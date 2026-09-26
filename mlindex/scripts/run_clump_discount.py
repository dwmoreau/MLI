"""Measure what a candidate is worth when it is not alone.

The convergence curve says how often ONE candidate starting a given distance from the true cell
refines to it. Every ensemble score then assumed candidates succeed independently, so a clump of k
counted as k chances. They do not: two candidates that start close together share their first
Miller-index assignment and most of their refinement. This measures by how much, and writes the
table `mlindex/utilities/ClumpDiscount.py` reads.

The discount is the only term in the ensemble score that reads crowding: a score built only on
distances from the true cell values a pile of candidates in one place the same as the same number
spread out.

WHAT IS MEASURED. Groups of k candidates whose centre sits at a shell of the lattice's own
convergence curve, with the members displaced from each other by delta, given as a fraction of
that shell's radius because lattice scales differ tenfold. How often a whole group fails, against
what independence predicts from each member's own distance, gives

    alpha = log P(the group all fail) / sum over members of log(1 - s(d_i))

At delta = 0 this is the identical-start case and reduces to k_eff/k. alpha = 1 is independence;
alpha near 1/k is a clump worth one candidate.

CUBIC IS REFUSED. A cubic cell has one free parameter, so a random direction is +1 or -1, every
member lands at exactly +/- delta/2, and half of each group is coincident whatever delta is asked.

Two stages, because the measurement costs node-hours and the reduction costs seconds:

    mpiexec -n 128 python -m mlindex.scripts.run_clump_discount --stage measure \\
        --bravais-lattices oP,mC,aP --dataset-directory mlindex/data/generated_datasets \\
        --roc-dir <the measured convergence curves> --out-dir results/clump

    python -m mlindex.scripts.run_clump_discount --stage reduce \\
        --out-dir results/clump --roc-dir <the measured convergence curves>

The reduction writes one {lattice}_clump_discount.npz, which is what the ensemble score loads.
"""
import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np

from mlindex.optimization import UtilitiesOptimizer
from mlindex.optimization.CandidateValidation import label_known_bl_batch
from mlindex.optimization.GroupedStarts import SeparatedStartManager
from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
from mlindex.scripts.run_ensemble_refine import (
    BROADENING_TAG, FACTORY_OF_SYSTEM, N_PEAKS, DEFAULT_N_PEAKS, check_mpi_world, commit,
    load_entries,
    load_second_phase_pool,
    needs_second_phase_pool,
    )
from mlindex.model_training import BenchmarkConditions
from mlindex.utilities.ClumpDiscount import discount_path
from mlindex.utilities.ConvergenceCurve import ROC_TAG, load_curve, success_of_distance
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM, get_partial_unit_cell

# Where the curve is steep enough to resolve a correlation. Near 0 or 1 the correlated and the
# independent predictions coincide and nothing can be told apart.
TARGET_SUCCESS_RATES = (0.80, 0.65, 0.50, 0.35, 0.20)
N_ITERATIONS = 100
CUBIC = ('cF', 'cI', 'cP')
# Clump sizes read out of one run: a group of m contains every smaller group as a subset.
SUBGROUPS = (2, 4, 8, 16, 32, 64)
# A clump is "candidates close enough to behave as one". Taken as the largest separation at which
# the discount has not yet risen appreciably above its identical-start value.
FLAT_TOLERANCE = 1.15


def pick_radii(radii, success, targets):
    """The shells whose success rate sits nearest each target."""
    chosen = [int(np.argmin(np.abs(success - target))) for target in targets]
    keep = sorted(set(chosen))
    return radii[keep], success[keep]


def measure(args):
    """Refine grouped starting clouds and record every candidate's fate and geometry.

    Returns this process's MPI rank, so a caller running both stages knows which one may reduce.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, n_ranks = comm.Get_rank(), comm.Get_size()
    check_mpi_world(comm, n_ranks)
    split_comm = comm.Split(color=rank, key=rank)
    out = Path(args.out_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    ratios = [float(v) for v in args.separation_ratios.split(',')]
    if ratios[0] != 0.0:
        raise SystemExit('the first separation ratio must be 0: it is the identical-start case '
                         'the rest is measured against')
    total = args.n_groups*args.group_size
    models_dir = Path(args.models_directory) if args.models_directory else _resolve_models_dir()
    package_root = Path(UtilitiesOptimizer.__file__).parent.parent.parent

    # Built once and across every lattice: a partner phase is not lattice-matched, so which
    # lattices this run was asked for must not change which partners exist.
    second_phase_pool = None
    if rank == 0 and needs_second_phase_pool(args.bundle):
        second_phase_pool = load_second_phase_pool(args.dataset_directory, args.seed)

    for bravais_lattice in args.bravais_lattices:
        # Every rank runs the same lattices and the work between them is blocking: send,
        # recv and gather. A rank that raises leaves the others waiting for a message that
        # never comes, and the job sits there until the allocation ends with nothing
        # written. Abort takes the whole job down with the traceback instead.
        try:
            _measure_lattice(args, comm, rank, n_ranks, split_comm, out, bravais_lattice,
                             ratios, total, models_dir, package_root, second_phase_pool)
        except Exception:
            import traceback
            print(f'RANK {rank} FAILED on {bravais_lattice}:', flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            comm.Abort(1)
    # The reduction reads what rank 0 has just written, so it must not start anywhere else.
    return rank


def _measure_lattice(args, comm, rank, n_ranks, split_comm, out, bravais_lattice,
                     ratios, total, models_dir, package_root, second_phase_pool):
    """One lattice: refine its grouped clouds at every separation, and write the raw run."""
    tag = ROC_TAG[bravais_lattice]
    n_peaks = N_PEAKS.get(bravais_lattice, DEFAULT_N_PEAKS)
    n_drop = int(re.search(r'drop(\d+)', tag).group(1))
    radii, success = pick_radii(*load_curve(args.roc_dir, bravais_lattice),
                                TARGET_SUCCESS_RATES)
    lattice_system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    options = {
        'convergence_testing': True, 'convergence_candidates': total,
        'convergence_distances': radii,
        'iteration_info': [{'worker': 'random_subsampling', 'n_iterations': N_ITERATIONS,
                            'n_peaks': n_peaks, 'n_drop': n_drop, 'uniform_sampling': False}],
        }
    if rank == 0:
        print(f'{bravais_lattice}: loading models', flush=True)
    factory = getattr(UtilitiesOptimizer, FACTORY_OF_SYSTEM[lattice_system])
    optimizer = factory(bravais_lattice, BROADENING_TAG, 1, split_comm,
                        project_path=package_root, options=options,
                        optimizer_class=SeparatedStartManager, seed=args.seed,
                        models_directory=models_dir)
    optimizer.group_size = args.group_size

    if rank == 0:
        entries, refused = load_entries(bravais_lattice, args.n_entries,
                                        args.dataset_directory, args.seed, args.bundle,
                                        second_phase_pool=second_phase_pool)
        print(f'{bravais_lattice}: {len(entries)} crystals ({refused} refused), '
              f'{args.n_groups} groups of {args.group_size}, {len(ratios)} separations',
              flush=True)
        for other in range(1, n_ranks):
            comm.send(entries.iloc[other::n_ranks], dest=other)
        mine = entries.iloc[0::n_ranks]
    else:
        mine = comm.recv(source=0)

    shape = (len(ratios), len(mine), radii.size, args.n_groups, args.group_size)
    correct = np.zeros(shape, dtype=bool)
    distance = np.zeros(shape, dtype=np.float32)
    separation = np.zeros(shape, dtype=np.float32)
    for ratio_index, ratio in enumerate(ratios):
        optimizer.separation_ratio = ratio
        for entry_index in range(len(mine)):
            entry = mine.iloc[entry_index]
            truth = get_partial_unit_cell(np.array(entry['reindexed_unit_cell']),
                                          lattice_system=lattice_system)
            for shell in range(radii.size):
                if rank == 0:
                    print(f'    {bravais_lattice} d/r={ratio} entry '
                          f'{entry_index + 1}/{len(mine)} shell '
                          f'{shell + 1}/{radii.size}', flush=True)
                optimizer.chunk_tag = (shell + 1)*1000 + ratio_index
                optimizer.opt_params['convergence_distances'] = radii[shell:shell + 1]
                optimizer.run(entry=entry, n_top_candidates=20)
                if optimizer.top_unit_cell.shape[0] != total:
                    raise RuntimeError(
                        f'{bravais_lattice} {entry["identifier"]}: '
                        f'{optimizer.top_unit_cell.shape[0]} candidates back against {total} '
                        f'generated; the row order is the measurement')
                is_correct, _ = label_known_bl_batch(
                    truth, optimizer.top_unit_cell, lattice_system)
                block = (args.n_groups, args.group_size)
                correct[ratio_index, entry_index, shell] = is_correct.reshape(block)
                distance[ratio_index, entry_index, shell] = \
                    optimizer.last_distance.reshape(block)
                separation[ratio_index, entry_index, shell] = \
                    optimizer.last_separation.reshape(block)
            if rank == 0:
                print(f'  {bravais_lattice} delta/r={ratio}: {entry_index + 1}/{len(mine)}',
                      flush=True)

    gathered = comm.gather({'correct': correct, 'distance': distance,
                            'separation': separation,
                            'identifiers': list(mine['identifier'])}, root=0)
    if rank == 0:
        np.savez_compressed(
            out/f'{bravais_lattice}_separation.npz',
            radii=radii, curve_success=success, separation_ratios=np.array(ratios),
            correct=np.concatenate([g['correct'] for g in gathered], axis=1),
            distance=np.concatenate([g['distance'] for g in gathered], axis=1),
            separation=np.concatenate([g['separation'] for g in gathered], axis=1),
            identifiers=np.array([n for g in gathered for n in g['identifiers']],
                                 dtype=object),
            )
        (out/f'{bravais_lattice}_manifest.json').write_text(json.dumps({
            'group_size': args.group_size, 'n_groups': args.n_groups,
            'separation_ratios': ratios, 'n_entries': args.n_entries, 'seed': args.seed,
            'bundle': args.bundle, 'commit': commit(), 'n_ranks': n_ranks,
            'platform': platform.platform(), 'machine': platform.machine(),
            'numpy': np.__version__, 'roc_file': f'{bravais_lattice}_roc_{tag}',
            }, indent=2, sort_keys=True), encoding='utf-8')
        print(f'{bravais_lattice}: written', flush=True)


def solve_alpha(correct, distance, radii_curve, success_curve, k):
    """alpha at clump size k, solved per shell and taken as the median over shells.

    Pooling the shells before taking the logarithm would be wrong: a shell where the curve reads
    0.8 and one where it reads 0.2 have failure rates orders of magnitude apart, so the log of
    their pooled rate is not the pooled log.
    """
    members = correct.shape[-1]
    if k > members:
        return np.nan
    n_sub = members//k
    correct = correct[..., :n_sub*k].reshape(correct.shape[:-1] + (n_sub, k))
    distance = distance[..., :n_sub*k].reshape(distance.shape[:-1] + (n_sub, k))
    all_failed = ~correct.any(axis=-1)
    rate = success_of_distance(radii_curve, success_curve, distance)
    independent = np.log1p(-np.clip(rate, 0.0, 1.0 - 1e-12)).sum(axis=-1)

    solved = []
    for shell in range(all_failed.shape[1]):
        p = float(all_failed[:, shell].mean())
        denominator = float(independent[:, shell].mean())
        if 0.0 < p < 1.0 and denominator < 0.0:
            # k_eff lies between 1 and k, so alpha lies in [1/k, 1]; outside is the solve failing.
            solved.append(float(np.clip(np.log(p)/denominator, 1.0/k, 1.0)))
    return float(np.median(solved)) if solved else np.nan


def reduce_run(args):
    """Turn each measured run into the (delta, k, alpha) table the ensemble score loads."""
    out = Path(args.out_dir)
    for path in sorted(out.glob('*_separation.npz')):
        bravais_lattice = path.name.split('_')[0]
        data = np.load(path, allow_pickle=True)
        ratios = data['separation_ratios']
        radii_curve, success_curve = load_curve(args.roc_dir, bravais_lattice)
        correct, distance, separation = data['correct'], data['distance'], data['separation']
        radii = data['radii']

        sizes = [k for k in SUBGROUPS if k <= correct.shape[-1]]
        alpha = np.array([[solve_alpha(correct[i], distance[i], radii_curve, success_curve, k)
                           for k in sizes] for i in range(len(ratios))])
        realised = np.array([np.mean([np.median(separation[i, :, j])/radii[j]
                                      for j in range(radii.size)]) for i in range(len(ratios))])

        # The clump radius: the largest separation at which the discount has not yet risen
        # appreciably above its identical-start value, so candidates that close still behave as
        # one. Read off the largest clump size measured, where the solve is best conditioned.
        column = alpha[:, -1]
        flat = [i for i in range(len(ratios))
                if np.isfinite(column[i]) and column[i] <= FLAT_TOLERANCE*column[0]]
        delta_ratio = realised[max(flat)] if flat else realised[0]
        delta = float(delta_ratio*np.median(radii))

        k = np.array([1.0] + [float(v) for v in sizes])
        table = np.array([1.0] + list(alpha[0]))
        if not np.all(np.isfinite(table)):
            raise SystemExit(f'{bravais_lattice}: alpha did not solve at every clump size; the '
                             f'run is too small to reduce')
        np.savez(discount_path(str(out), bravais_lattice), delta=delta, k=k, alpha=table)
        print(f'{bravais_lattice}: clump radius {delta:.3e} ({delta_ratio:.2f} r), '
              f'alpha {" ".join(f"{v:.3f}" for v in table)} at k {k.astype(int)}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m mlindex.scripts.run_clump_discount',
        description='Measure what a candidate is worth when it is not alone.')
    parser.add_argument('--stage', default='all', choices=('all', 'measure', 'reduce'))
    parser.add_argument('--bravais-lattices', default=None, metavar='A,B',
                        help='Comma-separated. Default: every lattice but the cubic ones, which '
                             'have one free parameter and cannot be scattered in a direction.')
    parser.add_argument('--out-dir', required=True, metavar='PATH')
    parser.add_argument('--roc-dir', required=True, metavar='PATH',
                        help='Directory holding the measured convergence curves.')
    parser.add_argument('--dataset-directory', default=None, metavar='PATH')
    parser.add_argument('--models-directory', default=None, metavar='PATH')
    parser.add_argument('--n-entries', type=int, default=200, metavar='N')
    parser.add_argument('--n-groups', type=int, default=128, metavar='N')
    parser.add_argument('--group-size', type=int, default=64, metavar='N')
    parser.add_argument('--separation-ratios', default='0,0.03,0.1,0.3,0.7,1.4', metavar='A,B',
                        help='Member separation as a fraction of the shell radius. The first '
                             'must be 0: it is the identical-start case the rest is read against.')
    parser.add_argument('--bundle', default='nominal', metavar='NAME')
    parser.add_argument('--seed', type=int, default=12345, metavar='N')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    known = [name for name in ROC_TAG if name not in CUBIC]
    args.bravais_lattices = ([name for name in args.bravais_lattices.split(',') if name]
                             if args.bravais_lattices else known)
    refused = [name for name in args.bravais_lattices if name in CUBIC]
    if refused:
        raise SystemExit(
            f'{", ".join(refused)}: a cubic cell has one free parameter, so a random direction is '
            f'+1 or -1, every member lands at exactly +/- delta/2 and half of each group is '
            f'coincident whatever separation is asked for. Measure the non-cubic lattices.')
    unknown = [name for name in args.bravais_lattices if name not in ROC_TAG]
    if unknown:
        raise SystemExit(f'not Bravais lattices this package knows: {", ".join(unknown)}')

    rank = 0
    if args.stage in ('all', 'measure'):
        if args.dataset_directory is None:
            raise SystemExit('--stage measure needs --dataset-directory')
        rank = measure(args)
    if args.stage in ('all', 'reduce') and rank == 0:
        reduce_run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
