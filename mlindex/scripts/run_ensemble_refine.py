"""Choose how much of the candidate budget each generator should get.

The indexer draws candidate unit cells from three generators -- a random forest (`trees`), a
neural network (`abnn`) and a Miller-index template library (`templates`) -- and refines the
mixture. The shares are fixed numbers in `UtilitiesOptimizer.py`. This driver measures what those
shares should be, by generating each generator's candidates separately for patterns whose answer
is known and scoring every mixture of them against the measured convergence curve.

It runs in two stages, because generating candidates costs node-hours and does not depend on the
score, while scoring costs seconds and is the thing that gets changed:

  --stage generate   draw each generator's pool for many patterns and write it to disk. MPI.
  --stage fit        read those pools and score every mixture on a grid. No MPI, seconds.

Scores, selected with --variant, and more than one may be given in a single fit:

  shipped   the score the current shares were originally chosen against. A pool is credited for
            every candidate it has beyond the number the curve says is needed, integrated over
            the curve, with a flat penalty if it never has enough. It rises without limit, so a
            pattern already certain to be indexed keeps earning credit.
  expected  the expected number of candidates that converge. The same shape as shipped, changing
            one thing: a candidate is worth its own chance rather than the whole tail of the curve
            beyond it.
  capped    the log of one over the chance that every candidate fails, capped. It adds up over
            candidates the same way, but stops crediting a pattern that is already safe.

Both are reported so that the difference between them is attributable.

Worked commands. One lattice end to end on a laptop, which takes a few minutes:

    MLINDEX_MODELS_DIR=$PWD/mlindex/models python -m mlindex.scripts.run_ensemble_refine \\
        --stage all --bravais-lattices cP --n-entries 20 \\
        --dataset-directory mlindex/data/generated_datasets \\
        --roc-dir docs/fom_production/artifacts/P08_inputs/data \\
        --pools /tmp/pools --out-dir /tmp/mix --variant shipped --variant capped

Generation for every lattice, on a node. Use mpiexec, not srun: a conda-built mpi4py does not
read srun's process management, so every task comes up alone and runs the whole job. The stage
refuses to start in that state.

    mpiexec -n 32 python -m mlindex.scripts.run_ensemble_refine --stage generate \\
        --dataset-directory mlindex/data/generated_datasets --pools results/pools

Then, refitting as often as the score changes, on a laptop, against those pools:

    python -m mlindex.scripts.run_ensemble_refine --stage fit --pools results/pools \\
        --roc-dir docs/fom_production/artifacts/P08_inputs/data --out-dir results/mix \\
        --variant shipped --variant capped
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mlindex
from mlindex.optimization import UtilitiesOptimizer
from mlindex.optimization.GeneratorPools import generate_candidate_pools
from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
from mlindex.utilities.ConvergenceCurve import load_curve
from mlindex.utilities.Digests import derived_seed
from mlindex.utilities.EnsembleObjective import VARIANTS, evaluate
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM


BRAVAIS_LATTICES = ('cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',
                    'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP')
FACTORY_OF_SYSTEM = {
    'cubic': 'get_cubic_optimizer',
    'hexagonal': 'get_hexagonal_optimizer',
    'rhombohedral': 'get_rhombohedral_optimizer',
    'tetragonal': 'get_tetragonal_optimizer',
    'orthorhombic': 'get_orthorhombic_optimizer',
    'monoclinic': 'get_monoclinic_optimizer',
    'triclinic': 'get_triclinic_optimizer',
    }
# The cubic curves were measured on ten peaks and the rest on twenty, so a pattern is cut to the
# same count the curve for its lattice was measured at.
N_PEAKS = {'cF': 10, 'cI': 10, 'cP': 10}
DEFAULT_N_PEAKS = 20
BROADENING_TAG = '1'
# The shipped score is minimised and the capped one is maximised. Multiplying by this makes larger
# mean better for both, so one grid search and one report serve either.
SENSE = {'shipped': -1.0, 'expected': 1.0, 'capped': 1.0}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def check_mpi_world(comm, n_ranks):
    """Refuse to run one copy of the whole job per task because the launcher never reached MPI.

    A conda-built mpi4py does not read the process-management information srun hands out, so every
    task comes up in a world of size one, believes it is the only rank, takes the whole crystal
    list and does the entire job. Nothing about that shows up in the results: it is N times the
    cost for one job's work, and N processes writing one file.
    """
    launched = int(os.environ.get('SLURM_NTASKS') or 0)
    if n_ranks == 1 and launched > 1:
        raise SystemExit(
            f'the launcher started {launched} tasks but MPI reports a world of 1, so every task '
            f'would run the whole job. Launch with mpiexec rather than srun.'
            )
    return comm


def shipped_mix_and_budget(optimizer):
    """The shares and the budget this lattice ships with, read off the optimizer.

    Taken from `generator_info` rather than transcribed, so the comparison is always against what
    the package actually does. A generator split across several split groups contributes each of
    its entries, which is why the counts are summed per name rather than read off one row.
    """
    counts = {}
    order = []
    for generator_info in optimizer.opt_params['generator_info']:
        name = generator_info['generator']
        if name not in counts:
            counts[name] = 0
            order.append(name)
        counts[name] += generator_info['n_unit_cells']
    budget = sum(counts.values())
    return order, [counts[name]/budget for name in order], budget


def load_entries(bravais_lattice, n_entries, dataset_directory, seed):
    """Training crystals with enough peaks, chosen reproducibly.

    Training crystals, not benchmark ones: the mix is a setting being selected, and selecting it
    on the population it is later reported against would be reading the answer first.
    """
    path = Path(dataset_directory)/f'dataset_{bravais_lattice}.parquet'
    if not path.is_file():
        raise SystemExit(
            f'no source dataset for {bravais_lattice} at {path}. Point --dataset-directory at a '
            f'tree that has it.'
            )
    n_peaks = N_PEAKS.get(bravais_lattice, DEFAULT_N_PEAKS)
    data = pd.read_parquet(path, columns=[
        'identifier', 'train', f'q2_{BROADENING_TAG}', 'reindexed_xnn'])
    data = data.loc[data['train']]
    peaks = data[f'q2_{BROADENING_TAG}']
    data = data.loc[peaks.apply(lambda q2: np.count_nonzero(q2) >= n_peaks)]
    data = data.sort_values('identifier', kind='stable', ignore_index=True)
    if data.shape[0] > n_entries:
        rng = np.random.default_rng(derived_seed(f'sample:{bravais_lattice}', seed))
        data = data.iloc[np.sort(rng.choice(data.shape[0], size=n_entries, replace=False))]
        data = data.reset_index(drop=True)
    q2 = np.zeros((data.shape[0], n_peaks))
    for index in range(data.shape[0]):
        q2[index] = np.array(data[f'q2_{BROADENING_TAG}'].iloc[index])[:n_peaks]
    data = data.copy()
    data['q2'] = list(add_q2_error(q2, None, 1, np.random.default_rng(seed)))
    return data


def commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def generate(args):
    """Draw every generator's pool for every chosen lattice and write them to disk."""
    from mpi4py import MPI

    comm = check_mpi_world(MPI.COMM_WORLD, MPI.COMM_WORLD.Get_size())
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    split_comm = comm.Split(color=rank, key=rank)
    out = Path(args.pools)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    models_directory = Path(args.models_directory) if args.models_directory else _resolve_models_dir()
    package_root = Path(mlindex.__path__[0]).parent
    manifest = {
        'n_entries': args.n_entries, 'budget_scale': args.budget_scale, 'seed': args.seed,
        'broadening_tag': BROADENING_TAG, 'commit': commit(), 'n_ranks': n_ranks,
        'platform': platform.platform(), 'machine': platform.machine(),
        'numpy': np.__version__, 'models_directory': str(models_directory),
        'population': 'training crystals, nominal error severity 1, no contaminants, no dropout',
        'lattices': {},
        }

    for bravais_lattice in args.bravais_lattices:
        factory = getattr(
            UtilitiesOptimizer, FACTORY_OF_SYSTEM[BL_TO_LATTICE_SYSTEM[bravais_lattice]])
        optimizer = factory(bravais_lattice, BROADENING_TAG, 1, split_comm,
                            project_path=package_root, seed=args.seed,
                            models_directory=models_directory)
        names, fractions, budget = shipped_mix_and_budget(optimizer)
        per_generator = int(round(args.budget_scale*budget))

        if rank == 0:
            entries = load_entries(bravais_lattice, args.n_entries, args.dataset_directory,
                                   args.seed)
            print(f'{bravais_lattice}: {len(entries)} crystals, {per_generator} candidates a '
                  f'generator, shipped budget {budget}', flush=True)
            for other in range(1, n_ranks):
                comm.send(entries.iloc[other::n_ranks], dest=other)
            mine = entries.iloc[0::n_ranks]
        else:
            mine = comm.recv(source=0)

        # The generators are seeded per crystal so that a subset of a run regenerates identically
        # and the rank count cannot change an answer.
        positions = []
        truths = []
        for index in range(len(mine)):
            entry = mine.iloc[index]
            rng = np.random.default_rng(derived_seed(
                f'pool:{bravais_lattice}:{entry["identifier"]}', args.seed))
            xnn, xnn_true, generated_names = generate_candidate_pools(
                optimizer, entry, candidates_per_model=per_generator, rng=rng)
            positions.append(xnn.astype(np.float32))
            truths.append(xnn_true)
        payload = {'xnn': positions, 'xnn_true': truths,
                   'identifiers': list(mine['identifier']), 'generator_names': generated_names}

        gathered = comm.gather(payload, root=0)
        if rank == 0:
            xnn = np.stack([block for part in gathered for block in part['xnn']])
            xnn_true = np.stack([block for part in gathered for block in part['xnn_true']])
            identifiers = [name for part in gathered for name in part['identifiers']]
            # Distances at full precision because the score is computed from them; positions at
            # single precision because they are only ever used to count near neighbours, where the
            # radii are thousands of times coarser than float32's resolution here.
            distance = np.linalg.norm(xnn - xnn_true[:, np.newaxis, np.newaxis, :], axis=-1)
            np.savez_compressed(
                out/f'{bravais_lattice}_pools.npz',
                xnn=xnn, xnn_true=xnn_true, distances=distance,
                identifiers=np.array(identifiers, dtype=object),
                generator_names=np.array(generated_names, dtype=object),
                )
            manifest['lattices'][bravais_lattice] = {
                'n_crystals': int(xnn.shape[0]), 'per_generator': per_generator,
                'shipped_budget': budget, 'shipped_mix': dict(zip(names, fractions)),
                'generator_names': generated_names,
                'n_peaks': N_PEAKS.get(bravais_lattice, DEFAULT_N_PEAKS),
                }
            # Rewritten after every lattice, not once at the end, so that a run which dies
            # partway leaves the lattices it finished in a state the fit stage can read.
            (out/'pools_manifest.json').write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
            print(f'{bravais_lattice}: wrote {xnn.shape[0]} crystals', flush=True)
    return 0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def mix_grid(step, n_generators):
    """Every mix on a regular grid over the simplex, as fractions summing to one."""
    n = int(round(1.0/step))
    if n_generators != 3:
        raise ValueError(f'the grid is written for three generators, not {n_generators}')
    return np.array([(a/n, b/n, (n - a - b)/n)
                     for a in range(n + 1) for b in range(n + 1 - a)])


def counts_for_mix(mix, budget):
    """Candidate counts that sum to the budget exactly, by largest remainder.

    Rounding each share independently makes the total drift with the mix, and the score rises with
    the number of candidates, so a mix that happened to round up would win on that alone.
    """
    exact = np.asarray(mix, dtype=float)*budget
    counts = np.floor(exact).astype(int)
    short = budget - int(counts.sum())
    if short:
        counts[np.argsort(-(exact - counts))[:short]] += 1
    return counts


def stack_pools(distance, counts):
    """(n_crystals, sum(counts)): the first counts[g] candidates of each generator, side by side."""
    return np.concatenate(
        [distance[:, :count, index] for index, count in enumerate(counts) if count], axis=1)


def score_every_mix(distance, grid, budget, curve, variant, cap):
    """(n_mixes, n_crystals) scores, oriented so that larger is always better."""
    scores = np.empty((grid.shape[0], distance.shape[0]))
    extra = {'cap': cap} if variant == 'capped' else {}
    for index, mix in enumerate(grid):
        pool = stack_pools(distance, counts_for_mix(mix, budget))
        scores[index] = SENSE[variant]*evaluate(variant, pool, curve, **extra)
    return scores


def choose_mix(scores, grid, reduction):
    """The mix a reduction picks, the pooled score over the grid, and how it got there.

    'pooled'       one mix for the lattice, against the mean over crystals.
    'per-pattern'  the best mix for each crystal, averaged. This is what the original fit did, by
                   averaging the logits it optimised; averaging the winning shares is the same
                   idea on a grid. Kept so that what it costs can be measured rather than assumed.

    For 'per-pattern' the third return says whether the average describes any pattern at all. A
    per-pattern optimum that hands almost the whole budget to one generator is not evidence for a
    blend, and an average of such optima is a mix no pattern asked for.
    """
    pooled = scores.mean(axis=1)
    if reduction == 'pooled':
        return grid[int(np.argmax(pooled))], pooled, {}
    if reduction == 'per-pattern':
        winners = grid[np.argmax(scores, axis=0)]
        return winners.mean(axis=0), pooled, {
            'share_all_or_nothing': float(np.mean(winners.max(axis=1) > 0.95)),
            'share_on_a_corner': float(np.mean(np.isclose(winners.max(axis=1), 1.0))),
            'winner_spread': float(np.mean(winners.std(axis=0))),
            }
    raise ValueError(f'unknown reduction {reduction!r}')


def screen(pooled, grid, chosen):
    """How decided the answer is, and whether it sits in a corner.

    A score that is flat gives every mix the same value, and `argmax` then returns whichever the
    grid happens to enumerate first. Counting how many mixes come within a hundredth of the whole
    range of the best one says whether there was anything to choose.
    """
    best = float(pooled.max())
    worst = float(pooled.min())
    spread = best - worst
    tied = int(np.count_nonzero(pooled >= best - 0.01*spread)) if spread > 0 else pooled.size
    return {
        'value_best': best,
        'value_worst': worst,
        'spread': spread,
        'n_tied': tied,
        'n_mixes': int(pooled.size),
        'determined': bool(tied < 0.02*pooled.size),
        'on_boundary': bool(np.any(np.isclose(chosen, 0.0)) or np.any(np.isclose(chosen, 1.0))),
        }


def fit(args):
    """Score every mix on the grid, under every variant and reduction asked for."""
    pools = Path(args.pools)
    manifest_path = pools/'pools_manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'no pools manifest at {manifest_path}; run --stage generate first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    report = ['# The generator mix, by lattice', '',
              f'pools: {pools}   commit that generated them: {manifest.get("commit")}',
              f'population: {manifest.get("population")}', '']

    for bravais_lattice in args.bravais_lattices:
        path = pools/f'{bravais_lattice}_pools.npz'
        if not path.is_file():
            continue
        data = np.load(path, allow_pickle=True)
        names = [str(name) for name in data['generator_names']]
        distance = np.asarray(data['distances'], dtype=float)
        if not np.all(np.isfinite(distance)):
            raise SystemExit(f'{path} holds non-finite distances; regenerate it')
        info = manifest['lattices'][bravais_lattice]
        shipped = np.array([info['shipped_mix'][name] for name in names])
        curve = np.vstack(load_curve(args.roc_dir, bravais_lattice))
        grid = mix_grid(args.step, len(names))

        print(f'{bravais_lattice}: scoring {grid.shape[0]} mixes x {len(args.variants)} '
              f'variants over {distance.shape[0]} crystals', flush=True)
        report.append(f'## {bravais_lattice}   {distance.shape[0]} crystals, '
                      f'{distance.shape[1]} candidates a generator, '
                      f'shipped budget {info["shipped_budget"]}')
        report.append('   shipped mix  ' + '  '.join(
            f'{name} {value:.2f}' for name, value in zip(names, shipped)))

        for budget_scale in args.budget_scales:
            budget = int(round(budget_scale*info['shipped_budget']))
            if budget > distance.shape[1]:
                report.append(f'   x{budget_scale:g}: skipped, the pools only go to '
                              f'{distance.shape[1]} a generator')
                continue
            for variant in args.variants:
                scores = score_every_mix(distance, grid, budget, curve, variant, args.cap)
                shipped_value = float(SENSE[variant]*np.mean(evaluate(
                    variant, stack_pools(distance, counts_for_mix(shipped, budget)), curve,
                    **({'cap': args.cap} if variant == 'capped' else {}))))
                for reduction in args.reductions:
                    for split, rows_of in _splits(distance.shape[0], args.split_seed):
                        chosen, pooled, how = choose_mix(scores[:, rows_of], grid, reduction)
                        measured = screen(pooled, grid, chosen)
                        measured.update(how)
                        rows.append(dict(
                            bravais_lattice=bravais_lattice, variant=variant,
                            reduction=reduction, split=split, budget=budget,
                            budget_scale=budget_scale, n_crystals=int(rows_of.size),
                            **{f'best_{name}': float(value)
                               for name, value in zip(names, chosen)},
                            **{f'shipped_{name}': float(value)
                               for name, value in zip(names, shipped)},
                            value_shipped=shipped_value, **measured))
                    _report_block(report, names, variant, reduction, budget_scale, rows)
        report.append('')
        # Written after every lattice rather than once at the end: the low-symmetry lattices are
        # the slow ones and they come last, so a run that dies on aP would otherwise take the
        # thirteen finished lattices with it.
        pd.DataFrame(rows).to_csv(out/'ensemble_mix.csv', index=False)

    frame = pd.DataFrame(rows)
    text = '\n'.join(report) + '\n'
    (out/'ensemble_mix.txt').write_text(text, encoding='utf-8')
    print(text)
    return 0


def _splits(n_crystals, seed):
    """The whole set, then two disjoint halves, so a mix that does not reproduce is visible."""
    everything = np.arange(n_crystals)
    shuffled = np.random.default_rng(seed).permutation(everything)
    half = n_crystals//2
    return [('all', everything),
            ('half-a', np.sort(shuffled[:half])),
            ('half-b', np.sort(shuffled[half:]))]


def _report_block(report, names, variant, reduction, budget_scale, rows):
    """The last three rows written are one setting's three splits."""
    block = rows[-3:]
    line = f'   x{budget_scale:g} {variant:>8s} {reduction:>12s}  '
    for row in block:
        mix = '/'.join(f'{row[f"best_{name}"]:.2f}' for name in names)
        line += f'{row["split"]}: {mix}  '
    whole = block[0]
    # The two halves are disjoint, so how far apart their answers are is the whole of the
    # stability screen. Reported as a number rather than left for the reader to eyeball.
    halves = [row for row in block if row['split'] != 'all']
    if len(halves) == 2:
        drift = max(abs(halves[0][f'best_{name}'] - halves[1][f'best_{name}']) for name in names)
        for row in block:
            row['half_to_half'] = drift
        line += f' | halves differ by {drift:.2f}'
    report.append(line)
    report.append(f'        best {whole["value_best"]:+.5f}  shipped {whole["value_shipped"]:+.5f}'
                  f'  spread {whole["spread"]:.5f}  {whole["n_tied"]} of {whole["n_mixes"]} tied'
                  f'  {"determined" if whole["determined"] else "NOT DETERMINED"}'
                  f'{"  optimum on a boundary" if whole["on_boundary"] else ""}')
    if 'share_all_or_nothing' in whole:
        report.append(f'        per-pattern optima: '
                      f'{100*whole["share_all_or_nothing"]:.0f}% give one generator over 95%, '
                      f'{100*whole["share_on_a_corner"]:.0f}% give it everything')


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m mlindex.scripts.run_ensemble_refine',
        description='Measure how much of the candidate budget each generator should get.')
    parser.add_argument('--stage', default='all', choices=('all', 'generate', 'fit'),
                        help='generate writes candidate pools (MPI, slow); fit scores mixes '
                             'against them (no MPI, seconds). Default: all.')
    parser.add_argument('--bravais-lattices', default=','.join(BRAVAIS_LATTICES), metavar='A,B',
                        help='Comma-separated. Default: all fourteen.')
    parser.add_argument('--pools', required=True, metavar='PATH',
                        help='Directory the candidate pools are written to and read from.')
    parser.add_argument('--out-dir', default=None, metavar='PATH',
                        help='Where the fit writes its table and report. Required for --stage '
                             'fit and all.')
    parser.add_argument('--roc-dir', default=None, metavar='PATH',
                        help='Directory holding the measured convergence curves. They are run '
                             'output and are not shipped with the package. Required to fit.')
    generate_group = parser.add_argument_group('generate')
    generate_group.add_argument('--dataset-directory', default=None, metavar='PATH',
                                help='Directory holding dataset_{lattice}.parquet.')
    generate_group.add_argument('--n-entries', type=int, default=300, metavar='N',
                                help='Source crystals per lattice (default: 300).')
    generate_group.add_argument('--budget-scale', type=float, default=2.0, metavar='X',
                                help='Candidates per generator, as a multiple of the shipped '
                                     'budget (default: 2.0, so any mix can be scored at twice it).')
    generate_group.add_argument('--models-directory', default=None, metavar='PATH',
                                help='Model tree. Default: the usual resolution order.')
    generate_group.add_argument('--seed', type=int, default=12345, metavar='N')
    fit_group = parser.add_argument_group('fit')
    fit_group.add_argument('--variant', action='append', dest='variants', default=None,
                           choices=VARIANTS,
                           help='Which score to fit against. Repeat for several in one pass. '
                                'Default: shipped.')
    fit_group.add_argument('--reduction', action='append', dest='reductions', default=None,
                           choices=('pooled', 'per-pattern'),
                           help='How per-crystal scores become one mix. Repeat for several. '
                                'Default: pooled.')
    fit_group.add_argument('--cap', type=float, default=5.0, metavar='X',
                           help='Where the capped score stops crediting a pattern (default: 5.0, '
                                'about a 0.7 percent chance that every candidate fails).')
    fit_group.add_argument('--step', type=float, default=0.02, metavar='X',
                           help='Grid spacing on the simplex (default: 0.02, so 1326 mixes).')
    fit_group.add_argument('--budget-scales', default='1', metavar='A,B',
                           help='Budgets to score, as multiples of the shipped one (default: 1).')
    fit_group.add_argument('--split-seed', type=int, default=12345, metavar='N',
                           help='Seed for the half-and-half stability check (default: 12345).')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.bravais_lattices = [name for name in args.bravais_lattices.split(',') if name]
    unknown = [name for name in args.bravais_lattices if name not in BRAVAIS_LATTICES]
    if unknown:
        raise SystemExit(f'not Bravais lattices this package knows: {", ".join(unknown)}')
    args.variants = args.variants or ['shipped']
    args.reductions = args.reductions or ['pooled']
    args.budget_scales = [float(value) for value in args.budget_scales.split(',')]

    if args.stage in ('all', 'generate'):
        if args.dataset_directory is None:
            raise SystemExit('--stage generate needs --dataset-directory')
        generate(args)
    if args.stage in ('all', 'fit'):
        if args.roc_dir is None:
            raise SystemExit('--stage fit needs --roc-dir')
        if args.out_dir is None:
            raise SystemExit('--stage fit needs --out-dir')
        fit(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
