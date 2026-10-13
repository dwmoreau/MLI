import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import argparse
import math
from pathlib import Path
import numpy as np

from mlindex.optimization.UtilitiesOptimizer import get_logger
from mlindex.optimization.UtilitiesOptimizer import get_mpi_organizer
from mlindex.optimization.UtilitiesOptimizer import get_optimizers
from mlindex.optimization.CandidateValidation import validate_candidate
from mlindex.utilities.gsas import load_pkslst
from mlindex.utilities.Ranker import load_ranker
from mlindex.utilities.Ranker import score_pool
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES
from mlindex.utilities.Reindexing import rhombohedral_to_hexagonal

_BL_MPI6_CFG = {
    'cF': (0, True),  'cI': (0, True),  'cP': (0, True),
    'hP': (1, True),  'hR': (2, True),
    'tI': (3, True),  'tP': (4, True),
    'oC': (1, False), 'oF': (2, False), 'oI': (3, False), 'oP': (4, False),
    'mC': (5, False), 'mP': (0, False), 'aP': (5, False),
}


# Per-lattice cost in seconds, as (gen, par). `gen` is candidate generation, which
# runs on a group's manager alone because only the manager holds models, so it does
# not divide by group size; `par` is the refinement loop, which does. A group of `k`
# processes owning lattice set S costs sum(gen) + sum(par)/k.
#
# Measured by `python -m mlindex.scripts.measure_bl_cost`, which times
# _generate_candidates_xnn and _run_loop directly. Re-run it to retune for a machine
# whose balance differs; it reports whether any allocation would actually change.
#
# These are steady-state costs. Roughly 1.1 s of one-time warm-up -- numba compiling,
# first-touch caching -- is paid once per process by whichever lattice it runs first,
# and is deliberately not in this table: it lands on every group alike, so it raises
# the true makespan above what `_group_cost` predicts without changing which plan
# wins. The previous table charged all of it to cF, which is first in
# BRAVAIS_LATTICES, and so claimed cF cost 1.25 s against the 0.01 s it really costs.
_BL_COST = {
    'cF': (0.01, 0.00), 'cI': (0.01, 0.00), 'cP': (0.01, 0.00),
    'hP': (0.87, 0.56), 'hR': (1.05, 0.68),
    'tI': (0.52, 0.43), 'tP': (0.52, 0.44),
    'oC': (0.82, 1.94), 'oF': (0.77, 1.91), 'oI': (0.70, 1.90), 'oP': (1.33, 2.16),
    'mC': (2.01, 4.37), 'mP': (2.16, 4.67), 'aP': (1.94, 3.28),
    }


def _group_cost(bravais_lattices, group_size):
    gen = sum(_BL_COST[bl][0] for bl in bravais_lattices)
    par = sum(_BL_COST[bl][1] for bl in bravais_lattices)
    return gen + par / group_size


def _plan_for_makespan(bravais_lattices, target, max_group_size):
    """Cheapest assignment reaching `target`, or None if it cannot be reached.

    A lattice whose solo cost exceeds the target is given enough processes to
    itself to meet it, up to `max_group_size`; everything else is
    first-fit-decreasing packed into single-process groups of capacity `target`.
    """
    if any(_BL_COST[bl][0] >= target for bl in bravais_lattices):
        # Generation alone already exceeds the target and never divides.
        return None
    groups = []
    light = []
    for bl in bravais_lattices:
        gen, par = _BL_COST[bl]
        if gen + par > target:
            group_size = int(math.ceil(par / (target - gen)))
            if group_size > max_group_size:
                return None
            groups.append(([bl], group_size))
        else:
            light.append(bl)
    bins = []
    for bl in sorted(light, key=lambda b: -sum(_BL_COST[b])):
        for existing in bins:
            if sum(sum(_BL_COST[b]) for b in existing) + sum(_BL_COST[bl]) <= target + 1e-9:
                existing.append(bl)
                break
        else:
            bins.append([bl])
    return groups + [(b, 1) for b in bins]


def allocate_lattice_groups(bravais_lattices, n_procs):
    """Assign Bravais lattices to `n_procs` processes as (lattice list, size) pairs.

    Binary search for the lowest makespan `_plan_for_makespan` can reach within
    the process budget. Cheap lattices share a process and expensive ones are split
    across several, whichever `_BL_COST` says is faster: the three cubics together
    cost 0.03 s while mP alone costs 6.83 s.

    Splitting a lattice stripes its candidates across the group, so **the result
    depends on `n_procs`**. A given `n_procs` always gives the same answer; a
    different one need not. That trade is deliberate -- refusing to split until
    every lattice had its own process left the makespan flat from eight processes
    to fourteen, six processes that bought nothing.

    The heaviest group comes first: the caller runs group 0 itself.
    """
    bravais_lattices = list(bravais_lattices)
    if n_procs <= 1 or len(bravais_lattices) <= 1:
        return [(bravais_lattices, max(1, n_procs))]
    low = max(_BL_COST[bl][0] for bl in bravais_lattices) + 1e-6
    high = sum(sum(_BL_COST[bl]) for bl in bravais_lattices)
    best = _plan_for_makespan(bravais_lattices, high, n_procs)
    for _ in range(60):
        mid = 0.5 * (low + high)
        plan = _plan_for_makespan(bravais_lattices, mid, n_procs)
        if plan is not None and sum(k for _, k in plan) <= n_procs:
            best, high = plan, mid
        else:
            low = mid
    return sorted(best, key=lambda group: -_group_cost(*group))


def build_base_parser(description="Start the display application"):
    """Return a parser pre-loaded with arguments common to all CLI entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--peaks",
        type=str,
        help="Comma- or space-separated list of peak positions; units set by --peak-units (default: d-spacing in Angstrom)"
    )
    parser.add_argument(
        "--peak-file",
        type=str,
        help="Path to peak list file (.npy array or GSAS-II .pkslst). For .npy, units set by --peak-units. For .pkslst, peaks are in 2-theta and --wavelength is required."
    )
    parser.add_argument(
        "--peak-units",
        type=str,
        choices=['d', 'q', 'q2', '2theta'],
        default="d",
        help="Units for peaks from --peaks or .npy files: 'd' = d-spacing (Angstrom, default), 'q' = 1/d (1/Angstrom), 'q2' = 1/d^2 (1/Angstrom^2), '2theta' = degrees (requires --wavelength). Not applied to .pkslst files.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="X-ray wavelength (Angstrom); required for .pkslst files, --zero-error, and --peak-units 2theta"
    )
    parser.add_argument(
        "--zero-error",
        action='store_true',
        help="Apply a correction for zero point error in theta"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="indexing_results.json",
        help="Output file path for results; a matching .txt file is also written (default: indexing_results.json)",
    )
    parser.add_argument(
        "--mpi",
        action='store_true',
        help="Use MPI parallelism (requires mpiexec -n 6)"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes for multiprocessing mode (default: 1 = serial)"
    )
    parser.add_argument(
        "--bravais-lattices",
        type=str,
        default=",".join(BRAVAIS_LATTICES),
        help=f"Comma-separated Bravais lattices to attempt (default: {','.join(BRAVAIS_LATTICES)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        # ASCII only: piping --help on Windows encodes through the locale codepage.
        help=("Seed for the candidate search (default: 12345). The same seed, peak list "
              "and --nproc always give the same result. Changing --nproc can change it, "
              "because the work is divided differently. There is normally no reason to "
              "change this seed."),
    )
    return parser


def parse_args(parser):
    """The parsed arguments, with `bravais_lattices` as a list of known lattices."""
    args = parser.parse_args()
    args.bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    invalid = [bl for bl in args.bravais_lattices if bl not in BRAVAIS_LATTICES]
    if invalid:
        parser.error(f"Unknown Bravais lattices: {', '.join(invalid)}")
    return args


def _convert_to_q2(peaks, units, wavelength):
    if units == 'q2':
        return peaks
    elif units == 'd':
        return 1.0 / peaks**2
    elif units == 'q':
        return peaks**2
    elif units == '2theta':
        assert wavelength, "--wavelength is required for --peak-units 2theta"
        theta_rad = peaks * (np.pi / 360.0)
        return (2.0 * np.sin(theta_rad) / wavelength) ** 2


def _load_peaks(args):
    if args.zero_error:
        assert args.wavelength, "--wavelength is required when --zero-error is set"

    if args.peaks is not None:
        raw = args.peaks.replace(',', ' ').split()
        peak_list = np.array([float(p) for p in raw])
        peak_list = _convert_to_q2(peak_list, args.peak_units, args.wavelength)
    elif args.peak_file is not None:
        if args.peak_file.endswith('.npy'):
            peak_list = np.load(args.peak_file)[:20]
            peak_list = _convert_to_q2(peak_list, args.peak_units, args.wavelength)
        elif args.peak_file.endswith('.pkslst'):
            assert args.wavelength, "--wavelength is required for .pkslst files"
            peak_list = load_pkslst(args.peak_file, args.wavelength)[:20]
        else:
            raise ValueError(f"Unsupported peak file format: {args.peak_file}")
    else:
        raise ValueError("Either --peaks or --peak-file must be provided")

    peak_list.sort()
    return peak_list


# How many candidates each Bravais lattice hands the final ranking (the ranker chooses which:
# `shortlist`). Twenty is the depth the packaged ranker was fitted at
# (`ranker_1/specification.json`, "depth").
SHORTLIST_DEPTH = 20


def _collect_results(result, bl, all_results, scores, rows):
    """One entry for each of `rows` of the lattice's pool, with its score.

    `result` is the lattice's `OptimizerBase.result()`; `scores` scores its pool, row for row.
    """
    pool = result['pool']
    for i in rows:
        unit_cell = pool['unit_cell'][i]
        if bl == 'hR':
            unit_cell = rhombohedral_to_hexagonal(unit_cell)
        entry = {
            'score': scores[i],
            'M20': pool['M20'][i],
            'M_sym': pool['M_sym'][i],
            'n_indexed': int(pool['n_indexed'][i]),
            'bravais_lattice': bl,
            'volume': get_unit_cell_volume(unit_cell[np.newaxis])[0],
            'a': unit_cell[0], 'b': unit_cell[1], 'c': unit_cell[2],
            'alpha': 180/np.pi * unit_cell[3],
            'beta':  180/np.pi * unit_cell[4],
            'gamma': 180/np.pi * unit_cell[5],
            'spacegroup': pool['spacegroup'][i],
        }
        all_results.append(entry)
    return all_results


def _write_results(output_data, ranker_name, output_file_base='indexing_results'):
    """Write the entries, in the order given, to `<base>.json` and `<base>.txt`, and print the
    first 20."""
    # Imported here rather than at module scope: this is its only use, and every
    # lattice group's manager re-imports this module when it is spawned.
    import pandas as pd

    output_df = pd.DataFrame(output_data)
    output_df.to_json(output_file_base + '.json')
    txt_cols = ['bravais_lattice', 'score', 'M20', 'M_sym', 'n_indexed', 'a', 'b', 'c',
                'alpha', 'beta', 'gamma', 'volume', 'spacegroup']
    txt_headers = {
        'bravais_lattice': 'Bravais Lattice', 'score': 'Score', 'M20': 'M20', 'M_sym': 'M_sym',
        'n_indexed': '# Indexed peaks',
        'a': 'A (Å)', 'b': 'B (Å)', 'c': 'C (Å)',
        'alpha': 'Alpha (°)', 'beta': 'Beta (°)', 'gamma': 'Gamma (°)',
        'volume': 'Volume (Å^3)', 'spacegroup': 'Space Group',
    }
    # encoding is explicit because the headers carry non-ASCII units: Windows would
    # otherwise use the locale codepage and fail to write this file outright
    # (cp1251/cp932 cannot encode 'Å').
    table = output_df.to_string(
        index=False,
        columns=txt_cols,
        header=[txt_headers[c] for c in txt_cols],
        formatters={
            'score': lambda x: f'{x:0.4f}',
            'M_sym': lambda x: f'{x:0.2f}',
            'volume': lambda x: f'{x:0.1f}',
            'a': lambda x: f'{x:0.4f}',
            'b': lambda x: f'{x:0.4f}',
            'c': lambda x: f'{x:0.4f}',
            'alpha': lambda x: f'{x:0.2f}',
            'beta': lambda x: f'{x:0.2f}',
            'gamma': lambda x: f'{x:0.2f}',
        }
    )
    with open(output_file_base + '.txt', 'w', encoding='utf-8') as handle:
        handle.write(f'Ranked by: {ranker_name}\n{table}\n')
    print(f'Ranked by: {ranker_name}')
    print(output_df[:20].to_string())


def _remove_duplicates(output_data):
    """Rank the entries, and drop each that repeats a higher-ranked cell of the same Bravais
    lattice in any basis (`Reindexing.same_lattice`, 0.5 % and 0.5 degrees).

    Ranked by score, and by M_sym where scores are equal -- the learned ranker's calibrated
    probability is often exactly 1.0 for several candidates of a clean pattern. The higher-ranked
    copy is kept, so first place never changes. Within one lattice only: two cells of different
    Bravais lattices are different answers even when they are one lattice metrically.

    P17 (fom-dev, cut 3.5): comparing parameters as printed missed the copies written in another
    basis, which held 1.7 of the top ten rows on dev_general and 2.9 on dev_hard; collapsing them
    puts the correct cell in the top ten 0.8 and 1.7 points more often. P17 also dropped
    `metric_subgroups` promotion, which ran here first: it removed 0.03 repeated rows a list and
    moved neither first place nor top ten, at ~5 ms a candidate.
    """
    from mlindex.utilities.Reindexing import niggli_cell, same_lattice

    ranked = sorted(output_data, key=lambda e: (e['score'], e['M_sym']), reverse=True)
    kept, kept_cells = [], []
    for entry in ranked:
        cell = niggli_cell([entry[key] for key in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')])
        if not any(ref['bravais_lattice'] == entry['bravais_lattice']
                   and same_lattice(cell, ref_cell, rtol=0.005, atol_deg=0.5)
                   for ref, ref_cell in zip(kept, kept_cells)):
            kept.append(entry)
            kept_cells.append(cell)
    return kept


def _write_output(output_file, results, ranker):
    """Rank and write every lattice's reported candidates.

    `results` is {Bravais lattice: `OptimizerBase.result()`}. Every survivor of every lattice is
    scored in one batch -- the ranker's context inputs compare a candidate with all of them --
    and each lattice reports the `SHORTLIST_DEPTH` the ranker's `shortlist` picks. `ranker` is a `Ranker.LearnedRanker` or `Ranker.MSymRanker`; the output names it.
    """
    scores = score_pool({bl: result['pool'] for bl, result in results.items()}, ranker)
    output_data = []
    for bl, result in results.items():
        rows = ranker.shortlist(result['pool'], scores[bl], SHORTLIST_DEPTH)
        _collect_results(result, bl, output_data, scores[bl], rows)
    output_data = _remove_duplicates(output_data)
    output_file_base = str(Path(output_file).with_suffix(''))
    _write_results(output_data, ranker.name, output_file_base=output_file_base)


def _run_mpi(args, peak_list, seed=12345):
    from mpi4py import MPI

    broadening_tag = '1'
    optimization_tag = '_0'
    n_top_candidates = 20

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    logger = get_logger(comm, optimization_tag)
    logger.info('Starting process')

    assert (n_ranks == 6) or (n_ranks == 1)

    bravais_lattices = args.bravais_lattices
    if n_ranks == 6:
        manager_rank = [_BL_MPI6_CFG[bl][0] for bl in bravais_lattices]
        serial = [_BL_MPI6_CFG[bl][1] for bl in bravais_lattices]
    else:
        manager_rank = [0] * len(bravais_lattices)
        serial = [True] * len(bravais_lattices)

    mpi_organizers = get_mpi_organizer(comm, bravais_lattices, manager_rank, serial)

    bl_string = ' '.join(bravais_lattices)
    logger.info(f'Including Bravais lattices {bl_string}')
    logger.info('Starting loading optimizers')
    optimizer = get_optimizers(rank, mpi_organizers, broadening_tag,
                               n_candidates_scale=1, logger=logger, seed=seed)

    if rank == 0:
        results = dict.fromkeys(bravais_lattices)

    for bravais_lattice in bravais_lattices:
        if rank in mpi_organizers[bravais_lattice].workers:
            if rank == mpi_organizers[bravais_lattice].manager:
                role = 'manager'
            else:
                role = 'worker'
            mpi_organizers[bravais_lattice].split_comm.barrier()
            logger.info(f'Starting optimization of {bravais_lattice} {role}')
            optimizer[bravais_lattice].run(
                q2=peak_list,
                n_top_candidates=n_top_candidates,
                zero_error=args.zero_error,
                wavelength=args.wavelength,
            )
            logger.info(f'Finishing optimization of {bravais_lattice} {role}')
    comm.barrier()

    logger.info('Gathering optimization results')
    for bravais_lattice in bravais_lattices:
        if rank == 0 and mpi_organizers[bravais_lattice].manager == 0:
            results[bravais_lattice] = optimizer[bravais_lattice].result()
        else:
            if rank == 0:
                results[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
            elif rank == mpi_organizers[bravais_lattice].manager:
                comm.send(optimizer[bravais_lattice].result(), dest=0)

    if rank == 0:
        _write_output(args.output_file, results, load_ranker())
    logger.info('Finished gathering optimization results')


def _run_mp(args, peak_list, n_procs, seed=12345):
    """Run every requested Bravais lattice, parallel across lattices AND candidates.

    `allocate_lattice_groups` deals the lattices into groups that run at the same
    time. Candidate generation holds the models, so it runs only on a group's own
    manager; with a single group it was 15.6 s of a 20.5 s pattern with every other
    process blocked waiting for it. Only the requested lattices are built.
    """
    from mlindex.optimization.MPOptimizer import (
        setup_lattice_groups, run_lattice_groups, shutdown_lattice_groups)

    broadening_tag = '1'
    n_top_candidates = 20

    assignment = allocate_lattice_groups(args.bravais_lattices, n_procs)
    groups, processes = setup_lattice_groups(
        assignment, broadening_tag, n_candidates_scale=1, seed=seed
    )

    results = run_lattice_groups(
        groups,
        q2=peak_list,
        zero_error=args.zero_error,
        wavelength=args.wavelength,
        n_top=n_top_candidates,
    )

    # Loaded once, here, after the search: the lattice groups never hold the ranker.
    ranker = load_ranker()
    shutdown_lattice_groups(groups, processes)
    _write_output(args.output_file,
                  {bl: results[bl] for bl in args.bravais_lattices}, ranker)


def main():
    args = parse_args(build_base_parser())
    peak_list = _load_peaks(args)
    if args.mpi:
        _run_mpi(args, peak_list, seed=args.seed)
    else:
        _run_mp(args, peak_list, n_procs=args.nproc, seed=args.seed)


if __name__ == "__main__":
    main()
