import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import argparse
import math
import types
from pathlib import Path
import numpy as np

from mlindex.optimization.UtilitiesOptimizer import get_logger
from mlindex.optimization.UtilitiesOptimizer import get_mpi_organizer
from mlindex.optimization.UtilitiesOptimizer import get_optimizers
from mlindex.optimization.CandidateValidation import validate_candidate
from mlindex.utilities.gsas import load_pkslst
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.Reindexing import rhombohedral_to_hexagonal


BRAVAIS_LATTICES = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']

_BL_MPI6_CFG = {
    'cF': (0, True),  'cI': (0, True),  'cP': (0, True),
    'hP': (1, True),  'hR': (2, True),
    'tI': (3, True),  'tP': (4, True),
    'oC': (1, False), 'oF': (2, False), 'oI': (3, False), 'oP': (4, False),
    'mC': (5, False), 'mP': (0, False), 'aP': (5, False),
}


# Per-lattice cost in seconds, measured on a 20-peak pattern with one process.
#
#   gen -- `_generate_candidates_xnn`. Runs on a group's manager alone, because
#          only the manager holds models, so it does NOT divide by group size.
#   par -- the iteration loop plus the post-loop block, which stripe across the
#          group and do divide.
#
# A group of `k` processes owning lattice set S therefore costs
#     sum(gen[bl] for bl in S) + sum(par[bl] for bl in S) / k
# which was checked against the shipped code by running single lattices at
# --nproc 1/2/4: mP measured 9.81/6.22/4.35 against 9.87/6.15/4.29 predicted,
# and eleven of twelve such points agree within 0.15 s.
#
# The table only has to RANK lattices, not predict wall clock on a given machine.
# It travels between patterns: over 11bmb, glucose and s178 the gen column
# totalled 15.20 / 15.59 / 15.68 s and the iteration half of par 21.68 / 21.65 /
# 21.94 s, because neither depends on how well a pattern indexes. Only the
# post-loop block moves, and it is the smallest of the three terms.
_BL_COST = {
    'cF': (1.00, 0.25), 'cI': (0.01, 0.00), 'cP': (0.01, 0.00),
    'hP': (1.07, 0.76), 'hR': (1.17, 0.72),
    'tI': (0.58, 0.54), 'tP': (0.63, 0.70),
    'oC': (0.89, 2.58), 'oF': (0.84, 2.86), 'oI': (0.65, 2.56), 'oP': (1.48, 5.54),
    'mC': (2.26, 6.62), 'mP': (2.43, 7.44), 'aP': (2.18, 5.02),
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

    Binary search on the makespan, with one rule covering fewer processes than
    lattices, exactly as many, and more.

    **While there are at least as many lattices as processes, every group holds
    exactly one process**, and the results are then bit-identical to `--nproc 1`
    on all fourteen lattices -- verified. That is a property the old
    candidate-striping design never had: at `--nproc 8` it reproduced `--nproc 1`
    on none of the fourteen, with mP's top M20 differing by 20 %.

    Only once processes outnumber lattices does a lattice get more than one, and
    the spare processes go to the heaviest lattices first. Those lattices are
    then searched by striping candidates across the group, exactly as the whole
    program used to be, so **above `len(bravais_lattices)` processes the results
    depend on `--nproc` again**. That is the only regime where they do.

    The heaviest group comes first: the caller runs group 0 itself.
    """
    bravais_lattices = list(bravais_lattices)
    if n_procs <= 1 or len(bravais_lattices) <= 1:
        return [(bravais_lattices, max(1, n_procs))]
    # Splitting a lattice buys speed at the cost of reproducibility, so it is
    # held back until there is nothing else left to spend a process on.
    if n_procs <= len(bravais_lattices):
        # Every group is one process, so this is plain longest-processing-time
        # bin packing. Done directly rather than through the search below,
        # which stops as soon as it meets the makespan and would leave
        # processes unused -- at eight it found a six-process plan, because
        # nothing can beat mP running alone. The makespan is the same either
        # way, but spreading the rest over every available process shortens
        # each group and so lowers the contention measured when several groups
        # generate candidates at once.
        n_groups = max(1, min(n_procs, len(bravais_lattices)))
        bins = [[] for _ in range(n_groups)]
        loads = [0.0] * n_groups
        for bl in sorted(bravais_lattices, key=lambda b: -sum(_BL_COST[b])):
            index = min(range(n_groups), key=lambda j: loads[j])
            bins[index].append(bl)
            loads[index] += sum(_BL_COST[bl])
        return sorted([(b, 1) for b in bins if b],
                      key=lambda group: -_group_cost(*group))
    max_group_size = n_procs
    low = max(_BL_COST[bl][0] for bl in bravais_lattices) + 1e-6
    high = sum(sum(_BL_COST[bl]) for bl in bravais_lattices)
    best = _plan_for_makespan(bravais_lattices, high, max_group_size)
    for _ in range(60):
        mid = 0.5 * (low + high)
        plan = _plan_for_makespan(bravais_lattices, mid, max_group_size)
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
        help="Random seed for reproducibility (default: 12345)",
    )
    return parser


def _parse_args():
    parser = build_base_parser()
    return parser.parse_args()


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


def _collect_results(optimizer, bl, all_results,
                     Minfos=None, spacegroups=None):
    for i in range(optimizer.top_M20.size):
        partial = optimizer.top_unit_cell[i]
        if bl in ['cF', 'cI', 'cP']:
            unit_cell = np.array([partial[0], partial[0], partial[0], np.pi/2, np.pi/2, np.pi/2])
        elif bl == 'hP':
            unit_cell = np.array([partial[0], partial[0], partial[1], 2*np.pi/3, np.pi/2, np.pi/2])
        elif bl == 'hR':
            unit_cell = rhombohedral_to_hexagonal(
                np.array([partial[0], partial[0], partial[0], partial[1], partial[1], partial[1]]))
        elif bl in ['tI', 'tP']:
            unit_cell = np.array([partial[0], partial[0], partial[1], np.pi/2, np.pi/2, np.pi/2])
        elif bl in ['oC', 'oF', 'oI', 'oP']:
            unit_cell = np.array([partial[0], partial[1], partial[2], np.pi/2, np.pi/2, np.pi/2])
        elif bl in ['mC', 'mP']:
            unit_cell = np.array([partial[0], partial[1], partial[2], np.pi/2, partial[3], np.pi/2])
        else:
            unit_cell = partial
        entry = {
            'M20': optimizer.top_M20[i],
            'n_indexed': int(optimizer.top_n_indexed[i]),
            'bravais_lattice': bl,
            'volume': get_unit_cell_volume(unit_cell[np.newaxis])[0],
            'a': unit_cell[0], 'b': unit_cell[1], 'c': unit_cell[2],
            'alpha': 180/np.pi * unit_cell[3],
            'beta':  180/np.pi * unit_cell[4],
            'gamma': 180/np.pi * unit_cell[5],
        }
        if Minfos is not None:             entry['Minfo']             = Minfos[i]
        if spacegroups is not None:        entry['spacegroup']        = spacegroups[i]
        all_results.append(entry)
    return all_results


def _write_results(output_data, output_file_base='indexing_results'):
    import pandas as pd
    output_df = pd.DataFrame(output_data)
    output_df.sort_values(by='M20', ascending=False, inplace=True, ignore_index=True)
    drop_columns = [c for c in ['Minfo'] if c in output_df.columns]
    output_df.drop(columns=drop_columns, inplace=True)
    output_df.to_json(output_file_base + '.json')
    txt_cols = [c for c in ['bravais_lattice', 'M20', 'n_indexed', 'a', 'b', 'c',
                             'alpha', 'beta', 'gamma', 'volume', 'spacegroup'] if c in output_df.columns]
    txt_headers = {
        'bravais_lattice': 'Bravais Lattice', 'M20': 'M20', 'n_indexed': '# Indexed peaks',
        'a': 'A (Å)', 'b': 'B (Å)', 'c': 'C (Å)',
        'alpha': 'Alpha (°)', 'beta': 'Beta (°)', 'gamma': 'Gamma (°)',
        'volume': 'Volume (Å^3)', 'spacegroup': 'Space Group',
    }
    # encoding is explicit because the headers carry non-ASCII units: Windows would
    # otherwise use the locale codepage and fail to write this file outright
    # (cp1251/cp932 cannot encode 'Å').
    output_df.to_string(
        output_file_base + '.txt',
        encoding='utf-8',
        index=False,
        columns=txt_cols,
        header=[txt_headers[c] for c in txt_cols],
        formatters={
            'volume': lambda x: f'{x:0.1f}',
            'a': lambda x: f'{x:0.4f}',
            'b': lambda x: f'{x:0.4f}',
            'c': lambda x: f'{x:0.4f}',
            'alpha': lambda x: f'{x:0.2f}',
            'beta': lambda x: f'{x:0.2f}',
            'gamma': lambda x: f'{x:0.2f}',
        }
    )
    print(output_df[:20].to_string())


_CS_CENTRING_TO_BL = {
    ('Cubic', 'P'): 'cP', ('Cubic', 'I'): 'cI', ('Cubic', 'F'): 'cF',
    ('Hexagonal', 'P'): 'hP',
    ('Trigonal', 'R'): 'hR', ('Trigonal', 'P'): 'hP',
    ('Tetragonal', 'P'): 'tP', ('Tetragonal', 'I'): 'tI',
    ('Orthorhombic', 'P'): 'oP', ('Orthorhombic', 'C'): 'oC',
    ('Orthorhombic', 'F'): 'oF', ('Orthorhombic', 'I'): 'oI',
    ('Monoclinic', 'P'): 'mP', ('Monoclinic', 'C'): 'mC',
    ('Triclinic', 'P'): 'aP',
}

_BL_RANK = {
    'aP': 0,
    'mP': 1, 'mC': 1,
    'oP': 2, 'oC': 2, 'oF': 2, 'oI': 2,
    'tP': 3, 'tI': 3,
    'hP': 4, 'hR': 4,
    'cP': 5, 'cI': 5, 'cF': 5,
}


def _is_same_cell(e1, e2, rtol, atol_deg):
    for key in ('a', 'b', 'c'):
        mean = 0.5 * (e1[key] + e2[key])
        if abs(e1[key] - e2[key]) / mean > rtol:
            return False
    for key in ('alpha', 'beta', 'gamma'):
        if abs(e1[key] - e2[key]) > atol_deg:
            return False
    return True


def promote_entries(output_data, delta=0.1):
    """Promote each entry to its highest-symmetry equivalent Bravais lattice.

    Pure per-entry work with no shared state, which is what lets it be split
    across processes. It is also the expensive half of `_conventional_cell`:
    `metric_subgroups` costs 6.5 ms per candidate against 280 candidates, and a
    thread pool cannot help because cctbx holds the GIL throughout -- measured
    at 1.201 s on one thread and 1.190 s on eight.
    """
    from cctbx import crystal as cctbx_crystal
    from cctbx.sgtbx.lattice_symmetry import metric_subgroups

    updated = []
    for entry in output_data:
        uc_params = (entry['a'], entry['b'], entry['c'],
                     entry['alpha'], entry['beta'], entry['gamma'])

        best_bl = entry['bravais_lattice']
        best_uc = uc_params
        best_volume = entry['volume']
        best_sg = entry.get('spacegroup')

        # Promotion is an optional improvement to one candidate, so a candidate cctbx cannot
        # analyse must not take the run down. metric_subgroups raises on cells it cannot reduce
        # ("Unsuitable value for rational rotation matrix"), which happens readily once triclinic
        # and monoclinic candidates from real data are in the pool -- 10 of 16 workers died this
        # way before the guard. Keep the un-promoted candidate; it is still a valid result.
        try:
            sym = cctbx_crystal.symmetry(unit_cell=uc_params, space_group_symbol='P 1')
            groups = metric_subgroups(sym, delta,
                                      enforce_max_delta_for_generated_two_folds=True)
            result_groups = groups.result_groups
        except (RuntimeError, ValueError):
            result_groups = []

        for group in result_groups:
            # Per group as well as per candidate, and the four fields are read before any is
            # committed, so a group that raises part way through cannot leave a cell whose
            # lattice, parameters and spacegroup disagree with each other.
            try:
                best_subsym = group['best_subsym']
                sg = best_subsym.space_group()
                candidate_bl = _CS_CENTRING_TO_BL.get(
                    (sg.crystal_system(), sg.conventional_centring_type_symbol())
                    )
                if candidate_bl is None:
                    continue
                if _BL_RANK.get(candidate_bl, 0) <= _BL_RANK.get(best_bl, 0):
                    continue
                promoted = (candidate_bl,
                            best_subsym.unit_cell().parameters(),
                            best_subsym.unit_cell().volume(),
                            best_subsym.space_group_info().type().lookup_symbol())
            except (RuntimeError, ValueError):
                continue
            best_bl, best_uc, best_volume, best_sg = promoted

        new_entry = dict(entry)
        new_entry['bravais_lattice'] = best_bl
        new_entry['a'], new_entry['b'], new_entry['c'] = best_uc[0], best_uc[1], best_uc[2]
        new_entry['alpha'], new_entry['beta'], new_entry['gamma'] = best_uc[3], best_uc[4], best_uc[5]
        new_entry['volume'] = best_volume
        if 'spacegroup' in new_entry:
            new_entry['spacegroup'] = best_sg
        updated.append(new_entry)

    return updated


def _conventional_cell(output_data, delta=0.1, promote=None):
    """Promote entries, then deduplicate near-identical cells within the same
    Bravais lattice, keeping the highest M20.

    `promote` defaults to running `promote_entries` here, in this process. The
    multiprocessing path passes one that fans the entries out over the lattice
    groups, which are otherwise idle by this point. The deduplication stays
    serial: it is O(n^2) over the promoted entries and cheap next to cctbx.
    """
    if promote is None:
        updated = promote_entries(output_data, delta=delta)
    else:
        updated = promote(output_data, delta)

    updated.sort(key=lambda e: e['M20'], reverse=True)
    kept = []
    for entry in updated:
        is_dup = any(
            ref['bravais_lattice'] == entry['bravais_lattice']
            and _is_same_cell(entry, ref, rtol=0.005, atol_deg=0.5)
            for ref in kept
        )
        if not is_dup:
            kept.append(entry)
    return kept


def _write_output(args, top_unit_cell, top_M20, top_Minfo, top_spacegroup,
                  top_n_indexed, promote=None):
    output_data = []
    for bl in args.bravais_lattices:
        mock = types.SimpleNamespace(
            top_unit_cell=top_unit_cell[bl],
            top_M20=top_M20[bl],
            top_n_indexed=top_n_indexed[bl],
        )
        _collect_results(
            mock, bl, output_data,
            Minfos=top_Minfo[bl],
            spacegroups=top_spacegroup[bl],
        )
    output_data = _conventional_cell(output_data, promote=promote)
    output_file_base = str(Path(args.output_file).with_suffix(''))
    _write_results(output_data, output_file_base=output_file_base)


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
        top_unit_cell = dict.fromkeys(bravais_lattices)
        top_M20 = dict.fromkeys(bravais_lattices)
        top_Minfo = dict.fromkeys(bravais_lattices)
        top_spacegroup = dict.fromkeys(bravais_lattices)
        top_n_indexed = dict.fromkeys(bravais_lattices)

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
            top_unit_cell[bravais_lattice] = optimizer[bravais_lattice].top_unit_cell
            top_M20[bravais_lattice] = optimizer[bravais_lattice].top_M20
            top_Minfo[bravais_lattice] = optimizer[bravais_lattice].top_Minfo
            top_spacegroup[bravais_lattice] = optimizer[bravais_lattice].top_spacegroup
            top_n_indexed[bravais_lattice] = optimizer[bravais_lattice].top_n_indexed
        else:
            if rank == 0:
                top_unit_cell[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_M20[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_Minfo[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_spacegroup[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_n_indexed[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
            elif rank == mpi_organizers[bravais_lattice].manager:
                comm.send(optimizer[bravais_lattice].top_unit_cell, dest=0)
                comm.send(optimizer[bravais_lattice].top_M20, dest=0)
                comm.send(optimizer[bravais_lattice].top_Minfo, dest=0)
                comm.send(optimizer[bravais_lattice].top_spacegroup, dest=0)
                comm.send(optimizer[bravais_lattice].top_n_indexed, dest=0)

    if rank == 0:
        _write_output(args, top_unit_cell, top_M20, top_Minfo, top_spacegroup,
                      top_n_indexed)
    logger.info('Finished gathering optimization results')


def _run_mp(args, peak_list, n_procs, seed=12345):
    """Run every requested Bravais lattice, parallel across lattices AND candidates.

    The lattices are dealt into groups by `allocate_lattice_groups`, and the
    groups run at the same time. That is the point: candidate generation holds
    the models and so runs only on a group's own manager, and with a single
    group -- which is what this function used to build -- it is 15.6 s of a
    20.5 s pattern with every other process blocked waiting for it.

    Only lattices the caller asked for are built, so `--bravais-lattices cP` no
    longer loads all fourteen lattices' models to use one of them.
    """
    from mlindex.optimization.MPOptimizer import (
        setup_lattice_groups, run_lattice_groups, shutdown_lattice_groups,
        promote_over_groups)

    broadening_tag = '1'
    n_top_candidates = 20

    assignment = allocate_lattice_groups(args.bravais_lattices, n_procs)
    groups, processes = setup_lattice_groups(
        assignment, broadening_tag, n_candidates_scale=1, seed=seed
    )

    try:
        results = run_lattice_groups(
            groups,
            q2=peak_list,
            zero_error=args.zero_error,
            wavelength=args.wavelength,
            n_top=n_top_candidates,
        )

        top_unit_cell = {}
        top_M20 = {}
        top_Minfo = {}
        top_spacegroup = {}
        top_n_indexed = {}
        for bravais_lattice in args.bravais_lattices:
            result = results[bravais_lattice]
            top_unit_cell[bravais_lattice] = result['top_unit_cell']
            top_M20[bravais_lattice] = result['top_M20']
            top_Minfo[bravais_lattice] = result['top_Minfo']
            top_spacegroup[bravais_lattice] = result['top_spacegroup']
            top_n_indexed[bravais_lattice] = result['top_n_indexed']

        # Before the shutdown below, not after: the conventional-cell promotion
        # is 6.5 ms of cctbx per candidate over roughly 280 of them, and it used
        # to run with every worker already dead.
        _write_output(args, top_unit_cell, top_M20, top_Minfo, top_spacegroup,
                      top_n_indexed,
                      promote=lambda entries, delta: promote_over_groups(
                          groups, entries, delta))
    finally:
        # In a finally block because every group manager and refinement worker
        # blocks on a queue with no timeout: an exception on the way out would
        # otherwise leave the run hanging instead of failing.
        shutdown_lattice_groups(groups, processes)


def main():
    args = _parse_args()
    selected = [bl.strip() for bl in args.bravais_lattices.split(',')]
    invalid = [bl for bl in selected if bl not in BRAVAIS_LATTICES]
    if invalid:
        raise SystemExit(f"Unknown Bravais lattices: {', '.join(invalid)}")
    args.bravais_lattices = selected
    peak_list = _load_peaks(args)
    if args.mpi:
        _run_mpi(args, peak_list, seed=args.seed)
    else:
        _run_mp(args, peak_list, n_procs=args.nproc, seed=args.seed)


if __name__ == "__main__":
    main()
