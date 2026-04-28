import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import argparse
import types
import numpy as np
import pandas as pd

from mlindex.optimization.UtilitiesOptimizer import get_logger
from mlindex.optimization.UtilitiesOptimizer import get_mpi_organizer
from mlindex.optimization.UtilitiesOptimizer import get_optimizers
from mlindex.optimization.CandidateValidation import validate_candidate
from mlindex.utilities.gsas import load_pkslst
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.Reindexing import rhombohedral_to_hexagonal


BRAVAIS_LATTICES = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']


def _parse_args():
    parser = argparse.ArgumentParser(description="Start the display application")
    parser.add_argument(
        "--peaks",
        type=str,
        help="Comma- or space-separated list of peak positions; units set by --peak-units (default: d-spacing in Å)"
    )
    parser.add_argument(
        "--peak-file",
        type=str,
        help="Path to peak list file (.npy array or GSAS-II .pkslst). For .npy, units set by --peak-units. For .pkslst, peaks are in 2θ and --wavelength is required."
    )
    parser.add_argument(
        "--peak-units",
        type=str,
        choices=['d', 'q', 'q2', '2theta'],
        default="d",
        help="Units for peaks from --peaks or .npy files: 'd' = d-spacing (Å, default), 'q' = 1/d (Å⁻¹), 'q2' = 1/d² (Å⁻²), '2theta' = degrees (requires --wavelength). Not applied to .pkslst files.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="X-ray wavelength (Å); required for .pkslst files, --zero-error, and --peak-units 2theta"
    )
    parser.add_argument(
        "--triplets-file",
        type=str,
        default=None,
        help="file name of the triplets file (numpy array)"
    )
    parser.add_argument(
        "--zero-error",
        action='store_true',
        help="Apply a correction for zero point error in theta"
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

    if args.triplets_file:
        triplet_obs = np.load(args.triplets_file)
    else:
        triplet_obs = None
    peak_list.sort()
    return peak_list, triplet_obs


def _collect_results(optimizer, bl, all_results,
                     Minfos=None, spacegroups=None, M_triplets=None, n_indexed_triplets=None):
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
        if M_triplets is not None:         entry['M_triplet']         = list(M_triplets[i])
        if n_indexed_triplets is not None: entry['n_indexed_triplet'] = n_indexed_triplets[i]
        all_results.append(entry)
    return all_results


def _write_results(output_data, output_file_base='indexing_results'):
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
    output_df.to_string(
        output_file_base + '.txt',
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


def _write_output(args, top_unit_cell, top_M20, top_Minfo, top_spacegroup,
                  top_n_indexed, top_M_triplets, top_n_indexed_triplets, triplet_obs):
    output_data = []
    for bl in BRAVAIS_LATTICES:
        mock = types.SimpleNamespace(
            top_unit_cell=top_unit_cell[bl],
            top_M20=top_M20[bl],
            top_n_indexed=top_n_indexed[bl],
        )
        _collect_results(
            mock, bl, output_data,
            Minfos=top_Minfo[bl],
            spacegroups=top_spacegroup[bl],
            M_triplets=top_M_triplets[bl] if triplet_obs is not None else None,
            n_indexed_triplets=top_n_indexed_triplets[bl] if triplet_obs is not None else None,
        )
    _write_results(output_data, output_file_base='indexing_results')


def _run_mpi(args, peak_list, triplet_obs):
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

    if n_ranks == 6:
        manager_rank = [   0,    0,    0,    1,    2,    3,    4,     1,     2,     3,     4,     5,     0,     5]
        serial =       [True, True, True, True, True, True, True, False, False, False, False, False, False, False]
    else:
        manager_rank = [0 for _ in range(len(BRAVAIS_LATTICES))]
        serial = [True for _ in range(len(BRAVAIS_LATTICES))]

    mpi_organizers = get_mpi_organizer(comm, BRAVAIS_LATTICES, manager_rank, serial)

    bl_string = ' '.join(BRAVAIS_LATTICES)
    logger.info(f'Including Bravais lattices {bl_string}')
    logger.info('Starting loading optimizers')
    optimizer = get_optimizers(rank, mpi_organizers, broadening_tag,
                               n_candidates_scale=1, logger=logger)

    if rank == 0:
        top_unit_cell = dict.fromkeys(BRAVAIS_LATTICES)
        top_M20 = dict.fromkeys(BRAVAIS_LATTICES)
        top_Minfo = dict.fromkeys(BRAVAIS_LATTICES)
        top_spacegroup = dict.fromkeys(BRAVAIS_LATTICES)
        top_n_indexed = dict.fromkeys(BRAVAIS_LATTICES)
        if triplet_obs is not None:
            top_M_triplets = dict.fromkeys(BRAVAIS_LATTICES)
            top_n_indexed_triplets = dict.fromkeys(BRAVAIS_LATTICES)

    for bravais_lattice in BRAVAIS_LATTICES:
        if rank in mpi_organizers[bravais_lattice].workers:
            if rank == mpi_organizers[bravais_lattice].manager:
                role = 'manager'
            else:
                role = 'worker'
            mpi_organizers[bravais_lattice].split_comm.barrier()
            logger.info(f'Starting optimization of {bravais_lattice} {role}')
            optimizer[bravais_lattice].run(
                q2=peak_list,
                triplets=triplet_obs,
                n_top_candidates=n_top_candidates,
                zero_error=args.zero_error,
                wavelength=args.wavelength,
            )
            logger.info(f'Finishing optimization of {bravais_lattice} {role}')
    comm.barrier()

    logger.info('Gathering optimization results')
    for bravais_lattice in BRAVAIS_LATTICES:
        if rank == 0 and mpi_organizers[bravais_lattice].manager == 0:
            top_unit_cell[bravais_lattice] = optimizer[bravais_lattice].top_unit_cell
            top_M20[bravais_lattice] = optimizer[bravais_lattice].top_M20
            top_Minfo[bravais_lattice] = optimizer[bravais_lattice].top_Minfo
            top_spacegroup[bravais_lattice] = optimizer[bravais_lattice].top_spacegroup
            top_n_indexed[bravais_lattice] = optimizer[bravais_lattice].top_n_indexed
            if triplet_obs is not None:
                top_n_indexed_triplets[bravais_lattice] = optimizer[bravais_lattice].top_n_indexed_triplets
                top_M_triplets[bravais_lattice] = optimizer[bravais_lattice].top_M_triplets
        else:
            if rank == 0:
                top_unit_cell[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_M20[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_Minfo[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_spacegroup[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_n_indexed[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                if triplet_obs is not None:
                    top_n_indexed_triplets[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                    top_M_triplets[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
            elif rank == mpi_organizers[bravais_lattice].manager:
                comm.send(optimizer[bravais_lattice].top_unit_cell, dest=0)
                comm.send(optimizer[bravais_lattice].top_M20, dest=0)
                comm.send(optimizer[bravais_lattice].top_Minfo, dest=0)
                comm.send(optimizer[bravais_lattice].top_spacegroup, dest=0)
                comm.send(optimizer[bravais_lattice].top_n_indexed, dest=0)
                if triplet_obs is not None:
                    comm.send(optimizer[bravais_lattice].top_n_indexed_triplets, dest=0)
                    comm.send(optimizer[bravais_lattice].top_M_triplets, dest=0)

    if rank == 0:
        top_M_triplets_out = top_M_triplets if triplet_obs is not None else None
        top_n_indexed_triplets_out = top_n_indexed_triplets if triplet_obs is not None else None
        _write_output(args, top_unit_cell, top_M20, top_Minfo, top_spacegroup,
                      top_n_indexed, top_M_triplets_out, top_n_indexed_triplets_out,
                      triplet_obs)
    logger.info('Finished gathering optimization results')


def _run_mp(args, peak_list, triplet_obs, n_procs):
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, run_mp_bl, shutdown_mp_workers

    broadening_tag = '1'
    n_top_candidates = 20

    optimizers, processes, task_queues = setup_mp_optimizers(
        n_procs, broadening_tag, n_candidates_scale=1
    )

    top_unit_cell = {}
    top_M20 = {}
    top_Minfo = {}
    top_spacegroup = {}
    top_n_indexed = {}
    top_M_triplets = {}
    top_n_indexed_triplets = {}

    for bravais_lattice in BRAVAIS_LATTICES:
        run_mp_bl(
            optimizers[bravais_lattice],
            bravais_lattice,
            task_queues,
            q2=peak_list,
            triplets=triplet_obs,
            zero_error=args.zero_error,
            wavelength=args.wavelength,
            n_top=n_top_candidates,
        )
        opt = optimizers[bravais_lattice]
        top_unit_cell[bravais_lattice] = opt.top_unit_cell
        top_M20[bravais_lattice] = opt.top_M20
        top_Minfo[bravais_lattice] = opt.top_Minfo
        top_spacegroup[bravais_lattice] = opt.top_spacegroup
        top_n_indexed[bravais_lattice] = opt.top_n_indexed
        if triplet_obs is not None:
            top_M_triplets[bravais_lattice] = opt.top_M_triplets
            top_n_indexed_triplets[bravais_lattice] = opt.top_n_indexed_triplets

    shutdown_mp_workers(processes, task_queues)

    top_M_triplets_out = top_M_triplets if triplet_obs is not None else None
    top_n_indexed_triplets_out = top_n_indexed_triplets if triplet_obs is not None else None
    _write_output(args, top_unit_cell, top_M20, top_Minfo, top_spacegroup,
                  top_n_indexed, top_M_triplets_out, top_n_indexed_triplets_out,
                  triplet_obs)


def main():
    args = _parse_args()
    peak_list, triplet_obs = _load_peaks(args)
    if args.mpi:
        _run_mpi(args, peak_list, triplet_obs)
    else:
        _run_mp(args, peak_list, triplet_obs, n_procs=args.nproc)


if __name__ == "__main__":
    main()
