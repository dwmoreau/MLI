#!/usr/bin/env python3
"""Command‑line entry point for the analytical high‑symmetry optimizer.

The script mirrors ``mlindex.command_line.run`` but uses the lightweight
``AnalyticOptimizer`` which implements a guess‑and‑check candidate generation.
It operates serially (MPI ``COMM_SELF``) and loops over the bravais lattices
``cF``, ``cI``, ``cP``, ``hP``, ``hR``, ``tI`` and ``tP``.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import argparse
from pathlib import Path
import numpy as np

from mlindex.optimization.AnalyticOptimizer import AnalyticOptimizer
from mlindex.optimization.MPIOptimizer import OptimizerWorker
from mlindex.command_line.run import _load_peaks, _collect_results, _write_results

_ALL_ANALYTIC_BL = ["cF", "cI", "cP", "hP", "hR", "tI", "tP", "oC", "oF", "oI", "oP", "mC", "mP", "aP"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical high‑symmetry indexing")
    parser.add_argument(
        "--peaks",
        type=str,
        help="Comma- or space-separated list of peak positions; units set by --peak-units (default: d-spacing in Å)"
    )
    parser.add_argument("--peak-file", type=str, help="Peak list file (.npy or .pkslst)")
    parser.add_argument(
        "--peak-units",
        type=str,
        choices=['d', 'q', 'q2', '2theta'],
        default="d",
        help="Units for peaks from --peaks or .npy files: 'd' = d-spacing (Å, default), 'q' = 1/d (Å⁻¹), 'q2' = 1/d² (Å⁻²), '2theta' = degrees (requires --wavelength). Not applied to .pkslst files.",
    )
    parser.add_argument("--wavelength", type=float, help="Wavelength for .pkslst files and --peak-units 2theta")
    parser.add_argument(
        "--zero-error",
        action='store_true',
        help="Apply a correction for zero point error in theta (requires --wavelength)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="analytic_results.json",
        help="File to write JSON results (a matching .txt file is also written)",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes for multiprocessing mode (default: 1 = serial)",
    )
    parser.add_argument(
        "--mpi",
        action='store_true',
        help="Use MPI for parallelism (requires mpiexec)",
    )
    parser.add_argument(
        "--bravais-lattices",
        type=str,
        default=",".join(_ALL_ANALYTIC_BL),
        help=f"Comma-separated Bravais lattices to attempt (default: {','.join(_ALL_ANALYTIC_BL)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
    )
    args = parser.parse_args()
    args.triplets_file = None   # _load_peaks requires this attribute; analytical mode has no triplets

    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    invalid = [bl for bl in bravais_lattices if bl not in _ALL_ANALYTIC_BL]
    if invalid:
        parser.error(f"Unknown Bravais lattices: {', '.join(invalid)}")

    q2_obs, _ = _load_peaks(args)

    n_ref_hkl_guess = {bl: 10 for bl in bravais_lattices}

    if args.mpi:
        _run_mpi_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess, seed=args.seed)
    elif args.nproc > 1:
        _run_mp_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess, seed=args.seed)
    else:
        _run_serial_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess, seed=args.seed)


def _run_serial_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess, seed=12345):
    from mlindex.optimization.MPOptimizer import LocalComm
    comm = LocalComm(n_ranks=1)

    all_results = []
    for bl in bravais_lattices:
        optimizer = AnalyticOptimizer(
            bravais_lattice=bl,
            comm=comm,
            n_peaks=q2_obs.size,
            n_ref_hkl_guess=n_ref_hkl_guess[bl],
            seed=seed,
        )
        optimizer.run(q2=q2_obs, zero_error=args.zero_error, wavelength=args.wavelength)
        all_results = _collect_results(optimizer, bl, all_results)

    output_file_base = str(Path(args.output_file).with_suffix(''))
    _write_results(all_results, output_file_base=output_file_base)


def _run_mp_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess, seed=12345):
    from mlindex.optimization.MPOptimizer import (
        setup_mp_analytic_optimizers, run_mp_bl, shutdown_mp_workers
    )
    optimizers, processes, task_queues = setup_mp_analytic_optimizers(
        args.nproc, q2_obs.size, n_ref_hkl_guess, bravais_lattices, seed=seed
    )

    all_results = []
    try:
        for bl in bravais_lattices:
            run_mp_bl(
                optimizers[bl], bl, task_queues,
                q2=q2_obs, triplets=None,
                zero_error=args.zero_error, wavelength=args.wavelength,
                n_top=20,
            )
            all_results = _collect_results(optimizers[bl], bl, all_results)
    finally:
        shutdown_mp_workers(processes, task_queues)
    output_file_base = str(Path(args.output_file).with_suffix(''))
    _write_results(all_results, output_file_base=output_file_base)


def _run_mpi_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess, seed=12345):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    all_results = []
    for bl in bravais_lattices:
        comm.barrier()
        if rank == 0:
            optimizer = AnalyticOptimizer(
                bravais_lattice=bl,
                comm=comm,
                n_peaks=q2_obs.size,
                n_ref_hkl_guess=n_ref_hkl_guess[bl],
                seed=seed,
            )
        else:
            optimizer = OptimizerWorker(comm=comm, fom='M20', seed=seed + rank)

        optimizer.run(q2=q2_obs, zero_error=args.zero_error, wavelength=args.wavelength)
        comm.barrier()

        if rank == 0:
            all_results = _collect_results(optimizer, bl, all_results)

    if rank == 0:
        output_file_base = str(Path(args.output_file).with_suffix(''))
        _write_results(all_results, output_file_base=output_file_base)


if __name__ == "__main__":
    main()
