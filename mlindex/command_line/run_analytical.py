#!/usr/bin/env python3
"""Command‑line entry point for the analytical high‑symmetry optimizer.

The script mirrors ``mlindex.command_line.run`` but uses the lightweight
``AnalyticOptimizer`` which implements a guess‑and‑check candidate generation.
It operates serially (MPI ``COMM_SELF``) and loops over the bravais lattices
``cF``, ``cI``, ``cP``, ``hP``, ``hR``, ``tI`` and ``tP``.
"""

import argparse
from pathlib import Path
import numpy as np

from mlindex.optimization.AnalyticOptimizer import AnalyticOptimizer
from mlindex.optimization.MPIOptimizer import OptimizerWorker
from mlindex.command_line.run import _load_peaks, _collect_results, _write_results

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
    args = parser.parse_args()
    args.triplets_file = None   # _load_peaks requires this attribute; analytical mode has no triplets

    q2_obs, _ = _load_peaks(args)

    # Bravais lattices to process (the high‑symmetry set)
    bravais_lattices = ["cF", "cI", "cP", "hP", "hR", "tI", "tP", "oC", "oF", "oI", "oP"]
    n_ref_hkl_guess = {bl: 10 for bl in bravais_lattices}

    if args.mpi:
        _run_mpi_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess)
    elif args.nproc > 1:
        _run_mp_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess)
    else:
        _run_serial_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess)


def _run_serial_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess):
    from mpi4py import MPI
    comm = MPI.COMM_SELF

    all_results = []
    for bl in bravais_lattices:
        optimizer = AnalyticOptimizer(
            bravais_lattice=bl,
            comm=comm,
            n_peaks=q2_obs.size,
            n_ref_hkl_guess=n_ref_hkl_guess[bl],
        )
        optimizer.run(q2=q2_obs, zero_error=args.zero_error, wavelength=args.wavelength)
        all_results = _collect_results(optimizer, bl, all_results)

    output_file_base = str(Path(args.output_file).with_suffix(''))
    _write_results(all_results, output_file_base=output_file_base)


def _run_mp_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess):
    from mlindex.optimization.MPOptimizer import (
        setup_mp_analytic_optimizers, run_mp_bl, shutdown_mp_workers
    )

    optimizers, processes, task_queues = setup_mp_analytic_optimizers(
        args.nproc, q2_obs.size, n_ref_hkl_guess
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


def _run_mpi_analytical(args, q2_obs, bravais_lattices, n_ref_hkl_guess):
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
            )
        else:
            optimizer = OptimizerWorker(comm=comm, fom='M20')

        optimizer.run(q2=q2_obs, zero_error=args.zero_error, wavelength=args.wavelength)
        comm.barrier()

        if rank == 0:
            all_results = _collect_results(optimizer, bl, all_results)

    if rank == 0:
        output_file_base = str(Path(args.output_file).with_suffix(''))
        _write_results(all_results, output_file_base=output_file_base)


if __name__ == "__main__":
    main()
