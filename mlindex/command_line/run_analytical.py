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
from pathlib import Path
import numpy as np

from mlindex.optimization.AnalyticOptimizer import AnalyticOptimizer
from mlindex.optimization.MPIOptimizer import OptimizerWorker
from mlindex.command_line.run import (
    _load_peaks, _collect_results, _write_results, _conventional_cell,
    build_base_parser, BRAVAIS_LATTICES,
)


def main() -> None:
    parser = build_base_parser(description="Analytical high-symmetry indexing")
    parser.set_defaults(output_file="analytic_results.json")
    args = parser.parse_args()

    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    invalid = [bl for bl in bravais_lattices if bl not in BRAVAIS_LATTICES]
    if invalid:
        parser.error(f"Unknown Bravais lattices: {', '.join(invalid)}")

    q2_obs = _load_peaks(args)

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

    all_results = _conventional_cell(all_results)
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
                q2=q2_obs,
                zero_error=args.zero_error, wavelength=args.wavelength,
                n_top=20,
            )
            all_results = _collect_results(optimizers[bl], bl, all_results)
    finally:
        shutdown_mp_workers(processes, task_queues)
    all_results = _conventional_cell(all_results)
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
        all_results = _conventional_cell(all_results)
        output_file_base = str(Path(args.output_file).with_suffix(''))
        _write_results(all_results, output_file_base=output_file_base)


if __name__ == "__main__":
    main()
