#!/usr/bin/env python3
"""Command‑line entry point for the analytical high‑symmetry optimizer.

The script mirrors ``mlindex.command_line.run`` but uses the lightweight
``AnalyticOptimizer`` which implements a guess‑and‑check candidate generation.
It operates serially (MPI ``COMM_SELF``) and loops over the bravais lattices
``cF``, ``cI``, ``cP``, ``hP``, ``hR``, ``tI`` and ``tP``.
"""

import argparse
import pandas as pd
from pathlib import Path
import json
import numpy as np

from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.gsas import load_pkslst
from mlindex.optimization.AnalyticOptimizer import AnalyticOptimizer
from mlindex.optimization.MPIOptimizer import OptimizerWorker

def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical high‑symmetry indexing")
    parser.add_argument("--peak-file", type=str, required=True, help="Peak list file (.npy or .pkslst)")
    parser.add_argument("--wavelength", type=float, help="Wavelength for .pkslst files")
    parser.add_argument(
        "--zero-error",
        action='store_true',
        help="Apply a correction for zero point error in theta (requires --wavelength)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="analytic_results.json",
        help="File to write JSON results",
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

    if args.zero_error:
        assert args.wavelength, "--wavelength is required when --zero-error is set"

    # Load observed q2 values (same logic as the original CLI)
    peak_path = Path(args.peak_file)
    if not peak_path.exists():
        raise FileNotFoundError(f"Peak file not found: {peak_path}")
    if peak_path.suffix == ".npy":
        q2_obs = np.load(peak_path).astype(float)
    elif peak_path.suffix == ".pkslst":
        if args.wavelength is None:
            raise RuntimeError("--wavelength is required for .pkslst files")
        q2_obs = load_pkslst(str(peak_path), args.wavelength).astype(float)
    else:
        raise ValueError("Unsupported peak file format: must be .npy or .pkslst")

    # Bravais lattices to process (the high‑symmetry set)
    bravais_lattices = ["cF", "cI", "cP", "hP", "hR", "tI", "tP", "oC", "oF", "oI", "oP"]
    #bravais_lattices = ["cI", "tP"]
    n_ref_hkl_guess = {
        "cF": 10,
        "cI": 10,
        "cP": 10,
        "hP": 10,
        "hR": 10,
        "tI": 10,
        "tP": 10,
        "oC": 10,
        "oF": 10,
        "oI": 10,
        "oP": 10,
    }

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

    _write_results(all_results, args.output_file)


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

    _write_results(all_results, args.output_file)


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
        _write_results(all_results, args.output_file)


def _collect_results(optimizer, bl, all_results):
    for result_index in range(optimizer.top_M20.size):
        partial_unit_cell = optimizer.top_unit_cell[result_index]
        # Reconstruct the full six‑parameter unit cell based on the bravais lattice
        if bl in ["cF", "cI", "cP"]:
            unit_cell = np.array([
                partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[0],
                np.pi / 2, np.pi / 2, np.pi / 2
            ])
        elif bl == "hP":
            unit_cell = np.array([
                partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[1],
                2 * np.pi / 3, np.pi / 2, np.pi / 2
            ])
        elif bl == "hR":
            unit_cell = np.array([
                partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[0],
                partial_unit_cell[1], partial_unit_cell[1], partial_unit_cell[1]
            ])
        elif bl in ["tI", "tP"]:
            unit_cell = np.array([
                partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[1],
                np.pi / 2, np.pi / 2, np.pi / 2
            ])
        elif bl in ['oC', 'oF', 'oI', 'oP']:
            unit_cell = np.array([
                partial_unit_cell[0], partial_unit_cell[1], partial_unit_cell[2],
                np.pi / 2, np.pi / 2, np.pi / 2
            ])
        else:
            unit_cell = partial_unit_cell

        a, b, c, alpha, beta, gamma = unit_cell.tolist()
        result = {
            "M20": float(optimizer.top_M20[result_index]),
            "n_indexed": int(optimizer.top_n_indexed[result_index]),
            "bravais_lattice": bl,
            "volume": get_unit_cell_volume(unit_cell[np.newaxis])[0],
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        }
        all_results.append(result)
    return all_results


def _write_results(all_results, output_file):
    all_results = pd.DataFrame(all_results)
    all_results.sort_values(by='M20', ascending=False, inplace=True, ignore_index=True)
    all_results.to_json(output_file)
    print(all_results[:20])
    print(f"Analytical indexing completed. Results written to {output_file}")


if __name__ == "__main__":
    main()
