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

from mpi4py import MPI

from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.gsas import load_pkslst
from mlindex.optimization.AnalyticOptimizer import AnalyticOptimizer
from mlindex.optimization.Optimizer import OptimizerWorker

def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical high‑symmetry indexing")
    parser.add_argument("--peak-file", type=str, required=True, help="Peak list file (.npy or .pkslst)")
    parser.add_argument("--wavelength", type=float, help="Wavelength for .pkslst files")
    parser.add_argument(
        "--output-file",
        type=str,
        default="analytic_results.json",
        help="File to write JSON results",
    )
    args = parser.parse_args()

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
    # Serial communicator for the AnalyticOptimizer
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

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

        # Run the optimizer; the parent class expects the observed q2 array via the ``q2`` argument
        optimizer.run(q2=q2_obs)
        comm.barrier()
        
        # The parent ``OptimizerManager`` stores results in ``top_*`` attributes.
        # We replicate the aggregation logic from ``mlindex.command_line.run`` to
        # build full unit‑cell arrays from the partial representation returned by
        # the analytic optimizer.
        if rank == 0:
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
                elif bl in ['oC',  'oF',  'oI',  'oP']:
                    unit_cell = np.array([
                        partial_unit_cell[0], partial_unit_cell[1], partial_unit_cell[2],
                        np.pi/2, np.pi/2, np.pi/2
                        ])
                else:
                    # Should not occur with the current set, but fallback to the raw values
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

    # Write combined results to JSON
    if rank == 0:
        all_results = pd.DataFrame(all_results)
        all_results.sort_values(by='M20', ascending=False, inplace=True, ignore_index=True)
        all_results.to_json(args.output_file)
        print(all_results[:20])
        print(f"Analytical indexing completed. Results written to {args.output_file}")


if __name__ == "__main__":
    main()
