#!/bin/bash
# Generate the candidate pools the generator mix is fitted against.
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
#   export MLI_OUT=$SCRATCH/ensemble_pools
#   sbatch mlindex/scripts/submit_ensemble_pools.sh
#
# Then, on a laptop, as often as the score changes -- this part needs no cluster and no MPI:
#
#   python -m mlindex.scripts.run_ensemble_refine --stage fit \
#       --pools <pulled copy of $MLI_OUT> --roc-dir <the measured convergence curves> \
#       --out-dir results/mix --variant shipped --variant expected --variant capped
#
# LAUNCHED WITH mpiexec, NOT srun. This driver is mpi4py, and a conda-built mpi4py does not read
# the process-management information srun hands out: every task then comes up in its own world of
# size one, believes it is the only rank, and runs the WHOLE job. That is N times the cost for one
# job's work and N processes writing one file, and it has actually happened here. The driver
# refuses to start when the launcher says it made several tasks and MPI reports a world of one.
# Set MLI_LAUNCHER yourself only if
#   srun -n 4 $MLI_PYTHON -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
# prints 4. None of this applies to run_benchmark.py, which must NOT be wrapped in srun because it
# makes its own processes and a bare `srun -n 1` pins them all to one core.
#
# WHAT IT COSTS. Measured on the development laptop: about 2 s per crystal on the most expensive
# lattice, generating twice the shipped budget from each of three generators, plus about 10 s to
# load one lattice's models. Fourteen lattices at 300 crystals is roughly two and a half hours of
# one core, so a few minutes of wall clock over a node. The two hours below is generous.
#
# WHAT IT WRITES. One .npz a lattice holding every candidate's position and its distance to the
# true cell, per generator, plus a manifest carrying the commit, the seed, the platform and each
# lattice's shipped mix and budget. The manifest is rewritten after every lattice, so a run that
# dies partway still leaves usable pools. Expect a few hundred MB a lattice on the low-symmetry
# ones.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J ensemble_pools
#SBATCH -A lcls
#SBATCH -t 2:00:00
#SBATCH -o ensemble_pools_%j.out

set -euo pipefail

: "${MLI_PYTHON:?set MLI_PYTHON to the interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the repository checkout}"
: "${MLI_OUT:?set MLI_OUT to the directory the pools are written to}"

MLI_RANKS="${MLI_RANKS:-128}"
MLI_LAUNCHER="${MLI_LAUNCHER:-mpiexec -n $MLI_RANKS}"
MLI_LATTICES="${MLI_LATTICES:-cF,cI,cP,hP,hR,tI,tP,oC,oF,oI,oP,mC,mP,aP}"
MLI_DATASETS="${MLI_DATASETS:-$MLI_REPO/mlindex/data/generated_datasets}"
MLI_MODELS="${MLI_MODELS:-$MLI_REPO/mlindex/models}"
MLI_N_ENTRIES="${MLI_N_ENTRIES:-300}"
MLI_BUDGET_SCALE="${MLI_BUDGET_SCALE:-2}"
MLI_SEED="${MLI_SEED:-12345}"

for MLI_PATH in "$MLI_REPO" "$MLI_DATASETS" "$MLI_MODELS"; do
    if [ ! -d "$MLI_PATH" ]; then
        echo "FATAL: no directory at $MLI_PATH" >&2
        exit 1
    fi
done

echo "ranks $MLI_RANKS | lattices $MLI_LATTICES | $MLI_N_ENTRIES crystals | budget x$MLI_BUDGET_SCALE"
echo "out $MLI_OUT"
cd "$MLI_REPO"
git rev-parse HEAD

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1 ATLAS_NUM_THREADS=1
export KERAS_BACKEND=torch

$MLI_LAUNCHER "$MLI_PYTHON" -m mlindex.scripts.run_ensemble_refine \
    --stage generate \
    --bravais-lattices "$MLI_LATTICES" \
    --dataset-directory "$MLI_DATASETS" \
    --models-directory "$MLI_MODELS" \
    --pools "$MLI_OUT" \
    --n-entries "$MLI_N_ENTRIES" \
    --budget-scale "$MLI_BUDGET_SCALE" \
    --seed "$MLI_SEED"

echo "done $MLI_OUT"
du -sh "$MLI_OUT"
