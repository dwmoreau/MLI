#!/bin/bash
# Re-derive every lattice's redistribution constants against the P09b ensemble score.
#
# RESEARCH CODE THAT NEEDS TO BE DELETED -- P09c. Goes with ensemble_arms.py when P09c closes.
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI          # checked out at the branch head
#   sbatch mlindex/scripts/submit_redistribution_fit.sh
#
# Then, on the laptop:
#
#   MLI_CAMPAIGN=fom_production docs/sync_record.sh pull-artifacts P09c_redistribution
#
# and read docs/fom_production/artifacts/P09c_redistribution/redistribution.csv. The row to take
# for each lattice is reduction per-pattern, split all; the two halves beside it say whether the
# answer reproduces, and at_grid_edge says whether the grid was wide enough. The chosen pair goes
# into that lattice's ENSEMBLE row, and only then can submit_ensemble_arms.sh run.
#
# WHAT IT READS. The candidate pools P09b's generator fractions were fitted on ($MLI_REDIST_POOLS, built
# at twice the shipped budget on the grid of contaminants and dropped peaks), the measured
# convergence curves and the measured clump discounts. It uses the fractions and budget in
# ENSEMBLE, so it must run from a checkout where those are P09b's.
#
# WHAT IT DOES. For each lattice except the three cubic ones, where no clump discount could be
# measured and the score cannot see redistribution: take --n-crystals crystals, build each one's
# pool at the ENSEMBLE fractions, redistribute it under each of about 42 settings around the
# shipped pair, --repeats times each, and score every result with the ensemble score and the clump
# discount. Writes redistribution.csv and one <lattice>_redistribution_scores.npz a lattice.
#
# WHAT IT COSTS. Measured on the development laptop at two repeats: about 20 s a crystal on oP and
# 50 s on aP. At three repeats and 1 000 crystals a lattice that is roughly 150 core-hours, so one
# to two hours over a node's 128 cores. Four hours is generous; the table is rewritten after every
# lattice, so a job killed at the limit keeps what it finished.
#
# NOT MPI and not wrapped in srun: it makes its own processes, and a bare `srun -n 1` pins them all
# to one core. Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.
#
# Variable names are MLI_-prefixed because bash silently discards an assignment to one of its own
# built-ins, and this script's own settings are MLI_REDIST_-prefixed so that none of them, left
# exported in a shell, changes another submit script's job.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J p09c_redistribution_fit
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH -o p09c_redistribution_fit_%j.out

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${MLI_PYTHON:?set MLI_PYTHON to the interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the checkout to run}"
MLI_REDIST_POOLS="${MLI_REDIST_POOLS:-$SCRATCH/p09b_pools}"
MLI_REDIST_ROC="${MLI_REDIST_ROC:-$MLI_REPO/docs/fom_production/artifacts/P08_inputs/data}"
MLI_REDIST_DISCOUNT="${MLI_REDIST_DISCOUNT:-$MLI_REPO/mlindex/characterization/clump_discount}"
MLI_REDIST_OUT="${MLI_REDIST_OUT:-$SCRATCH/fom_production/artifacts/P09c_redistribution}"
MLI_REDIST_N_CRYSTALS="${MLI_REDIST_N_CRYSTALS:-1000}"
MLI_REDIST_REPEATS="${MLI_REDIST_REPEATS:-3}"
MLI_REDIST_LATTICES="${MLI_REDIST_LATTICES:-hP,hR,tI,tP,oC,oF,oI,oP,mC,mP,aP}"

for MLI_PATH in "$MLI_REDIST_POOLS" "$MLI_REDIST_ROC" "$MLI_REDIST_DISCOUNT"; do
    if [ ! -d "$MLI_PATH" ]; then
        echo "FATAL: no directory at $MLI_PATH" >&2
        exit 1
    fi
done
if [ ! -f "$MLI_REDIST_POOLS/pools_manifest.json" ]; then
    echo "FATAL: $MLI_REDIST_POOLS has no pools_manifest.json" >&2
    exit 1
fi

MLI_CORES="${SLURM_CPUS_ON_NODE:-8}"
MLI_REDIST_NPROC="${MLI_REDIST_NPROC:-$((MLI_CORES / 2))}"

cd "$MLI_REPO"
echo "commit $(git rev-parse HEAD) | pools $MLI_REDIST_POOLS | processes $MLI_REDIST_NPROC | out $MLI_REDIST_OUT"
"$MLI_PYTHON" -m mlindex.scripts.run_ensemble_refine --stage redistribution \
    --bravais-lattices "$MLI_REDIST_LATTICES" \
    --pools "$MLI_REDIST_POOLS" \
    --roc-dir "$MLI_REDIST_ROC" \
    --clump-discount "$MLI_REDIST_DISCOUNT" \
    --out-dir "$MLI_REDIST_OUT" \
    --n-crystals "$MLI_REDIST_N_CRYSTALS" \
    --repeats "$MLI_REDIST_REPEATS" \
    --nproc "$MLI_REDIST_NPROC"
echo "done"
