#!/bin/bash
# Generate, score and reduce every P09c benchmark run: one array task per run.
#
# RESEARCH CODE THAT NEEDS TO BE DELETED -- P09c. Goes with ensemble_arms.py when P09c closes.
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI          # checked out at the branch head
#   export MLI_SPLIT_MANIFEST=$MLI_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet
#   sbatch mlindex/scripts/submit_ensemble_arms.sh
#
# Then, on the laptop:
#
#   MLI_CAMPAIGN=fom_production docs/sync_record.sh pull-artifacts P09c_arms
#   python -m mlindex.scripts.ensemble_arms compare \
#       --tables-dir docs/fom_production/artifacts/P09c_arms/tables \
#       --floor-dir docs/fom_production/artifacts/P04b_arms \
#       --out-dir docs/fom_production/artifacts/P09c_arms/compare
#
# WHICH RUNS. `python -m mlindex.scripts.ensemble_arms list` prints all of them with their index
# and the exact run_benchmark command each one is. The array covers every index; to re-run one
# that failed, `sbatch --array=<index> mlindex/scripts/submit_ensemble_arms.sh`. A run refuses to
# write into a directory that already holds something, so remove a failed run's pool first.
#
# ALL RUNS COME FROM ONE COMMIT. The code holds the new generator fractions and redistribution
# constants; a run testing an old setting names it on the command line, so nothing needs checking
# out between tasks and every pair of runs differs only in the settings the manifest records
# under `ensemble`. The runs refuse to start until the re-derived redistribution constants are in
# the code -- run submit_redistribution_fit.sh and land its answer first.
#
# WHERE THE OUTPUT GOES. Pools, tens of GB a run, go to $MLI_POOLS_DIR and stay on scratch. The
# reduced per-entry tables, a few MB, go to $MLI_TABLES_DIR, under the artifacts directory that
# `sync_record.sh pull-artifacts` copies -- so the pools must NOT be put there, or the pull moves
# them too.
#
# WALLTIME. A general run is ~1 590 patterns and a hard run ~1 800, at up to ~90 s a pattern on one
# core, over 128 pools: ~20-25 min each. Two hours is generous on purpose; a job killed at the limit
# leaves an unstamped run and the work is lost.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the pools.
# Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.
#
# Variable names are MLI_-prefixed because bash silently discards an assignment to one of its own
# built-ins.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J p09c_ensemble_arms
#SBATCH -A lcls
#SBATCH -t 2:00:00
#SBATCH --array=0-25
#SBATCH -o p09c_ensemble_arms_%A_%a.out

set -euo pipefail

# One thread per process: the pools already fill the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${MLI_PYTHON:?set MLI_PYTHON to the interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the checkout to run}"
: "${MLI_SPLIT_MANIFEST:?set MLI_SPLIT_MANIFEST to the frozen split manifest}"
MLI_POOLS_DIR="${MLI_POOLS_DIR:-$SCRATCH/p09c_pools}"
MLI_TABLES_DIR="${MLI_TABLES_DIR:-$SCRATCH/fom_production/artifacts/P09c_arms/tables}"

if [ ! -f "$MLI_SPLIT_MANIFEST" ]; then
    echo "FATAL: no split manifest at $MLI_SPLIT_MANIFEST" >&2
    exit 1
fi

MLI_CORES="${SLURM_CPUS_ON_NODE:-8}"
MLI_POOLS="${MLI_POOLS:-$((MLI_CORES / 2))}"
MLI_TASK="${SLURM_ARRAY_TASK_ID:-0}"

cd "$MLI_REPO"
echo "commit $(git rev-parse HEAD) | task $MLI_TASK | pools $MLI_POOLS | tables $MLI_TABLES_DIR"
"$MLI_PYTHON" -m mlindex.scripts.ensemble_arms generate \
    --index "$MLI_TASK" \
    --pools-dir "$MLI_POOLS_DIR" \
    --tables-dir "$MLI_TABLES_DIR" \
    --split-manifest "$MLI_SPLIT_MANIFEST" \
    --n-pools "$MLI_POOLS"
echo "done task $MLI_TASK"
