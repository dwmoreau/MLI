#!/bin/bash
# Generate, score and reduce one batch of P09c's benchmark runs: one array task per run.
#
# RESEARCH CODE THAT NEEDS TO BE DELETED -- P09c. Goes with ensemble_arms.py when P09c closes.
#
# From the root of the checkout to run, with the environment that has mlindex installed active:
#
#   conda activate /global/cfs/cdirs/m4064/dwmoreau/envs/onnx
#   cd /global/cfs/cdirs/m4064/dwmoreau/MLI
#   sbatch mlindex/scripts/submit_ensemble_arms.sh                            # the fractions batch
#   MLI_BATCH=budget sbatch --array=0-17 mlindex/scripts/submit_ensemble_arms.sh   # the budget batch
#
# Nothing needs exporting. The checkout is the directory sbatch was run from, the interpreter is
# the `python` of the environment active then (the job inherits it), and the split manifest is
# the one the floor was measured on, $MLI_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet.
# Any of them can still be set: MLI_REPO, MLI_PYTHON, MLI_SPLIT_MANIFEST.
#
# WHICH RUNS. `python -m mlindex.scripts.ensemble_arms list` prints each batch and its runs by
# task number. The array below fits the default batch, `fractions` (4 runs); `budget` has 18 and
# needs the --array shown above. A whole-batch array that does not match its batch stops at once. To
# re-run a task that failed, `sbatch --array=<task> ...`, after removing that run's pool: a run
# refuses to write into a directory that already holds something. Both batches must come from one
# commit, since the budget runs are read against the fractions batch's `control`.
#
# Then, on the laptop:
#
#   MLI_CAMPAIGN=fom_production docs/sync_record.sh pull-artifacts P09c_arms
#   python -m mlindex.scripts.ensemble_arms compare \
#       --tables-dir docs/fom_production/artifacts/P09c_arms/tables \
#       --floor-dir docs/fom_production/artifacts/P04b_arms \
#       --out-dir docs/fom_production/artifacts/P09c_arms/compare
#
# ALL RUNS COME FROM ONE COMMIT. The code holds the new generator fractions; the run testing the
# old ones names them on the command line, so nothing needs checking out between tasks and every
# pair of runs differs only in the settings the manifest records under `ensemble`.
#
# WHERE THE OUTPUT GOES. Pools, tens of GB a run, go to $MLI_POOLS_DIR and stay on scratch. The
# reduced per-entry tables, a few MB, go to $MLI_TABLES_DIR, under the artifacts directory that
# `sync_record.sh pull-artifacts` copies -- so the pools must NOT be put there, or the pull moves
# them too.
#
# WALLTIME. A general run is ~1 590 patterns and a hard run ~1 800. Measured in P09c at 135 s a
# pattern on one node core over 128 pools: ~35 min a general run. Two hours is generous on purpose;
# a job killed at the limit leaves an unstamped run and the work is lost.
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
#SBATCH --array=0-3
#SBATCH -o p09c_ensemble_arms_%A_%a.out

set -euo pipefail

# One thread per process: the pools already fill the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MLI_BATCH="${MLI_BATCH:-fractions}"
MLI_REPO="${MLI_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
MLI_PYTHON="${MLI_PYTHON:-$(command -v python || true)}"
if [ ! -f "$MLI_REPO/mlindex/scripts/ensemble_arms.py" ]; then
    echo "FATAL: $MLI_REPO is not an MLI checkout. Run sbatch from the repository root, or set MLI_REPO." >&2
    exit 1
fi
if [ -z "$MLI_PYTHON" ] || ! "$MLI_PYTHON" -c "import mlindex" 2>/dev/null; then
    echo "FATAL: no python with mlindex installed ('$MLI_PYTHON'). Activate that environment before sbatch, or set MLI_PYTHON." >&2
    exit 1
fi
MLI_SPLIT_MANIFEST="${MLI_SPLIT_MANIFEST:-$MLI_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet}"
MLI_POOLS_DIR="${MLI_POOLS_DIR:-$SCRATCH/p09c_pools}"
MLI_TABLES_DIR="${MLI_TABLES_DIR:-$SCRATCH/fom_production/artifacts/P09c_arms/tables}"

if [ ! -f "$MLI_SPLIT_MANIFEST" ]; then
    echo "FATAL: no split manifest at $MLI_SPLIT_MANIFEST" >&2
    exit 1
fi

MLI_CORES="${SLURM_CPUS_ON_NODE:-8}"
MLI_POOLS="${MLI_POOLS:-$((MLI_CORES / 2))}"
# MLI_POOLS is a process count here, as in submit_benchmark_arms.sh. Refuse anything else rather
# than pass it on: a value left exported by another job reaches this line silently.
case "$MLI_POOLS" in
    ''|*[!0-9]*)
        echo "FATAL: MLI_POOLS must be a number of processes, got '$MLI_POOLS'. Unset it or set a number." >&2
        exit 1
        ;;
esac
MLI_TASK="${SLURM_ARRAY_TASK_ID:-0}"
# A whole-batch submission is an array running from 0 without gaps, and must be the batch's size;
# a re-run of chosen tasks is not, and is not checked.
MLI_ARRAY_SIZE=""
if [ "${SLURM_ARRAY_TASK_MIN:-}" = 0 ] && [ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ] \
        && [ "${SLURM_ARRAY_TASK_COUNT}" -eq $(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 )) ]; then
    MLI_ARRAY_SIZE="$SLURM_ARRAY_TASK_COUNT"
fi

cd "$MLI_REPO"
echo "commit $(git rev-parse HEAD) | batch $MLI_BATCH task $MLI_TASK | processes $MLI_POOLS | pools $MLI_POOLS_DIR | tables $MLI_TABLES_DIR | python $MLI_PYTHON"
"$MLI_PYTHON" -m mlindex.scripts.ensemble_arms generate \
    --batch "$MLI_BATCH" --task "$MLI_TASK" ${MLI_ARRAY_SIZE:+--array-size "$MLI_ARRAY_SIZE"} \
    --pools-dir "$MLI_POOLS_DIR" \
    --tables-dir "$MLI_TABLES_DIR" \
    --split-manifest "$MLI_SPLIT_MANIFEST" \
    --n-pools "$MLI_POOLS"
echo "done batch $MLI_BATCH task $MLI_TASK"
