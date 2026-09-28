#!/bin/bash
# Replay the final refinement step under nine settings, score and reduce each: one task per run.
#
# RESEARCH CODE THAT NEEDS TO BE DELETED -- P10. Goes with p10_replay.py when P10 closes.
#
# From the root of the checkout to run, with the environment that has mlindex installed active:
#
#   conda activate /global/cfs/cdirs/m4064/dwmoreau/envs/onnx
#   cd /global/cfs/cdirs/m4064/dwmoreau/MLI
#   sbatch mlindex/scripts/submit_p10_replay.sh
#
# Nothing needs exporting. The checkout is the directory sbatch was run from, the interpreter is
# the `python` of the environment active then, and the split manifest is
# $MLI_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet. Any of them can still be set:
# MLI_REPO, MLI_PYTHON, MLI_SPLIT_MANIFEST.
#
# THE RUNS. Each writes one pool per setting (no final step; the shipped 0.95 before the
# minimum-peaks rule; thresholds 0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999), all from one search.
#
#   task  split      population  crystals a lattice  conditions  patterns
#   0     fom-train  general     80                  3 (error)   ~3 200
#   1     fom-train  hard        240                 5           ~3 600
#   2     fom-dev    general     40                  3 (error)   ~1 590
#   3     fom-dev    hard        120                 5           ~1 800
#
# The threshold is chosen on tasks 0-1; tasks 2-3 report the chosen one. fom-dev uses the sizes
# and conditions the P04b floors were measured on.
#
# Then, on the laptop:
#
#   MLI_CAMPAIGN=fom_production docs/sync_record.sh pull-artifacts P10_replay
#
# and compare the tables, e.g. for task 1:
#
#   T=docs/fom_production/artifacts/P10_replay/<commit>/train_hard
#   python -m mlindex.scripts.run_benchmark --stage contrast --scores M_sym,M20 \
#       --reference t0.95 --arm t0.95=$T/t0.95 --arm no_step=$T/no_step ... --out-dir $T/contrast
#
# WHERE THE OUTPUT GOES. Pools, tens of GB a run, go to $SCRATCH/p10_replay/<commit>/ and stay
# there. The reduced per-entry tables go under $SCRATCH/fom_production/artifacts/P10_replay/, which
# `sync_record.sh pull-artifacts` copies.
#
# WALLTIME. Measured on the laptop: 228 s a pattern for the search plus nine replays. Over 128
# processes the largest run (task 1) takes ~1.8 h, then ~1.2 h to score its nine pools in
# parallel. Six hours is generous on purpose; a job killed at the limit leaves unstamped pools.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the pools.
# Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.
#
# Variable names are MLI_P10R_-prefixed so a setting left exported by another submit script
# cannot reach this one.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J p10_replay
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH --array=0-3
#SBATCH -o p10_replay_%A_%a.out

set -euo pipefail

# One thread per process: the pools already fill the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MLI_P10R_NAMES=(train_general train_hard dev_general dev_hard)
MLI_P10R_SPLITS=(fom-train fom-train fom-dev fom-dev)
MLI_P10R_POPULATIONS=(general hard general hard)
MLI_P10R_PER_LATTICE=(80 240 40 120)
MLI_P10R_GENERAL_BUNDLES="b1_error0.5_cont0,b1_error1_cont0,b1_error2_cont0"
MLI_P10R_VARIANTS=(no_step t0.95_noA t0.00 t0.50 t0.80 t0.90 t0.95 t0.99 t0.999)

MLI_P10R_REPO="${MLI_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
MLI_P10R_PYTHON="${MLI_PYTHON:-$(command -v python || true)}"
if [ ! -f "$MLI_P10R_REPO/mlindex/scripts/p10_replay.py" ]; then
    echo "FATAL: $MLI_P10R_REPO is not an MLI checkout with p10_replay.py. Run sbatch from the repository root on branch p10-refinement, or set MLI_REPO." >&2
    exit 1
fi
if [ -z "$MLI_P10R_PYTHON" ] || ! "$MLI_P10R_PYTHON" -c "import mlindex" 2>/dev/null; then
    echo "FATAL: no python with mlindex installed ('$MLI_P10R_PYTHON'). Activate that environment before sbatch, or set MLI_PYTHON." >&2
    exit 1
fi
MLI_P10R_SPLIT_MANIFEST="${MLI_SPLIT_MANIFEST:-$MLI_P10R_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet}"
if [ ! -f "$MLI_P10R_SPLIT_MANIFEST" ]; then
    echo "FATAL: no split manifest at $MLI_P10R_SPLIT_MANIFEST" >&2
    exit 1
fi

MLI_P10R_TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$MLI_P10R_TASK" -ge "${#MLI_P10R_NAMES[@]}" ]; then
    echo "FATAL: task $MLI_P10R_TASK does not exist; the runs are 0-$(( ${#MLI_P10R_NAMES[@]} - 1 ))." >&2
    exit 1
fi
MLI_P10R_NAME="${MLI_P10R_NAMES[$MLI_P10R_TASK]}"
MLI_P10R_POPULATION="${MLI_P10R_POPULATIONS[$MLI_P10R_TASK]}"
MLI_P10R_BUNDLE_ARGS=()
if [ "$MLI_P10R_POPULATION" = "general" ]; then
    MLI_P10R_BUNDLE_ARGS=(--bundles "$MLI_P10R_GENERAL_BUNDLES")
fi

cd "$MLI_P10R_REPO"
MLI_P10R_COMMIT="$(git rev-parse --short=7 HEAD)"
MLI_P10R_OUT="$SCRATCH/p10_replay/$MLI_P10R_COMMIT/$MLI_P10R_NAME"
MLI_P10R_TABLES="$SCRATCH/fom_production/artifacts/P10_replay/$MLI_P10R_COMMIT/$MLI_P10R_NAME"
MLI_P10R_PROCS=$(( ${SLURM_CPUS_ON_NODE:-8} / 2 ))

echo "commit $MLI_P10R_COMMIT | task $MLI_P10R_TASK $MLI_P10R_NAME | processes $MLI_P10R_PROCS | out $MLI_P10R_OUT | python $MLI_P10R_PYTHON"

"$MLI_P10R_PYTHON" -m mlindex.scripts.p10_replay \
    --split-manifest "$MLI_P10R_SPLIT_MANIFEST" \
    --split "${MLI_P10R_SPLITS[$MLI_P10R_TASK]}" \
    --population "$MLI_P10R_POPULATION" \
    --per-lattice "${MLI_P10R_PER_LATTICE[$MLI_P10R_TASK]}" \
    ${MLI_P10R_BUNDLE_ARGS[@]+"${MLI_P10R_BUNDLE_ARGS[@]}"} \
    --n-procs "$MLI_P10R_PROCS" \
    --out-dir "$MLI_P10R_OUT"

# The nine pools are scored and reduced side by side; scoring one is a single process. Each job
# is waited on by its own pid, because a bare `wait` reports success whatever the jobs did.
MLI_P10R_PIDS=()
for MLI_P10R_VARIANT in "${MLI_P10R_VARIANTS[@]}"; do
    (
        "$MLI_P10R_PYTHON" -m mlindex.scripts.run_benchmark --stage sidecars \
            --pool "$MLI_P10R_OUT/$MLI_P10R_VARIANT"
        "$MLI_P10R_PYTHON" -m mlindex.scripts.run_benchmark --stage reduce \
            --pool "$MLI_P10R_OUT/$MLI_P10R_VARIANT" --scores M_sym,M20 \
            --out-dir "$MLI_P10R_TABLES"
    ) &
    MLI_P10R_PIDS+=($!)
done
MLI_P10R_FAILED=0
for MLI_P10R_PID in "${MLI_P10R_PIDS[@]}"; do
    wait "$MLI_P10R_PID" || MLI_P10R_FAILED=$(( MLI_P10R_FAILED + 1 ))
done
if [ "$MLI_P10R_FAILED" -gt 0 ]; then
    echo "FATAL: $MLI_P10R_FAILED of ${#MLI_P10R_VARIANTS[@]} pools failed to score or reduce; see above." >&2
    exit 1
fi

echo "done task $MLI_P10R_TASK $MLI_P10R_NAME"
