#!/bin/bash
# Generate the learned ranker's candidate pools: fom-train in full, and the two fom-dev samples.
#
# From the root of the checkout to run, with the environment that has mlindex installed active:
#
#   conda activate /global/cfs/cdirs/m4064/dwmoreau/envs/onnx
#   cd /global/cfs/cdirs/m4064/dwmoreau/MLI
#   sbatch mlindex/scripts/submit_benchmark_pools.sh
#
# That one command is the whole run. Array task 0 also submits the finalize job, which waits for
# every task of this array to succeed, merges the train shards and reduces all three arms.
#
# Nothing needs exporting. The checkout is the directory sbatch was run from, the interpreter is
# the `python` of the environment active then, and the split manifest is
# $MLI_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet, checked against its frozen
# sha256 before anything is drawn. Any of them can still be set: MLI_REPO, MLI_PYTHON,
# MLI_SPLIT_MANIFEST. The checkout must have no uncommitted changes to tracked files; the driver
# refuses otherwise, because the manifest would name a commit that is not the code that ran.
#
# REHEARSE FIRST, at one crystal a lattice, through this same script:
#
#   sbatch --array=0-3 mlindex/scripts/submit_benchmark_pools.sh rehearse
#
# It writes under .../rehearsal_<commit>/ and exercises every path the real run takes: two train
# shards, the finalize job, both sidecars and the reductions. Each pool prints "N s each"; that
# per-pattern rate is what WALLTIME below must be checked against before the real run.
#
# THE ARMS. Every pattern is searched in all fourteen Bravais lattices, at M20 cut 1.5, and every
# candidate is kept. --seed and --search-seed are 12345 throughout.
#
#   tasks   arm            split      crystals                         conditions  patterns
#   0-15    train_general  fom-train  all 11 396 (every lattice's)      all 10      113 960
#   16      dev_general    fom-dev    40 a lattice (cF 20, cI 30)       all 10        5 300
#   17      dev_hard       fom-dev    120 each of aP, mP, mC            5 hard        1 800
#
# The train arm is divided into 16 shards, one node each. Every shard draws the whole arm and
# builds the second-phase partners from all of it, so the division changes no pattern. The hard
# population on fom-train is read from train_general (aP, mP, mC under the five hard conditions);
# it is not generated separately.
#
# RE-RUNNING A TASK. A shard refuses to write where it has written before, so a failed shard is
# re-run by removing its stripes and stamp and resubmitting that task alone, then finalizing by
# hand once it has succeeded (a partial resubmission does not submit the finalize job):
#
#   rm -rf $SCRATCH/fom_production/P12_pools/<commit>/train_general/parts/007_*
#   rm -f  $SCRATCH/fom_production/P12_pools/<commit>/train_general/shards/007.json
#   sbatch --array=7 mlindex/scripts/submit_benchmark_pools.sh
#   sbatch --array=0 mlindex/scripts/submit_benchmark_pools.sh finalize
#
# `--array=0` is required on a finalize: without it the #SBATCH line below would start eighteen
# finalize jobs on the same arm, and the script refuses to run as more than one.
#
# A failed dev task is re-run by removing its whole arm directory. If any task fails, the finalize
# job's dependency can never be met and it stays pending; cancel it with scancel.
#
# WHERE THE OUTPUT GOES. Pools, ~850 GB for train_general, go to
# $SCRATCH/fom_production/P12_pools/<commit>/<arm>/. $SCRATCH is purged, so copy them somewhere
# durable once verified. The reduced per-entry tables go under
# $SCRATCH/fom_production/artifacts/P12_pools/<commit>/, which, on the laptop,
#
#   MLI_CAMPAIGN=fom_production docs/sync_record.sh pull-artifacts P12_pools
#
# copies back.
#
# WALLTIME AND MEMORY, measured on the first run (2026-09-30): a train shard took 2.4 h and peaked
# at ~295 GB, ~146 s a pattern and ~2.3 GB a process on a full node; dev_general 2.1 h, dev_hard
# 0.8 h; the finalize job 5.6 h at 248 GB; ~47 node-hours in all. Six hours a task and twelve for
# the finalize leave room on purpose: only the time used is charged, and a shard killed at the limit
# leaves no stamp and has to be run again whole. A rehearsal leaves most of a node idle and runs
# faster (80 s a pattern), so do not size a run from one.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the pools.
# Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.
#
# Variable names are MLI_P12_-prefixed so a setting left exported by another submit script cannot
# reach this one, and so none can collide with a bash built-in (GROUPS cost this project a run).

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J benchmark_pools
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH --array=0-17
#SBATCH -o benchmark_pools_%A_%a.out

set -euo pipefail

# One thread per process: the pools already fill the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MLI_P12_MODE="${1:-run}"
MLI_P12_SPLIT_SHA256=3dd52c5eb2546dacca3034ebd2fd052dcd2acd4a8f9af24ce972fe4e0a210969
MLI_P12_SEED=12345
MLI_P12_CUT=1.5

# The run and its rehearsal differ only in size and in where they write; each has its own
# finalize mode, which the array's task 0 submits.
case "$MLI_P12_MODE" in
    run|finalize)
        MLI_P12_N_SHARDS=16
        MLI_P12_TRAIN_PER_LATTICE=1800   # the largest lattice's count in fom-train: every crystal
        MLI_P12_TRAIN_CRYSTALS=11396
        MLI_P12_DEV_GENERAL_PER_LATTICE=40
        MLI_P12_DEV_HARD_PER_LATTICE=120
        MLI_P12_LABEL=""
        MLI_P12_FINALIZE_MODE=finalize
        ;;
    rehearse|finalize_rehearsal)
        MLI_P12_N_SHARDS=2
        MLI_P12_TRAIN_PER_LATTICE=1
        MLI_P12_TRAIN_CRYSTALS=14
        MLI_P12_DEV_GENERAL_PER_LATTICE=1
        MLI_P12_DEV_HARD_PER_LATTICE=1
        MLI_P12_LABEL="rehearsal_"
        MLI_P12_FINALIZE_MODE=finalize_rehearsal
        ;;
    *)
        echo "FATAL: unknown mode '$MLI_P12_MODE'; use no argument, 'rehearse' or 'finalize'." >&2
        exit 1
        ;;
esac

MLI_P12_REPO="${MLI_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
MLI_P12_PYTHON="${MLI_PYTHON:-$(command -v python || true)}"
if [ ! -f "$MLI_P12_REPO/mlindex/scripts/run_benchmark.py" ]; then
    echo "FATAL: $MLI_P12_REPO is not an MLI checkout. Run sbatch from the repository root, or set MLI_REPO." >&2
    exit 1
fi
if [ -z "$MLI_P12_PYTHON" ] || ! "$MLI_P12_PYTHON" -c "import mlindex" 2>/dev/null; then
    echo "FATAL: no python with mlindex installed ('$MLI_P12_PYTHON'). Activate that environment before sbatch, or set MLI_PYTHON." >&2
    exit 1
fi
MLI_P12_SPLIT_MANIFEST="${MLI_SPLIT_MANIFEST:-$MLI_P12_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet}"
if [ ! -f "$MLI_P12_SPLIT_MANIFEST" ]; then
    echo "FATAL: no split manifest at $MLI_P12_SPLIT_MANIFEST" >&2
    exit 1
fi

cd "$MLI_P12_REPO"
MLI_P12_COMMIT="$(git rev-parse --short=7 HEAD)"
MLI_P12_OUT="$SCRATCH/fom_production/P12_pools/${MLI_P12_LABEL}${MLI_P12_COMMIT}"
MLI_P12_TABLES="$SCRATCH/fom_production/artifacts/P12_pools/${MLI_P12_LABEL}${MLI_P12_COMMIT}"
MLI_P12_ARMS=(train_general dev_general dev_hard)

if [ "$MLI_P12_MODE" = "$MLI_P12_FINALIZE_MODE" ]; then
    if [ "${SLURM_ARRAY_TASK_COUNT:-1}" -ne 1 ]; then
        echo "FATAL: a finalize must run as one job, not ${SLURM_ARRAY_TASK_COUNT} array tasks merging the same arm at once. Submit it with --array=0." >&2
        exit 1
    fi
    echo "finalize commit $MLI_P12_COMMIT | out $MLI_P12_OUT"
    "$MLI_P12_PYTHON" -m mlindex.scripts.run_benchmark --stage finalize \
        --out-pool "$MLI_P12_OUT/train_general"
    MLI_P12_FOUND="$("$MLI_P12_PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['n_source_entries'])" "$MLI_P12_OUT/train_general/manifest.json")"
    if [ "$MLI_P12_FOUND" != "$MLI_P12_TRAIN_CRYSTALS" ]; then
        echo "FATAL: train_general drew $MLI_P12_FOUND crystals, not the $MLI_P12_TRAIN_CRYSTALS fom-train holds." >&2
        exit 1
    fi
    # One arm at a time: a condition bundle of the train arm is ~230 M candidate rows in memory.
    for MLI_P12_ARM in "${MLI_P12_ARMS[@]}"; do
        "$MLI_P12_PYTHON" -m mlindex.scripts.run_benchmark --stage reduce \
            --pool "$MLI_P12_OUT/$MLI_P12_ARM" --scores M20,M_sym \
            --out-dir "$MLI_P12_TABLES/$MLI_P12_ARM"
    done
    echo "done finalize $MLI_P12_OUT"
    exit 0
fi

MLI_P12_TASK="${SLURM_ARRAY_TASK_ID:-0}"
MLI_P12_N_TASKS=$(( MLI_P12_N_SHARDS + 2 ))
if [ "$MLI_P12_TASK" -ge "$MLI_P12_N_TASKS" ]; then
    echo "FATAL: task $MLI_P12_TASK does not exist; in mode $MLI_P12_MODE the tasks are 0-$(( MLI_P12_N_TASKS - 1 ))." >&2
    exit 1
fi

# Task 0 of a complete array submits the finalize job, to start once every task has succeeded.
# A partial resubmission (fewer tasks) does not, so a re-run never queues a second finalize.
if [ "$MLI_P12_TASK" -eq 0 ] && [ "${SLURM_ARRAY_TASK_COUNT:-0}" -eq "$MLI_P12_N_TASKS" ]; then
    sbatch --array=0 -t 12:00:00 -J benchmark_pools_finalize \
        -o "benchmark_pools_finalize_%j.out" \
        --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
        mlindex/scripts/submit_benchmark_pools.sh "$MLI_P12_FINALIZE_MODE"
fi

MLI_P12_PROCS=$(( ${SLURM_CPUS_ON_NODE:-8} / 2 ))
MLI_P12_SHARD_ARGS=()
if [ "$MLI_P12_TASK" -lt "$MLI_P12_N_SHARDS" ]; then
    MLI_P12_ARM=train_general
    MLI_P12_ARGS=(--split fom-train --population general
                  --per-lattice "$MLI_P12_TRAIN_PER_LATTICE")
    MLI_P12_SHARD_ARGS=(--shard "$MLI_P12_TASK" --n-shards "$MLI_P12_N_SHARDS")
elif [ "$MLI_P12_TASK" -eq "$MLI_P12_N_SHARDS" ]; then
    MLI_P12_ARM=dev_general
    MLI_P12_ARGS=(--split fom-dev --population general
                  --per-lattice "$MLI_P12_DEV_GENERAL_PER_LATTICE")
else
    MLI_P12_ARM=dev_hard
    MLI_P12_ARGS=(--split fom-dev --population hard
                  --per-lattice "$MLI_P12_DEV_HARD_PER_LATTICE")
fi

echo "commit $MLI_P12_COMMIT | mode $MLI_P12_MODE | task $MLI_P12_TASK $MLI_P12_ARM ${MLI_P12_SHARD_ARGS[*]+${MLI_P12_SHARD_ARGS[*]}} | processes $MLI_P12_PROCS | out $MLI_P12_OUT/$MLI_P12_ARM | python $MLI_P12_PYTHON"

"$MLI_P12_PYTHON" -m mlindex.scripts.run_benchmark --stage generate \
    --out-pool "$MLI_P12_OUT/$MLI_P12_ARM" \
    --split-manifest "$MLI_P12_SPLIT_MANIFEST" \
    --split-sha256 "$MLI_P12_SPLIT_SHA256" \
    "${MLI_P12_ARGS[@]}" \
    ${MLI_P12_SHARD_ARGS[@]+"${MLI_P12_SHARD_ARGS[@]}"} \
    --cut "$MLI_P12_CUT" \
    --seed "$MLI_P12_SEED" \
    --search-seed "$MLI_P12_SEED" \
    --n-pools "$MLI_P12_PROCS" \
    --pool-size 1

echo "done task $MLI_P12_TASK $MLI_P12_ARM"
