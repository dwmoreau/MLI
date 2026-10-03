#!/bin/bash
# Tune the learned ranker on the P12 pools: export them, fit the grid, then fit the chosen point
# at three seeds and score it on fom-dev.
#
# From the root of the checkout to run, with the environment that has mlindex installed active:
#
#   conda activate /global/cfs/cdirs/m4064/dwmoreau/envs/onnx
#   cd /global/cfs/cdirs/m4064/dwmoreau/MLI
#   sbatch mlindex/scripts/submit_ranker_tuning.sh
#
# That runs the first half: the fourteen export tasks of this array, and -- submitted by task 0,
# to start once every export has succeeded -- the 25 fits below. Each fit writes its learning
# curve on the selection crystals (curve.csv). Those curves choose the encoding and the setting
# (artifacts/P13_hyperparameters.md, section 2). The second half is then one more command:
#
#   sbatch mlindex/scripts/submit_ranker_tuning.sh seeds <encoding> <learning rate> <leaves> <trees>
#   e.g.  sbatch mlindex/scripts/submit_ranker_tuning.sh seeds onehot 0.04 63 1200
#
# which fits that point at the three seeds and, when they have succeeded, scores all three on the
# four fom-dev exports.
#
# Nothing needs exporting. The checkout is the directory sbatch was run from, the interpreter is
# the `python` of the environment active then, and the pools are P12's, under
# $SCRATCH/fom_production/P12_pools/dadd59d/; MLI_REPO, MLI_PYTHON and MLI_P13_POOLS override
# them. The checkout must have no uncommitted changes to tracked files: every output names its
# commit, and the driver refuses to run otherwise.
#
# REHEARSE FIRST. On a laptop, with small pools from run_benchmark --stage generate:
#
#   MLI_P13_POOLS=<dir holding train_general and dev_general> SCRATCH=<scratch dir> \
#       bash mlindex/scripts/submit_ranker_tuning.sh rehearse
#
# runs every stage serially with one grid cell, a short curve and two seeds.
#
# THE EXPORTS (array tasks 0-13). Training rows for the 25 fits, and evaluation rows: the pool
# restricted to an M20 cut and a depth per lattice.
#
#   tasks  pool           cut  depth  holds
#   0-9    train_general  3.5  20     one condition bundle each: training rows for seeds 12345,
#                                     777 and 20260826, and the selection crystals' evaluation rows
#   10     dev_general    3.5  20     the report at the inherited deployment
#   11     dev_hard       3.5  20
#   12     dev_general    1.5  all    the report over the whole pool, for P16
#   13     dev_hard       1.5  all
#
# THE FITS (one node each, all of its cores).
#
#   0-17   the grid, seed 12345: encoding onehot and ordinal x learning rate 0.02, 0.04, 0.08 x
#          leaves 31, 63, 127, to 4000, 2000 and 1000 trees respectively
#   18-24  the encoding check at learning rate 0.04, 63 leaves, 2000 trees: onehot and ordinal at
#          seeds 777 and 20260826, and native (reference only, never exported) at all three
#
# RE-RUNNING. No stage writes over a result. A failed export is re-run by removing its directory's
# files for that bundle (export_<bundle>.json and the bundle's parquets) and resubmitting that task
# with --array=<task>; a failed fit by removing its directory and resubmitting
#   sbatch --array=<task> mlindex/scripts/submit_ranker_tuning.sh fit
# A partial resubmission of the exports does not submit the fits; submit them with the command
# above and --array=0-24.
#
# WHERE THE OUTPUT GOES. The export frames, tens of GB, stay on the cluster under
# $SCRATCH/fom_production/P13_ranker/export/<commit>/. The results -- each fit's model, ONNX file
# and learning curve, and each fom-dev evaluation -- go under
# $SCRATCH/fom_production/artifacts/P13_ranker/{fits,evaluate}/<commit>/, which, on the laptop,
#
#   MLI_CAMPAIGN=fom_production docs/sync_record.sh pull-artifacts P13_ranker
#
# copies back to docs/fom_production/artifacts/P13_ranker/.
#
# WALLTIME: not yet measured on Perlmutter. Estimated on the laptop (10 cores, 2026-10-01): an
# export bundle holds ~265 M candidate rows (~70 M in its largest lattice file), ~3.5 us a row, so
# well under an hour; a fit is 0.4-0.55 s a tree on ~3.4 M rows plus ~0.35 s a tree for its curve,
# so ~1 h for the longest (4000 trees). Six hours a task leaves room, and only the time used is
# charged.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core. Variable names are
# MLI_P13_-prefixed so a setting exported by another submit script cannot reach this one.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J ranker_export
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH --array=0-13
#SBATCH -o ranker_tuning_%x_%A_%a.out

set -euo pipefail

MLI_P13_MODE="${1:-export}"
MLI_P13_SEEDS=12345,777,20260826
MLI_P13_SELECTION=0.15
MLI_P13_CALIBRATION=0.2
MLI_P13_REPO="${MLI_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
MLI_P13_PYTHON="${MLI_PYTHON:-$(command -v python || true)}"
if [ ! -f "$MLI_P13_REPO/mlindex/scripts/run_ranker.py" ]; then
    echo "FATAL: $MLI_P13_REPO is not an MLI checkout. Run sbatch from the repository root, or set MLI_REPO." >&2
    exit 1
fi
if [ -z "$MLI_P13_PYTHON" ] || ! "$MLI_P13_PYTHON" -c "import mlindex" 2>/dev/null; then
    echo "FATAL: no python with mlindex installed ('$MLI_P13_PYTHON'). Activate that environment before sbatch, or set MLI_PYTHON." >&2
    exit 1
fi
if [ -z "${SCRATCH:-}" ]; then
    echo "FATAL: SCRATCH is not set; it is where the outputs go." >&2
    exit 1
fi
MLI_P13_POOLS="${MLI_P13_POOLS:-$SCRATCH/fom_production/P12_pools/dadd59d}"
cd "$MLI_P13_REPO"
MLI_P13_COMMIT="$(git rev-parse HEAD | cut -c1-7)"
MLI_P13_EXPORT="$SCRATCH/fom_production/P13_ranker/export"
MLI_P13_OUT="$SCRATCH/fom_production/artifacts/P13_ranker"
MLI_P13_TRAIN_EXPORT="$MLI_P13_EXPORT/$MLI_P13_COMMIT/train_general_cut3.5_depth20"
MLI_P13_CPUS=$(( ${SLURM_CPUS_ON_NODE:-8} / 2 ))

ranker() {
    "$MLI_P13_PYTHON" -m mlindex.scripts.run_ranker "$@"
}

export_task() {   # $1: task number 0-13
    local bundle
    if [ "$1" -le 9 ]; then
        bundle=$("$MLI_P13_PYTHON" -c "import sys; from mlindex.model_training import Benchmark; bundles = Benchmark.available_bundles(sys.argv[1]); assert len(bundles) == 10, f'train_general has {len(bundles)} condition bundles, not 10'; print(bundles[int(sys.argv[2])])" "$MLI_P13_POOLS/train_general" "$1")
        ranker --stage export --pool "$MLI_P13_POOLS/train_general" --bundle "$bundle" \
            --seeds "$MLI_P13_SEEDS" --selection-fraction "$MLI_P13_SELECTION" --out-dir "$MLI_P13_EXPORT"
        return
    fi
    case "$1" in
        10) ranker --stage export --pool "$MLI_P13_POOLS/dev_general" --out-dir "$MLI_P13_EXPORT" ;;
        11) ranker --stage export --pool "$MLI_P13_POOLS/dev_hard" --out-dir "$MLI_P13_EXPORT" ;;
        12) ranker --stage export --pool "$MLI_P13_POOLS/dev_general" --cut 1.5 --depth all --out-dir "$MLI_P13_EXPORT" ;;
        13) ranker --stage export --pool "$MLI_P13_POOLS/dev_hard" --cut 1.5 --depth all --out-dir "$MLI_P13_EXPORT" ;;
        *) echo "FATAL: export task $1 does not exist; they are 0-13." >&2; exit 1 ;;
    esac
}

# The 25 fits as "encoding learning-rate leaves trees seed", in task order.
fit_settings() {
    local encoding rate leaves
    for encoding in onehot ordinal; do
        for rate in 0.02 0.04 0.08; do
            for leaves in 31 63 127; do
                case "$rate" in 0.02) trees=4000 ;; 0.04) trees=2000 ;; 0.08) trees=1000 ;; esac
                echo "$encoding $rate $leaves $trees 12345"
            done
        done
    done
    for seed in 777 20260826; do
        echo "onehot 0.04 63 2000 $seed"
        echo "ordinal 0.04 63 2000 $seed"
    done
    for seed in 12345 777 20260826; do
        echo "native 0.04 63 2000 $seed"
    done
}

fit_one() {   # encoding rate leaves trees seed [checkpoint-every]
    OMP_NUM_THREADS="$MLI_P13_CPUS" ranker --stage fit --export-dir "$MLI_P13_TRAIN_EXPORT" \
        --out-dir "$MLI_P13_OUT/fits" --encoding "$1" --learning-rate "$2" \
        --max-leaf-nodes "$3" --max-iter "$4" --seed "$5" --checkpoint-every "${6:-100}" \
        --calibration-fraction "$MLI_P13_CALIBRATION"
}

evaluate_seeds() {   # encoding rate leaves trees
    local models=() seed export
    for seed in ${MLI_P13_SEEDS//,/ }; do
        models+=(--model-dir "$MLI_P13_OUT/fits/$MLI_P13_COMMIT/${1}_lr${2}_leaves${3}_iter${4}_seed${seed}")
    done
    for export in dev_general_cut3.5_depth20 dev_hard_cut3.5_depth20 dev_general_cut1.5_depthall dev_hard_cut1.5_depthall; do
        ranker --stage evaluate --export-dir "$MLI_P13_EXPORT/$MLI_P13_COMMIT/$export" \
            "${models[@]}" --out-dir "$MLI_P13_OUT/evaluate"
    done
}

echo "commit $MLI_P13_COMMIT | mode $MLI_P13_MODE | task ${SLURM_ARRAY_TASK_ID:-none} | pools $MLI_P13_POOLS | out $MLI_P13_OUT | python $MLI_P13_PYTHON"

case "$MLI_P13_MODE" in
    export)
        MLI_P13_TASK="${SLURM_ARRAY_TASK_ID:-0}"
        if [ "$MLI_P13_TASK" -eq 0 ] && [ "${SLURM_ARRAY_TASK_COUNT:-0}" -eq 14 ]; then
            sbatch --array=0-24 -t 6:00:00 -J ranker_fit --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
                mlindex/scripts/submit_ranker_tuning.sh fit
        fi
        export_task "$MLI_P13_TASK"
        ;;
    fit)
        MLI_P13_TASK="${SLURM_ARRAY_TASK_ID:?a fit runs as an array task; use --array=0-24}"
        MLI_P13_N_FITS=$(fit_settings | wc -l | tr -d ' ')
        if [ "$MLI_P13_TASK" -ge "$MLI_P13_N_FITS" ]; then
            echo "FATAL: fit task $MLI_P13_TASK does not exist; they are 0-$(( MLI_P13_N_FITS - 1 ))." >&2
            exit 1
        fi
        # shellcheck disable=SC2046
        fit_one $(fit_settings | sed -n "$(( MLI_P13_TASK + 1 ))p")
        ;;
    seeds)
        if [ "$#" -ne 5 ]; then
            echo "FATAL: seeds takes <encoding> <learning rate> <leaves> <trees>, e.g. seeds onehot 0.04 63 1200" >&2
            exit 1
        fi
        if [ -z "${MLI_P13_SEED_FIT:-}" ]; then
            # A plain sbatch of this mode runs as the header's 14-task array. Task 0 submits one
            # fit per seed, marked so they know they are the fits, and the evaluation to follow
            # them; every other task has nothing to do.
            if [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
                exit 0
            fi
            MLI_P13_JOB=$(sbatch --parsable --array=0-2 -t 6:00:00 -J ranker_seeds \
                --export=ALL,MLI_P13_SEED_FIT=1 \
                mlindex/scripts/submit_ranker_tuning.sh seeds "$2" "$3" "$4" "$5")
            sbatch --array=0 -t 12:00:00 -J ranker_evaluate --dependency="afterok:${MLI_P13_JOB}" \
                mlindex/scripts/submit_ranker_tuning.sh evaluate "$2" "$3" "$4" "$5"
            exit 0
        fi
        fit_one "$2" "$3" "$4" "$5" "$(echo "$MLI_P13_SEEDS" | cut -d, -f$(( SLURM_ARRAY_TASK_ID + 1 )))"
        ;;
    evaluate)
        evaluate_seeds "$2" "$3" "$4" "$5"
        ;;
    rehearse)
        # A rehearsal pool has a crystal or two a lattice; holding out half at each step leaves every part some.
        MLI_P13_SELECTION=0.5
        MLI_P13_CALIBRATION=0.5
        for MLI_P13_TASK in $(seq 0 13); do
            if [ "$MLI_P13_TASK" -eq 11 ] || [ "$MLI_P13_TASK" -eq 13 ]; then
                [ -d "$MLI_P13_POOLS/dev_hard" ] || continue
            fi
            export_task "$MLI_P13_TASK"
        done
        fit_one onehot 0.08 31 200 12345 50
        fit_one native 0.08 31 200 12345 50
        MLI_P13_SEEDS=12345,777
        fit_one onehot 0.08 31 200 777 50
        models=(--model-dir "$MLI_P13_OUT/fits/$MLI_P13_COMMIT/onehot_lr0.08_leaves31_iter200_seed12345"
                --model-dir "$MLI_P13_OUT/fits/$MLI_P13_COMMIT/onehot_lr0.08_leaves31_iter200_seed777")
        ranker --stage evaluate --export-dir "$MLI_P13_EXPORT/$MLI_P13_COMMIT/dev_general_cut3.5_depth20" \
            "${models[@]}" --out-dir "$MLI_P13_OUT/evaluate"
        echo "done rehearsal $MLI_P13_OUT"
        ;;
    *)
        echo "FATAL: unknown mode '$MLI_P13_MODE'; use no argument, fit, seeds, evaluate or rehearse." >&2
        exit 1
        ;;
esac
echo "done $MLI_P13_MODE"
