#!/bin/bash
# S15 -- score every ranking merit over every arm, where the pools sit. One job.
#
#   sbatch --dependency=afterok:<generate jobid> submit_fom_e2e_reduce.sh
#   # then, from the laptop:
#   docs/sync_record.sh pull-artifacts 'S15_*'
#   python mlindex/scripts/run_fom_end_to_end.py --stage analyse
#   python mlindex/scripts/run_fom_end_to_end.py --stage figure
#   python mlindex/scripts/run_fom_end_to_end.py --stage report
#
# WHAT IT SCORES. M20 and `M_sym` (stored columns), S12's `plus_probation` at full scale (the tree
# plus its per-lattice calibrators, `mlindex/models/fom_combiner_c2_fullscale/`), and the two
# floors, in one pass per pool through `FomMetrics.reduce_many`. The reduction is one row per
# (crystal, condition) and is the sufficient statistic: both pool depths, every threshold and every
# stratum are computed from it on the laptop. Thresholds are S12's and are frozen in
# S15_design.json; nothing is chosen here.
#
# THE CUT-1.5 ARM is the general population's fourth real run and lives in two pools already on
# disk: the three clean bundles in `full_pool` (S08) and the five contaminated ones in
# `contaminant_pool` (S12). Both are fully retained -- the reduce refuses a thinned pool -- and
# both need their `merits/` and `structural/` sidecars, which the contaminant job wrote and the
# full-retained one may not have here: the guard below says what to do.
#
# THE RESTRICTION PASS replays `prune_below_m20` on the cut-1.5 pools at 5.0, 3.5 and 3.0 and
# scores the same merits, so the analyse stage can put "what a restriction says" beside "what the
# real run did" -- campaign 1 measured five entries of 210 in one direction and S15 quantifies it
# here. It is the long pole (~90 min a pool, C2-F-139); walltime 8 h.
#
# WHY THE MODELS ARE CHECKED FIRST. A reduce that silently loads the wrong models produces a table
# that looks entirely normal (C2-F-141).

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_e2e_reduce
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 8:00:00
#SBATCH -o fom_e2e_reduce_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO" || exit 1

OUTROOT="$SCRATCH/fom_campaign2"
MODEL="$REPO/mlindex/models/fom_combiner_c2_fullscale/plus_probation_seed12345"
FULL_POOL="$OUTROOT/full_pool"
CONTAM_POOL="$OUTROOT/contaminant_pool"

if [ ! -f "$MODEL/model.joblib" ] || [ ! -f "$MODEL/calibrators.npz" ]; then
    echo "FATAL: S12's full-scale model is not at $MODEL." >&2
    echo "From the laptop:" >&2
    echo "  rsync -avz mlindex/models/fom_combiner_c2_fullscale/plus_probation_seed12345/ \\" >&2
    echo "      <nersc>:$MODEL/" >&2
    exit 1
fi
if [ ! -f "$REPO/docs/fom_campaign2/artifacts/S15_design.json" ]; then
    echo "FATAL: S15_design.json missing; run --stage plan on the laptop and sync_record.sh push" >&2
    exit 1
fi

set -e
# Reductions go to $SCRATCH/fom_campaign2/artifacts, which is where `docs/sync_record.sh
# pull-artifacts` reads (C2-F-135: an export that wrote elsewhere moved the wrong files).
ARTIFACTS="$OUTROOT/artifacts"
mkdir -p "$ARTIFACTS"

echo "=== the six generated arms ==="
for POPULATION in general hard; do
    for CUT in 5.0 3.5 3.0; do
        POOL="$OUTROOT/e2e/$POPULATION/cut${CUT%.0}_pool"
        if [ ! -f "$POOL/manifest.json" ]; then
            echo "FATAL: $POOL has no manifest; the generate array did not finish this arm" >&2
            exit 1
        fi
        echo "--- $POPULATION cut $CUT: $POOL"
        "$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage reduce \
            --population "$POPULATION" --cut "$CUT" --pool "$POOL" \
            --learned "plus_probation=$MODEL" --artifact-dir "$ARTIFACTS"
    done
done

echo "=== the cut-1.5 arm: two fully retained pools, one reduction ==="
for POOL in "$FULL_POOL" "$CONTAM_POOL"; do
    if [ ! -d "$POOL/structural" ] || [ ! -d "$POOL/merits" ]; then
        echo "FATAL: $POOL lacks its merits/ or structural/ sidecar." >&2
        echo "For full_pool, write them here first (they exist on the laptop copy):" >&2
        echo "  $PYTHON mlindex/scripts/run_fom_floor_merits.py --pool $POOL --processes 64" >&2
        echo "  $PYTHON mlindex/scripts/run_fom_structural_features.py --pool $POOL --processes 64" >&2
        echo "or reduce the clean half on the laptop over mlindex/data/fom_full_c2_pool." >&2
        exit 1
    fi
done
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage reduce \
    --population general --cut 1.5 --pool "$FULL_POOL" --pool "$CONTAM_POOL" \
    --learned "plus_probation=$MODEL" --artifact-dir "$ARTIFACTS"

echo "=== restriction versus run: the cut-1.5 pools replayed at 5.0, 3.5, 3.0 ==="
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage restrict \
    --population general --pool "$FULL_POOL" --pool "$CONTAM_POOL" --cuts 5.0,3.5,3.0 \
    --learned "plus_probation=$MODEL" --artifact-dir "$ARTIFACTS"

echo
echo "DONE. Reductions are in $ARTIFACTS/S15_reduced_*"
echo "From the laptop:"
echo "  docs/sync_record.sh pull-artifacts 'S15_*'"
echo "  python mlindex/scripts/run_fom_end_to_end.py --stage analyse \\"
echo "      --existing-pool general:1.5=mlindex/data/fom_full_c2_pool"
echo "  python mlindex/scripts/run_fom_end_to_end.py --stage figure"
echo "  python mlindex/scripts/run_fom_end_to_end.py --stage report"
