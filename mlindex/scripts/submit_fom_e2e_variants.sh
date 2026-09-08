#!/bin/bash
# S18 -- the variant arms: the S15 arm at the DECIDED cut (3.5) with ONE optimizer setting changed,
# over the same crystals, seeds and pool size, so every cell pairs with the S15 arm it varies from.
# Six tasks: three variants x two populations. Each generates, stamps, writes the sidecars and
# reduces its own arm, and copies the entry table and manifest where `pull-artifacts` finds them.
#
#   # laptop, once, if not already done for S15:
#   python mlindex/scripts/run_fom_end_to_end.py --stage plan
#   docs/sync_record.sh push
#   # here:
#   sbatch submit_fom_e2e_variants.sh
#   # then, from the laptop:
#   docs/sync_record.sh pull-artifacts 'S15_reduced_*'
#   docs/sync_record.sh pull-artifacts 'S15_pool_*'
#   python mlindex/scripts/run_fom_end_to_end.py --stage variants
#
# THE THREE VARIANTS (mlindex/model_training/FomEndToEnd.py, VARIANTS):
#   _mask       assignment_statistic=posterior assignment_threshold=0.99   C2-Q-021: the posterior
#                                                                          peak filter in refine_cell
#   _nofilter   assignment_threshold=0.0                                   C2-Q-021: no peak filter;
#                                                                          the live contrast
#   _mask95     assignment_statistic=posterior assignment_threshold=0.95   C2-Q-034: the posterior
#                                                                          filter at rho's nominal
#                                                                          threshold (tasks 6-7)
#   _posterior  hkl_source=posterior                                       C2-Q-020: the analytic
#                                                                          posterior in place of the
#                                                                          IntegralFilter's network
# The shipped `rho` filter at 0.95 is the base arm itself and is refuted (decision 2026-08-28), so
# the question is "no filter against the posterior filter", read under plus_probation, M_sym and
# M20. Values are strings here; `--opt-param` parses them as JSON on the way in (0.99 -> float).
#
# WHY CUT 3.5. DWMM decided the deployment configuration on 2026-09-07: rank by plus_probation at
# prune cut 3.5. A variant is judged on whether it changes what ships, so it runs at 3.5 and pairs
# with S15's cut-3.5 arm, which must already exist here (the pre-flight refuses otherwise).
#
# PURGE. $SCRATCH is purge-managed (C2-R-014). Task 0 first copies both populations' cut-3.5
# pools to CFS with rsync --ignore-existing; the other tasks do not wait for it, and it is a copy
# rather than a move, so nothing here depends on it.
#
# COST. The S15 grid ran a general cut-3.5 arm in ~2 h of generation and the hard arm in under an
# hour; a variant costs the same. Walltime 6 h is slack.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the 128
# processes. Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_e2e_var
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH --array=0-7
#SBATCH -o fom_e2e_var_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO" || exit 1

ARTIFACTS="$REPO/docs/fom_campaign2/artifacts"
MANIFEST="$ARTIFACTS/S06_split_manifest.parquet"
DESIGN="$ARTIFACTS/S15_design.json"
OUTROOT="$SCRATCH/fom_campaign2"
OUT_ARTIFACTS="$OUTROOT/artifacts"
BACKUP=${BACKUP:-/global/cfs/cdirs/m4064/dwmoreau/fom_campaign2_pools}
MODEL="$REPO/mlindex/models/fom_combiner_c2_fullscale/plus_probation_seed12345"
CUT=3.5
NPOOLS=64
POOLSIZE=2
PROCESSES=64

# Tasks 0-5 ran on 2026-09-07. Tasks 6-7 (added 2026-09-08, C2-Q-034: the posterior filter at
# rho's own nominal threshold) are submitted alone with `sbatch --array=6-7 <this script>`; a
# full re-submit would regenerate nothing (existing bundles are skipped) but would re-reduce.
VARIANTS=(_mask _nofilter _posterior _mask _nofilter _posterior _mask95 _mask95)
POPULATIONS=(general general general hard hard hard general hard)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
POPULATION=${POPULATIONS[$SLURM_ARRAY_TASK_ID]}
ENTRIES="$ARTIFACTS/S15_entries_${POPULATION}.csv"
case "$VARIANT" in
    _mask)      OPTS=(--opt-param assignment_statistic=posterior --opt-param assignment_threshold=0.99) ;;
    _nofilter)  OPTS=(--opt-param assignment_threshold=0.0) ;;
    _posterior) OPTS=(--opt-param hkl_source=posterior) ;;
    _mask95)    OPTS=(--opt-param assignment_statistic=posterior --opt-param assignment_threshold=0.95) ;;
    *) echo "FATAL: unknown variant $VARIANT" >&2; exit 1 ;;
esac

BASE_POOL="$OUTROOT/e2e/$POPULATION/cut${CUT}_pool"
for NEEDED in "$MANIFEST" "$DESIGN" "$ENTRIES" "$BASE_POOL/manifest.json" "$BASE_POOL/entries.parquet" \
              "$MODEL/model.joblib" "$MODEL/calibrators.npz"; do
    if [ ! -f "$NEEDED" ]; then
        echo "FATAL: $NEEDED is missing." >&2
        echo "The design and entry lists come from --stage plan on the laptop + sync_record.sh push;" >&2
        echo "the base pool is S15's cut-3.5 arm (submit_fom_e2e_generate.sh); the model is rsynced" >&2
        echo "from the laptop (see submit_fom_e2e_reduce.sh)." >&2
        exit 1
    fi
done

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads)." >&2
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "=== task 0: copy the S15 cut-$CUT pools to CFS before the purge takes them (C2-R-014) ==="
    mkdir -p "$BACKUP"
    for POP in general hard; do
        rsync -a --ignore-existing "$OUTROOT/e2e/$POP/cut${CUT}_pool/" "$BACKUP/e2e/$POP/cut${CUT}_pool/" \
            && echo "    $POP: -> $BACKUP/e2e/$POP/cut${CUT}_pool/" \
            || echo "WARNING: backup of $POP failed; the arms do not depend on it" >&2
    done
fi

set -e
mkdir -p "$OUT_ARTIFACTS"

echo "=== S18 generate: $POPULATION$VARIANT cut $CUT (${OPTS[*]}) -> $OUTROOT/e2e/$POPULATION$VARIANT/ ==="
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage generate \
    --population "$POPULATION" --arm-suffix "$VARIANT" --cut "$CUT" \
    --out-root "$OUTROOT" --split-manifest "$MANIFEST" \
    --n-pools "$NPOOLS" --pool-size "$POOLSIZE" "${OPTS[@]}"

echo "=== stamp the arm complete (refused if any bundle is missing its manifest) ==="
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage complete \
    --population "$POPULATION" --arm-suffix "$VARIANT" --cut "$CUT" --out-root "$OUTROOT"

echo "=== consolidate + the sidecars S12's model reads, each followed by its --verify ==="
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage sidecars \
    --population "$POPULATION" --arm-suffix "$VARIANT" --cut "$CUT" --out-root "$OUTROOT" \
    --processes "$PROCESSES" --python "$PYTHON" --execute

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] || [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "=== re-reduce the S15 base arm at cut $CUT, so its reduction carries the wrong-lattice columns ==="
    # `n_other_lattice_above_best_correct` / `n_same_lattice_above_best_correct` were added to the
    # reduction for S18; the S15 reductions predate them. Same code for every existing column, so
    # the numbers do not move -- the pull overwrites the laptop's S15 files with a superset.
    "$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage reduce \
        --population "$POPULATION" --cut "$CUT" --pool "$BASE_POOL" \
        --learned "plus_probation=$MODEL" --artifact-dir "$OUT_ARTIFACTS"
fi

echo "=== reduce: every merit over the variant's pool ==="
POOL="$OUTROOT/e2e/$POPULATION$VARIANT/cut${CUT}_pool"
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage reduce \
    --population "$POPULATION" --arm-suffix "$VARIANT" --cut "$CUT" --pool "$POOL" \
    --learned "plus_probation=$MODEL" --artifact-dir "$OUT_ARTIFACTS"

echo "=== the entry tables and manifests, flat, for pull-artifacts and the digest check ==="
for ARM in "$POPULATION" "$POPULATION$VARIANT"; do
    P="$OUTROOT/e2e/$ARM/cut${CUT}_pool"
    cp "$P/entries.parquet" "$OUT_ARTIFACTS/S15_pool_entries_${ARM}_cut${CUT}.parquet"
    cp "$P/manifest.json"   "$OUT_ARTIFACTS/S15_pool_manifest_${ARM}_cut${CUT}.json"
done

echo
echo "DONE: $POOL"
du -sh "$POOL" 2>/dev/null || true
echo
echo "When all six tasks are done, from the laptop:"
echo "  docs/sync_record.sh pull-artifacts 'S15_reduced_*'"
echo "  docs/sync_record.sh pull-artifacts 'S15_pool_*'"
echo "  python mlindex/scripts/run_fom_end_to_end.py --stage variants"
