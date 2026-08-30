#!/bin/bash
# S07 -- regenerate the entries the first Benchmark B array lost, without redoing the rest.
#
# WHY THIS EXISTS. The first array (job 57701127) produced no rhombohedral entries at all.
# BRAVAIS_HOLOHEDRY['hR'] named the HEXAGONAL setting while `reindexed_unit_cell` stores hR on
# rhombohedral axes, so cctbx rejected every hR cell inside `is_degenerate`, which the driver calls
# once per entry INSIDE its per-entry failure guard. Every hR entry became a recorded skip; entries
# are striped across 4 shards x 64 pools, so no pool saw MAX_CONSECUTIVE_FAILURES in a row; all 24
# tasks exited 0 and 1 400 core + 210 mechanism entries were simply absent. Fixed in e773808, with
# two guards so a lattice cannot be lost quietly again. See C2-F-071.
#
# WHY A SUPPLEMENT RATHER THAN A RERUN. The pool is bit-identical across `n_pools` and shard count;
# only `pool_size` changes the search (C2-F-069), and this keeps it at 2. So the entries this
# generates are exactly the ones the first array would have written, and they merge into the
# existing pool. 7 840 cells of 106 339 -- about 3 node-hours against 43.
#
#   ~1 400 hR entries x 5 core bundles  +  ~210 hR entries x 4 mechanism bundles
#
# THE SHARD COUNT IS DELIBERATE AND MUST NOT BE 4 OR 1. Output files are named
# `<stream>_<tag>_shard<NN>of<NN>_pool<NN>`, and the consolidator refuses a basename it has already
# seen, because the same (shard, pool) consolidated twice doubles its rows. The first array used
# `of04` for core bundles and `of01` for mechanism ones; `of03` collides with neither.
#
# ================================================================================================
# RUNBOOK
#
#   1. ON THE LAPTOP: git push origin fom_campaign2 && docs/sync_record.sh push
#   2. HERE:          git pull && git log --oneline -1
#   3.                sbatch submit_fom_dump_supplement.sh
#
#      Tasks are independent and the driver skips pools already written, so a partial attempt
#      is resumable: `sbatch --array=5-8 submit_fom_dump_supplement.sh` re-runs the mechanism
#      half alone. The first attempt (2026-08-29) lost exactly those four to a wrong entry
#      list here; its five core tasks completed and must NOT be re-run.
#   4. When it finishes, consolidate BOTH roots together -- the consolidator merges a bundle that
#      appears in two directories and refuses any repeated shard file:
#
#        $PYTHON run_fom_dump_consolidate.py \
#            --dump-root $SCRATCH/fom_campaign2/benchmark \
#                        $SCRATCH/fom_campaign2/benchmark_supplement \
#            --out-dir   $SCRATCH/fom_campaign2/pool \
#            --artifact-dir $REPO/docs/fom_campaign2/artifacts \
#            --processes 9
#
#      Delete the old $SCRATCH/fom_campaign2/pool first; it was consolidated without hR.
#
#   5. Check the coverage table: every bundle's `n_missing_vs_arm` should be 0 except
#      c2_error1_cont0_phase3, whose ~101 are second-phase lines that could not be placed. Those
#      are expected and are recorded rather than intersected away (R14).
#   6. Then the gate. Record findings in STATUS_nersc_inbox.md, never STATUS.md.
# ================================================================================================

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_dump_supp
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 1:00:00
#SBATCH --array=0-8
#SBATCH -o fom_dump_supp_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
OUTROOT="$SCRATCH/fom_campaign2/benchmark_supplement"
# TWO id lists, one per arm. `--arm mechanism` restricts the manifest to the nested ~15 %
# subset BEFORE the id file is applied, and only 210 of the 1 400 hR entries are in it -- so
# handing the mechanism tasks the full list makes the driver refuse "1190 of 1400 requested
# entries were not found". That is the guard working correctly and the script being wrong; it
# cost the first supplement attempt all four of its mechanism tasks.
IDS_CORE="$SCRATCH/fom_campaign2/hR_ids_core.csv"
IDS_MECH="$SCRATCH/fom_campaign2/hR_ids_mechanism.csv"
SEED=12345

# MUST match the first array exactly: pool_size is part of the benchmark's identity (C2-F-069).
NPOOLS=64
POOLSIZE=2
NSHARDS=3
PREDOWNSAMPLE=4

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    exit 1
fi

# The entry lists, derived from the frozen manifest itself so they cannot drift from what the
# first array was given. Regenerated on every task: they are cheap, and a stale file from an
# earlier attempt is exactly the kind of thing that survives a fix.
mkdir -p "$(dirname "$IDS_CORE")"
$PYTHON "$REPO/mlindex/scripts/_hr_entry_lists.py" "$MANIFEST" "$IDS_CORE" "$IDS_MECH" \
    || exit 1

# One task per bundle. Each runs all NSHARDS shards in sequence -- an hR-only bundle is ~1 400
# entries at most, so the whole task is minutes, and this keeps the array small.
TASKS=(
    "control      core"
    "nominal      core"
    "noisy        core"
    "contaminated core"
    "second_phase core"
    "sparse2      mechanism"
    "sparse4      mechanism"
    "sparse6      mechanism"
    "error_shape  mechanism"
)
read -r CONDITION ARM <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

TAG=$($PYTHON run_fom_dump.py --condition "$CONDITION" --print-tag) || exit 1
ARM_FLAG=""
IDS="$IDS_CORE"
if [ "$ARM" = "mechanism" ]; then
    ARM_FLAG="--arm mechanism"
    IDS="$IDS_MECH"        # 210 entries, not 1 400 -- see the note above the two lists
fi

echo "task $SLURM_ARRAY_TASK_ID: $CONDITION ($ARM), hR only -> $TAG"
echo "  entry list: $IDS ($(($(wc -l < "$IDS") - 1)) entries)"

STATUS=0
for SHARD in $(seq 0 $((NSHARDS - 1))); do
    echo "--- shard $SHARD of $NSHARDS ---"
    $PYTHON run_fom_dump.py \
        --condition "$CONDITION" \
        $ARM_FLAG \
        --split-manifest "$MANIFEST" \
        --entry-ids-file "$IDS" \
        --n-pools "$NPOOLS" --pool-size "$POOLSIZE" \
        --shard "$SHARD" --n-shards "$NSHARDS" \
        --predownsample-entries "$PREDOWNSAMPLE" \
        --seed "$SEED" \
        --out-dir "$OUTROOT/$TAG" || STATUS=$?
done

echo "task $SLURM_ARRAY_TASK_ID exited $STATUS"
exit $STATUS
