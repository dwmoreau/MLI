#!/bin/bash
# S12 -- the four contaminated / sparse bundles, fully retained.
#
#   sbatch submit_fom_contaminant_retained.sh
#   then, when all four tasks are done:
#   sbatch --dependency=afterok:<jobid> submit_fom_contaminant_postprocess.sh
#
# WHY THIS EXISTS. Everything S12 reports rests on three condition bundles -- 0.1x, 1x and 2x
# Gaussian peak-position error, all with NO contaminants. The benchmark HAS contaminated bundles and
# the combiner is FITTED on them; it has never been REPORTED on one, because a rank claim needs a
# pool nothing was thinned from and the fully retained pool was generated with the three severity
# bundles alone (C2-R-024).
#
# That is the gap DWMM named on 2026-09-04: real patterns carry one or two peaks no cell can index,
# and nothing in the record says what the learned score does with them. This job closes it.
#
#   task 0  sparse2       c2_error1_cont1_drop2   1 contaminant, 2 peaks dropped, 31 peaks
#   task 1  sparse4       c2_error1_cont1_drop4   1 contaminant, 4 peaks dropped, 31 peaks
#   task 2  sparse6       c2_error1_cont1_drop6   1 contaminant, 6 peaks dropped, 31 peaks
#   task 3  contaminated  c2_error1_cont2         2 contaminants, no dropout, 60 peaks
#
# `icept4` (a 4x sigma floor) and `phase3` (three second-phase lines from QAHDIP) are the other two
# bundles the retained pool lacks. They are not in this job because they were not asked for; add
# `error_shape` and `second_phase` to CONDITIONS to include them.
#
# **THE SAME 530 CRYSTALS AS THE EXISTING POOL, AND THAT IS THE POINT.** `S08_floor_entries.csv` is
# the entry list `fom_full_c2_pool` was built from -- verified 530 of 530 against the pool's own
# `fom-dev` ids. Reusing it makes every new bundle PAIRED with the three already reported: the same
# crystal under a new condition, so a McNemar over crystals is valid and a difference cannot be a
# difference of populations. Do not re-derive this list.
#
# COST. 530 crystals x 4 bundles = 2 120 cells. The hard-stratum arm measured 1 080 cells in under
# half a node-hour on this configuration, so ~0.9 node-hours of generation. `--no-subsample` keeps
# every candidate, which is what makes the ranks exact.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_cont_c2
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH --array=0-3
#SBATCH -o fom_cont_c2_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
ENTRIES="$REPO/docs/fom_campaign2/artifacts/S08_floor_entries.csv"
OUTROOT="$SCRATCH/fom_campaign2/contaminant_retained"
SEED=12345
OPTIMIZER_SEED=12345
NPOOLS=64
POOLSIZE=2

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first." >&2
    exit 1
fi
if [ ! -f "$ENTRIES" ]; then
    echo "FATAL: entry list missing at $ENTRIES" >&2
    echo "It is the list fom_full_c2_pool was built from. Do NOT re-derive it: a different list" >&2
    echo "makes the new bundles a different population and the comparison unpaired." >&2
    exit 1
fi

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads)." >&2
fi

CONDITIONS=("sparse2" "sparse4" "sparse6" "contaminated")
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$CONDITION" ]; then
    echo "FATAL: no condition for array index '$SLURM_ARRAY_TASK_ID'" >&2
    exit 1
fi
TAG=$("$PYTHON" run_fom_dump.py --condition "$CONDITION" --print-tag)
OUTDIR="$OUTROOT/$TAG"

set -e

echo "=== fully retained: condition $CONDITION ($TAG) -> $OUTDIR ==="
echo "    530 fom-dev crystals, the SAME list fom_full_c2_pool was built from, every candidate kept"
"$PYTHON" run_fom_dump.py \
    --condition "$CONDITION" \
    --split-manifest "$MANIFEST" \
    --entry-ids-file "$ENTRIES" \
    --seed "$SEED" \
    --optimizer-seed "$OPTIMIZER_SEED" \
    --n-pools "$NPOOLS" \
    --pool-size "$POOLSIZE" \
    --no-subsample \
    --predownsample-entries 0 \
    --out-dir "$OUTDIR"

echo
echo "DONE: $CONDITION -> $OUTDIR"
du -sh "$OUTDIR"
echo
echo "When ALL FOUR tasks are done -- check with"
echo "  sacct -j \$SLURM_ARRAY_JOB_ID --format=JobID,State,Elapsed,MaxRSS"
echo "then consolidate and build the sidecars:"
echo "  sbatch submit_fom_contaminant_postprocess.sh"
