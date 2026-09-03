#!/bin/bash
# S12 -- the full-scale export, one condition bundle per array task.
#
#   sbatch submit_fom_combiner_export.sh
#   then, from the laptop:
#       docs/sync_record.sh pull-artifacts 'S12_combiner_*fullscale*'
#       python mlindex/scripts/run_fom_combiner.py --stage fit \
#           --fit-frame 'docs/fom_campaign2/artifacts/S12_combiner_fit_frame_fullscale_*.parquet' \
#           --suffix _fullscale
#
# WHY THIS IS AN ARRAY AND submit_fom_combiner_fullscale.sh WAS NOT.
#
# That job ran out of a 6 h walltime having finished neither the hold-out sidecars nor the export,
# and lost two workers to the OOM killer on the way (C2-F-139). The export is the part that cannot
# be fixed with more walltime:
#
#   `bundle_frames` yields ONE FRAME PER CONDITION BUNDLE with all fourteen lattices together.
#   That is not negotiable -- the ranking is cross-lattice, and a per-lattice frame would compute
#   different context features and reduce a ranking that does not exist (PROTOCOL section 10).
#   At Benchmark B's scale one such frame is ~98 M rows joined against four sidecar directories.
#   `run_export_fit` builds nine of them, then nine more for the calibration half, single-process.
#
# The header of the old script budgeted 30 minutes for that. On a laptop the same code takes about
# half an hour on a pool twenty times smaller. So: one bundle per task, nine tasks at once, and the
# shards are globbed back together by `--stage fit`. Each task holds one bundle, so the peak memory
# is a ninth of what the serial version would have needed at its worst.
#
# **A LOST TASK IS REFUSED, NOT FITTED AROUND.** An array loses tasks quietly and the survivors
# still glob into a frame that loads and fits -- a model fitted on seven conditions while its
# write-up says nine, with no symptom but a slightly wrong number. Every shard records the bundle
# list its invocation was given, and `_assert_shards_complete` checks the union against what was
# read back. Check the array's exit codes too: `sacct -j <jobid> --format=JobID,State,MaxRSS`.
#
# THE SIDECARS THIS NEEDS MUST ALREADY EXIST. `structural`, `merits` and `merits_soft` are written
# and verified. `holdout_merits` is NOT, and FEATURE_GROUPS below therefore omits `holdout` -- see
# the note there before adding it back.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_export
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH --array=0-8
#SBATCH -o fom_export_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

POOL="$SCRATCH/fom_campaign2/pool"
OUTDIR="$SCRATCH/fom_campaign2/artifacts"
mkdir -p "$OUTDIR"

# The nine condition bundles, in the order `available_bundles` returns them. One per task.
BUNDLES=(c2_error0.1_cont0 c2_error1_cont0 c2_error1_cont0_icept4 c2_error1_cont0_phase3 \
         c2_error1_cont1_drop2 c2_error1_cont1_drop4 c2_error1_cont1_drop6 c2_error1_cont2 \
         c2_error2_cont0)
BUNDLE="${BUNDLES[$SLURM_ARRAY_TASK_ID]}"

# **`holdout` is deliberately NOT here, and this export should not wait for it.**
#
# `campaign1_raw` and `probation` come out of the `structural` sidecar rather than a directory of
# their own, so they are free. `soft` cost a pass and is complete and verified. `holdout` cost a
# pass that has now failed twice: nine files remain, they are the nine largest, and each is an
# all-or-nothing unit of work because the worker holds a whole file's output in memory and writes
# once at the end. Three hours on nine processes produced not one of them (C2-F-139).
#
# It serves ONE arm, `plus_ho_M20`, which is not in the settled 14-feature model and was unsettled
# across three fit seeds -- it kept its sign but reached p = 0.092 at its worst (C2-F-132).
# Everything this export exists to decide -- C2-F-130, whether dropping the whole structural family
# still wins at 70x the fit size, and whether 14 features is minimal or only minimal at 157
# crystals -- needs the four groups below and nothing else.
#
# So: export now, and let `submit_fom_holdout_finish.sh` run on its own clock. If the arm is wanted
# later, add `holdout` here and re-export; the precondition below refuses to start until its
# sidecars verify clean, because a missing sidecar joins as a NULL COLUMN and would fit that arm on
# a feature that is silently absent.
FEATURE_GROUPS="raw,structural,context,counts,campaign1_raw,probation,soft"

if [ -z "$BUNDLE" ]; then
    echo "FATAL: no bundle for array index '$SLURM_ARRAY_TASK_ID'" >&2
    exit 1
fi
# **Every sidecar this group list needs, COMPLETE, before nine tasks start.** A directory that
# exists is not a directory that is finished: the hold-out pass died with 117 of 126 files written
# and the nine survivors would have joined as null columns, which is a feature silently absent
# rather than a job that fails (C2-F-139). Task 0 alone runs the verifies -- they are metadata-only
# and take seconds, but nine copies would be nine times the metadata I/O for one answer.
for sidecar in merits structural merits_soft; do
    if [ ! -d "$POOL/$sidecar" ]; then
        echo "FATAL: $POOL/$sidecar is missing. This job assembles, it does not build." >&2
        exit 1
    fi
done
case "$FEATURE_GROUPS" in
  *holdout*)
    if [ ! -d "$POOL/holdout_merits" ]; then
        echo "FATAL: FEATURE_GROUPS asks for holdout and $POOL/holdout_merits does not exist." >&2
        echo "Run submit_fom_holdout_finish.sh, or drop 'holdout' from FEATURE_GROUPS." >&2
        exit 1
    fi
    if ! "$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --verify > /dev/null 2>&1; then
        echo "FATAL: the hold-out sidecars are INCOMPLETE. They would join as null columns and" >&2
        echo "plus_ho_M20 would be fitted on a feature that is silently absent. Run" >&2
        echo "  $PYTHON run_fom_holdout_merits.py --pool $POOL --verify" >&2
        echo "then submit_fom_holdout_finish.sh, or drop 'holdout' from FEATURE_GROUPS." >&2
        exit 1
    fi
    ;;
esac

# Validated against the real list before the expensive part, not by substring: a substring test
# passes `raw,structrual` and would burn the whole task before the driver noticed (C2-F-135).
if ! "$PYTHON" -c "
import sys
from mlindex.model_training import FomCombiner
known = set(FomCombiner.FEATURE_GROUPS)
asked = [g for g in sys.argv[1].split(',') if g]
unknown = [g for g in asked if g not in known]
if unknown or not asked:
    print('unknown feature group(s) %r; known: %s' % (unknown, sorted(known)), file=sys.stderr)
    raise SystemExit(1)
" "$FEATURE_GROUPS"; then
    echo "FATAL: FEATURE_GROUPS is '$FEATURE_GROUPS', which is not a valid group list." >&2
    exit 1
fi

set -e

echo "=== export $BUNDLE (array task $SLURM_ARRAY_TASK_ID) ==="
"$PYTHON" run_fom_combiner.py --stage export-fit \
    --fit-pool "$POOL" \
    --bundles "$BUNDLE" \
    --groups "$FEATURE_GROUPS" \
    --n-negatives 40 \
    --calibration-negatives 400 \
    --out-dir "$OUTDIR" \
    --suffix _fullscale

echo
echo "DONE: $BUNDLE"
echo "When ALL NINE tasks have finished -- check with"
echo "  sacct -j \$SLURM_ARRAY_JOB_ID --format=JobID,State,Elapsed,MaxRSS"
echo "  ls $OUTDIR/S12_combiner_fit_frame_fullscale_*.parquet | wc -l   # must be 9"
echo "then from the laptop:"
echo "  docs/sync_record.sh pull-artifacts 'S12_combiner_*fullscale*'"
echo "  run_fom_combiner.py --stage fit --suffix _fullscale \\"
echo "      --fit-frame 'docs/fom_campaign2/artifacts/S12_combiner_fit_frame_fullscale_*.parquet'"
