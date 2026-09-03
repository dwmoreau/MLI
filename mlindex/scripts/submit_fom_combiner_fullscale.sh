#!/bin/bash
# S12 -- the full-scale fit. Build the design matrix where the pool is; fit where the report is.
#
#   sbatch submit_fom_combiner_fullscale.sh
#   rsync the two *_frame.parquet back to the laptop (~1.5 GB), then there:
#       python mlindex/scripts/run_fom_combiner.py --stage fit \
#           --fit-frame docs/fom_campaign2/artifacts/S12_combiner_fit_frame_fullscale.parquet \
#           --suffix _fullscale
#       python mlindex/scripts/run_fom_combiner.py --stage reduce  --suffix _fullscale
#       python mlindex/scripts/run_fom_combiner.py --stage analyse --suffix _fullscale
#
# WHAT THIS DECIDES.
#
# S12 session 1 fitted on the Benchmark B **slice**: 157 `fom-train` crystals. The benchmark holds
# about 11 000. That difference is the leading explanation for the one result in the step that
# contradicts an earlier finding -- **C2-F-130**, where dropping the entire eighteen-column
# structural family is the BEST arm at +7.30 pp of operating point, against C2-F-040's −1.675 pp
# on Benchmark A. It is not a train/test shift: every structural column is bit-identical across the
# two pools. But several of them (`final_rank`, `n_entering`, `pool_size_full`, `log_volume`) are
# effectively per-pattern constants, and with 157 training crystals a tree can key on individual
# patterns rather than learn a rule. **Until this runs, the structural family is not cut.**
#
# It also re-decides every other arm at 70x the fit size, including the two that were null on the
# slice: `plus_dropped_merits` (+1.07 pp, p = 0.061), which is what licenses the cut from
# seventeen merits to seven, and `drop_mrev_family` (+0.06 pp, p = 1.00).
#
# WHY THE DESIGN MATRIX TRAVELS AND NOTHING ELSE DOES. The pool is 122 GB on $SCRATCH and the
# laptop has ~13 GB free, so the pool cannot come. The **report** pool -- the fully retained one
# whose ranks are exact for a learned score (C2-R-013) -- is 23 GB and already on the laptop, so it
# cannot go. The subsampled fit frame is ~0.5 GB and its calibration sibling ~1.0 GB. So: assemble
# and subsample here, fit and report there.
#
# COST. The sidecar passes dominate, at 880 704 233 candidates:
#   structural  255 us/candidate -> ~62 core-hours -> ~1 h on 64 processes   REQUIRED
#   merits_soft ~136 us          -> ~33 core-hours -> ~35 min                only for plus_X_N_soft
#   holdout     ~226 us          -> ~55 core-hours -> ~55 min                only for plus_ho_M20
#   export      assembly + subsample                -> ~30 min
# Walltime 6 h against ~3.5 h expected. `merits/` already exists from submit_fom_zoo_merits.sh and
# MUST NOT be rebuilt -- it cost 33 core-hours and is verified.
#
# IF YOU WANT ONLY THE STRUCTURAL ANSWER, set GROUPS below to the core four and skip the two
# optional sidecars: that is a ~1.5 h job. The arms that need the missing groups then skip
# themselves and record the reason in the fit table.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core. Read SLURM_CPUS_ON_NODE.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_comb_full
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH -o fom_comb_full_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

POOL="$SCRATCH/fom_campaign2/pool"
# $SCRATCH, NOT the repo's docs/ tree, and both halves of that matter.
#
# `docs/sync_record.sh pull-artifacts` reads $SCRATCH/$CAMPAIGN/artifacts and nothing else, so
# a job that writes anywhere else is a job whose output cannot be fetched by the documented
# route -- which is how the first run of this script produced 1.5 GB of frames that
# `pull-artifacts` silently did not see. And `sync_record.sh` says in as many words that $CFS
# is not where a batch job should write at volume.
#
# PURGE RISK comes with it, and it is C2-R-014: nothing on $SCRATCH is backed up. Pull these
# the day the job finishes and run `snapshot` afterwards.
OUTDIR="$SCRATCH/fom_campaign2/artifacts"
mkdir -p "$OUTDIR"
PROCESSES=64

# Every feature group any arm needs. Drop to 'raw,structural,context,counts' to answer the
# structural question alone and skip the two optional sidecar passes below.
#
# **NOT named GROUPS.** `GROUPS` is a bash BUILT-IN array holding the current user's Unix group
# ids, and bash silently IGNORES assignments to it -- so `GROUPS="raw,..."` had no effect and
# `"$GROUPS"` expanded to the primary gid. The first run of this script therefore spent an hour
# building the structural sidecars and then died with `unknown feature group(s) ['97050']`, and the
# two `case` tests below matched nothing, so the soft and hold-out passes were skipped without a
# word. Nothing was lost, because the expensive pass had already written its output -- but the
# lesson is to keep shell variables out of bash's reserved namespace.
FEATURE_GROUPS="raw,structural,context,counts,campaign1_raw,probation,soft,holdout"

if [ ! -d "$POOL" ]; then
    echo "FATAL: Benchmark B is not at $POOL" >&2
    exit 1
fi
if [ ! -d "$POOL/merits" ]; then
    echo "FATAL: the merit sidecars are missing from $POOL/merits." >&2
    echo "They cost 33 core-hours and are written by submit_fom_zoo_merits.sh. Do not rebuild" >&2
    echo "them casually; check whether \$SCRATCH was purged (C2-R-014)." >&2
    exit 1
fi

# Checked BEFORE the hour-long pass, not after: the driver rejects an unknown group, but it is the
# last step here and a typo would cost the whole job's compute before saying so.
case "$FEATURE_GROUPS" in
    *raw*structural*) ;;
    *) echo "FATAL: FEATURE_GROUPS looks wrong: '$FEATURE_GROUPS'" >&2
       echo "Expected a comma-separated list of feature groups starting raw,structural,..." >&2
       exit 1 ;;
esac

set -e

echo "=== structural features: the columns the design matrix needs and the pool does not store ==="
"$PYTHON" run_fom_structural_features.py --pool "$POOL" --processes "$PROCESSES"
"$PYTHON" run_fom_structural_features.py --pool "$POOL" --verify \
    --sweep-dir "$POOL/extinction_sweep"

case "$FEATURE_GROUPS" in
  *soft*)
    echo "=== soft counting merits (C2-F-102), for the plus_X_N_soft arm ==="
    "$PYTHON" run_fom_floor_merits.py --pool "$POOL" --soft \
        --out-dir "$POOL/merits_soft" --processes "$PROCESSES"
    ;;
esac
case "$FEATURE_GROUPS" in
  *holdout*)
    echo "=== hold-out merits (S10), for the plus_ho_M20 arm ==="
    "$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --processes "$PROCESSES"
    ;;
esac

echo "=== export the fit and calibration frames ==="
"$PYTHON" run_fom_combiner.py --stage export-fit \
    --fit-pool "$POOL" \
    --groups "$FEATURE_GROUPS" \
    --n-negatives 40 \
    --calibration-negatives 400 \
    --out-dir "$OUTDIR" \
    --suffix _fullscale

echo
echo "DONE. Now, from the laptop:"
echo "  docs/sync_record.sh pull-artifacts 'S12_combiner_*fullscale*'"
echo "     -- both frames must arrive: the calibrator has to be fitted on rows the model was"
echo "        not, so --stage fit refuses a fit frame whose _cal_frame sibling is missing"
echo "  run_fom_combiner.py --stage fit     --fit-frame <the fit frame> --suffix _fullscale"
echo "  run_fom_combiner.py --stage reduce  --suffix _fullscale"
echo "  run_fom_combiner.py --stage analyse --suffix _fullscale"
echo
echo "40 negatives a pattern here against the slice run's 400, deliberately: 400 was chosen to"
echo "recover campaign 1's ROW count from 157 crystals, and at ~11 000 crystals 40 gives more"
echo "rows than that while keeping the frame small enough to carry."
