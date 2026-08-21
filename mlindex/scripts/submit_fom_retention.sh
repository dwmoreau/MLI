#!/bin/bash
# S14 -- measure what the two destructive cuts delete (Q31, F-065).
#
# Eight array tasks: the four hard condition bundles x two prune thresholds, over the 243
# hard-stratum entries of docs/fom/artifacts/S14_hard_entries.csv. Each task runs the full
# 14-Bravais-lattice pipeline and dumps every candidate ENTERING deduplication, not just
# the survivors Benchmark A kept.
#
# WHY. F-136 measured a perfect re-scorer at 0.6961 against a 0.6468 operating point, so
# 4.93 pp is the entire remaining prize for every figure of merit this project will build,
# while 30.4% of entries -- 90% on the hard stratum -- have no correct candidate anywhere.
# METRICS.md section 3 assigns that bucket to S14. Two cuts could be responsible and
# neither has ever been measured:
#
#   prune_below_m20   discards everything below M20 5.0 -- 94.2% of generated candidates
#                     at C1 nominal (F-049), and 99.8% on a hard-stratum sample. Q31, R1.
#   deduplication     keeps the highest-M20 member of each xnn neighbourhood. Benchmark A
#                     stored only survivors, so the discarded rows exist nowhere. F-065, R2.
#
# ================================================================================================
# RUNBOOK. Read it. docs/ is git-ignored, so `git pull` does NOT deliver the S14 handoff,
# STATUS.md, or the entry list -- this header is the only copy of these instructions that
# reaches this machine.
#
#   1. ON THE LAPTOP, push the record and the entry list across:
#
#        docs/sync_record.sh push
#
#      That carries docs/fom/artifacts/S14_hard_entries.csv, which this script requires and
#      which must NOT be re-derived here: it is a filter on S02's frozen split (PROTOCOL
#      section 3 rule 5), and it deliberately excludes the 60 fom-test entries, which stay
#      sealed until S15.
#
#   2. Pull the code:
#
#        cd /global/cfs/cdirs/m4064/dwmoreau/MLI && git checkout fom && git pull
#        git log --oneline -1
#
#      NERSC cannot push to origin (F-047), so never commit here expecting it to travel.
#
#   3. Run:
#
#        sbatch submit_fom_retention.sh
#
#      Tasks are independent and run_fom_dump.py resumes rather than redoes: a pool whose
#      output is already written and readable is skipped, and the pre-deduplication shard
#      counts towards "written" when --dump-predownsample was asked for.
#
#   4. Verify BEFORE reading any number. The gate stage is cheap and it is the difference
#      between a physics result and a plumbing artefact:
#
#        $PYTHON run_fom_retention_report.py --stage gate \
#            --t0-root ../characterization/fom/retention/t0 \
#            --t5-root ../characterization/fom/retention/t5 \
#            --benchmark-dir ../data/fom_benchmark \
#            --artifact-dir ../../docs/fom/artifacts
#
#      It checks that the peak lists reproduce Benchmark A, that n_entering is consistent
#      with the dumped row counts, and that each arm ran at the threshold it claims. ON THIS
#      MACHINE THE q2 DIGESTS SHOULD MATCH EXACTLY -- the pool was generated here. A digest
#      mismatch on the laptop is the known arm64/x86 one-ULP contaminant difference (R9,
#      R13) and is expected; a mismatch HERE is not, and means the patterns differ.
#
#   5. Then the measurement. Labelling is ~9 ms/candidate and is the entire cost, so give
#      it the node:
#
#        $PYTHON run_fom_retention_report.py --stage analyse \
#            --t0-root ../characterization/fom/retention/t0 \
#            --t5-root ../characterization/fom/retention/t5 \
#            --artifact-dir ../../docs/fom/artifacts --n-processes 128
#
#   6. Record findings in docs/fom/STATUS_nersc_inbox.md and NOTHING in STATUS.md. The
#      laptop copy is authoritative and `sync_record.sh push` overwrites this one; the inbox
#      is excluded from that push so it survives. Editing STATUS.md here is exactly how
#      F-045 was lost. Then, on the laptop: docs/sync_record.sh pull-inbox
# ================================================================================================
#
# SIZING. Measured on the laptop over three hard-stratum entries at 1 pool x 4 processes:
# 22.0 s/entry at threshold 5 and 29.1 s/entry at threshold 0, so the threshold-0 arm costs
# only ~1.3x -- not the 10-30x its 400x larger candidate population suggests, because the
# expensive stages are vectorised. 243 entries over 32 pools is ~8 entries per pool, so a
# task is ~4 min of work plus the ~111 s per-pool model loading. Hard-stratum entries are
# the slow ones (aP carries ~6 000 candidates against cubic's 100), so 1 h is generous
# slack over an estimate of ~20 min, not a measurement.
#
# STORAGE. 59 B/row measured, ~58 500 pre-deduplication rows per entry at threshold 0. So
# ~3.4 MB/entry, ~830 MB per threshold-0 bundle, ~3.4 GB for the four. The threshold-5 arm
# is three orders smaller.
#
# NOT passing --conventional-cell, matching the S02 grid and the S04 dump. See F-039, F-040, Q19.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_retention
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 1:00:00
#SBATCH --array=0-7

# Absolute interpreter rather than `module load conda; conda activate`, per PROTOCOL section 6.
PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

NPOOLS=32         # 32 x 4 = 128 = the node's PHYSICAL core count; SLURM_CPUS_ON_NODE
POOLSIZE=4        # reports 256, which counts both threads per core.
NPERBL=500        # must match S04, so the sampling reproduces the frozen set before the
SEED=12345        # entry list filters it
ENTRY_IDS="$REPO/docs/fom/artifacts/S14_hard_entries.csv"
OUTROOT="$REPO/mlindex/characterization/fom/retention"

if [ ! -f "$ENTRY_IDS" ]; then
    echo "FATAL: hard-stratum entry list missing at $ENTRY_IDS" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first -- step 1 of the runbook above." >&2
    exit 1
fi

# The four hard condition bundles: C2-C5, the ones the hard stratum is defined over
# (STATUS section 6, 2026-08-17). C1 and C6 are deliberately absent -- C6 postdates the
# stratum definition and is reported as its own stratum.
BUNDLES=(
    "2 0 0  noisy"
    "1 2 0  contaminated"
    "1 1 6  sparse"
    "1 1 10 aggressive"
)
THRESHOLDS=(5.0 0.0)
THRESHOLD_TAGS=(t5 t0)

BUNDLE_INDEX=$(( SLURM_ARRAY_TASK_ID % 4 ))
THRESHOLD_INDEX=$(( SLURM_ARRAY_TASK_ID / 4 ))
read -r ERROR CONT DROP NAME <<< "${BUNDLES[$BUNDLE_INDEX]}"
THRESHOLD="${THRESHOLDS[$THRESHOLD_INDEX]}"
THRESHOLD_TAG="${THRESHOLD_TAGS[$THRESHOLD_INDEX]}"

TAG="error${ERROR}_cont${CONT}"
if [ "$DROP" -gt 0 ]; then
    TAG="${TAG}_drop${DROP}"
fi
OUTDIR="$OUTROOT/$THRESHOLD_TAG/$TAG"

echo "task $SLURM_ARRAY_TASK_ID: bundle $TAG ($NAME), prune threshold $THRESHOLD -> $OUTDIR"

$PYTHON run_fom_dump.py \
    --error-multiplier "$ERROR" \
    --n-contaminants "$CONT" \
    --n-dropout "$DROP" \
    --n-entries-per-bl "$NPERBL" \
    --seed "$SEED" \
    --entry-ids-file "$ENTRY_IDS" \
    --prune-threshold "$THRESHOLD" \
    --dump-predownsample \
    --n-pools "$NPOOLS" \
    --pool-size "$POOLSIZE" \
    --shard 0 --n-shards 1 \
    --out-dir "$OUTDIR"
