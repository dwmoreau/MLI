#!/bin/bash
# S04 -- generate the frozen FOM benchmark candidate pool.
#
# Seven condition bundles, one per job-array task, over the 5 955 entries of S02's frozen manifest.
# Each task runs the full 14-Bravais-lattice pipeline per entry and dumps every surviving
# candidate, not the twenty per lattice ranking keeps, with the columns needed to recompute any
# figure of merit offline. This is Benchmark A: generate once, evaluate forever.
#
# ================================================================================================
# RUNBOOK. Read it. docs/ is git-ignored, so `git pull` does NOT deliver the S04 handoff, STATUS.md,
# or the frozen split manifest -- this header is the only copy of these instructions that reaches
# this machine.
#
#   1. ON THE LAPTOP, push the record and the split manifest across, and note the checksum:
#
#        docs/sync_record.sh push
#
#      It prints the sha256 of docs/fom/artifacts/S02_mirror_manifest.parquet. That file supplies
#      the fom-train/dev/test split, which PROTOCOL section 3 rule 5 forbids re-deriving; without
#      it run_fom_dump.py exits rather than inventing one. Confirm it arrived intact:
#
#        sha256sum /global/cfs/cdirs/m4064/dwmoreau/MLI/docs/fom/artifacts/S02_mirror_manifest.parquet
#
#   2. Pull the code and check you have what the laptop has:
#
#        cd /global/cfs/cdirs/m4064/dwmoreau/MLI && git checkout fom && git pull
#        git log --oneline -1
#
#      NERSC cannot push to origin (F-047), so never commit here expecting it to travel.
#
#   3. Time it before spending the array:
#
#        sbatch submit_fom_dump_calibration.sh
#
#      It compares 32x4 against 16x8 on a small entry count in the debug queue. If s/entry/pool
#      lands far from the ~48 s the mirror measured, adjust NPOOLS/POOLSIZE or the walltime here
#      before submitting.
#
#   4. Run the grid:
#
#        sbatch submit_fom_dump.sh
#
#      Tasks are independent. A task that dies can be requeued on its own -- run_fom_dump.py skips
#      pools whose output is already written and readable, so a requeue resumes rather than redoes.
#
#   5. Check completeness, then consolidate. The gate recomputes M20, Minfo and n_indexed from the
#      dumped columns alone and fails if any column needed to reconstruct a score is missing:
#
#        PY=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#        for d in ../characterization/fom/benchmark/*/; do
#            $PY run_fom_dump_gate.py --dump-dir "$d" --artifact-dir ../../docs/fom/artifacts \
#                --tag "S04_roundtrip_$(basename "$d")" || echo "GATE FAILED: $d"
#        done
#        $PY run_fom_dump_consolidate.py --dump-root ../characterization/fom/benchmark \
#            --out-dir ../data/fom_benchmark --artifact-dir ../../docs/fom/artifacts
#
#      Consolidation exits non-zero if the acceptance gate fails, and writes the row-count table,
#      the reachability ceiling and the gate verdict to docs/fom/artifacts/S04_row_counts.md.
#
#   6. Record findings in docs/fom/STATUS_nersc_inbox.md and NOTHING in STATUS.md. The laptop copy
#      is authoritative and `sync_record.sh push` overwrites this one; the inbox is excluded from
#      that push so it survives to be merged. Editing STATUS.md here is exactly how F-045 was lost.
#      Then, on the laptop: docs/sync_record.sh pull-inbox
# ================================================================================================
#
# SIZING. A bundle is 5 955 entries. At the 48.3 s/entry/pool the S02 calibration measured for
# 32x4, that is 5955/32 * 48.3 s ~ 2.5 h, against a 5 h walltime -- 2x headroom, which the dump
# needs more of than the mirror did because it writes ~566 rows per entry rather than one. Seven
# tasks run concurrently, so the grid finishes in one bundle's wall-time and a failure costs one
# bundle. Storage is ~3.5 GB for the whole grid (F-049: ~148 B/row).
#
# NOT passing --conventional-cell, matching the S02 grid. It changed nothing measurable there
# (ceiling identical, operating point inside noise) and it crashed 10 of 16 pools on a cctbx
# metric_subgroups error. The guard for that is now on main, but the mirror this pool sits
# alongside was generated without it, so it stays off for comparability. See F-039, F-040, Q19.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_dump
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 5:00:00
#SBATCH --array=0-6

# Absolute interpreter rather than `module load conda; conda activate`, per PROTOCOL section 6.
# With seven array tasks a shell-initialisation failure would take all of them down at once,
# after queueing.
PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

NPOOLS=32         # independent managers, run concurrently. 32x4 measured 1.45x faster than 16x8
POOLSIZE=4        # processes per pool: 1 manager + 3 workers. NPOOLS * POOLSIZE = 128 = core count
NPERBL=500
SEED=12345
MANIFEST="$REPO/docs/fom/artifacts/S02_mirror_manifest.parquet"
OUTROOT="$REPO/mlindex/characterization/fom/benchmark"

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first -- step 1 of the runbook above." >&2
    exit 1
fi

# The condition bundles, from PLAN section 6.3 as amended on 2026-08-16 (STATUS section 6).
#
#   C0 ideal         the control. A near-100% ceiling here checks nothing upstream is broken; if
#                    it fails, no other number in S04 is trustworthy.
#   C1 nominal       error multiplier 1 IS the characterised sigma(q2) for this instrument
#                    population, so the nominal is a measurement, not a fit. The frozen split
#                    manifest is pinned to this bundle.
#   C2 noisy         twice the characterised error.
#   C3 contaminated  two contaminant lines placed independently at random.
#   C4 sparse        6 holes punched in the nominal 20-peak window, backfilled from higher q2 as
#                    undetected weak reflections do. Median q2_20 stretches to 1.27x nominal.
#   C5 aggressive    10 holes, which is ErrorAdder.MAX_INTERIOR_DROPOUT. Median q2_20 1.46x, so
#                    this is where the handoff's 1.5x target landed. The ceiling exists because
#                    the median entry carries ~60 peaks: swept further, the mechanism empties the
#                    nominal window entirely and becomes a wholesale translation to high angle
#                    rather than interior dropout. Both values are measured in
#                    docs/fom/artifacts/S04_dropout_calibration.md.
#   C6 second phase  3 lines from a real second cell drawn at random from the database, weighted
#                    towards low angle. Unlike C3 these are CORRELATED -- consistent with an actual
#                    lattice -- which is what makes them hard to reject and C3 optimistic.
#
# C5 is NOT the truncation-to-14 condition the handoff originally specified. That remains blocked:
# the ONNX generators reject a peak vector shorter than 20 at the model input, before any figure of
# merit is reached (F-044, Q24). C5 keeps 20 peaks and attacks the low-angle window instead.
BUNDLES=(
    "0 0 0 0 ideal"
    "1 0 0 0 nominal"
    "2 0 0 0 noisy"
    "1 2 0 0 contaminated"
    "1 1 6 0 sparse"
    "1 1 10 0 aggressive"
    "1 0 0 3 second_phase"
)

read -r ERROR CONT DROP PHASE NAME <<< "${BUNDLES[$SLURM_ARRAY_TASK_ID]}"

# The tag run_fom_dump.py will derive, so each bundle gets its own directory. One manifest.json is
# written per --out-dir, so bundles sharing a directory would overwrite each other's.
TAG="error${ERROR}_cont${CONT}"
[ "$DROP" -gt 0 ] && TAG="${TAG}_drop${DROP}"
[ "$PHASE" -gt 0 ] && TAG="${TAG}_phase${PHASE}"

echo "array task $SLURM_ARRAY_TASK_ID: $NAME -- error $ERROR, contaminants $CONT, dropout $DROP, "\
     "second-phase lines $PHASE -> $TAG"

$PYTHON run_fom_dump.py \
    --error-multiplier "$ERROR" \
    --n-contaminants "$CONT" \
    --n-dropout "$DROP" \
    --second-phase-lines "$PHASE" \
    --n-entries-per-bl "$NPERBL" \
    --n-pools "$NPOOLS" --pool-size "$POOLSIZE" \
    --shard 0 --n-shards 1 \
    --seed "$SEED" \
    --split-manifest "$MANIFEST" \
    --out-dir "$OUTROOT/$TAG"
