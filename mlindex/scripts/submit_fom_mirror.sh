#!/bin/bash
# S02 -- the synthetic mirror condition grid.
#
# 21 bundles, one per job-array task: q2 error multiplier {0, 0.25, 0.5, 0.75, 1, 1.5, 2} x
# contaminants {0, 1, 2}. Each task runs the full 14-Bravais-lattice pipeline per entry and pools the
# candidates across lattices, which is what the paper's operating point is defined on.
#
#   sbatch submit_fom_mirror.sh
#
# ONE BUNDLE PER JOB, deliberately. At 500 entries per lattice a bundle is ~5 955 entries and, at the
# 48.3 s/entry/pool the calibration measured for 32x4, ~2.5 h of node time. Twenty-one of those in a
# single job is ~52 h; as an array they run concurrently, each task has 2x headroom inside the 5 h
# walltime, and one bundle dying costs only that bundle. (An earlier version of this script asked for
# 12 h to run ten bundles sequentially, which would have been killed partway and silently truncated
# the grid.)
#
# Parallelism within a task is two-level, and the shape matters. Inside a pool only the optimisation
# iterations distribute -- candidate generation runs on the manager -- so one 128-wide pool spends
# most of its processes waiting on a serial section. Measured on a whole node at 224 entries:
#
#     16 pools x 8 procs   34.9 s/entry/pool   -> 3.61 h/bundle
#     32 pools x 4 procs   48.3 s/entry/pool   -> 2.49 h/bundle   <- 1.45x faster, used here
#
# Fitting those two points to S + P/p gives ~21.5 s serial and ~107 s parallelisable per entry, which
# predicts 64x2 at ~1.9 h and 128x1 at ~1.7 h -- i.e. still improving. Worth one more calibration
# before a rerun, with an eye on memory: only managers load models (~3 GB each), so 32 managers is
# ~96 GB of the node's 512 GB, but 128 would be ~384 GB and is where this stops being free.
#
# Entries are striped across pools so every pool gets the same lattice mix and therefore roughly the
# same total cost.
#
# Deliberately NOT enabling --conventional-cell, for two measured reasons. It changes nothing that
# matters -- on 118 calibration entries the ceiling was identical (0.890) and the operating point
# moved 0.525 -> 0.534, inside noise, with zero entries losing their correct cell. And it *crashes*:
# 10 of 16 pools died on `cctbx Internal Error: rot_mx.cpp(36): Unsuitable value for rational
# rotation matrix` from metric_subgroups. That is a live bug in run.py's own default output path
# (F-040), not something this flag introduced.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_mirror
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 5:00:00
#SBATCH --array=0-20

# Absolute interpreter path rather than `module load conda; conda activate`, which is what
# PROTOCOL section 6 documents and what the calibration run was verified with. With 21 array tasks a
# shell-initialisation failure would take all of them down at once, after queueing.
PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
cd /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/scripts || exit 1

NPOOLS=32         # independent managers, run concurrently. 32x4 measured 1.45x faster than 16x8
POOLSIZE=4        # processes per pool: 1 manager + 3 workers. NPOOLS * POOLSIZE = 128 = core count

# THE GRID SPANS THE REGIME; IT IS NOT TUNED TO A TARGET. S02's original gate asked the mirror to
# reproduce the sealed benchmark's 75.1% / 89.0%, and that gate was retired on 2026-08-12 (STATUS 6)
# because meeting it meant choosing conditions -- and adding mechanisms -- to match the answers of
# the benchmark S16 is supposed to test blind. Conditions are justified physically instead:
#
#   error multiplier 1 is the characterised sigma(q2) for this instrument population, so it is the
#   nominal by measurement, not by fit. PLAN 6.3 labels (1, 0) as C1 nominal, (0, 0) as C0 ideal,
#   (2, 0) as C2 noisy and (1, 2) as C3 contaminated -- all designated before any comparison existed.
#
# The error axis runs below 1 as well as above so the grid brackets the nominal on both sides and the
# development set spans easy to hard; 0.25-0.75 was originally added to chase the CNRS numbers and is
# kept because a spanning grid is useful in its own right. Error 0 with 0 contaminants is the control
# -- a near-100% ceiling there checks that nothing upstream is broken, and if it fails no other
# number in S02 is trustworthy.
#
# 7 error levels x 3 contaminant counts = 21 tasks.
ERRORS=(0 0.25 0.5 0.75 1 1.5 2)
CONTS=(0 1 2)
NPERBL=500

NCONT=${#CONTS[@]}
ERROR=${ERRORS[$((SLURM_ARRAY_TASK_ID / NCONT))]}
CONT=${CONTS[$((SLURM_ARRAY_TASK_ID % NCONT))]}
echo "array task $SLURM_ARRAY_TASK_ID: error $ERROR, contaminants $CONT, $NPERBL entries/lattice"

$PYTHON run_fom_mirror.py --n-pools $NPOOLS --pool-size $POOLSIZE \
    --error-multiplier "$ERROR" --n-contaminants "$CONT" \
    --n-entries-per-bl "$NPERBL" --shard 0 --n-shards 1


# ------------------------------------------------------------------------------------------------
# Additional realistic conditions, for a wider development set. Two mechanisms are implemented and
# not yet run, both justified physically rather than as gap-closers:
#
#   --n-dropout N        interior peak dropout: delete N peaks from within the low-q2 window and
#                        backfill from higher q2, as undetected weak reflections do. Justified by
#                        detection limits. Headroom exists: 86% of cP and 93-99.9% of every other
#                        lattice's validation entries carry >=28 peaks.
#   --contaminant-bias B  weight contaminant positions towards low q2, where a second phase's strong
#                        reflections actually sit. B=2 is uniform in q; the fraction landing within
#                        the first five real peaks goes 35% (B=1) -> 61% (B=2) -> 72% (B=3).
#
# Both widen the condition space rather than targeting a number, and both go in the bundle tag so
# they cannot collide with the 21 bundles already on disk. C5 (truncation to 14 peaks) is impossible:
# the ONNX generators reject anything under the nominal peak count (F-044).
#
# Old note, kept for the reasoning: extending the error axis further would not have helped. The
# achievable (ceiling, operating point) locus passes beneath the sealed benchmark's point at every
# error level, so the mismatch was never on this axis -- which is part of why the gate was replaced
# rather than chased (F-042).
# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# Scoring, CNRS weighting, and the frozen manifest -- run once, after the whole array has finished.
# Report every bundle first, then pass --select-bundle to freeze S02_mirror_manifest.parquet against
# the chosen one. Not part of the array: it consumes all bundles at once.
# ------------------------------------------------------------------------------------------------
#/global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python run_fom_mirror_analysis.py
#/global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python run_fom_mirror_analysis.py \
#    --select-bundle error1_cont1
