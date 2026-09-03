#!/bin/bash
# S12 -- turn the hard-stratum generation output into a scoreable pool. Run AFTER the array.
#
#   sbatch submit_fom_hard_retained.sh                    # the array: 3 tasks, ~30 min
#   sbatch submit_fom_hard_postprocess.sh                 # this: 1 task, ~40 min
#
# Or chain them and forget the ordering:
#   JOB=$(sbatch --parsable submit_fom_hard_retained.sh)
#   sbatch --dependency=afterok:$JOB submit_fom_hard_postprocess.sh
#
# WHY THIS IS A SCRIPT AND NOT A COMMENT. It used to be a comment -- a block of six commands in the
# footer of `submit_fom_hard_retained.sh` -- and the first run of that job therefore produced
# candidates and no pool, because comments do not execute. The consolidation genuinely cannot live
# inside the array: it reads all three condition tags at once and an array task only knows its own.
# So it is a second job, and being a second job it should be a file rather than a docstring.
#
# WHAT A POOL NEEDS BEFORE ANYTHING CAN SCORE IT. The array writes candidates. Four sidecar passes
# make them usable, and `FomBenchmark.bundle_frames(require_merits=True)` RAISES rather than
# quietly leaving columns null if one is missing -- which is the right behaviour, since a null
# merit sorts last and reads as the worst score in the zoo rather than as an error.
#
#   merits/            the seven ranking merits              ~15 min
#   merits_soft/       the posterior-built counting merits   ~10 min
#   structural/        S12's design-matrix columns           ~15 min
#   holdout_merits/    the S10 hold-out family               ~15 min
#
# COST. The hard pool is ~1 080 cells of low-symmetry crystals, so its candidate count is larger per
# cell than the benchmark median but the pool is small: expect ~40 min in total on 64 processes.
# Walltime 2 h, which is slack rather than expectation.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core. Read SLURM_CPUS_ON_NODE.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_hard_post
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 2:00:00
#SBATCH -o fom_hard_post_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

OUTROOT="$SCRATCH/fom_campaign2/hard_retained"
POOL="$SCRATCH/fom_campaign2/hard_pool"
PROCESSES=64

if [ ! -d "$OUTROOT" ]; then
    echo "FATAL: no generation output at $OUTROOT" >&2
    echo "Run submit_fom_hard_retained.sh first; this consolidates what it wrote." >&2
    exit 1
fi
TAGS=$(find "$OUTROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$TAGS" -ne 3 ]; then
    echo "FATAL: expected 3 condition directories under $OUTROOT, found $TAGS." >&2
    echo "The array has 3 tasks; consolidating a partial set would silently build a pool over" >&2
    echo "fewer conditions than every downstream number assumes." >&2
    find "$OUTROOT" -mindepth 1 -maxdepth 1 -type d >&2
    exit 1
fi

set -e

echo "=== consolidate: $OUTROOT -> $POOL ==="
"$PYTHON" run_fom_dump_consolidate.py --dump-root "$OUTROOT" --out-dir "$POOL"

echo "=== the seven ranking merits ==="
"$PYTHON" run_fom_floor_merits.py --pool "$POOL" --processes "$PROCESSES"

echo "=== the posterior-built counting merits (C2-F-102) ==="
"$PYTHON" run_fom_floor_merits.py --pool "$POOL" --soft \
    --out-dir "$POOL/merits_soft" --processes "$PROCESSES"

echo "=== S12's structural columns ==="
"$PYTHON" run_fom_structural_features.py --pool "$POOL" --processes "$PROCESSES"

echo "=== the hold-out family (S10) ==="
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --processes "$PROCESSES"

echo "=== verify: every sidecar present, complete and populated ==="
# Exit code 0 is not evidence on its own -- C2-F-071 is an entire Bravais lattice lost from
# Benchmark B while all 24 generation tasks exited 0.
"$PYTHON" run_fom_structural_features.py --pool "$POOL" --verify
"$PYTHON" run_fom_floor_merits.py --pool "$POOL" --verify

echo
echo "DONE. Pool at $POOL"
du -sh "$POOL"
echo
echo "Now, from the laptop -- an explicit rsync, because sync_record.sh moves docs/ and not data:"
echo "  rsync -avz --progress \\"
echo "      <nersc>:$POOL/ mlindex/data/fom_hard_c2_pool/"
echo
echo "Then S12 can report a hard stratum for the first time: 360 crystals with exact ranks,"
echo "against the 20 cells over 20 crystals every hard number currently rests on (C2-R-019)."
