#!/bin/bash
# S12 -- consolidate the four contaminated / sparse bundles and build their sidecars.
#
#   sbatch --dependency=afterok:<generation jobid> submit_fom_contaminant_postprocess.sh
#
# Written as a script and not as a footer comment, because the first version of the hard-stratum
# job put exactly these commands in a comment block and comments do not execute (C2-F-135).
#
# WALLTIME IS SIZED FROM THE HOLD-OUT PASS, WHICH IS THE EXPENSIVE ONE AND WAS BADLY
# UNDER-ESTIMATED TWICE. It runs at >506 microseconds a candidate on this hardware -- measured, not
# assumed (C2-F-139) -- and a FILE is an all-or-nothing unit of work, so the walltime has to cover
# the LARGEST file rather than the total over processes. This pool is 530 crystals over 4 bundles
# against the 43 M-candidate pool's 530 over 3, so its largest file is comparable to that pool's
# and well inside 6 h. If a task dies, `--verify` says whether anything is truncated before you
# resume; nothing here may be resumed on the strength of an exit code alone.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_cont_post
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH -o fom_cont_post_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

OUTROOT="$SCRATCH/fom_campaign2/contaminant_retained"
POOL="$SCRATCH/fom_campaign2/contaminant_pool"
PROCESSES=64

if [ ! -d "$OUTROOT" ]; then
    echo "FATAL: no generation output at $OUTROOT" >&2
    echo "Run submit_fom_contaminant_retained.sh first; this consolidates what it wrote." >&2
    exit 1
fi
TAGS=$(find "$OUTROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$TAGS" -ne 4 ]; then
    echo "FATAL: expected 4 condition directories under $OUTROOT, found $TAGS." >&2
    echo "The array has 4 tasks; consolidating a partial set would silently build a pool over" >&2
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

# 16, not 64. Sixty-four workers each holding a 2 M-row chunk with six peak budgets is what
# OOM-killed the full-scale hold-out pass, and above one process per file nothing extra is used
# (C2-F-139). This pool has 56 files, so 16 is comfortably parallel and comfortably inside memory.
echo "=== the hold-out family (S10) ==="
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --processes 16 --chunk-rows 1000000

echo "=== verify: every sidecar present, complete and populated ==="
# Exit code 0 is not evidence. C2-F-071 lost an entire Bravais lattice while all 24 generation
# tasks exited 0; C2-F-135 is two cluster jobs that failed while exiting 0; and C2-F-139 is a pass
# that lost nine files to an OOM and would have been resumed over silently without this.
"$PYTHON" run_fom_structural_features.py --pool "$POOL" --verify
"$PYTHON" run_fom_floor_merits.py --pool "$POOL" --verify
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --verify

echo
echo "DONE. Pool at $POOL"
du -sh "$POOL"
echo
echo "Now, from the laptop -- an explicit rsync, because sync_record.sh moves docs/ and not data:"
echo "  rsync -avz --progress \\"
echo "      <nersc>:$POOL/ mlindex/data/fom_contaminant_c2_pool/"
echo
echo "Then S12 can answer the question it currently cannot: what the learned score does on a"
echo "pattern carrying peaks no cell can index. The 530 crystals are the SAME ones the three clean"
echo "bundles use, so every comparison is paired over crystals."
