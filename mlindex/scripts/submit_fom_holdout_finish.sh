#!/bin/bash
# S12 -- finish the nine hold-out sidecars, as an array by lattice.
#
#   sbatch submit_fom_holdout_finish.sh
#
# THIS IS THE SECOND ATTEMPT AND THE FIRST ONE'S FAILURE IS THE DESIGN INPUT.
#
# `submit_fom_combiner_fullscale.sh` lost nine files to a walltime plus two OOM kills. `--verify`
# then said exactly what was lost, and said the useful thing: 738 754 260 of 880 704 233 candidates
# carry hold-out merits, nine problems, ALL of them "NO SIDECAR" -- nothing truncated, no
# wholly-null column. The missing rows come to 141 949 973, exactly the difference. So a resume is
# safe, and it is safe because it was checked (C2-F-139).
#
# The first finish attempt then ran nine processes for three hours and produced NOT ONE FILE. Two
# things explain that, and only the second is fixable here:
#
#   1. **A file is an all-or-nothing unit of work.** `score_file` accumulates every chunk in `out`
#      and writes once at the end, so a killed worker loses the whole file however far it got.
#      `--chunk-rows` bounds the READ, not the output. Nothing partial is ever written -- which is
#      why `--verify` found no truncation, and also why three hours bought nothing.
#   2. **The per-candidate rate here is at least twice the laptop's.** Measured on this laptop with
#      `--sample-row-groups 20`: mP 252 us/candidate, aP 115. The largest missing file is 21 358 889
#      candidates, so 1.5 h at the laptop's mP rate -- and it did not finish in 3 h, which puts the
#      per-core rate above 506 us/candidate. At 1000 us that file alone is 5.9 h.
#
# HENCE: an array by lattice, `--processes` matched to each lattice's file count, and 8 hours.
# One task per lattice means a task that dies loses one lattice rather than all nine files, and
# each task's wall clock is its own LARGEST file, not the sum -- there is no parallelism inside a
# file, so more processes than files buys nothing.
#
#   task 0  oP  5 files, largest 16 262 997   ~4.5 h at 1000 us
#   task 1  aP  2 files, largest 21 358 889   ~5.9 h at 1000 us   <- the binding one
#   task 2  tP  2 files, largest  9 691 182   ~2.7 h at 1000 us
#
# **THE EXPORT DOES NOT WAIT FOR THIS.** `holdout` serves one arm, `plus_ho_M20`, which is not in
# the settled 14-feature model and was unsettled across three fit seeds (C2-F-132). Run
# `submit_fom_combiner_export.sh` now; fold this in later with a re-export if the arm is wanted.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_ho_finish
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 8:00:00
#SBATCH --array=0-2
#SBATCH -o fom_ho_finish_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

POOL="$SCRATCH/fom_campaign2/pool"

# One lattice per task, with as many processes as that lattice has missing files. Files already
# present are skipped, so a lattice that is finished costs seconds.
LATTICES=(oP aP tP)
PROCS=(5 2 2)
LATTICE="${LATTICES[$SLURM_ARRAY_TASK_ID]}"
NPROC="${PROCS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$LATTICE" ]; then
    echo "FATAL: no lattice for array index '$SLURM_ARRAY_TASK_ID'" >&2
    exit 1
fi

set -e

echo "=== task $SLURM_ARRAY_TASK_ID: $LATTICE on $NPROC process(es), 1 M-row chunks ==="
# Not 64. Sixty-four workers each holding a 2 M-row chunk with six peak budgets is what OOM-killed
# the first attempt, and above one process per file nothing extra is used anyway.
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" \
    --bravais-lattices "$LATTICE" \
    --processes "$NPROC" \
    --chunk-rows 1000000

echo
echo "=== verify $LATTICE ==="
# Exit code 0 is not evidence. C2-F-071 is an entire Bravais lattice lost from Benchmark B while
# all 24 generation tasks exited 0, and C2-F-135 is two cluster jobs that failed while exiting 0.
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --bravais-lattices "$LATTICE" --verify

echo
echo "DONE: $LATTICE. When ALL THREE tasks have finished, check the whole pool:"
echo "  $PYTHON run_fom_holdout_merits.py --pool $POOL --verify"
echo "It must print 'all sidecars complete and populated'. Only then is it worth adding"
echo "'holdout' to FEATURE_GROUPS in submit_fom_combiner_export.sh and re-exporting."
