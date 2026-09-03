#!/bin/bash
# S12 -- finish the nine hold-out sidecars the OOM took with it.
#
#   sbatch submit_fom_holdout_finish.sh
#
# WHAT HAPPENED. `submit_fom_combiner_fullscale.sh` hit its 6 h walltime part-way through the
# hold-out pass and lost two workers to the OOM killer on the way (C2-F-139). `--verify` then said
# exactly what was lost:
#
#   738 754 260 of 880 704 233 candidates carry hold-out merits
#   9 problems, ALL of them "NO SIDECAR" -- no truncated file, no wholly-null column
#
# That distinction is the whole reason `--verify` exists. These jobs resume by skipping files that
# already exist, so a sidecar an OOM had truncated mid-write would be skipped forever and every
# number downstream would rest on it. Nothing was truncated: the workers died before writing, not
# during. So a plain resume is safe here, and it is safe BECAUSE it was checked, not because
# parquet writes happen to be late.
#
# The nine are the nine largest files -- oP x5, aP x2, tP x2, 9.5 M to 21.4 M candidates each,
# 141 949 973 in total, which is exactly 880 704 233 - 738 754 260. That is the OOM's signature:
# the scheduler runs the biggest files last, and by then 64 workers were each holding a 2 M-row
# chunk with six peak budgets on it.
#
# SO: FAR FEWER PROCESSES, AND A SMALLER CHUNK. There are only nine files, so more than nine
# processes buys nothing at all -- and nine large files at 64-way concurrency is what failed.
# 8.9 core-hours of work; ~1.5 h on 9 processes.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_ho_finish
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 3:00:00
#SBATCH -o fom_ho_finish_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

POOL="$SCRATCH/fom_campaign2/pool"

# Only the three lattices that lost files. The other eleven are complete and would be skipped
# anyway, but naming them keeps the job's intent legible in its own log.
LATTICES="oP aP tP"

set -e

echo "=== before: what is missing ==="
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --verify || true

echo
echo "=== finishing $LATTICES on 9 processes, 1 M-row chunks ==="
# 9, not 64: there are nine files left, so nothing above nine is used, and 64-way on files this
# large is what triggered the OOM. Half the default chunk for headroom.
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" \
    --bravais-lattices $LATTICES \
    --processes 9 \
    --chunk-rows 1000000

echo
echo "=== after: this must print 'all sidecars complete and populated' ==="
# Exit code 0 is not evidence. C2-F-071 is an entire Bravais lattice lost from Benchmark B while
# all 24 generation tasks exited 0, and C2-F-135 is two cluster jobs that failed while exiting 0.
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --verify

echo
echo "DONE. All four sidecar families are now complete, so the export can run:"
echo "  sbatch submit_fom_combiner_export.sh"
