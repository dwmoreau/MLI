#!/bin/bash
# S04 stage 0 -- time the dump before spending the array on it.
#
#   sbatch submit_fom_dump_calibration.sh
#
# or interactively:
#
#   salloc -N 1 -C cpu -q interactive -t 1:00:00 -A lcls
#   ./submit_fom_dump_calibration.sh
#
# WHY THIS RUNS FIRST. submit_fom_dump.sh sizes its 5 h walltime from the 48.3 s/entry/pool the S02
# mirror measured at 32x4. The dump is not the mirror: it writes ~566 candidate rows per entry
# instead of one summary row, and it holds them in the pool process until the end. If that costs
# materially more than the mirror did, seven array tasks discover it simultaneously, five hours in,
# and the grid is lost. Thirty debug-queue minutes is the cheap version of that question.
#
# The S02 calibration's two hard-won details, both reproduced here:
#
#   * DO NOT wrap the driver in `srun`. A bare `srun -n 1` pins CPU affinity to a single core and
#     strangles the 128 processes; the driver forks its own pools and wants the whole node.
#   * Read SLURM_CPUS_ON_NODE, not `nproc`. Under salloc the login shell gets a narrow affinity
#     mask and nproc reports a handful of cores rather than the node's.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J fom_dump_cal
#SBATCH -A lcls
#SBATCH -t 0:30:00

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom/artifacts/S02_mirror_manifest.parquet"
OUTROOT="$REPO/mlindex/characterization/fom/benchmark_cal"

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first." >&2
    exit 1
fi

NCORES=${SLURM_CPUS_ON_NODE:-128}
echo "node reports $NCORES cores"

# Two entries per lattice is ~28 entries: enough for a stable s/entry, small enough for the debug
# queue. C5 is the timing worst case of the seven -- dropout pushes the window to higher q2, where
# the reference line lists are denser, so it is the bundle to size against.
NPERBL=2

for TOPOLOGY in "32 4" "16 8"; do
    read -r NPOOLS POOLSIZE <<< "$TOPOLOGY"
    TAG="cal_${NPOOLS}x${POOLSIZE}"
    echo
    echo "=== $NPOOLS pools x $POOLSIZE processes = $((NPOOLS * POOLSIZE)) ==="
    START=$(date +%s)
    $PYTHON run_fom_dump.py \
        --error-multiplier 1 --n-contaminants 1 --n-dropout 10 \
        --n-entries-per-bl "$NPERBL" \
        --n-pools "$NPOOLS" --pool-size "$POOLSIZE" \
        --shard 0 --n-shards 1 \
        --split-manifest "$MANIFEST" \
        --allow-unassigned-split \
        --out-dir "$OUTROOT/$TAG"
    echo "$TAG wall: $(( $(date +%s) - START )) s"
done

echo
echo "Per-pool s/entry is printed by each run above. Project a full bundle as:"
echo "    5955 entries / NPOOLS * (s/entry/pool) / 3600 = hours"
echo "and confirm it fits inside submit_fom_dump.sh's 5 h with headroom before submitting."
echo
echo "Also check the pool's peak RSS: only managers load models, so NPOOLS is the memory-limited"
echo "axis. 32 managers at ~3 GB is ~96 GB of the node's 512 GB; the dump adds its accumulated"
echo "rows on top, ~85 MB per pool over a full bundle, which is not the constraint."
