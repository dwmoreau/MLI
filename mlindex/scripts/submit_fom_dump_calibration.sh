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

# ENOUGH ENTRIES THAT EACH POOL GETS SEVERAL. The first version used NPERBL=2, which is 28 entries
# over 32 pools: one entry per pool, four pools idle, and every measurement equal to
# startup + one entry. Startup is ~111 s per pool (model loading, and only managers load models),
# so that measured almost nothing but model loading -- seff reported 2.8% CPU efficiency, and the
# two topologies came out within 1% of each other because both were reporting the same constant.
#
# The number that sizes the walltime is the MARGINAL cost per entry, and separating it from startup
# needs several entries per pool. NPERBL=20 is 280 entries: ~9 per pool at 32 pools, ~18 at 16, ~4
# at 64. Each run then reports s/entry close to the marginal rate rather than to the startup.
#
# C5 is the timing worst case of the seven -- dropout pushes the window to higher q2, where the
# reference line lists are denser, so it is the bundle to size against.
NPERBL=20

# 128 = the node's PHYSICAL core count. SLURM_CPUS_ON_NODE reports 256 on Perlmutter CPU nodes,
# which counts the two hardware threads per core; the S02 calibration used 128 and that is what
# these topologies keep. Memory is not the axis that binds: 32 pools measured 62.4 GB of the node's
# 476 GB, i.e. ~2 GB per manager, so even 64 pools is ~125 GB.
for TOPOLOGY in "32 4" "16 8" "64 2"; do
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
echo "Each pool above ran $((NPERBL * 14 / 32)) or so entries, so its s/entry still carries a share"
echo "of the ~111 s startup. Subtract it before projecting:"
echo
echo "    marginal  = (s_per_entry * n_entries_in_pool - 111) / n_entries_in_pool"
echo "    hours     = (111 + 5955 / NPOOLS * marginal) / 3600"
echo
echo "Confirm the winner fits inside submit_fom_dump.sh's 5 h with real headroom. The marginal rate"
echo "depends on POOLSIZE -- only the optimisation iterations distribute to workers, candidate"
echo "generation runs on the manager -- so a rate measured at one topology does NOT transfer to"
echo "another. That is why all three are run here rather than two."
echo
echo "This script does not measure memory. Get it from the scheduler afterwards:"
echo "    seff \$SLURM_JOB_ID"
echo "Only managers load models, so NPOOLS is the memory-limited axis: 32 pools measured 62.4 GB"
echo "of the node's 476 GB (~2 GB per manager), and the dump's accumulated rows add ~85 MB per pool"
echo "over a full bundle, which is not the constraint."
