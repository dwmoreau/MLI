#!/bin/bash
# S07 -- generate Benchmark B, the frozen candidate pool campaign 2 is developed against.
#
# 24 array tasks: 5 core bundles x 4 shards, plus 4 mechanism bundles unsharded. ~43 node-hours.
#
#   sbatch submit_fom_dump_calibration.sh     <- FIRST. Read its runbook header.
#   sbatch submit_fom_dump.sh
#
# THE GRID (frozen by S06 -- do not re-derive it from --n-entries-per-bl, the driver reads the
# entry list from the manifest and a sampling parameter cannot reproduce it, C2-F-048):
#
#   core arm       18 991 crystals x 5 bundles = 94 955 cells
#                  control, nominal, noisy, contaminated, second_phase
#   mechanism arm   2 846 crystals x 4 bundles = 11 384 cells   (a NESTED subset of the core arm)
#                  sparse2, sparse4, sparse6, error_shape
#                                              ------
#                                             106 339 cells
#
# The arms nest deliberately: every mechanism entry is also a core entry, assigned once in the
# manifest, so the two arms are comparable. Sizing them independently would have drawn different
# crystals.
#
# SHARDING. A core bundle is 18 991 cells at ~1.5 s/cell over 128 processes, so ~7.8 h whole --
# too long for one task. Four shards gives ~2 h tasks, each independently requeueable:
# run_fom_dump.py skips pools whose output is already written AND readable, so a requeued task
# resumes rather than restarts. A mechanism bundle is ~1.2 h and runs unsharded.
#
# WALLTIME is 4 h against a projected ~2 h. That slack is deliberate over a quantity measured on a
# laptop and projected onto a Perlmutter node -- replace it with the calibration job's number.
#
# ONE DIRECTORY PER BUNDLE. One manifest.json is written per --out-dir, so two bundles sharing a
# directory silently overwrite each other's. The consolidator now refuses the mirror-image mistake
# (two directories holding one bundle), but nothing catches this one but the layout.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the 128
# processes. Read SLURM_CPUS_ON_NODE, not nproc -- and HALVE it, because it counts both
# hyperthreads while NPOOLS x POOLSIZE should equal the 128 physical cores.
#
# --conventional-cell stays off, and there is no flag for it here. Campaign 1 left it off for
# comparability and had seen it crash 10 of 16 pools on a cctbx error.
#
# DISK: ~119 GB of survivors + ~74 GB pre-deduplication + a consolidated copy, against the 1 TB
# budget on $SCRATCH (DWMM, 2026-08-27). Check headroom before submitting:  df -h $SCRATCH

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_dump_c2
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH --array=0-23
#SBATCH -o fom_dump_c2_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
OUTROOT="$SCRATCH/fom_campaign2/benchmark"
SEED=12345

# Set from submit_fom_dump_calibration.sh. NPOOLS x POOLSIZE must equal the PHYSICAL core count.
NPOOLS=32
POOLSIZE=4

# The pre-deduplication stream is ~7.7x the survivor stream, so it is written for a stratified
# subsample of entries rather than all of them. Entries arrive in manifest order and the manifest
# is lattice-stratified, so a prefix is a stratified subsample. 1000 entries over the 5 core
# bundles is ~74 GB. The ratio is re-measured on aP by the calibration job.
PREDOWNSAMPLE=8

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first. The driver reads its ENTRY LIST" >&2
    echo "from this file, so a missing or stale copy is a different benchmark, not a missing" >&2
    echo "column. Expected sha256 3dd52c5eb2546dacca3034ebd2fd052dcd2acd4a8f9af24ce972fe4e0a210969" >&2
    exit 1
fi

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads). Undersubscribing wastes the reservation;" >&2
    echo "oversubscribing makes it slower." >&2
fi

# task -> (condition, arm, shard, n_shards). The core bundles are sharded four ways; the mechanism
# bundles run whole. `--arm mechanism` selects the nested subset from the manifest's own column.
TASKS=(
    "control      core      0 4" "control      core      1 4"
    "control      core      2 4" "control      core      3 4"
    "nominal      core      0 4" "nominal      core      1 4"
    "nominal      core      2 4" "nominal      core      3 4"
    "noisy        core      0 4" "noisy        core      1 4"
    "noisy        core      2 4" "noisy        core      3 4"
    "contaminated core      0 4" "contaminated core      1 4"
    "contaminated core      2 4" "contaminated core      3 4"
    "second_phase core      0 4" "second_phase core      1 4"
    "second_phase core      2 4" "second_phase core      3 4"
    "sparse2      mechanism 0 1"
    "sparse4      mechanism 0 1"
    "sparse6      mechanism 0 1"
    "error_shape  mechanism 0 1"
)

read -r CONDITION ARM SHARD NSHARDS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

# The tag rule has ONE implementation and it is in Python. Campaign 1 kept a partial copy of it in
# bash which omitted two components; this asks the driver instead.
TAG=$($PYTHON run_fom_dump.py --condition "$CONDITION" --print-tag) || exit 1

ARM_FLAG=""
[ "$ARM" = "mechanism" ] && ARM_FLAG="--arm mechanism"

echo "array task $SLURM_ARRAY_TASK_ID: $CONDITION ($ARM arm), shard $SHARD of $NSHARDS -> $TAG"
echo "topology ${NPOOLS}x${POOLSIZE}, seed $SEED, out $OUTROOT/$TAG"

$PYTHON run_fom_dump.py \
    --condition "$CONDITION" \
    $ARM_FLAG \
    --split-manifest "$MANIFEST" \
    --n-pools "$NPOOLS" --pool-size "$POOLSIZE" \
    --shard "$SHARD" --n-shards "$NSHARDS" \
    --predownsample-entries "$PREDOWNSAMPLE" \
    --seed "$SEED" \
    --out-dir "$OUTROOT/$TAG"
STATUS=$?

echo "task $SLURM_ARRAY_TASK_ID exited $STATUS"

# WHEN THE WHOLE ARRAY IS DONE, on a login node:
#
#   $PYTHON run_fom_dump_consolidate.py \
#       --dump-root $SCRATCH/fom_campaign2/benchmark \
#       --out-dir   $SCRATCH/fom_campaign2/pool \
#       --artifact-dir $REPO/docs/fom_campaign2/artifacts
#
#   $PYTHON run_fom_dump_gate.py floor \
#       --split-manifest $MANIFEST \
#       --reachability   <calibration per-lattice reachability csv> \
#       --artifact-dir   $REPO/docs/fom_campaign2/artifacts
#
#   $PYTHON run_fom_dump_gate.py check \
#       --pool      $SCRATCH/fom_campaign2/pool \
#       --full-pool $SCRATCH/fom_campaign2/pool_full \
#       --artifact-dir $REPO/docs/fom_campaign2/artifacts
#
# `pool_full` is a held-back fully-retained shard -- one task re-run with --no-subsample over a
# reserved entry subset. It is the only check that the retention rule did not change what the
# benchmark measures (gate 6), so reserve it deliberately rather than skipping it.
exit $STATUS
