#!/bin/bash
# S08 -- the reproducibility floor's ensemble: the same patterns indexed again under different
# search seeds, so the spread of a reported number over runs that are nominally the same can be
# measured rather than assumed.
#
#   python run_fom_floor_entries.py        <- FIRST, on the laptop. Writes the entry list.
#   docs/sync_record.sh push               <- gets that list to NERSC; docs/ is git-ignored
#   sbatch submit_fom_floor_arms.sh
#   python run_fom_floor_report.py --arm-root $SCRATCH/fom_campaign2/floor
#
# WHY THIS IS CHEAP, given it looks like a repeat of a 43-node-hour job. It is not a
# re-generation of Benchmark B. Two things make it small:
#
#   * It covers 530 patterns, not 106 235 (entry x bundle) cells. 530 x 3 conditions = 1 590
#     cells an arm.
#   * BENCHMARK B IS THE FIRST ARM, for free. It was generated at a recorded --optimizer-seed,
#     and C2-F-058 established that a run restricted to a subset of entries reproduces the full
#     run bit for bit -- 0 differing columns of 34 -- so its existing rows for these 530 entries
#     ARE an arm. Only the other three have to be generated.
#
#   3 arms x 1 590 cells = 4 770 cells at ~189 process-seconds a cell over 128 processes
#     = ~1.9 node-hours, against the benchmark's 43. Under 5 %.
#
# WHAT MUST NOT MOVE. Every arm shares --seed, which fixes the entry sample and the per-entry
# noise and therefore the peak lists; only --optimizer-seed moves, and it reaches the search
# alone. If --seed moved as well the arms would differ in their DATA and the spread would be
# generation noise and scoring noise together, which is the one distinction the floor is made of.
# run_fom_floor_report.py compares every arm's q2_digest entry by entry and REFUSES to report a
# number if they disagree; that check is not optional and is not a warning.
#
# POOLSIZE IS PART OF THE BENCHMARK'S IDENTITY (C2-F-069). The per-pattern search seed is keyed on
# the rank and the rank count IS pool_size, so an arm at a different pool_size is a different
# search and its spread against Benchmark B would measure that instead of the seed. Benchmark B
# was generated at 64 x 2; POOLSIZE must stay 2. NPOOLS is free -- the pool is invariant to it.
#
# THE CUT stays at the driver's default of 1.5, the value Benchmark B was generated at. A floor
# measured at a different cut is a floor for a different pool.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the 128
# processes. Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.
#
# DISK: ~1 590 cells x ~1.1 MB x 3 arms = ~5 GB. Not a constraint.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_floor_c2
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 1:00:00
#SBATCH --array=0-8
#SBATCH -o fom_floor_c2_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
ENTRIES="$REPO/docs/fom_campaign2/artifacts/S08_floor_entries.csv"
OUTROOT="$SCRATCH/fom_campaign2/floor"

# Shared by every arm. Moving this moves the peak lists and the floor stops being a floor.
SEED=12345

NPOOLS=64
POOLSIZE=2

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first." >&2
    exit 1
fi
if [ ! -f "$ENTRIES" ]; then
    echo "FATAL: floor entry list missing at $ENTRIES" >&2
    echo "Run run_fom_floor_entries.py on the laptop, then 'docs/sync_record.sh push'." >&2
    echo "The list must NOT be re-drawn here: a second draw is a different sample, and the" >&2
    echo "aggregate is composed from the split's per-lattice counts recorded beside it." >&2
    exit 1
fi

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads)." >&2
fi

# task -> (optimizer seed, condition). Three arms beyond Benchmark B's own, over three conditions.
#
# Three rather than two, for one specific reason: several arms give several independent pairs,
# which is what lets the floor be checked two ways -- derived from the rate at which a pattern's
# outcome flips between arms, and measured directly from the spread across arms. Campaign 1 got
# 0.366 pp derived against 0.360 pp reported that way, and matching that standard is S08
# acceptance condition 3. Two arms give one pair and no cross-check.
#
# The conditions are the reference, one harder and the near-noise-free control, so that "the
# floor barely moves with the condition" is re-measured rather than inherited from F-150.
TASKS=(
    "202 nominal" "202 noisy" "202 control"
    "303 nominal" "303 noisy" "303 control"
    "404 nominal" "404 noisy" "404 control"
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
read -r ARM_SEED CONDITION <<< "$TASK"

TAG=$("$PYTHON" run_fom_dump.py --condition "$CONDITION" --print-tag)
OUTDIR="$OUTROOT/seed${ARM_SEED}/${TAG}"

echo "=== arm $ARM_SEED, condition $CONDITION ($TAG) -> $OUTDIR ==="
echo "    base seed $SEED (shared by every arm), optimizer seed $ARM_SEED (the only difference)"

"$PYTHON" run_fom_dump.py \
    --condition "$CONDITION" \
    --split-manifest "$MANIFEST" \
    --entry-ids-file "$ENTRIES" \
    --seed "$SEED" \
    --optimizer-seed "$ARM_SEED" \
    --n-pools "$NPOOLS" \
    --pool-size "$POOLSIZE" \
    --predownsample-entries 0 \
    --out-dir "$OUTDIR"
