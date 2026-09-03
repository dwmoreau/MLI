#!/bin/bash
# S12 -- a FULLY RETAINED arm over the whole hard stratum of `fom-dev`. One node, under an hour.
#
#   python mlindex/scripts/run_fom_floor_entries.py --stratum hard --tag S12_hard   <- laptop FIRST
#   docs/sync_record.sh push                        <- gets the entry list here; docs/ is git-ignored
#   sbatch submit_fom_hard_retained.sh
#
# WHY THIS EXISTS, AND WHY IT IS THE CHEAPEST THING THE CAMPAIGN CAN BUY RIGHT NOW.
#
# Two separate problems meet on the hard stratum and one run fixes both.
#
#   1. C2-R-013. A learned score is not one of the seven merits the negative subsampler ranked on,
#      so on Benchmark B the candidates that would have outranked a correct one were kept at 5 %
#      and every rank metric for it is optimistic. `FomMetrics.evaluate` REFUSES to report one.
#      The remedy the register names is a fully retained arm over the entries the claim covers.
#
#   2. C2-R-019. The fully retained pool that does exist is S08's FLOOR SAMPLE -- 530 crystals
#      drawn balanced across lattices to measure reproducibility, not a stratified benchmark. Its
#      hard stratum is **20 (entry, condition) cells over 20 crystals, of which 6 are reachable**,
#      and in-sample M20 itself scores exactly 0.0000 on it. So every hard-stratum number in S12
#      is a statement about twenty patterns and says nothing.
#
# The hard stratum is where this campaign's gains are supposed to live -- mP, mC and aP at high
# volume -- and it is the stratum S06 sized the split to be able to carry. `fom-dev` holds **360**
# of them. This generates all 360 under the same three conditions the floor arms used, keeping
# every candidate.
#
# COST. 360 patterns x 3 conditions = 1 080 cells at the benchmark's own measured 2 470 cells per
# node-hour, so **under half a node-hour**. Walltime is 1 h against ~13 min deliberately: that rate
# is Benchmark B's aggregate and this sample is entirely low-symmetry, where pools are largest.
#
# DISK: ~4-6 GB. A hard-lattice cell holds more survivors than the benchmark median, and nothing is
# thinned. `--no-subsample` still writes `sampling_weight`, `retained_reason` and `pool_size_full`,
# all 1.0 / `correct` / the true count, so the output reads through the same loader.
#
# WHY IT REPRODUCES BENCHMARK B'S CANDIDATES. Every parameter below is Benchmark B's own, read from
# its manifest: seed 12345, optimizer seed 12345, pool_size 2, cut 1.5. C2-F-058 established that a
# run restricted to a subset of entries reproduces the full run bit for bit, so this generates the
# same candidates and simply keeps all of them.
#
# POOLSIZE IS PART OF THE BENCHMARK'S IDENTITY, NOT A PERFORMANCE KNOB (C2-F-069). The per-pattern
# search seed is keyed on the rank and the rank count IS pool_size, so a different value is a
# different search and this stops reproducing Benchmark B at all. NPOOLS is free.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the 128
# processes. Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_hard_c2
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 1:00:00
#SBATCH --array=0-2
#SBATCH -o fom_hard_c2_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
ENTRIES="$REPO/docs/fom_campaign2/artifacts/S12_hard_entries.csv"
OUTROOT="$SCRATCH/fom_campaign2/hard_retained"

# Benchmark B's own. Changing any of these stops this reproducing its candidates, and the point of
# the arm is that it is the same search with nothing thrown away.
SEED=12345
OPTIMIZER_SEED=12345
NPOOLS=64
POOLSIZE=2

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first." >&2
    exit 1
fi
if [ ! -f "$ENTRIES" ]; then
    echo "FATAL: hard-stratum entry list missing at $ENTRIES" >&2
    echo "On the laptop: python mlindex/scripts/run_fom_floor_entries.py --stratum hard \\" >&2
    echo "                   --tag S12_hard" >&2
    echo "then 'docs/sync_record.sh push'. Do NOT re-derive it here: the list must be the frozen" >&2
    echo "manifest's own hard stratum, or the arm covers different patterns from the claim." >&2
    exit 1
fi

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads)." >&2
fi

# The same three conditions the floor arms and the existing retained pool used, so the two retained
# pools can be concatenated into one reporting population rather than compared across conditions.
CONDITIONS=("nominal" "noisy" "control")
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

TAG=$("$PYTHON" run_fom_dump.py --condition "$CONDITION" --print-tag)
OUTDIR="$OUTROOT/$TAG"

echo "=== hard stratum, fully retained: condition $CONDITION ($TAG) -> $OUTDIR ==="
echo "    360 fom-dev crystals of mP/mC/aP at volume decile >= 8, every candidate kept"

"$PYTHON" run_fom_dump.py \
    --condition "$CONDITION" \
    --split-manifest "$MANIFEST" \
    --entry-ids-file "$ENTRIES" \
    --seed "$SEED" \
    --optimizer-seed "$OPTIMIZER_SEED" \
    --n-pools "$NPOOLS" \
    --pool-size "$POOLSIZE" \
    --no-subsample \
    --predownsample-entries 0 \
    --out-dir "$OUTDIR"

# AFTERWARDS: `sbatch submit_fom_hard_postprocess.sh`.
#
# That is a SCRIPT, not a list of commands in this comment, and the difference is not cosmetic --
# the first run of this job left the six post-processing steps unrun because they were prose here,
# so the array produced candidates and no pool. The consolidation cannot live inside the array
# either: it reads all three condition tags at once and an array task only knows its own.
#
# To make the ordering impossible to get wrong:
#
#   JOB=$(sbatch --parsable submit_fom_hard_retained.sh)
#   sbatch --dependency=afterok:$JOB submit_fom_hard_postprocess.sh
#
# What it does: consolidates $OUTROOT into $SCRATCH/fom_campaign2/hard_pool, writes the four
# sidecar sets a pool needs before anything can score it, and verifies them. ~40 min.
