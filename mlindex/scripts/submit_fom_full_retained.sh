#!/bin/bash
# S08 -- the fully retained pool: the same patterns as the floor sample, with NOTHING thinned.
#
#   python run_fom_floor_entries.py     <- FIRST, on the laptop. Writes the entry list.
#   docs/sync_record.sh push            <- gets that list here; docs/ is git-ignored
#   sbatch submit_fom_full_retained.sh
#
# ONE JOB, THREE PURPOSES. It is worth being deliberate about which entries this covers, because
# the same ~40 minutes answers three separate outstanding questions:
#
#   1. S07's GATE 6, the one acceptance condition Benchmark B came back `partial` on.
#      `run_fom_dump_gate.py`'s weights layer has two halves. That every correct candidate survived
#      and every certain retention carries weight 1.0 already passes -- it is checkable from the
#      thinned pool alone. That the WEIGHTED candidate count reproduces the true full-pool count
#      has never run, because no full pool existed; the gate says so in as many words ("no
#      --full-pool given, so the weighted-count check is NOT run"). It is the only evidence that
#      negative subsampling did not change what the benchmark measures, and every fit in phase 4
#      uses `sampling_weight`, so if the weights are wrong every one of those fits is biased and
#      nothing else in the pipeline would show it.
#
#   2. THE TIE-BREAK FLOOR, which is an S08 deliverable and provably cannot be measured on the
#      thinned pool. A constant score puts every candidate in a tie, so rank is decided entirely by
#      the tie-break order -- and that order is unrelated to the merits the retention rule ranked
#      on, so retention is effectively random with respect to it and a correct candidate is scored
#      against ~31 % of its true field. Campaign 1's 0.2657 was measured on an unsubsampled pool
#      and is not comparable to anything measured on this one.
#
#   3. A FULLY RETAINED ARM 1, which is the remedy C2-R-013 names. Rank exactness on Benchmark B
#      holds only for the seven merits the subsampler ranked on (C2-F-077); a learned score's rank
#      there is optimistic by an amount nobody has measured. This pool is where S12 and S14 can
#      measure it, or report on it directly, without regenerating anything.
#
# WHY IT REPRODUCES BENCHMARK B EXACTLY, which is what makes (1) a valid comparison rather than two
# different candidate sets being differenced. Every parameter below is Benchmark B's own, read from
# its manifest: seed 12345, optimizer seed 12345, pool_size 2, cut 1.5. C2-F-058 established that a
# run restricted to a subset of entries reproduces the full run bit for bit -- 0 differing columns
# of 34 -- so this generates the same candidates and simply keeps all of them.
#
# POOLSIZE IS PART OF THE BENCHMARK'S IDENTITY, NOT A PERFORMANCE KNOB (C2-F-069). The per-pattern
# search seed is keyed on the rank and the rank count IS pool_size, so a different value is a
# different search and this stops reproducing Benchmark B at all. NPOOLS is free.
#
# --no-subsample IS NOT --no-label. The retention rule keeps every correct candidate, so labelling
# has to happen either way; `--no-subsample` keeps every row AND still writes `sampling_weight`,
# `retained_reason` and `pool_size_full`, so the output reads through exactly the same loader.
#
# COST. 530 patterns x 3 conditions = 1 590 cells at the benchmark's own measured 2 470 cells per
# node-hour, so ~13 min a task and ~0.65 node-hours in total. Subsampling happens AFTER the search,
# so keeping everything costs no extra compute -- only disk. Walltime is 1 h against 13 min
# deliberately: the rate is Benchmark B's aggregate and this sample has a different lattice mix.
#
# DISK: ~6 GB, and likely less. A cell holds a median 26 734 survivors, but that median is over a
# 20-per-lattice sample; this one is balanced too, and the low-symmetry lattices whose pools
# dominate are a smaller share of it than of Benchmark B.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the 128
# processes. Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_full_c2
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 1:00:00
#SBATCH --array=0-2
#SBATCH -o fom_full_c2_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
ENTRIES="$REPO/docs/fom_campaign2/artifacts/S08_floor_entries.csv"
OUTROOT="$SCRATCH/fom_campaign2/full_retained"

# Benchmark B's own, from its manifest. Changing any of these breaks the reproduction and with it
# gate 6's comparison, which would then be differencing two different candidate sets.
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
    echo "FATAL: floor entry list missing at $ENTRIES" >&2
    echo "Run run_fom_floor_entries.py on the laptop, then 'docs/sync_record.sh push'." >&2
    echo "It must NOT be re-drawn here: this pool has to cover the same patterns as the floor" >&2
    echo "arms, or it serves only purpose (1) of the three above." >&2
    exit 1
fi

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads)." >&2
fi

# One task per condition, matching the floor sample's three.
CONDITIONS=("nominal" "noisy" "control")
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

TAG=$("$PYTHON" run_fom_dump.py --condition "$CONDITION" --print-tag)
OUTDIR="$OUTROOT/$TAG"

echo "=== fully retained: condition $CONDITION ($TAG) -> $OUTDIR ==="
echo "    seed $SEED, optimizer seed $OPTIMIZER_SEED, ${NPOOLS}x${POOLSIZE} -- Benchmark B's own,"
echo "    so this reproduces its candidates and keeps all of them rather than thinning to K=200"

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

# AFTERWARDS, on NERSC, to discharge gate 6. `--pool` is the THINNED pool being audited, which is
# Benchmark B itself; `--full-pool` is what this job just wrote. The gate compares weighted counts
# against true counts over the (entry, bundle, lattice) pools the two share, and passes at a mean
# relative error under 0.10 -- a sampling tolerance, not a numerical one.
#
#   $PYTHON run_fom_dump_consolidate.py --dump-root $OUTROOT --out-dir $SCRATCH/fom_campaign2/full_pool
#   $PYTHON run_fom_dump_gate.py check \
#       --pool      $SCRATCH/fom_campaign2/pool \
#       --full-pool $SCRATCH/fom_campaign2/full_pool \
#       --artifact-dir $REPO/docs/fom_campaign2/artifacts --keep-going
#
# Then copy the full pool back for the tie-break floor, which cannot be measured on a thinned one.
