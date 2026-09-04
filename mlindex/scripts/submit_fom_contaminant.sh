#!/bin/bash
# S12 -- the five contaminated / sparse / second-phase bundles, fully retained. ONE JOB.
#
#   docs/sync_record.sh push        # on the laptop: this reads the manifest and entry list
#   sbatch submit_fom_contaminant.sh
#
# WHY THIS EXISTS. Everything S12 reports rests on three condition bundles -- 0.1x, 1x and 2x
# Gaussian peak-position error, all with NO contaminants. The benchmark HAS contaminated bundles and
# the combiner is FITTED on them; it has never been REPORTED on one, because a rank claim needs a
# pool nothing was thinned from and the fully retained pool was generated with the three severity
# bundles alone (C2-R-024). Nothing in the record says what the learned score does with a peak no
# cell can index, which is what real data carries. This closes that.
#
# THE CONTAMINANTS ARE IN THE PEAKS THE CANDIDATE IS RANKED ON. Measured on the slice, counting
# hkl = (0,0,0) rows of the 20-peak `q2_obs`:
#
#   c2_error1_cont2         1.90 of 2 in the window, 0.10 in the surplus, 100% of entries hit
#   c2_error1_cont1_drop*   1.00 of 1 in the window, 0.00 in the surplus, 100% hit
#   c2_error1_cont0_phase3  2.78 of 3 in the window, 0.16 in the surplus, 100% hit
#
# `add_contaminants` draws over the WINDOW's own q2 range and re-truncates to the same length, so a
# contaminant is placed inside the ranking budget by construction AND DISPLACES A REAL REFLECTION
# into the hold-out. A cont2 pattern is 18.1 real reflections + 1.9 unindexable lines, not 20 + 2 --
# you lose signal and gain noise together, as a real impurity phase does.
#
# WHAT EACH BUNDLE IS FOR, because they are not interchangeable:
#
#   contaminated  c2_error1_cont2         2 independent lines, 60-peak pattern, NO dropout.
#                                         The clean read: contaminants and nothing else moving.
#   second_phase  c2_error1_cont0_phase3  3 lines from a REAL PARTNER CELL from the database.
#                                         **Correlated, not independent** -- they follow a lattice,
#                                         so a wrong cell can genuinely index some of them. That is
#                                         what makes them hard to reject and makes this the case
#                                         closest to a real two-phase sample.
#   sparse2/4/6   c2_error1_cont1_drop*   1 contaminant AND 2/4/6 peaks dropped AND a 31-peak
#                                         pattern. THREE things move at once, so read this series
#                                         as a sparsity-plus-contaminant trend, not as a
#                                         contaminant measurement.
#
# **THE SAME 530 CRYSTALS AS THE EXISTING POOL, AND THAT IS THE POINT.** `S08_floor_entries.csv` is
# the entry list `fom_full_c2_pool` was built from -- verified 530 of 530 against the pool's own
# `fom-dev` ids. Reusing it makes every new bundle PAIRED with the three already reported: the same
# crystal under a new condition, so a McNemar over crystals is valid and a difference cannot be a
# difference of populations. Do not re-derive this list.
#
# ONE JOB, NOT AN ARRAY. Generation is ~9 minutes a condition on a full node, so five of them run
# serially in under an hour and the sidecar passes dominate the job either way. An array would have
# saved ~35 minutes of wall clock at the cost of four extra node allocations and a dependency chain.
# Generation SKIPS a condition whose output already exists, so this script is safe to re-submit
# after a failure: it resumes rather than repeating.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_cont_c2
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 8:00:00
#SBATCH -o fom_cont_c2_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
ENTRIES="$REPO/docs/fom_campaign2/artifacts/S08_floor_entries.csv"
OUTROOT="$SCRATCH/fom_campaign2/contaminant_retained"
POOL="$SCRATCH/fom_campaign2/contaminant_pool"
SEED=12345
OPTIMIZER_SEED=12345
NPOOLS=64
POOLSIZE=2
PROCESSES=64

CONDITIONS=(contaminated second_phase sparse2 sparse4 sparse6)

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first." >&2
    exit 1
fi
if [ ! -f "$ENTRIES" ]; then
    echo "FATAL: entry list missing at $ENTRIES" >&2
    echo "It is the list fom_full_c2_pool was built from. Do NOT re-derive it: a different list" >&2
    echo "makes the new bundles a different population and every comparison unpaired." >&2
    exit 1
fi

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads)." >&2
fi

set -e

for CONDITION in "${CONDITIONS[@]}"; do
    TAG=$("$PYTHON" run_fom_dump.py --condition "$CONDITION" --print-tag)
    OUTDIR="$OUTROOT/$TAG"
    if [ -n "$(find "$OUTDIR" -name '*.parquet' -print -quit 2>/dev/null)" ]; then
        echo "=== $CONDITION ($TAG): output exists, skipping generation ==="
        continue
    fi
    echo "=== generate $CONDITION ($TAG) -> $OUTDIR ==="
    echo "    530 fom-dev crystals, the SAME list fom_full_c2_pool was built from"
    echo "    --no-subsample: every candidate kept, which is what makes the ranks exact"
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
done

TAGS=$(find "$OUTROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$TAGS" -ne "${#CONDITIONS[@]}" ]; then
    echo "FATAL: expected ${#CONDITIONS[@]} condition directories under $OUTROOT, found $TAGS." >&2
    echo "Consolidating a partial set would silently build a pool over fewer conditions than" >&2
    echo "every downstream number assumes." >&2
    find "$OUTROOT" -mindepth 1 -maxdepth 1 -type d >&2
    exit 1
fi

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
# OOM-killed the full-scale hold-out pass, and above one process per file nothing extra is used --
# a file is an all-or-nothing unit of work, so processes beyond the file count buy nothing
# (C2-F-139).
echo "=== the hold-out family (S10) ==="
"$PYTHON" run_fom_holdout_merits.py --pool "$POOL" --processes 16 --chunk-rows 1000000

echo "=== verify: every sidecar present, complete and populated ==="
# Exit code 0 is not evidence. C2-F-071 lost an entire Bravais lattice while all 24 generation
# tasks exited 0; C2-F-135 is two cluster jobs that failed while exiting 0; C2-F-139 is a pass that
# lost nine files to an OOM and would have been resumed over in silence without this.
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
echo "Then S12 can answer what it currently cannot: what the learned score does on a pattern"
echo "carrying peaks no cell can index. The 530 crystals are the SAME ones the three clean bundles"
echo "use, so every comparison is paired over crystals."
