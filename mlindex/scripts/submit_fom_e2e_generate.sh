#!/bin/bash
# S15 -- the end-to-end grid: REAL runs of the indexer at cuts 5.0, 3.5 and 3.0, both populations,
# every candidate kept, then the sidecars the ranking merits read. Six tasks, one per (population,
# cut); each loops over its bundles serially and resumes after a failure.
#
#   python mlindex/scripts/run_fom_end_to_end.py --stage plan     <- laptop FIRST: writes the design
#   docs/sync_record.sh push                                       <- gets it here; docs/ is git-ignored
#   sbatch submit_fom_e2e_generate.sh
#   sbatch --dependency=afterok:<jobid> submit_fom_e2e_reduce.sh
#
# WHY THE GENERATOR AND NOT run.py. The shipped indexer ranks by M20 and nothing else, and the prune
# threshold is deliberately not a CLI option (decision 2026-08-24, C2-F-008). `run_fom_dump.py` IS
# the indexer -- the same optimizer through the same `setup_mp_optimizers` / `run_mp_bl` path, the
# cut reaching it as `opt_params['prune_m20_threshold']` -- and it persists the whole
# post-deduplication pool with `final_rank`, `in_top_n`, `n_entering` and `pool_size_full`, which is
# what a merit is then scored over. `in_top_n` is the top twenty per lattice production hands to its
# final sort. With `--no-subsample` every candidate is kept, so every rank is exact (C2-R-013).
#
# WHY THESE PARAMETERS ARE FIXED. Seed 12345, optimizer seed 12345, pool size 2: Benchmark B's own,
# so the cut-1.5 pools already on disk for the 530 general crystals are the SAME search at a lower
# cut and join the factorial as a fourth real run. Pool size is part of the identity (C2-F-069);
# NPOOLS is free. The peak list depends on the seed, the entry and the condition and never on the
# cut, and the analyse stage checks that digest for digest across every arm (gate 5).
#
# WHAT THE TASKS ARE.
#   0 general 5.0    1 general 3.5    2 general 3.0     530 crystals x 9 bundles = 4 770 cells each
#   3 hard    5.0    4 hard    3.5    5 hard    3.0     360 crystals x 5 hard bundles = 1 800 cells
#
# COST. Benchmark B generated at ~2 470 cells per node-hour at cut 1.5; a higher cut is cheaper per
# cell (the search dominates and is cut-independent, S03), so a general task is ~2 h of generation
# and the hard task under an hour. The sidecars are minutes here: a cut-5.0 pool holds ~100-2 000
# candidates a cell against ~27 000 at 1.5. Walltime 6 h is deliberate slack.
#
# --record-timing writes per-entry wall clock onto the entry table, so `S15_cost.csv` is measured on
# the run that made the pool rather than projected from a model. It is as ONE pool saw it; node
# throughput is NPOOLS / seconds_total.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the 128
# processes. Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_e2e
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 6:00:00
#SBATCH --array=0-5
#SBATCH -o fom_e2e_%A_%a.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO" || exit 1

ARTIFACTS="$REPO/docs/fom_campaign2/artifacts"
MANIFEST="$ARTIFACTS/S06_split_manifest.parquet"
DESIGN="$ARTIFACTS/S15_design.json"
OUTROOT="$SCRATCH/fom_campaign2"
NPOOLS=64
POOLSIZE=2
PROCESSES=64

POPULATIONS=(general general general hard hard hard)
CUTS=(5.0 3.5 3.0 5.0 3.5 3.0)
POPULATION=${POPULATIONS[$SLURM_ARRAY_TASK_ID]}
CUT=${CUTS[$SLURM_ARRAY_TASK_ID]}
ENTRIES="$ARTIFACTS/S15_entries_${POPULATION}.csv"

for NEEDED in "$MANIFEST" "$DESIGN" "$ENTRIES"; do
    if [ ! -f "$NEEDED" ]; then
        echo "FATAL: $NEEDED is missing." >&2
        echo "Run 'python mlindex/scripts/run_fom_end_to_end.py --stage plan' on the laptop," >&2
        echo "then 'docs/sync_record.sh push'. The entry lists are NOT re-drawn here: they are" >&2
        echo "the crystals the cut-1.5 pools were built from, and a different list makes every" >&2
        echo "cross-cut comparison unpaired." >&2
        exit 1
    fi
done

CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
    echo "WARNING: NPOOLS x POOLSIZE = $((NPOOLS * POOLSIZE)) but the node has $PHYSICAL" >&2
    echo "physical cores ($CPUS hyperthreads)." >&2
fi

set -e

echo "=== S15 generate: $POPULATION cut $CUT -> $OUTROOT/e2e/$POPULATION/ ==="
echo "    $(wc -l < "$ENTRIES") lines in $ENTRIES; ${NPOOLS}x${POOLSIZE}; every bundle in turn,"
echo "    skipping any whose manifest exists, so a re-submit resumes rather than repeats"
# The driver loops over every bundle of the population, writes provenance.json BEFORE the first
# bundle, and builds each run_fom_dump argv itself -- one implementation of the invocation.
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage generate \
    --population "$POPULATION" --cut "$CUT" \
    --out-root "$OUTROOT" --split-manifest "$MANIFEST" \
    --n-pools "$NPOOLS" --pool-size "$POOLSIZE"

echo "=== stamp the arm complete (refused if any bundle is missing its manifest) ==="
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage complete \
    --population "$POPULATION" --cut "$CUT" --out-root "$OUTROOT"

echo "=== consolidate + the sidecars S12's model reads, each followed by its --verify ==="
# Exit code 0 is not evidence (C2-F-071, C2-F-135, C2-F-139): every producer is verified.
"$PYTHON" mlindex/scripts/run_fom_end_to_end.py --stage sidecars \
    --population "$POPULATION" --cut "$CUT" --out-root "$OUTROOT" \
    --processes "$PROCESSES" --python "$PYTHON" --execute

echo
echo "DONE: $OUTROOT/e2e/$POPULATION/cut${CUT%.0}_pool"
du -sh "$OUTROOT/e2e/$POPULATION/"*_pool 2>/dev/null || true
echo
echo "When all six tasks are done: sbatch submit_fom_e2e_reduce.sh"
