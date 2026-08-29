#!/bin/bash
# S07 -- time the generation run before spending the array on it, and measure the two numbers
# the array's sizing is still soft on.
#
# ~20 min in the debug queue. It answers three questions, and the third is a gate input:
#
#   1. WHICH POOL TOPOLOGY. NPOOLS x POOLSIZE must equal 128, the node's PHYSICAL core count, but
#      how to split it is not obvious: only managers load models (~3 GB each), so more pools costs
#      memory, while a bigger pool spends more time in the queue handoff. Campaign 1's mirror
#      measured 32x4 best of the two it compared; the dump's own comparison was never conclusive.
#
#   2. THE PRE-DEDUPLICATION RATIO ON aP. It is 7.7x the survivor stream, measured on cP and tP,
#      whose pools are one or two candidates -- so it is measured where there is nothing to
#      measure. aP survives at ~6 100 candidates per pattern. This sets --predownsample-entries
#      for the array, and it is ~74 GB of the ~310 GB budget.
#
#   3. PER-LATTICE REACHABILITY, which sets the acceptance gate's correct-candidate floor. It is
#      measured today on the hard stratum only (77.4 %, Wilson low 71.7 %, C2-F-049), and the
#      floor currently applies that one rate to all fourteen lattices. THE FLOOR TABLE IS WRITTEN
#      BEFORE THE ARRAY IS SUBMITTED, NEVER AFTER (PROTOCOL section 7).
#
# CAMPAIGN 1'S CALIBRATION MEASURED THE WRONG THING and it is easy to repeat: it gave each pool a
# single entry, so it measured model loading rather than throughput and the topologies it compared
# came out within 1 % of each other. Subtract the per-pool startup before projecting, which is why
# every arm below runs enough entries per pool for the startup to amortise.
#
# AND DO NOT COMPARE THE ARMS ON `Elapsed (wall clock)` OR ON `seconds_total`. The arms run in
# sequence inside one job, so the FIRST pays cold-cache model loading and the later ones read the
# weights from page cache. Job 57700621 measured startup at 163 s for the first arm against 48 and
# 77 s for the two that followed -- a 3x penalty with nothing to do with the topology. The
# cache-independent number is the per-pool `s/entry`, whose clock starts after
# `setup_mp_optimizers` returns; whole-node throughput is `n_pools / (s per entry)`. On 57700621
# that ordered the arms 64x2 (5.87 entries/s) > 32x4 (3.80) > 16x8 (3.18), while wall clock put
# 16x8 first. Throughput is what the array should be sized from: a 4 748-entry shard amortises
# startup over hours, so only the marginal rate matters.
#
# ================================================================================================
# RUNBOOK. docs/ is git-ignored, so `git pull` does NOT deliver the S07 handoff, STATUS.md, or the
# frozen split manifest. This header is the only copy of these instructions that reaches NERSC.
#
#   1. ON THE LAPTOP, push the record and the split manifest across:
#
#        docs/sync_record.sh push
#        docs/sync_record.sh checksum
#
#      It must print 3dd52c5eb2546dacca3034ebd2fd052dcd2acd4a8f9af24ce972fe4e0a210969. The driver
#      reads its ENTRY LIST from that file, not just the split, so a stale copy is a different
#      benchmark rather than a missing column. Confirm it arrived intact:
#
#        sha256sum /global/cfs/cdirs/m4064/dwmoreau/MLI/docs/fom_campaign2/artifacts/S06_split_manifest.parquet
#
#   2. Pull the code and check the commit matches the laptop's:
#
#        cd /global/cfs/cdirs/m4064/dwmoreau/MLI && git checkout fom_campaign2 && git pull
#        git log --oneline -1
#
#      NERSC cannot push to origin, so never commit here expecting it to travel.
#
#   3. sbatch submit_fom_dump_calibration.sh          <- this script
#   4. Read the summary it prints, set NPOOLS/POOLSIZE/walltime/--predownsample-entries in
#      submit_fom_dump.sh from it, and write the floor table with the measured reachability.
#   5. sbatch submit_fom_dump.sh
#
#   6. Record findings in docs/fom_campaign2/STATUS_nersc_inbox.md and NOTHING in STATUS.md. The
#      laptop copy is authoritative and `sync_record.sh push` overwrites this one; the inbox is
#      excluded from that push so it survives. Then, on the laptop: docs/sync_record.sh pull-inbox
# ================================================================================================

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J fom_dump_cal
#SBATCH -A lcls
#SBATCH -t 00:30:00
#SBATCH -o fom_dump_calibration_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

MANIFEST="$REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet"
OUTROOT="$SCRATCH/fom_campaign2/calibration"

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: frozen split manifest missing at $MANIFEST" >&2
    echo "Run 'docs/sync_record.sh push' on the laptop first -- step 1 of the runbook above." >&2
    exit 1
fi

# SLURM_CPUS_ON_NODE reports 256 on a Perlmutter CPU node, counting BOTH hyperthreads. The pools
# want the 128 physical cores; oversubscribing by 2x makes the run slower, not faster.
CPUS=${SLURM_CPUS_ON_NODE:-256}
PHYSICAL=$((CPUS / 2))
echo "SLURM_CPUS_ON_NODE=$CPUS -> $PHYSICAL physical cores (nproc says $(nproc), which is wrong "
echo "under salloc: the login shell inherits a narrow affinity mask and has reported 2)"

# Enough entries per pool that the ~111 s startup amortises rather than dominating. aP carries
# 3 000 manifest entries and the run takes shard 0 of NSHARDS of them, so the entry count is
# DERIVED from NSHARDS rather than the other way round. Computing NSHARDS from a target count
# and then reporting the target is how a projection gets built on a number the job did not run.
AP_ENTRIES=3000
NSHARDS=15
N_ENTRIES=$((AP_ENTRIES / NSHARDS))

# aP is the arm that matters: two thirds of every pattern's pool is aP, mP and mC (C2-F-052), and
# the pre-deduplication ratio has never been measured on any of them.
for TOPOLOGY in "32 4" "16 8" "64 2"; do
    read -r NPOOLS POOLSIZE <<< "$TOPOLOGY"
    if [ $((NPOOLS * POOLSIZE)) -ne "$PHYSICAL" ]; then
        echo "skipping ${NPOOLS}x${POOLSIZE}: does not fill $PHYSICAL physical cores"
        continue
    fi
    echo ""
    echo "=== ${NPOOLS} pools x ${POOLSIZE} processes, $N_ENTRIES aP entries ($((N_ENTRIES / NPOOLS)) per pool) ==="
    # NOT wrapped in srun. A bare `srun -n 1` pins CPU affinity to one core and strangles the 128
    # processes -- campaign 1's batch driver invokes the interpreter directly for this reason.
    /usr/bin/time -v $PYTHON run_fom_dump.py \
        --condition nominal \
        --split-manifest "$MANIFEST" \
        --bravais-lattices aP \
        --n-pools "$NPOOLS" --pool-size "$POOLSIZE" \
        --shard 0 --n-shards "$NSHARDS" \
        --predownsample-entries 2 \
        --out-dir "$OUTROOT/topology_${NPOOLS}x${POOLSIZE}" 2>&1 | \
        grep -E "s/entry|done:|labelled|subsampled|Elapsed|Maximum resident"
done

echo ""
echo "=== the three numbers this job exists to produce ==="
$PYTHON - <<'PY'
import glob, json, os
import pandas as pd

root = os.path.join(os.environ['SCRATCH'], 'fom_campaign2', 'calibration')
rows = []
for directory in sorted(glob.glob(os.path.join(root, 'topology_*'))):
    manifest_path = os.path.join(directory, 'manifest.json')
    if not os.path.exists(manifest_path):
        continue
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    candidates = pd.concat([pd.read_parquet(p) for p in
                            glob.glob(os.path.join(directory, 'candidates_*.parquet'))],
                           ignore_index=True)
    predownsample = [pd.read_parquet(p) for p in
                     glob.glob(os.path.join(directory, 'predownsample_*.parquet'))]
    predownsample = pd.concat(predownsample, ignore_index=True) if predownsample else None
    n_entries = manifest['n_entries']
    entries = pd.concat([pd.read_parquet(p) for p in
                         glob.glob(os.path.join(directory, 'entries_*.parquet'))],
                        ignore_index=True)
    # Survivors BEFORE subsampling, which is the number the disk projection is built on.
    survivors = int(entries['pool_size_full'].sum())
    reachable = entries['entry_id'].isin(
        candidates.loc[candidates['is_correct'], 'entry_id']).sum()
    row = {
        'topology': os.path.basename(directory).replace('topology_', ''),
        'n_entries': n_entries,
        'seconds_total': manifest['seconds_total'],
        's_per_entry': round(manifest['seconds_total'] / max(1, n_entries), 2),
        'survivors_per_pattern': round(survivors / max(1, len(entries)), 1),
        'retained_frac': round(len(candidates) / max(1, survivors), 4),
        'reachable_frac': round(float(reachable) / max(1, len(entries)), 4),
        }
    if predownsample is not None:
        # JOINED ON entry_id, not sliced. --predownsample-entries is per POOL, so the
        # pre-deduplication stream covers n_pools x that many entries while a `.head(2)` of the
        # entry table covers two -- which made the reported ratio scale with the pool count
        # (21.82 / 37.82 / 80.74 at 16 / 32 / 64 pools, i.e. the same number times n_pools).
        # Divide like by like: the entries that actually have both streams.
        per_entry = predownsample.groupby('entry_id').size()
        survivors = entries.set_index('entry_id')['pool_size_full']
        shared = per_entry.index.intersection(survivors.index)
        if len(shared):
            row['predownsample_ratio_aP'] = round(
                float(per_entry[shared].sum()) / max(1, int(survivors[shared].sum())), 2)
            row['predownsample_entries_measured'] = int(len(shared))
    rows.append(row)

table = pd.DataFrame(rows)
print(table.to_string(index=False))
if not table.empty:
    # `seconds_total` is confounded by cold-cache model loading on whichever arm ran first, so
    # the verdict is taken on THROUGHPUT instead -- see the header. Ranking on seconds_total
    # would tend to crown whichever arm ran last.
    print("\nNOTE: seconds_total and wall clock are NOT comparable across arms -- the first arm")
    print("pays cold-cache model loading. Ranking below is on per-pool throughput.")
    best = table.loc[table['entries_per_s'].idxmax()] if 'entries_per_s' in table \
        else table.loc[table['s_per_entry'].idxmin()]
    print(f"\nFASTEST TOPOLOGY: {best['topology']}")
    print(f"  NOTE this arm ran aP ONLY. A full-lattice cell is the sum over fourteen lattices;")
    print(f"  scale by the S06 pilot's ratio before projecting the array, do not use it directly.")
    print(f"\nPRE-DEDUPLICATION RATIO on aP: "
          f"{best.get('predownsample_ratio_aP', 'not measured')} "
          f"over {best.get('predownsample_entries_measured', 0)} entries.")
    print("  The 7.7x in the array sizing was measured on cP and tP, where deduplication\n"
          "  collapses almost everything (cF 118 -> 1.2). aP barely collapses at all, so the\n"
          "  ratio is near 1 there and the two numbers are not in conflict -- they are\n"
          "  different lattices. Size --predownsample-entries on THIS number, since aP, mP\n"
          "  and mC are 65 % of every pattern's pool (C2-F-052).")
    print(f"\naP REACHABILITY: {best['reachable_frac']:.3f}. Feed the per-lattice version of this")
    print( "  into `run_fom_dump_gate.py floor --reachability`, BEFORE submitting the array.")
PY

echo ""
echo "Next: set NPOOLS/POOLSIZE and the walltime in submit_fom_dump.sh, write the floor table,"
echo "then sbatch submit_fom_dump.sh. Record findings in STATUS_nersc_inbox.md, never STATUS.md."
