#!/bin/bash
# S09 prerequisite -- recompute the reduced merit set for Benchmark B and store it beside the pool.
#
#   sbatch submit_fom_zoo_merits.sh
#
# WHY THIS EXISTS AS ITS OWN JOB. `SCHEMA.md` stores `M20` and `Minfo` on a candidate row and no
# more: the other six of the reduced core -- `M_tilde`, `M_rev`, `M_sym`, `X_N`, `n_over`,
# `max_gap` -- are recomputable from `xnn`, the peak list and the extinction group, and by the
# schema's own rule a recomputable column does not earn storage in a 122 GB pool. That is the right
# call. It is the wrong call to act on repeatedly: measured on the floor arms, the recompute runs at
# **136 microseconds a candidate**, so Benchmark B's 880 704 233 candidates is **33 core-hours**.
#
# S09, S10, S11 and S12 all need those six columns. Recomputing them once per session is precisely
# what PROTOCOL section 3 rule 8 forbids, and campaign 1 lost this four separate times: a basin
# count, a generator provenance column, a posterior's own denominator, and correctness labels on a
# 57-million-row dump that every later analysis pass then recomputed.
#
# COST: 33 core-hours. WALL CLOCK is NOT 33/128 hours, because the unit of parallelism is the file
# and the files are wildly uneven -- aP, mP and mC are two thirds of every pattern's pool, so a
# single aP shard is ~24 M candidates and takes ~55 minutes on its own core. Expect **~1 hour**
# bounded by that file, whatever the process count. Walltime is 3 h against it.
#
# DISK: ~26 GB of sidecars, at 30 bytes a candidate. Check headroom first:  df -h $SCRATCH
#
# SIDECARS, NOT REWRITTEN CANDIDATE FILES. The merits go to <pool>/merits/<same filename> carrying
# the four join keys and the six columns. Rewriting the pool would duplicate 122 GB, and a pool file
# that differs from the one the generation array wrote can no longer be checksummed against it.
#
# RESUME: the script skips any file whose sidecar already exists, so a requeue continues rather than
# restarting. Pass --overwrite to force.
#
# MEMORY: each worker holds --chunk-rows candidate rows at a time, not a whole file. At 1 M rows
# and 64 workers that is roughly 100 GB of the node's 476 GB. Do not raise both together.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the workers.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_zoo_merits
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 3:00:00
#SBATCH -o fom_zoo_merits_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO/mlindex/scripts" || exit 1

POOL=${POOL:-$SCRATCH/fom_campaign2/pool}
PROCESSES=${PROCESSES:-64}
CHUNK_ROWS=${CHUNK_ROWS:-1000000}

if [ ! -d "$POOL" ]; then
    echo "FATAL: pool not found at $POOL" >&2
    exit 1
fi

echo "=== reduced merits for $POOL ==="
echo "    $PROCESSES processes, $CHUNK_ROWS rows a chunk, sidecars -> $POOL/merits"
df -h "$SCRATCH" | tail -1

"$PYTHON" run_fom_floor_merits.py \
    --pool "$POOL" \
    --processes "$PROCESSES" \
    --chunk-rows "$CHUNK_ROWS"

# Afterwards, a cheap check that every candidate file got a sidecar of the same length:
#
#   $PYTHON - <<'PY'
#   import glob, os, pyarrow.parquet as pq
#   pool = os.environ['SCRATCH'] + '/fom_campaign2/pool'
#   bad = []
#   for f in sorted(glob.glob(pool + '/candidates*.parquet')):
#       s = os.path.join(pool, 'merits', os.path.basename(f))
#       if not os.path.exists(s):
#           bad.append((os.path.basename(f), 'MISSING')); continue
#       n, m = pq.ParquetFile(f).metadata.num_rows, pq.ParquetFile(s).metadata.num_rows
#       if n != m:
#           bad.append((os.path.basename(f), f'{m} of {n}'))
#   print('all sidecars complete' if not bad else bad)
#   PY
