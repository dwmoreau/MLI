#!/bin/bash
# S02 calibration -- RUN THIS FIRST, before submitting submit_fom_mirror.sh.
#
# Works either as a batch job or inside an interactive allocation:
#
#   sbatch submit_fom_mirror_calibration.sh
#
#   salloc -N 1 -C cpu -q interactive -t 1:00:00 -A lcls
#   cd /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/scripts
#   bash submit_fom_mirror_calibration.sh 2>&1 | tee ~/fom_cal.log
#
# It calls the environment's python by absolute path rather than `module load conda; conda
# activate`, because `conda activate` is unreliable in a non-interactive shell and the absolute path
# is what PROTOCOL section 6 documents anyway. Do NOT wrap the python calls in `srun`: a bare
# `srun -n 1` pins the task to one core's worth of CPU affinity and would strangle the 128
# processes. The batch driver does not use srun either.
#
# It answers the three things that cannot be measured from a login node:
#   1. Throughput per node, which is what the grid's walltime is sized against.
#   2. Which pool topology wins. Inside a pool only the optimisation iterations distribute --
#      candidate generation runs on the manager -- so whether 16x8 or 32x4 is faster depends on how
#      manager-bound the pools are. Keep n_pools * pool_size at the core count either way.
#   3. How much run.py's _conventional_cell step (F-039) changes the pooled ranking, and what it
#      costs. It postdates the published 450/599, so the grid leaves it off; this measures what that
#      choice is worth. Compare its ceiling and operating point against the 16x8 run.
#
# READ TWO DIFFERENT NUMBERS OUT OF THE OUTPUT, they mean different things:
#   * per-pool `seconds_per_entry` in summary_*_pool*.json -- steady-state cost per entry. The pool
#     clock starts after its optimizer set is built, so this EXCLUDES model loading. Extrapolate the
#     grid from this one: 5955 entries per bundle / n_pools * seconds_per_entry.
#   * the final "shard ... finished ... s/entry aggregate" line -- wall-clock including the ~3 GB of
#     model loading each of the 16 managers does at startup. On a short run that startup dominates,
#     which is why NPERBL below is 16 and not 4: 16 x 14 = 224 entries is 14 per pool, enough for
#     the steady-state figure to mean something.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J fom_mirror_cal
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 0:30:00

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
cd /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/scripts || exit 1

NPERBL=16

echo "host $(hostname)  nproc $(nproc)  SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"
# Check SLURM's count, not nproc. Under salloc the shell inherits a narrow CPU affinity mask -- an
# interactive allocation on a whole node reported `nproc 2` against SLURM_CPUS_ON_NODE=256 -- so
# nproc fires a spurious warning on a perfectly good node, and a warning nobody can trust is worse
# than none. The spawned processes are not bound by the login shell's mask.
CORES=${SLURM_CPUS_ON_NODE:-$(nproc)}
if [ "$CORES" -lt 128 ]; then
    echo "WARNING: SLURM reports only $CORES cores. The topologies below assume a whole CPU node;"
    echo "         timings from a partial node will not extrapolate to the grid."
fi

echo "=================== 16 pools x 8 processes ==================="
$PYTHON run_fom_mirror.py --n-pools 16 --pool-size 8 \
    --error-multiplier 1 --n-contaminants 1 --n-entries-per-bl $NPERBL \
    --out-dir ../characterization/fom/mirror_cal_16x8

echo "=================== 32 pools x 4 processes ==================="
$PYTHON run_fom_mirror.py --n-pools 32 --pool-size 4 \
    --error-multiplier 1 --n-contaminants 1 --n-entries-per-bl $NPERBL \
    --out-dir ../characterization/fom/mirror_cal_32x4

echo "============ 16x8 with _conventional_cell, for F-039 ============"
$PYTHON run_fom_mirror.py --n-pools 16 --pool-size 8 --conventional-cell \
    --error-multiplier 1 --n-contaminants 1 --n-entries-per-bl $NPERBL \
    --out-dir ../characterization/fom/mirror_cal_conventional

echo "=================== summary ==================="
for DIR in mirror_cal_16x8 mirror_cal_32x4 mirror_cal_conventional; do
    $PYTHON - "$DIR" <<'PY'
import json, statistics, sys
from pathlib import Path
name = sys.argv[1]
paths = sorted(Path('../characterization/fom') .joinpath(name).glob('summary_*_pool*.json'))
if not paths:
    print(f'{name}: no output'); raise SystemExit
runs = [json.load(open(p)) for p in paths]
per_entry = [r['seconds_per_entry'] for r in runs if r['n_entries_run']]
entries = sum(r['n_entries_run'] for r in runs)
failures = sum(r['n_failures'] for r in runs)
lost = sum(r['correct_candidates_lost_to_dedup'] for r in runs)
aborted = [r['pool'] for r in runs if r['aborted']]
n_pools, pool_size = runs[0]['n_pools'], runs[0]['pool_size']
print(f'{name}: {n_pools}x{pool_size}  {entries} entries  '
      f'steady-state {statistics.median(per_entry):.1f} s/entry/pool  '
      f'-> projected {5955 / n_pools * statistics.median(per_entry) / 3600:.2f} h per bundle  '
      f'failures {failures}  correct-lost-to-dedup {lost}'
      + (f'  ABORTED pools {aborted}' if aborted else ''))
PY
done
