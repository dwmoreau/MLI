#!/bin/bash
# The end-to-end table: the list mlindex.run prints, under the learned ranker, M_sym and M20, on
# both fom-dev populations, replayed from searched pools by `run_benchmark --stage end-to-end`.
#
# From the root of the checkout to run, with the environment that has mlindex installed active:
#
#   conda activate /global/cfs/cdirs/m4064/dwmoreau/envs/onnx
#   cd /global/cfs/cdirs/m4064/dwmoreau/MLI
#   sbatch mlindex/scripts/submit_end_to_end.sh
#
# That one command is the whole run. It prints DONE and the command that copies the tables to the
# laptop. Nothing needs exporting; these can be set to override the defaults:
#
#   MLI_E2E_POOLS   the pools, space-separated
#                   (default: the dev_general and dev_hard pools P18's fix2 arm searched,
#                   $SCRATCH/fom_production/artifacts/P18_arms/pools/fix2/{dev_general,dev_hard})
#   MLI_E2E_OUT     where the tables go
#                   (default: $SCRATCH/fom_production/artifacts/P18b_end_to_end/<commit>)
#   MLI_REPO, MLI_PYTHON    the checkout and the interpreter (default: here, and `python`)
#
# WHAT IS CHECKED, before anything is written: the checkout has no uncommitted changes to tracked
# files (the tables name its commit); each pool is complete, was searched at the indexer's cut,
# and with the same model files as the tree this run reads; the learned ranker is in that tree.
# Each pool's table records the search's commit and the replay's beside every number.
#
# OUTPUT, per pool: end_to_end_<pool>.csv (levels and paired contrasts with their SE, per
# lattice, lattice system and all lattices, with the ceiling and the tie-break floor),
# end_to_end_<pool>.png, and printed_<pool>.parquet (every printed row).
#
# REHEARSE on the laptop against any small pool, e.g. one generated at one crystal a lattice:
#
#   MLINDEX_MODELS_DIR=$PWD/mlindex/models MLI_E2E_POOLS=<pool> MLI_E2E_OUT=<dir> \
#       bash mlindex/scripts/submit_end_to_end.sh
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core. Variables are
# MLI_E2E_-prefixed so a setting exported by another submit script cannot reach this one.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J end_to_end
#SBATCH -A lcls
#SBATCH -t 2:00:00
#SBATCH -o end_to_end_%j.out

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MLI_E2E_REPO="${MLI_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
MLI_E2E_PYTHON="${MLI_PYTHON:-$(command -v python || true)}"
cd "$MLI_E2E_REPO"
if [ ! -f mlindex/scripts/run_benchmark.py ]; then
    echo "FATAL: $MLI_E2E_REPO is not an MLI checkout. Run sbatch from the repository root." >&2
    exit 1
fi
if [ -z "$MLI_E2E_PYTHON" ]; then
    echo "FATAL: no python. Activate the environment that has mlindex installed, or set MLI_PYTHON." >&2
    exit 1
fi
export MLINDEX_MODELS_DIR="${MLINDEX_MODELS_DIR:-$MLI_E2E_REPO/mlindex/models}"
if [ ! -d "$MLINDEX_MODELS_DIR/cubic_1/abnn" ] || [ ! -f "$MLINDEX_MODELS_DIR/ranker_1/specification.json" ]; then
    echo "FATAL: $MLINDEX_MODELS_DIR is not a model tree with ranker_1 (run git lfs pull)." >&2
    exit 1
fi

MLI_E2E_COMMIT=$(git rev-parse --short=7 HEAD)
if [ -z "${MLI_E2E_POOLS:-}" ]; then
    if [ -z "${SCRATCH:-}" ]; then
        echo "FATAL: SCRATCH is not set and MLI_E2E_POOLS is not given." >&2
        exit 1
    fi
    MLI_E2E_FIX2="$SCRATCH/fom_production/artifacts/P18_arms/pools/fix2"
    MLI_E2E_POOLS="$MLI_E2E_FIX2/dev_general $MLI_E2E_FIX2/dev_hard"
fi
MLI_E2E_OUT="${MLI_E2E_OUT:-${SCRATCH:-}/fom_production/artifacts/P18b_end_to_end/$MLI_E2E_COMMIT}"
read -r -a MLI_E2E_POOL_LIST <<< "$MLI_E2E_POOLS"
for MLI_E2E_POOL in "${MLI_E2E_POOL_LIST[@]}"; do
    if [ ! -f "$MLI_E2E_POOL/complete.json" ]; then
        echo "FATAL: $MLI_E2E_POOL is not a complete pool (no complete.json)." >&2
        exit 1
    fi
done
mkdir -p "$MLI_E2E_OUT"
echo "commit $MLI_E2E_COMMIT | models $MLINDEX_MODELS_DIR | python $MLI_E2E_PYTHON | out $MLI_E2E_OUT"

MLI_E2E_ARGS=()
for MLI_E2E_POOL in "${MLI_E2E_POOL_LIST[@]}"; do
    MLI_E2E_ARGS+=(--pool "$MLI_E2E_POOL")
done
# A bundle a process: ten bundles is the most any pool has.
"$MLI_E2E_PYTHON" -m mlindex.scripts.run_benchmark --stage end-to-end "${MLI_E2E_ARGS[@]}" \
    --n-pools 10 --out-dir "$MLI_E2E_OUT"

echo DONE
echo "On the laptop, copy the tables back with:"
echo "  rsync -avz dwmoreau@perlmutter.nersc.gov:$MLI_E2E_OUT/ docs/fom_production/artifacts/P18b_end_to_end/"
