#!/bin/bash
# Reduce a before arm and an after arm to the small tables a paired contrast needs, WHERE THEY ARE.
#
# `submit_benchmark_arms.sh` generates; this turns two arms into the per-entry tables that travel.
# Every session that changes the printed answer needs this pair -- P05, P10, P11, P16, P17 -- so it
# takes the two arm roots as variables rather than naming a session.
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI      # the branch under test
#   export MLI_BEFORE_OUT=$SCRATCH/benchmark_p04b             # where the baseline arms live
#   export MLI_AFTER_OUT=$SCRATCH/p05_abnn_arms               # where this session's arms live
#   sbatch mlindex/scripts/submit_benchmark_contrast.sh
#
#   docs/sync_record.sh pull-artifacts 'P05_arms'             # from the laptop, afterwards
#
# WHY BOTH SIDES ARE RE-REDUCED HERE. The baseline arm may already have per-entry tables on the
# laptop, and they are the wrong ones: tables reduced before the manifest and the true lattice
# travelled with them carry neither, so the contrast refuses the pairing and the per-lattice
# breakdown comes out empty. Reducing both arms with the same code in the same job is what makes
# the two sides comparable, and it costs minutes.
#
# WHY THE TWO ARMS GO TO DIFFERENT DIRECTORIES. A reduced table is named for its pool directory,
# and both arms of a contrast are the same population at the same search seed -- so both are
# `general_seed12345` and they would overwrite each other in one output directory.
#
# WHAT IT REFUSES. Any of the four pools missing its completion stamp stops the job before
# anything is written. A killed generation leaves valid shards behind, and reducing those gives a
# table that pairs silently and is a table for a different population.
#
# THE CONTRAST ITSELF IS NOT RUN HERE. It is arithmetic over a few thousand rows and belongs on
# the analysis machine with the floor beside it; the command is printed at the end.
#
# Variable names are MLI_-prefixed because bash discards an assignment to one of its own built-ins.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J benchmark_contrast
#SBATCH -A lcls
#SBATCH -t 2:00:00
#SBATCH -o benchmark_contrast_%j.out

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${MLI_PYTHON:?set MLI_PYTHON to the interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the checkout to run}"
: "${MLI_BEFORE_OUT:?set MLI_BEFORE_OUT to the directory the baseline arms were written to}"
# No apostrophes in a :? message: bash reads one as an opening quote even inside double
# quotes, and the script then fails to parse rather than to run.
: "${MLI_AFTER_OUT:?set MLI_AFTER_OUT to the directory the new arms were written to}"

# Written where `docs/sync_record.sh pull-artifacts` already looks, so pulling needs no override.
MLI_RESULTS="${MLI_RESULTS:-$SCRATCH/fom_production/artifacts/P05_arms}"
MLI_SEED="${MLI_SEED:-12345}"
MLI_POPULATIONS=(${MLI_POPULATIONS:-general hard})
MLI_SCORES="${MLI_SCORES:-M20,M_sym}"
# Taken once: nesting a command substitution inside a quoted echo below is legal in bash 5 and
# an unterminated string in the bash 3.2 that ships with macOS, where these get syntax-checked.
MLI_RESULTS_NAME=$(basename "$MLI_RESULTS")

cd "$MLI_REPO"
mkdir -p "$MLI_RESULTS"

# Refuse before writing anything, rather than half way down the list.
MLI_MISSING=0
for MLI_POPULATION in "${MLI_POPULATIONS[@]}"; do
    for MLI_SIDE in before after; do
        if [ "$MLI_SIDE" = before ]; then MLI_ROOT="$MLI_BEFORE_OUT"; else MLI_ROOT="$MLI_AFTER_OUT"; fi
        MLI_ARM="$MLI_ROOT/${MLI_POPULATION}_seed${MLI_SEED}"
        if [ ! -f "$MLI_ARM/complete.json" ]; then
            echo "MISSING: $MLI_ARM has no completion stamp" >&2
            MLI_MISSING=1
        fi
    done
done
if [ "$MLI_MISSING" -ne 0 ]; then
    echo "FATAL: a contrast over arms that did not finish is a contrast between different" >&2
    echo "       populations. Re-run the missing array tasks and submit this again." >&2
    exit 1
fi

for MLI_POPULATION in "${MLI_POPULATIONS[@]}"; do
    for MLI_SIDE in before after; do
        if [ "$MLI_SIDE" = before ]; then MLI_ROOT="$MLI_BEFORE_OUT"; else MLI_ROOT="$MLI_AFTER_OUT"; fi
        MLI_ARM="$MLI_ROOT/${MLI_POPULATION}_seed${MLI_SEED}"
        echo "=== $MLI_POPULATION $MLI_SIDE: per-entry outcomes from $MLI_ARM ==="
        "$MLI_PYTHON" -m mlindex.scripts.run_benchmark --stage reduce \
            --pool "$MLI_ARM" \
            --scores "$MLI_SCORES" \
            --out-dir "$MLI_RESULTS/${MLI_POPULATION}_${MLI_SIDE}"
    done
done

echo
echo "Reduced tables are in $MLI_RESULTS. Pull them and read the contrast on the laptop:"
echo
echo "  docs/sync_record.sh pull-artifacts '$MLI_RESULTS_NAME'"
echo
for MLI_POPULATION in "${MLI_POPULATIONS[@]}"; do
    MLI_PREFIX="${MLI_POPULATION}_seed${MLI_SEED}"
    echo "  python -m mlindex.scripts.run_benchmark --stage contrast \\"
    echo "      --arm before=docs/fom_production/artifacts/$MLI_RESULTS_NAME/${MLI_POPULATION}_before/${MLI_PREFIX} \\"
    echo "      --arm after=docs/fom_production/artifacts/$MLI_RESULTS_NAME/${MLI_POPULATION}_after/${MLI_PREFIX} \\"
    echo "      --reference before --vary commit --scores $MLI_SCORES \\"
    echo "      --floor-table docs/fom_production/artifacts/P04b_arms/${MLI_POPULATION}_floor/floor.csv \\"
    echo "      --out-dir results/p05_${MLI_POPULATION}"
    echo
done
