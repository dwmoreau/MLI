#!/bin/bash
# Reduce the generated arms to the run-to-run floor, WHERE THE ARMS ARE.
#
# `submit_benchmark_arms.sh` generates and scores; this turns the result into numbers. It is a
# separate job because the two have nothing in common: generation is node-hours over 128 processes,
# this is minutes over one, and a cluster should not be asked to reserve the former to do the
# latter.
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI      # on branch p04b-generation
#   export MLI_OUT=$SCRATCH/benchmark_p04b                     # where the arms were written
#   sbatch mlindex/scripts/submit_benchmark_floor.sh
#
#   MLI_CAMPAIGN=fom_production docs/sync_record.sh pull-artifacts   # from the laptop, afterwards
#
# WHY IT RUNS HERE AND NOT ON THE LAPTOP. The floor reads every candidate of every arm -- about
# 73 GB across the eight -- and returns an aggregate and fourteen per-lattice numbers, which is
# kilobytes. Moving the pool to the analysis machine moves 73 GB to produce a file that fits in an
# email. The record's standing rule is heavy reduction on the cluster, analysis on the laptop; the
# per-entry tables this writes are the small thing that travels.
#
# WHAT IT REFUSES. The driver checks that the arms of a floor differ only in their search seed --
# not in commit, machine, condition set, pool size, threshold or split. An arm missing its
# completion stamp is refused too: a killed run leaves valid shards behind and pairs silently.
# Neither check is overridable here, deliberately; --allow-incomplete exists for the campaign's
# older arms and has no business in a fresh measurement.
#
# Variable names are MLI_-prefixed because bash discards an assignment to one of its own built-ins.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J benchmark_floor
#SBATCH -A lcls
#SBATCH -t 2:00:00
#SBATCH -o benchmark_floor_%j.out

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${MLI_PYTHON:?set MLI_PYTHON to the interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the checkout to run}"
: "${MLI_OUT:?set MLI_OUT to the directory the arms were written to}"
MLI_RESULTS="${MLI_RESULTS:-$MLI_OUT/results}"
MLI_SEARCH_SEEDS=(${MLI_SEARCH_SEEDS:-12345 202 303 404})
MLI_SCORES="${MLI_SCORES:-M20,M_sym}"

cd "$MLI_REPO"
mkdir -p "$MLI_RESULTS"

for MLI_POPULATION in general hard; do
    MLI_ARM_ARGS=()
    MLI_MISSING=0
    for MLI_SEED in "${MLI_SEARCH_SEEDS[@]}"; do
        MLI_ARM="$MLI_OUT/${MLI_POPULATION}_seed${MLI_SEED}"
        if [ ! -f "$MLI_ARM/complete.json" ]; then
            echo "MISSING: $MLI_ARM has no completion stamp" >&2
            MLI_MISSING=1
            continue
        fi
        MLI_ARM_ARGS+=(--arm "seed${MLI_SEED}=$MLI_ARM")
    done
    if [ "$MLI_MISSING" -ne 0 ]; then
        echo "FATAL: $MLI_POPULATION is incomplete; a floor over the arms that did finish would" >&2
        echo "       be a floor for a different population. Re-run the missing array tasks." >&2
        exit 1
    fi

    echo "=== $MLI_POPULATION: per-entry outcomes, one table per arm ==="
    for MLI_SEED in "${MLI_SEARCH_SEEDS[@]}"; do
        "$MLI_PYTHON" -m mlindex.scripts.run_benchmark --stage reduce \
            --pool "$MLI_OUT/${MLI_POPULATION}_seed${MLI_SEED}" \
            --scores "$MLI_SCORES" \
            --out-dir "$MLI_RESULTS/${MLI_POPULATION}_per_entry"
    done

    echo "=== $MLI_POPULATION: the floor, aggregate and per lattice ==="
    "$MLI_PYTHON" -m mlindex.scripts.run_benchmark --stage floor \
        "${MLI_ARM_ARGS[@]}" \
        --scores "$MLI_SCORES" \
        --out-dir "$MLI_RESULTS/${MLI_POPULATION}_floor"
done

echo "results in $MLI_RESULTS -- pull these, not the pools"
du -sh "$MLI_RESULTS"
