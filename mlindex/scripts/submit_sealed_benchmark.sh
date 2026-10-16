#!/bin/bash
# The sealed real-data benchmark: the CNRS/opXRD patterns searched as `mlindex.run --zero-error`
# and `mlindex.run_analytical --zero-error` search them, every printed list kept
# (`mlindex/scripts/run_real_benchmark.py`). From the checkout's root, environment active:
#
#   sbatch mlindex/scripts/submit_sealed_benchmark.sh
#
# Overrides: MLI_SEALED_DATA, MLI_SEALED_OUT, MLI_SEALED_PARTS, MLI_PYTHON.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J sealed_benchmark
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH -o sealed_benchmark_%j.out

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export MLINDEX_MODELS_DIR="${MLINDEX_MODELS_DIR:-$PWD/mlindex/models}"
MLI_SEALED_PYTHON="${MLI_PYTHON:-python}"
MLI_SEALED_DATA="${MLI_SEALED_DATA:-mlindex/data/opxrd/CNRS_output_data_verified_final3.json}"
MLI_SEALED_OUT="${MLI_SEALED_OUT:-${SCRATCH:-.}/fom_production/artifacts/P19_sealed/$(git rev-parse --short=7 HEAD)}"
MLI_SEALED_PARTS="${MLI_SEALED_PARTS:-$(( ${SLURM_CPUS_ON_NODE:-8} / 2 ))}"
mkdir -p "$MLI_SEALED_OUT"

for MLI_SEALED_INDEXER in ml analytical; do
    for (( MLI_SEALED_PART = 0; MLI_SEALED_PART < MLI_SEALED_PARTS; MLI_SEALED_PART++ )); do
        "$MLI_SEALED_PYTHON" -m mlindex.scripts.run_real_benchmark --patterns "$MLI_SEALED_DATA" \
            --indexer "$MLI_SEALED_INDEXER" --part "$MLI_SEALED_PART" --n-parts "$MLI_SEALED_PARTS" \
            --out-dir "$MLI_SEALED_OUT" > "$MLI_SEALED_OUT/log_${MLI_SEALED_INDEXER}_${MLI_SEALED_PART}.txt" 2>&1 &
    done
    wait
done
echo "DONE: $(find "$MLI_SEALED_OUT" -maxdepth 1 -name 'printed_*.parquet' | wc -l) of $(( 2 * MLI_SEALED_PARTS )) parts written to $MLI_SEALED_OUT"
