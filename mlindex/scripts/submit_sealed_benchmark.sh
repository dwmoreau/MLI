#!/bin/bash
# The sealed real-data benchmark, run once: the CNRS/opXRD patterns searched as
# `mlindex.run --zero-error` and `mlindex.run_analytical --zero-error` search them, every printed
# list kept (`mlindex/scripts/run_real_benchmark.py`). From the checkout's root, environment active:
#
#   sbatch mlindex/scripts/submit_sealed_benchmark.sh
#
# Refuses a data file or a model tree without the pre-registered digest, and an output directory
# that exists. Overrides: MLI_SEALED_DATA, MLI_SEALED_SHA256, MLI_SEALED_MODELS_DIGEST,
# MLI_SEALED_OUT, MLI_SEALED_PARTS, MLI_PYTHON.
# Not wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J sealed_benchmark
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH -o sealed_benchmark_%j.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
MLI_SEALED_PYTHON="${MLI_PYTHON:-python}"
export MLINDEX_MODELS_DIR="${MLINDEX_MODELS_DIR:-$PWD/mlindex/models}"
MLI_SEALED_DATA="${MLI_SEALED_DATA:-mlindex/data/opxrd/CNRS_output_data_verified_final3.json}"
MLI_SEALED_SHA256="${MLI_SEALED_SHA256:-30b5e253aa08d1b59b4e7dbb6d568329397484bc593cb2a6a402a293416ff6c4}"
# The model tree and ranker of P18b's table (hub v3), as pre-registered.
MLI_SEALED_MODELS_DIGEST="${MLI_SEALED_MODELS_DIGEST:-a6cec7d0e0cf1219af9e186e1f6e516abbaaeb3ce45c3aa3c75493b7dadd304d}"
MLI_SEALED_OUT="${MLI_SEALED_OUT:-${SCRATCH:-}/fom_production/artifacts/P19_sealed/$(git rev-parse --short=7 HEAD)}"
MLI_SEALED_PARTS="${MLI_SEALED_PARTS:-$(( ${SLURM_CPUS_ON_NODE:-8} / 2 ))}"

MLI_SEALED_FOUND=$("$MLI_SEALED_PYTHON" -c "import sys; from mlindex.utilities.Digests import file_digest; print(file_digest(sys.argv[1]))" "$MLI_SEALED_DATA")
if [ "$MLI_SEALED_FOUND" != "$MLI_SEALED_SHA256" ]; then
    echo "FATAL: $MLI_SEALED_DATA has sha256 $MLI_SEALED_FOUND, not $MLI_SEALED_SHA256." >&2
    exit 1
fi
MLI_SEALED_MODELS_FOUND=$("$MLI_SEALED_PYTHON" -c "from mlindex.model_training.BenchmarkPrinted import environment; print(environment()['models_digest'])")
if [ "$MLI_SEALED_MODELS_FOUND" != "$MLI_SEALED_MODELS_DIGEST" ]; then
    echo "FATAL: the model tree $MLINDEX_MODELS_DIR has digest $MLI_SEALED_MODELS_FOUND, not $MLI_SEALED_MODELS_DIGEST." >&2
    exit 1
fi
if [ -e "$MLI_SEALED_OUT" ]; then
    echo "FATAL: $MLI_SEALED_OUT exists. The sealed benchmark is run once." >&2
    exit 1
fi
mkdir -p "$MLI_SEALED_OUT"

for MLI_SEALED_INDEXER in ml analytical; do
    for (( MLI_SEALED_PART = 0; MLI_SEALED_PART < MLI_SEALED_PARTS; MLI_SEALED_PART++ )); do
        "$MLI_SEALED_PYTHON" -m mlindex.scripts.run_real_benchmark --patterns "$MLI_SEALED_DATA" \
            --indexer "$MLI_SEALED_INDEXER" --part "$MLI_SEALED_PART" --n-parts "$MLI_SEALED_PARTS" \
            --out-dir "$MLI_SEALED_OUT" > "$MLI_SEALED_OUT/log_${MLI_SEALED_INDEXER}_${MLI_SEALED_PART}.txt" 2>&1 &
    done
    wait
done
MLI_SEALED_WRITTEN=$(ls "$MLI_SEALED_OUT"/printed_*.parquet 2>/dev/null | wc -l)
if [ "$MLI_SEALED_WRITTEN" -ne $(( 2 * MLI_SEALED_PARTS )) ]; then
    echo "FATAL: $MLI_SEALED_WRITTEN of $(( 2 * MLI_SEALED_PARTS )) parts wrote; see $MLI_SEALED_OUT/log_*.txt" >&2
    exit 1
fi
echo DONE
echo "rsync -avz dwmoreau@perlmutter.nersc.gov:$MLI_SEALED_OUT/ docs/fom_production/artifacts/P19_sealed/"
