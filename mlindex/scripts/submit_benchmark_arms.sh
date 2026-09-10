#!/bin/bash
# Generate the benchmark arms that measure the run-to-run floor.
#
# Four arms per population, differing ONLY in the search seed. The spread between them is the
# noise floor every later gate is read against; a gate is quoted in multiples of it and never in
# percentage points.
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
#   export MLI_SPLIT_MANIFEST=$MLI_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet
#   export MLI_OUT=$SCRATCH/benchmark_p04b
#   sbatch mlindex/scripts/submit_benchmark_arms.sh
#
# Then, on the laptop, read the floor off the four arms of one population:
#
#   python -m mlindex.scripts.run_benchmark --stage floor \
#       --arm seed12345=$MLI_OUT/general_seed12345 \
#       --arm seed202=$MLI_OUT/general_seed202 \
#       --arm seed303=$MLI_OUT/general_seed303 \
#       --arm seed404=$MLI_OUT/general_seed404 \
#       --scores M20,M_sym --out-dir results/floor_general
#
# WHAT MUST NOT MOVE. Every arm shares --seed, which fixes the crystals drawn and the noise put on
# their peaks, and therefore the peak lists themselves. Only --search-seed moves, and it reaches
# the search alone. If --seed moved too the arms would differ in their DATA, and the spread would
# be generation noise and search noise together -- which is the one distinction a floor is made of.
# The reduction refuses to pair arms that disagree on anything else; see Benchmark.manifest_identity.
#
# POOL SIZE IS PART OF AN ARM'S IDENTITY. Above one process per pool a Bravais lattice is split
# across processes and the answer depends on how many, by decision. These run at 1: highest
# throughput, nothing split, and the search key is then (peak list, lattice, seed) alone, so any
# subset of an arm reproduces on its own. --n-pools is free -- it divides the crystals and nothing
# else -- and is the memory-limited axis, since every pool holds its own copy of the models.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core and strangles the pools.
# Read SLURM_CPUS_ON_NODE, not nproc, and halve it -- it counts both hyperthreads.
#
# MEMORY. One pool manager holding all fourteen lattices measures 0.8-1.0 GB resident, so 128 of
# them fit a 512 GB node with room to spare. Do not halve MLI_POOLS out of caution: the models are
# smaller than the ~3 GB this project's notes used to quote, and the pools are what the throughput
# is made of.
#
# WALLTIME. A pattern costs 59 s of one core on the development laptop, measured. A node core runs
# it in at most ~1.5x that -- derived from the campaign's own arm, 1 590 patterns over 64 pools in
# 2 124 s, which is 85.5 s a pattern at --pool-size 2 -- so ~90 s is the pessimistic figure. An arm
# of 1 590 over 128 pools is then ~20 min, and a hard arm of 1 800 is ~22 min. Four hours is
# deliberately generous: a job killed at the limit leaves an unstamped arm and the work is lost.
#
# Variable names are MLI_-prefixed because bash silently discards an assignment to one of its own
# built-ins, and GROUPS cost this project a 62 core-hour pass.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J benchmark_arms
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH --array=0-7
#SBATCH -o benchmark_arms_%A_%a.out

set -euo pipefail

# One thread per process. n_pools already fills the node, so a BLAS that spawns its own threads
# oversubscribes it by an order of magnitude and the run gets slower the more cores it is given.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${MLI_PYTHON:?set MLI_PYTHON to the interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the checkout to run}"
: "${MLI_SPLIT_MANIFEST:?set MLI_SPLIT_MANIFEST to the frozen split manifest}"
MLI_OUT="${MLI_OUT:-$SCRATCH/benchmark_arms}"

# Shared by every arm. Moving this moves the peak lists and the floor stops being a floor.
MLI_SEED="${MLI_SEED:-12345}"
MLI_SEARCH_SEEDS=(${MLI_SEARCH_SEEDS:-12345 202 303 404})
MLI_CUT="${MLI_CUT:-1.5}"

# The general floor is measured on the error-severity axis with no contamination; the hard
# population is all five of its own bundles, because those five are what "hard" means.
MLI_GENERAL_BUNDLES="${MLI_GENERAL_BUNDLES:-b1_error0.5_cont0,b1_error1_cont0,b1_error2_cont0}"
MLI_GENERAL_PER_LATTICE="${MLI_GENERAL_PER_LATTICE:-40}"
MLI_HARD_PER_LATTICE="${MLI_HARD_PER_LATTICE:-120}"

if [ ! -f "$MLI_SPLIT_MANIFEST" ]; then
    echo "FATAL: no split manifest at $MLI_SPLIT_MANIFEST" >&2
    exit 1
fi

# Half of SLURM_CPUS_ON_NODE, because it counts both hyperthreads, and one pool per physical core
# at --pool-size 1. Capped by memory: each pool loads its own models.
MLI_CORES="${SLURM_CPUS_ON_NODE:-8}"
MLI_POOLS="${MLI_POOLS:-$((MLI_CORES / 2))}"

MLI_TASK="${SLURM_ARRAY_TASK_ID:-0}"
MLI_POPULATIONS=(general hard)
MLI_POPULATION="${MLI_POPULATIONS[$((MLI_TASK / 4))]}"
MLI_SEARCH_SEED="${MLI_SEARCH_SEEDS[$((MLI_TASK % 4))]}"

if [ "$MLI_POPULATION" = "general" ]; then
    MLI_BUNDLE_ARGS=(--bundles "$MLI_GENERAL_BUNDLES")
    MLI_PER_LATTICE="$MLI_GENERAL_PER_LATTICE"
else
    MLI_BUNDLE_ARGS=()
    MLI_PER_LATTICE="$MLI_HARD_PER_LATTICE"
fi

MLI_ARM="$MLI_OUT/${MLI_POPULATION}_seed${MLI_SEARCH_SEED}"
echo "arm $MLI_ARM | pools $MLI_POOLS | per-lattice $MLI_PER_LATTICE | cut $MLI_CUT"

cd "$MLI_REPO"
"$MLI_PYTHON" -m mlindex.scripts.run_benchmark --stage generate \
    --out-pool "$MLI_ARM" \
    --split-manifest "$MLI_SPLIT_MANIFEST" \
    --population "$MLI_POPULATION" \
    --per-lattice "$MLI_PER_LATTICE" \
    "${MLI_BUNDLE_ARGS[@]}" \
    --cut "$MLI_CUT" \
    --seed "$MLI_SEED" \
    --search-seed "$MLI_SEARCH_SEED" \
    --n-pools "$MLI_POOLS" \
    --pool-size 1

# Merits are computed where the pool is: the shards stay on the cluster and only the reduced
# per-entry tables come back.
"$MLI_PYTHON" -m mlindex.scripts.run_benchmark --stage sidecars --pool "$MLI_ARM"

echo "done $MLI_ARM"
