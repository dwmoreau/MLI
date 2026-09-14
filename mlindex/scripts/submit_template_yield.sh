#!/bin/bash
# Screen the Miller-index template ranker's input sets on NERSC: train, then measure yield.
#
# RESEARCH CODE THAT NEEDS TO BE DELETED at P07. It runs P06's one-off comparison of input sets and
# is removed when P07 ships the retrained template models.
#
# One script, three uses, chosen with MLI_STAGE. Each array task is one (fit seed, search seed) pair
# from MLI_SEEDS; task 0 is the first screen, tasks 1 and 2 are the floor re-runs.
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
#   export MLI_SPLIT_MANIFEST=$MLI_REPO/docs/fom_campaign2/artifacts/S06_split_manifest.parquet
#   export MLI_OUT=$SCRATCH/template_yield_p06
#
#   # 1. the screen: every input set, seed pair 0 -- train, then measure on the select crystals
#   MLI_STAGE=train sbatch --array=0 -t 2:00:00 mlindex/scripts/submit_template_yield.sh
#   MLI_STAGE=yield sbatch --array=0 --dependency=afterok:<train job id> \
#       mlindex/scripts/submit_template_yield.sh
#
#   # 2. the floor: rho and the shortlist only, seed pairs 1 and 2
#   MLI_INPUT_SETS=rho,<shortlisted sets> MLI_STAGE=train sbatch --array=1-2 -t 2:00:00 \
#       mlindex/scripts/submit_template_yield.sh
#   MLI_INPUT_SETS=rho,<shortlisted sets> MLI_STAGE=yield sbatch --array=1-2 \
#       --dependency=afterok:<train job id> mlindex/scripts/submit_template_yield.sh
#
#   # 3. the confirmation: the chosen sets on fom-dev, reusing seed pair 0's trained rankers
#   MLI_INPUT_SETS=rho,<chosen sets> MLI_GROUP=report MLI_STAGE=yield sbatch --array=0 \
#       mlindex/scripts/submit_template_yield.sh
#
# Every run writes WORK/<population>_<group>/per_entry.parquet and manifest.json under
# $MLI_OUT/seed<fit>_<search>/. Only those come back; the trained rankers stay here. On the laptop:
#
#   rsync -a --include='*/' --include='per_entry.parquet' --include='manifest.json' \
#       --include='training_manifest.json' --exclude='*' \
#       perlmutter:$SCRATCH/template_yield_p06/ template_yield_p06/
#   python -m mlindex.scripts.run_template_yield --stage report \
#       --runs template_yield_p06/seed12345_12345/hard_select --out-dir template_yield_p06/report_hard
#
# WHAT MUST NOT MOVE between seed pairs: MLI_SEED, which fixes the crystals drawn, their fit/select
# group and the noise on their peaks. Only the fit seed and the search seed move, so the spread
# between pairs is how much a ranker's measured yield changes with training noise and search noise,
# on identical patterns. The report refuses runs that differ in anything else.
#
# PARALLELISM. A job is one input set on one Bravais lattice, run serially inside, and the driver
# runs up to MLI_PROCESSES of them at once, each held to one thread. The screen has 7 x 14 training
# jobs and 8 x 14 yield jobs, fewer than a node's cores, so the largest single job sets the wall
# time: aP, about 9 s a pattern on the laptop.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core. Read SLURM_CPUS_ON_NODE,
# not nproc, and halve it -- it counts both hyperthreads.
#
# Variable names are MLI_-prefixed because bash silently discards an assignment to one of its own
# built-ins, and GROUPS cost this project a 62 core-hour pass.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J template_yield
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH -o template_yield_%A_%a.out

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Nothing on this path imports keras, but the training environment defaults it to tensorflow,
# which is not installed.
export KERAS_BACKEND=torch

: "${MLI_PYTHON:?set MLI_PYTHON to the training interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the checkout to run}"
: "${MLI_SPLIT_MANIFEST:?set MLI_SPLIT_MANIFEST to the frozen split manifest}"
: "${MLI_STAGE:?set MLI_STAGE to train or yield}"
MLI_OUT="${MLI_OUT:-$SCRATCH/template_yield_p06}"
MLI_MODELS_DIR="${MLI_MODELS_DIR:-$MLI_REPO/mlindex/models}"
MLI_DATASETS="${MLI_DATASETS:-$MLI_REPO/mlindex/data/generated_datasets}"

MLI_INPUT_SETS="${MLI_INPUT_SETS:-rho,rho_sigma,posterior_sigma,merits,merits_structure,merits_structure_context,scalars_only}"
MLI_LATTICES="${MLI_LATTICES:-aP,cF,cI,cP,hP,hR,mC,mP,oC,oF,oI,oP,tI,tP}"
MLI_PER_LATTICE="${MLI_PER_LATTICE:-100}"
MLI_GROUP="${MLI_GROUP:-select}"
MLI_SEED="${MLI_SEED:-12345}"
# Space-separated fit:search pairs; the array task picks one.
MLI_SEEDS=(${MLI_SEEDS:-12345:12345 202:202 303:303})
# Empty means each population's own bundles: all ten for general, the five severe ones for hard.
MLI_GENERAL_BUNDLES="${MLI_GENERAL_BUNDLES:-}"
MLI_HARD_BUNDLES="${MLI_HARD_BUNDLES:-}"
# The hard population's lattices. Each must be among MLI_LATTICES, which is what was trained.
MLI_HARD_LATTICES="${MLI_HARD_LATTICES:-aP,mC,mP}"
MLI_CUT="${MLI_CUT:-1.5}"
# The commit the rankers and the yield stage need: seeded training and the parallel thread limit.
MLI_REQUIRED_COMMIT="${MLI_REQUIRED_COMMIT:-fc185a4}"

MLI_TASK="${SLURM_ARRAY_TASK_ID:-0}"
MLI_PAIR="${MLI_SEEDS[$MLI_TASK]}"
MLI_FIT_SEED="${MLI_PAIR%%:*}"
MLI_SEARCH_SEED="${MLI_PAIR##*:}"
MLI_WORK="$MLI_OUT/seed${MLI_FIT_SEED}_${MLI_SEARCH_SEED}"
MLI_CORES="${SLURM_CPUS_ON_NODE:-8}"
MLI_PROCESSES="${MLI_PROCESSES:-$((MLI_CORES / 2))}"

cd "$MLI_REPO"

# ---- pre-flight: refuse before the node is spent -------------------------------------------------
fail() { echo "FATAL: $*" >&2; exit 1; }

[ -f "$MLI_SPLIT_MANIFEST" ] || fail "no split manifest at $MLI_SPLIT_MANIFEST"
[ -d "$MLI_MODELS_DIR/cubic_1/abnn" ] || fail "$MLI_MODELS_DIR is not a full model tree"
# Ask whether the commit exists before asking whether it is an ancestor: a missing object also
# makes --is-ancestor fail, and that would read as "the checkout is too old".
git cat-file -e "${MLI_REQUIRED_COMMIT}^{commit}" 2>/dev/null \
    || fail "commit $MLI_REQUIRED_COMMIT is not in this checkout; fetch the p06 branch"
git merge-base --is-ancestor "$MLI_REQUIRED_COMMIT" HEAD \
    || fail "HEAD $(git rev-parse --short HEAD) does not contain $MLI_REQUIRED_COMMIT"
for lattice in ${MLI_LATTICES//,/ }; do
    [ -f "$MLI_DATASETS/dataset_${lattice}.parquet" ] || fail "no dataset_${lattice}.parquet in $MLI_DATASETS"
done
if [ "$MLI_STAGE" = "yield" ]; then
    for lattice in ${MLI_HARD_LATTICES//,/ }; do
        [[ ",$MLI_LATTICES," == *",$lattice,"* ]] \
            || fail "hard lattice $lattice is not in MLI_LATTICES ($MLI_LATTICES)"
    done
    for set in ${MLI_INPUT_SETS//,/ }; do
        [ -f "$MLI_WORK/models/$set/training_manifest.json" ] \
            || fail "no trained $set rankers in $MLI_WORK; run MLI_STAGE=train for task $MLI_TASK first"
    done
fi

# The versions that decide whether a ranker can be saved, and a round trip that proves it can.
"$MLI_PYTHON" - <<'EOF' || fail "this interpreter cannot export a gradient-boosted model to ONNX"
import tempfile
import google.protobuf, numpy, onnx, skl2onnx, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from mlindex.utilities.IOManagers import SKLearnManager
print(f"scikit-learn {sklearn.__version__}, skl2onnx {skl2onnx.__version__}, onnx {onnx.__version__}, "
      f"protobuf {google.protobuf.__version__}, numpy {numpy.__version__}")
rng = numpy.random.default_rng(0)
X = rng.normal(size=(2000, 5)).astype(numpy.float32)
model = HistGradientBoostingRegressor(max_iter=5).fit(X, X[:, 0])
with tempfile.TemporaryDirectory() as directory:
    SKLearnManager(filename=f"{directory}/probe", model_type="onnx").save(model=model, n_features=5)
    loaded = SKLearnManager(filename=f"{directory}/probe", model_type="onnx")
    loaded.load()
    assert numpy.allclose(loaded.predict(X)[:, 0], model.predict(X), atol=1e-5)
print("ONNX round trip ok")
EOF

echo "stage $MLI_STAGE | task $MLI_TASK | fit seed $MLI_FIT_SEED | search seed $MLI_SEARCH_SEED"
echo "commit $(git rev-parse HEAD) | sets $MLI_INPUT_SETS | processes $MLI_PROCESSES | work $MLI_WORK"

# ---- the work ------------------------------------------------------------------------------------
MLI_COMMON=(--split-manifest "$MLI_SPLIT_MANIFEST" --models-dir "$MLI_MODELS_DIR"
            --work-dir "$MLI_WORK" --dataset-directory "$MLI_DATASETS"
            --input-sets "$MLI_INPUT_SETS" --per-lattice "$MLI_PER_LATTICE"
            --seed "$MLI_SEED" --processes "$MLI_PROCESSES")

case "$MLI_STAGE" in
    train)
        "$MLI_PYTHON" -m mlindex.scripts.run_template_yield --stage train "${MLI_COMMON[@]}" \
            --bravais-lattices "$MLI_LATTICES" --fit-seed "$MLI_FIT_SEED"
        ;;
    yield)
        # The general population over every lattice, then the hard one over its own. An empty
        # bundle array must expand to nothing: `set -u` makes a bare empty-array expansion an error
        # on bash 3.2, which is what the `+` form avoids.
        MLI_GENERAL_ARGS=()
        [ -n "$MLI_GENERAL_BUNDLES" ] && MLI_GENERAL_ARGS=(--bundles "$MLI_GENERAL_BUNDLES")
        MLI_HARD_ARGS=()
        [ -n "$MLI_HARD_BUNDLES" ] && MLI_HARD_ARGS=(--bundles "$MLI_HARD_BUNDLES")
        "$MLI_PYTHON" -m mlindex.scripts.run_template_yield --stage yield "${MLI_COMMON[@]}" \
            --bravais-lattices "$MLI_LATTICES" --population general \
            ${MLI_GENERAL_ARGS[@]+"${MLI_GENERAL_ARGS[@]}"} \
            --group "$MLI_GROUP" --search-seed "$MLI_SEARCH_SEED" --cut "$MLI_CUT" \
            --out-dir "$MLI_WORK/general_$MLI_GROUP"
        "$MLI_PYTHON" -m mlindex.scripts.run_template_yield --stage yield "${MLI_COMMON[@]}" \
            --bravais-lattices "$MLI_HARD_LATTICES" --population hard \
            ${MLI_HARD_ARGS[@]+"${MLI_HARD_ARGS[@]}"} \
            --group "$MLI_GROUP" --search-seed "$MLI_SEARCH_SEED" --cut "$MLI_CUT" \
            --out-dir "$MLI_WORK/hard_$MLI_GROUP"
        ;;
    *)
        fail "MLI_STAGE must be train or yield, not $MLI_STAGE"
        ;;
esac

echo "done $MLI_STAGE task $MLI_TASK"
