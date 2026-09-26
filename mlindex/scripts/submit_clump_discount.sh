#!/bin/bash
# How much do two candidates a distance DELTA apart share their fate?
#
#   export MLI_PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
#   export MLI_REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
#   export MLI_OUT=$SCRATCH/clump_discount
#   sbatch mlindex/scripts/submit_clump_discount.sh
#
# Then, on the laptop:
#   rsync -a <nersc>:$MLI_OUT/ results/clump_discount/
#   python -m mlindex.scripts.run_clump_discount --stage reduce \
#       --out-dir <pulled copy of $MLI_OUT> --roc-dir <the measured convergence curves>
#
# The reduction writes one {lattice}_clump_discount.npz, which is the file the ensemble score
# loads. Without it the mix fit refuses to start, deliberately:
# assuming candidates are independent is what makes every fitted mix a corner.
#
# WHY THIS RUN EXISTS. P09 measured the correlation between candidates starting from an IDENTICAL
# cell -- separation exactly zero. A production pool contains almost none of those: the median
# candidate has no neighbour within 1e-6, and the clumps that do exist sit at 1e-5 to 1e-3. So the
# crowding discount P09b applied was measured where nothing occurs and applied where nothing was
# measured. This measures the regime that occurs, and the number it produces is what any crowding
# term in an ensemble score has to be built on.
#
# WHAT IT MEASURES. Groups of k candidates whose centre sits at a shell where the curve reads
# s = 0.80, 0.65, 0.50, 0.35 or 0.20 -- P09's own rule -- with the members displaced from each
# other by delta, given as a ratio of that shell's radius. delta = 0 is the control and must
# reproduce P09's numbers; on the laptop pilot it did, to within 5 % at k = 16 on both lattices
# tried.
#
# CUBIC IS EXCLUDED and the driver refuses it. A cubic cell has one free parameter, so a random
# direction is +1 or -1, every member lands at exactly +/- delta/2, and half of each group is
# coincident whatever delta is asked. P09 excluded cubic from its correlation for the same reason.
#
# LAUNCHED WITH mpiexec, NOT srun. A conda-built mpi4py does not read srun's process management,
# so every task comes up in a world of size one, believes it is the only rank, and runs the whole
# job. The driver refuses to start in that state. Set MLI_LAUNCHER yourself only if
#   srun -n 4 $MLI_PYTHON -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
# prints 4.
#
# WHAT IT COSTS. Five shells x six separations is 0.6 of P09's per-crystal work, and at 200
# crystals against P09's 500 the whole run is about a quarter of P09's correlation job -- roughly
# one node-hour. Two hours below is generous.
#
# WHAT IT WRITES. One .npz a lattice: every candidate's outcome, its realised distance from the
# true cell, its realised distance from its group's centre, and its final refined cell. Plus a
# manifest with the commit, the seed and the platform.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J clump_discount
#SBATCH -A lcls
#SBATCH -t 2:00:00
#SBATCH -o clump_discount_%j.out

set -euo pipefail

: "${MLI_PYTHON:?set MLI_PYTHON to the interpreter that has mlindex installed}"
: "${MLI_REPO:?set MLI_REPO to the repository checkout}"
: "${MLI_OUT:?set MLI_OUT to the directory the results are written to}"

MLI_RANKS="${MLI_RANKS:-128}"
MLI_LAUNCHER="${MLI_LAUNCHER:-mpiexec -n $MLI_RANKS}"
# Every lattice but the three cubic ones, which the driver refuses.
MLI_LATTICES="${MLI_LATTICES:-hP,hR,tI,tP,oC,oF,oI,oP,mC,mP,aP}"
MLI_DATASETS="${MLI_DATASETS:-$MLI_REPO/mlindex/data/generated_datasets}"
MLI_ROC="${MLI_ROC:-$MLI_REPO/mlindex/characterization/roc/data}"
MLI_MODELS="${MLI_MODELS:-$MLI_REPO/mlindex/models}"
MLI_N_ENTRIES="${MLI_N_ENTRIES:-200}"
MLI_N_GROUPS="${MLI_N_GROUPS:-128}"
MLI_GROUP_SIZE="${MLI_GROUP_SIZE:-64}"
MLI_RATIOS="${MLI_RATIOS:-0,0.03,0.1,0.3,0.7,1.4}"
MLI_SEED="${MLI_SEED:-12345}"

for MLI_PATH in "$MLI_REPO" "$MLI_DATASETS" "$MLI_ROC" "$MLI_MODELS"; do
    if [ ! -d "$MLI_PATH" ]; then
        echo "FATAL: no directory at $MLI_PATH" >&2
        exit 1
    fi
done

echo "ranks $MLI_RANKS | lattices $MLI_LATTICES | $MLI_N_ENTRIES crystals"
echo "groups $MLI_N_GROUPS of $MLI_GROUP_SIZE | separations $MLI_RATIOS"
cd "$MLI_REPO"
git rev-parse HEAD

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 KERAS_BACKEND=torch

$MLI_LAUNCHER "$MLI_PYTHON" -m mlindex.scripts.run_clump_discount \
    --stage all \
    --bravais-lattices "$MLI_LATTICES" \
    --dataset-directory "$MLI_DATASETS" \
    --roc-dir "$MLI_ROC" \
    --models-directory "$MLI_MODELS" \
    --out-dir "$MLI_OUT" \
    --n-entries "$MLI_N_ENTRIES" \
    --n-groups "$MLI_N_GROUPS" \
    --group-size "$MLI_GROUP_SIZE" \
    --separation-ratios "$MLI_RATIOS" \
    --seed "$MLI_SEED"

echo "done $MLI_OUT"
du -sh "$MLI_OUT"
