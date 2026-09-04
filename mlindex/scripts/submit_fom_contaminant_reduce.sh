#!/bin/bash
# S12 -- reduce the learned score over the contaminated pool, where that pool already sits.
#
#   sbatch submit_fom_contaminant_reduce.sh
#
# WHY A BATCH JOB. This walks 45 GB and takes ~90 minutes; the clean 28 GB pool took ~80. A login
# node will kill it, and it writes NOTHING until the end, so a kill and a success look identical
# until the last minute.
#
# WHY THE POOL DOES NOT TRAVEL. A reduction is 0.1 MB an arm and the pool is 45 GB against a laptop
# with 22 GB free. So the models come here -- rsync'd to mlindex/models/fom_combiner_c2_fullscale --
# and only the per-entry reductions go back.
#
# WHY THE THRESHOLD IS REUSED. `--calibration-from _fullscale` copies the thresholds the clean
# full-scale run chose. A threshold is a property of the MODEL and of the rows it was chosen on, not
# of the pool being reported: recomputing it here would make part of any clean-vs-contaminated
# difference a difference of thresholds. It also means the fit pool need not exist on this machine,
# which it does not.
#
# WHAT COMES BACK, and what each bundle can answer:
#   c2_error1_cont2         2 independent unindexable lines, 60 peaks, no dropout -- the clean read
#   c2_error1_cont0_phase3  3 CORRELATED lines from a real partner cell; a wrong candidate can
#                           genuinely index some of them, which is the real two-phase failure
#   c2_error1_cont1_drop2/4/6   a contaminant AND 2/4/6 dropped peaks AND a 31-peak pattern -- read
#                           as a trend, not as three contaminant measurements

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_cont_reduce
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH -o fom_cont_reduce_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO" || exit 1

POOL="$SCRATCH/fom_campaign2/contaminant_pool"
MODELS="$REPO/mlindex/models/fom_combiner_c2_fullscale"

if [ ! -d "$POOL" ]; then
    echo "FATAL: contaminated pool missing at $POOL" >&2
    exit 1
fi
if [ -z "$(ls -A "$MODELS" 2>/dev/null)" ]; then
    echo "FATAL: no models at $MODELS." >&2
    echo "From the laptop:" >&2
    echo "  rsync -avz mlindex/models/fom_combiner_c2_fullscale/ \\" >&2
    echo "      <nersc>:$MODELS/" >&2
    echo "Without them the reduce falls back to fom_combiner_c2 and reports the wrong models" >&2
    echo "with no visible sign -- which is C2-F-141." >&2
    exit 1
fi

set -e

echo "=== reduce over the contaminated pool ==="
"$PYTHON" mlindex/scripts/run_fom_combiner.py --stage reduce \
    --report-pool "$POOL" \
    --calibration-from _fullscale \
    --suffix _contam \
    --arms base drop_structural plus_probation

echo
echo "DONE. Now, from the laptop:"
echo "  rsync -avz --progress \\"
echo "    '<nersc>:$REPO/docs/fom_campaign2/artifacts/S12_combiner_*_contam*' \\"
echo "    docs/fom_campaign2/artifacts/"
echo "  python mlindex/scripts/run_fom_combiner.py --stage analyse --suffix _contam"
