#!/bin/bash
# S14 -- the network's inputs for S12's exported full-scale rows, one condition bundle per task.
#
#   docs/sync_record.sh push                                   # the record, so the job can read it
#   rsync -avz mlindex/models/fom_prior/main/global/ <nersc>:$REPO/mlindex/models/fom_prior/main/global/
#   sbatch submit_fom_neural_inputs.sh
#   then, from the laptop:
#       docs/sync_record.sh pull-artifacts 'S14_neural_inputs_fullscale*'
#       python mlindex/scripts/run_fom_neural_score.py --stage fit --suffix _fullscale \
#           --fit-frame 'docs/fom_campaign2/artifacts/S12_combiner_fit_frame_fullscale_*.parquet' \
#           --inputs-dir docs/fom_campaign2/artifacts/S14_neural_inputs_fullscale
#
# WHY THIS EXISTS. S12's full-scale frames (2 381 244 fit rows, 343 884 correct; C2-F-143) are on
# the laptop, but they carry no `xnn` and no `q2_obs`, and the network's inputs -- the twenty
# per-peak assignment posteriors and the prior read at the claimed pair -- need both. The pool
# those rows came from is here. So this computes the inputs for EXACTLY the keyed rows in those
# frames (about 7.6 M) rather than for the 880 M candidates of the pool, and ships one directory of
# sidecars back. `--keys-from` refuses to finish if a single requested key is not found in the pool:
# a key that is silently missing becomes a NaN input on the laptop with no other symptom.
#
# The prior network needs keras on the torch backend, which is why this runs in `envs/pytorch`
# and not the inference-only `envs/onnx` the S12 export used. The entries stage runs once, in
# task 0, before the array's candidate passes; the other tasks wait for its output file.
#
# Read the .out file, never the exit status (S12's nine silent failures, C2-F-135/138).

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_s14_inputs
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 4:00:00
#SBATCH --array=0-8
#SBATCH -o fom_s14_inputs_%A_%a.out

module load conda
conda activate /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch
export KERAS_BACKEND=torch
PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO" || exit 1

POOL="$SCRATCH/fom_campaign2/pool"
FRAMES="$SCRATCH/fom_campaign2/artifacts"
OUTDIR="$SCRATCH/fom_campaign2/artifacts/S14_neural_inputs_fullscale"
PRIOR="$REPO/mlindex/models/fom_prior/main/global"
PROCESSES="${SLURM_CPUS_ON_NODE:-32}"

BUNDLES=(c2_error0.1_cont0 c2_error1_cont0 c2_error1_cont0_icept4 c2_error1_cont0_phase3 \
         c2_error1_cont1_drop2 c2_error1_cont1_drop4 c2_error1_cont1_drop6 c2_error1_cont2 \
         c2_error2_cont0)
BUNDLE="${BUNDLES[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$BUNDLE" ]; then
    echo "FATAL: no bundle for array index '$SLURM_ARRAY_TASK_ID'" >&2
    exit 1
fi
for f in "$FRAMES/S12_combiner_fit_frame_fullscale_$BUNDLE.parquet" \
         "$FRAMES/S12_combiner_cal_frame_fullscale_$BUNDLE.parquet" \
         "$PRIOR/prior.weights.h5" "$POOL/entries.parquet"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: $f is missing. This job computes inputs for S12's exported rows; it does not export them." >&2
        exit 1
    fi
done
mkdir -p "$OUTDIR"
KEYS="$FRAMES/S12_combiner_fit_frame_fullscale_$BUNDLE.parquet $FRAMES/S12_combiner_cal_frame_fullscale_$BUNDLE.parquet"
ALL_KEYS="$FRAMES/S12_combiner_fit_frame_fullscale_*.parquet $FRAMES/S12_combiner_cal_frame_fullscale_*.parquet"

set -e
# The entry-level block once, for EVERY bundle's entries, by task 0. The other tasks poll for it:
# the candidate pass reads the joint tables it writes, and nine copies of a keras forward pass over
# the same entries would be nine times the work for one answer.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "=== entries stage (all bundles) ==="
    # shellcheck disable=SC2086
    "$PYTHON" mlindex/scripts/run_fom_neural_inputs.py --pool "$POOL" --out-dir "$OUTDIR" \
        --stage entries --prior-dir "$PRIOR" --keys-from $ALL_KEYS
    touch "$OUTDIR/.entries_done"
else
    for _ in $(seq 1 240); do
        [ -f "$OUTDIR/.entries_done" ] && break
        sleep 30
    done
    if [ ! -f "$OUTDIR/.entries_done" ]; then
        echo "FATAL: task 0 has not written the entry tables after two hours" >&2
        exit 1
    fi
fi

echo "=== candidates stage: $BUNDLE (array task $SLURM_ARRAY_TASK_ID, $PROCESSES processes) ==="
# shellcheck disable=SC2086
"$PYTHON" mlindex/scripts/run_fom_neural_inputs.py --pool "$POOL" --out-dir "$OUTDIR" \
    --stage candidates --processes "$PROCESSES" --keys-from $KEYS

echo
echo "DONE: $BUNDLE"
echo "When ALL NINE tasks have finished:"
echo "  sacct -j \$SLURM_ARRAY_JOB_ID --format=JobID,State,Elapsed,MaxRSS"
echo "  $PYTHON mlindex/scripts/run_fom_neural_inputs.py --pool $POOL --out-dir $OUTDIR --verify"
echo "  ls $OUTDIR/candidates_*.parquet | wc -l     # 9 bundles x 14 lattices, minus any with no keys"
echo "then from the laptop: docs/sync_record.sh pull-artifacts 'S14_neural_inputs_fullscale*'"
