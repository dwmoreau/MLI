#!/bin/bash
# S14 -- retrain the prior network on all FOURTEEN Bravais lattices (cubic included).
#
#   docs/sync_record.sh push                  # the record, including the frozen split manifest
#   git push origin fom_campaign2 (laptop) ; git pull (NERSC)
#   sbatch mlindex/scripts/submit_fom_prior_retrain.sh
#   if it hits the walltime, resubmit the SAME command: --resume continues from the last epoch
#   then, from the laptop -- both live in the repo on $CFS, NOT in $SCRATCH, so
#   `sync_record.sh pull-artifacts` (which reads $SCRATCH) does not see them:
#       rsync -avz <nersc>:/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/models/fom_prior_c2/ \
#                  mlindex/models/fom_prior_c2/
#       rsync -avz '<nersc>:/global/cfs/cdirs/m4064/dwmoreau/MLI/docs/fom_campaign2/artifacts/S14_prior_main14_*' \
#                  docs/fom_campaign2/artifacts/
#
# Walltime: campaign 1's `main` arm drew 6 000 patterns per class over eleven lattices for 30
# epochs (66 000 rows an epoch) and took 14.9 h on the laptop, most of it training. This job draws
# 20 000 per class over fourteen (280 000 rows an epoch), 4.2x the rows, so 48 h is asked for and
# may still not be enough -- which is why the fit now checkpoints after every epoch and a
# resubmission with --resume loses at most the epoch in progress.
#
# Campaign 1 trained on eleven lattices, excluding cubic because cubic is indexed on ten peaks
# where the extraction window takes twenty -- but the cubic ENTRIES' peak lists are twenty peaks
# long (only the cubic candidates are refined on ten), so a fourteen-class fit is well posed. The
# shipped checkpoint's three untrained classes read as probabilities of e^-19 (F-117 point 4);
# S14 masks them on the shipped model and measures this retrain as the other fix (decision
# 2026-09-05). The configuration is campaign 1's `main` arm (30 epochs, 128 volume branches,
# 1024 filters, layers 512/256/128), which took ~15 h on the laptop; a Perlmutter CPU node is
# faster and this job is walltime-slack rather than tight.
#
# The split guard reads campaign 2's frozen manifest (`fom_split_c2.parquet`, column `fom_split`)
# and refuses any manifest without a split column; `docs/sync_record.sh push` is what puts the
# manifest here, and `checksum` proves it is the one frozen on the laptop.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_s14_prior
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 48:00:00
#SBATCH -o fom_s14_prior_%j.out

module load conda
conda activate /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch
export KERAS_BACKEND=torch
PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO" || exit 1

DATASETS="$REPO/mlindex/data/generated_datasets"
MODELS="$REPO/mlindex/models/fom_prior_c2"
ARTIFACTS="$REPO/docs/fom_campaign2/artifacts"

# The split guard accepts either copy of the frozen split: the sidecar beside the datasets
# (column `fom_split`) or the record's own manifest (column `split`). They carry the same 18 991
# identifiers with identical splits (checked 2026-09-08). The sidecar is git-ignored and
# `sync_record.sh push` moves only docs/, so on a fresh checkout only the second exists here.
MANIFEST="$DATASETS/fom_split_c2.parquet"
if [ ! -f "$MANIFEST" ]; then
    MANIFEST="$ARTIFACTS/S06_split_manifest.parquet"
    echo "split sidecar absent; using the record's manifest $MANIFEST"
fi

for f in "$MANIFEST" "$DATASETS/dataset_cP.parquet" "$DATASETS/dataset_mP.parquet"; do
    if [ ! -f "$f" ]; then
        echo "FATAL: $f is missing" >&2
        exit 1
    fi
done
sha256sum "$MANIFEST"      # the record's manifest must match `docs/sync_record.sh checksum`

set -e
echo "=== prior retrain, 14 lattices, campaign 1's main configuration ==="
"$PYTHON" mlindex/scripts/run_fom_prior.py --stage main --arm main14 --include-cubic \
    --manifest "$MANIFEST" --datasets-dir "$DATASETS" --models-dir "$MODELS" \
    --artifact-dir "$ARTIFACTS" --tag S14_prior_main14 \
    --limit-per-lattice 0 --per-class 20000 --epochs 30 --n-volumes 128 --n-filters 1024 \
    --eval-source heldout --resume

echo
echo "DONE. Model at $MODELS/main14/global; artefacts S14_prior_main14_* in $ARTIFACTS"
echo "Check the .out for the per-epoch lines: every epoch must list 14 lattices (check_balanced)."
echo "Per-epoch checkpoint: $MODELS/main14/global/history.json holds one entry per finished epoch."
