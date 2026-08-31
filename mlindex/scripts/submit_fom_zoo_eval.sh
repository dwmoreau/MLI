#!/bin/bash
# S09 -- reduce Benchmark B for the merit zoo. The half of the analysis that needs the pool.
#
#   sbatch submit_fom_zoo_eval.sh
#
# WHY A REDUCE-ONLY JOB. The pool is 122 GB and its merit sidecars another 26 GB; the record, the
# figures and the paper live on a laptop with ~14 GB free. `FomMetrics.evaluate` splits into
# `reduce_to_per_entry`, which touches the pool, and `summarise_per_entry`, which does not -- and
# the reduction is ONE ROW PER (entry, condition), a few hundred megabytes for the whole pool. It
# is a sufficient statistic: every threshold, every metric, every stratum, McNemar and the
# bootstrap are functions of it. So this job reduces, and the laptop runs
#
#   python mlindex/scripts/run_fom_zoo_eval.py --analyse --tag S09_zoo
#   python mlindex/scripts/run_fom_zoo_explain.py --tag S09_zoo
#
# afterwards, through exactly the code the slice run already exercised.
#
# DO NOT RUN submit_fom_zoo_merits.sh. The sidecars exist and verify at 880 704 233 of
# 880 704 233. Recomputing them is 33 core-hours for no change: this job reads them.
#
# WHAT IT COSTS. One pool pass per (merit, split): 7 merits x 2 splits = 14 passes. Measured on
# the 13.2 M-candidate slice at ~19 s a pass; the pool is ~67x that, so ~21 min a pass and ~5 h
# total. Walltime is 8 h against it. Memory is bounded by one condition bundle at a time.
#
# THE UNFLOORED ARM IS NOT RUN HERE, and the driver refuses it on this pool. The subsampler ranked
# on the FLOORED M_rev, so a saturated fit scored 0.0, ranked last and was kept at only 5 %;
# unfloored, those same rows rank first. The arm would be scored against a field with its own
# strongest rivals removed and would come out flattered, understating what the floor is worth. It
# belongs on a fully retained pool and runs on the laptop. See C2-F-084.
#
# GETTING THE RESULT BACK. `docs/` is git-ignored and `sync_record.sh push` goes laptop -> NERSC
# and overwrites by name, so do NOT let it run before the reductions are copied off. Pull them
# with an explicit rsync from the laptop:
#
#   rsync -av <nersc>:$SCRATCH/fom_campaign2/artifacts/S09_zoo_reduced_*.parquet \
#             <nersc>:$SCRATCH/fom_campaign2/artifacts/S09_zoo_reduced_meta.json \
#             docs/fom_campaign2/artifacts/
#
# The tag differs from the laptop's slice run (`S09_zoo` here, `S09_zoo_slice` there) precisely so
# the two cannot overwrite each other.
#
# APPEND FINDINGS TO STATUS_nersc_inbox.md, NEVER TO STATUS.md. The laptop copy is authoritative
# and editing STATUS.md here is how campaign 1 lost a finding.
#
# NOT wrapped in srun: a bare `srun -n 1` pins CPU affinity to one core.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_zoo_eval
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 8:00:00
#SBATCH -o fom_zoo_eval_%j.out

PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
REPO=/global/cfs/cdirs/m4064/dwmoreau/MLI
cd "$REPO" || exit 1

POOL=${POOL:-$SCRATCH/fom_campaign2/pool}
ARTIFACTS=${ARTIFACTS:-$SCRATCH/fom_campaign2/artifacts}
TAG=${TAG:-S09_zoo}

if [ ! -d "$POOL" ]; then
    echo "FATAL: pool not found at $POOL" >&2
    exit 1
fi
if [ ! -d "$POOL/merits" ]; then
    echo "FATAL: no merit sidecars at $POOL/merits. They are a prerequisite and they exist;" >&2
    echo "       if this fires, find out what happened to them before recomputing." >&2
    exit 1
fi

mkdir -p "$ARTIFACTS"
echo "=== S09 reduce: $POOL -> $ARTIFACTS (tag $TAG) ==="
df -h "$SCRATCH" | tail -1

# Sidecars first, cheaply, from parquet metadata alone. A sidecar that is short leaves merit
# columns null after the join, and NaN ranks last -- the merit would report as uniformly
# worthless rather than raising. The driver also refuses this case per file, but finding out
# here costs seconds instead of hours.
"$PYTHON" mlindex/scripts/run_fom_floor_merits.py --pool "$POOL" --verify || exit 1

"$PYTHON" mlindex/scripts/run_fom_zoo_eval.py \
    --pool "$POOL" \
    --artifact-dir "$ARTIFACTS" \
    --tag "$TAG" \
    --n-bootstrap 1000 \
    --reduce

echo "=== done. Copy the reductions to the laptop, then run --analyse there. ==="
ls -la "$ARTIFACTS" | grep "${TAG}_reduced" | head -20
