#!/bin/bash
# S02 -- the two additional failure mechanisms, as extra conditions for the FOM development set.
#
#   sbatch submit_fom_mirror_mechanisms.sh
#
# 18 bundles, one per array task, ADDITIVE to the 21 already under characterization/fom/mirror --
# every tag here is distinct from those, checked, so nothing is overwritten.
#
# WHY THESE CONDITIONS. Both mechanisms are justified physically, not chosen to move an aggregate;
# S02's original "reproduce 75%/89%" gate was retired because meeting it meant tuning the generator
# to the sealed benchmark's answers (STATUS section 6). The error axis is anchored on multiplier 1,
# the characterised sigma(q2) for this instrument population, with 0.5 and 2 spanning it.
#
#   --n-dropout N        Interior peak dropout. Deletes N peaks from within the low-q2 window and
#                        backfills from higher q2, which is what undetected weak reflections do to a
#                        real pattern. It punches holes where the systematic-absence pattern lives,
#                        so it degrades candidate *discovery* rather than only the fit -- the one
#                        kind of damage the existing q2-scatter model cannot produce. Measured:
#                        N=3 removes 2 of the original lowest 10 peaks and pushes q2_20 out 1.12x,
#                        N=6 removes 3 and pushes it 1.25x. Headroom exists -- 86% of cP and
#                        93-99.9% of every other lattice's validation entries carry >=28 peaks.
#                        A thin entry drops fewer rather than returning a short list, because the
#                        ONNX generators reject anything under the nominal count (F-044); the
#                        achieved dropout is recorded per entry as n_dropout_achieved.
#
#   --contaminant-bias B  Weights contaminant positions towards low q2, where a second phase's
#                        strong reflections actually sit. B=2 is uniform in q rather than in q2.
#                        Fraction landing within the first five real peaks, where the generators
#                        take their information: 35% at B=1 (the old uniform draw), 61% at B=2,
#                        72% at B=3. This is the cheap stand-in for a full second-phase generator:
#                        same attack on discovery, without needing several mutually-consistent
#                        lines and therefore without a high contaminant count.
#
# Bias only appears with at least one contaminant -- with cont 0 it would regenerate an existing
# bundle under a new name and waste a node.
#
# SAME ENTRIES AS THE FROZEN MANIFEST. Sampling depends only on --seed and --n-entries-per-bl, so
# these bundles cover exactly the 5 955 entries in S02_mirror_manifest.parquet and inherit its
# fom-train / fom-dev / fom-test split. No re-splitting, ever.

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J fom_mech
#SBATCH --mail-user=dwmoreau@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A lcls
#SBATCH -t 5:00:00
#SBATCH --array=0-17

# Absolute interpreter path, not `module load conda; conda activate`: with 18 array tasks a shell
# initialisation failure would take all of them down at once, after queueing.
PYTHON=/global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python
cd /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/scripts || exit 1

NPOOLS=32         # 32x4 measured 1.45x faster than 16x8 on a whole node
POOLSIZE=4        # 1 manager + 3 workers per pool. NPOOLS * POOLSIZE = 128 = core count
NPERBL=500        # matches the frozen manifest

# error  contaminants  bias  dropout
BUNDLES=(
    # interior dropout across the error range, clean patterns
    "0.5  0  1  3"
    "0.5  0  1  6"
    "1    0  1  3"
    "1    0  1  6"
    "2    0  1  3"
    "2    0  1  6"
    # interior dropout with one contaminant, at the nominal error
    "1    1  1  3"
    "1    1  1  6"
    # low-angle contaminants across the error range
    "0.5  1  2  0"
    "0.5  1  3  0"
    "1    1  2  0"
    "1    1  3  0"
    "2    1  2  0"
    "2    1  3  0"
    # low-angle contaminants, two of them, at the nominal error
    "1    2  2  0"
    "1    2  3  0"
    # both mechanisms together, at the nominal error
    "1    1  2  3"
    "1    1  3  6"
    )

read -r ERROR CONT BIAS DROP <<< "${BUNDLES[$SLURM_ARRAY_TASK_ID]}"
echo "array task $SLURM_ARRAY_TASK_ID: error $ERROR, contaminants $CONT, bias $BIAS, dropout $DROP"

$PYTHON run_fom_mirror.py --n-pools $NPOOLS --pool-size $POOLSIZE \
    --error-multiplier "$ERROR" --n-contaminants "$CONT" \
    --contaminant-bias "$BIAS" --n-dropout "$DROP" \
    --n-entries-per-bl "$NPERBL" --shard 0 --n-shards 1


# ------------------------------------------------------------------------------------------------
# Afterwards, once the whole array has finished. The report covers all 39 bundles at once; the
# manifest stays frozen against C1 nominal (error1_cont0) and must NOT be re-frozen against one of
# these -- they are extra conditions on the same entries, not a new nominal.
# ------------------------------------------------------------------------------------------------
#/global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python run_fom_mirror_analysis.py
