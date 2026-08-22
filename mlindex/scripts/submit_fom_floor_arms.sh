#!/bin/bash
# S06b -- the reproducibility floor's ensemble: the same patterns indexed K times under
# different search seeds.
#
# The floor is the spread of a *reported* number over runs that are nominally the same, so
# the ensemble has to be runs of the program rather than an operator invented to stand in for
# one. F-147 records why the stand-in was abandoned: the pool's stored cells are not fixed
# points of any refinement in the pipeline, so "perturb and refine back" has nothing to return
# to, and displacing them far enough to break the incumbency of the stored cell measures the
# basin instead (no plateau; F-149).
#
# Every arm shares --seed, so the entry sample, the per-entry noise and therefore the peak
# lists are identical -- the entry tables' q2_digest columns are compared by the report stage
# and must match. Only --optimizer-seed moves, which is the search RNG alone.
#
# Laptop, 10 cores: 5 pools x 2 processes measures ~9.6 s/entry (1 x 10 measures 21 s/entry --
# the manager, not the workers, is the constraint). 250 entries x 4 arms is ~2.7 h.
#
#   bash mlindex/scripts/submit_fom_floor_arms.sh
#
# Run from the repository root. MLINDEX_MODELS_DIR is pinned to the checkout because a stale
# $XDG_DATA_HOME/mlindex/models otherwise wins over it (F-051).
set -euo pipefail

PYTHON=${PYTHON:-/Users/DWMoreau/miniforge3/envs/mli/bin/python}
ENTRIES=${ENTRIES:-docs/fom/artifacts/S06b_floor_entries.csv}
OUT_ROOT=${OUT_ROOT:-mlindex/data/fom_floor/arms}
BUNDLE_ARGS=${BUNDLE_ARGS:---error-multiplier 1 --n-contaminants 0}
SEEDS=${SEEDS:-101 202 303 404}

export MLINDEX_MODELS_DIR="${MLINDEX_MODELS_DIR:-$PWD/mlindex/models}"
mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
    out="$OUT_ROOT/seed${seed}"
    if [ -f "$out/manifest.json" ]; then
        echo "arm $seed already complete, skipping"
        continue
    fi
    echo "=== arm $seed -> $out ==="
    "$PYTHON" mlindex/scripts/run_fom_dump.py \
        $BUNDLE_ARGS \
        --n-entries-per-bl 500 \
        --seed 12345 \
        --optimizer-seed "$seed" \
        --entry-ids-file "$ENTRIES" \
        --pool-size 2 \
        --n-pools 5 \
        --out-dir "$out"
done
echo "all arms written under $OUT_ROOT"
