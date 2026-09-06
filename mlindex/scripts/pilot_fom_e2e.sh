#!/bin/bash
# S15 -- the laptop pilot: the whole chain on 28 crystals (two per lattice), nominal bundle only,
# real runs at cuts 5.0, 3.5 and 3.0, then the cut-1.5 pool already on disk restricted to the
# same crystals, reduced, restricted at each cut, analysed, drawn and written up. What it proves:
# the driver end to end on real output; the digest check across cuts; provenance; the restriction-
# versus-run comparison in its conservative direction; and per-entry wall clock on this machine.
#
#   nohup bash mlindex/scripts/pilot_fom_e2e.sh > pilot_fom_e2e.log 2>&1 &
#
# Pool size 2 is the benchmark's identity and is kept; two pools is what 16 GB carries (each manager
# loads ~3 GB of models). ~2.5 h of generation and ~2 h over the 32 GB cut-1.5 pool.
set -e
PYTHON=${PYTHON:-/Users/DWMoreau/miniforge3/envs/mli/bin/python}
cd "$(dirname "$0")/../.." || exit 1
ARTIFACTS=docs/fom_campaign2/artifacts
PILOT="$ARTIFACTS/S15_pilot_entries.csv"
OUTROOT=${OUTROOT:-mlindex/data/fom_e2e_pilot}
FULL_POOL=${FULL_POOL:-mlindex/data/fom_full_c2_pool}
NPOOLS=${NPOOLS:-2}
PROCESSES=${PROCESSES:-4}
MODEL=mlindex/models/fom_combiner_c2_fullscale/plus_probation_seed12345
DRIVER="$PYTHON mlindex/scripts/run_fom_end_to_end.py"

[ -f "$PILOT" ] || { echo "run --stage plan --pilot first" >&2; exit 1; }
echo "=== S15 pilot: $(date) ==="
for CUT in 5.0 3.5 3.0; do
    echo "--- generate general cut $CUT, nominal, $(wc -l < "$PILOT") lines: $(date)"
    $DRIVER --stage generate --population general --cut "$CUT" --condition nominal \
        --entries-file "$PILOT" --out-root "$OUTROOT" --n-pools "$NPOOLS" --pool-size 2
    $DRIVER --stage complete --population general --cut "$CUT" --out-root "$OUTROOT"
    $DRIVER --stage sidecars --population general --cut "$CUT" --out-root "$OUTROOT" \
        --processes "$PROCESSES" --python "$PYTHON" --execute
    $DRIVER --stage reduce --population general --cut "$CUT" --out-root "$OUTROOT" \
        --learned "plus_probation=$MODEL" --suffix _pilot
done
echo "--- cut 1.5 from the existing pool, restricted to the pilot crystals: $(date)"
$DRIVER --stage reduce --population general --cut 1.5 --pool "$FULL_POOL" \
    --keep-entries "$PILOT" --learned "plus_probation=$MODEL" --suffix _pilot
echo "--- restriction versus run: $(date)"
$DRIVER --stage restrict --population general --pool "$FULL_POOL" --keep-entries "$PILOT" \
    --cuts 5.0,3.5,3.0 --learned "plus_probation=$MODEL" --suffix _pilot
echo "--- analyse / figure / report: $(date)"
$DRIVER --stage analyse --out-root "$OUTROOT" --suffix _pilot --keep-entries "$PILOT" \
    --existing-pool "general:1.5=$FULL_POOL" --n-bootstrap 200
$DRIVER --stage figure --suffix _pilot
$DRIVER --stage report --suffix _pilot
echo "=== S15 pilot DONE: $(date) ==="
