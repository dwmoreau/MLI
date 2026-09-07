#!/bin/bash
# S18 -- the laptop pilot of the variant arms: the three variants at cut 3.5 on the 28 pilot
# crystals (nominal bundle), each generated, stamped, given its sidecars and reduced, then the
# `variants` stage against the S15 pilot's cut-3.5 arm. What it proves: the driver end to end on
# real output for a variant arm; the digest check between a variant and its base; the contrast,
# mechanism and cost tables; the report. Nothing in it is a result -- 28 crystals, one bundle.
#
#   nohup bash mlindex/scripts/pilot_fom_e2e_variants.sh > pilot_fom_e2e_variants.log 2>&1 &
#
# Needs the S15 pilot to have run first (mlindex/scripts/pilot_fom_e2e.sh): the base arm is
# mlindex/data/fom_e2e_pilot/e2e/general/cut3.5_pool and its reductions are S15_reduced_general_cut3.5_pilot_*.
set -e
PYTHON=${PYTHON:-/Users/DWMoreau/miniforge3/envs/mli/bin/python}
cd "$(dirname "$0")/../.." || exit 1
ARTIFACTS=docs/fom_campaign2/artifacts
PILOT="$ARTIFACTS/S15_pilot_entries.csv"
OUTROOT=${OUTROOT:-mlindex/data/fom_e2e_pilot}
NPOOLS=${NPOOLS:-2}
PROCESSES=${PROCESSES:-4}
CUT=3.5
MODEL=mlindex/models/fom_combiner_c2_fullscale/plus_probation_seed12345
DRIVER="$PYTHON mlindex/scripts/run_fom_end_to_end.py"

[ -f "$PILOT" ] || { echo "run --stage plan --pilot first" >&2; exit 1; }
[ -f "$OUTROOT/e2e/general/cut${CUT}_pool/manifest.json" ] || { echo "run pilot_fom_e2e.sh first" >&2; exit 1; }
[ -f "$ARTIFACTS/S15_reduced_general_cut${CUT}_pilot_meta.json" ] || { echo "the S15 pilot's cut-$CUT reduction is missing" >&2; exit 1; }

echo "=== S18 variants pilot: $(date) ==="
for VARIANT in _mask _nofilter _posterior; do
    case "$VARIANT" in
        _mask)      OPTS=(--opt-param assignment_statistic=posterior --opt-param assignment_threshold=0.99) ;;
        _nofilter)  OPTS=(--opt-param assignment_threshold=0.0) ;;
        _posterior) OPTS=(--opt-param hkl_source=posterior) ;;
    esac
    echo "--- generate general$VARIANT cut $CUT, nominal, $(wc -l < "$PILOT") lines: $(date)"
    $DRIVER --stage generate --population general --arm-suffix "$VARIANT" --cut "$CUT" \
        --condition nominal --entries-file "$PILOT" --out-root "$OUTROOT" \
        --n-pools "$NPOOLS" --pool-size 2 "${OPTS[@]}"
    $DRIVER --stage complete --population general --arm-suffix "$VARIANT" --cut "$CUT" --out-root "$OUTROOT"
    $DRIVER --stage sidecars --population general --arm-suffix "$VARIANT" --cut "$CUT" --out-root "$OUTROOT" \
        --processes "$PROCESSES" --python "$PYTHON" --execute
    $DRIVER --stage reduce --population general --arm-suffix "$VARIANT" --cut "$CUT" --out-root "$OUTROOT" \
        --learned "plus_probation=$MODEL" --suffix _pilot
done
echo "--- variants analysis: $(date)"
$DRIVER --stage variants --out-root "$OUTROOT" --suffix _pilot --n-bootstrap 200
echo "=== S18 variants pilot DONE: $(date) ==="
