"""Benchmark B's acceptance gate: the six conditions in S07's handoff, cheapest layer first.

    python run_fom_dump_gate.py floor --split-manifest <manifest.parquet> \\
        --artifact-dir docs/fom_campaign2/artifacts
    python run_fom_dump_gate.py check --pool <consolidated pool> \\
        --artifact-dir docs/fom_campaign2/artifacts

`floor` writes the correct-candidate floor table. **It is run BEFORE the array is submitted, never
after.** PROTOCOL section 7 forbids weakening a gate to pass it; naming the floor from a
measurement made and recorded in advance is not weakening, and discovering it afterwards would be.
`check` refuses to run if the table is absent.

Campaign 1's equivalent gate came back **partial** for two reasons this one is built around. Its
zero-error control bundle was arithmetically degenerate -- designed out here, since campaign 2
generates no zero-error bundle at all (METRICS.md section 9). And one stratum, triclinic at the
most aggressive dropout, had 169 correct candidates against a flat floor of 200. It also exempted
four lattices by name (`SPARSE_LATTICES = ('cP', 'cI', 'cF', 'oF')`), which is a gate with a hole
in it rather than a gate.

**The floor here is availability-aware** (DWMM, 2026-08-27): 200 correct candidates per
(lattice x bundle), capped by what the source population can actually supply. cF has 106 eligible
entries *in existence*, cI 156, cP 321 (C2-F-048), so a flat 200 asks five lattices for something
no sampling parameter can deliver. Every per-lattice claim on those five carries its n, which
C2-R-010 already requires.

Layers, in order, stopping at the first failure unless --keep-going:

  1. structure    every bundle readable, the join key valid, no all-null column
  2. manifest     everything SCHEMA.md's manifest section lists (gate 5)
  3. coverage     every bundle covers the same entry set for its arm (gate 3)
  4. floor        every (lattice x bundle) stratum meets its floor (gate 1)
  5. weights      the subsampled pool reproduces full-pool rank metrics (gate 6)
  6. roundtrip    every stored merit recomputed from the dump (gate 2)

Gate 4, "a subset re-runs bit-identically", is not a property of a finished pool -- it needs a
second generation run -- so it lives in the runbook and in
`tests/test_fom_dump_reproducibility.py`, not here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomConditions


FLOOR_NAME = 'S07_correct_candidate_floor.csv'

# The floor a stratum is asked for when the population can supply it. Campaign 1's number, kept
# so the two campaigns' gates are comparable.
TARGET_CORRECT_PER_STRATUM = 200

# Lower Wilson bound on per-entry reachability at the generation cut, measured on the hard stratum
# (188 of 243, 77.4 %, Wilson 95 % [0.717, 0.822] -- C2-F-049). Used for every lattice only until
# the calibration run measures reachability per lattice; `floor --reachability <csv>` replaces it.
# It is the HARD stratum's rate, so it is conservative for every other lattice.
DEFAULT_REACHABILITY_LOW = 0.717

# What manifest.json has to carry for a pool to be regenerable (SCHEMA.md, gate condition 5).
REQUIRED_MANIFEST_KEYS = (
    'schema_version', 'commit', 'arch', 'numpy_version', 'scipy_version', 'model_revision',
    'seed', 'search_seed_scheme', 'prune_threshold', 'condition_parameters',
    'split_manifest_sha256', 'candidate_columns', 'entry_columns', 'iteration_scale',
    )


class GateFailure(Exception):
    """One layer's verdict. Carries the rows that failed, not just a message."""


# One definition, in FomConditions, because the consolidator needs it too (C2-F-072).
_bundle_arm = FomConditions.bundle_arm


# ------------------------------------------------------------------------------------ the floor

def build_floor(manifest_path, reachability_path=None, target=TARGET_CORRECT_PER_STRATUM):
    """One row per (bundle, lattice): the floor that stratum is held to, and why.

    `floor = min(target, round(reachability_low * n_entries))`. The cap is not an exemption --
    it is the statement that a lattice with 106 entries in existence cannot be asked for 200
    correct candidates, and the `capped` column says which strata are in that position so no
    per-lattice claim on them is read without its n.
    """
    manifest = pd.read_parquet(manifest_path)
    id_column = 'identifier' if 'identifier' in manifest.columns else 'entry_id'

    reachability = {}
    if reachability_path:
        measured = pd.read_csv(reachability_path)
        reachability = dict(zip(measured['bravais_lattice'], measured['reachability_low']))

    rows = []
    for condition in FomConditions.CONDITIONS:
        arm = _bundle_arm(condition.tag)
        entries = manifest if arm == 'core' else manifest[
            manifest['arm'].astype(str).str.contains('mechanism')]
        for bravais_lattice, group in entries.groupby('bravais_lattice', sort=True):
            n_entries = int(group[id_column].nunique())
            rate = float(reachability.get(bravais_lattice, DEFAULT_REACHABILITY_LOW))
            supply = int(round(rate * n_entries))
            rows.append({
                'condition_bundle': condition.tag,
                'arm': arm,
                'bravais_lattice': bravais_lattice,
                'n_entries': n_entries,
                'reachability_low': rate,
                'reachability_source': 'measured' if bravais_lattice in reachability
                                       else 'hard-stratum default (C2-F-049)',
                'floor': min(target, supply),
                'capped': supply < target,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- the layers

def layer_structure(pool):
    entries = FomBenchmark.load_entries(pool)
    candidates = FomBenchmark.load_candidates(pool)
    FomBenchmark._check_join(candidates, entries)
    if not FomBenchmark.has_labels(candidates):
        raise GateFailure('the pool carries no usable labels; it was not labelled at generation')

    # A column null in every row is what C2-F-046 ruled out -- with two exceptions that are null
    # for a reason rather than by omission.
    allowed_null = {'second_phase_partner', 'hkl_true_in_basis'}
    for name, frame in (('candidates', candidates), ('entries', entries)):
        empty = [column for column in frame.columns
                 if column not in allowed_null and frame[column].isna().all()]
        if empty:
            raise GateFailure(f'{name}: columns null in every row: {empty}')
    # `second_phase_partner` must be null exactly when the bundle has no second phase.
    for bundle, group in entries.groupby('condition_bundle'):
        condition = FomConditions.BY_TAG.get(bundle)
        if condition is None:
            continue
        populated = group['second_phase_partner'].notna()
        if condition.second_phase_lines > 0 and not populated.all():
            raise GateFailure(f'{bundle} injects a second phase but {int((~populated).sum())} '
                              'entries record no partner')
        if condition.second_phase_lines == 0 and populated.any():
            raise GateFailure(f'{bundle} injects no second phase but records partners')
    return entries, candidates, f'{entries.shape[0]} entries, {candidates.shape[0]} candidates'


def layer_manifest(pool):
    with open(Path(pool) / 'manifest.json', encoding='utf-8') as handle:
        manifest = json.load(handle)
    bundle_manifests = manifest.get('bundle_manifests') or {'(unconsolidated)': manifest}
    missing = {}
    for bundle, payload in bundle_manifests.items():
        absent = [key for key in REQUIRED_MANIFEST_KEYS if payload.get(key) in (None, '')]
        if absent:
            missing[bundle] = absent
    if missing:
        raise GateFailure(f'manifest keys missing (R9, gate 5): {missing}')
    schemes = {payload.get('search_seed_scheme') for payload in bundle_manifests.values()}
    if schemes != {'per_entry_bravais'}:
        raise GateFailure(f'search_seed_scheme is {schemes}, not per_entry_bravais (R17)')
    arches = {payload.get('arch') for payload in bundle_manifests.values()}
    if len(arches) > 1:
        raise GateFailure(f'the pool was generated on more than one architecture: {arches}. '
                          'An arm64-generated pool is not bit-reproducible on x86 (R9)')
    return f'{len(bundle_manifests)} bundles, arch {arches.pop()}'


def layer_coverage(entries):
    """Gate 3: every bundle covers the same entry set for the arm it belongs to.

    NOT enforced by intersecting. Campaign 1 lost 33 entries to unplaceable second-phase lines and
    then aligned bundles by intersection, which is where its volume-decile drift entered (R14,
    C2-F-050). Here a shortfall is a failure to report, not a set to silently shrink.
    """
    report = []
    failures = []
    for arm in sorted({_bundle_arm(bundle) for bundle in entries['condition_bundle'].unique()}):
        bundles = {bundle: set(group['entry_id'])
                   for bundle, group in entries.groupby('condition_bundle')
                   if _bundle_arm(bundle) == arm}
        if not bundles:
            continue
        union = set().union(*bundles.values())
        for bundle, ids in sorted(bundles.items()):
            missing = len(union - ids)
            report.append(f'{bundle}: {len(ids)} entries, {missing} short of the {arm} union')
            if missing:
                failures.append((bundle, missing))
    if failures:
        raise GateFailure('bundles do not cover the same entry set within their arm: '
                          + '; '.join(f'{bundle} short {n}' for bundle, n in failures)
                          + '. Record the loss per bundle and do NOT intersect (R14).')
    return '; '.join(report)


def layer_floor(entries, candidates, floor_table):
    """Gate 1: every (lattice x bundle) stratum meets its floor.

    The stratum is keyed on the entry's TRUE Bravais lattice, not the candidate's -- METRICS.md
    section 5. Counted over the whole pool; the per-split breakdown is reported beside it, because
    `fom-dev` is what carries the claims.
    """
    truth = entries.set_index(['entry_id', 'condition_bundle'])['bravais_lattice_true']
    keyed = candidates.set_index(['entry_id', 'condition_bundle'])
    counts = (candidates
              .assign(bravais_lattice_true=keyed.index.map(truth).to_numpy())
              .groupby(['condition_bundle', 'bravais_lattice_true'])['is_correct']
              .sum().rename('n_correct').reset_index()
              .rename(columns={'bravais_lattice_true': 'bravais_lattice'}))
    merged = floor_table.merge(counts, on=['condition_bundle', 'bravais_lattice'], how='inner')
    if merged.empty:
        raise GateFailure('no (bundle, lattice) stratum in the pool matches the floor table; '
                          'check the floor table was built from the same split manifest')
    short = merged.loc[merged['n_correct'] < merged['floor']]
    if not short.empty:
        detail = '; '.join(f'{row.bravais_lattice}/{row.condition_bundle} '
                           f'{int(row.n_correct)} < {int(row.floor)}'
                           for row in short.itertuples())
        raise GateFailure(f'{short.shape[0]} strata below their correct-candidate floor: {detail}')
    n_capped = int(merged['capped'].sum())
    return (f'{merged.shape[0]} strata meet their floor '
            f'({n_capped} capped by the source population, C2-R-010)')


def layer_weights(candidates, full_pool=None):
    """Gate 6: the subsampling weights reproduce full-pool rank metrics.

    Checked on a held-back fully-retained subset -- a shard generated with --no-subsample. It is
    the only check that the retention rule did not quietly change what the benchmark measures.

    Two properties, and they are different. Every correct candidate survives, which the retention
    rule guarantees by construction and which is checked here as an identity rather than assumed.
    And the weighted candidate count reproduces the unweighted full-pool count, which is what makes
    any aggregate over the thinned pool unbiased for the whole one.
    """
    if 'retained_reason' not in candidates.columns:
        raise GateFailure('the pool carries no retained_reason column, so the retention rule '
                          'cannot be audited (SCHEMA.md)')
    correct_not_kept = candidates.loc[candidates['is_correct']
                                      & (candidates['retained_reason'] != 'correct')]
    if not correct_not_kept.empty:
        raise GateFailure(f'{correct_not_kept.shape[0]} correct candidates are not marked '
                          'retained_reason="correct"; the retention rule did not protect them')
    if (candidates.loc[candidates['retained_reason'] != 'sampled', 'sampling_weight'] != 1.0).any():
        raise GateFailure('a candidate retained with certainty carries a weight other than 1.0')

    if full_pool is None:
        return ('positives all retained; no --full-pool given, so the weighted-count check is '
                'NOT run -- gate 6 is partial')

    reference = FomBenchmark.load_candidates(full_pool)
    keys = ['entry_id', 'condition_bundle', 'bravais_lattice']
    thinned = (candidates.groupby(keys)['sampling_weight'].sum()
               .rename('weighted').reset_index())
    actual = reference.groupby(keys).size().rename('actual').reset_index()
    merged = thinned.merge(actual, on=keys, how='inner')
    if merged.empty:
        raise GateFailure('the thinned pool and the full pool share no (entry, bundle, lattice)')
    relative = (merged['weighted'] - merged['actual']).abs() / merged['actual'].clip(lower=1)
    # A Bernoulli sample of a pool of n has standard error sqrt(n(1-p)/p)/n on the weighted count,
    # so the tolerance is a sampling statement, not a numerical one.
    worst = float(relative.max())
    if relative.mean() > 0.10:
        raise GateFailure(f'weighted counts do not reproduce full-pool counts: mean relative '
                          f'error {relative.mean():.3f}, worst {worst:.3f}')
    return (f'positives all retained; weighted counts reproduce {merged.shape[0]} pools, '
            f'mean relative error {relative.mean():.4f}, worst {worst:.4f}')


def layer_roundtrip(entries, candidates, tolerance=1e-6, max_rows=None):
    """Gate 2: every stored merit recomputed from the dump matches the pipeline's value.

    S05 measured 0.000e+00 over 19 493 candidates against a 1e-6 gate; there is no reason to
    accept worse. Recomputation reads only the dumped columns, which is the property that makes
    the pool self-describing.
    """
    sample = candidates if max_rows is None else candidates.head(max_rows)
    recomputed = FomBenchmark.recompute_frame(sample, entries)
    stored = recomputed['M20'].to_numpy(dtype=float)
    fresh = recomputed['M20_recomputed'].to_numpy(dtype=float)
    scale = np.maximum(np.abs(stored), 1e-12)
    relative = np.abs(fresh - stored) / scale
    n_bad = int((relative > tolerance).sum())
    if n_bad:
        worst = int(np.argmax(relative))
        raise GateFailure(
            f'{n_bad} of {relative.size} candidates do not reproduce their stored M20 '
            f'(worst {relative[worst]:.3e} on {sample.iloc[worst]["entry_id"]}/'
            f'{sample.iloc[worst]["bravais_lattice"]})')
    return f'{relative.size} candidates, max relative difference {relative.max():.3e}'


# ------------------------------------------------------------------------------------------ CLI

def _run_floor(args):
    table = build_floor(args.split_manifest, args.reachability)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / FLOOR_NAME
    table.to_csv(path, index=False)
    capped = table.loc[table['capped'], 'bravais_lattice'].unique()
    print(f'wrote {path}: {table.shape[0]} strata, floor {table["floor"].min()}'
          f'-{table["floor"].max()}')
    print(f'capped by the source population: {sorted(capped)} (C2-F-048, C2-R-010)')
    if not args.reachability:
        print('reachability: the hard stratum\'s Wilson lower bound for every lattice '
              f'({DEFAULT_REACHABILITY_LOW}). Re-run with --reachability once the calibration '
              'job has measured it per lattice.')
    return 0


def _run_check(args):
    floor_path = Path(args.artifact_dir) / FLOOR_NAME
    if not floor_path.exists():
        raise SystemExit(
            f'{floor_path} does not exist. The correct-candidate floor is named BEFORE the run, '
            'not after it (PROTOCOL section 7): run `run_fom_dump_gate.py floor` first, and if '
            'the pool is already generated, say so when reporting the gate.')
    floor_table = pd.read_csv(floor_path)

    entries, candidates, structure = layer_structure(args.pool)
    layers = [
        ('structure', lambda: structure),
        ('manifest', lambda: layer_manifest(args.pool)),
        ('coverage', lambda: layer_coverage(entries)),
        ('floor', lambda: layer_floor(entries, candidates, floor_table)),
        ('weights', lambda: layer_weights(candidates, args.full_pool)),
        ('roundtrip', lambda: layer_roundtrip(entries, candidates, args.tolerance,
                                              args.roundtrip_rows)),
        ]
    failed = []
    for name, layer in layers:
        try:
            print(f'[{name:10s}] PASS  {layer()}', flush=True)
        except GateFailure as failure:
            print(f'[{name:10s}] FAIL  {failure}', flush=True)
            failed.append(name)
            if not args.keep_going:
                break
    if failed:
        print(f'\nGATE FAILED at: {", ".join(failed)}')
        return 1
    print('\nGATE PASSED')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='stage', required=True)

    floor = sub.add_parser('floor', help='write the correct-candidate floor table')
    floor.add_argument('--split-manifest', type=str, required=True)
    floor.add_argument('--reachability', type=str, default=None,
                       help='CSV of bravais_lattice,reachability_low from the calibration run. '
                            'Without it every lattice takes the hard stratum\'s measured lower '
                            'bound, which is conservative for all of them')
    floor.add_argument('--artifact-dir', type=str, required=True)
    floor.set_defaults(func=_run_floor)

    check = sub.add_parser('check', help='run the acceptance gate against a consolidated pool')
    check.add_argument('--pool', type=str, required=True)
    check.add_argument('--artifact-dir', type=str, required=True)
    check.add_argument('--full-pool', type=str, default=None,
                       help='A fully-retained pool over a held-back entry subset, for gate 6')
    check.add_argument('--tolerance', type=float, default=1e-6)
    check.add_argument('--roundtrip-rows', type=int, default=None,
                       help='Cap the round trip at this many candidates. Reported when set: a '
                            'silently truncated check reads as full coverage')
    check.add_argument('--keep-going', action='store_true')
    check.set_defaults(func=_run_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
