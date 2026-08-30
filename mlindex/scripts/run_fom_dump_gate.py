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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

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

# One row group at a time, projected to the columns a check needs. The pool is ~880 M candidate
# rows across 34 columns, four of them list-valued, so `load_candidates` -- which concatenates the
# lot into one pandas frame -- needs far more memory than a node has and killed the first gate run
# outright (C2-F-074). Everything below streams instead: batches are bounded by the row-group size
# the consolidator wrote, and no column is read that a check does not use.
BATCH_ROWS = 131072


def candidate_files(pool):
    paths = sorted(Path(pool).glob('candidates_*.parquet'))
    if not paths:
        raise GateFailure(f'no candidates_*.parquet under {pool}')
    return paths


def _batches(path, columns):
    """Row-group batches of one file, projected. Absent columns are simply not requested."""
    available = set(pq.read_schema(path).names)
    wanted = [column for column in columns if column in available]
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_ROWS, columns=wanted):
        yield pa.Table.from_batches([batch])


def scan_candidates(pool, entries):
    """One streaming pass over every candidate file, accumulating what the cheap layers need.

    Returns a dict. Doing it once rather than per layer matters: each pass is a read of 122 GB.
    """
    truth = dict(zip(entries['entry_id'].astype(str) + '\x00'
                     + entries['condition_bundle'].astype(str),
                     entries['bravais_lattice_true']))
    digests = dict(zip(entries['entry_id'].astype(str) + '\x00'
                       + entries['condition_bundle'].astype(str),
                       entries['q2_digest']))

    scan = {'n_rows': 0, 'non_null': {}, 'columns': None, 'floor': {},
            'correct_not_marked': 0, 'bad_weight': 0, 'weighted': {}, 'join_errors': [],
            'n_entry_keys': len(digests),
            'sample_entry_keys': [repr(key) for key in list(digests)[:2]],
            'sample_candidate_keys': []}

    needed = ['entry_id', 'condition_bundle', 'q2_digest', 'bravais_lattice',
              'is_correct', 'retained_reason', 'sampling_weight']
    for path in candidate_files(pool):
        schema = pq.read_schema(path)
        if scan['columns'] is None:
            scan['columns'] = list(schema.names)
        # Null counts come from the footer's statistics -- no rows are read at all.
        metadata = pq.ParquetFile(path).metadata
        for index, name in enumerate(schema.names):
            total = 0
            for group in range(metadata.num_row_groups):
                column = metadata.row_group(group).column(index)
                total += column.num_values - (column.statistics.null_count
                                              if column.statistics is not None else 0)
            scan['non_null'][name] = scan['non_null'].get(name, 0) + total

        for table in _batches(path, needed):
            scan['n_rows'] += table.num_rows
            pairs = table.group_by(['entry_id', 'condition_bundle', 'q2_digest']).aggregate([])
            for row in pairs.to_pylist():
                key = f"{row['entry_id']}\x00{row['condition_bundle']}"
                if len(scan['sample_candidate_keys']) < 2:
                    scan['sample_candidate_keys'].append(repr(key))
                expected = digests.get(key)
                if expected is None:
                    scan['join_errors'].append(f"{row['entry_id']}/{row['condition_bundle']} "
                                               'is absent from the entry table')
                elif expected != row['q2_digest']:
                    scan['join_errors'].append(
                        f"{row['entry_id']}/{row['condition_bundle']} carries q2_digest "
                        f"{row['q2_digest']} but its entry row says {expected}")

            if 'is_correct' in table.column_names:
                keys = pa.array([truth.get(f'{e}\x00{b}') for e, b in
                                 zip(table.column('entry_id').to_pylist(),
                                     table.column('condition_bundle').to_pylist())])
                tagged = table.append_column('bravais_lattice_true', keys)
                for row in (tagged.group_by(['condition_bundle', 'bravais_lattice_true'])
                            .aggregate([('is_correct', 'sum')]).to_pylist()):
                    scan['floor'][(row['condition_bundle'], row['bravais_lattice_true'])] = (
                        scan['floor'].get((row['condition_bundle'],
                                           row['bravais_lattice_true']), 0)
                        + int(row['is_correct_sum'] or 0))

            if {'is_correct', 'retained_reason'} <= set(table.column_names):
                correct = pc.fill_null(table.column('is_correct'), False)
                reason = table.column('retained_reason')
                scan['correct_not_marked'] += pc.sum(pc.and_(
                    correct, pc.not_equal(reason, 'correct'))).as_py() or 0
            if {'retained_reason', 'sampling_weight'} <= set(table.column_names):
                certain = pc.not_equal(table.column('retained_reason'), 'sampled')
                weights = table.column('sampling_weight')
                scan['bad_weight'] += pc.sum(pc.and_(
                    certain, pc.not_equal(weights, 1.0))).as_py() or 0
            if {'entry_id', 'condition_bundle', 'bravais_lattice',
                    'sampling_weight'} <= set(table.column_names):
                for row in (table.group_by(['entry_id', 'condition_bundle', 'bravais_lattice'])
                            .aggregate([('sampling_weight', 'sum')]).to_pylist()):
                    key = (row['entry_id'], row['condition_bundle'], row['bravais_lattice'])
                    scan['weighted'][key] = (scan['weighted'].get(key, 0.0)
                                             + float(row['sampling_weight_sum'] or 0.0))
    return scan


def layer_structure(pool, entries, scan):
    """Join integrity, labels, and no unexplained all-null column -- from the streaming scan.

    The null counts come from the parquet footers' own statistics, so this touches no rows for
    that check at all.
    """
    if scan['join_errors']:
        # Counted per (batch, triple), not per candidate -- one bad key recurs in every batch it
        # appears in, so the number is an occurrence count and saying "candidates" overstates it.
        n_distinct = len(set(scan['join_errors']))
        detail = '; '.join(sorted(set(scan['join_errors']))[:3])
        diagnosis = ''
        if n_distinct >= scan['n_entry_keys']:
            # EVERY key failing is not a data fault, it is a lookup fault. Say so, with the two
            # sides side by side, rather than making someone bisect a 122 GB pool for it.
            diagnosis = (
                f"\n  ALL {n_distinct} distinct keys failed, which is a lookup fault rather than "
                f"missing data.\n"
                f"  entry-table keys: {scan['n_entry_keys']}, e.g. {scan['sample_entry_keys']}\n"
                f"  candidate keys  : e.g. {scan['sample_candidate_keys']}\n"
                f"  If those look identical, compare their types and repr(), not their printed "
                f"form.")
        raise GateFailure(f'{n_distinct} distinct (entry, bundle) keys on candidate rows do not '
                          f'match the entry table, over {len(scan["join_errors"])} occurrences. '
                          f'e.g. {detail}{diagnosis}')

    required = [column for column in FomBenchmark.LABEL_COLUMNS]
    absent = [column for column in required if column not in (scan['columns'] or [])]
    if absent:
        raise GateFailure(f'the pool carries no {absent}; it was not labelled at generation')
    if scan['non_null'].get('is_correct', 0) != scan['n_rows']:
        raise GateFailure('is_correct is null on some rows; the pool was not fully labelled')

    # A column null in every row is what C2-F-046 ruled out -- with two exceptions that are null
    # for a reason rather than by omission.
    allowed_null = {'second_phase_partner', 'hkl_true_in_basis'}
    empty = [name for name, count in scan['non_null'].items()
             if count == 0 and name not in allowed_null]
    if empty:
        raise GateFailure(f'candidates: columns null in every row: {sorted(empty)}')
    empty_entries = [column for column in entries.columns
                     if column not in allowed_null and entries[column].isna().all()]
    if empty_entries:
        raise GateFailure(f'entries: columns null in every row: {empty_entries}')

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
    return f"{entries.shape[0]} entries, {scan['n_rows']} candidates"


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


def recorded_failures(dump_roots):
    """{bundle: entries the generation run recorded as failed}, from `failures_*.json`.

    A bundle can legitimately cover fewer entries than its arm: contaminant and second-phase
    placement are rejection-sampled, and an entry whose lines cannot be placed is skipped and
    RECORDED. Campaign 1 lost 33 entries that way. What the gate has to separate is a loss with a
    reason on disk from a loss with none -- the second is what condition 3 exists to catch, and
    failing on the first would make the gate unpassable on a run that behaved correctly.
    """
    counts = {}
    for root in dump_roots or ():
        for path in sorted(Path(root).glob('*/failures_*.json')):
            try:
                with open(path, encoding='utf-8') as handle:
                    failures = json.load(handle)
            except Exception:
                continue
            bundle = path.parent.name
            counts.setdefault(bundle, set()).update(
                failure.get('identifier') for failure in failures)
    return {bundle: len(ids) for bundle, ids in counts.items()}


def layer_coverage(entries, dump_roots=None):
    """Gate 3: every bundle covers the same entry set for the arm it belongs to.

    NOT enforced by intersecting. Campaign 1 lost 33 entries to unplaceable second-phase lines and
    then aligned bundles by intersection, which is where its volume-decile drift entered (R14,
    C2-F-050). Here a shortfall is reported, and it fails only when it is UNACCOUNTED -- larger
    than the entries the run itself recorded as failed.
    """
    accounted = recorded_failures(dump_roots)
    report, failures = [], []
    for arm in sorted({FomConditions.bundle_arm(bundle)
                       for bundle in entries['condition_bundle'].unique()}):
        bundles = {bundle: set(group['entry_id'])
                   for bundle, group in entries.groupby('condition_bundle')
                   if FomConditions.bundle_arm(bundle) == arm}
        if not bundles:
            continue
        union = set().union(*bundles.values())
        for bundle, ids in sorted(bundles.items()):
            missing = len(union - ids)
            known = accounted.get(bundle, 0)
            if missing == 0:
                report.append(f'{bundle}: {len(ids)}, complete')
            elif dump_roots is None:
                failures.append(f'{bundle} short {missing}, and no --dump-root was given to '
                                'check whether the run recorded them')
            elif missing <= known:
                report.append(f'{bundle}: {len(ids)}, short {missing} -- all accounted for by '
                              f'{known} recorded generation failures')
            else:
                failures.append(f'{bundle} short {missing} with only {known} recorded failures, '
                                f'so {missing - known} are unexplained')
    if failures:
        raise GateFailure('; '.join(failures)
                          + '. Record the loss per bundle and do NOT intersect (R14).')
    return '; '.join(report)


def layer_floor(scan, floor_table):
    """Gate 1: every (lattice x bundle) stratum meets its floor.

    Keyed on the entry's TRUE Bravais lattice, not the candidate's -- METRICS.md section 5.
    Counted during the streaming scan.
    """
    counts = pd.DataFrame(
        [{'condition_bundle': bundle, 'bravais_lattice': lattice, 'n_correct': n}
         for (bundle, lattice), n in scan['floor'].items() if lattice is not None])
    if counts.empty:
        raise GateFailure('no correct candidates were counted anywhere in the pool')
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


def layer_weights(scan, pool, full_pool=None):
    """Gate 6: the subsampling weights reproduce full-pool rank metrics.

    Two properties, and they are different. Every correct candidate survives, which the retention
    rule guarantees by construction and which is checked here as an identity rather than assumed.
    And the weighted candidate count reproduces the unweighted full-pool count, which is what makes
    any aggregate over the thinned pool unbiased for the whole one.
    """
    if 'retained_reason' not in (scan['columns'] or []):
        raise GateFailure('the pool carries no retained_reason column, so the retention rule '
                          'cannot be audited (SCHEMA.md)')
    if scan['correct_not_marked']:
        raise GateFailure(f"{scan['correct_not_marked']} correct candidates are not marked "
                          'retained_reason="correct"; the retention rule did not protect them')
    if scan['bad_weight']:
        raise GateFailure(f"{scan['bad_weight']} candidates retained with certainty carry a "
                          'weight other than 1.0')

    if full_pool is None:
        return ('positives all retained; no --full-pool given, so the weighted-count check is '
                'NOT run -- gate 6 is partial')

    reference = {}
    for path in candidate_files(full_pool):
        for table in _batches(path, ['entry_id', 'condition_bundle', 'bravais_lattice']):
            for row in (table.group_by(['entry_id', 'condition_bundle', 'bravais_lattice'])
                        .aggregate([([], 'count_all')]).to_pylist()):
                key = (row['entry_id'], row['condition_bundle'], row['bravais_lattice'])
                reference[key] = reference.get(key, 0) + int(row['count_all'])
    shared = [key for key in reference if key in scan['weighted']]
    if not shared:
        raise GateFailure('the thinned pool and the full pool share no (entry, bundle, lattice)')
    relative = np.array([abs(scan['weighted'][key] - reference[key]) / max(reference[key], 1)
                         for key in shared])
    # A Bernoulli sample of a pool of n has standard error sqrt(n(1-p)/p)/n on the weighted count,
    # so the tolerance is a sampling statement, not a numerical one.
    if relative.mean() > 0.10:
        raise GateFailure(f'weighted counts do not reproduce full-pool counts: mean relative '
                          f'error {relative.mean():.3f}, worst {relative.max():.3f}')
    return (f'positives all retained; weighted counts reproduce {len(shared)} pools, '
            f'mean relative error {relative.mean():.4f}, worst {relative.max():.4f}')


def layer_roundtrip(pool, entries, tolerance=1e-6, max_rows=2_000_000):
    """Gate 2: every stored merit recomputed from the dump matches the pipeline's value.

    S05 measured 0.000e+00 over 19 493 candidates against a 1e-6 gate; there is no reason to accept
    worse. Recomputation reads only the dumped columns, which is the property that makes the pool
    self-describing.

    CAPPED, and it says so. Recomputing all ~880 M candidates means rebuilding every reference line
    list, which is hours and far more memory than a node has. `max_rows` takes a prefix of each
    file in turn, so the sample spans every (bundle, lattice) rather than whichever happened to
    sort first. Pass `--roundtrip-rows 0` for no cap.
    """
    checked, worst, worst_where = 0, 0.0, None
    for path in candidate_files(pool):
        if max_rows and checked >= max_rows:
            break
        budget = (max_rows - checked) if max_rows else None
        frame = pd.read_parquet(path)
        if budget is not None and frame.shape[0] > budget:
            frame = frame.head(budget)
        if frame.empty:
            continue
        recomputed = FomBenchmark.recompute_frame(frame, entries)
        stored = recomputed['M20'].to_numpy(dtype=float)
        fresh = recomputed['M20_recomputed'].to_numpy(dtype=float)
        relative = np.abs(fresh - stored) / np.maximum(np.abs(stored), 1e-12)
        n_bad = int((relative > tolerance).sum())
        if n_bad:
            position = int(np.argmax(relative))
            raise GateFailure(
                f'{n_bad} of {relative.size} candidates in {path.name} do not reproduce their '
                f'stored M20 (worst {relative[position]:.3e} on '
                f'{frame.iloc[position]["entry_id"]}/{frame.iloc[position]["bravais_lattice"]})')
        if relative.size and relative.max() > worst:
            worst, worst_where = relative.max(), path.name
        checked += frame.shape[0]
    capped = ' (CAPPED -- not the whole pool)' if max_rows and checked >= max_rows else ''
    return (f'{checked} candidates, max relative difference {worst:.3e}'
            + (f' in {worst_where}' if worst_where else '') + capped)


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

    # The entry table is one row per (entry, bundle) -- ~106 000 rows -- so it is loaded whole.
    # The candidates are ~880 M rows and are never loaded whole; one streaming pass feeds every
    # layer but the round trip (C2-F-074).
    entries = FomBenchmark.load_entries(args.pool)
    print(f'[{"scan":10s}] streaming {len(candidate_files(args.pool))} candidate files...',
          flush=True)
    scan = scan_candidates(args.pool, entries)

    layers = [
        ('structure', lambda: layer_structure(args.pool, entries, scan)),
        ('manifest', lambda: layer_manifest(args.pool)),
        ('coverage', lambda: layer_coverage(entries, args.dump_root)),
        ('floor', lambda: layer_floor(scan, floor_table)),
        ('weights', lambda: layer_weights(scan, args.pool, args.full_pool)),
        ('roundtrip', lambda: layer_roundtrip(args.pool, entries, args.tolerance,
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
    check.add_argument('--dump-root', type=str, default=None, nargs='+',
                       help='The generation output the pool was consolidated from. Read for its '
                            'failures_*.json, so a bundle covering fewer entries than its arm can '
                            'be checked against what the run RECORDED as failed rather than '
                            'failing outright. Without it any shortfall is unexplained by '
                            'definition')
    check.add_argument('--full-pool', type=str, default=None,
                       help='A fully-retained pool over a held-back entry subset, for gate 6')
    check.add_argument('--tolerance', type=float, default=1e-6)
    check.add_argument('--roundtrip-rows', type=int, default=2_000_000,
                       help='Cap the round trip at this many candidates, 0 for no cap. It is the '
                            'one layer that must rebuild reference line lists, so the whole pool '
                            'is hours and more memory than a node has. The cap is REPORTED in the '
                            'result line: a silently truncated check reads as full coverage')
    check.add_argument('--keep-going', action='store_true')
    check.set_defaults(func=_run_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
