"""S06 step 1: every figure of merit in the zoo, over the frozen benchmark pool, computed once.

`FigureOfMerits.compute_all` costs ~268 get_M20-equivalents (docs/fom/artifacts/S01_fom_cost.csv).
S06's main table ranks ~20 merits, S07 standardises them and S08 combines them, so evaluating the
zoo inside a `FomMetrics.evaluate(score=callable)` would pay that cost once per merit per pass.
This script pays it once and writes a feature matrix that every later step joins to the pool.

What it writes, per condition bundle, into --out-dir:

    features_<bundle>.parquet     ZOO_KEY_COLUMNS + one float64 column per merit

and one gate report into --artifact-dir. The gate is free and exact: `compute_all` returns M20
last and on a copy, so the recomputed value must equal the pipeline's stored `M20` column. S04's
round trip established that it does to ~1e-12 on the six evaluable bundles (F-054 explains why C0
is not one of them), so any disagreement here means the line assignment was rebuilt wrongly, not
that the tolerance was optimistic.

Two conditions are excluded by default and both are decisions, not oversights.
C0 (`error0_cont0`) is excluded from every metric because its M20 diverges arithmetically -- zero
error means zero residual, and 9.49% of its candidates score above 1e9 (F-054, STATUS section 6).
`fom-test` is excluded because it stays sealed until S15 (PROTOCOL section 3 rule 2).

    python mlindex/scripts/run_fom_zoo_features.py --calibrate     # measure, then decide
    python mlindex/scripts/run_fom_zoo_features.py --n-processes 8

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.model_training import FomMetrics  # noqa: E402

# The six bundles any metric is computed over. C0 is the generation control and is excluded
# from every metric by decision (F-054); it is still reachable with --bundles for the
# singularity audit S06 owes F-054.
EVALUABLE_BUNDLES = (
    'error1_cont0',
    'error2_cont0',
    'error1_cont2',
    'error1_cont1_drop6',
    'error1_cont1_drop10',
    'error1_cont0_phase3',
    )

# fom-test is sealed until S15, so it is never read here.
DEFAULT_SPLITS = ('fom-train', 'fom-dev')

# The M20 round trip is exact by construction, so the tolerance only absorbs the reassociation
# float64 does inside a different call order. S04 measured ~1e-12 on these bundles.
M20_RTOL = 1e-9


def commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=BASE, stderr=subprocess.DEVNULL
            ).decode().strip()
    except Exception:
        return 'unknown'


def _feature_chunk(payload):
    """Module-level and picklable, so this survives the spawn start method."""
    candidates, entries, g_min, min_discrepancy, models_directory = payload
    features, treatments = FomBenchmark.zoo_features(
        candidates, entries, g_min=g_min, min_discrepancy=min_discrepancy,
        models_directory=models_directory,
        )
    residual = np.abs(features['M20'].to_numpy() - candidates['M20'].to_numpy())
    scale = np.maximum(np.abs(candidates['M20'].to_numpy()), 1e-12)
    return features, treatments, float(np.max(residual/scale)), float(np.max(residual))


def _chunks(candidates, entries, entry_ids, chunk_size, g_min, min_discrepancy,
            models_directory):
    """One payload per group of entries.

    Chunked by entry, not by row: `zoo_features` estimates the entrywise residual scale from an
    entry's whole pool, so splitting an entry across workers would change the estimate. Only the
    entry columns `zoo_features` reads are shipped, and only for the chunk's own entries, so a
    payload is a few MB rather than the whole 41 454-row entry table.
    """
    needed = [column for column in ('entry_id', 'condition_bundle', 'q2_obs')
              if column in entries.columns]
    by_entry = {entry_id: group for entry_id, group in candidates.groupby('entry_id', sort=False)}
    for start in range(0, len(entry_ids), chunk_size):
        batch = entry_ids[start:start + chunk_size]
        frame = pd.concat([by_entry[entry_id] for entry_id in batch], ignore_index=True)
        truth = entries.loc[entries['entry_id'].isin(set(batch)), needed]
        yield frame, truth, g_min, min_discrepancy, models_directory


def build_bundle(root, bundle, keep_entry_ids, g_min, min_discrepancy, n_processes,
                 chunk_size, models_directory=None, limit_entries=None):
    """The feature matrix for one condition bundle, plus its round-trip residual and timing."""
    candidates = FomBenchmark.load_candidates(
        root, bundles=[bundle], columns=list(FomBenchmark.ZOO_CANDIDATE_COLUMNS),
        )
    candidates = candidates.loc[candidates['entry_id'].isin(keep_entry_ids)]
    entries = FomBenchmark.load_entries(root)
    entries = entries.loc[entries['condition_bundle'] == bundle]

    entry_ids = list(pd.unique(candidates['entry_id']))
    if limit_entries is not None:
        entry_ids = entry_ids[:limit_entries]
        candidates = candidates.loc[candidates['entry_id'].isin(set(entry_ids))]
    candidates = candidates.reset_index(drop=True)

    payloads = _chunks(candidates, entries, entry_ids, chunk_size, g_min, min_discrepancy,
                       models_directory)
    start = time.perf_counter()
    if n_processes <= 1:
        results = [_feature_chunk(payload) for payload in payloads]
    else:
        from multiprocessing import Pool
        with Pool(processes=n_processes) as pool:
            results = list(pool.imap_unordered(_feature_chunk, payloads, chunksize=1))
    elapsed = time.perf_counter() - start

    features = pd.concat([item[0] for item in results], ignore_index=True)
    treatments = {}
    for item in results:
        treatments.update(item[1])
    return {
        'bundle': bundle,
        'condition': FomMetrics.BUNDLE_LABELS.get(bundle, '?'),
        'features': features,
        'treatments': treatments,
        'n_entries': len(entry_ids),
        'n_candidates': int(features.shape[0]),
        'max_M20_relative_error': max(item[2] for item in results) if results else 0.0,
        'max_M20_absolute_error': max(item[3] for item in results) if results else 0.0,
        'seconds': elapsed,
        }


AUDITED_FEATURES = ('M_info_clipped', 'null_tail_nll', 'nll_exponential', 'bic')


def coincidence_audit(out_dir, bundles):
    """Evidence for the `min_discrepancy` choice, rather than a guess at it (F-026).

    The two information-type merits carry a -log(1 - exp(-|dQ|/Delta)) term that diverges as the
    discrepancy goes to zero, so a floor is mandatory *if* exact coincidences occur. On de Wolff's
    printed table three did, and without a floor they contributed 690 of 725 nats. Whether any
    occur here is a measurement, and it has to be taken over *every* bundle -- a floor that binds
    on one condition and not another is still a floor that binds.

    Read back from the parquets rather than accumulated in memory, so what is reported is what
    was written.
    """
    rows = []
    for bundle in bundles:
        path = Path(out_dir)/f'features_{bundle}.parquet'
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=list(AUDITED_FEATURES))
        for name in frame.columns:
            values = frame[name].to_numpy()
            finite = np.isfinite(values)
            rows.append({
                'bundle': bundle,
                'feature': name,
                'n': int(values.size),
                'n_non_finite': int((~finite).sum()),
                'max_finite': float(np.max(values[finite])) if finite.any() else np.nan,
                'p99_finite': float(np.percentile(values[finite], 99)) if finite.any() else np.nan,
                })
    return pd.DataFrame(rows)


def write_report(path, summary, treatments, coincidences, args, passed):
    verdict = 'PASS' if passed else 'FAIL'
    total_candidates = int(summary['n_candidates'].sum())
    lines = [
        f'# S06 zoo feature matrix: {verdict}',
        '',
        f'*Generated by `mlindex/scripts/run_fom_zoo_features.py`. Do not edit by hand.*',
        '',
        f'- commit `{commit_hash()}`',
        f'- pool `{args.benchmark_dir}`, features `{args.out_dir}`',
        f'- bundles {", ".join(args.bundles)}',
        f'- splits {", ".join(args.splits)} (fom-test is sealed until S15)',
        f'- `g_min` = {args.g_min}, `min_discrepancy` = {args.min_discrepancy}',
        f'- {total_candidates:,} candidates, {len(treatments)} features',
        '',
        '## 1. The round trip, which is the gate',
        '',
        '`compute_all` evaluates `get_M20` last and on a copy, so its M20 must equal the value the',
        'pipeline stored. This is an exactness check on the rebuilt line assignment, not a',
        f'tolerance to be negotiated; it passes at a relative tolerance of {M20_RTOL:g}.',
        '',
        '| bundle | condition | entries | candidates | max rel. dM20 | max abs. dM20 | seconds |',
        '|---|---|---|---|---|---|---|',
        ]
    for row in summary.itertuples():
        lines.append(
            f'| `{row.bundle}` | {row.condition} | {row.n_entries:,} | {row.n_candidates:,} | '
            f'{row.max_M20_relative_error:.2e} | {row.max_M20_absolute_error:.2e} | '
            f'{row.seconds:.0f} |'
            )
    lines += [
        '',
        '## 2. Sigma treatment, per feature',
        '',
        'A column labelled anything but `free` is not sigma-free, and PROTOCOL section 3 rule 4',
        'applies to it: `assumed` columns are reference points only and are never a deliverable,',
        'and `in-sample` columns need a sigma-sensitivity curve and rest on an estimator that is',
        'itself unvalidated (Q7).',
        '',
        '| treatment | features |',
        '|---|---|',
        ]
    for treatment in ('free', 'in-sample', 'assumed'):
        names = sorted(name for name, value in treatments.items() if value == treatment)
        lines.append(f'| `{treatment}` | {", ".join(f"`{n}`" for n in names) or "--"} |')
    lines += [
        '',
        '## 3. The `min_discrepancy` floor, and whether it binds',
        '',
        'F-026: the information-type merits diverge when an observation lands exactly on a',
        'calculated line. The synthetic q2 here are full-precision floats and every evaluable',
        'bundle carries added error, so exact coincidences should be measure-zero -- unlike C0,',
        'where zero error makes them the rule (F-054). Measured rather than assumed:',
        '',
        coincidences.to_markdown(index=False),
        '',
        ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Compute the classical FOM zoo over the frozen benchmark pool (S06 step 1).'
        )
    parser.add_argument('--benchmark-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--out-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_features'))
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--bundles', nargs='+', default=list(EVALUABLE_BUNDLES))
    parser.add_argument('--splits', nargs='+', default=list(DEFAULT_SPLITS))
    parser.add_argument('--g-min', type=float, default=1.0,
                        help='Werner precision floor. Enters V_over_Vcrit and M_werner_frac '
                             'multiplicatively, so any other floor is a rescale of the stored '
                             'columns (Q14).')
    parser.add_argument('--min-discrepancy', type=float, default=0.0,
                        help='Floor on |dQ| for the information-type merits (F-026).')
    parser.add_argument('--n-processes', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Entries per worker payload. Chunked by entry because the '
                             'entrywise sigma estimate is a property of an entry whole pool.')
    parser.add_argument('--calibrate', action='store_true',
                        help='Measure the marginal rate on a small slice and stop, so the full '
                             'run is budgeted rather than guessed at.')
    parser.add_argument('--calibrate-entries', type=int, default=50)
    parser.add_argument('--report-only', action='store_true',
                        help='Regenerate the report and the coincidence audit from the parquets '
                             'already on disk, without rebuilding them.')
    parser.add_argument('--limit-entries', type=int, default=None,
                        help='Build only this many entries per bundle and still write the '
                             'parquet. Used for the C0 singularity audit F-054 asks S06 for, '
                             'where the whole 13.9M-candidate bundle is neither needed nor '
                             'affordable.')
    parser.add_argument('--tag', default='S06_zoo_features')
    args = parser.parse_args()

    root = Path(args.benchmark_dir)
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    entries = FomBenchmark.load_entries(root)
    keep = set(entries.loc[entries['split'].isin(args.splits), 'entry_id'])
    print(f'{len(keep):,} source entries in {"+".join(args.splits)}')

    limit = args.calibrate_entries if args.calibrate else args.limit_entries
    summaries = []
    treatments = {}
    if args.report_only:
        summary = pd.read_csv(artifact_dir/f'{args.tag}.csv')
        treatments = json.loads(
            (artifact_dir/f'{args.tag}_sigma_treatment.json').read_text(encoding='utf-8')
            )
        passed = bool((summary['max_M20_relative_error'] < M20_RTOL).all())
        write_report(artifact_dir/f'{args.tag}.md', summary, treatments,
                     coincidence_audit(out_dir, args.bundles), args, passed)
        print(f'regenerated {args.tag}.md from the parquets on disk')
        return 0 if passed else 1

    for bundle in args.bundles:
        result = build_bundle(
            root, bundle, keep, args.g_min, args.min_discrepancy, args.n_processes,
            args.chunk_size, limit_entries=limit,
            )
        treatments.update(result['treatments'])
        rate = 1e3*result['seconds']/max(result['n_candidates'], 1)
        print(f"{bundle:24s} {result['condition']:3s} "
              f"{result['n_candidates']:9,d} candidates  {result['seconds']:7.1f} s  "
              f"{rate:5.3f} ms/cand  max rel dM20 {result['max_M20_relative_error']:.2e}")
        if not args.calibrate:
            path = out_dir / f'features_{bundle}.parquet'
            result['features'].to_parquet(path, index=False)
        summaries.append({key: value for key, value in result.items()
                          if key not in ('features', 'treatments')})

    summary = pd.DataFrame(summaries)
    passed = bool((summary['max_M20_relative_error'] < M20_RTOL).all())

    if args.calibrate:
        total = summary['seconds'].sum()
        candidates = summary['n_candidates'].sum()
        rate = total/max(candidates, 1)
        # 12.53M candidates over C1-C6; scale by the split fraction actually being built.
        projected = 12.53e6*len(keep)/5922*rate
        print(f'\nmarginal rate {1e3*rate:.3f} ms/candidate on {args.n_processes} processes')
        print(f'projected full build: {projected/60:.0f} min for the six evaluable bundles')
        print(f'round trip {"PASS" if passed else "FAIL"} '
              f'(max rel dM20 {summary["max_M20_relative_error"].max():.2e})')
        return 0 if passed else 1

    coincidences = coincidence_audit(out_dir, args.bundles)
    summary.to_csv(artifact_dir / f'{args.tag}.csv', index=False, encoding='utf-8')
    (artifact_dir / f'{args.tag}_sigma_treatment.json').write_text(
        json.dumps(treatments, indent=2, sort_keys=True), encoding='utf-8',
        )
    write_report(artifact_dir / f'{args.tag}.md', summary, treatments, coincidences, args, passed)

    print(f'\n{summary["n_candidates"].sum():,} candidates, {len(treatments)} features')
    print(f'round trip {"PASS" if passed else "FAIL"} '
          f'(max rel dM20 {summary["max_M20_relative_error"].max():.2e})')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
