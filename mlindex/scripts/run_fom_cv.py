"""S10: build the predictive feature matrix, then ask what it is worth.

Stages, each written so a later one reads the earlier one's artefacts rather than recomputing:

    features   mlindex/data/fom_cv/cv_<bundle>.parquet -- three fold schemes and the hold-out
    main       the leaderboard against M20 and M_sym, paired, CNRS-weighted
    scaling    does the penalty scale with the number of free cell parameters? The gate's second
               condition, and the one that says whether the implementation is right at all
    gate       the symmetry-lowering stratum (primary) and the literal over-prediction stratum
    confound   is it measuring over-fitting capacity, or volume, or M20 in disguise?
    cost       seconds per candidate against get_M20, which is what S14 inherits
    combiner   what does it add to S08's finished 65-feature model

Bounds that belong on every number this writes, and which the report repeats: the pool is censored
at M20 >= 5 (R1); every candidate in it was already Gauss-Newton refined against the peaks it is
then scored on (R10), which is precisely the advantage cross-validation is trying to remove and
means the measured null is not de Wolff's; cubic is scored on ten peaks and everything else on
twenty (R5); and Variant A's hold-out lines were re-synthesised rather than stored, carry no
contaminants, and come from a second noise draw (R13).

    python mlindex/scripts/run_fom_cv.py --stage features --n-processes 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'mlindex', 'scripts'))

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.model_training import FomMetrics  # noqa: E402
from run_fom_zoo_features import commit_hash  # noqa: E402


# is_M20 is de Wolff's own arithmetic on the peaks the cell was fitted to, so it must reproduce the
# M20 the pipeline stored. That is this step's round-trip gate, and it is the only check that the
# fold machinery is scoring the same object the benchmark ranked. Measured at 1.0e-12 on 8 994
# candidates; the tolerance is the same 1e-9 S06's feature build asks for.
M20_RTOL = 1e-9

EVALUABLE_BUNDLES = (
    'error1_cont0',
    'error2_cont0',
    'error1_cont2',
    'error1_cont1_drop6',
    'error1_cont1_drop10',
    'error1_cont0_phase3',
    )

SCHEMES = ('random', 'contiguous', 'high_q')

# The four merits, per scheme, plus Variant A. Direction matters: cv_raw and cv_chi2 are errors and
# rank the other way round from cv_M and cv_tail_nll, which are merits.
CV_MERITS = {
    'cv_M': True,
    'cv_tail_nll': True,
    'cv_raw': False,
    'cv_chi2': False,
    }
HOLDOUT_MERITS = {'ho_M': True, 'ho_tail_nll': True, 'ho_raw': False, 'ho_chi2': False}

# What S06 and S08 measured, so the new merits are quoted against both rather than against M20
# alone -- the standing policy since S07 (STATUS section 6, 2026-08-17 and 2026-08-18).
BASELINES = {'M20': True, 'M_sym': True}


def _cv_chunk(payload):
    """Module-level and picklable, so this survives the spawn start method."""
    candidates, entries, holdout, schemes, n_folds, seed, min_discrepancy, models_directory = payload
    features, treatments = FomBenchmark.cv_features(
        candidates, entries, schemes=schemes, n_folds=n_folds, seed=seed,
        min_discrepancy=min_discrepancy, holdout_peaks=holdout,
        models_directory=models_directory,
        )
    return features, treatments


def _chunks(candidates, entries, holdout, entry_ids, chunk_size, schemes, n_folds, seed,
            min_discrepancy, models_directory):
    """One payload per group of entries.

    Chunked by entry rather than by row for the same reason `zoo_features` is: the entrywise
    residual scale is estimated from an entry's whole pool, so splitting an entry across workers
    would change it.
    """
    needed = [column for column in ('entry_id', 'condition_bundle', 'q2_obs')
              if column in entries.columns]
    by_entry = {entry_id: group for entry_id, group in candidates.groupby('entry_id', sort=False)}
    for start in range(0, len(entry_ids), chunk_size):
        batch = set(entry_ids[start:start + chunk_size])
        frame = pd.concat([by_entry[entry_id] for entry_id in entry_ids[start:start + chunk_size]],
                          ignore_index=False)
        truth = entries.loc[entries['entry_id'].isin(batch), needed]
        extra = None
        if holdout is not None:
            extra = holdout.loc[holdout['entry_id'].isin(batch)]
        yield (frame, truth, extra, schemes, n_folds, seed, min_discrepancy, models_directory)


def build_bundle(args, bundle, keep_entry_ids):
    """The predictive feature matrix for one condition bundle, plus its timing."""
    candidates = FomBenchmark.load_candidates(
        args.benchmark_dir, bundles=[bundle], columns=list(FomBenchmark.CV_CANDIDATE_COLUMNS),
        )
    candidates = candidates.loc[candidates['entry_id'].isin(keep_entry_ids)]
    entries = FomBenchmark.load_entries(args.benchmark_dir)
    entries = entries.loc[entries['condition_bundle'] == bundle]

    holdout = None
    holdout_path = os.path.join(args.out_dir, f'holdout_peaks_{bundle}.parquet')
    if os.path.exists(holdout_path):
        holdout = pd.read_parquet(holdout_path)
        holdout = holdout.loc[holdout['n_holdout'] > 0]
    else:
        print(f'  no hold-out peaks for {bundle}; run run_fom_cv_holdout.py for Variant A')

    entry_ids = list(pd.unique(candidates['entry_id']))
    if args.limit_entries is not None:
        entry_ids = entry_ids[:args.limit_entries]
        candidates = candidates.loc[candidates['entry_id'].isin(set(entry_ids))]
    candidates = candidates.reset_index(drop=True)

    payloads = _chunks(candidates, entries, holdout, entry_ids, args.chunk_size, tuple(args.schemes),
                       args.n_folds, args.seed, args.min_discrepancy, args.models_dir)
    started = time.perf_counter()
    if args.n_processes <= 1:
        results = [_cv_chunk(payload) for payload in payloads]
    else:
        from multiprocessing import Pool
        with Pool(processes=args.n_processes) as pool:
            results = list(pool.imap_unordered(_cv_chunk, payloads, chunksize=1))
    elapsed = time.perf_counter() - started

    features = pd.concat([item[0] for item in results], ignore_index=True)
    treatments = {}
    for item in results:
        treatments.update(item[1])

    residual = np.nan
    if 'is_M20' in features.columns:
        keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
        check = features[keys + ['is_M20']].merge(
            candidates[keys + ['M20']], on=keys, how='inner', validate='1:1')
        stored = check['M20'].to_numpy(dtype=np.float64)
        residual = float(np.max(np.abs(check['is_M20'].to_numpy(dtype=np.float64) - stored)
                                / np.maximum(np.abs(stored), 1e-12)))
    return features, treatments, elapsed, len(entry_ids), residual


def run_features(args, entries):
    keep = set(entries.loc[entries['split'].isin(set(args.splits)), 'entry_id'])
    rows = []
    treatments = {}
    for bundle in args.bundles:
        print(f'{bundle} ...', flush=True)
        features, bundle_treatments, elapsed, n_entries, residual = build_bundle(
            args, bundle, keep)
        treatments.update(bundle_treatments)
        path = os.path.join(args.out_dir, f'cv_{bundle}.parquet')
        features.to_parquet(path, index=False)
        rows.append(dict(bundle=bundle, n_entries=n_entries, n_candidates=int(features.shape[0]),
                         seconds=elapsed,
                         seconds_per_candidate=elapsed/max(features.shape[0], 1),
                         max_M20_relative_error=residual,
                         roundtrip_passed=bool(residual <= M20_RTOL)))
        print(f'  {features.shape[0]} candidates in {elapsed:.1f}s; M20 round trip '
              f'{residual:.2e} -> {path}', flush=True)
    table = pd.DataFrame(rows)
    if not table['roundtrip_passed'].all():
        print('WARNING: the is_M20 round trip failed; the fold machinery is not scoring the '
              'object the benchmark ranked. Do not report these numbers.')
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_build.csv'), index=False)
    with open(os.path.join(args.artifact_dir, f'{args.tag}_treatments.json'),
              'w', encoding='utf-8') as handle:
        json.dump(treatments, handle, indent=2, sort_keys=True)
    return table


def main():
    parser = argparse.ArgumentParser(description='S10 cross-validated and hold-out figures of merit.')
    parser.add_argument('--stage', default='features',
                        choices=['features', 'main', 'scaling', 'gate', 'confound', 'cost', 'combiner'])
    parser.add_argument('--benchmark-dir',
                        default=os.path.join('mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--feature-dir', default=os.path.join('mlindex', 'data', 'fom_features'))
    parser.add_argument('--out-dir', default=os.path.join('mlindex', 'data', 'fom_cv'))
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--null-dir', default=os.path.join('mlindex', 'models', 'fom_null'))
    parser.add_argument('--combiner-models-dir', dest='models_dir_combiner',
                        default=os.path.join('mlindex', 'models'))
    parser.add_argument('--bundles', nargs='+', default=list(EVALUABLE_BUNDLES))
    parser.add_argument('--schemes', nargs='+', default=list(SCHEMES))
    parser.add_argument('--splits', nargs='+', default=['fom-train', 'fom-dev'],
                        help='fom-test is sealed until S15')
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--min-discrepancy', type=float, default=0.0,
                        help='floor on |dQ|; F-064 measured that zero does not bind here')
    parser.add_argument('--n-processes', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--limit-entries', type=int, default=None)
    parser.add_argument('--cost-entries', type=int, default=6)
    parser.add_argument('--cost-repeats', type=int, default=3)
    parser.add_argument('--tag', default='S10_cv')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)
    entries = FomBenchmark.load_entries(args.benchmark_dir)

    started = time.perf_counter()
    if args.stage == 'features':
        run_features(args, entries)
    else:
        import run_fom_cv_analysis as analysis
        analysis.dispatch(args, entries)
    print(f'stage {args.stage} finished in {time.perf_counter() - started:.1f}s')


if __name__ == '__main__':
    main()
