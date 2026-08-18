"""S07: fit each merit's incorrect-candidate null, then re-rank the frozen pool by `-log p`.

Fits on `fom-train` and reports on `fom-dev`, through `FomMetrics.evaluate` -- the only route to a
number in this project -- and writes the fitted nulls as npz under `mlindex/models/fom_null/`.

The premise is F-074's measurement, not an assumption: within a Bravais lattice the whole zoo scores
0.44-0.685 on top-10 while across lattices it spreads 0.000-0.618, so most of S06's leaderboard is
scale transfer. This asks how much of that gap a null calibration closes, per merit and per method.

Four things about the protocol, each of which is a finding this script was written around:

  * **The pool and the fit share their censoring.** `prune_below_m20` keeps the better-fitting wrong
    cells, lattice-dependently (R1, F-049/F-068), so every fitted null here partly learns the prune
    boundary. `analytic` alone is immune, because nothing is fitted. The bound is printed with the
    results and stated in the results document, not only in a finding.
  * **Thresholds are selected on `fom-train` and applied unchanged**, with `check_threshold_transfer`
    enforcing it, and every merit is additionally reported at the matched false-positive budget
    M20 incurs at de Wolff's threshold of 10 -- the same convention as F-066/F-067, so these numbers
    sit directly beside S06's.
  * **The hard stratum is reported twice.** Rank metrics on the literal `{mP,mC,aP} x decile>=8`
    definition; threshold metrics at decile >= 6, because at decile >= 8 `fom-dev` holds 16
    reachable entries and every merit scores exactly 0.0000 (F-063). Q32, resolved by DWMM.
  * **Per-lattice deltas are always reported**, because an aggregate gain that is really the model
    learning "triclinic candidates are usually wrong" is the anti-pattern this step is most exposed
    to (PROTOCOL section 10).

    python mlindex/scripts/run_fom_null.py

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.model_training import FomMetrics  # noqa: E402
from mlindex.model_training import FomNull  # noqa: E402
from mlindex.scripts.run_fom_zoo_eval import DEWOLFF_THRESHOLD  # noqa: E402
from mlindex.scripts.run_fom_zoo_eval import MERITS, POOL_COLUMNS  # noqa: E402
from mlindex.scripts.run_fom_zoo_features import EVALUABLE_BUNDLES, commit_hash  # noqa: E402

# Q32, resolved by DWMM 2026-08-17: hard-stratum *threshold* metrics are reported at this volume
# cut, because the literal decile >= 8 stratum has 16 reachable entries on `fom-dev` and cannot
# produce them at all (F-063). Rank metrics stay on the literal definition.
HARD_THRESHOLD_DECILE = 6

# `binned` and `gbm` cost roughly ten times what `rank` costs to fit and are bounded by R1 either
# way, so they run on the merits where the transfer gap is largest plus M20 and M_sym as controls,
# rather than over the whole zoo. Stated here rather than left implicit in a runtime flag.
CONDITIONAL_MERITS = (
    'M20', 'M_sym', 'M_rev', 'M_tilde', 'Minfo', 'M_info_clipped', 'null_tail_nll', 'M_star',
    'M_star_corrected', 'M_1', 'M_wu', 'n_over', 'max_gap', 'M_werner_frac', 'chi2_fixed', 'bic',
    )

# The look-elsewhere correction is a separate claim from the standardisation and is reported on a
# representative set rather than on all twenty-one: two merits whose raw scale already transfers
# (M20, M_sym), two whose scale is the problem (null_tail_nll, Minfo), the over-prediction pair
# (M_rev, M_1) and the two that rank well and score not at all (n_over, max_gap, F-066).
TRIALS_MERITS = ('M20', 'M_sym', 'M_rev', 'M_1', 'Minfo', 'null_tail_nll', 'n_over', 'max_gap')

# Columns the nulls condition on, over and above the merit itself.
COVARIATE_COLUMNS = ('n_peaks', 'spacegroup', 'volume', 'V_over_Vcrit', 'n_entering')


def method_specs(merits, methods, trials_column=None):
    """(merit, method, trials, higher_is_better) for everything that will be fitted.

    A method is skipped rather than approximated when it does not apply: `analytic` and
    `analytic_rho` need a closed-form null and only two merits in the zoo have one, and the table
    says so instead of substituting something else.

    Every method is fitted twice when `trials_column` is set -- with and without the look-elsewhere
    correction -- because the pair is the only way to attribute a gain to the right half. The null
    standardisation and the trial-count correction are separate claims and they are reported
    separately.
    """
    specs = []
    for name, higher, _source, _note in merits:
        for method in methods:
            if method in ('analytic', 'analytic_rho') and name not in FomNull.NULL_LAWS:
                continue
            if method in ('binned', 'gbm') and name not in CONDITIONAL_MERITS:
                continue
            specs.append((name, method, None, higher))
            # The look-elsewhere factor turns P(one draw >= x) into P(best of m draws >= x), so it
            # is only defined on a score that *is* a tail probability. `z` is not one -- it is a
            # standardised merit that can be negative -- so a trial-corrected z-score would be
            # arithmetic on a quantity that has no probability interpretation. `gbm` is excluded
            # for cost, not for meaning.
            if (trials_column and name in TRIALS_MERITS
                    and method not in ('gbm', 'z')):
                specs.append((name, method, trials_column, higher))
    return specs


def load_split(benchmark_dir, feature_dir, bundles, keep_entry_ids, merit_names):
    """Every bundle of one split, in memory, with only the columns S07 reads.

    Held rather than streamed because the alternative is one parquet read per (merit, method,
    evaluation) -- a hundred-odd fits and three hundred-odd evaluations against 2.3 GB of features,
    which is hours of I/O for no benefit. The projection is what makes it affordable: the six
    evaluable bundles are ten million candidates, and this keeps ~30 columns of them.
    """
    keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
    wanted = sorted(set(merit_names) | set(COVARIATE_COLUMNS))
    frames = []
    for bundle in bundles:
        pool = FomBenchmark.load_candidates(
            benchmark_dir, bundles=[bundle], columns=list(POOL_COLUMNS),
            )
        pool = pool.loc[pool['entry_id'].isin(keep_entry_ids)]
        if not pool.shape[0]:
            continue
        missing = [column for column in wanted if column not in pool.columns]
        if missing:
            features = pd.read_parquet(
                Path(feature_dir)/f'features_{bundle}.parquet', columns=keys + missing,
                )
            pool = pool.merge(features, on=keys, how='inner', validate='1:1')
        if pool.shape[0]:
            frames.append(pool.reset_index(drop=True))
    return frames


def fit_nulls(frames, specs, models_dir, min_count, gbm_sample, seed, reuse=False):
    """One fitted null per (merit, method), saved as npz and returned in memory."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for merit, method, trials, higher in specs:
        suffix = f'{method}_trials' if trials else method
        path = models_dir/f'{merit}_{suffix}.npz'
        started = time.perf_counter()
        if reuse and path.exists():
            fitted[(merit, method, trials)] = FomNull.FomNull.load(path)
            print(f'  {merit:18s} {suffix:20s} loaded')
            continue
        # The trial-corrected variant differs from its parent only at apply time, so it reuses the
        # fitted tables rather than repeating the fit.
        parent = fitted.get((merit, method, None))
        if trials and parent is not None:
            null = FomNull.FomNull(
                merit=merit, method=method, higher_is_better=higher, tables=parent.tables,
                counts=parent.counts, binner=parent.binner, pooled_from=parent.pooled_from,
                rho=parent.rho, tail=parent.tail, meta=dict(parent.meta, trials=trials),
                trials=trials,
                )
        else:
            null = FomNull.FomNull.fit(
                frames, merit, method, higher_is_better=higher, min_count=min_count,
                gbm_sample=gbm_sample, seed=seed, trials=trials,
                )
        null.save(path)
        fitted[(merit, method, trials)] = null
        print(f'  {merit:18s} {suffix:20s} groups {len(null.tables) + len(null.rho):5d}  '
              f'null {null.meta["n_null"]:,}  ({time.perf_counter() - started:.0f} s)')
    return fitted


def evaluate_score(frames, entries, score, higher_is_better, threshold, score_columns, strata,
                   n_bootstrap, seed, split_label, pool='cross_bl',
                   hard_min_decile=FomMetrics.HARD_MIN_DECILE):
    return FomMetrics.evaluate(
        frames, score=score, higher_is_better=higher_is_better, threshold=threshold,
        score_columns=score_columns, entries=entries, strata=strata, split=split_label,
        n_bootstrap=n_bootstrap, seed=seed, pool=pool, hard_min_decile=hard_min_decile,
        )


def per_lattice(result, prefix):
    """Per-true-lattice top-10, so a gain that is really a base-rate shift is visible.

    Per-lattice rows are never CNRS-reweighted -- that would be circular -- so these are the raw
    within-lattice numbers and they are what a "did it help everywhere or only where the prior
    does the work" question is answered on.
    """
    rows = result.stratum('bravais_lattice')
    return {f'{prefix}_{level}': float(value)
            for level, value in zip(rows['level'], rows['top10'])}


def main():
    parser = argparse.ArgumentParser(
        description='Fit and evaluate the S07 null calibrations on the frozen benchmark pool.'
        )
    parser.add_argument('--benchmark-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--feature-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_features'))
    parser.add_argument('--models-dir',
                        default=os.path.join(BASE, 'mlindex', 'models', 'fom_null'))
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--bundles', nargs='+', default=list(EVALUABLE_BUNDLES))
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--merits', nargs='+', default=None)
    # The default is the three normalisers S07 kept (DWMM, STATUS section 6). `binned` and `gbm`
    # are still implemented and still reproduce the comparison that retired them -- pass them
    # explicitly -- but they cost 4.0 MB and 3.2 MB per merit against `z`'s 5.8 kB, they are
    # R1-bounded where `analytic` is not, and they beat `z` only on merits whose operating point is
    # at or below 0.03 (F-077). Running them by default would regenerate 131 MB for nothing.
    parser.add_argument('--methods', nargs='+', default=['analytic', 'analytic_rho', 'z', 'rank'])
    parser.add_argument('--min-count', type=int, default=FomNull.MIN_GROUP_COUNT)
    parser.add_argument('--gbm-sample', type=int, default=300000)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--trials-column', default='n_entering',
                        help=('Candidate column holding the number of cells tried for that '
                              '(entry, lattice); pass an empty string to skip the look-elsewhere '
                              'correction entirely.'))
    parser.add_argument('--reuse-fits', action='store_true')
    parser.add_argument('--tag', default='S07_null')
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    merits = [row for row in MERITS if args.merits is None or row[0] in set(args.merits)]
    merit_names = [row[0] for row in merits]
    specs = method_specs(merits, args.methods, trials_column=args.trials_column or None)

    entries = FomBenchmark.load_entries(args.benchmark_dir)
    train_ids = set(entries.loc[entries['split'] == args.train_split, 'entry_id'])
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])
    print(f'{len(train_ids):,} {args.train_split} / {len(dev_ids):,} {args.report_split} '
          f'source entries; {len(specs)} (merit, method) pairs')

    started = time.perf_counter()
    train = load_split(args.benchmark_dir, args.feature_dir, args.bundles, train_ids, merit_names)
    dev = load_split(args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, merit_names)
    print(f'loaded {sum(f.shape[0] for f in train):,} train and {sum(f.shape[0] for f in dev):,} '
          f'dev candidates in {time.perf_counter() - started:.0f} s')

    # ------------------------------------------------------------------ the nulls
    print('\nfitting')
    fitted = fit_nulls(train, specs, args.models_dir, args.min_count, args.gbm_sample, args.seed,
                       reuse=args.reuse_fits)

    # ------------------------------------------------------------------ the matched budget
    # M20's own false-positive rate at de Wolff's threshold on fom-train, exactly as S06 chose it,
    # so every operating point below is comparable with F-066's table and with every other merit.
    m20_train = evaluate_score(train, entries, 'M20', True, DEWOLFF_THRESHOLD, (), (), 0,
                               args.seed, args.train_split)
    budget = float(m20_train.metric('false_positive'))
    print(f'matched false-positive budget: {budget:.4f}')

    strata = ('bravais_lattice', 'volume_decile', 'condition_bundle')
    rows = []
    results = {}
    for label, merit, method, trials, higher in (
            [(f'{name} (raw)', name, None, None, higher) for name, higher, _s, _n in merits]
            + [(f'{merit}/{method}{"+trials" if trials else ""}', merit, method, trials, higher)
               for merit, method, trials, higher in specs]):
        started = time.perf_counter()
        if method is None:
            score, score_columns, oriented = merit, (), higher
        else:
            null = fitted[(merit, method, trials)]
            score, score_columns, oriented = null.apply, null.required_columns, True

        train_result = evaluate_score(train, entries, score, oriented, None, score_columns, (),
                                      0, args.seed, args.train_split)
        choice = FomMetrics.select_threshold(train_result, objective='youden')
        threshold = choice.threshold if oriented else -choice.threshold
        budgeted_choice = FomMetrics.select_threshold(
            train_result, objective='operating_point', max_false_positive_rate=budget,
            )
        budgeted_threshold = (budgeted_choice.threshold if oriented
                              else -budgeted_choice.threshold)

        dev_result = evaluate_score(dev, entries, score, oriented, threshold, score_columns,
                                    strata, args.n_bootstrap, args.seed, args.report_split)
        FomMetrics.check_threshold_transfer(choice, dev_result)
        budgeted = evaluate_score(dev, entries, score, oriented, budgeted_threshold,
                                  score_columns, (), 0, args.seed, args.report_split)
        FomMetrics.check_threshold_transfer(budgeted_choice, budgeted)
        per_bl = evaluate_score(dev, entries, score, oriented, None, score_columns, (), 0,
                                args.seed, args.report_split, pool='per_bl')
        widened = evaluate_score(dev, entries, score, oriented, budgeted_threshold,
                                 score_columns, (), 0, args.seed, args.report_split,
                                 hard_min_decile=HARD_THRESHOLD_DECILE)

        results[label] = dev_result
        row = dict(
            label=label, merit=merit, method=method or 'raw', trials=trials or '',
            r1_bounded=bool(method not in (None, 'analytic')),
            operating_point=dev_result.metric('operating_point'),
            operating_point_matched_fpr=budgeted.metric('operating_point'),
            false_positive_rate_matched=budgeted.metric('false_positive'),
            top10=dev_result.metric('top10'),
            top10_per_bl=per_bl.metric('top10'),
            top1=dev_result.metric('top1'),
            mrr=dev_result.metric('mrr'),
            threshold_only=dev_result.metric('threshold_only'),
            ceiling_rescorer=dev_result.metric('ceiling_rescorer'),
            reported=dev_result.metric('reported'),
            false_positive_rate=dev_result.metric('false_positive'),
            abstains_always=bool(dev_result.metric('reported') == 0.0),
            hard_top10=dev_result.metric('top10', scope='hard'),
            hard_n_entries=dev_result.metric('n_entries', scope='hard'),
            hard_op_given_found_d6=widened.metric('operating_point_given_found', scope='hard'),
            hard_op_d6=widened.metric('operating_point', scope='hard'),
            hard_n_entries_d6=widened.metric('n_entries', scope='hard'),
            threshold=float(threshold), threshold_matched_fpr=float(budgeted_threshold),
            seconds=time.perf_counter() - started,
            )
        if method is not None:
            row.update({f'diag_{key}': value
                        for key, value in fitted[(merit, method, trials)].diagnostics.items()})
        row.update(per_lattice(dev_result, 'top10_bl'))
        rows.append(row)
        print(f'  {label:32s} op@fpr {row["operating_point_matched_fpr"]:.4f}  '
              f'top10 {row["top10"]:.4f}  per-BL {row["top10_per_bl"]:.4f}  '
              f'({row["seconds"]:.0f} s)')

    table = pd.DataFrame(rows)

    # Gains against the two baselines that matter: the gate's (raw M20) and the incumbent's
    # (M_sym, S06's winner). Quoted as a fraction of headroom as well as in points, because
    # percentage points are not the unit (F-042).
    for baseline in ('M20 (raw)', 'M_sym (raw)'):
        if baseline not in set(table['label']):
            continue
        reference = table.loc[table['label'] == baseline].iloc[0]
        suffix = baseline.split()[0]
        table[f'delta_op_vs_{suffix}'] = (table['operating_point_matched_fpr']
                                          - reference['operating_point_matched_fpr'])
        headroom = reference['ceiling_rescorer'] - reference['operating_point_matched_fpr']
        table[f'headroom_fraction_vs_{suffix}'] = table[f'delta_op_vs_{suffix}']/max(headroom, 1e-9)
    table.to_csv(artifact_dir/f'{args.tag}_main_table.csv', index=False, encoding='utf-8')

    # ------------------------------------------------------------------ paired comparisons
    print('\npaired comparisons (McNemar, same entries)')
    comparisons = []
    for label in results:
        for baseline in ('M20 (raw)', 'M_sym (raw)'):
            if label == baseline or baseline not in results:
                continue
            for scope in (None, 'hard'):
                try:
                    test = FomMetrics.mcnemar(results[baseline], results[label],
                                              metric='operating_point', subset=scope)
                except ValueError as error:
                    comparisons.append(dict(label=label, baseline=baseline,
                                            subset=scope or 'all', error=str(error)))
                    continue
                comparisons.append(dict(test, label=label, baseline=baseline,
                                        subset=scope or 'all'))
    pd.DataFrame(comparisons).to_csv(artifact_dir/f'{args.tag}_mcnemar.csv', index=False,
                                     encoding='utf-8')

    # ------------------------------------------------------------------ the gate's second half
    print('\nnull homogeneity across lattices and extinction groups')
    homogeneity = []
    for merit, method, trials, higher in (
            [(name, None, None, higher) for name, higher, _s, _n in merits] + specs):
        if method is None:
            def score(frame, name=merit, oriented=higher):
                values = frame[name].to_numpy(dtype=np.float64)
                return values if oriented else -values
        else:
            score = fitted[(merit, method, trials)].apply
        for by in ('bravais_lattice', 'spacegroup'):
            table_h = FomNull.null_homogeneity(dev, score, by=by)
            table_h.insert(0, 'method', (method or 'raw') + ('+trials' if trials else ''))
            table_h.insert(0, 'merit', merit)
            homogeneity.append(table_h)
    pd.concat(homogeneity, ignore_index=True).to_csv(
        artifact_dir/f'{args.tag}_homogeneity.csv', index=False, encoding='utf-8')

    # ------------------------------------------------------------------ de Wolff's regularity ratio
    regularity = []
    for merit in FomNull.NULL_LAWS:
        if merit not in merit_names:
            continue
        for by in (('bravais_lattice',), ('bravais_lattice', 'extinction_group')):
            frame = FomNull.regularity_ratio(train, merit=merit, by=by)
            frame.insert(0, 'merit', merit)
            frame.insert(1, 'grouping', '+'.join(by))
            regularity.append(frame)
    if regularity:
        pd.concat(regularity, ignore_index=True).to_csv(
            artifact_dir/f'{args.tag}_regularity.csv', index=False, encoding='utf-8')

    # ------------------------------------------------------------------ pooling and clamping
    pooling = []
    for (merit, method, trials), null in fitted.items():
        levels = pd.Series([entry['level'] for entry in null.pooled_from.values()])
        pooling.append(dict(
            merit=merit, method=method, trials=trials or '',
            n_groups=len(null.tables) + len(null.rho),
            n_finest_labels=len(null.pooled_from),
            n_pooled=int((levels < levels.max()).sum()) if levels.size else 0,
            **{f'n_at_level_{level}': int(count) for level, count in
               (levels.value_counts().items() if levels.size else [])},
            **null.diagnostics,
            ))
    pd.DataFrame(pooling).to_csv(artifact_dir/f'{args.tag}_pooling.csv', index=False,
                                 encoding='utf-8')

    meta = dict(
        commit=commit_hash(), bundles=list(args.bundles), train_split=args.train_split,
        report_split=args.report_split, seed=args.seed, n_bootstrap=args.n_bootstrap,
        matched_fpr_budget=budget, hard_threshold_decile=HARD_THRESHOLD_DECILE,
        min_count=args.min_count, gbm_sample=args.gbm_sample, trials_column=args.trials_column,
        models_dir=str(args.models_dir),
        n_train_candidates=int(sum(f.shape[0] for f in train)),
        n_dev_candidates=int(sum(f.shape[0] for f in dev)),
        conditional_merits=list(CONDITIONAL_MERITS),
        r1_note=('Every method except `analytic` is fitted on negatives censored at M20 >= 5 by '
                 '`prune_below_m20`, lattice-dependently (R1, F-049/F-068), so it partly learns '
                 'the prune boundary. Provisional against a Benchmark B dumped at threshold 0.'),
        )
    with open(artifact_dir/f'{args.tag}_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print(f'\nwrote {args.tag}_* to {artifact_dir}')


if __name__ == '__main__':
    main()
