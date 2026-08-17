"""S06 steps 2-4: the loss decomposition, the leaderboard, and the oracle gap.

Reads the feature matrix `run_fom_zoo_features.py` wrote, re-ranks the frozen pool by each merit
in turn, and reports through `FomMetrics.evaluate` -- which is the only route to a number in this
project, so nothing here re-implements a metric.

Three things about the protocol, each of which is a finding the handoff was written before:

  * **The loss decomposition is reported first, and by stratum.** It orders S07 against S08. The
    aggregate says the loss is almost purely a threshold failure (F-061: re-ranking is worth
    0.1-0.8 pp against re-scoring's 23-48), but in the hard region 63% of reachable cases also
    have the correct cell outside the top ten, so both halves fail together and land in
    `lost_both` (F-062). Reported once, that inverts the conclusion.
  * **Thresholds cannot be chosen by maximising the operating point** -- it is monotone in the
    threshold, so the maximiser is minus infinity (F-060). `select_threshold`'s default objective
    is `operating_point - false_positive_rate`, chosen on `fom-train` and applied unchanged to
    `fom-dev`, with the false-positive rate reported beside every threshold.
  * **The hard stratum's headline is `operating_point_given_found`** (Q30, resolved by DWMM):
    87.1% of it has no correct candidate in the pool, so the target is the reachable ~10% and the
    unconditional number is context. Its effective sample is 104 source entries, so every hard row
    carries a bootstrap interval and a small difference is not a difference.

    python mlindex/scripts/run_fom_zoo_eval.py

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
from mlindex.scripts.run_fom_zoo_features import EVALUABLE_BUNDLES  # noqa: E402
from mlindex.scripts.run_fom_zoo_features import commit_hash  # noqa: E402

# The literature's own threshold, kept as its own row so the comparison to the paper is direct.
DEWOLFF_THRESHOLD = 10.0

# (name, higher_is_better, source, note). `source` is 'pool' for a column the dump already
# carries and 'features' for one this project computes. Orientation is taken from
# run_fom_li6b4o9.py, which is where the zoo's direction-of-merit was first pinned.
MERITS = (
    ('M20', True, 'pool', 'de Wolff 1968 -- the baseline'),
    ('Minfo', True, 'pool', 'the repo Taupin-style likelihood merit'),
    ('M_tilde', True, 'features', 'Oishi-Tomiyasu (i)+(ii): N_cal and a restricted range'),
    ('M_rev', True, 'features', 'O-T (iii) reversed -- the over-prediction penalty'),
    ('M_sym', True, 'features', 'O-T symmetric, M_tilde * M_rev'),
    ('M_wu', True, 'features', 'Wu 1988 exact mean gap; continuous under perturbation'),
    ('M_star', True, 'features', 'Wu S/(V^2/3 delta); no line counting at all'),
    ('M_star_corrected', True, 'features', 'M_star with Wu corrected symmetry factors'),
    ('M_1', True, 'features', 'Shirley 1980 local per-line epsilon'),
    ('M_nn', True, 'features', 'O-T/Tanaka/Nakagawa 2021 nearest-neighbour form'),
    ('M_info_clipped', True, 'features', 'Taupin eq. 25, neighbour-clipped'),
    ('null_tail_nll', True, 'features', "de Wolff 1961 analytic null tail -- S07's backbone"),
    ('F_N_q', True, 'features', 'Smith & Snyder 1979 in q-space (no wavelength here)'),
    ('M_werner_frac', True, 'features', 'Werner 1976 M_N/M_N,max; ranking is g_min-invariant'),
    ('X_N', False, 'features', 'de Wolff unindexed-line count'),
    ('n_over', False, 'features', 'calculated lines in range with no observation nearby'),
    ('max_gap', False, 'features', 'longest run of unaccounted calculated lines'),
    ('bic', False, 'features', 'explicit complexity penalty'),
    ('chi2_entrywise', False, 'features', 'sigma from the entry own pool -- IN-SAMPLE, Q7'),
    ('chi2_fixed', False, 'features', "the repo's global sigma -- ASSUMED, reference only"),
    ('nll_exponential', False, 'features', 'NEGATIVE CONTROL: not a merit, ranks backwards'),
    )

# Descriptive columns kept for the diagnostics in run_fom_zoo_explain.py; never ranked here.
DESCRIPTIVE = (
    'N_cal', 'delta_dewolff61', 'n_dewolff61', 'zone_dominance', 'V_over_Vcrit',
    'M_werner_max', 'chi2_taupin_scale', 'chi2_fixed_pvalue', 'chi2_entrywise_pvalue',
    )

# `volume_ratio_to_truth` is a stored *label*, not a feature (F-055), so it comes off the candidate
# shard rather than the feature matrix. The over-prediction diagnostic needs it and would otherwise
# look for it in the wrong parquet.
POOL_COLUMNS = tuple(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + (
    'Minfo', 'M20', 'volume', 'lattice_system', 'volume_ratio_to_truth',
    )


def werner_strict(frame):
    """Werner 1976's strengthened criterion, as a score rather than a second threshold.

    de Wolff allowed up to two unindexed lines below Q20; Werner claims that if *all* of them are
    indexed then M20 > 10 guarantees substantially correct indexing. That is a conjunction, so it
    is expressed as M20 for candidates explaining every line and minus infinity for the rest --
    the threshold then does the second half of the work and the usual machinery applies unchanged.
    """
    values = frame['M20'].to_numpy(dtype=np.float64).copy()
    values[frame['X_N'].to_numpy(dtype=np.float64) > 0] = -np.inf
    return values


def bundle_frames(benchmark_dir, feature_dir, bundles, keep_entry_ids, feature_columns):
    """Yield one merged (pool + features) frame per bundle.

    A generator rather than a cached list: `evaluate` consumes shards one at a time, and holding
    six bundles of the pool joined to thirty feature columns is several GB for no benefit. The
    cost is one parquet read per bundle per merit, which is under a second.
    """
    keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
    for bundle in bundles:
        pool = FomBenchmark.load_candidates(
            benchmark_dir, bundles=[bundle], columns=list(POOL_COLUMNS),
            )
        wanted = [column for column in feature_columns if column not in pool.columns]
        if wanted:
            features = pd.read_parquet(
                Path(feature_dir)/f'features_{bundle}.parquet', columns=keys + wanted,
                )
            # Inner, because the feature matrix covers fom-train and fom-dev only: fom-test is
            # sealed until S15 and was never computed. The join is 1:1 on the four keys.
            pool = pool.merge(features, on=keys, how='inner', validate='1:1')
        pool = pool.loc[pool['entry_id'].isin(keep_entry_ids)]
        if pool.shape[0]:
            yield pool.reset_index(drop=True)


def evaluate_merit(benchmark_dir, feature_dir, bundles, keep_entry_ids, entries, score,
                   higher_is_better, threshold, score_columns, strata, n_bootstrap, seed,
                   split_label):
    """One merit, one split. The split filter is applied here, not by `evaluate`.

    `evaluate` filters by split only when it is handed a benchmark *root*; given an iterable of
    shards it takes them as final. `split=` is still passed so the choice is recorded in the
    result's metadata and `check_threshold_transfer` has something to report.
    """
    needed = list(score_columns) + ([score] if isinstance(score, str) else [])
    shards = bundle_frames(benchmark_dir, feature_dir, bundles, keep_entry_ids, needed)
    return FomMetrics.evaluate(
        shards, score=score, higher_is_better=higher_is_better, threshold=threshold,
        entries=entries, strata=strata, split=split_label, n_bootstrap=n_bootstrap, seed=seed,
        )


def measure_cost(benchmark_dir, bundle='error1_cont0', n_entries=3, repeats=3):
    """Seconds per candidate for each merit, so the main table has a cost column.

    S01 measured only `get_M20` and `compute_all` as a whole and recorded that S14 would need the
    per-function numbers; this is them. Measured on real candidate pools from the frozen pool
    rather than on a synthetic one, so the reference-line lengths are the ones production sees.
    """
    from mlindex.utilities import FigureOfMerits as fom

    candidates = FomBenchmark.load_candidates(
        benchmark_dir, bundles=[bundle], columns=list(FomBenchmark.ZOO_CANDIDATE_COLUMNS),
        )
    entries = FomBenchmark.load_entries(benchmark_dir)
    entries = entries.loc[entries['condition_bundle'] == bundle]
    peaks = entries.set_index('entry_id')['q2_obs']

    cases = []
    for entry_id in pd.unique(candidates['entry_id'])[:n_entries]:
        group = candidates.loc[(candidates['entry_id'] == entry_id)
                               & (candidates['bravais_lattice'] == 'mP')]
        if not group.shape[0]:
            continue
        for spacegroup, chunk in group.groupby('spacegroup', sort=False):
            q2_obs = np.asarray(peaks.loc[entry_id], dtype=np.float64)[:int(chunk['n_peaks'].iloc[0])]
            xnn = np.vstack([np.asarray(v, dtype=np.float64) for v in chunk['xnn']])
            q2_ref_calc, _, _, q2_calc = FomBenchmark.assign_lines(
                q2_obs, xnn, 'monoclinic', 'mP', spacegroup,
                )
            cases.append((q2_obs, q2_calc, q2_ref_calc, xnn))

    n_candidates = sum(case[3].shape[0] for case in cases)
    calls = {
        'M20': lambda o, c, r, x: fom.get_M20(o, c, r.copy()),
        'Minfo': lambda o, c, r, x: fom.get_M20_likelihood(
            o, c, 'mP', np.ones(c.shape[0])),
        'M_tilde': lambda o, c, r, x: fom.get_M_rev_sym(o, c, r),
        'M_rev': lambda o, c, r, x: fom.get_M_rev_sym(o, c, r),
        'M_sym': lambda o, c, r, x: fom.get_M_rev_sym(o, c, r),
        'M_wu': lambda o, c, r, x: fom.get_M_wu(o, c, r),
        'M_star': lambda o, c, r, x: fom.get_M_star(o, c, np.ones(c.shape[0]), 'monoclinic'),
        'M_star_corrected': lambda o, c, r, x: fom.get_M_star(
            o, c, np.ones(c.shape[0]), 'monoclinic', corrected=True),
        'M_1': lambda o, c, r, x: fom.get_M_1(o, c, r),
        'M_nn': lambda o, c, r, x: fom.get_M_nn(o, c, r),
        'M_info_clipped': lambda o, c, r, x: fom.get_M_info_clipped(
            o, c, x, 'monoclinic', 'mP'),
        'null_tail_nll': lambda o, c, r, x: fom.get_null_tail_nll(o, c, x, 'monoclinic', 'mP'),
        'F_N_q': lambda o, c, r, x: fom.get_F_N(o, c, r),
        'X_N': lambda o, c, r, x: fom.get_X_N(o, c, r),
        'n_over': lambda o, c, r, x: fom.get_n_over(o, c, r),
        'max_gap': lambda o, c, r, x: fom.get_n_over(o, c, r),
        'bic': lambda o, c, r, x: fom.get_bic(o, c, x, 'monoclinic', 'mP'),
        'chi2_fixed': lambda o, c, r, x: fom.get_chi2(o, c, 'monoclinic', variant='fixed'),
        'chi2_entrywise': lambda o, c, r, x: fom.get_chi2(
            o, c, 'monoclinic', sigma=1e-3, variant='entrywise'),
        'nll_exponential': lambda o, c, r, x: fom.get_nll_exponential(
            o, c, x, 'monoclinic', 'mP'),
        'M_werner_frac': lambda o, c, r, x: fom.get_V_over_Vcrit(
            np.ones(c.shape[0]), 1/np.sqrt(np.maximum(c[:, -1], 1e-300)), 1.0, 2),
        }
    rows = []
    baseline = None
    for name, call in calls.items():
        best = np.inf
        for _ in range(repeats):
            start = time.perf_counter()
            for case in cases:
                call(*case)
            best = min(best, time.perf_counter() - start)
        per_candidate = best/max(n_candidates, 1)
        if name == 'M20':
            baseline = per_candidate
        rows.append({'merit': name, 'seconds_per_candidate': per_candidate})
    frame = pd.DataFrame(rows)
    frame['cost_vs_M20'] = frame['seconds_per_candidate']/baseline
    frame['n_candidates_timed'] = n_candidates
    return frame


def main():
    parser = argparse.ArgumentParser(
        description='Rank the classical FOM zoo on the frozen benchmark pool (S06 steps 2-4).'
        )
    parser.add_argument('--benchmark-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--feature-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_features'))
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--bundles', nargs='+', default=list(EVALUABLE_BUNDLES))
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--merits', nargs='+', default=None,
                        help='Restrict to these merits, for a quick pass.')
    parser.add_argument('--skip-cost', action='store_true')
    parser.add_argument('--tag', default='S06_zoo')
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    entries = FomBenchmark.load_entries(args.benchmark_dir)
    train_ids = set(entries.loc[entries['split'] == args.train_split, 'entry_id'])
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])
    print(f'{len(train_ids):,} {args.train_split} / {len(dev_ids):,} {args.report_split} '
          f'source entries')

    merits = [row for row in MERITS if args.merits is None or row[0] in set(args.merits)]
    strata = ('bravais_lattice', 'volume_decile', 'condition_bundle', 'dominant_zone')

    # ------------------------------------------------------------------ step 2: loss decomposition
    print('\nstep 2: the loss decomposition for M20, by stratum')
    m20_dev = evaluate_merit(
        args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries, 'M20', True,
        DEWOLFF_THRESHOLD, (), strata, args.n_bootstrap, args.seed, args.report_split,
        )
    m20_dev.loss.to_csv(artifact_dir/f'{args.tag}_loss_decomposition.csv', index=False,
                        encoding='utf-8')
    m20_dev.by_stratum.to_csv(artifact_dir/f'{args.tag}_m20_by_stratum.csv', index=False,
                              encoding='utf-8')
    for scope in ('all', 'hard'):
        row = m20_dev.loss.loc[m20_dev.loss['stratum'] == scope]
        if row.shape[0]:
            row = row.iloc[0]
            print(f'  {scope:5s} op {row["operating_point"]:.4f}  '
                  f'rank {row["share_rank_failure"]:.3f}  '
                  f'threshold {row["share_threshold_failure"]:.3f}  '
                  f'both {row["share_both"]:.3f}  '
                  f'(not found {row["lost_not_found"]:.3f})')

    # ------------------------------------------------------------------ step 3: the main table
    # Two threshold policies, because one of them degenerates and that is itself a result.
    #
    # `youden` maximises operating_point - false_positive_rate. For a merit whose top-ranked
    # candidate is usually wrong, abstaining scores 0 - 0 = 0 and every real threshold scores
    # less, so the optimum is "never report an answer" -- op 0.0000 at fpr 0.0000. Measured, that
    # happens to most of the zoo, and it is the sharpest possible statement of what S07 exists to
    # fix. But it also makes the headline column uninformative, so every merit is *also* thresholded
    # against a matched false-positive budget: the rate M20 itself incurs at de Wolff's threshold
    # of 10 on fom-train. That is the apples-to-apples comparison -- equal cost in wrong answers
    # reported, so the operating points are comparable rather than each merit choosing its own
    # willingness to answer.
    m20_train = evaluate_merit(
        args.benchmark_dir, args.feature_dir, args.bundles, train_ids, entries, 'M20', True,
        DEWOLFF_THRESHOLD, (), (), 0, args.seed, args.train_split,
        )
    budget = float(m20_train.metric('false_positive'))
    print(f'\nmatched false-positive budget from M20 @ {DEWOLFF_THRESHOLD:g} on '
          f'{args.train_split}: {budget:.4f}')

    print(f'\nstep 3: the leaderboard, threshold on {args.train_split}, reported on '
          f'{args.report_split}')
    rows = []
    results = {}
    thresholds = {}
    for name, higher, source, note in merits:
        score_columns = ()
        started = time.perf_counter()
        train = evaluate_merit(
            args.benchmark_dir, args.feature_dir, args.bundles, train_ids, entries, name,
            higher, None, score_columns, (), 0, args.seed, args.train_split,
            )
        choice = FomMetrics.select_threshold(train, objective='youden')
        # per_entry stores scores already oriented higher-is-better, so a lower-is-better
        # merit's chosen threshold comes back negated and has to be turned round again
        # before `evaluate` mirrors it a second time.
        threshold = choice.threshold if higher else -choice.threshold
        dev = evaluate_merit(
            args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries, name,
            higher, threshold, score_columns, strata, args.n_bootstrap, args.seed,
            args.report_split,
            )
        FomMetrics.check_threshold_transfer(choice, dev)

        budgeted_choice = FomMetrics.select_threshold(
            train, objective='operating_point', max_false_positive_rate=budget,
            )
        budgeted_threshold = (budgeted_choice.threshold if higher
                              else -budgeted_choice.threshold)
        budgeted = evaluate_merit(
            args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries, name,
            higher, budgeted_threshold, score_columns, (), 0, args.seed, args.report_split,
            )
        FomMetrics.check_threshold_transfer(budgeted_choice, budgeted)

        results[name] = dev
        thresholds[name] = choice
        row = _table_row(name, higher, source, note, dev, threshold, choice)
        row['threshold_matched_fpr'] = budgeted_threshold
        row['operating_point_matched_fpr'] = budgeted.metric('operating_point')
        row['false_positive_rate_matched'] = budgeted.metric('false_positive')
        row['abstains_always'] = bool(
            row['reported'] == 0.0 and row['operating_point'] == 0.0
            )
        rows.append(row)
        print(f'  {name:18s} op {row["operating_point"]:.4f}  '
              f'op@fpr {row["operating_point_matched_fpr"]:.4f}  '
              f'top10 {row["top10"]:.4f}  '
              f'hard|found {row["hard_operating_point_given_found"]:.4f}  '
              f'{"ABSTAINS" if row["abstains_always"] else "        "}  '
              f'({time.perf_counter() - started:.0f} s)')

    # M20 at de Wolff's own threshold, and Werner's strengthened conjunction, as extra rows.
    literature = _table_row('M20 @ 10 (literature)', True, 'pool',
                            "de Wolff's published threshold, not selected here",
                            m20_dev, DEWOLFF_THRESHOLD, None)
    literature['operating_point_matched_fpr'] = literature['operating_point']
    literature['false_positive_rate_matched'] = literature['false_positive_rate']
    literature['threshold_matched_fpr'] = DEWOLFF_THRESHOLD
    literature['abstains_always'] = False
    rows.append(literature)
    werner_train = evaluate_merit(
        args.benchmark_dir, args.feature_dir, args.bundles, train_ids, entries, werner_strict,
        True, None, ('M20', 'X_N'), (), 0, args.seed, args.train_split,
        )
    werner_choice = FomMetrics.select_threshold(werner_train, objective='youden')
    werner_dev = evaluate_merit(
        args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries, werner_strict,
        True, werner_choice.threshold, ('M20', 'X_N'), strata, args.n_bootstrap, args.seed,
        args.report_split,
        )
    FomMetrics.check_threshold_transfer(werner_choice, werner_dev)
    werner_budgeted_choice = FomMetrics.select_threshold(
        werner_train, objective='operating_point', max_false_positive_rate=budget,
        )
    werner_budgeted = evaluate_merit(
        args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries, werner_strict,
        True, werner_budgeted_choice.threshold, ('M20', 'X_N'), (), 0, args.seed,
        args.report_split,
        )
    results['werner_strict'] = werner_dev
    werner_row = _table_row('werner_strict', True, 'features',
                            'M20 where every line below Q20 is indexed, else -inf',
                            werner_dev, werner_choice.threshold, werner_choice)
    werner_row['threshold_matched_fpr'] = werner_budgeted_choice.threshold
    werner_row['operating_point_matched_fpr'] = werner_budgeted.metric('operating_point')
    werner_row['false_positive_rate_matched'] = werner_budgeted.metric('false_positive')
    werner_row['abstains_always'] = bool(werner_row['reported'] == 0.0)
    rows.append(werner_row)
    print(f'  {"werner_strict":18s} op {werner_row["operating_point"]:.4f}  '
          f'op@fpr {werner_row["operating_point_matched_fpr"]:.4f}')

    table = pd.DataFrame(rows)

    # ------------------------------------------------------------------ paired comparisons
    print('\npaired comparisons against M20 (McNemar, same entries)')
    comparisons = []
    for name, higher, source, note in merits:
        if name == 'M20':
            continue
        for scope in (None, 'hard'):
            try:
                test = FomMetrics.mcnemar(results[name], results['M20'],
                                          metric='operating_point', subset=scope)
            except ValueError as error:
                comparisons.append({'merit': name, 'scope': scope or 'all',
                                    'error': str(error)})
                continue
            comparisons.append({'merit': name, 'scope': scope or 'all', **dict(test)})
    comparison_frame = pd.DataFrame(comparisons)

    # ------------------------------------------------------------------ step 4: the oracle gap
    print('\nstep 4: the oracle gap, by stratum')
    oracle_rows = []
    for degenerates in ('exclude', 'include'):
        result = m20_dev if degenerates == 'exclude' else evaluate_merit(
            args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries, 'M20', True,
            DEWOLFF_THRESHOLD, (), strata, 0, args.seed, args.report_split,
            )
        for scope, frame in (('aggregate', result.aggregate), ('hard', result.hard)):
            if not frame.shape[0]:
                continue
            row = frame.iloc[0]
            oracle_rows.append({
                'degenerates': degenerates, 'scope': scope,
                'operating_point': row['operating_point'],
                'ceiling_reranker': row['ceiling_reranker'],
                'ceiling_rescorer': row['ceiling_rescorer'],
                'headroom_reranker': row['headroom_reranker'],
                'headroom_rescorer': row['headroom_rescorer'],
                'degenerate_only': row['degenerate_only'],
                'n_entries': row['n_entries'],
                })
        if degenerates == 'exclude':
            for row in result.by_stratum.itertuples():
                oracle_rows.append({
                    'degenerates': degenerates, 'scope': f'{row.stratum}={row.level}',
                    'operating_point': row.operating_point,
                    'ceiling_reranker': row.ceiling_reranker,
                    'ceiling_rescorer': row.ceiling_rescorer,
                    'headroom_reranker': row.headroom_reranker,
                    'headroom_rescorer': row.headroom_rescorer,
                    'degenerate_only': row.degenerate_only,
                    'n_entries': row.n_entries,
                    })
    oracle = pd.DataFrame(oracle_rows)

    # ------------------------------------------------- the hard stratum, on the rank metrics only
    #
    # F-059 measured 104 reachable source entries in the hard stratum across *all three splits*.
    # Divided by the split it is 64 train / 16 dev / 24 test, and sixteen entries cannot rank
    # twenty-one figures of merit -- on fom-dev alone every merit scores exactly zero and McNemar
    # reports no discordant pairs at all. So the hard-stratum leaderboard is reported on the
    # metrics that involve no threshold (top1, top10, MRR, rank_only, and the two ceilings), over
    # fom-train and fom-dev together, which is 80 reachable source entries.
    #
    # This is not a hole in PROTOCOL section 8. The only thing selected on fom-train in this
    # session is each merit's threshold, and a rank metric does not see a threshold. It stops
    # being true the moment anything is *fitted* on fom-train, so S07 and S08 cannot reuse this
    # licence without re-deriving it.
    print('\nthe hard stratum, rank metrics only, on train+dev (16 dev entries is not a sample)')
    hard_rows = []
    hard_results = {}
    combined = train_ids | dev_ids
    for name, higher, source, note in merits:
        result = evaluate_merit(
            args.benchmark_dir, args.feature_dir, args.bundles, combined, entries, name,
            higher, None, (), (), args.n_bootstrap, args.seed, 'fom-train+fom-dev',
            )
        hard_results[name] = result
        if not result.hard.shape[0]:
            continue
        row = result.hard.iloc[0]
        hard_rows.append({
            'merit': name, 'higher_is_better': higher,
            'n_entries': row['n_entries'], 'n_found': row['n_found'],
            'top1': row['top1'], 'top10': row['top10'], 'rank_only': row['rank_only'],
            'mrr': row['mrr'], 'ceiling_rescorer': row['ceiling_rescorer'],
            'found_ci_low': row['found_ci_low'], 'found_ci_high': row['found_ci_high'],
            })
        print(f'  {name:18s} top1 {row["top1"]:.4f}  top10 {row["top10"]:.4f}  '
              f'mrr {row["mrr"]:.4f}  (n={int(row["n_entries"])} rows, '
              f'{row["ceiling_rescorer"]:.3f} reachable)')
    hard_frame = pd.DataFrame(hard_rows).sort_values('top10', ascending=False)

    # The hard leaderboard's significance test. McNemar on `operating_point` is vacuous there --
    # every merit scores zero, so there are no discordant pairs -- so the paired comparison is run
    # on `top10`, which is the metric the hard table is actually ranked on. Paired over the same
    # (entry, condition) rows, clustered nowhere: McNemar is exact on the pairs themselves.
    print('\nhard-stratum paired comparisons against M20 (McNemar on top10)')
    hard_comparisons = []
    for name in hard_results:
        if name == 'M20' or 'M20' not in hard_results:
            continue
        try:
            test = FomMetrics.mcnemar(hard_results[name], hard_results['M20'],
                                      metric='top10', subset='hard')
        except ValueError as error:
            hard_comparisons.append({'merit': name, 'error': str(error)})
            continue
        hard_comparisons.append({'merit': name, **dict(test)})
        print(f'  {name:18s} delta {test["delta"]:+.4f}  '
              f'discordant {int(test["n_a_only"])}/{int(test["n_b_only"])}  '
              f'p {test["p_value"]:.4f}')
    hard_comparison_frame = pd.DataFrame(hard_comparisons)

    # ------------------------------------------------------------------ per-bundle, and C5 alone
    print('\nper-bundle (C5 is the aggressive-dropout condition and is read on its own)')
    per_bundle = []
    for name, result in results.items():
        frame = result.by_stratum
        for row in frame.loc[frame['stratum'] == 'condition_bundle'].itertuples():
            per_bundle.append({
                'merit': name, 'condition_bundle': row.level,
                'condition': FomMetrics.BUNDLE_LABELS.get(row.level, '?'),
                'operating_point': row.operating_point,
                'operating_point_given_found': row.operating_point_given_found,
                'top1': row.top1, 'top10': row.top10, 'mrr': row.mrr,
                'ceiling_rescorer': row.ceiling_rescorer, 'n_entries': row.n_entries,
                })
    per_bundle_frame = pd.DataFrame(per_bundle)

    # ------------------------------------------------------------------ per-entry outcomes
    # Persisted because the complementarity matrix and the union oracle are joins over these
    # flags, and recomputing them would mean re-running every merit. One row per
    # (merit, entry, condition); ~600k rows, so it costs nothing to keep.
    per_entry = []
    for name, result in results.items():
        frame = result.per_entry
        keep = ['entry_id', 'condition_bundle', 'bravais_lattice', 'volume_decile', 'is_hard',
                'top1', 'top10', 'rank_only', 'threshold_only', 'operating_point', 'found',
                'reported', 'false_positive', 'reciprocal_rank', 'rank_best_correct']
        subset = frame[[column for column in keep if column in frame.columns]].copy()
        subset.insert(0, 'merit', name)
        per_entry.append(subset)
    per_entry_frame = pd.concat(per_entry, ignore_index=True)

    # ------------------------------------------------------------------ write
    # Written *before* the cost measurement, which is an optional extra. The first version of
    # this script measured cost first and lost twenty minutes of leaderboard to a TypeError in a
    # timing lambda; nothing cheap and failure-prone should sit between expensive work and its
    # artefact.
    table.to_csv(artifact_dir/f'{args.tag}_main_table.csv', index=False, encoding='utf-8')
    comparison_frame.to_csv(artifact_dir/f'{args.tag}_mcnemar.csv', index=False, encoding='utf-8')
    oracle.to_csv(artifact_dir/f'{args.tag}_oracle.csv', index=False, encoding='utf-8')
    per_bundle_frame.to_csv(artifact_dir/f'{args.tag}_per_bundle.csv', index=False,
                            encoding='utf-8')
    hard_frame.to_csv(artifact_dir/f'{args.tag}_hard_table.csv', index=False, encoding='utf-8')
    hard_comparison_frame.to_csv(artifact_dir/f'{args.tag}_hard_mcnemar.csv', index=False,
                                 encoding='utf-8')
    per_entry_frame.to_parquet(artifact_dir/f'{args.tag}_per_entry.parquet', index=False)
    (artifact_dir/f'{args.tag}_thresholds.json').write_text(
        json.dumps({name: choice.to_dict() for name, choice in thresholds.items()},
                   indent=2, sort_keys=True, default=str),
        encoding='utf-8',
        )
    print(f'\nwrote {args.tag}_{{main_table,hard_table,loss_decomposition,mcnemar,oracle,'
          f'per_bundle}}.csv to {artifact_dir}')

    # ------------------------------------------------------------------ cost, last and optional
    if not args.skip_cost:
        print('\nper-merit cost')
        cost = measure_cost(args.benchmark_dir)
        cost.to_csv(artifact_dir/f'{args.tag}_cost.csv', index=False, encoding='utf-8')
        table = table.merge(cost[['merit', 'cost_vs_M20']], on='merit', how='left')
        table.to_csv(artifact_dir/f'{args.tag}_main_table.csv', index=False, encoding='utf-8')
        print(cost.round(6).to_string(index=False))
    return 0


def _table_row(name, higher, source, note, result, threshold, choice):
    aggregate = result.aggregate.iloc[0]
    hard = result.hard.iloc[0] if result.hard.shape[0] else None
    label = name if isinstance(name, str) else getattr(name, '__name__', str(name))
    return {
        'merit': label,
        'higher_is_better': higher,
        'source': source,
        'note': note,
        'threshold': threshold,
        'threshold_objective': None if choice is None else choice.objective,
        'threshold_split': None if choice is None else choice.split,
        'operating_point': aggregate['operating_point'],
        'operating_point_ci_low': aggregate['operating_point_ci_low'],
        'operating_point_ci_high': aggregate['operating_point_ci_high'],
        'false_positive_rate': aggregate['false_positive'],
        'precision': aggregate['precision'],
        'reported': aggregate['reported'],
        'top1': aggregate['top1'],
        'top10': aggregate['top10'],
        'mrr': aggregate['mrr'],
        'rank_only': aggregate['rank_only'],
        'threshold_only': aggregate['threshold_only'],
        'ceiling_rescorer': aggregate['ceiling_rescorer'],
        'hard_operating_point': np.nan if hard is None else hard['operating_point'],
        'hard_operating_point_given_found':
            np.nan if hard is None else hard['operating_point_given_found'],
        'hard_ceiling_rescorer': np.nan if hard is None else hard['ceiling_rescorer'],
        'hard_n_entries': np.nan if hard is None else hard['n_entries'],
        # Deferred to S06b: it needs cell perturbation and re-refinement to convergence, which
        # is new compute and its own harness. Left explicitly empty rather than guessed at.
        'stability': np.nan,
        }


if __name__ == '__main__':
    raise SystemExit(main())
