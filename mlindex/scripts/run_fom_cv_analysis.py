"""S10's analysis stages: what the predictive merits are worth, and whether they mean what we think.

Imported by `run_fom_cv.py`, which owns the arguments and the feature build. Split out because the
build is minutes and the analysis is seconds, and because a stage that reads the matrix should not
be able to rebuild it by accident.

Every number here goes through `FomMetrics.evaluate`; thresholds are selected on `fom-train` and
passed through `check_threshold_transfer` before anything is reported on `fom-dev`.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics
from mlindex.utilities.FigureOfMerits import N_CELL_PARAMETERS
from run_fom_zoo_features import commit_hash


DEWOLFF_THRESHOLD = 10.0

POOL_COLUMNS = tuple(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + (
    'M20', 'volume', 'lattice_system', 'volume_ratio_to_truth', 'n_peaks', 'spacegroup',
    )

# Direction of merit. cv_raw and cv_chi2 are errors; the rest are merits.
#
# `M20` is de Wolff's own statistic moved out of sample -- global Q_N/(2N) over a *mean* |dQ| --
# and `M` is the same idea with the local Delta(Q) and a median. Both are here because a single
# comparison against the M20 baseline could not otherwise separate "cross-validation costs
# something" from "the local normalisation costs something".
MERIT_DIRECTION = {
    'M20': True, 'M': True, 'tail_nll': True, 'raw': False, 'chi2': False,
    }

BASELINES = (('M20', True), ('M_sym', True))

STRATA = ('bravais_lattice', 'volume_decile', 'condition_bundle')


def cv_merit_names(schemes):
    """(column, higher_is_better, family) for everything worth ranking."""
    names = []
    for scheme in schemes:
        for stem, higher in MERIT_DIRECTION.items():
            names.append((f'cv_{stem}__{scheme}', higher, f'cv/{scheme}'))
    for stem, higher in MERIT_DIRECTION.items():
        names.append((f'is_{stem}', higher, 'in-sample'))
        names.append((f'ho_{stem}', higher, 'holdout'))
    return names


def bundle_frames(args, keep_entry_ids, columns, require=()):
    """Yield one (pool + zoo features + cv features) frame per bundle.

    A generator for the same reason S06's is: `evaluate` consumes shards one at a time and holding
    six bundles joined three ways is several GB for no benefit.

    `require` names columns that must be finite for a row to be kept. That is how the Variant A
    rows are restricted to the entries that *have* a hold-out set: an entry with no surplus lines
    has NaN there for every one of its candidates, and leaving those in would score "this entry has
    no extra peaks" as "this candidate failed its extra peaks".
    """
    import pyarrow.parquet as pq

    keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
    for bundle in args.bundles:
        pool = FomBenchmark.load_candidates(
            args.benchmark_dir, bundles=[bundle], columns=list(POOL_COLUMNS),
            )
        for directory, prefix in ((args.feature_dir, 'features'), (args.out_dir, 'cv')):
            path = os.path.join(directory, f'{prefix}_{bundle}.parquet')
            if not os.path.exists(path):
                continue
            # Project at the read rather than after it: the CV matrix is 10M rows by ~30 float
            # columns and a whole-file read per merit would dominate the analysis.
            available = set(pq.ParquetFile(path).schema_arrow.names)
            wanted = [column for column in columns
                      if column not in pool.columns and column in available]
            if not wanted:
                continue
            stored = pd.read_parquet(path, columns=keys + wanted)
            pool = pool.merge(stored, on=keys, how='inner', validate='1:1')
        pool = pool.loc[pool['entry_id'].isin(keep_entry_ids)]
        for column in require:
            if column in pool.columns:
                pool = pool.loc[np.isfinite(pool[column].to_numpy(dtype=np.float64))]
        if pool.shape[0]:
            yield pool.reset_index(drop=True)


def load_split(args, keep_entry_ids, columns):
    """Materialise one split's shards once.

    The leaderboard ranks twenty-odd merits over the same rows, and streaming them would re-read
    the 10M-row CV matrix once per merit. Strings are cast to categories on the way in, which is
    most of the memory: `spacegroup` alone has ~151 levels over millions of rows.
    """
    frames = []
    for frame in bundle_frames(args, keep_entry_ids, columns):
        for column in ('bravais_lattice', 'lattice_system', 'spacegroup', 'condition_bundle'):
            if column in frame.columns:
                frame[column] = frame[column].astype('category')
        frames.append(frame)
    return frames


def _restrict(frames, require):
    """Shards with the rows that have no value for `require` dropped.

    Variant A only exists where the entry had surplus peaks. An entry without them has NaN there
    for every one of its candidates, so dropping the rows drops the entry -- which is what the
    comparison needs, since "this entry has no extra peaks" is not "this candidate failed them".
    """
    if not require:
        return frames
    kept = []
    for frame in frames:
        mask = np.ones(frame.shape[0], dtype=bool)
        for column in require:
            if column in frame.columns:
                mask &= np.isfinite(frame[column].to_numpy(dtype=np.float64))
        subset = frame.loc[mask]
        if subset.shape[0]:
            kept.append(subset.reset_index(drop=True))
    return kept


def evaluate_frames(frames, entries, score, higher, threshold, split_label, seed,
                    strata=(), n_bootstrap=0):
    return FomMetrics.evaluate(
        frames, score=score, higher_is_better=higher, threshold=threshold, entries=entries,
        strata=strata, split=split_label, n_bootstrap=n_bootstrap, seed=seed,
        )


def evaluate_merit(args, entries, keep_entry_ids, score, higher, threshold, split_label,
                   strata=(), n_bootstrap=0, require=()):
    columns = [score] if isinstance(score, str) else []
    shards = bundle_frames(args, keep_entry_ids, columns + list(require), require=require)
    return FomMetrics.evaluate(
        shards, score=score, higher_is_better=higher, threshold=threshold, entries=entries,
        strata=strata, split=split_label, n_bootstrap=n_bootstrap, seed=args.seed,
        )


def _row(name, family, higher, result, threshold, budgeted):
    aggregate = result.aggregate.iloc[0]
    hard = result.hard.iloc[0] if result.hard.shape[0] else None
    return {
        'merit': name,
        'family': family,
        'higher_is_better': higher,
        'n_entries': int(aggregate['n_entries']),
        'threshold': threshold,
        'operating_point': aggregate['operating_point'],
        'operating_point_ci_low': aggregate['operating_point_ci_low'],
        'operating_point_ci_high': aggregate['operating_point_ci_high'],
        'operating_point_matched_fpr': np.nan if budgeted is None
        else budgeted.metric('operating_point'),
        'false_positive_rate': aggregate['false_positive'],
        'precision': aggregate['precision'],
        'reported': aggregate['reported'],
        'top1': aggregate['top1'],
        'top10': aggregate['top10'],
        'mrr': aggregate['mrr'],
        'rank_only': aggregate['rank_only'],
        'ceiling_rescorer': aggregate['ceiling_rescorer'],
        'hard_operating_point_given_found':
            np.nan if hard is None else hard['operating_point_given_found'],
        'hard_n_entries': np.nan if hard is None else hard['n_entries'],
        }


# --------------------------------------------------------------------------------------------
# stage: main
# --------------------------------------------------------------------------------------------
def run_main(args, entries):
    train_ids = set(entries.loc[entries['split'] == args.train_split, 'entry_id'])
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    merits = list(BASELINES) + [(name, higher) for name, higher, _ in cv_merit_names(args.schemes)]
    families = {name: family for name, _, family in cv_merit_names(args.schemes)}
    columns = sorted({name for name, _ in merits})

    print('loading shards...', flush=True)
    started = time.perf_counter()
    train_frames = load_split(args, train_ids, columns)
    dev_frames = load_split(args, dev_ids, columns)
    print(f'  {time.perf_counter() - started:.0f}s; '
          f'{sum(f.shape[0] for f in train_frames):,} train / '
          f'{sum(f.shape[0] for f in dev_frames):,} dev candidates', flush=True)

    m20_train = evaluate_frames(train_frames, entries, 'M20', True, DEWOLFF_THRESHOLD,
                                args.train_split, args.seed)
    budget = float(m20_train.metric('false_positive'))
    print(f'matched false-positive budget from M20 @ {DEWOLFF_THRESHOLD:g} on '
          f'{args.train_split}: {budget:.5f}')

    rows, results = [], {}
    for name, higher in merits:
        family = families.get(name, 'baseline')
        # Variant A only exists where the entry had surplus peaks; restrict rather than impute.
        require = ('ho_M',) if family == 'holdout' else ()
        started = time.perf_counter()
        fit_on = _restrict(train_frames, require)
        report_on = _restrict(dev_frames, require)
        train = evaluate_frames(fit_on, entries, name, higher, None, args.train_split, args.seed)
        choice = FomMetrics.select_threshold(train, objective='youden')
        threshold = choice.threshold if higher else -choice.threshold
        dev = evaluate_frames(report_on, entries, name, higher, threshold, args.report_split,
                              args.seed, strata=STRATA, n_bootstrap=args.n_bootstrap)
        FomMetrics.check_threshold_transfer(choice, dev)

        budgeted_choice = FomMetrics.select_threshold(
            train, objective='operating_point', max_false_positive_rate=budget,
            )
        budgeted = evaluate_frames(
            report_on, entries, name, higher,
            budgeted_choice.threshold if higher else -budgeted_choice.threshold,
            args.report_split, args.seed,
            )
        results[name] = dev
        rows.append(_row(name, family, higher, dev, threshold, budgeted))
        print(f'  {name:22s} {family:12s} op {rows[-1]["operating_point"]:.4f}  '
              f'op@fpr {rows[-1]["operating_point_matched_fpr"]:.4f}  '
              f'top10 {rows[-1]["top10"]:.4f}  n={rows[-1]["n_entries"]}  '
              f'({time.perf_counter() - started:.0f}s)', flush=True)

    table = pd.DataFrame(rows).sort_values('top10', ascending=False)
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_main_table.csv'), index=False,
                 encoding='utf-8')

    # Paired tests, only between results covering the same entries. The Variant A rows sit on a
    # smaller set, so they are paired against baselines re-evaluated on that same set rather than
    # against the full-set baselines -- pairing across different denominators is not pairing.
    # Variant A's rows cover fewer entries than everything else, so its baselines are re-evaluated
    # on that same restricted set. Pairing across different denominators is not pairing.
    holdout_frames = _restrict(dev_frames, ('ho_M',))
    holdout_baselines = {}
    for baseline, higher in BASELINES:
        holdout_baselines[baseline] = evaluate_frames(
            holdout_frames, entries, baseline, higher, None, args.report_split, args.seed,
            )
    paired = []
    for name in results:
        is_holdout = families.get(name) == 'holdout'
        for baseline, higher in BASELINES:
            if name == baseline:
                continue
            other = holdout_baselines[baseline] if is_holdout else results[baseline]
            for metric in ('operating_point', 'top10'):
                if is_holdout and metric == 'operating_point':
                    # The re-evaluated baseline carries no threshold, so it has no operating point
                    # to be paired on; top-10 is threshold-free and is the comparable metric.
                    continue
                test = FomMetrics.mcnemar(results[name], other, metric=metric)
                test['merit'] = name
                test['baseline'] = baseline
                test['restricted_to_holdout_entries'] = bool(is_holdout)
                paired.append(test)
    if paired:
        pd.DataFrame(paired).to_csv(
            os.path.join(args.artifact_dir, f'{args.tag}_mcnemar.csv'), index=False,
            encoding='utf-8')

    best = table.loc[~table['merit'].isin([name for name, _ in BASELINES])].iloc[0]['merit']
    results[best].by_stratum.to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_by_stratum.csv'), index=False,
        encoding='utf-8')

    meta = dict(commit=commit_hash(), tag=args.tag, stage='main', seed=args.seed,
                bundles=list(args.bundles), schemes=list(args.schemes),
                train_split=args.train_split, report_split=args.report_split,
                matched_fpr_budget=budget, n_bootstrap=args.n_bootstrap,
                best_predictive_merit=str(best),
                n_entries_train=len(train_ids), n_entries_dev=len(dev_ids))
    _write_meta(args, 'main', meta)
    return table


# --------------------------------------------------------------------------------------------
# stage: scaling -- the gate's second condition
# --------------------------------------------------------------------------------------------
def run_scaling(args, entries):
    """Does the cross-validation penalty grow with the number of free cell parameters?

    Two measurements, and the second is the one that counts. The cross-lattice version compares
    candidates of different symmetry against each other, which confounds the parameter count with
    everything else that differs between lattices -- volume, line density, pool composition, and
    R5's ten-peaks-for-cubic. The **identical-peaks** version compares candidates of different
    symmetry *within the same entry*, so the peaks, the noise and the entry's difficulty are held
    fixed by construction and only the parameter count moves.

    The penalty is `is_M / cv_M`: the same statistic computed on the peaks the cell was fitted to,
    over the same statistic computed on peaks it was not. One means cross-validation found no
    fitting advantage to remove.
    """
    dev_ids = set(entries.loc[entries['split'].isin({args.train_split, args.report_split}),
                              'entry_id'])
    scheme = args.schemes[0]
    columns = ['is_M', 'is_M20', f'cv_M__{scheme}', f'cv_M20__{scheme}',
               f'cv_n_voided__{scheme}', f'cv_max_leverage__{scheme}', f'cv_n_scored__{scheme}']
    blocks = []
    for frame in bundle_frames(args, dev_ids, columns):
        block = frame[['entry_id', 'condition_bundle', 'bravais_lattice', 'lattice_system',
                       'is_correct', 'M20', 'volume']].copy()
        block['is_M'] = frame['is_M'].to_numpy(dtype=np.float64)
        block['is_M20'] = frame['is_M20'].to_numpy(dtype=np.float64)
        block['cv_M'] = frame[f'cv_M__{scheme}'].to_numpy(dtype=np.float64)
        block['cv_M20'] = frame[f'cv_M20__{scheme}'].to_numpy(dtype=np.float64)
        block['n_voided'] = frame[f'cv_n_voided__{scheme}'].to_numpy(dtype=np.float64)
        block['max_leverage'] = frame[f'cv_max_leverage__{scheme}'].to_numpy(dtype=np.float64)
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    frame['n_free'] = frame['lattice_system'].map(N_CELL_PARAMETERS)
    # Two penalties, in the two normalisations, so a scaling seen in one and not the other is
    # attributable. `penalty` is the local-Delta form; `penalty_dewolff` is M20's own.
    frame['penalty'] = frame['is_M']/frame['cv_M'].replace(0.0, np.nan)
    frame['penalty_dewolff'] = frame['is_M20']/frame['cv_M20'].replace(0.0, np.nan)
    usable = frame.loc[np.isfinite(frame['penalty']) & ~FomMetrics.as_bool(frame['is_correct'])]

    across = usable.groupby('n_free').agg(
        n_candidates=('penalty', 'size'),
        median_penalty=('penalty', 'median'),
        median_penalty_dewolff=('penalty_dewolff', 'median'),
        median_is_M=('is_M', 'median'),
        median_cv_M=('cv_M', 'median'),
        median_is_M20=('is_M20', 'median'),
        median_cv_M20=('cv_M20', 'median'),
        median_leverage=('max_leverage', 'median'),
        mean_voided=('n_voided', 'mean'),
        ).reset_index()
    across['comparison'] = 'cross-lattice'

    # The identical-peaks control. Within each (entry, condition) the peaks are literally the same
    # array, so a difference between parameter counts cannot be a difference between entries.
    paired_rows = []
    grouped = usable.groupby(['entry_id', 'condition_bundle'])
    per_entry = grouped.apply(
        lambda group: group.groupby('n_free')['penalty'].median(), include_groups=False,
        )
    if isinstance(per_entry, pd.Series):
        per_entry = per_entry.unstack()
    reference = 3 if 3 in per_entry.columns else int(min(per_entry.columns))
    for n_free in sorted(per_entry.columns):
        # Selecting [[reference, n_free]] would duplicate the column when the two coincide, and
        # `both[n_free]` would then return a frame rather than a series. Name them instead.
        both = pd.DataFrame({
            'reference': per_entry[reference], 'value': per_entry[n_free],
            }).dropna()
        if both.shape[0] < 20:
            continue
        ratio = both['value']/both['reference']
        paired_rows.append(dict(
            n_free=int(n_free), reference_n_free=int(reference), n_entries=int(both.shape[0]),
            median_penalty_ratio=float(ratio.median()),
            fraction_above_one=float((ratio > 1).mean()),
            median_penalty=float(both['value'].median()),
            ))
    within = pd.DataFrame(paired_rows)
    within['comparison'] = 'identical-peaks'

    across.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_scaling_cross.csv'), index=False,
                  encoding='utf-8')
    within.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_scaling_within.csv'), index=False,
                  encoding='utf-8')
    print('cross-lattice:\n', across.to_string(index=False))
    print('\nidentical peaks (penalty relative to the same entry\'s '
          f'{reference}-parameter candidates):\n', within.to_string(index=False))

    monotone = bool(across.sort_values('n_free')['median_penalty'].is_monotonic_increasing)
    monotone_dewolff = bool(
        across.sort_values('n_free')['median_penalty_dewolff'].is_monotonic_increasing)
    monotone_within = bool(
        within.sort_values('n_free')['median_penalty_ratio'].is_monotonic_increasing
        ) if within.shape[0] else False
    _write_meta(args, 'scaling', dict(
        commit=commit_hash(), tag=args.tag, stage='scaling', scheme=scheme,
        n_candidates=int(usable.shape[0]),
        penalty_definition='is_M / cv_M on incorrect candidates',
        monotone_cross_lattice=monotone, monotone_identical_peaks=monotone_within,
        monotone_cross_lattice_dewolff=monotone_dewolff,
        reference_n_free=int(reference),
        ))
    return across, within


# --------------------------------------------------------------------------------------------
# stage: gate -- the strata the acceptance condition is stated on
# --------------------------------------------------------------------------------------------
def entry_geometry(args, entries, keep_entry_ids):
    """Per (entry, condition): what beat the correct cell under M20, and how.

    Built the way `run_fom_zoo_explain.build_geometry` builds it, so the stratum here is the same
    object F-069 measured rather than a parallel definition of it.
    """
    rank = {lattice: position
            for position, lattice in enumerate(FomMetrics.BRAVAIS_LATTICES)}
    blocks = []
    for frame in bundle_frames(args, keep_entry_ids, ['M20']):
        values = frame['M20'].to_numpy(dtype=np.float64)
        reduced = FomMetrics.reduce_pool(frame, values, pool='cross_bl')
        order = pd.Categorical(frame['bravais_lattice'],
                               categories=FomMetrics.BRAVAIS_LATTICES).codes
        ordered = frame.iloc[np.lexsort((frame['candidate_id'].to_numpy(), order, -values))]
        top = ordered.groupby(['entry_id', 'condition_bundle'], sort=False).head(1)
        top = top.set_index(['entry_id', 'condition_bundle'])
        reduced = reduced.set_index(['entry_id', 'condition_bundle'])
        block = pd.DataFrame(index=reduced.index)
        block['has_correct'] = reduced['has_correct_all'].to_numpy()
        block['top_is_correct'] = reduced['top_is_correct_all'].to_numpy()
        block['bravais_lattice_top'] = reduced['bravais_lattice_top_all'].to_numpy()
        block['top_volume_ratio_to_truth'] = (
            top['volume_ratio_to_truth'].reindex(reduced.index).to_numpy())
        blocks.append(block.reset_index())
    geometry = pd.concat(blocks, ignore_index=True)
    context = FomMetrics.entry_context(entries)
    geometry = geometry.merge(
        context[['entry_id', 'condition_bundle', 'bravais_lattice', 'is_hard']],
        on=['entry_id', 'condition_bundle'], how='left',
        )
    beaten = (FomMetrics.as_bool(geometry['has_correct'])
              & ~FomMetrics.as_bool(geometry['top_is_correct']))
    lower = (geometry['bravais_lattice_top'].map(rank) > geometry['bravais_lattice'].map(rank))
    larger = geometry['top_volume_ratio_to_truth'] > 1.0
    geometry['stratum_symmetry_lowering'] = beaten & lower
    geometry['stratum_over_prediction'] = beaten & lower & larger
    geometry['stratum_beaten'] = beaten
    return geometry


def _mask_for(result, geometry, column):
    index = result.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index().index
    flags = geometry.set_index(['entry_id', 'condition_bundle'])[column]
    return flags.reindex(index).fillna(False).to_numpy(dtype=bool)


def run_gate(args, entries):
    """The acceptance gate's first condition, on both strata.

    The handoff's stratum is "a larger-volume, lower-symmetry candidate outranks the correct one".
    F-069 measured that only 40.5% of wrong winners are larger while 85.5% are lower symmetry, so
    the primary stratum here is symmetry lowering and the literal conjunction is reported beside
    it, unchanged (STATUS section 6, this session). Both are reported; neither is dropped.
    """
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])
    geometry = entry_geometry(args, entries, dev_ids)
    counts = {column: int(geometry[column].sum())
              for column in ('stratum_beaten', 'stratum_symmetry_lowering',
                             'stratum_over_prediction')}
    print('stratum sizes on', args.report_split, counts)

    main_table = pd.read_csv(os.path.join(args.artifact_dir, f'{args.tag}_main_table.csv'))
    train_ids = set(entries.loc[entries['split'] == args.train_split, 'entry_id'])

    # Two champions, not one. The gate is written about the cross-validated variant, so the best
    # `cv_*` merit is tested whatever else wins; and the best predictive merit overall is tested
    # because that is what a deployed score would be. They sit on different entry sets -- Variant A
    # only exists where the entry had surplus peaks -- so each carries its own baselines,
    # re-evaluated on its own set. Pairing across different denominators is not pairing.
    champions = []
    for family in ('cv', 'any'):
        subset = main_table.loc[main_table['family'] != 'baseline']
        if family == 'cv':
            subset = subset.loc[subset['family'].str.startswith('cv/')]
        if subset.shape[0]:
            row = subset.iloc[0]
            champions.append((str(row['merit']), bool(row['higher_is_better'])))
    champions = list(dict.fromkeys(champions))

    rows, paired = [], []
    for champion, higher_champion in champions:
        require = ('ho_M',) if champion.startswith('ho_') else ()
        candidates = [(champion, higher_champion)] + list(BASELINES)
        results = {}
        for name, higher in candidates:
            train = evaluate_merit(args, entries, train_ids, name, higher, None,
                                   args.train_split, require=require)
            choice = FomMetrics.select_threshold(train, objective='youden')
            results[name] = evaluate_merit(
                args, entries, dev_ids, name, higher,
                choice.threshold if higher else -choice.threshold, args.report_split,
                require=require,
                )
        for stratum in ('stratum_symmetry_lowering', 'stratum_over_prediction',
                        'stratum_beaten'):
            mask = _mask_for(results[champion], geometry, stratum)
            for name, _ in candidates:
                per_entry = results[name].per_entry.set_index(
                    ['entry_id', 'condition_bundle']).sort_index()
                for metric in ('operating_point', 'top10'):
                    rows.append(dict(champion=champion, stratum=stratum, merit=name,
                                     metric=metric, n_entries=int(mask.sum()),
                                     value=float(per_entry.loc[mask, metric].mean())))
            for baseline, _ in BASELINES:
                for metric in ('operating_point', 'top10'):
                    test = FomMetrics.mcnemar(results[champion], results[baseline],
                                              metric=metric, subset=mask)
                    test['stratum'] = stratum
                    test['merit'] = champion
                    test['baseline'] = baseline
                    paired.append(test)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_gate.csv'), index=False,
                 encoding='utf-8')
    pd.DataFrame(paired).to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_gate_mcnemar.csv'), index=False,
        encoding='utf-8')
    print(table.pivot_table(index=['champion', 'stratum', 'metric'], columns='merit',
                            values='value').to_string())
    _write_meta(args, 'gate', dict(commit=commit_hash(), tag=args.tag, stage='gate',
                                   stratum_sizes=counts,
                                   primary_stratum='stratum_symmetry_lowering',
                                   champions=[name for name, _ in champions]))
    return table


# --------------------------------------------------------------------------------------------
# stage: confound -- is it measuring over-fitting capacity, or something duller?
# --------------------------------------------------------------------------------------------
def run_confound(args, entries):
    """Three controls and a cost table.

    A merit that ranks well because it is really volume, or really M20, is not the novel thing this
    step claims to have built. Each control removes one alternative explanation:

      within-volume-decile   is it re-ranking on cell size? Rank correlation with correctness
                             computed inside a decile, where size barely varies.
      residual on M20        cv_M is monotone in the mean held-out |dQ|, and M20 is built on the
                             in-sample one. If they agree closely there is no new information.
      M20-matched            are wrong candidates simply less converged? Compare inside narrow M20
                             bands, where convergence is held roughly fixed.
    """
    from scipy import stats

    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])
    scheme = args.schemes[0]
    merit = f'cv_M__{scheme}'
    blocks = []
    for frame in bundle_frames(args, dev_ids, [merit, 'is_M', 'M_sym']):
        keep = ['entry_id', 'condition_bundle', 'bravais_lattice', 'is_correct', 'M20', 'volume']
        block = frame[keep].copy()
        block['cv_M'] = frame[merit].to_numpy(dtype=np.float64)
        block['is_M'] = frame['is_M'].to_numpy(dtype=np.float64)
        block['M_sym'] = frame['M_sym'].to_numpy(dtype=np.float64)
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    frame['is_correct'] = FomMetrics.as_bool(frame['is_correct'])
    frame = frame.loc[np.isfinite(frame['cv_M']) & np.isfinite(frame['M20'])]
    frame['volume_decile'] = pd.qcut(frame['volume'], 10, labels=False, duplicates='drop')
    frame['M20_band'] = pd.qcut(frame['M20'], 20, labels=False, duplicates='drop')

    rows = []

    def _auc(group, column):
        positive = group.loc[group['is_correct'], column]
        negative = group.loc[~group['is_correct'], column]
        if positive.shape[0] < 5 or negative.shape[0] < 5:
            return np.nan
        return float(stats.mannwhitneyu(positive, negative).statistic
                     / (positive.shape[0]*negative.shape[0]))

    for column in ('cv_M', 'is_M', 'M20', 'M_sym'):
        rows.append(dict(control='pooled', column=column, level='all',
                         n=int(frame.shape[0]), auc=_auc(frame, column)))
        for control, key in (('within-volume-decile', 'volume_decile'),
                             ('M20-matched', 'M20_band')):
            values = [_auc(group, column) for _, group in frame.groupby(key)]
            values = [value for value in values if np.isfinite(value)]
            rows.append(dict(control=control, column=column, level='mean over bands',
                             n=int(frame.shape[0]),
                             auc=float(np.mean(values)) if values else np.nan))

    correlation = frame[['cv_M', 'is_M', 'M20', 'M_sym']].corr(method='spearman')
    correlation.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_confound_correlation.csv'),
                       encoding='utf-8')
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_confound.csv'), index=False,
                 encoding='utf-8')
    print(table.pivot(index='column', columns='control', values='auc').to_string())
    print('\nSpearman correlation:\n', correlation.round(3).to_string())
    _write_meta(args, 'confound', dict(commit=commit_hash(), tag=args.tag, stage='confound',
                                       scheme=scheme, n_candidates=int(frame.shape[0])))
    return table


def run_cost(args, entries):
    """Seconds per candidate against `get_M20`, in the S06_zoo_cost.csv format.

    F-085 and F-092 make this the number S14 actually needs: the distilled model is 3% of the
    inner-loop budget and the *features* are all of it, so a merit's price is what decides whether
    it can be deployed at all. Measured on real candidate pools from the frozen pool, so the
    reference-line lengths are the ones production sees, and with the numba kernels warm, since a
    cold JIT would price the compile rather than the arithmetic.
    """
    from mlindex.utilities import FigureOfMerits as fom

    candidates = FomBenchmark.load_candidates(
        args.benchmark_dir, bundles=[args.bundles[0]],
        columns=list(FomBenchmark.CV_CANDIDATE_COLUMNS),
        )
    entry_table = FomBenchmark.load_entries(args.benchmark_dir)
    entry_table = entry_table.loc[entry_table['condition_bundle'] == args.bundles[0]]
    peaks = entry_table.set_index('entry_id')['q2_obs']

    cases = []
    for entry_id in pd.unique(candidates['entry_id'])[:args.cost_entries]:
        group = candidates.loc[candidates['entry_id'] == entry_id]
        for keys, chunk in group.groupby(['lattice_system', 'bravais_lattice', 'spacegroup',
                                          'n_peaks'], sort=False):
            lattice_system, bravais_lattice, spacegroup, n_peaks = keys
            q2_obs = np.asarray(peaks.loc[entry_id], dtype=np.float64)[:int(n_peaks)]
            xnn = np.vstack([np.asarray(v, dtype=np.float64) for v in chunk['xnn']])
            q2_ref_calc, _, hkl, q2_calc = FomBenchmark.assign_lines(
                q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, args.models_dir,
                )
            hkl_ref = FomBenchmark.hkl_ref_for(
                lattice_system, bravais_lattice, spacegroup, args.models_dir)
            cases.append((q2_obs, xnn, hkl, hkl_ref, lattice_system, bravais_lattice,
                          q2_calc, q2_ref_calc))
    n_candidates = sum(case[1].shape[0] for case in cases)

    # Each row is one *call*, and a call emits the whole family at once -- get_cv_fom returns
    # cv_M20, cv_M, cv_tail_nll, cv_raw and cv_chi2 from the same folds. So these are the prices of
    # the families, which is what a deployment pays; there is no cheaper way to get cv_M20 alone.
    calls = {'M20': lambda c: fom.get_M20(c[0], c[6], c[7].copy()),
             'is_* (all in-sample)': lambda c: fom.get_insample_fom(
                 c[0], c[1], c[2], c[4], c[5], q2_calc=c[6], q2_ref_calc=c[7]),
             'ho_* (5 peaks)': lambda c: fom.get_holdout_fom(
                 c[0][:5], c[1], c[3], c[4], c[5]),
             }
    for scheme in args.schemes:
        calls[f'cv_* ({scheme})'] = (
            lambda c, scheme=scheme: fom.get_cv_fom(
                c[0], c[1], c[2], c[3], c[4], c[5], scheme=scheme, n_folds=args.n_folds)
            )

    rows, baseline = [], None
    for name, call in calls.items():
        best = np.inf
        for _ in range(args.cost_repeats + 1):        # the first pass warms the numba kernels
            start = time.perf_counter()
            for case in cases:
                call(case)
            best = min(best, time.perf_counter() - start)
        per_candidate = best/max(n_candidates, 1)
        if name == 'M20':
            baseline = per_candidate
        rows.append(dict(merit=name, seconds_per_candidate=per_candidate))
    table = pd.DataFrame(rows)
    table['cost_vs_M20'] = table['seconds_per_candidate']/baseline
    table['n_candidates_timed'] = n_candidates
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_cost.csv'), index=False,
                 encoding='utf-8')
    print(table.to_string(index=False))
    _write_meta(args, 'cost', dict(commit=commit_hash(), tag=args.tag, stage='cost',
                                   bundle=args.bundles[0], n_folds=args.n_folds,
                                   n_candidates_timed=int(n_candidates),
                                   note='first pass discarded so the numba JIT is warm'))
    return table


def _write_meta(args, stage, meta):
    meta = dict(meta)
    meta.setdefault('commit', commit_hash())
    with open(os.path.join(args.artifact_dir, f'{args.tag}_{stage}_meta.json'),
              'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, default=str)


def dispatch(args, entries):
    if args.stage == 'main':
        return run_main(args, entries)
    if args.stage == 'scaling':
        return run_scaling(args, entries)
    if args.stage == 'gate':
        return run_gate(args, entries)
    if args.stage == 'confound':
        return run_confound(args, entries)
    if args.stage == 'cost':
        return run_cost(args, entries)
    if args.stage == 'combiner':
        import run_fom_cv_combiner
        return run_fom_cv_combiner.run(args, entries)
    raise ValueError(f'unknown stage {args.stage!r}')
