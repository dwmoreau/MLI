"""Does the cross-validated merit add anything to S08's finished combiner?

S08 is done and reports 0.6341 on `fom-dev` against raw `M_sym`'s 0.4681. A new merit that beats
M20 is no longer interesting on its own -- twenty-nine of them already do. The question S10 has to
answer is whether it carries information the existing feature set does not, and the only honest way
to ask that is to fit the same architecture twice, with and without, and pair the two on the same
entries.

Deliberately *not* a change to S08's default feature set: `cv` is its own group and
`FomCombiner.DEFAULT_GROUPS` is untouched, so every number in STATUS section 2 still means what it
meant. The comparison is made here, and adopting it would be a later decision with its own record.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from mlindex.model_training import FomCombiner
from mlindex.model_training import FomMetrics
from mlindex.model_training import FomNull
import run_fom_combiner as s08
from run_fom_zoo_features import commit_hash


# S08's winning arm and its tuned hyperparameters, taken from its own artefacts rather than
# re-tuned: a paired comparison has to hold the architecture fixed, or the difference measured is
# the tuning and not the features.
BASELINE_GROUPS = ('scaled', 'structural', 'context')
CV_GROUPS = BASELINE_GROUPS + ('cv',)
LOAD_GROUPS = ('raw', 'scaled', 'structural', 'context', 'in_sample', 'cv')


def _s08_settings(artifact_dir):
    path = os.path.join(artifact_dir, 'S08_combiner_meta.json')
    with open(path, encoding='utf-8') as handle:
        meta = json.load(handle)
    tuning = pd.read_csv(os.path.join(artifact_dir, 'S08_combiner_tuning.csv'))
    best = tuning.sort_values('average_precision', ascending=False).iloc[0]
    params = dict(max_iter=int(best['max_iter']), learning_rate=float(best['learning_rate']),
                  max_leaf_nodes=int(best['max_leaf_nodes']))
    return meta, params


def load_frames(args, entries, covariates, scalers, keep_ids, cv_dir, n_negatives=None, seed=12345):
    frames = []
    for frame in FomCombiner.combiner_frames(
            args.benchmark_dir, args.feature_dir, args.bundles, keep_ids, covariates, scalers,
            groups=LOAD_GROUPS, cv_dir=cv_dir):
        if n_negatives is not None:
            frame = FomCombiner.subsample_negatives(frame, n_negatives, seed)
        frames.append(frame)
    return frames


def run(args, entries):
    meta, params = _s08_settings(args.artifact_dir)
    budget = float(meta['matched_fpr_budget'])
    n_negatives = int(meta.get('n_negatives', 40))
    holdout_fraction = float(meta.get('holdout_fraction', 0.2))

    # fit_arm reads these off `args`; run_fom_cv.py's parser does not carry them because they
    # belong to S08's experiment rather than to this one.
    args.objective = 'pointwise'
    args.model_params = params

    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = s08.split_ids(entries, args.train_split, holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    print('loading frames...', flush=True)
    started = time.perf_counter()
    fit_frames = load_frames(args, entries, covariates, scalers, fit_ids, args.out_dir,
                             n_negatives=n_negatives, seed=args.seed)
    cal_frames = load_frames(args, entries, covariates, scalers, cal_ids, args.out_dir)
    dev_frames = load_frames(args, entries, covariates, scalers, dev_ids, args.out_dir)
    print(f'  {time.perf_counter() - started:.0f}s; '
          f'{sum(f.shape[0] for f in fit_frames):,} fit / '
          f'{sum(f.shape[0] for f in dev_frames):,} dev candidates', flush=True)

    rows, results, models = [], {}, {}
    for name, groups in (('S08 baseline', BASELINE_GROUPS), ('S08 + cv', CV_GROUPS)):
        combiner, dev, row = s08.fit_arm(
            name, groups, fit_frames, cal_frames, dev_frames, entries, scalers, budget, args,
            )
        rows.append(row)
        results[name] = dev
        models[name] = combiner
        print(f'  {name:14s} op {row["operating_point"]:.4f}  top10 {row["top10"]:.4f}  '
              f'{row["n_features"]} features', flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_combiner_table.csv'), index=False,
                 encoding='utf-8')

    paired = []
    for metric in ('operating_point', 'top10'):
        test = FomMetrics.mcnemar(results['S08 + cv'], results['S08 baseline'], metric=metric)
        test['comparison'] = 'S08 + cv vs S08 baseline'
        paired.append(test)
        for lattice in FomMetrics.BRAVAIS_LATTICES:
            index = results['S08 + cv'].per_entry.set_index(
                ['entry_id', 'condition_bundle']).sort_index()
            mask = (index['bravais_lattice'] == lattice).to_numpy(dtype=bool)
            if mask.sum() < 30:
                continue
            per_lattice = FomMetrics.mcnemar(results['S08 + cv'], results['S08 baseline'],
                                             metric=metric, subset=mask)
            per_lattice['comparison'] = f'{lattice}'
            paired.append(per_lattice)
    pd.DataFrame(paired).to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_combiner_mcnemar.csv'), index=False,
        encoding='utf-8')

    # Saved so a later session can ask a question of this model without refitting it. The fit is
    # ~15 minutes and nothing about it is interesting enough to repeat.
    models['S08 + cv'].save(os.path.join(getattr(args, 'models_dir_combiner', os.path.join('mlindex', 'models')),
                     'fom_combiner_cv'))

    # Where the CV columns sit in the ordering the model actually uses. Permutation importance on
    # dev, not impurity importance -- the S08 handoff asks for that explicitly.
    importance = _importance(models['S08 + cv'], dev_frames, args.seed)
    importance.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_combiner_importance.csv'),
                      index=False, encoding='utf-8')
    print('\ncv columns by permutation importance:')
    print(importance.loc[importance['is_cv']].to_string(index=False))

    _meta = dict(commit=commit_hash(), tag=args.tag, stage='combiner',
                 baseline_groups=list(BASELINE_GROUPS), cv_groups=list(CV_GROUPS),
                 params=params, matched_fpr_budget=budget, n_negatives=n_negatives,
                 holdout_fraction=holdout_fraction, seed=args.seed,
                 cv_features=list(FomCombiner.CV_MERITS),
                 delta_operating_point=float(table.iloc[1]['operating_point']
                                             - table.iloc[0]['operating_point']),
                 delta_top10=float(table.iloc[1]['top10'] - table.iloc[0]['top10']))
    with open(os.path.join(args.artifact_dir, f'{args.tag}_combiner_meta.json'),
              'w', encoding='utf-8') as handle:
        json.dump(_meta, handle, indent=2, default=str)
    return table


# ---------------------------------------------------------------------------------------
# Q35 -- does `ho_M20` alone carry S10's +1.83 pp, or is the cross-validated half needed?
# ---------------------------------------------------------------------------------------
#
# F-099 ranks `ho_M20` **second of all seventy-eight features** at 8.7x `get_M20`, an order of
# magnitude cheaper than `cv_M20__random` at 36.7x, and four of the thirteen CV columns carry
# *negative* importance. If the cheap column is the whole gain then S10 delivers a merit S14 can
# actually afford, and the cross-validation half can be dropped outright.
#
# Held to the same discipline as `run()`: one architecture, one set of hyperparameters taken from
# S08's own artefacts, four arms differing only in which columns they may see, paired on the same
# entries. `fit_arm(names_override=...)` refits the same model class on a subset of its own
# columns, which is F-093's ablation template rather than a new one.

# The thirteen CV_MERITS split three ways by what they cost to compute.
HOLDOUT_MERITS = ('ho_M20', 'ho_M', 'ho_tail_nll')
CROSSVAL_MERITS = tuple(name for name in FomCombiner.CV_MERITS if name not in HOLDOUT_MERITS)


def family_arms(full_names):
    """(label, names, note) per arm, as subsets of the full 78-column feature set."""
    full = tuple(full_names)
    baseline = tuple(n for n in full if n not in set(FomCombiner.CV_MERITS))
    holdout_only = tuple(n for n in full if n in set(baseline) | set(HOLDOUT_MERITS))
    crossval_only = tuple(n for n in full if n in set(baseline) | set(CROSSVAL_MERITS))
    return [
        ('S08 baseline', baseline, 'no predictive columns at all'),
        ('+ ho_* only', holdout_only, f'{len(HOLDOUT_MERITS)} columns, ho_M20 at 8.7x get_M20'),
        ('+ cv_*/is_* only', crossval_only,
         f'{len(CROSSVAL_MERITS)} columns, cv_M20__random at 36.7x'),
        ('+ both (S10 as shipped)', full, 'all thirteen'),
        ]


def _report(combiner, cal_frames, dev_frames, entries, budget, args, name, note=''):
    """Calibrate, choose a threshold on the calibration split, report on dev -- `fit_arm`'s tail.

    Separated from the fit so an arm that is a restriction of an already-fitted model does not
    pay for a second full fit.
    """
    combiner.fit_calibrators(cal_frames)
    columns = combiner.score_columns
    cal = s08.evaluate_score(cal_frames, entries, combiner.score, None, args.train_split, columns)
    choice, rule = s08.choose_threshold(cal, budget)
    dev = s08.evaluate_score(dev_frames, entries, combiner.score, float(choice.threshold),
                             args.report_split, columns, n_bootstrap=args.n_bootstrap,
                             seed=args.seed)
    FomMetrics.check_threshold_transfer(choice, dev)
    row = s08.scope_row(dev, arm=name, question=note, n_features=len(combiner.names),
                        threshold=float(choice.threshold), threshold_rule=rule)
    row.update(s08.per_lattice(dev, 'dev'))
    return dev, row


def run_family(args, entries):
    meta, params = _s08_settings(args.artifact_dir)
    budget = float(meta['matched_fpr_budget'])
    n_negatives = int(meta.get('n_negatives', 40))
    holdout_fraction = float(meta.get('holdout_fraction', 0.2))
    args.objective = 'pointwise'
    args.model_params = params

    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = s08.split_ids(entries, args.train_split, holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    print('loading frames...', flush=True)
    started = time.perf_counter()
    fit_frames = load_frames(args, entries, covariates, scalers, fit_ids, args.out_dir,
                             n_negatives=n_negatives, seed=args.seed)
    cal_frames = load_frames(args, entries, covariates, scalers, cal_ids, args.out_dir)
    dev_frames = load_frames(args, entries, covariates, scalers, dev_ids, args.out_dir)
    print(f'  {time.perf_counter() - started:.0f}s; '
          f'{sum(f.shape[0] for f in fit_frames):,} fit / '
          f'{sum(f.shape[0] for f in dev_frames):,} dev candidates', flush=True)

    # The full model is fitted once and every arm is a refit of the same class on a subset of
    # its own columns, so each arm costs one fit rather than two -- run_fom_neural's ablation
    # stage does the same, for the same reason. Taking every arm's columns from the *full*
    # model's list is also what keeps them comparable: fitting each from its own group list
    # would let the column order drift between arms.
    print('fitting the full model...', flush=True)
    full = FomCombiner.FomCombiner.fit(
        fit_frames, groups=CV_GROUPS, scalers=scalers, objective='pointwise', seed=args.seed,
        **params)
    print(f'  {len(full.names)} features', flush=True)

    rows, results = [], {}
    for label, names, note in family_arms(tuple(full.names)):
        model = full if len(names) == len(full.names) else s08._restrict(
            full, list(names), fit_frames, args)
        dev, row = _report(model, cal_frames, dev_frames, entries, budget, args, label, note)
        rows.append(row)
        results[label] = dev
        print(f'  {label:24s} {len(names):3d} features  op {row["operating_point"]:.4f}  '
              f'top10 {row["top10"]:.4f}', flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_family_table.csv'), index=False,
                 encoding='utf-8')

    # Every arm against the baseline, and the two cheap-versus-expensive contrasts that answer
    # the question directly.
    contrasts = [(label, 'S08 baseline') for label in results if label != 'S08 baseline']
    contrasts += [('+ both (S10 as shipped)', '+ ho_* only'),
                  ('+ ho_* only', '+ cv_*/is_* only')]
    # Both scopes, always. PROTOCOL §3 rule 6: the hard stratum carries the claim, and an
    # aggregate alone is dominated by the easy cases. It matters here -- the two scopes do
    # not agree, and reporting only the aggregate would have recommended deleting the very
    # columns the hard stratum depends on.
    paired = []
    for metric in ('operating_point', 'top10'):
        for scope in (None, 'hard'):
            for treatment, control in contrasts:
                test = FomMetrics.mcnemar(results[treatment], results[control], metric=metric,
                                          subset=scope)
                test['comparison'] = f'{treatment} vs {control}'
                test['scope'] = 'aggregate' if scope is None else 'hard'
                paired.append(test)
    mcnemar = pd.DataFrame(paired)
    mcnemar.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_family_mcnemar.csv'), index=False,
                   encoding='utf-8')
    print()
    print(mcnemar.to_string(index=False))

    with open(os.path.join(args.artifact_dir, f'{args.tag}_family_meta.json'),
              'w', encoding='utf-8') as handle:
        json.dump(dict(commit=commit_hash(), tag=args.tag, stage='family',
                       question='Q35', params=params, matched_fpr_budget=budget,
                       n_negatives=n_negatives, holdout_fraction=holdout_fraction,
                       seed=args.seed, holdout_merits=list(HOLDOUT_MERITS),
                       crossval_merits=list(CROSSVAL_MERITS),
                       arms={label: len(names)
                             for label, names, _ in family_arms(tuple(full.names))}),
                  handle, indent=2, default=str)
    return table


def _importance(combiner, dev_frames, seed, n_repeats=1, sample=30000):
    """Permutation importance of each feature on average precision, on the dev split.

    Deliberately small. A 600-iteration gradient-boosted tree costs ~4.5 us per row per call, so a
    full-size permutation pass over 78 features is hours rather than minutes -- measured, after a
    first attempt at 200 000 rows and three repeats ran for three of them. Thirty thousand rows and
    one repeat is enough to *order* the features, which is the only claim made from this table; the
    magnitudes are not quoted anywhere.
    """
    frame = pd.concat(dev_frames, ignore_index=True)
    if frame.shape[0] > sample:
        frame = frame.sample(sample, random_state=seed)
    matrix = combiner.design_matrix(frame)
    target = FomMetrics.as_bool(frame['is_correct'])
    base = FomMetrics.average_precision(combiner.predict_batch(matrix), target)
    rng = np.random.default_rng(seed)
    rows = []
    for position, name in enumerate(combiner.names):
        drops = []
        for _ in range(n_repeats):
            shuffled = matrix.copy()
            shuffled[:, position] = shuffled[rng.permutation(shuffled.shape[0]), position]
            drops.append(base - FomMetrics.average_precision(
                combiner.predict_batch(shuffled), target))
        rows.append(dict(feature=name, importance=float(np.mean(drops)),
                         is_cv=name in set(FomCombiner.CV_MERITS)))
    return pd.DataFrame(rows).sort_values('importance', ascending=False)
