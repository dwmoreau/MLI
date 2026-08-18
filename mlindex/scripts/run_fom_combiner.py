"""S08: fit the learned combiner over the frozen pool, and report it.

Reads the feature matrix `run_fom_zoo_features.py` wrote and the normalisers `run_fom_null.py`
fitted, and reports through `FomMetrics.evaluate` -- which is the only route to a number in this
project, so nothing here re-implements a metric.

Four things about the protocol that the handoff was written before:

  * **The first experiment is raw features against scaled features.** S07 cannot answer it -- a
    normaliser has no ranking of its own to test -- and it is the question DWMM's reframing was
    made to pose. Three arms, same model, same split, McNemar over the same entries.
  * **Every number is quoted against raw M20 *and* raw `M_sym`.** The written gate is +5 pp over
    raw M20, which `M_sym` clears by 17.7 with no learning at all, so a gate-only report would
    call a model successful for reproducing an existing merit. The gate is not moved (PROTOCOL
    section 7); a second baseline is reported beside it (DWMM, 2026-08-18).
  * **The isotonic calibrators are fitted on a held-out slice of `fom-train`, not on `fom-dev`.**
    The handoff says `fom-dev`; PROTOCOL section 8 forbids reporting a number on the split it was
    fitted on, and an ECE quoted on `fom-dev` after calibrating there is exactly that.
  * **Hard-stratum threshold metrics are reported at volume decile >= 6** and rank metrics at the
    literal >= 8 (Q32, resolved 2026-08-17). S08 fits on `fom-train`, so it cannot inherit S06's
    licence to pool the hard stratum over train+dev.

    python mlindex/scripts/run_fom_combiner.py

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
from mlindex.model_training import FomCombiner  # noqa: E402
from mlindex.model_training import FomMetrics  # noqa: E402
from mlindex.scripts.run_fom_zoo_features import EVALUABLE_BUNDLES  # noqa: E402
from mlindex.scripts.run_fom_zoo_features import commit_hash  # noqa: E402

# The literature's own threshold, and the matched false-positive budget every merit is compared
# at. Recomputed here rather than hard-coded so the number is derived from the same pool it
# describes -- it comes out 0.24573, as it did for S06 and S07.
DEWOLFF_THRESHOLD = 10.0

# The two baselines. `M_sym` is not the gate but is what a deployed score would have to beat.
BASELINES = (('M20', True), ('M_sym', True))

# Q32: threshold metrics on the hard stratum are reported at this volume decile, rank metrics at
# `FomMetrics.HARD_MIN_DECILE`.
HARD_THRESHOLD_DECILE = 6

# Every column the arms between them need, so the frames are loaded once.
LOAD_GROUPS = ('raw', 'scaled', 'structural', 'context', 'in_sample')

# The experiment ladder. `groups` selects columns out of the loaded frames.
ARMS = (
    ('raw', ('raw', 'structural', 'context')),
    ('scaled', ('scaled', 'structural', 'context')),
    ('raw+scaled', ('raw', 'scaled', 'structural', 'context')),
    )

# Feature-group ablations, run on the winning arm's group set.
ABLATIONS = (
    ('drop_structural', ('structural',)),
    ('drop_context', ('context',)),
    )

# The hyperparameter grid, searched by GroupKFold *inside* `fom-train`'s fit part and scored by
# candidate-level average precision. Small on purpose: tuning on `fom-dev` and then reporting it
# is the second anti-pattern in the S08 handoff's pitfall list, and a wide grid searched on the
# right split still costs wall-clock that the ablations need more.
#
# Average precision rather than the operating point, because the fit split has had its negatives
# subsampled, so a per-entry ranking metric measured on it describes a pool that does not exist.
# METRICS.md section 7 sanctions exactly this use and forbids it as a headline.
TUNING_GRID = (
    dict(max_iter=200, learning_rate=0.10, max_leaf_nodes=31),
    dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63),
    dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63),
    dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=127),
    )

# Merits `S06_zoo_cost.csv` prices below one `get_M20`. The inner-loop variant is restricted to
# these; `M_sym`, `M_rev`, `M_tilde`, `n_over` and `max_gap` are 24-30x and are what the budget
# cannot afford.
CHEAP_MERITS = (
    'M20', 'Minfo', 'M_star', 'M_star_corrected', 'M_info_clipped', 'null_tail_nll', 'F_N_q',
    'X_N', 'M_werner_frac', 'nll_exponential',
    )

# `get_M20`'s own cost and the summed merit costs, from `S06_zoo_cost.csv`, so a timing can be
# quoted in the units the acceptance gate is written in. The merit sums are lower bounds: S06 timed
# the twenty-one merits but not the six descriptive covariates, which are byproducts of merit
# routines that were themselves timed (F-085).
GET_M20_SECONDS = 2.1625390721473303e-06
CHEAP_MERIT_COST = 4.68
FULL_MERIT_COST = 145.25


def paired_by_lattice(result, baseline, label, baseline_label):
    """One McNemar per true Bravais lattice, for the saved model against a baseline.

    Separate from the aggregate test because the aggregate cannot see the failure this task is
    most likely to produce: a model that learns "triclinic candidates are usually wrong", posts a
    good weighted mean and makes triclinic entries worse (S08 handoff, "Pitfalls"). A per-lattice
    delta with no paired test beside it cannot distinguish that from noise, which is exactly the
    question a negative delta raises.
    """
    lattices = result.per_entry['bravais_lattice'].to_numpy()
    ordered = (result.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index()
               ['bravais_lattice'].to_numpy())
    rows = []
    for lattice in sorted(set(lattices)):
        mask = ordered == lattice
        if not mask.any():
            continue
        for metric in ('operating_point', 'top10'):
            test = FomMetrics.mcnemar(result, baseline, metric=metric, subset=mask)
            rows.append(dict(test, bravais_lattice=lattice, label=label,
                             baseline=baseline_label))
    return pd.DataFrame(rows)


def run_paired_only(args, artifact_dir):
    """Re-score `fom-dev` with the saved combiner and emit the per-lattice paired tables.

    A separate entry point rather than a second script: it reuses the loader, the threshold the
    main run chose (read back from the main table) and the saved model, so the numbers are the
    same numbers rather than a near-identical rederivation.
    """
    entries = FomBenchmark.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])
    dev_frames = load_frames(args, entries, covariates, scalers, dev_ids)

    main_table = pd.read_csv(artifact_dir/f'{args.tag}_main_table.csv')
    combiner = FomCombiner.FomCombiner.load(args.models_dir)
    # The winning arm is read from the run's own metadata rather than inferred from the feature
    # count: several arms share a count, and the saved model is whichever one won.
    with open(artifact_dir/f'{args.tag}_meta.json', encoding='utf-8') as handle:
        winner = json.load(handle)['winner']
    threshold = float(main_table.loc[main_table['arm'] == winner, 'threshold'].iloc[0])

    model = evaluate_score(dev_frames, entries, combiner.score, threshold, args.report_split,
                           combiner.score_columns, n_bootstrap=0)
    tables = []
    for merit, higher in BASELINES:
        merit_threshold = float(main_table.loc[main_table['arm'] == f'{merit} (raw)',
                                               'threshold'].iloc[0])
        baseline = evaluate_score(dev_frames, entries, merit, merit_threshold, args.report_split,
                                  higher_is_better=higher, n_bootstrap=0)
        tables.append(paired_by_lattice(model, baseline, winner, f'{merit} (raw)'))
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(artifact_dir/f'{args.tag}_mcnemar_by_lattice.csv', index=False)
    losses = out.loc[(out['metric'] == 'top10') & (out['delta'] < 0)]
    print(f'wrote {artifact_dir}/{args.tag}_mcnemar_by_lattice.csv')
    print(f'lattices with a negative top-10 delta: '
          f'{losses[["bravais_lattice", "baseline", "delta", "p_value"]].to_string(index=False)}')


# ---------------------------------------------------------------------------------------
# Session 2: does it hold up?
# ---------------------------------------------------------------------------------------
# Bravais-lattice mixes for the prior-sensitivity check (PLAN section 6.6). A learned prior is by
# construction the CSD/COD symmetry distribution -- legitimate prior information, but the paper has
# to quantify the dependence rather than assert it. F-086 makes this sharper than it was when the
# plan was written: the extinction group is the single most important feature, so the model *is*
# mostly a symmetry prior and this is the check that matters most.
LATTICE_MIXES = {
    'cnrs': None,
    'uniform': {lattice: 1.0 for lattice in FomMetrics.BRAVAIS_LATTICES},
    'low_symmetry_x3': {lattice: FomMetrics.CNRS_WEIGHTS[lattice]*(
        3.0 if lattice in ('mP', 'mC', 'aP') else 1.0)
        for lattice in FomMetrics.BRAVAIS_LATTICES},
    'high_symmetry_x3': {lattice: FomMetrics.CNRS_WEIGHTS[lattice]*(
        3.0 if lattice in ('cP', 'cI', 'cF', 'tP', 'tI', 'hP', 'hR') else 1.0)
        for lattice in FomMetrics.BRAVAIS_LATTICES},
    }

# Volume tilts, applied on top of the CNRS lattice weighting. The decile is within-lattice
# (METRICS.md section 5), so a tilt shifts the population towards larger or smaller cells *for its
# own symmetry* rather than in absolute volume -- which is the axis the hard stratum is cut on.
DECILE_TILTS = {
    'flat': np.ones(FomMetrics.N_VOLUME_DECILES),
    'large_cells': np.arange(1, FomMetrics.N_VOLUME_DECILES + 1, dtype=float),
    'small_cells': np.arange(FomMetrics.N_VOLUME_DECILES, 0, -1, dtype=float),
    }


def reweighted(per_entry, metric, lattice_weights=None, decile_weights=None):
    """A metric re-weighted over (Bravais lattice x volume decile) cells.

    `FomMetrics.weighted_mean` does this over lattices alone, which is the standing CNRS
    reweighting; PLAN section 6.6 asks for a deliberately shifted volume *and* symmetry mix, so the
    same arithmetic is applied on a second axis. The cell means come from the per-entry flags the
    metrics module produced -- only the weighting is done here.

    Cells with no entries drop out and the weights renormalise over what is present, exactly as
    `weighted_mean` does, so a tilt cannot manufacture a number from an empty stratum.
    """
    lattice_weights = lattice_weights or FomMetrics.CNRS_WEIGHTS
    decile_weights = (np.ones(FomMetrics.N_VOLUME_DECILES) if decile_weights is None
                      else np.asarray(decile_weights, dtype=float))
    frame = per_entry.loc[per_entry[metric].notna(),
                          ['bravais_lattice', 'volume_decile', metric]]
    cells = frame.groupby(['bravais_lattice', 'volume_decile'])[metric].mean()
    weights = np.array([lattice_weights.get(lattice, 0.0)*decile_weights[int(decile)]
                        for lattice, decile in cells.index])
    total = weights.sum()
    return float((cells.to_numpy()*weights).sum()/total) if total > 0 else float('nan')


def per_lattice_thresholds(result, budget):
    """One threshold per Bravais lattice, each at the same false-positive budget *within* it.

    Q33: the combiner's triclinic loss is on the operating point and not on top-10 (F-088), which
    says the score ranks aP as well as `M_sym` does and is *thresholded* worse. A single global
    threshold on a calibrated probability is the suspect: a well-calibrated score reports lower
    confidence where the base rate is lower, and the operating point then penalises the honesty.

    Built from `FomMetrics.threshold_curve` on a lattice-filtered per-entry frame, unweighted --
    within one lattice the CNRS weighting is a constant and cancels.
    """
    thresholds = {}
    for lattice, block in result.per_entry.groupby('bravais_lattice'):
        curve = FomMetrics.threshold_curve(block, weighted=False)
        affordable = curve.loc[curve['false_positive_rate'] <= budget]
        if not affordable.shape[0]:
            affordable = curve.loc[[curve['false_positive_rate'].idxmin()]]
        best = affordable.loc[affordable['operating_point'].idxmax()]
        thresholds[str(lattice)] = float(best['threshold'])
    return thresholds


def offset_score(combiner, thresholds):
    """The combiner's probability minus its own lattice's threshold, so 0 is the accept boundary.

    Subtracting a per-lattice constant is monotone *within* a lattice, so every lattice's internal
    ranking is untouched and only the cross-lattice comparison moves -- the same property S07's
    transforms have, and it is asserted in the tests. The consequence is that this is not purely a
    threshold change: it is a per-lattice recalibration, and its effect on top-10 is reported
    beside its effect on the operating point rather than assumed to be nil.
    """
    def score(frame):
        values = combiner.score(frame)
        offsets = np.array([thresholds.get(str(lattice), 0.0)
                            for lattice in frame['bravais_lattice'].to_numpy()])
        return values - offsets
    return score


def hybrid_score(combiner, knots, lattice='aP'):
    """The combiner everywhere except one lattice, which is scored by `M_sym` put on the same scale.

    The deployable form of "use `M_sym` for triclinic": an isotonic fitted on the calibration split
    maps that lattice's `M_sym` onto P(correct), so the two halves are commensurable and the
    cross-lattice pooling still means something. Scoring one lattice with a raw merit and the rest
    with a probability would not be a score at all.
    """
    def score(frame):
        values = combiner.score(frame)
        mask = (frame['bravais_lattice'].to_numpy() == lattice)
        if mask.any():
            values[mask] = np.interp(frame['M_sym'].to_numpy(dtype=np.float64)[mask], *knots)
        return values
    return score


def run_robustness(args, artifact_dir):
    """Q33 (the triclinic loss), prior sensitivity (PLAN 6.6), and condition transfer (R11)."""
    from sklearn.isotonic import IsotonicRegression

    entries = FomBenchmark.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = split_ids(entries, args.train_split, args.holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    with open(artifact_dir/f'{args.tag}_meta.json', encoding='utf-8') as handle:
        meta = json.load(handle)
    # The feature groups and hyperparameters come off the *saved model's* specification rather than
    # being re-derived from the arm's name: the specification is what the reported number was
    # produced with, and a name is not a spec.
    combiner = FomCombiner.FomCombiner.load(args.models_dir)
    groups = tuple(combiner.groups)
    args.model_params = dict(combiner.meta.get('params')
                             or dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63))
    print(f'groups {groups}; params {args.model_params}')

    print('loading frames...')
    started = time.perf_counter()
    fit_frames = load_frames(args, entries, covariates, scalers, fit_ids,
                             n_negatives=args.n_negatives, seed=args.seed)
    cal_frames = load_frames(args, entries, covariates, scalers, cal_ids)
    dev_frames = load_frames(args, entries, covariates, scalers, dev_ids)
    print(f'  loaded ({time.perf_counter() - started:.0f} s)')

    columns = combiner.score_columns
    main_table = pd.read_csv(artifact_dir/f'{args.tag}_main_table.csv')
    winner = meta['winner']
    threshold = float(main_table.loc[main_table['arm'] == winner, 'threshold'].iloc[0])
    budget = float(meta['matched_fpr_budget'])

    # ---- Q33 -------------------------------------------------------------------------
    cal = evaluate_score(cal_frames, entries, combiner.score, threshold, args.train_split,
                         columns)
    thresholds = per_lattice_thresholds(cal, budget)
    print('per-lattice thresholds: '
          + ', '.join(f'{k} {v:.4f}' for k, v in sorted(thresholds.items())))

    cal_pool = pd.concat(cal_frames, ignore_index=True)
    ap = cal_pool.loc[cal_pool['bravais_lattice'] == 'aP']
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    isotonic.fit(ap['M_sym'].to_numpy(dtype=np.float64),
                 FomMetrics.as_bool(ap['is_correct']).astype(np.float64))
    knots = (np.asarray(isotonic.X_thresholds_, dtype=np.float64),
             np.asarray(isotonic.y_thresholds_, dtype=np.float64))
    del cal_pool, ap

    variants = [
        ('combiner (global threshold)', combiner.score, threshold, columns),
        ('combiner (per-lattice thresholds)', offset_score(combiner, thresholds), 0.0, columns),
        ('hybrid: M_sym-calibrated for aP', hybrid_score(combiner, knots), threshold,
         tuple(sorted(set(columns) | {'M_sym'}))),
        ]
    rows, results = [], {}
    for label, score, cut, needed in variants:
        dev = evaluate_score(dev_frames, entries, score, cut, args.report_split, needed,
                             n_bootstrap=args.n_bootstrap, seed=args.seed)
        row = scope_row(dev, arm=label, groups='+'.join(groups), objective='none',
                        n_features=len(combiner.names), threshold=cut, threshold_rule='q33',
                        seconds_fit=0.0, per_lattice_models=False)
        row.update(per_lattice(dev, 'dev'))
        rows.append(row)
        results[label] = dev
        print(f'  {label:36s} op {row["operating_point"]:.4f}  top10 {row["top10"]:.4f}')

    # The per-lattice diagnosis: which half of the criterion is aP losing?
    diagnosis = []
    for merit, higher in BASELINES:
        merit_threshold = float(main_table.loc[main_table['arm'] == f'{merit} (raw)',
                                              'threshold'].iloc[0])
        baseline = evaluate_score(dev_frames, entries, merit, merit_threshold, args.report_split,
                                  higher_is_better=higher, n_bootstrap=0)
        block = baseline.stratum('bravais_lattice').copy()
        block['arm'] = f'{merit} (raw)'
        diagnosis.append(block)
        results[f'{merit} (raw)'] = baseline
    for label in results:
        if label.startswith('combiner') or label.startswith('hybrid'):
            block = results[label].stratum('bravais_lattice').copy()
            block['arm'] = label
            diagnosis.append(block)
    pd.concat(diagnosis, ignore_index=True).to_csv(
        artifact_dir/f'{args.tag}_q33_by_lattice.csv', index=False)

    paired = []
    for label in ('combiner (per-lattice thresholds)', 'hybrid: M_sym-calibrated for aP'):
        for baseline in ('combiner (global threshold)', 'M_sym (raw)'):
            paired.append(paired_by_lattice(results[label], results[baseline], label, baseline))
    pd.concat(paired, ignore_index=True).to_csv(
        artifact_dir/f'{args.tag}_q33_mcnemar.csv', index=False)

    # ---- prior sensitivity ------------------------------------------------------------
    sensitivity = []
    for label, result in results.items():
        for mix, lattice_weights in LATTICE_MIXES.items():
            for tilt, decile_weights in DECILE_TILTS.items():
                sensitivity.append(dict(
                    arm=label, lattice_mix=mix, volume_tilt=tilt,
                    operating_point=reweighted(result.per_entry, 'operating_point',
                                               lattice_weights, decile_weights),
                    top10=reweighted(result.per_entry, 'top10', lattice_weights, decile_weights),
                    found=reweighted(result.per_entry, 'found', lattice_weights, decile_weights),
                    ))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(artifact_dir/f'{args.tag}_prior_sensitivity.csv', index=False)
    base = sensitivity[(sensitivity['lattice_mix'] == 'cnrs')
                       & (sensitivity['volume_tilt'] == 'flat')].set_index('arm')
    worst = sensitivity.assign(
        delta=sensitivity['operating_point'] - sensitivity['arm'].map(base['operating_point']))
    print('prior sensitivity, worst shift per arm:')
    for arm, block in worst.groupby('arm'):
        row = block.loc[block['delta'].idxmin()]
        print(f'  {arm:36s} {row["lattice_mix"]}/{row["volume_tilt"]}  '
              f'{row["operating_point"]:.4f} ({row["delta"]:+.4f})')

    # ---- leave one condition out ------------------------------------------------------
    loco = []
    for held_out in args.bundles:
        keep = [index for index, bundle in enumerate(args.bundles) if bundle != held_out]
        held = [index for index, bundle in enumerate(args.bundles) if bundle == held_out]
        if not held or len(keep) < 2:
            continue
        trained = FomCombiner.FomCombiner.fit(
            [fit_frames[index] for index in keep], groups=groups, scalers=scalers,
            objective='pointwise', seed=args.seed, **args.model_params)
        trained.fit_calibrators([cal_frames[index] for index in keep])
        in_cal = evaluate_score([cal_frames[index] for index in keep], entries, trained.score,
                                None, args.train_split, trained.score_columns)
        choice, rule = choose_threshold(in_cal, budget)
        target = [dev_frames[index] for index in held]
        out_of = evaluate_score(target, entries, trained.score, float(choice.threshold),
                                args.report_split, trained.score_columns, n_bootstrap=0)
        within = evaluate_score(target, entries, combiner.score, threshold, args.report_split,
                                columns, n_bootstrap=0)
        loco.append(dict(
            held_out_bundle=held_out,
            label=FomMetrics.BUNDLE_LABELS.get(held_out, held_out),
            n_entries=int(out_of.aggregate['n_entries'].iloc[0]),
            threshold_rule=rule,
            op_trained_without=float(out_of.metric('operating_point')),
            op_trained_with=float(within.metric('operating_point')),
            top10_trained_without=float(out_of.metric('top10')),
            top10_trained_with=float(within.metric('top10')),
            ceiling=float(out_of.metric('ceiling_rescorer')),
            ))
        loco[-1]['op_transfer_cost'] = (loco[-1]['op_trained_with']
                                       - loco[-1]['op_trained_without'])
        print(f'  LOCO {held_out:22s} without {loco[-1]["op_trained_without"]:.4f}  '
              f'with {loco[-1]["op_trained_with"]:.4f}  '
              f'cost {loco[-1]["op_transfer_cost"]:+.4f}')
    pd.DataFrame(loco).to_csv(artifact_dir/f'{args.tag}_loco.csv', index=False)

    pd.DataFrame(rows).to_csv(artifact_dir/f'{args.tag}_q33_table.csv', index=False)
    with open(artifact_dir/f'{args.tag}_robustness_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(dict(commit=commit_hash(), per_lattice_thresholds=thresholds,
                       global_threshold=threshold, matched_fpr_budget=budget,
                       groups=list(groups), params=args.model_params,
                       lattice_mixes=sorted(LATTICE_MIXES), volume_tilts=sorted(DECILE_TILTS),
                       seconds=time.perf_counter() - started), handle, indent=2)
    print(f'wrote {artifact_dir}/{args.tag}_{{q33_table,q33_by_lattice,q33_mcnemar,'
          f'prior_sensitivity,loco}}.csv')


def run_distil(args, artifact_dir):
    """The fast form, and STATUS Q4: is a small MLP as numpy matmuls slower than `get_M20`?

    F-085 measured the tree at 2.46x `get_M20`, so the inner-loop budget is missed before a single
    feature is computed. Two students are fitted, because the two halves of that cost are separate
    problems: one over the full feature set, which answers Q4 on its own terms, and one over the ten
    merits priced below 1x, which is the variant S14 could actually deploy.
    """
    entries = FomBenchmark.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = split_ids(entries, args.train_split, args.holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    with open(artifact_dir/f'{args.tag}_meta.json', encoding='utf-8') as handle:
        meta = json.load(handle)
    budget = float(meta['matched_fpr_budget'])
    teacher = FomCombiner.FomCombiner.load(args.models_dir)
    args.model_params = dict(teacher.meta.get('params')
                             or dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63))

    print('loading frames...')
    started = time.perf_counter()
    fit_frames = load_frames(args, entries, covariates, scalers, fit_ids,
                             n_negatives=args.n_negatives, seed=args.seed)
    cal_frames = load_frames(args, entries, covariates, scalers, cal_ids)
    dev_frames = load_frames(args, entries, covariates, scalers, dev_ids)
    print(f'  loaded ({time.perf_counter() - started:.0f} s)')

    # The cheap teacher is refitted here rather than read back: the main stage measured it but did
    # not persist it, and a student needs its own teacher's exact feature list.
    cheap_names = FomCombiner.affordable_features(teacher.names, CHEAP_MERITS)
    cheap_teacher = _restrict(teacher, cheap_names, fit_frames, args)
    cheap_teacher.fit_calibrators(cal_frames)

    students = {}
    for label, parent in (('full', teacher), ('cheap', cheap_teacher)):
        student = FomCombiner.DistilledCombiner.distil(
            parent, cal_frames, hidden=(32, 16), seed=args.seed)
        students[label] = (parent, student)
        print(f'  distilled {label:6s} {len(parent.names):3d} features, '
              f'teacher correlation {student.meta["train_correlation"]:.4f}')
        student.save(Path(args.models_dir)/f'distilled_{label}')

    # ---- timing ----------------------------------------------------------------------
    timings = []
    sample = dev_frames[0]
    for label, (parent, student) in students.items():
        matrix = parent.design_matrix(sample)
        for name, model in (('tree', parent), ('mlp', student)):
            best = np.inf
            for _ in range(5):
                mark = time.perf_counter()
                model.predict_batch(matrix)
                best = min(best, time.perf_counter() - mark)
            timings.append(dict(
                feature_set=label, form=name, n_features=len(parent.names),
                n_candidates=int(matrix.shape[0]),
                seconds_per_candidate=best/matrix.shape[0],
                cost_vs_get_M20=(best/matrix.shape[0])/GET_M20_SECONDS,
                merit_cost_vs_get_M20=(CHEAP_MERIT_COST if label == 'cheap'
                                       else FULL_MERIT_COST),
                ))
            print(f'  {label:6s} {name:4s} {timings[-1]["cost_vs_get_M20"]:.2f}x get_M20')
    pd.DataFrame(timings).to_csv(artifact_dir/f'{args.tag}_distil_cost.csv', index=False)

    # ---- what the fast form costs in accuracy ----------------------------------------
    rows = []
    for label, (parent, student) in students.items():
        for name, model in (('tree', parent), ('mlp', student)):
            columns = model.score_columns
            cal = evaluate_score(cal_frames, entries, model.score, None, args.train_split,
                                 columns)
            choice, rule = choose_threshold(cal, budget)
            dev = evaluate_score(dev_frames, entries, model.score, float(choice.threshold),
                                 args.report_split, columns, n_bootstrap=args.n_bootstrap,
                                 seed=args.seed)
            row = scope_row(dev, arm=f'{label}/{name}', groups='+'.join(parent.groups),
                            objective='none', n_features=len(parent.names),
                            threshold=float(choice.threshold), threshold_rule=rule,
                            seconds_fit=0.0, per_lattice_models=False)
            row.update(per_lattice(dev, 'dev'))
            rows.append(row)
            print(f'  {label}/{name:4s} operating point {row["operating_point"]:.4f}  '
                  f'top10 {row["top10"]:.4f}')
    pd.DataFrame(rows).to_csv(artifact_dir/f'{args.tag}_distil_table.csv', index=False)

    with open(artifact_dir/f'{args.tag}_distil_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(dict(commit=commit_hash(), hidden=[32, 16], seed=args.seed,
                       cheap_features=list(cheap_names), cheap_merits=list(CHEAP_MERITS),
                       get_M20_seconds=GET_M20_SECONDS,
                       cheap_merit_cost=CHEAP_MERIT_COST, full_merit_cost=FULL_MERIT_COST,
                       students={label: student.meta
                                 for label, (_, student) in students.items()},
                       seconds=time.perf_counter() - started), handle, indent=2)
    print(f'wrote {artifact_dir}/{args.tag}_distil_{{cost,table,meta}}.*')


def _uniform_random(frame):
    """A score with no information and no ties, so the tie-break contributes nothing."""
    return np.random.default_rng(12345).random(frame.shape[0])


def split_ids(entries, split, holdout_fraction, seed):
    """Source entries of one split, divided into a fit part and a calibration part.

    By **source entry**, never by candidate (PROTOCOL section 3 rule 5): one crystal appears under
    six condition bundles with correlated noise, so splitting rows would put near-duplicates of
    the same pattern on both sides of the calibration boundary.
    """
    ids = np.array(sorted(set(entries.loc[entries['split'] == split, 'entry_id'])))
    if not holdout_fraction:
        return set(ids), set()
    rng = np.random.default_rng(seed)
    held = rng.permutation(ids.size) < int(round(holdout_fraction*ids.size))
    return set(ids[~held]), set(ids[held])


def load_frames(args, entries, covariates, scalers, keep_ids, n_negatives=None, seed=12345):
    """Assembled frames for one set of entries, held in memory.

    Held rather than streamed because every arm, ablation and permutation makes another pass over
    the same rows, and re-reading 2.3 GB of parquet per pass is hours of I/O for no benefit --
    the same argument `run_fom_null.load_split` makes.
    """
    frames = []
    for frame in FomCombiner.combiner_frames(args.benchmark_dir, args.feature_dir, args.bundles,
                                             keep_ids, covariates, scalers, groups=LOAD_GROUPS):
        if n_negatives is not None:
            frame = FomCombiner.subsample_negatives(frame, n_negatives, seed)
        frames.append(frame)
    return frames


def tune(fit_frames, groups, scalers, args):
    """Pick hyperparameters by GroupKFold over source entries, inside the fit split.

    Grouped by `entry_id` and not by row: one crystal appears under six condition bundles with
    correlated noise, so an ungrouped fold would score a model on near-duplicates of what it was
    fitted on and every configuration would look equally good.
    """
    from sklearn.model_selection import GroupKFold

    frame = pd.concat(fit_frames, ignore_index=True) if len(fit_frames) > 1 else fit_frames[0]
    names, categorical = FomCombiner.feature_specification(groups, scalers)
    combiner = FomCombiner.FomCombiner(
        names=names, categorical=categorical,
        categories={name: {value: code + 1 for code, value in
                           enumerate(sorted(pd.unique(frame[name].astype(str))))}
                    for name in categorical},
        groups=groups)
    matrix = combiner.design_matrix(frame)
    target = FomMetrics.as_bool(frame['is_correct']).astype(np.int32)
    entry = frame['entry_id'].to_numpy()

    rows = []
    for settings in TUNING_GRID:
        scores = []
        for train_index, test_index in GroupKFold(n_splits=3).split(matrix, target, entry):
            model = FomCombiner._fit_pointwise(
                matrix[train_index], target[train_index], combiner.categorical_indices,
                args.seed, **settings)
            probability = model.predict_proba(matrix[test_index])[:, 1]
            scores.append(FomMetrics.average_precision(probability, target[test_index] == 1))
        rows.append(dict(settings, average_precision=float(np.mean(scores))))
        print(f'    {settings} -> AP {rows[-1]["average_precision"]:.4f}')
    table = pd.DataFrame(rows).sort_values('average_precision', ascending=False)
    best = {key: table.iloc[0][key] for key in TUNING_GRID[0]}
    return {key: int(value) if key != 'learning_rate' else float(value)
            for key, value in best.items()}, table


def evaluate_score(frames, entries, score, threshold, split_label, score_columns=(),
                   higher_is_better=True, n_bootstrap=0, seed=12345, strata=None,
                   hard_min_decile=FomMetrics.HARD_MIN_DECILE):
    return FomMetrics.evaluate(
        frames, score=score, higher_is_better=higher_is_better, threshold=threshold,
        entries=entries, split=split_label, score_columns=list(score_columns),
        strata=strata if strata is not None else FomMetrics.DEFAULT_STRATA,
        n_bootstrap=n_bootstrap, seed=seed, hard_min_decile=hard_min_decile,
        )


def choose_threshold(selection_result, budget):
    """The matched false-positive budget, or Youden if the budget is unreachable.

    A model that never scores above the budget's implied threshold has no operating point to
    maximise there, and `select_threshold` raises rather than inventing one. Falling back is
    recorded on the row so a threshold chosen the other way is never mistaken for a matched one.
    """
    try:
        choice = FomMetrics.select_threshold(selection_result, objective='operating_point',
                                             max_false_positive_rate=budget)
        return choice, 'matched_fpr'
    except ValueError:
        return FomMetrics.select_threshold(selection_result, objective='youden'), 'youden'


def scope_row(result, **identifiers):
    """One flat row of the metrics a leaderboard needs, from a `MetricsResult`."""
    row = dict(identifiers)
    for metric in ('operating_point', 'top1', 'top10', 'rank_only', 'threshold_only', 'mrr',
                   'ceiling_rescorer', 'reported', 'false_positive', 'precision',
                   'operating_point_given_found'):
        row[metric] = float(result.metric(metric))
    row['operating_point_ci_low'] = float(result.metric('operating_point_ci_low'))
    row['operating_point_ci_high'] = float(result.metric('operating_point_ci_high'))
    hard = result.hard
    if hard is not None and hard.shape[0]:
        row['hard_top10'] = float(hard['top10'].iloc[0])
        row['hard_operating_point'] = float(hard['operating_point'].iloc[0])
        row['hard_op_given_found'] = float(hard['operating_point_given_found'].iloc[0])
        row['hard_ceiling'] = float(hard['ceiling_rescorer'].iloc[0])
        row['hard_n_entries'] = int(hard['n_entries'].iloc[0])
    return row


def per_lattice(result, prefix):
    """Top-10 and operating point per *true* Bravais lattice, never reweighted.

    The named failure mode for this task is a model that learns "triclinic candidates are usually
    wrong", posts a good aggregate and makes triclinic entries worse. That is only visible here.
    """
    table = result.stratum('bravais_lattice')
    row = {}
    for _, entry in table.iterrows():
        row[f'{prefix}_top10_{entry["level"]}'] = float(entry['top10'])
        row[f'{prefix}_op_{entry["level"]}'] = float(entry['operating_point'])
    return row


def fit_arm(name, groups, fit_frames, cal_frames, dev_frames, entries, scalers, budget, args,
            names_override=None, per_lattice_models=False):
    """One model: fit, calibrate, choose a threshold on the calibration split, report on dev."""
    started = time.perf_counter()
    combiner = FomCombiner.FomCombiner.fit(
        fit_frames, groups=groups, scalers=scalers, objective=args.objective, seed=args.seed,
        **args.model_params)
    if names_override is not None:
        combiner = _restrict(combiner, names_override, fit_frames, args)
    combiner.fit_calibrators(cal_frames)
    fitted = time.perf_counter() - started

    columns = combiner.score_columns
    cal = evaluate_score(cal_frames, entries, combiner.score, None, 'fom-train', columns)
    choice, rule = choose_threshold(cal, budget)
    dev = evaluate_score(dev_frames, entries, combiner.score, float(choice.threshold), 'fom-dev',
                         columns, n_bootstrap=args.n_bootstrap, seed=args.seed)
    FomMetrics.check_threshold_transfer(choice, dev)
    widened = evaluate_score(dev_frames, entries, combiner.score, float(choice.threshold),
                             'fom-dev', columns, hard_min_decile=HARD_THRESHOLD_DECILE)

    row = scope_row(dev, arm=name, groups='+'.join(groups), objective=args.objective,
                    n_features=len(combiner.names), threshold=float(choice.threshold),
                    threshold_rule=rule, seconds_fit=fitted,
                    per_lattice_models=bool(per_lattice_models))
    row['hard_op_given_found_d6'] = float(widened.hard['operating_point_given_found'].iloc[0])
    row['hard_n_entries_d6'] = int(widened.hard['n_entries'].iloc[0])
    row.update(per_lattice(dev, 'dev'))
    return combiner, dev, row


def _shuffle_labels(frame, seed):
    """`is_correct` permuted within each (entry, bundle), leaving every feature untouched."""
    codes, _ = FomMetrics._group_codes(frame['entry_id'].to_numpy(),
                                       frame['condition_bundle'].to_numpy())
    labels = FomMetrics.as_bool(frame['is_correct'])
    rng = np.random.default_rng(seed)
    # Both orderings walk the groups in the same sequence; `base` lists each group's rows in
    # their original order and `drawn` in a random one, so the assignment is a permutation
    # confined to the group.
    base = np.lexsort((np.arange(codes.size), codes))
    drawn = np.lexsort((rng.random(codes.size), codes))
    shuffled = np.empty_like(labels)
    shuffled[base] = labels[drawn]
    return frame.assign(is_correct=shuffled)


def _restrict(combiner, names, fit_frames, args):
    """Refit the same model class on a subset of its own feature columns."""
    restricted = FomCombiner.FomCombiner(
        names=names, categorical=tuple(n for n in combiner.categorical if n in names),
        categories={k: v for k, v in combiner.categories.items() if k in names},
        groups=combiner.groups, objective=combiner.objective,
        )
    frame = pd.concat(fit_frames, ignore_index=True) if len(fit_frames) > 1 else fit_frames[0]
    matrix = restricted.design_matrix(frame)
    target = FomMetrics.as_bool(frame['is_correct']).astype(np.int32)
    restricted.model = FomCombiner._fit_pointwise(
        matrix, target, restricted.categorical_indices, args.seed, **args.model_params)
    restricted.meta = dict(combiner.meta, restricted_to=len(names))
    return restricted


def main():
    parser = argparse.ArgumentParser(
        description='S08: fit and report the learned combiner over the FOM zoo.')
    parser.add_argument('--benchmark-dir', default=os.path.join('mlindex', 'data',
                                                                'fom_benchmark'))
    parser.add_argument('--feature-dir', default=os.path.join('mlindex', 'data', 'fom_features'))
    parser.add_argument('--null-dir', default=os.path.join('mlindex', 'models', 'fom_null'))
    parser.add_argument('--models-dir', default=os.path.join('mlindex', 'models',
                                                             'fom_combiner'))
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--bundles', nargs='+', default=list(EVALUABLE_BUNDLES))
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--holdout-fraction', type=float, default=0.2,
                        help='fraction of fom-train held out for calibration and thresholds')
    parser.add_argument('--n-negatives', type=int, default=40,
                        help='incorrect candidates kept per entry-bundle when fitting')
    parser.add_argument('--objective', default='pointwise',
                        choices=('pointwise', 'lambdarank'))
    parser.add_argument('--max-iter', type=int, default=None,
                        help='fix the boosting rounds instead of tuning them')
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--skip-tuning', action='store_true')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--stage', default='main',
                        choices=('main', 'paired', 'robustness', 'distil'),
                        help='main fits every arm; paired emits the per-lattice paired tests; '
                             'robustness runs Q33, prior sensitivity and leave-one-condition-out; '
                             'distil fits the fast form and times it. Each later stage reads the '
                             'artefacts and the saved model the main stage wrote.')
    parser.add_argument('--skip-lambdarank', action='store_true')
    parser.add_argument('--skip-importance', action='store_true')
    parser.add_argument('--importance-repeats', type=int, default=1)
    parser.add_argument('--tag', default='S08_combiner')
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.stage != 'main':
        {'paired': run_paired_only, 'robustness': run_robustness,
         'distil': run_distil}[args.stage](args, artifact_dir)
        return

    entries = FomBenchmark.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = split_ids(entries, args.train_split, args.holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])
    print(f'{len(scalers)} normalisers; entries fit/cal/dev '
          f'{len(fit_ids)}/{len(cal_ids)}/{len(dev_ids)}')

    print('loading frames...')
    fit_frames = load_frames(args, entries, covariates, scalers, fit_ids,
                             n_negatives=args.n_negatives, seed=args.seed)
    cal_frames = load_frames(args, entries, covariates, scalers, cal_ids)
    dev_frames = load_frames(args, entries, covariates, scalers, dev_ids)
    n_fit = sum(frame.shape[0] for frame in fit_frames)
    n_cal = sum(frame.shape[0] for frame in cal_frames)
    n_dev = sum(frame.shape[0] for frame in dev_frames)
    print(f'  candidates fit/cal/dev {n_fit:,}/{n_cal:,}/{n_dev:,}  '
          f'({time.perf_counter() - started:.0f} s)')

    # ---- hyperparameters, chosen inside fom-train ------------------------------------
    if args.skip_tuning or args.max_iter is not None:
        args.model_params = dict(max_iter=args.max_iter or 400,
                                 learning_rate=args.learning_rate or 0.06)
        tuning = pd.DataFrame([dict(args.model_params, average_precision=np.nan,
                                    note='tuning skipped')])
    else:
        print('tuning on the fit split (GroupKFold over source entries)...')
        args.model_params, tuning = tune(fit_frames, ARMS[-1][1], scalers, args)
        print(f'  chosen {args.model_params}')
    tuning.to_csv(artifact_dir/f'{args.tag}_tuning.csv', index=False)

    # ---- baselines -------------------------------------------------------------------
    # The budget is raw M20's own false-positive rate at de Wolff's threshold of 10, measured on
    # the calibration split so every score in the table is thresholded on the same entries.
    m20_cal = evaluate_score(cal_frames, entries, 'M20', DEWOLFF_THRESHOLD, args.train_split)
    budget = float(m20_cal.metric('false_positive'))
    print(f'matched false-positive budget {budget:.5f}')

    rows, results = [], {}

    # The floor a *degenerate* score inherits for free. `reduce_pool` breaks ties by Bravais
    # lattice in `FomMetrics.BRAVAIS_LATTICES` order, which runs cubic first, so a score with many
    # ties is silently given a symmetry prior -- and F-069 says symmetry lowering is the dominant
    # failure, so that prior is a good one. The tie-break is label-independent, as METRICS.md says,
    # but it is not *prior*-independent, and every rank metric in this project is measured above
    # this line rather than above zero. Reported so the reader can see where zero actually is.
    for label, score in (('constant (tie-break floor)', lambda frame: np.ones(frame.shape[0])),
                         ('uniform random', _uniform_random)):
        dev = evaluate_score(dev_frames, entries, score, None, args.report_split,
                             n_bootstrap=args.n_bootstrap, seed=args.seed)
        row = scope_row(dev, arm=label, groups='floor', objective='none', n_features=0,
                        threshold=np.nan, threshold_rule='none', seconds_fit=0.0,
                        per_lattice_models=False)
        row.update(per_lattice(dev, 'dev'))
        rows.append(row)
        results[label] = dev
        print(f'  floor {label:26s} top10 {row["top10"]:.4f}')

    for merit, higher in BASELINES:
        cal = evaluate_score(cal_frames, entries, merit, None, args.train_split,
                             higher_is_better=higher)
        choice, rule = choose_threshold(cal, budget)
        threshold = float(choice.threshold) if higher else -float(choice.threshold)
        dev = evaluate_score(dev_frames, entries, merit, threshold, args.report_split,
                             higher_is_better=higher, n_bootstrap=args.n_bootstrap,
                             seed=args.seed)
        FomMetrics.check_threshold_transfer(choice, dev)
        widened = evaluate_score(dev_frames, entries, merit, threshold, args.report_split,
                                 higher_is_better=higher,
                                 hard_min_decile=HARD_THRESHOLD_DECILE)
        row = scope_row(dev, arm=f'{merit} (raw)', groups='baseline', objective='none',
                        n_features=1, threshold=threshold, threshold_rule=rule, seconds_fit=0.0,
                        per_lattice_models=False)
        row['hard_op_given_found_d6'] = float(
            widened.hard['operating_point_given_found'].iloc[0])
        row['hard_n_entries_d6'] = int(widened.hard['n_entries'].iloc[0])
        row.update(per_lattice(dev, 'dev'))
        rows.append(row)
        results[f'{merit} (raw)'] = dev
        print(f'  baseline {merit:8s} operating point {row["operating_point"]:.4f}  '
              f'top10 {row["top10"]:.4f}')

    # ---- the arms --------------------------------------------------------------------
    models = {}
    for name, groups in ARMS:
        combiner, dev, row = fit_arm(name, groups, fit_frames, cal_frames, dev_frames, entries,
                                     scalers, budget, args)
        rows.append(row)
        results[name] = dev
        models[name] = combiner
        print(f'  arm {name:12s} features {row["n_features"]:3d}  '
              f'operating point {row["operating_point"]:.4f}  top10 {row["top10"]:.4f}  '
              f'({row["seconds_fit"]:.0f} s fit)')

    best = max(ARMS, key=lambda arm: results[arm[0]].metric('operating_point'))
    print(f'best arm: {best[0]}')

    # ---- objective, ablations, and the inner-loop variant -----------------------------
    if not args.skip_lambdarank:
        try:
            ranker_args = argparse.Namespace(**vars(args))
            ranker_args.objective = 'lambdarank'
            combiner, dev, row = fit_arm('lambdarank', best[1], fit_frames, cal_frames,
                                         dev_frames, entries, scalers, budget, ranker_args)
            rows.append(row)
            results['lambdarank'] = dev
            models['lambdarank'] = combiner
            print(f'  lambdarank operating point {row["operating_point"]:.4f}')
        except ImportError:
            print('  lambdarank skipped: lightgbm is not installed (optional, training only)')

    for name, dropped in ABLATIONS:
        groups = tuple(group for group in best[1] if group not in dropped)
        if not groups:
            continue
        _, dev, row = fit_arm(name, groups, fit_frames, cal_frames, dev_frames, entries,
                              scalers, budget, args)
        rows.append(row)
        results[name] = dev
        print(f'  ablation {name:16s} operating point {row["operating_point"]:.4f}')

    in_sample_groups = tuple(sorted(set(best[1]) | {'in_sample'}))
    _, dev, row = fit_arm('add_in_sample_sigma', in_sample_groups, fit_frames, cal_frames,
                          dev_frames, entries, scalers, budget, args)
    rows.append(row)
    results['add_in_sample_sigma'] = dev
    print(f'  in-sample sigma operating point {row["operating_point"]:.4f}')

    # Two controls, and they answer different questions. Labels are permuted **within each
    # entry**, so the positive count per entry is preserved and only the association between a
    # candidate and its correctness is destroyed.
    #
    #   * `prior_only_control` shuffles the *fit* labels and leaves the per-lattice isotonic
    #     calibrators fitted on real ones. The estimator then carries no fit information at all
    #     and the only thing left is the per-lattice base rate -- so this measures what a pure
    #     Bravais-lattice prior is worth, which is PLAN section 3's third term on its own.
    #     Isotonic is monotone, so it cannot reorder *within* a lattice; everything it buys is
    #     cross-lattice, which is exactly where F-074 says the problem lives.
    #   * `label_shuffled_control` shuffles both. It has to land at chance. If it does not, the
    #     harness is measuring itself and no other number here means anything.
    shuffled_fit = [_shuffle_labels(frame, args.seed) for frame in fit_frames]
    shuffled_cal = [_shuffle_labels(frame, args.seed + 1) for frame in cal_frames]
    for control, cal_source in (('prior_only_control', cal_frames),
                                ('label_shuffled_control', shuffled_cal)):
        _, dev, row = fit_arm(control, best[1], shuffled_fit, cal_source, dev_frames, entries,
                              scalers, budget, args)
        rows.append(row)
        results[control] = dev
        print(f'  {control:24s} operating point {row["operating_point"]:.4f}  '
              f'top10 {row["top10"]:.4f}')

    cheap_names = FomCombiner.affordable_features(models[best[0]].names, CHEAP_MERITS)
    _, dev, row = fit_arm('cheap_features', best[1], fit_frames, cal_frames, dev_frames, entries,
                          scalers, budget, args, names_override=cheap_names)
    rows.append(row)
    results['cheap_features'] = dev
    print(f'  cheap-feature variant {len(cheap_names)} features, '
          f'operating point {row["operating_point"]:.4f}')

    table = pd.DataFrame(rows)
    for baseline in ('M20 (raw)', 'M_sym (raw)'):
        reference = float(table.loc[table['arm'] == baseline, 'operating_point'].iloc[0])
        ceiling = float(table.loc[table['arm'] == baseline, 'ceiling_rescorer'].iloc[0])
        suffix = 'M20' if baseline.startswith('M20') else 'M_sym'
        table[f'delta_op_vs_{suffix}'] = table['operating_point'] - reference
        table[f'headroom_fraction_vs_{suffix}'] = (
            (table['operating_point'] - reference)/max(ceiling - reference, 1e-12))
    table.to_csv(artifact_dir/f'{args.tag}_main_table.csv', index=False)

    # ---- paired comparisons ----------------------------------------------------------
    paired = []
    for label, result in results.items():
        for baseline in ('M20 (raw)', 'M_sym (raw)'):
            if label == baseline:
                continue
            for subset in (None, 'hard'):
                for metric in ('operating_point', 'top10'):
                    test = FomMetrics.mcnemar(result, results[baseline], metric=metric,
                                              subset=subset)
                    paired.append(dict(test, label=label, baseline=baseline))
    pd.DataFrame(paired).to_csv(artifact_dir/f'{args.tag}_mcnemar.csv', index=False)

    # ---- calibration -----------------------------------------------------------------
    winner_name = best[0]
    if 'lambdarank' in models and (results['lambdarank'].metric('operating_point') >
                                   results[best[0]].metric('operating_point')):
        winner_name = 'lambdarank'
    winner = models[winner_name]
    print(f'winner: {winner_name}')
    calibration_rows = []
    for frame in dev_frames:
        calibration_rows.append(pd.DataFrame({
            'probability': winner.score(frame),
            'is_correct': FomMetrics.as_bool(frame['is_correct']),
            'bravais_lattice': frame['bravais_lattice'].to_numpy(),
            }))
    calibration = pd.concat(calibration_rows, ignore_index=True)
    reliability, ece, brier = FomMetrics.reliability(calibration['probability'].to_numpy(),
                                                    calibration['is_correct'].to_numpy())
    reliability['scope'] = 'aggregate'
    per_bl = []
    for lattice, block in calibration.groupby('bravais_lattice'):
        if block.shape[0] < 200:
            continue
        table_bl, ece_bl, brier_bl = FomMetrics.reliability(
            block['probability'].to_numpy(), block['is_correct'].to_numpy())
        table_bl['scope'] = str(lattice)
        table_bl['ece'] = ece_bl
        table_bl['brier'] = brier_bl
        per_bl.append(table_bl)
    reliability['ece'] = ece
    reliability['brier'] = brier
    pd.concat([reliability] + per_bl, ignore_index=True).to_csv(
        artifact_dir/f'{args.tag}_calibration.csv', index=False)
    print(f'  calibration ECE {ece:.4f}  Brier {brier:.5f}')

    # ---- permutation importance ------------------------------------------------------
    if not args.skip_importance:
        importance = permutation_importance(winner, dev_frames, entries, args)
        importance.to_csv(artifact_dir/f'{args.tag}_importance.csv', index=False)
        print(f'  importance: top feature '
              f'{importance.iloc[0]["feature"]} ({importance.iloc[0]["delta_top10"]:+.4f})')

    # ---- stratified deltas -----------------------------------------------------------
    strata_rows = []
    for label in ('M20 (raw)', 'M_sym (raw)', best[0], 'cheap_features'):
        for stratum in ('bravais_lattice', 'volume_decile', 'condition_bundle'):
            block = results[label].stratum(stratum).copy()
            block['arm'] = label
            strata_rows.append(block)
    pd.concat(strata_rows, ignore_index=True).to_csv(
        artifact_dir/f'{args.tag}_by_stratum.csv', index=False)

    # ---- persist the winner and the metadata -----------------------------------------
    winner.save(args.models_dir)
    cost = timing(winner, dev_frames[0])
    pd.DataFrame([cost]).to_csv(artifact_dir/f'{args.tag}_cost.csv', index=False)
    meta = dict(
        commit=commit_hash(), tag=args.tag, bundles=list(args.bundles),
        train_split=args.train_split, report_split=args.report_split,
        holdout_fraction=args.holdout_fraction, n_negatives=args.n_negatives,
        seed=args.seed, n_bootstrap=args.n_bootstrap, matched_fpr_budget=budget,
        hard_threshold_decile=HARD_THRESHOLD_DECILE, objective=args.objective,
        n_entries_fit=len(fit_ids), n_entries_cal=len(cal_ids), n_entries_dev=len(dev_ids),
        n_candidates_fit=int(n_fit), n_candidates_cal=int(n_cal), n_candidates_dev=int(n_dev),
        n_features=len(winner.names), features=list(winner.names),
        best_arm=best[0], winner=winner_name, models_dir=str(args.models_dir),
        ece=float(ece), brier=float(brier),
        cheap_features=list(cheap_names), cheap_merits=list(CHEAP_MERITS),
        r1_note=('every scaled feature except the two analytic ones is fitted on negatives '
                 'censored at M20 >= 5, lattice-dependently (STATUS section 7, R1)'),
        seconds=time.perf_counter() - started,
        )
    with open(artifact_dir/f'{args.tag}_meta.json', 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print(f'done in {meta["seconds"]/60:.1f} min')


def permutation_importance(combiner, frames, entries, args):
    """Drop in cross-lattice top-10 when one feature is shuffled, on `fom-dev`.

    Permutation rather than impurity importance, which the handoff asks for and which matters
    here: impurity importance is biased towards high-cardinality features, and this feature set
    has an extinction group with ~151 levels sitting next to eighty continuous columns.

    Shuffled **within the condition bundle** rather than globally, so the permuted feature keeps
    its marginal distribution under that condition and the measurement is of the feature's
    information rather than of a distribution shift.
    """
    rng = np.random.default_rng(args.seed)
    matrices = [combiner.design_matrix(frame) for frame in frames]
    context = FomMetrics.entry_context(entries)
    baseline = _top10(combiner, frames, matrices, context)
    rows = []
    for index, name in enumerate(combiner.names):
        deltas = []
        for _ in range(args.importance_repeats):
            saved = [matrix[:, index].copy() for matrix in matrices]
            for matrix in matrices:
                rng.shuffle(matrix[:, index])
            deltas.append(_top10(combiner, frames, matrices, context) - baseline)
            for matrix, column in zip(matrices, saved):
                matrix[:, index] = column
        rows.append(dict(feature=name, baseline_top10=baseline,
                         delta_top10=float(np.mean(deltas))))
    return pd.DataFrame(rows).sort_values('delta_top10').reset_index(drop=True)


def _top10(combiner, frames, matrices, context):
    """Cross-lattice pooled top-10 on the supplied matrices, CNRS-weighted.

    Goes through `reduce_pool`/`derive_flags` rather than `evaluate` because importance needs one
    number eighty-odd times and the bootstrap, strata and loss tables are all wasted work there.
    The reduction is the same code, so the number is the same number.
    """
    reductions = []
    for frame, matrix in zip(frames, matrices):
        raw = combiner.predict_batch(matrix)
        reductions.append(FomMetrics.reduce_pool(frame, raw, pool='cross_bl'))
    per_entry = context.merge(FomMetrics._combine_reductions(reductions),
                              on=['entry_id', 'condition_bundle'], how='inner', validate='1:1')
    per_entry = FomMetrics.derive_flags(per_entry)
    return float(FomMetrics.weighted_mean(per_entry, 'top10', True))


def timing(combiner, frame, repeats=3):
    """Seconds per candidate for the model alone, and for the features it depends on.

    Two numbers, not one, because gate condition 3 is about the inner loop and the *features* are
    what spend its budget: `S06_zoo_cost.csv` prices `M_sym` at 24 `get_M20`-equivalents and the
    two over-prediction counters at 29 and 30. Reporting only the model's own cost would answer a
    question nobody asked.
    """
    assembly = np.inf
    for _ in range(repeats):
        started = time.perf_counter()
        matrix = combiner.design_matrix(frame)
        assembly = min(assembly, time.perf_counter() - started)
    inference = np.inf
    for _ in range(repeats):
        started = time.perf_counter()
        combiner.predict_batch(matrix)
        inference = min(inference, time.perf_counter() - started)
    n_candidates = int(matrix.shape[0])
    return dict(n_candidates=n_candidates, n_features=len(combiner.names),
                seconds_per_candidate_model=inference/n_candidates,
                seconds_per_candidate_design=assembly/n_candidates)


if __name__ == '__main__':
    main()
