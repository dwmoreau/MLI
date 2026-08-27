"""S13 -- scoring the per-peak assignment table, and choosing the mask threshold.

The measurement half of `run_fom_assignment.py`; kept separate because the table costs minutes to
build and is scored many times.

Two questions, and they are not the same question.

**Which estimator is right** (`--stage analytic`). Brier, expected calibration error and AUC per
form, per Bravais lattice and as an unweighted aggregate, on the whole population and on the
**well-posed** stratum -- the candidates that are the right cell in the truth's own setting, where
"the correct Miller index" names one reflection at all. Fitted on `fom-train`, reported on
`fom-dev` (PROTOCOL section 8).

**Where the mask should cut** (`--stage threshold`). `Candidates.refine_cell` admits a peak to the
final Gauss-Newton step when its probability clears `assignment_threshold`, which ships at 0.95 for
every lattice. That number was chosen against `rho`, which states 0.87 on a population whose base
rate is a few percent, so it does not transfer to a calibrated statistic and porting it would be
the mistake this step exists to avoid. This sweeps it: for each statistic and each cut, how many
peaks are admitted and what fraction of them carry their correct Miller index.

Selection is on `fom-train`. `fom-dev` is reported and not chosen on.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.scripts.run_fom_assignment import ANALYTIC_FORMS  # noqa: E402
from mlindex.scripts.run_fom_assignment import SETTING_CUTS  # noqa: E402
from mlindex.scripts.run_fom_assignment import SIGMA_MULTIPLIERS  # noqa: E402
from mlindex.scripts.run_fom_assignment import commit_hash  # noqa: E402
from mlindex.scripts.run_fom_assignment import load_peaks  # noqa: E402

# The shipped cut, in all seven per-lattice-system parameter dictionaries
# (UtilitiesOptimizer.py). The analytical CLI uses 0.90; the ML path, which is what S13 measures,
# uses this.
SHIPPED_THRESHOLD = 0.95

# The grid the mask is swept over. Wide, because a calibrated posterior and an over-confident rho
# do not live on the same part of [0, 1]: rho puts almost everything above 0.8, so a cut that
# means "most peaks" for one means "almost none" for the other.
THRESHOLD_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99,
                  0.995, 0.999, 0.9995, 0.9999, 0.99999)

# The statistics a mask could read. `dewolff` is here because it is the one genuinely different
# analytic form -- the same link on de Wolff 1961's Delta(Q) rather than Taupin's density -- and
# S13's handoff asks for all three.
MASK_FORMS = ('rho', 'posterior', 'dewolff')

STRATA = (
    ('all', lambda frame: frame),
    # Where the question is well posed: the candidate is the right cell *in the truth's own
    # setting*, so "the correct Miller index" names one reflection (R15).
    ('well_posed', lambda frame: frame.loc[frame['is_correct'] & frame['same_setting']]),
    ('well_posed_reachable', lambda frame: frame.loc[
        frame['is_correct'] & frame['same_setting'] & frame['reachable']]),
    ('alternative_setting', lambda frame: frame.loc[
        frame['is_correct'] & ~frame['same_setting']]),
    ('correct_candidate', lambda frame: frame.loc[frame['is_correct']]),
    ('wrong_candidate', lambda frame: frame.loc[~frame['is_correct']]),
    ('real_peaks', lambda frame: frame.loc[~frame['is_contaminant']]),
    ('contaminant_peaks', lambda frame: frame.loc[frame['is_contaminant']]),
    ('unreachable_peaks', lambda frame: frame.loc[~frame['reachable']]),
    )


def reliability(confidence, correct, n_bins=10):
    """Equal-count reliability table and its expected calibration error.

    Equal-count rather than equal-width bins, so a bin with three points cannot set the headline.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(confidence)
    rows, ece = [], 0.0
    for index, rows_in_bin in enumerate(np.array_split(order, n_bins)):
        if rows_in_bin.size == 0:
            continue
        mean_confidence = float(confidence[rows_in_bin].mean())
        mean_accuracy = float(correct[rows_in_bin].mean())
        ece += (rows_in_bin.size/confidence.size)*abs(mean_confidence - mean_accuracy)
        rows.append(dict(bin=index, n=int(rows_in_bin.size), confidence=mean_confidence,
                         accuracy=mean_accuracy, gap=mean_accuracy - mean_confidence))
    return rows, float(ece)


def roc_auc(score, label):
    """Rank AUC, ties averaged. Invariant under any monotone transform of the score.

    That invariance is the point: `rho`, `taupin` and the isotonic recalibration are all monotone
    functions of one `arg`, so they must return the *same* number here. Reporting it is how "these
    differ in calibration and in nothing else" stops being an assertion.
    """
    label = np.asarray(label, dtype=bool)
    if label.all() or not label.any():
        return np.nan
    from scipy.stats import rankdata

    ranks = rankdata(np.asarray(score, dtype=np.float64))
    n_positive = int(label.sum())
    n_negative = int(label.size - n_positive)
    return float((ranks[label].sum() - n_positive*(n_positive + 1)/2)/(n_positive*n_negative))


def cluster_bootstrap(values, clusters, n_bootstrap=1000, seed=12345):
    """Percentile interval for a mean, resampling **source entries** rather than rows.

    One crystal contributes twenty peaks per candidate and hundreds of candidates, all correlated;
    resampling rows would treat those as independent and shrink the interval by an order of
    magnitude (PROTOCOL section 8).
    """
    values = np.asarray(values, dtype=np.float64)
    codes, index = np.unique(np.asarray(clusters), return_inverse=True)
    order = np.argsort(index, kind='stable')
    edges = np.searchsorted(index[order], np.arange(len(codes) + 1))
    sums = np.add.reduceat(values[order], edges[:-1])
    counts = np.diff(edges)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(n_bootstrap, len(codes)))
    means = sums[draws].sum(axis=1)/np.maximum(counts[draws].sum(axis=1), 1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fit_isotonic(peaks, column='argument', increasing=False):
    """The monotone ceiling for one statistic: the best calibration any function of it can reach.

    Isotonic regression minimises the Brier score over **all** monotone transforms, so this is not
    one recalibration among many -- it is the ceiling, and it is the bar `rho` and `taupin` have to
    be read against, since both are monotone functions of the same `arg`. `increasing=False`
    because `arg` runs the other way: a large expected number of coincidences means a *less*
    trustworthy assignment.
    """
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip', increasing=increasing)
    model.fit(peaks[column].to_numpy(dtype=np.float64), peaks['label'].to_numpy(dtype=float))
    return model


def score_rows(peaks, form, n_bins=10, n_bootstrap=0, seed=12345):
    """The scalar row for one probability form on one slice of the peak table."""
    probability = peaks[form].to_numpy(dtype=np.float64)
    label = peaks['label'].to_numpy(dtype=bool)
    _, ece = reliability(probability, label, n_bins=n_bins)
    row = dict(form=form, n=int(len(peaks)), n_source_entries=int(peaks['entry_id'].nunique()),
               base_rate=float(label.mean()), mean_probability=float(probability.mean()),
               ece=float(ece),
               brier=float(np.mean((probability - label.astype(float))**2)),
               auc=roc_auc(probability, label))
    if n_bootstrap:
        low, high = cluster_bootstrap(
            (probability - label.astype(float))**2, peaks['entry_id'].to_numpy(),
            n_bootstrap, seed)
        row['brier_low'], row['brier_high'] = low, high
    return row


def setting_cut_table(peaks, cuts=SETTING_CUTS):
    """The evidence behind SETTING_TOLERANCE, rather than an assertion of it.

    Label rate against the setting residual, on correct candidates only. The cut is where the rate
    collapses; the reachable ceiling stays flat across it, which is what says the collapse is the
    basis and not the reference list.
    """
    correct = peaks.loc[peaks['is_correct']]
    rows, lower = [], 0.0
    for cut in cuts:
        band = correct.loc[(correct['setting_residual'] >= lower)
                           & (correct['setting_residual'] < cut)]
        rows.append(dict(setting_residual_low=lower, setting_residual_high=float(cut),
                         n=int(len(band)), n_candidates=int(
                             band.groupby(['entry_id', 'condition_bundle', 'bravais_lattice',
                                           'candidate_id'], sort=False).ngroups) if len(band)
                         else 0,
                         label_rate=float(band['label'].mean()) if len(band) else np.nan,
                         reachable_rate=float(band['reachable'].mean()) if len(band) else np.nan))
        lower = float(cut)
    return pd.DataFrame(rows)


def unweighted(frame, column, by='bravais_lattice'):
    """The campaign's aggregate: the unweighted mean over lattices, never over rows.

    PROTOCOL section 3 rule 6. A row-weighted aggregate is dominated by whichever lattice happened
    to generate the most candidates, which is aP by a wide margin.
    """
    return float(frame.groupby(by)[column].mean().mean())


def run_analytic(args):
    peaks = load_peaks(args.peaks_root, args.population)
    artifact_dir = Path(BASE)/args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train = peaks.loc[peaks['split'] == 'fom-train']
    dev = peaks.loc[peaks['split'] == 'fom-dev']
    if not len(train) or not len(dev):
        raise SystemExit(f'need both splits; have {peaks["split"].value_counts().to_dict()}')

    isotonic = fit_isotonic(train)
    well_posed_train = train.loc[train['is_correct'] & train['same_setting']]
    isotonic_well_posed = (fit_isotonic(well_posed_train)
                           if len(well_posed_train) > 100 else None)
    base_rate = float(train['label'].mean())

    def score_frame(frame):
        frame = frame.copy()
        frame['constant'] = base_rate
        frame['isotonic'] = isotonic.predict(frame['argument'].to_numpy(dtype=np.float64))
        if isotonic_well_posed is not None:
            frame['isotonic_well_posed'] = isotonic_well_posed.predict(
                frame['argument'].to_numpy(dtype=np.float64))
        return frame

    dev = score_frame(dev)
    forms = list(ANALYTIC_FORMS) + (['isotonic_well_posed'] if isotonic_well_posed is not None
                                    else [])

    # ---- per stratum, per lattice, and the unweighted aggregate ------------------------------
    rows = []
    for stratum, select in STRATA:
        block = select(dev)
        if not len(block):
            continue
        for form in forms:
            rows.append(dict(score_rows(block, form, n_bootstrap=args.bootstrap),
                             stratum=stratum, bravais_lattice='ALL_pooled'))
            for lattice, per_lattice in block.groupby('bravais_lattice'):
                if len(per_lattice) < args.min_rows:
                    continue
                rows.append(dict(score_rows(per_lattice, form), stratum=stratum,
                                 bravais_lattice=lattice))
    table = pd.DataFrame(rows)
    table.to_csv(artifact_dir/f'S13_assignment_forms_{args.population}.csv', index=False)

    # PROTOCOL rule 6: the headline is the unweighted mean over lattices, and the pooled row is
    # kept beside it rather than instead of it.
    aggregate = []
    for stratum in table['stratum'].unique():
        block = table.loc[(table['stratum'] == stratum)
                          & (table['bravais_lattice'] != 'ALL_pooled')]
        for form in forms:
            per_form = block.loc[block['form'] == form]
            if not len(per_form):
                continue
            aggregate.append(dict(
                stratum=stratum, form=form, n_lattices=int(per_form['bravais_lattice'].nunique()),
                brier=float(per_form['brier'].mean()), ece=float(per_form['ece'].mean()),
                auc=float(per_form['auc'].mean()),
                base_rate=float(per_form['base_rate'].mean()),
                mean_probability=float(per_form['mean_probability'].mean())))
    aggregate = pd.DataFrame(aggregate)
    aggregate.to_csv(artifact_dir/f'S13_assignment_aggregate_{args.population}.csv', index=False)

    # ---- the sigma sensitivity curve the protocol requires -----------------------------------
    sigma_rows = []
    for stratum in ('all', 'well_posed'):
        block = dict(STRATA)[stratum](dev)
        if not len(block):
            continue
        for name in ['posterior'] + [f'posterior_sigma{m:g}' for m in SIGMA_MULTIPLIERS]:
            multiplier = 1.0 if name == 'posterior' else float(name.split('sigma')[1])
            sigma_rows.append(dict(score_rows(block, name), stratum=stratum,
                                   sigma_multiplier=multiplier))
    pd.DataFrame(sigma_rows).to_csv(
        artifact_dir/f'S13_assignment_sigma_{args.population}.csv', index=False)

    setting_cut_table(dev).to_csv(
        artifact_dir/f'S13_assignment_setting_cut_{args.population}.csv', index=False)

    (artifact_dir/f'S13_assignment_provenance_{args.population}.json').write_text(json.dumps({
        'commit': commit_hash(), 'population': args.population,
        'n_peak_rows': int(len(peaks)), 'n_train_rows': int(len(train)),
        'n_dev_rows': int(len(dev)), 'train_base_rate': base_rate,
        'isotonic_well_posed_fitted': isotonic_well_posed is not None,
        }, indent=2), encoding='utf-8')

    show = aggregate.loc[aggregate['stratum'].isin(('all', 'well_posed'))]
    print(show.to_string(index=False))
    print(f'\nwrote {artifact_dir}/S13_assignment_aggregate_{args.population}.csv')


def mask_table(frame, forms=MASK_FORMS, grid=THRESHOLD_GRID):
    """Per (form, threshold): how many peaks the mask admits, and how many of them are right.

    `precision` is what `refine_cell` actually cares about -- an admitted peak carrying the wrong
    Miller index pulls the final Gauss-Newton step towards the wrong cell -- and `admitted_per
    _candidate` is the other half of the trade, because a mask that admits three peaks cannot
    determine six cell parameters however clean those three are.
    """
    label = frame['label'].to_numpy(dtype=bool)
    n_candidates = frame.groupby(
        ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id'], sort=False).ngroups
    rows = []
    for form in forms:
        probability = frame[form].to_numpy(dtype=np.float64)
        for threshold in grid:
            admitted = probability > threshold
            n_admitted = int(admitted.sum())
            rows.append(dict(
                form=form, threshold=float(threshold), n_admitted=n_admitted,
                admitted_per_candidate=n_admitted/max(n_candidates, 1),
                precision=float(label[admitted].mean()) if n_admitted else np.nan,
                recall=float(admitted[label].mean()) if label.any() else np.nan,
                n_correct_admitted=int((admitted & label).sum()),
                n_wrong_admitted=int((admitted & ~label).sum()),
                n_candidates=int(n_candidates)))
    return pd.DataFrame(rows)


def run_threshold(args):
    """Sweep the mask cut, and name the posterior threshold the arms should run at.

    **The rule was replaced once, for cause, and both are reported.** The rule stated first was
    matched count: a mask trades how many peaks reach the refinement against how many of them are
    right, so compare two statistics at the same point on that trade -- the posterior cut whose
    admitted peaks per candidate is closest to `rho` at the shipped 0.95.

    That rule assumes the two statistics lie on one frontier, and they do not. At its most
    demanding grid point the posterior still admits **more** peaks than the incumbent *and* admits
    them at higher precision, so there is no matched point and nothing to trade. The matched-count
    row is kept as the diagnostic that says so, and the choice is made by a rule that a dominated
    incumbent admits:

        **the largest admitted count whose precision is at least the incumbent's.**

    That is the right objective for `refine_cell` on its own terms -- a cell is better determined
    by more peaks, provided they are not the wrong ones -- and it reduces to the matched-count
    rule whenever a genuine trade exists. Recorded as a decision rather than moved silently
    (PROTOCOL section 7).

    Chosen on `fom-train`, in both senses PROTOCOL section 8 means: the grid is swept there and
    the winner is read there. `fom-dev` is scored at the chosen value and never searched over.
    """
    peaks = load_peaks(args.peaks_root, args.population)
    artifact_dir = Path(BASE)/args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train = peaks.loc[peaks['split'] == 'fom-train']
    dev = peaks.loc[peaks['split'] == 'fom-dev']

    rows = []
    for stratum in ('all', 'well_posed', 'correct_candidate'):
        for split_name, frame in (('fom-train', train), ('fom-dev', dev)):
            block = dict(STRATA)[stratum](frame)
            if not len(block):
                continue
            rows.append(mask_table(block).assign(stratum=stratum, split=split_name,
                                                 bravais_lattice='ALL_pooled'))
            for lattice, per_lattice in block.groupby('bravais_lattice'):
                if len(per_lattice) < args.min_rows:
                    continue
                rows.append(mask_table(per_lattice).assign(
                    stratum=stratum, split=split_name, bravais_lattice=lattice))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(artifact_dir/f'S13_mask_threshold_{args.population}.csv', index=False)

    # ---- the matched-count choice, on fom-train, on the well-posed stratum --------------------
    # Well-posed because that is where an admitted peak's correctness is a meaningful claim; the
    # mask runs on every candidate, but a threshold chosen on a population where the label is
    # mostly False by construction would be chosen against the base rate rather than the peak.
    selection = table.loc[(table['split'] == 'fom-train')
                          & (table['stratum'] == 'well_posed')
                          & (table['bravais_lattice'] == 'ALL_pooled')]
    incumbent = selection.loc[(selection['form'] == 'rho')
                              & (selection['threshold'] == SHIPPED_THRESHOLD)]
    if not len(incumbent):
        raise SystemExit('the shipped threshold is not on the grid')
    target = float(incumbent['admitted_per_candidate'].iloc[0])

    target_precision = float(incumbent['precision'].iloc[0])

    chosen = []
    for form in MASK_FORMS:
        block = selection.loc[selection['form'] == form].copy()
        matched = block.assign(
            count_gap=(block['admitted_per_candidate'] - target).abs()
            ).sort_values('count_gap').iloc[0]
        # The rule: as many peaks as possible, at no worse precision than the incumbent. If
        # nothing clears the incumbent's precision the form has no admissible cut and the
        # strictest grid point is reported, which is the honest failure.
        admissible = block.loc[block['precision'] >= target_precision]
        best = (admissible.sort_values('admitted_per_candidate').iloc[-1] if len(admissible)
                else block.sort_values('threshold').iloc[-1])
        chosen.append(dict(
            form=form, threshold=float(best['threshold']),
            admitted_per_candidate=float(best['admitted_per_candidate']),
            precision=float(best['precision']), recall=float(best['recall']),
            dominates_incumbent=bool(len(admissible)
                                     and best['admitted_per_candidate'] >= target),
            matched_count_threshold=float(matched['threshold']),
            matched_count_admitted=float(matched['admitted_per_candidate']),
            matched_count_precision=float(matched['precision']),
            target_admitted_per_candidate=target, target_precision=target_precision))
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(artifact_dir/f'S13_mask_choice_{args.population}.csv', index=False)

    print(f'incumbent: rho at {SHIPPED_THRESHOLD} admits {target:.3f} peaks per candidate '
          f'at precision {float(incumbent["precision"].iloc[0]):.4f} '
          '(fom-train, well-posed)\n')
    print(chosen.to_string(index=False))
    print(f'\nwrote {artifact_dir}/S13_mask_choice_{args.population}.csv')


# -------------------------------------------------------------------------------------------
# The refine_cell replay -- consumer 1, measured on the cell rather than on a proxy
# -------------------------------------------------------------------------------------------
# The tightness ladder the refined cell is scored on. `is_correct_known_bl_batch` at 0.01 is the
# campaign's own definition of a correct cell, so a ladder around it says "how much closer" in the
# only units this project already agrees on, instead of a bespoke distance nobody can compare with
# anything (C2-F-006's lesson about numbers that cannot be traced to an artefact).
RTOL_LADDER = (0.02, 0.01, 0.005, 0.002, 0.001)

# Coarser than the sweep the proxy uses: every point here costs a Gauss-Newton pass over the whole
# replayed pool, and the proxy's curve already says where the interesting region is.
REPLAY_GRID = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 0.9999)


def replay_refine_cell(q2_obs, xnn, hkl, best_M20, indexed, hkl_ref, q2_calculator,
                       lattice_system, rng):
    """`Candidates.refine_cell`'s arithmetic, on an arbitrary peak mask.

    Line for line what `refine_cell` does (Candidates.py): group the candidates by how many peaks
    their mask admits so each group can be batched, take one Gauss-Newton step on the admitted
    peaks only, repair, re-assign against the full reference list, and keep the refined cell
    **only where M20 improved** -- that acceptance test is part of the consumer and dropping it
    would measure a different function.

    `rng` feeds `fix_unphysical`'s repair of unphysical cells. Production draws from the search's
    own stream; a replay cannot reproduce that, so every arm here is handed a generator seeded
    identically and the difference between arms stays attributable to the mask. The zero-error
    branch is not replayed -- the benchmark does not run it.
    """
    from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
    from mlindex.utilities.FigureOfMerits import get_M20
    from mlindex.utilities.UnitCellTools import fix_unphysical
    from mlindex.utilities.numba_functions import fast_assign

    n_indexed = np.sum(indexed, axis=1)
    refined_xnn = xnn.copy()
    for n in np.unique(n_indexed):
        selected = n_indexed == n
        subsampled = np.argwhere(indexed[selected])[:, 1].reshape((int(selected.sum()), int(n)))
        hkl_subsampled = np.take_along_axis(hkl[selected], subsampled[:, :, np.newaxis], axis=1)
        q2_subsampled = np.take(q2_obs, subsampled)
        target = CandidateOptLoss(q2_subsampled, lattice_system=lattice_system)
        target.update(hkl_subsampled, refined_xnn[selected])
        refined_xnn[selected] += target.gauss_newton_step(refined_xnn[selected])
    refined_xnn = fix_unphysical(xnn=refined_xnn, rng=rng, lattice_system=lattice_system)

    q2_ref_calc = q2_calculator.get_q2(refined_xnn)
    hkl_assign = fast_assign(q2_obs, q2_ref_calc)
    refined_q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
    # get_M20 writes into q2_ref_calc (np.putmask), so it is the last thing to touch it.
    refined_M20 = get_M20(q2_obs, refined_q2_calc, q2_ref_calc)
    improved = refined_M20 > best_M20
    accepted = xnn.copy()
    accepted[improved] = refined_xnn[improved]
    kept_M20 = np.where(improved, refined_M20, best_M20)
    return accepted, kept_M20, improved


def correctness_ladder(unit_cell_true, unit_cell_pred, lattice_system):
    from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
    from mlindex.scripts.run_fom_prune_rerun import TRUTH_SLICE

    truth = np.asarray(unit_cell_true, dtype=np.float64)[TRUTH_SLICE[lattice_system]]
    return {f'correct_rtol{rtol:g}': is_correct_known_bl_batch(
        truth, unit_cell_pred, lattice_system, rtol=rtol) for rtol in RTOL_LADDER}


def run_replay(args):
    """Consumer 1: what the mask does to the cell `refine_cell` produces.

    **This is a replay, not the production state, and the difference is stated rather than
    buried.** `refine_cell` runs once per pool on the cell the search left; the capture stores the
    cell after the whole post-search chain, so what is replayed here is one further masked step
    from a candidate's final position. It measures the mask -- does admitting these peaks rather
    than those move the cell towards the truth -- and it does not predict a run. The paired real
    arms in `run_fom_assignment_arms.py` are the verdict; this is what chooses the cut they run at,
    on `fom-train`.

    Restricted to candidates on the entry's own true Bravais lattice, because only there does a
    distance to the truth exist. That restriction is the reason this is a diagnostic.
    """
    from mlindex.utilities.Q2Calculator import Q2Calculator
    from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
    from mlindex.scripts.run_fom_assignment import (CAPTURE_COLUMNS, capture_shards,
                                                    entry_table, hkl_reference, subsample)
    from mlindex.model_training import FomBenchmark as Bench
    from mlindex.utilities import FigureOfMerits as fom

    entries = entry_table(args.population)
    rng_pool = np.random.default_rng(args.seed)
    artifact_dir = Path(BASE)/args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    grid = (tuple(float(v) for v in args.replay_grid.split(','))
            if getattr(args, 'replay_grid', None) else REPLAY_GRID)
    forms = (tuple(args.replay_forms.split(','))
             if getattr(args, 'replay_forms', None) else ('rho', 'dewolff', 'posterior'))
    suffix = getattr(args, 'replay_suffix', '') or ''

    rows = []
    for shard in capture_shards(args.population):
        candidates = pd.read_parquet(shard, columns=list(CAPTURE_COLUMNS))
        candidates, _ = subsample(candidates, args.max_candidates, rng_pool)
        # Only the true lattice can hold a correct cell, so only there is "closer to the truth" a
        # measurable thing. Everything else is False by construction and would dilute the arm.
        true_lattice = entries.reset_index(drop=True).set_index(
            ['entry_id', 'condition_bundle'])['bravais_lattice_true']
        keys = list(zip(candidates['entry_id'], candidates['condition_bundle']))
        candidates = candidates.loc[
            candidates['bravais_lattice'].to_numpy() == true_lattice.loc[keys].to_numpy()]
        if not len(candidates):
            continue

        grouped = candidates.groupby(
            ['entry_id', 'condition_bundle', 'bravais_lattice', 'spacegroup'], sort=False)
        for (entry_id, bundle, bravais_lattice, spacegroup), group in grouped:
            entry = entries.loc[(entry_id, bundle)]
            lattice_system = group['lattice_system'].iloc[0]
            n_peaks = int(group['n_peaks'].iloc[0])
            q2_obs = np.asarray(entry['q2_obs'], dtype=np.float64)[:n_peaks]
            xnn = np.stack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
            split = group['split'].iloc[0]

            q2_ref_calc, _, hkl, q2_calc = Bench.assign_lines(
                q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, args.models_dir)
            reciprocal_volume = get_unit_cell_volume_from_xnn(xnn, lattice_system)
            statistics = {name: builder() for name, builder in (
                ('rho', lambda: fom.get_assignment_probability(
                    q2_obs, q2_calc, bravais_lattice, reciprocal_volume, form='rho')),
                ('dewolff', lambda: fom.get_assignment_probability_dewolff(
                    q2_obs, q2_calc, xnn, lattice_system, bravais_lattice)),
                ('posterior', lambda: fom.get_assignment_posterior(
                    q2_obs, q2_ref_calc, lattice_system)),
                ) if name in forms}
            # get_M20 mutates its reference array, so the baseline is scored on a copy and the
            # untouched q2_ref_calc stays available to the posterior above.
            from mlindex.utilities.FigureOfMerits import get_M20
            best_M20 = get_M20(q2_obs, q2_calc, q2_ref_calc.copy())
            q2_calculator = Q2Calculator(
                lattice_system=lattice_system,
                hkl=Bench.hkl_ref_for(lattice_system, bravais_lattice, spacegroup,
                                      args.models_dir),
                tensorflow=False, representation='xnn')
            hkl_ref = hkl_reference(lattice_system, bravais_lattice, args.models_dir)

            unrefined = correctness_ladder(
                entry['unit_cell_true'],
                get_unit_cell_from_xnn(xnn, partial_unit_cell=True,
                                       lattice_system=lattice_system),
                lattice_system)
            for form, probability in statistics.items():
                for threshold in grid:
                    accepted, kept_M20, improved = replay_refine_cell(
                        q2_obs, xnn, hkl, best_M20, probability > threshold, hkl_ref,
                        q2_calculator, lattice_system, np.random.default_rng(args.seed))
                    ladder = correctness_ladder(
                        entry['unit_cell_true'],
                        get_unit_cell_from_xnn(accepted, partial_unit_cell=True,
                                               lattice_system=lattice_system),
                        lattice_system)
                    rows.append(dict(
                        entry_id=entry_id, condition_bundle=bundle, split=split,
                        bravais_lattice=bravais_lattice, form=form, threshold=float(threshold),
                        n_candidates=int(len(group)),
                        mean_admitted=float(np.mean(np.sum(probability > threshold, axis=1))),
                        accepted_rate=float(improved.mean()),
                        delta_M20=float(np.mean(kept_M20 - best_M20)),
                        **{f'before_{k}': int(v.sum()) for k, v in unrefined.items()},
                        **{f'after_{k}': int(v.sum()) for k, v in ladder.items()}))
        print(f'{shard.name}: {len(rows)} arm rows', flush=True)

    replay = pd.DataFrame(rows)
    replay.to_parquet(artifact_dir/f'S13_replay_per_entry_{args.population}{suffix}.parquet', index=False)

    # Per (form, threshold): the campaign's unweighted-over-lattices aggregate, and the pooled row
    # beside it. Selection is on fom-train; fom-dev is carried so the choice can be reported
    # without having been made on it.
    summary = []
    for split_name in ('fom-train', 'fom-dev'):
        block = replay.loc[replay['split'] == split_name]
        if not len(block):
            continue
        for (form, threshold), arm in block.groupby(['form', 'threshold']):
            per_lattice = arm.groupby('bravais_lattice')[
                [c for c in arm.columns if c.startswith(('before_', 'after_'))] + ['n_candidates']
                ].sum()
            rates = {}
            for rtol in RTOL_LADDER:
                before = per_lattice[f'before_correct_rtol{rtol:g}']/per_lattice['n_candidates']
                after = per_lattice[f'after_correct_rtol{rtol:g}']/per_lattice['n_candidates']
                rates[f'before_rtol{rtol:g}'] = float(before.mean())
                rates[f'after_rtol{rtol:g}'] = float(after.mean())
                rates[f'delta_pp_rtol{rtol:g}'] = float((after - before).mean()*100)
            summary.append(dict(
                split=split_name, form=form, threshold=float(threshold),
                n_lattices=int(len(per_lattice)), n_candidates=int(arm['n_candidates'].sum()),
                mean_admitted=float(arm['mean_admitted'].mean()),
                accepted_rate=float(arm['accepted_rate'].mean()),
                delta_M20=float(arm['delta_M20'].mean()), **rates))
    summary = pd.DataFrame(summary)
    summary.to_csv(artifact_dir/f'S13_replay_{args.population}{suffix}.csv', index=False)

    show = summary.loc[summary['split'] == 'fom-train', [
        'form', 'threshold', 'mean_admitted', 'accepted_rate', 'delta_M20',
        'before_rtol0.01', 'after_rtol0.01', 'delta_pp_rtol0.01', 'delta_pp_rtol0.002']]
    print(show.to_string(index=False))
    print(f'\nwrote {artifact_dir}/S13_replay_{args.population}{suffix}.csv')


def get_unit_cell_volume_from_xnn(xnn, lattice_system):
    """V* exactly as get_M20_likelihood_from_xnn computes it, never from a stored column."""
    from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
    from mlindex.utilities.UnitCellTools import get_unit_cell_volume

    return get_unit_cell_volume(
        get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=lattice_system),
        partial_unit_cell=True, lattice_system=lattice_system)


# -------------------------------------------------------------------------------------------
# Consumer 2: the reported n_indexed column
# -------------------------------------------------------------------------------------------
def run_n_indexed(args):
    """What the swap does to `n_indexed`, and how much of a decision that column makes.

    **It makes none.** S13's handoff describes `n_indexed` as "reported to the user and available
    as a deduplication tie-break (`MPIOptimizer.DEDUP_TIEBREAK_FOMS`)". That constant is on `fom`
    only: it belongs to the multi-merit iterate retention campaign 2 dropped, and on this branch
    -- and on `main` -- `_downsample_chunk` collapses each neighbourhood to its highest-M20 member
    with no tie-break at all. So changing `n_indexed` changes a printed column and nothing else.

    That makes the interesting question a different one. `n_indexed` is the only per-candidate
    number a user sees beside M20, and they read it as "how much of the pattern this cell
    explains". So: does it separate correct candidates from incorrect ones, and does the swap make
    it better at that? Reported as an AUC per lattice, at each statistic's own chosen cut.
    """
    peaks = load_peaks(args.peaks_root, args.population)
    artifact_dir = Path(BASE)/args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cuts = {'rho': SHIPPED_THRESHOLD, 'posterior': args.posterior_threshold,
            'dewolff': SHIPPED_THRESHOLD}
    keys = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']
    per_candidate = peaks.groupby(keys, sort=False).agg(
        is_correct=('is_correct', 'first'), split=('split', 'first'),
        bravais_lattice_true=('bravais_lattice_true', 'first'),
        same_setting=('same_setting', 'first'), n_correct_peaks=('label', 'sum')).reset_index()
    for form, cut in cuts.items():
        admitted = peaks.assign(**{f'_{form}': peaks[form] > cut}).groupby(
            keys, sort=False)[f'_{form}'].sum()
        per_candidate[f'n_indexed_{form}'] = admitted.to_numpy()

    rows = []
    for split_name, block in per_candidate.groupby('split'):
        for lattice, arm in list(block.groupby('bravais_lattice')) + [('ALL_pooled', block)]:
            if len(arm) < args.min_rows:
                continue
            row = dict(split=split_name, bravais_lattice=lattice, n_candidates=int(len(arm)),
                       n_correct=int(arm['is_correct'].sum()),
                       mean_true_indexed=float(arm['n_correct_peaks'].mean()))
            for form, cut in cuts.items():
                column = arm[f'n_indexed_{form}'].to_numpy(dtype=float)
                row[f'mean_{form}'] = float(column.mean())
                row[f'bias_{form}'] = float((column - arm['n_correct_peaks']).mean())
                row[f'mae_{form}'] = float(np.abs(column - arm['n_correct_peaks']).mean())
                row[f'auc_{form}'] = roc_auc(column, arm['is_correct'].to_numpy(dtype=bool))
                row[f'threshold_{form}'] = cut
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(artifact_dir/f'S13_n_indexed_{args.population}.csv', index=False)

    dev = table.loc[(table['split'] == 'fom-dev') & (table['bravais_lattice'] != 'ALL_pooled')]
    print(f'n_indexed on fom-dev, unweighted over {dev["bravais_lattice"].nunique()} lattices')
    for form in cuts:
        print(f'  {form:<10} mean {dev[f"mean_{form}"].mean():6.2f}  '
              f'bias {dev[f"bias_{form}"].mean():+6.2f}  '
              f'MAE {dev[f"mae_{form}"].mean():5.2f}  '
              f'AUC for is_correct {dev[f"auc_{form}"].mean():.4f}')
    print(f'  truth      mean {dev["mean_true_indexed"].mean():6.2f}')
    print(f'\nwrote {artifact_dir}/S13_n_indexed_{args.population}.csv')


def run_choose(args):
    """The mask cut, chosen across both populations rather than on either alone.

    The replay's own optimum differs by population -- the general arm is a broad plateau that
    keeps climbing to 0.99999, the hard arm peaks at 0.99 and has fallen threefold by 0.9999 --
    and a cut chosen on the general arm alone would be near-optimal where the pipeline already
    works and a third as good where campaign 2's gains are supposed to come from.

    So the rule is **max-min**: normalise each population's curve by its own maximum and take the
    threshold whose *worst* fraction is largest. It needs no weighting between two populations
    that are not on a common scale, and it is the conservative direction -- it refuses a cut that
    is excellent on one arm and poor on the other.

    Read on `fom-train` in both populations. `fom-dev` is scored at the chosen value and never
    searched over.
    """
    artifact_dir = Path(BASE)/args.artifact_dir
    frames = {}
    for population in ('general', 'hard'):
        parts = [artifact_dir/f'S13_replay_{population}.csv']
        tail = artifact_dir/f'S13_replay_{population}_tail.csv'
        if tail.exists():
            parts.append(tail)
        frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        frames[population] = frame.loc[(frame['split'] == 'fom-train')
                                       & (frame['form'] == args.mask_form)]

    metric = f'delta_pp_rtol{args.mask_metric_rtol:g}'
    common = sorted(set(frames['general']['threshold']) & set(frames['hard']['threshold']))
    rows = []
    for threshold in common:
        row = {'threshold': threshold}
        for population, frame in frames.items():
            at = frame.loc[np.isclose(frame['threshold'], threshold)]
            row[f'{population}_pp'] = float(at[metric].iloc[0])
            row[f'{population}_admitted'] = float(at['mean_admitted'].iloc[0])
            row[f'{population}_frac_of_max'] = float(at[metric].iloc[0]/frame[metric].max())
        row['min_frac_of_max'] = min(row['general_frac_of_max'], row['hard_frac_of_max'])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values('threshold')
    table.to_csv(artifact_dir/f'S13_mask_choice_joint_{args.mask_form}.csv', index=False)
    chosen = table.loc[table['min_frac_of_max'].idxmax()]
    print(table.to_string(index=False))
    print(f"\nchosen {args.mask_form} threshold: {chosen['threshold']:g} "
          f"({chosen['general_pp']:+.3f} pp general, {chosen['hard_pp']:+.3f} pp hard, "
          f"worst case {chosen['min_frac_of_max']:.3f} of that population's own maximum)")
    print(f"\nwrote {artifact_dir}/S13_mask_choice_joint_{args.mask_form}.csv")


def run_figure(args):
    """The step's picture: what the peak mask is worth, against the cut it is made at.

    Two panels, one per population, on a shared story. The horizontal line is the no-mask arm --
    admit every peak -- because that is the comparison the shipped cut has to beat and does not.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'axes.axisbelow': True, 'grid.alpha': 0.25,
                         'grid.linewidth': 0.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                         'axes.linewidth': 0.7})
    artifact_dir = Path(BASE)/args.artifact_dir
    colours = {'rho': '#C1571A', 'posterior': '#0B5D91', 'dewolff': '#6B7A8F'}
    labels = {'rho': r'$\rho$  (shipped)', 'posterior': 'assignment posterior',
              'dewolff': "de Wolff $\\Delta$"}

    figure, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    for axis, population in zip(axes, ('general', 'hard')):
        parts = [artifact_dir/f'S13_replay_{population}.csv']
        tail = artifact_dir/f'S13_replay_{population}_tail.csv'
        if tail.exists():
            parts.append(tail)
        frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        frame = frame.loc[frame['split'] == 'fom-train']
        metric = f'delta_pp_rtol{args.mask_metric_rtol:g}'
        no_mask = frame.loc[np.isclose(frame['threshold'], 0.0), metric].mean()
        axis.axhline(no_mask, color='#444444', lw=0.9, ls=(0, (4, 2)), zorder=1)
        axis.annotate('no mask: admit every peak', (0.02, no_mask), xycoords=('axes fraction',
                      'data'), fontsize=6.5, color='#444444', va='bottom')
        for form, block in frame.groupby('form'):
            block = block.sort_values('threshold')
            # Plotted against 1 - threshold on a log axis: the interesting region for a
            # calibrated statistic is 0.99 to 0.99999, which a linear axis compresses to a point.
            axis.plot(1 - block['threshold'] + 1e-6, block[metric], marker='o', ms=3, lw=1.2,
                      color=colours[form], label=labels[form], zorder=3)
        shipped = frame.loc[(frame['form'] == 'rho') & np.isclose(frame['threshold'], 0.95)]
        chosen = frame.loc[(frame['form'] == 'posterior')
                           & np.isclose(frame['threshold'], args.posterior_threshold)]
        for point, colour, text in ((shipped, colours['rho'], 'shipped 0.95'),
                                    (chosen, colours['posterior'],
                                     f'chosen {args.posterior_threshold:g}')):
            if len(point):
                axis.scatter(1 - point['threshold'] + 1e-6, point[metric], s=42,
                             facecolor='white', edgecolor=colour, zorder=4, lw=1.2)
                axis.annotate(text, (1 - float(point['threshold'].iloc[0]) + 1e-6,
                                     float(point[metric].iloc[0])),
                              textcoords='offset points',
                              xytext=(6, 6) if text.startswith('shipped') else (6, -11),
                              fontsize=6.5, color=colour)
        axis.set_xscale('log')
        axis.invert_xaxis()
        axis.set_title('general population' if population == 'general' else 'hard stratum',
                       pad=6)
    axes[0].set_ylabel(f'refined cells correct at rtol {args.mask_metric_rtol:g}\n'
                       '(percentage points gained)')
    axes[0].legend(loc='center left', frameon=False)
    # One shared label: two would collide, and the axis is the same quantity in both panels.
    figure.supxlabel('mask threshold $t$, on a $1-t$ log scale -- stricter to the right',
                     fontsize=8, y=-0.04)
    figure.suptitle('The shipped peak mask is worse than not masking at all', fontsize=9.5,
                    y=1.02)
    destination = artifact_dir/'S13_refine_threshold.png'
    figure.savefig(destination)
    plt.close(figure)
    print(f'wrote {destination}')
