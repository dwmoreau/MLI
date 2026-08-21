"""S11 session 2 -- block B: the per-peak Miller-index assignment probability.

    python -m mlindex.scripts.run_fom_assignment --stage analytic --lattices mP
    python -m mlindex.scripts.run_fom_assignment --stage main     --lattices mP
    python -m mlindex.scripts.run_fom_assignment --stage cost     --lattices mP

Given a peak list and one candidate cell, block B asks for the probability that each observed peak
is assigned its correct Miller index. PLAN section 4 calls this the weakest link of the three-block
architecture and says to build a network here **only** if the analytic estimators are miscalibrated
in a way features cannot repair -- so `--stage analytic` runs first, is reported whatever the
network does, and on its own discharges S01-C.

**The two analytic estimators the handoff names are one statistic under two link functions.**
`get_M20_likelihood` computes `arg = 2*eps*n` and returns both `rho = 1/(1 + arg)` and, inside
Minfo, Taupin's `1 - exp(-arg)`. Monotone links do not reorder anything, so those two differ in
calibration and in nothing else, and the honest bar is the **isotonic recalibration of `arg`** --
the best any monotone function of it can do (DWMM, STATUS section 6). Four forms are therefore
measured: `rho`, `taupin`, `dewolff` (the same link on de Wolff 1961's Delta(Q), the only one that
is a different statistic) and `isotonic`, fitted on `fom-train` and applied unchanged to `fom-dev`.

The population is the frozen benchmark: real candidates from real indexing runs, per-peak truth
stored as `hkl_true`, and the assignment rebuilt exactly by `FomBenchmark.assign_lines`. It is
bounded by R1 -- the pool is censored at M20 >= 5 -- and that bound is stated on every number.
"""
import argparse
import json
import numpy as np
import os
import subprocess
import time

os.environ.setdefault('KERAS_BACKEND', 'torch')

from mlindex.model_training import AssignmentModel as Assign
from mlindex.model_training import FomBenchmark as Bench
from mlindex.model_training.FomMetrics import BRAVAIS_LATTICES
from mlindex.scripts.run_fom_prior import commit_hash
from mlindex.scripts.run_fom_prior import reliability
from mlindex.utilities import FigureOfMerits as fom


DEFAULT_DATASETS = os.path.join('mlindex', 'data', 'generated_datasets')
DEFAULT_MANIFEST = os.path.join('docs', 'fom', 'artifacts', 'S02_mirror_manifest.parquet')
DEFAULT_BENCHMARK = os.path.join('mlindex', 'data', 'fom_benchmark')
DEFAULT_ARTIFACTS = os.path.join('docs', 'fom', 'artifacts')
DEFAULT_MODELS = os.path.join('mlindex', 'models', 'fom_assignment')
DEFAULT_PEAKS = os.path.join('mlindex', 'data', 'fom_assignment')

# C0 is excluded everywhere: zero error means zero residual, so M20 diverges and 9.5% of its
# candidates score above 1e9 (F-054, METRICS section 9). It is a control for candidate generation,
# not a condition anything is calibrated on.
CONTROL_BUNDLES = ('error0_cont0',)

# PROTOCOL section 3 rule 4: anything that uses a sigma reports a sensitivity curve over it. The
# posterior estimates sigma in sample rather than assuming it, so this asks what a mis-estimate
# of that scale costs -- which is the honest form of the question for an estimator rather than an
# assumption.
SIGMA_MULTIPLIERS = (0.25, 0.5, 2.0, 4.0)

# The probability forms, in the order they are reported. `arg` is the shared statistic and is kept
# as a column so the recalibrations have something to fit to.
#
# `constant` is not padding. F-083 is this project's standing warning that a scoring rule can be
# dominated by the base rate: a *constant* score already reaches 0.27 on top-10 there. On a
# population whose base rate is 4% a constant predictor scores a Brier of 0.04, so any claim that
# a probability is "good" has to clear that line first, and the pooled isotonic will sit close to
# it whenever `arg` carries little information about the thing being asked.
#
# `isotonic` is the bar the network has to beat (DWMM, STATUS section 6), and the reason is exact:
# **isotonic regression achieves the lowest Brier score attainable by any monotone transform of a
# statistic.** rho, taupin and dewolff are monotone transforms of arg, so they cannot beat it, and
# neither can any relabelling of arg. A network that beats it is using information that is not in
# arg; a network that does not is supplying a link function that already exists.
ANALYTIC_FORMS = ('rho', 'taupin', 'dewolff', 'constant', 'isotonic',
                  'posterior', 'posterior_robust')

# The posterior forms answer a different question from the other three -- "which line produced this
# peak" rather than "could a random cell have come this close" -- and the difference is what makes
# them calibrated with nothing fitted. See `FigureOfMerits.get_assignment_posterior`.
POSTERIOR_FORMS = ('posterior', 'posterior_robust')

# Fitted on the well-posed subpopulation and reported only there -- see STRATA. At deployment you
# do not know whether a candidate is correct, so this is not a deployable score; it is the answer
# to A7's question ("is the analytic estimator well calibrated at all") on the population where
# "the correct Miller index" names one reflection.
WELL_POSED_FORM = 'isotonic_well_posed'
NETWORK_WELL_POSED_FORM = 'network_well_posed'

CANDIDATE_COLUMNS = (
    'candidate_id', 'entry_id', 'xnn', 'spacegroup', 'lattice_system', 'bravais_lattice',
    'is_correct', 'is_off_by_two', 'M20', 'n_peaks', 'reciprocal_volume', 'in_top_n',
    )
ENTRY_COLUMNS = (
    'entry_id', 'condition_bundle', 'split', 'q2_obs', 'hkl_true', 'bravais_lattice_true',
    'volume_true', 'n_contaminants',
    )

# Shirley (1980): M20's own reproducibility is ~10%, so a difference smaller than that is not a
# difference, whatever its p-value (F-009, PROTOCOL section 8).
REPRODUCIBILITY_FLOOR = 0.10

# How close a candidate has to sit to the truth's own *setting* for "the correct Miller index" to
# mean anything. Measured in error scales: the median over real peaks of |q2(hkl_true) - q2_obs|
# evaluated through the **candidate's** cell, so a candidate that describes the right lattice in a
# different basis fails it while a candidate refined onto the truth passes at ~1.
#
# This is not fussiness. Most of the candidates the benchmark labels `is_correct` are the right
# lattice in an alternative monoclinic setting -- the P2_1/a against P2_1/n choice, and more
# generally any a -> a + n c shear -- and their Miller indices are therefore expressed in a
# different basis from `hkl_true`. Index identity scores those 0.19 while a same-setting correct
# candidate scores 0.86, so pooling the two measures the basis convention rather than the
# assignment. The benchmark stores no candidate-to-truth transformation, hence the direct test.
#
# The cut is 1.0 by measurement, not by taste: `S11_B_setting_cut.csv` sweeps it, and the label
# rate falls from 0.86 below 1 to 0.66 between 1 and 3 and to 0.19 above 3, while the *reachable*
# ceiling stays flat at ~0.9 throughout. A candidate refined onto the truth sits at ~1 error scale
# because that is what refinement leaves; anything materially above it is a different basis.
SETTING_TOLERANCE = 1.0

# Sweep reported beside the choice, so the cut is evidenced rather than asserted (F-064's habit).
SETTING_CUTS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf)


def evaluable_bundles(root):
    return tuple(
        bundle for bundle in Bench.available_bundles(root) if bundle not in CONTROL_BUNDLES
        )


# -------------------------------------------------------------------------------------------
# The per-peak table
# -------------------------------------------------------------------------------------------
def setting_residuals(q2_obs, hkl_true, xnn, lattice_system):
    """How far each candidate is from describing these peaks at the *true* Miller indices.

    q2 for `hkl_true` computed through the **candidate's** cell, compared with the observed peaks
    and divided by the error scale the generator drew from. A candidate refined onto the truth
    comes back at ~1; the same lattice in a different setting comes back at tens to hundreds,
    because its indices label different reflections.

    Contaminant peaks carry `(0, 0, 0)` and are excluded -- they have no true reflection to place.

    Returns (n_candidates,), the median over real peaks.
    """
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info

    real = np.any(np.asarray(hkl_true) != 0, axis=1)
    if not real.any():
        return np.full(len(xnn), np.inf)
    design = Assign.canonical_hkl(hkl_true, lattice_system)
    q2_at_true = design@np.asarray(xnn).T
    params = get_peak_generation_info()['q2_error_params']
    scale = (params[0] + q2_obs*params[1])[:, np.newaxis]
    return np.median((np.abs(q2_at_true - q2_obs[:, np.newaxis])/scale)[real], axis=0)


def reachable_peaks(hkl_true, lattice_system, bravais_lattice, spacegroup,
                    models_directory=None):
    """Is each peak's true reflection even in the list the candidate assigns from?

    `assign_lines` picks the nearest line of `hkl_ref_for(..., spacegroup)`, which is the model's
    truncated reference list narrowed to one extinction group. A true reflection outside it cannot
    be recovered by any assignment rule, learned or analytic, so this is a hard ceiling on the
    label and it belongs beside every number rather than inside the residual.

    Two mechanisms put a reflection outside: the truncation to `hkl_ref_length` lines, and the
    extinction group -- a candidate that picked the wrong one has deleted whole families of
    reflections from its own vocabulary. S01's truncation audit (F-023, Q10) measured the first on
    *true* cells and found it never binds; S01-C carried the same measurement forward to real
    candidates, and this is it.

    Contaminants carry `(0, 0, 0)`, which is the reference list's own sentinel row, so they would
    match trivially; they are marked unreachable instead, which is what they are.
    """
    from mlindex.model_training.FomBenchmark import hkl_ref_for

    hkl_true = np.asarray(hkl_true, dtype=float)
    reference = hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory)
    known = set(map(
        bytes, np.ascontiguousarray(Assign.canonical_hkl(reference, lattice_system)),
        ))
    canonical = np.ascontiguousarray(Assign.canonical_hkl(hkl_true, lattice_system))
    present = np.array([row.tobytes() in known for row in canonical])
    return present & np.any(hkl_true != 0, axis=1)


def setting_cut_table(peaks, cuts=SETTING_CUTS):
    """Label rate and reachable ceiling against the setting cut, for the correct candidates."""
    import pandas as pd

    correct = peaks.loc[peaks['is_correct'] & ~peaks['is_contaminant']]
    grouped = correct.groupby(['entry_id', 'condition_bundle', 'candidate_id'], sort=False)
    per_candidate = grouped.agg(
        label_rate=('label', 'mean'), reachable=('reachable', 'mean'),
        residual=('setting_residual', 'first'),
        )
    rows, lower = [], 0.0
    for cut in cuts:
        band = per_candidate.loc[
            (per_candidate['residual'] >= lower) & (per_candidate['residual'] < cut)
            ]
        rows.append(dict(
            setting_residual_low=lower, setting_residual_high=float(cut),
            n_candidates=int(len(band)),
            label_rate=float(band['label_rate'].mean()) if len(band) else np.nan,
            reachable_ceiling=float(band['reachable'].mean()) if len(band) else np.nan,
            ))
        lower = float(cut)
    return pd.DataFrame(rows)


def collect_peaks(root, bravais_lattice, bundles, split, max_entries=None,
                  max_candidates=None, seed=12345, models_directory=None, verbose=True):
    """One row per (candidate, observed peak), with the label and the analytic probabilities.

    This is the object everything in block B is measured on, and it is built once so the analytic
    forms and the network are scored on **identical rows** -- a Brier comparison between two
    scores computed over different subsamples is not paired and does not mean anything.

    Both caps subsample; neither is silent. `n_entries_available`/`n_candidates_dropped` come back
    in the summary so a bounded number is reported as bounded (PROTOCOL section 10).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    lattice_system = Assign.lattice_system_of(bravais_lattice)
    hkl_ref = Assign.hkl_reference(bravais_lattice, models_directory)

    entries = Bench.load_entries(root)
    entries = entries.loc[
        entries['condition_bundle'].isin(bundles) & (entries['split'] == split),
        list(ENTRY_COLUMNS),
        ].reset_index(drop=True)
    n_available = len(entries)
    if max_entries is not None and len(entries) > max_entries:
        # Subsample by *source entry*, never by row, so one crystal's conditions travel together
        # and the cluster bootstrap downstream stays valid (PROTOCOL section 8).
        sources = np.sort(entries['entry_id'].unique())
        keep = rng.choice(sources, size=min(max_entries, len(sources)), replace=False)
        entries = entries.loc[entries['entry_id'].isin(set(keep))].reset_index(drop=True)

    candidates = Bench.load_candidates(
        root, bravais_lattices=[bravais_lattice], bundles=list(bundles),
        columns=list(CANDIDATE_COLUMNS),
        )
    keys = set(zip(entries['entry_id'], entries['condition_bundle']))
    candidates = candidates.loc[
        [key in keys for key in zip(candidates['entry_id'], candidates['condition_bundle'])]
        ].reset_index(drop=True)
    n_candidates_available = len(candidates)
    if max_candidates is not None:
        chosen = []
        for _, group in candidates.groupby(['entry_id', 'condition_bundle'], sort=False):
            if len(group) > max_candidates:
                # Keep every correct candidate -- they are 0.5% of the pool and the whole
                # positive class -- and subsample the rest.
                correct = group.loc[group['is_correct'].astype(bool)]
                rest = group.drop(index=correct.index)
                take = max(max_candidates - len(correct), 0)
                if len(rest) > take:
                    rest = rest.iloc[np.sort(rng.choice(len(rest), size=take, replace=False))]
                group = pd.concat([correct, rest])
            chosen.append(group)
        candidates = pd.concat(chosen, ignore_index=True)
    n_candidates_dropped = n_candidates_available - len(candidates)

    entry_lookup = {
        (row.entry_id, row.condition_bundle): row
        for row in entries.itertuples(index=False)
        }
    blocks, candidate_blocks, n_groups = [], [], 0
    grouped = candidates.groupby(['entry_id', 'condition_bundle', 'spacegroup'], sort=False)
    for (entry_id, bundle, spacegroup), group in grouped:
        entry = entry_lookup[(entry_id, bundle)]
        n_peaks = int(group['n_peaks'].iloc[0])
        q2_obs = np.asarray(entry.q2_obs, dtype=np.float64)[:n_peaks]
        hkl_true = np.asarray(entry.hkl_true, dtype=np.float64).reshape(-1, 3)[:n_peaks]
        xnn = np.stack([np.asarray(value, dtype=np.float64) for value in group['xnn']])
        reciprocal_volume = np.asarray(group['reciprocal_volume'], dtype=np.float64)

        q2_ref_calc, _, hkl_assigned, q2_calc = Bench.assign_lines(
            q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, models_directory,
            )
        argument = fom.get_assignment_argument(
            q2_obs, q2_calc, bravais_lattice, reciprocal_volume,
            )
        dewolff = fom.get_assignment_probability_dewolff(
            q2_obs, q2_calc, xnn, lattice_system, bravais_lattice,
            )
        # Computed here rather than in a second pass because it needs `q2_ref_calc`, the full set
        # of calculated lines, which is far too large to store -- (n_candidates, up to 1000) per
        # group -- and is already in hand from `assign_lines`.
        posterior = fom.get_assignment_posterior(q2_obs, q2_ref_calc, lattice_system)
        posterior_robust = fom.get_assignment_posterior(
            q2_obs, q2_ref_calc, lattice_system, robust=True,
            )
        sigma_curve = {
            f'posterior_sigma{multiplier:g}': fom.get_assignment_posterior(
                q2_obs, q2_ref_calc, lattice_system, sigma_multiplier=multiplier,
                )
            for multiplier in SIGMA_MULTIPLIERS
            }
        label = Assign.assignment_labels(hkl_assigned, hkl_true[np.newaxis], lattice_system)
        assign_class = Assign.hkl_class_index(hkl_assigned, hkl_ref, lattice_system)
        true_class = Assign.hkl_class_index(hkl_true, hkl_ref, lattice_system)
        setting_residual = setting_residuals(q2_obs, hkl_true, xnn, lattice_system)
        reachable = reachable_peaks(
            hkl_true, lattice_system, bravais_lattice, spacegroup, models_directory,
            )

        n_candidates, n_peaks = q2_calc.shape
        peak_index = np.tile(np.arange(n_peaks), n_candidates)
        blocks.append(pd.DataFrame(dict(
            entry_id=np.repeat(group['entry_id'].to_numpy(), n_peaks),
            condition_bundle=bundle,
            candidate_id=np.repeat(group['candidate_id'].to_numpy(), n_peaks),
            bravais_lattice=bravais_lattice,
            bravais_lattice_true=entry.bravais_lattice_true,
            is_correct=np.repeat(np.asarray(group['is_correct'], dtype=bool), n_peaks),
            is_off_by_two=np.repeat(
                np.asarray(group['is_off_by_two']).astype(bool), n_peaks,
                ),
            setting_residual=np.repeat(setting_residual, n_peaks).astype(np.float32),
            same_setting=np.repeat(setting_residual < SETTING_TOLERANCE, n_peaks),
            in_top_n=np.repeat(np.asarray(group['in_top_n']).astype(bool), n_peaks),
            peak_index=peak_index,
            q2_obs=np.tile(q2_obs, n_candidates).astype(np.float32),
            q2_calc=q2_calc.reshape(-1).astype(np.float32),
            # float64 for the shared statistic: rho and taupin are derived from it, and in
            # float32 the two links round differently enough to move their AUCs apart in the
            # fourth decimal -- which would look like the ordering difference this whole section
            # says does not exist.
            argument=argument.reshape(-1),
            dewolff=dewolff.reshape(-1),
            posterior=posterior.reshape(-1),
            posterior_robust=posterior_robust.reshape(-1),
            **{name: values.reshape(-1) for name, values in sigma_curve.items()},
            assign_class=assign_class.reshape(-1).astype(np.int32),
            true_class=np.tile(true_class, n_candidates).astype(np.int32),
            is_contaminant=np.tile(
                np.all(hkl_true == 0, axis=1), n_candidates,
                ),
            reachable=np.tile(reachable, n_candidates),
            label=label.reshape(-1),
            )))
        # The candidate-level twin of the same rows. The network takes (q2_obs, xnn) per
        # candidate, not per peak, and it has to be scored on exactly the rows the analytic forms
        # were scored on -- so the two tables are written together and keyed the same way rather
        # than rebuilt separately and hoped to line up.
        candidate_blocks.append(pd.DataFrame(dict(
            entry_id=group['entry_id'].to_numpy(),
            condition_bundle=bundle,
            candidate_id=group['candidate_id'].to_numpy(),
            spacegroup=spacegroup,
            is_correct=np.asarray(group['is_correct'], dtype=bool),
            setting_residual=setting_residual,
            same_setting=setting_residual < SETTING_TOLERANCE,
            q2_obs=[q2_obs.copy() for _ in range(len(group))],
            xnn=[row.copy() for row in xnn],
            assign_class=[row.copy() for row in assign_class],
            true_class=[true_class.copy() for _ in range(len(group))],
            )))
        n_groups += 1
        if verbose and n_groups % 500 == 0:
            print(f'  {bravais_lattice} {split}: {n_groups} groups, '
                  f'{sum(len(block) for block in blocks)} peak rows', flush=True)

    peaks = pd.concat(blocks, ignore_index=True)
    candidate_frame = pd.concat(candidate_blocks, ignore_index=True)
    peaks['rho'] = 1.0/(1.0 + peaks['argument'])
    peaks['taupin'] = np.exp(-peaks['argument'])
    summary = dict(
        bravais_lattice=bravais_lattice, split=split,
        n_entries_available=int(n_available), n_entries=int(len(entries)),
        n_candidates_available=int(n_candidates_available),
        n_candidates=int(len(candidates)), n_candidates_dropped=int(n_candidates_dropped),
        n_peak_rows=int(len(peaks)), n_source_entries=int(peaks['entry_id'].nunique()),
        )
    return peaks, candidate_frame, summary


# -------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------
def fit_isotonic(peaks, column='argument', increasing=False):
    """The monotone ceiling for one statistic: the best calibration any function of it can reach.

    Isotonic regression minimises the Brier score over **all** monotone transforms, so this is not
    one recalibration among many -- it is the ceiling. Fitted on `fom-train` and applied unchanged
    to `fom-dev` (PROTOCOL section 8).

    Two things are fitted with it, and the symmetry is the point. Applied to `arg` it says what
    the analytic estimator is worth once its link function is chosen optimally; applied to the
    network's own output it says the same for the network. Comparing a calibrated statistic with
    an uncalibrated one would measure the link, which is exactly the confusion this session's
    first finding is about. `increasing=False` for `arg`, which runs the other way -- a large
    expected number of coincidences means a *less* trustworthy assignment.
    """
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds='clip', increasing=increasing,
        )
    model.fit(peaks[column].to_numpy(dtype=np.float64), peaks['label'].to_numpy(dtype=float))
    return model


def score_frame(peaks, isotonic, base_rate, isotonic_well_posed=None):
    """Attach a probability column per form.

    Every fitted object here comes from `fom-train` and is applied unchanged to `fom-dev`
    (PROTOCOL section 8). `base_rate` is the training population's, never the evaluation split's,
    for the same reason block A takes its base-rate entropy from the training marginal.
    """
    peaks = peaks.copy()
    peaks['constant'] = float(base_rate)
    peaks['isotonic'] = isotonic.predict(peaks['argument'].to_numpy(dtype=np.float64))
    if isotonic_well_posed is not None:
        peaks[WELL_POSED_FORM] = isotonic_well_posed.predict(
            peaks['argument'].to_numpy(dtype=np.float64)
            )
    return peaks


def roc_auc(score, label):
    """Rank AUC, ties averaged. Invariant under any monotone transform of the score.

    That invariance is the point: rho, taupin, dewolff and the two recalibrations are all monotone
    functions of the same `arg`, so they must return the *same* number here. Reporting it is how
    the claim "these differ in calibration and in nothing else" stops being an assertion.
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
    magnitude. `FomMetrics` does the same thing for its entry-level metrics (METRICS section 8).
    """
    values = np.asarray(values, dtype=np.float64)
    codes, index = np.unique(np.asarray(clusters), return_inverse=True)
    order = np.argsort(index, kind='stable')
    sorted_values = values[order]
    edges = np.searchsorted(index[order], np.arange(len(codes) + 1))
    sums = np.add.reduceat(sorted_values, edges[:-1])
    counts = np.diff(edges)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(n_bootstrap, len(codes)))
    means = sums[draws].sum(axis=1)/np.maximum(counts[draws].sum(axis=1), 1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_rows(peaks, form, n_bins=10, n_bootstrap=0, seed=12345):
    """The scalar row for one probability form on one slice of the peak table."""
    probability = peaks[form].to_numpy(dtype=np.float64)
    label = peaks['label'].to_numpy(dtype=bool)
    table, ece = reliability(probability, label, n_bins=n_bins)
    brier = float(np.mean((probability - label.astype(float))**2))
    from mlindex.model_training.FomMetrics import average_precision

    row = dict(
        form=form, n=int(len(peaks)), n_source_entries=int(peaks['entry_id'].nunique()),
        base_rate=float(label.mean()), mean_probability=float(probability.mean()),
        ece=float(ece), brier=brier, auc=roc_auc(probability, label),
        average_precision=float(average_precision(probability, label)),
        )
    if n_bootstrap:
        squared = (probability - label.astype(float))**2
        low, high = cluster_bootstrap(
            squared, peaks['entry_id'].to_numpy(), n_bootstrap, seed,
            )
        row['brier_low'], row['brier_high'] = low, high
    return row, table


def modelled_count(peaks, form):
    """S01-C item 4: sum(p) as an estimator of how many peaks are correctly indexed.

    Reported per candidate, because that is where it is used -- `refine_cell` masks on rho > 0.95
    and the assignment threshold counts indexed lines, so a bias in the sum is a bias in what the
    optimiser fits to.
    """
    grouped = peaks.groupby(['entry_id', 'condition_bundle', 'candidate_id'], sort=False)
    modelled = grouped[form].sum()
    observed = grouped['label'].sum()
    return dict(
        form=form,
        n_candidates=int(len(modelled)),
        mean_modelled=float(modelled.mean()),
        mean_observed=float(observed.mean()),
        bias=float((modelled - observed).mean()),
        mean_abs_error=float((modelled - observed).abs().mean()),
        spread=float((modelled - observed).std()),
        )


def paired_delta(peaks, form_a, form_b, n_bootstrap=1000, seed=12345):
    """Paired Brier difference between two forms on the same peaks, clustered on source entry."""
    label = peaks['label'].to_numpy(dtype=float)
    squared_a = (peaks[form_a].to_numpy(dtype=np.float64) - label)**2
    squared_b = (peaks[form_b].to_numpy(dtype=np.float64) - label)**2
    difference = squared_a - squared_b
    low, high = cluster_bootstrap(difference, peaks['entry_id'].to_numpy(), n_bootstrap, seed)
    mean_b = float(squared_b.mean())
    return dict(
        form=form_a, against=form_b, delta_brier=float(difference.mean()),
        delta_low=low, delta_high=high,
        relative=float(difference.mean()/mean_b) if mean_b else np.nan,
        beats_floor=bool(abs(difference.mean()/mean_b) > REPRODUCIBILITY_FLOOR)
        if mean_b else False,
        )


# -------------------------------------------------------------------------------------------
# Stages
# -------------------------------------------------------------------------------------------
STRATA = (
    ('all', lambda frame: frame),
    # The stratum where the question is well posed: the candidate is the right cell *in the
    # truth's own setting*, so "the correct Miller index" names one reflection. This is S01-C's
    # population and the one A7 is about.
    ('well_posed', lambda frame: frame.loc[frame['is_correct'] & frame['same_setting']]),
    ('well_posed_real_peaks', lambda frame: frame.loc[
        frame['is_correct'] & frame['same_setting'] & ~frame['is_contaminant']]),
    # The peaks whose true reflection is in the candidate's own vocabulary at all. Anything
    # outside it is unassignable by construction, so this is where a probability model is being
    # asked a question it could in principle answer.
    ('well_posed_reachable', lambda frame: frame.loc[
        frame['is_correct'] & frame['same_setting'] & frame['reachable']]),
    ('unreachable_peaks', lambda frame: frame.loc[~frame['reachable']]),
    ('alternative_setting', lambda frame: frame.loc[
        frame['is_correct'] & ~frame['same_setting']]),
    ('correct_candidate', lambda frame: frame.loc[frame['is_correct']]),
    ('wrong_candidate', lambda frame: frame.loc[~frame['is_correct']]),
    ('same_lattice', lambda frame: frame.loc[
        frame['bravais_lattice'] == frame['bravais_lattice_true']]),
    ('contaminant_peaks', lambda frame: frame.loc[frame['is_contaminant']]),
    ('real_peaks', lambda frame: frame.loc[~frame['is_contaminant']]),
    ('low_q2_half', lambda frame: frame.loc[frame['peak_index'] < 10]),
    ('high_q2_half', lambda frame: frame.loc[frame['peak_index'] >= 10]),
    )


# -------------------------------------------------------------------------------------------
# What block C actually consumes
# -------------------------------------------------------------------------------------------
def candidate_summaries(peaks, form):
    """Per-candidate summaries of a per-peak probability, and the label they will be judged on.

    Block C is handed a candidate, not a peak, so the number that decides which per-peak form to
    ship is not its per-peak calibration -- it is whether a *summary* of it separates a correct
    candidate from a wrong one. Two summaries, both of which the existing pipeline already has
    analogues of:

      - `n_modelled = sum(P)`, the expected number of correctly indexed peaks. This is what
        `refine_cell`'s rho > 0.95 mask and the assignment threshold are counting today.
      - `mean_log = mean(log P)`, a per-peak log-likelihood. Taupin's merit is this shape, and it
        weights a single confidently-wrong peak far more heavily than the count does.
    """
    probability = np.clip(peaks[form].to_numpy(dtype=np.float64), 1e-12, 1.0)
    frame = peaks[['entry_id', 'condition_bundle', 'candidate_id', 'is_correct']].copy()
    frame['probability'] = probability
    frame['log_probability'] = np.log(probability)
    grouped = frame.groupby(['entry_id', 'condition_bundle', 'candidate_id'], sort=False)
    summary = grouped.agg(
        n_modelled=('probability', 'sum'), mean_log=('log_probability', 'mean'),
        is_correct=('is_correct', 'first'),
        ).reset_index()
    return summary


def summary_scores(summary, form, summary_name, n_bootstrap=0, seed=12345):
    """Discrimination of one summary, pooled and per entry.

    Pooled AUC says whether the summary separates correct candidates from wrong ones at all.
    `top1_rate` is the number that matches what `run.py` does: within each entry's own pool, is
    the highest-scoring candidate a correct one. An entry with no correct candidate is excluded
    rather than counted as a loss -- that is a generation failure and belongs to S14, not here
    (METRICS section 3).
    """
    labels = summary['is_correct'].to_numpy(dtype=bool)
    values = summary[summary_name].to_numpy(dtype=np.float64)
    row = dict(form=form, summary=summary_name, n_candidates=int(len(summary)),
               n_correct=int(labels.sum()), auc=roc_auc(values, labels))

    reachable, wins = 0, 0
    for _, group in summary.groupby(['entry_id', 'condition_bundle'], sort=False):
        correct = group['is_correct'].to_numpy(dtype=bool)
        if not correct.any():
            continue
        reachable += 1
        scores = group[summary_name].to_numpy(dtype=np.float64)
        best = np.flatnonzero(scores == scores.max())
        # A tie is scored as the fraction of the tied set that is correct, so a form that cannot
        # separate two candidates is not credited with picking the right one by file order --
        # F-083's warning, which found a *constant* score scoring 0.27 on top-10 for exactly that
        # reason.
        wins += float(correct[best].mean())
    row['n_entries_reachable'] = reachable
    row['top1_rate'] = wins/reachable if reachable else np.nan
    return row

def run_analytic(args):
    import pandas as pd

    started = time.time()
    bundles = tuple(args.bundles) if args.bundles else evaluable_bundles(args.benchmark_dir)
    os.makedirs(args.peaks_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    baseline_rows, reliability_rows, modelled_rows, paired_rows, summaries = [], [], [], [], []
    setting_rows, summary_rows = [], []
    for lattice in args.lattices:
        train, train_candidates, summary_train = collect_peaks(
            args.benchmark_dir, lattice, bundles, 'fom-train', args.train_entries,
            args.max_candidates, args.seed, args.models_directory, args.verbose,
            )
        dev, dev_candidates, summary_dev = collect_peaks(
            args.benchmark_dir, lattice, bundles, 'fom-dev', args.dev_entries,
            args.max_candidates, args.seed + 7, args.models_directory, args.verbose,
            )
        summaries.extend([summary_train, summary_dev])
        print(f'{lattice}: fom-train {summary_train["n_peak_rows"]} peak rows over '
              f'{summary_train["n_source_entries"]} entries; fom-dev '
              f'{summary_dev["n_peak_rows"]} over {summary_dev["n_source_entries"]}. '
              f'Dropped {summary_dev["n_candidates_dropped"]} dev candidates to the per-entry cap.',
              flush=True)

        isotonic = fit_isotonic(train)
        sigma_forms = [f'posterior_sigma{multiplier:g}' for multiplier in SIGMA_MULTIPLIERS]
        well_posed_train = train.loc[train['is_correct'] & train['same_setting']]
        isotonic_well_posed = (
            fit_isotonic(well_posed_train) if len(well_posed_train) > 100 else None
            )
        base_rate = float(train['label'].mean())
        train = score_frame(train, isotonic, base_rate, isotonic_well_posed)
        dev = score_frame(dev, isotonic, base_rate, isotonic_well_posed)
        print(f'{lattice}: fom-train base rate {base_rate:.4f}; well-posed rows '
              f'{len(well_posed_train)} at base rate '
              f'{well_posed_train["label"].mean() if len(well_posed_train) else float("nan"):.4f}',
              flush=True)
        train.to_parquet(os.path.join(args.peaks_dir, f'peaks_{lattice}_fom-train.parquet'))
        dev.to_parquet(os.path.join(args.peaks_dir, f'peaks_{lattice}_fom-dev.parquet'))
        train_candidates.to_parquet(
            os.path.join(args.peaks_dir, f'candidates_{lattice}_fom-train.parquet'),
            )
        dev_candidates.to_parquet(
            os.path.join(args.peaks_dir, f'candidates_{lattice}_fom-dev.parquet'),
            )

        setting_rows.append(setting_cut_table(dev).assign(bravais_lattice=lattice))

        # The number that decides which form block C is given: not per-peak calibration, but
        # whether a *summary* of the per-peak probabilities separates a correct candidate from a
        # wrong one.
        well_posed_dev_frame = dev.loc[dev['is_correct'] & dev['same_setting']]
        for form in list(ANALYTIC_FORMS) + sigma_forms:
            for stratum, frame in (('all', dev), ('well_posed', well_posed_dev_frame)):
                if not len(frame):
                    continue
                summary = candidate_summaries(frame, form)
                for summary_name in ('n_modelled', 'mean_log'):
                    summary_rows.append(dict(
                        bravais_lattice=lattice, stratum=stratum,
                        **summary_scores(summary, form, summary_name),
                        ))
        for stratum, select in STRATA:
            slice_ = select(dev)
            if not len(slice_):
                continue
            forms = list(ANALYTIC_FORMS) + sigma_forms
            if stratum.startswith('well_posed') and WELL_POSED_FORM in dev.columns:
                forms.append(WELL_POSED_FORM)
            for form in forms:
                row, table = score_rows(
                    slice_, form, n_bootstrap=args.n_bootstrap if stratum == 'all' else 0,
                    seed=args.seed,
                    )
                baseline_rows.append(dict(
                    bravais_lattice=lattice, stratum=stratum, **row,
                    ))
                if stratum == 'all':
                    reliability_rows.extend(
                        dict(arm='analytic', bravais_lattice=lattice, target=form, **entry)
                        for entry in table
                        )
        for bundle in bundles:
            slice_ = dev.loc[dev['condition_bundle'] == bundle]
            if not len(slice_):
                continue
            for form in ANALYTIC_FORMS:
                row, _ = score_rows(slice_, form)
                baseline_rows.append(dict(
                    bravais_lattice=lattice, stratum=f'bundle:{bundle}', **row,
                    ))
        for form in ANALYTIC_FORMS:
            modelled_rows.append(dict(bravais_lattice=lattice, **modelled_count(dev, form)))
        for form in ('rho', 'taupin', 'dewolff', 'constant'):
            paired_rows.append(dict(
                bravais_lattice=lattice, stratum='all',
                **paired_delta(dev, form, 'isotonic', args.n_bootstrap or 1000, args.seed),
                ))
        well_posed_dev = dev.loc[dev['is_correct'] & dev['same_setting']]
        if len(well_posed_dev) and WELL_POSED_FORM in dev.columns:
            for form in ('rho', 'taupin', 'dewolff', 'constant', 'posterior',
                         'posterior_robust'):
                paired_rows.append(dict(
                    bravais_lattice=lattice, stratum='well_posed',
                    **paired_delta(
                        well_posed_dev, form, WELL_POSED_FORM,
                        args.n_bootstrap or 1000, args.seed,
                        ),
                    ))

    baselines = pd.DataFrame(baseline_rows)
    if setting_rows:
        write(pd.concat(setting_rows, ignore_index=True), args, 'setting_cut')
    if summary_rows:
        write(pd.DataFrame(summary_rows), args, 'candidate_summary')
    write(baselines, args, 'analytic_baselines')
    write(pd.DataFrame(reliability_rows), args, 'reliability')
    write(pd.DataFrame(modelled_rows), args, 'modelled_count')
    write(pd.DataFrame(paired_rows), args, 'analytic_paired')
    write_meta(args, 'analytic', dict(
        bundles=list(bundles), lattices=list(args.lattices), summaries=summaries,
        wall_clock_seconds=round(time.time() - started, 1),
        ))
    print(baselines.loc[baselines['stratum'] == 'all'].to_string(index=False))
    return baselines


# -------------------------------------------------------------------------------------------
# The network
# -------------------------------------------------------------------------------------------
def split_pool(frame, fraction, seed):
    """Hold out whole source structures, never rows (PROTOCOL section 3 rule 5)."""
    identifiers = np.sort(frame['identifier'].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(identifiers)
    n_held = max(1, int(round(fraction*len(identifiers))))
    held = set(identifiers[:n_held])
    return (frame.loc[~frame['identifier'].isin(held)].reset_index(drop=True),
            frame.loc[frame['identifier'].isin(held)].reset_index(drop=True))


def check_composition(names, expected, epoch):
    """Block B's `check_balanced`.

    F-121 cost three thirty-epoch runs because a sampler was fed the wrong array and the training
    set decayed to one lattice from epoch 2, while the loss curve looked plausible throughout. The
    lesson recorded there is to assert the *batch*, every epoch, rather than to trust the loss. The
    thing that can decay here is the mixture of candidate kinds -- if the perturbation ladder or
    the benchmark draw silently empties, the model trains on true cells alone and every probability
    it emits is meaningless on a wrong one.
    """
    counts = {name: int(np.sum(names == name)) for name in np.unique(names)}
    for name, wanted in expected.items():
        if counts.get(name, 0) != wanted:
            raise RuntimeError(
                f'epoch {epoch}: batch holds {counts} against the intended {expected}. '
                f'The candidate mixture has come apart.'
                )


def training_batch(pool, rng, lattice_system, hkl_ref, benchmark, args, n_structures):
    """One epoch's batch: a fresh condition draw, fresh wrong cells, and the benchmark's own.

    Three sources, and the mixture is asserted rather than assumed:

      - `true`, the correct cell, which is the only source of a high per-peak label rate;
      - the `PERTURBATION_LADDER` rungs, wrong cells at known xnn distances from it;
      - `benchmark`, real candidates the indexer actually produced on `fom-train` entries, which
        are the only ones carrying the *refined* wrong-cell geometry R10 says nothing else has.

    Every structure is drawn fresh each epoch -- new noise, new contaminants -- so oversampling a
    thin pool gives new realisations rather than the same pattern repeated (block A's argument for
    the same construction).
    """
    rows = pool.iloc[rng.choice(len(pool), size=n_structures, replace=len(pool) < n_structures)]
    rows = rows.reset_index(drop=True)
    q2, hkl, _, _ = Assign.draw_peak_lists_with_hkl(rows, rng, lattice_system)
    classes = Assign.hkl_class_index(hkl, hkl_ref, lattice_system)
    xnn_true = Assign.partial_xnn(np.stack(rows['xnn_full'].to_numpy()), lattice_system)

    perturbed, rungs = Assign.perturbed_candidates(xnn_true, lattice_system, rng)
    repeat = len(Assign.PERTURBATION_LADDER)
    q2_all = [q2, np.repeat(q2, repeat, axis=0)]
    xnn_all = [xnn_true, perturbed]
    class_all = [classes, np.repeat(classes, repeat, axis=0)]
    names = [np.full(len(rows), 'true'), rungs]

    if benchmark is not None and args.benchmark_rows:
        take = benchmark.iloc[rng.choice(len(benchmark), size=args.benchmark_rows, replace=False)]
        q2_all.append(np.stack(take['q2_obs'].to_numpy()))
        xnn_all.append(np.stack(take['xnn'].to_numpy()))
        class_all.append(np.stack(take['true_class'].to_numpy()))
        names.append(np.full(len(take), 'benchmark'))

    return (np.concatenate(q2_all), np.concatenate(xnn_all),
            np.concatenate(class_all), np.concatenate(names))


def expected_composition(n_structures, args):
    expected = {'true': n_structures}
    for name, _ in Assign.PERTURBATION_LADDER:
        expected[name] = n_structures
    if args.benchmark_rows:
        expected['benchmark'] = args.benchmark_rows
    return expected


def check_loss_is_possible(model, q2, xnn, classes, reported, n_classes):
    """`fit`'s number against the model's own predictions, before anything is concluded from it.

    Twice in session 1 a run reported a loss that no predictor could have produced -- 0.32 on a
    128-class problem whose chance level is 4.85 -- and both times the numbers looked like a
    result for long enough to draw conclusions from (F-118, F-121). The check is cheap: evaluate
    the same rows through the compiled graph, and recompute the cross entropy in numpy from
    `predict`. All three must agree, and none may sit below what the predictions imply.
    """
    evaluated = model.calibration_model.evaluate(
        (model.scale_peaks(q2), np.asarray(xnn, dtype=np.float32)), classes,
        batch_size=Assign.predict_batch_size(classes.shape[1], n_classes), verbose=0,
        )
    evaluated = float(evaluated[0] if isinstance(evaluated, (list, tuple)) else evaluated)
    softmax = model.predict_softmax(q2, xnn)
    by_hand = float(np.mean(-np.log(np.maximum(
        np.take_along_axis(softmax, classes[:, :, np.newaxis], axis=2)[:, :, 0], 1e-30,
        ))))
    chance = float(np.log(n_classes))
    print(f'  loss check: fit {reported:.4f} | evaluate {evaluated:.4f} | by hand {by_hand:.4f} '
          f'| chance {chance:.4f}', flush=True)
    if not np.isclose(evaluated, by_hand, rtol=1e-3, atol=1e-3):
        raise RuntimeError(
            f'evaluate reports {evaluated:.4f} while the model own predictions imply '
            f'{by_hand:.4f}. The compiled loss is not the loss it appears to be (F-118).'
            )
    return dict(fit_loss=float(reported), evaluate_loss=evaluated, by_hand_loss=by_hand,
                chance_loss=chance)


def fit_arm(args, lattice):
    import pandas as pd

    lattice_system = Assign.lattice_system_of(lattice)
    hkl_ref = Assign.hkl_reference(lattice, args.models_directory)
    pool = Assign.load_assignment_frame(
        args.datasets_dir, args.manifest, lattice, limit=args.limit_structures,
        seed=args.seed,
        )
    fit_pool, held_pool = split_pool(pool, args.test_fraction, args.seed)
    print(f'{lattice}: {len(fit_pool)} fit structures, {len(held_pool)} held out, '
          f'{len(hkl_ref)} classes', flush=True)

    benchmark = None
    path = os.path.join(args.peaks_dir, f'candidates_{lattice}_fom-train.parquet')
    if args.benchmark_rows and os.path.exists(path):
        benchmark = pd.read_parquet(path)
        print(f'{lattice}: {len(benchmark)} real fom-train candidates available to mix in',
              flush=True)

    rng = np.random.default_rng(args.seed)
    # The scale is a property of the peak lists, so it is taken once from a draw of the fit pool
    # and then frozen -- it is baked into the pairwise-difference layer.
    scale_q2, _, _, _ = Assign.draw_peak_lists_with_hkl(
        fit_pool.iloc[:min(len(fit_pool), 2000)], rng, lattice_system,
        )
    model = Assign.AssignmentModel(
        lattice, dict(calibration_params=dict(
            layers=args.layers, learning_rate=args.learning_rate,
            batch_size=args.batch_size, epsilon_pds=args.epsilon_pds,
            )),
        os.path.join(args.models_dir, args.arm), float(scale_q2.std()), seed=args.seed,
        models_directory=args.models_directory,
        ).build()

    expected = expected_composition(args.per_epoch, args)
    history = []
    for epoch in range(args.epochs):
        q2, xnn, classes, names = training_batch(
            fit_pool, rng, lattice_system, hkl_ref, benchmark, args, args.per_epoch,
            )
        check_composition(names, expected, epoch + 1)
        record = model.calibration_model.fit(
            (model.scale_peaks(q2), np.asarray(xnn, dtype=np.float32)), classes,
            epochs=1, verbose=0, batch_size=model.model_params['calibration_params']['batch_size'],
            )
        entry = {key: float(value[-1]) for key, value in record.history.items()}
        entry.update(epoch=epoch + 1, n_rows=len(q2),
                     composition=' '.join(f'{name}:{int(np.sum(names == name))}'
                                          for name in sorted(set(names))))
        history.append(entry)
        print(f'  {lattice} epoch {epoch + 1}/{args.epochs} loss {entry["loss"]:.4f} '
              f'accuracy {entry.get("accuracy", float("nan")):.4f} | {entry["composition"]}',
              flush=True)
        if epoch == 0:
            checks = check_loss_is_possible(
                model, q2[:256], xnn[:256], classes[:256], entry['loss'], len(hkl_ref),
                )
            history[-1].update(checks)

    model.save_assignment()
    return model, pd.DataFrame(history), held_pool


def evaluate_arm(args, lattice, model, split='fom-dev'):
    """Score the network on exactly the rows the analytic forms were scored on."""
    import pandas as pd

    peaks = pd.read_parquet(os.path.join(args.peaks_dir, f'peaks_{lattice}_{split}.parquet'))
    candidates = pd.read_parquet(
        os.path.join(args.peaks_dir, f'candidates_{lattice}_{split}.parquet'),
        )
    q2 = np.stack(candidates['q2_obs'].to_numpy())
    xnn = np.stack(candidates['xnn'].to_numpy())
    assign_class = np.stack(candidates['assign_class'].to_numpy())
    started = time.time()
    at_assignment, at_argmax, argmax = model.assignment_probability(q2, xnn, assign_class)
    seconds = time.time() - started
    print(f'{lattice}: scored {len(candidates)} candidates in {seconds:.1f} s', flush=True)

    key = ['entry_id', 'condition_bundle', 'candidate_id']
    n_peaks = assign_class.shape[1]
    scored = pd.DataFrame({
        'entry_id': np.repeat(candidates['entry_id'].to_numpy(), n_peaks),
        'condition_bundle': np.repeat(candidates['condition_bundle'].to_numpy(), n_peaks),
        'candidate_id': np.repeat(candidates['candidate_id'].to_numpy(), n_peaks),
        'peak_index': np.tile(np.arange(n_peaks), len(candidates)),
        'network': at_assignment.reshape(-1),
        'network_argmax': at_argmax.reshape(-1),
        'network_agrees': (argmax == assign_class).reshape(-1),
        })
    merged = peaks.merge(scored, on=key + ['peak_index'], how='left', validate='1:1')
    assert not merged['network'].isna().any(), 'a peak row lost its network score in the join'
    return merged, seconds

def run_main(args):
    """Fit one model per lattice and score it against the analytic forms on the same peaks."""
    import pandas as pd

    started = time.time()
    os.makedirs(args.artifact_dir, exist_ok=True)
    rows, reliability_rows, paired_rows, histories, timings = [], [], [], [], []
    for lattice in args.lattices:
        model, history, _ = fit_arm(args, lattice)
        scored, seconds = evaluate_arm(args, lattice, model, 'fom-dev')

        # The network's raw softmax mass is no more calibrated than `arg` is, so it gets the same
        # treatment: an isotonic fitted on `fom-train` and applied unchanged here. Only then is the
        # comparison between the two about information rather than about link functions.
        scored_train, _ = evaluate_arm(args, lattice, model, 'fom-train')
        network_isotonic = fit_isotonic(scored_train, 'network', increasing=True)
        scored['network_calibrated'] = network_isotonic.predict(
            scored['network'].to_numpy(dtype=np.float64),
            )
        well_posed_train = scored_train.loc[
            scored_train['is_correct'] & scored_train['same_setting']
            ]
        if len(well_posed_train) > 100:
            network_well_posed = fit_isotonic(well_posed_train, 'network', increasing=True)
            scored[NETWORK_WELL_POSED_FORM] = network_well_posed.predict(
                scored['network'].to_numpy(dtype=np.float64),
                )
        histories.append(history.assign(bravais_lattice=lattice))
        timings.append(dict(bravais_lattice=lattice, seconds=seconds,
                            n_candidates=int(len(scored)/scored['peak_index'].max()
                                             if scored['peak_index'].max() else 1)))

        forms = list(ANALYTIC_FORMS) + ['network', 'network_argmax', 'network_calibrated']
        for stratum, select in STRATA:
            slice_ = select(scored)
            if not len(slice_):
                continue
            local = list(forms)
            if stratum.startswith('well_posed'):
                local.extend(
                    name for name in (WELL_POSED_FORM, NETWORK_WELL_POSED_FORM)
                    if name in scored.columns
                    )
            for form in local:
                row, table = score_rows(
                    slice_, form, n_bootstrap=args.n_bootstrap if stratum == 'all' else 0,
                    seed=args.seed,
                    )
                rows.append(dict(bravais_lattice=lattice, stratum=stratum, **row))
                if stratum in ('all', 'well_posed_reachable'):
                    reliability_rows.extend(
                        dict(arm=args.arm, bravais_lattice=lattice, stratum=stratum,
                             target=form, **entry)
                        for entry in table
                        )

        # The gate: the network against every analytic form, paired on the same peaks, on the
        # population that carries the number (benchmark fom-dev, pooled) and on the one where the
        # question is well posed.
        for form in ('network', 'network_calibrated'):
            for against in ('rho', 'taupin', 'dewolff', 'constant', 'isotonic'):
                paired_rows.append(dict(
                    bravais_lattice=lattice, stratum='all',
                    **paired_delta(
                        scored, form, against, args.n_bootstrap or 1000, args.seed,
                        ),
                    ))
        well_posed = scored.loc[scored['is_correct'] & scored['same_setting']]
        if len(well_posed):
            targets = ['rho', 'taupin']
            if WELL_POSED_FORM in scored.columns:
                targets.append(WELL_POSED_FORM)
            sources = ['network']
            if NETWORK_WELL_POSED_FORM in scored.columns:
                sources.append(NETWORK_WELL_POSED_FORM)
            for form in sources:
                for against in targets:
                    if form == against:
                        continue
                    paired_rows.append(dict(
                        bravais_lattice=lattice, stratum='well_posed',
                        **paired_delta(
                            well_posed, form, against, args.n_bootstrap or 1000, args.seed,
                            ),
                        ))

    write(pd.DataFrame(rows), args, 'network_table')
    write(pd.DataFrame(reliability_rows), args, 'network_reliability')
    write(pd.DataFrame(paired_rows), args, 'network_paired')
    write(pd.concat(histories, ignore_index=True), args, 'history')
    write_meta(args, 'main', dict(
        lattices=list(args.lattices), arm=args.arm, epochs=args.epochs,
        per_epoch=args.per_epoch, benchmark_rows=args.benchmark_rows,
        limit_structures=args.limit_structures, layers=args.layers,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        epsilon_pds=args.epsilon_pds, inference=timings,
        wall_clock_seconds=round(time.time() - started, 1),
        production_configuration=dict(
            note='development scale on the laptop. A production run trains every lattice on the '
                 'full pool for many more epochs, on NERSC.',
            limit_structures=None, per_epoch=20000, epochs=60,
            ),
        ))
    table = pd.DataFrame(rows)
    print(table.loc[table['stratum'].isin(('all', 'well_posed_reachable'))].to_string(index=False))
    return table


def run_cost(args):
    """What each per-peak probability costs, in `get_M20` equivalents. S14 needs the price.

    Same protocol as `run_fom_cv_analysis.py`: real candidate pools so the reference-line lengths
    are production's, best of N with the first pass discarded so a cold JIT is not being priced,
    and `get_M20` timed in the same loop on the same cases as the baseline.

    `assign_lines` is timed as its own row because every form here needs it and none of them can
    avoid it -- including `get_M20`. Reading the analytic rows as a *marginal* cost over an
    assignment the pipeline already has is the right reading for the inner loop; reading the
    network row that way is not, because it re-renders the whole pairwise-difference tensor.
    """
    import pandas as pd

    bundle = (args.bundles or ['error1_cont0'])[0]
    lattice = args.lattices[0]
    lattice_system = Assign.lattice_system_of(lattice)
    candidates = Bench.load_candidates(
        args.benchmark_dir, bravais_lattices=[lattice], bundles=[bundle],
        columns=list(CANDIDATE_COLUMNS),
        )
    entries = Bench.load_entries(args.benchmark_dir)
    entries = entries.loc[entries['condition_bundle'] == bundle]
    peaks = entries.set_index('entry_id')['q2_obs']

    cases = []
    for entry_id in pd.unique(candidates['entry_id'])[:args.cost_entries]:
        group = candidates.loc[candidates['entry_id'] == entry_id]
        for (spacegroup, n_peaks), chunk in group.groupby(['spacegroup', 'n_peaks'], sort=False):
            q2_obs = np.asarray(peaks.loc[entry_id], dtype=np.float64)[:int(n_peaks)]
            xnn = np.vstack([np.asarray(value, dtype=np.float64) for value in chunk['xnn']])
            reciprocal_volume = np.asarray(chunk['reciprocal_volume'], dtype=np.float64)
            q2_ref_calc, assign, _, q2_calc = Bench.assign_lines(
                q2_obs, xnn, lattice_system, lattice, spacegroup, args.models_directory,
                )
            cases.append((q2_obs, xnn, reciprocal_volume, q2_calc, q2_ref_calc, spacegroup,
                          assign))
    n_candidates = sum(case[1].shape[0] for case in cases)
    print(f'cost: {n_candidates} candidates over {len(cases)} groups', flush=True)

    model = Assign.AssignmentModel.load_assignment(
        os.path.join(args.models_dir, args.arm, lattice), seed=args.seed,
        models_directory=args.models_directory,
        )
    isotonic = None
    train_path = os.path.join(args.peaks_dir, f'peaks_{lattice}_fom-train.parquet')
    if os.path.exists(train_path):
        isotonic = fit_isotonic(pd.read_parquet(train_path, columns=['argument', 'label']))

    calls = {
        'get_M20': lambda case: fom.get_M20(case[0], case[3], case[4].copy()),
        'assign_lines (shared)': lambda case: Bench.assign_lines(
            case[0], case[1], lattice_system, lattice, case[5], args.models_directory,
            ),
        'rho / taupin': lambda case: fom.get_assignment_probability(
            case[0], case[3], lattice, case[2],
            ),
        'dewolff': lambda case: fom.get_assignment_probability_dewolff(
            case[0], case[3], case[1], lattice_system, lattice,
            ),
        }
    if isotonic is not None:
        calls['isotonic'] = lambda case: isotonic.predict(
            fom.get_assignment_argument(case[0], case[3], lattice, case[2]).reshape(-1),
            )
    calls['network'] = lambda case: model.assignment_probability(
        np.repeat(case[0][np.newaxis], len(case[1]), axis=0), case[1], case[6],
        )

    rows, baseline = [], None
    for name, call in calls.items():
        best = np.inf
        for _ in range(args.cost_repeats + 1):
            start = time.perf_counter()
            for case in cases:
                call(case)
            best = min(best, time.perf_counter() - start)
        per_candidate = best/max(n_candidates, 1)
        if name == 'get_M20':
            baseline = per_candidate
        rows.append(dict(merit=name, seconds_per_candidate=per_candidate))
    table = pd.DataFrame(rows)
    table['cost_vs_M20'] = table['seconds_per_candidate']/baseline
    table['n_candidates_timed'] = n_candidates
    write(table, args, 'cost')
    write_meta(args, 'cost', dict(
        bundle=bundle, lattice=lattice, n_candidates_timed=int(n_candidates),
        note='first pass discarded so the numba JIT is warm; assign_lines is the shared '
             'prerequisite of every row including get_M20',
        ))
    print(table.to_string(index=False))
    return table


def run_block_c(args):
    """The session's actual question: which per-peak form should block C be given?

    Per-peak calibration does not answer it. Block C is handed a candidate and asks "is this cell
    right", so a per-peak form earns its place only if a *summary* of it adds discrimination on top
    of what the combiner already has. Two baselines, and the second is the one that matters:

      - **M20 alone**, the merit the pipeline ranks on today;
      - **M20 + Minfo**, which is what the benchmark actually stores -- and Minfo is built from the
        same `arg` as rho, so "add rho" and "add Minfo" are close to the same experiment. Measuring
        against M20 alone flatters any form built on that statistic.

    A logistic on standardised features, fitted on `fom-train` and reported on `fom-dev`, per
    lattice. Deliberately a linear model rather than a tree: this is a question about whether the
    information is *present*, not about how well it can be exploited, and S08 already owns the
    second question.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    key = ['entry_id', 'condition_bundle', 'candidate_id']
    feature_sets = {
        'M20': ['m'],
        'M20 + rho': ['m', 'rho_n', 'rho_l'],
        'M20 + posterior': ['m', 'post_n', 'post_l'],
        'M20 + rho + posterior': ['m', 'rho_n', 'rho_l', 'post_n', 'post_l'],
        'M20 + Minfo (stored)': ['m', 'mi'],
        'M20 + Minfo + posterior': ['m', 'mi', 'post_n', 'post_l'],
        }

    def summarise(peaks):
        frame = peaks.copy()
        for column in ('rho', 'posterior'):
            frame[f'log_{column}'] = np.log(np.clip(frame[column], 1e-12, 1.0))
        return frame.groupby(key, sort=False).agg(
            rho_n=('rho', 'sum'), post_n=('posterior', 'sum'),
            rho_l=('log_rho', 'mean'), post_l=('log_posterior', 'mean'),
            is_correct=('is_correct', 'first'),
            ).reset_index()

    rows = []
    for lattice in args.lattices:
        paths = {
            split: os.path.join(args.peaks_dir, f'peaks_{lattice}_{split}.parquet')
            for split in ('fom-train', 'fom-dev')
            }
        if not all(os.path.exists(path) for path in paths.values()):
            print(f'{lattice}: no peak table, run --stage analytic first', flush=True)
            continue
        frames = {split: summarise(pd.read_parquet(path)) for split, path in paths.items()}
        merits = Bench.load_candidates(
            args.benchmark_dir, bravais_lattices=[lattice],
            columns=['candidate_id', 'entry_id', 'M20', 'Minfo'],
            )[key + ['M20', 'Minfo']]
        for split, frame in frames.items():
            frame = frame.merge(merits, on=key, how='left', validate='1:1')
            frame['m'] = np.log1p(frame['M20'].fillna(0.0).to_numpy())
            frame['mi'] = frame['Minfo'].fillna(0.0).to_numpy()
            frames[split] = frame
        train, dev = frames['fom-train'], frames['fom-dev']
        if train['is_correct'].nunique() < 2 or dev['is_correct'].nunique() < 2:
            print(f'{lattice}: one class only, skipped', flush=True)
            continue

        row = dict(bravais_lattice=lattice, n_dev=int(len(dev)),
                   n_correct=int(dev['is_correct'].sum()))
        labels = dev['is_correct'].to_numpy(dtype=bool)
        for name, columns in feature_sets.items():
            scaler = StandardScaler().fit(train[columns])
            model = LogisticRegression(max_iter=2000).fit(
                scaler.transform(train[columns]), train['is_correct'].astype(int),
                )
            row[name] = roc_auc(model.predict_proba(scaler.transform(dev[columns]))[:, 1], labels)
        rows.append(row)

    table = pd.DataFrame(rows)
    for name in ('M20 + rho', 'M20 + posterior', 'M20 + rho + posterior'):
        table[f'delta_vs_M20: {name}'] = 100*(table[name] - table['M20'])
    table['delta_vs_stored: + posterior'] = 100*(
        table['M20 + Minfo + posterior'] - table['M20 + Minfo (stored)']
        )
    write(table, args, 'block_c_features')
    write_meta(args, 'block_c', dict(
        lattices=list(args.lattices), feature_sets={k: v for k, v in feature_sets.items()},
        model='logistic on standardised features, fitted on fom-train, reported on fom-dev',
        ))
    print(table.round(3).to_string(index=False))
    return table


def write(frame, args, name):
    path = os.path.join(args.artifact_dir, f'{args.tag}_{name}.csv')
    frame.to_csv(path, index=False, encoding='utf-8')
    print(f'wrote {path}', flush=True)


def write_meta(args, stage, extra):
    meta = dict(
        commit=commit_hash(), seed=args.seed, stage=stage, tag=args.tag,
        benchmark=args.benchmark_dir, broadening_tag=Assign.BROADENING_TAG,
        scale='development -- laptop, subsampled. NOT a production result.',
        bounds=[
            'R1: the pool is censored at M20 >= 5, so every calibration curve here describes the '
            'surviving candidates only and says nothing about a candidate the prune removed.',
            'R10: no pre-refinement candidate is stored, so "wrong candidate" always means a '
            'Gauss-Newton refined wrong candidate, not an arbitrary one.',
            'R11: the grid has no different error *law*, so robustness to one is untested rather '
            'than passed.',
            'R12: the whole benchmark is one instrument at one broadening tag.',
            ],
        **extra,
        )
    path = os.path.join(args.artifact_dir, f'{args.tag}_{stage}_meta.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print(f'wrote {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='S11 block B: per-peak Miller-index assignment probability.'
        )
    parser.add_argument('--stage', default='analytic', choices=('analytic', 'main', 'block_c', 'cost'))
    parser.add_argument('--lattices', nargs='+', default=list(BRAVAIS_LATTICES))
    parser.add_argument('--bundles', nargs='+', default=None,
                        help='condition bundles; default is every bundle except the control')
    parser.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK)
    parser.add_argument('--datasets-dir', default=DEFAULT_DATASETS)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACTS)
    parser.add_argument('--models-dir', default=DEFAULT_MODELS)
    parser.add_argument('--peaks-dir', default=DEFAULT_PEAKS)
    parser.add_argument('--models-directory', default=None,
                        help='override the mlindex models tree hkl_ref is read from')
    parser.add_argument('--tag', default='S11_B')
    parser.add_argument('--train-entries', type=int, default=250)
    parser.add_argument('--dev-entries', type=int, default=250)
    parser.add_argument('--max-candidates', type=int, default=40,
                        help='per (entry, bundle); every correct candidate is kept regardless')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--arm', default='main')
    parser.add_argument('--limit-structures', type=int, default=20000,
                        help='source structures per lattice; <=0 for the whole pool')
    parser.add_argument('--per-epoch', type=int, default=1500,
                        help='structures drawn per epoch; each contributes one true cell and one '
                             'candidate per perturbation rung')
    parser.add_argument('--benchmark-rows', type=int, default=3000,
                        help='real fom-train candidates mixed into each epoch')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epsilon-pds', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.15)
    parser.add_argument('--cost-entries', type=int, default=40)
    parser.add_argument('--cost-repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    args = parser.parse_args()
    if args.limit_structures is not None and args.limit_structures <= 0:
        args.limit_structures = None

    if args.stage == 'analytic':
        run_analytic(args)
    elif args.stage == 'main':
        run_main(args)
    elif args.stage == 'block_c':
        run_block_c(args)
    elif args.stage == 'cost':
        run_cost(args)
    else:
        raise SystemExit(f'stage {args.stage} is not implemented yet')


if __name__ == '__main__':
    main()
