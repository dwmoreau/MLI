"""S11 session 1 -- block A: fit and evaluate the volume + Bravais prior network.

    python -m mlindex.scripts.run_fom_prior --stage main
    python -m mlindex.scripts.run_fom_prior --stage heads     # A5: conditional / marginal / joint
    python -m mlindex.scripts.run_fom_prior --stage grid      # the n_volumes sweep

Training data is `mlindex/data/generated_datasets/`, filtered against the frozen split manifest so
no `fom-dev` or `fom-test` structure is ever seen. **Evaluation is on the benchmark's own peak
lists** (`mlindex/data/fom_benchmark/entries.parquet`, `split == 'fom-dev'`), not on a re-draw of
the training conditions -- that is the population block C will consume, and evaluating on anything
else would measure the draw rather than the model.

Every number here is calibration-first, because that is the gate (S12/S13): expected calibration
accuracy on the volume branch and on the lattice, predictive-interval coverage for the volume, and
information gain in bits over the base rate for the lattice heads. Accuracy is reported and is
explicitly not the criterion -- a 46%-accurate calibrated prior is useful and a confident wrong one
is harmful.

The scale of these runs is a development scale, on the laptop, by decision. The production
configuration is named in the metadata of every artefact so no number here is later read as one.
"""
import argparse
import json
import numpy as np
import os
import subprocess
import time

os.environ.setdefault('KERAS_BACKEND', 'torch')

from mlindex.model_training import PriorNetwork as Prior


DEFAULT_DATASETS = os.path.join('mlindex', 'data', 'generated_datasets')
DEFAULT_MANIFEST = os.path.join('docs', 'fom', 'artifacts', 'S02_mirror_manifest.parquet')
DEFAULT_BENCHMARK = os.path.join('mlindex', 'data', 'fom_benchmark')
DEFAULT_ARTIFACTS = os.path.join('docs', 'fom', 'artifacts')
DEFAULT_MODELS = os.path.join('mlindex', 'models', 'fom_prior')

# Nominal coverage levels for the volume interval check. S12's gate is "within +/-10% of nominal".
COVERAGE_LEVELS = (0.5, 0.8, 0.95)


def commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL
            ).strip()
    except Exception:
        return 'unknown'


# -----------------------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------------------
def reliability(confidence, correct, n_bins=10):
    """Equal-count reliability table and its expected calibration error.

    Equal-count rather than equal-width bins, as `FomMetrics.reliability` uses, so a bin with three
    points cannot set the headline.
    """
    order = np.argsort(confidence)
    bins = np.array_split(order, n_bins)
    rows, ece = [], 0.0
    for index, rows_in_bin in enumerate(bins):
        if rows_in_bin.size == 0:
            continue
        mean_confidence = float(confidence[rows_in_bin].mean())
        mean_accuracy = float(correct[rows_in_bin].mean())
        weight = rows_in_bin.size/confidence.size
        ece += weight*abs(mean_confidence - mean_accuracy)
        rows.append(dict(bin=index, n=int(rows_in_bin.size), confidence=mean_confidence,
                         accuracy=mean_accuracy, gap=mean_accuracy - mean_confidence))
    return rows, float(ece)


def information_gain_bits(log_probability, labels, base_rate):
    """Bits the model gains over the base rate, per entry.

    `H(base rate) - cross entropy(model)`, both in bits. This is a lower bound on the mutual
    information between the peak list and the label, and it is exactly the quantity S13 asks to be
    reported -- "how much BL information is in a realistic peak list" -- in the units Taupin's
    bit-valued merit uses. It is negative for a model that is worse than knowing nothing, which is
    the honest outcome for a confidently wrong prior and the reason it is reported this way rather
    than as an accuracy.

    The base rate is the *training population's* marginal, never the evaluation split's, so the
    baseline cannot borrow information from the data the model is being scored on.
    """
    base_rate = np.asarray(base_rate, dtype=float)
    base_rate = base_rate/base_rate.sum()
    entropy = float(-(base_rate*np.log2(np.maximum(base_rate, 1e-300))).sum())
    cross_entropy = float(-(log_probability[np.arange(labels.size), labels]/np.log(2)).mean())
    return entropy - cross_entropy, entropy, cross_entropy


# -----------------------------------------------------------------------------------------
# Volume
# -----------------------------------------------------------------------------------------
def branch_volumes(model):
    """The direct-space volume each branch stands for.

    `ExtractionLayer.get_branch_labels` matches on
    `scale = (1/V / q2_obs_scale**2)**(2/3) / volume_normalization`, so inverting that gives the
    volume a branch is centred on and turns the branch softmax into a distribution over volume.
    """
    volumes = np.asarray(model.extraction_layer.volumes).ravel().astype(float)
    reciprocal = (volumes*model.extraction_layer.volume_normalization)**1.5*model.q2_obs_scale**2
    return 1.0/reciprocal


def volume_metrics(probability, volumes, volume_true, levels=COVERAGE_LEVELS):
    """Interval coverage, CRPS and NLL for the predictive distribution over volume.

    The distribution is discrete over branches, so CRPS is computed from the step CDF directly and
    the intervals are the narrowest contiguous branch ranges reaching each nominal level. The
    branches are geometrically spaced, so everything is done in log volume -- a fixed ratio, not a
    fixed difference, is what "close" means here, which is the same convention `get_branch_labels`
    matches in.
    """
    order = np.argsort(volumes)
    volumes = volumes[order]
    probability = probability[:, order]
    log_volumes = np.log(volumes)
    log_true = np.log(np.asarray(volume_true, dtype=float))

    probability = np.asarray(probability, dtype=np.float32)
    cdf = np.cumsum(probability, axis=1)
    indicator = (log_volumes[np.newaxis] >= log_true[:, np.newaxis]).astype(float)
    widths = np.diff(log_volumes, prepend=log_volumes[0])
    crps = float(((cdf - indicator)**2*widths[np.newaxis]).sum(axis=1).mean())

    nearest = np.abs(log_volumes[np.newaxis] - log_true[:, np.newaxis]).argmin(axis=1)
    nll = float(-np.log(np.maximum(probability[np.arange(log_true.size), nearest], 1e-300)).mean())

    coverage = {}
    ranked = np.argsort(-probability, axis=1).astype(np.int32)
    ordered_mass = np.take_along_axis(probability, ranked, axis=1).cumsum(axis=1)
    for level in levels:
        # Highest-density set: add branches by probability until the level is reached, then ask
        # whether the truth's branch is in it. This is the honest reading of a predictive interval
        # for a multimodal distribution, and multimodality is expected here (a pattern consistent
        # with V and with 2V is the case S12 says must not be averaged away).
        included = ordered_mass - np.take_along_axis(probability, ranked, axis=1) < level
        # Scatter the "is in the set" flags from rank order back to branch order, then read off the
        # truth's branch. Same answer as testing membership row by row, without the python loop.
        membership = np.zeros_like(included)
        np.put_along_axis(membership, ranked, included, axis=1)
        coverage[level] = float(membership[np.arange(log_true.size), nearest].mean())
    return dict(crps=crps, nll=nll, coverage=coverage,
                median_abs_log_ratio=float(np.median(np.abs(
                    (probability*log_volumes[np.newaxis]).sum(axis=1) - log_true
                    ))))


# -----------------------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------------------
def split_pool(pool, fraction, seed):
    """Split the training pool by source structure, never by row.

    A structure appears many times over training under different drawn conditions, so splitting by
    row would put the same crystal on both sides and make the held-out number meaningless.
    """
    rng = np.random.default_rng(seed)
    identifiers = np.array(sorted(set(pool['identifier'])))
    rng.shuffle(identifiers)
    n_selected = int(round(fraction*identifiers.size))
    selected = set(identifiers[:n_selected])
    mask = pool['identifier'].isin(selected).to_numpy()
    return pool.loc[~mask].reset_index(drop=True), pool.loc[mask].reset_index(drop=True)


def load_evaluation(benchmark_dir, split, bundles=None, limit=None, seed=12345):
    """The benchmark's own peak lists and truth, for one split."""
    import pandas as pd
    frame = pd.read_parquet(
        os.path.join(benchmark_dir, 'entries.parquet'),
        columns=['entry_id', 'split', 'condition_bundle', 'q2_obs', 'volume_true',
                 'bravais_lattice_true'],
        )
    frame = frame.loc[frame['split'] == split]
    keep = bundles if bundles else [name['name'] for name in Prior.CONDITION_BUNDLES]
    frame = frame.loc[frame['condition_bundle'].isin(keep)].reset_index(drop=True)
    if limit is not None and len(frame) > limit:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(frame), size=limit, replace=False))
        frame = frame.iloc[rows].reset_index(drop=True)
    for name, values in Prior.target_codes(frame['bravais_lattice_true'].to_numpy()).items():
        frame[f'target_{name}'] = values
    return frame


# -----------------------------------------------------------------------------------------
# Fit and evaluate
# -----------------------------------------------------------------------------------------
def fit_arm(pool_fit, model_params, args, tag):
    """Fit one arm. A fresh condition draw per epoch, which is what makes balancing honest."""
    rng = np.random.default_rng(args.seed)
    grid_q2, _, _ = Prior.draw_peak_lists(pool_fit, rng)
    grid_frame = pool_fit.assign(q2_window=list(grid_q2))

    model = Prior.PriorNetwork(
        data_params={}, model_params=model_params,
        save_to=os.path.join(args.models_dir, tag), seed=args.seed,
        )
    model.build_model(data=grid_frame)

    codes = pool_fit['target_bravais'].to_numpy()
    history = []
    for epoch in range(model.model_params['epochs']):
        rows = Prior.balanced_indices(
            codes, len(Prior.BRAVAIS_LATTICES), rng, args.per_class,
            )
        batch = pool_fit.iloc[rows].reset_index(drop=True)
        q2, _, _ = Prior.draw_peak_lists(batch, rng)
        branch = model.get_branch_labels(batch)
        joint = model.model_params['loss_mode'] == 'joint'
        width = model.model_params.get('volume_target_width', 0.0)
        targets = {'volume_branch': (
            Prior.soft_branch_targets(branch, model.model_params['n_volumes'], width)
            if width > 0 else branch
            )}
        for name in Prior.TARGETS:
            codes = batch[f'target_{name}'].to_numpy()
            targets[name] = Prior.joint_targets(codes, branch) if joint else codes
        record = model.model.fit(
            model.scale_peaks(q2), targets, epochs=1, verbose=0,
            batch_size=model.model_params['batch_size'],
            )
        history.append({key: float(value[-1]) for key, value in record.history.items()})
        if args.verbose:
            print(f'  {tag} epoch {epoch + 1}/{model.model_params["epochs"]} '
                  f'loss {history[-1]["loss"]:.4f}', flush=True)
    return model, history


def evaluate_arm(model, evaluation, base_rates):
    """Score the model on unseen structures. Accuracy first; nothing is fitted here."""
    evaluation_q2 = np.stack(evaluation['q2_obs'].to_numpy()).astype(float)
    raw = predict_in_batches(model, evaluation_q2)

    results, tables = {}, {'reliability': [], 'per_lattice': []}

    # The class heads already emit log probabilities -- the marginalisation happens inside the
    # graph -- so they are read straight out. The branch head emits logits, so it is normalised
    # once here and nowhere else.
    branch_log = np.asarray(raw['volume_branch'], dtype=float)
    branch_log = branch_log - np.log(np.exp(branch_log).sum(axis=1, keepdims=True))
    results['volume'] = volume_metrics(
        np.exp(branch_log), branch_volumes(model), evaluation['volume_true'].to_numpy(),
        )
    branch_labels = model.get_branch_labels(evaluation)
    results['volume']['branch_accuracy'] = float(
        (branch_log.argmax(axis=1) == branch_labels).mean()
        )
    results['volume']['branch_within_1'] = float(
        (np.abs(branch_log.argmax(axis=1) - branch_labels) <= 1).mean()
        )

    joint_mode = model.model_params.get('loss_mode') == 'joint'
    for target in Prior.TARGETS:
        log_probability = np.asarray(raw[target], dtype=float)
        if joint_mode:
            # (n, n_volumes, n_classes) -> the marginal, so every number stays comparable with
            # the marginally-trained arm. The joint itself is reported separately.
            log_probability = _logsumexp(
                branch_log[:, :, np.newaxis] + log_probability, axis=1,
                )
        labels = evaluation[f'target_{target}'].to_numpy()
        probability = np.exp(log_probability)
        predicted = probability.argmax(axis=1)
        confidence = probability.max(axis=1)
        correct = (predicted == labels).astype(float)

        rows, ece = reliability(confidence, correct)
        for row in rows:
            tables['reliability'].append(dict(target=target, **row))
        gain, entropy, cross_entropy = information_gain_bits(
            log_probability, labels, base_rates[target],
            )
        top3 = float(np.mean([
            labels[row] in np.argsort(-probability[row])[:3] for row in range(labels.size)
            ])) if probability.shape[1] > 3 else float('nan')
        results[target] = dict(
            accuracy=float(correct.mean()), top3=top3, ece=ece,
            information_gain_bits=gain, base_rate_entropy_bits=entropy,
            cross_entropy_bits=cross_entropy,
            brier=float(np.mean(np.sum(
                (probability - np.eye(probability.shape[1])[labels])**2, axis=1,
                ))),
            )

    lattice_log = np.asarray(raw['bravais'], dtype=float)
    if joint_mode:
        lattice_log = _logsumexp(branch_log[:, :, np.newaxis] + lattice_log, axis=1)
    labels = evaluation['target_bravais'].to_numpy()
    for index, code in enumerate(Prior.BRAVAIS_LATTICES):
        rows = np.flatnonzero(labels == index)
        if rows.size == 0:
            continue
        gain, _, _ = information_gain_bits(
            lattice_log[rows], labels[rows], base_rates['bravais'],
            )
        tables['per_lattice'].append(dict(
            bravais_lattice=code, n=int(rows.size),
            accuracy=float((lattice_log[rows].argmax(axis=1) == index).mean()),
            information_gain_bits=gain,
            mean_probability_on_truth=float(np.exp(lattice_log[rows, index]).mean()),
            ))
    return results, tables


def _logsumexp(values, axis):
    peak = values.max(axis=axis, keepdims=True)
    return (peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True))).squeeze(axis)


def grid_diagnostics(model):
    """Whether the extraction grid's own safeguards are binding, which for a global grid they do.

    `ExtractionLayer` clips sigma into [2, 6] x the filter spacing, and stores the *clipped* value
    as `sigma_init`, so the two cannot be compared to detect a clip. The ratio can: a sigma sitting
    exactly on 2 x spacing is one the floor is holding up, which means the filter grid is too coarse
    to resolve the peaks it is rendering and the model is reading a smeared pattern.

    This is the one thing a single grid spanning all fourteen lattices makes worse than a
    per-split-group one -- the pooled q2 range is wider, so the same filter count buys coarser
    spacing -- and it is why `--stage grid` sweeps the grid rather than assuming a size.
    """
    filters = np.asarray(model.extraction_layer.filters).ravel()
    spacing = float(filters[1] - filters[0])
    sigma = float(np.asarray(model.extraction_layer.sigma))
    ratio = sigma/spacing
    volumes = np.sort(np.asarray(model.extraction_layer.volumes).ravel())
    return dict(
        sigma=sigma, filter_spacing=spacing, sigma_over_spacing=ratio,
        sigma_at_floor=bool(abs(ratio - 2.0) < 1e-3),
        sigma_at_ceiling=bool(abs(ratio - 6.0) < 1e-3),
        branches_per_decade=float(volumes.size/np.log10(volumes[-1]/volumes[0])),
        )


def base_rates_from(pool):
    """The training population's marginal for each target -- the baseline every head is scored on."""
    rates = {}
    for target in Prior.TARGETS:
        counts = np.bincount(
            pool[f'target_{target}'].to_numpy(), minlength=Prior.TARGET_CLASSES[target],
            ).astype(float)
        rates[target] = counts/counts.sum()
    return rates


# -----------------------------------------------------------------------------------------
# Stages
# -----------------------------------------------------------------------------------------
def bootstrap_scope(model, evaluation, base_rates, n_bootstrap=1000, seed=12345):
    """Clustered confidence intervals for the headline calibration numbers.

    Resampling is over **source entries**, not rows, because each of the 1 191 `fom-dev` entries
    appears once per condition bundle and its six rows share a crystal and correlated noise. Treating
    them as independent would divide the interval by about the square root of six and make a
    development-scale run look far more certain than it is (PROTOCOL section 8, METRICS.md).

    The model is only run forward once; the replicates resample its stored per-row outputs, so this
    costs seconds rather than a refit.
    """
    rng = np.random.default_rng(seed)
    evaluation_q2 = np.stack(evaluation['q2_obs'].to_numpy()).astype(float)
    raw = predict_in_batches(model, evaluation_q2)

    entries = evaluation['entry_id'].to_numpy()
    unique = np.unique(entries)
    membership = {entry: np.flatnonzero(entries == entry) for entry in unique}

    rows = []
    for target in Prior.TARGETS:
        log_probability = np.asarray(raw[target], dtype=float)
        labels = evaluation[f'target_{target}'].to_numpy()
        probability = np.exp(log_probability)
        confidence = probability.max(axis=1)
        correct = (probability.argmax(axis=1) == labels).astype(float)

        gains, eces, accuracies = [], [], []
        for _ in range(n_bootstrap):
            drawn = rng.choice(unique, size=unique.size, replace=True)
            picked = np.concatenate([membership[entry] for entry in drawn])
            gain, _, _ = information_gain_bits(
                log_probability[picked], labels[picked], base_rates[target],
                )
            gains.append(gain)
            eces.append(reliability(confidence[picked], correct[picked])[1])
            accuracies.append(float(correct[picked].mean()))
        point_gain, _, _ = information_gain_bits(log_probability, labels, base_rates[target])
        rows.append(dict(
            target=target, n_entries=int(unique.size), n_rows=int(labels.size),
            information_gain_bits=point_gain,
            gain_ci_low=float(np.percentile(gains, 2.5)),
            gain_ci_high=float(np.percentile(gains, 97.5)),
            ece=reliability(confidence, correct)[1],
            ece_ci_low=float(np.percentile(eces, 2.5)),
            ece_ci_high=float(np.percentile(eces, 97.5)),
            accuracy=float(correct.mean()),
            accuracy_ci_low=float(np.percentile(accuracies, 2.5)),
            accuracy_ci_high=float(np.percentile(accuracies, 97.5)),
            ))
    return rows


def run_bootstrap(args):
    """Load the fitted model and attach clustered intervals to what it already reported."""
    import pandas as pd

    pool_fit, pool_test = build_pools(args)
    evaluation = (
        holdout_evaluation(pool_test, np.random.default_rng(args.seed + 7))
        if args.eval_source == 'heldout' else
        load_evaluation(args.benchmark_dir, 'fom-dev', limit=args.limit_evaluation, seed=args.seed)
        )
    # `save_prior` writes under the split group, which for this model is always 'global'.
    model = Prior.PriorNetwork.load_prior(
        os.path.join(args.models_dir, args.arm, Prior.GLOBAL_SPLIT_GROUP), seed=args.seed,
        )
    rows = bootstrap_scope(
        model, evaluation, base_rates_from(pool_fit),
        n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
    table = pd.DataFrame(rows)
    path = os.path.join(args.artifact_dir, f'{args.tag}_bootstrap.csv')
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    return table


def run_a5(args):
    """A5, paired: does conditioning the lattice on the volume branch actually buy anything?

    Comparing three arms by their point estimates is not an answer -- F-103's clustered interval on
    a single arm is about +/-0.06 bits wide, and the arms sit closer together than that. So the
    three saved models are run over the **same** evaluation rows and the *difference* in per-row log
    likelihood is bootstrapped, clustered on source entries. A paired difference has far less
    variance than the difference of two independent estimates, which is the whole reason to do it
    this way.

    Capacity is matched by construction and that matters for the comparison to mean anything: all
    three arms apply the same `Dense(d_model -> n_classes)` and differ only in where the
    normalisation happens -- marginalised over branches after (`conditional`), pooled over branches
    before (`marginal`), or one softmax over the flattened branch x class grid (`joint`).
    """
    import pandas as pd

    pool_fit, pool_test = build_pools(args)
    evaluation = (
        holdout_evaluation(pool_test, np.random.default_rng(args.seed + 7))
        if args.eval_source == 'heldout' else
        load_evaluation(args.benchmark_dir, 'fom-dev', limit=args.limit_evaluation, seed=args.seed)
        )
    evaluation_q2 = np.stack(evaluation['q2_obs'].to_numpy()).astype(float)
    entries = evaluation['entry_id'].to_numpy()
    unique = np.unique(entries)
    rng = np.random.default_rng(args.seed)

    per_arm = {}
    for arm in args.a5_arms:
        model = Prior.PriorNetwork.load_prior(
            os.path.join(args.models_dir, arm, Prior.GLOBAL_SPLIT_GROUP), seed=args.seed,
            )
        raw = predict_in_batches(model, evaluation_q2)
        per_target = {}
        for target in Prior.TARGETS:
            log_probability = np.asarray(raw[target], dtype=float)
            labels = evaluation[f'target_{target}'].to_numpy()
            # Per-row log likelihood on the truth, in bits. This is the quantity the information
            # gain averages, so pairing it pairs exactly what F-103 reports.
            per_target[target] = log_probability[np.arange(labels.size), labels]/np.log(2)
        per_arm[arm] = per_target

    # Clustered bootstrap without rebuilding an index array per replicate. Resampling entries with
    # replacement is a multinomial draw over entry counts, and the mean of a per-row quantity under
    # that draw is (counts . per-entry sums) / (counts . per-entry row counts). So the whole thing
    # is two matrix products against a (replicates x entries) count matrix -- exact, not an
    # approximation of the index-building version, and fast enough to raise the replicate count
    # rather than lower it.
    codes = np.searchsorted(unique, entries)
    entry_n = np.bincount(codes, minlength=unique.size).astype(float)
    counts = rng.multinomial(
        unique.size, np.full(unique.size, 1.0/unique.size), size=args.n_bootstrap,
        ).astype(float)
    denominator = counts @ entry_n

    rows = []
    reference = args.a5_arms[0]
    for arm in args.a5_arms[1:]:
        for target in Prior.TARGETS:
            difference = per_arm[reference][target] - per_arm[arm][target]
            entry_sum = np.bincount(codes, weights=difference, minlength=unique.size)
            replicates = (counts @ entry_sum)/denominator
            low = float(np.percentile(replicates, 2.5))
            high = float(np.percentile(replicates, 97.5))
            rows.append(dict(
                target=target, reference=reference, arm=arm,
                delta_bits=float(difference.mean()), ci_low=low, ci_high=high,
                significant=bool(low > 0 or high < 0),
                n_entries=int(unique.size), n_rows=int(difference.size),
                ))
    table = pd.DataFrame(rows)
    path = os.path.join(args.artifact_dir, f'{args.tag}_a5.csv')
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    return table


def run_conditions(args):
    """Per-condition breakdown: the sigma-sensitivity curve, and the answer to Q37.

    Two standing requirements are discharged by the same pass. **PLAN section 2.5** demands a
    sigma-sensitivity curve for anything that touches sigma, and this model is trained on peak lists
    noised with the repo's own sigma(q2) -- the leakage path F-008 names -- so the error x1 against
    error x2 bundles are the curve. **Q37** asks whether centring is weak because dropout destroys
    systematic absences or because centring is simply harder, and the dropout bundles against the
    clean one separate those.

    The bound that does not move: the grid varies the error *scale*, never the error *law*
    (**R11**), so nothing here is evidence about a different instrument or a heavier tail.
    """
    import pandas as pd

    pool_fit, pool_test = build_pools(args)
    evaluation = (
        holdout_evaluation(pool_test, np.random.default_rng(args.seed + 7))
        if args.eval_source == 'heldout' else
        load_evaluation(args.benchmark_dir, 'fom-dev', limit=args.limit_evaluation, seed=args.seed)
        )
    base_rates = base_rates_from(pool_fit)
    model = Prior.PriorNetwork.load_prior(
        os.path.join(args.models_dir, args.arm, Prior.GLOBAL_SPLIT_GROUP), seed=args.seed,
        )
    raw = predict_in_batches(
        model, np.stack(evaluation['q2_obs'].to_numpy()).astype(float),
        )

    bundles = evaluation['condition_bundle'].to_numpy()
    rows = []
    for bundle in sorted(set(bundles)):
        selected = np.flatnonzero(bundles == bundle)
        for target in Prior.TARGETS:
            log_probability = np.asarray(raw[target], dtype=float)[selected]
            labels = evaluation[f'target_{target}'].to_numpy()[selected]
            probability = np.exp(log_probability)
            correct = (probability.argmax(axis=1) == labels).astype(float)
            gain, entropy, _ = information_gain_bits(log_probability, labels, base_rates[target])
            rows.append(dict(
                condition_bundle=bundle, target=target, n=int(selected.size),
                information_gain_bits=gain,
                share_of_available=gain/entropy if entropy else float('nan'),
                ece=reliability(probability.max(axis=1), correct)[1],
                accuracy=float(correct.mean()),
                ))
    table = pd.DataFrame(rows)
    path = os.path.join(args.artifact_dir, f'{args.tag}_conditions.csv')
    table.to_csv(path, index=False)
    print(table.pivot(index='condition_bundle', columns='target',
                      values='information_gain_bits').to_string())
    return table


def holdout_evaluation(pool, rng, n_peaks=20):
    """Evaluation frame built from held-out *training* structures, under drawn conditions.

    This is the network's own measurement -- does it predict the volume branch and the Bravais
    lattice -- and it is the headline. The FOM benchmark is a different question (what block C will
    consume) and is reported beside it, not instead of it.
    """
    q2, bundle_index, _ = Prior.draw_peak_lists(pool, rng, n_peaks=n_peaks)
    frame = pool.copy()
    frame['q2_obs'] = list(q2)
    frame['entry_id'] = frame['identifier']
    frame['condition_bundle'] = [
        Prior.CONDITION_BUNDLES[index]['name'] for index in bundle_index
        ]
    frame['volume_true'] = frame['volume']
    return frame


def build_dense_baseline(n_peaks, targets, hidden=(1024, 512, 256, 128, 64, 32),
                         learning_rate=0.001):
    """The reference model: a plain dense stack on raw q2, exactly as `scripts/classification.py`.

    Architecture copied from that script -- gelu, no bias, [1024, 512, 256, 128, 64, 32] into a
    softmax -- because the point is to reproduce the comparison DWMM already trusts, not to invent
    a new baseline. Two deliberate departures, both to make the comparison isolate the *
    representation* rather than the experiment around it: it is trained through this module's data
    path (same structures, same drawn conditions, same held-out split, same class balancing as the
    prior network), and it carries the same five heads off one trunk rather than being refitted per
    target.

    What it cannot have is a volume-branch head: there is no branch grid without the extraction
    layer, which is precisely the thing being compared. So the prior network is scored against this
    on the five class heads and stands alone on the volume.
    """
    import keras

    inputs = keras.Input(shape=(n_peaks,), name='q2_obs_scaled', dtype='float32')
    x = inputs
    for index, width in enumerate(hidden):
        x = keras.layers.Dense(
            width, activation='gelu', use_bias=False, name=f'dense_{index}',
            )(x)
    outputs = {
        target: keras.layers.Dense(
            n_classes, activation='linear', name=target,
            )(x)
        for target, n_classes in targets.items()
        }
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={target: keras.losses.SparseCategoricalCrossentropy(from_logits=True)
              for target in targets},
        metrics={target: ['sparse_categorical_accuracy'] for target in targets},
        )
    return model


class DenseBaseline:
    """Thin wrapper so the baseline reads and reports through exactly the same code as the network.

    Per-feature `StandardScaler` on the raw q2, which is `classification.py`'s own choice and is
    strictly more flexible than the prior network's single scalar -- so the baseline is not being
    handicapped on normalisation.
    """

    def __init__(self, n_peaks, targets, seed=12345):
        self.targets = dict(targets)
        self.n_peaks = n_peaks
        self.seed = seed
        self.scaler = None
        self.model = None
        self.model_params = {'peak_length': n_peaks, 'head_mode': 'dense_baseline'}

    def fit_scaler(self, q2):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(np.asarray(q2, dtype=float))
        return self

    def scale_peaks(self, q2):
        return self.scaler.transform(np.asarray(q2, dtype=float)).astype('float32')

    def build(self, learning_rate=0.001):
        import keras
        keras.utils.set_random_seed(self.seed)
        self.model = build_dense_baseline(self.n_peaks, self.targets, learning_rate=learning_rate)
        return self


def run_baseline(args):
    """Train the dense reference on the same data the prior network sees, and score it the same way."""
    import pandas as pd

    pool_fit, pool_test = build_pools(args)
    evaluation = holdout_evaluation(pool_test, np.random.default_rng(args.seed + 7))
    base_rates = base_rates_from(pool_fit)
    rng = np.random.default_rng(args.seed)

    model = DenseBaseline(Prior.PRIOR_MODEL_DEFAULTS['peak_length'], Prior.TARGET_CLASSES,
                          seed=args.seed)
    scaler_q2, _, _ = Prior.draw_peak_lists(pool_fit.iloc[:20000], rng)
    model.fit_scaler(scaler_q2).build()
    if args.verbose:
        print(f'dense baseline: {model.model.count_params():,} parameters, '
              f'fit on {len(pool_fit):,} structures', flush=True)

    codes = pool_fit['target_bravais'].to_numpy()
    for epoch in range(args.epochs):
        rows = Prior.balanced_indices(codes, len(Prior.BRAVAIS_LATTICES), rng, args.per_class)
        batch = pool_fit.iloc[rows].reset_index(drop=True)
        q2, _, _ = Prior.draw_peak_lists(batch, rng)
        targets = {name: batch[f'target_{name}'].to_numpy() for name in Prior.TARGETS}
        record = model.model.fit(
            model.scale_peaks(q2), targets, epochs=1, verbose=0, batch_size=64,
            )
        if args.verbose:
            print(f'  baseline epoch {epoch + 1}/{args.epochs} '
                  f'loss {float(record.history["loss"][-1]):.4f}', flush=True)

    directory = os.path.join(args.models_dir, 'dense_baseline')
    os.makedirs(directory, exist_ok=True)
    model.model.save_weights(os.path.join(directory, 'baseline.weights.h5'))

    raw = predict_in_batches(
        model, np.stack(evaluation['q2_obs'].to_numpy()).astype(float),
        )
    rows = []
    for target in Prior.TARGETS:
        logits = np.asarray(raw[target], dtype=float)
        log_probability = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        labels = evaluation[f'target_{target}'].to_numpy()
        gain, entropy, _ = information_gain_bits(log_probability, labels, base_rates[target])
        rows.append(dict(
            model='dense_baseline', target=target,
            accuracy=float((log_probability.argmax(axis=1) == labels).mean()),
            information_gain_bits=gain, base_rate_entropy_bits=entropy,
            n=int(labels.size),
            ))

    # Per-lattice diagonals and the full confusion, so this lines up row for row with the network's
    # own `_per_lattice.csv` and `_confusion.csv` and with the archived `classification.py` matrix.
    # A macro average over lattices is reported beside the pooled accuracy because the evaluation
    # population is the natural one and is 36% mP -- a pooled number is dominated by two classes.
    lattices = list(active_lattices(args) or Prior.BRAVAIS_LATTICES)
    active = np.array([Prior.BRAVAIS_LATTICES.index(code) for code in lattices])
    bravais = np.exp(np.asarray(raw['bravais'], dtype=float))[:, active]
    bravais = bravais/bravais.sum(axis=1, keepdims=True)
    labels = evaluation['target_bravais'].to_numpy()
    position = {int(index): slot for slot, index in enumerate(active)}
    slots = np.array([position[int(value)] for value in labels])

    per_lattice, confusion = [], []
    for slot, code in enumerate(lattices):
        selected = np.flatnonzero(slots == slot)
        if selected.size == 0:
            continue
        block = bravais[selected]
        per_lattice.append(dict(
            model='dense_baseline', bravais_lattice=code, n=int(selected.size),
            accuracy=float((block.argmax(axis=1) == slot).mean()),
            mean_probability_on_truth=float(block[:, slot].mean()),
            ))
        for other, other_code in enumerate(lattices):
            confusion.append(dict(
                model='dense_baseline', true=code, predicted=other_code,
                argmax_fraction=float((block.argmax(axis=1) == other).mean()),
                mean_probability=float(block[:, other].mean()),
                ))
    per_lattice_table = pd.DataFrame(per_lattice)
    per_lattice_table.to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_baseline_per_lattice.csv'), index=False,
        )
    pd.DataFrame(confusion).to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_baseline_confusion.csv'), index=False,
        )
    rows.append(dict(
        model='dense_baseline', target='bravais_macro',
        accuracy=float(per_lattice_table['accuracy'].mean()),
        information_gain_bits=float('nan'), base_rate_entropy_bits=float('nan'),
        n=int(len(evaluation)),
        ))
    print(per_lattice_table.round(4).to_string(index=False))
    print()

    table = pd.DataFrame(rows)
    path = os.path.join(args.artifact_dir, f'{args.tag}_baseline.csv')
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    return table


# Peak memory in a forward pass is set by the extraction render, not by the outputs.
# `ExtractionLayer.call` materialises (batch, n_volumes, n_filters, extraction_peak_length) for the
# filter-to-peak differences and again for their exponential before summing the peak axis away, so
# a single sample costs n_volumes * n_filters * extraction_peak_length * 4 bytes, twice over. At
# 128 x 1024 x 12 that is 12.6 MB per sample, and a 512-row batch asks for ~6.4 GB -- which on a
# 16 GB laptop means the kernel spends its time paging rather than computing.
PREDICT_BYTES_BUDGET = 512*1024*1024


def predict_batch_size(model, budget=PREDICT_BYTES_BUDGET):
    """Largest forward batch whose extraction render stays inside `budget`."""
    params = model.model_params
    per_sample = 2*4*(
        params.get('n_volumes', 1)*params.get('n_filters', 1)
        *params.get('extraction_peak_length', 1)
        )
    return int(max(8, min(512, budget//max(per_sample, 1))))


def predict_in_batches(model, q2, budget=PREDICT_BYTES_BUDGET):
    """Forward pass in memory-safe chunks, concatenated back into one dict of arrays."""
    batch = predict_batch_size(model, budget)
    scaled = model.scale_peaks(q2)
    collected = {}
    for start in range(0, scaled.shape[0], _PREDICT_CHUNK):
        chunk = model.model.predict(
            scaled[start:start + _PREDICT_CHUNK], batch_size=batch, verbose=0,
            )
        for name, values in chunk.items():
            collected.setdefault(name, []).append(np.asarray(values, dtype=np.float32))
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


# Rows per predict() call. Keras allocates its own output buffers for the whole call, so this caps
# them independently of the per-batch render above.
_PREDICT_CHUNK = 20000


def active_lattices(args):
    """Which Bravais lattices this run covers.

    Cubic is excluded by default, matching `mlindex/scripts/classification.py`, which comments the
    cubic datasets out of its own data list -- so the dense reference and the prior network are
    measured on the same eleven lattices and six crystal systems rather than on problems of
    different sizes. `--include-cubic` restores all fourteen, which is the set block C ultimately
    needs since the figure of merit has to rank cubic candidates too.

    The label encodings are deliberately left at their full width. A class that never appears simply
    carries zero mass: the base-rate entropy is computed from counts, so it correctly reflects the
    classes present, and the alternative -- re-deriving five target encodings per run -- is more
    machinery than the question needs.
    """
    if args.include_cubic:
        return None
    return tuple(code for code in Prior.BRAVAIS_LATTICES if not code.startswith('c'))


def run_confusion(args):
    """Where the wrong-class mass goes, and whether it already respects symmetry distance.

    The question this answers is whether a cost-sensitive loss would buy anything. Cross entropy is
    `-log p(true)`, so it is blind to *which* wrong class receives the mass -- but it is a proper
    scoring rule, so at the optimum the model reproduces the true conditional distribution, which
    already encodes whatever confusability is really in the data. Whether it has done so is an
    empirical question, and this is the measurement.

    The test: take each row's predicted mass on the wrong classes, renormalise it, and compute the
    expected symmetry distance `|n_free(true) - n_free(predicted)|` under it. Compare against the
    same quantity when the wrong mass is spread uniformly over the available wrong classes. Below
    the uniform line means errors concentrate on symmetry-adjacent lattices without being told to.
    """
    import pandas as pd

    pool_fit, pool_test = build_pools(args)
    evaluation = holdout_evaluation(pool_test, np.random.default_rng(args.seed + 7))
    model = Prior.PriorNetwork.load_prior(
        os.path.join(args.models_dir, args.arm, Prior.GLOBAL_SPLIT_GROUP), seed=args.seed,
        )
    raw = predict_in_batches(
        model, np.stack(evaluation['q2_obs'].to_numpy()).astype(float),
        )
    probability = np.exp(np.asarray(raw['bravais'], dtype=float))
    labels = evaluation['target_bravais'].to_numpy()

    lattices = list(active_lattices(args) or Prior.BRAVAIS_LATTICES)
    active = np.array([Prior.BRAVAIS_LATTICES.index(code) for code in lattices])
    free = np.array([
        Prior.N_FREE_OF[Prior.LATTICE_SYSTEM_OF[Prior.BRAVAIS_LATTICES[index]]]
        for index in active
        ], dtype=float)
    # Restrict to the classes this run covers; the encoding is full width and the absent ones
    # carry no trained mass, so leaving them in would only add rounding.
    probability = probability[:, active]
    probability = probability/probability.sum(axis=1, keepdims=True)
    position = {int(index): slot for slot, index in enumerate(active)}
    rows_label = np.array([position[int(value)] for value in labels])

    distance = np.abs(free[:, np.newaxis] - free[np.newaxis, :])

    confusion, summary = [], []
    for slot, code in enumerate(lattices):
        rows = np.flatnonzero(rows_label == slot)
        if rows.size == 0:
            continue
        block = probability[rows]
        for other, other_code in enumerate(lattices):
            confusion.append(dict(
                true=code, predicted=other_code, n=int(rows.size),
                mean_probability=float(block[:, other].mean()),
                argmax_fraction=float((block.argmax(axis=1) == other).mean()),
                ))
        wrong = block.copy()
        wrong[:, slot] = 0.0
        total = wrong.sum(axis=1, keepdims=True)
        keep = total.ravel() > 0
        wrong = wrong[keep]/total[keep]
        model_distance = float((wrong*distance[slot][np.newaxis]).sum(axis=1).mean())
        others = np.array([index for index in range(len(lattices)) if index != slot])
        uniform_distance = float(distance[slot][others].mean())
        summary.append(dict(
            true=code, n=int(rows.size), n_free=float(free[slot]),
            accuracy=float((block.argmax(axis=1) == slot).mean()),
            mean_probability_on_truth=float(block[:, slot].mean()),
            wrong_mass_mean_symmetry_distance=model_distance,
            uniform_baseline=uniform_distance,
            ratio=model_distance/uniform_distance if uniform_distance else float('nan'),
            ))

    pd.DataFrame(confusion).to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_confusion.csv'), index=False,
        )
    table = pd.DataFrame(summary)
    table.to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_symmetry_distance.csv'), index=False,
        )
    weighted = float(
        (table['wrong_mass_mean_symmetry_distance']*table['n']).sum()/table['n'].sum()
        )
    baseline = float((table['uniform_baseline']*table['n']).sum()/table['n'].sum())
    print(table.to_string(index=False))
    print(f'\npooled: wrong-class mass sits at mean symmetry distance {weighted:.3f} '
          f'against {baseline:.3f} if it were spread uniformly '
          f'({100*(1 - weighted/baseline):+.1f}% closer)')
    return table


def build_pools(args):
    """Split the training pool in two by source structure: fit, and a held-out test set.

    Two ways, not three: nothing is fitted after training any more, so there is no third slice to
    reserve. The held-out set is what "does the network predict the volume branch and the lattice"
    is answered on -- structures it has never seen, under freshly drawn conditions. The FOM
    benchmark answers a different question (the population block C consumes) and is available
    through `--eval-source benchmark`, beside this rather than instead of it.
    """
    pool = Prior.load_prior_frame(
        args.datasets_dir, args.manifest, limit_per_lattice=args.limit_per_lattice,
        bravais_lattices=active_lattices(args), seed=args.seed,
        )
    return split_pool(pool, args.test_fraction, args.seed)


def run(args):
    started = time.time()
    pool_fit, pool_test = build_pools(args)
    if args.eval_source == 'heldout':
        evaluation = holdout_evaluation(pool_test, np.random.default_rng(args.seed + 7))
    else:
        evaluation = load_evaluation(
            args.benchmark_dir, 'fom-dev', limit=args.limit_evaluation, seed=args.seed,
            )
    base_rates = base_rates_from(pool_fit)
    if args.verbose:
        print(f'fit {len(pool_fit)} structures / held out {len(pool_test)}; '
              f'evaluation {len(evaluation)} entry-conditions '
              f'({args.eval_source})', flush=True)

    if args.stage == 'heads':
        arms = [('conditional', {}), ('marginal', {'head_mode': 'marginal'}),
                ('joint', {'head_mode': 'joint'})]
    elif args.stage == 'grid':
        arms = [(f'n_volumes_{value}', {'n_volumes': value}) for value in args.n_volumes_sweep]
        arms += [(f'n_filters_{value}', {'n_filters': value}) for value in args.n_filters_sweep]
    else:
        arms = [('main', {})]

    records = []
    for tag, overrides in arms:
        params = dict(overrides)
        params.setdefault('epochs', args.epochs)
        params.setdefault('n_volumes', args.n_volumes)
        params.setdefault('n_filters', args.n_filters)
        if args.extraction_peaks is not None:
            params.setdefault('extraction_peak_length', args.extraction_peaks)
        if args.loss_mode is not None:
            params.setdefault('loss_mode', args.loss_mode)
        if args.volume_target_width is not None:
            params.setdefault('volume_target_width', args.volume_target_width)
        model, history = fit_arm(pool_fit, params, args, tag)
        results, tables = evaluate_arm(model, evaluation, base_rates)
        model.save_prior()
        records.append(dict(arm=tag, params=params, results=results, tables=tables,
                            history=history,
                            **grid_diagnostics(model)))
        if args.verbose:
            summary = results['bravais']
            print(f'  {tag}: bravais acc {summary["accuracy"]:.3f} ECE {summary["ece"]:.4f} '
                  f'gain {summary["information_gain_bits"]:+.3f} bits', flush=True)

    write_outputs(args, records, pool_fit, pool_test, evaluation, base_rates, started)
    return records


def write_outputs(args, records, pool_fit, pool_test, evaluation, base_rates, started):
    import pandas as pd

    os.makedirs(args.artifact_dir, exist_ok=True)
    rows = []
    for record in records:
        row = dict(arm=record['arm'])
        for key in ('sigma', 'filter_spacing', 'sigma_over_spacing', 'sigma_at_floor',
                    'sigma_at_ceiling', 'branches_per_decade'):
            row[key] = record[key]
        row.update({f'volume_{key}': value for key, value in record['results']['volume'].items()
                    if key != 'coverage'})
        for level, value in record['results']['volume']['coverage'].items():
            row[f'volume_coverage_{int(level*100)}'] = value
        for target in Prior.TARGETS:
            for key, value in record['results'][target].items():
                row[f'{target}_{key}'] = value
        rows.append(row)
    main = pd.DataFrame(rows)
    main.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_main_table.csv'), index=False)

    pd.DataFrame([
        dict(arm=record['arm'], **entry)
        for record in records for entry in record['tables']['reliability']
        ]).to_csv(os.path.join(args.artifact_dir, f'{args.tag}_reliability.csv'), index=False)
    pd.DataFrame([
        dict(arm=record['arm'], **entry)
        for record in records for entry in record['tables']['per_lattice']
        ]).to_csv(os.path.join(args.artifact_dir, f'{args.tag}_per_lattice.csv'), index=False)

    meta = dict(
        commit=commit_hash(), seed=args.seed, stage=args.stage,
        n_fit_structures=int(len(pool_fit)), n_heldout_structures=int(len(pool_test)),
        n_evaluation_rows=int(len(evaluation)), eval_source=args.eval_source,
        limit_per_lattice=args.limit_per_lattice, per_class=args.per_class, epochs=args.epochs,
        n_volumes=args.n_volumes, n_filters=args.n_filters,
        broadening_tag=Prior.BROADENING_TAG,
        lattices=list(active_lattices(args) or Prior.BRAVAIS_LATTICES),
        condition_bundles=[bundle['name'] for bundle in Prior.CONDITION_BUNDLES],
        base_rates={key: list(map(float, value)) for key, value in base_rates.items()},
        wall_clock_seconds=round(time.time() - started, 1),
        scale='development -- laptop, subsampled. NOT a production result.',
        production_configuration=dict(
            limit_per_lattice=None, per_class=20000, epochs=60,
            n_volumes=256, n_filters=1024, d_model=512, layers=[1000, 600, 300, 100, 50],
            note='the second campaign runs this on NERSC; nothing here is fitted at that scale',
            ),
        bounds=['R11: no perturbed error-model bundle exists, so robustness to a different error '
                'law is untested rather than passed',
                'R12: broadening tag 1 only, so no instrument-transfer claim is available'],
        )
    with open(os.path.join(args.artifact_dir, f'{args.tag}_meta.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print(main.to_string(index=False))
    return main


def main():
    parser = argparse.ArgumentParser(
        description='Fit and evaluate the volume + Bravais prior network (S11 block A).',
        )
    parser.add_argument('--stage', default='main',
                        choices=('main', 'heads', 'grid', 'bootstrap', 'a5',
                                 'conditions', 'baseline', 'confusion'))
    parser.add_argument('--arm', default='main',
                        help='Saved arm to re-evaluate, for --stage bootstrap.')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--a5-arms', nargs='+',
                        default=['conditional', 'marginal', 'joint'],
                        help='Saved arms to pair; the first is the reference.')
    parser.add_argument('--datasets-dir', default=DEFAULT_DATASETS)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK)
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACTS)
    parser.add_argument('--models-dir', default=DEFAULT_MODELS)
    parser.add_argument('--tag', default='S11_A_prior')
    parser.add_argument('--limit-per-lattice', type=int, default=4000,
                        help='Structures per Bravais lattice, after the split filter. Development '
                             'scale; None uses the whole pool.')
    parser.add_argument('--limit-evaluation', type=int, default=None,
                        help='Cap on evaluation rows, for a fast pass.')
    parser.add_argument('--per-class', type=int, default=2000,
                        help='Balanced draws per Bravais lattice per epoch.')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--n-volumes', type=int, default=128)
    parser.add_argument('--n-filters', type=int, default=1024)
    parser.add_argument('--volume-target-width', type=float, default=None,
                        help='Width in branches of the soft volume target. 0 keeps the '
                             'ordinal-blind one-hot label.')
    parser.add_argument('--loss-mode', default=None, choices=('marginal', 'joint'),
                        help="'joint' trains -log P(v*) - log P(c*|v*), the likelihood "
                             'of the observed (volume, lattice) pair, instead of the '
                             'branch-averaged marginal.')
    parser.add_argument('--extraction-peaks', type=int, default=None,
                        help='Peaks the render reads. Default 12, inherited from the '
                             'triclinic ABNN config; peaks 13-20 land inside the filter '
                             'grid for 100%% of aP and 99%% of mP patterns and outside it '
                             'for tI and oF, so raising it targets the failing lattices.')
    parser.add_argument('--n-volumes-sweep', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--n-filters-sweep', type=int, nargs='+', default=[256, 512, 1024, 2048])
    parser.add_argument('--test-fraction', type=float, default=0.15,
                        help='Structures held out of training entirely, for the headline.')
    parser.add_argument('--eval-source', default='heldout',
                        choices=('heldout', 'benchmark'),
                        help="'heldout' = unseen training structures (the network's own "
                             "measurement); 'benchmark' = the FOM pool block C consumes.")
    parser.add_argument('--include-cubic', action='store_true',
                        help='Cover all fourteen lattices. Default excludes cubic, to '
                             'match scripts/classification.py.')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    args = parser.parse_args()
    if args.limit_per_lattice is not None and args.limit_per_lattice <= 0:
        args.limit_per_lattice = None
    if args.stage == 'bootstrap':
        run_bootstrap(args)
    elif args.stage == 'a5':
        run_a5(args)
    elif args.stage == 'conditions':
        run_conditions(args)
    elif args.stage == 'baseline':
        run_baseline(args)
    elif args.stage == 'confusion':
        run_confusion(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
