"""S14 -- block A: fit and evaluate the volume + Bravais prior network.

    python -m mlindex.scripts.run_fom_prior --stage main --include-cubic

Campaign 1's `S11` driver (`fom@7c137c3`), cut to the one stage S14 uses: the `main` fit, its
held-out evaluation, and a per-lattice table that now carries macro F1 with the predicted share
beside it (campaign 1's F-115: class balancing had oF predicted 26x more often than it occurs, at
2.6 % precision, which recall alone hides). The `heads`, `grid`, `a5`, `conditions`, `baseline`,
`bootstrap` and `confusion` stages stay on `fom`; their questions were answered there
(`docs/fom_campaign2/INHERITED.md` section 1) and are not re-asked.

Training data is `mlindex/data/generated_datasets/`, filtered against campaign 2's frozen split
manifest (`fom_split_c2.parquet`, column `fom_split`) so no `fom-dev` or `fom-test` structure is
ever seen. `--eval-source benchmark` evaluates on a campaign-2 pool's own peak lists instead of a
re-draw of the training conditions; pass `--bundles` to pick the pool's condition bundles, since
their names (`c2_...`) are not the training mix's.

The model records the lattices it was trained on as `support`, and every consumer reads it through
that (`PriorNetwork.entry_tables`). `--include-cubic` trains all fourteen; the default keeps campaign
1's eleven so the two can be compared on `S14_prior_interface.csv`.

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
DEFAULT_MANIFEST = os.path.join('mlindex', 'data', 'generated_datasets', 'fom_split_c2.parquet')
DEFAULT_BENCHMARK = os.path.join('mlindex', 'data', 'fom_full_c2_pool')
DEFAULT_ARTIFACTS = os.path.join('docs', 'fom_campaign2', 'artifacts')
DEFAULT_MODELS = os.path.join('mlindex', 'models', 'fom_prior_c2')

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
    # Every bundle the pool carries unless told otherwise. Campaign 1 defaulted to the training
    # mix's names, which a campaign-2 pool (`c2_error1_cont0`, ...) never matches -- the evaluation
    # frame then came back empty with no symptom (S14, 2026-09-05).
    if bundles:
        missing = sorted(set(bundles) - set(frame['condition_bundle']))
        if missing:
            raise ValueError(f'{benchmark_dir} has no rows for condition bundle(s) {missing} '
                             f'in split {split}; it carries '
                             f'{sorted(set(frame["condition_bundle"]))}')
        frame = frame.loc[frame['condition_bundle'].isin(set(bundles))]
    frame = frame.reset_index(drop=True)
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
def check_balanced(batch, pool, per_class, epoch):
    """Assert the invariant balanced sampling is supposed to guarantee, every epoch.

    A sampler that silently narrows its own pool cost two 80-minute runs, four wrong hypotheses and
    two withdrawn findings before anyone looked at what was actually in a batch (F-121). Balanced
    sampling has exactly one invariant -- every class present, in equal numbers -- so it is checked
    rather than inferred from a loss curve.
    """
    present = batch['bravais_lattice'].nunique()
    expected = pool['bravais_lattice'].nunique()
    if present != expected or len(batch) != per_class*expected:
        raise RuntimeError(
            f'epoch {epoch}: balanced batch holds {len(batch)} rows over {present} lattices, '
            f'expected {per_class*expected} over {expected}. The sampler has lost the pool.'
            )


def fit_arm(pool_fit, model_params, args, tag):
    """Fit one arm. A fresh condition draw per epoch, which is what makes balancing honest."""
    rng = np.random.default_rng(args.seed)
    grid_q2, _, _ = Prior.draw_peak_lists(pool_fit, rng)
    grid_frame = pool_fit.assign(q2_window=list(grid_q2))

    model_params = dict(model_params)
    # The lattices this fit can speak for, written into the checkpoint so a consumer never reads
    # an untrained class as a probability (`PriorNetwork.support`). It is the pool's own lattice
    # set, not the CLI's intent: a pool that lost a lattice to a filter would otherwise claim it.
    model_params['support'] = [code for code in Prior.BRAVAIS_LATTICES
                               if code in set(pool_fit['bravais_lattice'])]
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
        check_balanced(batch, pool_fit, args.per_class, epoch + 1)
        q2, _, _ = Prior.draw_peak_lists(batch, rng)
        branch = model.get_branch_labels(batch)
        joint = model.model_params['loss_mode'] == 'joint'
        width = model.model_params.get('volume_target_width', 0.0)
        targets = {'volume_branch': (
            Prior.soft_branch_targets(branch, model.model_params['n_volumes'], width)
            if width > 0 else branch
            )}
        for name in Prior.TARGETS:
            # `values`, not `codes`. Rebinding `codes` here shadowed the sampler's input, which is
            # bound once outside this loop: from epoch 2 onward `balanced_indices` was handed the
            # previous batch's `high_symmetry` labels -- a 2-class array of *batch* length -- so its
            # row indices were bounded by that length and `iloc` took the front of the pool. The
            # pool is concatenated per lattice, so the front is tP, and the batch converged to
            # 6 000 identical-lattice rows by epoch 5. The five class losses then read 0.000
            # because they were predicting a constant, and two 30-epoch runs plus two wrong
            # conclusions came out of it (F-120).
            values = batch[f'target_{name}'].to_numpy()
            targets[name] = Prior.joint_targets(values, branch) if joint else values
        record = model.model.fit(
            model.scale_peaks(q2), targets, epochs=1, verbose=0,
            batch_size=model.model_params['batch_size'],
            )
        history.append({key: float(value[-1]) for key, value in record.history.items()})
        if args.verbose:
            # Per-head losses, not just the total. `fit` has carried them all along and logging only
            # the total cost three runs: two collapsed with a total that was arithmetically
            # impossible for the predictions they made, and which head died was only inferable
            # afterwards. A head going flat is visible here on the epoch it happens (F-120).
            parts = ' '.join(
                f'{key[:-5]} {value:.3f}'
                for key, value in sorted(history[-1].items())
                if key.endswith('_loss') and key != 'loss'
                )
            print(f'  {tag} epoch {epoch + 1}/{model.model_params["epochs"]} '
                  f'loss {history[-1]["loss"]:.4f} | {parts}', flush=True)
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
    tables['per_lattice'] = macro_f1_table(lattice_log, labels, model.support,
                                           base_rates['bravais'])
    results['bravais']['macro_f1'] = float(np.nanmean(
        [row['f1'] for row in tables['per_lattice'] if row['in_support']]
        ))
    return results, tables


def macro_f1_table(lattice_log, labels, support, base_rate):
    """Per-lattice precision, recall, F1 and predicted share, plus what campaign 1 reported.

    Recall alone is the number that flattered campaign 1 (F-115): class balancing had oF predicted
    26x more often than it occurs, so its 67.6 % recall rested on 2.6 % precision. The predicted
    share beside the true share is what makes that visible, and macro F1 over the support is the
    headline the S14 gate asks for. Classes outside the support are still scored -- they are the
    before/after row of `S14_prior_interface.csv` -- and flagged.

    The argmax is taken over the raw fourteen-class head, deliberately: this table is the
    measurement of what the head does, support mask or not.
    """
    lattice_log = np.asarray(lattice_log, dtype=float)
    labels = np.asarray(labels)
    predicted = lattice_log.argmax(axis=1)
    support = set(support)
    rows = []
    for index, code in enumerate(Prior.BRAVAIS_LATTICES):
        truth = labels == index
        claim = predicted == index
        n_true, n_pred = int(truth.sum()), int(claim.sum())
        hits = int((truth & claim).sum())
        precision = hits/n_pred if n_pred else float('nan')
        recall = hits/n_true if n_true else float('nan')
        f1 = (2*precision*recall/(precision + recall)
              if n_pred and n_true and hits else (0.0 if (n_pred or n_true) else float('nan')))
        gain = (information_gain_bits(lattice_log[truth], labels[truth], base_rate)[0]
                if n_true else float('nan'))
        rows.append(dict(
            bravais_lattice=code, n=n_true, in_support=code in support,
            accuracy=recall if n_true else float('nan'),
            precision=precision, recall=recall, f1=f1,
            predicted_share=n_pred/labels.size, true_share=n_true/labels.size,
            information_gain_bits=gain,
            mean_probability_on_truth=(float(np.exp(lattice_log[truth, index]).mean())
                                       if n_true else float('nan')),
            median_log_probability=float(np.median(lattice_log[:, index])),
            median_rank=float(np.median(
                (lattice_log > lattice_log[:, index][:, np.newaxis]).sum(axis=1) + 1
                )),
            max_probability=float(np.exp(lattice_log[:, index]).max()),
            ))
    return rows


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


# -----------------------------------------------------------------------------------------
# Stage: interface -- the untrained-class defect, before and after
# -----------------------------------------------------------------------------------------
def run_interface(args):
    """`S14_prior_interface.csv`: every lattice under every prior dir given, raw head and masked.

    The defect (F-117 point 4; S14 gate condition 1): the shipped head has fourteen classes,
    three never trained, and reads cubic at median log P about -19 for every pattern including the
    cubic ones. This table measures that, then measures the two fixes -- the support mask on the
    same checkpoint, and a fourteen-lattice retrain where one exists -- on one evaluation frame:
    a campaign-2 pool's own `fom-dev` peak lists, which carry all fourteen true lattices.

    Macro F1 with the predicted share beside it, per gate condition 3; never recall alone.
    """
    import pandas as pd

    evaluation = load_evaluation(args.benchmark_dir, 'fom-dev', bundles=args.bundles,
                                 limit=args.limit_evaluation, seed=args.seed)
    labels = evaluation['target_bravais'].to_numpy()
    q2 = np.stack(evaluation['q2_obs'].to_numpy()).astype(float)
    counts = np.bincount(labels, minlength=len(Prior.BRAVAIS_LATTICES)).astype(float)
    base_rate = counts/counts.sum()
    rows = []
    for label, directory in zip(args.interface_labels, args.interface_dirs):
        model = Prior.PriorNetwork.load_prior(directory)
        raw = predict_in_batches(model, q2)
        raw_log = np.asarray(raw['bravais'], dtype=float)
        masked_log = model.entry_tables(q2, batch_size=256)['bravais_lp']
        for readout, lattice_log in (('raw_head', raw_log), ('support_masked', masked_log)):
            for row in macro_f1_table(np.where(np.isnan(lattice_log), -np.inf, lattice_log),
                                      labels, model.support, base_rate):
                row.update(model=label, readout=readout, prior_dir=str(directory),
                           support_defaulted=bool(model.support_defaulted),
                           n_evaluation=int(labels.size))
                rows.append(row)
        supported = np.array([code in set(model.support) for code in Prior.BRAVAIS_LATTICES])
        for readout, lattice_log in (('raw_head', raw_log), ('support_masked', masked_log)):
            table = [r for r in rows if r['model'] == label and r['readout'] == readout]
            f1_support = float(np.nanmean([r['f1'] for r in table if r['in_support']]))
            f1_all = float(np.nanmean([r['f1'] for r in table]))
            print(f'{label:12s} {readout:15s} macro F1 over support {f1_support:.3f}, over all '
                  f'fourteen {f1_all:.3f}; support {len(model.support)} lattices'
                  f'{" (defaulted)" if model.support_defaulted else ""}')
            _ = supported
    out = pd.DataFrame(rows)
    os.makedirs(args.artifact_dir, exist_ok=True)
    path = os.path.join(args.artifact_dir, f'{args.tag}_interface.csv')
    out.to_csv(path, index=False)
    print(f'wrote {path}')
    return out


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
            args.benchmark_dir, 'fom-dev', bundles=args.bundles, limit=args.limit_evaluation,
            seed=args.seed,
            )
    base_rates = base_rates_from(pool_fit)
    if args.verbose:
        print(f'fit {len(pool_fit)} structures / held out {len(pool_test)}; '
              f'evaluation {len(evaluation)} entry-conditions '
              f'({args.eval_source})', flush=True)

    arms = [(args.arm, {})]

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

    pd.DataFrame([
        dict(arm=record['arm'], epoch=index + 1, **entry)
        for record in records for index, entry in enumerate(record['history'])
        ]).to_csv(os.path.join(args.artifact_dir, f'{args.tag}_history.csv'), index=False)

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
            note='campaign 1 named this as its production configuration and never ran it; '
                 'S14 retrains at campaign 1\'s `main` configuration with cubic included',
            ),
        support=list(records[0]['params'].get('support', [])) if records else [],
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
    parser.add_argument('--stage', default='main', choices=('main', 'interface'))
    parser.add_argument('--interface-dirs', nargs='*',
                        default=[os.path.join('mlindex', 'models', 'fom_prior', 'main', 'global')],
                        help='Prior checkpoints to compare in --stage interface')
    parser.add_argument('--interface-labels', nargs='*', default=['shipped_11'],
                        help='One label per --interface-dirs entry')
    parser.add_argument('--arm', default='main',
                        help='Name of the fitted arm; the model is saved under '
                             '<models-dir>/<arm>/global/.')
    parser.add_argument('--bundles', nargs='*', default=None,
                        help='Condition bundles to evaluate on with --eval-source benchmark. '
                             'Default: every bundle the pool carries.')
    parser.add_argument('--datasets-dir', default=DEFAULT_DATASETS)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK)
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACTS)
    parser.add_argument('--models-dir', default=DEFAULT_MODELS)
    parser.add_argument('--tag', default='S14_prior')
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
    if args.stage == 'interface':
        if len(args.interface_dirs) != len(args.interface_labels):
            raise SystemExit('--interface-dirs and --interface-labels must pair up')
        run_interface(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
