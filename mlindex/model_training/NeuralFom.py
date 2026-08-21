"""S11 block C: a network over the assembled features, rather than the tree that assembles them.

**This is the last rung, not the first.** PLAN section 4 says to escalate to a network only on
measured residual structure, and the four tree arms in `run_fom_neural.py --stage combiner` are
what measure it. The network is run afterwards regardless of what they show (DWMM, 2026-08-20), so
the architecture question is answered rather than inferred -- but it is answered against a tree
that has already been given the same columns, which is the only comparison that isolates the
architecture from the features.

Two things it exists to measure that the tree cannot:

  * **F-081, re-measured as that finding instructs.** S07's scale normalisers make no difference to
    a tree -- 0.5 pp across raw, scaled and both -- because `z` and `rank` are monotone within a
    lattice and a tree is invariant to a monotone transform. A network is not. F-081 says S11 and
    S14 must re-measure rather than assume, so `run_fom_neural.py --stage network` fits the same
    architecture on raw and on scaled features and pairs them.
  * **Whether the interactions are the kind a tree cannot express.** A gradient-boosted tree with
    63 leaves already builds deep conjunctions of thresholds. What it cannot build is a smooth
    function of many features at once, which is what `P(volume, lattice) x P(peaks fit)` is if the
    factorisation is real.

Held the same way `DistilledCombiner` is -- plain arrays, imputed with the training median,
standardised, `relu` between layers -- because F-092 answered Q4 with that form at 0.17x `get_M20`
and a deployable block C has to be the same shape. The difference is what it is fitted on: the
labels, not a teacher's output.

**The training checks are not optional here** (S11 session 1, F-121). A shadowed variable once
decayed a training set to a single lattice from epoch 2, cost three 30-epoch runs and two withdrawn
findings, and the reported loss was arithmetically impossible for the model's own predictions the
whole time. Both invariants are asserted below: the batch composition before the fit, and the
reported loss against the model's own predictions after it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomCombiner
from mlindex.model_training import FomMetrics


class CompositionError(AssertionError):
    """The training matrix is not what the caller believes it is."""


def check_composition(matrix, target, expected_positive_rate=None, tolerance=0.5):
    """The batch, not the loss (F-121). Raises rather than warning, because a warning was missed.

    Two ways the assembled matrix can be silently wrong, both of which have happened in this
    project: it can lose its positives (a filter applied one step too early), and it can go
    non-finite in a column the tree tolerated and the network does not.
    """
    if matrix.shape[0] != target.shape[0]:
        raise CompositionError(f'{matrix.shape[0]} rows against {target.shape[0]} labels')
    positive = float(target.mean())
    if positive <= 0.0 or positive >= 1.0:
        raise CompositionError(f'one class only: positive rate {positive}')
    if expected_positive_rate is not None:
        low = expected_positive_rate*(1.0 - tolerance)
        high = expected_positive_rate*(1.0 + tolerance)
        if not low <= positive <= high:
            raise CompositionError(
                f'positive rate {positive:.5f} outside [{low:.5f}, {high:.5f}] -- the training '
                'set is not the one that was assembled'
                )
    return dict(n_rows=int(matrix.shape[0]), n_features=int(matrix.shape[1]),
                positive_rate=positive,
                n_non_finite=int((~np.isfinite(matrix)).sum()))


def check_not_constant(score, target, minimum_auc=0.55):
    """The model must have learned something. A constant predictor is the failure mode here.

    **This is not a formality, it caught the first fit of this class.** `MLPClassifier`'s
    `early_stopping` scores its validation split on **accuracy**, and at this project's base rate a
    predictor that answers "incorrect" to everything scores 0.95 -- so `best_validation_score_`
    reached 0.95 on the first epoch, `n_iter_no_change` expired at iteration 10, and the fit
    returned a model at AUC 0.46 while reporting a validation score of 0.95. It is F-115's warning
    in a new place: an aggregate metric under class imbalance reports the wrong winner, and here it
    reported a winner that had not started.
    """
    score = np.asarray(score, dtype=np.float64)
    if np.ptp(score) < 1e-9:
        raise CompositionError(f'the fitted model is constant at {score[0]:.6f}')
    auc = float(FomMetrics.average_precision(score, np.asarray(target, dtype=bool)))
    order = np.argsort(score)
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(score.size)
    positive = np.asarray(target, dtype=bool)
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    roc = ((ranks[positive].sum() - n_positive*(n_positive - 1)/2)
           / max(n_positive*n_negative, 1))
    if roc < minimum_auc:
        raise CompositionError(
            f'the fitted model scores AUC {roc:.4f} on its own training rows, which is not a '
            'model that has been trained'
            )
    return dict(train_roc_auc=float(roc), train_average_precision=auc,
                score_range=[float(score.min()), float(score.max())])


def check_loss_is_possible(model, matrix, target, factor=3.0):
    """`model.loss_` against the log loss of the model's own predictions on the same rows.

    F-121's other half. A loss that no set of predictions could produce is the strongest signal
    available that the training loop is not training on what it reports -- and it is a signal that
    was sitting in the logs of three failed runs before anyone checked it. Not an equality: with
    `early_stopping` sklearn's `loss_` is the last epoch on the training portion only, so this asks
    whether the number is of the right order rather than whether it matches.
    """
    probability = np.clip(model.predict_proba(matrix)[:, 1], 1e-12, 1.0 - 1e-12)
    observed = float(-np.mean(target*np.log(probability)
                              + (1 - target)*np.log(1 - probability)))
    reported = float(getattr(model, 'loss_', np.nan))
    if np.isfinite(reported) and not (reported/factor <= observed <= reported*factor):
        raise CompositionError(
            f'reported loss {reported:.6f} is not consistent with the log loss of the model\'s '
            f'own predictions, {observed:.6f}'
            )
    return dict(reported_loss=reported, observed_loss=observed)


class NeuralFom(FomCombiner.FomCombiner):
    """A calibrated P(correct) from an MLP over the same design matrix the tree is given.

    Everything about how the columns are assembled, calibrated and thresholded is inherited, which
    is the point: a network that built its features differently from the tree would be measuring
    the features and not the architecture.
    """

    def __init__(self, names=(), categorical=(), categories=None, groups=(), weights=(),
                 biases=(), centre=None, scale=None, median=None, calibrators=None, meta=None):
        super().__init__(names=names, categorical=categorical, categories=categories,
                         groups=groups, objective='pointwise')
        self.calibrators = calibrators or {}
        self.weights = [np.asarray(weight, dtype=np.float64) for weight in weights]
        self.biases = [np.asarray(bias, dtype=np.float64) for bias in biases]
        self.centre = None if centre is None else np.asarray(centre, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.median = None if median is None else np.asarray(median, dtype=np.float64)
        self.meta = meta or {}

    @classmethod
    def fit(cls, frames, groups=FomCombiner.DEFAULT_GROUPS, scalers=(), objective='pointwise',
            seed=12345, hidden=(64, 32), max_iter=80, learning_rate=1e-3, batch_size=1024,
            sample=None, expected_positive_rate=None, **ignored):
        """Fit on the labels over the assembled features, with both F-121 checks armed."""
        from sklearn.neural_network import MLPClassifier

        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        if not frames:
            raise ValueError('no frames to fit on')
        names, categorical = FomCombiner.feature_specification(groups, scalers)
        # The same construction the tree uses, so the two models see identical columns in an
        # identical encoding. `_UNSEEN_CODE` is 0 and real categories start at 1, which is what
        # makes an unseen extinction group land in its own bin rather than colliding with cP.
        categories = {name: {value: code + 1 for code, value in
                             enumerate(sorted(pd.unique(pd.concat(
                                 [frame[name] for frame in frames]).astype(str))))}
                      for name in categorical}
        model_shell = cls(names=names, categorical=categorical, categories=categories,
                          groups=groups)

        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        matrix = model_shell.design_matrix(frame)
        target = FomMetrics.as_bool(frame['is_correct']).astype(np.int32)

        rng = np.random.default_rng(seed)
        if sample and matrix.shape[0] > sample:
            keep = rng.choice(matrix.shape[0], size=sample, replace=False)
            matrix, target = matrix[keep], target[keep]
        composition = check_composition(matrix, target, expected_positive_rate)

        median = np.nanmedian(matrix, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        clean = FomCombiner._impute(matrix, median)
        centre = clean.mean(axis=0)
        scale = clean.std(axis=0)
        scale[scale < 1e-12] = 1.0

        # `early_stopping=False` deliberately. Sklearn's early stopping scores its held-out split
        # on *accuracy*, which a constant "incorrect" answer maximises at this base rate -- see
        # `check_not_constant`. With it off, `n_iter_no_change` and `tol` watch the training loss
        # instead, which is a criterion that means something on an imbalanced problem. The honest
        # stopping signal is the calibration split, which the caller already holds out.
        standardised = (clean - centre)/scale
        network = MLPClassifier(
            hidden_layer_sizes=tuple(hidden), activation='relu', solver='adam',
            learning_rate_init=learning_rate, batch_size=batch_size, max_iter=max_iter,
            random_state=seed, early_stopping=False, n_iter_no_change=10, tol=1e-5,
            )
        network.fit(standardised, target)
        loss_check = check_loss_is_possible(network, standardised, target)
        trained = check_not_constant(network.predict_proba(standardised)[:, 1], target)

        model_shell.weights = [np.asarray(w, dtype=np.float64) for w in network.coefs_]
        model_shell.biases = [np.asarray(b, dtype=np.float64) for b in network.intercepts_]
        model_shell.centre, model_shell.scale, model_shell.median = centre, scale, median
        model_shell.meta = dict(hidden=list(hidden), seed=int(seed), max_iter=int(max_iter),
                                learning_rate=float(learning_rate), n_features=len(names),
                                n_iterations=int(network.n_iter_),
                                composition=composition, loss_check=loss_check,
                                trained=trained)
        return model_shell

    def predict_batch(self, matrix):
        """`relu` between layers, logistic at the output. Three matmuls, no per-candidate Python."""
        activations = (FomCombiner._impute(matrix, self.median) - self.centre)/self.scale
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = activations@weight + bias
            if index < len(self.weights) - 1:
                np.maximum(activations, 0.0, out=activations)
        # Clipped before the exponential rather than after: a confident negative logit overflows
        # `exp(-x)` to inf, which gives the right answer (1/inf -> 0) through a RuntimeWarning, and
        # a warning that is always harmless is a warning nobody reads when it stops being.
        logit = np.clip(activations.ravel(), -500.0, 500.0)
        return 1.0/(1.0 + np.exp(-logit))

    def raw_score(self, frame):
        return self.predict_batch(self.design_matrix(frame))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {'centre': self.centre, 'scale': self.scale, 'median': self.median}
        for name, (thresholds, targets) in self.calibrators.items():
            arrays[f'calibrator_{name}__x'] = thresholds
            arrays[f'calibrator_{name}__y'] = targets
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f'weight_{index}'] = weight
            arrays[f'bias_{index}'] = bias
        np.savez_compressed(directory/'neural.npz', **arrays)
        with open(directory/'neural.json', 'w', encoding='utf-8') as handle:
            json.dump(dict(names=list(self.names), categorical=list(self.categorical),
                           categories={name: {str(k): int(v) for k, v in lookup.items()}
                                       for name, lookup in (self.categories or {}).items()},
                           groups=list(self.groups), n_layers=len(self.weights),
                           meta=self.meta), handle, indent=2, default=str)
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        arrays = np.load(directory/'neural.npz')
        with open(directory/'neural.json', encoding='utf-8') as handle:
            specification = json.load(handle)
        n_layers = int(specification['n_layers'])
        calibrators = {}
        for key in arrays.files:
            if key.startswith('calibrator_') and key.endswith('__x'):
                name = key[len('calibrator_'):-len('__x')]
                calibrators[name] = (arrays[key], arrays[f'calibrator_{name}__y'])
        return cls(names=specification['names'], categorical=specification['categorical'],
                   categories={name: {key: int(value) for key, value in lookup.items()}
                               for name, lookup in specification['categories'].items()},
                   groups=specification.get('groups', ()),
                   weights=[arrays[f'weight_{index}'] for index in range(n_layers)],
                   biases=[arrays[f'bias_{index}'] for index in range(n_layers)],
                   centre=arrays['centre'], scale=arrays['scale'], median=arrays['median'],
                   calibrators=calibrators, meta=specification.get('meta', {}))
