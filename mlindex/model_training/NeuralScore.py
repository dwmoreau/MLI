"""S14 -- the neural scoring network: a plain dense model that scores a candidate cell directly.

DWMM's design (`docs/fom_campaign2/handoffs/S14_neural_score.md`): ~50 inputs, almost none of
them classical merits -- the prior network's lattice distribution and per-lattice volume for the
pattern (block A), the twenty per-peak assignment posteriors for the candidate (block B), and the
candidate's own volume and Bravais lattice. It is NOT campaign 1's `NeuralFom`, which was an MLP
over the tree's own 108 columns and lost by 181 entries to 53; that module was deliberately left
on `fom` so this one could not become a retry of it (`CHERRY_PICK.md`).

What is reused from the tree, by direct assignment as `DistilledCombiner` does: `design_matrix`
(so the network reads exactly the columns `feature_specification` names and the same leakage check
runs), `fit_calibrators` and `score` (the same per-lattice isotonic, so the tree and the network
are compared as calibrated probabilities), and `score_columns`. What differs:

  * The Bravais lattice is a ONE-HOT block, never an ordinal code. Campaign 1's network arms
    consumed a 158-level categorical as an alphabetical ordinal and nobody recorded it.
  * Every input with a missing value in training gets a missingness indicator. A cubic candidate
    is scored on ten peaks, so `asg_p10`..`asg_p19` are NaN there by construction, and a network
    without the indicator conflates "cubic" with "no information".
  * Training is plain PyTorch, weighted binary cross entropy on `sampling_weight`, and inference
    is numpy matmuls from the saved arrays -- no torch, no keras, no session, per F-092.

**The three guards campaign 1 paid for (F-121)** are not optional and are written so that they
would have failed on the run that cost three 30-epoch fits: every batch's composition is checked
(one class or one lattice raises), the per-epoch composition is logged into the saved
specification, and the training loss the loop reports is compared with the loss the finished
model's own predictions imply -- a loss that is arithmetically impossible for the predictions is
the symptom that went unnoticed for two runs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomCombiner
from mlindex.model_training import FomMetrics
from mlindex.model_training.FomCombiner import _UNSEEN_CODE
from mlindex.model_training.FomCombiner import feature_specification
from mlindex.model_training.FomCombiner import fit_weights

MODEL_TYPE = 'neural_score'
DEFAULT_HIDDEN = (64, 32)


class CompositionError(RuntimeError):
    """A training batch or epoch that has lost the population it was drawn from."""


def check_composition(target, lattice_codes, where, minimum_lattices=2):
    """Raise unless both classes and at least `minimum_lattices` lattices are present.

    Campaign 1's balanced sampler decayed the training set to one lattice from epoch 2 and every
    loss read 0.000 because the model was predicting a constant. A row-count check passed that
    batch; this one would not have.
    """
    target = np.asarray(target)
    if target.size == 0:
        raise CompositionError(f'{where}: empty')
    positives = int(target.sum())
    if positives == 0 or positives == target.size:
        raise CompositionError(f'{where}: one class only ({positives} of {target.size} positive)')
    present = np.unique(np.asarray(lattice_codes))
    if present.size < minimum_lattices:
        raise CompositionError(f'{where}: {present.size} lattice(s) present, need '
                               f'{minimum_lattices}; the sampler has lost the pool')


def check_not_constant(score, target, minimum_auc=0.55):
    """A fitted score must vary and must beat a coin flip on its own training rows."""
    from sklearn.metrics import roc_auc_score

    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise CompositionError('the fitted score has non-finite values')
    if np.ptp(score) < 1e-9:
        raise CompositionError('the fitted score is constant')
    auc = float(roc_auc_score(np.asarray(target).astype(int), score))
    if auc < minimum_auc:
        raise CompositionError(f'training AUC {auc:.3f} is below {minimum_auc}; the model has '
                               f'not learned the label')
    return auc


def check_loss_is_possible(reported, implied, factor=3.0):
    """The loss the loop reported and the loss the model's predictions imply must agree."""
    if not (np.isfinite(reported) and np.isfinite(implied)) or implied <= 0:
        raise CompositionError(f'loss check: reported {reported}, implied {implied}')
    ratio = reported/implied
    if ratio > factor or ratio < 1.0/factor:
        raise CompositionError(
            f'reported training loss {reported:.4f} is not consistent with the loss the fitted '
            f'model\'s own predictions imply, {implied:.4f} (ratio {ratio:.2f}); the loop is '
            f'descending a quantity that is not the one being evaluated')
    return ratio


def weighted_log_loss(probability, target, weight):
    """Weighted binary cross entropy in nats, the quantity every check above compares."""
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    t = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    loss = -(t*np.log(p) + (1 - t)*np.log(1 - p))
    return float((w*loss).sum()/w.sum())


class NeuralScore:
    """A dense network over the S14 inputs, held as numpy arrays once fitted."""

    design_matrix = FomCombiner.FomCombiner.design_matrix
    categorical_indices = FomCombiner.FomCombiner.categorical_indices
    score_columns = FomCombiner.FomCombiner.score_columns
    fit_calibrators = FomCombiner.FomCombiner.fit_calibrators
    score = FomCombiner.FomCombiner.score

    def __init__(self, names=(), categorical=(), categories=None, groups=(), weights=(),
                 biases=(), centre=None, scale=None, indicator_columns=(), calibrators=None,
                 meta=None):
        self.names = tuple(names)
        self.categorical = tuple(categorical)
        self.categories = categories or {}
        self.groups = tuple(groups)
        self.objective = 'pointwise'
        self.weights = [np.asarray(weight, dtype=np.float64) for weight in weights]
        self.biases = [np.asarray(bias, dtype=np.float64) for bias in biases]
        self.centre = None if centre is None else np.asarray(centre, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.indicator_columns = tuple(int(index) for index in indicator_columns)
        self.calibrators = calibrators or {}
        self.meta = meta or {}

    # -- the input layer ------------------------------------------------------------------
    @property
    def numeric_indices(self):
        return [index for index, name in enumerate(self.names) if name not in self.categorical]

    def one_hot_width(self, name):
        return len(self.categories[name])

    def expand(self, matrix):
        """The network's input from `design_matrix`'s output: standardised numerics with NaN set
        to zero AFTER standardisation, one missingness indicator per training-time-NaN column, and
        a one-hot block per categorical (an unseen category is all zeros)."""
        numeric = matrix[:, self.numeric_indices].astype(np.float64, copy=True)
        missing = ~np.isfinite(numeric)
        standardised = (numeric - self.centre)/self.scale
        standardised[missing] = 0.0
        blocks = [standardised]
        if self.indicator_columns:
            blocks.append(missing[:, list(self.indicator_columns)].astype(np.float64))
        for name in self.categorical:
            index = self.names.index(name)
            codes = matrix[:, index].astype(np.int64)
            width = self.one_hot_width(name)
            block = np.zeros((matrix.shape[0], width), dtype=np.float64)
            valid = (codes >= 1) & (codes <= width)
            block[np.flatnonzero(valid), codes[valid] - 1] = 1.0
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    @property
    def input_width(self):
        return (len(self.numeric_indices) + len(self.indicator_columns)
                + sum(self.one_hot_width(name) for name in self.categorical))

    # -- fitting ---------------------------------------------------------------------------
    @classmethod
    def fit(cls, frames, groups, drop=(), seed=12345, weight_column='sampling_weight',
            hidden=DEFAULT_HIDDEN, epochs=30, batch_size=4096, learning_rate=1e-3,
            validation_fraction=0.1, patience=5, minimum_lattices=2, log=None,
            threads=None):
        """Fit on assembled frames. Weighted by `weight_column` (None fits unweighted).

        Early stopping watches the weighted log-loss on a validation split drawn BY ENTRY from the
        fit frames (never by row: one crystal under nine bundles is one draw). The composition of
        every batch and every epoch is checked and logged; the loss the loop reports is checked
        against the loss the finished model implies on the same rows.
        """
        import torch

        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        if not frames:
            raise ValueError('no frames to fit on')
        names, categorical = feature_specification(groups, drop=drop)
        categories = {name: {value: code + 1 for code, value in
                             enumerate(sorted(pd.unique(pd.concat(
                                 [frame[name] for frame in frames]).astype(str))))}
                      for name in categorical}
        model = cls(names=names, categorical=categorical, categories=categories, groups=groups)

        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        matrix = model.design_matrix(frame)
        target = FomMetrics.as_bool(frame['is_correct']).astype(np.float64)
        weights = (np.ones(frame.shape[0]) if weight_column is None
                   else fit_weights(frame, weight_column))
        lattice = frame['bravais_lattice'].to_numpy().astype(str)
        check_composition(target, lattice, 'the whole fit frame', minimum_lattices)

        numeric = matrix[:, model.numeric_indices]
        finite = np.isfinite(numeric)
        centre = np.array([numeric[finite[:, j], j].mean() if finite[:, j].any() else 0.0
                           for j in range(numeric.shape[1])])
        scale = np.array([numeric[finite[:, j], j].std() if finite[:, j].any() else 1.0
                          for j in range(numeric.shape[1])])
        scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
        model.centre, model.scale = centre, scale
        model.indicator_columns = tuple(int(j) for j in np.flatnonzero(~finite.all(axis=0)))
        inputs = model.expand(matrix)

        rng = np.random.default_rng(seed)
        entries = frame['entry_id'].to_numpy().astype(str)
        unique_entries = np.array(sorted(set(entries)))
        held = set(unique_entries[rng.permutation(unique_entries.size)
                                  < int(round(validation_fraction*unique_entries.size))])
        validation = np.array([entry in held for entry in entries])
        if validation.all() or not validation.any():
            validation = np.zeros(entries.size, dtype=bool)
        train_rows = np.flatnonzero(~validation)
        valid_rows = np.flatnonzero(validation)

        torch.manual_seed(seed)
        if threads:
            torch.set_num_threads(int(threads))
        layers = []
        width = inputs.shape[1]
        for size in hidden:
            layers += [torch.nn.Linear(width, int(size)), torch.nn.ELU()]
            width = int(size)
        layers.append(torch.nn.Linear(width, 1))
        network = torch.nn.Sequential(*layers).double()
        optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        x_all = torch.from_numpy(inputs)
        y_all = torch.from_numpy(target)
        w_all = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float64).copy())

        def implied_loss(rows):
            network.eval()
            with torch.no_grad():
                logits = network(x_all[rows]).squeeze(1).numpy()
            network.train()
            return weighted_log_loss(1.0/(1.0 + np.exp(-logits)), target[rows], weights[rows])

        lattice_code = pd.Series(lattice).map(categories['bravais_lattice']).to_numpy() \
            if 'bravais_lattice' in categories else np.zeros(entries.size, dtype=int)
        history, best, best_state, since_best = [], np.inf, None, 0
        for epoch in range(int(epochs)):
            order = rng.permutation(train_rows)
            running, seen = 0.0, 0.0
            epoch_target = np.zeros(0)
            epoch_lattices = set()
            per_lattice_positive = {}
            for start in range(0, order.size, int(batch_size)):
                rows = order[start:start + int(batch_size)]
                check_composition(target[rows], lattice_code[rows],
                                  f'epoch {epoch + 1} batch at row {start}', minimum_lattices)
                optimiser.zero_grad()
                logits = network(x_all[rows]).squeeze(1)
                loss = (criterion(logits, y_all[rows])*w_all[rows]).sum()/w_all[rows].sum()
                loss.backward()
                optimiser.step()
                running += float(loss.item())*float(weights[rows].sum())
                seen += float(weights[rows].sum())
                epoch_lattices.update(np.unique(lattice_code[rows]).tolist())
                for code, is_positive in zip(lattice_code[rows], target[rows]):
                    per_lattice_positive[int(code)] = per_lattice_positive.get(int(code), 0) \
                        + int(is_positive)
            epoch_target = target[order]
            check_composition(epoch_target, lattice_code[order], f'epoch {epoch + 1}',
                              minimum_lattices)
            train_loss = running/seen
            valid_loss = implied_loss(valid_rows) if valid_rows.size else float('nan')
            record = dict(epoch=epoch + 1, rows=int(order.size),
                          positives=int(epoch_target.sum()),
                          positive_fraction=float(epoch_target.mean()),
                          lattices_present=int(len(epoch_lattices)),
                          per_lattice_positives=json.dumps(
                              {str(k): v for k, v in sorted(per_lattice_positive.items())}),
                          train_loss=float(train_loss), validation_loss=float(valid_loss))
            history.append(record)
            if log is not None:
                log(f'    epoch {epoch + 1:3d} loss {train_loss:.4f} val {valid_loss:.4f} '
                    f'rows {order.size:,} pos {epoch_target.mean():.4f} '
                    f'lattices {len(epoch_lattices)}')
            watched = valid_loss if valid_rows.size else train_loss
            if watched < best - 1e-6:
                best, since_best = watched, 0
                best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            else:
                since_best += 1
                if since_best >= patience:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)

        # Arrays out, torch gone.
        model.weights = [layer.weight.detach().numpy().T.copy() for layer in network
                         if isinstance(layer, torch.nn.Linear)]
        model.biases = [layer.bias.detach().numpy().copy() for layer in network
                        if isinstance(layer, torch.nn.Linear)]

        # The guards: the reported loss against the one the finished model implies, on the same
        # rows and through the numpy path that will be served; and the score is not a constant.
        fitted = model.predict_batch(matrix)
        implied = weighted_log_loss(fitted[train_rows], target[train_rows], weights[train_rows])
        last_reported = history[-1]['train_loss']
        loss_ratio = check_loss_is_possible(last_reported, implied)
        auc = check_not_constant(fitted[train_rows], target[train_rows])
        torch_logits = network(x_all[:min(4096, inputs.shape[0])]).detach().squeeze(1).numpy()
        numpy_logits = _logit(fitted[:min(4096, inputs.shape[0])])
        if not np.allclose(torch_logits, numpy_logits, atol=1e-6, rtol=1e-6):
            raise CompositionError('the numpy forward pass does not reproduce the torch one')

        model.meta = dict(
            model_type=MODEL_TYPE, seed=int(seed), groups=list(groups), dropped=sorted(drop),
            weight_column=weight_column, weight_sum=float(weights.sum()),
            n_rows=int(frame.shape[0]), n_positive=int(target.sum()), n_features=len(names),
            input_width=int(inputs.shape[1]), hidden=[int(size) for size in hidden],
            epochs_requested=int(epochs), epochs_run=len(history),
            best_validation_loss=float(best) if valid_rows.size else None,
            n_validation_entries=len(held), batch_size=int(batch_size),
            learning_rate=float(learning_rate), composition=history,
            loss_check=dict(reported=float(last_reported), implied=float(implied),
                            ratio=float(loss_ratio)),
            train_auc=float(auc), indicator_columns=list(model.indicator_columns),
            indicator_names=[names[model.numeric_indices[j]] for j in model.indicator_columns],
            )
        return model

    # -- scoring ---------------------------------------------------------------------------
    def predict_batch(self, matrix, chunk=1_000_000):
        """Uncalibrated P(correct): ELU between layers, sigmoid at the output, numpy only.

        Chunked because `design_matrix` is float64 over the whole shard and a 15 M-row bundle
        times 85 inputs is 10 GB before the first matmul.
        """
        out = np.empty(matrix.shape[0], dtype=np.float64)
        for start in range(0, matrix.shape[0], int(chunk)):
            activations = self.expand(matrix[start:start + int(chunk)])
            for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                activations = activations@weight + bias
                if index < len(self.weights) - 1:
                    negative = activations < 0
                    activations[negative] = np.expm1(activations[negative])
            out[start:start + int(chunk)] = 1.0/(1.0 + np.exp(-activations.ravel()))
        return out

    def raw_score(self, frame, chunk=500_000):
        """Uncalibrated P(correct) per row, assembled and scored in row chunks.

        The design matrix is float64 over every named column, so a 15 M-row bundle at 55 inputs
        is 6.6 GB before the first matmul -- built whole, it drove a 16 GB laptop into swap and a
        six-minute pass took twenty (S14, 2026-09-05). Chunking the assembly, not only the
        matmul, is what keeps the peak at a few hundred megabytes.
        """
        out = np.empty(frame.shape[0], dtype=np.float64)
        for start in range(0, frame.shape[0], int(chunk)):
            block = frame.iloc[start:start + int(chunk)]
            out[start:start + int(chunk)] = self.predict_batch(self.design_matrix(block))
        return out

    # -- persistence -----------------------------------------------------------------------
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {'centre': self.centre, 'scale': self.scale,
                  'indicator_columns': np.asarray(self.indicator_columns, dtype=np.int64)}
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f'weight_{index}'] = weight
            arrays[f'bias_{index}'] = bias
        np.savez_compressed(directory/'neural.npz', **arrays)
        calibrators = {}
        for name, (thresholds, targets) in self.calibrators.items():
            calibrators[f'{name}__x'] = thresholds
            calibrators[f'{name}__y'] = targets
        np.savez_compressed(directory/'calibrators.npz', **calibrators)
        meta = dict(self.meta)
        meta['model_type'] = MODEL_TYPE
        meta['n_layers'] = len(self.weights)
        with open(directory/'specification.json', 'w', encoding='utf-8') as handle:
            json.dump(dict(names=list(self.names), categorical=list(self.categorical),
                           categories={name: {str(k): int(v) for k, v in lookup.items()}
                                       for name, lookup in self.categories.items()},
                           groups=list(self.groups), objective='pointwise', meta=meta),
                      handle, indent=2)
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory/'specification.json', encoding='utf-8') as handle:
            specification = json.load(handle)
        meta = specification.get('meta', {})
        if meta.get('model_type') != MODEL_TYPE:
            raise ValueError(f'{directory} is not a {MODEL_TYPE} model '
                             f'(model_type={meta.get("model_type")!r})')
        arrays = np.load(directory/'neural.npz')
        n_layers = int(meta['n_layers'])
        stored = np.load(directory/'calibrators.npz')
        names = {key.rsplit('__', 1)[0] for key in stored.files}
        calibrators = {name: (stored[f'{name}__x'], stored[f'{name}__y']) for name in names}
        return cls(names=specification['names'], categorical=specification['categorical'],
                   categories=specification['categories'], groups=specification['groups'],
                   weights=[arrays[f'weight_{index}'] for index in range(n_layers)],
                   biases=[arrays[f'bias_{index}'] for index in range(n_layers)],
                   centre=arrays['centre'], scale=arrays['scale'],
                   indicator_columns=arrays['indicator_columns'].tolist(),
                   calibrators=calibrators, meta=meta)


def _logit(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def chunked_score(model, frame, chunk=500_000):
    """`model.score(frame)` in row chunks, for any model with a `score` over a frame.

    The tree's `raw_score` (`FomCombiner.py`) builds its design matrix over the whole frame, and
    at 86 features on a 15 M-row bundle that is 10 GB of float64 on top of the frame itself. The
    per-lattice isotonic maps rows independently, so scoring a frame in pieces is exactly the
    same arithmetic with a bounded peak.
    """
    out = np.empty(frame.shape[0], dtype=np.float64)
    for start in range(0, frame.shape[0], int(chunk)):
        block = frame.iloc[start:start + int(chunk)]
        out[start:start + int(chunk)] = model.score(block)
    return out


def load_any(directory):
    """Whichever model type `specification.json` says lives here: the tree, or this network.

    The reduce stage globs a model directory and scores what it finds, which is how three
    slice-fitted trees were once reported as full-scale results (C2-F-141). Dispatching on the
    recorded type, and letting the caller assert `meta['n_rows']`, is the cheap half of not
    repeating that.
    """
    directory = Path(directory)
    with open(directory/'specification.json', encoding='utf-8') as handle:
        specification = json.load(handle)
    if specification.get('meta', {}).get('model_type') == MODEL_TYPE:
        return NeuralScore.load(directory)
    return FomCombiner.FomCombiner.load(directory)
