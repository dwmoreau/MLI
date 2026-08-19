"""Block A of the neural figure of merit: volume scale and Bravais lattice from the peak list.

One model for **every** Bravais lattice, unlike the shipped generators, which are one model per
split group. It reads a peak list and nothing else, and emits two things the figure of merit needs
as its prior term (PLAN section 3): a distribution over the unit cell's volume scale, and a
distribution over its lattice, the second conditioned on the first.

This is S12 (the volume prior) and S13 (the Bravais prior) built as one network with the lattice
head conditioned on the volume branch, which is PLAN section 4's assumption **A5** -- "is
P(BL | volume scale) x P(volume) the right decomposition" -- promoted from a footnote in both
handoffs to the thing actually being tested. `head_mode` selects between the three arms A5 needs:
the conditional factorisation, the marginals, and an unfactorised joint softmax.

**The gate here is calibration, not accuracy**, and that is the whole point of the task. A prior
does not have to be sharp; it has to be honest. Supplemental Fig. S18 measured a dense Bravais
classifier collapsing from 93.6% on idealised peak lists to 46.2% on realistic ones, and a
46%-accurate model that says so is usable while a confidently wrong one is actively harmful. So
nothing here is trained or selected on accuracy, and `run_fom_prior.py` reports ECE, reliability
and mutual information against the base-rate baseline before it reports anything else.

Three things about this file that are load-bearing.

**It subclasses `IntegralFilter` rather than reimplementing it.** The whole harness -- the
extraction grid, the attention block, the branch labels, the training loop -- already exists and is
tested. Only `build_model`, the head builder, `compile_model` and `get_branch_labels` are
overridden, and they are overridden because a global model has no single lattice system to convert
a unit cell through and no xnn regression head to rescale.

**The extraction layer has no learned weights.** `Networks.ExtractionLayer`'s `volumes`, `filters`
and `sigma` are all `trainable=False` and are assigned from data statistics -- percentiles of the
reciprocal volume distribution and a quantile of projected peak separations. So there is no
pre-trained representation to transfer here, and the grid is refitted on this task's own pooled
distribution. That is not a shortcut; it is the only thing the construction admits.

**The training pool overlaps the evaluation benchmark and the guard is not optional.**
`mlindex/data/generated_datasets/dataset_{BL}.parquet` and the frozen split manifest key on the
same `identifier`, and 40% of the manifest's entries are `fom-dev` or `fom-test`. `load_prior_frame`
drops them. The datasets also carry their own `train` boolean, which is a *different* split from a
different task and must never be used for this one.
"""
import numpy as np
import os

from mlindex.model_training.IntegralFilter import IntegralFilter
from mlindex.model_training.FomMetrics import BRAVAIS_LATTICES
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import add_second_phase
from mlindex.utilities.ErrorAdder import select_peaks_with_dropout


# The five targets, all of them deterministic functions of the true Bravais lattice and therefore
# free -- no extra labelling, no extra data. DWMM's requirement is the coarse end of this list
# ("getting a close BL ... we just need to know if it is low or high symmetry"), and `n_free` is
# included because the number of free cell parameters is the axis F-096 and F-099 measured the
# degrees-of-freedom mechanism along, so it is the one that connects to the rest of the project.
LATTICE_SYSTEM_OF = {
    'cP': 'cubic', 'cI': 'cubic', 'cF': 'cubic',
    'tP': 'tetragonal', 'tI': 'tetragonal',
    'hP': 'hexagonal', 'hR': 'rhombohedral',
    'oP': 'orthorhombic', 'oC': 'orthorhombic', 'oF': 'orthorhombic', 'oI': 'orthorhombic',
    'mP': 'monoclinic', 'mC': 'monoclinic',
    'aP': 'triclinic',
    }
LATTICE_SYSTEMS = ('cubic', 'tetragonal', 'hexagonal', 'rhombohedral',
                   'orthorhombic', 'monoclinic', 'triclinic')
CENTRINGS = ('P', 'I', 'F', 'C', 'R')
# Free cell parameters per lattice system, which is the ordinal F-099's per-lattice table is sorted
# by: cubic 1, the three two-parameter systems 2, orthorhombic 3, monoclinic 4, triclinic 6.
N_FREE_OF = {'cubic': 1, 'tetragonal': 2, 'hexagonal': 2, 'rhombohedral': 2,
             'orthorhombic': 3, 'monoclinic': 4, 'triclinic': 6}
N_FREE_LEVELS = (1, 2, 3, 4, 6)
# "High symmetry" is cut at two free parameters, so it is a coarsening of `n_free` rather than an
# independent judgement. F-069 measured that the dominant failure is symmetry *lowering*, so this
# is the axis the prior is wanted on.
HIGH_SYMMETRY_MAX_FREE = 2

TARGETS = ('bravais', 'system', 'centring', 'n_free', 'high_symmetry')
TARGET_CLASSES = {
    'bravais': len(BRAVAIS_LATTICES),
    'system': len(LATTICE_SYSTEMS),
    'centring': len(CENTRINGS),
    'n_free': len(N_FREE_LEVELS),
    'high_symmetry': 2,
    }

# The six evaluable condition bundles, read off `entries.parquet` rather than off their names, so a
# training draw covers exactly the conditions the model is evaluated under. C0 (`error0_cont0`) is
# excluded here as it is everywhere else: zero error makes M20 diverge (F-054), and an idealised
# peak list is the thing S12's pitfall list warns against training on.
CONDITION_BUNDLES = (
    dict(name='error1_cont0', multiplier=1.0, n_contaminants=0, n_dropout=0, second_phase=0),
    dict(name='error2_cont0', multiplier=2.0, n_contaminants=0, n_dropout=0, second_phase=0),
    dict(name='error1_cont2', multiplier=1.0, n_contaminants=2, n_dropout=0, second_phase=0),
    dict(name='error1_cont1_drop6', multiplier=1.0, n_contaminants=1, n_dropout=6, second_phase=0),
    dict(name='error1_cont1_drop10', multiplier=1.0, n_contaminants=1, n_dropout=10, second_phase=0),
    dict(name='error1_cont0_phase3', multiplier=1.0, n_contaminants=0, n_dropout=0, second_phase=3),
    )
CONTAMINANT_BIAS = 1.0
SECOND_PHASE_BIAS = 2.0

# Every block in S11 trains and evaluates on broadening tag `1` and the `*_1` model set, which is
# what S02 froze (decision 2026-08-11) and what produced the published 450/599. `dataset_*.parquet`
# also carries tag `sa`; it is deliberately out of scope, so rebuild row R12 -- that the whole
# benchmark is one instrument -- is not addressed by anything here.
BROADENING_TAG = '1'

# There is one split group here, by construction: the whole point of block A is a single
# model spanning every Bravais lattice, so `IntegralFilter`'s per-split-group directory
# layout collapses to one name.
GLOBAL_SPLIT_GROUP = 'global'

PRIOR_MODEL_DEFAULTS = {
    'peak_length': 20,
    'extraction_peak_length': 12,
    'n_volumes': 128,
    'n_filters': 1024,
    'layers': [256, 128],
    'l1_regularization': 0.0,
    'learning_rate': 0.0005,
    'd_model': 128,
    'n_heads': 4,
    'epochs': 10,
    'batch_size': 128,
    'volume_lower_percentile': 0.001,
    'volume_upper_percentile': 0.999,
    'volume_spacing_blend': 0.5,
    'head_mode': 'conditional',
    'model_type': 'prior',
    # The volume branch carries the factorisation, so it is a primary term here rather than the
    # 0.2-weighted auxiliary it is in the generators.
    'volume_loss_weight': 1.0,
    'target_loss_weights': {'bravais': 1.0, 'system': 1.0, 'centring': 0.5,
                            'n_free': 0.5, 'high_symmetry': 0.5},
    }


def target_codes(bravais_lattice):
    """The five class labels for an array of Bravais lattice codes, as a dict of int arrays."""
    bravais_lattice = np.asarray(bravais_lattice, dtype=object)
    systems = np.array([LATTICE_SYSTEM_OF[code] for code in bravais_lattice], dtype=object)
    n_free = np.array([N_FREE_OF[system] for system in systems], dtype=np.int64)
    return {
        'bravais': np.array([BRAVAIS_LATTICES.index(code) for code in bravais_lattice]),
        'system': np.array([LATTICE_SYSTEMS.index(system) for system in systems]),
        'centring': np.array([CENTRINGS.index(code[1]) for code in bravais_lattice]),
        'n_free': np.array([N_FREE_LEVELS.index(value) for value in n_free]),
        'high_symmetry': (n_free <= HIGH_SYMMETRY_MAX_FREE).astype(np.int64),
        }


def held_out_identifiers(manifest_path):
    """Identifiers the frozen split assigns to `fom-dev` or `fom-test`.

    The single most damaging mistake available in this task is training on these. The generated
    datasets and the manifest key on the same identifier -- verified, all 500 mP manifest entries
    are present in `dataset_mP.parquet` and 200 of them are held out -- so the overlap is real and
    silent. Anything the manifest does not mention was never in the benchmark and is free to train
    on.
    """
    import pandas as pd
    manifest = pd.read_parquet(manifest_path, columns=['identifier', 'split'])
    return set(manifest.loc[manifest['split'] != 'fom-train', 'identifier'])


def load_prior_frame(datasets_dir, manifest_path, n_peaks=20, bravais_lattices=None,
                     limit_per_lattice=None, broadening=BROADENING_TAG, seed=12345):
    """The training pool: one row per source structure, with its full peak list and its labels.

    Reads `dataset_{BL}.parquet` per lattice, drops every identifier the frozen split holds out,
    and keeps only structures with at least `n_peaks` usable lines -- an entry that cannot fill the
    nominal window is not one the indexer would have run on.

    `limit_per_lattice` subsamples for development. It is applied *after* the split filter and with
    a fixed seed, so a small run is a subset of a large one rather than a different experiment.
    """
    import pandas as pd

    held_out = held_out_identifiers(manifest_path)
    codes = tuple(bravais_lattices) if bravais_lattices else BRAVAIS_LATTICES
    rng = np.random.default_rng(seed)

    frames = []
    for code in codes:
        path = os.path.join(datasets_dir, f'dataset_{code}.parquet')
        frame = pd.read_parquet(path, columns=[
            'identifier', 'bravais_lattice', 'reindexed_volume', f'q2_{broadening}'
            ])
        frame = frame.rename(columns={f'q2_{broadening}': 'q2_full',
                                      'reindexed_volume': 'volume'})
        frame = frame.loc[~frame['identifier'].isin(held_out)]
        lengths = frame['q2_full'].map(lambda values: int(np.sum(np.asarray(values) > 0)))
        frame = frame.loc[lengths >= n_peaks]
        if limit_per_lattice is not None and len(frame) > limit_per_lattice:
            keep = rng.choice(len(frame), size=limit_per_lattice, replace=False)
            frame = frame.iloc[np.sort(keep)]
        frames.append(frame.reset_index(drop=True))

    pool = pd.concat(frames, ignore_index=True)
    for name, values in target_codes(pool['bravais_lattice'].to_numpy()).items():
        pool[f'target_{name}'] = values
    return pool


def draw_peak_lists(frame, rng, n_peaks=20, bundles=CONDITION_BUNDLES, bundle_index=None):
    """One realistic peak list per row: window with interior dropout, error, contaminants, phase.

    The mechanisms are S02/S04's own (`mlindex/utilities/ErrorAdder.py`), applied in the order the
    benchmark applies them, so a training pattern and a benchmark pattern are the same object drawn
    twice rather than two different constructions.

    A fresh draw per epoch is what makes the class balancing honest: oversampling a thin lattice
    then gives it new noise, dropout and contaminant realisations rather than the same 714 patterns
    repeated. Pass `bundle_index` to pin the condition, which the grid fit and the per-condition
    evaluation both need.
    """
    n_rows = len(frame)
    if bundle_index is None:
        bundle_index = rng.integers(0, len(bundles), size=n_rows)
    else:
        bundle_index = np.full(n_rows, int(bundle_index))

    full = [np.asarray(values, dtype=float) for values in frame['q2_full']]
    q2 = np.zeros((n_rows, n_peaks), dtype=float)
    for row in range(n_rows):
        window = select_peaks_with_dropout(
            full[row], n_peaks, bundles[bundle_index[row]]['n_dropout'], rng
            )
        # A thin entry can come back short of the nominal window; pad by repeating the last line so
        # the fixed-length input still holds. `load_prior_frame` filters these out, so this is a
        # guard rather than a path anything normally takes.
        if window.size < n_peaks:
            window = np.concatenate([window, np.full(n_peaks - window.size, window[-1])])
        q2[row] = window[:n_peaks]

    # Error is added to the whole batch at once because add_q2_error is vectorised over rows, and
    # it is added before the contaminants for the same reason the benchmark does it in that order:
    # a contaminant is a line from somewhere else, not a mismeasured line of this pattern.
    #
    # Both contaminant mechanisms can refuse a row -- a dense pattern with no gap wide enough, or a
    # partner with no placeable line inside the observed range. The row is then left with its clean
    # window rather than dropped, and the refusals are counted and returned, because "this entry
    # could not be contaminated" and "this entry was not selected for contamination" are different
    # statements and silently merging them would bias the condition mix.
    n_refused = 0
    for index in range(len(bundles)):
        rows = np.flatnonzero(bundle_index == index)
        if rows.size == 0:
            continue
        bundle = bundles[index]
        q2[rows] = add_q2_error(q2[rows], None, bundle['multiplier'], rng)
        if bundle['n_contaminants']:
            for row in rows:
                try:
                    q2[row] = add_contaminants(
                        q2[row][np.newaxis].copy(), None, bundle['n_contaminants'], rng,
                        low_angle_bias=CONTAMINANT_BIAS, max_attempts=200,
                        )[0]
                except ContaminantPlacementError:
                    n_refused += 1
        if bundle['second_phase']:
            partners = rng.choice(n_rows, size=rows.size, replace=True)
            for offset, row in enumerate(rows):
                try:
                    q2[row] = add_second_phase(
                        q2[row][np.newaxis].copy(), None, full[partners[offset]],
                        bundle['second_phase'], rng, low_angle_bias=SECOND_PHASE_BIAS,
                        )[0]
                except ContaminantPlacementError:
                    n_refused += 1
    return q2, bundle_index, n_refused


def balanced_indices(codes, n_classes, rng, per_class):
    """Row indices giving every class `per_class` draws.

    Sampling with replacement wherever a class is thinner than the quota, which for the Bravais
    lattices it very much is: mP has 202 271 structures and cF has 714, a ratio of 283 to 1. What
    makes the repetition acceptable is that `draw_peak_lists` re-draws the conditions each epoch,
    so a repeated structure arrives as a different pattern.

    The cost has to be stated wherever this is used: **balancing removes the base rate**, and the
    base rate is part of the prior the combiner wants (F-086 found the extinction group is S08's
    dominant feature). Block C therefore takes the likelihood from this model and is given the base
    rate back as its own feature.
    """
    codes = np.asarray(codes)
    picks = []
    for label in range(n_classes):
        rows = np.flatnonzero(codes == label)
        if rows.size == 0:
            continue
        picks.append(rng.choice(rows, size=per_class, replace=rows.size < per_class))
    order = np.concatenate(picks)
    rng.shuffle(order)
    return order


class MarginalizeOverBranches:
    """Placeholder replaced at import time by the keras subclass; see `_marginalize_layer`."""


def _marginalize_layer():
    """`log sum_v P(v) P(c | v)` as a keras layer, built lazily so importing this module is cheap.

    Doing the marginalisation inside the graph rather than after it is what makes the conditional
    head trainable against a plain cross entropy: the output *is* the marginal log probability, and
    softmax of a log probability vector is that vector's probability, so the standard sparse
    categorical loss applies unchanged with `from_logits=True`.
    """
    import keras

    class _Marginalize(keras.layers.Layer):
        def call(self, inputs):
            branch_logits, class_logits = inputs
            log_p_branch = keras.ops.log_softmax(branch_logits, axis=1)
            log_p_class_given_branch = keras.ops.log_softmax(class_logits, axis=2)
            joint = keras.ops.expand_dims(log_p_branch, axis=2) + log_p_class_given_branch
            return keras.ops.logsumexp(joint, axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[1][0], input_shape[1][2])

    return _Marginalize


class PriorNetwork(IntegralFilter):
    """One network, every Bravais lattice: P(volume branch) and P(lattice | volume branch).

    Constructed with `split_group='global'` and no `hkl_ref`, because neither means anything for a
    model that spans all fourteen lattices. `unit_cell_length` is zero for the same reason -- there
    is no regression head here, so there is no xnn to rescale and `MetricVolumeRescale` is not used.
    """

    def __init__(self, data_params, model_params, save_to, seed=12345):
        data_params = dict(data_params)
        data_params.setdefault('lattice_system', 'global')
        data_params.setdefault('unit_cell_length', 0)
        data_params.setdefault('unit_cell_indices', [])
        data_params.setdefault('n_peaks', PRIOR_MODEL_DEFAULTS['peak_length'])
        os.makedirs(save_to, exist_ok=True)
        model_params = dict(model_params)
        for key, value in PRIOR_MODEL_DEFAULTS.items():
            model_params.setdefault(key, value)
        super().__init__(
            GLOBAL_SPLIT_GROUP, data_params, model_params, save_to, seed, hkl_ref=None,
            )
        self.targets = dict(TARGET_CLASSES)

    # -------------------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------------------
    def build_model(self, data=None):
        """Fit the global extraction grid and assemble the heads.

        `data`, when given, is a frame carrying a `q2_window` column of **realistic** peak lists and
        a `volume` column. Realistic rather than idealised on purpose: the grid's filter positions
        and its sigma are fitted from the peak positions and separations it is handed, so fitting
        them on noiseless lines would set a width the model never sees again (S12's first pitfall).
        """
        import keras
        from mlindex.model_training.Networks import ExtractionLayer

        keras.utils.set_random_seed(self.seed)
        if data is not None:
            q2_obs = np.stack(data['q2_window'].to_numpy())
            q2_obs = q2_obs[:, :self.model_params['extraction_peak_length']]
            self.q2_obs_scale = float(q2_obs.std())
            # 1/V rather than a per-lattice reciprocal cell conversion: the two agree to 4e-15 on
            # triclinic and monoclinic and 5e-10 on cubic, and a global model has no single lattice
            # system to convert through.
            reciprocal_volume = 1.0/np.asarray(data['volume'].to_numpy(), dtype=float)
            self.extraction_layer = ExtractionLayer(
                self.model_params, q2_obs, None, reciprocal_volume, self.q2_obs_scale,
                name='extraction_layer',
                )
        else:
            assert hasattr(self, 'q2_obs_scale'), (
                'build_model(data=None) needs q2_obs_scale to already be set; load() does that.'
                )
            self.extraction_layer = ExtractionLayer(
                self.model_params, None, None, None, self.q2_obs_scale, name='extraction_layer',
                )

        inputs = keras.Input(
            shape=(self.model_params['peak_length'],), name='q2_obs_scaled', dtype='float32',
            )
        self.model = keras.Model(inputs, self.model_builder_prior(inputs))
        self.compile_model()

    def _trunk(self, inputs):
        import keras
        from mlindex.model_training.Networks import IntraVolume_MultiHeadAttention

        metric = self.extraction_layer(
            inputs[:, :self.model_params['extraction_peak_length']]
            )
        attended = IntraVolume_MultiHeadAttention(
            d_model=self.model_params['d_model'], n_heads=self.model_params['n_heads'],
            )(metric)
        x = keras.layers.UnitNormalization(axis=2)(attended)
        for index, width in enumerate(self.model_params['layers']):
            x = keras.layers.Dense(
                width, activation=keras.activations.elu, name=f'prior_dense_{index}',
                use_bias=False,
                kernel_regularizer=keras.regularizers.L1(l1=self.model_params['l1_regularization']),
                kernel_initializer=keras.initializers.HeUniform,
                )(x)
        return x

    def model_builder_prior(self, inputs):
        """The two heads, in whichever of A5's three arms `head_mode` selects.

        `conditional` is the factorisation the brief describes and the default: a branch logit per
        volume, a class logit per (volume, class), and the reported class distribution is the
        marginal over branches. `marginal` pools the branch axis away before the class head, so the
        class prediction cannot use the volume scale at all -- the control that says whether the
        conditioning earns anything. `joint` replaces the factorisation with one softmax over the
        flattened (volume, class) grid, which is the unfactorised alternative A5 names.
        """
        import keras

        x = self._trunk(inputs)
        n_volumes = self.model_params['n_volumes']
        head_mode = self.model_params['head_mode']
        marginalize = _marginalize_layer()

        branch = keras.layers.Dense(1, activation='linear', name='volume_dense')(x)
        branch = keras.layers.Reshape((n_volumes,), name='volume_branch')(branch)

        outputs = {'volume_branch': branch}
        for target, n_classes in self.targets.items():
            if head_mode == 'marginal':
                pooled = keras.layers.GlobalAveragePooling1D(name=f'{target}_pool')(x)
                logits = keras.layers.Dense(
                    n_classes, activation='linear', name=f'{target}_dense',
                    )(pooled)
                outputs[target] = keras.layers.Activation(
                    'log_softmax', name=target,
                    )(logits)
            elif head_mode == 'joint':
                logits = keras.layers.Dense(
                    n_classes, activation='linear', name=f'{target}_dense',
                    )(x)
                flat = keras.layers.Reshape((n_volumes*n_classes,))(logits)
                flat = keras.layers.Activation('log_softmax')(flat)
                grid = keras.layers.Reshape((n_volumes, n_classes))(flat)
                outputs[target] = keras.layers.Lambda(
                    lambda values: keras.ops.logsumexp(values, axis=1),
                    output_shape=(n_classes,), name=target,
                    )(grid)
            else:
                logits = keras.layers.Dense(
                    n_classes, activation='linear', name=f'{target}_dense',
                    )(x)
                outputs[target] = marginalize(name=target)([branch, logits])
        return outputs

    def compile_model(self):
        """Cross entropy on the branch and on every class head, all `from_logits=True`.

        The class heads emit log probabilities rather than raw logits, which the same loss accepts
        unchanged: softmax of a log probability vector returns that vector's probabilities.
        """
        import keras

        losses = {'volume_branch': keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
        weights = {'volume_branch': self.model_params['volume_loss_weight']}
        metrics = {'volume_branch': ['sparse_categorical_accuracy']}
        for target in self.targets:
            losses[target] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            weights[target] = self.model_params['target_loss_weights'].get(target, 1.0)
            metrics[target] = ['sparse_categorical_accuracy']
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.model_params['learning_rate']),
            loss=losses, loss_weights=weights, metrics=metrics, run_eagerly=False,
            )

    # -------------------------------------------------------------------------------------
    # Labels and inference
    # -------------------------------------------------------------------------------------
    def get_branch_labels(self, data):
        """Index of the volume branch matching each row's true volume.

        Overridden because `IntegralFilter`'s version routes the unit cell through a per lattice
        system reciprocal conversion, and this model has no single lattice system. The reciprocal
        volume is 1/V either way.
        """
        volume = np.asarray(data['volume'].to_numpy(), dtype=float)
        return self.extraction_layer.get_branch_labels(1.0/volume)

    def scale_peaks(self, q2):
        """Peak lists on the grid's own scale. Every entry point goes through this, once."""
        return np.asarray(q2, dtype='float32')/self.q2_obs_scale

    def predict_distributions(self, q2, batch_size=512):
        """P(volume branch) and the marginal class probabilities, as plain numpy."""
        import keras

        raw = self.model.predict(self.scale_peaks(q2), batch_size=batch_size, verbose=0)
        out = {'volume_branch': np.asarray(
            keras.ops.convert_to_numpy(keras.ops.softmax(raw['volume_branch'], axis=1))
            )}
        for target in self.targets:
            out[target] = np.exp(np.asarray(raw[target], dtype=float))
        return out

    # -------------------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------------------
    def save_prior(self, directory=None):
        """Weights plus the three arrays and one scalar the graph cannot rebuild for itself.

        Weights rather than a whole-model `.keras` save because `ExtractionLayer` takes the training
        distribution through its constructor and so is not serializable -- which is also why the
        shipped generators are served from an ONNX export rather than a checkpoint. Implementing
        `get_config` on it would mean editing a production module for a research task, and rebuild
        + `load_weights` reproduces the graph exactly without that.

        Deliberately *not* `IntegralFilter.save`, which writes the quantized ONNX pair the
        generators are served from. Export is a deployment question belonging to block C and S14,
        and fixing a serving interface here would fix it before the model has been shown to be
        worth serving.
        """
        import json

        directory = directory or self.save_to_split_group
        os.makedirs(directory, exist_ok=True)
        self.model.save_weights(os.path.join(directory, 'prior.weights.h5'))
        np.savez(
            os.path.join(directory, 'extraction_grid.npz'),
            q2_obs_scale=np.asarray(self.q2_obs_scale),
            volumes=np.asarray(self.extraction_layer.volumes),
            filters=np.asarray(self.extraction_layer.filters),
            sigma=np.asarray(self.extraction_layer.sigma),
            volume_normalization=np.asarray(self.extraction_layer.volume_normalization),
            )
        with open(os.path.join(directory, 'model_params.json'), 'w', encoding='utf-8') as handle:
            json.dump({'model_params': self.model_params, 'data_params': self.data_params,
                       'seed': self.seed}, handle, indent=2, default=str)
        return directory

    @classmethod
    def load_prior(cls, directory, seed=12345):
        """Rebuild the graph from `model_params.json` and restore the weights.

        `volume_normalization` is restored explicitly: it is a plain attribute rather than a weight,
        it is only computed on the fitting path, and `branch_volumes` needs it to turn the branch
        softmax back into a distribution over volume.
        """
        import json

        with open(os.path.join(directory, 'model_params.json'), encoding='utf-8') as handle:
            saved = json.load(handle)
        grid = np.load(os.path.join(directory, 'extraction_grid.npz'))
        model = cls(
            data_params=saved['data_params'], model_params=saved['model_params'],
            save_to=os.path.dirname(os.path.normpath(directory)) or directory, seed=seed,
            )
        model.q2_obs_scale = float(grid['q2_obs_scale'])
        model.build_model(data=None)
        model.model.load_weights(os.path.join(directory, 'prior.weights.h5'))
        model.extraction_layer.volume_normalization = float(grid['volume_normalization'])
        return model
