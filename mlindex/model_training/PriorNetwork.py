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

**Ported to campaign 2 by S14 (2026-09-05) from `fom@7c137c3`, with one change of substance.** A
saved model now records its `support` -- the Bravais lattices it was trained on -- and every
consumer-facing readout (`entry_tables`, `score_candidates`, `claimed_pair_readout`) is renormalised
over that set and NaN outside it. Campaign 1 trained on eleven lattices and kept a fourteen-class
head, so the three untrained cubic classes emitted log-probabilities near -19 and were consumed as
probabilities (F-117, point 4; C2 S14 handoff). The training path is otherwise the campaign-1 one.
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

# The lattices a model without a recorded `support` is assumed to have been trained on. Campaign 1
# excluded cubic (cP, cI, cF): cubic is indexed on ten peaks where the extraction window takes
# twenty, and the dense reference it was measured against excluded them too
# (`run_fom_prior.active_lattices`). The shipped checkpoint predates the key, so `load_prior` falls
# back to this and says so; a model trained by campaign 2's driver always writes the key.
DEFAULT_SUPPORT = tuple(code for code in BRAVAIS_LATTICES if not code.startswith('c'))

# The condition mix, specified by DWMM (2026-08-19): simulated error at x1 only, no dropout, no
# second phase, and 0/1/2 contaminants drawn at relative frequencies 1 : 0.5 : 0.25 -- so a
# two-contaminant pattern is four times rarer than a clean one.
#
# This deliberately departs from S02/S04's six evaluable bundles, which the first pass used. Those
# span error x2, six and ten interior dropout holes and a three-line second phase, and training
# across all of them was measured to cost about fourteen points of Bravais accuracy: your dense
# reference reached a 52.1% macro diagonal on its own data and 26.5% pooled under the S04 grid,
# which is the same collapse the prior network showed. The grid is a stress test, not the operating
# condition, and block A is a prior over what patterns normally look like.
CONDITION_BUNDLES = (
    dict(name='error1_cont0', multiplier=1.0, n_contaminants=0, n_dropout=0, second_phase=0,
         weight=1.0),
    dict(name='error1_cont1', multiplier=1.0, n_contaminants=1, n_dropout=0, second_phase=0,
         weight=0.5),
    dict(name='error1_cont2', multiplier=1.0, n_contaminants=2, n_dropout=0, second_phase=0,
         weight=0.25),
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
    'layers': [512, 256, 128],
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
    # 'marginal': train each class head on log sum_v P(v) P(c|v), the branch-averaged
    # posterior. 'joint': train on log P(v*) + log P(c* | v*), the likelihood of the
    # observed (volume, lattice) pair. The second supervises the cell of the 128 x C
    # table a candidate is actually scored against; the first only ever sees its shadow,
    # which leaves P(c | v) free to be arbitrary wherever it does not move the marginal.
    'loss_mode': 'marginal',
    'model_type': 'prior',
    # The volume branch carries the factorisation, so it is a primary term here rather than the
    # 0.2-weighted auxiliary it is in the generators.
    'volume_loss_weight': 1.0,
    # Width, in branches, of the soft target for the volume head. 0 keeps the one-hot label and
    # plain sparse cross entropy, which is ordinal-blind: with branches ordered by volume, missing
    # by one is penalised exactly as much as missing by ninety. Because the branches discretise a
    # continuous quantity, spreading the target over neighbours is more faithful to the truth
    # rather than less -- unlike a cost matrix over the *unordered* lattice classes, which would
    # distort a posterior block C reads as a probability.
    'volume_target_width': 0.0,
    'target_loss_weights': {'bravais': 1.0, 'system': 1.0, 'centring': 0.5,
                            'n_free': 0.5, 'high_symmetry': 0.5},
    }


def _log_normalise(values, axis):
    """log_softmax in numpy, stable, without routing float64 through the keras backend."""
    peak = values.max(axis=axis, keepdims=True)
    return values - (peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True)))


def _logsumexp_nan(values, axis):
    """logsumexp treating NaN cells as absent (log 0). All-NaN slices come back NaN."""
    finite = np.where(np.isnan(values), -np.inf, values)
    peak = finite.max(axis=axis, keepdims=True)
    safe_peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = safe_peak + np.log(np.exp(finite - safe_peak).sum(axis=axis, keepdims=True))
    out = np.where(np.isfinite(peak), out, np.nan)
    return np.squeeze(out, axis=axis)


def _entropy_nan(log_probability, axis):
    """Shannon entropy in nats of a log-probability vector with NaN meaning "not in support"."""
    probability = np.where(np.isnan(log_probability), 0.0, np.exp(log_probability))
    terms = np.where(probability > 0, probability*np.where(np.isnan(log_probability), 0.0,
                                                           log_probability), 0.0)
    return -terms.sum(axis=axis)


def claimed_pair_readout(joint, log_branch_volumes, rows, volumes, class_indices):
    """The per-candidate block-A readout from precomputed entry tables, no keras needed.

    `joint` is `entry_tables(...)['joint']` for the entries, `rows` indexes each candidate's
    entry into it, `volumes` and `class_indices` are the candidate's own claimed pair. Returns
    `prior_joint` (log P at the claimed pair; NaN where the claimed lattice is untrained),
    `prior_joint_margin` (the same minus the table's own maximum, so a respectable absolute value
    that is only the model being uncertain reads as such) and `prior_in_support`.
    """
    rows = np.asarray(rows, dtype=int)
    class_indices = np.asarray(class_indices, dtype=int)
    claimed = np.log(np.asarray(volumes, dtype=np.float64))
    nearest = np.abs(
        np.asarray(log_branch_volumes)[np.newaxis] - claimed[:, np.newaxis]
        ).argmin(axis=1)
    at_pair = joint[rows, nearest, class_indices]
    table_max = np.nanmax(joint.reshape(joint.shape[0], -1), axis=1)[rows]
    in_support = ~np.isnan(at_pair)
    return {
        'prior_joint': at_pair,
        'prior_joint_margin': at_pair - table_max,
        'prior_in_support': in_support.astype(np.int8),
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


def held_out_identifiers(manifest_path, split_column=None):
    """Identifiers the frozen split assigns to `fom-dev` or `fom-test`.

    Campaign 1's manifest names its split column `split`; campaign 2's
    (`mlindex/data/generated_datasets/fom_split_c2.parquet`) names it `fom_split`. Both are read;
    a manifest with neither raises rather than returning an empty set, because an empty set here
    means training on the benchmark with no symptom.

    The single most damaging mistake available in this task is training on these. The generated
    datasets and the manifest key on the same identifier -- verified, all 500 mP manifest entries
    are present in `dataset_mP.parquet` and 200 of them are held out -- so the overlap is real and
    silent. Anything the manifest does not mention was never in the benchmark and is free to train
    on.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    present = set(pq.ParquetFile(manifest_path).schema_arrow.names)
    if split_column is None:
        candidates = [name for name in ('fom_split', 'split') if name in present]
        if not candidates:
            raise KeyError(f'{manifest_path} carries neither `fom_split` nor `split`; a manifest '
                           f'without a split column cannot guard the training pool')
        split_column = candidates[0]
    manifest = pd.read_parquet(manifest_path, columns=['identifier', split_column])
    return set(manifest.loc[manifest[split_column] != 'fom-train', 'identifier'])


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
        # Weighted, not uniform: the conditions have different frequencies of occurrence and the
        # training mix should reflect that rather than over-representing the rare, hard ones.
        weights = np.array([bundle.get('weight', 1.0) for bundle in bundles], dtype=float)
        bundle_index = rng.choice(len(bundles), size=n_rows, p=weights/weights.sum())
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



def soft_branch_targets(branch_labels, n_volumes, width):
    """A Gaussian target over branches, centred on the branch the true volume falls in.

    In branch index rather than in log volume, which is the same thing: the grid is spaced
    geometrically, so log V is linear in the index and a fixed width in one is a fixed ratio in the
    other. `width` 0 returns the one-hot label unchanged.
    """
    branch_labels = np.asarray(branch_labels, dtype=np.int64)
    if width <= 0:
        target = np.zeros((branch_labels.size, n_volumes), dtype=np.float32)
        target[np.arange(branch_labels.size), branch_labels] = 1.0
        return target
    grid = np.arange(n_volumes, dtype=np.float64)[np.newaxis]
    weight = np.exp(-0.5*((grid - branch_labels[:, np.newaxis])/float(width))**2)
    return (weight/weight.sum(axis=1, keepdims=True)).astype(np.float32)


def _joint_loss(n_classes):
    """-log P(c* | v*): the conditional evaluated at the branch the true volume falls in.

    `y_true` packs (class, branch) because a keras loss only sees one label tensor per output, and
    this loss needs both -- which is the whole point, since it is the *pair* being supervised.
    Added to the volume head's own -log P(v*), the two terms are the joint negative log likelihood
    of the observed (volume, lattice) pair.
    """
    import keras

    def loss(y_true, y_pred):
        # `take_along_axis`, not `y_pred[rows, branch]`. Python-style advanced indexing with an
        # `arange` over the batch works eagerly and silently gathers the wrong elements inside the
        # compiled training step, where the batch dimension is not resolved the same way. That cost
        # a full 30-epoch run: `fit` reported a loss of 0.32 while the same weights scored 31.75
        # when evaluated, and the model had descended a quantity that did not exist.
        y_true = keras.ops.cast(y_true, 'int32')
        target = y_true[:, 0:1]
        branch = y_true[:, 1]
        at_branch = keras.ops.take_along_axis(
            y_pred, branch[:, None, None], axis=1,
            )[:, 0, :]
        return -keras.ops.take_along_axis(at_branch, target, axis=1)[:, 0]

    return loss


def joint_targets(class_codes, branch_labels):
    """Pack (class, branch) into the single label tensor a keras loss receives."""
    return np.stack([np.asarray(class_codes, dtype=np.int64),
                     np.asarray(branch_labels, dtype=np.int64)], axis=1)


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
        # A fresh model's support is whatever the caller says it is trained on; the driver sets it
        # from the lattices in its pool. Absent, every class is assumed in scope, which is right
        # for a model about to be trained on all of them and wrong for a loaded checkpoint -- so
        # `load_prior` handles the absent key itself rather than inheriting this default.
        support = self.model_params.get('support')
        self.set_support(tuple(support) if support else tuple(BRAVAIS_LATTICES))
        self.support_defaulted = not support

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
                if self.model_params['loss_mode'] == 'joint':
                    # The output *is* the per-branch conditional, so the cell a candidate is
                    # scored against is the thing being trained and the thing being served.
                    outputs[target] = keras.layers.Activation(
                        'log_softmax', name=target,
                        )(logits)
                else:
                    outputs[target] = marginalize(name=target)([branch, logits])
        return outputs

    def compile_model(self):
        """Cross entropy on the branch and on every class head, all `from_logits=True`.

        The class heads emit log probabilities rather than raw logits, which the same loss accepts
        unchanged: softmax of a log probability vector returns that vector's probabilities.
        """
        import keras

        joint = self.model_params['loss_mode'] == 'joint'
        soft = self.model_params.get('volume_target_width', 0.0) > 0
        losses = {'volume_branch': (
            keras.losses.CategoricalCrossentropy(from_logits=True) if soft
            else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            )}
        weights = {'volume_branch': self.model_params['volume_loss_weight']}
        metrics = {'volume_branch': [] if soft else ['sparse_categorical_accuracy']}
        for target, n_classes in self.targets.items():
            losses[target] = (
                _joint_loss(n_classes) if joint
                else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                )
            weights[target] = self.model_params['target_loss_weights'].get(target, 1.0)
            # Kept even in joint mode: the metric is what makes a head going flat visible
            # during training rather than three runs later (F-120).
            metrics[target] = [] if joint else ['sparse_categorical_accuracy']
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
    # Support: which classes a consumer may read as a probability
    # -------------------------------------------------------------------------------------
    def set_support(self, support):
        """Record the Bravais lattices this model is trained on, in `BRAVAIS_LATTICES` order."""
        unknown = [code for code in support if code not in BRAVAIS_LATTICES]
        if unknown or not len(support):
            raise ValueError(f'support must be a non-empty subset of {BRAVAIS_LATTICES}, '
                             f'got {tuple(support)}')
        self.support = tuple(code for code in BRAVAIS_LATTICES if code in set(support))
        self.model_params['support'] = list(self.support)

    def support_mask(self):
        """Boolean over `BRAVAIS_LATTICES`: True where the class was trained."""
        return np.array([code in set(self.support) for code in BRAVAIS_LATTICES], dtype=bool)

    def masked_bravais_joint(self, q2, batch_size=None):
        """`log P(volume branch, lattice)` renormalised over the support, NaN outside it.

        This is the only table a consumer should read. The raw head puts mass of order e^-19 on an
        untrained class, which is a statement about the initialiser and not about the pattern;
        renormalising over the trained classes removes it, and NaN where the class is untrained
        makes a consumer that forgot to check fail loudly rather than read a number.

        Returns `(joint, branch_lp)`: the (n, n_volumes, n_classes) table and the (n, n_volumes)
        branch marginal under the same renormalisation.
        """
        tables = self.joint_log_probabilities(q2, targets=('bravais',), batch_size=batch_size)
        joint = np.array(tables['bravais'], dtype=np.float64, copy=True)
        mask = self.support_mask()
        inside = joint[:, :, mask]
        flat = inside.reshape(inside.shape[0], -1)
        peak = flat.max(axis=1)
        norm = peak + np.log(np.exp(flat - peak[:, np.newaxis]).sum(axis=1))
        joint = joint - norm[:, np.newaxis, np.newaxis]
        joint[:, :, ~mask] = np.nan
        branch_lp = _logsumexp_nan(joint, axis=2)
        return joint, branch_lp

    def predict_log_marginal(self, q2, target, batch_size=None):
        """One head's log marginal over its classes, as the trained graph emits it."""
        raw = self.model.predict(self.scale_peaks(q2), batch_size=batch_size or 512, verbose=0)
        return _log_normalise(np.asarray(raw[target], dtype=np.float64), axis=1)

    def entry_tables(self, q2, batch_size=None):
        """Everything block A says about a peak list, as the per-entry inputs S14 consumes.

        All of it is constant within an entry. `bravais_lp` / `bravais_p` are the lattice marginal
        over the support (NaN outside), `logv` is `E[log V | lattice]` read from the joint -- the
        per-lattice volume DWMM's design asks for, and the reading F-117 measured to recover 37 %
        of the volume error against the plain marginal -- and `joint` is the masked table itself,
        kept so a per-candidate readout can be made later without another forward pass.
        """
        joint, branch_lp = self.masked_bravais_joint(q2, batch_size=batch_size)
        bravais_lp = _logsumexp_nan(joint, axis=1)                      # (n, n_classes)
        log_volumes = np.log(self.branch_volumes())                     # (n_volumes,)
        conditional = np.exp(joint - bravais_lp[:, np.newaxis, :])     # P(v | c), NaN outside
        logv = (conditional*log_volumes[np.newaxis, :, np.newaxis]).sum(axis=1)
        logv[:, ~self.support_mask()] = np.nan
        # The lattice-marginal volume, E[log V] under the support-renormalised branch marginal:
        # DWMM's "predict a volume" as one number per pattern (decision 2026-09-05).
        logv_marginal = (np.exp(branch_lp)*log_volumes[np.newaxis, :]).sum(axis=1)
        # The expected number of free cell parameters from the `n_free` head, renormalised over
        # the levels the support can reach: level 1 is cubic and only cubic, so a model trained
        # without cubic has never seen it and it is masked exactly as the cubic lattices are.
        levels = np.asarray(N_FREE_LEVELS, dtype=np.float64)
        reachable = np.array([any(N_FREE_OF[LATTICE_SYSTEM_OF[code]] == level
                                  for code in self.support) for level in N_FREE_LEVELS])
        n_free_lp = np.asarray(self.predict_log_marginal(q2, 'n_free', batch_size=batch_size),
                               dtype=np.float64)
        n_free_lp[:, ~reachable] = np.nan
        n_free_lp = n_free_lp - _logsumexp_nan(n_free_lp, axis=1)[:, np.newaxis]
        dof_expected = (np.where(np.isnan(n_free_lp), 0.0, np.exp(n_free_lp))
                        *levels[np.newaxis, :]).sum(axis=1)
        return {
            'logv_marginal': logv_marginal,
            'n_free_lp': n_free_lp,
            'dof_expected': dof_expected,
            'dof_levels_reachable': reachable,
            'joint': joint,
            'branch_lp': branch_lp,
            'bravais_lp': bravais_lp,
            'bravais_p': np.exp(bravais_lp),
            'logv': logv,
            'bravais_entropy': _entropy_nan(bravais_lp, axis=1),
            'branch_entropy': _entropy_nan(branch_lp, axis=1),
            'in_support': self.support_mask(),
            'log_branch_volumes': log_volumes,
            }

    # -------------------------------------------------------------------------------------
    # The joint, which is what a candidate is actually scored against
    # -------------------------------------------------------------------------------------
    def branch_volumes(self):
        """The direct-space volume each branch stands for.

        `get_branch_labels` matches on `(1/V / q2_obs_scale**2)**(2/3) / volume_normalization`, so
        inverting that turns a branch index back into the volume it represents.
        """
        volumes = np.asarray(self.extraction_layer.volumes).ravel().astype(float)
        reciprocal = (
            (volumes*self.extraction_layer.volume_normalization)**1.5*self.q2_obs_scale**2
            )
        return 1.0/reciprocal

    def _conditional_model(self):
        """A view onto `P(class | branch)`, which the trained graph computes and then discards.

        `MarginalizeOverBranches` collapses the branch axis before the output leaves the model, so
        the per-branch conditional is not reachable from `predict`. It is the same weights either
        way -- this exposes them rather than recomputing anything.
        """
        import keras

        if getattr(self, '_conditional_cache', None) is None:
            self._conditional_cache = keras.Model(
                self.model.input,
                {'volume_branch': self.model.get_layer('volume_branch').output,
                 **{target: self.model.get_layer(f'{target}_dense').output
                    for target in self.targets}},
                )
        return self._conditional_cache

    def joint_log_probabilities(self, q2, targets=None, batch_size=None):
        """Every head's joint table from **one** forward pass, as `{target: (n, n_v, n_c)}`.

        The graph computes all five heads together and `joint_log_probability` then throws four of
        them away, which is fine for one head and is five redundant passes for a caller that wants
        the lot -- block C reads all five at each candidate's claimed pair. Same arithmetic, same
        results; it just stops discarding what has already been computed.

        The branch log-probability is returned under `volume_branch`, since it is the (n, n_v)
        marginal every table in the dict shares and the caller would otherwise recover it by
        summing one of them back up.
        """
        raw = self._conditional_model().predict(
            self.scale_peaks(q2), batch_size=batch_size or 32, verbose=0,
            )
        # Normalised in numpy, not through keras.ops: MPS has no float64, and these are log
        # probabilities that get exponentiated and multiplied, so the precision is worth having.
        branch = _log_normalise(np.asarray(raw['volume_branch'], dtype=np.float64), axis=1)
        tables = {'volume_branch': branch}
        for target in (targets if targets is not None else self.targets):
            conditional = _log_normalise(np.asarray(raw[target], dtype=np.float64), axis=2)
            tables[target] = branch[:, :, np.newaxis] + conditional
        return tables

    def joint_log_probability(self, q2, target='bravais', batch_size=None):
        """log P(volume branch, class) for each pattern -- the (n, n_volumes, n_classes) table.

        This is what block C reads: a candidate arrives claiming a lattice *and* a volume, and the
        honest question is the mass at that cell. The two marginals are projections of this table
        and lose the correlation between the axes -- measured at 37% of the volume error (F-117),
        because the volume marginal silently inherits the volume scale of whichever lattice the
        model happens to believe.
        """
        return self.joint_log_probabilities(
            q2, targets=(target,), batch_size=batch_size,
            )[target]

    def score_candidates(self, q2, volumes, class_indices, target='bravais', batch_size=None):
        """log P(V, class) at each candidate's own claimed pair -- one number per candidate.

        The claimed lattice is used, never the true one: a candidate that claims a lattice whose
        volume prior it violates should score badly, and that is the whole point.
        """
        if target == 'bravais':
            # Renormalised over the support and NaN outside it: a claim on an untrained lattice
            # reads as "no prior", never as a probability of e^-19.
            joint, _ = self.masked_bravais_joint(q2, batch_size=batch_size)
        else:
            joint = self.joint_log_probability(q2, target=target, batch_size=batch_size)
        grid = np.log(self.branch_volumes())
        nearest = np.abs(
            grid[np.newaxis] - np.log(np.asarray(volumes, dtype=float))[:, np.newaxis]
            ).argmin(axis=1)
        rows = np.arange(joint.shape[0])
        return joint[rows, nearest, np.asarray(class_indices, dtype=int)]

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
        model_params = dict(saved['model_params'])
        support_defaulted = not model_params.get('support')
        if support_defaulted:
            # The campaign-1 checkpoint predates the key. Its training list is known (eleven,
            # cubic excluded) and is recorded here rather than guessed at by every consumer.
            model_params['support'] = list(DEFAULT_SUPPORT)
            print(f'PriorNetwork.load_prior: {directory} records no support; assuming the '
                  f'campaign-1 training set {DEFAULT_SUPPORT} (cubic untrained)')
        model = cls(
            data_params=saved['data_params'], model_params=model_params,
            save_to=os.path.dirname(os.path.normpath(directory)) or directory, seed=seed,
            )
        model.support_defaulted = support_defaulted
        model.q2_obs_scale = float(grid['q2_obs_scale'])
        model.build_model(data=None)
        model.model.load_weights(os.path.join(directory, 'prior.weights.h5'))
        model.extraction_layer.volume_normalization = float(grid['volume_normalization'])
        return model
