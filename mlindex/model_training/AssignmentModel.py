"""Block B of the neural figure of merit: which observed peaks a candidate cell indexes correctly.

Given a peak list and one candidate unit cell, this asks for **one probability per observed peak**
-- that the Miller index the indexer assigned to that peak is the right one. Block C consumes the
summary; PLAN section 4's assumptions **A6** ("a learned per-peak model beats the analytic
estimator") and **A7** ("the analytic estimator is well calibrated at all") turn on it, and
measuring the analytic forms discharges **S01-C**, blocked on S04 since 2026-08-10.

**One model per Bravais lattice**, unlike blocks A and C, because the reference line list and the
extinction groups are per lattice: the softmax is over that lattice's own `hkl_ref`, which is 1000
classes for mP/mC/hR, 750 for oP/hP, 500 for aP/tP/tI and 100 for the cubic lattices.

Four things about this file that are load-bearing.

**It subclasses `IntegralFilter` and changes almost nothing.** `build_calibration_model`
(`IntegralFilter.py:400`) and `model_builder_calibration` (`:526`) already take
`(q2_obs_scaled, xnn)` and emit a per-peak softmax over the reference lines, through pairwise
differences and the `epsilon/(|pds| + epsilon)` soft-match kernel. That is the shipped convention
and it is reused as it stands. What this class replaces is where the *candidate cell* and the
*labels* come from: `IntegralFilter.train_calibration` hardwires "the top-1 cell my own regression
head predicted" and a `data['train']` column, and block B has neither.

**The label space is the shipped `hkl_ref`, and its last row is the "unindexed" class.**
`mlindex/models/{system}_1/data/hkl_ref_{BL}.npy` is `Wrapper.setup_hkl`'s frozen output -- sorted
by mean calculated q2 over the training cells, truncated, with `[0, 0, 0]` appended last. A peak
whose true reflection is not in the list, and every contaminant, lands on that sentinel. Loading
the shipped file rather than re-deriving the sort matters: it is the list `Candidates` and
`FomBenchmark.assign_lines` index into at inference, so a locally re-sorted copy would put the
model's classes out of register with the pipeline's.

**"The same reflection" means equal under `get_hkl_matrix`.** Two Miller indices that map to the
same row of the lattice system's design matrix have the same calculated q2 for *every* cell of
that system, so they are indistinguishable from peak positions alone -- which is the only evidence
this project is allowed (PROTOCOL section 3 rule 3). That is the repo's own convention, the one
`Wrapper.setup_hkl` builds its labels with, and it is used here for both the class index and the
correctness label so the two cannot disagree.

**The training pool overlaps the evaluation benchmark and the guard is not optional.** The
generated datasets and the frozen split manifest key on the same `identifier`, and 40% of the
manifest is `fom-dev` or `fom-test` (F-101). `load_assignment_frame` drops them, reusing
`PriorNetwork.held_out_identifiers`. The datasets' own `train` boolean is a different task's split
and is never projected.
"""
import numpy as np
import os

from mlindex.model_training.IntegralFilter import IntegralFilter
from mlindex.model_training.PriorNetwork import CONDITION_BUNDLES
from mlindex.model_training.PriorNetwork import CONTAMINANT_BIAS
from mlindex.model_training.PriorNetwork import LATTICE_SYSTEM_OF
from mlindex.model_training.PriorNetwork import held_out_identifiers
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import perturb_xnn
from mlindex.utilities.UnitCellTools import get_hkl_matrix
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell


BROADENING_TAG = '1'

# Which components of the full six-vector xnn a lattice system actually uses. These are the same
# lists `Wrapper` sets (Wrapper.py:74-95) and the same order `get_hkl_matrix` emits its columns in,
# so `xnn_full[:, UNIT_CELL_INDICES[system]]` is the partial form the models take.
UNIT_CELL_INDICES = {
    'cubic': [0],
    'tetragonal': [0, 2],
    'hexagonal': [0, 2],
    'rhombohedral': [0, 3],
    'orthorhombic': [0, 1, 2],
    'monoclinic': [0, 1, 2, 4],
    'triclinic': [0, 1, 2, 3, 4, 5],
    }

# Distances in xnn used to manufacture wrong candidates around a true cell. `perturb_xnn`
# normalises each perturbation to exactly the requested L2 distance, so this is a ladder from
# "the refiner would walk back to truth" to "a different cell entirely", and every rung is
# labelled so the batch composition can be asserted rather than trusted.
PERTURBATION_LADDER = (
    ('near', 0.002),
    ('mid', 0.01),
    ('far', 0.05),
    )

ASSIGNMENT_MODEL_DEFAULTS = {
    'peak_length': 20,
    'calibration_params': {
        'epsilon_pds': 0.1,
        'layers': 3,
        'l1_regularization': 0.0,
        'learning_rate': 1e-3,
        'epochs': 1,
        'batch_size': 64,
        'n_heads': 5,
        'early_stopping_patience': 5,
        },
    }

# The forward pass materialises (batch, n_peaks, hkl_ref_length) at least twice -- the pairwise
# differences and their transform -- before every Dense layer of the same width. Budget against
# that rather than against a fixed batch, which is how a hardcoded 512 once asked for 6.4 GB in
# block A and turned an evaluation into fifteen hours.
PREDICT_BYTES_BUDGET = 512*1024*1024


def lattice_system_of(bravais_lattice):
    return LATTICE_SYSTEM_OF[bravais_lattice]


def hkl_reference(bravais_lattice, models_directory=None):
    """The lattice's frozen reference list: (hkl_ref_length, 3), `[0, 0, 0]` last.

    Deliberately the *unfiltered* per-lattice list rather than
    `FomBenchmark.hkl_ref_for(..., spacegroup)`, which narrows to one extinction group. A
    per-lattice model is not told the extinction group, so its class space is the whole list and
    the absences are something it has to read out of the pattern. That is a real difference from
    the shipped per-split-group generators and it is recorded rather than hidden.
    """
    from mlindex.model_training.FomBenchmark import _hkl_ref_path

    path = _hkl_ref_path(lattice_system_of(bravais_lattice), bravais_lattice, models_directory)
    hkl_ref = np.load(path)
    assert np.all(hkl_ref[-1] == 0), (
        f'{path} does not end in the [0, 0, 0] sentinel, so the unindexed class is not where '
        f'Wrapper.setup_hkl and MITemplates believe it is'
        )
    return hkl_ref


def canonical_hkl(hkl, lattice_system):
    """Miller indices reduced to what a peak position can distinguish, as an (..., k) integer array.

    `get_hkl_matrix` maps hkl to the design row whose dot product with xnn is q2, so two indices
    with equal rows have equal calculated q2 for *every* cell of the system. Equality of these
    rows is therefore the operational meaning of "the same reflection" here.

    **Returned as int64, and that is a correctness requirement rather than a tidiness one.** The
    design row is built from products of integers -- h^2, k^2, l^2, hl and so on -- so it is exact
    in float, but `hl` for h = -1, l = 0 evaluates to **negative zero**, which compares equal to
    +0.0 and has *different bytes*. Any lookup keyed on the byte representation therefore misses
    it. That is not hypothetical: it sent one real reflection in twenty-two to the "unindexed"
    sentinel, which is the network's training target and the reachability ceiling both, until it
    was caught by an impossible stratum -- peaks marked unreachable and simultaneously assigned
    correctly. Casting to int64 maps -0 to 0 and makes the representation canonical in fact.
    """
    design = get_hkl_matrix(np.asarray(hkl, dtype=float), lattice_system)
    rounded = np.rint(design)
    assert np.allclose(design, rounded, atol=1e-9), (
        'the hkl design matrix should be integer-valued for integer Miller indices'
        )
    return rounded.astype(np.int64)


def hkl_class_index(hkl, hkl_ref, lattice_system):
    """Class index into `hkl_ref` for each Miller index, with the sentinel for anything absent.

    Vectorised replacement for `Wrapper.setup_hkl`'s per-entry, per-peak `np.argwhere` loop
    (Wrapper.py:508-520), which is O(entries x peaks) Python calls and takes minutes on a pool
    this size. Same answer -- a test asserts it against `setup_hkl`'s own construction.

    `hkl` is (..., 3); returns an int array of the leading shape.
    """
    hkl = np.asarray(hkl, dtype=float)
    reference = canonical_hkl(hkl_ref, lattice_system)
    lookup = {row.tobytes(): index for index, row in enumerate(np.ascontiguousarray(reference))}
    # The sentinel is the last row, which is [0, 0, 0] -- so an unmatched peak and a contaminant
    # land on the same class, which is what they are: lines this reference list cannot index.
    sentinel = len(hkl_ref) - 1
    flat = np.ascontiguousarray(canonical_hkl(hkl.reshape(-1, 3), lattice_system))
    codes = np.fromiter(
        (lookup.get(row.tobytes(), sentinel) for row in flat), dtype=np.int64, count=len(flat),
        )
    return codes.reshape(hkl.shape[:-1])


def assignment_labels(hkl_assigned, hkl_true, lattice_system):
    """Per-peak truth: did the candidate assign this peak its correct Miller index?

    Both sides are reduced by `canonical_hkl` in the *candidate's* lattice system. For a candidate
    of the truth's own lattice this is exactly "the same reflection". For a candidate of a
    different lattice the label is essentially always False, which is correct -- a wrong cell does
    not index a peak correctly -- but the two lists are then in different bases, so a coincidental
    match is possible in principle and the drivers report how often it happens.

    Contaminants carry `(0, 0, 0)` as their true index (SCHEMA.md line 91), so they can only be
    called correct by a candidate that assigned them the sentinel, which no assignment does: the
    sentinel's calculated q2 is zero. They are therefore always False, which is the truth -- a
    line from another phase has no correct Miller index in this cell.
    """
    assigned = canonical_hkl(hkl_assigned, lattice_system)
    truth = canonical_hkl(hkl_true, lattice_system)
    return np.all(assigned == truth, axis=-1)


def load_assignment_frame(datasets_dir, manifest_path, bravais_lattice, n_peaks=20,
                          limit=None, broadening=BROADENING_TAG, seed=12345):
    """The training pool for one lattice: a source structure per row, with per-peak truth.

    `PriorNetwork.load_prior_frame` with the projection widened to the per-peak Miller indices and
    the true cell, because block B needs both and block A needed neither. The split filter and the
    "never read the datasets' own `train` column" rule are unchanged and are the point.
    """
    import pandas as pd

    held_out = held_out_identifiers(manifest_path)
    rng = np.random.default_rng(seed)
    path = os.path.join(datasets_dir, f'dataset_{bravais_lattice}.parquet')
    frame = pd.read_parquet(path, columns=[
        'identifier', 'bravais_lattice', 'reindexed_volume', 'reindexed_xnn',
        f'q2_{broadening}', f'reindexed_h_{broadening}', f'reindexed_k_{broadening}',
        f'reindexed_l_{broadening}',
        ])
    frame = frame.rename(columns={
        f'q2_{broadening}': 'q2_full', 'reindexed_volume': 'volume', 'reindexed_xnn': 'xnn_full',
        })
    frame = frame.loc[~frame['identifier'].isin(held_out)]

    # The peak list and the three Miller-index columns are parallel and equal length per row; the
    # window is the first n_peaks of them, and an entry that cannot fill it is not one the indexer
    # would have run on.
    lengths = frame['q2_full'].map(lambda values: int(np.sum(np.asarray(values) > 0)))
    frame = frame.loc[lengths >= n_peaks]
    if limit is not None and len(frame) > limit:
        keep = rng.choice(len(frame), size=limit, replace=False)
        frame = frame.iloc[np.sort(keep)]
    frame = frame.reset_index(drop=True)

    hkl_full = [
        np.stack([
            np.asarray(row[f'reindexed_h_{broadening}'], dtype=float),
            np.asarray(row[f'reindexed_k_{broadening}'], dtype=float),
            np.asarray(row[f'reindexed_l_{broadening}'], dtype=float),
            ], axis=1)
        for _, row in frame.iterrows()
        ]
    for index, hkl in enumerate(hkl_full):
        assert len(hkl) == len(frame['q2_full'].iloc[index]), (
            f'row {index}: {len(hkl)} Miller indices against '
            f'{len(frame["q2_full"].iloc[index])} peaks -- the columns are not parallel'
            )
    frame['hkl_full'] = hkl_full
    frame = frame.drop(columns=[
        f'reindexed_h_{broadening}', f'reindexed_k_{broadening}', f'reindexed_l_{broadening}',
        ])
    return frame


def partial_xnn(xnn_full, lattice_system):
    """The lattice system's free xnn components, from the datasets' full six-vector."""
    return np.asarray(xnn_full, dtype=float)[..., UNIT_CELL_INDICES[lattice_system]]


def draw_peak_lists_with_hkl(frame, rng, lattice_system, n_peaks=20, bundles=CONDITION_BUNDLES,
                             bundle_index=None, check=True):
    """One realistic peak list per row, with the true Miller indices still attached to it.

    Block A's `draw_peak_lists` passes `hkl=None` to every mechanism because it only needed the
    positions. Block B's labels live on the peaks, so the indices have to survive the same
    re-sorting, insertion and truncation the positions do -- which `add_q2_error` and
    `add_contaminants` will do, but only if they are handed the array.

    DWMM's conditions for this block: error x1, no dropout, no second phase, contaminants 0/1/2 at
    relative frequency 1 : 0.5 : 0.25. That is `PriorNetwork.CONDITION_BUNDLES` exactly, so it is
    imported rather than restated.

    `check` re-derives q2 from the true cell at the returned indices and requires it to agree with
    the returned peaks to within the error that was drawn. **This runs on every draw, not only in
    the tests**: a silent misalignment between a peak and its label would not fail anywhere
    downstream, it would just quietly train and score the wrong thing.

    Returns (q2, hkl, bundle_index, n_refused).
    """
    n_rows = len(frame)
    if bundle_index is None:
        weights = np.array([bundle.get('weight', 1.0) for bundle in bundles], dtype=float)
        bundle_index = rng.choice(len(bundles), size=n_rows, p=weights/weights.sum())
    else:
        bundle_index = np.full(n_rows, int(bundle_index))

    q2 = np.zeros((n_rows, n_peaks), dtype=float)
    hkl = np.zeros((n_rows, n_peaks, 3), dtype=float)
    for row in range(n_rows):
        full_q2 = np.asarray(frame['q2_full'].iloc[row], dtype=float)
        full_hkl = np.asarray(frame['hkl_full'].iloc[row], dtype=float)
        positive = full_q2 > 0
        q2[row] = full_q2[positive][:n_peaks]
        hkl[row] = full_hkl[positive][:n_peaks]
    clean = q2.copy()

    n_refused = 0
    for index in range(len(bundles)):
        rows = np.flatnonzero(bundle_index == index)
        if rows.size == 0:
            continue
        bundle = bundles[index]
        assert not bundle['n_dropout'] and not bundle['second_phase'], (
            'block B runs error x1 with contaminants only; a bundle carrying dropout or a second '
            'phase needs its hkl bookkeeping written first'
            )
        q2[rows], hkl[rows] = add_q2_error(q2[rows], hkl[rows], bundle['multiplier'], rng)
        if bundle['n_contaminants']:
            for row in rows:
                try:
                    contaminated, contaminated_hkl = add_contaminants(
                        q2[row][np.newaxis].copy(), hkl[row][np.newaxis].copy(),
                        bundle['n_contaminants'], rng, low_angle_bias=CONTAMINANT_BIAS,
                        max_attempts=200,
                        )
                except ContaminantPlacementError:
                    n_refused += 1
                    continue
                q2[row], hkl[row] = contaminated[0], contaminated_hkl[0]

    if check:
        check_alignment(frame, q2, hkl, clean, lattice_system)
    return q2, hkl, bundle_index, n_refused


def check_alignment(frame, q2, hkl, clean, lattice_system, tolerance=8.0):
    """Assert every non-contaminant peak still carries the index it was generated with.

    The test is not "did the arrays come back the same shape" -- they always do -- but whether the
    Miller index sitting beside a peak still predicts that peak's position through the *true* cell.
    A permutation that lost track of one row shows up immediately and nowhere else.

    `tolerance` is in units of the drawn error scale, generously wide because the point is to catch
    a permutation, not to re-measure sigma(q2).
    """
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info

    xnn = partial_xnn(np.stack(frame['xnn_full'].to_numpy()), lattice_system)
    q2_true = np.sum(canonical_hkl(hkl, lattice_system)*xnn[:, np.newaxis, :], axis=2)
    real = np.any(hkl != 0, axis=2)
    if not real.any():
        return
    params = get_peak_generation_info()['q2_error_params']
    scale = params[0] + q2*params[1]
    residual = np.abs(q2 - q2_true)/scale
    worst = residual[real].max()
    assert worst < tolerance, (
        f'peak and Miller index have come apart: worst residual {worst:.1f} error scales over '
        f'{int(real.sum())} real peaks. add_q2_error and add_contaminants re-sort hkl alongside '
        f'q2 only when they are given it, and the failure is silent everywhere downstream'
        )
    # `clean` is kept as the pre-error window so a caller can see the draw actually did something;
    # a bundle that changed nothing at all would be a different bug.
    assert clean.shape == q2.shape


def perturbed_candidates(xnn_true, lattice_system, rng, ladder=PERTURBATION_LADDER,
                         minimum_unit_cell=2.0, maximum_unit_cell=250.0):
    """Wrong cells around a true one, at a labelled ladder of xnn distances.

    `ErrorAdder.perturb_xnn` is the repo's own mechanism and is reused, but note it returns
    *partial unit cells*, not xnn -- so the result is converted back. One candidate per rung per
    structure, and the rung name travels with it so a batch's composition can be asserted.

    Returns (xnn, rung_names), the first (n_rows*len(ladder), n_components).
    """
    xnn_true = np.atleast_2d(np.asarray(xnn_true, dtype=float))
    distances = [distance for _, distance in ladder]
    names, rows = [], []
    for index in range(len(xnn_true)):
        unit_cells = perturb_xnn(
            xnn_true[index], 1, distances, minimum_unit_cell, maximum_unit_cell,
            lattice_system, rng,
            )
        rows.append(get_xnn_from_unit_cell(
            unit_cells, partial_unit_cell=True, lattice_system=lattice_system,
            ))
        names.extend(name for name, _ in ladder)
    return np.concatenate(rows, axis=0), np.array(names)


def predict_batch_size(n_peaks, hkl_ref_length, budget=PREDICT_BYTES_BUDGET):
    """Rows per forward pass, sized from the tensor the model actually materialises."""
    per_sample = 2*4*n_peaks*hkl_ref_length
    return int(max(8, min(512, budget//max(per_sample, 1))))


class AssignmentModel(IntegralFilter):
    """One lattice's per-peak Miller-index model: (q2_obs, xnn) -> P(hkl) per observed peak.

    Constructed against a Bravais lattice rather than a split group, so `hkl_ref` is the lattice's
    whole reference list and the extinction group is not an input. There is no regression head and
    no extraction layer here -- `IntegralFilter.build_calibration_model` needs only `q2_obs_scale`
    and `hkl_ref` -- so `build_model`, `model_builder_metric` and `MetricVolumeRescale` are unused.
    """

    def __init__(self, bravais_lattice, model_params, save_to, q2_obs_scale, seed=12345,
                 models_directory=None, data_params=None):
        lattice_system = lattice_system_of(bravais_lattice)
        hkl_ref = hkl_reference(bravais_lattice, models_directory)
        data_params = dict(data_params or {})
        data_params.setdefault('lattice_system', lattice_system)
        data_params.setdefault('unit_cell_indices', UNIT_CELL_INDICES[lattice_system])
        data_params.setdefault('unit_cell_length', len(UNIT_CELL_INDICES[lattice_system]))
        data_params.setdefault('n_peaks', ASSIGNMENT_MODEL_DEFAULTS['peak_length'])
        data_params.setdefault('hkl_ref_length', len(hkl_ref))
        model_params = dict(model_params or {})
        for key, value in ASSIGNMENT_MODEL_DEFAULTS.items():
            model_params.setdefault(key, dict(value) if isinstance(value, dict) else value)
        calibration = dict(ASSIGNMENT_MODEL_DEFAULTS['calibration_params'])
        calibration.update(model_params.get('calibration_params', {}))
        model_params['calibration_params'] = calibration
        # The parent does a bare os.mkdir on save_to/split_group, so the parent directory has to
        # exist first, and re-running a stage must not raise on a directory it already made.
        os.makedirs(os.path.join(save_to, bravais_lattice), exist_ok=True)
        super().__init__(
            bravais_lattice, data_params, model_params, save_to, seed, hkl_ref=hkl_ref,
            )
        self.bravais_lattice = bravais_lattice
        self.q2_obs_scale = float(q2_obs_scale)

    def build(self):
        """Build and compile the calibration model, inherited unchanged.

        Split out from `__init__` so a test can construct the object without importing keras, and
        so the two happen in a fixed order rather than as a side effect.
        """
        import keras

        keras.utils.set_random_seed(self.seed)
        self.build_calibration_model()
        return self

    def scale_peaks(self, q2):
        return np.asarray(q2, dtype=np.float32)/self.q2_obs_scale

    def predict_softmax(self, q2, xnn, batch_size=None):
        """The (n_rows, n_peaks, hkl_ref_length) softmax, batched.

        Materialising the whole thing is 10 GB for triclinic at any real pool size
        (`IntegralFilter.py:1444-1448`), so callers that only need a few numbers per peak should
        use `assignment_probability`, which reduces inside the loop. This exists for the tests and
        for small diagnostic slices.
        """
        batch_size = batch_size or predict_batch_size(
            self.data_params['n_peaks'], self.hkl_ref.shape[0],
            )
        inputs = (self.scale_peaks(q2), np.asarray(xnn, dtype=np.float32))
        return np.asarray(
            self.calibration_model.predict(inputs, batch_size=batch_size, verbose=0),
            dtype=np.float64,
            )

    def assignment_probability(self, q2, xnn, hkl_assign, batch_size=None, chunk=4096):
        """Two probabilities per peak, reduced inside the batch loop.

        `at_assignment` is the mass the model puts on the line the indexer actually assigned --
        the number that pairs like for like with the analytic estimators, which score exactly that
        assignment. `at_argmax` is the mass on the model's own best class, which is a strictly
        more capable predictor because the model is allowed to disagree with the assignment; it is
        reported beside, never instead.

        `hkl_assign` is (n_rows, n_peaks) class indices into this lattice's `hkl_ref`.
        """
        q2 = np.asarray(q2, dtype=float)
        hkl_assign = np.asarray(hkl_assign, dtype=int)
        at_assignment = np.zeros(hkl_assign.shape, dtype=np.float64)
        at_argmax = np.zeros(hkl_assign.shape, dtype=np.float64)
        argmax = np.zeros(hkl_assign.shape, dtype=np.int64)
        for start in range(0, len(q2), chunk):
            stop = min(start + chunk, len(q2))
            softmax = self.predict_softmax(q2[start:stop], xnn[start:stop], batch_size)
            at_assignment[start:stop] = np.take_along_axis(
                softmax, hkl_assign[start:stop][:, :, np.newaxis], axis=2,
                )[:, :, 0]
            argmax[start:stop] = softmax.argmax(axis=2)
            at_argmax[start:stop] = softmax.max(axis=2)
        return at_assignment, at_argmax, argmax

    def save_assignment(self, directory=None):
        """Weights, parameters and the reference list, in `PriorNetwork.save_prior`'s style.

        Not `IntegralFilter.save_calibration`, which also exports ONNX and a dynamically quantised
        copy. Quantisation is a deployment decision and S14's to make once there is a result worth
        deploying; writing it here would price a model that has not passed its gate yet.
        """
        import json

        directory = directory or self.save_to_split_group
        os.makedirs(directory, exist_ok=True)
        self.calibration_model.save_weights(os.path.join(directory, 'assignment.weights.h5'))
        np.save(os.path.join(directory, 'hkl_ref.npy'), self.hkl_ref)
        with open(os.path.join(directory, 'model_params.json'), 'w', encoding='utf-8') as handle:
            json.dump({
                'bravais_lattice': self.bravais_lattice,
                'q2_obs_scale': self.q2_obs_scale,
                'model_params': self.model_params,
                'data_params': {
                    key: value for key, value in self.data_params.items()
                    if key != 'unit_cell_indices'
                    },
                'unit_cell_indices': list(self.data_params['unit_cell_indices']),
                }, handle, indent=2)
        return directory

    @classmethod
    def load_assignment(cls, directory, seed=12345, models_directory=None):
        import json

        with open(os.path.join(directory, 'model_params.json'), encoding='utf-8') as handle:
            saved = json.load(handle)
        model = cls(
            saved['bravais_lattice'], saved['model_params'], os.path.dirname(directory),
            saved['q2_obs_scale'], seed=seed, models_directory=models_directory,
            data_params=dict(saved['data_params'],
                             unit_cell_indices=saved['unit_cell_indices']),
            ).build()
        model.calibration_model.load_weights(os.path.join(directory, 'assignment.weights.h5'))
        return model
