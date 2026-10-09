"""The learned ranker's per-candidate inputs, its design matrix and its calibration.

A candidate's inputs are computed from its cell, its extinction group, the observed peak list and
the lattice's reference reflections. Two reference lists are involved and they are not
interchangeable: the structural inputs use the candidate's own extinction group's lines, the
absence counts the lattice's full list, because they count the lines the group removes.

    from mlindex.utilities.Ranker import structural_inputs
    values = structural_inputs(q2_obs, xnn, q2_ref_calc, 'monoclinic', 'mP')
"""
import csv
import json
from pathlib import Path

import numpy as np

from mlindex.utilities.FigureOfMerits import HIGHER_IS_BETTER
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES
from mlindex.utilities.UnitCellTools import lattice_order_of

LATTICE_FEATURE = 'bravais_lattice'

# Columns no input may be: labels, quantities derived from the true cell, properties of the
# synthetic conditions, the thinning weights, and constants of the generation run.
FORBIDDEN_COLUMNS = frozenset({
    'is_correct', 'is_off_by_two', 'is_degenerate', 'split', 'condition_bundle',
    'q2_error_multiplier', 'intercept_scale', 'n_contaminants', 'n_contaminants_achieved',
    'n_dropout', 'n_dropout_achieved', 'second_phase_lines', 'second_phase_achieved',
    'second_phase_partner', 'sampling_weight', 'm20_at_prune', 'merit_at_prune',
    'in_top_n', 'prune_threshold', 'downsample_radius', 'assignment_threshold', 'q2_digest',
    'ctx_pool_size',
    # The peak count of the crystal's whole simulated pattern: set by its true cell and symmetry,
    # and not available from a peak list.
    'n_peaks_available',
    })
FORBIDDEN_SUFFIX = '_true'

# The calibrator a lattice without its own falls back to.
POOLED = '__pooled__'

# The packaged ranker's directory inside a models tree.
RANKER_DIRECTORY = 'ranker_1'

# The precision floor in Werner's critical volume. It scales `V_over_Vcrit` by the same factor
# for every candidate.
WERNER_G_MIN = 1.0


def check_no_leakage(names):
    """Raise if any input is a label, is derived from the truth, or describes the generator."""
    offenders = sorted({name for name in names
                        if name in FORBIDDEN_COLUMNS or name.endswith(FORBIDDEN_SUFFIX)})
    if offenders:
        raise ValueError(f'features that are unavailable at inference or derived from the truth: '
                         f'{offenders}')


def oriented(values, merit):
    """The merit's values with their sign set so that larger is better."""
    values = np.asarray(values, dtype=np.float64)
    return values if HIGHER_IS_BETTER[merit] else -values


def absent_fraction(n_removed, n_in_range):
    """The share of the reference lines in range that the extinction group removes; NaN where no
    line is in range."""
    n_in_range = np.asarray(n_in_range, dtype=np.float64)
    return np.where(n_in_range > 0,
                    np.asarray(n_removed, dtype=np.float64)/np.maximum(n_in_range, 1.0), np.nan)


def structural_inputs(q2_obs, xnn, q2_ref_calc, lattice_system, bravais_lattice):
    """`zone_dominance`, `V_over_Vcrit`, `n_dewolff61`, `M_wu`, `F_N_q` and M20, for candidates of
    one extinction group.

    `q2_ref_calc` is (n_candidates, n_ref), from that group's reference list. Returns a dict of
    arrays.
    """
    from mlindex.utilities.FigureOfMerits import (
        get_F_N, get_M20, get_M_wu, get_multiplicity_taupin88, get_n_dewolff61,
        get_V_over_Vcrit, get_zone_dominance)
    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.UnitCellTools import (
        get_reciprocal_unit_cell_from_xnn, get_unit_cell_volume)

    q2_calc = np.take_along_axis(q2_ref_calc, fast_assign(q2_obs, q2_ref_calc), axis=1)
    cutoff = q2_calc[:, -1]
    reciprocal_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system)
    volume = 1/np.maximum(get_unit_cell_volume(
        reciprocal_cell, partial_unit_cell=True, lattice_system=lattice_system), 1e-300)
    d_n = 1/np.sqrt(np.maximum(cutoff, 1e-300))
    over_critical, _ = get_V_over_Vcrit(
        volume, d_n, WERNER_G_MIN, get_multiplicity_taupin88(bravais_lattice)[0])
    return {
        'zone_dominance': get_zone_dominance(xnn, lattice_system),
        'V_over_Vcrit': over_critical,
        'n_dewolff61': get_n_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)[:, -1],
        'M_wu': get_M_wu(q2_obs, q2_calc, q2_ref_calc),
        'F_N_q': get_F_N(q2_obs, q2_calc, q2_ref_calc)[1],
        'M20': get_M20(q2_obs, q2_calc, q2_ref_calc),
        }


def absence_inputs(q2_obs, q2_ref_calc_full, spacegroups, keep_masks):
    """For each candidate, the reference lines its extinction group removes below the cutoff, and
    all reference lines below the cutoff.

    `q2_ref_calc_full` is (n_candidates, n_ref) from the lattice's full reference list, and the
    cutoff is the line of that list the last observed peak is assigned to. `keep_masks` is
    `SpaceGroups.get_spacegroup_keep_masks` for the list, keyed as `spacegroups` is. Returns a
    dict of int64 arrays.
    """
    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.SpaceGroups import count_absences_in_range

    cutoff = np.take_along_axis(
        q2_ref_calc_full, fast_assign(q2_obs, q2_ref_calc_full), axis=1)[:, -1]
    spacegroups = np.asarray(spacegroups)
    n = spacegroups.size
    values = {name: np.empty(n, dtype=np.int64)
              for name in ('n_absent_extra_in_range', 'n_ref_in_range')}
    for spacegroup in dict.fromkeys(spacegroups.tolist()):
        local = np.flatnonzero(spacegroups == spacegroup)
        removed, in_range = count_absences_in_range(
            q2_ref_calc_full[local], keep_masks[spacegroup], cutoff[local])
        values['n_absent_extra_in_range'][local] = removed
        values['n_ref_in_range'][local] = in_range
    return values


# The inputs `candidate_inputs` computes, beside M20 and n_indexed, which the search provides.
CANDIDATE_INPUTS = ('M_rev', 'M_sym', 'X_N', 'n_over', 'M_wu', 'F_N_q', 'zone_dominance',
                    'V_over_Vcrit', 'n_dewolff61', 'f_absent_extra')
# The merits whose gap to the pattern's best value is an input.
CONTEXT_MERITS = ('M20', 'M_sym', 'n_over')


def candidate_inputs(q2_obs, xnn, q2_ref_calc_full, spacegroups, keep_masks, lattice_system,
                     bravais_lattice):
    """The inputs of one lattice's candidates that are read from the candidate and the peaks.

    `q2_obs` is the peak list the lattice was fitted on; `q2_ref_calc_full` is (n_candidates,
    n_ref), every candidate's lines from the lattice's full reference list (shifted by its
    zero-point where there is one); `keep_masks` is `SpaceGroups.get_spacegroup_keep_masks` of
    that list, keyed as `spacegroups` is. Each candidate is scored against its own extinction
    group's lines, and its absences counted against the full list. Returns {name: array} for
    `CANDIDATE_INPUTS`, and the M20 those lines give, under `M20_recomputed`.
    """
    from mlindex.utilities.FigureOfMerits import merit_set

    spacegroups = np.asarray(spacegroups)
    values = {name: np.empty(spacegroups.size) for name in CANDIDATE_INPUTS + ('M20_recomputed',)}
    absences = absence_inputs(q2_obs, q2_ref_calc_full, spacegroups, keep_masks)
    values['f_absent_extra'] = absent_fraction(absences['n_absent_extra_in_range'],
                                               absences['n_ref_in_range'])
    for spacegroup in dict.fromkeys(spacegroups.tolist()):
        local = np.flatnonzero(spacegroups == spacegroup)
        q2_ref_calc = q2_ref_calc_full[local][:, keep_masks[spacegroup]]
        merits = merit_set(q2_obs, q2_ref_calc)
        structural = structural_inputs(q2_obs, xnn[local], q2_ref_calc, lattice_system,
                                       bravais_lattice)
        values['M20_recomputed'][local] = structural['M20']
        for name in CANDIDATE_INPUTS:
            if name in merits:
                values[name][local] = merits[name]
            elif name in structural:
                values[name][local] = structural[name]
    return values


def context_gaps(columns):
    """Each candidate's gap to the best value of `CONTEXT_MERITS` over the candidates given, all
    of one pattern: 0 for the best, negative below it."""
    gaps = {}
    for merit in CONTEXT_MERITS:
        values = oriented(columns[merit], merit)
        gaps[f'ctx_{merit}_gap_to_best'] = values - np.nanmax(values)
    return gaps


def read_group_frequency(path):
    """{(Bravais lattice, extinction group key): the share of the lattice's known structures in
    that group}, from a table `make_group_frequency` wrote."""
    with open(path, newline='', encoding='utf-8') as handle:
        return {(row['bravais_lattice'], row['spacegroup']): float(row['group_frequency'])
                for row in csv.DictReader(handle)}


def group_frequency(bravais_lattice, spacegroups, table):
    """Each candidate's extinction group's share of its lattice's known structures, 0 for a group
    the table does not list. `bravais_lattice` is one lattice or one per candidate."""
    spacegroups = np.asarray(spacegroups)
    lattices = np.broadcast_to(np.asarray(bravais_lattice), spacegroups.shape)
    return np.array([table.get((lattice, spacegroup), 0.0)
                     for lattice, spacegroup in zip(lattices.tolist(), spacegroups.tolist())],
                    dtype=np.float64)


def design_matrix(columns, features, encoding):
    """The float32 (n_candidates, n_columns) array the classifier and its ONNX export read.

    `columns` maps each input to its values (a DataFrame does). The lattice enters as its position
    in `BRAVAIS_LATTICES` plus one (`ordinal` and `native`), or as fourteen indicator columns
    (`onehot`).
    """
    missing = [name for name in features if name not in columns]
    if missing:
        raise KeyError(f'missing input column(s): {missing}')
    parts = []
    for name in features:
        if name != LATTICE_FEATURE:
            parts.append(np.asarray(columns[name], dtype=np.float32)[:, np.newaxis])
            continue
        position = lattice_order_of(np.asarray(columns[LATTICE_FEATURE]).astype(str))
        if encoding == 'onehot':
            parts.append(np.eye(len(BRAVAIS_LATTICES), dtype=np.float32)[position])
        else:
            parts.append((position + 1).astype(np.float32)[:, np.newaxis])
    return np.concatenate(parts, axis=1)


def apply_calibration(raw, lattice, calibrators):
    """The calibrated probability: each row's raw score through its lattice's isotonic knots,
    or the pooled knots for a lattice without its own."""
    raw = np.asarray(raw, dtype=np.float64)
    lattice = np.asarray(lattice)
    out = np.empty(raw.size, dtype=np.float64)
    for name in np.unique(lattice):
        mask = lattice == name
        thresholds, values = calibrators.get(str(name), calibrators[POOLED])
        out[mask] = np.interp(raw[mask], thresholds, values)
    return out


def write_calibrators(path, calibrators):
    """Save {lattice or POOLED: (thresholds, values)} as `<name>__x` / `<name>__y` arrays."""
    np.savez_compressed(path, **{f'{name}__{part}': array
                                 for name, (x, y) in calibrators.items()
                                 for part, array in (('x', x), ('y', y))})


def read_calibrators(path):
    """The calibrators `write_calibrators` saved."""
    with np.load(path) as arrays:
        names = {key.rsplit('__', 1)[0] for key in arrays.files}
        return {name: (arrays[f'{name}__x'], arrays[f'{name}__y']) for name in names}


class LearnedRanker:
    """The packaged ranker: an ONNX classifier, its per-lattice calibrators and the
    extinction-group frequency table, as `mlindex.scripts.package_ranker` writes them.

    Load it once and score a whole pool per call: the classifier has a fixed cost per call.
    """

    def __init__(self, features, encoding, classifier, calibrators, group_frequency,
                 specification):
        check_no_leakage(features)
        self.features = tuple(features)
        self.encoding = encoding
        self.classifier = classifier
        self.calibrators = calibrators
        self.group_frequency = group_frequency
        self.specification = specification

    @classmethod
    def load(cls, directory):
        from mlindex.utilities.IOManagers import SKLearnManager

        directory = Path(directory)
        with open(directory / 'specification.json', encoding='utf-8') as handle:
            specification = json.load(handle)
        classifier = SKLearnManager(
            filename=str(directory / Path(specification['onnx']).stem), model_type='onnx')
        classifier.load()
        return cls(specification['features'], specification['encoding'], classifier,
                   read_calibrators(directory / 'calibrators.npz'),
                   read_group_frequency(directory / specification['group_frequency']),
                   specification)

    def design_matrix(self, columns):
        """The float32 matrix of the inputs, in the order the classifier reads them."""
        return design_matrix(columns, self.features, self.encoding)

    def predict_batch(self, matrix):
        """The classifier's probability for each row of a design matrix, before calibration."""
        return np.asarray(self.classifier.predict_proba(matrix), dtype=np.float64)[:, 1]

    def score(self, columns):
        """The calibrated probability that each candidate is correct."""
        return apply_calibration(self.predict_batch(self.design_matrix(columns)),
                                 columns[LATTICE_FEATURE], self.calibrators)


def load_ranker():
    """(the packaged ranker, a line saying what ranks the output). The ranker is None, and the
    line says why, when the model tree holds no `ranker_1`: the output is then ranked by M_sym."""
    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir

    try:
        directory = _resolve_models_dir() / RANKER_DIRECTORY
    except FileNotFoundError:
        directory = None
    if directory is None or not (directory / 'specification.json').is_file():
        return None, (f'M_sym (fallback): no learned ranker found at {directory}; run '
                      'python -m mlindex.download_models to fetch it')
    ranker = LearnedRanker.load(directory)
    return ranker, f'learned ranker ({RANKER_DIRECTORY}, seed {ranker.specification["seed"]})'


def score_pool(pools, ranker):
    """Score every candidate of one pattern in one batch.

    `pools` maps each Bravais lattice to its candidates' `candidate_inputs` beside `M20`,
    `n_indexed` and `spacegroup`. The context gaps are taken over all of them together. With
    `ranker` None the score is M_sym. Returns {lattice: score array}, in each pool's order.
    """
    lattices = [lattice for lattice in pools if len(pools[lattice]['M20'])]
    if not lattices:
        return {lattice: np.zeros(0) for lattice in pools}
    columns = {name: np.concatenate([np.asarray(pools[lattice][name]) for lattice in lattices])
               for name in pools[lattices[0]]}
    columns[LATTICE_FEATURE] = np.concatenate(
        [np.full(len(pools[lattice]['M20']), lattice) for lattice in lattices])
    if ranker is None:
        score = np.asarray(columns['M_sym'], dtype=np.float64)
    else:
        columns.update(context_gaps(columns))
        columns['group_frequency'] = group_frequency(
            columns[LATTICE_FEATURE], columns['spacegroup'], ranker.group_frequency)
        score = ranker.score(columns)
    ends = np.cumsum([len(pools[lattice]['M20']) for lattice in lattices])
    scores = dict(zip(lattices, np.split(score, ends[:-1])))
    return {lattice: scores.get(lattice, np.zeros(0)) for lattice in pools}
