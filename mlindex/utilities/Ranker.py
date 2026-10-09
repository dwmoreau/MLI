"""The learned ranker's per-candidate inputs, its design matrix and its calibration.

A candidate's inputs are computed from its cell, its extinction group, the observed peak list and
the lattice's reference reflections. Two reference lists are involved and they are not
interchangeable: the structural inputs use the candidate's own extinction group's lines, the
absence counts the lattice's full list, because they count the lines the group removes.

    from mlindex.utilities.Ranker import structural_inputs
    values = structural_inputs(q2_obs, xnn, q2_ref_calc, 'monoclinic', 'mP')
"""
import csv

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
