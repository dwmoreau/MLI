import numpy as np

from mlindex.utilities.Reindexing import rhombohedral_settings
from mlindex.utilities.Reindexing import niggli_cell
from mlindex.utilities.Reindexing import same_lattice
from mlindex.utilities.UnitCellTools import BL_TO_CENTRING
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import get_full_unit_cell
from mlindex.utilities.UnitCellTools import get_partial_unit_cell
from mlindex.utilities.UnitCellTools import get_unit_cell_volume


# The sub- and super-cell multipliers each lattice system is tested against. A candidate whose
# axes are a small rational multiple of the truth found the right lattice at the wrong scale,
# which is its own outcome and never counted as correct.
MULTIPLIERS_HALF_DOUBLE = np.array([1 / 2, 2])
MULTIPLIERS_THIRDS = np.array([1 / 3, 1 / 2, 1, 2, 3])
MULTIPLIERS_UNIT = np.array([1 / 2, 1, 2])


def validate_candidate(entry, top_unit_cell, top_M20):
    """Classify one pattern's candidates against its truth, `entry`.

    `top_unit_cell` and `top_M20` map each Bravais lattice to its candidates' partial cells and
    M20s. Returns `(found, off_by_two, incorrect_bl, found_explainer)`, each true when any
    candidate of any lattice is so, labelled by `label_known_bl_batch`.
    """
    unit_cell_true = np.array(entry['reindexed_unit_cell'])
    found = off_by_two = incorrect_bl = found_explainer = False
    for bravais_lattice_pred, unit_cell_pred in top_unit_cell.items():
        correct, candidate_off_by_two = label_known_bl_batch(
            get_partial_unit_cell(unit_cell_true, bravais_lattice=bravais_lattice_pred),
            unit_cell_pred, bravais_lattice_pred)
        if np.any(correct):
            if bravais_lattice_pred == entry['bravais_lattice']:
                found = True
            else:
                incorrect_bl = True
        off_by_two |= bool(np.any(candidate_off_by_two))
        found_explainer |= bool(np.any(top_M20[bravais_lattice_pred] > 1000))
    return found, off_by_two, incorrect_bl, found_explainer


def validate_candidate_known_bl(unit_cell_true, unit_cell_pred, bravais_lattice_pred, rtol=1e-2):
    """Is `unit_cell_pred` the true cell, and if not, is it a sub- or super-cell of it?

    Returns (correct, off_by_two) for one candidate, as `label_known_bl_batch` labels it.
    `unit_cell_true` is the full six-parameter cell; `unit_cell_pred` is the partial cell of the
    predicted Bravais lattice.
    """
    correct, off_by_two = label_known_bl_batch(
        get_partial_unit_cell(np.asarray(unit_cell_true), bravais_lattice=bravais_lattice_pred),
        np.asarray(unit_cell_pred)[np.newaxis], bravais_lattice_pred, rtol=rtol)
    return bool(correct[0]), bool(off_by_two[0])


def _lattice_system_of(bravais_lattice, function):
    """The lattice system of `bravais_lattice`, raising for one the labeller does not know.

    A quiet False for an unknown lattice reads as "no correct candidate in the pool", which is a
    generation failure and must stay distinguishable from a ranking one.
    """
    if bravais_lattice not in BL_TO_LATTICE_SYSTEM:
        raise ValueError(f'{function} does not implement {bravais_lattice!r}.')
    return BL_TO_LATTICE_SYSTEM[bravais_lattice]


# The angle tolerance of the monoclinic and triclinic comparison, which is made between reduced
# cells. 1 degree is what the relative tolerance of 1 % gave the old comparison at the angles near
# 100 degrees that reduced cells have; an absolute tolerance does not change when the reduction writes
# an angle as its supplement.
ANGLE_TOLERANCE_DEG = 1.0


def _same_reduced_lattice(unit_cell_true, unit_cell_pred, centring, rtol):
    """Is each full cell of `unit_cell_pred` (n, 6, radians) the lattice of `unit_cell_true`?

    Both are reduced to the Niggli cell of their primitive lattice (`Reindexing.niggli_cell`) and
    compared by `Reindexing.same_lattice`, `rtol` in the lengths and `ANGLE_TOLERANCE_DEG` in the
    angles. A cell cctbx refuses outright (non-finite or impossible parameters) is not the true
    cell, and must not stop a pool's labels.

    `same_lattice` rejects a pair whose volumes differ by more than three times `rtol` before
    comparing; that test is made here first, on every row at once and with a hair of slack, so the
    rows it would reject -- most of a pool -- are never reduced. Both cells have the same
    centring, so their conventional volumes stand in the ratio their primitive ones do.
    """
    degrees = lambda cell: np.concatenate([cell[:3], np.degrees(cell[3:])])
    truth = niggli_cell(degrees(unit_cell_true), centring)
    correct = np.zeros(unit_cell_pred.shape[0], dtype=bool)
    volume_true = get_unit_cell_volume(unit_cell_true[np.newaxis])[0]
    with np.errstate(invalid='ignore'):
        volume = get_unit_cell_volume(unit_cell_pred)
        near = np.abs(volume - volume_true) <= 3.01*rtol*0.5*(volume + volume_true)
    for index in np.flatnonzero(near):
        cell = unit_cell_pred[index]
        try:
            correct[index] = same_lattice(niggli_cell(degrees(cell), centring), truth, rtol=rtol,
                                          atol_deg=ANGLE_TOLERANCE_DEG)
        except (RuntimeError, ValueError):
            continue
    return correct


def is_correct_known_bl_batch(unit_cell_true, unit_cell_pred, bravais_lattice, rtol=1e-2):
    """`validate_candidate_known_bl(...)[0]` for a batch sharing one truth and one Bravais lattice.

    `unit_cell_true` is the truth already sliced to the lattice system's free parameters, by
    `UnitCellTools.get_partial_unit_cell`, and `unit_cell_pred` is (n, k) of partial cells; both
    are in `bravais_lattice`'s setting (`UnitCellTools.BL_TO_CENTRING`). Returns a boolean array.
    """
    lattice_system = _lattice_system_of(bravais_lattice, 'is_correct_known_bl_batch')
    unit_cell_true = np.asarray(unit_cell_true, dtype=np.float64)
    unit_cell_pred = np.atleast_2d(np.asarray(unit_cell_pred, dtype=np.float64))
    if unit_cell_pred.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    if lattice_system in ('cubic', 'tetragonal', 'hexagonal'):
        # No basis walk: the scalar routine compares the cell as given.
        return np.all(np.isclose(unit_cell_pred, unit_cell_true, rtol=rtol), axis=1)

    if lattice_system in ('monoclinic', 'triclinic'):
        # Compared as lattices, not as cells: both reduced, each as the primitive cell of its
        # centred lattice. Walking small basis changes from the cells as given missed a correct
        # cell whenever the two settings were more than one shear apart (beta 151 against 156
        # degrees needs c + 2a), passed a cell whose basis change moved the centring point to
        # another face -- another lattice -- and let 1 % of a long sheared axis pass a cell several
        # percent off in its shortest ones (P17, STATUS P-F-207, P-F-209).
        return _same_reduced_lattice(
            get_full_unit_cell(unit_cell_true[np.newaxis], lattice_system)[0],
            get_full_unit_cell(unit_cell_pred, lattice_system), BL_TO_CENTRING[bravais_lattice],
            rtol)

    if lattice_system == 'orthorhombic':
        # Both sides are sorted before comparing, so an axis permutation is correct rather than
        # merely close.
        return np.all(np.isclose(np.sort(unit_cell_pred, axis=1), np.sort(unit_cell_true),
                                 rtol=rtol), axis=1)

    reindexed = rhombohedral_settings(unit_cell_pred)

    with np.errstate(invalid='ignore'):
        matches = np.all(np.isclose(reindexed, unit_cell_true, rtol=rtol), axis=2)
    return np.any(matches, axis=0)


def off_by_two_known_bl_batch(unit_cell_true, unit_cell_pred, bravais_lattice, rtol=1e-2):
    """`validate_candidate_known_bl(...)[1]` for a batch sharing one truth and one Bravais lattice.

    The multiplier loop runs over the grid and vectorises over candidates and bases, rather than
    the other way round: the transposed form materialises a (grid, bases, n, k) array for no gain.

    The grids differ per system, and the ones containing 1 re-test the correctness comparison, so
    a correct candidate can come back set here; `label_known_bl_batch` clears it.
    """
    lattice_system = _lattice_system_of(bravais_lattice, 'off_by_two_known_bl_batch')
    unit_cell_true = np.asarray(unit_cell_true, dtype=np.float64)
    unit_cell_pred = np.atleast_2d(np.asarray(unit_cell_pred, dtype=np.float64))
    if unit_cell_pred.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    def any_multiple(values, multipliers, true_values):
        """OR over the grid of `np.all(isclose(multiplier * values, true_values))`.

        `values` is (..., n, k) and `multipliers` is (m, k), so one grid step is a whole batch.
        """
        found = np.zeros(values.shape[-2], dtype=bool)
        for multiplier in multipliers:
            with np.errstate(invalid='ignore'):
                matched = np.all(np.isclose(multiplier * values, true_values, rtol=rtol), axis=-1)
            found |= matched if matched.ndim == 1 else np.any(matched, axis=0)
        return found

    def grid_of(multipliers, repeat, pad=0):
        """The `repeat`-fold product of `multipliers`, padded with ones to the cell's width."""
        mesh = np.stack(np.meshgrid(*([multipliers] * repeat), indexing='ij'), axis=-1)
        flat = mesh.reshape(-1, repeat)
        if pad:
            flat = np.concatenate([flat, np.ones((flat.shape[0], pad))], axis=1)
        return flat

    if lattice_system == 'cubic':
        return any_multiple(unit_cell_pred, MULTIPLIERS_HALF_DOUBLE[:, np.newaxis], unit_cell_true)

    if lattice_system in ('tetragonal', 'hexagonal'):
        return any_multiple(unit_cell_pred, grid_of(MULTIPLIERS_THIRDS, 2), unit_cell_true)

    if lattice_system == 'orthorhombic':
        # The multiplier is applied BEFORE the sort, so a scaled axis can change places. Sorting
        # first is not the same operation and would label differently.
        true_sorted = np.sort(unit_cell_true)
        found = np.zeros(unit_cell_pred.shape[0], dtype=bool)
        for multiplier in grid_of(MULTIPLIERS_UNIT, 3):
            with np.errstate(invalid='ignore'):
                found |= np.all(np.isclose(np.sort(multiplier * unit_cell_pred, axis=1),
                                           true_sorted, rtol=rtol), axis=1)
        return found

    if lattice_system == 'rhombohedral':
        grid = np.stack([MULTIPLIERS_HALF_DOUBLE, np.ones_like(MULTIPLIERS_HALF_DOUBLE)], axis=-1)
        return any_multiple(rhombohedral_settings(unit_cell_pred), grid, unit_cell_true)

    # Monoclinic and triclinic: compared as lattices, by the test the correctness arm makes
    # (`_same_reduced_lattice`). The candidate with its axes scaled is tested against the truth,
    # and the candidate against the truth with its axes scaled, so a doubling along an axis of
    # either cell as written is found in whatever setting the other is in.
    truth = get_full_unit_cell(unit_cell_true[np.newaxis], lattice_system)[0]
    cells = get_full_unit_cell(unit_cell_pred, lattice_system)
    centring = BL_TO_CENTRING[bravais_lattice]
    found = np.zeros(cells.shape[0], dtype=bool)
    for multiplier in grid_of(MULTIPLIERS_UNIT, 3, pad=3):
        if np.all(multiplier == 1):
            continue
        found |= _same_reduced_lattice(truth, multiplier * cells, centring, rtol)
        found |= _same_reduced_lattice(multiplier * truth, cells, centring, rtol)
    return found


def label_known_bl_batch(unit_cell_true, unit_cell_pred, bravais_lattice, rtol=1e-2):
    """`validate_candidate_known_bl` for a whole (entry, Bravais lattice) block.

    Returns `(is_correct, is_off_by_two)` as boolean arrays. `unit_cell_true` is the truth
    sliced by `UnitCellTools.get_partial_unit_cell`, the same call the scalar routine makes on
    its own argument. This is what a benchmark labels with: over a pool of millions of
    candidates the scalar routine is not an option.
    """
    correct = is_correct_known_bl_batch(unit_cell_true, unit_cell_pred, bravais_lattice, rtol=rtol)
    off_by_two = off_by_two_known_bl_batch(unit_cell_true, unit_cell_pred, bravais_lattice,
                                           rtol=rtol)
    # A correct candidate is never also off by two. The grids containing the identity flag it as
    # its own sub-cell; this was masked for four systems only, so a correct monoclinic cell came
    # back as both.
    return correct, off_by_two & ~correct


def get_best_candidates(self, report_counts):
    found = False
    found_best = False
    found_not_best = False
    found_off_by_two = False

    xnn_averaged, M20_averaged = self.remove_duplicates()
    unit_cell_averaged = get_unit_cell_from_xnn(
        xnn_averaged, partial_unit_cell=True, lattice_system=self.lattice_system
        )
    sort_indices = np.argsort(M20_averaged)[::-1]
    unit_cell = unit_cell_averaged[sort_indices][:20]
    M20 = M20_averaged[sort_indices][:20]

    for index in range(unit_cell.shape[0]):
        correct, off_by_two = self.validate_candidate(unit_cell[index])
        if correct and index == 0:
            found_best = True
            found = True
        elif correct:
            found_not_best = True
            found = True
        elif off_by_two:
            found_off_by_two = True
            found = True

    if found_best:
        report_counts['Found and best'] += 1
    elif found_not_best:
        report_counts['Found but not best'] += 1
    elif found_off_by_two:
        report_counts['Found but off by two'] += 1
    elif found:
        report_counts['Found explainers'] += 1
    else:
        report_counts['Not found'] += 1
    return report_counts, found
