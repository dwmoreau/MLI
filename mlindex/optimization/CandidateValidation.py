import numpy as np

from mlindex.utilities.UnitCellTools import get_partial_unit_cell


def validate_candidate(entry, top_unit_cell, top_M20):
    found = False
    off_by_two = False
    incorrect_bl = False
    found_explainer = False

    unit_cell_true = np.array(entry['reindexed_unit_cell'])
    bravais_lattice_true = entry['bravais_lattice']

    for bravais_lattice_pred in top_unit_cell.keys():
        for candidate_index in range(top_unit_cell[bravais_lattice_pred].shape[0]):
            correct, off_by_two = validate_candidate_known_bl(
                unit_cell_true=unit_cell_true,
                unit_cell_pred=top_unit_cell[bravais_lattice_pred][candidate_index],
                bravais_lattice_pred=bravais_lattice_pred,
                )
            if correct:
                if bravais_lattice_pred == bravais_lattice_true:
                    found = True
                else:
                    incorrect_bl = True
            if off_by_two:
                off_by_two = True
            if np.any(top_M20[bravais_lattice_pred] > 1000):
                found_explainer = True
    return found, off_by_two, incorrect_bl, found_explainer


def validate_candidate_known_bl(unit_cell_true, unit_cell_pred, bravais_lattice_pred, rtol=1e-2):
    # This should probably be replace with distance measurements in NCDIST
    from mlindex.utilities.Reindexing import reindex_entry_triclinic
    if bravais_lattice_pred in ['cF', 'cI', 'cP']:
        lattice_system_pred = 'cubic'
        unit_cell_true = unit_cell_true[0]
    elif bravais_lattice_pred == 'hP':
        lattice_system_pred = 'hexagonal'
        unit_cell_true = unit_cell_true[[0, 2]]
    elif bravais_lattice_pred == 'hR':
        lattice_system_pred = 'rhombohedral'
        unit_cell_true = unit_cell_true[[0, 3]]
    elif bravais_lattice_pred in ['tI', 'tP']:
        lattice_system_pred = 'tetragonal'
        unit_cell_true = unit_cell_true[[0, 2]]
    elif bravais_lattice_pred in ['oC', 'oF', 'oI', 'oP']:
        lattice_system_pred = 'orthorhombic'
        unit_cell_true = unit_cell_true[:3]
    elif bravais_lattice_pred in ['mC', 'mP']:
        lattice_system_pred = 'monoclinic'
        unit_cell_true = unit_cell_true[[0, 1, 2, 4]]
    elif bravais_lattice_pred == 'aP':
        lattice_system_pred = 'triclinic'

    if lattice_system_pred == 'cubic':
        if np.isclose(unit_cell_pred, unit_cell_true, rtol=rtol):
            return True, False
        mult_factors = np.array([1/2, 2])
        for mf in mult_factors:
            if np.isclose(mf * unit_cell_pred, unit_cell_true, rtol=rtol):
                return False, True
    elif lattice_system_pred in ['tetragonal', 'hexagonal']:
        if np.all(np.isclose(unit_cell_pred, unit_cell_true, rtol=rtol)):
            return True, False
        mult_factors = np.array([1/3, 1/2, 1, 2, 3])
        for mf0 in mult_factors:
            for mf1 in mult_factors:
                mf = np.array([mf0, mf1])
                if np.all(np.isclose(mf * unit_cell_pred, unit_cell_true, rtol=rtol)):
                    return False, True
    elif lattice_system_pred == 'rhombohedral':
        if np.all(np.isclose(unit_cell_pred, unit_cell_true, rtol=rtol)):
            return True, False
        mult_factors = np.array([1/2, 2])
        transformations = [
            np.eye(3),
            np.array([
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                ]),
            np.array([
                [3, -1, -1],
                [-1, 3, -1],
                [-1, -1, 3],
                ]),
            np.array([
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
                ]),
            np.array([
                [0.50, 0.25, 0.25],
                [0.25, 0.50, 0.25],
                [0.25, 0.25, 0.50],
                ])
            ]
        ax = unit_cell_pred[0]
        bx = unit_cell_pred[0]*np.cos(unit_cell_pred[1])
        by = unit_cell_pred[0]*np.sin(unit_cell_pred[1])
        cx = unit_cell_pred[0]*np.cos(unit_cell_pred[1])
        arg = (np.cos(unit_cell_pred[1]) - np.cos(unit_cell_pred[1])**2) / np.sin(unit_cell_pred[1])
        cy = unit_cell_pred[0] * arg
        cz = unit_cell_pred[0] * np.sqrt(np.sin(unit_cell_pred[1])**2 - arg**2)
        ucm = np.array([
            [ax, bx, cx],
            [0,  by, cy],
            [0,  0,  cz]
            ])
        found = False
        off_by_two = False
        for trans in transformations:
            rucm = ucm @ trans
            reindexed_unit_cell = np.zeros(2)
            reindexed_unit_cell[0] = np.linalg.norm(rucm[:, 0])
            reindexed_unit_cell[1] = np.arccos(np.dot(rucm[:, 1], rucm[:, 2]) / reindexed_unit_cell[0]**2)
            if np.all(np.isclose(reindexed_unit_cell, unit_cell_true, rtol=rtol)):
                found = True
            mult_factors = np.array([1/2, 2])
            for mf in mult_factors:
                if np.all(np.isclose(np.array([mf, 1]) * reindexed_unit_cell, unit_cell_true, rtol=rtol)):
                    off_by_two = True
        return found, off_by_two
    elif lattice_system_pred == 'orthorhombic':
        unit_cell_true_sorted = np.sort(unit_cell_true)
        unit_cell_pred_sorted = np.sort(unit_cell_pred)
        if np.all(np.isclose(unit_cell_pred_sorted, unit_cell_true_sorted, rtol=rtol)):
            return True, False
        mult_factors = np.array([1/2, 1, 2])
        for mf0 in mult_factors:
            for mf1 in mult_factors:
                for mf2 in mult_factors:
                    mf = np.array([mf0, mf1, mf2])
                    if np.all(np.isclose(np.sort(mf * unit_cell_pred), unit_cell_true_sorted, rtol=rtol)):
                        return False, True
    elif lattice_system_pred == 'monoclinic':
        mult_factors = np.array([1/2, 1, 2])
        obtuse_reindexer = [
            np.eye(3),
            np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
                ])
            ]
        ac_reindexer = [
            np.eye(3),
            np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0],
                ])
            ]
        transformations = [
            np.eye(3),
            np.array([
                [-1, 0, 1],
                [0, 1, 0],
                [-1, 0, 0],
                ]),
            np.array([
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, -1],
                ]),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 1],
                ]),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                ]),
            ]

        ucm = np.array([
            [unit_cell_pred[0], 0,            unit_cell_pred[2] * np.cos(unit_cell_pred[3])],
            [0,            unit_cell_pred[1], 0],
            [0,            0,            unit_cell_pred[2] * np.sin(unit_cell_pred[3])],
            ])
        found = False
        off_by_two = False
        for trans in transformations:
            for perm in ac_reindexer:
                for obt in obtuse_reindexer:
                    rucm = ucm @ obt @ perm @ trans
                    reindexed_unit_cell = np.zeros(4)
                    reindexed_unit_cell[0] = np.linalg.norm(rucm[:, 0])
                    reindexed_unit_cell[1] = np.linalg.norm(rucm[:, 1])
                    reindexed_unit_cell[2] = np.linalg.norm(rucm[:, 2])
                    dot_product = np.dot(rucm[:, 0], rucm[:, 2])
                    mag = reindexed_unit_cell[0] * reindexed_unit_cell[2]
                    reindexed_unit_cell[3] = np.arccos(dot_product / mag)
                    if np.all(np.isclose(reindexed_unit_cell, unit_cell_true, rtol=rtol)):
                        found = True
                    mult_factors = np.array([1/2, 1, 2])
                    for mf0 in mult_factors:
                        for mf1 in mult_factors:
                            for mf2 in mult_factors:
                                mf = np.array([mf0, mf1, mf2, 1])
                                if np.all(np.isclose(mf * reindexed_unit_cell, unit_cell_true, rtol=rtol)):
                                    off_by_two = True
        return found, off_by_two
    elif lattice_system_pred == 'triclinic':
        reindexed_unit_cell, _ = reindex_entry_triclinic(unit_cell_pred)
        found = False
        off_by_two = False
        if np.all(np.isclose(unit_cell_pred, unit_cell_true, rtol=rtol)):
            found = True
        mult_factors = np.array([1/2, 1, 2])
        for mf0 in mult_factors:
            for mf1 in mult_factors:
                for mf2 in mult_factors:
                    mf = np.array([mf0, mf1, mf2, 1, 1, 1])
                    if np.all(np.isclose(mf * reindexed_unit_cell, unit_cell_true, rtol=rtol)):
                        off_by_two = True
        return found, off_by_two
    return False, False


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


# ---------------------------------------------------------------------------------------------
# Batched correctness labelling.
#
# Cherry-picked from `fom` for S03 Phase 2, which produces a fresh candidate pool that has to be
# labelled from scratch. `validate_candidate_known_bl` above answers the same question one
# candidate at a time; this answers it for a whole (entry, lattice system) block, and campaign 1
# measured the pair at 1 584x with zero disagreements over 57.4 M rows (F-166). At that scale the
# difference is 56 seconds against a day.
#
# Nothing here is new: the transformation sets are the same matrices in the same order as the
# scalar routine walks, so the two agree by construction rather than by test.
# ---------------------------------------------------------------------------------------------

# Which columns of a stored six-parameter `unit_cell_true` the labeller compares, per lattice
# system. The truth table stores (a, b, c, alpha, beta, gamma); the batch labellers want the free
# parameters only, in the order `get_unit_cell_from_xnn(partial_unit_cell=True)` returns them.
#
# These are INDEX LISTS, not ranges -- monoclinic takes beta and not alpha, and rhombohedral takes
# alpha and not c, so a contiguous slice silently compares the wrong angle.
#
# DERIVED, not restated. `UnitCellTools.get_partial_unit_cell` already encodes this selection, and
# writing the indices out again here would be a second copy of one rule -- which is the defect
# `FomConditions` was written to close after campaign 1 kept its condition-tag rule in four places
# and they drifted. Running the library function over a probe vector recovers the indices it
# selects, so a change there follows through to the labeller instead of silently disagreeing with
# it. `tests/test_candidate_validation_batch.py` pins the values so the derivation cannot go wrong
# quietly either.
LATTICE_SYSTEMS = ('cubic', 'tetragonal', 'hexagonal', 'rhombohedral', 'orthorhombic',
                   'monoclinic', 'triclinic')

TRUTH_SLICE = {
    system: np.atleast_1d(
        get_partial_unit_cell(np.arange(6), lattice_system=system)).tolist()
    for system in LATTICE_SYSTEMS
    }

# The multiplier grids `validate_candidate_known_bl` walks to set `off_by_two`, per system and in
# its own order. Each is a sub- or super-cell test: a candidate whose axes are a small rational
# multiple of the truth's found the right lattice at the wrong scale, which is its own class and
# never a failure (SCHEMA.md).
_MULT_HALF_DOUBLE = np.array([1 / 2, 2])
_MULT_THIRDS = np.array([1 / 3, 1 / 2, 1, 2, 3])
_MULT_UNIT = np.array([1 / 2, 1, 2])


def _monoclinic_basis_changes():
    """The 20 fixed products `obt @ perm @ trans` of `validate_candidate_known_bl`.

    Built here in the same order and from the same matrices, so the two routines walk an
    identical set. `rucm = ucm @ obt @ perm @ trans`, and the three factors are constant, so
    their product can be formed once instead of per candidate.
    """
    transformations = [
        np.eye(3),
        np.array([[-1, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
        np.array([[0, 0, -1], [0, 1, 0], [1, 0, -1]], dtype=np.float64),
        np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.float64),
        ]
    ac_reindexer = [
        np.eye(3),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
        ]
    obtuse_reindexer = [
        np.eye(3),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64),
        ]
    return np.stack([obt @ perm @ trans
                     for trans in transformations
                     for perm in ac_reindexer
                     for obt in obtuse_reindexer])


_MONOCLINIC_BASIS_CHANGES = _monoclinic_basis_changes()


def is_correct_known_bl_batch(unit_cell_true, unit_cell_pred, lattice_system, rtol=1e-2):
    """`validate_candidate_known_bl(...)[0]` for a batch sharing one truth and one system.

    `unit_cell_true` is the *sliced* truth the scalar routine builds for that system --
    `[a, b, c, beta]` for monoclinic, the full six for triclinic -- and `unit_cell_pred` is
    (n, k) of partial cells in the optimizer's own representation. Returns a boolean array.

    All seven lattice systems are implemented. An unknown one raises rather than silently
    returning False, because a quiet False here would read as "no correct candidate" and be
    indistinguishable from a generation failure (PROTOCOL section 8 keeps those buckets apart).
    """
    unit_cell_true = np.asarray(unit_cell_true, dtype=np.float64)
    unit_cell_pred = np.atleast_2d(np.asarray(unit_cell_pred, dtype=np.float64))
    if unit_cell_pred.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    if lattice_system == 'triclinic':
        # The scalar routine compares the *unreindexed* prediction for `found`; only the
        # off-by-two arm reads the Selling-reduced form. Preserved exactly, including the
        # argument order, since np.isclose is asymmetric in rtol.
        return np.all(np.isclose(unit_cell_pred, unit_cell_true, rtol=rtol), axis=1)

    if lattice_system in ('cubic', 'tetragonal', 'hexagonal'):
        # One or two free parameters and no basis ambiguity to walk: `found` is a direct
        # comparison in the scalar routine too.
        return np.all(np.isclose(unit_cell_pred, unit_cell_true, rtol=rtol), axis=1)

    if lattice_system == 'orthorhombic':
        # The scalar routine sorts both sides before comparing, so an axis permutation is
        # correct rather than merely close.
        return np.all(np.isclose(np.sort(unit_cell_pred, axis=1), np.sort(unit_cell_true),
                                 rtol=rtol), axis=1)

    if lattice_system == 'rhombohedral':
        return _rhombohedral_batch(unit_cell_true, unit_cell_pred, rtol)

    if lattice_system != 'monoclinic':
        raise ValueError(
            f'is_correct_known_bl_batch does not implement {lattice_system!r}. '
            'Use validate_candidate_known_bl for it.')

    reindexed = _monoclinic_reindexed(unit_cell_pred)
    with np.errstate(invalid='ignore'):
        matches = np.all(np.isclose(reindexed, unit_cell_true, rtol=rtol), axis=2)
    return np.any(matches, axis=0)


def _monoclinic_reindexed(unit_cell_pred):
    """(k, n, 4) -- every monoclinic candidate under every one of the 20 basis changes.

    Split out of `is_correct_known_bl_batch` so the off-by-two arm walks the *same* array rather
    than a second construction of it. The scalar routine computes both flags inside one loop over
    the bases, so sharing the reindexed cells here is what keeps the two agreeing by construction.
    """
    a, b, c, beta = (unit_cell_pred[:, 0], unit_cell_pred[:, 1],
                     unit_cell_pred[:, 2], unit_cell_pred[:, 3])
    n = a.shape[0]
    ucm = np.zeros((n, 3, 3), dtype=np.float64)
    ucm[:, 0, 0] = a
    ucm[:, 0, 2] = c * np.cos(beta)
    ucm[:, 1, 1] = b
    ucm[:, 2, 2] = c * np.sin(beta)

    # (k, n, 3, 3): every candidate under every basis change.
    rucm = np.einsum('nij,kjl->knil', ucm, _MONOCLINIC_BASIS_CHANGES)
    lengths = np.linalg.norm(rucm, axis=2)                     # (k, n, 3)
    dot = np.einsum('kni,kni->kn', rucm[:, :, :, 0], rucm[:, :, :, 2])
    magnitude = lengths[:, :, 0] * lengths[:, :, 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        # arccos of a magnitude above one yields NaN, which compares False -- the same
        # outcome the scalar routine reaches, reached the same way.
        angle = np.arccos(dot / magnitude)
    return np.concatenate([lengths, angle[:, :, np.newaxis]], axis=2)        # (k, n, 4)


_RHOMBOHEDRAL_TRANSFORMS = np.stack([
    np.eye(3),
    np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64),
    np.array([[3, -1, -1], [-1, 3, -1], [-1, -1, 3]], dtype=np.float64),
    np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=np.float64),
    np.array([[0.50, 0.25, 0.25], [0.25, 0.50, 0.25], [0.25, 0.25, 0.50]], dtype=np.float64),
    ])


def _rhombohedral_reindexed(unit_cell_pred):
    """The rhombohedral arm of `is_correct_known_bl_batch`.

    Reproduces `validate_candidate_known_bl`'s cell-matrix construction exactly, including
    the two places it writes `unit_cell_pred[1]` where `unit_cell_pred[0]` might be expected
    -- `cx` is built from the same cosine as `bx`. That is the production definition of a
    correct rhombohedral candidate whether or not it was intended, so it is what the batch
    form has to match, and the equivalence test would fail if it were "corrected" here.
    """
    a, alpha = unit_cell_pred[:, 0], unit_cell_pred[:, 1]
    n = a.shape[0]
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    with np.errstate(invalid='ignore', divide='ignore'):
        arg = (cos_alpha - cos_alpha ** 2) / sin_alpha
        cz = a * np.sqrt(sin_alpha ** 2 - arg ** 2)
    ucm = np.zeros((n, 3, 3), dtype=np.float64)
    ucm[:, 0, 0] = a
    ucm[:, 0, 1] = a * cos_alpha
    ucm[:, 0, 2] = a * cos_alpha
    ucm[:, 1, 1] = a * sin_alpha
    ucm[:, 1, 2] = a * arg
    ucm[:, 2, 2] = cz

    rucm = np.einsum('nij,kjl->knil', ucm, _RHOMBOHEDRAL_TRANSFORMS)
    first = np.linalg.norm(rucm[:, :, :, 0], axis=2)
    dot = np.einsum('kni,kni->kn', rucm[:, :, :, 1], rucm[:, :, :, 2])
    with np.errstate(invalid='ignore', divide='ignore'):
        angle = np.arccos(dot / first ** 2)
    return np.stack([first, angle], axis=2)                                  # (k, n, 2)


def _rhombohedral_batch(unit_cell_true, unit_cell_pred, rtol):
    reindexed = _rhombohedral_reindexed(unit_cell_pred)
    with np.errstate(invalid='ignore'):
        matches = np.all(np.isclose(reindexed, unit_cell_true, rtol=rtol), axis=2)
    return np.any(matches, axis=0)


def off_by_two_known_bl_batch(unit_cell_true, unit_cell_pred, lattice_system, rtol=1e-2):
    """`validate_candidate_known_bl(...)[1]` for a batch sharing one truth and one system.

    A candidate is "off by two" when its axes are a small rational multiple of the truth's: it
    found the right lattice at the wrong scale, which SCHEMA.md classes separately and never as a
    failure. `is_correct_known_bl_batch` deliberately drops this grid, because `found` never reads
    it and the grid is what makes the scalar routine cost ~9 ms a candidate. `FomMetrics` reports
    an `off_by_two` outcome bucket, though, so the column has to be produced -- and at 2.5 billion
    survivor rows it has to be produced in a batch.

    The multiplier loop is over the grid (2 to 27 terms) and vectorised over candidates *and*
    bases, rather than the other way round: the transposed form would materialise a
    (27, 20, n, 4) array, which is 86 MB for a single mP pool and gains nothing.

    Two asymmetries in the scalar routine that are reproduced here rather than tidied:

    * The multiplier grids differ per system -- {1/2, 2} for cubic and rhombohedral,
      {1/3, 1/2, 1, 2, 3} squared for tetragonal and hexagonal, {1/2, 1, 2} cubed elsewhere -- and
      the ones containing 1 therefore re-test the `found` comparison. In the four systems where
      the scalar routine *returns early* on a match that cannot show, which is what
      `label_known_bl_batch` restores.
    * Triclinic compares the **Selling-reduced** cell here and the **unreduced** one for `found`.
      That is the production definition, so it is the one the batch form has to match.
    """
    unit_cell_true = np.asarray(unit_cell_true, dtype=np.float64)
    unit_cell_pred = np.atleast_2d(np.asarray(unit_cell_pred, dtype=np.float64))
    if unit_cell_pred.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    def _any_multiple(values, multipliers, true_values):
        """OR over the grid of `np.all(isclose(mf * values, true_values))`.

        `values` is (..., n, d) and `multipliers` is (m, d), so one grid step is a whole batch.
        """
        found = np.zeros(values.shape[-2], dtype=bool)
        for multiplier in multipliers:
            with np.errstate(invalid='ignore'):
                matched = np.all(np.isclose(multiplier * values, true_values, rtol=rtol), axis=-1)
            found |= matched if matched.ndim == 1 else np.any(matched, axis=0)
        return found

    if lattice_system == 'cubic':
        grid = _MULT_HALF_DOUBLE[:, np.newaxis]
        return _any_multiple(unit_cell_pred, grid, unit_cell_true)

    if lattice_system in ('tetragonal', 'hexagonal'):
        grid = np.stack(np.meshgrid(_MULT_THIRDS, _MULT_THIRDS, indexing='ij'), axis=-1)
        return _any_multiple(unit_cell_pred, grid.reshape(-1, 2), unit_cell_true)

    if lattice_system == 'orthorhombic':
        # The comparison is between SORTED cells, and the multiplier is applied before the sort --
        # so a scaled axis can change places. Sorting after multiplying is what the scalar routine
        # does and the two are not interchangeable.
        true_sorted = np.sort(unit_cell_true)
        grid = np.stack(np.meshgrid(_MULT_UNIT, _MULT_UNIT, _MULT_UNIT, indexing='ij'), axis=-1)
        found = np.zeros(unit_cell_pred.shape[0], dtype=bool)
        for multiplier in grid.reshape(-1, 3):
            with np.errstate(invalid='ignore'):
                found |= np.all(np.isclose(np.sort(multiplier * unit_cell_pred, axis=1),
                                           true_sorted, rtol=rtol), axis=1)
        return found

    if lattice_system == 'rhombohedral':
        reindexed = _rhombohedral_reindexed(unit_cell_pred)
        grid = np.stack([_MULT_HALF_DOUBLE, np.ones_like(_MULT_HALF_DOUBLE)], axis=-1)
        return _any_multiple(reindexed, grid, unit_cell_true)

    if lattice_system == 'monoclinic':
        reindexed = _monoclinic_reindexed(unit_cell_pred)
        grid = np.stack(np.meshgrid(_MULT_UNIT, _MULT_UNIT, _MULT_UNIT, indexing='ij'), axis=-1)
        grid = np.concatenate([grid.reshape(-1, 3), np.ones((grid.size // 3, 1))], axis=1)
        return _any_multiple(reindexed, grid, unit_cell_true)

    if lattice_system == 'triclinic':
        from mlindex.utilities.Reindexing import reindex_entry_triclinic
        reindexed, _ = reindex_entry_triclinic(unit_cell_pred)
        grid = np.stack(np.meshgrid(_MULT_UNIT, _MULT_UNIT, _MULT_UNIT, indexing='ij'), axis=-1)
        grid = np.concatenate([grid.reshape(-1, 3), np.ones((grid.size // 3, 3))], axis=1)
        return _any_multiple(reindexed, grid, unit_cell_true)

    raise ValueError(
        f'off_by_two_known_bl_batch does not implement {lattice_system!r}. '
        'Use validate_candidate_known_bl for it.')


# The four systems whose scalar arm `return`s on the first match, so `found` and `off_by_two` are
# mutually exclusive there. In the other three the scalar routine accumulates both inside one loop
# over the bases and a candidate can carry both flags, which is why this is a set and not a rule.
_EARLY_RETURN_SYSTEMS = frozenset({'cubic', 'tetragonal', 'hexagonal', 'orthorhombic'})


def label_known_bl_batch(unit_cell_true, unit_cell_pred, lattice_system, rtol=1e-2):
    """`validate_candidate_known_bl` for a whole (entry, lattice system) block.

    Returns `(is_correct, is_off_by_two)` as boolean arrays -- both of the scalar routine's
    returns, which `is_correct_known_bl_batch` gives only the first of. This is what the benchmark
    driver labels with: at 2.5 billion survivor rows the scalar routine is not an option, and
    `FomMetrics` reads both flags.

    `unit_cell_true` is the *sliced* truth for that system -- use `TRUTH_SLICE`. Gated against the
    scalar routine over a real pool in `tests/test_candidate_validation_batch.py`; a disagreement
    there is a defect in this function, never a reason to relax the test.
    """
    correct = is_correct_known_bl_batch(unit_cell_true, unit_cell_pred, lattice_system, rtol=rtol)
    off_by_two = off_by_two_known_bl_batch(unit_cell_true, unit_cell_pred, lattice_system,
                                           rtol=rtol)
    if lattice_system in _EARLY_RETURN_SYSTEMS:
        # The scalar routine returns `(True, False)` before it ever reaches the grid, and every
        # one of those grids contains the identity, so without this mask a correct candidate would
        # come back flagged as its own sub-cell.
        off_by_two &= ~correct
    return correct, off_by_two


def basis_change_known_bl_batch(unit_cell_true, unit_cell_pred, lattice_system, rtol=1e-2):
    """The basis change taking each *correct* candidate's cell to the truth's, or NaN.

    Returns (n, 3, 3). Row `i` is the matrix `M` with `A_true = A_pred @ M`, where the columns of
    `A` are the direct-space basis vectors -- which is exactly the matrix
    `is_correct_known_bl_batch` matched on, recovered rather than re-derived. NaN wherever no
    basis change matched, which is every incorrect candidate.

    **This is what `hkl_true_in_basis` is built from.** `is_correct` is a claim about the
    *lattice*, and a monoclinic lattice admits many equivalent cells, so "the truth's Miller index"
    is meaningless until it is expressed in the setting the candidate actually chose. Campaign 1
    stored no transformation at all, so pooling per-peak assignment claims without the setting cut
    moved its base rate from 0.83 to 0.38 (R15, F-126).

    The reciprocal basis is `B = A^-T`, so `B_true = B_pred @ M^-T` and a reflection column
    transforms as `h_pred = M^-T @ h_true`. `hkl_in_candidate_basis` does that step.

    Only correct candidates get a matrix. An off-by-two candidate is a sub- or super-cell, so the
    relation is a scaling rather than a basis change and the integer Miller indices do not survive
    it; those rows are null **because the quantity does not exist for them**, which is a different
    thing from a column that ships empty (C2-F-046).
    """
    unit_cell_true = np.asarray(unit_cell_true, dtype=np.float64)
    unit_cell_pred = np.atleast_2d(np.asarray(unit_cell_pred, dtype=np.float64))
    n = unit_cell_pred.shape[0]
    result = np.full((n, 3, 3), np.nan)
    if n == 0:
        return result

    if lattice_system in ('cubic', 'tetragonal', 'hexagonal', 'triclinic'):
        # No basis walk: the scalar routine compares the cell as given, so a match is the
        # identity setting and the truth's Miller indices are already in the candidate's basis.
        matched = is_correct_known_bl_batch(unit_cell_true, unit_cell_pred, lattice_system,
                                            rtol=rtol)
        result[matched] = np.eye(3)
        return result

    if lattice_system == 'orthorhombic':
        # The comparison sorts both sides, so the setting is an axis permutation: the k-th
        # shortest candidate axis is the k-th shortest true axis.
        true_order = np.argsort(unit_cell_true)
        pred_order = np.argsort(unit_cell_pred, axis=1)
        matched = np.all(np.isclose(np.take_along_axis(unit_cell_pred, pred_order, axis=1),
                                    unit_cell_true[true_order], rtol=rtol), axis=1)
        for position in np.flatnonzero(matched):
            permutation = np.zeros((3, 3))
            permutation[pred_order[position], true_order] = 1.0
            result[position] = permutation
        return result

    if lattice_system == 'monoclinic':
        reindexed = _monoclinic_reindexed(unit_cell_pred)
        changes = _MONOCLINIC_BASIS_CHANGES
    elif lattice_system == 'rhombohedral':
        reindexed = _rhombohedral_reindexed(unit_cell_pred)
        changes = _RHOMBOHEDRAL_TRANSFORMS
    else:
        raise ValueError(
            f'basis_change_known_bl_batch does not implement {lattice_system!r}.')

    with np.errstate(invalid='ignore'):
        matches = np.all(np.isclose(reindexed, unit_cell_true, rtol=rtol), axis=2)   # (k, n)
    # The FIRST matching basis change, in the order the scalar routine walks them, so two
    # settings that both match resolve the same way here as they do there.
    any_match = np.any(matches, axis=0)
    first = np.argmax(matches, axis=0)
    result[any_match] = changes[first[any_match]]
    return result


def hkl_in_candidate_basis(hkl_true, basis_change):
    """The truth's reflections re-expressed in one candidate's basis, or None.

    `hkl_true` is (m, 3) integers in the true cell's setting and `basis_change` is that
    candidate's `M` from `basis_change_known_bl_batch`. Returns (m, 3) int16, or None when the
    candidate has no basis change -- which is every candidate that is not correct.

    `h_pred = M^-T h_true`, from `B_true = B_pred @ M^-T`. The result is rounded and then checked:
    a basis change between two descriptions of one lattice maps integer Miller indices to integer
    Miller indices, so a residual above a small tolerance means the matrix does not describe a
    lattice correspondence and the answer is withheld rather than rounded into looking valid.
    """
    if basis_change is None or not np.all(np.isfinite(basis_change)):
        return None
    hkl_true = np.asarray(hkl_true, dtype=np.float64).reshape(-1, 3)
    transform = np.linalg.inv(np.asarray(basis_change, dtype=np.float64)).T
    exact = hkl_true @ transform.T
    rounded = np.rint(exact)
    if np.max(np.abs(exact - rounded)) > 1e-6:
        return None
    return rounded.astype(np.int16)
