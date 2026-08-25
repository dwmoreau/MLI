import numpy as np


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
    reindexed = np.concatenate([lengths, angle[:, :, np.newaxis]], axis=2)   # (k, n, 4)
    with np.errstate(invalid='ignore'):
        matches = np.all(np.isclose(reindexed, unit_cell_true, rtol=rtol), axis=2)
    return np.any(matches, axis=0)


_RHOMBOHEDRAL_TRANSFORMS = np.stack([
    np.eye(3),
    np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64),
    np.array([[3, -1, -1], [-1, 3, -1], [-1, -1, 3]], dtype=np.float64),
    np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=np.float64),
    np.array([[0.50, 0.25, 0.25], [0.25, 0.50, 0.25], [0.25, 0.25, 0.50]], dtype=np.float64),
    ])


def _rhombohedral_batch(unit_cell_true, unit_cell_pred, rtol):
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
    reindexed = np.stack([first, angle], axis=2)
    with np.errstate(invalid='ignore'):
        matches = np.all(np.isclose(reindexed, unit_cell_true, rtol=rtol), axis=2)
    return np.any(matches, axis=0)
