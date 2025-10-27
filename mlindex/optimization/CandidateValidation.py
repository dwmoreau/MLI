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
