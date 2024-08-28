from collections import namedtuple
import logging
import numpy as np
import scipy.spatial

from MPIFileHandler import MPIFileHandler
from Optimizer_mpi import OptimizerManager
from Optimizer_mpi import OptimizerWorker
from Reindexing import reindex_entry_triclinic


def validate_candidate(entry, top_unit_cell, top_M20):
    found = False
    off_by_two = False
    incorrect_bl = False
    found_explainer = False

    unit_cell_true = np.array(entry['reindexed_unit_cell'])
    bravais_lattice_true = entry['bravais_lattice']

    for bravais_lattice_pred in top_unit_cell.keys():
        print()
        print(bravais_lattice_pred)
        print(np.column_stack((top_unit_cell[bravais_lattice_pred], top_M20[bravais_lattice_pred])))
        print()
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
    #print(self.unit_cell_true)
    #print(np.concatenate((
    #    unit_cell.round(decimals=4), M20.round(decimals=4)[:, np.newaxis]
    #    ),
    #    axis=1))
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


def get_cubic_optimizer(bravais_lattice, broadening_tag, comm):
    data_params = {
        'tag': f'cubic_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        }
    template_params = {bravais_lattice: {'tag': f'cubic_{broadening_tag}'}}
    reg_params = {f'{bravais_lattice}_0': {'tag': f'cubic_{broadening_tag}'}}
    pitf_params = {f'{bravais_lattice}_0': {'tag': f'cubic_{broadening_tag}'}}
    random_params = {bravais_lattice: {'tag': f'cubic_{broadening_tag}'}}
    generator_info = [
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': 100},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': 150},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': 100},
        {'generator': 'templates', 'n_unit_cells': 150},
        #{'generator': 'random', 'n_unit_cells': 100},
        #{'generator': 'distribution_volume', 'n_unit_cells': 100},
        {'generator': 'predicted_volume', 'n_unit_cells': 150},
        ]
    iteration_info = [{
        'worker': 'random_subsampling',
        'n_iterations': 100,
        'n_peaks': 10,
        'n_drop': 5,
        }]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 10,
        'neighbor_radius': 0.00005,
        }
    return OptimizerManager(data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm)


def get_tetragonal_optimizer(bravais_lattice, broadening_tag, comm):
    data_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        }
    template_params = {bravais_lattice: {'tag': f'tetragonal_{broadening_tag}'}}
    reg_group_params = {'tag': f'tetragonal_{broadening_tag}'}
    reg_params = {
        f'{bravais_lattice}_0_00': reg_group_params,
        f'{bravais_lattice}_1_00': reg_group_params,
        f'{bravais_lattice}_0_01': reg_group_params,
        f'{bravais_lattice}_1_01': reg_group_params,
        }
    pitf_group_params = {'tag': f'tetragonal_{broadening_tag}'}
    pitf_params = {
        f'{bravais_lattice}_0_00': pitf_group_params,
        f'{bravais_lattice}_1_00': pitf_group_params,
        f'{bravais_lattice}_0_01': pitf_group_params,
        f'{bravais_lattice}_1_01': pitf_group_params,
        }
    random_params = {bravais_lattice: {'tag': f'tetragonal_{broadening_tag}'}}
    iteration_info = [{
        'worker': 'random_subsampling',
        'n_iterations': 100,
        'n_peaks': 20,
        'n_drop': 10,
        }]
    generator_info = [
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': 100},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 125},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': 125},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 125},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': 125},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': 100},
        {'generator': 'templates', 'n_unit_cells': 200},
        {'generator': 'random', 'n_unit_cells': 200},
        #{'generator': 'distribution_volume', 'n_unit_cells': 100},
        {'generator': 'predicted_volume', 'n_unit_cells': 200},
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 100,
        'neighbor_radius': 0.0005,
        }
    return OptimizerManager(data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm)


def get_hexagonal_optimizer(bravais_lattice, broadening_tag, comm):
    data_params = {
        'tag': f'hexagonal_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        }
    template_params = {bravais_lattice: {'tag': f'hexagonal_{broadening_tag}'}}
    reg_group_params = {'tag': f'hexagonal_{broadening_tag}'}
    reg_params = {
        f'{bravais_lattice}_0_00': reg_group_params,
        f'{bravais_lattice}_0_01': reg_group_params,
        f'{bravais_lattice}_0_02': reg_group_params,
        f'{bravais_lattice}_0_03': reg_group_params,
        f'{bravais_lattice}_1_00': reg_group_params,
        f'{bravais_lattice}_1_01': reg_group_params,
        f'{bravais_lattice}_1_02': reg_group_params,
        f'{bravais_lattice}_1_03': reg_group_params,
        }
    pitf_group_params = {'tag': f'hexagonal_{broadening_tag}'}
    pitf_params = {
        f'{bravais_lattice}_0_00': pitf_group_params,
        f'{bravais_lattice}_0_01': pitf_group_params,
        f'{bravais_lattice}_0_02': pitf_group_params,
        f'{bravais_lattice}_0_03': pitf_group_params,
        f'{bravais_lattice}_1_00': pitf_group_params,
        f'{bravais_lattice}_1_01': pitf_group_params,
        f'{bravais_lattice}_1_02': pitf_group_params,
        f'{bravais_lattice}_1_03': pitf_group_params,
        }
    random_params = {bravais_lattice: {'tag': f'hexagonal_{broadening_tag}'}}
    generator_info = [
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': 100},
        #{'generator': 'nn', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': 100},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 60},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 60},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': 60},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': 60},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': 60},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': 60},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': 60},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': 60},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': 100},
        {'generator': 'templates', 'n_unit_cells': 300},
        #{'generator': 'random', 'n_unit_cells': 100},
        #{'generator': 'distribution_volume', 'n_unit_cells': 100},
        {'generator': 'predicted_volume', 'n_unit_cells': 200},
        ]
    iteration_info = [{
        'worker': 'random_subsampling',
        'n_iterations': 100,
        'n_peaks': 20,
        'n_drop': 10,
        }]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 20,
        'neighbor_radius': 0.0005,
        }
    return OptimizerManager(data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm)


def get_rhombohedral_optimizer(bravais_lattice, broadening_tag, comm):
    data_params = {
        'tag': f'rhombohedral_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        }
    template_params = {bravais_lattice: {'tag': f'rhombohedral_{broadening_tag}'}}
    reg_group_params = {'tag': f'rhombohedral_{broadening_tag}'}
    reg_params = {
        f'{bravais_lattice}_00': reg_group_params,
        f'{bravais_lattice}_01': reg_group_params,
        }
    pitf_group_params = {'tag': f'rhombohedral_{broadening_tag}'}
    pitf_params = {
        f'{bravais_lattice}_00': pitf_group_params,
        f'{bravais_lattice}_01': pitf_group_params,
        }
    random_params = {bravais_lattice: {'tag': f'rhombohedral_{broadening_tag}'}}
    generator_info = [
        {'generator': 'nn', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': 200},
        {'generator': 'nn', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': 200},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': 200},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': 200},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': 100},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': 100},
        {'generator': 'templates', 'n_unit_cells': 200},
        #{'generator': 'random', 'n_unit_cells': 100},
        #{'generator': 'distribution_volume', 'n_unit_cells': 100},
        {'generator': 'predicted_volume', 'n_unit_cells': 200},
        ]
    iteration_info = [{
        'worker': 'random_subsampling',
        'n_iterations': 100,
        'n_peaks': 20,
        'n_drop': 10,
        }]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 100,
        'neighbor_radius': 0.0005,
        }
    return OptimizerManager(data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm)


def get_orthorhombic_optimizer(bravais_lattice, broadening_tag, comm):
    data_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        }
    template_params = {bravais_lattice: {'tag': f'orthorhombic_{broadening_tag}'}}
    reg_group_params = {'tag': f'orthorhombic_{broadening_tag}'}
    pitf_group_params = {'tag': f'orthorhombic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'orthorhombic_{broadening_tag}'}}
    if bravais_lattice == 'oF':
        reg_params = {
            f'{bravais_lattice}_0_00': reg_group_params,
            f'{bravais_lattice}_0_01': reg_group_params,
            }
        pitf_params = {
            f'{bravais_lattice}_0_00': pitf_group_params,
            f'{bravais_lattice}_0_01': pitf_group_params,
            }
        generator_info = [
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 750},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 750},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 750},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 750},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 100},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': 100},
            {'generator': 'templates', 'n_unit_cells': 2000},
            #{'generator': 'random', 'n_unit_cells': 200},
            #{'generator': 'distribution_volume', 'n_unit_cells': 100},
            {'generator': 'predicted_volume', 'n_unit_cells': 2000},
            ]
    elif bravais_lattice == 'oI':
        reg_params = {f'{bravais_lattice}_0_00': reg_group_params,}
        pitf_params = {f'{bravais_lattice}_0_00': pitf_group_params,}
        generator_info = [
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 1000},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 1000},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 100},
            {'generator': 'templates', 'n_unit_cells': 2000},
            #{'generator': 'random', 'n_unit_cells': 200},
            #{'generator': 'distribution_volume', 'n_unit_cells': 100},
            {'generator': 'predicted_volume', 'n_unit_cells': 2000},
            ]
    elif bravais_lattice == 'oC':
        reg_params = {
            f'{bravais_lattice}_0_00': reg_group_params,
            f'{bravais_lattice}_2_00': reg_group_params,
            }
        pitf_params = {
            f'{bravais_lattice}_0_00': pitf_group_params,
            f'{bravais_lattice}_2_00': pitf_group_params,
            }
        generator_info = [
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 500},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': 500},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 500},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': 500},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': 100},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': 100},
            {'generator': 'templates', 'n_unit_cells': 2000},
            #{'generator': 'random', 'n_unit_cells': 200},
            #{'generator': 'distribution_volume', 'n_unit_cells': 100},
            {'generator': 'predicted_volume', 'n_unit_cells': 1000},
            ]
    elif bravais_lattice == 'oP':
        reg_params = {
            f'{bravais_lattice}_0_00': reg_group_params,
            f'{bravais_lattice}_0_01': reg_group_params,
            f'{bravais_lattice}_0_02': reg_group_params,
            f'{bravais_lattice}_0_03': reg_group_params,
            f'{bravais_lattice}_0_04': reg_group_params,
            f'{bravais_lattice}_0_05': reg_group_params,
            f'{bravais_lattice}_0_06': reg_group_params,
            f'{bravais_lattice}_0_07': reg_group_params,
            f'{bravais_lattice}_0_08': reg_group_params,
            }
        pitf_params = {
            f'{bravais_lattice}_0_00': pitf_group_params,
            f'{bravais_lattice}_0_01': pitf_group_params,
            f'{bravais_lattice}_0_02': pitf_group_params,
            f'{bravais_lattice}_0_03': pitf_group_params,
            f'{bravais_lattice}_0_04': pitf_group_params,
            f'{bravais_lattice}_0_05': pitf_group_params,
            f'{bravais_lattice}_0_06': pitf_group_params,
            f'{bravais_lattice}_0_07': pitf_group_params,
            f'{bravais_lattice}_0_08': pitf_group_params,
            }
        f = 0.6
        generator_info = [
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_04', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_05', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_06', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_07', 'n_unit_cells': int(f*100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_08', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_04', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_05', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_06', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_07', 'n_unit_cells': int(f*100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_08', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_04', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_05', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_06', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_07', 'n_unit_cells': int(f*100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_08', 'n_unit_cells': int(f*100)},
            {'generator': 'templates', 'n_unit_cells': int(f*1000)},
            #{'generator': 'random', 'n_unit_cells': int(f*200)},
            #{'generator': 'distribution_volume', 'n_unit_cells': int(f*100)},
            {'generator': 'predicted_volume', 'n_unit_cells': int(f*1000)},
            ]
    iteration_info = [{
        'worker': 'random_subsampling',
        'n_iterations': 20,
        'n_peaks': 20,
        'n_drop': 10,
        }]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 100,
        'neighbor_radius': 0.0005,
        }
    return OptimizerManager(data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm)


def get_monoclinic_optimizer(bravais_lattice, broadening_tag, comm):
    data_params = {
        'tag': f'monoclinic_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        }
    template_params = {bravais_lattice: {'tag': f'monoclinic_{broadening_tag}'}}
    reg_group_params = {'tag': f'monoclinic_{broadening_tag}'}
    pitf_group_params = {'tag': f'monoclinic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'monoclinic_{broadening_tag}'}}
    f = 1
    if bravais_lattice == 'mC':
        reg_params = {
            f'{bravais_lattice}_0_02': reg_group_params,
            f'{bravais_lattice}_0_03': reg_group_params,
            f'{bravais_lattice}_1_02': reg_group_params,
            f'{bravais_lattice}_1_03': reg_group_params,
            f'{bravais_lattice}_4_02': reg_group_params,
            f'{bravais_lattice}_4_03': reg_group_params,
            }
        pitf_params = {
            f'{bravais_lattice}_0_02': pitf_group_params,
            f'{bravais_lattice}_0_03': pitf_group_params,
            f'{bravais_lattice}_1_02': pitf_group_params,
            f'{bravais_lattice}_1_03': pitf_group_params,
            f'{bravais_lattice}_4_02': pitf_group_params,
            f'{bravais_lattice}_4_03': pitf_group_params,
            }
        generator_info = [
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': 140},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': 140},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': 140},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': 140},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': 140},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': 140},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': 140},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': 140},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': 140},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': 140},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': 140},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': 140},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': 100},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': 100},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': 100},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': 100},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': 100},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': 100},
            {'generator': 'templates', 'n_unit_cells': 1000},
            #{'generator': 'random', 'n_unit_cells': 500},
            #{'generator': 'distribution_volume', 'n_unit_cells': 100},
            {'generator': 'predicted_volume', 'n_unit_cells': 1000},
            ]
    elif bravais_lattice == 'mP':
        reg_params = {
            f'{bravais_lattice}_0_00': reg_group_params,
            f'{bravais_lattice}_0_01': reg_group_params,
            f'{bravais_lattice}_1_00': reg_group_params,
            f'{bravais_lattice}_1_01': reg_group_params,
            f'{bravais_lattice}_4_00': reg_group_params,
            f'{bravais_lattice}_4_01': reg_group_params,
            }
        pitf_params = {
            f'{bravais_lattice}_0_00': pitf_group_params,
            f'{bravais_lattice}_0_01': pitf_group_params,
            f'{bravais_lattice}_1_00': pitf_group_params,
            f'{bravais_lattice}_1_01': pitf_group_params,
            f'{bravais_lattice}_4_00': pitf_group_params,
            f'{bravais_lattice}_4_01': pitf_group_params,
            }
        generator_info = [
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(f * 100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(f * 100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(f * 100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(f * 100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(f * 100)},
            {'generator': 'nn', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(f * 100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(f * 100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(f * 100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(f * 100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(f * 100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(f * 100)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(f * 100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(f * 100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(f * 100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(f * 100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(f * 100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(f * 100)},
            #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(f * 100)},
            {'generator': 'templates', 'n_unit_cells': int(f * 500)},
            #{'generator': 'random', 'n_unit_cells': int(f * 500)},
            #{'generator': 'distribution_volume', 'n_unit_cells': int(f * 100)},
            {'generator': 'predicted_volume', 'n_unit_cells': int(f * 500)},
            ]
    iteration_info = [{
        'worker': 'random_subsampling',
        'n_iterations': 100,
        'n_peaks': 20,
        'n_drop': 10,
        }]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 100,
        'neighbor_radius': 0.0005,
        }
    return OptimizerManager(data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm)


def get_triclinic_optimizer(bravais_lattice, broadening_tag, comm):
    data_params = {
        'tag': f'triclinic_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        }
    template_params = {bravais_lattice: {'tag': f'triclinic_{broadening_tag}'}}
    reg_group_params = {'tag': f'triclinic_{broadening_tag}'}
    pitf_group_params = {'tag': f'triclinic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'triclinic_{broadening_tag}'}}
    reg_params = {
        f'{bravais_lattice}_00': reg_group_params,
        }
    pitf_params = {
        f'{bravais_lattice}_00': pitf_group_params,
        }
    f = 1
    generator_info = [
        {'generator': 'nn', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(f * 1000)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(f * 1000)},
        #{'generator': 'pitf', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(f * 100)},
        {'generator': 'templates', 'n_unit_cells': int(f * 1000)},
        #{'generator': 'random', 'n_unit_cells': int(f * 1000)},
        #{'generator': 'distribution_volume', 'n_unit_cells': int(f * 100)},
        {'generator': 'predicted_volume', 'n_unit_cells': int(f * 1000)},
        ]
    iteration_info = [{
        'worker': 'random_subsampling',
        'n_iterations': 100,
        'n_peaks': 20,
        'n_drop': 10,
        }]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 50,
        'neighbor_radius': 0.003
        }
    return OptimizerManager(data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm)



def get_logger(comm):
    logger = logging.getLogger(f'rank[{comm.rank}]')
    logger.setLevel(logging.DEBUG)                                                 
    mh = MPIFileHandler('logfile.log')
    mh.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))                                                
    logger.addHandler(mh)
    return logger


def get_mpi_organizer(comm, bravais_lattices, manager_rank, serial):
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    mpi_organizer = namedtuple('mpi_organizer', ['manager', 'workers', 'color', 'split_comm'])
    mpi_organizers = dict.fromkeys(bravais_lattices)
    serial_split_comm = comm.Split(color=rank, key=0)
    for bl_index, bravais_lattice in enumerate(bravais_lattices):
        if serial[bl_index]:
            if rank == manager_rank[bl_index]:
                mpi_organizers[bravais_lattice] = mpi_organizer(
                    manager_rank[bl_index],
                    [manager_rank[bl_index]],
                    manager_rank[bl_index],
                    serial_split_comm
                    )
            else:
                mpi_organizers[bravais_lattice] = mpi_organizer(
                    manager_rank[bl_index],
                    [manager_rank[bl_index]],
                    rank,
                    None
                    )
        else:
            if rank == manager_rank[bl_index]:
                key = 0
            else:
                key = rank + 1
            mpi_organizers[bravais_lattice] = mpi_organizer(
                manager_rank[bl_index],
                [i for i in range(n_ranks)],
                bl_index,
                comm.Split(color=bl_index, key=key)
                )
    return mpi_organizers


def get_optimizers(rank, mpi_organizers, broadening_tag, logger=None):
    bravais_lattices = mpi_organizers.keys()
    optimizer = dict.fromkeys(bravais_lattices)
    for bl_index, bravais_lattice in enumerate(bravais_lattices):
        if rank == mpi_organizers[bravais_lattice].manager:
            # These function calls return an OptimizerManager object
            if bravais_lattice in ['cF', 'cI', 'cP']:
                optimizer[bravais_lattice] = get_cubic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    mpi_organizers[bravais_lattice].split_comm,
                    )
            elif bravais_lattice in ['hP']:
                optimizer[bravais_lattice] = get_hexagonal_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    mpi_organizers[bravais_lattice].split_comm,
                    )
            elif bravais_lattice in ['hR']:
                optimizer[bravais_lattice] = get_rhombohedral_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    mpi_organizers[bravais_lattice].split_comm,
                    )
            elif bravais_lattice in ['tI', 'tP']:
                optimizer[bravais_lattice] = get_tetragonal_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    mpi_organizers[bravais_lattice].split_comm,
                    )
            elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
                optimizer[bravais_lattice] = get_orthorhombic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    mpi_organizers[bravais_lattice].split_comm,
                    )
            elif bravais_lattice in ['mC', 'mP']:
                optimizer[bravais_lattice] = get_monoclinic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    mpi_organizers[bravais_lattice].split_comm,
                    )
            elif bravais_lattice in ['aP']:
                optimizer[bravais_lattice] = get_triclinic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    mpi_organizers[bravais_lattice].split_comm,
                    )
            if not logger is None:
                logger.info(f'Loaded manager optimizer for {bravais_lattice}')
        elif rank in mpi_organizers[bravais_lattice].workers:
            optimizer[bravais_lattice] = OptimizerWorker(mpi_organizers[bravais_lattice].split_comm)
            if not logger is None:
                logger.info(f'Loaded worker optimizer for {bravais_lattice}')
    return optimizer