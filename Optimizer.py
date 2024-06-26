import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.optimize
import scipy.spatial
import scipy.special
import time
from tqdm import tqdm

from Indexing import Indexing
from Reindexing import get_different_monoclinic_settings
from Reindexing import reindex_entry_triclinic
from Reindexing import get_s6_from_unit_cell
from TargetFunctions import CandidateOptLoss
from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_reciprocal_unit_cell_from_xnn
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import get_xnn_from_unit_cell
from Utilities import get_unit_cell_from_xnn
from Utilities import Q2Calculator
from Utilities import reciprocal_uc_conversion


class Candidates:
    def __init__(self, entry, unit_cell, lattice_system, bravais_lattice, minimum_unit_cell, maximum_unit_cell, tolerance, epsilon, max_neighbors, radius):
        self.q2_obs = np.array(entry['q2'])
        self.q2_obs_scaled = np.array(entry['q2_scaled'])
        self.n_points = self.q2_obs.size

        self.n = unit_cell.shape[0]
        self.n_uc = unit_cell.shape[1]

        self.lattice_system = lattice_system
        self.bravais_lattice = bravais_lattice
        self.minimum_unit_cell = minimum_unit_cell
        self.maximum_unit_cell = maximum_unit_cell
        if self.lattice_system == 'rhombohedral':
            self.maximum_angle = 2*np.pi/3
            self.minimum_angle = 0.01
        elif self.lattice_system in ['monoclinic', 'triclinic']:
            self.maximum_angle = np.pi
            self.minimum_angle = np.pi/2
        self.rng = np.random.default_rng()

        self.epsilon = epsilon
        self.min_loss = np.sum(np.log(np.sqrt(2*np.pi * self.q2_obs * epsilon)))
        self.tolerance = tolerance

        unit_cell_true = np.array(entry['reindexed_unit_cell'])
        reciprocal_unit_cell_true = reciprocal_uc_conversion(unit_cell_true[np.newaxis])[0]
        xnn_true = get_xnn_from_reciprocal_unit_cell(reciprocal_unit_cell_true[np.newaxis])[0]
        if lattice_system == 'cubic':
            self.unit_cell_true = unit_cell_true[0][np.newaxis]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[0][np.newaxis]
            self.xnn_true = xnn_true[0][np.newaxis]
        elif lattice_system in ['tetragonal', 'hexagonal']:
            self.unit_cell_true = unit_cell_true[[0, 2]]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[[0, 2]]
            self.xnn_true = xnn_true[[0, 2]]
        elif lattice_system == 'rhombohedral':
            self.unit_cell_true = unit_cell_true[[0, 3]]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[[0, 3]]
            self.xnn_true = xnn_true[[0, 3]]
        elif lattice_system == 'orthorhombic':
            self.unit_cell_true = unit_cell_true[:3]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[:3]
            self.xnn_true = xnn_true[:3]
        elif lattice_system == 'monoclinic':
            self.unit_cell_true = unit_cell_true[[0, 1, 2, 4]]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[[0, 1, 2, 4]]
            self.xnn_true = xnn_true[[0, 1, 2, 4]]
        elif lattice_system == 'triclinic':
            self.unit_cell_true = unit_cell_true
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true
            self.xnn_true = xnn_true

        self.hkl_true = np.array(entry['reindexed_hkl'])
        self.hkl_labels_true = np.array(entry['hkl_labels'])
        self.bl_true = entry['bravais_lattice']
        self.sg_true = int(entry['spacegroup_number'])
        self.spacegroup_symbol_hm_true = entry['reindexed_spacegroup_symbol_hm']

        self.df_columns = ['unit_cell', 'xnn', 'loss', 'candidate_index']
        self.explainers = pd.DataFrame(columns=self.df_columns)

        self.unit_cell = unit_cell
        self.update_xnn_from_unit_cell()
        self.hkl_true_check = get_hkl_matrix(self.hkl_true, self.lattice_system)
        self.best_loss = np.zeros(self.n)
        self.best_xnn = self.xnn.copy()
        self.xnn_for_reset = None
        self.candidate_index = np.arange(self.n)

        self.fix_out_of_range_candidates()
        # Exhaustive search parameters
        self.max_neighbors = max_neighbors
        self.radius = radius
        self.redistribute_candidates()
        self.starting_xnn = self.xnn.copy()

    def fix_bad_conversions(self):
        bad_conversions = np.sum(np.isnan(self.reciprocal_unit_cell), axis=1) > 0
        good_indices = np.arange(self.n)[~bad_conversions]
        n_bad = np.sum(bad_conversions)
        if n_bad > 0:
            if n_bad > bad_conversions.size - n_bad:
                good_indices = self.rng.choice(good_indices, replace=True, size=n_bad)
            else:
                good_indices = self.rng.choice(good_indices, replace=False, size=n_bad)
            self.xnn[bad_conversions] = self.xnn[good_indices]
            self.reciprocal_unit_cell[bad_conversions] = self.reciprocal_unit_cell[good_indices]
            self.unit_cell[bad_conversions] = self.unit_cell[good_indices]

    def update_xnn_from_unit_cell(self):
        self.reciprocal_unit_cell = reciprocal_uc_conversion(
            self.unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.xnn = get_xnn_from_reciprocal_unit_cell(
            self.reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.fix_bad_conversions()

    def update_unit_cell_from_xnn(self):
        self.reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            self.xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.unit_cell = reciprocal_uc_conversion(
            self.reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.fix_bad_conversions()

    def diagnostics(self, hkl_ref_length):
        reciprocal_unit_cell_rms = 1/np.sqrt(self.n_uc) * np.linalg.norm(self.reciprocal_unit_cell - self.reciprocal_unit_cell_true, axis=1)
        reciprocal_unit_cell_max_diff = np.max(np.abs(self.reciprocal_unit_cell - self.reciprocal_unit_cell_true), axis=1)

        hkl_pred_check = get_hkl_matrix(self.hkl, self.lattice_system)
        hkl_correct = self.hkl_true_check[np.newaxis] == hkl_pred_check
        hkl_accuracy = np.count_nonzero(hkl_correct, axis=1) / self.n_points

        impossible = np.any(self.hkl_labels_true == hkl_ref_length - 1)

        if self.lattice_system in ['monoclinic', 'orthorhombic', 'triclinic']:
            h_info = self.n_points - np.sum(self.hkl_true[:, 0]**2 == 0)
            k_info = self.n_points - np.sum(self.hkl_true[:, 1]**2 == 0)
            l_info = self.n_points - np.sum(self.hkl_true[:, 2]**2 == 0)
            dominant_axis_info = np.sort([h_info, k_info, l_info])[1]
            dominant_zone_info = np.min([h_info, k_info, l_info])
        elif self.lattice_system == 'tetragonal':
            hk = self.hkl_true[:, 0]**2 + self.hkl_true[:, 1]**2
            hk_info = self.n_points - np.sum(hk == 0)
            l2_info = self.n_points - np.sum(self.hkl_true[:, 2]**2 == 0)
            dominant_axis_info = min(hk_info, l2_info)
            dominant_zone_info = self.n_points
        elif self.lattice_system == 'hexagonal':
            hk = self.hkl_true[:, 0]**2 + self.hkl_true[:, 0]*self.hkl_true[:, 1] + self.hkl_true[:, 1]**2
            hk_info = self.n_points - np.sum(hk == 0)
            l2_info = self.n_points - np.sum(self.hkl_true[:, 2]**2 == 0)
            dominant_axis_info = min(hk_info, l2_info)
            dominant_zone_info = self.n_points
        elif self.lattice_system in ['cubic', 'rhombohedral']:
            dominant_axis_info = self.n_points
            dominant_zone_info = self.n_points

        if self.lattice_system == 'monoclinic':
            unit_cell_true = get_different_monoclinic_settings(self.unit_cell_true, partial_unit_cell=True)
            reciprocal_unit_cell_true = reciprocal_uc_conversion(
                unit_cell_true, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            xnn_true = get_xnn_from_reciprocal_unit_cell(
                reciprocal_unit_cell_true, partial_unit_cell=True, lattice_system=self.lattice_system
                )

            distance_ruc = np.linalg.norm(
                self.reciprocal_unit_cell[np.newaxis, :, :3] - reciprocal_unit_cell_true[:, np.newaxis, :3], 
                axis=2
                ).min(axis=0)
            distance_uc = np.linalg.norm(
                self.unit_cell[np.newaxis, :, :3] - unit_cell_true[:, np.newaxis, :3],
                axis=2
                ).min(axis=0)
            distance_xnn = np.linalg.norm(
                self.xnn[np.newaxis, :, :3] - xnn_true[:, np.newaxis, :3],
                axis=2
                ).min(axis=0)
        elif self.lattice_system == 'triclinic':
            distance_ruc = np.linalg.norm(
                self.reciprocal_unit_cell[:, :3] - self.reciprocal_unit_cell_true[np.newaxis, :3], 
                axis=1
                )
            distance_uc = np.linalg.norm(
                self.unit_cell[:, :3] - self.unit_cell_true[np.newaxis, :3],
                axis=1
                )
            distance_xnn = np.linalg.norm(self.xnn - self.xnn_true[np.newaxis], axis=1)
            s6 = get_s6_from_unit_cell(self.unit_cell)
            s6_true = get_s6_from_unit_cell(self.unit_cell_true[np.newaxis])[0]
            distance_s6 = np.linalg.norm(s6 - s6_true[np.newaxis], axis=1)
        elif self.lattice_system == 'rhombohedral':
            distance_ruc = np.abs(self.reciprocal_unit_cell[:, 0] - self.reciprocal_unit_cell_true[np.newaxis, 0])
            distance_uc = np.abs(self.unit_cell[:, 0] - self.unit_cell_true[np.newaxis, 0])
            distance_xnn = np.abs(self.xnn[:, 0] - self.xnn_true[np.newaxis, 0])
        else:
            distance_ruc = np.linalg.norm(
                self.reciprocal_unit_cell - self.reciprocal_unit_cell_true[np.newaxis], axis=1
                )
            distance_uc = np.linalg.norm(self.unit_cell - self.unit_cell_true[np.newaxis], axis=1)
            distance_xnn = np.linalg.norm(self.xnn - self.xnn_true[np.newaxis], axis=1)
        counts_ruc = [np.sum(distance_ruc < i) for i in [0.005, 0.01, 0.02]]
        counts_uc = [np.sum(distance_uc < i) for i in [1, 2, 3]]
        counts_xnn = [np.sum(distance_xnn < i) for i in [0.003, 0.004, 0.005]]
        if self.lattice_system == 'triclinic':
            counts_s6 = [np.sum(distance_s6 < i) for i in [25, 50, 75]]

        print(f'Starting # candidates:       {self.n}')
        print(f'Minimum Loss:                {np.round(self.min_loss, decimals=2)}')
        print(f'Impossible:                  {impossible}')
        print(f'True dominant axis info:     {dominant_axis_info}')
        print(f'True dominant zone info:     {dominant_zone_info}')
        print(f'True unit cell:              {np.round(self.unit_cell_true, decimals=4)}')
        print(f'True reciprocal unit cell:   {np.round(self.reciprocal_unit_cell_true, decimals=4)}')
        print(f'True Xnn:                    {np.round(self.xnn_true, decimals=4)}')
        print(f'Closest unit cell:           {np.round(self.reciprocal_unit_cell[np.argmin(reciprocal_unit_cell_rms)], decimals=4)}')
        print(f'Closest unit cell rms:       {reciprocal_unit_cell_rms.min():2.2f}')
        print(f'Smallest unit cell max diff: {reciprocal_unit_cell_max_diff.min():2.2f}')
        print(f'Mean unit cell rms:          {reciprocal_unit_cell_rms.mean():2.2f}')
        print(f'Best HKL accuracy:           {hkl_accuracy.max():1.2f}')
        print(f'Mean HKL accuracy:           {hkl_accuracy.mean():1.2f}')
        print(f'Close Unit Cell:             {counts_uc[0]}, {counts_uc[1]}, {counts_uc[2]}')
        print(f'Close Reciprocal UC:         {counts_ruc[0]}, {counts_ruc[1]}, {counts_ruc[2]}')
        print(f'Close Xnn:                   {counts_xnn[0]}, {counts_xnn[1]}, {counts_xnn[2]}')
        if self.lattice_system == 'triclinic':
            print(f'Close S6:                    {counts_s6[0]}, {counts_s6[1]}, {counts_s6[2]}')
        print(f'Bravais Lattice:             {self.bl_true}')
        print(f'Spacegroup:                  {self.sg_true} {self.spacegroup_symbol_hm_true}')

        output_dict = {
            'entry_index': None,
            'true_unit_cell': self.reciprocal_unit_cell_true,
            'closest_unit_cell': self.reciprocal_unit_cell[np.argmin(reciprocal_unit_cell_rms)],
            'best_hkl_accuracy': hkl_accuracy.max(),
            'mean_hkl_accuracy': hkl_accuracy.mean(),
            'bravais_lattice': self.bl_true,
            'spacegroup': self.sg_true,
            'impossible': impossible,
            'found': False,
            'counts_uc': counts_uc,
            'counts_ruc': counts_ruc,
            'counts_xnn': counts_xnn,
            'dominant_axis_info': dominant_axis_info,
            'dominant_zone_info': dominant_zone_info,
            }

        return pd.Series(output_dict)

    def fix_out_of_range_candidates(self):
        self.xnn = fix_unphysical(
            xnn=self.xnn,
            rng=self.rng,
            minimum_unit_cell=self.minimum_unit_cell, 
            maximum_unit_cell=self.maximum_unit_cell,
            lattice_system=self.lattice_system,
            )
        self.update_unit_cell_from_xnn()

    def get_neighbor_distance(self, xnn0, xnn1):
        # scipy.spatial is considerably better than a pure numpy distance
        # Faster and less memory load
        distance = scipy.spatial.distance.cdist(xnn0, xnn1)
        return distance

    def redistribute_and_perturb_xnn(self, xnn, from_indices, to_indices, norm_factor=None):
        n_indices = from_indices.size
        if not norm_factor is None:
            norm_factor = self.rng.uniform(low=norm_factor[0], high=norm_factor[1], size=n_indices)
        else:
            norm_factor = np.ones(n_indices)

        perturbation = self.rng.uniform(low=-1, high=1, size=(n_indices, self.n_uc))
        perturbation *= (self.radius*norm_factor/np.linalg.norm(perturbation, axis=1))[:, np.newaxis]
        xnn[from_indices] = xnn[to_indices] + perturbation
        xnn = fix_unphysical(
            xnn=xnn,
            rng=self.rng,
            minimum_unit_cell=self.minimum_unit_cell, 
            maximum_unit_cell=self.maximum_unit_cell,
            lattice_system=self.lattice_system
            )

        # Enforce the constraints on the unit cells.
        # This is the reindexing task
        if self.lattice_system == 'monoclinic':
            # a < c can be enforced
            # Order lattice
            unit_cell = get_unit_cell_from_xnn(xnn, partial_unit_cell=True, lattice_system=self.lattice_system)
            permute_indices = unit_cell[:, 0] > unit_cell[:, 2]
            unit_cell[permute_indices] = np.column_stack((
                unit_cell[permute_indices, 2],
                unit_cell[permute_indices, 1],
                unit_cell[permute_indices, 0],
                unit_cell[permute_indices, 3],
                ))
            xnn = get_xnn_from_unit_cell(unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        elif self.lattice_system == 'orthorhombic':
            if self.bravais_lattice in ['oF', 'oI', 'oP']:
                order = np.argsort(xnn, axis=1)[:, ::-1]
                xnn = np.take_along_axis(xnn, order, axis=1)
            elif self.bravais_lattice == 'oC':
                permute_indices = xnn[:, 0] < xnn[:, 1]
                xnn[permute_indices] = np.column_stack((
                    xnn[permute_indices, 1],
                    xnn[permute_indices, 0],
                    xnn[permute_indices, 2],
                    ))
        elif self.lattice_system == 'triclinic':
            unit_cell = get_unit_cell_from_xnn(xnn)
            unit_cell, _ = reindex_entry_triclinic(unit_cell)
            xnn = get_xnn_from_unit_cell(unit_cell)
        return xnn

    def redistribute_candidates(self):
        redistributed_xnn = self.xnn.copy()
        largest_neighborhood = self.max_neighbors + 1
        n_redistributed = 0
        iteration = 0
        while largest_neighborhood > self.max_neighbors and iteration < 10:
            distance = self.get_neighbor_distance(redistributed_xnn, redistributed_xnn)
            neighbor_array = distance < self.radius
            neighbor_count = np.sum(neighbor_array, axis=1)
            largest_neighborhood = neighbor_count.max()
            if largest_neighborhood > self.max_neighbors:
                from_indices = []
                high_density_indices = np.where(neighbor_count > self.max_neighbors)[0]
                for high_density_index in high_density_indices:
                    neighbor_indices = np.where(neighbor_array[high_density_index])[0]
                    excess_neighbors = neighbor_indices.size - self.max_neighbors
                    if excess_neighbors <= neighbor_indices.size:
                        replace = False
                    else:
                        replace = True
                    from_indices.append(neighbor_indices[
                        self.rng.choice(neighbor_indices.size, size=excess_neighbors, replace=replace)
                        ])
                from_indices = np.unique(np.concatenate(from_indices))
                excess_neighbors = from_indices.size
                n_redistributed += excess_neighbors
        
                low_density_indices = np.where(neighbor_count < self.max_neighbors)[0]
                if low_density_indices.size == 0:
                    break
                prob = self.max_neighbors - neighbor_count[low_density_indices]
                prob = prob / prob.sum()
                if excess_neighbors <= low_density_indices.size:
                    replace = False
                else:
                    replace = True
                to_indices = low_density_indices[
                    self.rng.choice(low_density_indices.size, size=excess_neighbors, replace=replace, p=prob)
                    ]
                redistributed_xnn = self.redistribute_and_perturb_xnn(
                    redistributed_xnn, from_indices, to_indices
                    )
            iteration += 1
        self.xnn = redistributed_xnn
        self.update_unit_cell_from_xnn()
        print(f'Redistributed {n_redistributed} candidates')

    def exhaustive_search(self):
        if self.lattice_system == 'triclinic':
            self.unit_cell, _ = reindex_entry_triclinic(get_unit_cell_from_xnn(self.xnn))
            self.xnn = get_xnn_from_unit_cell(self.unit_cell)

        # This get the best candidate during this round for use in monoclinic_reset()
        if self.xnn_for_reset is None:
            self.xnn_for_reset = self.best_xnn.copy()
        else:
            self.xnn_for_reset = np.row_stack((self.xnn_for_reset, self.best_xnn))

        distance = self.get_neighbor_distance(self.xnn, self.starting_xnn)
        neighbor_array = distance < self.radius
        neighbor_count = np.sum(neighbor_array, axis=1)
        from_indices = np.where(neighbor_count > self.max_neighbors)[0]
        excess_neighbors = from_indices.size
        low_density_indices = np.where(neighbor_count < self.max_neighbors)[0]

        if low_density_indices.size > 0:
            prob = self.max_neighbors - neighbor_count[low_density_indices]
            prob = prob / prob.sum()
            if excess_neighbors <= low_density_indices.size:
                replace = False
            else:
                replace = True
            choice = self.rng.choice(low_density_indices.size, size=excess_neighbors, replace=replace, p=prob)
            to_indices = low_density_indices[choice]
            self.xnn = self.redistribute_and_perturb_xnn(self.xnn, from_indices, to_indices)

            not_chosen = np.delete(np.arange(low_density_indices.size), choice)
            not_chosen_indices = low_density_indices[not_chosen]
            
            self.xnn = self.redistribute_and_perturb_xnn(
                self.xnn, not_chosen_indices, not_chosen_indices, norm_factor=[0.25, 0.75]
                )
        else:
            not_chosen_indices = np.ones(self.xnn.shape[0], dtype=bool)
            self.xnn = self.redistribute_and_perturb_xnn(
                self.xnn, not_chosen_indices, not_chosen_indices, norm_factor=[0.25, 0.75]
                )

        print(f'Exhaustive search redistributed {excess_neighbors} candidates')
        self.starting_xnn = np.row_stack((self.starting_xnn, self.xnn))
        self.update_unit_cell_from_xnn()
        self.best_xnn = self.xnn.copy()
        self.best_loss = np.zeros(self.xnn.shape[0])
        start = self.candidate_index.max() + 1
        self.candidate_index = np.arange(start, start + self.n)
        self.fix_out_of_range_candidates()

    def pick_explainers(self):
        found = self.loss < self.tolerance*self.min_loss
        if np.count_nonzero(found) > 0:
            # If I keep the 'hkl' column I get an error:
            #  ValueError: all the input array dimensions except for the concatenation axis must match exactly
            # I believe this is due to a data type mismatch. The easiest way to deal with this was to drop the column
            if len(self.explainers) == 0:
                self.explainers = pd.DataFrame({
                    'unit_cell': list(self.unit_cell[found]),
                    'xnn': list(self.xnn[found]),
                    'loss': self.loss[found],
                    'candidate_index': self.candidate_index[found],
                    })
            else:
                found_index = self.candidate_index[found]
                explainers_index = np.array(self.explainers['candidate_index'])
                new = np.ones(np.sum(found), dtype=bool)
                for index in range(np.sum(found)):
                    if found_index[index] in explainers_index:
                        new[index] = False
                if np.sum(new) > 0:
                    new_explainers = pd.DataFrame({
                        'unit_cell': list(self.unit_cell[found][new]),
                        'xnn': list(self.xnn[found][new]),
                        'loss': self.loss[found][new],
                        'candidate_index': self.candidate_index[found][new],
                        })
                    self.explainers = pd.concat(
                        [self.explainers, new_explainers], ignore_index=False
                        )
                    self.explainers.sort_values(by='loss', inplace=True)

    def validate_candidate(self, unit_cell):
        atol = 1e-2
        if self.lattice_system == 'cubic':
            if np.isclose(self.unit_cell_true, unit_cell, atol=atol):
                return True, False
            mult_factors = np.array([1/2, 2])
            for mf in mult_factors:
                if np.isclose(self.unit_cell_true, mf * unit_cell, atol=atol):
                    return False, True
        elif self.lattice_system in ['tetragonal', 'hexagonal']:
            if np.all(np.isclose(self.unit_cell_true, unit_cell, atol=atol)):
                return True, False
            mult_factors = np.array([1/3, 1/2, 1, 2, 3])
            for mf0 in mult_factors:
                for mf1 in mult_factors:
                    mf = np.array([mf0, mf1])
                    if np.all(np.isclose(self.unit_cell_true, mf * unit_cell, atol=atol)):
                        return False, True
        elif self.lattice_system == 'rhombohedral':
            if np.all(np.isclose(self.unit_cell_true, unit_cell, atol=atol)):
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
            ax = unit_cell[0]
            bx = unit_cell[0]*np.cos(unit_cell[1])
            by = unit_cell[0]*np.sin(unit_cell[1])
            cx = unit_cell[0]*np.cos(unit_cell[1])
            arg = (np.cos(unit_cell[1]) - np.cos(unit_cell[1])**2) / np.sin(unit_cell[1])
            cy = unit_cell[0] * arg
            cz = unit_cell[0] * np.sqrt(np.sin(unit_cell[1])**2 - arg**2)
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
                if np.all(np.isclose(self.unit_cell_true, reindexed_unit_cell, atol=atol)):
                    found = True
                mult_factors = np.array([1/2, 2])
                for mf in mult_factors:
                    if np.all(np.isclose(self.unit_cell_true, np.array([mf, 1]) * reindexed_unit_cell, atol=atol)):
                        off_by_two = True
            return found, off_by_two
        elif self.lattice_system == 'orthorhombic':
            unit_cell_true_sorted = np.sort(self.unit_cell_true)
            unit_cell_sorted = np.sort(unit_cell)
            if np.all(np.isclose(unit_cell_true_sorted, unit_cell_sorted, atol=atol)):
                return True, False
            mult_factors = np.array([1/2, 1, 2])
            for mf0 in mult_factors:
                for mf1 in mult_factors:
                    for mf2 in mult_factors:
                        mf = np.array([mf0, mf1, mf2])
                        if np.all(np.isclose(unit_cell_true_sorted, np.sort(mf * unit_cell), atol=atol)):
                            return False, True
        elif self.lattice_system == 'monoclinic':
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
                [unit_cell[0], 0,            unit_cell[2] * np.cos(unit_cell[3])],
                [0,            unit_cell[1], 0],
                [0,            0,            unit_cell[2] * np.sin(unit_cell[3])],
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
                        if np.all(np.isclose(self.unit_cell_true, reindexed_unit_cell, atol=atol)):
                            found = True
                        mult_factors = np.array([1/2, 1, 2])
                        for mf0 in mult_factors:
                            for mf1 in mult_factors:
                                for mf2 in mult_factors:
                                    mf = np.array([mf0, mf1, mf2, 1])
                                    if np.all(np.isclose(self.unit_cell_true, mf * reindexed_unit_cell, atol=atol)):
                                        off_by_two = True
            return found, off_by_two
        elif self.lattice_system == 'triclinic':
            reindexed_unit_cell, _ = reindex_entry_triclinic(unit_cell)
            found = False
            off_by_two = False
            if np.all(np.isclose(self.unit_cell_true, unit_cell, atol=atol)):
                found = True
            mult_factors = np.array([1/2, 1, 2])
            for mf0 in mult_factors:
                for mf1 in mult_factors:
                    for mf2 in mult_factors:
                        mf = np.array([mf0, mf1, mf2, 1, 1, 1])
                        if np.all(np.isclose(self.unit_cell_true, mf * reindexed_unit_cell, atol=atol)):
                            off_by_two = True
            return found, off_by_two
        return False, False

    def get_best_candidates(self, report_counts):
        found = False
        found_best = False
        found_not_best = False
        found_off_by_two = False

        if len(self.explainers) == 0:
            best_unit_cell = get_unit_cell_from_xnn(
                self.best_xnn,
                partial_unit_cell=True,
                lattice_system=self.lattice_system
                )
            indices = np.argsort(self.best_loss)[:20]
            unit_cell = best_unit_cell[indices]
            loss = self.best_loss[indices]
        else:
            found = True
            unit_cell = np.stack(self.explainers['unit_cell'])
            loss = np.array(self.explainers['loss'])

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

        print(np.concatenate((
            unit_cell.round(decimals=3), loss.round(decimals=1)[:, np.newaxis]
            ),
            axis=1))

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

    def monoclinic_reset(self, n_best, n_angles):
        if self.ending_unit_cells is None:
            reciprocal_unit_cell = np.stack(self.candidates['reciprocal_unit_cell'])
        else:
            reciprocal_unit_cell = reciprocal_uc_conversion(
                self.ending_unit_cells, partial_unit_cell=True, lattice_system=self.lattice_system
                )
        lengths = np.round(reciprocal_unit_cell[:, :3].ravel(), decimals=4)
        unique, counts = np.unique(lengths, return_counts=True)
        sort_indices = np.argsort(counts)[::-1]
        unique = unique[sort_indices]
        counts = counts[sort_indices]

        n_configurations = n_best * (n_best-1) * (n_best-2)
        new_reciprocal_unit_cell = np.zeros((n_configurations, 4))
        index = 0
        for ai in range(n_best):
            bi_indices = np.delete(np.arange(n_best), ai)
            for bi in bi_indices:
                if bi > ai:
                    ci_indices = np.delete(bi_indices, bi - 1)
                else:
                    ci_indices = np.delete(bi_indices, bi)
                for ci in ci_indices:
                    new_reciprocal_unit_cell[index, :3] = [unique[ai], unique[bi], unique[ci]]
                    index += 1

        new_reciprocal_unit_cell = np.repeat(new_reciprocal_unit_cell, repeats=n_angles, axis=0)
        for index, new_angle in enumerate(np.arccos(np.linspace(0, 0.99, n_angles))):
            start = index * n_configurations
            stop = (index + 1) * n_configurations
            new_reciprocal_unit_cell[start: stop, 3] = new_angle

        self.candidates = pd.DataFrame(columns=self.df_columns)
        self.candidates['reciprocal_unit_cell'] = list(new_reciprocal_unit_cell)
        self.n = n_configurations * n_angles
        reciprocal_unit_cell_full = np.zeros((self.n, 6))
        reciprocal_unit_cell_full[:, :3] = new_reciprocal_unit_cell[:, :3]
        reciprocal_unit_cell_full[:, 4] = new_reciprocal_unit_cell[:, 3]
        reciprocal_unit_cell_full[:, 3] = np.pi/2
        reciprocal_unit_cell_full[:, 5] = np.pi/2

        xnn_full = get_xnn_from_reciprocal_unit_cell(reciprocal_unit_cell_full)
        xnn = xnn_full[:, [0, 1, 2, 4]]
        self.candidates['xnn'] = list(xnn)

        unit_cell_full = reciprocal_uc_conversion(reciprocal_unit_cell_full)
        unit_cell = unit_cell_full[:, [0, 1, 2, 4]]
        self.candidates['unit_cell'] = list(unit_cell)


class Optimizer:
    def __init__(self, assign_params, data_params, opt_params, reg_params, template_params, bravais_lattice, seed=12345):
        self.assign_params = assign_params
        self.data_params = data_params
        self.opt_params = opt_params
        self.reg_params = reg_params
        self.template_params = template_params
        self.bravais_lattice = bravais_lattice
        self.rng = np.random.default_rng(seed)

        opt_params_defaults = {
            'n_candidates_nn': 320,
            'n_candidates_rf': 80,
            'n_candidates_template': 1920,
            'minimum_uc': 2,
            'maximum_uc': 500,
            'found_tolerance': -240,
            'load_predictions': False,
            'max_explainers': 20,
            }
        for key in opt_params_defaults.keys():
            if key not in self.opt_params.keys():
                self.opt_params[key] = opt_params_defaults[key]
        for key in self.assign_params[self.bravais_lattice]:
            self.assign_params[self.bravais_lattice][key]['load_from_tag'] = True
            self.assign_params[self.bravais_lattice][key]['mode'] = 'inference'
        for key in self.reg_params:
            self.reg_params[key]['load_from_tag'] = True
            self.reg_params[key]['alpha_params'] = {}
            self.reg_params[key]['beta_params'] = {}
            self.reg_params[key]['mean_params'] = {}
            self.reg_params[key]['var_params'] = {}
            self.reg_params[key]['head_params'] = {}
        self.data_params['load_from_tag'] = True
        self.template_params[self.bravais_lattice]['load_from_tag'] = True

        self.save_to = os.path.join(
            self.data_params['base_directory'], 'models', self.data_params['tag'], 'optimizer'
            )
        if not os.path.exists(self.save_to):
            os.mkdir(self.save_to)
        figure_directory = os.path.join(self.save_to, 'figures')
        if not os.path.exists(figure_directory):
            os.mkdir(figure_directory)

        self.indexer = Indexing(
            assign_params=self.assign_params, 
            data_params=self.data_params,
            reg_params=self.reg_params, 
            template_params=self.template_params, 
            seed=12345, 
            )
        self.indexer.setup_from_tag()
        self.indexer.load_data_from_tag(
            load_augmented=False,
            load_train=False,
            load_bravais_lattice=self.bravais_lattice
            )
        self.indexer.setup_regression()
        self.indexer.setup_miller_index_templates()

        self.opt_params['minimum_uc_scaled'] = \
            (self.opt_params['minimum_uc'] - self.indexer.uc_scaler.mean_[0]) / self.indexer.uc_scaler.scale_[0]
        self.opt_params['maximum_uc_scaled'] = \
            (self.opt_params['maximum_uc'] - self.indexer.uc_scaler.mean_[0]) / self.indexer.uc_scaler.scale_[0]

        if self.indexer.data_params['lattice_system'] == 'rhombohedral':
            # Maximum angle = 120 degrees
            # 120 degrees is a hard maximum because the reciprocal space conversion fails
            # The numerically stable lower angle limit is 0.5 degrees
            # for the reciprocal space conversion.
            self.opt_params['maximum_angle_scaled'] = np.cos(0.01)
            self.opt_params['minimum_angle_scaled'] = np.cos(2*np.pi/3)
        elif self.indexer.data_params['lattice_system'] in ['monoclinic', 'triclinic']:
            # Minimum / Maximum angle = 90 / 180 degrees
            # Monoclinic angles are restricted to be above 90 degrees because
            # the same monoclinic unit cell can be represented with either an
            # obtuse or acute angle.
            # Scaling is np.cos(angle)
            self.opt_params['maximum_angle_scaled'] = 0
            self.opt_params['minimum_angle_scaled'] = -1

        self.n_groups = len(self.indexer.data_params['split_groups'])
        self.q2_calculator = Q2Calculator(
            lattice_system=self.indexer.data_params['lattice_system'],
            hkl=self.indexer.hkl_ref[self.bravais_lattice],
            tensorflow=False,
            representation='xnn'
            )

    def predictions(self):
        self.N = self.indexer.data.shape[0]
        uc_scaled_mean_filename = os.path.join(
            f'{self.save_to}', f'{self.bravais_lattice}_{self.data_params["tag"]}_uc_scaled_mean.npy'
            )
        uc_scaled_var_filename = os.path.join(
            f'{self.save_to}', f'{self.bravais_lattice}_{self.data_params["tag"]}_uc_scaled_var.npy'
            )
        if self.opt_params['load_predictions']:
            self.uc_scaled_mean = np.load(uc_scaled_mean_filename)
            self.uc_scaled_var = np.load(uc_scaled_var_filename)
        else:
            self.uc_scaled_mean = np.zeros((self.N, self.n_groups, self.indexer.data_params['n_outputs']))
            self.uc_scaled_var = np.zeros((self.N, self.n_groups, self.indexer.data_params['n_outputs']))
            for group_index, group in enumerate(self.indexer.data_params['split_groups']):
                print(f'Performing predictions with {group}')
                uc_mean_scaled_group, uc_var_scaled_group = self.indexer.unit_cell_generator[group].do_predictions(
                    data=self.indexer.data, verbose=0, batch_size=2048
                    )
                self.uc_scaled_mean[:, group_index, :] = uc_mean_scaled_group
                self.uc_scaled_var[:, group_index, :] = uc_var_scaled_group
            np.save(uc_scaled_mean_filename, self.uc_scaled_mean)
            np.save(uc_scaled_var_filename, self.uc_scaled_var)

    def evaluate_regression(self, N_entries, n_candidates_steps, n_evaluations, threshold):
        candidates_per_model = min(
            len(self.indexer.data_params['split_groups'])*self.opt_params['n_candidates_nn'],
            len(self.indexer.data_params['split_groups'])*self.opt_params['n_candidates_rf'],
            self.opt_params['n_candidates_template'],
            )
        candidate_steps = np.round(
            np.linspace(10, candidates_per_model, n_candidates_steps),
            decimals=0
            ).astype(int)

        efficiency = np.zeros((N_entries, n_candidates_steps, 7))
        failure_rate = np.zeros((N_entries, n_candidates_steps, 7))
        print(f'Performing evaluation assuming each group generates {candidates_per_model} candidates')
        for entry_index in tqdm(range(N_entries)):
            entry = self.indexer.data.iloc[entry_index]
            for group_index, group in enumerate(self.indexer.data_params['split_groups']):
                candidate_uc_nn = self.generate_unit_cells(
                    self.uc_scaled_mean[entry_index, group_index, :],
                    self.uc_scaled_var[entry_index, group_index, :]
                    )

                # Get candidates from the random forest model
                q2_scaled = np.array(entry['q2_scaled'])[np.newaxis]
                # candidates_scaled_tree: n_entries, n_outputs, n_trees
                #   n_entries = 1 because there is only one q2_scaled input
                _, _, candidates_scaled_tree = \
                    self.indexer.unit_cell_generator[group].do_predictions_trees(q2_scaled=q2_scaled)
                tree_indices = self.rng.choice(
                    candidates_scaled_tree.shape[2],
                    size=self.opt_params['n_candidates_rf'],
                    replace=False
                    )
                candidate_uc_rf = self.indexer.revert_predictions(
                    uc_pred_scaled=candidates_scaled_tree[0, :, tree_indices]
                    )

            candidate_uc_tm = self.generate_unit_cells_miller_index_templates(np.array(entry['q2']))

            candidate_uc_nn = fix_unphysical(
                unit_cell=candidate_uc_nn,
                rng=self.rng,
                minimum_unit_cell=self.opt_params['minimum_uc'],
                maximum_unit_cell=self.opt_params['maximum_uc'],
                lattice_system=self.indexer.data_params['lattice_system']
                )
            candidate_uc_rf = fix_unphysical(
                unit_cell=candidate_uc_rf,
                rng=self.rng,
                minimum_unit_cell=self.opt_params['minimum_uc'],
                maximum_unit_cell=self.opt_params['maximum_uc'],
                lattice_system=self.indexer.data_params['lattice_system']
                )
            candidate_uc_tm = fix_unphysical(
                unit_cell=candidate_uc_tm,
                rng=self.rng,
                minimum_unit_cell=self.opt_params['minimum_uc'],
                maximum_unit_cell=self.opt_params['maximum_uc'],
                lattice_system=self.indexer.data_params['lattice_system']
                )
            if self.indexer.data_params['lattice_system'] == 'triclinic':
                candidate_uc_nn, _ = reindex_entry_triclinic(candidate_uc_nn)
                candidate_uc_rf, _ = reindex_entry_triclinic(candidate_uc_rf)
                candidate_uc_tm, _ = reindex_entry_triclinic(candidate_uc_tm)
            
            reindexed_unit_cell_true = np.array(entry['reindexed_unit_cell'])
            xnn_true = np.array(entry['reindexed_xnn'])
            candidate_xnn_nn = get_xnn_from_unit_cell(
                candidate_uc_nn,
                partial_unit_cell=True,
                lattice_system=self.indexer.data_params['lattice_system']
                )
            candidate_xnn_rf = get_xnn_from_unit_cell(
                candidate_uc_rf,
                partial_unit_cell=True,
                lattice_system=self.indexer.data_params['lattice_system']
                )
            candidate_xnn_tm = get_xnn_from_unit_cell(
                candidate_uc_tm,
                partial_unit_cell=True,
                lattice_system=self.indexer.data_params['lattice_system']
                )
            distance_nn_all = np.linalg.norm(candidate_xnn_nn - xnn_true[np.newaxis], axis=1)
            distance_rf_all = np.linalg.norm(candidate_xnn_rf - xnn_true[np.newaxis], axis=1)
            distance_tm_all = np.linalg.norm(candidate_xnn_tm - xnn_true[np.newaxis], axis=1)
            for step_index, step_size in enumerate(candidate_steps):
                efficiency_step = np.zeros((n_evaluations, 7))
                for eval_index in range(n_evaluations):
                    indices = self.rng.choice(candidates_per_model, step_size, replace=False)
                    distance_nn = distance_nn_all[indices]
                    distance_rf = distance_rf_all[indices]
                    distance_tm = distance_tm_all[indices]
                    efficiency_step[eval_index, 0] = np.sum(distance_nn < threshold) / step_size
                    efficiency_step[eval_index, 1] = np.sum(distance_rf < threshold) / step_size
                    efficiency_step[eval_index, 2] = np.sum(distance_tm < threshold) / step_size
                    efficiency_step[eval_index, 3] = (np.sum(distance_nn < threshold) + np.sum(distance_rf < threshold)) / (2*step_size)
                    efficiency_step[eval_index, 4] = (np.sum(distance_nn < threshold) + np.sum(distance_tm < threshold)) / (2*step_size)
                    efficiency_step[eval_index, 5] = (np.sum(distance_rf < threshold) + np.sum(distance_tm < threshold)) / (2*step_size)
                    efficiency_step[eval_index, 6] = (np.sum(distance_nn < threshold) + np.sum(distance_rf < threshold) + np.sum(distance_tm < threshold)) / (3*step_size)

                efficiency[entry_index, step_index, :] = efficiency_step.mean(axis=0)
                failure_rate[entry_index, step_index, :] = np.sum(efficiency_step == 0, axis=0) / n_evaluations
        color_cycle_indices = [0, 1, 2, 3, 9, 8, 5, 6, 7, 4]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
        labels = ['NN', 'RF', 'Temp', 'NN & RF', 'NN & Temp', 'RF & Temp', 'All']
        for i in range(3):
            axes[0, 0].plot(candidate_steps, 100 * efficiency[:, :, i].mean(axis=0), label=labels[i], color=colors[color_cycle_indices[i]])
            axes[0, 1].plot(candidate_steps, 100 * failure_rate[:, :, i].mean(axis=0), label=labels[i], color=colors[color_cycle_indices[i]])
        for i in range(3, 7):
            axes[1, 0].plot(candidate_steps, 100 * efficiency[:, :, i].mean(axis=0), label=labels[i], color=colors[color_cycle_indices[i]])
            axes[1, 1].plot(candidate_steps, 100 * failure_rate[:, :, i].mean(axis=0), label=labels[i], color=colors[color_cycle_indices[i]])

        axes[1, 0].set_xlabel('Number of candidates')
        axes[1, 1].set_xlabel('Number of candidates')
        for i in range(2):
            axes[i, 0].set_ylabel('Efficiency (%)')
            axes[i, 1].set_ylabel('Failure Rate (%)')
            axes[i, 1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(
            self.save_to,
            f'{self.bravais_lattice}_EfficiencyFailures_{self.opt_params["tag"]}.png'
            ))
        plt.show()

    def run(self):
        report_counts = {
            'Not found': 0,
            'Found and best': 0,
            'Found but not best': 0,
            'Found but off by two': 0,
            'Found explainers': 0,
            }
        diagnostic_df = pd.DataFrame(columns=[
            'entry_index',
            'true_unit_cell',
            'closest_unit_cell',
            'best_hkl_accuracy',
            'mean_hkl_accuracy',
            'bravais_lattice',
            'spacegroup',
            'impossible',
            'found',
            'counts_uc',
            'counts_ruc',
            'counts_xnn',
            'dominant_axis_info',
            'dominant_zone_info',
            ])
        if self.opt_params['rerun_failures']:
            print()
            print('Loading failures')
            diagnostics = pd.read_json(
                os.path.join(self.save_to, f'{self.bravais_lattice}_optimization_diagnostics.json')
                )
            entry_list = np.array(diagnostics[~diagnostics['found']]['entry_index'])
        else:
            entry_list = np.arange(self.N)
        for entry_index in entry_list:
            start = time.time()
            candidates = self.generate_candidates(
                self.uc_scaled_mean[entry_index],
                self.uc_scaled_var[entry_index],
                self.indexer.data.iloc[entry_index],
                entry_index,
                )
            candidates, diagnostic_df_entry = self.optimize_entry(candidates, entry_index)
            report_counts, found = candidates.get_best_candidates(report_counts)
            diagnostic_df_entry['found'] = found
            diagnostic_df_entry['entry_index'] = entry_index
            diagnostic_df.loc[entry_index] = diagnostic_df_entry
            if self.opt_params['rerun_failures'] == False:
                diagnostic_df.to_json(os.path.join(
                    self.save_to,
                    f'{self.bravais_lattice}_optimization_diagnostics.json'
                    ))
            print(report_counts)
            end = time.time()
            print(end - start)

    def generate_unit_cells(self, uc_scaled_mean, uc_scaled_var):
        if self.indexer.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            candidates_scaled = self.rng.normal(
                loc=uc_scaled_mean,
                scale=np.sqrt(uc_scaled_var),
                size=(self.opt_params['n_candidates_nn'], self.indexer.data_params['n_outputs']),
                )
        elif self.indexer.data_params['lattice_system'] == 'rhombohedral':
            uniform_angle = False
            if uc_scaled_mean[1] <= self.opt_params['minimum_angle_scaled']:
                uniform_angle = True
            elif uc_scaled_mean[1] >= self.opt_params['maximum_angle_scaled']:
                uniform_angle = True
            if uniform_angle:
                candidates_scaled = np.zeros((self.opt_params['n_candidates_nn'], 2))
                candidates_scaled[:, 0] = self.rng.normal(
                    loc=uc_scaled_mean[0],
                    scale=np.sqrt(uc_scaled_var[0]),
                    size=self.opt_params['n_candidates_nn']
                    )
                candidates_scaled[:, 1] = self.rng.uniform(
                    low=self.opt_params['minimum_angle_scaled'],
                    high=self.opt_params['maximum_angle_scaled'],
                    size=self.opt_params['n_candidates_nn']
                    )
            else:
                candidates_scaled = self.rng.normal(
                    loc=uc_scaled_mean,
                    scale=np.sqrt(uc_scaled_var),
                    size=(self.opt_params['n_candidates_nn'], self.indexer.data_params['n_outputs']),
                    )
        elif self.indexer.data_params['lattice_system'] == 'monoclinic':
            uniform_angle = False
            if uc_scaled_mean[3] <= self.opt_params['minimum_angle_scaled']:
                uniform_angle = True
            elif uc_scaled_mean[3] >= self.opt_params['maximum_angle_scaled']:
                uniform_angle = True
            if uniform_angle:
                candidates_scaled = np.zeros((self.opt_params['n_candidates_nn'], 4))
                candidates_scaled[:, :3] = self.rng.normal(
                    loc=uc_scaled_mean[:3],
                    scale=np.sqrt(uc_scaled_var[:3]),
                    size=(self.opt_params['n_candidates_nn'], 3),
                    )
                candidates_scaled[:, 3] = self.rng.uniform(
                    low=self.opt_params['minimum_angle_scaled'],
                    high=self.opt_params['maximum_angle_scaled'],
                    size=self.opt_params['n_candidates_nn']
                    )
            else:
                candidates_scaled = self.rng.normal(
                    loc=uc_scaled_mean,
                    scale=np.sqrt(uc_scaled_var),
                    size=(self.opt_params['n_candidates_nn'], self.indexer.data_params['n_outputs']),
                    )
        elif self.indexer.data_params['lattice_system'] == 'triclinic':
            uniform_alpha = False
            uniform_beta = False
            uniform_gamma = False
            if uc_scaled_mean[3] <= self.opt_params['minimum_angle_scaled']:
                uniform_alpha = True
            elif uc_scaled_mean[3] >= self.opt_params['maximum_angle_scaled']:
                uniform_alpha = True
            if uc_scaled_mean[4] <= self.opt_params['minimum_angle_scaled']:
                uniform_beta = True
            elif uc_scaled_mean[4] >= self.opt_params['maximum_angle_scaled']:
                uniform_beta = True
            if uc_scaled_mean[5] <= self.opt_params['minimum_angle_scaled']:
                uniform_gamma = True
            elif uc_scaled_mean[5] >= self.opt_params['maximum_angle_scaled']:
                uniform_gamma = True
            candidates_scaled = self.rng.normal(
                loc=uc_scaled_mean,
                scale=np.sqrt(uc_scaled_var),
                size=(self.opt_params['n_candidates_nn'], self.indexer.data_params['n_outputs'])
                )
            if uniform_alpha:
                candidates_scaled[:, 3] = self.rng.uniform(
                    low=self.opt_params['minimum_alpha_scaled'],
                    high=self.opt_params['maximum_angle_scaled'],
                    size=self.opt_params['n_candidates_nn']
                    )
            if uniform_beta:
                candidates_scaled[:, 4] = self.rng.uniform(
                    low=self.opt_params['minimum_beta_gamma_scaled'],
                    high=self.opt_params['maximum_angle_scaled'],
                    size=self.opt_params['n_candidates_nn']
                    )
            if uniform_gamma:
                candidates_scaled[:, 5] = self.rng.uniform(
                    low=self.opt_params['minimum_beta_gamma_scaled'],
                    high=self.opt_params['maximum_angle_scaled'],
                    size=self.opt_params['n_candidates_nn']
                    )

        candidate_unit_cells = self.indexer.revert_predictions(uc_pred_scaled=candidates_scaled)
        if self.indexer.data_params['lattice_system'] == 'cubic':
            candidate_unit_cells = candidate_unit_cells[:, np.newaxis]
        return candidate_unit_cells

    def generate_unit_cells_miller_index_templates(self, q2_obs):
        unit_cell_templates = self.indexer.miller_index_templator[self.bravais_lattice].do_predictions(
            q2_obs, n_templates=self.opt_params['n_candidates_template'],
            )
        return unit_cell_templates

    def plot_candidate_unit_cells(self, candidate_uc, entry, entry_index):
        if self.indexer.data_params['lattice_system'] in ['cubic', 'tetragonal', 'hexagonal', 'rhombohedral']:
            return None
        candidates_nn = np.zeros((
            self.n_groups * self.opt_params['n_candidates_nn'],
            self.indexer.data_params['n_outputs']
            ))
        candidates_rf = np.zeros((
            self.n_groups * self.opt_params['n_candidates_rf'],
            self.indexer.data_params['n_outputs']
            ))
        n_candidates = self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf']
        for group_index, group in enumerate(self.indexer.data_params['split_groups']):
            start_candidates = group_index * n_candidates
            stop_candidates = (group_index + 1) * n_candidates

            start_nn = group_index * self.opt_params['n_candidates_nn']
            stop_nn = (group_index + 1) * self.opt_params['n_candidates_nn']
            candidates_nn[start_nn: stop_nn] = candidate_uc[
                start_candidates: start_candidates + self.opt_params['n_candidates_nn']
                ]

            start_rf = group_index * self.opt_params['n_candidates_rf']
            stop_rf = (group_index + 1) * self.opt_params['n_candidates_rf']
            candidates_rf[start_rf: stop_rf] = candidate_uc[
                start_candidates + self.opt_params['n_candidates_nn']: stop_candidates
                ]
        candidates_template = candidate_uc[self.n_groups * n_candidates:]

        if self.indexer.data_params['lattice_system'] == 'monoclinic':
            #fig, axes = plt.subplots(3, 4, figsize=(8, 6))
            reindexed_unit_cell_true = get_different_monoclinic_settings(
                np.array(entry['reindexed_unit_cell']), partial_unit_cell=False
                )
            distance_nn = np.linalg.norm(
                candidates_nn[np.newaxis, :, :3] - reindexed_unit_cell_true[:, np.newaxis, :3],
                axis=2
                ).min(axis=0)
            distance_rf = np.linalg.norm(
                candidates_rf[np.newaxis, :, :3] - reindexed_unit_cell_true[:, np.newaxis, :3],
                axis=2
                ).min(axis=0)
            distance_template = np.linalg.norm(
                candidates_template[np.newaxis, :, :3] - reindexed_unit_cell_true[:, np.newaxis, :3],
                axis=2
                ).min(axis=0)
            angle_bins = np.linspace(np.pi/2, np.pi, 101)
            angle_centers = (angle_bins[1:] + angle_bins[:-1]) / 2
            w = angle_bins[1] - angle_bins[0]
        elif self.indexer.data_params['lattice_system'] == 'orthorhombic':
            #fig, axes = plt.subplots(3, 3, figsize=(8, 6))
            reindexed_unit_cell_true = np.array(entry['reindexed_unit_cell'])[:3]
            distance_nn = np.linalg.norm(
                candidates_nn - reindexed_unit_cell_true[np.newaxis],
                axis=1
                )
            distance_rf = np.linalg.norm(
                candidates_rf - reindexed_unit_cell_true[np.newaxis],
                axis=1
                )
            distance_template = np.linalg.norm(
                candidates_template - reindexed_unit_cell_true[np.newaxis],
                axis=1
                )
        elif self.indexer.data_params['lattice_system'] == 'triclinic':
            fig, axes = plt.subplots(3, 6, figsize=(8, 8))
            reindexed_unit_cell_true = np.array(entry['reindexed_unit_cell'])
            distance_nn = np.linalg.norm(
                candidates_nn[:, :3] - reindexed_unit_cell_true[np.newaxis, :3],
                axis=1
                )
            distance_rf = np.linalg.norm(
                candidates_rf[:, :3] - reindexed_unit_cell_true[np.newaxis, :3],
                axis=1
                )
            distance_template = np.linalg.norm(
                candidates_template[:, :3] - reindexed_unit_cell_true[np.newaxis, :3],
                axis=1
                )
            angle_bins = np.linspace(np.pi/2, np.pi, 101)
            angle_centers = (angle_bins[1:] + angle_bins[:-1]) / 2
            w = angle_bins[1] - angle_bins[0]

            ms = 1
            alpha = 0.25
            
            counts = [np.sum(distance_nn < i) for i in [1, 2, 3, 5]]
            axes[0, 1].set_title(f'Candidates within 1, 2, 3, 5A, (total): {counts[0]}, {counts[1]}, {counts[2]}, {counts[3]} ({candidates_nn.shape[0]})')
            axes[0, 0].plot(candidates_nn[:, 0], candidates_nn[:, 1], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[0, 1].plot(candidates_nn[:, 0], candidates_nn[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[0, 2].plot(candidates_nn[:, 1], candidates_nn[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            if self.indexer.data_params['lattice_system'] == 'monoclinic':
                hist_nn, _ = np.histogram(candidates_nn[:, 3], bins=angle_bins, density=True)
                axes[0, 3].bar(angle_centers, hist_nn, width=w)
            elif self.indexer.data_params['lattice_system'] == 'triclinic':
                for angle_index in range(3, 6):
                    hist_nn, _ = np.histogram(candidates_nn[:, angle_index], bins=angle_bins, density=True)
                    axes[0, angle_index].bar(angle_centers, hist_nn, width=w)

            counts = [np.sum(distance_rf < i) for i in [1, 2, 3, 5]]
            axes[1, 1].set_title(f'Candidates within 1, 2, 3, 5A, (total): {counts[0]}, {counts[1]}, {counts[2]}, {counts[3]} ({candidates_rf.shape[0]})')
            axes[1, 0].plot(candidates_rf[:, 0], candidates_rf[:, 1], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[1, 1].plot(candidates_rf[:, 0], candidates_rf[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[1, 2].plot(candidates_rf[:, 1], candidates_rf[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            if self.indexer.data_params['lattice_system'] == 'monoclinic':
                hist_rf, _ = np.histogram(candidates_rf[:, 3], bins=angle_bins, density=True)
                axes[1, 3].bar(angle_centers, hist_rf, width=w)
            elif self.indexer.data_params['lattice_system'] == 'triclinic':
                for angle_index in range(3, 6):
                    hist_rf, _ = np.histogram(candidates_rf[:, angle_index], bins=angle_bins, density=True)
                    axes[1, angle_index].bar(angle_centers, hist_rf, width=w)

            counts = [np.sum(distance_template < i) for i in [1, 2, 3, 5]]
            axes[2, 1].set_title(f'Candidates within 1, 2, 3, 5A, (total): {counts[0]}, {counts[1]}, {counts[2]}, {counts[3]} ({candidates_template.shape[0]})')
            axes[2, 0].plot(candidates_template[:, 0], candidates_template[:, 1], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[2, 1].plot(candidates_template[:, 0], candidates_template[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[2, 2].plot(candidates_template[:, 1], candidates_template[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            if self.indexer.data_params['lattice_system'] == 'monoclinic':
                hist_template, _ = np.histogram(candidates_template[:, 3], bins=angle_bins, density=True)
                axes[2, 3].bar(angle_centers, hist_template, width=w)
            elif self.indexer.data_params['lattice_system'] == 'triclinic':
                for angle_index in range(3, 6):
                    hist_template, _ = np.histogram(candidates_template[:, angle_index], bins=angle_bins, density=True)
                    axes[2, angle_index].bar(angle_centers, hist_template, width=w)

            if self.indexer.data_params['lattice_system'] == 'monoclinic':
                for reindexed_index in range(reindexed_unit_cell_true.shape[0]):
                    for row in range(3):
                        axes[row, 0].plot(
                            reindexed_unit_cell_true[reindexed_index, 0],
                            reindexed_unit_cell_true[reindexed_index, 1], 
                            linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0]
                            )
                        axes[row, 1].plot(
                            reindexed_unit_cell_true[reindexed_index, 0],
                            reindexed_unit_cell_true[reindexed_index, 2],
                            linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0]
                            )
                        axes[row, 2].plot(
                            reindexed_unit_cell_true[reindexed_index, 1],
                            reindexed_unit_cell_true[reindexed_index, 2], 
                            linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0]
                            )
                        ylim = axes[row, 3].get_ylim()
                        if reindexed_unit_cell_true[reindexed_index, 3] > np.pi/2:
                            angle = reindexed_unit_cell_true[reindexed_index, 3]
                        else:
                            angle = np.pi - reindexed_unit_cell_true[reindexed_index, 3]
                        axes[row, 3].plot(
                            [angle, angle],
                            0.5*np.array(ylim),
                            color=[0.8, 0, 0], linewidth=1
                            )
                        axes[row, 3].set_ylim(ylim)
            else:
                for row in range(3):
                    axes[row, 0].plot(
                        reindexed_unit_cell_true[0],
                        reindexed_unit_cell_true[1], 
                        linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0]
                        )
                    axes[row, 1].plot(
                        reindexed_unit_cell_true[0],
                        reindexed_unit_cell_true[2],
                        linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0]
                        )
                    axes[row, 2].plot(
                        reindexed_unit_cell_true[1],
                        reindexed_unit_cell_true[2], 
                        linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0]
                        )
                    if self.indexer.data_params['lattice_system'] == 'triclinic':
                        for angle_index in range(3, 6):
                            ylim = axes[row, angle_index].get_ylim()
                            angle = reindexed_unit_cell_true[angle_index]
                            axes[row, angle_index].plot(
                                [angle, angle],
                                0.5*np.array(ylim),
                                color=[0.8, 0, 0], linewidth=1
                                )
                            axes[row, angle_index].set_ylim(ylim)

            for col in range(3):
                ylim0 = axes[0, col].get_ylim()
                ylim1 = axes[1, col].get_ylim()
                ylim = [min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1])]
                for row in range(3):
                    axes[row, col].set_ylim(ylim)

                xlim0 = axes[0, col].get_xlim()
                xlim1 = axes[1, col].get_xlim()
                xlim = [min(xlim0[0], xlim1[0]), max(xlim0[1], xlim1[1])]
                for row in range(3):
                    axes[row, col].set_xlim(xlim)

            axes[0, 0].set_ylabel('NN predictions\nb')
            axes[1, 0].set_ylabel('RF predictions\nb')
            axes[2, 0].set_ylabel('Template predictions\nb')
            for row in range(3):
                axes[row, 0].set_xlabel('a')
                axes[row, 1].set_xlabel('a')
                axes[row, 1].set_ylabel('c')
                axes[row, 2].set_xlabel('b')
                axes[row, 2].set_ylabel('c')
                if self.indexer.data_params['lattice_system'] == 'monoclinic':
                    axes[row, 3].set_xlabel('beta')
                elif self.indexer.data_params['lattice_system'] == 'triclinic':
                    axes[row, 3].set_xlabel('alpha')
                    axes[row, 4].set_xlabel('beta')
                    axes[row, 5].set_xlabel('gamma')
            fig.tight_layout()
            fig.savefig(os.path.join(
                self.save_to,
                'figures',
                f'{self.bravais_lattice}_initial_candidates_{entry_index:03d}_{self.opt_params["tag"]}.png'
                ))
            # This is a bit scorched earth, but it helps with memory leaks.
            plt.cla()
            plt.clf()
            plt.close('all')

    def generate_candidates(self, uc_scaled_mean, uc_scaled_var, entry, entry_index):
        n_candidates = self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf']
        candidate_unit_cells = np.zeros((
            self.n_groups * n_candidates + self.opt_params['n_candidates_template'],
            self.indexer.data_params['n_outputs']
            ))
        for group_index, group in enumerate(self.indexer.data_params['split_groups']):
            start = group_index * n_candidates
            stop = (group_index + 1) * n_candidates
            # Get candidates from the neural network model
            candidate_unit_cells[start: start + self.opt_params['n_candidates_nn'], :] = \
                self.generate_unit_cells(
                    uc_scaled_mean[group_index, :], uc_scaled_var[group_index, :]
                    )

            # Get candidates from the random forest model
            q2_scaled = np.array(entry['q2_scaled'])[np.newaxis]
            # candidates_scaled_tree: n_entries, n_outputs, n_trees
            #   n_entries = 1 because there is only one q2_scaled input
            _, _, candidates_scaled_tree = \
                self.indexer.unit_cell_generator[group].do_predictions_trees(q2_scaled=q2_scaled)
            tree_indices = self.rng.choice(
                candidates_scaled_tree.shape[2],
                size=self.opt_params['n_candidates_rf'],
                replace=False
                )
            candidate_unit_cells_tree = self.indexer.revert_predictions(
                uc_pred_scaled=candidates_scaled_tree[0, :, tree_indices]
                )
            candidate_unit_cells[start + self.opt_params['n_candidates_nn']: stop, :] = \
                candidate_unit_cells_tree

        candidate_unit_cells[self.n_groups * n_candidates:] = \
            self.generate_unit_cells_miller_index_templates(np.array(entry['q2']))

        candidate_unit_cells = fix_unphysical(
            unit_cell=candidate_unit_cells,
            rng=self.rng,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            lattice_system=self.indexer.data_params['lattice_system']
            )
        if self.indexer.data_params['lattice_system'] == 'triclinic':
            candidate_unit_cells, _ = reindex_entry_triclinic(candidate_unit_cells)

        #self.plot_candidate_unit_cells(candidate_unit_cells, entry, entry_index)

        candidates = Candidates(
            entry=entry,
            unit_cell=candidate_unit_cells,
            lattice_system=self.indexer.data_params['lattice_system'],
            bravais_lattice=self.bravais_lattice,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            tolerance=self.opt_params['found_tolerance'],
            epsilon=self.opt_params['epsilon'],
            max_neighbors=self.opt_params['max_neighbors'],
            radius=self.opt_params['neighbor_radius'],
            )

        candidates = self.assign_hkls_closest(candidates)

        target_function = CandidateOptLoss(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        target_function.update(candidates.hkl, candidates.xnn)
        candidates.loss = target_function.get_loss(candidates.xnn)
        candidates.best_loss = candidates.loss.copy()
        candidates.best_xnn = candidates.xnn.copy()
        return candidates

    def optimize_entry(self, candidates, entry_index):
        diagnostic_df_entry = candidates.diagnostics(self.indexer.data_params['hkl_ref_length'])
        for iteration_info in self.opt_params['iteration_info']:
            if iteration_info['worker'] == 'monoclinic_reset':
                candidates = self.monoclinic_reset(candidates, iteration_info)
                print_list = [
                    f'{candidates.n},',
                    f'{len(candidates.explainers)},',
                    f'{candidates.loss.mean():0.2f},',
                    f'{candidates.loss.min():0.2f},',
                    f'{iteration_info["worker"]},',
                    ]
                print(' '.join(print_list))
            else:
                for iter_index in range(iteration_info['n_iterations']):
                    if iter_index > 0 and iter_index % iteration_info['exhaustive_search_period'] == 0:
                        candidates.exhaustive_search()
                    candidates = self.random_subsampling(candidates, iteration_info)
                    if len(candidates.explainers) > self.opt_params['max_explainers']:
                        return candidates, diagnostic_df_entry
                    print_list = [
                        f'{candidates.n},',
                        f'{len(candidates.explainers)},',
                        f'{candidates.loss.mean():0.2f},',
                        f'{candidates.loss.min():0.2f},',
                        f'{iteration_info["worker"]},',
                        ]
                    print(' '.join(print_list))
                save_to = os.path.join(
                    self.save_to,
                    'figures',
                    f'{self.bravais_lattice}_{entry_index:03d}_{self.opt_params["tag"]}'
                    )
        return candidates, diagnostic_df_entry

    def assign_hkls_closest(self, candidates):
        q2_ref_calc = self.q2_calculator.get_q2(candidates.xnn)
        pairwise_differences = scipy.spatial.distance.cdist(
            candidates.q2_obs[:, np.newaxis], q2_ref_calc.ravel()[:, np.newaxis]
            ).reshape((self.indexer.data_params['n_points'], candidates.n, self.indexer.data_params['hkl_ref_length']))
        hkl_assign = pairwise_differences.argmin(axis=2).T
        hkl_pred = np.take(self.indexer.hkl_ref[self.bravais_lattice], hkl_assign, axis=0)
        candidates.hkl = hkl_pred
        return candidates

    def random_subsampling(self, candidates, iteration_info):
        if type(iteration_info['n_drop']) == list:
            n_drop = self.rng.choice(iteration_info['n_drop'], size=1)[0]
        else:
            n_drop = iteration_info['n_drop']
        n_keep = self.indexer.data_params['n_points'] - n_drop
        subsampled_indices = self.rng.permuted(
            np.repeat(np.arange(self.indexer.data_params['n_points'])[np.newaxis], candidates.n, axis=0),
            axis=1
            )[:, :n_keep]

        hkl_subsampled = np.take_along_axis(candidates.hkl, subsampled_indices[:, :, np.newaxis], axis=1)
        q2_subsampled = np.take(candidates.q2_obs, subsampled_indices)

        target_function = CandidateOptLoss(
            q2_subsampled, 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        target_function.update(hkl_subsampled, candidates.xnn)
        delta_gn = target_function.gauss_newton_step(candidates.xnn)
        candidates.xnn = candidates.xnn + delta_gn

        candidates.fix_out_of_range_candidates()
        candidates = self.assign_hkls_closest(candidates)
        target_function = CandidateOptLoss(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        target_function.update(candidates.hkl, candidates.xnn)
        candidates.loss = target_function.get_loss(candidates.xnn)

        improved_loss = candidates.loss < candidates.best_loss
        candidates.best_loss[improved_loss] = candidates.loss[improved_loss]
        candidates.best_xnn[improved_loss] = candidates.xnn[improved_loss]
        candidates.pick_explainers()
        return candidates

    def monoclinic_reset(self, candidates, iteration_info):
        assert self.indexer.data_params['lattice_system'] == 'monoclinic'

        candidates.monoclinic_reset(iteration_info['n_best'], iteration_info['n_angles'])
        candidates = self.assign_hkls_closest(candidates)

        xnn = np.stack(candidates.candidates['xnn'])
        target_function = CandidateOptLoss(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        target_function.update(
            np.stack(candidates.candidates['hkl']),
            xnn
            )
        candidates.candidates['loss'] = target_function.get_loss(xnn)
        return candidates
