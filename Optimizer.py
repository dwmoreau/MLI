import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.optimize
import scipy.special
import time

from Indexing import Indexing
from Reindexing import get_different_monoclinic_settings
from Reindexing import reindex_entry_triclinic
from TargetFunctions import CandidateOptLoss_xnn
from Utilities import fix_unphysical_rhombohedral
from Utilities import fix_unphysical_triclinic
from Utilities import get_hkl_matrix
from Utilities import get_reciprocal_unit_cell_from_xnn
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import reciprocal_uc_conversion


def vectorized_subsampling(p, n_picks, rng):
    n_entries = p.shape[0]
    n_choices = p.shape[1]
    choices = np.repeat(np.arange(n_choices)[np.newaxis], repeats=n_entries, axis=0) 
    chosen = np.zeros((n_entries, n_picks), dtype=int)
    for index in range(n_picks):
        # cumsum: n_entries, n_peaks
        # random_value: n_entries
        # q: n_entries, n_peaks
        n_peaks = p.shape[1]
        cumsum = p.cumsum(axis=1)
        random_value = rng.random(n_entries)
        q = cumsum >= random_value[:, np.newaxis]
        chosen_indices = q.argmax(axis=1)
        chosen[:, index] = choices[np.arange(n_entries), chosen_indices]
        p_flat = p.ravel()
        choices_flat = choices.ravel()
        delete_indices = np.arange(n_entries) * n_peaks + chosen_indices
        p = np.delete(p_flat, delete_indices).reshape((n_entries, n_peaks - 1))
        choices = np.delete(choices_flat, delete_indices).reshape((n_entries, n_peaks - 1))
    chosen = np.sort(chosen, axis=1)
    return chosen


def vectorized_resampling(softmaxes, rng):
    # This is a major performance bottleneck

    # This function randomly resamples the peaks using the algorithm
    #  1: Pick a peak at random
    #  2: Assign Miller index according to softmaxes
    #  3: Set the assigned Miller index softmax to zero for all other peaks
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]

    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    random_values = rng.random(size=(n_entries, n_peaks))
    point_order = rng.permutation(n_peaks)
    softmaxes_zeroed = softmaxes.copy()
    i = 0
    for point_index in point_order:
        # Fast random selection:
        #  1: make cummulative sum along the distribution's axis (this is a cdf)
        #  2: selection is the first point in cummulative sum greater than random value
        #    - fastest way to do this, convert to bool array and find first True with argmax
        #    - To account for adding zeros to the softmax array, the random values are scaled
        #      instead of scaling the softmax array

        # This line is slow (60% of execution time)
        cumsum = np.cumsum(softmaxes_zeroed[:, point_index, :], axis=1)
        q = cumsum >= (random_values[:, point_index] * cumsum[:, -1])[:, np.newaxis]
        hkl_assign[:, point_index] = np.argmax(q, axis=1)
        i += 1
        if i < n_peaks:
            np.put_along_axis(
                softmaxes_zeroed,
                hkl_assign[:, point_index][:, np.newaxis, np.newaxis],
                values=0,
                axis=2
                )

    softmax = np.take_along_axis(softmaxes, hkl_assign[:, :, np.newaxis], axis=2)[:, :, 0]
    return hkl_assign, softmax


def best_assign_nocommon_original(softmaxes):
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]
    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)

    peak_choice = np.argsort(np.max(softmaxes, axis=2), axis=1)
    for candidate_index in range(n_entries):
        softmaxes_zeroed = softmaxes[candidate_index].copy()
        for peak_index in peak_choice[candidate_index]:
            choice = np.argmax(softmaxes_zeroed[peak_index, :])
            hkl_assign[candidate_index, peak_index] = choice
            softmaxes_zeroed[:, hkl_assign[candidate_index, peak_index]] = 0

    softmax_assign = np.take_along_axis(softmaxes, hkl_assign[:, :, np.newaxis], axis=2)
    return hkl_assign, softmax_assign


def best_assign_nocommon(softmaxes):
    # This is three times faster than the version above.
    # It picks the first occurance as opposed to the best occurance.
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]
    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    softmax_assign = np.zeros((n_entries, n_peaks))
    for peak_index in range(n_peaks):
        softmaxes_peak = softmaxes[:, peak_index, :]
        hkl_assign[:, peak_index] = np.argmax(softmaxes_peak, axis=1)
        softmax_assign[:, peak_index] = np.take_along_axis(
            softmaxes_peak, hkl_assign[:, peak_index][:, np.newaxis],
            axis=1
            )[:, 0]
        np.put(softmaxes, hkl_assign[:, np.newaxis, :], 0)
    return hkl_assign, softmax_assign


class Candidates:
    def __init__(self, entry, unit_cell, lattice_system, bravais_lattice, minimum_unit_cell, maximum_unit_cell, tolerance):
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
        elif self.lattice_system == 'monoclinic':
            self.maximum_angle = np.pi
            self.minimum_angle = np.pi/2
        elif self.lattice_system == 'triclinic':
            self.maximum_angle = np.pi
            self.minimum_angle_beta_gamma = np.pi/2
            self.minimum_angle_alpha = 0.01
        self.rng = np.random.default_rng()
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

        self.hkl_true = np.array(entry['reindexed_hkl'])[:, :, 0]
        self.hkl_labels_true = np.array(entry['hkl_labels'])
        self.bl_true = entry['bravais_lattice']
        self.sg_true = int(entry['spacegroup_number'])
        self.spacegroup_symbol_hm_true = entry['reindexed_spacegroup_symbol_hm']

        self.df_columns = [
            'unit_cell',
            'reciprocal_unit_cell',
            'xnn',
            'hkl',
            'softmax',
            'loss',
            'best_loss',
            'best_xnn',
            'best_returns_trajectory',
            'neighbors',
            'accepted',
            'loss_trajectory',
            'xnn_trajectory',
            'xnn_drift',
            'acceptance_trajectory',
            ]
        self.candidates = pd.DataFrame(columns=self.df_columns)
        self.explainers = pd.DataFrame(columns=self.df_columns)

        self.candidates['unit_cell'] = list(unit_cell)    
        self.update_xnn_from_unit_cell()
        self.hkl_true_check = get_hkl_matrix(self.hkl_true, self.lattice_system)
        self.candidates['best_returns'] = np.zeros(unit_cell.shape[0], dtype=int)
        self.candidates['best_loss'] = np.zeros(unit_cell.shape[0])
        self.candidates['best_xnn'] = self.candidates['xnn'].copy()

    def update_xnn_from_unit_cell(self):
        unit_cell = np.stack(self.candidates['unit_cell'])
        reciprocal_unit_cell = reciprocal_uc_conversion(
            unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        xnn = get_xnn_from_reciprocal_unit_cell(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )

        unit_cell = np.stack(self.candidates['unit_cell'])
        for i in range(unit_cell.shape[0]):
            if np.sum(np.isnan(reciprocal_unit_cell[i])) > 0:
                print('Get Xnn from unit cell')
                print(unit_cell[i])
                print(reciprocal_unit_cell[i])
                print(xnn[i])
                print()

        bad_conversions = np.sum(np.isnan(reciprocal_unit_cell), axis=1) > 0
        good_indices = np.arange(self.n)[~bad_conversions]
        n_bad = np.sum(bad_conversions)
        if n_bad > 0:
            if n_bad > bad_conversions.size - n_bad:
                good_indices = self.rng.choice(good_indices, replace=True, size=n_bad)
            else:
                good_indices = self.rng.choice(good_indices, replace=False, size=n_bad)
            xnn[bad_conversions] = xnn[good_indices]
            reciprocal_unit_cell[bad_conversions] = reciprocal_unit_cell[good_indices]
            unit_cell[bad_conversions] = unit_cell[good_indices]

        self.candidates['reciprocal_unit_cell'] = list(reciprocal_unit_cell)
        self.candidates['unit_cell'] = list(unit_cell)
        self.candidates['xnn'] = list(xnn)

    def update_unit_cell_from_xnn(self):
        xnn = np.stack(self.candidates['xnn'])
        # This forces the xnn components x_hh, x_kk, and x_ll to their bounds
        if self.lattice_system in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            too_small = xnn < (1 / self.maximum_unit_cell)**2
            too_large = xnn > (1 / self.minimum_unit_cell)**2
            xnn[too_small] = (1 / self.maximum_unit_cell)**2
            xnn[too_large] = (1 / self.minimum_unit_cell)**2
        elif self.lattice_system == 'rhombohedral':
            xnn = fix_unphysical_rhombohedral(xnn=xnn, rng=self.rng)
        elif self.lattice_system == 'monoclinic':
            too_small = xnn[:, :3] < (1 / self.maximum_unit_cell)**2
            too_large = xnn[:, :3] > (1 / self.minimum_unit_cell)**2
            xnn[:, :3][too_small] = (1 / self.maximum_unit_cell)**2
            xnn[:, :3][too_large] = (1 / self.minimum_unit_cell)**2
            cos_rbeta = xnn[:, 3] / (2 * np.sqrt(xnn[:, 0] * xnn[:, 2]))
            too_small = cos_rbeta < -1
            too_large = cos_rbeta > 1
            xnn[too_small, 3] = -2*0.999 * np.sqrt(xnn[too_small, 0] * xnn[too_small, 2])
            xnn[too_large, 3] = 2*0.999 * np.sqrt(xnn[too_large, 0] * xnn[too_large, 2])
        elif self.lattice_system == 'triclinic':
            xnn = fix_unphysical_triclinic(xnn=xnn, rng=self.rng)

        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        unit_cell = reciprocal_uc_conversion(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        for i in range(unit_cell.shape[0]):
            if np.sum(np.isnan(reciprocal_unit_cell[i])) > 0:
                print('Get unit cell from Xnn')
                print(unit_cell[i])
                print(reciprocal_unit_cell[i])
                print(xnn[i])
                print()

        bad_conversions = np.sum(np.isnan(reciprocal_unit_cell), axis=1) > 0
        good_indices = np.arange(self.n)[~bad_conversions]
        n_bad = np.sum(bad_conversions)
        if n_bad > 0:
            if n_bad > bad_conversions.size - n_bad:
                good_indices = self.rng.choice(good_indices, replace=True, size=n_bad)
            else:
                good_indices = self.rng.choice(good_indices, replace=False, size=n_bad)
            xnn[bad_conversions] = xnn[good_indices]
            reciprocal_unit_cell[bad_conversions] = reciprocal_unit_cell[good_indices]
            unit_cell[bad_conversions] = unit_cell[good_indices]

        self.candidates['reciprocal_unit_cell'] = list(reciprocal_unit_cell)
        self.candidates['unit_cell'] = list(unit_cell)
        self.candidates['xnn'] = list(xnn)

    def diagnostics(self, hkl_ref_length):
        unit_cell = np.stack(self.candidates['unit_cell'])
        reciprocal_unit_cell = np.stack(self.candidates['reciprocal_unit_cell'])
        xnn = np.stack(self.candidates['xnn'])
        reciprocal_unit_cell_rms = 1/np.sqrt(self.n_uc) * np.linalg.norm(reciprocal_unit_cell - self.reciprocal_unit_cell_true, axis=1)
        reciprocal_unit_cell_max_diff = np.max(np.abs(reciprocal_unit_cell - self.reciprocal_unit_cell_true), axis=1)

        hkl = np.stack(self.candidates['hkl'])
        hkl_pred_check = get_hkl_matrix(hkl, self.lattice_system)
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
                reciprocal_unit_cell[np.newaxis, :, :3] - reciprocal_unit_cell_true[:, np.newaxis, :3], 
                axis=2
                ).min(axis=0)
            distance_uc = np.linalg.norm(
                unit_cell[np.newaxis, :, :3] - unit_cell_true[:, np.newaxis, :3],
                axis=2
                ).min(axis=0)
            distance_xnn = np.linalg.norm(
                xnn[np.newaxis, :, :3] - xnn_true[:, np.newaxis, :3],
                axis=2
                ).min(axis=0)
        elif self.lattice_system == 'triclinic':
            distance_ruc = np.linalg.norm(
                reciprocal_unit_cell[:, :3] - self.reciprocal_unit_cell_true[np.newaxis, :3], 
                axis=1
                )
            distance_uc = np.linalg.norm(
                unit_cell[:, :3] - self.unit_cell_true[np.newaxis, :3],
                axis=1
                )
            distance_xnn = np.linalg.norm(
                xnn[:, :3] - self.xnn_true[np.newaxis, :3],
                axis=1
                )
        elif self.lattice_system == 'rhombohedral':
            distance_ruc = np.abs(reciprocal_unit_cell[:, 0] - self.reciprocal_unit_cell_true[np.newaxis, 0])
            distance_uc = np.abs(unit_cell[:, 0] - self.unit_cell_true[np.newaxis, 0])
            distance_xnn = np.abs(xnn[:, 0] - self.xnn_true[np.newaxis, 0])
        else:
            distance_ruc = np.linalg.norm(
                reciprocal_unit_cell - self.reciprocal_unit_cell_true[np.newaxis], axis=1
                )
            distance_uc = np.linalg.norm(unit_cell - self.unit_cell_true[np.newaxis], axis=1)
            distance_xnn = np.linalg.norm(xnn - self.xnn_true[np.newaxis], axis=1)
        counts_ruc = [np.sum(distance_ruc < i) for i in [0.005, 0.01, 0.02]]
        counts_uc = [np.sum(distance_uc < i) for i in [1, 2, 3]]
        counts_xnn = [np.sum(distance_xnn < i) for i in [0.0005, 0.001, 0.002]]

        print(f'Starting # candidates:       {self.n}')
        print(f'Impossible:                  {impossible}')
        print(f'True dominant axis info:     {dominant_axis_info}')
        print(f'True dominant zone info:     {dominant_zone_info}')
        print(f'True unit cell:              {np.round(self.unit_cell_true, decimals=4)}')
        print(f'True reciprocal unit cell:   {np.round(self.reciprocal_unit_cell_true, decimals=4)}')
        print(f'True Xnn:                    {np.round(self.xnn_true, decimals=4)}')
        print(f'Closest unit cell:           {np.round(reciprocal_unit_cell[np.argmin(reciprocal_unit_cell_rms)], decimals=4)}')
        print(f'Closest unit cell rms:       {reciprocal_unit_cell_rms.min():2.2f}')
        print(f'Smallest unit cell max diff: {reciprocal_unit_cell_max_diff.min():2.2f}')
        print(f'Mean unit cell rms:          {reciprocal_unit_cell_rms.mean():2.2f}')
        print(f'Best HKL accuracy:           {hkl_accuracy.max():1.2f}')
        print(f'Mean HKL accuracy:           {hkl_accuracy.mean():1.2f}')
        print(f'Close Unit Cell:             {counts_uc[0]}, {counts_uc[1]}, {counts_uc[2]}')
        print(f'Close Reciprocal UC:         {counts_ruc[0]}, {counts_ruc[1]}, {counts_ruc[2]}')
        print(f'Close Xnn:                   {counts_xnn[0]}, {counts_xnn[1]}, {counts_xnn[2]}')
        print(f'Bravais Lattice:             {self.bl_true}')
        print(f'Spacegroup:                  {self.sg_true} {self.spacegroup_symbol_hm_true}')

        output_dict = {
            'entry_index': None,
            'true_unit_cell': self.reciprocal_unit_cell_true,
            'closest_unit_cell': reciprocal_unit_cell[np.argmin(reciprocal_unit_cell_rms)],
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
        unit_cells = np.stack(self.candidates['unit_cell'])
        if self.lattice_system in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            too_small_lengths = unit_cells < self.minimum_unit_cell
            too_large_lengths = unit_cells > self.maximum_unit_cell
            if np.sum(too_small_lengths) > 0:
                indices = np.argwhere(too_small_lengths)
                unit_cells[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=self.minimum_unit_cell,
                    high=1.05*self.minimum_unit_cell,
                    size=np.sum(too_small_lengths)
                    )
            if np.sum(too_large_lengths) > 0:
                indices = np.argwhere(too_large_lengths)
                unit_cells[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=0.95*self.maximum_unit_cell,
                    high=self.maximum_unit_cell,
                    size=np.sum(too_large_lengths)
                    )
        elif self.lattice_system == 'rhombohedral':
            unit_cells = fix_unphysical_rhombohedral(
                unit_cell=unit_cells,
                rng=self.rng,
                minimum_unit_cell=self.minimum_unit_cell, 
                maximum_unit_cell=self.maximum_unit_cell,
                )
        elif self.lattice_system == 'triclinic':
            unit_cells = fix_unphysical_triclinic(
                unit_cell=unit_cells,
                rng=self.rng,
                minimum_unit_cell=self.minimum_unit_cell, 
                maximum_unit_cell=self.maximum_unit_cell,
                )
            for index in range(unit_cells.shape[0]):
                unit_cells[index], _ = reindex_entry_triclinic(unit_cells[index], radians=True)
        elif self.lattice_system == 'monoclinic':
            too_small_lengths = unit_cells[:, :3] < self.minimum_unit_cell
            too_large_lengths = unit_cells[:, :3] > self.maximum_unit_cell
            if np.sum(too_small_lengths) > 0:
                indices = np.argwhere(too_small_lengths)
                unit_cells[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=self.minimum_unit_cell,
                    high=1.05*self.minimum_unit_cell,
                    size=np.sum(too_small_lengths)
                    )
            if np.sum(too_large_lengths) > 0:
                indices = np.argwhere(too_large_lengths)
                unit_cells[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=0.95*self.maximum_unit_cell,
                    high=self.maximum_unit_cell,
                    size=np.sum(too_large_lengths)
                    )

            too_small_angles = unit_cells[:, 3] < self.minimum_angle
            too_large_angles = unit_cells[:, 3] > self.maximum_angle
            if np.sum(too_small_angles) > 0:
                unit_cells[too_small_angles, 3] = np.pi - unit_cells[too_small_angles, 3]
            if np.sum(too_large_angles) > 0:
                unit_cells[too_large_angles, 3] = self.rng.uniform(
                    low=0.95*self.maximum_angle,
                    high=self.maximum_angle,
                    size=np.sum(too_large_angles)
                    )
            
        self.candidates['unit_cell'] = list(unit_cells)
        self.update_xnn_from_unit_cell()

    def get_neighbor_distance(self, uc0, uc1):
        if self.lattice_system in ['monoclinic', 'triclinic']:
            distance = np.linalg.norm(uc0[:, np.newaxis, :3] - uc1[np.newaxis, :, :3], axis=2)
        elif self.lattice_system in ['tetragonal', 'hexagonal', 'cubic', 'orthorhombic']:
            distance = np.linalg.norm(uc0[:, np.newaxis, :] - uc1[np.newaxis, :, :], axis=2)
        elif self.lattice_system == 'rhombohedral':
            distance = np.abs(uc0[:, np.newaxis, 0] - uc1[np.newaxis, :, 0])
        return distance

    def redistribute_and_perturb_unit_cells(self, unit_cells, from_indices, to_indices, norm_factor=None):
        n_indices = from_indices.size
        if not norm_factor is None:
            norm_factor = self.rng.uniform(low=norm_factor[0], high=norm_factor[1], size=n_indices)
        else:
            norm_factor = np.ones(n_indices)
        if self.lattice_system in ['monoclinic', 'triclinic']:
            perturbation = self.rng.uniform(low=-1, high=1, size=(n_indices, 3))
            perturbation /= (norm_factor/np.linalg.norm(perturbation, axis=1))[:, np.newaxis]
            unit_cells[from_indices, :3] = unit_cells[to_indices, :3] + perturbation
            unit_cells[:, :3] = np.abs(unit_cells[:, :3])
            if self.lattice_system == 'monoclinic':
                # Randomly reassign the monoclinic angle
                unit_cells[from_indices, 3] = np.arccos(
                    self.rng.uniform(low=-1, high=0, size=n_indices)
                    )
            elif self.lattice_system == 'triclinic':
                unit_cells[from_indices, 3:] = unit_cells[to_indices, 3:]
                # This leads to unphysical unit cells. They cannot be converted to reciprocal space
                unit_cells[from_indices, 3] = np.arccos(
                    self.rng.uniform(low=-1, high=1, size=n_indices)
                    )
                unit_cells[from_indices, 4:] = np.arccos(
                    self.rng.uniform(low=-1, high=0, size=(n_indices, 2))
                    )
        elif self.lattice_system in ['cubic', 'tetragonal', 'hexagonal', 'orthorhombic']:
            perturbation = self.rng.uniform(low=-1, high=1, size=(n_indices, self.n_uc))
            perturbation /= (norm_factor/np.linalg.norm(perturbation, axis=1))[:, np.newaxis]
            unit_cells[from_indices] = unit_cells[to_indices] + perturbation
            unit_cells = np.abs(unit_cells)
        elif self.lattice_system == 'rhombohedral':
            perturbation = self.rng.uniform(low=-1, high=1, size=n_indices)
            unit_cells[from_indices, 0] = unit_cells[to_indices, 0] + norm_factor*perturbation
            unit_cells[:, 0] = np.abs(unit_cells[:, 0])
            unit_cells[from_indices, 1] = np.arccos(
                self.rng.uniform(low=-0.5, high=1, size=n_indices)
                )

        if self.lattice_system == 'monoclinic':
            permute_indices = unit_cells[:, 0] > unit_cells[:, 2]
            unit_cells[permute_indices] = np.column_stack((
                unit_cells[permute_indices, 2],
                unit_cells[permute_indices, 1],
                unit_cells[permute_indices, 0],
                unit_cells[permute_indices, 3],
                ))
        elif self.lattice_system == 'orthorhombic':
            if self.bravais_lattice != 'oC':
                # base centered orthombic unit cells are not ordered
                order = np.argsort(unit_cells, axis=1)
                unit_cells = np.take_along_axis(unit_cells, order, axis=1)
        elif self.lattice_system == 'triclinic':
            # This is incorrect for reindexing, the angles don't let the axes
            # simply permute.
            # But the angles are randomly generated, so it is fine.
            order = np.argsort(unit_cells, axis=1)
            unit_cells = np.take_along_axis(unit_cells, order, axis=1)
        return unit_cells

    def redistribute_unit_cells(self, max_neighbors, radius):
        unit_cells = np.stack(self.candidates['unit_cell'])
        redistributed_unit_cells = unit_cells.copy()
        largest_neighborhood = max_neighbors + 1
        n_redistributed = 0 
        while largest_neighborhood > max_neighbors:
            distance = self.get_neighbor_distance(redistributed_unit_cells, redistributed_unit_cells)
            neighbor_array = distance < radius
            neighbor_count = np.sum(neighbor_array, axis=1)
            largest_neighborhood = neighbor_count.max()
            if largest_neighborhood > max_neighbors:
                from_indices = []
                high_density_indices = np.where(neighbor_count > max_neighbors)[0]
                for high_density_index in high_density_indices:
                    neighbor_indices = np.where(neighbor_array[high_density_index])[0]
                    excess_neighbors = neighbor_indices.size - max_neighbors
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
        
                low_density_indices = np.where(neighbor_count < max_neighbors)[0]
                if low_density_indices.size == 0:
                    break
                prob = max_neighbors - neighbor_count[low_density_indices]
                prob = prob / prob.sum()
                if excess_neighbors <= low_density_indices.size:
                    replace = False
                else:
                    replace = True
                to_indices = low_density_indices[
                    self.rng.choice(low_density_indices.size, size=excess_neighbors, replace=replace, p=prob)
                    ]
                unit_cells = self.redistribute_and_perturb_unit_cells(
                    redistributed_unit_cells, from_indices, to_indices
                    )
                
        print(f'Redistributed {n_redistributed} candidates')
        self.candidates['unit_cell'] = list(redistributed_unit_cells)
        self.fix_out_of_range_candidates()

    def setup_exhaustive_search(self, max_neighbors, radius):
        self.ending_unit_cells = None
        self.starting_unit_cells = np.stack(self.candidates['unit_cell']).copy()
        self.max_neighbors = max_neighbors
        self.radius = radius

    def exhaustive_search(self):
        unit_cells = np.stack(self.candidates['unit_cell']).copy()
        if self.ending_unit_cells is None:
            self.ending_unit_cells = unit_cells.copy()
        else:
            self.ending_unit_cells = np.row_stack((self.ending_unit_cells, unit_cells))

        distance = self.get_neighbor_distance(unit_cells, self.starting_unit_cells)
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
            unit_cells = self.redistribute_and_perturb_unit_cells(unit_cells, from_indices, to_indices)

            not_chosen = np.delete(np.arange(low_density_indices.size), choice)
            not_chosen_indices = low_density_indices[not_chosen]
            
            unit_cells = self.redistribute_and_perturb_unit_cells(
                unit_cells, not_chosen_indices, not_chosen_indices, norm_factor=[0.25, 0.75]
                )
        else:
            not_chosen_indices = np.ones(unit_cells.shape[0], dtype=bool)
            unit_cells = self.redistribute_and_perturb_unit_cells(
                unit_cells, not_chosen_indices, not_chosen_indices, norm_factor=[0.25, 0.75]
                )

        self.starting_unit_cells = np.row_stack((self.starting_unit_cells, unit_cells))
        print(f'Exhaustive search redistributed {excess_neighbors} candidates')
        self.candidates['unit_cell'] = list(unit_cells)
        start = self.candidates.index.max() + 1
        self.candidates.reset_index(
            names=list(np.arange(start, start + self.n)), drop=True, inplace=True
            )
        self.fix_out_of_range_candidates()

    def update(self):
        if self.n > 1:
            bad_candidates = np.isnan(self.candidates['loss'])
            if np.sum(bad_candidates) > 0:
                self.candidates = self.candidates.loc[~bad_candidates]
                self.n = len(self.candidates)

            # Fix candidates with too small or too large unit cells
            self.fix_out_of_range_candidates()
            self.update_history()
        if len(self.candidates) > 1:
            self.pick_explainers()
        self.candidates = self.candidates.sort_values(by='best_loss')

    def update_history(self):
        returned = np.array(self.candidates['loss'] == self.candidates['best_loss'])
        improved = np.array(self.candidates['loss'] < self.candidates['best_loss'])
        if np.sum(improved) > 0:
            best_loss = np.array(self.candidates['best_loss'])
            best_loss[improved] = np.array(self.candidates['loss'])[improved]
            best_xnn = np.stack(self.candidates['best_xnn'])
            best_xnn[improved] = np.stack(self.candidates['xnn'])[improved]
            self.candidates['best_loss'] = best_loss
            self.candidates['best_xnn'] = list(best_xnn)

        acceptance_trajectory = np.stack(self.candidates['acceptance_trajectory'])
        best_returns_trajectory = np.stack(self.candidates['best_returns_trajectory'])
        loss_trajectory = np.stack(self.candidates['loss_trajectory'])
        xnn_trajectory = np.stack(self.candidates['xnn_trajectory'])
        xnn_current = np.stack(self.candidates['xnn'])
        xnn_drift = np.stack(self.candidates['xnn_drift'])
        if np.isnan(np.sum(loss_trajectory)):
            # This catches the first iteration when there is no loss
            self.candidates['acceptance_trajectory'] = np.zeros(self.n, dtype=bool)
            self.candidates['best_returns_trajectory'] = np.zeros(self.n, dtype=int)
            self.candidates['loss_trajectory'] = np.array(self.candidates['loss'])
            self.candidates['xnn_trajectory'] = np.array(self.candidates['xnn'])
            self.candidates['xnn_drift'] = np.zeros(self.n)
        elif len(loss_trajectory.shape) == 1:
            self.candidates['acceptance_trajectory'] = list(np.concatenate((
                acceptance_trajectory[:, np.newaxis], np.array(self.candidates['accepted'])[:, np.newaxis]
                ),
                axis=1
                ))
            new_best_returns = best_returns_trajectory.copy()
            new_best_returns[returned] += 1
            new_best_returns[improved] = 0
            self.candidates['best_returns_trajectory'] = list(np.concatenate((
                best_returns_trajectory[:, np.newaxis], new_best_returns[:, np.newaxis]
                ),
                axis=1
                ))
            self.candidates['loss_trajectory'] = list(np.concatenate((
                loss_trajectory[:, np.newaxis], np.array(self.candidates['loss'])[:, np.newaxis]
                ),
                axis=1
                ))
            self.candidates['xnn_trajectory'] = list(np.concatenate((
                xnn_trajectory[:, :, np.newaxis], xnn_current[:, :, np.newaxis]
                ),
                axis=2
                ))
            xnn_drift_current = np.linalg.norm(
                xnn_trajectory[:, :3] - xnn_current[:, :3],
                axis=1
                )
            self.candidates['xnn_drift'] = list(np.concatenate((
                xnn_drift[:, np.newaxis], xnn_drift_current[:, np.newaxis]
                ),
                axis=1
                ))
        else:
            self.candidates['acceptance_trajectory'] = list(np.concatenate((
                acceptance_trajectory, np.array(self.candidates['accepted'])[:, np.newaxis]
                ),
                axis=1
                ))
            new_best_returns = best_returns_trajectory[:, -1].copy()
            new_best_returns[returned] += 1
            new_best_returns[improved] = 0
            self.candidates['best_returns_trajectory'] = list(np.concatenate((
                best_returns_trajectory, new_best_returns[:, np.newaxis]
                ),
                axis=1
                ))
            self.candidates['loss_trajectory'] = list(np.concatenate((
                loss_trajectory, np.array(self.candidates['loss'])[:, np.newaxis]
                ),
                axis=1
                ))
            self.candidates['xnn_trajectory'] = list(np.concatenate((
                xnn_trajectory, xnn_current[:, :, np.newaxis]
                ),
                axis=2
                ))
            xnn_drift_current = np.array(np.linalg.norm(
                xnn_trajectory[:, :3, 0] - xnn_current[:, :3],
                axis=1
                ))
            self.candidates['xnn_drift'] = list(np.concatenate((
                np.stack(self.candidates['xnn_drift']), xnn_drift_current[:, np.newaxis]
                ),
                axis=1
                ))

    def plot_history(self, save_to):
        """
        - loss vs iteration
        - xnn drift
        - number of times reaching best value
        - moving average of acceptance
        """
        loss_trajectory = np.stack(self.candidates['loss_trajectory'])
        xnn_drift = np.stack(self.candidates['xnn_drift'])
        best_returns_trajectory = np.stack(self.candidates['best_returns_trajectory'])
        acceptance_trajectory = np.stack(self.candidates['acceptance_trajectory'])
        xnn_trajectory = np.stack(self.candidates['xnn_trajectory'])
        found = np.min(loss_trajectory, axis=1) < self.tolerance

        unit_cell_true = get_different_monoclinic_settings(self.unit_cell_true, partial_unit_cell=True)
        reciprocal_unit_cell_true = reciprocal_uc_conversion(
            unit_cell_true, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        xnn_true = get_xnn_from_reciprocal_unit_cell(
            reciprocal_unit_cell_true, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        true_error = np.linalg.norm(
            xnn_trajectory[np.newaxis, :, :3, :] - xnn_true[:, np.newaxis, :3, np.newaxis],
            axis=2
            ).min(axis=0)
        """
        true_error = np.linalg.norm(
            xnn_trajectory[:, :3, :] - self.xnn_true[np.newaxis, :3, np.newaxis],
            axis=1
            )
        """
        """
        xnn_cc = np.zeros(xnn_trajectory.shape)
        for entry_index in range(xnn_trajectory.shape[0]):
            for axis_index in range(self.n_uc):
                xnn_cc[entry_index, axis_index] = scipy.signal.correlate(
                    xnn_trajectory[entry_index, axis_index],
                    xnn_trajectory[entry_index, axis_index],
                    method='fft'
                    )[xnn_trajectory.shape[2]-1:]

        xnn_cc /= xnn_cc.max(axis=2)[:, :, np.newaxis]
        fig, axes = plt.subplots(self.n_uc, 1, figsize=(5, 3 + self.n_uc*1.5), sharex=True)
        for axis_index in range(self.n_uc):
            axes[axis_index].plot(xnn_cc[~found, axis_index, :].T, alpha=0.1)
            axes[axis_index].plot(xnn_cc[found, axis_index, :].T, linewidth=2, color=[0, 0, 0])
        axes[0].set_title('Auto Correlation Plots')
        fig.tight_layout()
        fig.savefig(save_to + '_auto_correlation.png')
        plt.cla()
        plt.clf()
        plt.close('all')
        """

        running_average = np.zeros(acceptance_trajectory.shape)
        for index in range(1, acceptance_trajectory.shape[1]):
            if index < 10:
                running_average[:, index] = np.mean(acceptance_trajectory[:, :index], axis=1)
            else:
                running_average[:, index] = np.mean(acceptance_trajectory[:, index-10:index], axis=1)

        fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharex=True)
        axes[0, 0].plot(loss_trajectory[~found].T, alpha=0.1)
        axes[0, 0].plot(loss_trajectory[found].T, linewidth=2, color=[0, 0, 0])
        axes[0, 1].semilogy(xnn_drift[~found, 1:].T, alpha=0.1)
        axes[0, 1].semilogy(xnn_drift[found, 1:].T, linewidth=2, color=[0, 0, 0])
        axes[0, 2].semilogy(true_error[~found, 1:].T, alpha=0.1)
        axes[0, 2].semilogy(true_error[found, 1:].T, linewidth=2, color=[0, 0, 0])
        axes[1, 0].plot(running_average[~found].T, alpha=0.1)
        axes[1, 0].plot(running_average[found].T, linewidth=2, color=[0, 0, 0])
        axes[1, 1].plot(best_returns_trajectory[~found].T, alpha=0.1)
        axes[1, 1].plot(best_returns_trajectory[found].T, linewidth=2, color=[0, 0, 0])

        axes[0, 0].set_ylabel('Loss')
        axes[1, 0].set_ylabel('Acceptance Moving Average')
        axes[0, 1].set_ylabel('RMS Xnn drift')
        axes[0, 2].set_ylabel('RMS Xnn - Xnn True')
        axes[1, 1].set_ylabel('Times returned to best')
        axes[1, 0].set_xlabel('MCMC Iteration')
        axes[1, 1].set_xlabel('MCMC Iteration')
        axes[1, 2].set_xlabel('MCMC Iteration')

        ylim = axes[0, 1].get_ylim()
        axes[0, 1].set_ylim([1e-8, ylim[1]])
        ylim = axes[0, 2].get_ylim()
        axes[0, 2].set_ylim([1e-8, ylim[1]])

        fig.tight_layout()
        fig.savefig(save_to + '_MCMC_diagnostics.png')
        plt.cla()
        plt.clf()
        plt.close('all')

    def pick_explainers(self):
        found = self.candidates['loss'] < self.tolerance
        if np.count_nonzero(found) > 0:
            # If I keep the 'hkl' column I get an error:
            #  ValueError: all the input array dimensions except for the concatenation axis must match exactly
            # I believe this is due to a data type mismatch. The easiest way to deal with this was to drop the column
            found_entries = self.candidates.loc[found].copy().drop(columns=['hkl'])
            if len(self.explainers) == 0:
                self.explainers = found_entries
            else:
                found_index = np.array(found_entries.index)
                explainers_index = np.array(self.explainers.index)
                new = np.ones(len(found_entries), dtype=bool)
                for index in range(len(found_entries)):
                    if found_index[index] in explainers_index:
                        new[index] = False
                if np.sum(new) > 0:
                    self.explainers = pd.concat(
                        [self.explainers, found_entries.loc[new]], ignore_index=False
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
            reindexed_unit_cell, _ = reindex_entry_triclinic(unit_cell, radians=True)
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
            unit_cell = np.stack(self.candidates['unit_cell'])[:20]
            loss = np.array(self.candidates['loss'])[:20]
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

    def save_candidates(self, save_to, index):
        # This is a diagnostic function that saves info to work with in isolated cases.
        np.save(f'{save_to}/{self.bravais_lattice}_uc_true_{index:03d}.npy', self.unit_cell_true)
        np.save(f'{save_to}/{self.bravais_lattice}_q2_obs_{index:03d}.npy', self.q2_obs)
        np.save(f'{save_to}/{self.bravais_lattice}_xnn_{index:03d}.npy', np.stack(self.candidates['xnn']))
        np.save(f'{save_to}/{self.bravais_lattice}_hkl_{index:03d}.npy', np.stack(self.candidates['hkl']))
        np.save(f'{save_to}/{self.bravais_lattice}_softmax_{index:03d}.npy', np.stack(self.candidates['softmax']))
        np.save(f'{save_to}/{self.bravais_lattice}_unit_cell_{index:03d}.npy', np.stack(self.candidates['unit_cell']))


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
            'assignment_batch_size': 'max',
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
        self.indexer.setup_assignment()
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
            self.opt_params['minimum_angle_scaled'] = (0.01 - np.pi/2) / self.indexer.angle_scale
            self.opt_params['maximum_angle_scaled'] = (2*np.pi/3 - np.pi/2) / self.indexer.angle_scale
        elif self.indexer.data_params['lattice_system'] == 'monoclinic':
            # Minimum / Maximum angle = 90 / 180 degrees
            # Monoclinic angles are restricted to be above 90 degrees because
            # the same monoclinic unit cell can be represented with either an
            # obtuse or acute angle.
            # Scaling is (angle - pi/2) / angle_scale
            self.opt_params['minimum_angle_scaled'] = (np.pi/2 - np.pi/2) / self.indexer.angle_scale
            self.opt_params['maximum_angle_scaled'] = (np.pi - np.pi/2) / self.indexer.angle_scale
        elif self.indexer.data_params['lattice_system'] == 'triclinic':
            self.opt_params['minimum_beta_gamma_scaled'] = (np.pi/2 - np.pi/2) / self.indexer.angle_scale
            self.opt_params['minimum_alpha_scaled'] = (0.0 - np.pi/2) / self.indexer.angle_scale
            self.opt_params['maximum_angle_scaled'] = (np.pi - np.pi/2) / self.indexer.angle_scale

        self.n_groups = len(self.indexer.data_params['split_groups'])
        if self.opt_params['assignment_batch_size'] == 'max':
            n_candidates = self.n_groups * (self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf'])
            self.opt_params['assignment_batch_size'] = n_candidates + self.opt_params['n_candidates_template']
            self.one_assignment_batch = True
        else:
            self.one_assignment_batch = False

    def predictions(self):
        self.N = self.indexer.data.shape[0]
        uc_scaled_mean_filename = os.path.join(
            f'{self.save_to}', f'{self.bravais_lattice}_{self.data_params["tag"]}_uc_scaled_mean.npy'
            )
        uc_scaled_cov_filename = os.path.join(
            f'{self.save_to}', f'{self.bravais_lattice}_{self.data_params["tag"]}_uc_scaled_cov.npy'
            )
        if self.opt_params['load_predictions']:
            self.uc_scaled_mean = np.load(uc_scaled_mean_filename)
            self.uc_scaled_cov = np.load(uc_scaled_cov_filename)
        else:
            self.uc_scaled_mean = np.zeros((self.N, self.n_groups, self.indexer.data_params['n_outputs']))
            self.uc_scaled_cov = np.zeros((
                self.N,
                self.n_groups,
                self.indexer.data_params['n_outputs'],
                self.indexer.data_params['n_outputs']
                ))
            for group_index, group in enumerate(self.indexer.data_params['split_groups']):
                print(f'Performing predictions with {group}')
                uc_mean_scaled_group, uc_cov_scaled_group = self.indexer.unit_cell_generator[group].do_predictions(
                    data=self.indexer.data, verbose=0, batch_size=2048
                    )
                self.uc_scaled_mean[:, group_index, :] = uc_mean_scaled_group
                self.uc_scaled_cov[:, group_index, :, :] = uc_cov_scaled_group
            np.save(uc_scaled_mean_filename, self.uc_scaled_mean)
            np.save(uc_scaled_cov_filename, self.uc_scaled_cov)

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
                self.uc_scaled_cov[entry_index],
                self.indexer.data.iloc[entry_index],
                entry_index,
                )
            candidates, diagnostic_df_entry = self.optimize_entry(candidates, entry_index)
            #candidates.save_candidates(self.save_to, entry_index)
            report_counts, found = candidates.get_best_candidates(report_counts)
            diagnostic_df_entry['found'] = found
            diagnostic_df_entry['entry_index'] = entry_index
            diagnostic_df.loc[entry_index] = diagnostic_df_entry
            if self.opt_params['rerun_failures'] == False:
                diagnostic_df.to_json(
                    os.path.join(self.save_to, f'{self.bravais_lattice}_optimization_diagnostics.json')
                    )
            print(report_counts)
            end = time.time()
            print(end - start)

    def generate_unit_cells(self, uc_scaled_mean, uc_scaled_cov):
        if self.indexer.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            candidates_scaled = self.rng.multivariate_normal(
                mean=uc_scaled_mean,
                cov=uc_scaled_cov,
                size=self.opt_params['n_candidates_nn'],
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
                    scale=np.sqrt(uc_scaled_cov[0, 0]),
                    size=self.opt_params['n_candidates_nn']
                    )
                candidates_scaled[:, 1] = self.rng.uniform(
                    low=self.opt_params['minimum_angle_scaled'],
                    high=self.opt_params['maximum_angle_scaled'],
                    size=self.opt_params['n_candidates_nn']
                    )
            else:
                candidates_scaled = self.rng.multivariate_normal(
                    mean=uc_scaled_mean,
                    cov=uc_scaled_cov,
                    size=self.opt_params['n_candidates_nn'],
                    )
        elif self.indexer.data_params['lattice_system'] == 'monoclinic':
            uniform_angle = False
            if uc_scaled_mean[3] <= self.opt_params['minimum_angle_scaled']:
                uniform_angle = True
            elif uc_scaled_mean[3] >= self.opt_params['maximum_angle_scaled']:
                uniform_angle = True
            if uniform_angle:
                candidates_scaled = np.zeros((self.opt_params['n_candidates_nn'], 4))
                candidates_scaled[:, :3] = self.rng.multivariate_normal(
                    mean=uc_scaled_mean[:3],
                    cov=uc_scaled_cov[:3, :3],
                    size=self.opt_params['n_candidates_nn'],
                    )
                candidates_scaled[:, 3] = self.rng.uniform(
                    low=self.opt_params['minimum_angle_scaled'],
                    high=self.opt_params['maximum_angle_scaled'],
                    size=self.opt_params['n_candidates_nn']
                    )
            else:
                candidates_scaled = self.rng.multivariate_normal(
                    mean=uc_scaled_mean,
                    cov=uc_scaled_cov,
                    size=self.opt_params['n_candidates_nn'],
                    )
        elif self.indexer.data_params['lattice_system'] == 'triclinic':
            uniform_alpha = False
            uniform_beta = False
            uniform_gamma = False
            if uc_scaled_mean[3] <= self.opt_params['minimum_alpha_scaled']:
                uniform_alpha = True
            elif uc_scaled_mean[3] >= self.opt_params['maximum_angle_scaled']:
                uniform_alpha = True
            if uc_scaled_mean[4] <= self.opt_params['minimum_beta_gamma_scaled']:
                uniform_beta = True
            elif uc_scaled_mean[4] >= self.opt_params['maximum_angle_scaled']:
                uniform_beta = True
            if uc_scaled_mean[5] <= self.opt_params['minimum_beta_gamma_scaled']:
                uniform_gamma = True
            elif uc_scaled_mean[3] >= self.opt_params['maximum_angle_scaled']:
                uniform_gamma = True
            candidates_scaled = self.rng.multivariate_normal(
                mean=uc_scaled_mean,
                cov=uc_scaled_cov,
                size=self.opt_params['n_candidates_nn'],
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
                np.array(entry['reindexed_unit_cell']), partial_unit_cell=False, radians=True
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
            angle_bins = np.linspace(0, np.pi, 101)
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

    def generate_candidates(self, uc_scaled_mean, uc_scaled_cov, entry, entry_index):
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
                    uc_scaled_mean[group_index, :], uc_scaled_cov[group_index, :, :]
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

        if self.indexer.data_params['lattice_system'] == 'triclinic':
            candidate_unit_cells = fix_unphysical_triclinic(
                unit_cell=candidate_unit_cells,
                rng=self.rng,
                minimum_unit_cell=self.opt_params['minimum_uc'],
                maximum_unit_cell=self.opt_params['maximum_uc'],
                )
            for index in range(candidate_unit_cells.shape[0]):
                candidate_unit_cells[index], _ = reindex_entry_triclinic(candidate_unit_cells[index], radians=True)
        elif self.indexer.data_params['lattice_system'] == 'rhombohedral':
            candidate_unit_cells = fix_unphysical_rhombohedral(
                unit_cell=candidate_unit_cells,
                rng=self.rng,
                minimum_unit_cell=self.opt_params['minimum_uc'],
                maximum_unit_cell=self.opt_params['maximum_uc'],
                )

        self.plot_candidate_unit_cells(candidate_unit_cells, entry, entry_index)

        candidates = Candidates(
            entry=entry,
            unit_cell=candidate_unit_cells,
            lattice_system=self.indexer.data_params['lattice_system'],
            bravais_lattice=self.bravais_lattice,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            tolerance=self.opt_params['found_tolerance'],
            )
        candidates.fix_out_of_range_candidates()
        if self.opt_params['redistribute_candidates']:
            candidates.redistribute_unit_cells(
                self.opt_params['max_neighbors'], self.opt_params['neighbor_radius']
                )
        if self.opt_params['initial_assigner_key'] == 'closest':
            candidates = self.assign_hkls_closest(candidates)
        else:
            candidates = self.assign_hkls(
                candidates,
                self.opt_params['initial_assigner_key'],
                'best'
                )

        target_function = CandidateOptLoss_xnn(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        xnn = np.stack(candidates.candidates['xnn'])
        target_function.update(
            np.stack(candidates.candidates['hkl']), 
            np.ones((candidates.n, self.indexer.data_params['n_points'])),
            xnn
            )
        candidates.candidates['loss'] = target_function.get_loss(xnn)
        candidates.update()
        return candidates

    def update_annealing(self, iteration_info, iter_index):
        if iteration_info['simulated_annealing']:
            if iter_index == 0:
                N = iteration_info['simulated_annealing_stop_iteration']
                temp_start = iteration_info['simulated_annealing_starting_temp']
                temp_end = iteration_info['simulated_annealing_ending_temp']
                a = -1/N * np.log(temp_end / temp_start)
                self._annealing_temperature_schedule = np.zeros(iteration_info['n_iterations'])
                self._annealing_temperature_schedule[:N] = temp_start * np.exp(-a*np.arange(N))
                self._annealing_temperature_schedule[N:] = temp_end
            self.temperature = self._annealing_temperature_schedule[iter_index]
        else:
            self.temperature = 1

    def exhaustive_search(self, candidates, iteration_info, iter_index):
        if iteration_info['exhaustive_search']:
            if iter_index == 0:
                candidates.setup_exhaustive_search(
                    self.opt_params['max_neighbors'], self.opt_params['neighbor_radius']
                    )
            elif iter_index % iteration_info['exhaustive_search_period'] == 0:
                candidates.exhaustive_search()

    def optimize_entry(self, candidates, entry_index):
        diagnostic_df_entry = candidates.diagnostics(self.indexer.data_params['hkl_ref_length'])
        acceptance_fraction = []
        for iteration_info in self.opt_params['iteration_info']:
            if iteration_info['worker'] == 'monoclinic_reset':
                #candidates.save_candidates(self.save_to, entry_index)
                candidates = self.monoclinic_reset(candidates, iteration_info)
                print_list = [
                    f'{candidates.n},',
                    f'{len(candidates.explainers)},',
                    f'{candidates.candidates["loss"].mean():0.2f},',
                    f'{candidates.candidates["loss"].min():0.2f},',
                    f'{iteration_info["worker"]},',
                    f'{iteration_info["assigner_key"]}',
                    ]
                print(' '.join(print_list))
            else:
                reindex_count = 0
                for iter_index in range(iteration_info['n_iterations']):
                    self.update_annealing(iteration_info, iter_index)
                    self.exhaustive_search(candidates, iteration_info, iter_index)
                    if iteration_info['worker'] in ['softmax_subsampling', 'random_subsampling']:
                        next_xnn = self.random_subsampling(candidates, iteration_info)
                    elif iteration_info['worker'] in ['resampling', 'resampling_softmax_subsampling']:
                        next_candidates = copy.copy(candidates)
                        next_candidates = self.assign_hkls(
                            next_candidates, iteration_info['assigner_key'], 'random'
                            )
                        if iteration_info['worker'] == 'resampling':
                            next_xnn = self.no_subsampling(next_candidates, iteration_info)
                        elif iteration_info['worker'] == 'resampling_softmax_subsampling':
                            next_xnn = self.random_subsampling(next_candidates, iteration_info)
                    elif iteration_info['worker'] == 'no_sampling':
                        next_xnn = self.no_subsampling(candidates, iteration_info)
                    else:
                        assert False
                    candidates, acceptance_fraction = self.update_candidates(
                        candidates,
                        iteration_info['assigner_key'],
                        iteration_info['acceptance_method'],
                        next_xnn,
                        acceptance_fraction 
                        )
                    if candidates.n <= 1:
                        return candidates, diagnostic_df_entry
                    if len(candidates.explainers) > self.opt_params['max_explainers']:
                        return candidates, diagnostic_df_entry
                    print_list = [
                        f'{candidates.n},',
                        f'{len(candidates.explainers)},',
                        f'{candidates.candidates["loss"].mean():0.2f},',
                        f'{candidates.candidates["loss"].min():0.2f},',
                        f'{iteration_info["worker"]},',
                        f'{iteration_info["assigner_key"]}',
                        ]
                    print(' '.join(print_list))
                save_to = os.path.join(
                    self.save_to,
                    'figures',
                    f'{self.bravais_lattice}_{entry_index:03d}_{self.opt_params["tag"]}'
                    )
                if iteration_info['plot_history']:
                    candidates.plot_history(save_to)
        print(np.mean(acceptance_fraction))
        return candidates, diagnostic_df_entry

    def get_softmaxes(self, candidates, assigner_key):
        """
        I'm using model.predict_on_batch as a workaround for memory leak issues
        - Inputs must be the same size each time predict_on_batch is called.
        - model.call() / model() and model.predict() lead to a bad memory leak
        - model.predict_on_batch when the batch size is different leads to the same memory leak
        - tf.nn.softmax gives a small memory leak as well.
        https://github.com/tensorflow/tensorflow/issues/44711
        https://github.com/keras-team/keras/issues/13118
        https://github.com/tensorflow/tensorflow/issues/33009        
        """
        n_batchs = candidates.n // self.opt_params['assignment_batch_size']
        left_over = candidates.n % self.opt_params['assignment_batch_size']
        xnn = np.stack(candidates.candidates['xnn'])
        batch_q2_scaled = np.repeat(
            candidates.q2_obs_scaled[np.newaxis, :],
            repeats=self.opt_params['assignment_batch_size'],
            axis=0
            )
        if self.one_assignment_batch:
            batch_xnn = np.zeros((
                self.opt_params['assignment_batch_size'], self.indexer.data_params['n_outputs']
                ))
            batch_xnn[:candidates.n] = xnn
            inputs = {
                'xnn': batch_xnn,
                'q2_scaled': batch_q2_scaled
                }
            # This is a bottleneck
            softmaxes = self.indexer.assigner[self.bravais_lattice][assigner_key].model.predict_on_batch(inputs)[:candidates.n]
        else:
            softmaxes = np.zeros((
                candidates.n,
                self.indexer.data_params['n_points'],
                self.indexer.data_params['hkl_ref_length']
                ))

            for batch_index in range(n_batchs + 1):
                start = batch_index * self.opt_params['assignment_batch_size']
                end = (batch_index + 1) * self.opt_params['assignment_batch_size']
                if batch_index == n_batchs:
                    batch_xnn = np.zeros((
                        self.opt_params['assignment_batch_size'], self.indexer.data_params['n_outputs']
                        ))
                    batch_xnn[:left_over] = xnn[start: start + left_over]
                    batch_xnn[left_over:] = batch_xnn[0]
                else:
                    batch_xnn = xnn[start: end]
                inputs = {
                    'xnn': batch_xnn,
                    'q2_scaled': batch_q2_scaled
                    }
                # This is a bottleneck
                softmaxes_batch = self.indexer.assigner[self.bravais_lattice][assigner_key].model.predict_on_batch(inputs)
                if batch_index == n_batchs:
                    softmaxes[start: start + left_over] = softmaxes_batch[:left_over]
                else:
                    softmaxes[start: end] = softmaxes_batch
        return softmaxes

    def assign_hkls(self, candidates, assigner_key, assignment_method):
        if assigner_key.startswith('random:'):
            choices = assigner_key.split('random:')[1].split(',')
            assigner_key = choices[self.rng.choice(len(choices), size=1)[0]]
        softmaxes = self.get_softmaxes(candidates, assigner_key)

        if assignment_method == 'best':
            hkl_assign, softmax_assign = best_assign_nocommon(softmaxes)
            candidates.candidates['softmax'] = list(softmax_assign)
        elif assignment_method == 'best_with_redundancies':
            hkl_assign = softmaxes.argmax(axis=2)
            candidates.candidates['softmax'] = list(softmaxes.max(axis=2))
        elif assignment_method == 'random':
            hkl_assign, softmax = vectorized_resampling(softmaxes, self.rng)
            candidates.candidates['softmax'] = list(softmax)
        else:
            assert False
        hkl_pred = np.zeros((candidates.n, self.indexer.data_params['n_points'], 3))
        for entry_index in range(candidates.n):
            hkl_pred[entry_index] = self.indexer.hkl_ref[self.bravais_lattice][hkl_assign[entry_index]]
        candidates.candidates['hkl'] = list(hkl_pred)
        return candidates

    def assign_hkls_closest(self, candidates):
        xnn = np.stack(candidates.candidates['xnn'])
        q2_obs_scaled = np.repeat(
            candidates.q2_obs_scaled[np.newaxis, :], repeats=candidates.n, axis=0
            )
        """
        pairwise_differences_scaled = self.indexer.assigner[self.bravais_lattice][self.opt_params['initial_assigner_key']].pairwise_difference_calculation_numpy.get_pairwise_differences(
            xnn, q2_obs_scaled
            )
        """
        pairwise_differences_scaled = self.indexer.assigner[self.bravais_lattice]['0'].pairwise_difference_calculation_numpy.get_pairwise_differences(
            xnn, q2_obs_scaled
            )
        candidates.candidates['softmax'] = list(np.ones((candidates.n, self.indexer.data_params['n_points'])))

        hkl_assign = np.abs(pairwise_differences_scaled).argmin(axis=2)
        hkl_pred = np.zeros((candidates.n, self.indexer.data_params['n_points'], 3))
        for entry_index in range(candidates.n):
            hkl_pred[entry_index] = self.indexer.hkl_ref[self.bravais_lattice][hkl_assign[entry_index]]
        candidates.candidates['hkl'] = list(hkl_pred)
        return candidates

    def montecarlo_acceptance(self, candidates, next_candidates, acceptance_method, acceptance_fraction):
        if acceptance_method == 'montecarlo':
            ratio = np.exp(-(next_candidates.candidates['loss'] - candidates.candidates['loss']) / self.temperature)
            probability = self.rng.random(candidates.n)
            accepted = probability < ratio
            acceptance_fraction.append(accepted.sum() / accepted.size)
            candidates.candidates.loc[accepted] = next_candidates.candidates.loc[accepted]
            candidates.candidates['accepted'] = accepted
        elif acceptance_method == 'always':
            candidates = next_candidates
            acceptance_fraction.append(1)
        else:
            assert False, 'Unrecognized acceptance method'
        return candidates, acceptance_fraction

    def update_candidates(self, candidates, assigner_key, acceptance_method, next_xnn, acceptance_fraction):
        next_candidates = copy.deepcopy(candidates)
        next_candidates.candidates['xnn'] = list(next_xnn)
        next_candidates.update_unit_cell_from_xnn()

        if assigner_key == 'closest':
            next_candidates = self.assign_hkls_closest(next_candidates)
        else:
            next_candidates = self.assign_hkls(next_candidates, assigner_key, 'best')
        target_function = CandidateOptLoss_xnn(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )

        target_function.update(
            np.stack(next_candidates.candidates['hkl']),
            np.ones((candidates.n, self.indexer.data_params['n_points'])),
            next_xnn
            )
        next_candidates.candidates['loss'] = target_function.get_loss(next_xnn)

        candidates, acceptance_fraction = self.montecarlo_acceptance(
            candidates,
            next_candidates,
            acceptance_method,
            acceptance_fraction
            )
        candidates.update()
        return candidates, acceptance_fraction

    def no_subsampling(self, candidates, iteration_info):
        target_function = CandidateOptLoss_xnn(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        xnn = np.stack(candidates.candidates['xnn'])

        target_function.update(
            np.stack(candidates.candidates['hkl']), 
            np.stack(candidates.candidates['softmax']), 
            xnn
            )
        delta_gn = target_function.gauss_newton_step(xnn)
        next_xnn = xnn + delta_gn
        return next_xnn

    def random_subsampling(self, candidates, iteration_info):
        hkl = np.stack(candidates.candidates['hkl'])
        softmax = np.stack(candidates.candidates['softmax'])
        xnn = np.stack(candidates.candidates['xnn'])
        n_keep = self.indexer.data_params['n_points'] - iteration_info['n_drop']
        if iteration_info['worker'] in ['random_subsampling', 'resampling_random_subsampling']:
            # The vectorized_subsampling function can be simplified and sped up considerably
            # in the case that each assignment has the same probability.
            # This implementation works, but would be faster for a function specific to this case.
            softmaxes_ = np.ones(softmax.shape)
            p = softmaxes_ / softmaxes_.sum(axis=1)[:, np.newaxis]
            subsampled_indices = vectorized_subsampling(p, n_keep, self.rng)
        elif iteration_info['worker'] in ['softmax_subsampling', 'resampling_softmax_subsampling']:
            p = softmax / softmax.sum(axis=1)[:, np.newaxis]
            subsampled_indices = vectorized_subsampling(p, n_keep, self.rng)

        hkl_subsampled = np.zeros((candidates.n, n_keep, 3))
        softmax_subsampled = np.zeros((candidates.n, n_keep))
        q2_subsampled = np.zeros((candidates.n, n_keep))
        for candidate_index in range(candidates.n):
            hkl_subsampled[candidate_index] = hkl[candidate_index, subsampled_indices[candidate_index]]
            softmax_subsampled[candidate_index] = softmax[candidate_index, subsampled_indices[candidate_index]]
            q2_subsampled[candidate_index] = candidates.q2_obs[subsampled_indices[candidate_index]]

        target_function = CandidateOptLoss_xnn(
            q2_subsampled, 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        target_function.update(hkl_subsampled, softmax_subsampled, xnn)
        delta_gn = target_function.gauss_newton_step(xnn)
        next_xnn = xnn + delta_gn
        return next_xnn

    def monoclinic_reset(self, candidates, iteration_info):
        assert self.indexer.data_params['lattice_system'] == 'monoclinic'

        # This is set to true so only one batch is run during Miller index assignmnent
        # Now we have a different amount of candidates, so we must use batched assignments
        self.one_assignment_batch = False
        candidates.monoclinic_reset(iteration_info['n_best'], iteration_info['n_angles'])
        if iteration_info['assigner_key'] == 'closest':
            candidates = self.assign_hkls_closest(candidates)
        else:
            candidates = self.assign_hkls(candidates, iteration_info['assigner_key'], 'best')

        xnn = np.stack(candidates.candidates['xnn'])
        target_function = CandidateOptLoss_xnn(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            epsilon=self.opt_params['epsilon'],
            )
        target_function.update(
            np.stack(candidates.candidates['hkl']),
            np.ones((candidates.n, self.indexer.data_params['n_points'])),
            xnn
            )
        candidates.candidates['loss'] = target_function.get_loss(xnn)
        return candidates
