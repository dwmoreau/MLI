import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.optimize
import scipy.special
import time

from Indexing import Indexing
from TargetFunctions import CandidateOptLoss_xnn
from Utilities import get_hkl_checks
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
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]

    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    random_values = rng.random(size=(n_entries, n_peaks))
    point_order = rng.permutation(n_peaks)
    softmaxes_zeroed = softmaxes.copy()
    for point_index in point_order:
        cumsum = np.cumsum(softmaxes_zeroed[:, point_index, :], axis=1)
        q = cumsum >= random_values[:, point_index][:, np.newaxis]
        hkl_assign[:, point_index] = np.argmax(q, axis=1)
        for candidate_index in range(n_entries):
            softmaxes_zeroed[candidate_index, :, hkl_assign[candidate_index, point_index]] = 0
        softmaxes_zeroed /= softmaxes_zeroed.sum(axis=2)[:, :, np.newaxis]

    softmax = np.zeros((n_entries, n_peaks))
    for candidate_index in range(n_entries):
        for point_index in range(n_peaks):
            softmax[candidate_index, point_index] = softmaxes[
                candidate_index,
                point_index,
                hkl_assign[candidate_index, point_index]
                ]
    return hkl_assign, softmax


def best_assign_nocommon(softmaxes):
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]
    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    softmax_assign = np.zeros((n_entries, n_peaks))

    peak_choice = np.argsort(np.max(softmaxes, axis=2), axis=1)
    for candidate_index in range(n_entries):
        softmaxes_zeroed = softmaxes[candidate_index].copy()
        for peak_index in peak_choice[candidate_index]:
            choice = np.argmax(softmaxes_zeroed[peak_index, :])
            hkl_assign[candidate_index, peak_index] = choice
            softmaxes_zeroed[:, hkl_assign[candidate_index, peak_index]] = 0
            softmax_assign[candidate_index, peak_index] = softmaxes[candidate_index, peak_index, choice]
    return hkl_assign, softmax_assign


class Candidates:
    def __init__(self,
        entry,
        unit_cell, unit_cell_scaled, unit_cell_pred,
        lattice_system,
        minimum_unit_cell, maximum_unit_cell,
        tolerance
        ):
        self.q2_obs = np.array(entry['q2'])
        self.q2_obs_scaled = np.array(entry['q2_scaled'])
        self.n_points = self.q2_obs.size

        self.unit_cell_pred = unit_cell_pred
        self.n = unit_cell.shape[0]
        self.n_uc = unit_cell.shape[1]

        self.lattice_system = lattice_system
        self.minimum_unit_cell = minimum_unit_cell
        self.maximum_unit_cell = maximum_unit_cell
        if self.lattice_system == 'rhombohedral':
            self.maximum_angle = 2*np.pi/3
            self.minimum_angle = 0.01
        elif self.lattice_system == 'monoclinic':
            self.maximum_angle = np.pi
            self.minimum_angle = 4*np.pi/9
        elif self.lattice_system == 'triclinic':
            assert False
            self.maximum_angle = np.pi
            self.minimum_angle = 0.01
        self.rng = np.random.default_rng()
        self.tolerance = tolerance

        unit_cell_true = np.array(entry['reindexed_unit_cell'])
        reciprocal_unit_cell_true = reciprocal_uc_conversion(unit_cell_true[np.newaxis])[0]
        if lattice_system == 'cubic':
            self.unit_cell_true = unit_cell_true[0][np.newaxis]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[0][np.newaxis]
        elif lattice_system in ['tetragonal', 'hexagonal']:
            self.unit_cell_true = unit_cell_true[[0, 2]]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[[0, 2]]
        elif lattice_system == 'rhombohedral':
            self.unit_cell_true = unit_cell_true[[0, 3]]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[[0, 3]]
        elif lattice_system == 'orthorhombic':
            self.unit_cell_true = unit_cell_true[:3]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[:3]
        elif lattice_system == 'monoclinic':
            self.unit_cell_true = unit_cell_true[[0, 1, 2, 4]]
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true[[0, 1, 2, 4]]
        elif lattice_system == 'triclinic':
            self.unit_cell_true = unit_cell_true
            self.reciprocal_unit_cell_true = reciprocal_unit_cell_true

        self.hkl_true = np.array(entry['reindexed_hkl'])[:, :, 0]
        self.hkl_labels_true = np.array(entry['hkl_labels'])
        self.bl_true = entry['bravais_lattice']
        self.sg_true = int(entry['spacegroup_number'])
        self.spacegroup_symbol_hm_true = entry['reindexed_spacegroup_symbol_hm']

        self.df_columns = [
            'unit_cell',
            'unit_cell_scaled',
            'reciprocal_unit_cell',
            'xnn',
            'hkl',
            'softmax',
            'loss',
            ]
        self.candidates = pd.DataFrame(columns=self.df_columns)
        self.explainers = pd.DataFrame(columns=self.df_columns)

        self.candidates['unit_cell'] = list(unit_cell)
        self.candidates['unit_cell_scaled'] = list(unit_cell_scaled)
        self.update_xnn_from_unit_cell()
        self.hkl_true_check = get_hkl_checks(self.hkl_true, self.lattice_system)

    def update_xnn_from_unit_cell(self):
        unit_cell = np.stack(self.candidates['unit_cell'])
        unit_cell_full = np.zeros((self.n, 6))
        if self.lattice_system == 'cubic':
            unit_cell_full[:, :3] = unit_cell[:, 0][:, np.newaxis]
            unit_cell_full[:, 3:] = np.pi/2
        elif self.lattice_system == 'tetragonal':
            unit_cell_full[:, :2] = unit_cell[:, 0][:, np.newaxis]
            unit_cell_full[:, 2] = unit_cell[:, 1]
            unit_cell_full[:, 3:] = np.pi/2
        elif self.lattice_system == 'hexagonal':
            unit_cell_full[:, :2] = unit_cell[:, 0][:, np.newaxis]
            unit_cell_full[:, 2] = unit_cell[:, 1]
            unit_cell_full[:, 3] = np.pi/2
            unit_cell_full[:, 4] = np.pi/2
            unit_cell_full[:, 5] = 2/3 * np.pi
        elif self.lattice_system == 'rhombohedral':
            unit_cell_full[:, :3] = unit_cell[:, 0][:, np.newaxis]
            unit_cell_full[:, 3:] = unit_cell[:, 1][:, np.newaxis]
        elif self.lattice_system == 'orthorhombic':
            unit_cell_full[:, :3] = unit_cell[:, :3]
            unit_cell_full[:, 3:] = np.pi/2
        elif self.lattice_system == 'monoclinic':
            unit_cell_full[:, :3] = unit_cell[:, :3]
            unit_cell_full[:, 4] = unit_cell[:, 3]
            unit_cell_full[:, 3] = np.pi/2
            unit_cell_full[:, 5] = np.pi/2
        elif self.lattice_system == 'triclinic':
            unit_cell_full = unit_cell_pred
        else:
            assert False

        reciprocal_unit_cell_full = reciprocal_uc_conversion(unit_cell_full)
        xnn_full = get_xnn_from_reciprocal_unit_cell(reciprocal_unit_cell_full)

        if self.lattice_system == 'cubic':
            reciprocal_unit_cell = reciprocal_unit_cell_full[:, 0][:, np.newaxis]
            xnn = xnn_full[:, 0][:, np.newaxis]
        elif self.lattice_system in ['tetragonal', 'hexagonal']:
            reciprocal_unit_cell = reciprocal_unit_cell_full[:, [0, 2]]
            xnn = xnn_full[:, [0, 2]]
        elif self.lattice_system == 'rhombohedral':
            reciprocal_unit_cell = reciprocal_unit_cell_full[:, [0, 3]]
            xnn = xnn_full[:, [0, 3]]
        elif self.lattice_system == 'orthorhombic':
            reciprocal_unit_cell = reciprocal_unit_cell_full[:, [0, 1, 2]]
            xnn = xnn_full[:, [0, 1, 2]]
        elif self.lattice_system == 'monoclinic':
            reciprocal_unit_cell = reciprocal_unit_cell_full[:, [0, 1, 2, 4]]
            xnn = xnn_full[:, [0, 1, 2, 4]]
        elif self.lattice_system == 'triclinic':
            reciprocal_unit_cell = reciprocal_unit_cell_full
            xnn = xnn_full

        self.candidates['reciprocal_unit_cell'] = list(reciprocal_unit_cell)
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
            too_small = xnn[:, 0] < (1 / self.maximum_unit_cell)**2
            too_large = xnn[:, 0] > (1 / self.minimum_unit_cell)**2
            xnn[:, 0][too_small] = (1 / self.maximum_unit_cell)**2
            xnn[:, 0][too_large] = (1 / self.minimum_unit_cell)**2
            cos_ralpha = xnn[:, 1] / (2*xnn[:, 0])
            too_small = cos_ralpha < -1
            too_large = cos_ralpha > 1
            xnn[too_small, 1] = -2*0.999 * xnn[too_small, 0]
            xnn[too_large, 1] = 2*0.999 * xnn[too_large, 0]
        elif self.lattice_system in ['monoclinic', 'triclinic']:
            too_small = xnn[:, :3] < (1 / self.maximum_unit_cell)**2
            too_large = xnn[:, :3] > (1 / self.minimum_unit_cell)**2
            xnn[:, :3][too_small] = (1 / self.maximum_unit_cell)**2
            xnn[:, :3][too_large] = (1 / self.minimum_unit_cell)**2
            if self.lattice_system == 'monoclinic':
                cos_rbeta = xnn[:, 3] / (2 * np.sqrt(xnn[:, 0] * xnn[:, 2]))
                too_small = cos_rbeta < -1
                too_large = cos_rbeta > 1
                xnn[too_small, 3] = -2*0.999 * np.sqrt(xnn[too_small, 0] * xnn[too_small, 2])
                xnn[too_large, 3] = 2*0.999 * np.sqrt(xnn[too_large, 0] * xnn[too_large, 2])
            elif self.lattice_system == 'triclinic':
                assert False

        self.candidates['xnn'] = list(xnn)

        unit_cell_full = np.zeros((self.n, 6))
        xnn_full = np.zeros((self.n, 6))
        if self.lattice_system == 'cubic':
            xnn_full[:, :3] = xnn[:, 0][:, np.newaxis]
        elif self.lattice_system == 'tetragonal':
            xnn_full[:, :2] = xnn[:, 0][:, np.newaxis]
            xnn_full[:, 2] = xnn[:, 1]
        elif self.lattice_system == 'hexagonal':
            xnn_full[:, :2] = xnn[:, 0][:, np.newaxis]
            xnn_full[:, 2] = xnn[:, 1]
            xnn_full[:, 5] = 2*xnn[:, 0]*np.cos(np.pi/3)
        elif self.lattice_system == 'rhombohedral':
            xnn_full[:, :3] = xnn[:, 0][:, np.newaxis]
            xnn_full[:, 3:] = xnn[:, 1][:, np.newaxis]
        elif self.lattice_system == 'orthorhombic':
            xnn_full[:, :3] = xnn[:, :3]
        elif self.lattice_system == 'monoclinic':
            xnn_full[:, :3] = xnn[:, :3]
            xnn_full[:, 4] = xnn[:, 3]
        elif self.lattice_system == 'triclinic':
            xnn_full = xnn

        reciprocal_unit_cell_full = get_reciprocal_unit_cell_from_xnn(xnn_full)
        unit_cell_full = reciprocal_uc_conversion(reciprocal_unit_cell_full)
        
        if self.lattice_system == 'cubic':
            unit_cell = unit_cell_full[:, 0][:, np.newaxis]
        elif self.lattice_system in ['tetragonal', 'hexagonal']:
            unit_cell = unit_cell_full[:, [0, 2]]
        elif self.lattice_system == 'rhombohedral':
            unit_cell = unit_cell_full[:, [0, 3]]
        elif self.lattice_system == 'orthorhombic':
            unit_cell = unit_cell_full[:, :3]
        elif self.lattice_system == 'monoclinic':
            unit_cell = unit_cell_full[:, [0, 1, 2, 4]]
        elif self.lattice_system == 'triclinic':
            unit_cell = unit_cell_full

        self.candidates['unit_cell'] = list(unit_cell)

    def diagnostics(self, hkl_ref_length):
        reciprocal_unit_cell = np.stack(self.candidates['reciprocal_unit_cell'])
        reciprocal_unit_cell_rms = 1/np.sqrt(self.n_uc) * np.linalg.norm(reciprocal_unit_cell - self.reciprocal_unit_cell_true, axis=1)
        reciprocal_unit_cell_max_diff = np.max(np.abs(reciprocal_unit_cell - self.reciprocal_unit_cell_true), axis=1)

        hkl = np.stack(self.candidates['hkl'])
        hkl_pred_check = get_hkl_checks(hkl, self.lattice_system)
        hkl_correct = self.hkl_true_check[np.newaxis] == hkl_pred_check
        hkl_accuracy = np.count_nonzero(hkl_correct, axis=1) / self.n_points

        impossible = np.any(self.hkl_labels_true == hkl_ref_length - 1)

        print(f'Starting # candidates:       {self.n}')
        print(f'Impossible:                  {impossible}')
        print(f'True unit cell:              {np.round(self.unit_cell_true, decimals=4)}')
        print(f'True reciprocal unit cell:   {np.round(self.reciprocal_unit_cell_true, decimals=4)}')
        print(f'Closest unit cell:           {np.round(reciprocal_unit_cell[np.argmin(reciprocal_unit_cell_rms)], decimals=4)}')
        print(f'Closest unit cell rms:       {reciprocal_unit_cell_rms.min():2.2f}')
        print(f'Smallest unit cell max diff: {reciprocal_unit_cell_max_diff.min():2.2f}')
        print(f'Mean unit cell rms:          {reciprocal_unit_cell_rms.mean():2.2f}')
        print(f'Best HKL accuracy:           {hkl_accuracy.max():1.2f}')
        print(f'Mean HKL accuracy:           {hkl_accuracy.mean():1.2f}')
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
            'found': False
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
            too_small_lengths = unit_cells[:, 0] < self.minimum_unit_cell
            too_large_lengths = unit_cells[:, 0] > self.maximum_unit_cell
            if np.sum(too_small_lengths) > 0:
                unit_cells[too_small_lengths, 0] = self.rng.uniform(
                    low=self.minimum_unit_cell,
                    high=1.05*self.minimum_unit_cell,
                    size=np.sum(too_small_lengths)
                    )
            if np.sum(too_large_lengths) > 0:
                unit_cells[too_large_lengths, 0] = self.rng.uniform(
                    low=0.95*self.maximum_unit_cell,
                    high=self.maximum_unit_cell,
                    size=np.sum(too_large_lengths)
                    )
            too_small_angles = unit_cells[:, 1] < self.minimum_angle
            too_large_angles = unit_cells[:, 1] > self.maximum_angle
            if np.sum(too_small_angles) > 0:
                unit_cells[too_small_angles, 1] = self.rng.uniform(
                    low=self.minimum_angle,
                    high=1.05*self.minimum_angle,
                    size=np.sum(too_small_angles)
                    )
            if np.sum(too_large_angles) > 0:
                unit_cells[too_large_angles, 0] = self.rng.uniform(
                    low=0.95*self.maximum_angle,
                    high=self.maximum_angle,
                    size=np.sum(too_large_angles)
                    )
        elif self.lattice_system in ['monoclinic', 'triclinic']:
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
            
            if self.lattice_system == 'triclinic':
                too_small_angles = unit_cells[:, 3:] < self.minimum_angle
                too_large_angles = unit_cells[:, 3:] > self.maximum_angle
                if np.sum(too_small_angles) > 0:
                    indices = np.argwhere(too_small_angles)
                    unit_cells[indices[:, 0], 3 + indices[:, 1]] = self.rng.uniform(
                        low=self.minimum_angle,
                        high=1.05*self.minimum_angle,
                        size=np.sum(too_small_angles)
                        )
                if np.sum(too_large_angles) > 0:
                    indices = np.argwhere(too_large_angles)
                    unit_cells[indices[:, 0], 3 + indices[:, 1]] = self.rng.uniform(
                        low=0.95*self.maximum_angle,
                        high=self.maximum_angle,
                        size=np.sum(too_large_angles)
                        )
            elif self.lattice_system == 'monoclinic':
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

    def update(self):
        if self.n > 1:
            bad_candidates = np.isnan(self.candidates['loss'])
            if np.sum(bad_candidates) > 0:
                self.candidates = self.candidates.loc[~bad_candidates]
                self.n = len(self.candidates)

            # Fix candidates with too small or too large unit cells
            self.fix_out_of_range_candidates()
        if len(self.candidates) > 1:
            self.pick_explainers()
        self.candidates = self.candidates.sort_values(by='loss')

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
                self.explainers = pd.concat([self.explainers, found_entries], ignore_index=True)
                self.explainers.sort_values(by='loss', inplace=True)
            self.candidates = self.candidates.loc[~found]
            self.n = len(self.candidates)

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
            mult_factors = np.array([1/2, 1, 2])
            for mf0 in mult_factors:
                for mf1 in mult_factors:
                    mf = np.array([mf0, mf1])
                    if np.all(np.isclose(self.unit_cell_true, mf * unit_cell, atol=atol)):
                        return False, True
        elif self.lattice_system == 'rhombohedral':
            if np.all(np.isclose(self.unit_cell_true, unit_cell, atol=atol)):
                return True, False
            mult_factors = np.array([1/2, 2])
            for mf in mult_factors:
                if np.isclose(self.unit_cell_true, np.array([mf0, 1]) * unit_cell, atol=atol):
                    return False, True
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
            if np.all(np.isclose(self.unit_cell_true, unit_cell, atol=atol)):
                return True, False
        return False, False

    def get_best_candidates(self, report_counts):
        found = False
        if len(self.explainers) == 0:
            print(np.stack(self.candidates['reciprocal_unit_cell'])[:10].round(decimals=4))
            report_counts['Not found'] += 1
        else:
            if len(self.explainers) == 1:
                print(self.explainers[['unit_cell', 'loss']].round(decimals={'unit_cell': 3, 'loss': 1}))
            else:
                uc_print = np.stack(self.explainers['unit_cell']).round(decimals=3)
                loss_print = np.array(self.explainers['loss']).round(decimals=1)
                print(np.concatenate((uc_print, loss_print[:, np.newaxis]), axis=1))
            found_best = False
            found_not_best = False
            found_off_by_two = False
            for explainer_index in range(len(self.explainers)):
                unit_cell = np.array(self.explainers.iloc[explainer_index]['unit_cell'])
                correct, off_by_two = self.validate_candidate(unit_cell)
                if correct and explainer_index == 0:
                    found_best = True
                elif correct:
                    found_not_best = True
                elif off_by_two:
                    found_off_by_two = True
            if found_best:
                report_counts['Found and best'] += 1
                found = True
            elif found_not_best:
                report_counts['Found but not best'] += 1
                found = True
            elif found_off_by_two:
                report_counts['Found but off by two'] += 1
                found = True
            else:
                report_counts['Found explainers'] += 1
                found = True
        return report_counts, found

    def monoclinic_reset(self, n_best, n_angles):
        reciprocal_unit_cell = np.stack(self.candidates['reciprocal_unit_cell'])
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
        np.save(f'{save_to}/uc_true_{index:03d}.npy', self.unit_cell_true)
        np.save(f'{save_to}/q2_obs_{index:03d}.npy', self.q2_obs)
        np.save(f'{save_to}/xnn_{index:03d}.npy', np.stack(self.candidates['xnn']))
        np.save(f'{save_to}/hkl_{index:03d}.npy', np.stack(self.candidates['hkl']))
        np.save(f'{save_to}/softmax_{index:03d}.npy', np.stack(self.candidates['softmax']))
        np.save(f'{save_to}/unit_cell_{index:03d}.npy', np.stack(self.candidates['unit_cell']))


class Optimizer:
    def __init__(self, assign_params, data_params, opt_params, reg_params):
        self.assign_params = assign_params
        self.data_params = data_params
        self.opt_params = opt_params
        self.reg_params = reg_params
        self.rng = np.random.default_rng()

        opt_params_defaults = {
            'n_candidates_nn': 30,
            'n_candidates_rf': 30,
            'n_candidates_template': 30,
            'minimum_uc': 2,
            'maximum_uc': 500,
            'found_tolerance': -200,
            'assignment_batch_size': 'max',
            'load_predictions': False,
            'max_explainers': 20,
            }
        for key in opt_params_defaults.keys():
            if key not in self.opt_params.keys():
                self.opt_params[key] = opt_params_defaults[key]
        for key in self.assign_params:
            self.assign_params[key]['load_from_tag'] = True
            self.assign_params[key]['mode'] = 'inference'
        for key in self.reg_params:
            self.reg_params[key]['load_from_tag'] = True
            self.reg_params[key]['alpha_params'] = {}
            self.reg_params[key]['beta_params'] = {}
            self.reg_params[key]['mean_params'] = {}
            self.reg_params[key]['var_params'] = {}
            self.reg_params[key]['head_params'] = {}
        self.data_params['load_from_tag'] = True

        self.save_to = os.path.join(
            self.data_params['base_directory'], 'models', self.data_params['tag'], 'optimizer'
            )
        if not os.path.exists(self.save_to):
            os.mkdir(self.save_to)

        self.indexer = Indexing(
            assign_params=self.assign_params, 
            data_params=self.data_params,
            reg_params=self.reg_params, 
            seed=12345, 
            )
        self.indexer.setup_from_tag()
        self.indexer.load_data_from_tag(load_augmented=False, load_train=False)
        self.indexer.setup_regression()
        self.indexer.setup_assignment()

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
            assert False

        self.n_groups = len(self.indexer.data_params['split_groups'])
        if self.opt_params['assignment_batch_size'] == 'max':
            n_candidates = self.n_groups * (self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf'])
            self.opt_params['assignment_batch_size'] = n_candidates + self.opt_params['n_candidates_template']
            self.one_assignment_batch = True
        else:
            self.one_assignment_batch = False

    def predictions(self):
        self.N = self.indexer.data.shape[0]
        if self.opt_params['load_predictions']:
            self.uc_scaled_mean = np.load(
                f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_mean.npy'
                )
            self.uc_scaled_cov = np.load(
                f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_cov.npy'
                )
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
            np.save(
                f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_mean.npy',
                self.uc_scaled_mean
                )
            np.save(
                f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_cov.npy',
                self.uc_scaled_cov
                )

    def run(self):
        uc_true = np.stack(
            self.indexer.data['reindexed_unit_cell']
            )[:, self.indexer.data_params['y_indices']]

        # self.uc_scaled_mean: n_entries, n_groups, unit_cell_length
        uc_mean = np.zeros(self.uc_scaled_mean.shape)
        for group_index in range(self.n_groups):
            uc_mean[:, group_index, :] = self.indexer.revert_predictions(
                uc_pred_scaled=self.uc_scaled_mean[:, group_index, :]
                )
        closest_prediction = np.linalg.norm(uc_true[:, np.newaxis, :] - uc_mean, axis=2).argmin(axis=1)
        uc_pred = uc_mean[np.arange(len(closest_prediction)), closest_prediction]

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
            ])
        if self.opt_params['rerun_failures']:
            print()
            print('Loading failures')
            diagnostics = pd.read_json(os.path.join(self.save_to, 'optimization_diagnostics.json'))
            entry_list = np.array(diagnostics[~diagnostics['found']]['entry_index'])
        else:
            entry_list = np.arange(self.N)
        for entry_index in entry_list:
            start = time.time()
            candidates = self.generate_candidates(
                self.uc_scaled_mean[entry_index],
                self.uc_scaled_cov[entry_index],
                uc_pred[entry_index],
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
                diagnostic_df.to_json(os.path.join(self.save_to, 'optimization_diagnostics.json'))
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
            too_small_lengths = candidates_scaled < self.opt_params['minimum_uc_scaled']
            too_large_lengths = candidates_scaled > self.opt_params['maximum_uc_scaled']
            if np.sum(too_small_lengths) > 0:
                indices = np.argwhere(too_small_lengths)
                candidates_scaled[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=self.opt_params['minimum_uc_scaled'],
                    high=self.opt_params['minimum_uc_scaled'] + 0.1*np.abs(self.opt_params['minimum_uc_scaled']),
                    size=np.sum(too_small_lengths)
                    )
            if np.sum(too_large_lengths) > 0:
                indices = np.argwhere(too_large_lengths)
                candidates_scaled[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=self.opt_params['maximum_uc_scaled'] - 0.1*np.abs(self.opt_params['maximum_uc_scaled']),
                    high=self.opt_params['maximum_uc_scaled'],
                    size=np.sum(too_large_lengths)
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

                too_small_angles = candidates_scaled[:, 1] < self.opt_params['minimum_angle_scaled']
                too_large_angles = candidates_scaled[:, 1] > self.opt_params['maximum_angle_scaled']
                # If the candidate angle is small than minimum_angle_scaled (0.5 degrees)
                candidates_scaled[too_small_angles, 1] = self.rng.uniform(
                    low=self.opt_params['minimum_angle_scaled'],
                    high=self.opt_params['minimum_angle_scaled'] + 0.1*np.abs(self.opt_params['minimum_angle_scaled']),
                    size=np.sum(too_small_angles)
                    )
                # If the candidate angle is larger than maximum_angle_scaled (120 degrees)
                # Then uniformly sample between 100 and 120 degrees
                candidates_scaled[too_large_angles, 1] = self.rng.uniform(
                    low=(5/9*np.pi - np.pi/2) / self.indexer.angle_scale,
                    high=self.opt_params['maximum_angle_scaled'],
                    size=np.sum(too_large_angles)
                    )

                too_small_lengths = candidates_scaled[:, 0] < self.opt_params['minimum_uc_scaled']
                too_large_lengths = candidates_scaled[:, 0] > self.opt_params['maximum_uc_scaled']
                candidates_scaled[too_small_lengths, 0] = self.rng.uniform(
                    low=self.opt_params['minimum_uc_scaled'],
                    high=self.opt_params['minimum_uc_scaled'] + 0.1*np.abs(self.opt_params['minimum_uc_scaled']),
                    size=np.sum(too_small_lengths)
                    )
                candidates_scaled[too_large_lengths, 0] = self.rng.uniform(
                    low=self.opt_params['maximum_uc_scaled'] - 0.1*np.abs(self.opt_params['maximum_uc_scaled']),
                    high=self.opt_params['maximum_uc_scaled'],
                    size=np.sum(too_large_lengths)
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

                too_small_angles = candidates_scaled[:, 3] < self.opt_params['minimum_angle_scaled']
                too_large_angles = candidates_scaled[:, 3] > self.opt_params['maximum_angle_scaled']
                # If the candidate angle is small than minimum_angle_scaled (0.5 degrees)
                candidates_scaled[too_small_angles, 3] = self.rng.uniform(
                    low=self.opt_params['minimum_angle_scaled'],
                    high=self.opt_params['minimum_angle_scaled'] + 0.1*np.abs(self.opt_params['minimum_angle_scaled']),
                    size=np.sum(too_small_angles)
                    )
                # If the candidate angle is larger than maximum_angle_scaled (180 degrees)
                # Then uniformly sample between 100 and 180 degrees
                candidates_scaled[too_large_angles, 3] = self.rng.uniform(
                    low=self.opt_params['maximum_angle_scaled'] - 0.1*np.abs(self.opt_params['maximum_angle_scaled']),
                    high=self.opt_params['maximum_angle_scaled'],
                    size=np.sum(too_large_angles)
                    )

                too_small_lengths = candidates_scaled[:, :3] < self.opt_params['minimum_uc_scaled']
                too_large_lengths = candidates_scaled[:, :3] > self.opt_params['maximum_uc_scaled']
                indices = np.argwhere(too_small_lengths)
                candidates_scaled[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=self.opt_params['minimum_uc_scaled'],
                    high=self.opt_params['minimum_uc_scaled'] + 0.1*np.abs(self.opt_params['minimum_uc_scaled']),
                    size=np.sum(too_small_lengths)
                    )
                indices = np.argwhere(too_large_lengths)
                candidates_scaled[indices[:, 0], indices[:, 1]] = self.rng.uniform(
                    low=self.opt_params['maximum_uc_scaled'] - 0.1*np.abs(self.opt_params['maximum_uc_scaled']),
                    high=self.opt_params['maximum_uc_scaled'],
                    size=np.sum(too_large_lengths)
                    )
        elif self.indexer.data_params['lattice_system'] == 'triclinic':
            assert False

        return candidates_scaled

    def generate_unit_cells_miller_index_templates(self, q2_obs):
        xnn = np.zeros((self.opt_params['n_candidates_template'], self.indexer.data_params['n_outputs']))
        order = np.zeros((self.opt_params['n_candidates_template'], self.indexer.data_params['n_points']))
        check_array = np.arange(self.indexer.data_params['n_points'])
        for template_index in range(self.opt_params['n_candidates_template']):
            hkl = self.indexer.hkl_ref[self.indexer.miller_index_templates[template_index]]
            if self.indexer.data_params['lattice_system'] == 'monoclinic':
                hkl2 = np.concatenate((
                    hkl**2,
                    (hkl[:, 0] * hkl[:, 2])[:, np.newaxis]
                    ),
                    axis=1
                    )
            else:
                assert False
            xnn[template_index], r, rank, s = np.linalg.lstsq(hkl2, q2_obs, rcond=None)
            i = 0
            status = True
            while status:
                xnn_loop, r, rank, s = np.linalg.lstsq(hkl2, q2_obs, rcond=None)
                q2_calc = hkl2 @ xnn_loop
                sort_indices = np.argsort(q2_calc)
                i += 1
                if np.all(sort_indices == check_array):
                    status = False
                elif i == 100:
                    status = False
                else:
                    hkl2 = hkl2[sort_indices]
            xnn[template_index] = xnn_loop

        if self.indexer.data_params['lattice_system'] == 'monoclinic':
            too_small = xnn[:, :3] < (1 / self.opt_params['maximum_uc'])**2
            too_large = xnn[:, :3] > (1 / self.opt_params['minimum_uc'])**2
            xnn[:, :3][too_small] = (1 / self.opt_params['maximum_uc'])**2
            xnn[:, :3][too_large] = (1 / self.opt_params['minimum_uc'])**2

            cos_rbeta = xnn[:, 3] / (2 * np.sqrt(xnn[:, 0] * xnn[:, 2]))

            too_small = cos_rbeta < -1
            too_large = cos_rbeta > 1
            xnn[too_small, 3] = -2*0.999 * np.sqrt(xnn[too_small, 0] * xnn[too_small, 2])
            xnn[too_large, 3] = 2*0.999 * np.sqrt(xnn[too_large, 0] * xnn[too_large, 2])

            xnn_full = np.zeros((self.opt_params['n_candidates_template'], 6))
            xnn_full[:, :3] = xnn[:, :3]
            xnn_full[:, 4] = xnn[:, 3]
            reciprocal_unit_cell_full = get_reciprocal_unit_cell_from_xnn(xnn_full)
            unit_cell_full = reciprocal_uc_conversion(reciprocal_unit_cell_full)

            acute = unit_cell_full[:, 4] < np.pi/2
            unit_cell_full[acute, 4] = np.pi - unit_cell_full[acute, 4]
            candidates_scaled = self.indexer.scale_predictions(uc_pred=unit_cell_full[:, [0, 1, 2, 4]])
        else:
            assert False
        return candidates_scaled

    def plot_candidate_unit_cells(self, candidate_uc, entry, entry_index):
        if self.indexer.data_params['lattice_system'] == 'monoclinic':
            candidates_nn = np.zeros((self.n_groups * self.opt_params['n_candidates_nn'], 4))
            candidates_rf = np.zeros((self.n_groups * self.opt_params['n_candidates_rf'], 4))

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
                candidates_rf[start_nn: stop_nn] = candidate_uc[
                    start_candidates + self.opt_params['n_candidates_nn']: stop_candidates
                    ]
            candidates_template = candidate_uc[self.n_groups * n_candidates:]

            unit_cell_true = np.array(entry['reindexed_unit_cell'])
            ms = 1
            alpha = 0.25
            angle_bins = np.linspace(np.pi/2, np.pi, 101)
            angle_centers = (angle_bins[1:] + angle_bins[:-1]) / 2
            w = angle_bins[1] - angle_bins[0]
            fig, axes = plt.subplots(3, 4, figsize=(8, 6))

            distance_nn = np.linalg.norm(candidates_nn[:, :3] - unit_cell_true[:3][np.newaxis], axis=1)
            counts = [np.sum(distance_nn < i) for i in [1, 2, 3, 5]]
            axes[0, 1].set_title(f'Candidates within 1, 2, 3, 5A, (total): {counts[0]}, {counts[1]}, {counts[2]}, {counts[3]} ({candidates_nn.shape[0]})')
            axes[0, 0].plot(candidates_nn[:, 0], candidates_nn[:, 1], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[0, 1].plot(candidates_nn[:, 0], candidates_nn[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[0, 2].plot(candidates_nn[:, 1], candidates_nn[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            hist_nn, _ = np.histogram(candidates_nn[:, 3], bins=angle_bins, density=True)
            axes[0, 3].bar(angle_centers, hist_nn, width=w)

            distance_rf = np.linalg.norm(candidates_rf[:, :3] - unit_cell_true[:3][np.newaxis], axis=1)
            counts = [np.sum(distance_rf < i) for i in [1, 2, 3, 5]]
            axes[1, 1].set_title(f'Candidates within 1, 2, 3, 5A, (total): {counts[0]}, {counts[1]}, {counts[2]}, {counts[3]} ({candidates_rf.shape[0]})')
            axes[1, 0].plot(candidates_rf[:, 0], candidates_rf[:, 1], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[1, 1].plot(candidates_rf[:, 0], candidates_rf[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[1, 2].plot(candidates_rf[:, 1], candidates_rf[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            hist_rf, _ = np.histogram(candidates_rf[:, 3], bins=angle_bins, density=True)
            axes[1, 3].bar(angle_centers, hist_rf, width=w)

            distance_template = np.linalg.norm(candidates_template[:, :3] - unit_cell_true[:3][np.newaxis], axis=1)
            counts = [np.sum(distance_template < i) for i in [1, 2, 3, 5]]
            axes[2, 1].set_title(f'Candidates within 1, 2, 3, 5A, (total): {counts[0]}, {counts[1]}, {counts[2]}, {counts[3]} ({candidates_template.shape[0]})')
            axes[2, 0].plot(candidates_template[:, 0], candidates_template[:, 1], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[2, 1].plot(candidates_template[:, 0], candidates_template[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            axes[2, 2].plot(candidates_template[:, 1], candidates_template[:, 2], linestyle='none', marker='.', markersize=ms, alpha=alpha)
            hist_template, _ = np.histogram(candidates_template[:, 3], bins=angle_bins, density=True)
            axes[2, 3].bar(angle_centers, hist_template, width=w)

            for row in range(3):
                axes[row, 0].plot(unit_cell_true[0], unit_cell_true[1], linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0])
                axes[row, 1].plot(unit_cell_true[0], unit_cell_true[2], linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0])
                axes[row, 2].plot(unit_cell_true[1], unit_cell_true[2], linestyle='none', marker='s', markersize=2*ms, color=[0.8, 0, 0])

                ylim = axes[row, 3].get_ylim()
                axes[row, 3].plot([unit_cell_true[4], unit_cell_true[4]], 0.5*np.array(ylim), color=[0.8, 0, 0], linewidth=1)
                axes[row, 3].set_ylim(ylim)

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
                axes[row, 3].set_xlabel('beta')
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_to, f'initial_candidates_{entry_index:03d}.png'))
            # This is a bit scorched earth, but it helps with memory leaks.
            plt.cla()
            plt.clf()
            plt.close('all')

    def generate_candidates(self, uc_scaled_mean, uc_scaled_cov, uc_pred, entry, entry_index):
        n_candidates = self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf']
        candidate_uc_scaled = np.zeros((
            self.n_groups * n_candidates + self.opt_params['n_candidates_template'],
            self.indexer.data_params['n_outputs']
            ))
        for group_index, group in enumerate(self.indexer.data_params['split_groups']):
            start = group_index * n_candidates
            stop = (group_index + 1) * n_candidates
            # Get candidates from the neural network model
            candidates_scaled = self.generate_unit_cells(
                uc_scaled_mean[group_index, :],
                uc_scaled_cov[group_index, :, :]
                )
            candidate_uc_scaled[start: start + self.opt_params['n_candidates_nn'], :] = candidates_scaled

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
            candidates_uc_scaled_tree = candidates_scaled_tree[0, :, tree_indices]
            candidate_uc_scaled[start + self.opt_params['n_candidates_nn']: stop, :] = \
                candidates_uc_scaled_tree

        candidate_uc_scaled[self.n_groups * n_candidates:] = \
            self.generate_unit_cells_miller_index_templates(np.array(entry['q2']))

        candidate_uc = self.indexer.revert_predictions(uc_pred_scaled=candidate_uc_scaled)
        self.plot_candidate_unit_cells(candidate_uc, entry, entry_index)
        candidates = Candidates(
            entry=entry,
            unit_cell=candidate_uc,
            unit_cell_scaled=candidate_uc_scaled,
            unit_cell_pred=uc_pred,
            lattice_system=self.indexer.data_params['lattice_system'],
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            tolerance=self.opt_params['found_tolerance'],
            )

        if self.opt_params['iteration_info'][0]['assigner_key'] == 'closest':
            candidates = self.assign_hkls_closest(candidates)
        else:
            candidates = self.assign_hkls(
                candidates,
                'initial',
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

    def optimize_entry(self, candidates, entry_index):
        diagnostic_df_entry = candidates.diagnostics(self.indexer.data_params['hkl_ref_length'])
        acceptance_fraction = []
        for iteration_info in self.opt_params['iteration_info']:
            if iteration_info['worker'] == 'monoclinic_reset':
                candidates.save_candidates(self.save_to, entry_index)
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
                    if iteration_info['worker'] in ['softmax_subsampling', 'random_subsampling']:
                        next_xnn = self.random_subsampling(candidates, iteration_info)
                    elif iteration_info['worker'] in ['resampling', 'resampling_softmax_subsampling']:
                        next_candidates = copy.copy(candidates)
                        next_candidates = self.assign_hkls(
                            next_candidates,
                            iteration_info['assigner_key'],
                            'random'
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
        unit_cell_scaled = np.stack(candidates.candidates['unit_cell_scaled'])
        batch_q2_scaled = np.repeat(
            candidates.q2_obs_scaled[np.newaxis, :],
            repeats=self.opt_params['assignment_batch_size'],
            axis=0
            )
        if self.one_assignment_batch:
            batch_unit_cell_scaled = np.zeros((
                self.opt_params['assignment_batch_size'], self.indexer.data_params['n_outputs']
                ))
            batch_unit_cell_scaled[:candidates.n] = unit_cell_scaled
            inputs = {
                'unit_cell_scaled': batch_unit_cell_scaled,
                'q2_scaled': batch_q2_scaled
                }
            # This is a bottleneck
            softmaxes = self.indexer.assigner[assigner_key].model.predict_on_batch(inputs)[:candidates.n]
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
                    batch_unit_cell_scaled = np.zeros((
                        self.opt_params['assignment_batch_size'], self.indexer.data_params['n_outputs']
                        ))
                    batch_unit_cell_scaled[:left_over] = unit_cell_scaled[start: start + left_over]
                    batch_unit_cell_scaled[left_over:] = batch_unit_cell_scaled[0]
                else:
                    batch_unit_cell_scaled = unit_cell_scaled[start: end]
                inputs = {
                    'unit_cell_scaled': batch_unit_cell_scaled,
                    'q2_scaled': batch_q2_scaled
                    }
                # This is a bottleneck
                softmaxes_batch = self.indexer.assigner[assigner_key].model.predict_on_batch(inputs)
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
            hkl_pred[entry_index] = self.indexer.hkl_ref[hkl_assign[entry_index]]
        candidates.candidates['hkl'] = list(hkl_pred)
        return candidates

    def assign_hkls_closest(self, candidates):
        unit_cell_scaled = np.stack(candidates.candidates['unit_cell_scaled'])
        q2_obs_scaled = np.repeat(
            candidates.q2_obs_scaled[np.newaxis, :], repeats=candidates.n, axis=0
            )
        pairwise_differences_scaled = \
            self.indexer.assigner['0'].pairwise_difference_calculation.get_pairwise_differences_from_uc_scaled(
                unit_cell_scaled, q2_obs_scaled
                )
        pds_inv = self.indexer.assigner['0'].transform_pairwise_differences(
            pairwise_differences_scaled, tensorflow=False
            )
        softmax_all = scipy.special.softmax(pds_inv, axis=2)
        candidates.candidates['softmax'] = list(softmax_all.max(axis=2))
        candidates.candidates['hkl'] = list(self.indexer.convert_softmax_to_assignments(softmax_all))
        return candidates

    def montecarlo_acceptance(self, candidates, next_candidates, acceptance_method, acceptance_fraction):
        if acceptance_method == 'montecarlo':
            ratio = np.exp(-(next_candidates.candidates['loss'] - candidates.candidates['loss']) /self.temperature)
            probability = self.rng.random(candidates.n)
            accepted = probability < ratio
            acceptance_fraction.append(accepted.sum() / accepted.size)
            candidates.candidates.loc[accepted] = next_candidates.candidates.loc[accepted]
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
        next_candidates.candidates['unit_cell_scaled'] = list(self.indexer.scale_predictions(
            uc_pred=np.stack(next_candidates.candidates['unit_cell'])
            ))
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
        candidates.candidates['unit_cell_scaled'] = list(self.indexer.scale_predictions(
            uc_pred=np.stack(candidates.candidates['unit_cell'])
            ))
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
