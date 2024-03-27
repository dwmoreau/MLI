import copy
import itertools
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
import scipy.optimize
import scipy.special
import time

from Indexing import Indexing
from Reindexing import unpermute_monoclinic_partial_unit_cell
from TargetFunctions import CandidateOptLoss
from TargetFunctions import CandidateOptLoss_xnn
from Utilities import get_mpi_logger
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

    cumsum = np.cumsum(softmaxes, axis=2)
    random_values = rng.random(size=(n_entries, n_peaks))
    q = cumsum >= random_values[:, :, np.newaxis]
    hkl_assign = np.argmax(q, axis=2)

    softmax = np.zeros((n_entries, n_peaks))
    for candidate_index in range(n_entries):
        for point_index in range(n_peaks):
            softmax[candidate_index, point_index] = softmaxes[
                candidate_index,
                point_index,
                hkl_assign[candidate_index, point_index]
                ]
    return hkl_assign, softmax


def get_out_of_range_candidates(unit_cells, lattice_system, minimum_length, maximum_length, minimum_angle, maximum_angle):
    if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
        bad_indices = np.logical_or(
            np.any(unit_cells < minimum_length, axis=1),
            np.any(unit_cells > maximum_length, axis=1)
            )
    elif lattice_system == 'monoclinic':
        bad_lengths = np.logical_or(
            np.any(unit_cells[:, :3] < minimum_length, axis=1),
            np.any(unit_cells[:, :3] > maximum_length, axis=1)
            )
        bad_angles = np.logical_or(
            unit_cells[:, 3] < minimum_angle,
            unit_cells[:, 3] > maximum_angle
            )
        bad_indices = np.logical_or(bad_lengths, bad_angles)
    elif lattice_system == 'triclinic':
        bad_lengths = np.logical_or(
            np.any(unit_cells[:, :3] < minimum_length, axis=1),
            np.any(unit_cells[:, :3] > maximum_length, axis=1)
            )
        bad_angles = np.logical_or(
            np.any(unit_cells[:, 3:] < minimum_angle, axis=1),
            np.any(unit_cells[:, 3:] > maximum_angle, axis=1)
            )
        bad_indices = np.logical_or(bad_lengths, bad_angles)
    return bad_indices


class Candidates:
    def __init__(self,
        entry,
        unit_cell, unit_cell_scaled, unit_cell_pred,
        lattice_system,
        minimum_unit_cell, maximum_unit_cell,
        tolerance, unit_cell_key, n_iterations
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
        self.tolerance = tolerance

        unit_cell_true = np.array(entry[f'{unit_cell_key}unit_cell'])
        if lattice_system == 'cubic':
            self.unit_cell_true = unit_cell_true[0]
        elif lattice_system == 'tetragonal':
            self.unit_cell_true = unit_cell_true[[0, 2]]
        elif lattice_system == 'orthorhombic':
            self.unit_cell_true = unit_cell_true[:3]
        elif lattice_system == 'monoclinic':
            self.unit_cell_true = unit_cell_true[[0, 1, 2, 4]]
        elif lattice_system == 'triclinic':
            self.unit_cell_true = unit_cell_true
        self.hkl_true = np.array(entry[f'{unit_cell_key}hkl'])[:, :, 0]
        self.bl_true = entry['bravais_lattice']
        self.sg_true = int(entry['spacegroup_number'])
        self.spacegroup_symbol_hm_true = entry[f'{unit_cell_key}spacegroup_symbol_hm']

        df_columns = [
            'unit_cell',
            'unit_cell_scaled',
            'reciprocal_unit_cell',
            'xnn',
            'hkl',
            'softmax',
            'loss',
            ]
        self.candidates = pd.DataFrame(columns=df_columns)
        self.explainers = pd.DataFrame(columns=df_columns)

        self.candidates['unit_cell'] = list(unit_cell)
        self.candidates['unit_cell_scaled'] = list(unit_cell_scaled)
        self.update_xnn_from_unit_cell()

        self.hkl_true_check = self.get_hkl_checks(self.hkl_true)

    def update_xnn_from_unit_cell(self):
        unit_cell = np.stack(self.candidates['unit_cell'])
        unit_cell_full = np.zeros((self.n, 6))
        if self.lattice_system == 'cubic':
            unit_cell_full[:, :3] = unit_cell[:, 0]
            unit_cell_full[:, 3:] = np.pi/2
        elif self.lattice_system == 'tetragonal':
            unit_cell_full[:, :2] = unit_cell[:, 0][:, np.newaxis]
            unit_cell_full[:, 2] = unit_cell[:, 1]
            unit_cell_full[:, 3:] = np.pi/2
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
            reciprocal_unit_cell = reciprocal_unit_cell_full[:, 0]
            xnn = xnn_full[:, 0]
        elif self.lattice_system == 'tetragonal':
            reciprocal_unit_cell = reciprocal_unit_cell_full[:, [0, 2]]
            xnn = xnn_full[:, [0, 2]]
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
        elif self.lattice_system in ['monoclinic', 'triclinic']:
            too_small = xnn[:, :3] < (1 / self.maximum_unit_cell)**2
            too_large = xnn[:, :3] > (1 / self.minimum_unit_cell)**2
            xnn[:, :3][too_small] = (1 / self.maximum_unit_cell)**2
            xnn[:, :3][too_large] = (1 / self.minimum_unit_cell)**2
            if self.lattice_system == 'monoclinic':
                cos_rbeta = xnn[:, 3] / (xnn[:, 0] * xnn[:, 2])
                too_small = cos_rbeta < -1
                too_large = cos_rbeta > 1
                xnn[too_small, 3] = -0.999 * (xnn[too_small, 0] * xnn[too_small, 2])
                xnn[too_large, 3] = 0.999 * (xnn[too_large, 0] * xnn[too_large, 2])
            elif self.lattice_system == 'triclinic':
                assert False

        self.candidates['xnn'] = list(xnn)

        unit_cell_full = np.zeros((self.n, 6))
        xnn_full = np.zeros((self.n, 6))
        if self.lattice_system == 'cubic':
            xnn_full[:, :3] = xnn[:, 0]
        elif self.lattice_system == 'tetragonal':
            xnn_full[:, :2] = xnn[:, 0][:, np.newaxis]
            xnn_full[:, 2] = xnn[:, 1]
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
            unit_cell = unit_cell_full[:, 0]
        elif self.lattice_system == 'tetragonal':
            unit_cell = unit_cell_full[:, [0, 2]]
        elif self.lattice_system == 'orthorhombic':
            unit_cell = unit_cell_full[:, :3]
        elif self.lattice_system == 'monoclinic':
            unit_cell = unit_cell_full[:, [0, 1, 2, 4]]
        elif self.lattice_system == 'triclinic':
            unit_cell = unit_cell_full

        self.candidates['unit_cell'] = list(unit_cell)

    def get_hkl_checks(self, hkl):
        if self.lattice_system == 'cubic':
            hkl_check = np.sum(hkl**2, axis=-1)
        elif self.lattice_system == 'tetragonal':
            hkl_check = np.stack((
                np.sum(hkl[..., :2]**2, axis=-1),
                hkl[..., 2]**2,
                ),
                axis=-1
                )
        elif self.lattice_system == 'orthorhombic':
            hkl_check = hkl**2
        elif self.lattice_system == 'monoclinic':
            hkl_check = np.stack((
                hkl[..., 0]**2,
                hkl[..., 1]**2,
                hkl[..., 2]**2,
                hkl[..., 0] * hkl[..., 2],
                ),
                axis=-1
                )
        elif self.lattice_system == 'triclinic':
            hkl_check = np.stack((
                hkl[..., 0]**2,
                hkl[..., 1]**2,
                hkl[..., 2]**2,
                hkl[..., 0] * hkl[..., 1],
                hkl[..., 0] * hkl[..., 2],
                hkl[..., 1] * hkl[..., 2],
                ),
                axis=-1
                )
        return hkl_check

    def diagnostics(self):
        unit_cell = np.stack(self.candidates['unit_cell'])
        unit_cell_rms = 1/np.sqrt(self.n_uc) * np.linalg.norm(unit_cell - self.unit_cell_true, axis=1)
        unit_cell_max_diff = np.max(np.abs(unit_cell - self.unit_cell_true), axis=1)

        hkl = np.stack(self.candidates['hkl'])
        hkl_pred_check = self.get_hkl_checks(hkl)
        hkl_correct = self.hkl_true_check[np.newaxis] == hkl_pred_check
        hkl_accuracy = np.count_nonzero(hkl_correct, axis=1) / self.n_points

        print(f'True unit cell:              {np.round(self.unit_cell_true, decimals=4)}')
        print(f'Predicted unit cell:         {np.round(self.unit_cell_pred, decimals=4)}')
        print(f'Closest unit cell:           {np.round(unit_cell[np.argmin(unit_cell_rms)], decimals=4)}')
        print(f'Closest unit cell rms:       {unit_cell_rms.min():2.2f}')
        print(f'Smallest unit cell max diff: {unit_cell_max_diff.min():2.2f}')
        print(f'Mean unit cell rms:          {unit_cell_rms.mean():2.2f}')
        print(f'Best HKL accuracy:           {hkl_accuracy.max():1.2f}')
        print(f'Mean HKL accuracy:           {hkl_accuracy.mean():1.2f}')
        print(f'Bravais Lattice:             {self.bl_true}')
        print(f'Spacegroup:                  {self.sg_true} {self.spacegroup_symbol_hm_true}')

    def update(self):
        if self.n > 1:
            self.drop_bad_optimizations()
        if len(self.candidates) > 1:
            self.pick_explainers()
        #if len(self.candidates) > 1:
        #    self.drop_identical_assignments()
        self.candidates.sort_values(by='loss', inplace=True)

    def drop_identical_assignments(self):
        """
        np.unique returns the first unique index
        Because the candidate data frame is sorted by ascending loss, taking the first found
        instance when there are multiple redundant hkl assignments is equivalent to taking the
        instance with the smallest loss.
        """
        hkl = np.stack(self.candidates['hkl'])
        _, unique_indices = np.unique(hkl, axis=0, return_index=True)
        self.candidates = self.candidates.iloc[unique_indices]
        self.n = len(self.candidates)

    def drop_bad_optimizations(self):
        # Remove bad loss's
        loss = np.array(self.candidates['loss'])
        z = (loss - np.median(loss)) / loss.std()
        self.candidates = self.candidates.loc[z < 4]
        # Remove candidates with too small or too large unit cells
        in_range = np.invert(get_out_of_range_candidates(
                np.stack(self.candidates['unit_cell']),
                self.lattice_system,
                minimum_length=self.minimum_unit_cell,
                maximum_length=self.maximum_unit_cell,
                minimum_angle=0,
                maximum_angle=180,
                ))
        self.candidates = self.candidates.loc[in_range]
        # order candidates based on loss
        self.n = len(self.candidates)

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

    def catch_off_by_two(self, uc_best_opt):
        mult_factors = np.array([1/2, 1, 2])
        if self.lattice_system == 'cubic':
            if np.isclose(self.unit_cell_true, 1/2 * uc_best_opt, atol=1e-3):
                return True
            elif np.isclose(self.unit_cell_true, 2 * uc_best_opt, atol=1e-3):
                return True
            else:
                return False
        elif self.lattice_system == 'tetragonal':
            for mf0 in mult_factors:
                for mf1 in mult_factors:
                    mf = np.array([mf0, mf1])
                    if np.all(np.isclose(self.unit_cell_true, mf * uc_best_opt, atol=1e-3)):
                        return True
            return False
        elif self.lattice_system == 'orthorhombic':
            for mf0 in mult_factors:
                for mf1 in mult_factors:
                    for mf2 in mult_factors:
                        mf = np.array([mf0, mf1, mf2])
                        if np.all(np.isclose(self.unit_cell_true, np.sort(mf * uc_best_opt), atol=1e-3)):
                            return True
            return False
        else:
            return False

    def get_best_candidates(self, report_counts):
        if len(self.explainers) == 0:
            uc_best_opt = self.candidates.iloc[0]['unit_cell']
            print(np.stack(self.candidates['unit_cell'])[:5])
            #print(uc_best_opt)
            if np.all(np.isclose(self.unit_cell_true, uc_best_opt, atol=1e-3)):
                report_counts['Found and best'] += 1
            else:
                report_counts['Not found'] += 1
        elif len(self.explainers) == 1:
            print(self.explainers[['unit_cell', 'loss']])
            uc_best_opt = self.explainers.iloc[0]['unit_cell']
            if np.all(np.isclose(self.unit_cell_true, uc_best_opt, atol=1e-3)):
                report_counts['Found and best'] += 1
            elif self.catch_off_by_two(uc_best_opt):
                report_counts['Found but off by two'] += 1
            else:
                report_counts['Not found'] += 1
        else:
            print(self.explainers[['unit_cell', 'loss']])
            uc_best_opt = np.array(self.explainers.iloc[0]['unit_cell'])
            found_best = False
            found_not_best = False
            found_off_by_two = False
            for explainer_index in range(len(self.explainers)):
                uc = np.array(self.explainers.iloc[explainer_index]['unit_cell'])
                if np.all(np.isclose(self.unit_cell_true, uc, atol=1e-3)):
                    if explainer_index == 0:
                        found_best = True
                    else:
                        found_not_best = True
                elif self.catch_off_by_two(uc):
                    found_off_by_two = True
            if found_best:
                report_counts['Found and best'] += 1
            elif found_not_best:
                report_counts['Found but not best'] += 1
            elif found_off_by_two:
                report_counts['Found but off by two'] += 1
            else:
                report_counts['Found explainers'] += 1

        return uc_best_opt, report_counts


class Optimizer:
    def __init__(self, assign_params, data_params, opt_params, reg_params):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_ranks = self.comm.Get_size()
        self.assign_params = assign_params
        self.data_params = data_params
        self.opt_params = opt_params
        self.reg_params = reg_params
        self.rng = np.random.default_rng()

        opt_params_defaults = {
            'n_candidates_nn': 30,
            'n_candidates_rf': 30,
            'minimum_uc': 2,
            'maximum_uc': 500,
            'tuning_param': 100,
            'n_pred_evals': 500,
            'found_tolerance': -200,
            'assignment_batch_size': 'max',
            'load_predictions': False,
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

        self.save_to = os.path.join(self.data_params['base_directory'], 'models', self.data_params['tag'], 'optimizer')

        self.indexer = Indexing(
            assign_params=self.assign_params, 
            data_params=self.data_params,
            reg_params=self.reg_params, 
            seed=12345, 
            )
        self.indexer.setup_from_tag()
        if self.rank == 0:
            self.indexer.load_data_from_tag(load_augmented=False, load_train=False)
            if not os.path.exists(self.save_to):
                os.mkdir(self.save_to)
        else:
            self.indexer.hkl_ref = None
        self.indexer.hkl_ref = self.comm.bcast(self.indexer.hkl_ref, root=0)
        self.indexer.setup_regression()
        self.indexer.setup_assignment()
        self.opt_params['minimum_uc_scaled'] = \
            (self.opt_params['minimum_uc'] - self.indexer.uc_scaler.mean_[0]) / self.indexer.uc_scaler.scale_[0]
        self.opt_params['maximum_uc_scaled'] = \
            (self.opt_params['maximum_uc'] - self.indexer.uc_scaler.mean_[0]) / self.indexer.uc_scaler.scale_[0]
        # Minimum / Maximum angle = 0 / 180 degrees
        self.opt_params['minimum_angle_scaled'] = (0 - np.pi/2) / self.indexer.angle_scale
        self.opt_params['maximum_angle_scaled'] = (np.pi - np.pi/2) / self.indexer.angle_scale

        self.logger = get_mpi_logger(self.rank, self.save_to, self.opt_params['tag'])

        if self.indexer.data_params['lattice_system'] in ['monoclinic', 'cubic']:
            self.unit_cell_key = ''
        elif self.indexer.data_params['lattice_system'] in ['tetragonal', 'orthorhombic']:
            self.unit_cell_key = 'reindexed_'

        self.n_groups = len(self.indexer.data_params['split_groups'])
        if self.opt_params['assignment_batch_size'] == 'max':
            n_candidates = self.n_groups * (self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf'])
            self.opt_params['assignment_batch_size'] = n_candidates
            self.one_assignment_batch = True
        else:
            self.one_assignment_batch = False

    def distribute_data(self):
        # Make predictions
        if self.opt_params['load_predictions'] and self.rank == 0:
            uc_scaled_mean_all = np.load(
                f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_mean.npy'
                )
            uc_scaled_cov_all = np.load(
                f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_cov.npy'
                )

        if self.rank == 0:
            for rank_index in range(1, self.n_ranks):
                rank_indices = np.arange(rank_index, self.indexer.N, self.n_ranks)
                self.comm.send(rank_indices, dest=rank_index, tag=0)
                self.comm.send(self.indexer.data.iloc[rank_indices], dest=rank_index, tag=1)
                if self.opt_params['load_predictions']:
                    self.comm.send(uc_scaled_mean_all[rank_indices], dest=rank_index, tag=2)
                    self.comm.send(uc_scaled_cov_all[rank_indices], dest=rank_index, tag=3)
            self.rank_indices = np.arange(0, self.indexer.data.shape[0], self.n_ranks)
            self.indexer.data = self.indexer.data.iloc[self.rank_indices]
            if self.opt_params['load_predictions']:
                self.uc_scaled_mean = uc_scaled_mean_all[self.rank_indices]
                self.uc_scaled_cov = uc_scaled_cov_all[self.rank_indices]
                self.N = self.indexer.data.shape[0]
                self.n_groups = len(self.indexer.data_params['split_groups'])
        else:
            self.rank_indices = self.comm.recv(source=0, tag=0)
            self.indexer.data = self.comm.recv(source=0, tag=1)
            if self.opt_params['load_predictions']:
                self.uc_scaled_mean = self.comm.recv(source=0, tag=2)
                self.uc_scaled_cov = self.comm.recv(source=0, tag=3)
                self.n_groups = len(self.indexer.data_params['split_groups'])
                self.N = self.indexer.data.shape[0]

        if not self.opt_params['load_predictions']:
            self.N = self.indexer.data.shape[0]
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
                if self.indexer.data_params['lattice_system'] == 'monoclinic':
                    monoclinic_angle = self.reg_params[group]['monoclinic_angle']
                    # Monoclinic predictions are performed in a 'reindexed' setting where axes lengths are a < b < c,
                    # and the monoclinic angle is not necessarily beta. Optimization is performed in the standard
                    # setting where beta is the monoclinic angle.
                    # 'permutation' is the permutation of the axes needed to go from the standard monoclinic angle
                    # at beta to the given monoclinic angle. This is the expectation of the unpermute function.
                    if monoclinic_angle == 'alpha':
                        permutation = 'bca'
                    elif monoclinic_angle == 'beta':
                        permutation = 'abc'
                    elif monoclinic_angle == 'gamma':
                        permutation = 'acb'
                    if monoclinic_angle in ['alpha', 'gamma']:
                        uc_mean_group, uc_cov_group = self.indexer.revert_predictions(
                            uc_pred_scaled=uc_mean_scaled_group, uc_pred_scaled_cov=uc_cov_scaled_group
                            )
                        # The unpermute function needs unscaled unit cells
                        for entry_index in range(len(self.indexer.data)):
                            uc_mean_group[entry_index], uc_cov_group[entry_index] = unpermute_monoclinic_partial_unit_cell(
                                uc_mean_group[entry_index], uc_cov_group[entry_index], permutation, radians=True
                                )
                        uc_mean_scaled_group, uc_cov_scaled_group = self.indexer.scale_predictions(
                            uc_pred=uc_mean_group, uc_pred_cov=uc_cov_group
                            )
                self.uc_scaled_mean[:, group_index, :] = uc_mean_scaled_group
                self.uc_scaled_cov[:, group_index, :, :] = uc_cov_scaled_group

            if self.rank == 0:
                uc_scaled_mean_all = [None for _ in range(self.n_ranks)]
                uc_scaled_cov_all = [None for _ in range(self.n_ranks)]
                rank_indices_all = [None for _ in range(self.n_ranks)]
                uc_scaled_mean_all[0] = self.uc_scaled_mean
                uc_scaled_cov_all[0] = self.uc_scaled_cov
                rank_indices_all[0] = self.rank_indices
                for rank_index in range(1, self.n_ranks):
                    uc_scaled_mean_all[rank_index] = self.comm.recv(source=rank_index, tag=2)
                    uc_scaled_cov_all[rank_index] = self.comm.recv(source=rank_index, tag=3)
                    rank_indices_all[rank_index] = self.comm.recv(source=rank_index, tag=4)
                uc_scaled_mean_all = np.row_stack(uc_scaled_mean_all)
                uc_scaled_cov_all = np.row_stack(uc_scaled_cov_all)
                rank_indices_all = np.concatenate(rank_indices_all)
                sort_indices = np.argsort(rank_indices_all)
                np.save(
                    f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_mean.npy',
                    uc_scaled_mean_all[sort_indices]
                    )
                np.save(
                    f'{self.save_to}/{self.data_params["tag"]}_uc_scaled_cov.npy',
                    uc_scaled_cov_all[sort_indices]
                    )
            else:
                self.comm.send(self.uc_scaled_mean, dest=0, tag=2)
                self.comm.send(self.uc_scaled_cov, dest=0, tag=3)
                self.comm.send(self.rank_indices, dest=0, tag=4)

    def run(self):
        n_iterations = 0
        for iteration_info in self.opt_params['iteration_info']:
            n_iterations += iteration_info['n_iterations']

        uc_true = np.stack(
            self.indexer.data[f'{self.unit_cell_key}unit_cell']
            )[:, self.indexer.data_params['y_indices']]

        # self.uc_scaled_mean: n_entries, n_groups, unit_cell_length
        uc_mean = np.zeros(self.uc_scaled_mean.shape)
        for group_index in range(self.n_groups):
            uc_mean[:, group_index, :] = self.indexer.revert_predictions(
                uc_pred_scaled=self.uc_scaled_mean[:, group_index, :]
                )
        closest_prediction = np.linalg.norm(uc_true[:, np.newaxis, :] - uc_mean, axis=2).argmin(axis=1)
        uc_pred = uc_mean[np.arange(len(closest_prediction)), closest_prediction]

        uc_best_opt = np.zeros((self.N, self.indexer.data_params['n_outputs']))
        percentage = 0
        report_counts = {
            'Not found': 0,
            'Found and best': 0,
            'Found but not best': 0,
            'Found but off by two': 0,
            'Found explainers': 0,
            }
        for entry_index in range(self.N):
            start = time.time()
            current_percentage = entry_index / self.N
            if current_percentage > percentage + 0.01:
                self.logger.info(f' {100*current_percentage:3.0f}% complete')
                percentage += 0.01
            print()
            candidates = self.generate_candidates(
                self.uc_scaled_mean[entry_index],
                self.uc_scaled_cov[entry_index],
                uc_pred[entry_index],
                self.indexer.data.iloc[entry_index],
                n_iterations
                )
            candidates = self.optimize_entry(candidates)
            uc_best_opt[entry_index], report_counts = candidates.get_best_candidates(report_counts)
            print(report_counts)
            end = time.time()
            print(end - start)
        self.indexer.data[f'{self.unit_cell_key}unit_cell_best_opt'] = list(uc_best_opt)

    def generate_candidates(self, uc_scaled_mean, uc_scaled_cov, uc_pred, entry, n_iterations):
        n_candidates = self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf']
        candidate_uc_scaled = np.zeros((self.n_groups * n_candidates, self.indexer.data_params['n_outputs']))
        for group_index, group in enumerate(self.indexer.data_params['split_groups']):
            start = group_index * n_candidates
            stop = (group_index + 1) * n_candidates
            # Get candidates from the neural network model
            candidates_scaled = self.rng.multivariate_normal(
                mean=uc_scaled_mean[group_index, :],
                cov=uc_scaled_cov[group_index, :, :],
                size=self.opt_params['n_candidates_nn'],
                )
            bad_indices = get_out_of_range_candidates(
                candidates_scaled,
                self.indexer.data_params['lattice_system'],
                self.opt_params['minimum_uc_scaled'],
                self.opt_params['maximum_uc_scaled'],
                self.opt_params['minimum_angle_scaled'],
                self.opt_params['maximum_angle_scaled'],
                )
            n_bad_indices = np.sum(bad_indices)
            while n_bad_indices > 0:
                candidates_scaled[bad_indices] = self.rng.multivariate_normal(
                    mean=uc_scaled_mean[group_index, :],
                    cov=uc_scaled_cov[group_index, :, :],
                    size=n_bad_indices,
                    )
                bad_indices = get_out_of_range_candidates(
                    candidates_scaled,
                    self.indexer.data_params['lattice_system'],
                    self.opt_params['minimum_uc_scaled'],
                    self.opt_params['maximum_uc_scaled'],
                    self.opt_params['minimum_angle_scaled'],
                    self.opt_params['maximum_angle_scaled'],
                    )
                n_bad_indices = np.sum(bad_indices)
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
            if self.indexer.data_params['lattice_system'] == 'monoclinic':
                # This only needs to be performed for the tree predictions.
                # The NN predictions were unpermuted in self.distribute_data.
                monoclinic_angle = self.reg_params[group]['monoclinic_angle']
                if monoclinic_angle == 'alpha':
                    permutation = 'bca'
                elif monoclinic_angle == 'beta':
                    permutation = 'abc'
                elif monoclinic_angle == 'gamma':
                    permutation = 'acb'
                if monoclinic_angle in ['alpha', 'gamma']:
                    # The unpermute function needs unscaled unit cells
                    candidates_uc_tree = self.indexer.revert_predictions(
                        uc_pred_scaled=candidates_uc_scaled_tree
                        )
                    for candidate_index in range(candidates_uc_tree.shape[0]):
                        candidates_uc_tree[candidate_index] = unpermute_monoclinic_partial_unit_cell(
                            candidates_uc_tree[candidate_index], None, permutation, radians=True
                            )
                    candidates_uc_scaled_tree = self.indexer.scale_predictions(
                        uc_pred=candidates_uc_tree
                        )
            candidate_uc_scaled[start + self.opt_params['n_candidates_nn']: stop, :] = \
                candidates_uc_scaled_tree

        candidates = Candidates(
            entry=entry,
            unit_cell=self.indexer.revert_predictions(uc_pred_scaled=candidate_uc_scaled),
            unit_cell_scaled=candidate_uc_scaled,
            unit_cell_pred=uc_pred,
            lattice_system=self.indexer.data_params['lattice_system'],
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            tolerance=self.opt_params['found_tolerance'],
            unit_cell_key=self.unit_cell_key,
            n_iterations=n_iterations
            )

        if self.opt_params['iteration_info'][0]['assigner_tag'] == 'closest':
            candidates = self.assign_hkls_closest(candidates)
        else:
            candidates = self.assign_hkls(
                candidates,
                self.opt_params['iteration_info'][0]['assigner_tag'],
                'best'
                )

        target_function = CandidateOptLoss_xnn(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system']
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

    def optimize_entry(self, candidates):
        """
        MCMC order:
            1: Assign Miller indices according to protocol
            2: Optimize given the assigned Miller indices
            3: Accept or reject new params.
        """
        candidates.diagnostics()
        for iteration_info in self.opt_params['iteration_info']:
            for iter_index in range(iteration_info['n_iterations']):
                if iteration_info['worker'] in ['softmax_subsampling', 'random_subsampling']:
                    next_xnn = self.random_subsampling(candidates, iteration_info)
                elif iteration_info['worker'] == 'resampling':
                    next_candidates = copy.copy(candidates)
                    next_candidates = self.assign_hkls(next_candidates, iteration_info['assigner_tag'], 'random')
                    next_xnn = self.no_subsampling(next_candidates, iteration_info)
                elif iteration_info['worker'] == 'no_subsampling':
                    next_xnn = self.no_subsampling(candidates, iteration_info)
                candidates = self.update_candidates(
                    candidates,
                    iteration_info['assigner_tag'],
                    iteration_info['acceptance_method'],
                    next_xnn
                    )
                if candidates.n <= 1:
                    return candidates
                if len(candidates.explainers) > 10:
                    return candidates
                print(f'{candidates.n}, {len(candidates.explainers)} {candidates.candidates["loss"].mean()}, {candidates.candidates["loss"].min()}, {iteration_info["assigner_tag"]}')
        return candidates

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

    def montecarlo_acceptance(self, candidates, next_candidates, acceptance_method):
        if acceptance_method == 'montecarlo':
            ratio = np.exp(-(next_candidates.candidates['loss'] - candidates.candidates['loss']))
            probability = self.rng.random(candidates.n)
            accepted = probability < ratio
            candidates.candidates.loc[accepted] = next_candidates.candidates.loc[accepted]
        elif acceptance_method == 'always':
            candidates = next_candidates
        else:
            assert False, 'Unrecognized acceptance method'
        return candidates

    def update_candidates(self, candidates, assigner_key, acceptance_method, next_xnn):
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
            lattice_system=self.indexer.data_params['lattice_system']
            )
        target_function.update(
            np.stack(next_candidates.candidates['hkl']),
            np.ones((candidates.n, self.indexer.data_params['n_points'])),
            next_xnn
            )
        next_candidates.candidates['loss'] = target_function.get_loss(next_xnn)
        candidates = self.montecarlo_acceptance(candidates, next_candidates, acceptance_method)
        candidates.update()
        return candidates

    def no_subsampling(self, candidates, iteration_info):
        target_function = CandidateOptLoss_xnn(
            q2_obs=np.repeat(candidates.q2_obs[np.newaxis, :], repeats=candidates.n, axis=0), 
            lattice_system=self.indexer.data_params['lattice_system'],
            )
        xnn = np.stack(candidates.candidates['xnn'])
        #np.save('mono_uc_true.npy', candidates.unit_cell_true)
        #np.save('mono_q2_obs.npy', candidates.q2_obs)
        #np.save('mono_xnn.npy', xnn)
        #np.save('mono_hkl.npy', np.stack(candidates.candidates['hkl']))
        #np.save('mono_softmax.npy', np.stack(candidates.candidates['softmax']))
        #np.save('mono_unit_cell.npy', np.stack(candidates.candidates['unit_cell']))
        #assert False
        target_function.update(
            np.stack(candidates.candidates['hkl']), 
            np.stack(candidates.candidates['softmax']), 
            xnn)
        delta_gn = target_function.gauss_newton_step(xnn)
        next_xnn = xnn + delta_gn
        return next_xnn

    def random_subsampling(self, candidates, iteration_info):
        hkl = np.stack(candidates.candidates['hkl'])
        softmax = np.stack(candidates.candidates['softmax'])
        xnn = np.stack(candidates.candidates['xnn'])
        n_keep = self.indexer.data_params['n_points'] - iteration_info['n_drop']
        if iteration_info['worker'] == 'random_subsampling':
            subsampled_indices = np.zeros((candidates.n, n_keep), dtype=int)
            for entry_index in range(candidates.n):
                subsampled_indices[entry_index] = self.rng(
                    self.indexer.data_params['n_points'],
                    size=n_keep,
                    replace=True
                    )
        elif iteration_info['worker'] == 'softmax_subsampling':
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
            )
        target_function.update(hkl_subsampled, softmax_subsampled, xnn)
        delta_gn = target_function.gauss_newton_step(xnn)
        next_xnn = xnn + delta_gn
        return next_xnn

    def gather_optimized_unit_cells(self):
        if self.rank == 0:
            optimized_data = [None for _ in range(self.n_ranks)]
            optimized_data[0] = self.indexer.data
            for rank_index in range(1, self.n_ranks):
                optimized_data[rank_index] = self.comm.recv(source=rank_index, tag=2)
            self.optimized_data = pd.concat(optimized_data)
            self.optimized_data[f'{self.indexer.hkl_prefactor}h'] = list(np.stack(self.optimized_data[f'{self.indexer.hkl_prefactor}hkl'])[:, :, 0, 0])
            self.optimized_data[f'{self.indexer.hkl_prefactor}k'] = list(np.stack(self.optimized_data[f'{self.indexer.hkl_prefactor}_hkl'])[:, :, 1, 0])
            self.optimized_data[f'{self.indexer.hkl_prefactor}l'] = list(np.stack(self.optimized_data[f'{self.indexer.hkl_prefactor}hkl'])[:, :, 2, 0])
            drop_columns = [
                f'{self.indexer.hkl_prefactor}hkl',
                f'{self.indexer.hkl_prefactor}unit_cell_pred_cov',
                f'{self.indexer.hkl_prefactor}unit_cell_pred_scaled_cov'
                ]
            self.optimized_data.drop(columns=drop_columns, inplace=True)
            self.optimized_data.to_parquet(f'{self.save_to}/{self.opt_params["tag"]}_optimized_data.parquet')
        else:
            self.comm.send(self.indexer.data, dest=0, tag=2)

    def evaluate(self):
        self.gather_optimized_unit_cells()
        if self.rank == 0:
            alpha = 0.1
            markersize = 0.5
            titles = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
            for bravais_lattice in self.indexer.data_params['bravais_lattices'] + ['All']:
                if bravais_lattice != 'All':
                    bl_data = self.optimized_data[self.optimized_data['bravais_lattice'] == bravais_lattice]
                else:
                    bl_data = self.optimized_data
                uc_true_bl = np.stack(bl_data[f'{self.indexer.hkl_prefactor}unit_cell'])[:, self.indexer.data_params['y_indices']]
                uc_best_opt_bl = np.stack(bl_data[f'{self.indexer.hkl_prefactor}unit_cell_best_opt'])
                uc_best_cand_bl = np.stack(bl_data[f'{self.indexer.hkl_prefactor}unit_cell_best_cand'])
                figsize = (self.indexer.data_params['n_outputs']*2 + 2, 8)
                fig, axes = plt.subplots(2, self.indexer.data_params['n_outputs'], figsize=figsize)
                uc_pred = [uc_best_opt_bl, uc_best_cand_bl]
                for uc_index in range(self.indexer.data_params['n_outputs']):
                    all_info = np.sort(uc_true_bl[:, uc_index])
                    lower = all_info[int(0.005*all_info.size)]
                    upper = all_info[int(0.995*all_info.size)]
                    for row in range(len(uc_pred)):
                        axes[row, uc_index].plot(
                            uc_true_bl[:, uc_index], uc_pred[row][:, uc_index], 
                            color=[0, 0, 0], alpha=alpha, 
                            linestyle='none', marker='.', markersize=markersize,
                            )
                        axes[row, uc_index].plot(
                            [lower, upper], [lower, upper],
                            color=[0.7, 0, 0], linestyle='dotted'
                            )
                        axes[row, uc_index].set_xlim([lower, upper])
                        axes[row, uc_index].set_ylim([lower, upper])

                        error = np.sort(np.abs(uc_true_bl[:, uc_index] - uc_pred[row][:, uc_index]))
                        p25 = error[int(0.25 * error.size)]
                        p50 = error[int(0.50 * error.size)]
                        p75 = error[int(0.75 * error.size)]
                        rmse = np.sqrt(1/uc_true_bl.shape[0] * np.linalg.norm(error)**2)
                        error_titles = [
                            titles[self.indexer.data_params['y_indices'][uc_index]],
                            f'RMSE: {rmse:0.4f}',
                            f'25%: {p25:0.4f}',
                            f'50%: {p50:0.4f}',
                            f'75%: {p75:0.4f}',
                            ]
                        axes[row, uc_index].set_title('\n'.join(error_titles), fontsize=12)
                        axes[row, uc_index].set_xlabel('True')
                        axes[row, 0].set_ylabel('Optimized')
                fig.tight_layout()
                fig.savefig(f'{self.save_to}/{self.opt_params["tag"]}_{bravais_lattice}.png')
                plt.close()
