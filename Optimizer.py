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
from Utilities import get_mpi_logger


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
        tolerance, unit_cell_key
        ):
        self.q2_obs = np.array(entry['q2'])
        self.q2_obs_scaled = np.array(entry['q2_scaled'])

        self.unit_cell_pred = unit_cell_pred
        unit_cell_true = np.array(entry[f'{unit_cell_key}unit_cell'])
        if lattice_system == 'cubic':
            self.unit_cell_true = unit_cell_true[0]
        elif lattice_system == 'tetragonal':
            self.unit_cell_true = unit_cell_true[:2]
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

        self.lattice_system = lattice_system
        self.minimum_unit_cell = minimum_unit_cell
        self.maximum_unit_cell = maximum_unit_cell
        self.tolerance = tolerance
        df_columns = [
            'unit_cell',
            'unit_cell_scaled',
            'hkl',
            'softmax',
            'loss',
            ]
        self.candidates = pd.DataFrame(columns=df_columns)
        self.explainers = pd.DataFrame(columns=df_columns)

        #best_index = np.argmin(np.linalg.norm(unit_cell - unit_cell_true, axis=1))
        #self.candidates['unit_cell'] = np.row_stack((unit_cell[best_index], unit_cell[best_index])).tolist()
        #self.candidates['unit_cell_scaled'] = np.row_stack((unit_cell_scaled[best_index], unit_cell_scaled[best_index])).tolist()
        self.candidates['unit_cell'] = list(unit_cell)
        self.candidates['unit_cell_scaled'] = list(unit_cell_scaled)
        self.n = len(self.candidates)
        self.n_uc = unit_cell.shape[1]
        self.n_points = self.q2_obs.size

        self.hkl_true_check = self.get_hkl_checks(self.hkl_true)

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
        if len(self.candidates) > 1:
            self.drop_identical_assignments()
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
            'assignment_batch_size': 64,
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
                self.n_groups = len(self.indexer.data_params['groups'])
        else:
            self.rank_indices = self.comm.recv(source=0, tag=0)
            self.indexer.data = self.comm.recv(source=0, tag=1)
            if self.opt_params['load_predictions']:
                self.uc_scaled_mean = self.comm.recv(source=0, tag=2)
                self.uc_scaled_cov = self.comm.recv(source=0, tag=3)
                self.n_groups = len(self.indexer.data_params['groups'])
                self.N = self.indexer.data.shape[0]

        if not self.opt_params['load_predictions']:
            self.n_groups = len(self.indexer.data_params['groups'])
            self.N = self.indexer.data.shape[0]
            self.uc_scaled_mean = np.zeros((self.N, self.n_groups, self.indexer.data_params['n_outputs']))
            self.uc_scaled_cov = np.zeros((
                self.N,
                self.n_groups,
                self.indexer.data_params['n_outputs'],
                self.indexer.data_params['n_outputs']
                ))
            for group_index, group in enumerate(self.indexer.data_params['groups']):
                print(f'Performing predictions with {group}')
                mean, cov = self.indexer.unit_cell_generator[group].do_predictions(
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
                        for entry_index in range(len(self.indexer.data)):
                            initial = mean[entry_index]
                            mean[entry_index], cov[entry_index] = unpermute_monoclinic_partial_unit_cell(
                                mean[entry_index], cov[entry_index], permutation, radians=True
                                )
                            #if monoclinic_angle == 'alpha':
                            #    print(f'{monoclinic_angle} {permutation} {initial} {mean[entry_index]}')
                self.uc_scaled_mean[:, group_index, :] = mean
                self.uc_scaled_cov[:, group_index, :, :] = cov

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
        uc_true = np.stack(
            self.indexer.data[f'{self.unit_cell_key}unit_cell']
            )[:, self.indexer.data_params['y_indices']]
        uc_mean = self.indexer.revert_predictions(uc_pred_scaled=self.uc_scaled_mean)
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
                self.indexer.data.iloc[entry_index]
                )
            candidates = self.optimize_entry(candidates)
            uc_best_opt[entry_index], report_counts = candidates.get_best_candidates(report_counts)
            print(report_counts)
            end = time.time()
            print(end - start)
        self.indexer.data[f'{self.indexer.hkl_prefactor}unit_cell_best_opt'] = list(uc_best_opt)

    def generate_candidates(self, uc_scaled_mean, uc_scaled_cov, uc_pred, entry):
        n_candidates = self.opt_params['n_candidates_nn'] + self.opt_params['n_candidates_rf']
        candidate_uc_scaled = np.zeros((self.n_groups * n_candidates, self.indexer.data_params['n_outputs']))
        for group_index, group in enumerate(self.indexer.data_params['groups']):
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
            _, _, candidates_scaled_tree = \
                self.indexer.unit_cell_generator[group].do_predictions_trees(q2_scaled=q2_scaled)
            tree_indices = self.rng.choice(
                candidates_scaled_tree.shape[2],
                size=self.opt_params['n_candidates_rf'],
                replace=False
                )
            if self.indexer.data_params['lattice_system'] == 'monoclinic':
                monoclinic_angle = self.reg_params[group]['monoclinic_angle']
                if monoclinic_angle == 'alpha':
                    permutation = 'bca'
                elif monoclinic_angle == 'beta':
                    permutation = 'abc'
                elif monoclinic_angle == 'gamma':
                    permutation = 'acb'
                if monoclinic_angle in ['alpha', 'gamma']:
                    for tree_index in tree_indices:
                        # candidates_scaled_tree: n_entries, n_outputs, n_trees
                        candidates_scaled_tree[0, :, tree_index] = unpermute_monoclinic_partial_unit_cell(
                            candidates_scaled_tree[0, :, tree_index], None, permutation, radians=True
                            )
            candidate_uc_scaled[start + self.opt_params['n_candidates_nn']: stop, :] = \
                candidates_scaled_tree[0, :, tree_indices]

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
            )
        return candidates

    def optimize_entry(self, candidates):
        """
        MCMC order:
            1: Assign Miller indices according to protocol
            2: Optimize given the assigned Miller indices
            3: Accept or reject new params.
        """
        candidates = self.assign_hkls(
            candidates,
            self.opt_params['iteration_info'][0]['assigner_tag'],
            assignment_method='best',
            )
        candidates = self.get_loss(candidates)
        candidates.diagnostics()
        for iteration_info in self.opt_params['iteration_info']:
            for iter_index in range(iteration_info['n_iterations']):
                next_candidates = copy.deepcopy(candidates)
                if iteration_info['worker'] in ['softmax_subsampling', 'random_subsampling']:
                    next_candidates = self.assign_hkls(
                        next_candidates, iteration_info['assigner_tag'], 'best'
                        )
                    next_candidates = self.random_subsampling(next_candidates, iteration_info)
                elif iteration_info['worker'] == 'resampling':
                    next_candidates = self.assign_hkls(
                        next_candidates, iteration_info['assigner_tag'], 'random'
                        )
                    next_candidates = self.no_subsampling(next_candidates, iteration_info)
                candidates = self.montecarlo_acceptance(
                    candidates,
                    next_candidates,
                    iteration_info['acceptance_method']
                    )
                candidates.update()
                if candidates.n <= 1:
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
            softmaxes_batch = self.indexer.assigner[assigner_key].model.predict_on_batch(inputs)
            if batch_index == n_batchs:
                softmaxes[start: start + left_over] = softmaxes_batch[:left_over]
            else:
                softmaxes[start: end] = softmaxes_batch
        return softmaxes

    def assign_hkls(self, candidates, assigner_key, assignment_method):
        softmaxes = self.get_softmaxes(candidates, assigner_key)
        if assignment_method == 'best':
            hkl_assign = softmaxes.argmax(axis=2)
            candidates.candidates['softmax'] = list(softmaxes.max(axis=2))
        elif assignment_method == 'random':
            # softmax output from tensorflow doesn't sum close enough to one for rng.choice
            softmaxes /= softmaxes.sum(axis=2)[:, :, np.newaxis]
            hkl_assign = np.zeros((candidates.n, self.indexer.data_params['n_points']), dtype=int)
            softmax = np.zeros((candidates.n, self.indexer.data_params['n_points']))
            for candidate_index in range(candidates.n):
                for point_index in range(self.indexer.data_params['n_points']):
                    hkl_assign[candidate_index, point_index] = self.rng.choice(
                        self.indexer.data_params['hkl_ref_length'],
                        size=1,
                        p=softmaxes[candidate_index, point_index]
                        )
                    softmax[candidate_index, point_index] = softmaxes[
                        candidate_index,
                        point_index,
                        hkl_assign[candidate_index, point_index]
                        ]
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
        next_candidates = self.get_loss(next_candidates)
        if acceptance_method == 'montecarlo':
            ratio = np.exp(-(next_candidates.candidates['loss'] - candidates.candidates['loss']))
            probability = self.rng.random(candidates.n)
            accepted = probability < ratio
            candidates.candidates.loc[accepted] = next_candidates.candidates.loc[accepted]
        elif acceptance_method == 'always':
            candidates = new_candidates
        else:
            assert False, 'Unrecognized acceptance method'
        return candidates

    def get_loss(self, candidates):
        target_function = CandidateOptLoss(
            q2_obs=candidates.q2_obs,
            lattice_system=self.indexer.data_params['lattice_system']
            )
        loss = np.zeros(candidates.n)
        hkl = np.stack(candidates.candidates['hkl'])
        unit_cell = np.stack(candidates.candidates['unit_cell'])
        for candidate_index in range(candidates.n):
            unit_cell_inv2 = target_function.update(
                hkl[candidate_index],
                np.ones(self.indexer.data_params['n_points']),
                unit_cell[candidate_index]
                )
            loss[candidate_index] = target_function.get_loss(unit_cell_inv2)
        candidates.candidates['loss'] = loss
        return candidates

    def optimize_unit_cell(self, q2_obs, hkl, softmax, unit_cell, optimizer):
        target_function = CandidateOptLoss(
            q2_obs=q2_obs,
            lattice_system=self.indexer.data_params['lattice_system']
            )
        unit_cell_inv2 = target_function.update(hkl, softmax, unit_cell)
        if optimizer in ['L-BFGS-B', 'BFGS', 'TNC', 'SLSQP']:
            results = scipy.optimize.minimize(
                target_function.loss_likelihood,
                x0=unit_cell_inv2,
                method=optimizer,
                jac=True,
                )
        elif optimizer in ['trust-krylov', 'trust-exact', 'trust-ncg', 'Newton-CG', 'dogleg', 'trust-constr']:
            """
            Brief testing showed 'trust-exact' had good performance.
            However it has a tendency to hang:
                https://github.com/scipy/scipy/pull/19668
                https://github.com/scipy/scipy/issues/12513
            'dogleg' also has good performance
            """
            results = scipy.optimize.minimize(
                target_function.loss_likelihood,
                x0=unit_cell_inv2,
                method=optimizer,
                jac=True,
                hess=target_function.loss_likelihood_hessian,
                options={'maxiter': 10},
                )
        optimized_unit_cell = target_function.get_optimized_uc(results.x)
        return optimized_unit_cell

    def no_subsampling(self, candidates, iteration_info):
        hkl = np.stack(candidates.candidates['hkl'])
        softmax = np.stack(candidates.candidates['softmax'])
        unit_cell = np.stack(candidates.candidates['unit_cell'])
        for candidate_index in range(candidates.n):
            unit_cell[candidate_index] = self.optimize_unit_cell(
                q2_obs=candidates.q2_obs,
                hkl=hkl[candidate_index],
                softmax=softmax[candidate_index],
                unit_cell=unit_cell[candidate_index],
                optimizer=iteration_info['optimizer'],
                )
        candidates.candidates['unit_cell'] = list(unit_cell)
        return candidates

    def random_subsampling(self, candidates, iteration_info):
        hkl = np.stack(candidates.candidates['hkl'])
        softmax = np.stack(candidates.candidates['softmax'])
        unit_cell = np.stack(candidates.candidates['unit_cell'])
        for candidate_index in range(candidates.n):
            if iteration_info['worker'] == 'random_subsampling':
                p = None
            elif iteration_info['worker'] == 'softmax_subsampling':
                p = softmax[candidate_index] / softmax[candidate_index].sum()
            subsampled_indices = np.sort(self.rng.choice(
                self.indexer.data_params['n_points'],
                size=self.indexer.data_params['n_points'] - iteration_info['n_drop'],
                replace=False,
                p=p,
                ))
            unit_cell[candidate_index] = self.optimize_unit_cell(
                q2_obs=candidates.q2_obs[subsampled_indices],
                hkl=hkl[candidate_index][subsampled_indices],
                softmax=softmax[candidate_index][subsampled_indices],
                unit_cell=unit_cell[candidate_index],
                optimizer=iteration_info['optimizer'],
                )
        candidates.candidates['unit_cell'] = list(unit_cell)
        return candidates

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
