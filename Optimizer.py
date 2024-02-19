import gc
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
import scipy.optimize
import scipy.special
import tensorflow as tf

from Indexing import Indexing
from TargetFunctions import CandidateOptLoss


class Candidates:
    def __init__(self,
        q2_obs, q2_obs_scaled, 
        unit_cell, unit_cell_scaled,
        minimum_unit_cell, maximum_unit_cell,
        unit_cell_generator_label,
        tolerance, tolerance_key,
        ):
        self.minimum_unit_cell = minimum_unit_cell
        self.maximum_unit_cell = maximum_unit_cell
        self.q2_obs = q2_obs
        self.q2_obs_scaled = q2_obs_scaled
        self.tolerance = tolerance
        self.tolerance_key = tolerance_key
        df_columns = [
            'unit_cell',
            'unit_cell_scaled',
            'unit_cell_initial',
            'hkl',
            'hkl_initial',
            'softmax',
            'q2_pred',
            'L1',
            'L2',
            'wL1',
            'wL2',
            'loss',
            'unit_cell_generator'
            ]
        self.candidates = pd.DataFrame(columns=df_columns)
        self.explainers = pd.DataFrame(columns=df_columns)

        self.candidates['unit_cell'] = list(unit_cell)
        self.candidates['unit_cell_scaled'] = list(unit_cell_scaled)
        self.candidates['unit_cell_initial'] = list(unit_cell)
        self.candidates['unit_cell_generator'] = unit_cell_generator_label
        self.n = unit_cell.shape[0]

    def initial_diagnostics(self, unit_cell_true, unit_cell_pred, unit_cell_pred_std, hkl_true, bl_true, sg_true):
        unit_cell_initial = np.stack(self.candidates['unit_cell_initial'])
        unit_cell_initial_rms = 1/unit_cell_true.size * np.linalg.norm(unit_cell_initial - unit_cell_true, axis=1)
        unit_cell_initial_max_diff = np.max(np.abs(unit_cell_initial - unit_cell_true), axis=1)

        hkl_initial = np.stack(self.candidates['hkl_initial'])
        hkl_correct = np.zeros(hkl_initial.shape, dtype=bool)
        for candidate_index in range(self.n):
            hkl_correct[candidate_index, :, :] = hkl_initial[candidate_index, :, :] == hkl_true
        hkl_accuracy = np.count_nonzero(np.all(hkl_correct, axis=2), axis=1) / self.q2_obs.size
        print(f'True unit cell:              {np.round(unit_cell_true, decimals=4)}')
        print(f'Predicted unit cell:         {np.round(unit_cell_pred, decimals=4)}')
        print(f'Predicted unit cell std:     {np.round(unit_cell_pred_std, decimals=4)}')
        print(f'Closest unit cell rms:       {unit_cell_initial_rms.min():2.4f}')
        print(f'Smallest unit cell max diff: {unit_cell_initial_max_diff.min():2.4f}')
        print(f'Mean unit cell rms:          {unit_cell_initial_rms.mean():2.4f}')
        print(f'Best HKL accuracy: {hkl_accuracy.max()}')
        print(f'Mean HKL accuracy: {hkl_accuracy.mean()}')
        print(f'Bravais Lattice: {bl_true}')
        print(f'Spacegroup: {sg_true}')

    def update(self):
        self.drop_bad_optimizations()
        self.pick_explainers()
        self.drop_identical_assignments()

    def drop_identical_assignments(self):
        """
        Because the candidate data frame is sorted by ascending loss, taking the first found 
        instance when there are multiple redundant hkl assignments is equivalent to taking the
        instance with the smallest loss.
        """
        hkl = np.stack(self.candidates['hkl'])
        _, unique_indices = np.unique(hkl, axis=0, return_index=True)
        self.candidates = self.candidates.iloc[unique_indices]
        self.candidates.sort_values(by=self.tolerance_key, inplace=True)
        self.n = len(self.candidates)

    def drop_bad_optimizations(self):
        # Remove really bad loss's
        loss = self.candidates[self.tolerance_key]
        z = (loss - np.median(loss)) / loss.std()
        self.candidates = self.candidates.loc[z < 4]
        # Remove candidates with too small or too large unit cells
        in_range = np.logical_and(
            np.all(np.stack(self.candidates['unit_cell']) >= self.minimum_unit_cell, axis=1),
            np.all(np.stack(self.candidates['unit_cell']) <= self.maximum_unit_cell, axis=1)
            )
        self.candidates = self.candidates.loc[in_range]
        # order candidates based on loss
        self.candidates.sort_values(by=self.tolerance_key, inplace=True)
        self.n = len(self.candidates)

    def pick_explainers(self):
        found = self.candidates[self.tolerance_key] < self.tolerance
        if np.count_nonzero(found) > 0:
            if len(self.explainers) == 0:
                self.explainers = self.candidates.loc[found].copy()
            else:
                self.explainers = pd.concat([self.explainers, self.candidates.loc[found]], ignore_index=True)
            self.explainers.sort_values(by=self.tolerance_key, inplace=True)
            self.candidates = self.candidates.loc[~found]
            self.n = len(self.candidates)

    def get_best_candidates(self, uc_true, bl_true, hkl_true, report_counts):
        if len(self.explainers) == 0:
            uc_best_opt = self.candidates.iloc[0]['unit_cell']
            print(uc_best_opt)
            if np.all(np.isclose(uc_true, np.sort(uc_best_opt), atol=1e-3)):
                report_counts['Found and best'] += 1
            else:
                report_counts['Not found'] += 1
        elif len(self.explainers) == 1:
            print(self.explainers['unit_cell'])
            #print(self.explainers['unit_cell_initial'])
            #print(bl_true)
            #print(self.explainers['unit_cell_generator'])
            uc_best_opt = self.explainers.iloc[0]['unit_cell']
            if np.all(np.isclose(uc_true, np.sort(uc_best_opt), atol=1e-3)):
                report_counts['Found and best'] += 1
            else:
                report_counts['Not found'] += 1
        else:
            print(self.explainers['unit_cell'])
            #print(self.explainers['unit_cell_initial'])
            explainer_uc, unique_indices = np.unique(
                np.round(np.stack(self.explainers['unit_cell']), decimals=4), 
                return_index=True, axis=0
                )
            uc_best_opt = explainer_uc[0]
            if explainer_uc.shape[0] == 1:
                print(uc_best_opt)
                if np.all(np.isclose(uc_true, np.sort(uc_best_opt), atol=1e-3)):
                    report_counts['Found and best'] += 1
                else:
                    report_counts['Not found'] += 1
            else:
                found_best = False
                found_not_best = False
                for explainer_index, uc in enumerate(explainer_uc):
                    if np.all(np.isclose(uc_true, np.sort(uc), atol=1e-3)):
                        if explainer_index == 0:
                            found_best = True
                        else:
                            found_not_best = True
                if found_best:
                    report_counts['Found and best'] += 1
                elif found_not_best:
                    report_counts['Found but not best'] += 1
                else:
                    report_counts['Not found'] += 1

        #difference = np.linalg.norm(uc_true[np.newaxis] - candidate_uc, axis=1)
        #uc_best_cand = candidate_uc[np.argmin(difference)]
        #print(f'{np.abs(uc_best_cand - uc_true)}')
        #print(f'{uc_best_cand} {loss[np.argmin(difference)]}')
        #for i in range(5):
        #    print(f'{candidate_uc[i]} {loss[i]}')
        #print()
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
            'n_candidates': 100,
            'subsampling_iterations': [[1, 5], [2, 3], [3, 1]],
            'minimum_uc': 2,
            'maximum_uc': 500,
            'tuning_param': [1, 10],
            'load_predictions': False,
            'n_pred_evals': 500,
            'found_tolerance_key': 'wL2',
            'found_tolerance': 1e-20,
            'assignment_batch_size': 64,
            }
        for key in opt_params_defaults.keys():
            if key not in self.opt_params.keys():
                self.opt_params[key] = opt_params_defaults[key]
        for key in self.assign_params:
            self.assign_params[key]['load_from_tag'] = True
            self.assign_params[key]['mode'] = 'inference'
        self.data_params['load_from_tag'] = True
        self.reg_params['load_from_tag'] = True
        self.save_to = f'models/{self.data_params["tag"]}/optimizer'

        self.indexer = Indexing(
            assign_params=self.assign_params, 
            class_params={'tag': None}, 
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
            self.indexer.N_bl = None
            self.indexer.hkl_ref = None
        self.indexer.N_bl = self.comm.bcast(self.indexer.N_bl, root=0)
        self.indexer.hkl_ref = self.comm.bcast(self.indexer.hkl_ref, root=0)
        self.indexer.setup_regression()
        self.indexer.setup_assignment()
        self.opt_params['minimum_uc_scaled'] = \
            (self.opt_params['minimum_uc'] - self.indexer.uc_scaler.mean_[0]) / self.indexer.uc_scaler.scale_[0]
        self.opt_params['maximum_uc_scaled'] = \
            (self.opt_params['maximum_uc'] - self.indexer.uc_scaler.mean_[0]) / self.indexer.uc_scaler.scale_[0]

        self.logger = logging.getLogger("rank[%i]"%self.comm.rank)
        self.logger.setLevel(logging.DEBUG)                                       
        mh = MPIFileHandler(f'{self.save_to}/{self.opt_params["tag"]}.log')
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
        mh.setFormatter(formatter)
        self.logger.addHandler(mh)

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
            self.rank_indices = np.arange(0, self.indexer.N, self.n_ranks)
            self.indexer.data = self.indexer.data.iloc[self.rank_indices]
            if self.opt_params['load_predictions']:
                self.uc_scaled_mean = uc_scaled_mean_all[self.rank_indices]
                self.uc_scaled_cov = uc_scaled_cov_all[self.rank_indices]
                self.N = self.indexer.data.shape[0]
                self.n_bl = len(self.indexer.data_params['bravais_lattices'])
        else:
            self.rank_indices = self.comm.recv(source=0, tag=0)
            self.indexer.data = self.comm.recv(source=0, tag=1)
            if self.opt_params['load_predictions']:
                self.uc_scaled_mean = self.comm.recv(source=0, tag=2)
                self.uc_scaled_cov = self.comm.recv(source=0, tag=3)
                self.N = self.indexer.data.shape[0]
                self.n_bl = len(self.indexer.data_params['bravais_lattices'])
        if self.opt_params['load_predictions'] == False:
            self.N = self.indexer.data.shape[0]
            self.n_bl = len(self.indexer.data_params['bravais_lattices'])
            self.uc_scaled_mean = np.zeros((self.N, self.n_bl, self.indexer.data_params['n_outputs']))
            self.uc_scaled_cov = np.zeros((
                self.N,
                self.n_bl,
                self.indexer.data_params['n_outputs'],
                self.indexer.data_params['n_outputs']
                ))
            for bl_index, bravais_lattice in enumerate(self.indexer.data_params['bravais_lattices']):
                mean, cov = self.indexer.unit_cell_generator[bravais_lattice].do_predictions(
                    data=self.indexer.data,
                    verbose=0,
                    n_evals=self.opt_params['n_pred_evals'],
                    )
                self.uc_scaled_mean[:, bl_index, :] = mean
                self.uc_scaled_cov[:, bl_index, :, :] = cov
                        
            if self.rank == 0:
                uc_scaled_mean_all = [None for i in range(self.n_ranks)]
                uc_scaled_cov_all = [None for i in range(self.n_ranks)]
                rank_indices_all = [None for i in range(self.n_ranks)]
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
        """
        for entry_index in range(self.N):
            print(self.indexer.data.iloc[entry_index]['bravais_lattice'])
            print(self.indexer.data.iloc[entry_index]['reordered_unit_cell'])
            for bl_index in range(self.n_bl):
                print(self.indexer.revert_predictions(uc_pred_scaled=self.uc_scaled_mean[entry_index, bl_index, :]))
            print()
        print(f'{self.rank} {self.uc_scaled_mean.sum()}, {self.uc_scaled_cov.sum()}')
        """

    def run(self):
        uc_true = np.stack(self.indexer.data['reordered_unit_cell'])[:, self.indexer.data_params['y_indices']]
        uc_pred = np.stack(self.indexer.data['reordered_unit_cell_pred'])
        uc_pred_cov = np.stack(self.indexer.data['reordered_unit_cell_pred_cov'])
        diag_indices = np.arange(uc_true.shape[1])
        uc_pred_std = np.sqrt(uc_pred_cov[:, diag_indices, diag_indices])
        bl_true = list(self.indexer.data['bravais_lattice'])
        sg_true = list(self.indexer.data['spacegroup_number'])
        hkl_true = np.stack(self.indexer.data['reordered_hkl'])[:, :, :, 0]

        uc_best_opt = np.zeros((self.N, self.indexer.data_params['n_outputs']))
        uc_best_cand = np.zeros((self.N, self.indexer.data_params['n_outputs']))
        hkl_best_opt = np.zeros((self.N, self.indexer.data_params['n_points'], 3))
        hkl_best_cand = np.zeros((self.N, self.indexer.data_params['n_points'], 3))

        percentage = 0
        report_counts = {
            'Not found': 0,
            'Found and best': 0,
            'Found but not best': 0
            }
        for entry_index in range(self.N):
            current_percentage = entry_index / self.N
            if current_percentage > percentage + 0.01:
                self.logger.info(f' {100*current_percentage:3.0f}% complete')
                percentage += 0.01
            print()
            #print(uc_true[entry_index])
            candidate_uc_scaled = np.zeros((
                self.n_bl * self.opt_params['n_candidates'],
                self.indexer.data_params['n_outputs']
                ))
            unit_cell_generator_labels = []
            for bl_index in range(self.n_bl):
                #print(self.indexer.revert_predictions(uc_pred_scaled=self.uc_scaled_mean[entry_index, bl_index, :]))
                start = bl_index * self.opt_params['n_candidates']
                stop = (bl_index + 1) * self.opt_params['n_candidates']
                unit_cell_generator_labels += [
                    self.indexer.data_params['bravais_lattices'][bl_index] 
                    for i in range(self.opt_params['n_candidates'])
                    ]
                candidates_scaled = self.rng.multivariate_normal(
                    mean=self.uc_scaled_mean[entry_index, bl_index, :],
                    cov=self.uc_scaled_cov[entry_index, bl_index, :, :],
                    size=self.opt_params['n_candidates'],
                    )
                ###!!! FIX FOR ANGLES
                bad_indices = np.logical_or(
                    np.any(candidates_scaled < self.opt_params['minimum_uc_scaled'], axis=1),
                    np.any(candidates_scaled > self.opt_params['maximum_uc_scaled'], axis=1)
                    )
                n_bad_indices = np.sum(bad_indices)
                while n_bad_indices > 0:
                    candidates_scaled[bad_indices] = self.rng.multivariate_normal(
                        mean=self.uc_scaled_mean[entry_index, bl_index, :],
                        cov=self.uc_scaled_cov[entry_index, bl_index, :, :],
                        size=n_bad_indices,
                        )
                    ###!!! FIX FOR ANGLES
                    bad_indices = np.any(candidates_scaled <= self.opt_params['minimum_uc_scaled'], axis=1)
                    n_bad_indices = np.sum(bad_indices)

                candidate_uc_scaled[start: stop, :] = candidates_scaled
            candidates = Candidates(
                q2_obs=np.array(self.indexer.data.iloc[entry_index]['q2']),
                q2_obs_scaled=np.array(self.indexer.data.iloc[entry_index]['q2_scaled']),
                unit_cell=self.indexer.revert_predictions(uc_pred_scaled=candidate_uc_scaled),
                unit_cell_scaled=candidate_uc_scaled,
                minimum_unit_cell=self.opt_params['minimum_uc'],
                maximum_unit_cell=self.opt_params['maximum_uc'],
                unit_cell_generator_label=unit_cell_generator_labels,
                tolerance=self.opt_params['found_tolerance'],
                tolerance_key=self.opt_params['found_tolerance_key']
                )
            candidates = self.assign_hkls(candidates, self.opt_params['iteration_info'][0][0])
            candidates.candidates['hkl_initial'] = candidates.candidates['hkl'].copy()
            candidates.initial_diagnostics(
                uc_true[entry_index], uc_pred[entry_index], uc_pred_std[entry_index], 
                hkl_true[entry_index], bl_true[entry_index], sg_true[entry_index]
                )
            candidates = self.optimize_entry(candidates)
            uc_best_opt[entry_index], report_counts = candidates.get_best_candidates(
                uc_true[entry_index], bl_true[entry_index], hkl_true[entry_index], report_counts
                )
            print(report_counts)
        self.indexer.data['reordered_unit_cell_best_opt'] = list(uc_best_opt)

    def optimize_entry(self, candidates):
        candidates = self.assign_hkls(candidates, self.opt_params['iteration_info'][0][0])
        candidates.candidates['hkl_initial'] = candidates.candidates['hkl'].copy()
        for iteration_info in self.opt_params['iteration_info']:
            assigner_key = iteration_info[0]
            n_opt_iterations = iteration_info[1]
            n_subsample = iteration_info[2]
            n_drop = iteration_info[3]
            for iter_index in range(n_opt_iterations):
                candidates = self.optimize_iteration(candidates, assigner_key, n_subsample, n_drop)
                #print(f'{candidates.n}, {candidates.candidates[self.opt_params["found_tolerance_key"]].mean()} {assigner_key}')
                #print(len(candidates.explainers))
        #candidate_uc, loss = self.multiple_assignments(candidate_uc, q2_scaled)
        """
        for subsampled_index in range(len(self.opt_params['subsampling_iterations'])):
            n_drop = self.opt_params['subsampling_iterations'][subsampled_index][0]
            n_iterations = self.opt_params['subsampling_iterations'][subsampled_index][1]
            for iter_index in range(n_iterations):
                candidate_uc, explainers, loss = self.deterministic_subsampling(
                    candidate_uc, explainers, q2, q2_scaled, n_drop
                    )
                print(f'{loss.size}, {loss.mean()}')
                print(len(explainers))
        """
        return candidates

    def assign_hkls(self, candidates, assigner_key, subsampled_candidates=None):
        """
        I'm using model.predict_on_batch as a work around for memory leak issues
        - Inputs must be the same size each time predict_on_batch is called.
        - model.call() / model() and model.predict() lead to a really bad memory leak
        - model.predict_on_batch when the batch size is different leads to the same memory leak
        - tf.nn.softmax gives a small memory leak as well.
        https://github.com/tensorflow/tensorflow/issues/44711
        https://github.com/keras-team/keras/issues/13118
        https://github.com/tensorflow/tensorflow/issues/33009        
        """
        if subsampled_candidates is None:
            unit_cell_scaled = np.stack(candidates.candidates['unit_cell_scaled'])
        else:
            unit_cell_scaled = np.stack(subsampled_candidates['unit_cell_scaled'])

        n_batchs = candidates.n // self.opt_params['assignment_batch_size']
        left_over = candidates.n % self.opt_params['assignment_batch_size']
        #print(f'{n_batchs} {left_over}')
        batch_q2_scaled = np.repeat(
            candidates.q2_obs_scaled[np.newaxis, :], 
            repeats=self.opt_params['assignment_batch_size'],
            axis=0
            )
        logits = np.zeros((
            candidates.n,
            self.indexer.data_params['n_points'],
            self.indexer.data_params['hkl_ref_length']
            ))
        for batch_index in range(n_batchs + 1):
            start = batch_index * self.opt_params['assignment_batch_size']
            if batch_index == n_batchs:
                batch_unit_cell_scaled = np.zeros((
                    self.opt_params['assignment_batch_size'], self.indexer.data_params['n_outputs']
                    ))
                batch_unit_cell_scaled[:left_over] = unit_cell_scaled[start: start + left_over]
                batch_unit_cell_scaled[left_over:] = batch_unit_cell_scaled[0]
            else:
                end = (batch_index + 1) * self.opt_params['assignment_batch_size']
                batch_unit_cell_scaled = unit_cell_scaled[start: end]
            inputs = {
                'unit_cell_scaled': batch_unit_cell_scaled,
                'q2_scaled': batch_q2_scaled
                }
            logits_batch = self.indexer.assigner[assigner_key].model.predict_on_batch(inputs)
            if batch_index == n_batchs:
                logits[start: start + left_over] = logits_batch[:left_over]
            else:
                logits[start: end] = logits_batch

        # Using tf.nn.softmax results in a memory leak
        # Switching to scipy solves the issue.
        # This is slightly faster than the scipy.special implementation
        exp_logits = np.exp(logits)
        softmax_all = exp_logits / exp_logits.sum(axis=2, keepdims=True)
        if subsampled_candidates is None:
            candidates.candidates['softmax'] = list(softmax_all.max(axis=2))
            candidates.candidates['hkl'] = list(self.indexer.convert_softmax_to_assignments(softmax_all))
            return candidates
        else:
            subsampled_candidates['softmax'] = list(softmax_all.max(axis=2))
            subsampled_candidates['hkl'] = list(self.indexer.convert_softmax_to_assignments(softmax_all))
            return subsampled_candidates

    def assign_hkls_closest(self, candidates):
        unit_cell_scaled = np.stack(candidates.candidates['unit_cell_scaled'])
        q2_obs_scaled = np.repeat(
            candidates.q2_obs_scaled[np.newaxis, :], repeats=candidates.n, axis=0
            )
        pairwise_differences_scaled = self.indexer.assigner['0'].pairwise_difference_calculation.get_pairwise_differences_from_uc_scaled(
            unit_cell_scaled, q2_obs_scaled
            )
        pds_inv = self.indexer.assigner['0'].transform_pairwise_differences(
            pairwise_differences_scaled, tensorflow=False
            )
        softmax_all = scipy.special.softmax(pds_inv, axis=2)
        candidates.candidates['softmax'] = list(softmax_all.max(axis=2))
        candidates.candidates['hkl'] = list(self.indexer.convert_softmax_to_assignments(softmax_all))
        return candidates

    def update_candidates(self, candidates, assigner_key, optimized_unit_cell):
        n_subsample = optimized_unit_cell.shape[1]
        if n_subsample == 1:
            candidates.candidates['unit_cell'] = list(optimized_unit_cell[:, 0, :])
            candidates.candidates['unit_cell_scaled'] = \
                list(self.indexer.scale_predictions(optimized_unit_cell[:, 0, :]))
            if assigner_key == 'closest':
                candidates = self.assign_hkls_closest(candidates)
            else:
                candidates = self.assign_hkls(candidates, assigner_key)
            metrics = np.zeros((candidates.n, 5))
            hkl = np.stack(candidates.candidates['hkl'])
            softmax = np.stack(candidates.candidates['softmax'])
            unit_cell = np.stack(candidates.candidates['unit_cell'])
            for candidate_index in range(candidates.n):
                self.target_function.update(
                    hkl[candidate_index], softmax[candidate_index], unit_cell[candidate_index]
                    )
                metrics[candidate_index, :] = self.target_function.metrics(unit_cell[candidate_index])
            candidates.candidates['L1'] = metrics[:, 0]
            candidates.candidates['L2'] = metrics[:, 1]
            candidates.candidates['wL1'] = metrics[:, 2]
            candidates.candidates['wL2'] = metrics[:, 3]
            candidates.candidates['loss'] = metrics[:, 4]

        else:
            subsampled_candidates = [candidates.candidates.copy() for i in range(n_subsample)]
            subsampled_metrics = np.zeros((candidates.n, n_subsample, 5))
            for subsampled_index in range(n_subsample):
                subsampled_candidates[subsampled_index]['unit_cell'] = \
                    list(optimized_unit_cell[:, subsampled_index, :])
                subsampled_candidates[subsampled_index]['unit_cell_scaled'] = \
                    list(self.indexer.scale_predictions(optimized_unit_cell[:, subsampled_index, :]))
                if assigner_key == 'closest':
                    assert False
                else:
                    subsampled_candidates[subsampled_index] = \
                        self.assign_hkls(candidates, assigner_key, subsampled_candidates[subsampled_index])
                hkl = np.stack(subsampled_candidates[subsampled_index]['hkl'])
                softmax = np.stack(subsampled_candidates[subsampled_index]['softmax'])
                unit_cell = np.stack(subsampled_candidates[subsampled_index]['unit_cell'])
                for candidate_index in range(candidates.n):
                    self.target_function.update(
                        hkl[candidate_index], softmax[candidate_index], unit_cell[candidate_index]
                        )
                    subsampled_metrics[candidate_index, subsampled_index, :] = \
                        self.target_function.metrics(unit_cell[candidate_index])
            best_subsample = np.argmin(subsampled_metrics[:, :, 3], axis=1)
            metrics = np.min(subsampled_metrics, axis=1)
            unit_cell = np.zeros((candidates.n, self.indexer.data_params['n_outputs']))
            unit_cell_scaled = np.zeros((candidates.n, self.indexer.data_params['n_outputs']))
            hkl = np.zeros((candidates.n, self.indexer.data_params['n_points'], 3))
            softmax = np.zeros((candidates.n, self.indexer.data_params['n_points']))
            for candidate_index in range(candidates.n):
                best = subsampled_candidates[best_subsample[candidate_index]].iloc[candidate_index]
                unit_cell[candidate_index] = best['unit_cell']
                unit_cell_scaled[candidate_index] = best['unit_cell_scaled']
                hkl[candidate_index] = best['hkl']
                softmax[candidate_index] = best['softmax']
            candidates.candidates['unit_cell'] = list(unit_cell)
            candidates.candidates['unit_cell_scaled'] = list(unit_cell_scaled)
            candidates.candidates['hkl'] = list(hkl)
            candidates.candidates['softmax'] = list(softmax)

        candidates.candidates['L1'] = metrics[:, 0]
        candidates.candidates['L2'] = metrics[:, 1]
        candidates.candidates['wL1'] = metrics[:, 2]
        candidates.candidates['wL2'] = metrics[:, 3]
        candidates.candidates['loss'] = metrics[:, 4]
        candidates.update()
        return candidates

    def optimize_iteration(self, candidates, assigner_key, n_subsample, n_drop):
        if n_drop == 0:
            n_subsample = 1

        optimized_unit_cell = np.zeros((
            candidates.n, n_subsample, self.indexer.data_params['n_outputs']
            ))
        hkl = np.stack(candidates.candidates['hkl'])
        softmax = np.stack(candidates.candidates['softmax'])
        unit_cell = np.stack(candidates.candidates['unit_cell'])
        if n_drop == 0:
            self.target_function = CandidateOptLoss(
                q2_obs=candidates.q2_obs, 
                lattice_system=self.indexer.data_params['lattice_system'],
                tuning_param=self.opt_params['tuning_param'][0],
                )
        for candidate_index in range(candidates.n):
            for subsampled_index in range(n_subsample):
                if n_drop != 0:
                    subsampled_indices = self.rng.choice(
                        self.indexer.data_params['n_points'],
                        size=self.indexer.data_params['n_points'] - n_drop,
                        replace=False,
                        p=softmax[candidate_index] / softmax[candidate_index].sum()
                        )
                    self.target_function = CandidateOptLoss(
                        q2_obs=candidates.q2_obs[subsampled_indices], 
                        lattice_system=self.indexer.data_params['lattice_system'],
                        tuning_param=self.opt_params['tuning_param'][0],
                        )
                    hkl_ = hkl[candidate_index][subsampled_indices]
                    softmax_ = softmax[candidate_index][subsampled_indices]
                else:
                    hkl_ = hkl[candidate_index]
                    softmax_ = softmax[candidate_index]

                self.target_function.update(hkl_, softmax_, unit_cell[candidate_index])
                """
                results = scipy.optimize.minimize(
                    self.target_function.loss_likelihood,
                    x0=unit_cell[candidate_index],
                    method='L-BFGS-B',
                    jac=True,
                    )
                """
                results = scipy.optimize.minimize(
                    self.target_function.loss_likelihood,
                    x0=unit_cell[candidate_index],
                    method='trust-krylov',
                    jac=True,
                    hess=self.target_function.loss_likelihood_hessian
                    )
                optimized_unit_cell[candidate_index, subsampled_index] = np.abs(results.x)
        if n_drop != 0:
            self.target_function = CandidateOptLoss(
                q2_obs=candidates.q2_obs, 
                lattice_system=self.indexer.data_params['lattice_system'],
                tuning_param=self.opt_params['tuning_param'][0],
                )
        candidates = self.update_candidates(candidates, assigner_key, optimized_unit_cell)
        return candidates

    """
    def multiple_assignments(self, candidate_uc, q2_scaled):
        # This should be broken
        def get_repeats(hkl_labels_pred):
            m = np.ones(hkl_labels_pred.shape, dtype=bool)
            m[np.unique(hkl_labels_pred, return_index=True)[1]] = False
            return np.unique(hkl_labels_pred[m])

        candidate_hkls, candidate_softmaxes, candidate_softmaxes_all = self.assign_hkls(candidate_uc, q2_scaled)
        candidate_hkl_labels = candidate_softmaxes_all.argmax(axis=2)
        for entry_index in range(candidate_uc.shape[0]):
            entry_hkl_labels = candidate_hkl_labels[entry_index]
            entry_softmaxes_all = candidate_softmaxes_all[entry_index]

            repeated = get_repeats(entry_hkl_labels)
            x = np.arange(self.indexer.data_params['n_points'])
            #print(entry_hkl_labels)
            #print(entry_softmaxes_all.max(axis=1))
            #print(np.unique(entry_hkl_labels))
            #print(repeated)
            if len(repeated) > 0:
                for repeat in repeated:
                    common_indices = entry_hkl_labels == repeat
                    for reaarange_index in np.argsort(entry_softmaxes_all[common_indices, repeat])[:-1]:
                        peak_to_reassign = x[common_indices][reaarange_index]
                        new_choice_indices = np.argsort(entry_softmaxes_all[peak_to_reassign])[::-1]
                        for new_index in new_choice_indices[1:]:
                            if not new_index in entry_hkl_labels:
                                entry_hkl_labels[peak_to_reassign] = new_index
                                break
            #print(entry_hkl_labels)
            #print(entry_softmaxes_all[x, entry_hkl_labels])
            #print(entry_hkls[x, ])
            #print()
            candidate_hkl_labels[entry_index] = entry_hkl_labels
            candidate_softmaxes[entry_index] = entry_softmaxes_all[x, entry_hkl_labels]
            #print(candidate_hkls[entry_index])
            candidate_hkls[entry_index] = self.indexer.hkl_ref[entry_hkl_labels]
            #print(candidate_hkls[entry_index])
            #print()
        candidate_hkls, candidate_softmaxes, candidate_uc = self.drop_identical_assignments(
            candidate_hkls, candidate_softmaxes, candidate_uc
            )
        candidate_uc, loss = self.optimize_candidates(candidate_hkls, candidate_softmaxes, candidate_uc)
        candidate_uc, candidate_hkls, candidate_softmaxes, loss = self.drop_bad_optimizations(candidate_uc, candidate_hkls, candidate_softmaxes, loss)
        return candidate_uc, loss
    """

    def deterministic_subsampling(self, candidate_uc, explainers, q2, q2_scaled, n_drop):
        candidate_hkls, candidate_softmaxes, candidate_softmaxes_all = \
            self.assign_hkls(candidate_uc, q2_scaled, self.opt_params['subsampling_assignment_key'])
        candidate_hkls, candidate_softmaxes, candidate_uc = \
            self.drop_identical_assignments(candidate_hkls, candidate_softmaxes, candidate_uc)

        n_candidates = candidate_uc.shape[0]
        n_points = self.indexer.data_params['n_points']
        if n_drop == 1:
            n_subsamples = n_points
        elif n_drop == 2:
            n_subsamples = int(n_points * (n_points - 1) / 2)
        elif n_drop == 3:
            n_subsamples = int(n_points * (n_points - 1) * (n_points - 2) / (3 * 2))
        print(f'drop {n_drop}')

        subsampled_uc = np.zeros((n_candidates, n_subsamples, candidate_uc.shape[1]))
        subsampled_loss = np.zeros((n_candidates, n_subsamples))
        if n_drop == 1:
            for point_index in range(n_points):
                subsampled_indices = np.ones(n_points, dtype=bool)
                subsampled_indices[point_index] = False
                self.target_function = CandidateOptLoss(
                    q2_obs=q2[subsampled_indices], 
                    lattice_system=self.indexer.data_params['lattice_system'],
                    tuning_param=self.opt_params['tuning_param'][0],
                    )
                subsampled_uc[:, point_index, :], subsampled_loss[:, point_index] = self.optimize_candidates(
                    candidate_hkls[:, subsampled_indices, :], 
                    candidate_softmaxes[:, subsampled_indices], 
                    candidate_uc
                    )
        elif n_drop == 2:
            point_index = 0
            for point_index_i in range(n_points - 1):
                for point_index_j in range(point_index_i + 1, n_points):
                    subsampled_indices = np.ones(n_points, dtype=bool)
                    subsampled_indices[point_index_i] = False
                    subsampled_indices[point_index_j] = False
                    self.target_function = CandidateOptLoss(
                        q2_obs=q2[subsampled_indices], 
                        lattice_system=self.indexer.data_params['lattice_system'],
                        tuning_param=self.opt_params['tuning_param'][0],
                        )
                    subsampled_uc[:, point_index, :], subsampled_loss[:, point_index] = self.optimize_candidates(
                        candidate_hkls[:, subsampled_indices, :], 
                        candidate_softmaxes[:, subsampled_indices], 
                        candidate_uc
                        )
                    point_index += 1
        elif n_drop == 3:
            point_index = 0
            for point_index_i in range(n_points - 2):
                for point_index_j in range(point_index_i + 1, n_points - 1):
                    for point_index_k in range(point_index_j + 1, n_points):
                        subsampled_indices = np.ones(n_points, dtype=bool)
                        subsampled_indices[point_index_i] = False
                        subsampled_indices[point_index_j] = False
                        subsampled_indices[point_index_k] = False
                        self.target_function = CandidateOptLoss(
                            q2_obs=q2[subsampled_indices], 
                            lattice_system=self.indexer.data_params['lattice_system'],
                            tuning_param=self.opt_params['tuning_param'][0],
                            )
                        subsampled_uc[:, point_index, :], subsampled_loss[:, point_index] = self.optimize_candidates(
                            candidate_hkls[:, subsampled_indices, :], 
                            candidate_softmaxes[:, subsampled_indices], 
                            candidate_uc
                            )
                        point_index += 1

        closest = np.argmin(subsampled_loss, axis=1)
        loss = np.min(subsampled_loss, axis=1)
        candidate_uc = subsampled_uc[np.arange(n_candidates), closest, :]
        candidate_uc, candidate_hkls, candidate_softmaxes, loss = self.drop_bad_optimizations(candidate_uc, candidate_hkls, candidate_softmaxes, loss)
        candidate_uc, explainers, loss = self.pick_explainers(candidate_uc, candidate_hkls, candidate_softmaxes, explainers, loss)
        return candidate_uc, explainers, loss

    def random_subsampling(self, candidate_uc, explainers, q2, q2_scaled, n_drop):
        candidate_hkls, candidate_softmaxes, candidate_softmaxes_all = \
            self.assign_hkls(candidate_uc, q2_scaled, self.opt_params['subsampling_assignment_key'])
        candidate_hkls, candidate_softmaxes, candidate_uc = \
            self.drop_identical_assignments(candidate_hkls, candidate_softmaxes, candidate_uc)

        n_candidates = candidate_uc.shape[0]
        n_points = self.indexer.data_params['n_points']
        if n_drop == 1:
            n_subsamples = n_points
        elif n_drop == 2:
            n_subsamples = int(n_points * (n_points - 1) / 2)
        elif n_drop == 3:
            n_subsamples = int(n_points * (n_points - 1) * (n_points - 2) / (3 * 2))

        subsampled_uc = np.zeros((n_candidates, n_subsamples, candidate_uc.shape[1]))
        subsampled_loss = np.zeros((n_candidates, n_subsamples))

    def cluster_candidates(self, candidate_uc, uc_true):
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].plot(candidate_uc[:, 0], candidate_uc[:, 1], linestyle='none', marker='.')
        axes[1].plot(candidate_uc[:, 0], candidate_uc[:, 2], linestyle='none', marker='.')
        axes[2].plot(candidate_uc[:, 1], candidate_uc[:, 2], linestyle='none', marker='.')
        axes[0].plot(uc_true[0], uc_true[1], linestyle='none', marker='x')
        axes[1].plot(uc_true[0], uc_true[2], linestyle='none', marker='x')
        axes[2].plot(uc_true[1], uc_true[2], linestyle='none', marker='x')
        plt.show()

    def gather_optimized_unit_cells(self):
        if self.rank == 0:
            optimized_data = [None for i in range(self.n_ranks)]
            optimized_data[0] = self.indexer.data
            for rank_index in range(1, self.n_ranks):
                optimized_data[rank_index] = self.comm.recv(source=rank_index, tag=2)
            self.optimized_data = pd.concat(optimized_data)
            self.optimized_data['reordered_h'] = list(np.stack(self.optimized_data['reordered_hkl'])[:, :, 0, 0])
            self.optimized_data['reordered_k'] = list(np.stack(self.optimized_data['reordered_hkl'])[:, :, 1, 0])
            self.optimized_data['reordered_l'] = list(np.stack(self.optimized_data['reordered_hkl'])[:, :, 2, 0])
            drop_columns = [
                'hkl',
                'reordered_hkl',
                'reordered_unit_cell_pred_cov',
                'reordered_unit_cell_pred_scaled_cov'
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
                uc_true_bl = np.stack(bl_data['reordered_unit_cell'])[:, self.indexer.data_params['y_indices']]
                uc_best_opt_bl = np.stack(bl_data['reordered_unit_cell_best_opt'])
                uc_best_cand_bl = np.stack(bl_data['reordered_unit_cell_best_cand'])
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


class MPIFileHandler(logging.FileHandler):
    """
    Code was copied from https://gist.github.com/muammar/2baec60fa8c7e62978720686895cdb9f

    Created on Wed Feb 14 16:17:38 2018
    This handler is used to deal with logging with mpi4py in Python3.
    @author: cheng
    @reference: 
        https://cvw.cac.cornell.edu/python/logging
        https://groups.google.com/forum/#!topic/mpi4py/SaNzc8bdj6U
        https://gist.github.com/JohnCEarls/8172807
    """                                  
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND ,
                 encoding='utf-8',  
                 delay=False,
                 comm=MPI.COMM_WORLD):                                                
        self.baseFilename = os.path.abspath(filename)                           
        self.mode = mode                                                        
        self.encoding = encoding                                            
        self.comm = comm                                                        
        if delay:                                                               
            #We don't open the stream, but we still need to call the            
            #Handler constructor to set level, formatter, lock etc.             
            logging.Handler.__init__(self)                                      
            self.stream = None                                                  
        else:                                                                   
           logging.StreamHandler.__init__(self, self._open())                   
                                                                                
    def _open(self):                                                            
        stream = MPI.File.Open( self.comm, self.baseFilename, self.mode )     
        stream.Set_atomicity(True)                                              
        return stream
                                                    
    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        
        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept 
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
            #self.flush()
        except Exception:
            self.handleError(record)
                                                         
    def close(self):                                                            
        if self.stream:                                                         
            self.stream.Sync()                                                  
            self.stream.Close()                                                 
            self.stream = None


if (__name__ == '__main__'):
    data_params = {
        'tag': 'Indexing_orthorhombic_20points',
        }
    oP_params = {
        'tag': 'mlp_20points',
        'load_from_tag': True,
        'var_est': 'alpha_beta',
        'alpha_params': {},
        'beta_params': {},
        'mean_params': {},
        'var_params': {},
        'head_params': {},
        }
    reg_params = {
        'oP': oP_params,
        'oC': oP_params,
        'oF': oP_params,
        'oI': oP_params,
        }
    assign_params = {
        '0': {'tag': 'mlp_20points_0'},
        '1': {'tag': 'mlp_20points_1'},
        '2': {'tag': 'mlp_20points_2'},
        '3': {'tag': 'mlp_20points_3'},
        '4': {'tag': 'mlp_20points_4'},
        '5': {'tag': 'mlp_20points_5'},
        }

    opt_params = {
        'tag': 'mlp_20points',
        'n_candidates': 200,
        'iteration_info': [
            ['1', 1, 10, 5],
            ['2', 10, 1, 0], 
            ['3', 10, 1, 0],
            ['4', 10, 1, 0],
            ['5', 10, 1, 0],
            ['closest', 10, 0, 0],
            ],
        'found_tolerance': 1e-20,
        'found_tolerance_key': 'wL2',
        'subsampling_iterations': [[1, 5], [2, 3]],
        'subsampling_assignment_key': '5',
        'minimum_uc': 2,
        'maximum_uc': 500,
        'tuning_param': [1, 5],
        'load_predictions': False,
        'n_pred_evals': 500,
        'assignment_batch_size': 64,
        }

    optimizer = Optimizer(assign_params, data_params, opt_params, reg_params)
    optimizer.distribute_data()
    optimizer.run()
    optimizer.evaluate()

