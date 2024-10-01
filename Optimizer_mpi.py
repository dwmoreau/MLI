import copy
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
pd.options.mode.copy_on_write = True
import scipy.optimize
import scipy.spatial
import scipy.special
import time
from tqdm import tqdm

from Indexing import Indexing
from Reindexing import get_different_monoclinic_settings
from Reindexing import get_s6_from_unit_cell
from Reindexing import monoclinic_standardization
from Reindexing import reindex_entry_basic
from Reindexing import selling_reduction
from TargetFunctions import CandidateOptLoss
from Utilities import fix_unphysical
from Utilities import get_extinction_group
from Utilities import get_hkl_matrix
from Utilities import get_M20
from Utilities import get_M20_from_xnn
from Utilities import get_M20_likelihood
from Utilities import get_M20_likelihood_from_xnn
from Utilities import get_M20_triplet
from Utilities import get_M20_triplet_from_xnn
from Utilities import get_reciprocal_unit_cell_from_xnn
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import get_xnn_from_unit_cell
from Utilities import get_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import Q2Calculator
from Utilities import reciprocal_uc_conversion
from Utilities import fast_assign


class Candidates:
    def __init__(self, q2_obs, triplets, xnn, hkl_ref, lattice_system, bravais_lattice, minimum_unit_cell, maximum_unit_cell):
        self.lattice_system = lattice_system
        self.bravais_lattice = bravais_lattice
        self.minimum_unit_cell = minimum_unit_cell
        self.maximum_unit_cell = maximum_unit_cell
        self.rng = np.random.default_rng()
        self.hkl_ref = hkl_ref
        self.hkl_ref_length = hkl_ref.shape[0]

        self.q2_obs = q2_obs
        self.n_peaks = self.q2_obs.size
        self.triplets = triplets
        self.n_triplets = triplets.shape[0]
        self.xnn = xnn
        self.n = self.xnn.shape[0]
        self.best_xnn = self.xnn.copy()
        self.best_M20 = np.zeros(self.n)
        self.candidate_index = np.arange(self.n)
        self.update_unit_cell_from_xnn()

        self.q2_calculator = Q2Calculator(
            lattice_system=self.lattice_system,
            hkl=self.hkl_ref,
            tensorflow=False,
            representation='xnn'
            )
        # self.assign_hkls() calculates the current M20 score
        self.assign_hkls()
        self.best_hkl = self.hkl.copy()

    def fix_bad_conversions(self):
        bad_conversions = np.sum(np.isnan(self.reciprocal_unit_cell), axis=1) > 0
        good_indices = np.arange(self.reciprocal_unit_cell.shape[0])[~bad_conversions]
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

    def fix_out_of_range_candidates(self):
        self.xnn = fix_unphysical(
            xnn=self.xnn,
            rng=self.rng,
            minimum_unit_cell=self.minimum_unit_cell, 
            maximum_unit_cell=self.maximum_unit_cell,
            lattice_system=self.lattice_system,
            )
        self.update_unit_cell_from_xnn()

    def assign_hkls(self):
        q2_ref_calc = self.q2_calculator.get_q2(self.xnn)
        hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
        self.hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
        if self.triplets is None:
            self.M20 = get_M20_from_xnn(
                self.q2_obs, self.xnn, self.hkl, self.hkl_ref, self.lattice_system
                )
        else:
            hkl_assign_triplets = fast_assign(self.triplets[:, 2], q2_ref_calc)
            self.hkl_triplets = np.take(self.hkl_ref, hkl_assign_triplets, axis=0)
            self.M20 = get_M20_triplet(
                self.q2_obs,
                self.triplets,
                self.hkl,
                self.xnn,
                self.lattice_system,
                self.bravais_lattice
                )

    def random_subsampling_original(self, iteration_info):
        n_keep = self.n_peaks - iteration_info['n_drop']
        subsampled_indices = self.rng.permuted(
            np.repeat(np.arange(self.n_peaks)[np.newaxis], self.n, axis=0),
            axis=1
            )[:, :n_keep]
        hkl_subsampled = np.take_along_axis(
            self.hkl, subsampled_indices[:, :, np.newaxis], axis=1
            )
        q2_subsampled = np.take(self.q2_obs, subsampled_indices)

        target_function = CandidateOptLoss(
            q2_subsampled, 
            lattice_system=self.lattice_system,
            )
        target_function.update(hkl_subsampled, self.xnn)

        self.xnn += target_function.gauss_newton_step(self.xnn)
        self.fix_out_of_range_candidates()
        self.assign_hkls()
        improved_M20 = self.M20 > self.best_M20
        self.best_M20[improved_M20] = self.M20[improved_M20]
        self.best_xnn[improved_M20] = self.xnn[improved_M20]
        self.best_hkl[improved_M20] = self.hkl[improved_M20]

    def random_subsampling(self, iteration_info):
        n_keep = self.n_peaks - iteration_info['n_drop']
        subsampled_indices = self.rng.permuted(
            np.repeat(np.arange(self.n_peaks)[np.newaxis], self.n, axis=0),
            axis=1
            )[:, :n_keep]
        hkl_subsampled = np.take_along_axis(
            self.hkl, subsampled_indices[:, :, np.newaxis], axis=1
            )
        q2_subsampled = np.take(self.q2_obs, subsampled_indices)

        if self.triplets is None:
            target_function = CandidateOptLoss(
                q2_subsampled, 
                lattice_system=self.lattice_system,
                )
            target_function.update(hkl_subsampled, self.xnn)
        else:
            n_keep_triplets = int(np.round(n_keep / self.n_peaks * self.n_triplets))
            subsampled_indices_triplets = self.rng.permuted(
                np.repeat(np.arange(self.n_triplets)[np.newaxis], self.n, axis=0),
                axis=1
                )[:, :n_keep_triplets]
            hkl_subsampled_triplets = np.take_along_axis(
                self.hkl_triplets, subsampled_indices_triplets[:, :, np.newaxis], axis=1
                )
            q2_subsampled_triplets = np.take(self.triplets[:, 2], subsampled_indices_triplets)

            hkl_subsampled_both = np.concatenate(
                (hkl_subsampled, hkl_subsampled_triplets), axis=1
                )
            q2_subsampled_both = np.concatenate(
                (q2_subsampled, q2_subsampled_triplets), axis=1
                )

            target_function = CandidateOptLoss(
                q2_subsampled_both, 
                lattice_system=self.lattice_system,
                )
            target_function.update(hkl_subsampled_both, self.xnn)

        self.xnn += target_function.gauss_newton_step(self.xnn)
        self.fix_out_of_range_candidates()
        self.assign_hkls()
        improved_M20 = self.M20 > self.best_M20
        self.best_M20[improved_M20] = self.M20[improved_M20]
        self.best_xnn[improved_M20] = self.xnn[improved_M20]
        self.best_hkl[improved_M20] = self.hkl[improved_M20]

    def refine_cell(self):
        _, probability, _ = get_M20_likelihood_from_xnn(
            q2_obs=self.q2_obs,
            xnn=self.best_xnn,
            hkl=self.best_hkl,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            )
        indexed_peaks = probability > 0.5
        n_indexed_peaks = np.sum(indexed_peaks, axis=1)
        unique_n_indexed_peaks = np.unique(n_indexed_peaks)
        refined_xnn = self.best_xnn.copy()
        for n in unique_n_indexed_peaks:
            candidate_indices = n_indexed_peaks == n
            subsampled_indices = np.argwhere(indexed_peaks[candidate_indices])
            # subsampled_indices: n_candidates x n_peaks
            # hkl:                n_candidates x n_peaks x 3
            subsampled_indices = subsampled_indices[:, 1].reshape((candidate_indices.sum(), n))
            hkl_subsampled = np.take_along_axis(
                self.best_hkl[candidate_indices],
                subsampled_indices[:, :, np.newaxis],
                axis=1
                )
            q2_subsampled = np.take(self.q2_obs, subsampled_indices)
            target_function = CandidateOptLoss(
                q2_subsampled, 
                lattice_system=self.lattice_system,
                )
            target_function.update(hkl_subsampled, refined_xnn[candidate_indices])
            refined_xnn[candidate_indices] += target_function.gauss_newton_step(refined_xnn[candidate_indices])

        q2_ref_calc = self.q2_calculator.get_q2(refined_xnn)
        hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
        refined_hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
        if self.triplets is None:
            refined_M20 = get_M20_from_xnn(
                self.q2_obs, refined_xnn, refined_hkl, self.hkl_ref, self.lattice_system
                )
        else:
            refined_M20 = get_M20_triplet(
                self.q2_obs,
                self.triplets,
                refined_hkl,
                refined_xnn,
                self.lattice_system,
                self.bravais_lattice
                )

        update = refined_M20 > self.best_M20
        self.best_hkl[update] = refined_hkl[update]
        self.best_M20[update] = refined_M20[update]
        self.best_xnn[update] = refined_xnn[update]

    def correct_off_by_two(self):
        # These do a quick standardization of monoclinic and triclinic candidates. It is just a
        # Selling reduction.
        # This is performed in this function because the Miller indices are reassigned at the 
        # end. Miller index transformations are not tracked in monoclinic_standardization
        if self.lattice_system in ['monoclinic', 'triclinic']:
            best_unit_cell = get_unit_cell_from_xnn(
                self.best_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            if self.lattice_system == 'monoclinic':
                best_unit_cell = monoclinic_standardization(best_unit_cell, partial_unit_cell=True)
            elif self.lattice_system == 'triclinic':
                best_unit_cell, _, _ = selling_reduction(best_unit_cell)
            self.best_xnn = get_xnn_from_unit_cell(
                best_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )

        mult_factor = np.array([1/2, 1, 2, 3, 4])
        if self.lattice_system == 'cubic':
            mult_factors = mult_factor[:, np.newaxis]
        elif self.lattice_system in ['hexagonal', 'tetragonal']:
            mult_factors = np.ones((mult_factor.size**2, 2))
            mf_index = 0
            for mf0 in mult_factor:
                for mf1 in mult_factor:
                    mult_factors[mf_index, 0] = mf0
                    mult_factors[mf_index, 1] = mf1
                    mf_index += 1
        elif self.lattice_system == 'rhombohedral':
            mult_factors = np.ones((mult_factor.size, 2))
            mult_factors[:, 0] = mult_factor
            mult_factors[:, 1] = mult_factor
        elif self.lattice_system in ['orthorhombic', 'monoclinic', 'triclinic']:
            mult_factors = np.ones((mult_factor.size**3, self.xnn.shape[1]))
            mf_index = 0
            for mf0 in mult_factor:
                for mf1 in mult_factor:
                    for mf2 in mult_factor:
                        mult_factors[mf_index, 0] = mf0
                        mult_factors[mf_index, 1] = mf1
                        mult_factors[mf_index, 2] = mf2
                        if self.lattice_system == 'monoclinic':
                            mult_factors[mf_index, 3] = np.sqrt(mf0 * mf2)
                        elif self.lattice_system == 'triclinic':
                            mult_factors[mf_index, 3] = np.sqrt(mf1 * mf2)
                            mult_factors[mf_index, 4] = np.sqrt(mf0 * mf2)
                            mult_factors[mf_index, 5] = np.sqrt(mf0 * mf1)
                        mf_index += 1

        n_test = int(0.05 * self.n)
        test_indices = np.argsort(self.best_M20)[::-1][:n_test]
        M20 = np.zeros([n_test, mult_factors.shape[0]])
        hkl = np.zeros([n_test, mult_factors.shape[0], self.n_peaks, 3])
        for mf_index in range(mult_factors.shape[0]):
            xnn_mult = mult_factors[mf_index, :][np.newaxis]**2 * self.best_xnn[test_indices]
            q2_ref_calc_mult = self.q2_calculator.get_q2(xnn_mult)
            hkl_assign = fast_assign(self.q2_obs, q2_ref_calc_mult)
            hkl[:, mf_index] = np.take(self.hkl_ref, hkl_assign, axis=0)
            if self.triplets is None:
                M20[:, mf_index] = get_M20_from_xnn(
                    self.q2_obs, xnn_mult, hkl[:, mf_index], self.hkl_ref, self.lattice_system
                    )
            else:
                M20[:, mf_index] = get_M20_triplet(
                    self.q2_obs,
                    self.triplets,
                    hkl[:, mf_index],
                    xnn_mult,
                    self.lattice_system,
                    self.bravais_lattice
                    )

        best_index = np.argmax(M20, axis=1)
        final_mult_factor = np.take(mult_factors, best_index, axis=0)        
        self.best_xnn[test_indices] *= final_mult_factor**2
        self.best_M20[test_indices] = np.take_along_axis(M20, best_index[:, np.newaxis], axis=1)[:, 0]
        self.best_hkl[test_indices] = np.take_along_axis(hkl, best_index[:, np.newaxis, np.newaxis, np.newaxis], axis=1)[:, 0]

        # do quick reindexing to enforce constraints
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            self.best_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        reciprocal_unit_cell = reindex_entry_basic(
            reciprocal_unit_cell,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            space='reciprocal'
            )
        self.best_xnn = get_xnn_from_reciprocal_unit_cell(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )

    def assign_extinction_group(self):
        self.best_M20, self.best_spacegroup = get_extinction_group(
            xnn=self.best_xnn,
            q2_obs=self.q2_obs,
            triplets_obs=self.triplets,
            hkl_ref_bl=self.hkl_ref,
            bravais_lattice=self.bravais_lattice,
            lattice_system=self.lattice_system
            )

    def calculate_peaks_indexed(self):
        _, probability, _ = get_M20_likelihood_from_xnn(
            q2_obs=self.q2_obs,
            xnn=self.best_xnn,
            hkl=self.best_hkl,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            )
        self.n_indexed = np.sum(probability > 0.5, axis=1)


class OptimizerBase:
    def __init__(self, comm):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.n_ranks = self.comm.Get_size()
        self.lattice_system = self.comm.bcast(self.lattice_system, root=self.root)
        self.bravais_lattice = self.comm.bcast(self.bravais_lattice, root=self.root)
        self.opt_params = self.comm.bcast(self.opt_params, root=self.root)
        self.hkl_ref_length = self.comm.bcast(self.hkl_ref_length, root=self.root)
        if self.rank != self.root:
            self.hkl_ref = np.zeros(self.hkl_ref_length)
        self.hkl_ref = self.comm.bcast(self.hkl_ref, root=self.root)
        self.n_peaks = self.comm.bcast(self.n_peaks, root=self.root)

    def generate_candidates_common(self, xnn_rank):
        candidates = Candidates(
            q2_obs=self.q2_obs,
            triplets=self.triplets,
            xnn=xnn_rank,
            hkl_ref=self.hkl_ref,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            )
        return candidates

    def run_common(self, n_top_candidates):
        self.comm.Bcast(self.q2_obs, root=self.root)
        self.triplets = self.comm.bcast(self.triplets, root=self.root)
        candidates = self.generate_candidates_rank()
        for iteration_info in self.opt_params['iteration_info']:
            for iter_index in range(iteration_info['n_iterations']):
                candidates.random_subsampling(iteration_info)

        # This meant to be run at the end of optimization to remove very similar candidates
        # If this isn't run, the results will be spammed with many candidates that are nearly
        # identical.
        # This method takes pairwise differences in Xnn space and combines candidates that are 
        # closer than some given radius
        # If this were performed with all the entries combined, it would be slow and memory intensive.
        # Instead the candidates are sorted by reciprocal unit cell volume and filtering is
        # performed in chunks.

        # Check to see if a better M20 score can be found by multiplying the unit cell by 2 along
        # each axis. This also performs a quick reindexing.
        # Check which spacegroup gives the best M20 score.
        # Then calculate the number of assigned peaks (probability > 50%)
        candidates.refine_cell()
        candidates.correct_off_by_two()
        candidates.assign_extinction_group()
        candidates.calculate_peaks_indexed()
        if self.opt_params['convergence_testing']:
            self.convergence_testing(candidates)
        else:
            self.downsample_candidates(candidates, n_top_candidates)


class OptimizerWorker(OptimizerBase):
    def __init__(self, comm):
        self.root = 0
        self.lattice_system = None
        self.bravais_lattice = None
        self.opt_params = None
        self.hkl_ref = None
        self.n_peaks = None
        self.hkl_ref_length = None
        self.rng = np.random.default_rng()
        super().__init__(comm)
        
    def run(self, entry=None, q2=None, triplets=None, n_top_candidates=20):
        self.q2_obs = np.zeros(self.n_peaks)
        self.triplets = None
        self.run_common(n_top_candidates=None)

    def generate_candidates_rank(self):
        candidate_xnn_rank = self.comm.recv(source=self.root)
        return self.generate_candidates_common(candidate_xnn_rank)

    def downsample_candidates(self, candidates, n_top_candidates):
        self.comm.Send(candidates.best_M20, dest=self.root)
        self.comm.Send(candidates.best_xnn, dest=self.root)
        self.comm.Send(candidates.n_indexed, dest=self.root)
        self.comm.send(candidates.best_spacegroup, dest=self.root)

    def convergence_testing(self, candidates):
        self.comm.Send(candidates.best_M20, dest=self.root)
        self.comm.Send(candidates.best_xnn, dest=self.root)


class OptimizerManager(OptimizerBase):
    def __init__(self, data_params, opt_params, reg_params, template_params, pitf_params, random_params, bravais_lattice, comm, seed=12345):
        self.root = comm.Get_rank()
        assert self.root == 0
        self.data_params = data_params
        self.opt_params = opt_params
        self.reg_params = reg_params
        self.pitf_params = pitf_params
        self.random_params = random_params
        self.template_params = template_params
        self.bravais_lattice = bravais_lattice
        self.rng = np.random.default_rng(seed)

        opt_params_defaults = {
            'minimum_uc': 2,
            'maximum_uc': 500,
            }
        for key in opt_params_defaults.keys():
            if key not in self.opt_params.keys():
                self.opt_params[key] = opt_params_defaults[key]
        for key in self.reg_params:
            self.reg_params[key]['load_from_tag'] = True
            self.reg_params[key]['alpha_params'] = {}
            self.reg_params[key]['beta_params'] = {}
            self.reg_params[key]['mean_params'] = {}
            self.reg_params[key]['var_params'] = {}
        for key in self.pitf_params:
            self.pitf_params[key]['load_from_tag'] = True
        self.data_params['load_from_tag'] = True
        self.template_params[self.bravais_lattice]['load_from_tag'] = True
        self.random_params[self.bravais_lattice]['load_from_tag'] = True

        self.indexer = Indexing(
            data_params=self.data_params,
            reg_params=self.reg_params,
            template_params=self.template_params,
            pitf_params=self.pitf_params,
            random_params=self.random_params,
            seed=12345,
            )
        self.indexer.setup_from_tag(load_bravais_lattice=self.bravais_lattice)
        load_random_forest = False
        load_nn = False
        load_pitf = False
        for generator_info in self.opt_params['generator_info']:
            if generator_info['generator'] == 'trees':
                load_random_forest = True
            elif generator_info['generator'] == 'nn':
                load_nn = True
            elif generator_info['generator'] == 'pitf':
                load_pitf = True
        if load_nn or load_random_forest:
            # eventually add an option so the NN doesn't get loaded if it won't be used
            self.indexer.setup_regression()
        if load_pitf:
            self.indexer.setup_pitf()
        self.indexer.setup_miller_index_templates()
        self.indexer.setup_random()

        self.n_groups = len(self.indexer.data_params['split_groups'])
        self.lattice_system = self.indexer.data_params['lattice_system']
        self.hkl_ref = self.indexer.hkl_ref[self.bravais_lattice]
        self.hkl_ref_length = self.indexer.data_params['hkl_ref_length']
        self.n_peaks = self.indexer.data_params['n_peaks']
        self.unit_cell_length = self.indexer.data_params['unit_cell_length']
        super().__init__(comm)

    def run(self, entry=None, q2=None, triplets=None, n_top_candidates=20):
        if entry is None:
            self.q2_obs = q2[:self.n_peaks]
        elif q2 is None:
            self.q2_obs = np.array(entry['q2'])[:self.n_peaks]
            if self.opt_params['convergence_testing']:
                self.xnn_true = np.array(entry['reindexed_xnn'])[self.indexer.data_params['unit_cell_indices']]
        self.triplets = triplets
        self.run_common(n_top_candidates=n_top_candidates)

    def generate_candidates_rank(self):
        candidate_unit_cells_all = []
        if self.opt_params['convergence_testing']:
            size = (self.opt_params['convergence_candidates'], self.indexer.data_params['unit_cell_length'])
            convergence_initial_xnn = []
            for distance in self.opt_params['convergence_distances']:
                perturbations = self.rng.uniform(low=-1, high=1, size=size)
                perturbations = distance * perturbations / np.linalg.norm(perturbations, axis=1)[:, np.newaxis]
                perturbed_xnn = self.xnn_true[np.newaxis] + perturbations
                perturbed_xnn = fix_unphysical(
                    xnn=perturbed_xnn,
                    rng=self.rng,
                    minimum_unit_cell=self.opt_params['minimum_uc'],
                    maximum_unit_cell=self.opt_params['maximum_uc'],
                    lattice_system=self.lattice_system
                    )
                convergence_initial_xnn.append(perturbed_xnn)
                candidate_unit_cells_all.append(get_unit_cell_from_xnn(
                    perturbed_xnn,
                    partial_unit_cell=True,
                    lattice_system=self.lattice_system
                    ))
            self.convergence_initial_xnn = np.concatenate(convergence_initial_xnn, axis=0)
        else:
            for generator_info in self.opt_params['generator_info']:
                if generator_info['generator'] in ['nn', 'trees']:
                    generator_unit_cells = self.indexer.unit_cell_generator[generator_info['split_group']].generate(
                        generator_info['n_unit_cells'], self.rng, self.q2_obs,
                        batch_size=1,
                        model=generator_info['generator'],
                        q2_scaler=self.indexer.q2_scaler,
                        )
                elif generator_info['generator'] == 'templates':
                    generator_unit_cells = self.indexer.miller_index_templator[self.bravais_lattice].generate(
                        generator_info['n_unit_cells'], self.rng, self.q2_obs,
                        )
                elif generator_info['generator'] == 'pitf':
                    generator_unit_cells = self.indexer.pitf_generator[generator_info['split_group']].generate(
                        generator_info['n_unit_cells'], self.rng, self.q2_obs,
                        batch_size=1,
                        )
                elif generator_info['generator'] in ['random', 'distribution_volume', 'predicted_volume']:
                    generator_unit_cells = self.indexer.random_unit_cell_generator[self.bravais_lattice].generate(
                        generator_info['n_unit_cells'], self.rng, self.q2_obs,
                        model=generator_info['generator'],
                        )
                candidate_unit_cells_all.append(generator_unit_cells)
        candidate_unit_cells_all = np.concatenate(candidate_unit_cells_all, axis=0)

        candidate_unit_cells_all = fix_unphysical(
            unit_cell=candidate_unit_cells_all,
            rng=self.rng,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            lattice_system=self.lattice_system
            )
        candidate_unit_cells_all = reindex_entry_basic(
            candidate_unit_cells_all,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            space='direct'
            )
        candidate_xnn_all = get_xnn_from_unit_cell(
            candidate_unit_cells_all,
            partial_unit_cell=True,
            lattice_system=self.lattice_system
            )
        if self.opt_params['convergence_testing'] == False:
            candidate_xnn_all = self.redistribute_xnn(candidate_xnn_all)

        self.sent_candidates = np.zeros(self.n_ranks, dtype=int)
        for rank_index in range(self.n_ranks):
            self.sent_candidates[rank_index] = candidate_xnn_all[rank_index::self.n_ranks].shape[0]
            if rank_index == self.root:
                candidate_xnn_rank = candidate_xnn_all[rank_index::self.n_ranks]
            else:
                self.comm.send(candidate_xnn_all[rank_index::self.n_ranks], dest=rank_index)

        return self.generate_candidates_common(candidate_xnn_rank)

    def redistribute_xnn(self, xnn):
        # This function is meant to be called only once before optimization starts
        # exhaustive_search is meant to redistribute candidates during optimization
        redistributed_xnn = xnn.copy()
        n_redistributed = 0
        iteration = 0
        # Capping the number of iterations is arbitrary.
        # Just an attempt to prevent an excessively long loop
        largest_neighborhood = self.opt_params['max_neighbors'] + 1
        from_indices = None
        while largest_neighborhood > self.opt_params['max_neighbors'] and iteration < 20:
            # This initial distance calculation is time intensive.
            # After the first iteration, only calculate distances after they have been updated.
            if from_indices is None:
                distance = scipy.spatial.distance.cdist(redistributed_xnn, redistributed_xnn)
                neighbor_array = distance < self.opt_params['neighbor_radius']
            else:
                distance_0 = scipy.spatial.distance.cdist(redistributed_xnn[from_indices], redistributed_xnn)
                distance[from_indices, :] = distance_0
                distance[:, from_indices] = distance_0.T
                neighbor_array[from_indices, :] = distance[from_indices, :] < self.opt_params['neighbor_radius']
                neighbor_array[:, from_indices] = distance[:, from_indices] < self.opt_params['neighbor_radius']
            neighbor_count = np.sum(neighbor_array, axis=1)
            largest_neighborhood = neighbor_count.max()
            if largest_neighborhood > self.opt_params['max_neighbors']:
                # This gets the candidate that has the most nearest neighbors and redistributes
                # a subsample of its neighbors such that it has the correct amount of neighbors
                highest_density_index = np.argmax(neighbor_count)
                neighbor_indices = np.where(neighbor_array[highest_density_index])[0]
                excess_neighbors = neighbor_indices.size - self.opt_params['max_neighbors']
                from_indices = neighbor_indices[
                    self.rng.choice(neighbor_indices.size, size=excess_neighbors, replace=False)
                    ]
                n_redistributed += excess_neighbors

                # We want to redistribute the excess only to regions where the density is low
                # Find candidates that have fewer than the number of maximum neighbors and
                # redistribute excess to neighborhoods near these candidates
                low_density_indices = np.where(neighbor_count < self.opt_params['max_neighbors'])[0]
                if low_density_indices.size == 0:
                    ### !!! FIX THIS CASE
                    ### !!! What should be done if there are no low density regions
                    break
                # Bias the redistribution to the lowest density regions by probabalistly sampling
                # the low density regions.
                prob = self.opt_params['max_neighbors'] - neighbor_count[low_density_indices]
                prob = prob / prob.sum()
                if excess_neighbors <= low_density_indices.size:
                    replace = False
                else:
                    replace = True
                to_indices = low_density_indices[self.rng.choice(
                    low_density_indices.size, size=excess_neighbors, replace=replace, p=prob
                    )]
                redistributed_xnn = self.redistribute_and_perturb_xnn(
                    redistributed_xnn, from_indices, to_indices
                    )
            iteration += 1
        return redistributed_xnn

    def redistribute_and_perturb_xnn(self, xnn, from_indices, to_indices, norm_factor=None):
        n_indices = from_indices.size
        if not norm_factor is None:
            norm_factor = self.rng.uniform(low=norm_factor[0], high=norm_factor[1], size=n_indices)
        else:
            norm_factor = np.ones(n_indices)
        norm_factor *= self.opt_params['neighbor_radius']
        perturbation = self.rng.uniform(low=-1, high=1, size=(n_indices, self.unit_cell_length))
        perturbation *= (norm_factor / np.linalg.norm(perturbation, axis=1))[:, np.newaxis]
        xnn[from_indices] = xnn[to_indices] + perturbation
        xnn[from_indices] = fix_unphysical(
            xnn=xnn[from_indices],
            rng=self.rng,
            minimum_unit_cell=self.opt_params['minimum_uc'], 
            maximum_unit_cell=self.opt_params['maximum_uc'],
            lattice_system=self.lattice_system
            )

        # Enforce the constraints on the unit cells by reindexing
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        # This reindexing is time intensive. Only reindex entries that were updated.
        reciprocal_unit_cell[from_indices] = reindex_entry_basic(
            reciprocal_unit_cell[from_indices],
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            space='reciprocal'
            )
        xnn = get_xnn_from_reciprocal_unit_cell(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return xnn        

    def downsample_candidates(self, candidates, n_top_candidates):
        best_M20_all = []
        best_xnn_all = []
        best_n_indexed_all = []
        best_spacegroup_all = []
        for rank_index in range(self.n_ranks):
            if rank_index == self.root:
                best_M20_all.append(candidates.best_M20)
                best_xnn_all.append(candidates.best_xnn)
                best_n_indexed_all.append(candidates.n_indexed)
                best_spacegroup_all += candidates.best_spacegroup
            else:
                best_M20_rank = np.zeros(self.sent_candidates[rank_index])
                best_xnn_rank = np.zeros((self.sent_candidates[rank_index], self.unit_cell_length))
                best_n_indexed_rank = np.zeros(self.sent_candidates[rank_index])
                self.comm.Recv(best_M20_rank, source=rank_index)
                self.comm.Recv(best_xnn_rank, source=rank_index)
                self.comm.Recv(best_n_indexed_rank, source=rank_index)
                best_spacegroup_rank = self.comm.recv(source=rank_index)
                best_M20_all.append(best_M20_rank)
                best_xnn_all.append(best_xnn_rank)
                best_n_indexed_all.append(best_n_indexed_rank)
                best_spacegroup_all += best_spacegroup_rank
        best_M20_all = np.concatenate(best_M20_all, axis=0)
        best_xnn_all = np.concatenate(best_xnn_all, axis=0)
        best_n_indexed_all = np.concatenate(best_n_indexed_all, axis=0)

        # Next remove nearly identical xnn's by selecting the xnn within an arbitrary radius
        # with the highest M20 score. The candidates are sorted by reciprocal volume so the
        # pairwise comparisons can be made within 'chunks' instead of over all candidates.
        reciprocal_volume = get_unit_cell_volume(get_reciprocal_unit_cell_from_xnn(
            best_xnn_all, partial_unit_cell=True, lattice_system=self.lattice_system
            ), partial_unit_cell=True, lattice_system=self.lattice_system)
        sort_indices = np.argsort(reciprocal_volume)

        best_xnn_all = best_xnn_all[sort_indices]
        best_M20_all = best_M20_all[sort_indices]
        best_n_indexed_all = best_n_indexed_all[sort_indices]
        best_spacegroup_all = [best_spacegroup_all[i] for i in sort_indices]
        chunk_size = 1000
        n_chunks = best_xnn_all.shape[0] // chunk_size + 1

        radius = self.opt_params['neighbor_radius'] / 10
        xnn_downsampled = []
        M20_downsampled = []
        n_indexed_downsampled = []
        spacegroup_downsampled = []
        for chunk_index in range(n_chunks):
            if chunk_index == n_chunks - 1:
                xnn_chunk = best_xnn_all[chunk_index * chunk_size:]
                M20_chunk = best_M20_all[chunk_index * chunk_size:]
                n_indexed_chunk = best_n_indexed_all[chunk_index * chunk_size:]
                spacegroup_chunk = best_spacegroup_all[chunk_index * chunk_size:]
            else:
                xnn_chunk = best_xnn_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                M20_chunk = best_M20_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                n_indexed_chunk = best_n_indexed_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                spacegroup_chunk = best_spacegroup_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
            status = True
            while status:
                distance = scipy.spatial.distance.cdist(xnn_chunk, xnn_chunk)
                neighbor_array = distance < radius
                neighbor_count = np.sum(neighbor_array, axis=1)
                if neighbor_count.size > 0 and neighbor_count.max() > 1:
                    highest_density_index = np.argmax(neighbor_count)
                    neighbor_indices = np.where(neighbor_array[highest_density_index])[0]
                    best_neighbor = np.argmax(M20_chunk[neighbor_indices])
                    xnn_best_neighbor = xnn_chunk[neighbor_indices][best_neighbor]
                    M20_best_neighbor = M20_chunk[neighbor_indices][best_neighbor]
                    n_indexed_best_neighbor = n_indexed_chunk[neighbor_indices][best_neighbor]
                    #spacegroup_best_neighbor = spacegroup_chunk[neighbor_indices][best_neighbor]
                    spacegroup_best_neighbor = [spacegroup_chunk[i] for i in neighbor_indices][best_neighbor]
                    xnn_chunk = np.row_stack((
                        np.delete(xnn_chunk, neighbor_indices, axis=0), 
                        xnn_best_neighbor
                        ))
                    M20_chunk = np.concatenate((
                        np.delete(M20_chunk, neighbor_indices), 
                        [M20_best_neighbor]
                        ))
                    n_indexed_chunk = np.concatenate((
                        np.delete(n_indexed_chunk, neighbor_indices), 
                        [n_indexed_best_neighbor]
                        ))
                    # neighbor indices are sorted in increasing order and must be reversed
                    # for this pop to remove them correctly.
                    for i in neighbor_indices[::-1]:
                        spacegroup_chunk.pop(i)
                    spacegroup_chunk += [spacegroup_best_neighbor]
                else:
                    status = False
            xnn_downsampled.append(xnn_chunk)
            M20_downsampled.append(M20_chunk)
            n_indexed_downsampled.append(n_indexed_chunk)
            spacegroup_downsampled += spacegroup_chunk
        xnn_downsampled = np.row_stack(xnn_downsampled)
        M20_downsampled = np.concatenate(M20_downsampled)
        n_indexed_downsampled = np.concatenate(n_indexed_downsampled)

        sort_indices = np.argsort(M20_downsampled)[::-1][:n_top_candidates]
        self.top_xnn = xnn_downsampled[sort_indices]
        self.top_M20 = M20_downsampled[sort_indices]
        self.top_n_indexed = n_indexed_downsampled[sort_indices]
        self.top_spacegroup = [spacegroup_downsampled[i] for i in sort_indices]
        self.top_unit_cell = get_unit_cell_from_xnn(
            self.top_xnn,
            partial_unit_cell=True,
            lattice_system=self.lattice_system,
            )

    def convergence_testing(self, candidates):
        n_candidates = self.opt_params['convergence_candidates'] * len(self.opt_params['convergence_distances'])
        self.top_M20 = np.zeros(n_candidates)
        self.top_xnn = np.zeros((n_candidates, self.indexer.data_params['unit_cell_length']))
        for rank_index in range(self.n_ranks):
            if rank_index == self.root:
                self.top_M20[rank_index::self.n_ranks] = candidates.best_M20
                self.top_xnn[rank_index::self.n_ranks] = candidates.best_xnn
            else:
                best_M20_rank = np.zeros(self.sent_candidates[rank_index])
                best_xnn_rank = np.zeros((self.sent_candidates[rank_index], self.unit_cell_length))
                self.comm.Recv(best_M20_rank, source=rank_index)
                self.comm.Recv(best_xnn_rank, source=rank_index)
                self.top_M20[rank_index::self.n_ranks] = best_M20_rank
                self.top_xnn[rank_index::self.n_ranks] = best_xnn_rank
        self.top_unit_cell = get_unit_cell_from_xnn(
            self.top_xnn,
            partial_unit_cell=True,
            lattice_system=self.lattice_system,
            )
