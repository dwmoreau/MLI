import numpy as np
import scipy.spatial

from mlindex.model_training.Wrapper import Wrapper
from mlindex.optimization.Candidates import Candidates
from mlindex.utilities.ErrorAdder import perturb_xnn
from mlindex.utilities.Reindexing import reindex_entry_basic
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_xnn_from_reciprocal_unit_cell
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume


class OptimizerBase:
    def __init__(self, comm, fom):
        self.comm = comm
        self.fom = fom
        self.rank = self.comm.Get_rank()
        self.n_ranks = self.comm.Get_size()
        self.lattice_system = self.comm.bcast(self.lattice_system, root=self.root)
        self.bravais_lattice = self.comm.bcast(self.bravais_lattice, root=self.root)
        self.opt_params = self.comm.bcast(self.opt_params, root=self.root)
        self.hkl_ref_length = self.comm.bcast(self.hkl_ref_length, root=self.root)
        if self.rank != self.root:
            self.hkl_ref = np.zeros((self.hkl_ref_length, 3))
        self.comm.Bcast(self.hkl_ref, root=self.root)
        self.n_peaks = self.comm.bcast(self.n_peaks, root=self.root)
        self.zero_error = False
        self.wavelength = None

    def generate_candidates_common(self, xnn_rank):
        candidates = Candidates(
            q2_obs=self.q2_obs,
            triplets=self.triplets,
            xnn=xnn_rank,
            hkl_ref=self.hkl_ref,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            opt_params=self.opt_params,
            rng=self.rng,
            fom=self.fom,
            zero_error=self.zero_error,
            wavelength=self.wavelength,
            )
        return candidates

    def _run_loop(self, n_top_candidates):
        candidates = self.generate_candidates_rank()
        if self.opt_params['redistribution_testing']:
            return None

        for iteration_info in self.opt_params['iteration_info']:
            for iter_index in range(iteration_info['n_iterations']):
                if iteration_info['worker'] == 'random_subsampling':
                    candidates.random_subsampling(iteration_info)
                elif iteration_info['worker'] == 'random_subsampling_power':
                    candidates.random_subsampling_power(iteration_info)
                elif iteration_info['worker'] == 'random_power':
                    candidates.random_power(iteration_info)
                elif iteration_info['worker'] == 'deterministic':
                    candidates.deterministic(iteration_info)

        # This meant to be run at the end of optimization to remove very similar candidates
        # If this isn't run, the results will be spammed with many candidates that are nearly
        # identical.
        # This method takes pairwise differences in Xnn space and combines candidates that are
        # closer than some given radius
        # If this were performed with all the entries, it would be slow and memory intensive.
        # Instead the candidates are sorted by reciprocal unit cell volume and filtering is
        # performed in chunks.

        # Check to see if a better M20 score can be found by multiplying the unit cell by 2 along
        # each axis. This also performs a quick reindexing.
        # Check which spacegroup gives the best M20 score.
        # Then calculate the number of assigned peaks (probability > 50%)
        candidates.refine_cell()
        candidates.standardize_cell()
        candidates.correct_off_by_two()
        candidates.assign_extinction_group()
        candidates.calculate_peaks_indexed()
        if self.opt_params['convergence_testing']:
            self.convergence_testing(candidates)
        else:
            self.downsample_candidates(candidates, n_top_candidates)

    def run_common(self, n_top_candidates):
        self.comm.Bcast(self.q2_obs, root=self.root)
        self.triplets = self.comm.bcast(self.triplets, root=self.root)
        self._run_loop(n_top_candidates)


class OptimizerWorker(OptimizerBase):
    def __init__(self, comm, fom, seed=12345):
        self.root = 0
        self.lattice_system = None
        self.bravais_lattice = None
        self.opt_params = None
        self.hkl_ref = None
        self.n_peaks = None
        self.hkl_ref_length = None
        self.rng = np.random.default_rng(seed)
        super().__init__(comm, fom)

    def run(self, entry=None, q2=None, triplets=None, n_top_candidates=20, zero_error=False, wavelength=None):
        self.q2_obs = np.zeros(self.n_peaks)
        self.triplets = None
        self.zero_error = zero_error
        self.wavelength = wavelength
        self.run_common(n_top_candidates=n_top_candidates)

    def generate_candidates_rank(self):
        candidate_xnn_rank = self.comm.recv(source=self.root)
        return self.generate_candidates_common(candidate_xnn_rank)

    def downsample_candidates(self, candidates, n_top_candidates):
        self.comm.Send(candidates.best_M20, dest=self.root)
        self.comm.Send(candidates.best_Minfo, dest=self.root)
        self.comm.Send(candidates.best_xnn, dest=self.root)
        self.comm.Send(candidates.n_indexed, dest=self.root)
        self.comm.send(candidates.best_spacegroup, dest=self.root)
        if not self.triplets is None:
            self.comm.Send(candidates.n_indexed_triplets, dest=self.root)
            self.comm.Send(candidates.best_M_triplets, dest=self.root)

    def convergence_testing(self, candidates):
        self.comm.Send(candidates.best_M20, dest=self.root)
        self.comm.Send(candidates.best_xnn, dest=self.root)


class OptimizerManager(OptimizerBase):
    def __init__(self, data_params, opt_params, rf_params, template_params, integral_filter_params, random_params, bravais_lattice, comm, fom, seed=12345):
        self.root = comm.Get_rank()
        assert self.root == 0
        self.data_params = data_params
        self.opt_params = opt_params
        self.rf_params = rf_params
        self.integral_filter_params = integral_filter_params
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
        for key in self.rf_params:
            self.rf_params[key]['load_from_tag'] = True
        for key in self.integral_filter_params:
            self.integral_filter_params[key]['load_from_tag'] = True
        self.data_params['load_from_tag'] = True
        self.template_params[self.bravais_lattice]['load_from_tag'] = True
        self.random_params[self.bravais_lattice]['load_from_tag'] = True

        self.wrapper = Wrapper(
            data_params=self.data_params,
            rf_params=self.rf_params,
            template_params=self.template_params,
            integral_filter_params=self.integral_filter_params,
            random_params=self.random_params,
            seed=12345,
            )
        self.wrapper.setup_from_tag(load_bravais_lattice=self.bravais_lattice)
        if self.opt_params['convergence_testing'] == False:
            load_random_forest = False
            load_integral_filter = False
            load_templates = False
            load_random = False
            for generator_info in self.opt_params['generator_info']:
                if generator_info['generator'] == 'trees':
                    load_random_forest = True
                elif generator_info['generator'] == 'integral_filter':
                    load_integral_filter = True
                elif generator_info['generator'] == 'templates':
                    load_templates = True
                elif generator_info['generator'] in ['predicted_volume', 'random']:
                    load_random = True
            if load_random_forest:
                self.wrapper.setup_random_forest()
            if load_integral_filter:
                self.wrapper.setup_integral_filter(mode='inference')
            if load_templates:
                self.wrapper.setup_miller_index_templates()
            if load_random:
                self.wrapper.setup_random()

        self.n_groups = len(self.wrapper.data_params['split_groups'])
        self.lattice_system = self.wrapper.data_params['lattice_system']
        self.hkl_ref = self.wrapper.hkl_ref[self.bravais_lattice]
        self.hkl_ref_length = self.wrapper.data_params['hkl_ref_length']

        self.n_peaks = self.wrapper.data_params['n_peaks']
        self.unit_cell_length = self.wrapper.data_params['unit_cell_length']
        super().__init__(comm, fom)

    def run(self, entry=None, q2=None, triplets=None, n_top_candidates=20, zero_error=False, wavelength=None):
        if (entry is None) and (not q2 is None):
            self.q2_obs = q2[:self.n_peaks]
        elif (not entry is None) and (q2 is None):
            self.q2_obs = np.array(entry['q2'])[:self.n_peaks]
            if self.opt_params['convergence_testing'] or self.opt_params['redistribution_testing']:
                self.xnn_true = np.array(entry['reindexed_xnn'])[self.wrapper.data_params['unit_cell_indices']]
        self.triplets = triplets
        self.zero_error = zero_error
        self.wavelength = wavelength
        if not self.triplets is None:
            good_indices = np.all(np.column_stack((
                self.triplets[:, 0] < self.n_peaks,
                self.triplets[:, 1] < self.n_peaks,
                )),
                axis=1
                )
            self.triplets = self.triplets[good_indices]
        self.run_common(n_top_candidates=n_top_candidates)
        if self.opt_params['redistribution_testing']:
            return self.opt_params['max_neighbors'], self.opt_params['neighbor_radius']

    def perform_predictions(self, q2, split_group, top_n=1):
        template_unit_cells = None
        tree_unit_cells = None
        volume_pred = None
        xnn_pred = None
        for generator_info in self.opt_params['generator_info']:
            if generator_info['generator'] == 'integral_filter':
                if generator_info['split_group'] == split_group:
                    xnn_pred, prob = self.wrapper.integral_filter_generator[split_group].predict_xnn(
                        top_n, q2_obs=q2[np.newaxis], batch_size=2
                        )
            elif generator_info['generator'] == 'templates':
                template_unit_cells = self.wrapper.miller_index_templator[self.bravais_lattice].generate(
                    top_n, self.rng, q2,
                    )
            elif generator_info['generator'] == 'predicted_volume':
                rec_volume_pred = self.wrapper.random_unit_cell_generator[self.bravais_lattice].random_forest_regressor.predict_individual_trees(
                    q2[np.newaxis],
                    n_outputs=1
                    )[0]
                volume_pred = 1 / rec_volume_pred
            elif generator_info['generator'] == 'trees':
                tree_unit_cells = self.wrapper.random_forest_generator[split_group].generate(
                    generator_info['n_unit_cells'], self.rng, q2,
                    )

        return xnn_pred, template_unit_cells, volume_pred, tree_unit_cells

    def _generate_candidates_xnn(self):
        if self.opt_params['convergence_testing']:
            candidate_unit_cells_all = perturb_xnn(
                self.xnn_true,
                convergence_candidates=self.opt_params['convergence_candidates'],
                convergence_distances=self.opt_params['convergence_distances'],
                minimum_uc=self.opt_params['minimum_uc'],
                maximum_uc=self.opt_params['maximum_uc'],
                lattice_system=self.lattice_system,
                rng=self.rng
            )
        else:
            candidate_unit_cells_all = []
            for generator_info in self.opt_params['generator_info']:
                if generator_info['generator'] == 'trees':
                    generator_unit_cells = self.wrapper.random_forest_generator[generator_info['split_group']].generate(
                        generator_info['n_unit_cells'], self.rng, self.q2_obs,
                        )
                elif generator_info['generator'] == 'templates':
                    generator_unit_cells = self.wrapper.miller_index_templator[self.bravais_lattice].generate(
                        generator_info['n_unit_cells'], self.rng, self.q2_obs,
                        )
                elif generator_info['generator'] == 'integral_filter':
                    # We only do one inference, so batch_size=total_size=1 makes sense
                    # but batch size of 2 is faster than one ....
                    generator_unit_cells = self.wrapper.integral_filter_generator[generator_info['split_group']].generate(
                        generator_info['n_unit_cells'], self.rng, self.q2_obs,
                        batch_size=2,
                        )
                elif generator_info['generator'] in ['random', 'predicted_volume']:
                    generator_unit_cells = self.wrapper.random_unit_cell_generator[self.bravais_lattice].generate(
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

        if self.opt_params['redistribution_testing']:
            self.redistrubution_testing(candidate_xnn_all)
        elif self.opt_params['convergence_testing'] == False:
            candidate_xnn_all = self.redistribute_xnn(candidate_xnn_all)

        return candidate_xnn_all

    def generate_candidates_rank(self):
        candidate_xnn_all = self._generate_candidates_xnn()
        self.sent_candidates = np.zeros(self.n_ranks, dtype=int)
        for rank_index in range(self.n_ranks):
            self.sent_candidates[rank_index] = candidate_xnn_all[rank_index::self.n_ranks].shape[0]
            if rank_index == self.root:
                candidate_xnn_rank = candidate_xnn_all[rank_index::self.n_ranks]
            else:
                self.comm.send(candidate_xnn_all[rank_index::self.n_ranks], dest=rank_index)
        return self.generate_candidates_common(candidate_xnn_rank)

    def _redistribution_testing_functional(self, neighbor_radius, xnn, x, N_success):
        self.opt_params['neighbor_radius'] = neighbor_radius
        redistributed_xnn = self.redistribute_xnn(xnn)
        distance = np.linalg.norm(redistributed_xnn - self.xnn_true[np.newaxis], axis=1)
        bins = np.concatenate([[0], x])
        distance_hist, _ = np.histogram(distance, bins=bins)
        N = np.cumsum(distance_hist)
        in_range = N_success != np.inf
        F = (N[in_range] - N_success[in_range]) / N_success[in_range]
        term_0 = 0
        if np.max(F) < 0:
            term_0 += 100
        term_1 = -np.mean(
            np.trapezoid(F, x[in_range]) / np.trapezoid(x[in_range])
            )
        return term_0 + term_1

    def redistrubution_testing(self, xnn):
        import scipy.optimize
        # Steps:
        #   Perform a grid search where max_neighbors is a prespecified grid. At each point, do
        #   an optimization for the best neighbor_radius
        opt_neighbor_radius = np.zeros(len(self.opt_params['max_neighbors_grid']))
        objective_function = np.zeros(len(self.opt_params['max_neighbors_grid']))
        convergence_radius = self.opt_params['convergence_radius'][self.bravais_lattice]
        x = convergence_radius[0]
        success_rate = convergence_radius[1]
        N_success = 1/success_rate
        in_range = success_rate > 0.01
        N_success[~in_range] = np.inf
        for index, max_neighors in enumerate(self.opt_params['max_neighbors_grid']):
            self.opt_params['max_neighbors'] = max_neighors
            opt_results = scipy.optimize.minimize_scalar(
                fun=self._redistribution_testing_functional,
                bounds=[0, 0.001],
                args=(xnn, x, N_success)
                )
            opt_neighbor_radius[index] = opt_results.x
            objective_function[index] = opt_results.fun
        best_index = np.argmin(objective_function)
        self.opt_params['max_neighbors'] = self.opt_params['max_neighbors_grid'][best_index]
        self.opt_params['neighbor_radius'] = opt_neighbor_radius[best_index]

    def redistribute_xnn(self, xnn):
        # This function is meant to be called only once before optimization starts
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
                if low_density_indices.size > 0:
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
                    norm_factor = 1
                else:
                    # In the case that there are no low density regions, perturb by selecting the
                    # lowest density indices, then perturb by a larger amount.
                    to_indices = np.argsort(neighbor_count)[:excess_neighbors]
                    norm_factor = 2
                redistributed_xnn = self.redistribute_and_perturb_xnn(
                    redistributed_xnn, from_indices, to_indices, norm_factor
                    )
            iteration += 1
        return redistributed_xnn

    def redistribute_and_perturb_xnn(self, xnn, from_indices, to_indices, norm_factor):
        n_indices = from_indices.size
        perturbation = self.rng.uniform(low=-1, high=1, size=(n_indices, self.unit_cell_length))
        perturbation *= (
            norm_factor*self.opt_params['neighbor_radius'] / np.linalg.norm(perturbation, axis=1)
            )[:, np.newaxis]
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

    def _downsample_computation(self, best_M20_all, best_Minfo_all, best_xnn_all,
                                best_n_indexed_all, best_spacegroup_all,
                                best_n_indexed_triplets_all, best_M_triplets_all,
                                n_top_candidates):
        best_M20_all = np.concatenate(best_M20_all, axis=0)
        best_Minfo_all = np.concatenate(best_Minfo_all, axis=0)
        best_xnn_all = np.concatenate(best_xnn_all, axis=0)
        best_n_indexed_all = np.concatenate(best_n_indexed_all, axis=0)
        if not self.triplets is None:
            best_n_indexed_triplets_all = np.concatenate(best_n_indexed_triplets_all, axis=0)
            best_M_triplets_all = np.concatenate(best_M_triplets_all, axis=0)

        # Remove any candidates with np.nan as a unit cell.
        # I believe these are caused by numerical issues with triclinic unit cells during the
        # Selling reduction
        good_indices = np.invert(np.any(np.isnan(best_xnn_all), axis=1))
        best_M20_all = best_M20_all[good_indices]
        best_Minfo_all = best_Minfo_all[good_indices]
        best_xnn_all = best_xnn_all[good_indices]
        best_n_indexed_all = best_n_indexed_all[good_indices]
        if not self.triplets is None:
            best_n_indexed_triplets_all = best_n_indexed_triplets_all[good_indices]
            best_M_triplets_all = best_M_triplets_all[good_indices]

        # Next remove nearly identical xnn's by selecting the xnn within an arbitrary radius
        # with the highest M20 score. The candidates are sorted by reciprocal volume so the
        # pairwise comparisons can be made within 'chunks' instead of over all candidates.
        reciprocal_volume = get_unit_cell_volume(get_reciprocal_unit_cell_from_xnn(
            best_xnn_all, partial_unit_cell=True, lattice_system=self.lattice_system
            ), partial_unit_cell=True, lattice_system=self.lattice_system)
        sort_indices = np.argsort(reciprocal_volume)

        best_xnn_all = best_xnn_all[sort_indices]
        best_M20_all = best_M20_all[sort_indices]
        best_Minfo_all = best_Minfo_all[sort_indices]
        best_n_indexed_all = best_n_indexed_all[sort_indices]
        best_spacegroup_all = [best_spacegroup_all[i] for i in sort_indices]
        if not self.triplets is None:
            best_n_indexed_triplets_all = best_n_indexed_triplets_all[sort_indices]
            best_M_triplets_all = best_M_triplets_all[sort_indices]
        chunk_size = 1000
        n_chunks = best_xnn_all.shape[0] // chunk_size + 1

        xnn_downsampled = []
        M20_downsampled = []
        Minfo_downsampled = []
        n_indexed_downsampled = []
        if not self.triplets is None:
            n_indexed_triplets_downsampled = []
            M_triplets_downsampled = []
        spacegroup_downsampled = []
        for chunk_index in range(n_chunks):
            if chunk_index == n_chunks - 1:
                xnn_chunk = best_xnn_all[chunk_index * chunk_size:]
                M20_chunk = best_M20_all[chunk_index * chunk_size:]
                Minfo_chunk = best_Minfo_all[chunk_index * chunk_size:]
                n_indexed_chunk = best_n_indexed_all[chunk_index * chunk_size:]
                spacegroup_chunk = best_spacegroup_all[chunk_index * chunk_size:]
                if not self.triplets is None:
                    n_indexed_triplets_chunk = best_n_indexed_triplets_all[chunk_index * chunk_size:]
                    M_triplets_chunk = best_M_triplets_all[chunk_index * chunk_size:]
            else:
                xnn_chunk = best_xnn_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                M20_chunk = best_M20_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                Minfo_chunk = best_Minfo_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                n_indexed_chunk = best_n_indexed_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                spacegroup_chunk = best_spacegroup_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                if not self.triplets is None:
                    n_indexed_triplets_chunk = best_n_indexed_triplets_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                    M_triplets_chunk = best_M_triplets_all[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
            status = True
            while status:
                distance = scipy.spatial.distance.cdist(xnn_chunk, xnn_chunk)
                neighbor_array = distance < self.opt_params['downsample_radius']
                neighbor_count = np.sum(neighbor_array, axis=1)
                if neighbor_count.size > 0 and neighbor_count.max() > 1:
                    highest_density_index = np.argmax(neighbor_count)
                    neighbor_indices = np.where(neighbor_array[highest_density_index])[0]
                    if self.triplets is None:
                        best_neighbor = np.argmax(M20_chunk[neighbor_indices])
                    else:
                        best_neighbor = np.argmax(
                            np.sum(M_triplets_chunk[neighbor_indices], axis=1)
                            )
                    xnn_best_neighbor = xnn_chunk[neighbor_indices][best_neighbor]
                    M20_best_neighbor = M20_chunk[neighbor_indices][best_neighbor]
                    Minfo_best_neighbor = Minfo_chunk[neighbor_indices][best_neighbor]
                    n_indexed_best_neighbor = n_indexed_chunk[neighbor_indices][best_neighbor]
                    spacegroup_best_neighbor = [spacegroup_chunk[i] for i in neighbor_indices][best_neighbor]
                    if not self.triplets is None:
                        n_indexed_triplets_best_neighbor = n_indexed_triplets_chunk[neighbor_indices][best_neighbor]
                        M_triplets_best_neighbor = M_triplets_chunk[neighbor_indices][best_neighbor]
                    xnn_chunk = np.row_stack((
                        np.delete(xnn_chunk, neighbor_indices, axis=0),
                        xnn_best_neighbor
                        ))
                    M20_chunk = np.concatenate((
                        np.delete(M20_chunk, neighbor_indices),
                        [M20_best_neighbor]
                        ))
                    Minfo_chunk = np.concatenate((
                        np.delete(Minfo_chunk, neighbor_indices),
                        [Minfo_best_neighbor]
                        ))
                    n_indexed_chunk = np.concatenate((
                        np.delete(n_indexed_chunk, neighbor_indices),
                        [n_indexed_best_neighbor]
                        ))
                    if not self.triplets is None:
                        n_indexed_triplets_chunk = np.concatenate((
                            np.delete(n_indexed_triplets_chunk, neighbor_indices),
                            [n_indexed_triplets_best_neighbor]
                            ))
                        M_triplets_chunk = np.row_stack((
                            np.delete(M_triplets_chunk, neighbor_indices, axis=0),
                            M_triplets_best_neighbor
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
            Minfo_downsampled.append(Minfo_chunk)
            n_indexed_downsampled.append(n_indexed_chunk)
            if not self.triplets is None:
                n_indexed_triplets_downsampled.append(n_indexed_triplets_chunk)
                M_triplets_downsampled.append(M_triplets_chunk)
            spacegroup_downsampled += spacegroup_chunk
        xnn_downsampled = np.row_stack(xnn_downsampled)
        M20_downsampled = np.concatenate(M20_downsampled)
        Minfo_downsampled = np.concatenate(Minfo_downsampled)
        n_indexed_downsampled = np.concatenate(n_indexed_downsampled)
        if not self.triplets is None:
            n_indexed_triplets_downsampled = np.concatenate(n_indexed_triplets_downsampled)
            M_triplets_downsampled = np.row_stack(M_triplets_downsampled)

        if self.triplets is None:
            sort_indices = np.argsort(M20_downsampled)[::-1][:n_top_candidates]
        else:
            sort_indices = np.argsort(
                np.sum(M_triplets_downsampled, axis=1)
                )[::-1][:n_top_candidates]
        self.top_xnn = xnn_downsampled[sort_indices]
        self.top_M20 = M20_downsampled[sort_indices]
        self.top_Minfo = Minfo_downsampled[sort_indices]
        self.top_n_indexed = n_indexed_downsampled[sort_indices]
        if not self.triplets is None:
            self.top_n_indexed_triplets = n_indexed_triplets_downsampled[sort_indices]
            self.top_M_triplets = M_triplets_downsampled[sort_indices]
        self.top_spacegroup = [spacegroup_downsampled[i] for i in sort_indices]
        self.top_unit_cell = get_unit_cell_from_xnn(
            self.top_xnn,
            partial_unit_cell=True,
            lattice_system=self.lattice_system,
            )

    def downsample_candidates(self, candidates, n_top_candidates):
        best_M20_all = []
        best_Minfo_all = []
        best_xnn_all = []
        best_n_indexed_all = []
        best_spacegroup_all = []
        if not self.triplets is None:
            best_n_indexed_triplets_all = []
            best_M_triplets_all = []
        else:
            best_n_indexed_triplets_all = None
            best_M_triplets_all = None
        for rank_index in range(self.n_ranks):
            if rank_index == self.root:
                best_M20_all.append(candidates.best_M20)
                best_Minfo_all.append(candidates.best_Minfo)
                best_xnn_all.append(candidates.best_xnn)
                best_n_indexed_all.append(candidates.n_indexed)
                best_spacegroup_all += candidates.best_spacegroup
                if not self.triplets is None:
                    best_n_indexed_triplets_all.append(candidates.n_indexed_triplets)
                    best_M_triplets_all.append(candidates.best_M_triplets)
            else:
                best_M20_rank = np.zeros(self.sent_candidates[rank_index])
                best_Minfo_rank = np.zeros(self.sent_candidates[rank_index])
                best_xnn_rank = np.zeros((self.sent_candidates[rank_index], self.unit_cell_length))
                best_n_indexed_rank = np.zeros(self.sent_candidates[rank_index], dtype=int)
                self.comm.Recv(best_M20_rank, source=rank_index)
                self.comm.Recv(best_Minfo_rank, source=rank_index)
                self.comm.Recv(best_xnn_rank, source=rank_index)
                self.comm.Recv(best_n_indexed_rank, source=rank_index)
                best_spacegroup_rank = self.comm.recv(source=rank_index)
                if not self.triplets is None:
                    best_n_indexed_triplets_rank = np.zeros(self.sent_candidates[rank_index], dtype=int)
                    self.comm.Recv(best_n_indexed_triplets_rank, source=rank_index)
                    best_n_indexed_triplets_all.append(best_n_indexed_triplets_rank)

                    best_M_triplets_rank = np.zeros((self.sent_candidates[rank_index], 2))
                    self.comm.Recv(best_M_triplets_rank, source=rank_index)
                    best_M_triplets_all.append(best_M_triplets_rank)

                best_M20_all.append(best_M20_rank)
                best_Minfo_all.append(best_Minfo_rank)
                best_xnn_all.append(best_xnn_rank)
                best_n_indexed_all.append(best_n_indexed_rank)
                best_spacegroup_all += best_spacegroup_rank

        self._downsample_computation(best_M20_all, best_Minfo_all, best_xnn_all,
                                     best_n_indexed_all, best_spacegroup_all,
                                     best_n_indexed_triplets_all, best_M_triplets_all,
                                     n_top_candidates)

    def convergence_testing(self, candidates):
        n_candidates = self.opt_params['convergence_candidates'] * len(self.opt_params['convergence_distances'])
        self.top_M20 = np.zeros(n_candidates)
        self.top_xnn = np.zeros((n_candidates, self.wrapper.data_params['unit_cell_length']))
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
