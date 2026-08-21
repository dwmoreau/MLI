from concurrent.futures import ThreadPoolExecutor
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


def _downsample_chunk(args):
    """Collapse each dense neighbourhood in a chunk to its highest-M20 member.

    The distance matrix is computed once and then maintained, rather than
    rebuilt after every collapse. Collapsing removes points and re-appends one
    of the removed points, so the pairwise distances among the survivors are
    unchanged -- recomputing them produced the same numbers ~110 times per
    chunk. Measured 9-22x on captured chunks with bit-identical output
    (tools/repro_downsample.py). redistribute_xnn below already avoided the
    same rebuild for the same reason.

    ``order`` holds original row indices in their current positions. That is
    what keeps this bit-identical rather than merely equivalent: np.argmax
    returns the *first* maximum, so both the densest-point choice and the
    best-neighbour choice depend on the current ordering, and the collapse
    permutes it in a specific way (survivors keep their relative order, the
    kept point moves to the end).
    """
    (xnn_chunk, M20_chunk, Minfo_chunk, n_indexed_chunk, spacegroup_chunk,
     downsample_radius) = args
    n = xnn_chunk.shape[0]
    if n == 0:
        return (xnn_chunk, M20_chunk, Minfo_chunk, n_indexed_chunk, spacegroup_chunk)

    neighbor_array = scipy.spatial.distance.cdist(xnn_chunk, xnn_chunk) < downsample_radius
    order = np.arange(n)
    # neighbor_count[o]: how many still-live points lie within the radius of o.
    # Maintained by subtracting removed columns instead of being recounted.
    neighbor_count = neighbor_array.sum(axis=1)

    while order.size:
        counts_in_position_order = neighbor_count[order]
        if counts_in_position_order.max() <= 1:
            break
        highest_density_position = int(np.argmax(counts_in_position_order))
        highest_density_index = order[highest_density_position]
        neighbor_positions = np.flatnonzero(neighbor_array[highest_density_index, order])
        neighbor_indices = order[neighbor_positions]
        best_neighbor = int(np.argmax(M20_chunk[neighbor_indices]))
        keep_index = neighbor_indices[best_neighbor]

        removed = neighbor_indices[neighbor_indices != keep_index]
        if removed.size:
            neighbor_count -= neighbor_array[:, removed].sum(axis=1)

        survivors = np.ones(order.size, dtype=bool)
        survivors[neighbor_positions] = False
        order = np.concatenate((order[survivors], [keep_index]))

    return (xnn_chunk[order], M20_chunk[order], Minfo_chunk[order],
            n_indexed_chunk[order], [spacegroup_chunk[i] for i in order])


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
        candidates.prune_below_m20(threshold=self.opt_params['prune_m20_threshold'])
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

    def run(self, entry=None, q2=None, n_top_candidates=20, zero_error=False, wavelength=None):
        self.q2_obs = np.zeros(self.n_peaks)
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
        self.comm.Send(candidates.m20_at_prune, dest=self.root)
        self.comm.send(candidates.best_spacegroup, dest=self.root)

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

        # Candidate dumping for the FOM benchmark (docs/fom/SCHEMA.md). Off unless the
        # driver sets opt_params['dump_candidates'] and dump_context; production is
        # untouched. The records are buffered rather than written because
        # _downsample_computation runs once per Bravais lattice per entry, and a shard per
        # call would mean hundreds of thousands of small files over a full grid.
        self.dump_context = None
        self._dump_records = []
        self._predownsample_records = []

        opt_params_defaults = {
            'minimum_uc': 2,
            'maximum_uc': 500,
            'dump_candidates': None,
            # S14 (Q31, F-065). Both default to the production behaviour, so an
            # optimizer built without options is byte-identical to one built before
            # these keys existed. prune_m20_threshold is read in _run_loop, which runs
            # on every rank, so it must arrive through get_optimizers(options=...) --
            # setting it after construction reaches the manager only.
            'prune_m20_threshold': 5.0,
            'dump_predownsample': None,
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

    def run(self, entry=None, q2=None, n_top_candidates=20, zero_error=False, wavelength=None):
        if (entry is None) and (not q2 is None):
            self.q2_obs = q2[:self.n_peaks]
        elif (not entry is None) and (q2 is None):
            self.q2_obs = np.array(entry['q2'])[:self.n_peaks]
            if self.opt_params['convergence_testing'] or self.opt_params['redistribution_testing']:
                self.xnn_true = np.array(entry['reindexed_xnn'])[self.wrapper.data_params['unit_cell_indices']]
        self.zero_error = zero_error
        self.wavelength = wavelength
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

    def _record_candidate_dump(self, xnn, M20, Minfo, n_indexed, spacegroup,
                               n_entering, n_top_candidates):
        """Buffer every deduplicated candidate, not just the top N that ranking keeps.

        Called from _downsample_computation *before* the truncation, so `final_rank`
        covers the whole survivor set and the rows that ranking is about to discard are
        still present. Read-only with respect to the optimizer: nothing here feeds back
        into the arrays the caller goes on to sort.

        The M20 recorded here is the post-`assign_extinction_group` value, which is the
        one ranking uses -- `Candidates.assign_extinction_group` rebinds `best_M20` to the
        maximum over extinction groups, and that is what arrives in this method.
        """
        if self.zero_error:
            raise NotImplementedError(
                'Candidate dumping does not support zero-error refinement: the per-'
                'candidate zeropoint stays in the worker and never reaches the manager, '
                'so the dumped columns would not reproduce the pipeline M20.'
                )
        from mlindex.model_training.FomBenchmark import q2_digest

        context = self.dump_context or {}
        # Descending M20 over every survivor, so rank 0 is the candidate ranking would
        # pick and the truncation is expressible as a boolean rather than a missing row.
        order = np.argsort(M20)[::-1]
        final_rank = np.empty(M20.shape[0], dtype=int)
        final_rank[order] = np.arange(M20.shape[0])

        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        unit_cell = get_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self._dump_records.append({
            'bravais_lattice': self.bravais_lattice,
            'lattice_system': self.lattice_system,
            'q2_digest': context.get('q2_digest') or q2_digest(self.q2_obs),
            'context': context,
            'n_peaks': int(self.n_peaks),
            'hkl_ref_length': int(self.hkl_ref_length),
            'n_entering': n_entering,
            'assignment_threshold': float(self.opt_params['assignment_threshold']),
            'downsample_radius': float(self.opt_params['downsample_radius']),
            'xnn': np.array(xnn, dtype=np.float64, copy=True),
            'unit_cell': np.array(unit_cell, dtype=np.float64, copy=True),
            'volume': get_unit_cell_volume(
                unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system),
            'reciprocal_volume': get_unit_cell_volume(
                reciprocal_unit_cell, partial_unit_cell=True,
                lattice_system=self.lattice_system),
            'M20': np.array(M20, dtype=np.float64, copy=True),
            'Minfo': np.array(Minfo, dtype=np.float64, copy=True),
            'n_indexed': np.array(n_indexed, dtype=int, copy=True),
            'spacegroup': list(spacegroup),
            'final_rank': final_rank,
            'in_top_n': final_rank < n_top_candidates,
            })

    def _record_predownsample_dump(self, xnn, M20, Minfo, n_indexed, spacegroup, n_entering,
                                   m20_at_prune):
        """Buffer every candidate entering deduplication, before any is collapsed.

        Deduplication keeps the highest-M20 member of each xnn-space neighbourhood and
        deletes the rest (`_downsample_chunk`). Benchmark A stored only the survivors, so
        what the tiebreak destroyed has never been measurable -- F-065, rebuild row R2.
        This is the missing rows.

        Deliberately a *separate* record stream from `_record_candidate_dump`, with fewer
        columns: the pre-deduplication population is ~2x the survivors at the production
        prune threshold and far larger at threshold 0, and unit_cell / volume /
        reciprocal_volume are all recoverable from xnn. Keeping it separate also leaves
        Benchmark A's candidate schema untouched, so existing shards and loaders are
        unaffected.

        Read-only with respect to the optimizer: the caller goes on to sort and collapse
        the same arrays, so everything stored here is copied.
        """
        if self.zero_error:
            raise NotImplementedError(
                'Candidate dumping does not support zero-error refinement: the per-'
                'candidate zeropoint stays in the worker and never reaches the manager, '
                'so the dumped columns would not reproduce the pipeline M20.'
                )
        from mlindex.model_training.FomBenchmark import q2_digest

        context = self.dump_context or {}
        self._predownsample_records.append({
            'bravais_lattice': self.bravais_lattice,
            'lattice_system': self.lattice_system,
            'q2_digest': context.get('q2_digest') or q2_digest(self.q2_obs),
            'context': context,
            'n_peaks': int(self.n_peaks),
            'hkl_ref_length': int(self.hkl_ref_length),
            'n_entering': n_entering,
            'prune_m20_threshold': float(self.opt_params['prune_m20_threshold']),
            'downsample_radius': float(self.opt_params['downsample_radius']),
            'xnn': np.array(xnn, dtype=np.float64, copy=True),
            'M20': np.array(M20, dtype=np.float64, copy=True),
            'Minfo': np.array(Minfo, dtype=np.float64, copy=True),
            'n_indexed': np.array(n_indexed, dtype=int, copy=True),
            'm20_at_prune': np.array(m20_at_prune, dtype=np.float64, copy=True),
            'spacegroup': list(spacegroup),
            })

    def drain_candidate_dump(self):
        """Hand the buffered records to the driver and reset, so memory stays bounded."""
        records = self._dump_records
        self._dump_records = []
        return records

    def drain_predownsample_dump(self):
        """As drain_candidate_dump, for the pre-deduplication stream."""
        records = self._predownsample_records
        self._predownsample_records = []
        return records

    def _downsample_computation(self, best_M20_all, best_Minfo_all, best_xnn_all,
                                best_n_indexed_all, best_spacegroup_all,
                                n_top_candidates, m20_at_prune_all=None):
        best_M20_all = np.concatenate(best_M20_all, axis=0)
        best_Minfo_all = np.concatenate(best_Minfo_all, axis=0)
        best_xnn_all = np.concatenate(best_xnn_all, axis=0)
        best_n_indexed_all = np.concatenate(best_n_indexed_all, axis=0)
        # Optional so convergence_testing and any caller predating it still work.
        if m20_at_prune_all is None:
            m20_at_prune_all = np.full(best_M20_all.shape, np.nan)
        else:
            m20_at_prune_all = np.concatenate(m20_at_prune_all, axis=0)
        # How many candidates survived prune_below_m20 across all ranks. Recorded rather
        # than inferred because the dump keeps only the deduplicated survivors, and the
        # ratio is the only trace left of how hard the M20 >= 5 cut bit for this entry.
        n_entering = int(best_xnn_all.shape[0])

        # Remove any candidates with np.nan as a unit cell.
        # I believe these are caused by numerical issues with triclinic unit cells during the
        # Selling reduction
        good_indices = np.invert(np.any(np.isnan(best_xnn_all), axis=1))
        best_M20_all = best_M20_all[good_indices]
        best_Minfo_all = best_Minfo_all[good_indices]
        best_xnn_all = best_xnn_all[good_indices]
        best_n_indexed_all = best_n_indexed_all[good_indices]
        m20_at_prune_all = m20_at_prune_all[good_indices]
        # best_spacegroup_all is a list and was left unfiltered here, while sort_indices
        # below index the *filtered* arrays -- so a single dropped row slid every later
        # spacegroup onto a different candidate's cell, silently, including onto the
        # highest-M20 candidate that ranking goes on to report.
        best_spacegroup_all = [
            spacegroup for spacegroup, keep in zip(best_spacegroup_all, good_indices) if keep
            ]

        # Every candidate that reaches deduplication, before it removes any of them. The
        # only trace this used to leave was the scalar n_entering, which is why the cost of
        # the tiebreak has never been measurable (F-065, rebuild row R2). Recorded here
        # rather than in _downsample_chunk because the chunker is a module-level function
        # under a ThreadPoolExecutor with no access to self, and its return contract is
        # pinned by a bit-identity harness.
        if self.opt_params.get('dump_predownsample'):
            self._record_predownsample_dump(
                best_xnn_all, best_M20_all, best_Minfo_all,
                best_n_indexed_all, best_spacegroup_all, n_entering,
                m20_at_prune_all,
                )

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
        chunk_size = 1000
        n_chunks = best_xnn_all.shape[0] // chunk_size + 1

        downsample_radius = self.opt_params['downsample_radius']

        chunk_args = []
        for chunk_index in range(n_chunks):
            start = chunk_index * chunk_size
            end = None if chunk_index == n_chunks - 1 else (chunk_index + 1) * chunk_size
            chunk_args.append((
                best_xnn_all[start:end],
                best_M20_all[start:end],
                best_Minfo_all[start:end],
                best_n_indexed_all[start:end],
                best_spacegroup_all[start:end],
                downsample_radius,
            ))

        with ThreadPoolExecutor(max_workers=self.n_ranks) as ex:
            chunk_results = list(ex.map(_downsample_chunk, chunk_args))

        xnn_downsampled = []
        M20_downsampled = []
        Minfo_downsampled = []
        n_indexed_downsampled = []
        spacegroup_downsampled = []
        for (xnn_chunk, M20_chunk, Minfo_chunk, n_indexed_chunk, spacegroup_chunk) in chunk_results:
            xnn_downsampled.append(xnn_chunk)
            M20_downsampled.append(M20_chunk)
            Minfo_downsampled.append(Minfo_chunk)
            n_indexed_downsampled.append(n_indexed_chunk)
            spacegroup_downsampled += spacegroup_chunk
        xnn_downsampled = np.vstack(xnn_downsampled)
        M20_downsampled = np.concatenate(M20_downsampled)
        Minfo_downsampled = np.concatenate(Minfo_downsampled)
        n_indexed_downsampled = np.concatenate(n_indexed_downsampled)

        if self.opt_params.get('dump_candidates'):
            self._record_candidate_dump(
                xnn_downsampled, M20_downsampled, Minfo_downsampled,
                n_indexed_downsampled, spacegroup_downsampled,
                n_entering, n_top_candidates,
                )

        sort_indices = np.argsort(M20_downsampled)[::-1][:n_top_candidates]
        self.top_xnn = xnn_downsampled[sort_indices]
        self.top_M20 = M20_downsampled[sort_indices]
        self.top_Minfo = Minfo_downsampled[sort_indices]
        self.top_n_indexed = n_indexed_downsampled[sort_indices]
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
        m20_at_prune_all = []
        for rank_index in range(self.n_ranks):
            if rank_index == self.root:
                best_M20_all.append(candidates.best_M20)
                best_Minfo_all.append(candidates.best_Minfo)
                best_xnn_all.append(candidates.best_xnn)
                best_n_indexed_all.append(candidates.n_indexed)
                m20_at_prune_all.append(candidates.m20_at_prune)
                best_spacegroup_all += candidates.best_spacegroup
            else:
                best_M20_rank = np.zeros(self.sent_candidates[rank_index])
                best_Minfo_rank = np.zeros(self.sent_candidates[rank_index])
                best_xnn_rank = np.zeros((self.sent_candidates[rank_index], self.unit_cell_length))
                best_n_indexed_rank = np.zeros(self.sent_candidates[rank_index], dtype=int)
                m20_at_prune_rank = np.zeros(self.sent_candidates[rank_index])
                self.comm.Recv(best_M20_rank, source=rank_index)
                self.comm.Recv(best_Minfo_rank, source=rank_index)
                self.comm.Recv(best_xnn_rank, source=rank_index)
                self.comm.Recv(best_n_indexed_rank, source=rank_index)
                self.comm.Recv(m20_at_prune_rank, source=rank_index)
                best_spacegroup_rank = self.comm.recv(source=rank_index)
                best_M20_all.append(best_M20_rank)
                best_Minfo_all.append(best_Minfo_rank)
                best_xnn_all.append(best_xnn_rank)
                best_n_indexed_all.append(best_n_indexed_rank)
                m20_at_prune_all.append(m20_at_prune_rank)
                best_spacegroup_all += best_spacegroup_rank

        self._downsample_computation(best_M20_all, best_Minfo_all, best_xnn_all,
                                     best_n_indexed_all, best_spacegroup_all,
                                     n_top_candidates, m20_at_prune_all)

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
