from concurrent.futures import ThreadPoolExecutor
import hashlib

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


def _scaled_iterations(iteration_info, scale):
    """How many passes of one `iteration_info` block to run at a given schedule scale.

    The shipped schedule is per lattice system -- cubic runs 1 deterministic pass plus 5 random
    subsampling passes on ten peaks, triclinic 1 plus 60 on twenty -- so a reduced schedule is a
    multiplier on those counts, not a single number applied everywhere.

    Two properties this rounding has to keep:

    * `scale == 1.0` returns the stored count unchanged, so the shipped search is untouched and
      byte-identical. It is compared exactly rather than approximately: 1.0 is the default and
      arrives as a literal, never as the result of arithmetic.
    * A block never scales to zero. `random_subsampling` is what makes the search stochastic at
      all, and a lattice system whose only random block vanished would silently become a
      one-shot deterministic solve rather than a cheaper search.

    Rounding is `round`, so cubic's 5 goes to 2 at a half schedule (banker's rounding on 2.5) and
    to 1 at a quarter. Cubic is excluded from the pilot's per-lattice comparison for exactly this
    reason -- with five passes it has too little resolution for a scale factor to mean the same
    thing it means for triclinic's sixty.
    """
    n_iterations = int(iteration_info['n_iterations'])
    if scale == 1.0:
        return n_iterations
    if scale <= 0.0:
        raise ValueError(f'iteration_scale must be positive, got {scale}')
    return max(1, int(round(n_iterations * scale)))


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


def _derived_seed(key, base_seed):
    """A stable seed for `key`. `hash()` will not do: it is salted per process."""
    digest = hashlib.sha256(f'{base_seed}:{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def _worker_payload(candidates):
    """What a rank hands back after the post-cut block.

    One dict rather than typed sends: a typed Recv needs its buffer sized before the message
    arrives, and the manager's only estimate is how many candidates it *sent* -- which is not what
    comes back, since prune_below_m20 drops rows and correct_off_by_two appends them.

    The at-prune columns ride along only when the research capture is on, so the shipped payload
    is unchanged.
    """
    payload = {
        'M20': candidates.best_M20,
        'Minfo': candidates.best_Minfo,
        'xnn': candidates.best_xnn,
        'n_indexed': candidates.n_indexed,
        'spacegroup': list(candidates.best_spacegroup),
        }
    if candidates.m20_at_prune is not None:
        payload['m20_at_prune'] = candidates.m20_at_prune
        payload['merit_at_prune'] = candidates.merit_at_prune
    return payload


def _collect_at_prune(candidates, results):
    """Assemble the at-prune columns across the manager's own candidates and every rank's.

    Returns None when the capture is off, which is what keeps `_downsample_computation` on its
    original path for every shipped run.
    """
    if candidates.m20_at_prune is None:
        return None
    columns = {'m20_at_prune': [candidates.m20_at_prune]}
    for name, values in candidates.merit_at_prune.items():
        columns[f'merit_at_prune_{name}'] = [values]
    for result in results:
        columns['m20_at_prune'].append(result['m20_at_prune'])
        for name, values in result['merit_at_prune'].items():
            columns[f'merit_at_prune_{name}'].append(values)
    return {name: np.concatenate(pieces) for name, pieces in columns.items()}


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

        # DWMM's compute lever: run the search for half, or a quarter, of the iterations. The
        # schedule is already tuned per lattice system -- triclinic gets twelve times cubic's --
        # so the knob is a scale factor on the existing counts rather than a flat number, and the
        # deterministic pass is never scaled away. Default 1.0, which is exactly the shipped
        # schedule; S06's pilot is what decides whether anything else is used.
        #
        # It is applied HERE, in the one place all three optimizer classes share. A scale placed
        # in `run_common` would be inert in multiprocessing mode, because MPOptimizerManager and
        # MPOptimizerWorker both override it -- which is precisely how the search's reseeding came
        # to look implemented while never firing at all (C2-F-042).
        iteration_scale = float(self.opt_params.get('iteration_scale', 1.0))
        for iteration_info in self.opt_params['iteration_info']:
            for iter_index in range(_scaled_iterations(iteration_info, iteration_scale)):
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
        # These two are the only steps here that change the candidate count:
        # prune_below_m20 drops rows, and correct_off_by_two appends an off-by-two
        # corrected copy beside the original rather than replacing it. Convergence
        # testing measures how far each perturbed starting cell travels, so it needs the
        # one-to-one correspondence with the cells it generated -- and it reassembles by
        # stride, which silently misaligns if the count moves. Neither step serves that
        # measurement, so skip both rather than trying to track the permutation.
        convergence = self.opt_params['convergence_testing']
        if not convergence:
            # `.get` with the shipped default, so an opt_params dict that does not carry the key
            # behaves exactly as before. This is the research route the decisions log blesses --
            # campaign 2 varies the cut through opt_params on its own branch and never through
            # the CLI, which is a settled question (C2-F-008). A per-lattice mapping is accepted
            # here too; see Candidates.prune_below_m20.
            candidates.prune_below_m20(
                threshold=self.opt_params.get('prune_m20_threshold', 5.0))
        candidates.refine_cell()
        candidates.standardize_cell()
        if not convergence:
            candidates.correct_off_by_two()
        candidates.assign_extinction_group()
        candidates.calculate_peaks_indexed()
        if convergence:
            self.convergence_testing(candidates)
        else:
            self.downsample_candidates(candidates, n_top_candidates)

    def run_common(self, n_top_candidates):
        self.comm.Bcast(self.q2_obs, root=self.root)
        self._reseed_for_pattern()
        self._run_loop(n_top_candidates)

    def _reseed_for_pattern(self):
        """Re-key the search RNG to this pattern, if the driver asked for it.

        **Off unless `opt_params['search_seed_scheme'] == 'per_entry_bravais'`.** It changes which
        candidates the search generates, so the shipped behaviour has to stay reproducible and the
        default is campaign 1's: one generator per pool, advanced by every entry.

        Why the benchmark needs it (PROTOCOL §6, R17). With one generator per pool, a 243-entry
        run stripes across pools differently from a 5 955-entry one, every entry meets the
        generators at a different state, and **no subset of the benchmark can be regenerated
        comparably**. That single fact forced a within-run restriction on every result in campaign
        1's final phase.

        The key is the observed peak list, the Bravais lattice and the rank -- not the entry
        identifier. That matters for a practical reason: `q2_obs` has just been broadcast, so
        every rank can derive the same key from what it already holds, and no worker protocol
        changes. It is also the more honest key, being a function of the entry *and* the condition
        rather than of the entry alone, so two bundles of one crystal are separate draws while the
        same bundle regenerates identically anywhere.
        """
        if self.opt_params.get('search_seed_scheme') != 'per_entry_bravais':
            return
        from mlindex.model_training.FomBenchmark import q2_digest
        base_seed = self.opt_params.get('search_base_seed', 12345)
        # `self.rank`, not `self.comm.Get_rank()`. Every class here carries `rank` --
        # `OptimizerBase.__init__` sets it from the communicator, and `MPOptimizerWorker` sets it
        # directly because it never calls that constructor -- but only the MPI classes and the MP
        # *manager* carry a `comm` at all. A worker in multiprocessing mode has none, so the
        # attribute lookup raised `AttributeError`, the pool reported `Worker N failed`, and the
        # scheme was unusable at any pool size above one. It went unseen because S05's gates ran
        # at `--pool-size 1`, where the manager is the only rank -- and multiprocessing is the
        # mode the benchmark generates in, which is the same blind spot C2-F-042 itself records.
        # The seeds are unchanged: rank 0 is rank 0 either way, so S05's gate results stand.
        key = f'search:{q2_digest(self.q2_obs)}:{self.bravais_lattice}:{self.rank}'
        self.rng = np.random.default_rng(_derived_seed(key, base_seed))

        # Reseeding the optimizer's own generator is NOT enough, and finding that out is what
        # gate 2 is for. `MITemplates` and `IntegralFilter` each construct their own
        # `default_rng` at setup and advance it on every call, so they carry state from one entry
        # to the next entirely outside `self.rng`. Left alone, the first entry of a run
        # reproduces and every later one drifts -- measured at 119 of 7 503 candidate rows over
        # three entries, with the first entry exact and cubic exact throughout, because cubic
        # draws from them least.
        wrapper = getattr(self, 'wrapper', None)
        if wrapper is None:
            return
        for attribute in ('random_forest_generator', 'integral_filter_generator',
                          'miller_index_templator', 'random_unit_cell_generator'):
            components = getattr(wrapper, attribute, None)
            if not isinstance(components, dict):
                continue
            for name, component in sorted(components.items(),
                                          key=lambda item: str(item[0])):
                if component is not None and hasattr(component, 'rng'):
                    component.rng = np.random.default_rng(
                        _derived_seed(f'{key}:{attribute}:{name}', base_seed))


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
        # One pickled message rather than four typed Sends plus a pickled list. A typed
        # Recv needs its buffer sized before the message arrives, and the manager's only
        # estimate is how many candidates it *sent* -- which is not what comes back:
        # prune_below_m20 drops rows and correct_off_by_two appends them. Sending
        # everything in one object means the arrays and the spacegroup list cannot
        # disagree about the count, which is the failure this replaces. MPOptimizer has
        # always worked this way; the two paths now match.
        self.comm.send(
            _worker_payload(candidates),
            dest=self.root,
            )

    def convergence_testing(self, candidates):
        self.comm.send(
            {'M20': candidates.best_M20, 'xnn': candidates.best_xnn}, dest=self.root
            )


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

        # Research capture, off unless a driver asks for it. `dump_context` carries the entry
        # identity down to the recorder: the hook fires per (entry, Bravais lattice) and would
        # otherwise have no idea which pattern it is looking at, which is how campaign 1's dump
        # came to leave `condition_bundle` in a filename and nowhere else (R8).
        self._dump_records = []
        self.dump_context = None
        self.predownsample = None

        opt_params_defaults = {
            'minimum_uc': 2,
            'maximum_uc': 500,
            'dump_candidates': False,
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

    def _record_candidate_dump(self, xnn, M20, Minfo, n_indexed, spacegroup, n_entering,
                               n_top_candidates, at_prune=None):
        """Buffer every candidate that survived deduplication, for the benchmark dump.

        Read-only with respect to the optimizer: the caller goes on to sort and truncate the same
        arrays, so everything stored here is copied.

        `final_rank` is the rank by descending M20 over ALL survivors, not over the printed
        twenty. That is what lets the truncation be expressed as a boolean rather than as a
        missing row, and it is computed here rather than on load because a within-lattice rank
        recomputed from surviving rows drifts whenever rows are dropped -- campaign 1 recomputed
        its volume decile that way and moved 114 entries, all upward (R14).

        The at-prune merits ride along keyed to the survivors, which is only possible because the
        deduplication carried row identity through. Without them `merit_at_prune` would again be
        unavailable at the one place it is needed (C2-R-001).
        """
        if self.zero_error:
            raise NotImplementedError(
                'Candidate dumping does not support zero-error refinement: the per-candidate '
                'zeropoint stays in the worker and never reaches the manager, so the dumped '
                'columns would not reproduce the pipeline M20.'
                )
        context = dict(self.dump_context or {})

        order = np.argsort(M20)[::-1]
        final_rank = np.empty(M20.shape[0], dtype=int)
        final_rank[order] = np.arange(M20.shape[0])

        unit_cell = get_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system)
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system)

        from mlindex.model_training.FomBenchmark import q2_digest

        record = {
            'bravais_lattice': self.bravais_lattice,
            'lattice_system': self.lattice_system,
            'context': context,
            # The join-integrity check. Falls back to computing it here so the hook is usable
            # without a driver-supplied context; a mis-joined shard is otherwise silent, since
            # every column still parses and the numbers simply attach to the wrong pattern.
            'q2_digest': context.get('q2_digest') or q2_digest(self.q2_obs),
            'n_peaks': int(self.n_peaks),
            'hkl_ref_length': int(self.hkl_ref_length),
            'n_entering': int(n_entering),
            'assignment_threshold': float(self.opt_params['assignment_threshold']),
            'downsample_radius': float(self.opt_params['downsample_radius']),
            'prune_m20_threshold': float(self.opt_params.get('prune_m20_threshold', 5.0)),
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
            }

        # S04's absence counts, which replace the 158-level extinction-group categorical as S12's
        # symmetry feature (C2-F-041). Both are table lookups keyed on (lattice, group), so they
        # cost nothing here. `n_absent_extra_in_range` is deliberately NOT stored: it needs the
        # candidate's own reference lines, and it is recomputable offline from `xnn`, the peak
        # list and the group -- which is the whole design of this dump. A column earns its place
        # by being expensive to recompute or by recording a value that no longer exists later.
        try:
            from mlindex.utilities.ExtinctionCounts import get_absence_counts
            from mlindex.utilities.ExtinctionCounts import get_n_groups_searched
            counts = get_absence_counts(self.bravais_lattice)
            record['n_absent_extra'] = np.array(
                [counts.get(group, 0) for group in spacegroup], dtype=np.int64)
            record['n_groups_searched'] = np.full(
                len(spacegroup), get_n_groups_searched(self.bravais_lattice), dtype=np.int64)
        except Exception:
            # A missing lookup table must not cost the whole dump; the columns are recoverable
            # from `spacegroup`, which is stored either way.
            pass
        for name, values in (at_prune or {}).items():
            record[name] = np.array(values, dtype=np.float64, copy=True)
        self._dump_records.append(record)

    def drain_candidate_dump(self):
        """Hand the buffered records to the driver and reset, so memory stays bounded."""
        records = self._dump_records
        self._dump_records = []
        return records

    def drain_predownsample_dump(self):
        """As drain_candidate_dump, for the pre-deduplication stream.

        Returns a list so the two streams have the same shape for the frame builders, even though
        the capture itself is one dict per (entry, Bravais lattice).
        """
        records = self.predownsample
        self.predownsample = None
        return [] if records is None else [records]

    def _downsample_computation(self, best_M20_all, best_Minfo_all, best_xnn_all,
                                best_n_indexed_all, best_spacegroup_all,
                                n_top_candidates, at_prune=None):
        best_M20_all = np.concatenate(best_M20_all, axis=0)
        best_Minfo_all = np.concatenate(best_Minfo_all, axis=0)
        best_xnn_all = np.concatenate(best_xnn_all, axis=0)
        best_n_indexed_all = np.concatenate(best_n_indexed_all, axis=0)

        # The arrays and the spacegroup list index the same candidates, and the sort below
        # indexes the list with positions taken from the arrays. If they ever disagree the
        # symptom is not a clean failure here: the NaN filter's `zip` truncates to the
        # shorter one and the sort then runs off the end of the list, several steps later
        # and in a way that reads like a sorting bug. Under MPI that killed the manager
        # rank and left every other rank blocked on the next barrier, hanging the job.
        if len(best_spacegroup_all) != best_M20_all.shape[0]:
            raise ValueError(
                f'got {best_M20_all.shape[0]} candidates but '
                f'{len(best_spacegroup_all)} spacegroups; they index the same candidates '
                'and must be the same length'
                )

        # Remove any candidates with np.nan as a unit cell.
        # I believe these are caused by numerical issues with triclinic unit cells during the
        # Selling reduction
        good_indices = np.invert(np.any(np.isnan(best_xnn_all), axis=1))
        best_M20_all = best_M20_all[good_indices]
        best_Minfo_all = best_Minfo_all[good_indices]
        best_xnn_all = best_xnn_all[good_indices]
        best_n_indexed_all = best_n_indexed_all[good_indices]
        # best_spacegroup_all is a list and was left unfiltered here, while sort_indices
        # below index the *filtered* arrays -- so a single dropped row slid every later
        # spacegroup onto a different candidate's cell, silently, including onto the
        # highest-M20 candidate that ranking goes on to report.
        best_spacegroup_all = [
            spacegroup for spacegroup, keep in zip(best_spacegroup_all, good_indices) if keep
            ]

        # Every candidate that reaches deduplication, before it removes any of them, and before
        # the reciprocal-volume sort reorders them. Only populated under the research capture;
        # the driver drains it after each lattice. Copied, because the caller goes on to sort and
        # collapse these same arrays.
        if at_prune is not None:
            self.predownsample = dict(
                {'xnn': best_xnn_all.copy(), 'M20': best_M20_all.copy(),
                 'Minfo': best_Minfo_all.copy(), 'n_indexed': best_n_indexed_all.copy(),
                 'spacegroup': list(best_spacegroup_all),
                 'bravais_lattice': self.bravais_lattice,
                 'lattice_system': self.lattice_system,
                 'n_peaks': int(self.n_peaks),
                 'hkl_ref_length': int(self.hkl_ref_length),
                 'downsample_radius': float(self.opt_params['downsample_radius']),
                 # Schema v3 needs these on the pre-deduplication stream too, so it joins on the
                 # same keys as the survivors and carries the condition on the row rather than in
                 # a filename (R8). `n_entering` is this stream's own row count by construction.
                 'context': dict(self.dump_context or {}),
                 'n_entering': int(best_M20_all.shape[0]),
                 'prune_m20_threshold': float(self.opt_params.get('prune_m20_threshold', 5.0)),
                 'q2_digest': (self.dump_context or {}).get('q2_digest')},
                **{name: values[good_indices] for name, values in at_prune.items()})

        # How many candidates reached deduplication. `n_entering` is a benchmark column in its
        # own right -- it is what makes the tie-break's cost measurable rather than inferred --
        # so it is taken here, after the NaN filter and before anything is collapsed.
        n_entering = int(best_M20_all.shape[0])
        at_prune_filtered = (None if at_prune is None
                             else {name: values[good_indices]
                                   for name, values in at_prune.items()})

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
        if at_prune_filtered is not None:
            at_prune_filtered = {name: values[sort_indices]
                                 for name, values in at_prune_filtered.items()}
        chunk_size = 1000
        n_chunks = best_xnn_all.shape[0] // chunk_size + 1

        downsample_radius = self.opt_params['downsample_radius']

        # The identity of each row, carried through `_downsample_chunk`'s spacegroup slot. That
        # function never reads the slot -- its only use is `[spacegroup_chunk[i] for i in order]`
        # -- so substituting row indices is bit-identical and hands back the survivor mapping for
        # free. Without it there is no way to attach a survivor to the at-prune merits it was cut
        # on, short of running the deduplication twice or matching cells by value. (This is the
        # same trick S03's deduplication emulator uses, for the same reason.)
        row_indices = list(range(best_M20_all.shape[0]))

        chunk_args = []
        for chunk_index in range(n_chunks):
            start = chunk_index * chunk_size
            end = None if chunk_index == n_chunks - 1 else (chunk_index + 1) * chunk_size
            chunk_args.append((
                best_xnn_all[start:end],
                best_M20_all[start:end],
                best_Minfo_all[start:end],
                best_n_indexed_all[start:end],
                row_indices[start:end],
                downsample_radius,
            ))

        with ThreadPoolExecutor(max_workers=self.n_ranks) as ex:
            chunk_results = list(ex.map(_downsample_chunk, chunk_args))

        xnn_downsampled = []
        M20_downsampled = []
        Minfo_downsampled = []
        n_indexed_downsampled = []
        survivor_indices = []
        for (xnn_chunk, M20_chunk, Minfo_chunk, n_indexed_chunk, index_chunk) in chunk_results:
            xnn_downsampled.append(xnn_chunk)
            M20_downsampled.append(M20_chunk)
            Minfo_downsampled.append(Minfo_chunk)
            n_indexed_downsampled.append(n_indexed_chunk)
            survivor_indices += index_chunk
        xnn_downsampled = np.vstack(xnn_downsampled)
        M20_downsampled = np.concatenate(M20_downsampled)
        Minfo_downsampled = np.concatenate(Minfo_downsampled)
        n_indexed_downsampled = np.concatenate(n_indexed_downsampled)
        spacegroup_downsampled = [best_spacegroup_all[i] for i in survivor_indices]

        # Every candidate that survived deduplication, with the rank it would be given over ALL
        # survivors rather than only the printed twenty. Off unless the driver asks for it.
        if self.opt_params.get('dump_candidates'):
            self._record_candidate_dump(
                xnn_downsampled, M20_downsampled, Minfo_downsampled, n_indexed_downsampled,
                spacegroup_downsampled, n_entering, n_top_candidates,
                None if at_prune_filtered is None
                else {name: values[survivor_indices]
                      for name, values in at_prune_filtered.items()},
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
        results = []
        for rank_index in range(self.n_ranks):
            if rank_index == self.root:
                best_M20_all.append(candidates.best_M20)
                best_Minfo_all.append(candidates.best_Minfo)
                best_xnn_all.append(candidates.best_xnn)
                best_n_indexed_all.append(candidates.n_indexed)
                best_spacegroup_all += candidates.best_spacegroup
            else:
                # `recv` returns the arrays at the length the rank actually produced, so
                # nothing has to be sized in advance. Sizing these from
                # `self.sent_candidates` is what used to hang the run: the typed Recv
                # zero-padded the arrays up to the outgoing count while the spacegroup
                # list arrived at its true, shorter length, and the two then disagreed in
                # `_downsample_computation`.
                result = self.comm.recv(source=rank_index)
                results.append(result)
                best_M20_all.append(result['M20'])
                best_Minfo_all.append(result['Minfo'])
                best_xnn_all.append(result['xnn'])
                best_n_indexed_all.append(result['n_indexed'])
                best_spacegroup_all += result['spacegroup']

        self._downsample_computation(best_M20_all, best_Minfo_all, best_xnn_all,
                                     best_n_indexed_all, best_spacegroup_all,
                                     n_top_candidates,
                                     at_prune=_collect_at_prune(candidates, results))

    def convergence_testing(self, candidates):
        n_candidates = self.opt_params['convergence_candidates'] * len(self.opt_params['convergence_distances'])
        self.top_M20 = np.zeros(n_candidates)
        self.top_xnn = np.zeros((n_candidates, self.wrapper.data_params['unit_cell_length']))
        for rank_index in range(self.n_ranks):
            if rank_index == self.root:
                self.top_M20[rank_index::self.n_ranks] = candidates.best_M20
                self.top_xnn[rank_index::self.n_ranks] = candidates.best_xnn
            else:
                result = self.comm.recv(source=rank_index)
                best_M20_rank = result['M20']
                best_xnn_rank = result['xnn']
                # Unlike the downsample path, this one scatters back into a strided view,
                # so it needs exactly the rows it sent -- a returned count that differs
                # would misalign every candidate with its starting perturbation rather
                # than merely shortening the pool. `_run_loop` skips the two steps that
                # change the count under convergence testing; this checks that it did.
                if best_M20_rank.shape[0] != self.sent_candidates[rank_index]:
                    raise RuntimeError(
                        f'rank {rank_index} returned {best_M20_rank.shape[0]} candidates '
                        f'for convergence testing against {self.sent_candidates[rank_index]} '
                        'sent; the strided reassembly below requires them to match'
                        )
                self.top_M20[rank_index::self.n_ranks] = best_M20_rank
                self.top_xnn[rank_index::self.n_ranks] = best_xnn_rank
        self.top_unit_cell = get_unit_cell_from_xnn(
            self.top_xnn,
            partial_unit_cell=True,
            lattice_system=self.lattice_system,
            )
