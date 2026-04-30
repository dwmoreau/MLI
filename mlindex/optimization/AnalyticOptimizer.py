from importlib.resources import files
import itertools
import numpy as np
from pathlib import Path

from mlindex.optimization.MPIOptimizer import OptimizerManager
from mlindex.optimization.Candidates import Candidates
from mlindex.utilities.Reindexing import reindex_entry_basic
from mlindex.utilities.UnitCellTools import (
    get_hkl_matrix,
    get_unit_cell_from_xnn,
    get_xnn_from_unit_cell,
    fix_unphysical,
    get_unit_cell_volume,
)
from mlindex.optimization.MPOptimizer import LocalComm


class AnalyticOptimizer(OptimizerManager):
    """Analytic optimizer using a guess‑and‑check candidate generation.

    Parameters
    ----------
    bravais_lattice : str
        Bravais lattice identifier (e.g. ``"cP"``, ``"tI``).
    comm : MPI communicator
        Communicator (serial ``MPI.COMM_SELF`` works for a single‑process run).
    n_peaks : int, optional
        Number of peaks to use in the optimization.
    n_peaks_guess : int, optional
        Number of low‑angle peaks to use for the initial linear solve.
    n_ref_hkl : int, optional
        Number of reference HKLs (from the pre‑computed file) to consider when
        building permutations.  The first ``n_ref_hkl`` entries are used.
    opt_params : dict, optional
        Optimisation parameters – if omitted a sensible default is constructed
        based on the lattice system.
    """

    def __init__(self, bravais_lattice, comm, n_peaks=10, n_peaks_guess=5, n_ref_hkl_guess=5, opt_params=None):
        # Basic attributes required by the parent class
        self.bravais_lattice = bravais_lattice
        self.n_peaks_guess = n_peaks_guess
        self.n_ref_hkl_guess = n_ref_hkl_guess

        # Determine lattice system from bravais label
        if bravais_lattice in ['cP', 'cF', 'cI']:
            self.lattice_system = 'cubic'
        elif bravais_lattice in ['tI', 'tP']:
            self.lattice_system = 'tetragonal'
        elif bravais_lattice == 'hP':
            self.lattice_system = 'hexagonal'
        elif bravais_lattice == 'hR':
            self.lattice_system = 'rhombohedral'
        elif bravais_lattice in ['oC', 'oP', 'oF', 'oI']:
            self.lattice_system = 'orthorhombic'
        elif bravais_lattice in ['mC', 'mP']:
            self.lattice_system = 'monoclinic'
        elif bravais_lattice == 'aP':
            self.lattice_system = 'triclinic'
        else:
            raise ValueError(f'Unsupported bravais lattice for analytic optimizer: {bravais_lattice}')

        # Load the pre‑computed HKL reference array (non‑redundant)
        # Get the absolute path to the MLI directory
        with files('mlindex').joinpath(
            'models', f'{self.lattice_system}_1', 'data', f'hkl_ref_{bravais_lattice}.npy'
        ).open('rb') as _f:
            self.hkl_ref = np.load(_f)
        self.hkl_ref_length = self.hkl_ref.shape[0]

        # Dimensionality of the reciprocal‑space parameter vector (xnn)
        dim_map = {
            'cubic': 1, 'tetragonal': 2, 'hexagonal': 2, 'rhombohedral': 2,
            'orthorhombic': 3, 'monoclinic': 4, 'triclinic': 6,
        }
        self.unit_cell_length = dim_map[self.lattice_system]
        self.n_peaks_guess = max(self.n_peaks_guess, self.unit_cell_length)

        # Default optimisation parameters (matching the user specification)
        if opt_params is None:
            iteration_info = [
                {
                    'worker': 'deterministic',
                    'n_iterations': 2,
                    'triplet_opt': False,
                },
                {
                    'worker': 'random_subsampling',
                    'n_iterations': 5,
                    'n_peaks': n_peaks,
                    'n_drop': n_peaks // 2,
                    'triplet_opt': True,
                    'uniform_sampling': False,
                }
            ]
            self.opt_params = {
                'iteration_info': iteration_info,
                'convergence_testing': False,
                'redistribution_testing': False,
                'assignment_threshold': 0.90,
                'figure_of_merit': 'M20',
                'downsample_radius': 0.0001,
                'minimum_uc': 1,
                'maximum_uc': 500,
            }
        else:
            self.opt_params = opt_params

        # Minimal placeholders required for ``run_common``
        self.rng = np.random.default_rng()

        self.comm = comm
        self.rank = comm.Get_rank()
        self.root = 0
        self.n_ranks = comm.Get_size()
        self.fom = 'M20'
        self.q2_obs = None
        self.triplets = None
        self.n_peaks = n_peaks

        self.lattice_system = self.comm.bcast(self.lattice_system, root=self.root)
        self.bravais_lattice = self.comm.bcast(self.bravais_lattice, root=self.root)
        self.opt_params = self.comm.bcast(self.opt_params, root=self.root)
        self.hkl_ref_length = self.comm.bcast(self.hkl_ref_length, root=self.root)
        self.comm.Bcast(self.hkl_ref, root=self.root)
        self.n_peaks = self.comm.bcast(self.n_peaks, root=self.root)

    # ---------------------------------------------------------------------
    # Overridden candidate generation – guess‑and‑check
    # ---------------------------------------------------------------------
    def _generate_analytic_candidates(self):
        """Build the full array of candidate xnn vectors from HKL permutations.

        Returns ``candidate_xnn_all`` with shape ``(n_candidates, unit_cell_length)``.
        """
        dim = self.unit_cell_length
        q2_guess = self.q2_obs[:self.n_peaks_guess]
        ref_hkls = self.hkl_ref[:self.n_ref_hkl_guess]
        hkl_permutations = np.stack(list(itertools.permutations(ref_hkls, dim)), axis=0)
        q2_permutations = np.stack(list(itertools.combinations(q2_guess, dim)), axis=0)
        n_q2 = q2_permutations.shape[0]
        n_templates = hkl_permutations.shape[0]
        hkl2 = get_hkl_matrix(hkl_permutations, self.lattice_system)
        A = np.repeat(hkl2, n_q2, axis=0)
        b = np.tile(q2_permutations, (n_templates, 1))
        candidate_xnn_all = np.zeros((n_templates * n_q2, dim))
        nonsingular = np.abs(np.linalg.det(A)) > 1e-12
        # linalg.solve gufunc signature is (m,m),(m,n)->(m,n); b needs a trailing dim
        candidate_xnn_all[nonsingular] = np.linalg.solve(
            A[nonsingular], b[nonsingular, :, np.newaxis]
        )[:, :, 0]

        candidate_xnn_all = fix_unphysical(
            xnn=candidate_xnn_all,
            rng=self.rng,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            lattice_system=self.lattice_system,
        )
    
        candidate_unit_cells_all = get_unit_cell_from_xnn(
            candidate_xnn_all, partial_unit_cell=True, lattice_system=self.lattice_system
        )

        candidate_unit_cells_all = reindex_entry_basic(
            candidate_unit_cells_all,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            space='direct'
        )
        candidate_unit_cells_all = fix_unphysical(
            unit_cell=candidate_unit_cells_all,
            rng=self.rng,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            lattice_system=self.lattice_system,
        )
        candidate_xnn_all = get_xnn_from_unit_cell(
            candidate_unit_cells_all,
            partial_unit_cell=True,
            lattice_system=self.lattice_system
        )
        candidate_xnn_all = np.repeat(candidate_xnn_all, repeats=3, axis=0)
        return candidate_xnn_all

    def generate_candidates_rank(self):
        """Distribute analytic candidates across MPI ranks."""
        candidate_xnn_all = self._generate_analytic_candidates()
        self.sent_candidates = np.zeros(self.n_ranks, dtype=int)
        for rank_index in range(self.n_ranks):
            self.sent_candidates[rank_index] = candidate_xnn_all[rank_index::self.n_ranks].shape[0]
            if rank_index == self.root:
                candidate_xnn_rank = candidate_xnn_all[rank_index::self.n_ranks]
            else:
                self.comm.send(candidate_xnn_all[rank_index::self.n_ranks], dest=rank_index)
        return self.generate_candidates_common(candidate_xnn_rank)


class MPAnalyticOptimizer(AnalyticOptimizer):
    """Multiprocessing variant of AnalyticOptimizer.

    Class-level queue attributes are set by setup_mp_analytic_optimizers() before
    any instances are constructed, then cleared after construction completes.
    """
    _mp_data_queues = None
    _mp_result_queues = None
    _mp_n_ranks = None

    def __init__(self, bravais_lattice, comm, n_peaks=10, n_peaks_guess=5,
                 n_ref_hkl_guess=5, opt_params=None):
        self._data_queues = MPAnalyticOptimizer._mp_data_queues
        self._result_queues = MPAnalyticOptimizer._mp_result_queues
        n_ranks = MPAnalyticOptimizer._mp_n_ranks
        # comm arg is ignored; LocalComm lets AnalyticOptimizer.__init__ run without MPI
        super().__init__(bravais_lattice, comm=LocalComm(n_ranks),
                         n_peaks=n_peaks, n_peaks_guess=n_peaks_guess,
                         n_ref_hkl_guess=n_ref_hkl_guess, opt_params=opt_params)
        self._init_workers()

    def _init_workers(self):
        init_data = (self.lattice_system, self.bravais_lattice, self.opt_params,
                     self.hkl_ref_length, self.hkl_ref.copy(), self.n_peaks)
        for r in range(1, self.n_ranks):
            self._data_queues[r].put(init_data)

    def run_common(self, n_top_candidates):
        for r in range(1, self.n_ranks):
            self._data_queues[r].put(self.q2_obs.copy())
            self._data_queues[r].put(self.triplets)
        self._run_loop(n_top_candidates)

    def generate_candidates_rank(self):
        candidate_xnn_all = self._generate_analytic_candidates()
        self.sent_candidates = np.zeros(self.n_ranks, dtype=int)
        for rank_index in range(self.n_ranks):
            self.sent_candidates[rank_index] = candidate_xnn_all[rank_index::self.n_ranks].shape[0]
            if rank_index == self.root:
                candidate_xnn_rank = candidate_xnn_all[rank_index::self.n_ranks]
            else:
                self._data_queues[rank_index].put(candidate_xnn_all[rank_index::self.n_ranks])
        return self.generate_candidates_common(candidate_xnn_rank)

    def downsample_candidates(self, candidates, n_top_candidates):
        from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
        best_M20_all = [candidates.best_M20]
        best_Minfo_all = [candidates.best_Minfo]
        best_xnn_all = [candidates.best_xnn]
        best_n_indexed_all = [candidates.n_indexed]
        best_spacegroup_all = list(candidates.best_spacegroup)
        if self.triplets is not None:
            best_n_indexed_triplets_all = [candidates.n_indexed_triplets]
            best_M_triplets_all = [candidates.best_M_triplets]
        else:
            best_n_indexed_triplets_all = None
            best_M_triplets_all = None
        for r in range(1, self.n_ranks):
            result = self._result_queues[r].get()
            if isinstance(result, Exception):
                raise RuntimeError(f"Worker {r} failed: {result}") from result
            best_M20_all.append(result['M20'])
            best_Minfo_all.append(result['Minfo'])
            best_xnn_all.append(result['xnn'])
            best_n_indexed_all.append(result['n_indexed'])
            best_spacegroup_all += result['spacegroup']
            if self.triplets is not None:
                best_n_indexed_triplets_all.append(result['n_indexed_triplets'])
                best_M_triplets_all.append(result['M_triplets'])
        self._downsample_computation(best_M20_all, best_Minfo_all, best_xnn_all,
                                     best_n_indexed_all, best_spacegroup_all,
                                     best_n_indexed_triplets_all, best_M_triplets_all,
                                     n_top_candidates)

    def convergence_testing(self, candidates):
        from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
        n_candidates = self.opt_params['convergence_candidates'] * len(self.opt_params['convergence_distances'])
        self.top_M20 = np.zeros(n_candidates)
        self.top_xnn = np.zeros((n_candidates, self.unit_cell_length))
        self.top_M20[self.root::self.n_ranks] = candidates.best_M20
        self.top_xnn[self.root::self.n_ranks] = candidates.best_xnn
        for r in range(1, self.n_ranks):
            result = self._result_queues[r].get()
            if isinstance(result, Exception):
                raise RuntimeError(f"Worker {r} failed: {result}") from result
            self.top_M20[r::self.n_ranks] = result['M20']
            self.top_xnn[r::self.n_ranks] = result['xnn']
        self.top_unit_cell = get_unit_cell_from_xnn(
            self.top_xnn, partial_unit_cell=True, lattice_system=self.lattice_system)
