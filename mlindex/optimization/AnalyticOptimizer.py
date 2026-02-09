'''Analytic optimizer for high‑symmetry lattice systems.

This class inherits from ``OptimizerManager`` and replaces the ML‑based candidate
generation with a deterministic, guess‑and‑check approach that works directly in
the reciprocal‑space ``xnn`` representation used throughout the code base.

The workflow mirrors ``mlindex.command_line.run`` but uses only the bravais
lattices ``cF``, ``cI``, ``cP``, ``hP``, ``hR``, ``tI`` and ``tP``.  Candidate
generation proceeds as follows:

1. Load the pre‑computed, non‑redundant HKL reference array for the supplied
   ``bravais_lattice`` (e.g. ``mlindex/models/tetragonal_1/data/hkl_ref_tP.npy``).
2. Take the first ``n_peaks_guess`` observed peaks (low‑angle ``q2`` values) and
   combine them with every permutation of the first ``n_ref_hkl`` reference HKLs
   of size equal to the dimensionality of the reciprocal‑space parameter vector
   (``dim`` = 1 for cubic, 2 for tetragonal/hexagonal, 3 for orthorhombic).
3. For each HKL permutation build the design matrix ``H`` via
   ``get_hkl_matrix`` and solve the linear system ``q2 = H·xnn`` using a least‑
   squares solve.  The resulting ``xnn`` vector is a candidate reciprocal‑space
   parameter set.
4. Convert each ``xnn`` to a conventional unit cell with
   ``get_unit_cell_from_xnn``, enforce physical limits (2 Å ≤ a,b,c ≤ 50 Å) via
   ``fix_unphysical``, and convert back to ``xnn``.
5. Feed the collection of ``xnn`` candidates to the existing ``Candidates``
   implementation.  The standard deterministic optimisation step (Gauss‑Newton
   refinement, HKL assignment, ``M20`` scoring, off‑by‑two correction, etc.) is
   then applied.

Only cubic, tetragonal and hexagonal systems are currently supported – the
rhombohedral case is omitted for simplicity.
'''

import itertools
from pathlib import Path

import numpy as np

from mlindex.optimization.Optimizer import OptimizerManager
from mlindex.optimization.Optimizer import Candidates
from mlindex.utilities.Reindexing import reindex_entry_basic
from mlindex.utilities.UnitCellTools import (
    get_hkl_matrix,
    get_unit_cell_from_xnn,
    get_xnn_from_unit_cell,
    fix_unphysical,
    get_unit_cell_volume,
)


class AnalyticOptimizer(OptimizerManager):
    """Analytic optimizer using a guess‑and‑check candidate generation.

    Parameters
    ----------
    bravais_lattice : str
        Bravais lattice identifier (e.g. ``"cP"``, ``"tI``).
    comm : MPI communicator
        Communicator (serial ``MPI.COMM_SELF`` works for a single‑process run).
    n_peaks_guess : int, optional
        Number of low‑angle peaks to use for the initial linear solve.
    n_ref_hkl : int, optional
        Number of reference HKLs (from the pre‑computed file) to consider when
        building permutations.  The first ``n_ref_hkl`` entries are used.
    opt_params : dict, optional
        Optimisation parameters – if omitted a sensible default is constructed
        based on the lattice system.
    """

    def __init__(self, bravais_lattice, comm, n_peaks_guess=5, n_ref_hkl_guess=5, opt_params=None):
        # Basic attributes required by the parent class
        self.comm = comm
        self.bravais_lattice = bravais_lattice
        self.n_peaks_guess = n_peaks_guess
        self.n_ref_hkl_guess = n_ref_hkl_guess
        self.root = comm.Get_rank()
        self.rank = self.root
        self.n_ranks = comm.Get_size()

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
        else:
            raise ValueError(f'Unsupported bravais lattice for analytic optimizer: {bravais_lattice}')

        # Load the pre‑computed HKL reference array (non‑redundant)
        data_path = (
            Path('mlindex')
            / 'models'
            / f'{self.lattice_system}_1'
            / 'data'
            / f'hkl_ref_{bravais_lattice}.npy'
        )
        self.hkl_ref = np.load(data_path)
        self.hkl_ref_length = self.hkl_ref.shape[0]

        # Dimensionality of the reciprocal‑space parameter vector (xnn)
        dim_map = {'cubic': 1, 'tetragonal': 2, 'hexagonal': 2, 'orthorhombic': 3}
        self.uc_length = dim_map.get(self.lattice_system, 2)

        # Default optimisation parameters (matching the user specification)
        if opt_params is None:
            iteration_info = [
                {
                    'worker': 'deterministic',
                    'n_iterations': 1,
                    'triplet_opt': False,
                }
            ]
            self.opt_params = {
                'iteration_info': iteration_info,
                'convergence_testing': False,
                'redistribution_testing': False,
                'assignment_threshold': 0.95,
                'figure_of_merit': 'M20',
                'max_neighbors': 64,
                'neighbor_radius': 0.000026,
                'downsample_radius': 0.002,
                'minimum_uc': 2,
                'maximum_uc': 50,
            }
            if self.lattice_system in {'tetragonal', 'hexagonal'}:
                self.opt_params.update({
                    'max_neighbors': 52,
                    'neighbor_radius': 0.000213,
                    'downsample_radius': 0.0001,
                })
            elif self.lattice_system == 'orthorhombic':
                self.opt_params.update({
                    'max_neighbors': 46,
                    'neighbor_radius': 0.000338,
                    'downsample_radius': 0.0001,
                })
        else:
            self.opt_params = opt_params

        # Minimal placeholders required for ``run_common``
        self.rng = np.random.default_rng()
        self.fom = 'M20'
        self.q2_obs = None
        self.triplets = None
        self.n_peaks = None

    # ---------------------------------------------------------------------
    # Overridden candidate generation – guess‑and‑check
    # ---------------------------------------------------------------------
    def generate_candidates_rank(self):
        """Create a ``Candidates`` instance from all HKL permutations.

        The method builds a list of candidate reciprocal‑space vectors ``xnn``
        by solving ``q2 = H·xnn`` for each permutation of ``dim`` reference HKLs.
        """
        # Use the first ``dim`` observed peaks (assumed sorted low‑angle)
        dim = self.uc_length
        q2_guess = self.q2_obs[:self.n_peaks_guess]
        # Restrict reference HKLs to the first ``n_ref_hkl`` entries
        ref_hkls = self.hkl_ref[:self.n_ref_hkl_guess]
        # Generate all ordered permutations of size ``dim``
        hkl_permutations = np.stack(list(itertools.permutations(ref_hkls, dim)), axis=0)
        q2_permutations = np.stack(list(itertools.permutations(q2_guess, dim)), axis=0)
        n_q2 = q2_permutations.shape[0]
        n_templates = hkl_permutations.shape[0]
        hkl2 = get_hkl_matrix(hkl_permutations, self.lattice_system)
        candidate_xnn_all = np.zeros((n_templates*n_q2, self.uc_length))
        index = 0
        for template_index in range(n_templates):
            for q2_index in range(n_q2):
                # Solve linear least‑squares for xnn (shape ``dim``)
                #print(q2_guess.shape, hkl2[index].shape)
                candidate_xnn_all[index], *_ = np.linalg.lstsq(
                    hkl2[template_index], q2_permutations[q2_index], rcond=None
                )
                index += 1

        # Physical‑parameter cleanup using the existing utilities
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
        candidate_xnn_all = get_xnn_from_unit_cell(
            candidate_unit_cells_all,
            partial_unit_cell=True,
            lattice_system=self.lattice_system
            )

        self.sent_candidates = np.zeros(self.n_ranks, dtype=int)
        for rank_index in range(self.n_ranks):
            self.sent_candidates[rank_index] = candidate_xnn_all[rank_index::self.n_ranks].shape[0]
            if rank_index == self.root:
                candidate_xnn_rank = candidate_xnn_all[rank_index::self.n_ranks]
            else:
                self.comm.send(candidate_xnn_all[rank_index::self.n_ranks], dest=rank_index)
        return self.generate_candidates_common(candidate_xnn_rank)
