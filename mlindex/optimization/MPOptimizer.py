import numpy as np
from multiprocessing import Process, Queue

from mlindex.optimization.MPIOptimizer import OptimizerManager, OptimizerWorker
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn


class LocalComm:
    """Dummy MPI communicator for use during MPOptimizerManager.__init__ only.

    Broadcasts are no-ops (manager already has all data). Point-to-point methods
    raise NotImplementedError as a safety net — the MP subclasses override all
    methods that would call them.
    """
    def __init__(self, n_ranks):
        self._n_ranks = n_ranks

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self._n_ranks

    def bcast(self, data, root=0):
        return data

    def Bcast(self, data, root=0):
        return data

    def barrier(self):
        pass

    def Split(self, color, key):
        return self

    def send(self, *a, **kw):
        raise NotImplementedError("MP mode: use queues directly")

    def recv(self, *a, **kw):
        raise NotImplementedError("MP mode: use queues directly")

    def Send(self, *a, **kw):
        raise NotImplementedError("MP mode: use queues directly")

    def Recv(self, *a, **kw):
        raise NotImplementedError("MP mode: use queues directly")


class MPOptimizerManager(OptimizerManager):
    """Multiprocessing variant of OptimizerManager.

    Class-level queue attributes are set by setup_mp_optimizers() before any
    instances are constructed, then cleared after construction completes.
    """
    _mp_data_queues = None
    _mp_result_queues = None
    _mp_n_ranks = None

    def __init__(self, data_params, opt_params, rf_params, template_params,
                 integral_filter_params, random_params, bravais_lattice, comm, fom,
                 seed=12345):
        self._data_queues = MPOptimizerManager._mp_data_queues
        self._result_queues = MPOptimizerManager._mp_result_queues
        n_ranks = MPOptimizerManager._mp_n_ranks
        # comm arg is ignored; LocalComm lets OptimizerBase.__init__ run without MPI
        super().__init__(data_params, opt_params, rf_params, template_params,
                         integral_filter_params, random_params, bravais_lattice,
                         comm=LocalComm(n_ranks), fom=fom, seed=seed)
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
        candidate_xnn_all = self._generate_candidates_xnn()
        self.sent_candidates = np.zeros(self.n_ranks, dtype=int)
        for rank_index in range(self.n_ranks):
            self.sent_candidates[rank_index] = candidate_xnn_all[rank_index::self.n_ranks].shape[0]
            if rank_index == self.root:
                candidate_xnn_rank = candidate_xnn_all[rank_index::self.n_ranks]
            else:
                self._data_queues[rank_index].put(candidate_xnn_all[rank_index::self.n_ranks])
        return self.generate_candidates_common(candidate_xnn_rank)

    def downsample_candidates(self, candidates, n_top_candidates):
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


class MPOptimizerWorker(OptimizerWorker):
    """Multiprocessing variant of OptimizerWorker.

    Does NOT call super().__init__() — receives all initialization data
    from the manager via the data queue instead of MPI broadcast.
    """
    def __init__(self, data_queue, result_queue, rank, n_ranks, fom, seed):
        self._data_q = data_queue
        self._result_q = result_queue
        self.root = 0
        self.rank = rank
        self.n_ranks = n_ranks
        self.fom = fom
        self.rng = np.random.default_rng(seed)
        self.zero_error = False
        self.wavelength = None
        # Receive init tuple sent by MPOptimizerManager._init_workers()
        (self.lattice_system, self.bravais_lattice, self.opt_params,
         self.hkl_ref_length, self.hkl_ref, self.n_peaks) = self._data_q.get()

    def run_common(self, n_top_candidates):
        self.q2_obs[:] = self._data_q.get()
        self.triplets = self._data_q.get()
        self._run_loop(n_top_candidates)

    def generate_candidates_rank(self):
        return self.generate_candidates_common(self._data_q.get())

    def downsample_candidates(self, candidates, n_top_candidates):
        result = {
            'M20': candidates.best_M20,
            'Minfo': candidates.best_Minfo,
            'xnn': candidates.best_xnn,
            'n_indexed': candidates.n_indexed,
            'spacegroup': list(candidates.best_spacegroup),
        }
        if self.triplets is not None:
            result['n_indexed_triplets'] = candidates.n_indexed_triplets
            result['M_triplets'] = candidates.best_M_triplets
        self._result_q.put(result)

    def convergence_testing(self, candidates):
        self._result_q.put({'M20': candidates.best_M20, 'xnn': candidates.best_xnn})


def _mp_worker_fn(rank, n_ranks, data_queue, result_queue, task_queue, fom=None):
    """Module-level worker function (picklable for macOS spawn start method).

    Builds all 14 MPOptimizerWorker objects at startup (reusing imports across
    all Bravais lattices), then loops waiting for task signals.
    """
    bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',
                        'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    workers = {}
    try:
        for bl in bravais_lattices:
            workers[bl] = MPOptimizerWorker(data_queue, result_queue, rank, n_ranks,
                                            fom, seed=rank)
        while True:
            msg = task_queue.get()
            if msg == 'shutdown':
                break
            bl, n_top, zero_error, wavelength = msg
            workers[bl].run(zero_error=zero_error, wavelength=wavelength,
                            n_top_candidates=n_top)
    except Exception as e:
        result_queue.put(e)


def setup_mp_optimizers(n_procs, broadening_tag, n_candidates_scale, logger=None):
    """Spawn worker processes and construct manager optimizers for all 14 BLs.

    Returns (optimizers, processes, task_queues).
    Call shutdown_mp_workers(processes, task_queues) when done.
    """
    from mlindex.optimization.UtilitiesOptimizer import get_optimizers
    import mlindex
    from pathlib import Path
    from types import SimpleNamespace

    data_queues   = [Queue() for _ in range(n_procs)]
    result_queues = [Queue() for _ in range(n_procs)]
    task_queues   = [Queue() for _ in range(n_procs)]

    processes = []
    for r in range(1, n_procs):
        p = Process(target=_mp_worker_fn,
                    args=(r, n_procs, data_queues[r], result_queues[r], task_queues[r]))
        p.start()
        processes.append(p)

    # Inject MP context into MPOptimizerManager class before factory construction.
    # Workers drain init tuples from data_queues as managers are constructed.
    MPOptimizerManager._mp_data_queues   = data_queues
    MPOptimizerManager._mp_result_queues = result_queues
    MPOptimizerManager._mp_n_ranks       = n_procs

    bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',
                        'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    mp_organizers = {bl: SimpleNamespace(manager=0, workers=list(range(n_procs)),
                                         split_comm=None, color=None)
                     for bl in bravais_lattices}

    optimizers = get_optimizers(0, mp_organizers, broadening_tag, n_candidates_scale,
                                logger=logger, optimizer_class=MPOptimizerManager)

    # Clean up class-level injection
    MPOptimizerManager._mp_data_queues   = None
    MPOptimizerManager._mp_result_queues = None
    MPOptimizerManager._mp_n_ranks       = None

    return optimizers, processes, task_queues


def run_mp_bl(optimizer, bl, task_queues, q2, triplets, zero_error, wavelength, n_top):
    """Signal workers to start BL run, then run manager's share."""
    for r in range(1, len(task_queues)):
        task_queues[r].put((bl, n_top, zero_error, wavelength))
    optimizer.run(q2=q2, triplets=triplets, zero_error=zero_error,
                  wavelength=wavelength, n_top_candidates=n_top)


def shutdown_mp_workers(processes, task_queues):
    """Send shutdown signal to all workers and wait for them to exit."""
    for r in range(1, len(task_queues)):
        task_queues[r].put('shutdown')
    for p in processes:
        p.join()


def _mp_analytic_worker_fn(rank, n_ranks, data_queue, result_queue, task_queue, fom=None):
    """Worker function for the analytic optimizer (11 Bravais lattices, no ML models).

    Like _mp_worker_fn but restricted to the analytic BL set so workers don't
    block waiting for init data from unsupported lattices.
    """
    bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',
                        'oC', 'oF', 'oI', 'oP']
    workers = {}
    try:
        for bl in bravais_lattices:
            workers[bl] = MPOptimizerWorker(data_queue, result_queue, rank, n_ranks,
                                            fom, seed=rank)
        while True:
            msg = task_queue.get()
            if msg == 'shutdown':
                break
            bl, n_top, zero_error, wavelength = msg
            workers[bl].run(zero_error=zero_error, wavelength=wavelength,
                            n_top_candidates=n_top)
    except Exception as e:
        result_queue.put(e)


def setup_mp_analytic_optimizers(n_procs, n_peaks, n_ref_hkl_guess):
    """Spawn worker processes and construct MPAnalyticOptimizer managers for 11 analytic BLs.

    Returns (optimizers, processes, task_queues).
    Call shutdown_mp_workers(processes, task_queues) when done.
    """
    from mlindex.optimization.AnalyticOptimizer import MPAnalyticOptimizer

    bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',
                        'oC', 'oF', 'oI', 'oP']

    data_queues   = [Queue() for _ in range(n_procs)]
    result_queues = [Queue() for _ in range(n_procs)]
    task_queues   = [Queue() for _ in range(n_procs)]

    processes = []
    for r in range(1, n_procs):
        p = Process(target=_mp_analytic_worker_fn,
                    args=(r, n_procs, data_queues[r], result_queues[r], task_queues[r]))
        p.start()
        processes.append(p)

    # Inject MP context into MPAnalyticOptimizer class before construction.
    # Workers drain init tuples from data_queues as managers are constructed.
    MPAnalyticOptimizer._mp_data_queues   = data_queues
    MPAnalyticOptimizer._mp_result_queues = result_queues
    MPAnalyticOptimizer._mp_n_ranks       = n_procs

    optimizers = {}
    for bl in bravais_lattices:
        optimizers[bl] = MPAnalyticOptimizer(
            bravais_lattice=bl,
            comm=None,
            n_peaks=n_peaks,
            n_ref_hkl_guess=n_ref_hkl_guess[bl],
        )

    # Clean up class-level injection
    MPAnalyticOptimizer._mp_data_queues   = None
    MPAnalyticOptimizer._mp_result_queues = None
    MPAnalyticOptimizer._mp_n_ranks       = None

    return optimizers, processes, task_queues
