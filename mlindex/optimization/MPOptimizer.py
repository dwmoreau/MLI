import numpy as np
from multiprocessing import Process, Queue

from mlindex.optimization.MPIOptimizer import OptimizerManager, OptimizerWorker
from mlindex.optimization.MPIOptimizer import _collect_at_prune, _worker_payload
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn


BRAVAIS_LATTICES_ALL = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',
                        'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']


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
        # Both MP subclasses override run_common to replace the MPI broadcast, so the base
        # class's per-pattern reseeding has to be re-issued here or it never fires in
        # multiprocessing mode -- which is the mode the benchmark dump runs in. Left out, the
        # reseeding silently does nothing and the pool looks non-reproducible for reasons that
        # have nothing to do with the seeding scheme.
        self._reseed_for_pattern()
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
        results = []
        for r in range(1, self.n_ranks):
            result = self._result_queues[r].get()
            if isinstance(result, Exception):
                raise RuntimeError(f"Worker {r} failed: {result}") from result
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
        self._reseed_for_pattern()
        self._run_loop(n_top_candidates)

    def generate_candidates_rank(self):
        return self.generate_candidates_common(self._data_q.get())

    def downsample_candidates(self, candidates, n_top_candidates):
        self._result_q.put(_worker_payload(candidates))

    def convergence_testing(self, candidates):
        self._result_q.put({'M20': candidates.best_M20, 'xnn': candidates.best_xnn})


def _mp_worker_fn(rank, n_ranks, data_queue, result_queue, task_queue, fom=None, seed=12345,
                  bravais_lattices=None):
    """Module-level worker function (picklable for macOS spawn start method).

    Builds one MPOptimizerWorker per Bravais lattice at startup (reusing imports
    across all of them), then loops waiting for task signals.

    `bravais_lattices` must list the lattices in the SAME ORDER the manager
    constructs them: each worker constructor drains one init tuple from the
    shared data queue, so a different order silently pairs a worker with another
    lattice's hkl_ref. Default None means all 14, which is what
    `setup_mp_optimizers` builds; `setup_lattice_groups` passes its group's own
    subset, and `get_optimizers` iterates the organizer dict in insertion order.
    """
    if bravais_lattices is None:
        bravais_lattices = list(BRAVAIS_LATTICES_ALL)
    workers = {}
    try:
        for bl in bravais_lattices:
            workers[bl] = MPOptimizerWorker(data_queue, result_queue, rank, n_ranks,
                                            fom, seed=seed + rank)
        while True:
            msg = task_queue.get()
            if msg == 'shutdown':
                break
            bl, n_top, zero_error, wavelength, run_seed = msg
            if run_seed is not None:
                # Reseed per task, so a run can be seeded per (entry, Bravais lattice) rather
                # than once per pool. Campaign 1 seeded once and advanced with every entry, which
                # meant no subset of a run could be regenerated comparably and forced a
                # within-run restriction on every result in its final phase (PROTOCOL section 6).
                # `+ rank` keeps the ranks on distinct streams, as at construction.
                workers[bl].rng = np.random.default_rng(run_seed + rank)
            workers[bl].run(zero_error=zero_error, wavelength=wavelength,
                            n_top_candidates=n_top)
    except Exception as e:
        result_queue.put(e)


def setup_mp_optimizers(n_procs, broadening_tag, n_candidates_scale, logger=None, seed=12345,
                        options=None):
    """Spawn worker processes and construct manager optimizers for all 14 BLs.

    Returns (optimizers, processes, task_queues).
    Call shutdown_mp_workers(processes, task_queues) when done.

    `options` is merged into every lattice system's opt_params by the factories, and
    `_init_workers` then ships the merged dict to each worker -- so a research setting reaches
    the whole pool without a second channel. Default None leaves every parameter exactly as
    shipped. This is the opt_params route, not a user-facing one: the prune threshold in
    particular is deliberately not a CLI option (C2-F-008).
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
                    args=(r, n_procs, data_queues[r], result_queues[r], task_queues[r]),
                    kwargs={'seed': seed})
        p.start()
        processes.append(p)

    # Inject MP context into MPOptimizerManager class before factory construction.
    # Workers drain init tuples from data_queues as managers are constructed.
    MPOptimizerManager._mp_data_queues   = data_queues
    MPOptimizerManager._mp_result_queues = result_queues
    MPOptimizerManager._mp_n_ranks       = n_procs

    bravais_lattices = list(BRAVAIS_LATTICES_ALL)
    mp_organizers = {bl: SimpleNamespace(manager=0, workers=list(range(n_procs)),
                                         split_comm=None, color=None)
                     for bl in bravais_lattices}

    optimizers = get_optimizers(0, mp_organizers, broadening_tag, n_candidates_scale,
                                logger=logger, optimizer_class=MPOptimizerManager, seed=seed,
                                options=options)

    # Clean up class-level injection
    MPOptimizerManager._mp_data_queues   = None
    MPOptimizerManager._mp_result_queues = None
    MPOptimizerManager._mp_n_ranks       = None

    return optimizers, processes, task_queues


def run_mp_bl(optimizer, bl, task_queues, q2, zero_error, wavelength, n_top, run_seed=None):
    """Signal workers to start BL run, then run manager's share.

    `run_seed` reseeds every rank for this one task. Default None leaves the streams exactly
    where the previous task left them, which is the shipped behaviour.
    """
    for r in range(1, len(task_queues)):
        task_queues[r].put((bl, n_top, zero_error, wavelength, run_seed))
    if run_seed is not None:
        optimizer.rng = np.random.default_rng(run_seed)
    optimizer.run(q2=q2, zero_error=zero_error,
                  wavelength=wavelength, n_top_candidates=n_top)


def shutdown_mp_workers(processes, task_queues):
    """Send shutdown signal to all workers and wait for them to exit."""
    for r in range(1, len(task_queues)):
        task_queues[r].put('shutdown')
    for p in processes:
        p.join()


# ---------------------------------------------------------------------------
# Lattice groups: parallelism ACROSS Bravais lattices, not only across candidates.
#
# `setup_mp_optimizers` above gives every process the same lattice at the same
# time, so only the manager -- the one process holding models -- can generate
# candidates, and every worker blocks in `generate_candidates_rank` while it
# does. Measured on a 20-peak pattern, that generation phase is 15.6 s of a
# 20.5 s pattern at eight processes and does not shrink as processes are added,
# which caps the whole program at about 2x however many cores it is given.
#
# A lattice group owns a subset of the lattices and holds only their models, so
# several groups generate at once. Generation for a lattice depends only on that
# lattice's own generator objects and the observed peak list -- verified: mP's
# candidates are bit-identical whether mP runs alone or after twelve other
# lattices -- so moving a lattice between processes does not perturb it, PROVIDED
# the assignment is static. The generator streams still advance once per pattern,
# so a lattice must stay with the same group for every pattern of a run.
# ---------------------------------------------------------------------------


def _build_group_optimizers(bl_list, group_size, data_queues, result_queues,
                            broadening_tag, n_candidates_scale, seed, options,
                            logger=None):
    """Construct the manager optimizers for one lattice group.

    Only the lattices in `bl_list` are built, so a group loads only the models it
    will use: 12-222 MB per lattice against 1.29 GB for all fourteen.
    """
    from mlindex.optimization.UtilitiesOptimizer import get_optimizers
    from types import SimpleNamespace

    MPOptimizerManager._mp_data_queues = data_queues
    MPOptimizerManager._mp_result_queues = result_queues
    MPOptimizerManager._mp_n_ranks = group_size
    organizers = {bl: SimpleNamespace(manager=0, workers=list(range(group_size)),
                                      split_comm=None, color=None)
                  for bl in bl_list}
    try:
        return get_optimizers(0, organizers, broadening_tag, n_candidates_scale,
                              logger=logger, optimizer_class=MPOptimizerManager,
                              seed=seed, options=options)
    finally:
        # Cleared even on failure. These are class attributes, so a manager left
        # pointing at a dead group's queues would be inherited by the next
        # construction in this process.
        MPOptimizerManager._mp_data_queues = None
        MPOptimizerManager._mp_result_queues = None
        MPOptimizerManager._mp_n_ranks = None


def _group_result(optimizer):
    """The five arrays `run.py` needs back from a lattice, ready to pickle."""
    return {
        'top_unit_cell': optimizer.top_unit_cell,
        'top_M20': optimizer.top_M20,
        'top_Minfo': optimizer.top_Minfo,
        'top_spacegroup': optimizer.top_spacegroup,
        'top_n_indexed': optimizer.top_n_indexed,
        }


def _mp_group_manager_fn(bl_list, group_size, data_queues, result_queues, task_queues,
                         control_queue, output_queue, broadening_tag,
                         n_candidates_scale, seed, options):
    """Manager process for one lattice group (module level, so spawn can pickle it).

    Holds the models for `bl_list` and drives the group's own refinement workers
    through `run_mp_bl`, exactly as the single manager does today. What is new is
    only that several of these run at once.
    """
    try:
        optimizers = _build_group_optimizers(
            bl_list, group_size, data_queues, result_queues,
            broadening_tag, n_candidates_scale, seed, options)
        output_queue.put(('ready', None))
        while True:
            msg = control_queue.get()
            if msg == 'shutdown':
                break
            if msg[0] == 'promote':
                # Imported here, not at module scope: run.py imports this
                # module, and cctbx costs a second to import in a process that
                # may never be asked to promote anything.
                from mlindex.command_line.run import promote_entries
                output_queue.put(('promoted', promote_entries(msg[1], delta=msg[2])))
                continue
            _, q2, zero_error, wavelength, n_top, run_seed = msg
            payload = {}
            for bl in bl_list:
                run_mp_bl(optimizers[bl], bl, task_queues, q2=q2,
                          zero_error=zero_error, wavelength=wavelength,
                          n_top=n_top, run_seed=run_seed)
                payload[bl] = _group_result(optimizers[bl])
            output_queue.put(('results', payload))
    except Exception as e:
        # Reported rather than raised, so the coordinator fails with this
        # exception as the cause instead of hanging on a queue nobody will fill.
        output_queue.put(('error', e))
    finally:
        for r in range(1, group_size):
            task_queues[r].put('shutdown')


def setup_lattice_groups(assignment, broadening_tag, n_candidates_scale,
                         logger=None, seed=12345, options=None):
    """Spawn one process group per (lattice list, group size) pair in `assignment`.

    The FIRST entry is run by the calling process, so exactly `sum(group_size)`
    processes do the work and the caller does not idle while the others run.
    Returns (groups, processes) for `run_lattice_groups` and
    `shutdown_lattice_groups`.
    """
    groups = []
    processes = []
    for group_index, (bl_list, group_size) in enumerate(assignment):
        bl_list = list(bl_list)
        data_queues = [Queue() for _ in range(group_size)]
        result_queues = [Queue() for _ in range(group_size)]
        task_queues = [Queue() for _ in range(group_size)]
        # The group's refinement workers, ranks 1..group_size-1. Spawned before
        # the manager is constructed, because each blocks in its constructor on
        # the init tuple the manager's `_init_workers` sends.
        for r in range(1, group_size):
            p = Process(target=_mp_worker_fn,
                        args=(r, group_size, data_queues[r], result_queues[r],
                              task_queues[r]),
                        kwargs={'seed': seed, 'bravais_lattices': bl_list})
            p.start()
            processes.append(p)
        # data_queues and result_queues are held here even though only the
        # group's own manager reads them. A Queue passed to Process() is
        # rebuilt from a named semaphore in the child, so if the parent drops
        # its last reference the semaphore is reclaimed before the child
        # finishes unpickling and every group dies with
        # `SemLock._rebuild ... FileNotFoundError` while the coordinator waits
        # forever for a `ready` that will never come.
        group = {'bravais_lattices': bl_list, 'size': group_size,
                 'data_queues': data_queues, 'result_queues': result_queues,
                 'task_queues': task_queues, 'control_queue': None,
                 'output_queue': None, 'optimizers': None}
        if group_index == 0:
            group['optimizers'] = _build_group_optimizers(
                bl_list, group_size, data_queues, result_queues,
                broadening_tag, n_candidates_scale, seed, options, logger=logger)
        else:
            control_queue = Queue()
            output_queue = Queue()
            p = Process(target=_mp_group_manager_fn,
                        args=(bl_list, group_size, data_queues, result_queues,
                              task_queues, control_queue, output_queue,
                              broadening_tag, n_candidates_scale, seed, options))
            p.start()
            processes.append(p)
            group['control_queue'] = control_queue
            group['output_queue'] = output_queue
        groups.append(group)

    # Wait out every remote group's model load here rather than inside the first
    # pattern, so a caller timing patterns is not also timing the load.
    for group in groups[1:]:
        tag, payload = group['output_queue'].get()
        if tag == 'error':
            raise RuntimeError('Lattice group failed while loading models') from payload
    return groups, processes


def run_lattice_groups(groups, q2, zero_error, wavelength, n_top, run_seed=None):
    """Run one pattern across every group concurrently.

    Returns {bravais_lattice: `_group_result` dict}. `run_seed` reseeds every
    rank for this one pattern, as in `run_mp_bl`.
    """
    message = ('run', q2, zero_error, wavelength, n_top, run_seed)
    for group in groups[1:]:
        group['control_queue'].put(message)

    results = {}
    local = groups[0]
    for bl in local['bravais_lattices']:
        run_mp_bl(local['optimizers'][bl], bl, local['task_queues'], q2=q2,
                  zero_error=zero_error, wavelength=wavelength, n_top=n_top,
                  run_seed=run_seed)
        results[bl] = _group_result(local['optimizers'][bl])

    for group in groups[1:]:
        tag, payload = group['output_queue'].get()
        if tag == 'error':
            raise RuntimeError('Lattice group failed during optimization') from payload
        results.update(payload)
    return results


def promote_over_groups(groups, entries, delta):
    """Run `run.promote_entries` over `entries`, split across the lattice groups.

    Called from `_write_output` while the groups are still alive. Each is idle
    at this point -- the optimization is over -- and cctbx holds the GIL, so
    processes are the only way to overlap this work at all.

    The entries are dealt out round-robin and reassembled by the same stride, so
    the promoted list is in the input order regardless of how many groups ran it
    and the result does not depend on `--nproc`.
    """
    n_groups = len(groups)
    if n_groups < 2 or len(entries) < 2 * n_groups:
        from mlindex.command_line.run import promote_entries
        return promote_entries(entries, delta=delta)

    for index, group in enumerate(groups[1:], start=1):
        group['control_queue'].put(('promote', entries[index::n_groups], delta))

    from mlindex.command_line.run import promote_entries
    promoted = {0: promote_entries(entries[0::n_groups], delta=delta)}
    for index, group in enumerate(groups[1:], start=1):
        tag, payload = group['output_queue'].get()
        if tag == 'error':
            raise RuntimeError('Lattice group failed during promotion') from payload
        promoted[index] = payload

    out = [None] * len(entries)
    for index in range(n_groups):
        out[index::n_groups] = promoted[index]
    return out


def shutdown_lattice_groups(groups, processes):
    """Stop every group manager and refinement worker, then join."""
    for group in groups[1:]:
        group['control_queue'].put('shutdown')
    local = groups[0]
    for r in range(1, local['size']):
        local['task_queues'][r].put('shutdown')
    for p in processes:
        p.join()


def _mp_analytic_worker_fn(rank, n_ranks, data_queue, result_queue, task_queue,
                           bravais_lattices, fom=None, seed=12345):
    workers = {}
    try:
        for bl in bravais_lattices:
            workers[bl] = MPOptimizerWorker(data_queue, result_queue, rank, n_ranks,
                                            fom, seed=seed + rank)
        while True:
            msg = task_queue.get()
            if msg == 'shutdown':
                break
            bl, n_top, zero_error, wavelength, run_seed = msg
            if run_seed is not None:
                # Reseed per task, so a run can be seeded per (entry, Bravais lattice) rather
                # than once per pool. Campaign 1 seeded once and advanced with every entry, which
                # meant no subset of a run could be regenerated comparably and forced a
                # within-run restriction on every result in its final phase (PROTOCOL section 6).
                # `+ rank` keeps the ranks on distinct streams, as at construction.
                workers[bl].rng = np.random.default_rng(run_seed + rank)
            workers[bl].run(zero_error=zero_error, wavelength=wavelength,
                            n_top_candidates=n_top)
    except Exception as e:
        result_queue.put(e)


def setup_mp_analytic_optimizers(n_procs, n_peaks, n_ref_hkl_guess, bravais_lattices, seed=12345):
    """Spawn worker processes and construct MPAnalyticOptimizer managers for the given BLs.

    Returns (optimizers, processes, task_queues).
    Call shutdown_mp_workers(processes, task_queues) when done.

    `options` is merged into every lattice system's opt_params by the factories, and
    `_init_workers` then ships the merged dict to each worker -- so a research setting reaches
    the whole pool without a second channel. Default None leaves every parameter exactly as
    shipped. This is the opt_params route, not a user-facing one: the prune threshold in
    particular is deliberately not a CLI option (C2-F-008).
    """
    from mlindex.optimization.AnalyticOptimizer import MPAnalyticOptimizer

    data_queues   = [Queue() for _ in range(n_procs)]
    result_queues = [Queue() for _ in range(n_procs)]
    task_queues   = [Queue() for _ in range(n_procs)]

    processes = []
    for r in range(1, n_procs):
        p = Process(target=_mp_analytic_worker_fn,
                    args=(r, n_procs, data_queues[r], result_queues[r], task_queues[r],
                          bravais_lattices),
                    kwargs={'seed': seed})
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
            seed=seed,
        )

    # Clean up class-level injection
    MPAnalyticOptimizer._mp_data_queues   = None
    MPAnalyticOptimizer._mp_result_queues = None
    MPAnalyticOptimizer._mp_n_ranks       = None

    return optimizers, processes, task_queues
