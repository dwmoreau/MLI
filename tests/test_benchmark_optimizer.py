"""The benchmark's recorder: what a run keeps beyond the twenty candidates it prints."""

import numpy as np
import pytest

from mlindex.model_training.BenchmarkOptimizer import BenchmarkOptimizer


def _recorder(**overrides):
    """A BenchmarkOptimizer carrying only what the recording path touches."""
    recorder = BenchmarkOptimizer.__new__(BenchmarkOptimizer)
    recorder.lattice_system = 'cubic'
    recorder.bravais_lattice = 'cP'
    recorder.n_ranks = 1
    recorder.zero_error = False
    recorder.n_peaks = 10
    recorder.hkl_ref_length = 500
    recorder.opt_params = {'downsample_radius': 1e-9, 'assignment_threshold': 0.95,
                           'prune_m20_threshold': 1.5}
    recorder.dump_context = {'entry_id': 'AAAAAA', 'condition_bundle': 'b1_error1_cont0',
                             'q2_digest': 'deadbeefdeadbeef'}
    recorder._records = []
    recorder.__dict__.update(overrides)
    return recorder


def _pool(m20):
    """One rank's return: distinct cells so nothing is collapsed."""
    n = len(m20)
    xnn = [np.arange(1.0, n + 1.0).reshape(n, 1)]
    return ([np.array(m20, dtype=float)], xnn, [np.arange(n)],
            [f'SG{i}' for i in range(n)])


def test_every_survivor_is_kept_and_ranked_over_all_of_them():
    """The indexer keeps twenty per lattice; the benchmark needs the rest, because a cell that
    was truncated away has no rank at all. Truncation is therefore a column, not a missing row."""
    recorder = _recorder()
    M20, xnn, n_indexed, spacegroup = _pool([10.0, 40.0, 20.0, 30.0])

    recorder._downsample_computation(M20, xnn, n_indexed, spacegroup, n_top_candidates=2)
    record, = recorder.drain()

    assert record['M20'].shape[0] == 4
    assert record['final_rank'].tolist() == [3, 0, 2, 1]
    assert record['in_top_n'].tolist() == [False, True, False, True]
    assert record['n_entering'] == 4
    assert record['candidate_id'].tolist() == [0, 1, 2, 3]
    # The manager still reports only its twenty -- here, two.
    assert recorder.top_M20.tolist() == [40.0, 30.0]


def test_a_survivor_keeps_its_own_spacegroup_and_cell():
    """The recorder reads the arrays the deduplication hands it, and the spacegroups travel as a
    list beside them. A row's label, cell and score have to describe one candidate."""
    recorder = _recorder()
    recorder.opt_params['downsample_radius'] = 1e-3
    # Rows 0 and 1 are near-duplicates, so one is collapsed away.
    xnn = [np.array([[1.0], [1.0000001], [5.0]])]
    M20 = [np.array([10.0, 20.0, 30.0])]
    n_indexed = [np.array([5, 6, 7])]
    spacegroup = ['A', 'B', 'C']

    recorder._downsample_computation(M20, xnn, n_indexed, spacegroup, n_top_candidates=20)
    record, = recorder.drain()

    by_score = dict(zip(record['M20'].tolist(), record['spacegroup']))
    assert by_score == {20.0: 'B', 30.0: 'C'}
    assert record['n_entering'] == 3
    assert record['M20'].shape[0] == 2


def test_a_pattern_that_cannot_name_itself_is_refused():
    """A candidate row joins to the truth on (entry_id, condition_bundle) and is checked against
    the entry table on q2_digest. A missing name does not fail loudly later: it joins to nothing,
    or worse, to the wrong crystal."""
    for missing in ('entry_id', 'condition_bundle', 'q2_digest'):
        context = {'entry_id': 'AAAAAA', 'condition_bundle': 'b', 'q2_digest': 'd'}
        del context[missing]
        recorder = _recorder(dump_context=context)
        M20, xnn, n_indexed, spacegroup = _pool([1.0, 2.0])
        with pytest.raises(ValueError, match=missing):
            recorder._downsample_computation(M20, xnn, n_indexed, spacegroup,
                                             n_top_candidates=20)


def test_the_digest_is_never_taken_from_the_optimizer_s_own_peak_list():
    """A manager truncates the peak list to what its lattice system is fitted on -- ten lines for
    cubic, twenty for the rest -- so a digest computed here would give one pattern several
    identities. Measured on a real run before this was refused: the same pattern digested
    8e0fc9e3... under mP and 48c37858... under cP. The driver's digest is used verbatim."""
    from mlindex.utilities.Digests import q2_digest

    recorder = _recorder()
    recorder.q2_obs = np.linspace(0.05, 0.5, 10)
    M20, xnn, n_indexed, spacegroup = _pool([1.0, 2.0])

    recorder._downsample_computation(M20, xnn, n_indexed, spacegroup, n_top_candidates=20)
    record, = recorder.drain()

    assert record['q2_digest'] == 'deadbeefdeadbeef'
    assert record['q2_digest'] != q2_digest(recorder.q2_obs)


def test_zero_error_refinement_is_refused():
    """The per-candidate zeropoint stays with the worker that fitted it and never reaches the
    manager, so a recorded cell would not reproduce the M20 the pipeline computed for it."""
    recorder = _recorder(zero_error=True)
    M20, xnn, n_indexed, spacegroup = _pool([1.0, 2.0])

    with pytest.raises(NotImplementedError, match='zero-error'):
        recorder._downsample_computation(M20, xnn, n_indexed, spacegroup, n_top_candidates=20)


def test_draining_empties_the_buffer():
    """A pool holds hundreds of patterns and a triclinic one produces thousands of survivors, so
    the driver takes the records per pattern rather than at the end."""
    recorder = _recorder()
    M20, xnn, n_indexed, spacegroup = _pool([1.0, 2.0])

    recorder._downsample_computation(M20, xnn, n_indexed, spacegroup, n_top_candidates=20)
    assert len(recorder.drain()) == 1
    assert recorder.drain() == []
