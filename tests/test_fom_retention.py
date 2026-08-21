"""S14 -- the instrumentation that makes the prune and the dedup tiebreak measurable.

Two cuts delete candidates before ranking sees them and neither had ever been measured:
`prune_below_m20` (Q31, rebuild row R1) and the highest-M20 deduplication tiebreak
(F-065, rebuild row R2). These tests pin the instrumentation, and -- more importantly --
pin that production behaviour is unchanged when it is switched off.
"""
import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomBenchmark
from mlindex.optimization.MPIOptimizer import OptimizerManager


def test_the_new_opt_params_default_to_the_production_behaviour():
    """The defaults are the whole safety argument: an optimizer built without options
    must behave exactly as it did before these keys existed."""
    import inspect
    source = inspect.getsource(OptimizerManager.__init__)
    assert "'prune_m20_threshold': 5.0" in source
    assert "'dump_predownsample': None" in source


def test_get_optimizers_accepts_options_so_a_worker_key_can_reach_the_workers():
    """prune_m20_threshold is read inside Candidates, which runs on every rank, and
    opt_params is shipped to workers at construction. A key set afterwards would apply to
    the manager's candidates only -- so the options path has to exist."""
    import inspect
    from mlindex.optimization.UtilitiesOptimizer import get_optimizers
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers
    assert 'options' in inspect.signature(get_optimizers).parameters
    assert 'options' in inspect.signature(setup_mp_optimizers).parameters


def _candidates(m20):
    """A Candidates stub carrying only what prune_below_m20 touches."""
    from mlindex.optimization.Candidates import Candidates
    stub = Candidates.__new__(Candidates)
    stub.best_M20 = np.asarray(m20, dtype=float)
    stub.best_xnn = np.arange(len(m20) * 2, dtype=float).reshape(len(m20), 2)
    stub.best_hkl = np.zeros((len(m20), 3, 3), dtype=float)
    stub.zero_error = False
    stub.retained = {}
    stub.retention_extra = ()
    stub.retained_by = np.zeros(len(m20), dtype=int)
    return stub


def test_prune_records_the_value_it_actually_tested():
    """The stored M20 is the post-assignment value; the prune tests the pre-extinction-group
    one (F-049). Without keeping the tested value the cut cannot be reconstructed."""
    stub = _candidates([9.0, 4.0, 7.0, 1.0])
    stub.prune_below_m20(threshold=5.0)
    assert stub.m20_at_prune.tolist() == [9.0, 7.0]
    assert stub.n == 2


def test_prune_at_zero_keeps_everything_and_still_records_the_scores():
    """Threshold 0 is how Q31 asks the question, so it must be a genuine no-op."""
    stub = _candidates([9.0, 4.0, 7.0, 1.0])
    stub.prune_below_m20(threshold=0.0)
    assert stub.n == 4
    assert stub.m20_at_prune.tolist() == [9.0, 4.0, 7.0, 1.0]


def test_the_fallback_keeps_one_candidate_that_failed_the_threshold():
    """`if not np.any(keep): keep[argmax] = True`. On the hard stratum this fallback, not
    the threshold, is what populates most lattices -- so a reconstruction of "would have
    been deleted" is an upper bound, and this is the reason why."""
    stub = _candidates([1.0, 2.0, 3.0])
    stub.prune_below_m20(threshold=5.0)
    assert stub.n == 1
    assert stub.m20_at_prune.tolist() == [3.0]


def test_the_predownsample_frame_round_trips():
    record = {
        'bravais_lattice': 'mP', 'lattice_system': 'monoclinic', 'q2_digest': 'abc',
        'context': {'entry_id': 'E1'}, 'n_peaks': 20, 'hkl_ref_length': 100,
        'n_entering': 3, 'prune_m20_threshold': 0.0, 'downsample_radius': 1e-4,
        'xnn': np.arange(12, dtype=float).reshape(3, 4),
        'M20': np.array([1.0, 2.0, 3.0]), 'Minfo': np.array([4.0, 5.0, 6.0]),
        'n_indexed': np.array([1, 2, 3]),
        'm20_at_prune': np.array([1.5, 2.5, 3.5]),
        'retained_by': np.array([0, 0, 1]),
        'spacegroup': ['P2/m', 'P21/m', 'P2/m'],
        }
    frame = FomBenchmark.predownsample_records_to_frame([record])
    assert list(frame.columns) == list(FomBenchmark.PREDOWNSAMPLE_COLUMNS)
    assert frame.shape[0] == 3
    assert frame['m20_at_prune'].tolist() == [1.5, 2.5, 3.5]
    # The provenance column is what makes item 1's ceiling a within-run restriction.
    assert frame['retained_by'].tolist() == [0, 0, 1]
    assert frame['spacegroup'].tolist() == ['P2/m', 'P21/m', 'P2/m']
    assert frame['xnn'].iloc[1].tolist() == [4.0, 5.0, 6.0, 7.0]


def test_an_empty_predownsample_frame_still_has_the_schema():
    """A lattice returning nothing is normal (cF and cI often do)."""
    frame = FomBenchmark.predownsample_records_to_frame([])
    assert list(frame.columns) == list(FomBenchmark.PREDOWNSAMPLE_COLUMNS)
    assert frame.empty


def test_the_spacegroup_list_is_filtered_with_the_arrays_it_is_aligned_to():
    """`sort_indices` index the NaN-filtered arrays. The spacegroup list was left
    unfiltered, so every entry after the first dropped row was attached to the wrong
    candidate. Guarded here because the failure is silent and the column is S08's most
    important feature."""
    import inspect
    from mlindex.optimization import MPIOptimizer
    source = inspect.getsource(MPIOptimizer.OptimizerManager._downsample_computation)
    filter_line = source.index('best_n_indexed_all = best_n_indexed_all[good_indices]')
    sort_line = source.index('best_spacegroup_all = [best_spacegroup_all[i] for i in sort_indices]')
    zipped = source.index('zip(best_spacegroup_all, good_indices)')
    assert filter_line < zipped < sort_line


# ---------------------------------------------------------------------------
# S14 item 1 -- multi-FOM iterate retention.
#
# The iteration loop keeps one iterate per candidate, the arg-max over M20. If the final
# score is not M20 that is the wrong iterate, and the correct cell leaves the pool before
# ranking ever sees it. These pin that the mechanism does what it claims *and* that it is
# invisible when it is off, which is the whole safety argument.
# ---------------------------------------------------------------------------

def _cubic_hkl_ref():
    hkl = np.array([[h, k, l] for h in range(4) for k in range(4) for l in range(4)
                    if (h, k, l) != (0, 0, 0)], dtype=float)
    return hkl[np.argsort((hkl**2).sum(axis=1))]


def _real_candidates(retention_foms=('M20',), xnn=(0.041, 0.038, 0.050),
                     zero_error=False):
    """A genuine Candidates -- constructor, Q2Calculator and all -- on a cubic cell.

    Cubic keeps xnn one-dimensional and the reference list small enough to write down, so
    this exercises the real assign_hkls path without needing the model files.
    """
    from mlindex.optimization.Candidates import Candidates
    hkl_ref = _cubic_hkl_ref()
    q2_obs = np.sort(np.unique((hkl_ref**2).sum(axis=1)))[:10]*0.04
    opt_params = {'minimum_uc': 2, 'maximum_uc': 500, 'assignment_threshold': 0.5,
                  'figure_of_merit': 'M20', 'retention_foms': tuple(retention_foms)}
    return Candidates(
        q2_obs=q2_obs, xnn=np.asarray(xnn, dtype=float)[:, np.newaxis], hkl_ref=hkl_ref,
        lattice_system='cubic', bravais_lattice='cP', opt_params=opt_params,
        rng=np.random.default_rng(0), fom=None, zero_error=zero_error, wavelength=None,
        )


def test_retention_defaults_to_m20_only_and_keeps_no_extra_state():
    candidates = _real_candidates()
    assert candidates.retention_foms == ('M20',)
    assert candidates.retention_extra == ()
    assert candidates.retained == {}
    assert candidates.retention_values == {}


def test_the_new_retention_keys_default_to_the_production_behaviour():
    """As with prune_m20_threshold: an optimizer built without options must behave exactly
    as it did before these keys existed."""
    import inspect
    source = inspect.getsource(OptimizerManager.__init__)
    assert "'retention_foms': ('M20',)" in source
    assert "'dedup_tiebreak_foms': ('M20',)" in source


def test_a_retained_merit_is_computed_on_the_arrays_assign_hkls_already_holds():
    candidates = _real_candidates(retention_foms=('M20', 'Minfo'))
    assert list(candidates.retention_values) == ['Minfo']
    assert candidates.retention_values['Minfo'].shape == candidates.M20.shape
    # The mechanism only exists because the two merits disagree about which cell is best.
    assert np.argmax(candidates.retention_values['Minfo']) != np.argmax(candidates.M20)


def test_an_unknown_retained_merit_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match='unknown retention FOM'):
        _real_candidates(retention_foms=('M20', 'M_nonesuch'))


def test_m20_cannot_be_dropped_from_the_retention_set():
    """Every other array in Candidates is aligned to the M20 track."""
    with pytest.raises(ValueError, match='must contain'):
        _real_candidates(retention_foms=('Minfo',))


def test_retention_refuses_the_zero_error_path_rather_than_approximating_it():
    """correct_zero_error leaves reciprocal_unit_cell one Gauss-Newton step stale and the
    per-candidate zeropoint would have to be carried per track."""
    with pytest.raises(NotImplementedError):
        _real_candidates(retention_foms=('M20', 'Minfo'), zero_error=True)


def _retention_stub(m20, minfo_track_xnn, m20_track_xnn, track_m20=None):
    """A Candidates carrying only what prune_below_m20 and merge_retained_iterates read."""
    from mlindex.optimization.Candidates import Candidates
    stub = Candidates.__new__(Candidates)
    n = len(m20)
    stub.best_M20 = np.asarray(m20, dtype=float)
    stub.best_xnn = np.asarray(m20_track_xnn, dtype=float)
    stub.best_hkl = np.zeros((n, 3, 3), dtype=float)
    stub.m20_at_prune = np.asarray(m20, dtype=float)
    stub.retained_by = np.zeros(n, dtype=int)
    stub.zero_error = False
    stub.retention_extra = ('Minfo',)
    stub.retained = {'Minfo': {
        'fom': np.arange(n, dtype=float),
        'xnn': np.asarray(minfo_track_xnn, dtype=float),
        'hkl': np.zeros((n, 3, 3), dtype=float),
        'M20': np.asarray(m20 if track_m20 is None else track_m20, dtype=float),
        }}
    stub.n = n
    return stub


def test_merging_appends_the_iterate_the_other_merit_kept():
    stub = _retention_stub(
        m20=[9.0, 8.0],
        m20_track_xnn=[[1.0], [2.0]],
        minfo_track_xnn=[[1.0], [7.0]],   # row 0 agrees, row 1 does not
        track_m20=[9.0, 3.0],
        )
    stub.merge_retained_iterates()
    assert stub.n == 3
    assert stub.best_xnn[:, 0].tolist() == [1.0, 2.0, 7.0]
    # The appended row carries *its own* M20, not the merit it was retained under, so
    # everything downstream that reads best_M20 stays interpretable.
    assert stub.best_M20.tolist() == [9.0, 8.0, 3.0]


def test_a_retained_iterate_identical_to_the_m20_one_is_not_duplicated():
    """correct_off_by_two's `best_mf_index != 0` test, applied to the cell."""
    stub = _retention_stub(
        m20=[9.0, 8.0],
        m20_track_xnn=[[1.0], [2.0]],
        minfo_track_xnn=[[1.0], [2.0]],
        )
    stub.merge_retained_iterates()
    assert stub.n == 2
    assert stub.best_xnn[:, 0].tolist() == [1.0, 2.0]


def test_an_appended_row_inherits_its_parents_m20_at_prune():
    """Had the parent been cut at the prune, the row would never have existed -- the same
    reasoning correct_off_by_two applies to its off-by-two rows."""
    stub = _retention_stub(
        m20=[9.0, 8.0],
        m20_track_xnn=[[1.0], [2.0]],
        minfo_track_xnn=[[5.0], [7.0]],
        )
    stub.m20_at_prune = np.array([9.5, 8.5])
    stub.merge_retained_iterates()
    assert stub.m20_at_prune.tolist() == [9.5, 8.5, 9.5, 8.5]
    assert stub.m20_at_prune.shape[0] == stub.best_xnn.shape[0]


def test_merging_is_a_no_op_when_only_m20_is_retained():
    stub = _candidates([9.0, 8.0])
    before = stub.best_xnn.copy()
    stub.n = 2
    stub.merge_retained_iterates()
    assert stub.n == 2
    assert np.array_equal(stub.best_xnn, before)


def test_the_prune_keeps_the_retention_tracks_row_aligned():
    """merge_retained_iterates attributes each appended row to a parent by position, so a
    track that fell out of step with best_xnn would credit the wrong candidate."""
    stub = _retention_stub(
        m20=[9.0, 1.0, 7.0],
        m20_track_xnn=[[1.0], [2.0], [3.0]],
        minfo_track_xnn=[[10.0], [20.0], [30.0]],
        )
    stub.prune_below_m20(threshold=5.0)
    assert stub.best_xnn[:, 0].tolist() == [1.0, 3.0]
    assert stub.retained['Minfo']['xnn'][:, 0].tolist() == [10.0, 30.0]
    assert stub.retained['Minfo']['fom'].tolist() == [0.0, 2.0]
    stub.merge_retained_iterates()
    assert stub.best_xnn[:, 0].tolist() == [1.0, 3.0, 10.0, 30.0]


def test_each_track_is_the_running_arg_max_of_its_own_merit():
    """The invariant, on the real loop. Not that the tracks *differ* -- on a toy cubic
    problem the search converges and they agree, which is the mechanism costing nothing
    rather than the mechanism being broken."""
    candidates = _real_candidates(retention_foms=('M20', 'Minfo'))
    iteration_info = {'worker': 'random_subsampling', 'n_iterations': 1, 'n_peaks': 10,
                      'n_drop': 4, 'uniform_sampling': True}
    for _ in range(20):
        candidates.random_subsampling(iteration_info)
        assert np.all(candidates.retained['Minfo']['fom']
                      >= candidates.retention_values['Minfo'])
        assert np.all(candidates.best_M20 >= candidates.M20)


def test_the_track_follows_its_own_merit_and_not_m20():
    """Directly: an iterate that is worse on M20 and better on Minfo is kept by the Minfo
    track and rejected by the M20 one. This is the whole of item 1 in four lines."""
    candidates = _real_candidates(retention_foms=('M20', 'Minfo'))
    candidates.best_M20 = np.array([10.0, 10.0, 10.0])
    candidates.retained['Minfo']['fom'] = np.array([1.0, 1.0, 1.0])
    candidates.retained['Minfo']['xnn'] = np.full((3, 1), -1.0)
    candidates.best_xnn = np.full((3, 1), -2.0)
    candidates.M20 = np.array([5.0, 5.0, 5.0])          # worse on M20
    candidates.retention_values = {'Minfo': np.array([9.0, 9.0, 9.0])}   # better on Minfo
    candidates.xnn = np.full((3, 1), 7.0)
    candidates.hkl = np.zeros((3, 10, 3))

    candidates._update_retained_iterates()

    assert candidates.retained['Minfo']['xnn'][:, 0].tolist() == [7.0, 7.0, 7.0]
    assert candidates.retained['Minfo']['M20'].tolist() == [5.0, 5.0, 5.0]
    assert candidates.best_xnn[:, 0].tolist() == [-2.0, -2.0, -2.0]


# ---------------------------------------------------------------------------
# S14 item 2 -- the deduplication tiebreak (F-065, rebuild row R2).
# ---------------------------------------------------------------------------

def _chunk(xnn, M20, Minfo, n_indexed, spacegroup, radius, tiebreak=None):
    from mlindex.optimization.MPIOptimizer import _downsample_chunk
    args = (np.asarray(xnn, dtype=float), np.asarray(M20, dtype=float),
            np.asarray(Minfo, dtype=float), np.asarray(n_indexed, dtype=int),
            list(spacegroup), radius)
    if tiebreak is not None:
        args = args + (tuple(tiebreak),)
    return _downsample_chunk(args)


def test_the_chunk_still_accepts_the_six_element_argument_tuple():
    """tools/repro_downsample.py pickles captured chunk arguments; the bit-identity
    harness has to keep replaying against the current code."""
    result = _chunk([[0.0], [1e-12], [5.0]], [1.0, 9.0, 2.0], [0.0, 0.0, 0.0],
                    [1, 1, 1], ['A', 'B', 'C'], 1e-9)
    assert result[1].tolist() == [2.0, 9.0]


def test_the_default_tiebreak_is_the_production_one():
    without = _chunk([[0.0], [1e-12], [5.0]], [1.0, 9.0, 2.0], [7.0, 0.0, 0.0],
                     [1, 1, 1], ['A', 'B', 'C'], 1e-9)
    with_default = _chunk([[0.0], [1e-12], [5.0]], [1.0, 9.0, 2.0], [7.0, 0.0, 0.0],
                          [1, 1, 1], ['A', 'B', 'C'], 1e-9, tiebreak=('M20',))
    assert np.array_equal(without[0], with_default[0])
    assert np.array_equal(without[1], with_default[1])
    assert without[4] == with_default[4]


def test_a_second_merit_rescues_the_member_the_m20_tiebreak_deletes():
    """Two cells inside one neighbourhood: M20 prefers the second, Minfo the first. The
    production rule deletes the first outright."""
    xnn = [[0.0], [1e-12], [5.0]]
    M20 = [1.0, 9.0, 2.0]
    Minfo = [7.0, 0.0, 0.0]
    baseline = _chunk(xnn, M20, Minfo, [1, 1, 1], ['A', 'B', 'C'], 1e-9)
    assert baseline[1].tolist() == [2.0, 9.0]
    assert 'A' not in baseline[4]

    rescued = _chunk(xnn, M20, Minfo, [1, 1, 1], ['A', 'B', 'C'], 1e-9,
                     tiebreak=('M20', 'Minfo'))
    # The production survivors are the prefix, unchanged; the rescue is appended.
    assert rescued[1].tolist() == [2.0, 9.0, 1.0]
    assert rescued[4] == ['C', 'B', 'A']


def test_a_rescued_point_is_frozen_and_cannot_collapse_against_its_own_winner():
    """The rescued point sits inside the winner's radius by construction. If it went back
    into the live set the collapse would fire again and one of them would be deleted --
    or, with both re-appended, never terminate."""
    xnn = [[0.0], [1e-12], [2e-12], [5.0]]
    result = _chunk(xnn, [1.0, 9.0, 3.0, 2.0], [7.0, 0.0, 0.0, 0.0], [1, 1, 1, 1],
                    ['A', 'B', 'C', 'D'], 1e-9, tiebreak=('M20', 'Minfo'))
    assert sorted(result[4]) == ['A', 'B', 'D']


def test_an_unknown_tiebreak_merit_is_refused_on_the_manager_not_in_a_thread():
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    manager = OptimizerManager.__new__(OptimizerManager)
    manager.lattice_system = 'cubic'
    manager.n_ranks = 1
    manager.zero_error = False
    manager.opt_params = {'downsample_radius': 1e-9, 'dump_candidates': None,
                          'dedup_tiebreak_foms': ('M20', 'M_nonesuch')}
    with pytest.raises(ValueError, match='unknown deduplication tiebreak FOM'):
        manager._downsample_computation(
            [np.array([1.0, 2.0])], [np.array([1.0, 2.0])],
            [np.array([[1.0], [2.0]])], [np.array([1, 1])], ['A', 'B'],
            n_top_candidates=10)


def test_the_tiebreak_reaches_the_reported_answer():
    """End of the manager path: a candidate the production tiebreak deletes is present in
    top_spacegroup and top_M20 once Minfo is allowed to rescue it."""
    from mlindex.optimization.MPIOptimizer import OptimizerManager

    def manager(tiebreak):
        stub = OptimizerManager.__new__(OptimizerManager)
        stub.lattice_system = 'cubic'
        stub.n_ranks = 1
        stub.zero_error = False
        stub.opt_params = {'downsample_radius': 1e-9, 'dump_candidates': None,
                           'dedup_tiebreak_foms': tiebreak}
        return stub

    arrays = ([np.array([1.0, 9.0, 2.0])], [np.array([7.0, 0.0, 0.0])],
              [np.array([[1.0], [1.0 + 1e-13], [5.0]])], [np.array([1, 1, 1])],
              ['A', 'B', 'C'])
    production = manager(('M20',))
    production._downsample_computation(*arrays, n_top_candidates=10)
    assert 'A' not in production.top_spacegroup

    rescued = manager(('M20', 'Minfo'))
    rescued._downsample_computation(*arrays, n_top_candidates=10)
    assert 'A' in rescued.top_spacegroup
    assert rescued.top_M20.tolist() == [9.0, 2.0, 1.0]


# ---------------------------------------------------------------------------
# Provenance. F-137 established that two arms of the same configuration are not
# comparable once the surviving row count differs, and retention changes the row count --
# so the ceiling before/after item 1 has to be a restriction *inside* one run, and that
# needs a column saying which track each row came from.
# ---------------------------------------------------------------------------

def test_every_production_row_is_labelled_as_the_m20_track():
    candidates = _real_candidates()
    assert candidates.retained_by.tolist() == [0, 0, 0]


def test_merged_rows_carry_the_track_they_came_from():
    stub = _retention_stub(
        m20=[9.0, 8.0],
        m20_track_xnn=[[1.0], [2.0]],
        minfo_track_xnn=[[5.0], [7.0]],
        )
    stub.merge_retained_iterates()
    assert stub.retained_by.tolist() == [0, 0, 1, 1]
    assert stub.retained_by.shape[0] == stub.best_xnn.shape[0]


def test_the_prune_keeps_provenance_aligned_too():
    stub = _retention_stub(
        m20=[9.0, 1.0, 7.0],
        m20_track_xnn=[[1.0], [2.0], [3.0]],
        minfo_track_xnn=[[10.0], [20.0], [30.0]],
        )
    stub.retained_by = np.array([0, 0, 0])
    stub.prune_below_m20(threshold=5.0)
    assert stub.retained_by.tolist() == [0, 0]
    stub.merge_retained_iterates()
    assert stub.retained_by.tolist() == [0, 0, 1, 1]


def test_the_manager_defaults_provenance_to_the_m20_track():
    """A caller predating retention passes no provenance and must still get a full column,
    or the pre-deduplication frame cannot be built."""
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    manager = OptimizerManager.__new__(OptimizerManager)
    manager.lattice_system = 'cubic'
    manager.n_ranks = 1
    manager.zero_error = False
    manager.bravais_lattice = 'cP'
    manager.n_peaks = 20
    manager.hkl_ref_length = 100
    manager.dump_context = None
    manager._dump_records = []
    manager._predownsample_records = []
    manager.q2_obs = np.linspace(0.1, 1.0, 20)
    manager.opt_params = {'downsample_radius': 1e-9, 'dump_candidates': None,
                          'dump_predownsample': True, 'assignment_threshold': 0.5,
                          'prune_m20_threshold': 5.0}
    manager._downsample_computation(
        [np.array([1.0, 2.0])], [np.array([1.0, 2.0])], [np.array([[1.0], [2.0]])],
        [np.array([1, 1])], ['A', 'B'], n_top_candidates=10)
    record = manager.drain_predownsample_dump()[0]
    assert record['retained_by'].tolist() == [0, 0]
    assert record['retention_foms'] == 'M20'
