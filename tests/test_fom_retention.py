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
        'spacegroup': ['P2/m', 'P21/m', 'P2/m'],
        }
    frame = FomBenchmark.predownsample_records_to_frame([record])
    assert list(frame.columns) == list(FomBenchmark.PREDOWNSAMPLE_COLUMNS)
    assert frame.shape[0] == 3
    assert frame['m20_at_prune'].tolist() == [1.5, 2.5, 3.5]
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
