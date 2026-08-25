"""Tests for S03's prune-criterion analysis (`mlindex/scripts/run_fom_prune_criterion.py`).

The script's own gates run against 70 million rows of untracked run output, which no test can
carry. What is testable without that data is the machinery those gates rest on: that the merit
recompute honours `get_M20`'s in-place mutation, that a cut's sign convention is right on the
three merits where a low value is the good one, and that the deduplication emulator is production's
own collapse rather than a second implementation of it.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from mlindex.optimization.MPIOptimizer import _downsample_chunk
from mlindex.utilities.FigureOfMerits import get_M20


def _load_module():
    """The script lives in mlindex/scripts/, which is not a package, so load it by path."""
    path = os.path.join(BASE, 'mlindex', 'scripts', 'run_fom_prune_criterion.py')
    spec = importlib.util.spec_from_file_location('run_fom_prune_criterion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRUNE = _load_module()


@pytest.fixture
def pool():
    """A small, deterministic (peaks, candidates, reference lines) set."""
    rng = np.random.default_rng(2026)
    q2_obs = np.sort(rng.uniform(0.01, 0.4, size=20))
    q2_ref_calc = np.sort(rng.uniform(0.005, 0.6, size=(64, 300)), axis=1)
    return q2_obs, q2_ref_calc


def test_merits_reproduce_get_M20_through_the_same_route(pool):
    """The M20 the recompute returns is the one `get_M20` returns on the same arrays.

    This is the property the 70-million-row gate checks at scale: the recompute walks the
    pipeline's own route rather than rebuilding q2_calc from Miller indices.
    """
    q2_obs, q2_ref_calc = pool
    from mlindex.utilities.numba_functions import fast_assign

    values = PRUNE.merits_on_reference(q2_obs, q2_ref_calc.copy())

    reference = q2_ref_calc.copy()
    hkl_assign = fast_assign(q2_obs, reference)
    q2_calc = np.take_along_axis(reference, hkl_assign, axis=1)
    expected = get_M20(q2_obs, q2_calc, reference)

    assert np.array_equal(values['M20'], expected)


def test_get_M20_is_called_last_so_the_other_merits_see_a_pristine_array(pool):
    """`get_M20` zeroes q2_ref_calc outside the cut-off via np.putmask.

    If it ran before the reversed and symmetric merits, they would read an array two thirds of
    which had been set to zero. Campaign 1 states this trap in three places; this is the test that
    would catch it being reordered.
    """
    q2_obs, q2_ref_calc = pool

    ordered = PRUNE.merits_on_reference(q2_obs, q2_ref_calc.copy())

    # The same merits computed with get_M20 deliberately run first, which is what a reordering
    # would produce. They must NOT agree -- if they do, the mutation has stopped mattering and
    # this test has stopped guarding anything.
    damaged = q2_ref_calc.copy()
    from mlindex.utilities.numba_functions import fast_assign
    hkl_assign = fast_assign(q2_obs, damaged)
    q2_calc = np.take_along_axis(damaged, hkl_assign, axis=1)
    get_M20(q2_obs, q2_calc, damaged)
    from mlindex.utilities.FigureOfMerits import get_M_rev_sym
    _, _, M_sym_after = get_M_rev_sym(q2_obs, q2_calc, damaged)

    assert not np.allclose(ordered['M_sym'], M_sym_after)


def test_criterion_scores_flip_the_sign_where_a_low_value_is_the_good_one():
    frame = pd.DataFrame({'M_sym_C': [1.0, 2.0], 'X_N_C': [0.0, 3.0], 'n_over_B': [4.0, 1.0],
                          'm20_at_prune': [3.0, 7.0]})

    assert np.array_equal(PRUNE.criterion_scores(frame, 'M_sym_C', None), [1.0, 2.0])
    assert np.array_equal(PRUNE.criterion_scores(frame, 'm20_at_prune', None), [3.0, 7.0])
    # X_N and n_over count things that should not be there, so a cut keeps the SMALL values.
    assert np.array_equal(PRUNE.criterion_scores(frame, 'X_N_C', None), [-0.0, -3.0])
    assert np.array_equal(PRUNE.criterion_scores(frame, 'n_over_B', None), [-4.0, -1.0])


def test_the_veto_composite_removes_candidates_at_every_threshold():
    """de Wolff's X20 <= 2 rider, expressed as a score so it needs no second rule."""
    frame = pd.DataFrame({'M_sym_C': [9.0, 9.0], 'X_N_C': [1.0, 5.0]})

    scores = PRUNE.criterion_scores(frame, 'M_sym_C', 'X_N_C')

    assert scores[0] == 9.0
    assert scores[1] == -np.inf


def test_deduplicate_is_production_collapse_and_carries_row_identity():
    """The emulator must return the same rows `_downsample_chunk` keeps, as original indices."""
    rng = np.random.default_rng(7)
    # One tight cluster plus scattered points, in a lattice system whose xnn is 1-dimensional.
    xnn = np.concatenate([np.full((6, 1), 0.0100) + rng.normal(0, 1e-5, (6, 1)),
                          np.linspace(0.02, 0.05, 9).reshape(-1, 1)])
    M20 = rng.uniform(1, 20, xnn.shape[0])
    Minfo = rng.uniform(1, 50, xnn.shape[0])
    n_indexed = rng.integers(1, 20, xnn.shape[0])

    kept = PRUNE.deduplicate(xnn, M20, Minfo, n_indexed, 'cubic', radius=0.002)

    # The cluster collapses to its highest-M20 member, and that member is the one returned.
    cluster = np.flatnonzero(xnn[:, 0] < 0.015)
    survivors_in_cluster = [index for index in kept if index in cluster]
    assert len(survivors_in_cluster) == 1
    assert survivors_in_cluster[0] == cluster[np.argmax(M20[cluster])]
    # Every returned index is a real row, and none is returned twice.
    assert len(set(kept)) == len(kept)
    assert set(kept) <= set(range(xnn.shape[0]))


def test_deduplicate_matches_downsample_chunk_on_a_single_chunk():
    """Below the 1 000-row chunk size the emulator is one call, so it must agree exactly."""
    rng = np.random.default_rng(11)
    xnn = np.sort(rng.uniform(0.01, 0.02, size=(40, 1)), axis=0)
    M20 = rng.uniform(1, 20, 40)
    Minfo = rng.uniform(1, 50, 40)
    n_indexed = rng.integers(1, 20, 40)

    kept = PRUNE.deduplicate(xnn, M20, Minfo, n_indexed, 'cubic', radius=0.0005)

    # deduplicate sorts by reciprocal volume first; reproduce that ordering, then call the
    # production chunker directly on the sorted arrays.
    from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
    from mlindex.utilities.UnitCellTools import get_unit_cell_volume
    volume = get_unit_cell_volume(
        get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True, lattice_system='cubic'),
        partial_unit_cell=True, lattice_system='cubic')
    order = np.argsort(volume)
    expected = _downsample_chunk((xnn[order], M20[order], Minfo[order], n_indexed[order],
                                  list(order), 0.0005))

    assert list(kept) == list(expected[4])


def test_exact_mismatches_treats_nan_as_equal_to_nan():
    stored = np.array([1.0, np.nan, 3.0])
    assert PRUNE.exact_mismatches(stored, np.array([1.0, np.nan, 3.0])) == 0
    assert PRUNE.exact_mismatches(stored, np.array([1.0, np.nan, 3.0000000001])) == 1
    assert PRUNE.exact_mismatches(stored, np.array([1.0, 2.0, 3.0])) == 1


def test_rate_ci_bootstraps_over_entries_not_candidates():
    """One entry contributing many candidates must move the rate as one observation.

    Two entries, one all-success and one all-failure, with wildly unequal candidate counts: a
    candidate-level bootstrap would give a narrow interval around 0.99, an entry-level one has to
    span both outcomes because resampling two entries often draws the same one twice.
    """
    successes = np.array([9900, 0])
    totals = np.array([10000, 100])
    rng = np.random.default_rng(3)

    low, high = PRUNE._rate_ci_by_entry(successes, totals, rng)

    assert low < 0.5 < high


def test_downstream_costs_shrink_as_the_cut_moves_later():
    """A cut placed later has fewer steps downstream of it, by construction."""
    assert set(PRUNE.DOWNSTREAM_OF['C']) < set(PRUNE.DOWNSTREAM_OF['B'])
    assert set(PRUNE.DOWNSTREAM_OF['B']) < set(PRUNE.DOWNSTREAM_OF['A'])
    assert set(PRUNE.DOWNSTREAM_OF['A']) == set(PRUNE.POST_PRUNE_STEPS)
    # assign_extinction_group is the expensive one, and it is downstream of A and B but not C --
    # which is what makes B and C different decisions rather than the same one.
    assert 'assign_extinction_group' in PRUNE.DOWNSTREAM_OF['B']
    assert 'assign_extinction_group' not in PRUNE.DOWNSTREAM_OF['C']


def test_criteria_list_carries_no_stage_A_merit_other_than_M20():
    """`best_xnn` at the cut was never stored, so no other merit HAS a stage-A value.

    The absence is a finding, not an omission; a criterion list that grew one would be reporting
    a number the dump cannot support.
    """
    stage_a = [label for label, point, _, _ in PRUNE.criteria_list() if point == 'A']

    assert stage_a == ['m20_at_prune']
