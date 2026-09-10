"""The prune-site capture, the per-lattice threshold, and the batched correctness labeller.

All three are research affordances and all three must be invisible to a
shipped run. The first two live inside `Candidates.prune_below_m20`, which every rank of every
mode executes, so the tests that matter most here are the ones asserting that nothing happens
when the flags are absent.
"""
import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from mlindex.optimization.Candidates import Candidates, PRUNE_CAPTURE_MERITS
from mlindex.utilities.numba_functions import fast_assign


def _candidates(capture=False, n=32, seed=7):
    """A cubic Candidates over a synthetic peak list, small enough to reason about.

    Cubic because its xnn is one parameter, so a cell is a single number and the repair path
    `fix_unphysical` takes is easy to keep out of the way -- every starting cell here is well
    inside the allowed range, which matters because a repaired cell is exactly the case
    describes.
    """
    rng = np.random.default_rng(seed)
    hkl_ref = np.load(os.path.join(BASE, 'mlindex', 'models', 'cubic_1', 'data',
                                   'hkl_ref_cF.npy'))
    xnn_true = np.array([[1.0 / 8.0 ** 2]])
    from mlindex.utilities.Q2Calculator import Q2Calculator
    q2_ref = Q2Calculator(lattice_system='cubic', hkl=hkl_ref, tensorflow=False,
                          representation='xnn').get_q2(xnn_true)[0]
    q2_obs = np.sort(q2_ref[q2_ref > 0])[:10]

    xnn = xnn_true + rng.normal(0, 2e-4, size=(n, 1))
    opt_params = {'minimum_uc': 2.0, 'maximum_uc': 60.0, 'assignment_threshold': 0.95,
                  'figure_of_merit': 'M20'}
    if capture:
        opt_params['prune_criterion_capture'] = True
    return Candidates(q2_obs=q2_obs, xnn=xnn, hkl_ref=hkl_ref, lattice_system='cubic',
                      bravais_lattice='cF', opt_params=opt_params,
                      rng=np.random.default_rng(seed), fom=None, zero_error=False,
                      wavelength=None)


def test_capture_is_off_unless_asked_for():
    """The shipped path must not pay for, or record, anything."""
    candidates = _candidates(capture=False)

    candidates.prune_below_m20(threshold=0.0)

    assert candidates.prune_criterion_capture is False
    assert candidates.m20_at_prune is None
    assert candidates.merit_at_prune is None


def test_capture_records_every_criterion_and_reproduces_the_value_the_rule_tested():
    """The gate on the whole capture: the recomputed M20 IS `best_M20`.

    Both come from `best_xnn` through the same get_q2 -> fast_assign route, so anything else means
    the route diverged and the captured merits belong to a different cell than the one kept.
    """
    candidates = _candidates(capture=True)

    candidates.prune_below_m20(threshold=0.0)

    assert set(candidates.merit_at_prune) == set(PRUNE_CAPTURE_MERITS)
    assert np.array_equal(candidates.merit_at_prune['M20'], candidates.m20_at_prune)
    assert np.array_equal(candidates.merit_at_prune['M20'], candidates.best_M20)
    # M_sym is the product of its two factors, so the three columns are not independent and a
    # mis-assembled dict would show up here.
    assert np.allclose(candidates.merit_at_prune['M_sym'],
                       candidates.merit_at_prune['M_tilde'] * candidates.merit_at_prune['M_rev'])


def test_capture_columns_stay_row_aligned_with_the_survivors():
    candidates = _candidates(capture=True)

    candidates.prune_below_m20(threshold=np.median(candidates.best_M20))

    assert candidates.m20_at_prune.shape[0] == candidates.best_xnn.shape[0]
    for values in candidates.merit_at_prune.values():
        assert values.shape[0] == candidates.best_xnn.shape[0]


def test_a_per_lattice_mapping_selects_this_candidate_set_s_own_lattice():
    """A cut expressed per Bravais lattice.

    The mapping carries a deliberately absurd value for every other lattice: if the wrong key were
    read, the pool would collapse to the arg-max rescue and the count would be 1.
    """
    scalar = _candidates(capture=False)
    scalar.prune_below_m20(threshold=6.0)

    mapped = _candidates(capture=False)
    mapped.prune_below_m20(threshold={'cF': 6.0, 'cI': 1e9, 'aP': 1e9})

    assert mapped.n == scalar.n
    assert np.array_equal(mapped.best_M20, scalar.best_M20)


def test_the_arg_max_is_rescued_when_nothing_clears_the_bar():
    """`prune_below_m20` never empties a rank, so retention is never exactly zero."""
    candidates = _candidates(capture=True)
    best = candidates.best_M20.max()

    candidates.prune_below_m20(threshold=best * 10)

    assert candidates.n == 1
    assert candidates.best_M20[0] == best
    assert candidates.m20_at_prune.shape[0] == 1


def test_off_by_two_children_inherit_their_parent_s_at_prune_values():
    """An appended row is a rescaling of its parent, and the parent is what the cut tested.

    Without this a restriction of a threshold-0 run would keep children whose parents the cut
    would have deleted, and the restriction would stop reproducing the real cut.
    """
    candidates = _candidates(capture=True)
    candidates.prune_below_m20(threshold=0.0)
    before = candidates.m20_at_prune.copy()
    n_before = candidates.n

    candidates.refine_cell()
    candidates.standardize_cell()
    candidates.correct_off_by_two()

    assert candidates.m20_at_prune.shape[0] == candidates.best_xnn.shape[0]
    # The originals keep their own values, in place; anything appended came from a parent.
    assert np.array_equal(candidates.m20_at_prune[:n_before], before)
    for values in candidates.merit_at_prune.values():
        assert values.shape[0] == candidates.best_xnn.shape[0]
    if candidates.n > n_before:
        appended = candidates.m20_at_prune[n_before:]
        assert np.all(np.isin(appended, before))


def test_capture_refuses_zero_error_rather_than_returning_a_stale_value():
    candidates = _candidates(capture=True)
    candidates.zero_error = True

    with pytest.raises(NotImplementedError, match='zero-error'):
        candidates.prune_below_m20(threshold=0.0)


# Index lists into the full six-parameter truth cell (a, b, c, alpha, beta, gamma), matching what
# `validate_candidate_known_bl` slices out for itself. Not contiguous ranges: monoclinic takes
# beta and skips alpha, rhombohedral takes alpha and skips c. Getting this wrong compares the
# wrong angle and mislabels silently, which is why it is asserted here as well as used.
TRUTH_SLICE = {
    'cubic': [0], 'tetragonal': [0, 2], 'hexagonal': [0, 2], 'rhombohedral': [0, 3],
    'orthorhombic': [0, 1, 2], 'monoclinic': [0, 1, 2, 4], 'triclinic': [0, 1, 2, 3, 4, 5],
    }


# ----------------------------------------------------------------------------------------------
# The stored ORDER, and the support N_cal. Both had to be settled before S07's array, because
# neither is repairable on a generated pool.
# ----------------------------------------------------------------------------------------------


def test_n_cal_is_captured_and_explains_every_floored_M_rev():
    """A floored row stores `M_rev` = 0.0 and nothing else says why.

    With `n_cal` beside it the three states that share 0.0 are distinguishable after the fact:
    floored (`n_cal` below the floor), no reference lines in the window at all (`n_cal` == 0), and
    a candidate that simply scores nothing (`n_cal` at or above the floor, `M_rev` == 0).
    """
    from mlindex.utilities.FigureOfMerits import get_M_rev_sym

    candidates = _candidates(capture=True)
    candidates.prune_below_m20(threshold=0.0)
    captured = candidates.merit_at_prune

    assert 'n_cal' in captured
    n_cal = captured['n_cal']
    M_rev = captured['M_rev']
    assert n_cal.shape == M_rev.shape
    assert np.all(n_cal >= 0)

    # The floor is ten. Every row below it must carry M_rev == 0, which is the floor's contract.
    below = n_cal < 10
    assert np.all(M_rev[below] == 0.0), 'a row below the support floor kept a non-zero M_rev'

    # And `n_cal` must be the count the merit itself used, not a recount that could differ.
    q2_ref_calc = candidates.q2_calculator.get_q2(candidates.best_xnn)
    hkl_assign = fast_assign(candidates.q2_obs, q2_ref_calc)
    q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
    _, _, _, n_cal_direct = get_M_rev_sym(
        candidates.q2_obs, q2_calc, q2_ref_calc, return_n_cal=True)
    assert np.array_equal(n_cal, n_cal_direct.astype(np.float64))


def test_the_floor_can_be_reconstructed_from_what_is_stored():
    """The point of storing it: recover the unfloored value for the rows the floor touched."""
    from mlindex.utilities.FigureOfMerits import get_M_rev_sym

    candidates = _candidates(capture=True)
    candidates.prune_below_m20(threshold=0.0)
    n_cal = candidates.merit_at_prune['n_cal']

    q2_ref_calc = candidates.q2_calculator.get_q2(candidates.best_xnn)
    hkl_assign = fast_assign(candidates.q2_obs, q2_ref_calc)
    q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
    _, unfloored, _ = get_M_rev_sym(
        candidates.q2_obs, q2_calc, q2_ref_calc, min_n_cal=None)

    # Which rows the floor touched is exactly `n_cal < 10`, and nothing else in the pool says so.
    touched = (n_cal < 10) & (unfloored > 0)
    assert np.all(candidates.merit_at_prune['M_rev'][touched] == 0.0)
    if touched.any():
        assert np.all(unfloored[touched] > 0), 'the floor should only zero non-zero values'
