"""The ensemble objective: the expected number of candidates that converge.

The score is a sum over candidates of their own chance of refining to the true cell, each weighted
by its share of its own clump -- see `tests/test_clump_discount.py` for that term. Two things
about it are deliberate and look like defects, so they are pinned here: a candidate beyond the
last measured shell is worth nothing rather than the last shell's rate, and a candidate whose rate
is at or below the cut is dropped.
"""
import numpy as np
import pytest

from mlindex.utilities.ConvergenceCurve import success_of_distance
from mlindex.utilities.EnsembleObjective import CONVERGENCE_CUT, expected_success_objective


def _curve(n_shells=40, floor=0.0):
    """A curve shaped like the real ones: log-spaced shells, a rate falling past the cut."""
    radii = np.logspace(-4, -2, n_shells)
    success = np.clip(np.linspace(0.85, floor, n_shells), 0.0, 1.0)
    return radii, success


def test_the_lookup_is_worth_nothing_beyond_the_curve():
    """Crediting a candidate past the curve extrapolates, and it is not harmless: the last
    shell's rate is small but not zero, so a few thousand hopeless candidates each credited with
    it drive any combined score to saturation and it stops separating one mix from another."""
    radii, success = _curve()
    rate = success_of_distance(
        radii, success, np.array([radii[0]/2, radii[3], radii[-1], radii[-1]*1.5]))
    assert rate[0] == success[0]      # closer than the first shell takes the first rate
    assert rate[1] == success[3]
    assert rate[2] == success[-1]     # exactly on the last shell still counts
    assert rate[3] == 0.0             # past it is worth nothing


def test_each_candidate_counts_at_its_own_rate():
    radii = np.array([1e-4, 2e-4, 3e-4])
    success = np.array([0.5, 0.3, 0.005])
    # two in the first shell, one in the second, one past the cut and so worth nothing
    distance = np.array([1e-4, 1e-4, 2e-4, 3e-4])
    assert expected_success_objective(distance, radii, success, np.ones(4)) == \
        pytest.approx(0.5 + 0.5 + 0.3)


def test_a_candidate_at_or_below_the_cut_contributes_nothing():
    radii = np.array([1e-4, 2e-4, 3e-4])
    success = np.array([0.5, CONVERGENCE_CUT, 0.011])
    one = np.ones(1)
    assert expected_success_objective(np.array([2e-4]), radii, success, one) == 0.0
    assert expected_success_objective(np.array([3e-4]), radii, success, one) == \
        pytest.approx(0.011)


def test_a_stack_of_pools_scores_the_same_as_the_pools_one_at_a_time():
    """The mix grid scores every candidate set at once; it must not become a second
    implementation of the same sum."""
    rng = np.random.default_rng(31)
    radii, success = _curve()
    stack = np.exp(rng.uniform(np.log(radii[0]/3), np.log(radii[-1]*3), (25, 600)))
    together = expected_success_objective(stack, radii, success, np.ones(stack.shape))
    assert together.shape == (25,)
    for index in range(stack.shape[0]):
        assert together[index] == expected_success_objective(
            stack[index], radii, success, np.ones(stack.shape[1]))


def test_the_score_rises_with_candidates_and_never_falls():
    """It is a sum of non-negative terms, which is why it cannot show that MORE candidates hurt --
    the reason the candidate budget is an end-to-end question and not one for this score."""
    rng = np.random.default_rng(5)
    radii, success = _curve()
    pool = np.exp(rng.uniform(np.log(radii[0]), np.log(radii[-1]), 400))
    running = [float(expected_success_objective(pool[:n], radii, success, np.ones(n)))
               for n in (50, 100, 200, 400)]
    assert running == sorted(running)
