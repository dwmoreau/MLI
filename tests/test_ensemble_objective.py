"""The ensemble objective, pinned to the two copies it replaces.

The score that chooses the generator mix used to be written out twice, once in
`OptimizerManager._redistribution_testing_functional` and once inside
`mlindex.scripts.run_ensemble_refine.ensemble_refine`. Both bodies are held below verbatim, and
this file is now their only copy: if `mlindex.utilities.EnsembleObjective` ever stops reproducing
them exactly, that is a change to a shipped number and it fails here first.

The comparisons are `==`, not `allclose`, on purpose. The point is that nothing moved at all.
"""
import numpy as np
import pytest

from mlindex.utilities.ConvergenceCurve import success_of_distance
from mlindex.utilities.EnsembleObjective import (
    CONVERGENCE_CUT,
    capped_log_objective,
    evaluate,
    excess_count_objective,
    shell_counts,
    shell_targets,
    )


# ---------------------------------------------------------------------------
# The originals, verbatim
# ---------------------------------------------------------------------------


def _targets_copy(success_rate):
    """From MPIOptimizer.redistrubution_testing, and from run_ensemble_refine.ensemble_refine."""
    N_success = 1/success_rate
    in_range = success_rate > 0.01
    N_success[~in_range] = np.inf
    return N_success


def _optimizer_copy(distance, x, N_success):
    """From OptimizerManager._redistribution_testing_functional, below the distance it is given."""
    bins = np.concatenate([[0], x])
    distance_hist, _ = np.histogram(distance, bins=bins)
    N = np.cumsum(distance_hist)
    in_range = N_success != np.inf
    F = (N[in_range] - N_success[in_range]) / N_success[in_range]
    term_0 = 0
    if np.max(F) < 0:
        term_0 += 100
    term_1 = -np.mean(
        np.trapezoid(F, x[in_range]) / np.trapezoid(x[in_range])
        )
    return term_0 + term_1


def _script_copy(distance, x, N_success):
    """From the inner target_function of run_ensemble_refine.ensemble_refine, below its pooling.

    Textually identical to the optimizer's copy from the histogram down. That is the duplication,
    and keeping both here is what makes the claim checkable rather than asserted.
    """
    bins = np.concatenate([[0], x])
    distance_hist, _ = np.histogram(distance, bins=bins)
    N = np.cumsum(distance_hist)
    in_range = N_success != np.inf
    F = (N[in_range] - N_success[in_range]) / N_success[in_range]
    term_0 = 0
    if np.max(F) < 0:
        term_0 += 100
    term_1 = -np.mean(
        np.trapezoid(F, x[in_range]) / np.trapezoid(x[in_range])
        )
    return term_0 + term_1


def _templater_copy(distance_convergence, success_rate, distance_train):
    """From MITemplates.fit, where the calibrator's regression targets are built."""
    indices_train = np.searchsorted(distance_convergence, distance_train)
    indices_train[indices_train < 0] = 0
    indices_train[indices_train >= success_rate.size] = success_rate.size - 1
    return success_rate[indices_train].ravel()


# ---------------------------------------------------------------------------
# Curves and pools to run them on
# ---------------------------------------------------------------------------


def _curve(n_shells=40, floor=0.0):
    """A curve shaped like the real ones: log-spaced shells, a rate falling past the cut."""
    radii = np.logspace(-4, -2, n_shells)
    success = np.clip(np.linspace(0.85, floor, n_shells), 0.0, 1.0)
    return radii, success


def _pools(rng, radii, n_pools=50, n_candidates=400):
    """Pools spanning the cases the score branches on, not only the comfortable middle."""
    pools = [
        np.full(n_candidates, radii[0]/2),          # everything inside the first shell
        np.full(n_candidates, radii[-1]*10),        # everything past the last shell
        np.array([radii[0]/2]),                     # a single close candidate
        np.linspace(radii[0], radii[-1], n_candidates),
        ]
    for _ in range(n_pools):
        pools.append(np.exp(rng.uniform(np.log(radii[0]/3), np.log(radii[-1]*3), n_candidates)))
    return pools


# ---------------------------------------------------------------------------
# The pinning tests
# ---------------------------------------------------------------------------


def test_the_shared_objective_reproduces_both_copies_bit_for_bit():
    rng = np.random.default_rng(12345)
    for floor in (0.0, 0.004, 0.01, 0.02):
        radii, success = _curve(floor=floor)
        shared_radii, shared_targets = shell_targets(np.vstack((radii, success)))
        reference_targets = _targets_copy(success.copy())
        np.testing.assert_array_equal(shared_targets, reference_targets)
        np.testing.assert_array_equal(shared_radii, radii)
        for distance in _pools(rng, radii):
            shared = excess_count_objective(distance, shared_radii, shared_targets)
            assert shared == _optimizer_copy(distance, radii, reference_targets.copy())
            assert shared == _script_copy(distance, radii, reference_targets.copy())


def test_a_shell_at_the_cut_is_out_of_range_and_one_just_above_it_is_not():
    success = np.array([0.5, 0.02, CONVERGENCE_CUT, 0.011, 0.0])
    radii = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4])
    _, n_success = shell_targets(np.vstack((radii, success)))
    assert np.isfinite(n_success[0]) and np.isfinite(n_success[1])
    assert n_success[2] == np.inf          # exactly at the cut: excluded
    assert np.isfinite(n_success[3])       # just above it: kept
    assert n_success[4] == np.inf          # a rate of zero, and no divide-by-zero warning


def test_a_rate_of_zero_does_not_raise_a_divide_warning():
    radii = np.array([1e-4, 2e-4])
    success = np.array([0.5, 0.0])
    with np.errstate(divide='raise'):
        shell_targets(np.vstack((radii, success)))


def test_candidates_beyond_the_last_shell_are_counted_nowhere():
    radii, _ = _curve()
    inside = np.array([radii[0], radii[5], radii[-1]*0.9])
    outside = np.concatenate((inside, [radii[-1]*2, radii[-1]*100]))
    np.testing.assert_array_equal(shell_counts(inside, radii), shell_counts(outside, radii))


def test_the_lookup_zeroes_beyond_the_curve_and_clamps_for_the_templater():
    radii, success = _curve()
    distance = np.array([radii[0]/2, radii[3], radii[-1], radii[-1]*1.5])
    zeroed = success_of_distance(radii, success, distance, beyond='zero')
    clamped = success_of_distance(radii, success, distance, beyond='clamp')
    np.testing.assert_array_equal(zeroed[:3], clamped[:3])
    assert zeroed[-1] == 0.0
    assert clamped[-1] == success[-1]


def test_the_templater_targets_are_unchanged_by_the_shared_lookup():
    """What stops this collapse from silently moving the targets 14 calibrators were fitted on."""
    rng = np.random.default_rng(7)
    radii, success = _curve()
    distance = np.exp(rng.uniform(np.log(radii[0]/3), np.log(radii[-1]*3), 500))
    np.testing.assert_array_equal(
        success_of_distance(radii, success, distance, beyond='clamp'),
        _templater_copy(radii, success, distance),
        )


def test_an_unknown_out_of_range_rule_is_refused():
    radii, success = _curve()
    with pytest.raises(ValueError, match='out-of-range rule'):
        success_of_distance(radii, success, np.array([1e-4]), beyond='last')


# ---------------------------------------------------------------------------
# The capped log score
# ---------------------------------------------------------------------------


def test_the_capped_score_never_exceeds_its_cap():
    radii, success = _curve()
    distance = np.full(5000, radii[0]/2)
    assert capped_log_objective(distance, radii, success, 5.0, np.ones(5000)) == 5.0


def test_a_candidate_below_the_cut_contributes_nothing():
    radii = np.array([1e-4, 2e-4, 3e-4])
    success = np.array([0.5, 0.005, 0.002])
    weight = np.ones(2)
    counted = capped_log_objective(np.array([1e-4, 1e-4]), radii, success, 100.0, weight)
    hopeless = capped_log_objective(np.array([1e-4, 3e-4]), radii, success, 100.0, weight)
    assert hopeless == pytest.approx(counted/2)
    assert hopeless == pytest.approx(-np.log(1 - 0.5))


def test_the_weight_scales_a_candidate_contribution():
    radii = np.array([1e-4, 2e-4])
    success = np.array([0.5, 0.4])
    distance = np.array([1e-4, 1e-4])
    full = capped_log_objective(distance, radii, success, 100.0, np.ones(2))
    half = capped_log_objective(distance, radii, success, 100.0, np.full(2, 0.5))
    assert half == pytest.approx(full/2)


def test_several_pools_are_scored_at_once():
    radii, success = _curve()
    distance = np.vstack((np.full(10, radii[0]), np.full(10, radii[-1]*10)))
    scored = capped_log_objective(distance, radii, success, 5.0, np.ones((2, 10)))
    assert scored.shape == (2,)
    assert scored[1] == 0.0


# ---------------------------------------------------------------------------
# The dispatcher refuses what it cannot use
# ---------------------------------------------------------------------------


def test_a_variant_refuses_a_constant_it_cannot_use():
    radii, success = _curve()
    curve = np.vstack((radii, success))
    distance = np.full(20, radii[0])
    with pytest.raises(ValueError, match='neither a cap nor a weight'):
        evaluate('shipped', distance, curve, cap=5.0)
    with pytest.raises(ValueError, match='needs a cap'):
        evaluate('capped', distance, curve)
    with pytest.raises(ValueError, match='use'):
        evaluate('capped', distance, curve, cap=5.0, weight=np.ones(20))
    with pytest.raises(ValueError, match='both a cap and a weight'):
        evaluate('capped_alpha', distance, curve, cap=5.0)


def test_an_unknown_variant_is_refused_rather_than_ignored():
    radii, success = _curve()
    with pytest.raises(ValueError, match='unknown ensemble objective variant'):
        evaluate('best', np.full(20, radii[0]), np.vstack((radii, success)))


def test_the_dispatcher_agrees_with_the_kernels_it_dispatches_to():
    rng = np.random.default_rng(3)
    radii, success = _curve()
    curve = np.vstack((radii, success))
    distance = np.exp(rng.uniform(np.log(radii[0]), np.log(radii[-1]), 300))
    shared_radii, targets = shell_targets(curve)
    assert evaluate('shipped', distance, curve) == excess_count_objective(
        distance, shared_radii, targets)
    weight = np.ones(300)
    assert evaluate('capped', distance, curve, cap=5.0) == capped_log_objective(
        distance, radii, success, 5.0, weight)
