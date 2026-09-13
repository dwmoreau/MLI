"""The per-peak assignment posterior: the estimator, its full-distribution form, and its floors.

Two things are pinned here, and each corresponds to a way this could be wrong without looking
wrong.

**The full distribution must be the same arithmetic as the scalar posterior.** The final
refinement's peak mask reads the scalar form and the candidate generator reads the distribution,
so the two consumers have to be reading one estimator rather than two implementations of one
description of it.

**The degenerate cases must be answers, not exceptions.** A candidate that fits its peaks exactly
drives the estimated scale to zero; squaring it underflows to exactly 0.0 and the kernel would
then divide by zero. The floor makes that case return the right answer -- every competing exponent
underflows, the nearest term is 1, so the peak is assigned with certainty -- and it must not move
anything that was not degenerate to begin with.
"""
import numpy as np
import pytest


def _random_case(seed, n_candidates, n_ref, n_peaks, lattice_system):
    rng = np.random.default_rng(seed)
    q2_obs = np.sort(rng.uniform(0.01, 0.5, n_peaks))
    q2_ref_calc = np.sort(rng.uniform(0.005, 0.6, (n_candidates, n_ref)), axis=1)
    return q2_obs, q2_ref_calc, lattice_system


def test_the_fortran_ordered_fallback_agrees_with_the_fast_path():
    """The non-C-contiguous branch is a safety net, and it has to compute the same thing.

    Not bit-for-bit: `np.sum(axis=1)` groups its additions differently over an F-ordered block,
    which `get_assignment_posterior`'s own comment records as deliberate.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_posterior

    q2_obs, q2_ref_calc, system = _random_case(6, 5, 71, 15, 'hexagonal')
    fortran = np.asfortranarray(q2_ref_calc)
    assert not fortran.flags.c_contiguous
    assert np.allclose(get_assignment_posterior(q2_obs, q2_ref_calc, system),
                       get_assignment_posterior(q2_obs, fortran, system), atol=1e-12)


# ------------------------------------------------------------------------------------------
# The scale floor
# ------------------------------------------------------------------------------------------
def test_a_candidate_that_fits_exactly_scores_one_rather_than_raising():
    """Zero residuals underflowed the posterior's scale to exactly 0 and the kernel divided by it.

    `get_assignment_sigma` clamps sigma at 1e-300, which looks like the degenerate case is handled;
    the consumer squares it and 1e-600 is 0.0. In numba that is a `ZeroDivisionError`, so the one
    candidate a synthetic pattern is guaranteed to contain -- the cell it was generated from --
    raised instead of scoring. The right answer is 1: an exact fit assigns with certainty.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_posterior

    q2_ref_calc = np.sort(np.random.default_rng(8).uniform(0.01, 0.6, (3, 64)), axis=1)
    q2_obs = q2_ref_calc[0, :20].copy()      # every residual exactly zero for candidate 0

    posterior = get_assignment_posterior(q2_obs, q2_ref_calc, 'orthorhombic')
    assert np.all(np.isfinite(posterior))
    assert np.allclose(posterior[0], 1.0)


def test_the_scale_floor_moves_nothing_that_was_not_degenerate():
    """The floor binds only where the unfloored scale would be zero or subnormal."""
    from mlindex.utilities.FigureOfMerits import _posterior_scale

    sigma = np.array([1e-3, 1.0, 1e-100, 1e-160, 1e-300])
    floored = _posterior_scale(sigma, 1.0)
    unfloored = 2*sigma**2
    ordinary = unfloored >= np.finfo(np.float64).tiny
    assert np.array_equal(floored[ordinary], unfloored[ordinary])
    assert np.all(floored[~ordinary] == np.finfo(np.float64).tiny)


@pytest.mark.parametrize('n_candidates, n_ref, n_peaks, lattice_system', [
    (300, 700, 20, 'triclinic'),
    (257, 1000, 20, 'monoclinic'),
    (5, 40, 10, 'cubic'),
    ])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_the_distribution_at_the_nearest_line_is_the_scalar_posterior(
        n_candidates, n_ref, n_peaks, lattice_system, order):
    """The two estimators are one estimator, and this is the equality that says so.

    The log-sum-exp shift makes the nearest line's own term exactly exp(0), so the normalised
    distribution read at that line is 1/sum(terms) -- the scalar posterior, from the same sum
    taken in the same place. Bit for bit, not to a tolerance: if these two ever drift, the peak
    mask and the candidate generator are reading different estimators.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution
    from mlindex.utilities.FigureOfMerits import get_assignment_posterior

    q2_obs, q2_ref_calc, system = _random_case(
        11, n_candidates, n_ref, n_peaks, lattice_system
        )
    if order == 'F':
        q2_ref_calc = np.asfortranarray(q2_ref_calc)
    posterior = get_assignment_posterior(q2_obs, q2_ref_calc, system)
    distribution = get_assignment_distribution(q2_obs, q2_ref_calc, system)
    nearest = np.argmin(
        np.abs(q2_ref_calc[:, np.newaxis, :] - q2_obs[np.newaxis, :, np.newaxis]), axis=2
        )
    at_nearest = np.take_along_axis(distribution, nearest[:, :, np.newaxis], axis=2)[:, :, 0]
    assert np.array_equal(at_nearest, posterior)


def test_the_normalised_distribution_sums_to_one_over_the_reference_list():
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution

    q2_obs, q2_ref_calc, system = _random_case(12, 64, 300, 20, 'orthorhombic')
    distribution = get_assignment_distribution(q2_obs, q2_ref_calc, system)
    assert distribution.shape == (64, 20, 300)
    assert np.allclose(np.sum(distribution, axis=2), 1.0, rtol=1e-14, atol=0)


def test_the_unnormalised_form_draws_the_same_miller_indices():
    """`vectorized_resampling` rescales each draw by the row's own cumulative total.

    So the generator may pass `normalise=False` and save an array pass; this is what makes that
    safe rather than merely plausible. A row of zeros cannot occur -- the nearest line's term is
    exactly 1 -- so there is no degenerate case where the two forms could diverge.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution
    from mlindex.utilities.MillerIndexAssignment import vectorized_resampling

    q2_obs, q2_ref_calc, system = _random_case(13, 24, 200, 20, 'tetragonal')
    normalised = get_assignment_distribution(q2_obs, q2_ref_calc, system)
    raw = get_assignment_distribution(q2_obs, q2_ref_calc, system, normalise=False)
    assigned_normalised, _ = vectorized_resampling(normalised, np.random.default_rng(7))
    assigned_raw, _ = vectorized_resampling(raw, np.random.default_rng(7))
    assert np.array_equal(assigned_normalised, assigned_raw)


# ------------------------------------------------------------------------------------------
# The soft counting merit
# ------------------------------------------------------------------------------------------