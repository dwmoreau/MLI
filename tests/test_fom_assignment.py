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


# ------------------------------------------------------------------------------------------
# The soft counting merit
# ------------------------------------------------------------------------------------------