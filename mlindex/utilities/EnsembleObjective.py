"""What a pool of candidate unit cells is worth, scored against a convergence curve.

The pool is what the generators produce before any refinement; the score says how likely that pool
is to contain a cell close enough to the true one to refine to it. It is what the generator mix
and the redistribution constants are chosen against.

One implementation. It was two, identical from the histogram down -- the inner `target_function`
of `mlindex.scripts.run_ensemble_refine` and `OptimizerManager._redistribution_testing_functional`
-- and `tests/test_ensemble_objective.py` holds both of the originals verbatim and requires this
module to reproduce each of them to the bit.
"""
import numpy as np

from mlindex.utilities.ConvergenceCurve import success_of_distance


# Shells at or below this success rate are left out of the score. Below it the number of
# candidates needed for one success runs away, and the curve's own measurement is thinnest there.
CONVERGENCE_CUT = 0.01

VARIANTS = ('shipped', 'capped', 'capped_alpha')


def shell_targets(curve):
    """(radii, n_success) from a two-row curve.

    `n_success` is how many candidates starting in a shell it takes for one of them to converge,
    and it is `inf` on the shells that are out of range, which is the mark every downstream mask
    tests for.
    """
    radii = np.asarray(curve[0], dtype=float)
    success_rate = np.asarray(curve[1], dtype=float)
    # A shell with a rate of zero would divide by zero; it is out of range anyway, and the next
    # line would overwrite the result, so skip the division rather than warn about it.
    n_success = np.divide(
        1.0, success_rate,
        out=np.full(success_rate.shape, np.inf),
        where=success_rate > 0,
        )
    n_success[success_rate <= CONVERGENCE_CUT] = np.inf
    return radii, n_success


def shell_counts(distance, radii):
    """N(r): how many candidates lie within each shell, cumulatively.

    A candidate further out than the last shell is counted in no bin at all. That is not an
    oversight to be repaired -- it is how this score has always excluded candidates beyond the
    measured curve, and on the low-symmetry lattices it is three quarters of them.
    """
    bins = np.concatenate([[0], radii])
    distance_hist, _ = np.histogram(distance, bins=bins)
    return np.cumsum(distance_hist)


def shell_excess(distance, radii, n_success):
    """(F, in_range). F is the pool's surplus over the count needed, per in-range shell."""
    N = shell_counts(distance, radii)
    in_range = n_success != np.inf
    F = (N[in_range] - n_success[in_range]) / n_success[in_range]
    return F, in_range


def excess_count_objective(distance, radii, n_success):
    """The shipped score, minimised: a flat penalty if the pool never reaches the count it needs,
    minus the normalised integral of its surplus where it does.

    One pattern, one pool: `distance` is (n_candidates,) and the result is a scalar. Combining
    patterns is the caller's business, because fitting each pattern separately and averaging the
    answers describes no pattern.

    The arithmetic below is written in the order the two originals wrote it and is not to be
    simplified into an algebraically equal form, because the test that pins this to them compares
    with `==`.
    """
    F, in_range = shell_excess(distance, radii, n_success)
    term_0 = 0
    if np.max(F) < 0:
        term_0 += 100
    term_1 = -np.mean(
        np.trapezoid(F, radii[in_range]) / np.trapezoid(radii[in_range])
        )
    return term_0 + term_1


def capped_log_objective(distance, radii, success_rate, cap, weight):
    """Lambda = sum of weight_i * -log(1 - s(d_i)) over the candidates worth counting, capped.

    Maximised, not minimised. Lambda is the log of one over the chance that every candidate
    fails, so it adds up over candidates and can be read at any budget; the cap stops a pool that
    is already certain from earning more credit for candidates it does not need.

    `weight` is required and has no default. Pass ones for an unweighted pool. A pool whose
    candidates sit in clumps passes each candidate's share of its clump, which the caller builds
    from the measured clump discount -- that table is research output and does not live here.

    `distance` and `weight` are (..., n_candidates); the result is (...).
    """
    rate = success_of_distance(radii, success_rate, distance, beyond='zero')
    # A shell measured at a rate of exactly 1 would make one candidate worth infinite credit. The
    # cap would absorb it, but the intermediate overflows, so hold the rate just below certainty.
    rate = np.clip(rate, 0.0, 1.0 - 1e-12)
    counted = rate > CONVERGENCE_CUT
    contribution = np.where(counted, -np.log1p(-rate), 0.0)
    return np.minimum(np.sum(weight*contribution, axis=-1), cap)


def evaluate(variant, distance, curve, cap=None, weight=None):
    """One entry point, so a driver can score the same cached pool under several variants.

    Each variant is given exactly the constants it can use and refuses the ones it cannot, because
    a constant a variant ignores is a constant nobody checks the value of.

      'shipped'       the score the generator mix was originally chosen against. Minimised.
                      Takes neither a cap nor a weight.
      'capped'        the capped log score, unweighted. Maximised. Needs a cap.
      'capped_alpha'  the same, with a per-candidate weight for clumping. Needs both.

    Production does not come through here: `_redistribution_testing_functional` calls
    `excess_count_objective` directly, so a variant added for a measurement has no route into the
    optimizer.
    """
    if variant == 'shipped':
        if cap is not None or weight is not None:
            raise ValueError("variant='shipped' takes neither a cap nor a weight")
        radii, n_success = shell_targets(curve)
        return excess_count_objective(distance, radii, n_success)
    if variant == 'capped':
        if cap is None:
            raise ValueError("variant='capped' needs a cap")
        if weight is not None:
            raise ValueError("variant='capped' is the unweighted one; use 'capped_alpha'")
        return capped_log_objective(
            distance, curve[0], curve[1], cap, np.ones(np.shape(distance)))
    if variant == 'capped_alpha':
        if cap is None or weight is None:
            raise ValueError("variant='capped_alpha' needs both a cap and a weight")
        return capped_log_objective(distance, curve[0], curve[1], cap, weight)
    raise ValueError(f'unknown ensemble objective variant {variant!r}')
