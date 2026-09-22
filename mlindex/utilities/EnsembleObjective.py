"""What a pool of candidate unit cells is worth, scored against a convergence curve.

The pool is what the generators produce before any refinement; the score says how likely it is to
hold a cell close enough to the true one to refine to it. It is what the generator mix and the
redistribution constants are chosen against.

`tests/test_ensemble_objective.py` holds the two copies this replaced, verbatim, and requires the
shipped score here to reproduce each of them to the bit.
"""
import numpy as np

from mlindex.utilities.ConvergenceCurve import success_of_distance


# Shells at or below this success rate are left out. Below it the number of candidates needed for
# one success runs away, and the curve's own measurement is thinnest there.
CONVERGENCE_CUT = 0.01

VARIANTS = ('shipped', 'expected', 'capped')


def shell_targets(curve):
    """(radii, n_success) from a two-row curve.

    `n_success` is how many candidates starting in a shell it takes for one to converge, and it is
    `inf` on the shells out of range, which is the mark every downstream mask tests for.
    """
    radii = np.asarray(curve[0], dtype=float)
    success_rate = np.asarray(curve[1], dtype=float)
    # A zero-rate shell is out of range anyway and the next line overwrites it, so skip the
    # division rather than warn about it.
    n_success = np.divide(1.0, success_rate, out=np.full(success_rate.shape, np.inf),
                          where=success_rate > 0)
    n_success[success_rate <= CONVERGENCE_CUT] = np.inf
    return radii, n_success


def shell_counts(distance, radii):
    """N(r): candidates within each shell, cumulatively. (..., n_candidates) -> (..., n_shells).

    A candidate further out than the last shell is counted in no bin. That is not an oversight to
    repair: it is how this score excludes candidates beyond the measured curve, and on the
    low-symmetry lattices that is three quarters of them.
    """
    distance = np.atleast_1d(distance)
    bins = np.concatenate([[0], radii])
    # np.histogram's bins are half-open except the last, which is closed at both ends; match that
    # so a distance landing exactly on the outermost shell keeps being counted.
    shell = np.searchsorted(bins, distance, side='right') - 1
    shell = np.where(distance == bins[-1], bins.size - 2, shell)
    inside = (shell >= 0) & (shell < radii.size)

    flat = shell.reshape(-1, distance.shape[-1])
    keep = inside.reshape(flat.shape)
    pool = np.repeat(np.arange(flat.shape[0]), flat.shape[1]).reshape(flat.shape)
    counted = np.bincount((pool[keep]*radii.size + flat[keep]).ravel(),
                          minlength=flat.shape[0]*radii.size).reshape(flat.shape[0], radii.size)
    return np.cumsum(counted, axis=-1).reshape(distance.shape[:-1] + (radii.size,))


def excess_count_objective(distance, radii, n_success):
    """The shipped score, MINIMISED: a flat penalty if the pool never reaches the count it needs,
    minus the normalised integral of its surplus where it does.

    `distance` is (..., n_candidates), the result (...). Combining patterns is the caller's.

    The arithmetic below is in the order the originals wrote it and is not to be simplified into
    an algebraically equal form: the test pinning this to them compares with `==`.
    """
    N = shell_counts(distance, radii)
    in_range = n_success != np.inf
    F = (N[..., in_range] - n_success[in_range]) / n_success[in_range]
    term_0 = np.where(np.max(F, axis=-1) < 0, 100, 0)
    term_1 = -(np.trapezoid(F, radii[in_range], axis=-1) / np.trapezoid(radii[in_range]))
    return term_0 + term_1


def expected_success_objective(distance, radii, success_rate):
    """The expected number of candidates that converge, sum of s(d_i). MAXIMISED.

    It differs from the shipped score in one thing: what a candidate is worth. The shipped score
    weights it by the whole tail of the curve beyond it; this weights it by its own chance.
    """
    rate = success_of_distance(radii, success_rate, distance)
    return np.sum(np.where(rate > CONVERGENCE_CUT, rate, 0.0), axis=-1)


def capped_log_objective(distance, radii, success_rate, cap, weight):
    """min(Lambda, cap), where Lambda = sum of weight_i * -log(1 - s(d_i)). MAXIMISED.

    Lambda is the log of one over the chance every candidate fails, so it adds up over candidates
    and can be read at any budget; the cap stops a pool that is already certain earning more.

    `weight` is required and has no default: pass ones for an unweighted pool, or each candidate's
    share of its clump for a crowding-discounted one.
    """
    rate = success_of_distance(radii, success_rate, distance)
    # A shell measured at exactly 1 would make one candidate worth infinite credit; the cap would
    # absorb it but the intermediate overflows.
    rate = np.clip(rate, 0.0, 1.0 - 1e-12)
    contribution = np.where(rate > CONVERGENCE_CUT, -np.log1p(-rate), 0.0)
    return np.minimum(np.sum(weight*contribution, axis=-1), cap)


def evaluate(variant, distance, curve, cap=None, weight=None):
    """One entry point, so a driver can score the same cached pool under several variants.

    Each variant is given exactly the constants it can use and refuses the others, because a
    constant a variant ignores is one nobody checks the value of.

    Production does not come through here: `_redistribution_testing_functional` calls
    `excess_count_objective` directly, so a variant added for a measurement cannot reach it.
    """
    if variant == 'shipped':
        if cap is not None or weight is not None:
            raise ValueError("variant='shipped' takes neither a cap nor a weight")
        return excess_count_objective(distance, *shell_targets(curve))
    if variant == 'expected':
        if cap is not None or weight is not None:
            raise ValueError("variant='expected' takes neither a cap nor a weight")
        return expected_success_objective(distance, curve[0], curve[1])
    if variant == 'capped':
        if cap is None:
            raise ValueError("variant='capped' needs a cap")
        if weight is not None:
            raise ValueError("variant='capped' is the unweighted one; call the kernel directly")
        return capped_log_objective(distance, curve[0], curve[1], cap,
                                    np.ones(np.shape(distance)))
    raise ValueError(f'unknown ensemble objective variant {variant!r}')
