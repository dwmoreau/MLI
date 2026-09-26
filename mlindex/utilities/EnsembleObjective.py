"""What a pool of candidate unit cells is worth, scored against a convergence curve.

The pool is what the generators produce before any refinement; the score says how likely it is to
hold a cell close enough to the true one to refine to it. It is what the generator mix is chosen
against.

The score is the expected number of candidates that converge. It replaced one that weighted a
candidate by the whole tail of the curve beyond it rather than by its own chance -- the two agree
on eleven of the fourteen lattices and this one is far better conditioned.

Candidates are NOT assumed independent. Each term carries that candidate's share of its own clump,
from `ClumpDiscount`, because two candidates starting close together succeed or fail together. The
weight is a required argument with no default: a caller that has not decided what its candidates
are worth cannot score a pool by accident.
"""
import numpy as np

from mlindex.utilities.ConvergenceCurve import success_of_distance


# Candidates whose success rate is at or below this are left out. Below it the number needed for
# one success runs away, and the curve's own measurement is thinnest there.
CONVERGENCE_CUT = 0.01


def expected_success_objective(distance, radii, success_rate, weight):
    """The expected number of INDEPENDENT candidates that converge: sum of w_i s(d_i). MAXIMISED.

    `distance` and `weight` are (..., n_candidates) and the result is (...), so one pool gives a
    scalar and a stack of them gives one score each. Combining patterns is the caller's business.

    `weight` is required and has no default. Pass `ClumpDiscount.clump_weights` for a pool whose
    positions are known, or ones only where the candidates are known to be independent -- which in
    a real generated pool they are not.
    """
    rate = success_of_distance(radii, success_rate, distance)
    return np.sum(weight*np.where(rate > CONVERGENCE_CUT, rate, 0.0), axis=-1)
