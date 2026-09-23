"""What a candidate is worth when it is not alone.

Two candidates that start close together do not fail independently: they share their first
Miller-index assignment and much of their refinement, so a clump of `k` is worth fewer than `k`
tries. Every ensemble score before this one counted it as `k`, which is the assumption
redistribution exists to exploit and the reason a score built without this term is blind to what
redistribution does.

`alpha` is the share of one independent candidate that one member of a clump is worth. It is
measured, not modelled: groups of candidates are started a set distance apart and how often the
whole group fails is compared with what independence predicts, so

    alpha = log P(the group all fail) / sum over members of log(1 - s(d_i))

which is exactly the factor this multiplies each candidate's contribution by. alpha = 1 is
independence; alpha -> 1/k is a clump worth a single candidate.

The table is run output, like the convergence curves, and is not shipped with the package: it is
loaded from a directory the caller names. It carries, per lattice, the separation `delta` within
which two candidates count as one clump, and `alpha` against clump size at that separation.
"""
import os

import numpy as np
from scipy.spatial import cKDTree


# The three cubic lattices, for which the discount cannot currently be measured.
CUBIC = ('cF', 'cI', 'cP')

# What is used for them instead: every candidate counted as fully independent.
#
# This is an ASSUMPTION and not a measurement, and it is the generous direction -- cubic
# candidates do clump, so a cubic pool is overvalued by however much they do. Two consequences
# worth knowing before reading a cubic fit:
#
#   * the score becomes a plain sum over candidates, which is LINEAR in the mix, so a pooled
#     optimum on the simplex is always a corner. A cubic fit will hand the whole budget to one
#     generator, and that is the objective's shape rather than a finding about the generators.
#   * it cannot see what redistribution does, redistribution existing precisely to break clumps.
#
# It is not measurable today because of how the separation job builds a group, not because of
# anything about cubic: SeparatedStartManager displaces members onto a SPHERE of radius delta/2
# about the group centre, and in one free parameter that sphere is two points, so half of every
# group is coincident at any separation asked for. Drawing from the ball, or from a Gaussian,
# would be non-degenerate in one dimension too. If that is ever done, drop a measured file in
# the discount directory and it takes precedence over this automatically.
NO_DISCOUNT = (0.0, np.array([1.0, 2.0]), np.array([1.0, 1.0]))


def discount_path(discount_directory, bravais_lattice):
    """The one place the clump-discount filename is composed."""
    return os.path.join(discount_directory, f'{bravais_lattice}_clump_discount.npz')


def load_clump_discount(discount_directory, bravais_lattice):
    """-> (delta, k, alpha), the separation that defines a clump and the discount against size."""
    path = discount_path(discount_directory, bravais_lattice)
    if not os.path.isfile(path):
        # A measured file always wins, so this never shadows one; see NO_DISCOUNT for what the
        # cubic fallback assumes and what it costs.
        if bravais_lattice in CUBIC:
            return NO_DISCOUNT
        raise FileNotFoundError(
            f'no clump discount for {bravais_lattice} at {path}. It is run output, measured by '
            f'the candidate-separation job, and is not shipped with the package.'
            )
    table = np.load(path)
    delta, k, alpha = float(table['delta']), table['k'], table['alpha']
    if k.shape != alpha.shape or k.size < 2:
        raise ValueError(f'{path}: k and alpha must be matching arrays of at least two points')
    if np.any(alpha > 1.0) or np.any(alpha <= 0.0):
        raise ValueError(f'{path}: alpha lies in (0, 1]; a clump member cannot be worth more '
                         f'than an independent candidate')
    return delta, k, alpha


def clump_weights(xnn, delta, k, alpha):
    """One weight per candidate, from how many of its own pool-mates sit within `delta`.

    Counted on the pool being scored, so a mix that piles candidates into one region is charged
    for it. Interpolated in log clump size, and held at the measured ends beyond them.
    """
    xnn = np.atleast_2d(xnn)
    if delta <= 0:
        return np.ones(xnn.shape[0])
    counts = cKDTree(xnn).query_ball_point(xnn, delta, return_length=True)
    return np.interp(np.log(np.maximum(counts, 1)), np.log(k), alpha)
