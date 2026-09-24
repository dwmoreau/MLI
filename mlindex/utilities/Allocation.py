"""Dividing a whole number of things among several claimants.

Candidate budgets are integers and the shares that divide them are not, so every generator that
splits a budget has to round -- and rounding each share on its own does not preserve the total.
The shortfall is not academic: the ensemble objective rises with the number of candidates, so a
mix whose shares happened to round up would win on that alone.
"""
import numpy as np


def largest_remainder(weight, total):
    """Whole counts proportional to `weight` that sum to `total` exactly.

    Each claimant takes its floor, then the units left over go to the claimants with the largest
    fractional parts -- the largest-remainder rule. A claimant with a small enough weight gets
    nothing, which is intended: that is how a share of zero is expressible at all.

    **Ties go to the earliest claimant.** Equal weights are the common case, not a corner: a
    generator's budget divided among its split groups has every remainder identical, and which
    group gets the odd candidate then rests entirely on this rule. An unstable sort would make
    that arbitrary, so the sort is stable.

    `weight` need not be normalised. It must be non-negative and not sum to zero.
    """
    weight = np.asarray(weight, dtype=float)
    if np.any(weight < 0):
        raise ValueError('largest_remainder needs non-negative weights')
    if not weight.sum() > 0:
        raise ValueError('largest_remainder needs at least one positive weight')
    exact = weight/weight.sum()*total
    counts = np.floor(exact).astype(int)
    short = int(total) - int(counts.sum())
    if short > 0:
        counts[np.argsort(-(exact - counts), kind='stable')[:short]] += 1
    return counts
