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


GENERATORS = ('trees', 'abnn', 'templates')


def check_generator_fractions(fractions):
    """Raise unless `fractions` gives every generator a non-negative share and the shares sum to 1.

    A lattice's fractions are three numbers that must be edited together. Raising one without
    lowering another is the likely mistake, and it would silently hand that lattice more
    candidates than its budget, so it is refused rather than renormalised.
    """
    if set(fractions) != set(GENERATORS):
        raise ValueError(
            f'generator fractions must name exactly {list(GENERATORS)}, got {sorted(fractions)}')
    if any(fractions[name] < 0 for name in GENERATORS):
        raise ValueError(f'generator fractions must be non-negative, got {fractions}')
    total = sum(fractions[name] for name in GENERATORS)
    if abs(total - 1) > 1e-9:
        raise ValueError(f'generator fractions must sum to 1, got {fractions} (sum {total})')


def generator_info_from_fractions(fractions, n_candidates, split_groups):
    """The per-generator, per-split-group candidate counts for one lattice.

    The forest and the network are trained per split group, so each one's share is divided evenly
    among `split_groups`; the templates are not split. Each count is `int(1/k*share*n_candidates)`
    for a generator with `k` groups, which floors every group separately -- a lattice can receive
    a few candidates fewer than its budget. A generator whose share is zero is left out.

    Returns the `generator_info` list the optimizer reads: forest groups, then network groups, then
    the templates.
    """
    check_generator_fractions(fractions)
    generator_info = []
    for name in ('trees', 'abnn'):
        if fractions[name] == 0:
            continue
        k = len(split_groups)
        for split_group in split_groups:
            generator_info.append({
                'generator': name,
                'split_group': split_group,
                'n_unit_cells': int(1/k*fractions[name]*n_candidates),
                })
    if fractions['templates'] != 0:
        generator_info.append({
            'generator': 'templates',
            'n_unit_cells': int(fractions['templates']*n_candidates),
            })
    return generator_info
