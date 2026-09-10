import numpy as np

from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn


class ContaminantPlacementError(RuntimeError):
    """A pattern too crowded to accept the contaminants asked of it."""


# Interior dropout leaves at most this many holes, whatever the caller asks for. Above it the
# mechanism stops being interior dropout: with a 20-peak window and 20 holes, almost every peak in
# the window is replaced by one from beyond it, which translates the window wholesale to high angle
# rather than punching holes in the low-angle region. Ten keeps half the window.
MAX_INTERIOR_DROPOUT = 10


def q2_sigma_params(intercept=None, slope=None):
    """The sigma(q2) = intercept + slope * q2 model, defaulting to the repository's own.

    Exposed so the error's *shape* can be varied and not only its magnitude: a multiplier scales
    both terms together, while the intercept alone changes sigma at low q2 relative to high q2.

    Passing neither argument reproduces the behaviour from before they existed exactly, including
    the number of random values consumed downstream.
    """
    if intercept is None or slope is None:
        from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
        defaults = get_peak_generation_info()['q2_error_params']
        if intercept is None:
            intercept = defaults[0]
        if slope is None:
            slope = defaults[1]
    return float(intercept), float(slope)


def add_q2_error(q2, hkl, multiplier, rng, intercept=None, slope=None):
    intercept, slope = q2_sigma_params(intercept, slope)
    sigma_error = multiplier * (intercept + q2 * slope)
    q2 += rng.normal(loc=0, scale=sigma_error)
    q2 = np.abs(q2)
    if hkl is None:
        return np.sort(q2, axis=1)
    else:
        sort_indices = np.argsort(q2, axis=1)
        q2 = np.take_along_axis(q2, sort_indices, axis=1)
        hkl = np.take_along_axis(hkl, sort_indices[:, :, np.newaxis], axis=1)
        return q2, hkl


def select_peaks_with_nested_dropout(q2_full, n_peaks, n_drop, rng, max_drop=None):
    """The observed window with `n_drop` interior peaks missing, plus everything left over.

    Interior dropout deletes peaks from *within* the nominal low-q2 window and backfills from
    higher q2, which is what undetected weak reflections do to a real pattern. It is a different
    attack from pushing the whole window outwards: it punches holes in the low-angle region, where
    the systematic-absence pattern lives and where the candidate generators take their information,
    so it degrades discovery rather than only the fit.

    The holes are *nested*. The permutation is always drawn at `max_drop` and the holes for a
    smaller `n_drop` are its prefix, so the two-hole set is a subset of the four-hole set, which is
    a subset of the six-hole set. A sparsity ladder built this way is one crystal degrading
    progressively rather than three independent noise realisations. A prefix of
    `rng.choice(..., replace=False)` is a subset by construction, which is the whole trick.

    Drawing at `max_drop` whatever `n_drop` is also keeps the number of random values consumed
    fixed across the ladder, so a bundle differing only in sparsity does not also get a different
    error realisation.

    The hole count is capped by the surplus the entry has to backfill with, by
    `MAX_INTERIOR_DROPOUT`, and by the window itself, so callers record the count achieved rather
    than the count requested.

    Returns (window, surplus, n_holes). The window is `n_peaks` long whenever the entry has any
    surplus at all, so a fixed-length model input is never violated. `surplus` is every remaining
    line, above the window and above the backfill.
    """
    if max_drop is None:
        max_drop = n_drop
    if n_drop > max_drop:
        raise ValueError(f'n_drop {n_drop} exceeds max_drop {max_drop}; the ladder must be drawn '
                         'at its maximum for the nesting to hold')
    q2_full = np.asarray(q2_full, dtype=float)
    q2_full = q2_full[q2_full > 0]
    n_surplus = q2_full.size - n_peaks

    # Drawn whenever the entry has surplus, including at n_drop == 0, so this stream's position
    # depends on the entry alone and never on which bundle is being generated.
    if n_surplus > 0:
        n_holes_max = min(max_drop, n_surplus, MAX_INTERIOR_DROPOUT, n_peaks)
    else:
        n_holes_max = 0
    if n_holes_max > 0:
        holes_all = rng.choice(n_peaks, size=n_holes_max, replace=False)
    else:
        holes_all = np.empty(0, dtype=int)

    n_holes = int(min(max(n_drop, 0), n_holes_max))
    holes = holes_all[:n_holes]
    if n_holes > 0:
        kept = np.delete(q2_full[:n_peaks], holes)
        backfill = q2_full[n_peaks:n_peaks + n_holes]
        window = np.sort(np.concatenate((kept, backfill)))
    else:
        window = q2_full[:n_peaks]
    # After the backfill, not after the nominal window: the backfilled lines are in the window.
    surplus = q2_full[n_peaks + n_holes:]
    return window, surplus, n_holes


def add_contaminants(q2, hkl, n_contaminants, rng, random_n_contaminants=False,
                     max_attempts=None):
    # The whole contaminant set is redrawn until every member clears every peak's half breadth, so
    # acceptance falls off exponentially in n_contaminants and a dense pattern can spin forever.
    # max_attempts=None keeps that unbounded behaviour; an integer caps the redraws and raises, so
    # a sweep over many entries can drop the ones that cannot be contaminated rather than hanging.
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
    q2_broadening_params = get_peak_generation_info()['broadening_params']
    # Breadth is specified as a linear model in q
    # Breadth in q^2 comes from error propagation
    breadth_q = q2_broadening_params[0] + q2_broadening_params[1] * np.sqrt(q2)
    breadth = 2 * breadth_q * np.sqrt(q2)
    n_peaks = q2.shape[1]
    for entry_index in range(q2.shape[0]):
        status = True
        n_attempts = 0
        while status:
            if max_attempts is not None and n_attempts >= max_attempts:
                raise ContaminantPlacementError(
                    f'Could not place {n_contaminants} contaminants in entry {entry_index} '
                    f'within {max_attempts} attempts'
                    )
            n_attempts += 1
            high = q2[entry_index, -1]
            if random_n_contaminants:
                n_contaminants_add = rng.choice(n_contaminants)
            else:
                n_contaminants_add = n_contaminants
            q2_contaminants = rng.uniform(
                low=0.5*q2[entry_index, 0],
                high=high,
                size=n_contaminants_add
                )
            if n_peaks is None:
                difference = np.abs(
                    q2_contaminants[np.newaxis]
                    - q2[entry_index][:, np.newaxis]
                    ).min(axis=0)
            else:
                difference = np.abs(
                    q2_contaminants[np.newaxis]
                    - q2[entry_index, :n_peaks][:, np.newaxis]
                    ).min(axis=0)
            status = np.any(difference[np.newaxis] < 0.5*breadth[entry_index][:, np.newaxis])

        q2_new = np.concatenate((q2[entry_index], q2_contaminants))
        if not hkl is None:
            hkl_new = np.concatenate(
                (hkl[entry_index], np.zeros((n_contaminants_add, 3))),
                axis=0
            )
        sort_indices = np.argsort(q2_new)
        q2[entry_index] = q2_new[sort_indices][:n_peaks]
        if not hkl is None:
            hkl[entry_index, :, 0] = hkl_new[sort_indices, 0][:n_peaks]
            hkl[entry_index, :, 1] = hkl_new[sort_indices, 1][:n_peaks]
            hkl[entry_index, :, 2] = hkl_new[sort_indices, 2][:n_peaks]
    if hkl is None:
        return q2
    else:
        return q2, hkl


def add_second_phase(q2, hkl, partner_q2, n_lines, rng, low_angle_bias=1.0):
    """Inject lines from a real partner cell into an observed pattern.

    `add_contaminants` draws each contaminant at an independent random position. Real contamination
    is a second crystalline phase, so its lines are *correlated* -- consistent with some other
    lattice, and a handful of them arrive together. Independently placed lines are easier to reject
    than real ones, so a bundle built from them is optimistic.

    Which lines: the datasets carry positions and no intensities, and a real second phase shows its
    strong low-angle reflections, so the selection is random but weighted towards low q2 rather
    than being the k lowest lines outright. `low_angle_bias` is a rank draw over the eligible
    lines, index = floor(n_eligible * u ** bias), with 1 a uniform pick.

    Placement is deliberately not `add_contaminants`' rejection loop. That function draws from a
    continuum, so redrawing until nothing collides is its only option. Here the partner offers a
    finite known set, so lines colliding with a real peak are filtered out up front and the draw is
    made from what remains. Rejection sampling on a discrete set degenerates: with two eligible
    lines and two to place there is exactly one possible set, and redrawing it cannot change the
    answer.

    Injected lines enter `hkl` as (0, 0, 0) before the list is re-sorted and truncated back to
    n_peaks, exactly as contaminants do.
    """
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
    q2_broadening_params = get_peak_generation_info()['broadening_params']
    breadth_q = q2_broadening_params[0] + q2_broadening_params[1] * np.sqrt(q2)
    breadth = 2 * breadth_q * np.sqrt(q2)
    n_peaks = q2.shape[1]
    partner_q2 = np.asarray(partner_q2, dtype=float)
    partner_q2 = np.sort(partner_q2[partner_q2 > 0])

    for entry_index in range(q2.shape[0]):
        low = 0.5*q2[entry_index, 0]
        high = q2[entry_index, -1]
        # Only the partner's lines that would fall inside the observed pattern. A line above the
        # last peak is not observable and one far below the first is outside the measured range;
        # these are the bounds add_contaminants draws within.
        eligible = partner_q2[(partner_q2 >= low) & (partner_q2 <= high)]
        if eligible.size == 0:
            raise ContaminantPlacementError(
                f'The second phase has no lines inside entry {entry_index}\'s observed range '
                f'[{low:.4f}, {high:.4f}]'
                )
        # A line too close to a real peak is not resolved as its own reflection, so it is not an
        # observable contaminant. This is add_contaminants' criterion exactly -- the distance to
        # the nearest peak against every peak's half breadth -- rather than the line's own
        # neighbour's breadth, which would be the more physical test. Matching matters more than
        # improving it: the independent-contaminant and second-phase bundles differ only in how
        # their lines are chosen, and a different rejection rule would confound that.
        separation = np.abs(
            eligible[np.newaxis] - q2[entry_index, :n_peaks][:, np.newaxis]
            ).min(axis=0)
        placeable = eligible[
            ~np.any(separation[np.newaxis] < 0.5*breadth[entry_index][:, np.newaxis], axis=0)
            ]
        if placeable.size == 0:
            raise ContaminantPlacementError(
                f'Every one of the second phase\'s {eligible.size} lines in entry '
                f'{entry_index}\'s range overlaps one of its peaks'
                )
        n_add = min(n_lines, placeable.size)

        # A rank draw rather than a position draw, because the partner offers a discrete set of
        # real lines and not a continuum. Without replacement, by redrawing index collisions --
        # which terminates because n_add <= placeable.size and every index stays reachable.
        chosen = set()
        while len(chosen) < n_add:
            if low_angle_bias == 1.0:
                index = int(rng.integers(placeable.size))
            else:
                index = int(placeable.size * rng.uniform()**low_angle_bias)
            chosen.add(min(index, placeable.size - 1))
        q2_second_phase = placeable[sorted(chosen)]

        q2_new = np.concatenate((q2[entry_index], q2_second_phase))
        if hkl is not None:
            hkl_new = np.concatenate((hkl[entry_index], np.zeros((n_add, 3))), axis=0)
        sort_indices = np.argsort(q2_new)
        q2[entry_index] = q2_new[sort_indices][:n_peaks]
        if hkl is not None:
            hkl[entry_index, :, 0] = hkl_new[sort_indices, 0][:n_peaks]
            hkl[entry_index, :, 1] = hkl_new[sort_indices, 1][:n_peaks]
            hkl[entry_index, :, 2] = hkl_new[sort_indices, 2][:n_peaks]
    if hkl is None:
        return q2
    return q2, hkl


def perturb_xnn(xnn_true, convergence_candidates, convergence_distances, minimum_uc, maximum_uc, lattice_system, rng):
    size = (convergence_candidates, xnn_true.size)
    perturbed_unit_cells = []
    for distance in convergence_distances:
        perturbations = rng.uniform(low=-1, high=1, size=size)
        perturbations = distance * perturbations / np.linalg.norm(perturbations, axis=1)[:, np.newaxis]
        perturbed_xnn = xnn_true[np.newaxis] + perturbations
        perturbed_xnn = fix_unphysical(
            xnn=perturbed_xnn,
            rng=rng,
            minimum_unit_cell=minimum_uc,
            maximum_unit_cell=maximum_uc,
            lattice_system=lattice_system
            )
        perturbed_unit_cells.append(get_unit_cell_from_xnn(
            perturbed_xnn,
            partial_unit_cell=True,
            lattice_system=lattice_system
            ))
    return np.concatenate(perturbed_unit_cells, axis=0)
