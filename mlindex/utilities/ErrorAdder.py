import numpy as np

from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn


class ContaminantPlacementError(RuntimeError):
    pass


# Interior dropout is capped here, not left to the caller. Measured over the 5 955 S04 entries, the
# median entry carries ~60 peaks, so the entry's own surplus almost never binds below 20 -- and at
# n_drop = 20 the mechanism removes 18.9 of the nominal 20 and refills from peaks 21-40. That is no
# longer punching holes in the low-angle window; it is translating the window wholesale to high
# angle, which is the distinct attack this function's docstring contrasts itself against. Ten keeps
# half the nominal window, so what the mechanism does still matches what it claims to do.
MAX_INTERIOR_DROPOUT = 10


def add_q2_error(q2, hkl, multiplier, rng):
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
    q2_error_params = get_peak_generation_info()['q2_error_params']
    sigma_error = multiplier * (q2_error_params[0] + q2 * q2_error_params[1])
    q2 += rng.normal(loc=0, scale=sigma_error)
    q2 = np.abs(q2)
    if hkl is None:
        return np.sort(q2, axis=1)
    else:
        sort_indices = np.argsort(q2, axis=1)
        q2 = np.take_along_axis(q2, sort_indices, axis=1)
        hkl = np.take_along_axis(hkl, sort_indices[:, :, np.newaxis], axis=1)
        return q2, hkl


def select_peaks_with_dropout(q2_full, n_peaks, n_drop, rng):
    # Interior dropout: delete peaks from *within* the nominal low-q2 window and backfill from
    # higher q2, which is what undetected weak reflections do to a real pattern. This is a different
    # attack from pushing the whole window outwards -- it punches holes in the low-angle region,
    # where the systematic-absence pattern lives and where the generators take their information, so
    # it degrades candidate *discovery* rather than just the fit. n_drop = 0 is a no-op.
    #
    # n_drop is the number of holes left in the NOMINAL first n_peaks, capped by how many surplus
    # peaks the entry has to backfill with and by MAX_INTERIOR_DROPOUT: a 22-peak entry can give up
    # 2 and no more, a 26-peak entry can give up 6, and a 20-peak entry cannot give up any. The
    # returned list is always as long as the input allows, so the fixed-length ONNX generator input
    # is never violated (F-044).
    #
    # This changed on 2026-08-16. It used to draw the deletions from a window of n_peaks + n_drop,
    # so a fraction n_drop/(n_peaks + n_drop) of them landed in the backfill region and did nothing,
    # leaving n_drop * n_peaks/(n_peaks + n_drop) holes rather than n_drop. The rationale was that
    # every peak in the window then has the same detection probability. The cost was that a hole
    # count could not be requested: at n_peaks=20 a 26-peak entry saturated at 4.62 holes on average
    # and no value of n_drop reached 6, because raising it widened the window as fast as it added
    # draws. S04's C5 needs "as aggressive as the entry allows", so the parameter now means what it
    # says. Callers still record the achieved count, since the cap binds on thin entries.
    #
    # NOTE: S02's 18 `_drop` mechanism bundles were generated under the old semantics and their
    # n_dropout is not comparable with S04's C4/C5. They are exploratory bundles, not the frozen C1
    # nominal the split manifest is pinned to, so nothing downstream depends on the difference.
    q2_full = np.asarray(q2_full, dtype=float)
    q2_full = q2_full[q2_full > 0]
    if n_drop <= 0:
        return q2_full[:n_peaks]
    n_surplus = q2_full.size - n_peaks
    if n_surplus <= 0:
        # Nothing to backfill from, so dropping a peak would shorten the list. Return what there is
        # and let the caller record that the requested dropout was not achieved.
        return q2_full[:n_peaks]
    # Three caps. The surplus is what the entry can backfill with; MAX_INTERIOR_DROPOUT is the
    # mechanism's own ceiling, above which it stops being interior dropout at all; n_peaks is the
    # arithmetic floor under both, and binds only if a caller shortens the nominal window.
    n_holes = min(n_drop, n_surplus, MAX_INTERIOR_DROPOUT, n_peaks)
    dropped = rng.choice(n_peaks, size=n_holes, replace=False)
    kept = np.delete(q2_full[:n_peaks], dropped)
    backfill = q2_full[n_peaks:n_peaks + n_holes]
    return np.sort(np.concatenate((kept, backfill)))


def add_contaminants(q2, hkl, n_contaminants, rng, random_n_contaminants=False, max_attempts=None,
                     low_angle_bias=1.0):
    # The whole contaminant set is redrawn until every member clears every peak's half breadth,
    # so acceptance falls off exponentially in n_contaminants and a dense pattern can spin
    # forever. max_attempts=None keeps that unbounded behaviour; an integer caps the redraws
    # and raises instead, so a caller sweeping many entries can drop the ones that cannot be
    # contaminated rather than hanging on them.
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
            if not max_attempts is None and n_attempts >= max_attempts:
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
            # low_angle_bias biases the draw towards low q2 via q2 = low + (high-low) * u**bias.
            # bias = 1 is the original uniform draw; bias = 2 is uniform in q rather than q2, which
            # is roughly where a second phase's visible lines sit. Measured over 400 mP entries, the
            # fraction of contaminants landing within the first five real peaks -- the region the
            # generators index from -- is 35% at bias 1, 50% at 1.5, 61% at 2 and 72% at 3, with the
            # median landing position moving from 47% of the q2 window to 8%.
            low = 0.5*q2[entry_index, 0]
            if low_angle_bias == 1.0:
                q2_contaminants = rng.uniform(low=low, high=high, size=n_contaminants_add)
            else:
                q2_contaminants = low + (high - low) * rng.uniform(
                    size=n_contaminants_add)**low_angle_bias
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


def add_second_phase(q2, hkl, partner_q2, n_lines, rng, max_attempts=None, low_angle_bias=1.0):
    # add_contaminants draws each contaminant at an independent random position. Real contamination
    # is a second crystalline phase, so its lines are *correlated* -- they are consistent with some
    # other lattice, and a handful of them arrive together. Uniform contaminants are therefore
    # easier to reject than real ones, which makes bundle C3 optimistic. This injects lines from an
    # actual second cell instead.
    #
    # Which lines, given that PROTOCOL section 3 rule 3 forbids intensities and the datasets carry
    # only positions: a real second phase shows its strong low-angle reflections, so the selection
    # is random but weighted towards low q2 rather than being the k lowest lines outright. The
    # weighting reuses add_contaminants' low_angle_bias convention as a rank draw over the eligible
    # lines -- index = floor(n_eligible * u**bias), bias 1 being a uniform pick -- so the parameter
    # means the same thing in both mechanisms.
    #
    # Structure otherwise follows add_contaminants exactly: the whole set is redrawn until every
    # member clears every real peak's half breadth, max_attempts turns an unplaceable pattern into
    # a recorded skip rather than a hang, and the injected lines enter hkl as (0, 0, 0) before the
    # list is re-sorted and truncated back to n_peaks.
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
        # Only the partner's lines that would actually fall inside the observed pattern. A line
        # above the last peak is not observable and one far below the first is outside the
        # measured range; these are the same bounds add_contaminants draws within.
        eligible = partner_q2[(partner_q2 >= low) & (partner_q2 <= high)]
        if eligible.size == 0:
            raise ContaminantPlacementError(
                f'The second phase has no lines inside entry {entry_index}\'s observed range '
                f'[{low:.4f}, {high:.4f}]'
                )
        n_add = min(n_lines, eligible.size)

        status = True
        n_attempts = 0
        while status:
            if not max_attempts is None and n_attempts >= max_attempts:
                raise ContaminantPlacementError(
                    f'Could not place {n_add} second-phase lines in entry {entry_index} '
                    f'within {max_attempts} attempts'
                    )
            n_attempts += 1
            # Rank draw rather than a position draw: the second phase offers a discrete set of
            # real lines, not a continuum. Sampled without replacement by redrawing collisions,
            # which is cheap because n_add is small and eligible is not.
            chosen = set()
            while len(chosen) < n_add:
                if low_angle_bias == 1.0:
                    index = rng.integers(eligible.size)
                else:
                    index = int(eligible.size * rng.uniform()**low_angle_bias)
                chosen.add(min(index, eligible.size - 1))
            q2_second_phase = eligible[sorted(chosen)]

            difference = np.abs(
                q2_second_phase[np.newaxis]
                - q2[entry_index, :n_peaks][:, np.newaxis]
                ).min(axis=0)
            status = np.any(difference[np.newaxis] < 0.5*breadth[entry_index][:, np.newaxis])

        q2_new = np.concatenate((q2[entry_index], q2_second_phase))
        if not hkl is None:
            hkl_new = np.concatenate(
                (hkl[entry_index], np.zeros((n_add, 3))),
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
