import numpy as np

from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn


class ContaminantPlacementError(RuntimeError):
    pass


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
    # Interior dropout: delete peaks from *within* the low-q2 range and backfill from higher q2,
    # which is what undetected weak reflections do to a real pattern. This is a different attack
    # from pushing the whole window outwards -- it punches holes in the low-angle region, where the
    # systematic-absence pattern lives and where the generators take their information, so it
    # degrades candidate *discovery* rather than just the fit. n_drop = 0 is a no-op.
    #
    # n_drop is the number of peaks REMOVED FROM THE FIRST n_peaks + n_drop, not the number of holes
    # left in the nominal list. A fraction n_drop/(n_peaks + n_drop) of the draws lands in the
    # backfill region, where dropping a peak merely selects a different high-q2 one and changes
    # nothing. So the expected hole count is n_drop * n_peaks/(n_peaks + n_drop): measured 2.60 for
    # n_drop=3 and 4.56 for n_drop=6 at n_peaks=20, against 2.61 and 4.62 predicted. This is
    # deliberate -- every peak in the window has the same detection probability, which is the
    # physical model -- but it means the parameter understates nothing and overstates the effect,
    # so callers should record the achieved count rather than assume n_drop.
    q2_full = np.asarray(q2_full, dtype=float)
    q2_full = q2_full[q2_full > 0]
    if n_drop <= 0:
        return q2_full[:n_peaks]
    window = q2_full[:n_peaks + n_drop]
    if window.size <= n_peaks:
        # Not enough peaks to drop any and still fill the list; return what there is and let the
        # caller decide. Reporting the achieved distribution matters more than forcing the count.
        return q2_full[:n_peaks]
    dropped = rng.choice(window.size, size=min(n_drop, window.size - n_peaks), replace=False)
    kept = np.delete(window, dropped)
    return np.sort(kept)[:n_peaks]


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
