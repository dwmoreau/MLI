import numpy as np

from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn


def add_q2_error(q2, hkl, multiplier, rng):
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


def add_contaminants(q2, hkl, n_contaminants, rng, random_n_contaminants=False):
    q2_broadening_params = get_peak_generation_info()['broadening_params']
    breadth = q2_broadening_params[0] + q2_broadening_params[1] * q2
    n_peaks = q2.shape[1]
    for entry_index in range(q2.shape[0]):
        status = True
        while status:
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
