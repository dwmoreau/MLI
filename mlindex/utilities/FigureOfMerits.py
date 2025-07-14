import numpy as np

from mlindex.utilities.UnitCellTools import get_hkl_matrix
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume


def get_M20_from_xnn(q2_obs, xnn, hkl, hkl_ref, lattice_system):
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    q2_ref_calc = np.sum(hkl2_ref * xnn[:, np.newaxis, :], axis=2)
    return get_M20(q2_obs, q2_calc, q2_ref_calc)


def get_M20(q2_obs, q2_calc, q2_ref_calc):
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    smaller_ref_peaks = q2_ref_calc < q2_calc[:, -1][:, np.newaxis]
    np.putmask(q2_ref_calc, ~smaller_ref_peaks, 0)
    last_smaller_ref_peak = np.max(q2_ref_calc, axis=1)
    N = np.sum(smaller_ref_peaks, axis=1)

    # There is an unknown issue that causes q2_calc to be all zero
    # These cases are caught and the M20 score is returned as zero.
    # Also catch cases where N == 0 for all peaks
    good_indices = np.logical_and(q2_calc.sum(axis=1) != 0, N != 0)
    expected_discrepancy = np.zeros(q2_calc.shape[0])
    expected_discrepancy[good_indices] = last_smaller_ref_peak[good_indices] / (
        2 * N[good_indices]
    )
    M20 = expected_discrepancy / discrepancy
    return M20


def get_M20_likelihood_from_xnn(q2_obs, xnn, hkl, lattice_system, bravais_lattice):
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
    )
    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
    )
    log_likelihood, probability, M = get_M20_likelihood(
        q2_obs, q2_calc, bravais_lattice, reciprocal_volume
    )
    return log_likelihood, probability, M


def get_M20_likelihood(q2_obs, q2_calc, bravais_lattice, reciprocal_volume):
    # This was inspired by Taupin 1988
    # Probability that a peak is correctly assigned:
    # arg = Expected number of peaks within error from random unit cell
    # P = 1 / (1 + arg)
    mu, nu = get_multiplicity_taupin88(bravais_lattice)
    observed_difference2 = (np.sqrt(q2_obs[np.newaxis]) - np.sqrt(q2_calc)) ** 2
    # There is an upstream error where reciprocal volumes can be very small.
    # Adding 1e-100 here prevents division by zero errors
    arg = (
        8
        * np.pi
        * q2_obs
        * np.sqrt(observed_difference2)
        / (reciprocal_volume[:, np.newaxis] * mu + 1e-100)
    )
    probability = 1 / (1 + arg)
    # The 1e-100 factor prevents np.log(~0) = -infinity
    M = -1 / np.log(2) * np.sum(np.log(1 - np.exp(-arg) + 1e-100), axis=1)
    return -np.sum(np.log(probability + 1e-100), axis=1), probability, M


def get_multiplicity_taupin88(bravais_lattice):
    # The commented out returns come from Taupin 1988
    # The others are from empirically plotting the
    # non systematic absences
    if bravais_lattice == "cF":
        return 4 * 32, 1
    elif bravais_lattice == "cI":
        return 2 * 32, 1
    elif bravais_lattice == "cP":
        return 1 * 32, 1
    elif bravais_lattice == "hP":
        # return 1*24, 2
        return 1 * 14, 2
    elif bravais_lattice == "hR":
        # return 1*24, 2
        return 1 * 8, 2
    elif bravais_lattice == "tI":
        # return 2*16, 2
        return 2 * 13, 2
    elif bravais_lattice == "tP":
        # return 1*16, 2
        return 1 * 13, 2
    elif bravais_lattice in ["oC", "oI"]:
        # return 2*8, 3
        return 2 * 7, 3
    elif bravais_lattice == "oF":
        # return 4*8, 3
        return 4 * 7, 3
    elif bravais_lattice == "oP":
        # return 1*8, 3
        return 1 * 7, 3
    elif bravais_lattice == "mC":
        # return 2*4, 4
        return 2 * 3.2, 4
    elif bravais_lattice == "mP":
        # return 1*4, 4
        return 1 * 3.5, 4
    elif bravais_lattice == "aP":
        # return 1*2, 6
        return 1 * 1.8, 6


def get_M20_sym_reversed(q2_obs, xnn, hkl, hkl_ref, lattice_system):
    # This function is broken because there is no get_multiplicity function
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    q2_ref_calc = np.sum(hkl2_ref * xnn[:, np.newaxis, :], axis=2)
    multiplicity = get_multiplicity(
        hkl.reshape((hkl.shape[0] * hkl.shape[1], hkl.shape[2])), lattice_system
    ).reshape(hkl.shape[:2])
    multiplicity_ref = get_multiplicity(hkl_ref, "monoclinic")

    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    smaller_ref_peaks = q2_ref_calc < q2_calc[:, -1][:, np.newaxis]
    last_smaller_ref_peak = np.zeros(q2_calc.shape[0])
    expected_discrepancy_reversed = (q2_obs[-1] - q2_obs[0]) / (2 * 20)
    discrepancy_reversed = np.zeros(q2_calc.shape[0])
    for i in range(q2_calc.shape[0]):
        q2_ref_smaller = q2_ref_calc[i, smaller_ref_peaks[i]]
        multiplicities_ref_smaller = multiplicity_ref[smaller_ref_peaks[i]]
        sort_indices = np.argsort(q2_ref_smaller)
        q2_ref_smaller = q2_ref_smaller[sort_indices]
        multiplicities_ref_smaller = multiplicities_ref_smaller[sort_indices]
        last_smaller_ref_peak[i] = q2_ref_smaller[-1]

        N_calc = np.sum(1 / multiplicities_ref_smaller)
        differences = np.min(
            np.abs(q2_ref_smaller[np.newaxis] - q2_obs[:, np.newaxis]), axis=0
        )
        discrepancy_reversed[i] = (
            np.sum(differences / multiplicities_ref_smaller) / N_calc
        )

    N = np.sum(smaller_ref_peaks, axis=1)
    expected_discrepancy = last_smaller_ref_peak / (2 * N)
    M20 = expected_discrepancy / discrepancy
    M20_reversed = expected_discrepancy_reversed / discrepancy_reversed
    M20_sym = M20 * M20_reversed
    return M20, M20_sym, M20_reversed


def get_q2_calc_triplets(triplets_obs, hkl, xnn, lattice_system):
    # This gets symmetry operations for a given lattice system
    mi_sym = get_hkl_triplet_symmetry(lattice_system)

    # triplets_obs columns are: peak_0 index, peak_1 index, |q0 - q1|**2, ???
    # triplets_obs is a float array, so round before casting to integer
    hkl0 = np.take(hkl, np.round(triplets_obs[:, 0], decimals=0).astype(int), axis=1)
    hkl1 = np.take(hkl, np.round(triplets_obs[:, 1], decimals=0).astype(int), axis=1)
    hkl0_sym = np.matmul(mi_sym, hkl0[:, :, np.newaxis, :, np.newaxis])[:, :, :, :, 0]

    # q0 - q1 is calculated from hkl_0 - hkl_1
    hkl_diff = hkl0_sym - hkl1[:, :, np.newaxis, :]
    hkl2_diff = get_hkl_matrix(hkl_diff, lattice_system)
    q2_diff_calc_sym = np.sum(xnn[:, np.newaxis, np.newaxis, :] * hkl2_diff, axis=3)
    difference = np.abs(
        triplets_obs[:, 2][np.newaxis, :, np.newaxis] - q2_diff_calc_sym
    )
    q2_diff_calc = np.take_along_axis(
        q2_diff_calc_sym, np.argmin(difference, axis=2)[:, :, np.newaxis], axis=2
    )[:, :, 0]
    return q2_diff_calc


def get_M_triplet_from_xnn(triplets_obs, hkl, xnn, lattice_system, bravais_lattice):
    q2_diff_calc = get_q2_calc_triplets(triplets_obs, hkl, xnn, lattice_system)
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
    )
    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
    )
    _, _, M20_triplet = get_M20_likelihood(
        triplets_obs[:, 2], q2_diff_calc, bravais_lattice, reciprocal_volume
    )
    return M20_triplet


def get_M_triplet(
    q2_obs, q2_calc, triplets_obs, hkl, xnn, lattice_system, bravais_lattice
):
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
    )
    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
    )
    _, _, M_likelihood_primary = get_M20_likelihood(
        q2_obs, q2_calc, bravais_lattice, reciprocal_volume
    )

    # q2_diff_calc is the magnitude of the calculated difference between q0 and q1
    # It is the calculated value of triplet_obs[:, 2]
    q2_diff_calc = get_q2_calc_triplets(triplets_obs, hkl, xnn, lattice_system)
    _, _, M_likelihood_triplet = get_M20_likelihood(
        triplets_obs[:, 2], q2_diff_calc, bravais_lattice, reciprocal_volume
    )
    return np.column_stack((M_likelihood_primary, M_likelihood_triplet))


def get_hkl_triplet_symmetry(lattice_system):
    mi_sym = np.stack(
        [
            np.eye(3),
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                ]
            ),
            np.array(
                [
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
        ],
        axis=0,
    )
    if lattice_system in ["hexagonal", "tetragonal", "cubic", "rhombohedral"]:
        # abc
        # bac
        mi_permutations = [
            np.eye(3),
            np.array(
                [
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                ]
            ),
        ]
    if lattice_system in ["cubic", "rhombohedral"]:
        # acb
        # bca
        # cba
        # cab
        mi_permutations += [
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            ),
        ]
    if lattice_system in ["hexagonal", "tetragonal", "cubic", "rhombohedral"]:
        mi_permutations = np.stack(mi_permutations, axis=0)
        mi_sym_extra = np.matmul(
            mi_sym[:, np.newaxis, :, :], mi_permutations[np.newaxis, :, :, :]
        )
        mi_sym = mi_sym_extra.reshape(
            (mi_sym_extra.shape[0] * mi_sym_extra.shape[1], 3, 3)
        )
    return mi_sym


def get_triplet_hkl_ref(hkl_ref, lattice_system):
    def pairing(k1, k2):
        k1 = k1 + 1000
        k2 = k2 + 1000
        return (k1 + k2) * (k1 + k2 + 1) / 2 + k2

    # mi_sym:           n_sym x 3 x 3
    # hkl_ref:          n_ref x 3
    # hkl_ref_sym:      n_ref x n_sym x 3
    # triplet_hkl_diff: n_ref, n_ref, n_sym, 3
    n_ref = hkl_ref.shape[0]
    mi_sym = get_hkl_triplet_symmetry(lattice_system)
    hkl_ref_sym = np.matmul(mi_sym[np.newaxis], hkl_ref[:, np.newaxis, :, np.newaxis])[
        :, :, :, 0
    ]
    triplet_hkl_diff = (
        hkl_ref_sym[:, np.newaxis, :, :] - hkl_ref[np.newaxis, :, np.newaxis, :]
    )
    triplet_hkl2_diff = get_hkl_matrix(triplet_hkl_diff, lattice_system)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    if lattice_system == "cubic":
        hash_triplet_diff = triplet_hkl2_diff[:, :, :, 0]
        hash_ref = hkl2_ref[:, 0]
    elif lattice_system in ["hexagonal", "tetragonal", "rhombohedral"]:
        hash_triplet_diff = pairing(
            triplet_hkl2_diff[:, :, :, 0], triplet_hkl2_diff[:, :, :, 1]
        )
        hash_ref = pairing(hkl2_ref[:, 0], hkl2_ref[:, 1])
    elif lattice_system == "orthorhombic":
        hash_triplet_diff = pairing(
            triplet_hkl2_diff[:, :, :, 0],
            pairing(triplet_hkl2_diff[:, :, :, 1], triplet_hkl2_diff[:, :, :, 2]),
        )
        hash_ref = pairing(hkl2_ref[:, 0], pairing(hkl2_ref[:, 1], hkl2_ref[:, 2]))
    elif lattice_system == "monoclinic":
        hash_triplet_diff = pairing(
            triplet_hkl2_diff[:, :, :, 0],
            pairing(
                triplet_hkl2_diff[:, :, :, 1],
                pairing(triplet_hkl2_diff[:, :, :, 2], triplet_hkl2_diff[:, :, :, 3]),
            ),
        )
        hash_ref = pairing(
            hkl2_ref[:, 0],
            pairing(hkl2_ref[:, 1], pairing(hkl2_ref[:, 2], hkl2_ref[:, 3])),
        )
    elif lattice_system == "triclinic":
        hash_triplet_diff = pairing(
            triplet_hkl2_diff[:, :, :, 0],
            pairing(
                triplet_hkl2_diff[:, :, :, 1],
                pairing(
                    triplet_hkl2_diff[:, :, :, 2],
                    pairing(
                        triplet_hkl2_diff[:, :, :, 3],
                        pairing(
                            triplet_hkl2_diff[:, :, :, 4], triplet_hkl2_diff[:, :, :, 5]
                        ),
                    ),
                ),
            ),
        )
        hash_ref = pairing(
            hkl2_ref[:, 0],
            pairing(
                hkl2_ref[:, 1],
                pairing(
                    hkl2_ref[:, 2],
                    pairing(
                        hkl2_ref[:, 3],
                        pairing(
                            hkl2_ref[:, 4],
                            hkl2_ref[:, 5],
                        ),
                    ),
                ),
            ),
        )

    triplet_hkl_ref = [[None for _ in range(n_ref)] for _ in range(n_ref)]
    same = (
        hash_triplet_diff[:, :, :, np.newaxis]
        == hash_ref[np.newaxis, np.newaxis, np.newaxis, :]
    )
    for i in range(n_ref - 1):
        for j in range(i + 1, n_ref):
            indices = []
            for k in range(mi_sym.shape[0]):
                indices_here = np.argwhere(same[i, j, k])
                if indices_here.size > 0:
                    indices.append(indices_here[0][0])
            triplet_hkl_ref[i][j] = list(set(indices))
    return triplet_hkl_ref
