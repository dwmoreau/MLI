import numpy as np


def get_partial_unit_cell(unit_cell, lattice_system=None, bravais_lattice=None):
    if lattice_system:
        if lattice_system == 'cubic':
            return unit_cell[[0]]
        elif lattice_system in ['hexagonal', 'tetragonal']:
            return unit_cell[[0, 2]]
        elif lattice_system == 'rhombohedral':
            return unit_cell[[0, 3]]
        elif lattice_system == 'orthorhombic':
            return unit_cell[[0, 1, 2]]
        elif lattice_system == 'monoclinic':
            return unit_cell[[0, 1, 2, 4]]
        else:
            return unit_cell
    elif bravais_lattice:
        if bravais_lattice in ['cP', 'cI', 'cF']:
            return unit_cell[[0]]
        elif bravais_lattice in ['hP', 'tP', 'tI']:
            return unit_cell[[0, 2]]
        elif bravais_lattice in ['hR']:
            return unit_cell[[0, 3]]
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            return unit_cell[[0, 1, 2]]
        elif bravais_lattice == ['mP', 'mC']:
            return unit_cell[[0, 1, 2, 4]]
        else:
            return unit_cell


def reciprocal_uc_conversion(unit_cell, partial_unit_cell=False, lattice_system=None):
    if partial_unit_cell and lattice_system != "triclinic":
        if lattice_system in ["cubic", "rhombohedral"]:
            a = unit_cell[:, 0]
            b = unit_cell[:, 0]
            c = unit_cell[:, 0]
        elif lattice_system in ["tetragonal", "hexagonal"]:
            a = unit_cell[:, 0]
            b = unit_cell[:, 0]
            c = unit_cell[:, 1]
        elif lattice_system in ["orthorhombic", "monoclinic"]:
            a = unit_cell[:, 0]
            b = unit_cell[:, 1]
            c = unit_cell[:, 2]
        if lattice_system in ["cubic", "tetragonal", "orthorhombic"]:
            alpha = np.pi / 2
            beta = np.pi / 2
            gamma = np.pi / 2
        elif lattice_system == "hexagonal":
            alpha = np.pi / 2
            beta = np.pi / 2
            gamma = 2 * np.pi / 3
        elif lattice_system == "rhombohedral":
            alpha = unit_cell[:, 1]
            beta = unit_cell[:, 1]
            gamma = unit_cell[:, 1]
        elif lattice_system == "monoclinic":
            alpha = np.pi / 2
            beta = unit_cell[:, 3]
            gamma = np.pi / 2
        else:
            assert False
    elif partial_unit_cell == False or lattice_system == "triclinic":
        a = unit_cell[:, 0]
        b = unit_cell[:, 1]
        c = unit_cell[:, 2]
        alpha = unit_cell[:, 3]
        beta = unit_cell[:, 4]
        gamma = unit_cell[:, 5]

    S = np.array(
        [
            [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)],
            [a * c * np.cos(beta), b * c * np.cos(alpha), c**2],
        ]
    ).T

    # Singular matrices are extremely rare here. They have been observed with triclinic lattices
    # during the off by two check. It is faster to use a try & except statement to catch exceptions
    # than to verify invertibility.
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        print(f"RECIPROCAL-DIRECT UNIT CELL CONVERSION FAILED: {e}")
        print("    RERUNNING CONVERSION WITH SINGULAR MATRIX CHECK")
        # nans in the S matrices makes them non-invertible. They also cause exceptions in the
        # numpy functions that check for invertibility.
        invertible = np.invert(np.any(np.isnan(S), axis=(1, 2)))
        # These are other checks to find which matrices are invertible.
        invertible[invertible] = np.linalg.det(S[invertible]) != 0
        invertible[invertible] = np.isfinite(np.linalg.cond(S[invertible]))
        invertible[invertible] = (
            np.linalg.matrix_rank(S[invertible], hermitian=True) == 3
        )
        S_inv = np.zeros(S.shape)
        S_inv[invertible] = np.linalg.inv(S[invertible])

    diag = np.diagonal(S_inv, axis1=1, axis2=2)
    nonphysical_lengths = np.any(diag <= 0, axis=1)
    if np.count_nonzero(nonphysical_lengths) == 0:
        a_inv = np.sqrt(S_inv[:, 0, 0])
        b_inv = np.sqrt(S_inv[:, 1, 1])
        c_inv = np.sqrt(S_inv[:, 2, 2])

        alpha_arg = S_inv[:, 1, 2] / (b_inv * c_inv)
        beta_arg = S_inv[:, 0, 2] / (a_inv * c_inv)
        gamma_arg = S_inv[:, 0, 1] / (a_inv * b_inv)
        nonphysical_angles = np.any(
            np.abs(np.stack((alpha_arg, beta_arg, gamma_arg), axis=1)) > 1, axis=1
        )
    else:
        physical_lengths = np.invert(nonphysical_lengths)
        a_inv = np.zeros(unit_cell.shape[0])
        b_inv = np.zeros(unit_cell.shape[0])
        c_inv = np.zeros(unit_cell.shape[0])

        a_inv[nonphysical_lengths] = np.nan
        b_inv[nonphysical_lengths] = np.nan
        c_inv[nonphysical_lengths] = np.nan

        a_inv[physical_lengths] = np.sqrt(S_inv[physical_lengths, 0, 0])
        b_inv[physical_lengths] = np.sqrt(S_inv[physical_lengths, 1, 1])
        c_inv[physical_lengths] = np.sqrt(S_inv[physical_lengths, 2, 2])

        alpha_arg = np.zeros(unit_cell.shape[0])
        beta_arg = np.zeros(unit_cell.shape[0])
        gamma_arg = np.zeros(unit_cell.shape[0])

        alpha_arg[nonphysical_lengths] = np.nan
        beta_arg[nonphysical_lengths] = np.nan
        gamma_arg[nonphysical_lengths] = np.nan

        alpha_arg[physical_lengths] = (
            S_inv[physical_lengths, 1, 2] / (b_inv * c_inv)[physical_lengths]
        )
        beta_arg[physical_lengths] = (
            S_inv[physical_lengths, 0, 2] / (a_inv * c_inv)[physical_lengths]
        )
        gamma_arg[physical_lengths] = (
            S_inv[physical_lengths, 0, 1] / (a_inv * b_inv)[physical_lengths]
        )

        nonphysical_angles = nonphysical_lengths

        nonphysical_angles[physical_lengths] = np.any(
            np.abs(
                np.stack(
                    (
                        alpha_arg[physical_lengths],
                        beta_arg[physical_lengths],
                        gamma_arg[physical_lengths],
                    ),
                    axis=1,
                )
            )
            > 1,
            axis=1,
        )

    if np.count_nonzero(nonphysical_angles) == 0:
        alpha_inv = np.arccos(alpha_arg)
        beta_inv = np.arccos(beta_arg)
        gamma_inv = np.arccos(gamma_arg)
    else:
        physical_angles = np.invert(nonphysical_angles)
        alpha_inv = np.zeros(unit_cell.shape[0])
        beta_inv = np.zeros(unit_cell.shape[0])
        gamma_inv = np.zeros(unit_cell.shape[0])

        alpha_inv[nonphysical_angles] = np.nan
        beta_inv[nonphysical_angles] = np.nan
        gamma_inv[nonphysical_angles] = np.nan

        alpha_inv[physical_angles] = np.arccos(alpha_arg[physical_angles])
        beta_inv[physical_angles] = np.arccos(beta_arg[physical_angles])
        gamma_inv[physical_angles] = np.arccos(gamma_arg[physical_angles])

    if partial_unit_cell and lattice_system != "triclinic":
        if lattice_system == "cubic":
            unit_cell_inv = a_inv[:, np.newaxis]
        elif lattice_system in ["tetragonal", "hexagonal"]:
            unit_cell_inv = np.column_stack([a_inv, c_inv])
        elif lattice_system == "orthorhombic":
            unit_cell_inv = np.column_stack([a_inv, b_inv, c_inv])
        elif lattice_system == "rhombohedral":
            unit_cell_inv = np.column_stack([a_inv, alpha_inv])
        elif lattice_system == "monoclinic":
            unit_cell_inv = np.column_stack([a_inv, b_inv, c_inv, beta_inv])
    elif partial_unit_cell == False or lattice_system == "triclinic":
        unit_cell_inv = np.column_stack(
            [a_inv, b_inv, c_inv, alpha_inv, beta_inv, gamma_inv]
        )
    return unit_cell_inv


def get_xnn_from_reciprocal_unit_cell(
    reciprocal_unit_cell, partial_unit_cell=False, lattice_system=None
):
    if partial_unit_cell and lattice_system != "triclinic":
        if lattice_system in ["cubic", "tetragonal", "hexagonal", "orthorhombic"]:
            xnn = reciprocal_unit_cell**2
        elif lattice_system == "rhombohedral":
            xnn = np.column_stack(
                [
                    reciprocal_unit_cell[:, 0] ** 2,
                    2
                    * reciprocal_unit_cell[:, 0] ** 2
                    * np.cos(reciprocal_unit_cell[:, 1]),
                ]
            )
        elif lattice_system == "monoclinic":
            xnn = np.column_stack(
                [
                    reciprocal_unit_cell[:, 0] ** 2,
                    reciprocal_unit_cell[:, 1] ** 2,
                    reciprocal_unit_cell[:, 2] ** 2,
                    2
                    * reciprocal_unit_cell[:, 0]
                    * reciprocal_unit_cell[:, 2]
                    * np.cos(reciprocal_unit_cell[:, 3]),
                ]
            )
    elif partial_unit_cell == False or lattice_system == "triclinic":
        xnn = np.column_stack(
            [
                reciprocal_unit_cell[:, 0] ** 2,
                reciprocal_unit_cell[:, 1] ** 2,
                reciprocal_unit_cell[:, 2] ** 2,
                2
                * reciprocal_unit_cell[:, 1]
                * reciprocal_unit_cell[:, 2]
                * np.cos(reciprocal_unit_cell[:, 3]),
                2
                * reciprocal_unit_cell[:, 0]
                * reciprocal_unit_cell[:, 2]
                * np.cos(reciprocal_unit_cell[:, 4]),
                2
                * reciprocal_unit_cell[:, 0]
                * reciprocal_unit_cell[:, 1]
                * np.cos(reciprocal_unit_cell[:, 5]),
            ]
        )
        xnn[reciprocal_unit_cell[:, 3] == np.pi / 2, 3] = 0
        xnn[reciprocal_unit_cell[:, 4] == np.pi / 2, 4] = 0
        xnn[reciprocal_unit_cell[:, 5] == np.pi / 2, 5] = 0
    return xnn


def get_reciprocal_unit_cell_from_xnn(
    xnn, partial_unit_cell=False, lattice_system=None
):
    if partial_unit_cell and lattice_system != "triclinic":
        if lattice_system in ["cubic", "tetragonal", "hexagonal", "orthorhombic"]:
            reciprocal_unit_cell = np.sqrt(xnn)
        elif lattice_system == "rhombohedral":
            reciprocal_unit_cell = np.column_stack(
                [
                    np.sqrt(xnn[:, 0]),
                    np.arccos(xnn[:, 1] / (2 * xnn[:, 0])),
                ]
            )
        elif lattice_system == "monoclinic":
            ra = np.sqrt(xnn[:, 0])
            rb = np.sqrt(xnn[:, 1])
            rc = np.sqrt(xnn[:, 2])
            rbeta = np.arccos(xnn[:, 3] / (2 * ra * rc))
            reciprocal_unit_cell = np.column_stack([ra, rb, rc, rbeta])
    elif partial_unit_cell == False or lattice_system == "triclinic":
        ra = np.sqrt(xnn[:, 0])
        rb = np.sqrt(xnn[:, 1])
        rc = np.sqrt(xnn[:, 2])
        ralpha = np.arccos(xnn[:, 3] / (2 * rb * rc))
        rbeta = np.arccos(xnn[:, 4] / (2 * ra * rc))
        rgamma = np.arccos(xnn[:, 5] / (2 * ra * rb))
        reciprocal_unit_cell = np.column_stack([ra, rb, rc, ralpha, rbeta, rgamma])
    return reciprocal_unit_cell


def get_unit_cell_from_xnn(xnn, partial_unit_cell=False, lattice_system=None):
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell, lattice_system
    )
    return reciprocal_uc_conversion(
        reciprocal_unit_cell, partial_unit_cell, lattice_system
    )


def get_xnn_from_unit_cell(unit_cell, partial_unit_cell=False, lattice_system=None):
    reciprocal_unit_cell = reciprocal_uc_conversion(
        unit_cell, partial_unit_cell, lattice_system
    )
    return get_xnn_from_reciprocal_unit_cell(
        reciprocal_unit_cell, partial_unit_cell, lattice_system
    )


def get_unit_cell_volume(unit_cell, partial_unit_cell=False, lattice_system=None):
    if partial_unit_cell and lattice_system != "triclinic":
        if lattice_system == "cubic":
            volume = unit_cell[:, 0] ** 3
        elif lattice_system == "tetragonal":
            volume = unit_cell[:, 0] ** 2 * unit_cell[:, 1]
        elif lattice_system == "hexagonal":
            volume = unit_cell[:, 0] ** 2 * unit_cell[:, 1] * np.sin(np.pi / 3)
        elif lattice_system == "rhombohedral":
            volume = unit_cell[:, 0] ** 3 * np.sqrt(
                1 - 3 * np.cos(unit_cell[:, 1]) ** 2 + 2 * np.cos(unit_cell[:, 1]) ** 3
            )
        elif lattice_system == "orthorhombic":
            volume = unit_cell[:, 0] * unit_cell[:, 1] * unit_cell[:, 2]
        elif lattice_system == "monoclinic":
            abc = unit_cell[:, 0] * unit_cell[:, 1] * unit_cell[:, 2]
            volume = abc * np.sin(unit_cell[:, 3])

    elif partial_unit_cell == False or lattice_system == "triclinic":
        a = unit_cell[:, 0]
        b = unit_cell[:, 1]
        c = unit_cell[:, 2]
        calpha = np.cos(unit_cell[:, 3])
        cbeta = np.cos(unit_cell[:, 4])
        cgamma = np.cos(unit_cell[:, 5])
        arg = 1 - calpha**2 - cbeta**2 - cgamma**2 + 2 * calpha * cbeta * cgamma
        volume = (a * b * c) * np.sqrt(arg)
    return volume


def get_hkl_matrix(hkl, lattice_system):
    last_axis = len(hkl.shape) - 1
    # hkl shape:
    # last_axis = 1: n_peaks, 3
    # last_axis = 2: n_entries, n_peaks, 3
    if lattice_system == "triclinic":
        hkl_matrix = np.concatenate(
            (
                hkl[..., :3] ** 2,
                (hkl[..., 1] * hkl[..., 2])[..., np.newaxis],
                (hkl[..., 0] * hkl[..., 2])[..., np.newaxis],
                (hkl[..., 0] * hkl[..., 1])[..., np.newaxis],
            ),
            axis=last_axis,
        )
    elif lattice_system == "monoclinic":
        hkl_matrix = np.concatenate(
            (
                hkl[..., :3] ** 2,
                (hkl[..., 0] * hkl[..., 2])[..., np.newaxis],
            ),
            axis=last_axis,
        )
    elif lattice_system == "orthorhombic":
        hkl_matrix = hkl**2
    elif lattice_system == "tetragonal":
        hkl_matrix = np.stack(
            (
                np.sum(hkl[..., :2] ** 2, axis=last_axis),
                hkl[..., 2] ** 2,
            ),
            axis=last_axis,
        )
    elif lattice_system == "hexagonal":
        hkl_matrix = np.stack(
            (
                hkl[..., 0] ** 2 + hkl[..., 0] * hkl[..., 1] + hkl[..., 1] ** 2,
                hkl[..., 2] ** 2,
            ),
            axis=last_axis,
        )
    elif lattice_system == "rhombohedral":
        hkl_matrix = np.stack(
            (
                np.sum(hkl**2, axis=last_axis),
                hkl[..., 0] * hkl[..., 1]
                + hkl[..., 0] * hkl[..., 2]
                + hkl[..., 1] * hkl[..., 2],
            ),
            axis=last_axis,
        )
    elif lattice_system == "cubic":
        hkl_matrix = np.sum(hkl**2, axis=last_axis)[..., np.newaxis]
    return hkl_matrix


def fix_unphysical(
    xnn=None,
    unit_cell=None,
    rng=None,
    lattice_system=None,
    minimum_unit_cell=2,
    maximum_unit_cell=500,
):
    if rng is None:
        rng = np.random.default_rng()
    if not xnn is None:
        if lattice_system == "triclinic":
            return fix_unphysical_triclinic(
                xnn=xnn,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )
        elif lattice_system == "monoclinic":
            return fix_unphysical_monoclinic(
                xnn=xnn,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )
        elif lattice_system == "rhombohedral":
            return fix_unphysical_rhombohedral(
                xnn=xnn,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )
        elif lattice_system in ["cubic", "orthorhombic", "tetragonal", "hexagonal"]:
            return fix_unphysical_box(
                xnn=xnn,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )
    elif not unit_cell is None:
        if lattice_system == "triclinic":
            return fix_unphysical_triclinic(
                unit_cell=unit_cell,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )
        elif lattice_system == "monoclinic":
            return fix_unphysical_monoclinic(
                unit_cell=unit_cell,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )
        elif lattice_system == "rhombohedral":
            return fix_unphysical_rhombohedral(
                unit_cell=unit_cell,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )
        elif lattice_system in ["cubic", "orthorhombic", "tetragonal", "hexagonal"]:
            return fix_unphysical_box(
                unit_cell=unit_cell,
                rng=rng,
                minimum_unit_cell=minimum_unit_cell,
                maximum_unit_cell=maximum_unit_cell,
            )


def fix_unphysical_triclinic(
    xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500
):
    """
    The purpose of this function is to ensure that RANDOMLY GENERATED triclinic unit cells are physically
    possible. This should not be used on known unit cells
    """
    if rng is None:
        rng = np.random.default_rng()

    if not xnn is None:

        def get_limits(cos_angle_0, cos_angle_1):
            pa = -1
            pb = 2 * cos_angle_0 * cos_angle_1
            pc = 1 - cos_angle_0**2 - cos_angle_1**2
            roots = [
                (-pb - np.sqrt(pb**2 - 4 * pa * pc)) / (2 * pa),
                (-pb + np.sqrt(pb**2 - 4 * pa * pc)) / (2 * pa),
            ]
            if min(roots) > 1:
                lower_root = 0
            else:
                lower_root = max(min(roots), 0)
            if max(roots) < 0:
                upper_root = 1
            else:
                upper_root = min(max(roots), 1)
            return lower_root, upper_root

        xnn[:, :3] = np.abs(xnn[:, :3])

        zero = xnn[:, :3] == 0
        zero_indices = np.sum(zero, axis=1) > 0
        xnn[zero_indices, :3] = np.mean(xnn[~zero_indices, :3])

        # alpha, beta, & gamma > pi/2
        # Then the reciprocal angles < pi/2
        ra = np.sqrt(xnn[:, 0])
        rb = np.sqrt(xnn[:, 1])
        rc = np.sqrt(xnn[:, 2])
        cos_ralpha = xnn[:, 3] / (2 * rb * rc)
        cos_rbeta = xnn[:, 4] / (2 * ra * rc)
        cos_rgamma = xnn[:, 5] / (2 * ra * rb)

        # In principal, cos_rangle could be up to 1.0, although this would likely never occur.
        # This is limited to 0.99 to prevent numerical errors.
        bad_ralpha = np.logical_or(cos_ralpha < 0, cos_ralpha > 0.95)
        bad_rbeta = np.logical_or(cos_rbeta < 0, cos_rbeta > 0.95)
        bad_rgamma = np.logical_or(cos_rgamma < 0, cos_rgamma > 0.95)
        if np.sum(bad_ralpha) > 0:
            xnn[bad_ralpha, 3] = (2 * rb * rc)[bad_ralpha] * rng.uniform(
                low=0, high=0.99, size=np.sum(bad_ralpha)
            )
        if np.sum(bad_rbeta) > 0:
            xnn[bad_rbeta, 4] = (2 * ra * rc)[bad_rbeta] * rng.uniform(
                low=0, high=0.99, size=np.sum(bad_rbeta)
            )
        if np.sum(bad_rgamma) > 0:
            xnn[bad_rgamma, 5] = (2 * ra * rb)[bad_rgamma] * rng.uniform(
                low=0, high=0.99, size=np.sum(bad_rgamma)
            )

        # Unit cell volume:
        # abc * sqrt[1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma)]
        #   Enforce the argument in the sqrt to be positive
        cos_ralpha = xnn[:, 3] / (2 * rb * rc)
        cos_rbeta = xnn[:, 4] / (2 * ra * rc)
        cos_rgamma = xnn[:, 5] / (2 * ra * rb)
        volume_arg = (
            1
            - cos_ralpha**2
            - cos_rbeta**2
            - cos_rgamma**2
            + 2 * cos_ralpha * cos_rbeta * cos_rgamma
        )
        for index in np.argwhere(volume_arg < 0):
            status = True
            while status:
                lower_root, upper_root = get_limits(cos_rbeta[index], cos_rgamma[index])
                cos_ralpha0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(
                    cos_ralpha[index], cos_rgamma[index]
                )
                cos_rbeta0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(cos_ralpha[index], cos_rbeta[index])
                cos_rgamma0 = rng.uniform(low=lower_root, high=upper_root)

                volume_arg = (
                    1
                    - cos_ralpha0**2
                    - cos_rbeta0**2
                    - cos_rgamma0**2
                    + 2 * cos_ralpha0 * cos_rbeta0 * cos_rgamma0
                )
                if volume_arg > 0:
                    status = False
                    xnn[index, 3] = cos_ralpha0 * (2 * rb[index] * rc[index])
                    xnn[index, 4] = cos_rbeta0 * (2 * ra[index] * rc[index])
                    xnn[index, 5] = cos_rgamma0 * (2 * ra[index] * rb[index])
        # If the reciprocal unit cell conversion fails due to a singular matrix, then the returned
        # unit cell is np.nan. This is extremely rare at this point
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=False
        )
        unit_cell = reciprocal_uc_conversion(
            reciprocal_unit_cell, partial_unit_cell=False
        )
        failed = np.any(np.isnan(unit_cell), axis=1)
        if np.count_nonzero(failed) > 0:
            xnn[failed, :] = np.array([1 / 5**2, 1 / 5**2, 1 / 5**2, 0, 0, 0])
        return xnn

    elif not unit_cell is None:

        def get_limits(cos_angle_0, cos_angle_1):
            pa = -1
            pb = 2 * cos_angle_0 * cos_angle_1
            pc = 1 - cos_angle_0**2 - cos_angle_1**2
            roots = [
                (-pb - np.sqrt(pb**2 - 4 * pa * pc)) / (2 * pa),
                (-pb + np.sqrt(pb**2 - 4 * pa * pc)) / (2 * pa),
            ]
            if min(roots) > 0:
                lower_root = -1
            else:
                lower_root = max(min(roots), -1)
            if max(roots) < -1:
                upper_root = 0
            else:
                upper_root = min(max(roots), 0)
            return lower_root, upper_root

        # Unit cell here is direct space. So angles are obtuse. The case above is for the Xnn
        # coordinates which are based on reciprocal space unit cells
        # Angles can occur up to pi or 180 degrees. This is limited to prevent numerical errors.
        maximum_angle = 0.95 * np.pi
        minimum_angle = np.pi / 2

        too_small_lengths = unit_cell[:, :3] < minimum_unit_cell
        too_large_lengths = unit_cell[:, :3] > maximum_unit_cell
        if np.sum(too_small_lengths) > 0:
            indices = np.argwhere(too_small_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=minimum_unit_cell,
                high=1.05 * minimum_unit_cell,
                size=np.sum(too_small_lengths),
            )
        if np.sum(too_large_lengths) > 0:
            indices = np.argwhere(too_large_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=0.95 * maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths),
            )

        too_small_angles = unit_cell[:, 3:] < minimum_angle
        too_large_angles = unit_cell[:, 3:] > maximum_angle

        if np.sum(too_large_angles) > 0:
            indices = np.argwhere(too_large_angles)
            unit_cell[indices[:, 0], 3 + indices[:, 1]] = rng.uniform(
                low=0.95 * maximum_angle,
                high=maximum_angle,
                size=np.sum(too_large_angles),
            )
        if np.sum(too_small_angles) > 0:
            indices = np.argwhere(too_small_angles)
            unit_cell[indices[:, 0], 3 + indices[:, 1]] = rng.uniform(
                low=minimum_angle,
                high=1.05 * minimum_angle,
                size=np.sum(too_small_angles),
            )

        # Unit cell volume:
        # abc * sqrt[1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma)]
        #   * Enforce the argument in the sqrt to be positive
        cos_alpha = np.cos(unit_cell[:, 3])
        cos_beta = np.cos(unit_cell[:, 4])
        cos_gamma = np.cos(unit_cell[:, 5])
        volume_arg = (
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )
        for index in np.argwhere(volume_arg < 0):
            status = True
            while status:
                lower_root, upper_root = get_limits(cos_beta[index], cos_gamma[index])
                cos_alpha0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(cos_alpha[index], cos_gamma[index])
                cos_beta0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(cos_alpha[index], cos_beta[index])
                cos_gamma0 = rng.uniform(low=lower_root, high=upper_root)

                volume_arg = (
                    1
                    - cos_alpha0**2
                    - cos_beta0**2
                    - cos_gamma0**2
                    + 2 * cos_alpha0 * cos_beta0 * cos_gamma0
                )
                if volume_arg > 0:
                    status = False
                    unit_cell[index, 3] = np.arccos(cos_alpha0)
                    unit_cell[index, 4] = np.arccos(cos_beta0)
                    unit_cell[index, 5] = np.arccos(cos_gamma0)

        # If the reciprocal unit cell conversion fails due to a singular matrix, then the returned
        # unit cell is np.nan. This is extremely rare at this point
        reciprocal_unit_cell = reciprocal_uc_conversion(
            unit_cell, partial_unit_cell=False
        )
        failed = np.any(np.isnan(reciprocal_unit_cell), axis=1)
        if np.count_nonzero(failed) > 0:
            unit_cell[failed, :] = np.array([5, 5, 5, np.pi / 2, np.pi / 2, np.pi / 2])
        return unit_cell


def fix_unphysical_rhombohedral(
    xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500
):
    if rng is None:
        rng = np.random.default_rng()

    if not xnn is None:
        if xnn.shape[1] != 2:
            assert False
        # xnn[:, 0] = a**2. Should be positive and not zero
        xnn[:, 0] = np.abs(xnn[:, 0])
        zero_indices = xnn[:, 0] == 0
        if np.sum(zero_indices) > 0:
            xnn[zero_indices, 0] = rng.normal(
                loc=xnn[~zero_indices, 0].mean(),
                scale=xnn[~zero_indices, 0].std(),
                size=np.sum(zero_indices),
            )

        # Direct space & Reciprocal unit cell angle must be
        # between 0 and 120 degrees (2/3 pi radians)
        cos_ralpha = xnn[:, 1] / (2 * xnn[:, 0])
        negative_angle = cos_ralpha >= 1
        large_angle = cos_ralpha < -0.5
        if np.sum(negative_angle) > 0:
            cos_ralpha[negative_angle] = rng.uniform(
                low=0.95, high=1, size=np.sum(negative_angle)
            )
            xnn[negative_angle, 1] = (
                cos_ralpha[negative_angle] * 2 * xnn[negative_angle, 0]
            )
        if np.sum(large_angle) > 0:
            cos_ralpha[large_angle] = rng.uniform(
                low=-0.5, high=-0.49, size=np.sum(large_angle)
            )
            xnn[large_angle, 1] = cos_ralpha[large_angle] * 2 * xnn[large_angle, 0]
        return xnn

    elif not unit_cell is None:
        if unit_cell.shape[1] != 2:
            assert False
        too_small_lengths = unit_cell[:, 0] < minimum_unit_cell
        too_large_lengths = unit_cell[:, 0] > maximum_unit_cell
        if np.sum(too_small_lengths) > 0:
            unit_cell[too_small_lengths, 0] = rng.uniform(
                low=minimum_unit_cell,
                high=1.05 * minimum_unit_cell,
                size=np.sum(too_small_lengths),
            )
        if np.sum(too_large_lengths) > 0:
            unit_cell[too_large_lengths, 0] = rng.uniform(
                low=0.95 * maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths),
            )

        bad_angle = np.logical_or(unit_cell[:, 1] <= 0, unit_cell[:, 1] > 2 * np.pi / 3)
        if np.sum(bad_angle) > 0:
            unit_cell[bad_angle, 1] = rng.uniform(
                low=0, high=2 * np.pi / 3, size=np.sum(bad_angle)
            )
        return unit_cell


def fix_unphysical_box(
    xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500
):
    if rng is None:
        rng = np.random.default_rng()
    if not xnn is None:
        xnn = np.abs(xnn)
        zero = xnn == 0
        zero_indices = np.sum(zero, axis=1) > 0
        xnn[zero_indices] = np.mean(xnn[~zero_indices])
        return xnn
    elif not unit_cell is None:
        too_small_lengths = unit_cell < minimum_unit_cell
        too_large_lengths = unit_cell > maximum_unit_cell
        if np.sum(too_small_lengths) > 0:
            indices = np.argwhere(too_small_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=minimum_unit_cell,
                high=1.05 * minimum_unit_cell,
                size=np.sum(too_small_lengths),
            )
        if np.sum(too_large_lengths) > 0:
            indices = np.argwhere(too_large_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=0.95 * maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths),
            )
        return unit_cell


def fix_unphysical_monoclinic(
    xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500
):
    if rng is None:
        rng = np.random.default_rng()
    if not xnn is None:
        xnn[:, :3] = np.abs(xnn[:, :3])
        zero = xnn[:, :3] == 0
        zero_indices = np.sum(zero[:, :3], axis=1) > 0
        xnn[zero_indices, :3] = np.mean(xnn[~zero_indices, :3])
        # reciprocal space => reciprocal_beta is acute
        # 1 > cos(reciprocal_beta) > 0
        xnn[:, 3] = np.abs(xnn[:, 3])
        cos_rbeta = xnn[:, 3] / (2 * np.sqrt(xnn[:, 0] * xnn[:, 2]))
        unphysical_angle = cos_rbeta >= 1
        if np.sum(unphysical_angle) > 0:
            cos_rbeta[unphysical_angle] = rng.uniform(
                low=0, high=1, size=np.sum(unphysical_angle)
            )
            xnn[unphysical_angle, 3] = cos_rbeta[unphysical_angle] * (
                2 * np.sqrt(xnn[unphysical_angle, 0] * xnn[unphysical_angle, 2])
            )
        return xnn
    elif not unit_cell is None:
        too_small_lengths = unit_cell[:, :3] < minimum_unit_cell
        too_large_lengths = unit_cell[:, :3] > maximum_unit_cell
        if np.sum(too_small_lengths) > 0:
            indices = np.argwhere(too_small_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=minimum_unit_cell,
                high=1.05 * minimum_unit_cell,
                size=np.sum(too_small_lengths),
            )
        if np.sum(too_large_lengths) > 0:
            indices = np.argwhere(too_large_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=0.95 * maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths),
            )

        too_small_angles = unit_cell[:, 3] < np.pi / 2
        too_large_angles = unit_cell[:, 3] > np.pi
        if np.sum(too_small_angles) > 0:
            unit_cell[too_small_angles, 3] = np.pi - unit_cell[too_small_angles, 3]
        if np.sum(too_large_angles) > 0:
            unit_cell[too_large_angles, 3] = rng.uniform(
                low=0.95 * np.pi, high=np.pi, size=np.sum(too_large_angles)
            )
        return unit_cell
