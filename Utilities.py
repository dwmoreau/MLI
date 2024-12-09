import csv
import gemmi
import numpy as np
import os
import scipy.spatial


def get_peak_generation_info():
    # This information gets used in GenerateDataset.py and Augmentor.py
    # The CCDC API generates diffraction patterns with the wavelength 1.54 (Cu K alpha)
    # The CCDC powder patterns are generated with 'broadening_fwhm'
    # The cctbx powder patterns are generated with 'broadening_params'
    # The wavelength of 0.827 is the approximate wavelength during 2024A8043 and 2023B8059
    # experiments at SACLA 
    # 'broadening_params': np.array([0.0001, 0.005]),
    # 'broadening_multiples': np.array([0.5, 1, 1.5]),
    #wavelength = 1.54

    wavelength = 0.827
    dtheta2 = 0.01
    theta2_min = 1
    theta2_max = 25
    theta2_pattern = np.arange(theta2_min, theta2_max, dtheta2)
    peak_generation_info = {
        'broadening_fwhm': 0.1,
        'broadening_params': np.array([0.00007, 0.002]),
        'broadening_multiples': np.array([2/3, 1, 1.5]),
        'broadening_tags': ['0.66', '1', '1.5'],
        'wavelength': wavelength,
        'dtheta2': dtheta2,
        'theta2_min': theta2_min,
        'theta2_max': theta2_max,
        'theta2_pattern': theta2_pattern,
        'q2_pattern': (2 * np.sin(theta2_pattern/2 * np.pi/180) / wavelength)**2,
        }
    return peak_generation_info


def reciprocal_uc_conversion(unit_cell, partial_unit_cell=False, lattice_system=None):
    if partial_unit_cell and lattice_system != 'triclinic':
        if lattice_system in ['cubic', 'rhombohedral']:
            a = unit_cell[:, 0]
            b = unit_cell[:, 0]
            c = unit_cell[:, 0]
        elif lattice_system in ['tetragonal', 'hexagonal']:
            a = unit_cell[:, 0]
            b = unit_cell[:, 0]
            c = unit_cell[:, 1]
        elif lattice_system in ['orthorhombic', 'monoclinic']:
            a = unit_cell[:, 0]
            b = unit_cell[:, 1]
            c = unit_cell[:, 2]
        if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
            alpha = np.pi/2
            beta = np.pi/2
            gamma = np.pi/2
        elif lattice_system == 'hexagonal':
            alpha = np.pi/2
            beta = np.pi/2
            gamma = 2*np.pi/3
        elif lattice_system == 'rhombohedral':
            alpha = unit_cell[:, 1]
            beta = unit_cell[:, 1]
            gamma = unit_cell[:, 1]
        elif lattice_system == 'monoclinic':
            alpha = np.pi/2
            beta = unit_cell[:, 3]
            gamma = np.pi/2
        else:
            assert False
    elif partial_unit_cell == False or lattice_system == 'triclinic':
        a = unit_cell[:, 0]
        b = unit_cell[:, 1]
        c = unit_cell[:, 2]
        alpha = unit_cell[:, 3]
        beta = unit_cell[:, 4]
        gamma = unit_cell[:, 5]

    S = np.array([
        [a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
        [a*b*np.cos(gamma), b**2, b*c*np.cos(alpha)],
        [a*c*np.cos(beta), b*c*np.cos(alpha), c**2]
        ]).T

    # Singular matrices are extremely rare here. They have been observed with triclinic lattices
    # during the off by two check.
    success = True
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        print(f'RECIPROCAL-DIRECT UNIT CELL CONVERSION FAILED: {e}')
        print('    RERUNNING CONVERSION WITH SINGULAR MATRIX CHECK')
        # Using np.linalg.matrix_rank(S, hermitian=True) == 3, gives an error here.
        invertible = np.all(np.column_stack((
            np.linalg.det(S) != 0,
            np.isfinite(np.linalg.cond(S))
            )), axis=1)
        S_inv = np.zeros(S.shape)
        S_inv[invertible] = np.linalg.inv(S[invertible])
        success = False

    if success:
        a_inv = np.sqrt(S_inv[:, 0, 0])
        b_inv = np.sqrt(S_inv[:, 1, 1])
        c_inv = np.sqrt(S_inv[:, 2, 2])

        alpha_inv = np.arccos(S_inv[:, 1, 2] / (b_inv * c_inv))
        beta_inv = np.arccos(S_inv[:, 0, 2] / (a_inv * c_inv))
        gamma_inv = np.arccos(S_inv[:, 0, 1] / (a_inv * b_inv))
    else:
        a_inv = np.zeros(unit_cell.shape[0])
        b_inv = np.zeros(unit_cell.shape[0])
        c_inv = np.zeros(unit_cell.shape[0])
        alpha_inv = np.zeros(unit_cell.shape[0])
        beta_inv = np.zeros(unit_cell.shape[0])
        gamma_inv = np.zeros(unit_cell.shape[0])

        a_inv[~invertible] = np.nan
        b_inv[~invertible] = np.nan
        c_inv[~invertible] = np.nan
        alpha_inv[~invertible] = np.nan
        beta_inv[~invertible] = np.nan
        gamma_inv[~invertible] = np.nan

        a_inv[invertible] = np.sqrt(S_inv[invertible, 0, 0])
        b_inv[invertible] = np.sqrt(S_inv[invertible, 1, 1])
        c_inv[invertible] = np.sqrt(S_inv[invertible, 2, 2])

        alpha_inv[invertible] = np.arccos(S_inv[invertible, 1, 2] / (b_inv * c_inv)[invertible])
        beta_inv[invertible] = np.arccos(S_inv[invertible, 0, 2] / (a_inv * c_inv)[invertible])
        gamma_inv[invertible] = np.arccos(S_inv[invertible, 0, 1] / (a_inv * b_inv)[invertible])
    
    if partial_unit_cell and lattice_system != 'triclinic':
        if lattice_system == 'cubic':
            unit_cell_inv = a_inv[:, np.newaxis]
        elif lattice_system in ['tetragonal', 'hexagonal']:
            unit_cell_inv = np.column_stack([a_inv, c_inv])
        elif lattice_system == 'orthorhombic':
            unit_cell_inv = np.column_stack([a_inv, b_inv, c_inv])
        elif lattice_system == 'rhombohedral':
            unit_cell_inv = np.column_stack([a_inv, alpha_inv])
        elif lattice_system == 'monoclinic':
            unit_cell_inv = np.column_stack([a_inv, b_inv, c_inv, beta_inv])
    elif partial_unit_cell == False or lattice_system == 'triclinic':
        unit_cell_inv = np.column_stack([a_inv, b_inv, c_inv, alpha_inv, beta_inv, gamma_inv])
    return unit_cell_inv


def get_xnn_from_reciprocal_unit_cell(reciprocal_unit_cell, partial_unit_cell=False, lattice_system=None):
    if partial_unit_cell and lattice_system != 'triclinic':
        if lattice_system in ['cubic', 'tetragonal', 'hexagonal', 'orthorhombic']:
            xnn = reciprocal_unit_cell**2
        elif lattice_system == 'rhombohedral':
            xnn = np.column_stack([
                reciprocal_unit_cell[:, 0]**2,
                2*reciprocal_unit_cell[:, 0]**2 * np.cos(reciprocal_unit_cell[:, 1]),
                ])
        elif lattice_system == 'monoclinic':
            xnn = np.column_stack([
                reciprocal_unit_cell[:, 0]**2,
                reciprocal_unit_cell[:, 1]**2,
                reciprocal_unit_cell[:, 2]**2,
                2*reciprocal_unit_cell[:, 0] * reciprocal_unit_cell[:, 2] * np.cos(reciprocal_unit_cell[:, 3]),
                ])
    elif partial_unit_cell == False or lattice_system == 'triclinic':
        xnn = np.column_stack([
            reciprocal_unit_cell[:, 0]**2,
            reciprocal_unit_cell[:, 1]**2,
            reciprocal_unit_cell[:, 2]**2,
            2*reciprocal_unit_cell[:, 1] * reciprocal_unit_cell[:, 2] * np.cos(reciprocal_unit_cell[:, 3]),
            2*reciprocal_unit_cell[:, 0] * reciprocal_unit_cell[:, 2] * np.cos(reciprocal_unit_cell[:, 4]),
            2*reciprocal_unit_cell[:, 0] * reciprocal_unit_cell[:, 1] * np.cos(reciprocal_unit_cell[:, 5]),
            ])
        xnn[reciprocal_unit_cell[:, 3] == np.pi/2, 3] = 0
        xnn[reciprocal_unit_cell[:, 4] == np.pi/2, 4] = 0
        xnn[reciprocal_unit_cell[:, 5] == np.pi/2, 5] = 0
    return xnn


def get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=False, lattice_system=None):
    if partial_unit_cell and lattice_system != 'triclinic':
        if lattice_system in ['cubic', 'tetragonal', 'hexagonal', 'orthorhombic']:
            reciprocal_unit_cell = np.sqrt(xnn)
        elif lattice_system == 'rhombohedral':
            reciprocal_unit_cell = np.column_stack([
                np.sqrt(xnn[:, 0]),
                np.arccos(xnn[:, 1] / (2 * xnn[:, 0])),
                ])
        elif lattice_system == 'monoclinic':
            ra = np.sqrt(xnn[:, 0])
            rb = np.sqrt(xnn[:, 1])
            rc = np.sqrt(xnn[:, 2])
            rbeta = np.arccos(xnn[:, 3] / (2 * ra * rc))    
            reciprocal_unit_cell = np.column_stack([ra, rb, rc, rbeta])
    elif partial_unit_cell == False or lattice_system == 'triclinic':
        ra = np.sqrt(xnn[:, 0])
        rb = np.sqrt(xnn[:, 1])
        rc = np.sqrt(xnn[:, 2])
        ralpha = np.arccos(xnn[:, 3] / (2 * rb * rc))
        rbeta = np.arccos(xnn[:, 4] / (2 * ra * rc))    
        rgamma = np.arccos(xnn[:, 5] / (2 * ra * rb))
        reciprocal_unit_cell = np.column_stack([ra, rb, rc, ralpha, rbeta, rgamma])
    return reciprocal_unit_cell


def get_unit_cell_from_xnn(xnn, partial_unit_cell=False, lattice_system=None):
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell, lattice_system)
    return reciprocal_uc_conversion(reciprocal_unit_cell, partial_unit_cell, lattice_system)


def get_xnn_from_unit_cell(unit_cell, partial_unit_cell=False, lattice_system=None):
    reciprocal_unit_cell = reciprocal_uc_conversion(unit_cell, partial_unit_cell, lattice_system)
    return get_xnn_from_reciprocal_unit_cell(reciprocal_unit_cell, partial_unit_cell, lattice_system)


def get_unit_cell_volume(unit_cell, partial_unit_cell=False, lattice_system=None):
    if partial_unit_cell and lattice_system != 'triclinic':
        if lattice_system == 'cubic':
            volume = unit_cell[:, 0]**3
        elif lattice_system == 'tetragonal':
            volume = unit_cell[:, 0]**2 * unit_cell[:, 1]
        elif lattice_system == 'hexagonal':
            volume = unit_cell[:, 0]**2 * unit_cell[:, 1] * np.sin(np.pi/3)
        elif lattice_system == 'rhombohedral':
            volume = unit_cell[:, 0]**3 * np.sqrt(1 - 3*np.cos(unit_cell[:, 1])**2 + 2*np.cos(unit_cell[:, 1])**3)
        elif lattice_system == 'orthorhombic':
            volume = unit_cell[:, 0] * unit_cell[:, 1] * unit_cell[:, 2]
        elif lattice_system == 'monoclinic':
            abc = unit_cell[:, 0] * unit_cell[:, 1] * unit_cell[:, 2]
            volume = abc * np.sin(unit_cell[:, 3])

    elif partial_unit_cell == False or lattice_system == 'triclinic':
        a = unit_cell[:, 0]
        b = unit_cell[:, 1]
        c = unit_cell[:, 2]
        calpha = np.cos(unit_cell[:, 3])
        cbeta = np.cos(unit_cell[:, 4])
        cgamma = np.cos(unit_cell[:, 5])
        arg = 1 - calpha**2 - cbeta**2 - cgamma**2 + 2*calpha*cbeta*cgamma
        volume = (a*b*c) * np.sqrt(arg)
    return volume


def fix_unphysical(xnn=None, unit_cell=None, rng=None, lattice_system=None, minimum_unit_cell=2, maximum_unit_cell=500):
    if rng is None:
        rng = np.random.default_rng()
    if not xnn is None:
        if lattice_system == 'triclinic':
            return fix_unphysical_triclinic(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'monoclinic':
            return fix_unphysical_monoclinic(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'rhombohedral':
            return fix_unphysical_rhombohedral(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system in ['cubic', 'orthorhombic', 'tetragonal', 'hexagonal']:
            return fix_unphysical_box(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
    elif not unit_cell is None:
        if lattice_system == 'triclinic':
            return fix_unphysical_triclinic(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'monoclinic':
            return fix_unphysical_monoclinic(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'rhombohedral':
            return fix_unphysical_rhombohedral(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system in ['cubic', 'orthorhombic', 'tetragonal', 'hexagonal']:
            return fix_unphysical_box(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )


def fix_unphysical_triclinic(xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500):
    """
    The purpose of this function is to ensure that RANDOMLY GENERATED triclinic unit cells are physically
    possible. This should not be used on known unit cells
    """
    if rng is None:
        rng = np.random.default_rng()

    if not xnn is None:
        def get_limits(cos_angle_0, cos_angle_1):
            pa = -1
            pb = 2*cos_angle_0*cos_angle_1
            pc = 1 - cos_angle_0**2 - cos_angle_1**2
            roots = [(-pb - np.sqrt(pb**2 - 4*pa*pc)) / (2*pa), (-pb + np.sqrt(pb**2 - 4*pa*pc)) / (2*pa)]
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

        bad_ralpha = np.logical_or(cos_ralpha < 0, cos_ralpha > 1)
        bad_rbeta = np.logical_or(cos_rbeta < 0, cos_rbeta > 1)
        bad_rgamma = np.logical_or(cos_rgamma < 0, cos_rgamma > 1)
        if np.sum(bad_ralpha) > 0:
            xnn[bad_ralpha, 3] = (2 * rb * rc)[bad_ralpha] * rng.uniform(low=0, high=1, size=np.sum(bad_ralpha))
        if np.sum(bad_rbeta) > 0:
            xnn[bad_rbeta, 4] = (2 * ra * rc)[bad_rbeta] * rng.uniform(low=0, high=1, size=np.sum(bad_rbeta))
        if np.sum(bad_rgamma) > 0:
            xnn[bad_rgamma, 5] = (2 * ra * rb)[bad_rgamma] * rng.uniform(low=0, high=1, size=np.sum(bad_rgamma))
        
        # Unit cell volume:
        # abc * sqrt[1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma)]
        #   Enforce the argument in the sqrt to be positive
        cos_ralpha = xnn[:, 3] / (2 * rb * rc)
        cos_rbeta = xnn[:, 4] / (2 * ra * rc)    
        cos_rgamma = xnn[:, 5] / (2 * ra * rb)
        volume_arg = 1 - cos_ralpha**2 - cos_rbeta**2 - cos_rgamma**2 + 2*cos_ralpha*cos_rbeta*cos_rgamma
        for index in np.argwhere(volume_arg < 0):
            status = True
            while status:
                lower_root, upper_root = get_limits(cos_rbeta[index], cos_rgamma[index])
                cos_ralpha0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(cos_ralpha[index], cos_rgamma[index])
                cos_rbeta0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(cos_ralpha[index], cos_rbeta[index])
                cos_rgamma0 = rng.uniform(low=lower_root, high=upper_root)

                volume_arg = 1 - cos_ralpha0**2 - cos_rbeta0**2 - cos_rgamma0**2 + 2*cos_ralpha0*cos_rbeta0*cos_rgamma0
                if volume_arg > 0:
                    status = False
                    xnn[index, 3] = cos_ralpha0 * (2 * rb[index] * rc[index])
                    xnn[index, 4] = cos_rbeta0 * (2 * ra[index] * rc[index])
                    xnn[index, 5] = cos_rgamma0 * (2 * ra[index] * rb[index])
        # If the reciprocal unit cell conversion fails due to a singular matrix, then the returned
        # unit cell is np.nan. This is extremely rare at this point
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=False)
        unit_cell = reciprocal_uc_conversion(reciprocal_unit_cell, partial_unit_cell=False)
        failed = np.any(np.isnan(unit_cell), axis=1)
        if np.count_nonzero(failed) > 0:
            xnn[failed, :] = np.array([1/5**2, 1/5**2, 1/5**2, 0, 0, 0])
        return xnn

    elif not unit_cell is None:
        def get_limits(cos_angle_0, cos_angle_1):
            pa = -1
            pb = 2*cos_angle_0*cos_angle_1
            pc = 1 - cos_angle_0**2 - cos_angle_1**2
            roots = [(-pb - np.sqrt(pb**2 - 4*pa*pc)) / (2*pa), (-pb + np.sqrt(pb**2 - 4*pa*pc)) / (2*pa)]
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
        maximum_angle = np.pi
        minimum_angle = np.pi/2

        too_small_lengths = unit_cell[:, :3] < minimum_unit_cell
        too_large_lengths = unit_cell[:, :3] > maximum_unit_cell
        if np.sum(too_small_lengths) > 0:
            indices = np.argwhere(too_small_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=minimum_unit_cell,
                high=1.05*minimum_unit_cell,
                size=np.sum(too_small_lengths)
                )
        if np.sum(too_large_lengths) > 0:
            indices = np.argwhere(too_large_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=0.95*maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths)
                )
        
        too_small_angles = unit_cell[:, 3:] < minimum_angle
        too_large_angles = unit_cell[:, 3:] > maximum_angle

        if np.sum(too_large_angles) > 0:
            indices = np.argwhere(too_large_angles)
            unit_cell[indices[:, 0], 3 + indices[:, 1]] = rng.uniform(
                low=0.95*maximum_angle,
                high=maximum_angle,
                size=np.sum(too_large_angles)
                )
        if np.sum(too_small_angles) > 0:
            indices = np.argwhere(too_small_angles)
            unit_cell[indices[:, 0], 3 + indices[:, 1]] = rng.uniform(
                low=minimum_angle,
                high=1.05*minimum_angle,
                size=np.sum(too_small_angles)
                )

        # Unit cell volume:
        # abc * sqrt[1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma)]
        #   * Enforce the argument in the sqrt to be positive
        cos_alpha = np.cos(unit_cell[:, 3])
        cos_beta = np.cos(unit_cell[:, 4])
        cos_gamma = np.cos(unit_cell[:, 5])
        volume_arg = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2*cos_alpha*cos_beta*cos_gamma
        for index in np.argwhere(volume_arg < 0):
            status = True
            while status:
                lower_root, upper_root = get_limits(cos_beta[index], cos_gamma[index])
                cos_alpha0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(cos_alpha[index], cos_gamma[index])
                cos_beta0 = rng.uniform(low=lower_root, high=upper_root)

                lower_root, upper_root = get_limits(cos_alpha[index], cos_beta[index])
                cos_gamma0 = rng.uniform(low=lower_root, high=upper_root)

                volume_arg = 1 - cos_alpha0**2 - cos_beta0**2 - cos_gamma0**2 + 2*cos_alpha0*cos_beta0*cos_gamma0
                if volume_arg > 0:
                    status = False
                    unit_cell[index, 3] = np.arccos(cos_alpha0)
                    unit_cell[index, 4] = np.arccos(cos_beta0)
                    unit_cell[index, 5] = np.arccos(cos_gamma0)

        # If the reciprocal unit cell conversion fails due to a singular matrix, then the returned
        # unit cell is np.nan. This is extremely rare at this point
        reciprocal_unit_cell = reciprocal_uc_conversion(unit_cell, partial_unit_cell=False)
        failed = np.any(np.isnan(reciprocal_unit_cell), axis=1)
        if np.count_nonzero(failed) > 0:
            unit_cell[failed, :] = np.array([5, 5, 5, np.pi/2, np.pi/2, np.pi/2])
        return unit_cell


def fix_unphysical_rhombohedral(xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500):
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
                size=np.sum(zero_indices)
                )

        # Direct space & Reciprocal unit cell angle must be
        # between 0 and 120 degrees (2/3 pi radians)
        cos_ralpha = xnn[:, 1] / (2 * xnn[:, 0])
        negative_angle = cos_ralpha >= 1
        large_angle = cos_ralpha < -0.5
        if np.sum(negative_angle) > 0:
            cos_ralpha[negative_angle] = rng.uniform(low=0.95, high=1, size=np.sum(negative_angle))
            xnn[negative_angle, 1] = cos_ralpha[negative_angle] * 2 * xnn[negative_angle, 0]
        if np.sum(large_angle) > 0:
            cos_ralpha[large_angle] = rng.uniform(low=-0.5, high=-0.49, size=np.sum(large_angle))
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
                high=1.05*minimum_unit_cell,
                size=np.sum(too_small_lengths)
                )
        if np.sum(too_large_lengths) > 0:
            unit_cell[too_large_lengths, 0] = rng.uniform(
                low=0.95*maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths)
                )

        bad_angle = np.logical_or(unit_cell[:, 1] <= 0, unit_cell[:, 1] > 2*np.pi/3)
        if np.sum(bad_angle) > 0:
            unit_cell[bad_angle, 1] = rng.uniform(low=0, high=2*np.pi/3, size=np.sum(bad_angle))
        return unit_cell


def fix_unphysical_box(xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500):
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
                high=1.05*minimum_unit_cell,
                size=np.sum(too_small_lengths)
                )
        if np.sum(too_large_lengths) > 0:
            indices = np.argwhere(too_large_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=0.95*maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths)
                )
        return unit_cell


def fix_unphysical_monoclinic(xnn=None, unit_cell=None, rng=None, minimum_unit_cell=2, maximum_unit_cell=500):
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
        cos_rbeta = xnn[:, 3] / (2 * np.sqrt(xnn[:, 0]*xnn[:, 2]))
        unphysical_angle = cos_rbeta >= 1
        if np.sum(unphysical_angle) > 0:
            cos_rbeta[unphysical_angle] = rng.uniform(low=0, high=1, size=np.sum(unphysical_angle))
            xnn[unphysical_angle, 3] = cos_rbeta[unphysical_angle] * (2 * np.sqrt(xnn[unphysical_angle, 0]*xnn[unphysical_angle, 2]))
        return xnn
    elif not unit_cell is None:
        too_small_lengths = unit_cell[:, :3] < minimum_unit_cell
        too_large_lengths = unit_cell[:, :3] > maximum_unit_cell
        if np.sum(too_small_lengths) > 0:
            indices = np.argwhere(too_small_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=minimum_unit_cell,
                high=1.05*minimum_unit_cell,
                size=np.sum(too_small_lengths)
                )
        if np.sum(too_large_lengths) > 0:
            indices = np.argwhere(too_large_lengths)
            unit_cell[indices[:, 0], indices[:, 1]] = rng.uniform(
                low=0.95*maximum_unit_cell,
                high=maximum_unit_cell,
                size=np.sum(too_large_lengths)
                )

        too_small_angles = unit_cell[:, 3] < np.pi / 2
        too_large_angles = unit_cell[:, 3] > np.pi
        if np.sum(too_small_angles) > 0:
            unit_cell[too_small_angles, 3] = np.pi - unit_cell[too_small_angles, 3]
        if np.sum(too_large_angles) > 0:
            unit_cell[too_large_angles, 3] = rng.uniform(
                low=0.95*np.pi,
                high=np.pi,
                size=np.sum(too_large_angles)
                )
        return unit_cell


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
    expected_discrepancy[good_indices] = last_smaller_ref_peak[good_indices] / (2*N[good_indices])
    M20 = expected_discrepancy / discrepancy
    return M20


def get_M20_likelihood_from_xnn(q2_obs, xnn, hkl, lattice_system, bravais_lattice):
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True, lattice_system=lattice_system)
    reciprocal_volume = get_unit_cell_volume(reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system)
    log_likelihood, probability, M = get_M20_likelihood(q2_obs, q2_calc, bravais_lattice, reciprocal_volume)
    return log_likelihood, probability, M


def get_M20_likelihood(q2_obs, q2_calc, bravais_lattice, reciprocal_volume):
    # This was inspired by Taupin 1988
    # Probability that a peak is correctly assigned:
    # arg = Expected number of peaks within error from random unit cell
    # P = 1 / (1 + arg)
    mu, nu = get_multiplicity_taupin88(bravais_lattice)
    observed_difference2 = (np.sqrt(q2_obs[np.newaxis]) - np.sqrt(q2_calc))**2
    # There is an upstream error where reciprocal volumes can be very small.
    # Adding 1e-100 here prevents division by zero errors
    arg = 8*np.pi*q2_obs * np.sqrt(observed_difference2) / (reciprocal_volume[:, np.newaxis] * mu + 1e-100)
    probability = 1 / (1 + arg)
    # The 1e-100 factor prevents np.log(~0) = -infinity
    M = -1/np.log(2) * np.sum(np.log(1 - np.exp(-arg) + 1e-100), axis=1)
    return -np.sum(np.log(probability + 1e-100), axis=1), probability, M


def get_multiplicity_taupin88(bravais_lattice):
    # The commented out returns come from Taupin 1988
    # The others are from empirically plotting the
    # non systematic absences
    if bravais_lattice == 'cF':
        return 4*32, 1
    elif bravais_lattice == 'cI':
        return 2*32, 1
    elif bravais_lattice == 'cP':
        return 1*32, 1
    elif bravais_lattice == 'hP':
        #return 1*24, 2
        return 1*14, 2
    elif bravais_lattice == 'hR':
        #return 1*24, 2
        return 1*8, 2
    elif bravais_lattice == 'tI':
        #return 2*16, 2
        return 2*13, 2
    elif bravais_lattice == 'tP':
        #return 1*16, 2
        return 1*13, 2
    elif bravais_lattice in ['oC', 'oI']:
        #return 2*8, 3
        return 2*7, 3
    elif bravais_lattice == 'oF':
        #return 4*8, 3
        return 4*7, 3
    elif bravais_lattice == 'oP':
        #return 1*8, 3
        return 1*7, 3
    elif bravais_lattice == 'mC':
        #return 2*4, 4
        return 2*3.2, 4
    elif bravais_lattice == 'mP':
        #return 1*4, 4
        return 1*3.5, 4
    elif bravais_lattice == 'aP':
        #return 1*2, 6
        return 1*1.8, 6


def get_M20_sym_reversed(q2_obs, xnn, hkl, hkl_ref, lattice_system):
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    q2_ref_calc = np.sum(hkl2_ref * xnn[:, np.newaxis, :], axis=2)
    multiplicity = get_multiplicity(hkl.reshape((hkl.shape[0]*hkl.shape[1], hkl.shape[2])), lattice_system).reshape(hkl.shape[:2])
    multiplicity_ref = get_multiplicity(hkl_ref, 'monoclinic')

    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    smaller_ref_peaks = q2_ref_calc < q2_calc[:, -1][:, np.newaxis]
    last_smaller_ref_peak = np.zeros(q2_calc.shape[0])
    expected_discrepancy_reversed = (q2_obs[-1] - q2_obs[0]) / (2*20)
    discrepancy_reversed = np.zeros(q2_calc.shape[0])
    for i in range(q2_calc.shape[0]):
        q2_ref_smaller = q2_ref_calc[i, smaller_ref_peaks[i]]
        multiplicities_ref_smaller = multiplicity_ref[smaller_ref_peaks[i]]
        sort_indices = np.argsort(q2_ref_smaller)
        q2_ref_smaller = q2_ref_smaller[sort_indices]
        multiplicities_ref_smaller = multiplicities_ref_smaller[sort_indices]
        last_smaller_ref_peak[i] = q2_ref_smaller[-1]
    
        N_calc = np.sum(1/multiplicities_ref_smaller)
        differences = np.min(np.abs(q2_ref_smaller[np.newaxis] - q2_obs[:, np.newaxis]), axis=0)
        discrepancy_reversed[i] = np.sum(differences/multiplicities_ref_smaller) / N_calc
    
    N = np.sum(smaller_ref_peaks, axis=1)
    expected_discrepancy = last_smaller_ref_peak / (2*N)
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
    difference = np.abs(triplets_obs[:, 2][np.newaxis, :, np.newaxis] - q2_diff_calc_sym)
    q2_diff_calc = np.take_along_axis(
        q2_diff_calc_sym,
        np.argmin(difference, axis=2)[:, :, np.newaxis],
        axis=2
        )[:, :, 0]
    return q2_diff_calc


def get_M_triplet_from_xnn(triplets_obs, hkl, xnn, lattice_system, bravais_lattice):
    q2_diff_calc = get_q2_calc_triplets(triplets_obs, hkl, xnn, lattice_system)
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True, lattice_system=lattice_system)
    reciprocal_volume = get_unit_cell_volume(reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system)
    _, _, M20_triplet = get_M20_likelihood(triplets_obs[:, 2], q2_diff_calc, bravais_lattice, reciprocal_volume)    
    return M20_triplet


def get_M_triplet_old(q2_obs, triplets_obs, hkl, xnn, lattice_system, bravais_lattice):
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
        )
    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
        )

    # This could be simplified
    _, _, M_likelihood_primary = get_M20_likelihood_from_xnn(
        q2_obs, xnn, hkl, lattice_system, bravais_lattice
        )

    # q2_diff_calc is the magnitude of the calculated difference between q0 and q1
    # It is the calculated value of triplet_obs[:, 2]
    q2_diff_calc = get_q2_calc_triplets(triplets_obs, hkl, xnn, lattice_system)    
    _, _, M_likelihood_triplet = get_M20_likelihood(
        triplets_obs[:, 2], q2_diff_calc, bravais_lattice, reciprocal_volume
        )
    return np.column_stack((M_likelihood_primary, M_likelihood_triplet))


def get_M_triplet(q2_obs, q2_calc, triplets_obs, hkl, xnn, lattice_system, bravais_lattice):
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
    mi_sym = np.stack([
        np.eye(3),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
            ]),
        np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
            ]),
        np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            ])
        ],
        axis=0
        )
    if lattice_system in ['hexagonal', 'tetragonal', 'cubic', 'rhombohedral']:
        # abc
        # bac
        mi_permutations = [
            np.eye(3),
            np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ])
            ]
    if lattice_system in ['cubic', 'rhombohedral']:
        # acb
        # bca
        # cba
        # cab
        mi_permutations += [
            np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                ]),
            np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                ]),
            np.array([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                ]),
            np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                ]),
            ]
    if lattice_system in ['hexagonal', 'tetragonal', 'cubic', 'rhombohedral']:
        mi_permutations = np.stack(mi_permutations, axis=0)
        mi_sym_extra = np.matmul(mi_sym[:, np.newaxis, :, :], mi_permutations[np.newaxis, :, :, :])
        mi_sym = mi_sym_extra.reshape((mi_sym_extra.shape[0] * mi_sym_extra.shape[1], 3, 3))
    return mi_sym


def get_triplet_hkl_ref(hkl_ref, lattice_system):
    def pairing(k1, k2):
        k1 = k1 + 1000
        k2 = k2 + 1000
        return (k1 + k2)*(k1 + k2 + 1)/2 + k2
    # mi_sym:           n_sym x 3 x 3
    # hkl_ref:          n_ref x 3
    # hkl_ref_sym:      n_ref x n_sym x 3
    # triplet_hkl_diff: n_ref, n_ref, n_sym, 3
    n_ref = hkl_ref.shape[0]
    mi_sym = get_hkl_triplet_symmetry(lattice_system)
    hkl_ref_sym = np.matmul(mi_sym[np.newaxis], hkl_ref[:, np.newaxis, :, np.newaxis])[:, :, :, 0]
    triplet_hkl_diff = hkl_ref_sym[:, np.newaxis, :, :] - hkl_ref[np.newaxis, :, np.newaxis, :]
    triplet_hkl2_diff = get_hkl_matrix(triplet_hkl_diff, lattice_system)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    if lattice_system == 'cubic':
        hash_triplet_diff = triplet_hkl2_diff[:, :, :, 0]
        hash_ref = hkl2_ref[:, 0]
    elif lattice_system in ['hexagonal', 'tetragonal', 'rhombohedral']:
        hash_triplet_diff = pairing(
            triplet_hkl2_diff[:, :, :, 0], triplet_hkl2_diff[:, :, :, 1]
            )
        hash_ref = pairing(hkl2_ref[:, 0], hkl2_ref[:, 1])
    elif lattice_system == 'orthorhombic':
        hash_triplet_diff = pairing(
            triplet_hkl2_diff[:, :, :, 0],
            pairing(
                triplet_hkl2_diff[:, :, :, 1], triplet_hkl2_diff[:, :, :, 2]
                )
            )
        hash_ref = pairing(hkl2_ref[:, 0], pairing(hkl2_ref[:, 1], hkl2_ref[:, 2]))
    elif lattice_system == 'monoclinic':
        hash_triplet_diff = pairing(
            triplet_hkl2_diff[:, :, :, 0],
            pairing(
                triplet_hkl2_diff[:, :, :, 1],
                pairing(
                    triplet_hkl2_diff[:, :, :, 2], triplet_hkl2_diff[:, :, :, 3]
                    )
                )
            )
        hash_ref = pairing(
            hkl2_ref[:, 0], 
            pairing(
                hkl2_ref[:, 1],
                pairing(
                    hkl2_ref[:, 2], hkl2_ref[:, 3]
                    )
                )
            )
    elif lattice_system == 'triclinic':
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
                            )
                        )
                    )
                )
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
                            hkl2_ref[:, 4], hkl2_ref[:, 5],
                            )
                        )
                    )
                )
            )

    triplet_hkl_ref = [ [ None for _ in range(n_ref)] for _ in range(n_ref)]
    same = hash_triplet_diff[:, :, :, np.newaxis] == hash_ref[np.newaxis, np.newaxis, np.newaxis, :]
    for i in range(n_ref - 1):
        for j in range(i + 1, n_ref):
            indices = []
            for k in range(mi_sym.shape[0]):
                indices_here = np.argwhere(same[i, j, k])
                if indices_here.size > 0:
                    indices.append(indices_here[0][0])
            triplet_hkl_ref[i][j] = list(set(indices))
    return triplet_hkl_ref


def get_hkl_matrix(hkl, lattice_system):
    last_axis = len(hkl.shape) - 1
    # hkl shape:
    # last_axis = 1: n_peaks, 3
    # last_axis = 2: n_entries, n_peaks, 3
    if lattice_system == 'triclinic':
        hkl_matrix = np.concatenate((
            hkl[..., :3]**2,
            (hkl[..., 1] * hkl[..., 2])[..., np.newaxis],
            (hkl[..., 0] * hkl[..., 2])[..., np.newaxis],
            (hkl[..., 0] * hkl[..., 1])[..., np.newaxis],
            ),
            axis=last_axis
            )
    elif lattice_system == 'monoclinic':
        hkl_matrix = np.concatenate((
            hkl[..., :3]**2,
            (hkl[..., 0] * hkl[..., 2])[..., np.newaxis],
            ),
            axis=last_axis
            )
    elif lattice_system == 'orthorhombic':
        hkl_matrix = hkl**2
    elif lattice_system == 'tetragonal':
        hkl_matrix = np.stack((
            np.sum(hkl[..., :2]**2, axis=last_axis),
            hkl[..., 2]**2,
            ),
            axis=last_axis
            )
    elif lattice_system == 'hexagonal':
        hkl_matrix = np.stack((
            hkl[..., 0]**2 + hkl[..., 0]*hkl[..., 1] + hkl[..., 1]**2,
            hkl[..., 2]**2,
            ),
            axis=last_axis
            )
    elif lattice_system == 'rhombohedral':
        hkl_matrix = np.stack((
            np.sum(hkl**2, axis=last_axis),
            hkl[..., 0]*hkl[..., 1] + hkl[..., 0]*hkl[..., 2] + hkl[..., 1]*hkl[..., 2],
            ),
            axis=last_axis
            )
    elif lattice_system == 'cubic':
        hkl_matrix = np.sum(hkl**2, axis=last_axis)[..., np.newaxis]
    return hkl_matrix


def get_spacegroup_hkl_ref(hkl_ref, bravais_lattice):
    # https://www.ba.ic.cnr.it/softwareic/expo/extinction_symbols/
    if bravais_lattice == 'cF':
        spacegroups =        ['F 2 3',   'F d -3',  'F 41 3 2', 'F -4 3 c', 'F d -3 c']
        extinction_groups =  ['F - - -', 'F d - -', 'F 41 - -', 'F - - c',  'F d - c']
    elif bravais_lattice == 'cI':
        spacegroups =       ['I 2 3',   'I a -3',  'I 41 3 2', 'I -4 3 d', 'I a -3 d']
        extinction_groups = ['I - - -', 'I a - -', 'I 41 - -', 'I - - d',  'I a - d']
    elif bravais_lattice == 'cP':
        spacegroups = [
            'P 2 3',   'P 21 3',    'P n -3',  'P a -3',  'P 43 3 2', 'P -43 n', 'P n -3 n'
            ]
        extinction_groups = [
            'P - - -', 'P 21 - -',  'P n - -', 'P a - -', 'P 41 - -', 'P - - n', 'P n - n'
            ]
    elif bravais_lattice == 'hR':
        spacegroups =       ['R 3',     'R 3 c']
        extinction_groups = ['R - - -', 'R - - c']
    elif bravais_lattice == 'hP':
        spacegroups =       [
            'P 3',     'P 31',     'P 3 c 1', 'P 3 1 c', 'P 61',     'P 62',     'P 63',     'P 6 c c'
            ]
        extinction_groups = [
            'P - - -', 'P 31 - -', 'P - c -', 'P - - c', 'P 61 - -', 'P 62 - -', 'P 63 - -', 'P - c c'
            ]
    elif bravais_lattice == 'tI':
        spacegroups = [
            'I 4',     'I 41',    'I 41/a',     'I 4 c m', 'I 41 m d', 'I 41 c d', 'I 41/a m d', 'I 41/a c d'
            ]
        extinction_groups = [
            'I - - -', 'I41 - -', 'I 41/a - -', 'I - c -', 'I - - d',  'I - c d',  'I a - d',    'I a c d'
            ]
    elif bravais_lattice == 'tP':
        spacegroups = [
            'P 4',     'P 41',     'P 42',     'P 4/n',   'P 42/n',     'P 4 21 2', 'P 41 21 2', 'P 42 21 2',
            'P 4 b m', 'P 42 c m', 'P 42 n m', 'P 4 c c', 'P 4 n c', 'P 42 m c', 'P 42 b c', 'P -4 21 c',
            'P 4/n b m', 'P 4/n n c', 'P 4/n c c', 'P 42/n b c', 'P 42/n n m', 'P 42/n m c', 'P 42/n c m'
            ]
        extinction_groups = [
            'P - - -', 'P 41 - -', 'P 42 - -', 'P n - -', 'P 42/n - -', 'P - 21 2', 'P 41 21 -', 'P 42 21 2',
            'P - b -', 'P - c -',  'P - n -',  'P - c c', 'P - n c', 'P - - c',  'P - b c',  'P - c1 c',
            'P n b -',   'P n n c',   'P n c c',   'P n b c',    'P n n -',    'P n - c',    'P n c -'  
            ]
    elif bravais_lattice == 'oC':
        # Do I need to add C m 2 b for example?
        spacegroups = [
            'C 2 2 2', 'C 2 2 21', 'C c c 2', 'C c 2 m', 'C 2 c m', 'C c 2 a', 'C 2 c b', 'C c c a', 'C m 2 a',
            ]
        extinction_groups = [
            'C - - -', 'C - - 21', 'C c c -', 'C c - -', 'C - c -', 'C c - a', 'C - c b', 'C c c a', 'C - - a',
            ]
    elif bravais_lattice == 'oF':
        spacegroups =       ['F 2 2 2', 'F d d d', 'F 2 d d', 'F d 2 d', 'F d d 2']
        extinction_groups = ['F - - -', 'F d d d', 'F - d d', 'F d - d', 'F d d -']
    elif bravais_lattice == 'oI':
        spacegroups = [
            'I m m m', 'I b c a', 'I b a 2', 'I 2 c b', 'I c 2 a', 'I b m 2', 'I m a 2', 'I m 2 a',
            ]
        extinction_groups = [
            'I - - -', 'I b c a', 'I b a -', 'I - c b', 'I c - a', 'I b m -', 'I - a -', 'I - - a',
            ]
    elif bravais_lattice == 'oP':
        spacegroups = [
            'P 2 2 2', 'P 21 2 2', 'P 2 21 2', 'P 2 2 21',
            'P 21 m a', 'P m 21 b', 'P m c 21',
            'P 21 a m', 'P b 21 m', 'P c m 21', 
            'P 2 a a', 'P b 2 b', 'P c c 2',
            'P 2 21 21', 'P 21 2 21', 'P 21 21 2',
            'P b c 21', 'P c a 21', 'P b 21 a', 'P c 21 b', 'P 21 c a', 'P 21 a b',
            'P 2 c b', 'P c 2 a', 'P b a 2',
            'P n c 2', 'P c n 2', 'P b 2 n', 'P n 2 b', 'P 2 n a', 'P 2 a n',
            'P n m 21', 'P m 21 n', 'P 21 n m', 
            'P n c b', 'P c n a', 'P b a n',
            'P c c b', 'P c c a', 'P b a a', 'P b c b', 'P c a a', 'P b a b',
            'P 21 21 21', 'P n n n', 'P b c a', 'P c a b',
            'P n a 21', 'P b n 21', 'P c 21 n', 'P n 21 a', 'P 21 n b', 'P 21 c n',
            'P 2 n n', 'P n 2 n', 'P n n 2', 'P n n a', 'P n n b', 'P n c n',
            'P c c n', 'P b n b', 'P n a a',
            'P b c n', 'P c a n', 'P b n a', 'P c n b', 'P n c a', 'P n a b'
            ]
        extinction_groups = [
            'P - - -', 'P 21 - -', 'P - 21 -', 'P - - 21',
            'P - - a', 'P - - b', 'P - c -',
            'P - a -', 'P b - -', 'P c - -', 
            'P - a a', 'P b - b', 'P c c -',
            'P - 21 21', 'P 21 - 21', 'P 21 21 -',
            'P b c -', 'P c a -', 'P b - a', 'P c - b', 'P - c a', 'P - a b',
            'P - c b', 'P c - a', 'P b a -',
            'P n c -', 'P c n -', 'P b - n', 'P n - b', 'P - n a', 'P - a n',
            'P n - -', 'P - - n', 'P - n -', 
            'P n c b', 'P c n a', 'P b a n',
            'P c c b', 'P c c a', 'P b a a', 'P b c b', 'P c a a', 'P b a b',
            'P 21 21 21', 'P n n n', 'P b c a', 'P c a b',
            'P n a -', 'P b n -', 'P c - n', 'P n - a', 'P - n b', 'P - c n',
            'P - n n', 'P n - n', 'P n n -', 'P n n a', 'P n n b', 'P n c n',
            'P c c n', 'P b n b', 'P n a a',
            'P b c n', 'P c a n', 'P b n a', 'P c n b', 'P n c a', 'P n a b'
            ]
    elif bravais_lattice == 'mC':
        spacegroups =       ['I 1 2 1', 'I 1 a 1']
        extinction_groups = ['I 1 - 1', 'I 1 a 1']
    elif bravais_lattice == 'mP':
        spacegroups = [
            'P 1 2 1', 'P 1 21 1', 'P 1 c 1', 'P 1 a 1', 'P 1 n 1', 'P 1 21/c 1', 'P 1 21/a 1', 'P 1 21/n 1'
            ]
        extinction_groups = [
            'P 1 - 1', 'P 1 21 1', 'P 1 c 1', 'P 1 a 1', 'P 1 n 1', 'P 1 21/c 1', 'P 1 21/a 1', 'P 1 21/n 1'
            ]
    elif bravais_lattice == 'aP':
        spacegroups =       ['P 1']
        extinction_groups = ['P -']
    """
    hkl_ref_sg = dict.fromkeys(spacegroups)
    for spacegroup in spacegroups:
        if bravais_lattice == 'hR':
            # gemmi gives the systematic absences for rhombohedral in the hexagonal setting.
            # The ':R' component tells gemmi to use the rhombohedral setting
            ops = gemmi.SpaceGroup(f'{spacegroup}:R').operations()
        else:
            ops = gemmi.SpaceGroup(spacegroup).operations()
        systematically_absent = ops.systematic_absences(hkl_ref)
        hkl_ref_sg[spacegroup] = hkl_ref[np.invert(systematically_absent)]
    """
    keys = [f'{i} e.g. {j}' for i, j in zip(extinction_groups, spacegroups)]
    hkl_ref_sg = dict.fromkeys(keys)
    for index, key in enumerate(keys):
        if bravais_lattice == 'hR':
            # gemmi gives the systematic absences for rhombohedral in the hexagonal setting.
            # The ':R' component tells gemmi to use the rhombohedral setting
            ops = gemmi.SpaceGroup(f'{spacegroups[index]}:R').operations()
        else:
            ops = gemmi.SpaceGroup(spacegroups[index]).operations()
        systematically_absent = ops.systematic_absences(hkl_ref)
        hkl_ref_sg[key] = hkl_ref[np.invert(systematically_absent)]
    return hkl_ref_sg


def map_spacegroup_to_extinction_group(spacegroup_symbol_hm):
    # These extinction groups are based on those used by EXPO
    #   https://www.ba.ic.cnr.it/softwareic/expo/extinction_symbols/
    # This function was mostly produced by ChatGPT.
    table_data = [
        {"Code": 15, "Extinction Group": "P 1 – 1", "Space Groups": ["P121", "P1m1", "P12/m1"]},
        {"Code": 16, "Extinction Group": "P 1 21 1", "Space Groups": ["P1211", "P121/m1"]},
        {"Code": 21, "Extinction Group": "P 1 n 1", "Space Groups": ["P1n1", "P12/n1"]},
        {"Code": 22, "Extinction Group": "P 1 21/n 1", "Space Groups": ["P121/n1"]},
        {"Code": 27, "Extinction Group": "I 1 - 1", "Space Groups": ["I121", "I1m1", "I12/m1"]},
        {"Code": 28, "Extinction Group": "I 1 a 1", "Space Groups": ["I1a1", "I12/a1"]},
        {"Code": 43, "Extinction Group": "P – – –", "Space Groups": ["P222", "Pmm2", "Pmmm", "Pm2m", "P2mm"]},
        {"Code": 44, "Extinction Group": "P – – 21", "Space Groups": ["P2221"]},
        {"Code": 45, "Extinction Group": "P – 21 –", "Space Groups": ["P2212"]},
        {"Code": 46, "Extinction Group": "P – 21 21", "Space Groups": ["P22121"]},
        {"Code": 47, "Extinction Group": "P 21 – –", "Space Groups": ["P2122"]},
        {"Code": 48, "Extinction Group": "P 21 – 21", "Space Groups": ["P21221"]},
        {"Code": 49, "Extinction Group": "P 21 21 –", "Space Groups": ["P21212"]},
        {"Code": 50, "Extinction Group": "P 21 21 21", "Space Groups": ["P212121"]},
        {"Code": 51, "Extinction Group": "P – – a", "Space Groups": ["Pm2a", "P21ma", "Pmma"]},
        {"Code": 52, "Extinction Group": "P – – b", "Space Groups": ["Pm21b", "P2mb", "Pmmb"]},
        {"Code": 53, "Extinction Group": "P – – n", "Space Groups": ["Pm21n", "P21mn", "Pmmn"]},
        {"Code": 54, "Extinction Group": "P – a –", "Space Groups": ["Pma2", "Pmam", "P21am"]},
        {"Code": 55, "Extinction Group": "P – a a", "Space Groups": ["P2aa", "Pmaa"]},
        {"Code": 56, "Extinction Group": "P – a b", "Space Groups": ["P21ab", "Pmab"]},
        {"Code": 57, "Extinction Group": "P – a n", "Space Groups": ["P2an", "Pman"]},
        {"Code": 58, "Extinction Group": "P – c –", "Space Groups": ["Pmc21", "P2cm", "Pmcm"]},
        {"Code": 59, "Extinction Group": "P – c a", "Space Groups": ["P21ca", "Pmca"]},
        {"Code": 60, "Extinction Group": "P – c b", "Space Groups": ["P2cb", "Pmcb"]},
        {"Code": 61, "Extinction Group": "P – c n", "Space Groups": ["P21cn", "Pmcn"]},
        {"Code": 62, "Extinction Group": "P – n –", "Space Groups": ["Pmn21", "P21nm", "Pmnm"]},
        {"Code": 63, "Extinction Group": "P – n a", "Space Groups": ["P2na", "Pmna"]},
        {"Code": 64, "Extinction Group": "P – n b", "Space Groups": ["P21nb", "Pmnb"]},
        {"Code": 65, "Extinction Group": "P – n n", "Space Groups": ["P2nn", "Pmnn"]},
        {"Code": 66, "Extinction Group": "P b – –", "Space Groups": ["Pbm2", "Pb21m", "Pbmm"]},
        {"Code": 67, "Extinction Group": "P b – a", "Space Groups": ["Pb21a", "Pbma"]},
        {"Code": 68, "Extinction Group": "P b – b", "Space Groups": ["Pb2b", "Pbmb"]},
        {"Code": 69, "Extinction Group": "P b – n", "Space Groups": ["Pb2n", "Pbmn"]},
        {"Code": 70, "Extinction Group": "P b a –", "Space Groups": ["Pba2", "Pbam"]},
        {"Code": 71, "Extinction Group": "P b a a", "Space Groups": ["Pbaa"]},
        {"Code": 72, "Extinction Group": "P b a b", "Space Groups": ["Pbab"]},
        {"Code": 73, "Extinction Group": "P b a n", "Space Groups": ["Pban"]},
        {"Code": 74, "Extinction Group": "P b c –", "Space Groups": ["Pbc21", "Pbcm"]},
        {"Code": 75, "Extinction Group": "P b c a", "Space Groups": ["Pbca"]},
        {"Code": 76, "Extinction Group": "P b c b", "Space Groups": ["Pbcb"]},
        {"Code": 77, "Extinction Group": "P b c n", "Space Groups": ["Pbcn"]},
        {"Code": 78, "Extinction Group": "P b n -", "Space Groups": ["Pbn21", "Pbnm"]},
        {"Code": 79, "Extinction Group": "P b n a", "Space Groups": ["Pbna"]},
        {"Code": 80, "Extinction Group": "P b n b", "Space Groups": ["Pbnb"]},
        {"Code": 81, "Extinction Group": "P b n n", "Space Groups": ["Pbnn"]},
        {"Code": 82, "Extinction Group": "P c – –", "Space Groups": ["Pcm21", "Pc2m", "Pcmm"]},
        {"Code": 83, "Extinction Group": "P c – a", "Space Groups": ["Pc2a", "Pcma"]},
        {"Code": 84, "Extinction Group": "P c – b", "Space Groups": ["Pc21b", "Pcmb"]},
        {"Code": 85, "Extinction Group": "P c – n", "Space Groups": ["Pc21n", "Pcmn"]},
        {"Code": 86, "Extinction Group": "P c a –", "Space Groups": ["Pca21", "Pcam"]},
        {"Code": 87, "Extinction Group": "P c a a", "Space Groups": ["Pcaa"]},
        {"Code": 88, "Extinction Group": "P c a b", "Space Groups": ["Pcab"]},
        {"Code": 89, "Extinction Group": "P c a n", "Space Groups": ["Pcan"]},
        {"Code": 90, "Extinction Group": "P c c –", "Space Groups": ["Pcc2", "Pccm"]},
        {"Code": 91, "Extinction Group": "P c c a", "Space Groups": ["Pcca"]},
        {"Code": 92, "Extinction Group": "P c c b", "Space Groups": ["Pccb"]},
        {"Code": 93, "Extinction Group": "P c c n", "Space Groups": ["Pccn"]},
        {"Code": 94, "Extinction Group": "P c n –", "Space Groups": ["Pcn2", "Pcnm"]},
        {"Code": 95, "Extinction Group": "P c n a", "Space Groups": ["Pcna"]},
        {"Code": 96, "Extinction Group": "P c n b", "Space Groups": ["Pcnb"]},
        {"Code": 97, "Extinction Group": "P c n n", "Space Groups": ["Pcnn"]},
        {"Code": 98, "Extinction Group": "P n – –", "Space Groups": ["Pnm21", "Pnmm", "Pn21m"]},
        {"Code": 99, "Extinction Group": "P n – a", "Space Groups": ["Pn21a", "Pnma"]},
        {"Code": 100, "Extinction Group": "P n – b", "Space Groups": ["Pn2b", "Pnmb"]},
        {"Code": 101, "Extinction Group": "P n – n", "Space Groups": ["Pn2n", "Pnmn"]},
        {"Code": 102, "Extinction Group": "P n a –", "Space Groups": ["Pna21", "Pnam"]},
        {"Code": 103, "Extinction Group": "P n a a", "Space Groups": ["Pnaa"]},
        {"Code": 104, "Extinction Group": "P n a b", "Space Groups": ["Pnab"]},
        {"Code": 105, "Extinction Group": "P n a n", "Space Groups": ["Pnan"]},
        {"Code": 106, "Extinction Group": "P n c –", "Space Groups": ["Pnc2", "Pncm"]},
        {"Code": 107, "Extinction Group": "P n c a", "Space Groups": ["Pnca"]},
        {"Code": 108, "Extinction Group": "P n c b", "Space Groups": ["Pncb"]},
        {"Code": 109, "Extinction Group": "P n c n", "Space Groups": ["Pncn"]},
        {"Code": 110, "Extinction Group": "P n n –", "Space Groups": ["Pnn2", "Pnnm"]},
        {"Code": 111, "Extinction Group": "P n n a", "Space Groups": ["Pnna"]},
        {"Code": 112, "Extinction Group": "P n n b", "Space Groups": ["Pnnb"]},
        {"Code": 113, "Extinction Group": "P n n n", "Space Groups": ["Pnnn"]},
        {"Code": 114, "Extinction Group": "C – – –", "Space Groups": ["C222", "Cmm2", "Cmmm", "Cm2m", "C2mm"]},
        {"Code": 115, "Extinction Group": "C – – 21", "Space Groups": ["C2221"]},
        {"Code": 116, "Extinction Group": "C – – (ab)", "Space Groups": ["Cm2a", "Cmma", "C2mb", "Cmmb"]},
        {"Code": 117, "Extinction Group": "C – c –", "Space Groups": ["Cmc21", "Cmcm", "C2cm"]},
        {"Code": 118, "Extinction Group": "C – c (ab)", "Space Groups": ["C2cb", "Cmca"]},
        {"Code": 119, "Extinction Group": "C c – –", "Space Groups": ["Ccm21", "Ccmm", "Cc2m"]},
        {"Code": 120, "Extinction Group": "C c – (ab)", "Space Groups": ["Cc2a", "Ccmb"]},
        {"Code": 121, "Extinction Group": "C c c –", "Space Groups": ["Ccc2", "Cccm"]},
        {"Code": 122, "Extinction Group": "C c c (ab)", "Space Groups": ["Ccca", "Cccb"]},
        {"Code": 123, "Extinction Group": "B – – –", "Space Groups": ["B222", "Bmm2", "Bmmm", "Bm2m", "B2mm"]},
        {"Code": 124, "Extinction Group": "B – 21 –", "Space Groups": ["B2212"]},
        {"Code": 125, "Extinction Group": "B – – b", "Space Groups": ["Bm21b", "Bmmb", "B2mb"]},
        {"Code": 126, "Extinction Group": "B – (ac)-", "Space Groups": ["Bma2", "Bmam", "B2cm", "Bmcm"]},
        {"Code": 127, "Extinction Group": "B – (ac)b", "Space Groups": ["B2cb", "Bmab"]},
        {"Code": 128, "Extinction Group": "B b – –", "Space Groups": ["Bbm2", "Bbmm", "Bb21m"]},
        {"Code": 129, "Extinction Group": "B b – b", "Space Groups": ["Bb2b", "Bbmb"]},
        {"Code": 130, "Extinction Group": "B b (ac)-", "Space Groups": ["Bba2", "Bbcm"]},
        {"Code": 131, "Extinction Group": "B b (ac)b", "Space Groups": ["Bbab", "Bbcb"]},
        {"Code": 132, "Extinction Group": "A – – –", "Space Groups": ["A222", "Amm2", "Ammm", "Am2m", "A2mm"]},
        {"Code": 133, "Extinction Group": "A 21 – –", "Space Groups": ["A2122"]},
        {"Code": 134, "Extinction Group": "A – – a", "Space Groups": ["Am2a", "Amma", "A21ma"]},
        {"Code": 135, "Extinction Group": "A – a –", "Space Groups": ["Ama2", "Amam", "A21am"]},
        {"Code": 136, "Extinction Group": "A – a a", "Space Groups": ["A2aa", "Amaa"]},
        {"Code": 137, "Extinction Group": "A(bc)- –", "Space Groups": ["Abm2", "Abmm", "Ac2m", "Acmm"]},
        {"Code": 138, "Extinction Group": "A(bc)- a", "Space Groups": ["Ac2a", "Abma"]},
        {"Code": 139, "Extinction Group": "A(bc)a –", "Space Groups": ["Aba2", "Acam"]},
        {"Code": 140, "Extinction Group": "A(bc)a a", "Space Groups": ["Abaa", "Acaa"]},
        {"Code": 141, "Extinction Group": "I – – –", "Space Groups": ["I222", "Imm2", "Immm", "I212121", "Im2m", "I2mm"]},
        {"Code": 142, "Extinction Group": "I – – (ab)", "Space Groups": ["Im2a", "Imma", "I2mb", "Immb"]},
        {"Code": 143, "Extinction Group": "I – (ac)-", "Space Groups": ["Ima2", "Imam", "I2cm", "Imcm"]},
        {"Code": 144, "Extinction Group": "I – c b", "Space Groups": ["I2cb", "Imcb"]},
        {"Code": 145, "Extinction Group": "I(bc)- –", "Space Groups": ["Ibm2", "Ibmm", "Ic2m", "Icmm"]},
        {"Code": 146, "Extinction Group": "I c – a", "Space Groups": ["Ic2a", "Icma"]},
        {"Code": 147, "Extinction Group": "I b a –", "Space Groups": ["Iba2", "Ibam"]},
        {"Code": 148, "Extinction Group": "I b c a", "Space Groups": ["Ibca", "Icab"]}, #I c a b is not a spacegroup. I added it to prevent a fail
        {"Code": 149, "Extinction Group": "F – – –", "Space Groups": ["F222", "Fmm2", "Fmmm", "Fm2m", "F2mm"]},
        {"Code": 150, "Extinction Group": "F – d d", "Space Groups": ["F2dd"]},
        {"Code": 151, "Extinction Group": "F d – d", "Space Groups": ["Fd2d"]},
        {"Code": 152, "Extinction Group": "F d d –", "Space Groups": ["Fdd2"]},
        {"Code": 153, "Extinction Group": "F d d d", "Space Groups": ["Fddd"]},
        {"Code": 154, "Extinction Group": "P – – –", "Space Groups": ["P4", "P-4", "P4/m", "P422", "P4mm", "P-42m", "P4/mmm", "P-4m2"]},
        {"Code": 155, "Extinction Group": "P – 21 –", "Space Groups": ["P4212", "P-421m"]},
        {"Code": 156, "Extinction Group": "P 42 – –", "Space Groups": ["P42", "P42/m", "P4222"]},
        {"Code": 157, "Extinction Group": "P 42 21 –", "Space Groups": ["P42212"]},
        {"Code": 158, "Extinction Group": "P 41 – –", "Space Groups": ["P41", "P43", "P4122", "P4322"]},
        {"Code": 159, "Extinction Group": "P 41 21 –", "Space Groups": ["P41212", "P43212"]},
        {"Code": 160, "Extinction Group": "P – – c", "Space Groups": ["P42mc", "P-42c", "P42/mmc"]},
        {"Code": 161, "Extinction Group": "P – 21 c", "Space Groups": ["P-421c"]},
        {"Code": 162, "Extinction Group": "P – b –", "Space Groups": ["P4bm", "P-4b2", "P4/mbm"]},
        {"Code": 163, "Extinction Group": "P – b c", "Space Groups": ["P42bc", "P42/mbc"]},
        {"Code": 164, "Extinction Group": "P – c –", "Space Groups": ["P42cm", "P-4c2", "P42/mcm"]},
        {"Code": 165, "Extinction Group": "P – c c", "Space Groups": ["P4cc", "P4/mcc"]},
        {"Code": 166, "Extinction Group": "P – n –", "Space Groups": ["P42nm", "P-4n2", "P42/mnm"]},
        {"Code": 167, "Extinction Group": "P – n c", "Space Groups": ["P4nc", "P4/mnc"]},
        {"Code": 168, "Extinction Group": "P n – –", "Space Groups": ["P4/n", "P4/nmm"]},
        {"Code": 169, "Extinction Group": "P 42/n – –", "Space Groups": ["P42/n"]},
        {"Code": 170, "Extinction Group": "P n – c", "Space Groups": ["P42/nmc"]},
        {"Code": 171, "Extinction Group": "P n b –", "Space Groups": ["P4/nbm"]},
        {"Code": 172, "Extinction Group": "P n b c", "Space Groups": ["P42/nbc"]},
        {"Code": 173, "Extinction Group": "P n c –", "Space Groups": ["P42/ncm"]},
        {"Code": 174, "Extinction Group": "P n c c", "Space Groups": ["P4/ncc"]},
        {"Code": 175, "Extinction Group": "P n n –", "Space Groups": ["P42/nnm"]},
        {"Code": 176, "Extinction Group": "P n n c", "Space Groups": ["P4/nnc"]},
        {"Code": 177, "Extinction Group": "I – – –", "Space Groups": ["I4", "I-4", "I4/m", "I422", "I4mm", "I-42m", "I4/mmm", "I-4m2"]},
        {"Code": 178, "Extinction Group": "I 41 – –", "Space Groups": ["I41", "I4122"]},
        {"Code": 179, "Extinction Group": "I – – d", "Space Groups": ["I41md", "I-42d"]},
        {"Code": 180, "Extinction Group": "I – c –", "Space Groups": ["I4cm", "I-4c2", "I4/mcm"]},
        {"Code": 181, "Extinction Group": "I – c d", "Space Groups": ["I41cd"]},
        {"Code": 182, "Extinction Group": "I 41/a – –", "Space Groups": ["I41/a"]},
        {"Code": 183, "Extinction Group": "I a – d", "Space Groups": ["I41/amd"]},
        {"Code": 184, "Extinction Group": "I a c d", "Space Groups": ["I41/acd"]},
        {"Code": 185, "Extinction Group": "P – – –", "Space Groups": ["P3", "P-3", "P321", "P3m1", "P-3m1", "P312", "P31m", "P-31m", "P6", "P-6", "P6/m", "P622", "P6mm", "P-62m", "P6/mmm", "P-6m2"]},
        {"Code": 186, "Extinction Group": "P 31 – –", "Space Groups": ["P31", "P3121", "P3112", "P32", "P3221", "P3212"]},
        {"Code": 187, "Extinction Group": "P – – c", "Space Groups": ["P31c", "P-31c", "P63mc", "P-62c", "P63/mmc"]},
        {"Code": 188, "Extinction Group": "P – c –", "Space Groups": ["P3c1", "P-3c1", "P63cm", "P-6c2", "P63/mcm"]},
        {"Code": 189, "Extinction Group": "R (obv) – –", "Space Groups": ["R3", "R-3", "R32", "R3m", "R-3m"]},
        {"Code": 190, "Extinction Group": "R (obv)- – c", "Space Groups": ["R3c", "R-3c"]},
        {"Code": 191, "Extinction Group": "R (rev) – –", "Space Groups": ["R3", "R-3", "R32", "R3m", "R-3m"]},
        {"Code": 192, "Extinction Group": "R (rev)- – c", "Space Groups": ["R3c", "R-3c"]},
        {"Code": 193, "Extinction Group": "R – – –", "Space Groups": ["R3", "R-3", "R32", "R3m", "R-3m"]},
        {"Code": 194, "Extinction Group": "R – – c", "Space Groups": ["R3c", "R-3c"]},
        {"Code": 195, "Extinction Group": "P 63 – –", "Space Groups": ["P63", "P63/m", "P6322"]},
        {"Code": 196, "Extinction Group": "P 62 – –", "Space Groups": ["P62", "P6222", "P64", "P6422"]},
        {"Code": 197, "Extinction Group": "P 61 – –", "Space Groups": ["P61", "P6122", "P65", "P6522"]},
        {"Code": 198, "Extinction Group": "P – c c", "Space Groups": ["P6cc", "P6/mcc"]},
        {"Code": 199, "Extinction Group": "P – – –", "Space Groups": ["P23", "Pm-3", "P432", "P-43m", "Pm-3m"]},
        {"Code": 200, "Extinction Group": "P 21 – –", "Space Groups": ["P213"]},
        {"Code": 201, "Extinction Group": "P 42 – –", "Space Groups": ["P4232"]},
        {"Code": 202, "Extinction Group": "P 41 – –", "Space Groups": ["P4132", "P4332"]},
        {"Code": 203, "Extinction Group": "P – – n", "Space Groups": ["P-43n", "Pm-3n"]},
        {"Code": 204, "Extinction Group": "P a – –", "Space Groups": ["Pa-3"]},
        {"Code": 205, "Extinction Group": "P n – –", "Space Groups": ["Pn-3", "Pn-3m"]},
        {"Code": 206, "Extinction Group": "P n – n", "Space Groups": ["Pn-3n"]},
        {"Code": 207, "Extinction Group": "I – – –", "Space Groups": ["I23", "I213", "Im-3", "I432", "I-43m", "Im-3m"]},
        {"Code": 208, "Extinction Group": "I 41 – –", "Space Groups": ["I4132"]},
        {"Code": 209, "Extinction Group": "I – – d", "Space Groups": ["I-43d"]},
        {"Code": 210, "Extinction Group": "I a – –", "Space Groups": ["Ia-3"]},
        {"Code": 211, "Extinction Group": "I a – d", "Space Groups": ["Ia-3d"]},
        {"Code": 212, "Extinction Group": "F – – –", "Space Groups": ["F23", "Fm-3", "F432", "F-43m", "Fm-3m"]},
        {"Code": 213, "Extinction Group": "F 41 – –", "Space Groups": ["F4132"]},
        {"Code": 214, "Extinction Group": "F – – c", "Space Groups": ["F-43c", "Fm-3c"]},
        {"Code": 215, "Extinction Group": "F d – –", "Space Groups": ["Fd-3", "Fd-3m"]},
        {"Code": 216, "Extinction Group": "F d – c", "Space Groups": ["Fd-3c"]},
        {"Code": 217, "Extinction Group": "P –", "Space Groups": ["P1", "P-1"]},
        ]

    # Create a lookup dictionary for space groups
    spacegroup_symbol = spacegroup_symbol_hm.replace(' ', '')
    for row in table_data:
        if spacegroup_symbol in row["Space Groups"]:
            return row['Extinction Group'], row['Code']
        elif spacegroup_symbol_hm in row["Space Groups"]:
            return row['Extinction Group'], row['Code']
    else:
        print(f'{spacegroup_symbol_hm} {spacegroup_symbol} Not in lookup')
        return None, None


class Q2Calculator:
    def __init__(self, lattice_system, hkl, tensorflow, representation):
        self.lattice_system = lattice_system
        self.representation = representation
        if len(hkl.shape) == 2:
            self.multiple_hkl_sets = False
        elif len(hkl.shape) == 3:
            self.multiple_hkl_sets = True
        else:
            assert False
        if tensorflow:
            import tensorflow as tf
            assert self.representation == 'xnn'
            self.newaxis = tf.newaxis
            self.sin = tf.math.sin
            self.cos = tf.math.cos
            self.zeros = tf.zeros
            self.zeros_like = tf.zeros_like
            self.array = tf.constant
            self.dtype = tf.float32
            self.stack = tf.stack
            self.concatenate = tf.concat
            self.matmul = tf.linalg.matmul
            self.get_q2_xnn = self.get_q2_xnn_tensorflow
            self.sum = tf.math.reduce_sum
            hkl = tf.cast(hkl, dtype=tf.float32)
        else:
            self.newaxis = np.newaxis
            self.sin = np.sin
            self.cos = np.cos
            self.zeros = np.zeros
            self.zeros_like = np.zeros_like
            self.array = np.array
            self.dtype = None
            self.stack = np.stack
            self.concatenate = np.concatenate
            self.get_q2_xnn = self.get_q2_xnn_numpy

        if self.representation in ['xnn', 'reciprocal_unit_cell']:
            if self.lattice_system == 'monoclinic':
                self.hkl2 = self.concatenate((
                    hkl**2, 
                    (hkl[:, 0] * hkl[:, 2])[:, self.newaxis]
                    ),
                    axis=1
                    )
            elif self.lattice_system == 'triclinic':
                self.hkl2 = self.concatenate((
                    hkl[:, :3]**2,
                    (hkl[:, 1] * hkl[:, 2])[:, self.newaxis],
                    (hkl[:, 0] * hkl[:, 2])[:, self.newaxis],
                    (hkl[:, 0] * hkl[:, 1])[:, self.newaxis],
                    ),
                    axis=1
                    )
            elif self.lattice_system == 'orthorhombic':
                self.hkl2 = hkl**2
            elif self.lattice_system == 'tetragonal':
                self.hkl2 = self.stack((
                    hkl[:, 0]**2 + hkl[:, 1]**2,
                    hkl[:, 2]**2
                    ),
                    axis=1
                    )
            elif self.lattice_system == 'hexagonal':
                self.hkl2 = self.stack((
                    (hkl[..., 0]**2 + hkl[..., 0]*hkl[..., 1] + hkl[..., 1]**2),
                    hkl[..., 2]**2
                    ),
                    axis=-1
                    )
            elif self.lattice_system == 'rhombohedral':
                self.hkl2 = self.stack((
                    (hkl[:, 0]**2 + hkl[:, 1]**2 + hkl[:, 2]**2),
                    (hkl[:, 0]*hkl[:, 1] + hkl[:, 0]*hkl[:, 2] + hkl[:, 1]*hkl[:, 2]),
                    ),
                    axis=1
                    )
            elif self.lattice_system == 'cubic':
                self.hkl2 = (hkl[:, 0]**2 + hkl[:, 1]**2 + hkl[:, 2]**2)[:, self.newaxis]
            
        elif self.representation == 'unit_cell':
            self.hkl = hkl
            if self.lattice_system == 'cubic':
                self.get_q2_unit_cell = self.get_q2_cubic_unit_cell
            elif self.lattice_system == 'tetragonal':
                self.get_q2_unit_cell = self.get_q2_tetragonal_unit_cell
            elif self.lattice_system == 'orthorhombic':
                self.get_q2_unit_cell = self.get_q2_orthorhombic_unit_cell
            elif self.lattice_system == 'monoclinic':
                self.get_q2_unit_cell = self.get_q2_monoclinic_unit_cell
            elif self.lattice_system == 'triclinic':
                self.get_q2_unit_cell = self.get_q2_triclinic_unit_cell
            elif self.lattice_system == 'hexagonal':
                self.get_q2_unit_cell = self.get_q2_hexagonal_unit_cell
            elif self.lattice_system == 'rhombohedral':
                self.get_q2_unit_cell = self.get_q2_rhombohedral_unit_cell
            else:
                assert False
        else:
            assert False

    def get_q2(self, inputs):
        if self.representation == 'unit_cell':
            return self.get_q2_unit_cell(inputs)
        elif self.representation == 'reciprocal_unit_cell':
            xnn = get_xnn_from_reciprocal_unit_cell(
                inputs, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            return self.get_q2_xnn(xnn)
        elif self.representation == 'xnn':
            return self.get_q2_xnn(inputs)

    def get_q2_xnn_numpy(self, xnn):
        if self.multiple_hkl_sets:
            arg = self.hkl2 * xnn[:, np.newaxis, :]
            q2_pred = np.sum(arg, axis=2)
            return q2_pred
        else:
            return np.matmul(xnn, self.hkl2.T)
            
    def get_q2_xnn_tensorflow(self, xnn):
        if self.multiple_hkl_sets:
            arg = self.hkl2 * xnn[:, self.newaxis, :]
            q2_pred = self.sum(arg, axis=2)
            return q2_pred
        else:
            return self.matmul(xnn, self.hkl2, transpose_b=True)

    def get_q2_cubic_unit_cell(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        q2_ref = (self.hkl[:, 0]**2 + self.hkl[:, 1]**2 + self.hkl[:, 2]**2) / a**2
        return q2_ref

    def get_q2_tetragonal_unit_cell(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        c = unit_cell[:, 1][:, self.newaxis]
        q2_ref = (self.hkl[:, 0]**2 + self.hkl[:, 1]**2) / a**2 + self.hkl[:, 2]**2 / c**2
        return q2_ref

    def get_q2_orthorhombic_unit_cell(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis] 
        b = unit_cell[:, 1][:, self.newaxis]
        c = unit_cell[:, 2][:, self.newaxis]
        q2_ref = self.hkl[:, 0]**2 / a**2 + self.hkl[:, 1]**2 / b**2 + self.hkl[:, 2]**2 / c**2
        return q2_ref

    def get_q2_monoclinic_unit_cell(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis] 
        b = unit_cell[:, 1][:, self.newaxis]
        c = unit_cell[:, 2][:, self.newaxis]
        cos_beta = self.cos(unit_cell[:, 3][:, self.newaxis])
        sin_beta = self.sin(unit_cell[:, 3][:, self.newaxis])

        term0 = self.hkl[:, 0]**2 / a**2
        term1 = self.hkl[:, 1]**2 * sin_beta**2 / b**2
        term2 = self.hkl[:, 2]**2 / c**2
        term3 = -2 * self.hkl[:, 0] * self.hkl[:, 2] * cos_beta / (a * c)
        term4 = 1 / sin_beta**2
        q2_ref = term4 * (term0 + term1 + term2 + term3)
        return q2_ref

    def get_q2_triclinic_unit_cell(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        b = unit_cell[:, 1][:, self.newaxis]
        c = unit_cell[:, 2][:, self.newaxis]
        calpha = self.cos(unit_cell[:, 3])[:, self.newaxis]
        cbeta = self.cos(unit_cell[:, 4])[:, self.newaxis]
        cgamma = self.cos(unit_cell[:, 5])[:, self.newaxis]
        salpha = self.sin(unit_cell[:, 3])[:, self.newaxis]
        sbeta = self.sin(unit_cell[:, 4])[:, self.newaxis]
        sgamma = self.sin(unit_cell[:, 5])[:, self.newaxis]

        denom = 1 + 2*calpha*cbeta*cgamma - calpha**2 - cbeta**2 - cgamma**2
        term0 = self.hkl[:, 0]**2 * salpha**2 / a**2
        term1 = self.hkl[:, 1]**2 * sbeta**2 / b**2
        term2 = self.hkl[:, 2]**2 * sgamma**2 / c**2
        term3a = 2*self.hkl[:, 0]*self.hkl[:, 1] / (a*b)
        term3b = calpha*cbeta - cgamma
        term3 = term3a * term3b
        term4a = 2*self.hkl[:, 1]*self.hkl[:, 2] / (b*c)
        term4b = cbeta*cgamma - calpha
        term4 = term4a * term4b
        term5a = 2*self.hkl[:, 0]*self.hkl[:, 2] / (a*c)
        term5b = calpha*cgamma - cbeta
        term5 = term5a * term5b
        q2_ref = (term0 + term1 + term2 + term3 + term4 + term5) / denom
        return q2_ref

    def get_q2_rhombohedral_unit_cell(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        alpha = unit_cell[:, 1][:, self.newaxis]
        cos_alpha = self.cos(alpha)
        sin_alpha = self.sin(alpha)
        h = self.hkl[:, 0]
        k = self.hkl[:, 1]
        l = self.hkl[:, 2]
        term0 = (h**2 + k**2 + l**2) * sin_alpha**2
        term1 = h*k + k*l + h*l
        term2 = 2 * (cos_alpha**2 - cos_alpha)
        term3 = a**2 * (1 - 3*cos_alpha**2 + 2*cos_alpha**3)
        q2_ref = (term0 + term1 * term2) / term3
        return q2_ref

    def get_q2_hexagonal_unit_cell(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        c = unit_cell[:, 1][:, self.newaxis]
        h = self.hkl[:, 0]
        k = self.hkl[:, 1]
        l = self.hkl[:, 2]
        q2_ref = 4/3 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2
        return q2_ref


class PairwiseDifferenceCalculator(Q2Calculator):
    def __init__(self, lattice_system, hkl_ref, tensorflow, q2_scaler):
        super().__init__(lattice_system, hkl_ref, tensorflow, 'xnn')
        self.q2_scaler = q2_scaler

    def get_pairwise_differences(self, xnn, q2_scaled, return_q2_ref=False):
        q2_ref = self.get_q2(xnn)
        q2_ref_scaled = (q2_ref - self.q2_scaler.mean_[0]) / self.q2_scaler.scale_[0]
        # d_spacing_ref: n_entries x hkl_ref_length
        # x: n_entries x n_peaks
        # differences = n_entries x n_peaks x hkl_ref_length
        pairwise_differences_scaled = q2_ref_scaled[:, self.newaxis, :] - q2_scaled[:, :, self.newaxis]
        if return_q2_ref:
            return pairwise_differences_scaled, q2_ref
        else:
            return pairwise_differences_scaled


def write_params(params, filename):
    with open(filename, 'w') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=params.keys())
        writer.writeheader()
        writer.writerow(params)


def read_params(filename):
    with open(filename, 'r') as params_file:
        reader = csv.DictReader(params_file)
        for row in reader:
            params = row
    return params


def vectorized_resampling(softmaxes, rng):
    # This is a major performance bottleneck

    # This function randomly resamples the peaks using the algorithm
    #  1: Pick a peak at random
    #  2: Assign Miller index according to softmaxes
    #  3: Set the assigned Miller index softmax to zero for all other peaks
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]

    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    random_values = rng.random(size=(n_entries, n_peaks))
    point_order = rng.permutation(n_peaks)
    softmaxes_zeroed = softmaxes.copy()
    i = 0
    for point_index in point_order:
        # Fast random selection:
        #  1: make cummulative sum along the distribution's axis (this is a cdf)
        #  2: selection is the first point in cummulative sum greater than random value
        #    - fastest way to do this, convert to bool array and find first True with argmax
        #    - To account for adding zeros to the softmax array, the random values are scaled
        #      instead of scaling the softmax array

        # This line is slow (60% of execution time)
        cumsum = np.cumsum(softmaxes_zeroed[:, point_index, :], axis=1)
        q = cumsum >= (random_values[:, point_index] * cumsum[:, -1])[:, np.newaxis]
        hkl_assign[:, point_index] = np.argmax(q, axis=1)
        i += 1
        if i < n_peaks:
            np.put_along_axis(
                softmaxes_zeroed,
                hkl_assign[:, point_index][:, np.newaxis, np.newaxis],
                values=0,
                axis=2
                )

    softmax = np.take_along_axis(softmaxes, hkl_assign[:, :, np.newaxis], axis=2)[:, :, 0]
    return hkl_assign, softmax


def vectorized_subsampling(p, n_picks, rng):
    n_entries = p.shape[0]
    n_choices = p.shape[1]
    choices = np.repeat(np.arange(n_choices)[np.newaxis], repeats=n_entries, axis=0) 
    chosen = np.zeros((n_entries, n_picks), dtype=int)
    for index in range(n_picks):
        # cumsum: n_entries, n_peaks
        # random_value: n_entries
        # q: n_entries, n_peaks
        n_peaks = p.shape[1]
        cumsum = p.cumsum(axis=1)
        random_value = rng.random(n_entries)
        q = cumsum >= random_value[:, np.newaxis]
        chosen_indices = q.argmax(axis=1)
        chosen[:, index] = choices[np.arange(n_entries), chosen_indices]
        p_flat = p.ravel()
        choices_flat = choices.ravel()
        delete_indices = np.arange(n_entries) * n_peaks + chosen_indices
        p = np.delete(p_flat, delete_indices).reshape((n_entries, n_peaks - 1))
        choices = np.delete(choices_flat, delete_indices).reshape((n_entries, n_peaks - 1))
    chosen = np.sort(chosen, axis=1)
    return chosen


def best_assign_nocommon_original(softmaxes):
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]
    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)

    peak_choice = np.argsort(np.max(softmaxes, axis=2), axis=1)
    for candidate_index in range(n_entries):
        softmaxes_zeroed = softmaxes[candidate_index].copy()
        for peak_index in peak_choice[candidate_index]:
            choice = np.argmax(softmaxes_zeroed[peak_index, :])
            hkl_assign[candidate_index, peak_index] = choice
            softmaxes_zeroed[:, hkl_assign[candidate_index, peak_index]] = 0

    softmax_assign = np.take_along_axis(softmaxes, hkl_assign[:, :, np.newaxis], axis=2)
    return hkl_assign, softmax_assign


def best_assign_nocommon(softmaxes):
    # This is three times faster than the version above.
    # It picks the first occurance as opposed to the best occurance.
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_ref_length = softmaxes.shape[2]
    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    softmax_assign = np.zeros((n_entries, n_peaks))
    for peak_index in range(n_peaks):
        softmaxes_peak = softmaxes[:, peak_index, :]
        hkl_assign[:, peak_index] = np.argmax(softmaxes_peak, axis=1)
        softmax_assign[:, peak_index] = np.take_along_axis(
            softmaxes_peak, hkl_assign[:, peak_index][:, np.newaxis],
            axis=1
            )[:, 0]
        np.put(softmaxes, hkl_assign[:, np.newaxis, :], 0)
    return hkl_assign, softmax_assign


def assign_hkl_triplets(triplets_obs, hkl_assign, triplet_hkl_ref, q2_ref_calc):
    top_n = hkl_assign.shape[2]
    n_candidates = hkl_assign.shape[0]
    n_triplets = triplets_obs.shape[0]
    hkl_assign_triplets = np.zeros((n_candidates, n_triplets), dtype=np.uint16)
    for candidate_index in range(n_candidates):
        hkl_assign_candidate = hkl_assign[candidate_index]
        q2_ref_calc_candidate = q2_ref_calc[candidate_index]
        for triplet_index in range(n_triplets): 
            triplet_loop = triplets_obs[triplet_index]
            hkl_assign_0_top_n = hkl_assign_candidate[int(triplet_loop[0])]
            hkl_assign_1_top_n = hkl_assign_candidate[int(triplet_loop[1])]
            hkl_assign_pair = []
            for top_n_index_0 in range(top_n):
                hkl_assign_0 = hkl_assign_0_top_n[top_n_index_0]
                for top_n_index_1 in range(top_n):
                    hkl_assign_1 = hkl_assign_1_top_n[top_n_index_1]
                    if hkl_assign_0 < hkl_assign_1:
                        hkl_assign_pair += triplet_hkl_ref[hkl_assign_0][hkl_assign_1]
                    elif hkl_assign_0 > hkl_assign_1:
                        hkl_assign_pair += triplet_hkl_ref[hkl_assign_1][hkl_assign_0]
            if len(hkl_assign_pair) > 0:
                diff = np.abs(triplet_loop[2] - q2_ref_calc_candidate[hkl_assign_pair])
                min_index = np.argmin(diff)
                hkl_assign_triplets[candidate_index, triplet_index] = hkl_assign_pair[min_index]
    return hkl_assign_triplets


from numba import jit
@jit(fastmath=True)
def fast_assign(q2_obs, q2_ref):
    n_obs = q2_obs.size
    n_candidates = q2_ref.shape[0]
    n_ref = q2_ref.shape[1]
    hkl_assign = np.zeros((n_candidates, n_obs), dtype=np.uint16)
    for candidate_index in range(n_candidates):
        for obs_index in range(n_obs):
            current_min = 100
            current_min_index = None
            for ref_index in range(n_ref):
                diff = abs(q2_obs[obs_index] - q2_ref[candidate_index, ref_index])
                if diff < current_min:
                    current_min = diff
                    current_min_index = ref_index
            hkl_assign[candidate_index, obs_index] = current_min_index
    return hkl_assign


@jit(fastmath=True)
def fast_assign_top_n(q2_obs, q2_ref, top_n):
    n_obs = q2_obs.size
    n_candidates = q2_ref.shape[0]
    n_ref = q2_ref.shape[1]
    hkl_assign = np.zeros((n_candidates, n_obs, top_n), dtype=np.uint16)
    for candidate_index in range(1):
        for obs_index in range(n_obs):
            current_min = [100 for _ in range(top_n)]
            current_min_index = [0 for _ in range(top_n)]
            for ref_index in range(n_ref):
                diff = abs(q2_obs[obs_index] - q2_ref[candidate_index, ref_index])
                # bisect.bisect_left could be used here, but it is not supported by numba
                status = True
                bisect_index = top_n - 1
                diff_index = top_n
                # Most reference peaks are far away, so look through array backwards
                while status:
                    if diff < current_min[bisect_index]:
                        diff_index = bisect_index
                    else:
                        status = False                        
                    bisect_index -= 1
                    if bisect_index < 0:
                        status = False
                if diff_index < top_n:
                    current_min.insert(diff_index, diff)
                    current_min.pop()
                    current_min_index.insert(diff_index, ref_index)
                    current_min_index.pop()
            hkl_assign[candidate_index, obs_index, :] = current_min_index
    return hkl_assign
