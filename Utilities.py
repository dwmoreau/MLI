import csv
import numpy as np
import os


def get_peak_generation_info():
    # This information gets used in GenerateDataset.py and Augmentor.py
    # The CCDC API generates diffraction patterns with the wavelength 1.54 (Cu K alpha)
    # The CCDC powder patterns are generated with 'broadening_fwhm'
    # The cctbx powder patterns are generated with 'broadening_params'
    wavelength = 1.54
    dtheta2 = 0.01
    theta2_min = 2
    theta2_max = 60
    theta2_pattern = np.arange(theta2_min, theta2_max, dtheta2)
    peak_generation_info = {
        'broadening_fwhm': 0.1,
        'broadening_params': np.array([0.0001, 0.005]),
        'broadening_multiples': np.array([0.5, 1, 1.5]),
        'broadening_tags': ['0.5', '1', '1.5'],
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
        ])
    S_inv = np.linalg.inv(S.T)

    a_inv = np.sqrt(S_inv[:, 0, 0])
    b_inv = np.sqrt(S_inv[:, 1, 1])
    c_inv = np.sqrt(S_inv[:, 2, 2])

    alpha_inv = np.arccos(S_inv[:, 1, 2] / (b_inv * c_inv))
    beta_inv = np.arccos(S_inv[:, 0, 2] / (a_inv * c_inv))
    gamma_inv = np.arccos(S_inv[:, 0, 1] / (a_inv * b_inv))
    
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
            xnn = fix_unphysical_triclinic(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'monoclinic':
            xnn = fix_unphysical_monoclinic(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'rhombohedral':
            xnn = fix_unphysical_rhombohedral(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system in ['cubic', 'orthorhombic', 'tetragonal', 'hexagonal']:
            xnn = fix_unphysical_box(
                xnn=xnn, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        return xnn
    elif not unit_cell is None:
        if lattice_system == 'triclinic':
            unit_cell = fix_unphysical_triclinic(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'monoclinic':
            unit_cell = fix_unphysical_monoclinic(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system == 'rhombohedral':
            unit_cell = fix_unphysical_rhombohedral(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        elif lattice_system in ['cubic', 'orthorhombic', 'tetragonal', 'hexagonal']:
            unit_cell = fix_unphysical_box(
                unit_cell=unit_cell, rng=rng, minimum_unit_cell=minimum_unit_cell, maximum_unit_cell=maximum_unit_cell
                )
        return unit_cell


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
    good_indices = q2_calc.sum(axis=1) != 0
    expected_discrepancy = np.zeros(q2_calc.shape[0])
    expected_discrepancy[good_indices] = last_smaller_ref_peak[good_indices] / (2*N[good_indices])
    M20 = expected_discrepancy / discrepancy
    return M20


def get_M20_likelihood_from_xnn(q2_obs, xnn, hkl, hkl_ref, lattice_system, bravais_lattice):
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    q2_ref_calc = np.sum(hkl2_ref * xnn[:, np.newaxis, :], axis=2)
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True, lattice_system=lattice_system)
    reciprocal_volume = get_unit_cell_volume(reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system)
    return get_M20_taupin88(q2_obs, q2_calc, q2_ref_calc, bravais_lattice, reciprocal_volume)


def get_M20_likelihood(q2_obs, q2_calc, q2_ref_calc, bravais_lattice, reciprocal_volume):
    # This was inspired by Taupin 1988
    # Probability that a peak is correctly assigned:
    # arg = Expected number of peaks within error from random unit cell
    # P = 1 / (1 + arg)
    mu, nu = get_multiplicity_taupin88(bravais_lattice)
    observed_difference2 = (np.sqrt(q2_obs[np.newaxis]) - np.sqrt(q2_calc))**2
    arg = 8*np.pi*q2_obs * np.sqrt(observed_difference2) / (reciprocal_volume[:, np.newaxis] * mu)
    return -np.sum(np.log(1/(1 + arg)), axis=1)


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


class Q2Calculator:
    def __init__(self, lattice_system, hkl, tensorflow, representation):
        self.lattice_system = lattice_system
        self.representation = representation
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
                    (hkl[:, 0]**2 + hkl[:, 0]*hkl[:, 1] + hkl[:, 1]**2),
                    hkl[:, 2]**2
                    ),
                    axis=1
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
        return np.matmul(xnn, self.hkl2.T)

    def get_q2_xnn_tensorflow(self, xnn):
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
