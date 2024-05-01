import csv
import numpy as np
import os



def get_fwhm_and_overlap_threshold():
    # This information gets used in GenerateDataset.py and Augmentor.py
    # This function removes duplicatation
    # The CCDC API generates diffraction patterns with the wavelength 1.54 (Cu K alpha)
    # by default.
    fwhm = 0.1
    overlap_threshold = fwhm / 1.5
    wavelength = 1.54
    return fwhm, overlap_threshold, wavelength


def reciprocal_uc_conversion(unit_cell, partial_unit_cell=False, lattice_system=None, radians=True):
    if radians:
        angle_multiplier = 1
    else:
        angle_multiplier = np.pi/180
    if partial_unit_cell:
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
            alpha = unit_cell[:, 1] * angle_multiplier
            beta = unit_cell[:, 1] * angle_multiplier
            gamma = unit_cell[:, 1] * angle_multiplier
        elif lattice_system == 'monoclinic':
            alpha = np.pi/2
            beta = unit_cell[:, 3] * angle_multiplier
            gamma = np.pi/2
        else:
            assert False
    elif partial_unit_cell == False or lattice_system == 'triclinic':
        a = unit_cell[:, 0]
        b = unit_cell[:, 1]
        c = unit_cell[:, 2]
        alpha = unit_cell[:, 3] * angle_multiplier
        beta = unit_cell[:, 4] * angle_multiplier
        gamma = unit_cell[:, 5] * angle_multiplier

    S = np.array([
        [a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
        [a*b*np.cos(gamma), b**2, b*c*np.cos(alpha)],
        [a*c*np.cos(beta), b*c*np.cos(alpha), c**2]
        ])
    S_inv = np.linalg.inv(S.T)
    a_inv = np.sqrt(S_inv[:, 0, 0])
    b_inv = np.sqrt(S_inv[:, 1, 1])
    c_inv = np.sqrt(S_inv[:, 2, 2])
    alpha_inv = np.arccos(S_inv[:, 1, 2] / (b_inv * c_inv)) /  angle_multiplier
    beta_inv = np.arccos(S_inv[:, 0, 2] / (a_inv * c_inv)) /  angle_multiplier
    gamma_inv = np.arccos(S_inv[:, 0, 1] / (a_inv * b_inv)) /  angle_multiplier
    
    if partial_unit_cell:
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


def get_xnn_from_reciprocal_unit_cell(reciprocal_unit_cell, partial_unit_cell=False, lattice_system=None, radians=True):
    if radians:
        angle_multiplier = 1
    else:
        angle_multiplier = np.pi/180
    if partial_unit_cell:
        if lattice_system in ['cubic', 'tetragonal', 'hexagonal', 'orthorhombic']:
            xnn = reciprocal_unit_cell**2
        elif lattice_system == 'rhombohedral':
            xnn = np.column_stack([
                reciprocal_unit_cell[:, 0]**2,
                2*reciprocal_unit_cell[:, 0]**2 * np.cos(angle_multiplier * reciprocal_unit_cell[:, 1]),
                ])
        elif lattice_system == 'monoclinic':
            xnn = np.column_stack([
                reciprocal_unit_cell[:, 0]**2,
                reciprocal_unit_cell[:, 1]**2,
                reciprocal_unit_cell[:, 2]**2,
                2*reciprocal_unit_cell[:, 0] * reciprocal_unit_cell[:, 2] * np.cos(angle_multiplier * reciprocal_unit_cell[:, 3]),
                ])
    elif partial_unit_cell == False or lattice_system == 'triclinic':
        xnn = np.column_stack([
            reciprocal_unit_cell[:, 0]**2,
            reciprocal_unit_cell[:, 1]**2,
            reciprocal_unit_cell[:, 2]**2,
            2*reciprocal_unit_cell[:, 1] * reciprocal_unit_cell[:, 2] * np.cos(angle_multiplier * reciprocal_unit_cell[:, 3]),
            2*reciprocal_unit_cell[:, 0] * reciprocal_unit_cell[:, 2] * np.cos(angle_multiplier * reciprocal_unit_cell[:, 4]),
            2*reciprocal_unit_cell[:, 0] * reciprocal_unit_cell[:, 1] * np.cos(angle_multiplier * reciprocal_unit_cell[:, 5]),
            ])
        xnn[reciprocal_unit_cell[:, 3] == 1/angle_multiplier * np.pi/2, 3] = 0
        xnn[reciprocal_unit_cell[:, 4] == 1/angle_multiplier * np.pi/2, 4] = 0
        xnn[reciprocal_unit_cell[:, 5] == 1/angle_multiplier * np.pi/2, 5] = 0
    return xnn


def get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=False, lattice_system=None):
    if partial_unit_cell:
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


def get_hkl_checks(hkl, lattice_system):
    if lattice_system == 'cubic':
        hkl_check = np.sum(hkl**2, axis=-1)
    elif lattice_system == 'tetragonal':
        hkl_check = np.stack((
            np.sum(hkl[..., :2]**2, axis=-1),
            hkl[..., 2]**2,
            ),
            axis=-1
            )
    elif lattice_system == 'orthorhombic':
        hkl_check = hkl**2
    elif lattice_system == 'monoclinic':
        hkl_check = np.stack((
            hkl[..., 0]**2,
            hkl[..., 1]**2,
            hkl[..., 2]**2,
            hkl[..., 0] * hkl[..., 2],
            ),
            axis=-1
            )
    elif lattice_system == 'triclinic':
        hkl_check = np.stack((
            hkl[..., 0]**2,
            hkl[..., 1]**2,
            hkl[..., 2]**2,
            hkl[..., 0] * hkl[..., 1],
            hkl[..., 0] * hkl[..., 2],
            hkl[..., 1] * hkl[..., 2],
            ),
            axis=-1
            )
    elif lattice_system == 'hexagonal':
        hkl_check = np.stack((
            hkl[..., 0]**2 + hkl[..., 0]*hkl[..., 1] + hkl[..., 1]**2,
            hkl[..., 2]**2,
            ),
            axis=-1
            )
    elif lattice_system == 'rhombohedral':
        hkl_check = np.stack((
            np.sum(hkl[..., :2]**2, axis=-1),
            hkl[..., 0]*hkl[..., 1] + hkl[..., 0]*hkl[..., 2] + hkl[..., 1]*hkl[..., 2],
            ),
            axis=-1
            )
    return hkl_check


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
            if self.lattice_system == 'cubic':
                self.hkl2 = (hkl[:, 0]**2 + hkl[:, 1]**2 + hkl[:, 2]**2)[:, self.newaxis]
            elif self.lattice_system == 'tetragonal':
                self.hkl2 = self.stack((
                    hkl[:, 0]**2 + hkl[:, 1]**2,
                    hkl[:, 2]**2
                    ),
                    axis=1
                    )
            elif self.lattice_system == 'orthorhombic':
                self.hkl2 = hkl**2
            elif self.lattice_system == 'monoclinic':
                self.hkl2 = self.concatenate((
                    hkl**2, 
                    (hkl[:, 0] * hkl[:, 2])[:, self.newaxis]
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
            elif self.lattice_system == 'triclinic':
                assert False
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

    def get_pairwise_differences(self, xnn, q2_scaled):
        q2_ref = self.get_q2(xnn)
        q2_ref_scaled = (q2_ref - self.q2_scaler.mean_[0]) / self.q2_scaler.scale_[0]
        # d_spacing_ref: n_entries x hkl_ref_length
        # x: n_entries x n_peaks
        # differences = n_entries x n_peaks x hkl_ref_length
        pairwise_differences_scaled = q2_ref_scaled[:, self.newaxis, :] - q2_scaled[:, :, self.newaxis]
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
