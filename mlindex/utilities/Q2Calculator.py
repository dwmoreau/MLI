import numpy as np

from mlindex.utilities.UnitCellTools import get_xnn_from_reciprocal_unit_cell


class Q2Calculator:
    def __init__(self, lattice_system, hkl, tensorflow, representation):
        self.lattice_system = lattice_system
        self.representation = representation
        self.tensorflow = tensorflow
        if len(hkl.shape) == 2:
            self.multiple_hkl_sets = False
        elif len(hkl.shape) == 3:
            self.multiple_hkl_sets = True
        else:
            assert False
        if tensorflow:
            import keras

            assert self.representation == "xnn"
            self.sin = keras.ops.sin
            self.cos = keras.ops.cos
            self.dtype = "float32"
            self.stack = keras.ops.stack
            self.concatenate = keras.ops.concatenate
            self.matmul = keras.ops.matmul
            self.get_q2_xnn = self.get_q2_xnn_tensorflow
            self.sum = keras.ops.sum
            self.transpose = keras.ops.transpose
            self._keras_expand_dims = keras.ops.expand_dims
            hkl = keras.ops.cast(hkl, dtype="float32")
        else:
            self.sin = np.sin
            self.cos = np.cos
            self.dtype = None
            self.stack = np.stack
            self.concatenate = np.concatenate
            self.get_q2_xnn = self.get_q2_xnn_numpy
            self.sum = np.sum

        if self.representation in ["xnn", "reciprocal_unit_cell"]:
            if self.lattice_system == "monoclinic":
                self.hkl2 = self.concatenate(
                    (hkl**2, self._expand_dims(hkl[:, 0] * hkl[:, 2], axis=1)), axis=1
                )
            elif self.lattice_system == "triclinic":
                self.hkl2 = self.concatenate(
                    (
                        hkl[:, :3] ** 2,
                        self._expand_dims(hkl[:, 1] * hkl[:, 2], axis=1),
                        self._expand_dims(hkl[:, 0] * hkl[:, 2], axis=1),
                        self._expand_dims(hkl[:, 0] * hkl[:, 1], axis=1),
                    ),
                    axis=1,
                )
            elif self.lattice_system == "orthorhombic":
                self.hkl2 = hkl**2
            elif self.lattice_system == "tetragonal":
                self.hkl2 = self.stack(
                    (hkl[:, 0] ** 2 + hkl[:, 1] ** 2, hkl[:, 2] ** 2), axis=1
                )
            elif self.lattice_system == "hexagonal":
                self.hkl2 = self.stack(
                    (
                        (
                            hkl[..., 0] ** 2
                            + hkl[..., 0] * hkl[..., 1]
                            + hkl[..., 1] ** 2
                        ),
                        hkl[..., 2] ** 2,
                    ),
                    axis=-1,
                )
            elif self.lattice_system == "rhombohedral":
                self.hkl2 = self.stack(
                    (
                        (hkl[:, 0] ** 2 + hkl[:, 1] ** 2 + hkl[:, 2] ** 2),
                        (
                            hkl[:, 0] * hkl[:, 1]
                            + hkl[:, 0] * hkl[:, 2]
                            + hkl[:, 1] * hkl[:, 2]
                        ),
                    ),
                    axis=1,
                )
            elif self.lattice_system == "cubic":
                self.hkl2 = self._expand_dims(
                    hkl[:, 0] ** 2 + hkl[:, 1] ** 2 + hkl[:, 2] ** 2, axis=1
                )

        elif self.representation == "unit_cell":
            self.hkl = hkl
            if self.lattice_system == "cubic":
                self.get_q2_unit_cell = self.get_q2_cubic_unit_cell
            elif self.lattice_system == "tetragonal":
                self.get_q2_unit_cell = self.get_q2_tetragonal_unit_cell
            elif self.lattice_system == "orthorhombic":
                self.get_q2_unit_cell = self.get_q2_orthorhombic_unit_cell
            elif self.lattice_system == "monoclinic":
                self.get_q2_unit_cell = self.get_q2_monoclinic_unit_cell
            elif self.lattice_system == "triclinic":
                self.get_q2_unit_cell = self.get_q2_triclinic_unit_cell
            elif self.lattice_system == "hexagonal":
                self.get_q2_unit_cell = self.get_q2_hexagonal_unit_cell
            elif self.lattice_system == "rhombohedral":
                self.get_q2_unit_cell = self.get_q2_rhombohedral_unit_cell
            else:
                assert False
        else:
            assert False

    def _expand_dims(self, arr, axis):
        if self.tensorflow:
            return self._keras_expand_dims(arr, axis=axis)
        else:
            if axis == 0:
                return arr[np.newaxis]
            elif axis == 1:
                return arr[:, np.newaxis]
            elif axis == 2:
                return arr[:, :, np.newaxis]
            elif axis == 3:
                return arr[:, :, :, np.newaxis]
            elif axis == 4:
                return arr[:, :, :, :, np.newaxis]

    def get_q2(self, inputs):
        if self.representation == "unit_cell":
            return self.get_q2_unit_cell(inputs)
        elif self.representation == "reciprocal_unit_cell":
            xnn = get_xnn_from_reciprocal_unit_cell(
                inputs, partial_unit_cell=True, lattice_system=self.lattice_system
            )
            return self.get_q2_xnn(xnn)
        elif self.representation == "xnn":
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
            arg = self.hkl2 * self._expand_dims(xnn, axis=1)
            q2_pred = self.sum(arg, axis=2)
            return q2_pred
        else:
            return self.matmul(xnn, self.transpose(self.hkl2))

    def get_q2_cubic_unit_cell(self, unit_cell):
        a = self._expand_dims(unit_cell[:, 0], axis=1)
        q2_ref = (
            self.hkl[:, 0] ** 2 + self.hkl[:, 1] ** 2 + self.hkl[:, 2] ** 2
        ) / a**2
        return q2_ref

    def get_q2_tetragonal_unit_cell(self, unit_cell):
        a = self._expand_dims(unit_cell[:, 0], axis=1)
        c = self._expand_dims(unit_cell[:, 1], axis=1)
        q2_ref = (self.hkl[:, 0] ** 2 + self.hkl[:, 1] ** 2) / a**2 + self.hkl[
            :, 2
        ] ** 2 / c**2
        return q2_ref

    def get_q2_orthorhombic_unit_cell(self, unit_cell):
        a = self._expand_dims(unit_cell[:, 0], axis=1)
        b = self._expand_dims(unit_cell[:, 1], axis=1)
        c = self._expand_dims(unit_cell[:, 2], axis=1)
        q2_ref = (
            self.hkl[:, 0] ** 2 / a**2
            + self.hkl[:, 1] ** 2 / b**2
            + self.hkl[:, 2] ** 2 / c**2
        )
        return q2_ref

    def get_q2_monoclinic_unit_cell(self, unit_cell):
        a = self._expand_dims(unit_cell[:, 0], axis=1)
        b = self._expand_dims(unit_cell[:, 1], axis=1)
        c = self._expand_dims(unit_cell[:, 2], axis=1)
        cos_beta = self._expand_dims(self.cos(unit_cell[:, 3]), axis=1)
        sin_beta = self._expand_dims(self.sin(unit_cell[:, 3]), axis=1)

        term0 = self.hkl[:, 0] ** 2 / a**2
        term1 = self.hkl[:, 1] ** 2 * sin_beta**2 / b**2
        term2 = self.hkl[:, 2] ** 2 / c**2
        term3 = -2 * self.hkl[:, 0] * self.hkl[:, 2] * cos_beta / (a * c)
        term4 = 1 / sin_beta**2
        q2_ref = term4 * (term0 + term1 + term2 + term3)
        return q2_ref

    def get_q2_triclinic_unit_cell(self, unit_cell):
        a = self._expand_dims(unit_cell[:, 0], axis=1)
        b = self._expand_dims(unit_cell[:, 1], axis=1)
        c = self._expand_dims(unit_cell[:, 2], axis=1)
        calpha = self._expand_dims(self.cos(unit_cell[:, 3]), axis=1)
        cbeta = self._expand_dims(self.cos(unit_cell[:, 4]), axis=1)
        cgamma = self._expand_dims(self.cos(unit_cell[:, 5]), axis=1)
        salpha = self._expand_dims(self.sin(unit_cell[:, 3]), axis=1)
        sbeta = self._expand_dims(self.sin(unit_cell[:, 4]), axis=1)
        sgamma = self._expand_dims(self.sin(unit_cell[:, 5]), axis=1)

        denom = 1 + 2 * calpha * cbeta * cgamma - calpha**2 - cbeta**2 - cgamma**2
        term0 = self.hkl[:, 0] ** 2 * salpha**2 / a**2
        term1 = self.hkl[:, 1] ** 2 * sbeta**2 / b**2
        term2 = self.hkl[:, 2] ** 2 * sgamma**2 / c**2
        term3a = 2 * self.hkl[:, 0] * self.hkl[:, 1] / (a * b)
        term3b = calpha * cbeta - cgamma
        term3 = term3a * term3b
        term4a = 2 * self.hkl[:, 1] * self.hkl[:, 2] / (b * c)
        term4b = cbeta * cgamma - calpha
        term4 = term4a * term4b
        term5a = 2 * self.hkl[:, 0] * self.hkl[:, 2] / (a * c)
        term5b = calpha * cgamma - cbeta
        term5 = term5a * term5b
        q2_ref = (term0 + term1 + term2 + term3 + term4 + term5) / denom
        return q2_ref

    def get_q2_rhombohedral_unit_cell(self, unit_cell):
        a = self._expand_dims(unit_cell[:, 0], axis=1)
        alpha = self._expand_dims(unit_cell[:, 1], axis=1)
        cos_alpha = self.cos(alpha)
        sin_alpha = self.sin(alpha)
        h = self.hkl[:, 0]
        k = self.hkl[:, 1]
        l = self.hkl[:, 2]
        term0 = (h**2 + k**2 + l**2) * sin_alpha**2
        term1 = h * k + k * l + h * l
        term2 = 2 * (cos_alpha**2 - cos_alpha)
        term3 = a**2 * (1 - 3 * cos_alpha**2 + 2 * cos_alpha**3)
        q2_ref = (term0 + term1 * term2) / term3
        return q2_ref

    def get_q2_hexagonal_unit_cell(self, unit_cell):
        a = self._expand_dims(unit_cell[:, 0], axis=1)
        c = self._expand_dims(unit_cell[:, 1], axis=1)
        h = self.hkl[:, 0]
        k = self.hkl[:, 1]
        l = self.hkl[:, 2]
        q2_ref = 4 / 3 * (h**2 + h * k + k**2) / a**2 + l**2 / c**2
        return q2_ref


class PairwiseDifferenceCalculator(Q2Calculator):
    def __init__(self, lattice_system, hkl_ref, tensorflow, q2_scaler):
        super().__init__(lattice_system, hkl_ref, tensorflow, "xnn")
        # The conversion to float prevents a failure if the q2_scaler is a numpy object
        # This would occur if it was calculated from np.std() for example.
        self.q2_scaler = float(q2_scaler)

    def get_pairwise_differences(self, xnn, q2_scaled, return_q2_ref=False):
        q2_ref = self.get_q2(xnn)
        q2_ref_scaled = q2_ref / self.q2_scaler
        # d_spacing_ref: n_entries x hkl_ref_length
        # x: n_entries x n_peaks
        # differences = n_entries x n_peaks x hkl_ref_length
        q2_ref_scaled = self._expand_dims(q2_ref_scaled, axis=1)
        q2_scaled = self._expand_dims(q2_scaled, axis=2)
        pairwise_differences_scaled = q2_ref_scaled - q2_scaled
        if return_q2_ref:
            return pairwise_differences_scaled, q2_ref
        else:
            return pairwise_differences_scaled
