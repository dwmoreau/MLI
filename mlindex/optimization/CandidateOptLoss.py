import numpy as np


class CandidateOptLoss:
    def __init__(self, q2_obs, lattice_system):
        self.q2_obs = q2_obs
        self.lattice_system = lattice_system
        self.delta_q_eps = 1e-10
        self.n_peaks = self.q2_obs.shape[1]
        self.n_entries = self.q2_obs.shape[0]

        if lattice_system == 'cubic':
            self.uc_length = 1
        elif lattice_system in ['tetragonal', 'hexagonal', 'rhombohedral']:
            self.uc_length = 2
        elif lattice_system == 'orthorhombic':
            self.uc_length = 3
        elif lattice_system == 'monoclinic':
            self.uc_length = 4
        elif lattice_system == 'triclinic':
            self.uc_length = 6

    def update(self, hkl, xnn_init, power=None, sigma_reduction=None):
        self.hkl = hkl
        if self.lattice_system == 'triclinic':
            self.hkl2 = np.concatenate((
                self.hkl**2, 
                (self.hkl[:, :, 1] * self.hkl[:, :, 2])[:, :, np.newaxis],
                (self.hkl[:, :, 0] * self.hkl[:, :, 2])[:, :, np.newaxis],
                (self.hkl[:, :, 0] * self.hkl[:, :, 1])[:, :, np.newaxis],
                ),
                axis=2
                )
        elif self.lattice_system == 'monoclinic':
            self.hkl2 = np.concatenate((
                self.hkl**2, 
                (self.hkl[:, :, 0] * self.hkl[:, :, 2])[:, :, np.newaxis]
                ),
                axis=2
                )
        elif self.lattice_system == 'cubic':
            self.hkl2 = (self.hkl[:, :, 0]**2 + self.hkl[:, :, 1]**2 + self.hkl[:, :, 2]**2)[:, :, np.newaxis]
        elif self.lattice_system == 'tetragonal':
            self.hkl2 = np.stack((
                self.hkl[:, :, 0]**2 + self.hkl[:, :, 1]**2,
                self.hkl[:, :, 2]**2
                ),
                axis=2
                )
        elif self.lattice_system == 'orthorhombic':
            self.hkl2 = self.hkl**2
        elif self.lattice_system == 'hexagonal':
            self.hkl2 = np.stack((
                (self.hkl[:, :, 0]**2 + self.hkl[:, :, 0]*self.hkl[:, :, 1] + self.hkl[:, :, 1]**2),
                self.hkl[:, :, 2]**2
                ),
                axis=2
                )
        elif self.lattice_system == 'rhombohedral':
            self.hkl2 = np.stack((
                (self.hkl[:, :, 0]**2 + self.hkl[:, :, 1]**2 + self.hkl[:, :, 2]**2),
                (self.hkl[:, :, 0]*self.hkl[:, :, 1] + self.hkl[:, :, 0]*self.hkl[:, :, 2] + self.hkl[:, :, 1]*self.hkl[:, :, 2]),
                ),
                axis=2
                )

        q2_pred_init = self.get_q2_pred(xnn_init, jac=False)
        delta_q2 = np.abs(q2_pred_init - self.q2_obs)
        if power is None:
            self.sigma = np.sqrt(self.q2_obs * (delta_q2 + self.delta_q_eps))
        else:
            self.sigma = np.sqrt((self.q2_obs**power) * (delta_q2 + self.delta_q_eps))
        if not sigma_reduction is None:
            self.sigma *= sigma_reduction
        self.prefactor = np.log(np.sqrt(2*np.pi) * self.sigma)
        self.hessian_prefactor = (1 / self.sigma**2)[:, :, np.newaxis, np.newaxis]

    def get_q2_pred(self, xnn, jac=True):
        # self.hkl2:     n_entries, n_peaks, xnn_length
        # xnn:           n_entries, xnn_length
        # q2_pred:       n_entries, n_peaks
        # dq2_pred_dxnn: n_entries, n_peaks, xnn_length
        arg = self.hkl2 * xnn[:, np.newaxis, :]
        q2_pred = np.sum(arg, axis=2)
        if jac:
            dq2_pred_dxnn = self.hkl2
            return q2_pred, dq2_pred_dxnn
        else:
            return q2_pred

    def gauss_newton_step(self, xnn):
        # q2_pred:       n_entries, n_peaks
        # dq2_pred_dxnn: n_entries, n_peaks, xnn_length
        # self.q2_obs:   n_peaks
        #
        # The arithmetic here is unchanged and the results are bit-identical to
        # what this produced before; only the failure handling is different. Two
        # ways this used to break, both reachable with ill-conditioned candidates:
        #
        #   * np.linalg.matrix_rank was called *outside* the try below, and it
        #     raises on a non-finite Hessian, so a single bad sigma anywhere in
        #     the batch took the whole run down.
        #   * numpy's batched inv raises for the *entire* batch when any one
        #     member is singular. The except clause caught it, but delta_gn was
        #     then all zeros, so one degenerate candidate out of tens of thousands
        #     cost every candidate in the batch its refinement step.
        #
        # Both are now isolated to the individual candidate. A candidate that
        # cannot be refined gets a zero step, which is what this code always
        # intended for a non-invertible candidate.
        #
        # tools/repro_hessian.py holds a faster, far leaner alternative
        # (gauss_newton_solve in numba_functions: 9.3-9.6x, peak working memory
        # 224 MB -> 1.3 MB). It is not wired in here because it perturbs the
        # Gauss-Newton step by ~1e-8 relative to the cell, which this optimizer
        # amplifies into a reordering of the lower half of the candidate list.
        q2_pred, dq2_pred_dxnn = self.get_q2_pred(xnn, jac=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        dlikelihood_dq2_pred = residuals / self.sigma
        dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
        term0 = np.matmul(dq2_pred_dxnn[:, :, :, np.newaxis], dq2_pred_dxnn[:, :, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=1)

        # Need to ensure H is invertible before inverting.
        # isfinite rather than the old ~isnan: an infinite unit cell leaves H
        # finite, so it passed the rank test and the inversion and then emerged
        # as a non-finite step that was added straight into xnn. NaN and Inf
        # cells are both unusable, so both are screened here.
        invertible = np.isfinite(xnn).all(axis=1)
        # A non-finite Hessian is what makes matrix_rank raise, so screen for it
        # rather than letting the exception escape. On healthy input every H is
        # finite and this changes nothing.
        invertible &= np.isfinite(H).all(axis=(1, 2))
        # The gradient can go non-finite while H stays finite, by the same route.
        invertible &= np.isfinite(dloss_dxnn).all(axis=1)
        if np.any(invertible):
            try:
                invertible[invertible] = np.linalg.matrix_rank(
                    H[invertible], hermitian=True) == self.uc_length
            except np.linalg.LinAlgError:
                invertible = self._rank_test_one_at_a_time(H, invertible)

        delta_gn = np.zeros((self.n_entries, self.uc_length))
        if np.any(invertible):
            try:
                delta_gn[invertible] = -np.matmul(
                    np.linalg.inv(H[invertible]),
                    dloss_dxnn[invertible, :, np.newaxis]
                    )[:, :, 0]
            except np.linalg.LinAlgError:
                self._invert_one_at_a_time(H, dloss_dxnn, invertible, delta_gn)
        # Last line of defence. A step is added straight into xnn, so letting a
        # non-finite one through poisons that candidate for the rest of the run
        # and shows up much later as a NaN unit cell. Zeroing it leaves the
        # candidate where it was, which is the same outcome as failing the
        # invertibility test above.
        unusable = ~np.isfinite(delta_gn).all(axis=1)
        if np.any(unusable):
            delta_gn[unusable] = 0.0
        return delta_gn

    def _rank_test_one_at_a_time(self, H, invertible):
        """Fallback when a batched matrix_rank refuses the whole batch."""
        candidates = np.flatnonzero(invertible)
        keep = np.zeros(candidates.size, dtype=bool)
        for position, candidate in enumerate(candidates):
            try:
                keep[position] = np.linalg.matrix_rank(
                    H[candidate], hermitian=True) == self.uc_length
            except np.linalg.LinAlgError:
                keep[position] = False
        invertible[candidates] = keep
        return invertible

    def _invert_one_at_a_time(self, H, dloss_dxnn, invertible, delta_gn):
        """Fallback when a batched inv refuses the whole batch.

        Solving candidate by candidate costs more than the batched call, but it
        only runs when the batched call has already failed, and it means a single
        singular Hessian no longer discards every other candidate's step.
        """
        n_failed = 0
        for candidate in np.flatnonzero(invertible):
            try:
                delta_gn[candidate] = -np.matmul(
                    np.linalg.inv(H[candidate]), dloss_dxnn[candidate])
            except np.linalg.LinAlgError:
                n_failed += 1
        if n_failed:
            print(f'GAUSS-NEWTON: {n_failed} of {int(invertible.sum())} candidates '
                  f'could not be inverted; their steps are zero')

    def correct_zeropoint(self, xnn, wavelength):
        # q2_pred:       n_entries, n_peaks
        q2_pred = self.get_q2_pred(xnn, jac=False)
        weights = 1 / self.q2_obs
        theta2 = 2 * np.arcsin(wavelength/2 * np.sqrt(self.q2_obs))
        prefactor = wavelength**2 / (4 * np.sin(theta2))
        residuals = self.q2_obs - q2_pred
        zeropoint = np.sum(weights * prefactor * residuals, axis=1) / np.sum(weights, axis=1)
        return zeropoint

    def apply_zeropoint(self, zeropoint, wavelength, q2):
        theta2 = 2 * np.arcsin(np.clip(wavelength/2 * np.sqrt(np.abs(q2)), -1.0, 1.0))
        return q2 + 4*np.sin(theta2)/wavelength**2 * zeropoint[:, np.newaxis]

    def gauss_newton_step_zero_error(self, xnn, wavelength, zeropoint=None):
        # q2_pred:       n_entries, n_peaks
        # dq2_pred_dxnn: n_entries, n_peaks, xnn_length
        # self.q2_obs:   n_peaks
        # xnn:           n_entries, xnn_length
        # hkl2:          n_entries, n_peaks, xnn_length

        sigma = self.sigma
        
        theta2 = 2 * np.arcsin(wavelength/2 * np.sqrt(self.q2_obs))
        prefactor = 4 * np.sin(theta2) / wavelength**2
        hkl2 = np.concatenate([self.hkl2, prefactor[:, :, np.newaxis]], axis=2)
        if zeropoint is None:
            xnn = np.concatenate([xnn, np.zeros([self.n_entries, 1])], axis=1)
        else:
            xnn = np.concatenate([xnn, zeropoint[:, np.newaxis]], axis=1)

        arg = hkl2 * xnn[:, np.newaxis, :]
        q2_pred = np.sum(arg, axis=2)
        dq2_pred_dxnn = hkl2
            
        residuals = (q2_pred - self.q2_obs) / sigma
        dlikelihood_dq2_pred = residuals / sigma
        dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
        term0 = np.matmul(dq2_pred_dxnn[:, :, :, np.newaxis], dq2_pred_dxnn[:, :, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=1)
        # Need to ensure H is invertible before inverting.
        #invertible = np.linalg.det(H) != 0 # This is the fastest, but leaves non-invertible matrices.
        try:
            invertible = np.linalg.matrix_rank(H, hermitian=True) == (self.uc_length + 1)
        except:
            for index in range(H.shape[0]):
                try:
                    np.linalg.matrix_rank(H[index], hermitian=True)
                except:
                    print(xnn[index])
                    print(H[index])
            assert False
        #invertible = np.isfinite(np.linalg.cond(H)) # This is slow
        delta_gn = np.zeros((self.n_entries, self.uc_length + 1))
        try:
            delta_gn[invertible] = -np.matmul(
                np.linalg.inv(H[invertible]),
                dloss_dxnn[invertible, :, np.newaxis]
                )[:, :, 0]
        except np.linalg.LinAlgError as e:
            print(f'GAUSS-NEWTON INVERSION FAILED: {e}')
        return delta_gn

    def linear_least_squares(self):
        # Weighted linear least squares
        # Results are identical to the gauss newton step - not extensively tested though
        # np.linalg.lstsq default solver uses SVD
        # q2 = H @ xnn <- crystallography equation
        # b  = a @ x   <- np.linalg.lstsq equation
        xnn = np.zeros((self.n_entries, self.uc_length))
        for index in range(self.n_entries):
            xnn[index], residuals, rank, s = np.linalg.lstsq(
                self.hkl2[index] / self.sigma[index, :, np.newaxis],
                self.q2_obs[index] / self.sigma[index],
                rcond=None
                )
        return xnn

    def _get_hessian_inverse(self, xnn):
        # q2_pred:       n_entries, n_peaks
        # dq2_pred_dxnn: n_entries, n_peaks, xnn_length
        # self.q2_obs:   n_peaks
        q2_pred, dq2_pred_dxnn = self.get_q2_pred(xnn, jac=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        dlikelihood_dq2_pred = residuals / self.sigma
        dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
        term0 = np.matmul(dq2_pred_dxnn[:, :, :, np.newaxis], dq2_pred_dxnn[:, :, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=1)
        good = np.linalg.matrix_rank(H, hermitian=True) == self.uc_length
        delta_gn = np.zeros((self.n_entries, self.uc_length))
        H_inv = np.zeros(H.shape)
        H_inv[good] = np.linalg.inv(H[good])
        return H_inv

    def _get_hessian(self, xnn):
        # Helper for getting derivative verification
        q2_pred, dq2_pred_dxnn = self.get_q2_pred(xnn, jac=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        dlikelihood_dq2_pred = residuals / self.sigma
        dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
        term0 = np.matmul(dq2_pred_dxnn[:, :, :, np.newaxis], dq2_pred_dxnn[:, :, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=1)
        return H

    def _get_gradient(self, xnn):
        # Helper for getting derivative verification
        q2_pred, dq2_pred_dxnn = self.get_q2_pred(xnn, jac=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        dlikelihood_dq2_pred = residuals / self.sigma
        dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
        return dloss_dxnn

    def get_loss(self, xnn):
        q2_pred = self.get_q2_pred(xnn, jac=False)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        likelihood = self.prefactor + 1/2 * residuals**2
        loss = np.sum(likelihood, axis=1)
        return loss
