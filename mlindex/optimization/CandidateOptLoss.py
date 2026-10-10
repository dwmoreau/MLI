import numpy as np

from mlindex.utilities.numba_functions import gauss_newton_solve


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

    # A Cholesky pivot at or below this fraction of the Hessian's largest
    # diagonal marks a candidate unrefinable. It replaces the eigenvalue
    # tolerance the old np.linalg.matrix_rank test used; on captured runs the two
    # classifications disagreed on 1 candidate in 39,753, and on that one both
    # produced a step below 1e-8 of the cell, so the value is not delicate.
    pivot_tolerance = 1e-12

    def gauss_newton_step(self, xnn):
        """One Gauss-Newton step per candidate, solved independently.

        gauss_newton_solve builds each candidate's k x k Hessian in a reused
        buffer, factorises it with a tolerance-guarded Cholesky and solves in
        place. It replaces a four-step numpy pipeline that built an
        (n_entries, n_peaks, k, k) intermediate, tested invertibility with a full
        eigendecomposition, then inverted.

        Robustness was the reason for writing it, not speed. The numpy version
        called np.linalg.matrix_rank outside its try block and matrix_rank raises
        on a non-finite Hessian, so one bad sigma ended the run; and numpy's
        batched inv raises for the *entire* batch when any single member is
        singular, which left every candidate with a zero step. There is no
        batched LAPACK call here to fail: a candidate that cannot be refined gets
        a zero step and its neighbours are untouched.

        Measured against captured inputs (tools/repro_hessian.py): 9.3-9.6x on
        this function, 8.6% end to end, and peak working memory 224 MB -> 1.3 MB
        on a 39,753-candidate chunk. The intermediate it removes would be ~549 MB
        at 100,000 triclinic candidates.

        This is not bit-identical to the numpy version. A different summation
        order perturbs each step by ~1e-8 relative to the cell, and roughly sixty
        iterations of a randomised optimizer amplify that into a reordered
        candidate list. Validated across the seventeen known-answer cases in
        mlindex/data/test_data (tools/validate_true_cell.py): the true cell is
        found in 16/17 under both implementations with identical rank-1, top-5,
        top-20 and median-rank counts. Its M20 does move in 7 of those 16 cases,
        by 1e-13 to 4e-3 relative; the two largest movements are improvements and
        the largest degradation is 4e-5.

        ``ok`` is discarded here because a skipped candidate already has the zero
        step this function has always given a non-invertible one, but the kernel
        returns it so callers can count skips if that is ever wanted.
        """
        delta_gn, _ok = gauss_newton_solve(
            np.ascontiguousarray(self.hkl2),
            np.ascontiguousarray(self.q2_obs),
            np.ascontiguousarray(self.sigma),
            np.ascontiguousarray(xnn),
            self.pivot_tolerance,
            )
        return delta_gn

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
        """Gauss-Newton step refining the zero-point offset alongside the cell.

        Same normal equations as gauss_newton_step, on a system augmented by one
        column: the zero-point enters q2 linearly through
        4 sin(2 theta) / lambda^2, so appending that to hkl2 and the current
        zero-point to xnn makes it just another parameter. That is why this can
        reuse the same kernel -- the returned step is (n_entries, uc_length + 1),
        with the zero-point correction in the final column.

        What this replaces mattered more than the arithmetic. The invertibility
        test was wrapped in a bare ``except:`` that printed the offending
        candidate and then hit ``assert False``, so a single non-finite Hessian
        ended the run by construction -- and a bare except also swallows
        KeyboardInterrupt. np.linalg.matrix_rank raises on exactly that input,
        which is reachable here: when wavelength/2 * sqrt(q2_obs) exceeds 1 the
        arcsin below is NaN and every entry of that candidate's Hessian follows.
        The kernel skips such a candidate and leaves its step at zero, the same
        outcome the code intended for a non-invertible one, without touching its
        neighbours.

        Note that apply_zeropoint clips the arcsin argument to [-1, 1] and this
        does not. Adding a clip here would change results for peaks beyond the
        wavelength's reachable range rather than dropping those candidates, so it
        is left as it was; the kernel means the difference is now a skipped
        candidate rather than a dead process.
        """
        theta2 = 2 * np.arcsin(wavelength/2 * np.sqrt(self.q2_obs))
        prefactor = 4 * np.sin(theta2) / wavelength**2
        hkl2 = np.concatenate([self.hkl2, prefactor[:, :, np.newaxis]], axis=2)
        if zeropoint is None:
            xnn = np.concatenate([xnn, np.zeros([self.n_entries, 1])], axis=1)
        else:
            xnn = np.concatenate([xnn, zeropoint[:, np.newaxis]], axis=1)

        delta_gn, _ok = gauss_newton_solve(
            np.ascontiguousarray(hkl2),
            np.ascontiguousarray(self.q2_obs),
            np.ascontiguousarray(self.sigma),
            np.ascontiguousarray(xnn),
            self.pivot_tolerance,
            )
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
