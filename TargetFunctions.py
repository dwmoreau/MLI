import numpy as np
from scipy.special import gamma
from sklearn.linear_model import LinearRegression
import tensorflow as tf


class LikelihoodLoss:
    def __init__(self, likelihood, n, beta_nll=None, tuning_param=None):
        """
        beta_nll: if None, use negative log likelihood target function
                  if not none, use beta-nll target function from Seitzer 2022
                  This has been found to speed up training
        tuning_param: degrees of freedom for the t-distribution likelihood
        self.prefactor is negative log of the normalization constant
        """
        self.n = n
        if likelihood == 'normal':
            self.likelihood_function = self.normal_likelihood
            self.prefactor = tf.cast(
                1/2 * tf.math.log(2 * np.pi),
                dtype=tf.float32,
                )
        elif likelihood == 't-dist':
            self.likelihood_function = self.t_dist_likelihood
            self.tuning_param = tuning_param
            v = self.tuning_param
            prefactor0 = tf.cast(
                -tf.math.lgamma((v + 1) / 2) / tf.math.lgamma(v / 2),
                dtype=tf.float32,
                )
            prefactor1 = tf.cast(
                1/2 * tf.math.log(v * np.pi),
                dtype=tf.float32,
                )
            self.prefactor = prefactor0 + prefactor1
        elif likelihood == 'alpha_beta':
            self.likelihood_function = self.alpha_beta_likelihood
            self.prefactor = tf.cast(
                1/2 * tf.math.log(2 * np.pi),
                dtype=tf.float32,
                )
        elif likelihood == 'mean_absolute_error':
            self.likelihood_function = self.mean_absolute_error
        elif likelihood == 'mean_squared_error':
            self.likelihood_function = self.mean_squared_error

        if beta_nll is None:
            self.beta_likelihood = False
        else:
            self.beta_likelihood = True
            self.beta_nll = beta_nll

    def __call__(self, y_true, y_pred):
        return self.likelihood_function(y_true, y_pred)

    def alpha_beta_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        alpha = y_pred[:, :, 1]
        beta = y_pred[:, :, 2]
        term0 = -alpha * tf.math.log(beta)
        term1 = -tf.math.lgamma(alpha + 1/2)
        term2 = tf.math.lgamma(alpha)
        term3 = (alpha + 1/2) * tf.math.log(beta + 1/2 * (y_true - mean)**2)
        likelihoods = self.prefactor + term0 + term1 + term2 + term3
        if self.beta_likelihood:
            var = beta / (alpha - 1)
            likelihoods = likelihoods * tf.stop_gradient(var**self.beta_nll)
        return tf.reduce_sum(likelihoods, axis=1)

    def normal_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = tf.math.log(var)
        term1 = 0.5 * (y_true - mean)**2 / var
        # likelihoods: n_batch x n_lattice_params
        likelihoods = self.prefactor + term0 + term1
        if self.beta_likelihood:
            likelihoods = likelihoods * tf.stop_gradient(var**self.beta_nll)
        return tf.reduce_sum(likelihoods, axis=1)

    def cauchy_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = tf.math.log(var)
        z2 = (y_true - mean)**2 / var
        term1 = tf.math.log(1 + z2)
        likelihoods = term0 + term1 + self.prefactor
        if self.beta_likelihood:
            likelihoods = likelihoods * tf.stop_gradient(var**self.beta_nll)
        return tf.reduce_sum(likelihoods, axis=1)

    def t_dist_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = tf.math.log(var)
        z2 = (y_true - mean)**2 / var
        term1 = (self.tuning_param + 1) / 2 * tf.math.log(1 + 1/self.tuning_param * z2)
        likelihoods = term0 + term1 + self.prefactor
        if self.beta_likelihood:
            likelihoods = likelihoods * tf.stop_gradient(var**self.beta_nll)
        return tf.reduce_sum(likelihoods, axis=1)

    def mean_squared_error(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        square_difference = (y_true - mean)**2
        return tf.reduce_mean(square_difference, axis=1)

    def mean_absolute_error(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        absolute_difference = tf.math.abs(y_true - mean)
        return tf.reduce_mean(absolute_difference, axis=1)


class CandidateOptLoss_xnn:
    def __init__(self, q2_obs, lattice_system):
        self.q2_obs = q2_obs
        self.lattice_system = lattice_system
        self.delta_q_eps = 1e-10
        #self.delta_q_eps = np.exp(-10)
        self.n_points = self.q2_obs.shape[1]
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

    def update(self, hkl, softmax, xnn_init):
        self.hkl = hkl
        self.softmax = softmax
        if self.lattice_system == 'cubic':
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
        elif self.lattice_system == 'monoclinic':
            self.hkl2 = np.concatenate((
                self.hkl**2, 
                (self.hkl[:, :, 0] * self.hkl[:, :, 2])[:, :, np.newaxis]
                ),
                axis=2
                )
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
        else:
            assert False

        q2_pred_init = self.get_q2_pred(xnn_init, jac=False)
        delta_q2 = np.abs(q2_pred_init - self.q2_obs)
        self.sigma = np.sqrt(self.q2_obs * (delta_q2 + self.delta_q_eps))

        self.prefactor = np.log(np.sqrt(2*np.pi) * self.sigma) - np.log(self.softmax)
        self.hessian_prefactor = (1 / self.sigma**2)[:, :, np.newaxis, np.newaxis]

    def get_q2_pred(self, xnn, jac=True):
        # self.hkl2:     n_entries, n_points, xnn_length
        # xnn:           n_entries, xnn_length
        # q2_pred:       n_entries, n_points
        # dq2_pred_dxnn: n_entries, n_points, xnn_length
        arg = self.hkl2 * xnn[:, np.newaxis, :]
        q2_pred = np.sum(arg, axis=2)
        if jac:
            dq2_pred_dxnn = self.hkl2
            return q2_pred, dq2_pred_dxnn
        else:
            return q2_pred

    def gauss_newton_step(self, xnn):
        # q2_pred:       n_entries, n_points
        # dq2_pred_dxnn: n_entries, n_points, xnn_length
        # self.q2_obs:   n_points
        q2_pred, dq2_pred_dxnn = self.get_q2_pred(xnn, jac=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        dlikelihood_dq2_pred = residuals / self.sigma
        dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
        term0 = np.matmul(dq2_pred_dxnn[:, :, :, np.newaxis], dq2_pred_dxnn[:, :, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=1)
        good = np.linalg.matrix_rank(H, hermitian=True) == self.uc_length
        delta_gn = np.zeros((self.n_entries, self.uc_length))
        delta_gn[good] = -np.matmul(np.linalg.inv(H[good]), dloss_dxnn[good, :, np.newaxis])[:, :, 0]
        return delta_gn

    def _get_hessian_inverse(self, xnn):
        # q2_pred:       n_entries, n_points
        # dq2_pred_dxnn: n_entries, n_points, xnn_length
        # self.q2_obs:   n_points
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


class CandidateOptLoss:
    def __init__(self, q2_obs, lattice_system):
        self.q2_obs = q2_obs
        self.lattice_system = lattice_system
        self.delta_q_eps = 1e-10
        #self.delta_q_eps = np.exp(-10)
        self.n_points = q2_obs.size

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

        self.__diag_indices = np.arange(self.uc_length)
        self.__term1 = np.zeros((self.n_points, self.uc_length, self.uc_length))

    def update(self, hkl, softmax, uc_init):
        self.hkl = hkl
        self.softmax = softmax
        if self.lattice_system == 'cubic':
            self.hkl2 = (self.hkl[:, 0]**2 + self.hkl[:, 1]**2 + self.hkl[:, 2]**2)[:, np.newaxis]
        elif self.lattice_system == 'tetragonal':
            self.hkl2 = np.column_stack((
                self.hkl[:, 0]**2 + self.hkl[:, 1]**2,
                self.hkl[:, 2]**2
                ))
        elif self.lattice_system == 'orthorhombic':
            self.hkl2 = self.hkl**2
        elif self.lattice_system == 'monoclinic':
            self.hkl2 = np.concatenate((
                self.hkl**2,
                (self.hkl[:, :, 0] * self.hkl[:, :, 2])[:, :, np.newaxis]
                ),
                axis=2
                )
        elif self.lattice_system == 'hexagonal':
            self.hkl2 = np.column_stack((
                4/3 * (self.hkl[:, 0]**2 + self.hkl[:, 0]*self.hkl[:, 1] + self.hkl[:, 1]**2),
                self.hkl[:, 2]**2
                ))

        if self.lattice_system in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            uc_inv2_init = 1 / uc_init**2
        elif self.lattice_system == 'monoclinic':
            uc_inv2_init = np.zeros(4)
            sin_beta = np.sin(uc_init[3])
            cos_beta = np.cos(uc_init[3])
            uc_inv2_init[0] = 1 / (uc_init[0] * sin_beta)**2
            uc_inv2_init[1] = 1 / uc_init[1]**2
            uc_inv2_init[2] = 1 / (uc_init[2] * sin_beta)**2
            uc_inv2_init[3] = -2 * cos_beta / (uc_init[0] * uc_init[2] * sin_beta**2)

        q2_pred_init, _ = self.get_q2_pred(uc_inv2_init)
        delta_q2 = np.abs(q2_pred_init - self.q2_obs)
        self.sigma = np.sqrt(self.q2_obs * (delta_q2 + self.delta_q_eps))

        self.prefactor = np.log(np.sqrt(2*np.pi) * self.sigma) - np.log(self.softmax)
        self.hessian_prefactor = (1 / self.sigma**2)[:, np.newaxis, np.newaxis]

        return uc_inv2_init

    def get_optimized_uc(self, uc_inv2):
        if self.lattice_system in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            uc = 1 / np.sqrt(np.abs(uc_inv2))
        elif self.lattice_system == 'monoclinic':
            a_sin_beta2 = 1 / np.abs(uc_inv2[0])
            b = 1 / np.sqrt(np.abs(uc_inv2[1]))
            c_sin_beta2 = 1 / np.abs(uc_inv2[2])
            aoc = np.sqrt(a_sin_beta2 / c_sin_beta2)
            aoc_cos_beta = -1/2 * uc_inv2[3] * a_sin_beta2
            cos_beta = aoc_cos_beta / aoc
            if np.abs(cos_beta) >= 1:
                # If cos_beta >= 1, then the monoclinic angle is not physically possible
                # and a, c, & beta will be calculated as np.nan.
                # Returning a very large unit cell will prevent nan errors downstream.
                # The candidate unit cell should be identifiable by this unit cell or large loss
                return np.array([1000, 1000, 1000, np.pi/2])
            else:
                beta = np.arccos(cos_beta)
                sin_beta = np.sin(beta)
                a = np.sqrt(a_sin_beta2) / sin_beta
                c = np.sqrt(c_sin_beta2) / sin_beta
                uc = np.array([a, b, c, beta])
        return uc

    def get_q2_pred(self, uc_inv2, jac=True):
        """
        cubic:
            uc_inv2 = 1 / a**2
            self.hkl2 = h**2 + k**2 + l**2
        tetragonal:
            uc_inv2[0] = 1 / a**2
            uc_inv2[1] = 1 / c**2
            self.hkl2 = h**2 + k**2, l**2
        orthorhombic:
            uc_inv2[0] = 1 / a**2
            uc_inv2[1] = 1 / b**2
            uc_inv2[2] = 1 / c**2
            self.hkl2 = h**2, k**2, l**2
        monoclinic:
            uc_inv2[0] = 1 / (a * sin(beta))**2
            uc_inv2[1] = 1 / b**2
            uc_inv2[2] = 1 / (c * sin(beta))**2
            uc_inv2[3] = -2 * cos(beta) / (a * c * sin(beta)**2)
            self.hkl2 = h**2, k**2, l**2, h*l
        hexagonal:
            uc_inv2[0] = 1 / a**2
            uc_inv2[1] = 1 / c**2
            self.hkl2 = 4/3 * (h**2 + h*k + k**2), l**2
        """
        arg = self.hkl2 * uc_inv2[np.newaxis, :]
        q2_pred = np.sum(arg, axis=1)
        if jac:
            dq2_pred_duc_inv2 = self.hkl2
            return q2_pred, dq2_pred_duc_inv2
        else:
            return q2_pred

    def least_squares(self):
        LS = LinearRegression(fit_intercept=False)
        LS.fit(self.hkl2, self.q2_obs, sample_weight=1/self.sigma**2)
        return self.get_optimized_uc(LS.coef_)

    def least_squares_svd(self):
        #https://sthalles.github.io/svd-for-regression/
        U,S,Vt = np.linalg.svd(self.hkl2, full_matrices=False)
        x_hat = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ self.q2_obs
        q2_pred = self.hkl2 @ x_hat
        print(x_hat)
        print(q2_pred)
        return self.get_optimized_uc(x_hat)

    def loss_likelihood(self, uc_inv2):
        q2_pred, dq2_pred_duc_inv2 = self.get_q2_pred(uc_inv2)

        residuals = (q2_pred - self.q2_obs) / self.sigma
        likelihood = self.prefactor + 1/2 * residuals**2

        dlikelihood_dq2_pred = residuals / self.sigma

        loss = np.sum(likelihood)
        dloss_duc_inv2 = np.sum(dlikelihood_dq2_pred[:, np.newaxis] * dq2_pred_duc_inv2, axis=0)
        return loss, dloss_duc_inv2

    def loss_likelihood_hessian(self, uc_inv2):
        q2_pred, dq2_pred_duc_inv2 = self.get_q2_pred(uc_inv2, jac=True)
        term0 = np.matmul(dq2_pred_duc_inv2[:, :, np.newaxis], dq2_pred_duc_inv2[:, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=0)
        return H

    def loss_likelihood_no_jac(self, uc_inv2):
        q2_pred = self.get_q2_pred(uc_inv2, jac=False)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        likelihood = self.prefactor + 1/2 * residuals**2
        loss = np.sum(likelihood)
        return loss, q2_pred

    def get_loss(self, uc_inv2):
        loss, q2_pred = self.loss_likelihood_no_jac(uc_inv2)
        return loss


class IndexingTargetFunction:
    def __init__(self, likelihood_function, error_fraction, n_points, tuning_param=1):
        self.error_fraction = error_fraction
        self.tuning_param = tuning_param
        prefactor0 = gamma((self.tuning_param + 1) / 2) / gamma(self.tuning_param / 2)
        prefactor1 = 1 / tf.math.sqrt(self.tuning_param * np.pi)
        self.prefactor = prefactor0 * prefactor1
        self.exponent = -(self.tuning_param + 1) / 2
        self.n_points = n_points

    def __call__(self, q2_true, y_pred):
        # softmaxes: n_batch x n_points x hkl_ref_length
        # q2_ref:    n_batch x hkl_ref_length            <- not transform!!!
        # q2_true:   n_batch x n_points                  <- not transform!!!
        # q2_error:  n_batch x n_points
        softmaxes = y_pred[:, :self.n_points, :]
        q2_ref = y_pred[:, self.n_points, :]
        q2_error = q2_true * self.error_fraction[tf.newaxis, :]
        
        # differences: n_batch x n_points x hkl_ref_length
        differences = q2_true[:, :, tf.newaxis] - q2_ref[:, tf.newaxis, :]
        residuals = differences / q2_error[:, :, np.newaxis]

        arg = 1 + 1/self.tuning_param * residuals**2
        difference_likelihoods = self.prefactor / q2_error[:, :, tf.newaxis] * arg**self.exponent
        # peak_likelihoods: n_batch x n_points x hkl_ref_length
        all_peaks_likelihoods = difference_likelihoods * softmaxes
        # peak_likelihoods: n_batch x n_points
        peak_log_likelihoods = tf.math.log(tf.math.reduce_sum(all_peaks_likelihoods, axis=2))
        neg_log_likelihood = -tf.math.reduce_sum(peak_log_likelihoods, axis=1)
        return neg_log_likelihood
