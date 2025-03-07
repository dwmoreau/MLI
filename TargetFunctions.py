import numpy as np
import scipy.special
from sklearn.linear_model import LinearRegression
import keras


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
            self.prefactor = keras.ops.cast(
                1/2 * keras.ops.log(2 * np.pi),
                dtype='float32',
                )
        elif likelihood == 't-dist':
            self.likelihood_function = self.t_dist_likelihood
            self.tuning_param = tuning_param
            v = self.tuning_param
            prefactor0 = keras.ops.cast(
                -scipy.special.gammaln((v + 1) / 2) / scipy.special.gammaln(v / 2),
                dtype='float32',
                )
            prefactor1 = keras.ops.cast(
                1/2 * keras.ops.log(v * np.pi),
                dtype='float32',
                )
            self.prefactor = prefactor0 + prefactor1
        elif likelihood == 'alpha_beta':
            self.likelihood_function = self.alpha_beta_likelihood
            self.prefactor = keras.ops.cast(
                1/2 * keras.ops.log(2 * np.pi),
                dtype='float32',
                )
        elif likelihood == 'mean_absolute_error':
            self.likelihood_function = self.mean_absolute_error
        elif likelihood == 'mean_squared_error':
            self.likelihood_function = self.mean_squared_error
        elif likelihood == 'log_cosh':
            self.likelihood_function = self.log_cosh_likelihood

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
        term0 = -alpha * keras.ops.log(beta)
        # lgamma is the log of the absolute value of the gamma function
        # Keras does not provide a backend agnostic gamma function
        # These terms are unnecessary anyways...
        #term1 = -tf.math.lgamma(alpha + 1/2)
        #term2 = tf.math.lgamma(alpha)
        term3 = (alpha + 1/2) * keras.ops.log(beta + 1/2 * (y_true - mean)**2)
        likelihoods = self.prefactor + term0 + term3 #+ term1 + term2
        if self.beta_likelihood:
            var = beta / (alpha - 1)
            likelihoods = likelihoods * keras.ops.stop_gradient(var**self.beta_nll)
        return keras.ops.sum(likelihoods, axis=1)

    def normal_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = 1/2 * keras.ops.log(var)
        term1 = 0.5 * (y_true - mean)**2 / var
        # likelihoods: n_batch x n_lattice_params
        likelihoods = self.prefactor + term0 + term1
        if self.beta_likelihood:
            likelihoods = likelihoods * keras.ops.stop_gradient(var**self.beta_nll)
        return keras.ops.sum(likelihoods, axis=1)

    def cauchy_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = keras.ops.log(var)
        z2 = (y_true - mean)**2 / var
        term1 = keras.ops.log(1 + z2)
        likelihoods = term0 + term1 + self.prefactor
        if self.beta_likelihood:
            likelihoods = likelihoods * keras.ops.stop_gradient(var**self.beta_nll)
        return keras.ops.sum(likelihoods, axis=1)

    def t_dist_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = keras.ops.log(var)
        z2 = (y_true - mean)**2 / var
        term1 = (self.tuning_param + 1) / 2 * keras.ops.log(1 + 1/self.tuning_param * z2)
        likelihoods = term0 + term1 + self.prefactor
        if self.beta_likelihood:
            likelihoods = likelihoods * keras.ops.stop_gradient(var**self.beta_nll)
        return keras.ops.sum(likelihoods, axis=1)

    def log_cosh_likelihood(self, y_true, y_pred):
        # This is not a likelihood function. The "likelihood" distinguishes this
        # call that uses variance from the "error" call that does not.
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        error = (y_true - mean) / keras.ops.sqrt(var)

        # This is to prevent an overflow error
        # keras.ops.cosh has a limit around +/- 80
        error = keras.ops.clip(error, -75.0, 75.0)
        log_cosh = keras.ops.log(keras.ops.cosh(error))

        # I don't know how to incorporate the beta_nll parameter here because this is not a likelihood.
        # If I had to guess, it might look like this:
        # if self.beta_likelihood:
        #   error = (y_true - mean) / keras.ops.sqrt(var) * keras.ops.stop_gradient(keras.ops.sqrt(var**self.beta_nll))
        # else:
        #   error = (y_true - mean) / keras.ops.sqrt(var)
        # or it might look like this:
        # if self.beta_likelihood:
        #   log_cosh = log_cosh * keras.ops.stop_gradient(var**self.beta_nll)
        return keras.ops.mean(log_cosh, axis=1)

    def mean_squared_error(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        square_difference = (y_true - mean)**2
        return keras.ops.mean(square_difference, axis=1)

    def log_cosh_error(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        error = (y_true - mean)
        error = keras.ops.clip(error, -75.0, 75.0)
        log_cosh = keras.ops.log(keras.ops.cosh(error))
        return keras.ops.mean(log_cosh, axis=1)

    def mean_absolute_error(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        absolute_difference = keras.ops.absolute(y_true - mean)
        return keras.ops.mean(absolute_difference, axis=1)


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
        q2_pred, dq2_pred_dxnn = self.get_q2_pred(xnn, jac=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        dlikelihood_dq2_pred = residuals / self.sigma
        dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
        term0 = np.matmul(dq2_pred_dxnn[:, :, :, np.newaxis], dq2_pred_dxnn[:, :, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=1)
        # Need to ensure H is invertible before inverting.
        invertible = np.linalg.det(H) != 0 # This is the fastest
        #invertible = np.linalg.matrix_rank(H, hermitian=True) == self.uc_length
        #invertible = np.isfinite(np.linalg.cond(H)) # This is slow
        delta_gn = np.zeros((self.n_entries, self.uc_length))
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
