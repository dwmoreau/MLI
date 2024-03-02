import numpy as np
from scipy.special import gamma
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
            #var = alpha / beta
            var = beta / (alpha - 1)
            likelihoods = likelihoods * tf.stop_gradient(var)**self.beta_nll
        return tf.reduce_sum(likelihoods, axis=1)

    def normal_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = tf.math.log(var)
        term1 = 0.5 * (y_true - mean)**2 / var
        # likelihoods: n_batch x n_lattice_params
        likelihoods = self.prefactor + term0 + term1
        if self.beta_likelihood:
            likelihoods = likelihoods * tf.stop_gradient(var)**self.beta_nll
        return tf.reduce_sum(likelihoods, axis=1)

    def cauchy_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = tf.math.log(var)
        z2 = (y_true - mean)**2 / var
        term1 = tf.math.log(1 + z2)
        likelihoods = term0 + term1 + self.prefactor
        if self.beta_likelihood:
            likelihoods = likelihoods * tf.stop_gradient(var)**self.beta_nll
        return tf.reduce_sum(likelihoods, axis=1)

    def t_dist_likelihood(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        var = y_pred[:, :, 1]
        term0 = tf.math.log(var)
        z2 = (y_true - mean)**2 / var
        term1 = (self.tuning_param + 1) / 2 * tf.math.log(1 + 1/self.tuning_param * z2)
        likelihoods = term0 + term1 + self.prefactor
        if self.beta_likelihood:
            likelihoods = likelihoods * tf.stop_gradient(var)**self.beta_nll
        return tf.reduce_sum(likelihoods, axis=1)

    def mean_squared_error(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        square_difference = (y_true - mean)**2
        return tf.reduce_mean(square_difference, axis=1)

    def mean_absolute_error(self, y_true, y_pred):
        mean = y_pred[:, :, 0]
        absolute_difference = tf.math.abs(y_true - mean)
        return tf.reduce_mean(absolute_difference, axis=1)


class CandidateOptLoss:
    def __init__(self, q2_obs, lattice_system, likelihood, tuning_param=None):
        self.q2_obs = q2_obs
        self.likelihood = likelihood
        self.lattice_system = lattice_system
        self.delta_q_eps = 1e-10
        self.n_points = q2_obs.size

        if lattice_system == 'cubic':
            self.uc_length = 1
        elif lattice_system == 'tetragonal':
            self.uc_length = 2
        elif lattice_system == 'orthorhombic':
            self.uc_length = 3
        elif lattice_system == 'monoclinic':
            self.get_q2_pred = self.get_q2_pred_monoclinic
            self.uc_length = 4
        elif lattice_system == 'triclinic':
            self.get_q2_pred = self.get_q2_pred_triclinic
            self.uc_length = 6
        elif lattice_system == 'hexagonal':
            self.get_q2_pred = self.get_q2_pred_hexagonal
            self.uc_length = 2
        elif lattice_system == 'rhombohedral':
            self.get_q2_pred = self.get_q2_pred_rhombohedral
            self.uc_length = 2

        if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
            self.get_q2_pred = self.get_q2_pred_cubic_tetragonal_orthorhombic
            self.__diag_indices = np.arange(self.uc_length)
            self.__term1 = np.zeros((self.n_points, self.uc_length, self.uc_length))

        if self.likelihood == 't_dist':
            self.tuning_param = tuning_param
            self.exponent = -(self.tuning_param + 1) / 2
            self.loss_likelihood = self.loss_likelihood_t
            self.loss_likelihood_no_jac = self.loss_likelihood_t_no_jac
            if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
                self.loss_likelihood_hessian = self.loss_likelihood_t_hessian_diagonal
        elif self.likelihood == 'normal':
            self.loss_likelihood = self.loss_likelihood_normal
            self.loss_likelihood_no_jac = self.loss_likelihood_normal_no_jac
            if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
                self.loss_likelihood_hessian = self.loss_likelihood_normal_hessian_diagonal

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
        elif self.lattice_system == 'hexagonal':
            self.hk_term = self.hkl[:, 0]**2 + self.hkl[:, 0]*self.hkl[:, 1] + self.hkl[:, 1]**2
        elif self.lattice_system == 'rhombohedral':
            self.hkl2_term = self.hkl[:, 0]**2 + self.hkl[:, 1]**2 + self.hkl[:, 2]**2
            self.hkl_term = self.hkl[:, 0]*self.hkl[:, 1] + self.hkl[:, 0]*self.hkl[:, 2] + self.hkl[:, 1]*self.hkl[:, 2]

        q2_pred_init, _ = self.get_q2_pred(uc_init)
        delta_q = np.abs(np.sqrt(q2_pred_init) - np.sqrt(self.q2_obs))
        self.sigma = np.sqrt(self.q2_obs * (delta_q + self.delta_q_eps))

        if self.likelihood == 't_dist':
            prefactor1 = gamma((self.tuning_param + 1) / 2) / gamma(self.tuning_param / 2)
            prefactor2 = 1 / np.sqrt(self.tuning_param * np.pi)
            prefactor0 = prefactor1 * prefactor2 / self.sigma
            self.prefactor = prefactor0 * self.softmax
        elif self.likelihood == 'normal':
            self.prefactor = np.log(np.sqrt(2*np.pi) * self.sigma) - np.log(self.softmax)
            self.hessian_prefactor = (1 / self.sigma**2)[:, np.newaxis, np.newaxis]

    def get_q2_pred_cubic_tetragonal_orthorhombic(self, uc, jac=True, hessian=False):
        arg = self.hkl2 / uc[np.newaxis, :]**2
        q2_pred = np.sum(arg, axis=1)
        if jac:
            dq2_pred_duc = -2 * arg / uc[np.newaxis, :]
        if hessian:
            d2q2_pred_duc2 = 6 * arg / uc[np.newaxis, :]**2
        if jac and not hessian:
            return q2_pred, dq2_pred_duc
        elif not jac and hessian:
            return q2_pred, d2q2_pred_duc2
        elif jac and hessian:
            return q2_pred, dq2_pred_duc, d2q2_pred_duc2
        elif not jac and not hessian:
            return q2_pred

    def get_q2_pred_hexagonal(self, uc):
        q2_pred = 4/3 * self.hk_term / uc[0]**2 + self.hkl[:, 2]**2 / uc[1]**2
        dq2_pred_duc = np.column_stack((
            -8/3 * self.hk_term / uc[0]**3,
            -2 * self.hkl[:, 2]**2 / uc[1]**3,
            ))
        return q2_pred, dq2_pred_duc

    def get_q2_pred_rhombohedral(self, uc):
        sin_alpha = np.sin(uc[1])
        cos_alpha = np.cos(uc[1])

        numer = self.hkl2_term * sin_alpha**2 + 2*self.hkl_term * (cos_alpha**2 - cos_alpha)
        denom = uc[0]**2 * (1 + 2*cos_alpha**3 - 3*cos_alpha**2)
        q2_pred = numer / denom

        dnumer_dalpha = 2*self.hkl2_term * sin_alpha * cos_alpha + 2*self.hkl_term * (sin_alpha - 2*cos_alpha*sin_alpha)
        ddenom_dalpha = uc[0]**2 * (6*cos_alpha*sin_alpha - 6*cos_alpha**2*sin_alpha)
        ddenom_da = 2*uc[0] * (1 + 2*cos_alpha**3 - 3*cos_alpha**2)
        dq2_pred_duc = np.column_stack((
            -1 * numer / denom**2 * ddenom_da,
            dnumer_dalpha/denom - numer / denom**2 * ddenom_dalpha,
            ))
        return q2_pred, dq2_pred_duc

    def get_q2_pred_triclinic(self, uc):
        a = uc[0]
        b = uc[1]
        c = uc[2]

        calpha = np.cos(uc[3])
        dcalpha_dalpha = -np.sin(uc[3])
        salpha = np.sin(uc[3])
        dsalpha_dalpha = np.cos(uc[3])

        cbeta = np.cos(uc[4])
        dcbeta_dbeta = -np.sin(uc[4])
        sbeta = np.sin(uc[4])
        dsbeta_dbeta = np.cos(uc[4])

        cgamma = np.cos(uc[5])
        dcgamma_dgamma = -np.sin(uc[5])
        sgamma = np.sin(uc[5])
        dsgamma_dgamma = np.cos(uc[5])

        denom = 1 + 2*calpha*cbeta*cgamma - calpha**2 - cbeta**2 - cgamma**2
        ddenom_dalpha = (2*cbeta*cgamma - 2*calpha) * dcalpha_dalpha
        ddenom_dbeta = (2*calpha*cgamma - 2*cbeta) * dcbeta_dbeta
        ddenom_dgamma = (2*calpha*cbeta - 2*cgamma) * dcgamma_dgamma

        term0 = self.hkl[:, 0]**2 * salpha**2 / a**2
        dterm0_da = -2 * self.hkl[:, 0]**2 * salpha**2 / a**3
        dterm0_dsalpha = 2*self.hkl[:, 0]**2 * salpha / a**2
        term1 = self.hkl[:, 1]**2 * sbeta**2 / b**2
        dterm1_db = -2 * self.hkl[:, 1]**2 * sbeta**2 / b**3
        dterm1_dsbeta = 2*self.hkl[:, 1]**2 * sbeta / b**2
        term2 = self.hkl[:, 2]**2 * sgamma**2 / c**2
        dterm2_dc = -2 * self.hkl[:, 2]**2 * sgamma**2 / c**3
        dterm2_dsgamma = 2*self.hkl[:, 2]**2 * sgamma / c**2

        term3a = 2*self.hkl[:, 0]*self.hkl[:, 1] / (a*b)
        dterm3a_da = -2*self.hkl[:, 0]*self.hkl[:, 1] / (a**2*b)
        dterm3a_db = -2*self.hkl[:, 0]*self.hkl[:, 1] / (a*b**2)
        term3b = calpha*cbeta - cgamma
        dterm3b_dcalpha = cbeta
        dterm3b_dcbeta = calpha
        dterm3b_dcgamma = -1
        term3 = term3a * term3b
        dterm3_da = dterm3a_da * term3b
        dterm3_db = dterm3a_db * term3b
        dterm3_dcalpha = term3a * dterm3b_dcalpha
        dterm3_dcbeta = term3a * dterm3b_dcbeta
        dterm3_dcgamma = term3a * dterm3b_dcgamma

        term4a = 2*self.hkl[:, 1]*self.hkl[:, 2] / (b*c)
        dterm4a_db = -2*self.hkl[:, 1]*self.hkl[:, 2] / (b**2*c)
        dterm4a_dc = -2*self.hkl[:, 1]*self.hkl[:, 2] / (b*c**2)
        term4b = cbeta*cgamma - calpha
        term4 = term4a * term4b
        dterm4b_dcalpha = -1
        dterm4b_dcbeta = cgamma
        dterm4b_dcgamma = cbeta
        dterm4_db = dterm4a_db * term4b
        dterm4_dc = dterm4a_dc * term4b
        dterm4_dcalpha = term4a * dterm4b_dcalpha
        dterm4_dcbeta = term4a * dterm4b_dcbeta
        dterm4_dcgamma = term4a * dterm4b_dcgamma

        term5a = 2*self.hkl[:, 0]*self.hkl[:, 2] / (a*c)
        dterm5a_da = -2*self.hkl[:, 0]*self.hkl[:, 2] / (a**2*c)
        dterm5a_dc = -2*self.hkl[:, 0]*self.hkl[:, 2] / (a*c**2)
        term5b = calpha*cgamma - cbeta
        term5 = term5a * term5b
        dterm5b_dcalpha = cgamma
        dterm5b_dcbeta = -1
        dterm5b_dcgamma = calpha
        dterm5_da = dterm5a_da * term5b
        dterm5_dc = dterm5a_dc * term5b
        dterm5_dcalpha = term5a * dterm5b_dcalpha
        dterm5_dcbeta = term5a * dterm5b_dcbeta
        dterm5_dcgamma = term5a * dterm5b_dcgamma

        numerator = term0 + term1 + term2 + term3 + term4 + term5
        dnum_da = dterm0_da + dterm3_da + dterm5_da
        dnum_db = dterm1_db + dterm3_db + dterm4_db
        dnum_dc = dterm2_dc + dterm4_dc + dterm5_dc
        dnum_dalpha = (
            dterm0_dsalpha * dsalpha_dalpha
            + (dterm3_dcalpha + dterm4_dcalpha + dterm5_dcalpha) * dcalpha_dalpha
            )
        dnum_dbeta = (
            dterm1_dsbeta * dsbeta_dbeta
            + (dterm3_dcbeta + dterm4_dcbeta + dterm5_dcbeta) * dcbeta_dbeta
            )
        dnum_dgamma = (
            dterm2_dsgamma * dsgamma_dgamma
            + (dterm3_dcgamma + dterm4_dcgamma + dterm5_dcgamma) * dcgamma_dgamma
            )

        q2_pred = numerator / denom
        dq2_pred_duc = np.column_stack((
            dnum_da/denom,
            dnum_db/denom,
            dnum_dc/denom,
            dnum_dalpha/denom - numerator/denom**2 * ddenom_dalpha,
            dnum_dbeta/denom - numerator/denom**2 * ddenom_dbeta,
            dnum_dgamma/denom - numerator/denom**2 * ddenom_dgamma,
            ))
        return q2_pred, dq2_pred_duc

    def get_q2_pred_monoclinic(self, uc):
        term0 = 1 / np.sin(uc[3])**2
        term1 = self.hkl[:, 0]**2 / uc[0]**2
        term2 = self.hkl[:, 1]**2 / uc[1]**2 * np.sin(uc[3])**2
        term3 = self.hkl[:, 2]**2 / uc[2]**2
        term4 = -2*self.hkl[:, 0] * self.hkl[:, 2] * np.cos(uc[3]) / (uc[0] * uc[2])
        q2_pred = term0 * (term1 + term2 + term3 + term4)

        dterm0_duc3 = -2 / np.sin(uc[3])**3 * np.cos(uc[3])
        dterm1_duc0 = -2 * self.hkl[:, 0]**2 / uc[0]**3
        dterm2_duc1 = -2 * self.hkl[:, 1]**2 / uc[1]**3 * np.sin(uc[3])**2
        dterm2_duc3 = 2 * self.hkl[:, 1]**2 / uc[1]**2 * np.sin(uc[3]) * np.cos(uc[3])
        dterm3_duc2 = -2 * self.hkl[:, 2]**2 / uc[2]**3
        dterm4_duc0 = 2*self.hkl[:, 0] * self.hkl[:, 2] * np.cos(uc[3]) / (uc[0] * uc[2])**2 * uc[2]
        dterm4_duc2 = 2*self.hkl[:, 0] * self.hkl[:, 2] * np.cos(uc[3]) / (uc[0] * uc[2])**2 * uc[0]
        dterm4_duc3 = 2*self.hkl[:, 0] * self.hkl[:, 2] * np.sin(uc[3]) / (uc[0] * uc[2])

        dq2_pred_duc = np.column_stack((
            term0 * (dterm1_duc0 + dterm4_duc0),
            term0 * dterm2_duc1,
            term0 * (dterm3_duc2 + dterm4_duc2),
            term0 * (dterm2_duc3 + dterm4_duc3) + dterm0_duc3 * (term1 + term2 + term3 + term4)
            ))
        return q2_pred, dq2_pred_duc

    def loss_likelihood_t(self, uc):
        q2_pred, dq2_pred_duc = self.get_q2_pred(uc)

        residuals = (q2_pred - self.q2_obs) / self.sigma
        arg = 1 + 1/self.tuning_param * residuals**2
        likelihood = np.log(self.sigma) + self.exponent * np.log(arg)

        dlikelihood_dq2_pred = 2 * self.exponent * residuals / (self.sigma * arg * self.tuning_param)

        loss = -np.sum(likelihood)
        dloss_duc = -np.sum(dlikelihood_dq2_pred[:, np.newaxis] * dq2_pred_duc, axis=0)
        return loss, dloss_duc

    def loss_likelihood_t_hessian_diagonal(self, uc):
        q2_pred, dq2_pred_duc, d2q2_pred_duc2 = self.get_q2_pred(uc, jac=True, hessian=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        arg = 1 + 1/self.tuning_param * residuals**2
        constant = (self.tuning_param + 1) / self.tuning_param
        
        prefactor = 1 / (arg * self.sigma**2)
        term00 = (1 - 2 * residuals**2 / (self.tuning_param * arg))
        term0 = term00[:, np.newaxis, np.newaxis] * np.matmul(
            dq2_pred_duc[:, :, np.newaxis], dq2_pred_duc[:, np.newaxis, :]
            )

        self.__term1[:, self.__diag_indices, self.__diag_indices] = \
            (self.sigma * residuals)[:, np.newaxis] * d2q2_pred_duc2
        hessian = constant * np.sum(prefactor[:, np.newaxis, np.newaxis] * (term0 + self.__term1), axis=0)
        return hessian

    def loss_likelihood_t_no_jac(self, uc):
        q2_pred = self.get_q2_pred(uc, jac=False)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        arg = 1 + 1/self.tuning_param * residuals**2
        all_likelihoods = self.prefactor * arg**self.exponent
        likelihood = np.log(all_likelihoods)
        loss = -np.sum(likelihood)
        return loss, q2_pred

    def loss_likelihood_normal(self, uc):
        q2_pred, dq2_pred_duc = self.get_q2_pred(uc)

        residuals = (q2_pred - self.q2_obs) / self.sigma
        likelihood = self.prefactor + 1/2 * residuals**2

        dlikelihood_dq2_pred = residuals / self.sigma

        loss = np.sum(likelihood)
        dloss_duc = np.sum(dlikelihood_dq2_pred[:, np.newaxis] * dq2_pred_duc, axis=0)
        return loss, dloss_duc

    def loss_likelihood_normal_hessian_diagonal(self, uc):
        q2_pred, dq2_pred_duc, d2q2_pred_duc2 = self.get_q2_pred(uc, jac=True, hessian=True)

        term0 = np.matmul(dq2_pred_duc[:, :, np.newaxis], dq2_pred_duc[:, np.newaxis, :])
        self.__term1[:, self.__diag_indices, self.__diag_indices] = \
            (q2_pred - self.q2_obs)[:, np.newaxis] * d2q2_pred_duc2
        H = np.sum(self.hessian_prefactor * (term0 + self.__term1), axis=0)
        return H

    def loss_likelihood_normal_no_jac(self, uc):
        q2_pred = self.get_q2_pred(uc, jac=False)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        likelihood = self.prefactor + 1/2 * residuals**2
        loss = np.sum(likelihood)
        return loss, q2_pred

    def get_loss(self, uc):
        loss, q2_pred = self.loss_likelihood_no_jac(uc)
        return loss


class CandidateOptLoss_inv2:
    def __init__(self, q2_obs, lattice_system, likelihood, tuning_param=None):
        self.q2_obs = q2_obs
        self.likelihood = likelihood
        self.lattice_system = lattice_system
        self.delta_q_eps = 1e-10
        self.n_points = q2_obs.size

        if lattice_system == 'cubic':
            self.uc_length = 1
        elif lattice_system == 'tetragonal':
            self.uc_length = 2
        elif lattice_system == 'orthorhombic':
            self.uc_length = 3
        elif lattice_system == 'monoclinic':
            self.get_q2_pred = self.get_q2_pred_monoclinic
            self.uc_length = 4
        elif lattice_system == 'triclinic':
            self.get_q2_pred = self.get_q2_pred_triclinic
            self.uc_length = 6
        elif lattice_system == 'hexagonal':
            self.get_q2_pred = self.get_q2_pred_hexagonal
            self.uc_length = 2
        elif lattice_system == 'rhombohedral':
            self.get_q2_pred = self.get_q2_pred_rhombohedral
            self.uc_length = 2

        if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
            self.get_q2_pred = self.get_q2_pred_cubic_tetragonal_orthorhombic
            self.__diag_indices = np.arange(self.uc_length)
            self.__term1 = np.zeros((self.n_points, self.uc_length, self.uc_length))

        if self.likelihood == 't_dist':
            self.tuning_param = tuning_param
            self.exponent = -(self.tuning_param + 1) / 2
            self.loss_likelihood = self.loss_likelihood_t
            self.loss_likelihood_no_jac = self.loss_likelihood_t_no_jac
            if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
                self.loss_likelihood_hessian = self.loss_likelihood_t_hessian_diagonal
        elif self.likelihood == 'normal':
            self.loss_likelihood = self.loss_likelihood_normal
            self.loss_likelihood_no_jac = self.loss_likelihood_normal_no_jac
            if lattice_system in ['cubic', 'tetragonal', 'orthorhombic']:
                self.loss_likelihood_hessian = self.loss_likelihood_normal_hessian_diagonal

    def update(self, hkl, softmax, uc_inv2_init):
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
        elif self.lattice_system == 'hexagonal':
            self.hk_term = self.hkl[:, 0]**2 + self.hkl[:, 0]*self.hkl[:, 1] + self.hkl[:, 1]**2
        elif self.lattice_system == 'rhombohedral':
            self.hkl2_term = self.hkl[:, 0]**2 + self.hkl[:, 1]**2 + self.hkl[:, 2]**2
            self.hkl_term = self.hkl[:, 0]*self.hkl[:, 1] + self.hkl[:, 0]*self.hkl[:, 2] + self.hkl[:, 1]*self.hkl[:, 2]

        q2_pred_init, _ = self.get_q2_pred(uc_inv2_init)
        delta_q = np.abs(np.sqrt(q2_pred_init) - np.sqrt(self.q2_obs))
        self.sigma = np.sqrt(self.q2_obs * (delta_q + self.delta_q_eps))

        if self.likelihood == 't_dist':
            prefactor1 = gamma((self.tuning_param + 1) / 2) / gamma(self.tuning_param / 2)
            prefactor2 = 1 / np.sqrt(self.tuning_param * np.pi)
            prefactor0 = prefactor1 * prefactor2 / self.sigma
            self.prefactor = prefactor0 * self.softmax
        elif self.likelihood == 'normal':
            self.prefactor = np.log(np.sqrt(2*np.pi) * self.sigma) - np.log(self.softmax)
            self.hessian_prefactor = (1 / self.sigma**2)[:, np.newaxis, np.newaxis]

    def get_q2_pred_cubic_tetragonal_orthorhombic(self, uc_inv2, jac=True):
        arg = self.hkl2 * uc_inv2[np.newaxis, :]
        q2_pred = np.sum(arg, axis=1)
        if jac:
            dq2_pred_duc_inv2 = self.hkl2
            return q2_pred, dq2_pred_duc_inv2
        else:
            return q2_pred

    def loss_likelihood_t(self, uc):
        q2_pred, dq2_pred_duc = self.get_q2_pred(uc)

        residuals = (q2_pred - self.q2_obs) / self.sigma
        arg = 1 + 1/self.tuning_param * residuals**2
        likelihood = np.log(self.sigma) + self.exponent * np.log(arg)

        dlikelihood_dq2_pred = 2 * self.exponent * residuals / (self.sigma * arg * self.tuning_param)

        loss = -np.sum(likelihood)
        dloss_duc = -np.sum(dlikelihood_dq2_pred[:, np.newaxis] * dq2_pred_duc, axis=0)
        return loss, dloss_duc

    def loss_likelihood_t_hessian_diagonal(self, uc):
        q2_pred, dq2_pred_duc, d2q2_pred_duc2 = self.get_q2_pred(uc, jac=True, hessian=True)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        arg = 1 + 1/self.tuning_param * residuals**2
        constant = (self.tuning_param + 1) / self.tuning_param

        prefactor = 1 / (arg * self.sigma**2)
        term00 = (1 - 2 * residuals**2 / (self.tuning_param * arg))
        term0 = term00[:, np.newaxis, np.newaxis] * np.matmul(
            dq2_pred_duc[:, :, np.newaxis], dq2_pred_duc[:, np.newaxis, :]
            )

        self.__term1[:, self.__diag_indices, self.__diag_indices] = \
            (self.sigma * residuals)[:, np.newaxis] * d2q2_pred_duc2
        hessian = constant * np.sum(prefactor[:, np.newaxis, np.newaxis] * (term0 + self.__term1), axis=0)
        return hessian

    def loss_likelihood_t_no_jac(self, uc):
        q2_pred = self.get_q2_pred(uc, jac=False)
        residuals = (q2_pred - self.q2_obs) / self.sigma
        arg = 1 + 1/self.tuning_param * residuals**2
        all_likelihoods = self.prefactor * arg**self.exponent
        likelihood = np.log(all_likelihoods)
        loss = -np.sum(likelihood)
        return loss, q2_pred

    def loss_likelihood_normal(self, uc_inv2):
        q2_pred, dq2_pred_duc_inv2 = self.get_q2_pred(uc_inv2)

        residuals = (q2_pred - self.q2_obs) / self.sigma
        likelihood = self.prefactor + 1/2 * residuals**2

        dlikelihood_dq2_pred = residuals / self.sigma

        loss = np.sum(likelihood)
        dloss_duc_inv2 = np.sum(dlikelihood_dq2_pred[:, np.newaxis] * dq2_pred_duc_inv2, axis=0)
        return loss, dloss_duc_inv2

    def loss_likelihood_normal_hessian_diagonal(self, uc_inv2):
        q2_pred, dq2_pred_duc_inv2 = self.get_q2_pred(uc_inv2, jac=True)
        term0 = np.matmul(dq2_pred_duc_inv2[:, :, np.newaxis], dq2_pred_duc_inv2[:, np.newaxis, :])
        H = np.sum(self.hessian_prefactor * term0, axis=0)
        return H

    def loss_likelihood_normal_no_jac(self, uc_inv2):
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
