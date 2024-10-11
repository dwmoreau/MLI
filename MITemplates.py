import joblib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import scipy.spatial
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing
from tqdm import tqdm

from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_M20_sym_reversed
from Utilities import get_unit_cell_from_xnn
from Utilities import read_params
from Utilities import write_params
from Utilities import get_reciprocal_unit_cell_from_xnn
from Utilities import reciprocal_uc_conversion
from Utilities import Q2Calculator
from Utilities import fast_assign


class MITemplates:
    def __init__(self, group, data_params, template_params, hkl_ref, save_to, seed):
        self.template_params = template_params
        template_params_defaults = {
            'templates_per_dominant_zone_bin': 1000,
            }

        for key in template_params_defaults.keys():
            if key not in self.template_params.keys():
                self.template_params[key] = template_params_defaults[key]

        self.lattice_system = data_params['lattice_system']
        self.n_peaks = data_params['n_peaks']
        self.unit_cell_length = data_params['unit_cell_length']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.hkl_ref_length = data_params['hkl_ref_length']
        self.hkl_ref = hkl_ref
        self.group = group
        self.save_to = save_to
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        template_params_defaults = {
            'templates_per_dominant_zone_bin': 2000,
            'calibrate': False,
            'parallelization': 'multiprocessing',
            'n_processes': 2,
            'radius': 0.01,
            'n_train': 100,
            'max_depth': 4,
            'min_samples_leaf': 100,
            'l2_regularization': 10,
            }

        for key in template_params_defaults.keys():
            if key not in self.template_params.keys():
                self.template_params[key] = template_params_defaults[key]

    def save(self):
        write_params(
            self.template_params,
            f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv'
            )
        if self.template_params['calibrate']:
            joblib.dump(
                self.hgbc_classifier,
                f'{self.save_to}/{self.group}_template_calibrator_{self.template_params["tag"]}.bin'
                )

    def load_from_tag(self):
        self.miller_index_templates = np.load(
            f'{self.save_to}/{self.group}_miller_index_templates_{self.template_params["tag"]}.npy'
            )
        self.miller_index_templates_prob = np.load(
            f'{self.save_to}/{self.group}_miller_index_templates_prob_{self.template_params["tag"]}.npy',
            )
        params = read_params(f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv')
        params_keys = [
            'tag',
            'templates_per_dominant_zone_bin',
            'calibrate',
            'n_templates',
            'parallelization',
            'n_processes',
            'n_train',
            'radius',
            'max_depth',
            'min_samples_leaf',
            'l2_regularization',
            'calibration_n_peaks',
            ]
        self.template_params = dict.fromkeys(params_keys)
        self.template_params['tag'] = params['tag']
        self.template_params['templates_per_dominant_zone_bin'] = int(params['templates_per_dominant_zone_bin'])
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]
        if 'calibrate' in params.keys():
            if params['calibrate'] == 'True':
                self.template_params['calibrate'] = True
                self.hgbc_classifier = joblib.load(
                    f'{self.save_to}/{self.group}_template_calibrator_{self.template_params["tag"]}.bin'
                    )
                self.template_params['parallelization'] = params['parallelization']
                self.template_params['n_processes'] = int(params['n_processes'])
                self.template_params['n_train'] = int(params['n_train'])
                self.template_params['radius'] = float(params['radius'])
                self.template_params['max_depth'] = int(params['max_depth'])
                self.template_params['min_samples_leaf'] = int(params['min_samples_leaf'])
                self.template_params['l2_regularization'] = float(params['l2_regularization'])
                self.template_params['calibration_n_peaks'] = int(params['calibration_n_peaks'])
            else:
                self.template_params['calibrate'] = False
        else:
            self.template_params['calibrate'] = False

    def setup_templates(self, data):
        def get_counts(hkl_labels_func, hkl_ref_length):
            hkl_labels_func = hkl_labels_func[hkl_labels_func != hkl_ref_length - 1]
            if hkl_labels_func.size > 0:
                counts_ = np.bincount(hkl_labels_func, minlength=hkl_ref_length)
                hist_ = np.zeros(hkl_ref_length)
                hist_ = counts_ / hkl_labels_func.size
                return hist_
            else:
                return None
            
        def make_sets(N_sets, N_peaks, hkl_labels, hkl_ref_length, rng):
            MI_sets = np.zeros((N_sets, N_peaks), dtype=int)
            hist_initial = np.zeros((N_peaks, hkl_ref_length))
            for peak_index in range(N_peaks):
                hist_initial[peak_index] = get_counts(hkl_labels[:, peak_index], hkl_ref_length)
            for set_index in range(N_sets):
                MI_sets[set_index, 0] = rng.choice(hkl_ref_length, p=hist_initial[0])
                hkl_labels_ = hkl_labels
                for peak_index in range(1, N_peaks):
                    indices = hkl_labels_[:, peak_index - 1] == MI_sets[set_index, peak_index - 1]
                    hkl_labels_ = hkl_labels_[indices]
                    hist_loop = get_counts(hkl_labels_[:, peak_index], hkl_ref_length)
                    if not hist_loop is None:
                        MI_sets[set_index, peak_index] = rng.choice(hkl_ref_length, p=hist_loop)
                    else:
                        MI_sets[set_index, peak_index] = rng.choice(hkl_ref_length, p=hist_initial[peak_index])
            return MI_sets

        training_data = data[data['train']]
        hkl_labels_all = np.stack(training_data['hkl_labels'])
        
        if self.lattice_system == 'cubic':
            # Cubic and rhombohedral do not have dominant zones
            miller_index_templates = make_sets(
                self.template_params['templates_per_dominant_zone_bin'],
                self.n_peaks,
                hkl_labels_all,
                self.hkl_ref_length,
                self.rng
                )
            sampling_probability = np.ones(self.template_params['templates_per_dominant_zone_bin'])
        elif self.lattice_system == 'rhombohedral':
            reindexed_xnn = np.stack(training_data['reindexed_xnn'])
            unit_cell_volume = np.array(training_data['reindexed_volume'])
            sorted_unit_cell_volume = np.sort(unit_cell_volume)
            volume_bins = np.linspace(
                sorted_unit_cell_volume[int(0.001*sorted_unit_cell_volume.size)],
                sorted_unit_cell_volume[int(0.999*sorted_unit_cell_volume.size)],
                11
                )
            ratio = reindexed_xnn[:, 3] / reindexed_xnn[:, 0]
            ratio_bins = np.linspace(-1, 2, 21)
            ra = np.sqrt(reindexed_xnn[:, 0])
            cos_ralpha = 1/2 * ratio
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(ra, cos_ralpha, linestyle='none', marker='.', markersize=1, alpha=0.2)
            axes[1].hist(ratio, bins=ratio_bins)
            axes[0].set_ylabel('$cos(\\alpha*)$')
            axes[0].set_xlabel('a*')
            axes[1].set_xlabel('$2 x cos(\\alpha*)$')
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/{self.group}_dominant_zone_ratio_{self.template_params["tag"]}.png')
            plt.close()

            mi_sets = []
            sampling_probability = []
            for i in range(20):
                indices = np.logical_and(ratio > ratio_bins[i], ratio <= ratio_bins[i + 1])
                if np.sum(indices) > 0:
                    hkl_labels_bin = hkl_labels_all[indices]
                    if hkl_labels_bin.shape[0] < self.template_params['templates_per_dominant_zone_bin']:
                        sampling_ratio = hkl_labels_bin.shape[0] / self.template_params['templates_per_dominant_zone_bin']
                        sampling_probability.append(
                            sampling_ratio * np.ones(self.template_params['templates_per_dominant_zone_bin'])
                            )
                        mi_sets.append(hkl_labels_bin)
                    else:
                        sampling_probability.append(
                            np.ones(self.template_params['templates_per_dominant_zone_bin'])
                            )
                        mi_sets.append(make_sets(
                            self.template_params['templates_per_dominant_zone_bin'],
                            self.n_peaks,
                            hkl_labels_bin,
                            self.hkl_ref_length,
                            self.rng
                            ))
            for i in range(10):
                indices = np.logical_and(
                    unit_cell_volume > volume_bins[i],
                    unit_cell_volume <= volume_bins[i + 1]
                    )
                if np.sum(indices) > 0:
                    hkl_labels_bin = hkl_labels_all[indices]
                    if hkl_labels_bin.shape[0] < self.template_params['templates_per_dominant_zone_bin']:
                        sampling_ratio = hkl_labels_bin.shape[0] / self.template_params['templates_per_dominant_zone_bin']
                        sampling_probability.append(
                            sampling_ratio * np.ones(self.template_params['templates_per_dominant_zone_bin'])
                            )
                        mi_sets.append(hkl_labels_bin)
                    else:
                        sampling_probability.append(
                            np.ones(self.template_params['templates_per_dominant_zone_bin'])
                            )
                        mi_sets.append(make_sets(
                            self.template_params['templates_per_dominant_zone_bin'],
                            self.n_peaks,
                            hkl_labels_bin,
                            self.hkl_ref_length,
                            self.rng
                            ))
            miller_index_templates = np.row_stack(mi_sets)
            sampling_probability = np.concatenate(sampling_probability)
        else:
            reindexed_xnn = np.stack(training_data['reindexed_xnn'])
            ratio_xnn = reindexed_xnn[:, :3].min(axis=1) / reindexed_xnn[:, :3].max(axis=1)

            reindexed_unit_cell = np.stack(training_data['reindexed_unit_cell'])
            ratio_unit_cell = reindexed_unit_cell[:, :3].min(axis=1) / reindexed_unit_cell[:, :3].max(axis=1)

            reindexed_rec_unit_cell = reciprocal_uc_conversion(reindexed_unit_cell, partial_unit_cell=False)
            ratio_rec_unit_cell = reindexed_rec_unit_cell[:, :3].min(axis=1) / reindexed_rec_unit_cell[:, :3].max(axis=1)

            reindexed_hkl = np.stack(training_data['reindexed_hkl'])
            hkl_information = np.sum(reindexed_hkl != 0, axis=1).min(axis=1)
            hkl_information_hist = np.bincount(hkl_information, minlength=self.n_peaks)

            unit_cell_volume = np.array(training_data['reindexed_volume'])
            sorted_unit_cell_volume = np.sort(unit_cell_volume)
            volume_bins = np.linspace(
                sorted_unit_cell_volume[int(0.001*sorted_unit_cell_volume.size)],
                sorted_unit_cell_volume[int(0.999*sorted_unit_cell_volume.size)],
                11
                )

            mean_ratio = np.zeros((self.n_peaks, 2, 2))
            for i in range(self.n_peaks):
                mean_ratio[i, 0, 0] = np.mean(ratio_xnn[hkl_information == i])
                mean_ratio[i, 1, 0] = np.std(ratio_xnn[hkl_information == i])
                mean_ratio[i, 0, 1] = np.mean(ratio_unit_cell[hkl_information == i])
                mean_ratio[i, 1, 1] = np.std(ratio_unit_cell[hkl_information == i])

            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            axes[0].hist(ratio_xnn, bins=np.linspace(0, 1, self.n_peaks + 1))
            axes[1].hist(ratio_unit_cell, bins=np.linspace(0, 1, self.n_peaks + 1))
            axes[2].hist(ratio_rec_unit_cell, bins=np.linspace(0, 1, self.n_peaks + 1))
            axes[3].bar(np.arange(self.n_peaks), hkl_information_hist, width=1)
            axes[4].plot(
                hkl_information, ratio_unit_cell,
                marker='.', linestyle='none', markersize=0.25, alpha=0.5
                )
            axes[4].errorbar(np.arange(self.n_peaks), mean_ratio[:, 0, 1], mean_ratio[:, 1, 1])

            axes[0].set_xlabel('Dominant zone ratio\n(Min/Max Xnn)')
            axes[1].set_xlabel('Dominant zone ratio\n(Min/Max Unit Cell)')
            axes[2].set_xlabel('Dominant zone ratio\n(Min/Max Reciprocal Unit Cell)')

            axes[0].set_ylabel('Counts')
            axes[1].set_ylabel('Counts')
            axes[2].set_ylabel('Counts')
            axes[3].set_ylabel('Counts')
            
            axes[3].set_xlabel('Minimum Information')
            axes[4].set_xlabel('Minimum Information')
            axes[4].set_ylabel('Dominant zone ratio (Unit Cell)')
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/{self.group}_dominant_zone_ratio_{self.template_params["tag"]}.png')
            plt.close()

            mi_sets = []
            sampling_probability = []
            n_ratio_bins = 10
            ratio_bins = np.linspace(0, 1, n_ratio_bins + 1)
            templates_per_information_bin = self.template_params['templates_per_dominant_zone_bin']
            templates_per_dominant_zone_bin = int(self.template_params['templates_per_dominant_zone_bin'] / n_ratio_bins)
            for i in range(self.n_peaks):
                indices = hkl_information == i
                if np.sum(indices) > 0:
                    hkl_labels_bin = hkl_labels_all[indices]
                    ratio_unit_cell_bin = ratio_unit_cell[indices]
                    if hkl_labels_bin.shape[0] < templates_per_information_bin:
                        sampling_ratio = templates_per_information_bin / hkl_labels_bin.shape[0]
                        sampling_probability.append(
                            sampling_ratio * np.ones(templates_per_information_bin)
                            )
                        mi_sets.append(hkl_labels_bin)
                    else:
                        for ratio_bin_index in range(n_ratio_bins):
                            ratio_indices = np.logical_and(
                                ratio_unit_cell_bin > ratio_bins[ratio_bin_index],
                                ratio_unit_cell_bin <= ratio_bins[ratio_bin_index + 1],
                                )
                            if np.sum(ratio_indices) > 0:
                                hkl_labels_ratio_bin = hkl_labels_bin[ratio_indices]
                                if hkl_labels_ratio_bin.shape[0] < templates_per_dominant_zone_bin:
                                    sampling_ratio = templates_per_dominant_zone_bin / hkl_labels_ratio_bin.shape[0]
                                    sampling_probability.append(
                                        sampling_ratio * np.ones(templates_per_dominant_zone_bin)
                                        )
                                    mi_sets.append(hkl_labels_ratio_bin)
                                else:
                                    sampling_probability.append(
                                        np.ones(templates_per_dominant_zone_bin)
                                        )
                                    mi_sets.append(make_sets(
                                        templates_per_dominant_zone_bin,
                                        self.n_peaks,
                                        hkl_labels_bin,
                                        self.hkl_ref_length,
                                        self.rng
                                        ))

            for i in range(10):
                indices = np.logical_and(
                    unit_cell_volume > volume_bins[i],
                    unit_cell_volume <= volume_bins[i + 1]
                    )
                if np.sum(indices) > 0:
                    hkl_labels_bin = hkl_labels_all[indices]
                    if hkl_labels_bin.shape[0] < self.template_params['templates_per_dominant_zone_bin']:
                        sampling_ratio = hkl_labels_bin.shape[0] / self.template_params['templates_per_dominant_zone_bin']
                        sampling_probability.append(
                            sampling_ratio * np.ones(self.template_params['templates_per_dominant_zone_bin'])
                            )
                        mi_sets.append(hkl_labels_bin)
                    else:
                        sampling_probability.append(
                            np.ones(self.template_params['templates_per_dominant_zone_bin'])
                            )
                        mi_sets.append(make_sets(
                            self.template_params['templates_per_dominant_zone_bin'],
                            self.n_peaks,
                            hkl_labels_bin,
                            self.hkl_ref_length,
                            self.rng
                            ))
            miller_index_templates = np.row_stack(mi_sets)
            sampling_probability = np.concatenate(sampling_probability)

        self.miller_index_templates, unique_indices = np.unique(
            miller_index_templates, axis=0, return_index=True
            )
        sampling_probability = sampling_probability[unique_indices]
        self.miller_index_templates_prob = sampling_probability / sampling_probability.sum()
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]
        np.save(
            f'{self.save_to}/{self.group}_miller_index_templates_{self.template_params["tag"]}.npy',
            self.miller_index_templates
            )
        np.save(
            f'{self.save_to}/{self.group}_miller_index_templates_prob_{self.template_params["tag"]}.npy',
            self.miller_index_templates_prob
            )

    def setup(self, data):
        self.setup_templates(data)
        if self.template_params['calibrate']:
            self.calibrate_templates(data)
        self.save()

    def generate_xnn_fast(self, q2_obs, indices=None):
        # This is still slow, but about 2 times faster than the other method
        # original I just did linear least squares iteratively until q2_calc was ordered (1st for loop)
        # It is faster to use Gauss-Newton non-linear least squares (2nd for loop)
        # This is mostly copied from TargetFunctions.py
        # I forgot the reason, but I don't use this function.

        if indices is None:
            hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)
            n_templates = self.template_params['n_templates']
        else:
            hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates[indices]], self.lattice_system)
            n_templates = indices.size

        # Calculate initial values for xnn using linear least squares methods
        xnn = np.zeros((n_templates, self.unit_cell_length))
        A = hkl2 / q2_obs[np.newaxis, :, np.newaxis]
        b = np.ones(self.n_peaks)
        for template_index in range(n_templates):
            xnn[template_index], r, rank, s = np.linalg.lstsq(
                A[template_index], b, rcond=None
                )

        # q2_calc should increase monotonically. Sort hkl2 then re-solve for xnn iteratively.
        sigma = q2_obs[np.newaxis]
        hessian_prefactor = (1 / sigma**2)[:, :, np.newaxis, np.newaxis]
        for index in range(5):
            if index == 4:
                q2_calculator = Q2Calculator(
                    lattice_system=self.lattice_system,
                    hkl=self.hkl_ref,
                    tensorflow=False,
                    representation='xnn'
                    )
                q2_ref_calc = q2_calculator.get_q2(xnn)
                hkl_assign = fast_assign(q2_obs, q2_ref_calc)
                hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
                hkl2 = get_hkl_matrix(hkl, self.lattice_system)
                q2_calc = (hkl2 @ xnn[:, :, np.newaxis])[:, :, 0]
                q2_calc_max = q2_calc.max(axis=1)
                N_pred = np.count_nonzero(q2_ref_calc < q2_calc_max[:, np.newaxis], axis=1)
            else:
                q2_calc = (hkl2 @ xnn[:, :, np.newaxis])[:, :, 0]
                sort_indices = q2_calc.argsort(axis=1)
                q2_calc = np.take_along_axis(q2_calc, sort_indices, axis=1)
                hkl2 = np.take_along_axis(hkl2, sort_indices[:, :, np.newaxis], axis=1)
            residuals = (q2_calc - q2_obs[np.newaxis]) / sigma
            dlikelihood_dq2_pred = residuals / sigma
            dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * hkl2, axis=1)
            term0 = np.matmul(hkl2[:, :, :, np.newaxis], hkl2[:, :, np.newaxis, :])
            H = np.sum(hessian_prefactor * term0, axis=1)
            good = np.linalg.matrix_rank(H, hermitian=True) == self.unit_cell_length
            delta_gn = np.zeros((n_templates, self.unit_cell_length))
            delta_gn[good] = -np.matmul(np.linalg.inv(H[good]), dloss_dxnn[good, :, np.newaxis])[:, :, 0]
            xnn += delta_gn
        residuals = np.abs(q2_calc - q2_obs[np.newaxis])
        
        xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
        return xnn, residuals, N_pred, q2_calc_max

    def generate_xnn(self, q2_obs, indices=None):
        # This is slower
        if indices is None:
            hkl2_all = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)
            n_templates = self.template_params['n_templates']
        else:
            hkl2_all = get_hkl_matrix(self.hkl_ref[self.miller_index_templates[indices]], self.lattice_system)
            n_templates = indices.size

        xnn = np.zeros((n_templates, self.unit_cell_length))
        residuals = np.zeros((n_templates, self.n_peaks))
        order = np.arange(self.n_peaks)
        
        for template_index in range(n_templates):
            sigma = q2_obs
            hkl2 = hkl2_all[template_index]
            status = True
            i = 0
            xnn_last = np.zeros(self.unit_cell_length)
            while status:
                xnn_current, r, rank, s = np.linalg.lstsq(
                    hkl2 / sigma[:, np.newaxis], q2_obs / sigma,
                    rcond=None
                    )
                q2_calc = hkl2 @ xnn_current            
                if np.all(q2_calc[1:] >= q2_calc[:-1]):
                    delta_q2 = np.abs(q2_obs - q2_calc)
                    if np.linalg.norm(xnn_current - xnn_last) < 0.01:
                        status = False
                else:
                    sort_indices = np.argsort(q2_calc)
                    hkl2 = hkl2[sort_indices]
                    delta_q2 = np.abs(q2_obs - q2_calc[sort_indices])
                sigma = np.sqrt(q2_obs * (delta_q2 + 1e-10))
                xnn_last = xnn_current
                if i == 10:
                    status = False
                i += 1
            xnn[template_index] = xnn_current
            residuals[template_index] = delta_q2
        xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
        return xnn, residuals

    def generate_uncalibrated(self, n_templates, rng, q2_obs):
        # This is primary used to generate candidates for the optimizer, which expects that the
        # sum of the template and sampled candidates to be the same. This is why n_samples gets
        # changed
        if n_templates == 'all':
            xnn_templates, _ = self.generate_xnn(q2_obs)
        elif n_templates < self.template_params['n_templates']:
            # requesting fewer templates than in the set
            # subsample
            indices = rng.choice(
                self.template_params['n_templates'],
                size=n_templates,
                replace=False,
                p=self.miller_index_templates_prob,
                )
            xnn_templates, _ = self.generate_xnn(q2_obs, indices)
        elif n_templates > self.template_params['n_templates']:
            xnn_templates, _ = self.generate_xnn(q2_obs)
            # requesting more templates than in the set
            # Just sample multiple times
            print('WARNING: Requesting more templates than available. Duplicates will be returned')
            difference = n_templates - self.template_params['n_templates']
            if difference > self.template_params['n_templates']:
                replace = True
            else:
                replace = False
            indices = rng.choice(
                self.template_params['n_templates'],
                size=difference,
                replace=replace,
                p=self.miller_index_templates_prob,
                )
            xnn_templates =  np.concatenate((xnn_templates, xnn_templates[indices]), axis=0)

        unit_cell_templates = get_unit_cell_from_xnn(
            xnn_templates, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return unit_cell_templates

    def generate_calibrated(self, n_templates, rng, q2_obs):
        xnn_templates_all, q2_error, N_pred, q2_calc_max = self.generate_xnn_fast(q2_obs)
        _, unique_indices = np.unique(
            np.round(xnn_templates_all, decimals=6), return_index=True, axis=0
            )
        xnn_templates_all = xnn_templates_all[unique_indices]
        q2_error = q2_error[unique_indices]
        N_pred = N_pred[unique_indices]
        q2_calc_max = q2_calc_max[unique_indices]

        inputs = np.concatenate((
            q2_error[:, :self.template_params['calibration_n_peaks']],
            N_pred[:, np.newaxis],
            q2_calc_max[:, np.newaxis],
            ), axis=1)
        probability_templates = self.hgbc_classifier.predict_proba(inputs)
        if n_templates == 'all':
            xnn_templates = xnn_templates_all
        elif n_templates < self.template_params['n_templates']:
            # The output of the classification is
            #   col 0: in far category (False)
            #   col 1: in close category (True)
            top_n_indices = np.argsort(probability_templates[:, 0])[:n_templates]
            xnn_templates = xnn_templates_all[top_n_indices]
        elif n_templates > self.template_params['n_templates']:
            # requesting more templates than in the set
            # Just sample multiple times
            print('WARNING: Requesting more templates than available. Duplicates will be returned')
            difference = n_templates - self.template_params['n_templates']
            if difference > self.template_params['n_templates']:
                replace = True
            else:
                replace = False
            indices = rng.choice(
                self.template_params['n_templates'],
                size=difference,
                replace=replace,
                p=self.miller_index_templates_prob,
                )
            xnn_templates =  np.concatenate((xnn_templates_all, xnn_templates_all[indices]), axis=0)
        unit_cell_templates = get_unit_cell_from_xnn(
            xnn_templates, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return unit_cell_templates

    def generate(self, n_templates, rng, q2_obs):
        if self.template_params['calibrate']:
            print('Generating Calibrated Candidates')
            return self.generate_calibrated(n_templates, rng, q2_obs)
        else:
            return self.generate_uncalibrated(n_templates, rng, q2_obs)

    def _get_inputs_worker(self, inputs):
        q2_obs = inputs[0]
        xnn_true = inputs[1]
        xnn, q2_error, N_pred, q2_calc_max = self.generate_xnn_fast(q2_obs)
        distance = scipy.spatial.distance.cdist(xnn, xnn_true[np.newaxis])
        neighbor = (distance < self.template_params['radius']).astype(int)[:, 0]
        return q2_error, neighbor, distance, xnn, N_pred, q2_calc_max

    def get_inputs(self, data, n_entries):
        q2_obs = np.stack(data['q2'])
        

        q2_error = []
        neighbor = []
        distance = []
        xnn = []
        N_pred = []
        q2_calc_max = []
        if n_entries is None:
            n_entries = len(data)
            indices = np.arange(n_entries)
        else:
            n_entries = min(n_entries, len(data))
            indices = self.rng.choice(len(data), n_entries, replace=False)
        xnn_true = np.stack(data['reindexed_xnn'])[:, self.unit_cell_indices]
        if self.template_params['parallelization'] is None:
            print(f'Setting up {n_entries} entries serially')
            for index in tqdm(indices):
                q2_error_entry, neighbor_entry, distance_entry, xnn_entry, N_pred_entry, q2_calc_max_entry = \
                    self._get_inputs_worker([q2_obs[index], xnn_true[index]])
                q2_error.append(q2_error_entry)
                neighbor.append(neighbor_entry)
                distance.append(distance_entry)
                xnn.append(xnn_entry)
                N_pred.append(N_pred_entry)
                q2_calc_max.append(q2_calc_max_entry)
        elif self.template_params['parallelization'] == 'multiprocessing':
            print(f'Setting up {n_entries} entries using multiprocessing')
            with multiprocessing.Pool(self.template_params['n_processes']) as p:
                outputs = p.map(self._get_inputs_worker, zip(q2_obs[indices], xnn_true[indices]))
            for i in range(n_entries):
                q2_error.append(outputs[i][0])
                neighbor.append(outputs[i][1])
                distance.append(outputs[i][2])
                xnn.append(outputs[i][3])
                N_pred.append(outputs[i][4])
                q2_calc_max.append(outputs[i][5])

        q2_error = np.row_stack(q2_error)
        neighbor = np.concatenate(neighbor)
        distance = np.concatenate(distance)
        xnn = np.stack(xnn, axis=0)
        N_pred = np.concatenate(N_pred)
        q2_calc_max = np.concatenate(q2_calc_max)
        return q2_error, neighbor, distance, xnn, xnn_true[indices], N_pred, q2_calc_max

    def calibrate_templates(self, data):
        unaugmented_data = data[~data['augmented']]
        training_data = unaugmented_data[unaugmented_data['train']]
        val_data = unaugmented_data[~unaugmented_data['train']]
        n_val = int(0.2*self.template_params['n_train'])
        q2_error_train, neighbor_train, distance_train, xnn_train_pred, xnn_train_true, N_pred_train, q2_calc_max_train = \
            self.get_inputs(training_data, self.template_params['n_train'])
        q2_error_val, neighbor_val, distance_val, xnn_val_pred, xnn_val_true, N_pred_val, q2_calc_max_val = \
            self.get_inputs(val_data, n_val)
        #np.save('MICal_test_q2_error_train.npy', q2_error_train)
        #np.save('MICal_test_q2_error_val.npy', q2_error_val)
        #np.save('MICal_test_neighbor_train.npy', neighbor_train)
        #np.save('MICal_test_neighbor_val.npy', neighbor_val)
        #np.save('MICal_test_distance_train.npy', distance_train)
        #np.save('MICal_test_distance_val.npy', distance_val)
        #np.save('MICal_test_xnn_train_pred.npy', xnn_train_pred)
        #np.save('MICal_test_xnn_val_pred.npy', xnn_val_pred)

        #q2_error_train = np.load('MICal_test_q2_error_train.npy') 
        #q2_error_val = np.load('MICal_test_q2_error_val.npy') 
        #neighbor_train = np.load('MICal_test_neighbor_train.npy') 
        #neighbor_val = np.load('MICal_test_neighbor_val.npy') 
        #distance_train = np.load('MICal_test_distance_train.npy') 
        #distance_val = np.load('MICal_test_distance_val.npy') 
        #xnn_train_pred = np.load('MICal_test_xnn_train_pred.npy')
        #xnn_val_pred = np.load('MICal_test_xnn_val_pred.npy')

        train_inputs = np.concatenate((
            q2_error_train[:, :self.template_params['calibration_n_peaks']],
            N_pred_train[:, np.newaxis],
            q2_calc_max_train[:, np.newaxis],
            ), axis=1)
        val_inputs = np.concatenate((
            q2_error_val[:, :self.template_params['calibration_n_peaks']],
            N_pred_val[:, np.newaxis],
            q2_calc_max_val[:, np.newaxis],
            ), axis=1)
        print('Fitting Hist Boosting Model')  
        self.hgbc_classifier = sklearn.ensemble.HistGradientBoostingClassifier(
            max_depth=self.template_params['max_depth'],
            min_samples_leaf=self.template_params['min_samples_leaf'],
            class_weight='balanced',
            l2_regularization=self.template_params['l2_regularization'],
            )
        self.hgbc_classifier.fit(train_inputs, neighbor_train)
        joblib.dump(
            self.hgbc_classifier,
            f'{self.save_to}/{self.group}_template_calibrator_{self.template_params["tag"]}.bin'
            )
        self.hgbc_classifier = joblib.load(
            f'{self.save_to}/{self.group}_template_calibrator_{self.template_params["tag"]}.bin'
            )

        probability_train = self.hgbc_classifier.predict_proba(train_inputs)
        score_train = self.hgbc_classifier.score(train_inputs, neighbor_train)
        probability_val = self.hgbc_classifier.predict_proba(val_inputs)
        score_val = self.hgbc_classifier.score(val_inputs, neighbor_val)

        alpha = 0.5
        ms = 0.5
        top_n = 200

        distance_close_train = distance_train[neighbor_train.astype(bool)]
        distance_far_train = distance_train[~neighbor_train.astype(bool)]
        distance_close_val = distance_val[neighbor_val.astype(bool)]
        distance_far_val = distance_val[~neighbor_val.astype(bool)]

        probability_close_train = probability_train[neighbor_train.astype(bool)][:, 1]
        probability_far_train = probability_train[~neighbor_train.astype(bool)][:, 1]
        probability_close_val = probability_val[neighbor_val.astype(bool)][:, 1]
        probability_far_val = probability_val[~neighbor_val.astype(bool)][:, 1]

        rng = np.random.default_rng(0)
        if distance_train.size > 100000:
            indices_train = rng.choice(distance_train.size, size=100000, replace=False)
        else:
            indices_train = np.arange(distance_train.size)
        if distance_val.size > 100000:
            indices_val = rng.choice(distance_val.size, size=100000, replace=False)
        else:
            indices_val = np.arange(distance_val.size)

        top_n_indices_train = np.argsort(probability_train[:, 0])[:top_n]
        top_n_distance_train = np.median(distance_train[top_n_indices_train])

        top_n_indices_val = np.argsort(probability_val[:, 0])[:top_n]
        top_n_distance_val = np.median(distance_val[top_n_indices_val])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
        axes[0].plot(
            probability_train[indices_train, 1], distance_train[indices_train],
            linestyle='none', marker='.', alpha=alpha, markersize=ms
            )
        axes[1].plot(
            probability_val[indices_val, 1], distance_val[indices_val],
            linestyle='none', marker='.', alpha=alpha, markersize=ms
            )
        axes[0].set_title(f'TRAIN Score: {score_train:0.3f}\nMedian distance (top {top_n} / all) {top_n_distance_train:0.5f} / {np.median(distance_train):0.5f}')
        axes[1].set_title(f'Val Score: {score_val:0.3f}\nMedian distance (top {top_n} / all) {top_n_distance_val:0.5f} / {np.median(distance_val):0.5f}')
        axes[0].set_ylim([-0.001, 0.05])
        xlim = axes[0].get_xlim()
        for i in range(2):
            axes[i].set_xlabel('Probability in convergence radius')
            axes[i].plot(xlim, self.template_params['radius']*np.ones(2), color=[0, 0, 0])
        axes[0].set_ylabel('Xnn distance')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.group}_calibration_{self.template_params["tag"]}.png')
        plt.close()

        ##############################
        # Plot unit cell evaluations #
        ##############################
        val_xnn = xnn_val_true
        val_unit_cell = get_unit_cell_from_xnn(
            val_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        probability_val = probability_val[:, 1].reshape((
            n_val, self.template_params['n_templates']
            ))
        val_xnn_pred_top5 = np.take_along_axis(
            xnn_val_pred,
            np.argsort(probability_val, axis=1)[:, ::-1][:, :5, np.newaxis],
            axis=1
            )
        val_unit_cell_pred_top5 = np.zeros(val_xnn_pred_top5.shape)
        for index in range(5):
            val_unit_cell_pred_top5[:, index, :] = get_unit_cell_from_xnn(
                val_xnn_pred_top5[:, index, :], partial_unit_cell=True, lattice_system=self.lattice_system
                )

        train_xnn = xnn_train_true
        train_unit_cell = get_unit_cell_from_xnn(
            train_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        probability_train = probability_train[:, 1].reshape((
            self.template_params['n_train'], self.template_params['n_templates']
            ))
        train_xnn_pred_top5 = np.take_along_axis(
            xnn_train_pred,
            np.argsort(probability_train, axis=1)[:, ::-1][:, :5, np.newaxis],
            axis=1
            )
        train_unit_cell_pred_top5 = np.zeros(train_xnn_pred_top5.shape)
        for index in range(5):
            train_unit_cell_pred_top5[:, index, :] = get_unit_cell_from_xnn(
                train_xnn_pred_top5[:, index, :], partial_unit_cell=True, lattice_system=self.lattice_system
                )

        figsize = (self.unit_cell_length*2 + 2, 6)
        fig, axes = plt.subplots(2, self.unit_cell_length, figsize=figsize)
        unit_cell_titles = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        xnn_titles = ['Xhh', 'Xkk', 'Xll', 'Xkl', 'Xhl', 'Xhk']
        alpha = 0.1
        markersize = 0.5
        for plot_index in range(2):
            if plot_index == 0:
                val_xnn_pred = val_xnn_pred_top5[:, 0, :]
                val_unit_cell_pred = val_unit_cell_pred_top5[:, 0, :]
                train_xnn_pred = train_xnn_pred_top5[:, 0, :]
                train_unit_cell_pred = train_unit_cell_pred_top5[:, 0, :]
                save_label = 'most_probable'
            elif plot_index == 1:
                val_diff = np.linalg.norm(val_xnn_pred_top5 - val_xnn[:, np.newaxis, :], axis=2)
                val_xnn_pred = np.take_along_axis(
                    val_xnn_pred_top5,
                    np.argmin(val_diff, axis=1)[:, np.newaxis, np.newaxis],
                    axis=1
                    )[:, 0, :]
                val_unit_cell_pred = np.take_along_axis(
                    val_unit_cell_pred_top5,
                    np.argmin(val_diff, axis=1)[:, np.newaxis, np.newaxis],
                    axis=1
                    )[:, 0, :]

                train_diff = np.linalg.norm(train_xnn_pred_top5 - train_xnn[:, np.newaxis, :], axis=2)
                train_xnn_pred = np.take_along_axis(
                    train_xnn_pred_top5,
                    np.argmin(train_diff, axis=1)[:, np.newaxis, np.newaxis],
                    axis=1
                    )[:, 0, :]
                train_unit_cell_pred = np.take_along_axis(
                    train_unit_cell_pred_top5,
                    np.argmin(train_diff, axis=1)[:, np.newaxis, np.newaxis],
                    axis=1
                    )[:, 0, :]
                save_label = 'best'

            train_unit_cell_error = np.abs(train_unit_cell_pred - train_unit_cell)
            val_unit_cell_error = np.abs(val_unit_cell_pred - val_unit_cell)
            train_xnn_error = np.abs(train_xnn_pred - train_xnn)
            val_xnn_error = np.abs(val_xnn_pred - val_xnn)
            for uc_index in range(self.unit_cell_length):
                sorted_unit_cell = np.sort(train_unit_cell[:, uc_index])
                lower_unit_cell = sorted_unit_cell[int(0.005*sorted_unit_cell.size)]
                upper_unit_cell = sorted_unit_cell[int(0.995*sorted_unit_cell.size)]
                if upper_unit_cell > lower_unit_cell:
                    axes[0, uc_index].plot(
                        train_unit_cell[:, uc_index], train_unit_cell_pred[:, uc_index],
                        color=[0, 0, 0], alpha=alpha,
                        linestyle='none', marker='.', markersize=markersize,
                        )
                    axes[0, uc_index].plot(
                        val_unit_cell[:, uc_index], val_unit_cell_pred[:, uc_index],
                        color=[0.8, 0, 0], alpha=alpha,
                        linestyle='none', marker='.', markersize=markersize,
                        )
                    axes[0, uc_index].plot(
                        [lower_unit_cell, upper_unit_cell], [lower_unit_cell, upper_unit_cell],
                        color=[0.7, 0, 0], linestyle='dotted'
                        )
                    axes[0, uc_index].set_xlim([lower_unit_cell, upper_unit_cell])
                    axes[0, uc_index].set_ylim([lower_unit_cell, upper_unit_cell])

                error_train = np.sort(train_unit_cell_error[:, uc_index])
                error_train = error_train[~np.isnan(error_train)]
                unit_cell_p25_train = error_train[int(0.25 * error_train.size)]
                unit_cell_p50_train = error_train[int(0.50 * error_train.size)]
                unit_cell_p75_train = error_train[int(0.75 * error_train.size)]
                unit_cell_rmse_train = np.sqrt(1/error_train.size * np.linalg.norm(error_train)**2)
                error_val = np.sort(val_unit_cell_error[:, uc_index])
                error_val = error_val[~np.isnan(error_val)]
                unit_cell_p25_val = error_val[int(0.25 * error_val.size)]
                unit_cell_p50_val = error_val[int(0.50 * error_val.size)]
                unit_cell_p75_val = error_val[int(0.75 * error_val.size)]
                unit_cell_rmse_val = np.sqrt(1/error_val.size * np.linalg.norm(error_val)**2)
                unit_cell_error_titles = [
                    unit_cell_titles[uc_index],
                    f'RMSE: {unit_cell_rmse_train:0.2f} / {unit_cell_rmse_val:0.2f}',
                    f'25%: {unit_cell_p25_train:0.2f} / {unit_cell_p25_val:0.2f}',
                    f'50%: {unit_cell_p50_train:0.2f} / {unit_cell_p50_val:0.2f}',
                    f'75%: {unit_cell_p75_train:0.2f} / {unit_cell_p75_val:0.2f}',
                    ]
                axes[0, uc_index].set_title('\n'.join(unit_cell_error_titles), fontsize=12)

                sorted_xnn = np.sort(train_xnn[:, uc_index])
                lower_xnn = sorted_xnn[int(0.005*sorted_xnn.size)]
                upper_xnn = sorted_xnn[int(0.995*sorted_xnn.size)]

                if upper_xnn > lower_xnn:
                    axes[1, uc_index].plot(
                        train_xnn[:, uc_index], train_xnn_pred[:, uc_index],
                        color=[0, 0, 0], alpha=alpha,
                        linestyle='none', marker='.', markersize=markersize,
                        )
                    axes[1, uc_index].plot(
                        val_xnn[:, uc_index], val_xnn_pred[:, uc_index],
                        color=[0.8, 0, 0], alpha=alpha,
                        linestyle='none', marker='.', markersize=markersize,
                        )
                    axes[1, uc_index].plot(
                        [lower_xnn, upper_xnn], [lower_xnn, upper_xnn],
                        color=[0.7, 0, 0], linestyle='dotted'
                        )
                    axes[1, uc_index].set_xlim([lower_xnn, upper_xnn])
                    axes[1, uc_index].set_ylim([lower_xnn, upper_xnn])

                error_train = np.sort(train_xnn_error[:, uc_index])
                error_train = error_train[~np.isnan(error_train)]
                xnn_p25_train = error_train[int(0.25 * error_train.size)]
                xnn_p50_train = error_train[int(0.50 * error_train.size)]
                xnn_p75_train = error_train[int(0.75 * error_train.size)]
                xnn_rmse_train = np.sqrt(1/error_train.size * np.linalg.norm(error_train)**2)
                error_val = np.sort(val_xnn_error[:, uc_index])
                error_val = error_val[~np.isnan(error_val)]
                xnn_p25_val = error_val[int(0.25 * error_val.size)]
                xnn_p50_val = error_val[int(0.50 * error_val.size)]
                xnn_p75_val = error_val[int(0.75 * error_val.size)]
                xnn_rmse_val = np.sqrt(1/error_val.size * np.linalg.norm(error_val)**2)
                xnn_error_titles = [
                    xnn_titles[uc_index],
                    f'RMSE: {100 * xnn_rmse_train:0.4f} / {100 * xnn_rmse_val:0.4f}',
                    f'25%: {100 * xnn_p25_train:0.4f} / {100 * xnn_p25_val:0.4f}',
                    f'50%: {100 * xnn_p50_train:0.4f} / {100 * xnn_p50_val:0.4f}',
                    f'75%: {100 * xnn_p75_train:0.4f} / {100 * xnn_p75_val:0.4f}',
                    ]
                axes[1, uc_index].set_title('\n'.join(xnn_error_titles), fontsize=12)

                axes[0, uc_index].set_xlabel('True')
                axes[1, uc_index].set_xlabel('True')
            axes[0, 0].set_ylabel('Predicted')
            axes[1, 0].set_ylabel('Predicted')
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/{self.group}_template_reg_eval_{self.template_params["tag"]}_{save_label}.png')
            plt.close()
