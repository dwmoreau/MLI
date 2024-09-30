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
        return xnn, residuals

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
        xnn_templates_all, q2_error = self.generate_xnn_fast(q2_obs)
        probability_templates = self.hgbc_classifier.predict_proba(
            q2_error[:, :self.template_params['calibration_n_peaks']]
            )
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
            return self.generate_calibrated(n_templates, rng, q2_obs)
        else:
            return self.generate_uncalibrated(n_templates, rng, q2_obs)

    def _get_inputs_worker(self, inputs):
        q2_obs = inputs[0]
        xnn_true = inputs[1]
        xnn, q2_error = self.generate_xnn_fast(q2_obs)
        distance = scipy.spatial.distance.cdist(xnn, xnn_true[np.newaxis])
        neighbor = (distance < self.template_params['radius']).astype(int)[:, 0]
        return q2_error, neighbor, distance

    def get_inputs(self, data, n_entries):
        q2_obs = np.stack(data['q2'])
        xnn_true = np.stack(data['reindexed_xnn'])[:, self.unit_cell_indices]

        q2_error = []
        neighbor = []
        distance = []
        if n_entries is None:
            n_entries = len(data)
            indices = np.arange(n_entries)
        else:
            n_entries = min(n_entries, len(data))
            indices = self.rng.choice(len(data), n_entries, replace=False)

        if self.template_params['parallelization'] is None:
            print(f'Setting up {n_entries} entries serially')
            for index in tqdm(indices):
                q2_error_entry, neighbor_entry, distance_entry = \
                    self._get_inputs_worker([q2_obs[index], xnn_true[index]])
                q2_error.append(q2_error_entry)
                neighbor.append(neighbor_entry)
                distance.append(distance_entry)
        elif self.template_params['parallelization'] == 'multiprocessing':
            print(f'Setting up {n_entries} entries using multiprocessing')
            with multiprocessing.Pool(self.template_params['n_processes']) as p:
                outputs = p.map(self._get_inputs_worker, zip(q2_obs[indices], xnn_true[indices]))
            for i in range(n_entries):
                q2_error.append(outputs[i][0])
                neighbor.append(outputs[i][1])
                distance.append(outputs[i][2])

        q2_error = np.row_stack(q2_error)
        neighbor = np.concatenate(neighbor)
        distance = np.concatenate(distance)
        return q2_error, neighbor, distance

    def calibrate_templates(self, data):
        unaugmented_data = data[~data['augmented']]
        training_data = unaugmented_data[unaugmented_data['train']]
        val_data = unaugmented_data[~unaugmented_data['train']]
        q2_error_train, neighbor_train, distance_train = self.get_inputs(
            training_data, self.template_params['n_train']
            )
        q2_error_val, neighbor_val, distance_val = self.get_inputs(
            val_data, int(0.2*self.template_params['n_train'])
            )
        #np.save('MICal_test_q2_error_train.npy', q2_error_train)
        #np.save('MICal_test_q2_error_val.npy', q2_error_val)
        #np.save('MICal_test_neighbor_train.npy', neighbor_train)
        #np.save('MICal_test_neighbor_val.npy', neighbor_val)
        #np.save('MICal_test_distance_train.npy', distance_train)
        #np.save('MICal_test_distance_val.npy', distance_val)

        #q2_error_train = np.load('MICal_test_q2_error_train.npy') 
        #q2_error_val = np.load('MICal_test_q2_error_val.npy') 
        #neighbor_train = np.load('MICal_test_neighbor_train.npy') 
        #neighbor_val = np.load('MICal_test_neighbor_val.npy') 
        #distance_train = np.load('MICal_test_distance_train.npy') 
        #distance_val = np.load('MICal_test_distance_val.npy') 

        print('Fitting Hist Boosting Model')  
        self.hgbc_classifier = sklearn.ensemble.HistGradientBoostingClassifier(
            max_depth=self.template_params['max_depth'],
            min_samples_leaf=self.template_params['min_samples_leaf'],
            class_weight='balanced',
            l2_regularization=self.template_params['l2_regularization'],
            )
        self.hgbc_classifier.fit(
            q2_error_train[:, :self.template_params['calibration_n_peaks']],
            neighbor_train
            )
        probability_train = self.hgbc_classifier.predict_proba(
            q2_error_train[:, :self.template_params['calibration_n_peaks']]
            )
        score_train = self.hgbc_classifier.score(
            q2_error_train[:, :self.template_params['calibration_n_peaks']],
            neighbor_train
            )
        probability_val = self.hgbc_classifier.predict_proba(
            q2_error_val[:, :self.template_params['calibration_n_peaks']]
            )
        score_val = self.hgbc_classifier.score(
            q2_error_val[:, :self.template_params['calibration_n_peaks']],
            neighbor_val
            )

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
