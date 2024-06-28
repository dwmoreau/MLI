import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.spatial
import sklearn.linear_model
from tqdm import tqdm

from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_M20
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
        self.n_points = data_params['n_points']
        self.n_outputs = data_params['n_outputs']
        self.y_indices = data_params['y_indices']
        self.hkl_ref_length = data_params['hkl_ref_length']
        self.hkl_ref = hkl_ref
        self.group = group
        self.save_to = save_to
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def _save_common(self):
        write_params(
            self.template_params,
            f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv'
            )

    def save(self):
        self._save_common()

    def _load_from_tag_common(self):
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
            'n_templates'
            ]
        self.template_params = dict.fromkeys(params_keys)
        self.template_params['tag'] = params['tag']
        self.template_params['templates_per_dominant_zone_bin'] = int(params['templates_per_dominant_zone_bin'])
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]

    def load_from_tag(self):
        self._load_from_tag_common()

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
                self.n_points,
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
                            self.n_points,
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
                            self.n_points,
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
            hkl_information_hist = np.bincount(hkl_information, minlength=self.n_points)

            unit_cell_volume = np.array(training_data['reindexed_volume'])
            sorted_unit_cell_volume = np.sort(unit_cell_volume)
            volume_bins = np.linspace(
                sorted_unit_cell_volume[int(0.001*sorted_unit_cell_volume.size)],
                sorted_unit_cell_volume[int(0.999*sorted_unit_cell_volume.size)],
                11
                )

            mean_ratio = np.zeros((self.n_points, 2, 2))
            for i in range(self.n_points):
                mean_ratio[i, 0, 0] = np.mean(ratio_xnn[hkl_information == i])
                mean_ratio[i, 1, 0] = np.std(ratio_xnn[hkl_information == i])
                mean_ratio[i, 0, 1] = np.mean(ratio_unit_cell[hkl_information == i])
                mean_ratio[i, 1, 1] = np.std(ratio_unit_cell[hkl_information == i])

            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            axes[0].hist(ratio_xnn, bins=np.linspace(0, 1, self.n_points + 1))
            axes[1].hist(ratio_unit_cell, bins=np.linspace(0, 1, self.n_points + 1))
            axes[2].hist(ratio_rec_unit_cell, bins=np.linspace(0, 1, self.n_points + 1))
            axes[3].bar(np.arange(self.n_points), hkl_information_hist, width=1)
            axes[4].plot(
                hkl_information, ratio_unit_cell,
                marker='.', linestyle='none', markersize=0.25, alpha=0.5
                )
            axes[4].errorbar(np.arange(self.n_points), mean_ratio[:, 0, 1], mean_ratio[:, 1, 1])

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
            for i in range(self.n_points):
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
                                        self.n_points,
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
                            self.n_points,
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
        self.save()

    def generate_xnn_fast(self, q2_obs):
        # This is still slow
        # original I just did linear least squares iteratively until q2_calc was ordered (1st for loop)
        # It is faster to use Gauss-Newton non-linear least squares (2nd for loop)
        # This is mostly copied from TargetFunctions.py
        # I forgot the reason, but I don't use this function. It might not be numerically stable

        hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)

        # Calculate initial values for xnn using linear least squares methods
        xnn = np.zeros((self.template_params['n_templates'], self.n_outputs))
        A = hkl2 / q2_obs[np.newaxis, :, np.newaxis]
        b = np.ones(self.n_points)
        for template_index in range(self.template_params['n_templates']):
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
            good = np.linalg.matrix_rank(H, hermitian=True) == self.n_outputs
            delta_gn = np.zeros((self.template_params['n_templates'], self.n_outputs))
            delta_gn[good] = -np.matmul(np.linalg.inv(H[good]), dloss_dxnn[good, :, np.newaxis])[:, :, 0]
            xnn += delta_gn
        loss = np.linalg.norm(1 - q2_calc/q2_obs[np.newaxis], axis=1)
        
        xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
        return xnn, loss

    def generate_xnn(self, q2_obs):
        # This is slower
        xnn = np.zeros((self.template_params['n_templates'], self.n_outputs))
        loss = np.zeros(self.template_params['n_templates'])
        order = np.arange(self.n_points)
        hkl2_all = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)
        for template_index in range(self.template_params['n_templates']):
            sigma = q2_obs
            hkl2 = hkl2_all[template_index]
            status = True
            i = 0
            xnn_last = np.zeros(self.n_outputs)
            while status:
                # Using this is only slightly faster than np.linalg.lstsq
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
            loss[template_index] = np.linalg.norm(1 - q2_calc/q2_obs)
        xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
        return xnn, loss

    def do_predictions(self, q2_obs, n_templates='all'):
        # This is primary used to generate candidates for the optimizer, which expects that the
        # sum of the template and sampled candidates to be the same. This is why n_samples gets
        # changed
        xnn_templates, _ = self.generate_xnn(q2_obs)
        if n_templates == 'all':
            pass
        elif n_templates < xnn_templates.shape[0]:
            # requesting fewer templates than in the set
            # subsample
            indices = self.rng.choice(
                xnn_templates.shape[0],
                size=n_templates,
                replace=False,
                p=self.miller_index_templates_prob,
                )
            xnn_templates =  xnn_templates[indices]
        elif n_templates > xnn_templates.shape[0]:
            # requesting more templates than in the set
            # Just sample multiple times
            print('WARNING: Requesting more templates than available. Duplicates will be returned')
            difference = n_templates - xnn_templates.shape[0]
            if difference > xnn_templates.shape[0]:
                replace = True
            else:
                replace = False
            indices = self.rng.choice(
                xnn_templates.shape[0],
                size=difference,
                replace=replace,
                p=self.miller_index_templates_prob,
                )
            xnn_templates =  np.concatenate((xnn_templates, xnn_templates[indices]), axis=0)

        unit_cell_templates = get_unit_cell_from_xnn(
            xnn_templates, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return unit_cell_templates


class MITemplates_calibrated(MITemplates):
    def __init__(self, group, data_params, template_params, hkl_ref, save_to, seed):
        super().__init__(group, data_params, template_params, hkl_ref, save_to, seed)
        template_params_defaults = {
            'parallelization': None,
            'n_processes': None,
            'inverse_regularization_strength': 1,
            'radius': 0.003,
            'q2_error_params': [0.0001, 0.001],
            'n_train': 100,
            'M20_evals': 10,
            }

        for key in template_params_defaults.keys():
            if key not in self.template_params.keys():
                self.template_params[key] = template_params_defaults[key]

        if self.template_params['parallelization'] == 'multiprocessing':
            import multiprocessing
        elif self.template_params['parallelization'] == 'message_passing':
            assert False

    def save(self):
        self._save_common()
        joblib.dump(
            self.logistic_regression,
            f'{self.save_to}/{self.group}_template_calibrator_{self.template_params["tag"]}.bin'
            )

    def load_from_tag(self):
        self._load_from_tag_common()
        self.logistic_regression = joblib.load(
            f'{self.save_to}/{self.group}_template_calibrator_{self.template_params["tag"]}.bin'
            )
        params = read_params(f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv')
        self.template_params['parallelization'] = params['parallelization']
        self.template_params['n_processes'] = int(params['n_processes'])
        self.template_params['n_train'] = int(params['n_train'])
        self.template_params['inverse_regularization_strength'] = \
            float(params['inverse_regularization_strength'])
        self.template_params['radius'] = float(params['radius'])
        self.template_params['q2_error_params'] = np.array(
            params['q2_error_params'].split('[')[1].split(']')[0].split(',')
            )

    def generate_xnn_M20(self, q2_obs):
        xnn = np.zeros((self.template_params['n_templates'], self.n_outputs))
        loss = np.zeros(self.template_params['n_templates'])
        q2_calc_all = np.zeros((self.template_params['n_templates'], self.n_points))
        order = np.arange(self.n_points)
        hkl2_all = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)
        for template_index in range(self.template_params['n_templates']):
            sigma = q2_obs
            hkl2 = hkl2_all[template_index]
            status = True
            i = 0
            xnn_last = np.zeros(self.n_outputs)
            while status:
                # Using this is only slightly faster than np.linalg.lstsq
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
            loss[template_index] = np.linalg.norm(1 - q2_calc/q2_obs)
            q2_calc_all[template_index] = q2_calc

        xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
        hkl2_ref = get_hkl_matrix(self.hkl_ref, self.lattice_system)
        q2_ref_calc_all = xnn @ hkl2_ref.T
        # Need to add some error to the peak positions to calculate M20.
        # Otherwise the observed discrepancy (M20 denominator) can be zero.
        M20 = np.zeros((self.template_params['n_templates'], self.template_params['M20_evals']))
        scale = self.template_params['q2_error_params'][0] + q2_obs*self.template_params['q2_error_params'][1]
        for M20_iteration in range(self.template_params['M20_evals']):
            q2_obs_err = q2_obs + self.rng.normal(loc=0, scale=scale)
            print(np.mean(np.abs(q2_obs_err - q2_obs)), M20_iteration)
            M20[:, M20_iteration] = get_M20(q2_obs_err, q2_calc_all, q2_ref_calc_all)
        return xnn, loss, M20.mean(axis=1)

    def _get_inputs_worker(self, q2_obs, xnn_true):
        xnn, loss, M20 = self.generate_xnn_M20(q2_obs)
        print(M20.mean(), M20.max(), M20.min())
        normalized_loss = loss.min()/loss
        normalized_M20 = M20 / M20.max()
        distance = scipy.spatial.distance.cdist(xnn, xnn_true[np.newaxis])
        neighbor = (distance < self.template_params['radius']).astype(int)[:, 0]
        return normalized_loss, normalized_M20, neighbor

    def get_inputs(self, data):
        training_data = data[data['train']]
        n_entries = len(training_data)
        metrics = np.zeros((n_entries, 2))
        q2_obs = np.stack(training_data['q2'])
        xnn_true = np.stack(training_data['reindexed_xnn'])[:, self.y_indices]

        normalized_loss = []
        normalized_M20 = []
        neighbor = []
        if self.template_params['n_train'] is None:
            n_train = len(training_data)
        else:
            n_train = min(self.template_params['n_train'], len(training_data))

        if self.template_params['parallelization'] is None:
            print(f'Setting up {n_train} entries serially')
            for i in tqdm(range(n_train)):
                normalized_loss_entry, normalized_M20_entry, neighbor_entry = \
                    self._get_inputs_worker(q2_obs[i], xnn_true[i])
                normalized_loss.append(normalized_loss_entry)
                normalized_M20.append(normalized_M20_entry)
                neighbor.append(neighbor_entry)
        elif self.template_params['parallelization'] == 'multiprocessing':
            print(f'Setting up {n_train} entries using multiprocessing')
            assert False
        elif self.template_params['parallelization'] == 'message_passing':
            assert False

        features = np.column_stack((
            np.concatenate(normalized_loss),
            np.concatenate(normalized_M20)
            ))
        neighbor = np.concatenate(neighbor)
        return features, neighbor

    def fit_model(self, data):
        features, neighbor = self.get_inputs(data)
        self.logistic_regression = sklearn.linear_model.LogisticRegression(
            C=self.template_params['inverse_regularization_strength'], verbose=1
            )
        self.logistic_regression.fit(features, neighbor)
        self.save()
