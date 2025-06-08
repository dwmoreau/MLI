import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import scipy.spatial
import sklearn.ensemble
from sklearn.model_selection import GridSearchCV

from numba_functions import fast_assign
from TargetFunctions import CandidateOptLoss
from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_M20_sym_reversed
from Utilities import get_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import get_reciprocal_unit_cell_from_xnn
from Utilities import reciprocal_uc_conversion
from Utilities import Q2Calculator
from Utilities import get_M20_likelihood
from IOManagers import read_params
from IOManagers import write_params
from IOManagers import SKLearnManager


class MITemplates:
    def __init__(self, bravais_lattice, data_params, template_params, hkl_ref, save_to, seed):
        self.lattice_system = data_params['lattice_system']
        self.unit_cell_length = data_params['unit_cell_length']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.hkl_ref_length = data_params['hkl_ref_length']
        self.hkl_ref = hkl_ref
        self.bravais_lattice = bravais_lattice
        self.save_to = save_to
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.template_params = template_params
        template_params_defaults = {
            'templates_per_dominant_zone_bin': 2000,
            'parallelization': 'multiprocessing',
            'n_processes': 4,
            'n_entries_train': 1000,
            'n_instances_train': 1000000,
            'loss': 'squared_error',
            'learning_rate': 0.1,
            'max_leaf_nodes': 31,
            'max_depth': 4,
            'min_samples_leaf': 100,
            'l2_regularization': 10,
            'n_peaks': data_params['n_peaks'],
            'n_peaks_template': min(10, data_params['n_peaks']),
            'n_peaks_calibration': min(20, data_params['n_peaks']),
            'max_distance': 0.05,
            'grid_search': None,
            }
        for key in template_params_defaults.keys():
            if key not in self.template_params.keys():
                self.template_params[key] = template_params_defaults[key]

    def save(self, train_inputs):
        write_params(
            self.template_params,
            os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_template_params_{self.template_params["tag"]}.csv'
                )
            )
        model_manager = SKLearnManager(
            filename=os.path.join(
                f'{self.save_to}', 
                f'{self.bravais_lattice}_template_regressor_{self.template_params["tag"]}'
                ),
            model_type='onnx'
            )
        model_manager.save(
            model=self.hgbc_regressor,
            n_features=2 + self.template_params['n_peaks_calibration'],
            )
        model_manager._save_sklearn(
            model=self.hgbc_regressor,
            )

    def load_from_tag(self):
        self.miller_index_templates = np.load(os.path.join(
            f'{self.save_to}',
            f'{self.bravais_lattice}_miller_index_templates_{self.template_params["tag"]}.npy'
            ))
        self.miller_index_templates_prob = np.load(os.path.join(
            f'{self.save_to}',
            f'{self.bravais_lattice}_miller_index_templates_prob_{self.template_params["tag"]}.npy',
            ))
        params = read_params(os.path.join(
            f'{self.save_to}',
            f'{self.bravais_lattice}_template_params_{self.template_params["tag"]}.csv'
            ))
        params_keys = [
            'tag',
            'templates_per_dominant_zone_bin',
            'n_templates',
            'parallelization',
            'n_processes',
            'n_entries_train',
            'n_instances_train',
            'max_depth',
            'min_samples_leaf',
            'l2_regularization',
            'n_peaks',
            'n_peaks_template',
            'n_peaks_calibration',
            ]
        self.template_params = dict.fromkeys(params_keys)
        self.template_params['tag'] = params['tag']
        self.template_params['templates_per_dominant_zone_bin'] = int(params['templates_per_dominant_zone_bin'])
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]
        if self.lattice_system == 'cubic':
            self.template_params['n_peaks'] = 10
        else:
            self.template_params['n_peaks'] = 20

        self.hgbc_regressor = SKLearnManager(
            filename=os.path.join(
                f'{self.save_to}', 
                f'{self.bravais_lattice}_template_regressor_{self.template_params["tag"]}'
                ),
            model_type='onnx'
            )
        self.hgbc_regressor.load()

        self.template_params['parallelization'] = params['parallelization']
        self.template_params['n_processes'] = int(params['n_processes'])
        self.template_params['n_entries_train'] = int(params['n_entries_train'])
        self.template_params['n_instances_train'] = int(params['n_instances_train'])
        self.template_params['max_depth'] = int(params['max_depth'])
        self.template_params['min_samples_leaf'] = int(params['min_samples_leaf'])
        self.template_params['l2_regularization'] = float(params['l2_regularization'])
        self.template_params['n_peaks_template'] = int(params['n_peaks_template'])
        self.template_params['n_peaks_calibration'] = int(params['n_peaks_calibration'])

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
                self.template_params['n_peaks'],
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
            fig.savefig(os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_dominant_zone_ratio_{self.template_params["tag"]}.png'
                ))
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
                            self.template_params['n_peaks'],
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
                            self.template_params['n_peaks'],
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
            hkl_information_hist = np.bincount(hkl_information, minlength=self.template_params['n_peaks'])

            unit_cell_volume = np.array(training_data['reindexed_volume'])
            sorted_unit_cell_volume = np.sort(unit_cell_volume)
            volume_bins = np.linspace(
                sorted_unit_cell_volume[int(0.001*sorted_unit_cell_volume.size)],
                sorted_unit_cell_volume[int(0.999*sorted_unit_cell_volume.size)],
                11
                )

            mean_ratio = np.zeros((self.template_params['n_peaks'], 2, 2))
            for i in range(self.template_params['n_peaks']):
                mean_ratio[i, 0, 0] = np.mean(ratio_xnn[hkl_information == i])
                mean_ratio[i, 1, 0] = np.std(ratio_xnn[hkl_information == i])
                mean_ratio[i, 0, 1] = np.mean(ratio_unit_cell[hkl_information == i])
                mean_ratio[i, 1, 1] = np.std(ratio_unit_cell[hkl_information == i])

            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            axes[0].hist(ratio_xnn, bins=np.linspace(0, 1, self.template_params['n_peaks'] + 1))
            axes[1].hist(ratio_unit_cell, bins=np.linspace(0, 1, self.template_params['n_peaks'] + 1))
            axes[2].hist(ratio_rec_unit_cell, bins=np.linspace(0, 1, self.template_params['n_peaks'] + 1))
            axes[3].bar(np.arange(self.template_params['n_peaks']), hkl_information_hist, width=1)
            axes[4].plot(
                hkl_information, ratio_unit_cell,
                marker='.', linestyle='none', markersize=0.25, alpha=0.5
                )
            axes[4].errorbar(np.arange(self.template_params['n_peaks']), mean_ratio[:, 0, 1], mean_ratio[:, 1, 1])

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
            fig.savefig(os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_dominant_zone_ratio_{self.template_params["tag"]}.png'
                ))
            plt.close()

            mi_sets = []
            sampling_probability = []
            n_ratio_bins = 10
            ratio_bins = np.linspace(0, 1, n_ratio_bins + 1)
            templates_per_information_bin = self.template_params['templates_per_dominant_zone_bin']
            templates_per_dominant_zone_bin = int(self.template_params['templates_per_dominant_zone_bin'] / n_ratio_bins)
            for i in range(self.template_params['n_peaks']):
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
                                        self.template_params['n_peaks'],
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
                            self.template_params['n_peaks'],
                            hkl_labels_bin,
                            self.hkl_ref_length,
                            self.rng
                            ))
            miller_index_templates = np.row_stack(mi_sets)
            sampling_probability = np.concatenate(sampling_probability)

        # Miller index templates are generated using all available peaks
        # The number of peaks per templates are reduced for templates here.
        # Then only the unique templates are retained.
        self.miller_index_templates, unique_indices = np.unique(
            miller_index_templates[:, :self.template_params['n_peaks_template']],
            axis=0, return_index=True
            )
        sampling_probability = sampling_probability[unique_indices]
        self.miller_index_templates_prob = sampling_probability / sampling_probability.sum()
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]
        np.save(
            os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_miller_index_templates_{self.template_params["tag"]}.npy'
                ),
            self.miller_index_templates
            )
        np.save(
            os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_miller_index_templates_prob_{self.template_params["tag"]}.npy'
                ),
            self.miller_index_templates_prob
            )

    def setup(self, data):
        self.setup_templates(data)
        train_inputs = self.calibrate_templates(data)
        self.save(train_inputs)

    def generate_xnn(self, q2_obs, indices=None):
        if indices is None:
            hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)
            n_templates = self.template_params['n_templates']
        else:
            hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates[indices]], self.lattice_system)
            n_templates = indices.size

        # q2_calc should increase monotonically. Sort hkl2 then re-solve for xnn iteratively.
        q2_obs_template = q2_obs[:self.template_params['n_peaks_template']]
        q2_obs_calibration = q2_obs[:self.template_params['n_peaks_calibration']]
        xnn = np.zeros((n_templates, self.unit_cell_length))
        sigma = q2_obs_template[np.newaxis]
        hessian_prefactor = (1 / sigma**2)[:, :, np.newaxis, np.newaxis]
        term0 = np.matmul(hkl2[:, :, :, np.newaxis], hkl2[:, :, np.newaxis, :])
        H = np.sum(hessian_prefactor * term0, axis=1)
        good = np.linalg.matrix_rank(H, hermitian=True) == self.unit_cell_length
        xnn = xnn[good]
        hkl2 = hkl2[good]
        for index in range(5):
            q2_calc = (hkl2 @ xnn[:, :, np.newaxis])[:, :, 0]
            if index != 0:
                sort_indices = q2_calc.argsort(axis=1)
                q2_calc = np.take_along_axis(q2_calc, sort_indices, axis=1)
                hkl2 = np.take_along_axis(hkl2, sort_indices[:, :, np.newaxis], axis=1)

            residuals = (q2_calc - q2_obs_template[np.newaxis]) / sigma
            dlikelihood_dq2_pred = residuals / sigma
            dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * hkl2, axis=1)
            term0 = np.matmul(hkl2[:, :, :, np.newaxis], hkl2[:, :, np.newaxis, :])
            H = np.sum(hessian_prefactor * term0, axis=1)
            delta_gn = -np.matmul(np.linalg.inv(H), dloss_dxnn[:, :, np.newaxis])[:, :, 0]
            xnn += delta_gn
            xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)

        # Now prepare each template for calibration, which does not involve the same
        # number of peaks as the templates.
        # First, find the best Miller index assignments using all calibration peaks
        q2_calculator = Q2Calculator(
            lattice_system=self.lattice_system,
            hkl=self.hkl_ref[:, :self.template_params['n_peaks_calibration']],
            tensorflow=False,
            representation='xnn'
            )
        q2_ref_calc = q2_calculator.get_q2(xnn)
        hkl_assign_calibration = fast_assign(q2_obs_calibration, q2_ref_calc)
        # Now remove templates that have non-unique Miller index assignments up to n_peaks_template
        hkl_assign_template, unique_indices = np.unique(
            hkl_assign_calibration[:, :self.template_params['n_peaks_template']],
            axis=0, return_index=True
            )
        n_templates = unique_indices.size
        xnn = xnn[unique_indices]
        hkl_assign_calibration = hkl_assign_calibration[unique_indices]
        hkl_template = np.take(
            self.hkl_ref[:, :self.template_params['n_peaks_template']], hkl_assign_template, axis=0
            )
        hkl_calibration = np.take(
            self.hkl_ref[:, :self.template_params['n_peaks_calibration']], hkl_assign_calibration, axis=0
            )

        # Second, update the unit cell given the assignments up to n_template_peaks
        target_function = CandidateOptLoss(
            np.repeat(q2_obs_template[np.newaxis], n_templates, axis=0), 
            lattice_system=self.lattice_system,
            )
        target_function.update(hkl_template[:, :self.template_params['n_peaks_template']], xnn)
        xnn += target_function.gauss_newton_step(xnn)
        xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
        hkl2 = get_hkl_matrix(hkl_calibration, self.lattice_system)
        q2_calc = (hkl2 @ xnn[:, :, np.newaxis])[:, :, 0]
        residuals = (q2_calc - q2_obs_calibration[np.newaxis]) / q2_obs_calibration[np.newaxis]

        # Third, downsample to removes redundant unit cells
        # Downsampling happens in chunks of xnn sorted by reciprocal space volume.
        # If the chunk size is large, downsampling is extremely slow.
        # If the chunk size is small, not enough redundant lattices get removed
        # Running it twice with a small chunk size removes more lattices
        # while also being reasonably fast.
        for _ in range(2):
            xnn, q2_calc = self.downsample_candidates(xnn, q2_calc, residuals)
        # Third, calculate the values needed for calibration
        reciprocal_volume = get_unit_cell_volume(get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            ), partial_unit_cell=True, lattice_system=self.lattice_system)
        q2_ref_calc = q2_calculator.get_q2(xnn)
        q2_calc_max = q2_calc.max(axis=1)
        N_pred = np.count_nonzero(q2_ref_calc < q2_calc_max[:, np.newaxis], axis=1)
        _, probability, _ = get_M20_likelihood(
            q2_obs=q2_obs_calibration,
            q2_calc=q2_calc,
            bravais_lattice=self.bravais_lattice,
            reciprocal_volume=reciprocal_volume
            )
        return xnn, probability, N_pred, q2_calc_max

    def generate(self, n_templates, rng, q2_obs):
        xnn_templates_all, probability, N_pred, q2_calc_max = self.generate_xnn(q2_obs)
        if n_templates == 'all':
            xnn_templates = xnn_templates_all
        elif n_templates <= xnn_templates_all.shape[0]:
            _, unique_indices = np.unique(
                np.round(xnn_templates_all, decimals=6), return_index=True, axis=0
                )
            xnn_templates_all = xnn_templates_all[unique_indices]
            probability = probability[unique_indices]
            N_pred = N_pred[unique_indices]
            q2_calc_max = q2_calc_max[unique_indices]

            inputs = np.concatenate((
                probability,
                N_pred[:, np.newaxis],
                q2_calc_max[:, np.newaxis],
                ), axis=1).astype(np.float32)
            success_pred_templates = self.hgbc_regressor.predict(inputs)[:, 0]    
            top_n_indices = np.argsort(success_pred_templates)[::-1][:n_templates]
            xnn_templates = xnn_templates_all[top_n_indices]
        elif n_templates > xnn_templates_all.shape[0]:
            # requesting more templates than in the set
            # Just sample multiple times
            n_replicates = n_templates // xnn_templates_all.shape[0]
            n_extra = n_templates % xnn_templates_all.shape[0]
            if n_replicates > 1:
                replicates = np.concatenate([xnn_templates_all for _ in range(n_replicates)], axis=0)
            else:
                replicates = xnn_templates_all
            extra = xnn_templates_all[:n_extra]
            xnn_templates =  np.concatenate((replicates, extra), axis=0)
            
        unit_cell_templates = get_unit_cell_from_xnn(
            xnn_templates, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return unit_cell_templates

    def downsample_candidates(self, xnn, q2_calc, residuals):
        chunk_size = 250
        n_chunks = xnn.shape[0] // chunk_size + 1

        reciprocal_volume = get_unit_cell_volume(get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            ), partial_unit_cell=True, lattice_system=self.lattice_system)
        sort_indices = np.argsort(reciprocal_volume)
        xnn = xnn[sort_indices]
        q2_calc = q2_calc[sort_indices]
        residuals = residuals[sort_indices]
        error = np.linalg.norm(residuals, axis=1)

        xnn_downsampled = []
        q2_calc_downsampled = []
        for chunk_index in range(n_chunks):
            if chunk_index == n_chunks - 1:
                xnn_chunk = xnn[chunk_index * chunk_size:]
                q2_calc_chunk = q2_calc[chunk_index * chunk_size:]
                error_chunk = error[chunk_index * chunk_size:]
            else:
                xnn_chunk = xnn[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                q2_calc_chunk = q2_calc[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                error_chunk = error[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
            status = True
            while status:
                distance = scipy.spatial.distance.cdist(xnn_chunk, xnn_chunk)
                neighbor_array = distance < 0.000001
                neighbor_count = np.sum(neighbor_array, axis=1)
                if neighbor_count.size > 0 and neighbor_count.max() > 1:
                    highest_density_index = np.argmax(neighbor_count)
                    neighbor_indices = np.where(neighbor_array[highest_density_index])[0]
                    best_neighbor = np.argmin(error[neighbor_indices])
                    xnn_best_neighbor = xnn_chunk[neighbor_indices][best_neighbor]
                    q2_calc_best_neighbor = q2_calc_chunk[neighbor_indices][best_neighbor]
                    error_best_neighbor = error_chunk[neighbor_indices][best_neighbor]
                    xnn_chunk = np.row_stack((
                        np.delete(xnn_chunk, neighbor_indices, axis=0), 
                        xnn_best_neighbor
                        ))
                    q2_calc_chunk = np.row_stack((
                        np.delete(q2_calc_chunk, neighbor_indices, axis=0), 
                        q2_calc_best_neighbor
                        ))
                    error_chunk = np.concatenate((
                        np.delete(error_chunk, neighbor_indices), 
                        [error_best_neighbor]
                        ))
                else:
                    status = False
            xnn_downsampled.append(xnn_chunk)
            q2_calc_downsampled.append(q2_calc_chunk)
        xnn_downsampled = np.row_stack(xnn_downsampled)
        q2_calc_downsampled = np.row_stack(q2_calc_downsampled)
        return xnn_downsampled, q2_calc_downsampled

    def _get_inputs_worker(self, inputs):
        q2_obs = inputs[0]
        xnn_true = inputs[1]
        xnn, probability, N_pred, q2_calc_max = self.generate_xnn(q2_obs)
        distance = scipy.spatial.distance.cdist(xnn, xnn_true[np.newaxis])
        return probability, distance, xnn, N_pred, q2_calc_max

    def get_inputs(self, data, n_entries):
        from tqdm import tqdm
        q2_obs = np.stack(data['q2'])
        probability = []
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
                probability_entry, distance_entry, xnn_entry, N_pred_entry, q2_calc_max_entry = \
                    self._get_inputs_worker([q2_obs[index], xnn_true[index]])
                probability.append(probability_entry)
                distance.append(distance_entry)
                xnn.append(xnn_entry)
                N_pred.append(N_pred_entry)
                q2_calc_max.append(q2_calc_max_entry)
        elif self.template_params['parallelization'] == 'multiprocessing':
            print(f'Setting up {n_entries} entries using multiprocessing')
            with multiprocessing.Pool(self.template_params['n_processes']) as p:
                outputs = p.map(self._get_inputs_worker, zip(q2_obs[indices], xnn_true[indices]))
            for i in range(n_entries):
                probability.append(outputs[i][0])
                distance.append(outputs[i][1])
                xnn.append(outputs[i][2])
                N_pred.append(outputs[i][3])
                q2_calc_max.append(outputs[i][4])

        probability = np.row_stack(probability)
        distance = np.concatenate(distance)
        xnn = np.row_stack(xnn)
        N_pred = np.concatenate(N_pred)
        q2_calc_max = np.concatenate(q2_calc_max)
        return probability, distance, xnn, xnn_true[indices], N_pred, q2_calc_max

    def calibrate_templates(self, data):
        unaugmented_data = data[~data['augmented']]
        training_data = unaugmented_data[unaugmented_data['train']]
        val_data = unaugmented_data[~unaugmented_data['train']]
        n_val = int(0.2*self.template_params['n_entries_train'])
        probability_train, distance_train, xnn_train_pred, xnn_train_true, N_pred_train, q2_calc_max_train = \
            self.get_inputs(training_data, self.template_params['n_entries_train'])
        probability_val, distance_val, xnn_val_pred, xnn_val_true, N_pred_val, q2_calc_max_val = \
            self.get_inputs(val_data, n_val)

        # Select only training and validation entries that are within 0.01 1/A**2
        # of the correct values.
        # distance_train has shape Nx1
        train_select = distance_train[:, 0] < self.template_params['max_distance']
        probability_train = probability_train[train_select]
        distance_train = distance_train[train_select]
        N_pred_train = N_pred_train[train_select]
        q2_calc_max_train = q2_calc_max_train[train_select]
        if distance_train.size > self.template_params['n_instances_train']:
            train_select = self.rng.choice(
                distance_train.size,
                size=self.template_params['n_instances_train'],
                replace=False
                )
            probability_train = probability_train[train_select]
            distance_train = distance_train[train_select]
            N_pred_train = N_pred_train[train_select]
            q2_calc_max_train = q2_calc_max_train[train_select]

        val_select = distance_val[:, 0] < self.template_params['max_distance']
        probability_val = probability_val[val_select]
        distance_val = distance_val[val_select]
        N_pred_val = N_pred_val[val_select]
        q2_calc_max_val = q2_calc_max_val[val_select]
        if distance_val.size > self.template_params['n_instances_train']:
            val_select = self.rng.choice(
                distance_val.size,
                size=self.template_params['n_instances_train'],
                replace=False
                )
            probability_val = probability_val[val_select]
            distance_val = distance_val[val_select]
            N_pred_val = N_pred_val[val_select]
            q2_calc_max_val = q2_calc_max_val[val_select]

        # Convert to float32 ensures that the ONNX conversion and the sklearn model
        # provide the same inferences.
        train_inputs = np.concatenate((
            probability_train,
            N_pred_train[:, np.newaxis],
            q2_calc_max_train[:, np.newaxis],
            ), axis=1).astype(np.float32)
        val_inputs = np.concatenate((
            probability_val,
            N_pred_val[:, np.newaxis],
            q2_calc_max_val[:, np.newaxis], 
            ), axis=1).astype(np.float32)

        roc = np.load(self.template_params['roc_file_name'].replace('!!', self.bravais_lattice))
        distance_convergence = roc[0]
        success_rate = roc[1]
        indices_train = np.searchsorted(distance_convergence, distance_train)
        indices_train[indices_train < 0] = 0
        indices_train[indices_train >= success_rate.size] = success_rate.size - 1
        train_outputs = success_rate[indices_train].ravel()
        indices_val = np.searchsorted(distance_convergence, distance_val)
        indices_val[indices_val < 0] = 0
        indices_val[indices_val >= success_rate.size] = success_rate.size - 1
        val_outputs = success_rate[indices_val].ravel()

        print('Fitting Hist Boosting Model')
        if self.template_params['grid_search'] is None:
            print('Instantiating')
            self.hgbc_regressor = sklearn.ensemble.HistGradientBoostingRegressor(
                loss=self.template_params['loss'],
                learning_rate=self.template_params['learning_rate'],
                max_leaf_nodes=self.template_params['max_leaf_nodes'],
                max_depth=self.template_params['max_depth'],
                min_samples_leaf=self.template_params['min_samples_leaf'],
                l2_regularization=self.template_params['l2_regularization'],
                )
            print('Fitting')
            self.hgbc_regressor.fit(train_inputs, train_outputs)
            print('Done')
        else:
            grid_search = GridSearchCV(
                estimator=sklearn.ensemble.HistGradientBoostingRegressor(
                    loss=self.template_params['loss'],
                    learning_rate=self.template_params['learning_rate'],
                    max_leaf_nodes=self.template_params['max_leaf_nodes'],
                    max_depth=self.template_params['max_depth'],
                    min_samples_leaf=self.template_params['min_samples_leaf'],
                    l2_regularization=self.template_params['l2_regularization'],
                    ),
                param_grid=self.template_params['grid_search'],
                cv=3,
                n_jobs=2,
                verbose=1,
                )
            grid_search.fit(train_inputs, train_outputs, sample_weight=train_outputs)
            self.template_params.update(grid_search.best_params_)
            print(grid_search.best_params_)
            self.hgbc_regressor = grid_search.best_estimator_

        success_pred_train = self.hgbc_regressor.predict(train_inputs)
        success_pred_val = self.hgbc_regressor.predict(val_inputs)

        alpha = 0.1
        ms = 0.1
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].plot(
            train_outputs, success_pred_train,
            linestyle='none', marker='.', alpha=alpha, markersize=ms
            )
        axes[1].plot(
            val_outputs, success_pred_val,
            linestyle='none', marker='.', alpha=alpha, markersize=ms
            )
        for index in range(2):
            lim = [0, 1.05]
            axes[index].plot(lim, lim, linestyle='dotted', color=[0, 0, 0], linewidth=1)
            axes[index].set_xlim(lim)
            axes[index].set_ylim(lim)
            axes[index].set_xlabel('True Success Rate')
        axes[0].set_ylabel('Predicted Success Rate')
        axes[0].set_title('Training')
        axes[1].set_title('Validation')
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to}',
            f'{self.bravais_lattice}_regression_{self.template_params["tag"]}.png'
            ))
        plt.close()

        return train_inputs
