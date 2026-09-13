import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import scipy.spatial
import sklearn.ensemble
from sklearn.model_selection import GridSearchCV

from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
from mlindex.utilities.FigureOfMerits import get_assignment_posterior
from mlindex.utilities.FigureOfMerits import get_assignment_sigma
from mlindex.utilities.FigureOfMerits import get_delta_dewolff61
from mlindex.utilities.FigureOfMerits import get_M20_likelihood
from mlindex.utilities.FigureOfMerits import get_n_dewolff61
from mlindex.utilities.FigureOfMerits import get_zone_dominance
from mlindex.utilities.FigureOfMerits import merit_set
from mlindex.utilities.IOManagers import read_params
from mlindex.utilities.IOManagers import write_params
from mlindex.utilities.IOManagers import SKLearnManager
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_hkl_matrix
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import reciprocal_uc_conversion


# RESEARCH CODE THAT NEEDS TO BE DELETED at P07. P06 screens which inputs the template regressor
# reads and P07 refits the chosen set; once those regressors ship, TEMPLATE_INPUT_SETS, the
# 'template_inputs' setting and the 'rho' family are removed, and get_M20_likelihood with them.
#
# An input set is an ordered tuple of feature families, and its columns appear in that order.
TEMPLATE_INPUT_SETS = {
    'rho': ('rho', 'line_count'),
    'rho_sigma': ('rho', 'line_count', 'log_sigma'),
    'posterior_sigma': ('posterior', 'line_count', 'log_sigma'),
    'merits': ('posterior', 'line_count', 'log_sigma', 'merits'),
    'merits_structure': ('posterior', 'line_count', 'log_sigma', 'merits', 'structure'),
    'merits_structure_context': (
        'posterior', 'line_count', 'log_sigma', 'merits', 'structure', 'context'),
    'scalars_only': ('line_count', 'log_sigma', 'merits', 'structure', 'context'),
    }
MERIT_NAMES = ('M20', 'M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap', 'n_cal')
STRUCTURE_NAMES = ('log_volume', 'zone_dominance', 'n_dewolff61', 'delta_dewolff61')
# Each context feature is a merit's distance from its best value over the pattern's template
# cells: the largest M20 and M_sym, the smallest n_over and max_gap.
CONTEXT_MERITS = (('M20', np.max), ('M_sym', np.max), ('n_over', np.min), ('max_gap', np.min))


def check_input_set(input_set):
    if input_set not in TEMPLATE_INPUT_SETS:
        raise ValueError(
            f'Unknown template input set {input_set!r}; known: {sorted(TEMPLATE_INPUT_SETS)}')
    return input_set


def family_width(family, n_peaks_calibration):
    widths = {
        'rho': n_peaks_calibration,
        'posterior': n_peaks_calibration,
        'line_count': 2,
        'log_sigma': 1,
        'merits': len(MERIT_NAMES),
        'structure': len(STRUCTURE_NAMES),
        'context': len(CONTEXT_MERITS),
        }
    return widths[family]


def n_template_features(template_params):
    """The regressor's input width under `template_params`' input set."""
    families = TEMPLATE_INPUT_SETS[check_input_set(template_params['template_inputs'])]
    return sum(family_width(family, template_params['n_peaks_calibration']) for family in families)


def template_features(families, q2_obs, xnn, q2_calc, q2_ref_calc, lattice_system, bravais_lattice):
    """Each requested feature family for a set of template cells, as (n_cells, width) arrays.

    `q2_obs` is the calibration peak list, `q2_calc` the lines the templates assigned to those
    peaks, and `q2_ref_calc` every reference line of each cell. A computation shared by several
    families runs once: sigma and the posterior come from one nearest-line scan, and the merits
    and their context from one `merit_set` call.
    """
    families = set(families)
    features = {}
    if families & {'rho', 'structure'}:
        reciprocal_volume = get_unit_cell_volume(get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=lattice_system
            ), partial_unit_cell=True, lattice_system=lattice_system)
    if 'rho' in families:
        _, features['rho'], _ = get_M20_likelihood(
            q2_obs=q2_obs,
            q2_calc=q2_calc,
            bravais_lattice=bravais_lattice,
            reciprocal_volume=reciprocal_volume
            )
    if 'line_count' in families:
        # How many reference lines fall below the highest assigned line, and that line.
        q2_calc_max = q2_calc.max(axis=1)
        N_pred = np.count_nonzero(q2_ref_calc < q2_calc_max[:, np.newaxis], axis=1)
        features['line_count'] = np.stack((N_pred, q2_calc_max), axis=1)
    if families & {'posterior', 'log_sigma'}:
        sigma, d1 = get_assignment_sigma(q2_obs, q2_ref_calc, lattice_system)
        if 'posterior' in families:
            features['posterior'] = get_assignment_posterior(
                q2_obs, q2_ref_calc, lattice_system, sigma=sigma, d1=d1)
        if 'log_sigma' in families:
            features['log_sigma'] = np.log(sigma)[:, np.newaxis]
    if families & {'merits', 'context'}:
        merits = merit_set(q2_obs, q2_ref_calc)
        if 'merits' in families:
            features['merits'] = np.stack([merits[name] for name in MERIT_NAMES], axis=1)
        if 'context' in families:
            features['context'] = np.stack(
                [merits[name] - best(merits[name]) for name, best in CONTEXT_MERITS], axis=1)
    if 'structure' in families:
        # de Wolff's expected line count and discrepancy are evaluated at the last peak.
        q2_last = q2_obs[-1:]
        features['structure'] = np.stack((
            -np.log(reciprocal_volume),
            get_zone_dominance(xnn, lattice_system),
            get_n_dewolff61(q2_last, xnn, lattice_system, bravais_lattice)[:, 0],
            get_delta_dewolff61(q2_last, xnn, lattice_system, bravais_lattice)[:, 0],
            ), axis=1)
    return features


def regressor_inputs(features, input_set):
    """The input set's families side by side, in float32.

    float32 is what makes the ONNX graph and the sklearn model return the same predictions.
    """
    return np.concatenate(
        [features[family] for family in TEMPLATE_INPUT_SETS[input_set]], axis=1
        ).astype(np.float32)


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
            'n_processes': 1,
            'n_entries_train': 1000,
            'n_instances_train': 1000000,
            'loss': 'squared_error',
            'learning_rate': 0.1,
            'max_leaf_nodes': 31,
            'max_depth': 4,
            'min_samples_leaf': 100,
            'l2_regularization': 10,
            'q2_error_multiplier_low': 0.25,
            'q2_error_multiplier_high': 1.75,
            'n_contaminants_max': 1,
            'n_peaks': data_params['n_peaks'],
            'n_peaks_template': min(10, data_params['n_peaks']),
            'n_peaks_calibration': min(20, data_params['n_peaks']),
            'max_distance': 0.05,
            'grid_search': None,
            'load_templates': False,
            'load_training_data': False,
            'template_inputs': 'rho',
            }
        for key in template_params_defaults.keys():
            if key not in self.template_params.keys():
                self.template_params[key] = template_params_defaults[key]
        check_input_set(self.template_params['template_inputs'])

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
            n_features=n_template_features(self.template_params),
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
            'template_inputs',
            ]
        self.template_params = dict.fromkeys(params_keys)
        self.template_params['tag'] = params['tag']
        # A model saved before this setting existed was fitted on 'rho'.
        self.template_params['template_inputs'] = check_input_set(
            params.get('template_inputs') or 'rho')
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
            miller_index_templates = np.vstack(mi_sets)
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
                                        hkl_labels_ratio_bin,
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
            miller_index_templates = np.vstack(mi_sets)
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
        # `load_templates` reuses a template library already saved under this tag, so a refit
        # changes the regressor alone; `load_training_data` separately reuses its training cache.
        if self.template_params['load_templates']:
            self.miller_index_templates = np.load(os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_miller_index_templates_{self.template_params["tag"]}.npy'
                ))
            self.miller_index_templates_prob = np.load(os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_miller_index_templates_prob_{self.template_params["tag"]}.npy',
                ))
            self.template_params['n_templates'] = self.miller_index_templates.shape[0]
        else:
            self.setup_templates(data)
        train_inputs = self.calibrate_templates(data)
        self.save(train_inputs)

    def generate_xnn(self, q2_obs, rng, indices=None):
        if indices is None:
            hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)
            n_templates = self.template_params['n_templates']
        else:
            hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates[indices]], self.lattice_system)
            n_templates = indices.size

        # q2_calc should increase monotonically. Sort hkl2 then re-solve for xnn iteratively.
        q2_obs_template = q2_obs[:self.template_params['n_peaks_template']]
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
            xnn = fix_unphysical(xnn=xnn, rng=rng, lattice_system=self.lattice_system)
        return self._generate_xnn_common(q2_obs, xnn, rng)

    def generate_xnn_true(self, q2_obs, xnn_true, rng):
        from mlindex.utilities.ErrorAdder import perturb_xnn
        if self.bravais_lattice in ['cF', 'cI', 'cP']:
            convergence_distances = np.logspace(-4, -0, 50)
        elif self.bravais_lattice in ['hP', 'hR', 'tI', 'tP']:
            convergence_distances = np.logspace(-4, -0.5, 50)
        elif self.bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            convergence_distances = np.logspace(-4, -1.5, 50)
        else:
            convergence_distances = np.logspace(-4, -2, 50)
        xnn = perturb_xnn(
            xnn_true,
            convergence_candidates=10,
            convergence_distances=convergence_distances,
            minimum_uc=2,
            maximum_uc=500,
            lattice_system=self.lattice_system,
            rng=rng,
        )
        return self._generate_xnn_common(q2_obs, xnn, rng)
        
    def _generate_xnn_common(self, q2_obs, xnn, rng):
        q2_obs_template = q2_obs[:self.template_params['n_peaks_template']]
        q2_obs_calibration = q2_obs[:self.template_params['n_peaks_calibration']]
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
        xnn = fix_unphysical(xnn=xnn, rng=rng, lattice_system=self.lattice_system)
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
        # Every reference line of each surviving cell, which the regressor's inputs are built from.
        q2_ref_calc = q2_calculator.get_q2(xnn)
        return xnn, q2_calc, q2_ref_calc

    def _inputs(self, q2_obs, xnn, q2_calc, q2_ref_calc):
        """The regressor's input matrix for these template cells, under this model's input set."""
        input_set = self.template_params['template_inputs']
        features = template_features(
            TEMPLATE_INPUT_SETS[input_set],
            q2_obs[:self.template_params['n_peaks_calibration']],
            xnn, q2_calc, q2_ref_calc, self.lattice_system, self.bravais_lattice,
            )
        return regressor_inputs(features, input_set)

    def generate(self, n_templates, rng, q2_obs):
        xnn_templates_all, q2_calc, q2_ref_calc = self.generate_xnn(q2_obs, rng)
        if n_templates == 'all':
            xnn_templates = xnn_templates_all
        elif n_templates <= xnn_templates_all.shape[0]:
            inputs = self._inputs(q2_obs, xnn_templates_all, q2_calc, q2_ref_calc)
            _, unique_indices = np.unique(
                np.round(xnn_templates_all, decimals=6), return_index=True, axis=0
                )
            xnn_templates_all = xnn_templates_all[unique_indices]
            inputs = inputs[unique_indices]
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
                    xnn_chunk = np.vstack((
                        np.delete(xnn_chunk, neighbor_indices, axis=0), 
                        xnn_best_neighbor
                        ))
                    q2_calc_chunk = np.vstack((
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
        xnn_downsampled = np.vstack(xnn_downsampled)
        q2_calc_downsampled = np.vstack(q2_calc_downsampled)
        return xnn_downsampled, q2_calc_downsampled

    def _get_inputs_worker(self, inputs):
        from mlindex.utilities.ErrorAdder import add_q2_error
        from mlindex.utilities.ErrorAdder import add_contaminants
        q2_obs = inputs[0]
        xnn_true = inputs[1]
        train = inputs[2]

        multiplier = self.rng.uniform(
            low=self.template_params['q2_error_multiplier_low'],
            high=self.template_params['q2_error_multiplier_high'],
        )
        q2_obs = add_q2_error(q2_obs[np.newaxis], None, multiplier, self.rng)[0]
        q2_obs = add_contaminants(
            q2=q2_obs[np.newaxis],
            hkl=None,
            n_contaminants=self.template_params['n_contaminants_max'],
            rng=self.rng,
            random_n_contaminants=True
        )[0]

        xnn, q2_calc, q2_ref_calc = self.generate_xnn(q2_obs, self.rng)
        inputs = self._inputs(q2_obs, xnn, q2_calc, q2_ref_calc)
        distance = scipy.spatial.distance.cdist(xnn, xnn_true[np.newaxis])[:, 0]
        select = distance < self.template_params['max_distance']
        return inputs[select], distance[select], xnn[select]

    def get_inputs(self, data, n_entries, train=False):
        from tqdm import tqdm
        q2_obs = np.stack(data['q2'])
        inputs = []
        distance = []
        xnn = []
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
                inputs_entry, distance_entry, xnn_entry = \
                    self._get_inputs_worker([q2_obs[index], xnn_true[index], train])
                inputs.append(inputs_entry)
                distance.append(distance_entry)
                xnn.append(xnn_entry)
        elif self.template_params['parallelization'] == 'multiprocessing':
            print(f'Setting up {n_entries} entries using multiprocessing')
            with multiprocessing.Pool(self.template_params['n_processes']) as p:
                if train:
                    train_array = np.ones(indices.size, dtype=bool)
                else:
                    train_array = np.zeros(indices.size, dtype=bool)
                outputs = p.map(self._get_inputs_worker, zip(q2_obs[indices], xnn_true[indices], train_array))
            for i in range(n_entries):
                inputs.append(outputs[i][0])
                distance.append(outputs[i][1])
                xnn.append(outputs[i][2])

        inputs = np.vstack(inputs)
        distance = np.concatenate(distance)
        xnn = np.vstack(xnn)
        return inputs, distance, xnn, xnn_true[indices]

    def _sample_training_val_data(self, distance, inputs):
        # Select only training and validation entries that are within a distance (1/A**2)
        # of the correct values.
        # distance_train has shape N
        select = distance < self.template_params['max_distance']
        inputs = inputs[select]
        distance = distance[select]
        if distance.size > self.template_params['n_instances_train']:
            n_bins = 10
            distance_bins = np.linspace(0, self.template_params['max_distance'], n_bins + 1)
            inputs_new = []
            distance_new = []
            n_per_bin = self.template_params['n_instances_train'] // n_bins
            n_instances = 0
            for bin_index in range(n_bins):
                bin_indices = np.logical_and(
                    distance >= distance_bins[bin_index],
                    distance < distance_bins[bin_index + 1],
                )
                if bin_indices.sum() > 0:
                    if bin_indices.sum() > n_per_bin:
                        select = self.rng.choice(
                            int(bin_indices.sum()),
                            size=n_per_bin,
                            replace=False
                            )
                        inputs_new.append(inputs[bin_indices][select])
                        distance_new.append(distance[bin_indices][select])
                        n_instances += n_per_bin
                    else:
                        inputs_new.append(inputs[bin_indices])
                        distance_new.append(distance[bin_indices])
                        if bin_index < (n_bins - 1):
                            n_instances += bin_indices.sum()
                            n_per_bins = (self.template_params['n_instances_train'] - n_instances) // (n_bins - bin_index)
                else:
                    if bin_index < (n_bins - 1):
                        n_per_bins = (self.template_params['n_instances_train'] - n_instances) // (n_bins - bin_index)
            inputs = np.concatenate(inputs_new, axis=0)
            distance = np.concatenate(distance_new)
        return distance, inputs
    
    def calibrate_templates(self, data):
        unaugmented_data = data[~data['augmented']]
        training_data = unaugmented_data[unaugmented_data['train']]
        val_data = unaugmented_data[~unaugmented_data['train']]
        n_val = int(0.2*self.template_params['n_entries_train'])

        # The cache holds the distance in column 0 and the inputs after it, so it is named for the
        # input set: a cache of one set read by another would be the wrong width or the wrong columns.
        cache_directory = os.path.join(f'{self.save_to}', 'data_cache')
        cache_name = f'{self.bravais_lattice}_{self.template_params["template_inputs"]}'
        if self.template_params['load_training_data']:
            training_cache = np.load(os.path.join(cache_directory, f'{cache_name}_train.npy'))
            distance_train = training_cache[:, 0]
            train_inputs = training_cache[:, 1:].astype(np.float32)
            val_cache = np.load(os.path.join(cache_directory, f'{cache_name}_val.npy'))
            distance_val = val_cache[:, 0]
            val_inputs = val_cache[:, 1:].astype(np.float32)
        else:
            train_inputs, distance_train, _, _ = \
                self.get_inputs(training_data, self.template_params['n_entries_train'], train=True)
            val_inputs, distance_val, _, _ = self.get_inputs(val_data, n_val)
            distance_train, train_inputs = self._sample_training_val_data(distance_train, train_inputs)
            distance_val, val_inputs = self._sample_training_val_data(distance_val, val_inputs)
            if not os.path.exists(cache_directory):
                os.mkdir(cache_directory)
            np.save(
                os.path.join(cache_directory, f'{cache_name}_train.npy'),
                np.concatenate((distance_train[:, np.newaxis], train_inputs), axis=1)
            )
            np.save(
                os.path.join(cache_directory, f'{cache_name}_val.npy'),
                np.concatenate((distance_val[:, np.newaxis], val_inputs), axis=1)
            )

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
                verbose=2,
                max_iter=25,
                # Above 10 000 rows the regressor holds out a random tenth for early stopping,
                # so without a seed every fit is a different model.
                random_state=self.seed,
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
                    random_state=self.seed,
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
        np.save(
            os.path.join(
                f'{self.save_to}',
                f'{self.bravais_lattice}_regression_{self.template_params["tag"]}_val.npy'
                ),
            np.stack((val_outputs, success_pred_val), axis=1)
            )

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
