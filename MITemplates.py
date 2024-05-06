import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import scipy.stats
# This prevents tensorflow from doing the import printout repeatitively during multiprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tqdm import tqdm

from Utilities import get_hkl_matrix
from Utilities import get_unit_cell_from_xnn
from Utilities import read_params
from Utilities import write_params


class MITemplates:
    def __init__(self, group, data_params, template_params, hkl_ref, save_to, seed):
        self.template_params = template_params
        template_params_defaults = {
            'n_dominant_zone_bins': 4,
            'templates_per_dominant_zone_bin': 1000,
            'parallelization': None,
            'n_processes': None,
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

        if self.template_params['parallelization'] == 'multiprocessing':
            import multiprocessing
        elif self.template_params['parallelization'] == 'message_passing':
            assert False

    def save(self):
        write_params(
            self.template_params,
            f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv'
            )

    def load_from_tag(self):
        self.miller_index_templates = np.load(f'{self.save_to}/{self.group}_miller_index_templates_{self.template_params["tag"]}.npy')
        params = read_params(f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv')
        params_keys = [
            'tag',
            'n_dominant_zone_bins',
            'templates_per_dominant_zone_bin',
            'n_templates'
            ]
        self.template_params = dict.fromkeys(params_keys)
        self.template_params['tag'] = params['tag']
        self.template_params['n_dominant_zone_bins'] = int(params['n_dominant_zone_bins'])
        self.template_params['templates_per_dominant_zone_bin'] = int(params['templates_per_dominant_zone_bin'])
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]

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
        
        if self.lattice_system in ['cubic', 'rhombohedral']:
            miller_index_templates = make_sets(
                self.template_params['templates_per_dominant_zone_bin'],
                self.n_points,
                hkl_labels_all,
                self.hkl_ref_length,
                self.rng
                )
        else:
            mi_sets = []
            reindexed_xnn = np.stack(training_data['reindexed_xnn'])
            ratio = reindexed_xnn[:, :3].min(axis=1) / reindexed_xnn[:, :3].max(axis=1)
            n_bins = self.template_params['n_dominant_zone_bins']
            bins = np.linspace(0, 1, n_bins + 1)

            fig, axes = plt.subplots(1, 1, figsize=(5, 3))
            axes.hist(ratio, bins=bins)
            axes.set_xlabel('Dominant zone ratio (Min(Xnn) / max(Xnn))')
            axes.set_ylabel('Counts')
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/{self.group}_dominant_zone_ratio_{self.template_params["tag"]}.png')

            indices = np.searchsorted(bins, ratio)
            for i in range(1, n_bins + 2):
                hkl_labels_bin = hkl_labels_all[indices == i]
                if hkl_labels_bin.shape[0] < self.template_params['templates_per_dominant_zone_bin']:
                    mi_sets.append(hkl_labels_bin)
                else:
                    mi_sets.append(make_sets(
                        self.template_params['templates_per_dominant_zone_bin'],
                        self.n_points,
                        hkl_labels_bin,
                        self.hkl_ref_length,
                        self.rng
                        ))
            miller_index_templates = np.row_stack(mi_sets)
        self.miller_index_templates = np.unique(miller_index_templates, axis=0)
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]
        np.save(
            f'{self.save_to}/{self.group}_miller_index_templates_{self.template_params["tag"]}.npy',
            self.miller_index_templates
            )

    def setup(self, data):
        self.setup_templates(data)
        self.save()

    def generate_xnn_fast(self, q2_obs):
        # This is slow
        # original I just did linear least squares iteratively until q2_calc was ordered (1st for loop)
        # It is faster to use Gauss-Newton non-linear least squares (2nd for loop)
        # This is mostly copied from TargetFunctions.py

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
        
        if self.lattice_system in ['monoclinic', 'orthorhombic', 'triclinic']:
            bad_indices = np.any(xnn[:, :3] == 0, axis=1)
            if np.sum(bad_indices) > 0:
                print(np.sum(bad_indices))
                xnn[bad_indices, :3] = xnn[:, :3].mean()
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
        if self.lattice_system in ['monoclinic', 'orthorhombic', 'triclinic']:
            bad_indices = np.any(xnn[:, :3] == 0, axis=1)
            if np.sum(bad_indices) > 0:
                xnn[bad_indices, :3] = xnn[:, :3].mean()
        return xnn, loss

    def do_predictions(self, q2_obs, n_templates='all'):
        # This is primary used to generate candidates for the optimizer, which expects that the
        # sum of the template and sampled candidates to be the same. This is why n_samples gets
        # changed
        xnn_templates, _ = self.generate_xnn(q2_obs)
        if n_templates == 'all':
            pass
        elif n_templates < xnn_templates.shape[0]:
            indices = self.rng.choice(xnn_templates.shape[0], size=n_templates)
            xnn_templates =  xnn_templates[indices]
        elif n_templates > xnn_templates.shape[0]:
            assert False

        unit_cell_templates = get_unit_cell_from_xnn(
            xnn_templates, partial_unit_cell=True, lattice_system=self.lattice_system, radians=True
            )
        return unit_cell_templates
  

class MITemplates_binning(MITemplates):
    def __init__(self, group, data_params, template_params, hkl_ref, save_to, seed):
        super(MITemplates_base, self).__init__(group, data_params, template_params, hkl_ref, save_to, seed)
        template_params_defaults = {
            'n_xnn_bins': 100,
            'filters': [16, 16, 16],
            'kernel_size': [10, 8, 4],
            'output_filters': 16,
            'output_kernel_size': 4,
            'output_bias_regularizer': 0.01,
            'dropout_rate': 0.1,
            'epochs': 20,
            'learning_rate': 0.0001,
            'batch_size': 32,
            }

        for key in template_params_defaults.keys():
            if key not in self.template_params.keys():
                self.template_params[key] = template_params_defaults[key]

        if not os.path.exists(f'{self.save_to}/validation_plots'):
            os.mkdir(f'{self.save_to}/validation_plots')

        if self.template_params['parallelization'] == 'multiprocessing':
            import multiprocessing
        elif self.template_params['parallelization'] == 'message_passing':
            assert False

        if self.lattice_system in ['cubic', 'hexagonal', 'rhombohedral', 'tetragonal', 'orthorhombic']:
            self.n_axes = self.n_outputs
        elif self.lattice_system in ['monoclinic', 'triclinic']:
            self.n_axes = 3

    def save(self):
        write_params(
            self.template_params,
            f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv'
            )
        for i in range(len(self.xnn_bins)):
            np.save(
                f'{self.save_to}/{self.group}_xnn_bins_axis_{i}_{self.template_params["tag"]}.npy',
                self.xnn_bins[i]
                )
        self.model.save_weights(f'{self.save_to}/{self.group}_model_weights_{self.template_params["tag"]}.h5')

    def load_from_tag(self):
        self.miller_index_templates = np.load(f'{self.save_to}/{self.group}_miller_index_templates_{self.template_params["tag"]}.npy')
        params = read_params(f'{self.save_to}/{self.group}_template_params_{self.template_params["tag"]}.csv')
        params_keys = [
            'tag',
            'n_dominant_zone_bins',
            'templates_per_dominant_zone_bin',
            'n_xnn_bins',
            'filters',
            'kernel_size',
            'output_filters',
            'output_kernel_size',
            'output_bias_regularizer',
            'dropout_rate',
            'epochs',
            'learning_rate',
            'batch_size',
            ]
        self.template_params = dict.fromkeys(params_keys)
        self.template_params['tag'] = params['tag']
        self.template_params['n_dominant_zone_bins'] = int(params['n_dominant_zone_bins'])
        self.template_params['templates_per_dominant_zone_bin'] = int(params['templates_per_dominant_zone_bin'])
        self.template_params['n_templates'] = self.miller_index_templates.shape[0]
        self.template_params['n_xnn_bins'] = int(params['n_xnn_bins'])
        self.template_params['dropout_rate'] = float(params['dropout_rate'])
        self.template_params['learning_rate'] = float(params['learning_rate'])
        self.template_params['epochs'] = int(params['epochs'])
        self.template_params['batch_size'] = int(params['batch_size'])
        self.template_params['filters'] = int(params['filters'])
        self.template_params['kernel_size'] = np.array(
            params['kernel_size'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.template_params['output_filters'] = int(params['output_filters'])
        self.template_params['output_kernel_size'] = int(params['output_kernel_size'])
        self.template_params['output_bias_regularizer'] = float(params['output_bias_regularizer'])

        self.build_model()
        self.model.load_weights(
            filepath=f'{self.save_to}/{self.group}_model_weights_{self.template_params["tag"]}.h5',
            by_name=True
            )
        self.compile_model()

        self.xnn_bins = []
        for i in range(self.n_axes):
            self.xnn_bins.append(
                np.load(f'{self.save_to}/{self.group}_xnn_bins_axis_{i}_{self.template_params["tag"]}.npy')
                )

    def setup_binning(self, data):
        training_data = data[data['train']]
        reindexed_xnn = np.stack(training_data['reindexed_xnn'])[:, self.y_indices]
        self.xnn_bins = []
        for i in range(self.n_axes):
            sorted_values = np.sort(reindexed_xnn[:, i])
            self.xnn_bins.append(np.linspace(
                    sorted_values[0],
                    sorted_values[int(0.9925*sorted_values.size)],
                    self.template_params['n_xnn_bins'] + 1
                ))

    def setup(self, data):
        self.setup_templates(data)
        self.setup_binning(data)

    def generate_distribution(self, q2_obs):
        xnn, loss = self.generate_xnn(q2_obs)
        if self.lattice_system in ['monoclinic', 'triclinic']:
            xnn = xnn[:, :3]
        xnn_histogram, _ = np.histogramdd(sample=xnn, bins=self.xnn_bins, density=False)
        xnn_distribution = xnn_histogram / xnn_histogram.sum()
        loss_distribution, _, _ = scipy.stats.binned_statistic_dd(
            sample=xnn,
            values=loss,
            bins=self.xnn_bins,
            statistic='mean',
            )
        if self.n_axes == 1:
            normalized_loss = np.zeros((
                self.template_params['n_xnn_bins']
                ))
        elif self.n_axes == 2:
            normalized_loss = np.zeros((
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                ))
        elif self.n_axes == 3:
            normalized_loss = np.zeros((
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                ))
        good_indices = np.invert(np.isnan(loss_distribution))
        if np.sum(good_indices) > 0:
            normalized_loss[good_indices] = \
                loss_distribution[good_indices].min() / loss_distribution[good_indices]
        return xnn, xnn_distribution, normalized_loss

    def do_predictions(self, q2_obs, n_templates='all', n_samples=None):
        # This is primary used to generate candidates for the optimizer, which expects that the
        # sum of the template and sampled candidates to be the same. This is why n_samples gets
        # changed
        if n_samples is None or n_samples == 0:
            xnn_templates, _ = self.generate_xnn(q2_obs)
            if n_templates == 'all':
                pass
            elif n_templates < xnn_templates.shape[0]:
                indices = self.rng.choice(xnn_templates.shape[0], size=n_templates)
                xnn_templates =  xnn_templates[indices]
            elif n_templates > xnn_templates.shape[0]:
                assert False
            unit_cell_templates = get_unit_cell_from_xnn(
                xnn_templates, partial_unit_cell=True, lattice_system=self.lattice_system, radians=False
                )
            return unit_cell_templates, None
        else:
            xnn_templates, xnn_distribution, normalized_loss = self.generate_distribution(
                q2_obs
                )

        if n_templates == 'all':
            pass
        elif n_templates < xnn_templates.shape[0]:
            indices = self.rng.choice(xnn_templates.shape[0], size=n_templates)
            xnn_templates = xnn_templates[indices]
        elif n_templates > xnn_templates.shape[0]:
            n_samples += n_templates - xnn_templates.shape[0]

        inputs = {
            'xnn_distribution': xnn_distribution[np.newaxis],
            'normalized_loss': normalized_loss[np.newaxis]
            }
        xnn_distribution_nn = np.array(self.model.predict_on_batch(inputs))[0]
        indices_flat = np.searchsorted(
            np.cumsum(xnn_distribution_nn), self.rng.random(size=n_samples)
            )
        indices = np.unravel_index(
            indices_flat,
            shape=(self.template_params['n_xnn_bins'], self.n_outputs)
            )
        xnn_sampled = np.zeros((n_samples, self.n_outputs))
        for i in range(self.n_outputs):
            xnn_sampled[:, i] = self.xnn_bins[i][indices[i]]
            # This randomly perturbs xnn from within a bin
            random_range = (self.xnn_bins[i][1] - self.xnn_bins[i][0])/2
            xnn_sampled[:, i] += self.rng.uniform(
                low=-random_range, high=random_range, size=n_samples
                )
        unit_cell_templates = get_unit_cell_from_xnn(
            xnn_templates, partial_unit_cell=True, lattice_system=self.lattice_system, radians=True
            )
        unit_cell_sampled = get_unit_cell_from_xnn(
            xnn_sampled, partial_unit_cell=True, lattice_system=self.lattice_system, radians=True
            )
        return unit_cell_templates, unit_cell_sampled

    def nn_results_plots(self, data, n_entries=10):
        validation_data = data[~data['train']]
        n_entries = min(len(validation_data), n_entries)
        q2_obs = np.stack(validation_data['q2'])
        xnn_true = np.stack(validation_data['reindexed_xnn'])[:, self.y_indices]
        
        centers = (self.xnn_bins[0][1:] + self.xnn_bins[0][:-1]) / 2
        bin_width = centers[1] - centers[0]
        print(f'Templating for {n_entries} validation entries with {self.template_params["n_templates"]} templates')
        for entry_index in range(n_entries):
            _, xnn_distribution, normalized_loss = self.generate_distribution(q2_obs[entry_index])
            inputs = {
                'xnn_distribution': xnn_distribution[np.newaxis],
                'normalized_loss': normalized_loss[np.newaxis]
                }
            xnn_distribution_nn = np.array(self.model(inputs))[0]
            if self.n_axes == 1:
                xnn_estimates = [
                    centers[np.argmax(xnn_distribution)],
                    centers[np.argmax(normalized_loss)],
                    ]

                fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
                axes[0].bar(centers, xnn_distribution, width=bin_width)
                axes[0].bar(centers, xnn_distribution_nn, width=bin_width, alpha=0.5)
                axes[1].bar(centers, normalized_loss, width=bin_width)
                title = [
                    f'True unit cell: {xnn_true[entry_index, 0]:0.2f}',
                    f'Most frequent: {xnn_estimates[0]:0.2f}',
                    f'Best loss: {xnn_estimates[1]:0.2f}',
                    ]
                axes[0].set_title('\n'.join(title))
                for i in range(2):
                    ylim = axes[i].get_ylim()
                    axes[i].plot(
                        [xnn_true[entry_index, 0], xnn_true[entry_index, 0]], ylim,
                        color=[0.8, 0, 0], linestyle='dotted'
                        )
                    axes[i].set_ylim(ylim)
                axes[1].set_xlabel('Xnn')
                axes[0].set_ylabel('Distribution')
                axes[1].set_ylabel('Normalized 1/sLoss')
            elif self.n_axes == 2:
                xnn_distribution_nn = xnn_distribution_nn.reshape((
                    self.template_params['n_xnn_bins'], self.template_params['n_xnn_bins']
                    ))
                vmin = min(xnn_distribution.min(), xnn_distribution_nn.min())
                vmax = min(xnn_distribution.max(), xnn_distribution_nn.max())
                extent = [
                    self.xnn_bins[1][0], self.xnn_bins[1][-1],
                    self.xnn_bins[0][0], self.xnn_bins[0][-1],
                    ]
                fig, axes = plt.subplots(1, 3, figsize=(8, 5))
                axes[0].imshow(
                    xnn_distribution,
                    origin='lower', extent=extent,
                    vmin=vmin, vmax=vmax, cmap='Greys'
                    )
                axes[1].imshow(
                    xnn_distribution_nn,
                    origin='lower', extent=extent,
                    vmin=vmin, vmax=vmax, cmap='Greys'
                    )
                axes[2].imshow(
                    normalized_loss,
                    origin='lower', extent=extent,
                    cmap='Greys'
                    )
                for i in range(3):
                    axes[i].plot(
                        xnn_true[entry_index, 0], xnn_true[entry_index, 1],
                        marker='.', markersize=10, color=[0.8, 0, 0]
                        )
                axes[0].set_ylabel('Xhh')
                for i in range(3):
                    axes[i].set_xlabel('Xll')
                axes[0].set_title('Xnn distribution\ninitial')
                axes[1].set_title('Xnn distribution\nrecalibrated')
                axes[2].set_title('Normalized loss')
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/validation_plots/{self.group}_entry_{entry_index:03d}_validation_plot_{self.template_params["tag"]}.png')
            plt.close()

    def build_model(self):
        if self.n_axes == 1:
            shape = (
                self.template_params['n_xnn_bins'],
                1
                    )
        elif self.n_axes == 2:
            shape = (
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                1
                )
        elif self.n_axes == 3:
            shape = (
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                1
                )
        inputs = {
            'xnn_distribution': tf.keras.Input(
                shape=shape,
                name='xnn_distribution',
                dtype=tf.float32,
                ),
            'normalized_loss': tf.keras.Input(
                shape=shape,
                name='normalized_loss',
                dtype=tf.float32,
                ),
            }
        self.model = tf.keras.Model(inputs, self.model_builder(inputs))
        self.model.summary()

    def model_builder(self, inputs):
        x_input = tf.keras.layers.Concatenate(axis=self.n_axes+1)((
            inputs['xnn_distribution'],
            inputs['normalized_loss'],
            ))
        if self.n_axes == 1:
            Conv = tf.keras.layers.Conv1D
            MaxPooling = tf.keras.layers.MaxPooling1D
            kernel_size = self.template_params['kernel_size']
            output_kernel_size = self.template_params['output_kernel_size']
        elif self.n_axes == 2:
            Conv = tf.keras.layers.Conv2D
            MaxPooling = tf.keras.layers.MaxPooling2D
            kernel_size = []
            for i in range(len(self.template_params['kernel_size'])):
                kernel_size.append([
                    self.template_params['kernel_size'][i],
                    self.template_params['kernel_size'][i],
                    ])
            output_kernel_size = [
                self.template_params['output_kernel_size'],
                self.template_params['output_kernel_size'],
                ]
        elif self.n_axes == 3:
            Conv = tf.keras.layers.Conv3D
            MaxPooling = tf.keras.layers.MaxPooling3D
            kernel_size = []
            for i in range(len(self.template_params['kernel_size'])):
                kernel_size.append([
                    self.template_params['kernel_size'][i],
                    self.template_params['kernel_size'][i],
                    self.template_params['kernel_size'][i],
                    ])
            output_kernel_size = [
                self.template_params['output_kernel_size'],
                self.template_params['output_kernel_size'],
                self.template_params['output_kernel_size'],
                ]

        x_xnn_distribution = Conv(
            filters=self.template_params['filters'][0],
            kernel_size=kernel_size[0],
            name='xnn_distribution_conv',
            use_bias=False,
            padding='same',
            )(inputs['xnn_distribution'])
        x_xnn_distribution = tf.keras.layers.LayerNormalization(
            name=f'xnn_distribution_layer_norm'
            )(x_xnn_distribution)
        x_xnn_distribution = tf.keras.activations.gelu(x_xnn_distribution)
        x_xnn_distribution = tf.keras.layers.Dropout(
            rate=self.template_params['dropout_rate'],
            name=f'xnn_distribution_dropout',
            )(x_xnn_distribution)

        x_normalized_loss = Conv(
            filters=self.template_params['filters'][0],
            kernel_size=kernel_size[0],
            name='normalized_loss_conv',
            use_bias=False,
            padding='same',
            )(inputs['normalized_loss'])
        x_normalized_loss = tf.keras.layers.LayerNormalization(
            name=f'normalized_loss_layer_norm'
            )(x_normalized_loss)
        x_normalized_loss = tf.keras.activations.gelu(x_normalized_loss)
        x_normalized_loss = tf.keras.layers.Dropout(
            rate=self.template_params['dropout_rate'],
            name=f'normalized_loss_dropout',
            )(x_normalized_loss)

        x = MaxPooling()(x_xnn_distribution + x_normalized_loss)

        for index in range(len(self.template_params['filters']) - 1):
            x = Conv(
                filters=self.template_params['filters'][index + 1],
                kernel_size=kernel_size[index + 1],
                name=f'conv_{index + 1}',
                use_bias=False,
                padding='same',
                )(x)
            x = tf.keras.layers.LayerNormalization(
                name=f'layer_norm_{index + 1}'
                )(x)
            x = tf.keras.activations.gelu(x)
            x = tf.keras.layers.Dropout(
                rate=self.template_params['dropout_rate'],
                name=f'dropout_{index + 1}',
                )(x)
            x = MaxPooling()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
                self.template_params['output_filters'],
                activation='linear',
                name=f'summation_params',
                use_bias=False,
                kernel_constraint=tf.keras.constraints.UnitNorm()
                )(x)
        x = tf.keras.activations.gelu(x)

        convolved_xnn_distribution = Conv(
            filters=self.template_params['output_filters'],
            kernel_size=output_kernel_size,
            use_bias=True,
            padding='same',
            bias_regularizer=tf.keras.regularizers.L2(l2=self.template_params['output_bias_regularizer'])
            )(inputs['xnn_distribution'])

        convolved_normalized_loss = Conv(
            filters=self.template_params['output_filters'],
            kernel_size=output_kernel_size,
            use_bias=True,
            padding='same',
            bias_regularizer=tf.keras.regularizers.L2(l2=self.template_params['output_bias_regularizer'])
            )(inputs['normalized_loss'])
        convolved_inputs = convolved_xnn_distribution + convolved_normalized_loss

        if self.n_axes == 1:
            weighted_sum = convolved_inputs * x[:, tf.newaxis, :]
        elif self.n_axes == 2:
            weighted_sum = convolved_inputs * x[:, tf.newaxis, tf.newaxis, :]
        elif self.n_axes == 3:
            weighted_sum = convolved_inputs * x[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        output_logits = tf.math.reduce_sum(weighted_sum, axis=self.n_axes+1)
        outputs_softmaxes = tf.keras.layers.Softmax(
            axis=1,
            name=f'output_softmaxes'
            )(tf.keras.layers.Flatten()(output_logits))
        return outputs_softmaxes

    def compile_model(self):
        optimizer = tf.optimizers.legacy.Adam(self.template_params['learning_rate'])
        loss_functions = {'output_softmaxes': tf.keras.losses.SparseCategoricalCrossentropy()}
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            )

    def get_inputs(self, data):
        n_entries = len(data)
        if self.lattice_system == 'cubic':
            xnn_distribution = np.zeros((
                n_entries, 
                self.template_params['n_xnn_bins']
                ))
        elif self.lattice_system in ['tetragonal', 'hexagonal', 'rhombohedral']:
            xnn_distribution = np.zeros((
                n_entries, 
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                ))
        else:
            xnn_distribution = np.zeros((
                n_entries, 
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                self.template_params['n_xnn_bins'],
                ))
        normalized_loss = np.zeros_like(xnn_distribution)
        q2_obs = np.stack(data['q2'])
        if self.template_params['parallelization'] is None:
            print(f'Setting up {n_entries} entries serially')
            for i in tqdm(range(n_entries)):
                _, xnn_distribution[i], normalized_loss[i] = self.generate_distribution(q2_obs[i])
        elif self.template_params['parallelization'] == 'multiprocessing':
            print(f'Setting up {n_entries} entries using multiprocessing')
            with multiprocessing.Pool(self.template_params['n_processes']) as p:
                outputs = p.map(self.generate_distribution, q2_obs)
            for i in range(n_entries):
                xnn_distribution[i] = outputs[i][1]
                normalized_loss[i] = outputs[i][2]
        elif self.template_params['parallelization'] == 'message_passing':
            assert False
        inputs = {
            'xnn_distribution': xnn_distribution[..., np.newaxis],
            'normalized_loss': normalized_loss[..., np.newaxis],
            }

        xnn = np.stack(data['reindexed_xnn'])[:, self.y_indices]
        ### !!! Replace this function with np.ravel_multi_index
        true = np.zeros(n_entries, dtype=int)
        for i in range(self.n_axes):
            true_axis = np.searchsorted(self.xnn_bins[i], xnn[:, i]) - 1
            true += true_axis * self.template_params['n_xnn_bins']**i
        return inputs, true

    def fit_model(self, data):
        self.build_model()
        self.compile_model()

        if self.template_params['load_training_data'] == False:
            train_inputs, train_true = self.get_inputs(data[data['train']])
            val_inputs, val_true = self.get_inputs(data[~data['train']])

            np.save(
                f'{self.save_to}/{self.group}_train_xnn_distribution_{self.template_params["tag"]}.npy',
                train_inputs['xnn_distribution']
                )
            np.save(
                f'{self.save_to}/{self.group}_train_normalized_loss_{self.template_params["tag"]}.npy',
                train_inputs['normalized_loss']
                )
            np.save(f'{self.save_to}/{self.group}_train_true_{self.template_params["tag"]}.npy', train_true)

            np.save(
                f'{self.save_to}/{self.group}_val_xnn_distribution_{self.template_params["tag"]}.npy',
                val_inputs['xnn_distribution']
                )
            np.save(
                f'{self.save_to}/{self.group}_val_normalized_loss_{self.template_params["tag"]}.npy',
                val_inputs['normalized_loss']
                )
            np.save(f'{self.save_to}/{self.group}_val_true_{self.template_params["tag"]}.npy', val_true)
        else:
            train_inputs = {
                'xnn_distribution': np.load(
                    f'{self.save_to}/{self.group}_train_xnn_distribution_{self.template_params["tag"]}.npy'
                    ),
                'normalized_loss': np.load(
                    f'{self.save_to}/{self.group}_train_normalized_loss_{self.template_params["tag"]}.npy'
                    )
                }
            train_true = np.load(f'{self.save_to}/{self.group}_train_true_{self.template_params["tag"]}.npy')
            val_inputs = {
                'xnn_distribution': np.load(
                    f'{self.save_to}/{self.group}_val_xnn_distribution_{self.template_params["tag"]}.npy'
                    ),
                'normalized_loss': np.load(
                    f'{self.save_to}/{self.group}_val_normalized_loss_{self.template_params["tag"]}.npy'
                    )
                }
            val_true = np.load(f'{self.save_to}/{self.group}_val_true_{self.template_params["tag"]}.npy')

        """
        centers = (self.xnn_bins[0][1:] + self.xnn_bins[0][:-1]) / 2
        bin_width = centers[1] - centers[0]
        train_xnn = np.stack(data[data['train']]['reindexed_xnn'])[:, self.y_indices]
        for index in range(10):
            fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
            axes[0].bar(
                centers, 
                train_inputs['xnn_distribution'][index, :, 0], 
                width=bin_width
                )
            axes[1].bar(
                centers, 
                train_inputs['normalized_loss'][index, :, 0], 
                width=bin_width
                )
            for i in range(2):
                ylim = np.array(axes[i].get_ylim())
                axes[i].plot(
                    [train_xnn[index], train_xnn[index]], ylim,
                    color=[0.8, 0, 0]
                    )
                axes[i].plot(
                    [centers[train_true[index]], centers[train_true[index]]], 0.5*ylim,
                    color=[0, 0.8, 0]
                    )
                axes[i].set_ylim(ylim)
            axes[1].set_xlabel('Xnn')
            axes[0].set_ylabel('Distribution')
            axes[1].set_ylabel('Normalized 1/Loss')
            fig.tight_layout()
            plt.show()
        """
        self.fit_history = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.template_params['epochs'],
            shuffle=True,
            batch_size=self.template_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            )

        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.plot(
            np.arange(self.template_params['epochs']),
            self.fit_history.history['loss'],
            marker='.', label='Training',
            )
        axes.plot(
            np.arange(self.template_params['epochs']),
            self.fit_history.history['val_loss'],
            marker='v', label='Validation',
            )
        axes.set_ylabel('Loss')
        axes.set_xlabel('Epoch')
        axes.legend()
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.group}_template_training_loss_{self.template_params["tag"]}.png')
        plt.close()
        self.save()



