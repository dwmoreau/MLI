"""
- Add regularization
- Do long training

- Deep model
"""
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import scipy.stats
import scipy.special
import scipy.optimize
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
from packaging import version
import time

from Networks import mlp_model_builder
from TargetFunctions import IndexingTargetFunction
from TargetFunctions import LikelihoodLoss
from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import get_xnn_from_unit_cell
from Utilities import PairwiseDifferenceCalculator
from Utilities import read_params
from Utilities import reciprocal_uc_conversion
from Utilities import Q2Calculator
from Utilities import vectorized_resampling
from Utilities import write_params


class ExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, model_params, q2_obs, xnn, reciprocal_volume, q2_obs_scale, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        self.seed = 0
        self.sigma = self.model_params['sigma']
        params_per_filter = self.model_params['filter_length'] * (1 + self.model_params['extraction_peak_length'])
        self.params_per_volume = params_per_filter * self.model_params['n_filters']
        if self.model_params['init_method'] == 'random':
            # q2_obs are all positive since the are divided by a scale factor.
            # q2_filter then should be positive. Hence the NonNeg constraint for volumes and filters
            #self.volumes = tf.cast(
            #        np.linspace(0.75, 7, self.model_params['n_volumes']),
            #        dtype=tf.float32
            #        )[:, tf.newaxis, tf.newaxis]

            self.volumes = tf.cast(
                    np.linspace(1, 5, self.model_params['n_volumes']),
                    dtype=tf.float32
                    )[:, tf.newaxis, tf.newaxis]

            self.filters = self.add_weight(
                shape=(1, self.model_params['n_filters'], self.model_params['filter_length']),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=0.01, maxval=1, seed=None
                    ),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=tf.keras.constraints.NonNeg(),
                name='filters'
                )

            self.amplitude_logits = self.add_weight(
                shape=(
                    1, 1,
                    self.model_params['n_filters'],
                    self.model_params['filter_length'],
                    self.model_params['extraction_peak_length']
                    ),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=None,
                name='amplitude_logits'
                )
        elif self.model_params['init_method'] == 'distribution':
            rng = np.random.default_rng(self.seed)
            reciprocal_volume_sorted = np.sort(reciprocal_volume)
            upper_volume_limit = reciprocal_volume_sorted[int(0.999*reciprocal_volume_sorted.size)]
            bins_vol = np.linspace(0, upper_volume_limit, 401)
            centers_vol = (bins_vol[1:] + bins_vol[:-1]) / 2
            reciprocal_volume_hist, _ = np.histogram(reciprocal_volume, bins=bins_vol, density=True)
            reciprocal_volume_hist = scipy.signal.medfilt(reciprocal_volume_hist, kernel_size=3)
            reciprocal_volume_rv = scipy.stats.rv_histogram(
                (reciprocal_volume_hist, bins_vol), density=True
                )
            reciprocal_volume_samples = reciprocal_volume_rv.ppf(np.linspace(
                0.001, 1, self.model_params['n_volumes']
                ))
            distribution_volumes = (reciprocal_volume_samples / q2_obs_scale**2)**(2/3)
            self.volumes = tf.cast(
                distribution_volumes, dtype=tf.float32
                )[:, tf.newaxis, tf.newaxis]

            q2_obs_scaled = q2_obs / q2_obs_scale
            q2_obs_scaled_sorted = np.sort(
                q2_obs_scaled[:, :self.model_params['extraction_peak_length']].ravel()
                )
            upper_q2_obs_scaled_limit = q2_obs_scaled_sorted[int(0.995*q2_obs_scaled_sorted.size)]
            bins_q2_obs_scaled = np.linspace(0, upper_q2_obs_scaled_limit, 201)
            centers_q2_obs_scaled = (bins_q2_obs_scaled[1:] + bins_q2_obs_scaled[:-1]) / 2
            q2_obs_scaled_rv = [None for _ in range(self.model_params['extraction_peak_length'])]
            for index in range(self.model_params['extraction_peak_length']):
                q2_obs_scaled_hist, _ = np.histogram(
                    q2_obs_scaled[:, index], bins=bins_q2_obs_scaled, density=True
                    )
                q2_obs_scaled_hist = scipy.signal.medfilt(q2_obs_scaled_hist, kernel_size=3)
                q2_obs_scaled_rv[index] = scipy.stats.rv_histogram(
                    (q2_obs_scaled_hist, bins_q2_obs_scaled), density=True
                    )

            q2_filters = np.zeros((self.model_params['n_filters'], self.model_params['filter_length']))
            for filter_index in range(self.model_params['n_filters']):
                peak_indices = np.sort(rng.choice(
                    self.model_params['extraction_peak_length'],
                    self.model_params['filter_length'],
                    replace=False
                    ))
                for filter_peak_index, peak_index in enumerate(peak_indices):
                    q2_filters[filter_index, filter_peak_index] = q2_obs_scaled_rv[peak_index].rvs()

            self.filters = self.add_weight(
                shape=(1, self.model_params['n_filters'], self.model_params['filter_length']),
                initializer=tf.keras.initializers.Zeros(),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=tf.keras.constraints.NonNeg(),
                name='filters'
                )
            self.filters.assign(tf.cast(q2_filters, dtype=tf.float32)[tf.newaxis])

            self.amplitude_logits = self.add_weight(
                shape=(
                    1, 1,
                    self.model_params['n_filters'],
                    self.model_params['filter_length'],
                    self.model_params['extraction_peak_length']
                    ),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=None,
                name='amplitude_logits'
                )

        self.filters_init = self.filters.numpy()[0]
        self.amplitude_logits_init = self.amplitude_logits.numpy()[0, 0]

    def call(self, q2_obs_scaled, **kwargs):
        # filters:     1, n_filters, filter_length
        # volumes:     n_volumes, 1, 1
        # q2_filters:  n_volumes, n_filters, filter_length
        # q2_obs:      batch_size, extraction_peak_length
        # difference: batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        q2_filters = (self.volumes * self.filters)[tf.newaxis, :, :, :, tf.newaxis]
        difference = q2_filters - q2_obs_scaled[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        amplitudes = self.model_params['filter_length'] * tf.keras.activations.softmax(self.amplitude_logits, axis=4) # peak axis
        distances = amplitudes * tf.math.exp(-1/2 * (difference / self.sigma)**2)
        # distances: batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        # metric:    batch_size, n_volumes, n_filters
        if self.model_params['metric_type'] == 'integral':
            metric = tf.reduce_sum(distances, axis=(3, 4))
        elif self.model_params['metric_type'] == 'kl_div':
            metric = tf.reduce_prod(tf.reduce_sum(distances, axis=4), axis=3)
        return metric

    def loss_function_common(self, y_true, y_pred):
        # y_true: batch_size, unit_cell_length
        # y_pred: batch_size, n_volumes, unit_cell_length + 1
        xnn_scaled_pred = y_pred[:, :, :self.model_params['unit_cell_length']]
        logits = y_pred[:, :, self.model_params['unit_cell_length']]
        probabilities = tf.nn.softmax(logits)
        errors = y_true[:, tf.newaxis, :] - xnn_scaled_pred
        return errors, probabilities

    def loss_function_log_cosh(self, y_true, y_pred):
        errors, probabilities = self.loss_function_common(y_true, y_pred)
        losses = tf.reduce_sum(tf.math.log(tf.math.cosh(errors)), axis=2)
        return tf.reduce_sum(losses * probabilities, axis=1)

    def loss_function_mse(self, y_true, y_pred):
        errors, probabilities = self.loss_function_common(y_true, y_pred)
        losses = 1/2 * tf.reduce_mean(errors**2, axis=2)
        return tf.reduce_sum(losses * probabilities, axis=1)

    def evaluate_weights(self, q2_obs, save_to, split_group, tag):
        filters_opt = self.filters.numpy()[0]
        amplitude_logits_opt = self.amplitude_logits.numpy()[0, 0]

        metric_max = np.zeros(q2_obs.shape[0])
        batch_size = 64
        n_batchs = q2_obs.shape[0] // batch_size
        for batch_index in range(n_batchs):
            start = batch_index * batch_size
            stop = (batch_index + 1) * batch_size
            metric = self.call(q2_obs[start:stop, :self.model_params['extraction_peak_length']])
            metric_max[start: stop] = metric.numpy().max(axis=(1, 2))
        start = (batch_index + 1) * batch_size
        metric = self.call(q2_obs[start:, :self.model_params['extraction_peak_length']])
        metric_max[start:] = metric.numpy().max(axis=(1, 2))

        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.hist(metric_max, bins=100)
        axes.set_xlabel('Maximum metric per entry')
        axes.set_ylabel('Counts')
        fig.tight_layout()
        fig.savefig(f'{save_to}/{split_group}_pitf_metric_max_{tag}.png')
        plt.close()

        # filter:             n_fitlers, filter_length
        amplitudes_opt = scipy.special.softmax(amplitude_logits_opt, axis=2)
        sort_indices = np.argsort(filters_opt, axis=1)
        for index in range(self.model_params['n_filters']):
            amplitudes_opt[index, :, :] = amplitudes_opt[index, sort_indices[index], :]
        # plot sigma & amplitudes vs position
        # amplitude mean, std, max
        fig, axes = plt.subplots(1, 3, figsize=(8, 2), sharex=True, sharey=True)
        imshow0 = axes[0].imshow(amplitudes_opt.mean(axis=0), aspect='auto')
        imshow1 = axes[1].imshow(amplitudes_opt.std(axis=0), aspect='auto')
        imshow2 = axes[2].imshow(amplitudes_opt.max(axis=0), aspect='auto')
        fig.colorbar(imshow0, ax=axes[0])
        fig.colorbar(imshow1, ax=axes[1])
        fig.colorbar(imshow2, ax=axes[2])
        axes[0].set_title('Mean Amplitude')
        axes[1].set_title('STD Amplitude')
        axes[2].set_title('Max Amplitude')
        for i in range(3):
            axes[i].set_xlabel('Peak List Position')
        axes[0].set_ylabel('Filter Position')
        fig.tight_layout()
        fig.savefig(f'{save_to}/{split_group}_pitf_amplitudes_{tag}.png')
        plt.close()

        ###############################################
        # Plot histogram of weights and their changes #
        ###############################################
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        alpha = 0.75
        fig, axes = plt.subplots(2, 2, figsize=(6, 5))
        axes[0, 0].hist(
            self.filters_init.ravel(), bins=10, color=colors[0], label='Init'
            )
        axes[0, 0].hist(
            filters_opt.ravel(), bins=10, color=colors[1], alpha=alpha, label='Optimized'
            )
        axes[1, 0].hist(
            self.filters_init.ravel() - filters_opt.ravel(), bins=10, color=colors[2], label='Init - Optimized'
            )

        axes[0, 1].hist(
            self.amplitude_logits_init.ravel(), bins=10, color=colors[0], label='Init'
            )
        axes[0, 1].hist(
            amplitude_logits_opt.ravel(), bins=10, color=colors[1], alpha=alpha, label='Optimized'
            )
        axes[1, 1].hist(
            self.amplitude_logits_init.ravel() - amplitude_logits_opt.ravel(), bins=10, color=colors[2], label='Init - Optimized'
            )
        
        axes[0, 0].set_title('Filter Weights')
        axes[0, 1].set_title('Amplitude Weights')
        for i in range(2):
            axes[0, i].set_xlabel('Value')
            axes[1, i].set_xlabel('Difference')
        axes[0, 0].legend()
        axes[1, 0].legend()
        fig.tight_layout()
        fig.savefig(f'{save_to}/{split_group}_pitf_weights_{tag}.png')
        plt.close()

    def evaluate_init(self, q2_obs_scaled, save_to, split_group, tag):
        q2_filters = (self.volumes * self.filters).numpy()
        
        bins = np.linspace(0, 5, 101)
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.hist(q2_obs_scaled.ravel(), bins=bins, label='q2_obs_scaled', density=True)
        axes.hist(q2_filters.ravel(), bins=bins, alpha=0.75, label='q2_filters', density=True)
        axes.set_xlabel('q2 scaled')
        axes.set_ylabel('distribution')
        axes.legend()
        fig.tight_layout()
        fig.savefig(f'{save_to}/{split_group}_pitf_filter_init_{tag}.png')
        plt.close()

        bins = np.linspace(0, 2, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        metric_zeros = 0
        metric_counts = 0
        metric_hist = np.zeros(100)
        batch_size = 32
        n_batchs = q2_obs_scaled.shape[0] // batch_size
        for batch_index in range(n_batchs + 1):
            start = batch_index * batch_size
            if batch_index == n_batchs:
                stop = -1
            else:
                stop = (batch_index + 1) * batch_size
            metric = self.call(q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']]).numpy()
            zero = np.isclose(metric, 0)
            metric_counts += metric.size
            metric_zeros += zero.sum()
            metric_hist_batch, _ = np.histogram(metric[~zero], bins=bins, density=False)
            metric_hist += metric_hist_batch

        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.bar(centers, metric_hist, width=bins[1] - bins[0])
        axes.set_title(f'{100*metric_zeros/metric_counts}% of metrics are zero')
        axes.set_xlabel('Metric')
        axes.set_ylabel('distribution')
        fig.tight_layout()
        fig.savefig(f'{save_to}/{split_group}_pitf_metric_init_{tag}.png')
        plt.close()


class PhysicsInformedModel:
    def __init__(self, split_group, data_params, model_params, save_to, seed, q2_scaler, xnn_scaler, hkl_ref):
        self.split_group = split_group
        self.data_params = data_params
        self.model_params = model_params

        self.n_peaks = data_params['n_peaks']
        self.unit_cell_length = data_params['unit_cell_length']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.save_to = save_to
        self.save_to_split_group = os.path.join(self.save_to, split_group)
        if not os.path.exists(self.save_to_split_group):
            os.mkdir(self.save_to_split_group)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.lattice_system = self.data_params['lattice_system']

        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()

    def setup(self, data):
        model_params_defaults = {
            'peak_length': 20,
            'extraction_peak_length': 6,
            'filter_length': 3,
            'n_volumes': 200,
            'n_filters': 200,
            'metric_layers': [200, 100, 50],
            'metric_dropout_rate_extraction': 0.0,
            'metric_dropout_rate': 0.0,
            'metric_orthogonal_regularization': 0.0,
            'base_line_layers': [200, 100, 50],
            'base_line_dropout_rate': 0.0,
            'learning_rate': 0.00005,
            'epochs': 50,
            'batch_size': 64,
            'loss_type': 'mse',
            'init_method': 'random',
            'augment': False,
            'model_type': 'metric',
            'metric_type': 'integral',
            }

        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        self.model_params['unit_cell_length'] = self.unit_cell_length
        self.build_model(data=data)

    def save(self):
        write_params(self.model_params, f'{self.save_to_split_group}/{self.split_group}_pitf_params_{self.model_params["tag"]}.csv')
        self.model.save_weights(f'{self.save_to_split_group}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5')
        np.save(
            f'{self.save_to_split_group}/{self.split_group}_q2_obs_scale_{self.model_params["tag"]}.npy',
            self.q2_obs_scale
            )
        np.save(
            f'{self.save_to_split_group}/{self.split_group}_xnn_scaler_{self.model_params["tag"]}.npy',
            np.array((self.xnn_mean, self.xnn_scale))
            )

    def load_from_tag(self):
        params = read_params(f'{self.save_to_split_group}/{self.split_group}_pitf_params_{self.model_params["tag"]}.csv')
        params_keys = [
            'tag',
            'peak_length',
            'extraction_peak_length',
            'filter_length',
            'n_volumes',
            'n_filters',
            'metric_layers',
            'metric_dropout_rate_extraction',
            'metric_dropout_rate',
            'metric_orthogonal_regularization',
            'base_line_layers',
            'base_line_dropout_rate',
            'learning_rate',
            'epochs',
            'batch_size',
            'loss_type',
            'init_method',
            'augment',
            'model_type',
            'metric_type',
            'sigma',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['peak_length'] = int(params['peak_length'])
        self.model_params['extraction_peak_length'] = int(params['extraction_peak_length'])
        self.model_params['filter_length'] = int(params['filter_length'])
        self.model_params['n_volumes'] = int(params['n_volumes'])
        self.model_params['n_filters'] = int(params['n_filters'])
        self.model_params['metric_layers'] = np.array(
            params['metric_layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['base_line_layers'] = np.array(
            params['base_line_layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['metric_dropout_rate_extraction'] = float(params['metric_dropout_rate_extraction'])
        self.model_params['metric_dropout_rate'] = float(params['metric_dropout_rate'])
        self.model_params['metric_orthogonal_regularization'] = float(params['metric_orthogonal_regularization'])
        self.model_params['base_line_dropout_rate'] = float(params['base_line_dropout_rate'])
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['epochs'] = int(params['epochs'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['loss_type'] = params['loss_type']
        self.model_params['init_method'] = params['init_method']
        if self.model_params['augment'] == 'True':
            self.model_params['augment'] = True
        else:
            self.model_params['augment'] = False
        self.model_params['model_type'] = params['model_type']
        self.model_params['metric_type'] = params['metric_type']
        self.model_params['sigma'] = float(params['metric_type'])

        self.build_model(data=None)
        self.compile_model()
        self.model.load_weights(
            filepath=f'{self.save_to_split_group}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )
        self.q2_obs_scale = np.load(
            f'{self.save_to_split_group}/{self.split_group}_q2_obs_scale_{self.model_params["tag"]}.npy',
            )
        self.xnn_mean, self.xnn_scale = np.load(
            f'{self.save_to_split_group}/{self.split_group}_xnn_scaler_{self.model_params["tag"]}.npy',
            )

    def build_model(self, data=None):
        data = data[data['train']]
        #data = data[~data['augmented']]
        q2_obs = np.stack(data['q2'])[:, :self.model_params['extraction_peak_length']]
        self.q2_obs_scale = q2_obs.std()
        unit_cell = np.stack(data['reindexed_unit_cell'])[:, self.unit_cell_indices]
        xnn = get_xnn_from_unit_cell(unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        reciprocal_unit_cell = reciprocal_uc_conversion(
            unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        reciprocal_volume = get_unit_cell_volume(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.extraction_layer = ExtractionLayer(
            self.model_params, q2_obs, xnn, reciprocal_volume, self.q2_obs_scale
            )
        inputs = {
            'q2_obs_scaled': tf.keras.Input(
                shape=self.model_params['peak_length'],
                name='q2_obs_scaled',
                dtype=tf.float32,
                )
            }
        if self.model_params['model_type'] == 'metric':
            self.model = tf.keras.Model(inputs, self.model_builder_metric(inputs))
        elif self.model_params['model_type'] == 'combined':
            self.model = tf.keras.Model(inputs, self.model_builder_combined(inputs))
        elif self.model_params['model_type'] == 'base_line':
            self.model = tf.keras.Model(inputs, self.model_builder_base_line(inputs))
        self.compile_model()
        self.model.summary()

    def model_builder_metric(self, inputs):
        # inputs['q2_obs_scaled']: batch_size, n_peaks
        # metric:                  batch_size, n_volumes, n_filters
        metric = self.extraction_layer(
            inputs['q2_obs_scaled'][:, :self.model_params['extraction_peak_length']]
            )
        x = tf.keras.layers.LayerNormalization(
                name=f'metric_layer_norm_extraction',
                axis=1,
                center=False,
                )(metric)
        x = tf.keras.layers.SpatialDropout1D(
            rate=self.model_params['metric_dropout_rate_extraction'],
            name=f'metric_dropout_extraction',
            )(x)

        if self.model_params['metric_orthogonal_regularization'] == 0:
            regularizer = None
        else:
            regularizer = tf.keras.regularizers.OrthogonalRegularizer(
                factor=self.model_params['metric_orthogonal_regularization'],
                mode='rows'
                )
        for index in range(len(self.model_params['metric_layers'])):
            x = tf.keras.layers.Dense(
                self.model_params['metric_layers'][index],
                activation='linear',
                name=f'metric_dense_{index}',
                use_bias=False,
                kernel_regularizer=regularizer,
                )(x)
            x = tf.keras.layers.LayerNormalization(
                name=f'metric_layer_norm_{index}',
                axis=2,
                center=True,
                )(x)
            x = tf.keras.activations.gelu(x)
            x = tf.keras.layers.Dropout(
                rate=self.model_params['metric_dropout_rate'],
                name=f'metric_dropout_{index}',
                )(x)

        # output: batch_size, n_volumes, unit_cell_length + 1
        output = tf.keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name='metric_xnn_scaled',
            )(x)
        return output

    def model_builder_base_line(self, inputs):
        # inputs['q2_obs_scaled']: batch_size, n_peaks
        # This is a 'Base line model' that does not use feature extraction
        x = inputs['q2_obs_scaled']
        for index in range(len(self.model_params['base_line_layers'])):
            x = tf.keras.layers.Dense(
                self.model_params['base_line_layers'][index],
                activation='linear',
                name=f'base_line_dense_{index}',
                use_bias=True,
                )(x)
            x = tf.keras.layers.LayerNormalization(
                name=f'base_line_layer_norm_{index}',
                center=False,
                )(x)
            x = tf.keras.activations.gelu(x)
            x = tf.keras.layers.Dropout(
                rate=self.model_params['base_line_dropout_rate'],
                name=f'base_line_dropout_{index}',
                )(x)

        # output: batch_size, n_volumes, unit_cell_length + 1
        output = tf.keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name='base_line_xnn_scaled',
            )(x[:, tf.newaxis, :])
        return output

    def model_builder_combined(self, inputs):
        return tf.keras.layers.Concatenate(axis=1, name='combined_xnn_scaled')((
            self.model_builder_base_line(inputs),
            self.model_builder_metric(inputs)
            ))

    def compile_model(self):
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate'])    # loss: 1.4367 - val_loss: 1.2484

        if self.model_params['loss_type'] == 'mse':
            loss_functions = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.loss_function_mse
                }
            loss_metrics = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.loss_function_log_cosh
                }
        else:
            loss_functions = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.loss_function_log_cosh
                }
            loss_metrics = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.loss_function_mse
                }
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            metrics=loss_metrics,
            run_eagerly=False,
            )

    def train(self, data):
        if self.model_params['augment'] == False:
            data = data[~data['augmented']]
        train = data[data['train']]
        val = data[~data['train']]

        train_q2_obs = np.stack(train['q2'])[:, :self.model_params['peak_length']]
        val_q2_obs = np.stack(val['q2'])[:, :self.model_params['peak_length']]
        train_q2_obs_scaled = train_q2_obs / self.q2_obs_scale
        val_q2_obs_scaled = val_q2_obs / self.q2_obs_scale

        train_unit_cell = np.stack(train['reindexed_unit_cell'])[:, self.unit_cell_indices]
        val_unit_cell = np.stack(val['reindexed_unit_cell'])[:, self.unit_cell_indices]
        train_xnn = get_xnn_from_unit_cell(train_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        val_xnn = get_xnn_from_unit_cell(val_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        
        self.xnn_mean = np.median(train_xnn, axis=0)[np.newaxis]
        self.xnn_scale = np.median(np.abs(train_xnn - self.xnn_mean), axis=0)[np.newaxis]
        train_xnn_scaled = (train_xnn - self.xnn_mean) / self.xnn_scale
        val_xnn_scaled = (val_xnn - self.xnn_mean) / self.xnn_scale

        fig, axes = plt.subplots(1, self.unit_cell_length + 1, figsize=(6, 3))
        bins0 = np.linspace(0, 5, 301)
        bins1 = np.linspace(-5, 5, 301)
        xnn_titles = ['Xhh', 'Xkk', 'Xll', 'Xkl', 'Xhl', 'Xhk']
        for index in range(self.unit_cell_length + 1):
            if index == 0:
                axes[index].hist(train_q2_obs_scaled.ravel(), bins=bins0, density=True)
                axes[index].plot(bins0, 2/np.sqrt(2*np.pi)*np.exp(-1/2*bins0**2), color=[1, 0, 0])
            else:
                axes[index].hist(train_xnn_scaled[:, index - 1], bins=bins1, density=True)
                axes[index].plot(bins1, 1/np.sqrt(2*np.pi)*np.exp(-1/2*bins1**2), color=[1, 0, 0])
        axes[0].set_title('q2_obs_scaled')
        fig.tight_layout()
        fig.savefig(f'{self.save_to_split_group}/{self.split_group}_pitf_io_{self.model_params["tag"]}.png')
        plt.close()

        train_true = {f'{self.model_params["model_type"]}_xnn_scaled': train_xnn_scaled}
        val_true = {f'{self.model_params["model_type"]}_xnn_scaled': val_xnn_scaled}
        train_inputs = {'q2_obs_scaled': train_q2_obs_scaled}
        val_inputs = {'q2_obs_scaled': val_q2_obs_scaled}
        if self.model_params['model_type'] in ['metric', 'combined']:
            self.extraction_layer.evaluate_init(
                train_q2_obs_scaled, self.save_to_split_group, self.split_group, self.model_params["tag"]
                )
        self.fit_history = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
        self.save()

        ##############################
        # Plot training loss vs time #
        ##############################
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        if self.model_params['loss_type'] == 'mse':
            mse_label = 'loss'
            log_cosh_label = 'loss_function_log_cosh'
        else:
            mse_label = 'loss_function_mse'
            log_cosh_label = 'loss'
        axes[0].plot(
            self.fit_history.history[mse_label], 
            label='Training', marker='.'
            )
        axes[0].plot(
            self.fit_history.history[f'val_{mse_label}'], 
            label='Validation', marker='v'
            )
        axes[1].plot(
            self.fit_history.history[log_cosh_label], 
            label='Training', marker='.'
            )
        axes[1].plot(
            self.fit_history.history[f'val_{log_cosh_label}'], 
            label='Validation', marker='v'
            )
        axes[0].set_ylabel('MSE')
        axes[1].set_ylabel('Log-Cosh Error')
        axes[1].set_xlabel('Epoch')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(f'{self.save_to_split_group}/{self.split_group}_pitf_training_loss_{self.model_params["tag"]}.png')
        plt.close()

    def evaluate(self, data):
        data = data[~data['augmented']]
        train = data[data['train']]
        val = data[~data['train']]

        train_q2_obs = np.stack(train['q2'])[:, :self.model_params['peak_length']]
        val_q2_obs = np.stack(val['q2'])[:, :self.model_params['peak_length']]
        train_q2_obs_scaled = train_q2_obs / self.q2_obs_scale
        val_q2_obs_scaled = val_q2_obs / self.q2_obs_scale

        train_inputs = {'q2_obs_scaled': train_q2_obs_scaled}
        val_inputs = {'q2_obs_scaled': val_q2_obs_scaled}

        train_unit_cell = np.stack(train['reindexed_unit_cell'])[:, self.unit_cell_indices]
        val_unit_cell = np.stack(val['reindexed_unit_cell'])[:, self.unit_cell_indices]
        train_xnn = get_xnn_from_unit_cell(train_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        val_xnn = get_xnn_from_unit_cell(val_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        train_xnn_scaled = (train_xnn - self.xnn_mean) / self.xnn_scale
        val_xnn_scaled = (val_xnn - self.xnn_mean) / self.xnn_scale

        val_pred = self.model.predict(val_inputs)

        val_all_xnn_scaled_pred = val_pred[:, :, :self.unit_cell_length]
        val_logits = val_pred[:, :, self.unit_cell_length]
        val_softmax = scipy.special.softmax(val_logits, axis=1)
        val_xnn_scaled_pred_top5 = np.take_along_axis(
            val_all_xnn_scaled_pred,
            np.argsort(val_softmax, axis=1)[:, ::-1][:, :5, np.newaxis],
            axis=1
            )
        val_xnn_pred_top5 = val_xnn_scaled_pred_top5*self.xnn_scale + self.xnn_mean
        val_unit_cell_pred_top5 = np.zeros(val_xnn_pred_top5.shape)
        for index in range(5):
            val_unit_cell_pred_top5[:, index, :] = get_unit_cell_from_xnn(
                val_xnn_pred_top5[:, index, :], partial_unit_cell=True, lattice_system=self.lattice_system
                )

        train_pred = self.model.predict(train_inputs)
        train_all_xnn_scaled_pred = train_pred[:, :, :self.unit_cell_length]
        train_logits = train_pred[:, :, self.unit_cell_length]
        train_softmax = scipy.special.softmax(train_logits, axis=1)
        train_xnn_scaled_pred_top5 = np.take_along_axis(
            train_all_xnn_scaled_pred,
            np.argsort(train_softmax, axis=1)[:, ::-1][:, :5, np.newaxis],
            axis=1
            )
        train_xnn_pred_top5 = train_xnn_scaled_pred_top5*self.xnn_scale + self.xnn_mean
        train_unit_cell_pred_top5 = np.zeros(train_xnn_pred_top5.shape)
        for index in range(5):
            train_unit_cell_pred_top5[:, index, :] = get_unit_cell_from_xnn(
                train_xnn_pred_top5[:, index, :], partial_unit_cell=True, lattice_system=self.lattice_system
                )

        for index in range(10):
            self.plot_predictions(
                val_xnn_scaled[index],
                val_all_xnn_scaled_pred[index],
                val_softmax[index],
                index
                )

        ##############################
        # Plot unit cell evaluations #
        ##############################
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
                val_diff = np.linalg.norm(val_xnn_scaled_pred_top5 - val_xnn_scaled[:, np.newaxis, :], axis=2)
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

                train_diff = np.linalg.norm(train_xnn_scaled_pred_top5 - train_xnn_scaled[:, np.newaxis, :], axis=2)
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
            fig.savefig(f'{self.save_to_split_group}/{self.split_group}_pitf_reg_eval_{self.model_params["tag"]}_{save_label}.png')
            plt.close()

        ##########################
        # Plot branch importance #
        ##########################
        if self.model_params['model_type'] in ['metric', 'combined']:
            if self.model_params['model_type'] == 'metric':
                n_branches = self.model_params['n_volumes']
            elif self.model_params['model_type'] == 'combined':
                n_branches = self.model_params['n_volumes'] + 1
            # number of times a branch is the most probable
            # number of times a branch is in the top 5, 10
            train_rankings = np.argsort(train_softmax, axis=1)[:, ::-1]
            val_rankings = np.argsort(val_softmax, axis=1)[:, ::-1]

            train_top1 = np.bincount(train_rankings[:, 0].ravel(), minlength=n_branches)
            val_top1 = np.bincount(val_rankings[:, 0].ravel(), minlength=n_branches)
            train_top5 = np.bincount(train_rankings[:, :5].ravel(), minlength=n_branches)
            val_top5 = np.bincount(val_rankings[:, :5].ravel(), minlength=n_branches)
            train_top10 = np.bincount(train_rankings[:, :10].ravel(), minlength=n_branches)
            val_top10 = np.bincount(val_rankings[:, :10].ravel(), minlength=n_branches)

            train_gt_10p = np.bincount(np.sum(train_softmax > 0.10, axis=1), minlength=10)
            val_gt_10p = np.bincount(np.sum(val_softmax > 0.10, axis=1), minlength=10)
            train_gt_5p = np.bincount(np.sum(train_softmax > 0.05, axis=1), minlength=20)
            val_gt_5p = np.bincount(np.sum(val_softmax > 0.05, axis=1), minlength=20)
            train_gt_1p = np.bincount(np.sum(train_softmax > 0.01, axis=1), minlength=100)
            val_gt_1p = np.bincount(np.sum(val_softmax > 0.01, axis=1), minlength=100)

            fig, axes = plt.subplots(2, 3, figsize=(10, 5))
            alpha = 0.75
            width = 1
            x = np.arange(n_branches)
            axes[0, 0].bar(x, train_top1, width=width, label='Training')
            axes[0, 0].bar(x, val_top1, width=width, alpha=alpha, label='Validation')
            axes[0, 1].bar(x, train_top5, width=width, label='Training')
            axes[0, 1].bar(x, val_top5, width=width, alpha=alpha, label='Validation')
            axes[0, 2].bar(x, train_top10, width=width, label='Training')
            axes[0, 2].bar(x, val_top10, width=width, alpha=alpha, label='Validation')
            axes[0, 0].set_title('Top 1 occurances')
            axes[0, 1].set_title('Top 5 occurances')
            axes[0, 2].set_title('Top 10 occurances')
            for i in range(3):
                axes[0, i].set_xlabel('Branch')
            bins_prob_frac = np.linspace(0, 1, 101)
            x_prob_frac = (bins_prob_frac[1:] + bins_prob_frac[:-1]) / 2
            axes[1, 0].bar(np.arange(10), train_gt_10p, width=width, label='Training')
            axes[1, 0].bar(np.arange(10), val_gt_10p, width=width, alpha=alpha, label='Validation')
            axes[1, 1].bar(np.arange(20), train_gt_5p, width=width, label='Training')
            axes[1, 1].bar(np.arange(20), val_gt_5p, width=width, alpha=alpha, label='Validation')
            axes[1, 2].bar(np.arange(100), train_gt_1p, width=width, label='Training')
            axes[1, 2].bar(np.arange(100), val_gt_1p, width=width, alpha=alpha, label='Validation')
            axes[1, 0].set_title('Number > 10%')
            axes[1, 1].set_title('Number > 5%')
            axes[1, 2].set_title('Number > 1%')
            for i in range(3):
                axes[1, i].set_xlabel('Counts')

            fig.tight_layout()
            fig.savefig(f'{self.save_to_split_group}/{self.split_group}_pitf_branch_importance_{self.model_params["tag"]}.png')
            plt.close()

            self.extraction_layer.evaluate_weights(
                train_inputs['q2_obs_scaled'], 
                self.save_to_split_group,
                self.split_group,
                self.model_params["tag"]
                )

    def plot_predictions(self, xnn_true, xnn_pred, softmax, index):
        if self.lattice_system == 'orthorhombic':
            fig, axes = plt.subplots(1, 3, figsize=(7, 3))
            axes[0].scatter(xnn_pred[:, 0], xnn_pred[:, 1], c=softmax)
            axes[1].scatter(xnn_pred[:, 0], xnn_pred[:, 2], c=softmax)
            axes[2].scatter(xnn_pred[:, 1], xnn_pred[:, 2], c=softmax)
            axes[0].plot(xnn_true[0], xnn_true[1], marker='X', color=[1, 0, 0])
            axes[1].plot(xnn_true[0], xnn_true[2], marker='X', color=[1, 0, 0])
            axes[2].plot(xnn_true[1], xnn_true[2], marker='X', color=[1, 0, 0])
            axes[0].set_xlabel('Xhh (scaled)')
            axes[1].set_xlabel('Xhh (scaled)')
            axes[2].set_xlabel('Xkk (scaled)')
            axes[0].set_ylabel('Xkk (scaled)')
            axes[1].set_ylabel('Xll (scaled)')
            axes[2].set_ylabel('Xll (scaled)')
        elif self.lattice_system in ['tetragonal', 'hexagonal']:
            fig, axes = plt.subplots(1, 1, figsize=(5, 3))
            axes.scatter(xnn_pred[:, 0], xnn_pred[:, 1], c=softmax)
            axes.plot(xnn_true[0], xnn_true[1], marker='X', color=[1, 0, 0])
            axes.set_ylabel('Xhh (scaled)')
            axes.set_xlabel('Xll (scaled)')
        else:
            return None
        fig.tight_layout()
        fig.savefig(f'{self.save_to_split_group}/{self.split_group}_pitf_example_{index}_{self.model_params["tag"]}.png')
        plt.close()


