"""
- "Optimized" approach to initialization
- Move to orthorhombic
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
        sigma = 0.02
        self.sigma_scale = 0.01
        self.sigma_mean = tf.math.log(tf.math.exp(sigma) - 1)
        self.sigma_params = self.add_weight(
            shape=(
                1, 1, 1,
                self.model_params['filter_length'],
                self.model_params['extraction_peak_length']
                ),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
            dtype=tf.float32,
            trainable=True,
            regularizer=None,
            name='sigma_params'
            )

        if self.model_params['init_method'] == 'random':
            # q2_obs are all positive since the are divided by a scale factor.
            # q2_filter then should be positive. Hence the NonNeg constraint for volumes and filters
            self.volumes = self.add_weight(
                shape=(self.model_params['n_volumes'], 1, 1),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=1.5, maxval=5, seed=None,
                    ),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=tf.keras.constraints.NonNeg(),
                name='volumes'
                )

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
            reciprocal_volume_rv = scipy.stats.rv_histogram(
                np.histogram(reciprocal_volume, bins=bins_vol, density=True),
                density=True
                )
            reciprocal_volume_samples = reciprocal_volume_rv.rvs(size=self.model_params['n_volumes'])
            random_volumes = (reciprocal_volume_samples / q2_obs_scale**2)**(2/3)
            self.volumes = self.add_weight(
                shape=(self.model_params['n_volumes'], 1, 1),
                initializer=tf.keras.initializers.Zeros(),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=tf.keras.constraints.NonNeg(),
                name='volumes'
                )
            self.volumes.assign(
                tf.cast(random_volumes, dtype=tf.float32)[:, tf.newaxis, tf.newaxis]
                )

            def filter_lims_fun(x, q2_obs_scaled, volumes, n_filters, filter_length, seed):
                rng = np.random.default_rng(seed)
                filters = rng.uniform(low=min(x), high=max(x), size=(n_filters, filter_length))
                q2_filters = volumes[:, np.newaxis, np.newaxis] * filters[np.newaxis]

                bins = np.linspace(0, 10, 201)
                db = bins[1] - bins[0]
                centers = (bins[1:] + bins[:-1]) / 2
                q2_obs_scaled_hist, _ = np.histogram(q2_obs_scaled.ravel(), bins=bins, density=False)
                q2_obs_scaled_hist = q2_obs_scaled_hist / (q2_obs_scaled.size * db)
                q2_filters_hist, _ = np.histogram(q2_filters.ravel(), bins=bins, density=False)
                q2_filters_hist = q2_filters_hist / (q2_filters.size * db)
                KL = scipy.special.kl_div(q2_filters_hist, q2_obs_scaled_hist)
                indices = np.invert(np.logical_or(np.isnan(KL), np.isinf(KL)))
                return np.trapz(KL[indices], centers[indices])

            filter_lims = np.zeros((10, 2))
            for seed in range(10):
                results = scipy.optimize.minimize(
                    fun=filter_lims_fun,
                    x0=[0, 1.5],
                    bounds=[[0, np.inf], [0, np.inf]],
                    args=(
                        q2_obs / q2_obs_scale,
                        random_volumes,
                        self.model_params['n_filters'],
                        self.model_params['filter_length'],
                        seed
                        ),
                    method='Nelder-Mead',
                    )
                filter_lims[seed, 0] = min(results.x)
                filter_lims[seed, 1] = max(results.x)
            q2_filters = rng.uniform(
                low=filter_lims.mean(axis=0)[0],
                high=filter_lims.mean(axis=0)[1],
                size=(self.model_params['n_filters'], self.model_params['filter_length'])
                )
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
        elif self.model_params['init_method'] == 'informed':
            mean_reciprocal_volume = reciprocal_volume.mean()
            amplitude_base = 1
            amplitude_frac = 1/2
            rng = np.random.default_rng(self.seed)
            if xnn.shape[0] >= self.model_params['n_filters']:
                replace = False
            else:
                replace = True
            entry_indices = rng.choice(
                xnn.shape[0], size=self.model_params['n_filters'], replace=replace
                )
            q2_filters = np.zeros((
                self.model_params['n_filters'],
                self.model_params['filter_length']
                ))
            amplitude_logits = np.zeros((
                self.model_params['n_filters'],
                self.model_params['filter_length'],
                self.model_params['extraction_peak_length']
                ))
            for filter_index, entry_index in enumerate(entry_indices):
                peak_indices = np.sort(rng.choice(
                    self.model_params['extraction_peak_length'],
                    size=self.model_params['filter_length'],
                    replace=False
                    ))
                q2_filters[filter_index] = q2_obs[entry_index, peak_indices]
                q2_filters[filter_index] *= (mean_reciprocal_volume / reciprocal_volume[entry_index])**(2/3)
                for filter_position in range(self.model_params['filter_length']):
                    peak_position = peak_indices[filter_position]
                    amplitude_logits[filter_index, filter_position, peak_position] = amplitude_base
                    if peak_position == 0:
                        amplitude_logits[filter_index, filter_position, peak_position + 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position + 2] = amplitude_base * amplitude_frac**2
                    elif peak_position == 1:
                        amplitude_logits[filter_index, filter_position, peak_position - 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position + 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position + 2] = amplitude_base * amplitude_frac**2
                    elif peak_position == self.model_params['extraction_peak_length'] - 1:
                        amplitude_logits[filter_index, filter_position, peak_position - 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position - 2] = amplitude_base * amplitude_frac**2
                    elif peak_position == self.model_params['extraction_peak_length'] - 2:
                        amplitude_logits[filter_index, filter_position, peak_position + 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position - 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position - 2] = amplitude_base * amplitude_frac**2
                    else:
                        amplitude_logits[filter_index, filter_position, peak_position + 2] = amplitude_base * amplitude_frac**2
                        amplitude_logits[filter_index, filter_position, peak_position + 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position - 1] = amplitude_base * amplitude_frac
                        amplitude_logits[filter_index, filter_position, peak_position - 2] = amplitude_base * amplitude_frac**2
            q2_filters /= q2_obs_scale
            q2_filters += rng.normal(
                loc=0, scale=0.1*sigma, size=q2_filters.shape
                )
            amplitude_logits += rng.normal(
                loc=0, scale=0.25*amplitude_base, size=amplitude_logits.shape
                )

            scale = (reciprocal_volume / mean_reciprocal_volume)**(2/3)
            bins = np.linspace(0, np.sort(scale)[int(0.99*scale.size)], 101)
            scale_distribution = scipy.stats.rv_histogram(
                histogram=np.histogram(scale, bins=bins, density=True)
                )
            random_volumes = scale_distribution.rvs(
                size=self.model_params['n_volumes'],
                random_state=rng
                )
            self.volumes = self.add_weight(
                shape=(self.model_params['n_volumes'], 1, 1),
                initializer=tf.keras.initializers.Zeros(),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=tf.keras.constraints.NonNeg(),
                name='volumes'
                )
            self.volumes.assign(tf.cast(random_volumes, dtype=tf.float32)[:, tf.newaxis, tf.newaxis])

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
                initializer=tf.keras.initializers.Zeros(),
                dtype=tf.float32,
                trainable=True,
                regularizer=None,
                constraint=None,
                name='amplitude_logits'
                )
            self.amplitude_logits.assign(tf.cast(amplitude_logits, dtype=tf.float32)[tf.newaxis, tf.newaxis])

        self.volumes_init = self.volumes.numpy()[:, 0, 0]
        self.filters_init = self.filters.numpy()[0]
        self.sigma_params_init = self.sigma_params.numpy()[0, 0, 0]
        self.amplitude_logits_init = self.amplitude_logits.numpy()[0, 0]

    def call(self, q2_obs_scaled, **kwargs):
        # filters:     1, n_filters, filter_length
        # volumes:     n_volumes, 1, 1
        # q2_filters:  n_volumes, n_filters, filter_length
        # q2_obs:      batch_size, extraction_peak_length
        # difference: batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        q2_filters = (self.volumes * self.filters)[tf.newaxis, :, :, :, tf.newaxis]
        difference = q2_filters - q2_obs_scaled[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        sigma = tf.math.softplus(self.sigma_params*self.sigma_scale + self.sigma_mean)
        arg = -1/2 * (difference / sigma)**2
        #amplitudes = self.model_params['filter_length'] * tf.nn.softmax(self.amplitude_logits, axis=3)
        amplitudes = self.model_params['filter_length'] * tf.nn.softmax(self.amplitude_logits, axis=4)
        #amplitudes = self.model_params['filter_length'] * tf.nn.softmax(self.amplitude_logits, axis=(3, 4))
        distances = amplitudes * tf.math.exp(arg)
        # distances: batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        # metric:    batch_size, n_volumes, n_filters
        metric = tf.reduce_sum(distances, axis=(3, 4))
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
        volumes_opt = self.volumes.numpy()[:, 0, 0]
        filters_opt = self.filters.numpy()[0]
        sigma_params_opt = self.sigma_params.numpy()[0, 0, 0]
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

        # sigma / amplitudes: n_filters, filter_length, extraction_peak_length
        # filter:             n_fitlers, filter_length
        sigma_opt = np.log(1 + np.exp(sigma_params_opt*self.sigma_scale + self.sigma_mean))
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

        """
        # sigma mean, amplitude weighted mean, std
        fig, axes = plt.subplots(2, 2, figsize=(5, 4), sharex=True, sharey=True)
        imshow00 = axes[0, 0].imshow(sigma_opt.mean(axis=0), aspect='auto')
        imshow01 = axes[0, 1].imshow(sigma_opt.std(axis=0), aspect='auto')
        imshow10 = axes[1, 0].imshow((sigma_opt * amplitudes_opt).mean(axis=0), aspect='auto')
        imshow11 = axes[1, 1].imshow((sigma_opt * amplitudes_opt).std(axis=0), aspect='auto')
        fig.colorbar(imshow00, ax=axes[0, 0])
        fig.colorbar(imshow01, ax=axes[0, 1])
        fig.colorbar(imshow10, ax=axes[1, 0])
        fig.colorbar(imshow11, ax=axes[1, 1])
        axes[0, 0].set_title('Mean Sigma')
        axes[0, 1].set_title('STD Sigma')
        axes[1, 0].set_title('Mean Amplitude x Sigma')
        axes[1, 1].set_title('STD Amplitude x Sigma')
        for i in range(2):
            axes[1, i].set_xlabel('Peak List Position')
            axes[i, 0].set_ylabel('Filter Position')
        fig.tight_layout()
        fig.savefig(f'{save_to}/{split_group}_pitf_sigmas_{tag}.png')
        plt.close()
        """
        ###############################################
        # Plot histogram of weights and their changes #
        ###############################################
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        alpha = 0.75
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        axes[0, 0].hist(
            self.volumes_init,
            bins=10, color=colors[0], label='Init'
            )
        axes[0, 0].hist(
            volumes_opt, bins=10, color=colors[1], alpha=alpha, label='Optimized'
            )
        axes[1, 0].hist(
            self.volumes_init - volumes_opt, bins=10, color=colors[2], label='Init - Optimized'
            )

        axes[0, 1].hist(
            self.filters_init.ravel(), bins=10, color=colors[0], label='Init'
            )
        axes[0, 1].hist(
            filters_opt.ravel(), bins=10, color=colors[1], alpha=alpha, label='Optimized'
            )
        axes[1, 1].hist(
            self.filters_init.ravel() - filters_opt.ravel(), bins=10, color=colors[2], label='Init - Optimized'
            )

        axes[0, 2].hist(
            self.sigma_params_init.ravel(), bins=10, color=colors[0], label='Init'
            )
        axes[0, 2].hist(
            sigma_params_opt.ravel(), bins=10, color=colors[1], alpha=alpha, label='Optimized'
            )
        axes[1, 2].hist(
            self.sigma_params_init.ravel() - sigma_params_opt.ravel(), bins=10, color=colors[2], label='Init - Optimized'
            )

        axes[0, 3].hist(
            self.amplitude_logits_init.ravel(), bins=10, color=colors[0], label='Init'
            )
        axes[0, 3].hist(
            amplitude_logits_opt.ravel(), bins=10, color=colors[1], alpha=alpha, label='Optimized'
            )
        axes[1, 3].hist(
            self.amplitude_logits_init.ravel() - amplitude_logits_opt.ravel(), bins=10, color=colors[2], label='Init - Optimized'
            )
        
        axes[0, 0].set_title('Volume Weights')
        axes[0, 1].set_title('Filter Weights')
        axes[0, 2].set_title('Sigma Weights')
        axes[0, 3].set_title('Amplitude Weights')
        for i in range(4):
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


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, rng, q2_obs_scaled, xnn_scaled, xnn, lattice_system, batch_size, super_batch_size):
        self.rng = rng
        self.q2_obs_scaled = q2_obs_scaled
        self.xnn_scaled = xnn_scaled
        self.batch_size = batch_size
        self.super_batch_size = super_batch_size
        self.n_entries = self.q2_obs_scaled.shape[0]
        self.n_batches = int(np.floor(self.n_entries / self.batch_size))
        self.n_super_batches = int(np.floor(self.n_entries / (self.super_batch_size * self.batch_size)))

        unit_cell = get_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=lattice_system
            )
        unit_cell_volume = get_unit_cell_volume(
            unit_cell, partial_unit_cell=True, lattice_system=lattice_system
            )
        if lattice_system in ['tetragonal', 'hexagonal']:
            self.sort_indices = [
                np.arange(self.n_entries),
                np.argsort(xnn[:, 0] / xnn[:, 1]),
                np.argsort(unit_cell_volume),
                ]
        elif lattice_system == 'orthorhombic':
            self.sort_indices = [
                np.arange(self.n_entries),
                np.argsort(xnn[:, 0] / xnn[:, 1]),
                np.argsort(xnn[:, 0] / xnn[:, 2]),
                np.argsort(xnn[:, 1] / xnn[:, 2]),
                np.argsort(unit_cell_volume),
                ]
        elif lattice_system == 'monoclinic':
            self.sort_indices = [
                np.arange(self.n_entries),
                np.argsort(xnn[:, 0] / xnn[:, 1]),
                np.argsort(xnn[:, 0] / xnn[:, 2]),
                np.argsort(xnn[:, 1] / xnn[:, 2]),
                np.argsort(unit_cell[:, 3]),
                np.argsort(unit_cell_volume),
                ]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # Pick sort type (a/c, b/c, angle)
        sort_indices = self.sort_indices[
            self.rng.choice(len(self.sort_indices), size=1)[0]
            ]
        self.shuffle_indices = np.zeros(self.n_entries, dtype=int)
        iterable = self.rng.permutation(np.arange(self.n_super_batches))
        start = 0
        for index, super_batch_index in enumerate(iterable):
            super_batch_start = super_batch_index*self.super_batch_size*self.batch_size
            if super_batch_index == self.n_super_batches - 1:
                super_batch_stop = self.n_entries
            else:
                super_batch_stop = (super_batch_index + 1)*self.super_batch_size*self.batch_size
            super_batch_indices = sort_indices[super_batch_start: super_batch_stop]
            stop = start + (super_batch_stop - super_batch_start)
            self.shuffle_indices[start: stop] = self.rng.permutation(super_batch_indices)
            start += stop - start

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        indices = self.shuffle_indices[start: stop]
        return self.q2_obs_scaled[indices], self.xnn_scaled[indices]


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
            'layers': [200, 100, 50],
            'dropout_rate_extraction': 0.1,
            'dropout_rate': 0.1,
            'orthogonal_regularization': 0.01,
            'learning_rate': 0.00005,
            'epochs': 50,
            'batch_size': 64,
            'loss_type': 'mse',
            'init_method': 'random',
            'superbatch_shuffle': False,
            'augment': False,
            'model_type': 'metric',
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
            'layers',
            'dropout_rate_extraction',
            'dropout_rate',
            'orthogonal_regularization'
            'learning_rate',
            'epochs',
            'batch_size',
            'loss_type',
            'init_method',
            'superbatch_shuffle',
            'augment',
            'model_type',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['peak_length'] = int(params['peak_length'])
        self.model_params['extraction_peak_length'] = int(params['extraction_peak_length'])
        self.model_params['filter_length'] = int(params['filter_length'])
        self.model_params['n_volumes'] = int(params['n_volumes'])
        self.model_params['n_filters'] = int(params['n_filters'])
        self.model_params['layers'] = np.array(
            params['layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['dropout_rate_extraction'] = float(params['dropout_rate_extraction'])
        self.model_params['dropout_rate'] = float(params['dropout_rate'])
        self.model_params['orthogonal_regularization'] = float(params['orthogonal_regularization'])
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['epochs'] = int(params['epochs'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['loss_type'] = params['loss_type']
        self.model_params['init_method'] = params['init_method']
        if self.model_params['superbatch_shuffle'] == 'True':
            self.model_params['superbatch_shuffle'] = True
        else:
            self.model_params['superbatch_shuffle'] = False
        if self.model_params['augment'] == 'True':
            self.model_params['augment'] = True
        else:
            self.model_params['augment'] = False
        self.model_params['model_type'] = params['model_type']

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
        inputs = {
            'q2_obs_scaled': tf.keras.Input(
                shape=self.model_params['peak_length'],
                name='q2_obs_scaled',
                dtype=tf.float32,
                )
            }

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
        if self.model_params['model_type'] == 'metric':
            self.model = tf.keras.Model(inputs, self.model_builder_metric(inputs))
        elif self.model_params['model_type'] == 'base_line':
            self.model = tf.keras.Model(inputs, self.model_builder_base_line(inputs))
        elif self.model_params['model_type'] == 'combined':
            self.model = tf.keras.Model(inputs, self.model_builder_combined(inputs))
        self.compile_model()
        #self.model.summary()

    def model_builder_metric(self, inputs):
        # inputs['q2_obs_scaled']: batch_size, n_peaks
        # metric:                  batch_size, n_volumes, n_filters
        metric = self.extraction_layer(
            inputs['q2_obs_scaled'][:, :self.model_params['extraction_peak_length']]
            )
        x = tf.keras.layers.SpatialDropout1D(
            rate=self.model_params['dropout_rate_extraction'],
            name=f'dropout_extraction',
            )(metric)

        regularizer = tf.keras.regularizers.OrthogonalRegularizer(
            factor=self.model_params['orthogonal_regularization'],
            mode='rows'
            )
        for index in range(len(self.model_params['layers'])):
            x = tf.keras.layers.Dense(
                self.model_params['layers'][index],
                activation='linear',
                name=f'dense_{index}',
                use_bias=False,
                kernel_regularizer=regularizer,
                )(x)
            x = tf.keras.layers.LayerNormalization(
                name=f'layer_norm_{index}',
                axis=2,
                center=True,
                )(x)
            x = tf.keras.activations.gelu(x)
            x = tf.keras.layers.Dropout(
                rate=self.model_params['dropout_rate'],
                name=f'dropout_{index}',
                )(x)

        # output: batch_size, n_volumes, unit_cell_length + 1
        output = tf.keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name='xnn_scaled',
            kernel_regularizer=regularizer,
            )(x)
        return output

    def model_builder_base_line(self, inputs):
        # inputs['q2_obs_scaled']: batch_size, n_peaks
        # This is a 'Base line model' that does not use feature extraction
        x = tf.keras.layers.RepeatVector(self.model_params['n_volumes'])(inputs['q2_obs_scaled'])
        for index in range(len(self.model_params['layers'])):
            x = tf.keras.layers.Dense(
                self.model_params['layers'][index],
                activation='linear',
                name=f'dense_{index}',
                use_bias=True,
                kernel_regularizer=regularizer,
                )(x)
            x = tf.keras.layers.LayerNormalization(
                name=f'layer_norm_{index}',
                axis=2,
                center=False,
                )(x)
            x = tf.keras.activations.gelu(x)
            x = tf.keras.layers.Dropout(
                rate=self.model_params['dropout_rate'],
                name=f'dropout_{index}',
                )(x)

        # output: batch_size, n_volumes, unit_cell_length + 1
        output = tf.keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name='xnn_scaled',
            kernel_regularizer=regularizer,
            )(x)
        return output

    def model_builder_combined(self, inputs):
        # inputs['q2_obs_scaled']: batch_size, n_peaks
        # metric:                  batch_size, n_volumes, n_filters
        metric = self.extraction_layer(
            inputs['q2_obs_scaled'][:, :self.model_params['extraction_peak_length']]
            )
        x_metric = tf.keras.layers.Dense(
            self.model_params['layers'][0],
            activation='linear',
            name=f'dense_metric',
            use_bias=False,
            )(metric)
        x_metric = tf.keras.layers.LayerNormalization(
            name=f'layer_norm_metric',
            axis=2,
            center=False,
            )(x_metric)
        x_metric = tf.keras.activations.gelu(x_metric)

        x_q2_obs = tf.keras.layers.Dense(
            self.model_params['layers'][0],
            activation='linear',
            name=f'dense_q2_obs',
            use_bias=False,
            )(inputs['q2_obs_scaled'])
        x_q2_obs = tf.keras.layers.LayerNormalization(
            name=f'layer_norm_q2_obs',
            center=True,
            )(x_q2_obs)
        x_q2_obs = tf.keras.activations.gelu(x_q2_obs)

        x = tf.keras.layers.Add(
            name='combine'
            )([x_metric, x_q2_obs[:, tf.newaxis, :]])
        x = tf.keras.layers.SpatialDropout1D(
            rate=self.model_params['dropout_rate_extraction'],
            name=f'dropout_combined',
            )(x)

        regularizer = tf.keras.regularizers.OrthogonalRegularizer(
            factor=self.model_params['orthogonal_regularization'],
            mode='rows'
            )
        for index in range(1, len(self.model_params['layers'])):
            x = tf.keras.layers.Dense(
                self.model_params['layers'][index],
                activation='linear',
                name=f'dense_{index}',
                use_bias=False,
                kernel_regularizer=regularizer,
                )(x)
            x = tf.keras.layers.LayerNormalization(
                name=f'layer_norm_{index}',
                axis=2,
                center=True,
                )(x)
            x = tf.keras.activations.gelu(x)
            x = tf.keras.layers.Dropout(
                rate=self.model_params['dropout_rate'],
                name=f'dropout_{index}',
                )(x)

        # output: batch_size, n_volumes, unit_cell_length + 1
        output = tf.keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name='xnn_scaled',
            kernel_regularizer=regularizer,
            )(x)
        return output

    def compile_model(self):
        #optimizer = tf.optimizers.legacy.RMSprop(self.model_params['learning_rate']) # loss: 1.4567 - val_loss: 1.1458
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate'])    # loss: 1.4367 - val_loss: 1.2484
        #optimizer = tf.optimizers.legacy.Adagrad(0.001) # loss: 2.0963 - val_loss: 1.5882

        if self.model_params['loss_type'] == 'mse':
            loss_functions = {
                'xnn_scaled': self.extraction_layer.loss_function_mse
                }
            loss_metrics = {
                'xnn_scaled': self.extraction_layer.loss_function_log_cosh
                }
        else:
            loss_functions = {
                'xnn_scaled': self.extraction_layer.loss_function_log_cosh
                }
            loss_metrics = {
                'xnn_scaled': self.extraction_layer.loss_function_mse
                }
        loss_weights = {
            'xnn_scaled': 1
            }
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
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

        train_true = {'xnn_scaled': train_xnn_scaled}
        val_true = {'xnn_scaled': val_xnn_scaled}
        train_inputs = {'q2_obs_scaled': train_q2_obs_scaled}
        val_inputs = {'q2_obs_scaled': val_q2_obs_scaled}

        self.extraction_layer.evaluate_init(
            train_q2_obs_scaled, self.save_to_split_group, self.split_group, self.model_params["tag"]
            )
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
        #xnn_error_titles = xnn_titles[uc_index]
        fig.tight_layout()
        fig.savefig(f'{self.save_to_split_group}/{self.split_group}_pitf_io_{self.model_params["tag"]}.png')
        plt.close()

        print(f'\nStarting training integral filter model for {self.split_group}')
        if self.model_params['superbatch_shuffle']:
            x = DataGenerator(
                self.rng,
                train_q2_obs_scaled,
                train_xnn_scaled,
                train_xnn,
                self.lattice_system,
                self.model_params['batch_size'],
                super_batch_size=4,
                )
            y = None
            shuffle = False
        else:
            x = train_inputs
            y = train_true
            shuffle = True
        self.fit_history = self.model.fit(
            x=x,
            y=y,
            epochs=self.model_params['epochs'],
            shuffle=shuffle,
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
        # number of times a branch is the most probable
        # number of times a branch is in the top 5, 10
        train_rankings = np.argsort(train_softmax, axis=1)[:, ::-1]
        val_rankings = np.argsort(val_softmax, axis=1)[:, ::-1]

        train_top1 = np.bincount(train_rankings[:, 0].ravel(), minlength=self.model_params['n_volumes'])
        val_top1 = np.bincount(val_rankings[:, 0].ravel(), minlength=self.model_params['n_volumes'])
        train_top5 = np.bincount(train_rankings[:, :5].ravel(), minlength=self.model_params['n_volumes'])
        val_top5 = np.bincount(val_rankings[:, :5].ravel(), minlength=self.model_params['n_volumes'])
        train_top10 = np.bincount(train_rankings[:, :10].ravel(), minlength=self.model_params['n_volumes'])
        val_top10 = np.bincount(val_rankings[:, :10].ravel(), minlength=self.model_params['n_volumes'])

        train_gt_10p = np.bincount(np.sum(train_softmax > 0.10, axis=1), minlength=10)
        val_gt_10p = np.bincount(np.sum(val_softmax > 0.10, axis=1), minlength=10)
        train_gt_5p = np.bincount(np.sum(train_softmax > 0.05, axis=1), minlength=20)
        val_gt_5p = np.bincount(np.sum(val_softmax > 0.05, axis=1), minlength=20)
        train_gt_1p = np.bincount(np.sum(train_softmax > 0.01, axis=1), minlength=100)
        val_gt_1p = np.bincount(np.sum(val_softmax > 0.01, axis=1), minlength=100)

        fig, axes = plt.subplots(2, 3, figsize=(10, 5))
        alpha = 0.75
        width = 1
        x = np.arange(self.model_params['n_volumes'])
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
            axes[0].plot(xnn_true[0], xnn_true[1], marker='x', color=[0, 0, 0])
            axes[1].plot(xnn_true[0], xnn_true[2], marker='x', color=[0, 0, 0])
            axes[2].plot(xnn_true[1], xnn_true[2], marker='x', color=[0, 0, 0])
            axes[0].set_xlabel('a')
            axes[1].set_xlabel('a')
            axes[2].set_xlabel('b')
            axes[0].set_ylabel('b')
            axes[1].set_ylabel('c')
            axes[2].set_ylabel('c')
            fig.tight_layout()
            fig.savefig(f'{self.save_to_split_group}/{self.split_group}_pitf_example_{index}_{self.model_params["tag"]}.png')
            plt.close()
