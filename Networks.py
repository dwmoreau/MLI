import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
import scipy.ndimage


def tensor_to_numpy(tensor):
    if os.environ["KERAS_BACKEND"] == 'tensorflow':
        return tensor.numpy()
    elif os.environ["KERAS_BACKEND"] == 'torch':
        return tensor.detach().cpu().numpy()


class ExtractionLayer(keras.layers.Layer):
    def __init__(self, model_params, q2_obs, xnn, reciprocal_volume, q2_obs_scale, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        self.seed = 0
        self.volumes = self.add_weight(
            shape=(self.model_params['n_volumes'], 1, 1),
            initializer=keras.initializers.Zeros(),
            dtype='float32',
            trainable=False,
            constraint=keras.constraints.NonNeg(),
            name='volumes'
            )
        self.filters = self.add_weight(
            shape=(1, self.model_params['n_filters'], self.model_params['filter_length']),
            initializer=keras.initializers.Zeros(),
            dtype='float32',
            trainable=True,
            constraint=keras.constraints.NonNeg(),
            name='filters'
            )

        self.sigma = self.add_weight(
            shape=(),
            initializer=keras.initializers.Zeros(),
            dtype='float32',
            trainable=True,
            constraint=keras.constraints.NonNeg(),
            name='sigma'
            )

        if self.model_params['model_type'] == 'deep':
            # If the .call method is not used, then the model
            # weights will not load properly.
            self.call = self.deep_model_call
        elif self.model_params['model_type'] == 'metric':
            self.call = self.basic_model_call

        if not q2_obs is None:
            rng = np.random.default_rng(self.seed)
            reciprocal_volume_sorted = np.sort(reciprocal_volume)
            upper_volume_limit = reciprocal_volume_sorted[int(0.990*reciprocal_volume_sorted.size)]
            lower_volume_limit = reciprocal_volume_sorted[int(0.005*reciprocal_volume_sorted.size)]
            bins_vol = np.linspace(lower_volume_limit, upper_volume_limit, 401)
            centers_vol = (bins_vol[1:] + bins_vol[:-1]) / 2
            reciprocal_volume_hist, _ = np.histogram(reciprocal_volume, bins=bins_vol, density=True)
            reciprocal_volume_hist_smoothed = scipy.ndimage.gaussian_filter1d(
                reciprocal_volume_hist, sigma=3, mode='constant'
                )
            reciprocal_volume_rv = scipy.stats.rv_histogram(
                (reciprocal_volume_hist_smoothed, bins_vol), density=True
                )
            reciprocal_volume_samples = reciprocal_volume_rv.ppf(np.linspace(
                0.001, 0.999, self.model_params['n_volumes']
                ))
            distribution_volumes = (reciprocal_volume_samples / q2_obs_scale**2)**(2/3)

            # Ideally this scaling of distribution_volumes should not be needed.
            # For primitive monoclinic and triclinic, the distribution of the volumes skews to the
            # large side and the q2_filter distribution is pushed to a much larger region than q2_obs.
            distribution_volumes /= np.median(distribution_volumes)
            self.volumes.assign(
                keras.ops.expand_dims(
                    keras.ops.expand_dims(
                        keras.ops.cast(distribution_volumes, dtype='float32'), axis=1
                        ),
                    axis=2
                    )
                )
            #fig, axes = plt.subplots(1, 1, figsize=(6, 3))
            #vol_scaled_x = (centers_vol / q2_obs_scale**2)**(2/3)
            #vol_scaled_bins = (bins_vol / q2_obs_scale**2)**(2/3)
            #axes.bar(
            #    vol_scaled_x, reciprocal_volume_hist,
            #    width=vol_scaled_bins[1:]- vol_scaled_bins[:-1]
            #    )
            #axes.plot(
            #    vol_scaled_x, reciprocal_volume_hist_smoothed,
            #    color=[1, 0, 0]
            #    )
            #axes.plot(
            #    vol_scaled_x, reciprocal_volume_rv.pdf(vol_scaled_x),
            #    color=[0, 1, 0]
            #    )
            #axes.set_xlabel(f'Reciprocal Volume - Scaled')
            #axes.set_ylabel('Distribution')
            #fig.tight_layout()
            #plt.show()

            volume_differences = distribution_volumes[1:] - distribution_volumes[:-1]
            sigma = min(max(np.median(volume_differences), 0.015), 0.035)
            self.sigma.assign(keras.ops.cast(sigma, dtype='float32'))

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
                q2_obs_scaled_hist_smoothed = scipy.signal.medfilt(q2_obs_scaled_hist, kernel_size=3)
                q2_obs_scaled_rv[index] = scipy.stats.rv_histogram(
                    (q2_obs_scaled_hist_smoothed, bins_q2_obs_scaled), density=True
                    )

                #fig, axes = plt.subplots(1, 1, figsize=(6, 3))
                #axes.bar(
                #    centers_q2_obs_scaled, q2_obs_scaled_hist,
                #    width=centers_q2_obs_scaled[1]- centers_q2_obs_scaled[0]
                #    )
                #axes.plot(
                #    centers_q2_obs_scaled, q2_obs_scaled_hist_smoothed,
                #    color=[1, 0, 0]
                #    )
                #axes.plot(
                #    centers_q2_obs_scaled, q2_obs_scaled_rv[index].pdf(centers_q2_obs_scaled),
                #    color=[0, 1, 0]
                #    )
                #axes.set_xlabel(f'q2 - Peak {index}')
                #axes.set_ylabel('Distribution')
                #fig.tight_layout()
                #plt.show()

            q2_filters = np.zeros((self.model_params['n_filters'], self.model_params['filter_length']))
            peak_position_sum = np.zeros(self.model_params['n_filters'])
            for filter_index in range(self.model_params['n_filters']):
                peak_indices = np.sort(rng.choice(
                    self.model_params['extraction_peak_length'],
                    self.model_params['filter_length'],
                    replace=False
                    ))
                peak_position_sum[filter_index] = np.sum(peak_indices)
                for filter_peak_index, peak_index in enumerate(peak_indices):
                    q2_filters[filter_index, filter_peak_index] = q2_obs_scaled_rv[peak_index].rvs()
            if self.model_params['model_type'] == 'deep':
                q2_filters = q2_filters[np.argsort(peak_position_sum)]
            self.filters.assign(
                keras.ops.expand_dims(keras.ops.cast(q2_filters, dtype='float32'), axis=0)
                )
            self.filters_init = self.filters.numpy()[0]

            #fig, axes = plt.subplots(1, 1, figsize=(6, 3))
            #q2_filt_scaled_hist, _ = np.histogram(
            #    q2_filters.ravel(),
            #    bins=bins_q2_obs_scaled, density=True
            #    )
            #q2_obs_scaled_hist, _ = np.histogram(
            #    q2_obs_scaled[:, :self.model_params['extraction_peak_length']].ravel(),
            #    bins=bins_q2_obs_scaled, density=True
            #    )
            #axes.bar(
            #    centers_q2_obs_scaled, q2_obs_scaled_hist,
            #    width=centers_q2_obs_scaled[1]- centers_q2_obs_scaled[0],
            #    label='Observations'
            #    )
            #axes.bar(
            #    centers_q2_obs_scaled, q2_filt_scaled_hist,
            #    width=centers_q2_obs_scaled[1]- centers_q2_obs_scaled[0],
            #    label='Filter',
            #    alpha=0.5
            #    )
            #axes.legend(frameon=False)
            #axes.set_xlabel(f'q2')
            #axes.set_ylabel('Distribution')
            #fig.tight_layout()
            #plt.show()
        else:
            self.filters_init = None

    def basic_model_call(self, q2_obs_scaled, amplitude_logits, **kwargs):
        # filters:     1, n_filters, filter_length
        # volumes:     n_volumes, 1, 1
        # q2_filters:  n_volumes, n_filters, filter_length
        # q2_obs:      batch_size, extraction_peak_length
        # difference: batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        q2_filters = keras.ops.expand_dims(
            keras.ops.expand_dims(self.volumes * self.filters, axis=0),
            axis=4
            )
        q2_obs_scaled = keras.ops.expand_dims(
            keras.ops.expand_dims(keras.ops.expand_dims(q2_obs_scaled, axis=1), axis=2), axis=3
            )
        difference = q2_filters - q2_obs_scaled

        # amplitudes: batch_size, 1, n_filters, filter_length, extraction_peak_length
        amplitudes = self.model_params['filter_length'] * keras.activations.softmax(
            amplitude_logits,
            axis=4
            ) # peak axis
        # Adding 0.001 to self.sigma prevents NaNs
        distances = amplitudes * keras.ops.exp(-1/2 * (difference / (self.sigma + 0.001))**2)
        # distances: batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        # metric:    batch_size, n_volumes, n_filters
        metric = keras.ops.sum(distances, axis=(3, 4))
        return metric

    def deep_model_call(self, q2_obs_scaled, **kwargs):
        # filters:     1, n_filters, filter_length
        # volumes:     n_volumes, 1, 1
        # q2_filters:  n_volumes, n_filters, filter_length
        # q2_obs:      batch_size, extraction_peak_length
        # difference:  batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        q2_filters = keras.ops.expand_dims(
            keras.ops.expand_dims(self.volumes * self.filters, axis=0),
            axis=4
            )
        q2_obs_scaled = keras.ops.expand_dims(
            keras.ops.expand_dims(keras.ops.expand_dims(q2_obs_scaled, axis=1), axis=2), axis=3
            )
        difference = q2_filters - q2_obs_scaled
        return keras.ops.exp(-1/2 * (difference / (self.sigma + 0.001))**2)

    def loss_function_common(self, y_true, y_pred):
        # y_true: batch_size, unit_cell_length
        # y_pred: batch_size, n_volumes, unit_cell_length + 1
        xnn_scaled_pred = y_pred[:, :, :self.model_params['unit_cell_length']]
        logits = y_pred[:, :, self.model_params['unit_cell_length']]
        probabilities = keras.ops.softmax(logits, axis=1)
        errors = keras.ops.expand_dims(y_true, axis=1) - xnn_scaled_pred
        # This is to prevent an overflow error
        # keras.ops.cosh has a limit around +/- 80 for dtype=float32
        errors = keras.ops.clip(errors, -75.0, 75.0)
        return errors, probabilities

    def loss_function_log_cosh(self, y_true, y_pred):
        errors, probabilities = self.loss_function_common(y_true, y_pred)
        losses = keras.ops.sum(keras.ops.log(keras.ops.cosh(errors)), axis=2)
        return keras.ops.sum(losses * probabilities, axis=1)

    def loss_function_mse(self, y_true, y_pred):
        errors, probabilities = self.loss_function_common(y_true, y_pred)
        losses = 1/2 * keras.ops.mean(errors**2, axis=2)
        return keras.ops.sum(losses * probabilities, axis=1)

    def evaluate_weights(self, q2_obs_scaled, save_to, split_group, tag):
        metric_max = np.zeros(q2_obs_scaled.shape[0])
        batch_size = 64
        n_batchs = q2_obs_scaled.shape[0] // batch_size
        
        amplitude_logits = keras.ops.ones(
            shape=(
                1, 1,
                self.model_params['n_filters'],
                self.model_params['filter_length'],
                self.model_params['extraction_peak_length']
                )
            )
        for batch_index in range(n_batchs):
            start = batch_index * batch_size
            stop = (batch_index + 1) * batch_size
            if self.model_params['model_type'] == 'metric':
                metric = self.call(
                    keras.ops.cast(
                        q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']],
                        dtype='float32'
                        ),
                    amplitude_logits,
                    )
            elif self.model_params['model_type'] == 'deep':
                metric = self.call(keras.ops.cast(
                    q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']],
                    dtype='float32'
                    ))
            metric_max[start: stop] = tensor_to_numpy(metric).max(axis=(1, 2))
        start = (batch_index + 1) * batch_size
        if self.model_params['model_type'] == 'metric':
            metric = self.call(
                keras.ops.cast(
                    q2_obs_scaled[start:, :self.model_params['extraction_peak_length']],
                    dtype='float32'
                    ),
                amplitude_logits
                )
        elif self.model_params['model_type'] == 'deep':
            metric = self.call(keras.ops.cast(
                q2_obs_scaled[start:, :self.model_params['extraction_peak_length']],
                dtype='float32'
                ))
        metric_max[start:] = tensor_to_numpy(metric).max(axis=(1, 2))

        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.hist(metric_max, bins=100)
        axes.set_xlabel('Maximum metric per entry')
        axes.set_ylabel('Counts')
        fig.tight_layout()
        fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_metric_max_{tag}.png'))
        plt.close()

        if not self.filters_init is None:
            filters_opt = self.filters.numpy()[0]
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            alpha = 0.75
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].hist(
                self.filters_init.ravel(), bins=10, color=colors[0], label='Init'
                )
            axes[0].hist(
                filters_opt.ravel(), bins=10, color=colors[1], alpha=alpha, label='Optimized'
                )
            axes[1].hist(
                self.filters_init.ravel() - filters_opt.ravel(), bins=10, color=colors[2], label='Init - Optimized'
                )
            axes[0].set_title('Filter Weights')
            axes[0].set_xlabel('Value')
            axes[1].set_xlabel('Difference')
            axes[0].legend()
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_weights_{tag}.png'))
            plt.close()

    def evaluate_init(self, q2_obs_scaled, save_to, split_group, tag):
        q2_filters = tensor_to_numpy(self.volumes * self.filters)
        volumes = self.volumes.numpy()[:, 0, 0]
        volume_differences = volumes[1:] - volumes[:-1]
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        axes.plot(volume_differences, marker='.')        
        xlim = axes.get_xlim()
        axes.plot(xlim, [self.sigma.numpy(), self.sigma.numpy()], linestyle='dashed')
        axes.plot(xlim, [0.01, 0.01], linestyle='dotted', color=[0, 0, 0])
        axes.plot(xlim, [0.04, 0.04], linestyle='dotted', color=[0, 0, 0])
        axes.set_xlim(xlim)
        axes.set_ylim([0, axes.get_ylim()[1]])
        axes.set_ylabel('Volume\nDifference')
        axes.set_xlabel('Volume Index')
        fig.tight_layout()
        fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_volume_diff_{tag}.png'))
        plt.close()

        bins = np.linspace(0, 5, 101)
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.hist(
            q2_obs_scaled[:, :self.model_params['extraction_peak_length']].ravel(),
            bins=bins, label='q2_obs_scaled', density=True
            )
        axes.hist(
            q2_filters.ravel(),
            bins=bins, alpha=0.75, label='q2_filters', density=True
            )
        axes.set_xlabel('q2 scaled')
        axes.set_ylabel('distribution')
        axes.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_filter_init_{tag}.png'))
        plt.close()

        bins = np.linspace(0, 2, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        metric_zeros = 0
        metric_counts = 0
        metric_hist = np.zeros(100)
        batch_size = 64
        filter_metric = np.zeros((q2_obs_scaled.shape[0], self.model_params['n_filters']))
        entry_max_metric = np.zeros(q2_obs_scaled.shape[0])
        n_batchs = q2_obs_scaled.shape[0] // batch_size
        amplitude_logits = keras.ops.ones(
            shape=(
                1, 1,
                self.model_params['n_filters'],
                self.model_params['filter_length'],
                self.model_params['extraction_peak_length']
                )
            )
        for batch_index in range(n_batchs + 1):
            start = batch_index * batch_size
            if batch_index == n_batchs:
                stop = -1
            else:
                stop = (batch_index + 1) * batch_size
            if self.model_params['model_type'] == 'metric':
                metric_tensor = self.call(
                    keras.ops.cast(
                        q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']],
                        dtype='float32'
                        ),
                    amplitude_logits
                    )

            elif self.model_params['model_type'] == 'deep':
                metric_tensor = self.call(keras.ops.cast(
                    q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']],
                    dtype='float32'
                    ))
            metric = tensor_to_numpy(metric_tensor)
            #filter_metric[start: stop, :] = metric.sum(axis=1) / self.model_params['n_volumes']
            filter_metric[start: stop, :] = metric.max(axis=1)

            entry_max_metric[start: stop] = metric.max(axis=(1, 2))

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
        fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_metric_init_{tag}.png'))
        plt.close()


def mlp_model_builder(x, tag, model_params, output_name):
    for index in range(len(model_params['layers'])):
        x = keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            use_bias=False,
            )(x)
        x = keras.layers.LayerNormalization(
            epsilon=model_params['epsilon'], 
            name=f'layer_norm_{tag}_{index}'
            )(x)
        x = keras.activations.gelu(x)
        x = keras.layers.Dropout(
            rate=model_params['dropout_rate'],
            name=f'dropout_{tag}_{index}',
            )(x)
    output = keras.layers.Dense(
        model_params['unit_cell_length'],
        activation=model_params['output_activation'],
        name=output_name,
        kernel_initializer=model_params['kernel_initializer'],
        bias_initializer=model_params['bias_initializer'],
        )(x)
    return output


def hkl_model_builder_additive(x_in, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10

    x = keras.layers.Dense(
        model_params['hkl_ref_length'],
        activation='linear',
        name=f'dense_{tag}_0',
        kernel_regularizer=keras.regularizers.OrthogonalRegularizer(
            factor=model_params['Ortho_kernel_reg'],
            mode='rows'
            ),
        use_bias=False,
        )(x_in)
    x = keras.layers.LayerNormalization(
        epsilon=model_params['epsilon'], 
        name=f'layer_norm_{tag}_0',
        )(x)
    x = keras.activations.gelu(x)
    x = keras.layers.Dropout(
        rate=model_params['dropout_rate'],
        name=f'dropout_{tag}_0'
        )(x)

    x = keras.layers.Dense(
        model_params['hkl_ref_length'],
        activation='linear',
        name=f'hkl_output_{tag}',
        kernel_regularizer=keras.regularizers.OrthogonalRegularizer(
            factor=model_params['Ortho_kernel_reg'],
            mode='rows'
            ),
        )(x)

    hkl_out = keras.layers.Softmax(
        axis=2,
        name=f'hkl_{tag}'
        )(x + x_in)
    return hkl_out


def hkl_model_builder(x, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10
    for index in range(len(model_params['layers'])):
        x = keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            kernel_regularizer=keras.regularizers.OrthogonalRegularizer(
                factor=model_params['Ortho_kernel_reg'],
                mode='rows'
                ),
            use_bias=False,
            )(x)
        x = keras.layers.LayerNormalization(
            epsilon=model_params['epsilon'], 
            name=f'layer_norm_{tag}_{index}',
            )(x)
        x = keras.activations.gelu(x)
        x = keras.layers.Dropout(
            rate=model_params['dropout_rate'],
            name=f'dropout_{tag}_{index}'
            )(x)

    x = keras.layers.Dense(
        model_params['hkl_ref_length'],
        activation='linear',
        name=f'hkl_output_{tag}',
        kernel_regularizer=keras.regularizers.OrthogonalRegularizer(
            factor=model_params['Ortho_kernel_reg'],
            mode='rows'
            ),
        )(x)
    hkl_out = keras.layers.Softmax(
        axis=2,
        name=f'hkl_{tag}'
        )(x)
    return hkl_out
