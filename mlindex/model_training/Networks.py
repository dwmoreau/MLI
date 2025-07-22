import keras
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def tensor_to_numpy(tensor):
    if os.environ["KERAS_BACKEND"] == 'tensorflow':
        return tensor.numpy()
    elif os.environ["KERAS_BACKEND"] == 'torch':
        return tensor.detach().cpu().numpy()


class SigmaDecayCallback(keras.callbacks.Callback):
    def __init__(self, extraction_layer, initial_multiplier=10, decay_rate=0.9):
        """
        Callback to decay sigma from initial_value to final_value exponentially.
        
        Args:
            custom_layer: The layer containing the sigma parameter
            initial_value: Starting value for sigma (default: 0.1)
            final_value: Target minimum value for sigma (default: 0.02)
            decay_rate: Rate of exponential decay (default: 0.9)
        """
        super().__init__()
        self.extraction_layer = extraction_layer
        self.initial_value = initial_multiplier*self.extraction_layer.sigma_init
        self.final_value = self.extraction_layer.sigma_init
        self.decay_rate = decay_rate
        
    def on_train_begin(self, logs=None):
        """Set sigma to initial value when training starts"""
        self.extraction_layer.sigma.assign(self.initial_value)
        print(f"Training started: sigma initialized to {self.initial_value:0.5f}")
        
    def on_epoch_begin(self, epoch, logs=None):
        """Update sigma using exponential decay formula"""
        # Calculate new sigma value using exponential decay
        new_sigma = self.final_value + (self.initial_value - self.final_value) * (self.decay_rate ** epoch)
        self.extraction_layer.sigma.assign(new_sigma)
        print(f"Epoch {epoch + 1}: sigma decayed to {new_sigma:0.5f}")


class ExtractionLayer(keras.layers.Layer):
    def __init__(self, model_params, q2_obs, xnn, reciprocal_volume, q2_obs_scale, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        self.seed = 0
        self.volumes = self.add_weight(
            shape=(self.model_params['n_volumes'], 1),
            initializer=keras.initializers.Zeros(),
            dtype='float32',
            trainable=False,
            constraint=keras.constraints.NonNeg(),
            name='volumes'
            )
        self.filters = self.add_weight(
            shape=(1, self.model_params['n_filters']),
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
            trainable=False,
            constraint=keras.constraints.NonNeg(),
            name='sigma'
            )

        if not q2_obs is None:
            import scipy.stats
            import scipy.ndimage
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
                keras.ops.expand_dims(keras.ops.cast(distribution_volumes, dtype='float32'), axis=1),
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
            sigma = min(max(2*np.median(volume_differences), 0.015), 0.04)
            self.sigma_init = sigma
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

            q2_filters = np.zeros(self.model_params['n_filters'])
            for filter_index in range(self.model_params['n_filters']):
                peak_index = rng.choice(
                    self.model_params['extraction_peak_length'],
                    replace=False
                    )
                q2_filters[filter_index] = q2_obs_scaled_rv[peak_index].rvs()
            
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

    def call(self, q2_obs_scaled, amplitude_logits, **kwargs):
        # filters:     1, n_filters
        # volumes:     n_volumes, 1
        # q2_filters:  1, n_volumes, n_filters, 1
        # q2_obs:      batch_size, extraction_peak_length
        # difference:  batch_size, n_volumes, n_filters, extraction_peak_length
        q2_filters = keras.ops.expand_dims(
            keras.ops.expand_dims(self.volumes * self.filters, axis=0),
            axis=3
            )
        q2_obs_scaled = keras.ops.expand_dims(keras.ops.expand_dims(q2_obs_scaled, axis=1), axis=2)
        difference = q2_filters - q2_obs_scaled

        # amplitudes: batch_size, 1, n_filters, extraction_peak_length
        amplitudes = keras.activations.softmax(
            amplitude_logits,
            axis=3
            ) # peak axis
        # Adding 0.001 to self.sigma prevents NaNs
        distances = amplitudes * keras.ops.exp(-1/2 * (difference / (self.sigma + 0.001))**2)
        # distances: batch_size, n_volumes, n_filters, extraction_peak_length
        # metric:    batch_size, n_volumes, n_filters
        metric = keras.ops.sum(distances, axis=3)
        return metric

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
                self.model_params['extraction_peak_length']
                )
            )
        for batch_index in range(n_batchs):
            start = batch_index * batch_size
            stop = (batch_index + 1) * batch_size
            metric = self.call(
                keras.ops.cast(
                    q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']],
                    dtype='float32'
                    ),
                amplitude_logits,
                )
            metric_max[start: stop] = tensor_to_numpy(metric).max(axis=(1, 2))
        start = (batch_index + 1) * batch_size
        metric = self.call(
            keras.ops.cast(
                q2_obs_scaled[start:, :self.model_params['extraction_peak_length']],
                dtype='float32'
                ),
            amplitude_logits
            )
        metric_max[start:] = tensor_to_numpy(metric).max(axis=(1, 2))

        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.hist(metric_max, bins=100)
        axes.set_xlabel('Maximum metric per entry')
        axes.set_ylabel('Counts')
        fig.tight_layout()
        fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_metric_max_{tag}.png'))
        plt.close()

        if not self.filters_init is None:
            print('Making weight plot')
            filters_opt = self.filters.numpy()[0]
            sigma_opt = self.sigma.numpy()
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
            axes[1].set_title(f'sigma init/opt: {self.sigma_init:0.4f} {sigma_opt:0.4f}')
            axes[0].legend()
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_weights_{tag}.png'))
            plt.close()
        else:
            print(self.filters_init)

    def evaluate_init(self, q2_obs_scaled, save_to, split_group, tag):
        q2_filters = tensor_to_numpy(self.volumes * self.filters)
        volumes = self.volumes.numpy()[:, 0]
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
                self.model_params['extraction_peak_length']
                )
            )
        for batch_index in range(n_batchs + 1):
            start = batch_index * batch_size
            if batch_index == n_batchs:
                stop = -1
            else:
                stop = (batch_index + 1) * batch_size
            metric_tensor = self.call(
                keras.ops.cast(
                    q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']],
                    dtype='float32'
                    ),
                amplitude_logits
                )

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


class IntraVolume_MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Single set of weights applied to all volumes
        self.W_q = keras.layers.Dense(
            d_model,
            use_bias=False,
            kernel_initializer=keras.initializers.HeUniform,
            #activation=keras.activations.gelu
        )
        self.W_k = keras.layers.Dense(
            d_model,
            use_bias=False,
            kernel_initializer=keras.initializers.HeUniform,
            #activation=keras.activations.gelu
        )
        self.W_v = keras.layers.Dense(
            d_model,
            use_bias=False,
            kernel_initializer=keras.initializers.HeUniform,
            #activation=keras.activations.gelu
        )
        self.W_o = keras.layers.Dense(
            d_model,
            use_bias=False,
            kernel_initializer=keras.initializers.HeUniform,
            #activation=keras.activations.gelu
        )
        
    def call(self, x):
        # x shape: (batch_size, n_volumes, n_filters)
        batch_size = keras.ops.shape(x)[0]
        n_volumes = keras.ops.shape(x)[1] 
        n_filters = keras.ops.shape(x)[2]
        
        # Reshape to treat each volume as a separate sequence
        # (batch_size * n_volumes, n_filters, 1)
        x_flat = keras.ops.reshape(x, (batch_size * n_volumes, n_filters, 1))
        
        # Generate Q, K, V - same operation applied to each volume
        Q = self.W_q(x_flat)
        K = self.W_k(x_flat)
        V = self.W_v(x_flat)
        
        # Reshape for multi-head attention
        Q = keras.ops.reshape(Q, (batch_size * n_volumes, n_filters, self.n_heads, self.d_k))
        K = keras.ops.reshape(K, (batch_size * n_volumes, n_filters, self.n_heads, self.d_k))
        V = keras.ops.reshape(V, (batch_size * n_volumes, n_filters, self.n_heads, self.d_k))
        
        # Transpose to (batch*n_volumes, n_heads, n_filters, d_k)
        Q = keras.ops.transpose(Q, [0, 2, 1, 3])
        K = keras.ops.transpose(K, [0, 2, 1, 3])
        V = keras.ops.transpose(V, [0, 2, 1, 3])
        
        # Compute attention scores within each volume
        # (batch*n_volumes, n_heads, n_filters, n_filters)
        scores = keras.ops.matmul(Q, keras.ops.transpose(K, [0, 1, 3, 2])) / math.sqrt(self.d_k)
        attention_weights = keras.ops.softmax(scores, axis=-1)
        
        # Apply attention to values
        # (batch*n_volumes, n_heads, n_filters, d_k)
        attended = keras.ops.matmul(attention_weights, V)
        
        # Concatenate heads and reshape back
        # (batch*n_volumes, n_filters, d_model)
        attended = keras.ops.transpose(attended, [0, 2, 1, 3])
        attended = keras.ops.reshape(attended, (batch_size * n_volumes, n_filters, self.d_model))
        
        # Final projection
        output = self.W_o(attended)
        
        # Reshape back to original structure
        # (batch_size, n_volumes, d_model)
        output = keras.ops.reshape(output, (batch_size, n_volumes, self.d_model))
        
        return output


class IntraVolume_Attention(keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
        self.W_q = keras.layers.Dense(
            d_model, 
            use_bias=False,
            #activation=keras.activations.gelu,
            kernel_initializer=keras.initializers.HeUniform
        )
        self.W_k = keras.layers.Dense(
            d_model, 
            use_bias=False,
            #activation=keras.activations.gelu,
            kernel_initializer=keras.initializers.HeUniform
        )
        self.W_v = keras.layers.Dense(
            d_model, 
            use_bias=False,
            #activation=keras.activations.gelu,
            kernel_initializer=keras.initializers.HeUniform
        )
        
    def call(self, x):
        # x shape: (batch_size, n_volumes, n_filters)
        batch_size = keras.ops.shape(x)[0]
        n_volumes = keras.ops.shape(x)[1]
        n_filters = keras.ops.shape(x)[2]
        
        # Add feature dimension: (batch_size, n_volumes, n_filters, 1)
        x_expanded = keras.ops.expand_dims(x, -1)
        
        # Apply same linear transformations to each volume
        Q = self.W_q(x_expanded)  # (batch_size, n_volumes, n_filters, d_model)
        K = self.W_k(x_expanded)
        V = self.W_v(x_expanded)
        
        # Compute attention scores for each volume independently
        # (batch_size, n_volumes, n_filters, n_filters)
        scores = keras.ops.matmul(Q, keras.ops.transpose(K, [0, 1, 3, 2])) / math.sqrt(self.d_model)
        attention_weights = keras.ops.softmax(scores, axis=-1)
        
        # Apply attention: (batch_size, n_volumes, n_filters, d_model)
        attended = keras.ops.matmul(attention_weights, V)
        
        # Reduce to final output: (batch_size, n_volumes, d_model)
        output = keras.ops.mean(attended, axis=2)  # or keras.ops.sum, depending on what you want
        
        return output