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
        self.q2_obs_scale = q2_obs_scale
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
            trainable=False,
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
            # The branch grid only spans these percentiles, so entries outside them have no branch
            # that renders them in register and are measurably the worst predicted group. Widening
            # costs alignment, since the same number of branches covers more range, so the two are
            # traded against each other rather than set independently.
            upper_volume_limit = reciprocal_volume_sorted[int(
                self.model_params['volume_upper_percentile']*reciprocal_volume_sorted.size
                )]
            lower_volume_limit = reciprocal_volume_sorted[int(
                self.model_params['volume_lower_percentile']*reciprocal_volume_sorted.size
                )]
            bins_vol = np.linspace(lower_volume_limit, upper_volume_limit, 401)
            centers_vol = (bins_vol[1:] + bins_vol[:-1]) / 2
            reciprocal_volume_hist, _ = np.histogram(reciprocal_volume, bins=bins_vol, density=True)
            reciprocal_volume_hist_smoothed = scipy.ndimage.gaussian_filter1d(
                reciprocal_volume_hist, sigma=3, mode='constant'
                )
            reciprocal_volume_rv = scipy.stats.rv_histogram(
                (reciprocal_volume_hist_smoothed, bins_vol), density=True
                )
            # Peak p of branch v is rendered at q2_p / v, so how far a peak moves between adjacent
            # branches is set by the step in 1/v, not by the step in probability. Spacing the
            # branches by equal probability gives every branch the same number of entries but lets
            # the step in 1/v vary enormously across the grid, so entries in the sparse tails land
            # far from any branch. blend interpolates between the two: 1.0 is equal probability,
            # 0.0 is equal steps in 1/v and therefore equal misalignment everywhere.
            blend = self.model_params['volume_spacing_blend']
            if blend >= 1.0:
                reciprocal_volume_samples = reciprocal_volume_rv.ppf(np.linspace(
                    0.001, 0.999, self.model_params['n_volumes']
                    ))
                distribution_volumes = (reciprocal_volume_samples / q2_obs_scale**2)**(2/3)
            else:
                quantiles = np.linspace(0.001, 0.999, 20001)
                reciprocal_v = 1.0/(
                    reciprocal_volume_rv.ppf(quantiles) / q2_obs_scale**2
                    )**(2/3)
                weight = np.abs(np.gradient(reciprocal_v, quantiles))**(1 - blend)
                cumulative = np.concatenate(
                    [[0], np.cumsum((weight[1:] + weight[:-1])/2 * np.diff(quantiles))]
                    )
                cumulative /= cumulative[-1]
                distribution_volumes = np.sort(1.0/np.interp(
                    np.linspace(0, 1, self.model_params['n_volumes']), cumulative, reciprocal_v
                    ))

            # Ideally this scaling of distribution_volumes should not be needed.
            # For primitive monoclinic and triclinic, the distribution of the volumes skews to the
            # large side and the q2_filter distribution is pushed to a much larger region than q2_obs.
            # Taken from the data rather than from the branch samples so that it does not move when
            # the branches are reallocated. With equal-probability spacing the two agree to 0.2%,
            # but a blended grid's median branch is not the distribution's median, and letting the
            # normalization follow it would silently rescale the grid and the fitted sigma with it.
            volume_normalization = np.median((reciprocal_volume / q2_obs_scale**2)**(2/3))
            distribution_volumes /= volume_normalization
            # Kept so an entry's own volume can be put on the same scale as the branches, which is
            # what the branch labels for the auxiliary loss need. Training only; on the load path
            # there is no data to fit and no labels to build.
            self.volume_normalization = volume_normalization
            self.volumes.assign(
                keras.ops.expand_dims(keras.ops.cast(distribution_volumes, dtype='float32'), axis=1),
                )

            # Equal probability spacing gave every branch the same number of entries, so the
            # auxiliary branch cross entropy was a balanced classification without anyone arranging
            # it. Any other spacing breaks that -- a blended grid leaves branch frequency varying
            # more than tenfold -- and cross entropy systematically under predicts rare classes,
            # which here are the sparsely populated small volume branches that most need the help.
            # Weighting by inverse frequency restores the balance the old grid provided implicitly.
            branch_counts = np.bincount(
                self.get_branch_labels(reciprocal_volume), minlength=self.model_params['n_volumes']
                ).astype(float)
            occupied = branch_counts > 0
            branch_class_weights = np.zeros_like(branch_counts)
            branch_class_weights[occupied] = 1.0/branch_counts[occupied]
            # Normalised so the mean weight over the training data is 1, which keeps the loss on the
            # same scale as the unweighted version and leaves branch_loss_weight meaning what it did.
            branch_class_weights /= (
                (branch_counts*branch_class_weights).sum()/branch_counts.sum()
                )
            self.branch_class_weights = branch_class_weights

            q2_obs_scaled = q2_obs / q2_obs_scale
            q2_obs_scaled_sorted = np.sort(
                q2_obs_scaled[:, :self.model_params['extraction_peak_length']].ravel()
                )

            upper_q2_obs_scaled_limit = q2_obs_scaled_sorted[int(0.98*q2_obs_scaled_sorted.size)]
            q2_filters = np.linspace(
                upper_q2_obs_scaled_limit/self.model_params['n_filters'],
                upper_q2_obs_scaled_limit,
                self.model_params['n_filters']
            )

            # sigma is the width of a peak on the fixed filter grid, so it is measured there. Each
            # training entry is projected onto the grid by its own volume scale, exactly as its
            # matched branch will project it, and sigma is set so that adjacent peaks of that
            # volume-normalized pattern stay resolved: two Gaussians of width sigma separated by d
            # only show a dip when d > 2*sigma. The 5% quantile rather than the minimum keeps a
            # handful of pathological entries from setting the width for the whole split group.
            entry_scales = (reciprocal_volume / q2_obs_scale**2)**(2/3) / volume_normalization
            projected_q2 = np.sort(
                q2_obs_scaled[:, :self.model_params['extraction_peak_length']]
                / entry_scales[:, np.newaxis],
                axis=1
                )
            peak_separations = np.diff(projected_q2, axis=1).ravel()
            peak_separations = peak_separations[peak_separations > 0]
            sigma_measured = np.quantile(peak_separations, 0.05) / 2
            # Kept so evaluate_init can plot the distribution sigma was actually fitted to.
            self.peak_separations_init = peak_separations

            # Safeguard. Too wide merges the peaks into a featureless blob; too narrow falls between
            # filters and will not train. Both bounds are multiples of the filter spacing so they
            # track n_filters instead of silently inverting if it changes.
            filter_spacing = q2_filters[1] - q2_filters[0]
            sigma = min(max(sigma_measured, 2*filter_spacing), 6*filter_spacing)
            self.sigma_init = sigma
            self.sigma.assign(keras.ops.cast(sigma, dtype='float32'))

            if sigma > sigma_measured:
                clipped = f'CLIPPED UP from {sigma_measured:0.5f} (floor 2*spacing)'
            elif sigma < sigma_measured:
                clipped = f'CLIPPED DOWN from {sigma_measured:0.5f} (ceiling 6*spacing)'
            else:
                clipped = 'not clipped'
            print(
                f'sigma = {sigma:0.5f} = {sigma/filter_spacing:0.2f} x filter spacing '
                f'({filter_spacing:0.5f}); {clipped}'
                )

            self.filters.assign(
                keras.ops.expand_dims(keras.ops.cast(q2_filters, dtype='float32'), axis=0)
                )
            self.filters_init = self.filters.numpy()[0]
        else:
            self.filters_init = None
            # No data to count branch occupancy from, and none needed: the load path never trains.
            self.branch_class_weights = None

    def call(self, q2_obs_scaled, **kwargs):
        # filters:     1, 1, n_filters, 1
        # volumes:     1, n_volumes, 1, 1
        # q2_obs:      batch_size, 1, 1, extraction_peak_length
        # difference:  batch_size, n_volumes, n_filters, extraction_peak_length
        filters = keras.ops.reshape(self.filters, (1, 1, -1, 1))
        volumes = keras.ops.reshape(self.volumes, (1, -1, 1, 1))
        q2_obs_scaled = keras.ops.expand_dims(keras.ops.expand_dims(q2_obs_scaled, axis=1), axis=2)

        # q2_obs is projected onto the fixed filter grid by the trial volume, rather than the filter
        # grid being stretched out to meet q2_obs. Both place peak p of branch v at q2_p / v, but
        # this form gives every branch the same Gaussian width instead of sigma / v, so a given unit
        # cell shape reaches the shared attention embedding as the same pattern at every volume.
        difference = filters - q2_obs_scaled / volumes

        distances = keras.ops.exp(-1/2 * (difference / self.sigma)**2)
        # distances: batch_size, n_volumes, n_filters, extraction_peak_length
        # metric:    batch_size, n_volumes, n_filters
        metric = keras.ops.sum(distances, axis=3)
        return metric

    def get_branch_labels(self, reciprocal_volume):
        """Index of the branch whose trial volume best matches each entry's true volume.

        Built exactly the way the branch grid itself was built, so the label is the branch that
        actually renders the entry in register. Matched in log space because the branches are
        spaced geometrically, so a fixed ratio, not a fixed difference, is what 'close' means.
        """
        scales = (reciprocal_volume/self.q2_obs_scale**2)**(2/3)/self.volume_normalization
        volumes = np.asarray(keras.ops.convert_to_numpy(self.volumes)).ravel()
        differences = np.abs(np.log(scales)[:, np.newaxis] - np.log(volumes)[np.newaxis])
        return differences.argmin(axis=1)

    def loss_function_common(self, y_true, y_pred):
        # y_true: batch_size, unit_cell_length + 1
        #         the xnn targets, then the index of the branch that matches the true volume
        # y_pred: batch_size, n_volumes, unit_cell_length + 1
        unit_cell_length = self.model_params['unit_cell_length']
        xnn_true = y_true[:, :unit_cell_length]
        xnn_scaled_pred = y_pred[:, :, :unit_cell_length]
        logits = y_pred[:, :, unit_cell_length]
        probabilities = keras.ops.softmax(logits, axis=1)
        errors = keras.ops.expand_dims(xnn_true, axis=1) - xnn_scaled_pred
        # This is to prevent an overflow error
        # keras.ops.cosh has a limit around +/- 80 for dtype=float32
        errors = keras.ops.clip(errors, -75.0, 75.0)
        return errors, probabilities, logits, y_true[:, unit_cell_length]

    def loss_function_log_cosh(self, y_true, y_pred):
        errors, probabilities, _, _ = self.loss_function_common(y_true, y_pred)
        losses = keras.ops.sum(keras.ops.log(keras.ops.cosh(errors)), axis=2)
        return keras.ops.sum(losses * probabilities, axis=1)

    def loss_function_mse(self, y_true, y_pred):
        errors, probabilities, _, _ = self.loss_function_common(y_true, y_pred)
        losses = 1/2 * keras.ops.mean(errors**2, axis=2)
        return keras.ops.sum(losses * probabilities, axis=1)

    def loss_function_branch(self, y_true, y_pred):
        """Cross entropy on the branch logits against the branch that matches the true volume.

        The regression losses weight each branch by its own softmax probability, which rewards the
        model for being confident where it happens to predict well. Nothing in them says which
        branch is physically correct, so the ranking is only ever supervised indirectly. This does
        say it.
        """
        _, _, logits, branch_true = self.loss_function_common(y_true, y_pred)
        labels = keras.ops.cast(branch_true, dtype='int32')
        losses = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        if self.branch_class_weights is not None:
            losses = losses*keras.ops.take(
                keras.ops.cast(self.branch_class_weights, dtype='float32'), labels, axis=0
                )
        return losses

    def branch_accuracy(self, y_true, y_pred):
        """Fraction of entries whose top ranked branch is the matching one."""
        _, _, logits, branch_true = self.loss_function_common(y_true, y_pred)
        predicted = keras.ops.cast(keras.ops.argmax(logits, axis=1), dtype='int32')
        return keras.ops.cast(
            keras.ops.equal(predicted, keras.ops.cast(branch_true, dtype='int32')),
            dtype='float32',
            )

    def training_loss_log_cosh(self, y_true, y_pred):
        return (
            self.loss_function_log_cosh(y_true, y_pred)
            + self.model_params.get('branch_loss_weight', 0.0)
            * self.loss_function_branch(y_true, y_pred)
            )

    def training_loss_mse(self, y_true, y_pred):
        return (
            self.loss_function_mse(y_true, y_pred)
            + self.model_params.get('branch_loss_weight', 0.0)
            * self.loss_function_branch(y_true, y_pred)
            )

    def evaluate_weights(self, q2_obs_scaled, save_to, split_group, tag):
        metric_max = np.zeros(q2_obs_scaled.shape[0])
        batch_size = 64
        n_batchs = q2_obs_scaled.shape[0] // batch_size
        
        for batch_index in range(n_batchs):
            start = batch_index * batch_size
            stop = (batch_index + 1) * batch_size
            metric = self.call(
                keras.ops.cast(
                    q2_obs_scaled[start:stop, :self.model_params['extraction_peak_length']],
                    dtype='float32'
                    )
                )
            metric_max[start: stop] = tensor_to_numpy(metric).max(axis=(1, 2))
        start = (batch_index + 1) * batch_size
        metric = self.call(
            keras.ops.cast(
                q2_obs_scaled[start:, :self.model_params['extraction_peak_length']],
                dtype='float32'
                )
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
        volumes = self.volumes.numpy()[:, 0]
        filters = self.filters.numpy()[0]
        sigma = self.sigma.numpy()
        filter_spacing = filters[1] - filters[0]

        # Every peak is rendered with the same width now, so what matters is how that width compares
        # to the grid it is sampled on and to how far a peak walks between adjacent volume branches.
        # A shift below ~1 sigma means adjacent branches overlap for that peak.
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        for q2_representative in (0.5, 4.0):
            branch_shift = q2_representative * np.diff(volumes) / (volumes[1:] * volumes[:-1])
            axes.plot(branch_shift / sigma, marker='.', label=f'q2 = {q2_representative}')
        axes.axhline(1.0, linestyle='dashed', color=[0, 0, 0])
        axes.set_yscale('log')
        axes.set_ylabel('Branch-to-branch\nshift / sigma')
        axes.set_xlabel('Volume Index')
        axes.set_title(
            f'sigma {sigma:0.5f} = {sigma/filter_spacing:0.2f} x spacing {filter_spacing:0.5f}'
            )
        axes.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_volume_diff_{tag}.png'))
        plt.close()

        # The volume-normalized peak separations sigma was fitted to, against the 2*sigma resolution
        # limit. Everything left of the dashed line is a peak pair that merges.
        if hasattr(self, 'peak_separations_init'):
            peak_separations = self.peak_separations_init
            fig, axes = plt.subplots(1, 1, figsize=(4, 3))
            axes.hist(peak_separations, bins=np.linspace(0, 0.5, 101), density=True)
            axes.axvline(2*sigma, linestyle='dashed', color=[0, 0, 0])
            merged = 100 * (peak_separations < 2*sigma).mean()
            axes.set_title(f'{merged:0.1f}% of peak pairs merge')
            axes.set_xlabel('Volume-normalized peak separation')
            axes.set_ylabel('distribution')
            fig.tight_layout()
            fig.savefig(os.path.join(f'{save_to}', f'{split_group}_pitf_separations_{tag}.png'))
            plt.close()

        # Grid coverage. Post-fix the grid is fixed and what moves across it is q2_obs / v, so that
        # projection is what has to land inside [filters.min(), filters.max()].
        bins = np.linspace(0, 5, 101)
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes.hist(
            q2_obs_scaled[:, :self.model_params['extraction_peak_length']].ravel(),
            bins=bins, label='q2_obs_scaled', density=True
            )
        # Subsampled: the full projection is n_entries x n_volumes x n_peaks.
        projection_sample = q2_obs_scaled[
            ::max(1, q2_obs_scaled.shape[0] // 2000), :self.model_params['extraction_peak_length']
            ]
        axes.hist(
            (projection_sample[:, np.newaxis, :] / volumes[np.newaxis, :, np.newaxis]).ravel(),
            bins=bins, alpha=0.75, label='q2_obs_scaled / volumes', density=True
            )
        axes.axvline(filters.min(), linestyle='dotted', color=[0, 0, 0])
        axes.axvline(filters.max(), linestyle='dotted', color=[0, 0, 0])
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
                    )
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


class MetricVolumeRescale(keras.layers.Layer):
    """Turn the regression head's output from an absolute xnn into a volume-normalized shape.

    The head predicts a unit cell shape; the absolute scale comes from the branch's own trial volume.
    Because q2 = h.G*.h^T, changing the unit cell volume multiplies every q2 by a common factor, so
    the shape a branch sees is the true cell divided by that branch's volume v. Reconstructing the
    absolute value and re-applying the model's own median/MAD normalization,

        shape       = (xnn_mean + xnn_scale * D) / q2_obs_scale
        xnn_raw     = shape * v * q2_obs_scale = (xnn_mean + xnn_scale * D) * v
        xnn_scaled  = (xnn_raw - xnn_mean) / xnn_scale = D * v + (xnn_mean / xnn_scale) * (v - 1)

    q2_obs_scale cancels. Two properties matter. At v = 1 this is exactly D, and ExtractionLayer
    normalizes the volumes to a median of 1, so the median branch is unchanged at initialization and
    this is a strict generalization of the previous head rather than a different model. And the
    (v - 1) term encodes the prior that xnn scales with the trial volume, which is what the geometry
    demands, so the head starts near the right magnitude at every branch instead of only the median
    one.

    Only the first unit_cell_length channels are rescaled. The last channel is the branch logit and
    passes through untouched, so the output layout stays (batch, n_volumes, unit_cell_length + 1) as
    loss_function_common expects.
    """
    def __init__(self, volumes_fn, xnn_mean, xnn_scale, unit_cell_length, **kwargs):
        super().__init__(**kwargs)
        # A plain callable, not the layer itself. Holding ExtractionLayer as an attribute would
        # register it as a sublayer and change the weights.h5 layout; this keeps the checkpoint
        # structure identical to the previous model. It also reads volumes live at call time, which
        # is required: on the load_from_tag path build_model(data=None) builds the graph while that
        # weight is still zeros, and it is only filled when the weights are loaded afterwards.
        self._volumes_fn = volumes_fn
        self.unit_cell_length = unit_cell_length
        # Baked in as a constant rather than a weight so that no new entry appears in weights.h5.
        self.xnn_offset = np.reshape(
            np.asarray(xnn_mean, dtype='float32') / np.asarray(xnn_scale, dtype='float32'),
            (1, 1, unit_cell_length)
            )

    def call(self, x):
        # x:       batch_size, n_volumes, unit_cell_length + 1
        # volumes: 1, n_volumes, 1
        volumes = keras.ops.reshape(self._volumes_fn(), (1, -1, 1))
        xnn_offset = keras.ops.cast(self.xnn_offset, dtype=x.dtype)
        xnn_scaled = x[:, :, :self.unit_cell_length]*volumes + xnn_offset*(volumes - 1.0)
        logits = x[:, :, self.unit_cell_length:]
        return keras.ops.concatenate([xnn_scaled, logits], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape


class IntraVolume_MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0
        
        self.W_q = keras.layers.Dense(
            d_model,
            use_bias=False,
            activation=keras.activations.elu,
            kernel_initializer=keras.initializers.HeUniform,
        )
        self.W_k = keras.layers.Dense(
            d_model,
            use_bias=False,
            activation=keras.activations.elu,
            kernel_initializer=keras.initializers.HeUniform,
        )
        self.W_v = keras.layers.Dense(
            d_model,
            use_bias=False,
            activation=keras.activations.elu,
            kernel_initializer=keras.initializers.HeUniform,
        )
        self.W_o = keras.layers.Dense(
            d_model,
            use_bias=False,
            activation=keras.activations.elu,
            kernel_initializer=keras.initializers.HeUniform,
        )
        
    def call(self, x):
        # x shape: (batch_size, n_volumes, n_filters)
        batch_size = keras.ops.shape(x)[0]
        n_volumes = keras.ops.shape(x)[1] 
        n_filters = keras.ops.shape(x)[2]
        
        # Generate Q, K, V - Dense automatically applies to each volume
        # (batch_size, n_volumes, d_model)
        Q = self.W_q(x) 
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = keras.ops.reshape(Q, (batch_size, n_volumes, self.n_heads, self.d_k))
        K = keras.ops.reshape(K, (batch_size, n_volumes, self.n_heads, self.d_k))
        V = keras.ops.reshape(V, (batch_size, n_volumes, self.n_heads, self.d_k))
        
        # Transpose to (batch_size, n_heads, n_volumes, d_k)
        Q = keras.ops.transpose(Q, [0, 2, 1, 3])
        K = keras.ops.transpose(K, [0, 2, 1, 3])
        V = keras.ops.transpose(V, [0, 2, 1, 3])
        
        # Compute attention scores within each head
        # (batch_size, n_heads, n_volumes, n_volumes)
        scores = keras.ops.matmul(Q, keras.ops.transpose(K, [0, 1, 3, 2])) / math.sqrt(self.d_k)
        attention_weights = keras.ops.softmax(scores, axis=-1)
        
        # Apply attention to values
        # (batch_size, n_heads, n_volumes, d_k)
        attended = keras.ops.matmul(attention_weights, V)
        
        # Concatenate heads and reshape back
        # (batch_size, n_volumes, d_model)
        attended = keras.ops.transpose(attended, [0, 2, 1, 3])
        attended = keras.ops.reshape(attended, (batch_size, n_volumes, self.d_model))
        # Final projection
        output = self.W_o(attended)  # (batch_size, n_volumes, d_model)
        return output


class IntraVolume_Attention(keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
        self.W_q = keras.layers.Dense(
            d_model, 
            use_bias=False,
            activation=keras.activations.elu,
            kernel_initializer=keras.initializers.HeUniform,
        )
        self.W_k = keras.layers.Dense(
            d_model, 
            use_bias=False,
            activation=keras.activations.elu,
            kernel_initializer=keras.initializers.HeUniform,
        )
        self.W_v = keras.layers.Dense(
            d_model, 
            use_bias=False,
            activation=keras.activations.elu,
            kernel_initializer=keras.initializers.HeUniform,
        )
        
    def call(self, x):
        # x shape: (batch_size, n_volumes, n_filters)
        batch_size = keras.ops.shape(x)[0]
        n_volumes = keras.ops.shape(x)[1]
        n_filters = keras.ops.shape(x)[2]
        
        # Add feature dimension: (batch_size, n_volumes, n_filters, 1)
        x_expanded = keras.ops.expand_dims(x, -1)
        
        # Apply same linear transformations to each volume
        # (batch_size, n_volumes, n_filters, d_model)
        Q = self.W_q(x_expanded)  
        K = self.W_k(x_expanded)
        V = self.W_v(x_expanded)
        
        # Compute attention scores for each volume independently
        # (batch_size, n_volumes, n_filters, n_filters)
        scores = keras.ops.matmul(Q, keras.ops.transpose(K, [0, 1, 3, 2])) / math.sqrt(self.d_model)
        attention_weights = keras.ops.softmax(scores, axis=-1)
        
        # Apply attention: (batch_size, n_volumes, n_filters, d_model)
        attended = keras.ops.matmul(attention_weights, V)
        
        # Reduce to final output: (batch_size, n_volumes, d_model)
        output = keras.ops.mean(attended, axis=2)
        return output
