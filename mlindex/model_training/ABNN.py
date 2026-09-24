import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special

from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
from mlindex.utilities.FigureOfMerits import get_assignment_distribution
from mlindex.utilities.IOManagers import NeuralNetworkManager
from mlindex.utilities.IOManagers import read_params
from mlindex.utilities.IOManagers import write_params
from mlindex.utilities.MillerIndexAssignment import vectorized_resampling
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.UnitCellTools import get_xnn_from_reciprocal_unit_cell
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import reciprocal_uc_conversion


class ABNN:
    def __init__(self, split_group, data_params, model_params, save_to, hkl_ref):
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
        self.lattice_system = self.data_params['lattice_system']
        self.hkl_ref = hkl_ref

    def get_titles(self, kind):
        """Labels for the active unit cell / xnn components of this lattice system.

        Both lists are indexed by unit_cell_indices, not positionally. For monoclinic the active
        components are indices [0, 1, 2, 4], so the fourth one is beta / Xhl, not alpha / Xkl --
        Xkl and Xhk are identically zero. Labelling positionally mislabels every lattice system
        whose indices are not a prefix of the full list.
        """
        all_titles = {
            'unit_cell': ['a', 'b', 'c', 'alpha', 'beta', 'gamma'],
            'xnn': ['Xhh', 'Xkk', 'Xll', 'Xkl', 'Xhl', 'Xhk'],
            }
        return [all_titles[kind][index] for index in self.unit_cell_indices]

    def get_branch_labels(self, data):
        """Index of the volume branch that matches each entry's true unit cell volume.

        The reciprocal volume is derived the same way build_model derives it for the branch grid, so
        the label lines up with the branches the model actually has.
        """
        unit_cell = np.stack(data['reindexed_unit_cell'])[:, self.unit_cell_indices]
        reciprocal_unit_cell = reciprocal_uc_conversion(
            unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        reciprocal_volume = get_unit_cell_volume(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return self.extraction_layer.get_branch_labels(reciprocal_volume)

    def setup(self, data):
        model_params_defaults = {
            'peak_length': 20,
            'extraction_peak_length': 6,
            'n_volumes': 200,
            'n_filters': 200,
            'layers': [200, 100, 50],
            'l1_regularization': 0.0,
            'base_line_layers': [200, 100, 50],
            'base_line_dropout_rate': 0.0,
            'learning_rate': 0.00005,
            'd_model': 512,
            'n_heads': 8,
            # Weight on the auxiliary cross entropy that supervises which volume branch is correct.
            # Without it the branch ranking is only supervised indirectly, through a regression loss
            # that weights each branch by its own confidence and never says which one is right.
            'branch_loss_weight': 0.2,
            # Percentiles of the reciprocal volume distribution the branch grid spans, and how the
            # branches are spaced within it. 1.0 spaces them by equal probability, 0.0 by equal
            # steps in 1/v and therefore equal misalignment; 0.5 trades a slightly worse median
            # alignment for far fewer badly misaligned entries, and the wider span leaves fewer
            # entries with no usable branch at all. Measured over the monoclinic split groups:
            # entries outside the grid 1.75% -> 0.4%, entries misaligned by more than a peak width
            # 5.3% -> 2.4%.
            'volume_lower_percentile': 0.001,
            'volume_upper_percentile': 0.999,
            'volume_spacing_blend': 0.5,
            'epochs': 20,
            'batch_size': 64,
            'loss_type': 'log_cosh',
            'model_type': 'metric',
            }

        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]

        self.model_params['unit_cell_length'] = self.unit_cell_length
        self.build_model(data=data)

    def save(self, train_inputs):
        import keras
        write_params(
            self.model_params,
            os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_abnn_params_{self.model_params["tag"]}.csv'
                )
            )
        np.save(
            os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_q2_obs_scale_{self.model_params["tag"]}.npy'
                ),
            self.q2_obs_scale
            )
        np.save(
            os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_xnn_scaler_{self.model_params["tag"]}.npy'
                ),
            np.array((self.xnn_mean, self.xnn_scale))
            )

        model_manager = NeuralNetworkManager(
            model_name=f'{self.split_group}_abnn_weights_{self.model_params["tag"]}',
            save_dir=f'{self.save_to_split_group}',
            )
        model_manager.save_keras_weights(self.model)
        model_manager.convert_to_onnx(
            self.model,
            example_inputs=train_inputs,
            input_signature=keras.Input(
                shape=(self.model_params['peak_length'],),
                name='q2_obs_scaled',
                dtype='float32',
                )
            )
        model_manager.quantize_onnx(
            method='dynamic',
            calibration_data=train_inputs
            )

    def load_from_tag(self, mode):
        params = read_params(os.path.join(
            f'{self.save_to_split_group}',
            f'{self.split_group}_abnn_params_{self.model_params["tag"]}.csv'
            ))
        params_keys = [
            'tag',
            'peak_length',
            'extraction_peak_length',
            'n_volumes',
            'n_filters',
            'layers',
            'd_model',
            'n_heads',
            'l1_regularization',
            'base_line_layers',
            'base_line_dropout_rate',
            'learning_rate',
            'epochs',
            'batch_size',
            'loss_type',
            'model_type',
            ]
        #self.model_params = dict.fromkeys(params_keys)
        assert mode in ['training', 'inference']
        self.model_params['mode'] = mode 
        self.model_params['tag'] = params['tag']
        self.model_params['peak_length'] = int(params['peak_length'])
        self.model_params['extraction_peak_length'] = int(params['extraction_peak_length'])
        self.model_params['n_volumes'] = int(params['n_volumes'])
        self.model_params['n_filters'] = int(params['n_filters'])
        self.model_params['d_model'] = int(params['d_model'])
        self.model_params['n_heads'] = int(params['n_heads'])
        self.model_params['layers'] = [int(i) for i in params['layers'].split('[')[1].split(']')[0].split(',')]
        self.model_params['base_line_layers'] = [int(i) for i in params['base_line_layers'].split('[')[1].split(']')[0].split(',')]
        self.model_params['l1_regularization'] = float(params['l1_regularization'])
        self.model_params['base_line_dropout_rate'] = float(params['base_line_dropout_rate'])
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['epochs'] = int(params['epochs'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['loss_type'] = params['loss_type']
        self.model_params['model_type'] = params['model_type']
        # Defaulted rather than required: models trained before the auxiliary branch loss existed
        # have no such column, and 0.0 is what they were trained with.
        # Recorded for provenance rather than needed to rebuild: the branch grid is stored in the
        # weights as ExtractionLayer.volumes, and build_model(data=None) never refits it. The
        # fallbacks are deliberately the values these settings had before they were settable, not
        # the current defaults, because they describe what a file written back then was trained
        # with rather than what a new model should use.
        self.model_params['branch_loss_weight'] = float(params.get('branch_loss_weight', 0.0))
        self.model_params['volume_lower_percentile'] = float(
            params.get('volume_lower_percentile', 0.005))
        self.model_params['volume_upper_percentile'] = float(
            params.get('volume_upper_percentile', 0.990))
        self.model_params['volume_spacing_blend'] = float(
            params.get('volume_spacing_blend', 1.0))

        self.q2_obs_scale = np.load(
            os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_q2_obs_scale_{self.model_params["tag"]}.npy'
                ),
            )
        self.xnn_mean, self.xnn_scale = np.load(os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_xnn_scaler_{self.model_params["tag"]}.npy'
                ),
            )

        model_manager = NeuralNetworkManager(
            model_name=f'{self.split_group}_abnn_weights_{self.model_params["tag"]}',
            save_dir=self.save_to_split_group,
            )
        if mode == 'training':
            self.build_model(data=None)
            self.compile_model()
            self.model = model_manager.load_keras_model(self.model)
        elif mode == 'inference':
            self.onnx_model = model_manager.load_onnx_model(quantized=True)

    def build_model(self, data=None):
        from mlindex.model_training.Networks import ExtractionLayer
        import keras
        # Build the ABNN model #
        keras.utils.set_random_seed(1)
        #tf.config.experimental.enable_op_determinism()
        if not data is None:
            training_data = data[data['train']]
            #training_data = training_data[~training_data['augmented']]
            q2_obs = np.stack(training_data['q2'])[:, :self.model_params['extraction_peak_length']]
            self.q2_obs_scale = q2_obs.std()
            unit_cell = np.stack(training_data['reindexed_unit_cell'])[:, self.unit_cell_indices]
            xnn = get_xnn_from_unit_cell(unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
            reciprocal_unit_cell = reciprocal_uc_conversion(
                unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            reciprocal_volume = get_unit_cell_volume(
                reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            self.extraction_layer = ExtractionLayer(
                self.model_params, q2_obs, xnn, reciprocal_volume, self.q2_obs_scale,
                name='extraction_layer'
                )

            # The regression head bakes these into the graph, so they have to exist before the model
            # is built rather than being computed in train(). Kept identical to what train() uses for
            # the targets: the reindexed_xnn column, not the xnn derived from reindexed_unit_cell
            # above, and the same rhombohedral filter.
            train_xnn = np.stack(training_data['reindexed_xnn'])[:, self.unit_cell_indices]
            if self.lattice_system == 'rhombohedral':
                train_xnn = train_xnn[np.max(np.abs(train_xnn), axis=1) < 0.05]
            self.xnn_mean = np.median(train_xnn, axis=0)[np.newaxis]
            self.xnn_scale = np.median(np.abs(train_xnn - self.xnn_mean), axis=0)[np.newaxis]
        else:
            # load_from_tag loads the scalers from .npy before calling build_model.
            assert hasattr(self, 'xnn_mean') and hasattr(self, 'xnn_scale'), (
                'build_model(data=None) needs xnn_mean and xnn_scale to already be set, because the '
                'regression head bakes them into the graph.'
                )
            self.extraction_layer = ExtractionLayer(
                self.model_params, None, None, None, self.q2_obs_scale,
                name='extraction_layer'
                )
        inputs = keras.Input(
            shape=(self.model_params['peak_length'],),
            name='q2_obs_scaled',
            dtype='float32',
            )
        if self.model_params['model_type'] == 'metric':
            self.model = keras.Model(inputs, self.model_builder_metric(inputs))
        elif self.model_params['model_type'] == 'base_line':
            self.model = keras.Model(inputs, self.model_builder_base_line(inputs))
        self.compile_model()
        #self.model.summary()

    def model_builder_base_line(self, inputs):
        import keras
        # inputs['q2_obs_scaled']: batch_size, n_peaks
        # This is a 'Base line model' that does not use feature extraction
        x = inputs
        for index in range(len(self.model_params['base_line_layers'])):
            x = keras.layers.Dense(
                self.model_params['base_line_layers'][index],
                activation='linear',
                name=f'base_line_dense_{index}',
                use_bias=True,
                )(x)
            x = keras.layers.LayerNormalization(
                name=f'base_line_layer_norm_{index}',
                center=False,
                )(x)
            x = keras.activations.relu(x)
            x = keras.layers.Dropout(
                rate=self.model_params['base_line_dropout_rate'],
                name=f'base_line_dropout_{index}',
                )(x)

        # output: batch_size, n_volumes, unit_cell_length + 1
        output = keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name='base_line_xnn_scaled',
            )(keras.ops.expand_dims(x, axis=1))
        return output

    def model_builder_metric(self, inputs):
        import keras
        from mlindex.model_training.Networks import IntraVolume_MultiHeadAttention
        from mlindex.model_training.Networks import MetricVolumeRescale
        # inputs: batch_size, n_peaks
        # metric: batch_size, n_volumes, n_filters

        #####################
        # Metric prediction #
        #####################
        metric = self.extraction_layer(
            inputs[:, :self.model_params['extraction_peak_length']],
            name='extraction_layer'
            )
        # attended: batch_size, n_volumes, d_model
        attended = IntraVolume_MultiHeadAttention(
            d_model=self.model_params['d_model'],
            n_heads=self.model_params['n_heads'],
        )(metric)
        x = keras.layers.UnitNormalization(axis=2)(attended)
        
        #################
        # Hidden layers #
        #################
        for index in range(len(self.model_params['layers'])):
            x = keras.layers.Dense(
                self.model_params['layers'][index],
                activation=keras.activations.elu,
                name=f'metric_dense_{index}',
                use_bias=False,
                kernel_regularizer=keras.regularizers.L1(
                    l1=self.model_params['l1_regularization']
                    ),
                kernel_initializer=keras.initializers.HeUniform
                )(x)

        # raw: batch_size, n_volumes, unit_cell_length + 1
        raw = keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name=f'{self.model_params["model_type"]}_raw',
            kernel_initializer=keras.initializers.HeUniform
            )(x)

        # The Dense weights are shared across all n_volumes branches, so it predicts a volume
        # normalized shape and each branch reapplies its own trial volume. This layer must keep the
        # xnn_scaled name because compile_model keys its loss and metric dicts on it.
        # output: batch_size, n_volumes, unit_cell_length + 1
        output = MetricVolumeRescale(
            volumes_fn=lambda: self.extraction_layer.volumes,
            xnn_mean=self.xnn_mean,
            xnn_scale=self.xnn_scale,
            unit_cell_length=self.unit_cell_length,
            name=f'{self.model_params["model_type"]}_xnn_scaled',
            )(raw)
        return output

    def compile_model(self):
        import keras
        # Create learning rate scheduler
        optimizer = keras.optimizers.Adam(
            learning_rate=self.model_params['learning_rate'],
            )
        # The training loss carries the auxiliary branch term; the metrics keep the pure regression
        # errors under their existing names so the loss curves stay comparable across runs.
        if self.model_params['loss_type'] == 'mse':
            loss_functions = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.training_loss_mse
                }
        else:
            loss_functions = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.training_loss_log_cosh
                }
        loss_metrics = {
            f'{self.model_params["model_type"]}_xnn_scaled': [
                self.extraction_layer.loss_function_log_cosh,
                self.extraction_layer.loss_function_mse,
                self.extraction_layer.loss_function_branch,
                self.extraction_layer.branch_accuracy,
                ]
            }
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            metrics=loss_metrics,
            run_eagerly=False,
            )

    def train(self, data):
        import keras
        train = data[data['train']]
        val = data[~data['train']]


        train_q2_obs = np.stack(train['q2'])[:, :self.model_params['peak_length']]
        val_q2_obs = np.stack(val['q2'])[:, :self.model_params['peak_length']]
        train_q2_obs_scaled = train_q2_obs / self.q2_obs_scale
        val_q2_obs_scaled = val_q2_obs / self.q2_obs_scale

        train_xnn = np.stack(train['reindexed_xnn'])[:, self.unit_cell_indices]
        val_xnn = np.stack(val['reindexed_xnn'])[:, self.unit_cell_indices]

        train_branch = self.get_branch_labels(train)
        val_branch = self.get_branch_labels(val)

        if self.lattice_system == 'rhombohedral':
            # There are very large values of xnn being included for rhombohedral.
            # These need to be excluded or NaNs will occur in the model.
            train_indices = np.max(np.abs(train_xnn), axis=1) < 0.05
            val_indices = np.max(np.abs(val_xnn), axis=1) < 0.05
            train_xnn = train_xnn[train_indices]
            train_q2_obs_scaled = train_q2_obs_scaled[train_indices]
            train_branch = train_branch[train_indices]
            val_xnn = val_xnn[val_indices]
            val_q2_obs_scaled = val_q2_obs_scaled[val_indices]
            val_branch = val_branch[val_indices]
            train_unaugmented = np.invert(train['augmented'][train_indices])
        else:
            train_unaugmented = np.invert(train['augmented'])

        # xnn_mean and xnn_scale are computed in build_model, which the regression head bakes into
        # the graph. Recomputing them here would risk the constants in the graph drifting from the
        # ones used to scale the targets.
        train_xnn_scaled = (train_xnn - self.xnn_mean) / self.xnn_scale
        val_xnn_scaled = (val_xnn - self.xnn_mean) / self.xnn_scale

        fig, axes = plt.subplots(1, self.unit_cell_length + 1, figsize=(8, 3))
        bins0 = np.linspace(0, 5, 301)
        bins1 = np.linspace(-5, 5, 301)
        for index in range(self.unit_cell_length + 1):
            if index == 0:
                axes[index].hist(
                    train_q2_obs_scaled[train_unaugmented, :self.model_params['extraction_peak_length']].ravel(),
                    bins=bins0, density=True, label='No Aug'
                    )
                axes[index].hist(
                    train_q2_obs_scaled[~train_unaugmented, :self.model_params['extraction_peak_length']].ravel(),
                    bins=bins0, density=True, label='Aug', alpha=0.5
                    )
                axes[index].plot(bins0, 2/np.sqrt(2*np.pi)*np.exp(-1/2*bins0**2), color=[1, 0, 0])
            else:
                axes[index].hist(train_xnn_scaled[train_unaugmented, index - 1], bins=bins1, density=True)
                axes[index].hist(train_xnn_scaled[~train_unaugmented, index - 1], bins=bins1, density=True, alpha=0.5)
                axes[index].plot(bins1, 1/np.sqrt(2*np.pi)*np.exp(-1/2*bins1**2), color=[1, 0, 0])
        axes[0].set_title('q2_obs_scaled')
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to_split_group}',
            f'{self.split_group}_abnn_io_{self.model_params["tag"]}.png'
            ))
        plt.close()

        # The branch label rides along as the last column of the target so the loss can supervise
        # the branch logits as well as the regression. Nothing downstream of the model changes.
        train_true = np.concatenate([train_xnn_scaled, train_branch[:, np.newaxis]], axis=1)
        val_true = np.concatenate([val_xnn_scaled, val_branch[:, np.newaxis]], axis=1)
        train_inputs = train_q2_obs_scaled
        val_inputs = val_q2_obs_scaled

        if self.model_params['model_type'] != 'base_line':
            self.extraction_layer.evaluate_init(
                train_q2_obs_scaled[train_unaugmented],
                self.save_to_split_group,
                self.split_group,
                self.model_params["tag"]
                )

        # Nothing is written until fit() returns, so a job killed part way through loses every epoch
        # it ran. That forces long walltime requests, which are the hardest thing for a scheduler to
        # backfill, so the job waits longer still. BackupAndRestore writes the weights, the optimizer
        # state and the epoch number after each epoch and restores all three on the next fit(), so an
        # interrupted run resumes instead of restarting. It keeps one checkpoint, overwritten in
        # place, and deletes the directory once fit() completes -- so the backup existing is itself
        # the signal that the previous run was interrupted. There is no flag to remember to set, and
        # a group that finished normally retrains from scratch exactly as before.
        backup_directory = os.path.join(
            self.save_to_split_group,
            f'{self.split_group}_training_backup_{self.model_params["tag"]}'
            )
        if os.path.exists(os.path.join(backup_directory, 'latest.weights.h5')):
            print(f'resuming {self.split_group} from the backup in {backup_directory}')
        self.fit_history = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'],
            shuffle=True,
            batch_size=self.model_params['batch_size'],
            validation_data=(val_inputs, val_true),
            callbacks=[keras.callbacks.BackupAndRestore(backup_dir=backup_directory)],
            )
        self.save(train_inputs)

        ##############################
        # Plot training loss vs time #
        ##############################
        fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
        # A resumed run only records the epochs it actually ran, so plotting the history against an
        # implicit index would relabel epoch 20 as epoch 0. fit_history.epoch carries the real
        # numbers. The epochs before the interruption are genuinely lost from the curves.
        epochs = self.fit_history.epoch
        for index, (train_key, val_key) in enumerate([
                ('loss', 'val_loss'),
                ('loss_function_log_cosh', 'val_loss_function_log_cosh'),
                ('loss_function_mse', 'val_loss_function_mse'),
                ('branch_accuracy', 'val_branch_accuracy'),
                ]):
            axes[index].plot(
                epochs, self.fit_history.history[train_key], label='Training', marker='.'
                )
            axes[index].plot(
                epochs, self.fit_history.history[val_key], label='Validation', marker='v'
                )
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Log-Cosh Error')
        axes[2].set_ylabel('MSE Error')
        axes[3].set_ylabel('Branch Accuracy')
        axes[3].set_xlabel('Epoch')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to_split_group}',
            f'{self.split_group}_abnn_training_loss_{self.model_params["tag"]}.png'
            ))
        plt.close()

        if self.model_params['model_type'] != 'base_line':
            self.extraction_layer.evaluate_weights(
                train_inputs, 
                self.save_to_split_group,
                self.split_group,
                self.model_params["tag"]
                )

    def predict_xnn(self, top_n, rng, data=None, inputs=None, q2_obs=None, batch_size=None):
        if not data is None:
            q2_obs = np.stack(data['q2'])[:, :self.model_params['peak_length']]
        elif not inputs is None:
            q2_obs = np.array(inputs['q2'])[:self.model_params['peak_length']]
        elif not q2_obs is None:
            q2_obs = q2_obs[:, :self.model_params['peak_length']]
        q2_obs_scaled = q2_obs / self.q2_obs_scale

        N = q2_obs_scaled.shape[0]
        xnn_pred_scaled = np.zeros((N, self.model_params['n_volumes'], self.unit_cell_length))
        logits_pred = np.zeros((N, self.model_params['n_volumes']))
        if self.model_params['mode'] == 'inference':
            for index in range(N):
                output = self.onnx_model.run(
                    None,
                    {'input': q2_obs_scaled[index].astype(np.float32)[np.newaxis]}
                    )[0]
                xnn_pred_scaled[index] = output[0, :, :self.unit_cell_length]
                logits_pred[index] = output[0, :, self.unit_cell_length]
        elif self.model_params['mode'] == 'training':
            if batch_size is None:
                batch_size = self.model_params['batch_size']
            n_batches = N // batch_size
            left_over = N % batch_size
            # predict_on_batch helps with a memory leak...
            for batch_index in range(n_batches + 1):
                start = batch_index * batch_size
                if batch_index == n_batches:
                    batch_inputs = np.zeros((batch_size, self.model_params['peak_length']))
                    batch_inputs[:left_over] = q2_obs_scaled[start: start + left_over]
                    batch_inputs[left_over:] = q2_obs_scaled[0]
                else:
                    batch_inputs = q2_obs_scaled[start: start + batch_size]

                outputs = self.model.predict_on_batch(batch_inputs)
                if batch_index == n_batches:
                    xnn_pred_scaled[start:] = outputs[:left_over, :, :self.unit_cell_length]
                    logits_pred[start:] = outputs[:left_over, :, self.unit_cell_length]
                else:
                    xnn_pred_scaled[start: start + batch_size] = outputs[:, :, :self.unit_cell_length]
                    logits_pred[start: start + batch_size] = outputs[:, :, self.unit_cell_length]
        softmax_pred = scipy.special.softmax(logits_pred, axis=1)
        xnn_pred_scaled_top_n = np.take_along_axis(
            xnn_pred_scaled,
            np.argsort(softmax_pred, axis=1)[:, ::-1][:, :top_n, np.newaxis],
            axis=1
            )
        softmax_pred_top_n = np.sort(softmax_pred, axis=1)[:, ::-1][:, :top_n]
        xnn_pred_top_n = xnn_pred_scaled_top_n*self.xnn_scale[:, np.newaxis] + self.xnn_mean[:, np.newaxis]
        for index in range(top_n):
            xnn_pred_top_n[:, index, :] = fix_unphysical(
                xnn=xnn_pred_top_n[:, index, :],
                lattice_system=self.data_params['lattice_system'],
                rng=rng
                )
        return xnn_pred_top_n, softmax_pred_top_n

    def generate(self, n_unit_cells, rng, q2_obs, top_n=None, batch_size=None):
        from mlindex.utilities.Q2Calculator import Q2Calculator
        from mlindex.utilities.numba_functions import fast_assign
        if top_n is None:
            top_n = self.model_params['n_volumes']

        q2_calculator = Q2Calculator(
            lattice_system=self.lattice_system,
            hkl=self.hkl_ref,
            tensorflow=False,
            representation='xnn'
            )

        if top_n > n_unit_cells:
            xnn_gen, _ = self.predict_xnn(n_unit_cells, rng, q2_obs=q2_obs[np.newaxis], batch_size=batch_size)
            xnn_gen = xnn_gen[0]
            q2_ref_calc = q2_calculator.get_q2(xnn_gen)
            hkl_assign = fast_assign(q2_obs, q2_ref_calc)
            hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
        else:
            n_unit_cells_per_pred = n_unit_cells // top_n
            n_extra = n_unit_cells % top_n
            xnn_gen = np.zeros((n_unit_cells, self.unit_cell_length))
            hkl_assign = np.zeros((n_unit_cells, self.data_params['n_peaks']), dtype=int)

            # If top_n == 5, then self.predict_xnn generates 5 unit cells
            # xnn_pred: 1, top_n, unit_cell_length
            xnn_pred, _ = self.predict_xnn(top_n, rng, q2_obs=q2_obs[np.newaxis], batch_size=batch_size)
            xnn_pred = xnn_pred[0]
            xnn_gen[:top_n] = xnn_pred
            q2_ref_calc = q2_calculator.get_q2(xnn_pred)
            hkl_assign[:top_n] = fast_assign(q2_obs, q2_ref_calc)

            # Resampling needs to generate n_unit_cells_per_pred - 1 unit cells from each
            # prediction, drawing a different Miller index labelling each time, so it needs a
            # distribution over this cell's reference lines for every peak.
            #
            # q2_ref_calc above is self.hkl_ref evaluated on exactly these predicted cells, so
            # the distribution is over exactly the lines fast_assign just chose between.
            # vectorized_resampling rescales its draws by each row's own cumulative total, so the
            # unnormalised form is what it wants and is one array pass cheaper.
            # hkl_softmax: top_n, n_peaks, hkl_ref_length
            hkl_softmax = get_assignment_distribution(
                q2_obs, q2_ref_calc, self.lattice_system, normalise=False
                )
            start = top_n
            for gen_index in range(n_unit_cells_per_pred - 1):
                # This generates top_n unit cells per iteration
                hkl_assign[start: start + top_n], _ = vectorized_resampling(hkl_softmax, rng)
                xnn_gen[start: start + top_n] = xnn_pred
                start += top_n
            hkl_assign[start: start + n_extra], _ = vectorized_resampling(hkl_softmax[:n_extra], rng)
            # The leftover candidates get a predicted cell too. Without this they keep the zeros
            # xnn_gen was allocated with, so a degenerate metric tensor enters the Gauss-Newton
            # step below and what comes out is noise, not a candidate. It is not a rare edge:
            # n_extra is n_unit_cells % top_n, which is 75 of hP's 175 candidates a split group,
            # 100 of mC's and mP's 550, and 50 of tI's and tP's 350.
            xnn_gen[start: start + n_extra] = xnn_pred[:n_extra]
            hkl = np.take_along_axis(self.hkl_ref[:, np.newaxis, :], hkl_assign[:, :, np.newaxis], axis=0)

        # hkl: n_unit_cells, n_peaks, 3
        target_function = CandidateOptLoss(
            np.repeat(q2_obs[np.newaxis], repeats=n_unit_cells, axis=0), 
            lattice_system=self.lattice_system,
            )
        target_function.update(hkl, xnn_gen)
        xnn_gen += target_function.gauss_newton_step(xnn_gen)
        xnn_gen = fix_unphysical(xnn=xnn_gen, rng=rng, lattice_system=self.lattice_system)
        unit_cell_gen = get_unit_cell_from_xnn(
            xnn_gen, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return unit_cell_gen

    def evaluate(self, data, quantitized_model=False):
        data = data[~data['augmented']]
        train = data[data['train']]
        val = data[~data['train']]

        train_q2_obs = np.stack(train['q2'])[:, :self.model_params['peak_length']]
        val_q2_obs = np.stack(val['q2'])[:, :self.model_params['peak_length']]
        train_q2_obs_scaled = train_q2_obs / self.q2_obs_scale
        val_q2_obs_scaled = val_q2_obs / self.q2_obs_scale

        train_inputs = train_q2_obs_scaled
        val_inputs = val_q2_obs_scaled

        train_unit_cell = np.stack(train['reindexed_unit_cell'])[:, self.unit_cell_indices]
        val_unit_cell = np.stack(val['reindexed_unit_cell'])[:, self.unit_cell_indices]
        train_xnn = get_xnn_from_unit_cell(train_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        val_xnn = get_xnn_from_unit_cell(val_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system)
        train_xnn_scaled = (train_xnn - self.xnn_mean) / self.xnn_scale
        val_xnn_scaled = (val_xnn - self.xnn_mean) / self.xnn_scale

        if quantitized_model:
            val_pred = np.zeros((
                val_q2_obs_scaled.shape[0],
                self.model_params['n_volumes'],
                self.unit_cell_length + 1
                ))
            for pred_index in range(val_q2_obs_scaled.shape[0]):
                val_pred[pred_index] = self.onnx_model.run(
                    None,
                    {'input': val_q2_obs_scaled[pred_index].astype(np.float32)[np.newaxis]}
                    )[0]
        else:
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
        if self.model_params['model_type'] == 'metric':
            for index in range(5):
                val_unit_cell_pred_top5[:, index, :] = get_unit_cell_from_xnn(
                    val_xnn_pred_top5[:, index, :], partial_unit_cell=True, lattice_system=self.lattice_system
                    )
        else:
            val_unit_cell_pred_top5[:, 0, :] = get_unit_cell_from_xnn(
                val_xnn_pred_top5[:, 0, :], partial_unit_cell=True, lattice_system=self.lattice_system
                )

        if quantitized_model:
            train_pred = np.zeros((
                train_q2_obs_scaled.shape[0],
                self.model_params['n_volumes'],
                self.unit_cell_length + 1
                ))
            for pred_index in range(train_q2_obs_scaled.shape[0]):
                train_pred[pred_index] = self.onnx_model.run(
                    None,
                    {'input': train_q2_obs_scaled[pred_index].astype(np.float32)[np.newaxis]}
                    )[0]
        else:
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
        if self.model_params['model_type'] == 'metric':
            for index in range(5):
                train_unit_cell_pred_top5[:, index, :] = get_unit_cell_from_xnn(
                    train_xnn_pred_top5[:, index, :], partial_unit_cell=True, lattice_system=self.lattice_system
                    )
        else:
            train_unit_cell_pred_top5[:, 0, :] = get_unit_cell_from_xnn(
                train_xnn_pred_top5[:, 0, :], partial_unit_cell=True, lattice_system=self.lattice_system
                )

        if not quantitized_model:
            for index in range(10):
                self.plot_predictions(
                    val_xnn_scaled[index],
                    val_all_xnn_scaled_pred[index],
                    val_softmax[index],
                    index
                    )

        ################################
        # Output unit cell evaluations #
        ################################
        if quantitized_model == False:
            val_xnn_pred = val_xnn_pred_top5[:, 0, :]
            val_unit_cell_pred = val_unit_cell_pred_top5[:, 0, :]
            train_xnn_pred = train_xnn_pred_top5[:, 0, :]
            train_unit_cell_pred = train_unit_cell_pred_top5[:, 0, :]

            train_unit_cell_error = np.abs(train_unit_cell_pred - train_unit_cell)
            val_unit_cell_error = np.abs(val_unit_cell_pred - val_unit_cell)
            train_xnn_error = np.abs(train_xnn_pred - train_xnn)
            val_xnn_error = np.abs(val_xnn_pred - val_xnn)
            train_X_error = np.linalg.norm(train_xnn_error, axis=1)
            val_X_error = np.linalg.norm(val_xnn_error, axis=1)

            train_size = train_xnn_pred.shape[0]
            val_size = val_xnn_pred.shape[0]

            unit_cell_titles = self.get_titles('unit_cell')
            xnn_titles = self.get_titles('xnn')
            output_dict = {}
            for uc_index in range(self.unit_cell_length):
                output_dict[f'rmse_train_{unit_cell_titles[uc_index]}'] = \
                    np.sqrt(1/train_size * np.linalg.norm(train_unit_cell_error[:, uc_index])**2)
                output_dict[f'rmse_val_{unit_cell_titles[uc_index]}'] = \
                    np.sqrt(1/val_size * np.linalg.norm(val_unit_cell_error[:, uc_index])**2)
                output_dict[f'rmse_train_{xnn_titles[uc_index]}'] = \
                    np.sqrt(1/train_size * np.linalg.norm(train_xnn_error[:, uc_index])**2)
                output_dict[f'rmse_val_{xnn_titles[uc_index]}'] = \
                    np.sqrt(1/val_size * np.linalg.norm(val_xnn_error[:, uc_index])**2)
                output_dict[f'mae_train_{unit_cell_titles[uc_index]}'] = \
                    np.nanmedian(train_unit_cell_error[:, uc_index])
                output_dict[f'mae_val_{unit_cell_titles[uc_index]}'] = \
                    np.nanmedian(val_unit_cell_error[:, uc_index])
                output_dict[f'mae_train_{xnn_titles[uc_index]}'] = \
                    np.nanmedian(train_xnn_error[:, uc_index])
                output_dict[f'mae_val_{xnn_titles[uc_index]}'] = \
                    np.nanmedian(val_xnn_error[:, uc_index])
            output_dict[f'rmse_train_X'] = \
                np.sqrt(1/train_size * np.linalg.norm(train_X_error)**2)
            output_dict[f'rmse_val_X'] = \
                np.sqrt(1/val_size * np.linalg.norm(val_X_error)**2)
            output_dict[f'mae_train_X'] = np.nanmedian(train_X_error)
            output_dict[f'mae_val_X'] = np.nanmedian(val_X_error)
            write_params(
                output_dict,
                os.path.join(
                    f'{self.save_to_split_group}',
                    f'{self.split_group}_abnn_reg_eval_{self.model_params["tag"]}_most_probable.csv'
                    )
                )

        ##############################
        # Plot unit cell evaluations #
        ##############################
        figsize = (self.unit_cell_length*2 + 2, 6)
        fig, axes = plt.subplots(2, self.unit_cell_length, figsize=figsize)
        if self.unit_cell_length == 1:
            axes = axes[:, np.newaxis]
        unit_cell_titles = self.get_titles('unit_cell')
        xnn_titles = self.get_titles('xnn')
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
            if quantitized_model:
                base_name = os.path.join(
                    f'{self.save_to_split_group}',
                    f'{self.split_group}_abnn_reg_eval_optimized_{self.model_params["tag"]}_{save_label}'
                    )
            else:
                base_name = os.path.join(
                    f'{self.save_to_split_group}',
                    f'{self.split_group}_abnn_reg_eval_{self.model_params["tag"]}_{save_label}'
                    )
            fig.savefig(f'{base_name}.png')
            plt.close()

            # The plot above bakes the numbers into pixels, so the validation points behind it are
            # also written out for figures that pool split groups. Validation only: the train points
            # are in the plot for diagnosis, but a pooled accuracy figure should not be reporting
            # error on entries the model fit. Shape is (2, n_val, unit_cell_length) with [0] true and
            # [1] predicted, so groups of one lattice system concatenate along axis 1. Angles stay in
            # radians, matching reindexed_unit_cell.
            np.save(
                f'{base_name}.npy',
                np.stack([val_unit_cell, val_unit_cell_pred], axis=0)
                )

        ##########################
        # Plot branch importance #
        ##########################
        if self.model_params['model_type'] == 'metric' and quantitized_model == False:
            n_branches = self.model_params['n_volumes']
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
            fig.savefig(os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_abnn_branch_importance_{self.model_params["tag"]}.png'
                ))
            plt.close()

            self.extraction_layer.evaluate_weights(
                train_inputs, 
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
        fig.savefig(os.path.join(
            f'{self.save_to_split_group}',
            f'{self.split_group}_abnn_example_{index}_{self.model_params["tag"]}.png'
            ))
        plt.close()

