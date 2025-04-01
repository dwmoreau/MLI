import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special

from IOManagers import read_params
from IOManagers import write_params
from IOManagers import NeuralNetworkManager
from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import get_xnn_from_unit_cell
from Utilities import get_reciprocal_unit_cell_from_xnn
from Utilities import PairwiseDifferenceCalculator
from Utilities import reciprocal_uc_conversion
from Utilities import Q2Calculator
from Utilities import vectorized_resampling
from TargetFunctions import CandidateOptLoss


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
        self.hkl_ref = hkl_ref
        self.global_q2_scaler = q2_scaler
        self.global_xnn_scaler = xnn_scaler

    def setup(self, data):
        model_params_defaults = {
            'peak_length': 20,
            'extraction_peak_length': 6,
            'filter_length': 3,
            'n_volumes': 200,
            'n_filters': 200,
            'n_volumes_depth': [256,  64,  16],
            'n_filters_depth': [200, 200, 200],
            'depth_layers': [400, 200, 100],
            'initial_layers': [400, 200, 100],
            'final_layers': [200, 100, 50],
            'l1_regularization': 0.0,
            'base_line_layers': [200, 100, 50],
            'base_line_dropout_rate': 0.0,
            'learning_rate': 0.00005,
            'epochs': 20,
            'batch_size': 64,
            'loss_type': 'log_cosh',
            'augment': True,
            'model_type': 'metric',
            'calibration_params': {
                'layers': 3,
                'n_peaks': self.n_peaks,
                'epsilon_pds': 0.1,
                'epochs': 20,
                'learning_rate': 0.0001,
                'augment': True,
                'batch_size': 64,
                'l1_regularization': 0.0,
                },
            }

        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        for key in model_params_defaults['calibration_params'].keys():
            if key not in self.model_params['calibration_params'].keys():
                self.model_params['calibration_params'][key] = model_params_defaults['calibration_params'][key]
        self.model_params['unit_cell_length'] = self.unit_cell_length
        if self.model_params['model_type'] == 'deep':
            self.model_params['n_volumes'] = self.model_params['n_volumes_depth'][0]
            self.model_params['n_filters'] = self.model_params['n_filters_depth'][-1]
        self.build_model(data=data)
        self.build_calibration_model()

    def save(self, train_inputs):
        import keras
        write_params(
            self.model_params,
            os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_pitf_params_{self.model_params["tag"]}.csv'
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
            model_name=f'{self.split_group}_pitf_weights_{self.model_params["tag"]}',
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

    def save_calibration(self, train_inputs):
        import keras
        write_params(
            self.model_params,
            os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_pitf_params_{self.model_params["tag"]}.csv'
                )
            )
        self.calibration_model.save_weights(
            os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_calibration_weights_{self.model_params["tag"]}.weights.h5'
                )
            )
        model_manager = NeuralNetworkManager(
            model_name=f'{self.split_group}_calibration_weights_{self.model_params["tag"]}',
            save_dir=f'{self.save_to_split_group}',
            )
        model_manager.save_keras_weights(self.calibration_model)
        model_manager.convert_to_onnx(
            self.calibration_model,
            example_inputs=train_inputs,
            input_signature=(
                keras.Input(
                    shape=(self.data_params['n_peaks'],),
                    name='q2_obs_scaled',
                    dtype='float32',
                    ),
                keras.Input(
                    shape=(self.unit_cell_length,),
                    name='xnn',
                    dtype='float32',
                    ),
                )
            )
        model_manager.quantize_onnx(
            method='dynamic',
            calibration_data=train_inputs
            )

    def load_from_tag(self, mode):
        initial_params = self.model_params.copy()
        params = read_params(os.path.join(
            f'{self.save_to_split_group}',
            f'{self.split_group}_pitf_params_{self.model_params["tag"]}.csv'
            ))
        params_keys = [
            'tag',
            'peak_length',
            'extraction_peak_length',
            'filter_length',
            'n_volumes',
            'n_filters',
            'initial_layers',
            'final_layers',
            'l1_regularization',
            'base_line_layers',
            'base_line_dropout_rate',
            'learning_rate',
            'epochs',
            'batch_size',
            'loss_type',
            'augment',
            'model_type',
            'n_volumes_depth',
            'n_filters_depth',
            'depth_layers',
            ]
        self.model_params = dict.fromkeys(params_keys)
        assert mode in ['training', 'inference']
        self.model_params['mode'] = mode 
        self.model_params['tag'] = params['tag']
        self.model_params['peak_length'] = int(params['peak_length'])
        self.model_params['extraction_peak_length'] = int(params['extraction_peak_length'])
        self.model_params['filter_length'] = int(params['filter_length'])
        self.model_params['n_volumes'] = int(params['n_volumes'])
        self.model_params['n_filters'] = int(params['n_filters'])
        self.model_params['initial_layers'] = np.array(
            params['initial_layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['final_layers'] = np.array(
            params['final_layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['n_volumes_depth'] = np.array(
            params['n_volumes_depth'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['n_filters_depth'] = np.array(
            params['n_filters_depth'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['base_line_layers'] = np.array(
            params['base_line_layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['l1_regularization'] = float(params['l1_regularization'])
        self.model_params['base_line_dropout_rate'] = float(params['base_line_dropout_rate'])
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['epochs'] = int(params['epochs'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['loss_type'] = params['loss_type']
        if self.model_params['augment'] == 'True':
            self.model_params['augment'] = True
        else:
            self.model_params['augment'] = False
        self.model_params['model_type'] = params['model_type']

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
            model_name=f'{self.split_group}_pitf_weights_{self.model_params["tag"]}',
            save_dir=self.save_to_split_group,
            )
        if mode == 'training':
            self.build_model(data=None)
            self.compile_model()
            self.model = model_manager.load_keras_model(self.model)
        elif mode == 'inference':
            self.onnx_model = model_manager.load_onnx_model(quantized=True)

        calibration_params_keys = [
            'layers',
            'l1_regularization',
            'n_peaks',
            'epsilon_pds',
            'epochs',
            'learning_rate',
            'augment',
            'batch_size',
            ]
        self.model_params['calibration_params'] = dict.fromkeys(params_keys)
        self.model_params['calibration_params']['l1_regularization'] = 0.0
        for element in params['calibration_params'].split('{')[1].split('}')[0].split(", '"):
            key = element.replace("'", "").split(':')[0]
            value = element.replace("'", "").split(':')[1]
            if key in ['dropout_rate', 'epsilon_pds', 'learning_rate', 'l1_regularization']:
                self.model_params['calibration_params'][key] = float(value)
            elif key in ['n_components', 'n_peaks', 'epochs', 'batch_size', 'layers']:
                self.model_params['calibration_params'][key] = int(value)
            elif key == 'augment':
                if value == 'True':
                    self.model_params['calibration_params'][key] = True
                elif value == 'False':
                    self.model_params['calibration_params'][key] = False
        if self.model_params['model_type'] != 'base_line':
            model_manager = NeuralNetworkManager(
                model_name=f'{self.split_group}_calibration_weights_{self.model_params["tag"]}',
                save_dir=self.save_to_split_group,
                )
            if mode == 'training':
                self.build_calibration_model()
                self.compile_calibration_model()
                self.calibration_model = model_manager.load_keras_model(self.calibration_model)
            else:
                self.calibration_onnx_model = model_manager.load_onnx_model(quantized=True)

    def build_model(self, data=None):
        from Networks import ExtractionLayer
        import keras
        # Build the integral filter model #
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
        else:
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
        elif self.model_params['model_type'] == 'deep':
            self.model = keras.Model(inputs, self.model_builder_deep(inputs))
        self.compile_model()
        #self.model.summary()

    def build_calibration_model(self):
        import keras
        self.pairwise_difference_calculator = PairwiseDifferenceCalculator(
            lattice_system=self.data_params['lattice_system'],
            hkl_ref=self.hkl_ref,
            tensorflow=True,
            q2_scaler=self.global_q2_scaler,
            )
        self.pairwise_difference_calculation_numpy = PairwiseDifferenceCalculator(
            lattice_system=self.data_params['lattice_system'],
            hkl_ref=self.hkl_ref,
            tensorflow=False,
            q2_scaler=self.global_q2_scaler,
            )
        inputs = (
            keras.Input(
                shape=(self.data_params['n_peaks'],),
                name='q2_obs_scaled',
                dtype='float32',
                ),
            keras.Input(
                shape=(self.unit_cell_length,),
                name='xnn',
                dtype='float32',
                ),
            )
        self.calibration_model = keras.Model(inputs, self.model_builder_calibration(inputs))
        self.compile_calibration_model()
        #self.calibration_model.summary()

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
            x = keras.activations.gelu(x)
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
        # inputs: batch_size, n_peaks
        # metric: batch_size, n_volumes, n_filters
        ################################################
        # Initial Dense layers that predict amplitudes #
        ################################################
        x_initial = inputs
        for index in range(len(self.model_params['initial_layers'])):
            x_initial = keras.layers.Dense(
                self.model_params['initial_layers'][index],
                activation=keras.activations.gelu,
                name=f'initial_metric_dense_{index}',
                use_bias=False,
                kernel_initializer=keras.initializers.HeUniform
                )(x_initial)
        amplitude_logits = keras.layers.Dense(
            self.model_params['n_filters']*self.model_params['filter_length']*self.model_params['extraction_peak_length'],
            activation='linear',
            name='initial_amplitude_logits',
            use_bias=False,
            kernel_initializer=keras.initializers.HeUniform
            )(x_initial)
        amplitude_logits = keras.layers.Reshape(
            (
                1,
                self.model_params['n_filters'],
                self.model_params['filter_length'],
                self.model_params['extraction_peak_length']
                )
            )(amplitude_logits)

        #####################
        # Metric prediction #
        #####################
        x = self.extraction_layer(
            inputs[:, :self.model_params['extraction_peak_length']],
            amplitude_logits,
            name='extraction_layer'
            )

        #################
        # Hidden layers #
        #################
        for index in range(len(self.model_params['final_layers'])):
            x = keras.layers.Dense(
                self.model_params['final_layers'][index],
                activation=keras.activations.gelu,
                name=f'metric_dense_{index}',
                use_bias=False,
                kernel_regularizer=keras.regularizers.L1(
                    l1=self.model_params['l1_regularization']
                    ),
                kernel_initializer=keras.initializers.HeUniform
                )(x)

        # output: batch_size, n_volumes, unit_cell_length + 1
        output = keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name=f'{self.model_params["model_type"]}_xnn_scaled',
            kernel_initializer=keras.initializers.HeUniform
            )(x)
        return output

    def model_builder_deep(self, inputs):
        import keras
        # inputs: batch_size, n_peaks
        # metric: batch_size, n_volumes, n_filters
        ################################################
        # Initial Dense layers that predict amplitudes #
        ################################################
        x_initial = inputs
        for index in range(len(self.model_params['initial_layers'])):
            x_initial = keras.layers.Dense(
                self.model_params['initial_layers'][index],
                activation=keras.activations.gelu,
                name=f'initial_metric_dense_{index}',
                use_bias=False,
                kernel_initializer=keras.initializers.HeUniform
                )(x_initial)
        amplitude_logits = keras.layers.Dense(
            self.model_params['n_filters_depth'][0]*self.model_params['filter_length']*self.model_params['extraction_peak_length'],
            activation='linear',
            name='initial_amplitude_logits',
            use_bias=False,
            kernel_initializer=keras.initializers.HeUniform
            )(x_initial)
        amplitude_logits = keras.layers.Reshape(
            (
                1,
                self.model_params['n_filters_depth'][0],
                self.model_params['filter_length'],
                self.model_params['extraction_peak_length']
                )
            )(amplitude_logits)
        amplitude = keras.activations.softmax(
            amplitude_logits, axis=4
            )

        ############################
        # Metric prediction layers #
        ############################
        # unweighted_metric: batch_size, n_volumes, n_filters, filter_length, extraction_peak_length
        # metric:            batch_size, n_volumes, n_filters
        unweighted_metric = self.extraction_layer(
            inputs[:, :self.model_params['extraction_peak_length']],
            )
        metric = keras.ops.sum(
            amplitude * unweighted_metric[:, :, :self.model_params['n_filters_depth'][0], :, :],
            axis=(3, 4)
            )
        for depth_index in range(1, len(self.model_params['n_volumes_depth'])):
            volume_rankings = keras.ops.sum(metric, axis=2)
            volume_sort_indices = tf.argsort(volume_rankings, axis=1, direction='DESCENDING')
            if depth_index == 1:
                volume_indices = volume_sort_indices[:, :self.model_params['n_volumes_depth'][depth_index]]
            else:
                volume_indices = tf.gather(
                    params=volume_indices,
                    indices=volume_sort_indices[:, :self.model_params['n_volumes_depth'][depth_index]],
                    axis=1,
                    batch_dims=1
                    )
            metric = tf.gather(
                params=metric,
                indices=volume_sort_indices[:, :self.model_params['n_volumes_depth'][depth_index]],
                axis=1,
                batch_dims=1,
                )
            # unweighted_metric: bs, n_volumes, n_filters, filter_length, peak_extraction_length
            unweighted_metric = tf.gather(
                params=unweighted_metric,
                indices=volume_sort_indices[:, :self.model_params['n_volumes_depth'][depth_index]],
                axis=1,
                batch_dims=1,
                )
            x_depth = metric
            for index in range(len(self.model_params['depth_layers'])):
                x_depth = keras.layers.Dense(
                    self.model_params['depth_layers'][index],
                    activation=keras.activations.gelu,
                    name=f'depth_metric_dense_{depth_index}_{index}',
                    use_bias=False,
                    kernel_initializer=keras.initializers.HeUniform
                    )(x_depth)
            amplitude_logits = keras.layers.Dense(
                self.model_params['n_filters_depth'][depth_index]*self.model_params['filter_length']*self.model_params['extraction_peak_length'],
                activation='linear',
                name=f'depth_amplitude_logits_{depth_index}',
                use_bias=False,
                kernel_initializer=keras.initializers.HeUniform,
                kernel_regularizer=keras.regularizers.L1(
                    l1=self.model_params['l1_regularization']
                    ),
                )(x_depth)
            amplitude_logits = keras.layers.Reshape(
                (
                    self.model_params['n_volumes_depth'][depth_index],
                    self.model_params['n_filters_depth'][depth_index],
                    self.model_params['filter_length'],
                    self.model_params['extraction_peak_length']
                    )
                )(amplitude_logits)
            amplitude = keras.activations.softmax(
                amplitude_logits, axis=4
                )
            metric = keras.ops.sum(
                amplitude * unweighted_metric[:, :, :self.model_params['n_filters_depth'][depth_index], :, :],
                axis=(3, 4)
                )
        # x_depth:         bs, n_volumes, arb_units
        # volumes_indices: bs, n_volumes
        if len(self.model_params['n_volumes_depth']) == 1:
            volume_rankings = keras.ops.sum(metric, axis=2)
            volume_sort_indices = tf.argsort(volume_rankings, axis=1, direction='DESCENDING')
            volume_indices = volume_sort_indices
            x = tf.gather(
                params=metric,
                indices=volume_sort_indices,
                axis=1,
                batch_dims=1,
                )
        elif len(self.model_params['n_volumes_depth']) > 1:
            volume_embeddings = keras.layers.Embedding(
                input_dim=self.model_params['n_volumes_depth'][0],
                output_dim=1,
                embeddings_initializer="uniform",
                embeddings_regularizer=None,
                input_length=None,
                sparse=False,
                )(volume_indices)
            #volume_embeddings = (volume_indices/self.model_params['n_volumes_depth'][0])[:, :, tf.newaxis]
            x = keras.layers.Concatenate(axis=2)((metric, volume_embeddings))

        ################
        # Hiden layers #
        ################
        for index in range(len(self.model_params['final_layers'])):
            x = keras.layers.Dense(
                self.model_params['final_layers'][index],
                activation=keras.activations.gelu,
                name=f'metric_dense_{index}',
                use_bias=False,
                kernel_regularizer=keras.regularizers.L1(
                    l1=self.model_params['l1_regularization']
                    ),
                kernel_initializer=keras.initializers.HeUniform
                )(x)
        # output: batch_size, n_volumes, unit_cell_length + 1
        output = keras.layers.Dense(
            self.unit_cell_length + 1,
            activation='linear',
            name=f'{self.model_params["model_type"]}_xnn_scaled',
            kernel_initializer=keras.initializers.HeUniform
            )(x)
        return output

    def transform_pairwise_differences(self, pairwise_differences_scaled, tensorflow):
        if tensorflow:
            import keras
            abs_func = keras.ops.absolute
        else:
            abs_func = np.abs
        epsilon = self.model_params['calibration_params']['epsilon_pds']
        return epsilon / (abs_func(pairwise_differences_scaled) + epsilon)

    def model_builder_calibration(self, inputs):
        import keras
        pairwise_differences_scaled, q2_ref = self.pairwise_difference_calculator.get_pairwise_differences(
            inputs[1], inputs[0], return_q2_ref=True
            )

        # hkl_logits:               n_batch x n_peaks x hkl_ref_length
        # pairwise_differences:     n_batch x n_peaks x hkl_ref_length
        # q2_ref:                   n_batch x hkl_ref_length
        pairwise_differences_transformed = self.transform_pairwise_differences(
            pairwise_differences_scaled, True
            )
        for index in range(self.model_params['calibration_params']['layers']):
            dense_layer = keras.layers.Dense(
                self.hkl_ref.shape[0],
                activation=keras.activations.gelu,
                name=f'dense_{index}',
                use_bias=False,
                kernel_initializer=keras.initializers.HeUniform,
                kernel_regularizer=keras.regularizers.L1(
                    l1=self.model_params['calibration_params']['l1_regularization']
                    ),
                )
            if index == 0:
                x = dense_layer(pairwise_differences_transformed)
            else:
                x = dense_layer(x)

        hkl_logits = keras.layers.Dense(
            self.hkl_ref.shape[0],
            activation='linear',
            name=f'hkl_logits',
            use_bias=False,
            )(x)

        hkl_softmax = keras.layers.Softmax(
            name='hkl_softmax',
            axis=2
            )(hkl_logits)
        return hkl_softmax

    def compile_model(self):
        import keras
        # Create learning rate scheduler
        optimizer = keras.optimizers.Adam(
            learning_rate=self.model_params['learning_rate'],
            )
        if self.model_params['loss_type'] == 'mse':
            loss_functions = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.loss_function_mse
                }
        else:
            loss_functions = {
                f'{self.model_params["model_type"]}_xnn_scaled': self.extraction_layer.loss_function_log_cosh
                }
        loss_metrics = {
            f'{self.model_params["model_type"]}_xnn_scaled': [
                self.extraction_layer.loss_function_log_cosh,
                self.extraction_layer.loss_function_mse
                ]
            }
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            metrics=loss_metrics,
            run_eagerly=False,
            )

    def compile_calibration_model(self):
        import keras
        optimizer = keras.optimizers.Adam(self.model_params['calibration_params']['learning_rate'])
        loss_metrics = {
            'hkl_softmax': 'accuracy',
            }
        loss_functions = {
            'hkl_softmax': keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            }
        self.calibration_model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            metrics=loss_metrics
            )

    def train(self, data):
        import keras
        from Networks import SigmaDecayCallback
        if self.model_params['augment'] == False:
            data = data[~data['augmented']]
        train = data[data['train']]
        val = data[~data['train']]

        #if self.lattice_system in ['triclinic', 'monoclinic']:
        #    # This helps with overflow error in the loss function
        #    train_xnn = np.stack(train['reindexed_xnn'])[:, self.unit_cell_indices]
        #    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        #        train_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
        #        )
        #    reciprocal_volume = get_unit_cell_volume(
        #        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
        #        )
        #    max_reciprocal_volume = np.sort(reciprocal_volume)[int(0.995*reciprocal_volume.size)]
        #    train = train[reciprocal_volume < max_reciprocal_volume]

        train_q2_obs = np.stack(train['q2'])[:, :self.model_params['peak_length']]
        val_q2_obs = np.stack(val['q2'])[:, :self.model_params['peak_length']]
        train_q2_obs_scaled = train_q2_obs / self.q2_obs_scale
        val_q2_obs_scaled = val_q2_obs / self.q2_obs_scale

        train_xnn = np.stack(train['reindexed_xnn'])[:, self.unit_cell_indices]
        val_xnn = np.stack(val['reindexed_xnn'])[:, self.unit_cell_indices]

        if self.lattice_system == 'rhombohedral':
            # There are very large values of xnn being included for rhombohedral.
            # These need to be excluded or NaNs will occur in the model.
            train_indices = np.max(np.abs(train_xnn), axis=1) < 0.05
            val_indices = np.max(np.abs(val_xnn), axis=1) < 0.05
            train_xnn = train_xnn[train_indices]
            train_q2_obs_scaled = train_q2_obs_scaled[train_indices]
            val_xnn = val_xnn[val_indices]
            val_q2_obs_scaled = val_q2_obs_scaled[val_indices]
            train_unaugmented = np.invert(train['augmented'][train_indices])
        else:
            train_unaugmented = np.invert(train['augmented'])

        self.xnn_mean = np.median(train_xnn, axis=0)[np.newaxis]
        self.xnn_scale = np.median(np.abs(train_xnn - self.xnn_mean), axis=0)[np.newaxis]
        train_xnn_scaled = (train_xnn - self.xnn_mean) / self.xnn_scale
        val_xnn_scaled = (val_xnn - self.xnn_mean) / self.xnn_scale

        fig, axes = plt.subplots(1, self.unit_cell_length + 1, figsize=(8, 3))
        bins0 = np.linspace(0, 5, 301)
        bins1 = np.linspace(-5, 5, 301)
        xnn_titles = ['Xhh', 'Xkk', 'Xll', 'Xkl', 'Xhl', 'Xhk']
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
            f'{self.split_group}_pitf_io_{self.model_params["tag"]}.png'
            ))
        plt.close()

        train_true = train_xnn_scaled
        val_true = val_xnn_scaled
        train_inputs = train_q2_obs_scaled
        val_inputs = val_q2_obs_scaled

        if self.model_params['model_type'] != 'base_line':
            self.extraction_layer.evaluate_init(
                train_q2_obs_scaled[train_unaugmented],
                self.save_to_split_group,
                self.split_group,
                self.model_params["tag"]
                )

            callbacks = [SigmaDecayCallback(
                extraction_layer=self.extraction_layer,
                decay_rate=0.8,
                initial_multiplier=2
                )]
        else:
            callbacks = None
        self.fit_history = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=callbacks,
            )
        self.save(train_inputs)

        ##############################
        # Plot training loss vs time #
        ##############################
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        axes[0].plot(
            self.fit_history.history['loss'], 
            label='Training', marker='.'
            )
        axes[0].plot(
            self.fit_history.history['val_loss'], 
            label='Validation', marker='v'
            )
        axes[1].plot(
            self.fit_history.history['loss_function_log_cosh'], 
            label='Training', marker='.'
            )
        axes[1].plot(
            self.fit_history.history['val_loss_function_log_cosh'], 
            label='Validation', marker='v'
            )
        axes[2].plot(
            self.fit_history.history['loss_function_mse'], 
            label='Training', marker='.'
            )
        axes[2].plot(
            self.fit_history.history['val_loss_function_mse'], 
            label='Validation', marker='v'
            )
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Log-Cosh Error')
        axes[2].set_ylabel('MSE Error')
        axes[2].set_xlabel('Epoch')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to_split_group}',
            f'{self.split_group}_pitf_training_loss_{self.model_params["tag"]}.png'
            ))
        plt.close()

        if self.model_params['model_type'] != 'base_line':
            self.extraction_layer.evaluate_weights(
                train_inputs, 
                self.save_to_split_group,
                self.split_group,
                self.model_params["tag"]
                )

    def train_calibration(self, data):
        import keras
        if self.model_params['model_type'] == 'base_line':
            return None
        if self.model_params['calibration_params']['augment'] == False:
            data = data[~data['augmented']]
        train = data[data['train']]
        val = data[~data['train']]

        # Get predictions
        val_q2_obs = np.stack(val['q2'])[:, :self.model_params['peak_length']]
        val_q2_obs_scaled = val_q2_obs / self.q2_obs_scale
        val_inputs = val_q2_obs_scaled
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

        train_q2_obs = np.stack(train['q2'])[:, :self.model_params['peak_length']]
        train_q2_obs_scaled = train_q2_obs / self.q2_obs_scale
        train_inputs = train_q2_obs_scaled
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

        train_inputs_calibration = (
            np.stack(train['q2_scaled']),
            train_xnn_pred_top5[:, 0],
            )
        val_inputs_calibration = (
            np.stack(val['q2_scaled']),
            val_xnn_pred_top5[:, 0],
            )
        train_true_calibration = np.stack(train['hkl_labels'])
        val_true_calibration = np.stack(val['hkl_labels'])
        self.calibration_fit_history = self.calibration_model.fit(
            x=train_inputs_calibration,
            y=train_true_calibration,
            epochs=self.model_params['calibration_params']['epochs'],
            shuffle=True,
            batch_size=self.model_params['calibration_params']['batch_size'], 
            validation_data=(val_inputs_calibration, val_true_calibration),
            callbacks=None,
            )
        self.save_calibration(train_inputs_calibration)

        fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        axes[0].plot(self.calibration_fit_history.history['loss'], label='Training', marker='.')
        axes[0].plot(self.calibration_fit_history.history['val_loss'], label='Validation', marker='.')
        axes[1].plot(self.calibration_fit_history.history['accuracy'], label='Training', marker='.')
        axes[1].plot(self.calibration_fit_history.history['val_accuracy'], label='Validation', marker='.')
        axes[1].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Accuracy')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to_split_group}',
            f'{self.split_group}_calibration_training_loss_{self.model_params["tag"]}.png'
            ))
        plt.close()

    def predict_xnn(self, top_n, data=None, inputs=None, q2_obs=None, batch_size=None):
        if not data is None:
            q2_obs = np.stack(data['q2'])[:, :self.model_params['peak_length']]
        elif not inputs is None:
            q2_obs = np.array(inputs['q2'])[:self.model_params['peak_length']]
        elif not q2_obs is None:
            print
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
                rng=self.rng
                )
        return xnn_pred_top_n, softmax_pred_top_n

    def predict_hkl(self, q2_obs, xnn, batch_size=None):
        q2_obs_scaled = (q2_obs - self.global_q2_scaler.mean_[0]) / self.global_q2_scaler.scale_[0]

        #print(f'\n Regression inferences for {self.split_group}')
        if batch_size is None:
            batch_size = self.model_params['batch_size']

        # predict_on_batch helps with a memory leak...
        N = q2_obs_scaled.shape[0]
        hkl_softmax = np.zeros((N, self.data_params['n_peaks'], self.hkl_ref.shape[0]))
        if self.model_params['mode'] == 'inference':
            q2_obs_scaled_f32 = q2_obs_scaled.astype(np.float32)
            xnn_f32 = xnn.astype(np.float32)
            for pred_index in range(xnn.shape[0]):
                inputs = {
                    'input_0': q2_obs_scaled_f32[pred_index][np.newaxis],
                    'input_1': xnn_f32[pred_index][np.newaxis]
                    }
                hkl_softmax[pred_index] = self.calibration_onnx_model.run(None, inputs)[0]
        elif self.model_params['mode'] == 'training':
            n_batches = N // batch_size
            left_over = N % batch_size

            for batch_index in range(n_batches + 1):
                start = batch_index * batch_size
                if batch_index == n_batches:
                    batch_inputs = (
                        np.zeros((batch_size, self.data_params['n_peaks'])),
                        np.zeros((batch_size, self.unit_cell_length))
                        )
                    batch_inputs[0][:left_over] = q2_obs_scaled[start: start + left_over]
                    batch_inputs[0][left_over:] = q2_obs_scaled[0]
                    batch_inputs[1][:left_over] = xnn[start: start + left_over]
                    batch_inputs[1][left_over:] = xnn[0]
                else:
                    batch_inputs = (
                        q2_obs_scaled[start: start + batch_size],
                        xnn[start: start + batch_size]
                        )

                outputs = self.calibration_model.predict_on_batch(batch_inputs)
                if batch_index == n_batches:
                    hkl_softmax[start:] = outputs[:left_over]
                else:
                    hkl_softmax[start: start + batch_size] = outputs
        return hkl_softmax

    def generate(self, n_unit_cells, rng, q2_obs, top_n=5, batch_size=None):
        n_unit_cells_per_pred = n_unit_cells // top_n
        n_extra = n_unit_cells % top_n
        xnn_gen = np.zeros((n_unit_cells, self.unit_cell_length))

        # If top_n == 5, then self.predict_xnn generates 5 unit cells
        # xnn_pred: 1, top_n, unit_cell_length
        xnn_pred, _ = self.predict_xnn(top_n, q2_obs=q2_obs[np.newaxis], batch_size=batch_size)
        xnn_gen[:top_n] = xnn_pred[0]

        # Resampling needs to generate n_unit_cells_per_pred - 1 unit cells from each prediction
        # hkl_softmax: top_n, n_peaks, hkl_ref_length
        hkl_softmax = self.predict_hkl(
            np.repeat(q2_obs[np.newaxis], repeats=top_n, axis=0),
            xnn_pred[0],
            batch_size=batch_size
            )
        hkl_assign = np.zeros((n_unit_cells, self.data_params['n_peaks']), dtype=int)
        hkl_assign[:top_n] = np.argmax(hkl_softmax, axis=2)
        start = top_n
        for gen_index in range(n_unit_cells_per_pred - 1):
            # This generates top_n unit cells per iteration
            hkl_assign[start: start + top_n], _ = vectorized_resampling(hkl_softmax, rng)
            xnn_gen[start: start + top_n] = xnn_pred[0]
            start += top_n
        hkl_assign[start: start + n_extra], _ = vectorized_resampling(hkl_softmax[:n_extra], rng)
        hkl = np.take_along_axis(self.hkl_ref[:, np.newaxis, :], hkl_assign[:, :, np.newaxis], axis=0)

        # hkl: n_unit_cells, n_peaks, 3
        target_function = CandidateOptLoss(
            np.repeat(q2_obs[np.newaxis], repeats=n_unit_cells, axis=0), 
            lattice_system=self.lattice_system,
            )
        target_function.update(hkl, xnn_gen)
        xnn_gen += target_function.gauss_newton_step(xnn_gen)
        xnn_gen = fix_unphysical(xnn=xnn_gen, rng=self.rng, lattice_system=self.lattice_system)
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

        if self.model_params['model_type'] != 'base_line':
            self.evaluate_indexing(
                train, val, train_xnn_pred_top5[:, 0, :], val_xnn_pred_top5[:, 0, :], quantitized_model
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
            
            unit_cell_titles = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
            xnn_titles = ['Xhh', 'Xkk', 'Xll', 'Xkl', 'Xhl', 'Xhk']
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
                    f'{self.split_group}_pitf_reg_eval_{self.model_params["tag"]}_most_probable.csv'
                    )
                )

        ##############################
        # Plot unit cell evaluations #
        ##############################
        figsize = (self.unit_cell_length*2 + 2, 6)
        fig, axes = plt.subplots(2, self.unit_cell_length, figsize=figsize)
        if self.unit_cell_length == 1:
            axes = axes[:, np.newaxis]
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
            if quantitized_model:
                fig.savefig(os.path.join(
                    f'{self.save_to_split_group}',
                    f'{self.split_group}_pitf_reg_eval_optimized_{self.model_params["tag"]}_{save_label}.png'
                    ))
            else:
                fig.savefig(os.path.join(
                    f'{self.save_to_split_group}',
                    f'{self.split_group}_pitf_reg_eval_{self.model_params["tag"]}_{save_label}.png'
                    ))
            plt.close()

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
                f'{self.split_group}_pitf_branch_importance_{self.model_params["tag"]}.png'
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
            f'{self.split_group}_pitf_example_{index}_{self.model_params["tag"]}.png'
            ))
        plt.close()

    def evaluate_indexing(self, train, val, train_xnn, val_xnn, quantitized_model):
        hkl_labels_true_train = np.stack(train['hkl_labels'])
        hkl_labels_true_val = np.stack(val['hkl_labels'])

        train_q2_obs_scaled = np.stack(train['q2_scaled'])
        val_q2_obs_scaled = np.stack(val['q2_scaled'])
        train_inputs_calibration = (
            train_q2_obs_scaled,
            train_xnn,
            )
        val_inputs_calibration = (
            val_q2_obs_scaled,
            val_xnn,
            )

        if quantitized_model:
            hkl_softmax_train = np.zeros((
                train_q2_obs_scaled.shape[0],
                self.data_params['n_peaks'],
                self.hkl_ref.shape[0]
                ))
            hkl_softmax_val = np.zeros((
                val_q2_obs_scaled.shape[0],
                self.data_params['n_peaks'],
                self.hkl_ref.shape[0]
                ))
            for pred_index in range(train_q2_obs_scaled.shape[0]):
                inputs = {
                    'input_0': train_inputs_calibration[0][pred_index].astype(np.float32)[np.newaxis],
                    'input_1': train_inputs_calibration[1][pred_index].astype(np.float32)[np.newaxis]
                    }
                hkl_softmax_train[pred_index] = self.calibration_onnx_model.run(None, inputs)[0]
            for pred_index in range(val_q2_obs_scaled.shape[0]):
                inputs = {
                    'input_0': val_inputs_calibration[0][pred_index].astype(np.float32)[np.newaxis],
                    'input_1': val_inputs_calibration[1][pred_index].astype(np.float32)[np.newaxis]
                    }
                hkl_softmax_val[pred_index] = self.calibration_onnx_model.run(None, inputs)[0]
        else:
            hkl_softmax_train = self.calibration_model.predict(train_inputs_calibration)
            hkl_softmax_val = self.calibration_model.predict(val_inputs_calibration)
        hkl_labels_pred_train = np.argmax(hkl_softmax_train, axis=2)
        hkl_labels_pred_val = np.argmax(hkl_softmax_val, axis=2)

        # correct shape: n_entries, n_peaks
        correct_pred_train = hkl_labels_true_train == hkl_labels_pred_train
        correct_pred_val = hkl_labels_true_val == hkl_labels_pred_val
        accuracy_pred_train = correct_pred_train.sum() / correct_pred_train.size
        accuracy_pred_val = correct_pred_val.sum() / correct_pred_val.size
        # accuracy for each entry
        accuracy_entry_train = correct_pred_train.sum(axis=1) / self.n_peaks
        accuracy_entry_val = correct_pred_val.sum(axis=1) / self.n_peaks
        # accuracy per peak position
        accuracy_peak_position_train = correct_pred_train.sum(axis=0) / correct_pred_train.shape[0]
        accuracy_peak_position_val = correct_pred_val.sum(axis=0) / correct_pred_val.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        bins = (np.arange(self.n_peaks + 2) - 0.5) / self.n_peaks
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        hist_train, _ = np.histogram(accuracy_entry_train, bins=bins, density=True)
        hist_val, _ = np.histogram(accuracy_entry_val, bins=bins, density=True)
        axes[0].bar(centers, hist_train, width=dbin, label='Predicted: Training')
        axes[0].bar(centers, hist_val, width=dbin, alpha=0.5, label='Predicted: Validation')
        axes[1].bar(
            np.arange(self.n_peaks), accuracy_peak_position_train,
            width=1, label='Predicted: Training'
            )
        axes[1].bar(
            np.arange(self.n_peaks), accuracy_peak_position_val,
            width=1, alpha=0.5, label='Predicted: Validation'
            )

        axes[1].legend(frameon=False)
        axes[0].set_title(f'Predicted accuracy: {accuracy_pred_train:0.3f}/{accuracy_pred_val:0.3f}')

        axes[0].set_xlabel('Accuracy')
        axes[1].set_xlabel('Peak Position')
        axes[0].set_ylabel('Entry Accuracy')
        axes[1].set_ylabel('Peak Accuracy')
        axes[1].set_ylim([0, 1])
        fig.tight_layout()
        if quantitized_model:
            fig.savefig(os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_calibration_accuracy_quantitized_{self.model_params["tag"]}.png'
                ))
        else:
            fig.savefig(os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_calibration_accuracy_{self.model_params["tag"]}.png'
                ))
        plt.close()    

        def calibration_plots(hkl_labels_true, hkl_softmax, n_peaks, n_bins=25):
            N = hkl_softmax.shape[0]
            hkl_labels_pred = hkl_softmax.argmax(axis=2)
            p_pred = np.zeros((N, n_peaks))
            metrics = np.zeros((n_bins, 4))
            ece = 0
            for entry_index in range(N):
                for point_index in range(n_peaks):
                    p_pred[entry_index, point_index] = hkl_softmax[
                        entry_index,
                        point_index,
                        hkl_labels_pred[entry_index, point_index]
                        ]

            bins = np.linspace(p_pred.min(), p_pred.max(), n_bins + 1)
            centers = (bins[1:] + bins[:-1]) / 2
            metrics[:, 0] = centers
            for bin_index in range(n_bins):
                indices = np.logical_and(
                    p_pred >= bins[bin_index],
                    p_pred < bins[bin_index + 1],
                    )
                if np.sum(indices) > 0:
                    p_pred_bin = p_pred[indices]
                    hkl_labels_pred_bin = hkl_labels_pred[indices]
                    hkl_labels_true_bin = hkl_labels_true[indices]
                    metrics[bin_index, 1] = np.sum(hkl_labels_pred_bin == hkl_labels_true_bin) / hkl_labels_true_bin.size
                    metrics[bin_index, 2] = p_pred_bin.mean()
                    metrics[bin_index, 3] = p_pred_bin.std()
                    prefactor = indices.sum() / indices.size
                    ece += prefactor * np.abs(metrics[bin_index, 2] - metrics[bin_index, 1])
            return metrics, ece

        metrics_train, ece_train = calibration_plots(hkl_labels_true_train, hkl_softmax_train, self.n_peaks)
        metrics_val, ece_val = calibration_plots(hkl_labels_true_val, hkl_softmax_val, self.n_peaks)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for i in range(2):
            axes[i].plot([0, 1], [0, 1], linestyle='dotted', color=[0, 0, 0])
            axes[i].set_xlabel('Confidence')
        axes[0].errorbar(metrics_train[:, 2], metrics_train[:, 1], yerr=metrics_train[:, 3], marker='.')
        axes[0].set_title(f'Expected Confidence Error: {ece_train:0.4f}')
        axes[0].set_ylabel('Accuracy')
        fig.tight_layout()
        if quantitized_model:
            fig.savefig(os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_calibration_cal_quantitized_{self.model_params["tag"]}.png'
                ))
        else:
            fig.savefig(os.path.join(
                f'{self.save_to_split_group}',
                f'{self.split_group}_calibration_cal_{self.model_params["tag"]}.png'
                ))
        plt.close()
