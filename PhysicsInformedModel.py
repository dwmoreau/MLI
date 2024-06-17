import numpy as np
import tensorflow as tf

from Networks import mlp_model_builder
from TargetFunctions import IndexingTargetFunction
from Utilities import PairwiseDifferenceCalculator


class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.calibration_0 = self.add_weight(
            shape=(1,),
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.05, seed=None),
            name='calibration_0',
            dtype=tf.float32,
            )
        self.calibration_1 = self.add_weight(
            shape=(1,),
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            name='calibration_1',
            dtype=tf.float32,
            )

    def call(self, inputs):
        return self.calibration_0**2 * inputs + self.calibration_1**2 * inputs**2


class PhysicsInformedModel:
    def __init__(self, bravais_lattice, data_params, model_params, save_to, seed, q2_scaler, xnn_scaler, hkl_ref):
        self.bravais_lattice = bravais_lattice
        self.data_params = data_params
        self.model_params = model_params
        self.model_params['model_params']['n_outputs'] = len(self.data_params['y_indices'])
        self.n_points = data_params['n_points']
        self.n_outputs = data_params['n_outputs']
        self.y_indices = data_params['y_indices']
        self.save_to = save_to
        self.seed = seed

        self.q2_scaler = q2_scaler
        self.xnn_scaler = xnn_scaler
        self.hkl_ref = hkl_ref

    def setup(self):
        self.build_model()
        #self.model.summary()

    def train(self, data):
        train = data[data['train']]
        val = data[~data['train']]
        train_inputs = {'q2_scaled': np.stack(train['q2_scaled'])}
        val_inputs = {'q2_scaled': np.stack(val['q2_scaled'])}

        train_true = {
            'xnn_scaled': np.stack(train['reindexed_xnn_scaled'])[:, self.data_params['y_indices']],
            'hkl_softmax': np.stack(train['hkl_labels']),
            'indexing_data': np.stack(train['q2']),
            }
        val_true = {
            'xnn_scaled': np.stack(val['reindexed_xnn_scaled'])[:, self.data_params['y_indices']],
            'hkl_softmax': np.stack(val['hkl_labels']),
            'indexing_data': np.stack(val['q2']),
            }

        self.fit_history = [None, None, None]
        self.compile_model('regression')
        #self.model.summary()
        print('\nStarting warm-up: Unit cell regression')
        self.fit_history[0] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs_regression'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
        self.compile_model('assignment')
        #self.model.summary()
        print('\nStarting fitting: Miller index assignments calibration')
        self.fit_history[1] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs_assignment'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
        
        self.compile_model('index')
        #self.model.summary()
        print('\nStarting indexing: Unit cell & assignment calibration')
        self.fit_history[2] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs_index'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
        #self.plot_training_loss()

    def transform_pairwise_differences(self, pairwise_differences_scaled, tensorflow):
        if tensorflow:
            abs_func = tf.math.abs
        else:
            abs_func = np.abs
        epsilon = self.model_params['epsilon_pds']
        return epsilon / (abs_func(pairwise_differences_scaled) + epsilon)

    def build_model(self):
        inputs = {
            'q2_scaled': tf.keras.Input(
                shape=self.data_params['n_points'],
                name='q2_scaled',
                dtype=tf.float32,
                ),
            }
        self.pairwise_difference_calculator = PairwiseDifferenceCalculator(
            lattice_system=self.data_params['lattice_system'],
            hkl_ref=self.hkl_ref,
            tensorflow=True,
            q2_scaler=self.q2_scaler,
            )

        self.regression_layer_names = []
        for index in range(len(self.model_params['model_params']['layers'])):
            self.regression_layer_names.append(f'dense_xnn_scaled_{index}')
            self.regression_layer_names.append(f'layer_norm_xnn_scaled_' + str(index))
        self.regression_layer_names.append('xnn_scaled')

        self.calibration_layer_names = ['calibration_0', 'calibration_1']

        self.model = tf.keras.Model(inputs, self.model_builder(inputs))

    def model_builder(self, inputs):
        xnn_scaled = mlp_model_builder(
            inputs['q2_scaled'],
            'xnn_scaled',
            self.model_params['model_params'],
            'xnn_scaled'
            )
        xnn = xnn_scaled * self.xnn_scaler.scale_[0] + self.xnn_scaler.mean_[0]
        pairwise_differences_scaled, q2_ref = self.pairwise_difference_calculator.get_pairwise_differences(
            xnn, inputs['q2_scaled'], return_q2_ref=True
            )

        pairwise_differences_transformed = self.transform_pairwise_differences(
            pairwise_differences_scaled, True
            )

        hkl_logits = ScalingLayer()(pairwise_differences_transformed)

        # hkl_logits:               n_batch x n_points x hkl_ref_length
        # pairwise_differences:     n_batch x n_points x hkl_ref_length
        # q2_ref:                   n_batch x hkl_ref_length
        hkl_softmax = tf.keras.layers.Softmax(
            name='hkl_softmax',
            axis=2
            )(hkl_logits)
        
        indexing_data = tf.keras.layers.Concatenate(
            axis=1,
            name='indexing_data'
            )((
                hkl_softmax,
                q2_ref[:, tf.newaxis, :],
                ))
        return [xnn_scaled, hkl_softmax, indexing_data]

    def compile_model(self, mode):
        indexing_loss = IndexingTargetFunction(
            likelihood_function='cauchy', 
            error_fraction=np.linspace(0.001, 0.01, self.data_params['n_points']),
            n_points=self.data_params['n_points'],
            tuning_param=1
            )

        if mode == 'regression':
            optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_regression'])
            loss_weights = {
                'xnn_scaled': 1,
                'hkl_softmax': 0,
                'indexing_data': 0,
                }
            for layer_name in self.regression_layer_names:
                self.model.get_layer(layer_name).trainable = True
            self.model.get_layer('scaling_layer').trainable = False
            loss_metrics = {
                'xnn_scaled': 'mse',
                }
        elif mode == 'assignment':
            optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_assignment'])
            loss_weights = {
                'xnn_scaled': 0,
                'hkl_softmax': 1,
                'indexing_data': 0,
                }
            for layer_name in self.regression_layer_names:
                self.model.get_layer(layer_name).trainable = False
            self.model.get_layer('scaling_layer').trainable = True
            loss_metrics = {
                'indexing_data': indexing_loss,
                }
        elif mode == 'index':
            optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_index'])
            loss_weights = {
                'xnn_scaled': 1,
                'hkl_softmax': 0,
                'indexing_data': 1,
                }
            for layer_name in self.regression_layer_names:
                self.model.get_layer(layer_name).trainable = True
            self.model.get_layer('scaling_layer').trainable = True
            loss_metrics = {
                'xnn_scaled': 'mse',
                'hkl_softmax': 'accuracy',
                'indexing_data': indexing_loss,
                }
        loss_functions = {
            'xnn_scaled': tf.keras.losses.MSE,
            'hkl_softmax': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            'indexing_data': indexing_loss,
            }
        
        
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics
            )
