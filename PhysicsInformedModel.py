import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special
from sklearn.preprocessing import StandardScaler
import shutil
import tensorflow as tf

from Networks import mlp_model_builder
from TargetFunctions import IndexingTargetFunction
from TargetFunctions import LikelihoodLoss
from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import PairwiseDifferenceCalculator
from Utilities import vectorized_resampling
from Utilities import read_params
from Utilities import write_params


def scaling_model_builder(x_in, tag, model_params):
    # x_in: batch_size x n_peaks x hkl_ref_length
    for index in range(len(model_params['layers'])):
        if index == 0:
            x = tf.keras.layers.Dense(
                model_params['layers'][index],
                activation='linear',
                name=f'dense_{tag}_{index}',
                use_bias=False,
                )(x_in)
        else:
            x = tf.keras.layers.Dense(
                model_params['layers'][index],
                activation='linear',
                name=f'dense_{tag}_{index}',
                use_bias=False,
                )(x)
        x = tf.keras.layers.LayerNormalization(
            name=f'layer_norm_{tag}_{index}',
            )(x)
        x = tf.keras.activations.gelu(x)
        x = tf.keras.layers.Dropout(
            rate=model_params['dropout_rate'],
            name=f'dropout_{tag}_{index}'
            )(x)

    # x: batch_size x n_peaks x units
    # scaler: batch_size x n_peaks x n_components
    scaler = tf.keras.layers.Concatenate(
        axis=1,
        name='scaler'
        )([
        tf.keras.layers.Dense(
            2*model_params['n_components'] - 2,
            activation='linear',
            name=f'calibration_scaler_{index}',
            use_bias=False,
            )(x[:, index, :])[:, tf.newaxis, :] for index in range(model_params['n_peaks'])
        ])

    hkl_logits = scaler[:, :, 0][:, :, tf.newaxis] * tf.ones_like(x_in)
    hkl_logits = tf.keras.layers.Add()([
        hkl_logits,
        scaler[:, :, 1][:, :, tf.newaxis] * x_in
        ])
    for power in range(2, model_params['n_components']):
        hkl_logits = tf.keras.layers.Add()([
            hkl_logits,
            scaler[:, :, power][:, :, tf.newaxis] * x_in**power
            ])
        hkl_logits = tf.keras.layers.Add()([
            hkl_logits,
            scaler[:, :, model_params['n_components'] + power - 2][:, :, tf.newaxis] * x_in**(1/power)
            ])
    return hkl_logits


class PhysicsInformedModel:
    def __init__(self, split_group, data_params, model_params, save_to, seed, q2_scaler, xnn_scaler, hkl_ref):
        self.split_group = split_group
        self.data_params = data_params
        self.model_params = model_params
        if not 'xnn_params' in self.model_params.keys():
            self.model_params['xnn_params'] = {}
        self.model_params['xnn_params']['unit_cell_length'] = len(self.data_params['unit_cell_indices'])
        self.n_peaks = data_params['n_peaks']
        self.unit_cell_length = data_params['unit_cell_length']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.save_to = save_to
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.q2_scaler = q2_scaler
        self.xnn_scaler = xnn_scaler
        self.hkl_ref = hkl_ref

    def setup(self, data):
        train = data[data['train']]
        train_rec_volume = 1 / np.array(train['reindexed_volume'])
        self.volume_scaler = StandardScaler()
        self.volume_scaler.fit(train_rec_volume[:, np.newaxis])

        model_params_defaults = {
            'xnn_params': {
                'layers': [2000, 1000, 1000],
                'dropout_rate': 0.05,
                'epsilon': 0.001,
                'output_activation': 'linear',
                },
            'volume_params': {
                'layers': [100, 100, 100],
                'dropout_rate': 0.05,
                'epsilon': 0.001,
                'unit_cell_length': 1,
                'output_activation': 'linear',
                },
            'assign_params': {
                'layers': [1000, 500, 100],
                'dropout_rate': 0.25,
                'n_components': 5,
                'n_peaks': self.n_peaks,
                },
            'epsilon_pds': 0.1,
            'learning_rate_volume': 0.001,
            'learning_rate_xnn': 0.001,
            'learning_rate_assign': 0.001,
            'learning_rate_full': 0.000001,
            'epochs_volume': 10,
            'epochs_xnn': 10,
            'epochs_assign': 10,
            'epochs_full': 10,
            'batch_size': 128,
            }

        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        self.model_params['xnn_params']['unit_cell_length'] = self.unit_cell_length

        for key in model_params_defaults['xnn_params'].keys():
            if key not in self.model_params['xnn_params'].keys():
                self.model_params['mean_params'][key] = model_params_defaults['xnn_params'][key]
        self.model_params['xnn_params']['kernel_initializer'] = None
        self.model_params['xnn_params']['bias_initializer'] = None

        for key in model_params_defaults['volume_params'].keys():
            if key not in self.model_params['volume_params'].keys():
                self.model_params['volume_params'][key] = model_params_defaults['volume_params'][key]
        self.model_params['volume_params']['kernel_initializer'] = None
        self.model_params['volume_params']['bias_initializer'] = None

        for key in model_params_defaults['assign_params'].keys():
            if key not in self.model_params['assign_params'].keys():
                self.model_params['assign_params'][key] = model_params_defaults['assign_params'][key]

        self.build_model()
        #self.model.summary()

    def save(self):
        write_params(self.model_params, f'{self.save_to}/{self.split_group}_pitf_params_{self.model_params["tag"]}.csv')
        self.model.save_weights(f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5')

    def load_from_tag(self):
        params = read_params(f'{self.save_to}/{self.split_group}_pitf_params_{self.model_params["tag"]}.csv')
        params_keys = [
            'tag',
            'epsilon_pds',
            'learning_rate_regression',
            'learning_rate_assignment',
            'learning_rate_index',
            'cycles_regression',
            'epochs_regression',
            'epochs_assignment',
            'epochs_index',
            'batch_size',
            'beta_nll',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['beta_nll'] = float(params['beta_nll'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['learning_rate_regression'] = float(params['learning_rate_regression'])
        self.model_params['learning_rate_assignment'] = float(params['learning_rate_assignment'])
        self.model_params['learning_rate_index'] = float(params['learning_rate_index'])
        self.model_params['cycles_regression'] = int(params['cycles_regression'])
        self.model_params['epochs_regression'] = int(params['epochs_regression'])
        self.model_params['epochs_assignment'] = int(params['epochs_assignment'])
        self.model_params['epochs_index'] = int(params['epochs_index'])
        self.model_params['epsilon_pds'] = float(params['epsilon_pds'])

        params_keys = [
            'dropout_rate',
            'epsilon',
            'layers',
            'output_activation',
            'output_name',
            'unit_cell_length',
            'kernel_initializer',
            'bias_initializer',
            ]
        network_keys = ['mean_params', 'var_params', 'calibration_params']
        for network_key in network_keys:
            self.model_params[network_key] = dict.fromkeys(params_keys)
            self.model_params[network_key]['unit_cell_length'] = self.unit_cell_length
            for element in params[network_key].split('{')[1].split('}')[0].split(", '"):
                key = element.replace("'", "").split(':')[0]
                value = element.replace("'", "").split(':')[1]
                if key in ['dropout_rate', 'epsilon']:
                    self.model_params[network_key][key] = float(value)
                elif key == 'layers':
                    self.model_params[network_key]['layers'] = np.array(
                        value.split('[')[1].split(']')[0].split(','),
                        dtype=int
                        )
                elif key == 'n_components':
                    self.model_params[network_key][key] = int(value)

            self.model_params[network_key]['kernel_initializer'] = None
            self.model_params[network_key]['bias_initializer'] = None

        self.model_params['calibration_params']['n_peaks'] = self.n_peaks

        self.build_model()
        self.compile_model()
        self.mean_model.load_weights(
            filepath=f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5',
            by_name=True, skip_mismatch=True
            )
        self.var_model.load_weights(
            filepath=f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5',
            by_name=True, skip_mismatch=True
            )
        self.assign_model.load_weights(
            filepath=f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5',
            by_name=True, skip_mismatch=True
            )
        self.model.load_weights(
            filepath=f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )

    def train(self, data):
        train = data[data['train']]
        val = data[~data['train']]

        train_rec_volume = 1 / np.array(train['reindexed_volume'])
        val_rec_volume = 1 / np.array(train['reindexed_volume'])
        train_rec_volume_scaled = self.volume_scaler.transform(
            train_rec_volume[:, np.newaxis]
            )[:, 0]
        val_rec_volume_scaled = self.volume_scaler.transform(
            val_rec_volume[:, np.newaxis]
            )[:, 0]

        train_xnn = np.stack(train['reindexed_xnn'])[:, self.data_params['unit_cell_indices']],
        val_xnn = np.stack(val['reindexed_xnn'])[:, self.data_params['unit_cell_indices']],
        train_xnn_volume_scaled = train_xnn / train_rec_volume[:, np.newaxis]**(2/3)
        val_xnn_volume_scaled = val_xnn / val_rec_volume[:, np.newaxis]**(2/3)

        train_inputs = {'q2_scaled': np.stack(train['q2_scaled'])}
        val_inputs = {'q2_scaled': np.stack(val['q2_scaled'])}

        train_true_volume = {'rec_volume_scaled': train_rec_volume_scaled}
        val_true_volume = {'rec_volume_scaled': val_rec_volume_scaled}

        train_true_xnn = {'xnn_volume_scaled': train_xnn_volume_scaled}
        val_true_xnn = {'xnn_volume_scaled': val_xnn_volume_scaled}

        train_true = {
            'rec_volume_scaled': train_rec_volume_scaled,
            'xnn_volume_scaled': train_xnn_volume_scaled,
            'hkl_softmax': np.stack(train['hkl_labels']),
            }
        val_true = {
            'rec_volume_scaled': val_rec_volume_scaled,
            'xnn_volume_scaled': val_xnn_volume_scaled,
            'hkl_softmax': np.stack(val['hkl_labels']),
            }

        self.fit_history = [None, None, None, None]
        print('Training Physics Informed Model')
        print(f'\n   Starting fitting: Volume {self.split_group}')
        self.fit_history[0] = self.volume_model.fit(
            x=train_inputs,
            y=train_true_volume,
            epochs=self.model_params['epochs_volume'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true_volume),
            callbacks=None,
            )

        print(f'\n   Starting fitting: Xnn {self.split_group}')
        self.transfer_weights('volume', 'xnn')
        self.fit_history[1] = self.xnn_model.fit(
            x=train_inputs,
            y=train_true_xnn,
            epochs=self.model_params['epochs_xnn'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true_xnn),
            callbacks=None,
            )

        print(f'\n   Starting fitting: Xnn {self.split_group}')
        self.transfer_weights('xnn', 'volume_xnn')
        self.fit_history[2] = self.volume_xnn_model.fit(
            x=train_inputs,
            y=train_true_xnn,
            epochs=self.model_params['epochs_volume_xnn'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true_xnn),
            callbacks=None,
            )

        print(f'\n   Starting fitting: Miller index assignments calibration {self.split_group}')
        self.transfer_weights('volume_xnn', 'assign')
        self.fit_history[3] = self.assign_model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs_assignment'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )

        self.transfer_weights('xnn', 'assign')
        self.fit_history[2] = self.assign_model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs_assignment'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )

        tf.keras.backend.clear_session()
        gc.collect()
        self.plot_training_loss()
        self.save()

    def plot_training_loss(self):
        # 0: Xnn loss & MSE
        # 1: Loss & accuracy
        # 2: Loss
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        total_epochs = 2 * self.model_params['cycles_regression'] * self.model_params['epochs_regression']
        total_epochs += self.model_params['epochs_assignment'] + self.model_params['epochs_index']
        metrics = np.zeros((total_epochs, 5, 2))
        metrics[:, :, :] = np.nan
        start = -self.model_params['epochs_regression']
        for cycle_index in range(self.model_params['cycles_regression']):
            start += self.model_params['epochs_regression']
            stop = start + self.model_params['epochs_regression']
            metrics[start: stop, 0, 0] = self.fit_history[2*cycle_index].history['loss']
            metrics[start: stop, 1, 0] = self.fit_history[2*cycle_index].history['mean_squared_error']
            metrics[start: stop, 0, 1] = self.fit_history[2*cycle_index].history['val_loss']
            metrics[start: stop, 1, 1] = self.fit_history[2*cycle_index].history['val_mean_squared_error']

            start += self.model_params['epochs_regression']
            stop = start + self.model_params['epochs_regression']
            metrics[start: stop, 0, 0] = self.fit_history[2*cycle_index + 1].history['loss']
            metrics[start: stop, 1, 0] = self.fit_history[2*cycle_index + 1].history['mean_squared_error']
            metrics[start: stop, 0, 1] = self.fit_history[2*cycle_index + 1].history['val_loss']
            metrics[start: stop, 1, 1] = self.fit_history[2*cycle_index + 1].history['val_mean_squared_error']

        start += self.model_params['epochs_regression']
        stop = start + self.model_params['epochs_assignment']
        history_index = 2*self.model_params['cycles_regression']
        metrics[start: stop, 2, 0] = self.fit_history[history_index].history['hkl_softmax_loss']
        metrics[start: stop, 3, 0] = self.fit_history[history_index].history['hkl_softmax_accuracy']
        metrics[start: stop, 2, 1] = self.fit_history[history_index].history['val_hkl_softmax_loss']
        metrics[start: stop, 3, 1] = self.fit_history[history_index].history['val_hkl_softmax_accuracy']

        start += self.model_params['epochs_assignment']
        stop = start + self.model_params['epochs_index']
        history_index = 2*self.model_params['cycles_regression'] + 1
        metrics[start: stop, 0, 0] = self.fit_history[history_index].history['xnn_scaled_loss']
        metrics[start: stop, 1, 0] = self.fit_history[history_index].history['xnn_scaled_mean_squared_error']
        metrics[start: stop, 2, 0] = self.fit_history[history_index].history['hkl_softmax_loss']
        metrics[start: stop, 3, 0] = self.fit_history[history_index].history['hkl_softmax_accuracy']
        metrics[start: stop, 4, 0] = self.fit_history[history_index].history['indexing_data_loss']
        metrics[start: stop, 0, 1] = self.fit_history[history_index].history['val_xnn_scaled_loss']
        metrics[start: stop, 1, 1] = self.fit_history[history_index].history['val_xnn_scaled_mean_squared_error']
        metrics[start: stop, 2, 1] = self.fit_history[history_index].history['val_hkl_softmax_loss']
        metrics[start: stop, 3, 1] = self.fit_history[history_index].history['val_hkl_softmax_accuracy']
        metrics[start: stop, 4, 1] = self.fit_history[history_index].history['val_indexing_data_loss']

        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes_r = [axes[i].twinx() for i in range(3)]
        axes[0].plot(metrics[:, 0, 0], label='Training Xnn Loss', color=colors[0], marker='.')
        axes[0].plot(metrics[:, 0, 1], label='Val Xnn Loss', color=colors[1], marker='x')
        axes_r[0].plot(metrics[:, 1, 0], label='Training Xnn MSE', color=colors[2], marker='.')
        axes_r[0].plot(metrics[:, 1, 1], label='Val Xnn MSE', color=colors[3], marker='x')
        axes[1].plot(metrics[:, 2, 0], label='Training HKL Loss', color=colors[0], marker='.')
        axes[1].plot(metrics[:, 2, 1], label='Val HKL Loss', color=colors[1], marker='x')
        axes_r[1].plot(metrics[:, 3, 0], label='Training HKL Accuracy', color=colors[2], marker='.')
        axes_r[1].plot(metrics[:, 3, 1], label='Val HKL Accuracy', color=colors[3], marker='x')
        axes[2].plot(metrics[:, 4, 0], label='Training Indexing Loss', color=colors[0], marker='.')
        axes[2].plot(metrics[:, 4, 1], label='Val Indexing Loss', color=colors[1], marker='x')

        for i in range(2):
            hl, ll = axes[i].get_legend_handles_labels()
            hr, lr = axes_r[i].get_legend_handles_labels()
            axes[i].legend(hl + hr, ll + lr, frameon=False)
        axes[2].legend(frameon=False)
        axes[2].set_xlabel('Epoch')
        axes[0].set_ylabel('Xnn Loss')
        axes_r[0].set_ylabel('Xnn MSE')
        axes[1].set_ylabel('HKL Loss')
        axes_r[1].set_ylabel('HKL Accuracy')
        axes[2].set_ylabel('Indexing Loss')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.split_group}_training_loss.png')
        plt.close()

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
                shape=self.data_params['n_peaks'],
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
        self.pairwise_difference_calculation_numpy = PairwiseDifferenceCalculator(
            lattice_system=self.data_params['lattice_system'],
            hkl_ref=self.hkl_ref,
            tensorflow=False,
            q2_scaler=self.q2_scaler,
            )

        self.volume_layer_names = []
        for index in range(len(self.model_params['volume_params']['layers'])):
            self.var_layer_names.append(f'dense_volume_scaled_{index}')
            self.var_layer_names.append(f'layer_norm_volume_scaled_{index}')
        self.volume_layer_names.append('volume_scaled')

        self.xnn_layer_names = []
        for index in range(len(self.model_params['xnn_params']['layers'])):
            self.xnn_layer_names.append(f'dense_xnn_volume_scaled_{index}')
            self.xnn_layer_names.append(f'layer_norm_xnn_volume_scaled_{index}')
        self.xnn_layer_names.append('xnn_volume_scaled')

        self.calibration_layer_names = []
        for index in range(len(self.model_params['assign_params']['layers'])):
            self.calibration_layer_names.append(f'dense_assign_{index}')
            self.calibration_layer_names.append(f'layer_norm_assign_{index}')
        for index in range(self.n_peaks):
            self.calibration_layer_names.append(f'assign_scaler_{index}')

        self.volume_model = tf.keras.Model(inputs, self.model_builder(inputs, 'volume'))
        self.xnn_model = tf.keras.Model(inputs, self.model_builder(inputs, 'xnn'))
        self.volume_xnn_model = tf.keras.Model(inputs, self.model_builder(inputs, 'xnn'))
        self.assign_model = tf.keras.Model(inputs, self.model_builder(inputs, 'assign'))
        self.model = tf.keras.Model(inputs, self.model_builder(inputs, 'assign'))
        self.compile_model()

    def model_builder(self, inputs, model_type):
        volume_scaled = mlp_model_builder(
            inputs['q2_scaled'],
            'volume_scaled',
            self.model_params['volume_params'],
            'volume_scaled'
            )
        if model_type == 'volume':
            return volume_scaled

        volume = volume_scaled*self.volume_scaler.scale_[0] + self.volume_scaler.mean_[0]
        q2 = inputs['q2_scaled']*self.q2_scaler.scale_[0] + self.q2_scaler.mean_[0]
        q2_volume_scaled = q2 / volume**(2/3)

        xnn_volume_scaled = mlp_model_builder(
            q2_volume_scaled,
            'xnn_volume_scaled',
            self.model_params['xnn_params'],
            'xnn_volume_scaled'
            )
        if model_type == 'xnn':
            return [volume_scaled, xnn_volume_scaled]

        xnn = xnn_volume_scaled * volume**(2/3)
        pairwise_differences_scaled = self.pairwise_difference_calculator.get_pairwise_differences(
            xnn, inputs['q2_scaled'], return_q2_ref=False
            )

        # hkl_logits:               n_batch x n_peaks x hkl_ref_length
        # pairwise_differences:     n_batch x n_peaks x hkl_ref_length
        # q2_ref:                   n_batch x hkl_ref_length
        pairwise_differences_transformed = self.transform_pairwise_differences(
            pairwise_differences_scaled, True
            )

        hkl_logits = scaling_model_builder(
            pairwise_differences_transformed,
            'calibration',
            self.model_params['calibration_params'],
            )

        hkl_softmax = tf.keras.layers.Softmax(
            name='hkl_softmax',
            axis=2
            )(hkl_logits)
        if model_type == 'assign':
            return [volume_scaled, xnn_volume_scaled, hkl_softmax]

    def compile_model(self):
        ################
        # volume model #
        ################
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_volume'])
        loss_weights = {
            'rec_volume_scaled': 1,
            }
        loss_metrics = {
            'rec_volume_scaled': None,
            }
        loss_functions = {
            'rec_volume_scaled': tf.keras.losses.MeanSquaredError(),
            }
        self.volume_model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics
            )

        #############
        # xnn model #
        #############
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_xnn'])
        for layer_name in self.volume_layer_names:
            self.xnn_model.get_layer(layer_name).trainable = False
        for layer_name in self.xnn_layer_names:
            self.xnn_model.get_layer(layer_name).trainable = True
        loss_weights = {
            'xnn_volume_scaled': 1,
            }
        loss_metrics = {
            'xnn_volume_scaled': None,
            }
        loss_functions = {
            'xnn_volume_scaled': tf.keras.losses.MeanSquaredError(),
            }
        self.xnn_model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics
            )

        ####################
        # volume xnn model #
        ####################
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_volume_xnn'])
        for layer_name in self.volume_layer_names:
            self.volume_xnn_model.get_layer(layer_name).trainable = True
        for layer_name in self.xnn_layer_names:
            self.volume_xnn_model.get_layer(layer_name).trainable = True
        loss_weights = {
            'volume_scaled': 1,
            'xnn_volume_scaled': 1,
            }
        loss_metrics = {
            'volume_scaled': None,
            'xnn_volume_scaled': None,
            }
        loss_functions = {
            'volume_scaled': tf.keras.losses.MeanSquaredError(),
            'xnn_volume_scaled': tf.keras.losses.MeanSquaredError(),
            }
        self.volume_xnn_model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics
            )

        ################
        # assign model #
        ################
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_assign'])
        for layer_name in self.volume_layer_names:
            self.assign_model.get_layer(layer_name).trainable = False
        for layer_name in self.xnn_layer_names:
            self.assign_model.get_layer(layer_name).trainable = False
        for layer_name in self.calibration_layer_names:
            self.assign_model.get_layer(layer_name).trainable = True
        loss_weights = {
            'hkl_softmax': 1,
            }
        loss_metrics = {
            'hkl_softmax': 'accuracy',
            }
        loss_functions = {
            'hkl_softmax': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            }
        self.assign_model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics
            )

        ##############
        # full model #
        ##############
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_full'])
        loss_weights = {
            'volume_scaled': 1,
            'xnn_volume_scaled': 1,
            'hkl_softmax': 1,
            }
        loss_metrics = {
            'volume_scaled': None,
            'xnn_volume_scaled': None,
            'hkl_softmax': 'accuracy',
            }
        loss_functions = {
            'volume_scaled': tf.keras.losses.MeanSquaredError(),
            'xnn_volume_scaled': tf.keras.losses.MeanSquaredError(),
            'hkl_softmax': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            }
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics
            )

    def transfer_weights(self, source, dest):
        tmp_dir = os.path.join(self.save_to, 'tmp')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        if source == 'volume' and dest == 'xnn':
            volume_name = os.path.join(tmp_dir, 'volume.h5')
            self.volume_model.save_weights(volume_name)
            self.xnn_model.load_weights(volume_name, by_name=True, skip_mismatch=True)
        elif source == 'xnn' and dest == 'assign':
            xnn_name = os.path.join(tmp_dir, 'xnn.h5')
            self.xnn_model.save_weights(xnn_name)
            self.assign_model.load_weights(xnn_name, by_name=True, skip_mismatch=True)
        shutil.rmtree(tmp_dir)

    def predict(self, data=None, inputs=None, q2_scaled=None, batch_size=None):
        if not data is None:
            q2_scaled = np.stack(data['q2_scaled'])
        elif not inputs is None:
            q2_scaled = inputs['q2_scaled']

        #print(f'\n Regression inferences for {self.split_group}')
        if batch_size is None:
            batch_size = self.model_params['batch_size']

        # predict_on_batch helps with a memory leak...
        N = q2_scaled.shape[0]
        n_batches = N // batch_size
        left_over = N % batch_size
        xnn_pred_scaled = np.zeros((N, self.unit_cell_length))
        xnn_pred_scaled_var = np.zeros((N, self.unit_cell_length))
        hkl_softmax = np.zeros((N, self.n_peaks, self.data_params['hkl_ref_length']))
        for batch_index in range(n_batches + 1):
            start = batch_index * batch_size
            if batch_index == n_batches:
                batch_inputs = {'q2_scaled': np.zeros((batch_size, self.n_peaks))}
                batch_inputs['q2_scaled'][:left_over] = q2_scaled[start: start + left_over]
                batch_inputs['q2_scaled'][left_over:] = q2_scaled[0]
            else:
                batch_inputs = {'q2_scaled': q2_scaled[start: start + batch_size]}

            outputs = self.model.predict_on_batch(batch_inputs)

            if batch_index == n_batches:
                xnn_pred_scaled[start:] = outputs[0][:left_over, :, 0]
                xnn_pred_scaled_var[start:] = outputs[0][:left_over, :, 1]
                hkl_softmax[start:] = outputs[1][:left_over]
            else:
                xnn_pred_scaled[start: start + batch_size] = outputs[0][:, :, 0]
                xnn_pred_scaled_var[start: start + batch_size] = outputs[0][:, :, 1]
                hkl_softmax[start: start + batch_size] = outputs[1]

        xnn_pred = xnn_pred_scaled * self.xnn_scaler.scale_[0] + self.xnn_scaler.mean_[0]
        xnn_pred_var = xnn_pred_scaled_var * self.xnn_scaler.scale_[0]**2
        xnn_pred = fix_unphysical(xnn=xnn_pred, lattice_system=self.data_params['lattice_system'], rng=self.rng)
        return xnn_pred, xnn_pred_var, hkl_softmax

    def generate(self, n_unit_cells, rng, q2_obs, batch_size=None):
        if batch_size is None:
            batch_size = self.model_params['batch_size']
        q2_scaled = (q2_obs - self.q2_scaler.mean_[0]) / self.q2_scaler.scale_[0]
        pairwise_difference_calculator_numpy = PairwiseDifferenceCalculator(
            lattice_system=self.data_params['lattice_system'],
            hkl_ref=self.hkl_ref,
            tensorflow=False,
            q2_scaler=self.q2_scaler,
            )
        # NN model assumes (batch_size, n_peaks). The generate function just does one entry at a time
        # So there needs to be a np.newaxis to make q2_scaled (1, n_peaks)
        xnn_pred, xnn_pred_var, hkl_softmax = self.predict(
            q2_scaled=q2_scaled[np.newaxis], batch_size=batch_size
            )
        unit_cell_gen = np.zeros((n_unit_cells, self.unit_cell_length))
        xnn_gen = np.zeros((n_unit_cells, self.unit_cell_length))
        loss = np.zeros(n_unit_cells)
        order = np.arange(self.n_peaks)

        hkl_assign_gen, _ = vectorized_resampling(
            np.repeat(hkl_softmax, n_unit_cells, axis=0), rng
            )
        hkl_assign_gen = np.unique(hkl_assign_gen, axis=0)
        if hkl_assign_gen.shape[0] < n_unit_cells:
            status = True
            while status:
                hkl_assign_gen_extra, _ = vectorized_resampling(
                    np.repeat(hkl_softmax, n_unit_cells - hkl_assign_gen.shape[0], axis=0), rng
                    )
                hkl_assign_gen = np.concatenate((hkl_assign_gen, hkl_assign_gen_extra), axis=0)
                hkl_assign_gen = np.unique(hkl_assign_gen, axis=0)
                if hkl_assign_gen.shape[0] >= n_unit_cells:
                    status = False

        hkl2_all = get_hkl_matrix(self.hkl_ref[hkl_assign_gen], self.data_params['lattice_system'])
        for candidate_index in range(n_unit_cells):
            sigma = q2_obs
            hkl2 = hkl2_all[candidate_index]
            status = True
            i = 0
            xnn_last = np.zeros(self.unit_cell_length)
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
            xnn_gen[candidate_index] = xnn_current
            loss[candidate_index] = np.linalg.norm(1 - q2_calc/q2_obs)
        xnn_gen = fix_unphysical(
            xnn=xnn_gen, rng=rng, lattice_system=self.data_params['lattice_system']
            )
        unit_cell_gen = get_unit_cell_from_xnn(
            xnn_gen, partial_unit_cell=True, lattice_system=self.data_params['lattice_system']
            )
        return unit_cell_gen

    def evaluate_indexing(self, data, xnn_key, unit_cell_indices):
        xnn = np.stack(data[xnn_key])[:, unit_cell_indices]

        labels_true = np.stack(data['hkl_labels'])
        labels_true_train = np.stack(data[data['train']]['hkl_labels'])
        labels_pred_train = np.stack(data[data['train']]['hkl_labels_pred'])
        labels_true_val = np.stack(data[~data['train']]['hkl_labels'])
        labels_pred_val = np.stack(data[~data['train']]['hkl_labels_pred'])

        # correct shape: n_entries, n_peaks
        correct_pred_train = labels_true_train == labels_pred_train
        correct_pred_val = labels_true_val == labels_pred_val
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
        fig.savefig(f'{self.save_to}/{self.split_group}_assignment_{self.model_params["tag"]}.png')
        plt.close()    

    def calibrate_indexing(self, data):
        def calibration_plots(softmaxes, n_peaks, n_bins=25):
            N = softmaxes.shape[0]
            y_pred = softmaxes.argmax(axis=2)
            p_pred = np.zeros((N, n_peaks))
            metrics = np.zeros((n_bins, 4))
            ece = 0
            for entry_index in range(N):
                for point_index in range(n_peaks):
                    p_pred[entry_index, point_index] = softmaxes[
                        entry_index,
                        point_index,
                        y_pred[entry_index, point_index]
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
                    y_pred_bin = y_pred[indices]
                    y_true_bin = y_true[indices]
                    metrics[bin_index, 1] = np.sum(y_pred_bin == y_true_bin) / y_true_bin.size
                    metrics[bin_index, 2] = p_pred_bin.mean()
                    metrics[bin_index, 3] = p_pred_bin.std()
                    prefactor = indices.sum() / indices.size
                    ece += prefactor * np.abs(metrics[bin_index, 2] - metrics[bin_index, 1])
            return metrics, ece

        softmaxes = np.stack(data['hkl_softmaxes'])
        y_true = np.stack(data['hkl_labels'])

        metrics, ece = calibration_plots(softmaxes, self.n_peaks)
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        axes.plot([0, 1], [0, 1], linestyle='dotted', color=[0, 0, 0])
        axes.set_xlabel('Confidence')
        axes.errorbar(metrics[:, 2], metrics[:, 1], yerr=metrics[:, 3], marker='.')

        axes.set_ylabel('Accuracy')
        axes.set_title(f'Unscaled\nExpected Confidence Error: {ece:0.4f}')
        axes.set_title(f'Expected Confidence Error: {ece:0.4f}')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.split_group}_pitf_assignment_calibration_{self.model_params["tag"]}.png')
        plt.close()
