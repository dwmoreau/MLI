import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import tensorflow as tf

from Networks import mlp_model_builder
from TargetFunctions import IndexingTargetFunction
from TargetFunctions import LikelihoodLoss
from Utilities import PairwiseDifferenceCalculator
from Utilities import vectorized_resampling
from Utilities import read_params
from Utilities import write_params


class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibration = self.add_weight(
            shape=(3,),
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.05, seed=None),
            name='calibration',
            dtype=tf.float32,
            )

    def call(self, inputs):
        outputs = self.calibration[0] * tf.math.sqrt(inputs)
        outputs += self.calibration[1] * inputs
        outputs += self.calibration[2] * inputs**2
        return outputs


class PhysicsInformedModel:
    def __init__(self, bravais_lattice, data_params, model_params, save_to, seed, q2_scaler, xnn_scaler, hkl_ref):
        self.bravais_lattice = bravais_lattice
        self.data_params = data_params
        self.model_params = model_params
        self.model_params['mean_params']['n_outputs'] = len(self.data_params['y_indices'])
        self.model_params['var_params']['n_outputs'] = len(self.data_params['y_indices'])
        self.n_points = data_params['n_points']
        self.n_outputs = data_params['n_outputs']
        self.y_indices = data_params['y_indices']
        self.save_to = save_to
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.q2_scaler = q2_scaler
        self.xnn_scaler = xnn_scaler
        self.hkl_ref = hkl_ref

    def setup(self):
        model_params_defaults = {
            'mean_params': {
                'layers': [5000, 2000, 1000],
                'dropout_rate': 0.5,
                'epsilon': 0.001,
                'output_activation': 'linear',
                'kernel_initializer': None,
                'bias_initializer': None,
                },
            'var_params': {
                'layers': [100, 100, 100],
                'dropout_rate': 0.5,
                'epsilon': 0.001,
                'output_activation': 'exponential',
                'kernel_initializer': None,
                'bias_initializer': None,
                },
            'epsilon_pds': 0.1,
            'learning_rate_regression': 0.001,
            'learning_rate_assignment': 0.001,
            'learning_rate_index': 0.000001,
            'cycles_regression': 2,
            'epochs_regression': 10,
            'epochs_assignment': 15,
            'epochs_index': 15,
            'batch_size': 128,
            'beta_nll': 0.5,
            }

        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        self.model_params['mean_params']['n_outputs'] = self.n_outputs
        self.model_params['var_params']['n_outputs'] = self.n_outputs

        for key in model_params_defaults['mean_params'].keys():
            if key not in self.model_params['mean_params'].keys():
                self.model_params['mean_params'][key] = model_params_defaults['mean_params'][key]
        self.model_params['mean_params']['kernel_initializer'] = None
        self.model_params['mean_params']['bias_initializer'] = None

        for key in model_params_defaults['var_params'].keys():
            if key not in self.model_params['var_params'].keys():
                self.model_params['var_params'][key] = model_params_defaults['var_params'][key]
        self.model_params['var_params']['kernel_initializer'] = None
        self.model_params['var_params']['bias_initializer'] = None

        self.build_model()
        #self.model.summary()

    def save(self):
        write_params(self.model_params, f'{self.save_to}/{self.bravais_lattice}_pitf_params_{self.model_params["tag"]}.csv')
        self.model.save_weights(f'{self.save_to}/{self.bravais_lattice}_pitf_weights_{self.model_params["tag"]}.h5')

    def load_from_tag(self):
        params = read_params(f'{self.save_to}/{self.bravais_lattice}_pitf_params_{self.model_params["tag"]}.csv')
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
            'n_outputs',
            'kernel_initializer',
            'bias_initializer',
            ]
        network_keys = ['mean_params', 'var_params']
        for network_key in network_keys:
            self.model_params[network_key] = dict.fromkeys(params_keys)
            self.model_params[network_key]['n_outputs'] = self.n_outputs
            for element in params[network_key].split('{')[1].split('}')[0].split(", '"):
                key = element.replace("'", "").split(':')[0]
                value = element.replace("'", "").split(':')[1]
                if key in ['dropout_rate', 'epsilon']:
                    self.model_params[network_key][key] = float(value)
                if key == 'layers':
                    self.model_params[network_key]['layers'] = np.array(
                        value.split('[')[1].split(']')[0].split(','),
                        dtype=int
                        )
            self.model_params[network_key]['kernel_initializer'] = None
            self.model_params[network_key]['bias_initializer'] = None

        self.build_model()
        self.model.load_weights(
            filepath=f'{self.save_to}/{self.bravais_lattice}_pitf_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )
        self.compile_model(mode='index')

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


        self.fit_history = [None for _ in range(2*self.model_params['cycles_regression'] + 2)]
        print('\nStarting fitting: Unit cell regression')
        for cycle_index in range(self.model_params['cycles_regression']):
            print(f'\n   Regression mean cycle: {cycle_index + 1}')
            self.compile_model('regression_mean')
            self.fit_history[2*cycle_index] = self.model.fit(
                x=train_inputs,
                y=train_true,
                epochs=self.model_params['epochs_regression'],
                shuffle=True,
                batch_size=self.model_params['batch_size'], 
                validation_data=(val_inputs, val_true),
                callbacks=None,
                )
            print(f'\n   Regression var cycle: {cycle_index + 1}')
            self.compile_model('regression_var')
            self.fit_history[2*cycle_index + 1] = self.model.fit(
                x=train_inputs,
                y=train_true,
                epochs=self.model_params['epochs_regression'],
                shuffle=True,
                batch_size=self.model_params['batch_size'], 
                validation_data=(val_inputs, val_true),
                callbacks=None,
                )

        print('\nStarting fitting: Miller index assignments calibration')
        self.compile_model('assignment')
        self.fit_history[2*self.model_params['cycles_regression']] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs_assignment'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
        
        print('\nStarting indexing: Unit cell & assignment calibration')
        self.compile_model('index')
        self.fit_history[2*self.model_params['cycles_regression'] + 1] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs_index'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
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
        start = -self.model_params['epochs_regression']
        for cycle_index in range(self.model_params['cycles_regression']):
            start += self.model_params['epochs_regression']
            stop = start + self.model_params['epochs_regression']
            metrics[start: stop, 0, 0] = self.fit_history[2*cycle_index].history['xnn_scaled_loss']
            metrics[start: stop, 1, 0] = self.fit_history[2*cycle_index].history['xnn_scaled_mean_squared_error']
            metrics[start: stop, 2, 0] = self.fit_history[2*cycle_index].history['hkl_softmax_loss']
            metrics[start: stop, 3, 0] = self.fit_history[2*cycle_index].history['hkl_softmax_accuracy']
            metrics[start: stop, 4, 0] = self.fit_history[2*cycle_index].history['indexing_data_loss']
            metrics[start: stop, 0, 1] = self.fit_history[2*cycle_index].history['val_xnn_scaled_loss']
            metrics[start: stop, 1, 1] = self.fit_history[2*cycle_index].history['val_xnn_scaled_mean_squared_error']
            metrics[start: stop, 2, 1] = self.fit_history[2*cycle_index].history['val_hkl_softmax_loss']
            metrics[start: stop, 3, 1] = self.fit_history[2*cycle_index].history['val_hkl_softmax_accuracy']
            metrics[start: stop, 4, 1] = self.fit_history[2*cycle_index].history['val_indexing_data_loss']

            start += self.model_params['epochs_regression']
            stop = start + self.model_params['epochs_regression']
            metrics[start: stop, 0, 0] = self.fit_history[2*cycle_index + 1].history['xnn_scaled_loss']
            metrics[start: stop, 1, 0] = self.fit_history[2*cycle_index + 1].history['xnn_scaled_mean_squared_error']
            metrics[start: stop, 2, 0] = self.fit_history[2*cycle_index + 1].history['hkl_softmax_loss']
            metrics[start: stop, 3, 0] = self.fit_history[2*cycle_index + 1].history['hkl_softmax_accuracy']
            metrics[start: stop, 4, 0] = self.fit_history[2*cycle_index + 1].history['indexing_data_loss']
            metrics[start: stop, 0, 1] = self.fit_history[2*cycle_index + 1].history['val_xnn_scaled_loss']
            metrics[start: stop, 1, 1] = self.fit_history[2*cycle_index + 1].history['val_xnn_scaled_mean_squared_error']
            metrics[start: stop, 2, 1] = self.fit_history[2*cycle_index + 1].history['val_hkl_softmax_loss']
            metrics[start: stop, 3, 1] = self.fit_history[2*cycle_index + 1].history['val_hkl_softmax_accuracy']
            metrics[start: stop, 4, 1] = self.fit_history[2*cycle_index + 1].history['val_indexing_data_loss']

        start += self.model_params['epochs_regression']
        stop = start + self.model_params['epochs_assignment']
        history_index = 2*self.model_params['cycles_regression']
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
        fig.savefig(f'{self.save_to}/{self.bravais_lattice}_training_loss.png')
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

        self.mean_layer_names = []
        for index in range(len(self.model_params['mean_params']['layers'])):
            self.mean_layer_names.append(f'dense_xnn_scaled_mean_{index}')
            self.mean_layer_names.append(f'layer_norm_xnn_scaled_mean_' + str(index))
        self.mean_layer_names.append('xnn_scaled_mean')

        self.var_layer_names = []
        for index in range(len(self.model_params['var_params']['layers'])):
            self.var_layer_names.append(f'dense_xnn_scaled_var_{index}')
            self.var_layer_names.append(f'layer_norm_xnn_scaled_var_' + str(index))
        self.var_layer_names.append('xnn_scaled_var')

        self.calibration_layer_names = ['calibration']

        self.model = tf.keras.Model(inputs, self.model_builder(inputs))

    def model_builder(self, inputs):
        xnn_scaled_mean = mlp_model_builder(
            inputs['q2_scaled'],
            'xnn_scaled_mean',
            self.model_params['mean_params'],
            'xnn_scaled_mean'
            )
        xnn_scaled_var = mlp_model_builder(
            inputs['q2_scaled'],
            'xnn_scaled_var',
            self.model_params['var_params'],
            'xnn_scaled_var'
            )
        xnn_scaled = tf.keras.layers.Concatenate(
            axis=2,
            name='xnn_scaled'
            )((
                xnn_scaled_mean[:, :, tf.newaxis],
                xnn_scaled_var[:, :, tf.newaxis],
                ))

        xnn = xnn_scaled_mean * self.xnn_scaler.scale_[0] + self.xnn_scaler.mean_[0]
        pairwise_differences_scaled, q2_ref = self.pairwise_difference_calculator.get_pairwise_differences(
            xnn, inputs['q2_scaled'], return_q2_ref=True
            )

        # hkl_logits:               n_batch x n_points x hkl_ref_length
        # pairwise_differences:     n_batch x n_points x hkl_ref_length
        # q2_ref:                   n_batch x hkl_ref_length
        pairwise_differences_transformed = self.transform_pairwise_differences(
            pairwise_differences_scaled, True
            )
        hkl_logits = ScalingLayer(
            name='scaling_layer'
            )(pairwise_differences_transformed)
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
            likelihood='normal', 
            error_fraction=np.linspace(0.01, 0.1, self.data_params['n_points']),
            n_points=self.data_params['n_points'],
            )

        if mode.startswith('regression'):
            optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_regression'])
            loss_weights = {
                'xnn_scaled': 1,
                'hkl_softmax': 0,
                'indexing_data': 0,
                }
            self.model.get_layer('scaling_layer').trainable = False
            if mode == 'regression_mean':
                reg_loss = LikelihoodLoss(
                    likelihood='normal',
                    n=self.n_outputs,
                    beta_nll=self.model_params['beta_nll'],
                    )
                for layer_name in self.mean_layer_names:
                    self.model.get_layer(layer_name).trainable = True
                for layer_name in self.var_layer_names:
                    self.model.get_layer(layer_name).trainable = False
            elif mode == 'regression_var':
                reg_loss = LikelihoodLoss(
                    likelihood='normal',
                    n=self.n_outputs,
                    beta_nll=None,
                    )
                for layer_name in self.mean_layer_names:
                    self.model.get_layer(layer_name).trainable = False
                for layer_name in self.var_layer_names:
                    self.model.get_layer(layer_name).trainable = True
        elif mode == 'assignment':
            reg_loss = LikelihoodLoss(
                likelihood='normal',
                n=self.n_outputs,
                beta_nll=None,
                )
            optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_assignment'])
            loss_weights = {
                'xnn_scaled': 0,
                'hkl_softmax': 1,
                'indexing_data': 0,
                }
            for layer_name in self.mean_layer_names:
                self.model.get_layer(layer_name).trainable = False
            for layer_name in self.var_layer_names:
                self.model.get_layer(layer_name).trainable = False
            self.model.get_layer('scaling_layer').trainable = True
        elif mode == 'index':
            reg_loss = LikelihoodLoss(
                likelihood='normal',
                n=self.n_outputs,
                beta_nll=None,
                )
            optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate_index'])
            loss_weights = {
                'xnn_scaled': 1,
                'hkl_softmax': 0,
                'indexing_data': 1,
                }
            for layer_name in self.mean_layer_names:
                self.model.get_layer(layer_name).trainable = True
            for layer_name in self.var_layer_names:
                self.model.get_layer(layer_name).trainable = False
            self.model.get_layer('scaling_layer').trainable = True

        loss_metrics = {
            'xnn_scaled': reg_loss.mean_squared_error,
            'hkl_softmax': 'accuracy',
            'indexing_data': indexing_loss,
            }
        loss_functions = {
            'xnn_scaled': reg_loss,
            'hkl_softmax': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            'indexing_data': indexing_loss,
            }

        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics
            )
        #self.model.summary()

    def do_predictions(self, data=None, inputs=None, q2_scaled=None, batch_size=None):
        if not data is None:
            q2_scaled = np.stack(data['q2_scaled'])
        elif not inputs is None:
            q2_scaled = inputs['q2_scaled']

        print(f'\n Regression inferences for {self.bravais_lattice}')
        if batch_size is None:
            batch_size = self.model_params['batch_size']

        # predict_on_batch helps with a memory leak...
        N = len(data)
        n_batches = N // batch_size
        left_over = N % batch_size
        xnn_pred_scaled = np.zeros((N, self.n_outputs))
        xnn_pred_scaled_var = np.zeros((N, self.n_outputs))
        hkl_softmax = np.zeros((N, self.n_points, self.data_params['hkl_ref_length']))
        for batch_index in range(n_batches + 1):
            start = batch_index * batch_size
            if batch_index == n_batches:
                batch_inputs = {'q2_scaled': np.zeros((batch_size, self.n_points))}
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
        return xnn_pred, xnn_pred_var, hkl_softmax

    def generate_candidates(self, q2_scaled=None, batch_size=None, n_candidates=None):
        # This is not debuged and 100% will not work
        pairwise_difference_calculator_numpy = PairwiseDifferenceCalculator(
            lattice_system=self.data_params['lattice_system'],
            hkl_ref=self.hkl_ref,
            tensorflow=False,
            q2_scaler=self.q2_scaler,
            )
        calibration_weights = np.array(self.model.get_weights('calibration'))
        xnn_pred, xnn_pred_var, hkl_softmax = self.do_predictions(
            q2_scaled=q2_scaled[np.newaxis], batch_size=batch_size
            )
        unit_cell_gen = np.zeros((n_candidates, self.n_outputs))
        xnn_gen = np.zeros((n_candidates, self.n_outputs))
        loss = np.zeros(n_candidates)
        order = np.arange(self.n_points)

        hkl_assign_gen, _ = vectorized_resampling(
            np.repeat(hkl_softmax, n_candidates, axis=0), self.rng
            )
        hkl_assign_gen = np.unique(hkl_assign_gen, axis=0)

        if hkl_assign_gen.shape[0] < n_candidates:
            status = True
            while status:
                xnn_pred_sampled = self.rng.normal(
                    loc=xnn_pred,
                    scale=np.sqrt(xnn_pred_var),
                    size=1
                    )
                pairwise_differences_scaled = pairwise_difference_calculator_numpy.get_pairwise_differences(
                    xnn_pred_sampled, q2_scaled[np.newaxis], return_q2_ref=False
                    )
                pairwise_differences_transformed = self.transform_pairwise_differences(
                    pairwise_differences_scaled, tensorflow=False
                    )
                hkl_logits = calibration_weights[0] * np.sqrt(pairwise_differences_transformed) \
                    + calibration_weights[1] * pairwise_differences_transformed \
                    + calibration_weights[2] * pairwise_differences_transformed**2
                hkl_assign_gen_loop, _ = vectorized_resampling(
                    np.repeat(scipy.special.softmax(hkl_logits, axis=1), n_candidates, axis=0),
                    self.rng
                    )
                hkl_assign_gen = np.stack((hkl_assign_gen, hkl_assign_gen_loop))
                hkl_assign_gen = np.unique(hkl_assign_gen, axis=0)
                if hkl_assign_gen.shape[0] >= n_candidates:
                    status = False

        hkl2_all = get_hkl_matrix(self.hkl_ref[hkl_assign_gen], self.lattice_system)
        for candidate_index in range(n_candidates):
            sigma = q2_obs
            hkl2 = hkl2_all[candidate_index]
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
            xnn_gen[candidate_index] = xnn_current
            loss[candidate_index] = np.linalg.norm(1 - q2_calc/q2_obs)
        xnn_gen = fix_unphysical(xnn=xnn_gen, rng=self.rng, lattice_system=self.lattice_system)
        unit_cell_gen = get_unit_cell_from_xnn(
            xnn_gen, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return unit_cell_gen