import copy
import csv
import gc
import joblib
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

from Networks import mlp_model_builder
from Networks import bnn_model_builder
from TargetFunctions import LikelihoodLoss


class Regression_base:
    def __init__(self, group, data_params, model_params, save_to, unit_cell_key, seed):
        self.model_params = model_params
        self.n_points = data_params['n_points']
        self.n_outputs = data_params['n_outputs']
        self.y_indices = data_params['y_indices']
        self.unit_cell_key = unit_cell_key
        self.group = group
        self.save_to = save_to
        self.seed = seed

    def train_regression(self, data):
        self.build_model()
        if self.model_params['fit_strategy'] == 'cycles':
            self.fit_model_cycles(data)
        elif self.model_params['fit_strategy'] == 'warmup':
            self.fit_model_warmup(data)
        self.save()

    def build_model(self):
        inputs = {
            'q2_scaled': tf.keras.Input(
                shape=self.n_points,
                name='input_points',
                dtype=tf.float32,
                )
            }
        if self.model_params['nn_type'] == 'mlp':
            self.model = tf.keras.Model(inputs, self.model_builder_mlp(inputs))
        elif self.model_params['nn_type'] == 'bnn':
            self.model = tf.keras.Model(inputs, self.model_builder_bnn(inputs))
        elif self.model_params['nn_type'] == 'mlp_head':
            self.model = tf.keras.Model(inputs, self.model_builder_mlp_head(inputs))
        elif self.model_params['nn_type'] == 'rnn_head':
            self.model = tf.keras.Model(inputs, self.model_builder_rnn_head(inputs))
        #self.model.summary()
        self.get_layer_names()

    def compile_model(self, mode):
        if mode == 'mean':
            mean_setting = True
            var_setting = False
            self.reg_loss.beta_likelihood = True
        elif mode == 'variance':
            mean_setting = False
            var_setting = True
            self.reg_loss.beta_likelihood = False
        elif mode == 'both':
            mean_setting = True
            var_setting = True
            self.reg_loss.beta_likelihood = True

        for layer_name in self.mean_layer_names:
            self.model.get_layer(layer_name).trainable = mean_setting
        for layer_name in self.var_layer_names:
            self.model.get_layer(layer_name).trainable = var_setting
        self.model.compile(
            optimizer=self.optimizer, 
            loss=self.loss_functions,
            loss_weights=self.loss_weights,
            metrics=self.loss_metrics
            )

    def _get_train_val(self, data):
        train = data[data['train']]
        val = data[~data['train']]
        train_inputs = {'q2_scaled': np.stack(train['q2_scaled'])}
        val_inputs = {'q2_scaled': np.stack(val['q2_scaled'])}
        if self.model_params['predict_pca']:
            unit_cell_scaled_train = np.stack(train[f'{self.unit_cell_key}_scaled'])[:, self.y_indices]
            unit_cell_scaled_val = np.stack(val[f'{self.unit_cell_key}_scaled'])[:, self.y_indices]
            self.pca = PCA(n_components=self.n_outputs).fit(unit_cell_scaled_train)
            train_true = {
                f'uc_pred_scaled_{self.group}': self.pca.transform(unit_cell_scaled_train)
                }
            val_true = {
                f'uc_pred_scaled_{self.group}': self.pca.transform(unit_cell_scaled_val)
                }
        else:
            train_true = {
                f'uc_pred_scaled_{self.group}': np.stack(
                    train[f'{self.unit_cell_key}_scaled']
                    )[:, self.y_indices],
                }
            val_true = {
                f'uc_pred_scaled_{self.group}': np.stack(
                    val[f'{self.unit_cell_key}_scaled']
                    )[:, self.y_indices],
                }
        return train_inputs, val_inputs, train_true, val_true

    def fit_model_cycles(self, data):
        self.fit_history = [None for j in range(2 * self.model_params['cycles'])] 
        train_inputs, val_inputs, train_true, val_true = self._get_train_val(data)

        for cycle_index in range(self.model_params['cycles']):
            self.compile_model('mean')
            print(f'\n Starting cycle {cycle_index} mean for {self.group}')
            self.fit_history[2*cycle_index] = self.model.fit(
                x=train_inputs,
                y=train_true,
                epochs=self.model_params['epochs'],
                shuffle=True,
                batch_size=self.model_params['batch_size'], 
                validation_data=(val_inputs, val_true),
                )
            tf.keras.backend.clear_session()
            gc.collect()
            self.compile_model('variance')
            print(f'\n Starting cycle {cycle_index} variance for {self.group}')
            self.fit_history[2*cycle_index + 1] = self.model.fit(
                x=train_inputs,
                y=train_true,
                epochs=self.model_params['epochs'],
                shuffle=True,
                batch_size=self.model_params['batch_size'], 
                validation_data=(val_inputs, val_true),
                )
            # https://github.com/tensorflow/tensorflow/issues/37505
            # these are to deal with a memory leak.
            # This does not solve the issue
            tf.keras.backend.clear_session()
            gc.collect()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        for cycle_index in range(self.model_params['cycles']):
            epochs_mean = np.arange(self.model_params['epochs']) \
                + 2 * cycle_index * self.model_params['epochs']
            if cycle_index == 0:
                train_label = 'Training'
                val_label = 'Validation'
            else:
                train_label = None
                val_label = None
            axes[0].plot(
                epochs_mean, 
                self.fit_history[2*cycle_index].history['loss'],
                marker='.', label=train_label, color=colors[0],
                )
            axes[0].plot(
                epochs_mean, 
                self.fit_history[2*cycle_index].history['val_loss'],
                marker='v', label=val_label, color=colors[1],
                )
            axes[1].plot(
                epochs_mean, 
                self.fit_history[2*cycle_index].history['mean_squared_error'],
                marker='.', label=train_label, color=colors[0],
                )
            axes[1].plot(
                epochs_mean, 
                self.fit_history[2*cycle_index].history['val_mean_squared_error'],
                marker='v', label=val_label, color=colors[1],
                )
            epochs_var = np.arange(self.model_params['epochs']) \
                + (2 * cycle_index + 1) * self.model_params['epochs']
            axes[0].plot(
                epochs_var, 
                self.fit_history[2*cycle_index + 1].history['loss'],
                marker='.', label=train_label, color=colors[0],
                )
            axes[0].plot(
                epochs_var, 
                self.fit_history[2*cycle_index + 1].history['val_loss'],
                marker='v', label=val_label, color=colors[1],
                )
            axes[1].plot(
                epochs_var, 
                self.fit_history[2*cycle_index + 1].history['mean_squared_error'],
                marker='.', label=train_label, color=colors[0],
                )
            axes[1].plot(
                epochs_var, 
                self.fit_history[2*cycle_index + 1].history['val_mean_squared_error'],
                marker='v', label=val_label, color=colors[1],
                )
        for col in range(2):
            ylims = axes[col].get_ylim()
            for cycle_index in range(2 * self.model_params['cycles']):
                epochs = (cycle_index + 1) * self.model_params['epochs']
                axes[col].plot(
                    [epochs, epochs], ylims,
                    color=0.1 * np.ones(3), linestyle='dotted'
                    )
            axes[col].set_ylim(ylims)
        axes[0].legend()
        axes[0].set_title(self.group)
        axes[0].set_ylabel('UC Loss')
        axes[1].set_ylabel('UC MSE')
        axes[1].set_xlabel('Epoch')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.group}_reg_training_loss_{self.model_params["tag"]}.png')
        plt.close()

    def fit_model_warmup(self, data):
        self.fit_history = [None for j in range(3)] 
        train_inputs, val_inputs, train_true, val_true = self._get_train_val(data)

        self.compile_model('mean')
        print(f'\n Starting warmup mean for {self.group}')
        self.fit_history[0] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'][0],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            sample_weight=None
            )
        self.compile_model('variance')
        print(f'\n Starting warmup variance for {self.group}')
        self.fit_history[1] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'][1],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            sample_weight=None
            )
        self.compile_model('both')
        print(f'\n Training mean & variance for {self.group}')
        self.fit_history[2] = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'][2],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            sample_weight=None
            )

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        epochs_warmup_mean = np.arange(self.model_params['epochs'][0])
        epochs_warmup_var = np.arange(self.model_params['epochs'][1]) \
            + self.model_params['epochs'][0]
        epochs_train = np.arange(self.model_params['epochs'][2]) \
            + self.model_params['epochs'][1] \
            + self.model_params['epochs'][0]
        epochs = [
            epochs_warmup_mean,
            epochs_warmup_var,
            epochs_train,
            ]
        axes_1r = axes[1].twinx()
        for cycle_index in range(3):
            if cycle_index == 0:
                train_label = 'Training'
                val_label = 'Validation'
                ratio_label = 'Val / Training'
            else:
                train_label = None
                val_label = None
                ratio_label = None
            axes[0].plot(
                epochs[cycle_index], 
                self.fit_history[cycle_index].history['loss'],
                marker='.', label=train_label, color=colors[0],
                )
            axes[0].plot(
                epochs[cycle_index],
                self.fit_history[cycle_index].history['val_loss'],
                marker='v', label=val_label, color=colors[1],
                )
            mse = np.array(self.fit_history[cycle_index].history['mean_squared_error'])
            val_mse = np.array(self.fit_history[cycle_index].history['val_mean_squared_error'])
            axes[1].plot(
                epochs[cycle_index], mse,
                marker='.', label=train_label, color=colors[0],
                )
            axes[1].plot(
                epochs[cycle_index], val_mse,
                marker='v', label=val_label, color=colors[1],
                )
            axes_1r.plot(
                epochs[cycle_index], val_mse / mse,
                marker='x', label=val_label, color=colors[2],
                )
        for col in range(2):
            ylims = axes[col].get_ylim()
            for cycle_index in range(3):
                last_epoch = epochs[cycle_index][-1]
                axes[col].plot(
                    [last_epoch, last_epoch], ylims,
                    color=0.1 * np.ones(3), linestyle='dotted'
                    )
            axes[col].set_ylim(ylims)

        hl, ll = axes[0].get_legend_handles_labels()
        hr, lr = axes_1r.get_legend_handles_labels()
        axes[0].legend(hl + hr, ll + lr)
        axes[0].set_title(self.group)
        axes[0].set_ylabel('UC Loss')
        axes[1].set_ylabel('UC MSE')
        axes_1r.set_ylabel('UC MSE: Val / Training')
        axes[1].set_xlabel('Epoch')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.group}_reg_training_loss_{self.model_params["tag"]}.png')
        plt.close()

    def do_predictions(self, data=None, inputs=None, verbose=1, n_evals=500):
        if self.model_params['nn_type'] in ['bnn']:
            uc_pred_scaled, uc_pred_scaled_cov = self.do_predictions_probabalistic(
                data=data,
                inputs=inputs,
                verbose=verbose,
                n_evals=n_evals
                )
        elif self.model_params['nn_type'] in ['mlp', 'mlp_head', 'rnn_head']:
            uc_pred_scaled, uc_pred_scaled_cov = self.do_predictions_deterministic(
                data=data,
                inputs=inputs,
                verbose=verbose,
                )
        return uc_pred_scaled, uc_pred_scaled_cov

    def evaluate(self, data):
        evaluate_regression(
            data=data,
            group=self.group,
            n_outputs=self.n_outputs,
            unit_cell_key=self.unit_cell_key,
            save_to_name=f'{self.save_to}/{self.group}_reg_{self.model_params["tag"]}.png',
            y_indices=self.y_indices,
            )

    def calibrate(self, data):
        calibrate_regression(
            data=data,
            group=self.group,
            n_outputs=self.n_outputs,
            unit_cell_key=self.unit_cell_key,
            save_to_name=f'{self.save_to}/{self.group}_reg_calibration_{self.model_params["tag"]}.png',
            y_indices=self.y_indices,
            )


class Regression_AlphaBeta(Regression_base):
    def __init__(self, group, data_params, model_params, save_to, unit_cell_key, seed=12345):
        super().__init__(group, data_params, model_params, save_to, unit_cell_key, seed)

    def setup(self):
        self.model_params['mean_params']['n_outputs'] = self.n_outputs
        self.model_params['alpha_params']['n_outputs'] = self.n_outputs
        self.model_params['beta_params']['n_outputs'] = self.n_outputs
        model_params_defaults = {
            'nn_type': 'mlp_bnn',
            'fit_strategy': 'cycles',
            'beta_nll': 0.5,
            'batch_size': 64,
            'learning_rate': 0.0002,
            }
        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]

        if self.model_params['fit_strategy'] == 'cycles':
            if not 'epochs' in self.model_params.keys():
                self.model_params['epochs'] = 5
            if not 'cycles' in self.model_params.keys():
                self.model_params['cycles'] = 10
        if self.model_params['fit_strategy'] == 'warmup':
            if not 'epochs' in self.model_params.keys():
                self.model_params['epochs'] = [10, 10, 80]

        self.optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate'])

        if self.model_params['nn_type'] in ['mlp_head', 'rnn_head']:
            head_params_defaults = {
                'dropout_rate': 0.0,
                'epsilon': 0.001,
                'layers': [60],
                'output_activation': 'linear',
                'output_name': 'head',
                }
            for key in head_params_defaults.keys():
                if key not in self.model_params['head_params'].keys():
                    self.model_params['head_params'][key] = head_params_defaults[key]
        self.model_params['head_params']['kernel_initializer'] = None
        self.model_params['head_params']['bias_initializer'] = None

        mean_params_defaults = {        
            'dropout_rate': 0.0,
            'epsilon': 0.001,
            'layers': [60, 60],
            'output_activation': 'linear',
            'output_name': 'uc_mean_scaled',
            }
        for key in mean_params_defaults.keys():
            if key not in self.model_params['mean_params'].keys():
                self.model_params['mean_params'][key] = mean_params_defaults[key]
        self.model_params['mean_params']['kernel_initializer'] = None
        self.model_params['mean_params']['bias_initializer'] = None

        alpha_params_defaults = {        
            'dropout_rate': 0.0,
            'epsilon': 0.001,
            'layers': [60, 60],
            'output_activation': 'softplus',
            'output_name': 'uc_alpha_scaled',
            }
        for key in alpha_params_defaults.keys():
            if key not in self.model_params['alpha_params'].keys():
                self.model_params['alpha_params'][key] = alpha_params_defaults[key]
        self.model_params['alpha_params']['kernel_initializer'] = None
        self.model_params['alpha_params']['bias_initializer'] = \
            tf.keras.initializers.RandomNormal(mean=2, stddev=0.05, seed=self.seed)

        beta_params_defaults = {
            'dropout_rate': 0.0,
            'epsilon': 0.001,
            'layers': [60, 60],
            'output_activation': 'softplus',
            'output_name': 'uc_beta_scaled',
            }
        for key in beta_params_defaults.keys():
            if key not in self.model_params['beta_params'].keys():
                self.model_params['beta_params'][key] = beta_params_defaults[key]
        self.model_params['beta_params']['kernel_initializer'] = None
        self.model_params['beta_params']['bias_initializer'] = \
            tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.05, seed=self.seed)

        self.reg_loss = LikelihoodLoss('alpha_beta', n=self.n_outputs, beta_nll=self.model_params['beta_nll'])
        self.loss_weights = {
            f'uc_pred_scaled_{self.group}': 1
            }
        self.loss_functions = {
            f'uc_pred_scaled_{self.group}': self.reg_loss
            }
        self.loss_metrics = {
            f'uc_pred_scaled_{self.group}': [self.reg_loss.mean_squared_error]
            }

    def save(self):
        model_params = copy.deepcopy(self.model_params)
        with open(f'{self.save_to}/{self.group}_reg_params_{self.model_params["tag"]}.csv', 'w') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=model_params.keys())
            writer.writeheader()
            writer.writerow(model_params)

        network_keys = ['mean_params', 'alpha_params', 'beta_params']
        if self.model_params['nn_type'] in ['mlp_head', 'rnn_head']:
            network_keys += ['head_params']
        for network_key in network_keys:
            params = copy.deepcopy(self.model_params[network_key])
            params.pop('kernel_initializer')
            params.pop('bias_initializer')
            with open(f'{self.save_to}/{self.group}_reg_{network_key}_{self.model_params["tag"]}.csv', 'w') as output_file:
                writer = csv.DictWriter(output_file, fieldnames=params.keys())
                writer.writeheader()
                writer.writerow(params)

        self.model.save_weights(f'{self.save_to}/{self.group}_reg_weights_{self.model_params["tag"]}.h5')
        if self.model_params['predict_pca']:
            joblib.dump(self.pca, f'{self.save_to}/{self.group}_pca.bin')

    def load_from_tag(self):
        with open(f'{self.save_to}/{self.group}_reg_params_{self.model_params["tag"]}.csv', 'r') as params_file:
            reader = csv.DictReader(params_file)
            for row in reader:
                params = row
        params_keys = [
            'tag',
            'nn_type'
            'mean_params',
            'alpha_params',
            'beta_params',
            'beta_nll',
            'batch_size',
            'epochs',
            'cycles',
            'learning_rate',
            'fit_strategy',
            'predict_pca',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['nn_type'] = params['nn_type']
        self.model_params['beta_nll'] = float(params['beta_nll'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['fit_strategy'] = params['fit_strategy']
        if self.model_params['fit_strategy'] == 'cycles':
            self.model_params['epochs'] = int(params['epochs'])
            self.model_params['cycles'] = int(params['cycles'])
        if self.model_params['fit_strategy'] == 'warmup':
            self.model_params['epochs'] = np.array(
                params['epochs'].split('[')[1].split(']')[0].split(','),
                dtype=int
                )
        if params['predict_pca'] == 'True':
            self.model_params['predict_pca'] = True
            self.pca = joblib.load(f'{self.save_to}/{self.group}_pca.bin')
        elif params['predict_pca'] == 'False':
            self.model_params['predict_pca'] = False

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
        network_keys = ['mean_params', 'alpha_params', 'beta_params']
        if self.model_params['nn_type'] in ['mlp_head', 'rnn_head']:
            network_keys += ['head_params']
        for network_key in network_keys:
            with open(f'{self.save_to}/{self.group}_reg_{network_key}_{self.model_params["tag"]}.csv', 'r') as params_file:
                reader = csv.DictReader(params_file)
                for row in reader:
                    params = row
            self.model_params[network_key] = dict.fromkeys(params_keys)
            self.model_params[network_key]['dropout_rate'] = float(params['dropout_rate'])
            self.model_params[network_key]['epsilon'] = float(params['epsilon'])
            self.model_params[network_key]['layers'] = np.array(
                params['layers'].split('[')[1].split(']')[0].split(','),
                dtype=int
                )
            self.model_params[network_key]['kernel_initializer'] = None
            self.model_params[network_key]['bias_initializer'] = None
            if network_key != 'head_params':
                self.model_params[network_key]['output_activation'] = params['output_activation']
                self.model_params[network_key]['output_name'] = params['output_name']
                self.model_params[network_key]['n_outputs'] = self.n_outputs

        self.build_model()
        self.model.load_weights(
            filepath=f'{self.save_to}/{self.group}_reg_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )
        self.compile_model(mode='both')

    def model_builder_rnn_head(self, inputs):
        head_outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.model_params['head_params']['layers'][0],
            return_sequences=True
            ),
            name=f'head_{self.group}',
            )(inputs['q2_scaled'][:, :, tf.newaxis])
        head_outputs = SeqSelfAttention(
            return_attention=True,
            name=f'head_{self.group}_attn'
            )(head_outputs)
        head_outputs = tf.keras.layers.concatenate(head_outputs)
        head_outputs = tf.keras.layers.Flatten()(head_outputs)

        uc_pred_mean_scaled = mlp_model_builder(
            head_outputs,
            tag=f'uc_pred_mean_scaled_{self.group}',
            model_params=self.model_params['mean_params'],
            output_name=f'{self.model_params["mean_params"]["output_name"]}_{self.group}',
            )
        uc_pred_alpha_scaled = 1 + mlp_model_builder(
            head_outputs,
            tag=f'uc_pred_alpha_scaled_{self.group}',
            model_params=self.model_params['alpha_params'],
            output_name=f'{self.model_params["alpha_params"]["output_name"]}_{self.group}',
            )
        uc_pred_beta_scaled = mlp_model_builder(
            head_outputs,
            tag=f'uc_pred_beta_scaled_{self.group}',
            model_params=self.model_params['beta_params'],
            output_name=f'{self.model_params["beta_params"]["output_name"]}_{self.group}',
            )

        # uc_pred_mean_scaled: n_batch x n_outputs
        uc_pred_scaled = tf.keras.layers.Concatenate(
            axis=2,
            name=f'uc_pred_scaled_{self.group}'
            )((
                uc_pred_mean_scaled[:, :, tf.newaxis],
                uc_pred_alpha_scaled[:, :, tf.newaxis],
                uc_pred_beta_scaled[:, :, tf.newaxis],
                ))
        return uc_pred_scaled

    def model_builder_mlp_head(self, inputs):
        x_head = inputs['q2_scaled']
        for layer_index in range(len(self.model_params['head_params']['layers'])):
            x_head = tf.keras.layers.Dense(
                self.model_params['head_params']['layers'][layer_index],
                activation='linear',
                name=f'dense_head_{layer_index}',
                )(x_head)
            x_head = tf.keras.layers.LayerNormalization(
                epsilon=self.model_params['head_params']['epsilon'], 
                name=f'layer_norm_head_{layer_index}'
                )(x_head)
            x_head = tf.keras.activations.gelu(x_head)
            x_head = tf.keras.layers.Dropout(
                rate=self.model_params['head_params']['dropout_rate'],
                name=f'dropout_head_{layer_index}',
                )(x_head)

        uc_pred_mean_scaled = mlp_model_builder(
            x_head,
            tag=f'uc_pred_mean_scaled_{self.group}',
            model_params=self.model_params['mean_params'],
            output_name=f'{self.model_params["mean_params"]["output_name"]}_{self.group}',
            )
        uc_pred_alpha_scaled = 1 + mlp_model_builder(
            x_head,
            tag=f'uc_pred_alpha_scaled_{self.group}',
            model_params=self.model_params['alpha_params'],
            output_name=f'{self.model_params["alpha_params"]["output_name"]}_{self.group}',
            )
        uc_pred_beta_scaled = mlp_model_builder(
            x_head,
            tag=f'uc_pred_beta_scaled_{self.group}',
            model_params=self.model_params['beta_params'],
            output_name=f'{self.model_params["beta_params"]["output_name"]}_{self.group}',
            )

        # uc_pred_mean_scaled: n_batch x n_outputs
        uc_pred_scaled = tf.keras.layers.Concatenate(
            axis=2,
            name=f'uc_pred_scaled_{self.group}'
            )((
                uc_pred_mean_scaled[:, :, tf.newaxis],
                uc_pred_alpha_scaled[:, :, tf.newaxis],
                uc_pred_beta_scaled[:, :, tf.newaxis],
                ))
        return uc_pred_scaled

    def model_builder_bnn(self, inputs):
        uc_pred_mean_scaled = bnn_model_builder(
            inputs['q2_scaled'],
            tag=f'uc_pred_mean_scaled_{self.group}',
            model_params=self.model_params['mean_params'],
            output_name=f'{self.model_params["mean_params"]["output_name"]}_{self.group}',
            N_train=self.model_params['N_train']
            )
        uc_pred_alpha_scaled = 1 + bnn_model_builder(
            inputs['q2_scaled'],
            tag=f'uc_pred_alpha_scaled_{self.group}',
            model_params=self.model_params['alpha_params'],
            output_name=f'{self.model_params["alpha_params"]["output_name"]}_{self.group}',
            N_train=self.model_params['N_train']
            )
        uc_pred_beta_scaled = bnn_model_builder(
            inputs['q2_scaled'],
            tag=f'uc_pred_beta_scaled_{self.group}',
            model_params=self.model_params['beta_params'],
            output_name=f'{self.model_params["beta_params"]["output_name"]}_{self.group}',
            N_train=self.model_params['N_train']
            )

        # uc_pred_mean_scaled: n_batch x n_outputs
        uc_pred_scaled = tf.keras.layers.Concatenate(
            axis=2,
            name=f'uc_pred_scaled_{self.group}'
            )((
                uc_pred_mean_scaled[:, :, tf.newaxis],
                uc_pred_alpha_scaled[:, :, tf.newaxis],
                uc_pred_beta_scaled[:, :, tf.newaxis],
                ))
        return uc_pred_scaled

    def model_builder_mlp(self, inputs):
        uc_pred_mean_scaled = mlp_model_builder(
            inputs['q2_scaled'],
            tag=f'uc_pred_mean_scaled_{self.group}',
            model_params=self.model_params['mean_params'],
            output_name=f'{self.model_params["mean_params"]["output_name"]}_{self.group}'
            )
        uc_pred_alpha_scaled = 1 + mlp_model_builder(
            inputs['q2_scaled'],
            tag=f'uc_pred_alpha_scaled_{self.group}',
            model_params=self.model_params['alpha_params'],
            output_name=f'{self.model_params["alpha_params"]["output_name"]}_{self.group}'
            )
        uc_pred_beta_scaled = mlp_model_builder(
            inputs['q2_scaled'],
            tag=f'uc_pred_beta_scaled_{self.group}',
            model_params=self.model_params['beta_params'],
            output_name=f'{self.model_params["beta_params"]["output_name"]}_{self.group}'
            )

        # uc_pred_mean_scaled: n_batch x n_outputs
        uc_pred_scaled = tf.keras.layers.Concatenate(
            axis=2,
            name=f'uc_pred_scaled_{self.group}'
            )((
                uc_pred_mean_scaled[:, :, tf.newaxis],
                uc_pred_alpha_scaled[:, :, tf.newaxis],
                uc_pred_beta_scaled[:, :, tf.newaxis],
                ))
        return uc_pred_scaled

    def get_layer_names(self):
        self.mean_layer_names = []
        for layer_index in range(len(self.model_params['mean_params']['layers'])):
            self.mean_layer_names.append(f'dense_uc_pred_mean_scaled_{self.group}_{layer_index}')
            self.mean_layer_names.append(f'layer_norm_uc_pred_mean_scaled_{self.group}_{layer_index}')
        self.mean_layer_names.append(f'{self.model_params["mean_params"]["output_name"]}_{self.group}')
        if self.model_params['nn_type'] in ['mlp_head']:
            for layer_index in range(len(self.model_params['head_params']['layers'])):
                self.mean_layer_names.append(f'dense_head_{layer_index}')
                self.mean_layer_names.append(f'layer_norm_head_{layer_index}')
        if self.model_params['nn_type'] in ['rnn_head']:
            self.mean_layer_names.append(f'head_{self.group}')
            self.mean_layer_names.append(f'head_{self.group}_attn')

        self.var_layer_names = []
        for layer_index in range(len(self.model_params['alpha_params']['layers'])):
            self.var_layer_names.append(f'dense_uc_pred_alpha_scaled_{self.group}_{layer_index}')
            self.var_layer_names.append(f'layer_norm_uc_pred_alpha_scaled_{self.group}_{layer_index}')
        self.var_layer_names.append(f'{self.model_params["alpha_params"]["output_name"]}_{self.group}')
        for layer_index in range(len(self.model_params['beta_params']['layers'])):
            self.var_layer_names.append(f'dense_uc_pred_beta_scaled_{self.group}_{layer_index}')
            self.var_layer_names.append(f'layer_norm_uc_pred_beta_scaled_{self.group}_{layer_index}')
        self.var_layer_names.append(f'{self.model_params["beta_params"]["output_name"]}_{self.group}')

    def do_predictions_deterministic(self, data=None, inputs=None, verbose=1):
        N = len(data)
        if not data is None:
            inputs = {'q2_scaled': np.stack(data['q2_scaled'])}
        if verbose == 1:
            print(f'\n Regression inferences for {self.group}')
        outputs = self.model.predict(inputs, batch_size=4096, verbose=verbose)
        pred = outputs[:, :, 0]
        pred_alpha = outputs[:, :, 1]
        pred_beta = outputs[:, :, 2]
        pred_var = pred_beta / (pred_alpha - 1)
        if self.model_params['predict_pca']:
            uc_pred_scaled = self.pca.inverse_transform(pred)
            temp = np.matmul(np.sqrt(pred_var), self.pca.components_)
            uc_pred_scaled_cov = np.matmul(temp[:, :, np.newaxis], temp[:, np.newaxis, :])
            #uc_pred_scaled_cov = np.matmul(pred_var[:, np.newaxis, :], self.pca.components_)
        else:
            uc_pred_scaled = pred
            diag_indices = np.diag_indices(self.n_outputs, ndim=2)
            uc_pred_scaled_cov = np.zeros((N, self.n_outputs, self.n_outputs))
            for index in range(N):
                uc_pred_scaled_cov[index, diag_indices[0], diag_indices[1]] = pred_var[index]
        return uc_pred_scaled, uc_pred_scaled_cov

    def do_predictions_probabalistic(self, data=None, inputs=None, verbose=1, n_evals=500):
        N = len(data)
        if not data is None:
            inputs = {'q2_scaled': np.stack(data['q2_scaled'])}
        if verbose == 1:
            print(f'\n Regression inferences for {self.group}')

        uc_pred_scaled_all = np.zeros((N, self.n_outputs, n_evals))
        uc_pred_scaled_var_all = np.zeros((N, self.n_outputs, n_evals))
        for index in range(n_evals):
            outputs = self.model.predict(inputs, batch_size=N, verbose=0)
            uc_pred_scaled_all[:, :, index] = outputs[:, :, 0]
            uc_pred_scaled_alpha = outputs[:, :, 1]
            uc_pred_scaled_beta = outputs[:, :, 2]
            uc_pred_scaled_var_all[:, :, index] = uc_pred_scaled_beta / (uc_pred_scaled_alpha - 1)
        uc_pred_scaled = uc_pred_scaled_all.mean(axis=2)
        uc_pred_scaled_model_var = uc_pred_scaled_all.std(axis=2)**2
        uc_pred_scaled_var = uc_pred_scaled_var_all.mean(axis=2)
        diag_indices = np.diag_indices(self.n_outputs, ndim=2)
        uc_pred_scaled_cov = np.zeros((N, self.n_outputs, self.n_outputs))
        for index in range(N):
            uc_pred_scaled_cov[index, diag_indices[0], diag_indices[1]] = \
                uc_pred_scaled_var[index] + 1/N*uc_pred_scaled_model_var[index]
        return uc_pred_scaled, uc_pred_scaled_cov

