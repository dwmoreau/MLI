import copy
import gc
import joblib
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import tensorflow as tf

from Networks import mlp_model_builder
from TargetFunctions import LikelihoodLoss
from Utilities import read_params
from Utilities import write_params


class RegressionBase:
    def __init__(self, group, data_params, model_params, save_to, seed):
        self.model_params = model_params
        self.n_points = data_params['n_points']
        self.n_outputs = data_params['n_outputs']
        self.y_indices = data_params['y_indices']
        self.group = group
        self.save_to = save_to
        self.seed = seed

    def train_regression(self, data):
        if self.model_params['fit_trees']:
            self.fit_trees(data)
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
            unit_cell_scaled_train = np.stack(train['reindexed_unit_cell_scaled'])[:, self.y_indices]
            unit_cell_scaled_val = np.stack(val['reindexed_unit_cell_scaled'])[:, self.y_indices]
            self.pca = PCA(n_components=self.n_outputs).fit(unit_cell_scaled_train)
            train_true = {
                f'uc_pred_scaled_{self.group}': self.pca.transform(unit_cell_scaled_train)
                }
            val_true = {
                f'uc_pred_scaled_{self.group}': self.pca.transform(unit_cell_scaled_val)
                }
        else:
            train_true = {
                f'uc_pred_scaled_{self.group}': np.stack(train['reindexed_unit_cell_scaled'])[:, self.y_indices],
                }
            val_true = {
                f'uc_pred_scaled_{self.group}': np.stack(val['reindexed_unit_cell_scaled'])[:, self.y_indices],
                }
        return train_inputs, val_inputs, train_true, val_true

    def fit_trees(self, data):
        train_inputs, val_inputs, train_true, val_true = self._get_train_val(data)
        self.random_forest_regressor = RandomForestRegressor(
            random_state=self.model_params['random_forest']['random_state'],
            n_estimators=self.model_params['random_forest']['n_estimators'],
            min_samples_leaf=self.model_params['random_forest']['min_samples_leaf'],
            max_depth=self.model_params['random_forest']['max_depth'],
            max_samples=self.model_params['random_forest']['subsample'],
            )
        # f'uc_pred_scaled_{self.group}' in train_true refers to the NN layer that goes to the target
        # function. The true values are 'reindexed_unit_cell_scaled'
        self.random_forest_regressor.fit(train_inputs['q2_scaled'], train_true[f'uc_pred_scaled_{self.group}'])

    def do_predictions_trees(self, data=None, inputs=None, q2_scaled=None):
        if not data is None:
            q2_scaled = np.stack(data['q2_scaled'])
        elif not inputs is None:
            q2_scaled = inputs['q2_scaled']
        N = q2_scaled.shape[0]
        uc_pred_scaled = self.random_forest_regressor.predict(q2_scaled)
        if self.n_outputs == 1:
            uc_pred_scaled = uc_pred_scaled[:, np.newaxis]
        uc_pred_scaled_tree = np.zeros((N, self.n_outputs, self.model_params['random_forest']['n_estimators']))
        for tree in range(self.model_params['random_forest']['n_estimators']):
            prediction = self.random_forest_regressor.estimators_[tree].predict(q2_scaled)
            if self.n_outputs == 1:
                uc_pred_scaled_tree[:, :, tree] = prediction[:, np.newaxis]
            else:
                uc_pred_scaled_tree[:, :, tree] = prediction
        uc_pred_scaled_var = uc_pred_scaled_tree.std(axis=2)**2

        diag_indices = np.diag_indices(self.n_outputs, ndim=2)
        uc_pred_scaled_cov = np.zeros((N, self.n_outputs, self.n_outputs))
        for index in range(N):
            uc_pred_scaled_cov[index, diag_indices[0], diag_indices[1]] = uc_pred_scaled_var[index]
        return uc_pred_scaled, uc_pred_scaled_cov, uc_pred_scaled_tree

    def fit_model_cycles(self, data):
        self.fit_history = [None for _ in range(2 * self.model_params['cycles'])]
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
        self.fit_history = [None for _ in range(3)]
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
            else:
                train_label = None
                val_label = None
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


class Regression_AlphaBeta(RegressionBase):
    def __init__(self, group, data_params, model_params, save_to, seed=12345):
        super().__init__(group, data_params, model_params, save_to, seed)

    def setup(self):
        model_params_defaults = {
            'nn_type': 'mlp',
            'fit_strategy': 'cycles',
            'predict_pca': False,
            'fit_trees': True,
            'epochs': 10,
            'cycles': 5,
            'beta_nll': 0.5,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'dropout_rate': 0.5,
            'mean_params': {
                'layers': [200, 100, 60],
                },
            'alpha_params': {
                'layers': [100, 60],
                },
            'beta_params': {
                'layers': [100, 60],
                },
            'head_params': {
                'layers': [100]
                },
            'random_forest': {},
            }
        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        self.model_params['mean_params']['n_outputs'] = self.n_outputs
        self.model_params['alpha_params']['n_outputs'] = self.n_outputs
        self.model_params['beta_params']['n_outputs'] = self.n_outputs
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
        random_forest_defaults = {
            'random_state': 0,
            'n_estimators': 80,
            'min_samples_leaf': 10,
            'max_depth': None,
            'subsample': 0.1,
            }
        for key in random_forest_defaults.keys():
            if key not in self.model_params['random_forest'].keys():
                self.model_params['random_forest'][key] = random_forest_defaults[key]

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
        write_params(model_params, f'{self.save_to}/{self.group}_reg_params_{self.model_params["tag"]}.csv')
        self.model.save_weights(f'{self.save_to}/{self.group}_reg_weights_{self.model_params["tag"]}.h5')

        if self.model_params['predict_pca']:
            joblib.dump(self.pca, f'{self.save_to}/{self.group}_pca.bin')
        if self.model_params['fit_trees']:
            joblib.dump(
                self.random_forest_regressor,
                f'{self.save_to}/{self.group}_random_forest_regressor.bin'
                )

    def load_from_tag(self):
        params = read_params(f'{self.save_to}/{self.group}_reg_params_{self.model_params["tag"]}.csv')
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
            'fit_trees',
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

        self.model_params['random_forest'] = {}
        for element in params['random_forest'].split('{')[1].split('}')[0].split(','):
            key = element.split(':')[0].split("'")[1]
            value = element.split(':')[1]
            if key in ['random_state', 'n_estimators', 'min_samples_leaf']:
                self.model_params['random_forest'][key] = int(value)
            elif key == 'subsample':
                self.model_params['random_forest'][key] = float(value)
            elif key == 'max_depth':
                if 'None' in value:
                    self.model_params['random_forest']['max_depth'] = 'None'
                else:
                    self.model_params['random_forest']['max_depth'] = int(value)
        if params['fit_trees'] == 'True':
            self.model_params['fit_trees'] = True
            self.random_forest_regressor = joblib.load(f'{self.save_to}/{self.group}_random_forest_regressor.bin')
        elif params['fit_trees'] == 'False':
            self.model_params['fit_trees'] = False

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
                if network_key != 'head_params':
                    if key in ['output_activation', 'output_name']:
                        self.model_params[network_key][key] = value.replace(' ', '')
            self.model_params[network_key]['kernel_initializer'] = None
            self.model_params[network_key]['bias_initializer'] = None

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

    def do_predictions(self, data=None, inputs=None, verbose=1, batch_size=None):
        if not data is None:
            q2_scaled = np.stack(data['q2_scaled'])
        else:
            q2_scaled = inputs['q2_scaled']
        if verbose == 1:
            print(f'\n Regression inferences for {self.group}')
        if batch_size is None:
            batch_size = self.model_params['batch_size']

        # predict_on_batch helps with a memory leak...
        N = len(data)
        n_batches = N // batch_size
        left_over = N % batch_size
        pred = np.zeros((N, self.n_outputs))
        pred_var = np.zeros((N, self.n_outputs))
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
                pred[start:] = outputs[:left_over, :, 0]
                pred_alpha = outputs[:left_over, :, 1]
                pred_beta = outputs[:left_over, :, 2]
                pred_var[start:] = pred_beta / (pred_alpha - 1)
            else:
                pred[start: start + batch_size] = outputs[:, :, 0]
                pred_alpha = outputs[:, :, 1]
                pred_beta = outputs[:, :, 2]
                pred_var[start: start + batch_size] = pred_beta / (pred_alpha - 1)

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
