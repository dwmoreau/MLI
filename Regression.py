import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from IOManagers import read_params
from IOManagers import write_params
from IOManagers import SKLearnManager
from IOManagers import NeuralNetworkManager


class Regression:
    def __init__(self, group, data_params, model_params, save_to, uc_scaler, angle_scale, seed=12345):
        self.model_params = model_params
        self.n_peaks = data_params['n_peaks']
        self.unit_cell_length = data_params['unit_cell_length']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.lattice_system = data_params['lattice_system']
        self.group = group
        self.save_to = save_to
        self.uc_scaler = uc_scaler
        self.angle_scale = angle_scale
        self.seed = seed

    def train_regression(self, data):
        self.fit_trees(data)
        self.build_model()
        if self.model_params['fit_strategy'] == 'cycles':
            self.fit_model_cycles(data)
        elif self.model_params['fit_strategy'] == 'warmup':
            self.fit_model_warmup(data)
        train_inputs, val_inputs, train_true, val_true, train_weights = self._get_train_val(data)
        self.save(train_inputs)

    def build_model(self):
        import keras
        inputs = keras.Input(
            shape=(self.n_peaks,),
            name='q2_scaled',
            dtype='float32',
            )
        self.model = keras.Model(inputs, self.model_builder(inputs))
        #self.model.summary()
        self.get_layer_names()

    def compile_model(self, mode):
        import keras
        from TargetFunctions import LikelihoodLoss
        if mode == 'mean':
            self.reg_loss = LikelihoodLoss(
                self.model_params['target_function'],
                n=self.unit_cell_length,
                beta_nll=self.model_params['beta_nll']
                )
            mean_setting = True
            var_setting = False
            self.reg_loss.beta_likelihood = True
        elif mode == 'variance':
            self.reg_loss = LikelihoodLoss(
                self.model_params['variance_model'],
                n=self.unit_cell_length,
                beta_nll=self.model_params['beta_nll']
                )
            mean_setting = False
            var_setting = True
            self.reg_loss.beta_likelihood = False
        elif mode == 'both':
            self.reg_loss = LikelihoodLoss(
                self.model_params['variance_model'], 
                n=self.unit_cell_length,
                beta_nll=self.model_params['beta_nll']
                )
            mean_setting = True
            var_setting = True
            self.reg_loss.beta_likelihood = True

        self.loss_functions = {
            f'uc_pred_scaled_{self.group}': self.reg_loss
            }
        self.loss_weights = {
            f'uc_pred_scaled_{self.group}': 1
            }
        self.loss_metrics = {
            f'uc_pred_scaled_{self.group}': [self.reg_loss.mean_squared_error, self.reg_loss.log_cosh_error]
            }
        for layer_name in self.mean_layer_names:
            self.model.get_layer(layer_name).trainable = mean_setting
        for layer_name in self.var_layer_names:
            self.model.get_layer(layer_name).trainable = var_setting

        self.model.compile(
            optimizer=keras.optimizers.Adam(self.model_params['learning_rate']), 
            loss=self.loss_functions,
            loss_weights=self.loss_weights,
            metrics=self.loss_metrics,
            run_eagerly=False,
            )

    def _get_train_val(self, data, limit_volume=False):
        train = data[data['train']]
        val = data[~data['train']]

        volume_train = np.stack(train['reindexed_volume'])
        volume_val = np.stack(val['reindexed_volume'])
        if limit_volume:
            # The large volume triclinic unit cells lead to an overflow error.
            # So the loss becomes nan
            max_volume = np.sort(volume_train)[int(0.95*volume_train.size)]
            train_indices = volume_train <= max_volume
            val_indices = volume_val <= max_volume
            volume_train = volume_train[train_indices]
            volume_val = volume_val[val_indices]
            train_inputs = np.stack(train['q2_scaled'])[train_indices]
            val_inputs = np.stack(val['q2_scaled'])[val_indices]
            train_true = np.stack(train['reindexed_unit_cell_scaled'])[train_indices][:, self.unit_cell_indices]
            val_true = np.stack(val['reindexed_unit_cell_scaled'])[val_indices][:, self.unit_cell_indices]

        else:
            train_inputs = np.stack(train['q2_scaled'])
            val_inputs = np.stack(val['q2_scaled'])
            train_true = np.stack(train['reindexed_unit_cell_scaled'])[:, self.unit_cell_indices]
            val_true = np.stack(val['reindexed_unit_cell_scaled'])[:, self.unit_cell_indices]

        volume_train_sorted = np.sort(volume_train)
        n_volume_bins = 25
        volume_bins = np.linspace(
            volume_train_sorted[int(0.01*volume_train_sorted.size)],
            volume_train_sorted[int(0.99*volume_train_sorted.size)],
            n_volume_bins + 1
            )
        volume_bins[0] = 0
        volume_bins[-1] = volume_train_sorted[-1]
        volume_bin_indices = np.searchsorted(volume_bins, volume_train) - 1
        volume_hist, _ = np.histogram(volume_train, bins=volume_bins, density=True)
        volume_bin_weights = np.ones(n_volume_bins)
        good = volume_hist > 0
        volume_bin_weights[good] = np.sqrt(1/volume_hist[good])
        volume_bin_weights /= volume_bin_weights[good].min()
        too_large = volume_bin_weights > 10
        volume_bin_weights[too_large] = 10
        volume_train_weights = volume_bin_weights[volume_bin_indices]
        return train_inputs, val_inputs, train_true, val_true, volume_train_weights

    def fit_trees(self, data):
        train_inputs, val_inputs, train_true, val_true, train_weights = self._get_train_val(data)
        if self.model_params['random_forest']['grid_search'] is not None:
            # param_grid input is a dictionary with the same names as the random forest model.
            # subsample needs to be correctly named 'max_samples'.
            param_grid = self.model_params['random_forest']['grid_search'].copy()
            subsample_value = param_grid['subsample']
            del param_grid['subsample']
            param_grid['max_samples'] = subsample_value

        if self.lattice_system == 'cubic':
            self.random_forest_regressor = RandomForestRegressor(
                random_state=self.model_params['random_forest']['random_state'],
                n_estimators=self.model_params['random_forest']['n_estimators'],
                min_samples_leaf=self.model_params['random_forest']['min_samples_leaf'],
                max_depth=self.model_params['random_forest']['max_depth'],
                max_samples=self.model_params['random_forest']['subsample'],
                )
            if self.model_params['random_forest']['grid_search'] is not None:
                # Set up GridSearchCV
                grid_search = GridSearchCV(
                    estimator=self.random_forest_regressor,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1,
                    verbose=1,
                    )
                grid_search.fit(train_inputs, train_true.ravel())
                best_params = grid_search.best_params_.copy()
                subsample_value = best_params['max_samples']
                del best_params['max_samples']
                best_params['subsample'] = subsample_value
                self.model_params['random_forest'].update(best_params)
                self.random_forest_regressor = grid_search.best_estimator_
            else:
                # Fit the RandomForestRegressor directly if no grid search is specified
                # f'uc_pred_scaled_{self.group}' in train_true refers to the NN layer that goes to the target
                # function. The actual values are 'reindexed_unit_cell_scaled'
                self.random_forest_regressor.fit(train_inputs, train_true.ravel())
        else:
            n_ratio_bins = self.model_params['random_forest']['n_dominant_zone_bins']
            train = data[data['train']]
            unit_cell_train = np.stack(train['reindexed_unit_cell'])

            if self.lattice_system == 'rhombohedral':
                # Ratio in rhombohedral is the cosine of the angle
                # angle is limited between 0 and 120 degrees or 1 and -1/2 (cos(alpha))
                ratio = np.cos(unit_cell_train[:, 3])
                ratio_sorted = np.sort(ratio)
                ratio_bins = [ratio_sorted[int(f*ratio_sorted.size)] for f in np.linspace(0, 0.999, n_ratio_bins + 1)]
                ratio_bins[0] = -1/2
                ratio_bins[-1] = 1
            else:
                # unit_cell_train is a N x 6 array. So the first three indices of the
                # first axis are unit cell magnitudes.
                ratio = unit_cell_train[:, :3].min(axis=1) / unit_cell_train[:, :3].max(axis=1)
                ratio_sorted = np.sort(ratio)
                ratio_bins = [ratio_sorted[int(f*ratio_sorted.size)] for f in np.linspace(0, 0.999, n_ratio_bins + 1)]
                ratio_bins[0] = 0
                ratio_bins[-1] = 1

            if self.model_params['random_forest']['grid_search']:
                grid_search = GridSearchCV(
                    estimator=RandomForestRegressor(
                        random_state=0, 
                        n_estimators=self.model_params['random_forest']['n_estimators'],
                        min_samples_leaf=self.model_params['random_forest']['min_samples_leaf'],
                        max_depth=self.model_params['random_forest']['max_depth'],
                        max_samples=self.model_params['random_forest']['subsample'],
                        ),
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=6,
                    verbose=1,
                    )
                indices_train = np.random.default_rng(0).choice(
                    train_inputs.shape[0],
                    size=int(train_inputs.shape[0]/n_ratio_bins),
                    replace=False
                    )
                grid_search.fit(
                    train_inputs[indices_train],
                    train_true[indices_train],
                    sample_weight=train_weights[indices_train]
                    )
                best_params = grid_search.best_params_.copy()
                subsample_value = best_params['max_samples']
                del best_params['max_samples']
                best_params['subsample'] = subsample_value
                self.model_params['random_forest'].update(best_params)

            self.random_forest_regressor = [
                RandomForestRegressor(
                    random_state=0, 
                    n_estimators=self.model_params['random_forest']['n_estimators'],
                    min_samples_leaf=self.model_params['random_forest']['min_samples_leaf'],
                    max_depth=self.model_params['random_forest']['max_depth'],
                    max_samples=self.model_params['random_forest']['subsample'],
                    ) for _ in range(n_ratio_bins)
                ]
            for ratio_index in range(n_ratio_bins):
                indices_train = np.logical_and(
                    ratio >= ratio_bins[ratio_index],
                    ratio < ratio_bins[ratio_index + 1]
                    )
                # f'uc_pred_scaled_{self.group}' in train_true refers to the NN layer that goes to the target
                # function. The true values are 'reindexed_unit_cell_scaled'
                self.random_forest_regressor[ratio_index].fit(
                    train_inputs[indices_train],
                    train_true[indices_train],
                    sample_weight=train_weights[indices_train]
                    )

    def fit_model_cycles(self, data):
        import keras
        self.fit_history = [None for _ in range(2 * self.model_params['cycles'])]
        if self.lattice_system == 'triclinic':
            limit_volume = True
        else:
            limit_volume = False
        train_inputs, val_inputs, train_true, val_true, train_weights = self._get_train_val(data, limit_volume)
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
                sample_weight=train_weights,
                )
            keras.backend.clear_session()
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
                sample_weight=train_weights,
                )
            # https://github.com/tensorflow/tensorflow/issues/37505
            # these are to deal with a memory leak.
            # This does not solve the issue
            keras.backend.clear_session()
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
        fig.savefig(os.path.join(
            f'{self.save_to}', 
            f'{self.group}_reg_training_loss_{self.model_params["tag"]}.png'
            ))
        plt.close()

    def fit_model_warmup(self, data):
        self.fit_history = [None for _ in range(3)]
        train_inputs, val_inputs, train_true, val_true, train_weights = self._get_train_val(data)

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
            sample_weight=train_weights
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
            sample_weight=train_weights
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
            sample_weight=train_weights
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
        fig.savefig(os.path.join(
            f'{self.save_to}',
            f'{self.group}_reg_training_loss_{self.model_params["tag"]}.png'
            ))
        plt.close()

    def setup(self):
        model_params_defaults = {
            'fit_strategy': 'cycles',
            'epochs': 15,
            'cycles': 3,
            'beta_nll': 0.5,
            'batch_size': 64,
            'learning_rate': 0.0005,
            'target_function': 'log_cosh',
            'variance_model': 'normal',
            'mean_params': {
                'layers': [200, 100, 60],
                'dropout_rate': 0.1,
                'epsilon': 0.001,
                'output_activation': 'linear',
                'output_name': 'uc_mean_scaled',
                },
            'var_params': {
                'layers': [100, 60],
                'dropout_rate': 0.25,
                'epsilon': 0.001,
                'output_activation': 'softplus',
                'output_name': 'uc_var_scaled',
                },
            'alpha_params': {
                'layers': [100, 60],
                'dropout_rate': 0.25,
                'epsilon': 0.001,
                'output_activation': 'softplus',
                'output_name': 'uc_alpha_scaled',
                },
            'beta_params': {
                'layers': [100, 60],
                'dropout_rate': 0.25,
                'epsilon': 0.001,
                'output_activation': 'softplus',
                'output_name': 'uc_beta_scaled',
                },
            'random_forest': {
                'random_state': 0,
                'n_estimators': 200,
                'min_samples_leaf': 2,
                'max_depth': 10,
                'subsample': 0.05,
                'n_dominant_zone_bins': 10,
                'grid_search': None,
                },
            }
        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        self.model_params['mean_params']['unit_cell_length'] = self.unit_cell_length
        self.model_params['var_params']['unit_cell_length'] = self.unit_cell_length
        self.model_params['alpha_params']['unit_cell_length'] = self.unit_cell_length
        self.model_params['beta_params']['unit_cell_length'] = self.unit_cell_length
        if self.model_params['fit_strategy'] == 'cycles':
            if not 'epochs' in self.model_params.keys():
                self.model_params['epochs'] = 5
            if not 'cycles' in self.model_params.keys():
                self.model_params['cycles'] = 10
        if self.model_params['fit_strategy'] == 'warmup':
            if not 'epochs' in self.model_params.keys():
                self.model_params['epochs'] = [10, 10, 80]

        for key in model_params_defaults['mean_params'].keys():
            if key not in self.model_params['mean_params'].keys():
                self.model_params['mean_params'][key] = model_params_defaults['mean_params'][key]
        self.model_params['mean_params']['kernel_initializer'] = None
        self.model_params['mean_params']['bias_initializer'] = None

        for key in model_params_defaults['var_params'].keys():
            if key not in self.model_params['var_params'].keys():
                self.model_params['var_params'][key] = model_params_defaults['var_params'][key]

        for key in model_params_defaults['alpha_params'].keys():
            if key not in self.model_params['alpha_params'].keys():
                self.model_params['alpha_params'][key] = model_params_defaults['alpha_params'][key]

        for key in model_params_defaults['beta_params'].keys():
            if key not in self.model_params['beta_params'].keys():
                self.model_params['beta_params'][key] = model_params_defaults['beta_params'][key]

        for key in model_params_defaults['random_forest'].keys():
            if key not in self.model_params['random_forest'].keys():
                self.model_params['random_forest'][key] = model_params_defaults['random_forest'][key]

    def save(self, train_inputs):
        import keras
        model_params = copy.deepcopy(self.model_params)
        write_params(
            model_params,
            os.path.join(
                f'{self.save_to}',
                f'{self.group}_reg_params_{self.model_params["tag"]}.csv'
                )
            )
        model_manager = NeuralNetworkManager(
            model=self.model,
            model_name=f'{self.group}_reg_weights_{self.model_params["tag"]}',
            save_dir=self.save_to,
            )
        model_manager.save_keras_weights()
        model_manager.convert_to_onnx(
            example_inputs=train_inputs,
            input_signature=(keras.Input(
                shape=(self.n_peaks,),
                name='q2_scaled',
                dtype='float32',
                ),)
            )
        model_manager.quantize_onnx(
            method='dynamic',
            calibration_data=train_inputs
            )

        if self.lattice_system == 'cubic':
            model_manager = SKLearnManager(
                filename=os.path.join(
                    f'{self.save_to}',
                    f'{self.group}_random_forest_regressor'
                    ),
                model_type='custom'
                )
            model_manager.save(
                model=self.random_forest_regressor,
                n_features=train_inputs.shape[1],
                )
            model_manager._save_sklearn(
                model=self.random_forest_regressor,
                )
        else:
            for ratio_index in range(self.model_params['random_forest']['n_dominant_zone_bins']):
                model_manager = SKLearnManager(
                    filename=os.path.join(
                        f'{self.save_to}',
                        f'{self.group}_{ratio_index}_random_forest_regressor'
                        ),
                    model_type='custom'
                    )
                model_manager.save(
                    model=self.random_forest_regressor[ratio_index],
                    n_features=train_inputs.shape[1],
                    )
                model_manager._save_sklearn(
                    model=self.random_forest_regressor[ratio_index],
                    )
                
    def load_from_tag(self, mode):
        params = read_params(os.path.join(
            f'{self.save_to}',
            f'{self.group}_reg_params_{self.model_params["tag"]}.csv'
            ))
        params_keys = [
            'tag',
            'mean_params',
            'var_params',
            'alpha_params',
            'beta_params',
            'beta_nll',
            'batch_size',
            'epochs',
            'cycles',
            'learning_rate',
            'fit_strategy',
            'variance_model',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['beta_nll'] = float(params['beta_nll'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['fit_strategy'] = params['fit_strategy']
        self.model_params['variance_model'] = params['variance_model']
        if self.model_params['fit_strategy'] == 'cycles':
            self.model_params['epochs'] = int(params['epochs'])
            self.model_params['cycles'] = int(params['cycles'])
        if self.model_params['fit_strategy'] == 'warmup':
            self.model_params['epochs'] = np.array(
                params['epochs'].split('[')[1].split(']')[0].split(','),
                dtype=int
                )

        self.model_params['random_forest'] = {}
        for element in params['random_forest'].split('{')[1].split('}')[0].split(','):
            key = element.split(':')[0].split("'")[1]
            value = element.split(':')[1]
            if key in ['random_state', 'n_estimators', 'min_samples_leaf', 'n_dominant_zone_bins']:
                self.model_params['random_forest'][key] = int(value)
            elif key == 'subsample':
                self.model_params['random_forest'][key] = float(value)
            elif key == 'max_depth':
                if 'None' in value:
                    self.model_params['random_forest']['max_depth'] = 'None'
                else:
                    self.model_params['random_forest']['max_depth'] = int(value)

        if mode in ['training', 'inference:both', 'inference:trees']:
            if self.lattice_system == 'cubic':
                #self.random_forest_regressor = SKLearnManager(
                #    filename=os.path.join(
                #        f'{self.save_to}',
                #        f'{self.group}_random_forest_regressor'
                #        ),
                #    model_type='custom'
                #    )
                self.random_forest_regressor = SKLearnManager(
                    filename=os.path.join(
                        f'{self.save_to}',
                        f'{self.group}_random_forest_regressor'
                        ),
                    model_type='sklearn'
                    )
                self.random_forest_regressor.load()
            else:
                self.random_forest_regressor = []
                for ratio_index in range(self.model_params['random_forest']['n_dominant_zone_bins']):
                    """
                    model_manager = SKLearnManager(
                        filename=os.path.join(
                            f'{self.save_to}',
                            f'{self.group}_{ratio_index}_random_forest_regressor'
                            ),
                        model_type='custom'
                        )
                    model_manager.load()
                    self.random_forest_regressor.append(model_manager)
                    """
                    model_manager = SKLearnManager(
                        filename=os.path.join(
                            f'{self.save_to}',
                            f'{self.group}_{ratio_index}_random_forest_regressor'
                            ),
                        model_type='sklearn'
                        )
                    model_manager.load()
                    self.random_forest_regressor.append(model_manager)

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

        network_keys = ['mean_params', 'var_params', 'alpha_params', 'beta_params']
        for network_key in network_keys:
            self.model_params[network_key] = dict.fromkeys(params_keys)
            self.model_params[network_key]['unit_cell_length'] = self.unit_cell_length
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
                if key in ['output_activation', 'output_name']:
                    self.model_params[network_key][key] = value.replace(' ', '')
            self.model_params[network_key]['kernel_initializer'] = None
            self.model_params[network_key]['bias_initializer'] = None

        if mode in ['training']:
            self.build_model()
            self.model = model_manager.load_keras_model(self.model)
            self.compile_model(mode='both')
        if mode in ['training', 'inference:both', 'inference:nn']:
            model_manager = NeuralNetworkManager(
                model_name=f'{self.group}_reg_weights_{self.model_params["tag"]}',
                save_dir=self.save_to,
                )
            self.onnx_model = model_manager.load_onnx_model(quantized=True)

    def model_builder(self, inputs):
        import keras
        from Networks import mlp_model_builder
        self.model_params['var_params']['kernel_initializer'] = None
        self.model_params['var_params']['bias_initializer'] = \
            keras.initializers.RandomNormal(mean=-1, stddev=0.05, seed=self.seed)
        self.model_params['alpha_params']['kernel_initializer'] = None
        self.model_params['alpha_params']['bias_initializer'] = \
            keras.initializers.RandomNormal(mean=2, stddev=0.05, seed=self.seed)
        self.model_params['beta_params']['kernel_initializer'] = None
        self.model_params['beta_params']['bias_initializer'] = \
            keras.initializers.RandomNormal(mean=0.5, stddev=0.05, seed=self.seed)

        uc_pred_mean_scaled = mlp_model_builder(
            inputs,
            tag=f'uc_pred_mean_scaled_{self.group}',
            model_params=self.model_params['mean_params'],
            output_name=f'{self.model_params["mean_params"]["output_name"]}_{self.group}'
            )
        if self.model_params['variance_model'] == 'normal':
            uc_pred_var_scaled = mlp_model_builder(
                inputs,
                tag=f'uc_pred_var_scaled_{self.group}',
                model_params=self.model_params['var_params'],
                output_name=f'{self.model_params["var_params"]["output_name"]}_{self.group}'
                )
            # uc_pred_mean_scaled: n_batch x unit_cell_length
            uc_pred_scaled = keras.layers.Concatenate(
                axis=2,
                name=f'uc_pred_scaled_{self.group}'
                )((
                    keras.ops.expand_dims(uc_pred_mean_scaled, axis=2),
                    keras.ops.expand_dims(uc_pred_var_scaled, axis=2),
                    ))
        elif self.model_params['variance_model'] == 'alpha_beta':
            uc_pred_alpha_scaled = 1 + mlp_model_builder(
                inputs,
                tag=f'uc_pred_alpha_scaled_{self.group}',
                model_params=self.model_params['alpha_params'],
                output_name=f'{self.model_params["alpha_params"]["output_name"]}_{self.group}'
                )
            uc_pred_beta_scaled = mlp_model_builder(
                inputs,
                tag=f'uc_pred_beta_scaled_{self.group}',
                model_params=self.model_params['beta_params'],
                output_name=f'{self.model_params["beta_params"]["output_name"]}_{self.group}'
                )

            # uc_pred_mean_scaled: n_batch x unit_cell_length
            uc_pred_scaled = keras.layers.Concatenate(
                axis=2,
                name=f'uc_pred_scaled_{self.group}'
                )((
                    keras.ops.expand_dims(uc_pred_mean_scaled, axis=2),
                    keras.ops.expand_dims(uc_pred_alpha_scaled, axis=2),
                    keras.ops.expand_dims(uc_pred_beta_scaled, axis=2),
                    ))
        return uc_pred_scaled

    def get_layer_names(self):
        self.mean_layer_names = []
        for layer_index in range(len(self.model_params['mean_params']['layers'])):
            self.mean_layer_names.append(f'dense_uc_pred_mean_scaled_{self.group}_{layer_index}')
            self.mean_layer_names.append(f'layer_norm_uc_pred_mean_scaled_{self.group}_{layer_index}')
        self.mean_layer_names.append(f'{self.model_params["mean_params"]["output_name"]}_{self.group}')

        self.var_layer_names = []
        if self.model_params['variance_model'] == 'normal':
            for layer_index in range(len(self.model_params['var_params']['layers'])):
                self.var_layer_names.append(f'dense_uc_pred_var_scaled_{self.group}_{layer_index}')
                self.var_layer_names.append(f'layer_norm_uc_pred_var_scaled_{self.group}_{layer_index}')
            self.var_layer_names.append(f'{self.model_params["var_params"]["output_name"]}_{self.group}')
        elif self.model_params['variance_model'] == 'alpha_beta':
            for layer_index in range(len(self.model_params['alpha_params']['layers'])):
                self.var_layer_names.append(f'dense_uc_pred_alpha_scaled_{self.group}_{layer_index}')
                self.var_layer_names.append(f'layer_norm_uc_pred_alpha_scaled_{self.group}_{layer_index}')
            self.var_layer_names.append(f'{self.model_params["alpha_params"]["output_name"]}_{self.group}')
            for layer_index in range(len(self.model_params['beta_params']['layers'])):
                self.var_layer_names.append(f'dense_uc_pred_beta_scaled_{self.group}_{layer_index}')
                self.var_layer_names.append(f'layer_norm_uc_pred_beta_scaled_{self.group}_{layer_index}')
            self.var_layer_names.append(f'{self.model_params["beta_params"]["output_name"]}_{self.group}')

    def revert_predictions(self, uc_pred_scaled=None, uc_pred_scaled_var=None):
        if not uc_pred_scaled is None:
            uc_pred = np.zeros(uc_pred_scaled.shape)
            if self.lattice_system in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred = uc_pred_scaled * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
            elif self.lattice_system == 'monoclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3] = self.angle_scale * uc_pred_scaled[:, 3] + np.pi/2
            elif self.lattice_system == 'triclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3:] = self.angle_scale * uc_pred_scaled[:, 3:] + np.pi/2
            elif self.lattice_system == 'rhombohedral':
                uc_pred[:, 0] = uc_pred_scaled[:, 0] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 1] = self.angle_scale * uc_pred_scaled[:, 1] + np.pi/2

        if not uc_pred_scaled_var is None:
            assert not uc_pred_scaled is None
            uc_pred_var = np.zeros(uc_pred_scaled_var.shape)
            if self.lattice_system in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred_var = uc_pred_scaled_var * self.uc_scaler.scale_[0]**2
            elif self.lattice_system == 'monoclinic':
                uc_pred_var[:, :3] = uc_pred_scaled_var[:, :3] * self.uc_scaler.scale_[0]**2
                uc_pred_var[:, 3] = uc_pred_scaled_var[:, 3] * self.angle_scale**2
            elif self.lattice_system == 'triclinic':
                uc_pred_var[:, :3] = uc_pred_scaled_var[:, :3] * self.uc_scaler.scale_[0]**2
                uc_pred_var[:, 3:] = uc_pred_scaled_var[:, 3:] * self.angle_scale**2
            elif self.lattice_system == 'rhombohedral':
                uc_pred_var[:, 0] = uc_pred_scaled_var[:, 0] * self.uc_scaler.scale_[0]**2
                uc_pred_var[:, 1] = uc_pred_scaled_var[:, 1] * self.angle_scale**2

        if not uc_pred_scaled is None and not uc_pred_scaled_var is None:
            return uc_pred, uc_pred_var
        elif not uc_pred_scaled is None and uc_pred_scaled_var is None:
            return uc_pred
        elif uc_pred_scaled is None and not uc_pred_scaled_var is None:
            return uc_pred_var

    def generate(self, n_unit_cells, rng, q2_obs, batch_size=None, model=None, q2_scaler=None):
        if model == 'nn':
            return self.generate_nn(n_unit_cells, rng, q2_obs[np.newaxis], batch_size, q2_scaler)
        elif model == 'trees':
            return self.generate_trees(n_unit_cells, rng, q2_obs[np.newaxis], q2_scaler)

    def generate_nn(self, n_unit_cells, rng, q2_obs, batch_size=None, q2_scaler=None):
        q2_scaled = (q2_obs - q2_scaler.mean_[0]) / q2_scaler.scale_[0]
        uc_pred, uc_pred_var = self.predict(q2_scaled=q2_scaled, batch_size=batch_size, model='nn')
        generated_unit_cells = rng.normal(
            loc=uc_pred,
            scale=np.sqrt(uc_pred_var),
            size=(n_unit_cells, self.unit_cell_length),
            )
        return generated_unit_cells

    def generate_trees(self, n_unit_cells, rng, q2_obs, q2_scaler=None):
        q2_scaled = (q2_obs - q2_scaler.mean_[0]) / q2_scaler.scale_[0]
        _, _, generated_unit_cells = self.predict(q2_scaled=q2_scaled, model='trees')
        # generated_unit_cells: n_entries, unit_cell_length, n_estimators
        # Expected output: n_estimators, unit_cell_length
        # We are only generating for one entry at a time so get the 0th element at the first axis
        if n_unit_cells <= generated_unit_cells.shape[2]:
            indices = rng.choice(generated_unit_cells.shape[2], size=n_unit_cells, replace=False)
            generated_unit_cells = generated_unit_cells[0, :, indices]
        else:
            # If more unit cells are requested than estimators, use the mean and covariance
            # of the estimators predictions to randomly generate unit cells
            if self.lattice_system == 'cubic':
                random_unit_cells = rng.normal(
                    loc=generated_unit_cells[0].mean(),
                    scale=generated_unit_cells[0].std(),
                    size=n_unit_cells - generated_unit_cells.shape[2]
                    )[:, np.newaxis]
            else:
                random_unit_cells = rng.multivariate_normal(
                    mean=generated_unit_cells[0].mean(axis=1),
                    cov=np.cov(generated_unit_cells[0], rowvar=True),
                    size=n_unit_cells - generated_unit_cells.shape[2]
                    )
            generated_unit_cells = np.concatenate((
                generated_unit_cells[0].T, random_unit_cells
                ), axis=0)
        return generated_unit_cells

    def predict(self, data=None, inputs=None, q2_scaled=None, batch_size=None, model=None):
        if not data is None:
            q2_scaled = np.stack(data['q2_scaled'])
        elif not inputs is None:
            q2_scaled = inputs['q2_scaled']

        if model == 'nn':
            uc_pred_scaled, uc_pred_scaled_var = self.predict_nn(q2_scaled, batch_size=None)
        elif model == 'trees':
            uc_pred_scaled, uc_pred_scaled_var, uc_pred_scaled_trees = self.predict_trees(q2_scaled)
            uc_pred_trees = self.revert_predictions(uc_pred_scaled=uc_pred_scaled_trees)
        uc_pred, uc_pred_var = self.revert_predictions(
            uc_pred_scaled=uc_pred_scaled, uc_pred_scaled_var=uc_pred_scaled_var
            )
        if model == 'nn':
            return uc_pred, uc_pred_var
        elif model == 'trees':
            return uc_pred, uc_pred_var, uc_pred_trees

    def predict_trees(self, q2_scaled):
        N = q2_scaled.shape[0]
        if self.lattice_system == 'cubic':
            uc_pred_scaled_tree = self.random_forest_regressor.predict_individual_trees(
                q2_scaled, n_outputs=self.unit_cell_length
                )
        else:
            uc_pred_scaled_tree = np.zeros((
                N,
                self.unit_cell_length,
                self.model_params['random_forest']['n_dominant_zone_bins']*self.model_params['random_forest']['n_estimators']
                ))
            for ratio_index in range(self.model_params['random_forest']['n_dominant_zone_bins']):
                start = ratio_index * self.model_params['random_forest']['n_estimators']
                stop = (ratio_index + 1) * self.model_params['random_forest']['n_estimators']
                uc_pred_scaled_tree[:, :, start: stop] = self.random_forest_regressor[ratio_index].predict_individual_trees(
                    q2_scaled, n_outputs=self.unit_cell_length
                    )
            """
            # This does a prediction with the onnx model
            # This feature is broekn right now
            uc_pred_scaled_tree = np.zeros((
                N,
                self.unit_cell_length,
                self.model_params['random_forest']['n_dominant_zone_bins']*self.model_params['random_forest']['n_estimators']
                ))
            for ratio_index in range(self.model_params['random_forest']['n_dominant_zone_bins']):
                start = ratio_index * self.model_params['random_forest']['n_estimators']
                stop = (ratio_index + 1) * self.model_params['random_forest']['n_estimators']
                uc_pred_scaled_tree[:, :, start: stop] = self.random_forest_regressor[ratio_index].predict_individual_trees(
                    q2_scaled,
                    )
            """

        uc_pred_scaled = uc_pred_scaled_tree.mean(axis=2)
        uc_pred_scaled_var = uc_pred_scaled_tree.std(axis=2)**2
        return uc_pred_scaled, uc_pred_scaled_var, uc_pred_scaled_tree

    def predict_nn(self, q2_scaled, batch_size=None):
        if batch_size is None:
            batch_size = self.model_params['batch_size']

        # predict_on_batch helps with a memory leak...
        N = q2_scaled.shape[0]
        uc_pred_scaled = np.zeros((N, self.unit_cell_length))
        uc_pred_scaled_var = np.zeros((N, self.unit_cell_length))
        for index in range(N):
            outputs = self.onnx_model.run(
                None,
                {'input': q2_scaled[index].astype(np.float32)[np.newaxis]}
                )[0]

            if self.model_params['variance_model'] == 'normal':
                uc_pred_scaled[index] = outputs[0, :, 0]
                uc_pred_scaled_var[index] = outputs[0, :, 1]
            elif self.model_params['variance_model'] == 'alpha_beta':
                uc_pred_scaled[index] = outputs[0, :, 0]
                pred_alpha = outputs[0, :, 1]
                pred_beta = outputs[0, :, 2]
                uc_pred_scaled_var[index] = pred_beta / (pred_alpha - 1)

        # [0.43981636] [0.20755097] <- Original Keras model
        # [0.43981636] [0.20755097] <- loading Keras model through manager
        # [0.43981636] [0.20755097] <- onnx model
        # [0.433568]   [0.20709202] <- int8 quantized model
        return uc_pred_scaled, uc_pred_scaled_var

    def predict_nn_old(self, q2_scaled, batch_size=None):
        '''
        This performs neural network predictions using the Keras model.
        Switching to inferences performed with the int8 quantized model.
        '''
        if batch_size is None:
            batch_size = self.model_params['batch_size']

        # predict_on_batch helps with a memory leak...
        N = q2_scaled.shape[0]
        n_batches = N // batch_size
        left_over = N % batch_size
        uc_pred_scaled = np.zeros((N, self.unit_cell_length))
        uc_pred_scaled_var = np.zeros((N, self.unit_cell_length))
        for batch_index in range(n_batches + 1):
            start = batch_index * batch_size
            if batch_index == n_batches:
                batch_inputs = np.zeros((batch_size, self.n_peaks))
                batch_inputs[:left_over] = q2_scaled[start: start + left_over]
                batch_inputs[left_over:] = q2_scaled[0]
            else:
                batch_inputs = q2_scaled[start: start + batch_size]

            outputs = self.model.predict_on_batch(batch_inputs)

            if self.model_params['variance_model'] == 'normal':
                if batch_index == n_batches:
                    uc_pred_scaled[start:] = outputs[:left_over, :, 0]
                    uc_pred_scaled_var[start:] = outputs[:left_over, :, 1]
                else:
                    uc_pred_scaled[start: start + batch_size] = outputs[:, :, 0]
                    uc_pred_scaled_var[start: start + batch_size] = outputs[:, :, 1]
            elif self.model_params['variance_model'] == 'alpha_beta':
                if batch_index == n_batches:
                    uc_pred_scaled[start:] = outputs[:left_over, :, 0]
                    pred_alpha = outputs[:left_over, :, 1]
                    pred_beta = outputs[:left_over, :, 2]
                    uc_pred_scaled_var[start:] = pred_beta / (pred_alpha - 1)
                else:
                    uc_pred_scaled[start: start + batch_size] = outputs[:, :, 0]
                    pred_alpha = outputs[:, :, 1]
                    pred_beta = outputs[:, :, 2]
                    uc_pred_scaled_var[start: start + batch_size] = pred_beta / (pred_alpha - 1)

        return uc_pred_scaled, uc_pred_scaled_var
