import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from mlindex.utilities.IOManagers import read_params
from mlindex.utilities.IOManagers import write_params
from mlindex.utilities.IOManagers import SKLearnManager


class RandomForest:
    def __init__(self, group, data_params, model_params, save_to, seed=12345):
        self.model_params = model_params
        self.n_peaks = data_params['n_peaks']
        self.unit_cell_length = data_params['unit_cell_length']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.lattice_system = data_params['lattice_system']
        self.group = group
        self.save_to = save_to
        self.seed = seed

    def train(self, data):
        self.fit(data)
        self.save()

    def _get_train_val(self, data):
        train = data[data['train']]
        val = data[~data['train']]

        volume_train = np.stack(train['reindexed_volume'])
        volume_val = np.stack(val['reindexed_volume'])

        train_inputs = np.stack(train['q2'])
        val_inputs = np.stack(val['q2'])
        train_true = np.stack(train['reindexed_unit_cell'])[:, self.unit_cell_indices]
        val_true = np.stack(val['reindexed_unit_cell'])[:, self.unit_cell_indices]

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

    def fit(self, data):
        train_inputs, val_inputs, train_true, val_true, train_weights = self._get_train_val(data)
        if self.model_params['grid_search'] is not None:
            # param_grid input is a dictionary with the same names as the random forest model.
            # subsample needs to be correctly named 'max_samples'.
            param_grid = self.model_params['grid_search'].copy()
            subsample_value = param_grid['subsample']
            del param_grid['subsample']
            param_grid['max_samples'] = subsample_value

        if self.lattice_system == 'cubic':
            self.random_forest_regressor = RandomForestRegressor(
                random_state=self.model_params['random_state'],
                n_estimators=self.model_params['n_estimators'],
                min_samples_leaf=self.model_params['min_samples_leaf'],
                max_depth=self.model_params['max_depth'],
                max_samples=self.model_params['subsample'],
                )
            if self.model_params['grid_search'] is not None:
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
                self.model_params.update(best_params)
                self.random_forest_regressor = grid_search.best_estimator_
            else:
                # Fit the RandomForestRegressor directly if no grid search is specified
                # f'uc_pred_{self.group}' in train_true refers to the NN layer that goes to the target
                # function. The actual values are 'reindexed_unit_cell'
                self.random_forest_regressor.fit(train_inputs, train_true.ravel())
        else:
            n_ratio_bins = self.model_params['n_dominant_zone_bins']
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

            if self.model_params['grid_search']:
                grid_search = GridSearchCV(
                    estimator=RandomForestRegressor(
                        random_state=0, 
                        n_estimators=self.model_params['n_estimators'],
                        min_samples_leaf=self.model_params['min_samples_leaf'],
                        max_depth=self.model_params['max_depth'],
                        max_samples=self.model_params['subsample'],
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
                self.model_params.update(best_params)

            self.random_forest_regressor = [
                RandomForestRegressor(
                    random_state=0, 
                    n_estimators=self.model_params['n_estimators'],
                    min_samples_leaf=self.model_params['min_samples_leaf'],
                    max_depth=self.model_params['max_depth'],
                    max_samples=self.model_params['subsample'],
                    ) for _ in range(n_ratio_bins)
                ]
            for ratio_index in range(n_ratio_bins):
                indices_train = np.logical_and(
                    ratio >= ratio_bins[ratio_index],
                    ratio < ratio_bins[ratio_index + 1]
                    )
                # f'uc_pred_{self.group}' in train_true refers to the NN layer that goes to the target
                # function. The true values are 'reindexed_unit_cell'
                self.random_forest_regressor[ratio_index].fit(
                    train_inputs[indices_train],
                    train_true[indices_train],
                    sample_weight=train_weights[indices_train]
                    )

    def setup(self):
        model_params_defaults = {
            'random_state': 0,
            'n_estimators': 200,
            'min_samples_leaf': 2,
            'max_depth': 10,
            'subsample': 0.05,
            'n_dominant_zone_bins': 10,
            'grid_search': None,
            }
        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]

    def save(self):
        model_params = copy.deepcopy(self.model_params)
        write_params(
            model_params,
            os.path.join(
                f'{self.save_to}',
                f'{self.group}_reg_params_{self.model_params["tag"]}.csv'
                )
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
                n_features=self.unit_cell_length,
                )
            model_manager._save_sklearn(
                model=self.random_forest_regressor,
                )
        else:
            for ratio_index in range(self.model_params['n_dominant_zone_bins']):
                model_manager = SKLearnManager(
                    filename=os.path.join(
                        f'{self.save_to}',
                        f'{self.group}_{ratio_index}_random_forest_regressor'
                        ),
                    model_type='custom'
                    )
                model_manager.save(
                    model=self.random_forest_regressor[ratio_index],
                    n_features=self.unit_cell_length,
                    )
                model_manager._save_sklearn(
                    model=self.random_forest_regressor[ratio_index],
                    )
                
    def load_from_tag(self):
        params = read_params(os.path.join(
            f'{self.save_to}',
            f'{self.group}_reg_params_{self.model_params["tag"]}.csv'
            ))
        params_keys = [
            'random_state',
            'n_estimators',
            'min_samples_leaf',
            'max_depth',
            'subsample',
            'n_dominant_zone_bins',
            'grid_search',
            ]
        self.model_params = dict.fromkeys(params_keys)
        for key in params_keys:
            if key in ['random_state', 'n_estimators', 'min_samples_leaf', 'n_dominant_zone_bins']:
                self.model_params[key] = int(params[key])
            elif key == 'subsample':
                self.model_params[key] = float(params[key])
            elif key == 'max_depth':
                if 'None' in params[key]:
                    self.model_params['max_depth'] = None
                else:
                    self.model_params['max_depth'] = int(params[key])

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
            for ratio_index in range(self.model_params['n_dominant_zone_bins']):
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

    def generate(self, n_unit_cells, rng, q2):
        _, _, generated_unit_cells = self.predict(q2=q2)
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

    def predict(self, data=None, inputs=None, q2=None):
        if not data is None:
            q2 = np.stack(data['q2'])
        elif not inputs is None:
            q2 = inputs['q2']

        N = q2.shape[0]
        if self.lattice_system == 'cubic':
            uc_pred_tree = self.random_forest_regressor.predict_individual_trees(
                q2, n_outputs=self.unit_cell_length
                )
        else:
            uc_pred_tree = np.zeros((
                N,
                self.unit_cell_length,
                self.model_params['n_dominant_zone_bins']*self.model_params['n_estimators']
                ))
            for ratio_index in range(self.model_params['n_dominant_zone_bins']):
                start = ratio_index * self.model_params['n_estimators']
                stop = (ratio_index + 1) * self.model_params['n_estimators']
                uc_pred_tree[:, :, start: stop] = self.random_forest_regressor[ratio_index].predict_individual_trees(
                    q2, n_outputs=self.unit_cell_length
                    )
            """
            # This does a prediction with the onnx model
            # This feature is broken right now
            uc_pred_tree = np.zeros((
                N,
                self.unit_cell_length,
                self.model_params['n_dominant_zone_bins']*self.model_params['n_estimators']
                ))
            for ratio_index in range(self.model_params['n_dominant_zone_bins']):
                start = ratio_index * self.model_params['n_estimators']
                stop = (ratio_index + 1) * self.model_params['n_estimators']
                uc_pred_tree[:, :, start: stop] = self.random_forest_regressor[ratio_index].predict_individual_trees(
                    q2,
                    )
            """

        uc_pred = uc_pred_tree.mean(axis=2)
        uc_pred_var = uc_pred_tree.std(axis=2)**2
        return uc_pred, uc_pred_var, uc_pred_tree

