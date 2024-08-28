import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Reindexing import reindex_entry_monoclinic
from Reindexing import reindex_entry_orthorhombic
from Reindexing import reindex_entry_triclinic
from Utilities import fix_unphysical
from Utilities import get_reciprocal_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import reciprocal_uc_conversion
from Utilities import read_params
from Utilities import write_params


class RandomGenerator:
    def __init__(self, bravais_lattice, data_params, model_params, save_to):
        self.lattice_system = data_params['lattice_system']
        self.unit_cell_length = data_params['unit_cell_length']
        self.bravais_lattice = bravais_lattice
        self.save_to = save_to
        self.model_params = model_params

    def setup(self):
        model_params_defaults = {
            'random_state': 0,
            'n_estimators': 100,
            'min_samples_leaf': 1,
            'max_depth': None,
            'subsample': 0.1,
            }
        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]

    def train(self, data):
        q2 = np.stack(data['q2'])
        # This is to correct for a bug in the dataset parsing.
        # This has been corrected and the following lines should be deleted when the 
        # dataset is regenerated next
        reciprocal_reindexed_unit_cell = np.stack(data['reciprocal_reindexed_unit_cell'])
        if np.sum(reciprocal_reindexed_unit_cell == 0) > 0:
            print('THERE ARE ZEROS IN THE RECIPROCAL REINDEXED UNIT CELLL')
            reindexed_unit_cell = np.stack(data['reindexed_unit_cell'])
            reciprocal_reindexed_unit_cell = reciprocal_uc_conversion(reindexed_unit_cell)
        vol = get_unit_cell_volume(reciprocal_reindexed_unit_cell)
        train = np.array(data['train'], dtype=bool)
            
        # Get limits for a continuous distribution of volume
        self.volume_lims = np.array([
            vol[train].min(),
            vol[train].max(),
            ])

        # Get an empirical unit cell volume
        bins = np.linspace(vol[train].min(), vol[train].max(), 101)
        centers = (bins[1:] + bins[:-1]) / 2
        pdf, _ = np.histogram(vol[train], bins=bins, density=True)
        cdf = np.cumsum(pdf) * (bins[1] - bins[0])
        self.volume_distribution = np.column_stack((centers, pdf, cdf))
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        axes.bar(centers, pdf, width=(bins[1] - bins[0]))
        axes_r = axes.twinx()
        axes_r.plot(centers, cdf, color=[0.8, 0, 0])
        axes.set_xlabel('Reciprocal Space Volume')
        axes.set_ylabel('Empirical PDF')
        axes_r.set_ylabel('Emprical CDF')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.bravais_lattice}_volume_distribution_{self.model_params["tag"]}.png')
        plt.close()

        # Random forest unit cell prediction
        print(f'\n  Training random forest for {self.bravais_lattice} volume prediction')
        self.random_forest_regressor = RandomForestRegressor(
            random_state=self.model_params['random_state'], 
            n_estimators=self.model_params['n_estimators'],
            min_samples_leaf=self.model_params['min_samples_leaf'],
            max_depth=self.model_params['max_depth'],
            max_samples=self.model_params['subsample'],
            )
        self.random_forest_regressor.fit(q2[train], vol[train])

        vol_pred_train = self.random_forest_regressor.predict(q2[train])
        vol_pred_val = self.random_forest_regressor.predict(q2[~train])

        rms_error_train = np.sqrt(np.mean((vol_pred_train - vol[train])**2))
        rms_error_val = np.sqrt(np.mean((vol_pred_val - vol[~train])**2))
        error_train = np.abs(vol_pred_train - vol[train])
        error_val = np.abs(vol_pred_val - vol[~train])
        preds = np.zeros((q2.shape[0], self.model_params['n_estimators']))
        for est_index in range(self.model_params['n_estimators']):
            preds[:, est_index] = self.random_forest_regressor.estimators_[est_index].predict(q2)
        std_train = preds[train].std(axis=1)
        std_val = preds[~train].std(axis=1)

        ms = 1
        alpha = 0.25
        bins = np.linspace(0, 3, 101)

        fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharey='row', sharex='row')
        axes[0, 0].set_title(f'Training: {rms_error_train:0.6f}')
        axes[0, 1].set_title(f'Validation: {rms_error_val:0.6f}')
        axes[0, 0].plot(
            vol[train], vol_pred_train,
            marker='.', linestyle='none', markersize=ms, alpha=alpha
            )
        axes[0, 1].plot(
            vol[~train], vol_pred_val,
            marker='.', linestyle='none', markersize=ms, alpha=alpha
            )
        axes[1, 0].hist(error_train/std_train, bins=bins, density=True)
        axes[1, 1].hist(error_val/std_val, bins=bins, density=True)
        for index in range(2):
            axes[1, index].plot(bins, 2/np.sqrt(2*np.pi) * np.exp(-1/2 * bins**2))
            axes[0, index].plot(self.volume_lims, self.volume_lims, linestyle='dotted', color=[0, 0, 0])
            axes[0, index].set_xlabel('Volume True')
            axes[0, index].set_xlim(self.volume_lims)
            axes[0, index].set_ylim(self.volume_lims)

        axes[0, 0].set_ylabel('Volume Predicted')
        axes[1, 0].set_ylabel('Error / std')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.bravais_lattice}_volume_pred_{self.model_params["tag"]}.png')
        plt.close()

        self.save()

    def save(self):
        write_params(
            self.model_params,
            f'{self.save_to}/{self.bravais_lattice}_random_params_{self.model_params["tag"]}.csv'
            )
        joblib.dump(
            self.random_forest_regressor,
            f'{self.save_to}/{self.bravais_lattice}_random_forest_regressor.bin'
            )
        np.save(
            f'{self.save_to}/{self.bravais_lattice}_volume_lims.npy',
            self.volume_lims
            )
        np.save(
            f'{self.save_to}/{self.bravais_lattice}_volume_distribution.npy',
            self.volume_distribution
            )

    def load_from_tag(self):
        params = read_params(
            f'{self.save_to}/{self.bravais_lattice}_random_params_{self.model_params["tag"]}.csv'
            )
        params_keys = [
            'random_state',
            'n_estimators',
            'min_samples_leaf',
            'max_depth',
            'subsample',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['random_state'] = int(params['random_state'])
        self.model_params['n_estimators'] = int(params['n_estimators'])
        self.model_params['min_samples_leaf'] = int(params['min_samples_leaf'])
        if params['max_depth'] == '':
            self.model_params['max_depth'] = None
        else:
            self.model_params['max_depth'] = int(params['max_depth'])
        self.model_params['subsample'] = float(params['subsample'])

        self.random_forest_regressor = joblib.load(
            f'{self.save_to}/{self.bravais_lattice}_random_forest_regressor.bin'
            )
        self.volume_lims = np.load(
            f'{self.save_to}/{self.bravais_lattice}_volume_lims.npy'
            )
        self.volume_distribution = np.load(
            f'{self.save_to}/{self.bravais_lattice}_volume_distribution.npy'
            )

    def generate(self, n_unit_cells, rng, q2_obs, model=None):
        if self.lattice_system in ['cubic', 'tetragonal', 'hexagonal', 'orthorhombic']:
            random_xnn = rng.uniform(low=0, high=1, size=(n_unit_cells, self.unit_cell_length))
        elif self.lattice_system == 'rhombohedral':
            random_xnn = np.zeros((n_unit_cells, 2))
            random_xnn[:, 0] = rng.uniform(low=0, high=1, size=n_unit_cells)
            random_cos_angle = rng.uniform(low=0, high=1, size=n_unit_cells)
            random_xnn[:, 1] = 2 * random_xnn[:, 0] * random_cos_angle
        elif self.lattice_system == 'monoclinic':
            random_xnn = np.zeros((n_unit_cells, 4))
            random_xnn[:, :3] = rng.uniform(low=0, high=1, size=(n_unit_cells, 3))
            random_cos_beta = rng.uniform(low=0, high=1, size=n_unit_cells)
            random_xnn[:, 3] = 2*np.sqrt(random_xnn[:, 0] * random_xnn[:, 2]) * random_cos_beta
        elif self.lattice_system == 'triclinic':
            random_xnn = np.zeros((n_unit_cells, 6))
            random_xnn[:, :3] = rng.uniform(low=0, high=1, size=(n_unit_cells, 3))
            random_cos_alpha = rng.uniform(low=0, high=1, size=n_unit_cells)
            random_cos_beta = rng.uniform(low=0, high=1, size=n_unit_cells)
            random_cos_gamma = rng.uniform(low=0, high=1, size=n_unit_cells)
            random_xnn[:, 3] = 2*np.sqrt(random_xnn[:, 0] * random_xnn[:, 1]) * random_cos_gamma
            random_xnn[:, 4] = 2*np.sqrt(random_xnn[:, 0] * random_xnn[:, 2]) * random_cos_beta
            random_xnn[:, 5] = 2*np.sqrt(random_xnn[:, 1] * random_xnn[:, 2]) * random_cos_alpha

        random_xnn = fix_unphysical(
            xnn=random_xnn,
            rng=rng,
            lattice_system=self.lattice_system,
            minimum_unit_cell=None,
            maximum_unit_cell=None,
            )
        random_reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            random_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        random_volume_generated = get_unit_cell_volume(
            random_reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        if model == 'random':
            random_volume_scale = rng.uniform(
                low=self.volume_lims[0], high=self.volume_lims[1], size=n_unit_cells
                )
        elif model == 'distribution_volume':
            rand = rng.random(size=n_unit_cells)
            indices = np.searchsorted(self.volume_distribution[:, 2], rand)
            random_volume_scale = self.volume_distribution[indices, 0]
        elif model == 'predicted_volume':
            preds = np.zeros(self.model_params['n_estimators'])
            for est_index in range(self.model_params['n_estimators']):
                preds[est_index] = self.random_forest_regressor.estimators_[est_index].predict(
                    q2_obs[np.newaxis]
                    )[0]

            if n_unit_cells > self.model_params['n_estimators']:
                replace = True
            else:
                replace = False
            indices = rng.choice(
                self.model_params['n_estimators'],
                size=n_unit_cells,
                replace=replace
                )
            random_volume_scale = preds[indices]

        scale = (random_volume_scale / random_volume_generated)**(1/3)
        if self.lattice_system in ['cubic', 'tetragonal', 'hexagonal', 'orthorhombic']:
            random_reciprocal_unit_cell *= scale[:, np.newaxis]
        elif self.lattice_system == 'rhombohedral':
            random_reciprocal_unit_cell[:, 0] *= scale
        elif self.lattice_system in ['monoclinic', 'triclinic']:
            random_reciprocal_unit_cell[:, :3] *= scale[:, np.newaxis]

        if self.lattice_system == 'monoclinic':
            if self.bravais_lattice == 'mC':
                standard_spacegroups = [
                    'I 1 2 1', 'I 1 m 1', 'I 1 a 1', 'I 1 2/m 1', 'I 1 2/a 1'
                    ]
            elif self.bravais_lattice == 'mP':
                standard_spacegroups = [
                    'P 1 2 1', 'P 1 m 1', 'P 1 2/m 1', 'P 1 21 1', 'P 1 21/m 1',
                    'P 1 n 1', 'P 1 2/n 1', 'P 1 21/n 1'
                    ]
            spacegroups = rng.choice(standard_spacegroups, n_unit_cells, replace=True)
            for index in range(n_unit_cells):
                random_reciprocal_unit_cell_ = np.array([
                    random_reciprocal_unit_cell[index, 0],
                    random_reciprocal_unit_cell[index, 1],
                    random_reciprocal_unit_cell[index, 2],
                    np.pi/2,
                    random_reciprocal_unit_cell[index, 3],
                    np.pi/2,
                    ])
                random_reciprocal_unit_cell_, _, _ = reindex_entry_monoclinic(
                    random_reciprocal_unit_cell_,
                    spacegroup_symbol=spacegroups[index],
                    space='reciprocal'
                    )
                random_reciprocal_unit_cell[index, :3] = random_reciprocal_unit_cell_[:3]
                random_reciprocal_unit_cell[index, 3] = random_reciprocal_unit_cell_[4]
        elif self.lattice_system == 'orthorhombic':
            if self.bravais_lattice in ['oF', 'oI', 'oP']:
                # In these cases, the unit cells are sorted to get a<b<c
                # The actual space group does not matter
                spacegroup_symbol = 'P 2 2 2'
                spacegroup_number = 16
            elif self.bravais_lattice == 'oC':
                # In these cases, the unit cells are sorted to get a<b
                spacegroup_symbol = 'C 2 2 2'
                spacegroup_number = 21
            random_unit_cell = reciprocal_uc_conversion(
                random_reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            for index in range(n_unit_cells):
                random_unit_cell_ = np.array([
                    random_unit_cell[index, 0],
                    random_unit_cell[index, 1],
                    random_unit_cell[index, 2],
                    np.pi/2,
                    np.pi/2,
                    np.pi/2,
                    ])
                _, _, random_unit_cell_, _ = reindex_entry_orthorhombic(
                    random_unit_cell_, spacegroup_symbol, spacegroup_number
                    )
                random_unit_cell[index] = random_unit_cell_[:3]
            random_reciprocal_unit_cell = reciprocal_uc_conversion(
                random_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
        elif self.lattice_system == 'triclinic':
            random_unit_cell = reciprocal_uc_conversion(
                random_reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            random_reciprocal_unit_cell, _ = reindex_entry_triclinic(
                random_unit_cell, space='direct'
                )
            random_reciprocal_unit_cell = reciprocal_uc_conversion(
                random_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
        random_unit_cell = reciprocal_uc_conversion(
            random_reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return random_unit_cell
