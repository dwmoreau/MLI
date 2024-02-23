"""
- Why did optimization stop working
    - Why is prediction with closest outperforming NN
        - remove the "flat" NN
    - move analysis to dials server to speed up work


* Optimization:
    - What differentiates a found / not found entry
        - large differences between prediction and true
    - common assignments:
        - drop during optimization but include in loss
        - use all hkl assignments with largest N likelihoods
    - Full softmax array optimization
    - assignment with group specific assigners
    - SVD

- Data
    - Get data from other databases:
        - Materials project
        - ICSD
    - more strict duplicate removal
        - entries that differ by one atom in the unit cell
    - experimental data from rruff
        - verify that unit cell is consistent with diffraction
    - redo dataset generation with new parameters based on RRUFF database

- SWE:
    * memory leak during cyclic training
    - get working on dials

- Augmentation
    - make peak drop rate a function of distance and q2
    - Bug in the miller index assignments / labeling. Very poor assigment model accuracy.

- Regression:
    - Improve hyperparameters
        - variance estimate almost always overfits
        - mean estimate tends to underfit
    - random forest predictions

    - prediction of PCA components
        - evaluation of fitting in the PCA / Scaled space
        - evaluation of covariance
    - read Stirn 2023 and implement
    Detlefsen 2019:
        - cluster input d-spacings
          - map d-spacings onto a single scalar correlated with volume
        - ??? extrapolation architecture

- Assignments
    - how to penalize multiple assignments to the same hkl
    - How to incorporate forward model
    - How would this look as a graphical NN
"""
import csv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from Assigner import Assigner
from Augmentor import Augmentor
from Evaluations import evaluate_regression
from Evaluations import calibrate_regression
from Regression import Regression_AlphaBeta
from Utilities import Q2Calculator


class Indexing:
    def __init__(self, assign_params=None, aug_params=None, data_params=None, reg_params=None, seed=12345):
        self.random_seed = seed
        self.rng = np.random.default_rng(self.random_seed)
        self.n_generated_points = 100
        self.save_to = dict.fromkeys(['results', 'data', 'regression', 'assigner', 'augmentor'])
        self.assign_params = assign_params
        self.aug_params = aug_params
        self.data_params = data_params
        self.reg_params = reg_params

        self.save_to = {
            'results': f'models/{self.data_params["tag"]}',
            'assigner': f'models/{self.data_params["tag"]}/assigner',
            'augmentor': f'models/{self.data_params["tag"]}/augmentor',
            'data': f'models/{self.data_params["tag"]}/data',
            'regression': f'models/{self.data_params["tag"]}/regression',
            }
        if not os.path.exists(self.save_to['results']):
            os.mkdir(self.save_to['results'])
            os.mkdir(self.save_to['assigner'])
            os.mkdir(self.save_to['augmentor'])
            os.mkdir(self.save_to['data'])
            os.mkdir(self.save_to['regression'])

        if self.data_params['load_from_tag']:
            self.setup_from_tag()
        else:
            self.setup()

    def setup(self):
        data_params_defaults = {
            'lattice_system': None,
            'unit_cell_representation': None,
            'group_by': 'groups',
            'data_dir': '/Users/DWMoreau/MLI/data',
            'augment': False,
            'train_fraction': 0.80,
            'n_max': 25000,
            'n_points': 20,
            'points_tag': 'intersect',
            'hkl_ref_length': 500,
            }

        for key in data_params_defaults.keys():
            if not data_params_defaults[key] is None:
                if key not in self.data_params.keys():
                    self.data_params[key] = data_params_defaults[key]

        if self.data_params['lattice_system'] == 'cubic':
            self.data_params['y_indices'] = [0]
            self.data_params['bravais_lattices'] = ['cF', 'cI', 'cP']
            self.data_params['groups'] = []
        elif self.data_params['lattice_system'] == 'tetragonal':
            self.data_params['y_indices'] = [0, 2]
            self.data_params['bravais_lattices'] = ['tI', 'tP']
            self.data_params['groups'] = []
        elif self.data_params['lattice_system'] == 'orthorhombic':
            self.data_params['y_indices'] = [0, 1, 2]
            self.data_params['bravais_lattices'] = ['oC', 'oF', 'oI', 'oP']
        elif self.data_params['lattice_system'] == 'monoclinic':
            self.data_params['y_indices'] = [0, 1, 2, 4]
            self.data_params['bravais_lattices'] = ['mC', 'mP']
            self.data_params['groups'] = []
        elif self.data_params['lattice_system'] == 'triclinic':
            self.data_params['y_indices'] = [0, 1, 2, 3, 4, 5]
            self.data_params['bravais_lattices'] = ['aP']
            self.data_params['groups'] = []
        elif self.data_params['lattice_system'] == 'rhombohedral':
            self.data_params['y_indices'] = [0, 3]
            self.data_params['groups'] = []
        elif self.data_params['lattice_system'] == 'hexagonal':
            self.data_params['y_indices'] = [0, 2]
            self.data_params['bravais_lattices'] = ['hP']
            self.data_params['groups'] = []
        self._setup_joint()

    def setup_from_tag(self):
        self.uc_scaler = joblib.load(f'{self.save_to["data"]}/uc_scaler.bin')
        self.volume_scaler = joblib.load(f'{self.save_to["data"]}/volume_scaler.bin')
        self.q2_scaler = joblib.load(f'{self.save_to["data"]}/q2_scaler.bin')

        with open(f'{self.save_to["data"]}/data_params.csv', 'r') as params_file:
            reader = csv.DictReader(params_file)
            for row in reader:
                params = row
        data_params_keys = [
            'augment',
            'bravais_lattices'
            'lattice_system',
            'unit_cell_representation',
            'data_dir',
            'train_fraction',
            'n_max',
            'y_indices',
            'n_points',
            'points_tag',
            'use_reduced_cell',
            'n_outputs',
            'hkl_ref_length',
            'groupspec_file_name',
            'groupspec_sheet',
            ]

        self.data_params = dict.fromkeys(data_params_keys)
        self.data_params['load_from_tag'] = True
        bravais_lattices = params['bravais_lattices'].replace(' ', '').replace("'", '')
        self.data_params['bravais_lattices'] = bravais_lattices.split('[')[1].split(']')[0].split(',')
        self.data_params['lattice_system'] = params['lattice_system']
        self.data_params['unit_cell_representation'] = params['unit_cell_representation']
        if params['augment'] == 'True':
            self.data_params['augment'] = True
        elif params['augment'] == 'False':
            self.data_params['augment'] = False
        self.data_params['data_dir'] = params['data_dir']
        self.data_params['y_indices'] = np.array(params['y_indices'].split('[')[1].split(']')[0].split(','), dtype=int)
        self.data_params['train_fraction'] = float(params['train_fraction'])
        self.data_params['n_max'] = int(params['n_max'])
        self.data_params['n_points'] = int(params['n_points'])
        self.data_params['points_tag'] = params['points_tag']
        self.data_params['hkl_ref_length'] = int(params['hkl_ref_length'])
        self.data_params['n_outputs'] = int(params['n_outputs'])
        self.data_params['groupspec_file_name'] = params['groupspec_file_name']
        self.data_params['groupspec_sheet'] = params['groupspec_sheet']
        self._setup_joint()

    def _setup_joint(self):
        if self.data_params['unit_cell_representation'] == 'conventional':
            self.unit_cell_key = 'unit_cell'
            self.volume_key = 'volume'
            self.hkl_key = 'hkl'
            self.hkl_prefactor = ''
        elif self.data_params['unit_cell_representation'] == 'reduced':
            self.unit_cell_key = 'reduced_unit_cell'
            self.volume_key = 'reduced_volume'
            self.hkl_key = 'hkl'
            self.hkl_prefactor = ''
        elif self.data_params['unit_cell_representation'] == 'reindexed':
            self.unit_cell_key = 'reindexed_unit_cell'
            self.volume_key = 'volume'
            self.hkl_key = 'reindexed_hkl'
            self.hkl_prefactor = 'reindexed_'
        else:
            print('Need to supply a unit_cell_representation')
            assert False

        for key in self.assign_params.keys():
            self.assign_params[key]['n_outputs'] = self.data_params['hkl_ref_length']
        self.data_params['n_outputs'] = len(self.data_params['y_indices'])
        self.reg_params['n_outputs'] = self.data_params['n_outputs']

        all_labels = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        self.uc_labels = [all_labels[index] for index in self.data_params['y_indices']]

        group_spec = pd.read_excel(
            os.path.join(self.data_params['data_dir'], self.data_params['groupspec_file_name']),
            sheet_name=self.data_params['groupspec_sheet'],
            )
        group_spec = group_spec.loc[group_spec['group'].notna()]
        group_spec['hm symbol'] = group_spec['hm symbol'].str.strip()
        self.data_params['groups'] = group_spec['group'].unique()
        self.group_mappings = dict.fromkeys(group_spec['hm symbol'].unique())
        for index in range(len(group_spec)):
            self.group_mappings[group_spec.iloc[index]['hm symbol']] = group_spec.iloc[index]['group']

    def load_data(self):
        read_columns = [
            'lattice_system',
            'bravais_lattice',
            'spacegroup_number',
            'volume',
                f'{self.hkl_prefactor}spacegroup_symbol_hm',
            f'{self.hkl_prefactor}unit_cell',
            f'd_spacing_{self.data_params["points_tag"]}',
            f'{self.hkl_prefactor}h_{self.data_params["points_tag"]}',
            f'{self.hkl_prefactor}k_{self.data_params["points_tag"]}',
            f'{self.hkl_prefactor}l_{self.data_params["points_tag"]}',
            ]

        if self.data_params['augment']:
            read_columns += [
                'd_spacing_sa',
                f'{self.hkl_prefactor}h_sa', f'{self.hkl_prefactor}k_sa', f'{self.hkl_prefactor}l_sa',
                ]

        data = []
        for index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            file_name = os.path.join(self.data_params['data_dir'], f'GeneratedDatasets/dataset_{bravais_lattice}.parquet')
            print(f'Loading data from {file_name}')
            data.append(pd.read_parquet(file_name, columns=read_columns))
        self.data = pd.concat(data, ignore_index=True)

        # Remove data that doesn't have enough peaks
        # A total of 60 or so peaks are included in the data set - for all entries
        # If there were less than 60 peaks, those get padded with zeros at the end of the array.
        #   - the 60 number is arbitrary and set in GenerateDataset_mpi.py

        points = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        indices = points.apply(len) >= self.data_params['n_points']
        self.data = self.data.loc[indices]
        points = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        enough_peaks = points.apply(np.count_nonzero) >= self.data_params['n_points']
        self.data = self.data.loc[enough_peaks]
        self.data['augmented'] = np.zeros(self.data.shape[0], dtype=bool)

        # Add label to data and down sample
        self.data['group'] = self.data['reindexed_spacegroup_symbol_hm'].map(lambda x: self.group_mappings[x])
        data_grouped = self.data.groupby('group')
        data_group = [None for i in range(len(data_grouped.groups.keys()))]
        for index, group in enumerate(data_grouped.groups.keys()):
            data_group[index] = data_grouped.get_group(group)
            data_group[index].insert(loc=0, column='group_label', value=index * np.ones(len(data_group[index])))
            data_group[index] = data_group[index].sample(
                n=min(len(data_group[index]), self.data_params['n_max']),
                replace=False,
                random_state=self.random_seed
                )
        self.data = pd.concat(data_group, ignore_index=True)

        d_spacing_pd = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        d_spacing = [np.zeros(self.data_params['n_points']) for i in range(self.data.shape[0])]
        for entry_index in range(self.data.shape[0]):
            d_spacing[entry_index] = d_spacing_pd.iloc[entry_index][:self.data_params['n_points']]
        self.data['d_spacing'] = d_spacing
        # The math is a bit easier logistically when we use q**2
        self.data['q2'] = 1 / self.data['d_spacing']**2
        if self.data_params['augment']:
            self.data['q2_sa'] = 1 / self.data['d_spacing_sa']**2
            self.data[f'q2_{self.data_params["points_tag"]}'] = 1 / self.data[f'd_spacing_{self.data_params["points_tag"]}']**2

        bl_label = np.zeros(self.data.shape[0], dtype=int)
        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            indices = np.where(self.data['bravais_lattice'] == bravais_lattice)[0]
            bl_label[indices] = bl_index
        self.data['bravais_lattice_label'] = bl_label

        spacegroups = self.data['spacegroup_number'].unique()
        spacegroup_label = np.zeros(self.data.shape[0], dtype=int)
        self.data_params['spacegroups'] = []
        for spacegroup_index, spacegroup in enumerate(spacegroups):
            indices = np.where(self.data['spacegroup_number'] == spacegroup)[0]
            spacegroup_label[indices] = spacegroup_index
            self.data_params['spacegroups'].append(spacegroup)
        self.data['spacegroup_label'] = spacegroup_label

        # This sets up the training / validation tags so that the validation set is taken
        # evenly from groups
        train_label = np.ones(self.data.shape[0], dtype=bool)
        for group_index, group in enumerate(self.data_params['groups']):
            indices = np.where(self.data['group'] == group)[0]
            n_val = int(indices.size * (1 - self.data_params['train_fraction']))
            val_indices = self.rng.choice(indices, size=n_val, replace=False)
            train_label[val_indices] = False
        self.data['train'] = train_label

        # put the hkl's together
        hkl = np.zeros((len(self.data), self.data_params['n_points'], 3), dtype=int)
        if self.data_params['augment']:
            hkl_sa = np.zeros((len(self.data), self.n_generated_points, 3), dtype=int)
            hkl_all = np.zeros((len(self.data), self.n_generated_points, 3), dtype=int)
        for entry_index in range(len(self.data)):
            entry = self.data.iloc[entry_index]
            hkl[entry_index, :, 0] = entry[f'{self.hkl_prefactor}h_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            hkl[entry_index, :, 1] = entry[f'{self.hkl_prefactor}k_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            hkl[entry_index, :, 2] = entry[f'{self.hkl_prefactor}l_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            if self.data_params['augment']:
                n_peaks_sa = entry[f'{self.hkl_prefactor}h_sa'].size
                hkl_sa[entry_index, :n_peaks_sa, 0] = entry[f'{self.hkl_prefactor}h_sa']
                hkl_sa[entry_index, :n_peaks_sa, 1] = entry[f'{self.hkl_prefactor}k_sa']
                hkl_sa[entry_index, :n_peaks_sa, 2] = entry[f'{self.hkl_prefactor}l_sa']
        self.data[f'{self.hkl_prefactor}hkl'] = list(hkl)
        if self.data_params['augment']:
            self.data[f'{self.hkl_prefactor}hkl_sa'] = list(hkl_sa)
        drop_columns = [
            f'{self.hkl_prefactor}h_{self.data_params["points_tag"]}',
            f'{self.hkl_prefactor}k_{self.data_params["points_tag"]}',
            f'{self.hkl_prefactor}l_{self.data_params["points_tag"]}'
            ]
        if self.data_params['augment']:
            drop_columns += [
                f'{self.hkl_prefactor}h_sa',
                f'{self.hkl_prefactor}k_sa',
                f'{self.hkl_prefactor}l_sa',
                ]
        self.data.drop(columns=drop_columns, inplace=True)

        self.setup_scalers()
        # Convert angles to radians
        unit_cell = np.stack(self.data[self.unit_cell_key])
        unit_cell[:, 3:] = np.pi/180 * unit_cell[:, 3:]
        self.data[self.unit_cell_key] = list(unit_cell)

        if self.data_params['augment']:
            self.augment_data()
            drop_columns = [
                f'q2_{self.data_params["points_tag"]}',
                'q2_sa',
                f'{self.hkl_prefactor}hkl_sa',
                ]
            self.data.drop(columns=drop_columns, inplace=True)

        self.setup_hkl()

        # This does another shuffle.
        self.data = self.data.sample(frac=1, replace=False, random_state=self.random_seed)
        self.plot_input()
        self.save()

    def load_data_from_tag(self, load_augmented, load_train):
        self.hkl_ref = np.load(f'{self.save_to["data"]}/hkl_ref.npy')
        self.data = pd.read_parquet(f'{self.save_to["data"]}/data.parquet')
        hkl = np.stack([
            np.stack(self.data[f'{self.hkl_prefactor}h'], axis=0)[:, :, np.newaxis],
            np.stack(self.data[f'{self.hkl_prefactor}k'], axis=0)[:, :, np.newaxis],
            np.stack(self.data[f'{self.hkl_prefactor}l'], axis=0)[:, :, np.newaxis]
            ], axis=2
            )
        self.data[f'{self.hkl_prefactor}hkl'] = list(hkl)
        self.data.drop(
            columns=[f'{self.hkl_prefactor}h', f'{self.hkl_prefactor}k', f'{self.hkl_prefactor}l'],
            inplace=True
            )

    def save(self):
        hkl = np.stack(self.data[f'{self.hkl_prefactor}hkl'])
        self.data[f'{self.hkl_prefactor}h'] = list(hkl[:, :, 0])
        self.data[f'{self.hkl_prefactor}k'] = list(hkl[:, :, 1])
        self.data[f'{self.hkl_prefactor}l'] = list(hkl[:, :, 2])
        self.data.drop(columns=[f'{self.hkl_prefactor}hkl'], inplace=True)
        self.data.to_parquet(f'{self.save_to["data"]}/data.parquet')

        np.save(f'{self.save_to["data"]}/hkl_ref.npy', self.hkl_ref)
        joblib.dump(self.uc_scaler, f'{self.save_to["data"]}/uc_scaler.bin')
        joblib.dump(self.volume_scaler, f'{self.save_to["data"]}/volume_scaler.bin')
        joblib.dump(self.q2_scaler, f'{self.save_to["data"]}/q2_scaler.bin')

        with open(f'{self.save_to["data"]}/data_params.csv', 'w') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.data_params.keys())
            writer.writeheader()
            writer.writerow(self.data_params)

    def setup_hkl(self):
        # This converts the true h, k, and l columns to a single label
        if self.data_params['lattice_system'] == 'cubic':
            hkl_ref_file_name = 'hkl_ref_cubic.npy'
        elif self.data_params['lattice_system'] == 'hexagonal':
            hkl_ref_file_name = 'hkl_ref_hexagonal.npy'
        elif self.data_params['lattice_system'] == 'monoclinic':
            hkl_ref_file_name = 'hkl_ref_monoclinic.npy'
        elif self.data_params['lattice_system'] == 'orthorhombic':
            hkl_ref_file_name = 'hkl_ref_orthorhombic.npy'
        elif self.data_params['lattice_system'] == 'rhombohedral':
            hkl_ref_file_name = 'hkl_ref_rhombohedral.npy'
        elif self.data_params['lattice_system'] == 'tetragonal':
            hkl_ref_file_name = 'hkl_ref_tetragonal.npy'
        elif self.data_params['lattice_system'] == 'triclinic':
            hkl_ref_file_name = 'hkl_ref_triclinic.npy'
        self.hkl_ref = np.load(os.path.join(
            self.data_params['data_dir'], hkl_ref_file_name
            ))[:2*self.data_params['hkl_ref_length']]
        indices = np.logical_and(self.data['train'], ~self.data['augmented'])
        train = self.data[indices]
        unit_cell = np.stack(train[self.unit_cell_key])[:, self.data_params['y_indices']]
        q2_ref_calculator = Q2Calculator(self.data_params['lattice_system'], self.hkl_ref, tensorflow=False)
        q2_ref = q2_ref_calculator.get_q2(unit_cell)
        sort_indices = np.argsort(q2_ref.mean(axis=0))
        self.hkl_ref = self.hkl_ref[sort_indices][:self.data_params['hkl_ref_length'] - 1]
        self.hkl_ref = np.concatenate((self.hkl_ref, np.zeros((1, 3))), axis=0)
        hkl = np.stack(self.data[self.hkl_key])

        if self.data_params['lattice_system'] == 'cubic':
            check_ref = self.hkl_ref[:, 0]**2 + self.hkl_ref[:, 1]**2 + self.hkl_ref[:, 2]**2
            check_data = hkl[:, :, 0]**2 + hkl[:, :, 1]**2 + hkl[:, :, 2]**2
            check_ref = check_ref[:, np.newaxis]
            check_data = check_data[:, :, np.newaxis]
        elif self.data_params['lattice_system'] == 'hexagonal':
            check_ref = np.column_stack((
                self.hkl_ref[:, 0]**2 + self.hkl_ref[:, 0]*self.hkl_ref[:, 1] + self.hkl_ref[:, 1]**2,
                self.hkl_ref[:, 2]**2
                ))
            check_data = np.stack((
                hkl[:, :, 0]**2 + hkl[:, :, 0] * hkl[:, :, 1] + hkl[:, :, 1]**2,
                hkl[:, :, 2]**2
                ),
                axis=2
                )
        elif self.data_params['lattice_system'] == 'monoclinic':
            check_ref = np.column_stack((
                self.hkl_ref[:, 0]**2,
                self.hkl_ref[:, 1]**2,
                self.hkl_ref[:, 2]**2,
                self.hkl_ref[:, 0]*self.hkl_ref[:, 2],
                ))
            check_data = np.stack((
                hkl[:, :, 0]**2,
                hkl[:, :, 1]**2,
                hkl[:, :, 2]**2,
                hkl[:, :, 0]*hkl[:, :, 2],
                ),
                axis=2
                )
        elif self.data_params['lattice_system'] == 'orthorhombic':
            check_ref = np.column_stack((
                self.hkl_ref[:, 0]**2,
                self.hkl_ref[:, 1]**2,
                self.hkl_ref[:, 2]**2
                ))
            check_data = np.stack((
                hkl[:, :, 0]**2,
                hkl[:, :, 1]**2,
                hkl[:, :, 2]**2
                ),
                axis=2
                )
        elif self.data_params['lattice_system'] == 'rhombohedral':
            check_ref = np.column_stack((
                self.hkl_ref[:, 0]**2 + self.hkl_ref[:, 1]**2 + self.hkl_ref[:, 2]**2,
                self.hkl_ref[:, 0]*self.hkl_ref[:, 1] + self.hkl_ref[:, 0]*self.hkl_ref[:, 2] + self.hkl_ref[:, 1]*self.hkl_ref[:, 2],
                ))
            check_data = np.stack((
                hkl[:, :, 0]**2 +  hkl[:, :, 1]**2 + hkl[:, :, 2]**2,
                hkl[:, :, 0]*hkl[:, :, 1] + hkl[:, :, 0]*hkl[:, :, 2] + hkl[:, :, 1]*hkl[:, :, 2]
                ),
                axis=2
                )
        elif self.data_params['lattice_system'] == 'tetragonal':
            check_ref = np.column_stack((
                self.hkl_ref[:, 0]**2 + self.hkl_ref[:, 1]**2,
                self.hkl_ref[:, 2]**2
                ))
            check_data = np.stack((
                hkl[:, :, 0]**2 + hkl[:, :, 1]**2,
                hkl[:, :, 2]**2
                ),
                axis=2
                )
        elif self.data_params['lattice_system'] == 'triclinic':
            check_ref = np.column_stack((
                self.hkl_ref[:, 0]**2,
                self.hkl_ref[:, 1]**2,
                self.hkl_ref[:, 2]**2,
                self.hkl_ref[:, 0]*self.hkl_ref[:, 1],
                self.hkl_ref[:, 0]*self.hkl_ref[:, 2],
                self.hkl_ref[:, 1]*self.hkl_ref[:, 2],
                ))
            check_data = np.stack((
                hkl[:, :, 0]**2,
                hkl[:, :, 1]**2,
                hkl[:, :, 2]**2,
                hkl[:, :, 0]*hkl[:, :, 1],
                hkl[:, :, 0]*hkl[:, :, 2],
                hkl[:, :, 1]*hkl[:, :, 2],
                ),
                axis=2
                )

        hkl_labels = (self.data_params['hkl_ref_length'] - 1) * np.ones((
            len(self.data), self.data_params['n_points']),
            dtype=int
            )

        for entry_index in range(len(self.data)):
            for point_index in range(self.data_params['n_points']):
                hkl_ref_index = np.argwhere(np.all(
                    check_ref[:, :] == check_data[entry_index, point_index, :],
                    axis=1
                    ))
                if len(hkl_ref_index) == 1:
                    hkl_labels[entry_index, point_index] = hkl_ref_index
        self.data['hkl_labels'] = list(hkl_labels)
        """
        print(f'Unlabeled peaks: {(hkl_labels == self.data_params["hkl_ref_length"] - 1).sum()}')
        print(f'Maximum label: {hkl_labels.max()}')
        for index in range(self.data_params['hkl_ref_length'] - 1):
            n = np.sum(hkl_labels == index)
            if n == 0:
                print(f'hkl {self.hkl_ref[index]} has no equivalents ({index})')
        """

    def augment_data(self):
        self.augmentor = Augmentor(
            aug_params=self.aug_params,
            data_params=self.data_params,
            min_unit_cell_scaled=self.min_unit_cell_scaled,
            n_generated_points=self.n_generated_points,
            save_to=self.save_to['augmentor'],
            seed=self.random_seed,
            uc_scaler=self.uc_scaler,
            unit_cell_key=self.unit_cell_key,
            hkl_key=self.hkl_key,
            )
        self.augmentor.setup(self.data)
        data_augmented = [None for i in range(len(self.data_params['groups']))]
        for group_index, group in enumerate(self.data_params['groups']):
            print(f'Augmenting {group}')
            group_data = self.data[self.data['group'] == group]
            data_augmented[group_index] = self.augmentor.augment(group_data, f'{self.hkl_prefactor}spacegroup_symbol_hm')
        data_augmented = pd.concat(data_augmented, ignore_index=True)
        self.data = pd.concat((self.data, data_augmented), ignore_index=True)

    def infer_unit_cell_volume(self, unit_cell):
        if self.data_params['lattice_system'] == 'cubic':
            volume = unit_cell**3
        elif self.data_params['lattice_system'] == 'tetragonal':
            volume = unit_cell[:, 0]**2 * unit_cell[:, 1]
        elif self.data_params['lattice_system'] == 'orthorhombic':
            volume = unit_cell[:, 0] * unit_cell[:, 1] * unit_cell[:, 2]
        elif self.data_params['lattice_system'] == 'hexagonal':
            pass
        elif self.data_params['lattice_system'] == 'rhombohedral':
            pass
        elif self.data_params['lattice_system'] == 'monoclinic':
            pass
        elif self.data_params['lattice_system'] == 'triclinic':
            a = unit_cell[:, 0]
            b = unit_cell[:, 1]
            c = unit_cell[:, 2]
            calpha = np.cos(unit_cell[:, 3])
            cbeta = np.cos(unit_cell[:, 4])
            cgamma = np.cos(unit_cell[:, 5])
            arg = 1 - calpha**2 - cbeta**2 - cgamma**2 + 2*calpha*cbeta*cgamma
            volume = (a*b*c) * np.sqrt(arg)
        return volume

    def q2_scale(self, q2):
        return self.q2_scaler.transform(q2.ravel()[:, np.newaxis]).reshape(q2.shape)

    def q2_revert(self, q2_scaled):
        q2 = self.q2_scaler.inverse_transform(
            q2_scaled.ravel()[:, np.newaxis]
            ).reshape(q2_scaled.shape)
        return q2

    def volume_scale(self, volume):
        return self.volume_scaler.transform(volume[:, np.newaxis]).reshape(volume.shape)

    def volume_revert(self, volume_scaled):
        volume = self.volume_scaler.inverse_transform(
            volume_scaled[:, np.newaxis]
            ).reshape(volume_scaled.shape)
        return volume

    def y_scale(self, y):
        y_scaled = np.zeros(y.shape)
        y_scaled[:3] = self.uc_scaler.transform(y[:3][:, np.newaxis])[:, 0]
        y_scaled[3:] = y[3:] - np.pi/2
        return y_scaled

    def y_revert(self, y):
        y_reverted = np.zeros(y.shape)
        y_reverted[:3] = self.uc_scaler.inverse_transform(y[:3][:, np.newaxis])[:, 0]
        y_reverted[3:] = y[3:] + np.pi/2
        return y_reverted

    def setup_scalers(self):
        # q2 scaling
        self.q2_scaler = StandardScaler()
        q2_train = np.stack(self.data[self.data['train']]['q2']).ravel()
        self.q2_scaler.fit(q2_train[:, np.newaxis])
        self.data['q2_scaled'] = self.data['q2'].apply(self.q2_scale)

        # Unit cell parameters scaling
        self.uc_scaler = StandardScaler()
        uc_train = np.stack(self.data[self.data['train']][self.unit_cell_key])
        self.uc_scaler.fit(uc_train[:, :3].ravel()[:, np.newaxis])
        self.data[f'{self.unit_cell_key}_scaled'] = self.data[self.unit_cell_key].apply(self.y_scale)
        # this hard codes the minimum allowed unit cell in augmented data to 1A
        self.min_unit_cell_scaled = (1 - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]

        # Volume scaling
        self.volume_scaler = StandardScaler()
        volume_train = np.array(self.data[self.data['train']][self.volume_key])
        self.volume_scaler.fit(volume_train[:, np.newaxis])
        self.data[f'{self.volume_key}_scaled'] = list(self.volume_scale(np.array(self.data[self.volume_key])))

    def plot_input(self):
        def make_hkl_plot(data, n_points, hkl_ref_length, save_to):
            fig, axes = plt.subplots(n_points, 1, figsize=(6, 10), sharex=True)
            hkl_labels = np.stack(data['hkl_labels']) # n_data x n_points
            bins = np.arange(0, hkl_ref_length + 1) - 0.5
            centers = (bins[1:] + bins[:-1]) / 2
            width = bins[1] - bins[0]
            for index in range(n_points):
                hist, _ = np.histogram(hkl_labels[:, index], bins=bins, density=True)
                axes[index].bar(centers, hist, width=width)
                axes[index].set_ylabel(f'Peak {index}')
            axes[n_points - 1].set_xlabel('HKL label')
            fig.tight_layout()
            fig.savefig(save_to)
            plt.close()

        y_labels = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        plot_volume_scale = 1000

        # Histogram of Groups
        x = np.arange(len(self.data_params['groups']))
        group_counts = np.zeros((len(self.data_params['groups']), 2))
        for index, group in enumerate(self.data_params['groups']):
            group_data = self.data[self.data['group'] == group]
            group_counts[index, 0] = group_data.shape[0]
            group_counts[index, 1] = np.sum(~group_data['augmented'])

        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes.bar(x, group_counts[:, 0], width=0.8, label='All data')
        axes.bar(x, group_counts[:, 1], width=0.8, alpha=0.5, label='Unaugmented')
        axes.set_xticks(x)
        axes.set_xticklabels(self.data_params['groups'])
        axes.tick_params(axis='x', rotation=90)
        axes.set_ylabel('Number of Entries')
        axes.set_xlabel('Group')
        axes.legend()
        fig.tight_layout()
        fig.savefig(f'{self.save_to["data"]}/group_counts.png')
        plt.close()

        # Histogram of Bravais lattices
        x = np.arange(len(self.data_params['bravais_lattices']))
        bl_counts = np.zeros((len(self.data_params['bravais_lattices']), 2))
        for index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            bl_data = self.data[self.data['bravais_lattice'] == bravais_lattice]
            bl_counts[index, 0] = bl_data.shape[0]
            bl_counts[index, 1] = np.sum(~bl_data['augmented'])

        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes.bar(x, bl_counts[:, 0], width=0.8, label='All data')
        axes.bar(x, bl_counts[:, 1], width=0.8, alpha=0.5, label='Unaugmented')
        axes.set_xticks(x)
        axes.set_xticklabels(self.data_params['bravais_lattices'])
        axes.set_ylabel('Number of Entriey')
        axes.set_xlabel('Bravais Lattice')
        axes.legend()
        fig.tight_layout()
        fig.savefig(f'{self.save_to["data"]}/bl_counts.png')
        plt.close()

        # histogram of true hkl order at peak position.
        unaugmented_data = self.data[~self.data['augmented']]
        make_hkl_plot(
            data=unaugmented_data,
            n_points=self.data_params['n_points'],
            hkl_ref_length=self.data_params['hkl_ref_length'],
            save_to=f'{self.save_to["data"]}/hkl_labels_unaugmented.png',
            )
        if self.data_params['augment']:
            augmented_data = self.data[self.data['augmented']]
            if len(augmented_data) > 0:
                make_hkl_plot(
                    data=augmented_data,
                    n_points=self.data_params['n_points'],
                    hkl_ref_length=self.data_params['hkl_ref_length'],
                    save_to=f'{self.save_to["data"]}/hkl_labels_augmented.png',
                    )

        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            make_hkl_plot(
                data=unaugmented_data[unaugmented_data['bravais_lattice'] == bravais_lattice],
                n_points=self.data_params['n_points'],
                hkl_ref_length=self.data_params['hkl_ref_length'],
                save_to=f'{self.save_to["data"]}/hkl_labels_unaugmented_{bravais_lattice}.png',
                )
            if self.data_params['augment']:
                bl_augmented_data = augmented_data[augmented_data['bravais_lattice'] == bravais_lattice]
                if len(bl_augmented_data) > 0:
                    make_hkl_plot(
                        data=bl_augmented_data,
                        n_points=self.data_params['n_points'],
                        hkl_ref_length=self.data_params['hkl_ref_length'],
                        save_to=f'{self.save_to["data"]}/hkl_labels_augmented_{bravais_lattice}.png',
                        )

        # Histograms of inputs
        fig, axes = plt.subplots(2, 8, figsize=(14, 5))
        bins_scaled = np.linspace(-4, 4, 101)
        centers_scaled = (bins_scaled[1:] + bins_scaled[:-1]) / 2
        dbin_scaled = bins_scaled[1] - bins_scaled[0]

        # q2
        data = self.data[~self.data['augmented']]
        q2_scaled = np.stack(data['q2_scaled']).ravel()
        q2_sorted = np.sort(np.stack(data['q2']).ravel())
        lower = q2_sorted[int(0.005*q2_sorted.size)]
        upper = q2_sorted[int(0.995*q2_sorted.size)]
        bins = np.linspace(lower, upper, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        hist, _ = np.histogram(q2_sorted, bins=bins, density=True)
        axes[0, 0].bar(centers, hist, width=dbin, label='Unaugmented')
        if self.data_params['augment']:
            data_augmented = self.data[self.data['augmented']]
            q2_augmented = np.stack(data_augmented['q2']).ravel()
            hist_augmented, _ = np.histogram(q2_augmented, bins=bins, density=True)
            axes[0, 0].bar(centers, hist_augmented, width=dbin, alpha=0.5, label='Augmented')
        hist_scaled, _ = np.histogram(q2_scaled, bins=bins_scaled, density=True)
        axes[1, 0].bar(centers_scaled, hist_scaled, width=dbin_scaled)
        axes[0, 0].legend()

        # volume
        volume_scaled = np.array(data[f'{self.volume_key}_scaled'])
        volume_sorted = np.sort(np.array(data[self.volume_key])) / plot_volume_scale
        lower = volume_sorted[int(0.005*volume_sorted.size)]
        upper = volume_sorted[int(0.995*volume_sorted.size)]
        bins = np.linspace(lower, upper, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        hist, _ = np.histogram(volume_sorted, bins=bins, density=True)
        axes[0, 1].bar(centers, hist, width=dbin)
        if self.data_params['augment']:
            volume_augmented = np.array(data_augmented[self.volume_key]) / plot_volume_scale
            hist_augmented, _ = np.histogram(volume_augmented, bins=bins, density=True)
            axes[0, 1].bar(centers, hist_augmented, width=dbin, alpha=0.5)
        hist_scaled, _ = np.histogram(volume_scaled, bins=bins_scaled, density=True)
        axes[1, 1].bar(centers_scaled, hist_scaled, width=dbin_scaled)

        # Unit cell
        unit_cell = np.stack(data[self.unit_cell_key])
        if self.data_params['augment']:
            unit_cell_augmented = np.stack(data_augmented[self.unit_cell_key])
        unit_cell_scaled = np.stack(data[f'{self.unit_cell_key}_scaled'])
        sorted_lengths = np.sort(unit_cell[:, :3].ravel())
        lower = sorted_lengths[int(0.005*sorted_lengths.size)]
        upper = sorted_lengths[int(0.995*sorted_lengths.size)]
        bins = np.linspace(lower, upper, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        for index in range(3):
            hist, _ = np.histogram(unit_cell[:, index], bins=bins, density=True)
            axes[0, index + 2].bar(centers, hist, width=dbin)
            if self.data_params['augment']:
                hist_augmented, _ = np.histogram(unit_cell_augmented[:, index], bins=bins, density=True)
                axes[0, index + 2].bar(centers, hist_augmented, width=dbin, alpha=0.5)
            hist_scaled, _ = np.histogram(
                unit_cell_scaled[:, index], bins=bins_scaled, density=True
                )
            axes[1, index + 2].bar(centers_scaled, hist_scaled, width=dbin_scaled)
            axes[0, index + 2].set_title(y_labels[index])

        sorted_angles = np.sort(unit_cell[:, 3:].ravel())
        lower = sorted_angles[int(0.005*sorted_angles.size)]
        upper = sorted_angles[int(0.995*sorted_angles.size)]
        bins = np.linspace(lower, upper, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        for index in range(3, 6):
            hist, _ = np.histogram(unit_cell[:, index], bins=bins, density=True)
            axes[0, index + 2].bar(centers, hist, width=dbin)
            if self.data_params['augment']:
                hist_augmented, _ = np.histogram(unit_cell_augmented[:, index], bins=bins, density=True)
                axes[0, index + 2].bar(centers, hist_augmented, width=dbin, alpha=0.5)

            hist_scaled, _ = np.histogram(unit_cell_scaled[:, index], bins=bins_scaled, density=True)
            axes[1, index + 2].bar(centers_scaled, hist_scaled, width=dbin_scaled)
            axes[0, index + 2].set_title(y_labels[index])

        axes[0, 0].set_ylabel('Raw data')
        axes[1, 0].set_ylabel('Standard Scaling')
        axes[0, 0].set_title('q2')
        axes[0, 1].set_title(f'Volume\n(x{plot_volume_scale})')
        fig.tight_layout()
        fig.savefig(f'{self.save_to["data"]}/regression_inputs.png')
        plt.close()

        # PCA components
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        unit_cell_cov = np.cov(unit_cell[:, self.data_params['y_indices']].T)
        unit_cell_scaled_cov = np.cov(unit_cell_scaled[:, self.data_params['y_indices']].T)
        cov_display = ConfusionMatrixDisplay(
            confusion_matrix=unit_cell_cov,
            display_labels=self.uc_labels,
            )
        cov_scaled_display = ConfusionMatrixDisplay(
            confusion_matrix=unit_cell_scaled_cov,
            display_labels=self.uc_labels,
            )
        cov_display.plot(ax=axes[0], colorbar=False, values_format='0.2f')
        cov_scaled_display.plot(ax=axes[1], colorbar=False, values_format='0.2f')
        axes[0].set_title('Unit cell covariance')
        axes[1].set_title('Scaled unit cell covariance')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        fig.tight_layout()
        fig.savefig(f'{self.save_to["data"]}/covariance_inputs.png')
        plt.close()

        # what is the order of the unit cell lengths
        if self.data_params['lattice_system'] == 'orthorhombic':
            print(self.unit_cell_key)
            unit_cell = np.stack(self.data[~self.data['augmented']][self.unit_cell_key])
            order = np.argsort(unit_cell[:, :3], axis=1)
            # order: [[shortest index, middle index, longest index], ... ]
            proportions = np.zeros((3, 3))
            if self.data_params['augment']:
                unit_cell_aug = np.stack(self.data[self.data['augmented']][self.unit_cell_key])
                order_aug = np.argsort(unit_cell_aug[:, :3], axis=1)
                proportions_aug = np.zeros((3, 3))
            for length_index in range(3):
                for uc_index in range(3):
                    proportions[length_index, uc_index] = np.sum(order[:, length_index] == uc_index)
                    if self.data_params['augment']:
                        proportions_aug[length_index, uc_index] = np.sum(order_aug[:, length_index] == uc_index)
            fig, axes = plt.subplots(1, 3, figsize=(8, 4))
            for length_index in range(3):
                axes[length_index].plot(proportions[length_index], marker='.', markersize=20, label='Unaugmented')
                if self.data_params['augment']:
                    axes[length_index].plot(proportions_aug[length_index], marker='v', markersize=10, label='Augmented')
                axes[length_index].set_xticks([0, 1, 2])
                axes[length_index].set_xticklabels(['a', 'b', 'c'])
            axes[0].legend()
            axes[0].set_title('Shortest axis position')
            axes[1].set_title('Middle axis position')
            axes[2].set_title('Longest axis position')
            fig.tight_layout()
            fig.savefig(f'{self.save_to["data"]}/axis_order.png')
            plt.close()

    def setup_regression(self):
        self.unit_cell_generator = dict.fromkeys(self.data_params['groups'])
        for group_index, group in enumerate(self.data_params['groups']):
            self.unit_cell_generator[group] = Regression_AlphaBeta(
                group,
                self.data_params,
                self.reg_params[group],
                self.save_to['regression'],
                self.unit_cell_key,
                self.random_seed,
                )
            self.unit_cell_generator[group].setup()
            if self.reg_params[group]['load_from_tag']:
                self.unit_cell_generator[group].load_from_tag()
            else:
                group_indices = self.data['group'] == group
                self.unit_cell_generator[group].train_regression(data=self.data[group_indices])

    def inferences_regression(self):
        uc_pred_scaled = np.zeros((len(self.data), self.data_params['n_outputs']))
        uc_pred_scaled_cov = np.zeros((len(self.data), self.data_params['n_outputs'], self.data_params['n_outputs']))

        for group_index, group in enumerate(self.data_params['groups']):
            group_indices = self.data['group'] == group
            uc_pred_scaled[group_indices, :], uc_pred_scaled_cov[group_indices, :, :] = \
                self.unit_cell_generator[group].do_predictions(data=self.data[group_indices], batch_size=1024)

        uc_pred, uc_pred_cov = self.revert_predictions(uc_pred_scaled, uc_pred_scaled_cov)
        self.data[self.volume_key + '_pred'] = list(self.infer_unit_cell_volume(uc_pred))
        self.data[self.unit_cell_key + '_pred'] = list(uc_pred)
        self.data[self.unit_cell_key + '_pred_cov'] = list(uc_pred_cov)
        self.data[self.unit_cell_key + '_pred_scaled'] = list(uc_pred_scaled)
        self.data[self.unit_cell_key + '_pred_scaled_cov'] = list(uc_pred_scaled_cov)

    def revert_predictions(self, uc_pred_scaled=None, uc_pred_scaled_cov=None):
        if not uc_pred_scaled is None:
            uc_pred = np.zeros(uc_pred_scaled.shape)
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred = uc_pred_scaled * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
            elif self.data_params['lattice_system'] == 'monoclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3] = uc_pred_scaled[:, 3] + np.pi/2
            elif self.data_params['lattice_system'] == 'triclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3:] = uc_pred_scaled[:, 3:] + np.pi/2
            elif self.data_params['lattice_system'] == 'rhombohedral':
                uc_pred[:, 0] = uc_pred_scaled[:, 0] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 1] = uc_pred_scaled[:, 1] + np.pi/2

        if not uc_pred_scaled_cov is None:
            uc_pred_cov = np.zeros(uc_pred_scaled_cov.shape)
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred_cov = uc_pred_scaled_cov * self.uc_scaler.scale_[0]**2
            elif self.data_params['lattice_system'] == 'monoclinic':
                pass
            elif self.data_params['lattice_system'] == 'triclinic':
                pass
            elif self.data_params['lattice_system'] == 'rhombohedral':
                pass

        if not uc_pred_scaled is None and not uc_pred_scaled_cov is None:
            return uc_pred, uc_pred_cov
        elif not uc_pred_scaled is None and uc_pred_scaled_cov is None:
            return uc_pred
        elif uc_pred_scaled is None and not uc_pred_scaled_cov is None:
            return uc_pred_cov

    def scale_predictions(self, uc_pred):
        if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            uc_pred_scaled = (uc_pred - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
        elif self.data_params['lattice_system'] == 'monoclinic':
            uc_pred_scaled = np.zeros(uc_pred.shape)
            uc_pred_scaled[:, :3] = (uc_pred[:, :3] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
            uc_pred_scaled[:, 3] = uc_pred[:, 3] - np.pi/2
        elif self.data_params['lattice_system'] == 'triclinic':
            uc_pred_scaled = np.zeros(uc_pred.shape)
            uc_pred_scaled[:, :3] = (uc_pred[:, :3] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
            uc_pred_scaled[:, 3:] = uc_pred[:, 3:] - np.pi/2
        elif self.data_params['lattice_system'] == 'rhombohedral':
            uc_pred_scaled = np.zeros(uc_pred.shape)
            uc_pred_scaled[:, 0] = (uc_pred[:, 0] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
            uc_pred_scaled[:, 1] = uc_pred[:, 1] - np.pi/2
        return uc_pred_scaled

    def evaluate_regression(self):
        for bravais_lattice in self.data_params['bravais_lattices']:
            evaluate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=self.unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg.png',
                y_indices=self.data_params['y_indices']
                )
            calibrate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=self.unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_calibration.png',
                y_indices=self.data_params['y_indices']
                )
        for group in self.data_params['groups']:
            evaluate_regression(
                data=self.data[self.data['group'] == group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=self.unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{group}_reg.png',
                y_indices=self.data_params['y_indices']
                )
            calibrate_regression(
                data=self.data[self.data['group'] == group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=self.unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{group}_reg_calibration.png',
                y_indices=self.data_params['y_indices']
                )
        evaluate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key=self.unit_cell_key,
            save_to_name=f'{self.save_to["regression"]}/All_reg.png',
            y_indices=self.data_params['y_indices']
            )
        calibrate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key=self.unit_cell_key,
            save_to_name=f'{self.save_to["regression"]}/All_reg_calibration.png',
            y_indices=self.data_params['y_indices']
            )

    def setup_assignment(self):
        self.assigner = dict.fromkeys(self.assign_params.keys())
        for key in self.assign_params.keys():
            self.assigner[key] = Assigner(
                self.data_params,
                self.assign_params[key],
                self.hkl_ref,
                self.uc_scaler,
                self.q2_scaler,
                self.save_to['assigner']
                )
            if self.assign_params[key]['load_from_tag']:
                self.assigner[key].load_from_tag(
                    self.assign_params[key]['tag'],
                    self.assign_params[key]['mode']
                    )
            else:
                if self.assign_params[key]['train_on'] == 'perturbed':
                    unit_cell_scaled_key = f'{self.unit_cell_key}_scaled'
                    y_indices = self.data_params['y_indices']
                elif self.assign_params[key]['train_on'] == 'predicted':
                    unit_cell_scaled_key = f'{self.unit_cell_key}_pred_scaled'
                    y_indices = None
                self.assigner[key].fit_model(
                    data=self.data[~self.data['augmented']],
                    unit_cell_scaled_key=unit_cell_scaled_key,
                    y_indices=y_indices,
                    )

    def inferences_assignment(self, keys):
        for key in keys:
            if self.assign_params[key]['train_on'] == 'perturbed':
                unit_cell_scaled_key = f'{self.unit_cell_key}_scaled'
                y_indices = self.data_params['y_indices']
            elif self.assign_params[key]['train_on'] == 'predicted':
                unit_cell_scaled_key = f'{self.unit_cell_key}_pred_scaled'
                y_indices = None

            unaugmented_data = self.data[~self.data['augmented']].copy()
            softmaxes = self.assigner[key].do_predictions(
                unaugmented_data,
                unit_cell_scaled_key=unit_cell_scaled_key,
                y_indices=y_indices,
                reload_model=False,
                batch_size=1024,
                )
            hkl_pred = self.convert_softmax_to_assignments(softmaxes)
            hkl_assign = softmaxes.argmax(axis=2)
            """
            indices = ~self.data['augmented']
            pairwise_differences_scaled = self.assigner[key].pairwise_difference_calculation.get_pairwise_differences_from_uc_scaled(
                tf.convert_to_tensor(np.stack(self.data[unit_cell_scaled_key])[:, y_indices]),
                tf.convert_to_tensor(np.stack(self.data['q2_scaled']))
                )
            np.save('pds.npy', pairwise_differences_scaled.numpy()[indices])
            np.save('hkl_labels_true.npy', np.stack(self.data['hkl_labels'][indices]))
            np.save('bl_labels.npy', np.stack(self.data['bravais_lattice_label'][indices]))
            np.save('hkl_labels_pred.npy', hkl_assign[indices])
            np.save('softmaxes.npy', softmaxes[indices])
            """

            unaugmented_data['hkl_labels_pred'] = list(hkl_assign)
            unaugmented_data['hkl_pred'] = list(hkl_pred)
            unaugmented_data['hkl_softmaxes'] = list(softmaxes)
            self.assigner[key].evaluate(
                unaugmented_data,
                self.data_params['bravais_lattices'],
                unit_cell_scaled_key=unit_cell_scaled_key,
                y_indices=y_indices,
                perturb_std=self.assign_params[key]['perturb_std']
                )

            self.assigner[key].calibrate(unaugmented_data)

    def convert_softmax_to_assignments(self, softmaxes):
        n_entries = softmaxes.shape[0]
        hkl_assign = softmaxes.argmax(axis=2)
        hkl_pred = np.zeros((n_entries, self.data_params['n_points'], 3))
        for entry_index in range(n_entries ):
            hkl_pred[entry_index] = self.hkl_ref[hkl_assign[entry_index]]
        return hkl_pred
