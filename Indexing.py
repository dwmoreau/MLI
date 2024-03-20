"""
lattice system | accuracy
-------------------------
cubic          | 99.5%
orthorhombic   | 90%
Monoclinic
    {'Not found': 157, 'Found and best': 86, 'Found but not best': 28, 'Found but off by two': 0, 'Found explainers': 66}

* Startover
    * Redo Generate dataset
        * orthorhombic
        * monoclinic

        - stricter intensity requirement
            - I > 0.005
            - d2I < -1
            - breadth = 0.10
    * Get reasonable hyperparameters for NN
    * Retest optimization

- Documentation
    * Add figures to Methods.md
    - Update README.md
    - Reread papers on indexing and update document
    - Reread ML technique papers
    - read about powder extinction classes
        Hahn, T., Ed. International Tables for X-ray Crystallography Volume A (Space Group Symmetry); Kluwer Academic Publishers: Dordrecht, The Netherlands, 1989

- Optimization:
    * profile with orthorhombic
    * Full softmax array optimization - Actual likelihood target function
    - What differentiates a found / not found entry
    - common assignments:
        - drop during optimization but include in loss
        - use all hkl assignments with largest N likelihoods

- Indexing.py

- Augmentation
    - Edit so the peak distributions match

- Assignments
    * smallest model that works
    - Figure out how to take a more holistic look at the pairwise difference array.
    - Penalize multiple assignments to the same hkl

- Data
    * Fix Hexagonal / Rhombohedral setting issues.
    - experimental data from rruff
        - verify that unit cell is consistent with diffraction
    - redo dataset generation with new parameters based on RRUFF database
    - Rewrite GenerateDataset.py
    - Get data from other databases:
        - Materials project
        - ICSD

- SWE:
    * Refactor code to work with reciprocal space unit cell parameters
        - verify conversions in Utilities.py
        - Refactor Optimizer.py
        - Refactor training
    * CombineDatabases.py bug where entries are being added more than once.
    * memory leak during cyclic training
        - Try saving and loading weights with two different models
    - MPI error: https://github.com/pmodels/mpich/issues/6547
    - get working on dials
        - generate data
        - get materials project cif files
        - redo counts & groups
        - get MLI working
        - Train ML models
    - profile optimization

- Regression:
    - prediction of PCA components
        - evaluation of fitting in the PCA / Scaled space
        - evaluation of covariance
    - Read about ensembles of weak learners - what other types of models can I utilize
    - Improve hyperparameters
    - read Stirn 2023 and implement
    Detlefsen 2019:
        - cluster input d-spacings
          - map d-spacings onto a single scalar correlated with volume
        - ??? extrapolation architecture
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Assigner import Assigner
from Augmentor import Augmentor
from Evaluations import evaluate_regression
from Evaluations import calibrate_regression
from Reindexing import get_permutation
from Reindexing import unpermute_monoclinic_partial_unit_cell
from Regression import Regression_AlphaBeta
from Utilities import Q2Calculator
from Utilities import read_params
from Utilities import write_params


class Indexing:
    def __init__(self, assign_params=None, aug_params=None, data_params=None, reg_params=None, seed=12345):
        self.random_seed = seed
        self.rng = np.random.default_rng(self.random_seed)
        self.n_generated_points = 60  # This is the peak length of the generated dataset.
        self.save_to = dict.fromkeys(['results', 'data', 'regression', 'assigner', 'augmentor'])
        self.assign_params = assign_params
        self.aug_params = aug_params
        self.data_params = data_params
        self.reg_params = reg_params

        results_directory = os.path.join(self.data_params['base_directory'], 'models', self.data_params['tag'])
        self.save_to = {
            'results': results_directory,
            'assigner': os.path.join(results_directory, 'assigner'),
            'augmentor': os.path.join(results_directory, 'augmentor'),
            'data': os.path.join(results_directory, 'data'),
            'regression': os.path.join(results_directory, 'regression'),
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
            'data_dir': os.path.join(self.data_params['base_directory'], 'data'),
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
        elif self.data_params['lattice_system'] == 'tetragonal':
            self.data_params['y_indices'] = [0, 2]
            self.data_params['bravais_lattices'] = ['tI', 'tP']
        elif self.data_params['lattice_system'] == 'orthorhombic':
            self.data_params['y_indices'] = [0, 1, 2]
            self.data_params['bravais_lattices'] = ['oC', 'oF', 'oI', 'oP']
        elif self.data_params['lattice_system'] == 'monoclinic':
            self.data_params['y_indices'] = [0, 1, 2, 4]
            self.data_params['bravais_lattices'] = ['mC', 'mP']
        elif self.data_params['lattice_system'] == 'triclinic':
            self.data_params['y_indices'] = [0, 1, 2, 3, 4, 5]
            self.data_params['bravais_lattices'] = ['aP']
        elif self.data_params['lattice_system'] == 'rhombohedral':
            self.data_params['y_indices'] = [0, 3]
        elif self.data_params['lattice_system'] == 'hexagonal':
            self.data_params['y_indices'] = [0, 2]
            self.data_params['bravais_lattices'] = ['hP']
        self._setup_joint()

    def setup_from_tag(self):
        self.angle_scale = np.load(f'{self.save_to["data"]}/angle_scale.npy')
        self.uc_scaler = joblib.load(f'{self.save_to["data"]}/uc_scaler.bin')
        self.volume_scaler = joblib.load(f'{self.save_to["data"]}/volume_scaler.bin')
        self.q2_scaler = joblib.load(f'{self.save_to["data"]}/q2_scaler.bin')

        params = read_params(f'{self.save_to["data"]}/data_params.csv')
        data_params_keys = [
            'augment',
            'bravais_lattices'
            'lattice_system',
            'data_dir',
            'train_fraction',
            'n_max',
            'y_indices',
            'n_points',
            'points_tag',
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
        if self.data_params['lattice_system'] in ['cubic', 'orthorhombic', 'monoclinic']:
            self.data_params['split_groups'] = self.data_params['groups']
        elif self.data_params['lattice_system'] == 'tetragonal':
            self.data_params['split_groups'] = []
            for group in self.data_params['groups']:
                self.data_params['split_groups'].append(group.replace('tetragonal_', 'tetragonal_0_'))
                self.data_params['split_groups'].append(group.replace('tetragonal_', 'tetragonal_1_'))

        self.group_mappings = dict.fromkeys(group_spec['hm symbol'].unique())
        for index in range(len(group_spec)):
            self.group_mappings[group_spec.iloc[index]['hm symbol']] = group_spec.iloc[index]['group']
        #for key in self.group_mappings.keys():
        #    print(f'{key} -> {self.group_mappings[key]}')

    def load_data(self):
        read_columns = [
            'lattice_system',
            'bravais_lattice',
            'spacegroup_number',
            'volume',
            'spacegroup_symbol_hm',
            'reindexed_spacegroup_symbol_hm',
            'unit_cell',
            'reindexed_unit_cell',
            f'd_spacing_{self.data_params["points_tag"]}',
            f'h_{self.data_params["points_tag"]}',
            f'k_{self.data_params["points_tag"]}',
            f'l_{self.data_params["points_tag"]}',
            f'reindexed_h_{self.data_params["points_tag"]}',
            f'reindexed_k_{self.data_params["points_tag"]}',
            f'reindexed_l_{self.data_params["points_tag"]}',
            ]

        if self.data_params['augment']:
            read_columns += [
                'd_spacing_sa',
                'h_sa', 'k_sa', 'l_sa',
                'reindexed_h_sa', 'reindexed_k_sa', 'reindexed_l_sa',
                ]

        data = []
        for index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            file_name = os.path.join(
                self.data_params['data_dir'],
                f'GeneratedDatasets/dataset_{bravais_lattice}.parquet'
                )
            print(f'Loading data from {file_name}')
            data.append(pd.read_parquet(file_name, columns=read_columns))
        self.data = pd.concat(data, ignore_index=True)

        # Remove data that doesn't have enough peaks
        # A total of 60 or so peaks are included in the data set - for all entries
        # If there were less than 60 peaks, those get padded with zeros at the end of the array.
        #   - the 60 number is arbitrary and set in GenerateDataset.py
        points = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        indices = points.apply(len) >= self.data_params['n_points']
        self.data = self.data.loc[indices]
        points = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        enough_peaks = points.apply(np.count_nonzero) >= self.data_params['n_points']
        self.data = self.data.loc[enough_peaks]
        self.data['augmented'] = np.zeros(self.data.shape[0], dtype=bool)

        # Add label to data and down sample
        self.data['group'] = self.data['reindexed_spacegroup_symbol_hm'].map(
            lambda x: self.group_mappings[x]
            )
        if self.data_params['lattice_system'] in ['cubic', 'orthorhombic']:
            self.data['split_group'] = self.data['group']
        elif self.data_params['lattice_system'] == 'tetragonal':
            # split_0: a < c
            # split_1: a > c
            unit_cell = np.stack(self.data['reindexed_unit_cell'])[:, [0, 2]]
            split_1 = unit_cell[:, 0] > unit_cell[:, 1]
            self.data['split_group'] = self.data['group'].map(
                lambda x: x.replace('tetragonal_', 'tetragonal_0_')
                )
            self.data.loc[split_1, 'split_group'] = self.data.loc[split_1, 'split_group'].map(
                lambda x: x.replace('tetragonal_0_', 'tetragonal_1_')
                )
        elif self.data_params['lattice_system'] == 'monoclinic':
            self.data['split_group'] = self.data['group']
            # make reindexed unit cell alpha = gamma = nan and beta != 90
            # This is not true, but makes it compatible with y_indices = [0, 1, 2, 4]
            reindexed_unit_cell = np.stack(self.data['reindexed_unit_cell'])
            new_reindexed_unit_cell = reindexed_unit_cell.copy()
            new_reindexed_unit_cell[:, [3, 5]] = np.nan
            reindexed_angle_index = np.zeros(len(self.data), dtype=int)
            for entry_index in range(len(self.data)):
                # conversion to radians happens latter
                reindexed_angle_index[entry_index] = 3 + np.argwhere(reindexed_unit_cell[entry_index, 3:] != 90)
                new_reindexed_unit_cell[entry_index, 4] = reindexed_unit_cell[entry_index, reindexed_angle_index[entry_index]]
            self.data['reindexed_unit_cell'] = list(new_reindexed_unit_cell)
            self.data['reindexed_angle_index'] = list(reindexed_angle_index)

        data_grouped = self.data.groupby('split_group')
        data_group = [None for _ in range(len(data_grouped.groups.keys()))]
        for index, group in enumerate(data_grouped.groups.keys()):
            data_group[index] = data_grouped.get_group(group)
            data_group[index].insert(loc=0, column='split_group_label', value=index * np.ones(len(data_group[index])))
            data_group[index] = data_group[index].sample(
                n=min(len(data_group[index]), self.data_params['n_max']),
                replace=False,
                random_state=self.random_seed
                )
        self.data = pd.concat(data_group, ignore_index=True)

        d_spacing_pd = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        d_spacing = [np.zeros(self.data_params['n_points']) for _ in range(self.data.shape[0])]
        for entry_index in range(self.data.shape[0]):
            d_spacing[entry_index] = d_spacing_pd.iloc[entry_index][:self.data_params['n_points']]
        self.data['d_spacing'] = d_spacing
        # The math is a bit easier logistically when we use q**2
        self.data['q2'] = 1 / self.data['d_spacing']**2
        if self.data_params['augment']:
            # When augmenting, we need all q2 for all the peaks
            #   *_sa: all nonsystematically absent peaks
            #   *_points_tag: Should be 'intersect', the tag for peaks dropped in a realistic manner.
            self.data['q2_sa'] = 1 / self.data['d_spacing_sa']**2
            self.data[f'q2_{self.data_params["points_tag"]}'] = 1 / self.data[f'd_spacing_{self.data_params["points_tag"]}']**2

        # This sets up the training / validation tags so that the validation set is taken
        # evenly from the spacegroup symbols. This is the finests hierarchy of the data
        train_label = np.ones(self.data.shape[0], dtype=bool)
        for symbol_index, symbol in enumerate(self.data['reindexed_spacegroup_symbol_hm'].unique()):
            indices = np.where(self.data['reindexed_spacegroup_symbol_hm'] == symbol)[0]
            n_val = int(indices.size * (1 - self.data_params['train_fraction']))
            val_indices = self.rng.choice(indices, size=n_val, replace=False)
            train_label[val_indices] = False
        self.data['train'] = train_label

        # put the hkl's together
        # I am saving the data into parquet format. It does not allow saving 2D arrays, so like hkl (20 x 3).
        # Converting to hdf5 would be very helpful here.
        hkl = np.zeros((len(self.data), self.data_params['n_points'], 3), dtype=int)
        reindexed_hkl = np.zeros((len(self.data), self.data_params['n_points'], 3), dtype=int)
        if self.data_params['augment']:
            hkl_sa = np.zeros((len(self.data), self.n_generated_points, 3), dtype=int)
            reindexed_hkl_sa = np.zeros((len(self.data), self.n_generated_points, 3), dtype=int)
        for entry_index in range(len(self.data)):
            entry = self.data.iloc[entry_index]
            hkl[entry_index, :, 0] = entry[f'h_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            hkl[entry_index, :, 1] = entry[f'k_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            hkl[entry_index, :, 2] = entry[f'l_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            reindexed_hkl[entry_index, :, 0] = entry[f'reindexed_h_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            reindexed_hkl[entry_index, :, 1] = entry[f'reindexed_k_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            reindexed_hkl[entry_index, :, 2] = entry[f'reindexed_l_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            if self.data_params['augment']:
                n_peaks_sa = entry[f'h_sa'].size
                hkl_sa[entry_index, :n_peaks_sa, 0] = entry['h_sa']
                hkl_sa[entry_index, :n_peaks_sa, 1] = entry['k_sa']
                hkl_sa[entry_index, :n_peaks_sa, 2] = entry['l_sa']
                reindexed_hkl_sa[entry_index, :n_peaks_sa, 0] = entry['reindexed_h_sa']
                reindexed_hkl_sa[entry_index, :n_peaks_sa, 1] = entry['reindexed_k_sa']
                reindexed_hkl_sa[entry_index, :n_peaks_sa, 2] = entry['reindexed_l_sa']
        self.data['hkl'] = list(hkl)
        self.data['reindexed_hkl'] = list(reindexed_hkl)
        if self.data_params['augment']:
            self.data['hkl_sa'] = list(hkl_sa)
            self.data['reindexed_hkl_sa'] = list(reindexed_hkl_sa)
        drop_columns = [
            f'h_{self.data_params["points_tag"]}',
            f'k_{self.data_params["points_tag"]}',
            f'l_{self.data_params["points_tag"]}',
            f'reindexed_h_{self.data_params["points_tag"]}',
            f'reindexed_k_{self.data_params["points_tag"]}',
            f'reindexed_l_{self.data_params["points_tag"]}',
            ]
        if self.data_params['augment']:
            drop_columns += [
                'h_sa', 'k_sa', 'l_sa',
                'reindexed_h_sa', 'reindexed_k_sa', 'reindexed_l_sa',
                ]
        self.data.drop(columns=drop_columns, inplace=True)

        # Convert angles to radians
        unit_cell = np.stack(self.data['unit_cell'])
        unit_cell[:, 3:] = np.pi/180 * unit_cell[:, 3:]
        self.data['unit_cell'] = list(unit_cell)
        reindexed_unit_cell = np.stack(self.data['reindexed_unit_cell'])
        reindexed_unit_cell[:, 3:] = np.pi/180 * reindexed_unit_cell[:, 3:]
        self.data['reindexed_unit_cell'] = list(reindexed_unit_cell)
        self.setup_scalers()

        if self.data_params['augment']:
            self.augment_data()
            drop_columns = [
                f'q2_{self.data_params["points_tag"]}',
                'q2_sa',
                'hkl_sa',
                'reindexed_hkl_sa',
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
        if 'h' in self.data.keys():
            hkl = np.stack([
                np.stack(self.data['h'], axis=0)[:, :, np.newaxis],
                np.stack(self.data['k'], axis=0)[:, :, np.newaxis],
                np.stack(self.data['l'], axis=0)[:, :, np.newaxis]
                ], axis=2
                )
            self.data['hkl'] = list(hkl)
            self.data.drop(
                columns=['h', 'k', 'l'],
                inplace=True
                )
        if 'reindexed_h' in self.data.keys():
            reindexed_hkl = np.stack([
                np.stack(self.data['reindexed_h'], axis=0)[:, :, np.newaxis],
                np.stack(self.data['reindexed_k'], axis=0)[:, :, np.newaxis],
                np.stack(self.data['reindexed_l'], axis=0)[:, :, np.newaxis]
                ], axis=2
                )
            self.data['reindexed_hkl'] = list(reindexed_hkl)
            self.data.drop(
                columns=['reindexed_h', 'reindexed_k', 'reindexed_l'],
                inplace=True
                )
        if not load_augmented:
            self.data = self.data[~self.data['augmented']]
        if not load_train:
            self.data = self.data[~self.data['train']]

    def save(self):
        hkl = np.stack(self.data['hkl'])
        self.data['h'] = list(hkl[:, :, 0])
        self.data['k'] = list(hkl[:, :, 1])
        self.data['l'] = list(hkl[:, :, 2])
        reindexed_hkl = np.stack(self.data['reindexed_hkl'])
        self.data['reindexed_h'] = list(reindexed_hkl[:, :, 0])
        self.data['reindexed_k'] = list(reindexed_hkl[:, :, 1])
        self.data['reindexed_l'] = list(reindexed_hkl[:, :, 2])
        self.data.drop(columns=['hkl', 'reindexed_hkl'], inplace=True)
        self.data.to_parquet(f'{self.save_to["data"]}/data.parquet')

        np.save(f'{self.save_to["data"]}/hkl_ref.npy', self.hkl_ref)
        np.save(f'{self.save_to["data"]}/angle_scale.npy', self.angle_scale)
        joblib.dump(self.uc_scaler, f'{self.save_to["data"]}/uc_scaler.bin')
        joblib.dump(self.volume_scaler, f'{self.save_to["data"]}/volume_scaler.bin')
        joblib.dump(self.q2_scaler, f'{self.save_to["data"]}/q2_scaler.bin')
        write_params(self.data_params, f'{self.save_to["data"]}/data_params.csv')

    def setup_hkl(self):
        print('Setting up the hkl labels')
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
        if self.data_params['lattice_system'] == 'monoclinic':
            unit_cell_key = 'unit_cell'
            hkl_key = 'hkl'
        else:
            unit_cell_key = 'reindexed_unit_cell'
            hkl_key = 'reindexed_hkl'
        unit_cell = np.stack(train[unit_cell_key])[:, self.data_params['y_indices']]
        q2_ref_calculator = Q2Calculator(self.data_params['lattice_system'], self.hkl_ref, tensorflow=False)
        q2_ref = q2_ref_calculator.get_q2(unit_cell)
        sort_indices = np.argsort(q2_ref.mean(axis=0))
        self.hkl_ref = self.hkl_ref[sort_indices][:self.data_params['hkl_ref_length'] - 1]
        self.hkl_ref = np.concatenate((self.hkl_ref, np.zeros((1, 3))), axis=0)
        hkl = np.stack(self.data[hkl_key])

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
                hkl[:, :, 0]**2 + hkl[:, :, 1]**2 + hkl[:, :, 2]**2,
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
        # This is slow, and I am sure it could be sped up.
        for entry_index in tqdm(range(len(self.data))):
            for point_index in range(self.data_params['n_points']):

                hkl_ref_index = np.argwhere(np.all(
                    check_ref[:, :] == check_data[entry_index, point_index, :],
                    axis=1
                    ))
                if len(hkl_ref_index) == 1:
                    hkl_labels[entry_index, point_index] = hkl_ref_index
                    """
                    if hkl_ref_index < point_index:
                        print(check_ref[:, :])
                        print(self.data.iloc[entry_index])
                        print(f'{point_index} {hkl_ref_index} {check_data[entry_index, point_index, :]}')
                        print(self.data.iloc[entry_index]['hkl'])
                        print(self.data.iloc[entry_index]['q2'])
                        print()
                    """
        self.data['hkl_labels'] = list(hkl_labels)

    def augment_data(self):
        self.augmentor = Augmentor(
            aug_params=self.aug_params,
            data_params=self.data_params,
            min_unit_cell_scaled=self.min_unit_cell_scaled,
            n_generated_points=self.n_generated_points,
            save_to=self.save_to['augmentor'],
            seed=self.random_seed,
            uc_scaler=self.uc_scaler,
            angle_scale=self.angle_scale,
            )
        self.augmentor.setup(self.data)
        data_augmented = [None for _ in range(len(self.data_params['split_groups']))]
        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            print(f'Augmenting {split_group}')
            split_group_data = self.data[self.data['split_group'] == split_group]
            data_augmented[split_group_index] = self.augmentor.augment(
                split_group_data, 'reindexed_spacegroup_symbol_hm'
                )
            print(f'  Unaugmented entries: {len(split_group_data)} augmented entries: {len(data_augmented[split_group_index])}')
        data_augmented = pd.concat(data_augmented, ignore_index=True)
        self.data = pd.concat((self.data, data_augmented), ignore_index=True)
        print('Finished Augmenting')

    def infer_unit_cell_volume_from_predictions(self, unit_cell):
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
            volume = unit_cell[:, 0] * unit_cell[:, 1] * unit_cell[:, 2] * np.sin(unit_cell[:, 3])
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
        y_scaled[3:] = (y[3:] - np.pi/2) / self.angle_scale
        return y_scaled

    def y_revert(self, y):
        y_reverted = np.zeros(y.shape)
        y_reverted[:3] = self.uc_scaler.inverse_transform(y[:3][:, np.newaxis])[:, 0]
        y_reverted[3:] = self.angle_scale * y[3:] + np.pi/2
        return y_reverted

    def setup_scalers(self):
        training_data = self.data.loc[np.logical_and(self.data['train'], ~self.data['augmented'])]
        # q2 scaling
        self.q2_scaler = StandardScaler()
        q2_train = np.stack(training_data['q2']).ravel()
        self.q2_scaler.fit(q2_train[:, np.newaxis])
        self.data['q2_scaled'] = self.data['q2'].apply(self.q2_scale)

        # Unit cell parameters scaling
        uc_train = np.stack(training_data['reindexed_unit_cell'])

        # lengths
        self.uc_scaler = StandardScaler()
        self.uc_scaler.fit(uc_train[:, :3].ravel()[:, np.newaxis])
        # angles
        if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic']:
            self.angle_scale = 1
        else:
            if self.data_params['lattice_system'] == 'monoclinic':
                angles = uc_train[:, 4]
            elif self.data_params['lattice_system'] == 'triclinic':
                angles = uc_train[:, 3:].ravel()
            self.angle_scale = angles[angles != np.pi/2].std()

        self.data['reindexed_unit_cell_scaled'] = self.data['reindexed_unit_cell'].apply(self.y_scale)
        self.data['unit_cell_scaled'] = self.data['unit_cell'].apply(self.y_scale)
        # this hard codes the minimum allowed unit cell in augmented data to 1 A
        self.min_unit_cell_scaled = (1 - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]


        # Volume scaling
        self.volume_scaler = StandardScaler()
        volume_train = np.array(training_data['volume'])
        self.volume_scaler.fit(volume_train[:, np.newaxis])
        self.data['volume_scaled'] = list(self.volume_scale(np.array(self.data['volume'])))

    def plot_input(self):
        def make_hkl_plot(data, n_points, hkl_ref_length, save_to):
            fig, axes = plt.subplots(n_points, 1, figsize=(6, 10), sharex=True)
            hkl_labels = np.stack(data['hkl_labels'])  # n_data x n_points
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
        x = np.arange(len(self.data_params['split_groups']))
        group_counts = np.zeros((len(self.data_params['split_groups']), 2))
        for index, group in enumerate(self.data_params['split_groups']):
            group_data = self.data[self.data['split_group'] == group]
            group_counts[index, 0] = group_data.shape[0]
            group_counts[index, 1] = np.sum(~group_data['augmented'])

        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes.bar(x, group_counts[:, 0], width=0.8, label='All data')
        axes.bar(x, group_counts[:, 1], width=0.8, alpha=0.5, label='Unaugmented')
        axes.set_xticks(x)
        axes.set_xticklabels(self.data_params['split_groups'])
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
        hist_scaled, _ = np.histogram(q2_scaled, bins=bins_scaled, density=True)
        axes[1, 0].bar(centers_scaled, hist_scaled, width=dbin_scaled)
        if self.data_params['augment']:
            data_augmented = self.data[self.data['augmented']]
            q2_augmented = np.stack(data_augmented['q2']).ravel()
            hist_augmented, _ = np.histogram(q2_augmented, bins=bins, density=True)
            axes[0, 0].bar(centers, hist_augmented, width=dbin, alpha=0.5, label='Augmented')
            q2_scaled_augmented = np.stack(data_augmented['q2_scaled']).ravel()
            hist_scaled_augmented, _ = np.histogram(q2_scaled_augmented, bins=bins_scaled, density=True)
            axes[1, 0].bar(centers_scaled, hist_scaled_augmented, width=dbin_scaled, alpha=0.5, label='Augmented')
        axes[0, 0].legend()

        # volume
        volume_scaled = np.array(data['volume_scaled'])
        volume_sorted = np.sort(np.array(data['volume'])) / plot_volume_scale
        lower = volume_sorted[int(0.005*volume_sorted.size)]
        upper = volume_sorted[int(0.995*volume_sorted.size)]
        bins = np.linspace(lower, upper, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        hist, _ = np.histogram(volume_sorted, bins=bins, density=True)
        axes[0, 1].bar(centers, hist, width=dbin)
        if self.data_params['augment']:
            volume_augmented = np.array(data_augmented['volume']) / plot_volume_scale
            hist_augmented, _ = np.histogram(volume_augmented, bins=bins, density=True)
            axes[0, 1].bar(centers, hist_augmented, width=dbin, alpha=0.5)
        hist_scaled, _ = np.histogram(volume_scaled, bins=bins_scaled, density=True)
        axes[1, 1].bar(centers_scaled, hist_scaled, width=dbin_scaled)

        # Unit cell
        unit_cell = np.stack(data['reindexed_unit_cell'])
        if self.data_params['augment']:
            unit_cell_augmented = np.stack(data_augmented['reindexed_unit_cell'])
            unit_cell_augmented_scaled = np.stack(data_augmented['reindexed_unit_cell_scaled'])
        unit_cell_scaled = np.stack(data['reindexed_unit_cell_scaled'])
        sorted_lengths = np.sort(unit_cell[:, :3].ravel())
        lower = sorted_lengths[int(0.005*sorted_lengths.size)]
        upper = sorted_lengths[int(0.995*sorted_lengths.size)]
        bins = np.linspace(lower, upper, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        for index in range(3):
            hist, _ = np.histogram(unit_cell[:, index], bins=bins, density=True)
            axes[0, index + 2].bar(centers, hist, width=dbin)
            hist_scaled, _ = np.histogram(
                unit_cell_scaled[:, index], bins=bins_scaled, density=True
                )
            axes[1, index + 2].bar(centers_scaled, hist_scaled, width=dbin_scaled)
            if self.data_params['augment']:
                hist_augmented, _ = np.histogram(unit_cell_augmented[:, index], bins=bins, density=True)
                axes[0, index + 2].bar(centers, hist_augmented, width=dbin, alpha=0.5)
                hist_augmented_scaled, _ = np.histogram(unit_cell_augmented_scaled[:, index], bins=bins_scaled, density=True)
                axes[1, index + 2].bar(centers_scaled, hist_augmented_scaled, width=dbin_scaled, alpha=0.5)
            axes[0, index + 2].set_title(y_labels[index])

        if self.data_params['lattice_system'] == 'monoclinic':
            sorted_angles = np.sort(unit_cell[:, 4])
        else:
            sorted_angles = np.sort(unit_cell[:, 3:].ravel())
        lower = sorted_angles[int(0.005*sorted_angles.size)]
        upper = sorted_angles[int(0.995*sorted_angles.size)]
        bins = np.linspace(lower, upper, 101)
        centers = (bins[1:] + bins[:-1]) / 2
        dbin = bins[1] - bins[0]
        for index in range(3, 6):
            indices = np.logical_and(
                unit_cell[:, index] != np.pi/2,
                ~np.isnan(unit_cell[:, index])
                )
            hist, _ = np.histogram(unit_cell[indices, index], bins=bins, density=True)
            axes[0, index + 2].bar(centers, hist, width=dbin)

            indices = np.logical_and(
                unit_cell_scaled[:, index] != 0,
                ~np.isnan(unit_cell_scaled[:, index])
                )
            hist_scaled, _ = np.histogram(unit_cell_scaled[indices, index], bins=bins_scaled, density=True)
            axes[1, index + 2].bar(centers_scaled, hist_scaled, width=dbin_scaled)
            axes[0, index + 2].set_title(y_labels[index])

            if self.data_params['augment']:
                indices = np.logical_and(
                    unit_cell_augmented[:, index] != np.pi/2,
                    ~np.isnan(unit_cell_augmented[:, index])
                    )
                hist_augmented, _ = np.histogram(unit_cell_augmented[indices, index], bins=bins, density=True)
                axes[0, index + 2].bar(centers, hist_augmented, width=dbin, alpha=0.5)

                indices = np.logical_and(
                    unit_cell_augmented_scaled[:, index] != 0,
                    ~np.isnan(unit_cell_augmented_scaled[:, index])
                    )
                hist_augmented_scaled, _ = np.histogram(unit_cell_augmented_scaled[indices, index], bins=bins_scaled, density=True)
                axes[1, index + 2].bar(centers_scaled, hist_augmented_scaled, width=dbin_scaled, alpha=0.5)

        axes[0, 0].set_ylabel('Raw data')
        axes[1, 0].set_ylabel('Standard Scaling')
        axes[0, 0].set_title('q2')
        axes[0, 1].set_title(f'Volume\n(x{plot_volume_scale})')
        fig.tight_layout()
        fig.savefig(f'{self.save_to["data"]}/regression_inputs.png')
        plt.close()

        # Covariance
        if self.data_params['lattice_system'] != 'cubic':
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
        if self.data_params['lattice_system'] in ['orthorhombic', 'monoclinic', 'triclinic']:
            unit_cell = np.stack(self.data[~self.data['augmented']]['reindexed_unit_cell'])
            order = np.argsort(unit_cell[:, :3], axis=1)
            # order: [[shortest index, middle index, longest index], ... ]
            proportions = np.zeros((3, 3))
            if self.data_params['augment']:
                unit_cell_aug = np.stack(self.data[self.data['augmented']]['reindexed_unit_cell'])
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
        self.unit_cell_generator = dict.fromkeys(self.data_params['split_groups'])
        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            self.unit_cell_generator[split_group] = Regression_AlphaBeta(
                split_group,
                self.data_params,
                self.reg_params[split_group],
                self.save_to['regression'],
                self.random_seed,
                )
            self.unit_cell_generator[split_group].setup()
            if self.reg_params[split_group]['load_from_tag']:
                self.unit_cell_generator[split_group].load_from_tag()
            else:
                split_group_indices = self.data['split_group'] == split_group
                self.unit_cell_generator[split_group].train_regression(data=self.data[split_group_indices])

    def inferences_regression(self):
        reindexed_uc_pred_scaled = np.zeros((len(self.data), self.data_params['n_outputs']))
        reindexed_uc_pred_scaled_cov = np.zeros((
            len(self.data), self.data_params['n_outputs'], self.data_params['n_outputs']
            ))
        reindexed_uc_pred_scaled_trees = np.zeros((len(self.data), self.data_params['n_outputs']))
        reindexed_uc_pred_scaled_cov_trees = np.zeros((
            len(self.data), self.data_params['n_outputs'], self.data_params['n_outputs']
            ))

        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            split_group_indices = self.data['split_group'] == split_group
            reindexed_uc_pred_scaled[split_group_indices, :], reindexed_uc_pred_scaled_cov[split_group_indices, :, :] = \
                self.unit_cell_generator[split_group].do_predictions(data=self.data[split_group_indices], batch_size=1024)
            reindexed_uc_pred_scaled_trees[split_group_indices, :], reindexed_uc_pred_scaled_cov_trees[split_group_indices, :, :], _ = \
                self.unit_cell_generator[split_group].do_predictions_trees(data=self.data[split_group_indices])

        reindexed_uc_pred, reindexed_uc_pred_cov = self.revert_predictions(
            reindexed_uc_pred_scaled, reindexed_uc_pred_scaled_cov
            )
        self.data['volume_pred'] = list(self.infer_unit_cell_volume_from_predictions(reindexed_uc_pred))
        self.data['reindexed_unit_cell_pred'] = list(reindexed_uc_pred)
        self.data['reindexed_unit_cell_pred_cov'] = list(reindexed_uc_pred_cov)
        self.data['reindexed_unit_cell_pred_scaled'] = list(reindexed_uc_pred_scaled)
        self.data['reindexed_unit_cell_pred_scaled_cov'] = list(reindexed_uc_pred_scaled_cov)

        reindexed_uc_pred_trees, reindexed_uc_pred_cov_trees = self.revert_predictions(
            reindexed_uc_pred_scaled_trees, reindexed_uc_pred_scaled_cov_trees
            )
        self.data['volume_pred_trees'] = list(self.infer_unit_cell_volume_from_predictions(reindexed_uc_pred_trees))
        self.data['reindexed_unit_cell_pred_trees'] = list(reindexed_uc_pred_trees)
        self.data['reindexed_unit_cell_pred_cov_trees'] = list(reindexed_uc_pred_cov_trees)
        self.data['reindexed_unit_cell_pred_scaled_trees'] = list(reindexed_uc_pred_scaled_trees)
        self.data['reindexed_unit_cell_pred_scaled_cov_trees'] = list(reindexed_uc_pred_scaled_cov_trees)

        if self.data_params['lattice_system'] == 'monoclinic':
            unit_cell = np.stack(self.data['unit_cell'])
            N = unit_cell.shape[0]
            unit_cell_pred = np.zeros((N, 4))
            unit_cell_pred_cov = np.zeros((N, 4, 4))
            unit_cell_pred_trees = np.zeros((N, 4))
            unit_cell_pred_cov_trees = np.zeros((N, 4, 4))

            for entry_index in range(unit_cell.shape[0]):
                permutation, _ = get_permutation(unit_cell[entry_index])
                unit_cell_pred[entry_index], unit_cell_pred_cov[entry_index] = unpermute_monoclinic_partial_unit_cell(
                    reindexed_uc_pred[entry_index],
                    reindexed_uc_pred_cov[entry_index],
                    permutation,
                    radians=True,
                    )
                unit_cell_pred_trees[entry_index], unit_cell_pred_cov_trees[entry_index] = unpermute_monoclinic_partial_unit_cell(
                    reindexed_uc_pred_trees[entry_index],
                    reindexed_uc_pred_cov_trees[entry_index],
                    permutation,
                    radians=True,
                    )

            self.data['unit_cell_pred'] = list(unit_cell_pred)
            self.data['unit_cell_pred_cov'] = list(unit_cell_pred_cov)
            self.data['unit_cell_pred_trees'] = list(unit_cell_pred_trees)
            self.data['unit_cell_pred_cov_trees'] = list(unit_cell_pred_cov_trees)

    def revert_predictions(self, uc_pred_scaled=None, uc_pred_scaled_cov=None):
        if not uc_pred_scaled is None:
            uc_pred = np.zeros(uc_pred_scaled.shape)
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred = uc_pred_scaled * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
            elif self.data_params['lattice_system'] == 'monoclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3] = self.angle_scale * uc_pred_scaled[:, 3] + np.pi/2
            elif self.data_params['lattice_system'] == 'triclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3:] = self.angle_scale * uc_pred_scaled[:, 3:] + np.pi/2
            elif self.data_params['lattice_system'] == 'rhombohedral':
                uc_pred[:, 0] = uc_pred_scaled[:, 0] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 1] = self.angle_scale * uc_pred_scaled[:, 1] + np.pi/2

        if not uc_pred_scaled_cov is None:
            uc_pred_cov = np.zeros(uc_pred_scaled_cov.shape)
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred_cov = uc_pred_scaled_cov * self.uc_scaler.scale_[0]**2
            elif self.data_params['lattice_system'] == 'monoclinic':
                uc_pred_cov[:, :3, :3] = uc_pred_scaled_cov[:, :3, :3] * self.uc_scaler.scale_[0]**2
                uc_pred_cov[:, 3, 3] = uc_pred_scaled_cov[:, 3, 3] * self.angle_scale**2
                uc_pred_cov[:, :3, 3] = uc_pred_scaled_cov[:, :3, 3] * self.uc_scaler.scale_[0] * self.angle_scale
                uc_pred_cov[:, 3, :3] = uc_pred_scaled_cov[:, 3, :3] * self.uc_scaler.scale_[0] * self.angle_scale
            elif self.data_params['lattice_system'] == 'triclinic':
                uc_pred_cov[:, :3, :3] = uc_pred_scaled_cov[:, :3, :3] * self.uc_scaler.scale_[0]**2
                uc_pred_cov[:, 3:, 3:] = uc_pred_scaled_cov[:, 3:, 3:] * self.angle_scale**2
                uc_pred_cov[:, :3, 3:] = uc_pred_scaled_cov[:, :3, 3:] * self.uc_scaler.scale_[0] * self.angle_scale
                uc_pred_cov[:, 3:, :3] = uc_pred_scaled_cov[:, 3:, :3] * self.uc_scaler.scale_[0] * self.angle_scale
            elif self.data_params['lattice_system'] == 'rhombohedral':
                uc_pred_cov[:, 0, 0] = uc_pred_scaled_cov[:, 0, 0] * self.uc_scaler.scale_[0]**2
                uc_pred_cov[:, 1, 1] = uc_pred_scaled_cov[:, 1, 1] * self.angle_scale**2
                uc_pred_cov[:, 0, 1] = uc_pred_scaled_cov[:, 0, 1] * self.uc_scaler.scale_[0] * self.angle_scale
                uc_pred_cov[:, 1, 0] = uc_pred_scaled_cov[:, 1, 0] * self.uc_scaler.scale_[0] * self.angle_scale

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
            uc_pred_scaled[:, 3] = (uc_pred[:, 3] - np.pi/2) / self.angle_scale
        elif self.data_params['lattice_system'] == 'triclinic':
            uc_pred_scaled = np.zeros(uc_pred.shape)
            uc_pred_scaled[:, :3] = (uc_pred[:, :3] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
            uc_pred_scaled[:, 3:] = (uc_pred[:, 3:] - np.pi/2) / self.angle_scale
        elif self.data_params['lattice_system'] == 'rhombohedral':
            uc_pred_scaled = np.zeros(uc_pred.shape)
            uc_pred_scaled[:, 0] = (uc_pred[:, 0] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
            uc_pred_scaled[:, 1] = (uc_pred[:, 1] - np.pi/2) / self.angle_scale
        return uc_pred_scaled

    def evaluate_regression(self):
        if self.data_params['lattice_system'] == 'monoclinic':
            unit_cell_key = 'unit_cell'
        else:
            unit_cell_key = 'reindexed_unit_cell'

        for bravais_lattice in self.data_params['bravais_lattices']:
            evaluate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg.png',
                y_indices=self.data_params['y_indices'],
                trees=False
                )
            evaluate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_tree.png',
                y_indices=self.data_params['y_indices'],
                trees=True
                )
            calibrate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_calibration.png',
                y_indices=self.data_params['y_indices'],
                trees=False
                )
            calibrate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_calibration_tree.png',
                y_indices=self.data_params['y_indices'],
                trees=True
                )
        for split_group in self.data_params['split_groups']:
            evaluate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg.png',
                y_indices=self.data_params['y_indices'],
                trees=False
                )
            evaluate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_tree.png',
                y_indices=self.data_params['y_indices'],
                trees=True
                )
            calibrate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_calibration.png',
                y_indices=self.data_params['y_indices'],
                trees=False
                )
            calibrate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key=unit_cell_key,
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_calibration_tree.png',
                y_indices=self.data_params['y_indices'],
                trees=True
                )
        evaluate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key=unit_cell_key,
            save_to_name=f'{self.save_to["regression"]}/All_reg.png',
            y_indices=self.data_params['y_indices'],
            trees=False
            )
        calibrate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key=unit_cell_key,
            save_to_name=f'{self.save_to["regression"]}/All_reg_calibration.png',
            y_indices=self.data_params['y_indices'],
            trees=False
            )
        evaluate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key=unit_cell_key,
            save_to_name=f'{self.save_to["regression"]}/All_reg_trees.png',
            y_indices=self.data_params['y_indices'],
            trees=True
            )
        calibrate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key=unit_cell_key,
            save_to_name=f'{self.save_to["regression"]}/All_reg_calibration_trees.png',
            y_indices=self.data_params['y_indices'],
            trees=True
            )

    def setup_assignment(self):
        self.assigner = dict.fromkeys(self.assign_params.keys())
        if self.data_params['lattice_system'] == 'monoclinic':
            unit_cell_key = 'unit_cell'
        else:
            unit_cell_key = 'reindexed_unit_cell'
        for key in self.assign_params.keys():
            self.assigner[key] = Assigner(
                self.data_params,
                self.assign_params[key],
                self.hkl_ref,
                self.uc_scaler,
                self.angle_scale,
                self.q2_scaler,
                self.save_to['assigner']
                )
            if self.assign_params[key]['load_from_tag']:
                self.assigner[key].load_from_tag(self.assign_params[key]['tag'], self.assign_params[key]['mode'])
            else:
                if self.assign_params[key]['train_on'] == 'perturbed':
                    unit_cell_scaled_key = f'{unit_cell_key}_scaled'
                    y_indices = self.data_params['y_indices']
                elif self.assign_params[key]['train_on'] == 'predicted':
                    unit_cell_scaled_key = f'{unit_cell_key}_pred_scaled'
                    y_indices = None
                self.assigner[key].fit_model(
                    data=self.data[~self.data['augmented']],
                    unit_cell_scaled_key=unit_cell_scaled_key,
                    y_indices=y_indices,
                    )

    def inferences_assignment(self, keys):
        if self.data_params['lattice_system'] == 'monoclinic':
            unit_cell_key = 'unit_cell'
        else:
            unit_cell_key = 'reindexed_unit_cell'
        for key in keys:
            if self.assign_params[key]['train_on'] == 'perturbed':
                unit_cell_scaled_key = f'{unit_cell_key}_scaled'
                y_indices = self.data_params['y_indices']
            elif self.assign_params[key]['train_on'] == 'predicted':
                unit_cell_scaled_key = f'{unit_cell_key}_pred_scaled'
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
        for entry_index in range(n_entries):
            hkl_pred[entry_index] = self.hkl_ref[hkl_assign[entry_index]]
        return hkl_pred
