"""
- Data
    * reindex all data to ordered unit cell axes

    - Get data from other databases:
        - COD
        - Materials project
        - ICSD

- Regression:
    * Make groups of spacegroups
    * Move predictions into the Regression class
    * bnn does not work during prediction

    - prediction of PCA components
        - evaluation of fitting in the PCA / Scaled space
        - evaluation of covariance
    - read Stirn 2023 and implement
    Detlefsen 2019:
        - cluster input d-spacings
          - map d-spacings onto a single scalar correlated with volume
        - ??? extrapolation architecture

- Optimization:
    * What differentiates a found / not found entry
        - large differences between prediction and true

    - assignment with group specific assigners
    - SVD
    - common assignments:
        - drop during optimization but include in loss
        - use all hkl assignments with largest N likelihoods

- Assignments
    - assigners specific to the unit cell generator
"""
import copy
import csv
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import scipy.special
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from Assigner import Assigner
from Augmentor import Augmentor
from BLClassifier import Classifier
from Regression import Regression_AlphaBeta
from Regression import Regression_MVE
from Utilities import IdentityTransformer
from Utilities import Q2Calculator
from TargetFunctions import CandidateOptLoss
from TargetFunctions import IndexingTargetFunction


class Indexing:
    def __init__(
            self,
            assign_params=None,
            aug_params=None,
            class_params=None, 
            data_params=None, 
            reg_params=None, 
            seed=12345, 
            ):
        self.random_seed = seed
        self.rng = np.random.default_rng(self.random_seed)
        self.n_generated_points = 60
        self.save_to = dict.fromkeys(['results', 'data', 'classifier', 'regression', 'assigner', 'augmentor'])
        self.assign_params = assign_params
        self.aug_params = aug_params
        self.class_params = class_params
        self.data_params = data_params
        self.reg_params = reg_params

        self.save_to = {
            'results': f'models/{self.data_params["tag"]}',
            'assigner': f'models/{self.data_params["tag"]}/assigner',
            'augmentor': f'models/{self.data_params["tag"]}/augmentor',
            'classifier': f'models/{self.data_params["tag"]}/classifier',
            'data': f'models/{self.data_params["tag"]}/data',
            'regression': f'models/{self.data_params["tag"]}/regression',
            }
        if not os.path.exists(self.save_to['results']):
            os.mkdir(self.save_to['results'])
            os.mkdir(self.save_to['assigner'])
            os.mkdir(self.save_to['augmentor'])
            os.mkdir(self.save_to['classifier'])
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
            'data_dir': '/Users/DWMoreau/unit_cell_ML/data',
            'augment': False,
            'train_fraction': 0.80,
            'n_max': 25000,
            'n_points': 20,
            'points_tag': 'intersect',
            'include_centered': True,
            'use_reduced_cell': False,
            'hkl_ref_length': 300,
            }
        for key in data_params_defaults.keys():
            if not data_params_defaults[key] is None:
                if key not in self.data_params.keys():
                    self.data_params[key] = data_params_defaults[key]
        if self.data_params['lattice_system'] == 'cubic':
            self.data_params['y_indices'] = [0]
            if self.data_params['include_centered']:
                self.data_params['bravais_lattices'] = ['cF', 'cI', 'cP']
            else:
                self.data_params['bravais_lattices'] = ['cP']
        elif self.data_params['lattice_system'] == 'tetragonal':
            self.data_params['y_indices'] = [0, 2]
            if self.data_params['include_centered']:
                self.data_params['bravais_lattices'] = ['tI', 'tP']
            else:
                self.data_params['bravais_lattices'] = ['tP']
        elif self.data_params['lattice_system'] == 'orthorhombic':
            self.data_params['y_indices'] = [0, 1, 2]
            if self.data_params['include_centered']:
                self.data_params['bravais_lattices'] = ['oC', 'oF', 'oI', 'oP']
            else:
                self.data_params['bravais_lattices'] = ['oP']
        elif self.data_params['lattice_system'] == 'monoclinic':
            self.data_params['y_indices'] = [0, 1, 2, 4]
            if self.data_params['include_centered']:
                self.data_params['bravais_lattices'] = ['mC', 'mP']
            else:
                self.data_params['bravais_lattices'] = ['mP']
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
            'include_centered',
            'use_reduced_cell',
            'n_outputs',
            'hkl_ref_length'
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
        if params['include_centered'] == 'True':
            self.data_params['include_centered'] = True
        elif params['include_centered'] == 'False':
            self.data_params['include_centered'] = False
        self.data_params['hkl_ref_length'] = int(params['hkl_ref_length'])
        self.data_params['n_outputs'] = int(params['n_outputs'])
        self._setup_joint()

    def _setup_joint(self):
        if self.data_params['unit_cell_representation'] == 'conventional':
            self.unit_cell_key = 'unit_cell'
            self.volume_key = 'volume'
            self.hkl_key = 'hkl'
        elif self.data_params['unit_cell_representation'] == 'reduced':
            self.unit_cell_key = 'reduced_unit_cell'
            self.volume_key = 'reduced_volume'
            self.hkl_key = 'hkl'
        elif self.data_params['unit_cell_representation'] == 'reordered':
            self.unit_cell_key = 'reordered_unit_cell'
            self.volume_key = 'volume'
            self.hkl_key = 'reordered_hkl'
        else:
            print('Need to supply a unit_cell_representation')
            assert False

        for key in self.assign_params.keys():
            self.assign_params[key]['n_outputs'] = self.data_params['hkl_ref_length']
        self.class_params['n_outputs'] = len(self.data_params['bravais_lattices'])
        self.data_params['n_outputs'] = len(self.data_params['y_indices'])
        self.reg_params['n_outputs'] = self.data_params['n_outputs']

        all_labels = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        self.uc_labels = [all_labels[index] for index in self.data_params['y_indices']]

    def load_data(self):
        read_columns = [
            'spacegroup_number',
            'setting',
            'crystal_family',
            'bravais_lattice',
            'centering',
            'd_spacing_' + self.data_params['points_tag'],
            'h_' + self.data_params['points_tag'],
            'k_' + self.data_params['points_tag'],
            'l_' + self.data_params['points_tag'],
            'unit_cell', 'volume',
            'reduced_unit_cell', 'reduced_volume',
            ]
        if self.data_params['augment']:
            read_columns += [
                'd_spacing_all', 'h_all', 'k_all', 'l_all',
                'd_spacing_sa', 'h_sa', 'k_sa', 'l_sa',
                ]
        data = []
        for index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            file_name = os.path.join(
                self.data_params['data_dir'],
                f'dataset_{bravais_lattice}.parquet'
                )
            print(file_name)
            bl_data = pd.read_parquet(file_name, columns=read_columns)
            n_entries = min(bl_data.shape[0], self.data_params['n_max'])
            # This shuffles the data and downsamples if needed
            bl_data = bl_data.sample(
                n=n_entries,
                replace=False,
                random_state=self.random_seed
                )
            bl_data['bravais_lattice_label'] = index * np.ones(bl_data.shape[0])
            data.append(bl_data)
            
        self.data = pd.concat(data, ignore_index=True)
        # Remove data that doesn't have enough peaks
        # A total of 60 or so peaks are included in the data set - for all extries
        # If there were less than 60 peaks, those get padded with zeros at the end of the array.
        #   - the 60 number is arbitrary and set in GenerateDataset_mpi.py
        points = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        indices = points.apply(len) >= self.data_params['n_points']
        self.data = self.data.loc[indices]
        points = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        enough_peaks = points.apply(np.count_nonzero) >= self.data_params['n_points']
        self.data = self.data.loc[enough_peaks]
        self.data['augmented'] = np.zeros(self.data.shape[0], dtype=bool)

        d_spacing_pd = self.data[f'd_spacing_{self.data_params["points_tag"]}']
        d_spacing = [np.zeros(self.data_params['n_points']) for i in range(self.data.shape[0])]
        for entry_index in range(self.data.shape[0]):
            entry = d_spacing_pd.iloc[entry_index]
            d_spacing[entry_index] = entry[:self.data_params['n_points']]
        self.data['d_spacing'] = d_spacing
        # The math is a bit easier logistically when we use q**2
        self.data['q2_sa'] = 1 / self.data['d_spacing_sa']**2
        self.data[f'q2_{self.data_params["points_tag"]}'] = 1 / self.data[f'd_spacing_{self.data_params["points_tag"]}']**2
        self.data['q2'] = 1 / self.data['d_spacing']**2

        # This sets up the training / validation tags so that the validation set is taken
        # evenly from spacegroups
        spacegroup_assign_label = 0
        train_label_spacegroup = np.ones(self.data.shape[0], dtype=bool)
        spacegroups = self.data['spacegroup_number'].unique()
        spacegroup_label = np.zeros(self.data.shape[0], dtype=int)
        self.data_params['spacegroups'] = []
        for spacegroup in spacegroups:
            indices = np.where(self.data['spacegroup_number'] == spacegroup)[0]
            n_val = int(indices.size * (1 - self.data_params['train_fraction']))
            val_indices = self.rng.choice(indices, size=n_val, replace=False)
            train_label_spacegroup[val_indices] = False
            spacegroup_label[indices] = spacegroup_assign_label
            spacegroup_assign_label += 1
            self.data_params['spacegroups'].append(spacegroup)
        self.data['spacegroup_label'] = spacegroup_assign_label
        self.data['train'] = train_label_spacegroup

        # put the hkl's together
        hkl = np.zeros((len(self.data), self.data_params['n_points'], 3), dtype=int)
        hkl_sa = np.zeros((len(self.data), self.n_generated_points, 3), dtype=int)
        for entry_index in range(len(self.data)):
            entry = self.data.iloc[entry_index]
            hkl[entry_index, :, 0] = entry[f'h_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            hkl[entry_index, :, 1] = entry[f'k_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            hkl[entry_index, :, 2] = entry[f'l_{self.data_params["points_tag"]}'][:self.data_params['n_points']]
            hkl_sa[entry_index, :entry[f'h_sa'].size, 0] = entry[f'h_sa']
            hkl_sa[entry_index, :entry[f'k_sa'].size, 1] = entry[f'k_sa']
            hkl_sa[entry_index, :entry[f'l_sa'].size, 2] = entry[f'l_sa']
        self.data['hkl'] = list(hkl)
        self.data['hkl_sa'] = list(hkl_sa)
        self.data.drop(
            columns=[
                'h_sa', 'k_sa', 'l_sa',
                f'h_{self.data_params["points_tag"]}',
                f'k_{self.data_params["points_tag"]}',
                f'l_{self.data_params["points_tag"]}'
                ],
            inplace=True
            )

        if self.data_params['unit_cell_representation'] == 'reordered':
            self.reorder_lattice()
        self.setup_scalers()
        # Convert angles to radians
        unit_cell = np.stack(self.data[self.unit_cell_key])
        unit_cell[:, 3:] = np.pi/180 * unit_cell[:, 3:]
        self.data[self.unit_cell_key] = list(unit_cell)

        if self.data_params['augment']:
            self.augment_data()
        self.setup_hkl()

        # This does another shuffle.
        self.data = self.data.sample(frac=1, replace=False, random_state=self.random_seed)
        self.N = len(self.data)
        self.N_train = np.sum(self.data['train'])
        self.N_val = self.N - self.N_train
        self.N_bl = np.zeros((len(self.data_params['bravais_lattices']), 3), dtype=int)
        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            self.N_bl[bl_index, 0] = np.sum(
                self.data['bravais_lattice'] == bravais_lattice
                )
            self.N_bl[bl_index, 1] = np.sum(
                self.data[self.data['train']]['bravais_lattice'] == bravais_lattice
                )
            self.N_bl[bl_index, 2] = np.sum(
                self.data[~self.data['train']]['bravais_lattice'] == bravais_lattice
                )

        self.plot_input()
        self.save()

    def load_data_from_tag(self, load_augmented, load_train):
        self.hkl_ref = np.load(f'{self.save_to["data"]}/hkl_ref.npy')
        self.data = pd.read_parquet(f'{self.save_to["data"]}/data.parquet')
        hkl = np.stack([
            np.stack(self.data['h'], axis=0)[:, :, np.newaxis],
            np.stack(self.data['k'], axis=0)[:, :, np.newaxis],
            np.stack(self.data['l'], axis=0)[:, :, np.newaxis]
            ], axis=2
            )
        self.data['hkl'] = list(hkl)
        self.data.drop(columns=['h', 'k', 'l'], inplace=True)
        if 'reordered_h' in self.data.keys():
            reordered_hkl = np.stack([
                np.stack(self.data['reordered_h'], axis=0)[:, :, np.newaxis],
                np.stack(self.data['reordered_k'], axis=0)[:, :, np.newaxis],
                np.stack(self.data['reordered_l'], axis=0)[:, :, np.newaxis]
                ], axis=2
                )
            self.data['reordered_hkl'] = list(reordered_hkl)
            self.data.drop(columns=['reordered_h', 'reordered_k', 'reordered_l'], inplace=True)
        self.N_bl = np.load(f'{self.save_to["data"]}/N_bl.npy')
        if load_augmented == False:
            self.data = self.data[~self.data['augmented']]
        if load_train == False:
            self.data = self.data[~self.data['train']]
        self.N = len(self.data)
        self.N_train = np.sum(self.data['train'])
        self.N_val = self.N - self.N_train
        print(self.N)

    def save(self):
        hkl = np.stack(self.data['hkl'])
        self.data['h'] = list(hkl[:, :, 0])
        self.data['k'] = list(hkl[:, :, 1])
        self.data['l'] = list(hkl[:, :, 2])
        hkl_sa = np.stack(self.data['hkl_sa'])
        self.data['h_sa'] = list(hkl_sa[:, :, 0])
        self.data['k_sa'] = list(hkl_sa[:, :, 1])
        self.data['l_sa'] = list(hkl_sa[:, :, 2])
        self.data.drop(columns=['hkl', 'hkl_sa'], inplace=True)
        if 'reordered_hkl' in self.data.keys():
            reordered_hkl = np.stack(self.data['reordered_hkl'])
            self.data['reordered_h'] = list(reordered_hkl[:, :, 0])
            self.data['reordered_k'] = list(reordered_hkl[:, :, 1])
            self.data['reordered_l'] = list(reordered_hkl[:, :, 2])
            reordered_hkl_sa = np.stack(self.data['reordered_hkl_sa'])
            self.data['reordered_h_sa'] = list(reordered_hkl_sa[:, :, 0])
            self.data['reordered_k_sa'] = list(reordered_hkl_sa[:, :, 1])
            self.data['reordered_l_sa'] = list(reordered_hkl_sa[:, :, 2])
            self.data.drop(columns=['reordered_hkl', 'reordered_hkl_sa'], inplace=True)
        self.data.to_parquet(f'{self.save_to["data"]}/data.parquet')
        np.save(f'{self.save_to["data"]}/hkl_ref.npy', self.hkl_ref)
        joblib.dump(self.uc_scaler, f'{self.save_to["data"]}/uc_scaler.bin')
        joblib.dump(self.volume_scaler, f'{self.save_to["data"]}/volume_scaler.bin')
        joblib.dump(self.q2_scaler, f'{self.save_to["data"]}/q2_scaler.bin')

        with open(f'{self.save_to["data"]}/data_params.csv', 'w') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.data_params.keys())
            writer.writeheader()
            writer.writerow(self.data_params)
        np.save(f'{self.save_to["data"]}/N_bl.npy', self.N_bl)

    def reorder_lattice(self):
        if self.data_params['lattice_system'] == 'cubic':
            self.reorder_lattice_cubic()
        elif self.data_params['lattice_system'] == 'tetragonal':
            self.reorder_lattice_tetragonal()
        elif self.data_params['lattice_system'] == 'orthorhombic':
            self.reorder_lattice_orthorhombic()
        elif self.data_params['lattice_system'] == 'hexagonal':
            self.reorder_lattice_hexagonal()
        elif self.data_params['lattice_system'] == 'rhombohedral':
            self.reorder_lattice_rhombohedral()
        elif self.data_params['lattice_system'] == 'monoclinic':
            self.reorder_lattice_monoclinic()
        elif self.data_params['lattice_system'] == 'triclinic':
            self.reorder_lattice_triclinic()

    def reorder_lattice_orthorhombic(self):
        unit_cell = np.stack(self.data['unit_cell'])[:, :3]
        hkl = np.stack(self.data['hkl'])
        hkl_sa = np.stack(self.data['hkl_sa'])

        reordered_unit_cell = np.zeros((unit_cell.shape[0], 6))
        reordered_unit_cell[:, 3:] = np.pi/2
        reordered_hkl = np.zeros(hkl.shape)
        reordered_hkl_sa = np.zeros(hkl_sa.shape)
        order = np.argsort(unit_cell, axis=1)
        for index in range(unit_cell.shape[0]):
            if np.all(order[index] == [0, 1, 2]):
                permutation = np.eye(3)
            elif np.all(order[index] == [0, 2, 1]):
                permutation = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    ])
            elif np.all(order[index] == [1, 0, 2]):
                permutation = np.array([
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    ])
            elif np.all(order[index] == [2, 0, 1]):
                permutation = np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                    ])
            elif np.all(order[index] == [1, 2, 0]):
                permutation = np.array([
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    ])
            elif np.all(order[index] == [2, 1, 0]):
                permutation = np.array([
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    ])
            reordered_unit_cell[index, :3] = np.matmul(permutation, unit_cell[index])
            reordered_hkl[index] = np.matmul(permutation, hkl[index][:, :, np.newaxis])[:, :, 0]
            reordered_hkl_sa[index] = np.matmul(permutation, hkl_sa[index][:, :, np.newaxis])[:, :, 0]
        self.data['reordered_unit_cell'] = list(reordered_unit_cell)
        self.data['reordered_hkl'] = list(reordered_hkl)
        self.data['reordered_hkl_sa'] = list(reordered_hkl_sa)

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
                hkl[:, :, 0]**2 +  hkl[:, :, 0]* hkl[:, :, 1] + hkl[:, :, 1]**2,
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
        self.augmentor = dict.fromkeys(self.data_params['bravais_lattices'])
        bl_data_augmented = [None for i in range(len(self.data_params['bravais_lattices']))]
        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            bl_data = self.data[self.data['bravais_lattice'] == bravais_lattice]
            self.augmentor[bravais_lattice] = Augmentor(
                aug_params=self.aug_params,
                bravais_lattice=bravais_lattice,
                data_params=self.data_params,
                min_unit_cell_scaled=self.min_unit_cell_scaled, 
                n_generated_points=self.n_generated_points,
                save_to=self.save_to['augmentor'],
                seed=self.random_seed,
                uc_scaler=self.uc_scaler,
                unit_cell_key=self.unit_cell_key,
                hkl_key=self.hkl_key,
                )
            self.augmentor[bravais_lattice].setup(bl_data)
            bl_data_augmented[bl_index] = self.augmentor[bravais_lattice].augment(bl_data)
        bl_data_augmented = pd.concat(bl_data_augmented, ignore_index=True)
        self.data = pd.concat((self.data, bl_data_augmented), ignore_index=True)

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
            unit_cell_aug = np.stack(self.data[self.data['augmented']][self.unit_cell_key])
            order = np.argsort(unit_cell[:, :3], axis=1)
            order_aug = np.argsort(unit_cell_aug[:, :3], axis=1)
            # order: [[shortest index, middle index, longest index], ... ]
            proportions = np.zeros((3, 3))
            proportions_aug = np.zeros((3, 3))
            for length_index in range(3):
                for uc_index in range(3):
                    proportions[length_index, uc_index] = np.sum(order[:, length_index] == uc_index)
                    proportions_aug[length_index, uc_index] = np.sum(order_aug[:, length_index] == uc_index)
            fig, axes = plt.subplots(1, 3, figsize=(8, 4))
            for length_index in range(3):
                axes[length_index].plot(proportions[length_index], marker='.', markersize=20, label='Unaugmented')
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

    def setup_networks(self):
        #self.setup_classifier()
        self.setup_regression()
        self.setup_assignment()

    def setup_classifier(self):
        self.classifier = Classifier(self.data_params, self.class_params, self.save_to['classifier'])
        if self.class_params['load_from_tag']:
            self.classifier.load_from_tag(self.class_params['tag'])
        else:
            self.classifier.do_classification(self.data)
            self.classifier.evaluate(self.data)

    def setup_regression(self):
        self.unit_cell_generator = dict.fromkeys(self.data_params['bravais_lattices'])
        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            self.reg_params[bravais_lattice]['N_train'] = self.N_bl[bl_index, 1]
            if self.reg_params[bravais_lattice]['var_est'] == 'alpha_beta':
                self.unit_cell_generator[bravais_lattice] = Regression_AlphaBeta(
                    bravais_lattice, 
                    self.data_params, 
                    self.reg_params[bravais_lattice], 
                    self.save_to['regression'], 
                    self.unit_cell_key,
                    self.random_seed,
                    )
            elif self.reg_params[bravais_lattice]['var_est'] == 'mve':
                self.unit_cell_generator[bravais_lattice] = Regression_MVE(
                    bravais_lattice, 
                    self.data_params, 
                    self.reg_params[bravais_lattice], 
                    self.save_to['regression'], 
                    self.unit_cell_key,
                    self.random_seed,
                    )
            self.unit_cell_generator[bravais_lattice].setup()
            if self.reg_params[bravais_lattice]['load_from_tag']:
                self.unit_cell_generator[bravais_lattice].load_from_tag()
            else:
                bl_indices = self.data['bravais_lattice'] == bravais_lattice
                self.unit_cell_generator[bravais_lattice].train_regression(data=self.data[bl_indices])
        self.inferences_regression()

    def inferences_regression(self):
        uc_pred_scaled = np.zeros((len(self.data), self.data_params['n_outputs']))
        uc_pred_scaled_cov = np.zeros((len(self.data), self.data_params['n_outputs'], self.data_params['n_outputs']))

        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            bl_indices = self.data['bravais_lattice'] == bravais_lattice
            uc_pred_scaled[bl_indices, :], uc_pred_scaled_cov[bl_indices, :, :] = \
                self.unit_cell_generator[bravais_lattice].do_predictions(data=self.data[bl_indices])

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
        do_all = False
        for bravais_lattice in self.data_params['bravais_lattices']:
            self.unit_cell_generator[bravais_lattice].evaluate(self.data)
            self.unit_cell_generator[bravais_lattice].calibrate(self.data)
            do_all = True
        if do_all:
            Regression_AlphaBeta.evaluate_regression(
                data=self.data,
                bravais_lattice='All', 
                n_outputs=self.data_params['n_outputs'], 
                unit_cell_key=self.unit_cell_key, 
                save_to_name=f'{self.save_to["regression"]}/All_reg.png', 
                y_indices=self.data_params['y_indices']
                )
            Regression_AlphaBeta.calibrate_regression(
                data=self.data,
                bravais_lattice='All', 
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
                    data=self.data, 
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

            logits = self.assigner[key].do_predictions(
                self.data, 
                unit_cell_scaled_key=unit_cell_scaled_key,
                y_indices=y_indices
                )
            softmaxes = scipy.special.softmax(logits, axis=2)
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
            self.data['hkl_labels_pred'] = list(hkl_assign)
            self.data['hkl_pred'] = list(hkl_pred)
            self.data['hkl_softmaxes'] = list(softmaxes)
            self.data['hkl_logits'] = list(logits)
            self.assigner[key].evaluate(
                self.data[~self.data['augmented']], 
                self.data_params['bravais_lattices'] + ['All'],
                unit_cell_scaled_key=unit_cell_scaled_key,
                y_indices=y_indices
                )
            self.assigner[key].calibrate(self.data[~self.data['augmented']])

    def convert_softmax_to_assignments(self, softmaxes):
        n_entries = softmaxes.shape[0]
        hkl_assign = softmaxes.argmax(axis=2)
        hkl_pred = np.zeros((n_entries, self.data_params['n_points'], 3))
        for entry_index in range(n_entries ):
            hkl_pred[entry_index] = self.hkl_ref[hkl_assign[entry_index]]
        return hkl_pred

    def evaluate_tetragonal_large_errors(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plot_volume_scale = 1000

        data = self.data[~self.data['augmented']]
        unit_cell_true = np.stack(data[self.unit_cell_key])[:, self.data_params['y_indices']]
        unit_cell_pred = np.stack(data[f'{self.unit_cell_key}_pred'])
        unit_cell_mse = 1/self.data_params['n_outputs'] * np.linalg.norm(unit_cell_pred - unit_cell_true, axis=1)**2
        large_errors = unit_cell_mse > np.sort(unit_cell_mse)[int(0.75 * data.shape[0])]
        N_small = np.sum(~large_errors)
        N_large = np.sum(large_errors)

        fig, axes = plt.subplots(2, 4, figsize=(10, 6))
        # volume
        volume = np.array(data[self.volume_key]) / plot_volume_scale
        axes[0, 0].boxplot([volume[~large_errors], volume[large_errors]])
        axes[0, 0].set_title(f'Volume (x{plot_volume_scale})')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xticks([1, 2])
        axes[0, 0].set_xticklabels(['Small err', 'Large err'])

        # minimum unit cell
        uc_min = unit_cell_true.min(axis=1)
        axes[0, 1].boxplot([uc_min[~large_errors], uc_min[large_errors]])
        axes[0, 1].set_title('Minimum unit cell')
        axes[0, 1].set_xticks([1, 2])
        axes[0, 1].set_xticklabels(['Small err', 'Large err'])

        # maximum unit cell
        uc_max = unit_cell_true.max(axis=1)
        axes[0, 2].boxplot([uc_max[~large_errors], uc_max[large_errors]])
        axes[0, 2].set_title('Maximum unit cell')
        axes[0, 2].set_xticks([1, 2])
        axes[0, 2].set_xticklabels(['Small err', 'Large err'])

        # dominant zone ratio
        ratio =  uc_max / uc_min 
        axes[0, 3].boxplot([ratio[~large_errors], ratio[large_errors]])
        axes[0, 3].set_title('Maximum / Minimum\nunit cell ratio')
        axes[0, 3].set_xticks([1, 2])
        axes[0, 3].set_xticklabels(['Small err', 'Large err'])

        # variation in unit cell sizes
        #   (a - b)**2 + (a - c)**2 + (b - c)**2
        variation = \
            ((unit_cell_true[:, 0] - unit_cell_true[:, 1]) / (0.5*(unit_cell_true[:, 0] + unit_cell_true[:, 1])))**2
        axes[1, 0].boxplot([variation[~large_errors], variation[large_errors]])
        axes[1, 0].set_title('Variation in unit cell')
        axes[1, 0].set_xticks([1, 2])
        axes[1, 0].set_xticklabels(['Small err', 'Large err'])

        # order of unit cell axis sizes
        order_small = np.argsort(unit_cell_true[~large_errors], axis=1)
        order_large = np.argsort(unit_cell_true[large_errors], axis=1)
        # order: [[shortest index, middle index, longest index], ... ]
        proportions_small = np.zeros((2, 2))
        proportions_large = np.zeros((2, 2))
        for length_index in range(2):
            for uc_index in range(2):
                proportions_small[length_index, uc_index] = np.sum(order_small[:, length_index] == uc_index)
                proportions_large[length_index, uc_index] = np.sum(order_large[:, length_index] == uc_index)
        proportions_small = proportions_small / N_small
        proportions_large = proportions_large / N_large
        axes[1, 1].bar([0, 1], proportions_small[0], color=colors[0], label='Small err')
        axes[1, 1].bar([3, 4], proportions_small[1], color=colors[0])

        axes[1, 1].bar([0, 1], proportions_large[0], color=colors[1], alpha=0.5, label='Large err')
        axes[1, 1].bar([3, 4], proportions_large[1], color=colors[1], alpha=0.5)

        axes[1, 1].set_title('Order of axes lengths')
        axes[1, 1].set_xticks([0.5, 2.5])
        axes[1, 1].set_xticklabels(['Shortest', 'Longest'])
        axes[1, 1].legend(frameon=False)

        # centering
        primitive = np.array(data['bravais_lattice'] == 'tP')
        body_centered = np.array(data['bravais_lattice'] == 'tI')

        # fraction centered
        centered_frac_small = np.sum(~primitive[~large_errors]) / N_small
        centered_frac_large = np.sum(~primitive[large_errors]) / N_large
        axes[1, 2].bar([0, 2], [centered_frac_small, centered_frac_large])
        axes[1, 2].set_xticks([0, 2])
        axes[1, 2].set_xticklabels(['Small error', 'Large error'])
        axes[1, 2].set_title('Fraction centered')

        # fraction Bravais lattice
        tI_frac_small = np.sum(body_centered[~large_errors]) / N_small
        tI_frac_large = np.sum(body_centered[large_errors]) / N_large

        axes[1, 3].bar(0, tI_frac_small, color=colors[0], label='Small err')
        axes[1, 3].bar(0, tI_frac_large, color=colors[1], alpha=0.5, label='Large err')
        axes[1, 3].set_xticks([0])
        axes[1, 3].set_xticklabels(['tI'])
        axes[1, 3].set_title('Bravais Lattice')

        fig.tight_layout()
        fig.savefig(self.save_to['results'] + '_tetragonal_error_eval.png')
        plt.close()

    def evaluate_orthorhombic_large_errors(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plot_volume_scale = 1000
        data = self.data[~self.data['augmented']]
        unit_cell_true = np.stack(data[self.unit_cell_key])[:, self.data_params['y_indices']]
        unit_cell_pred = np.stack(data[f'{self.unit_cell_key}_pred'])
        unit_cell_mse = 1/self.data_params['n_outputs'] * np.linalg.norm(unit_cell_pred - unit_cell_true, axis=1)**2
        large_errors = unit_cell_mse > np.sort(unit_cell_mse)[int(0.75 * data.shape[0])]
        N_small = np.sum(~large_errors)
        N_large = np.sum(large_errors)

        fig, axes = plt.subplots(2, 4, figsize=(10, 6))
        # volume
        volume = np.array(data[self.volume_key]) / plot_volume_scale
        axes[0, 0].boxplot([volume[~large_errors], volume[large_errors]])
        axes[0, 0].set_title(f'Volume (x{plot_volume_scale})')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xticks([1, 2])
        axes[0, 0].set_xticklabels(['Small err', 'Large err'])

        # minimum unit cell
        uc_min = unit_cell_true.min(axis=1)
        axes[0, 1].boxplot([uc_min[~large_errors], uc_min[large_errors]])
        axes[0, 1].set_title('Minimum unit cell')
        axes[0, 1].set_xticks([1, 2])
        axes[0, 1].set_xticklabels(['Small err', 'Large err'])

        # maximum unit cell
        uc_max = unit_cell_true.max(axis=1)
        axes[0, 2].boxplot([uc_max[~large_errors], uc_max[large_errors]])
        axes[0, 2].set_title('Maximum unit cell')
        axes[0, 2].set_xticks([1, 2])
        axes[0, 2].set_xticklabels(['Small err', 'Large err'])

        # dominant zone ratio
        ratio =  uc_max / uc_min 
        axes[0, 3].boxplot([ratio[~large_errors], ratio[large_errors]])
        axes[0, 3].set_title('Maximum / Minimum\nunit cell ratio')
        axes[0, 3].set_xticks([1, 2])
        axes[0, 3].set_xticklabels(['Small err', 'Large err'])

        # variation in unit cell sizes
        #   (a - b)**2 + (a - c)**2 + (b - c)**2
        variation = \
            ((unit_cell_true[:, 0] - unit_cell_true[:, 1]) / (0.5*(unit_cell_true[:, 0] + unit_cell_true[:, 1])))**2 \
            + ((unit_cell_true[:, 0] - unit_cell_true[:, 2]) / (0.5*(unit_cell_true[:, 0] + unit_cell_true[:, 2])))**2 \
            + ((unit_cell_true[:, 1] - unit_cell_true[:, 2]) / (0.5*(unit_cell_true[:, 1] + unit_cell_true[:, 2])))**2
        axes[1, 0].boxplot([variation[~large_errors], variation[large_errors]])
        axes[1, 0].set_title('Variation in unit cell')
        axes[1, 0].set_xticks([1, 2])
        axes[1, 0].set_xticklabels(['Small err', 'Large err'])

        # order of unit cell axis sizes
        order_small = np.argsort(unit_cell_true[~large_errors], axis=1)
        order_large = np.argsort(unit_cell_true[large_errors], axis=1)
        # order: [[shortest index, middle index, longest index], ... ]
        proportions_small = np.zeros((3, 3))
        proportions_large = np.zeros((3, 3))
        for length_index in range(3):
            for uc_index in range(3):
                proportions_small[length_index, uc_index] = np.sum(order_small[:, length_index] == uc_index)
                proportions_large[length_index, uc_index] = np.sum(order_large[:, length_index] == uc_index)
        proportions_small = proportions_small / N_small
        proportions_large = proportions_large / N_large
        axes[1, 1].bar([0, 1, 2], proportions_small[0], color=colors[0], label='Small err')
        axes[1, 1].bar([4, 5, 6], proportions_small[1], color=colors[0])
        axes[1, 1].bar([8, 9, 10], proportions_small[2], color=colors[0])

        axes[1, 1].bar([0, 1, 2], proportions_large[0], color=colors[1], alpha=0.5, label='Large err')
        axes[1, 1].bar([4, 5, 6], proportions_large[1], color=colors[1], alpha=0.5)
        axes[1, 1].bar([8, 9, 10], proportions_large[2], color=colors[1], alpha=0.5)

        axes[1, 1].set_title('Order of axes lengths')
        axes[1, 1].set_xticks([1, 5, 9])
        axes[1, 1].set_xticklabels(['Shortest', 'Middle', 'Longest'])
        axes[1, 1].legend(frameon=False)

        # centering
        primitive = np.array(data['bravais_lattice'] == 'oP')
        base_centered = np.array(data['bravais_lattice'] == 'oC')
        body_centered = np.array(data['bravais_lattice'] == 'oI')
        face_centered = np.array(data['bravais_lattice'] == 'oF')

        # fraction centered
        centered_frac_small = np.sum(~primitive[~large_errors]) / N_small
        centered_frac_large = np.sum(~primitive[large_errors]) / N_large
        axes[1, 2].bar([0, 2], [centered_frac_small, centered_frac_large])
        axes[1, 2].set_xticks([0, 2])
        axes[1, 2].set_xticklabels(['Small error', 'Large error'])
        axes[1, 2].set_title('Fraction centered')

        # fraction Bravais lattice
        oC_frac_small = np.sum(base_centered[~large_errors]) / N_small
        oC_frac_large = np.sum(base_centered[large_errors]) / N_large

        oI_frac_small = np.sum(body_centered[~large_errors]) / N_small
        oI_frac_large = np.sum(body_centered[large_errors]) / N_large

        oF_frac_small = np.sum(face_centered[~large_errors]) / N_small
        oF_frac_large = np.sum(face_centered[large_errors]) / N_large

        frac_small = [oC_frac_small, oI_frac_small, oF_frac_small]
        frac_large = [oC_frac_large, oI_frac_large, oF_frac_large]

        axes[1, 3].bar([0, 1, 2], frac_small, color=colors[0], label='Small err')
        axes[1, 3].bar([0, 1, 2], frac_large, color=colors[1], alpha=0.5, label='Large err')
        axes[1, 3].set_xticks([0, 1, 2])
        axes[1, 3].set_xticklabels(['oC', 'oI', 'oF'])
        axes[1, 3].set_title('Bravais Lattice')

        fig.tight_layout()
        fig.savefig(self.save_to['results'] + '_orthorhombic_error_eval.png')
        plt.close()
