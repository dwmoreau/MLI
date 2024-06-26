"""
Readings:
    - Bergmann 2004: Renewed interest in powder diffraction indexing
        - round robin comparison of existing programs
            - High quality data is the most important requirement for success
            - Most programs will find the solution with high quality data
            - Not a push button task. Need expert knowledge and manual intervention
            - Trying all available programs is a recommended approach.
    - Werner 1985: 
        - Attempt indexing as a 2D crystal first to help with dominant zones
        - Also used the 
        - Idea is to assume the Miller index of the first line is known, then try different
            miller indices for the rest of the pattern.
    - Werner 1964:
        - q2 error should be less than 0.0005 for all lines is CuKalpha radiation is used. Citing R. Hesse
        - Crazy flow diagram
    - Shirley 2003: Easy if all peaks are known
    - Altomare 2008: Precision and accuracy in determining peak positions is critical
    - Oishi-Tomiyasu 2013: Revised M20 scores
    - Bergmann 2007
    - Altomare 2009: Index-Heuristics
    - Tam & Compton (1995)
    - Paszkowicz 1996
    - Kariuki 1999
    - Le Bail 2004
    - Le Bail 2008
    - Harris 2000

Bravais Lattice | accuracy
--------------------------
cF              | 100.0%
cI              | 100.0%
cP              | 100.0%
tI              | 99.2%
tP              | 99.7%
hP              | 99.6%
hR              | 99.9%
oC              | 98.5%
oF              | 99.0%
oI              | 99.0%
oP              | 99.5%
mC              | 81 - 92%
mP              | 85%
aP              | 95 - 98%


* Generalization & SWE
    * Training
        - triclinic
    * Optimization
        x orthorhombic
        x monoclinic
        - triclinic
    * Find good redistribution and exhaustive search parameters
    - Different broadening
    - Incorporate positional error

- Dataset generation
    - put test / train split at datset generation
    - One large communication instead of many small communications

- Documentation
    - One page summary
    - Update after generalization has been implemented
    - Delete excess
    - Add discussion on mixture integer linear programing
    - Add section on physics informed target function model

- Optimization:
    - Performance
        - Find better parameters
            - Maximum number of explainers. Would 10 work fine?
            - Number of candidates
            - Number of exhaustive search cycles
            - Exhaustive search period

- Physics informed target function
    - Evaluations
        - Assignment accuracy
        - Assignment calibration
    - Generative
        - sampling from predicted Miller indices and updating the unit cell using least squares 
    - Incorporate into optimization
    - Make sure the beta-nll for regression is being used correctly
    - Optimization of uncertainty in the indexing
    - More general scaling

- Templating
    - Use a logistic regression model to predict if a candidate is within the correct neighborhood of the true unit cell
        - Inputs: 
            - normalized residuals
            - ???

- data
    - peak list
        - SACLA data
        - LCLS data
        - RRUFF

- Predictions for a single unknown candidate
    - Make a plan for 
- Dominant zone:
    - 2D and 1D optimization

- SWE
    - put test / train split at datset generation
    - Change back to a standard scaler for angle
    - Reduce number of communications in GenerateDataset.py
    - Use capital communications in GenerateDataset.py
- Regression
- Indexing.py
- Augmentation
- Assignments
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
from Evaluations import evaluate_regression_pitf
from Evaluations import calibrate_regression
from MITemplates import MITemplates
from MITemplates import MITemplates_binning
from PhysicsInformedModel import PhysicsInformedModel
from Regression import Regression_AlphaBeta
from Utilities import get_hkl_matrix
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import get_unit_cell_from_xnn
from Utilities import Q2Calculator
from Utilities import read_params
from Utilities import reciprocal_uc_conversion
from Utilities import write_params


class Indexing:
    def __init__(self, assign_params=None, aug_params=None, data_params=None, reg_params=None, template_params=None, pitf_params=None, seed=12345):
        self.random_seed = seed
        self.rng = np.random.default_rng(self.random_seed)
        self.n_generated_points = 60  # This is the peak length of the generated dataset.
        self.save_to = dict.fromkeys(['results', 'data', 'regression', 'assigner', 'augmentor'])
        self.assign_params = assign_params
        self.aug_params = aug_params
        self.data_params = data_params
        self.reg_params = reg_params
        self.template_params = template_params
        self.pitf_params = pitf_params

        results_directory = os.path.join(self.data_params['base_directory'], 'models', self.data_params['tag'])
        self.save_to = {
            'results': results_directory,
            'assigner': os.path.join(results_directory, 'assigner'),
            'augmentor': os.path.join(results_directory, 'augmentor'),
            'data': os.path.join(results_directory, 'data'),
            'regression': os.path.join(results_directory, 'regression'),
            'template': os.path.join(results_directory, 'template'),
            'pitf': os.path.join(results_directory, 'pitf'),
            }
        if not os.path.exists(self.save_to['results']):
            os.mkdir(self.save_to['results'])
            os.mkdir(self.save_to['assigner'])
            os.mkdir(self.save_to['augmentor'])
            os.mkdir(self.save_to['data'])
            os.mkdir(self.save_to['regression'])
            os.mkdir(self.save_to['template'])
            os.mkdir(self.save_to['pitf'])

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
            'points_tag': '1',
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
            self.data_params['bravais_lattices'] = ['hR']
        elif self.data_params['lattice_system'] == 'hexagonal':
            self.data_params['y_indices'] = [0, 2]
            self.data_params['bravais_lattices'] = ['hP']
        self._setup_joint()

    def setup_from_tag(self):
        self.uc_scaler = joblib.load(f'{self.save_to["data"]}/uc_scaler.bin')
        self.volume_scaler = joblib.load(f'{self.save_to["data"]}/volume_scaler.bin')
        self.q2_scaler = joblib.load(f'{self.save_to["data"]}/q2_scaler.bin')
        self.xnn_scaler = joblib.load(f'{self.save_to["data"]}/xnn_scaler.bin')

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
        # This is here because during optimization, not all bravais lattices are used for the
        # assignment model
        for bravais_lattice in self.assign_params.keys():
            for key in self.assign_params[bravais_lattice].keys():
                self.assign_params[bravais_lattice][key]['n_outputs'] = self.data_params['hkl_ref_length']
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
        
        if self.data_params['lattice_system'] == 'tetragonal':
            self.data_params['split_groups'] = []
            for group in self.data_params['groups']:
                self.data_params['split_groups'].append(group.replace('tetragonal_', 'tetragonal_0_'))
                self.data_params['split_groups'].append(group.replace('tetragonal_', 'tetragonal_1_'))
        elif self.data_params['lattice_system'] == 'monoclinic':
            self.data_params['split_groups'] = []
            for group in self.data_params['groups']:
                self.data_params['split_groups'].append(group.replace('monoclinic_', 'monoclinic_0_'))
                self.data_params['split_groups'].append(group.replace('monoclinic_', 'monoclinic_1_'))
                self.data_params['split_groups'].append(group.replace('monoclinic_', 'monoclinic_2_'))
                self.data_params['split_groups'].append(group.replace('monoclinic_', 'monoclinic_3_'))
                self.data_params['split_groups'].append(group.replace('monoclinic_', 'monoclinic_4_'))
                self.data_params['split_groups'].append(group.replace('monoclinic_', 'monoclinic_5_'))
        elif self.data_params['lattice_system'] == 'orthorhombic':
            self.data_params['split_groups'] = []
            for group in self.data_params['groups']:
                self.data_params['split_groups'].append(group.replace('orthorhombic_', 'orthorhombic_0_'))
                self.data_params['split_groups'].append(group.replace('orthorhombic_', 'orthorhombic_1_'))
                self.data_params['split_groups'].append(group.replace('orthorhombic_', 'orthorhombic_2_'))
        else:
            self.data_params['split_groups'] = self.data_params['groups']
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
            'reindexed_volume',
            'spacegroup_symbol_hm',
            'reindexed_spacegroup_symbol_hm',
            'unit_cell',
            'reindexed_unit_cell',
            'reindexed_xnn',
            f'q2_{self.data_params["points_tag"]}',
            f'h_{self.data_params["points_tag"]}',
            f'k_{self.data_params["points_tag"]}',
            f'l_{self.data_params["points_tag"]}',
            f'reindexed_h_{self.data_params["points_tag"]}',
            f'reindexed_k_{self.data_params["points_tag"]}',
            f'reindexed_l_{self.data_params["points_tag"]}',
            'permutation',
            'split'
            ]

        if self.data_params['augment']:
            # These are all the non-systematically absent peaks and are used during augmentation
            # to pick new peaks.
            read_columns += [
                'q2_sa',
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
        points = self.data[f'q2_{self.data_params["points_tag"]}']
        indices = points.apply(len) >= self.data_params['n_points']
        self.data = self.data.loc[indices]
        points = self.data[f'q2_{self.data_params["points_tag"]}']
        enough_peaks = points.apply(np.count_nonzero) >= self.data_params['n_points']
        self.data = self.data.loc[enough_peaks]
        self.data['augmented'] = np.zeros(self.data.shape[0], dtype=bool)

        # Add label to data and down sample
        self.data['group'] = self.data['reindexed_spacegroup_symbol_hm'].map(
            lambda x: self.group_mappings[x]
            )
        if self.data_params['lattice_system'] in ['cubic', 'hexagonal', 'rhombohedral', 'triclinic']:
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
        elif self.data_params['lattice_system'] in ['monoclinic', 'orthorhombic']:
            # Orthorhombic & Monoclinic groups are split into different permutations of abc
            # For Orthorhombic, this is only the case for C-centered. 
            # I, F, & P centered have unit cells ordered as a < b < c
            split_group = []
            group = list(self.data['group'])
            split = np.array(self.data['split']).astype(int)
            for entry_index in range(len(self.data)):
                split_group.append(group[entry_index].replace(
                    f'{self.data_params["lattice_system"]}_',
                    f'{self.data_params["lattice_system"]}_{split[entry_index]}_')
                    )
            self.data['split_group'] = split_group
            # Spacegroups where a & c have no symmetry elements are reindexed so a < c
            # These groups won't have entries, so this just pulls out the groups with entries.
            self.data_params['split_groups'] = sorted(list(self.data['split_group'].unique()))

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

        q2_pd = self.data[f'q2_{self.data_params["points_tag"]}']
        q2 = [np.zeros(self.data_params['n_points']) for _ in range(self.data.shape[0])]
        for entry_index in range(self.data.shape[0]):
            q2[entry_index] = q2_pd.iloc[entry_index][:self.data_params['n_points']]
        self.data['q2'] = q2

        # This sets up the training / validation tags so that the validation set is taken
        # evenly from the spacegroup symbols.
        # This should be updated to reflect that entries with the same spacegroup symbol
        # could be in different groups
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

    def load_data_from_tag(self, load_augmented, load_train, load_bravais_lattice='all'):
        self.hkl_ref = dict.fromkeys(self.data_params['bravais_lattices'])
        for bravais_lattice in self.data_params['bravais_lattices']:
            self.hkl_ref[bravais_lattice] = np.load(
                f'{self.save_to["data"]}/hkl_ref_{bravais_lattice}.npy'
                )
        if os.path.exists(f'{self.save_to["data"]}/miller_index_templates.npy'):
            self.miller_index_templates = np.load(f'{self.save_to["data"]}/miller_index_templates.npy')
        self.data = pd.read_parquet(f'{self.save_to["data"]}/data.parquet')
        if 'h' in self.data.keys():
            hkl = np.stack([
                np.stack(self.data['h'], axis=0),
                np.stack(self.data['k'], axis=0),
                np.stack(self.data['l'], axis=0),
                ], axis=2
                )
            self.data['hkl'] = list(hkl)
            self.data.drop(
                columns=['h', 'k', 'l'],
                inplace=True
                )
        if 'reindexed_h' in self.data.keys():
            reindexed_hkl = np.stack([
                np.stack(self.data['reindexed_h'], axis=0),
                np.stack(self.data['reindexed_k'], axis=0),
                np.stack(self.data['reindexed_l'], axis=0),
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
        if load_bravais_lattice != 'all':
            # This should be used during optimization when only one bravais lattice is
            # considered at a time
            self.data = self.data[self.data['bravais_lattice'] == load_bravais_lattice]
            self.data_params['bravais_lattices'] = [load_bravais_lattice]

        self.data_params['split_groups'] = sorted(list(self.data['split_group'].unique()))

    def save(self):
        hkl = np.stack(self.data['hkl'])
        reindexed_hkl = np.stack(self.data['reindexed_hkl'])
        save_to_data = self.data.copy()
        save_to_data['h'] = list(hkl[:, :, 0])
        save_to_data['k'] = list(hkl[:, :, 1])
        save_to_data['l'] = list(hkl[:, :, 2])
        save_to_data['reindexed_h'] = list(reindexed_hkl[:, :, 0])
        save_to_data['reindexed_k'] = list(reindexed_hkl[:, :, 1])
        save_to_data['reindexed_l'] = list(reindexed_hkl[:, :, 2])
        save_to_data.drop(columns=['hkl', 'reindexed_hkl'], inplace=True)
        save_to_data.to_parquet(f'{self.save_to["data"]}/data.parquet')

        for bravais_lattice in self.data_params['bravais_lattices']:
            np.save(
                f'{self.save_to["data"]}/hkl_ref_{bravais_lattice}.npy',
                self.hkl_ref[bravais_lattice]
                )
        joblib.dump(self.uc_scaler, f'{self.save_to["data"]}/uc_scaler.bin')
        joblib.dump(self.volume_scaler, f'{self.save_to["data"]}/volume_scaler.bin')
        joblib.dump(self.q2_scaler, f'{self.save_to["data"]}/q2_scaler.bin')
        joblib.dump(self.xnn_scaler, f'{self.save_to["data"]}/xnn_scaler.bin')
        write_params(self.data_params, f'{self.save_to["data"]}/data_params.csv')

    def setup_hkl(self):
        print('Setting up the hkl labels')
        indices = np.logical_and(self.data['train'], ~self.data['augmented'])
        self.hkl_ref = dict.fromkeys(self.data_params['bravais_lattices'])
        hkl_labels = (self.data_params['hkl_ref_length'] - 1) * np.ones((
            len(self.data), self.data_params['n_points']),
            dtype=int
            )
        for bravais_lattice in self.data_params['bravais_lattices']:
            self.hkl_ref[bravais_lattice] = np.load(
                os.path.join(self.data_params['data_dir'], f'hkl_ref_{bravais_lattice}.npy')
                )[:2*self.data_params['hkl_ref_length']]

            bl_indices = self.data['bravais_lattice'] == bravais_lattice
            bl_data = self.data[bl_indices]
            indices = np.logical_and(
                bl_data['bravais_lattice'] == bravais_lattice,
                bl_data['augmented'] == False
                )
            bl_train_data = bl_data[indices]

            unit_cell = np.stack(bl_train_data['reindexed_unit_cell'])[:, self.data_params['y_indices']]
            q2_ref_calculator = Q2Calculator(
                self.data_params['lattice_system'],
                self.hkl_ref[bravais_lattice],
                tensorflow=False,
                representation='unit_cell'
                )
            q2_ref = q2_ref_calculator.get_q2(unit_cell)
            sort_indices = np.argsort(q2_ref.mean(axis=0))
            if self.data_params['lattice_system'] == 'rhombohedral':
                # Rhombohedral struggles with the previous sorting based on q2 position
                # The entries with large unit cell angles (~120) tend to have many unlabeled peaks
                # This essentially sorts based on average increasing q2 given all unit cell
                # angles are 110, cos(110) = -1/2. Using a large hkl ref helps as well
                # number of entries with unlabeled peaks using a given angle (hkl_ref_length = 600):
                #   120: 200
                #   115: 193
                #   110: 189
                #   105: 184
                #   100: 249
                #   Other sorting: 870
                bl_hkl = self.hkl_ref[bravais_lattice]
                term0 = np.sum(bl_hkl**2, axis=1)
                cosine_term = np.cos(105 * np.pi/180)
                term1 = 2 * cosine_term * (bl_hkl[:, 0]*bl_hkl[:, 2] + bl_hkl[:, 1]*bl_hkl[:, 2] + bl_hkl[:, 0]*bl_hkl[:, 1])
                sort_indices = np.argsort(term0 + term1)

            self.hkl_ref[bravais_lattice] = self.hkl_ref[bravais_lattice][sort_indices]
            self.hkl_ref[bravais_lattice] = self.hkl_ref[bravais_lattice][:self.data_params['hkl_ref_length'] - 1]
            self.hkl_ref[bravais_lattice] = np.concatenate((self.hkl_ref[bravais_lattice], np.zeros((1, 3))), axis=0)

            check_ref = get_hkl_matrix(self.hkl_ref[bravais_lattice], self.data_params['lattice_system'])
            check_data = get_hkl_matrix(np.stack(bl_data['reindexed_hkl']), self.data_params['lattice_system'])

            hkl_labels_bl = (self.data_params['hkl_ref_length'] - 1) * np.ones((
                len(bl_data), self.data_params['n_points']),
                dtype=int
                )

            n_missing = 0
            for entry_index in tqdm(range(len(bl_data))):
                missing = False
                for point_index in range(self.data_params['n_points']):
                    hkl_ref_index = np.argwhere(np.all(
                        check_ref[:, :] == check_data[entry_index, point_index, :],
                        axis=1
                        ))
                    if len(hkl_ref_index) == 1:
                        hkl_labels_bl[entry_index, point_index] = hkl_ref_index
                    elif len(hkl_ref_index) == 0:
                        missing = True
                if missing:
                    n_missing += 1
            empty_ref = self.data_params['hkl_ref_length'] - np.unique(hkl_labels_bl).size
            print(f'{bravais_lattice} has {n_missing} entries with unlabeled peaks')
            print(f'{bravais_lattice} has {empty_ref} unused hkls in reference')
            hkl_labels[bl_indices] = hkl_labels_bl
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
            volume = unit_cell[:, 0]**2 * unit_cell[:, 1] * np.sin(np.pi/3)
        elif self.data_params['lattice_system'] == 'rhombohedral':
            calpha = np.cos(unit_cell[:, 1])
            volume = unit_cell[:, 0]**3 * np.sqrt(1 - 3*calpha**2 + 2*calpha**3)
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
        y_scaled[3:] = np.cos(y[3:])
        return y_scaled

    def y_revert(self, y):
        y_reverted = np.zeros(y.shape)
        y_reverted[:3] = self.uc_scaler.inverse_transform(y[:3][:, np.newaxis])[:, 0]
        y_reverted[3:] = np.arccos(y[3:])
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

        self.data['reindexed_unit_cell_scaled'] = self.data['reindexed_unit_cell'].apply(self.y_scale)
        self.data['unit_cell_scaled'] = self.data['unit_cell'].apply(self.y_scale)
        # this hard codes the minimum allowed unit cell in augmented data to 1 A
        self.min_unit_cell_scaled = (1 - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]

        # xnn scaling
        self.xnn_scaler = StandardScaler()
        xnn_train = np.stack(training_data['reindexed_xnn'])
        self.xnn_scaler.fit(xnn_train[xnn_train != 0][:, np.newaxis])
        self.data['reindexed_xnn_scaled'] = list(
            (np.stack(self.data['reindexed_xnn']) - self.xnn_scaler.mean_[0]) / self.xnn_scaler.scale_[0]
            )

        # Volume scaling
        self.volume_scaler = StandardScaler()
        volume_train = np.array(training_data['reindexed_volume'])
        self.volume_scaler.fit(volume_train[:, np.newaxis])
        self.data['reindexed_volume_scaled'] = list(self.volume_scale(np.array(self.data['reindexed_volume'])))

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

        unaugmented_data = self.data[~self.data['augmented']]
        augmented_data = self.data[self.data['augmented']]
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
        for split_group in self.data_params['split_groups']:
            group_data = self.data.loc[self.data['split_group'] == split_group]
            fig, axes = plt.subplots(2, 8, figsize=(14, 5))
            bins_scaled = np.linspace(-4, 4, 101)
            centers_scaled = (bins_scaled[1:] + bins_scaled[:-1]) / 2
            dbin_scaled = bins_scaled[1] - bins_scaled[0]

            # q2
            data = group_data[~group_data['augmented']]
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
                data_augmented = group_data[group_data['augmented']]
                q2_augmented = np.stack(data_augmented['q2']).ravel()
                hist_augmented, _ = np.histogram(q2_augmented, bins=bins, density=True)
                axes[0, 0].bar(centers, hist_augmented, width=dbin, alpha=0.5, label='Augmented')
                q2_scaled_augmented = np.stack(data_augmented['q2_scaled']).ravel()
                hist_scaled_augmented, _ = np.histogram(
                    q2_scaled_augmented,
                    bins=bins_scaled, density=True
                    )
                axes[1, 0].bar(
                    centers_scaled, hist_scaled_augmented,
                    width=dbin_scaled, alpha=0.5, label='Augmented'
                    )
            axes[0, 0].legend()

            # volume
            volume_scaled = np.array(data['reindexed_volume_scaled'])
            volume_sorted = np.sort(np.array(data['reindexed_volume'])) / plot_volume_scale
            lower = volume_sorted[int(0.005*volume_sorted.size)]
            upper = volume_sorted[int(0.995*volume_sorted.size)]
            bins = np.linspace(lower, upper, 101)
            centers = (bins[1:] + bins[:-1]) / 2
            dbin = bins[1] - bins[0]
            hist, _ = np.histogram(volume_sorted, bins=bins, density=True)
            axes[0, 1].bar(centers, hist, width=dbin)
            if self.data_params['augment']:
                volume_augmented = np.array(data_augmented['reindexed_volume']) / plot_volume_scale
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
                    hist_augmented, _ = np.histogram(
                        unit_cell_augmented[:, index],
                        bins=bins, density=True
                        )
                    axes[0, index + 2].bar(centers, hist_augmented, width=dbin, alpha=0.5)
                    hist_augmented_scaled, _ = np.histogram(
                        unit_cell_augmented_scaled[:, index],
                        bins=bins_scaled, density=True
                        )
                    axes[1, index + 2].bar(
                        centers_scaled, hist_augmented_scaled,
                        width=dbin_scaled, alpha=0.5
                        )
                axes[0, index + 2].set_title(y_labels[index])

            if self.data_params['lattice_system'] in ['monoclinic', 'rhombohedral', 'triclinic']:
                if self.data_params['lattice_system'] == 'monoclinic':
                    sorted_angles = np.sort(unit_cell[:, 4])
                elif self.data_params['lattice_system'] == 'rhombohedral':
                    sorted_angles = np.sort(unit_cell[:, 3])
                elif self.data_params['lattice_system'] == 'triclinic':
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
                    hist_scaled, _ = np.histogram(
                        unit_cell_scaled[indices, index],
                        bins=bins_scaled, density=True
                        )
                    axes[1, index + 2].bar(centers_scaled, hist_scaled, width=dbin_scaled)
                    axes[0, index + 2].set_title(y_labels[index])

                    if self.data_params['augment']:
                        indices = np.logical_and(
                            unit_cell_augmented[:, index] != np.pi/2,
                            ~np.isnan(unit_cell_augmented[:, index])
                            )
                        hist_augmented, _ = np.histogram(
                            unit_cell_augmented[indices, index],
                            bins=bins, density=True
                            )
                        axes[0, index + 2].bar(centers, hist_augmented, width=dbin, alpha=0.5)

                        indices = np.logical_and(
                            unit_cell_augmented_scaled[:, index] != 0,
                            ~np.isnan(unit_cell_augmented_scaled[:, index])
                            )
                        hist_augmented_scaled, _ = np.histogram(
                            unit_cell_augmented_scaled[indices, index],
                            bins=bins_scaled, density=True
                            )
                        axes[1, index + 2].bar(
                            centers_scaled, hist_augmented_scaled,
                            width=dbin_scaled, alpha=0.5
                            )

            axes[0, 0].set_ylabel('Raw data')
            axes[1, 0].set_ylabel('Standard Scaling')
            axes[0, 0].set_title('q2')
            axes[0, 1].set_title(f'Volume\n(x{plot_volume_scale})')
            fig.tight_layout()
            fig.savefig(f'{self.save_to["data"]}/regression_inputs_{split_group}.png')
            plt.close()

        # Histograms of Xnn
        xnn_labels = ['Xhh', 'Xkk', 'Xll', 'Xhk', 'Xhl', 'Xkl']
        for split_group in self.data_params['split_groups']:
            group_data = self.data.loc[self.data['split_group'] == split_group]
            fig, axes = plt.subplots(2, 6, figsize=(14, 5))
            bins_scaled = np.linspace(-4, 4, 101)
            centers_scaled = (bins_scaled[1:] + bins_scaled[:-1]) / 2
            dbin_scaled = bins_scaled[1] - bins_scaled[0]

            # Unit cell
            xnn = np.stack(data['reindexed_xnn'])
            if self.data_params['augment']:
                xnn_augmented = np.stack(data_augmented['reindexed_xnn'])
                xnn_augmented_scaled = np.stack(data_augmented['reindexed_xnn_scaled'])
            xnn_scaled = np.stack(data['reindexed_xnn_scaled'])
            sorted_xnn = np.sort(xnn.ravel())
            lower = sorted_xnn[int(0.005*sorted_xnn.size)]
            upper = sorted_xnn[int(0.995*sorted_xnn.size)]
            bins = np.linspace(lower, upper, 101)
            centers = (bins[1:] + bins[:-1]) / 2
            dbin = bins[1] - bins[0]
            for index in range(6):
                hist, _ = np.histogram(xnn[:, index], bins=bins, density=True)
                axes[0, index].bar(centers, hist, width=dbin)
                hist_scaled, _ = np.histogram(
                    xnn_scaled[:, index], bins=bins_scaled, density=True
                    )
                axes[1, index].bar(centers_scaled, hist_scaled, width=dbin_scaled)
                if self.data_params['augment']:
                    hist_augmented, _ = np.histogram(
                        xnn_augmented[:, index], bins=bins, density=True
                        )
                    axes[0, index].bar(centers, hist_augmented, width=dbin, alpha=0.5)
                    hist_augmented_scaled, _ = np.histogram(
                        xnn_augmented_scaled[:, index],
                        bins=bins_scaled, density=True
                        )
                    axes[1, index].bar(
                        centers_scaled, hist_augmented_scaled,
                        width=dbin_scaled, alpha=0.5
                        )
                axes[0, index].set_title(xnn_labels[index])

            axes[0, 0].set_ylabel('Raw data')
            axes[1, 0].set_ylabel('Standard Scaling')
            fig.tight_layout()
            fig.savefig(f'{self.save_to["data"]}/xnn_inputs_{split_group}.png')
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
            for split_group in self.data_params['split_groups']:
                data = self.data.loc[self.data['split_group'] == split_group]
                unit_cell = np.stack(data.loc[~data['augmented']]['reindexed_unit_cell'])
                order = np.argsort(unit_cell[:, :3], axis=1)
                # order: [[shortest index, middle index, longest index], ... ]
                proportions = np.zeros((3, 3))
                if self.data_params['augment']:
                    unit_cell_aug = np.stack(data.loc[data['augmented']]['reindexed_unit_cell'])
                    order_aug = np.argsort(unit_cell_aug[:, :3], axis=1)
                    proportions_aug = np.zeros((3, 3))
                for length_index in range(3):
                    for uc_index in range(3):
                        proportions[length_index, uc_index] = np.sum(order[:, length_index] == uc_index)
                        if self.data_params['augment']:
                            proportions_aug[length_index, uc_index] = np.sum(order_aug[:, length_index] == uc_index)
                fig, axes = plt.subplots(1, 3, figsize=(8, 4))
                for length_index in range(3):
                    axes[length_index].plot(
                        proportions[length_index],
                        marker='.', markersize=20, label='Unaugmented'
                        )
                    if self.data_params['augment']:
                        axes[length_index].plot(
                            proportions_aug[length_index],
                            marker='v', markersize=10, label='Augmented'
                            )
                    axes[length_index].set_xticks([0, 1, 2])
                    axes[length_index].set_xticklabels(['a', 'b', 'c'])
                axes[0].legend()
                axes[0].set_title('Shortest axis position')
                axes[1].set_title('Middle axis position')
                axes[2].set_title('Longest axis position')
                fig.tight_layout()
                fig.savefig(f'{self.save_to["data"]}/axis_order_{split_group}.png')
                plt.close()

    def setup_miller_index_templates(self):
        self.miller_index_templator = dict.fromkeys(self.data_params['bravais_lattices'])
        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            self.miller_index_templator[bravais_lattice] = MITemplates(
                group=bravais_lattice,
                data_params=self.data_params,
                template_params=self.template_params[bravais_lattice],
                hkl_ref=self.hkl_ref[bravais_lattice],
                save_to=self.save_to['template'],
                seed=self.random_seed
                )
            if self.template_params[bravais_lattice]['load_from_tag']:
                self.miller_index_templator[bravais_lattice].load_from_tag()
            else:
                self.miller_index_templator[bravais_lattice].setup(
                    self.data[self.data['bravais_lattice'] == bravais_lattice]
                    )

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
        reindexed_uc_pred_scaled_var = np.zeros((len(self.data), self.data_params['n_outputs']))

        reindexed_uc_pred_scaled_trees = np.zeros((len(self.data), self.data_params['n_outputs']))
        reindexed_uc_pred_scaled_var_trees = np.zeros((len(self.data), self.data_params['n_outputs']))

        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            split_group_indices = self.data['split_group'] == split_group
            reindexed_uc_pred_scaled[split_group_indices, :], reindexed_uc_pred_scaled_var[split_group_indices, :] = \
                self.unit_cell_generator[split_group].do_predictions(data=self.data[split_group_indices], batch_size=1024)
            reindexed_uc_pred_scaled_trees[split_group_indices, :], reindexed_uc_pred_scaled_var_trees[split_group_indices, :], _ = \
                self.unit_cell_generator[split_group].do_predictions_trees(data=self.data[split_group_indices])

        reindexed_uc_pred, reindexed_uc_pred_var = self.revert_predictions(
            reindexed_uc_pred_scaled, reindexed_uc_pred_scaled_var
            )
        self.data['reindexed_volume_pred'] = list(self.infer_unit_cell_volume_from_predictions(reindexed_uc_pred))
        self.data['reindexed_unit_cell_pred'] = list(reindexed_uc_pred)
        self.data['reindexed_unit_cell_pred_var'] = list(reindexed_uc_pred_var)
        self.data['reindexed_unit_cell_pred_scaled'] = list(reindexed_uc_pred_scaled)
        self.data['reindexed_unit_cell_pred_scaled_var'] = list(reindexed_uc_pred_scaled_var)

        reindexed_uc_pred_trees, reindexed_uc_pred_var_trees = self.revert_predictions(
            reindexed_uc_pred_scaled_trees, reindexed_uc_pred_scaled_var_trees
            )
        self.data['reindexed_volume_pred_trees'] = list(self.infer_unit_cell_volume_from_predictions(reindexed_uc_pred_trees))
        self.data['reindexed_unit_cell_pred_trees'] = list(reindexed_uc_pred_trees)
        self.data['reindexed_unit_cell_pred_var_trees'] = list(reindexed_uc_pred_var_trees)
        self.data['reindexed_unit_cell_pred_scaled_trees'] = list(reindexed_uc_pred_scaled_trees)
        self.data['reindexed_unit_cell_pred_scaled_var_trees'] = list(reindexed_uc_pred_scaled_var_trees)

    def setup_pitf(self):
        self.pitf_generator = dict.fromkeys(self.data_params['bravais_lattices'])
        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            self.pitf_generator[bravais_lattice] = PhysicsInformedModel(
                bravais_lattice,
                self.data_params,
                self.pitf_params[bravais_lattice],
                self.save_to['pitf'],
                self.random_seed,
                self.q2_scaler,
                self.xnn_scaler,
                self.hkl_ref[bravais_lattice]
                )
            self.pitf_generator[bravais_lattice].setup()
            if self.pitf_params[bravais_lattice]['load_from_tag']:
                self.pitf_generator[bravais_lattice].load_from_tag()
            else:
                bl_data = self.data[self.data['bravais_lattice'] == bravais_lattice]
                self.pitf_generator[bravais_lattice].train(data=bl_data[~bl_data['augmented']])

    def inferences_pitf(self):
        reindexed_xnn_pred = np.zeros((len(self.data), self.data_params['n_outputs']))
        reindexed_xnn_pred_var = np.zeros((len(self.data), self.data_params['n_outputs']))

        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            bl_indices = self.data['bravais_lattice'] == bravais_lattice
            reindexed_xnn_pred[bl_indices, :], reindexed_xnn_pred_var[bl_indices, :], _ = \
                self.pitf_generator[bravais_lattice].do_predictions(data=self.data[bl_indices], batch_size=1024)

        self.data['reindexed_xnn_pred_pitf'] = list(reindexed_xnn_pred)
        self.data['reindexed_xnn_pred_pitf_var'] = list(reindexed_xnn_pred_var)
        self.data['reindexed_unit_cell_pred_pitf'] = list(get_unit_cell_from_xnn(
            reindexed_xnn_pred,
            partial_unit_cell=True, 
            lattice_system=self.data_params['lattice_system'],
            ))

    def evaluate_pitf(self):
        for bravais_lattice in self.data_params['bravais_lattices']:
            evaluate_regression_pitf(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                save_to_name=f'{self.save_to["pitf"]}/{bravais_lattice}_reg_pitf.png',
                y_indices=self.data_params['y_indices'],
                )

    def revert_predictions(self, uc_pred_scaled=None, uc_pred_scaled_var=None):
        if not uc_pred_scaled is None:
            uc_pred = np.zeros(uc_pred_scaled.shape)
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred = uc_pred_scaled * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
            elif self.data_params['lattice_system'] == 'monoclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3] = np.arccos(uc_pred_scaled[:, 3])
            elif self.data_params['lattice_system'] == 'triclinic':
                uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 3:] = np.arccos(uc_pred_scaled[:, 3:])
            elif self.data_params['lattice_system'] == 'rhombohedral':
                uc_pred[:, 0] = uc_pred_scaled[:, 0] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
                uc_pred[:, 1] = np.arccos(uc_pred_scaled[:, 1])

        if not uc_pred_scaled_var is None:
            assert not uc_pred_scaled is None
            uc_pred_var = np.zeros(uc_pred_scaled_var.shape)
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred_var = uc_pred_scaled_var * self.uc_scaler.scale_[0]**2
            elif self.data_params['lattice_system'] == 'monoclinic':
                uc_pred_var[:, :3] = uc_pred_scaled_var[:, :3] * self.uc_scaler.scale_[0]**2
                uc_pred_var[:, 3] = uc_pred_scaled_var[:, 3] / (1 - uc_pred_scaled[:, 3]**2)
            elif self.data_params['lattice_system'] == 'triclinic':
                uc_pred_var[:, :3] = uc_pred_scaled_var[:, :3] * self.uc_scaler.scale_[0]**2
                uc_pred_var[:, 3:] = uc_pred_scaled_var[:, 3:] / (1 - uc_pred_scaled[:, 3:]**2)
            elif self.data_params['lattice_system'] == 'rhombohedral':
                uc_pred_var[:, 0] = uc_pred_scaled_var[:, 0] * self.uc_scaler.scale_[0]**2
                uc_pred_var[:, 1] = uc_pred_scaled_var[:, 1] / (1 - uc_pred_scaled[:, 1]**2)

        if not uc_pred_scaled is None and not uc_pred_scaled_var is None:
            return uc_pred, uc_pred_var
        elif not uc_pred_scaled is None and uc_pred_scaled_var is None:
            return uc_pred
        elif uc_pred_scaled is None and not uc_pred_scaled_var is None:
            return uc_pred_var

    def scale_predictions(self, uc_pred=None, uc_pred_var=None):
        if not uc_pred is None:
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred_scaled = (uc_pred - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
            elif self.data_params['lattice_system'] == 'monoclinic':
                uc_pred_scaled = np.zeros(uc_pred.shape)
                uc_pred_scaled[:, :3] = (uc_pred[:, :3] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
                uc_pred_scaled[:, 3] = np.cos(uc_pred[:, 3])
            elif self.data_params['lattice_system'] == 'triclinic':
                uc_pred_scaled = np.zeros(uc_pred.shape)
                uc_pred_scaled[:, :3] = (uc_pred[:, :3] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
                uc_pred_scaled[:, 3:] = np.cos(uc_pred[:, 3:])
            elif self.data_params['lattice_system'] == 'rhombohedral':
                uc_pred_scaled = np.zeros(uc_pred.shape)
                uc_pred_scaled[:, 0] = (uc_pred[:, 0] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
                uc_pred_scaled[:, 1] = np.cos(uc_pred[:, 1])
        if not uc_pred_var is None:
            uc_pred_scaled_var = np.zeros(uc_pred_var.shape)
            if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
                uc_pred_scaled_var = uc_pred_var / self.uc_scaler.scale_[0]**2
            elif self.data_params['lattice_system'] == 'monoclinic':
                uc_pred_scaled_var[:, :3] = uc_pred_var[:, :3] / self.uc_scaler.scale_[0]**2
                uc_pred_scaled_var[:, 3] = uc_pred_var[:, 3] * np.sin(uc_pred[:, 3])**2
            elif self.data_params['lattice_system'] == 'triclinic':
                uc_pred_scaled_var[:, :3] = uc_pred_var[:, :3] / self.uc_scaler.scale_[0]**2
                uc_pred_scaled_var[:, 3] = uc_pred_var[:, 3:] * np.sin(uc_pred[:, 3:])**2
            elif self.data_params['lattice_system'] == 'rhombohedral':
                uc_pred_scaled_var[:, 0] = uc_pred_var[:, 0] / self.uc_scaler.scale_[0]**2
                uc_pred_scaled_var[:, 1] = uc_pred_var[:, 1] * np.sin(uc_pred[:, 1])**2

        if not uc_pred is None and not uc_pred_var is None:
            return uc_pred_scaled, uc_pred_scaled_var
        elif not uc_pred is None and uc_pred_var is None:
            return uc_pred_scaled
        elif uc_pred is None and not uc_pred_var is None:
            return uc_pred_scaled_var

    def evaluate_regression(self):
        for bravais_lattice in self.data_params['bravais_lattices']:
            evaluate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg.png',
                y_indices=self.data_params['y_indices'],
                model='nn'
                )
            evaluate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_tree.png',
                y_indices=self.data_params['y_indices'],
                model='trees'
                )
            calibrate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_calibration.png',
                y_indices=self.data_params['y_indices'],
                model='nn'
                )
            calibrate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_calibration_tree.png',
                y_indices=self.data_params['y_indices'],
                model='trees'
                )
        for split_group in self.data_params['split_groups']:
            evaluate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg.png',
                y_indices=self.data_params['y_indices'],
                model='nn'
                )
            evaluate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_tree.png',
                y_indices=self.data_params['y_indices'],
                model='trees'
                )
            calibrate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_calibration.png',
                y_indices=self.data_params['y_indices'],
                model='nn'
                )
            calibrate_regression(
                data=self.data[self.data['split_group'] == split_group],
                n_outputs=self.data_params['n_outputs'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_calibration_tree.png',
                y_indices=self.data_params['y_indices'],
                model='trees'
                )
        evaluate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg.png',
            y_indices=self.data_params['y_indices'],
            model='nn'
            )
        calibrate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg_calibration.png',
            y_indices=self.data_params['y_indices'],
            model='nn'
            )
        evaluate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg_trees.png',
            y_indices=self.data_params['y_indices'],
            model='trees'
            )
        calibrate_regression(
            data=self.data,
            n_outputs=self.data_params['n_outputs'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg_calibration_trees.png',
            y_indices=self.data_params['y_indices'],
            model='trees'
            )

    def setup_assignment(self):
        # Assignments use the xnn unit cell representation.
        # This should be updated in the ParseDatabases.py file then these lines can be deleted
        # once the datasets are regenerated.
        reindexed_unit_cell = np.stack(self.data['reindexed_unit_cell'])
        reindexed_reciprocal_unit_cell = reciprocal_uc_conversion(
            reindexed_unit_cell, partial_unit_cell=False,
            )
        reindexed_xnn = get_xnn_from_reciprocal_unit_cell(
            reindexed_reciprocal_unit_cell, partial_unit_cell=False
            )
        self.data['reindexed_xnn'] = list(reindexed_xnn)
        self.assigner = dict.fromkeys(self.data_params['bravais_lattices'])
        for bravais_lattice in self.data_params['bravais_lattices']:
            self.assigner[bravais_lattice] = dict.fromkeys(self.assign_params[bravais_lattice].keys())
            bl_data = self.data[self.data['bravais_lattice'] == bravais_lattice]
            for key in self.assign_params[bravais_lattice].keys():
                self.assigner[bravais_lattice][key] = Assigner(
                    self.data_params,
                    self.assign_params[bravais_lattice][key],
                    self.hkl_ref[bravais_lattice],
                    self.q2_scaler,
                    self.save_to['assigner']
                    )
                if self.assign_params[bravais_lattice][key]['load_from_tag']:
                    self.assigner[bravais_lattice][key].load_from_tag(
                        self.assign_params[bravais_lattice][key]['tag'],
                        self.assign_params[bravais_lattice][key]['mode']
                        )
                else:
                    self.assigner[bravais_lattice][key].fit_model(
                        data=bl_data[~bl_data['augmented']],
                        xnn_key='reindexed_xnn',
                        y_indices=self.data_params['y_indices'],
                        )

    def inferences_assignment(self, keys):
        for bravais_lattice in self.data_params['bravais_lattices']:
            bl_data = self.data[self.data['bravais_lattice'] == bravais_lattice]
            for key in keys:
                unaugmented_bl_data = bl_data[~bl_data['augmented']].copy()
                softmaxes = self.assigner[bravais_lattice][key].do_predictions(
                    unaugmented_bl_data,
                    xnn_key='reindexed_xnn',
                    y_indices=self.data_params['y_indices'],
                    reload_model=False,
                    batch_size=1024,
                    )
                hkl_pred = self.convert_softmax_to_assignments(
                    softmaxes, self.hkl_ref[bravais_lattice]
                    )
                hkl_assign = softmaxes.argmax(axis=2)

                unaugmented_bl_data['hkl_labels_pred'] = list(hkl_assign)
                unaugmented_bl_data['hkl_pred'] = list(hkl_pred)
                unaugmented_bl_data['hkl_softmaxes'] = list(softmaxes)
                self.assigner[bravais_lattice][key].evaluate(
                    unaugmented_bl_data,
                    bravais_lattice,
                    xnn_key='reindexed_xnn',
                    y_indices=self.data_params['y_indices'],
                    perturb_std=self.assign_params[bravais_lattice][key]['perturb_std']
                    )

                self.assigner[bravais_lattice][key].calibrate(unaugmented_bl_data)

    def convert_softmax_to_assignments(self, softmaxes, hkl_ref):
        n_entries = softmaxes.shape[0]
        hkl_assign = softmaxes.argmax(axis=2)
        hkl_pred = np.zeros((n_entries, self.data_params['n_points'], 3))
        for entry_index in range(n_entries):
            hkl_pred[entry_index] = hkl_ref[hkl_assign[entry_index]]
        return hkl_pred
