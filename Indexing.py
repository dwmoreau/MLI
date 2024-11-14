"""
- Model ensemble
    - minimize failure rate, maximize efficiency, maximize variance

- Integral filter model
    - Retrain PITF Models with new sigma approach
        - cubic
        - tetragonal
        - hexagonal
        - rhombohedral
        - orthorhombic
    - Test load_by_tag
    - Run baseline for comparison

- Optimization
    - Reprofile
    - standardization of monoclinic cells
    - Add mechanism for analysis of failed entries    

- 2D Indexing
    * convergence radius testing
        - regenerate peak lists
        - Remove the indexing step and just use FOM
        - Figure out what happened with MI2-58
    - Start working on 2D specific algorithms using the indexed reflections
        - Can I decompose frames into basis vectors

- Documentation
    - Rewrite
    - One page summary
    - github README.md

- Experimental Data
    - Automate triplet picking
    - GSASII tutorials
        - Create a refined peak list and attempt optimization for each powder pattern
        - https://advancedphotonsource.github.io/GSAS-II-tutorials/tutorials.html

- Spacegroup assignments:
    - Add a validation
    - https://www.markvardsen.net/projects/ExtSym/main.html
    - https://journals.iucr.org/paper?S0021889808031087
    - https://www.ba.ic.cnr.it/softwareic/expo/extinction_symbols/
    - https://journals.iucr.org/paper?fe5024
    - https://journals.iucr.org/paper?buy=yes&cnor=ce5126&showscheme=yes&sing=yes

- Regression
- Templating
- SWE
- Augmentation
- Random unit cell generator
- Indexing.py
- Data
    - Reindex triclinic in reciprocal space

Readings:
    - Look more into TREOR
        https://www.ba.ic.cnr.it/softwareic/expo/ntreor/
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
        - q2 error should be less than 0.0005 for all lines if CuKalpha radiation is used. Citing R. Hesse
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
"""
import joblib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Augmentor import Augmentor
from Evaluations import evaluate_regression
from Evaluations import evaluate_regression_pitf
from Evaluations import calibrate_regression
from Evaluations import calibrate_regression_pitf
from MITemplates import MITemplates
from PhysicsInformedModel import PhysicsInformedModel
from RandomGenerator import RandomGenerator
from Regression import Regression
from Utilities import get_hkl_matrix
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import get_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import Q2Calculator
from Utilities import read_params
from Utilities import reciprocal_uc_conversion
from Utilities import write_params


class Indexing:
    def __init__(self, aug_params=None, data_params=None, reg_params=None, template_params=None, pitf_params=None, random_params=None, seed=12345, load_bravais_lattice='all'):
        self.random_seed = seed
        self.rng = np.random.default_rng(self.random_seed)
        self.n_generated_points = 60  # This is the peak length of the generated dataset.
        self.save_to = dict.fromkeys(['results', 'data', 'regression', 'augmentor'])
        self.aug_params = aug_params
        self.data_params = data_params
        self.random_params = random_params
        self.reg_params = reg_params
        self.template_params = template_params
        self.pitf_params = pitf_params

        results_directory = os.path.join(self.data_params['base_directory'], 'models', self.data_params['tag'])
        self.save_to = {
            'results': results_directory,
            'augmentor': os.path.join(results_directory, 'augmentor'),
            'data': os.path.join(results_directory, 'data'),
            'random': os.path.join(results_directory, 'random'),
            'regression': os.path.join(results_directory, 'regression'),
            'template': os.path.join(results_directory, 'template'),
            'pitf': os.path.join(results_directory, 'pitf'),
            }

        if not os.path.exists(self.save_to['results']):
            os.mkdir(self.save_to['results'])
        if not os.path.exists(self.save_to['augmentor']):
            os.mkdir(self.save_to['augmentor'])
        if not os.path.exists(self.save_to['data']):
            os.mkdir(self.save_to['data'])
        if not os.path.exists(self.save_to['random']):
            os.mkdir(self.save_to['random'])
        if not os.path.exists(self.save_to['regression']):
            os.mkdir(self.save_to['regression'])
        if not os.path.exists(self.save_to['template']):
            os.mkdir(self.save_to['template'])
        if not os.path.exists(self.save_to['pitf']):
            os.mkdir(self.save_to['pitf'])

        if self.data_params['load_from_tag']:
            self.setup_from_tag(load_bravais_lattice)
        else:
            self.setup()

    def setup(self):
        data_params_defaults = {
            'lattice_system': None,
            'data_dir': os.path.join(self.data_params['base_directory'], 'data'),
            'augment': False,
            'n_max_group': 25000,
            'n_peaks': 20,
            'broadening_tag': '1',
            'hkl_ref_length': 500,
            }

        for key in data_params_defaults.keys():
            if not data_params_defaults[key] is None:
                if key not in self.data_params.keys():
                    self.data_params[key] = data_params_defaults[key]

        if self.data_params['lattice_system'] == 'cubic':
            self.data_params['unit_cell_indices'] = [0]
            self.data_params['bravais_lattices'] = ['cF', 'cI', 'cP']
        elif self.data_params['lattice_system'] == 'tetragonal':
            self.data_params['unit_cell_indices'] = [0, 2]
            self.data_params['bravais_lattices'] = ['tI', 'tP']
        elif self.data_params['lattice_system'] == 'orthorhombic':
            self.data_params['unit_cell_indices'] = [0, 1, 2]
            self.data_params['bravais_lattices'] = ['oC', 'oF', 'oI', 'oP']
        elif self.data_params['lattice_system'] == 'monoclinic':
            self.data_params['unit_cell_indices'] = [0, 1, 2, 4]
            self.data_params['bravais_lattices'] = ['mC', 'mP']
        elif self.data_params['lattice_system'] == 'triclinic':
            self.data_params['unit_cell_indices'] = [0, 1, 2, 3, 4, 5]
            self.data_params['bravais_lattices'] = ['aP']
        elif self.data_params['lattice_system'] == 'rhombohedral':
            self.data_params['unit_cell_indices'] = [0, 3]
            self.data_params['bravais_lattices'] = ['hR']
        elif self.data_params['lattice_system'] == 'hexagonal':
            self.data_params['unit_cell_indices'] = [0, 2]
            self.data_params['bravais_lattices'] = ['hP']
        self.data_params['unit_cell_length'] = len(self.data_params['unit_cell_indices'])
        all_labels = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        self.uc_labels = [all_labels[index] for index in self.data_params['unit_cell_indices']]

        group_spec = pd.read_excel(
            os.path.join(self.data_params['data_dir'], self.data_params['groupspec_file_name']),
            sheet_name=self.data_params['groupspec_sheet'],
            )
        group_spec = group_spec.loc[group_spec['group'].notna()]
        group_spec['hm symbol'] = group_spec['hm symbol'].str.strip()
        self.data_params['groups'] = group_spec['group'].unique()
        self.data_params['split_groups'] = []
        if self.data_params['lattice_system'] == 'tetragonal':
            for group in self.data_params['groups']:
                if group.startswith('tP'):
                    self.data_params['split_groups'].append(group.replace('tP_', 'tP_0_'))
                    self.data_params['split_groups'].append(group.replace('tP_', 'tP_1_'))
                elif group.startswith('tI'):
                    self.data_params['split_groups'].append(group.replace('tI_', 'tI_0_'))
                    self.data_params['split_groups'].append(group.replace('tI_', 'tI_1_'))
        elif self.data_params['lattice_system'] == 'hexagonal':
            for group in self.data_params['groups']:
                self.data_params['split_groups'].append(group.replace('hP_', 'hP_0_'))
                self.data_params['split_groups'].append(group.replace('hP_', 'hP_1_'))
        elif self.data_params['lattice_system'] == 'monoclinic':
            for group in self.data_params['groups']:
                if group.startswith('mP'):
                    self.data_params['split_groups'].append(group.replace('mP_', 'mP_0_'))
                    self.data_params['split_groups'].append(group.replace('mP_', 'mP_1_'))
                    self.data_params['split_groups'].append(group.replace('mP_', 'mP_4_'))
                elif group.startswith('mC'):
                    self.data_params['split_groups'].append(group.replace('mC_', 'mC_0_'))
                    self.data_params['split_groups'].append(group.replace('mC_', 'mC_1_'))
                    self.data_params['split_groups'].append(group.replace('mC_', 'mC_4_'))
        elif self.data_params['lattice_system'] == 'triclinic':
            self.data_params['split_groups'] = self.data_params['groups']
        elif self.data_params['lattice_system'] == 'orthorhombic':
            for group in self.data_params['groups']:
                if group.startswith('oC'):
                    self.data_params['split_groups'].append(group.replace('oC_', 'oC_0_'))
                    self.data_params['split_groups'].append(group.replace('oC_', 'oC_1_'))
                    self.data_params['split_groups'].append(group.replace('oC_', 'oC_2_'))
                elif group.startswith('oF'):
                    self.data_params['split_groups'].append(group.replace('oF_', 'oF_0_'))
                elif group.startswith('oI'):
                    self.data_params['split_groups'].append(group.replace('oI_', 'oI_0_'))
                elif group.startswith('oP'):
                    self.data_params['split_groups'].append(group.replace('oP_', 'oP_0_'))
        else:
            self.data_params['split_groups'] = self.data_params['groups']
        self.group_mappings = dict.fromkeys(group_spec['hm symbol'].unique())
        for index in range(len(group_spec)):
            self.group_mappings[group_spec.iloc[index]['hm symbol']] = group_spec.iloc[index]['group']
        #for key in self.group_mappings.keys():
        #    print(f'{key} -> {self.group_mappings[key]}')

    def setup_from_tag(self, load_bravais_lattice='all'):
        self.angle_scale = np.load(f'{self.save_to["data"]}/angle_scale.npy')
        self.uc_scaler = joblib.load(f'{self.save_to["data"]}/uc_scaler.bin')
        self.volume_scaler = joblib.load(f'{self.save_to["data"]}/volume_scaler.bin')
        self.q2_scaler = joblib.load(f'{self.save_to["data"]}/q2_scaler.bin')
        self.xnn_scaler = joblib.load(f'{self.save_to["data"]}/xnn_scaler.bin')

        params = read_params(f'{self.save_to["data"]}/data_params.csv')
        data_params_keys = [
            'augment',
            'bravais_lattices',
            'split_groups',
            'lattice_system',
            'data_dir',
            'n_max_group',
            'unit_cell_indices',
            'n_peaks',
            'broadening_tag',
            'unit_cell_length',
            'hkl_ref_length',
            'groupspec_file_name',
            'groupspec_sheet',
            ]

        self.data_params = dict.fromkeys(data_params_keys)
        self.data_params['load_from_tag'] = True
        bravais_lattices = params['bravais_lattices'].replace(' ', '').replace("'", '')
        self.data_params['bravais_lattices'] = bravais_lattices.split('[')[1].split(']')[0].split(',')
        split_groups = params['split_groups'].replace("'", '').replace('[', '').replace(']', '').replace(',', '')
        self.data_params['split_groups'] = split_groups.split(' ')

        self.data_params['lattice_system'] = params['lattice_system']
        if params['augment'] == 'True':
            self.data_params['augment'] = True
        elif params['augment'] == 'False':
            self.data_params['augment'] = False
        self.data_params['data_dir'] = params['data_dir']
        self.data_params['unit_cell_indices'] = np.array(params['unit_cell_indices'].split('[')[1].split(']')[0].split(','), dtype=int)
        all_labels = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        self.uc_labels = [all_labels[index] for index in self.data_params['unit_cell_indices']]

        self.data_params['n_max_group'] = int(params['n_max_group'])
        self.data_params['n_peaks'] = int(params['n_peaks'])
        self.data_params['broadening_tag'] = params['broadening_tag']
        self.data_params['hkl_ref_length'] = int(params['hkl_ref_length'])
        self.data_params['unit_cell_length'] = int(params['unit_cell_length'])
        self.data_params['groupspec_file_name'] = params['groupspec_file_name']
        self.data_params['groupspec_sheet'] = params['groupspec_sheet']
        self.reg_params['unit_cell_length'] = self.data_params['unit_cell_length']

        if load_bravais_lattice != 'all':
            self.data_params['bravais_lattices'] = [load_bravais_lattice]
            self.hkl_ref = dict.fromkeys(self.data_params['bravais_lattices'])
            self.hkl_ref[load_bravais_lattice] = np.load(
                f'{self.save_to["data"]}/hkl_ref_{load_bravais_lattice}.npy'
                )
            split_groups = []
            for split_group in self.data_params['split_groups']:
                if split_group.startswith(load_bravais_lattice):
                    split_groups.append(split_group)
                    
            self.data_params['split_groups'] = split_groups
        else:
            self.hkl_ref = dict.fromkeys(self.data_params['bravais_lattices'])
            for bravais_lattice in self.data_params['bravais_lattices']:
                self.hkl_ref[bravais_lattice] = np.load(
                    f'{self.save_to["data"]}/hkl_ref_{bravais_lattice}.npy'
                    )

    def save(self):
        reindexed_hkl = np.stack(self.data['reindexed_hkl'])
        save_to_data = self.data.copy()
        save_to_data['reindexed_h'] = list(reindexed_hkl[:, :, 0])
        save_to_data['reindexed_k'] = list(reindexed_hkl[:, :, 1])
        save_to_data['reindexed_l'] = list(reindexed_hkl[:, :, 2])
        save_to_data.drop(columns=['reindexed_hkl'], inplace=True)
        save_to_data.to_parquet(f'{self.save_to["data"]}/data.parquet')

        for bravais_lattice in self.data_params['bravais_lattices']:
            np.save(
                f'{self.save_to["data"]}/hkl_ref_{bravais_lattice}.npy',
                self.hkl_ref[bravais_lattice]
                )
        np.save(f'{self.save_to["data"]}/angle_scale.npy', self.angle_scale)
        joblib.dump(self.uc_scaler, f'{self.save_to["data"]}/uc_scaler.bin')
        joblib.dump(self.volume_scaler, f'{self.save_to["data"]}/volume_scaler.bin')
        joblib.dump(self.q2_scaler, f'{self.save_to["data"]}/q2_scaler.bin')
        joblib.dump(self.xnn_scaler, f'{self.save_to["data"]}/xnn_scaler.bin')
        write_params(self.data_params, f'{self.save_to["data"]}/data_params.csv')

    def load_data(self):
        read_columns = [
            'lattice_system',
            'bravais_lattice',
            'train',
            f'q2_{self.data_params["broadening_tag"]}',
            f'reindexed_h_{self.data_params["broadening_tag"]}',
            f'reindexed_k_{self.data_params["broadening_tag"]}',
            f'reindexed_l_{self.data_params["broadening_tag"]}',
            'reindexed_spacegroup_symbol_hm',
            'reindexed_unit_cell',
            'reindexed_volume',
            'reindexed_xnn',
            'reciprocal_reindexed_unit_cell',
            'split',
            ]

        if self.data_params['augment']:
            # These are all the non-systematically absent peaks and are used during augmentation
            # to pick new peaks.
            read_columns += ['q2_sa', 'reindexed_h_sa', 'reindexed_k_sa', 'reindexed_l_sa']

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
        points = self.data[f'q2_{self.data_params["broadening_tag"]}']
        indices = points.apply(len) >= self.data_params['n_peaks']
        self.data = self.data.loc[indices]
        points = self.data[f'q2_{self.data_params["broadening_tag"]}']
        enough_peaks = points.apply(np.count_nonzero) >= self.data_params['n_peaks']
        self.data = self.data.loc[enough_peaks]
        self.data['augmented'] = np.zeros(self.data.shape[0], dtype=bool)

        # Add label to data and down sample
        self.data['group'] = self.data['reindexed_spacegroup_symbol_hm'].map(
            lambda x: self.group_mappings[x]
            )
        if self.data_params['lattice_system'] in ['cubic', 'rhombohedral']:
            self.data['split_group'] = self.data['group']
        elif self.data_params['lattice_system'] == 'triclinic':
            self.data['split_group'] = self.data['group']
        elif self.data_params['lattice_system'] in ['tetragonal', 'hexagonal']:
            # split_0: a < c
            # split_1: a > c
            unit_cell = np.stack(self.data['reindexed_unit_cell'])[:, [0, 2]]
            split_1 = unit_cell[:, 0] > unit_cell[:, 1]
            self.data['split_group'] = self.data['group'].map(
                lambda x: x.replace('_', '_0_')
                )
            self.data.loc[split_1, 'split_group'] = self.data.loc[split_1, 'split_group'].map(
                lambda x: x.replace('_0_', '_1_')
                )
        elif self.data_params['lattice_system'] == 'monoclinic':
            # Orthorhombic & Monoclinic groups are split into different permutations of abc
            # For Orthorhombic, this is only the case for C-centered. 
            # I, F, & P centered have unit cells ordered as a < b < c
            split_group = []
            group = list(self.data['group'])
            split = np.array(self.data['split']).astype(int)
            for entry_index in range(len(self.data)):
                split_group.append(group[entry_index].replace(f'_', f'_{split[entry_index]}_'))
            self.data['split_group'] = split_group
            # Spacegroups where a & c have no symmetry elements are reindexed so a < c
            # These groups won't have entries, so this just pulls out the groups with entries.
            self.data_params['split_groups'] = sorted(list(self.data['split_group'].unique()))
        elif self.data_params['lattice_system'] == 'orthorhombic':
            # Orthorhombic & Monoclinic groups are split into different permutations of abc
            # For Orthorhombic, this is only the case for C-centered. 
            # I, F, & P centered have unit cells ordered as a < b < c
            split_group = []
            group = list(self.data['group'])
            split = np.array(self.data['split']).astype(int)
            for entry_index in range(len(self.data)):
                split_group.append(group[entry_index].replace(f'_', f'_{split[entry_index]}_'))
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
                n=min(len(data_group[index]), self.data_params['n_max_group']),
                replace=False,
                random_state=self.random_seed
                )
        self.data = pd.concat(data_group, ignore_index=True)

        q2_pd = self.data[f'q2_{self.data_params["broadening_tag"]}']
        q2 = [np.zeros(self.data_params['n_peaks']) for _ in range(self.data.shape[0])]
        for entry_index in range(self.data.shape[0]):
            q2[entry_index] = q2_pd.iloc[entry_index][:self.data_params['n_peaks']]
        self.data['q2'] = q2

        # put the hkl's together
        # I am saving the data into parquet format. It does not allow saving 2D arrays, so like hkl (20 x 3).
        # Converting to hdf5 would be very helpful here.
        reindexed_hkl = np.zeros((len(self.data), self.data_params['n_peaks'], 3), dtype=int)
        if self.data_params['augment']:
            reindexed_hkl_sa = np.zeros((len(self.data), self.n_generated_points, 3), dtype=int)
        if self.data_params['lattice_system'] in ['monoclinic', 'triclinic']:
            reciprocal_reindexed_hkl = np.zeros((len(self.data), self.data_params['n_peaks'], 3), dtype=int)
            if self.data_params['augment']:
                reciprocal_reindexed_hkl_sa = np.zeros((len(self.data), self.n_generated_points, 3), dtype=int)
        for entry_index in range(len(self.data)):
            entry = self.data.iloc[entry_index]
            reindexed_hkl[entry_index, :, 0] = entry[f'reindexed_h_{self.data_params["broadening_tag"]}'][:self.data_params['n_peaks']]
            reindexed_hkl[entry_index, :, 1] = entry[f'reindexed_k_{self.data_params["broadening_tag"]}'][:self.data_params['n_peaks']]
            reindexed_hkl[entry_index, :, 2] = entry[f'reindexed_l_{self.data_params["broadening_tag"]}'][:self.data_params['n_peaks']]
            if self.data_params['augment']:
                n_peaks_sa = entry[f'reindexed_h_sa'].size
                reindexed_hkl_sa[entry_index, :n_peaks_sa, 0] = entry['reindexed_h_sa']
                reindexed_hkl_sa[entry_index, :n_peaks_sa, 1] = entry['reindexed_k_sa']
                reindexed_hkl_sa[entry_index, :n_peaks_sa, 2] = entry['reindexed_l_sa']
        self.data['reindexed_hkl'] = list(reindexed_hkl)
        if self.data_params['augment']:
            self.data['reindexed_hkl_sa'] = list(reindexed_hkl_sa)

        drop_columns = [
            f'reindexed_h_{self.data_params["broadening_tag"]}',
            f'reindexed_k_{self.data_params["broadening_tag"]}',
            f'reindexed_l_{self.data_params["broadening_tag"]}',
            ]
        if self.data_params['augment']:
            drop_columns += [
                'reindexed_h_sa', 'reindexed_k_sa', 'reindexed_l_sa',
                ]

        self.data.drop(columns=drop_columns, inplace=True)
        self.setup_scalers()

        if self.data_params['augment']:
            self.augment_data()
            drop_columns = [
                f'q2_{self.data_params["broadening_tag"]}',
                'q2_sa',
                'reindexed_hkl_sa',
                ]
            self.data.drop(columns=drop_columns, inplace=True)

        self.setup_hkl()
        if self.data_params['augment']:
            self.evaluate_augmentation()

        # This does another shuffle.
        self.data = self.data.sample(frac=1, replace=False, random_state=self.random_seed)
        self.plot_input()
        self.save()

    def load_data_from_tag(self, load_augmented, load_train, load_bravais_lattice='all'):
        self.data = pd.read_parquet(f'{self.save_to["data"]}/data.parquet')
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

    def setup_hkl(self):
        print('Setting up the hkl labels')
        indices = np.logical_and(self.data['train'], ~self.data['augmented'])
        self.hkl_ref = dict.fromkeys(self.data_params['bravais_lattices'])
        hkl_labels = (self.data_params['hkl_ref_length'] - 1) * np.ones((
            len(self.data), self.data_params['n_peaks']),
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

            unit_cell = np.stack(bl_train_data['reindexed_unit_cell'])[:, self.data_params['unit_cell_indices']]
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
                len(bl_data), self.data_params['n_peaks']),
                dtype=int
                )

            n_missing = 0
            for entry_index in tqdm(range(len(bl_data))):
                missing = False
                for point_index in range(self.data_params['n_peaks']):
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
            max_unit_cell_scaled=self.max_unit_cell_scaled,
            n_generated_points=self.n_generated_points,
            save_to=self.save_to['augmentor'],
            seed=self.random_seed,
            uc_scaler=self.uc_scaler,
            angle_scale=self.angle_scale,
            xnn_scaler=self.xnn_scaler,
            q2_scaler=self.q2_scaler,
            )
        self.augmentor.setup(self.data, self.data_params['split_groups'])
        data_augmented = [None for _ in range(len(self.data_params['split_groups']))]
        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            print()
            print(f'Augmenting {split_group}')
            split_group_data = self.data[self.data['split_group'] == split_group]
            data_augmented[split_group_index] = self.augmentor.augment(
                split_group_data, 'reindexed_spacegroup_symbol_hm'
                )
            print(f'  Unaugmented entries: {len(split_group_data)} augmented entries: {len(data_augmented[split_group_index])}')
        data_augmented = pd.concat(data_augmented, ignore_index=True)
        self.data = pd.concat((self.data, data_augmented), ignore_index=True)
        print('Finished Augmenting')

    def evaluate_augmentation(self):
        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            print(f'Evaluating {split_group} Augmentation')
            split_group_data = self.data[self.data['split_group'] == split_group]
            self.augmentor.evaluate(split_group_data, split_group)

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
        if self.data_params['lattice_system'] in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            self.angle_scale = 1
        else:
            if self.data_params['lattice_system'] == 'monoclinic':
                angles = uc_train[:, 4]
            elif self.data_params['lattice_system'] == 'triclinic':
                angles = uc_train[:, 3:].ravel()
            elif self.data_params['lattice_system'] == 'rhombohedral':
                angles = uc_train[:, 3]
            self.angle_scale = angles[angles != np.pi/2].std()
        self.data['reindexed_unit_cell_scaled'] = self.data['reindexed_unit_cell'].apply(self.y_scale)

        # This hard codes the minimum allowed unit cell in augmented data to 2.25 A
        # The smallest unit cell found in the unique csd entries is 2.45A. 2.25A is about 90%.
        # This minimum is primarily used for augmentation. Because the augmented unit cells
        # are randomly generated, the smallest augmented unit cell will be very close to
        # this value.
        # This is important for performing regression in Xnn coordinates.
        # Setting this to a minimum of 1A for example will have a maximum Xnn of 1 1/A^2.
        # For monoclinic this led to a maximum scaled Xnn of 115 which gave an unreasonably
        # large training loss.

        # The maximum unit cell in the unique csd entries is 225A. The maximum unit cell
        # is set to 250.
        self.min_unit_cell_scaled = (2.25 - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
        self.max_unit_cell_scaled = (250 - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]

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
        def make_hkl_plot(data, n_peaks, hkl_ref_length, save_to):
            fig, axes = plt.subplots(n_peaks, 1, figsize=(6, 10), sharex=True)
            hkl_labels = np.stack(data['hkl_labels'])  # n_data x n_peaks
            bins = np.arange(0, hkl_ref_length + 1) - 0.5
            centers = (bins[1:] + bins[:-1]) / 2
            width = bins[1] - bins[0]
            for index in range(n_peaks):
                hist, _ = np.histogram(hkl_labels[:, index], bins=bins, density=True)
                axes[index].bar(centers, hist, width=width)
                axes[index].set_ylabel(f'Peak {index}')
            axes[n_peaks - 1].set_xlabel('HKL label')
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
                n_peaks=self.data_params['n_peaks'],
                hkl_ref_length=self.data_params['hkl_ref_length'],
                save_to=f'{self.save_to["data"]}/hkl_labels_unaugmented_{bravais_lattice}.png',
                )
            if self.data_params['augment']:
                bl_augmented_data = augmented_data[augmented_data['bravais_lattice'] == bravais_lattice]
                if len(bl_augmented_data) > 0:
                    make_hkl_plot(
                        data=bl_augmented_data,
                        n_peaks=self.data_params['n_peaks'],
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
            unit_cell_cov = np.cov(unit_cell[:, self.data_params['unit_cell_indices']].T)
            unit_cell_scaled_cov = np.cov(unit_cell_scaled[:, self.data_params['unit_cell_indices']].T)
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
        if self.data_params['lattice_system'] in ['orthorhombic', 'monoclinic', 'triclinic', 'hexagonal', 'tetragonal']:
            for split_group in self.data_params['split_groups']:
                data = self.data.loc[self.data['split_group'] == split_group]
                if self.data_params['lattice_system'] in ['monoclinic', 'triclinic']:
                    unit_cell = np.stack(data.loc[~data['augmented']]['reciprocal_reindexed_unit_cell'])
                else:
                    unit_cell = np.stack(data.loc[~data['augmented']]['reindexed_unit_cell'])
                order = np.argsort(unit_cell[:, :3], axis=1)
                # order: [[shortest index, middle index, longest index], ... ]
                proportions = np.zeros((3, 3))
                if self.data_params['augment']:
                    if self.data_params['lattice_system'] in ['monoclinic', 'triclinic']:
                        unit_cell_aug = np.stack(data.loc[data['augmented']]['reciprocal_reindexed_unit_cell'])
                    else:
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

        # split group similarity
        # This was worked out in a Jupyter Notebook. It is a bit hackish...
        unaugmented_data = self.data[~self.data['augmented']]
        unit_cell_volume = np.array(unaugmented_data['reindexed_volume'])
        split_group = unaugmented_data['split_group']
        split_groups = list(split_group.unique())
        if len(split_groups[0].split('_')) == 3:
            groups = ['_'.join((i.split('_')[0], i.split('_')[2])) for i in split_groups]
            groups = list(set(groups))
        elif len(split_groups[0].split('_')) == 2:
            groups = split_groups
        groups_index = np.arange(len(groups))
        spacegroup = unaugmented_data['reindexed_spacegroup_symbol_hm']
        spacegroups = list(spacegroup.unique())
        hkl_labels = np.stack(unaugmented_data['hkl_labels'])

        spacegroup_mapping = dict.fromkeys(spacegroups)
        for index in range(len(unaugmented_data)):
            if len(split_groups[0].split('_')) == 3:
                spacegroup_mapping[spacegroup[index]] = '_'.join((split_group[index].split('_')[0], split_group[index].split('_')[2]))
            elif len(split_groups[0].split('_')) == 2:
                spacegroup_mapping[spacegroup[index]] = split_group[index]

        ordered_spacegroups_ = [[] for _ in range(len(groups))]
        for sg in spacegroups:
            ordered_spacegroups_[groups.index(spacegroup_mapping[sg])].append(sg)
        ordered_spacegroups = []
        n_spacegroups_group = []
        for index in range(len(groups)):
            n_spacegroups_group.append(len(ordered_spacegroups_[index]))
            ordered_spacegroups += ordered_spacegroups_[index]

        sorted_volume = np.sort(unit_cell_volume)
        volume_bins = np.linspace(0, sorted_volume[int(0.995*sorted_volume.size)], 101)
        volume_centers = (volume_bins[1:] + volume_bins[:-1]) / 2
        volume_distributions = dict.fromkeys(ordered_spacegroups)
        n_peaks = 10
        hkl_distributions = dict.fromkeys(ordered_spacegroups)
        for sg in ordered_spacegroups:
            indices = spacegroup == sg
            sg_volume = unit_cell_volume[indices]
            sg_hkl_labels = hkl_labels[indices]
            volume_distributions[sg], _ = np.histogram(sg_volume, bins=volume_bins, density=True)
            hkl_distributions[sg] = np.zeros((hkl_labels.max(), n_peaks))
            for peak_index in range(n_peaks):
                bad_indices = sg_hkl_labels[:, peak_index] == hkl_labels.max()
                sg_hkl_labels[bad_indices, peak_index] = hkl_labels.max() - 1
                hkl_distributions[sg][:, peak_index] = np.bincount(sg_hkl_labels[:, peak_index], minlength=hkl_labels.max()) / indices.sum()

        volume_distance = np.zeros((len(ordered_spacegroups), len(ordered_spacegroups)))
        hkl_distance = np.zeros((len(ordered_spacegroups), len(ordered_spacegroups)))
        for sg0_index, sg0 in enumerate(ordered_spacegroups):
            for sg1_index, sg1 in enumerate(ordered_spacegroups):
                volume_distance[sg0_index, sg1_index] = np.trapz(np.abs(volume_distributions[sg0] - volume_distributions[sg1]), volume_centers)
                hkl_distance[sg0_index, sg1_index] = np.trapz(np.abs(hkl_distributions[sg1] - hkl_distributions[sg0]).ravel()) / n_peaks
        volume_distance[np.arange(len(ordered_spacegroups)), np.arange(len(ordered_spacegroups))] = np.nan
        hkl_distance[np.arange(len(ordered_spacegroups)), np.arange(len(ordered_spacegroups))] = np.nan

        if self.data_params['lattice_system'] == 'orthorhombic':
            fig, axes = plt.subplots(1, 1, figsize=(20, 20))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.imshow(volume_distance, cmap='seismic')
        start = 0
        for n in n_spacegroups_group:
            rect = patches.Rectangle((start - 0.5, start - 0.5), n, n, linewidth=2, edgecolor=[0, 0, 0], facecolor='none')
            axes.add_patch(rect)
            start += n
        axes.set_yticks(np.arange(len(ordered_spacegroups)))
        axes.set_xticks(np.arange(len(ordered_spacegroups)))
        axes.set_yticklabels(ordered_spacegroups)
        axes.set_xticklabels(ordered_spacegroups, rotation=90)
        fig.tight_layout()
        fig.savefig(f'{self.save_to["data"]}/volume_similarity.png')
        plt.close()

        if self.data_params['lattice_system'] == 'orthorhombic':
            fig, axes = plt.subplots(1, 1, figsize=(20, 20))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.imshow(hkl_distance, cmap='seismic')
        start = 0
        for n in n_spacegroups_group:
            rect = patches.Rectangle((start - 0.5, start - 0.5), n, n, linewidth=2, edgecolor=[0, 0, 0], facecolor='none')
            axes.add_patch(rect)
            start += n
        axes.set_yticks(np.arange(len(ordered_spacegroups)))
        axes.set_xticks(np.arange(len(ordered_spacegroups)))
        axes.set_yticklabels(ordered_spacegroups)
        axes.set_xticklabels(ordered_spacegroups, rotation=90)
        fig.tight_layout()
        fig.savefig(f'{self.save_to["data"]}/hkl_similarity.png')
        plt.close()

    def setup_random(self):
        self.random_unit_cell_generator = dict.fromkeys(self.data_params['bravais_lattices'])
        for bl_index, bravais_lattice in enumerate(self.data_params['bravais_lattices']):
            self.random_unit_cell_generator[bravais_lattice] = RandomGenerator(
                bravais_lattice=bravais_lattice,
                data_params=self.data_params,
                model_params=self.random_params[bravais_lattice],
                save_to=self.save_to['random'],
                )
            if self.random_params[bravais_lattice]['load_from_tag']:
                self.random_unit_cell_generator[bravais_lattice].load_from_tag()
            else:
                bl_data = self.data[self.data['bravais_lattice'] == bravais_lattice]
                bl_data = bl_data[~bl_data['augmented']]
                self.random_unit_cell_generator[bravais_lattice].setup()
                self.random_unit_cell_generator[bravais_lattice].train(bl_data)

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
                #unit_cell_templates = self.miller_index_templator[bravais_lattice].generate(
                #    102,
                #    self.rng,
                #    np.array(self.data.iloc[0]['q2'])
                #    )
                #print(self.data.iloc[0]['reindexed_unit_cell'])
                #for i in range(unit_cell_templates.shape[0]):
                #    print(unit_cell_templates[i])
            else:
                self.miller_index_templator[bravais_lattice].setup(
                    self.data[self.data['bravais_lattice'] == bravais_lattice]
                    )

    def setup_regression(self):
        self.unit_cell_generator = dict.fromkeys(self.data_params['split_groups'])
        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            self.unit_cell_generator[split_group] = Regression(
                split_group,
                self.data_params,
                self.reg_params[split_group],
                self.save_to['regression'],
                self.uc_scaler,
                self.angle_scale,
                self.random_seed,
                )
            self.unit_cell_generator[split_group].setup()
            if self.reg_params[split_group]['load_from_tag']:
                self.unit_cell_generator[split_group].load_from_tag()
            else:
                split_group_indices = self.data['split_group'] == split_group
                self.unit_cell_generator[split_group].train_regression(data=self.data[split_group_indices])

    def inferences_regression(self):
        reindexed_uc_pred = np.zeros((len(self.data), self.data_params['unit_cell_length']))
        reindexed_uc_pred_var = np.zeros((len(self.data), self.data_params['unit_cell_length']))

        reindexed_uc_pred_trees = np.zeros((len(self.data), self.data_params['unit_cell_length']))
        reindexed_uc_pred_var_trees = np.zeros((len(self.data), self.data_params['unit_cell_length']))

        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            split_group_indices = self.data['split_group'] == split_group
            reindexed_uc_pred[split_group_indices, :], reindexed_uc_pred_var[split_group_indices, :] = \
                self.unit_cell_generator[split_group].predict(data=self.data[split_group_indices], batch_size=1024, model='nn')
            reindexed_uc_pred_trees[split_group_indices, :], reindexed_uc_pred_var_trees[split_group_indices, :], _ = \
                self.unit_cell_generator[split_group].predict(data=self.data[split_group_indices], model='trees')

        reindexed_unit_cell_volume = get_unit_cell_volume(
            reindexed_uc_pred, partial_unit_cell=True, lattice_system=self.data_params['lattice_system']
            )
        self.data['reindexed_volume_pred'] = list(reindexed_unit_cell_volume)
        self.data['reindexed_unit_cell_pred'] = list(reindexed_uc_pred)
        self.data['reindexed_unit_cell_pred_var'] = list(reindexed_uc_pred_var)

        reindexed_unit_cell_volume_trees = get_unit_cell_volume(
            reindexed_uc_pred_trees, partial_unit_cell=True, lattice_system=self.data_params['lattice_system']
            )
        self.data['reindexed_volume_pred_trees'] = list(reindexed_unit_cell_volume_trees)
        self.data['reindexed_unit_cell_pred_trees'] = list(reindexed_uc_pred_trees)
        self.data['reindexed_unit_cell_pred_var_trees'] = list(reindexed_uc_pred_var_trees)

    def setup_pitf(self):
        self.pitf_generator = dict.fromkeys(self.data_params['split_groups'])
        for split_group_index, split_group in enumerate(self.data_params['split_groups']):
            bravais_lattice = split_group[:2]
            split_group_data = self.data[self.data['split_group'] == split_group]
            self.pitf_generator[split_group] = PhysicsInformedModel(
                split_group,
                self.data_params,
                self.pitf_params[split_group],
                self.save_to['pitf'],
                self.random_seed,
                self.q2_scaler,
                self.xnn_scaler,
                self.hkl_ref[bravais_lattice]
                )
            if self.pitf_params[split_group]['load_from_tag']:
                print('Doing nothing')
                #self.pitf_generator[split_group].load_from_tag()
                #self.pitf_generator[split_group].evaluate(split_group_data)
            else:
                self.pitf_generator[split_group].setup(split_group_data)
                self.pitf_generator[split_group].train(data=split_group_data)
                self.pitf_generator[split_group].train_calibration(data=split_group_data)
                self.pitf_generator[split_group].evaluate(split_group_data)

    def evaluate_regression(self):
        for bravais_lattice in self.data_params['bravais_lattices']:
            evaluate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='nn'
                )
            evaluate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_tree.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='trees'
                )
            calibrate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_calibration.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='nn'
                )
            calibrate_regression(
                data=self.data[self.data['bravais_lattice'] == bravais_lattice],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{bravais_lattice}_reg_calibration_tree.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='trees'
                )
        for split_group in self.data_params['split_groups']:
            evaluate_regression(
                data=self.data[self.data['split_group'] == split_group],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='nn'
                )
            evaluate_regression(
                data=self.data[self.data['split_group'] == split_group],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_tree.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='trees'
                )
            calibrate_regression(
                data=self.data[self.data['split_group'] == split_group],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_calibration.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='nn'
                )
            calibrate_regression(
                data=self.data[self.data['split_group'] == split_group],
                unit_cell_length=self.data_params['unit_cell_length'],
                unit_cell_key='reindexed_unit_cell',
                save_to_name=f'{self.save_to["regression"]}/{split_group}_reg_calibration_tree.png',
                unit_cell_indices=self.data_params['unit_cell_indices'],
                model='trees'
                )
        evaluate_regression(
            data=self.data,
            unit_cell_length=self.data_params['unit_cell_length'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg.png',
            unit_cell_indices=self.data_params['unit_cell_indices'],
            model='nn'
            )
        calibrate_regression(
            data=self.data,
            unit_cell_length=self.data_params['unit_cell_length'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg_calibration.png',
            unit_cell_indices=self.data_params['unit_cell_indices'],
            model='nn'
            )
        evaluate_regression(
            data=self.data,
            unit_cell_length=self.data_params['unit_cell_length'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg_trees.png',
            unit_cell_indices=self.data_params['unit_cell_indices'],
            model='trees'
            )
        calibrate_regression(
            data=self.data,
            unit_cell_length=self.data_params['unit_cell_length'],
            unit_cell_key='reindexed_unit_cell',
            save_to_name=f'{self.save_to["regression"]}/All_reg_calibration_trees.png',
            unit_cell_indices=self.data_params['unit_cell_indices'],
            model='trees'
            )
