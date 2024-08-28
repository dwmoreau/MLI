import iotbx.cif
import gc
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
pd.options.mode.copy_on_write = True
import scipy.signal
import shutil
import sys

from EntryHelpers import save_identifiers
from Utilities import Q2Calculator
from Utilities import get_peak_generation_info


def edit_undefined_scatters(cif_file_name):
    new_cif_file_name = cif_file_name.replace('.cif', '_editted.cif')
    with open(cif_file_name, 'r') as cif_file, open(new_cif_file_name, 'w') as new_cif_file:
        in_loop_header = False
        in_atom_loop = False
        for line in cif_file:
            if line.startswith('loop_'):
                in_loop_header = True
                in_atom_loop = False
                count = 0
                atom_site_type_symbol_index = None
            elif in_loop_header and line.startswith('_'):
                if line.startswith('_atom_site_type_symbol'):
                    atom_site_type_symbol_index = count
                else:
                    count += 1
            elif in_loop_header and not line.startswith('_'):
                in_loop_header = False
                if atom_site_type_symbol_index is None:
                    in_atom_loop = False
                else:
                    in_atom_loop = True

            if in_atom_loop:
                new_elements = []
                for element_index, element in enumerate(line[:-1].split(' ')):
                    if element_index == atom_site_type_symbol_index:
                        element = element.replace('+', '').replace('-', '').replace('.', '')
                        element = ''.join([i for i in element if not i.isdigit()])
                    new_elements.append(element)
                new_line = ' '.join(new_elements) + '\n'
                if not 'SASH' in new_line:
                    new_cif_file.write(new_line)
            else:
                new_cif_file.write(line)
    cif_info = iotbx.cif.reader(new_cif_file_name)
    cif_structure = cif_info.build_crystal_structures()
    return cif_structure


def test_train_split(data_set, rng, train_fraction):
    # This sets up the training / validation tags so that the validation set is taken
    # evenly from the spacegroup symbols.
    # This should be updated to reflect that entries with the same spacegroup symbol
    # could be in different groups
    train_label = np.ones(data_set.shape[0], dtype=bool)
    for symbol_index, symbol in enumerate(data_set['reindexed_spacegroup_symbol_hm'].unique()):
        indices = np.where(data_set['reindexed_spacegroup_symbol_hm'] == symbol)[0]
        n_val = int(indices.size * (1 - 0.8))
        val_indices = rng.choice(indices, size=n_val, replace=False)
        train_label[val_indices] = False
    data_set['train'] = train_label
    return data_set


class EntryGenerator:
    def __init__(self, lattice_system):
        self.lattice_system = lattice_system
        self.peak_length = 60
        peak_generation_info = get_peak_generation_info()
        broadening_params = peak_generation_info['broadening_params']
        broadening_muliples = peak_generation_info['broadening_multiples']
        self.broadening_params = broadening_muliples[np.newaxis] * broadening_params[:, np.newaxis]
        self.broadening_tags = peak_generation_info['broadening_tags']
        self.wavelength = peak_generation_info['wavelength']
        self.d_min = self.wavelength / (2*np.sin(peak_generation_info['theta2_min']/2 * np.pi/180))
        self.d_max = self.wavelength / (2*np.sin(peak_generation_info['theta2_max']/2 * np.pi/180))
        self.theta2_pattern = peak_generation_info['theta2_pattern']
        self.q2_pattern = peak_generation_info['q2_pattern']
        # This is the information to store for each peak
        self.peak_components = {
            'q2': 'float64',
            'reindexed_h': 'int',
            'reindexed_k': 'int',
            'reindexed_l': 'int',
            }
        self.data_set_components = {
            'database': 'string',
            'identifier': 'string',
            'cif_file_name': 'string',
            'spacegroup_number': 'int',
            'bravais_lattice': 'string',
            'lattice_system': 'string',
            'reindexed_spacegroup_symbol_hm': 'string',
            'reindexed_unit_cell': 'float64',
            'reindexed_volume': 'float64',
            'hkl_reindexer': 'float64',
            'reciprocal_reindexed_unit_cell': 'float64',
            'reciprocal_reindexed_volume': 'float64',
            'reindexed_xnn': 'float64',
            'permutation': 'string',
            'split': 'int',
            }

        # These are the keys to load from the unique_entries.parquet file and directly copy into
        # the output entry
        self.data_frame_keys_to_keep = [
            'database',
            'identifier',
            'cif_file_name',
            'spacegroup_number',
            'bravais_lattice',
            'lattice_system',
            'spacegroup_symbol_hm',
            'volume',
            'unit_cell',
            'reindexed_spacegroup_symbol_hm',
            'reindexed_unit_cell',
            'reindexed_volume',
            'reciprocal_reindexed_unit_cell',
            'reindexed_xnn',
            'permutation',
            'hkl_reindexer',
            'split',
            ]

        # sa is for non-systematically absence peaks
        for tag in self.broadening_tags + ['sa']:
            for component in self.peak_components.keys():
                self.data_set_components.update({f'{component}_{tag}': self.peak_components[component]})

        # cctbx is not always able to make a structure factor for charged atoms.
        self.check_strings = ['+', '-', '.']

    def get_peak_list(self, data):
        def pick_peaks(I_pattern, q2_pattern, q2_peaks, hkl, peak_length):
            I_pattern /= np.trapz(I_pattern, q2_pattern)
            found_indices_pattern, _ = scipy.signal.find_peaks(I_pattern, prominence=2, distance=5)
            q2_found = q2_pattern[found_indices_pattern[:peak_length]]
            found_indices_peaks = np.argmin(np.abs(q2_found[:, np.newaxis] - q2_peaks[np.newaxis]), axis=1)
            return I_pattern, q2_peaks[found_indices_peaks], hkl[found_indices_peaks]

        cif_file_name = data['cif_file_name']
        unit_cell = data['unit_cell']
        hkl_reindexer = np.array(data['hkl_reindexer']).reshape((3, 3))
        lattice_system = data['lattice_system']

        try:
            cif_info = iotbx.cif.reader(cif_file_name)
            cif_structure = cif_info.build_crystal_structures()

            if len(list(cif_structure.keys())) == 0:
                return False, None
            key = list(cif_structure.keys())[0]
            change_scatters_name = False
            for scattering_type in cif_structure[key].scattering_types():
                for check in self.check_strings:
                    if check in scattering_type:
                        change_scatters_name = True
            if change_scatters_name:
                cif_structure = edit_undefined_scatters(cif_file_name)
                data['cif_file_name'] = cif_file_name.replace('.cif', '_editted.cif')

            miller_indices = cif_structure[key].build_miller_set(
                d_min=self.d_max, d_max=self.d_min, anomalous_flag=False
                )
            structure_factors = miller_indices.structure_factors_from_scatterers(
                cif_structure[key], algorithm='direct'
                )
            intensities = structure_factors.f_calc().as_intensity_array().data().as_numpy_array()
            hkl_peaks = miller_indices.indices().as_vec3_double().as_numpy_array()
            theta2_peaks = np.pi/180 * miller_indices.two_theta(self.wavelength, True).data().as_numpy_array()
        except Exception as error_message:
            #print(cif_file_name)
            #print(error_message)
            return False, None

        if len(cif_structure) == 0 or intensities.size < 10:
            return False, None
        
        q2_peaks = (2 * np.sin(theta2_peaks/2) / self.wavelength)**2
        # Lorentz-Polarization factor
        intensities *= (1 + np.cos(theta2_peaks)**2) / (2 * np.sin(theta2_peaks))

        sort_indices = np.argsort(q2_peaks)
        q2_peaks = q2_peaks[sort_indices]
        hkl_peaks = hkl_peaks[sort_indices]
        intensities = intensities[sort_indices]
        redundant = np.where((q2_peaks[1:] - q2_peaks[:-1]) == 0)[0] + 1
        if len(redundant) > 0:
            intensities[redundant - 1] = intensities[redundant] + intensities[redundant - 1]
            intensities = np.delete(intensities, redundant, axis=0)
            q2_peaks = np.delete(q2_peaks, redundant, axis=0)
            hkl_peaks =  np.delete(hkl_peaks, redundant, axis=0)

        breadths_q2_peaks = self.broadening_params[0, :] + self.broadening_params[1, :] * q2_peaks[:, np.newaxis]
        prefactor = 1/np.sqrt(2*np.pi*breadths_q2_peaks[:, np.newaxis]**2)
        arg = (self.q2_pattern[np.newaxis, :, np.newaxis] - q2_peaks[:, np.newaxis, np.newaxis]) / breadths_q2_peaks[:, np.newaxis]
        kernel = prefactor * np.exp(-1/2 * arg**2)
        I_pattern = np.sum(intensities[:, np.newaxis, np.newaxis] * kernel, axis=0)

        peaks_dict = {}
        for broadening_index, broadening_tag in enumerate(self.broadening_tags):
            I_norm, q2_found, hkl_found = pick_peaks(
                I_pattern[:, broadening_index], self.q2_pattern, q2_peaks, hkl_peaks, self.peak_length
                )
            reindexed_hkl_found = np.matmul(hkl_found, hkl_reindexer).round(decimals=0).astype(int)
            peak_length = min(q2_found.size, self.peak_length)
            peaks_dict.update({
                f'q2_{broadening_tag}': q2_found[:peak_length],
                f'reindexed_h_{broadening_tag}': reindexed_hkl_found[:peak_length, 0],
                f'reindexed_k_{broadening_tag}': reindexed_hkl_found[:peak_length, 1],
                f'reindexed_l_{broadening_tag}': reindexed_hkl_found[:peak_length, 2],
                })

            """
            q2_calculator = Q2Calculator(lattice_system='triclinic', hkl=hkl_found, tensorflow=False, representation='unit_cell')
            q2 = q2_calculator.get_q2(unit_cell[np.newaxis])[0]

            reindexed_q2_calculator = Q2Calculator(lattice_system='triclinic', hkl=reindexed_hkl_found, tensorflow=False, representation='unit_cell')
            reindexed_q2 = reindexed_q2_calculator.get_q2(np.array(data['reindexed_unit_cell'])[np.newaxis])[0]

            reciprocal_reindexed_q2_calculator = Q2Calculator(lattice_system='triclinic', hkl=reindexed_hkl_found, tensorflow=False, representation='reciprocal_unit_cell')
            reciprocal_reindexed_q2 = reciprocal_reindexed_q2_calculator.get_q2(np.array(data['reciprocal_reindexed_unit_cell'])[np.newaxis])[0]

            check0 = np.all(np.isclose(q2, reindexed_q2))
            check1 = np.all(np.isclose(q2, reciprocal_reindexed_q2))
            print(check0, check1)
            if not check0:
                print()
                print(np.round(np.column_stack((q2, reindexed_q2, reciprocal_reindexed_q2)), decimals=4))
            """
        peak_length = min(q2_peaks.size, self.peak_length)
        reindexed_hkl_peaks = np.matmul(
            hkl_peaks, hkl_reindexer
            ).round(decimals=0).astype(int)
        peaks_dict.update({
            'q2_sa': q2_peaks,
            'reindexed_h_sa': reindexed_hkl_peaks[:peak_length, 0],
            'reindexed_k_sa': reindexed_hkl_peaks[:peak_length, 1],
            'reindexed_l_sa': reindexed_hkl_peaks[:peak_length, 2],
            })
        return True, peaks_dict


def generate_dataset(bravais_lattice, train_fraction, entries_per_group, seed=12345):
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    n_ranks = COMM.Get_size()

    if bravais_lattice in ['cP', 'cI', 'cF']:
        lattice_system = 'cubic'
    elif bravais_lattice in ['hP']:
        lattice_system = 'hexagonal'
    elif bravais_lattice in ['hR']:
        lattice_system = 'rhombohedral'
    elif bravais_lattice in ['tI', 'tP']:
        lattice_system = 'tetragonal'
    elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
        lattice_system = 'orthorhombic'
    elif bravais_lattice in ['mC', 'mP']:
        lattice_system = 'monoclinic'
    elif bravais_lattice in ['aP']:
        lattice_system = 'triclinic'
    rng = np.random.default_rng(seed=seed)
    entry_generator = EntryGenerator(lattice_system)

    if rank == 0:
        if not os.path.exists('data/GeneratedDatasets'):
            os.mkdir('data/GeneratedDatasets')
        if os.path.exists('data/GeneratedDatasets/tmp'):
            shutil.rmtree('data/GeneratedDatasets/tmp')
        os.mkdir('data/GeneratedDatasets/tmp')
        # opening and accessing the giant data frame is only done on rank 0
        entries = []
        entries_csd = pd.read_parquet(
            'data/unique_entries_csd.parquet',
            columns=entry_generator.data_frame_keys_to_keep
            )
        entries_cod = pd.read_parquet(
            'data/unique_cod_entries_not_in_csd.parquet',
            columns=entry_generator.data_frame_keys_to_keep
            )
        entries_csd = entries_csd.loc[entries_csd['bravais_lattice'] == bravais_lattice]
        entries_cod = entries_cod.loc[entries_cod['bravais_lattice'] == bravais_lattice]

        if entries_per_group == 'all':
            entries = pd.concat([entries_csd, entries_cod], ignore_index=True)
        else:
            entries = []
            hm_groups_csd = entries_csd.groupby('reindexed_spacegroup_symbol_hm')
            hm_groups_cod = entries_cod.groupby('reindexed_spacegroup_symbol_hm')
            for hm_group_key in hm_groups_csd.groups.keys():
                final_group_csd = hm_groups_csd.get_group(hm_group_key)
                counts_csd = int(len(final_group_csd))
                if hm_group_key in hm_groups_cod.groups.keys():
                    final_group_cod = hm_groups_cod.get_group(hm_group_key)
                    counts_cod = int(len(final_group_cod))
                else:
                    group_entries_cod = None
                    counts_cod = 0

                if counts_csd >= entries_per_group:
                    indices_csd = rng.choice(counts_csd, entries_per_group, replace=False)
                    entries.append(final_group_csd.iloc[indices_csd])
                else:
                    n_group_entries_cod = entries_per_group - counts_csd
                    if n_group_entries_cod >= counts_cod:
                        entries.append(pd.concat([final_group_csd, final_group_cod], ignore_index=True))
                    else:
                        indices_cod = rng.choice(counts_cod, n_group_entries_cod, replace=False)
                        entries.append(pd.concat([final_group_csd, final_group_cod.iloc[indices_cod]], ignore_index=True))
            entries = pd.concat(entries, ignore_index=True)
        entries['rank'] = np.arange(len(entries)) % n_ranks
        entries_rank = entries[entries['rank'] == 0]
        for rank_index in range(1, n_ranks):
            COMM.send(entries[entries['rank'] == rank_index].copy(), dest=rank_index)
    else:
        entries_rank = COMM.recv(source=0)

    entries_rank.reset_index(inplace=True)
    success = np.zeros(entries_rank.shape[0], dtype=bool)
    keys = entry_generator.broadening_tags + ['sa']
    q2 = dict.fromkeys(entry_generator.broadening_tags + ['sa'])
    reindexed_h = dict.fromkeys(entry_generator.broadening_tags + ['sa'])
    reindexed_k = dict.fromkeys(entry_generator.broadening_tags + ['sa'])
    reindexed_l = dict.fromkeys(entry_generator.broadening_tags + ['sa'])
    for broadening_tag in entry_generator.broadening_tags + ['sa']:
        q2[broadening_tag] = []
        reindexed_h[broadening_tag] = []
        reindexed_k[broadening_tag] = []
        reindexed_l[broadening_tag] = []

    for entry_index in range(entries_rank.shape[0]):
        success[entry_index], peaks_dict = entry_generator.get_peak_list(entries_rank.iloc[entry_index])
        if success[entry_index]:
            for broadening_tag in entry_generator.broadening_tags + ['sa']:
                q2[broadening_tag].append(peaks_dict[f'q2_{broadening_tag}'])
                reindexed_h[broadening_tag].append(peaks_dict[f'reindexed_h_{broadening_tag}'])
                reindexed_k[broadening_tag].append(peaks_dict[f'reindexed_k_{broadening_tag}'])
                reindexed_l[broadening_tag].append(peaks_dict[f'reindexed_l_{broadening_tag}'])

    entries_rank = entries_rank.iloc[success]
    for broadening_tag in entry_generator.broadening_tags + ['sa']:
        entries_rank[f'q2_{broadening_tag}'] = q2[broadening_tag]
        entries_rank[f'reindexed_h_{broadening_tag}'] = reindexed_h[broadening_tag]
        entries_rank[f'reindexed_k_{broadening_tag}'] = reindexed_k[broadening_tag]
        entries_rank[f'reindexed_l_{broadening_tag}'] = reindexed_l[broadening_tag]

    entries_rank.to_parquet(f'data/GeneratedDatasets/tmp/chunk_{rank:02d}.parquet')

    COMM.Barrier()
    if rank == 0:
        chunks = []
        for file_name in os.listdir('data/GeneratedDatasets/tmp'):
            if file_name.startswith('chunk'):
                chunks.append(pd.read_parquet('data/GeneratedDatasets/tmp/' + file_name))
        chunks = pd.concat(chunks, ignore_index=True)
        chunks = test_train_split(chunks, rng, train_fraction)
        chunks.to_parquet(
            f'data/GeneratedDatasets/dataset_{bravais_lattice}.parquet'
            )
        shutil.rmtree('data/GeneratedDatasets/tmp')
    else:
        MPI.Finalize()

if __name__ == '__main__':
    if sys.argv[2] == 'all':
        entries_per_group = 'all'
    else:
        entries_per_group = int(sys.argv[2])
    generate_dataset(
        bravais_lattice=sys.argv[1],
        train_fraction=0.8,
        entries_per_group=entries_per_group
        )
