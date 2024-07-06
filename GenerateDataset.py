import iotbx.cif
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
import scipy.signal
import shutil

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
            'h': 'int8',
            'k': 'int8',
            'l': 'int8',
            'reindexed_h': 'int8',
            'reindexed_k': 'int8',
            'reindexed_l': 'int8',
            }
        self.data_set_components = {
            'database': 'string',
            'identifier': 'string',
            'cif_file_name': 'string',
            'spacegroup_number': 'int8',
            'bravais_lattice': 'string',
            'lattice_system': 'string',
            'spacegroup_symbol_hm': 'string',
            'volume': 'float64',
            'unit_cell': 'float64',
            'reindexed_spacegroup_symbol_hm': 'string',
            'reindexed_unit_cell': 'float64',
            'reindexed_xnn': 'float64',
            'reindexed_volume': 'float64',
            'hkl_reindexer': 'float64',
            'permutation': 'string',
            'split': 'int8',
            'reduced_unit_cell': 'float64',
            'reduced_volume': 'float64',
            'pattern': 'float64',
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
            'reindexed_xnn',
            'reindexed_volume',
            'permutation',
            'hkl_reindexer',
            'split',
            'reduced_unit_cell',
            'reduced_volume',
            ]

        # sa is for non-systematically absence peaks
        for tag in self.broadening_tags + ['sa']:
            for component in self.peak_components.keys():
                self.data_set_components.update({f'{component}_{tag}': self.peak_components[component]})

        # cctbx is not always able to make a structure factor for charged atoms.
        self.check_strings = ['+', '-', '.', 'SASH']

    def get_peak_list(self, data):
        def pick_peaks(I_pattern, q2_pattern, q2, hkl):
            I_pattern /= np.trapz(I_pattern, q2_pattern)
            found_indices_pattern, _ = scipy.signal.find_peaks(I_pattern, prominence=2, distance=5)
            q2_found = q2_pattern[found_indices_pattern]
            found_indices_peaks = np.argmin(np.abs(q2_found[:, np.newaxis] - q2[np.newaxis]), axis=1)
            return I_pattern, q2[found_indices_peaks], hkl[found_indices_peaks]

        cif_file_name = data['cif_file_name']
        unit_cell = data['unit_cell']
        reindexed_unit_cell = data['reindexed_unit_cell']
        hkl_reindexer = np.array(data['hkl_reindexer']).reshape([3, 3])

        try:
            cif_info = iotbx.cif.reader(cif_file_name)
            cif_structure = cif_info.build_crystal_structures()

            if len(list(cif_structure.keys())) == 0:
                data['failed'] = True
                return data
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
            print(cif_file_name)
            print(error_message)
            data['failed'] = True
            return data

        if len(cif_structure) == 0:
            data['failed'] = True
            return data

        if intensities.size < 10:
            data['failed'] = True
            return data
        
        q2_peaks = (2 * np.sin(theta2_peaks/2) / self.wavelength)**2
        # Lorentz-Polarization factor
        intensities *= (1 + np.cos(theta2_peaks)**2) / (2 * np.sin(theta2_peaks))

        sort_indices = np.argsort(q2_peaks)
        q2_peaks = q2_peaks[sort_indices]
        hkl_peaks = hkl_peaks[sort_indices]
        intensities = intensities[sort_indices]
        redundant_indices = np.where((q2_peaks[1:] - q2_peaks[:-1]) == 0)[0] + 1

        breadths_q2_peaks = self.broadening_params[0, :] + self.broadening_params[1, :] * q2_peaks[:, np.newaxis]
        prefactor = 1/np.sqrt(2*np.pi*breadths_q2_peaks[:, np.newaxis]**2)
        arg = (self.q2_pattern[np.newaxis, :, np.newaxis] - q2_peaks[:, np.newaxis, np.newaxis]) / breadths_q2_peaks[:, np.newaxis]
        kernel = prefactor * np.exp(-1/2 * arg**2)
        I_pattern = np.sum(intensities[:, np.newaxis, np.newaxis] * kernel, axis=0)

        peaks_dict = {}
        #fig, axes = plt.subplots(3, 1, figsize=(40, 10), sharex=True)
        #for broadening_index in range(3):
        #    axes[broadening_index].plot(self.q2_pattern, I_pattern[:, broadening_index])
        for broadening_index, broadening_tag in enumerate(self.broadening_tags):
            I_norm, q2_found, hkl_found = pick_peaks(
                I_pattern[:, broadening_index], self.q2_pattern, q2_peaks, hkl_peaks
                )
            #ylim = axes[broadening_index].get_ylim()
            #for i in range(q2_found.size):
            #    axes[broadening_index].plot([q2_found[i], q2_found[i]], ylim, color=[0,0,0], linewidth=1, linestyle='dotted')
            #axes[broadening_index].set_ylim(ylim)
            #if broadening_index == 1:
            #    for i in range(q2_peaks.size):
            #        axes[1].plot([q2_peaks[i], q2_peaks[i]], [ylim[0], 0.5*ylim[1]], color=[0.8,0,0], linewidth=1, linestyle='dotted')

            reindexed_hkl_found = np.matmul(hkl_found, hkl_reindexer).round(decimals=0).astype(int)
            if self.lattice_system in ['monoclinic', 'orthorhombic', 'triclinic']:
                q2_calc = Q2Calculator(
                    lattice_system='triclinic',
                    hkl=hkl_found,
                    tensorflow=False,
                    representation='unit_cell'
                    ).get_q2(unit_cell[np.newaxis])
                reindexed_q2_calc = Q2Calculator(
                    lattice_system='triclinic',
                    hkl=reindexed_hkl_found,
                    tensorflow=False,
                    representation='unit_cell'
                    ).get_q2(reindexed_unit_cell[np.newaxis])
                check = np.isclose(q2_calc[0], reindexed_q2_calc[0]).all()
                if not check:
                    print('Reindexing Failure')
                    print(unit_cell)
                    print(reindexed_unit_cell)
                    print()
            elif self.lattice_system == 'rhombohedral':
                if np.all(unit_cell[3:] == [np.pi/2, np.pi/2, 2*np.pi/3]):
                    # If the rhombohedral was initially in the hexagonal setting it was reindexed
                    q2_calc = Q2Calculator(
                        lattice_system='hexagonal',
                        hkl=hkl_found,
                        tensorflow=False,
                        representation='unit_cell'
                        ).get_q2(unit_cell[[0, 2]][np.newaxis])
                    reindexed_q2_calc = Q2Calculator(
                        lattice_system='rhombohedral',
                        hkl=reindexed_hkl_found,
                        tensorflow=False,
                        representation='unit_cell'
                        ).get_q2(reindexed_unit_cell[[0, 3]][np.newaxis])
                    check = np.isclose(q2_calc[0], reindexed_q2_calc[0]).all()
                    if not check:
                        print('Reindexing Failure')
                        print(unit_cell)
                        print(reindexed_unit_cell)
                        print(np.column_stack((q2[0], reindexed_q2[0])).round(decimals=4))
                        print()

            peak_length = min(q2_found.size, self.peak_length)
            peaks_dict.update({
                f'q2_{broadening_tag}': q2_found[:peak_length],
                f'h_{broadening_tag}': hkl_found[:peak_length, 0],
                f'k_{broadening_tag}': hkl_found[:peak_length, 1],
                f'l_{broadening_tag}': hkl_found[:peak_length, 2],
                f'reindexed_h_{broadening_tag}': reindexed_hkl_found[:peak_length, 0],
                f'reindexed_k_{broadening_tag}': reindexed_hkl_found[:peak_length, 1],
                f'reindexed_l_{broadening_tag}': reindexed_hkl_found[:peak_length, 2],
                })

        q2_peaks = np.delete(q2_peaks, redundant_indices)
        hkl_peaks =  np.delete(hkl_peaks, redundant_indices, axis=0)
        reindexed_hkl_peaks = np.matmul(hkl_peaks, hkl_reindexer).round(decimals=0).astype(int)
        peak_length = min(q2_peaks.size, self.peak_length)
        peaks_dict.update({
            'q2_sa': q2_peaks,
            'h_sa': hkl_peaks[:peak_length, 0],
            'k_sa': hkl_peaks[:peak_length, 1],
            'l_sa': hkl_peaks[:peak_length, 2],
            'reindexed_h_sa': reindexed_hkl_peaks[:peak_length, 0],
            'reindexed_k_sa': reindexed_hkl_peaks[:peak_length, 1],
            'reindexed_l_sa': reindexed_hkl_peaks[:peak_length, 2],
            })

        #fig.tight_layout()
        #plt.show()
        data['failed'] = False
        for broadening_tag in self.broadening_tags + ['sa']:
            for component in self.peak_components:
                key = f'{component}_{broadening_tag}'
                data[key] = peaks_dict[key]
        return data
    

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    n_ranks = COMM.Get_size()

    train_fraction = 0.8
    #entries_per_group = 100
    entries_per_group = 'all'
    entries_per_chunk = 1000
    lattice_system = 'triclinic'
    bad_identifiers_csd = []
    bad_identifiers_cod = []
    rng = np.random.default_rng(seed=12345)
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
        entries_csd = entries_csd.loc[entries_csd['lattice_system'] == lattice_system]
        entries_cod = entries_cod.loc[entries_cod['lattice_system'] == lattice_system]

        if entries_per_group == 'all':
            entries = pd.concat([entries_csd, entries_cod], ignore_index=True)
        else:
            bl_groups_csd = entries_csd.groupby('bravais_lattice')
            bl_groups_cod = entries_cod.groupby('bravais_lattice')
            for bravais_lattice in bl_groups_csd.groups.keys():
                bl_data_set = pd.DataFrame(columns=entry_generator.data_set_components.keys())
                bl_data_set = bl_data_set.astype(entry_generator.data_set_components)
                pattern_index = 0
                bl_group_csd = bl_groups_csd.get_group(bravais_lattice)
                bl_group_cod = bl_groups_cod.get_group(bravais_lattice)

                hm_groups_csd = bl_group_csd.groupby('reindexed_spacegroup_symbol_hm')
                hm_groups_cod = bl_group_cod.groupby('reindexed_spacegroup_symbol_hm')
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
            COMM.send(entries[entries['rank'] == rank_index], dest=rank_index)
    else:
        entries_rank = COMM.recv(source=0)

    n_chunks = entries_rank.shape[0] // entries_per_chunk
    n_extra = entries_rank.shape[0] - n_chunks * entries_per_chunk
    for chunk_index in range(n_chunks + 1):
        start = chunk_index * entries_per_chunk
        if chunk_index < n_chunks:
            stop = (chunk_index + 1) * entries_per_chunk
        else:
            stop = start + n_extra
        new_entries_chunk = []
        for entry_index in range(start, stop):
            new_entry_chunk = entry_generator.get_peak_list(entries_rank.iloc[entry_index].copy())
            if new_entry_chunk['failed'] == False:
                new_entries_chunk.append(new_entry_chunk)
        pd.DataFrame(new_entries_chunk).to_parquet(
            f'data/GeneratedDatasets/tmp/chunk_{rank:02d}_{chunk_index:03d}.parquet'
            )
    COMM.Barrier()
    if rank == 0:
        chunks = []
        for file_name in os.listdir('data/GeneratedDatasets/tmp'):
            if file_name.startswith('chunk'):
                chunks.append(pd.read_parquet('data/GeneratedDatasets/tmp/' + file_name))
        chunks = pd.concat(chunks, ignore_index=True)
        chunks = test_train_split(chunks, rng, train_fraction)
        bl_chunks = chunks.groupby('bravais_lattice')
        for bravais_lattice in bl_chunks.groups.keys():
            bl_chunks.get_group(bravais_lattice).to_parquet(
                f'data/GeneratedDatasets/dataset_{bravais_lattice}.parquet'
                )
        shutil.rmtree('data/GeneratedDatasets/tmp')
    else:
        MPI.Finalize()
