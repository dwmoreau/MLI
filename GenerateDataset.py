import iotbx.cif
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
import scipy.signal
import time

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


def copy_info(input_info, output_container, keys):
    for key in keys:
        output_container[key] = input_info[key]
    return output_container


def generate_group_dataset(n_group_entries, n_ranks, counts, rng, entry_generator, group_entries, pattern_index, group_data_set):
    n_iterations = n_group_entries // n_ranks
    n_extra = n_group_entries % n_ranks

    # Randomness is because I am only using a subset of the entries. This prevents using just the first
    # group of entries that might have a lot of similar entries.
    indices = np.arange(counts)
    rng.shuffle(indices)
    bad_identifiers = []
    for iteration in range(n_iterations):
        iteration_indices = indices[iteration*n_ranks: (iteration + 1) * n_ranks]
        data_iteration = [dict.fromkeys(entry_generator.data_set_components) for _ in range(n_ranks)]

        # rank 0 gets the identifiers and sends them to the other ranks
        for rank_index in range(n_ranks):
            data_iteration[rank_index] = copy_info(
                group_entries.iloc[iteration_indices[rank_index]],
                data_iteration[rank_index],
                entry_generator.data_frame_keys_to_keep
                )
        for rank_index in range(1, n_ranks):
            COMM.send(data_iteration[rank_index], dest=rank_index)

        # rank 0 processes its entry
        data_iteration[0] = entry_generator.get_peak_list(data_iteration[0])
        # Receive generated patterns and add to dataframe
        for rank_index in range(1, n_ranks):
            data_iteration[rank_index] = COMM.recv(source=rank_index)
        for rank_index in range(n_ranks):
            if data_iteration[rank_index]['failed']:
                bad_identifiers.append(data_iteration[rank_index]['identifier'])
            else:
                group_data_set.loc[pattern_index] = data_iteration[rank_index]
                pattern_index += 1

    # do the extra entries
    for extra_index in range(n_extra):
        data_extra = dict.fromkeys(entry_generator.data_set_components)
        data_extra = copy_info(
            group_entries.iloc[indices[n_iterations*n_ranks + extra_index]],
            data_extra,
            entry_generator.data_frame_keys_to_keep
            )
        data_extra = entry_generator.get_peak_list(data_extra)
        if data_extra['failed']:
            bad_identifiers.append(data_extra['identifier'])
        else:
            group_data_set.loc[pattern_index] = data_extra
            pattern_index += 1
    return bad_identifiers, group_data_set, pattern_index


if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    n_ranks = COMM.Get_size()

    entries_per_group = 100000
    lattice_system = 'triclinic'
    bad_identifiers_csd = []
    bad_identifiers_cod = []
    rng = np.random.default_rng(seed=1234)
    entry_generator = EntryGenerator(lattice_system)

    if rank == 0:
        if not os.path.exists('data/GeneratedDatasets'):
            os.mkdir('data/GeneratedDatasets')
        # opening and accessing the giant data frame is only done on rank 0
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
                group_entries_csd = hm_groups_csd.get_group(hm_group_key)
                counts_csd = int(len(group_entries_csd))
                if hm_group_key in hm_groups_cod.groups.keys():
                    group_entries_cod = hm_groups_cod.get_group(hm_group_key)
                    counts_cod = int(len(group_entries_cod))
                else:
                    group_entries_cod = None
                    counts_cod = 0

                # One iteration is when one identifier is sent out to each rank.
                # There is an extra iteration where the remainder get processed with rank 0
                if counts_csd >= entries_per_group:
                    n_group_entries_csd = entries_per_group
                    n_group_entries_cod = 0
                else:
                    n_group_entries_csd = counts_csd
                    n_group_entries_cod = min(entries_per_group - n_group_entries_csd, counts_cod)

                # CSD
                bad_identifiers, bl_data_set, pattern_index = generate_group_dataset(
                    n_group_entries_csd,
                    n_ranks,
                    counts_csd,
                    rng,
                    entry_generator,
                    group_entries_csd,
                    pattern_index,
                    bl_data_set
                    )
                bad_identifiers_csd += bad_identifiers

                # COD
                bad_identifiers, bl_data_set, pattern_index = generate_group_dataset(
                    n_group_entries_cod,
                    n_ranks,
                    counts_cod,
                    rng,
                    entry_generator,
                    group_entries_cod,
                    pattern_index,
                    bl_data_set
                    )
                bad_identifiers_cod += bad_identifiers
            bl_data_set.to_parquet(f'data/GeneratedDatasets/dataset_{bravais_lattice}.parquet')

        for rank_index in range(1, n_ranks):
            COMM.send(None, dest=rank_index)
        save_identifiers('bad_identifiers_csd.txt', bad_identifiers_csd)
        save_identifiers('bad_identifiers_cod.txt', bad_identifiers_cod)
    else:
        # These are the worker ranks (rank != 0)
        # They receive information about the entry, get the peak list,
        # then return the updated information to rank 0.
        status = True
        while status:
            #start = time.time()
            data_iteration_rank = COMM.recv(source=0)
            #timepoint_0 = time.time()
            # when rank 0 is finished with the databases, it sends out a None to let the workers know
            if data_iteration_rank is None:
                status = False
            else:
                data_iteration_rank = entry_generator.get_peak_list(data_iteration_rank)
                #timepoint_1 = time.time()
                COMM.send(data_iteration_rank, dest=0)
                #timepoint_2 = time.time()
                #total = timepoint_2 - start
                #t0 = (timepoint_0 - start) / total
                #t1 = (timepoint_1 - timepoint_0) / total
                #t2 = (timepoint_2 - timepoint_1) / total
                #print(f'{rank}: {t0:0.3f} {t1:0.3f} {t2:0.3f}')
