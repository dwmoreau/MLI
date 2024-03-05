from ccdc.io import CrystalReader
from ccdc.io import EntryReader
from ccdc.descriptors import CrystalDescriptors
from functools import reduce
from mpi4py import MPI
import numpy as np
import pandas as pd

from EntryHelpers import save_identifiers
from Reindexing import get_permutation
from Reindexing import reset_monoclinic


class EntryGenerator:
    def __init__(self):
        self.peak_length = 60
        self.peak_removal_tags = ['all', 'sa', 'strong', '2der', 'overlaps', 'intersect']
        # This is the information to store for each peak
        self.peak_components = {
            'theta2': 'float64',
            'd_spacing': 'float64',
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
            'permutation',
            'split',
            'reduced_unit_cell',
            'reduced_volume',
            ]

        for tag in self.peak_removal_tags:
            for component in self.peak_components.keys():
                self.data_set_components.update({f'{component}_{tag}': self.peak_components[component]})

        fwhm = 0.1
        self.dtheta2 = 0.02
        theta2_min = 0
        theta2_max = 60
        self.theta2 = np.arange(theta2_min, theta2_max, self.dtheta2)[:, np.newaxis]
        self.overlap_threshold = fwhm / 1.5

        self.pattern_generator = CrystalDescriptors.PowderPattern
        self.pattern_generator.Settings.full_width_at_half_maximum = fwhm
        self.pattern_generator.Settings.two_theta_minimum = theta2_min
        self.pattern_generator.Settings.two_theta_maximum = theta2_max - self.dtheta2
        self.pattern_generator.Settings.two_theta_step = self.dtheta2

    def remove_overlaps(self, peaks, I_pattern):
        # Calculate distance between peaks to find all the peaks closer than
        # some threshold
        new_peaks = peaks.copy()
        peaks = peaks[np.newaxis]
        diff = peaks - peaks.T
        too_close = np.argwhere(np.logical_and(diff > 0, diff < self.overlap_threshold))
        if too_close.shape[0] > 0:
            I_peaks = np.zeros(peaks.size)
            unique_all = np.unique(too_close)
            unique = np.unique(too_close[:, 0])
            I_peaks[unique_all] = I_pattern[(np.abs(self.theta2 - peaks[:, unique_all])).argmin(axis=0)]
            for index in unique:
                if new_peaks[index] != 0:
                    indices = np.where(too_close[:, 0] == index)[0]
                    others = too_close[indices[0]: indices[-1] + 1, 1]
                    cluster_indices = np.concatenate(([index], others))
                    I_cluster = I_peaks[cluster_indices]
                    argsort = np.argsort(I_cluster)
                    smaller_peaks = cluster_indices[argsort][:-1]
                    new_peaks[smaller_peaks] = 0
            not_overlapped = new_peaks != 0
            return not_overlapped
        else:
            return np.ones(peaks.size, dtype=bool)

    def get_pattern(self, crystal, unit_cell, lattice_system):
        def reset_peaks(peaks, hkl_order, indices, axes):
            n = np.count_nonzero(indices)
            peaks[:n, :, axes] = peaks[indices, :, axes]
            peaks[n:, :, axes] = 0
            hkl_order[:n, :, axes] = hkl_order[indices, :, axes]
            hkl_order[n:, :, axes] = -1
            return peaks, hkl_order

        base = self.pattern_generator.from_crystal(crystal)

        I_pattern = np.array(base.intensity)
        norm = np.linalg.norm(I_pattern)
        if norm == 0. or len(base.tick_marks) == 0:
            return None, None
        I_pattern /= norm

        ticks = base.tick_marks

        peaks = np.zeros((len(ticks), 2, 6))
        absent = np.ones(len(ticks), dtype=bool)
        hkl_order = -1 * np.ones((len(ticks), 4, 6), dtype=int)
        for i in range(len(ticks)):
            peaks[i, 0, :] = ticks[i].two_theta
            peaks[i, 1, :] = ticks[i].miller_indices.d_spacing
            hkl_order[i, :3, :] = np.array(ticks[i].miller_indices.hkl)[:, np.newaxis]
            hkl_order[i, 3, :] = i
            absent[i] = ticks[i].is_systematically_absent
        present = np.invert(absent)
        peaks, hkl_order = reset_peaks(peaks, hkl_order, present, 1)

        locations = np.argmin(np.abs(self.theta2 - peaks[:, 0, 0][np.newaxis]), axis=0)
        I_peaks = I_pattern[locations]
        large = I_peaks > 0.001
        peaks, hkl_order = reset_peaks(peaks, hkl_order, large, 2)

        diff2_I = np.gradient(np.gradient(I_pattern)) / (self.dtheta2**2)
        diff2_peaks = diff2_I[locations]
        with_2nd_der = diff2_peaks < 0
        peaks, hkl_order = reset_peaks(peaks, hkl_order, with_2nd_der, 3)

        not_overlapped = self.remove_overlaps(peaks[:, 0, 0], I_pattern)
        n_not_overlapped = np.count_nonzero(not_overlapped)
        peaks[:n_not_overlapped, :, 4] = peaks[not_overlapped, :, 0]
        peaks[n_not_overlapped:, :, 4] = 0
        hkl_order[:n_not_overlapped, :, 4] = hkl_order[not_overlapped, :, 0]
        hkl_order[n_not_overlapped:, :, 4] = 0

        order_all_removed = reduce(
            np.intersect1d,
            ((
                hkl_order[:, 3, 0],
                hkl_order[:, 3, 1],
                hkl_order[:, 3, 2],
                hkl_order[:, 3, 3],
                hkl_order[:, 3, 4]
            ))
            )
        _, indices, _ = np.intersect1d(hkl_order[:, 3, 0], order_all_removed, return_indices=True)
        peaks[:indices.size, :, 5] = peaks[indices, :, 0]
        peaks[indices.size:, :, 5] = 0
        hkl_order[:indices.size, :, 5] = hkl_order[indices, :, 0]
        hkl_order[indices.size:, :, 5] = 0
        if lattice_system == 'monoclinic' and unit_cell[4] != 90:
            unit_cell, reset_permuter = reset_monoclinic(unit_cell, radians=False)
        _, permuter = get_permutation(unit_cell)
        peaks_df = {}
        for index, tag in enumerate(self.peak_removal_tags):
            if lattice_system == 'monoclinic' and unit_cell[4] != 90:
                hkl = np.matmul(hkl_order[:, :3, index], permuter)
            else:
                hkl = hkl_order[:, :3, index]
            permuted_hkl = np.matmul(hkl, permuter)
            if tag == 'all':
                check_sa = peaks[:self.peak_length, 1, index]
            elif tag == 'intersect':
                check_i = peaks[:self.peak_length, 1, index]
            peaks_df.update({
                f'theta2_{tag}': peaks[:self.peak_length, 0, index],
                f'd_spacing_{tag}': peaks[:self.peak_length, 1, index],
                f'h_{tag}': hkl_order[:self.peak_length, 0, index],
                f'k_{tag}': hkl_order[:self.peak_length, 1, index],
                f'l_{tag}': hkl_order[:self.peak_length, 2, index],
                f'reindexed_h_{tag}': permuted_hkl[:self.peak_length, 0],
                f'reindexed_k_{tag}': permuted_hkl[:self.peak_length, 1],
                f'reindexed_l_{tag}': permuted_hkl[:self.peak_length, 2],
                })

        check_i = check_i[check_i > check_sa.max()]
        bad_counts = 0
        for value in check_i:
            if not value in check_sa:
                print(value)
                bad_counts += 1
        if bad_counts > 0:
            print(check_sa)
            print(check_i)
            print(bad_counts)
            print()
        return I_pattern, peaks_df


def fill_data_iteration(data_iteration, peaks_df, peak_removal_tags, peak_components):
    if peaks_df is not None:
        for tag in peak_removal_tags:
            for component in peak_components:
                key = f'{component}_{tag}'
                data_iteration[key] = peaks_df[key]
    return data_iteration


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
        if data_iteration[0]['database'] == 'csd':
            crystal = csd_entry_reader.entry(data_iteration[0]['identifier']).crystal
        elif data_iteration[0]['database'] == 'cod':
            try:
                crystal = CrystalReader(data_iteration[0]['cif_file_name'])[0]
            except:
                crystal = None
        if crystal is not None:
            data_iteration[0]['pattern'], peaks_df = entry_generator.get_pattern(
                crystal,
                data_iteration[0]['unit_cell'],
                data_iteration[0]['lattice_system'],
                )
            data_iteration[0] = fill_data_iteration(
                data_iteration[0], peaks_df, entry_generator.peak_removal_tags, entry_generator.peak_components
                )

        # Receive generated patterns and add to dataframe
        for rank_index in range(1, n_ranks):
            data_iteration[rank_index] = COMM.recv(source=rank_index)
        for rank_index in range(n_ranks):
            if data_iteration[rank_index]['pattern'] is None:
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
        if data_extra['database'] == 'csd':
            crystal = csd_entry_reader.entry(data_extra['identifier']).crystal
        elif data_extra['database'] == 'cod':
            try:
                crystal = CrystalReader(data_extra['cif_file_name'])[0]
            except:
                crystal = None
        if crystal is not None:
            data_extra['pattern'], peaks_df = entry_generator.get_pattern(
                crystal,
                data_extra['unit_cell'],
                data_extra['lattice_system'],
                )
            data_extra = fill_data_iteration(
                data_extra, peaks_df, entry_generator.peak_removal_tags, entry_generator.peak_components
                )
        if data_extra['pattern'] is None:
            bad_identifiers.append(data_extra['identifier'])
        else:
            group_data_set.loc[pattern_index] = data_extra
            pattern_index += 1
    return bad_identifiers, group_data_set, pattern_index

if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    n_ranks = COMM.Get_size()

    entries_per_group = 5000
    bad_identifiers_csd = []
    bad_identifiers_cod = []
    csd_entry_reader = EntryReader('CSD')
    entry_generator = EntryGenerator()
    rng = np.random.default_rng(seed=12345)
    if rank == 0:
        # opening and accessing the giant data frame is only done on rank 0
        entries_csd = pd.read_parquet(
            'data/unique_entries_csd.parquet',
            columns=entry_generator.data_frame_keys_to_keep
            )
        entries_cod = pd.read_parquet(
            'data/unique_cod_entries_not_in_csd.parquet',
            columns=entry_generator.data_frame_keys_to_keep
            )
        entries_csd = entries_csd.loc[entries_csd['lattice_system'] == 'monoclinic']
        entries_cod = entries_cod.loc[entries_cod['lattice_system'] == 'monoclinic']

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
                print(f'{hm_group_key} {counts_csd} {counts_cod}')

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
        status = True
        while status:
            data_iteration_rank = COMM.recv(source=0)
            if data_iteration_rank is None:
                status = False
            else:
                if data_iteration_rank['database'] == 'csd':
                    crystal = csd_entry_reader.entry(data_iteration_rank['identifier']).crystal
                elif data_iteration_rank['database'] == 'cod':
                    try:
                        crystal = CrystalReader(data_iteration_rank['cif_file_name'])[0]
                    except:
                        crystal = None
                if crystal is not None:
                    data_iteration_rank['pattern'], peaks_df = entry_generator.get_pattern(
                        crystal, data_iteration_rank['unit_cell'], data_iteration_rank['lattice_system']
                        )
                    data_iteration_rank = fill_data_iteration(
                        data_iteration_rank, peaks_df, entry_generator.peak_removal_tags, entry_generator.peak_components
                        )
                COMM.send(data_iteration_rank, dest=0)
