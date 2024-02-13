from ccdc.io import EntryReader
from ccdc.descriptors import CrystalDescriptors
from functools import reduce
from mpi4py import MPI
import numpy as np
import pandas as pd

from CCDCEntryHelpers import save_identifiers


def remove_overlaps(peaks, I_pattern, theta2, threshold):
    # Calculate distance between peaks to find all the peaks closer than
    # some threshold
    new_peaks = peaks.copy()
    peaks = peaks[np.newaxis]
    diff = peaks - peaks.T
    too_close = np.argwhere(np.logical_and(diff > 0, diff < threshold))
    if too_close.shape[0] > 0:
        I_peaks = np.zeros(peaks.size)
        unique_all = np.unique(too_close)
        unique = np.unique(too_close[:, 0])
        I_peaks[unique_all] = I_pattern[(np.abs(theta2 - peaks[:, unique_all])).argmin(axis=0)]
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


def get_pattern(entry, pattern_generator, peak_length, theta2, threshold, peak_removal_tags):
    def reset_peaks(peaks, hkl_order, indices, axes):
        n = np.count_nonzero(indices)
        peaks[:n, :, axes] = peaks[indices, :, axes]
        peaks[n:, :, axes] = 0
        hkl_order[:n, :, axes] = hkl_order[indices, :, axes]
        hkl_order[n:, :, axes] = -1
        return peaks, hkl_order

    base = pattern_generator.from_crystal(entry.crystal)

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

    locations = np.argmin(np.abs(theta2 - peaks[:, 0, 0][np.newaxis]), axis=0)
    I_peaks = I_pattern[locations]
    large = I_peaks > 0.001
    peaks, hkl_order = reset_peaks(peaks, hkl_order, large, 2)

    diff2_I = np.gradient(np.gradient(I_pattern)) / (dtheta2**2)
    diff2_peaks = diff2_I[locations]
    with_2nd_der = diff2_peaks < 0
    peaks, hkl_order = reset_peaks(peaks, hkl_order, with_2nd_der, 3)

    not_overlapped = remove_overlaps(peaks[:, 0, 0], I_pattern, theta2, threshold)
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

    peaks_df = {}
    for index, tag in enumerate(peak_removal_tags):
        peaks_df.update({
            f'theta2_{tag}': peaks[:peak_length, 0, index],
            f'd_spacing_{tag}': peaks[:peak_length, 1, index],
            f'h_{tag}': hkl_order[:peak_length, 0, index],
            f'k_{tag}': hkl_order[:peak_length, 1, index],
            f'l_{tag}': hkl_order[:peak_length, 2, index],
            f'order_{tag}': hkl_order[:peak_length, 3, index],
            })
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


if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    n_ranks = COMM.Get_size()

    peak_length = 60
    entries_per_group = 25000
    peak_removal_tags = ['all', 'sa', 'strong', '2der', 'overlaps', 'intersect']
    # This is the information to store for each peak
    peak_components = {
        'theta2': 'float64',
        'd_spacing': 'float64',
        'h': 'int8',
        'k': 'int8',
        'l': 'int8',
        'order': 'int8',
        }
    data_set_components = {
        'pattern': 'float64',
        'unit_cell': 'float64',
        'volume': 'float64',
        'reduced_unit_cell': 'float64',
        'reduced_volume': 'float64',
        's6': 'float64',
        'g6': 'float64',
        'identifier': 'string',
        'crystal_family': 'string',
        'crystal_system': 'string',
        'lattice_system': 'string',
        'bravais_lattice': 'string',
        'spacegroup_number': 'int8',
        'setting': 'int8',
        'centering': 'string',
        }

    # These are the keys to load from the unique_entries.parquet file and directly copy into
    # the output entry
    data_frame_keys_to_keep = [
        'identifier',
        'unit_cell',
        'volume',
        'reduced_unit_cell',
        'reduced_volume',
        'g6',
        's6',
        'crystal_family',
        'crystal_system',
        'lattice_system',
        'bravais_lattice',
        'spacegroup_number',
        'setting',
        'centering',
        ]

    for tag in peak_removal_tags:
        for component in peak_components.keys():
            data_set_components.update({
                f'{component}_{tag}': peak_components[component]
                })

    fwhm = 0.1
    dtheta2 = 0.02
    theta2_min = 0
    theta2_max = 60
    theta2 = np.arange(theta2_min, theta2_max, dtheta2)
    theta2 = theta2.reshape((theta2.size, 1))
    pattern_length = theta2.shape[0]
    overlap_threshold = fwhm / 1.5

    pattern_generator = CrystalDescriptors.PowderPattern
    pattern_generator.Settings.full_width_at_half_maximum = fwhm
    pattern_generator.Settings.two_theta_minimum = theta2_min
    pattern_generator.Settings.two_theta_maximum = theta2_max - dtheta2
    pattern_generator.Settings.two_theta_step = dtheta2

    bad_identifiers = []
    csd_entry_reader = EntryReader('CSD')
    if rank == 0:
        # opening and accessing the giant data frame is only done on rank 0
        entries = pd.read_parquet('data/unique_entries.parquet', columns=data_frame_keys_to_keep)
        groups = entries.groupby('bravais_lattice')
        rng = np.random.default_rng(seed=12345)
        for group_key in groups.groups.keys():
            group_entries = groups.get_group(group_key)
            counts = int(len(group_entries))
            print(f'{group_key} {counts}')

            # One iteration is when one identifier is sent out to each rank.
            # There is an extra iteration where the remainder get processed with rank 0
            n_patterns = min((entries_per_group, counts))
            n_iterations = n_patterns // n_ranks
            n_extra = n_patterns % n_ranks

            group_data_set = pd.DataFrame(columns=data_set_components.keys())
            group_data_set = group_data_set.astype(data_set_components)

            indices = np.arange(counts)
            rng.shuffle(indices)

            pattern_index = 0
            for iteration in range(n_iterations):
                iteration_indices = indices[iteration*n_ranks: (iteration + 1) * n_ranks]
                data_iteration = [dict.fromkeys(data_set_components) for i in range(n_ranks)]

                # rank 0 gets the identifiers and sends them to the other ranks
                iteration_identifiers = group_entries.iloc[iteration_indices].identifier.to_list()
                for rank_index in range(n_ranks):
                    entry = group_entries.iloc[iteration_indices[rank_index]]
                    data_iteration[rank_index] = copy_info(
                        entry,
                        data_iteration[rank_index],
                        data_frame_keys_to_keep
                        )
                for rank_index in range(1, n_ranks):
                    COMM.send(data_iteration[rank_index], dest=rank_index)

                # rank 0 processes its entry
                csd_entry = csd_entry_reader.entry(data_iteration[0]['identifier'])
                data_iteration[0]['pattern'], peaks_df = get_pattern(
                    csd_entry,
                    pattern_generator,
                    peak_length,
                    theta2,
                    overlap_threshold,
                    peak_removal_tags
                    )
                data_iteration[0] = fill_data_iteration(
                    data_iteration[0], peaks_df, peak_removal_tags, peak_components
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
                data_extra = dict.fromkeys(data_set_components)
                entry = group_entries.iloc[indices[n_iterations*n_ranks + extra_index]]
                data_extra = copy_info(entry, data_extra, data_frame_keys_to_keep)
                csd_entry = csd_entry_reader.entry(data_extra['identifier'])
                data_extra['pattern'], peaks_df = \
                    get_pattern(
                        csd_entry,
                        pattern_generator,
                        peak_length,
                        theta2,
                        overlap_threshold,
                        peak_removal_tags
                        )
                data_extra = fill_data_iteration(
                    data_extra, peaks_df, peak_removal_tags, peak_components
                    )
                if data_extra['pattern'] is None:
                    bad_identifiers.append(data_extra['identifier'])
                else:
                    group_data_set.loc[pattern_index] = data_iteration[rank_index]
                    pattern_index += 1
            group_data_set.to_parquet(f'data/dataset_{group_key}.parquet')
        for rank_index in range(1, n_ranks):
            COMM.send(None, dest=rank_index)
        save_identifiers('bad_identifiers.txt', bad_identifiers)
    else:
        status = True
        while status:
            data_iteration_rank = COMM.recv(source=0)
            if data_iteration_rank is None:
                status = False
            else:
                csd_entry = csd_entry_reader.entry(data_iteration_rank['identifier'])
                data_iteration_rank['pattern'], peaks_df = \
                    get_pattern(
                        csd_entry,
                        pattern_generator,
                        peak_length,
                        theta2,
                        overlap_threshold,
                        peak_removal_tags
                        )
                data_iteration_rank = fill_data_iteration(
                    data_iteration_rank, peaks_df, peak_removal_tags, peak_components
                    )
                COMM.send(data_iteration_rank, dest=0)
