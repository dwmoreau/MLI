import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
from mlindex.utilities.IOManagers import write_params
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.Reindexing import get_split_group
from mlindex.utilities.Reindexing import reindex_entry_triclinic
from mlindex.utilities.Reindexing import reindex_entry_monoclinic
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.UnitCellTools import get_xnn_from_reciprocal_unit_cell
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell
from mlindex.utilities.UnitCellTools import reciprocal_uc_conversion


class Augmentor:
    def __init__(
            self,
            aug_params,
            data_params,
            min_unit_cell,
            max_unit_cell,
            n_generated_points,
            save_to,
            seed,
            ):
        self.aug_params = aug_params
        self.n_generated_points = n_generated_points
        self.min_unit_cell = min_unit_cell
        self.max_unit_cell = max_unit_cell
        self.save_to = save_to
        self.n_peaks = data_params['n_peaks']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.unit_cell_length = data_params['unit_cell_length']
        self.broadening_tag = data_params['broadening_tag']
        self.lattice_system = data_params['lattice_system']
        self.n_max_group = data_params['n_max_group']
        self.hkl_ref_length = data_params['hkl_ref_length']
        self.rng = np.random.default_rng(seed)
        if type(self.aug_params['augment_shift']) == float:
            self.augment_shift = self.aug_params['augment_shift']
        elif type(self.aug_params['augment_shift']) == list:
            self.augment_shift = np.linspace(
                self.aug_params['augment_shift'][0],
                self.aug_params['augment_shift'][1],
                self.unit_cell_length
                )
        # Save the augmentation parameters so they can be viewed to verify what was done.
        write_params(
            self.aug_params,
            os.path.join(
                f'{self.save_to}',
                f'aug_params_{self.aug_params["tag"]}.csv'
                )
            )

    def _get_ratio_xnn(self, reindexed_xnn):
        if self.lattice_system == 'cubic':
            return reindexed_xnn[:, 0] / reindexed_xnn[:, 0].max()
        elif self.lattice_system == 'rhombohedral':
            return 1/2 * reindexed_xnn[:, 3] / reindexed_xnn[:, 0]
        else:
            return reindexed_xnn[:, :3].min(axis=1) / reindexed_xnn[:, :3].max(axis=1)
        
    def setup(self, data, split_groups):
        print(f'\n Setting up augmentation {self.aug_params["tag"]}')
        # Calculate the number of times each entry is to be augmented
        training_data = data[data['train']]
        n_groups = len(split_groups)

        # Get the mean and standard deviation of the unit cell parameters.
        # The UC parameters are perturbed in a scaled space.
        unit_cell = np.stack(training_data['reindexed_unit_cell'])[:, self.unit_cell_indices]
        self.mean_unit_cell = unit_cell.mean(axis=0)[np.newaxis]
        self.std_unit_cell = unit_cell.std(axis=0)[np.newaxis]

        reindexed_xnn = np.stack(training_data['reindexed_xnn'])
        ratio_xnn = self._get_ratio_xnn(reindexed_xnn)
        if self.lattice_system == 'cubic':
            self.ratio_bins = self.ratio_bins = np.linspace(0, 1, 21)
        elif self.lattice_system == 'rhombohedral':
            self.ratio_bins = np.linspace(-1, 2, 21)
        else:
            self.ratio_bins = np.linspace(0, 1, 21)
        self.ratio_hist, _ = np.histogram(ratio_xnn, bins=self.ratio_bins, density=True)

        if self.aug_params['augment_method'] in ['cov', 'pca']:
            self.volume_bins = dict.fromkeys(split_groups)
            n_bins = dict.fromkeys(split_groups)
            for split_group_index, split_group in enumerate(split_groups):
                split_group_data = training_data[training_data['split_group'] == split_group]
                unit_cell_volume = np.array(split_group_data['reindexed_volume'])
                if self.aug_params['n_per_volume'] is None:
                    n_bins[split_group] = 1
                    self.volume_bins[split_group] = None
                else:
                    n_bins[split_group] = len(split_group_data) // self.aug_params['n_per_volume'] + 1
                    n_per_bin = int(len(split_group_data) / n_bins[split_group])
                    sorted_unit_cell_volume = np.sort(unit_cell_volume)
                    self.volume_bins[split_group] = sorted_unit_cell_volume[::n_per_bin]
                    if len(split_group_data) % n_bins[split_group] == 0:
                        self.volume_bins[split_group] = np.concatenate((
                            self.volume_bins[split_group], [sorted_unit_cell_volume[-1]]
                            ))
                    else:
                        self.volume_bins[split_group][-1] = sorted_unit_cell_volume[-1]

        if self.aug_params['augment_method'] == 'random':
            self.perturb_unit_cell = self.perturb_unit_cell_std

        elif self.aug_params['augment_method'] == 'cov':
            self.perturb_unit_cell = self.perturb_unit_cell_cov
            self.cov = dict.fromkeys(split_groups)
            for split_group_index, split_group in enumerate(split_groups):
                split_group_data = training_data[training_data['split_group'] == split_group]
                unit_cell = np.stack(split_group_data['reindexed_unit_cell'])
                unit_cell_scaled = (unit_cell - self.mean_unit_cell) / self.std_unit_cell
                unit_cell_volume = np.array(split_group_data['reindexed_volume'])
                if n_bins[split_group] == 1:
                    self.cov[split_group] = np.cov(unit_cell_scaled.T)
                    fig, axes = plt.subplots(1, 1, figsize=(4, 3))
                    cov_display = ConfusionMatrixDisplay(confusion_matrix=self.cov[split_group])
                    cov_display.plot(ax=axes, colorbar=False, values_format='0.2f')
                    axes.set_xlabel('')
                    axes.set_ylabel(split_group)
                    axes.set_title('Unit cell covariance')
                else:
                    print(f'Creating {n_bins[split_group]} COVs from {len(split_group_data)} entries for {split_group}')
                    self.cov[split_group] = [None for _ in range(n_bins[split_group])]
                    fig, axes = plt.subplots(1, n_bins[split_group], figsize=(2 + 1.5*n_bins[split_group], 3))
                    for bin_index in range(n_bins[split_group]):
                        bin_data_indices = np.logical_and(
                            unit_cell_volume >= self.volume_bins[split_group][bin_index],
                            unit_cell_volume <= self.volume_bins[split_group][bin_index + 1],
                            )
                        bin_unit_cell_scaled = unit_cell_scaled[bin_data_indices]
                        self.cov[split_group][bin_index] = np.cov(bin_unit_cell_scaled.T)
                        cov_display = ConfusionMatrixDisplay(confusion_matrix=self.cov[split_group][bin_index])
                        cov_display.plot(ax=axes[bin_index], colorbar=False, values_format='0.2f')
                        axes[bin_index].set_xlabel('')
                        axes[bin_index].set_ylabel('')
                    axes[0].set_title(f'Unit cell covariance\n{split_group}')
                fig.tight_layout()
                fig.savefig(os.path.join(
                    f'{self.save_to}',
                    f'aug_unit_cell_cov_{split_group}_{self.aug_params["tag"]}.png'
                    ))
                plt.close()

        elif self.aug_params['augment_method'] == 'pca':
            self.perturb_unit_cell = self.perturb_unit_cell_pca
            self.pca = dict.fromkeys(split_groups)
            self.stddev = dict.fromkeys(split_groups)
            for split_group_index, split_group in enumerate(split_groups):
                split_group_data = training_data[training_data['split_group'] == split_group]
                unit_cell = np.stack(split_group_data['reindexed_unit_cell'])[:, self.unit_cell_indices]
                unit_cell_scaled = (unit_cell - self.mean_unit_cell) / self.std_unit_cell
                unit_cell_volume = np.array(split_group_data['reindexed_volume'])
                print(f'Creating {n_bins[split_group]} PCAs from {len(split_group_data)} entries for {split_group}')
                if n_bins[split_group] == 1:
                    self.pca[split_group] = PCA(n_components=self.unit_cell_length).fit(unit_cell_scaled)
                    unit_cell_scaled_transformed = self.pca[split_group].transform(unit_cell_scaled)
                    self.stddev[split_group] = np.std(unit_cell_scaled_transformed, axis=0)

                    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                    pca_display = ConfusionMatrixDisplay(confusion_matrix=self.pca[split_group].components_)
                    pca_display.plot(ax=axes[0], colorbar=False, values_format='0.2f')
                    axes[1].plot(self.pca[split_group].singular_values_, marker='.')
                    axes[0].set_xlabel('')
                    axes[0].set_ylabel(split_group)
                    axes[0].set_title('PCA Components')
                    axes[1].set_title('PCA Singular values')
                else:
                    self.pca[split_group] = [None for _ in range(n_bins[split_group])]
                    self.stddev[split_group] = [None for _ in range(n_bins[split_group])]
                    fig, axes = plt.subplots(n_bins[split_group], 3, figsize=(6, 2 + 1.5*n_bins[split_group]))
                    for bin_index in range(n_bins[split_group]):
                        bin_data_indices = np.logical_and(
                            unit_cell_volume >= self.volume_bins[split_group][bin_index],
                            unit_cell_volume <= self.volume_bins[split_group][bin_index + 1],
                            )
                        bin_unit_cell_scaled = unit_cell_scaled[bin_data_indices]
                        self.pca[split_group][bin_index] = PCA(n_components=self.unit_cell_length).fit(bin_unit_cell_scaled)
                        bin_unit_cell_scaled_transformed = self.pca[split_group][bin_index].transform(bin_unit_cell_scaled)
                        self.stddev[split_group][bin_index] = np.std(bin_unit_cell_scaled_transformed, axis=0)
                        pca_display = ConfusionMatrixDisplay(confusion_matrix=self.pca[split_group][bin_index].components_)
                        pca_display.plot(ax=axes[bin_index, 0], colorbar=False, values_format='0.2f')
                        axes[bin_index, 1].plot(self.pca[split_group][bin_index].singular_values_, marker='.')
                        axes[bin_index, 2].plot(self.stddev[split_group][bin_index], marker='.')
                        axes[bin_index, 0].set_xlabel('')
                        axes[bin_index, 0].set_ylabel(split_group)
                        axes[bin_index, 0].set_title('PCA Components')
                        axes[bin_index, 1].set_title('PCA Singular values')
                        axes[bin_index, 2].set_title('STD of transformation')
                fig.tight_layout()
                fig.savefig(os.path.join(
                    f'{self.save_to}',
                    f'aug_unit_cell_pca_{split_group}_{self.aug_params["tag"]}.png'
                    ))
                plt.close()

        # calculate the order of the peak in the list of sa peaks
        if self.lattice_system == 'cubic':
            n_bins = 20
            n_bins_q2 = 10
            difference_bins = np.logspace(-3, -2, n_bins + 1)
        else:
            n_bins = 50
            n_bins_q2 = 15
            difference_bins = np.logspace(-4, -2, n_bins + 1)
        self.difference_centers = (difference_bins[1:] + difference_bins[:-1]) / 2

        q2_bins = np.linspace(0, 0.25, n_bins_q2 + 1)
        self.q2_centers = (q2_bins[1:] + q2_bins[:-1]) / 2
        keep_sum = np.zeros((n_bins, n_bins_q2))
        drop_sum = np.zeros((n_bins, n_bins_q2))
        first_peak = np.zeros(len(training_data))
        differences = []
        for entry_index in range(len(training_data)):
            p = False
            q2_sa = np.array(training_data.iloc[entry_index]['q2_sa'])
            q2 = np.array(training_data.iloc[entry_index][f'q2_{self.broadening_tag}'])
            q2_sa = q2_sa[~np.isinf(q2_sa)]
            q2 = q2[~np.isinf(q2)]

            # This is a time saving measure
            q2_sa = q2_sa[q2_sa < 0.25]
            q2 = q2[q2 < 0.25]
            first_peak_index = np.argwhere(q2_sa == q2[0])
            # this catches two cases in oP that are problematic entries
            if len(first_peak_index) > 0:
                first_peak[entry_index] = first_peak_index[0][0]
                difference = q2_sa[1:] - q2_sa[:-1]
                differences.append(difference)
                for peak_index in range(1, q2_sa.size - 2):
                    #diff_0 = q2_1 - q2_0
                    #diff_1 = q2_2 - q2_1
                    #diff_n-1 = q2_n - q2_n-1
                    min_separation = min(difference[peak_index], difference[peak_index - 1])
                    insert_index = np.searchsorted(difference_bins, min_separation) - 1
                    insert_index_q2 = min(np.searchsorted(q2_bins, q2_sa[peak_index]) - 1, n_bins_q2 - 1)
                    if n_bins > insert_index >= 0:
                        if q2_sa[peak_index] in q2:
                            if q2_sa[peak_index + 1] in q2:
                                keep_sum[insert_index, insert_index_q2] += 1
                            else:
                                drop_sum[insert_index, insert_index_q2] += 1

        total_sum = keep_sum + drop_sum
        centers = np.arange(self.n_generated_points)
        bins = np.append(centers - 0.5, self.n_generated_points + 0.5)
        hist, _ = np.histogram(first_peak, bins=bins, density=True)
        indices = hist > 0
        pdf = hist[indices]
        cdf = np.cumsum(pdf)
        self.first_probability = {
            'x': centers[indices],
            'pdf': pdf,
            'cdf': cdf
            }

        y = 0.5 * np.ones(keep_sum.shape)
        indices = total_sum != 0
        y[indices] = keep_sum[indices] / total_sum[indices]
        footprint = np.array([
            [False, True, False],
            [True, True, True],
            [False, True, False],
            ])
        y_filt = scipy.ndimage.median_filter(y, footprint=footprint, mode='nearest')
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            points=(self.difference_centers, self.q2_centers),
            values=y_filt,
            bounds_error=False,
            fill_value=np.nan,
            )

        difference_hist, _ = np.histogram(np.concatenate(differences), bins=difference_bins, density=True)
        fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharex='col')
        axes[0, 0].plot(self.first_probability['x'], self.first_probability['pdf'], marker='.', linestyle='none')
        axes[1, 0].plot(self.first_probability['x'], self.first_probability['cdf'], marker='.', linestyle='none')
        axes[0, 0].set_title(f'Distribution of first peak')
        axes[0, 0].set_ylabel('PDF')
        axes[1, 0].set_ylabel('CDF')
        axes[1, 0].set_xlabel('Peak index')
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        skip = int(n_bins_q2 / 5)
        for color_index, index in enumerate(range(0, n_bins_q2, skip)):
            axes[0, 1].plot(self.difference_centers, y[:, index], color=colors[color_index], linestyle='none', marker='.')
            curve = self.interpolator((self.difference_centers, self.q2_centers[index]))
            axes[0, 1].plot(self.difference_centers, curve, color=colors[color_index], label=f'{self.q2_centers[index]:0.3f}')
        axes[0, 1].legend(frameon=False, title='$q^2$', labelspacing=0.1)
        axes[0, 1].set_ylabel('Keep Rate')

        axes[1, 1].bar(self.difference_centers, difference_hist, width=difference_bins[1:] - difference_bins[:-1])
        axes[1, 1].set_ylabel('Distribution')
        axes[1, 1].set_xlabel('$q^2$ spacing')
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to}',
            f'aug_setup_{self.aug_params["tag"]}.png'
            ))
        plt.close()

    def augment(self, data, subgroup_label):
        subgroups = data[subgroup_label].unique()
        n_subgroups = len(subgroups)
        n_target_entries = self.n_max_group // n_subgroups
        n_group_entries = len(data)
        rarity = np.zeros(n_group_entries)
        for subgroup in subgroups:
            subgroup_indices = data[subgroup_label] == subgroup
            subgroup_data = data[subgroup_indices]
            n_subgroup_entries = len(subgroup_data)
            n_augment_all = n_target_entries - n_subgroup_entries
            subgroup_proportion = n_subgroup_entries / n_group_entries
            if n_augment_all > 0:
                ratio_xnn = self._get_ratio_xnn(np.stack(subgroup_data['reindexed_xnn']))
                bin_indices = np.searchsorted(self.ratio_bins, ratio_xnn) - 1
                ratio_proportion = np.take(self.ratio_hist, bin_indices)
                rarity[subgroup_indices] = 1 / (subgroup_proportion * ratio_proportion)

        n_augment = dict.fromkeys(subgroups)
        rarity_scale = self.aug_params['median_augmentation'] / np.median(rarity)
        n_augment_allgroups = np.round(rarity_scale * rarity, decimals=0).astype(int)
        cap_entries = n_augment_allgroups > self.aug_params['max_augmentation']
        n_augment_allgroups[cap_entries] = self.aug_params['max_augmentation']
        for subgroup in subgroups:
            subgroup_indices = data[subgroup_label] == subgroup
            n_augment[subgroup] = n_augment_allgroups[subgroup_indices]
            if n_augment[subgroup].sum() > n_target_entries:
                n_estimated_entries = n_augment[subgroup].sum() + subgroup_indices.sum()
                scale = n_estimated_entries / n_target_entries
                n_augment[subgroup] = np.round(scale * n_augment[subgroup], decimals=0).astype(int)

        augmented_entries = []
        for subgroup in subgroups:
            subgroup_data = data[data[subgroup_label] == subgroup]
            total_failures = 0
            print(subgroup)
            print(f'    Total entries: {len(subgroup_data)}')
            print(f'    Average n_augment: {n_augment[subgroup].mean()}')
            for entry_index in tqdm(range(len(subgroup_data))):
                entry = subgroup_data.iloc[entry_index]
                for augment_index in range(n_augment[subgroup][entry_index]):
                    augmented_entry = None
                    attempts = 0
                    while augmented_entry is None:
                        augmented_entry = self.augment_entry(entry)
                        attempts += 1
                        if attempts == 100:
                            break
                    if augmented_entry is not None:
                        augmented_entries.append(augmented_entry)
                    else:
                        total_failures += 1
            print(f'    Total failures: {total_failures}')
            print()
        augmented_entries = pd.DataFrame(augmented_entries)

        # Reindexing for triclinic and monoclinic is performed in reciprocal space
        #   For these lattice systems update:
        #       reindexed_unit_cell
        #       reindexed_hkl
        reindexed_unit_cell = np.stack(augmented_entries['reindexed_unit_cell'])
        if self.lattice_system in ['monoclinic']:
            reciprocal_reindexed_unit_cell = reciprocal_uc_conversion(reindexed_unit_cell)
            if self.lattice_system == 'monoclinic':
                hkl_reindexer = np.zeros((reindexed_unit_cell.shape[0], 3, 3))
                reindexed_spacegroup_symbol_hm = list(augmented_entries['reindexed_spacegroup_symbol_hm'])
                split_group = [None for i in range(reindexed_unit_cell.shape[0])]
                group = list(augmented_entries['group'])
                for entry_index in range(reindexed_unit_cell.shape[0]):
                    reciprocal_reindexed_unit_cell[entry_index], _, hkl_reindexer[entry_index] = \
                        reindex_entry_monoclinic(
                            reciprocal_reindexed_unit_cell[entry_index],
                            spacegroup_symbol=reindexed_spacegroup_symbol_hm[entry_index],
                            space='reciprocal'
                            )
                    split = get_split_group(
                        'monoclinic',
                        reciprocal_reindexed_unit_cell=reciprocal_reindexed_unit_cell[entry_index],
                        )
                    split_group[entry_index] = group[entry_index].replace(f'_', f'_{split}_')
                augmented_entries['split_group'] = split_group
            # triclinic reindexing in reciprocal space is broken
            #elif self.lattice_system == 'triclinic':
            #    reciprocal_reindexed_unit_cell, hkl_reindexer = \
            #        reindex_entry_triclinic(reciprocal_reindexed_unit_cell, space='reciprocal')
            reindexed_unit_cell = reciprocal_uc_conversion(reciprocal_reindexed_unit_cell)
            augmented_entries['reindexed_unit_cell'] = list(reindexed_unit_cell)
            reindexed_hkl = np.stack(augmented_entries['reindexed_hkl'])
            for entry_index in range(reindexed_unit_cell.shape[0]):
                reindexed_hkl[entry_index] = reindexed_hkl[entry_index] @ hkl_reindexer[entry_index]
            augmented_entries['reindexed_hkl'] = list(reindexed_hkl)

        augmented_entries['reciprocal_reindexed_unit_cell'] = list(reciprocal_uc_conversion(reindexed_unit_cell))
        reindexed_xnn = get_xnn_from_unit_cell(reindexed_unit_cell, partial_unit_cell=False)
        augmented_entries['reindexed_xnn'] = list(reindexed_xnn)
        augmented_entries['reindexed_volume'] = list(get_unit_cell_volume(
            reindexed_unit_cell, partial_unit_cell=False
            ))
        return augmented_entries

    def augment_entry(self, entry):
        augmented_entry = copy.deepcopy(entry)
        augmented_entry['augmented'] = True
        reindexed_unit_cell = np.array(augmented_entry['reindexed_unit_cell'])
        reindexed_volume = np.array(augmented_entry['reindexed_volume'])
        perturbed_reindexed_unit_cell = self.perturb_unit_cell_common(
            reindexed_unit_cell,
            augmented_entry['split_group'],
            reindexed_volume,
            )
        augmented_entry['reindexed_unit_cell'] = perturbed_reindexed_unit_cell

        # calculate new d-spacings
        reindexed_hkl_sa = np.stack(augmented_entry['reindexed_hkl_sa']).round(decimals=0).astype(int)

        q2_calculator = Q2Calculator(
            self.lattice_system,
            reindexed_hkl_sa,
            tensorflow=False,
            representation='unit_cell'
            )
        q2_sa = q2_calculator.get_q2(perturbed_reindexed_unit_cell[self.unit_cell_indices][np.newaxis, :])[0]
        existing_peaks = np.any(reindexed_hkl_sa != 0, axis=1)

        q2_sa = q2_sa[existing_peaks]
        reindexed_hkl_sa = reindexed_hkl_sa[existing_peaks]
        order = np.argsort(q2_sa)
        q2_sa = q2_sa[order]
        reindexed_hkl_sa = reindexed_hkl_sa[order]

        augmented_entry['d_spacing_sa'] = 1 / np.sqrt(q2_sa)
        augmented_entry['q2_sa'] = q2_sa

        # choose new peaks
        if self.broadening_tag == 'sa':
            q2 = q2_sa.copy()
            reindexed_hkl = reindexed_hkl_sa.copy()
        else:
            first_peak_index = np.searchsorted(self.first_probability['cdf'], self.rng.random())
            if first_peak_index >= q2_sa.size:
                return None
            q2 = [q2_sa[first_peak_index]]
            reindexed_hkl = [reindexed_hkl_sa[first_peak_index]]

            previous_kept_index = first_peak_index

            peak_generation_info = get_peak_generation_info()
            broadening_params = peak_generation_info['broadening_params']
            broadening_multiplier = peak_generation_info['broadening_multiples'][
                peak_generation_info['broadening_tags'].index(self.broadening_tag)
                ]

            for index in range(first_peak_index + 1, q2_sa.size):
                # cases:
                # 1) Close to previous_kept: Reject
                # 2) Far from previous_kept and next: Use formula
                # 3) Far from previous_kept, close to next: Accept with 1 x formula probability
                #    Originally this was 0.5, but I changed it to 0.5 and it doesn't seem to matter
                #    There is a problem with the last 5 q2_obs values being distributed to higher 
                #    values in the augmented data that I can't seem to resolve.

                # There is a problem with really large q2 values. Like ~100 I believe they are comming from
                # setting the hkl to [-100, -100, -100] for empty peaks in the peak list
                if q2_sa[index] < 1:
                    peak_breadth_std = broadening_multiplier * (broadening_params[0] + q2_sa[index]*broadening_params[1])
                    # STD / FWHM conversion
                    # 2.35 = 2*np.sqrt(2*np.log(2))
                    overlap_threshold = peak_breadth_std * 2*np.sqrt(2*np.log(2)) / 1.5 # over rejects
                    #overlap_threshold = peak_breadth_std / 2 # over rejects
                    #overlap_threshold = 0 # over rejects
                    distance_previous = q2_sa[index] - q2_sa[previous_kept_index]
                    # Case 1 will not pass beyond this
                    if distance_previous > overlap_threshold: 
                        if index == q2_sa.size - 1:
                            separation = distance_previous
                            distance_next = distance_previous
                        else:
                            distance_next = q2_sa[index + 1] - q2_sa[index]
                            separation = min(distance_previous, distance_next)
                        
                        # If the query values are out of the interpolation ranges the output is np.nan
                        # Catch these cases and use the bounds
                        check_0 = self.difference_centers[0] < separation < self.difference_centers[-1]
                        check_1 = self.q2_centers[0] < q2_sa[index] < self.q2_centers[-1]
                        if check_0 and check_1:
                            keep_prob = self.interpolator((separation, q2_sa[index]))
                        elif separation < self.difference_centers[0] and q2_sa[index] < self.q2_centers[0]:
                            keep_prob = self.interpolator((self.difference_centers[0], self.q2_centers[0]))
                        elif separation < self.difference_centers[0] and q2_sa[index] > self.q2_centers[-1]:
                            keep_prob = self.interpolator((self.difference_centers[0], self.q2_centers[-1]))
                        elif separation > self.difference_centers[-1] and q2_sa[index] < self.q2_centers[0]:
                            keep_prob = self.interpolator((self.difference_centers[-1], self.q2_centers[0]))
                        elif separation > self.difference_centers[-1] and q2_sa[index] > self.q2_centers[-1]:
                            keep_prob = self.interpolator((self.difference_centers[-1], self.q2_centers[-1]))
                        elif separation < self.difference_centers[0]:
                            keep_prob = self.interpolator((self.difference_centers[0], q2_sa[index]))
                        elif separation > self.difference_centers[-1]:
                            keep_prob = self.interpolator((self.difference_centers[-1], q2_sa[index]))
                        elif q2_sa[index] < self.q2_centers[0]:
                            keep_prob = self.interpolator((separation, self.q2_centers[0]))
                        elif q2_sa[index] > self.q2_centers[-1]:
                            keep_prob = self.interpolator((separation, self.q2_centers[-1]))

                        if distance_next > overlap_threshold:
                            multiplier = 1
                        else:
                            multiplier = 1
                        if self.rng.random() < multiplier*keep_prob:
                            q2.append(q2_sa[index])
                            reindexed_hkl.append(reindexed_hkl_sa[index])
                            previous_kept_index = index

        if len(q2) >= self.n_peaks:
            # This sort might be unneccessary, but not harmful.
            q2 = np.array(q2)
            check = np.sum((q2[1:] - q2[:-1]) < 0)
            if check > 0:
                print('q2 is not sorted. This is a bug')
            sort_indices = np.argsort(q2)
            q2 = q2[sort_indices][:self.n_peaks]

            reindexed_hkl = np.array(reindexed_hkl, dtype=int)[sort_indices][:self.n_peaks]
            augmented_entry[f'q2_{self.broadening_tag}'] = q2
            augmented_entry['q2'] = q2[:self.n_peaks]

            augmented_entry[f'd_spacing_{self.broadening_tag}'] = 1 / np.sqrt(q2)
            augmented_entry[f'd_spacing'] = 1 / np.sqrt(q2[:self.n_peaks])
            augmented_entry['reindexed_hkl'] = reindexed_hkl
            return augmented_entry
        else:
            return None

    def _check_in_range(self, perturbed_unit_cell):
        if self.lattice_system == 'monoclinic':
            minimum_angle = np.pi/2
            maximum_angle = np.pi
            good_angle = False
            good_lengths = False
            if np.all(perturbed_unit_cell[:3] > self.min_unit_cell):
                if np.all(perturbed_unit_cell[:3] < self.max_unit_cell):
                    good_lengths = True
            if minimum_angle  < perturbed_unit_cell[3] < maximum_angle:
                good_angle = True
            if good_angle & good_lengths:
                return True
        elif self.lattice_system == 'triclinic':
            maximum_angle = np.pi
            good_angles = False
            good_lengths = False
            if np.all(perturbed_unit_cell[:3] > self.min_unit_cell):
                if np.all(perturbed_unit_cell[:3] < self.max_unit_cell):
                    good_lengths = True
            if np.all(perturbed_unit_cell[3:] <= maximum_angle):
                good_angles = True
            if good_angles & good_lengths:
                return True
        elif self.lattice_system == 'rhombohedral':
            good_length = False
            good_angle = False
            reciprocalable = False
            if perturbed_unit_cell[0] > self.min_unit_cell:
                if perturbed_unit_cell[0] < self.max_unit_cell:
                    good_length = True
            if 0 < perturbed_unit_cell[1]:
                if perturbed_unit_cell[1] < 2*np.pi/3:
                    good_angle = True
            if good_length == False:
                return None
            if good_angle == False:
                return None

            reciprocal_unit_cell = reciprocal_uc_conversion(
                perturbed_unit_cell[np.newaxis],
                partial_unit_cell=True,
                lattice_system='rhombohedral',
                )[0]
            reciprocalable = np.invert(np.any(np.isnan(reciprocal_unit_cell)))
            if reciprocalable:
                return True
        else:
            if np.all(perturbed_unit_cell > self.min_unit_cell):
                if np.all(perturbed_unit_cell < self.max_unit_cell):
                    return True

    def _permute_perturbed_unit_cell(self, perturbed_unit_cell , unit_cell ):
        if self.lattice_system == 'monoclinic':
            initial_order = np.argsort(unit_cell[:3])
            initial_inverse_sort = np.argsort(initial_order)
            current_sort = np.argsort(perturbed_unit_cell[:3])
            perturbed_unit_cell[:3] = perturbed_unit_cell[:3][current_sort][initial_inverse_sort]
            current_order = np.argsort(perturbed_unit_cell[:3])
            if np.any(initial_order != current_order):
                print(unit_cell )
                print(perturbed_unit_cell )
                print()
        elif self.lattice_system in ['hexagonal', 'orthorhombic', 'tetragonal']:
            initial_order = np.argsort(unit_cell)
            initial_inverse_sort = np.argsort(initial_order)
            current_sort = np.argsort(perturbed_unit_cell)
            perturbed_unit_cell  = perturbed_unit_cell[current_sort][initial_inverse_sort]
            current_order = np.argsort(perturbed_unit_cell)
            if np.any(initial_order != current_order):
                print(unit_cell)
                print(perturbed_unit_cell)
                print()
        elif self.lattice_system == 'triclinic':
            perturbed_unit_cell, _ = reindex_entry_triclinic(perturbed_unit_cell, space='direct')
        return perturbed_unit_cell 

    def perturb_unit_cell_common(self, unit_cell, split_group, reindexed_volume):
        if self.aug_params['n_per_volume'] is None:
            volume_bin_index = None
        elif len(self.volume_bins[split_group]) == 2:
            # This catches the case were there is only one volume bin
            volume_bin_index = None
        else:
            volume_bin_index = np.searchsorted(self.volume_bins[split_group], reindexed_volume) - 1
            if volume_bin_index < 0:
                volume_bin_index = 0
            elif volume_bin_index >= len(self.volume_bins[split_group]) - 1:
                volume_bin_index = len(self.volume_bins[split_group]) - 2

        perturbed_unit_cell = unit_cell.copy()
        perturbed_unit_cell[self.unit_cell_indices] = self.perturb_unit_cell(
            unit_cell[self.unit_cell_indices],
            split_group,
            volume_bin_index
            )
        if self.lattice_system == 'cubic':
            perturbed_unit_cell[:3] = perturbed_unit_cell[0]
        elif self.lattice_system in ['tetragonal', 'hexagonal']:
            perturbed_unit_cell[1] = perturbed_unit_cell[0]
        elif self.lattice_system == 'rhombohedral':
            perturbed_unit_cell[:3] = perturbed_unit_cell[0]
            perturbed_unit_cell[3:] = perturbed_unit_cell[3]
        return perturbed_unit_cell

    def perturb_unit_cell_std(self, unit_cell, split_group, volume_bin_index):
        # perturb unit cell
        status = True
        i = 0
        unit_cell_scaled = (unit_cell - self.mean_unit_cell[0]) / self.std_unit_cell[0]
        while status:
            perturbed_unit_cell_scaled = self.rng.normal(
                loc=unit_cell_scaled,
                scale=self.augment_shift,
                )[0]
            perturbed_unit_cell = perturbed_unit_cell_scaled * self.std_unit_cell[0] + self.mean_unit_cell[0]
            i += 1
            if self._check_in_range(perturbed_unit_cell):
                status = False
        perturbed_unit_cell = self._permute_perturbed_unit_cell(
            perturbed_unit_cell, unit_cell
            )
        return perturbed_unit_cell

    def perturb_unit_cell_cov(self, unit_cell, split_group, volume_bin_index):
        # perturb unit cell
        status = True
        if volume_bin_index is None:
            cov = self.cov[split_group]
        else:
            cov = self.cov[split_group][volume_bin_index]
        unit_cell_scaled = (unit_cell - self.mean_unit_cell[0]) / self.std_unit_cell[0]
        while status:
            perturbed_unit_cell_scaled = self.rng.multivariate_normal(
                mean=unit_cell_scaled,
                cov=self.augment_shift**2 * cov,
                size=1
                )[0]
            perturbed_unit_cell = perturbed_unit_cell_scaled * self.std_unit_cell[0] + self.mean_unit_cell[0]
            if self._check_in_range(perturbed_unit_cell):
                status = False
        perturbed_unit_cell = self._permute_perturbed_unit_cell(perturbed_unit_cell, unit_cell)
        return perturbed_unit_cell

    def perturb_unit_cell_pca(self, unit_cell, split_group, volume_bin_index):
        # perturb unit cell
        status = True
        if volume_bin_index is None:
            pca = self.pca[split_group]
            stddev = self.stddev[split_group]
        else:
            pca = self.pca[split_group][volume_bin_index]
            stddev = self.stddev[split_group][volume_bin_index]
        unit_cell_scaled = (unit_cell - self.mean_unit_cell[0]) / self.std_unit_cell[0]
        unit_cell_scaled_transformed = pca.transform(unit_cell_scaled[np.newaxis, :])[0]
        while status:
            perturbed_unit_cell_scaled_transformed = self.rng.normal(
                loc=unit_cell_scaled_transformed,
                scale=self.augment_shift * stddev,
                )
            perturbed_unit_cell_scaled = pca.inverse_transform(
                perturbed_unit_cell_scaled_transformed[np.newaxis, :]
                )[0, :]
            perturbed_unit_cell = perturbed_unit_cell_scaled * self.std_unit_cell[0] + self.mean_unit_cell[0]
            perturbed_unit_cell = self._permute_perturbed_unit_cell(perturbed_unit_cell, unit_cell)
            if self._check_in_range(perturbed_unit_cell):
                status = False
        return perturbed_unit_cell

    def evaluate(self, data, split_group):
        train = data[data['train']]
        train_unaugmented = train[~train['augmented']]
        train_augmented = train[train['augmented']]
        # There are usually an order of magnitude more augmented datasets than unaugmented
        # Downsample the augmented data so the balance is reasonable.
        if len(train_augmented) > len(train_unaugmented):
            train_augmented = train_augmented.sample(n=len(train_unaugmented), replace=False)
            train = pd.concat((train_unaugmented, train_augmented), ignore_index=True)
        train_uc = np.stack(train['reindexed_unit_cell'])[:, self.unit_cell_indices]
        train_uc_volume = get_unit_cell_volume(
            train_uc, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        train_y = np.array(train['augmented'], dtype=int)
        train_q2_obs = np.stack(train['q2'])

        val = data[~data['train']]
        val_unaugmented = val[~val['augmented']]
        val_augmented = val[val['augmented']]
        if len(val_augmented) > len(val_unaugmented):
            val_augmented = val_augmented.sample(n=len(val_unaugmented), replace=False)
            val = pd.concat((val_unaugmented, val_augmented), ignore_index=True)
        val_uc = np.stack(val['reindexed_unit_cell'])[:, self.unit_cell_indices]
        val_uc_volume = get_unit_cell_volume(
            val_uc, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        val_y = np.array(val['augmented'], dtype=int)
        val_q2_obs = np.stack(val['q2'])

        # Concatenate features
        train_features = np.hstack((train_uc, train_q2_obs))
        val_features = np.hstack((val_uc, val_q2_obs))

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(
                class_weight='balanced_subsample',
                ),
            param_grid={
                'max_depth': [6],
                'min_samples_leaf': [4],
                'min_samples_split': [4],
                'n_estimators': [400],
                },
            cv=5,
            n_jobs=6,
            verbose=2,
            )
        grid_search.fit(X=train_features, y=train_y)
        classifier = grid_search.best_estimator_

        train_accuracy = classifier.score(X=train_features, y=train_y)
        val_accuracy = classifier.score(X=val_features, y=val_y)

        train_pred = classifier.predict(train_features)
        train_correct = train_pred == train_y

        val_pred = classifier.predict(val_features)
        val_correct = val_pred == val_y

        n_bins = 40
        bins = [None for _ in range(self.unit_cell_length + 1)]
        centers = [None for _ in range(self.unit_cell_length + 1)]
        train_true_frac = np.zeros((n_bins, self.unit_cell_length + 1))
        val_true_frac = np.zeros((n_bins, self.unit_cell_length + 1))

        for uc_index in range(self.unit_cell_length + 1):
            if uc_index == self.unit_cell_length:
                sorted_uc = np.sort(train_uc_volume)
            else:
                sorted_uc = np.sort(train_uc[:, uc_index])
            bins[uc_index] = np.linspace(sorted_uc[0], sorted_uc[int(0.99*sorted_uc.size)], n_bins + 1)
            centers[uc_index] = (bins[uc_index][1:] + bins[uc_index][:-1]) / 2
            if uc_index == self.unit_cell_length:
                train_bin_indices = np.searchsorted(bins[uc_index], train_uc_volume) - 1
                val_bin_indices = np.searchsorted(bins[uc_index], val_uc_volume) - 1
            else:
                train_bin_indices = np.searchsorted(bins[uc_index], train_uc[:, uc_index]) - 1
                val_bin_indices = np.searchsorted(bins[uc_index], val_uc[:, uc_index]) - 1
            for bin_index in range(n_bins):
                train_select = train_correct[train_bin_indices == bin_index]
                val_select = val_correct[val_bin_indices == bin_index]
                if train_select.size > 0:
                    train_true_frac[bin_index, uc_index] = train_select.sum() / train_select.size
                else:
                    train_true_frac[bin_index, uc_index] = np.nan
                if val_select.size > 0:
                    val_true_frac[bin_index, uc_index] = val_select.sum() / val_select.size
                else:
                    val_true_frac[bin_index, uc_index] = np.nan

        figsize = (5 + 1.5**(self.unit_cell_length + 1), 3)
        fig, axes = plt.subplots(1, self.unit_cell_length + 1, figsize=figsize)
        if self.lattice_system == 'cubic':
            labels = ['a', 'Volume']
        elif self.lattice_system in ['tetragonal', 'hexagonal']:
            labels = ['a', 'b', 'Volume']
        elif self.lattice_system == 'orthorhombic':
            labels = ['a', 'b', 'c', 'Volume']
        elif self.lattice_system == 'monoclinic':
            labels = ['a', 'b', 'c', 'beta', 'Volume']
        elif self.lattice_system == 'triclinic':
            labels = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volume']
        elif self.lattice_system == 'rhombohedral':
            labels = ['a', 'alpha', 'Volume']
        axes_r = [axes[i].twinx() for i in range(self.unit_cell_length + 1)]
        for uc_index in range(self.unit_cell_length + 1):
            if uc_index == self.unit_cell_length:
                unaugmented_uc = train_uc_volume[~train['augmented']]
                augmented_uc = train_uc_volume[train['augmented']]
            else:
                unaugmented_uc = train_uc[~train['augmented'], uc_index]
                augmented_uc = train_uc[train['augmented'], uc_index]
            axes[uc_index].hist(unaugmented_uc, bins=bins[uc_index], label='Original', density=True)
            axes[uc_index].hist(augmented_uc, bins=bins[uc_index], label='Augmented', density=True, alpha=0.5)
            axes_r[uc_index].plot(centers[uc_index], train_true_frac[:, uc_index], color=[1, 0, 0], label='Train Accuracy')
            axes_r[uc_index].plot(centers[uc_index], val_true_frac[:, uc_index], color=[0, 1, 0], label='Val Accuracy')
            axes_r[uc_index].plot(centers[uc_index], 0.5*np.ones(centers[uc_index].size), color=[0, 0, 0], linestyle='dotted')
            axes[uc_index].set_xlabel(labels[uc_index])
        axes[0].set_ylabel('Distribution')
        axes_r[self.unit_cell_length].set_ylabel('Prediction Accuracy')
        axes[0].legend(frameon=False, framealpha=0.5)
        axes_r[self.unit_cell_length].legend(frameon=False, framealpha=0.5)
        axes[0].set_title('\n'.join([
            f'Train / Val Acc: {train_accuracy:0.2f} / {val_accuracy:0.2f}',
            ]))
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to}',
            f'aug_predictions_{split_group}_{self.aug_params["tag"]}.png'
            ))
        plt.close()

        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.plot(
            classifier.feature_importances_[:self.unit_cell_length],
            marker='.', label='Unit Cell'
            )
        axes.plot(
            classifier.feature_importances_[self.unit_cell_length:],
            marker='.', label='q2 obs'
            )
        ylim = axes.get_ylim()
        axes.plot(
            [self.unit_cell_length + 0.5, self.unit_cell_length + 0.5], ylim,
            color=[0, 0, 0], linestyle='dotted'
            )
        axes.set_ylim(ylim)
        axes.legend(frameon=False)
        axes.set_xlabel('Feature Importance')
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to}',
            f'aug_feature_importance_{split_group}_{self.aug_params["tag"]}.png'
            ))
        plt.close()

        n_important = 5
        feature_indices = np.argsort(classifier.feature_importances_[self.unit_cell_length:])[::-1][:n_important]
        sorted_q2 = np.sort(train_q2_obs.ravel())
        q2_bins = np.linspace(0, sorted_q2[int(0.9975*sorted_q2.size)], 101)
        unaugmented_q2 = train_q2_obs[~train['augmented']]
        augmented_q2 = train_q2_obs[train['augmented']]
        fig, axes = plt.subplots(1, n_important + 1, figsize=(10, 4), sharex=True)
        axes[0].hist(unaugmented_q2.ravel(), bins=q2_bins, density=True, label='Original')
        axes[0].hist(augmented_q2.ravel(), bins=q2_bins, density=True, alpha=0.5, label='Augmented')
        axes[0].set_title('All Peaks')
        for axes_index, feature_index in enumerate(feature_indices):
            axes[axes_index + 1].hist(unaugmented_q2[:, feature_index], bins=q2_bins, density=True, label='Original')
            axes[axes_index + 1].hist(augmented_q2[:, feature_index], bins=q2_bins, density=True, alpha=0.5, label='Augmented')
            axes[axes_index + 1].set_title(f'q2 index: {feature_index}')
        for axes_index in range(n_important + 1):
            axes[axes_index].set_xlabel('q2_obs')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            f'{self.save_to}',
            f'q2_distributions_{split_group}_{self.aug_params["tag"]}.png'
            ))
        plt.close()
