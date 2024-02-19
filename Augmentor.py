import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
from sklearn.decomposition import PCA
from tqdm import tqdm

from Utilities import Q2Calculator


class Augmentor:
    def __init__(
            self,
            aug_params,
            data_params,
            min_unit_cell_scaled,
            n_generated_points,
            save_to,
            seed,
            uc_scaler,
            unit_cell_key,
            hkl_key
            ):
        self.aug_params = aug_params
        self.unit_cell_key = unit_cell_key
        self.hkl_key = hkl_key
        self.n_generated_points = n_generated_points
        self.min_unit_cell_scaled = min_unit_cell_scaled
        self.save_to = save_to
        self.n_points = data_params['n_points']
        self.y_indices = data_params['y_indices']
        self.n_outputs = data_params['n_outputs']
        self.points_tag = data_params['points_tag']
        self.lattice_system = data_params['lattice_system']
        self.n_max = data_params['n_max']
        self.hkl_ref_length = data_params['hkl_ref_length']
        self.rng = np.random.default_rng(seed)
        self.uc_scaler = uc_scaler
        if data_params['unit_cell_representation'] == 'permuted':
            self.permute_unit_cell = True
        else:
            self.permute_unit_cell = False

    def setup(self, data):
        print(f'\n Setting up augmentation {self.aug_params["tag"]}')
        # Calculate the number of times each entry is to be augmented
        training_data = data[data['train']]
        unit_cell_scaled = np.stack(training_data[f'{self.unit_cell_key}_scaled'])[:, self.y_indices]
        if self.aug_params['augment_method'] == 'random':
            self.perturb_unit_cell = self.perturb_unit_cell_std
            self.stddev = unit_cell_scaled.std(axis=0)
        elif self.aug_params['augment_method'] == 'cov':
            self.perturb_unit_cell = self.perturb_unit_cell_cov
            self.cov = np.cov(unit_cell_scaled.T)
        elif self.aug_params['augment_method'] == 'pca':
            self.perturb_unit_cell = self.perturb_unit_cell_pca
            self.pca = PCA(n_components=self.n_outputs).fit(unit_cell_scaled)
            unit_cell_scaled_transformed = self.pca.transform(unit_cell_scaled)
            self.std = np.std(unit_cell_scaled_transformed, axis=0)

        # calculate the order of the peak in the list of sa peaks
        n_bins = 100
        difference_bins = np.logspace(-4, -1.75, n_bins + 1)
        difference_centers = (difference_bins[1:] + difference_bins[:-1]) / 2
        keep_sum = np.zeros(n_bins)
        drop_sum = np.zeros(n_bins)
        first_peak = np.zeros(len(training_data))
        differences = []
        for entry_index in range(len(training_data)):
            q2_sa = np.array(training_data.iloc[entry_index]['q2_sa'])
            q2 = np.array(training_data.iloc[entry_index][f'q2_{self.points_tag}'])
            q2_sa = q2_sa[~np.isinf(q2_sa)]
            q2 = q2[~np.isinf(q2)]
            first_peak_index = np.argwhere(q2_sa == q2[0])
            if len(first_peak_index) > 0:
                # this catches two cases in oP that are problematic entries
                first_peak[entry_index] = first_peak_index[0][0]
                difference = q2_sa[1:] - q2_sa[:-1]
                differences.append(difference)
                for peak_index in range(q2.size - 1):
                    insert_index = np.searchsorted(difference_bins, difference[peak_index])
                    if n_bins > insert_index >= 0:
                        if q2[peak_index] in q2_sa:
                            if q2[peak_index + 1] in q2_sa:
                                keep_sum[insert_index] += 1
                            else:
                                drop_sum[insert_index] += 1            
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

        self.keep_rate = lambda x, r0, r1: (1 - np.exp(-r0 * x))**r1
        y = keep_sum / total_sum
        self.keep_rate_params, pcov = scipy.optimize.curve_fit(
            f=self.keep_rate,
            xdata=difference_centers[~np.isnan(y)],
            ydata=y[~np.isnan(y)],
            p0=(10, 0.001),
            )

        difference_hist, _ = np.histogram(np.concatenate(differences), bins=difference_bins, density=True)
        fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharex='col')
        axes[0, 0].plot(self.first_probability['x'], self.first_probability['pdf'], marker='.', linestyle='none')
        axes[1, 0].plot(self.first_probability['x'], self.first_probability['cdf'], marker='.', linestyle='none')
        axes[0, 0].set_title(f'Distribution of first peak')
        axes[0, 0].set_ylabel('PDF')
        axes[1, 0].set_ylabel('CDF')
        axes[1, 0].set_xlabel('Peak index')
        
        axes[0, 1].plot(difference_centers, keep_sum / total_sum)
        axes[0, 1].plot(difference_centers, self.keep_rate(difference_centers, *self.keep_rate_params))
        axes[1, 1].bar(difference_centers, difference_hist, width=difference_bins[1:] - difference_bins[:-1])
        axes[0, 1].set_ylabel('Keep Rate')
        axes[1, 1].set_ylabel('Distribution')
        axes[1, 1].set_xlabel('$q^2$ spacing')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/aug_setup_{self.aug_params["tag"]}.png')
        plt.close()

    def augment(self, data):
        groups = data['group'].unique()
        n_augment = dict.fromkeys(groups)
        for group in groups:
            N = np.sum(data['group'] == group)
            n_augment_all = self.n_max - N
            n_augment_times = n_augment_all // N
            if n_augment_times >= self.aug_params['max_augmentation']:
                n_augment[group] = self.aug_params['max_augmentation'] * np.ones(N, dtype=int)
            else:
                n_augment[group] = n_augment_times * np.ones(N, dtype=int)
                n_augment_remainder = n_augment_all % N
                indices = self.rng.choice(N, size=n_augment_remainder, replace=False)
                n_augment[group][indices] += 1

        augmented_entries = []
        for group in groups:
            group_data = data[data['group'] == group]
            for entry_index in range(len(group_data)):
                entry = group_data.iloc[entry_index]
                for augment_index in range(n_augment[group][entry_index]):
                    augmented_entry = None
                    while augmented_entry is None:
                        augmented_entry = self.augment_entry(entry)
                    augmented_entries.append(augmented_entry)
        return pd.DataFrame(augmented_entries)

    def augment_entry(self, entry):
        augmented_entry = copy.deepcopy(entry)
        augmented_entry['augmented'] = True
        unit_cell_scaled = np.array(augmented_entry[f'{self.unit_cell_key}_scaled'])
        perturbed_unit_cell_scaled = unit_cell_scaled.copy()
        perturbed_unit_cell_scaled[self.y_indices] = self.perturb_unit_cell(unit_cell_scaled[self.y_indices])
        perturbed_unit_cell = np.zeros(6)
        perturbed_unit_cell[:3] = perturbed_unit_cell_scaled[:3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
        perturbed_unit_cell[3:] = perturbed_unit_cell_scaled[3:] + np.pi/2
        augmented_entry[f'{self.unit_cell_key}_scaled'] = perturbed_unit_cell_scaled
        augmented_entry[f'{self.unit_cell_key}'] = perturbed_unit_cell
        # calculate new d-spacings
        hkl_sa = augmented_entry[f'{self.hkl_key}_sa']
        q2_calculator = Q2Calculator(self.lattice_system, hkl_sa, tensorflow=False)
        q2_sa = q2_calculator.get_q2(perturbed_unit_cell[np.newaxis, :])[0]

        existing_peaks = np.any(hkl_sa != 0, axis=1)
        q2_sa = np.sort(q2_sa[existing_peaks])
        augmented_entry['d_spacing_sa'] = 1 / np.sqrt(q2_sa)
        augmented_entry['q2_sa'] = q2_sa

        # choose new peaks
        first_peak_index = self.first_probability['x'][
            np.searchsorted(self.first_probability['cdf'], self.rng.random())
            ]
        q2 = [q2_sa[first_peak_index]]
        hkl = [augmented_entry[f'{self.hkl_key}_sa'][first_peak_index]]
        for index in range(first_peak_index + 1, q2_sa.size):
            if self.rng.random() < self.keep_rate(q2_sa[index] - q2_sa[index - 1], *self.keep_rate_params):
                q2.append(q2_sa[index])
                hkl.append(augmented_entry[f'{self.hkl_key}_sa'][index])

        if len(q2) >= self.n_points:
            q2 = np.array(q2)[:self.n_points]
            sort_indices = np.argsort(q2)
            q2 = q2[sort_indices]
            hkl = np.array(hkl)[:self.n_points][sort_indices]
            augmented_entry[f'q2_{self.points_tag}'] = q2
            augmented_entry[f'd_spacing_{self.points_tag}'] = 1 / np.sqrt(q2)
            augmented_entry[f'{self.hkl_key}'] = hkl
            return augmented_entry
        else:
            return None

    def perturb_unit_cell_std(self, unit_cell_scaled):
        # perturb unit cell
        status = True
        while status:
            perturbed_unit_cell_scaled = self.rng.normal(
                loc=unit_cell_scaled,
                scale=self.aug_params['augment_shift'] * self.stddev
                )[0]
            if np.all(perturbed_unit_cell_scaled > self.min_unit_cell_scaled):
                status = False
        if self.permute_unit_cell:
            perturbed_unit_cell_scaled = np.sort(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled

    def perturb_unit_cell_cov(self, unit_cell_scaled):
        # perturb unit cell
        status = True
        while status:
            perturbed_unit_cell_scaled = self.rng.multivariate_normal(
                mean=unit_cell_scaled,
                cov=self.aug_params['augment_shift']**2 * self.cov,
                size=1
                )[0]
            if np.all(perturbed_unit_cell_scaled > self.min_unit_cell_scaled):
                status = False
        if self.permute_unit_cell:
            perturbed_unit_cell_scaled = np.sort(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled

    def perturb_unit_cell_pca(self, unit_cell_scaled):
        # perturb unit cell
        status = True
        unit_cell_scaled_transformed = self.pca.transform(unit_cell_scaled[np.newaxis, :])[0]
        while status:
            perturbed_unit_cell_scaled_transformed = self.rng.normal(
                loc=unit_cell_scaled_transformed,
                scale=self.aug_params['augment_shift'] * self.std,
                )
            perturbed_unit_cell_scaled = self.pca.inverse_transform(
                perturbed_unit_cell_scaled_transformed[np.newaxis, :]
                )[0, :]
            if np.all(perturbed_unit_cell_scaled > self.min_unit_cell_scaled):
                status = False
        if self.permute_unit_cell:
            perturbed_unit_cell_scaled = np.sort(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled
