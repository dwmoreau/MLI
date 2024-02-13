import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
from sklearn.decomposition import PCA
from tqdm import tqdm


class Augmentor:
    def __init__(
            self,
            aug_params,
            bravais_lattice,
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
        self.bravais_lattice = bravais_lattice
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
        if data_params['unit_cell_representation'] == 'reordered':
            self.reorder_unit_cell = True
        else:
            self.reorder_unit_cell = False

    def setup(self, data):
        print(f'\n Setting up augmentation for {self.bravais_lattice}')
        # Calculate the number of times each entry is to be augmented
        augmented_entries = []
        training_data = data[data['train']]
        N = len(data)
        n_augment_all = self.n_max - N
        n_augment_times = n_augment_all // N
        if n_augment_times >= self.aug_params['max_augmentation']:
            self.n_augment = self.aug_params['max_augmentation'] * np.ones(N, dtype=int)
        else:
            self.n_augment = n_augment_times * np.ones(N, dtype=int)
            n_augment_remainder = n_augment_all % N
            indices = self.rng.choice(N, size=n_augment_remainder, replace=False)
            self.n_augment[indices] += 1

        unit_cell_scaled = np.stack(
            training_data[f'{self.unit_cell_key}_scaled']
            )[:, self.y_indices]
        if self.aug_params['augment_method'] == 'cov':
            self.cov = np.cov(unit_cell_scaled.T)
        elif self.aug_params['augment_method'] == 'pca':
            self.pca = PCA(n_components=self.n_outputs).fit(unit_cell_scaled)
            unit_cell_scaled_transformed = self.pca.transform(unit_cell_scaled)
            self.std = np.std(unit_cell_scaled_transformed, axis=0)

        # calculate the order of the peak in the list of sa peaks
        order = np.zeros((N, self.n_generated_points), dtype=int)
        n_bins = 100
        difference_bins = np.logspace(-4, -1.75, n_bins + 1)
        difference_centers = (difference_bins[1:] + difference_bins[:-1]) / 2
        keep_sum = np.zeros(n_bins)
        drop_sum = np.zeros(n_bins)
        first_peak = np.zeros(N)
        differences = []
        for entry_index in range(N):
            q2_sa = np.array(data.iloc[entry_index][f'q2_sa'])            
            q2 = np.array(
                data.iloc[entry_index][f'q2_{self.points_tag}']
                )
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
                    if insert_index < n_bins and insert_index >= 0:
                        if q2[peak_index] in q2_sa:
                            if q2[peak_index + 1] in q2_sa:
                                keep_sum[insert_index] += 1
                            else:
                                drop_sum[insert_index] += 1            
        total_sum = keep_sum + drop_sum

        centers = np.arange(self.n_generated_points)
        bins = np.append(centers - 0.5, self.n_generated_points + 0.5)
        width = bins[1] - bins[0]
        hist, _ = np.histogram(first_peak, bins=bins, density=True)
        indices = hist > 0
        pdf = hist[indices]
        cdf = np.cumsum(pdf)
        self.first_probability = {
            'x': centers[indices],
            'pdf': pdf,
            'cdf': cdf
            }

        self.keep_rate = lambda x, r0, r1 : (1 - np.exp(-r0 * x))**r1
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
        axes[0, 0].set_title(f'Distribution of first peak {self.bravais_lattice}')
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
        fig.savefig(f'{self.save_to}/{self.bravais_lattice}_aug_setup_{self.aug_params["tag"]}.png')
        plt.close()

    def augment(self, data):
        augmented_entries = []
        for entry_index in tqdm(range(len(data))):
            for augment_index in range(self.n_augment[entry_index]):
                augmented_entry = self.augment_entry(data.iloc[entry_index])
                if not augmented_entry is None:
                    augmented_entries.append(augmented_entry)
        return pd.DataFrame(augmented_entries)

    def augment_entry(self, entry):
        augmented_entry = copy.deepcopy(entry)
        augmented_entry['augmented'] = True
        unit_cell_scaled = np.array(augmented_entry[f'{self.unit_cell_key}_scaled'])
        perturbed_unit_cell_scaled = unit_cell_scaled.copy()
        if self.aug_params['augment_method'] == 'cov':
            perturbed_unit_cell_scaled[self.y_indices] = self.perturb_unit_cell_cov(
                unit_cell_scaled[self.y_indices]
                )
        elif self.aug_params['augment_method'] == 'pca':
            perturbed_unit_cell_scaled[self.y_indices] = self.perturb_unit_cell_pca(
                unit_cell_scaled[self.y_indices]
                )
        perturbed_unit_cell = np.zeros(6)
        perturbed_unit_cell[:3] = perturbed_unit_cell_scaled[:3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
        perturbed_unit_cell[3:] = perturbed_unit_cell_scaled[3:] + np.pi/2
        augmented_entry[f'{self.unit_cell_key}_scaled'] = perturbed_unit_cell_scaled
        augmented_entry[f'{self.unit_cell_key}'] = perturbed_unit_cell
        # calculate new d-spacings
        hkl_sa = augmented_entry[f'{self.hkl_key}_sa']
        if self.lattice_system == 'cubic':
            q2_sa = (hkl_sa[:, 0]**2 + hkl_sa[:, 1]**2 + hkl_sa[:, 2]**2) / perturbed_unit_cell[0]**2
        elif self.lattice_system == 'tetragonal':
            q2_sa = (hkl_sa[:, 0]**2 + hkl_sa[:, 1]**2) / perturbed_unit_cell[0]**2 \
                   + hkl_sa[:, 2]**2 / unit_cell[2]**2
        elif self.lattice_system == 'orthorhombic':
            q2_sa = hkl_sa[:, 0]**2 / perturbed_unit_cell[0]**2 \
                  + hkl_sa[:, 1]**2 / perturbed_unit_cell[1]**2 \
                  + hkl_sa[:, 2]**2 / perturbed_unit_cell[2]**2
        elif self.lattice_system == 'hexagonal':
            assert False
        elif self.lattice_system == 'rhombohedral':
            assert False
        elif self.lattice_system == 'monoclinic':
            assert False
        elif self.lattice_system == 'triclinic':
            assert False
        else:
            assert False
        existing_peaks =  np.all(hkl_sa != 0, axis=1)
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

    def perturb_unit_cell_cov(self, unit_cell_scaled):
        # perturb unit cell
        status = True
        while status:
            perturbed_unit_cell_scaled = self.rng.multivariate_normal(
                mean=unit_cell_scaled,
                cov=self.aug_params['augment_shift'] * self.cov,
                size=1
                )[0]
            if np.all(perturbed_unit_cell_scaled > self.min_unit_cell_scaled):
                status = False
        if self.reorder_unit_cell:
            perturbed_unit_cell_scaled = np.sort(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled

    def perturb_unit_cell_pca(self, unit_cell_scaled):
        # perturb unit cell
        status = True
        unit_cell_scaled_transformed = self.pca.transform(unit_cell_scaled[np.newaxis, :])[0]
        while status:
            perturbed_unit_cell_scaled_transformed = self.rng.normal(
                loc=unit_cell_scaled_transformed,
                scale=np.sqrt(self.aug_params['augment_shift']) * self.std,
                )
            perturbed_unit_cell_scaled = self.pca.inverse_transform(
                perturbed_unit_cell_scaled_transformed[np.newaxis, :]
                )[0, :]
            if np.all(perturbed_unit_cell_scaled > self.min_unit_cell_scaled):
                status = False
        if self.reorder_unit_cell:
            perturbed_unit_cell_scaled = np.sort(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled
