import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

from Reindexing import get_permutation
from Reindexing import unpermute_monoclinic_full_unit_cell
from Utilities import Q2Calculator
from Utilities import get_fwhm_and_overlap_threshold


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
            angle_scale,
            ):
        self.aug_params = aug_params
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
        self.angle_scale = angle_scale

    def setup(self, data):
        print(f'\n Setting up augmentation {self.aug_params["tag"]}')
        # Calculate the number of times each entry is to be augmented
        training_data = data[data['train']]
        unit_cell_scaled = np.stack(training_data['reindexed_unit_cell_scaled'])[:, self.y_indices]
        if self.aug_params['augment_method'] == 'random':
            self.perturb_unit_cell = self.perturb_unit_cell_std

        elif self.aug_params['augment_method'] == 'cov':
            self.perturb_unit_cell = self.perturb_unit_cell_cov
            self.cov = np.cov(unit_cell_scaled.T)
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
            cov_display = ConfusionMatrixDisplay(confusion_matrix=self.cov)
            cov_display.plot(ax=axes, colorbar=False, values_format='0.2f')
            axes.set_title('Unit cell covariance')
            axes.set_xlabel('')
            axes.set_ylabel('')
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/aug_unit_cell_cov_{self.aug_params["tag"]}.png')
            plt.close()

        elif self.aug_params['augment_method'] == 'pca':
            self.perturb_unit_cell = self.perturb_unit_cell_pca
            self.pca = PCA(n_components=self.n_outputs).fit(unit_cell_scaled)
            unit_cell_scaled_transformed = self.pca.transform(unit_cell_scaled)
            self.stddev = np.std(unit_cell_scaled_transformed, axis=0)
            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            pca_display = ConfusionMatrixDisplay(confusion_matrix=self.pca.components_)
            pca_display.plot(ax=axes[0], colorbar=False, values_format='0.2f')
            axes[1].plot(self.pca.singular_values_, marker='.')
            axes[0].set_title('PCA Components')
            axes[0].set_xlabel('')
            axes[0].set_ylabel('')
            axes[1].set_title('PCA Singular values')
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/aug_unit_cell_pca_{self.aug_params["tag"]}.png')
            plt.close()

        # calculate the order of the peak in the list of sa peaks
        n_bins = 50
        difference_bins = np.logspace(-4, -2, n_bins + 1)
        difference_centers = (difference_bins[1:] + difference_bins[:-1]) / 2

        keep_sum = np.zeros(n_bins)
        drop_sum = np.zeros(n_bins)
        first_peak = np.zeros(len(training_data))
        differences = []
        for entry_index in range(len(training_data)):
            p = False
            q2_sa = np.array(training_data.iloc[entry_index]['q2_sa'])
            q2 = np.array(training_data.iloc[entry_index][f'q2_{self.points_tag}'])
            q2_sa = q2_sa[~np.isinf(q2_sa)]
            q2 = q2[~np.isinf(q2)]
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
                    if n_bins > insert_index >= 0:
                        if q2_sa[peak_index] in q2:
                            if q2_sa[peak_index + 1] in q2:
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

        self.keep_rate = lambda x, r0, r1, c: c*(1 - np.exp(-r0 * x))**r1
        y = keep_sum / total_sum
        self.keep_rate_params, _ = scipy.optimize.curve_fit(
            f=self.keep_rate,
            xdata=difference_centers[~np.isnan(y)],
            ydata=y[~np.isnan(y)],
            p0=(10, 0.001, 1),
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

    def augment(self, data, subgroup_label):
        sub_groups = data[subgroup_label].unique()
        n_subgroups = len(sub_groups)
        n_target_entries = self.n_max // n_subgroups
        n_augment = dict.fromkeys(sub_groups)
        for sub_group in sub_groups:
            n_subgroup_entries = np.sum(data[subgroup_label] == sub_group)
            n_augment_all = n_target_entries - n_subgroup_entries
            if n_augment_all > 0:
                n_augment_times = n_augment_all // n_subgroup_entries
                if n_augment_times >= self.aug_params['max_augmentation']:
                    n_augment[sub_group] = self.aug_params['max_augmentation'] * np.ones(n_subgroup_entries, dtype=int)
                else:
                    n_augment[sub_group] = n_augment_times * np.ones(n_subgroup_entries, dtype=int)
                    n_augment_remainder = n_augment_all % n_subgroup_entries
                    indices = self.rng.choice(n_subgroup_entries, size=n_augment_remainder, replace=False)
                    n_augment[sub_group][indices] += 1
            else:
                n_augment[sub_group] = np.zeros(n_subgroup_entries, dtype=int)

        augmented_entries = []
        for sub_group in sub_groups:
            sub_group_data = data[data[subgroup_label] == sub_group]
            for entry_index in range(len(sub_group_data)):
                entry = sub_group_data.iloc[entry_index]
                for augment_index in range(n_augment[sub_group][entry_index]):
                    augmented_entry = None
                    attempts = 0
                    while augmented_entry is None:
                        augmented_entry = self.augment_entry(entry)
                        attempts += 1
                        if attempts == 10:
                            break
                    if augmented_entry is not None:
                        augmented_entries.append(augmented_entry)
        return pd.DataFrame(augmented_entries)

    def augment_entry(self, entry):
        augmented_entry = copy.deepcopy(entry)
        augmented_entry['augmented'] = True
        reindexed_unit_cell_scaled = np.array(augmented_entry['reindexed_unit_cell_scaled'])
        perturbed_reindexed_unit_cell_scaled = reindexed_unit_cell_scaled.copy()
        perturbed_reindexed_unit_cell_scaled[self.y_indices] = self.perturb_unit_cell(reindexed_unit_cell_scaled[self.y_indices])
        perturbed_reindexed_unit_cell = np.zeros(6)
        perturbed_reindexed_unit_cell[:3] = perturbed_reindexed_unit_cell_scaled[:3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
        perturbed_reindexed_unit_cell[3:] = self.angle_scale * perturbed_reindexed_unit_cell_scaled[3:] + np.pi/2
        augmented_entry['reindexed_unit_cell_scaled'] = perturbed_reindexed_unit_cell_scaled
        augmented_entry['reindexed_unit_cell'] = perturbed_reindexed_unit_cell
        if self.lattice_system == 'monoclinic':
            # The perturbed unit cell is based on 'reindexed_unit_cell'
            # The 'reindexed_unit_cell' has alpha = gamma = np.nan as a convenience, even though the true reindexed
            # monoclinic angle might be alpha or gamma.
            # Perturbation does not change the monoclinic angle.
            # 'unit_cell' requires no nan's and always has the monoclinic angle at beta

            # This sets up a perturbed_reindexed_unit_cell with the monoclinic angle at the correct location
            perturbed_reindexed_unit_cell_ = np.zeros(6)
            perturbed_reindexed_unit_cell_[:3] = perturbed_reindexed_unit_cell[:3]
            perturbed_reindexed_unit_cell_[3:] = np.pi/2
            perturbed_reindexed_unit_cell_[augmented_entry['reindexed_angle_index']] = perturbed_reindexed_unit_cell[4]
            # This unreindexes the perturbed_reindexed_unit_cell
            # Conversion to radians happens before augmentation
            permutation, _ = get_permutation(augmented_entry['unit_cell'])
            perturbed_unit_cell = unpermute_monoclinic_full_unit_cell(
                perturbed_reindexed_unit_cell_, permutation, radians=True
                )
            perturbed_unit_cell_scaled = np.zeros(6)
            perturbed_unit_cell_scaled[:3] = (perturbed_unit_cell[:3] - self.uc_scaler.mean_[0]) / self.uc_scaler.scale_[0]
            perturbed_unit_cell_scaled[3:] = (perturbed_unit_cell[3:] - np.pi/2) / self.angle_scale

            augmented_entry['unit_cell_scaled'] = perturbed_unit_cell_scaled
            augmented_entry['unit_cell'] = perturbed_unit_cell
            if np.all(perturbed_unit_cell[3:] == np.pi/2):
                print('FAILED')
                print(permutation)
                print(perturbed_reindexed_unit_cell)
                print(perturbed_reindexed_unit_cell_)
                print(perturbed_unit_cell)
                print()
        elif self.lattice_system == 'orthorhombic':
            order = np.concatenate((
                np.argsort(np.array(augmented_entry['unit_cell_scaled'])[:3]), [3, 4, 5]
                ))
            augmented_entry['unit_cell_scaled'] = perturbed_reindexed_unit_cell_scaled[order]
            augmented_entry['unit_cell'] =  perturbed_reindexed_unit_cell[order]
        elif self.lattice_system == 'triclinic':
            assert False
        elif self.lattice_system in ['cubic', 'tetragonal']:
            augmented_entry['unit_cell_scaled'] = perturbed_reindexed_unit_cell_scaled
            augmented_entry['unit_cell'] = perturbed_reindexed_unit_cell
        else:
            assert False
        # calculate new d-spacings
        if self.lattice_system == 'monoclinic':
            q2_calculator = Q2Calculator(self.lattice_system, augmented_entry['hkl_sa'], tensorflow=False)
            q2_sa = q2_calculator.get_q2(np.array(augmented_entry['unit_cell'])[np.newaxis, :])[0]
            existing_peaks = np.any(augmented_entry['hkl_sa'] != 0, axis=1)
        else:
            q2_calculator = Q2Calculator(self.lattice_system, augmented_entry['reindexed_hkl_sa'], tensorflow=False)
            q2_sa = q2_calculator.get_q2(perturbed_reindexed_unit_cell[np.newaxis, :])[0]
            existing_peaks = np.any(augmented_entry['reindexed_hkl_sa'] != 0, axis=1)

        q2_sa = np.sort(q2_sa[existing_peaks])
        augmented_entry['d_spacing_sa'] = 1 / np.sqrt(q2_sa)
        augmented_entry['q2_sa'] = q2_sa

        # choose new peaks
        first_peak_index = self.first_probability['x'][
            np.searchsorted(self.first_probability['cdf'], self.rng.random())
            ]
        if first_peak_index >= q2_sa.size:
            return None
        q2 = [q2_sa[first_peak_index]]
        hkl = [augmented_entry['hkl_sa'][first_peak_index]]
        reindexed_hkl = [augmented_entry['reindexed_hkl_sa'][first_peak_index]]

        previous_kept_index = first_peak_index
        # overlap_threshold is the minimum separation from allowed during GenerateDataset.py
        # overlap_threshold = q2(2theta + overlap_threshold_theta2 / 2) - q2(2theta - overlap_threshold_theta2 / 2)
        # overlap_threshold = dq2_dtheta2 * overlap_threshold_theta2
        def get_overlap_threshold_q2(q2, fwhm, overlap_threshold_theta2, wavelength):
            fwhm = 0.1 * np.pi/180 # in radians
            overlap_threshold_theta2 = fwhm / 1.5
            theta2 = 2*np.arcsin(np.sqrt(q2)*wavelength/2)
            dq2_dtheta2 = (2/wavelength)**2 * np.sin(theta2/2) * np.cos(theta2/2)
            overlap_threshold_q2 = dq2_dtheta2 * overlap_threshold_theta2
            return overlap_threshold_q2

        keep_next = False
        fwhm, overlap_threshold_theta2, wavelength = get_fwhm_and_overlap_threshold()
        for index in range(first_peak_index + 1, q2_sa.size):
            # cases:
            # 1) Close to previous_kept: Reject
            # 2) Far from previous_kept and next: Use formula
            # 3) Far from previous_kept, close to next: Accept with 50% probability. If rejected, accept the next

            # There is a problem with really large q2 values. Like ~100 I believe they are comming from
            # setting the hkl to [-100, -100, -100] for empty peaks in the peak list
            if q2_sa[index] < 1:
                overlap_threshold_q2 = get_overlap_threshold_q2(q2_sa[index], fwhm, overlap_threshold_theta2, wavelength)
                distance_previous = q2_sa[index] - q2_sa[previous_kept_index]
                if index == q2_sa.size - 1:
                    separation = distance_previous
                else:
                    distance_next = q2_sa[index + 1] - q2_sa[index]
                    separation = min(distance_previous, distance_next)
                keep = False
                if keep_next:
                    keep = True
                    keep_next = False
                elif distance_previous > overlap_threshold_q2:
                    keep_next = False
                    if distance_next > overlap_threshold_q2:
                        if self.rng.random() < self.keep_rate(separation, *self.keep_rate_params):
                            keep = True
                    else:
                        if self.rng.random() < 1/2:
                            keep = True
                            keep_next = False
                        else:
                            keep_next = True
                if keep:
                    q2.append(q2_sa[index])
                    hkl.append(augmented_entry['hkl_sa'][index])
                    reindexed_hkl.append(augmented_entry['reindexed_hkl_sa'][index])
                    previous_kept_index = index

        if len(q2) >= self.n_points:
            q2 = np.array(q2)
            sort_indices = np.argsort(q2)
            q2 = q2[sort_indices][:self.n_points]

            hkl = np.array(hkl)[sort_indices][:self.n_points]
            reindexed_hkl = np.array(reindexed_hkl)[sort_indices][:self.n_points]
            augmented_entry[f'q2_{self.points_tag}'] = q2
            augmented_entry[f'd_spacing_{self.points_tag}'] = 1 / np.sqrt(q2)
            augmented_entry['hkl'] = hkl
            augmented_entry['reindexed_hkl'] = reindexed_hkl
            return augmented_entry
        else:
            return None

    def _check_in_range(self, perturbed_unit_cell_scaled):
        if self.lattice_system in ['monoclinic', 'triclinic']:
            limit = np.pi/2 / self.angle_scale
            if np.all(perturbed_unit_cell_scaled[:3] > self.min_unit_cell_scaled):
                if self.lattice_system == 'monoclinic':
                    if -limit < perturbed_unit_cell_scaled[3] < limit:
                        return True
                elif self.lattice_system == 'triclinic':
                    assert False
        else:
            if np.all(perturbed_unit_cell_scaled > self.min_unit_cell_scaled):
                return True

    def _permute_perturbed_unit_cell(self, perturbed_unit_cell_scaled):
        if self.lattice_system in ['monoclinic', 'triclinic']:
            perturbed_unit_cell_scaled[:3] = np.sort(perturbed_unit_cell_scaled[:3])
        elif self.lattice_system == 'orthorhombic':
            perturbed_unit_cell_scaled = np.sort(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled

    def perturb_unit_cell_std(self, unit_cell_scaled):
        # perturb unit cell
        status = True
        i = 0
        while status:
            perturbed_unit_cell_scaled = self.rng.normal(
                loc=unit_cell_scaled,
                scale=self.aug_params['augment_shift'],
                )[0]
            i += 1
            if self._check_in_range(perturbed_unit_cell_scaled):
                status = False
        perturbed_unit_cell_scaled = self._permute_perturbed_unit_cell(perturbed_unit_cell_scaled)
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
            if self._check_in_range(perturbed_unit_cell_scaled):
                status = False
        perturbed_unit_cell_scaled = self._permute_perturbed_unit_cell(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled

    def perturb_unit_cell_pca(self, unit_cell_scaled):
        # perturb unit cell
        status = True
        unit_cell_scaled_transformed = self.pca.transform(unit_cell_scaled[np.newaxis, :])[0]
        while status:
            perturbed_unit_cell_scaled_transformed = self.rng.normal(
                loc=unit_cell_scaled_transformed,
                scale=self.aug_params['augment_shift'] * self.stddev,
                )
            perturbed_unit_cell_scaled = self.pca.inverse_transform(
                perturbed_unit_cell_scaled_transformed[np.newaxis, :]
                )[0, :]
            if self._check_in_range(perturbed_unit_cell_scaled):
                status = False
        perturbed_unit_cell_scaled = self._permute_perturbed_unit_cell(perturbed_unit_cell_scaled)
        return perturbed_unit_cell_scaled
