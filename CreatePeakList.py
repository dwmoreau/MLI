"""
Improve peak fitting
    - show difference between fit and data
    - Plot breadths and asymetry

General
    - combine redundant functions
        - slice_refls
        - get_scattering_vectors
        - min_separation_check

    - variable rename
        - s1 to s1_lab
        - s1_norm to s1
"""
from cctbx import sgtbx
from cctbx import uctbx
from cctbx.crystal import symmetry
import cctbx.miller
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentList
from dxtbx import flumpy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.signal
import scipy.spatial.distance
import sklearn.metrics
import subprocess


def slice_refls(q2_obs, s1, start, refl_counts, refl_mask, mask):
    if mask:
        expt_refl_mask = refl_mask[start: start + refl_counts]
        expt_q2_obs = q2_obs[start: start + refl_counts][expt_refl_mask]
        expt_s1 = s1[start: start + refl_counts][expt_refl_mask]
    else:
        expt_q2_obs = q2_obs[start: start + refl_counts]
        expt_s1 = s1[start: start + refl_counts]
    start += refl_counts
    return expt_q2_obs, expt_s1, start


def get_scattering_vectors(s0, s1):
    wavelength = 1 / np.linalg.norm(s0)
    s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
    q = s1_normed - s0
    return q


def min_separation_check(q, min_separation):
    if min_separation is None:
        return True
    else:
        q2_diff_all = np.linalg.norm(q[:, np.newaxis, :] - q[np.newaxis, :, :], axis=2)**2
        indices = np.triu_indices(q.shape[0], k=1)
        q2_diff_lattice = q2_diff_all[indices[0], indices[1]]
        if np.min(q2_diff_lattice) > min_separation:
            return True
        else:
            return False


def calc_pairwise_diff(q, q_ref=None, metric='euclidean'):
    """
    Calculate pairwise differences between vectors
    
    Parameters:
    q : array of shape (N, 3)
        Array of N 3D vectors
    metric : str
        'euclidean' for Euclidean distance
        'angular' for angular distance in radians
    
    Returns:
    distances : array of shape (N*(N-1)/2,)
        Condensed distance matrix
    """
    if metric == 'euclidean':
        return scipy.spatial.distance.pdist(q, metric='euclidean')
    
    elif metric == 'angular':
        if q_ref is None:
            assert False
        # Normalize vectors
        q_norm = q / np.linalg.norm(q, axis=1)[:, np.newaxis]
        
        # Calculate cosine distances and convert to angles
        #cos_angles = scipy.spatial.distance.pdist(q_norm, metric='cosine')
        i, j = np.triu_indices(len(q), k=1)
        cos_angles = np.dot(q_norm, q_norm.T)
        valid = np.logical_and(cos_angles >= -1, cos_angles <= 1)
        angles = np.full(cos_angles.shape, np.nan)
        angles[valid] = np.arccos(cos_angles[valid])
        angles = angles[i, j]
        
        # Calculate cross products for handedness
        # Convert to full matrix indices
        cross_products = np.cross(q_norm[i], q_norm[j])
        
        # Use z-component sign for handedness
        signs = np.sign(np.matmul(q_ref, cross_products.T))
        return signs * angles


class PeakListCreator:
    def __init__(
        self, 
        tag,
        save_to_directory=None,
        load_combined=False,
        overwrite_combined=False,
        runs=None,
        run_limits=None,
        run_limits_sacla=None,
        input_path_template=None,
        suffix='_strong.expt',
        min_reflections_per_experiment=3,
        max_reflections_per_experiment=100,
        known_unit_cell=None, 
        known_space_group=None,
        variable_detector=False,
    ):
        
        if type(input_path_template) == str or input_path_template is None:
            self.multiple_run_groups = False
            self.input_path_template = [input_path_template]
        elif type(input_path_template) == list:
            self.multiple_run_groups = True
            self.input_path_template = input_path_template

        if not run_limits_sacla is None:
            assert False
            self.runs = []
            for run_index in range(run_limits_sacla[0], run_limits_sacla[1] + 1):
                for sub_run_index in range(3):
                    self.runs.append(f'{run_index}-{sub_run_index}')
        elif not run_limits is None:
            if self.multiple_run_groups:
                self.runs = [np.arange(rl[0], rl[1] + 1) for rl in run_limits]
            else:
                self.runs = [np.arange(run_limits[0], run_limits[1] + 1)]
        elif runs is None:
            self.runs = [[None]]
        else:
            if self.multiple_run_groups:
                self.runs = runs
            else:
                self.runs = [runs]

        self.max_reflections_per_experiment = max_reflections_per_experiment
        self.min_reflections_per_experiment = min_reflections_per_experiment

        self.variable_detector = variable_detector
        
        self.suffix = suffix
        self.tag = tag
        self.load_combined = load_combined
        self.overwrite_combined = overwrite_combined
        if save_to_directory is None:
            self.save_to_directory = os.path.join(os.getcwd(), self.tag)
        else:
            self.save_to_directory = os.path.join(save_to_directory, self.tag)
        if not os.path.exists(self.save_to_directory):
            os.mkdir(self.save_to_directory)
        
        self.expt_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined_all.expt')
        self.refl_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined_all.refl')
        if self.load_combined == False:
            self._combine_expt_refl_files()
            self._parse_refl_file()
        else:
            self.q2_obs = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_q2_obs.npy'),
                )
            self.refl_counts = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_refl_counts.npy'),
                )
            self.expt_indices = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_expt_indices.npy'),
                )
            self.s0 = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_s0.npy'),
                )
            self.s1 = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_s1.npy'),
                )
        self.beam_delta = np.zeros(2)
        self.refl_mask = np.ones(self.q2_obs.size, dtype=bool)
        self.known_unit_cell = known_unit_cell
        self.known_space_group = known_space_group
        self.error = None
        self.triplets_obs = None

    def _run_combine_experiments(self, expt_file_names, refl_file_names, run_str):
        command = ['dials.combine_experiments']
        command += expt_file_names
        command += refl_file_names
        if self.variable_detector == False:
            command += ['reference_from_experiment.detector=0']
        command += [
            f'max_reflections_per_experiment={self.max_reflections_per_experiment}',
            f'min_reflections_per_experiment={self.min_reflections_per_experiment}',
            f'output.experiments_filename={self.tag}_combined_{run_str}.expt',
            f'output.reflections_filename={self.tag}_combined_{run_str}.refl',
            ]
        log_file_name = os.path.join(
            self.save_to_directory,
            f'{self.tag}_combine_experiments_{run_str}.log'
            )

        output_refl_file_name = os.path.join(
            self.save_to_directory,
            f'{self.tag}_combined_{run_str}.refl'
            )
        if self.overwrite_combined == False and os.path.exists(output_refl_file_name):
            print(f'Loading combined expt and refls for run {run_str}')
        else:
            print(f'Combining experiments in run {run_str}')
            with open(log_file_name, 'w') as log_file:
                subprocess.run(command, cwd=self.save_to_directory, stdout=log_file)
        with open(log_file_name, 'r') as log_file:
            expt_counts = 0
            refl_counts = 0
            count = False
            for line in log_file:
                if line == '+--------------+-------------------------+\n':
                    count = False
                if count:
                    expt_counts = int(line.split('|')[1])
                    refl_counts += int(line.split('|')[2])
                if line == '|--------------+-------------------------|\n':
                    count = True
        print(f'    Run {run_str} has {expt_counts} experiments and {refl_counts} reflections')
        return refl_counts

    def _combine_expt_refl_files(self):
        expt_file_names = []
        refl_file_names = []
        for rg_index in range(len(self.input_path_template)):
            expt_file_names_rg = []
            refl_file_names_rg = []
            for run in self.runs[rg_index]:
                expt_file_names_run = []
                refl_file_names_run = []
                if type(run) == str:
                    run_str = run
                elif run is None:
                    run_str = ''
                else:
                    run_str = f'{run:04d}'
                input_path = self.input_path_template[rg_index].replace('!!!!', run_str)
                if os.path.exists(input_path):
                    for file_name in os.listdir(input_path):
                        if file_name.endswith(self.suffix):
                            expt_file_name = os.path.join(input_path, file_name)
                            refl_file_name = os.path.join(input_path, file_name.replace('.expt', '.refl'))
                            if os.path.exists(expt_file_name) and os.path.exists(refl_file_name):
                                expt_file_names_run.append(expt_file_name)
                                refl_file_names_run.append(refl_file_name)
                    if len(expt_file_names_run) > 0:
                        refl_counts = self._run_combine_experiments(
                            expt_file_names_run, refl_file_names_run, run_str
                            )
                        if refl_counts > 0:
                            expt_file_names_rg.append(os.path.join(
                                self.save_to_directory, f'{self.tag}_combined_{run_str}.expt'
                                ))
                            refl_file_names_rg.append(os.path.join(
                                self.save_to_directory, f'{self.tag}_combined_{run_str}.refl'
                                ))
            self._run_combine_experiments(
                expt_file_names_rg, refl_file_names_rg, f'rg_index_{rg_index}'
                )
            expt_file_names.append(os.path.join(
                self.save_to_directory, f'{self.tag}_combined_rg_index_{rg_index}.expt'
                ))
            refl_file_names.append(os.path.join(
                self.save_to_directory, f'{self.tag}_combined_rg_index_{rg_index}.refl'
                ))
        self._run_combine_experiments(
            expt_file_names, refl_file_names, 'all'
            )
    
    def _get_s1_from_xyz(self, panel, xyz, wavelength):
        s1 = flumpy.to_numpy(
                panel.get_lab_coord(panel.pixel_to_millimeter(flex.vec2_double(
                    flex.double(xyz[:, 0].ravel()),
                    flex.double(xyz[:, 1].ravel())
                )))
            )
        # s1 is the vector going from the interation point to the peak with magnitude 1/wavelength
        s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
        return s1_normed, s1

    def _get_q2_from_xyz(self, panel, xyz, s0):
        return np.array([1 / panel.get_resolution_at_pixel(s0, xyz[i][0:2])**2 for i in range(len(xyz))])

    def _get_q2_spacing(self, s1, s0):
        wavelength = 1 / np.linalg.norm(s0)
        dot_product = np.matmul(s1, s0)
        magnitudes = np.linalg.norm(s1, axis=1) * np.linalg.norm(s0)
        theta2 = np.arccos(dot_product / magnitudes)
        return ((2 * np.sin(theta2 / 2)) / wavelength)**2

    def _parse_refl_file(self):
        expts = ExperimentList.from_file(self.expt_file_name, check_format=False)
        refls = flex.reflection_table.from_file(self.refl_file_name)
        q2 = []
        s1 = []
        s0 = []
        expt_indices = []
        refl_counts = []
        print('Parsing Reflection File')
        for expt_index, expt in enumerate(expts):
            refls_expt = refls.select(refls['id'] == expt_index)
            if len(refls_expt) > 0:
                wavelength = expt.beam.get_wavelength()
                s0_lattice = expt.beam.get_s0() #|s0| = 1/wavelength
                # s1 is the vector going from the interaction point to the crystal
                # s1_normed has magnitude 1/wavelength
                s1_normed_lattice = []
                s1_lattice = []
                for panel_index, panel in enumerate(expt.detector):
                    refls_panel = refls_expt.select(refls_expt['panel'] == panel_index)
                    if len(refls_panel) > 0:
                        s1_normed_panel, s1_panel = self._get_s1_from_xyz(
                            panel, 
                            flumpy.to_numpy(refls_panel['xyzobs.px.value']), 
                            wavelength,
                            )
                        s1_normed_lattice.append(s1_normed_panel)
                        s1_lattice.append(s1_panel)
                s1_lattice = np.row_stack(s1_lattice)
                refl_counts.append(s1_lattice.shape[0])
                # s0 and s1 are retained for constructing secondary peaks and beam center optimization
                s1.append(np.row_stack(s1_lattice))
                s0.append(s0_lattice)
                expt_indices.append(expt_index*np.ones(s1_lattice.shape[0], dtype=int))
                # q2_lattice is the magnitude**2 of the scattering vector
                q2.append(self._get_q2_spacing(
                    np.row_stack(s1_normed_lattice), s0_lattice)
                    )
        self.q2_obs = np.concatenate(q2)
        self.refl_counts = np.array(refl_counts)
        self.expt_indices = np.concatenate(expt_indices)
        self.s0 = np.row_stack(s0)
        self.s1 = np.row_stack(s1)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_q2_obs.npy'),
            self.q2_obs
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_refl_counts.npy'),
            self.refl_counts
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_expt_indices.npy'),
            self.expt_indices
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_s0.npy'),
            self.s0
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_s1.npy'),
            self.s1
            )

    def quick_mask(self, n_bins=1000, threshold=20, pad=5, llur=None):
        """
        Masking algorithm:
            1: Calculate a 2D histogram of the reflection positions on the detector surface.
            2: Calculate an azimuthal average of the reflection positions.
            3: Project the azimuthal average onto the detector surface.
            4: Mask regions where the histogram is much larger than the azimuthal average.
        """
        # Array rows are coordinate y
        # Array cols are coordinate x
        bins_x = np.linspace(self.s1[:, 1].min(), self.s1[:, 1].max(), n_bins + 1)
        bins_y = np.linspace(self.s1[:, 0].min(), self.s1[:, 0].max(), n_bins + 1)
        centers_x = (bins_x[1:] + bins_x[:-1]) / 2
        centers_y = (bins_y[1:] + bins_y[:-1]) / 2

        # 2D histogram of the reflection positions
        hist, _, _ = np.histogram2d(x=self.s1[:, 1], y=self.s1[:, 0], bins=[bins_x, bins_y])

        # This maps the reflections onto the histogram coordinates
        refl_x = np.searchsorted(bins_x, self.s1[:, 1]) - 1
        refl_y = np.searchsorted(bins_y, self.s1[:, 0]) - 1
        refl_x[refl_x == -1] = 0
        refl_y[refl_y == -1] = 0

        # This should be the correct way to do this, the detector distance should be the average of
        # the detector distance of the reflections in the xy bin. This does not work though.
        # Using the same detector distance of 
        #centers_z, _, _, _ = scipy.stats.binned_statistic_2d(
        #    x=self.s1[:, 1],
        #    y=self.s1[:, 0], 
        #    values=self.s1[:, 2],
        #    bins=[bins_x, bins_y],
        #    statistic='mean'
        #    )
        #centers_z[np.isnan(centers_z)] = np.nanmean(centers_z)
        centers_z = self.s1[:, 2].mean()

        # This performs the azimuthal average and projection onto the detector surface.
        s1_lab_mag_centers = centers_x[np.newaxis, :]**2 + centers_y[:, np.newaxis]**2 + centers_z**2
        s1_lab_mag_bins = np.linspace(s1_lab_mag_centers.min(), s1_lab_mag_centers.max(), int(n_bins/2) + 1)
        azimuthal_mean, _, _ = scipy.stats.binned_statistic(
            x=s1_lab_mag_centers.ravel(), values=hist.ravel(), statistic='mean', bins=s1_lab_mag_bins
            )
        indices = np.searchsorted(s1_lab_mag_bins, s1_lab_mag_centers) - 1
        indices[indices == -1] = 0
        mean_projection = np.take(azimuthal_mean, indices)
        # This takes all the zero pixels and makes them nonzero to prevent large amounts of false positives
        mean_projection[mean_projection < mean_projection.mean()] = mean_projection.mean()

        # Create a detector surface mask and then pad it.
        mask = hist > threshold*mean_projection
        mask_indices_minimal = np.column_stack(np.nonzero(mask))
        mask_indices = []
        for index in range(mask_indices_minimal.shape[0]):
            mask_x = mask_indices_minimal[index, 1]
            mask_y = mask_indices_minimal[index, 0]
            for pad_x in range(-pad + mask_x, pad + mask_x + 1):
                for pad_y in range(-pad + mask_y, pad + mask_y + 1):
                    if 0 <= pad_x < n_bins:
                        if 0 <= pad_y < n_bins:
                            mask_indices.append([pad_y, pad_x])
        mask_indices = np.row_stack((mask_indices))
        mask[mask_indices[:, 0], mask_indices[:, 1]] = True



        # Mask for the reflections that fit within the detector mask
        # self.refl_mask is created in the __init__ method
        # Remaking it resets the mask
        self.refl_mask = np.ones(self.q2_obs.size, dtype=bool)
        for index in range(mask_indices.shape[0]):
            indices = np.logical_and(
                refl_x == mask_indices[index, 0],
                refl_y == mask_indices[index, 1]
                )
            self.refl_mask[indices] = False
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1[:, 0], self.s1[:, 1],
            s=0.01, color=[0, 0, 0], alpha=0.1
            )
        axes.imshow(
            mask, cmap='Reds', alpha=0.4,
            origin='lower', extent=(centers_x[0], centers_x[-1], centers_y[0], centers_y[-1])
            )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('Scatter Plot Of Reflection Coordinates\nMask in red\nThere is a bug and the mask and reflections are offset')
        fig.tight_layout()
        plt.show()

        # Make sure the masked reflections actually line up with the mask
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1[self.refl_mask, 0], self.s1[self.refl_mask, 1],
            s=0.01, color=[0, 0, 0], alpha=0.1
            )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('Scatter Plot Of Masked Reflection Coordinates')
        fig.tight_layout()
        plt.show()
        
        """
        # Diagnostic plots

        # Make sure the masked reflections actually line up with the mask
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1[self.refl_mask, 0], self.s1[self.refl_mask, 1],
            s=0.01, color=[0, 0, 0], alpha=0.1
            )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('Scatter Plot Of Reflection Coordinates\nCoordindates Masked')
        fig.tight_layout()
        plt.show()

        # Azimuthal mean
        fig, axes = plt.subplots(1, 1, figsize=(7, 3))
        axes.plot(azimuthal_mean)
        plt.show()

        # 2D image of the s1_lab
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.imshow(s1_lab_mag_centers, origin='lower')
        fig.tight_layout()
        plt.show()

        # 2D Histogram of the reflections
        vmax = np.sort(hist.ravel())[int(0.999*hist.size)]
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes.imshow(hist, cmap='gray_r', vmin=0, vmax=vmax, origin='lower')
        fig.tight_layout()
        plt.show()

        # Projection of the azimuthal mean onto the detector surface.
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes.imshow(mean_projection, cmap='gray_r', vmin=0, vmax=vmax, origin='lower')
        fig.tight_layout()
        plt.show()
        """

    def make_histogram(self, n_bins=1000, d_min=60, d_max=3.5, q2_min=None, q2_max=None, mask=True):
        if q2_min is None:
            self.d_min = d_min
            self.q2_min = 1 / self.d_min**2
        else:
            self.q2_min = q2_min
            self.d_min = 1/np.sqrt(q2_min)
        if q2_max is None:
            self.d_max = d_max
            self.q2_max = 1 / self.d_max**2
        else:
            self.q2_max = q2_max
            self.d_max = 1/np.sqrt(q2_max)
        self.q2_bins = np.linspace(self.q2_min, self.q2_max, n_bins + 11)
        self.q2_centers = (self.q2_bins[1:] + self.q2_bins[:-1]) / 2
        if mask:
            self.q2_hist, _ = np.histogram(self.q2_obs[self.refl_mask], bins=self.q2_bins)
        else:
            self.q2_hist, _ = np.histogram(self.q2_obs, bins=self.q2_bins)

    def pick_peaks(self, exclude_list=[], exclude_max=20, add_peaks=[], shift={}, prominence=30, plot_kapton_peaks=False, yscale=None):
        found_peak_indices = scipy.signal.find_peaks(self.q2_hist, prominence=prominence)
        found_peaks = self.q2_centers[found_peak_indices[0]]
        found_peaks = np.delete(found_peaks[:exclude_max], exclude_list)
        peaks = np.sort(np.concatenate((found_peaks, add_peaks)))
    
        fig, axes = plt.subplots(1, 1, figsize=(30, 6))
        axes.plot(self.q2_centers, self.q2_hist, label='Histogram')
        for p_index, p in enumerate(peaks):
            if p_index in shift.keys():
                peaks[p_index] += shift[p_index]
            if p in add_peaks:
                color = [0.8, 0, 0]
            else:
                color = [0, 0, 0]
            axes.plot(
                [p, p], [0, self.q2_hist.max()],
                linestyle='dotted', linewidth=1, color=color
                )
            axes.annotate(p_index, xy=(p-0.001, (1-p_index/peaks.size) * self.q2_hist.max()))
        if plot_kapton_peaks:
            kapton_peaks = [15.25, 7.625, 5.083333333, 3.8125, 3.05]
            for p in kapton_peaks:
                if p > self.d_max:
                    axes.plot([1/p**2, 1/p**2], [0, self.q2_hist.max()], linestyle='dotted', linewidth=2, color=[0, 0.7, 0], label='Kapton Peaks')
        axes.set_xlabel('q2 (1/$\mathrm{\AA}^2$')
        if yscale == 'log':
            axes.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_peaks.png'))
        plt.show()
        self.q2_peaks = peaks
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_peaks.npy'),
            self.q2_peaks
            )
        print(repr(self.q2_peaks))
        print(repr(1/np.sqrt(self.q2_peaks)))

    def fit_peaks(self, n_max, ind_peak_indices, fit_shift=True, exclude_fit_shift=[]):
        def get_I_calc(amplitudes, q2_centers, broadening_params, q2, jac=False):
            breadths = (broadening_params[0] + broadening_params[1]*q2_centers)[:, np.newaxis]
            prefactor = 1 / np.sqrt(2*np.pi * breadths**2)
            exponential = np.exp(-1/2 * ((q2_centers[:, np.newaxis] - q2[np.newaxis]) / breadths)**2)
            I_calc = np.sum(amplitudes[:, np.newaxis] * prefactor * exponential, axis=0)
            if jac:
                dI_calc_damplitudes = prefactor * exponential

                dexponential_dq2_centers = -exponential * (q2_centers[:, np.newaxis] - q2[np.newaxis]) / breadths**2
                dI_calc_dq2_centers = amplitudes[:, np.newaxis] * prefactor * dexponential_dq2_centers
                return I_calc, dI_calc_damplitudes, dI_calc_dq2_centers
            else:
                return I_calc
        def fit_loss(x, amplitudes, q2_centers, mask, broadening_params, I_obs, q2, mode):
            amplitudes_all = np.zeros(mask.size)
            q2_centers_all = np.zeros(mask.size)
            if mode == 'amplitudes':
                amplitudes_all[mask] = x
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all = q2_centers
            elif mode == 'amplitudes_centers':
                amplitudes_all[mask] = x[:mask.sum()]
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all[mask] = x[mask.sum():]
                q2_centers_all[~mask] = q2_centers[~mask]
            I_calc = get_I_calc(amplitudes_all, q2_centers_all, broadening_params, q2, False)
            L = I_calc - I_obs
            return L
        def fit_jac(x, amplitudes, q2_centers, mask, broadening_params, I_obs, q2, mode):
            amplitudes_all = np.zeros(mask.size)
            q2_centers_all = np.zeros(mask.size)
            if mode == 'amplitudes':
                amplitudes_all[mask] = x
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all = q2_centers
            elif mode == 'amplitudes_centers':
                amplitudes_all[mask] = x[:mask.sum()]
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all[mask] = x[mask.sum():]
                q2_centers_all[~mask] = q2_centers[~mask]
            I_calc, dI_calc_damplitudes, dI_calc_dq2_centers = get_I_calc(amplitudes_all, q2_centers_all, broadening_params, q2, True)
            if mode == 'amplitudes':
                jac = dI_calc_damplitudes[mask].T
            elif mode == 'amplitudes_centers':
                jac = np.concatenate((dI_calc_damplitudes[mask], dI_calc_dq2_centers[mask]), axis=0).T
            return jac
        def basic_gaussian(p, x):
            return p[0] / np.sqrt(2*np.pi*p[1]**2) * np.exp(-1/2 * ((x - p[2]) / p[1])**2)
        def basic_gaussian_loss(p, x, y):
            return basic_gaussian(p, x) - y

        # Start by fitting individual peaks
        # Peaks fit individually will be fixed in the next stages when peaks during the profile fit.
        ind_amplitudes = np.zeros(len(ind_peak_indices))
        ind_breadths = np.zeros(len(ind_peak_indices))
        ind_q2_centers = np.zeros(len(ind_peak_indices))

        delta = int(0.003 / (self.q2_centers[1] - self.q2_centers[0]))
        for index, peak_index in enumerate(ind_peak_indices):
            loc = np.searchsorted(self.q2_centers, self.q2_peaks[peak_index])
            low = max(0, loc - delta)
            high = min(self.q2_centers.size, loc + delta)
            sigma = 0.0001
            amplitude = (self.q2_hist[low: high].max() - self.q2_hist[low: high].min()) * np.sqrt(2*np.pi)*sigma
            results = scipy.optimize.least_squares(
                basic_gaussian_loss,
                x0=(amplitude, sigma, self.q2_peaks[peak_index]),
                args=(self.q2_centers[low: high], self.q2_hist[low: high])
                )
            ind_amplitudes[index] = np.abs(results.x[0])
            ind_breadths[index] = np.abs(results.x[1])
            ind_q2_centers[index] = np.abs(results.x[2])

        broadening_params_polyfit = np.polyfit(x=ind_q2_centers, y=ind_breadths, deg=1)
        self.broadening_params = np.array([broadening_params_polyfit[1], broadening_params_polyfit[0]])
        self.q2_breadths = np.polyval(broadening_params_polyfit, self.q2_peaks)

        mask = np.ones(n_max, dtype=bool)
        amplitudes = np.zeros(n_max)
        q2_centers = self.q2_peaks[:n_max].copy()
        for index, peak_index in enumerate(ind_peak_indices):
            if peak_index < n_max:
                mask[peak_index] = False
                amplitudes[peak_index] = ind_amplitudes[index]
                q2_centers[peak_index] = ind_q2_centers[index]

        # Fit breadths and amplitudes
        max_index = np.searchsorted(self.q2_centers, self.q2_peaks[n_max]) + 20
        results = scipy.optimize.least_squares(
            fit_loss,
            x0=amplitudes[mask],
            jac=fit_jac,
            args=(
                amplitudes,
                q2_centers,
                mask,
                self.broadening_params,
                self.q2_hist[:max_index],
                self.q2_centers[:max_index],
                'amplitudes'
                ),
            method='lm',
            )
        amplitudes[mask] = results.x
        print(results)
        if fit_shift:
            # Fit breadths, amplitudes, and shift
            x0 = np.concatenate((amplitudes[mask], q2_centers[mask]))
            print(x0.shape, x0) 
            results = scipy.optimize.least_squares(
                fit_loss,
                x0=x0,
                jac=fit_jac,
                args=(
                    amplitudes,
                    q2_centers,
                    mask,
                    self.broadening_params,
                    self.q2_hist[:max_index],
                    self.q2_centers[:max_index],
                    'amplitudes_centers'
                    ),
                method='lm'
                )
            print(results)
            amplitudes[mask] = results.x[:mask.sum()]
            q2_centers[mask] = results.x[mask.sum():]
            q2_peaks_original = self.q2_peaks[:n_max].copy()
            for peak_index in range(self.q2_peaks[:n_max].size):
                if not peak_index in exclude_fit_shift:
                    self.q2_peaks[peak_index] = q2_centers[peak_index]

        I_calc = get_I_calc(amplitudes, q2_centers, self.broadening_params, self.q2_centers[:max_index])
        fig, axes = plt.subplots(1, 1, figsize=(30,  8), sharex=True)
        axes.plot(self.q2_centers[:max_index], self.q2_hist[:max_index])
        axes.plot(self.q2_centers[:max_index], I_calc)
        ylim = axes.get_ylim()
        for peak_index, p in enumerate(self.q2_peaks[:n_max]):
            if p in ind_peak_indices:
                color = [0.8, 0, 0]
            else:
                color = [0, 0, 0]
            axes.plot([p, p], ylim, color=color, linestyle='dotted')
        if fit_shift:
            for i in range(n_max):
                shift = self.q2_peaks[i] - q2_peaks_original[i]
                axes.annotate(
                    f'{shift:0.5f}',
                    xy=(self.q2_peaks[i], 0.9 * ylim[1]),
                    rotation=90
                    )
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.plot(self.q2_peaks[ind_peak_indices], ind_breadths, marker='.')
        axes.plot(self.q2_peaks[ind_peak_indices], np.polyval(broadening_params_polyfit, self.q2_peaks[ind_peak_indices]))
        plt.show()

    def optimize_beam_center(self, primary_peak_indices, mask=True):
        def get_q2_spacing(s1, s0):
            wavelength = 1 / np.linalg.norm(s0)
            dot_product = np.matmul(s1, s0)
            magnitudes = np.linalg.norm(s1) * np.linalg.norm(s0)
            theta2 = np.arccos(dot_product / magnitudes)
            return ((2 * np.sin(theta2 / 2)) / wavelength)**2
    
        def functional(delta, s1_list, s0_list):
            L = 0
            for peak_index in range(len(s1_list)):
                s1 = s1_list[peak_index]
                s0 = s0_list[peak_index]
                q2_calc = np.zeros(s1.shape[0])
                for i in range(s1.shape[0]):
                    s1_delta = s1[i].copy()
                    s1_delta[:2] += delta
                    q2_calc[i] = get_q2_spacing(s1_delta, s0[i])
                L += q2_calc.std()
            return L

        s1 = []
        s0 = []
        if mask:
            q2_obs_masked = self.q2_obs[self.refl_mask]
            s1_masked = self.s1[self.refl_mask]
            expt_indices_masked = self.expt_indices[self.refl_mask]
        else:
            q2_obs_masked = self.q2_obs
            s1_masked = self.s1
            expt_indices_masked = self.expt_indices

        for peak_index in primary_peak_indices:
            differences = np.abs(q2_obs_masked - self.q2_peaks[peak_index])
            indices = differences < 3*self.q2_breadths[peak_index]
            s1.append(s1_masked[indices])
            s0.append(self.s0[expt_indices_masked[indices]])
    
        initial_simplex = np.array([
            [0.05, 0.025],
            [0.001, -0.01],
            [-0.025, -0.05],
            ])
        print(functional(np.zeros(2), s1, s0))
        
        results = scipy.optimize.minimize(
            fun=functional,
            x0=[0, 0],
            args=(s1, s0),
            method='Nelder-Mead',
            options={'initial_simplex': initial_simplex}
            )
        print(results)
        self.beam_delta = results.x[:2]
        self.s1[:, :2] += self.beam_delta
        start = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            self.q2_obs[start: start + refl_counts] = self._get_q2_spacing(
                self.s1[start: start + refl_counts], self.s0[expt_index]
                )
            start += refl_counts

    def bump_detector_distance(self, bump):
        self.s1[:, 2] += bump
        q2 = []
        start = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            q2_obs = self.q2_obs[start: start + refl_counts]
            s1 = self.s1[start: start + refl_counts]
            s0 = self.s0[expt_index]
            wavelength = 1 / np.linalg.norm(s0)
            s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
            # q2_lattice is the magnitude**2 of the scattering vector
            q2.append(self._get_q2_spacing(s1_normed, s0))
            start += refl_counts
        self.q2_obs = np.concatenate(q2)

    def filter_peaks(self, n_peaks=20, max_difference=None, delta=None, max_refl_counts=None, threshold=0.50, mask=True):
        # assign peaks and get distances
        # The :n_peaks+1 appears unnecessary, but is important
        # If the peak gets assigned to the n_peaks index, it is probably out of the range of diffraction
        # This catches those cases for them to be ignored
        differences_all = np.abs(self.q2_obs[:, np.newaxis] - self.q2_peaks[np.newaxis, :n_peaks + 1])
        assignment = np.argmin(differences_all, axis=1)
        differences = np.take_along_axis(differences_all, assignment[:, np.newaxis], axis=1)[:, 0]

        joint_occurances = np.zeros((n_peaks, n_peaks))
        ind_occurances = np.zeros(n_peaks)
        start = 0
        n_experiments = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            if mask:
                expt_refl_mask = self.refl_mask[start: start + refl_counts]
                assignment_expt = assignment[start: start + refl_counts][expt_refl_mask]
                differences_expt = differences[start: start + refl_counts][expt_refl_mask]
                masked_refl_counts = np.sum(expt_refl_mask)
            else:
                assignment_expt = assignment[start: start + refl_counts]
                differences_expt = differences[start: start + refl_counts]
                masked_refl_counts = refl_counts
            if masked_refl_counts > 0:
                if max_refl_counts is None or masked_refl_counts < max_refl_counts:
                    if not max_difference is None:
                        assignment_expt = assignment_expt[differences_expt < max_difference]
                    elif not delta is None:
                        peak_breadths = np.take(self.q2_breadths, assignment_expt)
                        tolerance = delta * peak_breadths
                        assignment_expt = assignment_expt[differences_expt < tolerance]
                        
                    unique_assignments = np.sort(np.unique(assignment_expt))
                    if unique_assignments.size > 0 and unique_assignments[-1] == n_peaks:
                        unique_assignments = unique_assignments[:-1]
                    #print(unique_assignments)
                    if unique_assignments.size > 0:
                        n_experiments += 1
                        for peak_index_0 in range(n_peaks):
                            if peak_index_0 in unique_assignments:
                                ind_occurances[peak_index_0] += 1
                                for peak_index_1 in range(n_peaks):
                                    if peak_index_1 in unique_assignments:
                                        joint_occurances[peak_index_0, peak_index_1] += 1
                    #print(ind_occurances)
                    #print(joint_occurances)
                    #print()
            start += refl_counts

        #joint_prob = joint_occurances / n_experiments
        #ind_prob = ind_occurances / n_experiments
        #separated_prob = 1/2*(ind_occurances[np.newaxis] + ind_occurances[:, np.newaxis]) / n_experiments
        #ratio = joint_prob/separated_prob

        ratio = joint_occurances / (ind_occurances[np.newaxis] * ind_occurances[:, np.newaxis] / n_experiments)

        ratio[np.arange(n_peaks), np.arange(n_peaks)] = np.nan
        paired = ratio > threshold

        print('Paired Peaks')
        for peak_index_0 in range(n_peaks):
            for peak_index_1 in range(peak_index_0, n_peaks):
                if paired[peak_index_0, peak_index_1]:
                    print(peak_index_0, peak_index_1)

        fig, axes = plt.subplots(1, 1, figsize=(10, 3))
        axes.bar(np.arange(n_peaks), ind_prob, width=1)
        axes.set_xlabel('Peak index')
        axes.set_ylabel('Occurance Probability')
        plt.show()

        cmap = 'binary'
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        separated_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=separated_prob)
        separated_disp.plot(include_values=False, ax=axes[0, 0], cmap=cmap)
        
        joint_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=joint_prob)
        joint_disp.plot(include_values=False, ax=axes[0, 1], cmap=cmap)

        ratio_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=ratio)
        ratio_disp.plot(include_values=False, ax=axes[1, 0], cmap=cmap)
        
        paired_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=paired)
        paired_disp.plot(include_values=False, ax=axes[1, 1], cmap=cmap)

        axes[0, 0].set_title('Separated Probability')
        axes[0, 1].set_title('Joint Probability')
        axes[1, 0].set_title('Joint/Separated Probability')
        axes[1, 1].set_title(f'Joint/Separated Probability > {threshold}')

        fig.tight_layout()
        plt.show()

    def output_json(self, note=None, extra_file_name=None):
        output = {
            'primary_peaks': self.q2_peaks,
            'secondary_peaks': self.q2_peaks_secondary,
            'primary_hist': np.column_stack((self.q2_centers, self.q2_hist)),
            'secondary_hist': np.column_stack((self.q2_diff_centers, self.q2_diff_hist)),
            'triplet_obs': self.triplets_obs,
            'broadening_params': self.broadening_params,
            'error': self.error,
            'note': note,
            }
        if extra_file_name is None:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info.json')
        else:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info_{extra_file_name}.json')
        pd.Series(output).to_json(file_name)

    def create_secondary_peaks(self, q2_max=None, max_difference=None, max_refl_counts=None, min_separation=None, n_bins=2000, mask=True):
        start = 0
        q2_diff = []
        min_separation_obs = []
        for expt_index, refl_counts in enumerate(self.refl_counts):
            if mask:
                expt_refl_mask = self.refl_mask[start: start + refl_counts]
                q2_obs = self.q2_obs[start: start + refl_counts][expt_refl_mask]
                s1 = self.s1[start: start + refl_counts][expt_refl_mask]
                masked_refl_counts = np.sum(expt_refl_mask)
            else:
                q2_obs = self.q2_obs[start: start + refl_counts]
                s1 = self.s1[start: start + refl_counts]
                masked_refl_counts = refl_counts
            if masked_refl_counts > 0:
                if max_refl_counts is None or masked_refl_counts < max_refl_counts:
                    s0 = self.s0[expt_index]
                    wavelength = 1 / np.linalg.norm(s0)
                    if not max_difference is None:
                        min_error = np.min(
                            np.abs(q2_obs[:, np.newaxis] - self.q2_peaks[np.newaxis]),
                            axis=1
                            )
                        indices = min_error < max_difference
                        q2_obs = q2_obs[indices]
                        s1 = s1[indices]
                    if not q2_max is None:
                        indices = q2_obs < q2_max
                        q2_obs = q2_obs[indices]
                        s1 = s1[indices]

                    if q2_obs.size > 1:
                        s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
                        q2_diff_all = np.linalg.norm(
                            s1_normed[np.newaxis, :, :] - s1_normed[:, np.newaxis, :],
                            axis=2
                            )**2
                        indices = np.triu_indices(s1.shape[0], k=1)
                        q2_diff_lattice = q2_diff_all[indices[0], indices[1]]
                        min_separation_obs.append(np.min(q2_diff_lattice))
                        if min_separation is None or np.min(q2_diff_lattice) > min_separation:
                            q2_diff.append(q2_diff_lattice)
            start += refl_counts
        self.q2_diff = np.concatenate(q2_diff)
        min_separation_obs = np.array(min_separation_obs)

        self.q2_diff_bins = np.linspace(0.00000001, self.q2_max, n_bins + 1)
        self.q2_diff_centers = (self.q2_diff_bins[1:] + self.q2_diff_bins[:-1]) / 2
        self.q2_diff_hist, _ = np.histogram(self.q2_diff, bins=self.q2_diff_bins)

        fig, axes = plt.subplots(1, 1, figsize=(40, 5))
        axes.plot(self.q2_diff_centers, self.q2_diff_hist)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_q2_diff_hist.npy'),
            np.column_stack((self.q2_diff_centers, self.q2_diff_hist))
            )

        fig, axes = plt.subplots(1, 3, figsize=(7, 3))

        indices = self.q2_obs < self.q2_peaks.max()
        min_error = np.min(np.abs(self.q2_obs[indices, np.newaxis] - self.q2_peaks[np.newaxis]), axis=1)
        axes[0].hist(min_error, bins=100, log=True)
        if not max_difference is None:
            ylim = axes[0].get_ylim()
            axes[0].plot([max_difference, max_difference], ylim, color=[0, 0, 0])
            axes[0].set_ylim(ylim)
        axes[0].set_title('Primary Peaks Distance\nfrom a picked peak (STD)')

        axes[1].bar(
            np.arange(self.refl_counts.max() + 1), np.bincount(self.refl_counts),
            width=1
            )
        if not max_refl_counts is None:
            ylim = axes[1].get_ylim()
            axes[1].plot([max_refl_counts, max_refl_counts], ylim, color=[0, 0, 0])
            axes[1].set_ylim(ylim)
        axes[1].set_xscale('log')
        axes[1].set_title('Counts per experiment')

        bins = np.linspace(0, 0.005, 1001)
        centers = (bins[1:] + bins[:-1]) / 2
        hist, _ = np.histogram(min_separation_obs, bins=bins)
        axes[2].bar(centers, hist, width=(bins[1] - bins[0]))
        if not min_separation is None:
            ylim = axes[2].get_ylim()
            axes[2].plot([min_separation, min_separation], ylim, color=[0, 0, 0])
            axes[2].set_ylim(ylim)
        axes[2].set_xscale('log')
        axes[2].set_title('Closest peaks per experiment')
        fig.tight_layout()
        plt.show()
        
    def pick_secondary_peaks(self, include_list=[], prominence=30, yscale=None):
        indices = scipy.signal.find_peaks(self.q2_diff_hist, prominence=prominence)
        self.q2_peaks_secondary = []

        fig, axes = plt.subplots(2, 1, figsize=(45, 6), sharex=True)
        axes[0].plot(self.q2_centers, self.q2_hist)
        axes[1].plot(self.q2_diff_centers, self.q2_diff_hist)

        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()
        for p_index, p in enumerate(self.q2_peaks):
            if p_index == 0:
                label = 'Primary Picked'
            else:
                label = None
            axes[0].plot([p, p], ylim0, linestyle='dotted', linewidth=1.5, color=[0, 0, 0])
            axes[1].plot([p, p], ylim1, linestyle='dotted', linewidth=1.5, color=[0, 0, 0], label=label)

        for p_index, p in enumerate(self.q2_diff_centers[indices[0]]):
            if p_index == 0:
                label = 'Secondary Found'
            else:
                label = None
            if p_index in include_list:
                self.q2_peaks_secondary.append(p)
            axes[1].plot([p, p], ylim1, linestyle='dashed', linewidth=1.5, color=[0.8, 0, 0], label=label)
            axes[1].annotate(p_index, xy=(p, 0.9*ylim1[1]))
        axes[0].set_ylim(ylim0)
        axes[1].set_ylim(ylim1)
        axes[0].set_ylabel('Primary Positions')
        axes[1].set_ylabel('Secondary Positions')
        axes[1].set_xlabel('1 / d_spacing ($\mathrm{\AA}$)')
        if yscale == 'log':
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')
        axes[1].legend(loc='upper left', frameon=False)

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.png'))
        plt.show()
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.npy'),
            1/np.sqrt(np.array(self.q2_peaks_secondary))
            )
        print(repr(1/np.sqrt(self.q2_peaks_secondary)))

    def plot_known_unit_cell(self, q2_max=0.5, unit_cell=None, space_group=None):
        if unit_cell is None:
            unit_cell = uctbx.unit_cell(parameters=self.known_unit_cell)
        else:
            unit_cell = uctbx.unit_cell(parameters=unit_cell)
        if space_group is None:
            sym = symmetry(unit_cell=unit_cell, space_group=self.known_space_group)
        else:
            sym = symmetry(unit_cell=unit_cell, space_group=space_group)

        hkl_list = cctbx.miller.build_set(sym, False, d_min=1/np.sqrt(q2_max))
        dspacings = unit_cell.d(hkl_list.indices()).as_numpy_array()
        q2_known = 1 / dspacings**2
   
        fig, axes = plt.subplots(1, 1, figsize=(40, 4), sharex=True)
        axes.plot(self.q2_centers, self.q2_hist)
        ylim0 = axes.get_ylim()
        for p in q2_known:
            axes.plot([p, p], ylim0, color=[0.8, 0, 0], linestyle='dotted', linewidth=2)
        for p_index, p in enumerate(self.q2_peaks):
            axes.plot([p, p], [ylim0[0], 0.75*ylim0[1]], color=[0, 0, 0], linestyle='dotted', linewidth=2)
        axes.set_ylim(ylim0)
        fig.tight_layout()
        plt.show()

    def make_triplets(self, triplet_peak_indices, delta=1, max_difference=False, min_separation=None, max_refl_counts=None, mask=True):
        if max_refl_counts is None:
            max_refl_counts = np.inf
        start = 0
        triplet_keys = []
        triplet_peaks = self.q2_peaks[triplet_peak_indices]
        triplet_breadths = np.abs(self.q2_breadths[triplet_peak_indices])
        for p0 in range(triplet_peaks.size):
            for p1 in range(p0, triplet_peaks.size):
                triplet_keys.append((triplet_peak_indices[p0], triplet_peak_indices[p1]))
        print(triplet_keys)
        self.triplets = dict.fromkeys(triplet_keys)
        for key in triplet_keys:
            self.triplets[key] = []
        for expt_index, refl_counts in enumerate(self.refl_counts):
            q2_obs, s1, start = slice_refls(self.q2_obs, self.s1, start, refl_counts, self.refl_mask, mask)
            # If there are too many refls on a frame, it might have multiple lattices.
            if q2_obs.size >= 2 and q2_obs.size < max_refl_counts:
                q = get_scattering_vectors(self.s0[expt_index], s1)
                
                # This removes peaks that are larger than the 1D peak list
                indices = q2_obs < (self.q2_peaks[-1] + delta*self.q2_breadths[-1])
                q2_obs = q2_obs[indices]
                q = q[indices]

                # Only consider peaks close to a peak in the picked peak list.
                if max_difference:
                    min_error = np.min(
                        np.abs(q2_obs[:, np.newaxis] - self.q2_peaks[np.newaxis]) / self.q2_breadths,
                        axis=1
                        )
                    indices = min_error < delta
                    q2_obs = q2_obs[indices]
                    q = q[indices]

                if q2_obs.size > 1:
                    q2_diff_lattice = calc_pairwise_diff(q, metric='euclidean')**2
                    angular_diff_lattice = calc_pairwise_diff(q, q_ref=self.s0[expt_index], metric='angular')
                    indices = np.triu_indices(q.shape[0], k=1)
                    q20_obs = q2_obs[indices[0]]
                    q21_obs = q2_obs[indices[1]]
                    # If there are peaks that are very close, it might be a multiple lattice.
                    if min_separation_check(q, min_separation):
                        q20_triplet_index = np.argmin(
                            np.abs(q20_obs[:, np.newaxis] - triplet_peaks[np.newaxis]) / triplet_breadths[np.newaxis],
                            axis=1
                            )
                        q21_triplet_index = np.argmin(
                            np.abs(q21_obs[:, np.newaxis] - triplet_peaks[np.newaxis]) / triplet_breadths[np.newaxis],
                            axis=1
                            )
                        for pair_index in range(q2_diff_lattice.size):
                            p0 = q20_triplet_index[pair_index]
                            p1 = q21_triplet_index[pair_index]
                            key = (
                                triplet_peak_indices[p0],
                                triplet_peak_indices[p1]
                                ) 
                            check0 = np.logical_and(
                                q20_obs[pair_index] > triplet_peaks[p0] - delta*triplet_breadths[p0],
                                q20_obs[pair_index] < triplet_peaks[p0] + delta*triplet_breadths[p0],
                                )
                            check1 = np.logical_and(
                                q21_obs[pair_index] > triplet_peaks[p1] - delta*triplet_breadths[p1],
                                q21_obs[pair_index] < triplet_peaks[p1] + delta*triplet_breadths[p1],
                                )
                            check2 = np.invert(np.isnan(angular_diff_lattice[pair_index]))
                            check3 = np.abs(angular_diff_lattice[pair_index]) < 2.8
                            if check0 and check1 and check2 and check3:
                                if key[0] < key[1]:
                                    q20_triplet_peak = triplet_peaks[p0]
                                    q21_triplet_peak = triplet_peaks[p1]
                                else:
                                    key = (key[1], key[0])
                                    q20_triplet_peak = triplet_peaks[p1]
                                    q21_triplet_peak = triplet_peaks[p0]
                                self.triplets[key].append([
                                    q20_triplet_peak,
                                    q21_triplet_peak,
                                    q2_diff_lattice[pair_index],
                                    angular_diff_lattice[pair_index],
                                    ])
        for key in triplet_keys:
            if len(self.triplets[key]) > 0:
                self.triplets[key] = np.row_stack(self.triplets[key])

    def pick_triplets(self, prominence_factor=5, hkl=None, xnn=None, lattice_system=None, auto=False):
        triplet_keys = list(self.triplets.keys())
        triplets_obs = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for key_index, key in enumerate(triplet_keys):
            if len(self.triplets[key]) > 0:
                triplets_obs_pair = []
                fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=False)
                bins_full = np.linspace(-np.pi, np.pi, 401)
                centers_full = (bins_full[1:] + bins_full[:-1]) / 2
                bins_half = np.linspace(0, np.pi/2, 101)
                centers_half = (bins_half[1:] + bins_half[:-1]) / 2
                
                q2_0 = self.triplets[key][0, 0]
                q2_1 = self.triplets[key][0, 1]
                q2_diff = self.triplets[key][:, 2]
                angles = self.triplets[key][:, 3]

                # Angles are between two points, A & B, on a common plane
                # Angles are between -180 to +180 where the sign comes from the dot product with
                #     the negative beam.
                # "Rotation about A" refer to a 180 degree rotation about the vector between 0 and A.
                # We can always find an angle between A & B such that 0 < angle < 90. This is
                #   because two points cannot fully define the orientation of the crystal.
                #   Inversions of B are valid - the rest of reciprocal space remains the same.
                #   Rotations about A are only valid if there is a mirror symmetry. Otherwise the
                #   rest of reciprocal space is changed.
                # If 0 < angle < 90: keep.
                #   angle_sym = angle
                # If 90 < angle < 180: Invert B, then rotate 180 about A.
                #   angle_sym = np.pi - angle
                # If -90 < angle < 0: Rotate 180 about A
                #   angle_sym = -angle
                # If -180 < angle < -90: Invert B
                #   angle_sym = np.pi + angle
                angles_sym = apply_simple_symmetry(angles)

                # Get the expected angle for peaks in the 1D primary peak list
                angles_peaks = law_of_cosines(q2_0, q2_1, self.q2_peaks)

                # Get the expected angle for peaks in the 1D secondary peak list
                angles_secondary_peaks = []
                if len(self.q2_peaks_secondary) > 0:
                    angles_secondary_peaks = law_of_cosines(q2_0, q2_1, self.q2_peaks_secondary)

                hist_half, _ = np.histogram(angles_sym, bins=bins_half)
                hist_full, _ = np.histogram(angles, bins=bins_full)
                axes.bar(centers_full, hist_full, width=bins_full[1] - bins_full[0], color=colors[0], label='Obs Diff.')
                axes.bar(centers_half, hist_half, width=bins_half[1] - bins_half[0], color=colors[3], alpha=0.5, label='Obs Diff. Sym.')
                ylim = axes.get_ylim()
                ylim = [ylim[0], ylim[1]*1.1]

                # scipy.signal.find_peaks does not find peaks at the first and last index.
                # Padding zeros at the start and end are attempts to pick them up.
                n_pad = 5

                diff_peak_indices, _ = scipy.signal.find_peaks(
                    np.concatenate((np.zeros(n_pad), hist_half, np.zeros(n_pad))),
                    prominence=prominence_factor*(np.std(hist_half) + 1)
                    )
                diff_peak_indices -= n_pad
                diff_peak_indices = diff_peak_indices[diff_peak_indices >= 0]
                diff_peak_indices = diff_peak_indices[diff_peak_indices <= hist_half.size]
                if diff_peak_indices.size > 0:
                    # This gets the peak positions in units of angle
                    diff_peaks = centers_half[diff_peak_indices]
                    weights = hist_half[diff_peak_indices]
                    for p_index, p in enumerate(diff_peaks):
                        selection = np.logical_and(
                            angles_sym > p - 0.02,
                            angles_sym < p + 0.02,
                            )
                        median_angle = np.nanmedian(angles_sym[selection])                        
                        median_difference = q2_0 + q2_1 - 2*np.sqrt(q2_0*q2_1)*np.cos(median_angle)
                        triplets_obs_pair.append([
                            key[0], key[1],
                            median_difference, median_angle,
                            weights[p_index]
                            ])
                        if p_index == 0:
                            label = 'Found Differences'
                        else:
                            label = None
                        axes.plot(
                            [median_angle, median_angle], ylim,
                            color=colors[1], alpha=0.75, label=label, linestyle='dotted'
                            )
                        axes.annotate(
                            f'{np.round(median_angle, decimals=3)}',
                            xy=[median_angle, ylim[1]*0.85],
                            )
                    axes.set_ylim(ylim)
                if hkl is None:
                    axes.set_ylabel(
                        str(key)
                        + f'\n{np.round(q2_0, decimals=5)} {np.round(q2_1, decimals=5)}'
                        )
                else:
                    from Utilities import get_hkl_matrix
                    axes.set_ylabel(
                        str(key)
                        + f'\n{np.round(q2_0, decimals=5)} {np.round(q2_1, decimals=5)}'
                        + f'\n{hkl[key[0]]}, {hkl[key[1]]}'
                        )
                    
                    mi_sym = [
                        np.array([1, 1, 1]),
                        np.array([1, 1, -1]),
                        np.array([1, -1, 1]),
                        np.array([-1, 1, 1]),
                        ]
                    permutations = [np.array([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        ])]
                    if lattice_system == 'cubic':
                        permutations.append(np.array([
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            ]))
                        permutations.append(np.array([
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            ]))
                        permutations.append(np.array([
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            ]))
                    elif lattice_system in ['tetragonal', 'hexagonal']:
                        permutations.append(np.array([
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            ]))
                        permutations.append(np.array([
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            ]))
                        permutations.append(np.array([
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            ]))
                    make_label = True
                    for i in range(len(mi_sym)):
                        for j in range(len(mi_sym)):
                            for k0 in range(len(permutations)):
                                for k1 in range(len(permutations)):
                                    hkl0 = np.matmul(permutations[k0], mi_sym[i]*hkl[key[0]])
                                    hkl1 = np.matmul(permutations[k1], mi_sym[j]*hkl[key[1]])
                                    hkl_diff = hkl0 - hkl1
                                    hkl2_diff = get_hkl_matrix(hkl_diff[np.newaxis], lattice_system)
                                    q2_diff_calc = np.sum(xnn * hkl2_diff, axis=1)[0]
                                    angle_diff_calc = law_of_cosines(q2_0, q2_1, q2_diff_calc, clip=False)
                                    if make_label:
                                        label = 'Pred Diff'
                                    else:
                                        label = None
                                    axes.plot(
                                        [angle_diff_calc, angle_diff_calc], [ylim[0], 0.15*ylim[1]],
                                        color=[0, 0.8, 0], label=label
                                        )
                                    axes.annotate(
                                        np.round(angle_diff_calc, decimals=3),
                                        xy=[angle_diff_calc, ylim[1]*0.2],
                                        )
                                    make_label = False
                for p_index, peak_angle in enumerate(angles_peaks):
                    if not np.isnan(peak_angle):
                        if p_index == 0:
                            label = 'Obs Peaks'
                        else:
                            label = None
                        axes.plot(
                            [peak_angle, peak_angle], [ylim[0], 0.1*ylim[1]],
                            color=[0, 0, 0], label=label
                            )
                for p_index, peak_angle in enumerate(angles_secondary_peaks):
                    if not np.isnan(peak_angle):
                        if p_index == 0:
                            label = 'Secondary Peaks'
                        else:
                            label = None
                        axes.plot(
                            [peak_angle, peak_angle], [ylim[0], 0.1*ylim[1]],
                            color=[0, 0, 0], linestyle='dashed', label=label
                            )
                axes.set_ylim()
                axes.legend(frameon=False)
                fig.tight_layout()
                plt.show(block=False)
                for index in range(len(triplets_obs_pair)):
                    if auto:
                        print(f'Found triplet at angle {triplets_obs_pair[index][3]:0.4f}')
                        if triplets_obs_pair[index][3] < 0.05:
                            triplets_obs_pair[index][3] = 0
                        elif triplets_obs_pair[index][3] > np.pi/2 - 0.05:
                            triplets_obs_pair[index][3] = np.pi/2
                        triplets_obs.append(triplets_obs_pair[index])
                    else:
                        print(f'Found triplet at angle {triplets_obs_pair[index][3]:0.4f}')
                        accept = input(f'   Accept with y, specify angle with 0 and 90$^o$ with 90')
                        if accept == 'y':
                            triplets_obs.append(triplets_obs_pair[index])
                        elif accept in [0, 0.0, '0']:
                            triplets_obs_pair[index][3] = 0
                            triplets_obs.append(triplets_obs_pair[index])
                        elif accept in [90, 90.0, '90']:
                            triplets_obs_pair[index][3] = np.pi/2
                            triplets_obs.append(triplets_obs_pair[index])
                plt.close()
        self.triplets_obs = np.stack(triplets_obs)
