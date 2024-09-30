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
import sklearn.metrics
import subprocess


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
        known_unit_cell=None, 
        known_space_group=None,
        ):

        if not run_limits_sacla is None:
            self.runs = []
            for run_index in range(run_limits_sacla[0], run_limits_sacla[1] + 1):
                for sub_run_index in range(3):
                    self.runs.append(f'{run_index}-{sub_run_index}')
        elif not run_limits is None:
            self.runs = np.arange(run_limits[0], run_limits[1] + 1)
        else:
            self.runs = runs
        self.min_reflections_per_experiment = min_reflections_per_experiment
        self.input_path_template = input_path_template
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
        self.known_unit_cell = known_unit_cell
        self.known_space_group = known_space_group
        self.error = None
        self.triplets_obs = None

    def _run_combine_experiments(self, expt_file_names, refl_file_names, run_str):
        command = ['dials.combine_experiments']
        command += expt_file_names
        command += refl_file_names
        command += [
            'reference_from_experiment.detector=0',
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
        for run in self.runs:
            expt_file_names_run = []
            refl_file_names_run = []
            if type(run) == str:
                run_str = run
            else:
                run_str = f'{run:04d}'
            input_path = self.input_path_template.replace('!!!!', run_str)
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
                        expt_file_names.append(os.path.join(
                            self.save_to_directory, f'{self.tag}_combined_{run_str}.expt'
                            ))
                        refl_file_names.append(os.path.join(
                            self.save_to_directory, f'{self.tag}_combined_{run_str}.refl'
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

    def make_histogram(self, n_bins=1000, d_min=60, d_max=3.5, q2_min=None, q2_max=None):
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
        self.q2_hist, _ = np.histogram(self.q2_obs, bins=self.q2_bins)

    def pick_peaks(self, exclude_list=[], exclude_max=20, add_peaks=[], shift={}, prominence=30, plot_kapton_peaks=False, yscale=None):
        found_peak_indices = scipy.signal.find_peaks(self.q2_hist, prominence=prominence)
        found_peaks = self.q2_centers[found_peak_indices[0]]
        found_peaks = np.delete(found_peaks[:exclude_max], exclude_list)
        peaks = np.sort(np.concatenate((found_peaks, add_peaks)))
    
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
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

    def optimize_beam_center(self, primary_peak_indices):
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
        for peak_index in primary_peak_indices:
            differences = np.abs(self.q2_obs - self.q2_peaks[peak_index])
            indices = differences < 3*self.q2_breadths[peak_index]
            s1.append(self.s1[indices])
            s0.append(self.s0[self.expt_indices[indices]])
    
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
        self.q2_obs = np.concatenate(q2)

    """
    def filter_peaks(self, max_difference=0.001, n_peaks=20):
        # assign peaks and get distances
        differences_all = np.abs(self.q2_primary[:, 0][:, np.newaxis] - self.q2_primary_picked[np.newaxis, :n_peaks + 1])
        assignment = np.argmin(differences_all, axis=1)
        differences = np.take_along_axis(differences_all, assignment[:, np.newaxis], axis=1)[:, 0]
        out_of_range = assignment == n_peaks

        fig, axes = plt.subplots(1, 2, figsize=(5, 3))
        axes[0].hist(differences[~out_of_range], bins=100)
        ylim = axes[0].get_ylim()
        axes[0].set_ylim(ylim)
        axes[0].plot([max_difference, max_difference], ylim, color=[0.8, 0, 0])
        axes[1].hist(assignment[~out_of_range], bins=n_peaks)
        fig.tight_layout()
        plt.show()
    
        close_indices = differences < max_difference
        q2_primary = self.q2_primary[close_indices]
        assignment = assignment[close_indices]
        
        occurance_frequency = np.zeros((n_peaks, n_peaks))
        for peak0_index, peak0 in enumerate(self.q2_primary_picked[:n_peaks]):
            experiments = q2_primary[assignment == peak0_index, 1]
            experiments = np.unique(experiments)
            n_experiments_peak0 = experiments.size
            if n_experiments_peak0 > 0:
                common_experiment_indices = np.isin(q2_primary[:, 1], experiments)
                assignment_common_experiment = assignment[common_experiment_indices]
                q2_primary_common_experiment = q2_primary[common_experiment_indices]
                for peak1_index, peak1 in enumerate(self.q2_primary_picked[:n_peaks]):
                    peak1_common_experiments = q2_primary_common_experiment[assignment_common_experiment == peak1_index, 1]
                    n_experiments_peak1 = np.unique(peak1_common_experiments).size
                    occurance_frequency[peak0_index, peak1_index] = n_experiments_peak1 / n_experiments_peak0
    
        fig, axes = plt.subplots(2, 1, figsize=(7, 7))
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=occurance_frequency)
        disp.plot(include_values=False, ax=axes[0])
        axes[0].set_xlabel('Probability Peak Y also occurs')
        axes[0].set_ylabel('Given Peak X occurs')
        axes[0].set_title('Frequency of common occurances')
        axes[1].plot(self.primary_centers_q2, self.primary_hist_q2)
        ylim = axes[1].get_ylim()
        for p_index, p in enumerate(self.q2_primary_picked[:n_peaks]):
            axes[1].plot([p, p], ylim, color=[0.8, 0, 0], linewidth=1, linestyle='dotted')
            axes[1].annotate(p_index, xy=(p, self.primary_hist_q2.max()))
        axes[1].set_ylim(ylim)
        axes[1].set_xlim([self.primary_centers_q2[0], self.q2_primary_picked[n_peaks + 1]])
        plt.show()
    """

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

    def create_secondary_peaks(self, q2_max=None, max_difference=None, max_refl_counts=None, min_separation=None, n_bins=2000):
        start = 0
        q2_diff = []
        min_separation_obs = []
        for expt_index, refl_counts in enumerate(self.refl_counts):
            if max_refl_counts is None or refl_counts < max_refl_counts:
                q2_obs = self.q2_obs[start: start + refl_counts]
                s1 = self.s1[start: start + refl_counts]
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
        plt.show()
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

    def make_triplets(self, triplet_peak_indices, delta=1, max_difference=None, min_separation=None, max_refl_counts=None):
        start = 0
        triplet_keys = []
        triplet_peaks = self.q2_peaks[triplet_peak_indices]
        triplet_breadths = np.abs(self.q2_breadths[triplet_peak_indices])
        for p0 in range(triplet_peaks.size - 1):
            for p1 in range(p0 + 1, triplet_peaks.size):
                triplet_keys.append((triplet_peak_indices[p0], triplet_peak_indices[p1]))
        print(triplet_keys)
        self.triplets = dict.fromkeys(triplet_keys)
        for key in triplet_keys:
            self.triplets[key] = []
        for expt_index, refl_counts in enumerate(self.refl_counts):
            # If there are too many refls on a frame, it might have multiple lattices.
            if max_refl_counts is None or refl_counts < max_refl_counts:
                q2_obs = self.q2_obs[start: start + refl_counts]
                s1 = self.s1[start: start + refl_counts]
                s0 = self.s0[expt_index]
                wavelength = 1 / np.linalg.norm(s0)
                
                # This removes peaks that are larger than the 1D peak list
                indices = q2_obs < (self.q2_peaks[-1] + 3*self.q2_breadths[-1])
                q2_obs = q2_obs[indices]
                s1 = s1[indices]

                # Only consider peaks close to a peak in the picked peak list.
                if not max_difference is None:
                    min_error = np.min(
                        np.abs(q2_obs[:, np.newaxis] - self.q2_peaks[np.newaxis]),
                        axis=1
                        )
                    indices = min_error < max_difference
                    q2_obs = q2_obs[indices]
                    s1 = s1[indices]

                if q2_obs.size > 1:
                    s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
                    q2_diff_all = np.linalg.norm(
                        s1_normed[:, np.newaxis, :] - s1_normed[np.newaxis, :, :],
                        axis=2
                        )**2
                    indices = np.triu_indices(s1.shape[0], k=1)
                    q2_diff_lattice = q2_diff_all[indices[0], indices[1]]
                    q20_obs = q2_obs[indices[0]]
                    q21_obs = q2_obs[indices[1]]
                    # If there are peaks that are very close, it might be a multiple lattice.
                    if min_separation is None or np.min(q2_diff_lattice) > min_separation:
                        q20_triplet_index = np.argmin(
                            np.abs(q20_obs[:, np.newaxis] - triplet_peaks[np.newaxis]),
                            axis=1
                            )
                        q21_triplet_index = np.argmin(
                            np.abs(q21_obs[:, np.newaxis] - triplet_peaks[np.newaxis]),
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
                            if check0 and check1:
                                if key[0] < key[1]:
                                    q20_triplet_peak = triplet_peaks[p0]
                                    q21_triplet_peak = triplet_peaks[p1]
                                else:
                                    key = (key[1], key[0])
                                    q20_triplet_peak = triplet_peaks[p1]
                                    q21_triplet_peak = triplet_peaks[p0]
                                if key[0] != key[1]:
                                    self.triplets[key].append([
                                        q20_triplet_peak, q21_triplet_peak, q2_diff_lattice[pair_index]
                                        ])

            start += refl_counts
        for key in triplet_keys:
            if len(self.triplets[key]) > 0:
                self.triplets[key] = np.row_stack(self.triplets[key])

    def pick_triplets(self, prominence_factor=5, hkl=None, xnn=None, lattice_system=None):
        triplet_keys = list(self.triplets.keys())
        triplets_obs = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for key_index, key in enumerate(triplet_keys):
            if len(self.triplets[key]) > 0:
                triplets_obs_pair = []
                fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=False)
                bins_full = np.linspace(0, np.pi, 201)
                centers_full = (bins_full[1:] + bins_full[:-1]) / 2
                bins_half = np.linspace(0, np.pi/2, 101)
                centers_half = (bins_half[1:] + bins_half[:-1]) / 2
                
                q20 = self.triplets[key][0, 0]
                q21 = self.triplets[key][0, 1]

                cos_angles_peaks = (q20 + q21 - self.q2_peaks) / (2 * np.sqrt(q20 * q21))
                good = np.logical_and(cos_angles_peaks > -1, cos_angles_peaks < 1)
                angles_peaks = np.arccos(cos_angles_peaks[good])
                
                not_overlapping = self.triplets[key][:, 2] > 0.0001
                q0_q1_2 = self.triplets[key][not_overlapping, 2]
                cos_angles = (q20 + q21 - q0_q1_2) / (2 * np.sqrt(q20 * q21))
                angles = np.zeros(cos_angles.size)
                lower = cos_angles > 1
                upper = cos_angles < -1
                both = np.logical_or(lower, upper)
                angles[~both] = np.arccos(np.abs(cos_angles[~both]))
                angles[lower] = 0
                angles[upper] = 0
                hist_half, _ = np.histogram(angles, bins=bins_half)
                hist_full, _ = np.histogram(np.arccos(cos_angles[~both]), bins=bins_full)
                axes.bar(centers_half, hist_half, width=bins_half[1] - bins_half[0], color=colors[0], alpha=0.9, label='Obs Differences')
                axes.bar(centers_full, hist_full, width=bins_full[1] - bins_full[0], color=colors[0], alpha=0.5)
                ylim = axes.get_ylim()
                ylim = [ylim[0], ylim[1]*1.1]
                diff_peak_indices, _ = scipy.signal.find_peaks(hist_half, prominence=prominence_factor*hist_half.std())
                if hist_half[0] > prominence_factor*hist_half.std():
                    diff_peak_indices = np.concatenate([diff_peak_indices, [0]])
                if hist_half[hist_half.size - 1] > prominence_factor*hist_half.std():
                    diff_peak_indices = np.concatenate([diff_peak_indices, [hist_half.size - 1]])
                if diff_peak_indices.size > 0:
                    diff_peaks = centers_half[diff_peak_indices]
                    for p_index, p in enumerate(diff_peaks):
                        selection = np.logical_and(
                            angles > p - 0.02,
                            angles < p + 0.02,
                            )
                        median = np.median(angles[selection])
                        median_difference = q20 + q21 - 2*np.sqrt(q20*q21)*np.cos(median)
                        triplets_obs_pair.append([key[0], key[1], median_difference, median])
                        if p_index == 0:
                            label = 'Found Differences'
                        else:
                            label=None
                        axes.plot(
                            [median, median], ylim, color=colors[1], alpha=0.75, label=label
                            )
                        axes.annotate(
                            np.round(median, decimals=3),
                            xy=[median, ylim[1]*0.85],
                            )
                        axes.plot(
                            [np.pi - median, np.pi - median], ylim, color=colors[1], alpha=0.5
                            )
                        axes.annotate(
                            np.round(np.pi - median, decimals=3),
                            xy=[np.pi - median, ylim[1]*0.85],
                            )
                    axes.set_ylim(ylim)
                if hkl is None:
                    axes.set_ylabel(
                        str(key)
                        + f'\n{np.round(q20, decimals=5)} {np.round(q21, decimals=5)}'
                        )
                else:
                    from Utilities import get_hkl_matrix
                    axes.set_ylabel(
                        str(key)
                        + f'\n{np.round(q20, decimals=5)} {np.round(q21, decimals=5)}'
                        + f'\n{hkl[key[0]]}, {hkl[key[1]]}'
                        )
                    
                    mi_sym = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]
                    make_label = True
                    for i in range(len(mi_sym)):
                        for j in range(len(mi_sym)):
                            hkl_diff = mi_sym[i]*hkl[key[0]] - mi_sym[j]*hkl[key[1]]
                            hkl2_diff = get_hkl_matrix(hkl_diff[np.newaxis], lattice_system)
                            q2_diff_calc = np.sum(xnn * hkl2_diff, axis=1)[0]
                            cos_angle_diff_calc = (q20 + q21 - q2_diff_calc) / (2 * np.sqrt(q20 * q21))
                            if cos_angle_diff_calc > 1:
                                angle_diff_calc = 0
                            elif cos_angle_diff_calc < -1:
                                angle_diff_calc = np.pi
                            else:
                                angle_diff_calc = np.arccos(cos_angle_diff_calc)
                            if make_label:
                                label = 'Pred Diff'
                            else:
                                label = None
                            axes.plot([angle_diff_calc, angle_diff_calc], [ylim[0], 0.5*ylim[1]], color=[0, 0.8, 0], label=label)
                            axes.annotate(
                                np.round(angle_diff_calc, decimals=3),
                                xy=[angle_diff_calc, ylim[1]*0.4],
                                )
                            make_label = False
                for p_index, peak_angle in enumerate(angles_peaks):
                    if p_index == 0:
                        label = 'Obs Peaks'
                    else:
                        label=None
                    axes.plot(
                        [peak_angle, peak_angle], [ylim[0], 0.25*ylim[1]], color=[0, 0, 0], label=label
                        )
                    axes.plot(
                        [np.pi - peak_angle, np.pi - peak_angle], [ylim[0], 0.25*ylim[1]], color=[0, 0, 0]
                        )
                axes.set_ylim()
                axes.legend(frameon=False)
                fig.tight_layout()
                plt.show(block=False)
                for index in range(len(triplets_obs_pair)):
                    accept = input(f'Accept triplet at angle {triplets_obs_pair[index][3]:0.4f} with y') 
                    if accept == 'y':
                        triplets_obs.append(triplets_obs_pair[index])
                plt.close()
        self.triplets_obs = np.stack(triplets_obs)
