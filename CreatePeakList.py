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
    def __init__(self, tag, save_to_directory=None, load_combined=False, runs=None, run_limits=None, run_limits_sacla=None, input_path_template=None, suffix='_strong.expt', min_reflections_per_experiment=3, known_unit_cell=None, known_space_group=None):
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
        if save_to_directory is None:
            self.save_to_directory = os.path.join(os.getcwd(), self.tag)
        else:
            self.save_to_directory = os.path.join(save_to_directory, self.tag)
        if not os.path.exists(self.save_to_directory):
            os.mkdir(self.save_to_directory)
        
        self.expt_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined.expt')
        self.refl_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined.refl')    
        if self.load_combined == False:
            self._combine_expt_refl_files()
            self._parse_refl_file()
        else:
            self.s0 = np.load(os.path.join(
                self.save_to_directory, f'{self.tag}_s0.npy'
                ))
            self.s1_primary = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_s1_primary.npy'),
                )
            self.q2_primary = np.load(os.path.join(
                self.save_to_directory, f'{self.tag}_q2_primary.npy'
                ))
            self.q2_secondary = np.load(os.path.join(
                self.save_to_directory, f'{self.tag}_q2_secondary.npy'
                ))
            self.refl_counts = np.load(os.path.join(
                self.save_to_directory, f'{self.tag}_refl_counts.npy'
                ))
            self.min_separation = np.load(os.path.join(
                self.save_to_directory, f'{self.tag}_min_separation.npy'
                ))
        self.known_unit_cell = known_unit_cell
        self.known_space_group = known_space_group
        self.error = None

    def output_json(self, note=None, extra_file_name=None):
        output = {
            'primary_peaks': self.q2_primary_picked,
            'secondary_peaks': self.secondary_peaks,
            'primary_hist': np.column_stack((self.primary_centers_q2, self.primary_hist_q2)),
            'secondary_hist': np.column_stack((self.secondary_centers_q2, self.secondary_hist_q2_difference)),
            'broadening_params': self.broadening_params,
            'error': self.error,
            'note': note,
            }
        if extra_file_name is None:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info.json')
        else:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info_{extra_file_name}.json')
        pd.Series(output).to_json(file_name)

    def _combine_expt_refl_files(self):
        expt_file_names = []
        refl_file_names = []
        for run in self.runs:
            if type(run) == str:
                input_path = self.input_path_template.replace('!!!!', run)
            else:
                input_path = self.input_path_template.replace('!!!!', f'{run:04d}')
            for file_name in os.listdir(input_path):
                if file_name.endswith(self.suffix):
                    expt_file_name = os.path.join(input_path, file_name)
                    refl_file_name = os.path.join(input_path, file_name.replace('.expt', '.refl'))
                    if os.path.exists(expt_file_name) and os.path.exists(refl_file_name):
                        expt_file_names.append(expt_file_name)
                        refl_file_names.append(refl_file_name)
        if len(expt_file_names) == 0:
            print(input_path)
            print(self.suffix)
            assert False
        command = ['dials.combine_experiments']
        command += expt_file_names
        command += refl_file_names
        command += [
            'reference_from_experiment.detector=0',
            f'min_reflections_per_experiment={self.min_reflections_per_experiment}',
            f'output.experiments_filename={self.tag}_combined.expt',
            f'output.reflections_filename={self.tag}_combined.refl',
            ]
        log_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combine_experiments.log')
        with open(log_file_name, 'w') as log_file:
            subprocess.run(command, cwd=self.save_to_directory, stdout=log_file)

    def _get_s1_from_xyz(self, panel, xyz, wavelength):
        s1 = flumpy.to_numpy(
                panel.get_lab_coord(panel.pixel_to_millimeter(flex.vec2_double(
                    flex.double(xyz[:, 0].ravel()),
                    flex.double(xyz[:, 1].ravel())
                )))
            )
        s1[:, :2] += self.delta
        # s1 is the vector going from the interation point to the peak with magnitude 1/wavelength
        s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
        return s1_normed, s1

    def _get_q2_from_xyz(self, panel, xyz, s0):
        return np.array([1 / panel.get_resolution_at_pixel(s0, xyz[i][0:2])**2 for i in range(len(xyz))])

    def _get_q2_spacing(self, s1, s0, wavelength):
        dot_product = np.matmul(s1, s0)
        magnitudes = np.linalg.norm(s1, axis=1) * np.linalg.norm(s0)
        theta2 = np.arccos(dot_product / magnitudes)
        return ((2 * np.sin(theta2 / 2)) / wavelength)**2

    def _parse_refl_file(self, update=False):
        expts = ExperimentList.from_file(self.expt_file_name, check_format=False)
        refls = flex.reflection_table.from_file(self.refl_file_name)
        q2_primary = []
        q2_secondary = []
        rlp_primary = []
        s1_primary = []
        s0_expt = []
        refl_counts = []
        min_separation = []
        if update == False:
            self.delta = np.zeros(2)
        for expt_index, expt in enumerate(expts):
            wavelength = expt.beam.get_wavelength()
            s0 = expt.beam.get_s0() #|s0| = 1/wavelength
            s0_expt.append(s0)
            refls_expt = refls.select(refls['id'] == expt_index)
            refl_counts.append(len(refls_expt))
            s1_normed_primary_lattice = []
            s1_primary_lattice = []
            q2_check_lattice = []
            for panel_index, panel in enumerate(expt.detector):
                refls_panel = refls_expt.select(refls_expt['panel'] == panel_index)
                if len(refls_panel) > 0:
                    s1_normed, s1 = self._get_s1_from_xyz(
                        panel, 
                        flumpy.to_numpy(refls_panel['xyzobs.px.value']), 
                        wavelength,
                        )
                    s1_normed_primary_lattice.append(s1_normed)
                    s1_primary_lattice.append(s1)
                    q2_check_lattice.append(self._get_q2_from_xyz(panel, refls_panel['xyzobs.px.value'], s0))
            s1_normed_primary_lattice = np.row_stack(s1_normed_primary_lattice)
            rlp_primary_lattice = s1_normed_primary_lattice - np.array(s0)[np.newaxis]
            s1_primary.append(np.row_stack(s1_primary_lattice))
            q2_primary_experiment = self._get_q2_spacing(s1_normed_primary_lattice, s0, wavelength)
            q2_rlp = np.linalg.norm(rlp_primary_lattice, axis=1)**2
            #print(np.column_stack((q2_primary_experiment, np.concatenate(q2_check_lattice), q2_rlp)))
            q2_primary.append(np.column_stack((
                q2_primary_experiment, expt_index*np.ones(q2_primary_experiment.size)
                )))
            
            N_primary = rlp_primary_lattice.shape[0]
            N_secondary = int(N_primary**2 / 2 - N_primary / 2)
            s1_secondary_lattice_difference = np.zeros((N_primary, N_primary, 3))
            for i in range(3):
                s1_secondary_lattice_difference[:, :, i] = rlp_primary_lattice[:, i][np.newaxis] - rlp_primary_lattice[:, i][:, np.newaxis]
            indices = np.triu_indices(N_primary, k=1)
            s1_secondary_lattice_difference = s1_secondary_lattice_difference[indices[0], indices[1], :]

            q2_secondary_lattice_difference = np.linalg.norm(s1_secondary_lattice_difference, axis=1)**2
            min_separation.append(q2_secondary_lattice_difference.min())
            q2_secondary_lattice = np.zeros((N_secondary, 5))
            q2_secondary_lattice[:, 0] = q2_primary[-1][indices[0], 0]
            q2_secondary_lattice[:, 1] = q2_primary[-1][indices[1], 0]
            q2_secondary_lattice[:, 2] = len(refls_expt)
            q2_secondary_lattice[:, 3] = q2_secondary_lattice_difference.min()
            q2_secondary_lattice[:, 4] = q2_secondary_lattice_difference
            q2_secondary.append(q2_secondary_lattice)

        self.refl_counts = np.array(refl_counts, dtype=int)
        self.min_separation = np.array(min_separation)
        self.s0 = np.row_stack(s0_expt)
        self.s1_primary = np.row_stack(s1_primary)
        self.q2_primary = np.row_stack(q2_primary)
        self.q2_secondary = np.row_stack(q2_secondary)
        print(f'{expt_index + 1} experiments')
        print(f'{self.q2_primary.shape[0]} reflections')
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_s0.npy'),
            self.s0
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_s1_primary.npy'),
            self.s1_primary
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_q2_primary.npy'),
            self.q2_primary
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_q2_secondary.npy'),
            self.q2_secondary
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_refl_counts.npy'),
            self.refl_counts
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_min_separation.npy'),
            self.min_separation
            )

    def optimize_beam_center(self, primary_peak_indices):
        def get_q2_spacing(s1, s0):
            wavelength = 1/np.linalg.norm(s0)
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
    
        def functional_angle(x0, s1_list, s0_list):
            delta = x0[:2]
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(x0[2]), -np.sin(x0[2])],
                [0, np.sin(x0[2]), np.cos(x0[2])]
                ])
            Ry = np.array([
                [np.cos(x0[3]), 0, np.sin(x0[3])],
                [0, 1, 0],
                [-np.sin(x0[3]), 0, np.cos(x0[3])]
                ])
            L = 0
            for peak_index in range(len(s1_list)):
                s1 = s1_list[peak_index]
                s0 = s0_list[peak_index]
                q2_calc = np.zeros(s1.shape[0])
                for i in range(s1.shape[0]):
                    s0_rot = Ry @ Rx @ s0[i]
                    s1_delta = s1[i].copy()
                    s1_delta[:2] += delta
                    q2_calc[i] = get_q2_spacing(s1_delta, s0_rot)
                L += q2_calc.std()
            return L
    
        s1 = []
        s0 = []
        for peak_index in primary_peak_indices:
            differences = np.abs(self.q2_primary[:, 0][:, np.newaxis] - self.q2_primary_picked[peak_index])
            indices = differences[:, 0] < 0.002
            s1.append(self.s1_primary[indices, :])
            s0.append(self.s0[self.q2_primary[indices, 1].astype(int)])
    
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
        self.delta = results.x[:2]
        self._parse_refl_file(update=True)

        # This optimizes the detector angle.
        #initial_simplex = np.array([
        #    [0.05, 0.025, 0.01, 0.025],
        #    [0.025, -0.05, 0.025, -0.01],
        #    [0.001, -0.01, 0.001, -0.001],
        #    [-0.01, 0.001, -0.001, 0.001],
        #    [-0.025, -0.05, -0.025, 0.01],
        #    ])
        #results = scipy.optimize.minimize(
        #    fun=functional_angle,
        #    x0=[0, 0, 0, 0],
        #    args=(s1, s0),
        #    method='Nelder-Mead',
        #    options={'initial_simplex': initial_simplex}
        #    )
        #self.delta = results.x[:2]
        #self.angle_x = results.x[2]
        #self.angle_y = results.x[3]
        #print(results)
        return None

    def make_primary_histogram(self, n_bins=1000, d_min=60, d_max=3.5):
        self.d_min = d_min
        self.d_max = d_max
        self.q2_min = 1 / self.d_min**2
        self.q2_max = 1 / self.d_max**2
        self.primary_bins_q2 = np.linspace(self.q2_min, self.q2_max, n_bins + 11)
        self.primary_centers_q2 = (self.primary_bins_q2[1:] + self.primary_bins_q2[:-1]) / 2
        self.primary_hist_q2, _ = np.histogram(self.q2_primary[:, 0], bins=self.primary_bins_q2)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_primary_hist.npy'),
            np.column_stack((self.primary_hist_q2, self.primary_centers_q2))
            )

    def pick_primary_peaks(self, exclude_list=[], exclude_max=100, add_peaks=[], shift={}, prominence=30, plot_kapton_peaks=False, yscale=None):
        found_peak_indices = scipy.signal.find_peaks(self.primary_hist_q2, prominence=prominence)
        found_peaks = self.primary_centers_q2[found_peak_indices[0]]
        found_peaks = np.delete(found_peaks[:exclude_max], exclude_list)
        primary_peaks = np.sort(np.concatenate((found_peaks, add_peaks)))
    
        fig, axes = plt.subplots(1, 1, figsize=(30, 6), sharex=True)
        axes.plot(self.primary_centers_q2, self.primary_hist_q2, label='Histogram')
        for p_index, p in enumerate(primary_peaks):
            if p_index in shift.keys():
                primary_peaks[p_index] += shift[p_index]
            if p in add_peaks:
                color = [0.8, 0, 0]
            else:
                color = [0, 0, 0]
            axes.plot(
                [p, p], [0, self.primary_hist_q2.max()],
                linestyle='dotted', linewidth=1, color=color
                )
            axes.annotate(p_index, xy=(p-0.001, (1-p_index/primary_peaks.size) * self.primary_hist_q2.max()))
        if plot_kapton_peaks:
            kapton_peaks = [15.25, 7.625, 5.083333333, 3.8125, 3.05]
            for p in kapton_peaks:
                if p > self.d_max:
                    axes.plot([1/p**2, 1/p**2], [0, self.primary_hist_q2.max()], linestyle='dotted', linewidth=2, color=[0, 0.7, 0], label='Kapton Peaks')
        axes.set_xlabel('q2 (1/$\mathrm{\AA}^2$')
        if yscale == 'log':
            axes.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_primary_peaks.png'))
        plt.show()
        self.q2_primary_picked = primary_peaks
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_primary_peaks.npy'),
            1/np.sqrt(self.q2_primary_picked)
            )
        print(repr(1/np.sqrt(self.q2_primary_picked)))

    def fit_peaks(self, n_max, ind_peak_indices, fit_shift=True):
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

        for index, peak_index in enumerate(ind_peak_indices):
            loc = np.searchsorted(self.primary_centers_q2, self.q2_primary_picked[peak_index])
            delta = 10
            low = max(0, loc - delta)
            high = min(self.primary_centers_q2.size, loc + delta)
            results = scipy.optimize.least_squares(
                basic_gaussian_loss,
                x0=(1, 0.00001, self.q2_primary_picked[peak_index]),
                args=(self.primary_centers_q2[low: high], self.primary_hist_q2[low: high])
                )
            ind_amplitudes[index] = np.abs(results.x[0])
            ind_breadths[index] = np.abs(results.x[1])
            ind_q2_centers[index] = np.abs(results.x[2])

        broadening_params_polyfit = np.polyfit(x=ind_q2_centers, y=ind_breadths, deg=1)
        self.broadening_params = np.array([broadening_params_polyfit[1], broadening_params_polyfit[0]])
        
        mask = np.ones(n_max, dtype=bool)
        amplitudes = np.zeros(n_max)
        q2_centers = self.q2_primary_picked[:n_max].copy()
        for index, peak_index in enumerate(ind_peak_indices):
            if peak_index < n_max:
                mask[peak_index] = False
                amplitudes[peak_index] = ind_amplitudes[index]
                q2_centers[peak_index] = ind_q2_centers[index]

        # Fit breadths and amplitudes
        max_index = np.searchsorted(self.primary_centers_q2, self.q2_primary_picked[n_max]) + 20
        results = scipy.optimize.least_squares(
            fit_loss,
            x0=amplitudes[mask],
            jac=fit_jac,
            args=(
                amplitudes,
                q2_centers,
                mask,
                self.broadening_params,
                self.primary_hist_q2[:max_index],
                self.primary_centers_q2[:max_index],
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
                    self.primary_hist_q2[:max_index],
                    self.primary_centers_q2[:max_index],
                    'amplitudes_centers'
                    ),
                method='lm'
                )
            print(results)
            amplitudes[mask] = results.x[:mask.sum()]
            q2_centers[mask] = results.x[mask.sum():]
            q2_primary_picked_original = self.q2_primary_picked[:n_max].copy()
            self.q2_primary_picked[:n_max] = q2_centers

        I_calc = get_I_calc(amplitudes, q2_centers, self.broadening_params, self.primary_centers_q2[:max_index])
        fig, axes = plt.subplots(1, 1, figsize=(30,  8), sharex=True)
        axes.plot(self.primary_centers_q2[:max_index], self.primary_hist_q2[:max_index])
        axes.plot(self.primary_centers_q2[:max_index], I_calc)
        ylim = axes.get_ylim()
        for peak_index, p in enumerate(self.q2_primary_picked[:n_max]):
            if p in ind_peak_indices:
                color = [0.8, 0, 0]
            else:
                color = [0, 0, 0]
            axes.plot([p, p], ylim, color=color, linestyle='dotted')
        if fit_shift:
            for i in range(n_max):
                shift = self.q2_primary_picked[i] - q2_primary_picked_original[i]
                axes.annotate(
                    f'{shift:0.5f}',
                    xy=(self.q2_primary_picked[i], 0.9 * ylim[1]),
                    rotation=90
                    )
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.plot(self.q2_primary_picked[ind_peak_indices], ind_breadths, marker='.')
        axes.plot(self.q2_primary_picked[ind_peak_indices], np.polyval(broadening_params_polyfit, self.q2_primary_picked[ind_peak_indices]))
        plt.show()

    def create_secondary_peaks(self, q2_max=None, max_difference=None, max_refl_count=None, min_separation=None, n_bins=800):
        def get_differences(q2_primary_picked, q2_secondary_source):
            ss_indices = np.searchsorted(q2_primary_picked, q2_secondary_source)
            low = ss_indices == 0
            high = ss_indices == q2_primary_picked.size
            middle = np.logical_and(~low, ~high)
            differences = np.zeros(q2_secondary_source.size)
            differences[low] = np.abs(q2_secondary_source[low] - q2_primary_picked[0])
            differences[high] = np.abs(q2_secondary_source[high] - q2_primary_picked[-1])
            differences[middle] = np.column_stack((
                np.abs(q2_secondary_source[middle] - q2_primary_picked[ss_indices[middle] - 1]),
                np.abs(q2_secondary_source[middle] - q2_primary_picked[ss_indices[middle]])
                )).min(axis=1)
            return differences
            
        # Only use secondary differences if they came from peaks lower in resolution than
        # The highest resolution picked peak.
        # This is unnecessary given the filtering based on difference, but helps speed up the execution
        if not q2_max is None:
            resolution_indices = np.max(self.q2_secondary[:, :2], axis=1) < q2_max
        else:
            resolution_indices = np.ones(self.q2_secondary.shape[0], dtype=bool)
    
        # These lines only select peaks close to a peak that was picked as a primary peak
        differences_1 = get_differences(self.q2_primary_picked, self.q2_secondary[resolution_indices, 0])
        differences_2 = get_differences(self.q2_primary_picked, self.q2_secondary[resolution_indices, 1])
        differences = np.column_stack((differences_1, differences_2)).max(axis=1)
        if not max_difference is None:
            difference_indices = differences < max_difference
        else:
            difference_indices = np.ones(resolution_indices.sum(), dtype=bool)

        if not max_refl_count is None:
            max_refl_indices = self.q2_secondary[resolution_indices, 2] < max_refl_count
        else:
            max_refl_indices = np.ones(resolution_indices.sum(), dtype=bool)

        if not min_separation is None:
            min_separation_indices = self.q2_secondary[resolution_indices, 3] > min_separation
        else:
            min_separation_indices = np.ones(resolution_indices.sum(), dtype=bool)
            
        filtered_indices = np.all(np.column_stack((
            difference_indices, max_refl_indices, min_separation_indices
            )), axis=1
            )
        
        self.q2_secondary_filtered = self.q2_secondary[resolution_indices][filtered_indices, 4]
    
        self.secondary_bins_q2 = np.linspace(0.00000001, self.q2_max, n_bins + 1)
        self.secondary_centers_q2 = (self.secondary_bins_q2[1:] + self.secondary_bins_q2[:-1]) / 2
        self.secondary_hist_q2_difference_unfiltered, _ = np.histogram(self.q2_secondary[:, 4], bins=self.secondary_bins_q2)
        self.secondary_hist_q2_difference, _ = np.histogram(self.q2_secondary_filtered, bins=self.secondary_bins_q2)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_secondary_hist_difference.npy'),
            np.column_stack((self.secondary_hist_q2_difference, self.secondary_centers_q2))
            )

        refl_counts_dist = np.bincount(self.refl_counts)
        edges = np.linspace(0, 0.005, 100)
        hist, _ = np.histogram(differences, bins=edges)
        d_centers = (edges[1:] + edges[:-1]) / 2
        fig, axes = plt.subplots(1, 3, figsize=(7, 3))
        axes[0].plot(d_centers, hist)
        if not max_difference is None:
            ylim = axes[0].get_ylim()
            axes[0].plot([max_difference, max_difference], ylim, color=[0, 0, 0])
            axes[0].set_ylim(ylim)
        axes[0].set_title('Primary Peaks Distance\nfrom a picked peak')

        axes[1].bar(np.arange(refl_counts_dist.size), refl_counts_dist, width=1)
        if not max_refl_count is None:
            ylim = axes[1].get_ylim()
            axes[1].plot([max_refl_count, max_refl_count], ylim, color=[0, 0, 0])
            axes[1].set_ylim(ylim)
        axes[1].set_title('Counts per experiment')

        bins = np.linspace(0, 0.005, 1001)
        centers = (bins[1:] + bins[:-1]) / 2
        hist, _ = np.histogram(self.min_separation, bins=bins)
        axes[2].bar(centers, hist, width=(bins[1] - bins[0]))
        if not min_separation is None:
            ylim = axes[2].get_ylim()
            axes[2].plot([min_separation, min_separation], ylim, color=[0, 0, 0])
            axes[2].set_ylim(ylim)
        axes[2].set_title('Closest peaks per experiment')
        fig.tight_layout()
        plt.show()
        
    def pick_secondary_peaks(self, include_list=[], prominence=30, yscale=None):
        indices = scipy.signal.find_peaks(self.secondary_hist_q2_difference, prominence=prominence)
        self.secondary_peaks = []

        fig, axes = plt.subplots(2, 1, figsize=(45, 6), sharex=True)
        axes[0].plot(self.primary_centers_q2, self.primary_hist_q2)
        axes[1].plot(self.secondary_centers_q2, self.secondary_hist_q2_difference_unfiltered, label='Difference - unfiltered')
        axes[1].plot(self.secondary_centers_q2, self.secondary_hist_q2_difference, label='Difference')

        ylim0 = axes[0].get_ylim()
        ylim1 = [0.1, self.secondary_hist_q2_difference.max() + 10]
        for p_index, p in enumerate(self.q2_primary_picked):
            if p_index == 0:
                label = 'Primary Picked'
            else:
                label = None
            axes[0].plot([p, p], [0.1, ylim0[1]], linestyle='dotted', linewidth=1.5, color=[0, 0, 0])
            axes[1].plot([p, p], ylim1, linestyle='dotted', linewidth=1.5, color=[0, 0, 0], label=label)

        for p_index, p in enumerate(self.secondary_centers_q2[indices[0]]):
            if p_index == 0:
                label = 'Secondary Found'
            else:
                label = None
            if p_index in include_list:
                self.secondary_peaks.append(p)
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
            1/np.sqrt(np.array(self.secondary_peaks))
            )
        print(repr(1/np.sqrt(self.secondary_peaks)))

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
        self.error = np.min(np.abs(self.q2_primary_picked[:, np.newaxis] - q2_known[np.newaxis]), axis=1)
    
        fig, axes = plt.subplots(2, 1, figsize=(40, 6), sharex=True)
        axes[0].plot(self.primary_centers_q2, self.primary_hist_q2)
        axes[1].plot(self.secondary_centers_q2, self.secondary_hist_q2_difference)
        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()
        for p in q2_known:
            axes[0].plot([p, p], ylim0, color=[0.8, 0, 0], linestyle='dotted', linewidth=2)
            axes[1].plot([p, p], ylim1, color=[0.8, 0, 0], linestyle='dotted', linewidth=2)
        for p_index, p in enumerate(self.q2_primary_picked):
            axes[0].plot([p, p], [ylim0[0], 0.75*ylim0[1]], color=[0, 0, 0], linestyle='dotted', linewidth=2)
            axes[0].annotate(
                f'{self.error[p_index]:0.5f}',
                xy=(self.q2_primary_picked[p_index], 0.7 * ylim0[1]),
                rotation=90
                )
        for p_index, p in enumerate(self.secondary_peaks):
            axes[1].plot([p, p], [ylim1[0], 0.75*ylim1[1]], color=[0, 0, 0], linestyle='dotted', linewidth=2)
            axes[1].annotate(
                f'{self.error[p_index]:0.5f}',
                xy=(self.q2_primary_picked[p_index], 0.7 * ylim1[1]),
                rotation=90
                )
        axes[0].set_ylim(ylim0)
        axes[1].set_ylim(ylim1)
        axes[1].set_xlim([0, q2_max])
        fig.tight_layout()
        plt.show()
    
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.plot(self.q2_primary_picked, self.error)
        plt.show()

