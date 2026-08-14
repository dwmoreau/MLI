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



_BL_TO_LATTICE_SYSTEM = {
    'cP': 'cubic',  'cI': 'cubic',  'cF': 'cubic',
    'tP': 'tetragonal', 'tI': 'tetragonal',
    'hP': 'hexagonal',
    'hR': 'rhombohedral',
    'oP': 'orthorhombic', 'oC': 'orthorhombic', 'oF': 'orthorhombic', 'oI': 'orthorhombic',
    'mP': 'monoclinic', 'mC': 'monoclinic',
    'aP': 'triclinic',
}


class PeakListCreator:
    def __init__(
        self, 
        tag,
        save_to_directory=None,
        load_combined=False,
        overwrite_combined=False,
        input_path_templates=None,
        expt_file=None,
        refl_file=None,
        suffix='_strong.expt',
        min_reflections_per_experiment=3,
        max_reflections_per_experiment=100,
        known_unit_cell=None, 
        known_space_group=None,
        variable_detector=False,
        mask_file=None,
        mask_pad=2,
    ):
        self.input_path_templates = input_path_templates
        self.max_reflections_per_experiment = max_reflections_per_experiment
        self.min_reflections_per_experiment = min_reflections_per_experiment

        self.variable_detector = variable_detector

        self.mask_pad = mask_pad
        if mask_file is not None:
            from libtbx import easy_pickle
            mask = easy_pickle.load(mask_file)
            self.panel_masks = [p.as_numpy_array() for p in mask]
        else:
            self.panel_masks = None
        
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

        if expt_file is None:
            self.expt_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined_all.expt')
            self.refl_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined_all.refl')
        else:
            self.expt_file_name = expt_file
            self.refl_file_name = refl_file
        if self.load_combined == False:
            if expt_file is None:
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
            self.s1_lab = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_s1.npy'),
                )
        self.beam_delta = np.zeros(2)
        self.refl_mask = np.ones(self.q2_obs.size, dtype=bool)
        self.known_unit_cell = known_unit_cell
        self.known_space_group = known_space_group
        self.error = None

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
        for run_index in range(len(self.input_path_templates)):
            input_path = self.input_path_templates[run_index]
            expt_file_names_run = []
            refl_file_names_run = []
            if os.path.exists(input_path):
                for file_name in os.listdir(input_path):
                    if file_name.endswith(self.suffix):
                        expt_file_name = os.path.join(input_path, file_name)
                        if self.suffix.endswith('refined.expt'):
                            refl_file_name = os.path.join(input_path, file_name.replace('.expt', '.refl'))
                            if not os.path.exists(refl_file_name):
                                refl_file_name = os.path.join(
                                    input_path,
                                    file_name.replace( '_refined.expt', '_indexed.refl')
                                )
                        else:
                            refl_file_name = os.path.join(input_path, file_name.replace('.expt', '.refl'))
                        if os.path.exists(expt_file_name) and os.path.exists(refl_file_name):
                            expt_file_names_run.append(expt_file_name)
                            refl_file_names_run.append(refl_file_name)
                if len(expt_file_names_run) > 0:
                    run_str =  os.path.split(input_path)[1]
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
        s1_lab = flumpy.to_numpy(
                panel.get_lab_coord(panel.pixel_to_millimeter(flex.vec2_double(
                    flex.double(xyz[:, 0].ravel()),
                    flex.double(xyz[:, 1].ravel())
                )))
            )
        # s1 has magnitude 1/wavelength
        s1 = s1_lab / (wavelength * np.linalg.norm(s1_lab, axis=1)[:, np.newaxis])
        return s1, s1_lab

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
        s1_lab = []
        s0 = []
        expt_indices = []
        refl_counts = []
        for expt_index, expt in enumerate(expts):
            refls_expt = refls.select(refls['id'] == expt_index)
            if len(refls_expt) > 0:
                wavelength = expt.beam.get_wavelength()
                s0_lattice = expt.beam.get_s0() #|s0| = 1/wavelength
                # s1 has magnitude 1/wavelength; s1_lab is the raw lab coordinate vector
                s1_lattice = []
                s1_lab_lattice = []
                for panel_index, panel in enumerate(expt.detector):
                    refls_panel = refls_expt.select(refls_expt['panel'] == panel_index)
                    if len(refls_panel) > 0:
                        xyz = flumpy.to_numpy(refls_panel['xyzobs.px.value'])
                        if self.panel_masks is not None:
                            panel_mask = self.panel_masks[panel_index]
                            pad = self.mask_pad
                            n_slow, n_fast = panel_mask.shape
                            keep = np.ones(xyz.shape[0], dtype=bool)
                            for i in range(xyz.shape[0]):
                                px = int(round(xyz[i, 0]))
                                py = int(round(xyz[i, 1]))
                                y0 = max(0, py - pad)
                                y1 = min(n_slow, py + pad + 1)
                                x0 = max(0, px - pad)
                                x1 = min(n_fast, px + pad + 1)
                                if not panel_mask[y0:y1, x0:x1].all():
                                    keep[i] = False
                            xyz = xyz[keep]
                            if xyz.shape[0] == 0:
                                continue
                        s1_panel, s1_lab_panel = self._get_s1_from_xyz(
                            panel,
                            xyz,
                            wavelength,
                            )
                        s1_lattice.append(s1_panel)
                        s1_lab_lattice.append(s1_lab_panel)
                s1_lab_lattice = np.vstack(s1_lab_lattice)
                refl_counts.append(s1_lab_lattice.shape[0])
                # s0 and s1_lab are retained for constructing secondary peaks and beam center optimization
                s1_lab.append(np.vstack(s1_lab_lattice))
                s0.append(s0_lattice)
                expt_indices.append(expt_index*np.ones(s1_lab_lattice.shape[0], dtype=int))
                # q2_lattice is the magnitude**2 of the scattering vector
                q2.append(self._get_q2_spacing(
                    np.vstack(s1_lattice), s0_lattice)
                    )
        self.q2_obs = np.concatenate(q2)
        self.refl_counts = np.array(refl_counts)
        self.expt_indices = np.concatenate(expt_indices)
        self.s0 = np.vstack(s0)
        self.s1_lab = np.vstack(s1_lab)
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
            self.s1_lab
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
        bins_x = np.linspace(self.s1_lab[:, 1].min(), self.s1_lab[:, 1].max(), n_bins + 1)
        bins_y = np.linspace(self.s1_lab[:, 0].min(), self.s1_lab[:, 0].max(), n_bins + 1)
        centers_x = (bins_x[1:] + bins_x[:-1]) / 2
        centers_y = (bins_y[1:] + bins_y[:-1]) / 2

        # 2D histogram of the reflection positions
        hist, _, _ = np.histogram2d(x=self.s1_lab[:, 1], y=self.s1_lab[:, 0], bins=[bins_x, bins_y])

        # This maps the reflections onto the histogram coordinates
        refl_x = np.searchsorted(bins_x, self.s1_lab[:, 1]) - 1
        refl_y = np.searchsorted(bins_y, self.s1_lab[:, 0]) - 1
        refl_x[refl_x == -1] = 0
        refl_y[refl_y == -1] = 0

        # This should be the correct way to do this, the detector distance should be the average of
        # the detector distance of the reflections in the xy bin. This does not work though.
        # Using the same detector distance of 
        #centers_z, _, _, _ = scipy.stats.binned_statistic_2d(
        #    x=self.s1_lab[:, 1],
        #    y=self.s1_lab[:, 0], 
        #    values=self.s1_lab[:, 2],
        #    bins=[bins_x, bins_y],
        #    statistic='mean'
        #    )
        #centers_z[np.isnan(centers_z)] = np.nanmean(centers_z)
        centers_z = self.s1_lab[:, 2].mean()

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
        mask_indices = np.vstack((mask_indices))
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
            self.s1_lab[:, 0], self.s1_lab[:, 1],
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
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_quick_mask.png'))
        plt.show()

        # Make sure the masked reflections actually line up with the mask
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1_lab[self.refl_mask, 0], self.s1_lab[self.refl_mask, 1],
            s=0.01, color=[0, 0, 0], alpha=0.1
            )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('Scatter Plot Of Masked Reflection Coordinates')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_masked_reflection.png'))
        plt.show()
        
        """
        # Diagnostic plots

        # Make sure the masked reflections actually line up with the mask
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1_lab[self.refl_mask, 0], self.s1_lab[self.refl_mask, 1],
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

    def pick_peaks(self, n_bins=1000, d_min=60, d_max=3.5, q2_min=None, q2_max=None, mask=True, exclude_max=20, prominence=30):
        if q2_min is None:
            self.d_min = d_min
            self.q2_min = 1 / self.d_min**2
        else:
            self.q2_min = q2_min
            self.d_min = 1 / np.sqrt(q2_min)
        if q2_max is None:
            self.d_max = d_max
            self.q2_max = 1 / self.d_max**2
        else:
            self.q2_max = q2_max
            self.d_max = 1 / np.sqrt(q2_max)
        self.q2_bins = np.linspace(self.q2_min, self.q2_max, n_bins + 11)
        self.q2_centers = (self.q2_bins[1:] + self.q2_bins[:-1]) / 2
        if mask:
            self.q2_hist, _ = np.histogram(self.q2_obs[self.refl_mask], bins=self.q2_bins)
        else:
            self.q2_hist, _ = np.histogram(self.q2_obs, bins=self.q2_bins)

        found_peak_indices = scipy.signal.find_peaks(self.q2_hist, prominence=prominence)
        peaks = list(self.q2_centers[found_peak_indices[0]][:exclude_max])

        fig, axes = plt.subplots(1, 1, figsize=(30, 6))
        axes.plot(self.q2_centers, self.q2_hist, label='Histogram')
        axes.set_xlabel(r'q2 (1/$\mathrm{\AA}^2$)')
        axes.set_title('Click to add peak  |  Shift+click to remove nearest peak  |  Close window when done')
        fig.tight_layout()

        line_objs = []
        ann_objs = []

        def redraw():
            for obj in line_objs:
                obj.remove()
            for obj in ann_objs:
                obj.remove()
            line_objs.clear()
            ann_objs.clear()
            peaks.sort()
            ymax = self.q2_hist.max()
            n = len(peaks)
            for p_index, p in enumerate(peaks):
                ln, = axes.plot([p, p], [0, ymax], linestyle='dotted', linewidth=1, color=[0, 0, 0])
                line_objs.append(ln)
                ann = axes.annotate(p_index, xy=(p - 0.001, (1 - p_index / max(n, 1)) * ymax))
                ann_objs.append(ann)
            fig.canvas.draw_idle()

        def on_click(event):
            if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode != '':
                return
            if event.inaxes is not axes or event.button != 1 or event.xdata is None:
                return
            x = event.xdata
            if event.key == 'shift':
                if peaks:
                    nearest = min(range(len(peaks)), key=lambda i: abs(peaks[i] - x))
                    peaks.pop(nearest)
            else:
                peaks.append(x)
            redraw()

        def on_close(event):
            self.q2_peaks = np.sort(np.array(peaks))
            fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_peaks.png'))
            np.save(os.path.join(self.save_to_directory, f'{self.tag}_peaks.npy'), self.q2_peaks)
            print(repr(self.q2_peaks))
            print(repr(1 / np.sqrt(self.q2_peaks)))

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('close_event', on_close)
        redraw()
        plt.show()

    def fit_peaks(self, n_max, fit_shift=True, exclude_fit_shift=[]):
        # Interactively select individual peaks used to initialise the broadening model.
        ind_peak_indices = []

        fig_sel, axes_sel = plt.subplots(1, 1, figsize=(30, 6))
        axes_sel.plot(self.q2_centers, self.q2_hist)
        axes_sel.set_xlabel(r'q2 (1/$\mathrm{\AA}^2$)')
        axes_sel.set_title('Click to select isolated peaks for broadening fit  |  Shift+click to remove  |  Close window when done')
        fig_sel.tight_layout()

        line_objs_sel = []
        ann_objs_sel = []

        def redraw_sel():
            for obj in line_objs_sel:
                obj.remove()
            for obj in ann_objs_sel:
                obj.remove()
            line_objs_sel.clear()
            ann_objs_sel.clear()
            ind_peak_indices.sort()
            ymax = self.q2_hist.max()
            for i, idx in enumerate(ind_peak_indices):
                p = self.q2_peaks[idx]
                ln, = axes_sel.plot([p, p], [0, ymax], linestyle='dotted', linewidth=1, color=[0.8, 0, 0])
                line_objs_sel.append(ln)
                ann = axes_sel.annotate(idx, xy=(p - 0.001, (1 - i / max(len(ind_peak_indices), 1)) * ymax))
                ann_objs_sel.append(ann)
            fig_sel.canvas.draw_idle()

        def on_click_sel(event):
            if event.inaxes is not axes_sel or event.button != 1 or event.xdata is None:
                return
            x = event.xdata
            diffs = np.abs(self.q2_peaks - x)
            nearest_idx = int(np.argmin(diffs))
            if event.key == 'shift':
                if nearest_idx in ind_peak_indices:
                    ind_peak_indices.remove(nearest_idx)
            else:
                if nearest_idx not in ind_peak_indices:
                    ind_peak_indices.append(nearest_idx)
            redraw_sel()

        fig_sel.canvas.mpl_connect('button_press_event', on_click_sel)
        fig_sel.canvas.mpl_connect('close_event', lambda e: None)
        redraw_sel()
        plt.show()

        ind_peak_indices = np.array(ind_peak_indices, dtype=int)

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
            if peak_index in ind_peak_indices:
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
        def get_q2_spacing(s1_lab, s0):
            wavelength = 1 / np.linalg.norm(s0)
            dot_product = np.matmul(s1_lab, s0)
            magnitudes = np.linalg.norm(s1_lab) * np.linalg.norm(s0)
            theta2 = np.arccos(dot_product / magnitudes)
            return ((2 * np.sin(theta2 / 2)) / wavelength)**2

        def functional(delta, s1_lab_list, s0_list):
            L = 0
            for peak_index in range(len(s1_lab_list)):
                s1_lab = s1_lab_list[peak_index]
                s0 = s0_list[peak_index]
                q2_calc = np.zeros(s1_lab.shape[0])
                for i in range(s1_lab.shape[0]):
                    s1_lab_delta = s1_lab[i].copy()
                    s1_lab_delta[:2] += delta
                    q2_calc[i] = get_q2_spacing(s1_lab_delta, s0[i])
                L += q2_calc.std()
            return L

        s1_lab = []
        s0 = []
        if mask:
            q2_obs_masked = self.q2_obs[self.refl_mask]
            s1_lab_masked = self.s1_lab[self.refl_mask]
            expt_indices_masked = self.expt_indices[self.refl_mask]
        else:
            q2_obs_masked = self.q2_obs
            s1_lab_masked = self.s1_lab
            expt_indices_masked = self.expt_indices

        for peak_index in primary_peak_indices:
            differences = np.abs(q2_obs_masked - self.q2_peaks[peak_index])
            indices = differences < 3*self.q2_breadths[peak_index]
            s1_lab.append(s1_lab_masked[indices])
            s0.append(self.s0[expt_indices_masked[indices]])

        initial_simplex = np.array([
            [0.05, 0.025],
            [0.001, -0.01],
            [-0.025, -0.05],
            ])
        print(functional(np.zeros(2), s1_lab, s0))

        results = scipy.optimize.minimize(
            fun=functional,
            x0=[0, 0],
            args=(s1_lab, s0),
            method='Nelder-Mead',
            options={'initial_simplex': initial_simplex}
            )
        print(results)
        self.beam_delta = results.x[:2]
        self.s1_lab[:, :2] += self.beam_delta
        start = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            self.q2_obs[start: start + refl_counts] = self._get_q2_spacing(
                self.s1_lab[start: start + refl_counts], self.s0[expt_index]
                )
            start += refl_counts

    def bump_detector_distance(self, bump):
        self.s1_lab[:, 2] += bump
        q2 = []
        start = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            q2_obs = self.q2_obs[start: start + refl_counts]
            s1_lab = self.s1_lab[start: start + refl_counts]
            s0 = self.s0[expt_index]
            wavelength = 1 / np.linalg.norm(s0)
            s1 = s1_lab / (wavelength * np.linalg.norm(s1_lab, axis=1)[:, np.newaxis])
            # q2_lattice is the magnitude**2 of the scattering vector
            q2.append(self._get_q2_spacing(s1, s0))
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
            'broadening_params': self.broadening_params,
            'error': self.error,
            'note': note,
            }
        if extra_file_name is None:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info.json')
        else:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info_{extra_file_name}.json')
        pd.Series(output).to_json(file_name)

    def output_optimization(self, exclude_primary=[], exclude_secondary=[]):
        fig, axes = plt.subplots(2, 1, figsize=(45, 6), sharex=True)
        axes[0].plot(self.q2_centers, self.q2_hist)
        if hasattr(self, 'q2_diff_centers'):
            axes[1].plot(self.q2_diff_centers, self.q2_diff_hist)

        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()
        for p_index, p in enumerate(self.q2_peaks):
            label = 'Primary Picked' if p_index == 0 else None
            axes[0].plot([p, p], ylim0, linestyle='dotted', linewidth=1.5, color=[0, 0, 0])
            if hasattr(self, 'q2_diff_centers'):
                axes[1].plot([p, p], ylim1, linestyle='dotted', linewidth=1.5, color=[0, 0, 0], label=label)
            axes[0].annotate(p_index, xy=(p, 0.9*ylim0[1]))

        if hasattr(self, 'q2_peaks_secondary'):
            for p_index, p in enumerate(self.q2_peaks_secondary):
                label = 'Secondary Picked' if p_index == 0 else None
                axes[0].plot([p, p], ylim0, linestyle='dashed', linewidth=1.5, color=[0.8, 0, 0], label=label)
                if hasattr(self, 'q2_diff_centers'):
                    axes[1].plot([p, p], ylim1, linestyle='dashed', linewidth=1.5, color=[0.8, 0, 0], label=label)
                    axes[1].annotate(p_index, xy=(p, 0.9*ylim1[1]))
        axes[0].set_ylim(ylim0)
        axes[1].set_ylim(ylim1)
        axes[0].set_ylabel('Primary Positions')
        axes[1].set_ylabel('Secondary Positions')
        axes[1].set_xlabel(r'1 / d_spacing ($\mathrm{\AA}$)')
        axes[1].legend(loc='upper left', frameon=False)
        fig.tight_layout()
        plt.show()

        q2_peaks_primary = np.delete(self.q2_peaks, exclude_primary)
        if hasattr(self, 'q2_peaks_secondary'):
            q2_peaks_secondary = np.delete(self.q2_peaks_secondary, exclude_secondary)
            q2_peaks = np.sort(np.concatenate((q2_peaks_primary, q2_peaks_secondary)))
        else:
            q2_peaks_secondary = np.array([])
            q2_peaks = q2_peaks_primary

        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_mlindex_peak_list.npy'),
            q2_peaks
        )

    def create_secondary_peaks(self, q2_max=None, max_difference=None, max_refl_counts=None, min_separation=None, n_bins=2000, mask=True):
        start = 0
        q2_diff = []
        min_separation_obs = []
        for expt_index, refl_counts in enumerate(self.refl_counts):
            if mask:
                expt_refl_mask = self.refl_mask[start: start + refl_counts]
                q2_obs = self.q2_obs[start: start + refl_counts][expt_refl_mask]
                s1_lab = self.s1_lab[start: start + refl_counts][expt_refl_mask]
                masked_refl_counts = np.sum(expt_refl_mask)
            else:
                q2_obs = self.q2_obs[start: start + refl_counts]
                s1_lab = self.s1_lab[start: start + refl_counts]
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
                        s1_lab = s1_lab[indices]
                    if not q2_max is None:
                        indices = q2_obs < q2_max
                        q2_obs = q2_obs[indices]
                        s1_lab = s1_lab[indices]

                    if q2_obs.size > 1:
                        s1 = s1_lab / (wavelength * np.linalg.norm(s1_lab, axis=1)[:, np.newaxis])
                        q2_diff_all = np.linalg.norm(
                            s1[np.newaxis, :, :] - s1[:, np.newaxis, :],
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
        
    def pick_secondary_peaks(self, prominence=30, yscale=None):
        found_peak_indices = scipy.signal.find_peaks(self.q2_diff_hist, prominence=prominence)
        peaks = list(self.q2_diff_centers[found_peak_indices[0]])

        fig, axes = plt.subplots(2, 1, figsize=(45, 6), sharex=True)
        axes[0].plot(self.q2_centers, self.q2_hist)
        axes[1].plot(self.q2_diff_centers, self.q2_diff_hist)

        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()
        for p_index, p in enumerate(self.q2_peaks):
            label = 'Primary Picked' if p_index == 0 else None
            axes[0].plot([p, p], ylim0, linestyle='dotted', linewidth=1.5, color=[0, 0, 0])
            axes[1].plot([p, p], ylim1, linestyle='dotted', linewidth=1.5, color=[0, 0, 0], label=label)
        axes[0].set_ylim(ylim0)
        axes[1].set_ylim(ylim1)
        axes[0].set_ylabel('Primary Positions')
        axes[1].set_ylabel('Secondary Positions')
        axes[1].set_xlabel(r'1 / d_spacing ($\mathrm{\AA}$)')
        axes[1].set_title('Click to add peak  |  Shift+click to remove nearest peak  |  Close window when done')
        if yscale == 'log':
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')
        axes[1].legend(loc='upper left', frameon=False)

        line_objs = []
        ann_objs = []

        def redraw():
            for obj in line_objs:
                obj.remove()
            for obj in ann_objs:
                obj.remove()
            line_objs.clear()
            ann_objs.clear()
            peaks.sort()
            ymax = ylim1[1]
            n = len(peaks)
            for p_index, p in enumerate(peaks):
                ln, = axes[1].plot([p, p], [0, ymax], linestyle='dashed', linewidth=1.5, color=[0.8, 0, 0])
                line_objs.append(ln)
                ann = axes[1].annotate(p_index, xy=(p - 0.001, (1 - p_index / max(n, 1)) * ymax))
                ann_objs.append(ann)
            fig.canvas.draw_idle()

        def on_click(event):
            if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode != '':
                return
            if event.inaxes is not axes[1] or event.button != 1 or event.xdata is None:
                return
            x = event.xdata
            if event.key == 'shift':
                if peaks:
                    nearest = min(range(len(peaks)), key=lambda i: abs(peaks[i] - x))
                    peaks.pop(nearest)
            else:
                peaks.append(x)
            redraw()

        def on_close(event):
            self.q2_peaks_secondary = np.sort(np.array(peaks))
            fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.png'))
            np.save(
                os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.npy'),
                1 / np.sqrt(self.q2_peaks_secondary),
            )
            print(repr(1 / np.sqrt(self.q2_peaks_secondary)))

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('close_event', on_close)
        fig.tight_layout()
        redraw()
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

    def refine_unit_cell(self, unit_cell, bravais_lattice, n_iterations=10):
        """Refine a unit cell against self.q2_peaks using Gauss-Newton optimization.

        Parameters
        ----------
        unit_cell : array-like, shape (6,)
            [a, b, c, alpha, beta, gamma] with angles in degrees.
        bravais_lattice : str
            Bravais lattice identifier, e.g. 'hP', 'cF'.
        n_iterations : int, optional
            Number of Gauss-Newton refinement steps (default: 10).

        Returns
        -------
        np.ndarray, shape (6,)
            Refined [a, b, c, alpha, beta, gamma] with angles in degrees.
        """
        from importlib.resources import files
        from mlindex.utilities.UnitCellTools import (
            get_partial_unit_cell,
            get_xnn_from_unit_cell,
            get_unit_cell_from_xnn,
            get_full_unit_cell,
        )
        from mlindex.utilities.Q2Calculator import Q2Calculator
        from mlindex.utilities.numba_functions import fast_assign
        from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
        from mlindex.utilities.FigureOfMerits import get_M20_from_xnn

        lattice_system = _BL_TO_LATTICE_SYSTEM[bravais_lattice]

        with files('mlindex').joinpath(
            'models', f'{lattice_system}_1', 'data', f'hkl_ref_{bravais_lattice}.npy'
        ).open('rb') as f:
            hkl_ref = np.load(f)

        uc = np.array(unit_cell, dtype=float)
        uc[3:] *= np.pi / 180

        partial_uc = get_partial_unit_cell(uc, lattice_system=lattice_system)[np.newaxis]
        xnn = get_xnn_from_unit_cell(partial_uc, partial_unit_cell=True, lattice_system=lattice_system)

        q2_calculator = Q2Calculator(
            lattice_system=lattice_system,
            hkl=hkl_ref,
            tensorflow=False,
            representation='xnn',
        )

        q2_obs = self.q2_peaks[:20]
        target_function = CandidateOptLoss(q2_obs[np.newaxis], lattice_system=lattice_system)

        q2_ref_calc = q2_calculator.get_q2(xnn)
        hkl_assign = fast_assign(q2_obs, q2_ref_calc)
        hkl = np.take(hkl_ref, hkl_assign, axis=0)
        m20 = get_M20_from_xnn(q2_obs, xnn, hkl, hkl_ref, lattice_system)
        print(f'Initial M20: {m20[0]:.2f}')

        for i in range(n_iterations):
            target_function.update(hkl, xnn)
            xnn += target_function.gauss_newton_step(xnn)
            q2_ref_calc = q2_calculator.get_q2(xnn)
            hkl_assign = fast_assign(q2_obs, q2_ref_calc)
            hkl = np.take(hkl_ref, hkl_assign, axis=0)
            m20 = get_M20_from_xnn(q2_obs, xnn, hkl, hkl_ref, lattice_system)
            print(f'Iteration {i + 1}: M20 = {m20[0]:.2f}')

        partial_refined = get_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=lattice_system
        )[0]
        full_rad = get_full_unit_cell(partial_refined, lattice_system)
        full_rad[3:] *= 180 / np.pi
        return full_rad
