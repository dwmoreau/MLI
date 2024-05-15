from cctbx.crystal import symmetry
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentList
from dxtbx import flumpy
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal


class PeakListCreator:
    def __init__(self, runs, input_path_template, tag, suffix='_strong.expt', d_min=60, d_max=3.5, save_to_directory=None):
        self.runs = runs
        self.input_path_template = input_path_template
        self.suffix = suffix
        self.tag = tag
        if save_to_directory is None:
            self.save_to_directory = os.path.join(os.getcwd(), self.tag)
        else:
            self.save_to_directory = os.path.join(save_to_directory, self.tag)
        if not os.path.exists(self.save_to_directory):
            os.mkdir(self.save_to_directory)
        self.d_min = d_min
        self.d_max = d_max
        self.q2_min = 1 / self.d_min**2
        self.q2_max = 1 / self.d_max**2
        self._get_expt_refl_filenames()
        self._parse_refl_files()

    def _get_s1_from_xy(self, panel, xyz, wavelength):
        s1 = flumpy.to_numpy(
                panel.get_lab_coord(panel.pixel_to_millimeter(flex.vec2_double(
                    flex.double(xyz[:, 0].ravel()),
                    flex.double(xyz[:, 1].ravel())
                )))
            )
        s1 = s1.reshape((xyz.shape[0], 3))
        s1 /= np.linalg.norm(s1, axis=1)[:, np.newaxis]
        s1 /= wavelength
        return s1

    def _get_d_spacing(self, s1, s0, wavelength):
        dot_product = np.matmul(s1, s0)
        magnitudes = np.linalg.norm(s1, axis=1) * np.linalg.norm(s0)
        theta2 = np.arccos(dot_product / magnitudes)
        return wavelength / (2 * np.sin(theta2 / 2))

    def _get_expt_refl_filenames(self):
        self.expt_file_names = []
        self.refl_file_names = []
        for run in self.runs:
            input_path = self.input_path_template.replace('!!!!', f'{run:04d}')
            for file_name in os.listdir(input_path):
                if file_name.endswith(self.suffix):
                    self.expt_file_names.append(os.path.join(input_path, file_name))
                    self.refl_file_names.append(os.path.join(input_path, file_name.replace('.expt', '.refl')))
        self.n_experiments = len(self.expt_file_names)

    def _parse_refl_files(self):
        d_primary = []
        d_secondary = []
        self.n_refls = 0
        for file_index in range(self.n_experiments):
            expts = ExperimentList.from_file(self.expt_file_names[file_index], check_format=False)
            refls = flex.reflection_table.from_file(self.refl_file_names[file_index])
            self.n_refls += len(refls)
            for expt_index, expt in enumerate(expts):
                wavelength = expt.beam.get_wavelength()
                s0 = expt.beam.get_s0()
                refls_expt = refls.select(refls['id'] == expt_index)
                s1_primary_lattice = []
                for panel_index, panel in enumerate(expt.detector):
                    refls_panel = refls_expt.select(refls_expt['panel'] == panel_index)
                    s1_primary_lattice.append(self._get_s1_from_xy(
                        panel, 
                        flumpy.to_numpy(refls_panel['xyzobs.px.value']), 
                        wavelength
                        ))
                s1_primary_lattice = np.row_stack(s1_primary_lattice)
                d_primary.append(self._get_d_spacing(
                    s1_primary_lattice, s0, wavelength
                    ))
                
                N_primary = s1_primary_lattice.shape[0]
                N_secondary = int(N_primary**2 / 2 - N_primary / 2)
                s1_secondary_lattice = np.zeros((N_primary, N_primary, 3))
                for i in range(3):
                    s1_secondary_lattice[:, :, i] = s1_primary_lattice[:, i][np.newaxis] - s1_primary_lattice[:, i][:, np.newaxis]
                indices = np.triu_indices(N_primary, k=1)
                s1_secondary_lattice = s1_secondary_lattice[indices[0], indices[1], :]
                d_secondary_lattice = np.zeros((N_secondary, 3))
                d_secondary_lattice[:, 0] = 1 / np.linalg.norm(s1_secondary_lattice, axis=1)
                d_secondary_lattice[:, 1] = d_primary[-1][indices[0]]
                d_secondary_lattice[:, 2] = d_primary[-1][indices[1]]
                d_secondary.append(d_secondary_lattice)
        d_primary = np.concatenate(d_primary)
        d_secondary = np.row_stack(d_secondary)
        self.q2_primary = 1 / d_primary**2
        self.q2_secondary = 1 / d_secondary**2

    def make_primary_histogram(self, n_bins=2000):
        self.primary_bins_q2 = np.linspace(self.q2_min, self.q2_max, n_bins + 11)
        self.primary_centers_q2 = (self.primary_bins_q2[1:] + self.primary_bins_q2[:-1]) / 2
        self.primary_hist_q2, _ = np.histogram(self.q2_primary, bins=self.primary_bins_q2)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_primary_hist.npy'),
            np.column_stack((self.primary_hist_q2, self.primary_centers_q2))
            )

    def pick_primary_peaks(self, exclude_list=[], exclude_max=100, add_peaks=[], shift={}, prominence=30, plot_kapton_peaks=False):
        found_peak_indices = scipy.signal.find_peaks(self.primary_hist_q2, prominence=prominence)
        found_peaks = self.primary_centers_q2[found_peak_indices[0]]
    
        primary_peaks = []
        for i in range(exclude_max, len(found_peaks)):
            exclude_list.append(i)
    
        fig, axes = plt.subplots(1, 1, figsize=(45, 6), sharex=True)
        axes.plot(self.primary_centers_q2, self.primary_hist_q2, label='Histogram')
        for p_index, p in enumerate(found_peaks):
            if not p_index in exclude_list:
                if p_index in shift.keys():
                    p += shift[p_index]
                primary_peaks.append(p)
                axes.plot([p, p], [0, self.primary_hist_q2.max()], linestyle='dotted', linewidth=1, color=[0, 0, 0], label='Found Peaks')
                axes.annotate(p_index, xy=(p, self.primary_hist_q2.max()))
        for p in add_peaks:
            primary_peaks.append(p)
            axes.plot([p, p], [0, self.primary_hist_q2.max()], linestyle='dotted', linewidth=1, color=[0.8, 0, 0], label='Added Peaks')
        if plot_kapton_peaks:
            kapton_peaks = [15.25, 7.625, 5.083333333, 3.8125, 3.05]
            for p in kapton_peaks:
                if p > self.d_max:
                    axes.plot([1/p**2, 1/p**2], [0, self.primary_hist_q2.max()], linestyle='dotted', linewidth=2, color=[0, 0.7, 0], label='Kapton Peaks')
        axes.set_xlabel('q2 (1/$\mathrm{\AA}^2$')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_primary_peaks.png'))
        plt.show()
        self.primary_peaks = np.sort(np.array(primary_peaks))
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_primary_peaks.npy'),
            1/np.sqrt(self.primary_peaks)
            )
        print(repr(1/np.sqrt(self.primary_peaks)))

    def create_secondary_peaks(self, q2_max=0.04, max_difference=0.00015):
        def get_differences(q2_primary, q2_secondary_source):
            ss_indices = np.searchsorted(q2_primary, q2_secondary_source)
            low = ss_indices == 0
            high = ss_indices == self.primary_peaks.size
            middle = np.logical_and(~low, ~high)
            differences = np.zeros(q2_secondary_source.size)
            differences[low] = np.abs(q2_secondary_source[low] - q2_primary[0])
            differences[high] = np.abs(q2_secondary_source[high] - q2_primary[-1])
            differences[middle] = np.column_stack((
                np.abs(q2_secondary_source[middle] - q2_primary[ss_indices[middle] - 1]),
                np.abs(q2_secondary_source[middle] - q2_primary[ss_indices[middle]])
                )).min(axis=1)
            return differences
            
        # Only use secondary differences if they came from peaks lower in resolution than
        # The highest resolution picked peak.
        # This is unnecessary given the filtering based on difference, but helps speed up the execution
    
        # These lines only select peaks close to a peak that was picked as a primary peak
        good_indices = np.max(self.q2_secondary[:, 1:], axis=1) < q2_max
        differences_1 = get_differences(self.primary_peaks, self.q2_secondary[good_indices, 1])
        differences_2 = get_differences(self.primary_peaks, self.q2_secondary[good_indices, 2])
    
        differences = np.column_stack((differences_1, differences_2)).max(axis=1)
        self.q2_secondary_filtered = self.q2_secondary[good_indices][differences < max_difference, 0]
    
        self.secondary_bins_q2 = np.linspace(0.00000001, self.q2_max, 801)
        self.secondary_centers_q2 = (self.secondary_bins_q2[1:] + self.secondary_bins_q2[:-1]) / 2
        self.secondary_hist_q2, _ = np.histogram(self.q2_secondary_filtered, bins=self.secondary_bins_q2)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_secondary_hist.npy'),
            np.column_stack((self.secondary_hist_q2, self.secondary_centers_q2))
            )
        
        edges = np.linspace(0, 0.005, 100)
        hist, _ = np.histogram(differences, bins=edges)
        d_centers = (edges[1:] + edges[:-1]) / 2
        fig, axes = plt.subplots(1, 1, figsize=(7, 3))
        axes.plot(d_centers, hist)
        ylim = axes.get_ylim()
        axes.plot([max_difference, max_difference], ylim, color=[0, 0, 0])
        axes.set_ylim(ylim)
        fig.tight_layout()
        plt.show
        
    def pick_secondary_peaks(self, include_list=[], prominence=30):
        indices = scipy.signal.find_peaks(self.secondary_hist_q2, prominence=prominence)
        self.secondary_peaks = []
        
        fig, axes = plt.subplots(2, 1, figsize=(45, 6), sharex=True)
        axes[0].plot(self.primary_centers_q2, self.primary_hist_q2)
        for p_index, p in enumerate(self.primary_peaks):
            axes[0].plot([p, p], [0, self.primary_hist_q2.max()], linestyle='dotted', linewidth=1, color=[0, 0, 0])
            axes[1].plot([p, p], [0, self.secondary_hist_q2.max()], linestyle='dotted', linewidth=2, color=[0, 0, 0])
        for p_index, p in enumerate(self.secondary_centers_q2[indices[0]]):
            if p_index in include_list:
                self.secondary_peaks.append(p)
            axes[1].plot([p, p], [0, self.secondary_hist_q2.max()], linestyle='dashed', linewidth=1, color=[0.8, 0, 0])
            axes[1].annotate(p_index, xy=(p, self.secondary_hist_q2.max()))
        axes[1].plot(self.secondary_centers_q2, self.secondary_hist_q2)
        axes[0].set_ylabel('Primary Positions')
        axes[1].set_ylabel('Secondary Positions')
        axes[1].set_xlabel('1 / d_spacing ($\mathrm{\AA}$)')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.png'))
        plt.show()
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.npy'),
            1/np.sqrt(np.array(self.secondary_peaks))
            )
        print(repr(1/np.sqrt(self.secondary_peaks)))
        

