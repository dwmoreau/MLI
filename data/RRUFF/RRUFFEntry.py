import gemmi
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.signal
import sys
sys.path.append('/Users/DWMoreau/MLI')

from Utilities import Q2Calculator
from EntryHelpers import verify_unit_cell_consistency_by_bravais_lattice
from EntryHelpers import get_unit_cell_volume
from EntryHelpers import spacegroup_to_symmetry
from Reindexing import hexagonal_to_rhombohedral_unit_cell
from Reindexing import hexagonal_to_rhombohedral_hkl
from Reindexing import reindex_entry_monoclinic
from Reindexing import reindex_entry_orthorhombic
from Reindexing import reindex_entry_triclinic


def get_sample_names(RRUFF_directory):
    sample_names = []
    for sample_name in os.listdir(os.path.join(RRUFF_directory, 'data')):
        if sample_name.startswith('D0'):
            sample_names.append(sample_name)
        elif sample_name.startswith('R0'): 
            sample_names.append(sample_name)
        elif sample_name.startswith('R1'): 
            sample_names.append(sample_name)
        elif sample_name.startswith('R2'): 
            sample_names.append(sample_name)
    sample_names.sort()
    return sample_names


class RRUFFEntry:
    def __init__(self, RRUFF_directory, sample_name, load):
        self.sample_name = sample_name
        self.directory_name = os.path.join(RRUFF_directory, 'data', self.sample_name)
        if load:
            self.load()
        else:
            self.dif_file_name = None
            self.processed_file_name = None
            for file_name in os.listdir(self.directory_name):
                if 'DIF_File' in file_name:
                    self.dif_file_name = os.path.join(self.directory_name, file_name)
                elif 'Processed' in file_name:
                    self.processed_file_name = os.path.join(self.directory_name, file_name)
            if self.dif_file_name and self.processed_file_name:
                self.status = True
            else:
                self.status = False

            self.wavelength = None
            self.spacegroup_number = None
            self.bravais_lattice = None
            self.crystal_system = None
            self.lattice_system = None
            self.unit_cell = None
            self.volume = None
            self.spacegroup_symbol_hall = None
            self.spacegroup_symbol_hm = None
            self.reindexed_spacegroup_symbol_hall = None
            self.reindexed_spacegroup_symbol_hm = None
            self.reindexed_unit_cell = None
            self.reindexed_volume = None
            self.I = None
            self.q2_pattern = None
            self.q2_found = None
            self.q2_peaks = None
            self.q2_sa = None
            self.q2_calc = None
            self.quality_diffraction = None

    def process_dif_file(self):
        self.unit_cell_degrees = np.zeros(6)
        with open(self.dif_file_name, 'r') as dif_file:
            read_peaks = False
            peaks_theta2 = []
            self.spacegroup_symbol = None
            for line in dif_file:
                if 'CELL PARAMETERS:' in line:
                    index = 0
                    for element in line[:-1].split('CELL PARAMETERS:')[1].split(' '):
                        if element != '':
                            try:
                                self.unit_cell_degrees[index] = float(element)
                            except:
                                self.status = False
                                return
                            index += 1
                if 'SPACE GROUP:' in line:
                    self.spacegroup_symbol = line[:-1].split('SPACE GROUP:')[1].replace(' ', '')
                if 'X-RAY WAVELENGTH:' in line:
                    try:
                        self.wavelength = float(line[:-1].split('X-RAY WAVELENGTH:')[1])
                    except:
                        self.status = False
                        return
                if read_peaks and (line.startswith('====') or 'Copyright' in line):
                    read_peaks = False
                if read_peaks:
                    index = 0
                    values = np.zeros(6)
                    for element in line[:-1].split(' '):
                        if element != '':
                            try:
                                values[index] = float(element)
                            except:
                                self.status = False
                                return
                            index += 1
                    peaks_theta2.append(values)
                if '2-THETA      INTENSITY    D-SPACING' in line:
                    read_peaks = True
        self.unit_cell = self.unit_cell_degrees.copy()
        self.unit_cell[3:] = self.unit_cell_degrees[3:] * np.pi/180
        if len(peaks_theta2) == 0 or self.spacegroup_symbol is None:
            self.status = False
            return
        peaks_theta2 = np.row_stack(peaks_theta2)
        self.hkl = peaks_theta2[:, 3:]
        self.d_spacing = self.wavelength / (2 * np.sin(peaks_theta2[:, 0] / 2 * np.pi/180))
        self.q2_peaks = 1 / self.d_spacing**2
        if self.spacegroup_symbol == 'P2_1/b':
            self.spacegroup_symbol = 'P 1 1 21/b'
        elif self.spacegroup_symbol == 'P2_1/n':
            self.spacegroup_symbol = 'P 1 21/n 1'
        elif self.spacegroup_symbol == 'P2_1/m':
            self.spacegroup_symbol = 'P 1 21/m 1'
        elif self.spacegroup_symbol == 'Pa3':
            self.spacegroup_symbol = 'Pa-3'
        elif self.spacegroup_symbol == 'Fd3m':
            self.spacegroup_symbol = 'Fd-3m'
        elif self.spacegroup_symbol == 'P2_1/a':
            self.spacegroup_symbol = 'P 1 21/a 1'
        elif self.spacegroup_symbol == 'P2_1/c':
            self.spacegroup_symbol = 'P 1 21/c 1'
        elif self.spacegroup_symbol == 'P2_1':
            self.spacegroup_symbol = 'P 1 21 1'
        elif self.spacegroup_symbol == 'Fm3m':
            self.spacegroup_symbol = 'Fm-3m'
        elif self.spacegroup_symbol == 'Pn3':
            self.spacegroup_symbol = 'Pn-3'
        elif self.spacegroup_symbol == 'Ia3':
            self.spacegroup_symbol = 'Ia-3'
        elif self.spacegroup_symbol == 'Ia3d':
            self.spacegroup_symbol = 'Ia-3d'
        elif self.spacegroup_symbol == 'Pn3m':
            self.spacegroup_symbol = 'Pn-3m'
        elif self.spacegroup_symbol == 'B2_1':
            self.spacegroup_symbol = 'B 2 21 2'

    def setup_entry(self):
        self.volume = get_unit_cell_volume(self.unit_cell)
        gemmi_spacegroup = gemmi.SpaceGroup(self.spacegroup_symbol)
        spacegroup_number = gemmi_spacegroup.number
        self.spacegroup_symbol_hm = gemmi_spacegroup.hm
        self.spacegroup_symbol_hall = gemmi_spacegroup.hall
        self.bravais_lattice, _, crystal_system, self.lattice_system = spacegroup_to_symmetry(spacegroup_number)
        self.crystal_system = gemmi_spacegroup.crystal_system_str()
        consistent_unit_cell = verify_unit_cell_consistency_by_bravais_lattice(
            bravais_lattice=self.bravais_lattice, unit_cell=self.unit_cell,
            )
        gemmi_cell = gemmi.UnitCell(*self.unit_cell_degrees)

        if self.lattice_system == 'cubic':
            self.reindexed_unit_cell = self.unit_cell
            self.reindexed_volume = self.volume
            self.reindexed_spacegroup_symbol_hm = self.spacegroup_symbol_hm
            uc = self.reindexed_unit_cell[[0]][np.newaxis]
        elif self.lattice_system == 'hexagonal':
            self.reindexed_unit_cell = self.unit_cell
            self.reindexed_volume = self.volume
            self.reindexed_spacegroup_symbol_hm = self.spacegroup_symbol_hm
            uc = self.reindexed_unit_cell[[0, 2]][np.newaxis]
        elif self.lattice_system == 'monoclinic':
            self.reindexed_unit_cell, self.reindexed_spacegroup_symbol_hm, hkl_reindexer = \
                reindex_entry_monoclinic(self.unit_cell, self.spacegroup_symbol_hm)
            self.hkl = self.hkl @ hkl_reindexer
            if self.reindexed_unit_cell is None:
                self.status = False
                return
            self.reindexed_volume = get_unit_cell_volume(self.reindexed_unit_cell)
            uc = self.reindexed_unit_cell[[0, 1, 2, 4]][np.newaxis]
        elif self.lattice_system == 'orthorhombic':
            self.reindexed_spacegroup_symbol_hm, _, self.reindexed_unit_cell, hkl_reindexer = \
                reindex_entry_orthorhombic(self.unit_cell, gemmi_spacegroup.hm, gemmi_spacegroup.number)
            self.reindexed_volume = self.volume
            self.hkl = self.hkl @ hkl_reindexer
            uc = self.reindexed_unit_cell[[0, 1, 2]][np.newaxis]
        elif self.lattice_system == 'rhombohedral':
            if np.all(self.unit_cell[3:] == [np.pi/2, np.pi/2, 4/3*np.pi/2]):
                self.reindexed_unit_cell, _ = hexagonal_to_rhombohedral_unit_cell(self.unit_cell)
                self.hkl = hexagonal_to_rhombohedral_hkl(self.hkl)
            else:
                self.reindexed_unit_cell = self.unit_cell
            self.reindexed_volume = get_unit_cell_volume(self.reindexed_unit_cell)
            self.reindexed_spacegroup_symbol_hm = self.spacegroup_symbol_hm
            uc = self.reindexed_unit_cell[[0, 3]][np.newaxis]
        elif self.lattice_system == 'tetragonal':
            self.reindexed_unit_cell = self.unit_cell
            self.reindexed_volume = self.volume
            self.reindexed_spacegroup_symbol_hm = self.spacegroup_symbol_hm
            uc = self.reindexed_unit_cell[[0, 2]][np.newaxis]
        elif self.lattice_system == 'triclinic':
            self.reindexed_unit_cell, hkl_reindexer = reindex_entry_triclinic(self.unit_cell)
            self.reindexed_volume = get_unit_cell_volume(self.reindexed_unit_cell)
            self.reindexed_spacegroup_symbol_hm = self.spacegroup_symbol_hm
            uc = self.reindexed_unit_cell[np.newaxis].copy()
            self.hkl = self.hkl @ hkl_reindexer
        self.reindexed_spacegroup_symbol_hall = gemmi.SpaceGroup(self.reindexed_spacegroup_symbol_hm).hall

        hkl_sa = gemmi.make_miller_array(gemmi_cell, gemmi_spacegroup, 0.95*self.d_spacing[-1], 100, True)

        self.q2_calc = Q2Calculator(
            lattice_system=self.lattice_system,
            hkl=self.hkl,
            tensorflow=False,
            representation='unit_cell',
            ).get_q2(uc)[0]

        self.q2_sa = Q2Calculator(
            lattice_system=self.lattice_system,
            hkl=hkl_sa,
            tensorflow=False,
            representation='unit_cell',
            ).get_q2(uc)[0]

        consistent_peaks = np.all(np.isclose(self.q2_calc, self.q2_peaks, atol=0.002))
        if not consistent_peaks or not consistent_unit_cell:
            self.status = False
            #print(self.lattice_system)
            #print(gemmi_spacegroup)
            #print(f'Unit cell BL consistent: {consistent_unit_cell}')
            #print(f'Consistent diffraction: {consistent_peaks}')
            #print(self.unit_cell)
            #print()

    def validate_quality(self, redo):
        validated_file_name = os.path.join(self.directory_name, f'{self.sample_name}_info_validated.json')
        if redo == False and os.path.exists(validated_file_name):
            return 
        print(self.lattice_system)
        print(self.reindexed_spacegroup_symbol_hm)
        print(self.reindexed_unit_cell)
        pattern = np.loadtxt(self.processed_file_name, comments='#', delimiter=',')
        resolution = self.wavelength / (2 * np.sin(pattern[:, 0] / 2 * np.pi/180))
        self.q2_pattern = 1 / resolution**2
        self.I = pattern[:, 1]

        fig, axes = plt.subplots(1, 1, figsize=(20, 6))
        axes.plot(self.q2_pattern, self.I, linewidth=2, zorder=1)
        ylim = axes.get_ylim()
        n_peaks = min(self.q2_peaks.shape[0], 20)
        for index in range(n_peaks):
            if index == 0:
                label0 = 'Read Log File Peaks'
                label1 = 'Calculated Log File Peaks'
            else:
                label0 = None
                label1 = None
            axes.plot(
                [self.q2_peaks[index], self.q2_peaks[index]], ylim,
                color=[1, 0, 0], linewidth=0.5, zorder=2,
                label=label0
                )
            axes.plot(
                [self.q2_calc[index], self.q2_calc[index]], [ylim[0], 0.66*ylim[1]],
                color=[0, 1, 0], linewidth=0.5, zorder=2,
                label=label1
                )
        for index in range(self.q2_sa.size):
            if index == 0:
                label = 'Non-systematically Absent'
            else:
                label = None
            axes.plot(
                [self.q2_sa[index], self.q2_sa[index]], [ylim[0], 0.33*ylim[1]],
                color=[1, 0, 1], linewidth=0.5, zorder=2,
                label=label,
                )
        axes.set_ylim(ylim)
        axes.set_xlim([0, 0.3])
        axes.legend(frameon=False, loc='upper left')
        plt.show()

        status = input('Good Pattern: y/n')
        if status != 'y':
            self.status = False
            self.diffraction_quality = False
        else:
            self.diffraction_quality = True
        self.export(validated=True)

    def find_peaks(self):
        status = 'n'
        prominence = 0.05
        while status == 'n':
            found_indices, _ = scipy.signal.find_peaks(self.I, prominence=prominence*I.max())
            self.q2_found = self.q2_pattern[found_indices]
            fig, axes = plt.subplots(1, 1, figsize=(100, 4))
            axes.plot(self.q2_pattern, self.I, linewidth=2, zorder=1)
            ylim = axes.get_ylim()
            for index in range(self.q2_found.size):
                axes.plot(
                    [self.q2_found[index], self.q2_found[index]], [ylim[0], ylim[1]],
                    color=0.1*np.ones(3), linewidth=1, zorder=2,
                    )
                axes.annotate(f'{index}', xy=(self.q2_found[index], 0.8*ylim[1]), fontsize=15)
            axes.set_ylim()
            plt.show()
            status = input('Good Prominence: y/n')
            if status != 'n':
                selected_indices = [int(p) for p in input('Select Peaks').split(',')]
                print(selected_indices)
                self.q2_found = self.q2_found[selected_indices]
            else:
                print(f'Current Prominence: {prominence}')
                prominence = float(input('Provide Prominence'))

    def export(self, validated=False):
        info = {
            'name': self.sample_name,
            'processed_file_name': self.processed_file_name,
            'dif_file_name': self.dif_file_name,
            'wavelength': self.wavelength,
            'spacegroup_number': self.spacegroup_number,
            'bravais_lattice': self.bravais_lattice,
            'crystal_system': self.crystal_system,
            'lattice_system': self.lattice_system,
            'unit_cell': self.unit_cell,
            'volume': self.volume,
            'spacegroup_symbol_hall': self.spacegroup_symbol_hall,
            'spacegroup_symbol_hm': self.spacegroup_symbol_hm,
            'reindexed_spacegroup_symbol_hall': self.reindexed_spacegroup_symbol_hall,
            'reindexed_spacegroup_symbol_hm': self.reindexed_spacegroup_symbol_hm,
            'reindexed_unit_cell': self.reindexed_unit_cell,
            'reindexed_volume': self.reindexed_volume,
            'pattern': self.I,
            'q2_pattern': self.q2_pattern,
            'q2_found': self.q2_found,
            'q2_log': self.q2_peaks,
            'q2_sa': self.q2_sa,
            'q2_calc': self.q2_calc,
            'status': self.status,
            'quality_diffraction': self.quality_diffraction,
            }
        if validated:
            file_name = os.path.join(self.directory_name, f'{self.sample_name}_info_validated.json')
        else:
            file_name = os.path.join(self.directory_name, f'{self.sample_name}_info.json')
        pd.Series(info).to_json(file_name)

    def load(self):
        file_name = os.path.join(self.directory_name, f'{self.sample_name}_info.json')
        if not os.path.exists(file_name):
            self.status = False
            return

        with open(file_name, 'r') as f:
            for line in f:
                info = json.loads(line)

        self.sample_name = info['name']
        self.processed_file_name = info['processed_file_name']
        self.dif_file_name = info['dif_file_name']
        self.wavelength = info['wavelength']
        self.spacegroup_number = info['spacegroup_number']
        self.bravais_lattice = info['bravais_lattice']
        self.crystal_system = info['crystal_system']
        self.lattice_system = info['lattice_system']
        self.unit_cell = None if info['unit_cell'] is None else np.array(info['unit_cell'])
        self.volume = info['volume']
        self.spacegroup_symbol_hall = info['spacegroup_symbol_hall']
        self.spacegroup_symbol_hm = info['spacegroup_symbol_hm']
        self.reindexed_spacegroup_symbol_hall = info['reindexed_spacegroup_symbol_hall']
        self.reindexed_spacegroup_symbol_hm = info['reindexed_spacegroup_symbol_hm']
        self.reindexed_unit_cell = None if info['reindexed_unit_cell'] is None else np.array(info['reindexed_unit_cell'])
        self.reindexed_volume = info['reindexed_volume']
        self.I = np.array(info['pattern'])
        self.q2_pattern = None if info['q2_pattern'] is None else np.array(info['q2_pattern'])
        self.q2_found = None if info['q2_found'] is None else np.array(info['q2_found'])
        self.q2_peaks = None if info['q2_log'] is None else np.array(info['q2_log'])
        self.q2_sa = None if info['q2_sa'] is None else np.array(info['q2_sa'])
        self.q2_calc = None if info['q2_calc'] is None else np.array(info['q2_calc'])
        self.status = info['status']
        self.quality_diffraction = info['quality_diffraction']
