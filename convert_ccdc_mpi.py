from ccdc.io import EntryReader
from mpi4py import MPI
import numpy as np
import pandas as pd
import time

from CCDCEntryHelpers import calculate_s6
from CCDCEntryHelpers import ChemicalFormulaHandler
from CCDCEntryHelpers import load_identifiers
from CCDCEntryHelpers import spacegroup_to_symmetry
from CCDCEntryHelpers import verify_crystal_system_bravais_lattice_consistency
from CCDCEntryHelpers import verify_unit_cell_consistency
from CCDCEntryHelpers import verify_volume


class ProcessEntry:
    def __init__(self, csd_entry):
        self.status = True
        self.reason = None
        self.csd_entry = csd_entry
        self.identifier = csd_entry.identifier
        self.chemical_dict = None
        self.chemical_string = None
        self.cell_lengths = [None, None, None]
        self.cell_angles = [None, None, None]
        self.volume = None
        self.reduced_cell_lengths = [None, None, None]
        self.reduced_cell_angles = [None, None, None]
        self.reduced_volume = None
        self.bravais_lattice = None
        self.lattice_system = None
        self.crystal_system = None
        self.crystal_family = None
        self.spacegroup_number = None
        self.setting = None

    def verify_entry_is_usable(self, bad_identifiers=[]):
        # These identifiers are saved if they fail to make a diffraction pattern later.
        if self.identifier in bad_identifiers:
            self.reason = 'Entry failed during GenerateDataset.py'
            self.status = False
            return None

        # Chemical formulas are used for duplicate identification
        if self.csd_entry.formula == '':
            self.reason = 'No chemical formula'
            self.status = False
            return None

        if self.csd_entry.crystal.lattice_centring == 'unknown centring':
            self.reason = 'Unknown centering'
            self.status = False
            return None

        try:
            self.spacegroup_number, self.setting = \
                self.csd_entry.crystal.spacegroup_number_and_setting
        except:
            self.reason = 'Could not read spacegroup_number_and_setting'
            self.status = False
            return None

        # Apparently there are legitimate unit cells larger than 230. Dan Paley mentioned that they
        if self.spacegroup_number and self.spacegroup_number > 230:
            self.reason = 'Spacegroup number greater than 230'
            self.status = False
            return None

        self.bravais_lattice, self.crystal_family, self.crystal_system, self.lattice_system = \
            spacegroup_to_symmetry(self.spacegroup_number)

        # We are getting the crystal system from the reported spacegroup number, but exclude
        # extries where this information is missing from the entry itself. This is so we can verify
        # that the reported crystal system and spacegroup are compatible.
        if self.csd_entry.crystal.crystal_system == '':
            self.reason = 'No crystal system'
            self.status = False
            return None
        if not verify_crystal_system_bravais_lattice_consistency(
                self.csd_entry.crystal.crystal_system, self.bravais_lattice
                ):
            self.reason = 'Crystal system bravais lattice mismatch'
            self.status = False
            return None

        # This is to prevent round off errors. Such as a hexagonal angle of 119.999999999999 ...
        self.cell_lengths = np.round(self.csd_entry.crystal.cell_lengths, decimals=6)
        self.cell_angles = np.round(self.csd_entry.crystal.cell_angles, decimals=6)
        self.volume = np.round(self.csd_entry.crystal.cell_volume, decimals=6)
        if np.any(self.cell_lengths == 0) or np.any(self.cell_angles == 0) or (self.volume == 0):
            self.reason = 'Zeros in unit cell parameters'
            self.status = False
            return None
        if not verify_unit_cell_consistency(
                self.crystal_system, self.bravais_lattice, self.cell_lengths, self.cell_angles
                ):
            self.reason = 'Inconsistent unit cell'
            self.status = False
            return None
        # Verify that the unit cell volume is calculated correctly
        if not verify_volume(self.cell_lengths, self.cell_angles, self.volume):
            self.reason = 'Volume is not consistent with unit cell parameters'
            self.status = False
            return None

        self.reduced_cell_lengths = np.round(
            self.csd_entry.crystal.reduced_cell.cell_lengths,
            decimals=6
            )
        self.reduced_cell_angles = np.round(
            self.csd_entry.crystal.reduced_cell.cell_angles,
            decimals=6
            )
        self.reduced_volume = np.round(self.csd_entry.crystal.reduced_cell.volume, decimals=6)
        check_lengths = np.any(self.reduced_cell_lengths == 0)
        check_angles = np.any(self.reduced_cell_angles == 0)
        check_volume = self.reduced_volume == 0
        if check_lengths or check_angles or check_volume:
            self.reason = 'Zeros in reduced unit cell parameters'
            self.status = False
            return None
        if not verify_volume(
                self.reduced_cell_lengths,
                self.reduced_cell_angles,
                self.reduced_volume
                ):
            self.reason = 'Reduced volume is not consistent with reduced unit cell parameters'
            self.status = False
            return None

        self.chemical_formula_handler = ChemicalFormulaHandler(self.csd_entry.formula)
        try:
            self.chemical_formula_handler.get_chemical_composition_dict()
            self.chemical_formula_handler.get_chemical_composition_string()
            self.chemical_dict = self.chemical_formula_handler.chemical_dict
            self.chemical_string = self.chemical_formula_handler.chemical_string
        except:
            self.reason = 'Could not parse chemical formula'
            self.status = False
            return None

        # Check if the crystal packing is in an odd range. The packing coefficient should be
        # between 0.65 and 0.80 for stable crystals:   https://www.iucr.org/education/pamphlets/21
        # Case in point, entry XIKTUJ has a packing coefficient of 1.0. The structure has 29 carbons
        # and one boron in a 3.6 x 3.6 x 3.6 A cubic unit cell...
        # This is commented out because it takes an extremly long time
        # pc = self.csd_entry.crystal.packing_coefficient
        # if pc is not None and pc != 0:
        #    if pc > 0.9 or pc < 0.3:
        #        #print(f'{self.identifier} has packing coefficient {pc}')
        #        self.reason = 'Unreasonable packing coefficient'
        #        self.status = False
        #        return None
        # Dan P. mentioned that the unit cell volume should be roughly 18 times the number of
        # non-hydrogen atoms in the unit cell. Do an order of magnitude check for this.
        self.chemical_formula_handler.count_non_hydrogen_atoms()
        expected_volume = 18 * self.chemical_formula_handler.n_non_hydrogen_atoms
        volume = self.csd_entry.crystal.cell_volume
        if volume < 0.1 * expected_volume or volume > 10 * expected_volume:
            self.reason = 'Number of atoms in unit cell are inconsistent with volume'
            self.status = False
            return None

    def reduce_cell(self):
        # The ccdc uses the "nearly Buerger reduced cell"
        #  Thomas 2010: WebCSD the online portal to the Cambridge Structural Database
        #  Andrews & Bernstein 1988 <- the goal of this algorithm is a numerically stable way to
        #    determine the Niggli reduced cell.
        #  https://downloads.ccdc.cam.ac.uk/documentation/API/descriptive_docs/reduced_cell_searching.html
        # This is verified to be the same as cctbx's Niggli reduced cell.
        # See testing/cctbx_reduction_comparison.ipynb
        if self.status:
            a = self.reduced_cell_lengths[0]
            b = self.reduced_cell_lengths[1]
            c = self.reduced_cell_lengths[2]
            alpha = self.reduced_cell_angles[0]
            beta = self.reduced_cell_angles[1]
            gamma = self.reduced_cell_angles[2]
            self.g6 = np.array([
                a**2,
                b**2,
                c**2,
                2*b*c*np.cos(np.radians(alpha)),
                2*a*c*np.cos(np.radians(beta)),
                2*a*b*np.cos(np.radians(gamma)),
                ])
            self.s6 = calculate_s6(self.g6)
        else:
            self.g6 = np.zeros(6)
            self.s6 = np.zeros(6)

    def make_output_dict(self):
        self.output_dict = {
            'identifier': self.csd_entry.identifier,
            'is_organometallic': self.csd_entry.is_organometallic,
            'is_organic': self.csd_entry.is_organic,
            'is_polymeric': self.csd_entry.is_polymeric,
            'formula': self.csd_entry.formula,
            'chemical_composition_dict': self.chemical_dict,
            'chemical_composition_string': self.chemical_string,
            'r_factor': self.csd_entry.r_factor,
            'unit_cell': np.concatenate((self.cell_lengths, self.cell_angles)),
            'volume': self.volume,
            'reduced_unit_cell': np.concatenate((
                self.reduced_cell_lengths,
                self.reduced_cell_angles
                )),
            'reduced_volume': self.reduced_volume,
            'g6': self.g6,
            's6': self.s6,
            'bravais_lattice': self.bravais_lattice,
            'crystal_system': self.csd_entry.crystal.crystal_system,
            'crystal_family': self.crystal_family,
            'lattice_system': self.lattice_system,
            'spacegroup_number': self.spacegroup_number,
            'setting': self.setting,
            'centering': self.csd_entry.crystal.lattice_centring,
            }


if __name__ == '__main__':
    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    n_ranks = COMM.Get_size()
    # MPI approach
    #   Each rank reads different entries
    #   Output data frames are saved to different .json files
    csd_entry_reader = EntryReader('CSD')
    n_total = len(csd_entry_reader)
    dicts = []
    failed_dicts = []
    output_dicts = []
    in_numeric_tag = False
    duplicate_base = ''
    if rank == 0:
        # The entries in these 'bad_identifiers' lists failed to generate a
        # data set duing GenerateDataset.py.
        bad_identifiers = []
        for file_index in range(4):
            bad_identifiers += load_identifiers(f'data/bad_identifiers_{file_index}.txt')
        print(len(bad_identifiers))
    else:
        bad_identifiers = None
    bad_identifiers = COMM.bcast(bad_identifiers, root=0)

    for index in range(rank, n_total, n_ranks):
        csd_entry = csd_entry_reader[index]
        entry = ProcessEntry(csd_entry)
        entry.verify_entry_is_usable(bad_identifiers=bad_identifiers)
        entry.reduce_cell()
        entry.make_output_dict()

        if entry.status:
            dicts.append(entry.output_dict)
        else:
            failed_dicts.append(entry.output_dict)

        if index % 10000 == 0:
            print(f'{100 * index / n_total: 2.2f}  {index} {len(dicts)}')

    entries_rank = pd.DataFrame(dicts)
    entries_rank.to_parquet(f'data/csd_{rank:02d}.parquet')

    failed_read = pd.DataFrame(failed_dicts)
    failed_read.to_parquet(f'data/failed_read_csd_{rank:02d}.parquet')