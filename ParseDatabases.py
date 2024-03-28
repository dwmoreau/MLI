import gemmi
import numpy as np

from EntryHelpers import ChemicalFormulaHandler
from EntryHelpers import get_unit_cell_volume
from EntryHelpers import rename_spacegroup_for_gemmi
from EntryHelpers import spacegroup_to_symmetry
from EntryHelpers import verify_crystal_system_bravais_lattice_consistency
from EntryHelpers import verify_unit_cell_consistency_by_bravais_lattice
from EntryHelpers import verify_volume
from Reindexing import get_permuter
from Reindexing import reindex_entry
from Reindexing import hexagonal_to_rhombohedral_unit_cell
from Reindexing import hexagonal_to_rhombohedral_hkl
from Utilities import Q2Calculator


class ProcessEntry:
    def __init__(self):
        self.database = None
        self.identifier = None
        self.cif_file_name = None

        self.chemical_formula = None
        self.chemical_composition_dict = None
        self.chemical_composition_string = None
        self.chemical_composition_string_strict = None

        self.spacegroup_number = None
        self.bravais_lattice = None
        self.lattice_system = None
        self.crystal_system = None
        self.crystal_family = None

        self.spacegroup_symbol_hall = None
        self.spacegroup_symbol_hm = None
        self.unit_cell = np.zeros(6)
        self.volume = None

        self.split = None
        self.reindexed_spacegroup_symbol_hall = None
        self.reindexed_spacegroup_symbol_hm = None
        self.reindexed_unit_cell = np.zeros(6)
        self.reindexed_volume = None
        self.permutation = None

        self.reduced_unit_cell = np.zeros(6)
        self.reduced_volume = None

        self.r_factor = None

        self.status = True
        self.reason = None

        self.hkl_ref_monoclinic = np.load('data/hkl_ref_monoclinic.npy')[:60]
        self.hkl_ref_hexagonal = np.load('data/hkl_ref_hexagonal.npy')[:60]

    def make_output_dict(self):
        self.output_dict = {
            'database': self.database,
            'identifier': self.identifier,
            'cif_file_name': self.cif_file_name,
            'chemical_formula': self.chemical_formula,
            'chemical_composition_dict': self.chemical_composition_dict,
            'chemical_composition_string': self.chemical_composition_string,
            'chemical_composition_string_strict': self.chemical_composition_string_strict,
            'spacegroup_number': self.spacegroup_number,
            'bravais_lattice': self.bravais_lattice,
            'lattice_system': self.lattice_system,
            'crystal_system': self.crystal_system,
            'crystal_family': self.crystal_family,
            'volume': self.volume,
            'spacegroup_symbol_hall': self.spacegroup_symbol_hall,
            'spacegroup_symbol_hm': self.spacegroup_symbol_hm,
            'unit_cell': self.unit_cell,
            'reindexed_spacegroup_symbol_hall': self.reindexed_spacegroup_symbol_hall,
            'reindexed_spacegroup_symbol_hm': self.reindexed_spacegroup_symbol_hm,
            'reindexed_unit_cell': self.reindexed_unit_cell,
            'reindexed_volume': self.reindexed_volume,
            'split': self.split,
            'permutation': self.permutation,
            'reduced_unit_cell': self.reindexed_unit_cell,
            'reduced_volume': self.reduced_volume,
            'r_factor': self.r_factor,
            'status': self.status,
            'reason': self.reason,
        }

    def verify_reindexing(self):
        unit_cell = self.unit_cell.copy()
        unit_cell[3:] *= np.pi/180
        q2_calculator = Q2Calculator(lattice_system='triclinic', hkl=self.hkl_ref_monoclinic, tensorflow=False)
        q2 = q2_calculator.get_q2(unit_cell[np.newaxis])

        reindexed_unit_cell = self.reindexed_unit_cell.copy()
        reindexed_unit_cell[3:] *= np.pi/180
        reindexed_hkl = np.matmul(self.hkl_ref_monoclinic, get_permuter(self.permutation))
        reindexed_q2_calculator = Q2Calculator(lattice_system='triclinic', hkl=reindexed_hkl, tensorflow=False)
        reindexed_q2 = reindexed_q2_calculator.get_q2(reindexed_unit_cell[np.newaxis])
        check = np.isclose(q2, reindexed_q2).all()
        return check

    def verify_reindexing_rhombohedral(self):
        unit_cell = self.unit_cell.copy()
        unit_cell[3:] *= np.pi/180
        q2_calculator = Q2Calculator(lattice_system='triclinic', hkl=self.hkl_ref_hexagonal, tensorflow=False)
        q2 = q2_calculator.get_q2(unit_cell[np.newaxis])

        reindexed_unit_cell = self.reindexed_unit_cell.copy()
        reindexed_unit_cell[3:] *= np.pi/180
        reindexed_hkl = hexagonal_to_rhombohedral_hkl(self.hkl_ref_hexagonal)
        reindexed_q2_calculator = Q2Calculator(lattice_system='triclinic', hkl=reindexed_hkl, tensorflow=False)
        reindexed_q2 = reindexed_q2_calculator.get_q2(reindexed_unit_cell[np.newaxis])
        check = np.isclose(q2, reindexed_q2).all()
        return check

    def common_verification(self):
        # Apparently there are legitimate unit cells larger than 230. Dan Paley mentioned that they
        if self.spacegroup_number and self.spacegroup_number > 230:
            self.reason = 'Spacegroup number greater than 230'
            self.status = False
            #print(f'{self.reason}')
            return None

        self.bravais_lattice, self.crystal_family, self.crystal_system, self.lattice_system = \
            spacegroup_to_symmetry(self.spacegroup_number)

        if not verify_crystal_system_bravais_lattice_consistency(
                self.crystal_system, self.bravais_lattice
        ):
            self.reason = 'Crystal system bravais lattice mismatch'
            self.status = False
            #print(f'{self.reason}')
            return None

        if self.lattice_system == 'monoclinic' and self.spacegroup_symbol_hm.startswith('F'):
            # There is like one of these entries in all the CSD. I'm excluding these to make
            # logistics easier.
            self.reason = 'Excluding F centered monoclinic'
            self.status = False
            return None

        if np.any(self.unit_cell == 0) or (self.volume == 0):
            self.reason = 'Zeros in unit cell parameters'
            self.status = False
            #print(f'{self.reason}')
            return None

        # Verify that the unit cell volume is calculated correctly
        if not verify_volume(self.unit_cell, self.volume):
            self.reason = 'Volume is not consistent with unit cell parameters'
            self.status = False
            #print(f'{self.reason}')
            return None

        if np.any(self.reduced_unit_cell == 0) or (self.reduced_volume == 0):
            self.reason = 'Zeros in reduced unit cell parameters'
            self.status = False
            #print(f'{self.reason}')
            return None

        # Verify that the unit cell volume is calculated correctly
        if not verify_volume(self.reduced_unit_cell, self.reduced_volume):
            self.reason = 'Reduced volume is not consistent with reduced unit cell parameters'
            self.status = False
            #print(f'{self.reason}')
            return None

        if not verify_unit_cell_consistency_by_bravais_lattice(self.bravais_lattice, self.unit_cell):
            self.reason = 'Inconsistent unit cell'
            self.status = False
            #print(f'{self.reason}: {self.bravais_lattice} {self.unit_cell}')
            return None

        chemical_formula_handler = ChemicalFormulaHandler(self.chemical_formula)
        try:
            chemical_formula_handler.get_chemical_composition_dict()
            chemical_formula_handler.get_chemical_composition_string()
            self.chemical_composition_dict = chemical_formula_handler.chemical_dict
            self.chemical_composition_string = chemical_formula_handler.chemical_string
            self.chemical_composition_string_strict = chemical_formula_handler.chemical_string_strict
        except:
            self.reason = 'Could not parse chemical formula'
            self.status = False
            #print(f'{self.reason}')
            return None

        # Check if the crystal packing is in an odd range. The packing coefficient should be
        # between 0.65 and 0.80 for stable crystals:   https://www.iucr.org/education/pamphlets/21
        # Case in point, entry XIKTUJ has a packing coefficient of 1.0. The structure has 29 carbons
        # and one boron in a 3.6 x 3.6 x 3.6 A cubic unit cell...
        # This is commented out because it takes an extremely long time
        # pc = csd_entry.crystal.packing_coefficient
        # if pc is not None and pc != 0:
        #    if pc > 0.9 or pc < 0.3:
        #        #print(f'{self.identifier} has packing coefficient {pc}')
        #        self.reason = 'Unreasonable packing coefficient'
        #        self.status = False
        #        return None
        # Dan P. mentioned that the unit cell volume should be roughly 18 times the number of
        # non-hydrogen atoms in the unit cell. Do an order of magnitude check for this.
        chemical_formula_handler.count_non_hydrogen_atoms()
        expected_volume = 18 * chemical_formula_handler.n_non_hydrogen_atoms
        if self.volume < 0.1 * expected_volume or self.volume > 10 * expected_volume:
            self.reason = 'Number of atoms in unit cell are inconsistent with volume'
            self.status = False
            #print(f'{self.reason} {self.volume} {expected_volume}')
            return None

        if self.lattice_system in ['orthorhombic', 'monoclinic']:
            try:
                self.reindexed_spacegroup_symbol_hm, self.permutation, self.reindexed_unit_cell = \
                    reindex_entry(self.lattice_system, self.unit_cell, self.spacegroup_symbol_hm, self.spacegroup_number)
                self.reindexed_volume = self.volume
            except:
                self.reason = 'Could not permute axes'
                self.status = False
                print(f'{self.reason} {self.spacegroup_number} {self.spacegroup_symbol_hm}')
                return None
            if not self.verify_reindexing():
                self.reason = 'Monoclinic / Orthorhombic Reindexing Error'
                self.status = False
                print(f'{self.reason} {self.unit_cell} {self.reindexed_unit_cell} {self.permutation}')
                return None
            try:
                self.reindexed_spacegroup_symbol_hall = gemmi.SpaceGroup(self.reindexed_spacegroup_symbol_hm).hall
            except:
                self.reason = 'Could not read reindexed spacegroup symbol'
                self.status = False
                print(f'{self.reason} {self.spacegroup_number}   {self.spacegroup_symbol_hm}   {self.reindexed_spacegroup_symbol_hm}')
                return None
        elif self.lattice_system == 'rhombohedral':
            if np.all(self.unit_cell[3:] == [90, 90, 120]):
                self.reindexed_unit_cell = hexagonal_to_rhombohedral_unit_cell(self.unit_cell)
                self.reindexed_volume = get_unit_cell_volume(self.reindexed_unit_cell)
                if not self.verify_reindexing_rhombohedral():
                    self.reason = 'Rhombohedral Reindexing Error'
                    self.status = False
                    print(f'{self.reason} {self.unit_cell} {self.reindexed_unit_cell} {self.permutation}')
                    return None
            else:
                self.reindexed_volume = self.volume
                self.reindexed_unit_cell = self.unit_cell
            self.permutation = 'abc'
            self.reindexed_spacegroup_symbol_hm = self.spacegroup_symbol_hm
            self.reindexed_spacegroup_symbol_hall = self.spacegroup_symbol_hall
        else:
            self.reindexed_unit_cell = self.unit_cell
            self.reindexed_volume = self.volume
            self.permutation = 'abc'
            self.reindexed_spacegroup_symbol_hm = self.spacegroup_symbol_hm
            self.reindexed_spacegroup_symbol_hall = self.spacegroup_symbol_hall

        if self.lattice_system in ['cubic', 'orthorhombic']:
            self.split = 0
        elif self.lattice_system == 'tetragonal':
            if self.unit_cell[0] < self.unit_cell[2]:
                self.split = 0
            else:
                self.split = 1
        elif self.lattice_system == 'monoclinic':
            if self.reindexed_unit_cell[3] != 90:
                self.split = 0
            elif self.reindexed_unit_cell[4] != 90:
                self.split = 1
            elif self.reindexed_unit_cell[5] != 90:
                self.split = 2
        else:
            self.split = 0


class ProcessCSDEntry(ProcessEntry):
    def __init__(self):
        super(ProcessCSDEntry, self).__init__()
        self.database = 'csd'

    def verify_entry(self, csd_entry, bad_identifiers=[]):
        self.identifier = csd_entry.identifier
        # These identifiers are saved if they fail to make a diffraction pattern later.
        if self.identifier in bad_identifiers:
            self.reason = 'Entry failed during GenerateDataset.py'
            self.status = False
            return None

        # Chemical formulas are used for duplicate identification
        if csd_entry.formula == '':
            self.reason = 'No chemical formula'
            self.status = False
            #print(f'{self.reason}')
            return None
        else:
            self.chemical_formula = csd_entry.formula

        try:
            self.spacegroup_number, setting = csd_entry.crystal.spacegroup_number_and_setting
            spacegroup_symbol_from_csd = csd_entry.crystal.spacegroup_symbol
        except:
            self.reason = 'Could not read spacegroup_number_and_setting'
            self.status = False
            #print(f'{self.reason}')
            return None
        if '*' in spacegroup_symbol_from_csd:
            self.reason = 'Spacegroup has a *'
            self.status = False
            #print(f'{self.reason} {spacegroup_symbol_from_csd} {self.spacegroup_number}')
            return None

        try:
            spacegroup = gemmi.SpaceGroup(rename_spacegroup_for_gemmi(spacegroup_symbol_from_csd))
            self.spacegroup_symbol_hall = spacegroup.hall
            self.spacegroup_symbol_hm = spacegroup.hm
        except:
            self.reason = 'Could not read spacegroup in Gemmi'
            self.status = False
            print(f'{self.reason} {spacegroup_symbol_from_csd} {self.spacegroup_number}')
            return None

        # We are getting the crystal system from the reported spacegroup number, but exclude
        # entries where this information is missing from the entry itself. This is so we can verify
        # that the reported crystal system and spacegroup are compatible.
        if csd_entry.crystal.crystal_system == '':
            self.reason = 'No crystal system'
            self.status = False
            #print(f'{self.reason}')
            return None
        else:
            self.crystal_system = csd_entry.crystal.crystal_system

        # This is to prevent round off errors. Such as a hexagonal angle of 119.999999999999 ...
        cell_lengths = np.round(csd_entry.crystal.cell_lengths, decimals=6)
        cell_angles = np.round(csd_entry.crystal.cell_angles, decimals=6)
        self.unit_cell = np.concatenate((cell_lengths, cell_angles))
        self.volume = np.round(csd_entry.crystal.cell_volume, decimals=6)

        try:
            reduced_cell_lengths = np.round(csd_entry.crystal.reduced_cell.cell_lengths, decimals=6)
            reduced_cell_angles = np.round(csd_entry.crystal.reduced_cell.cell_angles, decimals=6)
            self.reduced_unit_cell = np.concatenate((reduced_cell_lengths, reduced_cell_angles))
            self.reduced_volume = np.round(csd_entry.crystal.reduced_cell.volume, decimals=6)
        except:
            self.reason = 'Could not reduce cell in ccdc API'
            self.status = False
            #print(f'{self.reason}')
            return None

        try:
            self.r_factor = csd_entry.r_factor
        except:
            self.reason = 'Could not read r-factor'
            self.status = False
            #print(f'{self.reason}')
            return None

        self.common_verification()


class ProcessCODEntry(ProcessEntry):
    def __init__(self):
        super(ProcessCODEntry, self).__init__()
        self.database = 'cod'

    def _parse_spacegroup_symbol(self, spacegroup_symbol):
        if '(' in spacegroup_symbol:
            spacegroup_symbol = spacegroup_symbol.split('(')[0]
        try:
            spacegroup = gemmi.SpaceGroup(rename_spacegroup_for_gemmi(spacegroup_symbol))
            return spacegroup
        except:
            return None

    def verify_entry(self, cif_file_name):
        self.cif_file_name = cif_file_name

        cif_file_block = gemmi.cif.read_file(cif_file_name).sole_block()

        # Chemical formulas are used for duplicate identification
        self.chemical_formula = cif_file_block.find_value('_chemical_formula_sum')
        if self.chemical_formula is None:
            self.chemical_formula = cif_file_block.find_value('_chemical_formula_structural')
        if self.chemical_formula is None:
            self.reason = 'No chemical formula'
            self.status = False
            #print(self.reason)
            return None
        elif "'" in self.chemical_formula:
            self.chemical_formula = self.chemical_formula.split("'")[1]

        spacegroup_number_string_0 = cif_file_block.find_value('_space_group_IT_number')
        if spacegroup_number_string_0 and '?' in spacegroup_number_string_0:
            spacegroup_number_string_0 = None
        spacegroup_number_string_1 = cif_file_block.find_value('_symmetry_Int_Tables_number')
        if spacegroup_number_string_1 and '?' in spacegroup_number_string_1:
            spacegroup_number_string_1 = None
        if spacegroup_number_string_0 is None:
            if spacegroup_number_string_1 is None:
                # In this case - try to read it from the spacegroup symbol
                self.spacegroup_number = None
            else:
                self.spacegroup_number = int(spacegroup_number_string_1)
        else:
            self.spacegroup_number = int(spacegroup_number_string_0)

        spacegroup_symbol_hm0 = cif_file_block.find_value('_symmetry_space_group_name_H-M')
        spacegroup_symbol_hm1 = cif_file_block.find_value('_space_group_name_H-M')
        spacegroup_symbol_hm2 = cif_file_block.find_value('_space_group_name_H-M_alt')
        spacegroup_symbol_hall0 = cif_file_block.find_value('_space_group_name_Hall')
        spacegroup_symbol_hall1 = cif_file_block.find_value('_symmetry_space_group_name_Hall')
        spacegroup = None
        if spacegroup_symbol_hm0:
            spacegroup = self._parse_spacegroup_symbol(spacegroup_symbol_hm0)
        if spacegroup is None and spacegroup_symbol_hm1:
            spacegroup = self._parse_spacegroup_symbol(spacegroup_symbol_hm1)
        if spacegroup is None and spacegroup_symbol_hm2:
            spacegroup = self._parse_spacegroup_symbol(spacegroup_symbol_hm2)
        if spacegroup is None and spacegroup_symbol_hall0:
            spacegroup = self._parse_spacegroup_symbol(spacegroup_symbol_hall0)
        if spacegroup is None and spacegroup_symbol_hall1:
            spacegroup = self._parse_spacegroup_symbol(spacegroup_symbol_hall1)

        if spacegroup:
            if self.spacegroup_number is None:
                self.spacegroup_number = spacegroup.number
            self.crystal_system = spacegroup.crystal_system_str()
            self.spacegroup_symbol_hall = spacegroup.hall
            self.spacegroup_symbol_hm = spacegroup.hm
        else:
            self.reason = 'Could not read spacegroup symbol'
            self.status = False
            #print(self.reason)
            return None

        unit_cell_keys = [
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
            '_cell_angle_alpha',
            '_cell_angle_beta',
            '_cell_angle_gamma',
            ]
        self.unit_cell = np.zeros(6)
        for uc_index, key in enumerate(unit_cell_keys):
            uc_string = cif_file_block.find_value(key)
            if uc_string:
                if '(' in uc_string:
                    self.unit_cell[uc_index] = float(uc_string.split('(')[0])
                else:
                    self.unit_cell[uc_index] = float(uc_string)

        # This is to prevent round off errors. Such as a hexagonal angle of 119.999999999999 ...
        self.unit_cell = np.round(self.unit_cell, decimals=6)
        self.volume = get_unit_cell_volume(self.unit_cell)

        gv = gemmi.GruberVector(gemmi.UnitCell(*self.unit_cell), spacegroup)
        gv.niggli_reduce()
        self.reduced_unit_cell = np.round(gv.cell_parameters(), decimals=6)
        self.reduced_volume = get_unit_cell_volume(self.reduced_unit_cell)

        try:
            self.r_factor = cif_file_block.find_value('_refine_ls_R_factor_gt')
        except:
            self.reason = 'Could not read r-factor'
            self.status = False
            #print(f'{self.reason}')
            return None

        self.common_verification()
