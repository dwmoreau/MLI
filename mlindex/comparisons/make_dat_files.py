import numpy as np
import pandas as pd
from pathlib import Path
from mlindex.utilities.UnitCellTools import get_unit_cell_volume


def q_to_two_theta(q, wavelength):
	return np.sort(np.degrees(2 * np.arcsin(np.array(q) * wavelength / 2)))


def get_cell_maxima(unit_cell, volume):
	"""Return (amax, bmax, cmax, volmax) with 0 where defaults suffice, else 25% above true value rounded to int."""
	amax = 0 if unit_cell[0] <= 20.0 else round(unit_cell[0] * 1.25)
	bmax = 0 if unit_cell[1] <= 20.0 else round(unit_cell[1] * 1.25)
	cmax = 0 if unit_cell[2] <= 20.0 else round(unit_cell[2] * 1.25)
	volmax = 0 if volume <= 1500.0 else round(volume * 1.25)
	return amax, bmax, cmax, volmax


def write_ito_dat(q, wavelength, directory, tag, unit_cell, volume):
	dat_dir = Path(directory, tag)
	dat_dir.mkdir(exist_ok=True)
	dat_file_name = Path(dat_dir, f'{tag}.dat')

	two_theta = q_to_two_theta(q, wavelength)

	# Parameter card 1: fixed-column format
	# Col 1: MAN=9 (suppress instruction printing)
	# Col 4: NSOLMX=9 (print up to 9 solutions)
	# Cols 5-10: NSYST all 0 (indifferent to crystal system)
	# Cols 11-20: spaces (default TOL2=3.0, TOL3=4.5)
	# Cols 21-30: WAVEL (F10.5)
	param_card1 = f'9009 0 0 0          {wavelength:10.5f}'

	with open(dat_file_name, 'w') as f:
		f.write(f'{tag}\n')
		f.write(f'{param_card1}\n')
		f.write(f' 0.000\n')
		for tt in two_theta:
			f.write(f'{tt:10.5f}\n')
		f.write('0.0\n')
		f.write('END\n')


def _write_treor_file(f, tag, two_theta, keywords):
	f.write(f' {tag}\n')
	for tt in two_theta:
		f.write(f'{tt:16.5f}\n')
	f.write('\n')
	f.write(f' {keywords}\n')
	f.write(' END*\n')
	f.write('  0.00\n')


def write_treor_dat(q, wavelength, directory, tag, unit_cell, volume):
	dat_dir = Path(directory, tag)
	dat_dir.mkdir(exist_ok=True)

	two_theta = q_to_two_theta(q, wavelength)

	cem = max(unit_cell[:3])
	cem_kw = f' CEM={round(cem * 1.25)},' if cem > 25 else ''

	# High-symmetry file: cubic/tetragonal/hexagonal/orthorhombic/monoclinic
	vol_high = round(volume * 1.25) if volume > 2000 else 2000
	high_kw = f'WAVE={wavelength}, CHOICE=3, MONO=130.0, VOL={vol_high},{cem_kw}'
	with open(Path(dat_dir, f'{tag}_high.dat'), 'w') as f:
		_write_treor_file(f, tag, two_theta, high_kw)

	# Triclinic file: triclinic only, VOL always set explicitly
	vol_tric = round(volume * 1.25)
	tric_kw = f'WAVE={wavelength}, CHOICE=3, TRIC=1, VOL={vol_tric},{cem_kw}'
	with open(Path(dat_dir, f'{tag}_tric.dat'), 'w') as f:
		_write_treor_file(f, tag, two_theta, tric_kw)


def write_dicvol_dat(q, wavelength, directory, tag, unit_cell, volume):
	dat_dir = Path(directory, tag)
	dat_dir.mkdir(exist_ok=True)
	dat_file_name = Path(dat_dir, f'{tag}.dat')

	two_theta = q_to_two_theta(q, wavelength)
	n = len(two_theta)
	amax, bmax, cmax, volmax = get_cell_maxima(unit_cell, volume)

	with open(dat_file_name, 'w') as f:
		f.write(f'{tag}\n')
		f.write(f'{n} 2 1 1 1 1 1 1\n')
		f.write(f'{amax} {bmax} {cmax} 0. {volmax} 0. 0.\n')
		f.write(f'{wavelength} 0. 0. 0.\n')
		f.write(f'0. 0. 0.\n')
		for tt in two_theta:
			f.write(f'{tt:.3f}\n')


df = pd.read_json('CNRS_output_data_verified_final3.json')
dicvol_dir = 'results/dicvol'
ito_dir = 'results/ito'
treor_dir = 'results/treor'

for index in range(len(df)):
	entry = df.iloc[index]
	tag = Path(entry.file_name).name.split('.json')[0]
	q = entry.peak_positions[:20]
	wavelength = np.round(entry.primary_wavelength, decimals=6)
	unit_cell = np.array(entry.unit_cell)
	volume = get_unit_cell_volume(unit_cell[np.newaxis])[0]

	write_dicvol_dat(q, wavelength, dicvol_dir, tag, unit_cell, volume)
	write_ito_dat(q, wavelength, ito_dir, tag, unit_cell, volume)
	write_treor_dat(q, wavelength, treor_dir, tag, unit_cell, volume)
