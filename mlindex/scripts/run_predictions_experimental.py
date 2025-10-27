import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from mpi4py import MPI
import numpy as np
import pandas as pd

from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.optimization.UtilitiesOptimizer import get_logger
from mlindex.optimization.UtilitiesOptimizer import get_mpi_organizer
from mlindex.optimization.UtilitiesOptimizer import get_optimizers


def print_results(save_to_dir, material, tag, broadening_tag, lattice_system, unit_cell, unit_cell_pred, average, print_true):
    xnn = get_xnn_from_unit_cell(
        unit_cell[np.newaxis],
        lattice_system=lattice_system,
        partial_unit_cell=True
        )
    volume = get_unit_cell_volume(
        unit_cell[np.newaxis],
        lattice_system=lattice_system,
        partial_unit_cell=True
        )[0]
    if print_true:
        print(unit_cell, volume)

    xnn_pred = get_xnn_from_unit_cell(
        unit_cell_pred,
        lattice_system=lattice_system,
        partial_unit_cell=True
        )
    volume_pred = get_unit_cell_volume(
        unit_cell_pred,
        lattice_system=lattice_system,
        partial_unit_cell=True
        )

    if average:
        unit_cell_top = unit_cell_pred.mean(axis=0)
        volume_top = volume_pred.mean(axis=0)
    else:
        unit_cell_top = unit_cell_pred[0]
        volume_top = volume_pred[0]
    difference = np.linalg.norm(xnn - xnn_pred, axis=1)
    order = np.argsort(difference)
    unit_cell_best = unit_cell_pred[order[0]]
    volume_best = volume_pred[order[0]]

    np.save(
        os.path.join(save_to_dir, material, f'{material}_{tag}_{broadening_tag}.npy'),
        np.stack((
            np.concatenate((unit_cell_top, [volume_top])),
            np.concatenate((unit_cell_best, [volume_best])),
            ))
        )
    print(unit_cell_top, np.round(volume_top, decimals=2), np.round(difference[0], decimals=4))
    print(unit_cell_best, np.round(volume_best, decimals=2), np.round(difference[order[0]], decimals=4), order[0])


broadening_tag = '1'
n_top_candidates = 20

"""
base_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/experimental_data/gsas'
entry_tags = {
    '11bmb_3844': ['cP_0', [4.15686, 4.15686, 4.15686, 90, 90, 90]],
    '11bmb_6231': ['mP_1_01', [6.76298, 13.80425, 6.82476, 90, 111.07, 90]], #acb
    '11bmb_8716': ['mP_0_00', [7.71395, 8.6628, 10.80787, 90, 102.982, 90]], #abc
    'CuCr2O4': ['tI_0_01', [6.0392,  6.0392,  7.71289, 90,  90,  90]],
    'FAP': ['hP_1_03', [9.372,   9.372,   6.886,   90,  90,  120]],
    'garnet': ['cI_0', [12.19, 12.19,   12.19,   90,  90,  90]],
    'La7Ca3MnO3_50K': ['oP_0_00', [5.46677, 5.48278,  7.72499, 90,  90,  90]],
    'LaMnO3_50K': ['oP_0_00', [5.53373,  5.73839, 7.69644, 90,  90,  90]],
    'PBSO4': ['oP_0_00', [5.40194,   6.96943, 8.49223, 90,  90,  90]],
    'C7N2O2Cl_ca': ['mP_1_01', [7.137, 16.408, 8.875, 90, 93.84, 90]], #acb
    'C7N2O2Cl_cb': ['mP_1_01', [7.137, 16.408, 8.875, 90, 93.84, 90]], #acb
    'C7N2O2Cl_da': ['mP_1_01', [7.137, 16.408, 8.875, 90, 93.84, 90]], #acb
    'C7N2O2Cl_db': ['mP_1_01', [7.137, 16.408, 8.875, 90, 93.84, 90]], #acb
    'C7N2O2Cl_ec': ['mP_1_01', [7.137, 16.408, 8.875, 90, 93.84, 90]], #acb
    'C7N2O2Cl_fd': ['mP_1_01', [7.137, 16.408, 8.875, 90, 93.84, 90]], #acb
}

"""
base_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/experimental_data/SACLA'
entry_tags = {
    '1napthyl': ['mP_4_01', [16.1859,    4.4307,  27.39048669, 90,  118.7605304, 90]],
    '26dimethyl': ['mP_4_01', [13.2746,  4.403,   28.54359645, 90,  107.5761866, 90]],
    '3hydroxy': ['mP_4_01', [15.4315,    5.8648,  23.33138477, 90,  142.5818928, 90]],
    '4MeO-Se': ['oC_2_00', [5.9488,  7.2798,  35.3671, 90,  90,  90]],
    'ag2c3': ['aP_00', [4.525,   10.28,   11.236,  93.51,   99.56,   90.88]],
    'ag2c4': ['mP_4_01', [13.8362,   4.5424,  20.97812518, 90,  111.2145451, 90]],
    'agstp': ['mP_4_01', [7.5127,    5.877,   30.2883, 90,  91.938,  90]],
    'cu5s': ['mP_1_01', [4.0446, 17.9233, 4.5574,  90,  102.367, 90]],
    'cuc4': ['mC_4_03', [8.9206, 4.0381,  32.98515788, 90,  105.6906205, 90]],
    'cuc6': ['mP_4_01', [20.6751,    4.0463,  22.53183514, 90,  156.512904,  90]],
    'CuS4MeO': ['oF_0_00', [4.0356,  27.4999, 52.1372, 90,  90,  90]],
    'glu_dehyd': ['oP_0_02', [4.6142,    14.1409, 27.9371, 90,  90,  90]],
    'glucose': ['oP_0_03', [4.6351,  14.3085, 27.9955, 90,  90,  90]],
    'homocys': ['mC_4_03', [9.9147,  4.6017,  31.02137747, 90,  104.7303099, 90]],
    'thiorene': ['mP_4_01', [7.3381, 5.9217,  28.6862439,  90,  101.0927744, 90]],
    '25dimethyl': ['tP_0_00', [5.1816,   5.1816,  28.9283, 90,  90,  90]],
    '2am5cl': ['mP_4_00', [7.4984,   6.5062,  15.0183, 90,  97.286,  90]],
    '2bromo': ['aP_00', [4.2775, 11.905,  16.0497, 120.932, 93.37,   97.054]],
    '2chloro': ['aP_00', [4.3862,    11.6123, 14.44926509, 102.8005258, 104.7643542, 93.04]],
    '2mercaptopyr': ['oP_0_03', [7.798,  10.5081, 21.1495, 90,  90,  90]],
    '2napth': ['oF_0_01', [7.473,    11.913,  75.454,  90,  90,  90]],
    '2napthyl': ['oP_0_03', [3.736,  5.955,   37.7277, 90,  90,  90]],
    '3meote': ['mC_4_02', [15.7821,  9.3675,  22.7802967,  90,  92.50215382, 90]],
    'Ag4C7': ['mC_4_02', [16.3295,   4.5673,  40.45569584, 90,  142.5181502, 90]],
    'Auc4t': ['oF_0_01', [6.0373,    6.9793,  29.82,   90,  90,  90]],
    'Auc5t': ['oF_0_01', [5.9917,    7.0536,  33.5171, 90,  90,  90]],
    'AuC6': ['oP_0_03', [3.0014, 3.5214,  39.0265, 90,  90,  90]],
    'Auc7t': ['oC_2_00', [5.956, 7.1153,  42.6654, 90,  90,  90]],
    'cc4t': ['tP_1_00', [15.5277, 15.5277, 4.5275,  90,  90,  90]],
    'cu7s': ['mP_1_01', [4.0557, 22.736,  4.5894,  90,  102.771, 90]],
    'cu8s': ['mP_1_00', [4.056,  25.605,  4.589,   90,  102.77,  90]],
    'Cuc4y': ['oP_0_03', [4.1302, 5.4783,  32.916,  90,  90,  90]],
    'cybu': ['tP_1_00', [14.623, 14.623,  4.248,   90,  90,  90]],
    'cyhx': ['tI_1_00', [17.6836, 17.6836, 4.6046,  90,  90,  90]],
    }

rng = np.random.default_rng()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

for entry_tag in entry_tags.keys():
    print(entry_tag)
    split_group = entry_tags[entry_tag][0]
    bravais_lattice = split_group.split('_')[0]
    if bravais_lattice in ['cI', 'cF', 'cP']:
        n_peaks = 10
    else:
        n_peaks = 20
    peak_list = np.load(base_dir + f'/{entry_tag}/{entry_tag}_peak_list.npy')[:n_peaks]

    unit_cell = np.array(entry_tags[entry_tag][1])
    unit_cell[3:] *= np.pi/180
    hkl_ref = np.load(f'/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/hkl_ref_{bravais_lattice}.npy')
    if bravais_lattice in ['cI', 'cF', 'cP']:
        lattice_system = 'cubic'
        unit_cell_partial = unit_cell[[0]]
    elif bravais_lattice in ['hP']:
        lattice_system = 'hexagonal'
        unit_cell_partial = unit_cell[[0, 2]]
    elif bravais_lattice in ['tP', 'tI']:
        lattice_system = 'tetragonal'
        unit_cell_partial = unit_cell[[0, 2]]
    elif bravais_lattice in ['hR']:
        lattice_system = 'rhombohedral'
        unit_cell_partial = unit_cell[[0, 3]]
    elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
        lattice_system = 'orthorhombic'
        unit_cell_partial = unit_cell[[0, 1, 2]]
    elif bravais_lattice in ['mC', 'mP']:
        lattice_system = 'monoclinic'
        unit_cell_partial = unit_cell[[0, 1, 2, 4]]
    else:
        lattice_system = 'triclinic'
        unit_cell_partial = unit_cell
    volume = get_unit_cell_volume(
        unit_cell_partial[np.newaxis],
        lattice_system=lattice_system,
        partial_unit_cell=True
        )[0]

    mpi_organizers = get_mpi_organizer(comm, [bravais_lattice], [0], [True])
    optimizer = get_optimizers(rank, mpi_organizers, broadening_tag, n_candidates_scale=1, logger=None)

    xnn_pred, template_unit_cells, volume_pred, tree_unit_cells = optimizer[bravais_lattice].perform_predictions(
        peak_list,
        split_group,
        100
        )
    xnn_pred = xnn_pred[0]
    volume_pred = volume_pred[0]

    unit_cell_pred = get_unit_cell_from_xnn(
        xnn_pred,
        partial_unit_cell=True,
        lattice_system=lattice_system
        )
    print(split_group)
    print('Integral Filter')
    print_results(
        base_dir, entry_tag, 'integral_filter', broadening_tag,
        lattice_system, unit_cell_partial, unit_cell_pred, False, True
        )
    print('Template')
    print_results(
        base_dir, entry_tag, 'template', broadening_tag,
        lattice_system, unit_cell_partial, template_unit_cells, False, False
        )
    print('RF')
    print_results(
        base_dir, entry_tag, 'rf', broadening_tag,
        lattice_system, unit_cell_partial, tree_unit_cells, True, False
        )

    volume_difference = np.abs(volume - volume_pred)
    volume_order = np.argsort(volume_difference)
    print('Volume')
    print(volume_pred.mean())
    print(volume_pred[volume_order][0])
    np.save(
        os.path.join(base_dir, entry_tag, f'{entry_tag}_volume_{broadening_tag}.npy'),
        np.array((volume_pred.mean(), volume_pred[volume_order][0]))
        )
    print()
