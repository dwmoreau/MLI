"""
extremely small error:
cF:  100% {'Not found': 0, 'Found': 238, 'Off by two': 0, 'Found explainers': 0}
cI: 99.6% {'Not found': 1, 'Found': 249, 'Off by two': 0, 'Found explainers': 0}
cP: 99.8% {'Not found': 1, 'Found': 533, 'Off by two': 0, 'Found explainers': 0}
hP: 98.6% {'Not found': 4, 'Found': 493, 'Off by two': 0, 'Found explainers': 3}
hR: 99.0% {'Not found': 5, 'Found': 495, 'Off by two': 0, 'Found explainers': 0}
tI: 98.8% {'Not found': 6, 'Found': 494, 'Off by two': 0, 'Found explainers': 0}
tP: 97.2% {'Not found': 3, 'Found': 486, 'Off by two': 0, 'Found explainers': 11}
oC: 
oF: 97.0% {'Not found': 7, 'Found': 485, 'Off by two': 0, 'Found explainers': 8}
oI:
oP: 98.5% {'Not found': 1, 'Found': 447, 'Off by two': 0, 'Found explainers': 6}
mC:
mP:
aP:
"""
from collections import namedtuple
from mpi4py import MPI
import numpy as np
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

from Optimizer_mpi import OptimizerManager
from Optimizer_mpi import OptimizerWorker

from UtilitiesOptimizer import get_cubic_optimizer
from UtilitiesOptimizer import get_hexagonal_optimizer
from UtilitiesOptimizer import get_monoclinic_optimizer
from UtilitiesOptimizer import get_orthorhombic_optimizer
from UtilitiesOptimizer import get_rhombohedral_optimizer
from UtilitiesOptimizer import get_tetragonal_optimizer
from UtilitiesOptimizer import get_triclinic_optimizer
from UtilitiesOptimizer import validate_candidate


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

broadening_tag = '0.5'
error_tag = '0.1'
n_entries = 2 # Maximum number of entries to load for each Bravais lattice
#q2_error_params = np.array([0.0001, 0.001]) / 1
q2_error_params = np.array([0.000000001, 0])
n_top_candidates = 20
rng = np.random.default_rng()
mpi_manager = namedtuple('mpi_manager', ['manager', 'workers', 'color', 'split_comm'])

# This setups a namedtuple (mpi_manager) that contains the information for the parallezation
# for each bravais lattice.
# comm.Split is described here fairly well:
#   https://stackoverflow.com/questions/50900655/mpi4py-create-multiple-groups-and-scatter-from-each-group
#   https://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
# After the split, the manager will have rank 0.

#bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',  'oC',  'oF',  'oI',  'oP',  'mC',  'mP',  'aP']
#manager_rank =     [   0,    1,    0,    1,    0,    1,    0,     1,     0,     1,     0,     1,     0,     1]
#serial =           [True, True, True, True, True, True, True, False, False, False, False, False, False, False]

#manager_rank =    [   0,    0,    1,    2,    3,    4,    5,     0,     0,     1,     2,     3,     4,     5]
#manager_rank =     [   0,    1,    2,    0,    1,    2,    0,     1,     2,     0,     1,     2,     0,     1]

bravais_lattices = ['cP', 'tP',  'oP']
manager_rank =     [   0,    1,     0]
serial =           [True, True, False]

data = dict.fromkeys(bravais_lattices)
optimizer = dict.fromkeys(bravais_lattices)
mpi_managers = dict.fromkeys(bravais_lattices)
serial_split_comm = comm.Split(color=rank, key=0)
for bl_index, bravais_lattice in enumerate(bravais_lattices):
    if serial[bl_index]:
        if rank == manager_rank[bl_index]:
            mpi_managers[bravais_lattice] = mpi_manager(
                manager_rank[bl_index],
                [manager_rank[bl_index]],
                manager_rank[bl_index],
                serial_split_comm
                )
        else:
            mpi_managers[bravais_lattice] = mpi_manager(
                manager_rank[bl_index],
                [manager_rank[bl_index]],
                rank,
                None
                )
    else:
        if rank == manager_rank[bl_index]:
            key = 0
        else:
            key = rank + 1
        mpi_managers[bravais_lattice] = mpi_manager(
            manager_rank[bl_index],
            [i for i in range(n_ranks)],
            bl_index,
            comm.Split(color=bl_index, key=key)
            )

# This next section loads the Optimizer objects
for bl_index, bravais_lattice in enumerate(bravais_lattices):
    if rank == mpi_managers[bravais_lattice].manager:
        #print('Manager', rank, mpi_managers[bravais_lattice], bravais_lattice)
        # These function calls return an OptimizerManager object
        if bravais_lattice in ['cF', 'cI', 'cP']:
            optimizer[bravais_lattice] = get_cubic_optimizer(
                bravais_lattice,
                broadening_tag,
                error_tag,
                mpi_managers[bravais_lattice].split_comm,
                )
        elif bravais_lattice in ['hP']:
            optimizer[bravais_lattice] = get_hexagonal_optimizer(
                bravais_lattice,
                broadening_tag,
                error_tag,
                mpi_managers[bravais_lattice].split_comm,
                )
        elif bravais_lattice in ['hR']:
            optimizer[bravais_lattice] = get_rhombohedral_optimizer(
                bravais_lattice,
                broadening_tag,
                error_tag,
                mpi_managers[bravais_lattice].split_comm,
                )
        elif bravais_lattice in ['tI', 'tP']:
            optimizer[bravais_lattice] = get_tetragonal_optimizer(
                bravais_lattice,
                broadening_tag,
                error_tag,
                mpi_managers[bravais_lattice].split_comm,
                )
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer[bravais_lattice] = get_orthorhombic_optimizer(
                bravais_lattice,
                broadening_tag,
                error_tag,
                mpi_managers[bravais_lattice].split_comm,
                )
        elif bravais_lattice in ['mC', 'mP']:
            optimizer[bravais_lattice] = get_monoclinic_optimizer(
                bravais_lattice,
                broadening_tag,
                error_tag,
                mpi_managers[bravais_lattice].split_comm,
                )
        elif bravais_lattice in ['aP']:
            optimizer[bravais_lattice] = get_triclinic_optimizer(
                bravais_lattice,
                broadening_tag,
                error_tag,
                mpi_managers[bravais_lattice].split_comm,
                )
    elif rank in mpi_managers[bravais_lattice].workers:
        #print('Worker', rank, mpi_managers[bravais_lattice], bravais_lattice)
        optimizer[bravais_lattice] = OptimizerWorker(mpi_managers[bravais_lattice].split_comm)

# This loads the data for each bravais lattice:
read_columns = [
    'lattice_system',
    'bravais_lattice',
    'train',
    f'q2_{broadening_tag}',
    f'reindexed_h_{broadening_tag}',
    f'reindexed_k_{broadening_tag}',
    f'reindexed_l_{broadening_tag}',
    'reindexed_spacegroup_symbol_hm',
    'reindexed_unit_cell',
    ]
drop_columns = [
    f'q2_{broadening_tag}',
    f'reindexed_h_{broadening_tag}',
    f'reindexed_k_{broadening_tag}',
    f'reindexed_l_{broadening_tag}',
    ]
for bl_index, bravais_lattice in enumerate(bravais_lattices):
    if rank == mpi_managers[bravais_lattice].manager:
        n_peaks = 20
        bravais_lattice_data = pd.read_parquet(
            f'/Users/DWMoreau/MLI/data/GeneratedDatasets/dataset_{bravais_lattice}.parquet',
            columns=read_columns
            )
        bravais_lattice_data = bravais_lattice_data.loc[~bravais_lattice_data['train']]
        peaks = bravais_lattice_data[f'q2_{broadening_tag}']
        bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(len) >= n_peaks]
        peaks = bravais_lattice_data[f'q2_{broadening_tag}']
        bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(np.count_nonzero) >= n_peaks]
        q2 = np.zeros((bravais_lattice_data.shape[0], n_peaks))
        hkl = np.zeros((bravais_lattice_data.shape[0], n_peaks, 3))
        for entry_index in range(bravais_lattice_data.shape[0]):
            q2[entry_index] = np.array(bravais_lattice_data[f'q2_{broadening_tag}'].iloc[entry_index])[:n_peaks]
            hkl[entry_index, :, 0] = np.array(bravais_lattice_data[f'reindexed_h_{broadening_tag}'].iloc[entry_index])[:n_peaks]
            hkl[entry_index, :, 1] = np.array(bravais_lattice_data[f'reindexed_k_{broadening_tag}'].iloc[entry_index])[:n_peaks]
            hkl[entry_index, :, 2] = np.array(bravais_lattice_data[f'reindexed_l_{broadening_tag}'].iloc[entry_index])[:n_peaks]
        sigma_error = q2_error_params[0] + q2 * q2_error_params[1]
        q2 += rng.normal(loc=0, scale=sigma_error)
        bravais_lattice_data['q2'] = list(q2)
        bravais_lattice_data['reindexed_hkl'] = list(hkl)
        bravais_lattice_data.drop(columns=drop_columns, inplace=True)
        if len(bravais_lattice_data) > n_entries:
            bravais_lattice_data = bravais_lattice_data.sample(
                n=n_entries, replace=False, random_state=rng, ignore_index=True
                )
        data[bravais_lattice] = bravais_lattice_data

report_counts = {
    'Not found': 0,
    'Found': 0,
    'Off by two': 0,
    'Incorrect BL': 0,
    'Found explainers': 0,
    }

for bl_index_data, bravais_lattice_data in enumerate(bravais_lattices):
    if rank == mpi_managers[bravais_lattice_data].manager:
        n_entries_bl = len(data[bravais_lattice_data])
    else:
        n_entries_bl = None
    n_entries_bl = comm.bcast(n_entries_bl, root=mpi_managers[bravais_lattice_data].manager)
    
    for entry_index in range(n_entries_bl):
        if rank == 0:
            top_unit_cell = dict.fromkeys(bravais_lattices)
            top_M20 = dict.fromkeys(bravais_lattices)
        top_unit_cell_bl = dict.fromkeys(bravais_lattices)
        top_M20_bl = dict.fromkeys(bravais_lattices)

        if rank == mpi_managers[bravais_lattice_data].manager:
            entry = data[bravais_lattice_data].iloc[entry_index]
        else:
            entry = None
        entry = comm.bcast(entry, root=mpi_managers[bravais_lattice_data].manager)

        for bravais_lattice in bravais_lattices:
            if rank == mpi_managers[bravais_lattice].manager:
                top_unit_cell_bl[bravais_lattice], top_M20_bl[bravais_lattice] = \
                    optimizer[bravais_lattice].run(entry, n_top_candidates)
            elif rank in mpi_managers[bravais_lattice].workers:
                optimizer[bravais_lattice].run()

        comm.barrier()
        for bravais_lattice in bravais_lattices:
            if rank == 0 and mpi_managers[bravais_lattice].manager == 0:
                top_unit_cell[bravais_lattice] = top_unit_cell_bl[bravais_lattice]
                top_M20[bravais_lattice] = top_M20_bl[bravais_lattice]
            else:
                if rank == 0:
                    top_unit_cell[bravais_lattice] = comm.recv(source=mpi_managers[bravais_lattice].manager)
                    top_M20[bravais_lattice] = comm.recv(source=mpi_managers[bravais_lattice].manager)
                elif rank == mpi_managers[bravais_lattice].manager:
                    comm.send(top_unit_cell_bl[bravais_lattice], dest=0)
                    comm.send(top_M20_bl[bravais_lattice], dest=0)

        if rank == 0:
            found, off_by_two, incorrect_bl, found_explainer = \
                validate_candidate(entry, top_unit_cell, top_M20)
            if found:
                report_counts['Found'] += 1
            elif incorrect_bl:
                report_counts['Incorrect BL'] += 1
            elif off_by_two:
                report_counts['Off by two'] += 1
            elif found_explainer:
                report_counts['Found explainers'] += 1
            else:
                report_counts['Not found'] += 1
            print(report_counts)
            print()
