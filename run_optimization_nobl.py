"""
extremely small error:
cF: 
cI: 
cP: 
hP: 
hR: 
tI: 
tP: 
oC: 
oF: 
oI:
oP: 
mC:
mP:
aP:
"""
from collections import namedtuple
import logging
from mpi4py import MPI
import numpy as np
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

from MPIFileHandler import MPIFileHandler
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

logger = logging.getLogger(f'rank[{comm.rank}]')
logger.setLevel(logging.DEBUG)                                                 
mh = MPIFileHandler('logfile.log')
mh.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))                                                
logger.addHandler(mh)
logger.info('Starting process')

broadening_tag = '0.5'
error_tag = '0.1'
n_entries = 1 # Maximum number of entries to load for each Bravais lattice
#q2_error_params = np.array([0.0001, 0.001]) / 1
q2_error_params = np.array([0.000000001, 0])
n_top_candidates = 20
rng = np.random.default_rng()

"""
This setups a namedtuple (mpi_manager) that contains the information for the parallezation
for each bravais lattice.
comm.Split is described here fairly well:
  https://stackoverflow.com/questions/50900655/mpi4py-create-multiple-groups-and-scatter-from-each-group
  https://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
After the split, the manager will have rank 0.
Right now there are two splits. One is for a serial communicator. This will be used for Bravais
lattices that are optimized serially. The next is for a joint communicator. This will be used for
Bravais lattices optimized in parallel. 
"""

#bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',  'oC',  'oF',  'oI',  'oP',  'mC',  'mP',  'aP']
#manager_rank =     [   0,    0,    1,    2,    3,    4,    5,     1,     2,     3,     4,     5,     0,     1]
#serial =           [True, True, True, True, True, True, True, False, False, False, False, False, False, False]

#manager_rank =    [   0,    0,    1,    2,    3,    4,    5,     0,     0,     1,     2,     3,     4,     5]
#manager_rank =     [   0,    1,    2,    0,    1,    2,    0,     1,     2,     0,     1,     2,     0,     1]

#bravais_lattices = ['cP', 'tP',  'oP']
#manager_rank =     [   0,    1,     0]
#serial =           [True, True, False]

bravais_lattices = ['cP', 'tP',  'oP']
manager_rank =     [   0,    1,     0]
serial =           [True, True, False]
bl_string = ''
for bl in bravais_lattices:
    bl_string += f' {bl}'
logger.info(f'Including Bravais lattices {bl_string}')

data = dict.fromkeys(bravais_lattices)
optimizer = dict.fromkeys(bravais_lattices)
if rank == 0:
    report_counts = dict.fromkeys(bravais_lattices)
    for bravais_lattice in bravais_lattices:
        report_counts[bravais_lattice] = {
            'Not found': 0,
            'Found': 0,
            'Off by two': 0,
            'Incorrect BL': 0,
            'Found explainers': 0,
            }

mpi_manager = namedtuple('mpi_manager', ['manager', 'workers', 'color', 'split_comm'])
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

logger.info('Starting loading optimizers')
# This next section loads the Optimizer objects
for bl_index, bravais_lattice in enumerate(bravais_lattices):
    if rank == mpi_managers[bravais_lattice].manager:
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
        logger.info(f'Loaded manager optimizer for {bravais_lattice}')
    elif rank in mpi_managers[bravais_lattice].workers:
        optimizer[bravais_lattice] = OptimizerWorker(mpi_managers[bravais_lattice].split_comm)
        logger.info(f'Loaded worker optimizer for {bravais_lattice}')

# This loads the data for each bravais lattice. Only manager ranks load data for a bravais lattice
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
logger.info(f'Starting data loading')
for bl_index, bravais_lattice in enumerate(bravais_lattices):
    if rank == mpi_managers[bravais_lattice].manager:
        #if bravais_lattice in ['cF', 'cI', 'cP']:
        #    n_peaks = 10
        #else:
        #    n_peaks = 20
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
        logger.info(f'Loaded data for {bravais_lattice}')


# The first two loops are over data. First, bravais lattice is looped through then entries
# with that bravais lattice
# Loop over data bravais lattice
for bl_index_data, bravais_lattice_data in enumerate(bravais_lattices):
    if rank == mpi_managers[bravais_lattice_data].manager:
        n_entries_bl = len(data[bravais_lattice_data])
    else:
        n_entries_bl = None
    n_entries_bl = comm.bcast(n_entries_bl, root=mpi_managers[bravais_lattice_data].manager)
    # loop over entries with a common bravais lattice
    for entry_index in range(n_entries_bl):
        logger.info(f'Starting entry {entry_index} of {bravais_lattice_data}')
        # The next portion is the optimization of a single entry
        # rank 0 will be the rank that compares results from all bravais lattices. 
        if rank == 0:
            top_unit_cell = dict.fromkeys(bravais_lattices)
            top_M20 = dict.fromkeys(bravais_lattices)

        if rank == mpi_managers[bravais_lattice_data].manager:
            entry = data[bravais_lattice_data].iloc[entry_index]
        else:
            entry = None
        entry = comm.bcast(entry, root=mpi_managers[bravais_lattice_data].manager)

        # This loop optimizes the entry given an assumed bravais lattice.
        for bravais_lattice in bravais_lattices:
            if rank in mpi_managers[bravais_lattice].workers:
                if rank == mpi_managers[bravais_lattice].manager:
                    role = 'manager'
                else:
                    role = 'worker'
                logger.info(f'Starting optimization of {bravais_lattice} {role}')
                optimizer[bravais_lattice].run(entry, n_top_candidates)
                logger.info(f'Finishing optimization of {bravais_lattice} {role}')
        comm.barrier()

        # Gather the optimization results
        logger.info(f'Starting gathering optimization results {entry_index} of {bravais_lattice_data}')
        for bravais_lattice in bravais_lattices:
            if rank == 0 and mpi_managers[bravais_lattice].manager == 0:
                top_unit_cell[bravais_lattice] = optimizer[bravais_lattice].top_unit_cell
                top_M20[bravais_lattice] = optimizer[bravais_lattice].top_M20
            else:
                if rank == 0:
                    top_unit_cell[bravais_lattice] = comm.recv(source=mpi_managers[bravais_lattice].manager)
                    top_M20[bravais_lattice] = comm.recv(source=mpi_managers[bravais_lattice].manager)
                elif rank == mpi_managers[bravais_lattice].manager:
                    comm.send(optimizer[bravais_lattice].top_unit_cell, dest=0)
                    comm.send(optimizer[bravais_lattice].top_M20, dest=0)
        logger.info(f'Finished gathering optimization results {entry_index} of {bravais_lattice_data}')

        # do a final validation
        if rank == 0:
            found, off_by_two, incorrect_bl, found_explainer = \
                validate_candidate(entry, top_unit_cell, top_M20)
            if found:
                report_counts[bravais_lattice_data]['Found'] += 1
            elif incorrect_bl:
                report_counts[bravais_lattice_data]['Incorrect BL'] += 1
            elif off_by_two:
                report_counts[bravais_lattice_data]['Off by two'] += 1
            elif found_explainer:
                report_counts[bravais_lattice_data]['Found explainers'] += 1
            else:
                report_counts[bravais_lattice_data]['Not found'] += 1
            print(report_counts)
            print()
        logger.info(f'Finished entry {entry_index} of {bravais_lattice_data}')

