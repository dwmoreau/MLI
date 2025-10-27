import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from mpi4py import MPI
import numpy as np
import pandas as pd

from mlindex.optimization.UtilitiesOptimizer import get_logger
from mlindex.optimization.UtilitiesOptimizer import get_mpi_organizer
from mlindex.optimization.UtilitiesOptimizer import get_optimizers
from mlindex.optimization.CandidateValidation import validate_candidate


broadening_tag = '1'
optimization_tag = '_0'
n_top_candidates = 20

"""
base_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/experimental_data/SACLA'
entry_tags = [
    '1napthyl',
    '2am5cl',
    '2bromo',
    '2chloro',
    '2mercaptopyr',
    '2napth',
    '2napthyl',
    '3hydroxy',
    '3meote',
    '4MeO-Se',
    '24-dimethyl',
    '25dimethyl',
    '26dimethyl',
    'Ag2BrPh',
    'ag2c3',
    'ag2c4',
    'Ag2C5',
    'Ag2ClPh',
    'Ag3C6',
    'Ag4C7',
    'agstp',
    'Auc4t',
    'Auc5t',
    'AuC6',
    'Auc7t',
    'AuC8',
    'benz',
    'cc4t',
    'cu5s',
    'cu7s',
    'cu8s',
    'cuc4',
    'Cuc4w',
    'Cuc4y',
    'cuc6',
    'Cuc8',
    'CuS3MeO',
    'CuS4MeO',
    'cybu',
    'cyhx',
    'cypn',
    'glu_dehyd',
    'glucose',
    'homocys',
    'Lcys-lowph',
    'thiorene',
    ]
"""

base_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/experimental_data/gsas'
entry_tags = [
    '11bmb_3844',
    '11bmb_6231',
    '11bmb_8716',
    'Carbidopa',
    'CuCr2O4',
    'FAP',
    'garnet',
    'La7Ca3MnO3_50K',
    'LaMnO3_50K',
    'PBSO4',
    #'C7N2O2Cl_ca',
    #'C7N2O2Cl_cb',
    #'C7N2O2Cl_da',
    #'C7N2O2Cl_db',
    #'C7N2O2Cl_ec',
    #'C7N2O2Cl_fd',
]


rng = np.random.default_rng()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()
logger = get_logger(comm, optimization_tag)
logger.info('Starting process')

#bravais_lattices = ['cP', 'hP', 'hR', 'tP', 'oP', 'mP', 'aP']
#manager_rank =     [   0,    0,    0,    0,    0,    0,    0]
#serial =           [True, True, True, True, True, True, True]

bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',  'oC',  'oF',  'oI',  'oP',  'mC',  'mP',  'aP']
manager_rank =     [   0,    0,    0,    1,    2,    3,    4,     1,     2,     3,     4,     5,     0,     5]
serial =           [True, True, True, True, True, True, True, False, False, False, False, False, False, False]

#bravais_lattices = [ 'oC',  'oF',  'oI',  'oP']
#manager_rank =     [    0,     0,     1,     1]
#serial =           [False, False, False, False]

#bravais_lattices = ['cF', 'cI', 'cP']
#manager_rank =     [   0,    0,    0]
#serial =           [True, True, True]

#bravais_lattices = ['cP']
#manager_rank =     [   0]
#serial =           [True]

#bravais_lattices = ['tP', 'tI']
#manager_rank =     [   0, 0]
#serial =           [True, True]

#bravais_lattices = ['tI']
#manager_rank =     [  0]
#serial =           [True]

#bravais_lattices = [ 'mP']
#manager_rank =     [    0]
#serial =           [False]

#bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',  'oC',  'oF',  'oI',  'oP',  'mC',  'mP', 'aP']
#manager_rank =     [   0,    0,    0,    0,    0,    0,    0,     0,     0,     0,     0,     0,     0,    0]
#serial =           [True, True, True, True, True, True, True,  True,  True,  True,  True,  True,  True, True]

mpi_organizers = get_mpi_organizer(comm, bravais_lattices, manager_rank, serial)

bl_string = ''
for bl in bravais_lattices:
    bl_string += f' {bl}'
logger.info(f'Including Bravais lattices {bl_string}')
logger.info('Starting loading optimizers')
optimizer = get_optimizers(rank, mpi_organizers, broadening_tag, n_candidates_scale=1, logger=logger)

for entry_tag in entry_tags:
    logger.info(f'Starting entry {entry_tag}')
    peak_list = np.load(base_dir + f'/{entry_tag}/{entry_tag}_peak_list.npy')[:20]
    triplet_file_name = base_dir + f'/{entry_tag}/{entry_tag}_triplets.npy'
    if os.path.exists(triplet_file_name):
        triplet_obs = np.load(triplet_file_name)
        logger.info(f'Entry {entry_tag} has triplet information')
        triplet_obs = None
    else:
        triplet_obs = None
    # The next portion is the optimization of a single entry
    # rank 0 will be the rank that compares results from all bravais lattices. 
    if rank == 0:
        top_unit_cell = dict.fromkeys(bravais_lattices)
        top_M20 = dict.fromkeys(bravais_lattices)
        top_Minfo = dict.fromkeys(bravais_lattices)
        top_spacegroup = dict.fromkeys(bravais_lattices)
        top_n_indexed = dict.fromkeys(bravais_lattices)
        if not triplet_obs is None:
            top_M_triplets = dict.fromkeys(bravais_lattices)
            top_n_indexed_triplets = dict.fromkeys(bravais_lattices)

    # This loop optimizes the entry given an assumed bravais lattice.
    for bravais_lattice in bravais_lattices:
        if rank in mpi_organizers[bravais_lattice].workers:
            if rank == mpi_organizers[bravais_lattice].manager:
                role = 'manager'
            else:
                role = 'worker'
            mpi_organizers[bravais_lattice].split_comm.barrier()
            logger.info(f'Starting optimization of {bravais_lattice} {role}')
            optimizer[bravais_lattice].run(q2=peak_list, triplets=triplet_obs, n_top_candidates=n_top_candidates)
            logger.info(f'Finishing optimization of {bravais_lattice} {role}')
    comm.barrier()

    # Gather the optimization results
    logger.info(f'Starting gathering optimization results of {entry_tag}')
    for bravais_lattice in bravais_lattices:
        if rank == 0 and mpi_organizers[bravais_lattice].manager == 0:
            top_unit_cell[bravais_lattice] = optimizer[bravais_lattice].top_unit_cell
            top_M20[bravais_lattice] = optimizer[bravais_lattice].top_M20
            top_Minfo[bravais_lattice] = optimizer[bravais_lattice].top_Minfo
            top_spacegroup[bravais_lattice] = optimizer[bravais_lattice].top_spacegroup
            top_n_indexed[bravais_lattice] = optimizer[bravais_lattice].top_n_indexed
            if not triplet_obs is None:
                top_n_indexed_triplets[bravais_lattice] = optimizer[bravais_lattice].top_n_indexed_triplets
                top_M_triplets[bravais_lattice] = optimizer[bravais_lattice].top_M_triplets
        else:
            if rank == 0:
                top_unit_cell[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_M20[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_Minfo[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_spacegroup[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                top_n_indexed[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                if not triplet_obs is None:
                    top_n_indexed_triplets[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
                    top_M_triplets[bravais_lattice] = comm.recv(source=mpi_organizers[bravais_lattice].manager)
            elif rank == mpi_organizers[bravais_lattice].manager:
                comm.send(optimizer[bravais_lattice].top_unit_cell, dest=0)
                comm.send(optimizer[bravais_lattice].top_M20, dest=0)
                comm.send(optimizer[bravais_lattice].top_Minfo, dest=0)
                comm.send(optimizer[bravais_lattice].top_spacegroup, dest=0)
                comm.send(optimizer[bravais_lattice].top_n_indexed, dest=0)
                if not triplet_obs is None:
                    comm.send(optimizer[bravais_lattice].top_n_indexed_triplets, dest=0)
                    comm.send(optimizer[bravais_lattice].top_M_triplets, dest=0)
    if rank == 0:
        output_data = []
        for bravais_lattice in bravais_lattices:
            for result_index in range(top_M20[bravais_lattice].size):
                partial_unit_cell = top_unit_cell[bravais_lattice][result_index]
                if bravais_lattice in ['cF', 'cI', 'cP']:
                    unit_cell = np.array([
                        partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[0],
                        np.pi/2, np.pi/2, np.pi/2
                        ])
                elif bravais_lattice == 'hP':
                    unit_cell = np.array([
                        partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[1],
                        2*np.pi/3, np.pi/2, np.pi/2
                        ])
                elif bravais_lattice == 'hR':
                    unit_cell = np.array([
                        partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[0],
                        partial_unit_cell[1], partial_unit_cell[1], partial_unit_cell[1],
                        ])
                elif bravais_lattice in ['tI', 'tP']:
                    unit_cell = np.array([
                        partial_unit_cell[0], partial_unit_cell[0], partial_unit_cell[1],
                        np.pi/2, np.pi/2, np.pi/2
                        ])
                elif bravais_lattice in ['oC',  'oF',  'oI',  'oP']:
                    unit_cell = np.array([
                        partial_unit_cell[0], partial_unit_cell[1], partial_unit_cell[2],
                        np.pi/2, np.pi/2, np.pi/2
                        ])
                elif bravais_lattice in ['mC',  'mP']:
                    unit_cell = np.array([
                        partial_unit_cell[0], partial_unit_cell[1], partial_unit_cell[2],
                        np.pi/2, partial_unit_cell[3], np.pi/2
                        ])
                elif bravais_lattice == 'aP':
                    unit_cell = partial_unit_cell
                if triplet_obs is None:
                    n_indexed_triplets = 0
                else:
                    n_indexed_triplets = top_n_indexed_triplets[bravais_lattice][result_index]
                if triplet_obs is None:
                    M_triplet_output = None
                    n_indexed_triplets_output = None
                else:
                    M_triplet_output = list(top_M_triplets[bravais_lattice][result_index])
                    n_indexed_triplets_output = n_indexed_triplets
                output_data.append({
                    'M20': top_M20[bravais_lattice][result_index],
                    'Minfo': top_Minfo[bravais_lattice][result_index],
                    'n_indexed': top_n_indexed[bravais_lattice][result_index],
                    'M_triplet': M_triplet_output,
                    'n_indexed_triplet': n_indexed_triplets_output,
                    'bravais_lattice': bravais_lattice,
                    'spacegroup': top_spacegroup[bravais_lattice][result_index],
                    'a': unit_cell[0],
                    'b': unit_cell[1],
                    'c': unit_cell[2],
                    'alpha': unit_cell[3],
                    'beta': unit_cell[4],
                    'gamma': unit_cell[5],
                    })
                print(output_data[-1])
                print()
        output_df = pd.DataFrame(output_data)
        output_df.sort_values(by='M20', ascending=False, inplace=True, ignore_index=True)
        output_df.to_json(base_dir + f'/{entry_tag}/{entry_tag}_indexing_results_{optimization_tag}.json')
    logger.info(f'Finished gathering optimization results of {entry_tag}')
