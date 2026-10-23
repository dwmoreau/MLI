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
from mlindex.command_line.run import printed_rows
from mlindex.utilities.Ranker import load_ranker


broadening_tag = '1'
optimization_tag = '_0'
n_top_candidates = 20

opxrd = False
zero_error = True

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
    #'11bmb_3844',
    #'11bmb_6231',
    #'11bmb_8716',
    #'Carbidopa',
    #'CuCr2O4',
    #'FAP',
    #'garnet',
    #'La7Ca3MnO3_50K',
    #'LaMnO3_50K',
    #'PBSO4',
    'C7N2O2Cl_aa',
    'C7N2O2Cl_ab',
    'C7N2O2Cl_ba',
    'C7N2O2Cl_bb',
    #'C7N2O2Cl_ca',
    #'C7N2O2Cl_cb',
    #'C7N2O2Cl_da',
    #'C7N2O2Cl_db',
    #'C7N2O2Cl_ec',
    #'C7N2O2Cl_fd',
]
"""

base_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/opxrd'
df = pd.read_json(os.path.join(base_dir, 'CNRS_output_data_verified_final2.json'))
opxrd = True
"""
#failed_entries = np.load(os.path.join(base_dir, 'failed_entries.npy'))
failed_entries = [37, 57, 110, 251, 337, 467, 496, 508]
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
##manager_rank =     [   0,    0,    0,    1,    2,    3,    4,     1,     2,     3,     4,     5,     0,     5]
##serial =           [True, True, True, True, True, True, True, False, False, False, False, False, False, False]
manager_rank =     [   0,    0,    0,    0,    0,    0,    0,     0,     0,     0,     0,     0,     0,     0]
serial =           [True, True, True, True, True, True, True, True, True, True, True, True, True, True]

#bravais_lattices = [ 'oC',  'oF',  'oI',  'oP']
#manager_rank =     [    0,     0,     1,     1]
#serial =           [False, False, False, False]

#bravais_lattices = ['cF', 'cI', 'cP']
#manager_rank =     [   0,    0,    0]
#serial =           [True, True, True]

#bravais_lattices = ['tP']
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
if rank == 0:
    # Falls back to ranking by M_sym, and says so in its name, when no learned ranker is found.
    ranker = load_ranker()
    logger.info(f'Ranking by {ranker.name}')

if opxrd:
    n = len(df)
else:
    n = len(entry_tags)
#for index in failed_entries:
for index in range(n):
    if opxrd:
        entry = df.iloc[index]
        print(entry)
        entry_tag = entry.file_name
        peak_list = np.array(entry.peak_positions)[:20]**2
        if zero_error:
            wavelength = entry.primary_wavelength
        else:
            wavelength = None
    else:
        entry_tag = entry_tags[index]
        peak_list = np.load(base_dir + f'/{entry_tag}/{entry_tag}_peak_list.npy')[:20]
        #wavelength = None
        #zero_error = False
        wavelength = 1.5418
        zero_error = True

    logger.info(f'Starting entry {entry_tag}')

    # The next portion is the optimization of a single entry
    # rank 0 will be the rank that compares results from all bravais lattices.
    if rank == 0:
        results = dict.fromkeys(bravais_lattices)

    # This loop optimizes the entry given an assumed bravais lattice.
    for bravais_lattice in bravais_lattices:
        if rank in mpi_organizers[bravais_lattice].workers:
            if rank == mpi_organizers[bravais_lattice].manager:
                role = 'manager'
            else:
                role = 'worker'
            mpi_organizers[bravais_lattice].split_comm.barrier()
            logger.info(f'Starting optimization of {bravais_lattice} {role}')
            optimizer[bravais_lattice].run(
                q2=peak_list,
                n_top_candidates=n_top_candidates,
                zero_error=zero_error,
                wavelength=wavelength
            )
            logger.info(f'Finishing optimization of {bravais_lattice} {role}')
    comm.barrier()

    # Gather the optimization results
    logger.info(f'Starting gathering optimization results of {entry_tag}')
    for bravais_lattice in bravais_lattices:
        manager = mpi_organizers[bravais_lattice].manager
        if rank == 0 and manager == 0:
            results[bravais_lattice] = optimizer[bravais_lattice].result()
        elif rank == 0:
            results[bravais_lattice] = comm.recv(source=manager)
        elif rank == manager:
            comm.send(optimizer[bravais_lattice].result(), dest=0)
    if rank == 0:
        # The ranked list the indexer prints: every lattice's pool scored together, each
        # lattice's shortlist, repeated cells removed. Columns include score, M20 and M_sym;
        # angles are in degrees.
        output_df = pd.DataFrame(printed_rows(results, ranker))
        if opxrd:
            file_name = entry.file_name.replace('.json', '_opt.json')
        else:
            file_name = base_dir + f'/{entry_tag}/{entry_tag}_indexing_results_{optimization_tag}.json'
        output_df.to_json(file_name)
        if opxrd:
            print(entry.bravais_lattice, entry.unit_cell)
            for i in range(min(5, len(output_df))):
                print(output_df.iloc[i])
                print()
    logger.info(f'Finished gathering optimization results of {entry_tag}')
