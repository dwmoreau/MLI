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
from mlindex.scripts.run_real_benchmark import peak_list as q2_from_peak_positions
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
opxrd = True

#failed_entries = np.load(os.path.join(base_dir, 'failed_entries.npy'))
failed_entries = [37, 57, 110, 251, 337, 467, 496, 508]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()
logger = get_logger(comm, optimization_tag)
logger.info('Starting process')

# Each rank indexes its own share of the entries from start to finish: it manages every lattice
# itself, serially, and writes its own results. The list is read and filtered once, on rank 0, so
# every rank splits the same list.
if opxrd:
    if rank == 0:
        df = pd.read_json(os.path.join(base_dir, 'CNRS_output_data_verified_final2.json'))
        # Entries already indexed by an earlier run are skipped.
        opt_files = df['file_name'].str.replace('.json', '_opt.json', regex=False)
        df = df[~opt_files.apply(os.path.exists)]
        shares = [df.iloc[i::n_ranks] for i in range(n_ranks)]
    else:
        shares = None
    entries = comm.scatter(shares, root=0)
else:
    entries = entry_tags[rank::n_ranks]

#bravais_lattices = ['cP', 'hP', 'hR', 'tP', 'oP', 'mP', 'aP']
bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',  'oC',  'oF',  'oI',  'oP',  'mC',  'mP',  'aP']
#bravais_lattices = [ 'oC',  'oF',  'oI',  'oP']
#bravais_lattices = ['cF', 'cI', 'cP']
#bravais_lattices = ['tP']
#bravais_lattices = ['tP', 'tI']
#bravais_lattices = ['tI']
#bravais_lattices = [ 'mP']
manager_rank = [rank for _ in bravais_lattices]
serial = [True for _ in bravais_lattices]

mpi_organizers = get_mpi_organizer(comm, bravais_lattices, manager_rank, serial)

bl_string = ''
for bl in bravais_lattices:
    bl_string += f' {bl}'
logger.info(f'Including Bravais lattices {bl_string}')
logger.info('Starting loading optimizers')
optimizer = get_optimizers(rank, mpi_organizers, broadening_tag, n_candidates_scale=1, logger=logger)
# Falls back to ranking by M_sym, and says so in its name, when no learned ranker is found.
ranker = load_ranker()
logger.info(f'Ranking by {ranker.name}')

#for index in failed_entries:
for index in range(len(entries)):
    if opxrd:
        entry = entries.iloc[index]
        print(entry)
        entry_tag = entry.file_name
        # Read as the real-data benchmark reads them: distinct peaks, the first 20, q -> q2.
        peak_list = q2_from_peak_positions(entry.peak_positions)
        if zero_error:
            wavelength = entry.primary_wavelength
        else:
            wavelength = None
    else:
        entry_tag = entries[index]
        peak_list = np.load(os.path.join(base_dir, entry_tag, f'{entry_tag}_peak_list.npy'))[:20]
        #wavelength = None
        #zero_error = False
        wavelength = 1.5418
        zero_error = True

    logger.info(f'Starting entry {entry_tag}')

    # This loop optimizes the entry given an assumed bravais lattice.
    results = dict.fromkeys(bravais_lattices)
    for bravais_lattice in bravais_lattices:
        logger.info(f'Starting optimization of {bravais_lattice}')
        optimizer[bravais_lattice].run(
            q2=peak_list,
            n_top_candidates=n_top_candidates,
            zero_error=zero_error,
            wavelength=wavelength
        )
        results[bravais_lattice] = optimizer[bravais_lattice].result()
        logger.info(f'Finishing optimization of {bravais_lattice}')

    # The ranked list the indexer prints: every lattice's pool scored together, each
    # lattice's shortlist, repeated cells removed. Columns include score, M20 and M_sym;
    # angles are in degrees.
    output_df = pd.DataFrame(printed_rows(results, ranker))
    if opxrd:
        file_name = entry.file_name.replace('.json', '_opt.json')
    else:
        file_name = os.path.join(base_dir, entry_tag, f'{entry_tag}_indexing_results_{optimization_tag}.json')
    print(file_name)
    output_df.to_json(file_name)
    if opxrd:
        print(entry.bravais_lattice, entry.unit_cell)
        for i in range(min(5, len(output_df))):
            print(output_df.iloc[i])
            print()
    logger.info(f'Finished entry {entry_tag}')
