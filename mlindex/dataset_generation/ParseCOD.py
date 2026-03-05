from mpi4py import MPI
import numpy as np
import os
import pandas as pd

from ParseDatabases import ProcessCODEntry
from RemoveDuplicates import remove_duplicates
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell

def get_next_dirs(dir):
    next_dirs = []
    for d in os.listdir(dir):
        if not d.startswith('.'):
            next_dirs.append(d)
    return next_dirs

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
n_ranks = COMM.Get_size()
# MPI approach
#   Each rank reads different entries
#   Output data frames are saved to different .json files
cod_dir = '/Users/DWMoreau/cod_cifs'
opxrd_dir = '/Users/DWMoreau/MLI/mlindex/data/opxrd'
if rank == 0:
    cif_file_names = []
    for dir_0_index in range(1, 10):
        dir_0 = os.path.join(cod_dir, str(dir_0_index))
        for dir_1_index in get_next_dirs(dir_0):
            dir_1 = os.path.join(dir_0, dir_1_index)
            for dir_2_index in get_next_dirs(dir_1):
                dir_2 = os.path.join(dir_1, dir_2_index)
                for cif_file_name in get_next_dirs(dir_2):
                    if not cif_file_name.endswith('_editted.cif'):
                        cif_file_names.append(os.path.join(dir_2, cif_file_name))
else:
    cif_file_names = None
cif_file_names = COMM.bcast(cif_file_names, root=0)
n_total = len(cif_file_names)

dicts = []
failed_dicts = []
output_dicts = []
cnrs_dicts = []
in_numeric_tag = False
duplicate_base = ''

cnrs_xnn = get_xnn_from_unit_cell(np.load(os.path.join(opxrd_dir, 'CNRS_lattices.npy')))
cnrs_tolerance = 0.0001
for index in range(rank, n_total, n_ranks):
    entry = ProcessCODEntry()
    entry.verify_entry(cif_file_names[index])
    entry.make_output_dict()

    if entry.status:
        entry_xnn = get_xnn_from_unit_cell(entry.output_dict['unit_cell'][np.newaxis])
        difference = np.nanmin(np.linalg.norm(entry_xnn - cnrs_xnn, axis=1))
        if difference < cnrs_tolerance:
            cnrs_dicts.append(entry.output_dict)
        else:
            dicts.append(entry.output_dict)
    else:
        failed_dicts.append(entry.output_dict)

    if index % 10000 == 0:
        print(f'{100 * index / n_total: 2.2f}  {index} {len(dicts)} {len(failed_dicts)} {len(cnrs_dicts)}')

cnrs_rank = pd.DataFrame(cnrs_dicts)
cnrs_rank.to_parquet(f'cnrs_{rank:02d}.parquet')

entries_rank = pd.DataFrame(dicts)
entries_rank.to_parquet(f'cod_{rank:02d}.parquet')

failed_read = pd.DataFrame(failed_dicts)
failed_read.to_parquet(f'failed_read_cod_{rank:02d}.parquet')

if rank == 0:
    remove_duplicates('cod', n_ranks)
else:
    MPI.Finalize()
