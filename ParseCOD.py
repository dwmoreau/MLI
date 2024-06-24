from mpi4py import MPI
import os
import pandas as pd

from ParseDatabases import ProcessCODEntry


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
cod_dir = '/Users/DWMoreau/MLI/data/cod_cifs/'
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
in_numeric_tag = False
duplicate_base = ''

for index in range(rank, n_total, n_ranks):
    entry = ProcessCODEntry()
    entry.verify_entry(cif_file_names[index])
    entry.make_output_dict()

    if entry.status:
        dicts.append(entry.output_dict)
    else:
        failed_dicts.append(entry.output_dict)

    if index % 10000 == 0:
        print(f'{100 * index / n_total: 2.2f}  {index} {len(dicts)} {len(failed_dicts)}')

entries_rank = pd.DataFrame(dicts)
entries_rank.to_parquet(os.path.join('data', f'cod_{rank:02d}.parquet'))

failed_read = pd.DataFrame(failed_dicts)
failed_read.to_parquet(os.path.join('data', f'failed_read_cod_{rank:02d}.parquet'))
