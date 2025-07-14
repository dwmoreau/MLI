from ccdc.io import EntryReader
from mpi4py import MPI
import os
import pandas as pd

from ParseDatabases import ProcessCSDEntry
from RemoveDuplicates import remove_duplicates


COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
n_ranks = COMM.Get_size()
# MPI approach
#   Each rank reads different entries
#   Output data frames are saved to different .json files
csd_entry_reader = EntryReader('CSD')
n_total = len(csd_entry_reader)
dicts = []
failed_dicts = []
output_dicts = []
in_numeric_tag = False
duplicate_base = ''

for index in range(rank, n_total, n_ranks):
    csd_entry = csd_entry_reader[index]
    entry = ProcessCSDEntry()
    entry.verify_entry(csd_entry)
    entry.make_output_dict()

    if entry.status:
        dicts.append(entry.output_dict)
    else:
        failed_dicts.append(entry.output_dict)

    if index % 10000 == 0:
        print(f'{100 * index / n_total: 2.2f}  {index} {len(dicts)}')

entries_rank = pd.DataFrame(dicts)
entries_rank.to_parquet(os.path.join('data', f'csd_{rank:02d}.parquet'))

failed_read = pd.DataFrame(failed_dicts)
failed_read.to_parquet(os.path.join('data', f'failed_read_csd_{rank:02d}.parquet'))

if rank == 0:
    remove_duplicates('csd', n_ranks)
else:
    MPI.Finalize()
