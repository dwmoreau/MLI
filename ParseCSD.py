from ccdc.io import EntryReader
from mpi4py import MPI
import os
import pandas as pd

from EntryHelpers import load_identifiers
from ParseDatabases import ProcessCSDEntry

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
if rank == 0:
    # The entries in these 'bad_identifiers' lists failed to generate a
    # data set during GenerateDataset.py.
    bad_identifiers = []
    for file_index in range(3):
        bad_identifiers += load_identifiers(
            os.path.join('data', f'bad_identifiers_{file_index}.txt')
        )
else:
    bad_identifiers = None
bad_identifiers = COMM.bcast(bad_identifiers, root=0)

for index in range(rank, n_total, n_ranks):
    csd_entry = csd_entry_reader[index]
    entry = ProcessCSDEntry()
    entry.verify_entry(csd_entry, bad_identifiers=bad_identifiers)
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