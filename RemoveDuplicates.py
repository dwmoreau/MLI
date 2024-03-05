import numpy as np
import pandas as pd


source = 'cod'

if source == 'csd':
    print('Removing duplicates from CSD dataset')
    n_ranks = 8
    entries = pd.concat([
        pd.read_parquet(f'data/csd_{rank:02d}.parquet') for rank in range(n_ranks)
        ], ignore_index=True)
elif source == 'cod':
    print('Removing duplicates from COD dataset')
    entries = pd.read_parquet(f'data/cod_00.parquet')
else:
    print('Source must be either csd or cod')
    assert False

all_unique_entries = []
all_duplicated_entries = []
groups = entries.groupby('crystal_family')
for key in groups.groups.keys():
    unique_entries = []
    duplicated_entries = []
    group_entries = groups.get_group(key)
    compositions_group = group_entries.groupby('chemical_composition_string')
    n_entries = len(group_entries)
    print(f'Group {key} has {n_entries} entries')
    for composition in compositions_group.groups.keys():
        common_composition = compositions_group.get_group(composition)
        if len(common_composition) == 1:
            unique_entries.append(common_composition)
        else:
            while len(common_composition) > 0:
                volume = common_composition.iloc[0]['volume']
                close_volume = common_composition.loc[
                    np.isclose(common_composition['volume'], volume, rtol=0.05)
                    ]
                if len(close_volume) == 1:
                    unique_entries.append(close_volume)
                else:
                    close_volume = close_volume.sort_values(by='r_factor')
                    unique_entries.append(close_volume.iloc[[0]])
                    duplicated_entries.append(close_volume.iloc[1:])
                common_composition = common_composition.drop(close_volume.index)
    if len(unique_entries) > 0:
        unique_entries = pd.concat(unique_entries)
        all_unique_entries.append(unique_entries)
        print(f'  {len(unique_entries)} unique entries')
    if len(duplicated_entries) > 0:
        duplicated_entries = pd.concat(duplicated_entries)
        all_duplicated_entries.append(duplicated_entries)
        print(f'  {len(duplicated_entries)} duplicate entries')

all_unique_entries = pd.concat(all_unique_entries)
all_duplicated_entries = pd.concat(all_duplicated_entries)

if source == 'csd':
    all_unique_entries.to_parquet('data/unique_entries_csd.parquet')
    all_duplicated_entries.to_parquet('data/duplicate_entries_csd.parquet')
elif source == 'cod':
    all_unique_entries.to_parquet('data/unique_entries_cod.parquet')
    all_duplicated_entries.to_parquet('data/duplicate_entries_cod.parquet')
