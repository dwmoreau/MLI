import numpy as np
import os
import pandas as pd
import sys


if sys.argv[1] == 'remove-cod-in-csd':
    entries_a = pd.read_parquet(os.path.join('data', 'unique_entries_csd.parquet'))
    entries_b = pd.read_parquet(os.path.join('data', 'unique_cod_entries_cnrs_removed.parquet'))
    output_file_name = os.path.join('data', 'unique_cod_entries_not_in_csd.parquet')
elif sys.argv[1] == 'remove-cnrs-in-csd':
    entries_a = pd.read_parquet(os.path.join('data', 'cnrs_entries.parquet'))
    entries_b = pd.read_parquet(os.path.join('data', 'unique_entries_csd.parquet'))
    output_file_name = os.path.join('data', 'unique_csd_entries_cnrs_removed.parquet')
elif sys.argv[1] == 'remove-cnrs-in-cod':
    entries_a = pd.read_parquet(os.path.join('data', 'cnrs_entries.parquet'))
    entries_b = pd.read_parquet(os.path.join('data', 'unique_entries_cod.parquet'))
    output_file_name = os.path.join('data', 'unique_cod_entries_cnrs_removed.parquet')

all_unique_entries = []
groups_a = entries_a.groupby('bravais_lattice')
groups_b = entries_b.groupby('bravais_lattice')
keys = groups_b.groups.keys()
for key in keys:
    group_entries_a = groups_a.get_group(key)
    group_entries_b = groups_b.get_group(key)
    print(f'CSD Group {key} has {len(group_entries_a)} unique entries')
    print(f'COD Group {key} has {len(group_entries_b)} unique entries')

    unique_entries = []
    counts = 0
    compositions_group_a = group_entries_a.groupby('chemical_composition_string_strict')
    compositions_group_b = group_entries_b.groupby('chemical_composition_string_strict')
    for composition in compositions_group_b.groups.keys():
        # get all the entries in the cod with a given composition
        common_composition_b = compositions_group_b.get_group(composition)
        if composition in compositions_group_a.groups.keys():
            # If that composition also exists in the csd, then check for duplication
            common_composition_a = compositions_group_a.get_group(composition)
            # There could be multiple entries in the csd database with the same composition
            # Get all the volumes for that composition and verify that the cod entry
            # is not close in volume to any of the csd entries
            volume_checks = np.array(common_composition_a['reindexed_volume'])
            for entry_index in range(len(common_composition_b)):
                good = True
                entry_volume = common_composition_b.iloc[entry_index]['reindexed_volume']
                for volume_check in volume_checks:
                    check = np.isclose(volume_check, entry_volume, rtol=0.05)
                    if check:
                        good = False
                if good:
                    unique_entries.append(common_composition_b.iloc[entry_index])
                    counts += 1
        else:
            # Otherwise, add the cod entries because they should be unique
            unique_entries.append(common_composition_b)
            counts += len(common_composition_b)
    if len(unique_entries) > 0:
        unique_entries = pd.concat(unique_entries, ignore_index=True)

        # This is hack.
        # There is a bug in this code where the same COD entry is being added more than once.
        _, unique_indices = np.unique(unique_entries['reindexed_volume'], return_index=True)
        unique_entries = unique_entries.loc[unique_indices]
        all_unique_entries.append(unique_entries)
        print(f'  COD has {len(unique_entries)} unique entries not in CSD')

all_unique_entries = pd.concat(all_unique_entries, ignore_index=True)
all_unique_entries.drop(columns=[0], inplace=True)
all_unique_entries.to_parquet(output_file_name)
