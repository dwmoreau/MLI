import numpy as np
import pandas as pd


entries_csd = pd.read_parquet(f'data/unique_entries_csd.parquet')
entries_cod = pd.read_parquet(f'data/unique_entries_cod.parquet')

all_unique_entries = []
groups_csd = entries_csd.groupby('bravais_lattice')
groups_cod = entries_cod.groupby('bravais_lattice')
keys = groups_cod.groups.keys()
for key in keys:
    group_entries_csd = groups_csd.get_group(key)
    group_entries_cod = groups_cod.get_group(key)
    print(f'CSD Group {key} has {len(group_entries_csd)} unique entries')
    print(f'COD Group {key} has {len(group_entries_cod)} unique entries')

    unique_entries = []
    counts = 0
    compositions_group_csd = group_entries_csd.groupby('chemical_composition_string_strict')
    compositions_group_cod = group_entries_cod.groupby('chemical_composition_string_strict')
    for composition in compositions_group_cod.groups.keys():
        # get all the entries in the cod with a given composition
        common_composition_cod = compositions_group_cod.get_group(composition)
        if composition in compositions_group_csd.groups.keys():
            # If that composition also exists in the csd, then check for duplication
            common_composition_csd = compositions_group_csd.get_group(composition)
            # There could be multiple entries in the csd database with the same composition
            # Get all the volumes for that composition and verify that the cod entry
            # is not close in volume to any of the csd entries
            volume_checks = np.array(common_composition_csd['volume'])
            for entry_index in range(len(common_composition_cod)):
                good = True
                entry_volume = common_composition_cod.iloc[entry_index]['volume']
                for volume_check in volume_checks:
                    check = np.isclose(volume_check, entry_volume, rtol=0.05)
                    if check:
                        good = False
                if good:
                    unique_entries.append(common_composition_cod.iloc[entry_index])
                    counts += 1
        else:
            # Otherwise, add the cod entries because they should be unique
            unique_entries.append(common_composition_cod)
            counts += len(common_composition_cod)
    if len(unique_entries) > 0:
        unique_entries = pd.concat(unique_entries, ignore_index=True)

        # This is hack.
        # There is a bug in this code where the same COD entry is being added more than once.
        _, unique_indices = np.unique(unique_entries['volume'], return_index=True)
        unique_entries = unique_entries.loc[unique_indices]
        all_unique_entries.append(unique_entries)
        print(f'  COD has {len(unique_entries)} unique entries not in CSD')

all_unique_entries = pd.concat(all_unique_entries, ignore_index=True)
all_unique_entries.drop(columns=[0], inplace=True)
all_unique_entries.to_parquet('data/unique_cod_entries_not_in_csd.parquet')
