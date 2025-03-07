import numpy as np
import pandas as pd


def remove_duplicates(source, n_ranks):
    print(f'Removing duplicates from {source} dataset')
    entries = pd.concat([
        pd.read_parquet(os.path.join(
            'data',
            f'{source}_{rank:02d}.parquet'
            )) for rank in range(n_ranks)
        ], ignore_index=True)

    all_unique_entries = []
    all_duplicated_entries = []
    bl_groups = entries.groupby('bravais_lattice')
    for key in bl_groups.groups.keys():
        duplicated_entries = []
        bl_group_entries = bl_groups.get_group(key)
        print(f'Group {key} has {len(bl_group_entries)} entries')

        # chemical_composition_string: All elements and counts in the unit cell
        # chemical_composition_string_string: remove hydrogrens & deuteriums and elements 
        # that make up less than 5% of the total atoms.
        unique_entries_composition = []
        compositions_group = bl_group_entries.groupby('chemical_composition_string_strict')
        for composition in compositions_group.groups.keys():
            common_composition = compositions_group.get_group(composition)
            if len(common_composition) == 1:
                unique_entries_composition.append(common_composition)
            else:
                while len(common_composition) > 0:
                    # Get the first entries volume
                    # reindexed volume is the same as volume for all entries except rhombohedral
                    # In this case, reindexed_volume is different for the entries in the
                    # hexagonal setting.
                    volume = common_composition.iloc[0]['reindexed_volume']
                    # Get all entries with the volume within 5%
                    close_volume = common_composition.loc[
                        np.isclose(common_composition['reindexed_volume'], volume, rtol=0.05)
                        ]
                    if len(close_volume) == 1:
                        # If there is only one entry with this volume, add it to unit_entries_compostion
                        unique_entries_composition.append(close_volume)
                    else:
                        # Otherwise, get the entry with the lowest r-factor and get rid of the rest.
                        close_volume = close_volume.sort_values(by='r_factor')
                        unique_entries_composition.append(close_volume.iloc[[0]])
                        duplicated_entries.append(close_volume.iloc[1:])
                    # Remove entries that have been verified to not be duplicates
                    common_composition = common_composition.drop(close_volume.index)
        print(f'  {len(unique_entries_composition)} unique entries after common composition removal')

        # This removes entries with exactly the same unit cell volume
        # I am trying to find entries with the same unit cell, so a,b,c, alpha, beta, gamma are 
        # exactly the same. Unit cell is a numpy array and unhashable, so it cannot be used in groupby
        if len(unique_entries_composition) > 0:
            unique_entries_composition = pd.concat(unique_entries_composition)
            unique_entries_unit_cell = []
            unit_cell_group = unique_entries_composition.groupby('reindexed_volume')
            for unit_cell in unit_cell_group.groups.keys():
                common_unit_cell = unit_cell_group.get_group(unit_cell)
                if len(common_unit_cell) == 1:
                    unique_entries_unit_cell.append(common_unit_cell)
                else:
                    common_unit_cell = common_unit_cell.sort_values(by='r_factor')
                    unique_entries_unit_cell.append(common_unit_cell.iloc[[0]])
                    duplicated_entries.append(common_unit_cell.iloc[1:])

            if len(unique_entries_unit_cell) > 0:
                unique_entries_unit_cell = pd.concat(unique_entries_unit_cell)
                all_unique_entries.append(unique_entries_unit_cell)
                print(f'  {len(unique_entries_unit_cell)} unique entries after common unit cell removal')

        if len(duplicated_entries) > 0:
            duplicated_entries = pd.concat(duplicated_entries)
            all_duplicated_entries.append(duplicated_entries)
            print(f'  {len(duplicated_entries)} duplicate entries')

    all_unique_entries = pd.concat(all_unique_entries)
    all_duplicated_entries = pd.concat(all_duplicated_entries)

    all_unique_entries.to_parquet(os.path.join('data', f'unique_entries_{source}.parquet'))
    all_duplicated_entries.to_parquet(os.path.join('data', f'duplicate_entries_{source}.parquet'))

if __name__ == '__main__':
    remove_duplicates(source='csd', n_ranks=8)
