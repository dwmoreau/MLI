import numpy as np
import pandas as pd

entries_csd = pd.read_parquet(f'data/unique_entries_csd.parquet')
entries_cod = pd.read_parquet(f'data/unique_entries_cod.parquet')

all_unique_entries = []
groups_csd = entries_csd.groupby('crystal_family')
groups_cod = entries_cod.groupby('crystal_family')
keys = groups_cod.groups.keys()
for key in keys:
    unique_entries = []
    group_entries_csd = groups_csd.get_group(key)
    group_entries_cod = groups_cod.get_group(key)
    print(f'CSD Group {key} has {len(group_entries_csd)} unique entries')
    print(f'COD Group {key} has {len(group_entries_cod)} unique entries')
    compositions_group_csd = group_entries_csd.groupby('chemical_composition_string')
    compositions_group_cod = group_entries_cod.groupby('chemical_composition_string')
    for composition in compositions_group_cod.groups.keys():
        common_composition_cod = compositions_group_cod.get_group(composition)
        if composition in compositions_group_csd.groups.keys():
            common_composition_csd = compositions_group_csd.get_group(composition)
            while len(common_composition_cod) > 0:
                close_volume = common_composition_csd.loc[
                    np.isclose(
                        common_composition_csd['volume'],
                        common_composition_cod.iloc[0]['volume'],
                        rtol=0.05
                        )
                    ]
                if len(close_volume) == 0:
                    unique_entries.append(common_composition_cod.iloc[0])
                common_composition_cod = common_composition_cod.iloc[1:]
        else:
            unique_entries.append(common_composition_cod)
    if len(unique_entries) > 0:
        unique_entries = pd.concat(unique_entries, ignore_index=True)
        all_unique_entries.append(unique_entries)
        print(f'  COD has {len(unique_entries)} unique entries not in CSD')
all_unique_entries = pd.concat(all_unique_entries, ignore_index=True)
all_unique_entries.drop(columns=[0], inplace=True)
all_unique_entries.to_parquet('data/unique_cod_entries_not_in_csd.parquet')
