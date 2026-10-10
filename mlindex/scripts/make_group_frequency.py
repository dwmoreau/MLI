"""Write the extinction-group frequency table the learned ranker reads.

For each Bravais lattice and each extinction group the indexer searches, the share of the
lattice's source structures whose space group belongs to that group. A group's members are the
space groups `SpaceGroups.EXTINCTION_GROUP_TABLE` lists under its EXPO code; a group whose
example setting the table does not list takes the members of the EXPO group that holds the
example's space-group type, or that type alone. Groups overlap, so a lattice's frequencies can sum
to more than one.

The fom-dev and fom-test structures of the split manifest are left out, so the table carries
nothing from the crystals the ranker is reported on.

    python -m mlindex.scripts.make_group_frequency \
        --split-manifest docs/fom_campaign2/artifacts/S06_split_manifest.parquet \
        --out group_frequency.csv

Needs the generated datasets (`dataset_<lattice>.parquet`, by default under
`mlindex/data/generated_datasets/`) and the model tree's reference reflections.
"""

import argparse
from pathlib import Path

import pandas as pd

from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

SEALED_SPLITS = ('fom-dev', 'fom-test')


def members_by_code():
    """{EXPO extinction-group code: the space-group numbers of its member space groups}."""
    from cctbx import sgtbx
    from mlindex.utilities.SpaceGroups import EXTINCTION_GROUP_TABLE

    members = {}
    for row in EXTINCTION_GROUP_TABLE:
        numbers = set()
        for symbol in row['Space Groups']:
            # A rhombohedral setting is written with its obverse/reverse suffix, which cctbx
            # does not read.
            for spelling in (symbol, symbol.replace('(obv)', '').replace('(rev)', '')):
                try:
                    numbers.add(int(sgtbx.space_group_info(spelling).type().number()))
                    break
                except RuntimeError:
                    continue
            else:
                print(f'cctbx does not read {symbol!r}; skipped')
        members[row['Code']] = numbers
    return members


def group_frequency_table(datasets, split_manifest):
    """One row per (Bravais lattice, extinction group key the indexer searches)."""
    from cctbx import sgtbx
    from mlindex.model_training.BenchmarkRuns import _hkl_reference
    from mlindex.utilities.SpaceGroups import get_spacegroup_keep_masks
    from mlindex.utilities.SpaceGroups import map_spacegroup_to_extinction_group

    manifest = pd.read_parquet(split_manifest, columns=['identifier', 'split'])
    sealed = set(manifest.loc[manifest['split'].isin(SEALED_SPLITS), 'identifier'])
    members = members_by_code()
    rows = []
    for lattice in BRAVAIS_LATTICES:
        source = pd.read_parquet(Path(datasets) / f'dataset_{lattice}.parquet',
                                 columns=['identifier', 'database', 'spacegroup_number'])
        kept = source.loc[~source['identifier'].isin(sealed)]
        numbers = kept['spacegroup_number'].astype(int)
        total = int(numbers.size)
        keys = get_spacegroup_keep_masks(
            _hkl_reference(lattice, BL_TO_LATTICE_SYSTEM[lattice]), lattice)
        for key in keys:
            example = key.split(' e.g. ')[1]
            group, code = map_spacegroup_to_extinction_group(example)
            found = members.get(code, set())
            if not found:
                number = int(sgtbx.space_group_info(example).type().number())
                holders = [c for c, held in members.items() if number in held]
                found = members[holders[0]] if holders else {number}
                code = holders[0] if holders else None
            count = int(numbers.isin(sorted(found)).sum())
            rows.append(dict(bravais_lattice=lattice, spacegroup=key, expo_group=group,
                             expo_code=code,
                             spacegroup_numbers=','.join(str(n) for n in sorted(found)),
                             structures=count, lattice_structures=total,
                             databases=','.join(sorted(kept['database'].unique())),
                             group_frequency=count/total if total else float('nan')))
        print(f'{lattice}: {total} structures ({source.shape[0] - total} sealed left out), '
              f'{len(keys)} groups')
    return pd.DataFrame(rows)


def build_parser():
    from mlindex.model_training.BenchmarkPatterns import DATASET_DIRECTORY

    parser = argparse.ArgumentParser(
        description='Write the extinction-group frequency table the learned ranker reads.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--split-manifest', required=True, metavar='PARQUET',
                        help='The split manifest; its fom-dev and fom-test structures are left '
                             'out.')
    parser.add_argument('--datasets', default=str(DATASET_DIRECTORY), metavar='DIR',
                        help='Directory holding dataset_<lattice>.parquet (default %(default)s).')
    parser.add_argument('--out', required=True, metavar='CSV', help='The table to write.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; the table is never overwritten')
    table = group_frequency_table(args.datasets, args.split_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
