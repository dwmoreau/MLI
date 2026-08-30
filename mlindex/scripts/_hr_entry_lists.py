"""Write the two hR entry lists the supplementary run needs, one per arm.

A helper rather than a heredoc inside `submit_fom_dump_supplement.sh`, so it can be tested.

The distinction is the whole point. `run_fom_dump.py --arm mechanism` restricts the manifest to
the nested subset BEFORE `--entry-ids-file` is applied, and only 210 of the 1 400 hR entries are
in that subset -- so a mechanism task given the full list is refused with "1190 of 1400 requested
entries were not found". That is the driver's guard working; the caller was wrong.
"""

import sys

import pandas as pd


def write_lists(manifest_path, core_path, mechanism_path, bravais_lattice='hR'):
    manifest = pd.read_parquet(manifest_path)
    identifier = 'identifier' if 'identifier' in manifest.columns else 'entry_id'
    selected = manifest.loc[manifest['bravais_lattice'] == bravais_lattice]
    mechanism = selected[selected['arm'].astype(str).str.contains('mechanism')]

    pd.DataFrame({'identifier': sorted(selected[identifier])}).to_csv(core_path, index=False)
    pd.DataFrame({'identifier': sorted(mechanism[identifier])}).to_csv(mechanism_path, index=False)
    return len(selected), len(mechanism)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        raise SystemExit('usage: _hr_entry_lists.py <manifest.parquet> <core.csv> <mechanism.csv>')
    n_core, n_mechanism = write_lists(*argv)
    if n_core == 0:
        raise SystemExit(f'{argv[0]} holds no entries for that lattice; nothing to regenerate')
    print(f'hR entries: {n_core} core, {n_mechanism} also in the mechanism arm')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
