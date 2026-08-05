"""Check that the run_training_*.py scripts will actually retrain what you think they will.

A training run costs hours per split group, and the ways it can silently do nothing are cheap to
miss by eye: a group still pointing at a load_from_tag dict trains nothing, a group named in the
script but absent from the data fails partway through, and a script that still sets a parameter by
hand quietly overrides a default that has since been tuned. Variable names are no guide either --
some scripts have a dict named '..._load' that sets load_from_tag False, and others the reverse --
so each group is resolved through to the value it actually gets.

    python mlindex/scripts/check_training_scripts.py [--systems cubic,monoclinic]
"""
import argparse
import os
import re

import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYSTEMS = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'rhombohedral', 'tetragonal',
           'triclinic']
# Settings that now have tuned defaults. A script naming one is overriding that default, which may
# be deliberate but is worth seeing.
TUNED_DEFAULTS = ['branch_loss_weight', 'volume_lower_percentile', 'volume_upper_percentile',
                  'volume_spacing_blend', 'early_stopping_patience']


def resolve(source, variable):
    """load_from_tag of the dict a group maps to, resolved by value rather than by name."""
    block = re.search(rf'^    {variable} = \{{(.*?)^        \}}', source, re.S | re.M)
    if block is None:
        return None
    found = re.search(r"'load_from_tag':\s*(True|False)", block.group(1))
    return None if found is None else found.group(1) == 'False'


def check(system):
    path = os.path.join(BASE, 'mlindex', 'scripts', f'run_training_{system}.py')
    source = open(path).read()
    problems = []

    block = re.search(r'^    integral_filter_params = \{(.*?)^        \}', source, re.S | re.M)
    if block is None:
        return [f'no integral_filter_params mapping found in {path}'], 0
    mapping = re.findall(r"'([A-Za-z]{2}_[0-9_]+)':\s*(\w+)", block.group(1))
    will_train, will_load, unresolved = [], [], []
    for group, variable in mapping:
        trains = resolve(source, variable)
        (will_train if trains is True else will_load if trains is False else unresolved
         ).append(group)
    if will_load:
        problems.append(f'set to load, so will not retrain: {sorted(will_load)}')
    if unresolved:
        problems.append(f'could not resolve load_from_tag for: {sorted(unresolved)}')

    parquet = os.path.join(BASE, 'mlindex', 'models', f'{system}_1', 'data', 'data.parquet')
    if not os.path.exists(parquet):
        problems.append(f'no data at {parquet}')
    else:
        present = set(pc.unique(
            pq.read_table(parquet, columns=['split_group'])['split_group']).to_pylist())
        named = {group for group, _ in mapping}
        if named - present:
            problems.append(f'named in script but absent from the data: {sorted(named - present)}')
        if present - named:
            problems.append(f'present in the data but not named in the script: '
                            f'{sorted(present - named)}')

    if re.search(r'^\s*wrapper\.data_params\[.split_groups.\]\s*=', source, re.M):
        problems.append('split_groups is overridden, which pins the run to a subset')
    overrides = [key for key in TUNED_DEFAULTS if f"'{key}'" in source]
    if overrides:
        problems.append(f'overrides tuned defaults: {overrides}')

    calls = re.findall(r'^\s*wrapper\.(\w+)\(', source, re.M)
    unexpected = [c for c in set(calls)
                  if c not in {'load_data_from_tag', 'load_data', 'setup_integral_filter'}]
    if unexpected:
        problems.append(f'also runs, and would overwrite, other components: {sorted(unexpected)}')

    return problems, len(will_train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', default=','.join(SYSTEMS))
    args = parser.parse_args()
    ready = True
    total = 0
    for system in args.systems.split(','):
        problems, n_train = check(system)
        total += 0 if problems else n_train
        print(('READY     ' if not problems else 'NOT READY ')
              + f'{system:14s} groups that would retrain: {n_train}')
        for problem in problems:
            print(f'{"":24s}{problem}')
        ready &= not problems
    print(f'\n{"all scripts ready" if ready else "not ready -- see above"}; '
          f'{total} split groups would retrain')
    raise SystemExit(0 if ready else 1)


if __name__ == '__main__':
    main()
