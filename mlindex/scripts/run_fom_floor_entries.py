"""S08 -- draw the patterns the reproducibility floor is measured on, and record their composition.

The floor is the spread of a reported number over runs of the indexer that differ only in the
search seed. This chooses which patterns those runs cover, and writes the composition table the
report needs to turn a per-lattice floor into an aggregate one.

    python mlindex/scripts/run_fom_floor_entries.py \
        --split-manifest docs/fom_campaign2/artifacts/S06_split_manifest.parquet \
        --artifact-dir docs/fom_campaign2/artifacts

**Balanced across lattices, not proportional -- and that choice changes the arithmetic.**

`fom-dev` runs 600 entries for each of aP, mC and mP against 20 for cF and 30 for cI (C2-F-048,
C2-R-010). Drawn proportionally at any affordable size the rare lattices get one or two patterns
each and have no per-lattice floor at all -- and PROTOCOL section 8 requires a per-lattice claim to
be read against *that lattice's* floor, because the floor is ordered by free cell parameters over
two orders of magnitude and the lattices where this campaign's gains are expected are exactly the
least reproducible ones.

So the draw is balanced, and the aggregate is composed afterwards from the reporting split's own
per-lattice counts rather than from the sample's. Campaign 1 could take the shortcut -- its sample
was drawn proportional to `fom-dev`, so scaling `n_bl` by the sample's composition was valid -- and
S08's acceptance condition 4 says in terms that a sample drawn any other way needs the split's own
counts instead. `floor_weights` below is those counts, written beside the sample so the report
cannot silently fall back on the sample's shape.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


# The reporting split. Thresholds are selected on `fom-train` and reported on `fom-dev`
# (PROTOCOL section 8), and the floor is a property of what is reported, so it is measured there.
REPORTING_SPLIT = 'fom-dev'

# Three conditions, so "the floor barely moves with the condition" is re-measured rather than
# inherited. Campaign 1 found the metric floor moving by 1.4 pp while the operating point itself
# collapsed by a factor of 5.7 between conditions (F-150), which is the evidence that a floor
# expressed as a fraction of the baseline is wrong in shape as well as in size.
DEFAULT_CONDITIONS = ('nominal', 'noisy', 'control')

# Benchmark B's own search seed is the first arm, so only the others have to be generated.
DEFAULT_ARM_SEEDS = (202, 303, 404)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Draw the entries the S08 reproducibility floor is measured on')
    parser.add_argument('--split-manifest', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts',
                                             'S06_split_manifest.parquet'),
                        help='The frozen split. The entry list is read from it, never re-sampled')
    parser.add_argument('--artifact-dir', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--entries-per-lattice', type=int, default=40,
                        help='Patterns drawn per Bravais lattice, capped by availability. '
                             'Balanced, not proportional -- see the module docstring')
    parser.add_argument('--split', type=str, default=REPORTING_SPLIT)
    parser.add_argument('--conditions', type=str, default=','.join(DEFAULT_CONDITIONS))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', type=str, default='S08_floor')
    return parser.parse_args(argv)


def draw_sample(manifest, entries_per_lattice, split, seed):
    """`entries_per_lattice` from each Bravais lattice of `split`, deterministically.

    Returns the drawn rows. A lattice with fewer entries than the quota contributes all of them
    and is flagged in the composition table rather than silently short -- five lattices are
    hard-capped by the source population and cannot reach any quota (C2-R-010).
    """
    pool = manifest.loc[manifest['split'] == split]
    if pool.empty:
        raise SystemExit(f'The manifest holds no {split} entries; found '
                         f'{sorted(manifest["split"].unique())}')
    rng = np.random.default_rng(seed)
    drawn = []
    for lattice, group in pool.groupby('bravais_lattice', sort=True):
        group = group.sort_values('identifier')
        take = min(int(entries_per_lattice), group.shape[0])
        positions = np.sort(rng.choice(group.shape[0], size=take, replace=False))
        drawn.append(group.iloc[positions])
    return pd.concat(drawn, ignore_index=True).sort_values('identifier').reset_index(drop=True)


def composition(manifest, sample, split):
    """Per lattice: what the split holds, what the sample drew, and the weight of each.

    `split_entries` is what an aggregate must be composed with. `sample_share` and `split_share`
    are written beside it so the departure from proportionality is visible in the artefact rather
    than only in this docstring -- a later reader who assumes the sample is proportional would get
    an aggregate floor wrong by the ratio between the two columns.
    """
    pool = manifest.loc[manifest['split'] == split]
    split_counts = pool.groupby('bravais_lattice').size()
    sample_counts = sample.groupby('bravais_lattice').size()
    table = pd.DataFrame({
        'split_entries': split_counts,
        'sample_entries': sample_counts,
        }).fillna(0).astype(int).reset_index()
    table['capped_by_population'] = table['sample_entries'] < table['sample_entries'].max()
    table['split_share'] = table['split_entries']/table['split_entries'].sum()
    table['sample_share'] = table['sample_entries']/table['sample_entries'].sum()
    # The composition weight: an aggregate over lattices is sum_bl (n_bl/N) * mean_bl, so this is
    # the column the report multiplies a per-lattice floor by. It is `split_share` by definition;
    # it is named separately because the report must never reach for `sample_share` by accident.
    table['floor_weight'] = table['split_share']
    return table


def main(argv=None):
    args = _parse_args(argv)
    conditions = [name.strip() for name in args.conditions.split(',') if name.strip()]
    manifest = pd.read_parquet(args.split_manifest)
    sample = draw_sample(manifest, args.entries_per_lattice, args.split, args.seed)
    table = composition(manifest, sample, args.split)

    artifact_dir = args.artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    entries_path = os.path.join(artifact_dir, f'{args.tag}_entries.csv')
    sample[['identifier']].to_csv(entries_path, index=False)
    composition_path = os.path.join(artifact_dir, f'{args.tag}_composition.csv')
    table.to_csv(composition_path, index=False)

    n_cells = sample.shape[0]*len(conditions)
    plan = dict(
        split=args.split,
        seed=int(args.seed),
        entries_per_lattice=int(args.entries_per_lattice),
        n_entries=int(sample.shape[0]),
        conditions=conditions,
        # Benchmark B is arm 1: it was generated at a recorded search seed and a subset run at the
        # same seed reproduces its rows exactly (C2-F-058), so only these have to be generated.
        extra_arm_seeds=list(DEFAULT_ARM_SEEDS),
        n_cells_per_arm=int(n_cells),
        n_cells_to_generate=int(n_cells*len(DEFAULT_ARM_SEEDS)),
        sampling='balanced_across_lattices',
        aggregate_composed_from='split_entries',
        )
    plan_path = os.path.join(artifact_dir, f'{args.tag}_plan.json')
    with open(plan_path, 'w', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)

    print(table.to_string(index=False))
    print(f'\n{sample.shape[0]} entries x {len(conditions)} conditions = {n_cells} cells an arm')
    print(f'{plan["n_cells_to_generate"]} cells to generate over '
          f'{len(DEFAULT_ARM_SEEDS)} extra arms; Benchmark B is the first arm')
    print(f'wrote {entries_path}, {composition_path}, {plan_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
