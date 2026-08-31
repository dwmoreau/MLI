"""Cut a small, balanced, still-valid slice out of Benchmark B, for development off NERSC.

Benchmark B is ~122 GB on Perlmutter scratch and the laptop has none of it, so the metrics module
cannot be developed or gated against real data without a slice. This writes one: same file layout,
same columns, a manifest that describes itself as a subset, and a stratified entry selection that
keeps the things a metrics gate actually needs to exercise.

    RUNBOOK -- run this ON NERSC, then copy the output directory back.

    python mlindex/scripts/run_fom_pool_subset.py \
        --pool /pscratch/sd/d/dwmoreau/fom_campaign2/pool \
        --out-dir $SCRATCH/fom_campaign2/pool_subset \
        --entries-per-lattice 20

    tar -czf pool_subset.tar.gz -C $SCRATCH/fom_campaign2 pool_subset

`docs/` is git-ignored and never reaches NERSC, so the runbook lives here rather than in a handoff.

Three properties this has to have, each of which is a way the slice could be quietly useless:

* **`fom-test` never leaves the pool.** PROTOCOL section 3 rule 2 seals it until S15. It is
  refused here rather than filtered, so a slice cannot be built that contains it by accident.
* **Balanced across Bravais lattices, not proportional.** The reporting split runs 600 entries for
  each of aP, mC and mP against 20 for cF and 30 for cI (C2-F-048, C2-R-010), so a proportional
  slice gives the rare lattices one entry each and no per-lattice number can be exercised at all.
* **The hard stratum is selected for**, not left to chance. It is mP/mC/aP at volume decile >= 8
  under the harder conditions, and it is the stratum every headline claim is carried by; a slice
  that happens to miss it passes a gate that means nothing.

Memory: one row group at a time, never one frame. The whole pool is 880.7 M rows and the S07
acceptance gate OOM-killed a node by loading it into a single pandas frame (C2-F-074); a subset
tool that scans the same files is the obvious place for that to happen twice.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mlindex.model_training import FomBenchmark


# Matches the consolidator, so a slice reads back through exactly the same loader as the pool.
ROW_GROUP_SIZE = 131072

# The splits a slice may contain. `fom-test` is sealed until S15 and is not one of them.
ALLOWED_SPLITS = ('fom-train', 'fom-dev')

SEALED_SPLIT = 'fom-test'

# The hard stratum's lattices and volume cut, as METRICS.md section 5 defines them. Imported
# rather than restated would be better, but FomMetrics imports pandas-heavy machinery this script
# does not need on a login node; `test_pool_subset_hard_stratum_matches_metrics` pins them equal.
HARD_LATTICES = ('mP', 'mC', 'aP')
HARD_MIN_DECILE = 8


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Write a small balanced subset of Benchmark B for off-cluster development')
    parser.add_argument('--pool', type=str, required=True,
                        help='The consolidated Benchmark B directory, holding entries.parquet, '
                             'manifest.json and the candidate parquets')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Where the subset pool is written')
    parser.add_argument('--entries-per-lattice', type=int, default=20,
                        help='Source entries kept per Bravais lattice, capped by availability. '
                             'The slice is roughly 1.1 MB per (entry x condition) cell, so 20 '
                             'over 14 lattices and 9 bundles is about 2 GB')
    parser.add_argument('--hard-fraction', type=float, default=0.5,
                        help='Share of a hard lattice quota reserved for entries at or above the '
                             'hard volume decile. Ignored for the other lattices')
    parser.add_argument('--bundles', type=str, default=None,
                        help='Comma-separated condition bundle tags. Default is every bundle in '
                             'the pool')
    parser.add_argument('--splits', type=str, default=','.join(ALLOWED_SPLITS),
                        help='Comma-separated splits to draw from. fom-test is refused')
    parser.add_argument('--with-predownsample', action='store_true',
                        help='Also slice the pre-deduplication stream. It is ~7.7x the survivor '
                             'stream and is a stratified subsample of entries, so most selected '
                             'entries will not appear in it')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Selection seed. The draw is deterministic given the pool and this')
    parser.add_argument('--row-group-size', type=int, default=ROW_GROUP_SIZE)
    return parser.parse_args(argv)


def select_entries(entries, entries_per_lattice, hard_fraction, splits, seed):
    """Choose the source entries, balanced across lattices and loaded toward the hard stratum.

    `entries` is the pool's entry table -- one row per (entry, condition bundle) -- so it is
    reduced to one row per source entry first. The split is by source entry and never by
    candidate (PROTOCOL section 3 rule 5), and an entry appears under every bundle it was
    generated for, so drawing on the (entry, bundle) rows would weight an entry by how many
    bundles happened to cover it.

    Within a lattice the draw is: fill `hard_fraction` of the quota from entries at or above the
    hard volume decile, then fill the rest from everything else, then top up from whichever pool
    still has entries. That keeps the hard stratum non-empty without making the slice a
    hard-stratum-only pool, which would leave the general population untested.
    """
    if SEALED_SPLIT in splits:
        raise SystemExit(
            f'{SEALED_SPLIT} is sealed until S15 (PROTOCOL section 3 rule 2) and must not be '
            f'copied off the cluster. Draw from {ALLOWED_SPLITS}.')
    unknown = [name for name in splits if name not in ALLOWED_SPLITS]
    if unknown:
        raise SystemExit(f'Unknown split(s) {unknown}; expected some of {ALLOWED_SPLITS}')
    if 'volume_decile' not in entries.columns:
        # Benchmark A has no such column -- that is R14, the defect schema v3 exists to fix -- so
        # this is also the check that the pool being sliced is a campaign-2 one.
        raise SystemExit(
            'The entry table has no volume_decile column, so the hard stratum cannot be selected '
            'for. Schema v3 stores it, read from the frozen split manifest; a pool without it is '
            "campaign 1's Benchmark A, which this script does not slice.")

    per_entry = (entries.loc[entries['split'].isin(splits)]
                 .drop_duplicates(subset=['entry_id'])
                 .sort_values('entry_id')
                 .reset_index(drop=True))
    if per_entry.empty:
        raise SystemExit(f'No entries in splits {splits}; the pool may already be filtered')

    rng = np.random.default_rng(seed)
    chosen = []
    for lattice, group in per_entry.groupby('bravais_lattice_true', sort=True):
        quota = min(int(entries_per_lattice), group.shape[0])
        if lattice in HARD_LATTICES:
            hard = group.loc[group['volume_decile'] >= HARD_MIN_DECILE]
            rest = group.loc[group['volume_decile'] < HARD_MIN_DECILE]
            n_hard = min(int(round(quota*float(hard_fraction))), hard.shape[0])
        else:
            hard = group.iloc[:0]
            rest = group
            n_hard = 0
        picked = pd.concat([
            _draw(hard, n_hard, rng),
            _draw(rest, quota - n_hard, rng),
            ])
        # Top up from whatever is left if either half ran short of its share.
        if picked.shape[0] < quota:
            remaining = group.loc[~group['entry_id'].isin(picked['entry_id'])]
            picked = pd.concat([picked, _draw(remaining, quota - picked.shape[0], rng)])
        chosen.append(picked)
    return pd.concat(chosen, ignore_index=True).sort_values('entry_id').reset_index(drop=True)


def _draw(frame, n, rng):
    """`n` rows without replacement, deterministically, tolerating an over-large `n`."""
    n = max(0, min(int(n), frame.shape[0]))
    if not n:
        return frame.iloc[:0]
    positions = rng.choice(frame.shape[0], size=n, replace=False)
    return frame.iloc[np.sort(positions)]


def filter_parquet(path, out_path, entry_ids, row_group_size):
    """Copy the rows of one parquet whose `entry_id` is in `entry_ids`, a row group at a time.

    Returns (rows_in, rows_out). Nothing is written when no row survives: an empty parquet for
    every (bundle, lattice) pair the slice does not reach would make the subset look complete
    while carrying no candidates, and `available_bundles` counts files.
    """
    source = pq.ParquetFile(path)
    keep = pa.array(sorted(entry_ids), type=pa.string())
    writer = None
    rows_in = rows_out = 0
    try:
        for index in range(source.num_row_groups):
            table = source.read_row_group(index)
            rows_in += table.num_rows
            mask = pc.is_in(table.column('entry_id'), value_set=keep)
            table = table.filter(mask)
            if not table.num_rows:
                continue
            if writer is None:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table, row_group_size=row_group_size)
            rows_out += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows_in, rows_out


def _stream_names(pool, prefix, bundles):
    """The candidate or pre-deduplication parquets, restricted to the wanted bundles.

    Both layouts are accepted: a consolidated pool writes `<prefix>_<bundle>_<BL>.parquet` and a
    raw generation run writes `<prefix>_<bundle>_shard<NN>ofNN_pool<NN>.parquet`. The bundle is
    recovered with the loader's own rule rather than by splitting the name here, because that rule
    has bitten this project before -- a bundle tag contains underscores.
    """
    paths = []
    for path in sorted(Path(pool).glob(f'{prefix}_*.parquet')):
        bundle = FomBenchmark.bundle_from_candidate_path(path)
        if bundles is None or bundle in bundles:
            paths.append((bundle, path))
    return paths


def main(argv=None):
    args = _parse_args(argv)
    pool = Path(args.pool)
    out_dir = Path(args.out_dir)
    splits = tuple(name.strip() for name in args.splits.split(',') if name.strip())
    bundles = (None if args.bundles is None
               else {name.strip() for name in args.bundles.split(',') if name.strip()})

    entries = FomBenchmark.load_entries(pool)
    selected = select_entries(entries, args.entries_per_lattice, args.hard_fraction, splits,
                              args.seed)
    entry_ids = set(selected['entry_id'].astype(str))
    print(f'selected {len(entry_ids)} source entries over '
          f'{selected["bravais_lattice_true"].nunique()} lattices')

    out_dir.mkdir(parents=True, exist_ok=True)
    kept_entries = entries.loc[entries['entry_id'].isin(entry_ids)].reset_index(drop=True)
    if (kept_entries['split'] == SEALED_SPLIT).any():
        raise SystemExit(f'Refusing to write: {SEALED_SPLIT} rows reached the entry table.')
    kept_entries.to_parquet(out_dir / 'entries.parquet', index=False,
                            row_group_size=args.row_group_size)

    counts = []
    prefixes = ['candidates'] + (['predownsample'] if args.with_predownsample else [])
    for prefix in prefixes:
        for bundle, path in _stream_names(pool, prefix, bundles):
            rows_in, rows_out = filter_parquet(path, out_dir / path.name, entry_ids,
                                               args.row_group_size)
            counts.append(dict(stream=prefix, bundle=bundle, file=path.name,
                               rows_in=rows_in, rows_out=rows_out))
            print(f'  {path.name}: {rows_out} of {rows_in}')

    manifest = FomBenchmark.load_manifest(pool) or {}
    # The slice describes itself as one. A manifest copied verbatim would claim the pool's own
    # entry and candidate counts, and every gate that reads it would compare against the full
    # pool's numbers and fail for the wrong reason.
    manifest.update(
        subset_of=str(pool),
        subset_seed=int(args.seed),
        subset_entries_per_lattice=int(args.entries_per_lattice),
        subset_hard_fraction=float(args.hard_fraction),
        subset_splits=list(splits),
        n_entries=int(kept_entries.shape[0]),
        n_source_entries=len(entry_ids),
        n_candidates=int(sum(row['rows_out'] for row in counts
                             if row['stream'] == 'candidates')),
        )
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    table = pd.DataFrame(counts)
    table.to_csv(out_dir / 'subset_row_counts.csv', index=False)
    composition = (selected.groupby('bravais_lattice_true')
                   .agg(n_entries=('entry_id', 'size'),
                        n_hard=('volume_decile', lambda column: int(
                            (column >= HARD_MIN_DECILE).sum())))
                   .reset_index())
    composition.to_csv(out_dir / 'subset_composition.csv', index=False)
    print(f'\n{composition.to_string(index=False)}')
    print(f'\nwrote {manifest["n_candidates"]} candidates over {kept_entries.shape[0]} '
          f'(entry x bundle) cells to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
