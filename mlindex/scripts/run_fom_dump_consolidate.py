"""Consolidate the per-bundle dumps into Benchmark B, the frozen pool every later step reads.

`run_fom_dump.py` writes one directory per condition bundle, each holding a parquet per
(shard, pool). This joins them into the pool S08 onwards loads, partitioned by
(bundle, Bravais lattice) so a step can pull one lattice at a time rather than the whole grid:

    mlindex/data/fom_benchmark_c2/candidates_<bundle>_<BL>.parquet
    mlindex/data/fom_benchmark_c2/predownsample_<bundle>_<BL>.parquet
    mlindex/data/fom_benchmark_c2/entries.parquet
    mlindex/data/fom_benchmark_c2/manifest.json

Two things campaign 1's consolidator did that this one must not.

**It dropped `condition_bundle`** -- `labelled.drop(columns=['condition_bundle'])`, on the grounds
that "the bundle lives in the filename". That is rebuild row R8: after consolidation `entry_id`
alone is not a key, a join on it fans every candidate out once per bundle, and
`(entry_id, q2_digest)` is not a substitute because two sparse bundles leave entries with
identical peak lists. The column is written on both streams and on the entry table here.

**It aligned bundles by intersecting their entry sets.** Campaign 1 lost 33 entries to unplaceable
second-phase lines and then intersected, which is where its volume-decile drift entered: dropping
rows raises the within-lattice rank of the survivors (R14, F-108 -- and C2-F-050 shows the drift is
two-sided, so "it was conservative" is not available as a defence). This one keeps every bundle's
own entry set and **records the loss per bundle** instead. The frozen decile is joined, never
recomputed, so an unequal entry set costs coverage and cannot move a stratum.

Bundles are processed one at a time and never all held at once.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark


# Parquet row groups. The default of ~1M rows was fine for campaign 1's 26M-row pool; this one is
# an order of magnitude larger and much wider, and S09-S12 read narrow column subsets of it. A
# smaller group means a column subset touches less of the file and a predicate skips more of it,
# at the cost of more metadata. Set deliberately rather than inherited -- S07's pitfall list.
ROW_GROUP_SIZE = 131072


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Consolidate the campaign-2 bundle dumps into Benchmark B')
    parser.add_argument('--dump-root', type=str, required=True,
                        help='Directory holding one subdirectory per condition bundle')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Where the frozen pool is written')
    parser.add_argument('--artifact-dir', type=str, default=None,
                        help='Where the row-count table is written, if anywhere')
    parser.add_argument('--row-group-size', type=int, default=ROW_GROUP_SIZE)
    return parser.parse_args(argv)


def bundle_directories(dump_root):
    """One directory per bundle, identified by its own manifest.json.

    A directory without one is not a bundle -- it is an incomplete run, and treating it as a
    bundle would consolidate a partial pool while reporting a total that looks whole.
    """
    root = Path(dump_root)
    directories = sorted(child for child in root.iterdir()
                         if child.is_dir() and (child / 'manifest.json').exists())
    if not directories:
        raise SystemExit(f'no bundle directory under {root} carries a manifest.json')
    return directories


def _read_stream(bundle_dir, pattern):
    paths = sorted(Path(bundle_dir).glob(pattern))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _write(frame, path, row_group_size):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, row_group_size=row_group_size)


def consolidate_bundle(bundle_dir, out_dir, row_group_size):
    """One bundle's shards and pools, written out partitioned by Bravais lattice.

    Returns (entries, counts) -- the entry table and one row per (bundle, lattice) with the
    candidate and correct-candidate counts the gate reads.
    """
    entries = _read_stream(bundle_dir, 'entries_*.parquet')
    candidates = _read_stream(bundle_dir, 'candidates_*.parquet')
    if entries is None or candidates is None:
        raise SystemExit(f'{bundle_dir} is missing an entry or candidate stream; do not '
                         'consolidate a partial bundle')

    bundles = entries['condition_bundle'].unique()
    if len(bundles) != 1:
        # One manifest.json is written per output directory, so two bundles sharing a directory
        # have already overwritten each other's. Catch it here rather than downstream.
        raise SystemExit(f'{bundle_dir} holds more than one bundle: {sorted(bundles)}. '
                         'One directory per bundle.')
    bundle = str(bundles[0])

    # R8: the bundle is a COLUMN on every stream, not a fact about the filename. Both are written
    # by the driver; this asserts it survived rather than assuming it.
    for name, frame in (('candidates', candidates), ('entries', entries)):
        if 'condition_bundle' not in frame.columns:
            raise SystemExit(f'{bundle_dir}: {name} carries no condition_bundle column (R8)')

    FomBenchmark._check_join(candidates, entries)

    for bravais_lattice, group in candidates.groupby('bravais_lattice', sort=True):
        _write(group.reset_index(drop=True),
               Path(out_dir) / f'candidates_{bundle}_{bravais_lattice}.parquet', row_group_size)

    predownsample = _read_stream(bundle_dir, 'predownsample_*.parquet')
    if predownsample is not None and not predownsample.empty:
        for bravais_lattice, group in predownsample.groupby('bravais_lattice', sort=True):
            _write(group.reset_index(drop=True),
                   Path(out_dir) / f'predownsample_{bundle}_{bravais_lattice}.parquet',
                   row_group_size)

    truth = entries.set_index('entry_id')['bravais_lattice_true']
    counts = (candidates
              .assign(bravais_lattice_true=candidates['entry_id'].map(truth))
              .groupby(['bravais_lattice_true'], sort=True)
              .agg(n_candidates=('entry_id', 'size'),
                   n_correct=('is_correct', 'sum'),
                   n_off_by_two=('is_off_by_two', 'sum'),
                   n_entries=('entry_id', 'nunique'))
              .reset_index())
    counts.insert(0, 'condition_bundle', bundle)
    counts['n_reachable_entries'] = (
        candidates.loc[candidates['is_correct']]
        .assign(bravais_lattice_true=lambda f: f['entry_id'].map(truth))
        .groupby('bravais_lattice_true')['entry_id'].nunique()
        .reindex(counts['bravais_lattice_true']).fillna(0).astype(int).to_numpy())
    return entries, counts


def main(argv=None):
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries, all_counts, manifests = [], [], {}
    seen = {}
    for bundle_dir in bundle_directories(args.dump_root):
        entries, counts = consolidate_bundle(bundle_dir, out_dir, args.row_group_size)
        bundle = counts['condition_bundle'].iloc[0]
        # Two directories holding the SAME bundle overwrite each other's consolidated output, and
        # the run would report a bundle count lower than the directory count while looking
        # successful. This is the "one manifest.json per --out-dir" trap one level up: there it is
        # two bundles sharing a directory, here it is one bundle spread over two. Both are silent.
        if bundle in seen:
            raise SystemExit(
                f'{bundle_dir.name} and {seen[bundle].name} both hold bundle {bundle!r}, so the '
                'second has overwritten the first. Consolidate one directory per bundle; if '
                'these are deliberately separate runs, consolidate them to separate --out-dirs.')
        seen[bundle] = bundle_dir
        with open(bundle_dir / 'manifest.json', encoding='utf-8') as handle:
            manifests[bundle] = json.load(handle)
        all_entries.append(entries)
        all_counts.append(counts)
        print(f'{bundle_dir.name}: {entries.shape[0]} entries, '
              f'{int(counts["n_candidates"].sum())} candidates, '
              f'{int(counts["n_correct"].sum())} correct', flush=True)

    entries = pd.concat(all_entries, ignore_index=True)
    counts = pd.concat(all_counts, ignore_index=True)
    _write(entries, out_dir / 'entries.parquet', args.row_group_size)

    # Coverage, NOT alignment. Every bundle keeps its own entry set; what is recorded is which
    # entries each one is missing, so a later step can restrict deliberately rather than inherit
    # an intersection nobody chose.
    per_bundle = entries.groupby('condition_bundle')['entry_id'].apply(set)
    union = set().union(*per_bundle) if len(per_bundle) else set()
    coverage = pd.DataFrame([
        {'condition_bundle': bundle,
         'n_entries': len(ids),
         'n_missing_vs_union': len(union - ids),
         'missing_examples': ','.join(sorted(union - ids)[:5])}
        for bundle, ids in per_bundle.items()]).sort_values('condition_bundle')

    FomBenchmark.write_manifest(
        out_dir,
        consolidated=True,
        bundles=sorted(manifests),
        n_entries=int(entries.shape[0]),
        n_source_entries=len(union),
        n_candidates=int(counts['n_candidates'].sum()),
        n_correct=int(counts['n_correct'].sum()),
        row_group_size=int(args.row_group_size),
        bundle_manifests=manifests,
        )

    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(artifact_dir / 'S07_row_counts.csv', index=False)
        coverage.to_csv(artifact_dir / 'S07_bundle_coverage.csv', index=False)
        print(f'wrote {artifact_dir}/S07_row_counts.csv and S07_bundle_coverage.csv', flush=True)

    print(f'\nconsolidated {len(manifests)} bundles -> {out_dir}')
    print(coverage.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
