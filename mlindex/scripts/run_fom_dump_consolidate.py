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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomConditions


# Parquet row groups. The default of ~1M rows was fine for campaign 1's 26M-row pool; this one is
# an order of magnitude larger and much wider, and S09-S12 read narrow column subsets of it. A
# smaller group means a column subset touches less of the file and a predicate skips more of it,
# at the cost of more metadata. Set deliberately rather than inherited -- S07's pitfall list.
ROW_GROUP_SIZE = 131072


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Consolidate the campaign-2 bundle dumps into Benchmark B')
    parser.add_argument('--dump-root', type=str, required=True, nargs='+',
                        help='One or more directories, each holding one subdirectory per '
                             'condition bundle. Several are merged per bundle, which is how a '
                             'supplementary run -- regenerating a lattice the first pass lost -- '
                             'is folded in without regenerating the rest')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Where the frozen pool is written')
    parser.add_argument('--artifact-dir', type=str, default=None,
                        help='Where the row-count table is written, if anywhere')
    parser.add_argument('--row-group-size', type=int, default=ROW_GROUP_SIZE)
    parser.add_argument('--processes', type=int, default=1,
                        help='Bundles consolidated concurrently. They are independent, so '
                             'this is the axis to parallelise on -- 9 is the maximum that '
                             'does anything. Each process holds one shard table at a time, '
                             'not a whole bundle. Do not run a large value on a login '
                             'node; use a compute node or an salloc')
    return parser.parse_args(argv)


def bundle_directories(dump_roots):
    """{bundle tag: [directories]}, across one or more dump roots.

    A bundle can legitimately be spread over two roots: the main array writes one, and a
    supplementary run -- regenerating a lattice the first pass lost, say -- writes another. They
    merge here because the output is one file per (bundle, lattice) regardless of which shard the
    rows came from, so the writers simply append.

    A directory without a manifest.json is not a bundle; it is an incomplete run, and treating it
    as one would consolidate a partial pool while reporting a total that looks whole.
    """
    groups = {}
    for root in dump_roots:
        root = Path(root)
        if not root.is_dir():
            raise SystemExit(f'{root} is not a directory')
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / 'manifest.json').exists():
                groups.setdefault(child.name, []).append(child)
    if not groups:
        raise SystemExit(f'no bundle directory under {list(dump_roots)} carries a manifest.json')
    return dict(sorted(groups.items()))


def _stream_paths(bundle_dirs, pattern):
    """Every shard of one stream, across the directories holding this bundle.

    REFUSES A REPEATED BASENAME. Shard files are named `<stream>_<tag>_shard<NN>of<NN>_pool<NN>`,
    so the same basename in two roots means the same (shard, pool) was generated twice and its
    entries would be counted and written twice. Guarding on the file name is exact, where the older
    guard -- one bundle may not appear in two directories -- was both too strict, forbidding a
    legitimate supplementary run, and too loose, saying nothing about what overlapped.
    """
    paths, seen = [], {}
    for bundle_dir in bundle_dirs:
        for path in sorted(Path(bundle_dir).glob(pattern)):
            if path.name in seen:
                raise SystemExit(
                    f'{path.name} appears in both {seen[path.name].parent} and {path.parent}. '
                    'The same (shard, pool) was generated twice; consolidating both would double '
                    'its rows. Give a supplementary run a shard count the first pass did not use.')
            seen[path.name] = path
            paths.append(path)
    return paths


def _write(frame, path, row_group_size):
    """The consolidated entry table. One row per (entry, bundle), so it stays small."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, row_group_size=row_group_size)


def _read_small(paths):
    """Concatenate the entry tables. They are small -- ~74 rows a file -- so pandas is fine."""
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


class _LatticeWriters:
    """One open ParquetWriter per Bravais lattice, written to incrementally.

    The point of writing this way rather than concatenating: a core bundle is ~25 GB of candidates
    across 256 shard files, and the old implementation built it as ONE pandas frame before writing
    it back out. Four of the 34 columns are list-valued (`xnn`, `unit_cell`, `merit_at_prune`,
    `hkl_true_in_basis`), and those are what make the Arrow -> pandas -> Arrow round trip
    expensive: each becomes an object column of numpy arrays, one Python object per row per column.
    Staying in Arrow skips both conversions, and streaming file by file bounds memory at one shard.
    """

    def __init__(self, out_dir, stream, bundle, row_group_size):
        self.out_dir, self.stream = Path(out_dir), stream
        self.bundle, self.row_group_size = bundle, row_group_size
        self._writers = {}

    def write(self, table, bravais_lattice):
        writer = self._writers.get(bravais_lattice)
        if writer is None:
            path = self.out_dir / f'{self.stream}_{self.bundle}_{bravais_lattice}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            # The first shard's schema is the file's schema. A later shard that disagrees raises
            # here rather than producing a file whose columns silently shift.
            writer = pq.ParquetWriter(path, table.schema)
            self._writers[bravais_lattice] = writer
        writer.write_table(table, row_group_size=self.row_group_size)

    def close(self):
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def _check_shard_join(table, digest_by_entry, path):
    """Every candidate finds its entry, and agrees with its peak list. Per shard, in Arrow.

    Reduced to the DISTINCT (entry_id, q2_digest) pairs first -- a shard holds ~74 entries and
    millions of rows, so checking pairs is O(entries) where checking rows is O(rows). The earlier
    version of this function called `.to_pylist()` on two full columns, which is the per-row Python
    object creation this rewrite exists to avoid.

    Both checks matter: a mis-joined shard is otherwise silent, since every column still parses and
    the numbers are simply attached to the wrong pattern.
    """
    pairs = table.group_by(['entry_id', 'q2_digest']).aggregate([]).to_pylist()
    for pair in pairs:
        entry_id, digest = pair['entry_id'], pair['q2_digest']
        expected = digest_by_entry.get(entry_id)
        if expected is None:
            raise SystemExit(f'{path.name}: entry {entry_id!r} is absent from the entry table')
        if expected != digest:
            raise SystemExit(
                f'{path.name}: entry {entry_id!r} carries q2_digest {digest} but its entry row '
                f'says {expected}. The shards do not belong to the same run.')


def _with_true_lattice(table, entry_ids, true_lattices):
    """Attach the entry's TRUE Bravais lattice, vectorised.

    The counts are grouped by the entry's true lattice while the files are partitioned by the
    candidate's -- METRICS.md section 5 defines the stratum on the truth, and the acceptance floor
    is keyed on it. `index_in` + `take` does the lookup without a Python loop over rows.
    """
    positions = pc.index_in(table.column('entry_id'), value_set=entry_ids)
    return table.append_column('bravais_lattice_true', pc.take(true_lattices, positions))


def _accumulate(table, tally, seen, reachable):
    """Fold one shard into the running counts, entirely in Arrow.

    `group_by(...).aggregate(...)` returns one row per lattice and one per (lattice, entry), both
    tiny, so the only Python-side work is proportional to the number of distinct groups rather than
    to the number of candidates.
    """
    has_off_by_two = 'is_off_by_two' in table.column_names
    aggregations = [('is_correct', 'sum'), ([], 'count_all')]
    if has_off_by_two:
        aggregations.insert(1, ('is_off_by_two', 'sum'))
    summary = table.group_by(['bravais_lattice_true']).aggregate(aggregations).to_pylist()
    for row in summary:
        lattice = row['bravais_lattice_true']
        entry = tally.setdefault(lattice, {'n_candidates': 0, 'n_correct': 0, 'n_off_by_two': 0})
        entry['n_candidates'] += int(row['count_all'])
        entry['n_correct'] += int(row['is_correct_sum'] or 0)
        if has_off_by_two:
            entry['n_off_by_two'] += int(row['is_off_by_two_sum'] or 0)

    for row in table.group_by(['bravais_lattice_true', 'entry_id']).aggregate([]).to_pylist():
        seen.setdefault(row['bravais_lattice_true'], set()).add(row['entry_id'])
    correct_only = table.filter(pc.fill_null(table.column('is_correct'), False))
    for row in (correct_only.group_by(['bravais_lattice_true', 'entry_id'])
                .aggregate([]).to_pylist()):
        reachable.setdefault(row['bravais_lattice_true'], set()).add(row['entry_id'])


def consolidate_bundle(bundle_dirs, out_dir, row_group_size):
    """One bundle's shards and pools, repartitioned by Bravais lattice without ever holding it all.

    Returns (entries, counts) -- the entry table and one row per (bundle, true lattice).

    The old implementation read all 256 shards with `pd.read_parquet` and concatenated them into a
    single frame before writing it back. A core bundle is ~77 M rows and four of the 34 columns are
    list-valued (`xnn`, `unit_cell`, `merit_at_prune`, `hkl_true_in_basis`), so that materialised
    hundreds of millions of Python objects and tens of GB of RAM. Here nothing larger than one
    shard is ever in memory and no column is converted to Python at all.
    """
    bundle_dirs = [Path(d) for d in ([bundle_dirs] if isinstance(bundle_dirs, (str, Path))
                                     else bundle_dirs)]
    entry_paths = _stream_paths(bundle_dirs, 'entries_*.parquet')
    candidate_paths = _stream_paths(bundle_dirs, 'candidates_*.parquet')
    if not entry_paths or not candidate_paths:
        raise SystemExit(f'{bundle_dirs} is missing an entry or candidate stream; do not '
                         'consolidate a partial bundle')

    entries = _read_small(entry_paths)
    bundles = entries['condition_bundle'].unique()
    if len(bundles) != 1:
        raise SystemExit(f'{bundle_dirs} hold more than one bundle: {sorted(bundles)}. '
                         'One directory per bundle.')
    bundle = str(bundles[0])
    if 'condition_bundle' not in entries.columns:
        raise SystemExit(f'{bundle_dirs}: entries carries no condition_bundle column (R8)')

    entry_ids = pa.array(entries['entry_id'].astype(str).tolist())
    true_lattices = pa.array(entries['bravais_lattice_true'].astype(str).tolist())
    digest_by_entry = dict(zip(entries['entry_id'], entries['q2_digest']))

    tally, seen, reachable = {}, {}, {}
    for stream, paths in (('candidates', candidate_paths),
                          ('predownsample',
                           _stream_paths(bundle_dirs, 'predownsample_*.parquet'))):
        if not paths:
            continue
        writers = _LatticeWriters(out_dir, stream, bundle, row_group_size)
        try:
            for path in paths:
                table = pq.read_table(path)
                if table.num_rows == 0:
                    continue
                if 'condition_bundle' not in table.column_names:
                    raise SystemExit(f'{path.name}: no condition_bundle column (R8)')
                if stream == 'candidates':
                    _check_shard_join(table, digest_by_entry, path)
                    _accumulate(_with_true_lattice(table, entry_ids, true_lattices),
                                tally, seen, reachable)
                lattice_column = table.column('bravais_lattice')
                for lattice in pc.unique(lattice_column).to_pylist():
                    writers.write(table.filter(pc.equal(lattice_column, lattice)), lattice)
        finally:
            writers.close()

    counts = pd.DataFrame([
        {'condition_bundle': bundle, 'bravais_lattice_true': lattice,
         'n_candidates': row['n_candidates'], 'n_correct': row['n_correct'],
         'n_off_by_two': row['n_off_by_two'], 'n_entries': len(seen.get(lattice, ())),
         'n_reachable_entries': len(reachable.get(lattice, ()))}
        for lattice, row in sorted(tally.items())])
    return entries, counts


def _consolidate_one(payload):
    """Module-level and picklable, so bundles can be consolidated in parallel under `spawn`."""
    tag, bundle_dirs, out_dir, row_group_size = payload
    entries, counts = consolidate_bundle(bundle_dirs, out_dir, row_group_size)
    return tag, entries, counts


def main(argv=None):
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = bundle_directories(args.dump_root)
    payloads = [(tag, [str(d) for d in dirs], str(out_dir), args.row_group_size)
                for tag, dirs in groups.items()]
    for tag, dirs in groups.items():
        if len(dirs) > 1:
            print(f'{tag}: merging {len(dirs)} directories -- '
                  + ', '.join(str(d.parent.name) for d in dirs), flush=True)

    # Bundles are independent -- each reads its own directory and writes its own files -- so this
    # is the axis to parallelise on. Nine of them, and one process holds one shard's table at a
    # time rather than a whole bundle, so the memory cost of the pool is bounded.
    if args.processes > 1 and len(payloads) > 1:
        from multiprocessing import Pool
        with Pool(processes=min(args.processes, len(payloads))) as pool:
            results = pool.map(_consolidate_one, payloads)
    else:
        results = [_consolidate_one(payload) for payload in payloads]

    all_entries, all_counts, manifests = [], [], {}
    for tag, entries, counts in results:
        bundle = counts['condition_bundle'].iloc[0]
        # The old guard here refused one bundle appearing in two directories. That is now allowed
        # and merged -- a supplementary run regenerating a lost lattice is a legitimate second
        # directory -- and the real hazard, the same (shard, pool) generated twice, is caught by
        # basename in `_stream_paths`, which is exact where the bundle-level guard was not.
        with open(groups[tag][0] / 'manifest.json', encoding='utf-8') as handle:
            manifests[bundle] = json.load(handle)
        all_entries.append(entries)
        all_counts.append(counts)
        print(f'{tag}: {entries.shape[0]} entries, '
              f'{int(counts["n_candidates"].sum())} candidates, '
              f'{int(counts["n_correct"].sum())} correct', flush=True)

    entries = pd.concat(all_entries, ignore_index=True)
    counts = pd.concat(all_counts, ignore_index=True)
    _write(entries, out_dir / 'entries.parquet', args.row_group_size)

    # Coverage, NOT alignment. Every bundle keeps its own entry set; what is recorded is which
    # entries each one is missing, so a later step can restrict deliberately rather than inherit
    # an intersection nobody chose (R14).
    #
    # COMPARED WITHIN AN ARM. The core arm runs every crystal in the manifest and the mechanism arm
    # a nested ~15 % subset, so a union taken across both makes every mechanism bundle look short
    # by the arm difference -- 14 955 entries, which is the design and not a loss. That reading
    # buried the one real signal in the first Benchmark B consolidation.
    per_bundle = entries.groupby('condition_bundle')['entry_id'].apply(set)
    arms = {bundle: FomConditions.bundle_arm(bundle) for bundle in per_bundle.index}
    union_by_arm = {}
    for bundle, ids in per_bundle.items():
        union_by_arm.setdefault(arms[bundle], set()).update(ids)
    coverage = pd.DataFrame([
        {'condition_bundle': bundle,
         'arm': arms[bundle],
         'n_entries': len(ids),
         'n_missing_vs_arm': len(union_by_arm[arms[bundle]] - ids),
         'missing_examples': ','.join(sorted(union_by_arm[arms[bundle]] - ids)[:5])}
        for bundle, ids in per_bundle.items()]).sort_values(['arm', 'condition_bundle'])

    FomBenchmark.write_manifest(
        out_dir,
        consolidated=True,
        bundles=sorted(manifests),
        n_entries=int(entries.shape[0]),
        n_source_entries=int(entries['entry_id'].nunique()),
        n_entries_by_arm={arm: len(ids) for arm, ids in sorted(union_by_arm.items())},
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
