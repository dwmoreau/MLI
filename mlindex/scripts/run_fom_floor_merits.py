"""Recompute the reduced merit set for a pool and persist it beside the data.

    python mlindex/scripts/run_fom_floor_merits.py \\
        --pool mlindex/data/fom_floor_c2/consolidated/seed202 --processes 8

`SCHEMA.md` stores only `M20` and `Minfo` on a candidate row: the other six of the reduced core --
`M_tilde`, `M_rev`, `M_sym`, `X_N`, `n_over`, `max_gap` -- are recomputable from `xnn`, the peak
list and the extinction group, and by the schema's own rule a recomputable column does not earn
storage. That is the right call for a 122 GB pool. It is the wrong call to make *repeatedly*:
measured here, the recompute runs at **136 microseconds a candidate**, so the floor's four arms at
13.08 M candidates each is **~2 hours** of one core.

So it is computed once and written beside the pool -- PROTOCOL section 3 rule 8, which exists
because campaign 1 lost this four times over: a basin count, a generator provenance column, a
posterior's own denominator, and correctness labels on a 57-million-row dump that every later pass
then recomputed.

**Sidecars, not rewritten candidate files.** The merits go to `merits/<same filename>` carrying the
join keys and the six columns, rather than being appended to the pool's own parquets. Two reasons:
rewriting would duplicate the whole pool on a disk that has room for the sidecars and not for a
second copy, and a pool file that differs from the one NERSC wrote is a pool that can no longer be
checksummed against it.

The files are independent -- one per (bundle, Bravais lattice) -- so `--processes` divides the
work almost linearly.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mlindex.model_training import FomBenchmark


# What the sidecar carries besides the merits. `candidate_id` is only unique within an
# (entry, bundle, lattice) pool, so all four are needed to join without fanning rows out.
JOIN_KEYS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id')

# All `reduced_merits` reads from the candidate side. Projecting to these is the difference between
# a 1 GB read and a 6 GB one on Benchmark B's larger files: `unit_cell`, `merit_at_prune` and
# `hkl_true_in_basis` are list-valued and become one Python object per row per column in pandas.
CANDIDATE_COLUMNS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id',
                     'lattice_system', 'spacegroup', 'n_peaks', 'xnn')

# And all it reads from the entry side. The full entry table carries q2_holdout, hkl_holdout,
# hkl_true and the ground-truth cell; on Benchmark B's 106 235 rows that is ~2 GB in pandas, and
# the unprojected version loaded it once PER FILE in every worker.
ENTRY_COLUMNS = ('entry_id', 'condition_bundle', 'q2_digest', 'q2_obs')

# Rows held at once. `reduced_merits` groups by (entry, lattice, extinction group) and is exact on
# any subset of a group -- the merits are per candidate given the entry's peak list -- so chunking
# costs a rebuilt reference list at a boundary and nothing else.
CHUNK_ROWS = 2_000_000

_ENTRY_CACHE = {}


def _entries_for(pool):
    """The projected entry table, once per worker process rather than once per file."""
    if pool not in _ENTRY_CACHE:
        frame = FomBenchmark.load_entries(pool)
        keep = [name for name in ENTRY_COLUMNS if name in frame.columns]
        _ENTRY_CACHE[pool] = frame[keep]
    return _ENTRY_CACHE[pool]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Recompute and persist the reduced merit set beside a pool')
    parser.add_argument('--pool', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Where the sidecars go. Default is <pool>/merits')
    parser.add_argument('--processes', type=int, default=1,
                        help='Files scored concurrently. One file per (bundle, lattice), so they '
                             'are independent; each worker holds one file')
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS,
                        help='Candidate rows held at once. Bounds memory per worker; the merits '
                             'are exact on any subset of an (entry, lattice, group) so chunking '
                             'costs a rebuilt reference list at a boundary and nothing else')
    parser.add_argument('--verify', action='store_true',
                        help='Check an existing set of sidecars instead of writing any: every '
                             'candidate file has one, the row counts match file by file, and no '
                             'merit column is wholly null. Reads parquet metadata, not data, so '
                             'it is seconds even on Benchmark B')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rescore files that already have a sidecar. Off by default, so an '
                             'interrupted run resumes instead of restarting')
    return parser.parse_args(argv)


def score_file(task):
    """One candidate file -> one sidecar, streamed. Module-level and picklable: spawn-safe."""
    path, out_path, pool, chunk_rows = task
    entries = _entries_for(pool)
    source = pq.ParquetFile(path)
    # `schema_arrow`, not `schema`: the parquet schema flattens a list column to its leaf
    # path, so `xnn` appears as `xnn.list.element` and a membership test drops it -- silently,
    # because the read then succeeds and `reduced_merits` raises later on the missing column.
    projection = [name for name in CANDIDATE_COLUMNS if name in source.schema_arrow.names]

    pieces, held, written = [], 0, 0
    out = []
    for index in range(source.num_row_groups):
        block = source.read_row_group(index, columns=projection).to_pandas()
        pieces.append(block)
        held += block.shape[0]
        if held < chunk_rows and index < source.num_row_groups - 1:
            continue
        chunk = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
        pieces, held = [], 0
        merits = FomBenchmark.reduced_merits(chunk, entries)
        keys = [key for key in JOIN_KEYS if key in chunk.columns]
        out.append(pd.concat([chunk[keys].reset_index(drop=True),
                              merits.reset_index(drop=True)], axis=1))
        written += chunk.shape[0]

    if not out:
        return path, 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(out_path, index=False)
    return path, written


MERIT_COLUMNS = ('M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap')


def verify(pool, out_dir):
    """Check the sidecars without reading their data. Returns (rows, problems).

    Exit code 0 is not evidence that this worked: C2-F-071 is an entire Bravais lattice lost from
    Benchmark B while all 24 generation tasks exited 0. So this checks the three things that go
    wrong quietly -- a sidecar missing, a sidecar short of its candidate file, and a merit column
    written wholly null because the recompute raised for that group and was swallowed.

    Null counts come from parquet's own column statistics, so a 26 GB sidecar set is checked in
    seconds rather than read.
    """
    problems, total = [], 0
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        sidecar = Path(out_dir)/path.name
        if not sidecar.exists():
            problems.append(f'{path.name}: NO SIDECAR')
            continue
        expected = pq.ParquetFile(path).metadata.num_rows
        metadata = pq.ParquetFile(sidecar).metadata
        if metadata.num_rows != expected:
            problems.append(f'{path.name}: {metadata.num_rows} rows against {expected}')
            continue
        total += metadata.num_rows
        names = list(metadata.schema.names)
        nulls = {}
        for group in range(metadata.num_row_groups):
            row_group = metadata.row_group(group)
            for column in range(row_group.num_columns):
                stats = row_group.column(column).statistics
                name = names[column]
                if stats is not None and name in MERIT_COLUMNS:
                    nulls[name] = nulls.get(name, 0) + stats.null_count
        for name in MERIT_COLUMNS:
            if name not in names:
                problems.append(f'{path.name}: no {name} column')
            elif nulls.get(name, 0) == metadata.num_rows:
                problems.append(f'{path.name}: {name} is wholly null')
    return total, problems


def main(argv=None):
    args = _parse_args(argv)
    pool = Path(args.pool)
    out_dir = Path(args.out_dir) if args.out_dir else pool / 'merits'

    if args.verify:
        total, problems = verify(pool, out_dir)
        print(f'{pool}: {total} candidates carry merits in {out_dir}')
        for problem in problems:
            print(f'  FAIL {problem}')
        print('all sidecars complete and populated' if not problems
              else f'{len(problems)} problem(s)')
        return 1 if problems else 0

    tasks = []
    for path in sorted(pool.glob('candidates*.parquet')):
        out_path = out_dir / path.name
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((str(path), str(out_path), str(pool), int(args.chunk_rows)))
    if not tasks:
        print(f'{pool}: every sidecar is already written')
        return 0

    processes = max(1, min(int(args.processes), len(tasks)))
    print(f'{pool}: scoring {len(tasks)} files over {processes} process(es)')
    total = 0
    if processes == 1:
        results = map(score_file, tasks)
    else:
        handle = Pool(processes)
        results = handle.imap_unordered(score_file, tasks)
    for path, rows in results:
        total += rows
        print(f'  {Path(path).name}: {rows}')
    if processes > 1:
        handle.close()
        handle.join()
    print(f'{pool}: {total} candidates scored -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
