"""S10a: score every candidate on the surplus peaks, and persist it beside the data.

    python mlindex/scripts/run_fom_holdout_merits.py \\
        --pool mlindex/data/fom_benchmark_c2 --processes 8

The indexer fits a cell to the first 20 observed peaks; the benchmark stores up to 60 lines an
entry, because `GenerateDataset.EntryGenerator` truncates there -- a storage cap, not a line
count, and one 54.9 % of entries sit exactly on (C2-F-103). A real pattern carries 20-25 peaks
and 30 at best, so the realistic budgets are the small ones. This scores each candidate on the
peaks it was never fitted to -- **no refit anywhere**, which is the whole claim -- at several
peak budgets at once, and writes the result to `<pool>/holdout_merits/`.

Deliberately a near-clone of `run_fom_floor_merits.py`. The two do the same job on different merit
sets and every hard-won property of that file applies here unchanged: sidecars rather than
rewritten pool files, so the pool stays checksummable against what NERSC wrote and no second copy
is needed on a disk that has no room for one; a projection to the columns actually read, because
the list-valued columns become one Python object per row in pandas; row-group chunking to bound
memory; and independence per (bundle, lattice) file so `--processes` divides the work.

**`--sample-row-groups N` reads N row groups per file instead of all of them.** Row groups here
average ~865 rows across 126 files, so `--sample-row-groups 1` is a stratified sample spanning
every bundle and every Bravais lattice for roughly 1/150th of the work. That is what S10a's
validity gates run on; S10b runs the full pass. **Sample output must go to `--out-dir` outside the
pool**, or S10b's resume-by-default would mistake a sampled file for a finished one -- the script
refuses to write a sampled run into the default location for exactly that reason.

Measured on this machine, warm: **226 microseconds a candidate** for the six-budget grid and 54 for
a single budget, against `reduced_merits`' 341 for the six-column classical set on the same rows.
So the whole hold-out family costs less than the sidecar pass that already ran.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mlindex.model_training import FomBenchmark


# `candidate_id` is only unique within an (entry, bundle, lattice) pool, so all four are needed to
# join without fanning rows out. Identical to the floor sidecar's, so the two join the same way.
JOIN_KEYS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id')

# Everything `holdout_merits` reads from the candidate side. `reciprocal_volume` is the one column
# beyond the floor sidecar's list: Minfo's coincidence density is 4 pi q^2 V / mu, so the hold-out
# form needs the candidate's own volume.
CANDIDATE_COLUMNS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id',
                     'lattice_system', 'spacegroup', 'n_peaks', 'xnn', 'reciprocal_volume')

# And all it reads from the entry side. Wider than the floor sidecar's, necessarily: the surplus
# peaks and their Miller indices are the object being scored, and the two injection counts set the
# rate at which contaminants are seeded into the surplus.
ENTRY_COLUMNS = ('entry_id', 'condition_bundle', 'q2_digest', 'q2_obs', 'q2_holdout',
                 'hkl_holdout', 'n_contaminants', 'second_phase_lines')

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
        description='Score candidates on the surplus peaks and persist it beside a pool')
    parser.add_argument('--pool', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Where the sidecars go. Default is <pool>/holdout_merits')
    parser.add_argument('--processes', type=int, default=1,
                        help='Files scored concurrently. One file per (bundle, lattice), so they '
                             'are independent; each worker holds one file')
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS,
                        help='Candidate rows held at once. Bounds memory per worker')
    parser.add_argument('--sample-row-groups', type=int, default=None,
                        help='Read only the first N row groups of each file. A stratified sample '
                             'across every bundle and lattice, for the validity gates. Requires '
                             '--out-dir, so a sampled run cannot be mistaken for a complete one')
    parser.add_argument('--n-extra', type=int, nargs='*', default=None,
                        help='Peak budgets to score, as surplus-peak counts. Default is the S10 '
                             'grid. n_extra IS the total peak budget minus 20, so 5 is a 25-peak '
                             'pattern -- which is what real data carries')
    parser.add_argument('--no-contaminate', action='store_true',
                        help='Do not seed contaminants into the surplus. The diagnostic arm: the '
                             'generator cannot place them there, so this is the optimistic case '
                             'and is a bound rather than a result')
    parser.add_argument('--mode', choices=FomBenchmark.HOLDOUT_MODES, default='surplus',
                        help="How the hold-out peak list is built. 'surplus' is the uniform "
                             "definition every other step means. 'free_window' gives a candidate "
                             'every peak above the ones it was fitted to, which on cubic is ten '
                             'free window peaks at a fixed pattern length (C2-Q-026); '
                             "'free_equal' takes the same peaks at a fixed COUNT instead. The two "
                             'free modes are PAIRED arms against the uniform one and must never '
                             'be adopted into an aggregate (F-088)')
    parser.add_argument('--bravais-lattices', nargs='+', default=None,
                        help='Score only these Bravais lattices. The pool is partitioned by '
                             '(bundle, lattice), so this is a file filter and costs nothing. Used '
                             'by the cubic arm, which changes nothing outside cF/cI/cP -- every '
                             'other lattice is fitted on all 20 window peaks, so a free-window '
                             'run there is byte-identical to the uniform one and writing it again '
                             'would be pure disk')
    parser.add_argument('--verify', action='store_true',
                        help='Check an existing set of sidecars instead of writing any. Reads '
                             'parquet metadata, not data, so it is seconds even on Benchmark B')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rescore files that already have a sidecar. Off by default, so an '
                             'interrupted run resumes instead of restarting')
    return parser.parse_args(argv)


def score_file(task):
    """One candidate file -> one sidecar, streamed. Module-level and picklable: spawn-safe."""
    path, out_path, pool, chunk_rows, n_extra, contaminate, sample_row_groups, mode = task
    entries = _entries_for(pool)
    source = pq.ParquetFile(path)
    # `schema_arrow`, not `schema`: the parquet schema flattens a list column to its leaf path, so
    # `xnn` appears as `xnn.list.element` and a membership test drops it -- silently, because the
    # read then succeeds and the scorer raises later on the missing column.
    projection = [name for name in CANDIDATE_COLUMNS if name in source.schema_arrow.names]

    groups = range(source.num_row_groups)
    if sample_row_groups is not None:
        groups = range(min(int(sample_row_groups), source.num_row_groups))

    pieces, held, written, out = [], 0, 0, []
    group_list = list(groups)
    for position, index in enumerate(group_list):
        block = source.read_row_group(index, columns=projection).to_pandas()
        pieces.append(block)
        held += block.shape[0]
        if held < chunk_rows and position < len(group_list) - 1:
            continue
        chunk = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
        pieces, held = [], 0
        merits = FomBenchmark.holdout_merits(
            chunk, entries, n_extra_values=n_extra, contaminate=contaminate, mode=mode)
        keys = [key for key in JOIN_KEYS if key in chunk.columns]
        out.append(pd.concat([chunk[keys].reset_index(drop=True),
                              merits.astype('float32').reset_index(drop=True)], axis=1))
        written += chunk.shape[0]

    if not out:
        return path, 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(out_path, index=False)
    return path, written


def verify(pool, out_dir, n_extra, bravais_lattices=None):
    """Check the sidecars without reading their data. Returns (rows, problems).

    Exit code 0 is not evidence that this worked: C2-F-071 is an entire Bravais lattice lost from
    Benchmark B while all 24 generation tasks exited 0. So this checks the three things that go
    wrong quietly -- a sidecar missing, a sidecar short of its candidate file, and a merit column
    written wholly null because the scorer raised for that group and was swallowed.

    `ho_*` columns are legitimately null wherever an entry's surplus is shorter than the budget, so
    *wholly* null is the test rather than any null at all. That is the missing-not-zero rule the
    sweep depends on, and it means a large budget on a short-surplus lattice is expected to be
    sparse rather than absent.
    """
    problems, total = [], 0
    expected_columns = FomBenchmark.holdout_columns(n_extra)
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        if bravais_lattices and not any(path.stem.endswith(f'_{one}') for one in bravais_lattices):
            continue
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
                if stats is not None and name in expected_columns:
                    nulls[name] = nulls.get(name, 0) + stats.null_count
        for name in expected_columns:
            if name not in names:
                problems.append(f'{path.name}: no {name} column')
            elif nulls.get(name, 0) == metadata.num_rows:
                problems.append(f'{path.name}: {name} is wholly null')
    return total, problems


def main(argv=None):
    args = _parse_args(argv)
    pool = Path(args.pool)
    n_extra = tuple(args.n_extra) if args.n_extra else FomBenchmark.HOLDOUT_N_EXTRA
    default_out = pool/'holdout_merits'
    out_dir = Path(args.out_dir) if args.out_dir else default_out

    # A sampled run writes files that look complete and are not. Refusing the default location is
    # cheaper than the alternative, which is S10b resuming over them and silently reporting a
    # hundredth of the pool.
    # A non-default mode changes what every column means, so it may not land in the pool's own
    # sidecar directory either -- an arm written there would be picked up by the next resume as
    # though it were the uniform definition.
    if args.mode != 'surplus' and out_dir == default_out:
        raise SystemExit(
            f'A --mode {args.mode} run must write to an explicit --out-dir outside the pool: its '
            f'columns are a different definition of hold-out, and resume-by-default would treat '
            f'them as the uniform one.')

    if args.sample_row_groups is not None and out_dir == default_out:
        raise SystemExit(
            'A --sample-row-groups run must write to an explicit --out-dir outside the pool: its '
            'sidecars are partial, and resume-by-default would treat them as finished.')

    if args.verify:
        total, problems = verify(pool, out_dir, n_extra, args.bravais_lattices)
        print(f'{pool}: {total} candidates carry hold-out merits in {out_dir}')
        for problem in problems:
            print(f'  FAIL {problem}')
        print('all sidecars complete and populated' if not problems
              else f'{len(problems)} problem(s)')
        return 1 if problems else 0

    tasks = []
    for path in sorted(pool.glob('candidates*.parquet')):
        if args.bravais_lattices and not any(
                path.stem.endswith(f'_{one}') for one in args.bravais_lattices):
            continue
        out_path = out_dir/path.name
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((str(path), str(out_path), str(pool), int(args.chunk_rows), n_extra,
                      not args.no_contaminate, args.sample_row_groups, args.mode))
    if not tasks:
        print(f'{pool}: every sidecar is already written')
        return 0

    processes = max(1, min(int(args.processes), len(tasks)))
    print(f'{pool}: scoring {len(tasks)} files over {processes} process(es) at n_extra={n_extra}, '
          f'contaminate={not args.no_contaminate}, mode={args.mode}')
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
