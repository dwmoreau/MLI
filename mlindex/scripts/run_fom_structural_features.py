"""Compute S12's structural feature columns for a pool and persist them beside the data.

    python mlindex/scripts/run_fom_structural_features.py \\
        --pool mlindex/data/fom_full_c2_pool --processes 8

The learned combiner's design matrix needs three things the pool does not store and the merit
sidecars do not carry: the Werner and de Wolff structural columns campaign 1's `structural` family
was built from, the two absence quantities S04 Phase 2 adopted, and -- for one arm -- the three
merits S00 left on probation. `FomBenchmark.structural_features` computes all three in one pass
over the reference lines; this script runs it file by file and writes the result to
`<pool>/structural/`.

**Why this and not `run_fom_zoo_eval`'s route through `zoo_features`.** That function pays
`compute_all`, measured on this branch at 558 microseconds a candidate against 255 here, to
produce twenty-odd merits S00's audit put below a constant score and which no campaign-2 feature
set may contain. The saving is entirely in merits not computed: every shared column is bit-identical
between the two routes and `tests/test_fom_structural_features.py` pins that, value for value.

At 255 microseconds the fully retained pool's 43.3 M candidates is ~3.1 core-hours, so ~23 minutes
on eight processes; the Benchmark B slice's 13.2 M is ~7 minutes. Computed once and written beside
the pool, per PROTOCOL section 3 rule 8.

**Two reference lists, two cutoffs, and the difference is measurable.** The structural columns use
the candidate's own extinction group's list, which is what `compute_all` does. The absence counts
use the lattice's generic list with the generic list's own cutoff, which is what
`run_fom_symmetry_arms` does -- and which is NOT what `FomBenchmark.extinction_group_sweep` does:
the sweep uses each group's own cutoff, so its window is wider and its count is greater by one on
0.3 to 1.6 % of rows. Both are defensible; this script matches S04 because S04's convention is the
one under which `n_absent_extra_in_range` earned the +0.522 pp that put it in the feature set
(C2-F-041). `--verify` reports the disagreement rather than treating it as a failure.
"""
import argparse
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mlindex.model_training import FomBenchmark


# What the sidecar carries besides the features. `candidate_id` is only unique within an
# (entry, bundle, lattice) pool, so all four are needed to join without fanning rows out.
JOIN_KEYS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id')

# All `structural_features` reads from the candidate side. Projecting to these is the difference
# between a 1 GB read and a 6 GB one: `unit_cell`, `merit_at_prune` and `hkl_true_in_basis` are
# list-valued and become one Python object per row per column in pandas.
CANDIDATE_COLUMNS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id',
                     'lattice_system', 'spacegroup', 'n_peaks', 'xnn')

# And all it reads from the entry side.
ENTRY_COLUMNS = ('entry_id', 'condition_bundle', 'q2_digest', 'q2_obs')

# Rows held at once. `structural_features` groups by (entry, lattice, extinction group) and is
# exact on any subset of a group, so chunking costs a rebuilt reference list at a boundary and
# nothing else. Lower than the merit producer's 2 M because the generic-list pass holds an
# (n_candidates, n_ref) block and the low-symmetry reference lists are large.
CHUNK_ROWS = 1_000_000

_ENTRY_CACHE = {}


def _entries_for(pool):
    """The projected entry table, once per worker process rather than once per file."""
    if pool not in _ENTRY_CACHE:
        frame = FomBenchmark.load_entries(pool)
        keep = [name for name in ENTRY_COLUMNS if name in frame.columns]
        _ENTRY_CACHE[pool] = frame[keep]
    return _ENTRY_CACHE[pool]


def feature_columns(probation=True, absences=True, dropped=True):
    """The columns a sidecar written with these options carries, besides the join keys."""
    names = list(FomBenchmark.STRUCTURAL_COLUMNS)
    if probation:
        names += list(FomBenchmark.PROBATION_MERIT_COLUMNS)
    if dropped:
        names += list(FomBenchmark.DROPPED_MERIT_COLUMNS)
    if absences:
        names += list(FomBenchmark.ABSENCE_COLUMNS)
    return tuple(names)


def score_file(task):
    """One candidate file -> one sidecar, streamed. Module-level and picklable: spawn-safe."""
    # The options ride in the tuple rather than in a module global: Windows and macOS both spawn,
    # so a global set in main() is not visible in the worker (CLAUDE.md's spawn-safety rule).
    path, out_path, pool, chunk_rows, probation, absences, dropped = task
    entries = _entries_for(pool)
    source = pq.ParquetFile(path)
    # `schema_arrow`, not `schema`: the parquet schema flattens a list column to its leaf path, so
    # `xnn` appears as `xnn.list.element` and a membership test drops it -- silently, because the
    # read then succeeds and `structural_features` raises later on the missing column.
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
        features = FomBenchmark.structural_features(
            chunk, entries, probation=probation, absences=absences, dropped=dropped)
        keys = [key for key in JOIN_KEYS if key in chunk.columns]
        out.append(pd.concat([chunk[keys].reset_index(drop=True),
                              features.reset_index(drop=True)], axis=1))
        written += chunk.shape[0]

    if not out:
        return path, 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(out_path, index=False)
    return path, written


def verify(pool, out_dir, sweep_dir=None, sweep_rows=200_000):
    """Check the sidecars. Returns (rows, problems, notes).

    Exit code 0 is not evidence that this worked: C2-F-071 is an entire Bravais lattice lost from
    Benchmark B while all 24 generation tasks exited 0. So this checks the three things that go
    wrong quietly -- a sidecar missing, a sidecar short of its candidate file, and a column written
    wholly null because the computation raised for one group and was swallowed. Null counts come
    from parquet's own column statistics, so the whole set is checked in seconds rather than read.

    `sweep_dir` adds the check that is worth more than the other three together: the absence counts
    against `extinction_group_sweep`'s, which reach the same quantity through an independent
    implementation. It is reported as a NOTE and not a failure, because the two use different
    counting windows on purpose -- see this module's docstring -- so the expected agreement is high
    but not exact, and a *drop* in it is the signal.
    """
    problems, notes, total = [], [], 0
    carried_names = set(feature_columns()) | set(feature_columns(probation=False, dropped=False))
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
                if stats is not None and names[column] in carried_names:
                    nulls[names[column]] = nulls.get(names[column], 0) + stats.null_count
        carried = [name for name in names if name not in JOIN_KEYS]
        if not carried:
            problems.append(f'{path.name}: no feature columns at all')
        for name in carried:
            if nulls.get(name, 0) == metadata.num_rows:
                problems.append(f'{path.name}: {name} is wholly null')

    if sweep_dir is not None:
        notes.extend(_sweep_agreement(pool, out_dir, sweep_dir, sweep_rows))
    return total, problems, notes


def _sweep_agreement(pool, out_dir, sweep_dir, sweep_rows):
    """`n_absent_extra_in_range` against the extinction sweep's, on a bounded sample per file."""
    notes = []
    for path in sorted(Path(sweep_dir).glob('candidates*.parquet')):
        sidecar = Path(out_dir)/path.name
        if not sidecar.exists():
            continue
        keys = list(JOIN_KEYS)
        mine = pd.read_parquet(sidecar, columns=keys + ['n_absent_extra_in_range'])
        theirs = pd.read_parquet(path, columns=keys + ['xg_M20_n_absent_in_range'])
        joined = mine.merge(theirs, on=keys, how='inner', validate='1:1')
        if not joined.shape[0]:
            notes.append(f'{path.name}: sweep joins no rows')
            continue
        joined = joined.head(int(sweep_rows))
        agree = joined['n_absent_extra_in_range'].to_numpy() == \
            joined['xg_M20_n_absent_in_range'].to_numpy()
        gap = (joined['xg_M20_n_absent_in_range'].to_numpy()
               - joined['n_absent_extra_in_range'].to_numpy())
        notes.append(f'{path.name}: sweep agreement {agree.mean():.4f} on {len(joined)} rows, '
                     f'sweep minus this in [{gap.min()}, {gap.max()}]')
    return notes


def _commit():
    """The commit the sidecars were written at, or None outside a checkout."""
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute S12's structural feature columns and persist them beside a pool")
    parser.add_argument('--pool', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Where the sidecars go. Default is <pool>/structural')
    parser.add_argument('--processes', type=int, default=1,
                        help='Files scored concurrently. One file per (bundle, lattice), so they '
                             'are independent; each worker holds one file')
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS,
                        help='Candidate rows held at once. Bounds memory per worker; the features '
                             'are exact on any subset of an (entry, lattice, group) so chunking '
                             'costs a rebuilt reference list at a boundary and nothing else')
    parser.add_argument('--no-probation', action='store_true',
                        help="Omit M_wu, M_1 and F_N_q. They cost 42 microseconds a candidate and "
                             'they are what lets S00 probation merits be decided by a retrained '
                             'arm rather than left undecided, so omitting them saves little')
    parser.add_argument('--no-dropped', action='store_true',
                        help='Omit the six merits S00 cut. They cost 1 microsecond a candidate '
                             "and they are what S12's plus_dropped_merits arm restores, which is "
                             'the retrained arm that licenses the cut from 17 merits to 7')
    parser.add_argument('--no-absences', action='store_true',
                        help='Omit n_absent_extra_in_range and n_ref_in_range. These need a second '
                             'pass over the generic reference list, so this is the option that '
                             'saves real time on the low-symmetry lattices')
    parser.add_argument('--verify', action='store_true',
                        help='Check an existing set of sidecars instead of writing any: every '
                             'candidate file has one, the row counts match file by file, and no '
                             'column is wholly null. Reads parquet metadata, not data')
    parser.add_argument('--sweep-dir', type=str, default=None,
                        help='With --verify, also compare the absence counts against an extinction '
                             'sweep in this directory. Reported as a note, not a failure: the two '
                             'use different counting windows by design')
    parser.add_argument('--sweep-rows', type=int, default=200_000,
                        help='Rows per file used for the sweep comparison')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rescore files that already have a sidecar. Off by default, so an '
                             'interrupted run resumes instead of restarting')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    pool = Path(args.pool)
    out_dir = Path(args.out_dir) if args.out_dir else pool / 'structural'
    probation = not args.no_probation
    absences = not args.no_absences
    dropped = not args.no_dropped

    if args.verify:
        total, problems, notes = verify(pool, out_dir, args.sweep_dir, args.sweep_rows)
        print(f'{pool}: {total} candidates carry structural features in {out_dir}')
        for note in notes:
            print(f'  note {note}')
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
        tasks.append((str(path), str(out_path), str(pool), int(args.chunk_rows),
                      probation, absences, dropped))
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
        print(f'  {Path(path).name}: {rows}', flush=True)
    if processes > 1:
        handle.close()
        handle.join()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'_meta.json').write_text(json.dumps({
        'pool': str(pool),
        'commit': _commit(),
        'columns': list(feature_columns(probation, absences, dropped)),
        'probation': probation,
        'absences': absences,
        'dropped': dropped,
        'chunk_rows': int(args.chunk_rows),
        'g_min': FomBenchmark.STRUCTURAL_G_MIN,
        'absence_window': 'generic list, generic cutoff (run_fom_symmetry_arms convention)',
        'n_candidates': int(total),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        }, indent=2, sort_keys=True), encoding='utf-8')
    print(f'{pool}: {total} candidates scored -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
