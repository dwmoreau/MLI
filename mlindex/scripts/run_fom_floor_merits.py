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

from mlindex.model_training import FomBenchmark


# What the sidecar carries besides the merits. `candidate_id` is only unique within an
# (entry, bundle, lattice) pool, so all four are needed to join without fanning rows out.
JOIN_KEYS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Recompute and persist the reduced merit set beside a pool')
    parser.add_argument('--pool', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Where the sidecars go. Default is <pool>/merits')
    parser.add_argument('--processes', type=int, default=1,
                        help='Files scored concurrently. One file per (bundle, lattice), so they '
                             'are independent; each worker holds one file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rescore files that already have a sidecar. Off by default, so an '
                             'interrupted run resumes instead of restarting')
    return parser.parse_args(argv)


def score_file(task):
    """One candidate file -> one sidecar. Module-level and picklable: spawn-safe."""
    path, out_path, pool = task
    entries = FomBenchmark.load_entries(pool)
    candidates = pd.read_parquet(path)
    merits = FomBenchmark.reduced_merits(candidates, entries)
    keys = [key for key in JOIN_KEYS if key in candidates.columns]
    sidecar = pd.concat([candidates[keys].reset_index(drop=True),
                         merits.reset_index(drop=True)], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_parquet(out_path, index=False)
    return path, sidecar.shape[0]


def main(argv=None):
    args = _parse_args(argv)
    pool = Path(args.pool)
    out_dir = Path(args.out_dir) if args.out_dir else pool / 'merits'

    tasks = []
    for path in sorted(pool.glob('candidates*.parquet')):
        out_path = out_dir / path.name
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((str(path), str(out_path), str(pool)))
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
