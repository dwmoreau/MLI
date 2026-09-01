"""S11: re-run the extinction-group argmax under every candidate criterion, and persist it.

    python mlindex/scripts/run_fom_extinction_sweep.py --gate --sample-row-groups 1
    python mlindex/scripts/run_fom_extinction_sweep.py --processes 8

`Candidates.assign_extinction_group` chooses which calculated lines a cell should produce, by
argmax of M20 over up to 68 extinction groups, and rebinds `best_M20` to the value at the winner.
This re-runs that argmax offline under each of `EXTINCTION_CRITERIA` and writes the outcome to
`<pool>/extinction_sweep/`, so the comparison is a restriction of one persisted pass rather than
five (PROTOCOL section 3 rule 8 -- campaign 1 lost four derived quantities by recomputing them).

It runs on `mlindex/data/fom_full_c2_pool`, the fully retained pool, and not on Benchmark B. Two
independent reasons, and either alone would be decisive. The new rule changes stored `M20`, one
of the seven merits the subsampler ranked on, so on a subsampled pool the arms fall outside the
retention rule and every rank metric comes out optimistic (C2-R-013, and C2-F-084 is the
precedent for a merit sharing a NAME with one of the seven and still falling outside). And a
subsampled pool distorts every candidate-level statistic, not only rank metrics -- the same
control reads 0.4558 on the slice and 0.5865 on the retained pool, a sign flip either side of
chance, and nothing in the code refuses it (C2-F-111, C2-R-020).

**aP is skipped.** It has one extinction group, so every arm is identically the same choice and
the rule cannot change anything. Its 645 correct candidates would only pad an aggregate with rows
that carry no information about the question (DWMM, 2026-09-01).

The gates matter more than the sweep and run first. In order: G0 the reference lists still match
what the pool recorded; G2 the masked q2 route is bit-identical (it is, and it is also SLOWER, so
it stays off -- see `extinction_group_sweep`); G1 the offline argmax under M20 reproduces the
stored `spacegroup` and the stored `M20` EXACTLY; G3 the sweep agrees with a real `Candidates`;
G4 how often the `M_rev` support floor fires, per lattice and per group. **G1 is the one that
licenses everything else** -- C2-F-036 got 310 807 of 310 807 on the pre-deduplication stream, and
this pool is post-deduplication, which is the one place the C2-F-036 mis-attachment could surface.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark
from mlindex.utilities.ExtinctionCounts import LATTICE_SYSTEM, get_absence_counts
from mlindex.utilities.FigureOfMerits import EXTINCTION_CRITERIA, m_rev_support_floor

# `candidate_id` is unique only within an (entry, bundle, lattice) pool, so all four are needed to
# join without fanning rows out. Identical to the hold-out and floor sidecars', so they all join
# the same way.
JOIN_KEYS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id')

CANDIDATE_COLUMNS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id',
                     'lattice_system', 'spacegroup', 'n_peaks', 'xnn', 'M20', 'is_correct',
                     'n_absent_extra', 'n_groups_searched', 'final_rank')

ENTRY_COLUMNS = ('entry_id', 'condition_bundle', 'q2_obs', 'bravais_lattice_true',
                 'extinction_group_true', 'split', 'volume_decile')

# One group with one choice. Nothing to measure, so nothing is read.
SKIP_LATTICES = ('aP',)

CHUNK_ROWS = 400_000

_ENTRY_CACHE = {}


def _entries_for(pool):
    """The projected entry table, once per worker process rather than once per file."""
    if pool not in _ENTRY_CACHE:
        frame = FomBenchmark.load_entries(pool)
        keep = [name for name in ENTRY_COLUMNS if name in frame.columns]
        _ENTRY_CACHE[pool] = frame[keep].set_index(['entry_id', 'condition_bundle'])
    return _ENTRY_CACHE[pool]


def sweep_chunk(chunk, entries, criteria):
    """Run the argmax over every group for one candidate frame. Returns a sidecar frame.

    Grouped by (entry, bundle, lattice, n_peaks) because the reference lists and the observed
    peaks are shared across every candidate of one pattern, so the cctbx-backed group lists are
    built once per group rather than once per row.
    """
    rows = []
    keys = ['entry_id', 'condition_bundle', 'bravais_lattice', 'lattice_system', 'n_peaks']
    for (entry_id, bundle, bravais_lattice, lattice_system, n_peaks), block in chunk.groupby(
            keys, sort=False):
        entry = entries.loc[(entry_id, bundle)]
        q2_obs = np.asarray(entry['q2_obs'], dtype=np.float64)[:int(n_peaks)]
        xnn = np.stack([np.asarray(v, dtype=np.float64) for v in block['xnn']])
        group_keys, winners, M20, scores, n_cal, n_absent = FomBenchmark.extinction_group_sweep(
            q2_obs, xnn, lattice_system, bravais_lattice, criteria=criteria)
        index = {key: position for position, key in enumerate(group_keys)}

        out = block[list(JOIN_KEYS)].reset_index(drop=True).copy()
        take = np.arange(xnn.shape[0])
        for criterion in criteria:
            winner = winners[criterion]
            out[f'xg_{criterion}_group_index'] = winner.astype(np.int16)
            out[f'xg_{criterion}_M20'] = M20[take, winner]
            out[f'xg_{criterion}_n_absent_in_range'] = n_absent[take, winner].astype(np.int16)
            out[f'xg_{criterion}_n_cal'] = n_cal[take, winner].astype(np.int16)
            if criterion != 'M20':
                out[f'xg_{criterion}_score'] = scores[criterion][take, winner]
        # The incumbent's own answer, so a disagreement rate is one subtraction on the sidecar
        # rather than a second join against the pool.
        out['xg_stored_group_index'] = block['spacegroup'].map(index).to_numpy()
        out['xg_n_floored_groups'] = (n_cal < m_rev_support_floor()).sum(axis=1).astype(np.int16)
        out['xg_n_groups'] = np.int16(len(group_keys))
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else None


def sweep_file(task):
    """One candidate file -> one sidecar, streamed. Module-level and picklable: spawn-safe."""
    path, out_path, pool, chunk_rows, criteria, sample_row_groups = task
    entries = _entries_for(pool)
    source = pq.ParquetFile(path)
    # `schema_arrow`, not `schema`: parquet flattens a list column to its leaf path, so `xnn`
    # appears as `xnn.list.element` and a membership test drops it -- silently, because the read
    # then succeeds and the sweep raises later on the missing column.
    projection = [name for name in CANDIDATE_COLUMNS if name in source.schema_arrow.names]

    groups = list(range(source.num_row_groups))
    if sample_row_groups is not None:
        groups = groups[:int(sample_row_groups)]

    pieces, held, out = [], 0, []
    for position, index in enumerate(groups):
        block = source.read_row_group(index, columns=projection).to_pandas()
        pieces.append(block)
        held += block.shape[0]
        if held < chunk_rows and position < len(groups) - 1:
            continue
        chunk = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
        pieces, held = [], 0
        result = sweep_chunk(chunk, entries, criteria)
        if result is not None:
            out.append(result)

    if not out:
        return path, 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(out, ignore_index=True)
    frame.to_parquet(out_path, index=False)
    return path, frame.shape[0]


def candidate_tasks(pool, out_dir, criteria, chunk_rows, sample_row_groups, bravais_lattices,
                    overwrite):
    tasks = []
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        lattice = path.stem.split('_')[-1]
        if lattice in SKIP_LATTICES:
            continue
        if bravais_lattices and lattice not in bravais_lattices:
            continue
        out_path = Path(out_dir)/path.name
        if out_path.exists() and not overwrite:
            continue
        tasks.append((str(path), str(out_path), pool, chunk_rows, tuple(criteria),
                      sample_row_groups))
    return tasks


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Re-run the extinction-group argmax under every criterion and persist it')
    parser.add_argument('--pool', default=os.path.join('mlindex', 'data', 'fom_full_c2_pool'),
                        help='The fully retained pool. Not Benchmark B: the arms fall outside its '
                             'retention rule (C2-R-013) and it distorts candidate-level '
                             'statistics (C2-R-020)')
    parser.add_argument('--out-dir', default=None,
                        help='Where the sidecars go. Default is <pool>/extinction_sweep')
    parser.add_argument('--criteria', nargs='+', default=list(EXTINCTION_CRITERIA),
                        help='Which criteria to sweep. They share the per-group q2 and '
                             'fast_assign, so five cost about twice one rather than five times')
    parser.add_argument('--processes', type=int, default=1,
                        help='Files swept concurrently; one file per (bundle, lattice)')
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS)
    parser.add_argument('--sample-row-groups', type=int, default=None,
                        help='Read only the first N row groups of each file: a stratified sample '
                             'across every bundle and lattice, for the gates. Requires --out-dir, '
                             'so a sampled run cannot be mistaken for a complete one')
    parser.add_argument('--bravais-lattices', nargs='+', default=None)
    parser.add_argument('--gate', action='store_true',
                        help='Run G0-G4 and write the gate report instead of sweeping')
    parser.add_argument('--gate-rows', type=int, default=200_000,
                        help='Candidate rows per lattice for G1, the reproduction gate')
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom_campaign2',
                                                               'artifacts'))
    parser.add_argument('--tag', default='S11')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-sweep files that already have a sidecar. Off by default, so an '
                             'interrupted run resumes instead of restarting')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    pool = args.pool
    out_dir = args.out_dir or os.path.join(pool, 'extinction_sweep')
    if args.sample_row_groups is not None and args.out_dir is None and not args.gate:
        raise SystemExit(
            '--sample-row-groups writes a partial sidecar; give it an explicit --out-dir so a '
            'resume can never mistake it for a finished one')

    if args.gate:
        from mlindex.scripts.run_fom_extinction_gates import run_gates
        return run_gates(pool, args.artifact_dir, args.tag, args.gate_rows, args.criteria,
                         args.bravais_lattices)

    tasks = candidate_tasks(pool, out_dir, args.criteria, args.chunk_rows,
                            args.sample_row_groups, args.bravais_lattices, args.overwrite)
    if not tasks:
        print('nothing to do -- every sidecar exists (use --overwrite to redo them)')
        return 0
    print(f'sweeping {len(tasks)} files into {out_dir}')
    if args.processes > 1:
        with Pool(args.processes) as workers:
            results = workers.map(sweep_file, tasks)
    else:
        results = [sweep_file(task) for task in tasks]

    total = sum(rows for _, rows in results)
    for path, rows in sorted(results):
        print(f'  {Path(path).name:52s} {rows:>10,}')
    print(f'{total:,} candidates swept')

    meta = {
        'pool': str(pool),
        'out_dir': str(out_dir),
        'criteria': list(args.criteria),
        'skipped_lattices': list(SKIP_LATTICES),
        'm_rev_support_floor': m_rev_support_floor(),
        'commit': FomBenchmark.commit_hash() if hasattr(FomBenchmark, 'commit_hash') else None,
        # Position is the only label a group index has -- read it back through this, never by
        # assumption. `merit_at_prune` mislabelled four of seven entries exactly this way, and
        # nothing was able to detect it (C2-F-067).
        #
        # These come from `spacegroup_reference_sets`, which is the dict the sweep itself
        # enumerates, and NOT from `get_absence_counts`. The committed absence-count JSON holds
        # the same keys in ALPHABETICAL order while `get_spacegroup_hkl_ref` yields them in
        # insertion order, and they differ on mP, oP, tP and oC at least -- so writing the JSON's
        # order here would mislabel every index in every sidecar, silently, which is C2-F-067
        # happening a second time in the same campaign.
        'group_keys': {
            lattice: list(FomBenchmark.spacegroup_reference_sets(LATTICE_SYSTEM[lattice], lattice))
            for lattice in sorted(get_absence_counts()) if lattice not in SKIP_LATTICES
            },
        'n_candidates': int(total),
        }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, '_meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
