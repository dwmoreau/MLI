"""S03 Phase 2 -- capture every candidate merit at the prune site, and re-run.

S03 Phase 1 settled the prune from campaign 1's threshold-0 dump: the cut is on the right merit at
the right stage and at the wrong value, because moving it later makes refinement and the
off-by-two check unconditional and paying for those on the whole pool costs the same as keeping
the whole pool where the cut already is (C2-F-021).

One question that dump cannot answer. It stored `m20_at_prune` and nothing else at the cut, so
"would a different merit make a better cut *where the cut already is*?" has never been measured
(C2-R-001). C2-F-021 is what makes it worth measuring: a criterion has to win at point A to
change anything, since every later-stage criterion is dominated at matched wall clock.

This script produces the pool that answers it. It runs the same entries, at threshold 0, with
`prune_criterion_capture` on, so every candidate carries `M20`, `M_tilde`, `M_rev`, `M_sym`,
`X_N`, `n_over` and `max_gap` **as the cut saw them** alongside the final values.

WHAT IT DOES NOT DO. It does not regenerate peak lists. Every archived `entries_*.parquet`
already carries `q2_obs`, the split and the truth columns, so the driver reads them -- which is
what keeps this a few hundred lines instead of a harness port (`CHERRY_PICK.md`).

THE POOL IS NOT CAMPAIGN 1's POOL, and that is the point. Different code and different seeding, so
points A, B and C all come from one run and are perfectly paired -- which a restriction of the
archived dump cannot give. It is relabelled from scratch. The archived dump becomes a cross-check
(does the censoring reproduce campaign 1's 72-88 % band?), not a dependency.

SEEDING. Per `(entry_id, bravais_lattice)`, from a stable digest rather than Python's salted
`hash`. Campaign 1 seeded once per pool and advanced with every entry, so no subset of its runs
could be regenerated comparably and every result in its final phase inherited a within-run
restriction (PROTOCOL section 6). This also pilots the scheme S07 needs.

    python mlindex/scripts/run_fom_prune_rerun.py --arm general --processes 8
    python mlindex/scripts/run_fom_prune_rerun.py --arm hard --processes 8

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn

BRAVAIS_LATTICES = ('cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP',
                    'aP')
BROADENING_TAG = '1'
BASE_SEED = 12345

# Which columns of `unit_cell_true` the labeller compares, per lattice system. The truth table
# stores a full six-parameter cell (a, b, c, alpha, beta, gamma); the batch labeller wants the
# free parameters only, in the order `get_unit_cell_from_xnn(partial_unit_cell=True)` returns
# them. These are INDEX LISTS, not ranges -- monoclinic takes beta and not alpha, and rhombohedral
# takes alpha and not c, so a contiguous slice silently compares the wrong angle.
TRUTH_SLICE = {
    'cubic': [0],
    'tetragonal': [0, 2],
    'hexagonal': [0, 2],
    'rhombohedral': [0, 3],
    'orthorhombic': [0, 1, 2],
    'monoclinic': [0, 1, 2, 4],
    'triclinic': [0, 1, 2, 3, 4, 5],
    }

ARMS = {
    'hard': os.path.join('mlindex', 'characterization', 'fom', 'retention', 't0'),
    'general': os.path.join('mlindex', 'characterization', 'fom', 'allstrata', 't0'),
    }

OUT_ROOT = os.path.join('mlindex', 'characterization', 'fom', 'prune_capture')


def derived_seed(entry_id, bravais_lattice):
    """A stable per-(entry, lattice) seed. `hash` is salted per process and cannot be used."""
    digest = hashlib.sha256(f'{BASE_SEED}:{entry_id}:{bravais_lattice}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') % (2 ** 31 - 1)


def bundle_directories(root):
    directories = {}
    for child in sorted(Path(root).iterdir()):
        if child.is_dir() and any(child.glob('entries_*.parquet')):
            directories[child.name] = child
    if not directories:
        raise SystemExit(f'no entry tables under {root}')
    return directories


def run_entry(optimizers, task_queues, entry, n_top_candidates):
    """All fourteen lattices for one pattern, at the capture settings. Returns one frame."""
    from mlindex.optimization.MPOptimizer import run_mp_bl

    q2_obs = np.asarray(entry['q2_obs'], dtype=np.float64)
    frames = []
    for bravais_lattice in BRAVAIS_LATTICES:
        optimizer = optimizers[bravais_lattice]
        run_mp_bl(optimizer, bravais_lattice, task_queues, q2_obs[:optimizer.n_peaks],
                  False, None, n_top_candidates,
                  run_seed=derived_seed(entry['entry_id'], bravais_lattice))
        pool = optimizer.predownsample
        if pool is None:
            raise RuntimeError(f'no capture for {entry["entry_id"]}/{bravais_lattice}; '
                               'is prune_criterion_capture set?')
        frame = pd.DataFrame({
            'entry_id': entry['entry_id'],
            'bravais_lattice': pool['bravais_lattice'],
            'lattice_system': pool['lattice_system'],
            'candidate_id': np.arange(pool['M20'].shape[0]),
            'xnn': list(pool['xnn']),
            'spacegroup': pool['spacegroup'],
            'hkl_ref_length': pool['hkl_ref_length'],
            'n_peaks': pool['n_peaks'],
            'M20': pool['M20'],
            'Minfo': pool['Minfo'],
            'n_indexed': pool['n_indexed'],
            'downsample_radius': pool['downsample_radius'],
            })
        for name, values in pool.items():
            if name == 'm20_at_prune' or name.startswith('merit_at_prune_'):
                frame[name] = values
        frames.append(frame)
        optimizer.predownsample = None
    return pd.concat(frames, ignore_index=True)


def label(frame, entries):
    """`is_correct` for one pattern's whole pool, batched per lattice system.

    Only rows on the entry's own true Bravais lattice can be correct, and they are labelled
    against the truth cell restricted to that system's free parameters. Everything else is False
    by construction, exactly as campaign 1's labelling pass recorded it.
    """
    truth = entries.set_index('entry_id')
    is_correct = np.zeros(frame.shape[0], dtype=bool)
    for entry_id, group in frame.groupby('entry_id', sort=False):
        true_lattice = truth.loc[entry_id, 'bravais_lattice_true']
        on_lattice = (group['bravais_lattice'].to_numpy() == true_lattice)
        if not on_lattice.any():
            continue
        system = group['lattice_system'].to_numpy()[on_lattice][0]
        unit_cell_true = np.asarray(truth.loc[entry_id, 'unit_cell_true'], dtype=np.float64)
        predicted = get_unit_cell_from_xnn(
            np.stack(group['xnn'].to_numpy()[on_lattice]), partial_unit_cell=True,
            lattice_system=system)
        rows = frame.index.get_indexer(group.index)[on_lattice]
        is_correct[rows] = is_correct_known_bl_batch(
            unit_cell_true[TRUTH_SLICE[system]], predicted, system, rtol=0.01)
    frame['is_correct'] = is_correct
    return frame


def capture_gate(frame):
    """The captured M20 must equal the value the rule tested, row for row.

    Both come from `best_xnn` through the same `get_q2` -> `fast_assign` route, so a difference
    means the route diverged. The one known exception is C2-F-026: candidates `fix_unphysical`
    repaired at construction carry a `best_xnn` from before the repair and a `best_M20` from
    after, so the two describe different cells. Reported as a count and a rate rather than raised,
    because it is a defect in the constructor rather than in the capture, and it is on the record.
    """
    captured = frame['merit_at_prune_M20'].to_numpy()
    tested = frame['m20_at_prune'].to_numpy()
    both_nan = np.isnan(captured) & np.isnan(tested)
    return int(np.sum(~both_nan & (captured != tested)))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', default='general', choices=sorted(ARMS))
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--out-root', default=OUT_ROOT)
    parser.add_argument('--n-top-candidates', type=int, default=20)
    parser.add_argument('--limit-entries', type=int, default=None,
                        help='smoke test only; never for a result')
    parser.add_argument('--shard-stride', type=int, default=1,
                        help='run every Nth shard, so N copies of this script can share the arm')
    parser.add_argument('--shard-offset', type=int, default=0,
                        help='which residue class of shards this copy takes')
    return parser.parse_args()


def main():
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, shutdown_mp_workers

    args = _parse_args()
    root = Path(BASE) / ARMS[args.arm]
    out_root = Path(BASE) / args.out_root / args.arm
    out_root.mkdir(parents=True, exist_ok=True)

    options = {'prune_m20_threshold': 0.0, 'prune_criterion_capture': True}
    optimizers, processes, task_queues = setup_mp_optimizers(
        args.processes, BROADENING_TAG, n_candidates_scale=1, seed=BASE_SEED, options=options)

    # One work unit per archived entry table. Striding lets several copies of this script share
    # the arm, which is the only parallelism that helps: the manager process is the bottleneck --
    # everything after the cut, and all of deduplication, runs there while its workers idle -- so
    # more workers inside one run buys little and more runs side by side buys a lot.
    jobs = [(bundle, shard)
            for bundle, bundle_dir in bundle_directories(root).items()
            for shard in sorted(bundle_dir.glob('entries_*.parquet'))]
    jobs = jobs[args.shard_offset::args.shard_stride]
    print(f'{len(jobs)} shards for offset {args.shard_offset} of stride {args.shard_stride}',
          flush=True)

    started = time.time()
    summary = []
    try:
        for bundle, shard in jobs:
            tag = shard.stem.split('_', 1)[1]
            destination = out_root / f'predownsample_{tag}.parquet'
            if destination.exists():
                print(f'  {bundle}/{tag}: already done, skipping', flush=True)
                continue
            entries = pd.read_parquet(shard)
            if args.limit_entries:
                entries = entries.head(args.limit_entries)
            frames = []
            for _, entry in entries.iterrows():
                frames.append(run_entry(optimizers, task_queues, entry,
                                        args.n_top_candidates))
            frame = label(pd.concat(frames, ignore_index=True), entries)
            frame['condition_bundle'] = bundle
            frame['split'] = frame['entry_id'].map(entries.set_index('entry_id')['split'])
            frame.to_parquet(destination, index=False)
            mismatches = capture_gate(frame)
            summary.append({'bundle': bundle, 'shard': tag, 'entries': int(entries.shape[0]),
                            'rows': int(frame.shape[0]),
                            'correct_rows': int(frame['is_correct'].sum()),
                            'capture_mismatches': mismatches})
            print(f'  {bundle}/{tag}: {frame.shape[0]:,} rows, '
                  f'{int(frame["is_correct"].sum())} correct, '
                  f'{mismatches} capture mismatches ({mismatches / frame.shape[0]:.2%}), '
                  f'{time.time() - started:.0f}s', flush=True)
    finally:
        shutdown_mp_workers(processes, task_queues)

    # One manifest per striding copy, so concurrent runs cannot overwrite each other's record of
    # what they produced. A reader concatenates them.
    name = ('manifest.json' if args.shard_stride == 1
            else f'manifest_{args.shard_offset}of{args.shard_stride}.json')
    with open(out_root / name, 'w', encoding='utf-8') as handle:
        json.dump({'arm': args.arm, 'source': ARMS[args.arm],
                   'prune_threshold': 0.0, 'prune_criterion_capture': True,
                   'seeding': 'sha256(base_seed:entry_id:bravais_lattice), per PROTOCOL section 6',
                   'base_seed': BASE_SEED, 'processes': args.processes,
                   'broadening_tag': BROADENING_TAG,
                   'shard_stride': args.shard_stride, 'shard_offset': args.shard_offset,
                   'seconds': round(time.time() - started, 1),
                   'shards': summary}, handle, indent=2)
    rows = sum(record['rows'] for record in summary)
    print(f'\n{rows:,} rows over {len(summary)} shards in {time.time() - started:.0f}s')
    print(f'wrote {out_root}')


if __name__ == '__main__':
    main()
