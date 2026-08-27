"""S06 -- price DWMM's iteration lever before the benchmark is generated behind it.

DWMM's fourth compute lever, 2026-08-24: *"We can also run the indexer for half, or a quarter of
the iterations."* This measures what that costs, on the only terms that matter.

WHAT THE INNER LOOP IS. `opt_params['iteration_info']` is one deterministic pass plus a block of
random-subsampling passes, and the block is already tuned per lattice system -- cubic 5, the
tetragonal family 30, orthorhombic 50, monoclinic and triclinic 60. So "half the iterations" is a
scale factor on an existing schedule, not a flat number, and `MPIOptimizer._scaled_iterations` is
where it is applied.

WHY IT IS NOT A FREE SAVING. The random subsampling fits every iterate to a random subset of the
peaks, which is Shirley's "slightly different refinement conditions" implemented by accident, and
it is what makes the reproducibility floor measurable at all. Changing the schedule changes that
noise as well as the pool, so S08's floor must be measured under whatever this chooses. Worse, a
merit selected on a cheap pool is being chosen against a distribution users will never see -- so
the reduction is taken ONLY if the ceiling does not move, and if it moves the honest options are
to keep the full schedule or to change the shipped default too.

THE ARMS NEST, AND THAT IS THE DESIGN. With `search_seed_scheme='per_entry_bravais'` the search
RNG is re-keyed per (peak list, Bravais lattice, rank), so the quarter arm draws exactly the same
random subsets as the full arm's first k iterations. The three schedules are prefixes of one
another rather than three independent searches, which is what makes the comparison paired rather
than a comparison of two different runs (the trap F-137 records). It is asserted directly, on the
drawn peak subsets, in `tests/test_iteration_scale.py` -- not assumed here.

WHAT IS MEASURED, per (entry, Bravais lattice) and per arm:
  * the pool -- every survivor of the cut, and whether a correct cell is among them. This is the
    CEILING, and the handoff calls it the decisive metric;
  * the rank of the best correct candidate in the pooled, M20-sorted list `run.py` actually
    prints, which is what a merit then has to work with;
  * wall clock, so the saving is priced rather than assumed.

The pool sizes are a second deliverable in their own right: S06 sizes the negative subsampler's
K against real survivor counts at the generation cut, and this is the only place they exist.

    python mlindex/scripts/run_fom_iteration_pilot.py --n-entries-per-bl 30
    python mlindex/scripts/run_fom_iteration_pilot.py --stage report

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomConditions
from mlindex.model_training import FomPatterns
from mlindex.model_training.FomBenchmark import q2_digest
from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
from mlindex.scripts.run_fom_prune_rerun import BRAVAIS_LATTICES, TRUTH_SLICE

# The generation cut S03 settled (C2-Q-001): loss-free on both threshold-0 arms, and every higher
# cut reconstructable from it by restriction. The pilot has to see the pool the benchmark will
# actually carry, so it runs behind the same cut rather than behind production's 5.0.
PRUNE_THRESHOLD = 1.5

# One deterministic pass, then the random block scaled. 1.0 is the shipped schedule.
DEFAULT_SCALES = (1.0, 0.5, 0.25)

OUT_ROOT = os.path.join('mlindex', 'characterization', 'fom', 'iteration_pilot')
ARTIFACT_DIR = os.path.join('docs', 'fom_campaign2', 'artifacts')


def scale_tag(scale):
    """Filename-safe name for an arm. `1.0 -> full`, `0.5 -> half`, `0.25 -> quarter`."""
    named = {1.0: 'full', 0.5: 'half', 0.25: 'quarter'}
    return named.get(float(scale), f'scale{float(scale):g}'.replace('.', 'p'))


def best_correct_rank(M20, correct):
    """Where the best correct candidate sits in the list `run.py` prints, or -1 if absent.

    Lifted from `run_fom_prune_confirm.best_correct_rank`, and stable-sorted for the same reason:
    `run.py` sorts the pooled candidates by M20 descending and prints from the top, so ties must
    keep the order the lattices were assembled in. Returning the rank rather than a top-N flag
    means every top-N figure is a restriction of one stored number, so top-1, top-10 and top-20
    cannot disagree with each other.
    """
    correct = np.asarray(correct, dtype=bool)
    if not correct.any():
        return -1
    order = np.argsort(-np.asarray(M20, dtype=np.float64), kind='stable')
    return int(np.argmax(correct[order]))


def label_pool(unit_cell_pred, unit_cell_true, lattice_system):
    """`is_correct` over a whole pool, batched.

    The scalar labeller costs ~9 ms a candidate and this pilot produces millions of them; the
    batch routine does 57.4 M rows in 56 seconds with zero disagreements (F-166). `TRUTH_SLICE`
    is not a range -- monoclinic takes beta and not alpha -- so a contiguous slice would compare
    the wrong angle silently.
    """
    if unit_cell_pred.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return is_correct_known_bl_batch(
        np.asarray(unit_cell_true, dtype=np.float64)[TRUTH_SLICE[lattice_system]],
        np.asarray(unit_cell_pred, dtype=np.float64), lattice_system, rtol=0.01)


def optimizer_options(scale):
    """The opt_params one arm runs under.

    `search_seed_scheme` is what makes the arms paired -- and it is off by default in the shipped
    optimizer, because it changes which candidates the search generates (C2-F-042). `run_mp_bl`
    is therefore called with `run_seed=None`: the optimizer re-keys itself per pattern, and a
    second reseed from the driver would fight it.
    """
    return {
        'prune_m20_threshold': float(PRUNE_THRESHOLD),
        'dump_candidates': True,
        'search_seed_scheme': 'per_entry_bravais',
        'search_base_seed': 12345,
        'iteration_scale': float(scale),
        }


def sample_pilot_entries(bravais_lattices, n_entries_per_bl, seed):
    """The pilot population: lattice-stratified, and the same entries under every arm."""
    frames = [FomPatterns.sample_entries(bl, n_entries_per_bl, seed,
                                         columns=FomPatterns.DUMP_READ_COLUMNS)
              for bl in bravais_lattices]
    return pd.concat(frames, ignore_index=True)


def run_one_entry(entry, condition, optimizers, task_queues, bravais_lattices, seed):
    """One pattern through all fourteen lattices, under whatever arm the optimizers carry.

    Returns (per-lattice rows, entry row). The per-lattice rows carry the pool size, which is the
    input to S06's K sizing; the entry row carries the pooled rank, which is what `run.py` prints.
    """
    from mlindex.optimization.MPOptimizer import run_mp_bl

    hkl_full = np.stack([
        np.asarray(entry[f'reindexed_{axis}_{FomPatterns.BROADENING_TAG}'], dtype=float)
        for axis in ('h', 'k', 'l')], axis=1)
    pattern = FomPatterns.prepare_peak_list(entry, condition, seed, hkl=hkl_full,
                                            second_phase_pool=None)
    q2_obs = np.asarray(pattern.q2_obs, dtype=np.float64)
    unit_cell_true = np.asarray(entry['reindexed_unit_cell'], dtype=np.float64)
    true_lattice = entry['bravais_lattice']
    digest = q2_digest(q2_obs)

    rows = []
    pooled_M20, pooled_correct = [], []
    entry_seconds = 0.0
    for bravais_lattice in bravais_lattices:
        optimizer = optimizers[bravais_lattice]
        started = time.perf_counter()
        run_mp_bl(optimizer, bravais_lattice, task_queues, q2=q2_obs,
                  zero_error=False, wavelength=None, n_top=FomPatterns.N_TOP_CANDIDATES)
        seconds = time.perf_counter() - started
        entry_seconds += seconds
        records = optimizer.drain_candidate_dump()

        # The full survivor pool: the ceiling is a statement about this, not about the top 20.
        pool_size = int(sum(record['xnn'].shape[0] for record in records))
        pool_correct = np.zeros(0, dtype=bool)
        pool_M20 = np.zeros(0)
        if records and bravais_lattice == true_lattice:
            system = optimizer.lattice_system
            pool_correct = np.concatenate([
                label_pool(record['unit_cell'], unit_cell_true, system) for record in records])
            pool_M20 = np.concatenate([np.asarray(record['M20'], dtype=np.float64)
                                       for record in records])

        # And the printed list: each lattice's top 20, pooled and sorted by M20, as `run.py`
        # assembles it. Only the true lattice can hold a correct cell -- that is how every label
        # in this campaign is defined -- so the rest are False by construction, not by measurement.
        top_M20 = np.asarray(optimizer.top_M20, dtype=np.float64)
        pooled_M20.append(top_M20)
        if bravais_lattice == true_lattice:
            pooled_correct.append(label_pool(
                np.asarray(optimizer.top_unit_cell, dtype=np.float64), unit_cell_true,
                optimizer.lattice_system))
        else:
            pooled_correct.append(np.zeros(top_M20.size, dtype=bool))

        rows.append({
            'entry_id': entry['identifier'],
            'q2_digest': digest,
            'bravais_lattice': bravais_lattice,
            'bravais_lattice_true': true_lattice,
            'lattice_system': optimizer.lattice_system,
            'is_true_lattice': bravais_lattice == true_lattice,
            'volume_true': float(entry['reindexed_volume']),
            'pool_size': pool_size,
            'n_correct_pool': int(pool_correct.sum()),
            'best_correct_rank_in_lattice': best_correct_rank(pool_M20, pool_correct),
            'top20_size': int(top_M20.size),
            'seconds': seconds,
            })

    pooled_M20 = np.concatenate(pooled_M20)
    pooled_correct = np.concatenate(pooled_correct)
    reachable = any(row['n_correct_pool'] > 0 for row in rows)
    entry_row = {
        'entry_id': entry['identifier'],
        'q2_digest': digest,
        'bravais_lattice_true': true_lattice,
        'lattice_system_true': entry['lattice_system'],
        'volume_true': float(entry['reindexed_volume']),
        # The ceiling: a correct cell exists ANYWHERE in the pool. A generation outcome, and the
        # metric the iteration decision rests on.
        'reachable': bool(reachable),
        # The outcome: where it lands in the list a user reads.
        'pooled_rank': best_correct_rank(pooled_M20, pooled_correct),
        'pool_size_full': int(sum(row['pool_size'] for row in rows)),
        'seconds': entry_seconds,
        }
    return rows, entry_row


def run_arm(args, scale, entries, condition):
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, shutdown_mp_workers

    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    out_dir = Path(BASE) / args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = scale_tag(scale)
    lattice_path = out_dir / f'lattice_{tag}.parquet'
    entry_path = out_dir / f'entry_{tag}.parquet'
    if lattice_path.exists() and entry_path.exists() and not args.force:
        print(f'[{tag}] already present, skipping', flush=True)
        return

    optimizers, processes, task_queues = setup_mp_optimizers(
        args.processes, FomPatterns.BROADENING_TAG, n_candidates_scale=1,
        seed=args.seed, options=optimizer_options(scale))

    lattice_rows, entry_rows, failures = [], [], []
    started = time.time()
    try:
        for position in range(entries.shape[0]):
            entry = entries.iloc[position]
            try:
                rows, entry_row = run_one_entry(entry, condition, optimizers, task_queues,
                                                bravais_lattices, args.seed)
            except Exception as error:                       # noqa: BLE001 -- recorded, not raised
                failures.append({'identifier': entry['identifier'], 'detail': repr(error)})
                continue
            for row in rows:
                row['iteration_scale'] = float(scale)
            entry_row['iteration_scale'] = float(scale)
            lattice_rows += rows
            entry_rows.append(entry_row)
            if (position + 1) % args.flush_every == 0:
                # Abort-safe, for the reason campaign 1 added it: an abort near the end of a long
                # bundle loses the whole bundle otherwise.
                pd.DataFrame(lattice_rows).to_parquet(lattice_path, index=False)
                pd.DataFrame(entry_rows).to_parquet(entry_path, index=False)
                elapsed = time.time() - started
                print(f'[{tag}] {position + 1}/{entries.shape[0]} entries, '
                      f'{elapsed:.0f} s, {elapsed / (position + 1):.1f} s/entry', flush=True)
    finally:
        shutdown_mp_workers(processes, task_queues)
        if lattice_rows:
            pd.DataFrame(lattice_rows).to_parquet(lattice_path, index=False)
            pd.DataFrame(entry_rows).to_parquet(entry_path, index=False)
        if failures:
            with open(out_dir / f'failures_{tag}.json', 'w', encoding='utf-8') as handle:
                json.dump(failures, handle, indent=2, sort_keys=True)
    print(f'[{tag}] wrote {len(entry_rows)} entries, {len(failures)} failures, '
          f'{time.time() - started:.0f} s', flush=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage', choices=['run', 'report'], default='run')
    parser.add_argument('--scales', default=','.join(f'{s:g}' for s in DEFAULT_SCALES),
                        help='Schedule scale factors, one arm each. 1 is the shipped schedule')
    parser.add_argument('--n-entries-per-bl', type=int, default=30)
    parser.add_argument('--bravais-lattices', default=','.join(BRAVAIS_LATTICES))
    parser.add_argument('--condition', default='nominal',
                        help='Condition bundle key from FomConditions')
    parser.add_argument('--processes', type=int, default=8,
                        help='One manager plus processes-1 workers')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--flush-every', type=int, default=10)
    parser.add_argument('--limit-entries', type=int, default=None)
    parser.add_argument('--force', action='store_true', help='Re-run an arm already on disk')
    parser.add_argument('--out-root', default=OUT_ROOT)
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    scales = [float(value) for value in args.scales.split(',')]
    if args.stage == 'run':
        condition = FomConditions.BY_KEY[args.condition]
        bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
        entries = sample_pilot_entries(bravais_lattices, args.n_entries_per_bl, args.seed)
        if args.limit_entries:
            entries = entries.iloc[:args.limit_entries].reset_index(drop=True)
        print(f'{entries.shape[0]} entries, {len(scales)} arms, condition {condition.tag}',
              flush=True)
        for scale in scales:
            run_arm(args, scale, entries, condition)
        return
    from mlindex.scripts import run_fom_iteration_pilot_report as report
    report.report(args, scales)


if __name__ == '__main__':
    main()
