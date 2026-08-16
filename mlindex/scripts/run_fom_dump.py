"""S03/S04 -- generate the frozen candidate pool that the FOM work is developed against.

Runs the full 14-Bravais-lattice pipeline over synthetic entries under one condition bundle
and writes every *surviving* candidate, not the twenty per lattice that ranking keeps, with
the columns needed to recompute any figure of merit offline. Two tables per pool:

    candidates_<tag>.parquet   one row per candidate
    entries_<tag>.parquet      one row per pattern, with the ground truth
    manifest.json              what makes the pool regenerable

This is deliberately a separate entry point from `mlindex.command_line.run`. The dump is a
research instrument; the end-user indexer should not grow a flag for it.

The condition mechanisms, the entry sampling and the identifier-derived noise seed are all
`run_fom_mirror`'s, imported rather than copied, so a pool generated here sees exactly the
patterns S02's grid saw. The one thing this script must not do is re-derive the
train/dev/test split: it is frozen by entry in S02's manifest and is read from there.

    python run_fom_dump.py --error-multiplier 1 --n-contaminants 0 \
        --n-entries-per-bl 5 --n-pools 3 --pool-size 2 --out-dir <dir>

Reproducibility caveat inherited from the mirror: each optimizer's search RNG is seeded once
per pool and advances with every entry it runs, so a pool is reproducible only at a fixed
(n_pools, pool_size, shard, n_shards, seed). All five go in the manifest.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['ATLAS_NUM_THREADS'] = '1'
os.environ['SKLEARN_N_JOBS'] = '1'
import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

import mlindex
from mlindex.command_line.run import BRAVAIS_LATTICES
from mlindex.model_training import FomBenchmark
from mlindex.utilities.ErrorAdder import ContaminantPlacementError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fom_mirror as mirror


# Ground truth the mirror does not read but the benchmark schema requires: the metric tensor
# of the true cell, and which reflection produced each observed peak.
DUMP_READ_COLUMNS = list(mirror.READ_COLUMNS) + [
    'reindexed_xnn',
    'reindexed_extinction_group',
    f'reindexed_h_{mirror.BROADENING_TAG}',
    f'reindexed_k_{mirror.BROADENING_TAG}',
    f'reindexed_l_{mirror.BROADENING_TAG}',
    ]

MANIFEST_PATH = (Path(mlindex.__path__[0]).parent / 'docs' / 'fom' / 'artifacts'
                 / 'S02_mirror_manifest.parquet')


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the FOM benchmark candidate pool (S03/S04)')
    parser.add_argument('--error-multiplier', type=float, required=True,
                        help='Multiplier on the sigma(q2) model; float, so 1.5 is expressible')
    parser.add_argument('--n-contaminants', type=int, required=True,
                        help='Contaminant peaks injected per pattern')
    parser.add_argument('--contaminant-bias', type=float, default=1.0,
                        help='Bias contaminant positions towards low q2; 1 = uniform draw')
    parser.add_argument('--n-dropout', type=int, default=0,
                        help='Interior peak dropout: peaks deleted from the low-q2 window '
                             'and backfilled from higher q2')
    parser.add_argument('--second-phase-lines', type=int, default=0,
                        help='Lines injected from a second crystalline phase drawn at random '
                             'from the database. Correlated, unlike --n-contaminants, which is '
                             'what makes them hard to reject (bundle C6)')
    parser.add_argument('--second-phase-bias', type=float, default=2.0,
                        help='Weight the choice of the partner\'s lines towards low q2, where a '
                             'second phase\'s strong reflections sit. 1 picks uniformly')
    parser.add_argument('--n-entries-per-bl', type=int, default=500,
                        help='Entries sampled per Bravais lattice, capped by availability')
    parser.add_argument('--bravais-lattices', type=str, default=','.join(BRAVAIS_LATTICES),
                        help='True Bravais lattices to draw entries from')
    parser.add_argument('--n-pools', type=int, default=1,
                        help='Independent optimizer pools run concurrently, each with its '
                             'own manager. Only managers load models (~3 GB each), so this '
                             'is the memory-limited axis')
    parser.add_argument('--pool-size', type=int, default=1,
                        help='Processes per pool: one manager plus pool_size-1 workers')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12345,
                        help='Base seed for entry sampling, per-entry noise and the optimizers')
    parser.add_argument('--split-manifest', type=str, default=str(MANIFEST_PATH),
                        help='Frozen S02 manifest supplying the fom-train/dev/test split. '
                             'The split is by source entry and must never be re-derived here')
    parser.add_argument('--allow-unassigned-split', action='store_true',
                        help='Proceed when sampled entries are absent from the frozen manifest. '
                             'Only for deliberately reduced runs: at the production settings the '
                             'sampling reproduces the manifest exactly, so a shortfall otherwise '
                             'means a parameter has drifted')
    parser.add_argument('--out-dir', type=str, required=True)
    return parser.parse_args()


def load_split(manifest_path):
    """entry identifier -> fom-train / fom-dev / fom-test, from S02's frozen manifest.

    PROTOCOL 3 rule 5: splits are by source entry, never by candidate, and the same crystal
    must never appear in two splits under different noise. Re-deriving the split here would
    silently break that across condition bundles.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise SystemExit(
            f'Frozen split manifest not found at {path}. It is S02 output and is required: '
            'the split must not be re-derived. Copy it across or pass --split-manifest.'
            )
    manifest = pd.read_parquet(path)
    id_column = 'identifier' if 'identifier' in manifest.columns else 'entry_id'
    split_column = 'split' if 'split' in manifest.columns else 'fom_split'
    if split_column not in manifest.columns:
        raise SystemExit(
            f'{path} has no split column; found {sorted(manifest.columns)}. '
            'S04 must consume S02\'s split rather than inventing one.'
            )
    return dict(zip(manifest[id_column], manifest[split_column]))


def entry_record(entry, args, q2_obs, hkl_true, n_dropout_achieved, split, partner_id):
    return {
        'entry_id': entry['identifier'],
        'q2_digest': FomBenchmark.q2_digest(q2_obs),
        'source_db': entry['database'],
        'split': split,
        'condition_bundle': mirror.bundle_tag(args),
        'q2_obs': np.asarray(q2_obs, dtype=np.float64),
        'n_peaks_available': int(np.count_nonzero(
            np.asarray(entry[f'q2_{mirror.BROADENING_TAG}'], dtype=float) > 0)),
        'q2_error_multiplier': float(args.error_multiplier),
        'n_contaminants': int(args.n_contaminants),
        'contaminant_bias': float(args.contaminant_bias),
        'n_dropout': int(args.n_dropout),
        'n_dropout_achieved': int(n_dropout_achieved),
        'second_phase_lines': int(args.second_phase_lines),
        'second_phase_bias': float(args.second_phase_bias),
        # Which entry contaminated this one. Null off C6, and needed there to reproduce the
        # pattern and to ask whether a particular partner lattice is what defeats the FOM.
        'second_phase_partner': partner_id,
        'xnn_true': np.asarray(entry['reindexed_xnn'], dtype=np.float64),
        'unit_cell_true': np.asarray(entry['reindexed_unit_cell'], dtype=np.float64),
        'volume_true': float(entry['reindexed_volume']),
        'bravais_lattice_true': entry['bravais_lattice'],
        'lattice_system_true': entry['lattice_system'],
        'spacegroup_true': entry['reindexed_spacegroup_symbol_hm'],
        'extinction_group_true': entry['reindexed_extinction_group'],
        'hkl_true': np.asarray(hkl_true, dtype=np.int16).reshape(-1),
        }


def _pool_output_is_complete(out_dir, pool_tag):
    """Both of a pool's tables present and readable, so a resumed task can skip it."""
    paths = [Path(out_dir) / f'entries_{pool_tag}.parquet',
             Path(out_dir) / f'candidates_{pool_tag}.parquet']
    if not all(path.exists() for path in paths):
        return False
    try:
        for path in paths:
            pd.read_parquet(path, columns=['entry_id'])
    except Exception:
        # Truncated by a kill mid-write. Regenerate rather than resume onto a corrupt shard.
        return False
    return True


def run_pool(pool_index, args, entries, split_by_id, out_dir, shard_tag,
             second_phase_pool=None):
    """One manager plus pool_size-1 workers, working through this pool's stripe of entries.

    Runs in its own process: the queue injection in setup_mp_optimizers is class-level and
    documented as non-reentrant, so concurrent pools cannot share an interpreter.
    """
    from mlindex.optimization.MPOptimizer import run_mp_bl
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers
    from mlindex.optimization.MPOptimizer import shutdown_mp_workers

    pool_tag = f'{shard_tag}_pool{pool_index:02d}'
    # Resume: a requeued array task must not redo pools that already finished. Both tables have to
    # be present and readable -- a pool killed mid-write can leave the entry table complete and the
    # candidate shard truncated, and silently skipping that would lose its candidates for good.
    if _pool_output_is_complete(out_dir, pool_tag):
        print(f'[pool {pool_index:02d}] already written, skipping', flush=True)
        return

    optimizers, processes, task_queues = setup_mp_optimizers(
        args.pool_size, mirror.BROADENING_TAG, n_candidates_scale=1,
        seed=args.seed + pool_index,
        )
    for bravais_lattice in BRAVAIS_LATTICES:
        optimizers[bravais_lattice].opt_params['dump_candidates'] = True

    entry_rows = []
    candidate_frames = []
    failures = []
    consecutive_failures = 0
    started = time.time()
    try:
        for entry_index in range(entries.shape[0]):
            entry = entries.iloc[entry_index]
            hkl_full = np.stack([
                np.asarray(entry[f'reindexed_{axis}_{mirror.BROADENING_TAG}'], dtype=float)
                for axis in ('h', 'k', 'l')
                ], axis=1)
            try:
                q2_obs, n_dropout_achieved, hkl_true, partner_id = mirror.prepare_peak_list(
                    entry, args, args.seed, hkl=hkl_full,
                    second_phase_pool=second_phase_pool)
            except ContaminantPlacementError as error:
                failures.append({'identifier': entry['identifier'],
                                 'reason': 'contaminant_placement', 'detail': str(error)})
                continue

            digest = FomBenchmark.q2_digest(q2_obs)
            context = {
                'entry_id': entry['identifier'],
                'q2_digest': digest,
                }
            records = []
            try:
                for bravais_lattice in BRAVAIS_LATTICES:
                    optimizer = optimizers[bravais_lattice]
                    optimizer.dump_context = context
                    run_mp_bl(optimizer, bravais_lattice, task_queues, q2=q2_obs,
                              zero_error=False, wavelength=None,
                              n_top=mirror.N_TOP_CANDIDATES)
                    records += optimizer.drain_candidate_dump()
            except Exception as error:
                failures.append({'identifier': entry['identifier'],
                                 'reason': 'optimizer', 'detail': repr(error)})
                consecutive_failures += 1
                if consecutive_failures >= mirror.MAX_CONSECUTIVE_FAILURES:
                    raise
                continue
            consecutive_failures = 0

            candidate_frames.append(FomBenchmark.records_to_frame(records))
            entry_rows.append(entry_record(
                entry, args, q2_obs, hkl_true, n_dropout_achieved,
                split_by_id.get(entry['identifier'], 'unassigned'), partner_id,
                ))

            if (entry_index + 1) % 25 == 0:
                elapsed = time.time() - started
                print(f'[pool {pool_index:02d}] {entry_index + 1}/{entries.shape[0]} entries, '
                      f'{elapsed / (entry_index + 1):.1f} s/entry', flush=True)
    except Exception as error:
        # Write what this pool has rather than losing hours of it. The mirror already does this
        # (run_fom_mirror.run_pool); the dump did not, so an abort near the end of a 2.5 h bundle
        # discarded every row the pool had accumulated. Re-raised after the tables are on disk so
        # the process still exits non-zero and main() reports the failure.
        aborted = error
    else:
        aborted = None
    finally:
        shutdown_mp_workers(processes, task_queues)

    # Failures were counted and then thrown away, so an unattended run left no way to tell a
    # handful of unplaceable contaminants from a systematic problem. On the first gate run that
    # hid four entries lost to one degenerate branch of add_second_phase.
    if failures:
        with open(Path(out_dir) / f'failures_{pool_tag}.json', 'w',
                  encoding='utf-8') as failure_file:
            json.dump(failures, failure_file, indent=2, sort_keys=True)
        reasons = {}
        for failure in failures:
            reasons[failure['reason']] = reasons.get(failure['reason'], 0) + 1
        print(f'[pool {pool_index:02d}] failures by reason: {reasons}', flush=True)

    if entry_rows:
        FomBenchmark.write_entry_table(
            pd.DataFrame(entry_rows, columns=list(FomBenchmark.ENTRY_COLUMNS)),
            out_dir, pool_tag)
    if candidate_frames:
        FomBenchmark.write_candidate_shard(
            pd.concat(candidate_frames, ignore_index=True), out_dir, pool_tag)
    elapsed = time.time() - started
    print(f'[pool {pool_index:02d}] {"ABORTED" if aborted else "done"}: '
          f'{len(entry_rows)}/{entries.shape[0]} entries, '
          f'{elapsed / max(1, len(entry_rows)):.1f} s/entry, {len(failures)} failures',
          flush=True)
    if aborted is not None:
        raise aborted


def _commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(mlindex.__path__[0]).parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    args = _parse_args()
    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    invalid = [bl for bl in bravais_lattices if bl not in BRAVAIS_LATTICES]
    if invalid:
        raise SystemExit(f"Unknown Bravais lattices: {', '.join(invalid)}")

    split_by_id = load_split(args.split_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = mirror.bundle_tag(args)
    shard_tag = f'{tag}_shard{args.shard:02d}of{args.n_shards:02d}'

    entries = []
    for bravais_lattice in bravais_lattices:
        sampled = mirror.sample_entries(bravais_lattice, args.n_entries_per_bl, args.seed,
                                        columns=DUMP_READ_COLUMNS)
        entries.append(sampled)
        print(f'{bravais_lattice}: sampled {sampled.shape[0]} of '
              f'{args.n_entries_per_bl} requested', flush=True)
    entries = pd.concat(entries, ignore_index=True)

    # Pre-flight, because the alternative is discovering it after the bundle has run: an entry with
    # no split is unusable downstream, and the usual cause is a sampling parameter that no longer
    # reproduces the frozen set. Verified for the production settings -- at --n-entries-per-bl 500
    # and --seed 12345 the sampling reproduces all 5 955 manifest entries exactly, per lattice --
    # so any shortfall here means a parameter has drifted, not that the manifest is incomplete.
    # A reduced-size smoke run legitimately samples outside it and passes --allow-unassigned-split.
    unassigned = [identifier for identifier in entries['identifier']
                  if identifier not in split_by_id]
    if unassigned:
        message = (f'{len(unassigned)} of {entries.shape[0]} sampled entries are absent from '
                   f'{args.split_manifest}, e.g. {sorted(unassigned)[:3]}. The split is frozen by '
                   'S02 and must not be re-derived, so these entries would carry no split.')
        if not args.allow_unassigned_split:
            raise SystemExit(
                f'{message}\nCheck --n-entries-per-bl and --seed match the manifest, or pass '
                '--allow-unassigned-split for a deliberately reduced run.'
                )
        print(f'WARNING: {message}', flush=True)

    entries = entries.iloc[args.shard::args.n_shards].reset_index(drop=True)
    print(f'shard {args.shard}/{args.n_shards}: {entries.shape[0]} entries, bundle {tag}; '
          f'{args.n_pools} pools x {args.pool_size} processes', flush=True)

    # Built before striping, so every pool and every shard draws partners from the same set.
    second_phase_pool = (mirror.build_second_phase_pool(entries)
                         if args.second_phase_lines > 0 else None)

    started = time.time()
    if args.n_pools == 1:
        run_pool(0, args, entries, split_by_id, out_dir, shard_tag, second_phase_pool)
    else:
        processes = []
        for pool_index in range(args.n_pools):
            stripe = entries.iloc[pool_index::args.n_pools].reset_index(drop=True)
            process = Process(target=run_pool,
                              args=(pool_index, args, stripe, split_by_id, out_dir, shard_tag,
                                    second_phase_pool))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [p.exitcode for p in processes if p.exitcode]
        if failed:
            raise SystemExit(f'{len(failed)} pools exited non-zero: {failed}')

    FomBenchmark.write_manifest(
        out_dir,
        bundle=tag,
        shard=args.shard,
        n_shards=args.n_shards,
        n_pools=args.n_pools,
        pool_size=args.pool_size,
        seed=args.seed,
        broadening_tag=mirror.BROADENING_TAG,
        n_top_candidates=mirror.N_TOP_CANDIDATES,
        n_entries_per_bl=args.n_entries_per_bl,
        bravais_lattices=bravais_lattices,
        error_multiplier=args.error_multiplier,
        n_contaminants=args.n_contaminants,
        contaminant_bias=args.contaminant_bias,
        n_dropout=args.n_dropout,
        second_phase_lines=args.second_phase_lines,
        second_phase_bias=args.second_phase_bias,
        prune_threshold=5.0,
        split_manifest=str(args.split_manifest),
        commit=_commit_hash(),
        seconds_total=round(time.time() - started, 1),
        )
    print(f'wrote {out_dir}', flush=True)


if __name__ == '__main__':
    main()
