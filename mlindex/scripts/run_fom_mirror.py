"""S02 -- the synthetic mirror of the CNRS gap.

Runs the full 14-Bravais-lattice pipeline over synthetic validation entries under one condition
bundle, pools the candidates across lattices exactly as `mlindex.command_line.run` does, and
records for every entry where the correct cell landed in the pooled ranking and what score it
carried. The two headline statistics follow from those two numbers:

    operating point  -- correct cell in the pooled top 10 and M20 > 10 (the paper's criterion)
    ceiling          -- correct cell anywhere in the pooled output

and so does the rank/threshold decomposition of the entries that are lost.

Scoring the top 20 of the *true* lattice instead of the pooled top 10 is a different and much
easier problem; that is what `run_success_rate.py` measures and it is not what the paper reports.

The peak-list noise is seeded from the entry identifier, so a given entry receives the same
perturbation under every condition bundle and paired comparisons across bundles stay valid.

Parallelism is two-level, because one big pool scales badly. Inside a pool only the optimisation
iterations distribute -- candidate generation (the ML models) runs on the manager -- so the serial
fraction puts a low ceiling on the speedup of a single wide pool. Instead:

    --n-pools 16 --pool-size 8    ->  16 independent managers, 7 workers each, 128 processes

Each pool is a separate process that builds its own optimizer set and works through its own stripe
of entries, so the 16 managers generate candidates concurrently. Memory is not the constraint: only
managers load models (OptimizerWorker takes no data_params), so this costs ~16 x 3 GB rather than
128 x 3 GB. --n-shards spreads a bundle across nodes on top of that.

    python run_fom_mirror.py --n-pools 16 --pool-size 8 --error-multiplier 1 --n-contaminants 1 \
        --n-entries-per-bl 500 --shard 0 --n-shards 4

One caveat on reproducibility. The peak-list noise is identifier-derived and therefore invariant,
but each optimizer's *search* RNG is seeded once per pool and advances with every entry it runs, so
an entry's candidate set depends on its position in its pool's queue. Results are reproducible for
a fixed (--n-pools, --pool-size, --shard, --n-shards, --seed) and those are recorded in the summary
JSON; they are not invariant to re-pooling. That is pre-existing behaviour -- `run.py` is
order-dependent in the same way -- and it does not bias any statistic, since entries are
independent.

Aggregation, CNRS weighting and the frozen manifest are in run_fom_mirror_analysis.py.
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
import hashlib
import json
import time
import types
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

import mlindex
from mlindex.command_line.run import BRAVAIS_LATTICES
from mlindex.command_line.run import _collect_results
from mlindex.optimization.CandidateValidation import label_candidates
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import select_peaks_with_dropout


BROADENING_TAG = '1'
N_TOP_CANDIDATES = 20
N_PEAKS = 20
# Cubic models take 10 peaks and everything else 20 (OptimizerManager truncates internally), but
# every entry must supply 20 so that all 14 lattices see the same list. The filter costs almost
# nothing even for cubic: cF keeps 106 of 136 validation entries, cP 321 of 344.

# The headline tolerance is the 1e-2 the handoff mandates; the draft quotes 2%, so both are
# recorded and the comparison against Table 1 never rests on an undocumented mismatch.
RTOL_HEADLINE = 1e-2
RTOL_ALTERNATE = 2e-2

# A contaminant draw is rejected unless it clears every peak's half breadth, and the whole set is
# redrawn on any rejection, so acceptance decays exponentially in the contaminant count. Uncapped
# this can spin forever on a crowded pattern; the cap turns that into a recorded skip.
CONTAMINANT_MAX_ATTEMPTS = 2000

# Give up on a shard rather than write a mostly-failed one, but tolerate isolated entry failures.
MAX_CONSECUTIVE_FAILURES = 10

DATASET_DIRECTORY = Path(mlindex.__path__[0]) / 'data' / 'generated_datasets'

READ_COLUMNS = [
    'identifier',
    'database',
    'bravais_lattice',
    'lattice_system',
    'train',
    'reindexed_unit_cell',
    'reindexed_volume',
    'reindexed_spacegroup_symbol_hm',
    f'q2_{BROADENING_TAG}',
    ]


def _parse_args():
    parser = argparse.ArgumentParser(description='S02 synthetic mirror: pooled 14-lattice run')
    parser.add_argument('--n-pools', type=int, default=1,
                        help='Independent optimizer pools run concurrently, each with its own '
                             'manager. Entries are striped across pools. Total processes is '
                             'n_pools * pool_size')
    parser.add_argument('--pool-size', type=int, default=1,
                        help='Processes per pool: one manager plus pool_size-1 workers. '
                             '1 runs each pool fully serially with no children')
    parser.add_argument('--error-multiplier', type=float, required=True,
                        help='Multiplier on the sigma(q2) model; float, so 1.5 is expressible')
    parser.add_argument('--n-contaminants', type=int, required=True,
                        help='Contaminant peaks injected per pattern')
    parser.add_argument('--contaminant-bias', type=float, default=1.0,
                        help='Bias contaminant positions towards low q2: q2 = lo + (hi-lo)*u**bias. '
                             '1 = the original uniform draw, 2 = uniform in q. Fraction landing '
                             'within the first five real peaks, where the generators take their '
                             'information: 35%% at 1, 61%% at 2, 72%% at 3')
    parser.add_argument('--n-dropout', type=int, default=0,
                        help='Interior peak dropout: delete this many peaks from within the '
                             'low-q2 window and backfill from higher q2, as undetected weak '
                             'reflections do. Attacks candidate discovery, not just the fit')
    parser.add_argument('--n-entries-per-bl', type=int, default=500,
                        help='Entries sampled per Bravais lattice, capped by availability')
    parser.add_argument('--bravais-lattices', type=str, default=','.join(BRAVAIS_LATTICES),
                        help='True Bravais lattices to draw entries from')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12345,
                        help='Base seed for entry sampling, per-entry noise and the optimizers')
    parser.add_argument('--conventional-cell', action='store_true',
                        help='Apply run.py\'s cctbx symmetry promotion and cross-lattice '
                             'deduplication before ranking. Added to run.py after the published '
                             '450/599 was measured, so it is off by default; see the sensitivity '
                             'recorded by the Stage 0 calibration')
    parser.add_argument('--no-candidate-dump', action='store_true',
                        help='Skip the per-candidate table and keep only the per-entry summary')
    parser.add_argument('--out-dir', type=str,
                        default=str(Path(mlindex.__path__[0]) / 'characterization' / 'fom' / 'mirror'))
    return parser.parse_args()


def bundle_tag(args):
    # Every non-default axis has to appear, or two bundles collide on their output filenames and
    # silently overwrite each other. Defaults are omitted so the 21 bundles already on disk keep
    # their names and stay comparable.
    tag = f'error{args.error_multiplier:g}_cont{args.n_contaminants:d}'
    if args.contaminant_bias != 1.0:
        tag += f'_bias{args.contaminant_bias:g}'
    if args.n_dropout:
        tag += f'_drop{args.n_dropout:d}'
    return tag


def derived_seed(key, base_seed):
    # Stable across processes and runs, unlike hash(). PROTOCOL 6 wants the per-entry seed derived
    # from the entry id so the same entry gets the same noise in every condition bundle.
    digest = hashlib.sha256(f'{base_seed}:{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def sample_entries(bravais_lattice, n_entries, base_seed, columns=None):
    # Uniform within the lattice, so the per-lattice success rate keeps the natural volume
    # distribution and stays comparable with draft Table 1. Volume stratification belongs to the
    # train/dev/test split and to reporting, not to the sampling.
    #
    # columns widens the read for callers needing ground truth the mirror does not use (the FOM
    # benchmark dump wants reindexed_xnn and the per-peak Miller indices). The sampling itself
    # depends only on the seed and the entry count, so a widened read selects the same entries.
    data = pd.read_parquet(DATASET_DIRECTORY / f'dataset_{bravais_lattice}.parquet',
                           columns=list(columns) if columns is not None else READ_COLUMNS)
    data = data.loc[~data['train']]
    peaks = data[f'q2_{BROADENING_TAG}']
    data = data.loc[peaks.apply(lambda q2: np.count_nonzero(q2) >= N_PEAKS)]
    data = data.sort_values('identifier', kind='stable', ignore_index=True)
    if data.shape[0] > n_entries:
        rng = np.random.default_rng(derived_seed(f'sample:{bravais_lattice}', base_seed))
        selected = np.sort(rng.choice(data.shape[0], size=n_entries, replace=False))
        data = data.iloc[selected].reset_index(drop=True)
    return data


def prepare_peak_list(entry, args, base_seed, hkl=None):
    # Applied in the order a real pattern acquires them: some reflections are never detected, the
    # instrument has a calibration offset and a random error, and the contaminant lines are then
    # placed relative to the peaks as observed.
    #
    # hkl is the ground-truth assignment for q2_full, supplied only by callers that need to know
    # which observed peak came from which reflection -- the FOM benchmark dump does, the mirror
    # does not. Passing it costs one extra return value and nothing else: add_q2_error and
    # add_contaminants already carry an hkl argument through their sort and insertion, and
    # contaminants arrive as (0, 0, 0) rows, which is the convention the benchmark schema wants.
    q2_full = np.asarray(entry[f'q2_{BROADENING_TAG}'], dtype=float)
    rng = np.random.default_rng(derived_seed(f'noise:{entry["identifier"]}', base_seed))

    q2 = select_peaks_with_dropout(q2_full, N_PEAKS, args.n_dropout, rng)
    # A thin entry cannot give up every requested peak, so the achieved dropout varies and is
    # recorded per entry rather than assumed from the parameter. Compared before any noise is
    # applied, so the values are still exact.
    nominal = q2_full[q2_full > 0][:N_PEAKS]
    n_dropout_achieved = int(np.sum(~np.isin(nominal, q2)))

    if hkl is not None:
        # select_peaks_with_dropout drops the zero padding and then selects by value, so the
        # surviving rows are recovered by looking the values up in the same filtered list. Peak
        # lists are stored in ascending q2, which searchsorted requires.
        positive = q2_full > 0
        q2_positive = q2_full[positive]
        hkl_positive = np.asarray(hkl, dtype=float)[positive]
        kept = np.searchsorted(q2_positive, q2)
        hkl = hkl_positive[kept][np.newaxis].copy()

    # add_q2_error perturbs in place, so hand it a copy and keep the unperturbed list intact.
    q2 = q2[np.newaxis].copy()
    if args.error_multiplier > 0:
        if hkl is None:
            q2 = add_q2_error(q2, None, args.error_multiplier, rng)
        else:
            q2, hkl = add_q2_error(q2, hkl, args.error_multiplier, rng)
    if args.n_contaminants > 0:
        if hkl is None:
            q2 = add_contaminants(q2, None, args.n_contaminants, rng,
                                  max_attempts=CONTAMINANT_MAX_ATTEMPTS,
                                  low_angle_bias=args.contaminant_bias)
        else:
            q2, hkl = add_contaminants(q2, hkl, args.n_contaminants, rng,
                                       max_attempts=CONTAMINANT_MAX_ATTEMPTS,
                                       low_angle_bias=args.contaminant_bias)
    if hkl is None:
        return q2[0], n_dropout_achieved
    return q2[0], n_dropout_achieved, hkl[0]


def pool_candidates(top_unit_cell, top_M20, top_Minfo, top_spacegroup, top_n_indexed,
                    labels, bravais_lattices, conventional_cell):
    # Expansion to full unit cells, the volume, and the M20-descending pool are all run.py's, so
    # the mirror ranks candidates the way the indexer does rather than by a parallel convention.
    rows = []
    for bravais_lattice in bravais_lattices:
        start = len(rows)
        optimizer_view = types.SimpleNamespace(
            top_unit_cell=top_unit_cell[bravais_lattice],
            top_M20=top_M20[bravais_lattice],
            top_n_indexed=top_n_indexed[bravais_lattice],
            )
        _collect_results(
            optimizer_view, bravais_lattice, rows,
            Minfos=top_Minfo[bravais_lattice],
            spacegroups=top_spacegroup[bravais_lattice],
            )
        for candidate_index, row in enumerate(rows[start:]):
            row['source_bravais_lattice'] = bravais_lattice
            row['candidate_index'] = candidate_index
            for rtol, correct, off_by_two in labels:
                suffix = '' if rtol == RTOL_HEADLINE else f'_rtol{rtol:g}'
                row[f'correct{suffix}'] = bool(correct[bravais_lattice][candidate_index])
                row[f'off_by_two{suffix}'] = bool(off_by_two[bravais_lattice][candidate_index])

    n_before = len(rows)
    n_correct_before = sum(row['correct'] for row in rows)
    if conventional_cell:
        # Promotion rewrites bravais_lattice and the cell but not the physical lattice, so the
        # labels computed above travel with the row. Deduplication can drop a labelled row in
        # favour of a higher-M20 one within 0.5%, which is inside the labelling tolerance, so a
        # correct candidate should never be lost -- the count is carried out so that assumption
        # is measured rather than assumed.
        from mlindex.command_line.run import _conventional_cell
        rows = _conventional_cell(rows)

    if not rows:
        # A lattice returning a single candidate is normal (cF and cI often do), so an entry whose
        # every lattice comes back empty is possible. Give it the expected columns rather than
        # letting an empty frame raise KeyError and take the whole shard down.
        columns = ['M20', 'Minfo', 'n_indexed', 'bravais_lattice', 'spacegroup', 'volume',
                   'a', 'b', 'c', 'alpha', 'beta', 'gamma',
                   'source_bravais_lattice', 'candidate_index']
        for rtol in (RTOL_HEADLINE, RTOL_ALTERNATE):
            suffix = '' if rtol == RTOL_HEADLINE else f'_rtol{rtol:g}'
            columns += [f'correct{suffix}', f'off_by_two{suffix}']
        return pd.DataFrame(columns=columns), n_before, n_correct_before

    pool = pd.DataFrame(rows).sort_values('M20', ascending=False, ignore_index=True)
    return pool, n_before, n_correct_before


def summarize_entry(entry, pool, q2_obs):
    summary = {
        'identifier': entry['identifier'],
        'database': entry['database'],
        'bravais_lattice_true': entry['bravais_lattice'],
        'lattice_system_true': entry['lattice_system'],
        'spacegroup_true': entry['reindexed_spacegroup_symbol_hm'],
        'volume_true': float(entry['reindexed_volume']),
        'n_peaks': int(q2_obs.size),
        'q2_20': float(q2_obs[-1]),
        'n_pooled': int(pool.shape[0]),
        'M20_top1': float(pool['M20'].iloc[0]) if pool.shape[0] else np.nan,
        }
    for rtol in (RTOL_HEADLINE, RTOL_ALTERNATE):
        suffix = '' if rtol == RTOL_HEADLINE else f'_rtol{rtol:g}'
        correct = pool[f'correct{suffix}'].to_numpy().astype(bool)
        same_bl = pool['source_bravais_lattice'].to_numpy() == entry['bravais_lattice']
        correct_indices = np.flatnonzero(correct)
        rank = int(correct_indices[0]) if correct_indices.size else -1
        summary[f'rank_best_correct{suffix}'] = rank
        summary[f'M20_best_correct{suffix}'] = (
            float(pool['M20'].iloc[rank]) if rank >= 0 else np.nan
            )
        summary[f'bl_of_best_correct{suffix}'] = (
            pool['source_bravais_lattice'].iloc[rank] if rank >= 0 else ''
            )
        # The paper counts a match regardless of the lattice label, so correct_any_bl is the
        # headline and correct_same_bl is the stricter reading.
        summary[f'correct_any_bl{suffix}'] = bool(correct.any())
        summary[f'correct_same_bl{suffix}'] = bool((correct & same_bl).any())
        summary[f'off_by_two_any{suffix}'] = bool(
            pool[f'off_by_two{suffix}'].to_numpy().astype(bool).any()
            )
    return summary


def run_pool(pool_index, args, entries, out_dir, shard_tag):
    # Runs in its own process. Builds a manager plus pool_size-1 workers, works through this pool's
    # stripe of entries, and writes its own shard files. The queue injection in
    # setup_mp_optimizers is class-level and documented as non-reentrant, which is exactly why each
    # pool must be a separate process rather than another set of queues in this one.
    from mlindex.optimization.MPOptimizer import run_mp_bl
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers
    from mlindex.optimization.MPOptimizer import shutdown_mp_workers

    pool_tag = f'{shard_tag}_pool{pool_index:02d}'
    # Vary the optimizer seed per pool so concurrent pools do not walk identical search sequences.
    optimizers, processes, task_queues = setup_mp_optimizers(
        args.pool_size, BROADENING_TAG, n_candidates_scale=1, seed=args.seed + pool_index
        )

    summaries = []
    candidate_frames = []
    failures = []
    consecutive_failures = 0
    correct_rows_collapsed = 0
    entries_losing_correct = 0
    aborted = None
    started = time.time()
    try:
        for entry_index in range(entries.shape[0]):
            entry = entries.iloc[entry_index]
            entry_started = time.time()
            try:
                q2_obs, n_dropout_achieved = prepare_peak_list(entry, args, args.seed)
            except ContaminantPlacementError as error:
                failures.append({'identifier': entry['identifier'],
                                 'reason': 'contaminant_placement', 'detail': str(error)})
                continue

            try:
                top_unit_cell = {}
                top_M20 = {}
                top_Minfo = {}
                top_spacegroup = {}
                top_n_indexed = {}
                for bravais_lattice in BRAVAIS_LATTICES:
                    run_mp_bl(optimizers[bravais_lattice], bravais_lattice, task_queues,
                              q2=q2_obs, zero_error=False, wavelength=None,
                              n_top=N_TOP_CANDIDATES)
                    optimizer = optimizers[bravais_lattice]
                    # run_mp_bl overwrites these on the next lattice, so snapshot now.
                    top_unit_cell[bravais_lattice] = optimizer.top_unit_cell.copy()
                    top_M20[bravais_lattice] = optimizer.top_M20.copy()
                    top_Minfo[bravais_lattice] = optimizer.top_Minfo.copy()
                    top_n_indexed[bravais_lattice] = optimizer.top_n_indexed.copy()
                    top_spacegroup[bravais_lattice] = list(optimizer.top_spacegroup)
            except Exception as error:
                failures.append({'identifier': entry['identifier'],
                                 'reason': 'optimizer', 'detail': repr(error)})
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise
                continue
            consecutive_failures = 0

            labels = [
                (rtol,) + label_candidates(entry, top_unit_cell, rtol=rtol)
                for rtol in (RTOL_HEADLINE, RTOL_ALTERNATE)
                ]
            pool, n_before, n_correct_before = pool_candidates(
                top_unit_cell, top_M20, top_Minfo, top_spacegroup, top_n_indexed,
                labels, BRAVAIS_LATTICES, args.conventional_cell,
                )
            if args.conventional_cell:
                # Two different questions, and the row count answers the uninteresting one.
                # Promotion collapses duplicate correct candidates, so rows disappearing is normal
                # and expected; what would actually matter is an entry going from having a correct
                # candidate to having none. Count both, and never quote the row count as a loss.
                n_correct_after = int(pool['correct'].to_numpy().astype(bool).sum())
                correct_rows_collapsed += max(0, n_correct_before - n_correct_after)
                if n_correct_before > 0 and n_correct_after == 0:
                    entries_losing_correct += 1

            summary = summarize_entry(entry, pool, q2_obs)
            summary['n_pooled_before_dedup'] = int(n_before)
            summary['n_dropout_achieved'] = n_dropout_achieved
            summary['seconds'] = time.time() - entry_started
            summaries.append(summary)

            if not args.no_candidate_dump:
                pool = pool.assign(identifier=entry['identifier'], pooled_rank=np.arange(pool.shape[0]))
                candidate_frames.append(pool)

            if (entry_index + 1) % 25 == 0:
                elapsed = time.time() - started
                print(f'[pool {pool_index:02d}] {entry_index + 1}/{entries.shape[0]} entries, '
                      f'{elapsed / (entry_index + 1):.1f} s/entry', flush=True)
    except Exception as error:
        # Keep the entries already run rather than losing a pool that failed hours in. The summary
        # JSON records the abort, so a partial pool cannot be mistaken for a complete one.
        aborted = repr(error)
        print(f'[pool {pool_index:02d}] ABORTED after {len(summaries)} entries: {aborted}',
              flush=True)
    finally:
        shutdown_mp_workers(processes, task_queues)

    # One set of files per pool. The analysis globs entries_*.parquet and recovers the bundle by
    # splitting on '_shard', so the pool suffix needs no change there.
    pd.DataFrame(summaries).to_parquet(out_dir / f'entries_{pool_tag}.parquet', index=False)
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_parquet(
            out_dir / f'candidates_{pool_tag}.parquet', index=False
            )

    elapsed = time.time() - started
    run_summary = {
        'bundle': bundle_tag(args),
        'error_multiplier': args.error_multiplier,
        'n_contaminants': args.n_contaminants,
        'contaminant_bias': args.contaminant_bias,
        'n_dropout': args.n_dropout,
        'broadening_tag': BROADENING_TAG,
        'shard': args.shard,
        'n_shards': args.n_shards,
        'pool': pool_index,
        'n_pools': args.n_pools,
        'pool_size': args.pool_size,
        'seed': args.seed,
        'optimizer_seed': args.seed + pool_index,
        'conventional_cell': bool(args.conventional_cell),
        'rtol_headline': RTOL_HEADLINE,
        'rtol_alternate': RTOL_ALTERNATE,
        'n_entries_requested_per_bl': args.n_entries_per_bl,
        'n_entries_run': len(summaries),
        'n_entries_expected': int(entries.shape[0]),
        'n_failures': len(failures),
        'failures': failures,
        'correct_rows_collapsed_by_promotion': correct_rows_collapsed,
        'entries_losing_correct_to_promotion': entries_losing_correct,
        'aborted': aborted,
        'seconds_total': elapsed,
        'seconds_per_entry': elapsed / max(1, len(summaries)),
        }
    with open(out_dir / f'summary_{pool_tag}.json', 'w') as summary_file:
        json.dump(run_summary, summary_file, indent=2)
    print(f'[pool {pool_index:02d}] done: {len(summaries)}/{entries.shape[0]} entries, '
          f'{run_summary["seconds_per_entry"]:.1f} s/entry, {len(failures)} failures'
          + ('' if aborted is None else ' (ABORTED)'), flush=True)
    if aborted is not None:
        raise SystemExit(1)


def main():
    args = _parse_args()

    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    invalid = [bl for bl in bravais_lattices if bl not in BRAVAIS_LATTICES]
    if invalid:
        raise SystemExit(f"Unknown Bravais lattices: {', '.join(invalid)}")

    tag = bundle_tag(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f'{tag}_shard{args.shard:02d}of{args.n_shards:02d}'

    # Sample every lattice, then take this shard's stride. The sample depends only on the seed and
    # the per-lattice count, so all bundles and all shards agree on which entries are in the set.
    entries = []
    for bravais_lattice in bravais_lattices:
        sampled = sample_entries(bravais_lattice, args.n_entries_per_bl, args.seed)
        entries.append(sampled)
        print(f'{bravais_lattice}: sampled {sampled.shape[0]} of {args.n_entries_per_bl} requested',
              flush=True)
    entries = pd.concat(entries, ignore_index=True)
    entries = entries.iloc[args.shard::args.n_shards].reset_index(drop=True)
    print(f'shard {args.shard}/{args.n_shards}: {entries.shape[0]} entries, bundle {tag}; '
          f'{args.n_pools} pools x {args.pool_size} processes = '
          f'{args.n_pools * args.pool_size} total', flush=True)

    # Stripe rather than block: entries are concatenated in lattice order, so striping gives every
    # pool the same lattice mix and therefore roughly the same total cost. Blocking would hand one
    # pool all the triclinic entries, which are the most expensive by a wide margin.
    pool_entries = [entries.iloc[index::args.n_pools].reset_index(drop=True)
                    for index in range(args.n_pools)]

    if args.n_pools == 1:
        # Run inline, so a single-pool debugging run has no extra process layer to obscure errors.
        run_pool(0, args, pool_entries[0], out_dir, shard_tag)
        return

    started = time.time()
    processes = []
    for pool_index in range(args.n_pools):
        process = Process(target=run_pool,
                          args=(pool_index, args, pool_entries[pool_index], out_dir, shard_tag))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    failed = [index for index, process in enumerate(processes) if process.exitcode != 0]
    n_run = 0
    for pool_index in range(args.n_pools):
        summary_path = out_dir / f'summary_{shard_tag}_pool{pool_index:02d}.json'
        if summary_path.exists():
            with open(summary_path) as summary_file:
                n_run += json.load(summary_file)['n_entries_run']
    elapsed = time.time() - started
    print(f'shard {args.shard}/{args.n_shards} finished: {n_run}/{entries.shape[0]} entries in '
          f'{elapsed / 3600:.2f} h wall, {elapsed / max(1, n_run):.2f} s/entry aggregate '
          f'({args.n_pools * args.pool_size} processes)', flush=True)
    if failed:
        raise SystemExit(f'pools {failed} exited non-zero; their partial output was still written')


if __name__ == '__main__':
    main()
