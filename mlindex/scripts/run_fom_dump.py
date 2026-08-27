"""Generate Benchmark B: the frozen candidate pool campaign 2 develops against.

Schema v3, specified by `docs/fom_campaign2/SCHEMA.md`. One invocation is one condition bundle,
one shard of the frozen split, run over `--n-pools` independent optimizer pools.

Differences from campaign 1's `run_fom_dump.py` that a reader should know about:

* The peak-list synthesis is `mlindex.model_training.FomPatterns`, a module, rather than a
  sibling script reached through `sys.path`.
* The condition set is `mlindex.model_training.FomConditions`, and `--condition` names a bundle
  from it. The tag rule has exactly one implementation; `--print-tag` is how the submit script
  obtains a tag instead of rebuilding the rule in bash.
* Search seeding is per (entry, condition, Bravais lattice) rather than one stream per pool, so
  any subset of the benchmark regenerates identically (PROTOCOL section 6, R17).
* **The entry list is READ from the frozen manifest, never re-derived by sampling.**
  `FomPatterns.sample_entries` draws `rng.choice(size=n, replace=False)`, so no single
  `--n-entries-per-bl` reproduces a manifest whose per-lattice counts run 106 / 1 400 / 3 000 --
  five lattices are hard-capped by the source population (C2-F-048). Sampling and then
  intersecting, which is what this driver did before, silently produced a different benchmark
  whenever a parameter drifted.
* **Labels are written at generation**, by the batched labeller, before anything is subsampled.
  The order is forced: the retention rule keeps every correct candidate, so correctness has to be
  known first (SCHEMA.md; R24).
* The surplus peaks, the labels and the condition are written for every stream.
"""

# One thread per process, set before numpy is imported. NPOOLS x POOLSIZE already fills the node's
# 128 physical cores, so a BLAS that helpfully spawns its own threads oversubscribes it by an order
# of magnitude and the run gets slower the more cores it is given.
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import hashlib
import json
import platform
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

import mlindex
from mlindex.command_line.run import BRAVAIS_LATTICES
from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomConditions
from mlindex.model_training import FomPatterns
from mlindex.utilities.ErrorAdder import ContaminantPlacementError


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate the campaign-2 FOM benchmark candidate pool (schema v3)')
    parser.add_argument('--condition', type=str, default='nominal',
                        help='Condition bundle key from FomConditions, e.g. nominal, noisy, '
                             'contaminated, sparse4, second_phase')
    parser.add_argument('--print-tag', action='store_true',
                        help='Print the bundle tag and exit. The submit script uses this so the '
                             'tag rule has one implementation rather than two')
    parser.add_argument('--n-entries-per-bl', type=int, default=4,
                        help='Entries sampled per Bravais lattice, capped by availability. Used '
                             'ONLY when no --split-manifest is given: a manifested run reads its '
                             'entry list from the manifest, because sampling cannot reproduce a '
                             'non-uniform per-lattice draw (C2-F-048)')
    parser.add_argument('--bravais-lattices', type=str, default=','.join(BRAVAIS_LATTICES))
    parser.add_argument('--n-pools', type=int, default=1,
                        help='Independent optimizer pools run concurrently, each with its own '
                             'manager in its own process. Only managers load models (~3 GB each), '
                             'so this is the memory-limited axis. n_pools x pool_size should '
                             'equal the physical core count')
    parser.add_argument('--pool-size', type=int, default=1,
                        help='Processes per pool: one manager plus pool-size-1 workers')
    parser.add_argument('--shard', type=int, default=0,
                        help='This task index. Entries are striped shard::n_shards, so shards '
                             'partition the bundle and each is independently recoverable')
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12345,
                        help='Base seed for entry sampling and the per-entry noise. Also seeds '
                             'the search unless --optimizer-seed overrides it')
    parser.add_argument('--optimizer-seed', type=int, default=None,
                        help='Base seed for the candidate SEARCH alone, leaving the entry sample '
                             'and the peak-list noise on --seed. Defaults to --seed, so an '
                             'invocation written before this flag existed produces the same pool. '
                             'This is what S08 measures the reproducibility floor with: four runs '
                             'differing only here differ only in the search, so the spread is '
                             'search noise and not generation noise')
    parser.add_argument('--entry-ids-file', type=str, default=None,
                        help='CSV holding an identifier column. Restricts the run to those '
                             'entries. Unlike campaign 1, a restricted run DOES reproduce a full '
                             'one candidate for candidate -- that is what the per-pattern search '
                             'seeding buys, and the subset gate proves it')
    parser.add_argument('--prune-threshold', type=float, default=1.5,
                        help='M20 below which a candidate is discarded before deduplication. '
                             'The benchmark generates at 1.5: the largest cut that loses no '
                             'reachable pattern-condition, so every higher cut is reconstructable '
                             'by restriction. Production ships 5.0 and that is unchanged')
    parser.add_argument('--top-k', type=int, default=200,
                        help='Negative subsampling depth. Every candidate inside the top K by '
                             'each reported merit is kept, which makes rank metrics exact at any '
                             'depth <= K. Sized by S06 against real pool sizes; the old default '
                             'of 500 was an assumption and is close to a no-op at measured pool '
                             'sizes')
    parser.add_argument('--negative-rate', type=float, default=0.05,
                        help='Retention probability for a candidate that is neither correct nor '
                             'in the top-K union. Its inverse becomes the row sampling_weight')
    parser.add_argument('--no-subsample', action='store_true',
                        help='Keep every candidate, with the bookkeeping columns still written. '
                             'What the gates run with, and what the held-back fully-retained '
                             'shard the weight check needs is generated with')
    parser.add_argument('--no-label', action='store_true',
                        help='Skip labelling. Subsampling then refuses, because the retention '
                             'rule keeps every correct candidate and cannot run blind')
    parser.add_argument('--predownsample-entries', type=int, default=None,
                        help='Write the pre-deduplication stream for this many entries per pool '
                             'rather than for all of them. It is ~7.7x the survivor stream and '
                             'SCHEMA.md specifies it as a stratified subsample. Omit to write it '
                             'for all; pass 0 to write none')
    parser.add_argument('--arm', type=str, default=None,
                        help='Restrict to entries whose manifest arm contains this string, e.g. '
                             'mechanism. Requires --split-manifest')
    parser.add_argument('--split-manifest', type=str, default=None,
                        help='Frozen manifest supplying the fom-train/dev/test split AND the '
                             'entry list. The split is by source entry and must never be '
                             're-derived here')
    # Not `required=True`: `--print-tag` is how the submit script asks what directory a
    # bundle should be written to, so it cannot be made to name that directory first.
    parser.add_argument('--out-dir', type=str, default=None)
    return parser.parse_args(argv)


def load_manifest(manifest_path):
    """The frozen split manifest, or None.

    PROTOCOL section 3 rule 5: splits are by source entry, never by candidate, and the same
    crystal must never appear in two splits under different noise. Re-deriving the split here
    would break that silently across condition bundles.

    It carries the frozen `volume_decile` too, which is the other half of R14: recomputing a
    within-lattice percentile rank downstream moved 114 of campaign 1's 5 922 entries and took
    the hard stratum with them. Read here, written onto every entry row, joined thereafter.
    """
    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        raise SystemExit(f'Frozen split manifest not found at {path}. The split must not be '
                         're-derived; copy it across or pass --split-manifest.')
    manifest = pd.read_parquet(path)
    id_column = 'identifier' if 'identifier' in manifest.columns else 'entry_id'
    split_column = 'split' if 'split' in manifest.columns else 'fom_split'
    if split_column not in manifest.columns:
        raise SystemExit(f'{path} has no split column; found {sorted(manifest.columns)}.')
    manifest = manifest.rename(columns={id_column: 'identifier', split_column: 'split'})
    if 'volume_decile' not in manifest.columns:
        raise SystemExit(f'{path} has no volume_decile column. It is frozen at split time and '
                         'must never be recomputed downstream (R14).')
    return manifest.set_index('identifier')


def manifest_sha256(manifest_path):
    """The split manifest's checksum, for the run manifest. Gate condition 5.

    `docs/` is git-ignored, so the manifest reaches NERSC only through `sync_record.sh push` and
    nothing else would notice if a stale copy were in place there. Recording the checksum beside
    the pool is how a later session proves which split the pool was generated against.
    """
    if manifest_path is None:
        return None
    digest = hashlib.sha256()
    with open(manifest_path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def entries_from_manifest(manifest, bravais_lattices, columns):
    """The manifest's entries, read out of the source datasets by identifier.

    THE MANIFEST IS THE ENTRY LIST, not merely a split lookup. `FomPatterns.sample_entries` draws
    `rng.choice(data.shape[0], size=n, replace=False)`, and a draw of 3 000 is not a superset of a
    draw of 1 400 -- so no single `--n-entries-per-bl` reproduces the frozen manifest, whose
    per-lattice counts are 106 for cF and 3 000 for each of mP, mC and aP (C2-F-048, C2-R-010).
    Sampling and then intersecting, which is what this driver used to do, quietly generated a
    different benchmark whenever a sampling parameter drifted.

    Refuses on any manifest entry it cannot find, rather than generating a smaller pool and
    reporting a count nobody reads.
    """
    wanted_by_lattice = {}
    for bravais_lattice in bravais_lattices:
        selected = manifest.index[manifest['bravais_lattice'] == bravais_lattice]
        if len(selected):
            wanted_by_lattice[bravais_lattice] = set(selected)

    frames = []
    for bravais_lattice, wanted in wanted_by_lattice.items():
        path = FomPatterns.DATASET_DIRECTORY / f'dataset_{bravais_lattice}.parquet'
        data = pd.read_parquet(path, columns=list(columns))
        data = data.loc[data['identifier'].isin(wanted)]
        # Stable order, independent of the parquet's row order and of how many lattices ran, so a
        # pool's entry sequence is a property of the manifest alone.
        data = data.sort_values('identifier', kind='stable', ignore_index=True)
        missing = wanted - set(data['identifier'])
        if missing:
            raise SystemExit(
                f'{len(missing)} of {len(wanted)} {bravais_lattice} manifest entries are not in '
                f'{path}, e.g. {sorted(missing)[:3]}. The manifest and the source datasets '
                'disagree; do not generate against a partial entry list.')
        frames.append(data)

    if not frames:
        raise SystemExit('the manifest holds no entries for any requested Bravais lattice')
    return pd.concat(frames, ignore_index=True)


def extinction_group_true(entry):
    """The truth's extinction group, derived when the source dataset did not store one.

    `reindexed_extinction_group` is **null on 62.6 % of the frozen manifest's entries** -- 11 879
    of 18 991, and at a similar rate in every one of the fourteen lattices. The nullness is not a
    property of the space group: `P 1 21/n 1` appears 105 996 times with the column empty and
    62 716 times with it filled, so it is an artefact of how the source datasets were generated.

    `reindexed_spacegroup_symbol_hm` is never null, and the repository already carries the mapping
    (`SpaceGroups.map_spacegroup_to_extinction_group`). Checked over 131 282 source rows across all
    fourteen lattices: it agrees with **every** stored value -- 48 815 of 48 815, zero
    disagreements -- and recovers all 82 467 null ones, with nothing raising. So deriving where the
    column is empty changes no entry that had a value and fills the rest.

    This matters because S11's whole subject is the extinction-group assignment rule, and without
    it the ground truth it is scored against would be absent on nearly two thirds of Benchmark B.
    A column null in most rows is exactly what C2-F-046 ruled out.
    """
    from mlindex.utilities.SpaceGroups import map_spacegroup_to_extinction_group
    stored = entry['reindexed_extinction_group']
    if isinstance(stored, str) and stored:
        return stored
    try:
        derived = map_spacegroup_to_extinction_group(entry['reindexed_spacegroup_symbol_hm'])
    except Exception:
        return None
    # The mapping returns (group, spacegroup_number); only the group is the truth column.
    return derived[0] if isinstance(derived, tuple) else derived


def entry_record(entry, condition, pattern, split, volume_decile, degeneracy):
    """One row per indexed pattern, with the ground truth and the conditions in force."""
    from mlindex.utilities.ErrorAdder import q2_sigma_params
    intercept, slope = q2_sigma_params()
    is_degenerate, accidental, systematic = degeneracy
    full_peaks = np.asarray(entry[f'q2_{FomPatterns.BROADENING_TAG}'], dtype=float)
    return {
        'entry_id': entry['identifier'],
        'q2_digest': FomBenchmark.q2_digest(pattern.q2_obs),
        'source_db': entry['database'],
        'split': split,
        'condition_bundle': condition.tag,
        'q2_obs': np.asarray(pattern.q2_obs, dtype=np.float64),
        'q2_holdout': np.asarray(pattern.q2_holdout, dtype=np.float64),
        'hkl_holdout': np.asarray(pattern.hkl_holdout, dtype=np.int16).reshape(-1),
        'n_peaks_available': int(np.count_nonzero(full_peaks > 0)),
        'volume_decile': volume_decile,
        'pool_size_full': -1,          # filled in once the candidates for this entry are known
        'q2_error_multiplier': float(condition.error_multiplier),
        'n_contaminants': int(condition.n_contaminants),
        'contaminant_bias': float(condition.contaminant_bias),
        'n_dropout': int(condition.n_dropout),
        'n_dropout_achieved': int(pattern.n_dropout_achieved),
        'second_phase_lines': int(condition.second_phase_lines),
        'second_phase_bias': float(condition.second_phase_bias),
        'second_phase_partner': pattern.second_phase_partner,
        'error_law': FomConditions.ERROR_LAW,
        'error_law_params': np.array([intercept * condition.intercept_scale, slope],
                                     dtype=np.float64),
        'intercept_scale': float(condition.intercept_scale),
        'broadening_tag': FomPatterns.BROADENING_TAG,
        'is_degenerate': bool(is_degenerate),
        'degeneracy_conditions': '|'.join(accidental),
        'degeneracy_systematic': '|'.join(systematic),
        'xnn_true': np.asarray(entry['reindexed_xnn'], dtype=np.float64),
        'unit_cell_true': np.asarray(entry['reindexed_unit_cell'], dtype=np.float64),
        'volume_true': float(entry['reindexed_volume']),
        'bravais_lattice_true': entry['bravais_lattice'],
        'lattice_system_true': entry['lattice_system'],
        'spacegroup_true': entry['reindexed_spacegroup_symbol_hm'],
        'extinction_group_true': extinction_group_true(entry),
        'hkl_true': np.asarray(pattern.hkl_obs, dtype=np.int16).reshape(-1),
        }


def preflight(args):
    """Refuse an impossible configuration BEFORE the search runs, not after.

    The retention rule needs `is_correct`, so a run that does not label cannot subsample.
    Discovering that at the end costs the whole bundle -- campaign 1 added abort-safety after
    losing an abort near the end of a 2.5 h run, and a guard that fires after the work is done
    reintroduces exactly that.
    """
    if args.no_label and not args.no_subsample:
        raise SystemExit(
            'Refusing to subsample an unlabelled pool. Negative subsampling keeps every correct '
            'candidate, so labelling must happen first (SCHEMA.md; S07 handoff, "Labelling, '
            'subsampling and consolidation -- in that order"). Drop --no-label, or add '
            '--no-subsample and subsample after a separate labelling pass.')
    if args.arm and not args.split_manifest:
        raise SystemExit('--arm selects on a manifest column and needs --split-manifest')

    # cctbx, checked in two seconds rather than discovered after the queue wait. Two independent
    # entry points need it and BOTH run in the batch job: `SpaceGroups.get_spacegroup_hkl_ref`,
    # which `Candidates.assign_extinction_group` calls on every pattern (so campaign 1's array
    # already proved this one), and `LatticeDegeneracy.reduced_cell`, which is new in campaign 2
    # and runs once per entry. It is an OPTIONAL dependency of this package -- the end-user path
    # must not acquire it -- so an inference-only environment can legitimately lack it.
    try:
        from cctbx import crystal  # noqa: F401
        from cctbx import sgtbx    # noqa: F401
    except ImportError as error:
        raise SystemExit(
            f'cctbx is not importable in this interpreter ({error}). The generation run needs it '
            'twice per pattern: for the extinction-group reference sets and for the Niggli '
            'reduction behind `is_degenerate`. Use the environment campaign 1 generated in '
            '(envs/onnx on NERSC), or install cctbx-base.')
    if not 1 <= args.n_shards or not 0 <= args.shard < args.n_shards:
        raise SystemExit(f'--shard must be in [0, {args.n_shards}); got {args.shard}')


def label_and_subsample(candidates, entry_rows, args, pool_index):
    """Label, then subsample, then hand back what the pool writes. In that order.

    THE ORDER IS FORCED AND IT IS EASY TO GET WRONG. The retention rule keeps every *correct*
    candidate, so correctness has to be known before anything is dropped. Subsampling blind would
    delete the entire signal at a base rate under 1 % and leave a pool that looks like a
    generation failure rather than a thinned one -- campaign 1's most expensive repeated mistake
    was labelling on *load*, and this would be the same mistake with the data gone.

    The subsampler ranks on all seven merits of the reduced core, not on M20 alone: K = 200 was
    measured as the size of the union over that set, ~3.3x K (C2-F-051). The six recomputed ones
    are dropped again before writing, because they are recomputable offline and by SCHEMA.md's own
    rule that means they do not earn a column.
    """
    if candidates.empty:
        return candidates, False
    entries = pd.DataFrame(entry_rows)

    if not args.no_label:
        started = time.time()
        candidates = FomBenchmark.label_frame(candidates, entries)
        print(f'[pool {pool_index:02d}] labelled {candidates.shape[0]} candidates in '
              f'{time.time() - started:.0f} s, {int(candidates["is_correct"].sum())} correct',
              flush=True)

    if args.no_subsample:
        # Not a no-op: the bookkeeping columns are still written, so a pool generated whole reads
        # through exactly the same loader as a thinned one.
        return FomBenchmark.subsample_negatives(
            candidates, merit_columns=('M20',), top_k=int(args.top_k), negative_rate=1.0,
            base_seed=int(args.seed)), False

    ranked = FomBenchmark.with_reduced_merits(candidates, entries)
    thinned = FomBenchmark.subsample_negatives(
        ranked, merit_columns=FomBenchmark.REDUCED_MERIT_COLUMNS, top_k=int(args.top_k),
        negative_rate=float(args.negative_rate), base_seed=int(args.seed))
    kept = [column for column in thinned.columns
            if column not in FomBenchmark.REDUCED_MERIT_COLUMNS or column == 'M20']
    print(f'[pool {pool_index:02d}] subsampled {candidates.shape[0]} -> {thinned.shape[0]} rows '
          f'({thinned.shape[0] / max(1, candidates.shape[0]):.1%} retained)', flush=True)
    return thinned[kept], True


def optimizer_options(args):
    """The opt_params overrides one invocation asks for.

    `search_seed_scheme` is the campaign-2 change and it is opt-in for a reason: it alters which
    candidates the search generates, so the shipped indexer keeps campaign 1's behaviour and the
    benchmark gets the reproducible one. It reads `self.rank`, not `self.comm` -- the latter is
    what made it crash every multiprocessing worker until 2026-08-27 (C2-F-047), which is the mode
    this driver runs in at every pool size above one.
    """
    return {
        'prune_m20_threshold': float(args.prune_threshold),
        'prune_criterion_capture': True,
        'dump_candidates': True,
        'search_seed_scheme': 'per_entry_bravais',
        'search_base_seed': int(search_seed(args)),
        }


def search_seed(args):
    """The base seed for the candidate search, which is `--seed` unless `--optimizer-seed` is set.

    THE POINT OF SEPARATING THEM. The reproducibility floor is the spread of a reported number
    over runs differing **only in the search**. `--seed` moves three things at once -- which
    entries are drawn, what noise is added to each peak list, and where the search starts -- so
    four runs at four `--seed` values measure generation noise and scoring noise together, which
    is exactly the conflation `METRICS.md` section 8 splits into three separate floors.

    Defaulting to `args.seed` rather than to a constant is deliberate: every invocation written
    before this flag existed keeps producing the pool it produced.
    """
    return int(args.seed if args.optimizer_seed is None else args.optimizer_seed)


def _pool_complete(out_dir, pool_tag, want_predownsample):
    """Every one of a pool's tables present *and readable*, so a resumed task can skip it.

    Readability, not just existence: a pool killed mid-write leaves the entry table complete and
    the candidate shard truncated, and silently skipping that would lose its candidates for good.
    The pre-deduplication shard counts only when it was asked for, or a task requeued after
    `--predownsample-entries` changed would skip pools holding only the survivors.
    """
    paths = [Path(out_dir) / f'entries_{pool_tag}.parquet',
             Path(out_dir) / f'candidates_{pool_tag}.parquet']
    if want_predownsample:
        paths.append(Path(out_dir) / f'predownsample_{pool_tag}.parquet')
    if not all(path.exists() for path in paths):
        return False
    try:
        for path in paths:
            pd.read_parquet(path, columns=['entry_id'])
    except Exception:
        # Truncated by a kill mid-write. Regenerate rather than resume onto a corrupt shard.
        return False
    return True


def run_pool(pool_index, args, entries, manifest, out_dir, shard_tag, second_phase_pool):
    """One manager plus pool_size-1 workers, working through this pool's stripe of entries.

    Runs in its own process: `setup_mp_optimizers` injects its queues at CLASS level on
    `MPOptimizerManager`, so concurrent pools cannot share an interpreter.
    """
    from mlindex.optimization.MPOptimizer import run_mp_bl
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers
    from mlindex.optimization.MPOptimizer import shutdown_mp_workers
    from mlindex.utilities.LatticeDegeneracy import is_degenerate

    condition = FomConditions.BY_KEY[args.condition]
    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    pool_tag = f'{shard_tag}_pool{pool_index:02d}'
    want_predownsample = args.predownsample_entries != 0
    if _pool_complete(out_dir, pool_tag, want_predownsample):
        print(f'[pool {pool_index:02d}] already written, skipping', flush=True)
        return

    optimizers, processes, task_queues = setup_mp_optimizers(
        args.pool_size, FomPatterns.BROADENING_TAG, n_candidates_scale=1,
        seed=search_seed(args) + pool_index, options=optimizer_options(args))

    entry_rows, candidate_frames, predownsample_frames, failures = [], [], [], []
    merit_at_prune_names = ()
    consecutive_failures = 0
    started = time.time()
    aborted = None
    try:
        for position in range(entries.shape[0]):
            entry = entries.iloc[position]
            hkl_full = np.stack([
                np.asarray(entry[f'reindexed_{axis}_{FomPatterns.BROADENING_TAG}'], dtype=float)
                for axis in ('h', 'k', 'l')], axis=1)
            try:
                pattern = FomPatterns.prepare_peak_list(
                    entry, condition, args.seed, hkl=hkl_full,
                    second_phase_pool=second_phase_pool)
            except ContaminantPlacementError as error:
                # The contaminant draw is rejection-sampled and the whole set is redrawn on any
                # collision, so acceptance decays exponentially in the contaminant count. The
                # max-attempts guard turns a hang into this recorded skip.
                failures.append({'identifier': entry['identifier'],
                                 'reason': 'contaminant_placement', 'detail': str(error)})
                continue

            context = {'entry_id': entry['identifier'],
                       'q2_digest': FomBenchmark.q2_digest(pattern.q2_obs),
                       'condition_bundle': condition.tag}
            records, predownsample_records = [], []
            try:
                for bravais_lattice in bravais_lattices:
                    optimizer = optimizers[bravais_lattice]
                    optimizer.dump_context = context
                    run_mp_bl(optimizer, bravais_lattice, task_queues, q2=pattern.q2_obs,
                              zero_error=False, wavelength=None,
                              n_top=FomPatterns.N_TOP_CANDIDATES)
                    records += optimizer.drain_candidate_dump()
                    if want_predownsample:
                        predownsample_records += optimizer.drain_predownsample_dump()
                # `is_degenerate` is cctbx-backed (Niggli reduction of the primitive setting)
                # and runs once per entry, so it is INSIDE this guard rather than after it. A
                # single cell cctbx refuses would otherwise abort the whole pool at that entry --
                # and cctbx raising on real cells is documented in this codebase, not
                # hypothetical: `_conventional_cell` did it for ~4.8 % of entries before the
                # guard now on `main` existed.
                degeneracy = is_degenerate(
                    np.asarray(entry['reindexed_unit_cell'], dtype=float),
                    entry['bravais_lattice'])
            except Exception as error:
                # An isolated entry failure is tolerable; a systematic one is not, and an
                # unattended run must not spend hours writing a mostly-failed shard.
                failures.append({'identifier': entry['identifier'],
                                 'reason': 'optimizer', 'detail': repr(error)})
                consecutive_failures += 1
                if consecutive_failures >= FomPatterns.MAX_CONSECUTIVE_FAILURES:
                    raise
                continue
            consecutive_failures = 0

            candidates = FomBenchmark.records_to_frame(records)
            # What the k-th entry of `merit_at_prune` means. Recorded once, in the manifest,
            # rather than as a column per criterion -- so the column set does not change when the
            # merit set does, and a loader can always name what it is reading (C2-R-001).
            merit_at_prune_names = candidates.attrs.get('merit_at_prune_names',
                                                        merit_at_prune_names)
            candidate_frames.append(candidates)
            # SCHEMA.md specifies the pre-deduplication stream as a stratified subsample of
            # entries, and it is the expensive one -- ~7.7x the survivor stream. Entries arrive in
            # manifest order and the manifest is lattice-stratified, so a prefix is a stratified
            # subsample.
            if want_predownsample and (args.predownsample_entries is None
                                       or len(predownsample_frames)
                                       < args.predownsample_entries):
                predownsample_frames.append(
                    FomBenchmark.predownsample_records_to_frame(predownsample_records))

            frozen = (manifest.loc[entry['identifier']]
                      if manifest is not None and entry['identifier'] in manifest.index
                      else None)
            row = entry_record(
                entry, condition, pattern,
                'unassigned' if frozen is None else str(frozen['split']),
                # READ, never recomputed (R14). -1 marks a run with no frozen manifest, which is
                # a run whose numbers cannot be stratified by volume -- not a default value.
                volume_decile=-1 if frozen is None else int(frozen['volume_decile']),
                degeneracy=degeneracy)
            row['pool_size_full'] = int(candidates.shape[0])
            entry_rows.append(row)

            if (position + 1) % 25 == 0:
                elapsed = time.time() - started
                print(f'[pool {pool_index:02d}] {position + 1}/{entries.shape[0]} entries, '
                      f'{elapsed / (position + 1):.1f} s/entry', flush=True)
    except Exception as error:
        # Write what this pool has rather than losing hours of it. Re-raised once the tables are
        # on disk, so the process still exits non-zero and main() reports the failure.
        aborted = error
    finally:
        shutdown_mp_workers(processes, task_queues)

    if failures:
        with open(Path(out_dir) / f'failures_{pool_tag}.json', 'w', encoding='utf-8') as handle:
            json.dump(failures, handle, indent=2, sort_keys=True)
        reasons = {}
        for failure in failures:
            reasons[failure['reason']] = reasons.get(failure['reason'], 0) + 1
        # Counted AND reported: campaign 1 counted failures and threw them away, so an unattended
        # run left no way to tell a handful of unplaceable contaminants from a systematic problem.
        print(f'[pool {pool_index:02d}] failures by reason: {reasons}', flush=True)

    if entry_rows:
        FomBenchmark.write_entry_table(
            pd.DataFrame(entry_rows, columns=list(FomBenchmark.ENTRY_COLUMNS)), out_dir, pool_tag)
    if candidate_frames:
        candidates = pd.concat(candidate_frames, ignore_index=True)
        candidates, _ = label_and_subsample(candidates, entry_rows, args, pool_index)
        FomBenchmark.write_candidate_shard(candidates, out_dir, pool_tag)
    if predownsample_frames:
        FomBenchmark.write_predownsample_shard(
            pd.concat(predownsample_frames, ignore_index=True), out_dir, pool_tag)

    elapsed = time.time() - started
    print(f'[pool {pool_index:02d}] {"ABORTED" if aborted else "done"}: '
          f'{len(entry_rows)}/{entries.shape[0]} entries, '
          f'{elapsed / max(1, len(entry_rows)):.1f} s/entry, {len(failures)} failures', flush=True)
    if aborted is not None:
        raise aborted


def _commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=Path(mlindex.__path__[0]).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _model_revision():
    """The pinned model revision, so a pool records which weights generated it."""
    try:
        path = Path(mlindex.__path__[0]) / 'model_metadata.json'
        with open(path, encoding='utf-8') as handle:
            return json.load(handle).get('model_revision')
    except Exception:
        return None


def _scipy_version():
    try:
        import scipy
        return scipy.__version__
    except Exception:
        return None


def run(args):
    condition = FomConditions.BY_KEY[args.condition]
    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    unknown = [bl for bl in bravais_lattices if bl not in BRAVAIS_LATTICES]
    if unknown:
        raise SystemExit(f"Unknown Bravais lattices: {', '.join(unknown)}")

    if args.out_dir is None:
        raise SystemExit('--out-dir is required for a generation run')
    preflight(args)
    manifest = load_manifest(args.split_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest is not None:
        wanted = manifest
        if args.arm:
            if 'arm' not in manifest.columns:
                raise SystemExit(f'--arm given but {args.split_manifest} has no arm column')
            wanted = manifest[manifest['arm'].astype(str).str.contains(args.arm)]
            if wanted.empty:
                raise SystemExit(f'no manifest entry has an arm containing {args.arm!r}')
        entries = entries_from_manifest(wanted, bravais_lattices, FomPatterns.DUMP_READ_COLUMNS)
        print(f'manifest: {len(wanted)} entries wanted, {entries.shape[0]} read', flush=True)
    else:
        # No manifest: a smoke run. It samples, it carries no split, and it says so.
        entries = pd.concat(
            [FomPatterns.sample_entries(bl, args.n_entries_per_bl, args.seed,
                                        columns=FomPatterns.DUMP_READ_COLUMNS)
             for bl in bravais_lattices],
            ignore_index=True)
        print(f'no split manifest: sampled {entries.shape[0]} entries, split "unassigned"',
              flush=True)

    # Built BEFORE the id restriction and BEFORE striping, from the whole entry set, so the
    # partner draw is a property of the manifest rather than of the run's shape. If it were built
    # after either, a subset run would place DIFFERENT second-phase lines from the full run --
    # different peak lists, so different candidates -- and the subset reproducibility gate would
    # be measuring the wrong thing while appearing to pass. Campaign 1's driver built it after the
    # restriction; its gate happened to use a bundle with no second phase, so this never showed.
    second_phase_pool = (FomPatterns.build_second_phase_pool(entries)
                         if condition.second_phase_lines > 0 else None)

    if args.entry_ids_file:
        wanted_ids = pd.read_csv(args.entry_ids_file)
        id_column = 'identifier' if 'identifier' in wanted_ids.columns else 'entry_id'
        wanted_ids = set(wanted_ids[id_column])
        missing = wanted_ids - set(entries['identifier'])
        if missing:
            raise SystemExit(f'{len(missing)} of {len(wanted_ids)} requested entries were not '
                             f'found, e.g. {sorted(missing)[:3]}.')
        entries = entries[entries['identifier'].isin(wanted_ids)].reset_index(drop=True)

    entries = entries.iloc[args.shard::args.n_shards].reset_index(drop=True)
    tag = condition.tag
    shard_tag = f'{tag}_shard{args.shard:02d}of{args.n_shards:02d}'
    print(f'shard {args.shard}/{args.n_shards}: {entries.shape[0]} entries, bundle {tag}; '
          f'{args.n_pools} pools x {args.pool_size} processes', flush=True)

    started = time.time()
    if args.n_pools == 1:
        run_pool(0, args, entries, manifest, out_dir, shard_tag, second_phase_pool)
    else:
        pool_processes = []
        for pool_index in range(args.n_pools):
            stripe = entries.iloc[pool_index::args.n_pools].reset_index(drop=True)
            process = Process(target=run_pool,
                              args=(pool_index, args, stripe, manifest, out_dir, shard_tag,
                                    second_phase_pool))
            process.start()
            pool_processes.append(process)
        for process in pool_processes:
            process.join()
        failed = [process.exitcode for process in pool_processes if process.exitcode]
        if failed:
            raise SystemExit(f'{len(failed)} pools exited non-zero: {failed}')

    FomBenchmark.write_manifest(
        out_dir,
        bundle=tag,
        condition=condition.key,
        condition_parameters=FomConditions.condition_row(condition),
        schema_version=FomBenchmark.SCHEMA_VERSION,
        commit=_commit_hash(),
        # An arm64-generated pool is not bit-reproducible on x86 and campaign 1's manifest
        # recorded the commit but not the machine (R9).
        arch=platform.machine(),
        platform=platform.platform(),
        numpy_version=np.__version__,
        scipy_version=_scipy_version(),
        python_version=platform.python_version(),
        model_revision=_model_revision(),
        seed=args.seed,
        # Both, always. They are equal on almost every run, and a manifest recording one number
        # cannot say whether two pools differ in their patterns or only in their search -- which
        # is the single distinction the reproducibility floor is made of.
        optimizer_seed=search_seed(args),
        search_seed_scheme='per_entry_bravais',
        # The shipped schedule, unchanged. Halving it leaves the ceiling at 0.00 pp but buys only
        # 4.7 % of wall clock for a 17.8 % larger pool (C2-F-053, C2-F-054), so it is refused on
        # price -- and a pool has to record which schedule produced it either way.
        iteration_scale=1.0,
        merit_at_prune_names=list(FomBenchmark.REDUCED_MERIT_COLUMNS),
        broadening_tag=FomPatterns.BROADENING_TAG,
        error_law=FomConditions.ERROR_LAW,
        n_peaks=FomPatterns.N_PEAKS,
        n_holdout=FomPatterns.N_HOLDOUT,
        n_top_candidates=FomPatterns.N_TOP_CANDIDATES,
        n_entries_per_bl=args.n_entries_per_bl if manifest is None else None,
        bravais_lattices=bravais_lattices,
        arm=args.arm,
        shard=args.shard,
        n_shards=args.n_shards,
        n_pools=args.n_pools,
        pool_size=args.pool_size,
        prune_threshold=float(args.prune_threshold),
        labelled=not args.no_label,
        # What the run ACTUALLY did, not what it was asked to do. An earlier version wrote
        # `subsampled: true` whenever the flag was absent, while the subsampler did not exist --
        # a manifest that misdescribes its own pool is worse than one that omits the field.
        subsampled=not args.no_subsample,
        top_k=int(args.top_k) if not args.no_subsample else None,
        negative_rate=float(args.negative_rate) if not args.no_subsample else None,
        subsample_rule=('correct + top_k union over ' +
                        ','.join(FomBenchmark.REDUCED_MERIT_COLUMNS) +
                        ' + bernoulli(negative_rate), weight = 1/rate on the sampled class'
                        if not args.no_subsample else None),
        predownsample_entries=args.predownsample_entries,
        split_manifest=args.split_manifest,
        split_manifest_sha256=manifest_sha256(args.split_manifest),
        n_entries=entries.shape[0],
        seconds_total=round(time.time() - started, 1),
        )
    print(f'wrote {out_dir}: {entries.shape[0]} entries, {time.time() - started:.0f} s',
          flush=True)


def main(argv=None):
    args = _parse_args(argv)
    if args.print_tag:
        print(FomConditions.BY_KEY[args.condition].tag)
        return
    run(args)


if __name__ == '__main__':
    main()
