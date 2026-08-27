"""Generate Benchmark B: the frozen candidate pool campaign 2 develops against.

Schema v3, specified by `docs/fom_campaign2/SCHEMA.md`. Nothing here is generated at scale by
S05; this is the harness S07 runs, plus the two gates that prove it works.

Differences from campaign 1's `run_fom_dump.py` that a reader should know about:

* The peak-list synthesis is `mlindex.model_training.FomPatterns`, a module, rather than a
  sibling script reached through `sys.path`.
* The condition set is `mlindex.model_training.FomConditions`, and `--condition` names a bundle
  from it. The tag rule has exactly one implementation; `--print-tag` is how the submit script
  obtains a tag instead of rebuilding the rule in bash.
* Search seeding is per (entry, condition, Bravais lattice) rather than one stream per pool, so
  any subset of the benchmark regenerates identically (PROTOCOL §6, R17).
* The surplus peaks, the labels and the condition are written for every stream.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
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
                        help='Entries sampled per Bravais lattice, capped by availability')
    parser.add_argument('--bravais-lattices', type=str, default=','.join(BRAVAIS_LATTICES))
    parser.add_argument('--pool-size', type=int, default=1,
                        help='Processes per pool: one manager plus pool-size-1 workers')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Base seed for entry sampling, the per-entry noise and the search')
    parser.add_argument('--entry-ids-file', type=str, default=None,
                        help='CSV holding an identifier column. Restricts the run to those '
                             'entries after sampling. Unlike campaign 1, a restricted run DOES '
                             'reproduce a full one candidate for candidate -- that is what the '
                             'per-pattern search seeding buys, and gate 2 proves it')
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
                        help='Keep every candidate. What the gates run with')
    parser.add_argument('--predownsample-entries', type=int, default=None,
                        help='Write the pre-deduplication stream for this many entries rather '
                             'than for all of them. It is far larger than the survivor stream '
                             'and SCHEMA.md specifies it as a stratified subsample; without this '
                             'the driver wrote it for every entry. Omit to write it for all')
    parser.add_argument('--arm', type=str, default=None,
                        help='Restrict to entries whose manifest arm contains this string, e.g. '
                             'mechanism. Requires --split-manifest')
    parser.add_argument('--split-manifest', type=str, default=None,
                        help='Frozen manifest supplying the fom-train/dev/test split. The split '
                             'is by source entry and must never be re-derived here. S06 produces '
                             'it; until then a run is unassigned and says so')
    parser.add_argument('--out-dir', type=str, required=True)
    return parser.parse_args(argv)


def load_manifest(manifest_path):
    """The frozen split manifest, or None.

    PROTOCOL §3 rule 5: splits are by source entry, never by candidate, and the same crystal must
    never appear in two splits under different noise. Re-deriving the split here would break that
    silently across condition bundles.

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
        'extinction_group_true': entry['reindexed_extinction_group'],
        'hkl_true': np.asarray(pattern.hkl_obs, dtype=np.int16).reshape(-1),
        }


def subsample_or_refuse(candidate_frames, args):
    """Thin the pool, or say why it will not be thinned. Returns (frames, subsampled).

    THE ORDER IS FORCED AND IT IS EASY TO GET WRONG. The retention rule keeps every *correct*
    candidate, so correctness has to be known before anything is dropped -- label, then subsample,
    then consolidate. Labels are not written by this driver yet, so rather than subsample blind,
    which would delete the entire signal at a base rate under 1 % and leave a pool that looks like
    a generation failure rather than a thinned one, it refuses and names the order.

    Campaign 1's most expensive repeated mistake was labelling on *load*; subsampling before
    labelling would be the same mistake with the data gone.
    """
    if args.no_subsample or not candidate_frames:
        return candidate_frames, False
    merged = pd.concat(candidate_frames, ignore_index=True)
    if 'is_correct' not in merged.columns or merged['is_correct'].isna().all():
        raise SystemExit(
            'Refusing to subsample an unlabelled pool. Negative subsampling keeps every correct '
            'candidate, so labelling must happen first (SCHEMA.md; S07 handoff, "Labelling, '
            'subsampling and consolidation -- in that order"). Re-run with --no-subsample, or '
            'label at generation before enabling it.')
    return [FomBenchmark.subsample_negatives(
        merged, merit_columns=('M20',), top_k=int(args.top_k),
        negative_rate=float(args.negative_rate), base_seed=int(args.seed))], True


def optimizer_options(args):
    """The opt_params overrides one invocation asks for.

    `search_seed_scheme` is the campaign-2 change and it is opt-in for a reason: it alters which
    candidates the search generates, so the shipped indexer keeps campaign 1's behaviour and the
    benchmark gets the reproducible one.
    """
    return {
        'prune_m20_threshold': float(args.prune_threshold),
        'prune_criterion_capture': True,
        'dump_candidates': True,
        'search_seed_scheme': 'per_entry_bravais',
        'search_base_seed': int(args.seed),
        }


def _commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=Path(mlindex.__path__[0]).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run(args):
    from mlindex.optimization.MPOptimizer import run_mp_bl
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers
    from mlindex.optimization.MPOptimizer import shutdown_mp_workers
    from mlindex.utilities.LatticeDegeneracy import is_degenerate

    condition = FomConditions.BY_KEY[args.condition]
    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    unknown = [bl for bl in bravais_lattices if bl not in BRAVAIS_LATTICES]
    if unknown:
        raise SystemExit(f"Unknown Bravais lattices: {', '.join(unknown)}")

    manifest = load_manifest(args.split_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = pd.concat(
        [FomPatterns.sample_entries(bl, args.n_entries_per_bl, args.seed,
                                    columns=FomPatterns.DUMP_READ_COLUMNS)
         for bl in bravais_lattices],
        ignore_index=True)

    if manifest is not None:
        # The manifest is the entry list, not merely a lookup. Re-deriving the population from a
        # sampling parameter is how the arms come apart: `sample_entries` draws
        # `rng.choice(size=n)`, so a run at a different `--n-entries-per-bl` selects a different
        # set and the core and mechanism arms stop being paired. Pre-flighting it here also turns
        # the usual cause -- a sampling parameter that has drifted -- into an abort rather than an
        # invented split (S07's handoff asks for exactly this).
        wanted = manifest
        if args.arm:
            if 'arm' not in manifest.columns:
                raise SystemExit(f'--arm given but {args.split_manifest} has no arm column')
            wanted = manifest[manifest['arm'].astype(str).str.contains(args.arm)]
        keep = entries['identifier'].isin(wanted.index)
        missing = int((~keep).sum())
        entries = entries[keep].reset_index(drop=True)
        unsampled = len(set(wanted.index) - set(entries['identifier']))
        print(f'manifest: {len(wanted)} entries wanted, {entries.shape[0]} sampled, '
              f'{missing} sampled entries not in the manifest, {unsampled} manifest entries '
              f'not reached by this sampling', flush=True)
        if entries.empty:
            raise SystemExit('the sampling and the manifest do not overlap at all; check '
                             '--n-entries-per-bl and --seed against the frozen manifest')

    if args.entry_ids_file:
        wanted = pd.read_csv(args.entry_ids_file)
        id_column = 'identifier' if 'identifier' in wanted.columns else 'entry_id'
        wanted_ids = set(wanted[id_column])
        missing = wanted_ids - set(entries['identifier'])
        if missing:
            raise SystemExit(f'{len(missing)} of {len(wanted_ids)} requested entries were not '
                             f'sampled, e.g. {sorted(missing)[:3]}.')
        entries = entries[entries['identifier'].isin(wanted_ids)].reset_index(drop=True)

    second_phase_pool = (FomPatterns.build_second_phase_pool(entries)
                         if condition.second_phase_lines > 0 else None)

    optimizers, processes, task_queues = setup_mp_optimizers(
        args.pool_size, FomPatterns.BROADENING_TAG, n_candidates_scale=1,
        seed=args.seed, options=optimizer_options(args))

    entry_rows, candidate_frames, predownsample_frames, failures = [], [], [], []
    merit_at_prune_names = ()
    started = time.time()
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
                failures.append({'identifier': entry['identifier'],
                                 'reason': 'contaminant_placement', 'detail': str(error)})
                continue

            context = {'entry_id': entry['identifier'],
                       'q2_digest': FomBenchmark.q2_digest(pattern.q2_obs),
                       'condition_bundle': condition.tag}
            records, predownsample_records = [], []
            for bravais_lattice in bravais_lattices:
                optimizer = optimizers[bravais_lattice]
                optimizer.dump_context = context
                run_mp_bl(optimizer, bravais_lattice, task_queues, q2=pattern.q2_obs,
                          zero_error=False, wavelength=None,
                          n_top=FomPatterns.N_TOP_CANDIDATES)
                records += optimizer.drain_candidate_dump()
                predownsample_records += optimizer.drain_predownsample_dump()

            candidates = FomBenchmark.records_to_frame(records)
            # What the k-th entry of `merit_at_prune` means. Recorded once, in the manifest,
            # rather than as a column per criterion -- so the column set does not change when the
            # merit set does, and a loader can always name what it is reading (C2-R-001).
            merit_at_prune_names = candidates.attrs.get('merit_at_prune_names',
                                                        merit_at_prune_names)
            candidate_frames.append(candidates)
            # SCHEMA.md specifies the pre-deduplication stream as a stratified subsample of
            # entries, and it is the expensive one -- ~59 000 rows per cell at threshold 0
            # against ~1 500 for the survivors. The driver wrote it for every entry, which is
            # most of the projected disk. Entries are taken in sampling order, and the sampling
            # is already lattice-stratified, so a prefix is a stratified subsample.
            if (args.predownsample_entries is None
                    or len(predownsample_frames) < args.predownsample_entries):
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
                degeneracy=is_degenerate(
                    np.asarray(entry['reindexed_unit_cell'], dtype=float),
                    entry['bravais_lattice']))
            row['pool_size_full'] = int(candidates.shape[0])
            entry_rows.append(row)
    finally:
        shutdown_mp_workers(processes, task_queues)

    tag = condition.tag
    candidate_frames, subsampled = subsample_or_refuse(candidate_frames, args)

    if entry_rows:
        FomBenchmark.write_entry_table(
            pd.DataFrame(entry_rows, columns=list(FomBenchmark.ENTRY_COLUMNS)), out_dir, tag)
    if candidate_frames:
        FomBenchmark.write_candidate_shard(
            pd.concat(candidate_frames, ignore_index=True), out_dir, tag)
    if predownsample_frames:
        FomBenchmark.write_predownsample_shard(
            pd.concat(predownsample_frames, ignore_index=True), out_dir, tag)
    if failures:
        with open(out_dir / f'failures_{tag}.json', 'w', encoding='utf-8') as handle:
            json.dump(failures, handle, indent=2, sort_keys=True)

    FomBenchmark.write_manifest(
        out_dir,
        bundle=tag,
        condition=condition.key,
        schema_version=FomBenchmark.SCHEMA_VERSION,
        commit=_commit_hash(),
        # An arm64-generated pool is not bit-reproducible on x86 and campaign 1's manifest
        # recorded the commit but not the machine (R9).
        arch=platform.machine(),
        platform=platform.platform(),
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        seed=args.seed,
        search_seed_scheme='per_entry_bravais',
        merit_at_prune_names=list(merit_at_prune_names),
        broadening_tag=FomPatterns.BROADENING_TAG,
        error_law=FomConditions.ERROR_LAW,
        n_peaks=FomPatterns.N_PEAKS,
        n_holdout=FomPatterns.N_HOLDOUT,
        n_top_candidates=FomPatterns.N_TOP_CANDIDATES,
        n_entries_per_bl=args.n_entries_per_bl,
        bravais_lattices=bravais_lattices,
        prune_threshold=float(args.prune_threshold),
        # What the run ACTUALLY did, not what it was asked to do. The previous version wrote
        # `subsampled: true` whenever the flag was absent, while the subsampler did not exist --
        # a manifest that misdescribes its own pool is worse than one that omits the field.
        top_k=int(args.top_k) if subsampled else None,
        negative_rate=float(args.negative_rate) if subsampled else None,
        subsample_rule=('correct + top_k union + bernoulli(negative_rate), '
                        'weight = 1/rate on the sampled class' if subsampled else None),
        subsampled=subsampled,
        predownsample_entries=args.predownsample_entries,
        split_manifest=args.split_manifest,
        n_entries=len(entry_rows),
        n_failures=len(failures),
        seconds_total=round(time.time() - started, 1),
        )
    print(f'wrote {out_dir}: {len(entry_rows)} entries, {len(failures)} failures, '
          f'{time.time() - started:.0f} s', flush=True)


def main(argv=None):
    args = _parse_args(argv)
    if args.print_tag:
        print(FomConditions.BY_KEY[args.condition].tag)
        return
    run(args)


if __name__ == '__main__':
    main()
