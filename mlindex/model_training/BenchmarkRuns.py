"""Running a benchmark arm, and reading two arms against each other.

An **arm** is one code configuration measured over one population: the same source crystals, the
same synthesised patterns, the same threshold, indexed and kept in full. Two arms are compared
paired, cell for cell, and the size of a difference is read against the run-to-run noise of the
search itself -- four arms that differ only in the search seed, which is what `floor_from_arms`
measures. A gate is quoted in multiples of that floor and never in percentage points.

Two seeds, and they do different jobs. `seed` fixes which crystals are drawn and what noise is put
on their peaks, so every arm of a comparison must share it or the arms differ in their data.
`search_seed` reaches the search alone. A floor moves the second and nothing else.
"""

import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import Benchmark
from mlindex.model_training import BenchmarkConditions
from mlindex.model_training import BenchmarkMetrics as metrics
from mlindex.model_training import BenchmarkPatterns
from mlindex.utilities.Digests import q2_digest
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

# Which condition bundles and which true lattices each population is made of. `hard` is the severe
# end of every axis, restricted to the three lattices where the correct cell is hardest to find.
POPULATIONS = {
    'general': {'bundles': BenchmarkConditions.tags(), 'bravais_lattices': BRAVAIS_LATTICES},
    'hard': {'bundles': BenchmarkConditions.HARD_BUNDLES,
             'bravais_lattices': ('aP', 'mP', 'mC')},
    }

REPORTING_SPLIT = 'fom-dev'

# How much of a condition bundle may fail to synthesise before the bundle itself is suspect. A
# second phase can only contaminate a pattern whose observed range its lines reach, and for the
# largest cells that range is low enough that some partners have nothing there -- so a handful of
# crystals legitimately have no pattern under that bundle. Losing many is a different thing: it
# means the condition is not producing what it claims, and a floor measured on the survivors would
# be a floor for a population nobody chose.
MAX_BUNDLE_FAILURE_FRACTION = 0.05

# How often a pool says where it has got to. A pattern takes tens of seconds, so this is
# a line every few minutes per pool.
PROGRESS_EVERY = 10


def file_digest(path):
    """The sha256 of a file, so a manifest can name the split it drew from and prove it."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def draw_entries(split_manifest, per_lattice, seed, split=REPORTING_SPLIT,
                 bravais_lattices=None):
    """The reporting sample: `per_lattice` crystals of each Bravais lattice, from the frozen split.

    Balanced across lattices rather than drawn in proportion, because every reported number is an
    unweighted mean over the fourteen: a proportional draw would make the headline mostly
    monoclinic and triclinic. A lattice with fewer than `per_lattice` crystals in the split
    contributes all of them -- cF and cI are capped this way and no seed changes that.

    **Read from the manifest, never re-derived by sampling the source datasets.** The split was
    frozen once; a draw that reproduces its per-lattice counts by chance does not reproduce its
    membership.
    """
    manifest = pd.read_parquet(split_manifest)
    manifest = manifest.loc[manifest['split'] == split]
    wanted = list(bravais_lattices if bravais_lattices is not None else BRAVAIS_LATTICES)
    chosen = []
    for lattice in wanted:
        block = manifest.loc[manifest['bravais_lattice'] == lattice]
        block = block.sort_values('identifier', kind='stable', ignore_index=True)
        if block.shape[0] > per_lattice:
            rng = np.random.default_rng(
                BenchmarkPatterns.derived_seed(f'draw:{lattice}', seed))
            positions = np.sort(rng.choice(block.shape[0], size=per_lattice, replace=False))
            block = block.iloc[positions].reset_index(drop=True)
        chosen.append(block)
    if not chosen:
        raise ValueError(f'No crystals in split {split!r} for lattices {wanted}.')
    return pd.concat(chosen, ignore_index=True)


def load_source_rows(chosen, dataset_directory=None):
    """The source datasets' rows for the drawn crystals, with the ground truth a label needs."""
    directory = (Path(dataset_directory) if dataset_directory is not None
                 else BenchmarkPatterns.DATASET_DIRECTORY)
    frames = []
    for lattice, block in chosen.groupby('bravais_lattice', sort=False):
        path = directory / f'dataset_{lattice}.parquet'
        if not path.is_file():
            raise FileNotFoundError(
                f'No source dataset for {lattice} at {path}. Point --dataset-directory at a tree '
                'that has it.')
        data = pd.read_parquet(path, columns=BenchmarkPatterns.TRUTH_COLUMNS)
        data = data.loc[data['identifier'].isin(set(block['identifier']))]
        found = set(data['identifier'])
        missing = sorted(set(block['identifier']) - found)
        if missing:
            raise ValueError(
                f'{len(missing)} {lattice} crystals of the frozen split are not in {path.name}, '
                f'first {missing[:3]}. The split and the source data have diverged; a run that '
                'quietly dropped them would report on a different population.')
        frames.append(data.sort_values('identifier', kind='stable', ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def _hkl_of(entry):
    """The crystal's reflection assignment, as one (n, 3) array."""
    tag = BenchmarkPatterns.BROADENING_TAG
    return np.stack([np.asarray(entry[f'reindexed_{axis}_{tag}'], dtype=float)
                     for axis in ('h', 'k', 'l')], axis=-1)


def entry_record(entry, condition, pattern, digest, pool_size_full=-1):
    """One row of the entry table: the pattern as synthesised, and the truth behind it."""
    from mlindex.utilities.ErrorAdder import q2_sigma_params

    intercept, slope = q2_sigma_params()
    full_peaks = np.asarray(entry[f'q2_{BenchmarkPatterns.BROADENING_TAG}'], dtype=float)
    return {
        'entry_id': entry['identifier'],
        'condition_bundle': condition.tag,
        'q2_digest': digest,
        'source_db': entry['database'],
        'split': REPORTING_SPLIT,
        'q2_obs': np.asarray(pattern.q2_obs, dtype=np.float64),
        'n_peaks_available': int(np.count_nonzero(full_peaks > 0)),
        'q2_error_multiplier': float(condition.error_multiplier),
        'intercept_scale': float(condition.intercept_scale),
        'error_law': BenchmarkConditions.ERROR_LAW,
        'error_law_params': np.array([intercept*condition.intercept_scale, slope],
                                     dtype=np.float64),
        'n_contaminants': int(condition.n_contaminants),
        'n_contaminants_achieved': int(pattern.n_contaminants_achieved),
        'n_dropout': int(condition.n_dropout),
        'n_dropout_achieved': int(pattern.n_dropout_achieved),
        'second_phase_lines': int(condition.second_phase_lines),
        'second_phase_achieved': int(pattern.n_second_phase_achieved),
        'second_phase_partner': pattern.second_phase_partner,
        'broadening_tag': BenchmarkPatterns.BROADENING_TAG,
        'xnn_true': np.asarray(entry['reindexed_xnn'], dtype=np.float64),
        'unit_cell_true': np.asarray(entry['reindexed_unit_cell'], dtype=np.float64),
        'volume_true': float(entry['reindexed_volume']),
        'bravais_lattice_true': entry['bravais_lattice'],
        'lattice_system_true': entry['lattice_system'],
        'spacegroup_true': entry['reindexed_spacegroup_symbol_hm'],
        'hkl_true': np.asarray(pattern.hkl_obs, dtype=np.int16).reshape(-1),
        'pool_size_full': int(pool_size_full),
        # Lattice degeneracy is not evaluated by this pass; the manifest's `degeneracy_rule`
        # says so, and no number here excludes a crystal on it. See P-Q-009.
        'is_degenerate': False,
        }


def _run_pool(part, pool_dir, source_rows, second_phase_pool, bundles, bravais_lattices,
              seed, search_seed, cut, pool_size):
    """Index one stripe of an arm's crystals and write it under `parts/NNN/`.

    Module level and taking only picklable arguments, because Windows and macOS spawn rather
    than fork. Every pool is given the SAME `search_seed`: the search re-keys itself per
    (peak list, Bravais lattice, rank), so an entry's candidates do not depend on which stripe
    it landed in and any subset of an arm reproduces on its own.
    """
    from mlindex.model_training.BenchmarkOptimizer import BenchmarkOptimizer
    from mlindex.optimization.MPOptimizer import (
        run_mp_bl, setup_mp_optimizers, shutdown_mp_workers)

    directory = Benchmark.part_dir(pool_dir, part)
    directory.mkdir(parents=True, exist_ok=True)
    optimizers, processes, task_queues = setup_mp_optimizers(
        pool_size, BenchmarkPatterns.BROADENING_TAG, 1, seed=search_seed,
        options={'prune_m20_threshold': float(cut)},
        optimizer_class=BenchmarkOptimizer)

    entry_rows = []
    failures = []
    started = time.perf_counter()
    done = 0
    total = len(bundles)*source_rows.shape[0]
    try:
        for bundle in bundles:
            condition = BenchmarkConditions.BY_TAG[bundle]
            records = []
            bundle_failures = 0
            for _, entry in source_rows.iterrows():
                try:
                    pattern = BenchmarkPatterns.prepare_peak_list(
                        entry, condition, seed, hkl=_hkl_of(entry),
                        second_phase_pool=second_phase_pool)
                except ContaminantPlacementError as error:
                    # This crystal has no pattern under this condition. Recorded and skipped
                    # rather than raised: one unlucky crystal must not cost a node-hours run, and
                    # the skip is deterministic, so every arm of a comparison skips the same one.
                    failures.append({'entry_id': entry['identifier'],
                                     'condition_bundle': bundle, 'reason': str(error)})
                    bundle_failures += 1
                    done += 1
                    continue
                q2 = np.asarray(pattern.q2_obs, dtype=np.float64)
                digest = q2_digest(q2)
                context = {'entry_id': entry['identifier'], 'condition_bundle': bundle,
                           'q2_digest': digest}
                pool_size_full = 0
                for lattice in bravais_lattices:
                    optimizer = optimizers[lattice]
                    optimizer.dump_context = context
                    run_mp_bl(optimizer, lattice, task_queues, q2=q2, zero_error=False,
                              wavelength=None, n_top=Benchmark.N_TOP_CANDIDATES)
                    drained = optimizer.drain()
                    records += drained
                    pool_size_full += sum(int(record['M20'].shape[0]) for record in drained)
                entry_rows.append(entry_record(entry, condition, pattern, digest,
                                               pool_size_full=pool_size_full))
                done += 1
                if done % PROGRESS_EVERY == 0 or done == total:
                    _report_progress(part, done, total, started)
            if bundle_failures > MAX_BUNDLE_FAILURE_FRACTION*source_rows.shape[0]:
                raise RuntimeError(
                    f'{bundle_failures} of {source_rows.shape[0]} crystals could not be given a '
                    f'pattern under {bundle}. That is past the point where this reads as a few '
                    'unlucky crystals; the condition is not producing what it claims, and a '
                    'number measured on the survivors would describe a population nobody chose.')
            _write_bundle(directory, bundle, records, entry_rows)
            del records
    finally:
        shutdown_mp_workers(processes, task_queues)

    Benchmark.write_entry_table(pd.DataFrame(entry_rows, columns=list(Benchmark.ENTRY_COLUMNS)),
                                directory)
    # Written even when empty, so a reader can tell "nothing failed" from "nobody looked".
    with open(directory/'failures.json', 'w', encoding='utf-8') as handle:
        json.dump(failures, handle, indent=2, sort_keys=True)
    return directory


def _report_progress(part, done, total, started):
    """How far a pool has got, and when it expects to finish.

    A pool writes nothing until it finishes a whole condition bundle, so without this a cluster
    job that will take hours is indistinguishable from one that hung in its first minute. Flushed,
    because the output is a file that somebody is tailing.
    """
    elapsed = time.perf_counter() - started
    rate = elapsed/done
    print(f'pool {part:02d}: {done}/{total} patterns, {rate:.1f} s each, '
          f'{(total - done)*rate/60:.1f} min left', flush=True)


def _write_bundle(directory, bundle, records, entry_rows):
    """One bundle's shards, written per Bravais lattice as soon as the bundle is finished.

    Written here rather than at the end of the stripe so that only one bundle's candidates are
    ever held: a triclinic pattern produces several thousand survivors and a bundle holds every
    crystal in the stripe.
    """
    if not records:
        return
    entries = pd.DataFrame(entry_rows)
    frame = Benchmark.records_to_frame(records)
    frame = Benchmark.label_frame(frame, entries)
    for lattice, block in frame.groupby('bravais_lattice', sort=False):
        Benchmark.write_candidate_shard(block.reset_index(drop=True), directory, bundle, lattice)


def run_arm(pool_dir, split_manifest, population='general', per_lattice=40, seed=12345,
            search_seed=12345, cut=1.5, pool_size=1, n_pools=1, bundles=None,
            dataset_directory=None, degeneracy_rule='not_evaluated'):
    """Generate one arm into `pool_dir`, and stamp it complete when every stripe has landed.

    Returns the manifest's metadata. The completion stamp is written last and only here: a killed
    run leaves valid shards behind, so an unstamped arm is refused by every reader.
    """
    import platform
    from multiprocessing import Process

    import mlindex

    pool_dir = Path(pool_dir)
    # Read before a single pattern is indexed, not at the end beside the rest of the manifest.
    # An arm takes hours, and a commit made while it runs would be recorded as the revision that
    # produced it -- which is both wrong and invisible, since the manifest still parses and the
    # identity check still compares it against other arms.
    commit = _commit()
    design = POPULATIONS[population]
    bundles = list(bundles or design['bundles'])
    unknown = [bundle for bundle in bundles if bundle not in BenchmarkConditions.BY_TAG]
    if unknown:
        raise ValueError(f'Unknown condition bundle(s) {unknown}. '
                         f'Known: {list(BenchmarkConditions.tags())}')

    chosen = draw_entries(split_manifest, per_lattice, seed,
                          bravais_lattices=design['bravais_lattices'])
    source_rows = load_source_rows(chosen, dataset_directory)
    # Built once, from every drawn crystal, and passed down: real contamination is not
    # lattice-matched, and a partner drawn from a stripe rather than from the whole arm would
    # depend on how the arm was divided.
    second_phase_pool = BenchmarkPatterns.build_second_phase_pool(source_rows)

    stripes = [source_rows.iloc[part::n_pools].reset_index(drop=True) for part in range(n_pools)]
    arguments = [(part, pool_dir, stripe, second_phase_pool, bundles, list(BRAVAIS_LATTICES),
                  seed, search_seed, cut, pool_size)
                 for part, stripe in enumerate(stripes) if stripe.shape[0]]
    if len(arguments) == 1:
        _run_pool(*arguments[0])
    else:
        processes = [Process(target=_run_pool, args=argument) for argument in arguments]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed = [process.exitcode for process in processes if process.exitcode]
        if failed:
            raise RuntimeError(
                f'{len(failed)} of {len(processes)} pools exited non-zero: {failed}. The arm is '
                'left unstamped, so nothing will read it as finished.')

    failures = _collect_failures(pool_dir)
    Benchmark.consolidate(pool_dir)
    metadata = {
        'population': population,
        'bundles': bundles,
        'bravais_lattices': list(BRAVAIS_LATTICES),
        'n_source_entries': int(source_rows.shape[0]),
        'n_patterns_refused': len(failures),
        'per_lattice': int(per_lattice),
        'seed': int(seed),
        'search_seed': int(search_seed),
        'prune_threshold': float(cut),
        'pool_size': int(pool_size),
        'n_pools': int(n_pools),
        'n_top_candidates': int(Benchmark.N_TOP_CANDIDATES),
        'broadening_tag': BenchmarkPatterns.BROADENING_TAG,
        'condition_set_digest': BenchmarkConditions.condition_set_digest(),
        'split_manifest': str(split_manifest),
        'split_manifest_sha256': file_digest(split_manifest),
        'degeneracy_rule': degeneracy_rule,
        'commit': commit,
        'arch': platform.machine(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'mlindex_version': getattr(mlindex, '__version__', 'unknown'),
        }
    Benchmark.write_manifest(pool_dir, **metadata)
    Benchmark.stamp_complete(pool_dir, n_source_entries=metadata['n_source_entries'],
                             n_bundles=len(bundles))
    return metadata


def _collect_failures(pool_dir):
    """Merge every pool's refused patterns into one file at the arm's root, before consolidation.

    Each pool writes its own, and `consolidate` removes the part directories -- so they have to be
    gathered first. The merged file is written even when empty: a reader can then tell "nothing was
    refused" from "nobody recorded it", which for a population that may be smaller than it looks is
    the difference that matters.
    """
    pool_dir = Path(pool_dir)
    failures = []
    for path in sorted((pool_dir/Benchmark.PART_DIR).glob('*/failures.json')):
        with open(path, encoding='utf-8') as handle:
            failures += json.load(handle)
        path.unlink()
    with open(pool_dir/'failures.json', 'w', encoding='utf-8') as handle:
        json.dump(failures, handle, indent=2, sort_keys=True)
    return failures


def _commit():
    """The revision the arm was generated at, or 'unknown' outside a checkout."""
    import subprocess

    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                cwd=str(Path(__file__).resolve().parent))
    except (OSError, ValueError):
        return 'unknown'
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


# ---------------------------------------------------------------------------
# Reading arms against each other
# ---------------------------------------------------------------------------


def select_entries(entries, limit, seed):
    """Cap the work by SOURCE CRYSTAL, not by row or by file.

    A cap applied per shard file gives each arm a different set of crystals whenever the shards
    differ in length, and the comparison stops being paired. Capping the crystal list keeps every
    condition of every chosen crystal.
    """
    if limit is None:
        return entries
    identifiers = np.sort(entries['entry_id'].unique())
    if identifiers.size <= limit:
        return entries
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(identifiers, size=limit, replace=False).tolist())
    return entries.loc[entries['entry_id'].isin(chosen)].reset_index(drop=True)


def report_arm(reductions, entries, top_n=10, depth='all'):
    """Per-scope metric tables, one per score, with the aggregate row first."""
    tables = {}
    for name, per_entry in reductions.items():
        flagged = metrics.derive_flags(per_entry, depth=depth, top_n=top_n)
        table = metrics.summarise_by_scope(flagged, entries=entries)
        table.insert(0, 'score', name)
        tables[name] = table
    return tables


def contrast_table(reductions, baseline, top_n=10, depth='all'):
    """Paired comparison of every score against the baseline, over the same patterns."""
    if baseline not in reductions:
        return pd.DataFrame()
    base = metrics.derive_flags(reductions[baseline], depth=depth, top_n=top_n)
    clusters = base['entry_id'].to_numpy()
    rows = []
    for name, per_entry in reductions.items():
        if name == baseline:
            continue
        arm = metrics.derive_flags(per_entry, depth=depth, top_n=top_n)
        for metric in ('top10', 'top1', 'found'):
            test = metrics.mcnemar(base[metric].to_numpy(), arm[metric].to_numpy())
            low, high = metrics.paired_delta_ci(
                base[metric].to_numpy(dtype=float), arm[metric].to_numpy(dtype=float), clusters)
            rows.append({'score': name, 'baseline': baseline, 'metric': metric,
                         'delta_pp': 100.0*test['delta'], 'ci_low_pp': 100.0*low,
                         'ci_high_pp': 100.0*high, 'n_discordant': test['n_discordant'],
                         'p_value': test['p_value'], 'n_pairs': test['n_pairs']})
    return pd.DataFrame(rows)


def floor_from_arms(arm_reductions, score, baseline, top_n=10, depth='all', lattices=None):
    """The run-to-run floor: how much the answer moves between arms that differ only in the seed.

    A gate is read in multiples of this, never in percentage points. The shift between two arms is
    clustered on the SOURCE CRYSTAL first: one crystal appears under every condition with
    correlated noise, so it is one draw and not several, and treating the rows as independent
    gives a floor that is too tight and every gate too permissive.

    The effect size beside it is the plain mean over arms. The campaign's version took it from the
    left arm of each ordered pair, which weights four arms 3:2:1:0 and drops the last one.

    `lattices` maps an entry to its TRUE Bravais lattice, and adds a row per lattice beside the
    aggregate. A per-lattice claim is read against that lattice's own floor and never against the
    aggregate: the spread is set mostly by how many crystals a lattice contributes, which is capped
    by the split for the scarce ones. No rescaling is applied, unlike the campaign's version --
    these arms are drawn from the same reporting sample the gates are read on, so the count the
    floor is measured at is already the count it will be used at.

    Returns one row per scope, aggregate first.
    """
    names = list(arm_reductions)
    contrasts = {}
    effects = []
    for arm in names:
        reductions = arm_reductions[arm]
        left = metrics.derive_flags(reductions[score], depth=depth, top_n=top_n)
        right = metrics.derive_flags(reductions[baseline], depth=depth, top_n=top_n)
        merged = left[['entry_id', 'condition_bundle', 'top10']].merge(
            right[['entry_id', 'condition_bundle', 'top10']],
            on=['entry_id', 'condition_bundle'], suffixes=('', '_base'), validate='1:1')
        merged['contrast'] = (merged['top10'].astype(float)
                              - merged['top10_base'].astype(float))
        contrasts[arm] = merged[['entry_id', 'condition_bundle', 'contrast']]
        effects.append(100.0*float(merged['contrast'].mean()))

    scopes = {'aggregate': None}
    if lattices is not None:
        for lattice in sorted(set(pd.Series(lattices).dropna())):
            scopes[lattice] = lattice

    rows = []
    for scope, lattice in scopes.items():
        rows.append(_floor_over(contrasts, effects, names, score, baseline, scope, lattice,
                                lattices))
    return pd.DataFrame(rows)


def _floor_over(contrasts, effects, names, score, baseline, scope, lattice, lattices):
    """The floor within one scope: the whole sample, or one true Bravais lattice."""
    if lattice is None:
        restricted = contrasts
        scope_effects = effects
    else:
        wanted = set(pd.Series(lattices)[pd.Series(lattices) == lattice].index)
        restricted = {name: frame.loc[frame['entry_id'].isin(wanted)]
                      for name, frame in contrasts.items()}
        scope_effects = [100.0*float(frame['contrast'].mean()) if frame.shape[0] else float('nan')
                         for frame in restricted.values()]

    floors = []
    for left_name, right_name in combinations(names, 2):
        joined = restricted[left_name].merge(restricted[right_name],
                                             on=['entry_id', 'condition_bundle'],
                                             suffixes=('_a', '_b'), validate='1:1')
        if joined.empty:
            continue
        joined['shift'] = joined['contrast_a'] - joined['contrast_b']
        clustered = joined.groupby('entry_id', as_index=False)['shift'].mean()
        shift = clustered['shift'].to_numpy(dtype=np.float64)*100.0
        if shift.size > 1:
            floors.append(float(np.std(shift, ddof=1)/np.sqrt(shift.size)))
    n_entries = int(pd.concat(restricted.values())['entry_id'].nunique()) if restricted else 0
    if not floors:
        if lattice is not None:
            # A lattice with one crystal has no spread to take; say so in the row rather than
            # refusing the whole table, and let the reader see which lattice it was.
            return {'score': score, 'baseline': baseline, 'metric': 'top10', 'scope': scope,
                    'n_entries': n_entries, 'n_arms': len(names), 'n_pairs': 0,
                    'effect_pp': float('nan'), 'floor_pp': float('nan'),
                    'standard_errors': float('nan')}
        raise ValueError(
            f'No arm pair produced a floor from {len(names)} arm(s). A floor needs at least two '
            'arms that share their patterns, and at least two source crystals to take a spread '
            'over.')
    floor_pp = float(np.mean(floors))
    effect_pp = float(np.nanmean(scope_effects))
    if lattice is None and not floor_pp > 0:
        raise ValueError(
            f'The measured floor is {floor_pp}, so every gate read against it would be infinite '
            f'or undefined. With {len(names)} arms and {len(floors)} pair(s) the arms did not '
            'differ anywhere, which at this size means the sample is too small rather than that '
            'the search is noiseless. Use four arms over the full reporting sample.')
    return {'score': score, 'baseline': baseline, 'metric': 'top10', 'scope': scope,
            'n_entries': n_entries, 'n_arms': len(names), 'n_pairs': len(floors),
            'effect_pp': effect_pp, 'floor_pp': floor_pp,
            'standard_errors': abs(effect_pp)/floor_pp if floor_pp else float('nan')}


# ---------------------------------------------------------------------------
# The merit sidecar
# ---------------------------------------------------------------------------

# The columns a merit sidecar carries, in the order `merit_set` returns them. `M20` is not among
# them: the candidate table already stores the value the pipeline computed, and a second column of
# the same name recomputed here would be one merit with two definitions.
SIDECAR_MERITS = ('M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap', 'n_cal')


def _hkl_reference(bravais_lattice, lattice_system):
    """The reference reflection list the search used for this lattice.

    Read from the resolved model tree rather than from the package, because `MLINDEX_MODELS_DIR`
    may point somewhere else and a merit computed against a different reference list is a
    different merit.
    """
    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir

    path = (_resolve_models_dir() / f'{lattice_system}_{BenchmarkPatterns.BROADENING_TAG}'
            / 'data' / f'hkl_ref_{bravais_lattice}.npy')
    if not path.is_file():
        raise FileNotFoundError(
            f'No reference reflections for {bravais_lattice} at {path}. The merit sidecar must '
            'use the same list the search did.')
    return np.load(path)


def merit_sidecar(pool_dir, bundles=None, bravais_lattices=None):
    """Score every stored candidate under every merit, beside the pool rather than inside it.

    One pass per (bundle, lattice) shard, reading `xnn` back and rebuilding the lines each
    candidate predicts. The merits are `FigureOfMerits.merit_set`, the same computation the
    at-prune capture uses, so `M_sym` has one definition in this repository.
    """
    from mlindex.utilities.FigureOfMerits import merit_set
    from mlindex.utilities.Q2Calculator import Q2Calculator

    pool_dir = Path(pool_dir)
    sidecar_dir = pool_dir / Benchmark.MERIT_SIDECAR
    entries = Benchmark.load_entries(pool_dir, columns=['entry_id', 'condition_bundle', 'q2_obs'])
    peaks = {(row.entry_id, row.condition_bundle): np.asarray(row.q2_obs, dtype=np.float64)
             for row in entries.itertuples()}

    written = []
    for bundle in (bundles or Benchmark.available_bundles(pool_dir)):
        for lattice, path in Benchmark.candidate_shards(pool_dir, bundle, bravais_lattices):
            frame = pd.read_parquet(path, columns=list(Benchmark.CANDIDATE_KEY)
                                    + ['lattice_system', 'xnn'])
            if frame.empty:
                continue
            lattice_system = frame['lattice_system'].iloc[0]
            calculator = Q2Calculator(
                lattice_system=lattice_system, hkl=_hkl_reference(lattice, lattice_system),
                tensorflow=False, representation='xnn')
            columns = {name: np.empty(frame.shape[0]) for name in SIDECAR_MERITS}
            for key, block in frame.groupby(['entry_id', 'condition_bundle'], sort=False):
                q2_obs = peaks[key]
                xnn = np.stack([np.asarray(row, dtype=np.float64) for row in block['xnn']])
                merits = merit_set(q2_obs, calculator.get_q2(xnn))
                positions = frame.index.get_indexer(block.index)
                for name in SIDECAR_MERITS:
                    columns[name][positions] = merits[name]
            sidecar = frame[list(Benchmark.CANDIDATE_KEY)].copy()
            for name in SIDECAR_MERITS:
                sidecar[name] = columns[name]
            written.append(Benchmark.write_candidate_shard(sidecar, sidecar_dir, bundle, lattice))
    if not written:
        raise FileNotFoundError(f'No candidate shards to score under {pool_dir}.')
    return written
