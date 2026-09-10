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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import Benchmark
from mlindex.model_training import BenchmarkConditions
from mlindex.model_training import BenchmarkMetrics as metrics
from mlindex.model_training import BenchmarkPatterns
from mlindex.utilities.Digests import q2_digest
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

# Which condition bundles and which true lattices each population is made of. `hard` is the severe
# end of every axis, restricted to the three lattices where the correct cell is hardest to find.
POPULATIONS = {
    'general': {'bundles': BenchmarkConditions.tags(), 'bravais_lattices': BRAVAIS_LATTICES},
    'hard': {'bundles': BenchmarkConditions.HARD_BUNDLES,
             'bravais_lattices': ('aP', 'mP', 'mC')},
    }

REPORTING_SPLIT = 'fom-dev'


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
    try:
        for bundle in bundles:
            condition = BenchmarkConditions.BY_TAG[bundle]
            records = []
            for _, entry in source_rows.iterrows():
                pattern = BenchmarkPatterns.prepare_peak_list(
                    entry, condition, seed, hkl=_hkl_of(entry),
                    second_phase_pool=second_phase_pool)
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
            _write_bundle(directory, bundle, records, entry_rows)
            del records
    finally:
        shutdown_mp_workers(processes, task_queues)

    Benchmark.write_entry_table(pd.DataFrame(entry_rows, columns=list(Benchmark.ENTRY_COLUMNS)),
                                directory)
    return directory


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

    Benchmark.consolidate(pool_dir)
    metadata = {
        'population': population,
        'bundles': bundles,
        'bravais_lattices': list(BRAVAIS_LATTICES),
        'n_source_entries': int(source_rows.shape[0]),
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
        'commit': _commit(),
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


def floor_from_arms(arm_reductions, score, baseline, top_n=10, depth='all'):
    """The run-to-run floor: how much the answer moves between arms that differ only in the seed.

    A gate is read in multiples of this, never in percentage points. The shift between two arms is
    clustered on the SOURCE CRYSTAL first: one crystal appears under every condition with
    correlated noise, so it is one draw and not several, and treating the rows as independent
    gives a floor that is too tight and every gate too permissive.

    The effect size beside it is the plain mean over arms. The campaign's version took it from the
    left arm of each ordered pair, which weights four arms 3:2:1:0 and drops the last one.
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

    floors = []
    for left_name, right_name in combinations(names, 2):
        joined = contrasts[left_name].merge(contrasts[right_name],
                                            on=['entry_id', 'condition_bundle'],
                                            suffixes=('_a', '_b'), validate='1:1')
        joined['shift'] = joined['contrast_a'] - joined['contrast_b']
        clustered = joined.groupby('entry_id', as_index=False)['shift'].mean()
        shift = clustered['shift'].to_numpy(dtype=np.float64)*100.0
        if shift.size > 1:
            floors.append(float(np.std(shift, ddof=1)/np.sqrt(shift.size)))
    floor_pp = float(np.mean(floors)) if floors else float('nan')
    effect_pp = float(np.mean(effects))
    return {'score': score, 'baseline': baseline, 'metric': 'top10', 'n_arms': len(names),
            'n_pairs': len(floors), 'effect_pp': effect_pp, 'floor_pp': floor_pp,
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
