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
from mlindex.utilities.Digests import derived_seed, file_digest, q2_digest, tree_digest
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

# The splits an arm may be drawn from. Choices are made on `fom-train` and reported on `fom-dev`;
# `fom-test` is sealed until the release benchmark and is not offered here.
SPLITS = ('fom-train', REPORTING_SPLIT)

# How much of a condition bundle may fail to synthesise before the bundle itself is suspect.
#
# A second phase can only contaminate a pattern whose observed range its lines reach, and for the
# largest cells that range is low enough that many partners have nothing there -- 5.6 % of the hard
# population observes nothing above q2 = 0.02 -- so a few crystals legitimately have no pattern
# under that bundle. The loss is not hidden: every refusal is in `failures.json` with its reason,
# and the count is in the manifest. So this guard exists to catch a condition that is broken
# outright, not to police a few percent, and it is set loosely on purpose.
#
# It is checked over the WHOLE ARM, never over one pool's stripe. At 128 pools a hard stripe is
# about three crystals, and a fraction of that is less than one -- so a per-stripe guard would
# abort the run on the first refusal anywhere, which is the opposite of what it is for.
MAX_BUNDLE_FAILURE_FRACTION = 0.20

# How often a pool says where it has got to. A pattern takes tens of seconds, so this is
# a line every few minutes per pool.
PROGRESS_EVERY = 10


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


def entry_record(entry, condition, pattern, digest, split, pool_size_full=-1):
    """One row of the entry table: the pattern as synthesised, and the truth behind it."""
    from mlindex.utilities.ErrorAdder import q2_sigma_params

    intercept, slope = q2_sigma_params()
    full_peaks = np.asarray(entry[f'q2_{BenchmarkPatterns.BROADENING_TAG}'], dtype=float)
    return {
        'entry_id': entry['identifier'],
        'condition_bundle': condition.tag,
        'q2_digest': digest,
        'source_db': entry['database'],
        'split': split,
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
              seed, search_seed, cut, pool_size, split):
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
        options={'prune_m20_threshold': float(cut), 'prune_criterion_capture': True},
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
                entry_rows.append(entry_record(entry, condition, pattern, digest, split,
                                               pool_size_full=pool_size_full))
                done += 1
                if done % PROGRESS_EVERY == 0 or done == total:
                    _report_progress(part, done, total, started)
            _write_bundle(directory, bundle, records, entry_rows)
            del records
    finally:
        shutdown_mp_workers(processes, task_queues)

    Benchmark.write_entry_table(pd.DataFrame(entry_rows, columns=list(Benchmark.ENTRY_COLUMNS)),
                                directory)
    # Scored here, on this stripe, rather than on the merged arm: a merged shard of a large arm
    # is millions of rows for one process, while the stripes are scored by every pool at once.
    if entry_rows:
        merit_sidecar(directory)
        feature_sidecar(directory)
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
    print(f'pool {part}: {done}/{total} patterns, {rate:.1f} s each, '
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


def ensemble_record():
    """Every lattice's candidate budget and generator fractions, as the manifest records them.

    Written out rather than left to the commit, so two arms can be compared on what they ran
    without reading the code they ran.
    """
    from mlindex.optimization.UtilitiesOptimizer import ENSEMBLE, lattice_budget
    return {'lattices': {lattice: {'n_candidates': lattice_budget(lattice, 1),
                                   'fractions': dict(ENSEMBLE[lattice]['fractions'])}
                         for lattice in BRAVAIS_LATTICES}}


def run_arm(pool_dir, split_manifest, split_sha256, population='general', per_lattice=40,
            seed=12345, search_seed=12345, cut=1.5, pool_size=1, n_pools=1, bundles=None,
            dataset_directory=None, degeneracy_rule='not_evaluated', split=REPORTING_SPLIT,
            true_lattices=None, shard=0, n_shards=1):
    """Generate one arm into `pool_dir`, or shard `shard` of `n_shards` of it.

    `split` is where the crystals are drawn from, and `true_lattices` narrows the population to
    crystals of those Bravais lattices. Every pattern is still searched in all fourteen lattices.
    `split_sha256` is the checksum the split manifest must have; anything else is refused before
    a crystal is drawn.

    Every shard draws the whole arm and builds its second-phase partners from all of it, then
    indexes `source_rows.iloc[shard::n_shards]`, so a pattern does not depend on how the arm was
    divided. Each of its `n_pools` processes writes its stripe, with both sidecars, under
    `parts/`, and the shard writes its stamp under `shards/` once they have all exited cleanly.
    `finalize_arm` merges the shards when every one is stamped; with one shard that happens here.

    Returns the arm's manifest metadata when finalized, otherwise the shard's stamp.
    """
    import platform
    from multiprocessing import Process

    import mlindex

    pool_dir = Path(pool_dir)
    if not 0 <= shard < n_shards:
        raise ValueError(f'shard {shard} is not one of 0..{n_shards - 1}.')
    design = POPULATIONS[population]
    if split not in SPLITS:
        raise ValueError(f'Unknown or sealed split {split!r}. Known: {SPLITS}')
    true_lattices = list(true_lattices or design['bravais_lattices'])
    outside = [lattice for lattice in true_lattices if lattice not in design['bravais_lattices']]
    if outside:
        raise ValueError(f'{outside} are not in the {population} population, which is '
                         f'{list(design["bravais_lattices"])}.')
    bundles = list(bundles or design['bundles'])
    unknown = [bundle for bundle in bundles if bundle not in BenchmarkConditions.BY_TAG]
    if unknown:
        raise ValueError(f'Unknown condition bundle(s) {unknown}. '
                         f'Known: {list(BenchmarkConditions.tags())}')
    split_manifest_sha256 = file_digest(split_manifest)
    if split_manifest_sha256 != split_sha256:
        raise ValueError(
            f'{split_manifest} has sha256 {split_manifest_sha256}, not the {split_sha256} it was '
            'expected to have. An arm drawn from a different split cannot be paired with any '
            'other, and could draw crystals from the sealed one.')
    _refuse_an_occupied_directory(pool_dir, shard, n_shards)
    # Read before a single pattern is indexed: an arm takes hours, and a commit made while it
    # runs would otherwise be recorded as the revision that produced it.
    provenance = _provenance()
    ensemble = ensemble_record()

    chosen = draw_entries(split_manifest, per_lattice, seed, split=split,
                          bravais_lattices=true_lattices)
    source_rows = load_source_rows(chosen, dataset_directory)
    # Built once, from every drawn crystal, and passed down: real contamination is not
    # lattice-matched, and a partner drawn from a stripe or a shard rather than from the whole
    # arm would depend on how the arm was divided.
    second_phase_pool = BenchmarkPatterns.build_second_phase_pool(source_rows)

    shard_rows = source_rows.iloc[shard::n_shards].reset_index(drop=True)
    stripes = [shard_rows.iloc[part::n_pools].reset_index(drop=True) for part in range(n_pools)]
    arguments = [(f'{shard:03d}_{part:03d}', pool_dir, stripe, second_phase_pool, bundles,
                  list(BRAVAIS_LATTICES), seed, search_seed, cut, pool_size, split)
                 for part, stripe in enumerate(stripes) if stripe.shape[0]]
    if len(arguments) == 1:
        _run_pool(*arguments[0])
    elif arguments:
        processes = [Process(target=_run_pool, args=argument) for argument in arguments]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed = [process.exitcode for process in processes if process.exitcode]
        if failed:
            raise RuntimeError(
                f'{len(failed)} of {len(processes)} pools exited non-zero: {failed}. The shard is '
                'left unstamped, so nothing will read it as finished.')

    stamp = {
        'population': population,
        'bundles': bundles,
        'bravais_lattices': list(BRAVAIS_LATTICES),
        'split': split,
        'true_lattices': true_lattices,
        'n_source_entries': int(source_rows.shape[0]),
        'per_lattice': int(per_lattice),
        'seed': int(seed),
        'search_seed': int(search_seed),
        'prune_threshold': float(cut),
        'pool_size': int(pool_size),
        'ensemble': ensemble,
        'n_pools': int(n_pools),
        'n_shards': int(n_shards),
        'n_top_candidates': int(Benchmark.N_TOP_CANDIDATES),
        'broadening_tag': BenchmarkPatterns.BROADENING_TAG,
        'condition_set_digest': BenchmarkConditions.condition_set_digest(),
        'split_manifest': str(split_manifest),
        'split_manifest_sha256': split_manifest_sha256,
        'degeneracy_rule': degeneracy_rule,
        **provenance,
        'arch': platform.machine(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'mlindex_version': getattr(mlindex, '__version__', 'unknown'),
        'shard': int(shard),
        'n_shard_entries': int(shard_rows.shape[0]),
        'failures': _collect_failures(pool_dir, shard),
        }
    path = pool_dir / Benchmark.SHARD_DIR / f'{shard:03d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(stamp, handle, indent=2, sort_keys=True)
    if n_shards == 1:
        return finalize_arm(pool_dir)
    return stamp


# What a shard's stamp carries that is its own rather than the arm's.
SHARD_FIELDS = ('shard', 'n_shard_entries', 'failures')


def finalize_arm(pool_dir):
    """Merge a generated arm's shards into one pool and stamp it complete.

    Refused unless every shard of the arm is stamped, every stamp describes the same arm, and
    together they hold every drawn crystal exactly once. A shard that died leaves valid stripes
    behind and no stamp, so its absence is the only thing that tells it apart from one that
    finished. Returns the manifest's metadata.
    """
    pool_dir = Path(pool_dir)
    if (pool_dir / Benchmark.COMPLETION_NAME).exists():
        raise FileExistsError(f'{pool_dir} is already finalized.')
    paths = sorted((pool_dir / Benchmark.SHARD_DIR).glob('*.json'))
    if not paths:
        raise FileNotFoundError(f'No shard stamps under {pool_dir / Benchmark.SHARD_DIR}.')
    stamps = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            stamps.append(json.load(handle))
    arm = {name: value for name, value in stamps[0].items() if name not in SHARD_FIELDS}
    for stamp in stamps[1:]:
        differing = sorted(name for name in set(arm) | set(stamp)
                           if name not in SHARD_FIELDS and stamp.get(name) != arm.get(name))
        if differing:
            raise ValueError(
                f'Shard {stamp["shard"]} differs from shard {stamps[0]["shard"]} in '
                f'{differing}; they are not shards of one arm.')
    found = sorted(stamp['shard'] for stamp in stamps)
    missing = sorted(set(range(arm['n_shards'])) - set(found))
    if missing or len(found) != arm['n_shards']:
        raise RuntimeError(
            f'Shards {missing} of {arm["n_shards"]} have no stamp (found {found}). The arm is '
            'left unfinalized: a merge now would describe the crystals that happened to finish.')
    n_entries = sum(stamp['n_shard_entries'] for stamp in stamps)
    if n_entries != arm['n_source_entries']:
        raise RuntimeError(f'The shards hold {n_entries} crystals and the arm drew '
                           f'{arm["n_source_entries"]}.')

    failures = [failure for stamp in stamps for failure in stamp['failures']]
    _refuse_a_broken_bundle(failures, arm['bundles'], arm['n_source_entries'])
    Benchmark.consolidate(pool_dir)
    # Written even when empty, so a reader can tell "nothing was refused" from "nobody recorded".
    with open(pool_dir/'failures.json', 'w', encoding='utf-8') as handle:
        json.dump(failures, handle, indent=2, sort_keys=True)
    metadata = dict(arm, n_patterns_refused=len(failures),
                    n_shard_entries=[stamp['n_shard_entries'] for stamp in stamps])
    Benchmark.write_manifest(pool_dir, **metadata)
    Benchmark.stamp_complete(pool_dir, n_source_entries=arm['n_source_entries'],
                             n_bundles=len(arm['bundles']), n_patterns_refused=len(failures),
                             n_shards=arm['n_shards'])
    return metadata


def _refuse_an_occupied_directory(pool_dir, shard=0, n_shards=1):
    """Refuse to generate where an earlier run left something this one would be merged with.

    A shard that died leaves its finished pools' stripes under `parts/`, and a re-run writing
    beside them would consolidate stale stripes with new ones. Nothing in the result would say
    which run a shard came from. An unsharded arm needs an empty directory; a shard needs no
    finished arm there and nothing of its own.
    """
    pool_dir = Path(pool_dir)
    if not pool_dir.exists():
        return
    if n_shards == 1:
        occupants = sorted(path.name for path in pool_dir.iterdir())
    else:
        occupants = sorted(
            [path.name for path in (pool_dir/Benchmark.COMPLETION_NAME,
                                    pool_dir/Benchmark.MANIFEST_NAME,
                                    pool_dir/Benchmark.SHARD_DIR/f'{shard:03d}.json')
             if path.exists()]
            + [path.name for path in (pool_dir/Benchmark.PART_DIR).glob(f'{shard:03d}_*')])
    if occupants:
        raise FileExistsError(
            f'{pool_dir} already holds {occupants[:4]}{"..." if len(occupants) > 4 else ""}. '
            'Generating into it would mix this run with whatever is there -- a failed run leaves '
            'its finished pools behind, and consolidation cannot tell them apart. Remove it and '
            're-run.')


def _refuse_a_broken_bundle(failures, bundles, n_crystals):
    """Stop an arm whose condition could not be applied to most of its crystals.

    Checked over the whole arm rather than over a pool's stripe: at 128 pools a stripe is a handful
    of crystals, and any fraction of a handful rounds to "abort on the first refusal".
    """
    for bundle in bundles:
        refused = sum(1 for failure in failures if failure['condition_bundle'] == bundle)
        if refused > MAX_BUNDLE_FAILURE_FRACTION*n_crystals:
            raise RuntimeError(
                f'{refused} of {n_crystals} crystals could not be given a pattern under {bundle} '
                f'({100*refused/n_crystals:.1f} %). Past this point it is the condition that is '
                'wrong rather than a few crystals being unlucky, and a number measured on the '
                'survivors would describe a population nobody chose. See failures.json.')


def _collect_failures(pool_dir, shard):
    """One shard's refused patterns, gathered from its pools before consolidation removes them."""
    pool_dir = Path(pool_dir)
    failures = []
    for path in sorted((pool_dir/Benchmark.PART_DIR).glob(f'{shard:03d}_*/failures.json')):
        with open(path, encoding='utf-8') as handle:
            failures += json.load(handle)
        path.unlink()
    return failures


def _git(*arguments):
    """A git command's output in the checkout this module is in, or None outside one."""
    import subprocess

    try:
        result = subprocess.run(['git', *arguments], capture_output=True, text=True,
                                cwd=str(Path(__file__).resolve().parent))
    except (OSError, ValueError):
        return None
    return result.stdout if result.returncode == 0 else None


def committed_checkout():
    """The commit this checkout is at, refused unless no tracked file is modified.

    A number has to be attributable to code somebody can read. The model tree is excluded from the
    check; a caller that reads model files identifies them by their own digest.
    """
    commit = (_git('rev-parse', 'HEAD') or '').strip()
    if not commit:
        raise RuntimeError('This has to run from a git checkout, so that its output can name the '
                           'commit that produced it.')
    modified = _git('status', '--porcelain', '--untracked-files=no', '--', ':(top)',
                    ':(top,exclude)mlindex/models')
    if modified is None or modified.strip():
        raise RuntimeError(
            f'The checkout has uncommitted changes to tracked files:\n{modified}\nCommit them '
            f'first; the output would name {commit[:10]}, which is not the code that ran.')
    return commit


def _provenance():
    """The code and the model files an arm is generated from: the commit, and a digest over the
    model directories the search reads."""
    from importlib.resources import files

    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
    from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM

    commit = committed_checkout()
    models_dir = _resolve_models_dir()
    systems = sorted(set(BL_TO_LATTICE_SYSTEM.values()))
    models_digest, n_model_files = tree_digest(
        models_dir, [f'{system}_{BenchmarkPatterns.BROADENING_TAG}' for system in systems])
    metadata = json.loads(files('mlindex').joinpath('model_metadata.json').read_text(
        encoding='utf-8'))
    return {'commit': commit, 'models_dir': str(models_dir), 'models_digest': models_digest,
            'n_model_files': n_model_files, 'model_revision': metadata['model_revision']}


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


# The rule P09b fixed before any arm ran: a change is read in multiples of the measured
# run-to-run floor, and only a move of more than this many either way is a result.
VERDICT_STANDARD_ERRORS = 2.0


def verdict(standard_errors):
    """'helps', 'hurts' or 'does not matter much' for a difference in floor multiples.

    Empty when no floor was given, because a difference in percentage points alone cannot be read.
    """
    if not np.isfinite(standard_errors):
        return ''
    if standard_errors > VERDICT_STANDARD_ERRORS:
        return 'helps'
    if standard_errors < -VERDICT_STANDARD_ERRORS:
        return 'hurts'
    return 'does not matter much'


def arm_contrast(arm_reductions, score, reference, top_n=10, depth='all', lattices=None,
                 floors=None):
    """One arm's outcome against another's, under the same score, paired on the pattern.

    This is the comparison every session after P04b actually needs: arm A is the code before a
    change and arm B after it, and the question is whether B finds and ranks the correct cell more
    often. `contrast_table` answers a different one -- two scores inside one arm -- and
    `floor_from_arms` a third, the spread of a score-contrast across arms that differ only by seed.

    `floors` maps (metric, scope) to that metric's measured run-to-run floor in that scope, and
    turns the difference into the multiple of it that a gate is actually read in; a metric with no
    measured floor is left unread. Without it the size is reported in
    percentage points and the caller is told, rather than left to assume the points mean something.

    Returns one row per metric per scope, aggregate first.
    """
    if reference not in arm_reductions:
        raise ValueError(f'No arm named {reference!r} to compare against; '
                         f'have {sorted(arm_reductions)}.')
    others = [name for name in arm_reductions if name != reference]
    if not others:
        raise ValueError('An arm contrast needs two arms.')

    base = metrics.derive_flags(arm_reductions[reference][score], depth=depth, top_n=top_n)
    scopes = {'aggregate': None}
    if lattices is not None:
        for lattice in sorted(set(pd.Series(lattices).dropna())):
            scopes[lattice] = lattice

    rows = []
    for name in others:
        arm = metrics.derive_flags(arm_reductions[name][score], depth=depth, top_n=top_n)
        merged = base[['entry_id', 'condition_bundle', 'found', 'top10', 'top1']].merge(
            arm[['entry_id', 'condition_bundle', 'found', 'top10', 'top1']],
            on=['entry_id', 'condition_bundle'], suffixes=('_base', '_arm'), validate='1:1')
        if lattices is not None:
            merged['lattice'] = [pd.Series(lattices).get(entry)
                                 for entry in merged['entry_id']]
        for scope, lattice in scopes.items():
            block = merged if lattice is None else merged.loc[merged['lattice'] == lattice]
            if block.empty:
                continue
            for metric in ('found', 'top10', 'top1'):
                left = block[f'{metric}_base'].to_numpy()
                right = block[f'{metric}_arm'].to_numpy()
                test = metrics.mcnemar(left, right)
                low, high = metrics.paired_delta_ci(
                    left.astype(float), right.astype(float),
                    block['entry_id'].to_numpy())
                floor = (floors or {}).get((metric, scope))
                standard_errors = (100.0*test['delta']/floor) if floor else float('nan')
                rows.append({
                    'arm': name, 'reference': reference, 'score': score, 'scope': scope,
                    'metric': metric, 'n_pairs': test['n_pairs'],
                    'reference_pct': 100.0*left.mean(), 'arm_pct': 100.0*right.mean(),
                    'delta_pp': 100.0*test['delta'],
                    'ci_low_pp': 100.0*low, 'ci_high_pp': 100.0*high,
                    'n_discordant': test['n_discordant'],
                    'n_rescued': int(np.sum(~left.astype(bool) & right.astype(bool))),
                    'n_broken': int(np.sum(left.astype(bool) & ~right.astype(bool))),
                    'p_value': test['p_value'],
                    'floor_pp': floor,
                    'standard_errors': standard_errors,
                    'verdict': verdict(standard_errors),
                    })
    return pd.DataFrame(rows)


def floors_from_table(table, score=None):
    """{(metric, scope): floor_pp} from a `floor.csv` this harness wrote, to read a contrast against.

    Keyed by metric as well as scope because a floor is measured for one metric -- top-10, as the
    floor stage writes it -- and nothing says another metric's run-to-run spread is the same, so a
    top-1 difference read against a top-10 floor would be read against the wrong noise.
    """
    if score is not None:
        table = table.loc[table['score'] == score]
    return {(row.metric, row.scope): float(row.floor_pp) for row in table.itertuples()
            if np.isfinite(row.floor_pp)}


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

    A merit has to be computed on exactly what the pipeline computed its M20 on, and two things
    make that narrower than "the cell and the peak list":

    * **The peak list is truncated to what the lattice system is fitted on** -- ten lines for
      cubic, twenty for the rest. The entry table stores the full twenty.
    * **The reference lines are the candidate's own extinction group's**, not the whole list.
      `assign_extinction_group` picks the group that maximises M20 and the stored M20 is the one
      with that group's absences removed. Scoring against the full list answers a different
      question, and silently: the numbers are all finite and ordered.

    Both are checked rather than trusted. The recomputed M20 must equal the stored M20 candidate
    for candidate, and a mismatch raises -- a merit sidecar that disagrees with its own pool is
    indistinguishable from a measurement once it is written.
    """
    from mlindex.utilities.FigureOfMerits import merit_set
    from mlindex.utilities.Q2Calculator import Q2Calculator
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    pool_dir = Path(pool_dir)
    sidecar_dir = pool_dir / Benchmark.MERIT_SIDECAR
    entries = Benchmark.load_entries(pool_dir, columns=['entry_id', 'condition_bundle', 'q2_obs'])
    peaks = {(row.entry_id, row.condition_bundle): np.asarray(row.q2_obs, dtype=np.float64)
             for row in entries.itertuples()}

    written = []
    for bundle in (bundles or Benchmark.available_bundles(pool_dir)):
        for lattice, path in Benchmark.candidate_shards(pool_dir, bundle, bravais_lattices):
            frame = pd.read_parquet(path, columns=list(Benchmark.CANDIDATE_KEY)
                                    + ['lattice_system', 'xnn', 'spacegroup', 'n_peaks', 'M20'])
            if frame.empty:
                continue
            lattice_system = frame['lattice_system'].iloc[0]
            n_peaks = int(frame['n_peaks'].iloc[0])
            by_group = get_spacegroup_hkl_ref(_hkl_reference(lattice, lattice_system),
                                              bravais_lattice=lattice)
            calculators = {}
            columns = {name: np.empty(frame.shape[0]) for name in SIDECAR_MERITS}
            recomputed = np.empty(frame.shape[0])
            for (entry_id, bundle_tag, spacegroup), block in frame.groupby(
                    ['entry_id', 'condition_bundle', 'spacegroup'], sort=False):
                if spacegroup not in calculators:
                    calculators[spacegroup] = Q2Calculator(
                        lattice_system=lattice_system, hkl=by_group[spacegroup],
                        tensorflow=False, representation='xnn')
                q2_obs = peaks[(entry_id, bundle_tag)][:n_peaks]
                xnn = np.stack([np.asarray(row, dtype=np.float64) for row in block['xnn']])
                merits = merit_set(q2_obs, calculators[spacegroup].get_q2(xnn))
                positions = frame.index.get_indexer(block.index)
                recomputed[positions] = merits['M20']
                for name in SIDECAR_MERITS:
                    columns[name][positions] = merits[name]
            _refuse_a_disagreeing_sidecar(recomputed, frame['M20'].to_numpy(dtype=np.float64),
                                          bundle, lattice)
            sidecar = frame[list(Benchmark.CANDIDATE_KEY)].copy()
            for name in SIDECAR_MERITS:
                sidecar[name] = columns[name]
            written.append(Benchmark.write_candidate_shard(sidecar, sidecar_dir, bundle, lattice))
    if not written:
        raise FileNotFoundError(f'No candidate shards to score under {pool_dir}.')
    return written


# ---------------------------------------------------------------------------
# The feature sidecar
# ---------------------------------------------------------------------------

# Per-candidate inputs of the learned ranker beyond the merit sidecar's. The structural and
# probation columns are computed against the candidate's own extinction group's reference list;
# the absence counts against the lattice's full list, because they count what the group removes.
STRUCTURAL_FEATURES = ('zone_dominance', 'V_over_Vcrit', 'M_werner_max', 'N_cal_full',
                       'delta_dewolff61', 'n_dewolff61')
PROBATION_FEATURES = ('M_wu', 'M_1', 'F_N_q')
ABSENCE_FEATURES = ('n_absent_extra', 'n_absent_extra_in_range', 'n_ref_in_range',
                    'n_groups_searched')
SIDECAR_FEATURES = STRUCTURAL_FEATURES + PROBATION_FEATURES + ABSENCE_FEATURES

# The precision floor in Werner's critical volume. It scales `V_over_Vcrit` and `M_werner_max`
# by the same factor for every candidate.
WERNER_G_MIN = 1.0


def _candidate_features(q2_obs, xnn, lattice_system, bravais_lattice, calculator):
    """The structural and probation features, and M20, for candidates of one extinction group.

    `calculator` holds that group's reference list. Returns a dict of arrays keyed by
    `STRUCTURAL_FEATURES + PROBATION_FEATURES` and 'M20'.
    """
    from mlindex.utilities.FigureOfMerits import (
        _sorted_lines_in_range, get_delta_dewolff61, get_F_N, get_M_1, get_M20, get_M_wu,
        get_multiplicity_taupin88, get_n_dewolff61, get_N_cal, get_V_over_Vcrit,
        get_zone_dominance)
    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.UnitCellTools import (
        get_reciprocal_unit_cell_from_xnn, get_unit_cell_volume)

    q2_ref_calc = calculator.get_q2(xnn)
    q2_calc = np.take_along_axis(q2_ref_calc, fast_assign(q2_obs, q2_ref_calc), axis=1)
    cutoff = q2_calc[:, -1]
    reciprocal_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system)
    volume = 1/np.maximum(get_unit_cell_volume(
        reciprocal_cell, partial_unit_cell=True, lattice_system=lattice_system), 1e-300)
    d_n = 1/np.sqrt(np.maximum(cutoff, 1e-300))
    over_critical, m_max = get_V_over_Vcrit(
        volume, d_n, WERNER_G_MIN, get_multiplicity_taupin88(bravais_lattice)[0])
    sorted_lines = _sorted_lines_in_range(q2_ref_calc, cutoff)
    return {
        'zone_dominance': get_zone_dominance(xnn, lattice_system),
        'V_over_Vcrit': over_critical,
        'M_werner_max': m_max,
        'N_cal_full': get_N_cal(q2_ref_calc, np.zeros(xnn.shape[0]), cutoff),
        'delta_dewolff61': np.mean(
            get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice), axis=1),
        'n_dewolff61': get_n_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)[:, -1],
        'M_wu': get_M_wu(q2_obs, q2_calc, q2_ref_calc, sorted_lines=sorted_lines),
        'M_1': get_M_1(q2_obs, q2_calc, q2_ref_calc, sorted_lines=sorted_lines),
        'F_N_q': get_F_N(q2_obs, q2_calc, q2_ref_calc)[1],
        # get_M20 writes into its reference array, so it gets a copy and goes last.
        'M20': get_M20(q2_obs, q2_calc, q2_ref_calc.copy()),
        }


def feature_sidecar(pool_dir, bundles=None, bravais_lattices=None):
    """Write `SIDECAR_FEATURES` for every stored candidate, beside the pool.

    The peak list is truncated to the lattice's own count and the structural features use the
    candidate's own extinction group's lines, as in `merit_sidecar`. The absence counts use the
    lattice's full reference list and its own cutoff: the line the last peak is assigned to.

    Two checks against what the pipeline wrote, either of which refuses the shard: the full
    reference list must be as long as the one the search used (`hkl_ref_length`), and the M20
    recomputed on each candidate's own group must equal the stored M20.
    """
    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.Q2Calculator import Q2Calculator
    from mlindex.utilities.SpaceGroups import count_absences_in_range, get_spacegroup_keep_masks

    pool_dir = Path(pool_dir)
    sidecar_dir = pool_dir / Benchmark.FEATURE_SIDECAR
    entries = Benchmark.load_entries(pool_dir, columns=['entry_id', 'condition_bundle', 'q2_obs'])
    peaks = {(row.entry_id, row.condition_bundle): np.asarray(row.q2_obs, dtype=np.float64)
             for row in entries.itertuples()}

    written = []
    for bundle in (bundles or Benchmark.available_bundles(pool_dir)):
        for lattice, path in Benchmark.candidate_shards(pool_dir, bundle, bravais_lattices):
            frame = pd.read_parquet(path, columns=list(Benchmark.CANDIDATE_KEY) + [
                'lattice_system', 'xnn', 'spacegroup', 'n_peaks', 'hkl_ref_length', 'M20'])
            if frame.empty:
                continue
            lattice_system = frame['lattice_system'].iloc[0]
            n_peaks = int(frame['n_peaks'].iloc[0])
            hkl_ref = _hkl_reference(lattice, lattice_system)
            stored_length = frame['hkl_ref_length'].unique()
            if stored_length.tolist() != [hkl_ref.shape[0]]:
                raise ValueError(
                    f'{bundle}/{lattice}: the search used a reference list of '
                    f'{stored_length.tolist()} lines and this models tree has '
                    f'{hkl_ref.shape[0]}. The features would describe different reflections.')
            keep_masks = get_spacegroup_keep_masks(hkl_ref, bravais_lattice=lattice)
            full = Q2Calculator(lattice_system=lattice_system, hkl=hkl_ref, tensorflow=False,
                                representation='xnn')
            calculators = {}
            columns = {name: np.empty(frame.shape[0]) for name in SIDECAR_FEATURES}
            recomputed = np.empty(frame.shape[0])
            columns['n_groups_searched'][:] = len(keep_masks)
            for (entry_id, bundle_tag), entry in frame.groupby(
                    ['entry_id', 'condition_bundle'], sort=False):
                q2_obs = peaks[(entry_id, bundle_tag)][:n_peaks]
                xnn = np.stack([np.asarray(row, dtype=np.float64) for row in entry['xnn']])
                rows = frame.index.get_indexer(entry.index)
                q2_ref_calc = full.get_q2(xnn)
                cutoff = np.take_along_axis(
                    q2_ref_calc, fast_assign(q2_obs, q2_ref_calc), axis=1)[:, -1]
                spacegroups = entry['spacegroup'].to_numpy()
                for spacegroup in pd.unique(spacegroups):
                    local = np.flatnonzero(spacegroups == spacegroup)
                    keep = keep_masks[spacegroup]
                    removed, in_range = count_absences_in_range(
                        q2_ref_calc[local], keep, cutoff[local])
                    columns['n_absent_extra'][rows[local]] = np.count_nonzero(~keep)
                    columns['n_absent_extra_in_range'][rows[local]] = removed
                    columns['n_ref_in_range'][rows[local]] = in_range
                    if spacegroup not in calculators:
                        calculators[spacegroup] = Q2Calculator(
                            lattice_system=lattice_system, hkl=hkl_ref[keep], tensorflow=False,
                            representation='xnn')
                    values = _candidate_features(q2_obs, xnn[local], lattice_system, lattice,
                                                 calculators[spacegroup])
                    recomputed[rows[local]] = values.pop('M20')
                    for name, value in values.items():
                        columns[name][rows[local]] = value
            _refuse_a_disagreeing_sidecar(recomputed, frame['M20'].to_numpy(dtype=np.float64),
                                          bundle, lattice)
            sidecar = frame[list(Benchmark.CANDIDATE_KEY)].copy()
            for name in SIDECAR_FEATURES:
                sidecar[name] = columns[name]
            for name in ABSENCE_FEATURES:
                sidecar[name] = sidecar[name].astype(np.int64)
            written.append(Benchmark.write_candidate_shard(sidecar, sidecar_dir, bundle, lattice))
    if not written:
        raise FileNotFoundError(f'No candidate shards to score under {pool_dir}.')
    return written


def _refuse_a_disagreeing_sidecar(recomputed, stored, bundle, lattice):
    """The merits must be computed on what the pipeline computed its own M20 on.

    M20 is the one merit that exists on both sides, so recomputing it is a free check that the
    peak list, the reference lines and the cell are all the ones the search used. Every other merit
    in the sidecar rides on the same three, so if M20 agrees they are being asked the same
    question.
    """
    difference = np.abs(recomputed - stored)
    worst = float(np.nanmax(difference)) if difference.size else 0.0
    if worst > 1e-6:
        n_bad = int(np.count_nonzero(difference > 1e-6))
        raise ValueError(
            f'The sidecar for {bundle}/{lattice} disagrees with the pool it sits beside: '
            f'{n_bad} of {difference.size} candidates recompute a different M20, worst '
            f'{worst:.4g}. The merits are being computed on a different peak list, a different '
            'reference list or a different cell from the one the search used, and every column '
            'written here would be a plausible number answering the wrong question.')


# ---------------------------------------------------------------------------
# The true cell, where the search did not find it
# ---------------------------------------------------------------------------

TRUTH_POOL_STAMP = 'truth_pool.json'
# The candidate id an added true cell carries; every searched candidate's is >= 0.
TRUTH_CANDIDATE_ID = -1


def refine_true_cell(q2_obs, unit_cell_true, bravais_lattice, lattice_system, hkl_ref, opt_params,
                     rng):
    """The true cell as the search would have finished it, had it been found.

    Starts from the true cell in its lattice's partial form and applies the search's last steps
    in the search's order: one Gauss-Newton refinement (kept only where it raises M20), the
    standard setting, the extinction group that maximises M20, and the indexed-peak count. The
    M20 cut and the off-by-two copies are not applied. Returns the `Candidates` holding one cell.
    """
    from mlindex.optimization.Candidates import Candidates
    from mlindex.utilities.UnitCellTools import get_partial_unit_cell, get_xnn_from_unit_cell

    partial = get_partial_unit_cell(np.asarray(unit_cell_true, dtype=np.float64),
                                    lattice_system=lattice_system)
    xnn = get_xnn_from_unit_cell(partial[np.newaxis], partial_unit_cell=True,
                                 lattice_system=lattice_system)
    candidates = Candidates(
        q2_obs=q2_obs, xnn=xnn, hkl_ref=hkl_ref, lattice_system=lattice_system,
        bravais_lattice=bravais_lattice, opt_params=opt_params, rng=rng, fom=None,
        zero_error=False, wavelength=None)
    candidates.refine_cell()
    candidates.standardize_cell()
    candidates.assign_extinction_group()
    candidates.calculate_peaks_indexed()
    return candidates


def truth_pool(pool_dir, out_dir, seed, bundles=None):
    """A companion pool holding the refined true cell of every pattern whose pool lacks one.

    For each pattern-condition with no correct candidate (`is_correct` under `label_frame`'s rule,
    which needs the true Bravais lattice), the true cell is finished by `refine_true_cell` and
    written as one candidate of the true lattice, in the pool's schema: `candidate_id`
    `TRUTH_CANDIDATE_ID`, `final_rank` where its M20 ranks among that lattice's survivors,
    `n_entering` that lattice's, and the run settings the pool recorded. Both sidecars are then
    computed on it by the functions that compute the pool's own. `bundles` limits it to some
    condition bundles. Returns the stamp it writes.
    """
    from mlindex.model_training.BenchmarkOptimizer import candidate_record
    from mlindex.optimization.Candidates import PRUNE_CAPTURE_MERITS
    from mlindex.optimization.UtilitiesOptimizer import MAXIMUM_UNIT_CELL, MINIMUM_UNIT_CELL
    from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM

    commit = committed_checkout()
    pool_dir, out_dir = Path(pool_dir), Path(out_dir)
    if (out_dir / TRUTH_POOL_STAMP).exists() or any(out_dir.glob('candidates_*.parquet')):
        raise FileExistsError(f'{out_dir} already holds a truth pool')
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = Benchmark.load_entries(pool_dir)
    columns = list(Benchmark.CANDIDATE_KEY) + [
        'is_correct', 'lattice_system', 'unit_cell', 'M20', 'n_entering', 'n_peaks', 'hkl_ref_length', 'assignment_threshold',
        'downsample_radius', 'prune_threshold']

    counts = {'patterns': int(entries.shape[0]), 'without_a_correct_cell': 0, 'added': 0,
              'correct_after_refinement': 0}
    bundles = list(bundles or Benchmark.available_bundles(pool_dir))
    entries = entries.loc[entries['condition_bundle'].isin(bundles)].reset_index(drop=True)
    for bundle in bundles:
        in_bundle = entries.loc[entries['condition_bundle'] == bundle]
        shards = dict(Benchmark.candidate_shards(pool_dir, bundle))
        for lattice in BRAVAIS_LATTICES:
            truth = in_bundle.loc[in_bundle['bravais_lattice_true'] == lattice]
            if not truth.shape[0] or lattice not in shards:
                continue
            lattice_system = BL_TO_LATTICE_SYSTEM[lattice]
            frame = Benchmark.load_candidates(pool_dir, bundle, columns=columns,
                                              bravais_lattices=[lattice], sidecars=())
            frame = frame.loc[frame['entry_id'].isin(truth['entry_id'])]
            found = set(frame.loc[Benchmark.relabelled(frame, entries), 'entry_id'])
            missing = truth.loc[~truth['entry_id'].isin(found)]
            counts['without_a_correct_cell'] += int(missing.shape[0])
            if not missing.shape[0]:
                continue
            hkl_ref = _hkl_reference(lattice, lattice_system)
            settings = frame.iloc[0]
            if int(settings['hkl_ref_length']) != hkl_ref.shape[0]:
                raise ValueError(f'{bundle}/{lattice}: the search used '
                                 f'{settings["hkl_ref_length"]} reference lines and this models '
                                 f'tree has {hkl_ref.shape[0]}')
            opt_params = {
                'minimum_uc': MINIMUM_UNIT_CELL, 'maximum_uc': MAXIMUM_UNIT_CELL,
                'figure_of_merit': 'M20',
                'assignment_threshold': float(settings['assignment_threshold']),
                'downsample_radius': float(settings['downsample_radius']),
                'prune_m20_threshold': float(settings['prune_threshold'])}
            n_peaks = int(settings['n_peaks'])
            by_entry = dict(tuple(frame.groupby('entry_id', sort=False)))
            records = []
            for entry in missing.itertuples():
                candidates = refine_true_cell(
                    np.asarray(entry.q2_obs, dtype=np.float64)[:n_peaks], entry.unit_cell_true,
                    lattice, lattice_system, hkl_ref, opt_params,
                    # Keyed by the pattern alone, so its refinement does not depend on which
                    # patterns were refined before it.
                    np.random.default_rng(derived_seed(f'{entry.entry_id}:{bundle}', seed)))
                pool = by_entry.get(entry.entry_id)
                m20 = float(candidates.best_M20[0])
                records.append(candidate_record(
                    {'entry_id': entry.entry_id, 'condition_bundle': bundle,
                     'q2_digest': entry.q2_digest},
                    lattice, lattice_system, n_peaks, hkl_ref.shape[0],
                    0 if pool is None else int(pool['n_entering'].iloc[0]), opt_params,
                    candidates.best_xnn, candidates.best_spacegroup, candidates.best_M20,
                    candidates.n_indexed,
                    [0 if pool is None else int((pool['M20'].to_numpy() > m20).sum())],
                    Benchmark.N_TOP_CANDIDATES, [m20],
                    np.full((1, len(PRUNE_CAPTURE_MERITS)), np.nan),
                    candidate_id=[TRUTH_CANDIDATE_ID]))
            added = Benchmark.label_frame(Benchmark.records_to_frame(records), entries)
            counts['added'] += int(added.shape[0])
            counts['correct_after_refinement'] += int(added['is_correct'].sum())
            Benchmark.write_candidate_shard(added, out_dir, bundle, lattice)
    Benchmark.write_entry_table(entries, out_dir)
    if counts['added']:
        merit_sidecar(out_dir)
        feature_sidecar(out_dir)
    stamp = dict(counts, bundles=bundles, source_pool=str(pool_dir), commit=commit, seed=int(seed),
                 source_commit=Benchmark.load_manifest(pool_dir).get('commit'))
    with open(out_dir / TRUTH_POOL_STAMP, 'w', encoding='utf-8') as handle:
        json.dump(stamp, handle, indent=2)
    return stamp
