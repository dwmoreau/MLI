"""S15 -- the end-to-end runs and the deployment recommendation. The logic; the script is thin.

Everything from S03 to S14 re-ranked a frozen pool. This step runs the indexer for real with the
prune cut and the ranking merit changed **together**, because the record has measured each alone:
a lower cut by itself is negative at every top-N and costs a quarter of the run (C2-F-032), and a
better merit by itself is capped by a ceiling only the cut can raise. The deliverable is a menu of
(cut, merit) with its measured cost, the success curves along the benchmark's condition axes, and
the hard-condition comparison campaign 1 owed and never delivered.

**The real indexer cannot rank by anything but M20**, and the prune threshold is deliberately not a
CLI option (decision 2026-08-24, C2-F-008). So a real run at a cut is a `run_fom_dump.py` run --
the same optimizer through the same `setup_mp_optimizers` / `run_mp_bl` path, with the cut
reaching it as `opt_params['prune_m20_threshold']` and every candidate kept -- and a merit is a
score over the persisted pool. Two depths are reported for every number: `all`, the whole
post-deduplication pool a re-ranker would see, and `in_top_n`, the top twenty per lattice that
production hands to its final sort (a strict superset of the printed list, because `run.py` still
promotes and deduplicates across lattices afterwards, which can only raise a correct cell's rank).

Nothing here is tuned. The thresholds are S12's, read from its full-scale table and frozen into
the design manifest with that file's checksum; the menu's recommendation follows a rule stated in
the same manifest and recorded as a decision, not a search over the result.

Every function below is pure enough to test without a pool, which is why they live here and not in
the script.
"""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomCombiner
from mlindex.model_training import FomConditions
from mlindex.model_training import FomMetrics
from mlindex.model_training.FomHoldoutReport import load_floors


BASE = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE/'docs'/'fom_campaign2'/'artifacts'
SCRIPTS_DIR = BASE/'mlindex'/'scripts'
TAG = 'S15'

BRAVAIS_LATTICES = tuple(FomMetrics.BRAVAIS_LATTICES)
HARD_LATTICES = ('aP', 'mC', 'mP')

# The benchmark's identity. `pool_size` is part of it (C2-F-069): the per-pattern search seed is
# keyed on the rank and the rank count IS the pool size, so a different value is a different search
# and the cut-1.5 pools already on disk would stop being the same candidates. NPOOLS is free.
SEED = 12345
OPTIMIZER_SEED = 12345
POOL_SIZE = 2
N_TOP_CANDIDATES = 20

# The grid. Cut 1.5 is the generation cut and its fully retained pools exist for the general
# population under eight of the nine bundles -- `error_shape` was never generated there -- so it
# joins the general factorial as a fourth REAL run rather than being regenerated.
CUTS = (5.0, 3.5, 3.0)
EXISTING_CUT = 1.5
ERROR_SHAPE_TAG = FomConditions.BY_KEY['error_shape'].tag
EXISTING_BUNDLES = tuple(tag for tag in FomConditions.tags() if tag != ERROR_SHAPE_TAG)

POPULATIONS = {
    'general': dict(entries='S08_floor_entries.csv', bundles=tuple(FomConditions.tags()),
                    n_entries=530, lattices=BRAVAIS_LATTICES,
                    # S08 drew 40 per lattice, capped by what exists: cF has 106 entries in the
                    # whole split and cI 156 (C2-F-048, C2-R-010).
                    per_lattice={'cF': 20, 'cI': 30}),
    'hard': dict(entries='S12_hard_entries.csv', bundles=tuple(FomConditions.HARD_BUNDLES),
                 n_entries=360, lattices=HARD_LATTICES, per_lattice={}),
    }

MERITS = ('M20', 'M_sym')
LEARNED = {'plus_probation': 'mlindex/models/fom_combiner_c2_fullscale/plus_probation_seed12345'}
INCUMBENT = (5.0, 'M20')
THRESHOLD_TABLE = 'S12_combiner_main_table_fullscale.csv'
POOL_SUBSETS = ('all', 'in_top_n')
METRICS = ('operating_point', 'top10')

# The floor a contrast is read against, per metric: S08 measured top-10 and S09 the operating
# point, both as `M_sym` vs M20 over four search seeds, aggregate and per lattice.
FLOOR_SOURCES = {
    'top10': ('S08_floor_contrast.csv', 'S08_floor_by_lattice.csv'),
    'operating_point': ('S09_floor_op_contrast.csv', 'S09_floor_op_by_lattice.csv'),
    }
FLOOR_REFERENCE = ('M_sym', 'M20')

# The menu rule, stated once. Per merit: the cut with the largest general operating point whose
# hard-population operating point is not below the incumbent's by more than this many standard
# errors of the hard lattices' own floor. Recorded as a decision; nothing is chosen on the result.
MENU_HARD_TOLERANCE_SE = 2.0
MENU_RULE = ('per merit, the cut with the largest general operating point whose hard-population '
             'operating point is not more than MENU_HARD_TOLERANCE_SE floor standard errors '
             'below the incumbent (5.0, M20)')


# ---------------------------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------------------------
def cut_label(cut):
    """`cut5`, `cut3.5`, `cut1.5` -- the `:g` form, so 5.0 and 5 name one directory."""
    return f'cut{float(cut):g}'


def arm_dir(root, population, cut):
    return Path(root)/'e2e'/population/cut_label(cut)


def pool_dir(root, population, cut):
    return Path(root)/'e2e'/population/f'{cut_label(cut)}_pool'


def bundle_dir(root, population, cut, tag):
    return arm_dir(root, population, cut)/tag


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


# ---------------------------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------------------------
def build_design(populations=POPULATIONS, cuts=CUTS, existing_cut=EXISTING_CUT,
                 existing_bundles=EXISTING_BUNDLES):
    """One row per (population, cut, condition bundle): what is generated, and what exists.

    Every row is a real run. `source` says whether S15 generates it (`e2e`) or reads a pool that
    already exists (`existing`, the cut-1.5 arm); it does NOT mark a restriction, because there
    are none in the factorial -- the restriction-vs-run comparison is a separate artefact.
    """
    rows = []
    for population, spec in populations.items():
        for cut in cuts:
            for tag in spec['bundles']:
                rows.append(dict(population=population, cut=float(cut), condition_bundle=tag,
                                 is_real_run=True, source='e2e', n_entries=spec['n_entries']))
        if population == 'general' and existing_cut is not None:
            for tag in existing_bundles:
                rows.append(dict(population=population, cut=float(existing_cut),
                                 condition_bundle=tag, is_real_run=True, source='existing',
                                 n_entries=spec['n_entries']))
    frame = pd.DataFrame(rows)
    frame['n_cells'] = frame['n_entries']
    return frame


def read_thresholds(artifact_dir, table=THRESHOLD_TABLE, scores=('M20', 'M_sym', 'plus_probation')):
    """S12's frozen thresholds, per score, with the checksum of the file they came from.

    A threshold is a property of the model and of the rows it was chosen on, never of the pool
    being reported (S12's `--calibration-from`). S15 chooses nothing: it reads the number S12
    chose on `fom-train` at M20's matched false-positive rate and carries it unchanged.
    """
    path = Path(artifact_dir)/table
    frame = pd.read_csv(path)
    out = {}
    for score in scores:
        rows = frame.loc[frame['arm'] == score]
        if rows.empty:
            raise SystemExit(f'{score} has no row in {path}; S15 cannot invent a threshold')
        row = rows.iloc[0]
        out[score] = dict(threshold=float(row['threshold']),
                          threshold_rule=str(row.get('threshold_rule', '')),
                          fit_seed=int(row['fit_seed']) if 'fit_seed' in row else None)
    return out, sha256_of(path)


def load_entry_list(artifact_dir, population, populations=POPULATIONS):
    path = Path(artifact_dir)/populations[population]['entries']
    frame = pd.read_csv(path)
    column = 'identifier' if 'identifier' in frame.columns else 'entry_id'
    return frame[column].astype(str).tolist(), path


def check_entry_list(identifiers, manifest, population, populations=POPULATIONS):
    """The list is the size the record says, every crystal is `fom-dev`, and the lattice mix is
    the one the population was drawn with. Returns the per-lattice table."""
    spec = populations[population]
    wanted = manifest.loc[manifest['identifier'].isin(identifiers)]
    missing = set(identifiers) - set(wanted['identifier'])
    if missing:
        raise SystemExit(f'{population}: {len(missing)} entries are not in the split manifest, '
                         f'e.g. {sorted(missing)[:3]}')
    if wanted.shape[0] != spec['n_entries']:
        raise SystemExit(f'{population}: {wanted.shape[0]} entries against the recorded '
                         f'{spec["n_entries"]}')
    splits = set(wanted['split'].astype(str))
    if splits != {'fom-dev'}:
        raise SystemExit(f'{population}: split must be fom-dev only, found {sorted(splits)} -- '
                         f'a fom-train crystal here would be reported on rows S12 chose its '
                         f'thresholds on, and fom-test is sealed')
    lattices = set(wanted['bravais_lattice'].astype(str))
    if lattices != set(spec['lattices']):
        raise SystemExit(f'{population}: lattices {sorted(lattices)} against the expected '
                         f'{sorted(spec["lattices"])}')
    counts = wanted.groupby('bravais_lattice').size()
    return counts.rename('n').reset_index()


def pilot_entries(manifest, identifiers, per_lattice=2, seed=SEED):
    """A small lattice-balanced subset of a population, for the laptop pilot."""
    wanted = manifest.loc[manifest['identifier'].isin(identifiers)].copy()
    wanted['identifier'] = wanted['identifier'].astype(str)
    rng = np.random.default_rng(seed)
    chosen = []
    for lattice, group in wanted.groupby('bravais_lattice'):
        ids = sorted(group['identifier'])
        chosen += [ids[i] for i in rng.choice(len(ids), size=min(per_lattice, len(ids)),
                                              replace=False)]
    return sorted(chosen)


# ---------------------------------------------------------------------------------------------
# Generation: the exact invocation, and the provenance beside it
# ---------------------------------------------------------------------------------------------
def generate_argv(population, cut, condition_key, root, n_pools, manifest_path, entry_ids_file,
                  pool_size=POOL_SIZE, seed=SEED, optimizer_seed=OPTIMIZER_SEED,
                  extra_opt_params=None, record_timing=True):
    """The argv for `run_fom_dump.main`: a real run at `cut`, every candidate kept, timed."""
    tag = FomConditions.BY_KEY[condition_key].tag
    argv = ['--condition', condition_key,
            '--split-manifest', str(manifest_path),
            '--entry-ids-file', str(entry_ids_file),
            '--seed', str(int(seed)),
            '--optimizer-seed', str(int(optimizer_seed)),
            '--prune-threshold', f'{float(cut):g}',
            '--n-pools', str(int(n_pools)),
            '--pool-size', str(int(pool_size)),
            '--no-subsample',
            '--predownsample-entries', '0',
            '--out-dir', str(bundle_dir(root, population, cut, tag))]
    if record_timing:
        argv.append('--record-timing')
    for key, value in sorted((extra_opt_params or {}).items()):
        argv += ['--opt-param', f'{key}={value}']
    return argv


def write_provenance(directory, **fields):
    """Written BEFORE the first bundle runs, so a killed job still says what it was."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(fields)
    payload.setdefault('commit', commit_hash())
    payload.setdefault('platform', platform.platform())
    payload.setdefault('arch', platform.machine())
    payload.setdefault('complete', False)
    path = directory/'provenance.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')
    return path


def stamp_complete(directory, **fields):
    """The completion stamp, only on the way out of a finished arm."""
    path = Path(directory)/'provenance.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload.update(fields)
    payload['complete'] = True
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')
    return payload


def load_provenance(directory, require_complete=True):
    """An arm's provenance; refuses an arm without the stamp.

    A half-finished arm is indistinguishable from a finished one by its contents alone, and a
    paired comparison over whichever crystals happened to finish would look entirely normal
    (S13's `load_arm`, same reason).
    """
    path = Path(directory)/'provenance.json'
    if not path.exists():
        raise SystemExit(f'{directory} has no provenance.json; it was not generated by this driver')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if require_complete and not payload.get('complete'):
        raise SystemExit(f'{directory} is an INCOMPLETE arm -- no completion stamp, so it was '
                         f'killed or is still running. Re-run it or delete it.')
    return payload


def arm_bundles_done(root, population, cut, bundles):
    """Which of an arm's bundles have a manifest, i.e. finished generation."""
    return [tag for tag in bundles
            if (bundle_dir(root, population, cut, tag)/'manifest.json').exists()]


# ---------------------------------------------------------------------------------------------
# Sidecars: only what the named models read
# ---------------------------------------------------------------------------------------------
def learned_groups(learned_dirs):
    """The union of feature groups the named saved models read, from their specifications."""
    groups = []
    for directory in learned_dirs.values():
        spec = json.loads((Path(directory)/'specification.json').read_text(encoding='utf-8'))
        for group in spec.get('groups', ()):
            if group not in groups:
                groups.append(group)
    return tuple(dict.fromkeys(tuple(FomCombiner.DEFAULT_GROUPS) + tuple(groups)))


def sidecar_commands(arm_directory, pool, processes, groups, python='python',
                     scripts_dir=SCRIPTS_DIR):
    """The argv list for consolidating an arm and writing the sidecars its models need.

    `merits/` and `structural/` are unconditional; `merits_soft/` and `holdout_merits/` are the two
    slowest passes and are added only when a named model's groups map to them. Every producer is
    followed by its own `--verify`, because exit code 0 is not evidence (C2-F-071, C2-F-139).
    """
    scripts_dir = Path(scripts_dir)
    pool = str(pool)
    commands = [[python, str(scripts_dir/'run_fom_dump_consolidate.py'),
                 '--dump-root', str(arm_directory), '--out-dir', pool],
                [python, str(scripts_dir/'run_fom_floor_merits.py'),
                 '--pool', pool, '--processes', str(int(processes))],
                [python, str(scripts_dir/'run_fom_structural_features.py'),
                 '--pool', pool, '--processes', str(int(processes))]]
    directories = {FomCombiner.SIDECAR_DIRS[group] for group in groups
                   if group in FomCombiner.SIDECAR_DIRS}
    if 'merits_soft' in directories:
        commands.append([python, str(scripts_dir/'run_fom_floor_merits.py'), '--pool', pool,
                         '--soft', '--out-dir', str(Path(pool)/'merits_soft'),
                         '--processes', str(int(processes))])
    if 'holdout_merits' in directories:
        commands.append([python, str(scripts_dir/'run_fom_holdout_merits.py'), '--pool', pool,
                         '--processes', str(min(16, int(processes)))])
    commands.append([python, str(scripts_dir/'run_fom_floor_merits.py'), '--pool', pool,
                     '--verify'])
    commands.append([python, str(scripts_dir/'run_fom_structural_features.py'), '--pool', pool,
                     '--verify'])
    if 'holdout_merits' in directories:
        commands.append([python, str(scripts_dir/'run_fom_holdout_merits.py'), '--pool', pool,
                         '--verify'])
    return commands


# ---------------------------------------------------------------------------------------------
# Scoring a pool
# ---------------------------------------------------------------------------------------------
def reference_scores(seed=SEED):
    """The stored columns and the two floors, as `run_fom_combiner.reference_scores` builds them.

    A constant score already reaches 0.2352 of top-10 on this population because ties break
    cubic-first (C2-F-083), so a rank metric is read against that, not against zero.
    """
    rng = np.random.default_rng(seed)
    return {'M20': 'M20', 'M_sym': 'M_sym',
            'constant': lambda frame: np.ones(frame.shape[0]),
            'uniform_random': lambda frame: rng.random(frame.shape[0])}


def load_scores(learned_dirs, seed=SEED):
    """`{name: column-or-callable}` for every stored merit and every named saved model."""
    from mlindex.model_training import NeuralScore
    scores = reference_scores(seed)
    for name, directory in learned_dirs.items():
        model = NeuralScore.load_any(directory)
        scores[name] = model.score
    return scores


def assert_report_split(entries, train_split='fom-train'):
    """No crystal the thresholds were chosen on may be in the report pool (S12's guard)."""
    leaked = entries.loc[entries['split'].astype(str) == train_split, 'entry_id']
    if len(leaked):
        raise SystemExit(f'{leaked.nunique()} {train_split} crystals are in the report pool, so a '
                         f'threshold reused from S12 may have been chosen on rows being reported')


def assert_fully_retained(pool):
    depth, subsampled = FomBenchmark.subsample_depth(Path(pool))
    if subsampled is None:
        raise SystemExit(f'{pool} has no readable manifest, so it cannot be certified unthinned')
    if subsampled:
        raise SystemExit(f'{pool} was negatively subsampled at K={depth}; a learned score\'s rank '
                         f'there is optimistic (C2-R-013). S15 reduces fully retained pools only')


def reduce_arm(pool, scores, entries, groups, split='fom-dev', on_shard=None, row_filter=None,
               keep_entry_ids=None, extra_columns=()):
    """One pass over a fully retained pool, every score at once. Returns `{name: (per_entry, meta)}`."""
    assert_fully_retained(pool)
    assert_report_split(entries)
    frames = FomCombiner.combiner_frames_c2(pool, entries, groups=groups,
                                            keep_entry_ids=keep_entry_ids,
                                            row_filter=row_filter, extra_columns=extra_columns)
    reduced = FomMetrics.reduce_many(frames, scores, entries=entries, splits={split: None},
                                     higher_is_better={name: True for name in scores},
                                     subsample_top_k=None, on_shard=on_shard)
    return {name: value for (name, found_split), value in reduced.items() if found_split == split}


def concatenate_reductions(parts):
    """Reductions of several pools over one arm (cut 1.5 is two pools) become one per score."""
    out = {}
    names = set().union(*(part.keys() for part in parts))
    for name in sorted(names):
        frames, metas = zip(*(part[name] for part in parts if name in part))
        per_entry = pd.concat(frames, ignore_index=True)
        duplicated = per_entry.duplicated(['entry_id', 'condition_bundle'])
        if duplicated.any():
            raise SystemExit(f'{name}: {int(duplicated.sum())} cells reduced twice across pools')
        meta = dict(metas[0])
        meta['n_candidates_seen'] = int(sum(m.get('n_candidates_seen', 0) for m in metas))
        meta['n_entries'] = int(per_entry.shape[0])
        meta['ranks_exact'] = all(bool(m.get('ranks_exact')) for m in metas)
        meta['pools'] = [m.get('pool') for m in metas]
        meta['entry_digest'] = FomMetrics.entry_digest(per_entry)
        out[name] = (per_entry, meta)
    return out


def reduction_paths(artifact_dir, population, cut, suffix=''):
    stem = f'{TAG}_reduced_{population}_{cut_label(cut)}{suffix}'
    return Path(artifact_dir)/f'{stem}_meta.json', stem


def write_reductions(reduced, artifact_dir, population, cut, suffix='', require_exact=True):
    """One parquet per score plus one meta file per arm. The reductions are the sufficient
    statistic: both pool depths, every threshold and every stratum come from these files."""
    meta_path, stem = reduction_paths(artifact_dir, population, cut, suffix)
    metas = {}
    for name, (per_entry, meta) in sorted(reduced.items()):
        if require_exact and not meta.get('ranks_exact'):
            raise SystemExit(f'{name} on {population} {cut_label(cut)} is not rank-exact: '
                             f'{meta.get("rank_exactness")}')
        per_entry.to_parquet(Path(artifact_dir)/f'{stem}_{name}.parquet', index=False)
        metas[name] = meta
    meta_path.write_text(json.dumps(metas, indent=2, sort_keys=True, default=str),
                         encoding='utf-8')
    return meta_path


def load_reductions(artifact_dir, population, cut, suffix=''):
    meta_path, stem = reduction_paths(artifact_dir, population, cut, suffix)
    if not meta_path.exists():
        return None
    metas = json.loads(meta_path.read_text(encoding='utf-8'))
    return {name: (pd.read_parquet(Path(artifact_dir)/f'{stem}_{name}.parquet'), meta)
            for name, meta in metas.items()}


# ---------------------------------------------------------------------------------------------
# Gate 5: the arms differ in nothing but the cut
# ---------------------------------------------------------------------------------------------
IDENTITY_KEYS = ('seed', 'optimizer_seed', 'pool_size', 'search_seed_scheme',
                 'split_manifest_sha256', 'arch', 'broadening_tag', 'iteration_scale')
WITNESS_COLUMNS = ('n_dropout_achieved', 'second_phase_partner')


def check_peak_digests(entries_by_cut, witness_columns=WITNESS_COLUMNS):
    """Every (entry, bundle) cell has the same peak list under every cut.

    `q2_digest` hashes the twenty-peak window and depends on the seed, the entry and the condition
    -- never on the cut -- so agreement here is exactly the handoff's "peak lists checked
    digest-for-digest". Returns a per-bundle table; raises naming the first offending cell on any
    disagreement or missing cell, because a factorial over different peak lists is not paired.
    """
    key = ['entry_id', 'condition_bundle']
    tables = {}
    for cut, entries in entries_by_cut.items():
        frame = entries[key + ['q2_digest'] + [c for c in witness_columns if c in entries.columns]]
        frame = frame.copy()
        frame['second_phase_partner'] = (frame['second_phase_partner'].astype(str)
                                         if 'second_phase_partner' in frame.columns else '')
        tables[float(cut)] = frame.set_index(key).sort_index()
    cuts = sorted(tables)
    if len(cuts) < 2:
        raise ValueError('need at least two cuts to compare')
    problems = []
    rows = []
    bundles = sorted(set().union(*(set(t.index.get_level_values(1)) for t in tables.values())))
    for bundle in bundles:
        per_cut = {cut: t.xs(bundle, level='condition_bundle', drop_level=False)
                   for cut, t in tables.items()}
        union = sorted(set().union(*(set(p.index) for p in per_cut.values())))
        common = sorted(set.intersection(*(set(p.index) for p in per_cut.values())))
        n_missing = len(union) - len(common)
        if n_missing:
            for cut, p in per_cut.items():
                gone = sorted(set(union) - set(p.index))
                if gone:
                    problems.append(f'{bundle}: {len(gone)} cell(s) missing at {cut_label(cut)}, '
                                    f'e.g. {gone[0][0]}')
        agree = disagree = 0
        for cell in common:
            digests = {cut: per_cut[cut].loc[cell, 'q2_digest'] for cut in cuts}
            same = len(set(digests.values())) == 1
            for column in witness_columns:
                if column in per_cut[cuts[0]].columns:
                    values = {str(per_cut[cut].loc[cell, column]) for cut in cuts}
                    same = same and len(values) == 1
            if same:
                agree += 1
            else:
                disagree += 1
                if len(problems) < 20:
                    problems.append(f'{bundle}: {cell[0]} differs across cuts '
                                    f'{[cut_label(c) for c in cuts]}')
        rows.append(dict(condition_bundle=bundle, cuts_compared=','.join(cut_label(c) for c in cuts),
                         n_cells=len(common), n_agree=agree, n_disagree=disagree,
                         n_missing=n_missing))
    table = pd.DataFrame(rows)
    if problems:
        raise ValueError('peak lists are NOT identical across cuts: ' + '; '.join(problems[:10]))
    return table


def manifest_identity(manifest, keys=IDENTITY_KEYS):
    """The identity fields of a pool manifest, from its bundle manifests when consolidated."""
    inner = list(manifest.get('bundle_manifests', {}).values()) or [manifest]
    identity = {}
    for key in keys:
        values = {json.dumps(m.get(key), sort_keys=True, default=str) for m in inner}
        identity[key] = sorted(values)[0] if len(values) == 1 else 'MIXED:' + '|'.join(sorted(values))
    identity['commit'] = sorted({str(m.get('commit')) for m in inner})
    identity['prune_threshold'] = sorted({str(m.get('prune_threshold')) for m in inner})
    return identity


def check_manifest_identity(manifests_by_cut, keys=IDENTITY_KEYS):
    """Every arm shares seed, optimizer seed, pool size, seed scheme, split manifest and arch."""
    identities = {cut: manifest_identity(m, keys) for cut, m in manifests_by_cut.items()}
    problems = []
    for key in keys:
        values = {cut: identity[key] for cut, identity in identities.items()}
        if len(set(values.values())) != 1:
            problems.append(f'{key}: {values}')
    if problems:
        raise ValueError('the arms differ in more than the cut: ' + '; '.join(problems))
    return pd.DataFrame([dict(cut=cut, **{k: (v if not isinstance(v, list) else ','.join(v))
                                          for k, v in identity.items()})
                         for cut, identity in sorted(identities.items())])


# ---------------------------------------------------------------------------------------------
# Analysis: levels, contrasts, floors
# ---------------------------------------------------------------------------------------------
def summarise(per_entry, meta, threshold, pool_subset, top_n=10, n_bootstrap=1000, seed=SEED):
    return FomMetrics.summarise_per_entry(per_entry, meta, threshold=threshold, top_n=top_n,
                                          strata=FomMetrics.DEFAULT_STRATA,
                                          pool_subset=pool_subset, n_bootstrap=n_bootstrap,
                                          seed=seed)


def common_keys(per_entries):
    """The (entry, bundle) cells present in every reduction, for a cross-pool pairing."""
    sets = [set(zip(frame['entry_id'].astype(str), frame['condition_bundle'].astype(str)))
            for frame in per_entries]
    return set.intersection(*sets)


def restrict_per_entry(per_entry, keys):
    pairs = list(zip(per_entry['entry_id'].astype(str), per_entry['condition_bundle'].astype(str)))
    mask = np.array([pair in keys for pair in pairs], dtype=bool)
    return per_entry.loc[mask].reset_index(drop=True), int((~mask).sum())


LEVEL_COLUMNS = ('operating_point', 'operating_point_ci_low', 'operating_point_ci_high',
                 'top1', 'top5', 'top10', 'top10_ci_low', 'top10_ci_high', 'rank_only',
                 'threshold_only', 'mrr', 'ceiling_rescorer', 'reported', 'false_positive',
                 'precision', 'operating_point_given_found', 'n_entries', 'n_clusters')


def level_row(result, scope='aggregate', **identifiers):
    """One row of the factorial's level table from a `MetricsResult`."""
    table = result.aggregate if scope == 'aggregate' else result.hard
    row = dict(identifiers)
    row['scope'] = scope
    for column in LEVEL_COLUMNS:
        row[column] = float(table[column].iloc[0]) if column in table.columns else np.nan
    return row


def stratum_rows(result, stratum, **identifiers):
    """Level rows per level of one stratum (`bravais_lattice`, `condition_bundle`)."""
    rows = []
    for _, line in result.stratum(stratum).iterrows():
        row = dict(identifiers)
        row['scope'] = f'{stratum}={line["level"]}'
        for column in LEVEL_COLUMNS:
            row[column] = float(line[column]) if column in line.index else np.nan
        rows.append(row)
    return rows


def contrast(reference, arm, metric, mask=None):
    """One paired contrast: McNemar plus the cluster-bootstrap interval, signed so that a
    positive delta means the arm is better (`run_fom_combiner._pair`'s convention)."""
    try:
        test = FomMetrics.mcnemar(reference, arm, metric=metric, subset=mask)
        interval = FomMetrics.paired_delta_ci(arm, reference, metric=metric, subset=mask)
    except (ValueError, KeyError):
        return None
    return dict(metric=metric, delta_pp=100*float(interval['delta']),
                ci_low_pp=100*float(interval['ci_low']), ci_high_pp=100*float(interval['ci_high']),
                gained=int(test['n_b_only']), lost=int(test['n_a_only']),
                p_value=float(test['p_value']), method=str(test.get('method', '')),
                n_entries=int(test['n_entries']), n_clusters=int(test['n_clusters']))


def scope_masks(result, lattices=BRAVAIS_LATTICES, bundles=None):
    """`{scope: mask}` in `mcnemar`'s sorted order: aggregate, hard, each lattice, each bundle."""
    masks = {'aggregate': None}
    if 'is_hard' in result.per_entry.columns:
        masks['hard'] = FomMetrics.stratum_mask(result, 'is_hard', True)
    present = set(result.per_entry['bravais_lattice'].astype(str))
    for lattice in lattices:
        if lattice in present:
            masks[f'bravais_lattice={lattice}'] = FomMetrics.stratum_mask(
                result, 'bravais_lattice', lattice)
    # The bundle is an index level once `mcnemar` sorts by (entry_id, condition_bundle), so
    # `stratum_mask` cannot see it as a column; the mask is built in the same sorted order here.
    sorted_bundles = result.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index() \
        .index.get_level_values('condition_bundle').astype(str).to_numpy()
    for bundle in (bundles or sorted(set(sorted_bundles))):
        masks[f'condition_bundle={bundle}'] = sorted_bundles == bundle
    return masks


def load_floor_tables(artifact_dir, sources=FLOOR_SOURCES, reference=FLOOR_REFERENCE):
    """`{metric: (aggregate floor pp, {lattice: se pp})}` from the S08 and S09 artefacts."""
    floors = {}
    for metric, (aggregate_artefact, per_lattice_artefact) in sources.items():
        floors[metric] = load_floors(artifact_dir, reference=(reference[0], reference[1], metric),
                                     aggregate_artefact=aggregate_artefact,
                                     per_lattice_artefact=per_lattice_artefact)
    return floors


def floor_for(floors, metric, scope, hard_lattices=HARD_LATTICES):
    """(floor pp, source) for a contrast at `scope`.

    A per-lattice scope uses that lattice's own floor (PROTOCOL section 8). The hard population has
    no measured floor of its own; the closest honest number is the mean of the hard lattices' own
    floors, and the source says so. Anything else uses the aggregate.
    """
    aggregate, per_lattice = floors.get(metric, (None, {}))
    if scope.startswith('bravais_lattice='):
        lattice = scope.split('=', 1)[1]
        value = per_lattice.get(lattice)
        return (np.nan if value is None else float(value)), f'{metric} floor, {lattice}'
    if scope == 'hard' or scope == 'hard_population':
        values = [per_lattice[l] for l in hard_lattices if l in per_lattice]
        if values:
            return float(np.mean(values)), f'{metric} floor, mean of {"/".join(hard_lattices)}'
    return (np.nan if aggregate is None else float(aggregate)), f'{metric} floor, aggregate'


def in_floor_ses(delta_pp, floor_pp):
    if floor_pp is None or not np.isfinite(floor_pp) or floor_pp <= 0:
        return np.nan
    return float(delta_pp)/float(floor_pp)


# ---------------------------------------------------------------------------------------------
# The success curves: the benchmark's own axes
# ---------------------------------------------------------------------------------------------
def success_curve_axes(conditions=FomConditions.CONDITIONS):
    """Which bundles sit on which axis, and at which x, with the caveat where an axis point moves
    more than one thing at once. Read off `FomConditions`, never hard-coded."""
    nominal = [c for c in conditions if c.error_multiplier == 1 and c.n_contaminants == 0
               and c.n_dropout == 0 and c.second_phase_lines == 0 and c.intercept_scale == 1][0]
    points = []
    # Error scale: the cont0, no-dropout, no-phase, nominal-intercept bundles.
    for c in conditions:
        if (c.n_contaminants == 0 and c.n_dropout == 0 and c.second_phase_lines == 0
                and c.intercept_scale == 1):
            points.append(dict(axis='error_scale', x=c.error_multiplier, condition_bundle=c.tag,
                               caveat=''))
    # Contaminant count at nominal error: 0 (nominal), 1 (the smallest-dropout cont1 bundle,
    # which also drops peaks), 2 (contaminated).
    cont1 = sorted([c for c in conditions if c.n_contaminants == 1 and c.error_multiplier == 1
                    and c.second_phase_lines == 0], key=lambda c: c.n_dropout)
    points.append(dict(axis='contaminant_count', x=0, condition_bundle=nominal.tag, caveat=''))
    if cont1:
        c = cont1[0]
        points.append(dict(axis='contaminant_count', x=1, condition_bundle=c.tag,
                           caveat=f'also drops {c.n_dropout} peaks'))
    for c in conditions:
        if c.n_contaminants >= 2 and c.error_multiplier == 1 and c.n_dropout == 0:
            points.append(dict(axis='contaminant_count', x=c.n_contaminants,
                               condition_bundle=c.tag, caveat=''))
    # Dropout: the cont1 series, with nominal as the x = 0 reference (which carries no
    # contaminant, hence the caveat).
    points.append(dict(axis='dropout', x=0, condition_bundle=nominal.tag,
                       caveat='no contaminant; the dropout series carries one'))
    for c in cont1:
        points.append(dict(axis='dropout', x=c.n_dropout, condition_bundle=c.tag, caveat=''))
    return pd.DataFrame(points)


# ---------------------------------------------------------------------------------------------
# Restriction versus run
# ---------------------------------------------------------------------------------------------
def restrict_at_cut(frame, cut, n_top=N_TOP_CANDIDATES, column='m20_at_prune'):
    """What a cut-1.5 pool says a higher cut would have admitted -- `prune_below_m20` replayed.

    Keeps every row at or above the cut and, where a (entry, bundle, lattice) pool has none, its
    best row (the production fallback, `Candidates.prune_below_m20`); then recomputes the two
    pool-position columns the cut moves, `final_rank` and `in_top_n`, and the per-entry survivor
    count. `n_entering` is left as the cut-1.5 run's, which is the approximation this comparison
    exists to measure: a restriction gives the candidates a higher cut would have admitted, not the
    cells a run at that cut would have refined.
    """
    if column not in frame.columns:
        raise KeyError(f'{column} is not on the frame; project it in with extra_columns')
    at_prune = frame[column].to_numpy(dtype=np.float64)
    group = frame.groupby(['entry_id', 'condition_bundle', 'bravais_lattice'], sort=False)
    keep = at_prune >= float(cut)
    best = group[column].transform('max').to_numpy(dtype=np.float64)
    none_cleared = group[column].transform(lambda s: bool((s >= float(cut)).sum() == 0)).to_numpy(dtype=bool)
    keep |= none_cleared & (at_prune == best)
    out = frame.loc[keep].copy()
    order = out.groupby(['entry_id', 'condition_bundle', 'bravais_lattice'], sort=False)['M20']
    out['final_rank'] = order.rank(method='first', ascending=False).astype(np.int64) - 1
    out['in_top_n'] = out['final_rank'].to_numpy() < int(n_top)
    if 'pool_size_full' in out.columns:
        out['pool_size_full'] = out.groupby(['entry_id', 'condition_bundle'])['M20'].transform(
            'size').to_numpy(dtype=np.float64)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# The deployment menu
# ---------------------------------------------------------------------------------------------
def build_menu(levels, contrasts, cost=None, incumbent=INCUMBENT, tolerance_se=MENU_HARD_TOLERANCE_SE,
               merits=None):
    """One row per (cut, merit, pool depth): the levels on both populations, the paired delta
    against the incumbent in floor standard errors, the worst lattice, the cost, and whether the
    stated rule recommends it."""
    aggregate = levels.loc[levels['scope'] == 'aggregate']
    pair = contrasts.loc[(contrasts['contrast_kind'] == 'pair') & (contrasts['scope'] == 'aggregate')
                         & (contrasts['metric'] == 'operating_point')]
    lattice = contrasts.loc[(contrasts['contrast_kind'] == 'pair')
                            & contrasts['scope'].str.startswith('bravais_lattice=')
                            & (contrasts['metric'] == 'operating_point')
                            & (contrasts['population'] == 'general')]
    rows = []
    merits = merits or sorted(aggregate['merit'].unique())
    for pool_subset in sorted(aggregate['pool_subset'].unique()):
        for merit in merits:
            for cut in sorted(aggregate.loc[aggregate['merit'] == merit, 'cut'].unique(), reverse=True):
                row = dict(cut=float(cut), merit=merit, pool_subset=pool_subset)
                for population in ('general', 'hard'):
                    level = aggregate.loc[(aggregate['population'] == population)
                                          & (aggregate['cut'] == cut) & (aggregate['merit'] == merit)
                                          & (aggregate['pool_subset'] == pool_subset)]
                    for column, name in (('operating_point', 'op'), ('top10', 'top10'),
                                         ('ceiling_rescorer', 'ceiling')):
                        row[f'{population}_{name}'] = (float(level[column].iloc[0])
                                                       if level.shape[0] else np.nan)
                    delta = pair.loc[(pair['population'] == population) & (pair['cut'] == cut)
                                     & (pair['merit'] == merit) & (pair['pool_subset'] == pool_subset)]
                    for column in ('delta_pp', 'ci_low_pp', 'ci_high_pp', 'p_value', 'standard_errors'):
                        row[f'{population}_{column}_vs_incumbent'] = (
                            float(delta[column].iloc[0]) if delta.shape[0] else np.nan)
                worst = lattice.loc[(lattice['cut'] == cut) & (lattice['merit'] == merit)
                                    & (lattice['pool_subset'] == pool_subset)]
                if worst.shape[0]:
                    line = worst.loc[worst['standard_errors'].idxmin()]
                    row['worst_lattice'] = line['scope'].split('=', 1)[1]
                    row['worst_lattice_delta_pp'] = float(line['delta_pp'])
                    row['worst_lattice_standard_errors'] = float(line['standard_errors'])
                else:
                    row['worst_lattice'], row['worst_lattice_delta_pp'] = '', np.nan
                    row['worst_lattice_standard_errors'] = np.nan
                if cost is not None:
                    line = cost.loc[(cost['population'] == 'general') & (cost['cut'] == cut)
                                    & (cost['condition_bundle'] == 'all')]
                    row['seconds_per_entry'] = (float(line['seconds_per_entry_median'].iloc[0])
                                                if line.shape[0] else np.nan)
                    row['pool_size_median'] = (float(line['pool_size_full_median'].iloc[0])
                                               if line.shape[0] else np.nan)
                row['is_incumbent'] = (float(cut) == float(incumbent[0]) and merit == incumbent[1])
                rows.append(row)
    menu = pd.DataFrame(rows)
    if cost is not None and 'seconds_per_entry' in menu.columns:
        base = menu.loc[menu['is_incumbent'] & (menu['pool_subset'] == menu['pool_subset'].iloc[0]),
                        'seconds_per_entry']
        base = float(base.iloc[0]) if base.shape[0] and np.isfinite(base.iloc[0]) else np.nan
        menu['seconds_vs_incumbent_pct'] = 100*(menu['seconds_per_entry']/base - 1)
    menu['recommended'] = False
    menu['rule'] = MENU_RULE
    for (pool_subset, merit), group in menu.groupby(['pool_subset', 'merit']):
        admissible = group.loc[~(group['hard_standard_errors_vs_incumbent'] < -tolerance_se)]
        if admissible.shape[0]:
            menu.loc[admissible['general_op'].idxmax(), 'recommended'] = True
    return menu
