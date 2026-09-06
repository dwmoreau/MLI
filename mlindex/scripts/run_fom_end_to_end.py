"""S15 -- end-to-end runs and the deployment recommendation. The thin driver.

    # laptop, once: the design manifest, the frozen thresholds and the entry lists
    python run_fom_end_to_end.py --stage plan [--pilot]

    # one real run per (population, cut, condition) -- run_fom_dump through the S15 naming
    python run_fom_end_to_end.py --stage generate --population general --cut 5.0 \
        --condition nominal --out-root $SCRATCH/fom_campaign2 --n-pools 64
    python run_fom_end_to_end.py --stage complete --population general --cut 5.0 --out-root ...

    # consolidate an arm and write the sidecars its models read
    python run_fom_end_to_end.py --stage sidecars --population general --cut 5.0 --out-root ... \
        --processes 64 [--execute]

    # score every merit over one arm's pool (several --pool roots for the cut-1.5 arm)
    python run_fom_end_to_end.py --stage reduce --population general --cut 5.0 \
        --pool <pool> [--pool <pool2>] [--learned plus_probation=<dir>]
    python run_fom_end_to_end.py --stage restrict --pool <cut-1.5 pool> [--pool ...] --cuts 5.0,3.5,3.0

    # laptop: the tables, the figures and the results document, from the reductions alone
    python run_fom_end_to_end.py --stage analyse [--out-root ...]
    python run_fom_end_to_end.py --stage figure
    python run_fom_end_to_end.py --stage report

The logic is in `mlindex/model_training/FomEndToEnd.py`; this file parses arguments, moves files
and prints. See that module's docstring for what the step measures and why it runs the generator
rather than `run.py`.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomConditions
from mlindex.model_training import FomEndToEnd as E2E
from mlindex.model_training import FomMetrics


MANIFEST = E2E.ARTIFACT_DIR/'S06_split_manifest.parquet'
DEFAULT_OUT_ROOT = E2E.BASE/'mlindex'/'data'/'fom_e2e'
SCORES = ('M20', 'M_sym', 'plus_probation', 'constant', 'uniform_random')
REPORTED = ('M20', 'M_sym', 'plus_probation')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='S15: end-to-end runs and the deployment menu')
    parser.add_argument('--stage', required=True,
                        choices=('plan', 'generate', 'complete', 'sidecars', 'reduce', 'restrict',
                                 'analyse', 'figure', 'report'))
    parser.add_argument('--artifact-dir', default=str(E2E.ARTIFACT_DIR))
    parser.add_argument('--out-root', default=str(DEFAULT_OUT_ROOT),
                        help='Where the arms live: <out-root>/e2e/<population>/cut<cut>/')
    parser.add_argument('--population', choices=tuple(E2E.POPULATIONS), default='general')
    parser.add_argument('--cut', type=float, default=None)
    parser.add_argument('--cuts', default=','.join(f'{c:g}' for c in E2E.CUTS),
                        help='restrict: the cuts to replay on the cut-1.5 pool')
    parser.add_argument('--condition', default=None,
                        help='generate: one FomConditions key; omit to run every bundle of the '
                             'population in turn')
    parser.add_argument('--split-manifest', default=str(MANIFEST))
    parser.add_argument('--entries-file', default=None,
                        help='generate: the entry list; defaults to the plan stage\'s '
                             'S15_entries_<population>.csv')
    parser.add_argument('--n-pools', type=int, default=4)
    parser.add_argument('--pool-size', type=int, default=E2E.POOL_SIZE)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--opt-param', action='append', default=None, metavar='KEY=VALUE',
                        help='generate: an extra optimizer setting, e.g. hkl_source=posterior for '
                             'the C2-Q-020 arm. Recorded in the provenance as the arm\'s options')
    parser.add_argument('--arm-suffix', default='',
                        help='generate/reduce: names a variant arm, e.g. _posterior, so its '
                             'directory and reductions do not collide with the main grid')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true',
                        help='sidecars: run the commands rather than printing them')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--pool', action='append', default=None,
                        help='reduce/restrict: a fully retained pool root; repeat for several')
    parser.add_argument('--learned', action='append', default=None, metavar='NAME=DIR',
                        help='reduce: a saved combiner to score, by name; defaults to S12\'s '
                             'plus_probation at full scale')
    parser.add_argument('--keep-entries', default=None,
                        help='reduce/restrict/analyse: CSV of identifiers, to work on a subset '
                             '(the pilot); analyse restricts the digest and cost tables to it')
    parser.add_argument('--suffix', default='',
                        help='reduce: namespaces the reduction files, e.g. _pilot')
    parser.add_argument('--pilot', action='store_true',
                        help='plan: also write S15_pilot_entries.csv (2 per lattice)')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--existing-pool', action='append', default=None,
                        metavar='POPULATION:CUT=PATH',
                        help='analyse: where an existing arm\'s pool lives, for the cost and the '
                             'digest tables (e.g. general:1.5=mlindex/data/fom_full_c2_pool)')
    return parser.parse_args(argv)


def _learned(args):
    if not args.learned:
        return {name: str(E2E.BASE/path) for name, path in E2E.LEARNED.items()}
    out = {}
    for item in args.learned:
        name, _, directory = item.partition('=')
        if not directory:
            raise SystemExit(f'--learned expects NAME=DIR, got {item!r}')
        out[name] = directory
    return out


def _population_dir_suffix(args):
    return args.arm_suffix or ''


# ---------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------
def run_plan(args):
    artifact_dir = Path(args.artifact_dir)
    manifest = pd.read_parquet(args.split_manifest)
    manifest['identifier'] = manifest['identifier'].astype(str)
    thresholds, threshold_sha = E2E.read_thresholds(artifact_dir)
    design = E2E.build_design()
    composition = {}
    entry_files = {}
    for population in E2E.POPULATIONS:
        identifiers, source = E2E.load_entry_list(artifact_dir, population)
        counts = E2E.check_entry_list(identifiers, manifest, population)
        composition[population] = dict(zip(counts['bravais_lattice'], counts['n'].astype(int)))
        wanted = manifest.loc[manifest['identifier'].isin(identifiers),
                              ['identifier', 'bravais_lattice', 'split', 'volume_decile']]
        wanted = wanted.sort_values('identifier')
        path = artifact_dir/f'{E2E.TAG}_entries_{population}.csv'
        wanted.to_csv(path, index=False)
        entry_files[population] = dict(path=str(path), name=path.name, sha256=E2E.sha256_of(path),
                                       source=str(source), n=int(wanted.shape[0]))
        print(f'{population}: {wanted.shape[0]} crystals from {source.name} -> {path.name}')
    if args.pilot:
        identifiers, _ = E2E.load_entry_list(artifact_dir, 'general')
        chosen = E2E.pilot_entries(manifest, identifiers, per_lattice=2)
        path = artifact_dir/f'{E2E.TAG}_pilot_entries.csv'
        manifest.loc[manifest['identifier'].isin(chosen), ['identifier', 'bravais_lattice']] \
            .sort_values('identifier').to_csv(path, index=False)
        entry_files['pilot'] = dict(path=str(path), name=path.name, sha256=E2E.sha256_of(path),
                                    n=len(chosen))
        print(f'pilot: {len(chosen)} crystals -> {path.name}')
    learned = {name: dict(directory=path,
                          specification_sha256=E2E.sha256_of(Path(E2E.BASE/path)/'specification.json'))
               for name, path in E2E.LEARNED.items()}
    payload = dict(
        tag=E2E.TAG, cuts=list(E2E.CUTS), existing_cut=E2E.EXISTING_CUT,
        existing_bundles=list(E2E.EXISTING_BUNDLES), populations={
            name: dict(bundles=list(spec['bundles']), n_entries=spec['n_entries'],
                       lattices=list(spec['lattices']), composition=composition[name])
            for name, spec in E2E.POPULATIONS.items()},
        merits=list(E2E.MERITS), learned=learned, incumbent=list(E2E.INCUMBENT),
        thresholds=thresholds, threshold_table=E2E.THRESHOLD_TABLE, threshold_table_sha256=threshold_sha,
        seed=E2E.SEED, optimizer_seed=E2E.OPTIMIZER_SEED, pool_size=E2E.POOL_SIZE,
        search_seed_scheme='per_entry_bravais', n_top_candidates=E2E.N_TOP_CANDIDATES,
        pool_subsets=list(E2E.POOL_SUBSETS), metrics=list(E2E.METRICS),
        floor_sources=E2E.FLOOR_SOURCES, menu_rule=E2E.MENU_RULE,
        menu_hard_tolerance_se=E2E.MENU_HARD_TOLERANCE_SE,
        entry_files=entry_files, split_manifest_sha256=E2E.sha256_of(args.split_manifest),
        commit=E2E.commit_hash(), n_cells=int(design['n_cells'].sum()),
        n_cells_generated=int(design.loc[design['source'] == 'e2e', 'n_cells'].sum()),
        )
    path = artifact_dir/f'{E2E.TAG}_design.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    design.to_csv(artifact_dir/f'{E2E.TAG}_design.csv', index=False)
    print(f'design: {payload["n_cells"]} cells, {payload["n_cells_generated"]} to generate; '
          f'thresholds {thresholds}; -> {path.name}')


def _design(args):
    path = Path(args.artifact_dir)/f'{E2E.TAG}_design.json'
    if not path.exists():
        raise SystemExit(f'{path} missing; run --stage plan first')
    return json.loads(path.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------------------------
# generate / complete
# ---------------------------------------------------------------------------------------------
def _entries_file(args, design):
    """The population's entry list, found under THIS machine's artifact directory.

    The design records the list's path for provenance, and that path is the laptop's -- the design
    is written there and travels to NERSC with `sync_record.sh push`. Resolving the recorded path
    literally sent every task of the first grid submission looking under /Users/... on Perlmutter
    (2026-09-06). The basename under `--artifact-dir` is the same file, and the sha256 check below
    proves it.
    """
    if args.entries_file:
        return Path(args.entries_file)
    recorded = design['entry_files'][args.population]
    return Path(args.artifact_dir)/recorded.get('name', Path(recorded['path']).name)


def _arm_root(args):
    """The population directory an arm lives under; a variant arm gets its own name."""
    return Path(args.out_root)


def run_generate(args):
    from mlindex.scripts import run_fom_dump
    design = _design(args)
    if args.cut is None:
        raise SystemExit('--cut is required')
    population = args.population + args.arm_suffix
    entries_file = _entries_file(args, design)
    if not args.entries_file:
        recorded = design['entry_files'][args.population]['sha256']
        if E2E.sha256_of(entries_file) != recorded:
            raise SystemExit(f'{entries_file} differs from the list the design was planned with; '
                             f're-run --stage plan or pass --entries-file deliberately')
    extra = {}
    for item in args.opt_param or []:
        key, _, value = item.partition('=')
        extra[key] = value
    bundles = (list(E2E.POPULATIONS[args.population]['bundles']) if args.condition is None
               else [FomConditions.BY_KEY[args.condition].tag])
    directory = E2E.arm_dir(args.out_root, population, args.cut)
    if not (directory/'provenance.json').exists() and not args.dry_run:
        E2E.write_provenance(
            directory, tag=E2E.TAG, population=args.population, arm_suffix=args.arm_suffix,
            cut=float(args.cut), bundles=bundles, seed=E2E.SEED, optimizer_seed=E2E.OPTIMIZER_SEED,
            pool_size=args.pool_size, n_pools=args.n_pools, extra_opt_params=extra,
            entry_list=str(entries_file), entry_list_sha256=E2E.sha256_of(entries_file),
            design_sha256=E2E.sha256_of(Path(args.artifact_dir)/f'{E2E.TAG}_design.json'),
            split_manifest_sha256=E2E.sha256_of(args.split_manifest),
            models_dir=os.environ.get('MLINDEX_MODELS_DIR', 'package default'),
            started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    keys = {FomConditions.BY_KEY[k].tag: k for k in FomConditions.BY_KEY}
    for tag in bundles:
        target = E2E.bundle_dir(args.out_root, population, args.cut, tag)
        if (target/'manifest.json').exists():
            print(f'{population} {E2E.cut_label(args.cut)} {tag}: exists, skipping', flush=True)
            continue
        argv = E2E.generate_argv(population, args.cut, keys[tag], args.out_root, args.n_pools,
                                 args.split_manifest, entries_file, pool_size=args.pool_size,
                                 extra_opt_params=extra)
        print(f'=== {population} {E2E.cut_label(args.cut)} {tag}: run_fom_dump ' + ' '.join(argv),
              flush=True)
        if args.dry_run:
            continue
        run_fom_dump.main(argv)


def run_complete(args):
    if args.cut is None:
        raise SystemExit('--cut is required')
    population = args.population + args.arm_suffix
    # The arm's OWN bundle list, from the provenance `generate` wrote before its first bundle --
    # a pilot arm carries one bundle and is complete when that one is done; the grid arms carry
    # the population's full set.
    provenance = E2E.load_provenance(E2E.arm_dir(args.out_root, population, args.cut),
                                     require_complete=False)
    bundles = list(provenance.get('bundles') or E2E.POPULATIONS[args.population]['bundles'])
    done = E2E.arm_bundles_done(args.out_root, population, args.cut, bundles)
    if set(done) != set(bundles):
        raise SystemExit(f'{population} {E2E.cut_label(args.cut)}: {len(done)} of {len(bundles)} '
                         f'bundles have a manifest; not stamping a partial arm')
    seconds = {}
    for tag in bundles:
        manifest = FomBenchmark.load_manifest(E2E.bundle_dir(args.out_root, population, args.cut, tag))
        seconds[tag] = manifest.get('seconds_total')
    # A digest-for-digest check within the arm is cheap here and catches a bundle generated under
    # another seed before anything is consolidated.
    payload = E2E.stamp_complete(E2E.arm_dir(args.out_root, population, args.cut),
                                 bundle_seconds_total=seconds,
                                 completed=time.strftime('%Y-%m-%dT%H:%M:%S'))
    print(f'stamped {population} {E2E.cut_label(args.cut)} complete: {payload["bundle_seconds_total"]}')


# ---------------------------------------------------------------------------------------------
# sidecars
# ---------------------------------------------------------------------------------------------
def run_sidecars(args):
    if args.cut is None:
        raise SystemExit('--cut is required')
    population = args.population + args.arm_suffix
    directory = E2E.arm_dir(args.out_root, population, args.cut)
    E2E.load_provenance(directory)
    pool = E2E.pool_dir(args.out_root, population, args.cut)
    groups = E2E.learned_groups(_learned(args))
    for command in E2E.sidecar_commands(directory, pool, args.processes, groups, python=args.python):
        print('$ ' + ' '.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


# ---------------------------------------------------------------------------------------------
# reduce / restrict
# ---------------------------------------------------------------------------------------------
def _keep(args):
    if not args.keep_entries:
        return None
    frame = pd.read_csv(args.keep_entries)
    column = 'identifier' if 'identifier' in frame.columns else 'entry_id'
    return set(frame[column].astype(str))


def _announce(started):
    def announce(frame):
        print(f'  {frame["condition_bundle"].iloc[0]:28s} {frame.shape[0]:>10,} candidates '
              f'({time.perf_counter() - started:.0f} s)', flush=True)
    return announce


def _pools(args):
    if args.pool:
        return [Path(p) for p in args.pool]
    if args.cut is None:
        raise SystemExit('--pool or --cut is required')
    return [E2E.pool_dir(args.out_root, args.population + args.arm_suffix, args.cut)]


def run_reduce(args, row_filter=None, suffix=None):
    if args.cut is None:
        raise SystemExit('--cut is required')
    learned = _learned(args)
    scores = E2E.load_scores(learned)
    groups = E2E.learned_groups(learned)
    keep = _keep(args)
    started = time.perf_counter()
    parts = []
    for pool in _pools(args):
        entries = FomBenchmark.load_entries(pool)
        if keep is not None:
            entries = entries.loc[entries['entry_id'].astype(str).isin(keep)]
        print(f'reducing {len(scores)} scores over {pool} '
              f'({entries["entry_id"].nunique()} crystals, '
              f'{len(FomBenchmark.available_bundles(pool))} bundles)', flush=True)
        parts.append(E2E.reduce_arm(pool, scores, entries, groups, on_shard=_announce(started),
                                    row_filter=row_filter, keep_entry_ids=keep,
                                    extra_columns=('m20_at_prune',) if row_filter else ()))
    reduced = E2E.concatenate_reductions(parts) if len(parts) > 1 else parts[0]
    population = args.population + args.arm_suffix
    path = E2E.write_reductions(reduced, args.artifact_dir, population, args.cut,
                                suffix=args.suffix if suffix is None else suffix)
    for name, (per_entry, meta) in sorted(reduced.items()):
        print(f'  {name:16s} {per_entry.shape[0]:>6,} cells, ranks exact {meta["ranks_exact"]}')
    print(f'-> {path} ({time.perf_counter() - started:.0f} s)')


def run_restrict(args):
    """Replay the higher cuts on the cut-1.5 pool, one reduction per cut, suffixed `_restricted`."""
    for cut in [float(c) for c in args.cuts.split(',')]:
        args.cut = cut
        print(f'=== restriction at {E2E.cut_label(cut)} ===', flush=True)
        run_reduce(args, row_filter=partial(E2E.restrict_at_cut, cut=cut),
                   suffix=args.suffix + '_restricted')


# ---------------------------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------------------------
def _existing_pools(args):
    out = {}
    for item in args.existing_pool or []:
        key, _, path = item.partition('=')
        population, _, cut = key.partition(':')
        out.setdefault((population, float(cut)), []).append(Path(path))
    if not out:
        default = E2E.BASE/'mlindex'/'data'/'fom_full_c2_pool'
        if default.exists():
            out[('general', E2E.EXISTING_CUT)] = [default]
    return out


def _arms(args, design):
    """Every (population, cut) with a reduction on disk, in cut order, plus the existing arm."""
    arms = []
    for population in E2E.POPULATIONS:
        cuts = list(design['cuts']) + ([design['existing_cut']] if population == 'general' else [])
        for cut in sorted(cuts, reverse=True):
            if E2E.load_reductions(args.artifact_dir, population, cut, args.suffix) is not None:
                arms.append((population, float(cut)))
    return arms


def run_analyse(args):
    design = _design(args)
    artifact_dir = Path(args.artifact_dir)
    thresholds = {name: spec['threshold'] for name, spec in design['thresholds'].items()}
    floors = E2E.load_floor_tables(artifact_dir)
    arms = _arms(args, design)
    if not arms:
        raise SystemExit('no reductions found; run --stage reduce first')
    print(f'arms with reductions: {arms}')

    # 1. Every result, summarised at both depths, per (population, cut, score).
    results = {}
    reductions = {}
    for population, cut in arms:
        reductions[(population, cut)] = E2E.load_reductions(artifact_dir, population, cut, args.suffix)

    # Pair every cut of a population on the cells all its cuts share, so a cross-cut contrast is
    # over one entry set. The count dropped is reported, not hidden.
    for population in E2E.POPULATIONS:
        keys = [k for k in reductions if k[0] == population]
        if not keys:
            continue
        frames = [reductions[k][name][0] for k in keys for name in reductions[k]]
        common = E2E.common_keys(frames)
        for k in keys:
            for name, (per_entry, meta) in reductions[k].items():
                restricted, dropped = E2E.restrict_per_entry(per_entry, common)
                meta = dict(meta, n_dropped_unpaired=dropped)
                reductions[k][name] = (restricted, meta)
        print(f'{population}: {len(common)} cells common to {len(keys)} arm(s)')

    levels, contrasts = [], []
    for (population, cut), reduced in reductions.items():
        for pool_subset in E2E.POOL_SUBSETS:
            for name in REPORTED:
                if name not in reduced:
                    continue
                per_entry, meta = reduced[name]
                result = E2E.summarise(per_entry, meta, thresholds.get(name), pool_subset,
                                       n_bootstrap=args.n_bootstrap)
                results[(population, cut, name, pool_subset)] = result
                ids = dict(population=population, cut=cut, merit=name, pool_subset=pool_subset,
                           threshold=thresholds.get(name), condition_bundle='all',
                           is_real_run=True, source='existing' if cut == design['existing_cut'] else 'e2e',
                           n_dropped_unpaired=meta.get('n_dropped_unpaired', 0))
                levels.append(E2E.level_row(result, 'aggregate', **ids))
                if pool_subset == 'in_top_n' or True:
                    levels.append(E2E.level_row(result, 'hard', **ids))
                levels += E2E.stratum_rows(result, 'bravais_lattice', **ids)
                levels += E2E.stratum_rows(result, 'condition_bundle', **ids)
    levels = pd.DataFrame(levels)

    def add_contrast(kind, reference_key, arm_key):
        reference, arm = results.get(reference_key), results.get(arm_key)
        if reference is None or arm is None:
            return
        population, ref_cut, ref_merit, pool_subset = reference_key
        _, cut, merit, _ = arm_key
        masks = E2E.scope_masks(arm)
        for scope, mask in masks.items():
            for metric in E2E.METRICS:
                row = E2E.contrast(reference, arm, metric, mask)
                if row is None:
                    continue
                floor_pp, source = E2E.floor_for(floors, metric, scope if population == 'general'
                                                 else ('hard' if scope == 'aggregate' else scope))
                row.update(population=population, contrast_kind=kind, reference_cut=ref_cut,
                           reference_merit=ref_merit, cut=cut, merit=merit, pool_subset=pool_subset,
                           scope=scope, floor_pp=floor_pp, floor_source=source,
                           standard_errors=E2E.in_floor_ses(row['delta_pp'], floor_pp))
                contrasts.append(row)

    for (population, cut, merit, pool_subset) in list(results):
        # merit at fixed cut
        for reference_merit in REPORTED:
            if reference_merit != merit and REPORTED.index(reference_merit) < REPORTED.index(merit):
                add_contrast('merit', (population, cut, reference_merit, pool_subset),
                             (population, cut, merit, pool_subset))
        # cut at fixed merit, against every higher cut
        for reference_cut in sorted({k[1] for k in results if k[0] == population}, reverse=True):
            if reference_cut > cut:
                add_contrast('cut', (population, reference_cut, merit, pool_subset),
                             (population, cut, merit, pool_subset))
        # the pair against the incumbent
        incumbent = (population, float(E2E.INCUMBENT[0]), E2E.INCUMBENT[1], pool_subset)
        if (cut, merit) != (float(E2E.INCUMBENT[0]), E2E.INCUMBENT[1]):
            add_contrast('pair', incumbent, (population, cut, merit, pool_subset))
    contrasts = pd.DataFrame(contrasts)

    levels.to_csv(artifact_dir/f'{E2E.TAG}_factorial{args.suffix}.csv', index=False)
    contrasts.to_csv(artifact_dir/f'{E2E.TAG}_factorial_contrasts{args.suffix}.csv', index=False)
    print(f'factorial: {levels.shape[0]} level rows, {contrasts.shape[0]} contrast rows')

    # 2. Success curves: the bundle-scope levels placed on the benchmark's axes.
    axes = E2E.success_curve_axes()
    curve_rows = []
    bundle_levels = levels.loc[levels['scope'].str.startswith('condition_bundle=')].copy()
    bundle_levels['bundle'] = bundle_levels['scope'].str.split('=', n=1).str[1]
    for _, point in axes.iterrows():
        hit = bundle_levels.loc[bundle_levels['bundle'] == point['condition_bundle']]
        for _, line in hit.iterrows():
            for metric in ('operating_point', 'top10'):
                curve_rows.append(dict(population=line['population'], cut=line['cut'],
                                       merit=line['merit'], pool_subset=line['pool_subset'],
                                       axis=point['axis'], x=point['x'],
                                       condition_bundle=point['condition_bundle'],
                                       caveat=point['caveat'], metric=metric,
                                       value=line[metric],
                                       ci_low=line.get(f'{metric}_ci_low', np.nan),
                                       ci_high=line.get(f'{metric}_ci_high', np.nan),
                                       n_entries=line['n_entries']))
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(artifact_dir/f'{E2E.TAG}_success_curves{args.suffix}.csv', index=False)
    print(f'success curves: {curves.shape[0]} rows')

    # 3. Cost: measured on the pools that produced the arms, never projected.
    cost = _cost_table(args, design, reductions)
    cost.to_csv(artifact_dir/f'{E2E.TAG}_cost{args.suffix}.csv', index=False)

    # 4. Digest check and manifest identity, over every arm whose pool is readable here.
    _digest_check(args, design, artifact_dir)

    # 5. Restriction versus run.
    _restriction_table(args, design, thresholds, artifact_dir)

    # 6. The menu.
    menu = E2E.build_menu(levels, contrasts, cost=cost)
    menu.to_csv(artifact_dir/f'{E2E.TAG}_deployment_menu{args.suffix}.csv', index=False)
    print('menu:')
    show = ['cut', 'merit', 'pool_subset', 'general_op', 'hard_op',
            'general_standard_errors_vs_incumbent', 'hard_standard_errors_vs_incumbent',
            'worst_lattice', 'worst_lattice_standard_errors', 'seconds_per_entry', 'recommended']
    print(menu[[c for c in show if c in menu.columns]].to_string(index=False))


def _pool_paths(args, design):
    """{(population, cut): [pool roots]} for every arm whose pool is readable on this machine."""
    pools = dict(_existing_pools(args))
    for population in E2E.POPULATIONS:
        for cut in design['cuts']:
            path = E2E.pool_dir(args.out_root, population, cut)
            if (path/'manifest.json').exists():
                pools[(population, float(cut))] = [path]
    return pools


def _load_arm_entries(args, paths):
    entries = pd.concat([FomBenchmark.load_entries(p) for p in paths], ignore_index=True)
    keep = _keep(args)
    if keep is not None:
        entries = entries.loc[entries['entry_id'].astype(str).isin(keep)].reset_index(drop=True)
    return entries


def _cost_table(args, design, reductions):
    rows = []
    for (population, cut), paths in _pool_paths(args, design).items():
        entries = _load_arm_entries(args, paths)
        manifests = [FomBenchmark.load_manifest(p) for p in paths]
        inner = []
        for manifest in manifests:
            inner += list(manifest.get('bundle_manifests', {}).values()) or [manifest]
        node_seconds = sum(float(m.get('seconds_total') or 0) for m in inner)
        n_pools = sorted({m.get('n_pools') for m in inner})
        machine = sorted({m.get('arch') for m in inner})
        timed = entries['seconds_search'] >= 0 if 'seconds_search' in entries.columns else \
            pd.Series(False, index=entries.index)
        reduced = reductions.get((population, cut), {})
        in_top = None
        if 'M20' in reduced:
            per_entry = reduced['M20'][0]
            in_top = per_entry[['entry_id', 'condition_bundle', 'n_candidates_in_top_n']]
        for bundle in ['all'] + sorted(entries['condition_bundle'].unique()):
            subset = entries if bundle == 'all' else entries.loc[entries['condition_bundle'] == bundle]
            mask = timed.loc[subset.index]
            row = dict(population=population, cut=float(cut), condition_bundle=bundle,
                       n_entries=int(subset.shape[0]), n_timed=int(mask.sum()),
                       seconds_per_entry_mean=(float(subset.loc[mask, 'seconds_total'].mean())
                                               if mask.any() else np.nan),
                       seconds_per_entry_median=(float(subset.loc[mask, 'seconds_total'].median())
                                                 if mask.any() else np.nan),
                       seconds_search_median=(float(subset.loc[mask, 'seconds_search'].median())
                                              if mask.any() else np.nan),
                       pool_size_full_median=float(subset['pool_size_full'].median()),
                       pool_size_full_mean=float(subset['pool_size_full'].mean()),
                       n_candidates_total=int(subset['pool_size_full'].sum()),
                       node_hours=(node_seconds/3600 if bundle == 'all' else np.nan),
                       n_pools=','.join(str(n) for n in n_pools), machine=','.join(map(str, machine)),
                       source='entries_seconds' if mask.any() else 'manifest_only')
            if in_top is not None:
                hit = in_top if bundle == 'all' else in_top.loc[in_top['condition_bundle'] == bundle]
                row['in_top_n_size_median'] = float(hit['n_candidates_in_top_n'].median())
            rows.append(row)
    cost = pd.DataFrame(rows)
    if cost.shape[0]:
        base = cost.loc[(cost['population'] == 'general') & (cost['cut'] == float(E2E.INCUMBENT[0]))
                        & (cost['condition_bundle'] == 'all'), 'seconds_per_entry_median']
        base = float(base.iloc[0]) if base.shape[0] else np.nan
        cost['seconds_vs_cut5_pct'] = 100*(cost['seconds_per_entry_median']/base - 1)
    print(f'cost: {cost.shape[0]} rows')
    return cost


def _digest_check(args, design, artifact_dir):
    pools = _pool_paths(args, design)
    tables, identities = [], []
    for population in E2E.POPULATIONS:
        by_cut = {cut: _load_arm_entries(args, paths)
                  for (pop, cut), paths in pools.items() if pop == population}
        manifests = {cut: FomBenchmark.load_manifest(paths[0])
                     for (pop, cut), paths in pools.items() if pop == population}
        if len(by_cut) < 2:
            print(f'{population}: {len(by_cut)} arm(s) readable here, digest check skipped')
            continue
        try:
            table = E2E.check_peak_digests(by_cut)
            table.insert(0, 'population', population)
            table['status'] = 'agree'
        except ValueError as error:
            table = pd.DataFrame([dict(population=population, status=f'FAIL: {error}')])
        tables.append(table)
        try:
            identity = E2E.check_manifest_identity(manifests)
            identity.insert(0, 'population', population)
            identity['status'] = 'agree'
        except ValueError as error:
            identity = pd.DataFrame([dict(population=population, status=f'FAIL: {error}')])
        identities.append(identity)
    if tables:
        out = pd.concat(tables, ignore_index=True)
        out.to_csv(artifact_dir/f'{E2E.TAG}_digest_check{args.suffix}.csv', index=False)
        print(out.to_string(index=False))
    if identities:
        pd.concat(identities, ignore_index=True).to_csv(
            artifact_dir/f'{E2E.TAG}_manifest_identity{args.suffix}.csv', index=False)


def _restriction_table(args, design, thresholds, artifact_dir):
    rows = []
    for cut in design['cuts']:
        restricted = E2E.load_reductions(artifact_dir, 'general', cut, args.suffix + '_restricted')
        real = E2E.load_reductions(artifact_dir, 'general', cut, args.suffix)
        if restricted is None or real is None:
            continue
        for name in REPORTED:
            if name not in restricted or name not in real:
                continue
            common = E2E.common_keys([restricted[name][0], real[name][0]])
            for pool_subset in E2E.POOL_SUBSETS:
                a = E2E.summarise(*E2E.restrict_per_entry(restricted[name][0], common)[:1],
                                  restricted[name][1], thresholds.get(name), pool_subset,
                                  n_bootstrap=0)
                b = E2E.summarise(*E2E.restrict_per_entry(real[name][0], common)[:1],
                                  real[name][1], thresholds.get(name), pool_subset, n_bootstrap=0)
                for metric in ('found', 'top10', 'operating_point'):
                    row = E2E.contrast(a, b, metric)
                    if row is None:
                        continue
                    row.update(cut=float(cut), merit=name, pool_subset=pool_subset,
                               restricted_rate=a.metric(metric) if metric != 'found'
                               else a.metric('ceiling_rescorer'),
                               real_rate=b.metric(metric) if metric != 'found'
                               else b.metric('ceiling_rescorer'), n_cells=len(common))
                    rows.append(row)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(artifact_dir/f'{E2E.TAG}_restriction_vs_run{args.suffix}.csv', index=False)
        print(f'restriction vs run: {frame.shape[0]} rows')
    else:
        print('restriction vs run: no restricted reductions found (run --stage restrict)')


# ---------------------------------------------------------------------------------------------
# figure / report
# ---------------------------------------------------------------------------------------------
MERIT_COLOURS = {'M20': '#8a817c', 'M_sym': '#1b4965', 'plus_probation': '#e09f3e'}
CUT_STYLES = {5.0: ('-', 'o'), 3.5: ('--', 's'), 3.0: (':', '^'), 1.5: ('-.', 'd')}
AXIS_LABELS = {'error_scale': 'peak-position error, x nominal',
               'contaminant_count': 'unindexable lines in the window',
               'dropout': 'interior peaks dropped (with one contaminant)'}


def run_figure(args):
    from mlindex.model_training.FomHoldoutReport import _style
    plt = _style()
    artifact_dir = Path(args.artifact_dir)
    curves = pd.read_csv(artifact_dir/f'{E2E.TAG}_success_curves{args.suffix}.csv')
    menu = pd.read_csv(artifact_dir/f'{E2E.TAG}_deployment_menu{args.suffix}.csv')
    curves = curves.loc[(curves['pool_subset'] == 'in_top_n') & (curves['metric'] == 'operating_point')]
    populations = [p for p in ('general', 'hard') if p in set(curves['population'])]
    axes_names = list(AXIS_LABELS)
    fig, panels = plt.subplots(len(populations), 3, figsize=(11, 3.4*len(populations)),
                               squeeze=False, sharey='row')
    recommended = set(zip(menu.loc[menu['recommended'] & (menu['pool_subset'] == 'in_top_n'), 'cut'],
                          menu.loc[menu['recommended'] & (menu['pool_subset'] == 'in_top_n'), 'merit']))
    for row, population in enumerate(populations):
        for col, axis in enumerate(axes_names):
            ax = panels[row][col]
            sub = curves.loc[(curves['population'] == population) & (curves['axis'] == axis)]
            for (merit, cut), line in sub.groupby(['merit', 'cut']):
                line = line.sort_values('x')
                style, marker = CUT_STYLES.get(float(cut), ('-', 'o'))
                colour = MERIT_COLOURS.get(merit, '#444444')
                ax.plot(line['x'], 100*line['value'], linestyle=style, color=colour,
                        linewidth=1.2 if (float(cut), merit) in recommended else 0.9,
                        label=f'{merit}, cut {cut:g}')
                # Per point: a caveated point (it moves more than one thing) is hollow.
                caveated = line['caveat'].fillna('').astype(str) != ''
                ax.plot(line.loc[~caveated, 'x'], 100*line.loc[~caveated, 'value'], linestyle='none',
                        marker=marker, color=colour, markersize=4)
                ax.plot(line.loc[caveated, 'x'], 100*line.loc[caveated, 'value'], linestyle='none',
                        marker=marker, color=colour, markerfacecolor='white', markersize=4)
                if (float(cut), merit) in recommended and line['ci_low'].notna().any():
                    ax.fill_between(line['x'], 100*line['ci_low'], 100*line['ci_high'],
                                    color=colour, alpha=0.12, linewidth=0)
            ax.set_xlabel(AXIS_LABELS[axis])
            if col == 0:
                ax.set_ylabel(f'{population}: operating point, %')
            if axis == 'error_scale':
                ax.set_xscale('log')
            ax.set_title(f'{population} - {axis.replace("_", " ")}')
    handles, labels = panels[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(6, max(1, len(labels))), fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('S15 - success rate on real runs, by cut and ranking merit '
                 '(production list depth; hollow markers: the point moves more than one thing)',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = artifact_dir/f'{E2E.TAG}_success_curves{args.suffix}.png'
    fig.savefig(path, bbox_inches='tight')
    print(f'-> {path}')

    # The factorial as two heat-maps.
    levels = pd.read_csv(artifact_dir/f'{E2E.TAG}_factorial{args.suffix}.csv')
    contrasts = pd.read_csv(artifact_dir/f'{E2E.TAG}_factorial_contrasts{args.suffix}.csv')
    agg = levels.loc[(levels['scope'] == 'aggregate') & (levels['pool_subset'] == 'in_top_n')]
    pair = contrasts.loc[(contrasts['contrast_kind'] == 'pair') & (contrasts['scope'] == 'aggregate')
                         & (contrasts['metric'] == 'operating_point')
                         & (contrasts['pool_subset'] == 'in_top_n')]
    fig, panels = plt.subplots(1, len(populations), figsize=(4.2*len(populations), 3.2), squeeze=False)
    for col, population in enumerate(populations):
        ax = panels[0][col]
        table = agg.loc[agg['population'] == population].pivot(index='cut', columns='merit',
                                                               values='operating_point')
        table = table.reindex(index=sorted(table.index, reverse=True),
                              columns=[m for m in REPORTED if m in table.columns])
        image = ax.imshow(100*table.to_numpy(dtype=float), cmap='viridis', aspect='auto')
        ax.set_xticks(range(table.shape[1]), table.columns)
        ax.set_yticks(range(table.shape[0]), [f'{c:g}' for c in table.index])
        ax.set_xlabel('ranking merit')
        ax.set_ylabel('prune cut')
        ax.set_title(f'{population}: operating point, % (production depth)')
        ax.grid(False)
        for i, cut in enumerate(table.index):
            for j, merit in enumerate(table.columns):
                value = table.loc[cut, merit]
                se = pair.loc[(pair['population'] == population) & (pair['cut'] == cut)
                              & (pair['merit'] == merit), 'standard_errors']
                text = f'{100*value:.1f}' + (f'\n{float(se.iloc[0]):+.1f} se' if se.shape[0] else '')
                ax.text(j, i, text, ha='center', va='center', fontsize=7,
                        color='white' if value < np.nanmean(table.to_numpy(dtype=float)) else 'black')
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = artifact_dir/f'{E2E.TAG}_factorial{args.suffix}.png'
    fig.savefig(path, bbox_inches='tight')
    print(f'-> {path}')


def _table(frame, columns, formats=None):
    from mlindex.model_training.FomHoldoutReport import _table as table
    return table(frame, columns, formats)


def run_report(args):
    design = _design(args)
    artifact_dir = Path(args.artifact_dir)
    suffix = args.suffix
    read = lambda name: pd.read_csv(artifact_dir/f'{E2E.TAG}_{name}{suffix}.csv')  # noqa: E731
    levels, contrasts, menu, cost = read('factorial'), read('factorial_contrasts'), \
        read('deployment_menu'), read('cost')
    curves = read('success_curves')
    digest = (read('digest_check') if (artifact_dir/f'{E2E.TAG}_digest_check{suffix}.csv').exists()
              else None)
    restriction = (read('restriction_vs_run')
                   if (artifact_dir/f'{E2E.TAG}_restriction_vs_run{suffix}.csv').exists() else None)
    pct = lambda v: '' if pd.isna(v) else f'{100*v:.2f}'  # noqa: E731
    pp = lambda v: '' if pd.isna(v) else f'{v:+.2f}'  # noqa: E731
    se = lambda v: '' if pd.isna(v) else f'{v:+.1f}'  # noqa: E731
    g = lambda v: '' if pd.isna(v) else f'{v:g}'  # noqa: E731

    parts = [f'# {E2E.TAG} - end-to-end runs and the deployment recommendation', '',
             f'**Design:** `{E2E.TAG}_design.json` - commit `{design["commit"]}`; cuts '
             f'{design["cuts"]} run for real, {design["existing_cut"]} read from the existing pools; '
             f'seed {design["seed"]}, optimizer seed {design["optimizer_seed"]}, pool size '
             f'{design["pool_size"]}, `{design["search_seed_scheme"]}`. Thresholds are S12\'s, frozen: '
             + ', '.join(f'{k} {v["threshold"]:.4g} ({v["threshold_rule"]})'
                         for k, v in design['thresholds'].items())
             + f' - from `{design["threshold_table"]}` (sha256 {design["threshold_table_sha256"][:12]}). '
             '**Nothing here is tuned.**', '',
             f'![success curves]({E2E.TAG}_success_curves{suffix}.png)', '',
             '## 0. How to read this', '',
             'Every number is from a **real run** of the indexer at the stated cut, through the '
             'benchmark generator (the same optimizer code, the cut reaching it as '
             '`opt_params[\'prune_m20_threshold\']`), every candidate kept. A merit is a score over '
             'that pool. Two depths: **`all`** is the whole post-deduplication pool a re-ranker '
             'would see; **`in_top_n`** is the top twenty per lattice that production hands to its '
             'final sort - a strict superset of the printed list, because `run.py` still promotes '
             'and deduplicates across lattices afterwards, which can only raise a correct cell\'s '
             'rank. The **operating point** is the correct cell in the pooled top ten AND above the '
             'merit\'s frozen threshold; **top-10** is the rank half alone. Every contrast is paired '
             'over the same (crystal, condition) cells with McNemar and a cluster bootstrap over '
             'crystals, and quoted in **standard errors of the contrast floor** - S08\'s top-10 '
             'floor and S09\'s operating-point floor, per lattice where the claim is per lattice. '
             'The hard population has no measured floor of its own; its claims are read against '
             'the mean of the hard lattices\' floors and say so.', '']

    if digest is not None:
        parts += ['## 1. Gate 5 - the arms differ in nothing but the cut', '',
                  _table(digest, [c for c in ('population', 'condition_bundle', 'cuts_compared',
                                              'n_cells', 'n_agree', 'n_disagree', 'n_missing',
                                              'note', 'status') if c in digest.columns]), '']

    parts += ['## 2. The factorial: cut x merit, both populations', '']
    for pool_subset in E2E.POOL_SUBSETS:
        for population in ('general', 'hard'):
            agg = levels.loc[(levels['scope'] == 'aggregate') & (levels['population'] == population)
                             & (levels['pool_subset'] == pool_subset)]
            if agg.empty:
                continue
            pair = contrasts.loc[(contrasts['contrast_kind'] == 'pair') & (contrasts['scope'] == 'aggregate')
                                 & (contrasts['population'] == population)
                                 & (contrasts['pool_subset'] == pool_subset)]
            table = agg.merge(pair.loc[pair['metric'] == 'operating_point',
                                       ['cut', 'merit', 'delta_pp', 'ci_low_pp', 'ci_high_pp',
                                        'gained', 'lost', 'p_value', 'standard_errors']],
                              on=['cut', 'merit'], how='left').sort_values(['cut', 'merit'],
                                                                           ascending=[False, True])
            parts += [f'### {population}, depth `{pool_subset}` ({int(agg["n_entries"].max())} cells)', '',
                      _table(table, ['cut', 'merit', 'operating_point', 'top10', 'ceiling_rescorer',
                                     'reported', 'precision', 'delta_pp', 'ci_low_pp', 'ci_high_pp',
                                     'gained', 'lost', 'p_value', 'standard_errors'],
                             {'cut': g, 'operating_point': pct, 'top10': pct, 'ceiling_rescorer': pct,
                              'reported': pct, 'precision': pct, 'delta_pp': pp, 'ci_low_pp': pp,
                              'ci_high_pp': pp, 'standard_errors': se,
                              'p_value': lambda v: '' if pd.isna(v) else f'{v:.2g}',
                              'gained': lambda v: '' if pd.isna(v) else f'{int(v)}',
                              'lost': lambda v: '' if pd.isna(v) else f'{int(v)}'}),
                      '', '`delta_pp` .. `standard_errors`: the pair against the incumbent '
                      '(5.0, M20) on the operating point.', '']

    parts += ['## 3. Per lattice, the recommended pair against the incumbent', '']
    rec = menu.loc[menu['recommended'] & (menu['pool_subset'] == 'in_top_n')]
    for _, line in rec.iterrows():
        lat = contrasts.loc[(contrasts['contrast_kind'] == 'pair') & (contrasts['population'] == 'general')
                            & (contrasts['cut'] == line['cut']) & (contrasts['merit'] == line['merit'])
                            & (contrasts['pool_subset'] == 'in_top_n')
                            & contrasts['scope'].str.startswith('bravais_lattice=')
                            & (contrasts['metric'] == 'operating_point')].copy()
        if lat.empty:
            continue
        lat['lattice'] = lat['scope'].str.split('=', n=1).str[1]
        parts += [f'### ({line["cut"]:g}, {line["merit"]}) vs (5.0, M20), operating point, `in_top_n`', '',
                  _table(lat, ['lattice', 'delta_pp', 'ci_low_pp', 'ci_high_pp', 'gained', 'lost',
                               'p_value', 'floor_pp', 'standard_errors', 'n_entries'],
                         {'delta_pp': pp, 'ci_low_pp': pp, 'ci_high_pp': pp, 'standard_errors': se,
                          'floor_pp': lambda v: f'{v:.2f}',
                          'p_value': lambda v: f'{v:.2g}'}), '']

    parts += ['## 4. Success curves', '',
              'The benchmark\'s own condition axes. A hollow marker (in the figure) and a caveat '
              '(here) mark a point that moves more than one thing: the contaminant-count point at '
              'x = 1 also drops two peaks, and the dropout series carries one contaminant where its '
              'x = 0 reference does not. `error_shape` and `second_phase` are factorial rows, not '
              'curve points.', '']
    c = curves.loc[(curves['pool_subset'] == 'in_top_n') & (curves['metric'] == 'operating_point')]
    if not c.empty:
        wide = c.pivot_table(index=['population', 'axis', 'x', 'condition_bundle', 'caveat'],
                             columns=['merit', 'cut'], values='value').reset_index()
        wide.columns = ['_'.join(str(p) for p in col if str(p) != '') if isinstance(col, tuple) else col
                        for col in wide.columns]
        parts += [_table(wide, list(wide.columns), {col: pct for col in wide.columns
                                                     if col not in ('population', 'axis', 'x',
                                                                    'condition_bundle', 'caveat')}), '']

    if restriction is not None:
        parts += ['## 5. A restriction is not a run', '',
                  'The cut-1.5 pool restricted at each cut (`prune_below_m20` replayed on the '
                  'stored `m20_at_prune`) against the real run at that cut, paired. Campaign 1 '
                  'measured five entries of 210 in one direction; this is the same comparison on '
                  'this pool.', '',
                  _table(restriction, ['cut', 'merit', 'pool_subset', 'metric', 'restricted_rate',
                                       'real_rate', 'delta_pp', 'gained', 'lost', 'p_value', 'n_cells'],
                         {'cut': g, 'restricted_rate': pct, 'real_rate': pct, 'delta_pp': pp,
                          'p_value': lambda v: f'{v:.2g}'}), '',
                  '`delta_pp` is real minus restricted: positive means the real run reaches more.', '']

    parts += ['## 6. Cost, measured', '',
              _table(cost.loc[cost['condition_bundle'] == 'all'],
                     ['population', 'cut', 'n_entries', 'n_timed', 'seconds_per_entry_median',
                      'seconds_vs_cut5_pct', 'pool_size_full_median', 'in_top_n_size_median',
                      'node_hours', 'n_pools', 'machine', 'source'],
                     {'cut': g, 'seconds_per_entry_median': lambda v: f'{v:.1f}',
                      'seconds_vs_cut5_pct': lambda v: '' if pd.isna(v) else f'{v:+.1f}',
                      'pool_size_full_median': lambda v: f'{v:.0f}',
                      'in_top_n_size_median': lambda v: '' if pd.isna(v) else f'{v:.0f}',
                      'node_hours': lambda v: '' if pd.isna(v) else f'{v:.2f}'}), '',
              'Seconds per entry are as ONE pool saw them (the fourteen-lattice search plus '
              'bookkeeping), on the machine named; node throughput divides by `n_pools`. A row '
              'with `manifest_only` was generated before `--record-timing` existed and carries no '
              'per-entry time.', '']

    parts += ['## 7. The deployment menu', '',
              f'Rule: {E2E.MENU_RULE} (tolerance {E2E.MENU_HARD_TOLERANCE_SE} se). The rule is a '
              'decision recorded in STATUS.md, not a search over this table.', '',
              _table(menu.loc[menu['pool_subset'] == 'in_top_n'],
                     ['cut', 'merit', 'general_op', 'general_top10', 'general_ceiling', 'hard_op',
                      'hard_top10', 'hard_ceiling', 'general_delta_pp_vs_incumbent',
                      'general_standard_errors_vs_incumbent', 'hard_standard_errors_vs_incumbent',
                      'worst_lattice', 'worst_lattice_standard_errors', 'seconds_per_entry',
                      'seconds_vs_incumbent_pct', 'pool_size_median', 'recommended'],
                     {'cut': g, 'general_op': pct, 'general_top10': pct, 'general_ceiling': pct,
                      'hard_op': pct, 'hard_top10': pct, 'hard_ceiling': pct,
                      'general_delta_pp_vs_incumbent': pp,
                      'general_standard_errors_vs_incumbent': se,
                      'hard_standard_errors_vs_incumbent': se, 'worst_lattice_standard_errors': se,
                      'seconds_per_entry': lambda v: '' if pd.isna(v) else f'{v:.1f}',
                      'seconds_vs_incumbent_pct': lambda v: '' if pd.isna(v) else f'{v:+.1f}',
                      'pool_size_median': lambda v: '' if pd.isna(v) else f'{v:.0f}'}), '',
              '## 8. Bounds', '',
              '- `in_top_n` is a strict superset of the printed list; ranks there bound the '
              'printed rank from above.',
              '- The hard population has no measured contrast floor; the hard lattices\' own floors '
              'from the general population stand in and are labelled.',
              '- One fit seed for the learned arm (S12\'s bound). C2-Q-032\'s block-A form is not '
              'in the grid; it can be scored on these pools without regenerating anything.',
              '- `error_shape` has no cut-1.5 arm; second-phase and error-shape are factorial rows, '
              'not curve points.',
              '- Cost rows are per pool on the machine named, not per node.', '']
    path = artifact_dir/f'{E2E.TAG}_end_to_end{suffix}.md'
    path.write_text('\n'.join(parts), encoding='utf-8')
    print(f'-> {path}')


def main(argv=None):
    args = _parse_args(argv)
    {'plan': run_plan, 'generate': run_generate, 'complete': run_complete,
     'sidecars': run_sidecars, 'reduce': run_reduce, 'restrict': run_restrict,
     'analyse': run_analyse, 'figure': run_figure, 'report': run_report}[args.stage](args)


if __name__ == '__main__':
    main()
