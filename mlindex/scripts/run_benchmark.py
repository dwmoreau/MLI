"""Measure the indexer on a benchmark pool, and compare two ways of ranking it.

This is the instrument for "did that change help?". It reads a stored pool of candidate unit
cells, ranks them by one or more scores, reduces the ranking to one outcome per pattern-condition
-- one crystal under one noise condition -- and reports how often the correct cell reaches the top
of the list. Two arms are compared paired, against the run-to-run noise of the search itself.

    # every score the pool can carry, aggregate and per lattice, written to a directory
    python -m mlindex.scripts.run_benchmark --pool mlindex/data/my_pool --out-dir results/run1

    # one bundle, a couple of scores, a quick look on the terminal
    python -m mlindex.scripts.run_benchmark --pool mlindex/data/my_pool \
        --bundles b1_error1_cont0 --scores M20,M_sym

    # cap the work to 40 source crystals, chosen reproducibly, for a laptop-sized check
    python -m mlindex.scripts.run_benchmark --pool mlindex/data/my_pool --limit-entries 40

    # the run-to-run floor: several arms of the same patterns differing only in the search seed
    python -m mlindex.scripts.run_benchmark --stage floor \
        --arm benchmark=mlindex/data/floor/arm1 \
        --arm seed202=mlindex/data/floor/seed202 \
        --arm seed303=mlindex/data/floor/seed303 \
        --out-dir results/floor

By default the whole chain runs and nothing has to be sequenced by hand. `--stage` exists because a
cluster needs the halves as separate jobs with different walltimes.

A number from this tool is only comparable with another if both came from the same condition set
and the same process count, so both are recorded beside every result and a mismatch is refused
rather than reported.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import Benchmark
from mlindex.model_training import BenchmarkMetrics as metrics
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

DEFAULT_SCORES = ('M20', 'M_sym')

# Where each score comes from: a column of the candidate table, or of a named sidecar.
SCORE_SOURCES = {'M20': None, 'M_sym': 'merits', 'M_tilde': 'merits', 'M_rev': 'merits',
                 'X_N': 'merits', 'n_over': 'merits', 'max_gap': 'merits'}

CANDIDATE_COLUMNS = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id',
                     'in_top_n', 'is_correct']
ENTRY_COLUMNS = ['entry_id', 'condition_bundle', 'is_degenerate', 'bravais_lattice_true']


def build_parser():
    parser = argparse.ArgumentParser(
        description='Measure the indexer on a benchmark pool and compare ways of ranking it.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Scores not stored on the candidate table are read from a sidecar; a missing '
               'sidecar is an error, because a missing score column would rank last and read '
               'like a bad score.')
    parser.add_argument('--pool', action='append', default=None, metavar='PATH',
                        help='A consolidated pool directory. Repeat to reduce several.')
    parser.add_argument('--arm', action='append', default=None, metavar='NAME=PATH',
                        help='A named arm, for --stage floor. Repeat once per arm.')
    parser.add_argument('--out-dir', default=None, metavar='PATH',
                        help='Where to write the tables (default: print only).')
    parser.add_argument('--stage', default='all',
                        choices=('all', 'reduce', 'report', 'floor'),
                        help='Which half to run (default: all). Separate stages exist because a '
                             'cluster needs them as separate jobs.')
    parser.add_argument('--scores', default=','.join(DEFAULT_SCORES), metavar='A,B',
                        help=f'Comma-separated scores to rank by (default: '
                             f'{",".join(DEFAULT_SCORES)}).')
    parser.add_argument('--baseline', default='M20', metavar='NAME',
                        help='The score every other score is contrasted against (default: M20).')
    parser.add_argument('--bundles', default=None, metavar='A,B',
                        help='Comma-separated condition bundles (default: every one in the pool).')
    parser.add_argument('--bravais-lattices', default=None, metavar='A,B',
                        help=f'Comma-separated lattices (default: all {len(BRAVAIS_LATTICES)}).')
    parser.add_argument('--top-n', type=int, default=10, metavar='N',
                        help='How many printed rows count as a success (default: 10).')
    parser.add_argument('--depth', default='all', choices=('all', 'in_top_n'),
                        help="Which pool to rank in: 'all' is everything retained, 'in_top_n' is "
                             'the top twenty per lattice production keeps (default: all).')
    parser.add_argument('--limit-entries', type=int, default=None, metavar='N',
                        help='Cap the number of SOURCE CRYSTALS, chosen reproducibly, so every '
                             'arm of a comparison sees the same ones.')
    parser.add_argument('--entry-seed', type=int, default=12345, metavar='N',
                        help='Seed for --limit-entries (default: 12345).')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Read an arm with no completion stamp. Off by default: a killed run '
                             'looks finished by its contents.')
    return parser


def _split(value):
    return [item for item in (value or '').split(',') if item]


def _parse_arms(values):
    arms = []
    for item in values or []:
        if '=' not in item:
            raise ValueError(f'--arm wants NAME=PATH, got {item!r}')
        name, path = item.split('=', 1)
        arms.append((name, Path(path)))
    return arms


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


def reduce_arm(pool_dir, scores, bundles=None, bravais_lattices=None, limit_entries=None,
               entry_seed=12345, require_complete=True):
    """Reduce one pool to per-pattern outcomes, one row per (crystal, condition) per score."""
    pool_dir = Path(pool_dir)
    Benchmark.check_complete(pool_dir, require=require_complete)
    entries = Benchmark.load_entries(pool_dir, bundles=bundles, columns=ENTRY_COLUMNS)
    entries = select_entries(entries, limit_entries, entry_seed)
    wanted_bundles = bundles or Benchmark.available_bundles(pool_dir)

    sidecars = sorted({SCORE_SOURCES.get(name) for name in scores} - {None})
    sidecar_columns = {name: [score for score in scores if SCORE_SOURCES.get(score) == name]
                       for name in sidecars}
    direct = [score for score in scores if SCORE_SOURCES.get(score) is None]

    pieces = {name: [] for name in scores}
    for bundle in wanted_bundles:
        frame = Benchmark.load_candidates(
            pool_dir, bundle, columns=CANDIDATE_COLUMNS + direct,
            bravais_lattices=bravais_lattices, sidecars=sidecars,
            sidecar_columns=sidecar_columns)
        frame = frame.loc[frame['entry_id'].isin(set(entries['entry_id']))]
        if frame.empty:
            continue
        frame = Benchmark.attach_entry_columns(frame, entries, ['is_degenerate'])
        for name, reduced in metrics.reduce_many(
                frame, {name: name for name in scores}).items():
            pieces[name].append(reduced)
        del frame
    return ({name: pd.concat(parts, ignore_index=True) for name, parts in pieces.items()
             if parts}, entries)


def report_arm(reductions, entries, top_n=10, depth='all'):
    """Per-scope metric tables, one per score, with the aggregate row first."""
    tables = {}
    for name, per_entry in reductions.items():
        flagged = metrics.derive_flags(per_entry, depth=depth, top_n=top_n)
        table = metrics.summarise_by_scope(flagged, entries=entries)
        table.insert(0, 'score', name)
        tables[name] = table
    return tables


def contrast_table(reductions, entries, baseline, top_n=10, depth='all'):
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


def _write(out_dir, name, table):
    if out_dir is None:
        return None
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding='utf-8')
    print(f'wrote {path}')
    return path


def main(argv=None):
    args = build_parser().parse_args(argv)
    scores = _split(args.scores)
    bundles = _split(args.bundles) or None
    lattices = _split(args.bravais_lattices) or None
    unknown = [name for name in scores if name not in SCORE_SOURCES]
    if unknown:
        raise SystemExit(f'Unknown score(s) {unknown}. Known: {sorted(SCORE_SOURCES)}')

    if args.stage == 'floor':
        arms = _parse_arms(args.arm)
        if len(arms) < 2:
            raise SystemExit('--stage floor needs at least two --arm NAME=PATH values.')
        arm_reductions = {}
        for name, path in arms:
            reductions, _ = reduce_arm(path, scores, bundles=bundles, bravais_lattices=lattices,
                                       limit_entries=args.limit_entries,
                                       entry_seed=args.entry_seed,
                                       require_complete=not args.allow_incomplete)
            arm_reductions[name] = reductions
            print(f'reduced arm {name}')
        rows = [floor_from_arms(arm_reductions, score, args.baseline, top_n=args.top_n,
                                depth=args.depth)
                for score in scores if score != args.baseline]
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        _write(args.out_dir, 'floor.csv', table)
        return 0

    pools = args.pool or []
    if not pools:
        raise SystemExit('--pool is required (or use --stage floor with --arm).')
    for pool in pools:
        reductions, entries = reduce_arm(
            pool, scores, bundles=bundles, bravais_lattices=lattices,
            limit_entries=args.limit_entries, entry_seed=args.entry_seed,
            require_complete=not args.allow_incomplete)
        tag = Path(pool).name
        if args.stage in ('all', 'reduce'):
            for name, per_entry in reductions.items():
                _write(args.out_dir, f'{tag}_per_entry_{name}.csv', per_entry)
        if args.stage in ('all', 'report'):
            tables = report_arm(reductions, entries, top_n=args.top_n, depth=args.depth)
            combined = pd.concat(tables.values(), ignore_index=True)
            print(f'\n=== {tag}, depth={args.depth}, top_n={args.top_n} ===')
            print(combined.loc[combined['scope'] == 'aggregate'].to_string(index=False))
            _write(args.out_dir, f'{tag}_metrics.csv', combined)
            contrast = contrast_table(reductions, entries, args.baseline, top_n=args.top_n,
                                      depth=args.depth)
            if not contrast.empty:
                print(f'\n--- against {args.baseline} ---')
                print(contrast.to_string(index=False))
                _write(args.out_dir, f'{tag}_contrast.csv', contrast)
    return 0


if __name__ == '__main__':
    sys.exit(main())
