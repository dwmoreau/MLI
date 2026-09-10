"""Measure the indexer on a benchmark pool, and compare two ways of ranking it.

This is the instrument for "did that change help?". It reads a stored pool of candidate unit
cells, ranks them by one or more scores, reduces the ranking to one outcome per pattern-condition
-- one crystal under one noise condition -- and reports how often the correct cell reaches the top
of the list. Two arms are compared paired, against the run-to-run noise of the search itself.

    # generate an arm, score it, reduce it and report it -- the whole chain, one command
    python -m mlindex.scripts.run_benchmark --stage all --out-pool arms/baseline \
        --split-manifest path/to/split_manifest.parquet \
        --population general --per-lattice 40 --cut 1.5 \
        --seed 12345 --search-seed 12345 --n-pools 4 --out-dir results/baseline

    # every score a stored pool can carry, aggregate and per lattice, written to a directory
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

By default the whole chain runs and nothing has to be sequenced by hand: generate, score, reduce,
report. `--stage` exists because a cluster needs the parts as separate jobs with different
walltimes -- generation is node-hours, everything after it is minutes.

Two seeds, doing different jobs. `--seed` fixes which crystals are drawn and what noise is put on
their peaks, so every arm of a comparison must share it. `--search-seed` reaches the search alone,
and is the one a run-to-run floor moves.

A number from this tool is only comparable with another if both came from the same condition set
and the same process count, so both are recorded beside every result and a mismatch is refused
rather than reported.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mlindex.model_training import Benchmark
from mlindex.model_training import BenchmarkMetrics as metrics
from mlindex.model_training import BenchmarkRuns as runs
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
                        choices=('all', 'generate', 'sidecars', 'reduce', 'report', 'floor'),
                        help='Which part to run (default: all, which is the whole chain). The '
                             'stages exist because a cluster needs them as separate jobs with '
                             'different walltimes.')
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

    generate = parser.add_argument_group(
        'generating an arm',
        'Indexing the benchmark patterns and keeping every candidate. An arm is written to '
        '--out-pool and stamped complete only when every pool has landed.')
    generate.add_argument('--out-pool', default=None, metavar='PATH',
                          help='Where to write the generated pool.')
    generate.add_argument('--split-manifest', default=None, metavar='PATH',
                          help='The frozen train/dev/test split to draw crystals from. Its '
                               'sha256 is recorded in the manifest.')
    generate.add_argument('--population', default='general',
                          choices=tuple(runs.POPULATIONS),
                          help="Which population to generate (default: general). 'hard' is the "
                               'severe end of every condition axis over the three lattices where '
                               'the correct cell is hardest to find.')
    generate.add_argument('--per-lattice', type=int, default=40, metavar='N',
                          help='Source crystals per Bravais lattice (default: 40). Balanced '
                               'rather than proportional, because results are an unweighted mean '
                               'over lattices. A lattice with fewer contributes all it has.')
    generate.add_argument('--cut', type=float, default=1.5, metavar='M20',
                          help='The M20 threshold candidates are pruned at before refinement '
                               '(default: 1.5). Generate low and retain everything; a higher '
                               'threshold is then a restriction rather than a second run.')
    generate.add_argument('--seed', type=int, default=12345, metavar='N',
                          help='Fixes which crystals are drawn and what noise goes on their '
                               'peaks (default: 12345). Every arm of a comparison shares it, or '
                               'the arms differ in their data rather than in the code.')
    generate.add_argument('--search-seed', type=int, default=12345, metavar='N',
                          help='Reaches the search alone (default: 12345). This is the one that '
                               'moves between the arms of a run-to-run floor.')
    generate.add_argument('--n-pools', type=int, default=1, metavar='N',
                          help='Independent pools of processes over the crystals (default: 1). '
                               'Each holds its own copy of the models, so this is the '
                               'memory-limited axis, not the core count.')
    generate.add_argument('--pool-size', type=int, default=1, metavar='N',
                          help='Processes within one pool (default: 1). Above one, a lattice is '
                               'split across them and the answer depends on the count, so this '
                               'is part of an arm identity and arms are paired only at equal '
                               'values.')
    generate.add_argument('--dataset-directory', default=None, metavar='PATH',
                          help='Where the per-lattice source datasets live (default: the '
                               'packaged mlindex/data/generated_datasets).')
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


def reduce_arm(pool_dir, scores, bundles=None, bravais_lattices=None, limit_entries=None,
               entry_seed=12345, require_complete=True):
    """Reduce one pool to per-pattern outcomes, one row per (crystal, condition) per score."""
    pool_dir = Path(pool_dir)
    Benchmark.check_complete(pool_dir, require=require_complete)
    entries = Benchmark.load_entries(pool_dir, bundles=bundles, columns=ENTRY_COLUMNS)
    entries = runs.select_entries(entries, limit_entries, entry_seed)
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

    if args.stage in ('all', 'generate') and args.out_pool:
        if not args.split_manifest:
            raise SystemExit('Generating an arm needs --split-manifest: the crystals are read '
                             'from the frozen split, never re-drawn by sampling.')
        metadata = runs.run_arm(
            args.out_pool, args.split_manifest, population=args.population,
            per_lattice=args.per_lattice, seed=args.seed, search_seed=args.search_seed,
            cut=args.cut, pool_size=args.pool_size, n_pools=args.n_pools, bundles=bundles,
            dataset_directory=args.dataset_directory)
        print(f"generated {metadata['n_source_entries']} crystals x "
              f"{len(metadata['bundles'])} bundles into {args.out_pool}")
        pools = [args.out_pool]
    elif args.stage == 'generate':
        raise SystemExit('--stage generate needs --out-pool.')
    else:
        pools = list(args.pool or [])

    if args.stage in ('all', 'sidecars') and pools:
        for pool in pools:
            written = runs.merit_sidecar(pool, bundles=bundles, bravais_lattices=lattices)
            print(f'scored {len(written)} shards of {pool}')
        if args.stage == 'sidecars':
            return 0
    elif args.stage == 'sidecars':
        raise SystemExit('--stage sidecars needs --pool.')

    if args.stage == 'generate':
        return 0

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
        rows = [runs.floor_from_arms(arm_reductions, score, args.baseline,
                                     top_n=args.top_n, depth=args.depth)
                for score in scores if score != args.baseline]
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        _write(args.out_dir, 'floor.csv', table)
        return 0

    if not pools:
        raise SystemExit('--pool is required (or --stage generate with --out-pool, or '
                         '--stage floor with --arm).')
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
            tables = runs.report_arm(reductions, entries, top_n=args.top_n,
                                     depth=args.depth)
            combined = pd.concat(tables.values(), ignore_index=True)
            print(f'\n=== {tag}, depth={args.depth}, top_n={args.top_n} ===')
            print(combined.loc[combined['scope'] == 'aggregate'].to_string(index=False))
            _write(args.out_dir, f'{tag}_metrics.csv', combined)
            contrast = runs.contrast_table(reductions, args.baseline,
                                           top_n=args.top_n, depth=args.depth)
            if not contrast.empty:
                print(f'\n--- against {args.baseline} ---')
                print(contrast.to_string(index=False))
                _write(args.out_dir, f'{tag}_contrast.csv', contrast)
    return 0


if __name__ == '__main__':
    sys.exit(main())
