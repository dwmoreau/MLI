"""The benchmark runs that decide P09c's candidate settings, and one command for each step.

RESEARCH CODE THAT NEEDS TO BE DELETED -- P09c. Once P09c's verdicts have landed, the runs below
describe nothing a later session runs. Delete this file and submit_ensemble_arms.sh in the commit
that removes the `fractions` setting.

Every run is a `run_benchmark` command; this file only names them, so that nobody has to assemble
one by hand. `list` prints each command in full, and any of them can be run on its own.

    # every run, numbered, with the run_benchmark command it stands for
    python -m mlindex.scripts.ensemble_arms list

    # on NERSC: generate, score and reduce one run (submit_ensemble_arms.sh runs them all)
    python -m mlindex.scripts.ensemble_arms generate --index 0 \
        --pools-dir $SCRATCH/p09c_pools \
        --tables-dir $SCRATCH/fom_production/artifacts/P09c_arms/tables \
        --split-manifest $MLI_SPLIT_MANIFEST --n-pools 128

    # on the laptop, after `docs/sync_record.sh pull-artifacts P09c_arms`
    python -m mlindex.scripts.ensemble_arms compare \
        --tables-dir docs/fom_production/artifacts/P09c_arms/tables \
        --floor-dir docs/fom_production/artifacts/P04b_arms \
        --out-dir docs/fom_production/artifacts/P09c_arms/compare

`control` is the code as it stands, with P09b's generator fractions in ENSEMBLE. The fractions
are compared with the old ones, and each budget run with `control`.

The populations, crystals per lattice and condition bundles are the ones the floor was measured
on. Any other choice and the floor no longer describes the runs' noise.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mlindex.scripts import run_benchmark
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM

# The generator fractions shipped before P09c. ENSEMBLE now holds P09b's; the `old_fractions` run
# names these, so every run comes from one commit. Trees, abnn, templates.
OLD_FRACTIONS = {lattice: (0.45, 0.45, 0.10) for lattice in ('cF', 'cI', 'cP')}
OLD_FRACTIONS.update({lattice: (0.05, 0.70, 0.25)
                      for lattice in ('hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP')})
OLD_FRACTIONS.update({'mC': (0.05, 0.55, 0.40), 'mP': (0.05, 0.55, 0.40),
                      'aP': (0.05, 0.40, 0.55)})
_OLD_FRACTIONS = [f'--fractions={lattice}={t},{a},{p}'
                  for lattice, (t, a, p) in OLD_FRACTIONS.items()]

FAMILIES = ('cubic', 'hexagonal', 'rhombohedral', 'tetragonal', 'orthorhombic', 'monoclinic',
            'triclinic')

# Each run's run_benchmark flags beyond the common ones. `control` is ENSEMBLE as it stands.
RUNS = {
    'control': [],
    'old_fractions': _OLD_FRACTIONS,
    }
for _family in FAMILIES:
    RUNS[f'budget_half_{_family}'] = [f'--budget-scale={_family}=0.5']
    RUNS[f'budget_double_{_family}'] = [f'--budget-scale={_family}=2']

# What the floor was measured on (submit_benchmark_arms.sh). The hard population takes the
# driver's default bundles, every severe one, and covers aP, mP and mC only.
POPULATIONS = {
    'general': {'per_lattice': 40,
                'flags': ['--bundles', 'b1_error0.5_cont0,b1_error1_cont0,b1_error2_cont0']},
    'hard': {'per_lattice': 120, 'flags': [], 'bravais_lattices': ('aP', 'mP', 'mC')},
    }
SCORES = 'M_sym,M20'   # M_sym first: the contrast reads its floor from the first score
SEED = 12345
CUT = 1.5

# The comparisons `compare` makes: (name, reference run, the runs read against it). The fractions
# are read with the old ones as reference, so a verdict of helps means P09b's are better; the
# budget runs against `control`, so helps means that change would improve on what ENSEMBLE ships.
QUESTIONS = (
    ('fractions', 'old_fractions', ('control',)),
    ('against_control', 'control', tuple(name for name in RUNS if name.startswith('budget_'))),
    )

def _touches(run, lattices):
    """Whether a run changes anything on these lattices, so is worth running on them."""
    if not run.startswith('budget_'):
        return True
    family = run.split('_', 2)[2]
    return any(BL_TO_LATTICE_SYSTEM[lattice] == family for lattice in lattices)


def jobs():
    """Every (population, run) to generate, in array-index order."""
    out = []
    for population, design in POPULATIONS.items():
        lattices = design.get('bravais_lattices', tuple(BL_TO_LATTICE_SYSTEM))
        out += [(population, run) for run in RUNS if _touches(run, lattices)]
    return out


def generate_argv(population, run, pools_dir, tables_dir, split_manifest, n_pools,
                  dataset_directory=None):
    """The run_benchmark command line for one run: generate, score its merits, reduce it.

    The pool, every candidate, goes under `pools_dir` and stays where it was made; the reduced
    per-entry tables, which are all a comparison reads, go to `tables_dir`.
    """
    design = POPULATIONS[population]
    argv = ['--stage', 'all',
            '--out-pool', str(Path(pools_dir) / f'{population}_{run}'),
            '--out-dir', str(tables_dir),
            '--split-manifest', str(split_manifest),
            '--population', population,
            '--per-lattice', str(design['per_lattice']),
            *design['flags'],
            '--cut', str(CUT), '--seed', str(SEED), '--search-seed', str(SEED),
            '--n-pools', str(n_pools), '--pool-size', '1',
            '--scores', SCORES,
            *RUNS[run]]
    if dataset_directory:
        argv += ['--dataset-directory', str(dataset_directory)]
    return argv


def compare(tables_dir, floor_dir, out_dir):
    """Every question, both populations, one table; prints which lattices each run moves."""
    tables_dir, out_dir = Path(tables_dir), Path(out_dir)
    pieces = []
    for population in POPULATIONS:
        floor_table = Path(floor_dir) / f'{population}_floor' / 'floor.csv'
        if not floor_table.is_file():
            raise SystemExit(f'No floor table at {floor_table}.')
        for question, reference, arms in QUESTIONS:
            present = [run for run in (reference, *arms)
                       if (tables_dir / f'{population}_{run}_manifest.json').is_file()]
            if reference not in present or len(present) < 2:
                print(f'skipping {population} {question}: have {present}')
                continue
            directory = out_dir / f'{population}_{question}'
            run_benchmark.main(
                ['--stage', 'contrast', '--reference', reference, '--vary', 'ensemble',
                 '--scores', SCORES, '--floor-table', str(floor_table), '--out-dir', str(directory)]
                + [f'--arm={run}={tables_dir / f"{population}_{run}"}' for run in present])
            table = pd.read_csv(directory / 'arm_contrast.csv')
            table.insert(0, 'question', question)
            table.insert(0, 'population', population)
            pieces.append(table)
    if not pieces:
        raise SystemExit(f'Nothing to compare in {tables_dir}.')
    summary = pd.concat(pieces, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'summary.csv', index=False, encoding='utf-8')

    read = summary.loc[(summary['scope'] != 'aggregate') & (summary['verdict'] != '')
                       & summary['verdict'].notna()]
    print('\nPer-lattice verdicts, top-10, against the reference run:')
    for (population, arm, score), block in read.groupby(['population', 'arm', 'score'], sort=False):
        helps = ' '.join(block.loc[block['verdict'] == 'helps', 'scope'])
        hurts = ' '.join(block.loc[block['verdict'] == 'hurts', 'scope'])
        print(f'  {population:7s} {arm:28s} {score:5s}  helps: {helps or "-":20s} '
              f'hurts: {hurts or "-"}')
    print(f'\nwrote {out_dir / "summary.csv"}')
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    commands.add_parser('list', help='Print every run with its index and command.')

    generate = commands.add_parser('generate', help='Generate, score and reduce one run.')
    which = generate.add_mutually_exclusive_group(required=True)
    which.add_argument('--index', type=int, metavar='N',
                       help='The run with this index in `list` (what the SLURM array passes).')
    which.add_argument('--run', metavar='POPULATION/NAME',
                       help='The run by name, for example general/budget_half_cubic.')
    generate.add_argument('--pools-dir', required=True, metavar='PATH',
                          help='Where the pool, every candidate, is written. Tens of GB a run.')
    generate.add_argument('--tables-dir', required=True, metavar='PATH',
                          help='Where the reduced per-entry tables go: a few MB, what compare reads.')
    generate.add_argument('--split-manifest', required=True, metavar='PATH',
                          help='The frozen train/dev/test split the floor was measured on.')
    generate.add_argument('--n-pools', type=int, default=1, metavar='N',
                          help='Independent processes over the crystals (default: 1).')
    generate.add_argument('--dataset-directory', default=None, metavar='PATH',
                          help='Where the per-lattice source datasets live (default: packaged).')
    generate.add_argument('--limit', type=int, default=None, metavar='N',
                          help='Crystals per lattice instead of the population default, for a '
                               'quick check only: a run made this way is not on the floor.')

    comparing = commands.add_parser('compare', help='Every comparison, one summary table.')
    comparing.add_argument('--tables-dir', required=True, metavar='PATH',
                           help='The reduced tables, as generate wrote them.')
    comparing.add_argument('--floor-dir', required=True, metavar='PATH',
                           help='Holds general_floor/floor.csv and hard_floor/floor.csv.')
    comparing.add_argument('--out-dir', required=True, metavar='PATH',
                           help='Where the per-question tables and summary.csv go.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'list':
        for index, (population, run) in enumerate(jobs()):
            command = generate_argv(population, run, '$POOLS_DIR', '$TABLES_DIR',
                                    '$MLI_SPLIT_MANIFEST', '$N_POOLS')
            print(f'{index:3d}  {population}/{run}\n'
                  f'     python -m mlindex.scripts.run_benchmark {" ".join(command)}')
        return 0
    if args.command == 'generate':
        listed = jobs()
        if args.index is not None:
            if not 0 <= args.index < len(listed):
                raise SystemExit(f'--index must be 0 to {len(listed) - 1}; see `list`.')
            population, run = listed[args.index]
        else:
            population, _, run = args.run.partition('/')
            if (population, run) not in listed:
                raise SystemExit(f'No run {args.run!r}; see `list`.')
        argv = generate_argv(population, run, args.pools_dir, args.tables_dir,
                             args.split_manifest, args.n_pools, args.dataset_directory)
        if args.limit is not None:
            argv[argv.index('--per-lattice') + 1] = str(args.limit)
        print(f'{population}/{run}: run_benchmark {" ".join(argv)}', flush=True)
        return run_benchmark.main(argv)
    compare(args.tables_dir, args.floor_dir, args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
