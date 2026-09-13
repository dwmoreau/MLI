"""Which inputs should the Miller-index template ranker read? Measured by the yield of its templates.

The templater turns a peak list into thousands of candidate cells, and a gradient-boosted ranker
keeps the few that go on to the search. This script trains that ranker on different input sets and
measures, for each, how often the templates it keeps lead to the true cell.

Run the stages in order.

    # 1. Train one ranker per input set on the `fit` crystals. The real training target is a
    #    success curve that exists only on NERSC, so a stand-in is used: -log10 of a template's
    #    distance to the true cell. Everything else is the templater's own training code.
    python -m mlindex.scripts.run_template_yield --stage train \
        --split-manifest docs/fom_campaign2/artifacts/S06_split_manifest.parquet \
        --models-dir mlindex/models --work-dir template_yield \
        --input-sets rho,posterior_sigma --bravais-lattices tP --per-lattice 50

    # 2. Measure the shipped ranker and every trained one on crystals none was trained on.
    python -m mlindex.scripts.run_template_yield --stage yield \
        --split-manifest docs/fom_campaign2/artifacts/S06_split_manifest.parquet \
        --models-dir mlindex/models --work-dir template_yield \
        --input-sets rho,posterior_sigma --bravais-lattices tP --per-lattice 50 \
        --group select --bundles b1_error1_cont0 --out-dir template_yield/select

Crystals come from the frozen split in three disjoint groups. `fit` and `select` divide the
`fom-train` crystals drawn for a lattice four to one by a hash of each crystal's identifier, so a
crystal's group does not depend on how many were drawn; `report` is `fom-dev`. Rankers are trained
on `fit`, compared on `select`, and reported once on `report`. `fom-test` is never read.

For each (crystal, condition bundle, ranker) the yield stage records two things:

  * Unrefined yield. The ranker orders every template; production keeps the first `n`, where `n`
    is the lattice's template count in `UtilitiesOptimizer`. The rank of the first template that
    is the true cell, at each --rtol, gives the yield at `n` and at any other depth, and whether any
    template at all is correct gives the ceiling.
  * Refined yield. The indexer's own search is run from the ranker's templates alone, and the
    pattern counts as found when any refined cell is the true cell. The choice between input sets
    is made on this number.

Only the pattern's true lattice is searched, since that is the templater the pattern tests.
"""

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlindex.model_training import Benchmark
from mlindex.model_training import BenchmarkConditions
from mlindex.model_training import BenchmarkPatterns
from mlindex.model_training import BenchmarkRuns as runs
from mlindex.model_training.MITemplates import TEMPLATE_INPUT_SETS
from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

GROUPS = ('fit', 'select', 'report')
# One crystal in this many of a lattice's fom-train draw is held out to compare rankers on.
SELECT_EVERY = 5
DEFAULT_RTOLS = (0.001, 0.003, 0.01, 0.03)
SHIPPED = 'shipped'
# The stand-in success curve: -log10(distance) tabulated on a log grid up to the lattice's
# max_distance, which is what calibrate_templates reads through `roc_file_name`.
PROXY_CURVE_POINTS = 400
PROXY_CURVE_SMALLEST_DISTANCE = 1e-7
# The shipped hyperparameters a trained ranker copies, and how each is read from the params CSV.
HYPERPARAMETERS = {
    'loss': str,
    'learning_rate': float,
    'max_depth': int,
    'max_leaf_nodes': int,
    'min_samples_leaf': int,
    'l2_regularization': float,
    'n_instances_train': int,
    'max_distance': float,
    'n_peaks_template': int,
    'n_peaks_calibration': int,
    'q2_error_multiplier_low': float,
    'q2_error_multiplier_high': float,
    'n_contaminants_max': int,
    }


def crystal_group(identifier, seed):
    """'select' for one fom-train crystal in SELECT_EVERY and 'fit' for the rest, fixed per crystal."""
    key = BenchmarkPatterns.derived_seed(f'group:{identifier}', seed)
    return 'select' if key % SELECT_EVERY == 0 else 'fit'


def draw_group(split_manifest, group, per_lattice, seed, bravais_lattices):
    """The crystals of one group: `per_lattice` drawn per lattice from the split, then divided."""
    if group not in GROUPS:
        raise ValueError(f'Unknown group {group!r}; known: {GROUPS}')
    if group == 'report':
        return runs.draw_entries(split_manifest, per_lattice, seed, split='fom-dev',
                                 bravais_lattices=bravais_lattices)
    chosen = runs.draw_entries(split_manifest, per_lattice, seed, split='fom-train',
                               bravais_lattices=bravais_lattices)
    groups = chosen['identifier'].map(lambda identifier: crystal_group(identifier, seed))
    return chosen.loc[groups == group].reset_index(drop=True)


class _Settings:
    """Takes an optimizer's place in a factory, so the factory returns its settings unloaded."""

    def __init__(self, data_params, opt_params, *args, **kwargs):
        self.opt_params = opt_params


def production_template_count(bravais_lattice, models_dir):
    """How many templates the shipped indexer asks this lattice's templater for."""
    from mlindex.optimization.UtilitiesOptimizer import get_optimizers

    organizers = {bravais_lattice: SimpleNamespace(manager=0, workers=[0], split_comm=None,
                                                   color=None)}
    settings = get_optimizers(0, organizers, BenchmarkPatterns.BROADENING_TAG, 1,
                              optimizer_class=_Settings, models_directory=models_dir)
    counts = [generator['n_unit_cells']
              for generator in settings[bravais_lattice].opt_params['generator_info']
              if generator['generator'] == 'templates']
    if len(counts) != 1:
        raise ValueError(f'{bravais_lattice} has {len(counts)} template generators; expected one.')
    return int(counts[0])


def ranked_templates(templator, q2_obs, seed):
    """Every template cell for a peak list, and the same cells in the order the ranker keeps them.

    Both calls draw from a generator seeded identically, so they produce the same cells. The
    second asks for all of them, so it returns the ranker's whole ordering, and its first `n` rows
    are exactly what `generate(n, ...)` returns.
    """
    cells = templator.generate('all', np.random.default_rng(seed), q2_obs)
    ranked = templator.generate(cells.shape[0], np.random.default_rng(seed), q2_obs)
    return cells, ranked


def first_correct_rank(truth, cells, lattice_system, rtol):
    """The position of the first cell that is the true cell, or -1 if none is."""
    correct = is_correct_known_bl_batch(truth, cells, lattice_system, rtol=rtol)
    return int(np.argmax(correct)) if correct.any() else -1


def _copy_once(source, destination):
    """Copy a file unless it is already there, atomically, so parallel jobs can share a tree."""
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f'.{destination.name}.{os.getpid()}.partial')
    shutil.copyfile(source, partial)
    os.replace(partial, destination)


def arm_models_dir(work_dir, input_set):
    return Path(work_dir) / 'models' / input_set


def _training_frame(rows, train):
    tag = BenchmarkPatterns.BROADENING_TAG
    q2 = [np.asarray(peaks, dtype=np.float64)[np.asarray(peaks) > 0][:BenchmarkPatterns.N_PEAKS]
          for peaks in rows[f'q2_{tag}']]
    return pd.DataFrame({'q2': q2, 'reindexed_xnn': list(rows['reindexed_xnn']),
                         'train': train, 'augmented': False})


def train_input_set(input_set, bravais_lattice, models_dir, work_dir, fit_rows, select_rows, seed):
    """Fit one lattice's ranker on `input_set` with the templater's own training code.

    The shipped template library is copied under the shipped tag into this input set's models
    directory, so every input set ranks the same templates and the directory loads exactly as the
    shipped one does. The `select` crystals are the validation rows calibrate_templates plots.
    """
    from mlindex.model_training.MITemplates import MITemplates
    from mlindex.model_training.Wrapper import Wrapper
    from mlindex.utilities.IOManagers import read_params

    system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    tag = f'{system}_{BenchmarkPatterns.BROADENING_TAG}'
    shipped = Path(models_dir) / tag
    arm = arm_models_dir(work_dir, input_set) / tag
    for path in (shipped / 'data').iterdir():
        if path.is_file():
            _copy_once(path, arm / 'data' / path.name)
    for kind in ('miller_index_templates', 'miller_index_templates_prob'):
        name = f'{bravais_lattice}_{kind}_{tag}.npy'
        _copy_once(shipped / 'template' / name, arm / 'template' / name)

    shipped_params = read_params(
        shipped / 'template' / f'{bravais_lattice}_template_params_{tag}.csv')
    params = {name: cast(shipped_params[name]) for name, cast in HYPERPARAMETERS.items()
              if shipped_params.get(name) not in (None, '')}
    distances = np.logspace(np.log10(PROXY_CURVE_SMALLEST_DISTANCE),
                            np.log10(params.get('max_distance', 0.05)), PROXY_CURVE_POINTS)
    curve_path = arm / 'template' / f'{bravais_lattice}_proxy_success_curve.npy'
    np.save(curve_path, np.stack((distances, -np.log10(distances))))

    data = pd.concat((_training_frame(fit_rows, True), _training_frame(select_rows, False)),
                     ignore_index=True)
    params.update({
        'tag': tag,
        'load_templates': True,
        'template_inputs': input_set,
        'roc_file_name': str(curve_path),
        'parallelization': None,
        'n_entries_train': int(fit_rows.shape[0]),
        })
    wrapper = Wrapper(
        data_params={'tag': tag, 'models_directory': str(models_dir), 'load_from_tag': True},
        rf_params={}, template_params={}, abnn_params={}, random_params={}, seed=seed,
        load_bravais_lattice=bravais_lattice)
    trainer = MITemplates(bravais_lattice, wrapper.data_params, params,
                          wrapper.hkl_ref[bravais_lattice], str(arm / 'template'), seed)
    started = time.perf_counter()
    trainer.setup(data)
    return {'input_set': input_set, 'bravais_lattice': bravais_lattice,
            'n_fit': int(fit_rows.shape[0]), 'n_select': int(select_rows.shape[0]),
            'seconds': time.perf_counter() - started}


def measure_lattice(arm, arm_dir, bravais_lattice, rows, second_phase_pool, bundles, seed,
                    search_seed, cut, rtols, n_templates):
    """One ranker's unrefined and refined yield over one lattice's crystals, one row per pattern."""
    from mlindex.model_training.BenchmarkOptimizer import BenchmarkOptimizer
    from mlindex.optimization.MPOptimizer import _build_group_optimizers
    from mlindex.utilities.Digests import q2_digest
    from mlindex.utilities.UnitCellTools import get_partial_unit_cell

    lattice_system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    options = {'generator_info': [{'generator': 'templates', 'n_unit_cells': int(n_templates)}],
               'prune_m20_threshold': float(cut)}
    optimizer = _build_group_optimizers(
        [bravais_lattice], 1, [Queue()], [Queue()], BenchmarkPatterns.BROADENING_TAG, 1,
        search_seed, options, optimizer_class=BenchmarkOptimizer,
        models_directory=arm_dir)[bravais_lattice]
    templator = optimizer.wrapper.miller_index_templator[bravais_lattice]

    results = []
    for bundle in bundles:
        condition = BenchmarkConditions.BY_TAG[bundle]
        for _, entry in rows.iterrows():
            row = {'arm': arm, 'entry_id': entry['identifier'], 'condition_bundle': bundle,
                   'bravais_lattice': bravais_lattice, 'n_templates': int(n_templates)}
            try:
                pattern = BenchmarkPatterns.prepare_peak_list(
                    entry, condition, seed, second_phase_pool=second_phase_pool)
            except ContaminantPlacementError as error:
                # Recorded rather than skipped silently: every arm refuses the same patterns,
                # because the peak list depends on the seed and the crystal alone.
                results.append(dict(row, refused=str(error)))
                continue
            q2 = np.asarray(pattern.q2_obs, dtype=np.float64)
            unit_cell_true = np.asarray(entry['reindexed_unit_cell'], dtype=np.float64)
            truth = get_partial_unit_cell(unit_cell_true, lattice_system=lattice_system)

            started = time.perf_counter()
            template_seed = BenchmarkPatterns.derived_seed(
                f'templates:{entry["identifier"]}:{bundle}:{bravais_lattice}', search_seed)
            cells, ranked = ranked_templates(templator, q2[:optimizer.n_peaks], template_seed)
            row['seconds_templates'] = time.perf_counter() - started
            row['n_cells'] = int(cells.shape[0])
            row['n_ranked'] = int(ranked.shape[0])
            for rtol in rtols:
                row[f'first_rank_{rtol:g}'] = first_correct_rank(truth, ranked, lattice_system, rtol)
                row[f'oracle_{rtol:g}'] = bool(
                    is_correct_known_bl_batch(truth, cells, lattice_system, rtol=rtol).any())

            started = time.perf_counter()
            optimizer.dump_context = {'entry_id': entry['identifier'], 'condition_bundle': bundle,
                                      'q2_digest': q2_digest(q2)}
            optimizer.run(q2=q2, zero_error=False, wavelength=None,
                          n_top_candidates=Benchmark.N_TOP_CANDIDATES)
            refined = Benchmark.label_frame(
                Benchmark.records_to_frame(optimizer.drain()),
                pd.DataFrame({'entry_id': [entry['identifier']], 'unit_cell_true': [unit_cell_true]}))
            row['seconds_refine'] = time.perf_counter() - started
            row['n_refined'] = int(refined.shape[0])
            row['refined_found'] = bool(refined['is_correct'].any())
            row['refined_in_top_n'] = bool((refined['is_correct'] & refined['in_top_n']).any())
            row['refused'] = ''
            results.append(row)
    return pd.DataFrame(results)


def _run_jobs(function, jobs, processes):
    """Run `function(*job)` for every job, in spawned processes when more than one is allowed."""
    if processes <= 1:
        return [function(*job) for job in jobs]
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=processes, mp_context=context) as executor:
        futures = [executor.submit(function, *job) for job in jobs]
        return [future.result() for future in futures]


def write_manifest(path, **fields):
    fields.update({
        'commit': runs._commit(),
        'arch': platform.machine(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(fields, handle, indent=2, sort_keys=True, default=str)
    return path


def run_train(args):
    lattices = args.bravais_lattices
    chosen = runs.draw_entries(args.split_manifest, args.per_lattice, args.seed, split='fom-train',
                               bravais_lattices=lattices)
    rows = runs.load_source_rows(chosen, args.dataset_directory)
    groups = rows['identifier'].map(lambda identifier: crystal_group(identifier, args.seed))
    jobs = []
    for input_set in args.input_sets:
        for lattice in lattices:
            of_lattice = rows['bravais_lattice'] == lattice
            jobs.append((input_set, lattice, args.models_dir, args.work_dir,
                         rows.loc[of_lattice & (groups == 'fit')].reset_index(drop=True),
                         rows.loc[of_lattice & (groups == 'select')].reset_index(drop=True),
                         args.seed))
    summary = pd.DataFrame(_run_jobs(train_input_set, jobs, args.processes))
    print(summary.to_string(index=False))
    for input_set in args.input_sets:
        write_manifest(arm_models_dir(args.work_dir, input_set) / 'training_manifest.json',
                       input_set=input_set, families=TEMPLATE_INPUT_SETS[input_set],
                       bravais_lattices=lattices, per_lattice=args.per_lattice, seed=args.seed,
                       select_every=SELECT_EVERY, target='-log10(distance to the true cell)',
                       split_manifest=str(args.split_manifest),
                       split_manifest_sha256=runs.file_digest(args.split_manifest),
                       shipped_models_dir=str(args.models_dir),
                       training=summary.loc[summary['input_set'] == input_set].to_dict('records'))


def run_yield(args):
    lattices = args.bravais_lattices
    bundles = args.bundles or list(runs.POPULATIONS[args.population]['bundles'])
    chosen = draw_group(args.split_manifest, args.group, args.per_lattice, args.seed, lattices)
    rows = runs.load_source_rows(chosen, args.dataset_directory)
    second_phase_pool = BenchmarkPatterns.build_second_phase_pool(rows)
    arms = [(SHIPPED, Path(args.models_dir))] + [
        (input_set, arm_models_dir(args.work_dir, input_set)) for input_set in args.input_sets]
    counts = {lattice: production_template_count(lattice, args.models_dir) for lattice in lattices}

    jobs = []
    for arm, arm_dir in arms:
        for lattice in lattices:
            system_tag = f'{BL_TO_LATTICE_SYSTEM[lattice]}_{BenchmarkPatterns.BROADENING_TAG}'
            regressor = (arm_dir / system_tag / 'template'
                         / f'{lattice}_template_regressor_{system_tag}.onnx')
            if not regressor.is_file():
                raise SystemExit(f'No {lattice} ranker for arm {arm!r} at {regressor}. '
                                 'Run --stage train for it first.')
            jobs.append((arm, arm_dir, lattice,
                         rows.loc[rows['bravais_lattice'] == lattice].reset_index(drop=True),
                         second_phase_pool, bundles, args.seed, args.search_seed, args.cut,
                         args.rtol, counts[lattice]))
    started = time.perf_counter()
    per_entry = pd.concat(_run_jobs(measure_lattice, jobs, args.processes), ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_entry.to_parquet(out_dir / 'per_entry.parquet', index=False)
    write_manifest(out_dir / 'manifest.json', stage='yield', group=args.group,
                   population=args.population, bundles=bundles, bravais_lattices=lattices,
                   per_lattice=args.per_lattice, seed=args.seed, search_seed=args.search_seed,
                   cut=args.cut, rtols=list(args.rtol), template_counts=counts,
                   arms={arm: str(arm_dir) for arm, arm_dir in arms},
                   condition_set_digest=BenchmarkConditions.condition_set_digest(),
                   split_manifest=str(args.split_manifest),
                   split_manifest_sha256=runs.file_digest(args.split_manifest),
                   seconds=time.perf_counter() - started)
    print(f'wrote {out_dir / "per_entry.parquet"}: {per_entry.shape[0]} rows')


def _split(value):
    return [item for item in (value or '').split(',') if item]


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train the Miller-index template ranker on different input sets and measure '
                    'how often the templates it keeps lead to the true cell.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=('train', 'yield'),
                        help='train: fit one ranker per input set. yield: measure the shipped '
                             'ranker and the trained ones.')
    parser.add_argument('--split-manifest', required=True, metavar='PATH',
                        help='The frozen train/dev/test split crystals are drawn from.')
    parser.add_argument('--models-dir', required=True, metavar='PATH',
                        help='The shipped model tree: the directory that directly contains '
                             'cubic_1/, hexagonal_1/, ... It supplies the template library, the '
                             'hyperparameters and the shipped ranker.')
    parser.add_argument('--work-dir', required=True, metavar='PATH',
                        help='Where trained rankers are written, one models directory per input '
                             'set under WORK_DIR/models/.')
    parser.add_argument('--input-sets', default='', metavar='A,B',
                        help=f'Comma-separated input sets. Known: {",".join(TEMPLATE_INPUT_SETS)}.')
    parser.add_argument('--bravais-lattices', default=','.join(BRAVAIS_LATTICES), metavar='A,B',
                        help='Comma-separated lattices (default: all fourteen).')
    parser.add_argument('--per-lattice', type=int, default=50, metavar='N',
                        help='Crystals drawn per lattice before the fit/select division, or from '
                             'fom-dev for --group report (default: 50).')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='Fixes the crystals drawn, their groups, their noise and the training '
                             'draws (default: 12345).')
    parser.add_argument('--processes', type=int, default=1, metavar='N',
                        help='Jobs run at once; one job is one input set on one lattice '
                             '(default: 1).')
    parser.add_argument('--dataset-directory', default=None, metavar='PATH',
                        help='Where the per-lattice source datasets live (default: the packaged '
                             'mlindex/data/generated_datasets).')

    measure = parser.add_argument_group('measuring yield')
    measure.add_argument('--group', default='select', choices=GROUPS,
                         help='Which crystals to measure on (default: select).')
    measure.add_argument('--population', default='general', choices=tuple(runs.POPULATIONS),
                         help='Supplies the condition bundles when --bundles is not given '
                              '(default: general).')
    measure.add_argument('--bundles', default='', metavar='A,B',
                         help='Comma-separated condition bundles (default: the population\'s).')
    measure.add_argument('--search-seed', type=int, default=12345, metavar='N',
                         help='Reaches template sampling and the search alone (default: 12345).')
    measure.add_argument('--cut', type=float, default=1.5, metavar='M20',
                         help='The M20 threshold refinement prunes at (default: 1.5).')
    measure.add_argument('--rtol', default=','.join(f'{rtol:g}' for rtol in DEFAULT_RTOLS),
                         metavar='A,B',
                         help='Relative tolerances a template must match the true cell to '
                              '(default: 0.001,0.003,0.01,0.03).')
    measure.add_argument('--out-dir', default=None, metavar='PATH',
                         help='Where the per-pattern table and its manifest are written.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.input_sets = _split(args.input_sets)
    args.bravais_lattices = _split(args.bravais_lattices)
    args.bundles = _split(args.bundles)
    args.rtol = [float(value) for value in _split(args.rtol)]
    unknown = [name for name in args.input_sets if name not in TEMPLATE_INPUT_SETS]
    if unknown:
        raise SystemExit(f'Unknown input set(s) {unknown}. Known: {list(TEMPLATE_INPUT_SETS)}')
    unknown = [name for name in args.bravais_lattices if name not in BRAVAIS_LATTICES]
    if unknown:
        raise SystemExit(f'Unknown Bravais lattice(s) {unknown}.')
    unknown = [name for name in args.bundles if name not in BenchmarkConditions.BY_TAG]
    if unknown:
        raise SystemExit(f'Unknown condition bundle(s) {unknown}. '
                         f'Known: {list(BenchmarkConditions.tags())}')
    if args.stage == 'train':
        if not args.input_sets:
            raise SystemExit('--stage train needs --input-sets.')
        run_train(args)
    elif args.stage == 'yield':
        if args.out_dir is None:
            raise SystemExit('--stage yield needs --out-dir.')
        run_yield(args)


if __name__ == '__main__':
    main()
