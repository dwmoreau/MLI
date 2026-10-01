"""Fit and measure the learned ranker on benchmark pools.

Three stages, each a separate job on a cluster:

    # 1. export: a pool as ranker frames, one condition bundle per call (or all of them)
    python -m mlindex.scripts.run_ranker --stage export --pool pools/train_general \
        --out-dir ranker/export --bundle b0_error0p5_cont0
    python -m mlindex.scripts.run_ranker --stage export --pool pools/dev_general \
        --out-dir ranker/export

    # 2. fit: one model, with its learning curve on the held-out selection crystals
    python -m mlindex.scripts.run_ranker --stage fit \
        --export-dir ranker/export/<commit>/train_general_cut3.5_depth20 \
        --out-dir ranker/fits --encoding onehot --learning-rate 0.04 --max-leaf-nodes 63 \
        --max-iter 2000 --seed 12345

    # 3. evaluate: saved models on a reporting pool, beside M20 and M_sym
    python -m mlindex.scripts.run_ranker --stage evaluate \
        --export-dir ranker/export/<commit>/dev_general_cut3.5_depth20 \
        --model-dir ranker/fits/<commit>/onehot_lr0.04_leaves63_iter2000_seed12345 \
        --out-dir ranker/evaluate

What each stage does:

* **export** reads a pool written by `run_benchmark --stage generate`. From a `fom-train` pool it
  holds out a fixed share of crystals (`--selection-fraction`, drawn once with `--split-seed`)
  for choosing settings, and writes the rest as training rows: every correct candidate and a
  thinned, seed-keyed sample of the wrong ones, for each of `--seeds`. The held-out crystals, and
  every crystal of a `fom-dev` pool, are written as evaluation rows: the pool restricted to
  `--cut`, the top `--depth` per lattice.
* **fit** divides the training crystals into fit and calibration parts by `--seed`, fits the
  classifier on the fit part, and at every `--checkpoint-every` trees calibrates on the
  calibration part and ranks the selection crystals. It writes the model and that curve.
* **evaluate** scores a reporting export with each saved model, checks that the model's ONNX
  export gives the same probabilities and the same ranking, and writes per-pattern outcomes and
  each model's calibration (Brier score and expected calibration error, per lattice).

Every ranking a fit or an evaluation reports comes three ways: through the per-lattice isotonic
maps (the shipped design), through one pooled map, and by the classifier's raw score, so whether
the per-lattice calibration earns its place in the ranking is measured on the same models.

Every output directory names the commit that wrote it, and no stage writes over an existing
result.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import Benchmark
from mlindex.model_training import FomCombiner as Ranker
from mlindex.model_training.BenchmarkMetrics import derive_flags
from mlindex.model_training.BenchmarkMetrics import reduce_many
from mlindex.model_training.BenchmarkRuns import POPULATIONS
from mlindex.model_training.BenchmarkRuns import committed_checkout

SEEDS = (12345, 777, 20260826)
CRYSTALS_NAME = 'crystals.parquet'
OUTCOMES = ('found', 'top1', 'top5', 'top10', 'reciprocal_rank')
REFERENCE_SCORES = ('M20', 'M_sym')
ONNX_CHECK_ROWS = 2_000_000
# How each calibration arm is named in a learning curve and in an evaluation's outcome files.
CURVE_NAMES = {'per_lattice': 'ranker', 'pooled': 'ranker_pooled', 'raw': 'ranker_raw'}
ARM_SUFFIX = {'per_lattice': '', 'pooled': '__pooled', 'raw': '__raw'}


def _integers(text):
    return tuple(int(value) for value in text.split(',') if value)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Fit and measure the learned ranker on benchmark pools.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--stage', required=True, choices=('export', 'fit', 'evaluate'))
    parser.add_argument('--out-dir', required=True, metavar='DIR',
                        help='Where results go; each stage writes under DIR/<commit>/.')

    export = parser.add_argument_group('export')
    export.add_argument('--pool', metavar='DIR', help='A pool from run_benchmark.')
    export.add_argument('--bundle', metavar='TAG',
                        help='One condition bundle; all of the pool\'s when omitted.')
    export.add_argument('--seeds', type=_integers, default=SEEDS, metavar='S1,S2,...',
                        help='Fit seeds to write training rows for (default %(default)s).')
    export.add_argument('--selection-fraction', type=float, default=0.15, metavar='F',
                        help='Share of fom-train crystals held out for choosing settings.')
    export.add_argument('--split-seed', type=int, default=12345, metavar='S',
                        help='Draws the selection crystals; fixed across fits.')
    export.add_argument('--cut', type=float, default=3.5, metavar='M20',
                        help='The M20 cut the evaluation rows are restricted to.')
    export.add_argument('--depth', default='20', metavar='N|all',
                        help='Evaluation rows kept per lattice: a number, or all.')
    export.add_argument('--top-k', type=int, default=200, metavar='K',
                        help='Wrong candidates kept with certainty per merit and lattice.')
    export.add_argument('--negative-rate', type=float, default=0.05, metavar='R',
                        help='Sampling rate for the other wrong candidates.')
    export.add_argument('--n-negatives', type=int, default=400, metavar='N',
                        help='Wrong candidates kept per pattern; the most any fit may use.')

    fit = parser.add_argument_group('fit')
    fit.add_argument('--export-dir', metavar='DIR', help='An export directory.')
    fit.add_argument('--encoding', choices=Ranker.LATTICE_ENCODINGS, default='onehot')
    fit.add_argument('--learning-rate', type=float, default=0.04)
    fit.add_argument('--max-leaf-nodes', type=int, default=63)
    fit.add_argument('--max-iter', type=int, default=600, metavar='TREES')
    fit.add_argument('--seed', type=int, default=SEEDS[0], metavar='S',
                     help='Draws the fit/calibration split and the wrong-candidate sample.')
    fit.add_argument('--checkpoint-every', type=int, default=100, metavar='TREES')
    fit.add_argument('--fit-negatives', type=int, default=40, metavar='N',
                     help='Wrong candidates per pattern in the fit rows.')
    fit.add_argument('--calibration-negatives', type=int, default=400, metavar='N',
                     help='Wrong candidates per pattern in the calibration rows.')
    fit.add_argument('--calibration-fraction', type=float, default=0.2, metavar='F',
                     help='Share of training crystals the calibrators are fitted on.')

    evaluate = parser.add_argument_group('evaluate')
    evaluate.add_argument('--model-dir', action='append', default=[], metavar='DIR',
                          help='A fitted model; repeat for several.')
    return parser


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------
def _depth(text):
    return None if text == 'all' else int(text)


def run_export(args, commit):
    pool = Path(args.pool)
    Benchmark.check_complete(pool)
    manifest = Benchmark.load_manifest(pool)
    split = manifest['split']
    depth = _depth(args.depth)
    target = (Path(args.out_dir) / commit[:7]
              / f'{pool.name}_cut{args.cut:g}_depth{args.depth}')
    target.mkdir(parents=True, exist_ok=True)

    entries = Benchmark.load_entries(pool)
    if set(entries['split']) != {split}:
        raise SystemExit(f'{pool}: the entry table holds splits {sorted(set(entries["split"]))}, '
                         f'not only the manifest\'s {split!r}')
    if split == 'fom-train':
        parts = Ranker.split_crystals(
            entries, {'selection': args.selection_fraction,
                      'training': 1.0 - args.selection_fraction},
            np.random.default_rng(args.split_seed))
    elif split == 'fom-dev':
        parts = {'selection': set(entries['entry_id']), 'training': set()}
    else:
        raise SystemExit(f'{pool} is a {split!r} pool; only fom-train and fom-dev are read here')
    _write_crystals(target, entries, parts, dict(
        pool=str(pool), pool_commit=manifest['commit'], split=split,
        selection_fraction=args.selection_fraction, split_seed=args.split_seed))

    bundles = [args.bundle] if args.bundle else Benchmark.available_bundles(pool)
    for bundle in bundles:
        stamp = target / f'export_{bundle}.json'
        if stamp.exists():
            raise SystemExit(f'{stamp} exists: this bundle is already exported to {target}')
        training, evaluation = Ranker.export_bundle(
            pool, bundle, entries, parts['training'], parts['selection'], args.seeds,
            cut=args.cut, n_top=depth or Benchmark.N_TOP_CANDIDATES,
            keep_all_depths=depth is None, top_k=args.top_k,
            negative_rate=args.negative_rate, n_negatives=args.n_negatives)
        for seed, frame in training.items():
            frame.to_parquet(target / f'training_{bundle}_seed{seed}.parquet', index=False)
        if evaluation is not None:
            evaluation.to_parquet(target / f'evaluation_{bundle}.parquet', index=False)
        record = dict(commit=commit, bundle=bundle, pool=str(pool),
                      pool_commit=manifest['commit'], split=split, cut=args.cut,
                      depth=args.depth, top_k=args.top_k, negative_rate=args.negative_rate,
                      n_negatives=args.n_negatives, seeds=sorted(training),
                      n_training_rows={str(seed): int(frame.shape[0])
                                       for seed, frame in training.items()},
                      n_evaluation_rows=0 if evaluation is None else int(evaluation.shape[0]))
        with open(stamp, 'w', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2)
        print(f'export {bundle}: {record["n_training_rows"]} training rows, '
              f'{record["n_evaluation_rows"]} evaluation rows -> {target}')
    return 0


def _write_crystals(target, entries, parts, settings):
    """The crystal division, written once; a second export of the same directory must agree."""
    crystals = entries[['entry_id', 'bravais_lattice_true']].drop_duplicates('entry_id')
    crystals = crystals.assign(part=np.where(crystals['entry_id'].isin(parts['training']),
                                             'training', 'selection'))
    crystals = crystals.sort_values('entry_id').reset_index(drop=True)
    path = target / CRYSTALS_NAME
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.equals(crystals):
            raise SystemExit(f'{path} holds a different crystal division; export into a fresh '
                             'directory or with the same --selection-fraction and --split-seed')
        return
    crystals.to_parquet(path, index=False)
    with open(target / 'crystals.json', 'w', encoding='utf-8') as handle:
        json.dump(settings, handle, indent=2)


def _read_export(directory):
    """The export's crystal table and per-bundle stamps, refusing one that is incomplete."""
    directory = Path(directory)
    crystals = pd.read_parquet(directory / CRYSTALS_NAME)
    stamps = []
    for path in sorted(directory.glob('export_*.json')):
        with open(path, encoding='utf-8') as handle:
            stamps.append(json.load(handle))
    if not stamps:
        raise SystemExit(f'{directory} holds no exported bundle')
    with open(directory / 'crystals.json', encoding='utf-8') as handle:
        pool = json.load(handle)['pool']
    expected = Benchmark.available_bundles(pool) if Path(pool).is_dir() else None
    found = sorted(stamp['bundle'] for stamp in stamps)
    if expected is not None and found != sorted(expected):
        raise SystemExit(f'{directory} has bundles {found}; its pool has {sorted(expected)}')
    for field in ('pool_commit', 'cut', 'depth', 'top_k', 'negative_rate', 'n_negatives',
                  'seeds'):
        if len({json.dumps(stamp[field]) for stamp in stamps}) != 1:
            raise SystemExit(f'{directory}: the bundles were exported with different {field}')
    return crystals, stamps


def _evaluation_frame(directory, crystals):
    frames = [pd.read_parquet(path) for path in sorted(Path(directory).glob('evaluation_*.parquet'))]
    frame = pd.concat(frames, ignore_index=True)
    return frame.merge(crystals[['entry_id', 'bravais_lattice_true']], on='entry_id',
                       how='left', validate='m:1')


# ---------------------------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------------------------
def outcomes(frame, scores):
    """Per pattern-condition outcome flags for each score, long form."""
    reductions = reduce_many(frame, scores, depths=('all',))
    rows = []
    lattice = frame.drop_duplicates(Benchmark.ENTRY_KEY)[Benchmark.ENTRY_KEY
                                                         + ['bravais_lattice_true']]
    for name, reduced in reductions.items():
        flags = derive_flags(reduced, depth='all')[Benchmark.ENTRY_KEY + list(OUTCOMES)]
        rows.append(flags.assign(score=name))
    return pd.concat(rows, ignore_index=True).merge(lattice, on=Benchmark.ENTRY_KEY, how='left')


def population_means(flags, by='score'):
    """Each outcome, as a mean over Bravais lattices of the per-lattice mean, per population and
    per value of `by`.

    The fom-train crystals are not balanced across lattices, so a plain mean would weight aP 27
    times cF; a mean of lattice means weights the lattices equally, as the fom-dev sample does.
    """
    rows = []
    for population, spec in POPULATIONS.items():
        mask = (flags['bravais_lattice_true'].isin(spec['bravais_lattices'])
                & flags['condition_bundle'].isin(spec['bundles']))
        subset = flags.loc[mask]
        if not subset.shape[0]:
            continue
        per_lattice = subset.groupby([by, 'bravais_lattice_true'])[list(OUTCOMES)].mean()
        means = per_lattice.groupby(by).mean().reset_index()
        rows.append(means.assign(population=population,
                                 n_patterns=subset.groupby(by).size().to_numpy()))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------------------------
def run_fit(args, commit):
    export_dir = Path(args.export_dir)
    crystals, stamps = _read_export(export_dir)
    if args.seed not in stamps[0]['seeds']:
        raise SystemExit(f'{export_dir} has training rows for seeds {stamps[0]["seeds"]}, '
                         f'not {args.seed}')
    for wanted in (args.fit_negatives, args.calibration_negatives):
        if wanted > stamps[0]['n_negatives']:
            raise SystemExit(f'the export kept {stamps[0]["n_negatives"]} wrong candidates a '
                             f'pattern; {wanted} cannot be drawn from it')
    name = (f'{args.encoding}_lr{args.learning_rate:g}_leaves{args.max_leaf_nodes}'
            f'_iter{args.max_iter}_seed{args.seed}')
    target = Path(args.out_dir) / commit[:7] / name
    if target.exists():
        raise SystemExit(f'{target} exists; every fit writes its own directory')

    training_ids = crystals.loc[crystals['part'] == 'training']
    parts = Ranker.split_crystals(
        training_ids, {'fit': 1.0 - args.calibration_fraction,
                       'calibration': args.calibration_fraction},
        np.random.default_rng(args.seed))
    if not parts['fit'] or not parts['calibration']:
        raise SystemExit(f'{len(training_ids)} training crystals give {len(parts["fit"])} to fit '
                         f'and {len(parts["calibration"])} to calibrate; both need some, so '
                         'raise --calibration-fraction or export more crystals')
    frames = [pd.read_parquet(path) for path in
              sorted(export_dir.glob(f'training_*_seed{args.seed}.parquet'))]
    training = pd.concat(frames, ignore_index=True)
    fit_rows = Ranker.cap_negatives(training.loc[training['entry_id'].isin(parts['fit'])],
                                    args.fit_negatives)
    calibration_rows = Ranker.cap_negatives(
        training.loc[training['entry_id'].isin(parts['calibration'])],
        args.calibration_negatives)
    del training, frames
    selection = _evaluation_frame(export_dir, crystals)
    if set(selection['entry_id']) & (parts['fit'] | parts['calibration']):
        raise SystemExit('a selection crystal is also a fit or calibration crystal')
    print(f'fit {fit_rows.shape[0]} rows ({len(parts["fit"])} crystals), calibrate '
          f'{calibration_rows.shape[0]} rows ({len(parts["calibration"])} crystals), select on '
          f'{selection.shape[0]} rows ({selection["entry_id"].nunique()} crystals)')

    combiner = Ranker.FomCombiner.fit(
        fit_rows, encoding=args.encoding, seed=args.seed,
        params=dict(max_iter=args.max_iter, learning_rate=args.learning_rate,
                    max_leaf_nodes=args.max_leaf_nodes))
    curve = learning_curve(combiner, calibration_rows, selection, args.checkpoint_every)
    combiner.fit_calibrators(calibration_rows)
    combiner.meta.update(commit=commit, export_dir=str(export_dir),
                         fit_negatives=args.fit_negatives,
                         calibration_negatives=args.calibration_negatives,
                         n_fit_crystals=len(parts['fit']),
                         n_calibration_crystals=len(parts['calibration']))
    combiner.save(target)
    curve.to_parquet(target / 'curve_outcomes.parquet', index=False)
    summary = population_means(curve, by='curve_point')
    summary.to_csv(target / 'curve.csv', index=False)
    print(summary.loc[summary['population'] == 'general'].tail(3).to_string(index=False))
    print(f'wrote {target}')
    return 0


def learning_curve(combiner, calibration, selection, every):
    """Selection outcomes at every `every` trees, calibrating on `calibration` at each stage.

    Each checkpoint is ranked three ways (`FomCombiner.CALIBRATION_ARMS`): `ranker@T` through
    the per-lattice maps, `ranker_pooled@T` through the pooled map, `ranker_raw@T` by the raw
    score.
    """
    from sklearn.metrics import log_loss

    model = combiner.model
    calibration_matrix = combiner.design_matrix(calibration)
    selection_matrix = combiner.design_matrix(selection)
    target = calibration['is_correct'].to_numpy(dtype=bool)
    weights = calibration['fit_weight'].to_numpy(dtype=np.float64)
    checkpoints = set(range(every, model.n_iter_ + 1, every)) | {model.n_iter_}
    rows = []
    stages = zip(model.staged_predict_proba(calibration_matrix),
                 model.staged_predict_proba(selection_matrix))
    for n_trees, (raw_calibration, raw_selection) in enumerate(stages, start=1):
        if n_trees not in checkpoints:
            continue
        calibrators = Ranker.fit_calibration(raw_calibration[:, 1], target,
                                             calibration['bravais_lattice'], weights)
        arms = Ranker.calibration_arms(raw_selection[:, 1], selection['bravais_lattice'],
                                       calibrators)
        flags = outcomes(selection, {CURVE_NAMES[arm]: score for arm, score in arms.items()})
        flags['curve_point'] = flags['score'] + f'@{n_trees:05d}'
        flags['n_trees'] = n_trees
        truth = selection['is_correct'].to_numpy(dtype=bool)
        losses = {CURVE_NAMES[arm]: log_loss(truth, np.clip(score, 1e-15, 1 - 1e-15))
                  for arm, score in arms.items()}
        flags['selection_log_loss'] = flags['score'].map(losses)
        rows.append(flags.drop(columns='score'))
    if not rows:
        raise SystemExit('no checkpoint reached; lower --checkpoint-every')
    reference = outcomes(selection, {name: name for name in REFERENCE_SCORES})
    reference = reference.rename(columns={'score': 'curve_point'}).assign(
        n_trees=0, selection_log_loss=np.nan)
    return pd.concat(rows + [reference], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------------------------
def calibration_rows(probability, correct, lattice, n_bins=10):
    """Brier score and expected calibration error, over all rows and per Bravais lattice.

    The calibration error is the row-weighted mean, over `n_bins` bins of equal row count by
    predicted probability, of |mean predicted - share correct|.
    """
    def one(p, y):
        order = np.argsort(p, kind='stable')
        bins = np.array_split(order, n_bins)
        gaps = [abs(p[b].mean() - y[b].mean())*b.size for b in bins if b.size]
        return dict(n_rows=int(p.size), base_rate=float(y.mean()), brier=float(np.mean((p - y)**2)),
                    ece=float(np.sum(gaps)/p.size))

    probability = np.asarray(probability, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    lattice = np.asarray(lattice)
    rows = [dict(scope='aggregate', **one(probability, correct))]
    for name in np.unique(lattice):
        mask = lattice == name
        rows.append(dict(scope=f'bravais_lattice={name}', **one(probability[mask], correct[mask])))
    return rows


def onnx_check_rows(frame, limit=ONNX_CHECK_ROWS, seed=12345):
    """Whole crystals, drawn in a fixed random order until `limit` rows: the rows the ONNX export
    is compared on. A whole-pool export is too large to score twice."""
    crystals = np.sort(frame['entry_id'].unique())
    order = crystals[np.random.default_rng(seed).permutation(crystals.size)]
    sizes = frame.groupby('entry_id').size().reindex(order).to_numpy()
    keep = order[:max(1, int(np.searchsorted(np.cumsum(sizes), limit, side='right')))]
    return frame['entry_id'].isin(keep)


def run_evaluate(args, commit):
    export_dir = Path(args.export_dir)
    crystals, stamps = _read_export(export_dir)
    if stamps[0]['split'] != 'fom-dev':
        raise SystemExit(f'{export_dir} is a {stamps[0]["split"]} export; the reporting split '
                         'is fom-dev')
    if not args.model_dir:
        raise SystemExit('name at least one --model-dir')
    frame = _evaluation_frame(export_dir, crystals)
    target = Path(args.out_dir) / commit[:7] / export_dir.name
    target.mkdir(parents=True, exist_ok=True)

    scores = {name: name for name in REFERENCE_SCORES}
    agreement, calibration = [], []
    sample = onnx_check_rows(frame)
    for directory in args.model_dir:
        directory = Path(directory)
        name = directory.name
        if (target / f'outcomes_{name}.parquet').exists():
            raise SystemExit(f'{target} already holds outcomes for {name}')
        combiner = Ranker.FomCombiner.load(directory)
        raw = combiner.raw_score(frame)
        arms = Ranker.calibration_arms(raw, frame['bravais_lattice'], combiner.calibrators)
        probability = arms['per_lattice']
        for arm, score in arms.items():
            scores[name + ARM_SUFFIX[arm]] = score
            calibration += [dict(model=name, calibration_arm=arm, **row) for row in
                            calibration_rows(score, frame['is_correct'].to_numpy(dtype=bool),
                                             frame['bravais_lattice'])]
        if combiner.encoding in Ranker.EXPORTABLE_ENCODINGS:
            check = sample.to_numpy()
            onnx_raw = Ranker.onnx_probability(directory / 'model.onnx',
                                               combiner.design_matrix(frame.loc[check]))
            onnx_probability = Ranker.apply_calibration(
                onnx_raw, frame.loc[check, 'bravais_lattice'], combiner.calibrators)
            same = outcomes(frame.loc[check].reset_index(drop=True),
                            {'sklearn': probability[check], 'onnx': onnx_probability})
            pivot = same.pivot_table(index=Benchmark.ENTRY_KEY, columns='score',
                                     values=['top1', 'top10'])
            agreement.append(dict(
                model=name, n_rows_checked=int(check.sum()),
                n_patterns_checked=int(pivot.shape[0]),
                max_abs_raw=float(np.max(np.abs(onnx_raw - raw[check]))),
                # A float32 input equal to a float64 split threshold rounded up takes the other
                # branch in ONNX, which compares in float32; this counts the rows it reaches.
                n_rows_raw_differ=int(np.sum(np.abs(onnx_raw - raw[check]) > 1e-5)),
                max_abs_calibrated=float(np.max(np.abs(onnx_probability - probability[check]))),
                top1_disagreements=int((pivot['top1']['sklearn'] != pivot['top1']['onnx']).sum()),
                top10_disagreements=int(
                    (pivot['top10']['sklearn'] != pivot['top10']['onnx']).sum())))
            print(agreement[-1])
    flags = outcomes(frame, scores)
    for name in scores:
        flags.loc[flags['score'] == name].to_parquet(target / f'outcomes_{name}.parquet',
                                                     index=False)
    population_means(flags).to_csv(target / 'summary.csv', index=False)
    pd.DataFrame(calibration).to_csv(target / 'calibration.csv', index=False)
    with open(target / 'onnx_agreement.json', 'w', encoding='utf-8') as handle:
        json.dump(dict(commit=commit, export_dir=str(export_dir), models=agreement), handle,
                  indent=2)
    print(population_means(flags).to_string(index=False))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    required = {'export': ('pool',), 'fit': ('export_dir',), 'evaluate': ('export_dir',)}
    missing = [name for name in required[args.stage] if getattr(args, name) is None]
    if missing:
        raise SystemExit(f'--stage {args.stage} needs ' + ', '.join(
            f'--{name.replace("_", "-")}' for name in missing))
    commit = committed_checkout()
    return {'export': run_export, 'fit': run_fit, 'evaluate': run_evaluate}[args.stage](
        args, commit)


if __name__ == '__main__':
    sys.exit(main())
