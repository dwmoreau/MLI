"""S14 -- the neural scoring network, fitted, reduced and analysed beside S12's tree.

    python mlindex/scripts/run_fom_neural_score.py --stage fit        [--fit-seed 777 --suffix _seed777]
    python mlindex/scripts/run_fom_neural_score.py --stage reduce --arms tree tree_fullscale
    python mlindex/scripts/run_fom_neural_score.py --stage reduce --arms network plus_prior_claimed ...
    python mlindex/scripts/run_fom_neural_score.py --stage analyse
    python mlindex/scripts/run_fom_neural_score.py --stage calibration
    python mlindex/scripts/run_fom_neural_score.py --stage cost
    python mlindex/scripts/run_fom_neural_score.py --stage combine

The same two-pool design as S12 (`run_fom_combiner.py`, decision 2026-09-01): fit and choose
thresholds on the Benchmark B slice's `fom-train`, report on the fully retained pool's `fom-dev`,
where a learned score's rank is exact. Everything that is a function of a reduction is imported
from the S12 driver rather than re-implemented -- the splits, the negative subsampling, the shard
guards, the threshold rule, the McNemar pairing -- so the two steps are measured by one procedure.

**What is different, and why.**

  * The arms are networks (`NeuralScore`) over DWMM's ~50 inputs, plus two trees: `tree`, S12's
    `plus_probation` feature set REFITTED on the same rows the networks see, so the tree-versus-
    network contrast is at equal fit size; and `tree_fullscale`, S12's shipped model, loaded from
    its own directory with its `n_rows` asserted, as the campaign's reference level. A network
    fitted on 363 k rows against a tree fitted on 2.4 M would be measuring scale, not design.
  * The reduce runs in TWO passes over the report pool, `--arms` naming which. One frame carrying
    S12's groups and S14's is ~8.5 GB and its float64 design matrix another 10 GB, which a 16 GB
    laptop cannot hold. Metas are merged into the run's meta file, never rewritten, so the second
    pass does not destroy the first (which is what `run_fom_combiner.run_reduce` would do).
  * `--inputs-dir` merges S14's inputs onto S12's exported full-scale frames for a full-scale fit,
    asserting every row found its inputs; `submit_fom_neural_inputs.sh` is what produces them.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomCombiner
from mlindex.model_training import FomMetrics
from mlindex.model_training import NeuralScore
from mlindex.scripts import run_fom_combiner as s12

BASE = s12.BASE
FIT_POOL = s12.FIT_POOL
REPORT_POOL = s12.REPORT_POOL
ARTIFACT_DIR = s12.ARTIFACT_DIR
MODELS_DIR = BASE/'mlindex'/'models'/'fom_neural_score'
TREE_FULLSCALE = BASE/'mlindex'/'models'/'fom_combiner_c2_fullscale'/'plus_probation_seed12345'
TREE_FULLSCALE_ROWS = 2_381_244        # C2-F-143; asserted on load, never assumed
SEED = s12.SEED
SEEDS = s12.SEEDS
SEED_SUFFIXES = s12.SEED_SUFFIXES
TAG = 'S14_neural'

# DWMM's base: block A (entry level), block B, the candidate's volume and lattice. `structural`
# is the group that carries `log_volume` and the `bravais_lattice` categorical; everything else in
# it is dropped by name, so the base arm reads exactly 14 + 14 + 2 entropies + 20 + 1 + 1 columns.
NETWORK_GROUPS = ('structural', 'prior_entry', 'prior_volume', 'assignment_peaks')
NETWORK_DROP = tuple(name for name in FomCombiner.STRUCTURAL_NUMERIC if name != 'log_volume') \
    + ('spacegroup',)
TREE_GROUPS = ('raw', 'structural', 'context', 'counts', 'probation')    # S12's plus_probation
NETWORK_PARAMS = dict(hidden=(64, 32), epochs=40, batch_size=4096, learning_rate=1e-3,
                      patience=5)

# (name, kind, groups, drop, weight_column, purpose). `kind` is 'network' or 'tree'.
ARMS = (
    ('network', 'network', NETWORK_GROUPS, NETWORK_DROP, 'sampling_weight',
     "DWMM's design: prior lattice distribution + per-lattice volume, 20 per-peak posteriors, "
     'candidate volume and lattice'),
    ('plus_prior_claimed', 'network', NETWORK_GROUPS + ('prior_claimed',), NETWORK_DROP,
     'sampling_weight', 'plus the joint read at the claimed pair -- the only block-A column that '
     'can reorder within an entry (the re-ranker question)'),
    ('plus_asg_sigma', 'network', NETWORK_GROUPS + ('assignment_sigma',), NETWORK_DROP,
     'sampling_weight', "plus the posterior's own denominator, log sigma (F-131)"),
    ('drop_A', 'network', ('structural', 'assignment_peaks'), NETWORK_DROP, 'sampling_weight',
     'block B alone: is the prior doing anything (super-additivity, handoff item 3)'),
    ('drop_B', 'network', ('structural', 'prior_entry', 'prior_volume'), NETWORK_DROP,
     'sampling_weight', 'block A alone: is the per-peak fit doing anything'),
    ('unweighted_fit', 'network', NETWORK_GROUPS, NETWORK_DROP, None,
     'the C2-Q-031 check: fit unweighted on the thinned negatives'),
    ('label_shuffled', 'network', NETWORK_GROUPS, NETWORK_DROP, 'sampling_weight',
     'control: labels permuted within each (entry, bundle); must land between the floors'),
    ('tree', 'tree', TREE_GROUPS, s12.BASE_DROP, 'sampling_weight',
     "S12's plus_probation feature set, refitted on exactly the rows the networks see"),
    ('tree_plus_blocks', 'tree', TREE_GROUPS + FomCombiner.NEURAL_ENTRY_GROUPS
     + FomCombiner.NEURAL_CANDIDATE_GROUPS, s12.BASE_DROP,
     'sampling_weight', 'the tree given the network\'s inputs too: inputs or architecture'),
    # DWMM's redirect (decision 2026-09-05): block A as two ratio features in the combiner.
    ('tree_ratio_marginal', 'tree',
     TREE_GROUPS + ('prior_ratio_volume_marginal', 'prior_ratio_dof'), s12.BASE_DROP,
     'sampling_weight', "S12's tree plus v_candidate/v_inferred (lattice-marginal volume) and "
     'dof_candidate/E[dof]'),
    ('tree_ratio_claimed', 'tree',
     TREE_GROUPS + ('prior_ratio_volume_claimed', 'prior_ratio_dof'), s12.BASE_DROP,
     'sampling_weight', "S12's tree plus v_candidate/v_inferred (volume at the claimed lattice) "
     'and dof_candidate/E[dof]'),
    ('tree_ratio_volume_only', 'tree', TREE_GROUPS + ('prior_ratio_volume_marginal',),
     s12.BASE_DROP, 'sampling_weight', 'the marginal volume ratio alone'),
    ('tree_ratio_dof_only', 'tree', TREE_GROUPS + ('prior_ratio_dof',), s12.BASE_DROP,
     'sampling_weight', 'the dof ratio alone'),
    ('tree_plus_joint', 'tree', TREE_GROUPS + ('prior_claimed',), s12.BASE_DROP,
     'sampling_weight', 'the principled alternative: the joint P(V, lattice) read at the claimed '
     'pair, its margin, and the support flag'),
    )
NETWORK_ARMS = tuple(name for name, kind, *_ in ARMS if kind == 'network')
TREE_ARMS = tuple(name for name, kind, *_ in ARMS if kind == 'tree')


def _log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------------------------
# stage: fit
# ---------------------------------------------------------------------------------------------
def models_directory(args):
    return Path(args.models_dir + args.suffix) if args.suffix else Path(args.models_dir)


def union_groups(names=None):
    groups = []
    for name, _, arm_groups, *_ in ARMS:
        if names and name not in names:
            continue
        groups.extend(arm_groups)
    return tuple(dict.fromkeys(tuple(FomCombiner.DEFAULT_GROUPS) + tuple(groups)))


def merge_inputs(frames, inputs_dir):
    """S14's inputs joined onto S12's exported frames, every row accounted for.

    Per-candidate columns from the keyed sidecars `run_fom_neural_inputs.py --keys-from` wrote,
    entry-level columns from its `prior_entries.parquet`; both joined 1:1 and a row without its
    inputs is a hard failure, because a NaN input here is indistinguishable from a cubic claim.
    """
    inputs_dir = Path(inputs_dir)
    shards = sorted(inputs_dir.glob('candidates*.parquet'))
    if not shards:
        raise SystemExit(f'no keyed input sidecars under {inputs_dir}')
    candidate_inputs = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    entry_inputs = pd.read_parquet(inputs_dir/FomCombiner.NEURAL_ENTRY_FILE)
    keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
    out = []
    for frame in frames:
        merged = frame.merge(candidate_inputs, on=keys, how='left', validate='1:1',
                             indicator=True)
        lost = int((merged['_merge'] != 'both').sum())
        if lost:
            raise SystemExit(f'{lost} of {frame.shape[0]} rows found no per-candidate inputs in '
                             f'{inputs_dir}; re-run the keyed input pass over these frames')
        merged = merged.drop(columns=['_merge']).merge(
            entry_inputs, on=['entry_id', 'condition_bundle'], how='left', validate='m:1',
            indicator=True)
        lost = int((merged['_merge'] != 'both').sum())
        if lost:
            raise SystemExit(f'{lost} rows found no entry-level prior in {inputs_dir}')
        out.append(merged.drop(columns=['_merge']))
    return out


def load_slice_frames(args):
    entries = FomBenchmark.load_entries(FIT_POOL)
    covariates = FomCombiner.neural_covariates(FIT_POOL, entries)
    fit_ids, cal_ids = s12.split_ids(entries, args.train_split, s12.HOLDOUT_FRACTION, SEED)
    report_entries = FomBenchmark.load_entries(REPORT_POOL)
    s12.assert_disjoint(set(fit_ids) | set(cal_ids), set(report_entries['entry_id']))
    _log(f'fit {len(fit_ids)} crystals, calibrate {len(cal_ids)}, report on '
         f'{report_entries["entry_id"].nunique()} disjoint crystals')
    groups = union_groups()
    fit_frames = s12.load_fit_frames(FIT_POOL, entries, fit_ids, groups, s12.N_NEGATIVES, SEED,
                                     covariates=covariates)
    cal_frames = s12.load_fit_frames(FIT_POOL, entries, cal_ids, groups, None, SEED,
                                     covariates=covariates)
    return fit_frames, cal_frames


def load_frame_files(args):
    """S12's exported full-scale frames plus S14's inputs (`--fit-frame` + `--inputs-dir`)."""
    fit_paths = sorted(Path().glob(args.fit_frame)) if any(c in args.fit_frame for c in '*?[') \
        else [Path(args.fit_frame)]
    if not fit_paths:
        raise SystemExit(f'--fit-frame matched nothing: {args.fit_frame}')
    pairs = []
    for fit_path in fit_paths:
        cal_path = fit_path.with_name(fit_path.name.replace('_fit_frame', '_cal_frame'))
        if cal_path == fit_path or not cal_path.exists():
            raise SystemExit(f'--fit-frame needs the `_cal_frame` sibling of {fit_path.name}')
        pairs.append((fit_path, cal_path))
    fit_frames = [pd.read_parquet(path) for path, _ in pairs]
    cal_frames = [pd.read_parquet(path) for _, path in pairs]
    covered = sorted({str(f['condition_bundle'].iloc[0]) for f in fit_frames if f.shape[0]})
    s12.assert_disjoint({e for f in fit_frames for e in f['entry_id'].unique()},
                        {e for f in cal_frames for e in f['entry_id'].unique()})
    s12._assert_shards_complete(pairs, covered, args.allow_partial_bundles)
    if not args.inputs_dir:
        raise SystemExit('--fit-frame needs --inputs-dir: S12\'s frames carry none of the S14 inputs')
    fit_frames = merge_inputs(fit_frames, args.inputs_dir)
    cal_frames = merge_inputs(cal_frames, args.inputs_dir)
    _log(f'read {len(pairs)} shard(s): {sum(f.shape[0] for f in fit_frames):,} fit rows, '
         f'{sum(f.shape[0] for f in cal_frames):,} calibration rows, {len(covered)} bundle(s), '
         f'inputs from {args.inputs_dir}')
    return fit_frames, cal_frames


def fit_arm(name, kind, groups, drop, weight_column, purpose, fit_frames, cal_frames, args):
    started = time.perf_counter()
    if kind == 'network':
        model = NeuralScore.NeuralScore.fit(
            fit_frames, groups=groups, drop=drop, seed=args.fit_seed,
            weight_column=weight_column, hidden=tuple(args.hidden), epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            patience=NETWORK_PARAMS['patience'], log=_log if args.verbose else None,
            threads=args.threads)
    else:
        model = FomCombiner.FomCombiner.fit(
            fit_frames, groups=groups, seed=args.fit_seed, drop=drop,
            weight_column=weight_column, **s12.MODEL_PARAMS)
    # The same asymmetry S12 uses (its `fit_one`): fit on `sampling_weight`, calibrate on
    # `fit_weight`, which undoes the driver's own negative subsampling where it applied.
    model.fit_calibrators(cal_frames,
                          weight_column=None if weight_column is None else 'fit_weight')
    model.meta['arm'] = name
    model.meta['purpose'] = purpose
    model.meta['kind'] = kind
    directory = models_directory(args)/f'{name}_seed{args.fit_seed}'
    model.save(directory)
    elapsed = time.perf_counter() - started
    _log(f'  {name:20s} {kind:8s} {len(model.names):>3d} features  {elapsed:6.0f} s  '
         f'-> {directory.name}')
    row = dict(arm=name, kind=kind, purpose=purpose, groups='+'.join(groups),
               n_features=len(model.names), dropped=';'.join(sorted(drop)),
               fit_seed=int(args.fit_seed), weight_column=weight_column or 'none',
               n_rows_fit=int(model.meta['n_rows']), n_positive_fit=int(model.meta['n_positive']),
               n_calibration_rows=int(model.meta['n_calibration_rows']),
               calibrated_lattices=len(model.meta['calibrated_lattices']),
               seconds=round(elapsed, 1))
    if kind == 'network':
        row.update(input_width=int(model.meta['input_width']),
                   epochs_run=int(model.meta['epochs_run']),
                   best_validation_loss=model.meta['best_validation_loss'],
                   train_auc=float(model.meta['train_auc']),
                   loss_ratio=float(model.meta['loss_check']['ratio']),
                   min_lattices_per_epoch=int(min(r['lattices_present']
                                                  for r in model.meta['composition'])))
    return row


def run_fit(args):
    if args.fit_frame:
        fit_frames, cal_frames = load_frame_files(args)
    else:
        fit_frames, cal_frames = load_slice_frames(args)
    for frame in fit_frames[:1]:
        s12._report_absent_features(frame, union_groups())
    _log(f'assembled: {sum(f.shape[0] for f in fit_frames):,} fit rows '
         f'({sum(int(FomMetrics.as_bool(f["is_correct"]).sum()) for f in fit_frames):,} correct), '
         f'{sum(f.shape[0] for f in cal_frames):,} calibration rows')
    wanted = set(args.arms) if args.arms else None
    rows = []
    for name, kind, groups, drop, weight_column, purpose in ARMS:
        if wanted and name not in wanted:
            continue
        this_fit, this_cal = fit_frames, cal_frames
        if name == 'label_shuffled':
            this_fit = [s12._shuffle_labels(frame, args.fit_seed) for frame in fit_frames]
            this_cal = [s12._shuffle_labels(frame, args.fit_seed) for frame in cal_frames]
        try:
            rows.append(fit_arm(name, kind, groups, drop, weight_column, purpose,
                                this_fit, this_cal, args))
        except (KeyError, ValueError, NeuralScore.CompositionError) as problem:
            _log(f'  {name:20s} SKIPPED -- {problem}')
            rows.append(dict(arm=name, kind=kind, purpose=purpose, skipped=str(problem),
                             fit_seed=int(args.fit_seed)))
    table = pd.DataFrame(rows)
    path = Path(args.artifact_dir)/f'{args.tag}_fit_table{args.suffix}.csv'
    table.to_csv(path, index=False)
    _log(f'\nwrote {path}')
    return 0


# ---------------------------------------------------------------------------------------------
# stage: reduce -- two passes over the report pool, metas merged
# ---------------------------------------------------------------------------------------------
def load_tree_fullscale(args):
    tree = NeuralScore.load_any(Path(args.tree_fullscale))
    n_rows = int(tree.meta.get('n_rows', -1))
    if n_rows != TREE_FULLSCALE_ROWS or tree.meta.get('arm') != 'plus_probation':
        raise SystemExit(
            f'{args.tree_fullscale} is not the full-scale plus_probation model: n_rows {n_rows} '
            f'(expected {TREE_FULLSCALE_ROWS}), arm {tree.meta.get("arm")!r}. C2-F-141 is the '
            f'reason this is asserted rather than assumed.')
    return tree


def load_arms(args):
    directory = models_directory(args)
    arms = {}
    for name in args.arms or ():
        if name == 'tree_fullscale':
            arms[name] = load_tree_fullscale(args)
            continue
        path = directory/f'{name}_seed{args.fit_seed}'
        if not path.exists():
            raise SystemExit(f'no fitted arm at {path}; run --stage fit')
        model = NeuralScore.load_any(path)
        if model.meta.get('arm') != name:
            raise SystemExit(f'{path} records arm {model.meta.get("arm")!r}, not {name!r}')
        arms[name] = model
    return arms


def _merge_meta(path, metas):
    existing = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    existing.update(metas)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str),
                    encoding='utf-8')


def run_reduce(args):
    if not args.arms:
        raise SystemExit('--stage reduce needs --arms (one pass of the two; see the module doc)')
    arms = load_arms(args)
    pool = s12.report_pool(args)
    _log(f'models: {models_directory(args)} -> {", ".join(sorted(arms))} (+ references: '
         f'{"yes" if args.with_references else "no"})')
    for name, model in arms.items():
        _log(f'  {name:20s} {model.meta.get("model_type", "tree"):13s} n_rows '
             f'{int(model.meta.get("n_rows", -1)):,}  features {len(model.names)}')
    report_entries = FomBenchmark.load_entries(pool)
    fit_entries = FomBenchmark.load_entries(FIT_POOL)
    fit_ids, cal_ids = s12.split_ids(fit_entries, args.train_split, s12.HOLDOUT_FRACTION, SEED)
    s12.assert_disjoint(set(fit_ids) | set(cal_ids), set(report_entries['entry_id']))

    # Chunked, for every model type: a whole-frame design matrix is 6-10 GB on a 15 M-row bundle
    # and put the laptop into swap for the length of a pool walk (S14, 2026-09-05).
    scores = {name: (lambda frame, model=model: NeuralScore.chunked_score(model, frame))
              for name, model in arms.items()}
    if args.with_references:
        scores.update(s12.reference_scores(args.fit_seed))
    orientation = {name: True for name in scores}
    groups = s12._union_groups(arms)
    artifact_dir = Path(args.artifact_dir)
    meta_path = artifact_dir/f'{args.tag}_reduced_meta{args.suffix}.json'
    started = time.perf_counter()

    def announce(frame):
        _log(f'  {frame["condition_bundle"].iloc[0]:24s} {frame.shape[0]:>10,} candidates '
             f'({time.perf_counter() - started:.0f} s)')

    _log(f'reducing {len(scores)} scores over {pool} with groups {groups}')
    reduced = FomMetrics.reduce_many(
        FomCombiner.combiner_frames_c2(pool, report_entries, groups=groups),
        scores, entries=report_entries, splits={args.report_split: None},
        higher_is_better=orientation, subsample_top_k=s12._pool_depth(pool), on_shard=announce)
    _merge_meta(meta_path, s12._write_reductions(reduced, artifact_dir, args.tag, args.suffix,
                                                 require_exact=True))

    _log('\nreducing the calibration split for threshold selection (ranks inexact by design)')
    calibration = FomMetrics.reduce_many(
        FomCombiner.combiner_frames_c2(FIT_POOL, fit_entries, groups=groups,
                                       keep_entry_ids=cal_ids),
        scores, entries=fit_entries, splits={args.train_split: None},
        higher_is_better=orientation, subsample_top_k=s12._pool_depth(FIT_POOL),
        allow_inexact_ranks=True)
    _merge_meta(meta_path, s12._write_reductions(calibration, artifact_dir, args.tag,
                                                 args.suffix, require_exact=False, kind='_cal'))
    _log(f'\nreductions -> {artifact_dir} ({meta_path.name} merged)')
    return 0


# ---------------------------------------------------------------------------------------------
# stage: analyse
# ---------------------------------------------------------------------------------------------
REFERENCE_ARM = 'network'
BASELINES = ('tree', 'tree_fullscale', 'M_sym', 'M20')
REFERENCE_ARMS = ('network', 'tree')     # every arm is paired against each of these
ANSWER_RATES = (0.75, 0.90)


def answer_rate_rows(name, per_entry, meta):
    """Operating point and precision at a matched answer rate (C2-F-142): the threshold is set
    so the arm reports on the same fraction of patterns as every other arm. This is the only
    fair threshold comparison between a calibrated probability and a raw merit."""
    scores = per_entry['score_top_in_top_n'].to_numpy(dtype=float)
    row = dict(arm=name, top10=float(FomMetrics.summarise_per_entry(
        per_entry, meta, n_bootstrap=0).metric('top10')))
    for rate in ANSWER_RATES:
        summary = FomMetrics.summarise_per_entry(
            per_entry, meta, threshold=float(np.nanquantile(scores, 1 - rate)), n_bootstrap=0)
        row[f'op_at_{int(rate*100)}'] = float(summary.metric('operating_point'))
        row[f'precision_at_{int(rate*100)}'] = float(summary.metric('precision'))
        row[f'reported_at_{int(rate*100)}'] = float(summary.metric('reported'))
    return row


def run_analyse(args):
    artifact_dir = Path(args.artifact_dir)
    reductions = s12.load_reductions(artifact_dir, args.tag, args.suffix)
    dev = {name: value for (name, split), value in reductions.items()
           if split == args.report_split}
    cal = {name: value for (name, split), value in reductions.items()
           if split.startswith(args.train_split)}
    if not dev or not cal or 'M20' not in cal:
        raise SystemExit('need the report and calibration reductions, including the references; '
                         'run --stage reduce with --with-references on one pass')
    m20_cal = FomMetrics.summarise_per_entry(cal['M20'][0], cal['M20'][1],
                                             threshold=s12.DEWOLFF_THRESHOLD, n_bootstrap=0)
    budget = float(m20_cal.metric('false_positive'))
    _log(f'matched false-positive budget from M20 at {s12.DEWOLFF_THRESHOLD}: {budget:.6f}')

    rows, results, rates = [], {}, []
    for name in sorted(dev):
        if name not in cal:
            _log(f'  {name}: no calibration reduction, skipped')
            continue
        selection = FomMetrics.summarise_per_entry(cal[name][0], cal[name][1], n_bootstrap=0)
        choice, rule = s12.choose_threshold(selection, budget)
        result = FomMetrics.summarise_per_entry(
            dev[name][0], dev[name][1], threshold=float(choice.threshold),
            strata=FomMetrics.DEFAULT_STRATA, n_bootstrap=args.n_bootstrap, seed=SEED)
        FomMetrics.check_threshold_transfer(choice, result)
        results[name] = result
        row = s12.scope_row(result, arm=name, threshold=float(choice.threshold),
                            threshold_rule=rule, fit_seed=args.fit_seed,
                            ranks_exact=bool(dev[name][1]['ranks_exact']))
        row.update(s12.per_lattice(result, 'dev'))
        rows.append(row)
        rates.append(answer_rate_rows(name, dev[name][0], dev[name][1]))
        _log(f'  {name:20s} op {row["operating_point"]:.4f}  top10 {row["top10"]:.4f}  '
             f'threshold_only {row["threshold_only"]:.4f}')

    table = pd.DataFrame(rows).sort_values('operating_point', ascending=False)
    fit_path = artifact_dir/f'{args.tag}_fit_table{args.suffix}.csv'
    if fit_path.exists():
        fit_table = pd.read_csv(fit_path)
        carry = [name for name in ('kind', 'n_features', 'n_rows_fit', 'n_positive_fit',
                                   'weight_column', 'dropped', 'purpose', 'epochs_run',
                                   'train_auc', 'loss_ratio', 'min_lattices_per_epoch')
                 if name in fit_table.columns]
        table = table.merge(fit_table[['arm'] + carry], on='arm', how='left')
    table.to_csv(artifact_dir/f'{args.tag}_main_table{args.suffix}.csv', index=False)
    pd.DataFrame(rates).to_csv(artifact_dir/f'{args.tag}_answer_rates{args.suffix}.csv',
                               index=False)
    write_contrasts(results, artifact_dir, args.tag, args.suffix)
    _log(f'\nwrote {args.tag}_main_table{args.suffix}.csv, contrasts, per-lattice McNemar and '
         f'answer rates to {artifact_dir}')
    return 0


def write_contrasts(results, artifact_dir, tag, suffix):
    """Every arm against `network` (the retrained paired arms), every arm against the baselines
    (the gate), and one paired McNemar per lattice for the arms a per-lattice claim is about."""
    contrasts, headline, by_lattice = [], [], []
    for name, result in sorted(results.items()):
        for metric in ('operating_point', 'top10', 'threshold_only'):
            for scope in (None, 'hard'):
                for reference in REFERENCE_ARMS:
                    if reference in results and name != reference:
                        contrasts.append(s12._pair(results[reference], result, reference,
                                                   name, metric, scope))
                for baseline in BASELINES:
                    if baseline in results and name != baseline \
                            and name not in ('constant', 'uniform_random'):
                        headline.append(s12._pair(results[baseline], result, baseline, name,
                                                  metric, scope))
    for arm in sorted(results):
        if arm in ('constant', 'uniform_random'):
            continue
        for baseline in ('tree', 'tree_fullscale', 'M_sym', REFERENCE_ARM):
            if baseline not in results or arm == baseline:
                continue
            for lattice in FomMetrics.BRAVAIS_LATTICES:
                try:
                    mask = FomMetrics.stratum_mask(results[arm], 'bravais_lattice', lattice)
                except KeyError:
                    continue
                if mask.sum() < 2:
                    continue
                for metric in ('top10', 'operating_point'):
                    row = s12._pair(results[baseline], results[arm], baseline, arm, metric,
                                    None, mask=mask)
                    if row:
                        row['scope'] = f'lattice={lattice}'
                        row['n_entries'] = int(mask.sum())
                        by_lattice.append(row)
    pd.DataFrame([r for r in contrasts if r]).to_csv(
        artifact_dir/f'{tag}_contrasts{suffix}.csv', index=False)
    pd.DataFrame([r for r in headline if r]).to_csv(
        artifact_dir/f'{tag}_mcnemar{suffix}.csv', index=False)
    pd.DataFrame(by_lattice).to_csv(artifact_dir/f'{tag}_by_lattice_mcnemar{suffix}.csv',
                                    index=False)


# ---------------------------------------------------------------------------------------------
# stage: combine -- what survives every seed
# ---------------------------------------------------------------------------------------------
def run_combine(args):
    artifact_dir = Path(args.artifact_dir)
    frames = []
    for suffix in SEED_SUFFIXES:
        for kind in ('contrasts', 'mcnemar'):
            path = artifact_dir/f'{args.tag}_{kind}{suffix}.csv'
            if not path.exists():
                _log(f'  missing {path.name}')
                continue
            frame = pd.read_csv(path)
            frame['seed_suffix'] = suffix or '_seed12345'
            frames.append(frame)
    if not frames:
        raise SystemExit('no per-seed contrast tables')
    everything = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=['reference', 'arm', 'metric', 'scope', 'seed_suffix'])
    rows = []
    for (reference, arm, metric, scope), group in everything.groupby(
            ['reference', 'arm', 'metric', 'scope'], sort=True):
        deltas = group['delta_pp'].to_numpy(dtype=float)
        pvalues = group['p_value'].to_numpy(dtype=float)
        rows.append(dict(
            reference=reference, arm=arm, metric=metric, scope=scope, n_seeds=int(group.shape[0]),
            delta_mean=float(np.mean(deltas)), delta_min=float(np.min(deltas)),
            delta_max=float(np.max(deltas)), p_max=float(np.max(pvalues)),
            same_sign_all_seeds=bool(np.all(deltas > 0) or np.all(deltas < 0)),
            significant_all_seeds=bool(np.all(pvalues < 0.05)),
            settled=bool((np.all(deltas > 0) or np.all(deltas < 0)) and np.all(pvalues < 0.05)
                         and group.shape[0] >= 3)))
    table = pd.DataFrame(rows).sort_values(['reference', 'metric', 'scope', 'delta_mean'],
                                           ascending=False)
    path = artifact_dir/f'{args.tag}_seed_summary.csv'
    table.to_csv(path, index=False)
    n_seeds = everything['seed_suffix'].nunique()
    _log(f'{n_seeds} seed(s) combined -> {path}')
    return 0


# ---------------------------------------------------------------------------------------------
# stage: calibration
# ---------------------------------------------------------------------------------------------
def run_calibration(args):
    arms = load_arms(args)
    if not arms:
        raise SystemExit('--stage calibration needs --arms')
    FomMetrics.check_candidate_statistic(REPORT_POOL, 'a reliability table and its ECE')
    entries = FomBenchmark.load_entries(REPORT_POOL)
    rng = np.random.default_rng(args.fit_seed)
    collected = {name: [] for name in arms}
    for frame in FomCombiner.combiner_frames_c2(REPORT_POOL, entries, groups=s12._union_groups(arms)):
        shard = frame.loc[rng.random(frame.shape[0]) < args.calibration_rate]
        if not shard.shape[0]:
            continue
        label = FomMetrics.as_bool(shard['is_correct'])
        for name, model in arms.items():
            collected[name].append(pd.DataFrame({'score': NeuralScore.chunked_score(model, shard),
                                                 'is_correct': label}))
        _log(f'  {frame["condition_bundle"].iloc[0]:24s} {frame.shape[0]:>10,} candidates, '
             f'{shard.shape[0]:>8,} sampled, {int(label.sum()):>5,} correct')
    rows, tables = [], []
    for name in sorted(collected):
        block = pd.concat(collected[name], ignore_index=True)
        score = block['score'].to_numpy(dtype=np.float64)
        label = block['is_correct'].to_numpy(dtype=bool)
        table, ece, brier = FomMetrics.reliability(score, label, n_bins=args.calibration_bins)
        table.insert(0, 'arm', name)
        tables.append(table)
        rows.append(dict(arm=name, ece=float(ece), brier=float(brier),
                         base_rate=float(label.mean()), mean_score=float(score.mean()),
                         share_at_one=float((score >= 1.0 - 1e-12).mean()),
                         n=int(score.size), n_positive=int(label.sum())))
        _log(f'  {name:20s} ECE {ece:.5f}  Brier {brier:.6f}  share at 1.0 '
             f'{rows[-1]["share_at_one"]:.4f}')
    artifact_dir = Path(args.artifact_dir)
    pd.DataFrame(rows).to_csv(artifact_dir/f'{args.tag}_calibration{args.suffix}.csv', index=False)
    pd.concat(tables, ignore_index=True).to_csv(
        artifact_dir/f'{args.tag}_reliability{args.suffix}.csv', index=False)
    return 0


# ---------------------------------------------------------------------------------------------
# stage: cost
# ---------------------------------------------------------------------------------------------
def run_cost(args):
    """Price block A, block B and the network against `get_M20` on a real low-symmetry block.

    Recorded, not gating (decision 2026-08-25). Block A is priced per ENTRY and amortised over the
    entry's own pool size, because one forward pass serves every candidate of a pattern; block B
    and the network per candidate; the network at the block size and at a realistic batch.
    """
    import pyarrow.parquet as pq
    from mlindex.utilities.FigureOfMerits import get_M20
    from mlindex.utilities.FigureOfMerits import get_assignment_posterior
    from mlindex.utilities.FigureOfMerits import get_assignment_sigma
    from mlindex.model_training import PriorNetwork as Prior

    pool = s12.report_pool(args)
    path = sorted(pool.glob(f'candidates*_{args.cost_lattice}.parquet'))
    if not path:
        raise SystemExit(f'no {args.cost_lattice} candidate file under {pool}')
    frame = pd.DataFrame(
        next(pq.ParquetFile(path[0]).iter_batches(batch_size=args.cost_rows)).to_pydict())
    entries = FomBenchmark.load_entries(pool)
    bundle = frame['condition_bundle'].iloc[0]
    entries = entries.loc[entries['condition_bundle'] == bundle]
    n = frame.shape[0]
    _log(f'{n:,} {args.cost_lattice} candidates from {path[0].name}')

    def timed(label, call, repeats=3, rows=None):
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            call()
            samples.append(time.perf_counter() - started)
        rows = n if rows is None else rows
        return dict(step=label, seconds=float(np.median(samples)), rows=int(rows),
                    microseconds_per_candidate=float(np.median(samples)/rows*1e6))

    blocks = []
    for key, group in frame.groupby(['entry_id', 'lattice_system', 'bravais_lattice',
                                     'spacegroup', 'n_peaks'], sort=False):
        entry_id, lattice_system, bravais_lattice, spacegroup, n_peaks = key
        q2_obs = np.asarray(entries.set_index('entry_id').loc[entry_id, 'q2_obs'],
                            dtype=np.float64)[:int(n_peaks)]
        xnn = np.vstack([np.asarray(value, dtype=np.float64) for value in group['xnn']])
        blocks.append((q2_obs, xnn, lattice_system, bravais_lattice, spacegroup))
    prepared = [FomBenchmark.assign_lines(*block) for block in blocks]

    def m20_only():
        for (q2_obs, *_), (q2_ref_calc, _, _, q2_calc) in zip(blocks, prepared):
            get_M20(q2_obs, q2_calc, q2_ref_calc.copy())

    def posterior_only():
        for (q2_obs, _, lattice_system, *_), (q2_ref_calc, *_) in zip(blocks, prepared):
            sigma, d1 = get_assignment_sigma(q2_obs, q2_ref_calc, lattice_system)
            get_assignment_posterior(q2_obs, q2_ref_calc, lattice_system, sigma=sigma, d1=d1)

    rows = [timed('get_M20 (the unit), reference lines already built', m20_only),
            timed('assign_lines: build the reference lines',
                  lambda: [FomBenchmark.assign_lines(*block) for block in blocks]),
            timed('block B: sigma + 20 per-peak posteriors, reference lines built',
                  posterior_only),
            timed('neural_inputs: the whole per-candidate sidecar pass',
                  lambda: FomBenchmark.neural_inputs(frame, entries, prior_tables=None))]

    prior = Prior.PriorNetwork.load_prior(args.prior_dir)
    entry_q2 = np.stack([np.asarray(v, dtype=np.float64)[:20] for v in entries['q2_obs']])
    n_entries = entry_q2.shape[0]
    pool_sizes = entries['pool_size_full'].to_numpy(dtype=float) if 'pool_size_full' \
        in entries.columns else np.full(n_entries, np.nan)
    per_entry = timed('block A: one forward pass + entry tables, per ENTRY',
                      lambda: prior.entry_tables(entry_q2, batch_size=256), rows=n_entries)
    rows.append(per_entry)
    amortised = dict(per_entry)
    amortised['step'] = 'block A per CANDIDATE, amortised over the entry\'s pool_size_full'
    amortised['rows'] = int(np.nansum(pool_sizes)) if np.isfinite(pool_sizes).any() else n
    amortised['microseconds_per_candidate'] = per_entry['seconds']/amortised['rows']*1e6
    rows.append(amortised)

    arms = load_arms(args)
    assembled = next(FomCombiner.combiner_frames_c2(
        pool, FomBenchmark.load_entries(pool), bundles=[bundle],
        keep_entry_ids=set(frame['entry_id']), groups=s12._union_groups(arms)))
    for name, model in arms.items():
        matrix = model.design_matrix(assembled)
        rows.append(timed(f'{name}: design_matrix', lambda: model.design_matrix(assembled),
                          rows=assembled.shape[0]))
        rows.append(timed(f'{name}: predict_batch at {matrix.shape[0]:,} rows',
                          lambda: model.predict_batch(matrix), rows=matrix.shape[0]))
        rows.append(timed(f'{name}: score (with per-lattice isotonic)',
                          lambda: model.score(assembled), rows=assembled.shape[0]))
        tiles = max(1, int(np.ceil(args.cost_batch/max(matrix.shape[0], 1))))
        large = np.repeat(matrix, tiles, axis=0) if tiles > 1 else matrix
        rows.append(timed(f'{name}: predict_batch at {large.shape[0]:,} rows',
                          lambda: model.predict_batch(large), repeats=2, rows=large.shape[0]))

    table = pd.DataFrame(rows)
    unit = table.loc[0, 'microseconds_per_candidate']
    table['get_M20_units'] = table['microseconds_per_candidate']/unit
    table.insert(0, 'lattice', args.cost_lattice)
    table.insert(0, 'n_candidates', n)
    out = Path(args.artifact_dir)/f'{args.tag}_cost{args.suffix}.csv'
    table.to_csv(out, index=False)
    _log(table[['step', 'microseconds_per_candidate', 'get_M20_units']].to_string(index=False))
    _log(f'\nwrote {out}. Cost decides nothing in campaign 2 (decision 2026-08-25).')
    return 0


# ---------------------------------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='S14: the neural scoring network')
    parser.add_argument('--stage', required=True,
                        choices=('fit', 'reduce', 'analyse', 'combine', 'calibration', 'cost'))
    parser.add_argument('--models-dir', default=str(MODELS_DIR))
    parser.add_argument('--artifact-dir', default=str(ARTIFACT_DIR))
    parser.add_argument('--tag', default=TAG)
    parser.add_argument('--suffix', default='',
                        help='Namespaces the models AND every table this run writes')
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--fit-seed', type=int, default=SEED)
    parser.add_argument('--arms', nargs='*', default=None,
                        help='Arms to fit / reduce / calibrate. Reduce needs it (two passes).')
    parser.add_argument('--with-references', action='store_true',
                        help='Reduce M20, M_sym and the two floors in this pass too')
    parser.add_argument('--fit-frame', default=None,
                        help="Glob over S12's exported `*_fit_frame*.parquet` shards")
    parser.add_argument('--inputs-dir', default=None,
                        help='Directory of keyed S14 input sidecars for those frames')
    parser.add_argument('--allow-partial-bundles', action='store_true')
    parser.add_argument('--report-pool', default=None)
    parser.add_argument('--tree-fullscale', default=str(TREE_FULLSCALE))
    parser.add_argument('--prior-dir', default='mlindex/models/fom_prior/main/global')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--hidden', type=int, nargs='+', default=list(NETWORK_PARAMS['hidden']))
    parser.add_argument('--epochs', type=int, default=NETWORK_PARAMS['epochs'])
    parser.add_argument('--batch-size', type=int, default=NETWORK_PARAMS['batch_size'])
    parser.add_argument('--learning-rate', type=float, default=NETWORK_PARAMS['learning_rate'])
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--calibration-rate', type=float, default=0.1)
    parser.add_argument('--calibration-bins', type=int, default=20)
    parser.add_argument('--cost-lattice', default='mP')
    parser.add_argument('--cost-rows', type=int, default=4000)
    parser.add_argument('--cost-batch', type=int, default=2_000_000)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    stage = {'fit': run_fit, 'reduce': run_reduce, 'analyse': run_analyse,
             'combine': run_combine, 'calibration': run_calibration, 'cost': run_cost}[args.stage]
    return stage(args)


if __name__ == '__main__':
    raise SystemExit(main())
