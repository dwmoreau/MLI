"""S11 block C: does the prior (block A) plus the fit statistic (block B) beat S08's combiner?

PLAN section 3 writes the target as `fit - null + prior`, and Phase 1 measured all three terms.
S07 supplied `- null` exactly and removing it made every merit *worse* (F-076, three times over).
S08 found the gain was the prior and had to infer it from the extinction group. This is the first
model in the project handed both surviving terms explicitly:

    P(candidate correct)  ~  P(cell plausible | peak list)  x  P(peaks fit | cell)
                                    block A                         block B

**The baseline is S08's tuned combiner at 78 features -- 0.6536 operating point, 0.6845 top-10 on
`fom-dev` -- and never raw M20.** `Minfo` is built from the same statistic as `rho`, so against M20
alone any fit-quality column looks like +3.5 pp and against the honest baseline it is worth nothing
(F-130). The architecture and hyperparameters are read from S08's own artefacts rather than
re-tuned, because a paired comparison has to hold everything but the features fixed.

Stages, in the order the handoff specifies:

  * `assignment` -- block B's three columns over the **full** pool. They exist today only for the
    218 038 candidates block B subsampled, and evaluating a combiner on a 40-candidate-per-entry
    pool would be optimistic and unpairable with 0.6536.
  * `prior`      -- block A's readout at each candidate's *claimed* (volume, lattice) pair.
  * `combiner`   -- the four arms and the paired tests. This is Q42.
  * `ablation`   -- which block carries the gain, and what can be deleted (F-093's template).
  * `cost`       -- `get_M20` equivalents, because S14 needs the price.
  * `network`    -- a network over the assembled features, run last.

Bounds that belong on every number this script produces: **R1** (the pool is censored at M20 >= 5),
**R10** (a wrong candidate here is a Gauss-Newton refined wrong cell, not an arbitrary one),
**R11** (the grid has no different error *law*, so robustness to one is untested rather than
passed), **R12** (one instrument, one broadening tag) and **R15** (per-peak truth is
basis-dependent). Block A additionally trained against the repo's own sigma(q^2) model, which is
the leakage path F-008 names and R11 bounds.
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

# Before anything can pull keras in transitively. It defaults to tensorflow, which is not
# installed, so a bare `import keras` fails outright (F-019b).
os.environ.setdefault('KERAS_BACKEND', 'torch')

from mlindex.model_training import FomBenchmark as Bench
from mlindex.utilities import FigureOfMerits as fom

DEFAULT_BENCHMARK = os.path.join('mlindex', 'data', 'fom_benchmark')
DEFAULT_FEATURES = os.path.join('mlindex', 'data', 'fom_features')
DEFAULT_CV = os.path.join('mlindex', 'data', 'fom_cv')
DEFAULT_OUT = os.path.join('mlindex', 'data', 'fom_neural')
DEFAULT_ARTIFACTS = os.path.join('docs', 'fom', 'artifacts')
DEFAULT_NULL = os.path.join('mlindex', 'models', 'fom_null')
DEFAULT_MODELS = os.path.join('mlindex', 'models')
DEFAULT_PRIOR = os.path.join('mlindex', 'models', 'fom_prior', 'main', 'global')
DEFAULT_MANIFEST = os.path.join('docs', 'fom', 'artifacts', 'S02_mirror_manifest.parquet')

BRAVAIS_LATTICES = ('cP', 'cI', 'cF', 'tP', 'tI', 'hP', 'hR', 'oP', 'oC', 'oF', 'oI',
                    'mP', 'mC', 'aP')

# `error0_cont0` is the control and its M20 is arithmetically degenerate (F-054), so the six
# evaluable bundles are what every S06-onward number is computed over.
EVALUABLE_BUNDLES = ('error1_cont0', 'error1_cont0_phase3', 'error1_cont1_drop10',
                     'error1_cont1_drop6', 'error1_cont2', 'error2_cont0')

# `fom-test` is sealed until S15 (PROTOCOL section 3 rule 2) and is never read here.
SPLITS = ('fom-train', 'fom-dev')

KEY = list(Bench.ZOO_KEY_COLUMNS)


# ---------------------------------------------------------------------------------------
# Stage: assignment -- block B's columns over the full pool
# ---------------------------------------------------------------------------------------
def assignment_columns(benchmark_dir, bundle, lattice, splits=SPLITS, models_directory=None):
    """`asg_sigma`, `asg_post_n`, `asg_post_l` for every candidate of one (lattice, bundle).

    One `assign_lines` call per (entry, extinction group), which is the granularity the reference
    line list is shared at, then `get_assignment_sigma` once and the posterior reusing its nearest
    -line distances. Both statistics come out of the single scan that dominates the cost.

    `asg_sigma` is carried as a log: F-131 fitted it that way, it spans orders of magnitude, and a
    tree is invariant to the transform but the network in the last stage is not.
    """
    import pandas as pd

    entries = Bench.load_entries(benchmark_dir)
    entries = entries.loc[
        (entries['condition_bundle'] == bundle) & (entries['split'].isin(splits)),
        ['entry_id', 'q2_obs'],
        ]
    if not entries.shape[0]:
        return pd.DataFrame(columns=KEY + list(_ASSIGNMENT_COLUMNS))
    q2_lookup = dict(zip(entries['entry_id'], entries['q2_obs']))

    candidates = Bench.load_candidates(
        benchmark_dir, bravais_lattices=[lattice], bundles=[bundle],
        columns=['candidate_id', 'entry_id', 'spacegroup', 'xnn', 'n_peaks', 'lattice_system'],
        )
    candidates = candidates.loc[candidates['entry_id'].isin(q2_lookup)]
    if not candidates.shape[0]:
        return pd.DataFrame(columns=KEY + list(_ASSIGNMENT_COLUMNS))
    lattice_system = candidates['lattice_system'].iloc[0]

    blocks = []
    for (entry_id, spacegroup), group in candidates.groupby(['entry_id', 'spacegroup'],
                                                            sort=False):
        n_peaks = int(group['n_peaks'].iloc[0])
        q2_obs = np.asarray(q2_lookup[entry_id], dtype=np.float64)[:n_peaks]
        xnn = np.stack([np.asarray(value, dtype=np.float64) for value in group['xnn']])
        q2_ref_calc, _, _, _ = Bench.assign_lines(
            q2_obs, xnn, lattice_system, lattice, spacegroup, models_directory,
            )
        sigma, d1 = fom.get_assignment_sigma(q2_obs, q2_ref_calc, lattice_system)
        posterior = fom.get_assignment_posterior(
            q2_obs, q2_ref_calc, lattice_system, sigma=sigma, d1=d1,
            )
        blocks.append(pd.DataFrame({
            'entry_id': group['entry_id'].to_numpy(),
            'condition_bundle': bundle,
            'bravais_lattice': lattice,
            'candidate_id': group['candidate_id'].to_numpy(),
            'asg_sigma': np.log(sigma),
            'asg_post_n': posterior.sum(axis=1),
            'asg_post_l': np.log(np.clip(posterior, 1e-12, 1.0)).mean(axis=1),
            }))
    return pd.concat(blocks, ignore_index=True)


_ASSIGNMENT_COLUMNS = ('asg_sigma', 'asg_post_n', 'asg_post_l')


def _assignment_worker(task):
    """Module-level and picklable, because Windows and macOS both spawn (CLAUDE.md)."""
    benchmark_dir, bundle, lattice, splits, models_directory = task
    started = time.perf_counter()
    frame = assignment_columns(benchmark_dir, bundle, lattice, splits, models_directory)
    return bundle, lattice, frame, time.perf_counter() - started


def run_assignment(args):
    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [(args.benchmark_dir, bundle, lattice, tuple(args.splits), args.models_directory)
             for bundle in args.bundles for lattice in args.lattices]
    print(f'{len(tasks)} (bundle, lattice) shards on {args.nproc} processes', flush=True)

    started = time.perf_counter()
    collected, timings = {}, []
    if args.nproc > 1:
        context = multiprocessing.get_context('spawn')
        with context.Pool(args.nproc) as pool:
            for bundle, lattice, frame, seconds in pool.imap_unordered(_assignment_worker, tasks):
                collected.setdefault(bundle, []).append(frame)
                timings.append(dict(bundle=bundle, bravais_lattice=lattice, n_rows=len(frame),
                                    seconds=seconds))
                print(f'  {bundle:22s} {lattice:3s} {len(frame):8,d} rows  {seconds:6.1f}s',
                      flush=True)
    else:
        for task in tasks:
            bundle, lattice, frame, seconds = _assignment_worker(task)
            collected.setdefault(bundle, []).append(frame)
            timings.append(dict(bundle=bundle, bravais_lattice=lattice, n_rows=len(frame),
                                seconds=seconds))
            print(f'  {bundle:22s} {lattice:3s} {len(frame):8,d} rows  {seconds:6.1f}s',
                  flush=True)

    total = 0
    for bundle, frames in sorted(collected.items()):
        frame = pd.concat(frames, ignore_index=True)
        # The join downstream is validate='1:1', so a duplicate key here would raise there rather
        # than silently double a candidate's weight. Assert it where it is cheap to diagnose.
        assert not frame.duplicated(subset=KEY).any(), f'{bundle}: duplicate zoo keys'
        path = os.path.join(args.out_dir, f'assignment_{bundle}.parquet')
        frame.to_parquet(path, index=False)
        total += len(frame)
        print(f'wrote {path}  {len(frame):,} rows', flush=True)

    elapsed = time.perf_counter() - started
    pd.DataFrame(timings).to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_assignment_shards.csv'), index=False,
        encoding='utf-8')
    print(f'{total:,} candidates in {elapsed:.0f}s '
          f'({1e6*elapsed*args.nproc/max(total, 1):.0f} us/candidate/core)', flush=True)
    write_meta(args, 'assignment', dict(
        n_rows=int(total), n_shards=len(tasks), seconds=float(elapsed),
        bundles=list(args.bundles), lattices=list(args.lattices), splits=list(args.splits),
        columns=list(_ASSIGNMENT_COLUMNS),
        ))
    return total


# ---------------------------------------------------------------------------------------
# Stage: prior -- block A read at each candidate's claimed (volume, lattice) pair
# ---------------------------------------------------------------------------------------
_PRIOR_HEADS = ('bravais', 'system', 'centring', 'n_free', 'high_symmetry')


def _entropy(log_probability, axis):
    """Shannon entropy in nats from log probabilities, without exponentiating twice."""
    probability = np.exp(log_probability)
    return -np.sum(probability*log_probability, axis=axis)


def _logsumexp(values, axis):
    peak = values.max(axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.exp(values - peak).sum(axis=axis))


def base_rate_by_lattice(benchmark_dir, bundles, train_split='fom-train'):
    """P(is_correct | claimed Bravais lattice), estimated on `fom-train` and nowhere else.

    Block A is trained class-balanced, which removes the base rate by construction -- and F-086
    measured that the base rate is where S08's gain came from. Handing it back explicitly is the
    handoff's instruction; estimating it anywhere but the training split would be the leak.

    Note S08's feature set already carries `bravais_lattice` as a structural categorical, so a tree
    can recover this for itself. The column earns its place mainly for the network stage, and the
    ablation is what says whether it earned it.
    """
    import pandas as pd

    entries = Bench.load_entries(benchmark_dir)
    keep = set(entries.loc[entries['split'] == train_split, 'entry_id'])
    counts = {}
    for bundle in bundles:
        pool = Bench.load_candidates(
            benchmark_dir, bundles=[bundle], columns=['entry_id', 'bravais_lattice', 'is_correct'],
            )
        pool = pool.loc[pool['entry_id'].isin(keep)]
        for lattice, group in pool.groupby('bravais_lattice', sort=False):
            correct, total = counts.get(lattice, (0, 0))
            counts[lattice] = (correct + int(group['is_correct'].sum()), total + len(group))
    return {lattice: correct/total for lattice, (correct, total) in counts.items() if total}


def prior_columns(model, q2_obs, candidates, class_codes, log_grid):
    """The block A readout for one chunk of entries and every candidate belonging to them.

    `candidates` carries a `row` column indexing into `q2_obs`. Everything is read at the
    candidate's **claimed** pair, never the true one: a candidate claiming a lattice whose volume
    prior it violates should score badly, and that is the entire point of the term.
    """
    import pandas as pd

    tables = model.joint_log_probabilities(q2_obs)
    branch_lp = tables['volume_branch']                     # (n_entries, n_volumes)
    rows = candidates['row'].to_numpy()

    # Nearest volume branch to the claimed volume, on a log grid -- the same match
    # `score_candidates` makes, lifted out because the branch index is itself a feature here.
    claimed = np.log(candidates['volume'].to_numpy(dtype=np.float64))
    branch = np.abs(log_grid[np.newaxis] - claimed[:, np.newaxis]).argmin(axis=1)

    out = {name: candidates[name].to_numpy() for name in KEY}
    joint = tables['bravais']                               # (n_entries, n_volumes, n_classes)
    bravais_index = class_codes['bravais']
    out['prior_joint'] = joint[rows, branch, bravais_index]
    for head in _PRIOR_HEADS[1:]:
        out[f'prior_joint_{head}'] = tables[head][rows, branch, class_codes[head]]

    out['prior_branch_lp'] = branch_lp[rows, branch]
    marginal = _logsumexp(joint, axis=1)                    # (n_entries, n_classes)
    out['prior_bravais_lp'] = marginal[rows, bravais_index]

    # How far the claimed pair sits from the table's own mode. A candidate can have a respectable
    # absolute log-probability simply because the model is uncertain, and the margin says which.
    out['prior_joint_margin'] = out['prior_joint'] - joint.reshape(joint.shape[0], -1).max(axis=1)[rows]

    # Rank of the claimed value in its own distribution, 0 = the model's first choice. An argmax
    # would throw away everything but "is it first"; the rank keeps the ordering and stays finite
    # where a log-probability underflows.
    out['prior_branch_rank'] = (branch_lp[rows] > branch_lp[rows, branch][:, np.newaxis]).sum(axis=1)
    out['prior_bravais_rank'] = (
        marginal[rows] > marginal[rows, bravais_index][:, np.newaxis]
        ).sum(axis=1)

    # Entry-level: constant within an entry, so they cannot reorder candidates inside one. They act
    # on the cross-entry threshold, which is what the operating point measures.
    for position, code in enumerate(BRAVAIS_LATTICES):
        out[f'prior_bravais_p_{code}'] = np.exp(marginal[rows, position])
    out['prior_branch_entropy'] = _entropy(branch_lp, axis=1)[rows]
    out['prior_bravais_entropy'] = _entropy(marginal, axis=1)[rows]
    return pd.DataFrame(out)


def run_prior(args):
    import pandas as pd

    from mlindex.model_training import PriorNetwork as Prior

    os.makedirs(args.out_dir, exist_ok=True)

    # PROTOCOL section 10's first anti-pattern, asserted rather than trusted: block A's training
    # filter was `held_out_identifiers`, and the benchmark's `entry_id` *is* the manifest's
    # `identifier` (verified: 5 922 of 5 955, splits agreeing exactly). So every entry scored on
    # the report split must be one block A was never shown.
    held_out = Prior.held_out_identifiers(args.manifest)
    entries_all = Bench.load_entries(args.benchmark_dir)
    reported = set(entries_all.loc[entries_all['split'] == args.report_split, 'entry_id'])
    leaked = reported - held_out
    if leaked:
        raise SystemExit(
            f'{len(leaked)} {args.report_split} entries are not in block A\'s held-out set '
            f'({sorted(leaked)[:5]}...); block A trained on entries this stage reports on'
            )
    print(f'leakage guard: all {len(reported):,} {args.report_split} entries were held out of '
          'block A', flush=True)

    model = Prior.PriorNetwork.load_prior(args.prior_dir)
    log_grid = np.log(model.branch_volumes())
    print(f'block A loaded from {args.prior_dir}: {log_grid.size} volume branches', flush=True)

    print('estimating the per-lattice base rate on fom-train...', flush=True)
    base_rate = base_rate_by_lattice(args.benchmark_dir, args.bundles, args.train_split)
    print('  ' + '  '.join(f'{k} {v:.4f}' for k, v in sorted(base_rate.items())), flush=True)

    started, total = time.perf_counter(), 0
    for bundle in args.bundles:
        entries = entries_all.loc[
            (entries_all['condition_bundle'] == bundle)
            & (entries_all['split'].isin(args.splits)),
            ['entry_id', 'q2_obs'],
            ].reset_index(drop=True)
        if not entries.shape[0]:
            continue
        position = {entry_id: index for index, entry_id in enumerate(entries['entry_id'])}

        candidates = Bench.load_candidates(
            args.benchmark_dir, bravais_lattices=list(args.lattices), bundles=[bundle],
            columns=['entry_id', 'candidate_id', 'bravais_lattice', 'volume'],
            )
        candidates = candidates.loc[candidates['entry_id'].isin(position)].reset_index(drop=True)
        candidates['row'] = candidates['entry_id'].map(position).to_numpy()

        blocks = []
        for start in range(0, entries.shape[0], args.prior_chunk):
            stop = min(start + args.prior_chunk, entries.shape[0])
            chunk = candidates.loc[candidates['row'].between(start, stop - 1)].copy()
            if not chunk.shape[0]:
                continue
            chunk['row'] = chunk['row'].to_numpy() - start
            q2_obs = np.stack([
                np.asarray(value, dtype=np.float64)
                for value in entries['q2_obs'].iloc[start:stop]
                ])
            codes = Prior.target_codes(chunk['bravais_lattice'].to_numpy())
            blocks.append(prior_columns(model, q2_obs, chunk, codes, log_grid))
            print(f'  {bundle:22s} entries {stop:5d}/{entries.shape[0]}  '
                  f'{sum(len(b) for b in blocks):8,d} candidates', flush=True)

        frame = pd.concat(blocks, ignore_index=True)
        frame['prior_base_rate'] = frame['bravais_lattice'].map(base_rate).astype(float)
        assert not frame.duplicated(subset=KEY).any(), f'{bundle}: duplicate zoo keys'
        path = os.path.join(args.out_dir, f'prior_{bundle}.parquet')
        frame.to_parquet(path, index=False)
        total += len(frame)
        print(f'wrote {path}  {len(frame):,} rows', flush=True)

    elapsed = time.perf_counter() - started
    print(f'{total:,} candidates in {elapsed:.0f}s', flush=True)
    write_meta(args, 'prior', dict(
        n_rows=int(total), seconds=float(elapsed), prior_dir=args.prior_dir,
        n_volume_branches=int(log_grid.size), base_rate={k: float(v) for k, v in base_rate.items()},
        heads=list(_PRIOR_HEADS), bundles=list(args.bundles), splits=list(args.splits),
        held_out_guard=f'all {len(reported)} {args.report_split} entries verified absent from '
                       "block A's training set",
        ))
    return total


# ---------------------------------------------------------------------------------------
# Stage: combiner -- the four arms. This is Q42.
# ---------------------------------------------------------------------------------------
# S08's winning arm plus S10's columns: the 78-feature model at 0.6536 / 0.6845 on `fom-dev`, and
# the only honest baseline. Reported against M20 alone, any fit-quality column looks like +3.5 pp
# and means nothing (F-130).
BASELINE_GROUPS = ('scaled', 'structural', 'context', 'cv')
LOAD_GROUPS = ('raw', 'scaled', 'structural', 'context', 'in_sample', 'cv', 'assignment', 'prior')
ARMS = (
    ('S08 baseline', BASELINE_GROUPS),
    ('+ block B', BASELINE_GROUPS + ('assignment',)),
    ('+ block A', BASELINE_GROUPS + ('prior',)),
    ('+ both', BASELINE_GROUPS + ('assignment', 'prior')),
    )

# Shirley (1980): M20's own reproducibility is ~10%, so a difference smaller than that is not a
# difference, whatever its p-value (F-009, PROTOCOL section 8).
REPRODUCIBILITY_FLOOR = 0.10
# A constant score already reaches this on top-10, because `reduce_pool` breaks ties cubic-first
# and F-069 makes that a good prior (F-083). It goes beside every rank metric.
CONSTANT_TOP10 = 0.2657
# Q32: hard-stratum *threshold* metrics are reported at volume decile >= 6, rank metrics at 8.
HARD_THRESHOLD_DECILE = 6


def _s08(module_name='run_fom_combiner'):
    """S08's script as a module, so its fit/threshold/report path is used rather than copied."""
    import importlib
    scripts = os.path.dirname(os.path.abspath(__file__))
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(module_name)


def s08_settings(artifact_dir):
    """S08's tuned hyperparameters and its matched-FPR budget, from its own artefacts.

    Read rather than re-tuned: a paired comparison has to hold the architecture fixed, or the
    difference measured is the tuning and not the features (`run_fom_cv_combiner`, F-099).
    """
    import pandas as pd

    with open(os.path.join(artifact_dir, 'S08_combiner_meta.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    tuning = pd.read_csv(os.path.join(artifact_dir, 'S08_combiner_tuning.csv'))
    best = tuning.sort_values('average_precision', ascending=False).iloc[0]
    return meta, dict(max_iter=int(best['max_iter']), learning_rate=float(best['learning_rate']),
                      max_leaf_nodes=int(best['max_leaf_nodes']))


def load_frames(args, covariates, scalers, keep_ids, n_negatives=None, seed=12345):
    from mlindex.model_training import FomCombiner

    frames = []
    for frame in FomCombiner.combiner_frames(
            args.benchmark_dir, args.feature_dir, args.bundles, keep_ids, covariates, scalers,
            groups=LOAD_GROUPS, cv_dir=args.cv_dir, assignment_dir=args.out_dir,
            prior_dir=args.out_dir):
        if n_negatives is not None:
            frame = FomCombiner.subsample_negatives(frame, n_negatives, seed)
        frames.append(frame)
    return frames


def coverage_report(frames, label):
    """How much of each new group actually joined. A left join hides its own misses.

    F-121's lesson applied to a merge rather than a batch: check the object, do not trust that the
    step that produced it did what it said. A column that is 100% NaN trains silently and reports
    a delta of exactly zero, which reads like a clean negative and is a plumbing failure.
    """
    import pandas as pd
    from mlindex.model_training import FomCombiner

    rows = []
    total = sum(frame.shape[0] for frame in frames)
    for group, columns in (('assignment', FomCombiner.ASSIGNMENT_MERITS),
                           ('prior', FomCombiner.PRIOR_MERITS)):
        for column in columns:
            present = sum(int(frame[column].notna().sum()) for frame in frames
                          if column in frame.columns)
            rows.append(dict(split=label, group=group, column=column, n_rows=total,
                             n_present=present, coverage=present/max(total, 1)))
    table = pd.DataFrame(rows)
    worst = table.sort_values('coverage').iloc[0]
    print(f'  {label}: {total:,} rows; worst join coverage '
          f'{worst["coverage"]:.4f} on {worst["column"]}', flush=True)
    if worst['coverage'] < 0.5:
        raise SystemExit(
            f'{worst["column"]} joined onto {worst["coverage"]:.1%} of {label} rows -- the '
            'feature matrix is not covering the pool; regenerate before fitting anything'
            )
    return table


def hard_masks(index):
    """The hard stratum at both cuts, and each restricted to entries with a reachable solution.

    F-059 measured the designated stratum at 87% *generation* failure, so the unconditional hard
    operating point is mostly a statement about the candidate generator rather than about the
    figure of merit. `operating_point_given_found` is the number Q30 chose as its headline, and
    the paired version of it is McNemar over the same metric restricted to the entries where a
    correct candidate exists at all.
    """
    import pandas as pd

    from mlindex.model_training import FomMetrics

    lattices = index['bravais_lattice'].isin(FomMetrics.HARD_LATTICES)
    # `condition_bundle` is half the index, not a column, so it is read off the level rather than
    # subscripted -- which raises rather than returning something wrong, but only at run time.
    bundle_level = pd.Index(index.index.get_level_values('condition_bundle'))
    bundles = pd.Series(bundle_level.isin(FomMetrics.HARD_BUNDLES), index=index.index)
    found = index['found'].to_numpy(dtype=bool)
    masks = {}
    for decile in (FomMetrics.HARD_MIN_DECILE, HARD_THRESHOLD_DECILE):
        base = (lattices & bundles & (index['volume_decile'] >= decile)).to_numpy(dtype=bool)
        masks[f'hard_d{decile}'] = base
        masks[f'hard_d{decile}_found'] = base & found
    return masks


def paired_tables(results, baseline_name, metrics=('operating_point', 'top10')):
    """McNemar against the baseline, aggregate and per Bravais lattice.

    Per-lattice is not optional here. F-084 is the named failure mode -- a model that has learned
    "triclinic candidates are usually wrong" posts a good aggregate and is useless -- and F-087
    found `mcnemar`'s subset argument raised on every call before S08, so no per-stratum paired
    test in this project predates that fix.
    """
    import pandas as pd
    from mlindex.model_training import FomMetrics

    baseline = results[baseline_name]
    aggregate, by_lattice = [], []
    for name, result in results.items():
        if name == baseline_name:
            continue
        index = result.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index()
        for metric in metrics:
            # `mcnemar` returns its own `metric`, `subset` and `n_entries`, so the identifiers are
            # merged *under* it rather than passed alongside -- naming one of them again is a
            # TypeError, and it is a TypeError raised after four models have been fitted.
            test = FomMetrics.mcnemar(result, baseline, metric=metric)
            aggregate.append({'arm': name, 'baseline': baseline_name, **test.to_dict()})
            for lattice in FomMetrics.BRAVAIS_LATTICES:
                mask = (index['bravais_lattice'] == lattice).to_numpy(dtype=bool)
                if mask.sum() < 30:
                    continue
                test = FomMetrics.mcnemar(result, baseline, metric=metric, subset=mask)
                by_lattice.append({'arm': name, 'baseline': baseline_name,
                                   'bravais_lattice': lattice, **test.to_dict()})
            # PROTOCOL section 3 rule 6: the hard stratum carries the claim, so it gets its own
            # paired test rather than only the unpaired summary `scope_row` already carries.
            # Both cuts, because Q32 settled that threshold metrics are reported at decile >= 6
            # (468 dev rows over 117 source entries) while rank metrics stay at the literal 8 --
            # and `mcnemar(subset='hard')` only knows the literal one.
            for label, mask in hard_masks(index).items():
                if mask.sum() < 30:
                    continue
                test = FomMetrics.mcnemar(result, baseline, metric=metric, subset=mask)
                by_lattice.append({'arm': name, 'baseline': baseline_name,
                                   'bravais_lattice': label, **test.to_dict()})
    # `n_a_only` is entries the arm wins and the baseline loses; `n_b_only` is the reverse. Named
    # here so no reader has to work out which way round the arguments went.
    for table in (aggregate, by_lattice):
        for row in table:
            row['n_gained'], row['n_lost'] = row['n_a_only'], row['n_b_only']
    return pd.DataFrame(aggregate), pd.DataFrame(by_lattice)


def calibration_table(model, dev_frames, arm):
    import pandas as pd
    from mlindex.model_training import FomMetrics

    rows = pd.concat([pd.DataFrame({
        'probability': model.score(frame),
        'is_correct': FomMetrics.as_bool(frame['is_correct']),
        'bravais_lattice': frame['bravais_lattice'].to_numpy(),
        }) for frame in dev_frames], ignore_index=True)
    table, ece, brier = FomMetrics.reliability(rows['probability'].to_numpy(),
                                               rows['is_correct'].to_numpy())
    table['scope'], table['ece'], table['brier'], table['arm'] = 'aggregate', ece, brier, arm
    blocks = [table]
    for lattice, block in rows.groupby('bravais_lattice'):
        if block.shape[0] < 200:
            continue
        sub, ece_bl, brier_bl = FomMetrics.reliability(
            block['probability'].to_numpy(), block['is_correct'].to_numpy())
        sub['scope'], sub['ece'], sub['brier'], sub['arm'] = str(lattice), ece_bl, brier_bl, arm
        blocks.append(sub)
    return pd.concat(blocks, ignore_index=True), float(ece), float(brier)


def run_combiner(args):
    import pandas as pd
    from mlindex.model_training import FomCombiner

    s08 = _s08()
    meta, params = s08_settings(args.artifact_dir)
    budget = float(meta['matched_fpr_budget'])
    n_negatives = int(meta.get('n_negatives', 40))
    holdout_fraction = float(meta.get('holdout_fraction', 0.2))
    args.objective = 'pointwise'
    args.model_params = params
    print(f"S08's architecture, read from its artefacts: {params}, matched-FPR budget {budget}",
          flush=True)

    entries = Bench.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = s08.split_ids(entries, args.train_split, holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    print('loading frames...', flush=True)
    started = time.perf_counter()
    fit_frames = load_frames(args, covariates, scalers, fit_ids, n_negatives, args.seed)
    cal_frames = load_frames(args, covariates, scalers, cal_ids)
    dev_frames = load_frames(args, covariates, scalers, dev_ids)
    print(f'  {time.perf_counter() - started:.0f}s; '
          f'{sum(f.shape[0] for f in fit_frames):,} fit / '
          f'{sum(f.shape[0] for f in cal_frames):,} calibration / '
          f'{sum(f.shape[0] for f in dev_frames):,} dev candidates', flush=True)
    coverage = pd.concat([coverage_report(fit_frames, 'fit'),
                          coverage_report(dev_frames, 'dev')], ignore_index=True)
    coverage.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_coverage.csv'), index=False,
                    encoding='utf-8')

    rows, results, models, calibrations = [], {}, {}, []
    for name, groups in ARMS:
        combiner, dev, row = s08.fit_arm(
            name, groups, fit_frames, cal_frames, dev_frames, entries, scalers, budget, args,
            )
        table, ece, brier = calibration_table(combiner, dev_frames, name)
        row['ece'], row['brier'] = ece, brier
        calibrations.append(table)
        rows.append(row)
        results[name] = dev
        models[name] = combiner
        print(f'  {name:14s} op {row["operating_point"]:.4f}  top10 {row["top10"]:.4f}  '
              f'ECE {ece:.4f}  {row["n_features"]} features', flush=True)

    table = pd.DataFrame(rows)
    table['delta_operating_point'] = table['operating_point'] - table['operating_point'].iloc[0]
    table['delta_top10'] = table['top10'] - table['top10'].iloc[0]
    table['constant_top10_floor'] = CONSTANT_TOP10
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_main_table.csv'), index=False,
                 encoding='utf-8')
    pd.concat(calibrations, ignore_index=True).to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_calibration.csv'), index=False,
        encoding='utf-8')

    best = table.sort_values('operating_point', ascending=False).iloc[0]['arm']
    models[best].save(os.path.join(args.models_dir, 'fom_combiner_blockc'))
    print(f'saved the best arm ({best}) to fom_combiner_blockc', flush=True)

    # Written before the paired step on purpose: the arms cost fifteen minutes and the paired
    # step is cheap, so nothing downstream of them should be able to discard them.
    aggregate, by_lattice = paired_tables(results, 'S08 baseline')
    aggregate.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_mcnemar.csv'), index=False,
                     encoding='utf-8')
    by_lattice.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_mcnemar_by_lattice.csv'),
                      index=False, encoding='utf-8')
    print('\npaired against the S08 baseline:')
    print(aggregate[['arm', 'metric', 'n_gained', 'n_lost', 'p_value']].to_string(index=False))

    write_meta(args, 'combiner', dict(
        arms={name: list(groups) for name, groups in ARMS}, params=params,
        matched_fpr_budget=budget, n_negatives=n_negatives, holdout_fraction=holdout_fraction,
        best_arm=str(best), reproducibility_floor=REPRODUCIBILITY_FLOOR,
        constant_top10=CONSTANT_TOP10,
        n_fit=int(sum(f.shape[0] for f in fit_frames)),
        n_dev=int(sum(f.shape[0] for f in dev_frames)),
        baseline_reference=dict(operating_point=0.6536, top10=0.6845,
                                source='S08 session 2 with S10 columns (F-099)'),
        ))
    return table


# ---------------------------------------------------------------------------------------
# Stage: ablation -- which block carries the gain, and what can be deleted
# ---------------------------------------------------------------------------------------
def _drop(names, removed):
    removed = set(removed)
    return tuple(name for name in names if name not in removed)


def ablation_sets(full_names):
    """(label, names, question) for each arm. The direction is F-093's: what can be *dropped*.

    F-093 is the template and the reason: it found that dropping the whole over-prediction family
    cost 0.28 pp at p = 0.85 while saving 57% of the feature budget, and that was the most useful
    thing S14 inherited. Asking "what helps" produces a longer feature list; asking "what can go"
    produces a deployable one.
    """
    from mlindex.model_training import FomCombiner as C

    return (
        ('full', full_names, 'both blocks, every column'),
        ('drop block B', _drop(full_names, C.ASSIGNMENT_MERITS), 'is the fit statistic load-bearing'),
        ('drop block A', _drop(full_names, C.PRIOR_MERITS), 'is the prior load-bearing'),
        ('drop both (= S08)', _drop(full_names, C.ASSIGNMENT_MERITS + C.PRIOR_MERITS),
         'the baseline, refitted inside this harness'),
        ('drop posterior', _drop(full_names, ('asg_post_n', 'asg_post_l')),
         "F-131's question: does the posterior earn its place beside sigma"),
        ('drop sigma', _drop(full_names, ('asg_sigma',)), 'or is sigma the whole of block B'),
        ('drop prior entry-level', _drop(full_names, C.PRIOR_ENTRY),
         'do the constant-within-entry columns pay, or only the claimed-pair ones'),
        ('drop prior claimed-pair', _drop(full_names, C.PRIOR_CLAIMED),
         'the converse'),
        ('drop base rate', _drop(full_names, ('prior_base_rate',)),
         'F-086: is the explicit base rate worth a column a tree can already infer'),
        )


def report_model(combiner, cal_frames, dev_frames, entries, budget, args, s08, name, note=''):
    """Calibrate, choose a threshold on the calibration split, report on dev. `fit_arm`'s tail."""
    from mlindex.model_training import FomMetrics

    combiner.fit_calibrators(cal_frames)
    columns = combiner.score_columns
    cal = s08.evaluate_score(cal_frames, entries, combiner.score, None, args.train_split, columns)
    choice, rule = s08.choose_threshold(cal, budget)
    dev = s08.evaluate_score(dev_frames, entries, combiner.score, float(choice.threshold),
                             args.report_split, columns, n_bootstrap=args.n_bootstrap,
                             seed=args.seed)
    FomMetrics.check_threshold_transfer(choice, dev)
    row = s08.scope_row(dev, arm=name, question=note, n_features=len(combiner.names),
                        threshold=float(choice.threshold), threshold_rule=rule)
    row.update(s08.per_lattice(dev, 'dev'))
    return dev, row


def run_ablation(args):
    import pandas as pd
    from mlindex.model_training import FomCombiner

    s08 = _s08()
    meta, params = s08_settings(args.artifact_dir)
    budget = float(meta['matched_fpr_budget'])
    n_negatives = int(meta.get('n_negatives', 40))
    holdout_fraction = float(meta.get('holdout_fraction', 0.2))
    args.objective = 'pointwise'
    args.model_params = params

    entries = Bench.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = s08.split_ids(entries, args.train_split, holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    print('loading frames...', flush=True)
    fit_frames = load_frames(args, covariates, scalers, fit_ids, n_negatives, args.seed)
    cal_frames = load_frames(args, covariates, scalers, cal_ids)
    dev_frames = load_frames(args, covariates, scalers, dev_ids)

    # The full model is fitted once and every ablation is a refit of the same class on a subset of
    # its own columns (`_restrict`), so each arm costs one fit rather than two.
    full_groups = BASELINE_GROUPS + ('assignment', 'prior')
    full = FomCombiner.FomCombiner.fit(
        fit_frames, groups=full_groups, scalers=scalers, objective='pointwise', seed=args.seed,
        **params)
    print(f'full model: {len(full.names)} features', flush=True)

    rows, results = [], {}
    for label, names, note in ablation_sets(tuple(full.names)):
        model = full if label == 'full' else s08._restrict(full, list(names), fit_frames, args)
        dev, row = report_model(model, cal_frames, dev_frames, entries, budget, args, s08,
                                label, note)
        rows.append(row)
        results[label] = dev
        print(f'  {label:24s} {len(names):4d} features  op {row["operating_point"]:.4f}  '
              f'top10 {row["top10"]:.4f}', flush=True)

    table = pd.DataFrame(rows)
    table['delta_operating_point'] = table['operating_point'] - table['operating_point'].iloc[0]
    table['delta_top10'] = table['top10'] - table['top10'].iloc[0]
    table['constant_top10_floor'] = CONSTANT_TOP10
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_ablation.csv'), index=False,
                 encoding='utf-8')

    aggregate, by_lattice = paired_tables(results, 'full')
    aggregate.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_ablation_mcnemar.csv'),
                     index=False, encoding='utf-8')
    by_lattice.to_csv(os.path.join(args.artifact_dir,
                                   f'{args.tag}_ablation_mcnemar_by_lattice.csv'),
                      index=False, encoding='utf-8')

    importance = permutation_importance(full, dev_frames, args.seed)
    importance.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_importance.csv'), index=False,
                      encoding='utf-8')
    print('\nnew columns by permutation importance (rank of ' + str(len(full.names)) + '):')
    print(importance.loc[importance['group'] != 'S08'].to_string(index=False))

    write_meta(args, 'ablation', dict(
        n_features_full=len(full.names), params=params, matched_fpr_budget=budget,
        arms=[label for label, _, _ in ablation_sets(tuple(full.names))],
        ))
    return table


def permutation_importance(combiner, dev_frames, seed, n_repeats=1, sample=30000):
    """Permutation importance on average precision, on the dev split.

    Deliberately small, for the reason `run_fom_cv_combiner._importance` records: a 600-iteration
    tree costs ~4.5 us per row per call, so a full-size pass over a hundred features is hours. This
    orders the features, which is the only claim made from it; the magnitudes are not quoted.
    """
    import pandas as pd
    from mlindex.model_training import FomCombiner, FomMetrics

    frame = pd.concat(dev_frames, ignore_index=True)
    if frame.shape[0] > sample:
        frame = frame.sample(sample, random_state=seed)
    matrix = combiner.design_matrix(frame)
    target = FomMetrics.as_bool(frame['is_correct'])
    base = FomMetrics.average_precision(combiner.predict_batch(matrix), target)
    rng = np.random.default_rng(seed)
    assignment, prior = set(FomCombiner.ASSIGNMENT_MERITS), set(FomCombiner.PRIOR_MERITS)
    rows = []
    for position, name in enumerate(combiner.names):
        drops = []
        for _ in range(n_repeats):
            shuffled = matrix.copy()
            shuffled[:, position] = shuffled[rng.permutation(shuffled.shape[0]), position]
            drops.append(base - FomMetrics.average_precision(
                combiner.predict_batch(shuffled), target))
        group = 'block B' if name in assignment else 'block A' if name in prior else 'S08'
        rows.append(dict(feature=name, group=group, importance=float(np.mean(drops))))
    table = pd.DataFrame(rows).sort_values('importance', ascending=False).reset_index(drop=True)
    table['rank'] = table.index + 1
    return table


# ---------------------------------------------------------------------------------------
# Stage: cost -- what S14 would pay, in get_M20 equivalents
# ---------------------------------------------------------------------------------------
def run_cost(args):
    """Seconds per candidate against `get_M20`, in the `S06_zoo_cost.csv` format.

    Gate condition 3 asks for a variant inside 2x `get_M20`, and it has failed on the *features*
    every time it has been measured -- F-085 priced the ten affordable merits at 4.68x and all
    seventeen at 145x, while F-092 put the model itself at 0.17x. So a new column's price is what
    decides whether it can be deployed at all, and S14 inherits this table.

    Two prices for block B, because they answer different questions. `assign_lines` rebuilds the
    reference line list, which the pipeline **already has** wherever it computes M20 at all; that
    is the protocol's own accounting, which times `get_M20` with `q2_ref_calc` in hand. So the
    marginal row is what an inner loop pays and the standalone row is what a caller starting from
    nothing pays. Reporting only one of them would flatter or damn the feature by a factor of ten.

    Block A is priced per *candidate* even though it runs per *peak list*: one forward pass serves
    every candidate of an entry, so its price is the pass divided by the pool size, and a pool of
    one would pay all of it. Both are reported.
    """
    import pandas as pd
    from mlindex.utilities import FigureOfMerits as fom

    bundle = args.bundles[0]
    candidates = Bench.load_candidates(
        args.benchmark_dir, bundles=[bundle],
        columns=['entry_id', 'candidate_id', 'xnn', 'spacegroup', 'lattice_system',
                 'bravais_lattice', 'n_peaks'],
        )
    entry_table = Bench.load_entries(args.benchmark_dir)
    entry_table = entry_table.loc[entry_table['condition_bundle'] == bundle]
    peaks = entry_table.set_index('entry_id')['q2_obs']

    chosen = pd.unique(candidates['entry_id'])[:args.cost_entries]
    cases, pool_sizes, q2_lists = [], [], []
    for entry_id in chosen:
        group = candidates.loc[candidates['entry_id'] == entry_id]
        pool_sizes.append(len(group))
        q2_lists.append(np.asarray(peaks.loc[entry_id], dtype=np.float64))
        for keys, chunk in group.groupby(['lattice_system', 'bravais_lattice', 'spacegroup',
                                          'n_peaks'], sort=False):
            lattice_system, bravais_lattice, spacegroup, n_peaks = keys
            q2_obs = np.asarray(peaks.loc[entry_id], dtype=np.float64)[:int(n_peaks)]
            xnn = np.vstack([np.asarray(value, dtype=np.float64) for value in chunk['xnn']])
            q2_ref_calc, _, _, q2_calc = Bench.assign_lines(
                q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, args.models_directory,
                )
            cases.append((q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, q2_calc,
                          q2_ref_calc))
    n_candidates = sum(case[1].shape[0] for case in cases)
    mean_pool = float(np.mean(pool_sizes))

    def _assignment(case):
        sigma, d1 = fom.get_assignment_sigma(case[0], case[6], case[2])
        return fom.get_assignment_posterior(case[0], case[6], case[2], sigma=sigma, d1=d1)

    def _assignment_standalone(case):
        q2_ref_calc, _, _, _ = Bench.assign_lines(
            case[0], case[1], case[2], case[3], case[4], args.models_directory,
            )
        sigma, d1 = fom.get_assignment_sigma(case[0], q2_ref_calc, case[2])
        return fom.get_assignment_posterior(case[0], q2_ref_calc, case[2], sigma=sigma, d1=d1)

    calls = {
        'M20': lambda case: fom.get_M20(case[0], case[5], case[6].copy()),
        'asg_* given the reference lines': _assignment,
        'asg_* including assign_lines': _assignment_standalone,
        }

    rows, baseline = [], None
    for name, call in calls.items():
        best = np.inf
        for _ in range(args.cost_repeats + 1):    # the first pass warms the numba kernels
            start = time.perf_counter()
            for case in cases:
                call(case)
            best = min(best, time.perf_counter() - start)
        per_candidate = best/max(n_candidates, 1)
        if name == 'M20':
            baseline = per_candidate
        rows.append(dict(feature=name, block='block B' if name != 'M20' else '-',
                         seconds_per_candidate=per_candidate, n_timed=n_candidates))

    # Block A: one forward pass per peak list, amortised over the entry's pool.
    from mlindex.model_training import PriorNetwork as Prior

    model = Prior.PriorNetwork.load_prior(args.prior_dir)
    q2_batch = np.stack(q2_lists)
    model.joint_log_probabilities(q2_batch, batch_size=args.prior_batch)   # warm
    best = np.inf
    for _ in range(args.cost_repeats):
        start = time.perf_counter()
        model.joint_log_probabilities(q2_batch, batch_size=args.prior_batch)
        best = min(best, time.perf_counter() - start)
    per_entry = best/len(q2_lists)
    rows.append(dict(feature='prior_* per peak list (all five heads)', block='block A',
                     seconds_per_candidate=per_entry, n_timed=len(q2_lists)))
    rows.append(dict(feature=f'prior_* per candidate (pool of {mean_pool:.0f})', block='block A',
                     seconds_per_candidate=per_entry/mean_pool, n_timed=len(q2_lists)))

    table = pd.DataFrame(rows)
    table['cost_vs_M20'] = table['seconds_per_candidate']/baseline
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_cost.csv'), index=False,
                 encoding='utf-8')
    print(table.to_string(index=False))
    write_meta(args, 'cost', dict(
        bundle=bundle, n_candidates_timed=int(n_candidates), n_entries_timed=len(q2_lists),
        mean_pool_size=mean_pool, prior_batch=args.prior_batch, repeats=args.cost_repeats,
        note='first pass discarded so the numba JIT and the torch graph are warm; get_M20 is '
             'timed with the reference lines already built, which is the S06_zoo_cost convention',
        ))
    return table


# ---------------------------------------------------------------------------------------
# Stage: network -- the last rung, run after everything else is reported
# ---------------------------------------------------------------------------------------
def residual_structure(tree, fit_frames, dev_frames, args):
    """Is there anything left in the tree's residual that the same features can predict?

    PLAN section 4 would have made this the gate on building the network at all; DWMM's instruction
    is to build it either way, so the measurement is made and reported rather than used as a gate.
    It is still the number that says *why* the network wins or does not.

    Fitted on the fit split's residual `y - p_tree` and scored on dev, so a positive R^2 means the
    structure generalises rather than that a second model memorised the first one's errors. A tree
    fitted to its own residual on its own training rows would report a large R^2 and mean nothing.
    """
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    from mlindex.model_training import FomMetrics

    fit_frame = pd.concat(fit_frames, ignore_index=True)
    dev_frame = pd.concat(dev_frames, ignore_index=True)
    if dev_frame.shape[0] > 400000:
        dev_frame = dev_frame.sample(400000, random_state=args.seed)

    fit_matrix, dev_matrix = tree.design_matrix(fit_frame), tree.design_matrix(dev_frame)
    fit_residual = (FomMetrics.as_bool(fit_frame['is_correct']).astype(float)
                    - tree.predict_batch(fit_matrix))
    dev_residual = (FomMetrics.as_bool(dev_frame['is_correct']).astype(float)
                    - tree.predict_batch(dev_matrix))

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                          random_state=args.seed)
    model.fit(fit_matrix, fit_residual)
    predicted = model.predict(dev_matrix)
    total = float(np.sum((dev_residual - dev_residual.mean())**2))
    explained = 1.0 - float(np.sum((dev_residual - predicted)**2))/max(total, 1e-30)
    return dict(r2_on_dev=explained,
                residual_mean=float(dev_residual.mean()),
                residual_std=float(dev_residual.std()),
                correlation=float(np.corrcoef(dev_residual, predicted)[0, 1]),
                n_dev_rows=int(dev_frame.shape[0]))


def run_network(args):
    import pandas as pd
    from mlindex.model_training import FomCombiner
    from mlindex.model_training import NeuralFom as Neural

    s08 = _s08()
    meta, params = s08_settings(args.artifact_dir)
    budget = float(meta['matched_fpr_budget'])
    n_negatives = int(meta.get('n_negatives', 40))
    holdout_fraction = float(meta.get('holdout_fraction', 0.2))
    args.objective = 'pointwise'
    args.model_params = params

    entries = Bench.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    fit_ids, cal_ids = s08.split_ids(entries, args.train_split, holdout_fraction, args.seed)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    print('loading frames...', flush=True)
    fit_frames = load_frames(args, covariates, scalers, fit_ids, n_negatives, args.seed)
    cal_frames = load_frames(args, covariates, scalers, cal_ids)
    dev_frames = load_frames(args, covariates, scalers, dev_ids)
    positive_rate = float(np.mean(np.concatenate(
        [frame['is_correct'].to_numpy(dtype=bool) for frame in fit_frames])))
    print(f'  fit positive rate {positive_rate:.5f}', flush=True)

    scaled_groups = BASELINE_GROUPS + ('assignment', 'prior')
    # F-081 says to re-measure rather than assume: `z` and `rank` are monotone within a lattice, so
    # a tree cannot tell scaled from raw, and a network can.
    raw_groups = ('raw', 'structural', 'context', 'cv', 'assignment', 'prior')

    rows, results, models = [], {}, {}
    tree = FomCombiner.FomCombiner.fit(
        fit_frames, groups=scaled_groups, scalers=scalers, objective='pointwise', seed=args.seed,
        **params)
    dev, row = report_model(tree, cal_frames, dev_frames, entries, budget, args, s08,
                            'tree (+ both)', 'the arm the network has to beat')
    rows.append(row)
    results['tree (+ both)'] = dev
    print(f'  {"tree (+ both)":24s} op {row["operating_point"]:.4f}  '
          f'top10 {row["top10"]:.4f}', flush=True)

    print('measuring residual structure...', flush=True)
    residual = residual_structure(tree, fit_frames, dev_frames, args)
    print(f'  residual R^2 on dev {residual["r2_on_dev"]:+.5f} '
          f'(correlation {residual["correlation"]:+.4f})', flush=True)

    for label, groups in (('network (scaled)', scaled_groups), ('network (raw)', raw_groups)):
        network = Neural.NeuralFom.fit(
            fit_frames, groups=groups, scalers=scalers, seed=args.seed,
            hidden=tuple(args.hidden), max_iter=args.epochs,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            expected_positive_rate=positive_rate,
            )
        dev, row = report_model(network, cal_frames, dev_frames, entries, budget, args, s08,
                                label, f'{len(network.names)} features, hidden {args.hidden}')
        row['n_iterations'] = network.meta['n_iterations']
        row['reported_loss'] = network.meta['loss_check']['reported_loss']
        row['observed_loss'] = network.meta['loss_check']['observed_loss']
        rows.append(row)
        results[label] = dev
        models[label] = network
        print(f'  {label:24s} op {row["operating_point"]:.4f}  top10 {row["top10"]:.4f}  '
              f'({network.meta["n_iterations"]} iterations, loss '
              f'{row["reported_loss"]:.5f} reported / {row["observed_loss"]:.5f} observed)',
              flush=True)

    table = pd.DataFrame(rows)
    table['delta_operating_point'] = table['operating_point'] - table['operating_point'].iloc[0]
    table['delta_top10'] = table['top10'] - table['top10'].iloc[0]
    table['constant_top10_floor'] = CONSTANT_TOP10
    table['residual_r2'] = residual['r2_on_dev']
    table.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_network.csv'), index=False,
                 encoding='utf-8')

    aggregate, by_lattice = paired_tables(results, 'tree (+ both)')
    aggregate.to_csv(os.path.join(args.artifact_dir, f'{args.tag}_network_mcnemar.csv'),
                     index=False, encoding='utf-8')
    by_lattice.to_csv(os.path.join(args.artifact_dir,
                                   f'{args.tag}_network_mcnemar_by_lattice.csv'),
                      index=False, encoding='utf-8')
    print('\npaired against the tree on the same features:')
    print(aggregate[['arm', 'metric', 'n_gained', 'n_lost', 'p_value']].to_string(index=False))

    best = table.sort_values('operating_point', ascending=False).iloc[0]['arm']
    if best in models:
        models[best].save(os.path.join(args.models_dir, 'fom_neural_blockc'))
    write_meta(args, 'network', dict(
        residual_structure=residual, hidden=list(args.hidden), epochs=args.epochs,
        learning_rate=args.learning_rate, batch_size=args.batch_size, best_arm=str(best),
        scaled_groups=list(scaled_groups), raw_groups=list(raw_groups),
        note='the network is run regardless of the residual measurement (DWMM, 2026-08-20); the '
             'measurement is reported because it says why the result came out as it did',
        ))
    return table


# ---------------------------------------------------------------------------------------
# Stage: diagnose -- where the remaining loss actually is
# ---------------------------------------------------------------------------------------
def run_diagnose(args):
    """Why block C gained so little, asked of the pool rather than of the model.

    The gate compared a gain in percentage points against a floor in percentage points, and
    METRICS.md section 3 says improvements are quoted as a **fraction of headroom** instead
    (F-042). The two disagree here, so this stage measures the headroom and then decomposes what
    is left of it:

      * `ceiling_rescorer` is the operating point a **perfect** score reaches -- a correct
        candidate exists somewhere in the pool. Everything above it is a *generation* failure and
        belongs to S14/S15, not to any figure of merit (METRICS.md section 3).
      * of the reachable entries that are still lost, `share_rank_failure` /
        `share_threshold_failure` / `share_both` say which half of the criterion failed. A rank
        failure cannot be fixed by a better-calibrated score and a threshold failure cannot be
        fixed by a better ranking, so the two point at different work.
      * and for block A specifically: the prior can only pay where it is right, so its accuracy on
        these entries bounds what it could ever contribute.
    """
    import pandas as pd
    from mlindex.model_training import FomCombiner
    from mlindex.model_training import FomMetrics

    s08 = _s08()
    meta, params = s08_settings(args.artifact_dir)
    holdout_fraction = float(meta.get('holdout_fraction', 0.2))
    args.objective, args.model_params = 'pointwise', params

    entries = Bench.load_entries(args.benchmark_dir)
    covariates = FomCombiner.entry_covariates(entries)
    scalers = FomCombiner.load_scalers(args.null_dir, groups=LOAD_GROUPS)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    print('loading frames...', flush=True)
    dev_frames = load_frames(args, covariates, scalers, dev_ids)

    combiner = FomCombiner.FomCombiner.load(
        os.path.join(args.models_dir, 'fom_combiner_blockc'))
    table = pd.read_csv(os.path.join(args.artifact_dir, f'{args.tag}_main_table.csv'))
    threshold = float(table.loc[table['arm'] == '+ both', 'threshold'].iloc[0])
    print(f'block C loaded: {len(combiner.names)} features, threshold {threshold:.6f}', flush=True)

    result = s08.evaluate_score(
        dev_frames, entries, combiner.score, threshold, args.report_split,
        combiner.score_columns, n_bootstrap=0,
        )

    # ---- 1. the headroom, and what fraction of it was taken -------------------------------
    baseline_op = float(table.loc[table['arm'] == 'S08 baseline', 'operating_point'].iloc[0])
    block_c_op = float(result.metric('operating_point'))
    ceiling = float(result.metric('ceiling_rescorer'))
    reranker = float(result.metric('ceiling_reranker'))
    headroom = ceiling - baseline_op
    rows = [dict(
        quantity='S08 baseline operating point', value=baseline_op,
        note='the 78-feature combiner, refitted in this harness'),
        dict(quantity='block C operating point', value=block_c_op, note='+ both arm'),
        dict(quantity='ceiling_rescorer', value=ceiling,
             note='a PERFECT score: a correct candidate exists in the pool at all'),
        dict(quantity='ceiling_reranker', value=reranker,
             note='reordering only, every candidate keeping its score (= threshold_only)'),
        dict(quantity='headroom above the baseline, pp', value=100*headroom,
             note='what ANY re-scorer could still win'),
        dict(quantity='reproducibility floor, pp', value=100*REPRODUCIBILITY_FLOOR*baseline_op,
             note='10% of the baseline (Shirley 1980); the gate asked for more than this'),
        dict(quantity='block C gain, pp', value=100*(block_c_op - baseline_op), note=''),
        dict(quantity='block C gain as a fraction of headroom',
             value=(block_c_op - baseline_op)/headroom if headroom else float('nan'),
             note="METRICS.md section 3's convention (F-042), which the gate did not use"),
        dict(quantity='entries with no correct candidate anywhere', value=1.0 - ceiling,
             note='generation failure; belongs to S14/S15, not to a figure of merit'),
        ]
    headroom_table = pd.DataFrame(rows)
    write(headroom_table, args, 'diagnose_headroom')
    print(headroom_table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # ---- 2. where the reachable-lost entries went -----------------------------------------
    decomposition = []
    for scope, frame in (('all', result.aggregate), ('hard', result.hard)):
        if frame is None or not len(frame):
            continue
        row = frame.iloc[0]
        decomposition.append(dict(
            scope=scope, n_entries=int(row['n_entries']),
            operating_point=float(row['operating_point']),
            ceiling_rescorer=float(row['ceiling_rescorer']),
            lost_not_found=float(row['lost_not_found']),
            lost_rank_failure=float(row['lost_rank_failure']),
            lost_threshold_failure=float(row['lost_threshold_failure']),
            lost_both=float(row['lost_both']),
            share_rank_failure=float(row['share_rank_failure']),
            share_threshold_failure=float(row['share_threshold_failure']),
            share_both=float(row['share_both']),
            ))
    for _, row in result.stratum('bravais_lattice').iterrows():
        decomposition.append(dict(
            scope=str(row['level']), n_entries=int(row['n_entries']),
            operating_point=float(row['operating_point']),
            ceiling_rescorer=float(row['ceiling_rescorer']),
            lost_not_found=float(row['lost_not_found']),
            lost_rank_failure=float(row['lost_rank_failure']),
            lost_threshold_failure=float(row['lost_threshold_failure']),
            lost_both=float(row['lost_both']),
            share_rank_failure=float(row['share_rank_failure']),
            share_threshold_failure=float(row['share_threshold_failure']),
            share_both=float(row['share_both']),
            ))
    decomposition = pd.DataFrame(decomposition)
    write(decomposition, args, 'diagnose_decomposition')
    print()
    print(decomposition.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # ---- 3. block A's own accuracy, which bounds what the prior could contribute -----------
    prior_columns = [f'prior_bravais_p_{code}' for code in BRAVAIS_LATTICES]
    per_entry = []
    for frame in dev_frames:
        available = [column for column in prior_columns if column in frame.columns]
        if len(available) != len(prior_columns):
            continue
        first = frame.groupby(['entry_id', 'condition_bundle'], sort=False).first().reset_index()
        per_entry.append(first[['entry_id', 'condition_bundle'] + prior_columns])
    prior_rows = None
    if per_entry:
        prior_rows = pd.concat(per_entry, ignore_index=True)
        truth = entries[['entry_id', 'condition_bundle', 'bravais_lattice_true']]
        prior_rows = prior_rows.merge(truth, on=['entry_id', 'condition_bundle'], how='left')
        matrix = prior_rows[prior_columns].to_numpy()
        order = np.argsort(-matrix, axis=1)
        codes = np.array(BRAVAIS_LATTICES)
        prior_rows['prior_top1'] = codes[order[:, 0]]
        true_index = np.array([list(BRAVAIS_LATTICES).index(code)
                               for code in prior_rows['bravais_lattice_true']])
        rank = np.empty(len(prior_rows), dtype=int)
        for position in range(matrix.shape[1]):
            rank[order[:, position] == true_index] = position
        prior_rows['true_rank'] = rank
        prior_rows['p_true'] = matrix[np.arange(len(prior_rows)), true_index]
        summary = prior_rows.groupby('bravais_lattice_true').agg(
            n=('true_rank', 'size'),
            top1_accuracy=('true_rank', lambda values: float((values == 0).mean())),
            top3_accuracy=('true_rank', lambda values: float((values < 3).mean())),
            mean_p_true=('p_true', 'mean'),
            ).reset_index()
        overall = pd.DataFrame([dict(
            bravais_lattice_true='ALL', n=len(prior_rows),
            top1_accuracy=float((prior_rows['true_rank'] == 0).mean()),
            top3_accuracy=float((prior_rows['true_rank'] < 3).mean()),
            mean_p_true=float(prior_rows['p_true'].mean()),
            )])
        summary = pd.concat([overall, summary], ignore_index=True)
        write(summary, args, 'diagnose_prior_accuracy')
        print()
        print('block A accuracy on the entries block C scores:')
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    write_meta(args, 'diagnose', dict(
        threshold=threshold, baseline_operating_point=baseline_op,
        block_c_operating_point=block_c_op, ceiling_rescorer=ceiling,
        headroom_pp=100*headroom, floor_pp=100*REPRODUCIBILITY_FLOOR*baseline_op,
        fraction_of_headroom=(block_c_op - baseline_op)/headroom if headroom else None,
        ))
    return headroom_table, decomposition


# ---------------------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------------------
def write(frame, args, name):
    """One artefact path convention for every stage, so a table cannot land somewhere else."""
    path = os.path.join(args.artifact_dir, f'{args.tag}_{name}.csv')
    frame.to_csv(path, index=False, encoding='utf-8')
    print(f'wrote {path}', flush=True)
    return path


def commit_hash():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def write_meta(args, stage, extra):
    meta = dict(
        commit=commit_hash(), tag=args.tag, stage=stage, seed=args.seed,
        benchmark=args.benchmark_dir, broadening_tag='1',
        scale='development -- laptop. Every artefact names its production configuration.',
        bounds=[
            'R1: the pool is censored at M20 >= 5, so every number here describes the surviving '
            'candidates only.',
            'R10: no pre-refinement candidate is stored, so "wrong candidate" always means a '
            'Gauss-Newton refined wrong cell.',
            'R11: the grid has no different error *law*, so robustness to one is untested rather '
            'than passed. Block A trained against the repo\'s own sigma(q^2) model, which is the '
            'same generator -- F-008\'s leakage path.',
            'R12: the whole benchmark is one instrument at one broadening tag.',
            'R15: per-peak truth is basis-dependent; nothing here consumes it.',
            ],
        **extra,
        )
    path = os.path.join(args.artifact_dir, f'{args.tag}_{stage}_meta.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, default=str)
    print(f'wrote {path}', flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(
        description='S11 block C: combine the volume/symmetry prior with the fit statistic.')
    parser.add_argument('--stage', default='assignment',
                        choices=('assignment', 'prior', 'combiner', 'ablation', 'cost',
                                 'network', 'diagnose'))
    parser.add_argument('--lattices', nargs='+', default=list(BRAVAIS_LATTICES))
    parser.add_argument('--bundles', nargs='+', default=list(EVALUABLE_BUNDLES))
    parser.add_argument('--splits', nargs='+', default=list(SPLITS))
    parser.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK)
    parser.add_argument('--feature-dir', default=DEFAULT_FEATURES)
    parser.add_argument('--cv-dir', default=DEFAULT_CV)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACTS)
    parser.add_argument('--null-dir', default=DEFAULT_NULL)
    parser.add_argument('--models-dir', default=DEFAULT_MODELS)
    parser.add_argument('--prior-dir', default=DEFAULT_PRIOR)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--models-directory', default=None,
                        help='override the mlindex models tree hkl_ref is read from')
    parser.add_argument('--prior-chunk', type=int, default=1024,
                        help='entries per block A forward pass; sized from the joint table it '
                             'materialises, which is n_entries x n_volumes x n_classes')
    parser.add_argument('--hidden', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--epochs', type=int, default=80,
                        help='max_iter for the MLP; early stopping usually ends it sooner')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--cost-entries', type=int, default=40)
    parser.add_argument('--cost-repeats', type=int, default=3)
    parser.add_argument('--prior-batch', type=int, default=64)
    parser.add_argument('--nproc', type=int, default=max(multiprocessing.cpu_count() - 2, 1))
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', default='S11_C')
    args = parser.parse_args()

    if args.stage == 'assignment':
        run_assignment(args)
    elif args.stage == 'prior':
        run_prior(args)
    elif args.stage == 'combiner':
        run_combiner(args)
    elif args.stage == 'ablation':
        run_ablation(args)
    elif args.stage == 'cost':
        run_cost(args)
    elif args.stage == 'network':
        run_network(args)
    elif args.stage == 'diagnose':
        run_diagnose(args)
    else:
        raise SystemExit(f'stage {args.stage} is not implemented yet')


if __name__ == '__main__':
    main()
