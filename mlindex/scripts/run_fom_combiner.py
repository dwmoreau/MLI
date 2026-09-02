"""S12: the learned combiner, cut hard. Fit on the slice, report on the fully retained pool.

    python mlindex/scripts/run_fom_combiner.py --stage fit
    python mlindex/scripts/run_fom_combiner.py --stage reduce
    python mlindex/scripts/run_fom_combiner.py --stage analyse

**Two pools, and neither can do both jobs.** A learned score is not one of the seven merits the
negative subsampler ranked on, so on Benchmark B every rank metric for it is optimistic by an
unmeasured amount and `FomMetrics` refuses to report one (C2-F-077, C2-R-013). The fully retained
pool has no such problem and is where every reported number comes from -- but it holds `fom-dev`
crystals only, so nothing can be fitted or thresholded on it. So:

  * **fit and threshold** on the Benchmark B slice's `fom-train` (196 source crystals, 9 condition
    bundles), weighted by `sampling_weight`, which is what makes a fit on a thinned pool unbiased
    and which `SCHEMA.md` requires in bold;
  * **report** on the fully retained pool's `fom-dev` (530 crystals, 3 bundles, 43.3 M candidates),
    where `ranks_exact` is True for any score at all.

The two entry sets are disjoint by split and the driver asserts it rather than assuming it.

**Every cut is a retrained paired arm.** PROTOCOL section 8, and this step is where it bites
hardest: permuting campaign 1's extinction group cost 7.28 pp of top-10 while retraining without it
cost 0.004 pp, a factor of 1 800, because permutation pushes a high-cardinality feature out of
distribution and measures the corruption. So there is no permutation importance here. The arms are
declared in `ARMS` as (name, extra groups, dropped columns, what it settles), each is a full refit
at three seeds, and each is paired against `base` by McNemar over the same entries.

**The ladder is ordered so that stopping early still answers the biggest question.** Block 0 is the
harness controls, block 1 the headline and the standing without-symmetry arm, block 2 the merit cut
itself. A session that gets through block 2 has done what the step is for.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomCombiner
from mlindex.model_training import FomMetrics


BASE = Path(__file__).resolve().parents[2]
FIT_POOL = BASE/'mlindex'/'data'/'fom_benchmark_c2'
REPORT_POOL = BASE/'mlindex'/'data'/'fom_full_c2_pool'
MODELS_DIR = BASE/'mlindex'/'models'/'fom_combiner_c2'
ARTIFACT_DIR = BASE/'docs'/'fom_campaign2'/'artifacts'

# The base feature space is `FomCombiner.DEFAULT_GROUPS`; the base ARM drops four columns from it,
# and each drop is a measurement rather than a preference:
#
#   `spacegroup`     C2-F-041: dropping the 158-level categorical is +0.275 pp of operating point,
#                    a gain, at every one of three fit seeds. `plus_spacegroup` puts it back.
#   `ctx_*_rank`,    Measured this session on the 25 crystals present in both pools: `gap_to_best`
#   `ctx_*_z`        is EXACTLY invariant to the thinning (Spearman 1.0000, zero shift) because the
#                    retention rule keeps the pool maximum of every context merit, while `rank` is
#                    3.35x smaller in the thinned pool and `z` moves with the median and spread.
#                    A tree splits on levels, so a split learned at thinned rank 50 means retained
#                    rank ~168. `plus_ctx_rank_z` measures whether they earn their place net of it.
#   `n_peaks`,       Exactly 0.000000 importance in campaign 1's 65-feature table, and `n_peaks` is
#   `hkl_ref_length` a two-valued function of the lattice system the tree already splits on.
#   `q2_max`         -0.000058 there. `plus_pool_structural` restores all three together.
CONTEXT_SKEWED = tuple(f'ctx_{merit}_{statistic}'
                       for merit, _ in FomCombiner.CONTEXT_MERITS
                       for statistic in ('rank', 'z'))
BASE_DROP = ('spacegroup', 'n_peaks', 'hkl_ref_length', 'q2_max') + CONTEXT_SKEWED

ARMS = (
    # -- block 1: the headline, and the arm DWMM requires beside every headline ---------------
    ('base', (), BASE_DROP,
     'the campaign-2 combiner: 7 merits, structural, counts, context gap-to-best'),
    ('no_symmetry', (), BASE_DROP + FomCombiner.SYMMETRY_COUNTS,
     "DWMM's standing instruction: reported beside every headline, never cited from once"),

    # -- block 2: the merit cut, which is what this step is for --------------------------------
    ('plus_dropped_merits', ('campaign1_raw',), BASE_DROP,
     'restores the ten merits S00 cut; a null here licenses 17 -> 7 in one arm'),
    ('drop_counting', (), BASE_DROP + ('X_N', 'n_over', 'max_gap'),
     'C2-F-097: 36 % of the aggregate union-oracle headroom and 56 % of the hard'),
    ('drop_mrev_family', (), BASE_DROP + ('M_tilde', 'M_rev', 'M_sym'),
     "campaign 1 measured 0.283 pp at p = 0.85; re-decided on an uncensored pool"),
    ('m20_only', (), BASE_DROP + ('M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap'),
     'the floor of the merit cut: what the structural and context columns are worth alone'),

    # -- block 3: probation, and the additions phase 3 named ------------------------------------
    ('plus_probation', ('probation',), BASE_DROP,
     'M_wu, M_1, F_N_q together. S09 dropped them for inexact ranks and said so reversibly'),
    ('plus_X_N_soft', ('soft',), BASE_DROP,
     'C2-F-102: the posterior-built count, as an addition to the hard three and not a swap'),
    ('plus_ho_M20', ('holdout',), BASE_DROP,
     "S10c's one surviving hold-out column, as a feature rather than as the score it loses as"),

    # -- block 4: the structural and context trims ----------------------------------------------
    ('drop_structural', (), BASE_DROP + FomCombiner.STRUCTURAL_NUMERIC,
     'C2-F-040 measured -1.675 pp at p < 0.001; confirm the family is load-bearing here'),
    ('drop_context', (), BASE_DROP + tuple(
        name for name in FomCombiner.context_names() if name not in CONTEXT_SKEWED),
     'the strongest non-symmetry family in campaign 1'),
    ('plus_ctx_rank_z', (), ('spacegroup', 'n_peaks', 'hkl_ref_length', 'q2_max'),
     'the two context statistics thinning distorts, measured net of the distortion'),
    ('plus_pool_structural', (), ('spacegroup',) + CONTEXT_SKEWED,
     'add-back of n_peaks, hkl_ref_length and q2_max, rather than an importance table'),
    ('plus_spacegroup', (), ('n_peaks', 'hkl_ref_length', 'q2_max') + CONTEXT_SKEWED,
     'C2-F-041 says dropping it is a gain; confirm on this pool'),

    # -- block 5: architecture and protocol sanity ----------------------------------------------
    ('unweighted_fit', (), BASE_DROP,
     'base with no sampling weight. A null closes the weighting question for S14'),
    )

# Fitted with the labels destroyed, so they measure the harness rather than the model. They are not
# in ARMS because they need a transform applied to the frames rather than a feature-set change.
CONTROL_ARMS = (
    ('label_shuffled',
     'fit AND calibration labels permuted within (entry, bundle). Must land on the tie-break '
     'floor S08 measured for this pool -- 0.2352, not campaign 10.2657'),
    ('prior_only',
     'fit labels permuted, calibration labels real. Isolates what the per-lattice prior alone is '
     'worth: isotonic is monotone so it cannot reorder within a lattice'),
    )

# Scored without being fitted, in the same pool pass, so the leaderboard has its baselines and its
# floors from the same rows as the arms.
REFERENCE_SCORES = ('M20', 'M_sym', 'constant', 'uniform_random')

# Campaign 1's values, kept so these arms sit on the same subsample and capacity as S04's.
SEED = 12345
SEEDS = (12345, 777, 20260826)
HOLDOUT_FRACTION = 0.2
MODEL_PARAMS = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63)
# Higher than campaign 1's 40. Its fit split held ~17 000 (entry, bundle) cells; the slice's
# `fom-train` holds ~1 764, so at 40 negatives the fit is ~72 000 rows against campaign 1's
# ~725 000. 400 restores the row count without changing the rule.
N_NEGATIVES = 400
DEWOLFF_THRESHOLD = 10.0

TAG = 'S12_combiner'


# ---------------------------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------------------------
def split_ids(entries, split, holdout_fraction, seed):
    """Source entries of one split, divided into a fit part and a calibration part.

    By **source entry**, never by candidate (PROTOCOL section 3 rule 5): one crystal appears under
    nine condition bundles with correlated noise, so splitting rows would put near-duplicates of
    the same pattern on both sides of the calibration boundary.
    """
    ids = np.array(sorted(set(entries.loc[entries['split'] == split, 'entry_id'])))
    if not holdout_fraction:
        return set(ids), set()
    rng = np.random.default_rng(seed)
    held = rng.permutation(ids.size) < int(round(holdout_fraction*ids.size))
    return set(ids[~held]), set(ids[held])


def assert_disjoint(fit_entries, report_entries):
    """The fit pool and the report pool must not share a crystal. Asserted, never assumed.

    They are disjoint by split -- one is `fom-train`, the other `fom-dev` -- but a split manifest
    is a file and files get regenerated, and a threshold reported on entries it was chosen on is
    the failure `check_threshold_transfer` exists to catch one level further down.
    """
    overlap = set(fit_entries) & set(report_entries)
    if overlap:
        raise SystemExit(
            f'{len(overlap)} source crystals are in BOTH the fit and report pools, e.g. '
            f'{sorted(overlap)[:5]}. Every reported number would be fitted on its own entries.')
    return True


def _shuffle_labels(frame, seed):
    """`is_correct` permuted within each (entry, bundle), leaving every feature untouched.

    Within the group rather than across it, so each entry keeps its own number of correct
    candidates and only the candidate-to-correctness association is destroyed. A global shuffle
    would also flatten the per-entry base rate, and the control would then be measuring two things.
    """
    codes, _ = FomMetrics._group_codes(frame['entry_id'].to_numpy(),
                                       frame['condition_bundle'].to_numpy())
    rng = np.random.default_rng(seed)
    # `source` lists the rows grouped by entry in RANDOM order within each group; `destination`
    # lists them grouped by entry in their original order. Assigning one through the other permutes
    # inside every group at once, with no Python loop over several thousand groups.
    source = np.lexsort((rng.random(frame.shape[0]), codes))
    destination = np.argsort(codes, kind='stable')
    values = FomMetrics.as_bool(frame['is_correct'])
    shuffled = values.copy()
    shuffled[destination] = values[source]
    return frame.assign(is_correct=shuffled)


def _commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True, cwd=str(BASE)).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------------------------
# stage: fit -- on the slice's fom-train, weighted
# ---------------------------------------------------------------------------------------------
def load_fit_frames(pool, entries, keep_ids, groups, n_negatives, seed, covariates=None):
    """Assembled frames for one set of entries, held in memory rather than streamed.

    Held because every arm makes another pass over the same rows and re-reading the slice per arm
    is hours of I/O for nothing. The fit half is negatively subsampled -- every positive and at
    most `n_negatives` incorrect candidates per (entry, bundle) -- and `subsample_negatives`
    writes `fit_weight`, which composes that draw with the generator's own `sampling_weight`.
    """
    frames = []
    for frame in FomCombiner.combiner_frames_c2(pool, entries, groups=groups,
                                                keep_entry_ids=keep_ids, covariates=covariates):
        if n_negatives is not None:
            frame = FomCombiner.subsample_negatives(frame, n_negatives, seed)
        elif 'sampling_weight' in frame.columns:
            frame = frame.assign(fit_weight=frame['sampling_weight'].to_numpy(dtype=np.float64))
        frames.append(frame)
    if not frames:
        raise SystemExit(f'no frames assembled from {pool} for {len(keep_ids)} entries')
    return frames


def arm_groups(extra):
    """The feature groups for one arm: the base set plus whatever the arm adds."""
    return tuple(dict.fromkeys(tuple(FomCombiner.DEFAULT_GROUPS) + tuple(extra)))


def run_fit(args):
    entries = FomBenchmark.load_entries(FIT_POOL)
    covariates = FomCombiner.entry_covariates(entries)
    fit_ids, cal_ids = split_ids(entries, args.train_split, HOLDOUT_FRACTION, SEED)
    report_entries = FomBenchmark.load_entries(REPORT_POOL)
    assert_disjoint(set(fit_ids) | set(cal_ids), set(report_entries['entry_id']))
    print(f'fit {len(fit_ids)} crystals, calibrate {len(cal_ids)}, '
          f'report on {report_entries["entry_id"].nunique()} disjoint crystals')

    # ONE assembly, with the union of every group any arm needs. An arm's feature set is a
    # selection of columns from the frame -- `design_matrix` reads the names `feature_specification`
    # returns -- so assembling per group set would re-read the same 9 M rows five times to produce
    # frames that differ only in which columns are present.
    union = tuple(dict.fromkeys(
        tuple(FomCombiner.DEFAULT_GROUPS)
        + tuple(group for _, extra, _, _ in ARMS for group in extra)))
    started = time.perf_counter()
    fit_frames = load_fit_frames(FIT_POOL, entries, fit_ids, union, args.n_negatives, SEED,
                                 covariates)
    cal_frames = load_fit_frames(FIT_POOL, entries, cal_ids, union, None, SEED, covariates)
    print(f'assembled {"+".join(union)}: {sum(f.shape[0] for f in fit_frames):,} fit rows '
          f'({sum(int(FomMetrics.as_bool(f["is_correct"]).sum()) for f in fit_frames):,} correct), '
          f'{sum(f.shape[0] for f in cal_frames):,} calibration rows '
          f'({time.perf_counter() - started:.0f} s)', flush=True)

    models_dir = Path(args.models_dir)
    rows = []
    for name, extra, drop, purpose in ARMS:
        rows.append(_fit_or_record(name, arm_groups(extra), drop, fit_frames, cal_frames,
                                   models_dir, args.fit_seed, purpose,
                                   None if name == 'unweighted_fit' else 'sampling_weight'))
    for name, purpose in CONTROL_ARMS:
        # The label-shuffled control destroys the association on BOTH splits; the prior-only
        # control shuffles the fit labels and keeps the calibration ones real, so the per-lattice
        # isotonic is the only thing left that knows anything. Isotonic is monotone, so it cannot
        # reorder candidates within a lattice -- everything it buys is cross-lattice, which is
        # exactly the quantity the control is there to bound.
        shuffled_fit = [_shuffle_labels(frame, args.fit_seed) for frame in fit_frames]
        shuffled_cal = ([_shuffle_labels(frame, args.fit_seed + 1) for frame in cal_frames]
                        if name == 'label_shuffled' else cal_frames)
        rows.append(_fit_or_record(name, arm_groups(()), BASE_DROP, shuffled_fit, shuffled_cal,
                                   models_dir, args.fit_seed, purpose, 'sampling_weight'))

    table = pd.DataFrame(rows)
    path = Path(args.artifact_dir)/f'{args.tag}_fit_table{args.suffix}.csv'
    table.to_csv(path, index=False)
    print(f'\n{len(rows)} arms fitted -> {path}')
    return 0


def _fit_or_record(name, groups, drop, fit_frames, cal_frames, models_dir, seed, purpose,
                   weight_column):
    """Fit one arm, or record why it could not be built and carry on down the ladder.

    The ladder is ordered so that a session that stops early has still answered the biggest
    question, and a hard stop at arm three would defeat that: one arm needing a sidecar column
    nobody wrote should cost that arm, not the fourteen below it. The reason lands in the fit
    table, so a missing arm is visible as a stated absence rather than as a gap.
    """
    try:
        return fit_one(name, groups, drop, fit_frames, cal_frames, models_dir, seed, purpose,
                       weight_column=weight_column)
    except (KeyError, ValueError) as problem:
        print(f'  {name:24s} SKIPPED -- {problem}', flush=True)
        return dict(arm=name, purpose=purpose, groups='+'.join(groups), skipped=str(problem),
                    fit_seed=int(seed))


def fit_one(name, groups, drop, fit_frames, cal_frames, models_dir, seed, purpose,
            weight_column='sampling_weight'):
    """One arm: fit, calibrate on rows it was not fitted on, save. Returns its table row.

    `sampling_weight`, not `fit_weight`. The generator's thinning is a bias and the weight corrects
    it; the driver's own negative subsampling is a deliberate rebalancing and weighting it back
    undoes the only reason it exists -- measured at -17.7 pp of top-10 (C2-F-127).
    """
    started = time.perf_counter()
    combiner = FomCombiner.FomCombiner.fit(
        fit_frames, groups=groups, seed=seed, drop=drop, weight_column=weight_column,
        **MODEL_PARAMS)
    combiner.fit_calibrators(cal_frames, weight_column=weight_column)
    combiner.meta['arm'] = name
    combiner.meta['purpose'] = purpose
    directory = models_dir/f'{name}_seed{seed}'
    combiner.save(directory)
    elapsed = time.perf_counter() - started
    print(f'  {name:24s} {len(combiner.names):>3d} features  {elapsed:6.0f} s  -> {directory.name}',
          flush=True)
    return dict(arm=name, purpose=purpose, groups='+'.join(groups), n_features=len(combiner.names),
                dropped=';'.join(sorted(drop)), fit_seed=int(seed),
                weight_column=weight_column or 'none',
                n_rows_fit=int(combiner.meta['n_rows']),
                n_positive_fit=int(combiner.meta['n_positive']),
                weight_sum=combiner.meta.get('weight_sum'),
                n_calibration_rows=int(combiner.meta['n_calibration_rows']),
                calibrated_lattices=len(combiner.meta['calibrated_lattices']),
                seconds=round(elapsed, 1))


# ---------------------------------------------------------------------------------------------
# stage: reduce -- the only stage that touches the 43 M-candidate report pool
# ---------------------------------------------------------------------------------------------
def load_arms(models_dir, seed, names=None):
    """Every saved arm for one fit seed, as {name: combiner}."""
    arms = {}
    for directory in sorted(Path(models_dir).glob(f'*_seed{seed}')):
        name = directory.name[:-len(f'_seed{seed}')]
        if names and name not in names:
            continue
        arms[name] = FomCombiner.FomCombiner.load(directory)
    if not arms:
        raise SystemExit(f'no fitted arms under {models_dir} for seed {seed}; run --stage fit')
    return arms


def reference_scores(seed):
    """The unfitted columns scored in the same pass: two baselines and two floors.

    The floors are not decoration. A constant score already reaches 0.2352 of top-10 on this pool
    because ties break cubic-first and the dominant failure is symmetry lowering (C2-F-083), so a
    rank metric read against zero rather than against that is read against the wrong thing. They
    are recomputed here rather than quoted because the tie-break floor is a property of the
    population being reported on.
    """
    rng = np.random.default_rng(seed)
    return {
        'M20': 'M20',
        'M_sym': 'M_sym',
        'constant': lambda frame: np.ones(frame.shape[0]),
        'uniform_random': lambda frame: rng.random(frame.shape[0]),
        }


def run_reduce(args):
    arms = load_arms(args.models_dir, args.fit_seed, args.arms)
    report_entries = FomBenchmark.load_entries(REPORT_POOL)
    fit_entries = FomBenchmark.load_entries(FIT_POOL)
    _, cal_ids = split_ids(fit_entries, args.train_split, HOLDOUT_FRACTION, SEED)
    assert_disjoint(cal_ids, set(report_entries['entry_id']))

    # A combiner's `score` is a bound method over a frame, which is exactly the callable shape
    # `reduce_many` takes -- a fitted model is not a stored column and never can be.
    scores = {name: combiner.score for name, combiner in arms.items()}
    scores.update(reference_scores(args.fit_seed))
    orientation = {name: True for name in scores}

    artifact_dir = Path(args.artifact_dir)
    started = time.perf_counter()
    seen = {'n': 0}

    def announce(frame):
        seen['n'] += frame.shape[0]
        print(f'  {frame["condition_bundle"].iloc[0]:24s} {frame.shape[0]:>10,} candidates '
              f'({time.perf_counter() - started:.0f} s)', flush=True)

    print(f'reducing {len(scores)} scores over the fully retained pool')
    reduced = FomMetrics.reduce_many(
        FomCombiner.combiner_frames_c2(REPORT_POOL, report_entries,
                                       groups=_union_groups(arms)),
        scores, entries=report_entries, splits={args.report_split: None},
        higher_is_better=orientation,
        # Explicit, never 'auto': `_resolve_subsampling` takes an iterable of frames for a full
        # pool, so 'auto' would certify a thinned one. The pool's own manifest is the authority.
        subsample_top_k=_pool_depth(REPORT_POOL), on_shard=announce,
        )
    metas = _write_reductions(reduced, artifact_dir, args.tag, args.suffix, require_exact=True)

    # The threshold has to be chosen somewhere the model was not fitted, and the report pool has no
    # fom-train crystals at all. So a second, much smaller reduction over the slice's calibration
    # entries, with the inexactness recorded rather than hidden: these rows feed a THRESHOLD and
    # never a rank claim.
    print('\nreducing the calibration split for threshold selection (ranks inexact by design)')
    calibration = FomMetrics.reduce_many(
        FomCombiner.combiner_frames_c2(FIT_POOL, fit_entries, groups=_union_groups(arms),
                                       keep_entry_ids=cal_ids),
        scores, entries=fit_entries, splits={args.train_split: None},
        higher_is_better=orientation, subsample_top_k=_pool_depth(FIT_POOL),
        allow_inexact_ranks=True,
        )
    metas.update(_write_reductions(calibration, artifact_dir, args.tag, args.suffix + '_cal',
                                   require_exact=False))
    (artifact_dir/f'{args.tag}_reduced_meta{args.suffix}.json').write_text(
        json.dumps(metas, indent=2, sort_keys=True, default=str), encoding='utf-8')
    print(f'\n{seen["n"]:,} candidates reduced -> {artifact_dir}')
    return 0


def _union_groups(arms):
    """Every feature group any loaded arm needs, so one pool pass serves all of them."""
    groups = set()
    for combiner in arms.values():
        groups.update(combiner.groups)
    return tuple(dict.fromkeys(tuple(FomCombiner.DEFAULT_GROUPS) + tuple(sorted(groups))))


def _pool_depth(pool):
    """The retention depth from the pool's own manifest: `None` when nothing was thinned."""
    depth, subsampled = FomBenchmark.subsample_depth(Path(pool))
    return depth if subsampled else None


def _write_reductions(reduced, artifact_dir, tag, suffix, require_exact):
    metas = {}
    for (name, split), (per_entry, meta) in sorted(reduced.items()):
        if require_exact and not meta['ranks_exact']:
            # Asserted rather than trusted: a reduction that silently lost its exactness
            # certificate is the one thing that cannot be detected downstream.
            raise SystemExit(f'{name} on {split} is not rank-exact: {meta["rank_exactness"]}')
        path = artifact_dir/f'{tag}_reduced_{name}_{split}{suffix}.parquet'
        per_entry.to_parquet(path, index=False)
        metas[f'{name}|{split}{suffix}'] = meta
        print(f'  {name:24s} {split:10s} {per_entry.shape[0]:>6,} cells from '
              f'{meta["n_candidates_seen"]:,} candidates', flush=True)
    return metas


# ---------------------------------------------------------------------------------------------
# stage: analyse -- no pool pass; everything is a function of the reductions
# ---------------------------------------------------------------------------------------------
def load_reductions(artifact_dir, tag, suffix):
    metas = json.loads(
        (Path(artifact_dir)/f'{tag}_reduced_meta{suffix}.json').read_text(encoding='utf-8'))
    out = {}
    for key, meta in metas.items():
        name, split = key.split('|')
        path = Path(artifact_dir)/f'{tag}_reduced_{name}_{split}.parquet'
        out[(name, split)] = (pd.read_parquet(path), meta)
    return out


def choose_threshold(selection, budget):
    """The matched false-positive budget, or Youden where the budget is unreachable.

    A model that never scores above the budget's implied threshold has no operating point to
    maximise there and `select_threshold` raises rather than inventing one. The fallback is
    recorded on the row, so a threshold chosen the other way is never mistaken for a matched one.
    """
    try:
        return FomMetrics.select_threshold(selection, objective='operating_point',
                                           max_false_positive_rate=budget), 'matched_fpr'
    except ValueError:
        return FomMetrics.select_threshold(selection, objective='youden'), 'youden'


def scope_row(result, **identifiers):
    """One flat row of what a leaderboard needs, aggregate and hard, from a `MetricsResult`."""
    row = dict(identifiers)
    for metric in ('operating_point', 'top1', 'top10', 'rank_only', 'threshold_only', 'mrr',
                   'ceiling_rescorer', 'reported', 'false_positive', 'precision',
                   'operating_point_given_found'):
        row[metric] = float(result.metric(metric))
    for bound in ('low', 'high'):
        row[f'operating_point_ci_{bound}'] = float(result.metric(f'operating_point_ci_{bound}'))
    hard = result.hard
    if hard is not None and hard.shape[0]:
        for metric in ('top10', 'operating_point', 'operating_point_given_found',
                       'ceiling_rescorer'):
            row[f'hard_{metric}'] = float(hard[metric].iloc[0])
        # Reported beside every hard number, always. The retained pool's hard stratum is 20 cells
        # over 20 crystals of which 6 are reachable (C2-R-019), so a hard delta here is a statement
        # about twenty patterns and must never be read as one about the stratum.
        row['hard_n_entries'] = int(hard['n_entries'].iloc[0])
    return row


def per_lattice(result, prefix):
    """Top-10 and operating point per *true* Bravais lattice, never reweighted.

    The named failure mode for this task is a model that learns "triclinic candidates are usually
    wrong", posts a good aggregate and makes triclinic entries worse. That is only visible here.
    """
    row = {}
    for _, entry in result.stratum('bravais_lattice').iterrows():
        row[f'{prefix}_top10_{entry["level"]}'] = float(entry['top10'])
        row[f'{prefix}_op_{entry["level"]}'] = float(entry['operating_point'])
        row[f'{prefix}_n_{entry["level"]}'] = int(entry['n_entries'])
    return row


def run_analyse(args):
    artifact_dir = Path(args.artifact_dir)
    reductions = load_reductions(artifact_dir, args.tag, args.suffix)
    dev = {name: value for (name, split), value in reductions.items()
           if split == args.report_split}
    cal = {name: value for (name, split), value in reductions.items()
           if split.startswith(args.train_split)}
    if not dev or not cal:
        raise SystemExit('need both the report and calibration reductions; run --stage reduce')

    # The budget is raw M20's own false-positive rate at de Wolff's 10.0, measured on THIS
    # calibration reduction. Campaign 1's 0.2313 describes a different population and is not reused.
    m20_cal = FomMetrics.summarise_per_entry(cal['M20'][0], cal['M20'][1],
                                             threshold=DEWOLFF_THRESHOLD, n_bootstrap=0)
    budget = float(m20_cal.metric('false_positive'))
    print(f'matched false-positive budget from M20 at {DEWOLFF_THRESHOLD}: {budget:.6f}')

    rows, results = [], {}
    for name in sorted(dev):
        selection = FomMetrics.summarise_per_entry(cal[name][0], cal[name][1], n_bootstrap=0)
        choice, rule = choose_threshold(selection, budget)
        result = FomMetrics.summarise_per_entry(
            dev[name][0], dev[name][1], threshold=float(choice.threshold),
            strata=FomMetrics.DEFAULT_STRATA, n_bootstrap=args.n_bootstrap, seed=SEED)
        # Raises if the threshold was chosen on entries it is now reported on. It cannot be, the
        # two pools being disjoint by split, which is exactly why the call is cheap to make.
        FomMetrics.check_threshold_transfer(choice, result)
        results[name] = result
        row = scope_row(result, arm=name, threshold=float(choice.threshold), threshold_rule=rule,
                        fit_seed=args.fit_seed,
                        ranks_exact=bool(dev[name][1]['ranks_exact']))
        row.update(per_lattice(result, 'dev'))
        rows.append(row)
        print(f'  {name:24s} op {row["operating_point"]:.4f}  top10 {row["top10"]:.4f}  '
              f'hard top10 {row.get("hard_top10", float("nan")):.4f}', flush=True)

    table = pd.DataFrame(rows).sort_values('operating_point', ascending=False)
    # The feature count and the fit size belong beside every reported number: a level from a
    # 363 000-row fit is not comparable with campaign 1's from 725 000, and an arm's whole claim is
    # about how many features it carries.
    fit_path = artifact_dir/f'{args.tag}_fit_table{args.suffix}.csv'
    if fit_path.exists():
        fit_table = pd.read_csv(fit_path)
        carry = [name for name in ('n_features', 'n_rows_fit', 'n_positive_fit', 'weight_column',
                                   'dropped', 'purpose') if name in fit_table.columns]
        table = table.merge(fit_table[['arm'] + carry], on='arm', how='left')
    table.to_csv(artifact_dir/f'{args.tag}_main_table{args.suffix}.csv', index=False)
    _write_contrasts(results, artifact_dir, args.tag, args.suffix)
    print(f'\nwrote {args.tag}_main_table{args.suffix}.csv and its contrasts to {artifact_dir}')
    return 0


def _write_contrasts(results, artifact_dir, tag, suffix):
    """Every arm against `base`, and every arm against the two raw baselines. McNemar, paired.

    Two tables because they answer different questions and mixing them is how a feature-set
    contrast gets read as a headline. `_contrasts` is arm-vs-base -- the retrained paired arm
    PROTOCOL section 8 requires for a cut -- and `_mcnemar` is arm-vs-baseline, which is what the
    acceptance gate reads.
    """
    contrasts, headline = [], []
    for name, result in sorted(results.items()):
        for metric in ('operating_point', 'top10'):
            for scope in (None, 'hard'):
                if 'base' in results and name != 'base':
                    contrasts.append(_pair(results['base'], result, 'base', name, metric, scope))
                for baseline in ('M20', 'M_sym'):
                    if baseline in results and name not in ('M20', 'M_sym'):
                        headline.append(
                            _pair(results[baseline], result, baseline, name, metric, scope))
    pd.DataFrame([row for row in contrasts if row]).to_csv(
        artifact_dir/f'{tag}_contrasts{suffix}.csv', index=False)
    pd.DataFrame([row for row in headline if row]).to_csv(
        artifact_dir/f'{tag}_mcnemar{suffix}.csv', index=False)


def _pair(reference, arm, reference_name, arm_name, metric, scope):
    """One McNemar row plus its paired interval, or None where the stratum is empty."""
    try:
        # `is_hard` is a boolean per-entry column, so the mask is built the same way a
        # per-lattice one is -- through `stratum_mask`, which sorts to `mcnemar`'s own order. A
        # mask taken straight off `per_entry` lines up with the wrong rows and nothing raises.
        mask = None if scope is None else FomMetrics.stratum_mask(arm, 'is_hard', True)
        # `mcnemar(reference, arm)` reports `n_b_only` as the arm's own wins, so the delta has to
        # be signed the same way or the table contradicts itself: `paired_delta_ci(a, b)` returns
        # `a - b`, which is reference MINUS arm. Passing (arm, reference) makes a positive delta
        # mean the arm is better, which is what `gained` already means.
        test = FomMetrics.mcnemar(reference, arm, metric=metric, subset=mask)
        interval = FomMetrics.paired_delta_ci(arm, reference, metric=metric, subset=mask)
    except (ValueError, KeyError):
        return None
    return dict(reference=reference_name, arm=arm_name, metric=metric,
                scope=scope or 'aggregate', delta_pp=100*float(interval['delta']),
                ci_low_pp=100*float(interval['ci_low']), ci_high_pp=100*float(interval['ci_high']),
                gained=int(test['n_b_only']), lost=int(test['n_a_only']),
                p_value=float(test['p_value']), method=str(test.get('method', '')))


# ---------------------------------------------------------------------------------------------
# stage: skew -- what the thinning does to a feature, measured rather than assumed
# ---------------------------------------------------------------------------------------------
def _skew_features():
    """Every feature whose value could differ between the two pools, deduplicated.

    The context block, because it is a statistic of the candidate's own pool and the two pools are
    thinned differently -- and the structural block, because a family whose ablation turns out to
    HELP has to be shown not to be shifting before that is reported as a property of the features
    rather than of the two-pool design.
    """
    return list(dict.fromkeys(['ctx_pool_size'] + list(FomCombiner.context_names())
                              + list(CONTEXT_SKEWED) + list(FomCombiner.STRUCTURAL_NUMERIC)))


def run_skew(args):
    """How far the context features move between the fit pool and the report pool.

    The fit pool is thinned 3.3x and the report pool is not, so any feature that is a statistic of
    a candidate's own pool means something different in the two places -- and `sampling_weight`
    cannot repair a shift on the feature side. C2-R-013 and C2-R-020 both name this exposure and
    neither measures it.

    Measurable because 25 `fom-dev` crystals happen to be in both pools over the three shared
    bundles, so the same candidate can be scored under both retention rules and differenced.
    Twenty-five crystals is thin, and the table says so; but the alternative is asserting it.
    """
    shared_bundles = sorted(set(FomBenchmark.available_bundles(FIT_POOL))
                            & set(FomBenchmark.available_bundles(REPORT_POOL)))
    fit_entries = FomBenchmark.load_entries(FIT_POOL)
    report_entries = FomBenchmark.load_entries(REPORT_POOL)
    shared = sorted(set(fit_entries.loc[fit_entries['split'] == args.report_split, 'entry_id'])
                    & set(report_entries['entry_id']))
    if not shared:
        raise SystemExit('the two pools share no crystals; there is nothing to compare')
    print(f'{len(shared)} crystals in both pools, over {len(shared_bundles)} shared bundles')

    def context_of(pool, entries):
        blocks = []
        for frame in FomCombiner.combiner_frames_c2(
                pool, entries, groups=('raw', 'structural', 'context'),
                bundles=shared_bundles, keep_entry_ids=shared, downcast=False):
            blocks.append(frame[list(FomBenchmark.ZOO_KEY_COLUMNS) + _skew_features()])
        return pd.concat(blocks, ignore_index=True)

    thin = context_of(FIT_POOL, fit_entries)
    full = context_of(REPORT_POOL, report_entries)
    joined = thin.merge(full, on=list(FomBenchmark.ZOO_KEY_COLUMNS), suffixes=('_thin', '_full'),
                        validate='1:1')
    rows = [dict(feature='ctx_pool_size', spearman=np.nan,
                 median_thin=float(joined['ctx_pool_size_thin'].median()),
                 median_full=float(joined['ctx_pool_size_full'].median()),
                 median_abs_shift_over_scale=np.nan)]
    for feature in [name for name in _skew_features() if name != 'ctx_pool_size']:
        left = joined[f'{feature}_thin'].to_numpy(dtype=np.float64)
        right = joined[f'{feature}_full'].to_numpy(dtype=np.float64)
        finite = np.isfinite(left) & np.isfinite(right)
        scale = float(np.median(np.abs(right[finite]))) or 1.0
        rows.append(dict(
            feature=feature,
            spearman=float(pd.Series(left[finite]).corr(pd.Series(right[finite]),
                                                        method='spearman')),
            median_thin=float(np.median(left[finite])), median_full=float(np.median(right[finite])),
            median_abs_shift_over_scale=float(np.median(np.abs(left[finite] - right[finite]))/scale),
            ))
    table = pd.DataFrame(rows)
    table.insert(0, 'n_paired_candidates', len(joined))
    table.insert(0, 'n_shared_entries', len(shared))
    path = Path(args.artifact_dir)/f'{args.tag}_retention_skew.csv'
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f'\nwrote {path}')
    return 0


# ---------------------------------------------------------------------------------------------
# stage: calibration -- is the score a probability, on the pool it is reported on
# ---------------------------------------------------------------------------------------------
def run_calibration(args):
    """Reliability, ECE and Brier for each arm, on a uniform sample of the report pool.

    A separate light pass rather than a column on the reduce, because the reduce's output is one
    row per (entry, condition) and calibration is a per-CANDIDATE question -- it needs the score and
    the label of every row, 43 M of them. A sample answers it: ECE is a mean of bin deviations and a
    few million candidates estimate it far more tightly than the 0.05 gate needs.

    **Uniform, with the positives NOT oversampled**, which is the choice that makes the number
    mean what it says. Keeping every correct candidate and sampling the negatives would put the
    positives in at fifty times their rate, and an ECE computed on that sample is the calibration of
    a population that does not exist. `FomMetrics.reliability` takes no weights, and reweighting a
    reliability table by hand is re-implementing a metric, which METRICS section 1 forbids. So the
    sample is a faithful miniature and `n_positive` is reported beside every row -- at a 0.03 % base
    rate the top bin is the thin one, and a reader has to be able to see how thin.
    """
    arms = load_arms(args.models_dir, args.fit_seed, args.arms)
    entries = FomBenchmark.load_entries(REPORT_POOL)
    rng = np.random.default_rng(args.fit_seed)
    collected = {name: [] for name in arms}
    seen = 0
    for frame in FomCombiner.combiner_frames_c2(REPORT_POOL, entries,
                                                groups=_union_groups(arms)):
        shard = frame.loc[rng.random(frame.shape[0]) < args.calibration_rate]
        if not shard.shape[0]:
            continue
        label = FomMetrics.as_bool(shard['is_correct'])
        for name, combiner in arms.items():
            collected[name].append(pd.DataFrame({'score': combiner.score(shard),
                                                 'is_correct': label}))
        seen += frame.shape[0]
        print(f'  {frame["condition_bundle"].iloc[0]:24s} {frame.shape[0]:>10,} candidates, '
              f'{shard.shape[0]:>8,} sampled, {int(label.sum()):>5,} correct', flush=True)

    rows, tables = [], []
    for name in sorted(collected):
        block = pd.concat(collected[name], ignore_index=True)
        score = block['score'].to_numpy(dtype=np.float64)
        label = block['is_correct'].to_numpy(dtype=bool)
        if not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
            # `evaluate` refuses calibration for a score outside [0, 1] and so does this: the
            # calibration of something that is not a probability is a number with no meaning.
            rows.append(dict(arm=name, ece=np.nan, brier=np.nan, n=int(score.size),
                             n_positive=int(label.sum()),
                             note='score is not a probability in [0, 1]'))
            continue
        table, ece, brier = FomMetrics.reliability(score, label, n_bins=args.calibration_bins)
        table.insert(0, 'arm', name)
        tables.append(table)
        rows.append(dict(arm=name, ece=float(ece), brier=float(brier),
                         base_rate=float(label.mean()), mean_score=float(score.mean()),
                         n=int(score.size), n_positive=int(label.sum()), note=''))
        print(f'  {name:24s} ECE {ece:.5f}  Brier {brier:.6f}  '
              f'{int(label.sum()):,} correct of {score.size:,}', flush=True)

    artifact_dir = Path(args.artifact_dir)
    pd.DataFrame(rows).to_csv(artifact_dir/f'{args.tag}_calibration{args.suffix}.csv', index=False)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            artifact_dir/f'{args.tag}_reliability{args.suffix}.csv', index=False)
    print(f'\n{seen:,} candidates seen -> {artifact_dir}')
    return 0


# ---------------------------------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='S12: fit the learned combiner on the slice, report on the retained pool')
    parser.add_argument('--stage', required=True,
                        choices=('fit', 'reduce', 'analyse', 'skew', 'calibration'))
    parser.add_argument('--models-dir', default=str(MODELS_DIR))
    parser.add_argument('--artifact-dir', default=str(ARTIFACT_DIR))
    parser.add_argument('--tag', default=TAG)
    parser.add_argument('--suffix', default='',
                        help='Appended to every output name. Use it to keep a smoke run from '
                             'overwriting a real one')
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--fit-seed', type=int, default=SEED,
                        help='The model seed. Three are run and combined, because a single-seed '
                             'contrast can invert (C2-F-061)')
    parser.add_argument('--n-negatives', type=int, default=N_NEGATIVES,
                        help='Incorrect candidates kept per (entry, bundle) for the fit')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--calibration-rate', type=float, default=0.10,
                        help='Uniform sampling rate for the calibration stage. Uniform, not '
                             'positive-enriched: an ECE computed on an enriched sample is the '
                             'calibration of a population that does not exist')
    parser.add_argument('--calibration-bins', type=int, default=10,
                        help='Equal-count reliability bins')
    parser.add_argument('--arms', nargs='*', default=None,
                        help='Reduce only these arms. The reduce is one pass over 43 M candidates, '
                             'so reduce what is reported and nothing else')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    print(f'S12 --stage {args.stage}  commit {_commit()}  seed {args.fit_seed}')
    return {'fit': run_fit, 'reduce': run_reduce, 'analyse': run_analyse,
            'skew': run_skew, 'calibration': run_calibration}[args.stage](args)


if __name__ == '__main__':
    raise SystemExit(main())
