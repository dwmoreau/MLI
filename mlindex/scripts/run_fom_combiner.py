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

# ---------------------------------------------------------------------------------------------
# The search ladder: backward elimination from what three seeds have already settled.
# ---------------------------------------------------------------------------------------------
# The main ladder above asks "does this family earn its place". Three seeds answered that for two
# of them and left twelve arms unsettled (C2-F-132), so a second ladder starts from the answer
# rather than from `base`: the **structural family is out**, robustly, at +8.74 pp of operating
# point across every seed. That leaves sixteen features, and the question becomes which of THOSE
# sixteen can go -- which is the acceptance gate's "at most ~25 features, every cut justified by a
# retrained paired arm".
#
# One column at a time, because a group drop cannot say which of its members carried the group.
# `drop_structural` removes thirteen at once and the main ladder could not tell whether all
# thirteen were harmful or one was and the rest were inert.
#
# `bravais_lattice` IS tested here, against campaign 1's instruction to keep it regardless. That
# instruction was wrong about the mechanism: `fit_calibrators` and `score` read the lattice from
# the FRAME, not from the design matrix, so dropping it as a *feature* leaves the per-lattice
# isotonic key intact. Whether it earns a place as a feature is therefore a measurable question and
# not a structural requirement.
LEAN_DROP = BASE_DROP + FomCombiner.STRUCTURAL_NUMERIC


def _lean_features():
    """The sixteen the search starts from: seven merits, four counts, four context, one lattice."""
    names, _ = FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS, drop=LEAN_DROP)
    return names


def search_arms():
    """`lean`, then `lean` less each of its own columns, one at a time.

    Declared as a function rather than a constant because the column list is derived from
    `feature_specification` -- writing it out would be a second copy of the feature set that could
    drift from the first, which is how `merit_at_prune` came to be mislabelled (C2-F-067).
    """
    arms = [('lean', (), LEAN_DROP,
             'the 16 features that survive dropping the structural family (C2-F-132)')]
    for column in _lean_features():
        arms.append((f'lean_minus_{column}', (), LEAN_DROP + (column,),
                     f'is {column} carrying anything once the structural family is gone'))
    # Two group drops for reference, so a single-column result can be read against the family's.
    arms.append(('lean_minus_counts', (), LEAN_DROP + FomCombiner.SYMMETRY_COUNTS,
                 'all four absence counts at once'))
    arms.append(('lean_minus_context', (), LEAN_DROP + tuple(
        name for name in FomCombiner.context_names() if name not in CONTEXT_SKEWED),
        'all four context columns at once'))
    return tuple(arms)


# ---------------------------------------------------------------------------------------------
# Cycle 2: joint removal. "Each is free alone" and "all are free together" are different claims.
# ---------------------------------------------------------------------------------------------
# Cycle 1 removed one column at a time from `lean` and settled five results at three seeds. Two
# columns are actively HARMFUL -- dropping `M20` gains +1.93 pp of operating point and dropping
# `M_tilde` +1.72 -- so `core` is `lean` less those two. Three are load-bearing:
# `ctx_M_sym_gap_to_best` (-4.86), `bravais_lattice` (-4.05) and the absence counts AS A GROUP
# (-1.99, while no individual count matters).
#
# The other eleven came back with no reliable effect either way, and that is the whole reason this
# cycle exists. **A column with no individual effect is not the same as a column that can be
# removed**, because these features are correlated by construction -- `M_sym` IS `M_tilde` x
# `M_rev`, and the four context columns are the same statistic over four merits. Removing any one
# leaves the information in its neighbours; removing them together may not. So cycle 2 tests the
# joint drops, and the smallest candidate set, directly.
CORE_DROP = LEAN_DROP + ('M20', 'M_tilde')

# Individually null in cycle 1, grouped by what they are.
WEAK_MERITS = ('M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap')
WEAK_CONTEXT = ('ctx_M20_gap_to_best', 'ctx_n_over_gap_to_best', 'ctx_max_gap_gap_to_best')
WEAK_COUNTS = ('n_absent_extra', 'f_absent_extra', 'n_groups_searched')

# The three ladders' surviving arms, by name, so `--stage transfer` and `--stage cost` can be
# pointed at the model the campaign actually settled on. Both stages hardcoded `base` while `base`
# was the only fitted arm; `core` is 14 features and 8.6 pp better, so a transfer or cost number
# quoted for `base` is a number for a model nobody is going to ship.
DROP_SETS = {'base': BASE_DROP, 'lean': LEAN_DROP, 'core': CORE_DROP}


def search2_arms():
    """`core`, the three joint family drops, and the minimal set they imply."""
    arms = [
        ('core', (), CORE_DROP,
         'lean less M20 and M_tilde, the two cycle 1 found actively harmful'),
        ('core_minus_weak_merits', (), CORE_DROP + WEAK_MERITS,
         'all five individually-null merits at once -- leaves no classical merit at all'),
        ('core_minus_weak_context', (), CORE_DROP + WEAK_CONTEXT,
         'the three context columns that are not ctx_M_sym'),
        ('core_minus_weak_counts', (), CORE_DROP + WEAK_COUNTS,
         'three of the four absence counts, keeping n_absent_extra_in_range'),
        ('core_minus_weak_merits_context', (), CORE_DROP + WEAK_MERITS + WEAK_CONTEXT,
         'both merit and context tails together'),
        ('minimal', (), CORE_DROP + WEAK_MERITS + WEAK_CONTEXT + WEAK_COUNTS,
         'everything cycle 1 could not show carries anything: the smallest defensible set'),
        # Keep one classical merit against the minimal set, since dropping every merit from a
        # figure-of-merit model is the kind of result that deserves its own control.
        ('minimal_plus_M_sym', (), CORE_DROP + tuple(
            name for name in WEAK_MERITS if name != 'M_sym') + WEAK_CONTEXT + WEAK_COUNTS,
         'the minimal set with M_sym restored -- is any classical merit needed at all'),
        ('minimal_plus_counts', (), CORE_DROP + WEAK_MERITS + WEAK_CONTEXT,
         'the minimal set with all four counts restored'),
        ]
    return tuple(arms)


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
    ladder = {'search': search_arms, 'search2': search2_arms}.get(
        args.ladder, lambda: ARMS)()
    union = tuple(dict.fromkeys(
        tuple(FomCombiner.DEFAULT_GROUPS)
        + tuple(group for _, extra, _, _ in ladder for group in extra)))
    started = time.perf_counter()
    if args.fit_frame:
        # Built elsewhere by `--stage export-fit`, which is how a fit on Benchmark B's ~11 000
        # `fom-train` crystals reaches a laptop that cannot hold the pool they came from.
        fit_path = Path(args.fit_frame)
        cal_path = fit_path.with_name(fit_path.name.replace('_fit_frame', '_cal_frame'))
        if cal_path == fit_path or not cal_path.exists():
            raise SystemExit(
                f'--fit-frame must name a `*_fit_frame*.parquet` with its `_cal_frame` sibling '
                f'beside it; looked for {cal_path}. The calibrator has to be fitted on rows the '
                f'model was not fitted on, so the pair travels together or neither does.')
        fit_frames = [pd.read_parquet(fit_path)]
        cal_frames = [pd.read_parquet(cal_path)]
        print(f'read {sum(f.shape[0] for f in fit_frames):,} fit rows and '
              f'{sum(f.shape[0] for f in cal_frames):,} calibration rows from {fit_path.parent}')
    else:
        fit_frames = load_fit_frames(FIT_POOL, entries, fit_ids, union, args.n_negatives, SEED,
                                     covariates)
        cal_frames = load_fit_frames(FIT_POOL, entries, cal_ids, union, None, SEED, covariates)
    print(f'assembled {"+".join(union)}: {sum(f.shape[0] for f in fit_frames):,} fit rows '
          f'({sum(int(FomMetrics.as_bool(f["is_correct"]).sum()) for f in fit_frames):,} correct), '
          f'{sum(f.shape[0] for f in cal_frames):,} calibration rows '
          f'({time.perf_counter() - started:.0f} s)', flush=True)

    models_dir = Path(args.models_dir)
    rows = []
    for name, extra, drop, purpose in ladder:
        rows.append(_fit_or_record(name, arm_groups(extra), drop, fit_frames, cal_frames,
                                   models_dir, args.fit_seed, purpose,
                                   None if name == 'unweighted_fit' else 'sampling_weight'))
    for name, purpose in (CONTROL_ARMS if args.ladder == 'main' else ()):
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
    # Calibrate on `fit_weight`, never on `sampling_weight`, and the asymmetry with the fit is the
    # point. The fit must NOT undo its own negative subsampling (C2-F-127) because it wants a
    # discriminative model; the calibrator MUST undo it, because its entire job is to state the
    # prior the subsampling removed. `subsample_negatives` sets `fit_weight` equal to
    # `sampling_weight` when nothing was thinned, so this is a no-op for an unsubsampled
    # calibration split and is correct for a subsampled one -- which is what a full-scale fit on
    # the cluster has to use, since an unsubsampled `fom-train` calibration split is 164 M rows.
    combiner.fit_calibrators(cal_frames,
                             weight_column=None if weight_column is None else 'fit_weight')
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
    metas.update(_write_reductions(calibration, artifact_dir, args.tag, args.suffix,
                                   require_exact=False, kind='_cal'))
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


def _write_reductions(reduced, artifact_dir, tag, run_suffix, require_exact, kind=''):
    """Write one reduction per (score, split) and return their metas.

    **`kind` and `run_suffix` are different things and were briefly the same one.** `kind` labels
    what the reduction IS -- '' for the report split, '_cal' for the calibration rows a threshold is
    chosen on -- and belongs in the meta key, because `run_analyse` looks the two up by split.
    `run_suffix` namespaces a whole RUN, which is how three fit seeds coexist in one artifact
    directory, and belongs only in the filename. Folding the second into the meta key made every
    split label read `fom-dev_seed777`, so the analyse stage matched nothing and a two-hour run
    ended in a stage that could not find its own output.
    """
    metas = {}
    for (name, split), (per_entry, meta) in sorted(reduced.items()):
        if require_exact and not meta['ranks_exact']:
            # Asserted rather than trusted: a reduction that silently lost its exactness
            # certificate is the one thing that cannot be detected downstream.
            raise SystemExit(f'{name} on {split} is not rank-exact: {meta["rank_exactness"]}')
        path = artifact_dir/f'{tag}_reduced_{name}_{split}{kind}{run_suffix}.parquet'
        per_entry.to_parquet(path, index=False)
        metas[f'{name}|{split}{kind}'] = meta
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
        path = Path(artifact_dir)/f'{tag}_reduced_{name}_{split}{suffix}.parquet'
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
                reference = next((name for name in ('core', 'lean', 'base') if name in results),
                                 'base')
                if reference in results and name != reference:
                    contrasts.append(
                        _pair(results[reference], result, reference, name, metric, scope))
                for baseline in ('M20', 'M_sym'):
                    if baseline in results and name not in ('M20', 'M_sym'):
                        headline.append(
                            _pair(results[baseline], result, baseline, name, metric, scope))
    pd.DataFrame([row for row in contrasts if row]).to_csv(
        artifact_dir/f'{tag}_contrasts{suffix}.csv', index=False)
    pd.DataFrame([row for row in headline if row]).to_csv(
        artifact_dir/f'{tag}_mcnemar{suffix}.csv', index=False)
    _write_per_lattice(results, artifact_dir, tag, suffix)


def _write_per_lattice(results, artifact_dir, tag, suffix):
    """One PAIRED test per lattice, for the arms a per-lattice claim is made about.

    The per-lattice table in the main table is a difference of two rates. A difference of rates is
    not a paired comparison and carries no interval, which is exactly the defect campaign 1 shipped
    -- every per-lattice claim in its zoo and null packages was an unpaired delta, because its
    McNemar routine's mask argument raised on every call (F-087, Q34). S09 was the first step in
    the project to run one paired, and a gate that FAILS on a per-lattice condition had better be
    failing on the same evidence a passing one would need.

    `stratum_mask` rather than a mask off `per_entry`: `mcnemar` sorts both results by
    (entry_id, condition_bundle) and applies the mask positionally, so a mask built in the pool's
    own order lines up with the wrong rows and nothing raises.
    """
    rows = []
    for arm in sorted(results):
        if arm in ('constant', 'uniform_random'):
            continue
        for baseline in ('M_sym', 'M20', 'base'):
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
                    row = _pair(results[baseline], results[arm], baseline, arm, metric, None,
                                mask=mask)
                    if row:
                        row['scope'] = f'lattice={lattice}'
                        row['n_entries'] = int(mask.sum())
                        rows.append(row)
    pd.DataFrame(rows).to_csv(artifact_dir/f'{tag}_by_lattice_mcnemar{suffix}.csv', index=False)


def _pair(reference, arm, reference_name, arm_name, metric, scope, mask=None):
    """One McNemar row plus its paired interval, or None where the stratum is empty."""
    try:
        # `is_hard` is a boolean per-entry column, so the mask is built the same way a
        # per-lattice one is -- through `stratum_mask`, which sorts to `mcnemar`'s own order. A
        # mask taken straight off `per_entry` lines up with the wrong rows and nothing raises.
        if mask is None:
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


# The three fit seeds. One is not enough: C2-F-061 measured two halves of one feature group
# swapping significance between seeds, with each half's seed-to-seed spread wider than the effect
# it was estimating. An arm verdict from one seed is a hypothesis.
SEED_SUFFIXES = ('', '_seed777', '_seed20260826')
SEARCH_SEED_SUFFIXES = ('_search_seed12345', '_search_seed777', '_search_seed20260826')
SEARCH2_SEED_SUFFIXES = ('_search2_seed12345', '_search2_seed777', '_search2_seed20260826')


# ---------------------------------------------------------------------------------------------
# stage: transfer -- does a model that never saw a condition still work on it
# ---------------------------------------------------------------------------------------------
def _transfer_drop_cells(frames, n_rows, seed):
    """Remove whole (entry, bundle) cells at random until at least `n_rows` rows are gone.

    **Whole cells, not rows.** One crystal appears under nine bundles with correlated noise, and a
    pattern's candidates are one unit; thinning rows inside a cell would change what the fit sees
    ABOUT each pattern rather than HOW MANY patterns it sees -- which is the wrong control for
    "one bundle fewer", since withholding a bundle also removes whole cells. PROTOCOL section 3
    rule 5 is the same rule one level up.

    Returns the surviving frames and the row count actually removed, which overshoots `n_rows` by
    at most one cell.
    """
    cells = [(i, entry) for i, frame in enumerate(frames)
             for entry in sorted(set(frame['entry_id']))]
    order = np.random.default_rng(seed).permutation(len(cells))
    removed, dropped = 0, {}
    for position in order:
        if removed >= n_rows:
            break
        i, entry = cells[position]
        removed += int((frames[i]['entry_id'] == entry).sum())
        dropped.setdefault(i, []).append(entry)
    out = []
    for i, frame in enumerate(frames):
        if i in dropped:
            frame = frame.loc[~frame['entry_id'].isin(dropped[i])]
        if frame.shape[0]:
            out.append(frame.reset_index(drop=True))
    return out, removed


def run_transfer(args):
    """Leave-one-condition-bundle-out: fit without a bundle, report on it.

    **This is NOT the error-law transfer the S12 handoff asks for, and that check cannot be run on
    this benchmark at all.** The handoff says "Benchmark B's error-law bundles make the real check
    possible for the first time". There are no error-law bundles: DWMM's decision of 2026-08-26 is
    Gaussian only, with severity (the multiplier) and shape (the sigma intercept) as the axes, and
    what that leaves untested is carried as **C2-R-008** rather than discharged. Reconfirmed
    2026-09-01. So this measures transfer across **conditions**, which is the check campaign 1
    already ran (leave-one-condition-out, 1.6 pp average and 2.7 pp worst), and it must be labelled
    as that wherever it is quoted. C2-R-008 stands unreduced.

    **And it can only cover three of the nine bundles.** A transfer claim needs the model reported
    on the condition it did not see, with exact ranks -- which means the fully retained pool, which
    carries the three severity bundles and none of the sparsity, contaminant or second-phase ones
    (C2-R-024). So what is measured is transfer across error *severity*: fit without 0.1x, or
    without 1x, or without 2x, and report on the one left out.
    """
    fit_entries = FomBenchmark.load_entries(FIT_POOL)
    report_entries = FomBenchmark.load_entries(REPORT_POOL)
    fit_ids, cal_ids = split_ids(fit_entries, args.train_split, HOLDOUT_FRACTION, SEED)
    assert_disjoint(set(fit_ids) | set(cal_ids), set(report_entries['entry_id']))
    covariates = FomCombiner.entry_covariates(fit_entries)

    shared = sorted(set(FomBenchmark.available_bundles(FIT_POOL))
                    & set(FomBenchmark.available_bundles(REPORT_POOL)))
    all_bundles = sorted(FomBenchmark.available_bundles(FIT_POOL))
    print(f'{len(all_bundles)} fit bundles, {len(shared)} of them reportable with exact ranks: '
          f'{", ".join(shared)}')

    groups = arm_groups(())
    drop = DROP_SETS[args.transfer_arm]

    # Assembled ONCE over all nine bundles and then partitioned, rather than re-read per held-out
    # bundle. Three reads of the fit pool cost three times the I/O for the same rows, and holding
    # them lets the size-matched control below be built from exactly the same subsample.
    started = time.perf_counter()
    all_fit = [FomCombiner.subsample_negatives(frame, args.n_negatives, SEED)
               for frame in FomCombiner.combiner_frames_c2(
                   FIT_POOL, fit_entries, groups=groups, bundles=all_bundles,
                   keep_entry_ids=fit_ids, covariates=covariates)]
    all_cal = [FomCombiner.subsample_negatives(frame, None, SEED)
               for frame in FomCombiner.combiner_frames_c2(
                   FIT_POOL, fit_entries, groups=groups, bundles=all_bundles,
                   keep_entry_ids=cal_ids, covariates=covariates)]
    print(f'  assembled {sum(f.shape[0] for f in all_fit):,} fit rows over {len(all_bundles)} '
          f'bundles ({time.perf_counter() - started:.0f} s)', flush=True)

    def _bundle_of(frame):
        return frame['condition_bundle'].iloc[0]

    def _fit(fit_frames, cal_frames):
        combiner = FomCombiner.FomCombiner.fit(
            fit_frames, groups=groups, seed=args.fit_seed, drop=drop,
            weight_column='sampling_weight', **MODEL_PARAMS)
        combiner.fit_calibrators(cal_frames, weight_column='fit_weight')
        return combiner

    models = {}
    for index, held_out in enumerate(shared):
        started = time.perf_counter()
        fit_frames = [f for f in all_fit if _bundle_of(f) != held_out]
        cal_frames = [f for f in all_cal if _bundle_of(f) != held_out]
        models[f'without_{held_out}'] = _fit(fit_frames, cal_frames)
        n_rows = sum(f.shape[0] for f in fit_frames)
        print(f'  fitted without {held_out}: {n_rows:,} rows '
              f'({time.perf_counter() - started:.0f} s)', flush=True)

        # **The control the contrast is worthless without.** Withholding a bundle withholds a
        # CONDITION and, at the same time, a slice of the fit set -- and this campaign has already
        # measured that fit size is the binding constraint here, 14 features beating 29 by 8.6 pp
        # at 157 crystals (C2-F-134). So a loss on the unseen condition is not evidence of failed
        # transfer until the same loss of rows, spread across the conditions the model DOES see,
        # has been shown to cost less. Same row count, same seed, cells drawn from all nine bundles.
        started = time.perf_counter()
        # A per-bundle seed, so the three controls are three independent draws rather than nested
        # prefixes of one permutation -- and derived from the bundle's POSITION, not `hash()`,
        # which is salted per process in Python 3 and would make this un-rerunnable. The
        # CALIBRATION rows are left whole and cover all nine bundles, matching the incumbent's:
        # the control must differ from the incumbent in row count and in nothing else, which is
        # what makes `held_out` against `control` the condition effect alone.
        control_fit, removed = _transfer_drop_cells(
            all_fit, sum(f.shape[0] for f in all_fit) - n_rows, SEED + 1 + index)
        models[f'size_matched_{held_out}'] = _fit(control_fit, all_cal)
        print(f'  size-matched control: {sum(f.shape[0] for f in control_fit):,} rows, '
              f'{removed:,} removed from all bundles ({time.perf_counter() - started:.0f} s)',
              flush=True)
        del fit_frames, cal_frames, control_fit
    del all_fit, all_cal

    # The incumbent, which saw everything, scored in the same pass so the comparison is paired.
    incumbent = Path(args.models_dir)/f'{args.transfer_arm}_seed{args.fit_seed}'
    if not incumbent.is_dir():
        raise SystemExit(
            f'no fitted {args.transfer_arm} arm at {incumbent}. The incumbent must be the SAME\n'
            f'model the held-out arms are, refit only without a bundle -- refitting it here\n'
            f'instead would confound the transfer contrast with fit noise. Point --models-dir\n'
            f'at the ladder that fitted it (core lives under fom_combiner_c2_search2).')
    models['all_bundles'] = FomCombiner.FomCombiner.load(incumbent)
    scores = {name: model.score for name, model in models.items()}
    scores['M_sym'] = 'M_sym'

    reduced = FomMetrics.reduce_many(
        FomCombiner.combiner_frames_c2(REPORT_POOL, report_entries,
                                       groups=_union_groups(models)),
        scores, entries=report_entries, splits={args.report_split: None},
        higher_is_better={name: True for name in scores},
        subsample_top_k=_pool_depth(REPORT_POOL))

    rows = []
    for held_out in shared:
        name = f'without_{held_out}'
        for scope_bundle in shared:
            for label, source in (('held_out', name), ('all_bundles', 'all_bundles'),
                                  ('size_matched', f'size_matched_{held_out}')):
                per_entry, meta = reduced[(source, args.report_split)]
                mask = per_entry['condition_bundle'] == scope_bundle
                if not mask.any():
                    continue
                result = FomMetrics.summarise_per_entry(
                    per_entry.loc[mask].reset_index(drop=True), meta, n_bootstrap=0)
                rows.append(dict(
                    fitted_without=held_out, reported_on=scope_bundle, arm=label,
                    is_the_unseen_condition=(scope_bundle == held_out),
                    n_entries=int(mask.sum()),
                    top10=float(result.metric('top10')),
                    top1=float(result.metric('top1')),
                    rank_only=float(result.metric('rank_only'))))
    table = pd.DataFrame(rows)
    index = ['fitted_without', 'reported_on', 'is_the_unseen_condition']
    wide = table.pivot_table(index=index, columns='arm', values='top10').reset_index()
    # Three quantities, and only the last of them is the transfer claim:
    #   delta_pp             held-out arm against the incumbent = condition lost AND rows lost
    #   size_effect_pp       size-matched control against the incumbent = rows lost alone
    #   condition_effect_pp  held-out arm against the control    = CONDITION lost alone
    wide['delta_pp'] = 100*(wide['held_out'] - wide['all_bundles'])
    if 'size_matched' in wide:
        wide['size_effect_pp'] = 100*(wide['size_matched'] - wide['all_bundles'])
        wide['condition_effect_pp'] = 100*(wide['held_out'] - wide['size_matched'])
    # The crystal count, so a reader can size the delta: on this pool 530 entries make one bundle,
    # so 3.4 pp is eighteen crystals. **The delta is a difference of RATES on the same crystals,
    # not a McNemar** -- which is the right quantity for a transfer bound (how much is lost) and
    # is NOT a significance claim. Nothing here should be quoted with a p-value.
    wide = wide.merge(table.groupby(index, as_index=False)['n_entries'].max(), on=index)
    wide['arm_features'] = args.transfer_arm
    path = (Path(args.artifact_dir)
            / f'{args.tag}_condition_transfer_{args.transfer_arm}{args.suffix}.csv')
    wide.to_csv(path, index=False)
    unseen = wide[wide['is_the_unseen_condition']]
    print(f'\ntop-10 when the {args.transfer_arm} model never saw the condition it is '
          'reported on:')
    columns = [c for c in ('fitted_without', 'held_out', 'size_matched', 'all_bundles',
                           'delta_pp', 'size_effect_pp', 'condition_effect_pp') if c in unseen]
    print(unseen[columns].to_string(index=False))
    if 'condition_effect_pp' in unseen:
        print('\ncondition_effect_pp is the transfer claim. delta_pp confounds it with the rows '
              'the withheld bundle took with it, and size_effect_pp is that confound measured.')
    print(f'\nwrote {path}')
    print('Transfer across CONDITIONS, not across error laws -- no error-law bundle exists '
          '(C2-R-008), and only 3 of 9 bundles are reportable with exact ranks (C2-R-024).')
    print('Reported as top-10, not as the operating point: this stage selects no threshold, so '
          'an operating point is undefined here. Do not compare these deltas with the '
          'operating-point deltas everywhere else in S12.')
    return 0


# ---------------------------------------------------------------------------------------------
# stage: cost -- what the score costs, in get_M20 units, for the record
# ---------------------------------------------------------------------------------------------
def run_cost(args):
    """Price the feature stack and the model against `get_M20`, on a real candidate block.

    **Nothing here decides anything.** Cost stopped being an exclusion criterion in campaign 2 on
    2026-08-25 -- a merit that outperforms the rest is kept whatever it costs -- and the reason is
    structural rather than generous: this campaign does not change the inner loop, and the prune and
    the final ranking each read a merit once per candidate, so a threefold difference is invisible
    against the Gauss-Newton refinement that produced the candidate. The table exists because the
    arithmetic is worth knowing and because campaign 1 excluded merits on prices that were wrong by
    ten to twenty times (C2-F-001).

    Priced per candidate on a real (entry, lattice) block rather than on synthetic input, because
    every one of these costs is dominated by the reference-line pass and its size is a property of
    the lattice. A low-symmetry block is the honest place to measure.
    """
    import time as _time
    from mlindex.utilities.FigureOfMerits import get_M20
    import pyarrow.parquet as pq

    pool = Path(args.report_pool or REPORT_POOL)
    path = sorted(pool.glob(f'candidates*_{args.cost_lattice}.parquet'))
    if not path:
        raise SystemExit(f'no {args.cost_lattice} candidate file under {pool}')
    frame = pd.DataFrame(
        next(pq.ParquetFile(path[0]).iter_batches(batch_size=args.cost_rows)).to_pydict())
    entries = FomBenchmark.load_entries(pool)
    bundle = frame['condition_bundle'].iloc[0]
    entries = entries.loc[entries['condition_bundle'] == bundle]
    n = frame.shape[0]
    print(f'{n:,} {args.cost_lattice} candidates from {path[0].name}')

    def timed(label, call, repeats=3):
        """Median of `repeats` timed calls, per candidate. Median rather than best-of: this is a
        cost for the record, not a benchmark, and the median is what a run actually pays."""
        samples = []
        for _ in range(repeats):
            started = _time.perf_counter()
            call()
            samples.append(_time.perf_counter() - started)
        rows_scored = getattr(call, 'rows', n)
        return dict(step=label, seconds=float(np.median(samples)), rows=int(rows_scored),
                    microseconds_per_candidate=float(np.median(samples)/rows_scored*1e6))

    # The unit. Everything else is quoted against this, and it is measured here rather than taken
    # from S02 so the ratio is internally consistent on one machine at one block size.
    blocks = []
    for key, group in frame.groupby(['entry_id', 'lattice_system', 'bravais_lattice',
                                     'spacegroup', 'n_peaks'], sort=False):
        entry_id, lattice_system, bravais_lattice, spacegroup, n_peaks = key
        q2_obs = np.asarray(entries.set_index('entry_id').loc[entry_id, 'q2_obs'],
                            dtype=np.float64)[:int(n_peaks)]
        xnn = np.vstack([np.asarray(value, dtype=np.float64) for value in group['xnn']])
        blocks.append((q2_obs, xnn, lattice_system, bravais_lattice, spacegroup))

    def assign_all():
        return [FomBenchmark.assign_lines(*block) for block in blocks]

    prepared = assign_all()

    def m20_only():
        for (q2_obs, *_), (q2_ref_calc, _, _, q2_calc) in zip(blocks, prepared):
            get_M20(q2_obs, q2_calc, q2_ref_calc.copy())

    rows = [timed('get_M20 (the unit), reference lines already built', m20_only),
            timed('assign_lines: build the reference lines', assign_all),
            timed('reduced_merits: the seven ranking merits',
                  lambda: FomBenchmark.reduced_merits(frame, entries)),
            timed('structural_features: the S12 design-matrix columns',
                  lambda: FomBenchmark.structural_features(frame, entries))]

    combiner = FomCombiner.FomCombiner.load(
        Path(args.models_dir)/f'{args.transfer_arm}_seed{args.fit_seed}')
    assembled = next(FomCombiner.combiner_frames_c2(
        pool, entries, bundles=[bundle], keep_entry_ids=set(frame['entry_id'])))
    rows.append(timed('design_matrix: assemble the model input',
                      lambda: combiner.design_matrix(assembled)))
    matrix = combiner.design_matrix(assembled)
    rows.append(timed('predict_batch: the tree, at this block size', lambda:
                      combiner.predict_batch(matrix)))
    rows.append(timed('score: tree plus per-lattice isotonic, at this block size',
                      lambda: combiner.score(assembled)))

    # **And the same two at a realistic batch, which is a different number by two orders of
    # magnitude.** A gradient-boosted tree carries a fixed per-CALL cost that a few thousand rows
    # do not amortise, and scoring happens over a whole pool: the reduce stage hands it a 14 M-row
    # bundle. Quoting the small-block figure would price the model at ~34x `get_M20` where it is
    # nearer 0.1x, which is exactly the kind of error that excluded merits in campaign 1.
    tiles = max(1, int(np.ceil(args.cost_batch/max(matrix.shape[0], 1))))
    large = np.repeat(matrix, tiles, axis=0) if tiles > 1 else matrix
    large_call = (lambda: combiner.predict_batch(large))
    large_call.rows = large.shape[0]
    rows.append(timed(f'predict_batch: the tree, at {large.shape[0]:,} rows', large_call,
                      repeats=2))

    table = pd.DataFrame(rows)
    unit = table.loc[0, 'microseconds_per_candidate']
    table['get_M20_units'] = table['microseconds_per_candidate']/unit
    table.insert(0, 'lattice', args.cost_lattice)
    table.insert(0, 'n_candidates', n)
    path_out = Path(args.artifact_dir)/f'{args.tag}_cost.csv'
    table.to_csv(path_out, index=False)
    print(table[['step', 'microseconds_per_candidate', 'get_M20_units']].to_string(index=False))
    print(f'\nwrote {path_out}')
    print('Cost decides nothing in campaign 2 (decision 2026-08-25); this is for the record.')
    return 0


# ---------------------------------------------------------------------------------------------
# stage: combine -- an arm verdict is what survives every seed
# ---------------------------------------------------------------------------------------------
def run_combine(args):
    """Pool the per-seed contrast tables into one verdict per arm.

    Reports the spread, not just the mean, and two booleans: whether every seed agreed on the
    **sign**, and whether every seed reached p < 0.05. Both are needed. C2-F-061's decomposition
    failed precisely because its two halves swapped which of them was significant between seeds
    while both means stayed positive -- a mean-of-three would have called it settled.

    An arm that is `same_sign_all_seeds` but not `significant_all_seeds` is a direction with weak
    evidence; an arm that is neither is noise, however large its mean.
    """
    artifact_dir = Path(args.artifact_dir)
    suffixes = {'search': SEARCH_SEED_SUFFIXES, 'search2': SEARCH2_SEED_SUFFIXES}.get(
        args.ladder, SEED_SUFFIXES)
    frames = []
    for suffix in suffixes:
        path = artifact_dir/f'{args.tag}_contrasts{suffix}.csv'
        if not path.exists():
            print(f'  missing {path.name} -- skipping this seed')
            continue
        frame = pd.read_csv(path)
        frame['seed_suffix'] = suffix or '_seed12345'
        frames.append(frame)
    if not frames:
        raise SystemExit(f'no per-seed contrast tables under {artifact_dir}')
    if len(frames) < len(suffixes):
        print(f'WARNING: combining {len(frames)} seeds of {len(suffixes)}. A verdict from '
              f'fewer than three is not what C2-F-061 asks for.')

    everything = pd.concat(frames, ignore_index=True)
    rows = []
    for (arm, metric, scope), group in everything.groupby(['arm', 'metric', 'scope'], sort=True):
        deltas = group['delta_pp'].to_numpy(dtype=float)
        pvalues = group['p_value'].to_numpy(dtype=float)
        rows.append(dict(
            arm=arm, metric=metric, scope=scope, n_seeds=int(group.shape[0]),
            delta_mean=float(np.mean(deltas)), delta_min=float(np.min(deltas)),
            delta_max=float(np.max(deltas)), p_max=float(np.max(pvalues)),
            same_sign_all_seeds=bool(np.all(deltas > 0) or np.all(deltas < 0)),
            significant_all_seeds=bool(np.all(pvalues < 0.05)),
            # The spread against the effect. C2-F-061's halves had spreads 6.6x and 8x the quantity
            # they estimated, which is the shape of a contrast that is not there.
            spread_over_mean=(float(np.ptp(deltas)/abs(np.mean(deltas)))
                              if np.mean(deltas) else np.nan),
            ))
    table = pd.DataFrame(rows).sort_values(['metric', 'scope', 'delta_mean'], ascending=False)
    path = artifact_dir/(f'{args.tag}_seed_summary'
                         + ('' if args.ladder == 'main' else f'_{args.ladder}') + '.csv')
    table.to_csv(path, index=False)

    settled = table[(table.scope == 'aggregate') & (table.metric == 'operating_point')
                    & table.same_sign_all_seeds & table.significant_all_seeds]
    reference_arm = {'search': 'lean', 'search2': 'core'}.get(args.ladder, 'base')
    print(f'(reference arm: {reference_arm})')
    print(f'\n{len(frames)} seeds combined -> {path}')
    print(f'\narms settled on the operating point (same sign AND p < 0.05 at every seed):')
    if settled.empty:
        print('  none')
    for _, row in settled.sort_values('delta_mean', ascending=False).iterrows():
        print(f'  {row["arm"]:24s} {row["delta_mean"]:+7.2f} pp  '
              f'[{row["delta_min"]:+.2f}, {row["delta_max"]:+.2f}]  p <= {row["p_max"]:.3g}')
    unsettled = table[(table.scope == 'aggregate') & (table.metric == 'operating_point')
                      & ~(table.same_sign_all_seeds & table.significant_all_seeds)]
    print(f'\narms NOT settled ({len(unsettled)}): '
          f'{", ".join(sorted(unsettled["arm"])) if len(unsettled) else "none"}')
    return 0


# ---------------------------------------------------------------------------------------------
# stage: export-fit -- build the training frame where the pool is, carry it to where the model is
# ---------------------------------------------------------------------------------------------
def run_export_fit(args):
    """Assemble and subsample a fit frame from a pool, and write it as one parquet.

    **This is what makes a full-scale fit possible without moving 122 GB.** S12 session 1 fitted on
    the Benchmark B *slice*, which is 196 `fom-train` crystals; the benchmark itself has about
    11 000, and the difference is the leading explanation for C2-F-130 -- the arm that drops the
    whole structural family winning by 7.30 pp, against C2-F-040's −1.675 pp. The pool cannot come
    to the laptop and the report pool cannot go to the cluster, so the *design matrix* travels:
    assembled and subsampled where the candidates are, fitted and reported where the retained pool
    is.

    Two frames, subsampled differently and for different reasons:

    * the **fit** rows keep `--n-negatives` incorrect candidates a pattern, to rebalance a 0.03 %
      base rate into something a tree can learn from;
    * the **calibration** rows keep `--calibration-negatives`, many more, because the calibrator's
      job is to state a prior and it needs the negative tail to state it from. Both carry
      `sampling_weight` and `fit_weight`, and `fit_one` uses the first for the fit and the second
      for the calibrator (C2-F-127).

    At Benchmark B's scale this is about 4 M fit rows and 8 M calibration rows -- a couple of
    gigabytes, against 122.
    """
    pool = Path(args.fit_pool)
    entries = FomBenchmark.load_entries(pool)
    covariates = FomCombiner.entry_covariates(entries)
    fit_ids, cal_ids = split_ids(entries, args.train_split, HOLDOUT_FRACTION, SEED)
    print(f'{len(fit_ids)} fit crystals, {len(cal_ids)} calibration crystals from {pool}')

    union = tuple(dict.fromkeys(
        tuple(FomCombiner.DEFAULT_GROUPS)
        + tuple(group for _, extra, _, _ in ARMS for group in extra)))
    if args.groups:
        # An escape hatch for a pool whose optional sidecars are not written yet. The arms that
        # need a missing group then skip themselves and say so in the fit table, which is a stated
        # absence rather than a gap -- but the arms that decide the structural question
        # (`base` against `drop_structural`) need only the default groups, so a core-only export is
        # a useful thing to be able to produce.
        union = tuple(name.strip() for name in args.groups.split(',') if name.strip())
        unknown = [name for name in union if name not in FomCombiner.FEATURE_GROUPS]
        if unknown:
            raise SystemExit(f'unknown feature group(s) {unknown}; '
                             f'known: {list(FomCombiner.FEATURE_GROUPS)}')
    print(f'groups: {"+".join(union)}')
    out_dir = Path(args.out_dir or args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for label, ids, negatives in (('fit', fit_ids, args.n_negatives),
                                  ('cal', cal_ids, args.calibration_negatives)):
        started = time.perf_counter()
        frames = load_fit_frames(pool, entries, ids, union, negatives, SEED, covariates)
        frame = pd.concat(frames, ignore_index=True)
        path = out_dir/f'{args.tag}_{label}_frame{args.suffix}.parquet'
        frame.to_parquet(path, index=False)
        correct = int(FomMetrics.as_bool(frame['is_correct']).sum())
        written[label] = dict(path=str(path), rows=int(frame.shape[0]), correct=correct,
                              negatives_per_cell=negatives,
                              megabytes=round(path.stat().st_size/1e6, 1))
        print(f'  {label:4s} {frame.shape[0]:>10,} rows, {correct:>7,} correct, '
              f'{written[label]["megabytes"]:.0f} MB ({time.perf_counter() - started:.0f} s)',
              flush=True)
        del frames, frame

    (out_dir/f'{args.tag}_export_meta{args.suffix}.json').write_text(json.dumps(dict(
        pool=str(pool), commit=_commit(), groups=list(union), train_split=args.train_split,
        holdout_fraction=HOLDOUT_FRACTION, split_seed=SEED, frames=written,
        ), indent=2, sort_keys=True), encoding='utf-8')
    print(f'\nwrote {len(written)} frames to {out_dir}')
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
    # Arms from more than one ladder in ONE pool pass. `core` and `lean` were fitted by the
    # search ladders and live under their own model directories, and the shipped model is `core`
    # -- so a calibration table that can only see `fom_combiner_c2/` reports the ECE of a 29-feature
    # model nobody will ship. A second pass over 43 M candidates to add one arm is 30 minutes for
    # nothing; loading both directories costs one glob.
    arms = {}
    for directory in [args.models_dir] + list(args.also_models_dir or ()):
        for name, combiner in load_arms(directory, args.fit_seed, args.arms).items():
            if name in arms:
                raise SystemExit(
                    f'arm {name!r} was fitted in two of the given model directories. They would '
                    f'be different models under one label; pass --arms to pick one.')
            arms[name] = combiner
    # C2-Q-028's guard, in the one place in this driver that computes a per-candidate statistic.
    # An ECE on a subsampled pool is the calibration of a population that does not exist, and the
    # failure is silent -- S10c's within-band control read either side of chance depending on which
    # pool it was computed on (C2-F-111).
    FomMetrics.check_candidate_statistic(REPORT_POOL, 'a reliability table and its ECE')
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
                        choices=('fit', 'reduce', 'analyse', 'skew', 'calibration',
                                 'export-fit', 'combine', 'cost', 'transfer'))
    parser.add_argument('--models-dir', default=str(MODELS_DIR))
    parser.add_argument('--also-models-dir', nargs='*', default=None,
                        help='Further model directories the calibration stage loads arms from, '
                             'so one pool pass covers arms fitted by different ladders')
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
    parser.add_argument('--fit-pool', default=str(FIT_POOL),
                        help='Which pool export-fit assembles from. Defaults to the Benchmark B '
                             'slice; point it at the full benchmark on the cluster')
    parser.add_argument('--fit-frame', default=None,
                        help='Path to a `*_fit_frame.parquet` written by --stage export-fit. Its '
                             '`_cal_frame` sibling is read beside it. Given this, --stage fit does '
                             'not touch a pool at all')
    parser.add_argument('--ladder', choices=('main', 'search', 'search2'), default='main',
                        help="'main' asks whether each feature FAMILY earns its place; 'search' "
                             'starts from the answer -- the sixteen features left once the '
                             'structural family is dropped -- and removes one column at a time, '
                             'which is what the acceptance gate means by justifying a cut')
    parser.add_argument('--transfer-arm', choices=tuple(DROP_SETS), default='core',
                        help='Which feature set the transfer and cost stages measure. Defaults\n'
                             'to the 14-feature core the search settled on, NOT the 29-feature\n'
                             'base -- pass --models-dir the ladder that holds it')
    parser.add_argument('--report-pool', default=None,
                        help='Pool the cost stage prices on. Defaults to the report pool')
    parser.add_argument('--cost-lattice', default='mP',
                        help='Which lattice to price on. Low-symmetry by default: these costs are '
                             'dominated by the reference-line pass and its size is a property of '
                             'the lattice, so a cubic block would flatter every row')
    parser.add_argument('--cost-rows', type=int, default=4000)
    parser.add_argument('--cost-batch', type=int, default=2_000_000,
                        help='Rows the model is priced on for the amortised figure. A tree has a '
                             'fixed per-call cost that a small block does not amortise, and '
                             'scoring runs over whole pools')
    parser.add_argument('--groups', default=None,
                        help='Comma-separated feature groups for export-fit, overriding the union '
                             'every arm needs. Use it when a pool lacks an optional sidecar; the '
                             'arms needing that group then skip themselves and record why')
    parser.add_argument('--out-dir', default=None,
                        help='Where export-fit writes. Defaults to --artifact-dir')
    parser.add_argument('--calibration-negatives', type=int, default=400,
                        help='Incorrect candidates a pattern kept in the CALIBRATION frame. Larger '
                             'than --n-negatives because a calibrator states a prior and needs the '
                             'negative tail to state it from')
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
            'skew': run_skew, 'calibration': run_calibration,
            'export-fit': run_export_fit, 'combine': run_combine,
            'cost': run_cost, 'transfer': run_transfer}[args.stage](args)


if __name__ == '__main__':
    raise SystemExit(main())
