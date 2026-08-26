"""S08: one learned score over the whole figure-of-merit zoo.

PLAN section 3 writes the target as `fit - null + prior`. S06 and S07 between them settled which
of those three terms is still open. S07 supplied `- null` exactly -- the calibration provably
works, equalising the incorrect-candidate distribution across lattices by a factor of 25 to 200 --
and *ranking on it is worse than the raw merit*, 1 126 entries lost against 1 gained on M20
(F-076). The reason is Shirley's 1980 answer to Snyder, now measured: M20's volume and symmetry
dependence was carrying the **prior** as well as the null, and removing the null removed both. So
this module owns the prior, and it is the only term left.

Two numbers scope it, and neither is the one the handoff's gate quotes:

  * `M_sym` already beats M20 by 44% of headroom with no learning at all (F-066). A gate written
    against raw M20 is therefore clearable by emitting an existing merit, so every number here is
    reported against raw M20 *and* raw `M_sym` (DWMM, 2026-08-18, STATUS section 6).
  * A **perfect** combiner over all twenty-one merits reaches 0.7587 top-10 against `M_sym`'s
    0.6986 (F-070). **The headroom is 6.0 points**, and the merits that supply them are `n_over`
    and `max_gap` -- the over-prediction counters, which rank well and cannot be thresholded at
    all, which is precisely what a combiner is for.

Features are FOM-derived and structural only. S09 was abandoned so no process signal exists, and
S10/S12/S13 are later phases whose outputs must not be pulled in here -- doing so would confound
"do these merits combine" with "does a learned prior help", which are separate questions asked at
different times (S08 handoff, phase note).

**Four things this module refuses to use, and why each is a real hazard rather than a formality:**

  * anything derived from the truth (`is_correct` aside, which is the target): the labels
    `is_off_by_two`, `xnn_distance_to_truth`, `volume_ratio_to_truth`, every `*_true` column, and
    the two strata `dominant_zone` and `zone_count_min`, which METRICS.md defines from the *true*
    cell and which read like ordinary geometry;
  * `condition_bundle` and every condition parameter (`q2_error_multiplier`, `n_contaminants`,
    `n_dropout*`, `second_phase_*`). These are properties of the synthetic generator and are not
    knowable at inference. They are strata, not features;
  * `chi2_fixed`, which uses the repo's global sigma model -- the same model the synthetic
    generator uses, so it is the leakage path F-008 names and it will look excellent here and
    will not transfer (PROTOCOL section 3 rule 4);
  * `M_nn`, which reproduces M20 exactly at s = 1 (asserted in S01's tests). Including both is a
    collinear pair, not two features.

`bic` and `chi2_entrywise` estimate sigma in-sample, which rule 4 permits with a validated
estimator, so they are their own group and reported with and without.

**R1 bounds the scaled features.** The pool is censored at M20 >= 5 by `prune_below_m20`,
lattice-dependently (F-049, F-068), and every S07 normaliser except the two `analytic` ones was
fitted on those censored negatives. The bound belongs in the results document, not only here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics

# The seventeen sigma-free merits of the zoo. `M_nn` and `chi2_fixed` are excluded for the reasons
# in the module docstring; `bic` and `chi2_entrywise` are in-sample and live in their own group.
# `nll_exponential` is kept deliberately: F-025 established it is not a figure of merit and ranks
# backwards cross-lattice, but F-074 then measured 0.4407 within a lattice, so its uselessness is a
# scale artefact on top of a weak ranker rather than an absence of signal. A tree can use it or
# ignore it, and the importance table says which.
RAW_MERITS = (
    'M20', 'Minfo', 'M_tilde', 'M_rev', 'M_sym', 'M_wu', 'M_star', 'M_star_corrected', 'M_1',
    'M_info_clipped', 'null_tail_nll', 'F_N_q', 'M_werner_frac', 'X_N', 'n_over', 'max_gap',
    'nll_exponential',
    )

# PROTOCOL section 3 rule 4: sigma is never assumed known, but an in-sample estimate is allowed if
# the estimator is validated. Q7 has not validated one, so these are reported with and without
# rather than folded into the headline.
IN_SAMPLE_MERITS = ('bic', 'chi2_entrywise')

# Kept as an explicit map rather than an omission, so a later session sees the reason without
# having to reconstruct it from a diff.
EXCLUDED_MERITS = {
    'M_nn': 'reproduces M20 exactly at s = 1 (S01); a collinear pair, not two features',
    'chi2_fixed': "the repo's global sigma model -- the generator's own, so F-008's leakage path",
    'chi2_taupin_scale': 'an in-sample sigma estimate, not a merit',
    'chi2_fixed_pvalue': 'monotone in chi2_fixed; same leakage',
    'chi2_entrywise_pvalue': 'monotone in chi2_entrywise',
    }

# Candidate-side structure. Every one of these is available at inference from the candidate and
# its peak list; none of them touches the truth. Split by *where it comes from*, because the
# loader has to ask three different parquets for them and asking the wrong one is an ArrowInvalid
# rather than a wrong number -- but only if the split is explicit.
POOL_STRUCTURAL = ('n_peaks', 'n_indexed', 'hkl_ref_length', 'n_entering', 'final_rank')
FEATURE_MATRIX_STRUCTURAL = (
    'N_cal', 'zone_dominance', 'V_over_Vcrit', 'delta_dewolff61', 'n_dewolff61', 'M_werner_max',
    )
DERIVED_STRUCTURAL = ('log_volume', 'q2_max', 'n_peaks_available')
STRUCTURAL_NUMERIC = POOL_STRUCTURAL + FEATURE_MATRIX_STRUCTURAL + DERIVED_STRUCTURAL

# `spacegroup` holds a diffraction symbol -- an extinction group, ~151 of them -- not a single
# space group, and Smith & Snyder 1979 measured the M20 dependence as being on that rather than on
# the Bravais lattice alone (F-011). `lattice_system` is omitted: it is a function of
# `bravais_lattice`, so it is a duplicated split, not a second feature.
STRUCTURAL_CATEGORICAL = ('bravais_lattice', 'spacegroup')

# A candidate's plausibility is relative to its competitors, and these are computed over the
# *pooled cross-lattice* entry, which is what `run.py` actually ranks. Label-free and legitimate at
# inference. They make the model a ranker, so its probability is calibrated and interpreted per
# entry -- stated rather than left implicit (S08 handoff, "Pitfalls").
#
# The four reference orderings are M20 (the incumbent), `M_sym` (the S06 winner) and the two
# over-prediction counters F-070 names as the source of S08's entire headroom.
CONTEXT_MERITS = (('M20', True), ('M_sym', True), ('n_over', False), ('max_gap', False))
CONTEXT_STATISTICS = ('rank', 'gap_to_best', 'z')

# S10's predictive merits, as their own droppable group. Not in DEFAULT_GROUPS: S08's model is
# done and reported, and adding a feature to its default would silently redefine every number in
# STATUS section 2. The group exists so S10 can measure what it adds by fitting the same
# architecture twice and pairing the two.
#
# A subset of the thirty columns the CV matrix carries, not all of them. `contiguous` and `random`
# differ only in whether the held-out peaks are adjacent and correlate above 0.99, so including
# both would be one feature counted twice; the diagnostics (n_scored, n_voided, max_leverage) are
# for reading the result rather than for the model. `is_*` is here because the *pair* is the
# feature -- what a tree can use is that the in-sample and held-out statistics disagree.
CV_MERITS = (
    'cv_M20__random', 'cv_M__random', 'cv_tail_nll__random', 'cv_raw__random', 'cv_chi2__random',
    'cv_M20__high_q', 'cv_M__high_q', 'cv_tail_nll__high_q',
    'is_M', 'is_tail_nll',
    'ho_M20', 'ho_M', 'ho_tail_nll',
    )
# `is_M20` is deliberately absent: it reproduces the pool's own `M20` to 1e-12, so including it
# would add a column that is exactly collinear with one the model already has. It is built and
# reported because it is the round-trip gate, not because it is a feature.

# S11 block B, as its own droppable group, on the same terms as `cv`: not in DEFAULT_GROUPS, so
# S08's numbers keep meaning what they meant. Three columns out of one `assign_lines` pass.
#
# `asg_sigma` is Taupin 1988's reduced chi-square -- the absolute misfit, normalised by the number
# of free cell parameters. F-131 measured it at +2.82 pp of AUC over M20 + Minfo on 10 of 14
# lattices, and it is *not* a merit on its own: alone it scores 0.615, twenty-two points below
# M20. A complement to a merit that already carries the volume and symmetry scale is what a fit
# term should look like.
#
# The two posterior summaries average to -0.41 pp and are kept for one measured interaction: mC
# goes 0.805 -> 0.864 when they sit beside `asg_sigma` (F-131), and mC is a hard-stratum lattice
# and S08's second-worst. The ablation is what decides whether they stay.
ASSIGNMENT_MERITS = ('asg_sigma', 'asg_post_n', 'asg_post_l')

# S11 block A, likewise. The candidate's claimed (volume, Bravais) pair read against what the peak
# list implies -- PLAN section 3's prior term, handed to the model explicitly for the first time.
#
# Three things about the shape, each of which is a handoff pitfall made concrete:
#
#   * `prior_joint` reads the *joint* table at the claimed pair rather than the two marginals.
#     F-117 measured that difference at 37% of the volume error, because the volume marginal
#     silently inherits the volume scale of whichever lattice the model happens to believe.
#   * The lattice distribution ships as fourteen columns, never an argmax. F-106 measured the
#     centring head delivering 1.0% of its available information under ten interior holes, so a
#     hard label from it would be noise presented as fact; a distribution says so itself.
#   * `prior_base_rate` gives back what class-balanced training removed. F-086 says the base rate
#     is where S08's gain came from, and block A is trained balanced. Estimated on `fom-train`
#     rows only -- see `base_rate_by_lattice`.
#
# The eight entry-level columns (the fourteen `p_*`, the two entropies) are constant within an
# entry, so they cannot reorder candidates inside one; they act on the cross-entry threshold,
# which is exactly what the operating point measures and the top-10 does not.
PRIOR_CLAIMED = (
    'prior_joint', 'prior_joint_system', 'prior_joint_centring', 'prior_joint_n_free',
    'prior_joint_high_symmetry', 'prior_branch_lp', 'prior_bravais_lp', 'prior_joint_margin',
    'prior_branch_rank', 'prior_bravais_rank',
    )
PRIOR_ENTRY = tuple(
    f'prior_bravais_p_{code}' for code in
    ('cP', 'cI', 'cF', 'tP', 'tI', 'hP', 'hR', 'oP', 'oC', 'oF', 'oI', 'mP', 'mC', 'aP')
    ) + ('prior_branch_entropy', 'prior_bravais_entropy')
PRIOR_MERITS = PRIOR_CLAIMED + PRIOR_ENTRY + ('prior_base_rate',)

SCALER_METHODS = ('analytic', 'z', 'rank')
# S04 Phase 2, 2026-08-26. Two rival encodings of the same physics, as their own droppable groups
# so each can be measured by a retrained paired arm rather than an importance table (PROTOCOL
# section 8). `counts` is DWMM's proposed replacement for the `spacegroup` categorical; `delta` is
# what S04 found actually carries the signal -- how far the merit moved when the group was applied,
# with the look-elsewhere count it is a selected maximum over (C2-F-034, C2-F-035).
SYMMETRY_COUNTS = ('n_absent_extra', 'n_absent_extra_in_range', 'f_absent_extra',
                   'n_groups_searched')
SYMMETRY_DELTA = ('delta_M20', 'delta_M_rev', 'n_groups_searched')

FEATURE_GROUPS = ('raw', 'structural', 'context', 'in_sample', 'cv', 'assignment',
                  'prior', 'counts', 'delta')
DEFAULT_GROUPS = ('raw', 'structural', 'context')

# The groups whose columns are joined in from a directory of per-bundle parquets rather than read
# out of the pool or the feature matrix. Kept as one map so `combiner_frames` has a single join
# path instead of one branch per group.
EXTERNAL_GROUPS = {
    'cv': ('cv', CV_MERITS),
    'assignment': ('assignment', ASSIGNMENT_MERITS),
    'prior': ('prior', PRIOR_MERITS),
    'counts': ('symmetry', SYMMETRY_COUNTS),
    'delta': ('symmetry', SYMMETRY_DELTA),
    }

# Enforced by `check_no_leakage`, which every fit and every score call runs. A deny-list rather
# than trust in the allow-list, because the two failure modes are different: the allow-list catches
# a feature nobody added, the deny-list catches one somebody added on purpose.
FORBIDDEN_COLUMNS = frozenset({
    'is_correct', 'is_off_by_two', 'is_degenerate', 'xnn_distance_to_truth',
    'volume_ratio_to_truth', 'dominant_zone', 'zone_count_min', 'split', 'condition_bundle',
    'q2_error_multiplier', 'n_contaminants', 'contaminant_bias', 'n_dropout',
    'n_dropout_achieved', 'second_phase_lines', 'second_phase_bias', 'second_phase_partner',
    'cluster', 'is_hard', 'volume_decile',
    })
FORBIDDEN_SUFFIX = '_true'

# `run_fom_zoo_eval.POOL_COLUMNS` plus the four the combiner needs and S06 did not: `n_indexed`,
# `hkl_ref_length` and `final_rank` are features, and `volume_ratio_to_truth` is dropped because
# it is a label. Projection-only either way.
POOL_COLUMNS = tuple(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + (
    'Minfo', 'M20', 'volume', 'n_peaks', 'spacegroup', 'n_entering', 'n_indexed',
    'hkl_ref_length', 'final_rank',
    )

# Entry-level covariates. `q2_max` is stored nowhere and is the largest observed q^2 the candidate
# was actually scored against, which differs between cubic (10 peaks) and everything else (20).
ENTRY_COLUMNS = ('entry_id', 'condition_bundle', 'q2_obs', 'n_peaks_available', 'split')

_EPSILON = 1e-12

# Reserved code for a category unseen at fit time. Real categories start at 1, so an extinction
# group the model has never met lands in its own bin rather than silently colliding with the
# first one -- which is the difference between a model that says "I do not know this symmetry"
# and one that says "this is P1".
_UNSEEN_CODE = 0


# ---------------------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------------------
def active_merits(groups):
    """The merit columns in play for a given set of feature groups."""
    merits = []
    if 'raw' in groups:
        merits.extend(RAW_MERITS)
    if 'in_sample' in groups:
        merits.extend(IN_SAMPLE_MERITS)
    return tuple(merits)


# NOT PORTED (S04 Phase 2, 2026-08-26): `load_scalers` and `scaled_names`, and with them the
# whole `scaled` feature group. They are the only reason this module imported `FomNull`, which is
# 1 011 lines this step does not otherwise use, and campaign 1's fitted scalers are not on disk to
# load anyway. PROTOCOL section 3 rule 10 -- take what the step uses; see `CHERRY_PICK.md`.
#
# The cost is bounded and known from campaign 1's own table: the `scaled` arm reaches an operating
# point of 0.633436 against the `raw` arm's 0.628497, so working from `raw+structural+context`
# gives up 0.49 pp of level. Every arm here is an ablation of the same base, so the CONTRAST -- the
# only thing S04 Phase 2 is measuring -- is unaffected.


def context_names():
    """The per-entry context columns `add_context` appends."""
    names = ['ctx_pool_size']
    for merit, _ in CONTEXT_MERITS:
        names.extend(f'ctx_{merit}_{statistic}' for statistic in CONTEXT_STATISTICS)
    return tuple(names)


def feature_specification(groups=DEFAULT_GROUPS, scalers=(), drop=()):
    """(names, categorical_names) for one choice of feature groups, less any dropped columns.

    The ablation in campaign 1's results document is a group-drop, so the group is the unit here
    rather than the column.

    `drop` exists because that unit turned out to be too coarse to answer the question it was used
    to answer. Campaign 1's `drop_structural` arm removes **sixteen** features at once --
    `spacegroup`, `bravais_lattice`, `final_rank`, `n_entering`, `log_volume` and eleven more -- and
    its 2.23 pp operating-point cost was then read as the cost of the symmetry prior. It is the cost
    of the family. Naming a single column here is what lets S04 Phase 2 measure `spacegroup` on its
    own (C2-F-039, C2-Q-013).
    """
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f'unknown feature group(s) {unknown}; known: {list(FEATURE_GROUPS)}')

    merits = active_merits(groups)
    names = []
    if 'raw' in groups:
        names.extend(RAW_MERITS)
    if 'in_sample' in groups:
        names.extend(IN_SAMPLE_MERITS)
    if 'cv' in groups:
        names.extend(CV_MERITS)
    if 'assignment' in groups:
        names.extend(ASSIGNMENT_MERITS)
    if 'prior' in groups:
        names.extend(PRIOR_MERITS)
    if 'counts' in groups:
        names.extend(SYMMETRY_COUNTS)
    if 'delta' in groups:
        names.extend(SYMMETRY_DELTA)
    categorical = ()
    if 'structural' in groups:
        names.extend(STRUCTURAL_NUMERIC)
        categorical = STRUCTURAL_CATEGORICAL
        names.extend(categorical)
    if 'context' in groups:
        names.extend(context_names())

    drop = set(drop)
    unknown_drop = drop - set(names)
    if unknown_drop:
        raise ValueError(f'cannot drop {sorted(unknown_drop)}: not in this feature set')
    seen, ordered = set(), []
    for name in names:
        if name not in seen and name not in drop:
            seen.add(name)
            ordered.append(name)
    check_no_leakage(ordered)
    return tuple(ordered), tuple(name for name in categorical if name not in drop)


def affordable_features(names, allowed_merits):
    """`names` with every column that depends on a merit outside `allowed_merits` removed.

    The inner-loop question S14 inherits. Gate condition 3 asks for a variant within 2x `get_M20`;
    `S06_zoo_cost.csv` prices `M_sym` at 24x, `n_over` at 29x and `max_gap` at 30x, and F-070 says
    those three are the whole of S08's headroom -- so the budget is spent computing the features
    before the model runs at all. This builds the variant that stays inside it, so what the budget
    costs is measured rather than asserted.

    **None of those three prices still holds.** F-172 and F-173 rewrote all three for
    bit-identical output: on the frozen pool `M_sym` is now 1.7x `get_M20`, and `n_over`/`max_gap`
    2.8x, against the 24x, 29x and 30x recorded here. `affordable_features` still builds the
    variant inside a stated budget, which is the durable part; what has changed is which merits
    fall outside it, and the answer now has to come from a fresh measurement
    (`tools/repro_fom_zoo.py cost`) rather than from the S06 table.
    """
    allowed = set(allowed_merits)
    known = (set(RAW_MERITS) | set(IN_SAMPLE_MERITS) | set(CV_MERITS)
             | set(ASSIGNMENT_MERITS) | set(PRIOR_MERITS))
    kept = []
    for name in names:
        # A CV column carries its fold scheme after the separator, and an assignment or prior
        # column has no scaled form at all, so both are their own merit; match the whole name
        # before the '__' split that unpacks a scaled one.
        merit = name if name in known else name.split('__', 1)[0]
        if name.startswith('ctx_') and name != 'ctx_pool_size':
            body = name[len('ctx_'):]
            merit = next((candidate for candidate, _ in CONTEXT_MERITS
                          if body.startswith(f'{candidate}_')), merit)
        if merit in known and merit not in allowed:
            continue
        kept.append(name)
    return tuple(kept)


def check_no_leakage(names):
    """Raise if any proposed feature is truth-derived or is a property of the generator.

    Cheap, run on every fit and every score, and the only guard between this task and the two
    ways it could produce a large number that means nothing.
    """
    offenders = sorted({name for name in names
                        if name in FORBIDDEN_COLUMNS or name.endswith(FORBIDDEN_SUFFIX)})
    if offenders:
        raise ValueError(
            f'feature list contains columns that are truth-derived or unavailable at inference: '
            f'{offenders}')
    return True


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------
def entry_covariates(entries):
    """Per (entry_id, condition_bundle): the observed q^2 ceiling, at both peak counts.

    `q2_max` is the largest observed q^2 the candidate was scored against, and that is 10 peaks
    for a cubic candidate and 20 for everything else (SCHEMA.md), so it is a property of the pair
    rather than of the entry. Both are computed here and the candidate picks one.
    """
    q2_obs = entries['q2_obs'].to_numpy()
    cubic = np.empty(len(q2_obs))
    full = np.empty(len(q2_obs))
    for index, values in enumerate(q2_obs):
        peaks = np.asarray(values, dtype=np.float64)
        cubic[index] = peaks[:10].max() if peaks.size else np.nan
        full[index] = peaks[:20].max() if peaks.size else np.nan
    return pd.DataFrame({
        'entry_id': entries['entry_id'].to_numpy(),
        'condition_bundle': entries['condition_bundle'].to_numpy(),
        'q2_max_cubic': cubic,
        'q2_max_full': full,
        'n_peaks_available': entries['n_peaks_available'].to_numpy(),
        })


def add_context(frame):
    """Append the per-entry context columns, over the pooled cross-lattice entry.

    Called on the **whole** bundle before any negative subsampling: a candidate's rank among its
    competitors is a property of the pool the indexer produced, and computing it after thinning
    the negatives would invent a feature that does not exist at inference.
    """
    codes, keys = FomMetrics._group_codes(frame['entry_id'].to_numpy(),
                                          frame['condition_bundle'].to_numpy())
    n_groups = len(keys)
    # The same total order `reduce_pool` ranks by, so `ctx_M20_rank` is the rank the metrics
    # module would report and not a second, nearly-identical ordering.
    lattice_order = pd.Categorical(frame['bravais_lattice'].to_numpy(),
                                   categories=FomMetrics.BRAVAIS_LATTICES).codes.astype(np.int64)
    candidate_id = frame['candidate_id'].to_numpy(dtype=np.int64)
    columns = {'ctx_pool_size': np.bincount(codes, minlength=n_groups)[codes].astype(np.float64)}
    for merit, higher_is_better in CONTEXT_MERITS:
        values = frame[merit].to_numpy(dtype=np.float64)
        oriented = values if higher_is_better else -values
        stats = pd.DataFrame({'code': codes, 'value': oriented}).groupby('code')['value'].agg(
            ['max', 'median', 'std']).reindex(np.arange(n_groups))
        best = stats['max'].to_numpy()[codes]
        median = stats['median'].to_numpy()[codes]
        spread = stats['std'].to_numpy()[codes]
        # NaN sorts last, so a non-finite merit ranks behind every finite one rather than winning.
        sort_key = -oriented.copy()
        sort_key[np.isnan(oriented)] = np.inf
        columns[f'ctx_{merit}_rank'] = FomMetrics._ranks_within(
            codes, sort_key, lattice_order, candidate_id).astype(np.float64)
        columns[f'ctx_{merit}_gap_to_best'] = oriented - best
        columns[f'ctx_{merit}_z'] = (oriented - median)/(spread + _EPSILON)
    return frame.assign(**columns)


def _merge_external(pool, keys, group, prefix, columns, directory, bundle):
    """Join one externally-generated feature group onto a bundle's pool, on the four zoo keys.

    **Left, not inner, and that is a decision rather than a default.** An entry with no surplus
    peaks has no `ho_` columns; a candidate whose assignment pass found no reference line has no
    `asg_` ones. Dropping those rows would silently change the denominator every metric is
    computed over. HistGradientBoosting takes NaN natively, and "this candidate had no such
    statistic" is a real inference-time state rather than missing data.

    One path for `cv`, `assignment` and `prior` because they differ only in a filename prefix, and
    three copies of this is how the second one drifts from the first.
    """
    if directory is None:
        raise ValueError(f"feature group '{group}' needs a directory; its matrix is not in "
                         'feature_dir')
    wanted = [column for column in columns if column not in pool.columns]
    if not wanted:
        return pool
    frame = pd.read_parquet(Path(directory)/f'{prefix}_{bundle}.parquet')
    wanted = [column for column in wanted if column in frame.columns]
    if not wanted:
        return pool
    return pool.merge(frame[keys + wanted], on=keys, how='left', validate='1:1')


def combiner_frames(benchmark_dir, feature_dir, bundles, keep_entry_ids, covariates, scalers,
                    groups=DEFAULT_GROUPS, cv_dir=None, assignment_dir=None, prior_dir=None,
                    symmetry_dir=None):
    """Yield one fully-assembled frame per bundle: pool, features, S07 scales, entry, context.

    The same assembly serves training and evaluation, which is the point -- a context feature
    computed one way at fit time and another at score time is the classic silent failure here.

    Not materialised: `evaluate` consumes shards one at a time and a caller that needs many
    passes over the same split holds the *result* of this generator, as `run_fom_null.load_split`
    does, rather than re-reading 2.3 GB of parquet per pass.
    """
    keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
    external_dirs = {'cv': cv_dir, 'assignment': assignment_dir, 'prior': prior_dir,
                     'counts': symmetry_dir, 'delta': symmetry_dir}
    merits = active_merits(groups)
    # The context features are computed for four reference orderings whatever the feature groups
    # are, so their merits are always read even when the raw group is dropped for an ablation.
    context = {merit for merit, _ in CONTEXT_MERITS}
    wanted_features = sorted((set(merits) | set(FEATURE_MATRIX_STRUCTURAL) | context)
                             - set(EXCLUDED_MERITS))
    for bundle in bundles:
        pool = FomBenchmark.load_candidates(
            benchmark_dir, bundles=[bundle], columns=list(POOL_COLUMNS),
            )
        pool = pool.loc[pool['entry_id'].isin(keep_entry_ids)]
        if not pool.shape[0]:
            continue
        missing = [column for column in wanted_features if column not in pool.columns]
        if missing:
            features = pd.read_parquet(
                Path(feature_dir)/f'features_{bundle}.parquet', columns=keys + missing,
                )
            # Inner: the feature matrix covers fom-train and fom-dev only, fom-test having never
            # been computed (it is sealed until S15). 1:1 on the four zoo keys.
            pool = pool.merge(features, on=keys, how='inner', validate='1:1')
        for group, (prefix, columns) in EXTERNAL_GROUPS.items():
            if group in groups:
                pool = _merge_external(pool, keys, group, prefix, columns,
                                       external_dirs.get(group), bundle)
        pool = pool.merge(covariates, on=['entry_id', 'condition_bundle'], how='left',
                          validate='m:1')
        pool['log_volume'] = np.log(pool['volume'].to_numpy(dtype=np.float64))
        pool['q2_max'] = np.where(pool['n_peaks'].to_numpy() <= 10,
                                  pool['q2_max_cubic'].to_numpy(),
                                  pool['q2_max_full'].to_numpy())
        pool = add_context(pool)
        if pool.shape[0]:
            yield pool.reset_index(drop=True)


def subsample_negatives(frame, n_negatives, seed):
    """Every positive, and at most `n_negatives` incorrect candidates per (entry, bundle).

    Each entry has of order one correct candidate among several hundred, so an unthinned fit
    spends almost all of its capacity on negatives that no threshold will ever reach. The prior
    shift this introduces is undone by the isotonic step, which is fitted on an *unsubsampled*
    calibration split -- which is why the two must not be the same rows.
    """
    if n_negatives is None:
        return frame
    correct = FomMetrics.as_bool(frame['is_correct'])
    codes, keys = FomMetrics._group_codes(frame['entry_id'].to_numpy(),
                                          frame['condition_bundle'].to_numpy())
    rng = np.random.default_rng(seed)
    # One draw per row, then keep the `n_negatives` smallest within each group: a group-wise
    # sample without a Python loop over seven thousand groups.
    draw = rng.random(frame.shape[0])
    draw[correct] = -1.0
    order = np.lexsort((draw, codes))
    position = np.arange(order.size) - np.searchsorted(
        codes[order], np.arange(len(keys)), side='left')[codes[order]]
    within = np.empty(order.size, dtype=np.int64)
    within[order] = position
    n_correct = np.bincount(codes[correct], minlength=len(keys))
    keep = correct | (within - n_correct[codes] < n_negatives)
    return frame.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------
class FomCombiner:
    """A calibrated P(correct) over the zoo, and the fast path that scores a candidate batch.

    One global model with the Bravais lattice as a categorical feature, not fourteen per-lattice
    models: the rare lattices cannot support their own (oF has two entries in the whole CNRS
    benchmark and cI has four), and S07's scaled inputs are already lattice-conditioned so the
    global model is not fighting a scale difference. The per-lattice variant is fitted anyway and
    reported as the ablation that justifies the choice.
    """

    def __init__(self, model=None, names=(), categorical=(), categories=None,
                 groups=DEFAULT_GROUPS, objective='pointwise', calibrators=None, meta=None):
        self.model = model
        self.names = tuple(names)
        self.categorical = tuple(categorical)
        self.categories = categories or {}
        self.groups = tuple(groups)
        self.objective = objective
        self.calibrators = calibrators or {}
        self.meta = meta or {}

    # -- feature assembly -------------------------------------------------------------
    def design_matrix(self, frame):
        """The (n_candidates, n_features) float array the estimator sees, in `names` order."""
        check_no_leakage(self.names)
        missing = [name for name in self.names if name not in frame.columns]
        if missing:
            raise KeyError(f'frame is missing {len(missing)} feature column(s): {missing[:8]}')
        matrix = np.empty((frame.shape[0], len(self.names)), dtype=np.float64)
        for index, name in enumerate(self.names):
            if name in self.categories:
                lookup = self.categories[name]
                codes = pd.Series(frame[name].to_numpy()).map(lookup).to_numpy()
                matrix[:, index] = np.where(pd.isna(codes), _UNSEEN_CODE,
                                            codes).astype(np.float64)
            else:
                matrix[:, index] = frame[name].to_numpy(dtype=np.float64)
        return matrix

    @property
    def categorical_indices(self):
        return [self.names.index(name) for name in self.categorical if name in self.names]

    # -- fitting ----------------------------------------------------------------------
    @classmethod
    def fit(cls, frames, groups=DEFAULT_GROUPS, scalers=(), objective='pointwise', seed=12345,
            drop=(), **params):
        """Fit on an iterable of assembled frames. `objective` is 'pointwise' or 'lambdarank'."""
        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        if not frames:
            raise ValueError('no frames to fit on')
        names, categorical = feature_specification(groups, scalers, drop=drop)
        categories = {name: {value: code + 1 for code, value in
                             enumerate(sorted(pd.unique(pd.concat(
                                 [frame[name] for frame in frames]).astype(str))))}
                      for name in categorical}
        combiner = cls(names=names, categorical=categorical, categories=categories,
                       groups=groups, objective=objective)

        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        matrix = combiner.design_matrix(frame)
        target = FomMetrics.as_bool(frame['is_correct']).astype(np.int32)
        if objective == 'pointwise':
            combiner.model = _fit_pointwise(matrix, target, combiner.categorical_indices, seed,
                                            **params)
        elif objective == 'lambdarank':
            combiner.model = _fit_lambdarank(matrix, target, frame,
                                             combiner.categorical_indices, seed, **params)
        else:
            raise ValueError(f"objective must be 'pointwise' or 'lambdarank', got {objective!r}")
        combiner.meta = dict(objective=objective, seed=int(seed), groups=list(groups),
                             dropped=sorted(drop),
                             n_rows=int(frame.shape[0]), n_positive=int(target.sum()),
                             n_features=len(names), params={k: v for k, v in params.items()})
        return combiner

    def fit_calibrators(self, frames, minimum=200):
        """Per-Bravais-lattice isotonic regression, on rows the model was not fitted on.

        The S08 handoff says to calibrate on `fom-dev`; PROTOCOL section 8 forbids reporting a
        number on the split it was fitted on, and an ECE quoted on `fom-dev` after fitting the
        calibrator there is exactly that. Fitted on a held-out slice of `fom-train` instead, which
        also absorbs the prior shift the negative subsampling introduced.

        A lattice with fewer than `minimum` rows falls back to the pooled calibrator rather than
        fitting noise -- oF has two entries in the whole CNRS benchmark.
        """
        from sklearn.isotonic import IsotonicRegression

        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        raw = self.raw_score(frame)
        target = FomMetrics.as_bool(frame['is_correct']).astype(np.float64)
        lattice = frame['bravais_lattice'].to_numpy()

        def knots(values, labels):
            fitted = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            fitted.fit(values, labels)
            return (np.asarray(fitted.X_thresholds_, dtype=np.float64),
                    np.asarray(fitted.y_thresholds_, dtype=np.float64))

        calibrators = {'__pooled__': knots(raw, target)}
        for name in np.unique(lattice):
            mask = lattice == name
            if int(mask.sum()) >= minimum and np.unique(target[mask]).size > 1:
                calibrators[str(name)] = knots(raw[mask], target[mask])
        self.calibrators = calibrators
        self.meta['n_calibration_rows'] = int(frame.shape[0])
        self.meta['calibrated_lattices'] = sorted(set(calibrators) - {'__pooled__'})
        return self

    # -- scoring ----------------------------------------------------------------------
    def raw_score(self, frame):
        """The estimator's uncalibrated output, higher meaning more likely correct."""
        return self.predict_batch(self.design_matrix(frame))

    def predict_batch(self, matrix):
        """Uncalibrated score for a prepared (n_candidates, n_features) matrix.

        The inner-loop entry point: no pandas, no column lookup, one call into the estimator. See
        the results document for why the *features* rather than this call are what decides whether
        an inner-loop variant is affordable.
        """
        if self.objective == 'lambdarank':
            return np.asarray(self.model.predict(matrix), dtype=np.float64)
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=np.float64)

    def score(self, frame):
        """Calibrated P(correct) per row. This is the callable `FomMetrics.evaluate` wants."""
        raw = self.raw_score(frame)
        if not self.calibrators:
            return raw
        out = np.empty(raw.size, dtype=np.float64)
        lattice = frame['bravais_lattice'].to_numpy()
        pooled = self.calibrators['__pooled__']
        for name in np.unique(lattice):
            mask = lattice == name
            thresholds, targets = self.calibrators.get(str(name), pooled)
            out[mask] = np.interp(raw[mask], thresholds, targets)
        return out

    @property
    def score_columns(self):
        """Columns `evaluate` must project for `score` to work."""
        return tuple(sorted(set(self.names) | {'bravais_lattice'}))

    # -- persistence ------------------------------------------------------------------
    def save(self, directory):
        """joblib for the estimator, npz for the calibrators, JSON for the specification.

        The estimator has to be pickled -- unlike `FomNull`, a fitted gradient-boosting model does
        not reduce to a lookup table -- but the *specification* does not, and keeping it in JSON
        is what makes a mismatched load fail loudly instead of scoring the wrong columns.
        """
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory/'model.joblib')
        arrays = {}
        for name, (thresholds, targets) in self.calibrators.items():
            arrays[f'{name}__x'] = thresholds
            arrays[f'{name}__y'] = targets
        np.savez_compressed(directory/'calibrators.npz', **arrays)
        with open(directory/'specification.json', 'w', encoding='utf-8') as handle:
            json.dump(dict(names=list(self.names), categorical=list(self.categorical),
                           categories={name: {str(k): int(v) for k, v in lookup.items()}
                                       for name, lookup in self.categories.items()},
                           groups=list(self.groups), objective=self.objective, meta=self.meta),
                      handle, indent=2)
        return directory

    @classmethod
    def load(cls, directory):
        import joblib

        directory = Path(directory)
        with open(directory/'specification.json', encoding='utf-8') as handle:
            specification = json.load(handle)
        arrays = np.load(directory/'calibrators.npz')
        names = {key.rsplit('__', 1)[0] for key in arrays.files}
        calibrators = {name: (arrays[f'{name}__x'], arrays[f'{name}__y']) for name in names}
        return cls(model=joblib.load(directory/'model.joblib'),
                   names=specification['names'], categorical=specification['categorical'],
                   categories=specification['categories'], groups=specification['groups'],
                   objective=specification['objective'], calibrators=calibrators,
                   meta=specification.get('meta', {}))


class PerLatticeCombiner:
    """Fourteen models, one per Bravais lattice -- the architecture S08 argued against.

    The handoff asserts one global model with the lattice as a categorical feature is preferable,
    on two grounds: the rare lattices cannot support their own (oF has two entries in the whole
    CNRS benchmark and cI has four), and S07's scaled inputs are already lattice-conditioned so the
    global model is not fighting a scale difference. It then asks for the per-lattice model as an
    ablation *to justify the choice*, which is the part that turns an argument into a measurement.

    Each sub-model is fitted only on its own lattice's candidates, so `bravais_lattice` is constant
    within it and the model cannot learn the cross-lattice prior at all -- which is the point.
    Every sub-model gets its own isotonic, because the pooled ranking needs the fourteen outputs on
    one probability scale; without that this arm would be measuring an arbitrary scale mismatch
    rather than an architecture.

    A lattice with too few positives to fit falls back to the global model, and which ones did is
    recorded rather than absorbed -- the fallback count is itself part of the answer.
    """

    def __init__(self, models=None, fallback=None, meta=None):
        self.models = models or {}
        self.fallback = fallback
        self.meta = meta or {}

    @classmethod
    def fit(cls, frames, fallback, groups=DEFAULT_GROUPS, scalers=(), seed=12345,
            min_positive=25, **params):
        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        correct = FomMetrics.as_bool(frame['is_correct'])
        models, fell_back, sizes = {}, [], {}
        for lattice in sorted(frame['bravais_lattice'].unique()):
            mask = (frame['bravais_lattice'] == lattice).to_numpy()
            positives = int(correct[mask].sum())
            sizes[str(lattice)] = dict(rows=int(mask.sum()), positives=positives)
            if positives < min_positive:
                fell_back.append(str(lattice))
                continue
            models[str(lattice)] = FomCombiner.fit(
                frame.loc[mask].reset_index(drop=True), groups=groups, scalers=scalers,
                objective='pointwise', seed=seed, **params)
        return cls(models=models, fallback=fallback,
                   meta=dict(n_models=len(models), fell_back=fell_back, sizes=sizes,
                             min_positive=int(min_positive), groups=list(groups)))

    def fit_calibrators(self, frames, minimum=200):
        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        for lattice, model in self.models.items():
            mask = (frame['bravais_lattice'] == lattice).to_numpy()
            if mask.any():
                model.fit_calibrators(frame.loc[mask].reset_index(drop=True), minimum=minimum)
        return self

    def score(self, frame):
        out = np.full(frame.shape[0], np.nan)
        lattices = frame['bravais_lattice'].to_numpy()
        for lattice in np.unique(lattices):
            mask = lattices == lattice
            model = self.models.get(str(lattice), self.fallback)
            out[mask] = model.score(frame.loc[mask])
        return out

    @property
    def names(self):
        model = next(iter(self.models.values()), self.fallback)
        return model.names

    @property
    def score_columns(self):
        model = next(iter(self.models.values()), self.fallback)
        return model.score_columns


class DistilledCombiner:
    """The combiner as three numpy matmuls, which is STATUS Q4 answered rather than assumed.

    Q4 asks whether a small MLP evaluated as plain matmuls is actually slower than `get_M20` in the
    inner loop. PLAN §4's guess was that it is not, and that the "networks are too slow" experience
    came from per-candidate or ONNX-session overhead rather than arithmetic. F-085 measured the
    600-tree ensemble at 2.46x `get_M20`, so the tree form misses the 2x budget *before any feature
    is computed* -- which makes this the half of the cost problem that a distillation can address.

    Fitted on the teacher's own output rather than on the labels: the target is a ranking, the
    teacher already encodes one, and regressing on it keeps the ordering the teacher learned
    instead of re-learning it from 0.9%-prevalence labels with far less capacity.

    Held as plain arrays, so scoring is `relu(relu(X@W0 + b0)@W1 + b1)@W2 + b2` and nothing is
    unpickled, no session is created, and there is no per-candidate Python. Non-finite features are
    imputed with the training median, which the tree did not need and an MLP does.
    """

    def __init__(self, names=(), categorical=(), categories=None, weights=(), biases=(),
                 centre=None, scale=None, median=None, calibrators=None, meta=None):
        self.names = tuple(names)
        self.categorical = tuple(categorical)
        self.categories = categories or {}
        # The student needs its own per-lattice isotonic, for the same reason the teacher does and
        # not because it inherits one: a regression fitted to the teacher's *output* reproduces the
        # ordering but not the scale, and an uncalibrated score has no threshold that meets a
        # false-positive budget -- measured as an operating point of exactly zero before this was
        # added, against a top-10 of 0.65.
        self.calibrators = calibrators or {}
        self.weights = [np.asarray(weight, dtype=np.float64) for weight in weights]
        self.biases = [np.asarray(bias, dtype=np.float64) for bias in biases]
        self.centre = None if centre is None else np.asarray(centre, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.median = None if median is None else np.asarray(median, dtype=np.float64)
        self.meta = meta or {}

    # `design_matrix`, `categorical_indices` and `score_columns` are identical to the teacher's, so
    # they are inherited rather than duplicated -- a distilled model that assembled its columns
    # differently from its teacher would be measuring something else.
    design_matrix = FomCombiner.design_matrix
    categorical_indices = FomCombiner.categorical_indices
    score_columns = FomCombiner.score_columns
    fit_calibrators = FomCombiner.fit_calibrators
    score = FomCombiner.score

    @classmethod
    def distil(cls, teacher, frames, hidden=(32, 16), seed=12345, max_iter=60, sample=400000):
        """Fit an MLP to reproduce `teacher`'s ranking, and keep only its arrays."""
        from sklearn.neural_network import MLPRegressor

        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        matrix = teacher.design_matrix(frame)
        target = teacher.raw_score(frame)

        rng = np.random.default_rng(seed)
        if sample and matrix.shape[0] > sample:
            keep = rng.choice(matrix.shape[0], size=sample, replace=False)
            matrix, target = matrix[keep], target[keep]

        median = np.nanmedian(matrix, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        clean = _impute(matrix, median)
        centre = clean.mean(axis=0)
        scale = clean.std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardised = (clean - centre)/scale

        model = MLPRegressor(hidden_layer_sizes=tuple(hidden), activation='relu',
                             random_state=seed, max_iter=max_iter, early_stopping=True,
                             n_iter_no_change=5, validation_fraction=0.1)
        model.fit(standardised, target)
        student = cls(names=teacher.names, categorical=teacher.categorical,
                      categories=teacher.categories,
                      weights=model.coefs_, biases=model.intercepts_,
                      centre=centre, scale=scale, median=median,
                      meta=dict(hidden=list(hidden), seed=int(seed), n_rows=int(matrix.shape[0]),
                                n_features=len(teacher.names), teacher=teacher.objective,
                                n_layers=len(model.coefs_)))
        student.meta['train_correlation'] = float(
            np.corrcoef(student.predict_batch(_undo(standardised, centre, scale)), target)[0, 1])
        return student

    def predict_batch(self, matrix):
        """`relu` between layers, identity at the output. One matmul per layer, no Python loop."""
        activations = (_impute(matrix, self.median) - self.centre)/self.scale
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = activations@weight + bias
            if index < len(self.weights) - 1:
                np.maximum(activations, 0.0, out=activations)
        return activations.ravel()

    def raw_score(self, frame):
        return self.predict_batch(self.design_matrix(frame))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {'centre': self.centre, 'scale': self.scale, 'median': self.median}
        for name, (thresholds, targets) in self.calibrators.items():
            arrays[f'calibrator_{name}__x'] = thresholds
            arrays[f'calibrator_{name}__y'] = targets
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f'weight_{index}'] = weight
            arrays[f'bias_{index}'] = bias
        np.savez_compressed(directory/'distilled.npz', **arrays)
        with open(directory/'distilled.json', 'w', encoding='utf-8') as handle:
            json.dump(dict(names=list(self.names), categorical=list(self.categorical),
                           categories={name: {str(k): int(v) for k, v in lookup.items()}
                                       for name, lookup in self.categories.items()},
                           meta=self.meta), handle, indent=2)
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        arrays = np.load(directory/'distilled.npz')
        with open(directory/'distilled.json', encoding='utf-8') as handle:
            specification = json.load(handle)
        n_layers = int(specification['meta']['n_layers'])
        names = {key[len('calibrator_'):].rsplit('__', 1)[0] for key in arrays.files
                 if key.startswith('calibrator_')}
        calibrators = {name: (arrays[f'calibrator_{name}__x'], arrays[f'calibrator_{name}__y'])
                       for name in names}
        return cls(names=specification['names'], categorical=specification['categorical'],
                   categories=specification['categories'],
                   weights=[arrays[f'weight_{index}'] for index in range(n_layers)],
                   biases=[arrays[f'bias_{index}'] for index in range(n_layers)],
                   centre=arrays['centre'], scale=arrays['scale'], median=arrays['median'],
                   calibrators=calibrators, meta=specification['meta'])


def _impute(matrix, median):
    """Non-finite entries replaced by the training median, without copying when nothing is bad."""
    bad = ~np.isfinite(matrix)
    if not bad.any():
        return matrix
    out = matrix.copy()
    out[bad] = np.take(median, np.nonzero(bad)[1])
    return out


def _undo(standardised, centre, scale):
    return standardised*scale + centre


def _fit_pointwise(matrix, target, categorical_indices, seed, **params):
    """The headline model: gradient-boosted trees, not a network (S08 handoff, "Model choice")."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    settings = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
                    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                    random_state=seed)
    settings.update(params)
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_indices or None, **settings)
    model.fit(matrix, target)
    return model


def _fit_lambdarank(matrix, target, frame, categorical_indices, seed, **params):
    """PLAN section 4's assumption A11: is per-entry ranking a better objective than pointwise?

    lightgbm is an optional, training-only dependency and is imported here rather than at module
    scope, exactly as `FomNull._distil_gbm` imports its regressor -- nothing on the inference path
    may acquire a dependency, and a saved combiner must load without it.
    """
    import lightgbm

    codes, keys = FomMetrics._group_codes(frame['entry_id'].to_numpy(),
                                          frame['condition_bundle'].to_numpy())
    order = np.argsort(codes, kind='stable')
    sizes = np.bincount(codes, minlength=len(keys))
    settings = dict(objective='lambdarank', n_estimators=400, learning_rate=0.06,
                    num_leaves=63, min_child_samples=40, reg_lambda=1.0, random_state=seed,
                    label_gain=[0, 1], verbose=-1)
    # The caller tunes in sklearn's vocabulary, because the pointwise model is the headline and
    # the two objectives have to be compared at the same capacity rather than at whatever each
    # library happens to default to.
    translation = {'max_iter': 'n_estimators', 'max_leaf_nodes': 'num_leaves',
                   'min_samples_leaf': 'min_child_samples', 'l2_regularization': 'reg_lambda'}
    settings.update({translation.get(key, key): value for key, value in params.items()})
    model = lightgbm.LGBMRanker(**settings)
    model.fit(matrix[order], target[order], group=sizes[sizes > 0],
              categorical_feature=categorical_indices or 'auto')
    return model
