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
# ---------------------------------------------------------------------------------------
# S12's cut, 2026-09-01. Seventeen raw merits become seven.
# ---------------------------------------------------------------------------------------
# These are exactly `FomBenchmark.REDUCED_MERIT_COLUMNS`, and that identity is worth stating: they
# are the seven the negative subsampler ranked on, so they are the only merits whose input
# distribution Benchmark B's retention rule was built to preserve (C2-F-077).
#
# The cut is DWMM's instruction -- "there are also too many traditional FOMs used; we should just
# use what was effective and discard the rest as noise" -- and its evidence is S00's merit audit,
# which reached it from per-entry outcomes rather than from an importance table: ten merits rank
# below a constant score, `M_nn` is analytically M20, and three calls reach 99.5 % of the
# twenty-two-merit union oracle. **But an audit is a prior, not a verdict.** PROTOCOL section 8
# settles a merit cut with a retrained paired arm and nothing else, so the arm that restores the
# dropped seven (`plus_dropped_merits`) is what licenses this tuple, and until it has run this is
# a hypothesis with a comment attached.
RAW_MERITS = ('M20', 'M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap')

# What campaign 1's `raw` group held, kept so the restoring arm is a one-line change and so the
# cut is legible as a decision rather than as an absence. `Minfo` is NOT here: SCHEMA.md forbids
# ranking, fitting or reporting on the stored column outright, so it cannot enter any arm.
CAMPAIGN1_RAW_MERITS = (
    'M20', 'M_tilde', 'M_rev', 'M_sym', 'M_wu', 'M_star', 'M_star_corrected', 'M_1',
    'M_info_clipped', 'null_tail_nll', 'F_N_q', 'M_werner_frac', 'X_N', 'n_over', 'max_gap',
    'nll_exponential',
    )

# S00 left three merits on probation and S09 declined to decide them: they are outside
# `RANK_EXACT_MERITS`, so their rank on Benchmark B is optimistic by an unmeasured amount, and its
# decision of 2026-08-31 records the drop as reversible on a fully retained pool. That pool exists,
# and `FomBenchmark.structural_features` emits all three for 42 microseconds a candidate because
# they need reference lines it has already built. One arm settles all three.
PROBATION_MERITS = ('M_wu', 'M_1', 'F_N_q')

# C2-F-102: `X_N` rebuilt on S13's per-peak assignment posterior goes from 0.2252 to 0.6164 of
# top-10 and its ties collapse from 3 832 to 1. The same treatment made `n_over` and `max_gap`
# WORSE -- a predicted line need not be observed at all, so normalising over peaks forces a
# distribution where "none" is the right answer -- and the hard counts still add more union-oracle
# headroom than the soft ones (+0.692 pp against +0.377, both together +0.818). So the soft form
# is an addition to the hard three, never a replacement, and only `X_N_soft` is carried.
SOFT_MERITS = ('X_N_soft',)

# S10c's recommendation, and the only hold-out column that survived it. All three posterior columns
# were beaten within narrow M20 bands by `ho_M20` (0.682 against 0.587-0.593) and none ships
# (C2-F-112). `ho_M20` itself loses to in-sample M20 as a RANKER by 39.66 pp (C2-F-104), which is
# a statement about it as a score and not about it as a feature -- that is a complementarity
# question and this step's `plus_ho_M20` arm is what answers it. `n5` is a 25-peak pattern, the
# budget real instruments supply (C2-F-103, C2-R-016).
HOLDOUT_MERITS = ('ho_M20__n5',)

# PROTOCOL section 3 rule 4: sigma is never assumed known, but an in-sample estimate is allowed if
# the estimator is validated. Q7 has not validated one, so these are reported with and without
# rather than folded into the headline.
IN_SAMPLE_MERITS = ('bic', 'chi2_entrywise')

# Kept as an explicit map rather than an omission, so a later session sees the reason without
# having to reconstruct it from a diff.
EXCLUDED_MERITS = {
    'M_nn': 'reproduces M20 exactly at s = 1 (S01); a collinear pair, not two features',
    # Campaign 2's cut, from S00's merit audit sections 3 and 4. Each of these ranks BELOW a
    # constant score on per-entry outcomes, and nine of the ten win no entry any other merit
    # misses (C2-F-012). Listed rather than omitted so the reason survives the diff.
    'Minfo': 'below a constant score, 0.041 top-10 against a 0.2352 tie-break floor; and '
             'SCHEMA.md forbids ranking, fitting or reporting on the stored column at all',
    'M_star': 'below a constant score (S00 section 3)',
    'M_star_corrected': 'below a constant score (S00 section 3)',
    'M_info_clipped': 'below a constant score, and redundant with Minfo (S00 section 3)',
    'null_tail_nll': 'below a constant score (S00 section 3)',
    'M_werner_frac': 'below a constant score (S00 section 3); the Werner quantity that carries '
                     'signal is V_over_Vcrit, which is in the structural group',
    'nll_exponential': 'a negative control, never a feature (S00 section 6)',
    'werner_strict': 'M20 with a strictness gate; strictly worse, not a distinct merit',
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
#
# `N_cal` and `N_cal_full` are two different counts and are carried under two names deliberately.
# `N_cal` is `get_M_rev_sym`'s support -- reference lines in [q_I, q_N], the quantity the M_rev
# floor tests, which fires on 63 % of cP group evaluations (C2-F-114) -- and comes from the merit
# sidecar. `N_cal_full` is `compute_all`'s, over [0, q_N], and is what campaign 1's feature of that
# name was. Measured on real mP candidates they agree on 0.07 % of rows, so a feature set that
# joins the sidecar and calls it `N_cal` is not carrying campaign 1's column.
FEATURE_MATRIX_STRUCTURAL = (
    'N_cal', 'N_cal_full', 'zone_dominance', 'V_over_Vcrit', 'delta_dewolff61', 'n_dewolff61',
    'M_werner_max',
    )
# `pool_size_full` replaces `ctx_pool_size`: the survivor count before subsampling, which is the
# same number on a thinned pool and a fully retained one. See FORBIDDEN_COLUMNS.
DERIVED_STRUCTURAL = ('log_volume', 'q2_max', 'n_peaks_available', 'pool_size_full')
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

# S14's input blocks (2026-09-05), as their own groups so each is a retrained paired arm rather
# than a line in an importance table. They are NOT the campaign-1 `prior` / `assignment` groups
# above, which keep their meaning and their flat-directory join path for the S12 record:
#
#   * `prior_entry`   -- `PRIOR_ENTRY` as before, but now read through the prior's SUPPORT: the
#                        fourteen probabilities are renormalised over the lattices the model was
#                        trained on and NaN outside it, so an untrained class can never be read
#                        as a probability (S14 gate condition 1; F-117 point 4).
#   * `prior_volume`  -- E[log V | lattice] per lattice from the joint table, DWMM's "volume for
#                        each BL"; the reading F-117 measured at 37 % of the volume error.
#   * `prior_claimed` -- the joint read at the candidate's OWN claimed pair, its margin to the
#                        table's mode, and whether the claim is inside the support at all. The
#                        only block-A column that can reorder candidates within an entry.
#   * `assignment_peaks` -- the twenty per-peak assignment posteriors themselves (C2-F-074), not
#                        the two summaries campaign 1 fitted and found worth -0.41 pp.
#   * `assignment_sigma` -- the posterior's own denominator, log-scaled (F-131: +2.82 pp beside
#                        two columns, a substitute for the posterior beside seventy-eight).
#
# The entry-level groups arrive through `neural_covariates` (one row per (entry, bundle)); the
# per-candidate groups through the `neural_inputs` sidecar. `run_fom_neural_inputs.py` writes both.
PRIOR_VOLUME = tuple(
    f'prior_logv_{code}' for code in
    ('cP', 'cI', 'cF', 'tP', 'tI', 'hP', 'hR', 'oP', 'oC', 'oF', 'oI', 'mP', 'mC', 'aP')
    )
PRIOR_SUPPORT_FLAG = 'prior_in_support'
PRIOR_CLAIMED_C2 = ('prior_joint', 'prior_joint_margin', PRIOR_SUPPORT_FLAG)
ASSIGNMENT_PEAKS = tuple(f'asg_p{index:02d}' for index in range(20))
ASSIGNMENT_SIGMA = ('asg_sigma',)
# DWMM's redirect (decision 2026-09-05): the prior's two summaries -- one predicted volume for the
# pattern (E[log V] over the branches, lattices summed out) and one predicted number of free cell
# parameters (E[dof] from the `n_free` head) -- enter the combiner as per-candidate RATIOS against
# the candidate's own volume and lattice. Entry-level summaries ride the covariates; the ratios
# are derived in `combiner_frames_c2` from them and the candidate row, so no sidecar pass is
# needed. The volume ratio exists in two readings, both fitted as arms: against the lattice
# marginal, and against E[log V | claimed lattice] (the F-117 reading, NaN outside the support).
PRIOR_SUMMARY = ('prior_logv_marginal', 'prior_dof_expected')
PRIOR_RATIO_VOLUME_MARGINAL = ('prior_volume_ratio_marginal',)
PRIOR_RATIO_VOLUME_CLAIMED = ('prior_volume_ratio_claimed',)
PRIOR_RATIO_DOF = ('prior_dof_ratio',)
# Free cell parameters per Bravais lattice: cubic 1, tetragonal/hexagonal/rhombohedral 2,
# orthorhombic 3, monoclinic 4, triclinic 6 (`PriorNetwork.N_FREE_OF`).
DOF_OF_LATTICE = {'cP': 1, 'cI': 1, 'cF': 1, 'tP': 2, 'tI': 2, 'hP': 2, 'hR': 2,
                  'oP': 3, 'oC': 3, 'oF': 3, 'oI': 3, 'mP': 4, 'mC': 4, 'aP': 6}

NEURAL_ENTRY_GROUPS = ('prior_entry', 'prior_volume')
NEURAL_CANDIDATE_GROUPS = ('prior_claimed', 'assignment_peaks', 'assignment_sigma')
PRIOR_RATIO_GROUPS = ('prior_ratio_volume_marginal', 'prior_ratio_volume_claimed',
                      'prior_ratio_dof')
NEURAL_GROUPS = NEURAL_ENTRY_GROUPS + NEURAL_CANDIDATE_GROUPS + PRIOR_RATIO_GROUPS
NEURAL_GROUP_COLUMNS = {
    'prior_entry': PRIOR_ENTRY,
    'prior_volume': PRIOR_VOLUME,
    'prior_claimed': PRIOR_CLAIMED_C2,
    'assignment_peaks': ASSIGNMENT_PEAKS,
    'assignment_sigma': ASSIGNMENT_SIGMA,
    'prior_ratio_volume_marginal': PRIOR_RATIO_VOLUME_MARGINAL,
    'prior_ratio_volume_claimed': PRIOR_RATIO_VOLUME_CLAIMED,
    'prior_ratio_dof': PRIOR_RATIO_DOF,
    }

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
                  'prior', 'counts', 'delta', 'probation', 'soft', 'holdout', 'campaign1_raw',
                  ) + NEURAL_GROUPS
# S12's base feature space. `counts` joins the default because S04 Phase 2 settled the symmetry
# question in its favour -- the absence counts beat the 158-level categorical by +0.522 pp of
# operating point at p <= 0.004 at every fit seed (C2-F-041) -- so a campaign-2 model that omitted
# them would be an ablation rather than the headline. `spacegroup` is still reachable, and is
# dropped by column in the arms rather than deleted here, so `plus_spacegroup` is the absence of a
# drop rather than a second definition of the feature set.
DEFAULT_GROUPS = ('raw', 'structural', 'context', 'counts')

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
    # A leakage class campaign 2 has and campaign 1 did not: everything describing the RETENTION
    # RULE. Benchmark B keeps every correct candidate, the union of seven top-200 lists and 5 % of
    # the rest; the fully retained pool keeps everything. So `retained_reason` is very nearly the
    # label, `sampling_weight` is 1.0 for every correct row, and a model reading either learns the
    # subsampler and transfers to no pool at all -- including the one it is reported on.
    'sampling_weight', 'fit_weight', 'retained_reason', 'retained_by',
    # `ctx_pool_size` counts the candidates that SURVIVED RETENTION, so it is 8 206 on Benchmark B
    # where it is 26 734 on a fully retained pool of the same patterns -- a 3.3x shift between the
    # pool a model is fitted on and the pool it is scored on, in a feature, which no weight
    # repairs. Campaign 1 measured it as the worst feature of 78 and "actively harmful"; here it
    # is a correctness problem rather than a weak one. `pool_size_full` is the same quantity
    # without the skew -- the survivor count BEFORE subsampling, identical on both pools by
    # construction, and available at inference -- so it is carried in its place.
    'ctx_pool_size',
    # And the quantities that describe the generation run rather than the candidate: the prune
    # tested `m20_at_prune` and stored what it tested, `in_top_n` is a threshold on `final_rank`
    # that the dump applied, and the last three are constants of the run.
    'm20_at_prune', 'merit_at_prune', 'in_top_n', 'prune_threshold', 'downsample_radius',
    'assignment_threshold', 'q2_digest',
    })
FORBIDDEN_SUFFIX = '_true'

# `run_fom_zoo_eval.POOL_COLUMNS` plus the four the combiner needs and S06 did not: `n_indexed`,
# `hkl_ref_length` and `final_rank` are features, and `volume_ratio_to_truth` is dropped because
# it is a label. Projection-only either way.
POOL_COLUMNS = tuple(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + (
    'Minfo', 'M20', 'volume', 'n_peaks', 'spacegroup', 'n_entering', 'n_indexed',
    'hkl_ref_length', 'final_rank',
    # The two absence columns the pool DOES store. SCHEMA.md keeps these on the candidate row and
    # leaves `n_absent_extra_in_range` to be recomputed, so the counts group is half a projection
    # and half a sidecar -- which is easy to half-fix and then find a column silently absent.
    'n_absent_extra', 'n_groups_searched',
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
    if 'campaign1_raw' in groups:
        merits.extend(CAMPAIGN1_RAW_MERITS)
    if 'in_sample' in groups:
        merits.extend(IN_SAMPLE_MERITS)
    if 'probation' in groups:
        merits.extend(PROBATION_MERITS)
    if 'soft' in groups:
        merits.extend(SOFT_MERITS)
    if 'holdout' in groups:
        merits.extend(HOLDOUT_MERITS)
    return tuple(dict.fromkeys(merits))


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
    """The per-entry context FEATURES. Not everything `add_context` appends.

    `ctx_pool_size` is computed and is deliberately not a feature: it counts the candidates that
    survived retention, which is a different number on a thinned pool and a fully retained one, so
    it cannot be fitted on one and scored on the other. It stays in the frame as a diagnostic --
    the retention shift is worth being able to look at -- and `FORBIDDEN_COLUMNS` stops it being
    fitted on by accident. `pool_size_full` is the unskewed version and is in the structural group.

    **This changes S04 Phase 2's feature set by one column if its arms are ever re-run.** Its
    published contrasts are unaffected, being ablations of a common base that all carried it.
    """
    names = []
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
    if 'probation' in groups:
        names.extend(PROBATION_MERITS)
    if 'soft' in groups:
        names.extend(SOFT_MERITS)
    if 'holdout' in groups:
        names.extend(HOLDOUT_MERITS)
    if 'campaign1_raw' in groups:
        # The restoring arm. Everything campaign 1's `raw` group held, so the seven-merit cut is
        # licensed by a retrained paired arm rather than by S00's audit alone (PROTOCOL section 8).
        names.extend(CAMPAIGN1_RAW_MERITS)
    for group in NEURAL_GROUPS:
        if group in groups:
            names.extend(NEURAL_GROUP_COLUMNS[group])
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


# The only two columns a fit may be weighted by, and the guard is the point rather than the list.
#
# `tests/test_fom_combiner_arms.py::test_no_public_entry_point_takes_a_weight` forbids a weight on
# `FomMetrics.evaluate`, `entry_context`, `threshold_curve` and `select_threshold`. That guard is
# about **reweighting entries to a target lattice distribution** -- campaign 1 defaulted
# `evaluate(weights='cnrs')`, so a caller who omitted the argument silently reweighted every
# aggregate to the sealed benchmark's shape and discarded 43 % of its effective sample. PROTOCOL
# section 3 rules 1 and 6.
#
# This is the opposite object: a per-row **inverse retention probability** written by the generator,
# which undoes a thinning rather than imposing a population. `SCHEMA.md` requires it in bold and
# `METRICS.md` section 1 repeats it. Keeping the two apart structurally -- a fit may take a weight,
# nothing on the scoring or metrics path may -- is what stops the guard and the requirement being
# read as contradicting each other.
ALLOWED_WEIGHT_COLUMNS = ('sampling_weight', 'fit_weight')


def fit_weights(frame, column='fit_weight'):
    """The per-row fit weight, refusing any column that is not an inclusion probability.

    Raises rather than defaulting, on both an unknown column name and an absent one. A silently
    unweighted fit on a subsampled pool is the failure this exists to prevent, and it is invisible
    in every downstream number.
    """
    if column not in ALLOWED_WEIGHT_COLUMNS:
        raise ValueError(
            f'{column!r} is not an inclusion-probability weight; allowed: '
            f'{list(ALLOWED_WEIGHT_COLUMNS)}. A per-lattice or per-entry weight would be a '
            'reweighting of the population, which PROTOCOL section 3 rule 6 forbids.')
    if column not in frame.columns:
        raise ValueError(
            f'{column!r} is not in the frame. Project it in the loader rather than fitting '
            'unweighted: on a negatively subsampled pool an unweighted fit is biased and nothing '
            'downstream shows it.')
    values = frame[column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f'{column!r} carries a non-positive or non-finite weight')
    return values


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
    covariates = pd.DataFrame({
        'entry_id': entries['entry_id'].to_numpy(),
        'condition_bundle': entries['condition_bundle'].to_numpy(),
        'q2_max_cubic': cubic,
        'q2_max_full': full,
        'n_peaks_available': entries['n_peaks_available'].to_numpy(),
        })
    # The survivor count BEFORE subsampling, which is why it is read here rather than counted from
    # the rows: counting gives `ctx_pool_size`, which is the retained count and means different
    # things on a thinned pool and a full one. See FORBIDDEN_COLUMNS.
    if 'pool_size_full' in entries.columns:
        covariates['pool_size_full'] = entries['pool_size_full'].to_numpy(dtype=np.float64)
    return covariates


def add_prior_ratios(frame):
    """The two block-A ratio features, per candidate, from the entry-level prior summaries.

    Log ratios rather than ratios: a tree is invariant to the monotone transform, and the log
    form keeps a volume off by 10x and one off by 1/10x symmetric about zero. `prior_volume_ratio_*`
    is `log V_candidate - E[log V]`; `prior_dof_ratio` is the candidate lattice's free-parameter
    count over the prior's expected count. The claimed-lattice volume reading is NaN where the
    claimed lattice is outside the prior's support (cubic, for the shipped model), and so is
    recorded as such rather than imputed here -- the tree handles a missing value natively.
    """
    for name in PRIOR_SUMMARY:
        if name not in frame.columns:
            raise KeyError(f'{name} is not on the frame; re-run run_fom_neural_inputs.py '
                           f'--stage entries with a PriorNetwork that emits it')
    log_volume = np.log(frame['volume'].to_numpy(dtype=np.float64))
    lattice = frame['bravais_lattice'].to_numpy().astype(str)
    frame['prior_volume_ratio_marginal'] = (
        log_volume - frame['prior_logv_marginal'].to_numpy(dtype=np.float64))
    claimed = np.empty(frame.shape[0], dtype=np.float64)
    for code in np.unique(lattice):
        mask = lattice == code
        column = f'prior_logv_{code}'
        claimed[mask] = (frame.loc[mask, column].to_numpy(dtype=np.float64)
                         if column in frame.columns else np.nan)
    frame['prior_volume_ratio_claimed'] = log_volume - claimed
    dof = np.array([DOF_OF_LATTICE.get(code, np.nan) for code in lattice], dtype=np.float64)
    frame['prior_dof_ratio'] = dof/frame['prior_dof_expected'].to_numpy(dtype=np.float64)
    return frame


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
    # Computed, never fitted on: see `context_names`.
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


# Which sidecar directory under a Benchmark B pool supplies each feature group. `merits` and
# `structural` are unconditional -- the raw merits and the structural family are in every arm --
# and the rest are joined only when their group is active, because a hold-out sidecar is 88 columns
# and 11 GB and no arm but one reads a single column of it.
SIDECAR_DIRS = {
    'raw': 'merits',
    'structural': 'structural',
    'counts': 'structural',
    'probation': 'structural',
    'soft': 'merits_soft',
    'holdout': 'holdout_merits',
    'prior_claimed': 'neural_inputs',
    'assignment_peaks': 'neural_inputs',
    'assignment_sigma': 'neural_inputs',
    }
NEURAL_ENTRY_FILE = 'prior_entries.parquet'


def neural_covariates(pool, entries):
    """`entry_covariates` plus the entry-level block-A columns from `<pool>/neural_inputs/`.

    One row per (entry_id, condition_bundle), joined 1:1, and it RAISES when an entry in `entries`
    has no prior row: `combiner_frames_c2` merges covariates m:1 and left, so a missing row would
    become NaN in thirty columns for every candidate of that entry, with no symptom.
    """
    covariates = entry_covariates(entries)
    path = Path(pool)/SIDECAR_DIRS['prior_claimed']/NEURAL_ENTRY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f'{path} is missing: write it with run_fom_neural_inputs.py --stage entries')
    prior = pd.read_parquet(path)
    wanted = ['entry_id', 'condition_bundle'] + list(PRIOR_ENTRY) + list(PRIOR_VOLUME) \
        + [name for name in PRIOR_SUMMARY if name in prior.columns]
    absent = [name for name in wanted if name not in prior.columns]
    if absent:
        raise KeyError(f'{path} lacks {absent}')
    merged = covariates.merge(prior[wanted], on=['entry_id', 'condition_bundle'], how='left',
                              validate='1:1', indicator=True)
    unmatched = merged['_merge'] != 'both'
    if unmatched.any():
        example = merged.loc[unmatched, ['entry_id', 'condition_bundle']].head(5)
        raise KeyError(
            f'{int(unmatched.sum())} (entry, bundle) pairs in the entry table have no row in '
            f'{path}, e.g. {example.to_dict("records")}. Re-run the entries stage over this '
            f'entry table.')
    return merged.drop(columns=['_merge'])

# Read from the pool but not fitted on: the labels, the keys the reduction groups by, and the two
# weights. `check_no_leakage` forbids every one of them as a feature, which is the point -- they
# have to be in the frame and must not be in the design matrix.
CARRIED_NOT_FITTED = ('is_correct', 'is_off_by_two', 'sampling_weight', 'retained_reason',
                      'in_top_n', 'volume', 'spacegroup', 'lattice_system')


def combiner_frames_c2(pool, entries, groups=DEFAULT_GROUPS, bundles=None, keep_entry_ids=None,
                       covariates=None, holdout_n_extra=5, downcast=True):
    """One assembled frame per condition bundle, for a **Benchmark B** pool. A generator.

    `combiner_frames` reads campaign 1's layout -- a `features_{bundle}.parquet` matrix beside a
    single-file pool -- which Benchmark B does not have. This reads the campaign-2 layout instead:
    one candidate parquet per (bundle, lattice) with the features in sidecar directories beside
    them, joined by `FomBenchmark.bundle_frames` on the four zoo keys. Everything after the join is
    the same as `combiner_frames` does and is deliberately so, because a context feature computed
    one way at fit time and another at score time is the classic silent failure here.

    **One frame per bundle, all fourteen lattices together, and that is not negotiable.** The
    context features and the ranking are cross-lattice -- that is the problem `run.py` actually
    solves -- so a per-lattice frame would compute a different feature and reduce a different
    ranking (PROTOCOL section 10's worst anti-pattern).

    `downcast` puts the float columns in float32 after the context features are computed, which
    roughly halves a 14.5 M-row bundle. The context statistics are computed in float64 first, so
    this costs precision in the design matrix and none in the feature definition.
    """
    from mlindex.model_training import FomBenchmark

    merits = list(active_merits(groups))
    wanted = set(merits) | set(FEATURE_MATRIX_STRUCTURAL) | {merit for merit, _ in CONTEXT_MERITS}
    if 'counts' in groups:
        wanted |= set(FomBenchmark.ABSENCE_COLUMNS)
    if 'holdout' in groups:
        wanted |= {FomBenchmark.holdout_column('ho_M20', holdout_n_extra)}
    for group in NEURAL_CANDIDATE_GROUPS:
        if group in groups:
            wanted |= set(NEURAL_GROUP_COLUMNS[group])
    # NOT `wanted -= EXCLUDED_MERITS`. That set documents which merits campaign 2 cut from the
    # default feature space, and six of them are exactly what the `campaign1_raw` group restores --
    # so subtracting it here silently emptied the projection for the one arm that licenses the cut.
    # `_sidecar_projection` already drops a name no sidecar carries, which is the right place for
    # that decision: it is a fact about the data, not about the feature policy.

    # Every group `SIDECAR_DIRS` knows, not a hard-coded list: a directory added to the map and
    # not to a list here joined nothing and the arm skipped itself downstream (S14, 2026-09-05).
    directories = []
    for group in SIDECAR_DIRS:
        if group in groups or group in ('raw', 'structural'):
            directory = Path(pool)/SIDECAR_DIRS[group]
            if directory not in directories:
                directories.append(directory)

    pool_columns = [name for name in
                    tuple(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + POOL_COLUMNS + CARRIED_NOT_FITTED
                    if name is not None]
    if covariates is None:
        # The entry-level block-A columns ride on the covariates, so any caller asking for those
        # groups gets them without knowing where they live -- and gets `neural_covariates`'s
        # refusal if the prior table is missing or short.
        if any(group in groups for group in NEURAL_ENTRY_GROUPS + PRIOR_RATIO_GROUPS):
            covariates = neural_covariates(pool, entries)
        else:
            covariates = entry_covariates(entries)

    for frame in FomBenchmark.bundle_frames(
            pool, merit_dir=directories, columns=list(dict.fromkeys(pool_columns)),
            require_merits=True, merit_columns=sorted(wanted)):
        if bundles is not None and frame['condition_bundle'].iloc[0] not in set(bundles):
            continue
        if keep_entry_ids is not None:
            frame = frame.loc[frame['entry_id'].isin(keep_entry_ids)]
        if not frame.shape[0]:
            continue
        frame = frame.reset_index(drop=True)
        frame = frame.merge(covariates, on=['entry_id', 'condition_bundle'], how='left',
                            validate='m:1')
        frame['log_volume'] = np.log(frame['volume'].to_numpy(dtype=np.float64))
        frame['q2_max'] = np.where(frame['n_peaks'].to_numpy() <= 10,
                                   frame['q2_max_cubic'].to_numpy(),
                                   frame['q2_max_full'].to_numpy())
        if 'counts' in groups:
            # Derived here rather than stored: a stored ratio is a third column that can disagree
            # with its own numerator, and both of its operands are already on the row.
            in_range = frame['n_ref_in_range'].to_numpy(dtype=np.float64)
            frame['f_absent_extra'] = np.where(
                in_range > 0,
                frame['n_absent_extra_in_range'].to_numpy(dtype=np.float64)/np.maximum(in_range, 1),
                np.nan)
        if 'holdout' in groups:
            frame['ho_M20__n5'] = frame[
                FomBenchmark.holdout_column('ho_M20', holdout_n_extra)].to_numpy(dtype=np.float64)
        if any(group in groups for group in PRIOR_RATIO_GROUPS):
            frame = add_prior_ratios(frame)
        frame = add_context(frame)
        if downcast:
            floats = frame.select_dtypes(include=['float64']).columns
            frame[floats] = frame[floats].astype(np.float32)
        yield frame


def subsample_negatives(frame, n_negatives, seed, weight_column='sampling_weight'):
    """Every positive, and at most `n_negatives` incorrect candidates per (entry, bundle).

    Each entry has of order one correct candidate among several hundred, so an unthinned fit
    spends almost all of its capacity on negatives that no threshold will ever reach. The prior
    shift this introduces is undone by the isotonic step, which is fitted on an *unsubsampled*
    calibration split -- which is why the two must not be the same rows.

    **Writes `fit_weight`, and a fit must NOT use it.** It is the composition of two thinnings:
    the generator's, recorded in `sampling_weight`, and this one. It is the correct inverse-
    inclusion weight for estimating a pool-level quantity from the subsample, and it is verified
    unbiased -- over 20 seeds it recovers the true negative weight mass to +0.44 % with a standard
    deviation of 2.78 %.

    It is the wrong weight for a **fit**, and measurably so. The two thinnings are not the same
    kind of thing. The generator's is a bias to correct: it kept the highest-scoring wrong
    candidates preferentially, and `sampling_weight` undoes that. This one is a deliberate
    **rebalancing** -- the pool is 0.026 % correct and no gradient-boosted tree learns anything
    useful at that rate, so the negatives are thinned to bring it to about 1.7 %. Weighting it back
    restores 0.026 % and undoes the only reason the subsample exists. Measured on this pool:
    fitting on `fit_weight` costs **17.7 pp of top-10 and 43.1 pp of top-1** against fitting on
    `sampling_weight`, and drops the model below raw M20 (C2-F-127).

    So: **fit on `sampling_weight`, calibrate on `sampling_weight` over an unsubsampled split,
    and keep `fit_weight` for pool-level estimators.** The prior the rebalancing removes is what
    the isotonic step puts back, which is why the calibration rows must not be the fit rows.
    """
    if n_negatives is None:
        if weight_column is not None and weight_column in frame.columns:
            return frame.assign(fit_weight=frame[weight_column].to_numpy(dtype=np.float64))
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

    if weight_column is None or weight_column not in frame.columns:
        return frame.loc[keep].reset_index(drop=True)
    base = frame[weight_column].to_numpy(dtype=np.float64)
    n_negative = np.bincount(codes[~correct], minlength=len(keys))
    n_kept = np.bincount(codes[keep & ~correct], minlength=len(keys))
    # Positives are kept whole, so they carry the generator's weight unchanged.
    inflation = np.where(n_kept > 0, n_negative/np.maximum(n_kept, 1), 1.0)[codes]
    fit_weight = np.where(correct, base, base*inflation)
    return frame.loc[keep].assign(fit_weight=fit_weight[keep]).reset_index(drop=True)


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
            drop=(), weight_column=None, **params):
        """Fit on an iterable of assembled frames. `objective` is 'pointwise' or 'lambdarank'.

        `weight_column` names a per-row weight in the frames, and on a negatively subsampled pool
        it is not optional -- see `fit_weights`. `None` fits unweighted, which is right for a fully
        retained pool and is also the `unweighted_fit` control arm.
        """
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
        weights = None if weight_column is None else fit_weights(frame, weight_column)
        if objective == 'pointwise':
            combiner.model = _fit_pointwise(matrix, target, combiner.categorical_indices, seed,
                                            sample_weight=weights, **params)
        elif objective == 'lambdarank':
            combiner.model = _fit_lambdarank(matrix, target, frame,
                                             combiner.categorical_indices, seed,
                                             sample_weight=weights, **params)
        else:
            raise ValueError(f"objective must be 'pointwise' or 'lambdarank', got {objective!r}")
        combiner.meta = dict(objective=objective, seed=int(seed), groups=list(groups),
                             dropped=sorted(drop), weight_column=weight_column,
                             weight_sum=None if weights is None else float(weights.sum()),
                             n_rows=int(frame.shape[0]), n_positive=int(target.sum()),
                             n_features=len(names), params={k: v for k, v in params.items()})
        return combiner

    def fit_calibrators(self, frames, minimum=200, weight_column=None):
        """Per-Bravais-lattice isotonic regression, on rows the model was not fitted on.

        The S08 handoff says to calibrate on `fom-dev`; PROTOCOL section 8 forbids reporting a
        number on the split it was fitted on, and an ECE quoted on `fom-dev` after fitting the
        calibrator there is exactly that. Fitted on a held-out slice of `fom-train` instead, which
        also absorbs the prior shift the negative subsampling introduced.

        A lattice with fewer than `minimum` rows falls back to the pooled calibrator rather than
        fitting noise -- oF has two entries in the whole CNRS benchmark.

        `weight_column` matters here for the same reason it matters in `fit`, and more sharply: the
        calibrator's whole job is to state a prior, and on a subsampled pool an unweighted isotonic
        states the *retained* prior rather than the pool's. The calibration rows are not negatively
        subsampled by this module, but on Benchmark B they still arrive already thinned by the
        generator, so they still carry `sampling_weight` of 1 or 20.
        """
        from sklearn.isotonic import IsotonicRegression

        frames = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        raw = self.raw_score(frame)
        target = FomMetrics.as_bool(frame['is_correct']).astype(np.float64)
        lattice = frame['bravais_lattice'].to_numpy()
        weights = None if weight_column is None else fit_weights(frame, weight_column)

        def knots(values, labels, sample_weight=None):
            fitted = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            fitted.fit(values, labels, sample_weight=sample_weight)
            return (np.asarray(fitted.X_thresholds_, dtype=np.float64),
                    np.asarray(fitted.y_thresholds_, dtype=np.float64))

        calibrators = {'__pooled__': knots(raw, target, weights)}
        for name in np.unique(lattice):
            mask = lattice == name
            if int(mask.sum()) >= minimum and np.unique(target[mask]).size > 1:
                calibrators[str(name)] = knots(raw[mask], target[mask],
                                               None if weights is None else weights[mask])
        self.calibrators = calibrators
        self.meta['n_calibration_rows'] = int(frame.shape[0])
        self.meta['calibration_weight_column'] = weight_column
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


def _fit_pointwise(matrix, target, categorical_indices, seed, sample_weight=None, **params):
    """The headline model: gradient-boosted trees, not a network (S08 handoff, "Model choice").

    `sample_weight` is the inverse retention probability of each row. Campaign 1's pool kept every
    candidate, so omitting it there was correct; Benchmark B keeps every correct candidate, the
    union of seven top-200 lists, and 5 % of everything else, so a fit that ignores it is fitted to
    a negative set enriched twentyfold in the highest-scoring wrong candidates. `SCHEMA.md` says
    "Every fit must use it" and `METRICS.md` section 1 says the same. See `fit_weights`.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    settings = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
                    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                    random_state=seed)
    settings.update(params)
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_indices or None, **settings)
    model.fit(matrix, target, sample_weight=sample_weight)
    return model


def _fit_lambdarank(matrix, target, frame, categorical_indices, seed, sample_weight=None,
                    **params):
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
              sample_weight=None if sample_weight is None else np.asarray(sample_weight)[order],
              categorical_feature=categorical_indices or 'auto')
    return model
