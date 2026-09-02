"""Every number the FOM project reports, computed in one place from a candidate pool and a score.

One module owns these definitions because the headline criterion is a *compound* one -- a
correct candidate inside the pooled top ten **and** scoring above a threshold -- and there are
several ways to get it subtly easier than the indexer's own problem. Pooling per Bravais
lattice instead of across all fourteen is the worst of them (PROTOCOL section 10); ranking
inside the top-twenty-per-lattice subset the pipeline happens to report is another, worth
+1.1 points of ceiling on C1 by itself. There is one correct pooling and it is encoded here,
once, so no downstream task re-derives it.

The pool is 26.4M candidates over 3.0 GB, so nothing here holds it. Candidates are reduced,
one condition bundle at a time, to one row per (entry_id, condition_bundle) carrying the rank
and the score of the best correct candidate; every metric, stratum and bootstrap replicate is
then computed over ~41 000 rows. The reduction stores ranks and scores rather than booleans
precisely so that `top_n` and `threshold` stay free afterwards -- S06 sweeps thresholds and
must not need a second pass over the pool to do it.

Four things in here are decisions rather than arithmetic. They are documented at their
definitions and again in docs/fom/METRICS.md:

  * the "oracle" of PLAN section 6.4 is two different numbers, because a re-ranker cannot
    change a candidate's score (`ceiling_reranker` against `ceiling_rescorer`);
  * "no correct candidate in the pool" is a generation failure and is reported beside the loss
    decomposition rather than inside it;
  * every aggregate is UNWEIGHTED, and there is no other option. Campaign 1 reweighted each one
    to the CNRS Bravais-lattice distribution, which discarded 43 % of its effective sample (Kish
    n 682 of 1 197) and inflated every standard error by 1.75x. PROTOCOL section 3 rules 1 and 6
    forbid it before S16, so campaign 2 removed the capability rather than the default -- see
    the note where the weights table used to be;
  * the operating point cannot be maximised over the threshold, so `select_threshold`
    optimises something that punishes reporting a wrong cell. See its docstring.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomConditions


# Tie-break order for candidates that score identically, and the canonical order of every
# per-lattice table. Defined here rather than imported from `command_line.run`, to keep this
# module off the inference path; `test_bravais_lattices_match_run_py` pins them equal.
BRAVAIS_LATTICES = ('cP', 'cI', 'cF', 'tP', 'tI', 'hP', 'hR',
                    'oP', 'oC', 'oF', 'oI', 'mP', 'mC', 'aP')

# NOT PORTED (S04 Phase 2, 2026-08-26), and deliberately not as a disabled default:
# `CNRS_TABLE1`, `CNRS_WEIGHTS`, `CNRS_TOTAL`, `CNRS_OPERATING_POINT`, `CNRS_CEILING`, and every
# code path that consumed them.
#
# On `fom` these hold the opXRD/CNRS benchmark's lattice composition, and `evaluate` defaulted to
# reweighting every aggregate to it. PROTOCOL section 3 rule 1 seals CNRS until S16 and forbids
# reweighting any aggregate to its distribution before then; rule 6 requires aggregates to be
# unweighted. A default of `weights='cnrs'` means any caller that simply forgets the argument
# violates both, silently and without leaving a trace in its output -- which is how campaign 1
# came to reweight everything.
#
# So the capability is removed, not defaulted off. `evaluate` no longer takes `weights` and there
# is no argument that reinstates it. S16, which is allowed to compute the weighted view exactly
# once, takes this table back from `fom` deliberately and records the row in `CHERRY_PICK.md` --
# that is the normal mechanism (PROTOCOL section 3 rule 10), not a repair.
#
# `CNRS_OPERATING_POINT` (450/599) and `CNRS_CEILING` (533/599) go with them: INHERITED section 2
# records that neither denominator has reproducible provenance, and nothing before S16 may quote
# them.

# The condition bundles, campaign 2's and campaign 1's, in one mapping.
#
# Both are here because both pools are live: Benchmark B is the campaign's own and carries `c2_`
# tags, while Benchmark A is still on disk and S03, S04 and S09's back-comparisons read it. The
# two tag namespaces are disjoint by construction -- `FomConditions.TAG_PREFIX` exists to make
# them so -- so a union cannot mislabel a bundle from either pool, and a single mapping means a
# caller never has to say which campaign it is holding.
BUNDLE_LABELS = dict(FomConditions.BUNDLE_LABELS)
BUNDLE_LABELS.update({
    'error0_cont0': 'C0',
    'error1_cont0': 'C1',
    'error2_cont0': 'C2',
    'error1_cont2': 'C3',
    'error1_cont1_drop6': 'C4',
    'error1_cont1_drop10': 'C5',
    'error1_cont0_phase3': 'C6',
})

# Bundles excluded from every metric.
#
# Campaign 1's C0 is the zero-error control and its M20 is arithmetically degenerate: 9.49 % of
# its candidates score above 1e9 and 248 are non-finite, because the residual denominator goes to
# zero when the observed peaks *are* the calculated ones (F-054). It is 13.9 M of Benchmark A's
# 26.4 M rows, so a loader that globs the directory picks it up silently -- PROTOCOL section 3
# rule 11.
#
# **Campaign 2 contributes none**, and that is a design decision rather than an omission:
# `FomConditions.CONTROL_BUNDLES` is empty because campaign 2's own control uses a small NON-zero
# error multiplier (0.1x), which keeps the control's purpose and every residual-denominator merit
# finite (METRICS section 9). So on Benchmark B this exclusion is a no-op and `c2_error0.1_cont0`
# is reported like any other bundle.
#
# The S08 handoff asked for the constant to be emptied and "the exclusion machinery" removed. It
# is kept: emptying it would delete rule 11's only enforcement while the pool it protects is
# still on disk and still read. Removing it becomes correct when Benchmark A does -- see
# C2-F-076.
CONTROL_BUNDLES = tuple(FomConditions.CONTROL_BUNDLES) + ('error0_cont0',)

# The hard stratum: low-symmetry lattice, large cell, hard condition (METRICS section 5).
#
# The definition is campaign 1's and is deliberately unchanged; what changed is that the split is
# now SIZED so the stratum can carry a claim. Campaign 1's held 104 reachable source entries in
# total, split 64/16/24, so on its reporting split every threshold metric was exactly 0.0000 and
# McNemar found no discordant pairs (R3, F-063). S06 sized against measured reachability -- 77.4 %
# per entry at the generation cut, not the 34.9 % its own handoff assumed -- and `fom-dev` now
# carries 360 hard entries of which ~258 are reachable (C2-F-049).
#
# So `n_reachable` is reported beside every hard-stratum number (it is `n_found`, in the metric
# block of every scope row), and the reachability sizing is a property of the split rather than a
# term in this predicate. The S08 handoff reads as though reachability enters the definition; it
# does not, and METRICS section 5 is the text that governs.
HARD_LATTICES = ('mP', 'mC', 'aP')
HARD_BUNDLES = tuple(FomConditions.HARD_BUNDLES) + (
    'error2_cont0', 'error1_cont2', 'error1_cont1_drop6', 'error1_cont1_drop10')
HARD_MIN_DECILE = 8

# `standardize_cell`'s write-back fix (Q27, 7ab633d) changed `best_xnn` for the monoclinic and
# triclinic lattice systems only, so S02's numbers for these three no longer describe S04's
# pool. The gate attribution restricts to the complement.
Q27_AFFECTED_LATTICES = ('mP', 'mC', 'aP')

N_VOLUME_DECILES = 10
DEFAULT_STRATA = ('bravais_lattice', 'volume_decile', 'condition_bundle')
DEFAULT_TOP_N = 10

# ---------------------------------------------------------------------------------------
# Negative subsampling, and the one thing it is NOT exact for
# ---------------------------------------------------------------------------------------
# Benchmark B keeps every correct candidate, every candidate inside the top *K* by each of the
# merits below, and a 5 % Bernoulli sample of the rest (SCHEMA.md; C2-F-051). A rank computed on
# the retained rows is therefore exact to depth *K* -- but **only for a score in this tuple**.
#
# The guarantee is merit-conditional and neither SCHEMA.md nor METRICS.md said so; both state it
# unqualified, as "nothing that could have entered the top K was dropped". For a score the
# subsampler did not rank on -- a learned combiner, a neural score, any merit added later -- the
# candidates above a correct one are retained at ~5 %, so its rank is measured against a thinned
# field and comes out **optimistic**. See C2-F-077, which measures the size of it.
#
# `FomBenchmark.subsample_negatives` takes the union per (entry, condition, Bravais lattice), so
# the guarantee survives the cross-lattice pooling this module ranks by: a candidate at pooled
# rank r has within-lattice rank <= r, so r <= K implies it was retained.
RANK_EXACT_MERITS = tuple(FomBenchmark.REDUCED_MERIT_COLUMNS)

# Which way each merit points. Held here, once, because `evaluate`'s `higher_is_better` defaults to
# True and a lower-is-better merit passed without it is not an error -- it is a silently reversed
# ranking that looks like a very bad merit. That is exactly what happened to `X_N` in every S08
# floor table (C2-F-085): three of the seven count something you want *less* of.
#
#   X_N      observed lines below the cut-off that no calculated line explains (de Wolff)
#   n_over   calculated lines in range that no observation accounts for
#   max_gap  the largest run of them
#
# Callers should read this rather than pass a literal, so a merit added here is oriented everywhere
# at once. `orientation_of` raises on an unknown name rather than guessing a direction.
HIGHER_IS_BETTER = {
    'M20': True, 'Minfo': True, 'M_tilde': True, 'M_rev': True, 'M_sym': True,
    'M_rev_unfloored': True, 'M_sym_unfloored': True,
    'X_N': False, 'n_over': False, 'max_gap': False,
    # The posterior-based forms (C2-Q-025). Same direction as the counts they replace:
    # each is the EXPECTED value of the integer its hard counterpart returns.
    'X_N_soft': False, 'n_over_soft': False, 'max_gap_soft': False,
    # S10's hold-out family: the same statistics computed on the surplus peaks the cell was never
    # fitted to. `ho_raw` is a median |dQ| and `ho_tail_nll` a summed negative log-likelihood, so
    # those two point the other way from the merits beside them -- which is the C2-F-085 trap
    # again, one family later. `ho_N_cal`, `ho_n_scored` and `ho_ref_reach` are support and
    # coverage diagnostics rather than merits and are deliberately absent: ranking on one is a
    # mistake this map should refuse rather than serve.
    'ho_M20': True, 'ho_M': True, 'ho_M_tilde': True, 'ho_M_rev': True, 'ho_M_sym': True,
    'ho_Minfo': True,
    'ho_raw': False, 'ho_chi2': False, 'ho_tail_nll': False,
    # S10c's posterior hold-out family. All three point the same way -- a cell that predicts the
    # surplus peaks well assigns them confidently, so a higher posterior, a higher mean log
    # posterior and a higher evidence are all better. `ho_evidence` is the denominator the
    # posterior divides away (F-131 found it worth more than the ratio), and it is a log of a sum
    # of positive terms, so it is higher-is-better for the same reason the others are.
    'ho_post': True, 'ho_post_logmean': True, 'ho_evidence': True,
    # S11's extinction-rule arms. Every one of these is M20 EVALUATED AT the group the named
    # criterion chose -- not the criterion's own value -- so they are all higher-is-better and all
    # on M20's scale, which is what makes them comparable to each other and to the incumbent. The
    # criterion never becomes the ranking score here; ranking on M_sym is S09's question and
    # answering it inside the assignment rule would confound the two changes.
    'M20_at_M20': True, 'M20_at_M_rev': True, 'M20_at_M_sym': True,
    'M20_at_M_rev_unfloored': True, 'M20_at_M_rev_then_M20': True,
    }


def orientation_of(merit):
    """`higher_is_better` for a named merit. Raises rather than assuming a direction."""
    try:
        return HIGHER_IS_BETTER[merit]
    except KeyError:
        raise KeyError(
            f'No recorded direction of merit for {merit!r}. Add it to '
            f'FomMetrics.HIGHER_IS_BETTER rather than passing higher_is_better at the call site: '
            f'a reversed ranking is indistinguishable from a poor merit (C2-F-085).'
            ) from None


def holdout_orientation_of(column):
    """`higher_is_better` for a suffixed hold-out column: `ho_M_sym__n5` -> the `ho_M_sym` entry.

    The sweep names a column per peak budget, and the budget cannot change which way a merit
    points. Splitting the suffix here keeps one direction per merit rather than one per column,
    so a seven-point sweep cannot orient two of its points differently by typo.
    """
    return orientation_of(str(column).split('__n')[0])


def rank_exactness(score, top_n, top_k, subsampled, mrr=True):
    """Whether a rank metric on this pool is exact, and if not, why not.

    Returns `(exact, reason)`. `reason` is `None` when exact and a sentence otherwise; it is
    recorded in `MetricsResult.meta` whether or not the caller chose to proceed, so a number
    computed on a thinned pool always carries the statement of what it is.

    `top_k` is the pool's depth, `None` for a pool that was not subsampled or whose manifest
    could not be read -- `subsampled` separates those two, because an absent manifest is not
    evidence of a full pool.
    """
    if subsampled is False:
        return True, None
    if subsampled is None:
        return False, ('The pool has no readable manifest, so whether it was negatively '
                       'subsampled is unknown and no rank metric can be certified exact. '
                       'Pass subsample_top_k explicitly.')
    name = score if isinstance(score, str) else getattr(score, '__name__', 'a callable score')
    if not isinstance(score, str) or score not in RANK_EXACT_MERITS:
        return False, (
            f'{name!r} is not one of the merits the subsampler ranked on '
            f'({", ".join(RANK_EXACT_MERITS)}), so the candidates above a correct one were '
            f'retained at the negative rate rather than in full and every rank metric is '
            f'optimistic. Score the full pool, or add this merit to the retention rule.'
            )
    if top_k is not None and int(top_n) > int(top_k):
        return False, (
            f'top_n={int(top_n)} is deeper than the pool retention depth K={int(top_k)}; '
            f'beyond K the pool is a weighted sample and a rank is no longer exact.'
            )
    if mrr:
        return True, (
            f'Exact to depth K={top_k} for {name!r}. Note `mrr` uses the full rank and is NOT '
            f'bounded by top_n, so it is exact only for entries whose best correct candidate '
            f'ranks within K.'
            )
    return True, None

# The columns the metrics need, whatever the score. One bundle of fourteen lattices is 0.3 s
# and 222 MB projected this way, against ~2 GB with `xnn` and `unit_cell` attached.
SCORE_INDEPENDENT_COLUMNS = (
    'entry_id',
    'bravais_lattice',
    'candidate_id',
    'in_top_n',
    'is_correct',
    'is_off_by_two',
    'is_degenerate',
)

# Columns of `SCORE_INDEPENDENT_COLUMNS` that a schema-v3 pool does NOT carry on the candidate row.
#
# `is_degenerate` is the only one, and it moved for a reason rather than by omission: campaign 2's
# definition is a statement about the **pattern's own true lattice** -- whether its Niggli reduced
# cell sits accidentally on one of Santoro's special-condition boundaries, so a different lattice
# reproduces its peak positions exactly -- which takes one value per pattern, not one per candidate
# (C2-F-043). Campaign 1 stored it per candidate and shipped it null, which is why it excluded
# degenerates at a *measured* zero rather than a known one.
#
# Projecting the fixed list onto a v3 pool raises `ArrowInvalid` inside the parquet reader, so
# **the module could not read Benchmark B at all** until this was handled (C2-F-080). The column is
# dropped from the projection when absent and broadcast from the entry table instead, which is
# exactly equivalent: if a pattern's true lattice is degenerate then every correct candidate for it
# is, and the entry leaves the loss decomposition's denominator rather than counting as a figure of
# merit's failure.
OPTIONAL_CANDIDATE_COLUMNS = ('is_degenerate',)

# Metrics that are means of a per-entry boolean, so they reweight and bootstrap identically.
_FLAG_METRICS = (
    'operating_point', 'top1', 'top5', 'top10', 'rank_only', 'threshold_only', 'found',
    'reported', 'false_positive', 'off_by_two', 'degenerate_only',
    'lost_rank_failure', 'lost_threshold_failure', 'lost_both', 'lost_not_found',
)

# Everything a threshold has to exist for. Reported as NaN when `threshold is None`, rather
# than as a number that silently assumes every candidate passes.
_THRESHOLD_METRICS = (
    'operating_point', 'operating_point_ci_low', 'operating_point_ci_high',
    'operating_point_given_found', 'threshold_only', 'reported', 'false_positive', 'precision',
    'ceiling_reranker', 'headroom_reranker', 'headroom_rescorer',
    'lost_rank_failure', 'lost_threshold_failure', 'lost_both',
    'share_rank_failure', 'share_threshold_failure', 'share_both',
)

METRIC_COLUMNS = (
    'n_entries', 'n_found',
    'operating_point', 'operating_point_ci_low', 'operating_point_ci_high',
    'operating_point_given_found',
    'top1', 'top5', 'top10', 'rank_only', 'mrr',
    'threshold_only',
    'found', 'found_ci_low', 'found_ci_high',
    'ceiling_reranker', 'ceiling_rescorer', 'headroom_reranker', 'headroom_rescorer',
    'reported', 'false_positive', 'precision',
    'off_by_two', 'degenerate_only',
    'lost_rank_failure', 'lost_threshold_failure', 'lost_both', 'lost_not_found',
    'share_rank_failure', 'share_threshold_failure', 'share_both',
)


class MetricsResult:
    """Everything one (pool, score, threshold) combination produces.

    `per_entry` is part of the result rather than an internal: `mcnemar` needs the paired
    per-entry flags, and a stratum someone thinks of later is a groupby over it instead of
    another pass over 3 GB.
    """

    def __init__(self, per_entry, aggregate, hard, by_stratum, by_cell, loss, curve,
                 calibration, meta):
        self.per_entry = per_entry
        self.aggregate = aggregate
        self.hard = hard
        self.by_stratum = by_stratum
        self.by_cell = by_cell
        self.loss = loss
        self.curve = curve
        self.calibration = calibration
        self.meta = meta

    def metric(self, name, scope='aggregate'):
        table = {'aggregate': self.aggregate, 'hard': self.hard}[scope]
        return float(table[name].iloc[0])

    def stratum(self, name, level=None):
        rows = self.by_stratum.loc[self.by_stratum['stratum'] == name]
        return rows if level is None else rows.loc[rows['level'] == level]

    def tables(self):
        return dict(aggregate=self.aggregate, hard=self.hard, by_stratum=self.by_stratum,
                    by_cell=self.by_cell, loss=self.loss, curve=self.curve,
                    calibration=self.calibration)

    def __repr__(self):
        return (f"MetricsResult(score={self.meta['score']!r}, "
                f"threshold={self.meta['threshold']}, "
                f"n_entries={int(self.aggregate['n_entries'].iloc[0])}, "
                f"operating_point={self.metric('operating_point'):.4f}, "
                f"ceiling_rescorer={self.metric('ceiling_rescorer'):.4f})")


def evaluate(candidates, score='M20', higher_is_better=True, threshold=None,
             pool='cross_bl', top_n=DEFAULT_TOP_N, strata=DEFAULT_STRATA, entries=None, bundles=None, split=None, bravais_lattices=None,
             pool_subset='all', degenerates='exclude', score_columns=(),
             include_control=False, calibration=False, n_bootstrap=1000, seed=12345,
             hard_min_decile=HARD_MIN_DECILE,
             subsample_top_k='auto', allow_inexact_ranks=False):
    """Turn a candidate pool plus a score into every number the project reports.

    `candidates` is a benchmark root, a DataFrame, or an iterable of DataFrames. A root is read
    one condition bundle at a time, because the pooled ranking is per (entry, bundle) and
    nothing needs two bundles resident at once.

    `score` is a column name or a callable taking a candidate frame and returning one value per
    row. A callable is invoked once per shard, so it sees one lattice at a time; name the extra
    columns it needs in `score_columns`.

    `pool_subset` selects which survivors are ranked: 'all' is every candidate in the pool,
    'in_top_n' is the top-twenty-per-lattice subset the pipeline actually reports and the only
    one comparable with S02's live numbers. The operating point is identical either way -- a
    member of the pooled top ten is necessarily inside its own lattice's top twenty -- so this
    changes the ceiling, not the headline.

    `subsample_top_k` is the pool's negative-subsampling depth *K*. 'auto' reads it from the
    pool's manifest when `candidates` is a root, and means "not subsampled" for a frame handed in
    directly, since a caller assembling its own frame knows what is in it. A rank metric that
    this pool cannot answer exactly raises, naming what is wrong; `allow_inexact_ranks=True`
    proceeds and records the reason in `meta['rank_exactness']` instead. It refuses rather than
    warns because an optimistic rank is indistinguishable from a good one.
    """
    reduced, calibration_rows, reduce_meta = reduce_to_per_entry(
        candidates, score=score, higher_is_better=higher_is_better, pool=pool, top_n=top_n,
        entries=entries, bundles=bundles, split=split, bravais_lattices=bravais_lattices,
        score_columns=score_columns, include_control=include_control, calibration=calibration,
        hard_min_decile=hard_min_decile, subsample_top_k=subsample_top_k,
        allow_inexact_ranks=allow_inexact_ranks,
        )
    return summarise_per_entry(
        reduced, reduce_meta, threshold=threshold, top_n=top_n, strata=strata,
        pool_subset=pool_subset, degenerates=degenerates, n_bootstrap=n_bootstrap, seed=seed,
        calibration=calibration, calibration_rows=calibration_rows,
        )


def reduce_to_per_entry(candidates, score='M20', higher_is_better=True, pool='cross_bl',
                        top_n=DEFAULT_TOP_N, entries=None, bundles=None, split=None,
                        bravais_lattices=None, score_columns=(), include_control=False,
                        calibration=False, hard_min_decile=HARD_MIN_DECILE,
                        subsample_top_k='auto', allow_inexact_ranks=False):
    """The half of `evaluate` that touches the candidate pool. The expensive half.

    Returns `(per_entry, calibration_rows, reduce_meta)`, where `per_entry` is one row per
    (entry, condition) carrying the entry context and the pooled reduction, **before** any
    threshold has been applied.

    **This is a sufficient statistic for everything downstream.** `derive_flags` and every
    summary read only columns that are already here, so one pool pass answers every threshold,
    every metric, every stratum, McNemar and the bootstrap. That is what lets a 122 GB pool be
    reduced on the cluster and analysed on a laptop: the reduction is a few hundred megabytes and
    `summarise_per_entry` needs nothing else.

    Split out for that reason, not for tidiness -- `evaluate` is exactly these two calls composed
    and behaves identically.
    """
    if pool not in ('cross_bl', 'per_bl'):
        raise ValueError(f"pool must be 'cross_bl' or 'per_bl', got {pool!r}")

    entries, shards, source = _resolve_inputs(
        candidates, entries, bundles=bundles, split=split, bravais_lattices=bravais_lattices,
        score=score, score_columns=score_columns,
        )
    top_k, subsampled = _resolve_subsampling(candidates, subsample_top_k)
    ranks_exact, exactness_reason = rank_exactness(score, top_n, top_k, subsampled)
    if not ranks_exact and not allow_inexact_ranks:
        raise ValueError(
            f'Refusing to report a rank metric on this pool. {exactness_reason} '
            f'Pass allow_inexact_ranks=True to proceed with the reason recorded in meta.'
            )
    context = entry_context(entries, hard_min_decile=hard_min_decile)
    excluded = ([bundle for bundle in CONTROL_BUNDLES
                 if (context['condition_bundle'] == bundle).any()]
                if not include_control else [])

    reductions = []
    diagnostics = dict(n_candidates_seen=0, n_non_finite_score=0, n_score_above_1e9=0)
    calibration_rows = []
    degenerate_entries = _degenerate_entries(entries)
    for frame in shards:
        frame = _prepare_shard(frame, include_control, degenerate_entries)
        if frame is None:
            continue
        values = _shard_scores(frame, score, higher_is_better)
        diagnostics['n_candidates_seen'] += int(values.size)
        diagnostics['n_non_finite_score'] += int(np.sum(~np.isfinite(values)))
        diagnostics['n_score_above_1e9'] += int(np.sum(np.abs(values) > 1e9))
        reductions.append(reduce_pool(frame, values, pool=pool))
        if calibration:
            calibration_rows.append(pd.DataFrame({
                'condition_bundle': frame['condition_bundle'].to_numpy(),
                'score': values,
                'is_correct': as_bool(frame['is_correct']),
                }))
    if not reductions:
        raise ValueError('No candidates to evaluate after filtering')

    per_entry = context.merge(_combine_reductions(reductions),
                             on=['entry_id', 'condition_bundle'], how='inner', validate='1:1')
    reduce_meta = dict(
        score=score if isinstance(score, str) else getattr(score, '__name__', 'callable'),
        higher_is_better=bool(higher_is_better),
        pool=pool,
        reduced_top_n=int(top_n),
        split=split,
        bundles_excluded=sorted(excluded),
        hard_min_decile=int(hard_min_decile),
        subsample_top_k=(None if top_k is None else int(top_k)),
        subsampled=subsampled,
        ranks_exact=bool(ranks_exact),
        rank_exactness=exactness_reason,
        source=source,
        )
    reduce_meta.update(diagnostics)
    return per_entry, calibration_rows, reduce_meta


def reduce_many(candidates, scores, entries=None, splits=None, higher_is_better=None,
                pool='cross_bl', top_n=DEFAULT_TOP_N, bundles=None, bravais_lattices=None,
                include_control=False, hard_min_decile=HARD_MIN_DECILE, subsample_top_k='auto',
                allow_inexact_ranks=False, on_shard=None):
    """`reduce_to_per_entry` for many scores and many splits, in **one** pass over the pool.

    Returns `{(score_name, split): (per_entry, reduce_meta)}`, each entry identical to what
    `reduce_to_per_entry` returns for that pair. Nothing is re-implemented: this calls
    `_prepare_shard`, `_shard_scores`, `reduce_pool` and `_combine_reductions`, and runs
    `rank_exactness` per score independently, so every guard fires exactly as it does there.

    **This is C2-Q-027.** `reduce_to_per_entry` reads the pool once per score, which is right for
    one merit and wrong for many: S10b would have paid 37 reads of a 43 M-candidate pool instead of
    one, so it wrote this loop in a script instead, and S10c then copied it. S12 is the fourth
    consumer -- it scores a dozen retrained arms over the same pool -- so the loop is folded in
    here rather than copied a third time. The single-score signature is untouched.

    `scores` maps a name to a column name or to a callable taking the shard and returning one value
    per row, which is how a learned score reaches this: a fitted combiner is not a stored column.

    `higher_is_better` maps a name to its orientation; a name absent from it is looked up in
    `HIGHER_IS_BETTER` via `orientation_of`, which raises on an unknown merit rather than defaulting
    to True -- an omitted orientation is a silently reversed ranking that looks like a bad merit
    (C2-F-085).

    `splits` maps a label to the entry ids reported under it; `None` reduces every entry under the
    single label `None`. `on_shard` is called with each prepared shard before scoring, for a
    diagnostic that would otherwise need its own pass over the pool.
    """
    if not scores:
        raise ValueError('no scores to reduce')
    # `split=None`: the splits are applied per shard below, because a multi-split reduce is the
    # whole reason this exists and `_resolve_inputs` can only filter to one.
    entries, shards, source = _resolve_inputs(
        candidates, entries, bundles=bundles, split=None, bravais_lattices=bravais_lattices,
        score=next(iter(scores.values())), score_columns=(),
        )
    top_k, subsampled = _resolve_subsampling(candidates, subsample_top_k)

    orientation, exactness = {}, {}
    for name, score in scores.items():
        if higher_is_better is not None and name in higher_is_better:
            orientation[name] = bool(higher_is_better[name])
        else:
            orientation[name] = orientation_of(name)
        # Certified on the NAME, not on the column it reads. The key is the merit's identity --
        # `{'M20': 'M20_recomputed'}` is still M20 -- and `{'learned': 'M_sym'}` is not `M_sym`
        # however it is computed. Keying on the column would let a learned score inherit an
        # exactness certificate from whatever column it happened to be stored in, which is the
        # exact confusion C2-R-013 exists to prevent.
        exact, reason = rank_exactness(name, top_n, top_k, subsampled)
        exactness[name] = (exact, reason)
        if not exact and not allow_inexact_ranks:
            raise ValueError(
                f'Refusing to report a rank metric on this pool for {name!r}. {reason} '
                f'Pass allow_inexact_ranks=True to proceed with the reason recorded in meta.'
                )

    if splits is None:
        splits = {None: None}
    context = entry_context(entries, hard_min_decile=hard_min_decile)
    degenerate_entries = _degenerate_entries(entries)
    excluded = ([bundle for bundle in CONTROL_BUNDLES
                 if (context['condition_bundle'] == bundle).any()] if not include_control else [])

    accumulated = {(name, label): [] for name in scores for label in splits}
    diagnostics = {name: dict(n_candidates_seen=0, n_non_finite_score=0, n_score_above_1e9=0)
                   for name in scores}
    for frame in shards:
        frame = _prepare_shard(frame, include_control, degenerate_entries)
        if frame is None:
            continue
        if on_shard is not None:
            on_shard(frame)
        # The split masks are computed once and the scores one at a time. Holding every score's
        # values at once would be one float64 array per score per shard -- twenty-one of them over
        # a 14 M-row bundle is 2.3 GB on top of a 5.3 GB frame, which is the difference between
        # fitting in memory and not.
        masks = {}
        for label, ids in splits.items():
            if ids is None:
                masks[label] = None
            else:
                mask = frame['entry_id'].isin(ids).to_numpy()
                if mask.any():
                    masks[label] = mask
        shards = {label: (frame if mask is None else frame.loc[mask])
                  for label, mask in masks.items()}
        for name, score in scores.items():
            values = _shard_scores(frame, score, orientation[name])
            diagnostics[name]['n_candidates_seen'] += int(values.size)
            diagnostics[name]['n_non_finite_score'] += int(np.sum(~np.isfinite(values)))
            diagnostics[name]['n_score_above_1e9'] += int(np.sum(np.abs(values) > 1e9))
            for label, mask in masks.items():
                accumulated[(name, label)].append(
                    reduce_pool(shards[label], values if mask is None else values[mask],
                                pool=pool))
            del values

    out = {}
    for (name, label), reductions in accumulated.items():
        if not reductions:
            continue
        per_entry = context.merge(_combine_reductions(reductions),
                                  on=['entry_id', 'condition_bundle'], how='inner',
                                  validate='1:1')
        score = scores[name]
        meta = dict(
            score=score if isinstance(score, str) else name,
            higher_is_better=bool(orientation[name]),
            pool=pool,
            reduced_top_n=int(top_n),
            split=label,
            bundles_excluded=sorted(excluded),
            hard_min_decile=int(hard_min_decile),
            subsample_top_k=(None if top_k is None else int(top_k)),
            subsampled=subsampled,
            ranks_exact=bool(exactness[name][0]),
            rank_exactness=exactness[name][1],
            source=source,
            )
        meta.update(diagnostics[name])
        out[(name, label)] = (per_entry, meta)
    if not out:
        raise ValueError('No candidates to evaluate after filtering')
    return out


def summarise_per_entry(per_entry, reduce_meta, threshold=None, top_n=DEFAULT_TOP_N,
                        strata=DEFAULT_STRATA, pool_subset='all', degenerates='exclude',
                        n_bootstrap=1000, seed=12345, calibration=False, calibration_rows=None):
    """The half of `evaluate` that needs no pool: flags, summaries, curve, intervals.

    Takes what `reduce_to_per_entry` returned. Every threshold policy, stratification and
    bootstrap is a re-run of this function over the same reduction, which is why a threshold
    sweep costs one pool pass rather than one per threshold.
    """
    if pool_subset not in ('all', 'in_top_n'):
        raise ValueError(f"pool_subset must be 'all' or 'in_top_n', got {pool_subset!r}")
    if degenerates not in ('exclude', 'include'):
        raise ValueError(f"degenerates must be 'exclude' or 'include', got {degenerates!r}")
    reduced_top_n = reduce_meta.get('reduced_top_n')
    if reduced_top_n is not None and top_n > reduced_top_n and reduce_meta.get('subsampled'):
        # The exactness check ran against the depth the reduction was certified at; asking for a
        # deeper top-n afterwards would quietly evade it.
        raise ValueError(
            f'This reduction was certified for top_n <= {reduced_top_n}; {top_n} was asked for. '
            f'Re-reduce at the deeper depth rather than summarising past the certificate.')

    # The threshold is quoted in the score's own orientation -- "accept below t" for a
    # lower-is-better merit -- so it is mirrored with the scores rather than by the caller. A
    # per-lattice mapping is mirrored value by value.
    higher_is_better = reduce_meta['higher_is_better']
    if threshold is None or higher_is_better:
        internal_threshold = threshold
    elif isinstance(threshold, dict):
        internal_threshold = {lattice: -cut for lattice, cut in threshold.items()}
    else:
        internal_threshold = -threshold
    per_entry = derive_flags(per_entry, threshold=internal_threshold, top_n=top_n,
                             pool_subset=pool_subset, degenerates=degenerates)

    has_threshold = threshold is not None
    replicates = _bootstrap_replicates(per_entry['cluster'].to_numpy(), n_bootstrap, seed)
    aggregate = _summarise(per_entry, 'aggregate', has_threshold, replicates)
    hard = _summarise(per_entry.loc[per_entry['is_hard']], 'hard', has_threshold, replicates)
    by_stratum = _summarise_by_stratum(per_entry, strata, has_threshold)
    by_cell = _summarise_cells(per_entry, strata, has_threshold)
    loss = _loss_table(per_entry, strata, has_threshold)
    curve = threshold_curve(per_entry)
    calibration_table, calibration_reason = _calibration_table(
        pd.concat(calibration_rows, ignore_index=True) if calibration_rows else None)
    if not calibration:
        calibration_reason = 'not requested (pass calibration=True for a probability score)'

    meta = dict(reduce_meta)
    meta.pop('reduced_top_n', None)
    meta.update(
        threshold=(None if threshold is None else
                   ({str(k): float(v) for k, v in threshold.items()}
                    if isinstance(threshold, dict) else float(threshold))),
        top_n=int(top_n),
        pool_subset=pool_subset,
        degenerates=degenerates,
        weights='none',
        bundles=sorted(per_entry['condition_bundle'].unique().tolist()),
        strata=list(strata),
        n_entries=int(per_entry.shape[0]),
        n_clusters=int(per_entry['cluster'].nunique()),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        entry_digest=entry_digest(per_entry),
        calibration_skipped_reason=calibration_reason,
        )
    return MetricsResult(per_entry, aggregate, hard, by_stratum, by_cell, loss, curve,
                         calibration_table, meta)


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------
def _resolve_inputs(candidates, entries, bundles, split, bravais_lattices, score,
                    score_columns):
    """Normalise the three accepted shapes of `candidates` into (entries, shard iterator)."""
    if isinstance(candidates, (str, Path)):
        root = Path(candidates)
        entries = FomBenchmark.load_entries(root) if entries is None else entries
        keep = None if split is None else set(entries.loc[entries['split'] == split, 'entry_id'])
        columns = _projection(score, score_columns,
                              available=FomBenchmark.candidate_columns_present(root))
        available = FomBenchmark.available_bundles(root)
        wanted = available if bundles is None else [b for b in available if b in set(bundles)]
        if not wanted:
            raise FileNotFoundError(f'No candidate shards under {root} for bundles {bundles}')
        shards = (_load_bundle(root, bundle, bravais_lattices, columns, keep)
                  for bundle in wanted)
        return entries, shards, str(root)
    if entries is None:
        raise ValueError('`entries` is required unless `candidates` is a benchmark root')
    if isinstance(candidates, pd.DataFrame):
        frame = candidates
        if bundles is not None:
            frame = frame.loc[frame['condition_bundle'].isin(bundles)]
        if bravais_lattices is not None:
            frame = frame.loc[frame['bravais_lattice'].isin(bravais_lattices)]
        if split is not None:
            keep = set(entries.loc[entries['split'] == split, 'entry_id'])
            frame = frame.loc[frame['entry_id'].isin(keep)]
        return entries, [frame], 'frame'
    return entries, candidates, 'iterable'


def _resolve_subsampling(candidates, subsample_top_k):
    """`(top_k, subsampled)` for the pool being evaluated.

    'auto' reads the manifest for a root and assumes a full pool for a frame or an iterable: a
    caller that assembled its own frame knows what went into it, and there is no manifest to
    consult. Anything else is taken as the depth itself, `None` meaning "not subsampled".
    """
    if subsample_top_k != 'auto':
        return subsample_top_k, subsample_top_k is not None
    if isinstance(candidates, (str, Path)):
        return FomBenchmark.subsample_depth(Path(candidates))
    return None, False


def _degenerate_entries(entries):
    """The entry ids whose true lattice is degenerate, or None when the table does not say.

    `None` and the empty set are different: a pool that never computed the quantity must not read
    as one that computed it and found none. That distinction is exactly what campaign 1 lost —
    it shipped the column null and so excluded degenerates at a *measured* zero, where the real
    rate is 2.86 % of entries and 16.7 % on mC (C2-F-043).
    """
    if 'is_degenerate' not in entries.columns:
        return None
    flags = pd.Series(entries['is_degenerate']).fillna(False).to_numpy(dtype=bool)
    return set(np.asarray(entries['entry_id'].astype(str))[flags])


def _load_bundle(root, bundle, bravais_lattices, columns, keep_entry_ids):
    frame = FomBenchmark.load_candidates(root, bundles=[bundle],
                                         bravais_lattices=bravais_lattices, columns=columns)
    if keep_entry_ids is not None:
        frame = frame.loc[frame['entry_id'].isin(keep_entry_ids)].reset_index(drop=True)
    return frame


def _projection(score, score_columns, available=None):
    """The columns to read. Optional ones are dropped when the pool does not carry them."""
    columns = [column for column in SCORE_INDEPENDENT_COLUMNS
               if available is None
               or column not in OPTIONAL_CANDIDATE_COLUMNS
               or column in available]
    if isinstance(score, str):
        columns.append(score)
    columns.extend(column for column in score_columns if column not in columns)
    return columns


def _prepare_shard(frame, include_control, degenerate_entries=None):
    """Drop what is not being evaluated, and refuse a frame that cannot be joined.

    `degenerate_entries` supplies `is_degenerate` for a pool that carries it on the entry table
    rather than on the candidate row -- every schema-v3 pool. Broadcasting the entry's flag onto
    its candidates is the faithful translation, not a convenience: the quantity is a property of
    the pattern's true lattice, so if it holds the pattern is ambiguous from peak positions alone
    and *all* of its correct candidates are degenerate.
    """
    if frame is None or frame.shape[0] == 0:
        return None
    if 'condition_bundle' not in frame.columns:
        raise ValueError(
            'Candidate frames must carry condition_bundle; load them with '
            'FomBenchmark.load_candidates, which reads it from the filename.'
            )
    if not include_control:
        frame = frame.loc[~frame['condition_bundle'].isin(CONTROL_BUNDLES)]
    if 'is_degenerate' not in frame.columns:
        frame = frame.copy()
        frame['is_degenerate'] = (
            False if not degenerate_entries
            else frame['entry_id'].isin(degenerate_entries).to_numpy())
    return frame if frame.shape[0] else None


def _shard_scores(frame, score, higher_is_better):
    if isinstance(score, str):
        if score not in frame.columns:
            raise KeyError(f'Score column {score!r} is not in the candidate frame')
        values = frame[score].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(score(frame), dtype=np.float64)
        if values.shape != (frame.shape[0],):
            raise ValueError(
                f'A callable score must return one value per row; got {values.shape} for '
                f'{frame.shape[0]} rows'
                )
    # Everything downstream assumes higher-is-better, so the orientation is applied once,
    # here, and `derive_flags` mirrors the threshold with it.
    return values if higher_is_better else -values


def as_bool(column):
    """Nullable or object booleans as a plain array, with NA read as False.

    `is_degenerate` arrives from parquet as an all-null column, so it is nullable by
    construction and `.to_numpy(dtype=bool)` would raise rather than tell us anything.
    """
    return np.asarray(pd.Series(column).fillna(False).to_numpy(dtype=bool))


# ---------------------------------------------------------------------------------------
# The per-entry reduction
# ---------------------------------------------------------------------------------------
def reduce_pool(frame, values, pool='cross_bl'):
    """Rank a shard's candidates and reduce them to one row per (entry, condition bundle).

    Ranking is one `np.lexsort` on (descending score, Bravais lattice, candidate_id). The
    tie-break is total and label-independent: putting `is_correct` anywhere in the key would
    make every ranking optimistic, and leaving ties to the input order would make the numbers
    depend on the glob order of the shards. Entries whose outcome hinged on a tie stay
    countable through `n_ties_at_best_correct`.

    Non-finite scores follow numpy's own ordering: +inf ranks first, which is right because for
    M20 it means a zero residual, and NaN ranks last because it carries no information.

    `pool='per_bl'` ranks within each (entry, lattice) instead, and the entry-level reduction
    then keeps the best rank any lattice achieved. That is a different and easier question than
    the indexer's, which is why it is a named mode rather than the default.
    """
    entry_code, keys = _group_codes(frame['entry_id'], frame['condition_bundle'])
    lattice = frame['bravais_lattice'].to_numpy()
    lattice_order = pd.Categorical(lattice, categories=BRAVAIS_LATTICES).codes.astype(np.int64)
    if (lattice_order < 0).any():
        unknown = sorted(set(np.asarray(lattice)[lattice_order < 0]))
        raise ValueError(f'Unknown Bravais lattice in the candidate frame: {unknown}')
    candidate_id = frame['candidate_id'].to_numpy(dtype=np.int64)
    rank_code = (entry_code if pool == 'cross_bl'
                 else pd.factorize(entry_code*len(BRAVAIS_LATTICES) + lattice_order,
                                   sort=True)[0])

    labels = dict(
        is_correct=as_bool(frame['is_correct']),
        is_degenerate=as_bool(frame['is_degenerate']),
        is_off_by_two=as_bool(frame['is_off_by_two']),
        )
    in_top_n = as_bool(frame['in_top_n'])

    columns = {}
    for subset, mask in (('all', None), ('in_top_n', in_top_n)):
        reduced = _reduce_subset(entry_code, rank_code, values, lattice, lattice_order,
                                 candidate_id, labels, mask, n_groups=len(keys))
        for name, column in reduced.items():
            columns[f'{name}_{subset}'] = column
    out = pd.DataFrame(columns)
    out.insert(0, 'entry_id', [key[0] for key in keys])
    out.insert(1, 'condition_bundle', [key[1] for key in keys])
    return out


def _group_codes(entry_id, condition_bundle):
    """Integer codes for (entry_id, condition_bundle), and the keys in code order.

    The two fields are factorized separately and combined arithmetically rather than pasted
    into one string: this runs over millions of rows per bundle, factorizing object tuples is
    an order of magnitude slower, and a string separator is not safe -- numpy truncates object
    string concatenation at a NUL byte, which silently produced 'ADOGEHerror1_cont0'.
    """
    entry_codes, entry_keys = pd.factorize(pd.Series(entry_id).astype(str).to_numpy(), sort=True)
    bundle_codes, bundle_keys = pd.factorize(
        pd.Series(condition_bundle).astype(str).to_numpy(), sort=True)
    stride = max(len(bundle_keys), 1)
    codes, composites = pd.factorize(entry_codes.astype(np.int64)*stride + bundle_codes,
                                     sort=True)
    keys = [(entry_keys[composite//stride], bundle_keys[composite % stride])
            for composite in composites]
    return codes.astype(np.int64), keys


def _reduce_subset(entry_code, rank_code, values, lattice, lattice_order, candidate_id, labels,
                   subset_mask, n_groups):
    """One ranking pass over a subset of the pool, and the gathers that hang off it."""
    if subset_mask is not None:
        keep = np.flatnonzero(subset_mask)
        entry_code = entry_code[keep]
        rank_code = rank_code[keep]
        values = values[keep]
        lattice = np.asarray(lattice)[keep]
        lattice_order = lattice_order[keep]
        candidate_id = candidate_id[keep]
        labels = {name: column[keep] for name, column in labels.items()}

    # NaN sorts last; +inf and -inf keep their natural places.
    sort_key = -values.copy()
    sort_key[np.isnan(values)] = np.inf
    rank = _ranks_within(rank_code, sort_key, lattice_order, candidate_id)
    # The reduction always sorts by *score* within the entry, so "the best correct candidate"
    # means the highest-scoring one under either pooling. What `pool` changes is only which
    # rank is attributed to it. Ordering the gather by rank instead would make `threshold_only`
    # pool-dependent -- under 'per_bl' it would pick a rank-0 candidate scoring 8 over a rank-3
    # candidate scoring 15 -- and `threshold_only` is a statement about scores alone.
    order = np.lexsort((candidate_id, lattice_order, sort_key, entry_code))
    entry_sorted = entry_code[order]
    values_sorted = values[order]
    rank_sorted = rank[order]
    lattice_sorted = np.asarray(lattice)[order]
    correct = labels['is_correct'][order]
    degenerate = labels['is_degenerate'][order]
    off_by_two = labels['is_off_by_two'][order]

    counts = np.bincount(entry_sorted, minlength=n_groups)
    columns = {
        'n_candidates': counts,
        'n_off_by_two': _count(entry_sorted, off_by_two, n_groups),
        'n_degenerate': _count(entry_sorted, degenerate, n_groups),
        'n_non_finite_score': _count(entry_sorted, ~np.isfinite(values_sorted), n_groups),
        }

    present = counts > 0
    starts = np.searchsorted(entry_sorted, np.arange(n_groups), side='left')
    score_top = np.full(n_groups, np.nan)
    top_is_correct = np.zeros(n_groups, dtype=bool)
    lattice_top = np.array([None]*n_groups, dtype=object)
    first = starts[present]
    score_top[present] = values_sorted[first]
    top_is_correct[present] = correct[first]
    lattice_top[present] = lattice_sorted[first]
    columns['score_top'] = score_top
    columns['top_is_correct'] = top_is_correct
    columns['bravais_lattice_top'] = lattice_top

    for suffix, mask in (('', correct & ~degenerate), ('_incl_degenerate', correct)):
        rank_best = np.full(n_groups, -1, dtype=np.int64)
        score_best = np.full(n_groups, np.nan)
        lattice_best = np.array([None]*n_groups, dtype=object)
        ties = np.zeros(n_groups, dtype=np.int64)
        hits = np.flatnonzero(mask)
        if hits.size:
            groups_hit = entry_sorted[hits]
            _, first_hit = np.unique(groups_hit, return_index=True)
            positions = hits[first_hit]
            winners = groups_hit[first_hit]
            rank_best[winners] = rank_sorted[positions]
            score_best[winners] = values_sorted[positions]
            lattice_best[winners] = lattice_sorted[positions]
            # How many candidates of this entry share the winning score, so an outcome that
            # hinged on a tie can be counted rather than argued about.
            ties = _count(entry_sorted, values_sorted == score_best[entry_sorted], n_groups)
        columns[f'rank_best_correct{suffix}'] = rank_best
        columns[f'score_best_correct{suffix}'] = score_best
        columns[f'bravais_lattice_best_correct{suffix}'] = lattice_best
        columns[f'n_correct{suffix}'] = _count(entry_sorted, mask, n_groups)
        columns[f'has_correct{suffix}'] = rank_best >= 0
        columns[f'n_ties_at_best_correct{suffix}'] = ties
    return columns


def _ranks_within(group_code, sort_key, lattice_order, candidate_id):
    """0-based rank of every row inside its group, by the total order the module defines."""
    order = np.lexsort((candidate_id, lattice_order, sort_key, group_code))
    group_sorted = group_code[order]
    n_groups = int(group_sorted[-1]) + 1 if group_sorted.size else 0
    starts = np.searchsorted(group_sorted, np.arange(n_groups), side='left')
    rank_sorted = np.arange(group_sorted.size) - starts[group_sorted]
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = rank_sorted
    return rank


def _count(group_code, flags, n_groups):
    return np.bincount(group_code, weights=np.asarray(flags, dtype=float),
                       minlength=n_groups).astype(np.int64)


def _combine_reductions(reductions):
    combined = pd.concat(reductions, ignore_index=True)
    duplicated = combined.duplicated(['entry_id', 'condition_bundle'])
    if duplicated.any():
        raise ValueError(
            f'{int(duplicated.sum())} (entry_id, condition_bundle) reduced twice. The shards of '
            'one bundle must be pooled before ranking, not reduced separately -- otherwise the '
            'ranking is per shard and the pooling claim is false.'
            )
    return combined


# ---------------------------------------------------------------------------------------
# Entry-level context: strata and weights
# ---------------------------------------------------------------------------------------
def entry_context(entries, hard_min_decile=HARD_MIN_DECILE):
    """The stratification variables, one row per (entry_id, condition_bundle).

    Deciles are computed over `entries` as given -- the whole table, not one split -- so
    `fom-train` and `fom-dev` share bins by construction and no edges have to be threaded
    between calls.

    **The decile is joined from the entry table when it is there, and only recomputed when it is
    not** -- see the comment at the assignment. That is the R14 fix and it is why `entries` is
    allowed to carry a `volume_decile` column at all.

    `hard_min_decile` widens the hard stratum's volume cut, and exists for one reason: at the
    literal cut of 8 the stratum holds 16 reachable source entries on `fom-dev`, where every merit
    scores exactly 0.0000 and McNemar finds no discordant pairs, so its *threshold* metrics cannot
    be produced from the reportable split at all (F-063). S06 got round that by pooling rank metrics
    over `fom-train`+`fom-dev`, a licence that holds only while nothing is fitted on `fom-train`;
    S07 fits, so it cannot inherit it. Q32 is resolved by reporting hard-stratum *threshold* metrics
    at decile >= 6 -- 538 reachable rows over 313 reachable entries against 146/104 (F-062) -- while
    rank metrics stay on the literal stratum. The default is unchanged, so every earlier number
    means what it did.
    """
    required = ['entry_id', 'condition_bundle', 'split', 'bravais_lattice_true',
                'lattice_system_true', 'volume_true']
    missing = [column for column in required if column not in entries.columns]
    if missing:
        raise ValueError(f'Entry table is missing {missing}')
    context = pd.DataFrame({
        'entry_id': entries['entry_id'].astype(str).to_numpy(),
        'condition_bundle': entries['condition_bundle'].astype(str).to_numpy(),
        'split': entries['split'].astype(str).to_numpy(),
        'bravais_lattice': entries['bravais_lattice_true'].astype(str).to_numpy(),
        'lattice_system': entries['lattice_system_true'].astype(str).to_numpy(),
        'volume_true': entries['volume_true'].to_numpy(dtype=np.float64),
        })
    context['condition_label'] = context['condition_bundle'].map(BUNDLE_LABELS).fillna('?')
    # JOINED, never recomputed, whenever the entry table carries it -- which schema v3 does, by
    # reading it from the frozen split manifest at generation time (SCHEMA.md, R14).
    #
    # This is the one-line change S06 exists to make. `volume_decile` below is a *within-lattice
    # percentile rank*, so it is a property of the row set it is computed over rather than of the
    # entry. Campaign 1 recomputed it from whatever row set a caller happened to hand in, so once
    # 33 entries were lost to unplaceable second-phase lines and the bundles were aligned by
    # intersection, 114 of 5 922 entries disagreed with the manifest, and the hard stratum with
    # them -- from the 286 entries the split was balanced over to the 298 the pipeline used. No
    # number was wrong; "the hard stratum" simply denoted two different sets of entries in two
    # different documents.
    #
    # Note the mechanism is NOT the one the inherited record states. F-108 and the S06 handoff
    # both say dropping rows "can only raise" a survivor's rank; a survivor only rises when the
    # dropped rows sat above it, so attrition uncorrelated with volume perturbs the decile in
    # BOTH directions (measured in tests/test_split_manifest.py). Campaign 1's 114 entries all
    # moving up says its attrition was correlated with volume. The fix is the same either way,
    # but a stratum on a recomputed decile can lose entries as well as gain them.
    #
    # The fallback is not laziness: Benchmark A has no such column and S03 and S04 still read it,
    # so a pool without the column keeps campaign 1's behaviour and says which it used.
    if 'volume_decile' in entries.columns:
        context['volume_decile'] = np.asarray(entries['volume_decile'], dtype=np.int64)
        context.attrs['volume_decile_source'] = 'stored'
    else:
        context['volume_decile'] = volume_decile(context)
        context.attrs['volume_decile_source'] = 'recomputed'
    # `pool_size_full` is the survivor count BEFORE negative subsampling, so it is the only
    # correct denominator for a percentile on a thinned pool. Counting retained rows instead
    # gives a percentile against a 5 % field, which is the same defect as an inexact rank one
    # level down. Carried through here so nothing downstream has to re-derive it (PROTOCOL
    # section 3 rule 8).
    for optional in ('n_peaks_available', 'n_dropout_achieved', 'second_phase_lines',
                     'pool_size_full'):
        if optional in entries.columns:
            context[optional] = entries[optional].to_numpy()
    context['zone_count_min'] = _zone_count_min(entries)
    context['axis_ratio'] = _axis_ratio(entries)
    context['dominant_zone'] = pd.cut(
        context['axis_ratio'], bins=[0.0, 0.25, 0.5, 1.0001], include_lowest=True,
        labels=['very_anisotropic', 'anisotropic', 'isotropic'],
        ).astype(object)
    # The bootstrap unit is the source entry, not the (entry, condition) row: one crystal
    # contributes up to seven rows whose noise is correlated by construction, because the
    # per-entry seed is derived from the entry identifier (PROTOCOL section 6).
    context['cluster'] = pd.factorize(context['entry_id'], sort=True)[0]
    context['is_hard'] = (
        context['bravais_lattice'].isin(HARD_LATTICES)
        & (context['volume_decile'] >= int(hard_min_decile))
        & context['condition_bundle'].isin(HARD_BUNDLES)
        )
    return context


def volume_decile(frame, n_deciles=N_VOLUME_DECILES):
    """Volume deciles **within each true Bravais lattice**, 0-based.

    Within-lattice, not global: this is what `run_fom_mirror_analysis.volume_decile` computes
    and what the frozen split manifest is stratified on, so a global decile would make the hard
    stratum a different set of entries from the one the split was balanced over (DWMM,
    2026-08-17; the S05 handoff's "over the whole benchmark" is superseded).

    `rank(method='first')` rather than `qcut`, following S02 exactly: no edge handling, and
    balanced cells even when many entries share a volume. Note the boundary behaviour that
    comes with reproducing it -- percentile ranks run from 1/n to 1, so the lowest cell is one
    entry short and the highest one long. It is reproduced rather than corrected because the
    frozen split manifest's own deciles were computed this way, and the hard stratum has to
    mean the same set of entries the split was balanced over.
    """
    ranked = frame.groupby('bravais_lattice')['volume_true'].rank(method='first', pct=True)
    return np.clip((ranked*n_deciles).astype(int), 0, n_deciles - 1)


def _zone_count_min(entries):
    """The random-forest grouping's dominant-zone count, from the true Miller indices.

    Supplemental 5.2.1: the minimum over the three axes of how many observed peaks carry a
    non-zero Miller index on that axis. A small value means one zone indexes most of the
    pattern, so the apparent agreement rests on few independent constraints.
    `FigureOfMerits.get_zone_dominance` implements Shirley's better-founded S/V*^(2/3) for a
    *candidate*; this is the entry-level variable the S05 handoff asks for.
    """
    if 'hkl_true' not in entries.columns:
        return np.full(entries.shape[0], -1, dtype=np.int64)
    counts = np.full(entries.shape[0], -1, dtype=np.int64)
    for position, flat in enumerate(entries['hkl_true'].to_numpy()):
        if flat is None:
            continue
        hkl = np.asarray(flat).reshape(-1, 3)
        if hkl.size == 0:
            continue
        counts[position] = int(np.min(np.count_nonzero(hkl, axis=0)))
    return counts


def _axis_ratio(entries):
    """min/max of the true cell's axis lengths -- the ratio the RF grouping currently uses."""
    if 'unit_cell_true' not in entries.columns:
        return np.full(entries.shape[0], np.nan)
    ratios = np.full(entries.shape[0], np.nan)
    for position, cell in enumerate(entries['unit_cell_true'].to_numpy()):
        if cell is None:
            continue
        axes = np.asarray(cell, dtype=np.float64)[:3]
        if axes.size < 3 or not np.all(np.isfinite(axes)) or np.max(axes) <= 0:
            continue
        ratios[position] = float(np.min(axes)/np.max(axes))
    return ratios


# ---------------------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------------------
def _resolve_threshold(threshold, frame, lattice_column):
    """A scalar cut, or one cut per row taken from `lattice_column`.

    A lattice absent from the mapping, and a row with no candidate to attribute (an entry with no
    correct candidate, or none at all), get `+inf` -- so the comparison is False rather than
    silently borrowing another lattice's cut.
    """
    if not isinstance(threshold, dict):
        return float(threshold)
    lattices = frame[lattice_column].to_numpy()
    return np.array([threshold.get(lattice, np.inf) if lattice is not None else np.inf
                     for lattice in lattices], dtype=np.float64)


def derive_flags(per_entry, threshold=None, top_n=DEFAULT_TOP_N, pool_subset='all',
                 degenerates='exclude'):
    """Attach the per-entry outcome flags for one (threshold, top_n) operating point.

    Split out from the reduction because these are the only quantities that depend on the
    threshold, so S06 can sweep it without re-reading the pool and `mcnemar` can pair two
    scores on exactly these columns.

    `threshold` is a scalar, or a **mapping from Bravais lattice to a cut**. The mapping exists
    because a per-lattice accept rule cannot be expressed as a transform of the score: subtracting
    a per-lattice offset changes the *cross-lattice ranking* as well as the accept test, and S08
    measured that conflation costing 3.9 pp of top-10 (F-089). Applied here the ranking is
    untouched and only the two comparisons move -- each against the lattice of the candidate it is
    testing, which is why the reduction records `bravais_lattice_best_correct` and
    `bravais_lattice_top` separately.
    """
    frame = per_entry.copy()
    suffix = '' if degenerates == 'exclude' else '_incl_degenerate'
    for name in ('rank_best_correct', 'score_best_correct', 'has_correct', 'n_correct',
                 'bravais_lattice_best_correct', 'n_ties_at_best_correct'):
        frame[name] = frame[f'{name}{suffix}_{pool_subset}']
    for name in ('score_top', 'top_is_correct', 'bravais_lattice_top', 'n_candidates',
                 'n_non_finite_score'):
        frame[name] = frame[f'{name}_{pool_subset}']

    rank = frame['rank_best_correct'].to_numpy()
    score_best = frame['score_best_correct'].to_numpy(dtype=np.float64)
    found = frame['has_correct'].to_numpy(dtype=bool)
    in_top = found & (rank < top_n)
    if threshold is None:
        # Rank-only reporting. The flags stay defined so the frame has one shape, and
        # `_metric_block` NaNs out every metric a threshold exists for.
        over = found
        reported = np.ones(found.shape, dtype=bool)
    else:
        cut_best = _resolve_threshold(threshold, frame, 'bravais_lattice_best_correct')
        cut_top = _resolve_threshold(threshold, frame, 'bravais_lattice_top')
        over = found & (score_best > cut_best)
        reported = frame['score_top'].to_numpy(dtype=np.float64) > cut_top

    frame['found'] = found
    frame['rank_only'] = in_top
    frame['top1'] = found & (rank < 1)
    frame['top5'] = found & (rank < 5)
    frame['top10'] = found & (rank < 10)
    frame['threshold_only'] = over
    frame['operating_point'] = in_top & over
    frame['reciprocal_rank'] = np.where(found, 1.0/(np.maximum(rank, 0) + 1.0), 0.0)
    # Whether the program would report an answer at all, and whether its top answer is wrong.
    # The threshold exists to refuse these, and the operating point cannot see them: it is
    # monotone in the threshold, so maximising it alone drives the threshold to minus infinity.
    frame['reported'] = reported
    frame['false_positive'] = reported & ~frame['top_is_correct'].to_numpy(dtype=bool)
    frame['abstain'] = ~reported
    # Off-by-two is its own class, not a failure (PLAN section 6.5).
    frame['off_by_two'] = frame[f'n_off_by_two_{pool_subset}'].to_numpy() > 0
    # An entry whose only correct candidates are Mighell-Santoro degenerates has no solution
    # reachable from positions alone, so it leaves the loss decomposition's denominator instead
    # of counting as a FOM failure.
    frame['degenerate_only'] = (
        frame[f'has_correct_incl_degenerate_{pool_subset}'].to_numpy(dtype=bool) & ~found
        )
    frame['lost_not_found'] = ~found & ~frame['degenerate_only'].to_numpy(dtype=bool)
    frame['lost_rank_failure'] = found & ~in_top & over
    frame['lost_threshold_failure'] = found & in_top & ~over
    frame['lost_both'] = found & ~in_top & ~over
    frame['lost_reachable'] = found & ~frame['operating_point'].to_numpy(dtype=bool)
    return frame


# ---------------------------------------------------------------------------------------
# Summaries, weighting and the bootstrap
# ---------------------------------------------------------------------------------------
def _bootstrap_replicates(clusters, n_bootstrap, seed):
    """One set of cluster draws, reused by every metric and every stratum.

    Sharing the replicates makes all the intervals and all the paired deltas mutually
    consistent, which is what lets a McNemar delta be quoted beside a metric's own interval.
    """
    n_clusters = int(clusters.max()) + 1 if clusters.size else 0
    if n_bootstrap <= 0 or n_clusters == 0:
        return np.zeros((0, n_clusters), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(int(n_bootstrap), n_clusters))


def _cluster_ci(flags, clusters, replicates):
    """Percentile 95% CI over resampled *source entries*.

    Resampling rows would treat one crystal's seven conditions as seven independent observations
    and shrink the interval by up to sqrt(7); PROTOCOL section 8 asks for entries.

    On `fom` this also took `lattice_codes` and `lattice_weights`, and recomputed per-lattice means
    inside every replicate so the interval reflected the CNRS reweighting. Campaign 2 has no
    reweighting, so that branch is gone with it.
    """
    if replicates.shape[0] == 0 or flags.size == 0:
        return (np.nan, np.nan)
    n_clusters = replicates.shape[1]
    success = np.bincount(clusters, weights=flags, minlength=n_clusters)
    count = np.bincount(clusters, minlength=n_clusters).astype(float)
    drawn = np.take(success, replicates).sum(axis=1)
    totals = np.take(count, replicates).sum(axis=1)
    values = np.divide(drawn, totals, out=np.full(drawn.shape, np.nan), where=totals > 0)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def unweighted_mean(frame, column):
    """The plain mean of a per-entry flag, over entries.

    On `fom` this was `weighted_mean` and reweighted to the CNRS lattice distribution by default.
    Campaign 2 has one aggregate and it is this one (PROTOCOL section 3 rule 6); the per-lattice
    rows are reported beside it and are where a lattice-specific claim is read.
    """
    values = frame[column].to_numpy(dtype=np.float64)
    if values.size == 0:
        return np.nan
    return float(np.nanmean(values))


def _metric_block(frame, has_threshold, replicates=None):
    """The metric columns for one scope."""
    block = {name: np.nan for name in METRIC_COLUMNS}
    block['n_entries'] = int(frame.shape[0])
    block['n_found'] = int(frame['found'].sum()) if frame.shape[0] else 0
    if frame.shape[0] == 0:
        return block
    for name in _FLAG_METRICS:
        block[name] = unweighted_mean(frame, name)
    block['mrr'] = unweighted_mean(frame, 'reciprocal_rank')
    # The operating point among the entries that *have* a reachable solution: the FOM's own job,
    # separated from the generator's. It matters wherever generation failure dominates -- in the
    # designated hard stratum the unconditional operating point is 0.05% against a ceiling of
    # 13%, so an unconditional number there is a statement about candidate generation (F-059).
    found = frame.loc[frame['found'].to_numpy(dtype=bool)]
    block['operating_point_given_found'] = (unweighted_mean(found, 'operating_point')
                                            if found.shape[0] else np.nan)
    # A perfect re-ranker permutes the pool but cannot change a candidate's score, so its
    # reachable operating point is exactly `threshold_only`. A perfect re-*scorer* reaches
    # every entry whose pool contains a correct candidate at all. Both are reported, because
    # the difference between them is the cleanest statement of why S07 exists.
    block['ceiling_reranker'] = block['threshold_only']
    block['ceiling_rescorer'] = block['found']
    block['headroom_reranker'] = block['ceiling_reranker'] - block['operating_point']
    block['headroom_rescorer'] = block['ceiling_rescorer'] - block['operating_point']
    reported = frame['reported'].to_numpy(dtype=bool)
    top_correct = frame['top_is_correct'].to_numpy(dtype=bool)
    block['precision'] = float(top_correct[reported].mean()) if reported.any() else np.nan
    # The shares are over the reachable-lost subset and sum to one: the three indicators are
    # mutually exclusive and exhaustive on it.
    reachable = frame.loc[frame['lost_reachable'].to_numpy(dtype=bool)]
    for name, source in (('share_rank_failure', 'lost_rank_failure'),
                         ('share_threshold_failure', 'lost_threshold_failure'),
                         ('share_both', 'lost_both')):
        block[name] = (unweighted_mean(reachable, source) if reachable.shape[0] else np.nan)
    if replicates is not None and replicates.shape[0]:
        clusters = frame['cluster'].to_numpy()
        for metric in ('operating_point', 'found'):
            low, high = _cluster_ci(
                frame[metric].to_numpy(dtype=np.float64), clusters, replicates)
            block[f'{metric}_ci_low'] = low
            block[f'{metric}_ci_high'] = high
    if not has_threshold:
        for name in _THRESHOLD_METRICS:
            block[name] = np.nan
    return block


def _scope_row(frame, has_threshold, replicates=None, **identifiers):
    row = dict(identifiers)
    row['weighted'] = False
    row['n_clusters'] = int(frame['cluster'].nunique()) if frame.shape[0] else 0
    row['n_lattices'] = int(frame['bravais_lattice'].nunique()) if frame.shape[0] else 0
    row.update(_metric_block(frame, has_threshold, replicates=replicates))
    return row


def _summarise(frame, scope, has_threshold, replicates):
    return pd.DataFrame([_scope_row(frame, has_threshold, replicates, scope=scope)])


def _summarise_by_stratum(per_entry, strata, has_threshold):
    """One-way marginals: every level of every stratification variable."""
    rows = []
    for stratum in strata:
        if stratum not in per_entry.columns:
            raise ValueError(f'Unknown stratum {stratum!r}')
        for level, group in per_entry.groupby(stratum, observed=True, dropna=False):
            rows.append(_scope_row(group, has_threshold, stratum=stratum, level=level))
    return pd.DataFrame(rows)


def _summarise_cells(per_entry, strata, has_threshold):
    """The full cross of the strata, for the cells a one-way marginal cannot show."""
    strata = [stratum for stratum in strata if stratum in per_entry.columns]
    if not strata:
        return pd.DataFrame()
    rows = []
    for levels, group in per_entry.groupby(list(strata), observed=True, dropna=False):
        levels = levels if isinstance(levels, tuple) else (levels,)
        rows.append(_scope_row(group, has_threshold, **dict(zip(strata, levels))))
    return pd.DataFrame(rows)


def _loss_table(per_entry, strata, has_threshold):
    """The loss decomposition, in aggregate, on the hard stratum, and by every stratum.

    Two denominators, both reported, because two different questions are asked of this table.
    `lost_*` are fractions of all entries, which is `run_fom_mirror_analysis.classify`'s
    convention and keeps S02's verdict comparable. `share_*` are fractions of the entries that
    were lost *and* had a reachable solution, which is the S05 handoff's denominator and sums
    to one.
    """
    scopes = [('all', 'all', per_entry), ('hard', 'hard', per_entry.loc[per_entry['is_hard']])]
    for stratum in strata:
        if stratum not in per_entry.columns:
            continue
        for level, group in per_entry.groupby(stratum, observed=True, dropna=False):
            scopes.append((stratum, level, group))
    rows = []
    for stratum, level, group in scopes:
        row = dict(stratum=stratum, level=level, weighted=False,
                   n_entries=int(group.shape[0]),
                   n_lost_reachable=int(group['lost_reachable'].sum()) if group.shape[0] else 0,
                   n_degenerate_only=(int(group['degenerate_only'].sum()) if group.shape[0]
                                      else 0))
        for name in ('operating_point', 'lost_rank_failure', 'lost_threshold_failure',
                     'lost_both', 'lost_not_found', 'share_rank_failure',
                     'share_threshold_failure', 'share_both'):
            row[name] = np.nan
        if group.shape[0]:
            for name in ('operating_point', 'lost_rank_failure', 'lost_threshold_failure',
                         'lost_both', 'lost_not_found'):
                row[name] = unweighted_mean(group, name)
            reachable = group.loc[group['lost_reachable']]
            for name, source in (('share_rank_failure', 'lost_rank_failure'),
                                 ('share_threshold_failure', 'lost_threshold_failure'),
                                 ('share_both', 'lost_both')):
                row[name] = (unweighted_mean(reachable, source)
                             if reachable.shape[0] else np.nan)
            if not has_threshold:
                for name in ('operating_point', 'lost_rank_failure', 'lost_threshold_failure',
                             'lost_both', 'share_rank_failure', 'share_threshold_failure',
                             'share_both'):
                    row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_curve(per_entry, scope='all'):
    """Operating point and false-positive rate at every threshold that changes a decision.

    The candidate thresholds are the scores themselves: no grid, no bin width, and no threshold
    reported as if it were distinguishable from the one beside it.
    """
    frame = per_entry if scope == 'all' else per_entry.loc[per_entry['is_hard']]
    if frame.shape[0] == 0:
        return pd.DataFrame()
    scores = np.concatenate([frame['score_best_correct'].to_numpy(dtype=np.float64),
                             frame['score_top'].to_numpy(dtype=np.float64)])
    scores = np.unique(scores[np.isfinite(scores)])
    if scores.size > 2000:
        # A curve is a curve; 2 000 points is more than any figure or any argmax needs.
        scores = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 2000)))
    rank_ok = frame['rank_only'].to_numpy(dtype=bool)
    score_best = frame['score_best_correct'].to_numpy(dtype=np.float64)
    score_top = frame['score_top'].to_numpy(dtype=np.float64)
    top_correct = frame['top_is_correct'].to_numpy(dtype=bool)
    codes, lattice_index = pd.factorize(frame['bravais_lattice'], sort=True)
    n_lattices = len(lattice_index)
    # One sort per (mask, lattice) and a searchsorted per threshold, rather than a groupby per
    # threshold: at benchmark scale with 2 000 thresholds the groupby version took 13 s, and S06
    # calls this once per figure of merit and again inside every threshold selection.
    counts = np.bincount(codes, minlength=n_lattices).astype(float)
    operating = _counts_above(score_best, rank_ok, codes, n_lattices, scores)
    reported = _counts_above(score_top, np.ones(codes.size, dtype=bool), codes, n_lattices,
                             scores)
    false_positive = _counts_above(score_top, ~top_correct, codes, n_lattices, scores)
    true_positive = _counts_above(score_top, top_correct, codes, n_lattices, scores)
    reported_total = reported.sum(axis=1)
    curve = pd.DataFrame(dict(
        threshold=scores.astype(float),
        operating_point=_pooled_rate(operating, counts),
        false_positive_rate=_pooled_rate(false_positive, counts),
        abstain_rate=_pooled_rate(counts[np.newaxis, :] - reported, counts),
        precision=np.divide(true_positive.sum(axis=1), reported_total,
                            out=np.full(scores.size, np.nan), where=reported_total > 0),
        ))
    curve['youden'] = curve['operating_point'] - curve['false_positive_rate']
    return curve


def _counts_above(scores, mask, codes, n_lattices, thresholds):
    """(n_thresholds, n_lattices) counts of masked entries scoring strictly above a threshold."""
    out = np.zeros((thresholds.size, n_lattices))
    for lattice in range(n_lattices):
        selected = scores[mask & (codes == lattice)]
        # NaN never exceeds a threshold and would break the search; +inf always does and stays.
        selected = np.sort(selected[~np.isnan(selected)])
        out[:, lattice] = selected.size - np.searchsorted(selected, thresholds, side='right')
    return out


def _pooled_rate(numerator, denominator):
    """One rate over every entry, pooled across lattices -- each entry counting exactly once.

    On `fom` this was `_weighted_rate`, which formed per-lattice rates and combined them with the
    CNRS weights. Setting those weights to one would NOT have made it unweighted: a mean of
    per-lattice rates still weights an entry by the reciprocal of its lattice's size, which is a
    reweighting to a uniform-over-lattices distribution and carries the same effective-sample loss
    PROTOCOL section 3 rule 6 objects to. Pooling is what "unweighted" means here.

    It also has to agree with `unweighted_mean`, which the summary tables use: `select_threshold`
    reads its budget off this curve while `evaluate` reports the operating point off those, and a
    macro curve against a micro summary would enforce the false-positive budget on a different
    quantity from the one reported.
    """
    totals = denominator.sum()
    if totals <= 0:
        return np.full(numerator.shape[0], np.nan)
    return numerator.sum(axis=1)/totals


# ---------------------------------------------------------------------------------------
# Paired comparison, threshold selection, calibration
# ---------------------------------------------------------------------------------------
def entry_digest(per_entry):
    """Short digest of the (entry, bundle) set a result covers.

    Two scores compared on different entry sets is the mistake the handoff asks the API to
    guard against. Comparing lengths would not catch it, so the key set itself is hashed --
    same construction as `FomBenchmark.q2_digest`.
    """
    keys = sorted(zip(per_entry['entry_id'].astype(str),
                      per_entry['condition_bundle'].astype(str)))
    payload = '\n'.join(f'{entry}\t{bundle}' for entry, bundle in keys).encode('utf-8')
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def mcnemar(result_a, result_b, metric='operating_point', subset=None):
    """Paired McNemar test of two scores on the same entries.

    Raises when the two results do not cover the same (entry_id, condition_bundle) set: a
    comparison across different entry sets is not a comparison, and the S05 handoff asks for
    that to be impossible rather than merely discouraged.
    """
    left = result_a.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index()
    right = result_b.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index()
    if result_a.meta['entry_digest'] != result_b.meta['entry_digest']:
        only_left = left.index.difference(right.index)
        only_right = right.index.difference(left.index)
        raise ValueError(
            'The two results cover different entry sets, so they cannot be paired: '
            f'{len(only_left)} only in A (e.g. {only_left[:3].tolist()}), '
            f'{len(only_right)} only in B (e.g. {only_right[:3].tolist()})'
            )
    right = right.reindex(left.index)
    # The string check is guarded by `isinstance` rather than written as `subset == 'hard'`:
    # comparing an ndarray to a string returns an array, and `elif` on it raises "truth value of
    # an array is ambiguous" -- which made the documented boolean-mask path unusable.
    if subset is None:
        mask = np.ones(left.shape[0], dtype=bool)
    elif isinstance(subset, str):
        if subset != 'hard':
            raise ValueError(f"subset must be None, 'hard', or a boolean mask; got {subset!r}")
        mask = left['is_hard'].to_numpy(dtype=bool)
    else:
        mask = np.asarray(subset, dtype=bool)
        if mask.shape != (left.shape[0],):
            raise ValueError(
                f'subset mask has shape {mask.shape}, expected ({left.shape[0]},). It must be '
                'aligned to the result sorted by (entry_id, condition_bundle).')
    flags_a = left.loc[mask, metric].to_numpy(dtype=bool)
    flags_b = right.loc[mask, metric].to_numpy(dtype=bool)
    n_a_only = int(np.sum(flags_a & ~flags_b))
    n_b_only = int(np.sum(~flags_a & flags_b))
    discordant = n_a_only + n_b_only
    if discordant == 0:
        statistic, p_value, method = 0.0, 1.0, 'no discordant pairs'
    elif discordant < 25:
        # The chi-square approximation is unreliable on few discordant pairs, and a hard
        # stratum cell can easily have fewer than 25.
        from scipy.stats import binomtest
        statistic = float(min(n_a_only, n_b_only))
        p_value = float(binomtest(min(n_a_only, n_b_only), discordant, 0.5).pvalue)
        method = 'exact'
    else:
        from scipy.stats import chi2
        statistic = float((abs(n_a_only - n_b_only) - 1)**2/discordant)
        p_value = float(chi2.sf(statistic, 1))
        method = 'chi2 (Edwards continuity correction)'
    return pd.Series(dict(
        metric=metric,
        subset='all' if subset is None else ('hard' if isinstance(subset, str) else 'mask'),
        n_entries=int(mask.sum()), n_clusters=int(left.loc[mask, 'cluster'].nunique()),
        n_both=int(np.sum(flags_a & flags_b)), n_a_only=n_a_only, n_b_only=n_b_only,
        n_neither=int(np.sum(~flags_a & ~flags_b)),
        delta=float(flags_a.mean() - flags_b.mean()),
        statistic=statistic, p_value=p_value, method=method,
        ))


def stratum_mask(result, column, value):
    """A boolean mask for `mcnemar`'s `subset`, aligned the way `mcnemar` requires.

    `mcnemar` sorts both results by `(entry_id, condition_bundle)` and then applies the mask
    positionally. A caller who builds a mask straight off `result.per_entry` -- whose natural order
    is the pool's, not sorted -- gets a mask that lines up with the wrong rows and a paired test
    that silently compares unrelated entries. Nothing raises; the numbers are simply wrong.

    S09 is the first consumer of the mask path at all (campaign 1's raised on every call, F-087),
    and the per-lattice leaderboard needs one mask per lattice, so the footgun would have been
    fired fourteen times before anyone looked. Build masks with this.
    """
    frame = result.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index()
    if column not in frame.columns:
        raise KeyError(
            f'{column!r} is not a per-entry column; available strata include '
            f'{sorted(c for c in frame.columns if frame[c].dtype == object)[:8]}')
    return (frame[column] == value).to_numpy(dtype=bool)


def paired_delta_ci(result_a, result_b, metric='top10', subset=None, n_bootstrap=1000,
                    seed=12345):
    """Cluster-bootstrap interval on the paired difference `a - b`, over source entries.

    `mcnemar` gives the sign and the p-value; it does not give an interval, and S09's acceptance
    gate asks for both. Resampling is over the **source crystal**, not the (entry, condition) row:
    one crystal appears under every condition bundle with correlated noise, so treating its rows as
    independent draws gives an interval up to sqrt(n_conditions) too tight (METRICS section 8).

    The replicates come from `_bootstrap_replicates` at the same seed `evaluate` uses, so this
    interval and the marginal intervals in the summary tables are drawn from one resampling and are
    mutually consistent rather than merely both correct.
    """
    left = result_a.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index()
    right = result_b.per_entry.set_index(['entry_id', 'condition_bundle']).sort_index()
    if result_a.meta['entry_digest'] != result_b.meta['entry_digest']:
        raise ValueError(
            'The two results cover different entry sets, so they cannot be paired. '
            '`mcnemar` refuses the same comparison for the same reason.')
    right = right.reindex(left.index)

    if subset is None:
        mask = np.ones(left.shape[0], dtype=bool)
    elif isinstance(subset, str):
        if subset != 'hard':
            raise ValueError(f"subset must be None, 'hard', or a boolean mask; got {subset!r}")
        mask = left['is_hard'].to_numpy(dtype=bool)
    else:
        mask = np.asarray(subset, dtype=bool)
        if mask.shape != (left.shape[0],):
            raise ValueError(
                f'subset mask has shape {mask.shape}, expected ({left.shape[0]},). Build it with '
                f'`stratum_mask`, which aligns it to the sorted order this function uses.')

    difference = (left.loc[mask, metric].to_numpy(dtype=float)
                  - right.loc[mask, metric].to_numpy(dtype=float))
    clusters = left.loc[mask, 'cluster'].to_numpy()
    point = float(difference.mean()) if difference.size else float('nan')
    if not difference.size or n_bootstrap <= 0:
        return pd.Series(dict(metric=metric, delta=point, ci_low=np.nan, ci_high=np.nan,
                              n_entries=int(mask.sum()), n_clusters=int(len(set(clusters))),
                              n_bootstrap=int(max(0, n_bootstrap))))

    codes, unique = pd.factorize(clusters)
    replicates = _bootstrap_replicates(codes, n_bootstrap, seed)
    totals = np.bincount(codes, weights=difference, minlength=unique.size)
    counts = np.bincount(codes, minlength=unique.size)
    drawn_totals = totals[replicates].sum(axis=1)
    drawn_counts = counts[replicates].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.where(drawn_counts > 0, drawn_totals/drawn_counts, np.nan)
    low, high = np.nanpercentile(means, [2.5, 97.5])
    return pd.Series(dict(metric=metric, delta=point, ci_low=float(low), ci_high=float(high),
                          n_entries=int(mask.sum()), n_clusters=int(unique.size),
                          n_bootstrap=int(n_bootstrap)))


class ThresholdChoice:
    """A threshold and the evidence for it, carried so it cannot be silently reused."""

    def __init__(self, threshold, objective, value, curve, entry_digest, split, meta):
        self.threshold = threshold
        self.objective = objective
        self.value = value
        self.curve = curve
        self.entry_digest = entry_digest
        self.split = split
        self.meta = meta

    def to_dict(self):
        payload = dict(threshold=self.threshold, objective=self.objective, value=self.value,
                       entry_digest=self.entry_digest, split=self.split)
        payload.update(self.meta)
        return payload

    def __repr__(self):
        return (f'ThresholdChoice(threshold={self.threshold:.6g}, '
                f'objective={self.objective!r}, value={self.value:.4f})')


def select_threshold(result, objective='youden', max_false_positive_rate=None, subset=None):
    """Choose an operating threshold on the split the caller passes in.

    **The operating point alone cannot be maximised.** It is monotone non-increasing in the
    threshold -- lowering the threshold can only admit more entries -- so its maximiser is
    always minus infinity, where the program reports an answer for every pattern and the
    threshold does nothing at all. The S06 handoff's "maximise the operating point" is
    therefore not implementable as written (see STATUS F-058), and a threshold is instead
    chosen against something that punishes reporting a wrong cell:

        'youden'           maximise operating_point - false_positive_rate (the default)
        'operating_point'  maximise operating_point subject to max_false_positive_rate

    S06 selects on `fom-train` and reports on `fom-dev`; `check_threshold_transfer` refuses to
    reuse a choice on the entries it was selected on.
    """
    frame = (result.per_entry if subset != 'hard'
             else result.per_entry.loc[result.per_entry['is_hard']])
    curve = threshold_curve(frame, scope='all')
    if curve.shape[0] == 0:
        raise ValueError('No thresholds to choose between; the result is empty')
    if objective == 'youden':
        index = int(curve['youden'].to_numpy().argmax())
        value = float(curve['youden'].iloc[index])
    elif objective == 'operating_point':
        if max_false_positive_rate is None:
            raise ValueError(
                "objective='operating_point' needs max_false_positive_rate: the unconstrained "
                'maximiser of the operating point is minus infinity, where the threshold '
                'rejects nothing.'
                )
        curve = curve.loc[curve['false_positive_rate'] <= max_false_positive_rate]
        if curve.shape[0] == 0:
            raise ValueError(
                f'No threshold keeps the false-positive rate at or below '
                f'{max_false_positive_rate}'
                )
        index = int(curve['operating_point'].to_numpy().argmax())
        value = float(curve['operating_point'].iloc[index])
    else:
        raise ValueError(f'Unknown objective {objective!r}')
    return ThresholdChoice(
        threshold=float(curve['threshold'].iloc[index]), objective=objective, value=value,
        curve=curve, entry_digest=result.meta['entry_digest'], split=result.meta.get('split'),
        meta=dict(score=result.meta['score'], subset='hard' if subset == 'hard' else 'all',
                  weighted=False, max_false_positive_rate=max_false_positive_rate),
        )


def check_threshold_transfer(choice, result, allow_same_entries=False):
    """Refuse a threshold chosen on the very entries it is about to be reported on."""
    if not allow_same_entries and choice.entry_digest == result.meta['entry_digest']:
        raise ValueError(
            'This threshold was selected on the same entries it is being applied to. '
            'PROTOCOL section 8: select on fom-train, report on fom-dev. Pass '
            'allow_same_entries=True only for a deliberate in-sample diagnostic.'
            )


def reliability(probability, is_correct, n_bins=10):
    """Equal-count reliability table, with the bin counts beside it.

    Equal-count rather than equal-width because a score's distribution is nothing like uniform:
    equal-width bins put almost every candidate in one bin and then report a confident ECE from
    the few dozen that landed elsewhere. Duplicate quantile edges are collapsed, so a constant
    predictor honestly reports one bin rather than ten copies of itself.
    """
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(is_correct, dtype=bool)
    if probability.size == 0:
        return pd.DataFrame(), np.nan, np.nan
    edges = np.unique(np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([probability[0] - 0.5, probability[0] + 0.5])
    index = np.clip(np.searchsorted(edges, probability, side='left') - 1, 0, edges.size - 2)
    rows = []
    for position in range(edges.size - 1):
        mask = index == position
        if not mask.any():
            continue
        rows.append(dict(bin=len(rows), p_low=float(edges[position]),
                         p_high=float(edges[position + 1]),
                         p_mean=float(probability[mask].mean()),
                         observed=float(labels[mask].mean()), n=int(mask.sum())))
    table = pd.DataFrame(rows)
    total = float(probability.size)
    ece = float((table['n']/total*(table['p_mean'] - table['observed']).abs()).sum())
    brier = float(np.mean((probability - labels.astype(float))**2))
    return table, ece, brier


def average_precision(score, is_correct):
    """Candidate-level PR AUC, with exact ties handled as one group.

    Useful for model selection during training, and deliberately *not* a headline: the task is
    per-entry ranking, not per-candidate classification, and the pool's positive prevalence
    varies several-fold between condition bundles (F-049, F-054).
    """
    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(is_correct, dtype=bool)
    if score.size == 0 or labels.sum() == 0:
        return np.nan
    order = np.argsort(-score, kind='stable')
    labels_sorted = labels[order]
    score_sorted = score[order]
    # Group exact ties: no threshold separates them, so precision and recall step once.
    ends = np.append(np.flatnonzero(np.diff(score_sorted)) + 1, score_sorted.size) - 1
    true_positive = np.cumsum(labels_sorted)[ends]
    precision = true_positive/(ends + 1)
    recall = true_positive/labels.sum()
    return float(np.sum(np.diff(np.append(0.0, recall))*precision))


def _calibration_table(rows):
    """Reliability per condition bundle, or the reason there is none.

    Refused outright for a score outside [0, 1]: an ECE for raw M20 would be a number with no
    meaning, and on C0 it would be a number dominated by 1.3M candidates scoring above 1e9
    (F-054). Calibration is defined on probabilities; turning a merit into one is S07's job.
    """
    if rows is None or rows.shape[0] == 0:
        return pd.DataFrame(), None
    scores = rows['score'].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        return pd.DataFrame(), 'the score has non-finite values'
    if scores.min() < 0.0 or scores.max() > 1.0:
        return pd.DataFrame(), (
            f'the score is not a probability (range {scores.min():.4g} to {scores.max():.4g}); '
            'calibrate it first'
            )
    tables = []
    for bundle, group in rows.groupby('condition_bundle', observed=True):
        table, ece, brier = reliability(group['score'], group['is_correct'])
        if table.shape[0] == 0:
            continue
        table.insert(0, 'condition_bundle', bundle)
        table['ece'] = ece
        table['brier'] = brier
        table['auc_pr'] = average_precision(group['score'], group['is_correct'])
        tables.append(table)
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), None


def write_result(result, out_dir, tag):
    """Persist a result's tables and metadata, so a reported number has an artefact."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in result.tables().items():
        if table is None or table.shape[0] == 0:
            continue
        path = out_dir / f'{tag}_{name}.csv'
        table.to_csv(path, index=False, encoding='utf-8')
        written[name] = str(path)
    path = out_dir / f'{tag}_meta.json'
    with open(path, 'w', encoding='utf-8') as meta_file:
        json.dump(result.meta, meta_file, indent=2, sort_keys=True, default=str)
    written['meta'] = str(path)
    return written
