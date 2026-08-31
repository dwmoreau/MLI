"""S08 -- the metrics module, which is the measuring instrument for the whole campaign.

Every number this project reports comes out of `FomMetrics`, so every definition that could
plausibly be implemented two ways gets a hand-built frame with an arithmetic answer rather than a
smoke test. Taken from `fom` @ `7c137c3` and reduced: the three tests that exercised the CNRS
lattice reweighting are gone with the capability itself (`CHERRY_PICK.md`), and campaign 1's
acceptance gate against Benchmark A is replaced by one against Benchmark B's slice.

Two tests here are load-bearing beyond their own assertion:

* `test_oracle_reranker_equals_threshold_only_and_rescorer_equals_found` is S08 acceptance
  condition 2. A re-ranker permutes the pool but cannot change a candidate's score, so its
  ceiling is *identically* the threshold-only rate -- and the test is built so an implementation
  that computes the two separately fails it.
* `test_mcnemar_accepts_a_boolean_mask_subset` pins a defect that made every per-stratum paired
  test in campaign 1 raise, so none was ever run. Comparing an ndarray to a string returns an
  array, and `elif` on it raises "truth value of an array is ambiguous".
"""
import json

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics


# Benchmark B's slice, written on NERSC by `run_fom_pool_subset.py` and copied here. The full pool
# is ~122 GB on Perlmutter scratch and never comes to a laptop; the gate that needs the whole pool
# runs there. Absent, the gate skips rather than failing -- it is untracked generated data.
SLICE_ROOT = FomBenchmark.Path(__file__).parent.parent / 'mlindex' / 'data' / 'fom_benchmark_c2'
GATE_TOLERANCE = 0.005


def _tiny(rows, entries=None, bundle='c2_error1_cont0'):
    """Build a candidate frame from (entry_id, bravais_lattice, score, is_correct) tuples.

    `rows` may carry two optional trailing fields, `is_off_by_two` and `is_degenerate`, and an
    optional `in_top_n`; everything else the module needs is filled in.
    """
    records = []
    per_entry_counter = {}
    for row in rows:
        entry_id, lattice, score, is_correct = row[:4]
        extra = list(row[4:]) + [False, False, True][len(row) - 4:]
        off_by_two, degenerate, in_top_n = extra[0], extra[1], extra[2]
        key = (entry_id, lattice)
        per_entry_counter[key] = per_entry_counter.get(key, -1) + 1
        records.append(dict(
            entry_id=entry_id, condition_bundle=bundle, bravais_lattice=lattice,
            candidate_id=per_entry_counter[key], score=float(score),
            is_correct=bool(is_correct), is_off_by_two=bool(off_by_two),
            is_degenerate=degenerate, in_top_n=bool(in_top_n),
            ))
    candidates = pd.DataFrame(records)
    candidates['is_degenerate'] = candidates['is_degenerate'].astype('boolean')
    if entries is None:
        entries = {entry_id: 'oP' for entry_id in candidates['entry_id'].unique()}
    entry_frame = pd.DataFrame([
        dict(entry_id=entry_id, condition_bundle=bundle, split='fom-dev',
             bravais_lattice_true=lattice, lattice_system_true='orthorhombic',
             volume_true=1000.0 + 10*position)
        for position, (entry_id, lattice) in enumerate(entries.items())
        ])
    return candidates, entry_frame


def _evaluate(rows, entries=None, **kwargs):
    candidates, entry_frame = _tiny(rows, entries=entries)
    kwargs.setdefault('score', 'score')
    kwargs.setdefault('n_bootstrap', 0)
    return FomMetrics.evaluate(candidates, entries=entry_frame, **kwargs)


# ---------------------------------------------------------------------------------------
# Pooling -- the project's worst anti-pattern, so it is the first test
# ---------------------------------------------------------------------------------------
def test_pooled_ranking_is_cross_lattice():
    # The correct cP candidate is 3rd of its own lattice but 6th once the four oP candidates
    # that outscore it join the pool. With top_n=5 the two poolings disagree, which is the
    # whole of PROTOCOL section 10's warning.
    rows = [
        ('E1', 'cP', 10.0, False), ('E1', 'cP', 9.0, False), ('E1', 'cP', 5.0, True),
        ('E1', 'oP', 8.0, False), ('E1', 'oP', 7.0, False), ('E1', 'oP', 6.5, False),
        ('E1', 'oP', 6.0, False),
        ]
    cross = _evaluate(rows, pool='cross_bl', top_n=5)
    per_bl = _evaluate(rows, pool='per_bl', top_n=5)
    assert cross.per_entry['rank_best_correct'].iloc[0] == 6
    assert per_bl.per_entry['rank_best_correct'].iloc[0] == 2
    assert cross.metric('rank_only') == 0.0
    assert per_bl.metric('rank_only') == 1.0


def test_threshold_only_does_not_depend_on_the_pooling():
    # `threshold_only` is a statement about scores alone, so it must be identical under both
    # poolings. It is not if the reduction picks the best-*ranked* correct candidate instead of
    # the best-*scoring* one: here cP's rank-0 correct candidate scores 8 and mP's rank-2 one
    # scores 15.
    rows = [
        ('E1', 'cP', 8.0, True),
        ('E1', 'mP', 30.0, False), ('E1', 'mP', 20.0, False), ('E1', 'mP', 15.0, True),
        ]
    cross = _evaluate(rows, pool='cross_bl', threshold=10)
    per_bl = _evaluate(rows, pool='per_bl', threshold=10)
    assert cross.metric('threshold_only') == 1.0
    assert per_bl.metric('threshold_only') == 1.0


def test_in_top_n_subset_changes_the_ceiling_but_not_the_operating_point():
    # A pooled top-10 member is necessarily inside its own lattice's top twenty, so restricting
    # to the reported subset can only remove candidates that were never going to be reported.
    rows = [
        ('E1', 'oP', 50.0, False, False, False, True),
        ('E1', 'oP', 20.0, True, False, False, True),
        ('E2', 'oP', 50.0, False, False, False, True),
        ('E2', 'oP', 1.0, True, False, False, False),   # correct, but not kept by the pipeline
        ]
    everything = _evaluate(rows, threshold=10, pool_subset='all')
    reported = _evaluate(rows, threshold=10, pool_subset='in_top_n')
    assert everything.metric('found') == 1.0
    assert reported.metric('found') == 0.5
    assert everything.metric('operating_point') == reported.metric('operating_point') == 0.5


# ---------------------------------------------------------------------------------------
# The loss decomposition -- the measurement that answers STATUS Q2
# ---------------------------------------------------------------------------------------
def test_loss_decomposition_buckets_are_exclusive_and_complete():
    rows = []
    # E1 succeeds: correct, rank 0, score above the threshold.
    rows += [('E1', 'oP', 20.0, True)]
    # E2 is a threshold failure: rank 0, score below.
    rows += [('E2', 'oP', 5.0, True)]
    # E3 is a rank failure: score above, but two better-scoring wrong candidates push it out.
    rows += [('E3', 'oP', 20.0, False), ('E3', 'oP', 19.0, False), ('E3', 'oP', 18.0, True)]
    # E4 fails both: below the threshold and outranked.
    rows += [('E4', 'oP', 20.0, False), ('E4', 'oP', 19.0, False), ('E4', 'oP', 5.0, True)]
    # E5 has no correct candidate at all -- a generation failure, not a ranking one.
    rows += [('E5', 'oP', 20.0, False)]
    result = _evaluate(rows, threshold=10, top_n=2)
    aggregate = result.aggregate.iloc[0]
    assert aggregate['operating_point'] == pytest.approx(0.2)
    assert aggregate['lost_threshold_failure'] == pytest.approx(0.2)
    assert aggregate['lost_rank_failure'] == pytest.approx(0.2)
    assert aggregate['lost_both'] == pytest.approx(0.2)
    assert aggregate['lost_not_found'] == pytest.approx(0.2)
    total = sum(aggregate[name] for name in ('operating_point', 'lost_rank_failure',
                                            'lost_threshold_failure', 'lost_both',
                                            'lost_not_found'))
    assert total == pytest.approx(1.0)
    # The three shares use the reachable-lost denominator and sum to one.
    shares = sum(aggregate[name] for name in ('share_rank_failure', 'share_threshold_failure',
                                              'share_both'))
    assert shares == pytest.approx(1.0)


def test_not_found_is_never_inside_the_decomposition():
    rows = [('E1', 'oP', 20.0, False)]
    result = _evaluate(rows, threshold=10)
    assert result.metric('lost_not_found') == 1.0
    assert result.aggregate['n_entries'].iloc[0] == 1
    loss = result.loss.loc[result.loss['stratum'] == 'all'].iloc[0]
    assert loss['n_lost_reachable'] == 0
    assert np.isnan(loss['share_rank_failure'])


def test_degenerates_leave_the_loss_denominator():
    # E1's only correct candidate is a Mighell-Santoro degenerate: no position-only FOM can
    # separate it, so it must not be counted as a FOM failure (PLAN section 6.5).
    rows = [('E1', 'oP', 5.0, True, False, True), ('E2', 'oP', 5.0, True, False, pd.NA)]
    excluded = _evaluate(rows, threshold=10, degenerates='exclude')
    included = _evaluate(rows, threshold=10, degenerates='include')
    assert excluded.metric('found') == 0.5
    assert excluded.metric('degenerate_only') == 0.5
    assert excluded.loss.loc[excluded.loss['stratum'] == 'all', 'n_lost_reachable'].iloc[0] == 1
    assert included.metric('found') == 1.0
    assert included.metric('degenerate_only') == 0.0
    assert included.loss.loc[included.loss['stratum'] == 'all', 'n_lost_reachable'].iloc[0] == 2


# ---------------------------------------------------------------------------------------
# Rank metrics
# ---------------------------------------------------------------------------------------
def test_mrr_counts_absent_entries_as_zero():
    rows = []
    for entry_id, rank in (('E1', 0), ('E2', 1), ('E3', 3)):
        rows += [('%s' % entry_id, 'oP', 100.0 - position, position == rank)
                 for position in range(rank + 1)]
    rows += [('E4', 'oP', 50.0, False)]
    result = _evaluate(rows)
    # 1 + 1/2 + 1/4 + 0, over four entries.
    assert result.metric('mrr') == pytest.approx(0.4375)


def test_rank_metrics_nest():
    rng = np.random.default_rng(0)
    rows = []
    for entry in range(30):
        n_candidates = int(rng.integers(1, 12))
        correct_at = int(rng.integers(0, n_candidates + 2))
        for position in range(n_candidates):
            rows.append((f'E{entry}', 'oP', float(100 - position), position == correct_at))
    result = _evaluate(rows, threshold=50)
    aggregate = result.aggregate.iloc[0]
    assert aggregate['top1'] <= aggregate['top5'] <= aggregate['top10'] <= aggregate['found']
    assert aggregate['operating_point'] <= min(aggregate['rank_only'],
                                               aggregate['threshold_only'])


def test_oracle_reranker_equals_threshold_only_and_rescorer_equals_found():
    # A perfect re-ranker permutes the pool but cannot change a candidate's score, so its
    # reachable operating point is exactly `threshold_only`. Constructed so a naive
    # `oracle = found` would differ: E2's correct candidate is unreachable at this threshold.
    rows = [('E1', 'oP', 20.0, False), ('E1', 'oP', 19.0, True),
            ('E2', 'oP', 20.0, False), ('E2', 'oP', 5.0, True)]
    result = _evaluate(rows, threshold=10, top_n=1)
    assert result.metric('found') == 1.0
    assert result.metric('ceiling_rescorer') == 1.0
    assert result.metric('ceiling_reranker') == 0.5
    assert result.metric('ceiling_reranker') == result.metric('threshold_only')
    assert result.metric('operating_point') == 0.0
    assert result.metric('headroom_reranker') == pytest.approx(0.5)
    assert result.metric('headroom_rescorer') == pytest.approx(1.0)


def test_operating_point_given_found_separates_the_fom_from_the_generator():
    # Two entries have no correct candidate at all -- a generation failure. The unconditional
    # operating point is 1/4; the FOM's own hit rate on the entries it could have got is 1/2.
    rows = [('E1', 'oP', 20.0, True), ('E2', 'oP', 5.0, True),
            ('E3', 'oP', 20.0, False), ('E4', 'oP', 20.0, False)]
    result = _evaluate(rows, threshold=10)
    assert result.metric('operating_point') == pytest.approx(0.25)
    assert result.metric('operating_point_given_found') == pytest.approx(0.5)
    assert result.aggregate['n_found'].iloc[0] == 2


def test_threshold_none_gives_rank_metrics_and_no_threshold_metrics():
    rows = [('E1', 'oP', 20.0, True)]
    result = _evaluate(rows, threshold=None)
    assert result.metric('top1') == 1.0
    assert np.isnan(result.metric('operating_point'))
    assert np.isnan(result.metric('threshold_only'))
    assert np.isnan(result.metric('lost_threshold_failure'))
    assert result.metric('found') == 1.0


# ---------------------------------------------------------------------------------------
# Ties, orientation and non-finite scores
# ---------------------------------------------------------------------------------------
def test_ties_break_deterministically_under_row_shuffling():
    rows = [('E1', 'aP', 10.0, True), ('E1', 'cP', 10.0, False), ('E1', 'oP', 10.0, False)]
    candidates, entries = _tiny(rows)
    values = candidates['score'].to_numpy()
    baseline = FomMetrics.reduce_pool(candidates, values)
    order = np.random.default_rng(3).permutation(candidates.shape[0])
    shuffled = candidates.iloc[order].reset_index(drop=True)
    shuffled_result = FomMetrics.reduce_pool(shuffled, shuffled['score'].to_numpy())
    assert baseline['rank_best_correct_all'].iloc[0] == shuffled_result['rank_best_correct_all'].iloc[0]
    # cP precedes aP in the canonical order, so the correct aP candidate cannot be rank 0.
    assert baseline['rank_best_correct_all'].iloc[0] == 2
    assert baseline['n_ties_at_best_correct_all'].iloc[0] == 3
    assert entries.shape[0] == 1


def test_non_finite_scores_follow_numpy_ordering():
    # +inf for M20 means a zero residual -- a perfect fit -- so it ranks first. NaN carries no
    # ordering information and ranks last.
    rows = [('E1', 'oP', np.inf, True), ('E1', 'oP', 50.0, False), ('E1', 'oP', np.nan, False),
            ('E2', 'oP', np.nan, True), ('E2', 'oP', 5.0, False)]
    result = _evaluate(rows, threshold=10)
    per_entry = result.per_entry.set_index('entry_id')
    assert per_entry.loc['E1', 'rank_best_correct'] == 0
    assert per_entry.loc['E2', 'rank_best_correct'] == 1
    assert result.meta['n_non_finite_score'] == 3
    assert per_entry.loc['E1', 'n_non_finite_score'] == 2


def test_higher_is_better_false_is_an_exact_mirror():
    rows = [('E1', 'oP', 20.0, False), ('E1', 'oP', 12.0, True), ('E2', 'oP', 5.0, True)]
    ascending = [(row[0], row[1], -row[2], row[3]) for row in rows]
    forward = _evaluate(rows, threshold=10, higher_is_better=True)
    mirrored = _evaluate(ascending, threshold=-10, higher_is_better=False)
    for name in ('operating_point', 'found', 'top1', 'threshold_only', 'mrr', 'reported'):
        assert forward.metric(name) == pytest.approx(mirrored.metric(name))


# ---------------------------------------------------------------------------------------
# Weighting, strata and the bootstrap
# ---------------------------------------------------------------------------------------
def test_volume_deciles_are_computed_within_each_lattice():
    # aP's volumes are all larger than mP's; within-lattice deciles must not notice, which is
    # the whole difference from a global decile.
    frame = pd.DataFrame({
        'bravais_lattice': ['mP']*100 + ['aP']*100,
        'volume_true': list(np.arange(100.0)) + list(np.arange(10000.0, 10100.0)),
        })
    deciles = FomMetrics.volume_decile(frame)
    for block in (deciles[:100], deciles[100:]):
        assert sorted(set(block)) == list(range(10))
        counts = np.bincount(block, minlength=10)
        assert counts.min() >= 9 and counts.max() <= 11
    # The top decile is the largest cells of each lattice separately, not of the pool.
    assert deciles[99] == 9 and deciles[199] == 9
    assert deciles[100] == 0


def test_volume_deciles_reproduce_the_frozen_split_manifest():
    """The deciles the split was stratified on must be recoverable, not merely similar.

    S06 froze `docs/fom_campaign2/artifacts/S06_split_manifest.parquet` with a `volume_decile`
    column, and the hard stratum is defined against decile >= 8. If this formula drifted from
    S06's, the hard stratum would quietly become a different set of entries.

    The module now *joins* the stored decile rather than recomputing it, which is the R14 fix --
    so this test no longer guards the number the pipeline uses. It guards the fallback, which is
    what a pool without the column still gets, and it is the check that the two agree.
    """
    manifest_path = (FomBenchmark.Path(__file__).parent.parent / 'docs' / 'fom_campaign2'
                     / 'artifacts' / 'S06_split_manifest.parquet')
    if not manifest_path.exists():
        pytest.skip('the frozen split manifest is absent (docs/ is git-ignored)')
    pytest.importorskip('pyarrow')
    manifest = pd.read_parquet(manifest_path)
    recomputed = FomMetrics.volume_decile(manifest[['bravais_lattice', 'volume_true']])
    assert (recomputed == manifest['volume_decile']).all()


def _stratum_entries():
    """20 entries per (lattice, bundle), so deciles are meaningful at this scale."""
    rows = []
    for lattice in ('mP', 'oP'):
        for bundle in ('error2_cont0', 'error1_cont0', 'error1_cont0_phase3'):
            for position in range(20):
                rows.append(dict(
                    entry_id=f'{lattice}-{position:02d}', condition_bundle=bundle,
                    split='fom-dev', bravais_lattice_true=lattice,
                    lattice_system_true='monoclinic' if lattice == 'mP' else 'orthorhombic',
                    volume_true=100.0 + position,
                    ))
    return pd.DataFrame(rows)


def test_hard_stratum_membership():
    context = FomMetrics.entry_context(_stratum_entries())
    hard = context.loc[context['is_hard']]
    # Only mP, only the top two deciles, only C2 of the three bundles present.
    assert set(hard['bravais_lattice']) == {'mP'}
    assert set(hard['condition_bundle']) == {'error2_cont0'}
    assert set(hard['volume_decile']) == {8, 9}
    assert hard.shape[0] == 4
    # C6 exists in the frame and is deliberately not in the hard set (DWMM, 2026-08-17).
    assert 'error1_cont0_phase3' in set(context['condition_bundle'])
    assert {'error2_cont0', 'error1_cont2', 'error1_cont1_drop6',
            'error1_cont1_drop10'} <= set(FomMetrics.HARD_BUNDLES)


def test_the_hard_stratum_covers_both_campaigns_bundle_names():
    """Benchmark A and Benchmark B are both live, and their tags are disjoint by construction.

    `FomConditions.TAG_PREFIX` exists so a campaign-2 bundle cannot be mistaken for a campaign-1
    one, which is what makes a union of the two safe: a pool carries tags from one namespace only,
    so the other half of the union can never match a row in it.
    """
    from mlindex.model_training import FomConditions
    assert set(FomConditions.HARD_BUNDLES) <= set(FomMetrics.HARD_BUNDLES)
    campaign_1 = {name for name in FomMetrics.HARD_BUNDLES if not name.startswith('c2_')}
    campaign_2 = {name for name in FomMetrics.HARD_BUNDLES if name.startswith('c2_')}
    assert campaign_1 and campaign_2
    assert not (campaign_1 & campaign_2)


def test_a_campaign_2_hard_stratum_is_selected_the_same_way():
    """The predicate is unchanged; only the bundle names differ. The S08 handoff reads as though
    the stratum is redefined in terms of reachability -- METRICS.md section 5 makes reachability
    the *sizing* criterion for the split, not a term in this predicate (C2-F-078)."""
    rows = []
    for lattice in ('mC', 'oP'):
        for bundle in ('c2_error2_cont0', 'c2_error1_cont0'):
            for position in range(20):
                rows.append(dict(
                    entry_id=f'{lattice}-{position:02d}', condition_bundle=bundle,
                    split='fom-dev', bravais_lattice_true=lattice,
                    lattice_system_true='monoclinic' if lattice == 'mC' else 'orthorhombic',
                    volume_true=100.0 + position,
                    ))
    context = FomMetrics.entry_context(pd.DataFrame(rows))
    hard = context.loc[context['is_hard']]
    assert set(hard['bravais_lattice']) == {'mC'}
    assert set(hard['condition_bundle']) == {'c2_error2_cont0'}
    assert set(hard['volume_decile']) == {8, 9}


def test_hard_min_decile_widens_the_stratum_and_defaults_to_the_literal_one():
    """Q32's widening must be opt-in, so every number measured before it still means what it did.

    S07 fits on `fom-train` and so cannot inherit S06's licence to pool the hard stratum over
    train+dev; at the literal decile >= 8 cut `fom-dev` holds sixteen reachable entries and every
    merit's threshold metrics come back exactly 0.0000 (F-063). DWMM resolved Q32 by reporting
    hard-stratum *threshold* metrics at decile >= 6. The default is unchanged and this pins that.
    """
    entries = _stratum_entries()
    default = FomMetrics.entry_context(entries)
    explicit = FomMetrics.entry_context(entries, hard_min_decile=FomMetrics.HARD_MIN_DECILE)
    assert default['is_hard'].equals(explicit['is_hard'])
    assert FomMetrics.HARD_MIN_DECILE == 8

    widened = FomMetrics.entry_context(entries, hard_min_decile=6)
    assert set(widened.loc[widened['is_hard'], 'volume_decile']) == {6, 7, 8, 9}
    # Strictly a superset: widening the volume cut cannot drop an entry that was already hard.
    assert widened['is_hard'].sum() > default['is_hard'].sum()
    assert not (default['is_hard'] & ~widened['is_hard']).any()


def test_bootstrap_resamples_source_entries_not_rows():
    # The same crystal under seven conditions is one observation, not seven. Resampling rows
    # would shrink the interval by up to sqrt(7) and produce the absurdly tight CIs the handoff
    # warns about.
    rng = np.random.default_rng(7)
    n_entries, n_bundles = 60, 7
    flags = np.repeat(rng.random(n_entries) < 0.5, n_bundles).astype(float)
    clusters = np.repeat(np.arange(n_entries), n_bundles)
    rows = np.arange(flags.size)
    cluster_replicates = FomMetrics._bootstrap_replicates(clusters, 400, 1)
    row_replicates = FomMetrics._bootstrap_replicates(rows, 400, 1)
    cluster_low, cluster_high = FomMetrics._cluster_ci(flags, clusters, cluster_replicates)
    row_low, row_high = FomMetrics._cluster_ci(flags, rows, row_replicates)
    assert (cluster_high - cluster_low) > 2*(row_high - row_low)


def test_bootstrap_interval_brackets_the_point_estimate():
    rows = [(f'E{position}', 'oP', 20.0 if position % 2 else 5.0, True) for position in range(40)]
    result = _evaluate(rows, threshold=10, n_bootstrap=200)
    aggregate = result.aggregate.iloc[0]
    assert aggregate['operating_point_ci_low'] <= aggregate['operating_point']
    assert aggregate['operating_point'] <= aggregate['operating_point_ci_high']


# ---------------------------------------------------------------------------------------
# Paired comparison and threshold selection
# ---------------------------------------------------------------------------------------
def test_mcnemar_on_a_known_table():
    scipy_stats = pytest.importorskip('scipy.stats')
    rows = [(f'E{position}', 'oP', 20.0, True) for position in range(12)]
    candidates, entries = _tiny(rows)
    candidates['other'] = candidates['score'] - 15.0   # everything now fails the threshold
    first = FomMetrics.evaluate(candidates, entries=entries, score='score', threshold=10,
                                 n_bootstrap=0)
    second = FomMetrics.evaluate(candidates, entries=entries, score='other', threshold=10,
                                  n_bootstrap=0)
    outcome = FomMetrics.mcnemar(first, second)
    assert outcome['n_a_only'] == 12
    assert outcome['n_b_only'] == 0
    assert outcome['method'] == 'exact'
    assert outcome['p_value'] == pytest.approx(
        scipy_stats.binomtest(0, 12, 0.5).pvalue)
    assert outcome['delta'] == pytest.approx(1.0)


def test_mcnemar_refuses_different_entry_sets():
    rows = [('E1', 'oP', 20.0, True), ('E2', 'oP', 20.0, True)]
    both = _evaluate(rows, threshold=10)
    one = _evaluate(rows[:1], threshold=10)
    with pytest.raises(ValueError, match='different entry sets'):
        FomMetrics.mcnemar(both, one)


def test_a_per_lattice_threshold_moves_the_accept_rule_and_not_the_ranking():
    """S08's Q33 needs a per-lattice accept rule that leaves the cross-lattice order alone.

    Expressing it as a score transform -- subtract each lattice's cut -- also reorders the pooled
    ranking, and S08 measured that conflation costing 3.9 pp of top-10. A mapping applied at the
    comparison instead has to leave every rank metric byte-identical.
    """
    rows = ([(f'A{position}', 'cP', 30.0, True) for position in range(6)]
            + [(f'B{position}', 'aP', 12.0, True) for position in range(6)])
    truth = {f'A{position}': 'cP' for position in range(6)}
    truth.update({f'B{position}': 'aP' for position in range(6)})
    candidates, entries = _tiny(rows, entries=truth)
    common = dict(entries=entries, score='score',  n_bootstrap=0,
                  strata=('bravais_lattice',))

    scalar = FomMetrics.evaluate(candidates, threshold=20.0, **common)
    mapped = FomMetrics.evaluate(candidates, threshold={'cP': 20.0, 'aP': 5.0}, **common)

    # The rank half is untouched by any threshold, per-lattice or not.
    for metric in ('top1', 'top10', 'rank_only', 'found'):
        assert mapped.metric(metric) == scalar.metric(metric)
    # The accept half moves, and only for the lattice whose cut changed.
    per_lattice = mapped.stratum('bravais_lattice').set_index('level')
    assert per_lattice.loc['aP', 'threshold_only'] == 1.0
    assert FomMetrics.evaluate(candidates, threshold=20.0, **common) \
        .stratum('bravais_lattice').set_index('level').loc['aP', 'threshold_only'] == 0.0
    assert mapped.meta['threshold'] == {'cP': 20.0, 'aP': 5.0}

    # A lattice the mapping omits is refused, not quietly given a neighbour's cut.
    partial = FomMetrics.evaluate(candidates, threshold={'cP': 20.0}, **common)
    assert partial.stratum('bravais_lattice').set_index('level').loc['aP', 'threshold_only'] == 0.0


def test_mcnemar_accepts_a_boolean_mask_subset():
    """The documented mask path, which raised "truth value of an array is ambiguous" until S08.

    `subset == 'hard'` was evaluated before the type was known, so comparing an ndarray with a
    string produced an array and the `elif` on it raised. That made every per-stratum paired test
    unreachable -- including the per-Bravais-lattice one, which is the only way to check whether a
    learned score has made one lattice worse while improving the aggregate (S08, F-080).
    """
    rows = [(f'E{position}', 'oP', 20.0, True) for position in range(12)]
    candidates, entries = _tiny(rows)
    candidates['other'] = candidates['score'] - 15.0
    first = FomMetrics.evaluate(candidates, entries=entries, score='score', threshold=10,
                                 n_bootstrap=0)
    second = FomMetrics.evaluate(candidates, entries=entries, score='other', threshold=10,
                                  n_bootstrap=0)

    everything = np.ones(first.per_entry.shape[0], dtype=bool)
    assert (int(FomMetrics.mcnemar(first, second, subset=everything)['n_a_only'])
            == int(FomMetrics.mcnemar(first, second)['n_a_only']))

    half = np.zeros(first.per_entry.shape[0], dtype=bool)
    half[:6] = True
    assert int(FomMetrics.mcnemar(first, second, subset=half)['n_entries']) == 6

    with pytest.raises(ValueError, match='boolean mask'):
        FomMetrics.mcnemar(first, second, subset='soft')
    with pytest.raises(ValueError, match='shape'):
        FomMetrics.mcnemar(first, second, subset=np.ones(3, dtype=bool))


def test_threshold_selection_needs_something_to_trade_against():
    # The operating point is monotone non-increasing in the threshold, so maximising it alone
    # drives the threshold to minus infinity. Selecting on it must therefore be constrained.
    rows = [('E1', 'oP', 20.0, True), ('E2', 'oP', 8.0, False), ('E3', 'oP', 4.0, True)]
    result = _evaluate(rows, threshold=10)
    with pytest.raises(ValueError, match='minus infinity'):
        FomMetrics.select_threshold(result, objective='operating_point')
    constrained = FomMetrics.select_threshold(result, objective='operating_point',
                                             max_false_positive_rate=0.0)
    assert constrained.curve['false_positive_rate'].max() <= 0.0
    youden = FomMetrics.select_threshold(result, objective='youden')
    assert np.isfinite(youden.threshold)


def test_a_lower_is_better_threshold_has_to_be_turned_back_round():
    """The orientation trap S06 walks into once per lower-is-better merit.

    `evaluate` flips a lower-is-better score once, on the way in, so everything downstream can
    assume higher-is-better -- which means `per_entry` stores the *negated* score and the
    threshold `select_threshold` picks off it is negated too. Feeding that number straight back
    into `evaluate(threshold=..., higher_is_better=False)` negates it a second time and silently
    selects on the wrong side of the distribution. The caller has to pass `-choice.threshold`.

    Here the correct candidates score 2 and 3 and the wrong one 30, so any useful threshold keeps
    scores below about 10.
    """
    rows = [('E1', 'oP', 2.0, True), ('E2', 'oP', 30.0, False), ('E3', 'oP', 3.0, True)]
    result = _evaluate(rows, higher_is_better=False)
    choice = FomMetrics.select_threshold(result)
    # Internal orientation: the stored scores are -2, -30, -3, so the chosen threshold is negative.
    assert choice.threshold < 0

    correct = _evaluate(rows, higher_is_better=False, threshold=-choice.threshold)
    wrong = _evaluate(rows, higher_is_better=False, threshold=choice.threshold)

    # Turned back round, both correct candidates clear the threshold and the wrong one does not.
    assert correct.metric('operating_point') == pytest.approx(2/3)
    assert correct.metric('false_positive') == pytest.approx(0.0)

    # Left negated, the threshold lands past the far tail and the program abstains on everything.
    # Note the failure mode: not a flood of wrong answers but a silent refusal to answer at all,
    # which is why this is worth a test -- a merit broken this way looks merely unimpressive.
    assert wrong.metric('operating_point') == pytest.approx(0.0)
    assert wrong.metric('reported') == pytest.approx(0.0)
    assert correct.metric('reported') > wrong.metric('reported')


def test_threshold_transfer_refuses_the_selection_entries():
    rows = [('E1', 'oP', 20.0, True), ('E2', 'oP', 4.0, True)]
    result = _evaluate(rows, threshold=10)
    choice = FomMetrics.select_threshold(result)
    with pytest.raises(ValueError, match='same entries'):
        FomMetrics.check_threshold_transfer(choice, result)
    FomMetrics.check_threshold_transfer(choice, result, allow_same_entries=True)


# ---------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------
def test_reliability_uses_equal_count_bins():
    probability = np.linspace(0.0, 1.0, 100)
    labels = probability > 0.5
    table, ece, brier = FomMetrics.reliability(probability, labels, n_bins=10)
    assert table.shape[0] == 10
    assert set(table['n']) == {10}
    assert np.isfinite(ece) and np.isfinite(brier)


def test_reliability_collapses_a_constant_predictor_to_one_bin():
    probability = np.full(100, 0.5)
    labels = np.arange(100) < 25
    table, ece, brier = FomMetrics.reliability(probability, labels)
    assert table.shape[0] == 1
    assert ece == pytest.approx(0.25)
    assert brier == pytest.approx(0.25)


def test_reliability_is_near_zero_for_a_calibrated_score():
    rng = np.random.default_rng(11)
    probability = rng.random(20000)
    labels = rng.random(20000) < probability
    _, ece, _ = FomMetrics.reliability(probability, labels, n_bins=20)
    assert ece < 0.02


def test_average_precision_matches_sklearn():
    metrics = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(5)
    score = np.round(rng.random(500), 2)     # rounding forces exact ties
    labels = rng.random(500) < 0.3
    assert FomMetrics.average_precision(score, labels) == pytest.approx(
        metrics.average_precision_score(labels, score), rel=1e-9)


def test_calibration_refuses_a_score_that_is_not_a_probability():
    # An ECE for raw M20 would be a number with no meaning, and on C0 it would be a number
    # dominated by the 9.5% of candidates scoring above 1e9 (F-054).
    rows = [('E1', 'oP', 12.5, True), ('E2', 'oP', 3.0, False)]
    result = _evaluate(rows, threshold=10, calibration=True)
    assert result.calibration.shape[0] == 0
    assert 'not a probability' in result.meta['calibration_skipped_reason']


def test_calibration_runs_for_a_probability_score():
    rng = np.random.default_rng(2)
    probability = rng.random(400)
    rows = [(f'E{position//4}', 'oP', float(probability[position]),
             bool(rng.random() < probability[position])) for position in range(400)]
    result = _evaluate(rows, threshold=0.5, calibration=True)
    assert result.meta['calibration_skipped_reason'] is None
    assert result.calibration['n'].sum() == 400
    assert result.calibration['ece'].notna().all()


# ---------------------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------------------
def test_bravais_lattices_match_run_py():
    from mlindex.command_line.run import BRAVAIS_LATTICES as production
    assert set(FomMetrics.BRAVAIS_LATTICES) == set(production)
    assert len(FomMetrics.BRAVAIS_LATTICES) == 14


def test_score_callable_is_called_once_per_shard():
    rows = [('E1', 'oP', 20.0, True), ('E2', 'oP', 5.0, True)]
    candidates, entries = _tiny(rows)
    calls = []

    def negated_score(frame):
        calls.append(frame.shape[0])
        return -frame['score'].to_numpy()

    result = FomMetrics.evaluate([candidates], entries=entries, score=negated_score,
                                 higher_is_better=False, threshold=-10, 
                                 n_bootstrap=0)
    assert calls == [2]
    assert result.metric('operating_point') == 0.5
    assert result.meta['score'] == 'negated_score'


def test_the_control_bundle_is_excluded_by_default():
    rows_c0 = [('E1', 'oP', 1e15, True)]
    candidates, entries = _tiny(rows_c0, bundle='error0_cont0')
    with pytest.raises(ValueError, match='No candidates to evaluate'):
        FomMetrics.evaluate(candidates, entries=entries, score='score', threshold=10,
                             n_bootstrap=0)
    kept = FomMetrics.evaluate(candidates, entries=entries, score='score', threshold=10,
                                n_bootstrap=0, include_control=True)
    assert kept.metric('operating_point') == 1.0
    assert kept.meta['n_score_above_1e9'] == 1


def test_reduction_refuses_shards_reduced_separately():
    rows = [('E1', 'oP', 20.0, True)]
    candidates, entries = _tiny(rows)
    with pytest.raises(ValueError, match='reduced twice'):
        FomMetrics.evaluate([candidates, candidates], entries=entries, score='score',
                            threshold=10,  n_bootstrap=0)


def test_write_result_leaves_an_artefact(tmp_path):
    rows = [('E1', 'oP', 20.0, True)]
    result = _evaluate(rows, threshold=10)
    written = FomMetrics.write_result(result, tmp_path, 'unit')
    assert (tmp_path / 'unit_aggregate.csv').exists()
    assert (tmp_path / 'unit_meta.json').exists()
    assert 'aggregate' in written and 'meta' in written


# ---------------------------------------------------------------------------------------
# Loader repair: the consolidated pool keys on (entry_id, condition_bundle)
# ---------------------------------------------------------------------------------------
def test_bundle_is_parsed_from_either_filename_layout():
    parse = FomBenchmark.bundle_from_candidate_path
    assert parse('candidates_error1_cont1_drop6_mP.parquet') == 'error1_cont1_drop6'
    assert parse('candidates_error1_cont0_phase3_aP.parquet') == 'error1_cont0_phase3'
    assert parse('candidates_error1_cont0_shard00of01_pool00.parquet') == 'error1_cont0'
    with pytest.raises(ValueError, match='Not a candidate shard'):
        parse('entries.parquet')


def test_loaders_join_a_two_bundle_consolidated_pool(tmp_path):
    """R8: after consolidation `entry_id` alone is not a key.

    Two bundles hold the same crystal, so a join on `entry_id` fans every candidate out once per
    bundle. `(entry_id, q2_digest)` is not a substitute either -- two sparse bundles leave 157
    entries with identical peak lists -- which is why `condition_bundle` is a stored column on
    both streams in schema v3.
    """
    pytest.importorskip('pyarrow')
    frames = []
    entry_rows = []
    for bundle in ('c2_error1_cont0', 'c2_error2_cont0'):
        candidates, entries = _tiny([('E1', 'oP', 20.0, True), ('E1', 'mP', 5.0, False)],
                                    bundle=bundle)
        # The stored schema carries no condition_bundle; the filename does.
        candidates = candidates.drop(columns=['condition_bundle'])
        # A consolidated pool carries its labels, so loading must not try to recompute them --
        # which it could not do here anyway, the tiny entry table having no ground-truth cell.
        # Schema v3's label set includes `hkl_true_in_basis`, null except on correct rows.
        candidates['xnn_distance_to_truth'] = 0.0
        candidates['volume_ratio_to_truth'] = 1.0
        candidates['hkl_true_in_basis'] = None
        candidates['lattice_system'] = 'orthorhombic'
        candidates['q2_digest'] = f'digest-{bundle}'
        entries['q2_digest'] = f'digest-{bundle}'
        for lattice, group in candidates.groupby('bravais_lattice'):
            FomBenchmark.write_candidate_shard(group.reset_index(drop=True), tmp_path,
                                               f'{bundle}_{lattice}')
        frames.append(candidates)
        entry_rows.append(entries)
    FomBenchmark._to_parquet(pd.concat(entry_rows, ignore_index=True),
                             tmp_path / 'entries.parquet')

    assert FomBenchmark.available_bundles(tmp_path) == ['c2_error1_cont0', 'c2_error2_cont0']
    loaded = FomBenchmark.load_candidates(tmp_path)
    assert set(loaded['condition_bundle']) == {'c2_error1_cont0', 'c2_error2_cont0'}
    # The join is the point: keyed on entry_id alone this raises, because E1 appears twice.
    joined = FomBenchmark.load_benchmark(tmp_path)
    assert joined.shape[0] == loaded.shape[0]
    assert set(joined['bravais_lattice_true']) == {'oP'}


def test_check_join_rejects_a_mismatched_digest(tmp_path):
    candidates, entries = _tiny([('E1', 'oP', 20.0, True)])
    candidates['q2_digest'] = 'right'
    entries['q2_digest'] = 'wrong'
    with pytest.raises(ValueError, match='disagrees with'):
        FomBenchmark._check_join(candidates, entries)


# ---------------------------------------------------------------------------------------
# Negative subsampling: what the pool can and cannot answer exactly
# ---------------------------------------------------------------------------------------
# Benchmark B keeps every correct candidate, everything in the top K by each of seven merits, and
# a 5 % sample of the rest. `SCHEMA.md` and `METRICS.md` both stated the resulting exactness
# without saying it holds only for those seven; for any other score the field a correct candidate
# is ranked against was thinned, so its rank comes out flattering. See C2-F-077 and C2-R-013.
def test_a_ranked_merit_within_the_depth_is_exact():
    for merit in FomMetrics.RANK_EXACT_MERITS:
        exact, _ = FomMetrics.rank_exactness(merit, top_n=10, top_k=200, subsampled=True)
        assert exact, merit


def test_a_score_the_subsampler_did_not_rank_on_is_refused():
    exact, reason = FomMetrics.rank_exactness('combiner_score', top_n=10, top_k=200,
                                              subsampled=True)
    assert not exact
    assert 'not one of the merits the subsampler ranked on' in reason


def test_a_callable_score_is_refused_on_a_subsampled_pool():
    """A callable cannot be one of the seven, so it is refused by construction rather than by
    name-matching something that happens to look like a merit column."""
    exact, reason = FomMetrics.rank_exactness(lambda frame: frame['score'], top_n=10, top_k=200,
                                              subsampled=True)
    assert not exact and 'subsampler ranked on' in reason


def test_a_depth_beyond_k_is_refused():
    exact, reason = FomMetrics.rank_exactness('M20', top_n=500, top_k=200, subsampled=True)
    assert not exact
    assert 'deeper than the pool retention depth' in reason


def test_an_unsubsampled_pool_answers_anything():
    exact, reason = FomMetrics.rank_exactness('anything', top_n=10_000, top_k=None,
                                              subsampled=False)
    assert exact and reason is None


def test_an_unreadable_manifest_is_refused_rather_than_assumed_full():
    """The dangerous default. A pool whose manifest cannot be read is not evidence of a full
    pool, and treating it as one silently reinstates exactly the bias this guard exists for."""
    exact, reason = FomMetrics.rank_exactness('M20', top_n=10, top_k=None, subsampled=None)
    assert not exact
    assert 'no readable manifest' in reason


def test_mrr_is_flagged_even_when_the_depth_is_exact():
    """`mrr` uses the full rank, so `top_n` does not bound it and K does not make it exact."""
    _, reason = FomMetrics.rank_exactness('M20', top_n=10, top_k=200, subsampled=True)
    assert 'mrr' in reason


def test_evaluate_refuses_an_inexact_rank_metric(tmp_path):
    candidates, entries = _tiny([('E1', 'oP', 20.0, True)])
    with pytest.raises(ValueError, match='Refusing to report a rank metric'):
        FomMetrics.evaluate(candidates, entries=entries, score='score', threshold=10,
                            n_bootstrap=0, subsample_top_k=200)


def test_evaluate_proceeds_when_told_to_and_records_why():
    candidates, entries = _tiny([('E1', 'oP', 20.0, True)])
    result = FomMetrics.evaluate(candidates, entries=entries, score='score', threshold=10,
                                 n_bootstrap=0, subsample_top_k=200,
                                 allow_inexact_ranks=True)
    assert result.meta['ranks_exact'] is False
    assert result.meta['subsample_top_k'] == 200
    assert 'subsampler ranked on' in result.meta['rank_exactness']


def test_a_frame_is_taken_at_face_value():
    """'auto' means "not subsampled" for a frame: there is no manifest, and a caller that built
    its own frame knows what went into it. Only a pool root is interrogated."""
    candidates, entries = _tiny([('E1', 'oP', 20.0, True)])
    result = FomMetrics.evaluate(candidates, entries=entries, score='score', threshold=10,
                                 n_bootstrap=0)
    assert result.meta['ranks_exact'] is True
    assert result.meta['subsampled'] is False


def test_subsample_depth_reads_the_smallest_k_across_bundles(tmp_path):
    """A consolidated pool nests one manifest per bundle, and the pool is only exact to the
    smallest depth any of them was written at."""
    payload = dict(schema_version='3', subsampled=True, top_k=200,
                   bundle_manifests={'a': dict(subsampled=True, top_k=200),
                                     'b': dict(subsampled=True, top_k=50)})
    with open(tmp_path / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    assert FomBenchmark.subsample_depth(tmp_path) == (50, True)


def test_subsample_depth_reports_an_absent_manifest_as_unknown(tmp_path):
    assert FomBenchmark.subsample_depth(tmp_path) == (None, None)


# ---------------------------------------------------------------------------------------
# The acceptance gate, against Benchmark B's slice
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def slice_root():
    pytest.importorskip('pyarrow')
    if not (SLICE_ROOT / 'entries.parquet').exists():
        pytest.skip('Benchmark B\'s slice is absent (untracked). Write it on NERSC with '
                    'run_fom_pool_subset.py and copy it to mlindex/data/fom_benchmark_c2.')
    return SLICE_ROOT


@pytest.mark.slow
def test_gate_the_module_against_the_real_slice(slice_root):
    """S08 acceptance condition 1, on as much of the pool as exists off the cluster.

    Not the full gate: the slice is a stratified subset, so its operating point is a property of
    the subset and cannot be compared against a published pool-wide number. What it does check is
    that the module runs end to end on real schema-v3 data, that the pool's own retention depth is
    read and honoured, and that the two ceilings stand in the relation METRICS.md section 3
    requires. The pool-wide comparison belongs to the NERSC session.
    """
    result = FomMetrics.evaluate(slice_root, score='M20', threshold=10, n_bootstrap=0)
    assert result.meta['ranks_exact'] is True
    assert result.meta['subsample_top_k'] is not None
    assert result.metric('ceiling_reranker') == pytest.approx(result.metric('threshold_only'))
    assert result.metric('ceiling_rescorer') >= result.metric('ceiling_reranker')
    assert 0.0 <= result.metric('operating_point') <= result.metric('ceiling_rescorer')
    # Every reported bundle is a campaign-2 one, so no campaign-1 control could have crept in.
    assert all(bundle.startswith('c2_') for bundle in result.meta['bundles'])
