"""Reducing a candidate pool to per-pattern outcomes.

These pin the decisions that make two numbers comparable -- the pooling, the tie-break, what counts
as found -- on small frames where the right answer can be read off by hand. The gate against real
campaign numbers is in the session artefact, not here; this is what fails first when one of those
decisions is changed by accident.
"""

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import BenchmarkMetrics as metrics


def _pool(rows):
    """A candidate frame from (bravais_lattice, candidate_id, score, is_correct) tuples."""
    return pd.DataFrame({
        'entry_id': ['E1']*len(rows),
        'condition_bundle': ['b1_error1_cont0']*len(rows),
        'bravais_lattice': [row[0] for row in rows],
        'candidate_id': [row[1] for row in rows],
        'score': [row[2] for row in rows],
        'is_correct': [row[3] for row in rows],
        'in_top_n': [row[4] if len(row) > 4 else True for row in rows],
        'is_degenerate': [False]*len(rows),
    })


def _rank(frame, depth='all'):
    reduced = metrics.reduce_pool(frame, frame['score'].to_numpy())
    return int(reduced[f'rank_best_correct_{depth}'].iloc[0])


def test_the_ranking_pools_across_lattices():
    """The indexer sorts one pool across all fourteen lattices, so the reduction must too.
    Ranking within a lattice and taking the best is a different, much easier question."""
    frame = _pool([('mP', 0, 10.0, True), ('cP', 0, 12.0, False), ('oP', 0, 11.0, False)])
    assert _rank(frame) == 2


def test_a_tie_breaks_towards_the_earlier_lattice_not_towards_the_correct_answer():
    """Putting is_correct in the sort key would make every ranking optimistic. The tie-break is
    lattice order then candidate id, which is total and label-independent."""
    frame = _pool([('mP', 0, 10.0, True), ('cF', 0, 10.0, False)])
    assert _rank(frame) == 1, 'cF precedes mP in the canonical order, so it wins the tie'

    frame = _pool([('cF', 0, 10.0, True), ('mP', 0, 10.0, False)])
    assert _rank(frame) == 0


def test_a_tie_within_one_lattice_breaks_on_candidate_id():
    frame = _pool([('mP', 7, 10.0, True), ('mP', 3, 10.0, False)])
    assert _rank(frame) == 1


def test_the_ranking_does_not_depend_on_the_order_shards_were_read_in():
    """Otherwise the numbers depend on a glob, which differs between machines."""
    rows = [('mP', 0, 10.0, True), ('cP', 1, 12.0, False), ('oP', 2, 11.0, False)]
    first = _rank(_pool(rows))
    second = _rank(_pool(list(reversed(rows))))
    assert first == second


def test_nan_scores_rank_last_and_infinite_ones_first():
    """NaN carries no information; +inf is a vanishing residual and is a real best."""
    frame = _pool([('mP', 0, np.nan, False), ('mP', 1, 5.0, True)])
    assert _rank(frame) == 0
    frame = _pool([('mP', 0, np.inf, False), ('mP', 1, 5.0, True)])
    assert _rank(frame) == 1


def test_a_pattern_with_no_correct_candidate_is_not_found_rather_than_ranked_last():
    """'No correct candidate in the pool' is a generation failure and has to stay separable from
    a ranking failure."""
    frame = _pool([('mP', 0, 10.0, False), ('cP', 1, 9.0, False)])
    reduced = metrics.reduce_pool(frame, frame['score'].to_numpy())
    flagged = metrics.derive_flags(reduced)
    assert not bool(flagged['found'].iloc[0])
    assert not bool(flagged['top10'].iloc[0])
    assert float(flagged['reciprocal_rank'].iloc[0]) == 0.0
    assert int(reduced['rank_best_correct_all'].iloc[0]) == -1


def test_a_degenerate_crystal_s_correct_cells_do_not_count_as_found():
    """Their correctness is an accident of the lattice rather than the search finding it. They are
    counted separately so the exclusion is visible."""
    frame = _pool([('mP', 0, 10.0, True)])
    frame['is_degenerate'] = True
    reduced = metrics.reduce_pool(frame, frame['score'].to_numpy())
    assert not bool(reduced['has_correct_all'].iloc[0])
    assert int(reduced['n_correct_all'].iloc[0]) == 0
    assert int(reduced['n_correct_incl_degenerate_all'].iloc[0]) == 1


def test_the_two_depths_come_from_one_pass_and_in_top_n_is_a_restriction():
    """`in_top_n` is what production hands to its final sort. Restricting can only improve a rank,
    because it removes competitors and never adds them."""
    frame = _pool([('cP', 0, 12.0, False, False), ('oP', 1, 11.0, False, True),
                   ('mP', 2, 10.0, True, True)])
    assert _rank(frame, 'all') == 2
    assert _rank(frame, 'in_top_n') == 1


def test_the_cross_lattice_loss_is_attributed():
    """The dominant ranking failure is a candidate from a different Bravais lattice, so a change
    that lifts correct cells within their own lattice buys nothing if it lifts the others too."""
    frame = _pool([('cP', 0, 12.0, False), ('cI', 1, 11.5, False), ('mP', 2, 10.0, True),
                   ('mP', 3, 11.0, False)])
    reduced = metrics.reduce_pool(frame, frame['score'].to_numpy())
    assert int(reduced['n_other_lattice_above_best_correct_all'].iloc[0]) == 2
    assert int(reduced['n_same_lattice_above_best_correct_all'].iloc[0]) == 1


def test_ties_at_the_winning_score_are_counted():
    frame = _pool([('mP', 0, 10.0, True), ('mP', 1, 10.0, False), ('oP', 2, 10.0, False)])
    reduced = metrics.reduce_pool(frame, frame['score'].to_numpy())
    assert int(reduced['n_ties_at_best_correct_all'].iloc[0]) == 3


def test_each_score_is_reduced_under_its_own_name():
    """One pass, several scores, and no score inherits another's ranking -- otherwise a learned
    score could take a certificate from the column it reads."""
    frame = _pool([('mP', 0, 10.0, True), ('cP', 1, 12.0, False)])
    frame['other'] = [12.0, 10.0]
    reductions = metrics.reduce_many(frame, {'score': 'score', 'other': 'other'})
    assert int(reductions['score']['rank_best_correct_all'].iloc[0]) == 1
    assert int(reductions['other']['rank_best_correct_all'].iloc[0]) == 0


def test_a_score_of_the_wrong_length_is_refused():
    frame = _pool([('mP', 0, 10.0, True)])
    with pytest.raises(ValueError, match='values for'):
        metrics.reduce_many(frame, {'bad': np.array([1.0, 2.0])})


def test_an_unknown_bravais_lattice_is_refused_rather_than_sorted_last():
    frame = _pool([('mP', 0, 10.0, True)])
    frame.loc[0, 'bravais_lattice'] = 'zz'
    with pytest.raises(ValueError, match='Unknown Bravais lattice'):
        metrics.reduce_pool(frame, frame['score'].to_numpy())


def test_top_n_is_the_only_thing_derive_flags_depends_on():
    """So a sweep over reporting depth costs no pool pass."""
    frame = _pool([(bl, i, 10.0 - i, i == 5)
                   for i, bl in enumerate(['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP'])])
    reduced = metrics.reduce_pool(frame, frame['score'].to_numpy())
    assert bool(metrics.derive_flags(reduced, top_n=10)['top10'].iloc[0])
    assert not bool(metrics.derive_flags(reduced, top_n=5)['top10'].iloc[0])


def test_an_unknown_depth_is_refused():
    frame = _pool([('mP', 0, 10.0, True)])
    reduced = metrics.reduce_pool(frame, frame['score'].to_numpy())
    with pytest.raises(ValueError, match="depth must be"):
        metrics.derive_flags(reduced, depth='nonsense')


# ----------------------------------------------------------------------------------------------
# Comparing two arms
# ----------------------------------------------------------------------------------------------

def test_mcnemar_counts_only_the_disagreements():
    """Two rankings agreeing on nine tenths of a benchmark differ only where they disagree."""
    a = np.array([True]*90 + [True, False, False, False, False, True, True, True, True, False])
    b = np.array([True]*90 + [False, True, True, True, True, True, True, True, True, False])
    result = metrics.mcnemar(a, b)
    assert result['n_pairs'] == 100
    assert result['a_only'] == 1
    assert result['b_only'] == 4
    assert result['n_discordant'] == 5
    assert result['delta'] == pytest.approx(0.03)


def test_mcnemar_on_identical_arms_is_not_significant():
    a = np.array([True, False, True, False])
    result = metrics.mcnemar(a, a.copy())
    assert result['n_discordant'] == 0
    assert result['p_value'] == 1.0


def test_mismatched_arms_are_refused():
    with pytest.raises(ValueError, match='same patterns'):
        metrics.mcnemar(np.array([True, False]), np.array([True]))


def test_the_bootstrap_resamples_crystals_not_pattern_conditions():
    """One crystal under nine conditions is one draw. Treating it as nine understates every
    interval, because the nine outcomes are far from independent."""
    rng = np.random.default_rng(0)
    n_crystals, n_conditions = 40, 9
    clusters = np.repeat(np.arange(n_crystals), n_conditions)
    a = rng.random(clusters.size) < 0.5
    b = a.copy()

    low, high = metrics.paired_delta_ci(a, b, clusters, n_bootstrap=200, seed=1)
    assert low == 0.0 and high == 0.0, 'identical arms have no spread whatever the clustering'

    # Every condition of a crystal flips together: the effective sample size is the crystal count,
    # so the interval must be wide despite there being 360 rows.
    flip = np.repeat(rng.random(n_crystals) < 0.5, n_conditions)
    b = np.where(flip, ~a, a)
    low, high = metrics.paired_delta_ci(a, b, clusters, n_bootstrap=400, seed=1)
    assert high - low > 0.05
