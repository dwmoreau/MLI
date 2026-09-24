"""The candidate-pool generator, and the one numpy behaviour it leans on.

`generate_candidate_pools` permutes each generator's column by drawing an index and applying it,
rather than by permuting the column's values, because the positions and the distances have to move
together. That is only safe if drawing the index takes the same numbers off the generator as
permuting the values would -- otherwise a pool regenerated at the same seed stops matching the one
on disk. This file pins that.
"""
import numpy as np
import pytest

from mlindex.optimization.GeneratorPools import (  # noqa: F401
    generate_candidate_pools,
    refuse_unfilled,
    )


def test_permuting_by_a_drawn_index_matches_permuting_the_values():
    for n in (1, 2, 10, 4000):
        values = np.arange(float(n))
        by_value = np.random.default_rng(12345)
        by_index = np.random.default_rng(12345)
        np.testing.assert_array_equal(
            by_value.permutation(values),
            values[by_index.permutation(n)],
            )
        assert by_value.bit_generator.state == by_index.bit_generator.state


def test_permuting_a_gathered_block_matches_permuting_the_gather_index():
    """The two-tier network permutation gathers first and permutes after; this is the swap."""
    rng_values = np.random.default_rng(7)
    rng_index = np.random.default_rng(7)
    column = np.linspace(0.0, 1.0, 50)
    gather = np.array([3, 17, 42, 8, 29])
    np.testing.assert_array_equal(
        rng_values.permutation(column[gather]),
        column[rng_index.permutation(gather)],
        )


def test_an_unfilled_candidate_slot_is_refused_rather_than_returned():
    """A share that does not divide among a generator's split groups leaves NaN behind, and a
    NaN distance is silently read as "beyond the curve" by everything downstream."""
    names = ['trees', 'abnn', 'templates']
    full = np.zeros((4, 3, 2))
    refuse_unfilled(full, names, 4)          # a full pool passes

    holed = np.zeros((4, 3, 2))
    holed[3, 1] = np.nan
    with pytest.raises(ValueError, match="'abnn': 1"):
        refuse_unfilled(holed, names, 4)


def test_a_prefix_of_the_network_s_column_is_its_best_ranked_predictions():
    """The predicted cells arrive sorted by the network's own confidence, and must stay so.

    predict_xnn sorts its predictions by softmax and generate writes them in that order, so
    rows 0..top_n-1 of each split group are rank-ordered. The column used to be permuted within
    that tier, which sampled the predictions at random -- but production does not sample: when a
    generator's share is below the model's prediction count, ABNN.generate returns the MOST
    PROBABLE n. A permuted tier makes the fit score something production would never build.

    Ranks are interleaved across split groups rather than concatenated, because each model's
    softmax is normalised within itself and ranks are not comparable between groups.
    """
    from mlindex.optimization.GeneratorPools import abnn_order

    counts = [7, 7, 7]          # three split groups
    top_n = 4                   # four predicted cells each, three siblings each
    order = abnn_order(counts, top_n, np.random.default_rng(0))

    assert order.size == sum(counts)
    assert sorted(order.tolist()) == list(range(sum(counts))), 'every candidate exactly once'

    # the predicted cells come first, one rank at a time across the groups
    starts = [0, 7, 14]
    expected = [start + rank for rank in range(top_n) for start in starts]
    assert order[:top_n*len(counts)].tolist() == expected

    # so any prefix uses only the best ranks, from every group
    for prefix in range(1, top_n*len(counts) + 1):
        ranks = [int(index) % 7 for index in order[:prefix]]
        assert max(ranks) <= (prefix - 1)//len(counts), (
            f'a prefix of {prefix} reached rank {max(ranks)}')

    # the siblings follow, and are permuted rather than left in generation order
    siblings = order[top_n*len(counts):]
    assert sorted(siblings.tolist()) == [s + r for s in starts for r in range(top_n, 7)]


def test_the_network_s_ranking_survives_an_uneven_split_group_division():
    """hexagonal has eight split groups and a budget that does not divide evenly among them."""
    from mlindex.optimization.GeneratorPools import abnn_order

    counts = [250, 250, 250, 250, 249, 249, 249, 249]
    order = abnn_order(counts, 100, np.random.default_rng(1))
    assert order.size == sum(counts)
    assert sorted(order.tolist()) == list(range(sum(counts)))
    starts = np.cumsum([0] + counts[:-1])
    assert order[:len(counts)].tolist() == starts.tolist(), 'rank 1 of every group comes first'
