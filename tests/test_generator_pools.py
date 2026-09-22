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
