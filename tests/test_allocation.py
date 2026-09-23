import numpy as np
import pytest

from mlindex.utilities.Allocation import largest_remainder


def test_counts_sum_to_the_total_whatever_the_weights():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        weight = rng.random(n)
        total = int(rng.integers(0, 5000))
        counts = largest_remainder(weight, total)
        assert counts.sum() == total
        assert np.all(counts >= 0)


def test_the_leftovers_go_to_the_largest_fractions():
    # 10 units on shares 0.25/0.25/0.5 is 2.5/2.5/5 -> floors 2/2/5, one unit left
    counts = largest_remainder(np.array([0.25, 0.25, 0.5]), 10)
    assert counts.sum() == 10
    assert counts[2] == 5
    assert sorted(counts[:2].tolist()) == [2, 3]


def test_a_small_enough_weight_gets_nothing():
    counts = largest_remainder(np.array([1.0, 1e-9]), 4)
    assert counts.tolist() == [4, 0]


def test_equal_weights_divide_as_evenly_as_possible():
    counts = largest_remainder(np.ones(4), 10)
    assert counts.sum() == 10
    assert set(counts.tolist()) == {2, 3}


def test_weights_that_cannot_be_divided_are_refused():
    with pytest.raises(ValueError):
        largest_remainder(np.zeros(3), 5)
    with pytest.raises(ValueError):
        largest_remainder(np.array([1.0, -1.0]), 5)


def test_ties_go_to_the_earliest_claimant():
    """Equal weights are the common case: a generator's budget split among its split groups.

    Which group gets the odd candidate rests entirely on the tie rule, so it has to be defined.
    This is also what makes the shared allocator a drop-in for the even split GeneratorPools
    used to do inline.
    """
    for n_claimants in range(1, 9):
        for total in range(0, 40):
            counts = largest_remainder(np.ones(n_claimants), total)
            base, remainder = divmod(total, n_claimants)
            expected = [base + (1 if index < remainder else 0)
                        for index in range(n_claimants)]
            assert counts.tolist() == expected, (n_claimants, total)
