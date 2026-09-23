"""What a candidate is worth when it is not alone.

The weight is the only term in the ensemble score that can see how crowded a pool is. Everything
else reads distances from the true cell, and redistribution barely changes those -- which is why a
score without this term reports the same value at every redistribution setting while a third of the
pool is being moved.
"""
import numpy as np
import pytest

from mlindex.utilities.ClumpDiscount import clump_weights, discount_path, load_clump_discount
from mlindex.utilities.EnsembleObjective import expected_success_objective

K = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
ALPHA = np.array([1.0, 0.6, 0.35, 0.2, 0.12])


def test_a_candidate_on_its_own_is_worth_a_whole_candidate():
    spread = np.arange(8.0)[:, np.newaxis]*np.array([1.0, 0.0])   # 1.0 apart, delta far smaller
    np.testing.assert_allclose(clump_weights(spread, 0.1, K, ALPHA), 1.0)


def test_a_clump_is_discounted_by_its_size():
    """Eight candidates on top of each other are worth alpha(8) each, not one each."""
    piled = np.zeros((8, 2))
    np.testing.assert_allclose(clump_weights(piled, 0.1, K, ALPHA), ALPHA[3])


def test_spreading_a_clump_out_raises_what_it_is_worth():
    """This is the whole point: it is the only way the score can notice redistribution."""
    piled = np.zeros((8, 2))
    spread = np.arange(8.0)[:, np.newaxis]*np.array([1.0, 0.0])
    assert clump_weights(spread, 0.1, K, ALPHA).sum() > clump_weights(piled, 0.1, K, ALPHA).sum()


def test_a_zero_radius_means_nothing_clumps():
    piled = np.zeros((8, 2))
    np.testing.assert_allclose(clump_weights(piled, 0.0, K, ALPHA), 1.0)


def test_the_score_falls_when_the_same_candidates_are_piled_up():
    """Same distances from the true cell, same count -- only the crowding differs."""
    radii = np.array([1e-4, 1e-3, 1e-2])
    success = np.array([0.8, 0.4, 0.05])
    distance = np.full(8, 1e-4)
    piled = np.zeros((8, 2))
    spread = np.arange(8.0)[:, np.newaxis]*np.array([1.0, 0.0])
    crowded = expected_success_objective(distance, radii, success,
                                         clump_weights(piled, 0.1, K, ALPHA))
    apart = expected_success_objective(distance, radii, success,
                                       clump_weights(spread, 0.1, K, ALPHA))
    assert apart > crowded
    assert apart == pytest.approx(8*0.8)          # independent: eight whole candidates
    assert crowded == pytest.approx(8*0.8*ALPHA[3])


def test_a_missing_discount_is_refused_with_the_path_it_wanted(tmp_path):
    with pytest.raises(FileNotFoundError, match='clump discount'):
        load_clump_discount(str(tmp_path), 'oP')


def test_an_impossible_discount_is_refused(tmp_path):
    """alpha above 1 would mean a clump member worth more than an independent candidate, which is
    the signature of the solve failing rather than a measurement."""
    np.savez(discount_path(str(tmp_path), 'oP'), delta=1e-4, k=K, alpha=ALPHA*10)
    with pytest.raises(ValueError, match='cannot be worth more'):
        load_clump_discount(str(tmp_path), 'oP')


def test_a_written_discount_round_trips(tmp_path):
    np.savez(discount_path(str(tmp_path), 'oP'), delta=3.4e-4, k=K, alpha=ALPHA)
    delta, k, alpha = load_clump_discount(str(tmp_path), 'oP')
    assert delta == pytest.approx(3.4e-4)
    np.testing.assert_array_equal(k, K)
    np.testing.assert_array_equal(alpha, ALPHA)


def test_cubic_falls_back_to_no_discount_because_it_cannot_be_measured(tmp_path):
    """A cubic cell has one free parameter, so the separation job cannot build a clump in it.

    SeparatedStartManager puts group members on a sphere of radius delta/2 about the centre, and
    in one dimension that sphere is two points -- half of every group coincident at any
    separation. So there is no cubic file to load and the fit would otherwise refuse the three
    lattices outright.
    """
    from mlindex.utilities.ClumpDiscount import CUBIC

    for bravais_lattice in CUBIC:
        delta, k, alpha = load_clump_discount(str(tmp_path), bravais_lattice)
        assert delta == 0.0
        assert np.all(alpha == 1.0)
        # and it really does switch the term off, even on a maximally clumped pool
        coincident = np.zeros((64, 1))
        assert np.all(clump_weights(coincident, delta, k, alpha) == 1.0)


def test_a_measured_cubic_discount_takes_precedence_over_the_fallback(tmp_path):
    """The fallback must never shadow a real measurement, or fixing the group construction
    later would silently have no effect."""
    np.savez(discount_path(str(tmp_path), 'cP'), delta=1e-3,
             k=np.array([1.0, 8.0]), alpha=np.array([1.0, 0.25]))
    delta, k, alpha = load_clump_discount(str(tmp_path), 'cP')
    assert delta == 1e-3
    assert alpha.tolist() == [1.0, 0.25]
    coincident = np.zeros((8, 1))
    assert clump_weights(coincident, delta, k, alpha)[0] < 1.0, 'the measured table must bite'


def test_a_non_cubic_lattice_still_refuses_a_missing_discount(tmp_path):
    """The fallback is for the three lattices that cannot be measured, not a general default."""
    for bravais_lattice in ('oP', 'mP', 'aP', 'hP'):
        with pytest.raises(FileNotFoundError, match='no clump discount'):
            load_clump_discount(str(tmp_path), bravais_lattice)
