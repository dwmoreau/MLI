"""The mechanisms that turn a simulated pattern into a realistic one.

The two functions that already shipped, `add_q2_error` and `add_contaminants`, gained keyword
arguments here. Their default path is pinned as unchanged because model training calls them.
"""

import numpy as np
import pytest

from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.ErrorAdder import MAX_INTERIOR_DROPOUT
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import add_second_phase
from mlindex.utilities.ErrorAdder import q2_sigma_params
from mlindex.utilities.ErrorAdder import select_peaks_with_nested_dropout


def _peaks(n=40, seed=0, low=0.02, high=0.8):
    return np.sort(np.random.default_rng(seed).uniform(low, high, size=n))


# ----------------------------------------------------------------------------------------------
# sigma(q2): severity against shape
# ----------------------------------------------------------------------------------------------

def test_omitting_both_parameters_gives_the_repository_model():
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info

    defaults = get_peak_generation_info()['q2_error_params']
    assert q2_sigma_params() == (float(defaults[0]), float(defaults[1]))


def test_either_parameter_can_be_overridden_alone():
    intercept, slope = q2_sigma_params(intercept=0.5)
    assert intercept == 0.5
    assert slope == q2_sigma_params()[1]
    intercept, slope = q2_sigma_params(slope=0.25)
    assert slope == 0.25
    assert intercept == q2_sigma_params()[0]


def test_the_intercept_reaches_what_a_multiplier_cannot():
    """A multiplier scales the whole of sigma. Raising the intercept alone changes sigma at low q2
    relative to high q2, which is the point of exposing it."""
    nominal_intercept, slope = q2_sigma_params()
    q2_low, q2_high = 0.02, 0.8

    def ratio(intercept):
        return ((intercept + q2_low * slope) / (intercept + q2_high * slope))

    assert ratio(4 * nominal_intercept) > ratio(nominal_intercept)


# ----------------------------------------------------------------------------------------------
# Interior dropout
# ----------------------------------------------------------------------------------------------

def test_the_window_keeps_its_length_whenever_there_is_surplus_to_backfill_from():
    q2_full = _peaks(40, seed=1)
    for n_drop in (0, 2, 4, 6):
        window, _, _ = select_peaks_with_nested_dropout(
            q2_full, 20, n_drop, np.random.default_rng(3), max_drop=6)
        assert window.size == 20, n_drop


def test_the_holes_are_nested_up_the_ladder():
    """The two-hole set is a subset of the four-hole set is a subset of the six-hole set, so the
    sparsity axis is one crystal degrading progressively rather than three unrelated draws."""
    q2_full = _peaks(40, seed=2)
    windows = {}
    for n_drop in (2, 4, 6):
        window, _, _ = select_peaks_with_nested_dropout(
            q2_full, 20, n_drop, np.random.default_rng(17), max_drop=6)
        windows[n_drop] = set(np.round(window, 12))

    nominal = set(np.round(q2_full[:20], 12))
    dropped = {n: nominal - window for n, window in windows.items()}
    assert len(dropped[2]) == 2 and len(dropped[4]) == 4 and len(dropped[6]) == 6
    assert dropped[2] < dropped[4] < dropped[6]


def test_the_draw_count_does_not_depend_on_the_rung():
    """A bundle differing only in sparsity must not also get a different error realisation, so the
    permutation is drawn at max_drop whatever n_drop is."""
    q2_full = _peaks(40, seed=4)
    positions = []
    for n_drop in (0, 2, 4, 6):
        rng = np.random.default_rng(23)
        select_peaks_with_nested_dropout(q2_full, 20, n_drop, rng, max_drop=6)
        positions.append(rng.random())
    assert len(set(positions)) == 1


def test_the_hole_count_is_capped_by_the_surplus_the_entry_actually_has():
    """A 22-peak entry can give up two and no more; the window would otherwise shorten."""
    q2_full = _peaks(22, seed=5)
    window, _, n_holes = select_peaks_with_nested_dropout(
        q2_full, 20, 6, np.random.default_rng(9), max_drop=6)
    assert n_holes == 2
    assert window.size == 20


def test_an_entry_with_no_surplus_loses_no_peaks():
    q2_full = _peaks(20, seed=6)
    window, surplus, n_holes = select_peaks_with_nested_dropout(
        q2_full, 20, 6, np.random.default_rng(9), max_drop=6)
    assert n_holes == 0
    assert surplus.size == 0
    np.testing.assert_array_equal(window, q2_full)


def test_the_mechanism_stops_before_it_stops_being_interior_dropout():
    """Above the cap almost every peak in the window is replaced by one from beyond it, which
    translates the window to high angle rather than punching holes in it."""
    q2_full = _peaks(80, seed=7)
    _, _, n_holes = select_peaks_with_nested_dropout(
        q2_full, 20, 40, np.random.default_rng(9), max_drop=40)
    assert n_holes == MAX_INTERIOR_DROPOUT


def test_the_surplus_starts_after_the_backfill_not_after_the_nominal_window():
    """The backfilled lines are in the window, so they must not also appear in the surplus."""
    q2_full = _peaks(40, seed=8)
    window, surplus, n_holes = select_peaks_with_nested_dropout(
        q2_full, 20, 6, np.random.default_rng(31), max_drop=6)
    assert n_holes == 6
    assert not (set(np.round(window, 12)) & set(np.round(surplus, 12)))
    assert window.size + surplus.size == q2_full.size - n_holes


def test_a_rung_above_its_own_maximum_is_refused():
    with pytest.raises(ValueError, match='max_drop'):
        select_peaks_with_nested_dropout(_peaks(40), 20, 8, np.random.default_rng(0), max_drop=6)


# ----------------------------------------------------------------------------------------------
# Contaminants and a second phase
# ----------------------------------------------------------------------------------------------

# A pattern whose peaks fill the range contaminants are drawn from, asked for enough of them that
# every one must simultaneously miss every peak. Acceptance decays exponentially in the count, so
# this is the shape that spins forever without a cap.
_CROWDED = np.linspace(0.20, 0.50, 60)[np.newaxis]


def test_contaminant_placement_gives_up_rather_than_spinning_forever():
    """A recorded failure lets a sweep drop the entry instead of hanging on it."""
    with pytest.raises(ContaminantPlacementError, match='within 1 attempts'):
        add_contaminants(_CROWDED.copy(), None, 30, np.random.default_rng(0), max_attempts=1)


def test_the_cap_does_not_fire_on_a_pattern_that_can_be_contaminated():
    """The cap must not turn an ordinary pattern into a failure."""
    q2 = _peaks(20, seed=21, low=0.05, high=0.5)[np.newaxis].copy()
    result = add_contaminants(q2, None, 2, np.random.default_rng(0), max_attempts=2000)
    assert result.shape == (1, 20)


def test_an_uncapped_call_still_has_no_ceiling():
    """max_attempts=None keeps the behaviour every existing caller relies on."""
    q2 = _peaks(20, seed=22, low=0.05, high=0.5)[np.newaxis].copy()
    result = add_contaminants(q2, None, 2, np.random.default_rng(0))
    assert result.shape == (1, 20)


def test_a_second_phase_injects_the_partner_s_own_lines():
    """The point of the mechanism: the injected lines are consistent with some other lattice
    rather than independently placed, which is what makes them hard to reject."""
    q2 = _peaks(20, seed=11, low=0.05, high=0.5)[np.newaxis].copy()
    partner = _peaks(60, seed=12, low=0.05, high=0.5)
    before = set(np.round(q2[0], 12))
    result = add_second_phase(q2, None, partner, 3, np.random.default_rng(5))
    injected = set(np.round(result[0], 12)) - before
    assert injected
    assert injected <= set(np.round(partner, 12))


def test_a_second_phase_keeps_the_window_length_and_sorted_order():
    q2 = _peaks(20, seed=13, low=0.05, high=0.5)[np.newaxis].copy()
    partner = _peaks(60, seed=14, low=0.05, high=0.5)
    result = add_second_phase(q2, None, partner, 3, np.random.default_rng(5))
    assert result.shape == (1, 20)
    assert np.all(np.diff(result[0]) >= 0)


def test_injected_lines_are_labelled_as_belonging_to_no_reflection():
    """They enter hkl as (0, 0, 0) exactly as contaminants do, so a labeller can tell them from
    real reflections."""
    q2 = _peaks(20, seed=15, low=0.05, high=0.5)[np.newaxis].copy()
    hkl = np.random.default_rng(16).integers(1, 5, size=(1, 20, 3)).astype(float)
    partner = _peaks(60, seed=17, low=0.05, high=0.5)
    _, hkl_out = add_second_phase(q2, hkl, partner, 3, np.random.default_rng(5))
    assert np.any(np.all(hkl_out[0] == 0, axis=1))


def test_a_partner_with_no_lines_in_range_is_refused_rather_than_silently_skipped():
    q2 = _peaks(20, seed=18, low=0.05, high=0.5)[np.newaxis].copy()
    partner = np.array([5.0, 6.0, 7.0])
    with pytest.raises(ContaminantPlacementError, match='observed range'):
        add_second_phase(q2, None, partner, 3, np.random.default_rng(5))


def test_the_low_angle_bias_moves_the_injected_lines_towards_low_q2():
    """Bias 1 picks uniformly among eligible lines; a higher bias weights the draw towards the low
    angle end, which is where a real second phase shows its strong reflections."""
    partner = _peaks(200, seed=19, low=0.05, high=0.5)
    positions = {}
    for bias in (1.0, 3.0):
        drawn = []
        for seed in range(40):
            q2 = _peaks(20, seed=100 + seed, low=0.05, high=0.5)[np.newaxis].copy()
            before = set(np.round(q2[0], 12))
            result = add_second_phase(q2, None, partner, 3, np.random.default_rng(seed),
                                      low_angle_bias=bias)
            drawn.extend(sorted(set(np.round(result[0], 12)) - before))
        positions[bias] = np.median(drawn)
    assert positions[3.0] < positions[1.0]
