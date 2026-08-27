"""Peak-list error injection: the sigma model, and the bounded contaminant rejection loop.

Run in both environments -- development first, then the runtime one, before anything is called
done:

    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python -m pytest tests/ -v
    /global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python -m pytest tests/ -v
"""
import numpy as np
import pytest

from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import add_second_phase
from mlindex.utilities.ErrorAdder import MAX_INTERIOR_DROPOUT
from mlindex.utilities.ErrorAdder import select_peaks_with_dropout


def _sparse_q2(n_peaks=20):
    # Widely spaced, so contaminants have room and the rejection loop terminates immediately.
    return np.linspace(0.05, 2.0, n_peaks)[np.newaxis].copy()


def _uncontaminatable_q2(n_peaks=600):
    # A pattern with no acceptable contaminant position anywhere, so the unbounded loop cannot
    # exit. Rejection compares each contaminant's distance to the *largest* peak's half breadth,
    # which is 0.00165 at q2 = 1; 600 peaks over [0.001, 1.0] are spaced 0.00167 apart, so the
    # rejection zones tile the whole [0.5*q2_min, q2_max] draw window. Verified by grid search:
    # the acceptable fraction of the window is 0.0000 here and 0.95 for _sparse_q2.
    return np.linspace(0.001, 1.0, n_peaks)[np.newaxis].copy()


def test_add_q2_error_scales_with_the_multiplier():
    q2_error_params = get_peak_generation_info()['q2_error_params']
    q2 = _sparse_q2()
    residuals = {}
    for multiplier in (1.0, 2.0):
        perturbed = add_q2_error(q2.copy(), None, multiplier, np.random.default_rng(0))
        expected_sigma = multiplier * (q2_error_params[0] + q2 * q2_error_params[1])
        residuals[multiplier] = (perturbed - q2) / expected_sigma
    # Same seed and the same standardising sigma, so the standardised residuals coincide -- the
    # multiplier scales both the intercept and the slope and nothing else.
    np.testing.assert_allclose(residuals[1.0], residuals[2.0], rtol=1e-12, atol=1e-12)


def test_add_q2_error_accepts_a_fractional_multiplier():
    q2 = _sparse_q2()
    perturbed = add_q2_error(q2.copy(), None, 1.5, np.random.default_rng(0))
    assert perturbed.shape == q2.shape
    assert np.all(np.diff(perturbed, axis=1) > 0)


def test_add_contaminants_conserves_the_peak_count():
    q2 = _sparse_q2()
    hkl = np.tile(np.arange(1, 21)[:, np.newaxis], (1, 3))[np.newaxis].astype(float)
    q2_out, hkl_out = add_contaminants(q2.copy(), hkl.copy(), 3, np.random.default_rng(0))
    assert q2_out.shape == q2.shape
    assert hkl_out.shape == hkl.shape
    # The contaminated list is re-sorted and truncated back to n_peaks, so contaminants displace
    # real peaks from the high-q2 end. A contaminant drawn above the new cut-off is itself
    # evicted, so the retained count is between 1 and 3 rather than exactly 3.
    assert q2_out[0, -1] < q2[0, -1]
    n_retained = np.count_nonzero(np.all(hkl_out[0] == 0, axis=1))
    assert 1 <= n_retained <= 3
    # The real peaks that survive are the lowest-q2 ones, in their original order.
    retained_real = hkl_out[0, :, 0][hkl_out[0, :, 0] != 0]
    np.testing.assert_array_equal(retained_real, np.arange(1, retained_real.size + 1))


def test_max_attempts_none_is_the_default_and_unbounded():
    # Default path on a pattern that accepts immediately: identical results with and without an
    # explicit max_attempts, so the cap changes nothing when it does not bind.
    q2 = _sparse_q2()
    without = add_contaminants(q2.copy(), None, 2, np.random.default_rng(7))
    with_cap = add_contaminants(q2.copy(), None, 2, np.random.default_rng(7), max_attempts=1000)
    np.testing.assert_array_equal(without, with_cap)


def test_max_attempts_raises_rather_than_hanging():
    q2 = _uncontaminatable_q2()
    with pytest.raises(ContaminantPlacementError):
        add_contaminants(q2.copy(), None, 2, np.random.default_rng(0), max_attempts=200)


def test_low_angle_bias_moves_contaminants_down_the_pattern():
    # The point of the bias is to land contaminants in the low-angle region the generators take
    # their information from, so the test measures where they land rather than just that the
    # parameter is accepted. Drawn over many entries: one fixed RNG would make u constant and the
    # comparison vacuous.
    q2 = _sparse_q2()
    positions = {}
    for bias in (1.0, 3.0):
        fractions = []
        for trial in range(300):
            row = q2.copy()
            low, high = 0.5 * row[0, 0], row[0, -1]
            before = set(np.round(row[0], 12))
            out = add_contaminants(row, None, 1, np.random.default_rng(trial),
                                   max_attempts=500, low_angle_bias=bias)
            new = [v for v in out[0] if round(v, 12) not in before]
            if new:
                fractions.append((new[0] - low) / (high - low))
        positions[bias] = np.median(fractions)
    assert positions[3.0] < positions[1.0] - 0.1, (
        f'bias should pull contaminants to low q2: {positions}')


def test_low_angle_bias_of_one_is_the_original_uniform_draw():
    q2 = _sparse_q2()
    default = add_contaminants(q2.copy(), None, 2, np.random.default_rng(4), max_attempts=500)
    explicit = add_contaminants(q2.copy(), None, 2, np.random.default_rng(4), max_attempts=500,
                                low_angle_bias=1.0)
    np.testing.assert_array_equal(default, explicit)


def test_dropout_punches_holes_at_low_q2_and_keeps_the_peak_count():
    # Distinct from pushing the window outwards: peaks must go missing from *within* the original
    # low-q2 range, which is what breaks the systematic-absence pattern, with the list refilled
    # from higher q2 so the count is unchanged.
    q2_full = np.linspace(0.05, 3.0, 40)
    n_peaks = 20

    nominal = select_peaks_with_dropout(q2_full, n_peaks, 0, np.random.default_rng(0))
    np.testing.assert_array_equal(nominal, q2_full[:n_peaks])

    dropped = select_peaks_with_dropout(q2_full, n_peaks, 5, np.random.default_rng(0))
    assert dropped.size == n_peaks, 'the list must be refilled to the nominal count'
    assert np.all(np.diff(dropped) > 0), 'output must stay sorted'
    assert dropped[-1] > nominal[-1], 'refilling must reach to higher q2'
    missing_from_low_range = set(np.round(nominal, 12)) - set(np.round(dropped, 12))
    assert len(missing_from_low_range) == 5, 'exactly the dropped peaks should be absent'


@pytest.mark.parametrize('n_available,n_drop,expected_dropped', [
    (40, 5, 5),     # plenty of headroom: the full request is honoured
    (21, 5, 1),     # only one spare peak, so only one can be dropped
    (20, 5, 0),     # no headroom at all
])
def test_dropout_drops_as_many_as_the_available_peaks_allow(n_available, n_drop, expected_dropped):
    # Returning a short list is not an option -- the ONNX generators reject anything under the
    # nominal count (F-044) -- so a thin entry drops fewer peaks rather than fewer than n_peaks.
    # The achieved dropout therefore varies per entry and the harness records it.
    n_peaks = 20
    q2_full = np.linspace(0.05, 3.0, n_available)
    out = select_peaks_with_dropout(q2_full, n_peaks, n_drop, np.random.default_rng(0))
    assert out.size == n_peaks
    nominal = q2_full[:n_peaks]
    assert int(np.sum(~np.isin(nominal, out))) == expected_dropped


@pytest.mark.parametrize('n_available,n_drop,expected_dropped', [
    (26, 6, 6),      # the case the old semantics could not reach at any n_drop
    (22, 1, 1),
    (22, 2, 2),
    (22, 6, 2),      # capped by the two surplus peaks
    (30, 10, 10),
    (30, 20, 10),    # MAX_INTERIOR_DROPOUT binds before the entry's surplus does
    (60, 20, 10),    # a peak-rich entry is still held to the mechanism's ceiling
    (60, 30, 10),    # n_drop above n_peaks: capped, not an error from rng.choice
])
def test_dropout_hole_count_is_exact_at_every_seed(n_available, n_drop, expected_dropped):
    """n_drop means holes in the nominal window, not draws from a widened one.

    Until 2026-08-16 the deletions were drawn from a window of n_peaks + n_drop, so a fraction of
    them landed in the backfill region and did nothing. The hole count was then a binomial mean
    rather than a guarantee -- 26 available peaks saturated at 4.62 of a requested 6, and no value
    of n_drop reached 6 because raising it widened the window as fast as it added draws. The
    existing cases above happened to pass on their seeds; these fail on most of them.
    """
    n_peaks = 20
    q2_full = np.linspace(0.05, 3.0, n_available)
    nominal = q2_full[:n_peaks]
    for seed in range(25):
        out = select_peaks_with_dropout(q2_full, n_peaks, n_drop,
                                        np.random.default_rng(seed))
        assert out.size == n_peaks, f'seed {seed} returned {out.size} peaks'
        assert int(np.sum(~np.isin(nominal, out))) == expected_dropped, f'seed {seed}'
        assert np.all(np.diff(out) > 0), f'seed {seed} returned an unsorted list'


def test_dropout_never_exceeds_the_mechanism_ceiling():
    """Past ~10 holes of 20 this stops being interior dropout and becomes a window translation.

    The nominal low-angle window is where the systematic-absence pattern lives and where the
    generators take their information; emptying it entirely is a different condition, not a more
    aggressive version of this one. The ceiling is enforced here rather than left to each caller,
    so no bundle can reach it by accident.
    """
    q2_full = np.linspace(0.05, 3.0, 80)
    n_peaks = 20
    nominal = q2_full[:n_peaks]
    for n_drop in (MAX_INTERIOR_DROPOUT, 15, 20, 40):
        out = select_peaks_with_dropout(q2_full, n_peaks, n_drop, np.random.default_rng(3))
        assert int(np.sum(~np.isin(nominal, out))) == MAX_INTERIOR_DROPOUT
        assert out.size == n_peaks


def test_zero_contaminants_terminates_without_a_redraw():
    # An empty draw has no rejectable member, so even the uncontaminatable pattern exits on the
    # first attempt. This is the guard that a cap of 1 is enough for the n_contaminants=0 bundle.
    q2 = _uncontaminatable_q2()
    q2_out = add_contaminants(q2.copy(), None, 0, np.random.default_rng(0), max_attempts=1)
    np.testing.assert_array_equal(q2_out, q2)


def _partner_lines(n_lines=200, low=0.02, high=2.5):
    # A second phase's line list: ascending, positive, and denser than the host pattern so the
    # eligibility window has plenty to choose from.
    return np.linspace(low, high, n_lines)


def test_second_phase_conserves_the_peak_count_and_marks_lines_as_contaminants():
    q2 = _sparse_q2()
    hkl = np.tile(np.arange(1, 21)[:, np.newaxis], (1, 3))[np.newaxis].astype(float)
    q2_out, hkl_out = add_second_phase(
        q2.copy(), hkl.copy(), _partner_lines(), 3, np.random.default_rng(0))

    assert q2_out.shape == q2.shape
    assert hkl_out.shape == hkl.shape
    assert np.all(np.diff(q2_out, axis=1) > 0)
    # Injected lines displace real peaks from the high-q2 end, exactly as add_contaminants does.
    n_injected = np.count_nonzero(np.all(hkl_out[0] == 0, axis=1))
    assert 1 <= n_injected <= 3
    retained_real = hkl_out[0, :, 0][hkl_out[0, :, 0] != 0]
    np.testing.assert_array_equal(retained_real, np.arange(1, retained_real.size + 1))


def test_second_phase_lines_come_from_the_partner_and_lie_in_the_observed_window():
    q2 = _sparse_q2()
    partner = _partner_lines()
    before = set(np.round(q2[0], 12))
    out = add_second_phase(q2.copy(), None, partner, 4, np.random.default_rng(1))

    injected = [value for value in out[0] if round(value, 12) not in before]
    assert injected, 'nothing was injected'
    for value in injected:
        assert np.isclose(partner, value).any(), 'an injected line is not one of the partner\'s'
        assert 0.5*q2[0, 0] <= value <= q2[0, -1], 'an injected line is outside the window'


def test_second_phase_draws_without_replacement():
    # A rank draw with collisions redrawn; a repeated line would be a silently weaker condition.
    q2 = _sparse_q2()
    partner = _partner_lines()
    for seed in range(20):
        before = set(np.round(q2[0], 12))
        out = add_second_phase(q2.copy(), None, partner, 5, np.random.default_rng(seed))
        injected = [round(value, 12) for value in out[0] if round(value, 12) not in before]
        assert len(injected) == len(set(injected)), f'seed {seed} injected a duplicate line'


def test_second_phase_bias_pulls_the_selection_towards_low_q2():
    # The whole point of the weighting: a real second phase shows its low-angle lines. Measured
    # over many entries, since one fixed RNG would make the draw constant.
    q2 = _sparse_q2()
    partner = _partner_lines()
    positions = {}
    for bias in (1.0, 3.0):
        fractions = []
        for trial in range(200):
            row = q2.copy()
            low, high = 0.5*row[0, 0], row[0, -1]
            before = set(np.round(row[0], 12))
            out = add_second_phase(row, None, partner, 1, np.random.default_rng(trial),
                                   low_angle_bias=bias)
            injected = [v for v in out[0] if round(v, 12) not in before]
            if injected:
                fractions.append((injected[0] - low) / (high - low))
        positions[bias] = np.median(fractions)
    assert positions[3.0] < positions[1.0] - 0.1, (
        f'bias should pull the selection to low q2: {positions}')


def test_second_phase_raises_when_the_partner_has_no_line_in_range():
    q2 = _sparse_q2()
    # Every partner line sits above the host's last peak, so none is observable.
    partner = np.linspace(10.0, 20.0, 50)
    with pytest.raises(ContaminantPlacementError):
        add_second_phase(q2.copy(), None, partner, 2, np.random.default_rng(0))


def test_second_phase_raises_when_every_eligible_line_overlaps_a_peak():
    # No redraw loop to hang in: the colliding lines are filtered out up front, so this is a
    # direct "nothing is placeable" answer rather than an attempt budget running out.
    q2 = _uncontaminatable_q2()
    with pytest.raises(ContaminantPlacementError):
        add_second_phase(q2.copy(), None, _partner_lines(high=1.0), 2, np.random.default_rng(0))


def test_second_phase_places_what_it_can_when_few_lines_are_eligible():
    """The failure that cost four of fifty-six entries on the first gate run.

    A partner with two eligible lines and two to place admits exactly one possible set, so the
    old rejection loop redrew that same set 2000 times and then gave up. Filtering collisions up
    front makes the sparse case succeed whenever any line is placeable, and only ever injects
    fewer than asked because fewer are available -- not because a draw was unlucky.
    """
    q2 = _sparse_q2()
    # Two lines in range, both clear of every peak: exactly the degenerate case.
    partner = np.array([0.6123, 1.457])
    before = set(np.round(q2[0], 12))
    out = add_second_phase(q2.copy(), None, partner, 3, np.random.default_rng(0))
    injected = [value for value in out[0] if round(value, 12) not in before]
    assert len(injected) == 2, f'both placeable lines should be used, got {injected}'
    for value in injected:
        assert np.isclose(partner, value).any()


# ---------------------------------------------------------------------------------------------
# Campaign 2 additions (S05, 2026-08-26). Three mechanisms, each covering a decision recorded in
# docs/fom_campaign2/STATUS.md section 6:
#   * the sigma(q2) parameterisation, Gaussian only, with severity and shape as separate axes;
#   * nested interior dropout, so the sparsity axis is paired rather than three noise draws;
#   * the displaced-peak overflow, which is what lets the hold-out set carry the window's
#     contaminants instead of being re-synthesised from the true structure (R13).
# ---------------------------------------------------------------------------------------------

from mlindex.utilities.ErrorAdder import q2_sigma_params
from mlindex.utilities.ErrorAdder import select_peaks_with_nested_dropout


def _old_add_q2_error(q2, hkl, multiplier, rng):
    """`add_q2_error` exactly as it stood before the sigma parameterisation.

    Kept verbatim rather than imported, so the bit-identity test compares against the real
    previous behaviour and not against a later edit of it.
    """
    q2_error_params = get_peak_generation_info()['q2_error_params']
    sigma_error = multiplier * (q2_error_params[0] + q2 * q2_error_params[1])
    q2 += rng.normal(loc=0, scale=sigma_error)
    q2 = np.abs(q2)
    if hkl is None:
        return np.sort(q2, axis=1)
    sort_indices = np.argsort(q2, axis=1)
    q2 = np.take_along_axis(q2, sort_indices, axis=1)
    hkl = np.take_along_axis(hkl, sort_indices[:, :, np.newaxis], axis=1)
    return q2, hkl


@pytest.mark.parametrize('multiplier', [0.05, 0.1, 1.0, 2.0, 3.5])
@pytest.mark.parametrize('with_hkl', [False, True])
def test_sigma_defaults_are_bit_identical_to_the_unparameterised_form(multiplier, with_hkl):
    # The hard constraint on the whole change: omitting the arguments must reproduce the old
    # behaviour to the last bit AND consume the same number of draws, or every bundle generated
    # before S05 stops being comparable with every bundle generated after it.
    setup = np.random.default_rng(4)
    base = np.sort(setup.uniform(0.01, 0.9, size=(4, 20)), axis=1)
    hkl = setup.integers(-6, 7, size=(4, 20, 3)).astype(float)

    old_rng, new_rng = np.random.default_rng(99), np.random.default_rng(99)
    if with_hkl:
        old_q2, old_hkl = _old_add_q2_error(base.copy(), hkl.copy(), multiplier, old_rng)
        new_q2, new_hkl = add_q2_error(base.copy(), hkl.copy(), multiplier, new_rng)
        assert (old_hkl == new_hkl).all()
    else:
        old_q2 = _old_add_q2_error(base.copy(), None, multiplier, old_rng)
        new_q2 = add_q2_error(base.copy(), None, multiplier, new_rng)

    assert old_q2.dtype == new_q2.dtype
    assert (old_q2 == new_q2).all()
    # Same stream position afterwards: a differing draw count desynchronises every later mechanism.
    assert old_rng.integers(0, 2 ** 62) == new_rng.integers(0, 2 ** 62)


def test_sigma_params_resolve_to_the_repository_model():
    expected = get_peak_generation_info()['q2_error_params']
    assert q2_sigma_params() == (float(expected[0]), float(expected[1]))
    # Either may be overridden alone; the other still falls back.
    assert q2_sigma_params(intercept=1e-3) == (1e-3, float(expected[1]))
    assert q2_sigma_params(slope=2e-3) == (float(expected[0]), 2e-3)


def test_the_intercept_is_a_shape_axis_a_multiplier_cannot_reach():
    # This is the entire justification for exposing the intercept separately (DWMM, 2026-08-26):
    # a multiplier scales sigma at low and high q2 by the same factor, so it cannot change the
    # *shape* of sigma(q2). Raising the intercept alone must move low q2 more than high q2.
    intercept, slope = q2_sigma_params()
    q2_low, q2_high = 0.02, 0.9

    def sigma(mult, icept):
        return mult * (icept + np.array([q2_low, q2_high]) * slope)

    multiplied = sigma(4.0, intercept) / sigma(1.0, intercept)
    reshaped = sigma(1.0, intercept * 4.0) / sigma(1.0, intercept)

    assert np.allclose(multiplied[0], multiplied[1])       # a multiplier is flat across q2
    assert reshaped[0] > reshaped[1] * 2                   # the intercept is not
    assert reshaped[1] < multiplied[1]


@pytest.mark.parametrize('n_available', [26, 40, 60])
def test_nested_dropout_holes_are_prefixes_of_one_another(n_available):
    # The sparsity axis is a paired comparison only if the smaller hole sets are subsets of the
    # larger ones -- the same crystal degrading, not three unrelated crystals.
    q2_full = np.sort(np.random.default_rng(3).uniform(0.01, 2.0, size=n_available))
    nominal = q2_full[:20]

    hole_sets = []
    for n_drop in (2, 4, 6):
        window, _, achieved = select_peaks_with_nested_dropout(
            q2_full, 20, n_drop, np.random.default_rng(11), n_holdout=20, max_drop=6)
        hole_sets.append(set(np.flatnonzero(~np.isin(nominal, window)).tolist()))
        assert achieved == min(n_drop, n_available - 20, MAX_INTERIOR_DROPOUT)

    assert hole_sets[0] < hole_sets[1] < hole_sets[2]


@pytest.mark.parametrize('n_drop', [0, 2, 4, 6])
def test_nested_dropout_consumes_the_same_draws_whatever_the_rung(n_drop):
    # Without this the hole count shifts every draw after it, so a bundle differing only in
    # sparsity also gets a different error realisation (C2-F-005).
    q2_full = np.sort(np.random.default_rng(3).uniform(0.01, 2.0, size=60))
    reference = np.random.default_rng(11)
    select_peaks_with_nested_dropout(q2_full, 20, 6, reference, n_holdout=20, max_drop=6)
    expected = reference.integers(0, 2 ** 62)

    rng = np.random.default_rng(11)
    select_peaks_with_nested_dropout(q2_full, 20, n_drop, rng, n_holdout=20, max_drop=6)
    assert rng.integers(0, 2 ** 62) == expected


@pytest.mark.parametrize('n_drop', [2, 4, 6, 10])
def test_nested_dropout_matches_the_unnested_form_at_its_maximum(n_drop):
    # At n_drop == max_drop the two make the same rng.choice call, so they must agree exactly.
    # This is what pins the nested version to the mechanism campaign 1 calibrated.
    q2_full = np.sort(np.random.default_rng(3).uniform(0.01, 2.0, size=60))
    unnested = select_peaks_with_dropout(q2_full, 20, n_drop, np.random.default_rng(11))
    nested, _, _ = select_peaks_with_nested_dropout(
        q2_full, 20, n_drop, np.random.default_rng(11), n_holdout=20, max_drop=n_drop)
    assert (unnested == nested).all()


def test_nested_dropout_holdout_starts_after_the_backfill():
    # The backfilled peaks are IN the window, so the surplus cannot start at the nominal 21st
    # peak or the two sets would overlap and the hold-out would score peaks it was fitted on.
    q2_full = np.sort(np.random.default_rng(3).uniform(0.01, 2.0, size=60))
    for n_drop in (0, 2, 4, 6):
        window, holdout, achieved = select_peaks_with_nested_dropout(
            q2_full, 20, n_drop, np.random.default_rng(11), n_holdout=20, max_drop=6)
        assert window.size == 20
        assert holdout.size == 20
        assert not set(window.tolist()) & set(holdout.tolist())
        assert np.isclose(holdout[0], q2_full[20 + achieved])


def test_nested_dropout_refuses_a_rung_above_its_maximum():
    q2_full = np.sort(np.random.default_rng(3).uniform(0.01, 2.0, size=60))
    with pytest.raises(ValueError, match='max_drop'):
        select_peaks_with_nested_dropout(q2_full, 20, 8, np.random.default_rng(11), max_drop=6)


@pytest.mark.parametrize('with_hkl', [False, True])
def test_contaminant_overflow_is_off_by_default_and_changes_nothing(with_hkl):
    setup = np.random.default_rng(5)
    base = np.sort(setup.uniform(0.05, 1.5, size=(1, 20)), axis=1)
    hkl = setup.integers(-5, 6, size=(1, 20, 3)).astype(float)

    plain_rng, overflow_rng = np.random.default_rng(21), np.random.default_rng(21)
    if with_hkl:
        plain_q2, plain_hkl = add_contaminants(base.copy(), hkl.copy(), 2, plain_rng,
                                               max_attempts=2000)
        (over_q2, over_hkl), _, _ = add_contaminants(base.copy(), hkl.copy(), 2, overflow_rng,
                                                     max_attempts=2000, return_overflow=True)
        assert (plain_hkl == over_hkl).all()
    else:
        plain_q2 = add_contaminants(base.copy(), None, 2, plain_rng, max_attempts=2000)
        over_q2, _, _ = add_contaminants(base.copy(), None, 2, overflow_rng, max_attempts=2000,
                                         return_overflow=True)

    assert (plain_q2 == over_q2).all()
    assert plain_rng.integers(0, 2 ** 62) == overflow_rng.integers(0, 2 ** 62)


@pytest.mark.parametrize('n_contaminants', [1, 2, 4])
def test_contaminant_overflow_is_exactly_the_displaced_real_peaks(n_contaminants):
    # The window is re-truncated to n_peaks after insertion, so a contaminant landing inside it
    # pushes a real reflection out. That reflection is not lost -- it belongs in the hold-out.
    setup = np.random.default_rng(5)
    base = np.sort(setup.uniform(0.05, 1.5, size=(1, 20)), axis=1)
    window, overflow, _ = add_contaminants(base.copy(), None, n_contaminants,
                                           np.random.default_rng(21), max_attempts=2000,
                                           return_overflow=True)

    # The overflow is exactly as many lines as were inserted, and NO real peak is lost: every
    # original peak is either still in the window or has moved into the overflow. That
    # conservation property is what the hold-out set relies on.
    assert overflow[0].size == n_contaminants
    kept_or_displaced = np.concatenate([window[0], overflow[0]])
    assert np.isin(base[0], kept_or_displaced).all(), 'a real peak was lost entirely'

    displaced = base[0][~np.isin(base[0], window[0])]
    assert np.isin(displaced, overflow[0]).all()
    # The overflow is not necessarily all real peaks: an inserted line drawn near the top of the
    # range can itself sort above the window and be truncated out. It still belongs in the
    # hold-out -- a contaminant above the fitted window is part of the observed pattern.
    assert overflow[0].size >= displaced.size
    # The window's upper edge moves DOWN under contamination, which is why "the surplus starts
    # at peak 21 of the true list" is wrong.
    assert overflow[0].min() >= window[0].max()


def test_second_phase_overflow_is_exactly_the_displaced_real_peaks():
    setup = np.random.default_rng(5)
    base = np.sort(setup.uniform(0.05, 1.5, size=(1, 20)), axis=1)
    partner = _partner_lines()

    plain = add_second_phase(base.copy(), None, partner, 3, np.random.default_rng(31))
    window, overflow, _ = add_second_phase(base.copy(), None, partner, 3,
                                           np.random.default_rng(31), return_overflow=True)

    assert (plain == window).all()
    # Same conservation property as for contaminants, and the same caveat: some of the overflow
    # may be injected second-phase lines that fell above the window rather than displaced real
    # peaks. Both belong in the hold-out.
    assert overflow[0].size == 3
    kept_or_displaced = np.concatenate([window[0], overflow[0]])
    assert np.isin(base[0], kept_or_displaced).all(), 'a real peak was lost entirely'
    displaced = base[0][~np.isin(base[0], window[0])]
    assert np.isin(displaced, overflow[0]).all()
    assert overflow[0].min() >= window[0].max()
