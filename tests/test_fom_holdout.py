"""S10: scoring a fitted cell on the surplus peaks it never saw.

`get_holdout_fom` shipped on this branch with **no behavioural coverage at all**. Its tests lived
in `tests/test_fom_cv.py`, which S02 dropped along with the cross-validated family those tests
mostly covered -- so the three cases that apply to the hold-out form went with them. They are
restored here, from `fom@7c137c3`, alongside the cases S10a's own machinery needs.

The properties that matter, and why each is here rather than assumed:

* **no refit.** The whole claim is that the cell is scored on peaks it was never fitted to. A refit
  would turn this into the cross-validated family campaign 2 dropped as a designed negative.
* **an entry with too short a surplus is missing, not bad.** It must drop out of a paired
  comparison rather than enter it scoring zero.
* **a peak budget cannot change which way a merit points**, and it cannot change the answer through
  a route other than the peaks it adds.
* **seeding contaminants into the surplus must not touch the window**, or the candidate pool it
  conditioned would no longer match the peak list.
"""

import numpy as np
import pytest

from mlindex.model_training import FomBenchmark as FB
from mlindex.model_training import FomMetrics as FM
from mlindex.utilities import ErrorAdder as EA
from mlindex.utilities.FigureOfMerits import SIGMA_TREATMENT
from mlindex.utilities.FigureOfMerits import get_hkl_matrix
from mlindex.utilities.FigureOfMerits import get_holdout_fom


LATTICE_SYSTEM = 'orthorhombic'
BRAVAIS_LATTICE = 'oP'


def _reference_list(max_index=5):
    """A generic hkl list with the origin removed.

    The origin is dropped here rather than inside the merit, which is where the production path
    drops it too -- `get_spacegroup_hkl_ref` never emits it. Left in, an all-zero row would give a
    literal 0.0 calculated line that a low-q2 peak could be assigned to and that de Wolff's N would
    count. Carried over verbatim from `fom:tests/test_fom_cv.py`.
    """
    grid = np.stack(np.meshgrid(*[np.arange(0, max_index)]*3, indexing='ij'), axis=-1)
    hkl_ref = grid.reshape(-1, 3)
    return hkl_ref[np.any(hkl_ref != 0, axis=1)]


@pytest.fixture
def cells():
    """Two candidates: one that generated the peaks, and one distorted away from it."""
    good = np.array([0.04, 0.02, 0.01])
    bad = good*np.array([1.7, 0.6, 1.3])
    return np.vstack([good, bad])


@pytest.fixture
def holdout(cells):
    """Five surplus peaks taken from the FIRST candidate's own spectrum, ascending."""
    hkl_ref = _reference_list()
    lines = np.matmul(cells[:1], get_hkl_matrix(hkl_ref, LATTICE_SYSTEM).T)[0]
    return np.sort(lines)[40:45]


# --------------------------------------------------------------------------------------------
# Restored from fom@7c137c3, tests/test_fom_cv.py
# --------------------------------------------------------------------------------------------

def test_the_holdout_form_needs_no_refit_and_rewards_the_cell_that_predicts(cells, holdout):
    features = get_holdout_fom(
        holdout, cells, _reference_list(), LATTICE_SYSTEM, BRAVAIS_LATTICE)
    assert features['ho_M'][0] > features['ho_M'][1]
    # Every surplus peak scored for both candidates: no silent voiding.
    assert features['ho_n_scored'].tolist() == [5.0, 5.0]


def test_the_holdout_form_does_not_modify_its_arguments(cells, holdout):
    hkl_ref = _reference_list()
    before = (holdout.copy(), cells.copy(), hkl_ref.copy())
    get_holdout_fom(holdout, cells, hkl_ref, LATTICE_SYSTEM, BRAVAIS_LATTICE)
    assert np.array_equal(holdout, before[0])
    assert np.array_equal(cells, before[1])
    assert np.array_equal(hkl_ref, before[2])


def test_every_emitted_column_declares_a_sigma_treatment(cells, holdout):
    features = get_holdout_fom(
        holdout, cells, _reference_list(), LATTICE_SYSTEM, BRAVAIS_LATTICE,
        sigma_entrywise=1e-4)
    for key in features:
        assert key in SIGMA_TREATMENT, key
    assert SIGMA_TREATMENT['ho_chi2'] == 'in-sample'
    assert SIGMA_TREATMENT['ho_M'] == 'free'


# --------------------------------------------------------------------------------------------
# S10a: the reference lines may be handed in, and every emitted column is oriented
# --------------------------------------------------------------------------------------------

def test_passing_the_reference_lines_changes_nothing(cells, holdout):
    """The sweep's whole optimisation, pinned bit-for-bit.

    `q2_ref_calc` depends on the cell and not on the peaks, so a caller sweeping the peak budget
    can build it once. That is only safe if the two routes agree exactly -- and the route matters
    to the last bit here, because de Wolff's cut-off IS one of the reference lines.
    """
    hkl_ref = _reference_list()
    derived = get_holdout_fom(holdout, cells, hkl_ref, LATTICE_SYSTEM, BRAVAIS_LATTICE)
    q2_ref_calc = np.matmul(cells, get_hkl_matrix(hkl_ref, LATTICE_SYSTEM).T)
    handed = get_holdout_fom(holdout, cells, hkl_ref, LATTICE_SYSTEM, BRAVAIS_LATTICE,
                             q2_ref_calc=q2_ref_calc)
    assert set(derived) == set(handed)
    for key in derived:
        assert np.array_equal(derived[key], handed[key], equal_nan=True), key


def test_every_holdout_merit_has_a_recorded_direction():
    """A merit ranked backwards looks like a bad merit rather than a bug (C2-F-085)."""
    merits = ('ho_M20', 'ho_M', 'ho_M_tilde', 'ho_M_rev', 'ho_M_sym', 'ho_Minfo',
              'ho_raw', 'ho_tail_nll')
    for name in merits:
        assert FM.holdout_orientation_of(FB.holdout_column(name, 5)) is FM.orientation_of(name)
    # The support and coverage diagnostics are NOT merits and must refuse to be ranked.
    for name in ('ho_N_cal', 'ho_n_scored', 'ho_ref_reach'):
        with pytest.raises(KeyError):
            FM.orientation_of(name)


def test_the_budget_suffix_round_trips():
    assert FB.holdout_column('ho_M_sym', 5) == 'ho_M_sym__n5'
    columns = FB.holdout_columns((1, 5))
    assert len(columns) == 2*len(FB.HOLDOUT_MERIT_NAMES)
    assert 'ho_M20__n1' in columns and 'ho_M20__n5' in columns


# --------------------------------------------------------------------------------------------
# S10a: seeding contaminants into the surplus
# --------------------------------------------------------------------------------------------

def _surplus():
    q2 = np.sort(np.random.default_rng(0).uniform(0.6, 1.4, 20))
    return q2, np.ones((20, 3), dtype=np.int16)


def test_a_clean_bundle_draws_nothing():
    """`cont0` must come back untouched, not merely rarely contaminated."""
    q2, hkl = _surplus()
    assert EA.surplus_contaminant_rate(0, 20) == 0.0
    out_q2, out_hkl, added = EA.add_surplus_contaminants(
        q2, hkl, 0.0, np.random.default_rng(7))
    assert added == 0
    assert np.array_equal(out_q2, q2)
    assert np.array_equal(out_hkl, hkl)


def test_seeded_contaminants_are_tagged_sorted_and_reproducible():
    q2, hkl = _surplus()
    # Seed 0 draws Poisson(2.0) = 2. Most seeds draw something, but a zero draw is a legitimate
    # outcome rather than a failure -- `test_a_clean_bundle_draws_nothing` covers the empty case --
    # so this one is pinned to a seed that adds, and asserts on what was added.
    first = EA.add_surplus_contaminants(q2, hkl, 2.0, np.random.default_rng(0))
    again = EA.add_surplus_contaminants(q2, hkl, 2.0, np.random.default_rng(0))
    assert first[2] > 0
    # Same stream, same lines, to the last bit -- PROTOCOL section 6.
    assert np.array_equal(first[0], again[0])
    assert first[0].size == q2.size + first[2]
    assert np.all(np.diff(first[0]) >= 0)
    # Each injected line is tagged (0, 0, 0), exactly as the window mechanisms tag theirs.
    assert int((np.abs(first[1]).sum(axis=1) == 0).sum()) == first[2]
    # The originals survive unmodified.
    assert np.array_equal(np.sort(q2), np.sort(first[0][np.abs(first[1]).sum(axis=1) != 0]))


def test_seeding_does_not_modify_its_input():
    """The stored surplus is shared across every candidate of the entry; mutating it corrupts them."""
    q2, hkl = _surplus()
    before = q2.copy()
    EA.add_surplus_contaminants(q2, hkl, 3.0, np.random.default_rng(3))
    assert np.array_equal(q2, before)


def test_seeded_lines_land_inside_the_surplus_range_not_at_its_head():
    """The displaced contaminants already in the data all sit at position 0; these must not.

    That difference is the point of seeding them: a merit scored on the nearest few surplus peaks
    would otherwise see every contaminant the benchmark has, which is a distribution no instrument
    produces.
    """
    q2, hkl = _surplus()
    positions = []
    for seed in range(60):
        out_q2, out_hkl, added = EA.add_surplus_contaminants(
            q2, hkl, 2.0, np.random.default_rng(seed))
        if added:
            positions.extend(np.flatnonzero(np.abs(out_hkl).sum(axis=1) == 0).tolist())
            assert out_q2[np.abs(out_hkl).sum(axis=1) == 0].min() >= q2[0]
            assert out_q2[np.abs(out_hkl).sum(axis=1) == 0].max() <= q2[-1]
    assert len(set(positions)) > 1, 'every seeded line landed at the same index'


# ---------------------------------------------------------------------------------------
# S10b: the peak list a candidate is scored on, and the cubic free-peaks arm (C2-Q-026)
# ---------------------------------------------------------------------------------------
def test_the_uniform_mode_ignores_the_window_entirely():
    """'surplus' must return the stored surplus untouched, whatever the candidate was fitted on.

    This is the definition every other step in the campaign means by hold-out, so it has to be
    provably unaffected by the arm added beside it.
    """
    q2_obs = np.linspace(0.1, 0.5, 20)
    q2_holdout = np.linspace(0.55, 0.8, 20)
    for n_peaks in (10, 20):
        peaks, offset = FB.holdout_peaks(q2_obs, q2_holdout, n_peaks, 'surplus')
        assert offset == 0
        np.testing.assert_array_equal(peaks, q2_holdout)


def test_free_window_hands_cubic_its_unused_window_peaks_and_nobody_else_anything():
    """The whole point of the arm: ten free peaks on cubic, none anywhere else.

    A cubic cell is fitted on ten peaks and everything else on twenty (R5), so `q2_obs[10:]` are
    already hold-out for it and `q2_holdout` does not contain them (C2-F-101). At a fixed pattern
    length a cubic candidate therefore scores 10 + n_extra peaks where an oP one scores n_extra.
    """
    q2_obs = np.linspace(0.1, 0.5, 20)
    q2_holdout = np.linspace(0.55, 0.8, 20)

    peaks, offset = FB.holdout_peaks(q2_obs, q2_holdout, 10, 'free_window')
    assert offset == 10
    np.testing.assert_allclose(peaks[:10], q2_obs[10:])
    np.testing.assert_allclose(peaks[10:], q2_holdout)
    # A budget of five surplus peaks becomes fifteen scored peaks -- the 50 % larger budget.
    assert peaks[:offset + 5].size == 15

    # And an ordinary lattice is byte-identical to the uniform definition, so an aggregate over
    # the arm cannot quietly move a non-cubic number.
    peaks, offset = FB.holdout_peaks(q2_obs, q2_holdout, 20, 'free_window')
    assert offset == 0
    np.testing.assert_array_equal(peaks, q2_holdout)


def test_free_equal_takes_the_same_peaks_at_the_same_count():
    """The secondary arm: which peaks, holding how many fixed.

    Separating the two is what makes the primary arm interpretable -- a gain at a larger budget
    could be the budget or could be the peaks, and only this arm tells them apart.
    """
    q2_obs = np.linspace(0.1, 0.5, 20)
    q2_holdout = np.linspace(0.55, 0.8, 20)
    peaks, offset = FB.holdout_peaks(q2_obs, q2_holdout, 10, 'free_equal')
    assert offset == 0
    # Five peaks, but peaks 11-15 of the pattern rather than 21-25.
    np.testing.assert_allclose(peaks[:5], q2_obs[10:15])


def test_an_unknown_mode_raises_rather_than_falling_back():
    """A typo must not silently score the uniform definition and be reported as the arm."""
    with pytest.raises(ValueError, match='mode must be one of'):
        FB.holdout_peaks(np.linspace(0.1, 0.5, 20), np.linspace(0.6, 0.8, 5), 10, 'free')


def test_what_the_budget_axis_consumes_in_each_mode():
    """The two arms count different things, and the difference is a result rather than a wrinkle.

    'surplus' and 'free_window' both consume `n_extra` SURPLUS peaks, so an n_extra of 5 is a
    25-peak pattern in both and the paired comparison between them is at a fixed pattern length.
    'free_equal' consumes `n_extra` HOLD-OUT peaks wherever they come from -- and for a cubic
    candidate the first ten of those are unused *window* peaks, so it consumes **no surplus at
    all** and describes a plain 20-peak pattern. A cubic cell can therefore be scored out of
    sample on data that carries no surplus whatever, which is the arm's most useful property.
    """
    q2_obs = np.linspace(0.1, 0.5, 20)
    q2_holdout = np.linspace(0.55, 0.8, 20)

    for mode in ('surplus', 'free_window'):
        peaks, offset = FB.holdout_peaks(q2_obs, q2_holdout, 10, mode)
        scored = peaks[:offset + 5]
        assert np.sum(np.isin(scored, q2_holdout)) == 5, mode

    peaks, offset = FB.holdout_peaks(q2_obs, q2_holdout, 10, 'free_equal')
    scored = peaks[:offset + 5]
    assert np.sum(np.isin(scored, q2_holdout)) == 0
    assert np.sum(np.isin(scored, q2_obs)) == 5


# ---------------------------------------------------------------------------------------
# S10b: the reporting half
# ---------------------------------------------------------------------------------------
def test_a_lattice_floor_is_used_for_a_lattice_claim():
    """PROTOCOL section 8: a per-lattice claim reads against that lattice's own floor.

    The floors span 2.1x, so reading a cF claim against the aggregate would be wrong by an order
    of magnitude (C2-F-081). This is the guard on that, not a formatting test.
    """
    import pandas as pd
    from mlindex.model_training import FomHoldoutReport as HR
    frame = pd.DataFrame({'scope': ['all', 'hard', 'cF', 'aP', 'unknown'],
                          'delta_pp': [5.0, 5.0, 5.0, 5.0, 5.0]})
    out = HR.in_standard_errors(frame, aggregate_floor=0.5,
                                per_lattice_floor={'cF': 2.5, 'aP': 1.0})
    by_scope = dict(zip(out['scope'], out['standard_errors']))
    assert by_scope['all'] == pytest.approx(10.0)
    assert by_scope['cF'] == pytest.approx(2.0)
    assert by_scope['aP'] == pytest.approx(5.0)
    # A lattice with no measured floor gets no gate, rather than borrowing the aggregate.
    assert np.isnan(by_scope['unknown'])


def test_coverage_rates_keep_their_denominators():
    """A rate without its denominator cannot be pooled, and the figure pools lattices into groups."""
    import pandas as pd
    from mlindex.model_training import FomHoldoutReport as HR
    coverage = pd.DataFrame({
        'n_extra': [5, 5], 'condition_bundle': ['b', 'b'], 'bravais_lattice': ['cF', 'aP'],
        'n_candidates': [100, 200], 'n_scored': [100, 200],
        'n_mrev_supported': [10, 180], 'n_ref_reached': [80, 200]})
    out = HR.coverage_table(coverage)
    assert {'n_mrev_supported', 'n_ref_reached', 'n_scored'} <= set(out.columns)
    overall = out.loc[out['scope'] == 'all'].iloc[0]
    # 190 of 300, not the mean of 10 % and 90 %.
    assert overall['mrev_support_rate'] == pytest.approx(190/300)
