"""The widened merit sidecar: both sides of the `M_rev` floor, from one evaluation.

`reduced_merits` stores `M_rev` floored at `M_REV_MIN_N_CAL` and `M_rev_unfloored` beside it, with
`N_cal` as the support the floor tested. The point of the widening is that a floored 0.0 used to
mean three different things -- the floor fired, N_cal was zero, the candidate is degenerate -- and
nothing stored could tell them apart (C2-Q-017, C2-F-086).

What has to hold, and what these tests pin:

  * deriving the floored value from the unfloored one is **exact**, not close. If it were merely
    close, the widening would silently change every stored `M_rev` in the campaign.
  * the two new columns are recomputed merits, so `run_fom_dump` must drop them again -- otherwise
    they leak into a regenerated pool's schema, which is C2-F-073's failure mode.
  * `M_sym` remains the product, on both sides of the floor.
"""

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics
from mlindex.utilities.FigureOfMerits import get_M_rev_sym

from tests.test_fom_literature import _m_rev_blowup_case

SLICE_ROOT = FomBenchmark.Path(__file__).parent.parent / 'mlindex' / 'data' / 'fom_benchmark_c2'


# ---------------------------------------------------------------------------------------
# The identity the widening rests on
# ---------------------------------------------------------------------------------------
def test_the_floor_is_a_mask_over_identical_arithmetic():
    """`where(N_cal >= 10, unfloored, 0)` reproduces the floored call exactly.

    Exactly, not approximately: `assert_array_equal`. The floor is `usable &= n_cal >= min_n_cal`
    applied to the same division on the same operands, so any difference at all would mean the two
    branches do not share arithmetic and the widening is not free after all.
    """
    case = _m_rev_blowup_case()
    args = (case['q2_obs'], case['q2_calc'], case['q2_ref_calc'])

    floored_tilde, floored_rev, floored_sym = get_M_rev_sym(*args)
    raw_tilde, raw_rev, _, n_cal = get_M_rev_sym(*args, min_n_cal=None, return_n_cal=True)

    derived = np.where(n_cal >= FomBenchmark.M_REV_MIN_N_CAL, raw_rev, 0.0)
    np.testing.assert_array_equal(derived, floored_rev)
    np.testing.assert_array_equal(raw_tilde, floored_tilde)
    np.testing.assert_array_equal(raw_tilde*derived, floored_sym)

    # And the case is not vacuous: this row really is floored, and its raw value is the blow-up
    # the floor exists to suppress. A test where the floor never fires would pass trivially.
    assert n_cal[0] == 3
    assert floored_rev[0] == 0.0
    assert raw_rev[0] > 1e11


def test_M_rev_unfloored_preserves_what_the_floor_discards():
    """The audit column answers the question the floored column cannot.

    Three states used to collapse onto 0.0. With `N_cal` stored, "the floor fired at N_cal = 3" is
    distinguishable from "N_cal was zero", and the raw value is recoverable.
    """
    case = _m_rev_blowup_case()
    _, _, _, n_cal = get_M_rev_sym(case['q2_obs'], case['q2_calc'], case['q2_ref_calc'],
                                   min_n_cal=None, return_n_cal=True)
    assert 0 < n_cal[0] < FomBenchmark.M_REV_MIN_N_CAL


# ---------------------------------------------------------------------------------------
# The column contract
# ---------------------------------------------------------------------------------------
def test_recomputed_columns_are_the_ranking_set_plus_the_audit_pair():
    assert FomBenchmark.RECOMPUTED_MERIT_COLUMNS == (
        'M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap', 'N_cal', 'M_rev_unfloored')
    # M20 is stored, never recomputed, so it is in the ranking set and not in this one. That
    # asymmetry is the whole reason the drop list keys on this constant.
    assert 'M20' in FomBenchmark.REDUCED_MERIT_COLUMNS
    assert 'M20' not in FomBenchmark.RECOMPUTED_MERIT_COLUMNS


def test_an_empty_pool_still_returns_every_recomputed_column():
    empty = pd.DataFrame(columns=['entry_id', 'condition_bundle', 'bravais_lattice',
                                  'lattice_system', 'spacegroup', 'n_peaks', 'xnn', 'candidate_id'])
    out = FomBenchmark.reduced_merits(empty, pd.DataFrame(columns=['entry_id', 'q2_obs']))
    assert list(out.columns) == list(FomBenchmark.RECOMPUTED_MERIT_COLUMNS)


def test_the_dump_drops_every_recomputed_column_so_none_leaks_into_the_schema():
    """The guard on C2-F-073's failure mode, asserted on the real drop expression.

    `run_fom_dump` keeps `column not in RECOMPUTED_MERIT_COLUMNS`. Keyed on the *ranking* set
    instead -- which is what it used to do -- `N_cal` and `M_rev_unfloored` would survive into a
    regenerated pool's candidate files, and a pool whose schema differs from the one the array
    wrote is a pool that can no longer be checksummed against it.
    """
    from mlindex.model_training.FomBenchmark import CANDIDATE_COLUMNS
    columns = list(CANDIDATE_COLUMNS) + list(FomBenchmark.RECOMPUTED_MERIT_COLUMNS)
    kept = [c for c in columns if c not in FomBenchmark.RECOMPUTED_MERIT_COLUMNS]
    assert 'M20' in kept, 'M20 is a stored column and must survive the drop'
    for added in FomBenchmark.RECOMPUTED_MERIT_COLUMNS:
        assert added not in kept


# ---------------------------------------------------------------------------------------
# Direction of merit -- C2-F-085
# ---------------------------------------------------------------------------------------
def test_the_counting_merits_are_lower_is_better():
    """Three of the seven count something you want less of, and `evaluate` defaults to True.

    A reversed ranking is not an error, it is a merit that looks terrible -- which is how `X_N`
    survived reversed through every S08 floor table.
    """
    for merit in ('X_N', 'n_over', 'max_gap'):
        assert FomMetrics.orientation_of(merit) is False
    for merit in ('M20', 'M_tilde', 'M_rev', 'M_sym'):
        assert FomMetrics.orientation_of(merit) is True
    assert set(FomMetrics.RANK_EXACT_MERITS) <= set(FomMetrics.HIGHER_IS_BETTER)


def test_an_unrecorded_merit_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match='No recorded direction of merit'):
        FomMetrics.orientation_of('some_merit_nobody_oriented')


# ---------------------------------------------------------------------------------------
# On real data
# ---------------------------------------------------------------------------------------
@pytest.mark.slow
def test_reduced_merits_on_a_real_slice_chunk():
    """The identity again, on real candidates rather than a constructed blow-up.

    The constructed case exercises one row at N_cal = 3. This exercises whatever the pool actually
    contains, including the rows where the floor does not fire, which is the majority.
    """
    pytest.importorskip('pyarrow')
    path = SLICE_ROOT / 'candidates_c2_error1_cont0_mP.parquet'
    if not path.exists():
        pytest.skip('Benchmark B\'s slice is absent (untracked).')

    entries = FomBenchmark.load_entries(SLICE_ROOT)
    candidates = pd.read_parquet(path).head(2000)
    merits = FomBenchmark.reduced_merits(candidates, entries)

    assert list(merits.columns) == list(FomBenchmark.RECOMPUTED_MERIT_COLUMNS)
    assert merits.notna().all().all()

    floored = np.where(merits['N_cal'].to_numpy() >= FomBenchmark.M_REV_MIN_N_CAL,
                       merits['M_rev_unfloored'].to_numpy(), 0.0)
    np.testing.assert_array_equal(floored, merits['M_rev'].to_numpy())
    np.testing.assert_array_equal(merits['M_tilde'].to_numpy()*merits['M_rev'].to_numpy(),
                                  merits['M_sym'].to_numpy())
    # The unfloored value is never below the floored one: the floor only ever removes.
    assert (merits['M_rev_unfloored'] >= merits['M_rev']).all()


# ---------------------------------------------------------------------------------------
# The posterior-based counting merits -- C2-Q-025
# ---------------------------------------------------------------------------------------
def test_the_soft_counts_are_the_expected_value_of_the_hard_ones():
    """Same question, same direction, same scale -- but continuous.

    Each soft count is the expectation of the integer its hard counterpart returns, so a pattern
    every peak explains confidently still scores near zero and lower is still better. If that
    correspondence broke, the soft form would be a different merit wearing the same name.
    """
    import numpy as np
    from mlindex.utilities.FigureOfMerits import get_soft_counts, get_X_N

    # Twenty peaks sitting exactly on twenty of two hundred calculated lines: nothing unexplained.
    q2_ref = np.linspace(0.1, 2.0, 200)[np.newaxis]
    index = np.linspace(0, 199, 20).astype(int)
    q2_obs = q2_ref[0][index]
    q2_calc = q2_ref[:, index]

    soft = get_soft_counts(q2_obs, q2_calc, q2_ref, 'orthorhombic')
    assert get_X_N(q2_obs, q2_calc, q2_ref)[0] == 0
    # Not exactly zero -- a posterior never awards probability 1 while a competitor exists -- but
    # far below the twenty peaks it is counting over.
    assert 0.0 <= soft['X_N_soft'][0] < 5.0
    for name in ('X_N_soft', 'n_over_soft', 'max_gap_soft'):
        assert np.isfinite(soft[name]).all()
        assert (soft[name] >= 0).all()


def test_the_soft_counts_break_the_ties_that_make_the_hard_ones_useless():
    """The point of C2-Q-025: a binary criterion cannot order a pool, a posterior can.

    `X_N` gives 1 522 of 8 272 real candidates the same score (C2-F-095), so its ranking is
    settled by the tie-break rather than by the merit. This is the property that has to change.
    """
    import numpy as np
    from mlindex.utilities.FigureOfMerits import get_soft_counts, get_X_N

    rng = np.random.default_rng(12345)
    q2_ref = np.sort(rng.uniform(0.1, 2.0, size=(40, 150)), axis=1)
    q2_obs = np.sort(rng.uniform(0.1, 1.5, size=12))
    q2_calc = q2_ref[:, :12]

    hard = get_X_N(q2_obs, q2_calc, q2_ref)
    soft = get_soft_counts(q2_obs, q2_calc, q2_ref, 'orthorhombic')['X_N_soft']
    assert len(np.unique(soft)) > len(np.unique(hard)), (
        f'soft resolved {len(np.unique(soft))} values, hard {len(np.unique(hard))}')


def test_the_soft_counts_are_oriented_and_kept_out_of_the_verified_sidecar():
    from mlindex.model_training import FomMetrics as FM
    for name in FomBenchmark.SOFT_MERIT_COLUMNS:
        assert FM.orientation_of(name) is False
        # They are NOT in the subsampler's retention rule, so they must not be smuggled into the
        # set that claims rank-exactness on Benchmark B (C2-R-013).
        assert name not in FM.RANK_EXACT_MERITS
        assert name not in FomBenchmark.RECOMPUTED_MERIT_COLUMNS
