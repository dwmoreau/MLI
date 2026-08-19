"""S10's predictive merits: does the cell predict a peak it was not fitted to?

Two things can go wrong here and neither announces itself. The refit can re-assign the retained
peaks, which lets the cell chase the held-out ones through the assignment and leaves nothing to
measure. And a rank-deficient retained set makes `gauss_newton_solve` return a *zero step*, so the
"refit" is the full fit and every held-out peak is predicted perfectly -- a silent, spectacular
false positive. Both are asserted rather than assumed.

The null calibration is the other load-bearing test. Under de Wolff's idealised null the held-out
ratio |dQ|/Delta is Exp(1), so `cv_M` is 1 by construction; that is what makes the merit comparable
across lattices without a fitted normalisation, and it is checked here on a construction that
actually satisfies the null. The benchmark's refined survivors do not satisfy it and are not
expected to.
"""
import numpy as np
import pytest

from mlindex.utilities import FigureOfMerits as fom
from mlindex.utilities.FigureOfMerits import SIGMA_TREATMENT
from mlindex.utilities.UnitCellTools import get_hkl_matrix


# ---------------------------------------------------------------------------------------------
# Fold design
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('scheme', ['random', 'contiguous'])
def test_every_peak_is_held_out_exactly_once(scheme):
    folds = fom._cv_folds(20, 5, scheme, seed=1)
    held = np.concatenate(folds)
    assert sorted(held.tolist()) == list(range(20))
    assert len(folds) == 5


def test_high_q_holds_out_the_top_block_only():
    folds = fom._cv_folds(20, 5, 'high_q', seed=1)
    assert len(folds) == 1
    assert folds[0].tolist() == [16, 17, 18, 19]


def test_contiguous_folds_are_contiguous_and_random_folds_are_not():
    contiguous = fom._cv_folds(20, 5, 'contiguous', seed=1)
    assert all(np.all(np.diff(fold) == 1) for fold in contiguous)
    scattered = fom._cv_folds(20, 5, 'random', seed=1)
    assert not all(np.all(np.diff(fold) == 1) for fold in scattered)


def test_random_folds_are_deterministic_under_a_seed():
    first = fom._cv_folds(20, 5, 'random', seed=7)
    second = fom._cv_folds(20, 5, 'random', seed=7)
    other = fom._cv_folds(20, 5, 'random', seed=8)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_an_unknown_scheme_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match='unknown cv scheme'):
        fom._cv_folds(20, 5, 'leave-one-out', seed=1)


# ---------------------------------------------------------------------------------------------
# The null calibration. cv_M = ln(2)/median(r), so r ~ Exp(1) must give 1.
# ---------------------------------------------------------------------------------------------
def test_cv_M_is_one_under_the_idealised_null():
    rng = np.random.default_rng(0)
    ratio = rng.exponential(1.0, size=(4000, 20))
    features = fom._reduce_predictive(ratio, ratio, None, 'cv')
    assert abs(np.median(features['cv_M']) - 1.0) < 0.02


def test_cv_M_is_one_regardless_of_the_number_of_scored_peaks():
    """The whole point of dividing by Delta: cubic is scored on ten peaks, everything else on
    twenty (R5), and the merit must not shift because of that."""
    rng = np.random.default_rng(1)
    wide = fom._reduce_predictive(*(2*[rng.exponential(1.0, size=(6000, 20))]), None, 'cv')
    narrow = fom._reduce_predictive(*(2*[rng.exponential(1.0, size=(6000, 10))]), None, 'cv')
    assert abs(np.median(wide['cv_M']) - np.median(narrow['cv_M'])) < 0.05


def test_the_tail_statistic_is_a_sum_and_records_how_many_peaks_it_summed():
    """cv_tail_nll is Gamma(n_scored, 1) under the null, so n_scored has to travel with it or it
    cannot be turned into a -log p (FomNull.analytic_neg_log_p takes n_peaks)."""
    rng = np.random.default_rng(2)
    ratio = rng.exponential(1.0, size=(2000, 20))
    features = fom._reduce_predictive(ratio, ratio, None, 'cv')
    assert np.all(features['cv_n_scored'] == 20)
    assert abs(np.mean(features['cv_tail_nll']) - 20.0) < 0.5


def test_a_voided_row_scores_zero_rather_than_nan():
    ratio = np.full((3, 20), np.nan)
    ratio[0] = 1.0
    features = fom._reduce_predictive(ratio, ratio, None, 'cv')
    assert features['cv_n_scored'].tolist() == [20.0, 0.0, 0.0]
    assert np.all(np.isfinite(features['cv_M']))
    assert features['cv_M'][1] == 0.0


def test_the_chi2_column_appears_only_when_a_scale_is_supplied():
    ratio = np.full((5, 20), 0.5)
    assert 'cv_chi2' not in fom._reduce_predictive(ratio, ratio, None, 'cv')
    assert 'cv_chi2' in fom._reduce_predictive(ratio, ratio, 1e-4, 'cv')


# ---------------------------------------------------------------------------------------------
# The refit. q2 = hkl2 @ xnn is linear, so at fixed weights one step is the exact optimum.
# ---------------------------------------------------------------------------------------------
def _orthorhombic_case(n_candidates=6, n_peaks=20, seed=0):
    """Peaks generated exactly from a known orthorhombic cell, plus its Miller indices."""
    rng = np.random.default_rng(seed)
    hkl = rng.integers(0, 4, size=(n_peaks, 3))
    hkl[np.all(hkl == 0, axis=1)] = [1, 0, 0]
    xnn_true = np.array([0.03, 0.02, 0.011])
    hkl2 = get_hkl_matrix(hkl, 'orthorhombic')
    q2_obs = np.sort(hkl2 @ xnn_true)
    order = np.argsort(hkl2 @ xnn_true)
    hkl = hkl[order]
    hkl_stack = np.repeat(hkl[np.newaxis], n_candidates, axis=0)
    xnn = np.repeat(xnn_true[np.newaxis], n_candidates, axis=0)
    return q2_obs, hkl_stack, xnn


def test_the_refit_recovers_the_true_cell_from_noiseless_retained_peaks():
    q2_obs, hkl, xnn = _orthorhombic_case()
    hkl2 = get_hkl_matrix(hkl, 'orthorhombic')
    start = xnn*1.05
    refit, ok = fom._refit_on_retained(q2_obs, start, hkl2, np.arange(0, 16))
    assert np.all(ok)
    assert np.allclose(refit, xnn, rtol=1e-8)


def test_the_refit_does_not_depend_on_where_it_started():
    """The forward model is linear, so a single step at fixed weights is the exact weighted
    optimum. If this fails, the refit is being treated as an iterative optimisation and the merit
    inherits a dependence on the full fit it is meant to be independent of."""
    q2_obs, hkl, xnn = _orthorhombic_case(seed=3)
    hkl2 = get_hkl_matrix(hkl, 'orthorhombic')
    keep = np.arange(0, 16)
    near, _ = fom._refit_on_retained(q2_obs, xnn*1.01, hkl2, keep)
    far, _ = fom._refit_on_retained(q2_obs, xnn*1.40, hkl2, keep)
    assert np.allclose(near, far, rtol=1e-6)


def test_a_rank_deficient_retained_set_is_reported_not_silently_zero_stepped():
    """gauss_newton_solve returns a zero step for a singular system, which would make the refit
    equal to the full fit and every held-out peak predict perfectly. It must come back not-ok."""
    q2_obs, hkl, xnn = _orthorhombic_case()
    hkl2 = get_hkl_matrix(hkl, 'orthorhombic')
    hkl2[:, :, 2] = 0.0          # the c* column carries no information any more
    _, ok = fom._refit_on_retained(q2_obs, xnn, hkl2, np.arange(0, 16))
    assert not np.any(ok)


# ---------------------------------------------------------------------------------------------
# get_cv_fom end to end, on a reference list built from the same generator as the peaks
# ---------------------------------------------------------------------------------------------
def _reference_list(max_index=5):
    grid = np.stack(np.meshgrid(*[np.arange(0, max_index)]*3, indexing='ij'), axis=-1)
    hkl_ref = grid.reshape(-1, 3)
    return hkl_ref[np.any(hkl_ref != 0, axis=1)]


def _as_the_pipeline_would(q2_obs, xnn, hkl_ref, lattice_system):
    """Each candidate carries the assignment `fast_assign` gives it under its own cell.

    Handing every candidate the *true* Miller indices instead would be a different experiment
    entirely: the model is linear in xnn, so a wrong cell with the right assignment simply refits
    back to the right cell and predicts perfectly. A wrong candidate is wrong in its assignment as
    much as in its parameters, and that is what the benchmark's candidates look like.
    """
    from mlindex.utilities.numba_functions import fast_assign
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    q2_ref_calc = xnn @ hkl2_ref.T
    return np.take(hkl_ref, fast_assign(q2_obs, q2_ref_calc), axis=0)


def test_a_correct_cell_predicts_its_held_out_peaks_and_a_wrong_one_does_not():
    q2_obs, _, xnn = _orthorhombic_case(n_candidates=2, seed=5)
    q2_obs = q2_obs + np.random.default_rng(5).normal(0.0, 2e-5, size=q2_obs.shape)
    cells = xnn.copy()
    cells[1] = xnn[1]*np.array([1.7, 0.6, 1.3])
    hkl_ref = _reference_list()
    hkl = _as_the_pipeline_would(q2_obs, cells, hkl_ref, 'orthorhombic')
    features = fom.get_cv_fom(
        q2_obs, cells, hkl, hkl_ref, 'orthorhombic', 'oP', scheme='contiguous',
        )
    assert features['cv_M'][0] > features['cv_M'][1]
    assert features['cv_raw'][0] < features['cv_raw'][1]


def test_a_wrong_cell_handed_the_true_assignment_refits_back_to_the_true_cell():
    """The reason the test above has to assign each candidate under its own cell, asserted rather
    than left as a comment. This is the linear model working, not a defect -- but a test that fed
    the true assignment to every candidate would be measuring nothing."""
    q2_obs, hkl, xnn = _orthorhombic_case(n_candidates=1, seed=5)
    hkl2 = get_hkl_matrix(hkl, 'orthorhombic')
    refit, ok = fom._refit_on_retained(q2_obs, xnn*np.array([1.7, 0.6, 1.3]), hkl2, np.arange(16))
    assert np.all(ok)
    assert np.allclose(refit, xnn, rtol=1e-6)


def test_the_retained_assignment_is_frozen_rather_than_recomputed():
    """The single most important property. If the retained peaks were re-assigned under the refit
    cell, corrupting the supplied assignment would make no difference -- the function would just
    find the same nearest lines again. It must change the answer."""
    q2_obs, hkl, xnn = _orthorhombic_case(n_candidates=3, seed=6)
    corrupted = hkl.copy()
    corrupted[:, :12] = corrupted[:, :12][:, ::-1]
    honest = fom.get_cv_fom(
        q2_obs, xnn, hkl, _reference_list(), 'orthorhombic', 'oP', scheme='contiguous',
        )
    tampered = fom.get_cv_fom(
        q2_obs, xnn, corrupted, _reference_list(), 'orthorhombic', 'oP', scheme='contiguous',
        )
    assert not np.allclose(honest['cv_M'], tampered['cv_M'])


def test_get_cv_fom_does_not_modify_its_arguments():
    q2_obs, hkl, xnn = _orthorhombic_case(seed=7)
    hkl_ref = _reference_list()
    before = (q2_obs.copy(), hkl.copy(), xnn.copy(), hkl_ref.copy())
    fom.get_cv_fom(q2_obs, xnn, hkl, hkl_ref, 'orthorhombic', 'oP')
    for original, current in zip(before, (q2_obs, hkl, xnn, hkl_ref)):
        assert np.array_equal(original, current)


def test_a_fold_larger_than_the_data_can_support_is_voided_not_fitted():
    """Triclinic has six free parameters; ask it to refit on five retained peaks and there is no
    refit to be had. The peaks must be voided rather than scored off a zero step."""
    rng = np.random.default_rng(8)
    hkl = rng.integers(1, 4, size=(1, 8, 3))
    xnn = np.array([[0.03, 0.02, 0.011, 0.001, 0.002, 0.0015]])
    hkl2 = get_hkl_matrix(hkl, 'triclinic')
    q2_obs = np.sort((hkl2 @ xnn[0])[0])
    features = fom.get_cv_fom(
        q2_obs, xnn, hkl, _reference_list(), 'triclinic', 'aP', scheme='contiguous', n_folds=2,
        )
    assert features['cv_n_voided'][0] == q2_obs.size


def test_every_emitted_column_declares_a_sigma_treatment():
    q2_obs, hkl, xnn = _orthorhombic_case(seed=9)
    features = fom.get_cv_fom(
        q2_obs, xnn, hkl, _reference_list(), 'orthorhombic', 'oP', sigma_entrywise=1e-4,
        )
    features.update(fom.get_holdout_fom(
        q2_obs[:5], xnn, _reference_list(), 'orthorhombic', 'oP', sigma_entrywise=1e-4,
        ))
    for name in features:
        assert name in SIGMA_TREATMENT, name
    assert SIGMA_TREATMENT['cv_chi2'] == 'in-sample'
    assert SIGMA_TREATMENT['cv_M'] == 'free'


# ---------------------------------------------------------------------------------------------
# get_holdout_fom
# ---------------------------------------------------------------------------------------------
def test_the_holdout_form_needs_no_refit_and_rewards_the_cell_that_predicts():
    q2_obs, hkl, xnn = _orthorhombic_case(n_candidates=2, seed=11)
    hkl2 = get_hkl_matrix(hkl, 'orthorhombic')
    extra = np.sort((hkl2[0] @ xnn[0])[:5]) + 0.0
    wrong = xnn.copy()
    wrong[1] = xnn[1]*np.array([1.7, 0.6, 1.3])
    features = fom.get_holdout_fom(extra, wrong, _reference_list(), 'orthorhombic', 'oP')
    assert features['ho_M'][0] > features['ho_M'][1]
    assert features['ho_n_scored'].tolist() == [5.0, 5.0]


def test_the_holdout_form_does_not_modify_its_arguments():
    q2_obs, hkl, xnn = _orthorhombic_case(seed=12)
    hkl_ref = _reference_list()
    before = (q2_obs.copy(), xnn.copy(), hkl_ref.copy())
    fom.get_holdout_fom(q2_obs[:5], xnn, hkl_ref, 'orthorhombic', 'oP')
    for original, current in zip(before, (q2_obs, xnn, hkl_ref)):
        assert np.array_equal(original, current)


# ---------------------------------------------------------------------------------------------
# The in-sample partner, which is what makes the scaling claim measurable
# ---------------------------------------------------------------------------------------------
def test_the_in_sample_form_uses_the_given_assignment_without_refitting_or_reassigning():
    q2_obs, hkl, xnn = _orthorhombic_case(n_candidates=2, seed=13)
    features = fom.get_insample_fom(q2_obs, xnn, hkl, 'orthorhombic', 'oP')
    # The peaks were generated from exactly this cell and assignment, so every residual is zero
    # and the merit is the infinity a perfect fit earns.
    assert np.all(features['is_raw'] == 0.0)
    assert np.all(np.isinf(features['is_M']))
    assert features['is_n_scored'].tolist() == [20.0, 20.0]


def test_the_in_sample_form_beats_the_cross_validated_one_on_the_same_peaks():
    """The whole premise: a cell always fits the peaks it was fitted to better than the ones it
    was not. If this came back the other way the folds would not be holding anything out."""
    q2_obs, _, xnn = _orthorhombic_case(n_candidates=4, seed=14)
    q2_obs = q2_obs + np.random.default_rng(14).normal(0.0, 2e-5, size=q2_obs.shape)
    hkl_ref = _reference_list()
    cells = xnn*np.array([[1.0], [1.15], [0.8], [1.4]])
    hkl = _as_the_pipeline_would(q2_obs, cells, hkl_ref, 'orthorhombic')
    in_sample = fom.get_insample_fom(q2_obs, cells, hkl, 'orthorhombic', 'oP')
    cross = fom.get_cv_fom(q2_obs, cells, hkl, hkl_ref, 'orthorhombic', 'oP', scheme='contiguous')
    assert np.all(in_sample['is_M'] >= cross['cv_M'])


def test_the_in_sample_de_wolff_form_is_get_M20_on_the_same_inputs():
    """`is_M20` is de Wolff's arithmetic on the fitted peaks, so it must BE get_M20.

    This is what licenses reading `cv_M20` as "M20, moved out of sample": if the in-sample form
    were a near-miss of get_M20, the held-out form would be a near-miss of something nobody has a
    baseline for. The tolerance in `_dewolff_baseline` exists for this comparison -- the cut-off is
    itself a reference line, and reaching it two ways differs in the last bit.
    """
    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.FigureOfMerits import get_M20

    q2_obs, _, xnn = _orthorhombic_case(n_candidates=4, seed=15)
    q2_obs = q2_obs + np.random.default_rng(15).normal(0.0, 3e-5, size=q2_obs.shape)
    cells = xnn*np.array([[1.0], [1.15], [0.8], [1.4]])
    hkl_ref = _reference_list()
    hkl2_ref = get_hkl_matrix(hkl_ref, 'orthorhombic')
    q2_ref_calc = cells @ hkl2_ref.T
    assign = fast_assign(q2_obs, q2_ref_calc)
    hkl = np.take(hkl_ref, assign, axis=0)
    q2_calc = np.take_along_axis(q2_ref_calc, assign, axis=1)

    expected = get_M20(q2_obs, q2_calc, q2_ref_calc.copy())
    measured = fom.get_insample_fom(
        q2_obs, cells, hkl, 'orthorhombic', 'oP', q2_calc=q2_calc, q2_ref_calc=q2_ref_calc,
        )['is_M20']
    assert np.array_equal(measured, expected)


def test_is_M20_is_absent_unless_the_caller_supplies_the_reference_lines():
    """Silence rather than a rederived near-miss: the column exists to be compared against the
    stored M20, and a version computed from the Miller indices alone is not that number."""
    q2_obs, hkl, xnn = _orthorhombic_case(seed=16)
    assert 'is_M20' not in fom.get_insample_fom(q2_obs, xnn, hkl, 'orthorhombic', 'oP')
