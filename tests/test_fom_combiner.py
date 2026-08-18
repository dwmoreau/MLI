"""S08's learned combiner: the feature contract, the leakage guards, and the round trip.

The tests that matter here are not the arithmetic ones. This task has two ways to produce a large
number that means nothing -- a feature derived from the truth, and a feature that is a property of
the synthetic generator rather than of the pattern -- and neither shows up as a failure anywhere
else in the pipeline. Both are asserted directly.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training import FomCombiner  # noqa: E402
from mlindex.model_training import FomMetrics  # noqa: E402


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------
def _pool(n_entries=40, n_candidates=30, seed=7):
    """A synthetic pool with the columns the combiner reads and a learnable signal in `M_sym`."""
    rng = np.random.default_rng(seed)
    lattices = np.array(['cP', 'mP', 'aP', 'oP'])
    rows = []
    for entry in range(n_entries):
        lattice = lattices[entry % lattices.size]
        correct = rng.integers(0, n_candidates)
        for candidate in range(n_candidates):
            is_correct = candidate == correct
            rows.append(dict(
                entry_id=f'E{entry:04d}',
                condition_bundle='error1_cont0',
                bravais_lattice=lattice,
                candidate_id=candidate,
                spacegroup=f'sg{candidate % 3}',
                is_correct=is_correct,
                is_off_by_two=False,
                is_degenerate=pd.NA,
                in_top_n=True,
                n_peaks=10 if lattice == 'cP' else 20,
                n_indexed=int(rng.integers(5, 20)),
                hkl_ref_length=300,
                n_entering=500,
                final_rank=candidate,
                volume=float(rng.uniform(200, 3000)),
                q2_max_cubic=0.04,
                q2_max_full=0.10,
                n_peaks_available=30,
                ))
            for merit in FomCombiner.RAW_MERITS + FomCombiner.IN_SAMPLE_MERITS:
                rows[-1][merit] = float(rng.normal(3.0 + 4.0*is_correct, 1.0))
            for column in FomCombiner.FEATURE_MATRIX_STRUCTURAL:
                rows[-1][column] = float(rng.uniform(1.0, 50.0))
    frame = pd.DataFrame(rows)
    frame['log_volume'] = np.log(frame['volume'])
    frame['q2_max'] = np.where(frame['n_peaks'] <= 10, frame['q2_max_cubic'],
                               frame['q2_max_full'])
    return FomCombiner.add_context(frame)


@pytest.fixture(scope='module')
def pool():
    return _pool()


# ---------------------------------------------------------------------------------------
# The leakage guards -- the whole reason this file exists
# ---------------------------------------------------------------------------------------
def test_no_feature_group_contains_a_truth_derived_column():
    """PLAN section 6.5's labels, and the two strata METRICS.md defines from the *true* cell."""
    names, _ = FomCombiner.feature_specification(FomCombiner.FEATURE_GROUPS, ())
    for forbidden in ('is_correct', 'is_off_by_two', 'xnn_distance_to_truth',
                      'volume_ratio_to_truth', 'is_degenerate', 'dominant_zone',
                      'zone_count_min'):
        assert forbidden not in names
    assert not [name for name in names if name.endswith('_true')]


def test_no_feature_is_a_property_of_the_synthetic_generator():
    """The condition and its parameters are strata, not features: at inference nobody knows them.

    Easier to add by accident than the truth columns, because `condition_bundle` is on every
    frame the loader produces and reads like an ordinary categorical.
    """
    names, _ = FomCombiner.feature_specification(FomCombiner.FEATURE_GROUPS, ())
    for forbidden in ('condition_bundle', 'q2_error_multiplier', 'n_contaminants', 'n_dropout',
                      'n_dropout_achieved', 'second_phase_lines', 'second_phase_partner',
                      'split'):
        assert forbidden not in names


def test_check_no_leakage_rejects_a_truth_column_however_it_arrives():
    with pytest.raises(ValueError, match='truth-derived'):
        FomCombiner.check_no_leakage(['M20', 'volume_ratio_to_truth'])
    with pytest.raises(ValueError, match='truth-derived'):
        FomCombiner.check_no_leakage(['M20', 'spacegroup_true'])
    with pytest.raises(ValueError, match='truth-derived'):
        FomCombiner.check_no_leakage(['M20', 'condition_bundle'])


def test_the_two_excluded_merits_are_excluded_and_the_reason_is_recorded():
    """`M_nn` is M20 (S01's s = 1 identity) and `chi2_fixed` is the generator's own sigma."""
    names, _ = FomCombiner.feature_specification(FomCombiner.FEATURE_GROUPS, ())
    assert 'M_nn' not in names and 'chi2_fixed' not in names
    assert set(FomCombiner.EXCLUDED_MERITS) >= {'M_nn', 'chi2_fixed'}
    assert all(FomCombiner.EXCLUDED_MERITS.values())


def test_in_sample_sigma_merits_are_their_own_group():
    """PROTOCOL section 3 rule 4: reportable with and without, never silently in the headline."""
    default, _ = FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS, ())
    widened, _ = FomCombiner.feature_specification(
        FomCombiner.DEFAULT_GROUPS + ('in_sample',), ())
    assert not set(FomCombiner.IN_SAMPLE_MERITS) & set(default)
    assert set(FomCombiner.IN_SAMPLE_MERITS) <= set(widened)


# ---------------------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------------------
def test_feature_groups_are_droppable_as_a_unit():
    names, _ = FomCombiner.feature_specification(('raw', 'structural'), ())
    assert set(FomCombiner.RAW_MERITS) <= set(names)
    assert not [name for name in names if name.startswith('ctx_')]
    assert not [name for name in names if '__' in name]


def test_an_unknown_feature_group_raises_rather_than_being_ignored():
    with pytest.raises(ValueError, match='unknown feature group'):
        FomCombiner.feature_specification(('raw', 'process'), ())


def test_context_features_are_invariant_to_row_order(pool):
    """A feature that depends on the order rows happen to arrive in is not a feature."""
    shuffled = pool.sample(frac=1.0, random_state=3).reset_index(drop=True)
    recomputed = FomCombiner.add_context(shuffled.drop(columns=list(FomCombiner.context_names())))
    key = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']
    left = pool.set_index(key).sort_index()
    right = recomputed.set_index(key).sort_index()
    for name in FomCombiner.context_names():
        np.testing.assert_allclose(left[name].to_numpy(), right[name].to_numpy())


def test_context_rank_matches_the_ranking_the_metrics_module_would_report(pool):
    """`ctx_M20_rank` has to be M20's own cross-lattice rank, not a second, similar ordering."""
    reduced = FomMetrics.reduce_pool(pool, pool['M20'].to_numpy(), pool='cross_bl')
    best = reduced.set_index('entry_id')['rank_best_correct_all']
    correct = pool.loc[FomMetrics.as_bool(pool['is_correct'])]
    observed = correct.set_index('entry_id')['ctx_M20_rank']
    np.testing.assert_array_equal(observed.loc[best.index].to_numpy(), best.to_numpy())


def test_context_is_computed_before_subsampling_not_after(pool):
    """Thinning the negatives must not move a surviving candidate's rank among its competitors."""
    thinned = FomCombiner.subsample_negatives(pool, n_negatives=5, seed=1)
    recomputed = FomCombiner.add_context(thinned.drop(columns=list(FomCombiner.context_names())))
    assert thinned['ctx_M20_rank'].max() > recomputed['ctx_M20_rank'].max()
    assert thinned['ctx_pool_size'].iloc[0] == pool['ctx_pool_size'].iloc[0]


def test_subsampling_keeps_every_positive(pool):
    thinned = FomCombiner.subsample_negatives(pool, n_negatives=3, seed=1)
    assert int(FomMetrics.as_bool(thinned['is_correct']).sum()) == \
        int(FomMetrics.as_bool(pool['is_correct']).sum())
    per_entry = thinned.groupby('entry_id')['is_correct'].agg(['size', 'sum'])
    assert (per_entry['size'] - per_entry['sum'] <= 3).all()


def test_affordable_features_drops_every_column_that_depends_on_an_expensive_merit():
    """Gate condition 3's variant: `M_sym` costs 24 get_M20-equivalents, its scaled forms too."""
    names, _ = FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS, ())
    cheap = FomCombiner.affordable_features(names, ('M20', 'null_tail_nll'))
    assert 'M_sym' not in cheap and 'M_sym__z' not in cheap and 'ctx_M_sym_rank' not in cheap
    assert 'M20' in cheap and 'ctx_M20_rank' in cheap and 'ctx_pool_size' in cheap
    assert set(FomCombiner.STRUCTURAL_NUMERIC) <= set(cheap)


# ---------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------
def test_fit_score_and_round_trip_are_exact(tmp_path, pool):
    groups = ('raw', 'structural', 'context')
    combiner = FomCombiner.FomCombiner.fit([pool], groups=groups, max_iter=25)
    combiner.fit_calibrators([pool], minimum=50)
    before = combiner.score(pool)

    combiner.save(tmp_path/'model')
    reloaded = FomCombiner.FomCombiner.load(tmp_path/'model')
    assert reloaded.names == combiner.names
    np.testing.assert_array_equal(reloaded.score(pool), before)

    with open(tmp_path/'model'/'specification.json', encoding='utf-8') as handle:
        specification = json.load(handle)
    FomCombiner.check_no_leakage(specification['names'])


def test_predict_batch_matches_the_frame_path(pool):
    combiner = FomCombiner.FomCombiner.fit([pool], groups=('raw', 'context'), max_iter=25)
    np.testing.assert_array_equal(combiner.predict_batch(combiner.design_matrix(pool)),
                                  combiner.raw_score(pool))


def test_scoring_a_frame_with_a_missing_feature_raises_rather_than_guessing(pool):
    combiner = FomCombiner.FomCombiner.fit([pool], groups=('raw', 'context'), max_iter=25)
    with pytest.raises(KeyError, match='missing'):
        combiner.raw_score(pool.drop(columns=['M_sym']))


def test_an_unseen_category_gets_its_own_code_rather_than_the_first_one(pool):
    """An extinction group the model has never met must not be silently read as another one."""
    combiner = FomCombiner.FomCombiner.fit([pool], groups=('raw', 'structural'), max_iter=25)
    index = combiner.names.index('spacegroup')
    novel = pool.assign(spacegroup='sg_never_seen')
    codes = combiner.design_matrix(novel)[:, index]
    assert np.all(codes == FomCombiner._UNSEEN_CODE)
    assert FomCombiner._UNSEEN_CODE not in set(combiner.categories['spacegroup'].values())


def test_calibrated_scores_are_probabilities_and_are_monotone_within_a_lattice(pool):
    """Isotonic cannot reorder inside a lattice; everything it changes is cross-lattice.

    That is the same invariant `FomNull` asserts, and for the same reason: if a calibration step
    reorders within a lattice it is doing something other than calibrating.
    """
    combiner = FomCombiner.FomCombiner.fit([pool], groups=('raw', 'context'), max_iter=25)
    combiner.fit_calibrators([pool], minimum=50)
    raw = combiner.raw_score(pool)
    calibrated = combiner.score(pool)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0
    for lattice in pool['bravais_lattice'].unique():
        mask = (pool['bravais_lattice'] == lattice).to_numpy()
        order = np.argsort(raw[mask], kind='stable')
        assert np.all(np.diff(calibrated[mask][order]) >= -1e-12)


def test_a_saved_model_loads_without_lightgbm(tmp_path, pool, monkeypatch):
    """The optional training-only dependency must not reach the inference path."""
    combiner = FomCombiner.FomCombiner.fit([pool], groups=('raw',), max_iter=10)
    combiner.save(tmp_path/'model')
    monkeypatch.setitem(sys.modules, 'lightgbm', None)
    reloaded = FomCombiner.FomCombiner.load(tmp_path/'model')
    assert np.isfinite(reloaded.raw_score(pool)).all()


def test_the_ranking_objective_is_refused_clearly_when_lightgbm_is_absent(pool, monkeypatch):
    monkeypatch.setitem(sys.modules, 'lightgbm', None)
    with pytest.raises((ImportError, TypeError, AttributeError)):
        FomCombiner.FomCombiner.fit([pool], groups=('raw',), objective='lambdarank')


def test_an_unknown_objective_raises(pool):
    with pytest.raises(ValueError, match='pointwise'):
        FomCombiner.FomCombiner.fit([pool], groups=('raw',), objective='listwise')


# ---------------------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------------------
def test_a_threshold_cannot_be_reported_on_the_entries_it_was_chosen_on(pool):
    """PROTOCOL section 8, enforced by the metrics module rather than by remembering."""
    entries = pd.DataFrame({
        'entry_id': sorted(pool['entry_id'].unique()),
        'condition_bundle': 'error1_cont0',
        'split': 'fom-train',
        'bravais_lattice_true': [pool.loc[pool['entry_id'] == entry, 'bravais_lattice'].iloc[0]
                                 for entry in sorted(pool['entry_id'].unique())],
        'lattice_system_true': 'triclinic',
        'volume_true': 1000.0,
        'unit_cell_true': [[10.0, 11.0, 12.0, 1.5, 1.5, 1.5]]*pool['entry_id'].nunique(),
        })
    result = FomMetrics.evaluate([pool], score='M20', entries=entries, weights=None,
                                 n_bootstrap=0, strata=('bravais_lattice',))
    choice = FomMetrics.select_threshold(result, objective='youden')
    with pytest.raises(ValueError, match='PROTOCOL'):
        FomMetrics.check_threshold_transfer(choice, result)


# ---------------------------------------------------------------------------------------
# The distilled form (STATUS Q4)
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def teacher(pool):
    combiner = FomCombiner.FomCombiner.fit([pool], groups=('raw', 'structural', 'context'),
                                           max_iter=30)
    return combiner.fit_calibrators([pool], minimum=20)


def test_the_student_reads_exactly_the_teachers_columns(pool, teacher):
    """A distilled model that assembled its features differently would measure something else."""
    student = FomCombiner.DistilledCombiner.distil(teacher, [pool], hidden=(8, 4), max_iter=30,
                                                   sample=None)
    assert student.names == teacher.names
    assert student.categorical == teacher.categorical
    np.testing.assert_array_equal(student.design_matrix(pool), teacher.design_matrix(pool))


def test_the_student_is_three_matmuls_and_nothing_else(pool, teacher):
    """`predict_batch` must be the forward pass written out, or the timing measures the wrong thing."""
    student = FomCombiner.DistilledCombiner.distil(teacher, [pool], hidden=(8, 4), max_iter=30,
                                                   sample=None)
    matrix = student.design_matrix(pool)
    activations = (matrix - student.centre)/student.scale
    for index, (weight, bias) in enumerate(zip(student.weights, student.biases)):
        activations = activations@weight + bias
        if index < len(student.weights) - 1:
            activations = np.maximum(activations, 0.0)
    np.testing.assert_allclose(student.predict_batch(matrix), activations.ravel())


def test_the_student_imputes_rather_than_propagating_nan(pool, teacher):
    """Trees take NaN natively and an MLP does not, so the gap has to be closed explicitly."""
    student = FomCombiner.DistilledCombiner.distil(teacher, [pool], hidden=(8, 4), max_iter=30,
                                                   sample=None)
    damaged = pool.copy()
    damaged.loc[damaged.index[:20], 'M20'] = np.nan
    damaged.loc[damaged.index[20:40], 'M_sym'] = np.inf
    assert np.isfinite(student.raw_score(damaged)).all()


def test_the_student_round_trips_without_pickling_anything(tmp_path, pool, teacher):
    student = FomCombiner.DistilledCombiner.distil(teacher, [pool], hidden=(8, 4), max_iter=30,
                                                   sample=None)
    student.fit_calibrators([pool], minimum=20)
    before = student.score(pool)
    student.save(tmp_path/'student')
    reloaded = FomCombiner.DistilledCombiner.load(tmp_path/'student')
    assert set(reloaded.calibrators) == set(student.calibrators)
    np.testing.assert_array_equal(reloaded.score(pool), before)
    arrays = np.load(tmp_path/'student'/'distilled.npz')
    assert all(arrays[key].dtype != object for key in arrays.files)


def test_the_student_is_a_probability_only_once_it_is_calibrated(pool, teacher):
    """A regression on the teacher's output reproduces the ordering, not the scale.

    Skipping the student's own isotonic left it with no threshold that met a false-positive
    budget: its operating point measured exactly 0.0000 against a top-10 of 0.65 (F-092). So the
    student calibrates like the teacher, and the uncalibrated form is deliberately *not* silently
    clipped into looking like a probability.
    """
    student = FomCombiner.DistilledCombiner.distil(teacher, [pool], hidden=(8, 4), max_iter=30,
                                                   sample=None)
    assert not student.calibrators
    student.fit_calibrators([pool], minimum=20)
    probability = student.score(pool)
    assert probability.min() >= 0.0 and probability.max() <= 1.0
    # Calibration is monotone within a lattice, so it cannot reorder there -- the same invariant
    # the teacher is held to.
    raw = student.raw_score(pool)
    for lattice in pool['bravais_lattice'].unique():
        mask = (pool['bravais_lattice'] == lattice).to_numpy()
        order = np.argsort(raw[mask], kind='stable')
        assert np.all(np.diff(probability[mask][order]) >= -1e-12)


# ---------------------------------------------------------------------------------------
# The per-lattice architecture (the ablation that justifies the global choice)
# ---------------------------------------------------------------------------------------
def test_per_lattice_models_are_fitted_only_on_their_own_lattice(pool, teacher):
    """Each sub-model must be blind to the cross-lattice prior -- that is what it tests."""
    per_bl = FomCombiner.PerLatticeCombiner.fit(
        [pool], fallback=teacher, groups=('raw', 'context'), max_iter=20, min_positive=5)
    assert set(per_bl.models) == set(pool['bravais_lattice'].unique())
    for lattice, model in per_bl.models.items():
        rows = int(model.meta['n_rows'])
        assert rows == int((pool['bravais_lattice'] == lattice).sum())


def test_a_lattice_too_thin_to_fit_falls_back_and_says_so(pool, teacher):
    """oF has two entries in the whole CNRS benchmark; silence there would be the wrong answer."""
    per_bl = FomCombiner.PerLatticeCombiner.fit(
        [pool], fallback=teacher, groups=('raw',), max_iter=20, min_positive=10**6)
    assert not per_bl.models
    assert set(per_bl.meta['fell_back']) == set(pool['bravais_lattice'].unique())
    np.testing.assert_array_equal(per_bl.score(pool), teacher.score(pool))


def test_per_lattice_scores_dispatch_to_the_right_sub_model(pool, teacher):
    per_bl = FomCombiner.PerLatticeCombiner.fit(
        [pool], fallback=teacher, groups=('raw', 'context'), max_iter=20, min_positive=5)
    per_bl.fit_calibrators([pool], minimum=20)
    combined = per_bl.score(pool)
    for lattice, model in per_bl.models.items():
        mask = (pool['bravais_lattice'] == lattice).to_numpy()
        np.testing.assert_array_equal(combined[mask], model.score(pool.loc[mask]))
    assert np.isfinite(combined).all()


def test_dropping_a_merit_family_removes_its_scaled_and_context_columns(pool):
    """The over-prediction ablation has to take the whole family, not just the raw columns."""
    names, _ = FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS, ())
    keep = (set(FomCombiner.RAW_MERITS) | set(FomCombiner.IN_SAMPLE_MERITS)
            - {'n_over', 'max_gap', 'M_rev'})
    kept = FomCombiner.affordable_features(names, keep - {'n_over', 'max_gap', 'M_rev'})
    for merit in ('n_over', 'max_gap', 'M_rev'):
        assert merit not in kept
        assert not [name for name in kept if name.startswith(f'{merit}__')]
        assert not [name for name in kept if name.startswith(f'ctx_{merit}_')]
    # M_sym survives, and it is M_tilde * M_rev -- so the family is not fully separable.
    assert 'M_sym' in kept
