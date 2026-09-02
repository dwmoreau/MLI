"""S12's driver: the arm ladder, the two-pool split, and the controls.

The driver's job is to keep three things straight that are easy to confuse and expensive to get
wrong: which pool a number is fitted on and which it is reported on, which columns each arm
ablates, and what a control destroys. None of that needs a pool to test.
"""

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomCombiner
from mlindex.scripts import run_fom_combiner as driver


# ---------------------------------------------------------------------------------------------
# the arm ladder
# ---------------------------------------------------------------------------------------------
def test_every_arm_has_a_unique_name_and_a_stated_purpose():
    names = [name for name, _, _, _ in driver.ARMS] + [name for name, _ in driver.CONTROL_ARMS]
    assert len(names) == len(set(names))
    for name, _, _, purpose in driver.ARMS:
        assert purpose and len(purpose) > 20, f'{name} has no stated purpose'


def test_every_arm_actually_ablates_what_its_name_says():
    """An arm that drops a column not in its own feature set ablates nothing and reports a null.

    `feature_specification` raises on an unknown drop, so this is really a check that every arm is
    *constructible* -- but the failure it prevents is the quiet one: a mistyped column name in a
    drop tuple would otherwise produce an arm indistinguishable from `base` and a contrast of zero
    that reads as evidence.
    """
    for name, extra, drop, _ in driver.ARMS:
        groups = driver.arm_groups(extra)
        names, _ = FomCombiner.feature_specification(groups, drop=drop)
        assert names, f'{name} has an empty feature set'
        for column in drop:
            assert column not in names


def test_the_base_arm_is_smaller_than_the_space_and_every_drop_is_recoverable():
    """Each dropped column returns in some arm, so no cut is made by fiat.

    `q2_max`, `n_peaks` and `hkl_ref_length` come back in `plus_pool_structural`; `spacegroup` in
    `plus_spacegroup`; the skewed context statistics in `plus_ctx_rank_z`. A column dropped in the
    base arm and restored nowhere would be a feature cut with no retrained arm behind it, which is
    what PROTOCOL section 8 forbids.
    """
    base, _ = FomCombiner.feature_specification(driver.arm_groups(()), drop=driver.BASE_DROP)
    space, _ = FomCombiner.feature_specification(driver.arm_groups(()))
    assert len(base) < len(space)
    restored = set()
    for _, extra, drop, _ in driver.ARMS:
        names, _ = FomCombiner.feature_specification(driver.arm_groups(extra), drop=drop)
        restored |= set(names)
    for column in driver.BASE_DROP:
        assert column in restored, f'{column} is dropped in the base arm and restored in no arm'


def test_the_headline_and_the_symmetry_drop_arm_differ_by_the_counts_and_nothing_else():
    """DWMM's standing instruction is that the without-symmetry arm rides beside every headline,
    so the pair has to differ in the symmetry features alone or the comparison means nothing."""
    base, _ = FomCombiner.feature_specification(driver.arm_groups(()), drop=driver.BASE_DROP)
    without, _ = FomCombiner.feature_specification(
        driver.arm_groups(()), drop=driver.BASE_DROP + FomCombiner.SYMMETRY_COUNTS)
    assert set(base) - set(without) == set(FomCombiner.SYMMETRY_COUNTS)


def test_no_arm_can_reach_a_truth_column_or_the_retention_rule():
    """`check_no_leakage` runs inside `feature_specification`, so constructing every arm is the
    check. `sampling_weight` is the one worth naming: it has to be in the frame and must never be
    in the matrix, and it is 1.0 for every correct candidate."""
    for name, extra, drop, _ in driver.ARMS:
        names, _ = FomCombiner.feature_specification(driver.arm_groups(extra), drop=drop)
        for forbidden in ('is_correct', 'sampling_weight', 'fit_weight', 'retained_reason',
                          'ctx_pool_size', 'm20_at_prune', 'in_top_n'):
            assert forbidden not in names, f'{name} would fit on {forbidden}'


def test_the_context_statistics_the_thinning_distorts_are_out_of_the_base_arm():
    """Measured, not assumed: `gap_to_best` is invariant to the retention rule because the pool
    maximum is always retained, while `rank` and `z` are not. `--stage skew` is what measures it."""
    assert set(driver.CONTEXT_SKEWED) <= set(driver.BASE_DROP)
    base, _ = FomCombiner.feature_specification(driver.arm_groups(()), drop=driver.BASE_DROP)
    assert any(name.endswith('_gap_to_best') for name in base)
    assert not any(name.endswith('_rank') or name.endswith('_z') for name in base
                   if name.startswith('ctx_'))


# ---------------------------------------------------------------------------------------------
# the two pools
# ---------------------------------------------------------------------------------------------
def test_split_ids_divides_by_source_crystal_and_is_reproducible():
    entries = pd.DataFrame({
        'entry_id': [f'E{index:02d}' for index in range(20) for _ in range(3)],
        'condition_bundle': ['b1', 'b2', 'b3']*20,
        'split': ['fom-train']*60,
        })
    fit, cal = driver.split_ids(entries, 'fom-train', 0.2, 12345)
    assert not fit & cal
    assert len(fit) + len(cal) == 20
    assert len(cal) == 4
    assert (fit, cal) == driver.split_ids(entries, 'fom-train', 0.2, 12345)


def test_assert_disjoint_refuses_an_overlap_rather_than_warning():
    assert driver.assert_disjoint({'A', 'B'}, {'C'}) is True
    with pytest.raises(SystemExit, match='BOTH'):
        driver.assert_disjoint({'A', 'B'}, {'B', 'C'})


# ---------------------------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------------------------
def test_label_shuffling_preserves_the_per_entry_positive_count_and_moves_the_labels():
    """Within the group, not across it. A global shuffle would also flatten the per-entry base
    rate, and the control would then be measuring two things at once."""
    frame = pd.DataFrame({
        'entry_id': ['A']*6 + ['B']*6,
        'condition_bundle': ['b']*12,
        'is_correct': [True, False, False, False, False, False,
                       True, True, False, False, False, False],
        'M20': np.arange(12.0),
        })
    shuffled = driver._shuffle_labels(frame, 7)
    pd.testing.assert_series_equal(shuffled['M20'], frame['M20'])
    for entry in ('A', 'B'):
        assert (shuffled.loc[shuffled.entry_id == entry, 'is_correct'].sum()
                == frame.loc[frame.entry_id == entry, 'is_correct'].sum())
    assert not shuffled['is_correct'].equals(frame['is_correct'])


def test_the_two_controls_are_declared_and_differ_in_what_they_shuffle():
    names = dict(driver.CONTROL_ARMS)
    assert set(names) == {'label_shuffled', 'prior_only'}
    assert 'calibration' in names['label_shuffled']
    assert 'prior' in names['prior_only']


# ---------------------------------------------------------------------------------------------
# the guard that a rank claim depends on
# ---------------------------------------------------------------------------------------------
def test_the_report_pool_is_read_as_fully_retained_and_the_fit_pool_is_not():
    """`_pool_depth` is what stops `reduce_many` being handed 'auto' on an iterable of frames,
    where it would assume a full pool and certify a rank it cannot answer."""
    if not driver.REPORT_POOL.exists() or not driver.FIT_POOL.exists():
        pytest.skip('the pools are not on this machine')
    assert driver._pool_depth(driver.REPORT_POOL) is None
    assert driver._pool_depth(driver.FIT_POOL) == 200


# ---------------------------------------------------------------------------------------------
# the two thinnings, which are not the same kind of thing (C2-F-127)
# ---------------------------------------------------------------------------------------------
def test_the_fit_is_weighted_by_sampling_weight_and_not_by_the_composed_weight():
    """The distinction cost 17.7 pp of top-10 to discover, so it is pinned rather than commented.

    The generator's thinning is a bias -- it kept the highest-scoring wrong candidates
    preferentially -- and `sampling_weight` corrects it. The driver's own negative subsampling is a
    deliberate rebalancing from 0.026 % correct to about 1.7 %, and weighting it back restores
    0.026 % and undoes the only reason the subsample exists.
    """
    import inspect
    signature = inspect.signature(driver.fit_one)
    assert signature.parameters['weight_column'].default == 'sampling_weight'
    source = inspect.getsource(driver.run_fit)
    assert "'fit_weight'" not in source, (
        'run_fit passes fit_weight to a fit again. That is C2-F-127: it undoes the rebalancing '
        'the negative subsample exists to create, and it took the model below raw M20 on top-1.')
    assert "'sampling_weight'" in source


def test_subsample_negatives_writes_the_composed_weight_and_is_unbiased():
    """`fit_weight` is still written and is still correct -- for a pool-level estimator, which is
    not what a fit is. Unbiasedness is the property that makes it worth keeping at all."""
    # Seeded well away from the subsample seeds below, and that is not fussiness. `numpy`'s
    # generator is deterministic in its seed, so building the weights with `default_rng(0)` and
    # then subsampling with seed 0 draws the SAME uniform stream -- and since the subsample keeps
    # the smallest draws while `choice([1, 20], p=[0.3, 0.7])` assigns 1.0 to the smallest draws,
    # every kept negative comes out with weight 1.0 and the estimator collapses. Harmless here
    # because the real generator indexes its draw by `candidate_id`, but it is exactly the kind of
    # coincidence that reads as a bug.
    rng = np.random.default_rng(20260901)
    n_groups, per_group = 40, 300
    frame = pd.DataFrame({
        'entry_id': np.repeat([f'E{index}' for index in range(n_groups)], per_group),
        'condition_bundle': 'b',
        'is_correct': np.tile([True] + [False]*(per_group - 1), n_groups),
        # The generator's own two weights, as the pool carries them.
        'sampling_weight': rng.choice([1.0, 20.0], size=n_groups*per_group, p=[0.3, 0.7]),
        })
    frame.loc[frame['is_correct'], 'sampling_weight'] = 1.0
    truth = frame.loc[~frame['is_correct'], 'sampling_weight'].sum()
    estimates = []
    for seed in range(30):
        kept = FomCombiner.subsample_negatives(frame, 40, seed)
        assert 'fit_weight' in kept.columns
        negatives = ~FomCombiner.FomMetrics.as_bool(kept['is_correct'])
        # Positives are kept whole, so they carry the generator's weight unchanged.
        assert (kept.loc[~negatives, 'fit_weight'] == 1.0).all()
        estimates.append(kept.loc[negatives, 'fit_weight'].sum())
    assert abs(np.mean(estimates)/truth - 1) < 0.02, (
        f'the composed weight is biased: mean {np.mean(estimates):.0f} against {truth:.0f}')


def test_the_two_weights_imply_different_positive_rates_which_is_the_whole_point():
    """One number explains C2-F-127: what share of the fit's weight sits on correct candidates."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        'entry_id': np.repeat([f'E{index}' for index in range(20)], 500),
        'condition_bundle': 'b',
        'is_correct': np.tile([True] + [False]*499, 20),
        'sampling_weight': rng.choice([1.0, 20.0], size=10_000, p=[0.3, 0.7]),
        })
    frame.loc[frame['is_correct'], 'sampling_weight'] = 1.0
    kept = FomCombiner.subsample_negatives(frame, 40, 12345)
    correct = FomCombiner.FomMetrics.as_bool(kept['is_correct'])
    shares = {column: kept.loc[correct, column].sum()/kept[column].sum()
              for column in ('sampling_weight', 'fit_weight')}
    unweighted = correct.mean()
    assert shares['fit_weight'] < shares['sampling_weight'] < unweighted, (
        f'expected fit_weight to push the positive share back down towards the pool rate; '
        f'got {shares} against an unweighted {unweighted:.4f}')
