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


# ---------------------------------------------------------------------------------------------
# the two-machine split: build the design matrix where the pool is, fit where the report pool is
# ---------------------------------------------------------------------------------------------
def test_export_fit_names_its_calibration_sibling_and_refuses_a_lone_frame():
    """The calibrator must be fitted on rows the model was not fitted on, so the pair travels
    together. A `--fit-frame` with no `_cal_frame` beside it would otherwise calibrate on the fit
    rows and report an expected calibration error that means nothing."""
    import inspect
    source = inspect.getsource(driver.run_fit)
    assert "_cal_frame" in source
    assert 'SystemExit' in source


def test_the_fit_and_the_calibrator_use_different_weights_on_purpose():
    """C2-F-127's asymmetry, and it is the whole reason `fit_weight` is still written.

    The fit must not undo its own negative subsampling -- that rebalancing is the only reason a
    tree can learn from a 0.03 % base rate. The calibrator must undo it, because stating the prior
    the rebalancing removed is its entire job. `subsample_negatives` makes the two weights equal
    when nothing was thinned, so this is a no-op on an unsubsampled calibration split.
    """
    import inspect
    source = inspect.getsource(driver.fit_one)
    assert "fit_calibrators" in source and "'fit_weight'" in source
    assert inspect.signature(driver.fit_one).parameters['weight_column'].default == \
        'sampling_weight'


def test_the_hard_stratum_draw_takes_every_entry_rather_than_a_sample():
    """360 crystals is the whole hard stratum of `fom-dev` and generating all of them fully
    retained is under half a node-hour, so sampling would cost the claim and buy nothing."""
    from mlindex.scripts import run_fom_floor_entries as entries_script
    manifest = pd.DataFrame({
        'identifier': [f'X{index:03d}' for index in range(300)],
        'split': ['fom-dev']*150 + ['fom-train']*150,
        'bravais_lattice': ['mP', 'mC', 'aP', 'oP', 'cF']*60,
        # The decile advances once per block of five, so every lattice meets every decile. A
        # cycle of ten against a cycle of five stays in lockstep and puts the hard lattices only
        # at low deciles, which made an earlier version of this fixture describe an empty stratum.
        'volume_decile': [(index//5) % 10 for index in range(300)],
        })
    hard = entries_script.draw_hard_stratum(manifest, 'fom-dev')
    assert set(hard['bravais_lattice']) <= set(entries_script.HARD_LATTICES)
    assert (hard['volume_decile'] >= entries_script.HARD_MIN_DECILE).all()
    assert (hard['split'] == 'fom-dev').all()
    # Every qualifying entry, not a draw from them.
    expected = manifest[(manifest.split == 'fom-dev')
                        & manifest.bravais_lattice.isin(entries_script.HARD_LATTICES)
                        & (manifest.volume_decile >= entries_script.HARD_MIN_DECILE)]
    assert len(hard) == len(expected)


def test_the_hard_predicate_matches_the_metrics_modules_own():
    """A second copy of the definition is how the two drift, and S08 already had to refuse one
    redefinition of this stratum (C2-F-078)."""
    from mlindex.model_training import FomMetrics as metrics
    from mlindex.scripts import run_fom_floor_entries as entries_script
    assert set(entries_script.HARD_LATTICES) == set(metrics.HARD_LATTICES)
    assert entries_script.HARD_MIN_DECILE == metrics.HARD_MIN_DECILE


def test_every_artifact_path_carries_the_suffix_that_namespaces_a_run(tmp_path):
    """`--suffix` exists so a second run cannot overwrite a first, and two paths dropped it.

    The cost stage is run once per ARM, so pricing `core` wrote over the `base` table the record
    cites -- silently, with a success message naming the file it had just destroyed. This is a
    source-level check because reproducing it needs a pool.
    """
    import re
    from pathlib import Path
    source = Path(driver.__file__).read_text(encoding='utf-8')
    unsuffixed = [line.strip() for line in source.splitlines()
                  if re.search(r"f'\{args\.tag\}_[a-z_]+\.(csv|json|parquet)'", line)]
    assert not unsuffixed, f'artifact paths that ignore --suffix: {unsuffixed}'


# ---------------------------------------------------------------------------------------------
# the sharded export
# ---------------------------------------------------------------------------------------------
def _shard(tmp_path, bundle, in_pool, rows=3):
    """One (fit, cal, meta) shard triple as an array task writes it.

    `in_pool` is the POOL's full bundle list, which every shard records. A per-shard list of what
    that shard covered cannot reveal a task that never ran -- see `_assert_shards_complete`.
    """
    import json
    for label in ('fit', 'cal'):
        pd.DataFrame({'condition_bundle': [bundle]*rows,
                      'entry_id': list(range(rows)),
                      'is_correct': [True] + [False]*(rows-1)}).to_parquet(
            tmp_path/f'S12_combiner_{label}_frame_fullscale_{bundle}.parquet', index=False)
    (tmp_path/f'S12_combiner_export_meta_fullscale_{bundle}.json').write_text(
        json.dumps({'bundles': [bundle], 'bundles_in_pool': in_pool}), encoding='utf-8')


def test_a_missing_array_task_is_refused_rather_than_fitted_around(tmp_path):
    """A SLURM array loses a task quietly and the survivors still glob into a usable frame.

    The failure is a model fitted on fewer conditions than its write-up claims, with no symptom
    but a slightly wrong number. Each shard's meta records the bundle list its invocation was
    given, so the union of those is what must be present.
    """
    pool = ['c2_error0.1_cont0', 'c2_error1_cont0', 'c2_error2_cont0']
    _shard(tmp_path, 'c2_error1_cont0', pool)
    _shard(tmp_path, 'c2_error2_cont0', pool)
    pairs = [(tmp_path/f'S12_combiner_fit_frame_fullscale_{b}.parquet',
              tmp_path/f'S12_combiner_cal_frame_fullscale_{b}.parquet')
             for b in ('c2_error1_cont0', 'c2_error2_cont0')]
    covered = ['c2_error1_cont0', 'c2_error2_cont0']
    with pytest.raises(SystemExit) as problem:
        driver._assert_shards_complete(pairs, covered)
    assert 'c2_error0.1_cont0' in str(problem.value)
    # And a deliberate subset is allowed through, loudly.
    driver._assert_shards_complete(pairs, covered, allow_partial=True)


def test_a_complete_set_of_shards_passes(tmp_path):
    pool = ['c2_error1_cont0', 'c2_error2_cont0']
    _shard(tmp_path, 'c2_error1_cont0', pool)
    _shard(tmp_path, 'c2_error2_cont0', pool)
    pairs = [(tmp_path/f'S12_combiner_fit_frame_fullscale_{b}.parquet',
              tmp_path/f'S12_combiner_cal_frame_fullscale_{b}.parquet')
             for b in ('c2_error1_cont0', 'c2_error2_cont0')]
    driver._assert_shards_complete(pairs, ['c2_error1_cont0', 'c2_error2_cont0'])


def test_shards_with_no_meta_do_not_raise(tmp_path):
    """An export from before sharding wrote no bundle list; absence is not evidence of a gap."""
    driver._assert_shards_complete([(tmp_path/'S12_combiner_fit_frame.parquet', tmp_path/'x')],
                                   ['c2_error1_cont0'])


# ---------------------------------------------------------------------------------------------
# a column the frame does not carry
# ---------------------------------------------------------------------------------------------
def test_an_absent_design_matrix_column_is_named_at_export_time(capsys):
    """`_sidecar_projection` drops a name no sidecar carries, silently and by design.

    Right per sidecar -- a pool has several and a caller asks once against all of them -- but a
    column present in NONE of them vanishes without a word. The full-scale export then shipped
    nine shards with no `N_cal`, and the failure surfaced as sixteen arms skipping themselves on a
    laptop hours later. The export has to say it where the job that could re-run cheaply is.
    """
    groups = driver.arm_groups(())
    wanted, _ = FomCombiner.feature_specification(groups, drop=())
    frame = pd.DataFrame({name: [0.0] for name in wanted if name != 'N_cal'})
    absent = driver._report_absent_features(frame, groups)
    assert absent == ['N_cal']
    assert 'N_cal' in capsys.readouterr().out


def test_a_complete_frame_says_nothing(capsys):
    groups = driver.arm_groups(())
    wanted, _ = FomCombiner.feature_specification(groups, drop=())
    frame = pd.DataFrame({name: [0.0] for name in wanted})
    assert driver._report_absent_features(frame, groups) == []
    assert capsys.readouterr().out == ''


def test_drop_columns_costs_one_feature_rather_than_every_arm():
    """One absent column skipped sixteen of seventeen arms. `--drop-columns` makes it cost one."""
    args = driver._parse_args(['--stage', 'fit', '--drop-columns', 'N_cal'])
    assert args.drop_columns == 'N_cal'
    extra = tuple(n.strip() for n in args.drop_columns.split(',') if n.strip())
    groups = driver.arm_groups(())
    full, _ = FomCombiner.feature_specification(groups, drop=driver.BASE_DROP)
    cut, _ = FomCombiner.feature_specification(groups, drop=driver.BASE_DROP + extra)
    assert len(cut) == len(full) - 1
    assert 'N_cal' in full and 'N_cal' not in cut
