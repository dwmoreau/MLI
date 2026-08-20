"""S11 block A: the leakage guard, the label algebra, and the pieces the grid is rebuilt from.

The arithmetic tests here are not the ones that matter. This task has exactly one way to produce a
large, wrong, publishable number -- training on the entries it is evaluated on -- and that failure
is silent everywhere else in the pipeline, because the two tables join on a column neither of them
advertises as a join key. It is asserted first, on the real manifest, not on a fixture.

The rest guard the things a later session would otherwise have to rediscover: that the class
balancer actually balances, that the condition draw keeps the peak list sorted and finite, and that
a saved model rebuilds to the same predictions.
"""
import numpy as np
import os
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training import PriorNetwork as Prior  # noqa: E402


REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPOSITORY, 'docs', 'fom', 'artifacts', 'S02_mirror_manifest.parquet')
DATASETS = os.path.join(REPOSITORY, 'mlindex', 'data', 'generated_datasets')

pytest.importorskip('pyarrow')


def _have(path):
    return os.path.exists(path)


# ---------------------------------------------------------------------------------------
# The leakage guard
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not _have(MANIFEST), reason='frozen split manifest not present')
def test_held_out_identifiers_are_dev_and_test_only():
    import pandas as pd

    manifest = pd.read_parquet(MANIFEST, columns=['identifier', 'split'])
    held_out = Prior.held_out_identifiers(MANIFEST)
    expected = set(manifest.loc[manifest['split'] != 'fom-train', 'identifier'])
    assert held_out == expected
    assert held_out.isdisjoint(set(manifest.loc[manifest['split'] == 'fom-train', 'identifier']))


@pytest.mark.skipif(not (_have(MANIFEST) and _have(DATASETS)),
                    reason='manifest or generated datasets not present')
def test_training_pool_contains_no_held_out_structure():
    """The one that matters. On the real data, because a fixture cannot reproduce the overlap.

    mP is the check worth making by hand: all 500 of its manifest entries are present in
    `dataset_mP.parquet` and 200 of them are held out, so a pool built without the filter would
    contain them and nothing downstream would notice.
    """
    import pandas as pd

    pool = Prior.load_prior_frame(
        DATASETS, MANIFEST, bravais_lattices=('mP',), limit_per_lattice=None,
        )
    held_out = Prior.held_out_identifiers(MANIFEST)
    assert len(pool) > 0
    assert set(pool['identifier']).isdisjoint(held_out)

    manifest = pd.read_parquet(MANIFEST, columns=['identifier', 'split', 'bravais_lattice'])
    manifest = manifest.loc[manifest['bravais_lattice'] == 'mP']
    raw = pd.read_parquet(os.path.join(DATASETS, 'dataset_mP.parquet'), columns=['identifier'])
    overlap = set(manifest['identifier']) & set(raw['identifier'])
    assert len(overlap) > 0, 'the overlap this filter exists for should be present'
    assert len(overlap & held_out) > 0, 'and some of it should be held out'


@pytest.mark.skipif(not (_have(MANIFEST) and _have(DATASETS)),
                    reason='manifest or generated datasets not present')
def test_pool_does_not_use_the_datasets_own_train_column():
    """`dataset_*.parquet` carries a `train` boolean from a different task's split.

    Using it would be a plausible-looking mistake with the same consequence as no filter at all, so
    the loaded frame does not carry the column and cannot be filtered on it by accident.
    """
    pool = Prior.load_prior_frame(
        DATASETS, MANIFEST, bravais_lattices=('cF',), limit_per_lattice=None,
        )
    assert 'train' not in pool.columns


# ---------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------
def test_target_codes_are_consistent_coarsenings():
    codes = Prior.target_codes(np.array(Prior.BRAVAIS_LATTICES, dtype=object))
    for index, lattice in enumerate(Prior.BRAVAIS_LATTICES):
        assert Prior.BRAVAIS_LATTICES[codes['bravais'][index]] == lattice
        system = Prior.LATTICE_SYSTEMS[codes['system'][index]]
        assert system == Prior.LATTICE_SYSTEM_OF[lattice]
        assert Prior.CENTRINGS[codes['centring'][index]] == lattice[1]
        n_free = Prior.N_FREE_LEVELS[codes['n_free'][index]]
        assert n_free == Prior.N_FREE_OF[system]
        # high_symmetry must be a strict coarsening of n_free, not an independent judgement
        assert codes['high_symmetry'][index] == int(n_free <= Prior.HIGH_SYMMETRY_MAX_FREE)


def test_every_bravais_lattice_has_a_target():
    for lattice in Prior.BRAVAIS_LATTICES:
        assert lattice in Prior.LATTICE_SYSTEM_OF
        assert lattice[1] in Prior.CENTRINGS
    for target, n_classes in Prior.TARGET_CLASSES.items():
        codes = Prior.target_codes(np.array(Prior.BRAVAIS_LATTICES, dtype=object))[target]
        assert codes.min() >= 0 and codes.max() < n_classes


# ---------------------------------------------------------------------------------------
# The condition draw
# ---------------------------------------------------------------------------------------
def _fake_pool(n_rows=60, n_lines=60, seed=3):
    import pandas as pd

    rng = np.random.default_rng(seed)
    lattices = np.array(Prior.BRAVAIS_LATTICES)[rng.integers(0, 14, size=n_rows)]
    rows = []
    for index in range(n_rows):
        q2 = np.sort(rng.uniform(0.005, 0.25, size=n_lines))
        rows.append(dict(identifier=f'X{index:04d}', bravais_lattice=lattices[index],
                         volume=float(rng.uniform(200, 8000)), q2_full=q2))
    frame = pd.DataFrame(rows)
    for name, values in Prior.target_codes(frame['bravais_lattice'].to_numpy()).items():
        frame[f'target_{name}'] = values
    return frame


def test_drawn_peak_lists_are_sorted_finite_and_the_right_length():
    pool = _fake_pool()
    q2, bundles, refused = Prior.draw_peak_lists(pool, np.random.default_rng(0), n_peaks=20)
    assert q2.shape == (len(pool), 20)
    assert np.isfinite(q2).all()
    assert (np.diff(q2, axis=1) >= 0).all(), 'a figure of merit assumes a sorted peak list'
    assert (q2 > 0).all()
    assert set(np.unique(bundles)).issubset(set(range(len(Prior.CONDITION_BUNDLES))))
    assert refused >= 0


def test_pinning_the_bundle_reproduces_that_bundles_conditions():
    pool = _fake_pool()
    clean, _, _ = Prior.draw_peak_lists(
        pool, np.random.default_rng(0), bundle_index=0,
        )
    noisy, _, _ = Prior.draw_peak_lists(
        pool, np.random.default_rng(0), bundle_index=1,
        )
    # bundle 1 is error x2 on the same windows, so its residual against the clean window is larger
    window = np.stack([np.asarray(values)[:20] for values in pool['q2_full']])
    assert np.abs(noisy - window).mean() > np.abs(clean - window).mean()


def test_dropout_still_punches_holes_when_a_bundle_asks_for_it():
    """The mechanism, not the configuration.

    DWMM's condition mix carries no dropout, so asserting against the default bundles would assert
    nothing. `draw_peak_lists` still supports it and S02/S04's grid still uses it, so the guard is
    written against an explicit bundle instead -- it survives the configuration changing again.
    """
    pool = _fake_pool()
    window = np.stack([np.asarray(values)[:20] for values in pool['q2_full']])
    bundles = (dict(name='drop10', multiplier=1.0, n_contaminants=0, n_dropout=10,
                    second_phase=0, weight=1.0),)
    dropped, _, _ = Prior.draw_peak_lists(
        pool, np.random.default_rng(0), bundles=bundles, bundle_index=0,
        )
    # backfilling from beyond the window pushes the last peak out to higher q2
    assert (dropped[:, -1] > window[:, -1]).mean() > 0.9


def test_conditions_are_drawn_at_their_specified_frequencies():
    """0/1/2 contaminants at 1 : 0.5 : 0.25, which is 57.1 / 28.6 / 14.3 per cent."""
    pool = _fake_pool(n_rows=4000)
    _, drawn, _ = Prior.draw_peak_lists(pool, np.random.default_rng(11))
    weights = np.array([bundle.get('weight', 1.0) for bundle in Prior.CONDITION_BUNDLES])
    expected = weights/weights.sum()
    observed = np.bincount(drawn, minlength=len(Prior.CONDITION_BUNDLES))/drawn.size
    np.testing.assert_allclose(observed, expected, atol=0.02)


def test_the_condition_mix_carries_no_dropout_or_second_phase():
    """DWMM's specification, asserted so a silent revert to the S04 grid cannot happen.

    Training across that grid cost about fourteen points of Bravais accuracy (F-110, F-112), and
    the failure mode was invisible -- the loss curve looked fine.
    """
    for bundle in Prior.CONDITION_BUNDLES:
        assert bundle['multiplier'] == 1.0
        assert bundle['n_dropout'] == 0
        assert bundle['second_phase'] == 0
        assert bundle['n_contaminants'] in (0, 1, 2)


# ---------------------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------------------
def test_balanced_indices_give_every_class_the_same_count():
    codes = np.concatenate([np.zeros(500, dtype=int), np.ones(3, dtype=int),
                            np.full(40, 2, dtype=int)])
    rows = Prior.balanced_indices(codes, 3, np.random.default_rng(0), per_class=50)
    counts = np.bincount(codes[rows], minlength=3)
    assert (counts == 50).all(), 'a 167:1 imbalance must come out flat'
    # the thin class is necessarily drawn with replacement; the fresh condition draw per epoch is
    # what makes that acceptable, and it is the reason this is a sampler rather than a loss weight
    assert len(set(rows[codes[rows] == 1])) <= 3


def test_balanced_indices_skip_an_absent_class():
    codes = np.array([0, 0, 2, 2])
    rows = Prior.balanced_indices(codes, 3, np.random.default_rng(0), per_class=4)
    assert set(np.unique(codes[rows])) == {0, 2}


# ---------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def keras_available():
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    return pytest.importorskip('keras')


def _tiny_model(tmp_path, head_mode='conditional'):
    pool = _fake_pool(n_rows=80)
    q2, _, _ = Prior.draw_peak_lists(pool, np.random.default_rng(0))
    pool = pool.assign(q2_window=list(q2))
    model = Prior.PriorNetwork(
        data_params={}, save_to=str(tmp_path),
        model_params={'n_volumes': 16, 'n_filters': 64, 'layers': [16], 'd_model': 16,
                      'n_heads': 2, 'epochs': 1, 'batch_size': 16, 'head_mode': head_mode},
        seed=7,
        )
    model.build_model(data=pool)
    return model, pool, q2


@pytest.mark.parametrize('head_mode', ('conditional', 'marginal', 'joint'))
def test_every_head_emits_a_normalised_distribution(keras_available, tmp_path, head_mode):
    """The heads emit log probabilities, and the loss depends on that being literally true."""
    model, _, q2 = _tiny_model(tmp_path, head_mode)
    distributions = model.predict_distributions(q2)
    for target, probability in distributions.items():
        assert probability.shape[1] == (
            model.model_params['n_volumes'] if target == 'volume_branch'
            else Prior.TARGET_CLASSES[target]
            )
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-5,
                                   err_msg=f'{target} under {head_mode}')


def test_branch_labels_land_on_the_nearest_branch_in_log_volume(keras_available, tmp_path):
    model, pool, _ = _tiny_model(tmp_path)
    labels = model.get_branch_labels(pool)
    scales = ((1.0/pool['volume'].to_numpy(dtype=float))/model.q2_obs_scale**2)**(2/3)
    scales = scales/model.extraction_layer.volume_normalization
    branches = np.asarray(model.extraction_layer.volumes).ravel().astype(float)
    expected = np.abs(
        np.log(scales)[:, np.newaxis] - np.log(branches)[np.newaxis]
        ).argmin(axis=1)
    np.testing.assert_array_equal(labels, expected)


def test_marginal_head_cannot_see_the_volume_branch(keras_available, tmp_path):
    """The A5 control has to be a control: its class head must not depend on the branch axis."""
    model, _, _ = _tiny_model(tmp_path, head_mode='marginal')
    names = [layer.name for layer in model.model.layers]
    assert 'bravais_pool' in names


def test_saved_model_reloads_to_the_same_predictions(keras_available, tmp_path):
    model, _, q2 = _tiny_model(tmp_path)
    before = model.predict_distributions(q2)
    directory = model.save_prior()
    restored = Prior.PriorNetwork.load_prior(directory, seed=7)
    after = restored.predict_distributions(q2)
    for target in before:
        np.testing.assert_allclose(before[target], after[target], atol=1e-6)
    assert restored.q2_obs_scale == model.q2_obs_scale
    assert restored.extraction_layer.volume_normalization == pytest.approx(
        model.extraction_layer.volume_normalization
        )


# ---------------------------------------------------------------------------------------
# The split, and what the dense reference is compared against
# ---------------------------------------------------------------------------------------
def _args(**overrides):
    import argparse

    values = dict(datasets_dir=DATASETS, manifest=MANIFEST, limit_per_lattice=40,
                  test_fraction=0.2, seed=5, include_cubic=False)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.skipif(not (_have(MANIFEST) and _have(DATASETS)),
                    reason='manifest or generated datasets not present')
def test_cubic_is_excluded_by_default_and_restorable():
    """The comparison against `scripts/classification.py` only means anything on the same lattices."""
    from mlindex.scripts import run_fom_prior

    default = run_fom_prior.active_lattices(_args())
    assert default is not None
    assert not any(code.startswith('c') for code in default)
    assert len(default) == 11
    assert run_fom_prior.active_lattices(_args(include_cubic=True)) is None


@pytest.mark.skipif(not (_have(MANIFEST) and _have(DATASETS)),
                    reason='manifest or generated datasets not present')
def test_fit_and_heldout_pools_share_no_structure():
    """Split by source structure, never by row.

    One structure yields many patterns over training, under different drawn conditions. A row split
    would put the same crystal on both sides and the held-out number would mean nothing.
    """
    from mlindex.scripts import run_fom_prior

    fit, held_out = run_fom_prior.build_pools(_args())
    assert len(fit) > 0 and len(held_out) > 0
    assert set(fit['identifier']).isdisjoint(set(held_out['identifier']))
    # and neither side may contain a benchmark structure the FOM is judged on
    assert set(fit['identifier']).isdisjoint(Prior.held_out_identifiers(MANIFEST))


@pytest.mark.skipif(not (_have(MANIFEST) and _have(DATASETS)),
                    reason='manifest or generated datasets not present')
def test_holdout_evaluation_carries_the_columns_the_scorer_reads():
    from mlindex.scripts import run_fom_prior

    _, held_out = run_fom_prior.build_pools(_args())
    frame = run_fom_prior.holdout_evaluation(held_out, np.random.default_rng(0))
    for column in ('entry_id', 'condition_bundle', 'q2_obs', 'volume_true', 'target_bravais'):
        assert column in frame.columns
    assert len(np.stack(frame['q2_obs'].to_numpy())[0]) == 20
    assert set(frame['condition_bundle']).issubset(
        {bundle['name'] for bundle in Prior.CONDITION_BUNDLES}
        )


def test_dense_baseline_matches_the_reference_architecture(keras_available):
    """The baseline is `scripts/classification.py`'s stack, not a new invention."""
    from mlindex.scripts import run_fom_prior

    model = run_fom_prior.build_dense_baseline(20, {'bravais': 14, 'system': 7})
    widths = [layer.units for layer in model.layers
              if layer.__class__.__name__ == 'Dense' and layer.name.startswith('dense_')]
    assert widths == [1024, 512, 256, 128, 64, 32]
    assert all(not layer.use_bias for layer in model.layers
               if layer.name.startswith('dense_'))
    assert set(model.output.keys()) == {'bravais', 'system'}
    # no volume-branch head: there is no branch grid without the extraction layer, which is the
    # thing under comparison
    assert 'volume_branch' not in model.output


def test_joint_loss_is_the_same_inside_fit_as_it_is_eagerly(keras_available):
    """The test that would have caught a 30-epoch run descending a quantity that did not exist.

    `_joint_loss` was first written with `y_pred[rows, branch]` and an `arange` over the batch.
    That is correct eagerly -- a direct call reproduced `-log P(c*|v*)` exactly -- and wrong inside
    the compiled training step, where the batch dimension is not resolved the same way. `fit`
    reported 0.32 while the identical weights evaluated to 31.75. So checking the loss in isolation
    is not enough: it has to be checked through `fit`.
    """
    import keras

    n_v, n_classes, batch = 8, 4, 6
    inputs = keras.Input(shape=(3,))
    hidden = keras.layers.Dense(n_v*n_classes, use_bias=False)(inputs)
    grid = keras.layers.Reshape((n_v, n_classes))(hidden)
    outputs = keras.layers.Activation('log_softmax', name='bravais')(grid)
    model = keras.Model(inputs, {'bravais': outputs})
    model.compile(optimizer='sgd', loss={'bravais': Prior._joint_loss(n_classes)})

    rng = np.random.default_rng(0)
    x = rng.normal(size=(batch, 3)).astype('float32')
    codes = rng.integers(0, n_classes, batch)
    branches = rng.integers(0, n_v, batch)
    labels = Prior.joint_targets(codes, branches)

    reported = model.evaluate(x, {'bravais': labels}, batch_size=batch, verbose=0)
    predicted = np.asarray(keras.ops.convert_to_numpy(model(x)['bravais']), dtype=float)
    expected = float(np.mean(-predicted[np.arange(batch), branches, codes]))
    assert reported == pytest.approx(expected, abs=1e-5), (
        f'loss through the graph {reported} != -log P(c*|v*) {expected}'
        )


def test_soft_volume_target_is_ordinal_and_normalised():
    """The branches discretise a continuous quantity, so the target should too.

    Width 0 must reproduce the one-hot label exactly, so the default path is unchanged.
    """
    hard = Prior.soft_branch_targets([5, 20], 32, 0.0)
    assert hard.shape == (2, 32)
    np.testing.assert_allclose(hard.sum(axis=1), 1.0)
    assert hard[0, 5] == 1.0 and hard[1, 20] == 1.0

    soft = Prior.soft_branch_targets([5, 20], 32, 2.0)
    np.testing.assert_allclose(soft.sum(axis=1), 1.0, atol=1e-6)
    assert soft[0].argmax() == 5 and soft[1].argmax() == 20
    # monotone decreasing away from the truth, which is the whole point
    assert soft[0, 5] > soft[0, 6] > soft[0, 8] > soft[0, 15]


def test_joint_api_scores_the_claimed_pair(keras_available, tmp_path):
    """`score_candidates` must read the candidate's own claimed lattice, never the true one."""
    model, pool, q2 = _tiny_model(tmp_path)
    joint = model.joint_log_probability(q2, batch_size=8)
    n_volumes = model.model_params['n_volumes']
    assert joint.shape == (len(pool), n_volumes, Prior.TARGET_CLASSES['bravais'])
    np.testing.assert_allclose(np.exp(joint).sum(axis=(1, 2)), 1.0, atol=1e-6)

    volumes = pool['volume'].to_numpy(dtype=float)
    claimed = pool['target_bravais'].to_numpy()
    scored = model.score_candidates(q2, volumes, claimed, batch_size=8)
    assert scored.shape == (len(pool),)
    # the same volume under a different claimed lattice must give a different score, or the
    # lattice axis is being ignored and the joint is doing nothing
    other = (claimed + 1) % len(Prior.BRAVAIS_LATTICES)
    assert not np.allclose(scored, model.score_candidates(q2, volumes, other, batch_size=8))
    assert np.all(model.branch_volumes() > 0)


def test_balanced_sampler_input_is_not_shadowed_by_the_target_loop():
    """The regression test for F-121, written against the shape of the bug rather than its symptom.

    `fit_arm` binds the sampler's input once outside the epoch loop and then builds per-target
    arrays inside it. A rebinding there fed `balanced_indices` the previous batch's
    `high_symmetry` labels -- a 2-class array of batch length -- so its indices were bounded by that
    length, `iloc` took the front of the lattice-ordered pool, and the training set decayed to one
    lattice by epoch 5. Every class loss then read 0.000 because it was predicting a constant.

    Simulated here rather than trained: the failure is in the index arithmetic, and it needs no
    model to reproduce.
    """
    rng = np.random.default_rng(0)
    n_classes = len(Prior.BRAVAIS_LATTICES)
    pool_codes = np.repeat(np.arange(n_classes), 300)          # lattice-ordered, as the real pool is
    per_class = 50

    codes = pool_codes
    seen = []
    for _ in range(6):
        rows = Prior.balanced_indices(codes, n_classes, rng, per_class)
        batch = pool_codes[rows]
        seen.append(len(np.unique(batch)))
        # the bug: rebinding the sampler's input to a per-batch target array
        codes = (batch >= n_classes - 4).astype(np.int64)       # stands in for `high_symmetry`
    assert seen[0] == n_classes, 'the first epoch is always fine, which is what hid this'
    assert min(seen) < n_classes, 'the shadowing must actually degrade the batch'

    # and the correct form keeps every class present at every epoch
    codes = pool_codes
    for _ in range(6):
        rows = Prior.balanced_indices(codes, n_classes, rng, per_class)
        batch = pool_codes[rows]
        assert len(np.unique(batch)) == n_classes
        assert len(batch) == per_class*n_classes
        values = (batch >= n_classes - 4).astype(np.int64)      # bound to a *different* name
        assert values.size == batch.size
