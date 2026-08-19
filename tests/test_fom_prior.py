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


def test_dropout_bundles_actually_punch_holes():
    pool = _fake_pool()
    window = np.stack([np.asarray(values)[:20] for values in pool['q2_full']])
    index = [bundle['n_dropout'] for bundle in Prior.CONDITION_BUNDLES].index(10)
    dropped, _, _ = Prior.draw_peak_lists(pool, np.random.default_rng(0), bundle_index=index)
    # backfilling from beyond the window pushes the last peak out to higher q2
    assert (dropped[:, -1] > window[:, -1]).mean() > 0.9


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
