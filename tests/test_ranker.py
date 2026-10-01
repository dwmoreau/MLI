"""The learned ranker's data path, fit, calibration and ONNX export.

These pin what decides whether a fitted ranker can be trusted: which rows a thinning keeps and
with what weight, that a seed changes the sample and nothing else does, that the pool-context
inputs do not depend on the thinning, and that a model's ONNX export scores as the model does.
Each is checked against a case whose answer is known, never against the implementation itself.
"""

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import BenchmarkMetrics as metrics
from mlindex.model_training import FomCombiner as ranker


def _pool(n_entries=3, n_per_lattice=60, lattices=('cP', 'mP', 'aP'), seed=0):
    """A synthetic candidate frame carrying every ranker input, with one correct cell a pattern."""
    rng = np.random.default_rng(seed)
    rows = []
    for entry in range(n_entries):
        for bundle in ('b0', 'b1'):
            for lattice in lattices:
                n = n_per_lattice
                frame = pd.DataFrame({
                    'entry_id': f'E{entry}', 'condition_bundle': bundle,
                    'bravais_lattice': lattice, 'candidate_id': np.arange(n)})
                for name in ranker.FEATURES:
                    if name != ranker.LATTICE_FEATURE:
                        frame[name] = rng.normal(size=n)
                frame['M20'] = rng.gamma(2.0, 3.0, size=n)
                frame['m20_at_prune'] = frame['M20'].to_numpy()
                frame['is_correct'] = False
                frame['in_top_n'] = True
                rows.append(frame)
    frame = pd.concat(rows, ignore_index=True)
    first = frame.groupby(['entry_id', 'condition_bundle']).head(1).index
    frame.loc[first, 'is_correct'] = True
    return frame


def test_a_keyed_draw_belongs_to_the_candidate_not_to_its_row():
    """Reading the same pool in another order or another grouping must not change which
    candidates a thinning keeps, so the draw is a function of the key and seed alone."""
    frame = _pool()
    draw = ranker.keyed_uniform(frame, 12345)
    shuffled = frame.sample(frac=1.0, random_state=1)
    redrawn = pd.Series(ranker.keyed_uniform(shuffled, 12345), index=shuffled.index)
    np.testing.assert_array_equal(redrawn.loc[frame.index].to_numpy(), draw)
    assert not np.array_equal(ranker.keyed_uniform(frame, 777), draw)
    assert not np.array_equal(ranker.keyed_uniform(frame, 12345, stream=1), draw)
    assert 0.0 <= draw.min() and draw.max() < 1.0


def test_thinning_keeps_the_best_candidates_by_a_lower_is_better_merit():
    """X_N, n_over and max_gap count errors. The top-k by those is the SMALLEST values; ranking
    them largest-first, as the campaign's thinning did, keeps the worst candidates instead."""
    frame = _pool(n_entries=1, n_per_lattice=50, lattices=('mP',))
    frame = frame.loc[frame['condition_bundle'] == 'b0'].reset_index(drop=True)
    for merit in ranker.RAW_MERITS:
        frame[merit] = 0.0
    frame['n_over'] = np.arange(frame.shape[0], dtype=float)
    thinned = ranker.thin_negatives(frame, top_k=5, negative_rate=1e-9, seed=12345)
    kept = set(thinned['candidate_id'])
    assert {0, 1, 2, 3, 4} <= kept
    assert not {45, 46, 47, 48, 49} & kept


def test_thinning_keeps_every_correct_candidate_and_weights_the_sampled_ones():
    frame = _pool()
    thinned = ranker.thin_negatives(frame, top_k=3, negative_rate=0.25, seed=12345)
    assert thinned['is_correct'].sum() == frame['is_correct'].sum()
    in_top_k = ranker.top_k_mask(thinned, 3)
    weights = thinned['sampling_weight'].to_numpy()
    certain = in_top_k | thinned['is_correct'].to_numpy()
    assert set(weights[certain]) == {1.0}
    assert set(weights[~certain]) == {4.0}
    # The sampled share is near the rate, over the rows not kept with certainty.
    n_other = (~ranker.top_k_mask(frame, 3) & ~frame['is_correct'].to_numpy()).sum()
    assert 0.15 < (~certain).sum()/n_other < 0.35


def test_the_cap_is_a_nested_uniform_sample_with_its_inclusion_weight():
    frame = _pool(n_per_lattice=40).assign(sampling_weight=2.0)
    frame['negative_order'] = ranker.negative_order(frame, 12345)
    frame['n_negatives_thinned'] = frame.groupby(['entry_id', 'condition_bundle'])[
        'is_correct'].transform(lambda values: (~values).sum())
    small = ranker.cap_negatives(frame, 10)
    large = ranker.cap_negatives(frame, 30)
    key = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']
    assert set(map(tuple, small[key].to_numpy())) <= set(map(tuple, large[key].to_numpy()))
    negatives = small.loc[~small['is_correct']]
    assert (negatives.groupby(['entry_id', 'condition_bundle']).size() == 10).all()
    # 119 wrong candidates a pattern, 10 kept: each stands for 11.9 of them.
    np.testing.assert_allclose(negatives['fit_weight'], 2.0*119/10)
    np.testing.assert_allclose(small.loc[small['is_correct'], 'fit_weight'], 2.0)
    other = ranker.cap_negatives(frame.assign(negative_order=ranker.negative_order(frame, 777)), 10)
    assert set(map(tuple, other[key].to_numpy())) != set(map(tuple, small[key].to_numpy()))


def test_the_context_gaps_do_not_depend_on_the_thinning():
    """Thinning keeps each merit's best candidate in every lattice, so a gap taken over the whole
    pattern is the same whether it is computed before or after thinning -- and differs when it is
    taken over one lattice alone, which is why the export takes it over all of them."""
    frame = _pool()
    whole = ranker.add_context(frame)
    thinned = ranker.thin_negatives(frame, top_k=2, negative_rate=0.01, seed=12345)
    gaps = ranker.add_context(thinned, ranker.context_best(frame))
    key = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']
    joined = gaps.merge(whole, on=key, suffixes=('', '_whole'))
    for name in ranker.CONTEXT_FEATURES:
        np.testing.assert_array_equal(joined[name], joined[f'{name}_whole'])
    one_lattice = ranker.add_context(frame.loc[frame['bravais_lattice'] == 'aP'])
    assert not np.allclose(one_lattice['ctx_M20_gap_to_best'],
                           whole.loc[whole['bravais_lattice'] == 'aP', 'ctx_M20_gap_to_best'])
    # Lower-is-better merits are negated, so every gap is at most zero and the best is zero.
    assert (whole[list(ranker.CONTEXT_FEATURES)].to_numpy() <= 0).all()


def test_restricting_at_a_cut_keeps_each_lattices_best_and_reranks():
    frame = pd.DataFrame({
        'entry_id': 'E', 'condition_bundle': 'b0',
        'bravais_lattice': ['cP', 'cP', 'cP', 'mP', 'mP'],
        'candidate_id': [0, 1, 2, 0, 1],
        'M20': [9.0, 4.0, 2.0, 3.0, 2.5],
        'm20_at_prune': [9.0, 4.0, 2.0, 3.0, 2.5],
        'final_rank': [0, 1, 2, 0, 1], 'in_top_n': True, 'pool_size_full': 5.0})
    out = metrics.restrict_at_cut(frame, 3.5, n_top=1)
    assert list(zip(out['bravais_lattice'], out['candidate_id'])) == [
        ('cP', 0), ('cP', 1), ('mP', 0)]
    assert out['final_rank'].tolist() == [0, 1, 0]
    assert out['in_top_n'].tolist() == [True, False, True]
    assert set(out['pool_size_full']) == {3.0}


def test_split_crystals_is_disjoint_stratified_and_seeded():
    entries = pd.DataFrame({
        'entry_id': [f'C{index:03d}' for index in range(200)],
        'bravais_lattice_true': ['aP']*150 + ['cF']*50})
    parts = ranker.split_crystals(entries, {'a': 0.8, 'b': 0.2}, np.random.default_rng(1))
    assert not parts['a'] & parts['b']
    assert len(parts['a'] | parts['b']) == 200
    assert sum(name in parts['b'] for name in entries['entry_id'][150:]) == 10
    again = ranker.split_crystals(entries, {'a': 0.8, 'b': 0.2}, np.random.default_rng(1))
    other = ranker.split_crystals(entries, {'a': 0.8, 'b': 0.2}, np.random.default_rng(2))
    assert again == parts and other != parts


def test_the_leakage_guard_refuses_labels_and_generator_columns():
    for name in ('is_correct', 'sampling_weight', 'm20_at_prune', 'volume_true'):
        with pytest.raises(ValueError):
            ranker.FomCombiner('onehot', features=ranker.FEATURES + (name,))
    ranker.check_no_leakage(ranker.FEATURES)


def test_the_lattice_encodings_shape_the_design_matrix():
    frame = _pool(n_entries=1, n_per_lattice=3)
    widths = {encoding: ranker.FomCombiner(encoding).design_matrix(frame).shape[1]
              for encoding in ranker.LATTICE_ENCODINGS}
    assert widths == {'onehot': 31 + 13, 'ordinal': 31, 'native': 31}
    with pytest.raises(ValueError):
        ranker.FomCombiner('ordinal').design_matrix(frame.assign(bravais_lattice='xX'))


def _fitted(encoding, seed=12345):
    frame = _pool(n_entries=12, n_per_lattice=30, lattices=('cF', 'tI', 'oC', 'mP', 'aP'))
    # A lattice dependence only a split on a set of lattices captures exactly.
    member = frame['bravais_lattice'].isin(['cF', 'oC', 'aP']).to_numpy()
    rng = np.random.default_rng(seed)
    frame['is_correct'] = rng.random(frame.shape[0]) < np.where(member, 0.6, 0.1)
    frame = frame.assign(sampling_weight=1.0, fit_weight=1.0)
    combiner = ranker.FomCombiner.fit(frame, encoding=encoding, seed=seed,
                                      params=dict(max_iter=40, max_leaf_nodes=15))
    return combiner.fit_calibrators(frame, minimum=20), frame


@pytest.mark.parametrize('encoding', ranker.EXPORTABLE_ENCODINGS)
def test_an_exportable_model_scores_the_same_through_onnx(tmp_path, encoding):
    combiner, frame = _fitted(encoding)
    combiner.save(tmp_path / 'model')
    onnx = ranker.onnx_probability(tmp_path / 'model' / 'model.onnx',
                                   combiner.design_matrix(frame))
    np.testing.assert_allclose(onnx, combiner.raw_score(frame), atol=1e-6)


def test_a_native_categorical_model_does_not_survive_onnx_conversion(tmp_path):
    """Why `native` is never exported: skl2onnx writes a categorical split as a threshold split,
    converts without complaint, and scores differently. This is the case the test above exists to
    catch, so it must fail it."""
    from mlindex.utilities.IOManagers import SKLearnManager

    combiner, frame = _fitted('native')
    SKLearnManager(filename=str(tmp_path / 'native'), model_type='onnx').save(
        model=combiner.model, n_features=len(combiner.matrix_names))
    onnx = ranker.onnx_probability(tmp_path / 'native.onnx', combiner.design_matrix(frame))
    assert np.max(np.abs(onnx - combiner.raw_score(frame))) > 1e-2


def test_a_saved_model_loads_identically_and_is_never_overwritten(tmp_path):
    combiner, frame = _fitted('ordinal')
    combiner.save(tmp_path / 'model')
    loaded = ranker.FomCombiner.load(tmp_path / 'model')
    np.testing.assert_array_equal(loaded.score(frame), combiner.score(frame))
    with pytest.raises(FileExistsError):
        combiner.save(tmp_path / 'model')
    assert not (tmp_path / 'native').exists()
    _fitted('native')[0].save(tmp_path / 'native')
    assert not (tmp_path / 'native' / 'model.onnx').exists()


def test_a_refit_with_the_same_seed_is_identical():
    """The classifier draws nothing of its own (no early-stopping holdout, no feature sampling),
    so a fit seed acts only through the split and the sample, which the tests above pin."""
    first, frame = _fitted('onehot', seed=1)
    again, _ = _fitted('onehot', seed=1)
    np.testing.assert_array_equal(first.raw_score(frame), again.raw_score(frame))


def test_calibration_falls_back_to_the_pooled_curve_for_a_small_lattice():
    raw = np.linspace(0, 1, 300)
    target = (raw > 0.5).astype(float)
    lattice = np.array(['aP']*250 + ['cF']*50)
    calibrators = ranker.fit_calibration(raw, target, lattice, np.ones(300), minimum=100)
    assert set(calibrators) == {ranker.POOLED, 'aP'}
    out = ranker.apply_calibration(np.array([0.9, 0.1]), np.array(['cF', 'cF']), calibrators)
    np.testing.assert_allclose(out, [1.0, 0.0])
