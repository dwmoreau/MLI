"""S14's network: the guards, the one-hot, the numpy path, and the round trip.

Everything here is synthetic and pool-free. The composition guard is tested against the SHAPE of
campaign 1's failure (F-121: a batch that has decayed to one lattice, and a loss that is
arithmetically impossible for the model's own predictions), not against its symptom.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training import FomCombiner  # noqa: E402
from mlindex.model_training import NeuralScore  # noqa: E402

pytest.importorskip('torch')

LATTICES = ('cP', 'tP', 'hP', 'oP', 'mP', 'aP')
GROUPS = ('structural', 'prior_entry', 'prior_volume', 'assignment_peaks')
DROP = tuple(name for name in FomCombiner.STRUCTURAL_NUMERIC if name != 'log_volume') \
    + ('spacegroup',)


def _frame(n_entries=60, per_entry=40, seed=0, lattices=LATTICES, positive_rate=0.1):
    """Candidates whose correctness is a smooth function of the per-peak posteriors."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for entry in range(n_entries):
        for candidate in range(per_entry):
            lattice = lattices[rng.integers(len(lattices))]
            n_peaks = 10 if lattice.startswith('c') else 20
            quality = rng.beta(2, 5)
            peaks = np.clip(rng.normal(quality, 0.15, size=20), 0, 1)
            peaks[n_peaks:] = np.nan
            correct = rng.random() < (0.02 + 0.9*quality**3)
            row = dict(entry_id=f'e{entry}', condition_bundle='c2_x', bravais_lattice=lattice,
                       candidate_id=candidate, is_correct=bool(correct),
                       sampling_weight=1.0 if correct else float(rng.choice([1.0, 20.0])),
                       log_volume=rng.normal(6.5, 0.5), asg_sigma=rng.normal(-6, 1))
            for index, value in enumerate(peaks):
                row[f'asg_p{index:02d}'] = value
            for name in FomCombiner.PRIOR_ENTRY:
                row[name] = rng.random()
            for name in FomCombiner.PRIOR_VOLUME:
                row[name] = np.nan if name.endswith(('cP', 'cI', 'cF')) else rng.normal(6.5, 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def _fit(frame, **overrides):
    options = dict(groups=GROUPS, drop=DROP, seed=1, epochs=20, batch_size=256, hidden=(16, 8),
                   learning_rate=1e-2)
    options.update(overrides)
    return NeuralScore.NeuralScore.fit(frame, **options)


def test_fit_learns_scores_and_records_its_guards():
    frame = _frame()
    model = _fit(frame)
    score = model.raw_score(frame)
    assert score.shape == (frame.shape[0],)
    assert ((score >= 0) & (score <= 1)).all()
    meta = model.meta
    assert meta['model_type'] == 'neural_score'
    assert meta['n_rows'] == frame.shape[0]
    assert meta['n_positive'] == int(frame['is_correct'].sum())
    assert meta['train_auc'] > 0.75
    assert len(meta['composition']) == meta['epochs_run'] >= 1
    for record in meta['composition']:
        assert record['lattices_present'] == len(LATTICES)
        assert 0 < record['positive_fraction'] < 1
    assert 1/3 < meta['loss_check']['ratio'] < 3
    # the columns with a NaN in training got an indicator: the ten cubic-empty peaks and the
    # three cubic volume readouts
    assert set(meta['indicator_names']) == (
        {f'asg_p{index:02d}' for index in range(10, 20)}
        | {'prior_logv_cP', 'prior_logv_cI', 'prior_logv_cF'})


def test_the_lattice_is_a_one_hot_block_never_an_ordinal():
    frame = _frame(n_entries=20)
    model = _fit(frame, epochs=1)
    matrix = model.design_matrix(frame)
    expanded = model.expand(matrix)
    width = len(model.categories['bravais_lattice'])
    assert width == len(LATTICES)
    block = expanded[:, -width:]
    np.testing.assert_array_equal(block.sum(axis=1), 1.0)
    assert set(np.unique(block)) == {0.0, 1.0}
    # an unseen lattice is all zeros, not the first bin and not an out-of-range code
    other = frame.head(3).assign(bravais_lattice='hR')
    block_other = model.expand(model.design_matrix(other))[:, -width:]
    np.testing.assert_array_equal(block_other, 0.0)
    # the ordinal code never reaches the network: the standardised numerics exclude it
    assert expanded.shape[1] == model.input_width
    assert 'bravais_lattice' not in [model.names[j] for j in model.numeric_indices]


def test_save_and_load_reproduce_the_scores_bit_for_bit(tmp_path):
    frame = _frame(n_entries=20)
    model = _fit(frame, epochs=2)
    model.fit_calibrators(frame, minimum=50)
    model.save(tmp_path/'net')
    loaded = NeuralScore.NeuralScore.load(tmp_path/'net')
    np.testing.assert_array_equal(loaded.raw_score(frame), model.raw_score(frame))
    np.testing.assert_array_equal(loaded.score(frame), model.score(frame))
    assert loaded.meta['n_rows'] == model.meta['n_rows']
    assert loaded.groups == tuple(GROUPS)
    assert (tmp_path/'net'/'neural.npz').exists()
    assert not (tmp_path/'net'/'model.joblib').exists()


def test_load_any_dispatches_on_the_recorded_model_type(tmp_path):
    frame = _frame(n_entries=20)
    net = _fit(frame, epochs=1)
    net.save(tmp_path/'net')
    assert isinstance(NeuralScore.load_any(tmp_path/'net'), NeuralScore.NeuralScore)
    tree = FomCombiner.FomCombiner.fit(frame, groups=GROUPS, drop=DROP, seed=1,
                                       weight_column='sampling_weight', max_iter=5)
    tree.save(tmp_path/'tree')
    assert isinstance(NeuralScore.load_any(tmp_path/'tree'), FomCombiner.FomCombiner)
    with pytest.raises(ValueError):
        NeuralScore.NeuralScore.load(tmp_path/'tree')


def test_the_composition_guard_fails_on_campaign_ones_decayed_batch():
    """F-121's batch: 6 000 rows of one lattice, both classes present, row count as expected."""
    target = np.array([0, 1]*3000)
    with pytest.raises(NeuralScore.CompositionError):
        NeuralScore.check_composition(target, np.full(6000, 5), 'epoch 2')
    with pytest.raises(NeuralScore.CompositionError):
        NeuralScore.check_composition(np.ones(100), np.arange(100) % 14, 'one class')
    NeuralScore.check_composition(target, np.arange(6000) % 14, 'healthy')


def test_fit_refuses_a_frame_that_has_lost_its_population():
    frame = _frame(n_entries=20, lattices=('mP',))
    with pytest.raises(NeuralScore.CompositionError):
        _fit(frame, epochs=1)
    all_correct = _frame(n_entries=20).assign(is_correct=True)
    with pytest.raises(NeuralScore.CompositionError):
        _fit(all_correct, epochs=1)


def test_the_loss_check_refuses_an_impossible_loss():
    with pytest.raises(NeuralScore.CompositionError):
        NeuralScore.check_loss_is_possible(0.05, 0.6)
    with pytest.raises(NeuralScore.CompositionError):
        NeuralScore.check_loss_is_possible(2.0, 0.3)
    assert NeuralScore.check_loss_is_possible(0.31, 0.30) == pytest.approx(0.31/0.30)
    with pytest.raises(NeuralScore.CompositionError):
        NeuralScore.check_not_constant(np.full(50, 0.3), np.arange(50) % 2)


def test_weighted_log_loss_is_the_bce_the_loop_minimises():
    p = np.array([0.9, 0.2, 0.6])
    t = np.array([1.0, 0.0, 1.0])
    w = np.array([1.0, 1.0, 2.0])
    expected = (-np.log(0.9) - np.log(0.8) - 2*np.log(0.6))/4
    assert NeuralScore.weighted_log_loss(p, t, w) == pytest.approx(expected)


def test_no_truth_column_can_enter_the_design():
    frame = _frame(n_entries=10)
    model = _fit(frame, epochs=1)
    assert not (set(model.names) & FomCombiner.FORBIDDEN_COLUMNS)
    assert 'is_correct' not in model.names and 'sampling_weight' not in model.names
    with pytest.raises(ValueError):
        FomCombiner.check_no_leakage(model.names + ('sampling_weight',))


def test_unweighted_fit_is_an_explicit_choice():
    frame = _frame(n_entries=20)
    model = _fit(frame, epochs=1, weight_column=None)
    assert model.meta['weight_column'] is None
    assert model.meta['weight_sum'] == frame.shape[0]
