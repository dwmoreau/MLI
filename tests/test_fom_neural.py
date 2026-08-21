"""S11 block C: the two joins, the leakage guard, and the identity with what session 2 measured.

Three of these are worth more than the rest, and each pins a way this stage could report a clean
number while being wrong.

The **leakage guard** is asserted on the real manifest rather than a fixture, because a fixture
cannot reproduce an overlap that exists only in the real files (F-101, and the same reasoning as
`test_fom_assignment`). It is asserted in both directions: that every reported entry was held out
of block A, *and* that block A's held-out set is non-empty -- without the second the test passes
just as happily when the guard has silently stopped selecting anything.

The **join key** is four columns, not three. `candidate_id` is unique within an (entry, lattice)
and **not** within an entry: a three-column join of block B's tables onto the pool silently
returned 3 805 rows where 3 655 went in, which is a duplicated candidate presented as a match. The
production path uses `FomBenchmark.ZOO_KEY_COLUMNS` throughout; this pins that it must.

The **reproduction of session 2** is what says the full-pool generation computed the same statistic
F-131 measured, rather than something that merely correlates with it. F-131's +2.82 pp is the
result block C is built on; if `asg_sigma` here is a different number, nothing downstream means
what it says.
"""
import numpy as np
import os
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'mlindex', 'scripts'))

from mlindex.model_training import FomBenchmark as Bench  # noqa: E402
from mlindex.model_training import FomCombiner  # noqa: E402
from mlindex.utilities import FigureOfMerits as fom  # noqa: E402

pytest.importorskip('pyarrow')

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPOSITORY, 'docs', 'fom', 'artifacts', 'S02_mirror_manifest.parquet')
BENCHMARK = os.path.join(REPOSITORY, 'mlindex', 'data', 'fom_benchmark')
FEATURES = os.path.join(REPOSITORY, 'mlindex', 'data', 'fom_features')
NEURAL = os.path.join(REPOSITORY, 'mlindex', 'data', 'fom_neural')
PEAKS = os.path.join(REPOSITORY, 'mlindex', 'data', 'fom_assignment')
BUNDLE = 'error1_cont0'

needs_pool = pytest.mark.skipif(
    not os.path.isdir(BENCHMARK),
    reason='the frozen candidate pool is untracked; a fresh clone has none of it',
    )
needs_columns = pytest.mark.skipif(
    not os.path.exists(os.path.join(NEURAL, f'assignment_{BUNDLE}.parquet')),
    reason="block C's generated columns are untracked; run --stage assignment first",
    )
needs_manifest = pytest.mark.skipif(
    not os.path.exists(MANIFEST), reason='the frozen split manifest is in git-ignored docs/')


# ---------------------------------------------------------------------------------------
# The feature specification
# ---------------------------------------------------------------------------------------
def test_the_new_groups_leave_s08s_feature_set_byte_identical():
    """Adding a group must not redefine a number STATUS section 2 already reports."""
    assert FomCombiner.DEFAULT_GROUPS == ('raw', 'scaled', 'structural', 'context')
    for group in ('assignment', 'prior'):
        assert group in FomCombiner.FEATURE_GROUPS
        assert group not in FomCombiner.DEFAULT_GROUPS


def test_the_baseline_is_still_seventy_eight_features():
    """S08 + S10's columns. The comparison is only paired if the baseline is the published one."""
    scalers = FomCombiner.load_scalers(
        os.path.join(REPOSITORY, 'mlindex', 'models', 'fom_null'),
        groups=('raw', 'scaled', 'structural', 'context', 'in_sample', 'cv'))
    baseline, _ = FomCombiner.feature_specification(
        ('scaled', 'structural', 'context', 'cv'), scalers)
    assert len(baseline) == 78
    with_b, _ = FomCombiner.feature_specification(
        ('scaled', 'structural', 'context', 'cv', 'assignment'), scalers)
    with_both, _ = FomCombiner.feature_specification(
        ('scaled', 'structural', 'context', 'cv', 'assignment', 'prior'), scalers)
    assert len(with_b) == 78 + len(FomCombiner.ASSIGNMENT_MERITS)
    assert len(with_both) == 78 + len(FomCombiner.ASSIGNMENT_MERITS) + len(
        FomCombiner.PRIOR_MERITS)
    # A group-drop has to remove the whole family at once or the ablation measures nothing.
    assert set(baseline).isdisjoint(FomCombiner.ASSIGNMENT_MERITS)
    assert set(baseline).isdisjoint(FomCombiner.PRIOR_MERITS)


def test_check_no_leakage_still_rejects_a_truth_column_in_a_new_group():
    """The deny-list catches a feature somebody added on purpose. It must still be reachable."""
    with pytest.raises(ValueError):
        FomCombiner.check_no_leakage(
            list(FomCombiner.ASSIGNMENT_MERITS) + ['volume_ratio_to_truth'])
    with pytest.raises(ValueError):
        FomCombiner.check_no_leakage(list(FomCombiner.PRIOR_MERITS) + ['is_correct'])
    # And the new names themselves must pass, or the fit path raises on every call.
    FomCombiner.check_no_leakage(
        list(FomCombiner.ASSIGNMENT_MERITS) + list(FomCombiner.PRIOR_MERITS))


def test_affordable_features_can_drop_a_new_column_by_name():
    """S14's cost-limited variant has to be able to delete these, like any other merit."""
    names = list(FomCombiner.ASSIGNMENT_MERITS) + ['M20', 'ctx_pool_size']
    kept = FomCombiner.affordable_features(names, allowed_merits=('M20',))
    assert 'M20' in kept and 'ctx_pool_size' in kept
    assert not set(kept) & set(FomCombiner.ASSIGNMENT_MERITS)


def test_an_external_group_without_its_directory_raises_rather_than_yielding_nothing():
    """A missing directory must not silently produce a frame with the columns absent."""
    with pytest.raises(ValueError, match='needs a directory'):
        FomCombiner._merge_external(
            None, list(Bench.ZOO_KEY_COLUMNS), 'prior', 'prior',
            FomCombiner.PRIOR_MERITS, None, BUNDLE)


# ---------------------------------------------------------------------------------------
# The leakage guard, on the real files
# ---------------------------------------------------------------------------------------
@needs_manifest
@needs_pool
def test_every_reported_entry_was_held_out_of_block_a():
    import pandas as pd

    from mlindex.model_training import PriorNetwork as Prior

    held_out = Prior.held_out_identifiers(MANIFEST)
    assert len(held_out) > 0, 'the guard selected nothing, so it is asserting nothing'

    entries = Bench.load_entries(BENCHMARK)[['entry_id', 'split']].drop_duplicates()
    reported = set(entries.loc[entries['split'] == 'fom-dev', 'entry_id'])
    assert reported, 'no fom-dev entries in the pool'
    assert not reported - held_out

    # And the converse, which is the half that catches a guard that has stopped selecting: the
    # entries block A *was* free to train on must not include any of them.
    trainable = set(entries.loc[entries['split'] == 'fom-train', 'entry_id'])
    assert trainable and not trainable & held_out


@needs_manifest
@needs_pool
def test_the_base_rate_is_estimated_on_the_training_split_only():
    import run_fom_neural as neural

    entries = Bench.load_entries(BENCHMARK)
    dev_ids = set(entries.loc[entries['split'] == 'fom-dev', 'entry_id'])

    on_train = neural.base_rate_by_lattice(BENCHMARK, [BUNDLE], 'fom-train')
    on_dev = neural.base_rate_by_lattice(BENCHMARK, [BUNDLE], 'fom-dev')
    assert set(on_train) and set(on_train) == set(on_dev)
    # If the split filter were a no-op the two would be identical. They are estimates of the same
    # quantity on disjoint entries, so they must be close but not equal.
    assert any(abs(on_train[k] - on_dev[k]) > 1e-9 for k in on_train)
    assert dev_ids


# ---------------------------------------------------------------------------------------
# The joins
# ---------------------------------------------------------------------------------------
@needs_pool
@needs_columns
def test_the_generated_columns_cover_the_feature_matrix_one_to_one():
    """A left join hides its own misses; a column that is all-NaN reports a delta of exactly zero.

    Which reads like a clean negative and is a plumbing failure -- F-121's lesson applied to a
    merge rather than a batch.
    """
    import pandas as pd

    keys = list(Bench.ZOO_KEY_COLUMNS)
    features = pd.read_parquet(os.path.join(FEATURES, f'features_{BUNDLE}.parquet'),
                               columns=keys)
    for prefix, columns in (('assignment', FomCombiner.ASSIGNMENT_MERITS),
                            ('prior', FomCombiner.PRIOR_MERITS)):
        path = os.path.join(NEURAL, f'{prefix}_{BUNDLE}.parquet')
        if not os.path.exists(path):
            pytest.skip(f'{prefix} columns not generated')
        frame = pd.read_parquet(path)
        assert not frame.duplicated(subset=keys).any()
        merged = features.merge(frame, on=keys, how='left', validate='1:1')
        assert len(merged) == len(features)
        for column in columns:
            assert merged[column].notna().all(), f'{column} did not cover the feature matrix'


@needs_pool
@needs_columns
def test_the_three_column_key_is_not_enough_and_the_four_column_one_is():
    """`candidate_id` repeats across Bravais lattices within one entry.

    A three-column join therefore matches a candidate against a different lattice's candidate and
    returns more rows than it was given. This is not hypothetical -- it happened while building
    this stage, on the first cross-check that was run.
    """
    import pandas as pd

    frame = pd.read_parquet(os.path.join(NEURAL, f'assignment_{BUNDLE}.parquet'))
    assert not frame.duplicated(subset=list(Bench.ZOO_KEY_COLUMNS)).any()
    assert frame.duplicated(subset=['entry_id', 'condition_bundle', 'candidate_id']).any()


# ---------------------------------------------------------------------------------------
# The identity with session 2
# ---------------------------------------------------------------------------------------
@needs_pool
@needs_columns
@pytest.mark.parametrize('lattice', ('cP', 'mP', 'aP'))
def test_the_full_pool_columns_reproduce_what_session_two_measured(lattice):
    """F-131's +2.82 pp is the result block C is built on. Same statistic, or nothing means what
    it says."""
    import pandas as pd

    from mlindex.utilities.FigureOfMerits import N_FREE_PARAMETERS

    path = os.path.join(PEAKS, f'peaks_{lattice}_fom-dev.parquet')
    if not os.path.exists(path):
        pytest.skip('block B peak tables not present')
    keys = list(Bench.ZOO_KEY_COLUMNS)
    peaks = pd.read_parquet(path, columns=keys + ['q2_obs', 'q2_calc', 'posterior'])
    peaks = peaks.loc[peaks['condition_bundle'] == BUNDLE]
    if not len(peaks):
        pytest.skip('no rows for this bundle')
    peaks['squared'] = (peaks['q2_obs'] - peaks['q2_calc'])**2
    peaks['log_posterior'] = np.log(np.clip(peaks['posterior'], 1e-12, 1.0))
    summary = peaks.groupby(keys, sort=False).agg(
        residual=('squared', 'sum'), n=('squared', 'size'),
        post_n=('posterior', 'sum'), post_l=('log_posterior', 'mean'),
        ).reset_index()

    system = Bench.load_candidates(
        BENCHMARK, bravais_lattices=[lattice], bundles=[BUNDLE],
        columns=['lattice_system'])['lattice_system'].iloc[0]
    n_free = N_FREE_PARAMETERS[system]
    summary['reference_sigma'] = np.log(
        np.sqrt(summary['residual']/np.maximum(summary['n'] - n_free, 1)) + 1e-12)

    generated = pd.read_parquet(os.path.join(NEURAL, f'assignment_{BUNDLE}.parquet'))
    merged = summary.merge(generated, on=keys, how='inner', validate='1:1')
    assert len(merged) == len(summary), 'the full-pool pass does not cover session 2s rows'

    # The posteriors are the same arithmetic and must agree to machine precision.
    assert np.allclose(merged['post_n'], merged['asg_post_n'], atol=1e-12)
    assert np.allclose(merged['post_l'], merged['asg_post_l'], atol=1e-12)
    # `asg_sigma` uses the *nearest* line where session 2's summary used the *assigned* one, which
    # differ for the handful of peaks where `fast_assign` does not take the nearest. On a log scale
    # the disagreement is bounded well below anything that could move a fitted number.
    assert np.abs(merged['reference_sigma'] - merged['asg_sigma']).max() < 1e-3


@needs_pool
@needs_columns
def test_the_prior_column_reproduces_the_shipped_score_candidates():
    """`prior_joint` must *be* `score_candidates`, not a reimplementation that agrees on average.

    Not asserted as exact equality, and the reason is worth knowing before someone tries to
    reproduce a number from this stage to the last bit. The network is bit-deterministic for a
    **fixed batch composition** -- the same call twice returns identical arrays -- but a float32
    matmul reassociates when the batch shape changes, so scoring the same candidate inside a
    different batch moves its log-probability in the last bits. Measured here: 192 of 200
    candidates identical, worst disagreement 1.6e-6 on a log-probability of order -15. That is
    seven orders of magnitude below anything a fitted number can see, and it is a property of the
    arithmetic rather than of the code.
    """
    import pandas as pd

    from mlindex.model_training import PriorNetwork as Prior

    prior_path = os.path.join(NEURAL, f'prior_{BUNDLE}.parquet')
    model_dir = os.path.join(REPOSITORY, 'mlindex', 'models', 'fom_prior', 'main', 'global')
    if not (os.path.exists(prior_path) and os.path.isdir(model_dir)):
        pytest.skip('block A columns or weights not present')

    frame = pd.read_parquet(prior_path).sample(200, random_state=0)
    pool = Bench.load_candidates(
        BENCHMARK, bundles=[BUNDLE], columns=['entry_id', 'candidate_id', 'bravais_lattice',
                                              'volume'])
    merged = frame.merge(pool, on=['entry_id', 'bravais_lattice', 'candidate_id'], how='inner',
                         validate='1:1')
    entries = Bench.load_entries(BENCHMARK)
    entries = entries.loc[entries['condition_bundle'] == BUNDLE].set_index('entry_id')['q2_obs']

    model = Prior.PriorNetwork.load_prior(model_dir)
    codes = Prior.target_codes(merged['bravais_lattice'].to_numpy())
    reference = model.score_candidates(
        np.stack([np.asarray(entries.loc[e], dtype=np.float64) for e in merged['entry_id']]),
        merged['volume'].to_numpy(), codes['bravais'], target='bravais',
        )
    generated = merged['prior_joint'].to_numpy()
    assert np.abs(reference - generated).max() < 1e-4
    # Most of them *are* identical; a systematic offset would not look like this.
    assert np.mean(reference == generated) > 0.5


def test_the_multi_head_joint_matches_the_single_head_one():
    """One forward pass for five heads must return exactly what five passes returned."""
    from mlindex.model_training import PriorNetwork as Prior

    model_dir = os.path.join(REPOSITORY, 'mlindex', 'models', 'fom_prior', 'main', 'global')
    if not os.path.isdir(model_dir):
        pytest.skip('block A weights not present')
    model = Prior.PriorNetwork.load_prior(model_dir)
    rng = np.random.default_rng(0)
    q2 = np.sort(rng.uniform(0.01, 0.5, size=(4, 20)), axis=1)
    tables = model.joint_log_probabilities(q2)
    for target in ('bravais', 'system', 'centring'):
        assert np.array_equal(tables[target], model.joint_log_probability(q2, target=target))
    # Every head's joint is a normalised distribution over the whole (branch, class) table.
    for target in ('bravais', 'system', 'centring', 'n_free', 'high_symmetry'):
        total = np.exp(tables[target]).sum(axis=(1, 2))
        assert np.allclose(total, 1.0, atol=1e-9)


def test_the_posterior_is_unchanged_when_its_inputs_are_passed_in():
    """The reuse that halves the generation cost must not change a single value."""
    rng = np.random.default_rng(1)
    q2_obs = np.sort(rng.uniform(0.05, 0.4, size=12))
    q2_ref_calc = np.sort(rng.uniform(0.05, 0.4, size=(5, 200)), axis=1)
    fresh = fom.get_assignment_posterior(q2_obs, q2_ref_calc, 'monoclinic')
    sigma, d1 = fom.get_assignment_sigma(q2_obs, q2_ref_calc, 'monoclinic')
    reused = fom.get_assignment_posterior(q2_obs, q2_ref_calc, 'monoclinic', sigma=sigma, d1=d1)
    assert np.array_equal(fresh, reused)


# ---------------------------------------------------------------------------------------
# The network, and the two checks S11 session 1 paid three runs to learn it needs
# ---------------------------------------------------------------------------------------
def _synthetic_frame(n=4000, seed=0):
    """A frame carrying exactly the `structural` group, with a learnable label."""
    import pandas as pd

    from mlindex.model_training import FomCombiner as C

    rng = np.random.default_rng(seed)
    columns = {}
    for name in C.STRUCTURAL_NUMERIC:
        columns[name] = rng.normal(size=n)
    columns['bravais_lattice'] = rng.choice(['cP', 'mP', 'aP'], size=n)
    columns['spacegroup'] = rng.choice(['P', 'C', 'I'], size=n)
    frame = pd.DataFrame(columns)
    signal = frame['n_peaks'] + 0.5*frame['n_indexed'] - 0.5*frame['hkl_ref_length']
    frame['is_correct'] = signal > np.quantile(signal, 0.95)
    frame['entry_id'] = np.repeat(np.arange(n//20), 20)[:n]
    frame['condition_bundle'] = 'error1_cont0'
    return frame


def test_the_network_fits_predicts_and_round_trips(tmp_path):
    from mlindex.model_training import NeuralFom as Neural

    frame = _synthetic_frame()
    model = Neural.NeuralFom.fit([frame], groups=('structural',), seed=0, hidden=(16, 8),
                                 max_iter=40, batch_size=256)
    scores = model.raw_score(frame)
    assert scores.shape == (frame.shape[0],)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    # It has to have learned *something*, or every downstream comparison is against noise.
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(frame['is_correct'].to_numpy(), scores) > 0.75

    assert model.meta['trained']['train_roc_auc'] > 0.75
    model.save(tmp_path/'net')
    reloaded = Neural.NeuralFom.load(tmp_path/'net')
    assert reloaded.names == model.names
    assert np.allclose(reloaded.raw_score(frame), scores)


def test_the_composition_check_catches_a_training_set_that_lost_its_positives():
    """F-121: a shadowed variable decayed the training set and the loss never said so."""
    from mlindex.model_training import NeuralFom as Neural

    matrix = np.zeros((100, 3))
    with pytest.raises(Neural.CompositionError, match='one class only'):
        Neural.check_composition(matrix, np.zeros(100, dtype=np.int32))
    with pytest.raises(Neural.CompositionError, match='rows against'):
        Neural.check_composition(matrix, np.zeros(99, dtype=np.int32))
    target = np.zeros(100, dtype=np.int32)
    target[:2] = 1
    with pytest.raises(Neural.CompositionError, match='outside'):
        Neural.check_composition(matrix, target, expected_positive_rate=0.5)
    assert Neural.check_composition(matrix, target, expected_positive_rate=0.02)


def test_the_constant_check_catches_a_model_that_never_started():
    """The guard that caught the first fit of this class: accuracy-based early stopping stopped it
    at iteration 10 with AUC 0.46 while reporting a validation score of 0.95."""
    from mlindex.model_training import NeuralFom as Neural

    rng = np.random.default_rng(0)
    target = (rng.random(2000) < 0.05).astype(np.int32)
    with pytest.raises(Neural.CompositionError, match='constant'):
        Neural.check_not_constant(np.full(2000, 0.05), target)
    with pytest.raises(Neural.CompositionError, match='not a model that has been trained'):
        Neural.check_not_constant(rng.random(2000), target)
    informative = rng.random(2000) + 0.6*target
    assert Neural.check_not_constant(informative, target)['train_roc_auc'] > 0.55


def test_the_loss_check_catches_a_loss_its_own_predictions_could_not_produce():
    """The other half of F-121: the reported loss was arithmetically impossible for three runs."""
    from mlindex.model_training import NeuralFom as Neural

    class _Liar:
        loss_ = 1e-6

        def predict_proba(self, matrix):
            half = np.full((matrix.shape[0], 2), 0.5)
            return half

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 3))
    target = (rng.random(200) > 0.5).astype(np.int32)
    with pytest.raises(Neural.CompositionError, match='not consistent'):
        Neural.check_loss_is_possible(_Liar(), matrix, target)

    class _Honest(_Liar):
        loss_ = float(np.log(2.0))

    assert Neural.check_loss_is_possible(_Honest(), matrix, target)['observed_loss'] > 0
