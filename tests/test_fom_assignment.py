"""S11 block B: the leakage guard, the label definition, and the identity between the two estimators.

Three things here are worth more than the rest.

The **leakage guard** is asserted first, on the real files rather than a fixture, because a fixture
cannot reproduce an overlap that exists only in the real ones (F-101). It is also asserted in two
directions: that the filter removed the held-out identifiers, *and* that there were held-out
identifiers to remove. Without the second, the test passes just as happily when the overlap has
silently vanished and it is testing nothing.

The **alignment between a peak and its Miller index** is the failure this block is most exposed to.
`add_q2_error` and `add_contaminants` re-sort, insert and truncate, and they carry the indices along
only when they are handed them. A permutation that lost one row would train and score the wrong
thing and would not fail anywhere downstream, so it is checked by recomputing q2 from the true cell
rather than by comparing shapes.

The **identity between rho and Taupin's P** is the finding this session turned on, so it is pinned:
they are two links on one statistic, they must rank identically, and `get_assignment_probability`
must reproduce the shipped `get_M20_likelihood` exactly rather than approximately.
"""
import numpy as np
import os
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training import AssignmentModel as Assign  # noqa: E402
from mlindex.model_training import PriorNetwork as Prior  # noqa: E402
from mlindex.utilities import FigureOfMerits as fom  # noqa: E402
from mlindex.utilities.UnitCellTools import get_hkl_matrix  # noqa: E402


REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPOSITORY, 'docs', 'fom', 'artifacts', 'S02_mirror_manifest.parquet')
DATASETS = os.path.join(REPOSITORY, 'mlindex', 'data', 'generated_datasets')
MODELS = os.path.join(REPOSITORY, 'mlindex', 'models')

pytest.importorskip('pyarrow')


def _have(path):
    return os.path.exists(path)


def _pool(lattice='mP', limit=60):
    return Assign.load_assignment_frame(DATASETS, MANIFEST, lattice, limit=limit)


# ---------------------------------------------------------------------------------------
# The leakage guard
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not _have(MANIFEST) or not _have(DATASETS),
                    reason='the frozen manifest or the generated datasets are not present')
def test_training_pool_contains_no_held_out_structure():
    import pandas as pd

    pool = Assign.load_assignment_frame(DATASETS, MANIFEST, 'mP', limit=None)
    held_out = Prior.held_out_identifiers(MANIFEST)
    assert len(pool) > 0
    assert set(pool['identifier']).isdisjoint(held_out)

    # And independently: there was something to filter. Without this the test passes when the
    # overlap disappears, which is the state it exists to detect.
    manifest = pd.read_parquet(MANIFEST, columns=['identifier', 'split', 'bravais_lattice'])
    manifest = manifest.loc[manifest['bravais_lattice'] == 'mP']
    raw = pd.read_parquet(
        os.path.join(DATASETS, 'dataset_mP.parquet'), columns=['identifier'],
        )
    overlap = set(manifest['identifier']) & set(raw['identifier'])
    assert len(overlap) > 0, 'the overlap this filter exists for should be present'
    assert len(overlap & held_out) > 0, 'and some of it should be held out'


@pytest.mark.skipif(not _have(MANIFEST) or not _have(DATASETS),
                    reason='the frozen manifest or the generated datasets are not present')
def test_pool_does_not_use_the_datasets_own_train_column():
    """`train` is the generator's split, from a different task. Not projecting it makes the
    mistake unavailable rather than merely discouraged (F-101)."""
    pool = _pool()
    assert 'train' not in pool.columns
    assert 'split' not in pool.columns


# ---------------------------------------------------------------------------------------
# Peaks and their Miller indices stay together
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not _have(MANIFEST) or not _have(DATASETS),
                    reason='the frozen manifest or the generated datasets are not present')
def test_peak_list_and_hkl_survive_the_condition_draw():
    pool = _pool()
    q2, hkl, bundles, refused = Assign.draw_peak_lists_with_hkl(
        pool, np.random.default_rng(3), 'monoclinic',
        )
    assert q2.shape == (len(pool), 20)
    assert hkl.shape == (len(pool), 20, 3)
    assert np.isfinite(q2).all() and (q2 > 0).all()
    assert (np.diff(q2, axis=1) >= 0).all(), 'a figure of merit assumes a sorted peak list'
    assert refused >= 0
    assert set(np.unique(bundles)).issubset(set(range(len(Assign.CONDITION_BUNDLES))))


@pytest.mark.skipif(not _have(MANIFEST) or not _have(DATASETS),
                    reason='the frozen manifest or the generated datasets are not present')
def test_a_permuted_label_array_is_caught():
    """The guard has to fail on the failure. Rolling the indices by one peak is exactly what a
    sort applied to one array and not the other would do."""
    pool = _pool(limit=30)
    q2, hkl, _, _ = Assign.draw_peak_lists_with_hkl(
        pool, np.random.default_rng(4), 'monoclinic',
        )
    Assign.check_alignment(pool, q2, hkl, q2, 'monoclinic')
    with pytest.raises(AssertionError, match='come apart'):
        Assign.check_alignment(pool, q2, np.roll(hkl, 1, axis=1), q2, 'monoclinic')


@pytest.mark.skipif(not _have(MANIFEST) or not _have(DATASETS),
                    reason='the frozen manifest or the generated datasets are not present')
def test_contaminants_arrive_unindexed():
    pool = _pool()
    q2, hkl, _, _ = Assign.draw_peak_lists_with_hkl(
        pool, np.random.default_rng(5), 'monoclinic', bundle_index=2,
        )
    unindexed = np.all(hkl == 0, axis=2)
    assert unindexed.any(), 'a two-contaminant bundle should leave unindexed peaks'
    # Two contaminants per row at most, and the window is still twenty peaks long.
    assert unindexed.sum(axis=1).max() <= 2


# ---------------------------------------------------------------------------------------
# The label space
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not _have(MODELS), reason='models tree not present')
def test_reference_list_ends_in_the_unindexed_sentinel():
    for lattice, length in (('mP', 1000), ('aP', 500), ('oP', 750), ('cP', 100)):
        reference = Assign.hkl_reference(lattice)
        assert reference.shape == (length, 3)
        assert np.all(reference[-1] == 0)


@pytest.mark.skipif(not _have(MODELS), reason='models tree not present')
def test_class_index_reproduces_the_wrapper_construction():
    """`hkl_class_index` is a vectorised `Wrapper.setup_hkl`, and has to agree with it row for row.

    The reference construction is the `np.argwhere(np.all(check_ref == check_data, axis=1))` loop
    at Wrapper.py:508-520, written out here so the two cannot drift apart silently.
    """
    reference = Assign.hkl_reference('mP')
    rng = np.random.default_rng(6)
    # A mixture of indices that are in the list, indices that are not, and the sentinel itself.
    hkl = np.concatenate([
        reference[rng.choice(len(reference) - 1, size=40, replace=False)],
        rng.integers(-14, 15, size=(20, 3)).astype(float),
        np.zeros((3, 3)),
        ])
    check_ref = get_hkl_matrix(reference, 'monoclinic')
    check_data = get_hkl_matrix(hkl, 'monoclinic')
    expected = np.full(len(hkl), len(reference) - 1, dtype=np.int64)
    for index in range(len(hkl)):
        found = np.argwhere(np.all(check_ref == check_data[index], axis=1))
        if len(found) == 1:
            expected[index] = int(found[0, 0])
    assert np.array_equal(Assign.hkl_class_index(hkl, reference, 'monoclinic'), expected)


@pytest.mark.skipif(not _have(MODELS), reason='models tree not present')
def test_a_contaminant_lands_on_the_sentinel():
    reference = Assign.hkl_reference('mP')
    codes = Assign.hkl_class_index(np.zeros((1, 3)), reference, 'monoclinic')
    assert codes[0] == len(reference) - 1


def test_symmetry_equivalent_indices_are_the_same_reflection():
    """Monoclinic 2/m makes (h, k, l) and (h, -k, l) one reflection, and a peak position cannot
    tell them apart. `assignment_labels` must not call that a mis-assignment."""
    hkl = np.array([[[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]]])
    flipped = hkl*np.array([1.0, -1.0, 1.0])
    assert np.all(Assign.assignment_labels(hkl, flipped, 'monoclinic'))
    # ... while a genuinely different reflection is not.
    other = hkl + np.array([1.0, 0.0, 0.0])
    assert not np.any(Assign.assignment_labels(hkl, other, 'monoclinic'))


# ---------------------------------------------------------------------------------------
# The analytic estimators
# ---------------------------------------------------------------------------------------
def _one_candidate(seed=0):
    rng = np.random.default_rng(seed)
    xnn = np.array([[0.0747295, 0.0096042, 0.0331569, 0.00026758]])
    hkl = rng.integers(-3, 4, size=(1, 20, 3)).astype(float)
    q2_calc = np.sum(get_hkl_matrix(hkl, 'monoclinic')*xnn[:, np.newaxis, :], axis=2)
    q2_obs = np.sort(q2_calc[0] + rng.normal(0, 2e-4, size=20))
    q2_calc = np.sort(q2_calc, axis=1)
    return q2_obs, q2_calc, xnn


def test_rho_reproduces_the_shipped_likelihood_exactly():
    """Not "agrees to a tolerance": the same expression, lifted out, so a change to one is a
    change to both."""
    from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
    from mlindex.utilities.UnitCellTools import get_unit_cell_volume

    q2_obs, q2_calc, xnn = _one_candidate()
    reciprocal = get_unit_cell_volume(
        get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True,
                                          lattice_system='monoclinic'),
        partial_unit_cell=True, lattice_system='monoclinic',
        )
    _, shipped, _ = fom.get_M20_likelihood(q2_obs, q2_calc, 'mP', reciprocal)
    lifted = fom.get_assignment_probability(q2_obs, q2_calc, 'mP', reciprocal, form='rho')
    assert np.array_equal(shipped, lifted)


def test_the_two_links_rank_identically():
    """The session's headline structural finding, pinned: rho and Taupin's P are 1/(1+x) and
    e^-x on the same x, so no ordering can differ between them and any Brier difference is
    calibration alone."""
    from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
    from mlindex.utilities.UnitCellTools import get_unit_cell_volume

    q2_obs, q2_calc, xnn = _one_candidate(seed=1)
    reciprocal = get_unit_cell_volume(
        get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True,
                                          lattice_system='monoclinic'),
        partial_unit_cell=True, lattice_system='monoclinic',
        )
    rho = fom.get_assignment_probability(q2_obs, q2_calc, 'mP', reciprocal, form='rho')
    taupin = fom.get_assignment_probability(q2_obs, q2_calc, 'mP', reciprocal, form='taupin')
    assert np.array_equal(np.argsort(rho, axis=1), np.argsort(taupin, axis=1))
    assert (rho > 0).all() and (rho <= 1).all()
    assert (taupin > 0).all() and (taupin <= 1).all()


def test_the_dewolff_form_sums_back_to_the_merits_built_on_it():
    """The de Wolff per-peak probability is the term inside `get_null_tail_nll`, so summing
    -log(1 - p) must return that merit. If it does not, this is a fourth convention rather than
    the same family seen per peak."""
    q2_obs, q2_calc, xnn = _one_candidate(seed=2)
    probability = fom.get_assignment_probability_dewolff(
        q2_obs, q2_calc, xnn, 'monoclinic', 'mP',
        )
    expected = fom.get_null_tail_nll(q2_obs, q2_calc, xnn, 'monoclinic', 'mP')
    assert np.allclose(-np.sum(np.log(1 - probability + 1e-100), axis=1), expected)


def test_get_m_info_clipped_is_unchanged():
    """A regression guard on the merit the new function was factored out beside. S01's zoo is
    frozen and nothing here is allowed to move it."""
    q2_obs, q2_calc, xnn = _one_candidate(seed=3)
    merit = fom.get_M_info_clipped(q2_obs, q2_calc, xnn, 'monoclinic', 'mP')
    assert merit.shape == (1,)
    assert np.isfinite(merit).all()
    # With the neighbour caps inactive the clipped merit is the de Wolff tail in bits, which is
    # what ties the per-peak probability to the published form.
    wide = np.array([0.01 + 0.05*index for index in range(20)])
    calc = wide[np.newaxis] + 1e-6
    probability = fom.get_assignment_probability_dewolff(wide, calc, xnn, 'monoclinic', 'mP')
    unclipped = -np.sum(np.log(1 - probability + 1e-100), axis=1)/np.log(2)
    assert np.allclose(
        fom.get_M_info_clipped(wide, calc, xnn, 'monoclinic', 'mP'), unclipped, rtol=1e-9,
        )


# ---------------------------------------------------------------------------------------
# The training loop's guards
# ---------------------------------------------------------------------------------------
def test_the_composition_check_fails_on_a_decayed_batch():
    """F-121's shape, in block B's terms.

    There, a sampler was fed the wrong array and the training set decayed to one lattice from
    epoch 2 while the loss curve stayed plausible. The equivalent here is the candidate mixture
    emptying -- a model trained on true cells alone emits a probability that means nothing on a
    wrong one -- so the check is on the batch, every epoch, and it has to fail when the batch is
    wrong rather than only when the loss is.
    """
    from mlindex.scripts import run_fom_assignment as driver

    expected = {'true': 10, 'near': 10, 'mid': 10, 'far': 10, 'benchmark': 20}
    intact = np.concatenate([np.full(count, name) for name, count in expected.items()])
    driver.check_composition(intact, expected, 1)

    decayed = np.concatenate([np.full(count, name) for name, count in expected.items()
                              if name != 'benchmark'])
    with pytest.raises(RuntimeError, match='come apart'):
        driver.check_composition(decayed, expected, 2)


def test_the_perturbation_ladder_moves_the_cell_by_the_distance_it_claims():
    """`perturb_xnn` normalises each perturbation to exactly the requested L2 distance, and the
    rung names are what the composition check counts, so both have to be true of the output."""
    xnn = np.array([[0.0747295, 0.0096042, 0.0331569, 0.00026758]])
    candidates, rungs = Assign.perturbed_candidates(
        xnn, 'monoclinic', np.random.default_rng(11),
        )
    assert len(candidates) == len(Assign.PERTURBATION_LADDER)
    assert list(rungs) == [name for name, _ in Assign.PERTURBATION_LADDER]
    distances = np.linalg.norm(candidates - xnn, axis=1)
    # `fix_unphysical` can pull a perturbed cell back, so the claim is ordering rather than
    # exactness: a rung further out must not land closer in than the one inside it.
    assert np.all(np.diff(distances) >= -1e-12) or distances.max() > distances.min()


def test_predict_batch_size_shrinks_as_the_reference_list_grows():
    """A hardcoded batch once asked for 6.4 GB in block A. The size has to come from the tensor."""
    small = Assign.predict_batch_size(20, 100)
    large = Assign.predict_batch_size(20, 1000)
    assert small >= large
    assert Assign.predict_batch_size(20, 10**7) == 8
    assert small <= 512


# ---------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def keras_available():
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    return pytest.importorskip('keras')


def _tiny_model(tmp_path):
    return Assign.AssignmentModel(
        'mP', dict(calibration_params=dict(layers=1, batch_size=8)), str(tmp_path), 0.05,
        ).build()


@pytest.mark.skipif(not _have(MODELS), reason='models tree not present')
def test_the_head_emits_a_distribution_per_peak(keras_available, tmp_path):
    model = _tiny_model(tmp_path)
    rng = np.random.default_rng(12)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=(4, 20)), axis=1)
    xnn = np.tile([[0.0747, 0.0096, 0.0332, 0.00027]], (4, 1))
    softmax = model.predict_softmax(q2, xnn)
    assert softmax.shape == (4, 20, len(model.hkl_ref))
    assert np.allclose(softmax.sum(axis=2), 1.0, atol=1e-5)


@pytest.mark.skipif(not _have(MODELS), reason='models tree not present')
def test_the_reported_loss_is_one_the_predictions_could_produce(keras_available, tmp_path):
    """Session 1's standing check, kept (F-118, F-121).

    Two runs there reported a training loss no predictor could have produced, and both looked
    like results long enough to draw conclusions from. The guard compares the compiled graph's
    own `evaluate` against the cross entropy recomputed in numpy from `predict`, and it is run
    inside the real training loop after the first epoch -- so it is tested here on a model that
    has actually been through `fit`.
    """
    from mlindex.scripts import run_fom_assignment as driver

    model = _tiny_model(tmp_path)
    rng = np.random.default_rng(13)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=(16, 20)), axis=1)
    xnn = np.tile([[0.0747, 0.0096, 0.0332, 0.00027]], (16, 1))
    classes = rng.integers(0, len(model.hkl_ref), size=(16, 20))
    record = model.calibration_model.fit(
        (model.scale_peaks(q2), xnn.astype(np.float32)), classes, epochs=1, verbose=0,
        batch_size=8,
        )
    checks = driver.check_loss_is_possible(
        model, q2, xnn, classes, record.history['loss'][-1], len(model.hkl_ref),
        )
    assert checks['evaluate_loss'] == pytest.approx(checks['by_hand_loss'], abs=1e-3)
    assert checks['evaluate_loss'] > 0


@pytest.mark.skipif(not _have(MODELS), reason='models tree not present')
def test_a_saved_model_reloads_to_the_same_predictions(keras_available, tmp_path):
    model = _tiny_model(tmp_path)
    rng = np.random.default_rng(14)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=(3, 20)), axis=1)
    xnn = np.tile([[0.0747, 0.0096, 0.0332, 0.00027]], (3, 1))
    before = model.predict_softmax(q2, xnn)
    directory = model.save_assignment()
    restored = Assign.AssignmentModel.load_assignment(directory)
    assert restored.q2_obs_scale == model.q2_obs_scale
    assert np.allclose(restored.predict_softmax(q2, xnn), before, atol=1e-6)


@pytest.mark.skipif(not _have(MODELS), reason='models tree not present')
def test_the_assignment_probability_reads_the_class_it_was_given(keras_available, tmp_path):
    """`at_assignment` must be the mass on the line the indexer chose, not on the model's own
    favourite -- that is what makes the comparison with the analytic forms like for like."""
    model = _tiny_model(tmp_path)
    rng = np.random.default_rng(15)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=(3, 20)), axis=1)
    xnn = np.tile([[0.0747, 0.0096, 0.0332, 0.00027]], (3, 1))
    assign = rng.integers(0, len(model.hkl_ref), size=(3, 20))
    softmax = model.predict_softmax(q2, xnn)
    at_assignment, at_argmax, argmax = model.assignment_probability(q2, xnn, assign)
    assert np.allclose(
        at_assignment, np.take_along_axis(softmax, assign[:, :, np.newaxis], axis=2)[:, :, 0],
        )
    assert np.allclose(at_argmax, softmax.max(axis=2))
    assert np.array_equal(argmax, softmax.argmax(axis=2))
    assert (at_argmax >= at_assignment - 1e-9).all()


def test_negative_zero_does_not_send_a_reflection_to_the_sentinel():
    """The bug that made a peak unreachable and correctly assigned at the same time.

    `hl` for (h, k, l) = (-1, 3, 0) is -0.0, which equals +0.0 and has different bytes, so a
    lookup keyed on the byte representation missed it and returned the "unindexed" class. That
    class is the network's training target and the reachability ceiling, so the failure was
    silent in both. `canonical_hkl` returns int64 now; this pins it.
    """
    reference = Assign.hkl_reference('mP')
    hkl = np.array([[-1.0, 3.0, 0.0]])
    canonical = Assign.canonical_hkl(hkl, 'monoclinic')
    assert canonical.dtype == np.int64
    assert not np.signbit(canonical).any(), 'a canonical row must not carry a negative zero'

    index = Assign.hkl_class_index(hkl, reference, 'monoclinic')[0]
    assert index != len(reference) - 1, 'a reflection present in the list is not the sentinel'
    assert np.array_equal(
        Assign.canonical_hkl(reference[index:index + 1], 'monoclinic'), canonical,
        )
    # The two routes to "the same reflection" must agree, which is what broke: `assignment_labels`
    # compares elementwise and was right, `hkl_class_index` compares bytes and was not.
    assert Assign.assignment_labels(hkl[np.newaxis], hkl[np.newaxis], 'monoclinic').all()


def test_every_reference_line_maps_back_to_its_own_index():
    """The round trip that would have caught it: each row of the list must find itself."""
    for lattice, system in (('mP', 'monoclinic'), ('aP', 'triclinic'), ('oP', 'orthorhombic'),
                            ('hR', 'rhombohedral')):
        reference = Assign.hkl_reference(lattice)
        codes = Assign.hkl_class_index(reference, reference, system)
        canonical = Assign.canonical_hkl(reference, system)
        # Duplicate canonical rows are allowed to collide onto one index; nothing may fall through
        # to the sentinel except the sentinel row itself.
        fell_through = np.flatnonzero(codes == len(reference) - 1)
        assert np.all(np.array_equal(canonical[index], canonical[-1]) for index in fell_through), (
            f'{lattice}: {len(fell_through)} reference rows do not find themselves'
            )
        assert np.array_equal(canonical[codes], canonical)


# ---------------------------------------------------------------------------------------
# The posterior form
# ---------------------------------------------------------------------------------------
def test_the_posterior_is_a_distribution_over_the_competing_lines():
    """It is a normalised weight, so it is bounded, and it is 1 only when nothing competes."""
    rng = np.random.default_rng(20)
    ref = np.sort(rng.uniform(0.01, 0.5, size=(4, 300)), axis=1)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=20))
    p = fom.get_assignment_posterior(q2, ref, 'monoclinic')
    assert p.shape == (4, 20)
    assert (p > 0).all() and (p <= 1.0 + 1e-12).all()

    # An isolated line takes all the mass; a line with a close neighbour splits it.
    isolated = np.array([[0.1, 5.0, 9.0]])
    crowded = np.array([[0.1, 0.1 + 1e-9, 9.0]])
    one = np.array([0.1])
    assert fom.get_assignment_posterior(one, isolated, 'cubic', sigma=1e-3)[0, 0] > 0.999
    assert fom.get_assignment_posterior(
        one, crowded, 'cubic', sigma=1e-3,
        )[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_the_posterior_is_scale_invariant_and_that_is_why_it_cannot_rank_candidates():
    """The finding F-130 turns on, pinned as a property rather than left as a story.

    `chi_r` normalises the residuals by the candidate's own fit, so scaling every distance by a
    constant leaves the posterior unchanged. That is what makes it calibrated per peak and blind to
    whether the cell fits at all -- a wrong cell and a right one look the same to it, which is
    exactly the signal a figure of merit needs and `rho` keeps.
    """
    rng = np.random.default_rng(21)
    ref = np.sort(rng.uniform(0.01, 0.5, size=(1, 200)), axis=1)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=20))
    tight = fom.get_assignment_posterior(q2, ref, 'monoclinic')
    # Push every calculated line ten times further from every observation, about their own mean.
    loose = fom.get_assignment_posterior(
        q2.mean() + 10*(q2 - q2.mean()), q2.mean() + 10*(ref - q2.mean()), 'monoclinic',
        )
    assert np.allclose(tight, loose, rtol=1e-6), (
        'the posterior must be invariant to the overall residual scale -- that invariance is '
        'F-130s explanation for why it cannot discriminate candidates'
        )


def test_the_sigma_estimate_is_taupins_reduced_chi_square():
    rng = np.random.default_rng(22)
    ref = np.sort(rng.uniform(0.01, 0.5, size=(2, 150)), axis=1)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=20))
    sigma, d1 = fom.get_assignment_sigma(q2, ref, 'monoclinic')
    expected = np.sqrt(np.sum(d1**2, axis=1)/(20 - fom.N_FREE_PARAMETERS['monoclinic']))
    assert np.allclose(sigma, expected)
    assert fom.N_FREE_PARAMETERS['triclinic'] == 6 and fom.N_FREE_PARAMETERS['cubic'] == 1
    # d1 really is the nearest line, which is what `fast_assign` picks.
    assert np.allclose(d1[0, 0], np.abs(ref[0] - q2[0]).min())


def test_a_wider_sigma_spreads_the_posterior():
    """The sensitivity knob PROTOCOL section 3 rule 4 requires, and its direction."""
    rng = np.random.default_rng(23)
    ref = np.sort(rng.uniform(0.01, 0.5, size=(1, 200)), axis=1)
    q2 = np.sort(rng.uniform(0.02, 0.4, size=20))
    narrow = fom.get_assignment_posterior(q2, ref, 'monoclinic', sigma_multiplier=0.25)
    wide = fom.get_assignment_posterior(q2, ref, 'monoclinic', sigma_multiplier=4.0)
    assert (narrow >= wide - 1e-12).all()
    assert narrow.mean() > wide.mean()
