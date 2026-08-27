"""The batched labeller must agree with the scalar one, candidate for candidate.

`label_known_bl_batch` is what S07 labels Benchmark B with, because the scalar
`validate_candidate_known_bl` costs ~9 ms a candidate and the survivor pool is ~2.5 billion rows.
The whole justification for the batch form is that it is the *same* function computed differently,
so the gate is exact agreement on both returned flags, not a tolerance.

Campaign 1 measured the pair at 1 584x with zero disagreements over 57.4 M rows (F-166); that
covered `is_correct` only, since the off-by-two grid was dropped. This covers both, over all seven
lattice systems, on cells built to land in every branch: exact matches, deliberate sub- and
super-cells at each multiplier the grids carry, and random cells that should match nothing.

A disagreement here is a defect in the batch form. It is never a reason to relax the assertion --
`is_correct` is the entire signal in the benchmark, and a quiet False reads as "no correct
candidate", which is indistinguishable from a generation failure (PROTOCOL section 8).
"""

import numpy as np
import pytest

from mlindex.optimization.CandidateValidation import TRUTH_SLICE
from mlindex.optimization.CandidateValidation import label_known_bl_batch
from mlindex.optimization.CandidateValidation import validate_candidate_known_bl
from mlindex.utilities.UnitCellTools import get_full_unit_cell


# One Bravais lattice per system, since the scalar routine dispatches on the lattice and the
# batch form on the system.
LATTICE_FOR_SYSTEM = {
    'cubic': 'cP',
    'tetragonal': 'tP',
    'hexagonal': 'hP',
    'rhombohedral': 'hR',
    'orthorhombic': 'oP',
    'monoclinic': 'mP',
    'triclinic': 'aP',
    }

RIGHT = np.pi / 2


def _true_cell(system, rng):
    """A full six-parameter true cell of the right metric symmetry, angles in radians.

    Built by drawing the system's free parameters and widening them with
    `UnitCellTools.get_full_unit_cell`, which is the repository's own inverse of
    `get_partial_unit_cell` -- so the constrained angles come from the one place that defines
    them rather than from a second copy of the rule here.
    """
    a, b, c = rng.uniform(4.0, 25.0, size=3)
    if system == 'cubic':
        return get_full_unit_cell(np.array([a]), system)
    if system in ('tetragonal', 'hexagonal'):
        return get_full_unit_cell(np.array([a, c]), system)
    if system == 'rhombohedral':
        alpha = rng.uniform(np.deg2rad(50), np.deg2rad(110))
        return get_full_unit_cell(np.array([a, alpha]), system)
    if system == 'orthorhombic':
        return get_full_unit_cell(np.array([a, b, c]), system)
    if system == 'monoclinic':
        beta = rng.uniform(np.deg2rad(92), np.deg2rad(125))
        return get_full_unit_cell(np.array([a, b, c, beta]), system)
    angles = rng.uniform(np.deg2rad(72), np.deg2rad(108), size=3)
    # Triclinic truth is returned SELLING-REDUCED. The scalar routine's off-by-two arm compares
    # the reduced prediction against the truth as given, so an unreduced truth makes that branch
    # unreachable and the test would pass while covering nothing -- which is what the coverage
    # assertions at the end of the test exist to catch.
    from mlindex.utilities.Reindexing import reindex_entry_triclinic
    reduced, _ = reindex_entry_triclinic(np.array([a, b, c, *angles]))
    return np.asarray(reduced, dtype=float)


def _predictions(system, unit_cell_true, rng):
    """Candidates spanning every branch: exact, scaled by each multiplier, and noise.

    The scaled rows are what exercise the off-by-two grids. Lengths are scaled and angles left
    alone, which is what a sub- or super-cell of the same lattice looks like.
    """
    partial = unit_cell_true[TRUTH_SLICE[system]]
    n_lengths = {'cubic': 1, 'tetragonal': 2, 'hexagonal': 2, 'rhombohedral': 1,
                 'orthorhombic': 3, 'monoclinic': 3, 'triclinic': 3}[system]

    rows = [partial.copy()]
    for factor in (0.5, 2.0, 1 / 3, 3.0):
        scaled = partial.copy()
        scaled[:n_lengths] = scaled[:n_lengths] * factor
        rows.append(scaled)
    if n_lengths > 1:
        # An anisotropic scaling, so the multi-axis grids are reached rather than only the
        # isotropic diagonal of them.
        mixed = partial.copy()
        mixed[0] = mixed[0] * 2.0
        rows.append(mixed)
    # Cells that should match nothing, and near-misses just outside the 1 % tolerance.
    for _ in range(6):
        rows.append(partial * rng.uniform(1.05, 1.6, size=partial.shape))
    rows.append(partial * (1.0 + rng.uniform(0.011, 0.02, size=partial.shape)))
    return np.stack(rows)


@pytest.mark.parametrize('system', sorted(LATTICE_FOR_SYSTEM))
def test_the_batch_labeller_agrees_with_the_scalar_one_on_both_flags(system):
    rng = np.random.default_rng(20260827)
    bravais_lattice = LATTICE_FOR_SYSTEM[system]
    disagreements = []
    n_correct = n_off_by_two = 0

    for _ in range(25):
        unit_cell_true = _true_cell(system, rng)
        predictions = _predictions(system, unit_cell_true, rng)

        batch_correct, batch_off = label_known_bl_batch(
            unit_cell_true[TRUTH_SLICE[system]], predictions, system, rtol=1e-2)

        for position, prediction in enumerate(predictions):
            # The scalar routine slices the truth itself, so it is handed the full six.
            scalar_correct, scalar_off = validate_candidate_known_bl(
                unit_cell_true.copy(), prediction.copy(), bravais_lattice, rtol=1e-2)
            scalar_correct, scalar_off = bool(scalar_correct), bool(scalar_off)
            if (scalar_correct, scalar_off) != (bool(batch_correct[position]),
                                                bool(batch_off[position])):
                disagreements.append((prediction, (scalar_correct, scalar_off),
                                      (bool(batch_correct[position]),
                                       bool(batch_off[position]))))
            n_correct += scalar_correct
            n_off_by_two += scalar_off

    assert not disagreements, (
        f'{len(disagreements)} of {25 * predictions.shape[0]} {system} candidates disagree, '
        f'first: {disagreements[0]}')
    # A test that never reached either branch would pass vacuously, which is how a labeller that
    # returns all-False passes an equivalence test against itself.
    assert n_correct > 0, f'no {system} candidate was labelled correct; the test proves nothing'
    assert n_off_by_two > 0, f'no {system} candidate was labelled off-by-two'


def test_an_empty_block_returns_empty_arrays_rather_than_raising():
    correct, off_by_two = label_known_bl_batch(
        np.array([5.0, 6.0, 7.0]), np.zeros((0, 3)), 'orthorhombic')
    assert correct.shape == (0,) and off_by_two.shape == (0,)
    assert correct.dtype == bool and off_by_two.dtype == bool


def test_an_unknown_lattice_system_raises_rather_than_returning_false():
    # A quiet False reads as "no correct candidate" and is indistinguishable from a generation
    # failure, which PROTOCOL section 8 requires be kept a separate bucket.
    with pytest.raises(ValueError, match='does not implement'):
        label_known_bl_batch(np.array([5.0]), np.array([[5.0]]), 'octahedral')


# ------------------------------------------------------------------------------------------
# The basis change, and `hkl_true_in_basis`.
# ------------------------------------------------------------------------------------------

def _direct_metric(unit_cell):
    """G_ij = |a_i||a_j| cos(angle between them), from a full six-parameter cell in radians."""
    a, b, c, alpha, beta, gamma = unit_cell
    return np.array([
        [a * a, a * b * np.cos(gamma), a * c * np.cos(beta)],
        [a * b * np.cos(gamma), b * b, b * c * np.cos(alpha)],
        [a * c * np.cos(beta), b * c * np.cos(alpha), c * c],
        ])


@pytest.mark.parametrize('system', sorted(LATTICE_FOR_SYSTEM))
def test_the_recovered_basis_change_relates_the_two_metric_tensors(system):
    """`A_true = A_pred @ M` means `G_true = M.T @ G_pred @ M`, and that is checkable directly.

    This is the property `hkl_true_in_basis` depends on: if it holds, re-expressing a reflection
    through M preserves q-squared, which is the whole content of "the same lattice in a different
    setting". If it did not hold, the stored Miller indices would be a plausible-looking answer to
    a different question.
    """
    from mlindex.optimization.CandidateValidation import basis_change_known_bl_batch

    rng = np.random.default_rng(4242)
    n_matched = 0
    for _ in range(25):
        unit_cell_true = _true_cell(system, rng)
        predictions = _predictions(system, unit_cell_true, rng)
        changes = basis_change_known_bl_batch(
            unit_cell_true[TRUTH_SLICE[system]], predictions, system, rtol=1e-2)

        metric_true = _direct_metric(unit_cell_true)
        for position, change in enumerate(changes):
            if not np.all(np.isfinite(change)):
                continue
            n_matched += 1
            metric_pred = _direct_metric(get_full_unit_cell(predictions[position], system))
            assert np.allclose(change.T @ metric_pred @ change, metric_true, rtol=2e-2,
                               atol=1e-6), (
                f'{system}: recovered basis change does not relate the metrics\n'
                f'{change}\n{change.T @ metric_pred @ change}\nvs\n{metric_true}')
    assert n_matched > 0, f'no {system} candidate matched; the test proves nothing'


def test_an_incorrect_candidate_gets_no_basis_change_and_no_reexpressed_indices():
    from mlindex.optimization.CandidateValidation import basis_change_known_bl_batch
    from mlindex.optimization.CandidateValidation import hkl_in_candidate_basis

    unit_cell_true = np.array([8.0, 11.0, 14.0, RIGHT, RIGHT, RIGHT])
    predictions = np.array([[8.0, 11.0, 14.0], [3.1, 4.7, 21.0]])
    changes = basis_change_known_bl_batch(unit_cell_true[TRUTH_SLICE['orthorhombic']],
                                          predictions, 'orthorhombic')
    assert np.all(np.isfinite(changes[0]))
    assert not np.any(np.isfinite(changes[1]))
    assert hkl_in_candidate_basis(np.array([[1, 2, 3]]), changes[1]) is None


def test_an_axis_permutation_is_recovered_and_permutes_the_indices():
    """The orthorhombic setting is a permutation, and it must move the Miller indices with it."""
    from mlindex.optimization.CandidateValidation import basis_change_known_bl_batch
    from mlindex.optimization.CandidateValidation import hkl_in_candidate_basis

    unit_cell_true = np.array([8.0, 11.0, 14.0, RIGHT, RIGHT, RIGHT])
    # The same lattice, axes listed in a different order -- which is a correct candidate.
    predictions = np.array([[14.0, 8.0, 11.0]])
    changes = basis_change_known_bl_batch(unit_cell_true[TRUTH_SLICE['orthorhombic']],
                                          predictions, 'orthorhombic')
    assert np.all(np.isfinite(changes[0]))
    reexpressed = hkl_in_candidate_basis(np.array([[1, 2, 3]]), changes[0])
    # a_true -> the candidate's second axis, b_true -> third, c_true -> first.
    assert reexpressed.tolist() == [[3, 1, 2]], reexpressed
    assert reexpressed.dtype == np.int16


def _monoclinic_cell_matrix(partial):
    a, b, c, beta = partial
    matrix = np.zeros((3, 3))
    matrix[0, 0] = a
    matrix[0, 2] = c * np.cos(beta)
    matrix[1, 1] = b
    matrix[2, 2] = c * np.sin(beta)
    return matrix


def _monoclinic_parameters(matrix):
    lengths = np.linalg.norm(matrix, axis=0)
    beta = np.arccos(np.dot(matrix[:, 0], matrix[:, 2]) / (lengths[0] * lengths[2]))
    return np.array([lengths[0], lengths[1], lengths[2], beta])


def test_alternative_monoclinic_settings_are_recovered_with_the_right_basis_change():
    """The case the metric-tensor test above cannot reach on random cells.

    A random candidate that happens to match the truth matches in the identity setting, so that
    test exercises `M = I` and nothing else. Here the candidates are built to be the *same*
    lattice described in each of the twenty settings the labeller walks -- which is the situation
    `hkl_true_in_basis` exists for, and the only one in which a wrong M would be visible.

    The assertion is on the metric relation rather than on which M comes back: two settings can
    both match, the labeller takes the first, and any matching setting is a correct answer.
    """
    from mlindex.optimization.CandidateValidation import _MONOCLINIC_BASIS_CHANGES
    from mlindex.optimization.CandidateValidation import basis_change_known_bl_batch
    from mlindex.optimization.CandidateValidation import hkl_in_candidate_basis

    truth_partial = np.array([7.0, 9.0, 13.0, np.deg2rad(104.0)])
    unit_cell_true = get_full_unit_cell(truth_partial, 'monoclinic')
    matrix_true = _monoclinic_cell_matrix(truth_partial)

    predictions = np.array([
        _monoclinic_parameters(matrix_true @ np.linalg.inv(change))
        for change in _MONOCLINIC_BASIS_CHANGES])

    changes = basis_change_known_bl_batch(truth_partial, predictions, 'monoclinic', rtol=1e-2)
    assert np.all(np.isfinite(changes)), 'every alternative setting of one lattice is correct'

    n_non_identity = 0
    metric_true = _direct_metric(unit_cell_true)
    hkl_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, -3], [2, -1, 4]])
    for prediction, change in zip(predictions, changes):
        metric_pred = _direct_metric(get_full_unit_cell(prediction, 'monoclinic'))
        assert np.allclose(change.T @ metric_pred @ change, metric_true, rtol=1e-6, atol=1e-8)
        if not np.allclose(change, np.eye(3)):
            n_non_identity += 1
        # And the point of it all: the re-expressed indices are integers, and they preserve
        # q-squared, which is what makes them the same reflections.
        reexpressed = hkl_in_candidate_basis(hkl_true, change)
        assert reexpressed is not None and reexpressed.dtype == np.int16
        reciprocal_true = np.linalg.inv(metric_true)
        reciprocal_pred = np.linalg.inv(metric_pred)
        q2_true = np.einsum('mi,ij,mj->m', hkl_true, reciprocal_true, hkl_true)
        q2_pred = np.einsum('mi,ij,mj->m', reexpressed.astype(float), reciprocal_pred,
                            reexpressed.astype(float))
        assert np.allclose(q2_true, q2_pred, rtol=1e-8), (
            'the re-expressed reflections are at different q-squared, so they are not the '
            'same reflections')

    assert n_non_identity >= 15, (
        f'only {n_non_identity} of {len(changes)} settings needed a non-identity basis change; '
        'the test has stopped covering what it was written for')


def test_the_truth_slice_is_the_one_the_repository_already_defines():
    """`TRUTH_SLICE` is derived from `UnitCellTools.get_partial_unit_cell`, and pinned here.

    Two failures this catches. If the library's selection changes, the literals below fail and
    someone has to decide deliberately -- rather than the labeller silently comparing a different
    angle, which is exactly what a contiguous slice would do to monoclinic. And if the derivation
    is ever replaced by a hand-written copy, this still passes, but the comment above the
    definition says why that would be a step backwards: campaign 1 kept one naming rule in four
    places and they drifted.
    """
    from mlindex.optimization.CandidateValidation import LATTICE_SYSTEMS
    from mlindex.utilities.UnitCellTools import get_partial_unit_cell

    assert TRUTH_SLICE == {
        'cubic': [0],
        'tetragonal': [0, 2],
        'hexagonal': [0, 2],
        'rhombohedral': [0, 3],
        'orthorhombic': [0, 1, 2],
        'monoclinic': [0, 1, 2, 4],
        'triclinic': [0, 1, 2, 3, 4, 5],
        }
    assert set(LATTICE_SYSTEMS) == set(TRUTH_SLICE)
    for system, indices in TRUTH_SLICE.items():
        derived = np.atleast_1d(get_partial_unit_cell(np.arange(6), lattice_system=system))
        assert derived.tolist() == indices, system


def test_the_truth_slice_selects_what_get_partial_unit_cell_selects_on_a_real_cell():
    """The property that matters: slicing a truth by TRUTH_SLICE is taking its partial cell."""
    from mlindex.utilities.UnitCellTools import get_partial_unit_cell

    rng = np.random.default_rng(11)
    for system in sorted(TRUTH_SLICE):
        unit_cell_true = _true_cell(system, rng)
        assert np.array_equal(
            unit_cell_true[TRUTH_SLICE[system]],
            np.atleast_1d(get_partial_unit_cell(unit_cell_true, lattice_system=system))), system
