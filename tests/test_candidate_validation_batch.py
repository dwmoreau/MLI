"""The batched labeller must agree with the scalar one, candidate for candidate.

`label_known_bl_batch` is the correctness oracle a benchmark labels with: it decides whether a
candidate cell *is* the true cell, allowing for reindexing and for sub- and super-cells. Every
top-N number rests on it, and over a pool of millions of candidates the scalar routine is not an
option. So the batch form is gated against the scalar form here. A disagreement is a defect in the
batch function, never a reason to relax this test.
"""

import numpy as np
import pytest

from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
from mlindex.optimization.CandidateValidation import label_known_bl_batch
from mlindex.optimization.CandidateValidation import off_by_two_known_bl_batch
from mlindex.optimization.CandidateValidation import validate_candidate_known_bl
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import get_partial_unit_cell

LATTICE_SYSTEMS = sorted(set(BL_TO_LATTICE_SYSTEM.values()))
BL_FOR_SYSTEM = {'cubic': 'cP', 'tetragonal': 'tP', 'hexagonal': 'hP', 'rhombohedral': 'hR',
                 'orthorhombic': 'oP', 'monoclinic': 'mP', 'triclinic': 'aP'}


def _sliced(unit_cell, lattice_system):
    """The truth cut down to a system's free parameters, by the function that defines the rule.

    The batch labeller takes the truth already sliced, and slicing it any other way than the scalar
    routine does compares the wrong angle -- monoclinic keeps beta and skips alpha, rhombohedral
    keeps alpha and skips c.
    """
    return np.atleast_1d(get_partial_unit_cell(unit_cell, lattice_system=lattice_system)).astype(
        float)


def _random_true_cell(system, rng):
    a, b, c = np.sort(rng.uniform(3.0, 25.0, size=3))
    angle = lambda: rng.uniform(np.deg2rad(65), np.deg2rad(115))
    if system == 'cubic':
        return np.array([a, a, a, np.pi / 2, np.pi / 2, np.pi / 2])
    if system == 'tetragonal':
        return np.array([a, a, c, np.pi / 2, np.pi / 2, np.pi / 2])
    if system == 'hexagonal':
        return np.array([a, a, c, np.pi / 2, np.pi / 2, 2 * np.pi / 3])
    if system == 'rhombohedral':
        alpha = angle()
        return np.array([a, a, a, alpha, alpha, alpha])
    if system == 'orthorhombic':
        return np.array([a, b, c, np.pi / 2, np.pi / 2, np.pi / 2])
    if system == 'monoclinic':
        return np.array([a, b, c, np.pi / 2, angle(), np.pi / 2])
    return np.array([a, b, c, angle(), angle(), angle()])


def _candidates(true_partial, system, rng, n_random=6):
    """A block mixing the exact truth, sub- and super-cells, and unrelated cells.

    Without the derived cells the comparison would be almost all `(False, False)` and would not
    exercise the arms that actually matter.
    """
    width = true_partial.size
    block = [true_partial.copy()]
    for scale in (0.5, 2.0, 1 / 3, 3.0):
        scaled = true_partial.copy()
        scaled[:3 if width > 3 else width] *= scale
        block.append(scaled)
    for axis in range(min(3, width)):
        scaled = true_partial.copy()
        scaled[axis] *= 2.0
        block.append(scaled)
    for _ in range(n_random):
        other = _random_true_cell(system, rng)
        block.append(_sliced(other, system))
    return np.array(block)


@pytest.mark.parametrize('system', LATTICE_SYSTEMS)
def test_the_batched_labeller_agrees_with_the_scalar_one(system):
    rng = np.random.default_rng(abs(hash(system)) % (2**32))
    bravais_lattice = BL_FOR_SYSTEM[system]
    n_compared = 0
    seen = set()

    for _ in range(25):
        true_full = _random_true_cell(system, rng)
        true_partial = _sliced(true_full, system)
        block = _candidates(true_partial, system, rng)

        correct, off_by_two = label_known_bl_batch(true_partial, block, system)

        for index in range(block.shape[0]):
            expected = validate_candidate_known_bl(true_full.copy(), block[index].copy(),
                                                   bravais_lattice)
            got = (bool(correct[index]), bool(off_by_two[index]))
            assert got == tuple(map(bool, expected)), (
                f'{system} candidate {index}: batch {got} != scalar {tuple(map(bool, expected))}\n'
                f'  true {true_partial}\n  pred {block[index]}')
            seen.add(got)
            n_compared += 1

    assert n_compared > 0
    # The comparison is worthless if every candidate landed in one bucket.
    assert len(seen) >= 2, f'{system}: only outcome(s) {seen} were exercised'


@pytest.mark.parametrize('system', LATTICE_SYSTEMS)
def test_the_two_arms_are_the_two_halves_of_the_scalar_return(system):
    """`is_correct_known_bl_batch` and `off_by_two_known_bl_batch` are usable separately, and must
    mean the same thing they do inside `label_known_bl_batch` -- apart from the early-return mask,
    which is the one documented difference."""
    rng = np.random.default_rng(11)
    true_full = _random_true_cell(system, rng)
    true_partial = _sliced(true_full, system)
    block = _candidates(true_partial, system, rng)

    correct, off_by_two = label_known_bl_batch(true_partial, block, system)
    np.testing.assert_array_equal(
        correct, is_correct_known_bl_batch(true_partial, block, system))
    raw = off_by_two_known_bl_batch(true_partial, block, system)
    assert np.all(off_by_two <= raw), 'the mask may only remove off-by-two flags, never add them'


@pytest.mark.parametrize('system', LATTICE_SYSTEMS)
def test_an_empty_block_returns_empty_arrays(system):
    """A Bravais lattice can contribute no candidates to a pattern, and that must not raise."""
    true_partial = _sliced(_random_true_cell(system, np.random.default_rng(0)), system)
    width = true_partial.size
    correct, off_by_two = label_known_bl_batch(true_partial, np.empty((0, width)), system)
    assert correct.shape == (0,) and off_by_two.shape == (0,)
    assert correct.dtype == bool and off_by_two.dtype == bool


def test_an_unimplemented_system_raises_rather_than_reporting_nothing_correct():
    """A quiet False reads as 'no correct candidate in the pool', which is a generation failure and
    has to stay distinguishable from a ranking failure."""
    with pytest.raises(ValueError, match='does not implement'):
        is_correct_known_bl_batch(np.array([5.0]), np.array([[5.0]]), 'nonsense')
    with pytest.raises(ValueError, match='does not implement'):
        off_by_two_known_bl_batch(np.array([5.0]), np.array([[5.0]]), 'nonsense')


def test_the_truth_is_sliced_the_way_the_scalar_routine_slices_it():
    """Monoclinic keeps beta and skips alpha; rhombohedral keeps alpha and skips c. Slicing the
    truth differently from the scalar routine compares the wrong angle and mislabels silently."""
    unit_cell = np.array([5.0, 6.0, 7.0, 1.1, 1.2, 1.3])
    expected = {'cubic': [5.0], 'tetragonal': [5.0, 7.0], 'hexagonal': [5.0, 7.0],
                'rhombohedral': [5.0, 1.1], 'orthorhombic': [5.0, 6.0, 7.0],
                'monoclinic': [5.0, 6.0, 7.0, 1.2],
                'triclinic': [5.0, 6.0, 7.0, 1.1, 1.2, 1.3]}
    for lattice_system, want in expected.items():
        np.testing.assert_array_equal(_sliced(unit_cell, lattice_system), np.array(want),
                                      err_msg=lattice_system)


def _in_setting(unit_cell, transformation):
    """The full cell `unit_cell` re-expressed in the basis `cell_matrix(...) @ transformation`."""
    from mlindex.utilities.Reindexing import cell_matrix, unit_cell_from_matrix
    basis = cell_matrix(np.asarray(unit_cell)[np.newaxis])[0]
    return unit_cell_from_matrix(basis @ transformation)


def test_a_triclinic_cell_in_another_reduced_setting_is_the_true_cell():
    """Replacing c by -(a+b+c), the fourth vector of the reduced set, gives another cell of the
    same lattice; the labeller compared triclinic cells only as given and called it wrong."""
    truth = np.array([12.67, 15.30, 21.50, np.radians(109.28), np.radians(99.16),
                      np.radians(123.40)])
    other = _in_setting(truth, np.array([[1, 0, -1], [0, 1, -1], [0, 0, -1]]))
    assert not np.allclose(other, truth, rtol=1e-2)
    distorted = other*np.array([1, 1, 1.05, 1, 1, 1])
    correct = is_correct_known_bl_batch(truth, np.stack([other, distorted]), 'triclinic')
    assert correct.tolist() == [True, False]


def test_a_monoclinic_cell_in_a_setting_outside_the_twenty_is_the_true_cell():
    """Of the basis changes that keep b unique with entries in {-1, 0, 1}, sixteen give this cell a
    setting the twenty MONOCLINIC_BASIS_CHANGES never reach; a cell in one of them is still the
    true cell. Here a stays, b is reversed, and c becomes c - a."""
    from mlindex.utilities.Reindexing import monoclinic_settings
    transformation = np.array([[-1, 0, -1], [0, -1, 0], [0, 0, 1]])
    truth = np.array([11.34, 18.49, 19.73, np.radians(93.13)])
    full_truth = np.array([11.34, 18.49, 19.73, np.pi/2, np.radians(93.13), np.pi/2])
    other = _in_setting(full_truth, transformation)
    assert np.allclose(other[[3, 5]], np.pi/2)
    candidate = other[[0, 1, 2, 4]]
    walked = monoclinic_settings(candidate[np.newaxis])[:, 0]
    assert not np.all(np.isclose(walked, truth, rtol=1e-2), axis=1).any()
    assert is_correct_known_bl_batch(truth, candidate[np.newaxis], 'monoclinic').tolist() == [True]
    assert is_correct_known_bl_batch(truth, (candidate*[1, 1.05, 1, 1])[np.newaxis],
                                     'monoclinic').tolist() == [False]
