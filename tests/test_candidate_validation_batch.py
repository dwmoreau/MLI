"""The batched labeller must agree with the scalar one, candidate for candidate.

`label_known_bl_batch` is the correctness oracle a benchmark labels with: it decides whether a
candidate cell *is* the true cell, allowing for reindexing and for sub- and super-cells. Every
top-N number rests on it, and over a pool of millions of candidates the scalar routine is not an
option. So the batch form is gated against the scalar form here. A disagreement is a defect in the
batch function, never a reason to relax this test.
"""

import numpy as np
import pytest

from mlindex.optimization.CandidateValidation import TRUTH_SLICE
from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
from mlindex.optimization.CandidateValidation import label_known_bl_batch
from mlindex.optimization.CandidateValidation import off_by_two_known_bl_batch
from mlindex.optimization.CandidateValidation import validate_candidate_known_bl
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM

LATTICE_SYSTEMS = sorted(set(BL_TO_LATTICE_SYSTEM.values()))
BL_FOR_SYSTEM = {'cubic': 'cP', 'tetragonal': 'tP', 'hexagonal': 'hP', 'rhombohedral': 'hR',
                 'orthorhombic': 'oP', 'monoclinic': 'mP', 'triclinic': 'aP'}


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
        block.append(np.atleast_1d(other[TRUTH_SLICE[system]]).astype(float))
    return np.array(block)


@pytest.mark.parametrize('system', LATTICE_SYSTEMS)
def test_the_batched_labeller_agrees_with_the_scalar_one(system):
    rng = np.random.default_rng(abs(hash(system)) % (2**32))
    bravais_lattice = BL_FOR_SYSTEM[system]
    n_compared = 0
    seen = set()

    for _ in range(25):
        true_full = _random_true_cell(system, rng)
        true_partial = np.atleast_1d(true_full[TRUTH_SLICE[system]]).astype(float)
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
    true_partial = np.atleast_1d(true_full[TRUTH_SLICE[system]]).astype(float)
    block = _candidates(true_partial, system, rng)

    correct, off_by_two = label_known_bl_batch(true_partial, block, system)
    np.testing.assert_array_equal(
        correct, is_correct_known_bl_batch(true_partial, block, system))
    raw = off_by_two_known_bl_batch(true_partial, block, system)
    assert np.all(off_by_two <= raw), 'the mask may only remove off-by-two flags, never add them'


@pytest.mark.parametrize('system', LATTICE_SYSTEMS)
def test_an_empty_block_returns_empty_arrays(system):
    """A Bravais lattice can contribute no candidates to a pattern, and that must not raise."""
    true_partial = np.atleast_1d(
        _random_true_cell(system, np.random.default_rng(0))[TRUTH_SLICE[system]]).astype(float)
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


def test_truth_slice_keeps_the_angle_each_system_actually_has():
    """Monoclinic keeps beta and skips alpha; rhombohedral keeps alpha and skips c. Slicing the
    truth differently from the scalar routine compares the wrong angle and mislabels silently."""
    assert TRUTH_SLICE['cubic'] == [0]
    assert TRUTH_SLICE['tetragonal'] == [0, 2]
    assert TRUTH_SLICE['hexagonal'] == [0, 2]
    assert TRUTH_SLICE['rhombohedral'] == [0, 3]
    assert TRUTH_SLICE['orthorhombic'] == [0, 1, 2]
    assert TRUTH_SLICE['monoclinic'] == [0, 1, 2, 4]
    assert TRUTH_SLICE['triclinic'] == [0, 1, 2, 3, 4, 5]
