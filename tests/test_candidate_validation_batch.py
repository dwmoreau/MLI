"""What the labeller decides, outcome by outcome.

`label_known_bl_batch` is the correctness oracle a benchmark labels with: it decides whether a
candidate cell *is* the true cell, allowing for reindexing, and whether it is a sub- or super-cell
of it. Every top-N number rests on it. It used to be gated against a scalar routine that compared
cells parameter by parameter; that routine flagged a correct monoclinic cell as off by two and
missed off-by-two cells in another setting, so it now calls the batch labeller and these tests
state the outcomes instead.
"""

import numpy as np
import pytest

from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
from mlindex.optimization.CandidateValidation import label_known_bl_batch
from mlindex.optimization.CandidateValidation import off_by_two_known_bl_batch
from mlindex.optimization.CandidateValidation import validate_candidate
from mlindex.optimization.CandidateValidation import validate_candidate_known_bl
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import get_partial_unit_cell

BRAVAIS_LATTICES = sorted(BL_TO_LATTICE_SYSTEM)


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


# The positions of the free axis lengths in each system's partial cell.
_LENGTHS = {'cubic': [0], 'tetragonal': [0, 1], 'hexagonal': [0, 1], 'rhombohedral': [0],
            'orthorhombic': [0, 1, 2], 'monoclinic': [0, 1, 2], 'triclinic': [0, 1, 2]}


@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_the_true_cell_is_correct_and_never_off_by_two(bravais_lattice):
    """The grids contain the identity, so without the mask a correct cell is flagged as its own
    sub-cell; it was masked for four systems only, and a correct monoclinic cell came back as
    both."""
    system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    rng = np.random.default_rng(BRAVAIS_LATTICES.index(bravais_lattice))
    truths = [_sliced(_random_true_cell(system, rng), system) for _ in range(10)]
    for truth in truths:
        correct, off_by_two = label_known_bl_batch(truth, truth[np.newaxis], bravais_lattice)
        assert (correct.tolist(), off_by_two.tolist()) == ([True], [False]), truth


@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_one_axis_doubled_or_halved_is_off_by_two(bravais_lattice):
    system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    rng = np.random.default_rng(BRAVAIS_LATTICES.index(bravais_lattice))
    for _ in range(10):
        truth = _sliced(_random_true_cell(system, rng), system)
        block = []
        for axis in _LENGTHS[system]:
            for scale in (0.5, 2.0):
                scaled = truth.copy()
                scaled[axis] *= scale
                block.append(scaled)
        correct, off_by_two = label_known_bl_batch(truth, np.array(block), bravais_lattice)
        assert not correct.any() and off_by_two.all(), (truth, off_by_two)


@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_a_cell_off_by_no_small_multiple_is_neither(bravais_lattice):
    system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    rng = np.random.default_rng(BRAVAIS_LATTICES.index(bravais_lattice))
    for _ in range(10):
        truth = _sliced(_random_true_cell(system, rng), system)
        other = truth.copy()
        other[_LENGTHS[system]] *= 1.37
        correct, off_by_two = label_known_bl_batch(truth, other[np.newaxis], bravais_lattice)
        assert (correct.tolist(), off_by_two.tolist()) == ([False], [False]), truth


def test_a_monoclinic_cell_doubled_in_another_setting_is_off_by_two():
    """DUWTAR's truth and its copy two shears away are one lattice; with b doubled the copy is off
    by two. Compared parameter by parameter from the cells as given it was neither."""
    truth = np.array([19.133, 14.349, 19.564, np.radians(151.04)])
    other = _monoclinic_in_setting(truth, np.array([[-1, 0, 2], [0, -1, 0], [0, 0, 1]]))
    for lattice in ('mP', 'mC'):
        correct, off_by_two = label_known_bl_batch(truth, np.stack([other, other*[1, 2, 1, 1]]),
                                                   lattice)
        assert correct.tolist() == [True, False], lattice
        assert off_by_two.tolist() == [False, True], lattice


def test_a_triclinic_cell_doubled_is_off_by_two_in_any_setting():
    """The candidate was reduced before the grid and the truth was not, so a triclinic cell with an
    axis doubled was missed even in the truth's own setting."""
    truth = np.array([6.1, 7.3, 9.4, np.radians(82.0), np.radians(97.0), np.radians(104.0)])
    other = _in_setting(truth, np.array([[1, 0, -1], [0, 1, -1], [0, 0, -1]]))
    doubled = np.array([2, 1, 1, 1, 1, 1])
    correct, off_by_two = label_known_bl_batch(truth, np.stack([truth*doubled, other*doubled]),
                                               'aP')
    assert correct.tolist() == [False, False]
    assert off_by_two.tolist() == [True, True]


def test_validate_candidate_reports_any_candidate_not_the_last():
    """`validate_candidate` reused the name of its result for each candidate's, so it reported
    whether the last candidate checked was off by two."""
    truth = np.array([10.0, 12.0, 15.0, np.pi/2, np.radians(103.0), np.pi/2])
    doubled = _sliced(truth, 'monoclinic')*[2, 1, 1, 1]
    unrelated = np.array([7.7, 8.8, 13.1, np.radians(95.0)])
    entry = {'reindexed_unit_cell': truth, 'bravais_lattice': 'mP'}
    for block in (np.stack([doubled, unrelated]), np.stack([unrelated, doubled])):
        assert validate_candidate(entry, {'mP': block}, {'mP': np.zeros(2)}) == (
            False, True, False, False)


def test_validate_candidate_known_bl_is_the_batch_label_of_one_candidate():
    truth = np.array([10.0, 12.0, 15.0, np.pi/2, np.radians(103.0), np.pi/2])
    partial = _sliced(truth, 'monoclinic')
    assert validate_candidate_known_bl(truth, partial, 'mP') == (True, False)
    assert validate_candidate_known_bl(truth, partial*[1, 2, 1, 1], 'mP') == (False, True)


@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_the_two_arms_are_the_two_halves_of_the_label(bravais_lattice):
    """`is_correct_known_bl_batch` and `off_by_two_known_bl_batch` are usable separately, and must
    mean the same thing they do inside `label_known_bl_batch` -- apart from the mask that clears
    off-by-two on a correct candidate."""
    system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    rng = np.random.default_rng(11)
    true_full = _random_true_cell(system, rng)
    true_partial = _sliced(true_full, system)
    block = _candidates(true_partial, system, rng)

    correct, off_by_two = label_known_bl_batch(true_partial, block, bravais_lattice)
    np.testing.assert_array_equal(
        correct, is_correct_known_bl_batch(true_partial, block, bravais_lattice))
    raw = off_by_two_known_bl_batch(true_partial, block, bravais_lattice)
    assert np.all(off_by_two <= raw), 'the mask may only remove off-by-two flags, never add them'


@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_an_empty_block_returns_empty_arrays(bravais_lattice):
    """A Bravais lattice can contribute no candidates to a pattern, and that must not raise."""
    system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    true_partial = _sliced(_random_true_cell(system, np.random.default_rng(0)), system)
    width = true_partial.size
    correct, off_by_two = label_known_bl_batch(true_partial, np.empty((0, width)),
                                               bravais_lattice)
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
    correct = is_correct_known_bl_batch(truth, np.stack([other, distorted]), 'aP')
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
    assert is_correct_known_bl_batch(truth, candidate[np.newaxis], 'mP').tolist() == [True]
    assert is_correct_known_bl_batch(truth, (candidate*[1, 1.05, 1, 1])[np.newaxis],
                                     'mP').tolist() == [False]


def _monoclinic_in_setting(partial, transformation):
    """A monoclinic partial cell (a, b, c, beta) re-expressed in another basis, as a partial cell."""
    full = np.array([partial[0], partial[1], partial[2], np.pi/2, partial[3], np.pi/2])
    other = _in_setting(full, transformation)
    assert np.allclose(other[[3, 5]], np.pi/2), 'the basis change must keep b unique'
    return other[[0, 1, 2, 4]]


def test_a_monoclinic_cell_two_shears_from_the_stored_truth_is_the_true_cell():
    """P17 (STATUS P-F-207): a third of the stored monoclinic truths are oblique, and the search
    prints oblique cells too. DUWTAR's truth has beta 151 degrees; its printed copy (-a, -b, 2a + c)
    has beta 156, and no basis change with entries in {-1, 0, 1} joins them, so
    walking from the cells as given called the right answer wrong."""
    truth = np.array([19.133, 14.349, 19.564, np.radians(151.04)])
    two_shears = _monoclinic_in_setting(truth, np.array([[-1, 0, 2], [0, -1, 0], [0, 0, 1]]))
    assert np.degrees(two_shears[3]) > 150 and not np.isclose(two_shears[2], truth[2], rtol=0.05)
    for lattice in ('mP', 'mC'):
        correct = is_correct_known_bl_batch(truth, np.stack([two_shears,
                                                             two_shears*[1, 1.03, 1, 1]]), lattice)
        assert correct.tolist() == [True, False], lattice


def test_an_mc_cell_whose_basis_change_moves_the_centring_point_is_another_lattice():
    """In the I setting the centring point is (a + b + c)/2. With c + a in place of c it lands on
    the A face, so the same six numbers labelled mC describe another lattice. The walk from the
    cells as given ignores centring and accepted it (STATUS P-F-209); as mP, where there is no
    centring point, it is the same lattice."""
    truth = np.array([14.988, 13.371, 17.384, np.radians(100.42)])
    moved = _monoclinic_in_setting(truth, np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]]))
    assert is_correct_known_bl_batch(truth, moved[np.newaxis], 'mC').tolist() == [False]
    assert is_correct_known_bl_batch(truth, moved[np.newaxis], 'mP').tolist() == [True]
    assert is_correct_known_bl_batch(truth, truth[np.newaxis], 'mC').tolist() == [True]


def test_one_percent_on_a_sheared_cell_is_not_one_percent_on_the_lattice():
    """A truth stored with beta 155 degrees, and a candidate 0.9 % off in beta alone: within the
    1 % tolerance as given, which is what the walk from the cells as given tested, but several
    percent off in the reduced cell, whose short axis is the difference of two long ones."""
    from mlindex.utilities.Reindexing import niggli_cell
    truth = np.array([10.0, 12.0, 25.0, np.radians(155.0)])
    candidate = truth*np.array([1, 1, 1, 1.009])
    reduced = [np.array(niggli_cell([cell[0], cell[1], cell[2], 90, np.degrees(cell[3]), 90],
                                    'P').parameters()[:3]) for cell in (truth, candidate)]
    assert np.max(np.abs(reduced[1] - reduced[0])/reduced[0]) > 0.02
    assert is_correct_known_bl_batch(truth, candidate[np.newaxis], 'mP').tolist() == [False]
