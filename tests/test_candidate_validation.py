"""Candidate labelling: the per-candidate labels and the per-entry flags reduced from them.

Run in both environments -- development first, then the runtime one, before anything is called
done:

    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python -m pytest tests/ -v
    /global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python -m pytest tests/ -v
"""
import numpy as np

from mlindex.optimization.CandidateValidation import label_candidates
from mlindex.optimization.CandidateValidation import validate_candidate
from mlindex.optimization.CandidateValidation import validate_candidate_known_bl


# The true cell, orthorhombic, and three oP predictions: exact, halved on a (off by two under
# the orthorhombic multiplier loop), and unrelated.
UNIT_CELL_TRUE = np.array([5.0, 7.0, 11.0, np.pi/2, np.pi/2, np.pi/2])
OP_EXACT = np.array([5.0, 7.0, 11.0])
OP_OFF_BY_TWO = np.array([2.5, 7.0, 11.0])
OP_UNRELATED = np.array([3.1, 8.3, 13.7])

ENTRY = {'reindexed_unit_cell': UNIT_CELL_TRUE, 'bravais_lattice': 'oP'}


def _flat_M20(top_unit_cell, value=50.0):
    return {bl: np.full(top_unit_cell[bl].shape[0], value) for bl in top_unit_cell}


def test_fixture_cells_label_as_intended():
    # Guard the fixture itself, so a failure below is about the reduction and not about the
    # three cells having stopped meaning what the other tests assume.
    assert validate_candidate_known_bl(UNIT_CELL_TRUE, OP_EXACT, 'oP') == (True, False)
    assert validate_candidate_known_bl(UNIT_CELL_TRUE, OP_OFF_BY_TWO, 'oP') == (False, True)
    assert validate_candidate_known_bl(UNIT_CELL_TRUE, OP_UNRELATED, 'oP') == (False, False)


def test_off_by_two_ors_across_candidates():
    # Regression: the off-by-two candidate is followed by one that is neither correct nor off
    # by two. The reduction must be an OR over candidates, not the last candidate's verdict.
    top_unit_cell = {'oP': np.stack([OP_OFF_BY_TWO, OP_UNRELATED])}
    found, off_by_two, incorrect_bl, found_explainer = validate_candidate(
        ENTRY, top_unit_cell, _flat_M20(top_unit_cell)
        )
    assert off_by_two
    assert not found
    assert not incorrect_bl
    assert not found_explainer


def test_off_by_two_ors_across_bravais_lattices():
    # Same OR, but with the off-by-two candidate in a lattice visited before another lattice
    # whose candidates are all unremarkable.
    top_unit_cell = {
        'oP': np.stack([OP_OFF_BY_TWO]),
        'oI': np.stack([OP_UNRELATED]),
        }
    _, off_by_two, _, _ = validate_candidate(ENTRY, top_unit_cell, _flat_M20(top_unit_cell))
    assert off_by_two


def test_found_requires_the_true_bravais_lattice():
    top_unit_cell = {'oP': np.stack([OP_UNRELATED, OP_EXACT])}
    found, off_by_two, incorrect_bl, _ = validate_candidate(
        ENTRY, top_unit_cell, _flat_M20(top_unit_cell)
        )
    assert found
    assert not incorrect_bl
    assert not off_by_two


def test_correct_cell_under_the_wrong_bravais_lattice_is_incorrect_bl():
    # a and c of the true cell read as a tetragonal pair. Correct geometry, wrong label, so it
    # must land in incorrect_bl and must not count as found.
    top_unit_cell = {'tP': np.array([[UNIT_CELL_TRUE[0], UNIT_CELL_TRUE[2]]])}
    found, _, incorrect_bl, _ = validate_candidate(
        ENTRY, top_unit_cell, _flat_M20(top_unit_cell)
        )
    assert incorrect_bl
    assert not found


def test_found_explainer_tracks_the_M20_cutoff():
    top_unit_cell = {'oP': np.stack([OP_UNRELATED])}
    _, _, _, below = validate_candidate(ENTRY, top_unit_cell, _flat_M20(top_unit_cell, 999.0))
    _, _, _, above = validate_candidate(ENTRY, top_unit_cell, _flat_M20(top_unit_cell, 1001.0))
    assert not below
    assert above


def test_label_candidates_shapes_and_alignment():
    top_unit_cell = {
        'oP': np.stack([OP_UNRELATED, OP_EXACT, OP_OFF_BY_TWO]),
        'tP': np.array([[UNIT_CELL_TRUE[0], UNIT_CELL_TRUE[2]]]),
        }
    correct, off_by_two = label_candidates(ENTRY, top_unit_cell)

    assert set(correct) == set(top_unit_cell)
    assert set(off_by_two) == set(top_unit_cell)
    for bl in top_unit_cell:
        assert correct[bl].shape == (top_unit_cell[bl].shape[0],)
        assert off_by_two[bl].shape == (top_unit_cell[bl].shape[0],)
        assert correct[bl].dtype == bool

    np.testing.assert_array_equal(correct['oP'], [False, True, False])
    np.testing.assert_array_equal(off_by_two['oP'], [False, False, True])
    np.testing.assert_array_equal(correct['tP'], [True])


def test_validate_candidate_reduces_label_candidates():
    # The two must not be able to disagree: validate_candidate is defined as the reduction.
    top_unit_cell = {
        'oP': np.stack([OP_UNRELATED, OP_OFF_BY_TWO]),
        'tP': np.array([[UNIT_CELL_TRUE[0], UNIT_CELL_TRUE[2]]]),
        'oI': np.stack([OP_EXACT]),
        }
    top_M20 = _flat_M20(top_unit_cell)
    correct, off_by_two = label_candidates(ENTRY, top_unit_cell)

    found, off_by_two_flag, incorrect_bl, _ = validate_candidate(ENTRY, top_unit_cell, top_M20)

    assert found == any(
        np.any(correct[bl]) for bl in top_unit_cell if bl == ENTRY['bravais_lattice']
        )
    assert incorrect_bl == any(
        np.any(correct[bl]) for bl in top_unit_cell if bl != ENTRY['bravais_lattice']
        )
    assert off_by_two_flag == any(np.any(off_by_two[bl]) for bl in top_unit_cell)


def test_rtol_is_threaded_through():
    # 3% off in a: outside the 1e-2 default, inside 5e-2.
    top_unit_cell = {'oP': np.stack([np.array([5.15, 7.0, 11.0])])}
    top_M20 = _flat_M20(top_unit_cell)
    assert not validate_candidate(ENTRY, top_unit_cell, top_M20)[0]
    assert validate_candidate(ENTRY, top_unit_cell, top_M20, rtol=5e-2)[0]
