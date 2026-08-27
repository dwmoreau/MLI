"""`label_frame` on a small pool: the columns SCHEMA.md specifies, and nothing null that should not be.

The batch labeller itself is gated against the scalar routine in
`tests/test_candidate_validation_batch.py`. What is tested here is the wiring: that the frame-level
function slices the truth per lattice system, that it groups correctly when one pool holds several
entries and several systems, and that `hkl_true_in_basis` appears exactly on the rows where a basis
change exists.
"""

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomBenchmark

RIGHT = np.pi / 2


def _entries():
    """Two patterns with different true lattices, so the grouping is actually exercised."""
    hkl_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.int16).reshape(-1)
    return pd.DataFrame([
        {'entry_id': 'ORTHO1', 'q2_digest': 'aaaa', 'condition_bundle': 'c2_error1_cont0',
         'unit_cell_true': np.array([8.0, 11.0, 14.0, RIGHT, RIGHT, RIGHT]),
         'volume_true': 8.0 * 11.0 * 14.0, 'bravais_lattice_true': 'oP',
         'lattice_system_true': 'orthorhombic', 'hkl_true': hkl_true},
        {'entry_id': 'CUBE1', 'q2_digest': 'bbbb', 'condition_bundle': 'c2_error1_cont0',
         'unit_cell_true': np.array([9.0, 9.0, 9.0, RIGHT, RIGHT, RIGHT]),
         'volume_true': 729.0, 'bravais_lattice_true': 'cP',
         'lattice_system_true': 'cubic', 'hkl_true': hkl_true},
        ])


def _candidates():
    def row(entry_id, digest, bravais_lattice, system, unit_cell, xnn, volume):
        return {'entry_id': entry_id, 'q2_digest': digest,
                'condition_bundle': 'c2_error1_cont0', 'bravais_lattice': bravais_lattice,
                'lattice_system': system, 'candidate_id': 0,
                'unit_cell': np.asarray(unit_cell, dtype=float),
                'xnn': np.asarray(xnn, dtype=float), 'volume': float(volume)}

    return pd.DataFrame([
        # Correct, identity setting.
        row('ORTHO1', 'aaaa', 'oP', 'orthorhombic', [8.0, 11.0, 14.0], [1 / 64, 1 / 121, 1 / 196],
            8.0 * 11.0 * 14.0),
        # Correct, axes permuted -- a different setting of the same lattice.
        row('ORTHO1', 'aaaa', 'oP', 'orthorhombic', [14.0, 8.0, 11.0], [1 / 196, 1 / 64, 1 / 121],
            8.0 * 11.0 * 14.0),
        # Wrong.
        row('ORTHO1', 'aaaa', 'oP', 'orthorhombic', [5.0, 6.0, 30.0], [0.04, 0.028, 0.0011], 900.0),
        # A sub-cell: half the axes, so off-by-two and not correct.
        row('CUBE1', 'bbbb', 'cP', 'cubic', [4.5], [1 / 20.25], 91.125),
        # Correct.
        row('CUBE1', 'bbbb', 'cP', 'cubic', [9.0], [1 / 81], 729.0),
        ])


def test_label_frame_populates_every_label_column():
    labelled = FomBenchmark.label_frame(_candidates(), _entries())
    for column in FomBenchmark.LABEL_COLUMNS:
        assert column in labelled.columns, column
    assert FomBenchmark.has_labels(labelled)

    assert labelled['is_correct'].tolist() == [True, True, False, False, True]
    assert labelled['is_off_by_two'].tolist() == [False, False, False, True, False]
    # A correct cubic candidate must not also be flagged as its own sub-cell: the scalar routine
    # returns before it reaches the multiplier grid, and every grid contains the identity.
    assert not labelled.loc[4, 'is_off_by_two']


def test_hkl_true_in_basis_appears_on_correct_rows_and_follows_the_setting():
    labelled = FomBenchmark.label_frame(_candidates(), _entries())
    reexpressed = labelled['hkl_true_in_basis']

    assert reexpressed[2] is None, 'a wrong cell has no basis change, so no re-expressed indices'
    assert reexpressed[3] is None, 'an off-by-two cell is a scaling, not a basis change'

    # Identity setting: unchanged.
    assert np.asarray(reexpressed[0]).reshape(-1, 3).tolist() == [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    # Permuted setting: the indices permute with the axes. The truth's a is the candidate's
    # second axis, b its third, c its first.
    assert np.asarray(reexpressed[1]).reshape(-1, 3).tolist() == [
        [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]]
    assert np.asarray(reexpressed[0]).dtype == np.int16


def test_the_distance_to_truth_is_only_defined_where_the_lattice_matches():
    candidates = _candidates()
    # An mP candidate against an oP truth: xnn vectors of different length and meaning.
    candidates.loc[5] = {'entry_id': 'ORTHO1', 'q2_digest': 'aaaa',
                         'condition_bundle': 'c2_error1_cont0', 'bravais_lattice': 'mP',
                         'lattice_system': 'monoclinic', 'candidate_id': 1,
                         'unit_cell': np.array([8.0, 11.0, 14.0, RIGHT]),
                         'xnn': np.array([0.015, 0.008, 0.005, 0.0]), 'volume': 1232.0}
    labelled = FomBenchmark.label_frame(candidates, _entries())
    assert np.isfinite(labelled.loc[0, 'xnn_distance_to_truth'])
    assert np.isnan(labelled.loc[5, 'xnn_distance_to_truth'])
    # The volume ratio is defined for every candidate, matching lattice or not.
    assert labelled['volume_ratio_to_truth'].notna().all()


def test_an_empty_pool_still_returns_the_label_columns():
    empty = _candidates().iloc[:0]
    labelled = FomBenchmark.label_frame(empty, _entries())
    assert list(FomBenchmark.LABEL_COLUMNS) == [c for c in FomBenchmark.LABEL_COLUMNS
                                                if c in labelled.columns]
    assert labelled.shape[0] == 0


def test_label_frame_matches_the_scalar_routine_row_for_row():
    from mlindex.optimization.CandidateValidation import validate_candidate_known_bl

    candidates, entries = _candidates(), _entries()
    labelled = FomBenchmark.label_frame(candidates, entries)
    truth = entries.set_index('entry_id')
    for position, row in candidates.iterrows():
        correct, off_by_two = validate_candidate_known_bl(
            np.asarray(truth.loc[row['entry_id'], 'unit_cell_true'], dtype=float).copy(),
            np.asarray(row['unit_cell'], dtype=float).copy(),
            row['bravais_lattice'], rtol=1e-2)
        assert bool(correct) == bool(labelled.loc[position, 'is_correct'])
        assert bool(off_by_two) == bool(labelled.loc[position, 'is_off_by_two'])
