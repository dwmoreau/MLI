"""Tests for the S04 Phase 2 additions to the ported combiner stack.

Three things arrived on this branch at S04 Phase 2 and none of them are campaign 1's: a column-level
`drop` on the feature specification, the two rival symmetry encodings as their own feature groups,
and the **removal** of the CNRS weighting from `FomMetrics`.

The last of those is a standing rule rather than a preference (PROTOCOL section 3 rules 1, 6 and 11),
so it is tested the way a rule is: not "unweighted is the default" but "there is no way to ask for
anything else".
"""
import os
import sys
import inspect

import numpy as np
import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from mlindex.model_training import FomCombiner
from mlindex.model_training import FomMetrics


# ---------------------------------------------------------------------------------------------
# the CNRS weighting is gone, and cannot be reinstated by an argument
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['CNRS_WEIGHTS', 'CNRS_TABLE1', 'CNRS_TOTAL',
                                  'CNRS_OPERATING_POINT', 'CNRS_CEILING', '_weight_series',
                                  '_weight_coverage', 'weighted_mean', '_weighted_rate'])
def test_the_cnrs_weighting_is_absent(name):
    """PROTOCOL section 3 rule 1: no aggregate is reweighted to the sealed benchmark before S16."""
    assert not hasattr(FomMetrics, name), (
        f'{name} is back in FomMetrics. Campaign 1 defaulted `evaluate` to weights="cnrs", so a '
        'caller who omitted the argument reweighted silently. S16 takes this back deliberately, '
        'with a CHERRY_PICK row.')


@pytest.mark.parametrize('function,parameter', [
    (FomMetrics.evaluate, 'weights'),
    (FomMetrics.entry_context, 'weights'),
    (FomMetrics.threshold_curve, 'weighted'),
    (FomMetrics.select_threshold, 'weighted'),
    ])
def test_no_public_entry_point_takes_a_weight(function, parameter):
    assert parameter not in inspect.signature(function).parameters


def test_pooled_rate_counts_every_entry_once():
    """The aggregate is pooled over entries, not averaged over lattices.

    Unit weights would still have been a reweighting -- to a uniform-over-lattices distribution --
    and would carry the same effective-sample loss rule 6 objects to. Two lattices, one with nine
    entries and one with one: pooled gives 9/10, a macro average would give 1/2.
    """
    numerator = np.array([[9.0, 0.0]])
    denominator = np.array([9.0, 1.0])
    assert FomMetrics._pooled_rate(numerator, denominator)[0] == pytest.approx(0.9)


def test_unweighted_mean_is_a_plain_mean_over_entries():
    frame = pd.DataFrame({'bravais_lattice': ['mP']*9 + ['cF'],
                          'flag': [1.0]*9 + [0.0]})
    assert FomMetrics.unweighted_mean(frame, 'flag') == pytest.approx(0.9)


# ---------------------------------------------------------------------------------------------
# the column-level drop, which is what lets `spacegroup` be measured on its own
# ---------------------------------------------------------------------------------------------

def test_drop_removes_exactly_one_column():
    base, base_categorical = FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS)
    dropped, categorical = FomCombiner.feature_specification(
        FomCombiner.DEFAULT_GROUPS, drop=('spacegroup',))
    assert len(dropped) == len(base) - 1
    assert 'spacegroup' in base and 'spacegroup' not in dropped
    assert set(base) - set(dropped) == {'spacegroup'}
    # It must leave the categorical list too, or the design matrix declares a column it lacks.
    assert 'spacegroup' in base_categorical and 'spacegroup' not in categorical
    assert 'bravais_lattice' in categorical


def test_dropping_a_column_that_is_not_there_is_an_error():
    """Silently ignoring it would make an arm that did not ablate what its label says it did."""
    with pytest.raises(ValueError, match='cannot drop'):
        FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS, drop=('no_such_feature',))


def test_the_structural_family_is_sixteen_features():
    """C2-F-039: campaign 1's `drop_structural` removes sixteen features, not the symmetry prior.

    This is the number the whole of C2-Q-013 turns on -- its 2.23 pp of operating point was read as
    the cost of the extinction group -- so it is pinned here rather than recounted by hand.
    """
    full, _ = FomCombiner.feature_specification(('raw', 'structural', 'context'))
    without, _ = FomCombiner.feature_specification(('raw', 'context'))
    assert len(full) - len(without) == 16
    assert {'spacegroup', 'bravais_lattice', 'final_rank', 'n_entering', 'log_volume'} <= set(full)


# ---------------------------------------------------------------------------------------------
# the two rival encodings, as groups
# ---------------------------------------------------------------------------------------------

def test_counts_and_delta_are_separate_droppable_groups():
    counts, _ = FomCombiner.feature_specification(('raw', 'structural', 'context', 'counts'))
    delta, _ = FomCombiner.feature_specification(('raw', 'structural', 'context', 'delta'))
    assert set(FomCombiner.SYMMETRY_COUNTS) <= set(counts)
    assert set(FomCombiner.SYMMETRY_DELTA) <= set(delta)
    # They share the look-elsewhere count and nothing else: the delta is a selected maximum and
    # `n_groups_searched` is what it has to be judged against, so it travels with both.
    shared = set(FomCombiner.SYMMETRY_COUNTS) & set(FomCombiner.SYMMETRY_DELTA)
    assert shared == {'n_groups_searched'}


def test_symmetry_features_never_reach_the_matrix_by_default():
    """Neither encoding is in DEFAULT_GROUPS, so no earlier number silently changes meaning."""
    names, _ = FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS)
    assert not (set(FomCombiner.SYMMETRY_COUNTS) - {'n_groups_searched'}) & set(names)
    assert not set(FomCombiner.SYMMETRY_DELTA) & set(names)


def test_no_leakage_check_still_rejects_truth_columns():
    """The deny-list is what makes the entry table safe to pass to `evaluate` alongside features."""
    with pytest.raises(ValueError):
        FomCombiner.check_no_leakage(['M20', 'volume_true'])
    with pytest.raises(ValueError):
        FomCombiner.check_no_leakage(['M20', 'is_correct'])
