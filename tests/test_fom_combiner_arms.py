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


def test_campaign_ones_structural_family_is_sixteen_features():
    """C2-F-039: campaign 1's `drop_structural` removes sixteen features, not the symmetry prior.

    This is the number the whole of C2-Q-013 turns on -- its 2.23 pp of operating point was read as
    the cost of the extinction group -- so it is pinned here rather than recounted by hand.

    Pinned against campaign 1's own column tuples rather than against the live group, because S12
    added two columns to the family (`N_cal_full` and `pool_size_full`) and the sixteen is a fact
    about the arm that produced the 2.23 pp, not about whatever the family holds today.
    """
    campaign1 = (('n_peaks', 'n_indexed', 'hkl_ref_length', 'n_entering', 'final_rank')
                 + ('N_cal', 'zone_dominance', 'V_over_Vcrit', 'delta_dewolff61', 'n_dewolff61',
                    'M_werner_max')
                 + ('log_volume', 'q2_max', 'n_peaks_available')
                 + ('bravais_lattice', 'spacegroup'))
    assert len(campaign1) == 16
    full, _ = FomCombiner.feature_specification(('raw', 'structural', 'context'))
    assert set(campaign1) <= set(full)


def test_S12_added_two_columns_to_the_structural_family_and_said_which():
    """`N_cal_full` and `pool_size_full`, each replacing something that was silently wrong.

    `N_cal_full` is `compute_all`'s count over [0, q_N]; the family's existing `N_cal` now comes
    from the merit sidecar and is `get_M_rev_sym`'s support over [q_I, q_N]. They agree on 0.07 %
    of real rows, so before the rename the family carried whichever one the loader happened to
    join. `pool_size_full` replaces the context group's `ctx_pool_size`, which counts RETAINED
    candidates and so means different things on the fit and report pools.
    """
    full, _ = FomCombiner.feature_specification(('raw', 'structural', 'context'))
    without, _ = FomCombiner.feature_specification(('raw', 'context'))
    assert len(full) - len(without) == 18
    assert {'N_cal', 'N_cal_full', 'pool_size_full'} <= set(full)
    assert 'ctx_pool_size' not in full
    assert 'ctx_pool_size' in FomCombiner.FORBIDDEN_COLUMNS


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


def test_the_counts_encoding_is_now_the_default_and_delta_is_not():
    """S12 reverses S04's deliberate default, on S04's own evidence.

    S04 kept both encodings out of `DEFAULT_GROUPS` so that no campaign-1 number silently changed
    meaning while it was measuring them. It then measured: the absence counts beat the 158-level
    categorical by +0.522 pp of operating point at p <= 0.004 at every fit seed (C2-F-041), and
    `delta_merit_extinction` reached +0.364 pp, significant at one seed of three, never beating
    counts. So `counts` joins the default and `delta` does not, and a campaign-2 model that omitted
    the counts would be an ablation rather than the headline.

    `spacegroup` is still IN the base space and is dropped by column in the arms, so
    `plus_spacegroup` is the absence of a drop rather than a second feature-set definition.
    """
    names, categorical = FomCombiner.feature_specification(FomCombiner.DEFAULT_GROUPS)
    assert set(FomCombiner.SYMMETRY_COUNTS) <= set(names)
    assert not (set(FomCombiner.SYMMETRY_DELTA) - {'n_groups_searched'}) & set(names)
    assert 'spacegroup' in categorical


def test_no_leakage_check_still_rejects_truth_columns():
    """The deny-list is what makes the entry table safe to pass to `evaluate` alongside features."""
    with pytest.raises(ValueError):
        FomCombiner.check_no_leakage(['M20', 'volume_true'])
    with pytest.raises(ValueError):
        FomCombiner.check_no_leakage(['M20', 'is_correct'])
