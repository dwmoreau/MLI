"""The S14 driver's arm table: what each arm reads, and that the tree reference is S12's."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training import FomCombiner  # noqa: E402
from mlindex.scripts import run_fom_combiner as s12  # noqa: E402
from mlindex.scripts import run_fom_neural_score as driver  # noqa: E402


def test_arm_names_are_unique_and_groups_are_known():
    names = [arm[0] for arm in driver.ARMS]
    assert len(names) == len(set(names))
    for name, kind, groups, drop, weight, purpose in driver.ARMS:
        assert kind in ('network', 'tree')
        FomCombiner.feature_specification(groups, drop=drop)      # raises on an unknown group
        assert weight in ('sampling_weight', None)
        assert purpose


def test_the_base_network_reads_dwmm_fifty_inputs_and_nothing_classical():
    spec = dict((arm[0], arm) for arm in driver.ARMS)['network']
    names, categorical = FomCombiner.feature_specification(spec[2], drop=spec[3])
    assert set(FomCombiner.PRIOR_ENTRY) <= set(names)
    assert set(FomCombiner.PRIOR_VOLUME) <= set(names)
    assert set(FomCombiner.ASSIGNMENT_PEAKS) <= set(names)
    assert 'log_volume' in names and categorical == ('bravais_lattice',)
    classical = set(FomCombiner.RAW_MERITS) | set(FomCombiner.PROBATION_MERITS) \
        | set(FomCombiner.context_names()) | {'M20', 'Minfo', 'spacegroup'}
    assert not (set(names) & classical)
    # 14 + 14 + 2 entropies + 20 + log_volume + bravais_lattice
    assert len(names) == 52


def test_the_tree_reference_is_s12s_plus_probation_feature_set():
    spec = dict((arm[0], arm) for arm in driver.ARMS)['tree']
    names, _ = FomCombiner.feature_specification(spec[2], drop=spec[3])
    s12_spec = dict((arm[0], arm) for arm in s12.ARMS)['plus_probation']
    s12_names, _ = FomCombiner.feature_specification(s12.arm_groups(s12_spec[1]),
                                                     drop=s12_spec[2])
    assert tuple(names) == tuple(s12_names)


def test_super_additivity_arms_partition_the_blocks():
    arms = dict((arm[0], arm) for arm in driver.ARMS)
    a_only, _ = FomCombiner.feature_specification(arms['drop_B'][2], drop=arms['drop_B'][3])
    b_only, _ = FomCombiner.feature_specification(arms['drop_A'][2], drop=arms['drop_A'][3])
    both, _ = FomCombiner.feature_specification(arms['network'][2], drop=arms['network'][3])
    assert set(FomCombiner.ASSIGNMENT_PEAKS).isdisjoint(a_only)
    assert set(FomCombiner.PRIOR_ENTRY).isdisjoint(b_only)
    assert set(a_only) | set(b_only) == set(both)


def test_the_unweighted_arm_is_the_only_unweighted_one():
    unweighted = [arm[0] for arm in driver.ARMS if arm[4] is None]
    assert unweighted == ['unweighted_fit']


def test_the_fullscale_tree_row_count_is_the_record_s():
    assert driver.TREE_FULLSCALE_ROWS == 2_381_244


def test_reduce_refuses_to_run_without_arms():
    args = driver._parse_args(['--stage', 'reduce'])
    with pytest.raises(SystemExit):
        driver.run_reduce(args)


def test_suffix_namespaces_the_models_directory():
    args = driver._parse_args(['--stage', 'fit', '--suffix', '_seed777'])
    assert str(driver.models_directory(args)).endswith('fom_neural_score_seed777')
