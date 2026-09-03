"""S12's results document: the sections that report the model the campaign settled on.

The document is generated, never transcribed (PROTOCOL section 5), which makes the generator the
only thing standing between a stored table and a claim someone quotes. These tests cover the two
sections added once the feature search had run and the transfer stage existed -- both of which
report a DIFFERENT model from the one the main ladder fitted, and both of which are easy to point
at the wrong file without noticing.
"""

import pandas as pd
import pytest

from mlindex.model_training import FomCombiner
from mlindex.scripts import run_fom_combiner as driver
from mlindex.scripts import run_fom_combiner_report as report


# ---------------------------------------------------------------------------------------------
# which model is measured
# ---------------------------------------------------------------------------------------------
def test_transfer_and_cost_default_to_the_settled_arm_not_the_first_one_fitted():
    """`base` was hardcoded while it was the only fitted arm, and `core` is 8.6 pp better.

    A transfer or cost number quoted for `base` describes a 29-feature model nobody is going to
    ship. The default has to be the arm the search settled on, and it has to be nameable.
    """
    args = driver._parse_args(['--stage', 'transfer'])
    assert args.transfer_arm == 'core'
    assert set(driver.DROP_SETS) == {'base', 'lean', 'core'}


def test_each_drop_set_leaves_the_feature_count_the_record_claims():
    expected = {'base': 29, 'lean': 16, 'core': 14}
    for arm, drop in driver.DROP_SETS.items():
        names, _ = FomCombiner.feature_specification(driver.arm_groups(()), drop=drop)
        assert len(names) == expected[arm], f'{arm} is {len(names)} features, not {expected[arm]}'


def test_core_carries_neither_M20_nor_M_tilde():
    """The headline claim of C2-F-134, asserted where a refactor would break it silently."""
    names, _ = FomCombiner.feature_specification(driver.arm_groups(()), drop=driver.CORE_DROP)
    assert 'M20' not in names and 'M_tilde' not in names
    assert 'M_sym' in names and 'bravais_lattice' in names


# ---------------------------------------------------------------------------------------------
# the transfer section
# ---------------------------------------------------------------------------------------------
def _transfer_table(tmp_path, name, delta):
    frame = pd.DataFrame([
        dict(fitted_without='c2_error2_cont0', reported_on='c2_error2_cont0',
             is_the_unseen_condition=True, all_bundles=0.84, held_out=0.84 + delta/100,
             delta_pp=delta, arm_features='core'),
        dict(fitted_without='c2_error2_cont0', reported_on='c2_error1_cont0',
             is_the_unseen_condition=False, all_bundles=0.84, held_out=0.84,
             delta_pp=0.0, arm_features='core'),
    ])
    frame.to_csv(tmp_path/f'S12_combiner_{name}.csv', index=False)
    return frame


def test_the_transfer_section_prefers_the_settled_arms_table(tmp_path):
    """Three arm-named files can coexist; the document must not quote whichever sorts first."""
    _transfer_table(tmp_path, 'condition_transfer_base', -9.0)
    _transfer_table(tmp_path, 'condition_transfer_core', -1.0)
    loaded = report._load_transfer(tmp_path, 'S12_combiner')
    assert float(loaded['delta_pp'].min()) == -1.0


def test_the_transfer_section_falls_back_to_an_unsuffixed_table(tmp_path):
    """A table written before the stage was parameterised must not be silently dropped."""
    _transfer_table(tmp_path, 'condition_transfer', -2.5)
    assert report._load_transfer(tmp_path, 'S12_combiner') is not None


def test_the_transfer_section_says_what_it_is_not_when_it_has_a_table(tmp_path):
    """C2-R-008 and C2-R-024, in the document rather than only in the record.

    The handoff asks for transfer across error LAWS. No error-law bundle exists, and only three of
    nine condition bundles can be reported with exact ranks. A reader who takes this table for the
    check the handoff named would over-read it, so the section must carry both bounds.
    """
    _transfer_table(tmp_path, 'condition_transfer_core', -1.0)
    text = '\n'.join(report._transfer(report._load_transfer(tmp_path, 'S12_combiner')))
    assert 'C2-R-008' in text and 'C2-R-024' in text
    assert 'not transfer across error laws' in text
    assert 'three of nine' in text
    assert '-1.00 pp' in text


def test_an_unrun_transfer_reports_the_gate_as_open_rather_than_omitting_it():
    text = '\n'.join(report._transfer(None))
    assert 'condition 5 is open' in text


# ---------------------------------------------------------------------------------------------
# the feature-search section
# ---------------------------------------------------------------------------------------------
def _search_ladder(tmp_path, suffix, seeds, levels, summary_rows):
    for seed in seeds:
        pd.DataFrame([dict(arm=arm, operating_point=value, top10=value, n_features=n)
                      for arm, value, n in levels]).to_csv(
            tmp_path/f'S12_combiner_main_table{suffix}_seed{seed}.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(
        tmp_path/f'S12_combiner_seed_summary{suffix}.csv', index=False)


def test_the_search_section_marks_an_arm_whose_contrast_changes_sign(tmp_path):
    """An arm is settled only if it keeps its sign at every seed (C2-F-061).

    `m20_only` was significant at every seed in OPPOSITE directions, so one seed would have given
    a confident wrong answer. A table that shows only the mean hides exactly that.
    """
    _search_ladder(
        tmp_path, '_search', (12345, 777),
        [('lean', 0.849, 16), ('lean_minus_X_N', 0.858, 15), ('lean_minus_bravais_lattice', 0.809, 15)],
        [dict(arm='lean_minus_X_N', metric='operating_point', scope='aggregate',
              delta_mean=0.90, delta_min=-0.38, delta_max=1.57),
         dict(arm='lean_minus_bravais_lattice', metric='operating_point', scope='aggregate',
              delta_mean=-4.05, delta_min=-4.65, delta_max=-3.21)])
    text = '\n'.join(report._search(
        report._load_search(tmp_path, 'S12_combiner', ('_search',), (12345, 777))))
    assert '`lean_minus_X_N` | 15 | 85.80' in text
    assert 'sign flips' in text
    # The settled one must NOT be marked, or the mark means nothing.
    bravais = [line for line in text.splitlines() if 'bravais_lattice' in line][0]
    assert 'sign flips' not in bravais
    # The reference arm has no contrast against itself.
    assert '| `lean` | 16 | 84.90 | 84.90 | reference |' in text


def test_the_search_section_reads_only_the_seeds_it_is_given(tmp_path):
    """A missing seed's table is skipped, not counted as zero."""
    _search_ladder(tmp_path, '_search2', (12345,), [('core', 0.8478, 14)], [])
    loaded = report._load_search(tmp_path, 'S12_combiner', ('_search2',), (12345, 777, 20260826))
    assert len(loaded) == 1
    assert int(loaded[0][1]['n_seeds'].iloc[0]) == 1


def test_no_ladder_reports_as_not_run_rather_than_raising(tmp_path):
    assert report._load_search(tmp_path, 'S12_combiner', ('_search',), (12345,)) == []
    assert 'Not run.' in '\n'.join(report._search([]))


def test_the_arm_comes_from_the_file_name_when_the_table_predates_the_column(tmp_path):
    """The transfer stage was parameterised after its first run, so an early table has no column."""
    frame = pd.DataFrame([dict(fitted_without='c2_error1_cont0', reported_on='c2_error1_cont0',
                               is_the_unseen_condition=True, all_bundles=0.84, held_out=0.83,
                               delta_pp=-1.0)])
    frame.to_csv(tmp_path/'S12_combiner_condition_transfer_core.csv', index=False)
    loaded = report._load_transfer(tmp_path, 'S12_combiner')
    assert loaded['arm_features'].iloc[0] == 'core'
    assert '`core` feature set' in '\n'.join(report._transfer(loaded))
