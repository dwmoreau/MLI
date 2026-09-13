"""The port-list checker, which exists because a check that can only say "no" is worse than none."""

import pytest

from mlindex.scripts import check_port_list


def test_a_symbol_in_the_tree_is_found_and_an_impossible_one_is_not():
    assert check_port_list.is_present('mcnemar')
    assert not check_port_list.is_present(check_port_list.absent_sentinel())


def test_the_absent_sentinel_is_generated_rather_than_written_down():
    """The first sentinel was a literal in the checker's own source, so once that file was tracked
    `git grep` found it there and the checker declared itself broken -- on main, after a merge, in
    code whose whole purpose is to not give a confident wrong answer."""
    first, second = check_port_list.absent_sentinel(), check_port_list.absent_sentinel()
    assert first != second
    assert not check_port_list.is_present(first)


def test_whole_word_matching_so_a_prefix_is_not_a_false_positive():
    """`reduce_many` must not make `reduce_many_more` look ported, and a substring of a real name
    must not make an absent symbol look present."""
    assert check_port_list.is_present('reduce_many')
    assert not check_port_list.is_present('reduce_man')


def test_the_checker_verifies_itself_before_reporting():
    """The first version used `\\b` in a POSIX ERE pattern, which is not a word boundary, so every
    symbol read as absent -- 27 of 27, including ones the caller was importing. That output looked
    like a finding rather than a broken tool."""
    check_port_list.verify_the_checker()

    broken = check_port_list.is_present
    try:
        check_port_list.is_present = lambda symbol, paths=('mlindex/',): False
        with pytest.raises(SystemExit, match='known to be present'):
            check_port_list.verify_the_checker()
        check_port_list.is_present = lambda symbol, paths=('mlindex/',): True
        with pytest.raises(SystemExit, match='impossible symbol'):
            check_port_list.verify_the_checker()
    finally:
        check_port_list.is_present = broken


def test_it_reports_the_absences_rather_than_failing_on_them(capsys):
    """An absence is present, renamed, or dropped by decision -- a list to account for, not a
    failure. Exiting non-zero would make a session suppress it."""
    sentinel = check_port_list.absent_sentinel()
    assert check_port_list.main(['--symbols', f'mcnemar,{sentinel}']) == 0
    out = capsys.readouterr().out
    assert '1 of 2 present' in out
    assert sentinel in out
