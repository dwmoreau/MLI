"""Tests for S03 Phase 2b's confirmation harness (`run_fom_prune_confirm.py`).

The harness exists to reproduce what `run.py` prints -- each Bravais lattice's top 20 pooled and
sorted by M20 -- so the property worth testing is that its ranking convention is `run.py`'s, not
merely a sensible one. Everything else it does is a run of the real optimizer, which no unit test
can stand in for.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def _load():
    path = os.path.join(BASE, 'mlindex', 'scripts', 'run_fom_prune_confirm.py')
    spec = importlib.util.spec_from_file_location('run_fom_prune_confirm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIRM = _load()


def test_rank_is_zero_based_and_counts_from_the_highest_M20():
    """Rank 0 is the candidate `run.py` names first, which is the highest M20."""
    M20 = np.array([3.0, 9.0, 5.0, 1.0])
    assert CONFIRM.best_correct_rank(M20, [False, True, False, False]) == 0
    assert CONFIRM.best_correct_rank(M20, [False, False, True, False]) == 1
    assert CONFIRM.best_correct_rank(M20, [True, False, False, False]) == 2
    assert CONFIRM.best_correct_rank(M20, [False, False, False, True]) == 3


def test_the_best_correct_candidate_is_the_one_reported():
    """Several correct candidates can survive; the rank reported is the best of them."""
    M20 = np.array([2.0, 8.0, 6.0])

    assert CONFIRM.best_correct_rank(M20, [True, False, True]) == 1


def test_absence_is_minus_one_and_not_confusable_with_rank_zero():
    """-1 has to be a distinct outcome: 'no correct cell anywhere' is a generation failure and is
    kept in its own bucket, never folded in with a bad rank."""
    assert CONFIRM.best_correct_rank(np.array([5.0, 3.0]), [False, False]) == -1
    assert CONFIRM.best_correct_rank(np.array([5.0, 3.0]), [True, False]) == 0


def test_ties_keep_the_order_the_lattices_were_assembled_in():
    """`run.py` sorts with pandas' default, which is stable, so a tie leaves assembly order alone.

    Without a stable sort the rank of a tied correct candidate would depend on the sort
    implementation rather than on the pipeline, and the two threshold arms could differ for a
    reason that has nothing to do with the threshold.
    """
    M20 = np.array([7.0, 7.0, 7.0, 7.0])

    assert CONFIRM.best_correct_rank(M20, [False, False, True, False]) == 2
    assert CONFIRM.best_correct_rank(M20, [False, True, True, False]) == 1


def test_top_n_figures_are_restrictions_of_one_number():
    """top-1, top-10 and top-20 all come from the same stored rank, so they cannot disagree."""
    ranks = pd.Series([-1, 0, 3, 9, 10, 19, 25])

    top1 = ranks.between(0, 0)
    top10 = ranks.between(0, 9)
    top20 = ranks.between(0, 19)

    assert top1.sum() == 1 and top10.sum() == 3 and top20.sum() == 5
    # Monotone by construction: anything in the top 1 is in the top 10 is in the top 20.
    assert (top1 <= top10).all() and (top10 <= top20).all()


def test_the_production_top_n_is_the_shipped_value():
    """The harness reproduces `run.py`'s answer, so it must use `run.py`'s candidate count."""
    assert CONFIRM.N_TOP_CANDIDATES == 20
