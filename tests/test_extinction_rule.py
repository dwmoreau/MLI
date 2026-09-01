"""S11: choosing a candidate's extinction group on something other than M20.

`assign_extinction_group` decides which calculated lines a cell should produce, by argmax over up
to 68 groups, and it REBINDS `best_M20` to the value at the winner -- so it sets the merit that
deduplication, `final_rank` and `run.py`'s printed list all read. Changing what that argmax runs
on therefore changes the shipped answer, which is why the criterion is a default-off flag and why
the first thing these tests pin is that the default does nothing at all.

What could be silently wrong here, in the order it would hurt:

  * The flag changes the default path. `score` is an ALIAS of `M20` when the criterion is 'M20',
    so this ought to be structural -- but "ought to be" is what a test is for.
  * The tie-break is skipped. Under a floored `M_rev`, ties at 0.0 are EXPECTED: deleting lines
    is what an extinction group does and what drives `N_cal` under the floor. `np.argmax` takes
    the first maximum and the generic zero-absence group is first in all fourteen key lists, so
    without an explicit tie-break every unsupported candidate silently receives the generic group
    and the rule looks like it made a choice (C2-F-059).
  * The offline sweep drifts from production. The `fom` branch already carries a hand-written
    second copy of this loop; `test_the_sweep_agrees_with_production` is what stops a third.
"""
import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from mlindex.model_training.FomBenchmark import extinction_group_sweep
from mlindex.optimization.Candidates import Candidates
from mlindex.utilities.FigureOfMerits import (
    EXTINCTION_CRITERIA, argmax_extinction_group, extinction_criterion_score, get_M20,
    get_M_rev_sym, m_rev_support_floor,
    )
from mlindex.utilities.Q2Calculator import Q2Calculator


def _candidates(criterion=None, n=24, seed=11, bravais_lattice='oP',
                lattice_system='orthorhombic'):
    """A real `Candidates` over a synthetic peak list, on a lattice with many groups.

    oP by default because it searches 68 groups, which is where every hazard in this module lives;
    a two-group lattice would pass these tests without exercising anything.
    """
    rng = np.random.default_rng(seed)
    hkl_ref = np.load(os.path.join(BASE, 'mlindex', 'models', f'{lattice_system}_1', 'data',
                                   f'hkl_ref_{bravais_lattice}.npy'))
    xnn_true = np.array([[1/8.0**2, 1/9.0**2, 1/11.0**2]])
    q2_ref = Q2Calculator(lattice_system=lattice_system, hkl=hkl_ref, tensorflow=False,
                          representation='xnn').get_q2(xnn_true)[0]
    q2_obs = np.sort(q2_ref[q2_ref > 0])[:20]
    xnn = xnn_true + rng.normal(0, 2e-5, size=(n, 3))
    opt_params = {'minimum_uc': 2.0, 'maximum_uc': 60.0, 'assignment_threshold': 0.95,
                  'figure_of_merit': 'M20'}
    if criterion is not None:
        opt_params['extinction_criterion'] = criterion
    return Candidates(q2_obs=q2_obs, xnn=xnn, hkl_ref=hkl_ref, lattice_system=lattice_system,
                      bravais_lattice=bravais_lattice, opt_params=opt_params,
                      rng=np.random.default_rng(seed), fom=None, zero_error=False,
                      wavelength=None)


def _assigned(criterion=None, **kwargs):
    candidates = _candidates(criterion, **kwargs)
    candidates.assign_extinction_group()
    return candidates


def test_the_criterion_is_M20_unless_asked_for():
    assert _candidates().extinction_criterion == 'M20'


def test_an_unknown_criterion_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match='extinction_criterion'):
        _candidates('M_revv')


def test_the_default_path_is_identical_to_naming_M20_explicitly():
    """The alias has to be an alias. If this fails the flag is not free."""
    absent, explicit = _assigned(None), _assigned('M20')
    assert absent.best_spacegroup == explicit.best_spacegroup
    assert np.array_equal(absent.best_M20, explicit.best_M20)
    assert np.array_equal(absent.best_hkl, explicit.best_hkl)


def test_the_default_path_does_not_record_tie_diagnostics():
    """The tie-break is not entered under M20, so its bookkeeping must not exist either."""
    assert not hasattr(_assigned(None), 'n_ties_at_best_extinction')
    assert hasattr(_assigned('M_rev'), 'n_ties_at_best_extinction')


@pytest.mark.parametrize('criterion', EXTINCTION_CRITERIA)
def test_every_declared_criterion_runs_and_reports_M20_at_its_winner(criterion):
    """`best_M20` is M20 at the chosen group -- never the criterion's own value.

    That is what keeps the arms comparable: the rule changes WHICH group is chosen, and the
    reported merit stays on M20's scale so a ranking difference is attributable to the choice.
    """
    candidates = _assigned(criterion)
    assert len(candidates.best_spacegroup) == candidates.n
    assert np.all(np.isfinite(candidates.best_M20))
    # M20 at any group is at most the maximum over groups, which is what the incumbent reports.
    assert np.all(candidates.best_M20 <= _assigned('M20').best_M20 + 1e-12)


def test_M20_scoring_is_the_untouched_function():
    rng = np.random.default_rng(3)
    q2_obs = np.sort(rng.uniform(0.01, 0.5, 20))
    q2_calc = np.sort(rng.uniform(0.01, 0.5, (5, 20)), axis=1)
    q2_ref = np.sort(rng.uniform(0.01, 0.8, (5, 300)), axis=1)
    score, n_cal = extinction_criterion_score('M20', q2_obs, q2_calc, q2_ref.copy())
    assert n_cal is None
    assert np.array_equal(score, get_M20(q2_obs, q2_calc, q2_ref.copy()))


def test_the_floor_is_read_from_get_M_rev_sym_rather_than_restated():
    """One definition of the support floor. A second copy of `10` is how the two drift."""
    import inspect
    assert m_rev_support_floor() == inspect.signature(
        get_M_rev_sym).parameters['min_n_cal'].default


def test_neither_merit_modifies_the_reference_lines_it_is_given():
    """Pinned because four comments in the tree still claim `get_M20` does.

    It used to zero the excluded lines in place with `np.putmask`, which forced callers to pass a
    copy or to call it last. `lines_below_cutoff` replaced that, so the ordering constraint the
    S11 handoff leads with is gone -- and the stale comments would otherwise reintroduce a
    defensive copy of a 48 MB array inside a 68-group loop.
    """
    rng = np.random.default_rng(5)
    q2_obs = np.sort(rng.uniform(0.01, 0.5, 20))
    q2_calc = np.sort(rng.uniform(0.01, 0.5, (4, 20)), axis=1)
    q2_ref = np.sort(rng.uniform(0.01, 0.8, (4, 250)), axis=1)
    untouched = q2_ref.copy()
    get_M20(q2_obs, q2_calc, q2_ref)
    assert np.array_equal(q2_ref, untouched)
    get_M_rev_sym(q2_obs, q2_calc, q2_ref)
    assert np.array_equal(q2_ref, untouched)


def test_ties_are_broken_on_M20_and_not_on_group_order():
    """The generic group is first in every key list, so an unbroken tie would always pick it."""
    n_groups = 5
    score = np.zeros((3, n_groups))          # every group tied at the floored zero
    M20 = np.array([[1.0, 9.0, 2.0, 3.0, 4.0],
                    [5.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 2.0, 3.0, 4.0, 9.0]])
    n_cal = np.zeros((3, n_groups))
    chosen, n_ties, n_floored = argmax_extinction_group('M_rev', score, M20, n_cal)
    assert list(chosen) == [1, 0, 4]
    assert list(n_ties) == [n_groups]*3
    assert list(n_floored) == [n_groups]*3
    # And the incumbent's own path is untouched by the tie-break machinery.
    assert list(argmax_extinction_group('M20', score, M20, n_cal)[0]) == [1, 0, 4]


def test_the_sweep_agrees_with_production_for_every_criterion():
    """The anti-drift gate in miniature: one loop, two callers, no second implementation."""
    reference = _candidates()
    keys, winners, _, _, _, _ = extinction_group_sweep(
        reference.q2_obs, reference.best_xnn, reference.lattice_system,
        reference.bravais_lattice, hkl_ref=reference.hkl_ref,
        )
    for criterion in EXTINCTION_CRITERIA:
        production = _assigned(criterion)
        assert [keys[i] for i in winners[criterion]] == production.best_spacegroup, criterion
