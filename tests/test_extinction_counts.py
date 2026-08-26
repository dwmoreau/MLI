"""Tests for S04's absence counts (`mlindex/utilities/ExtinctionCounts.py`).

The feature is a subtraction -- how many reference lines an extinction group deletes beyond its
Bravais lattice's own -- so what has to hold is that the committed lookup still describes the
reference lists it was built from, that the two routes to the count agree, and that triclinic is
the structural zero the whole diagnostic uses as its negative control.

The lookup exists so that inference never imports cctbx. So the tests that only READ it must run
in the inference-only environment, and only the regeneration test is allowed to require cctbx.
"""
import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from mlindex.utilities.ExtinctionCounts import (
    LATTICE_SYSTEM, absent_in_range, build_absence_counts, build_group_masks, get_absence_counts,
    get_generic_group, get_n_groups_searched,
    )

# The counts the S04 handoff states, kept as a fixture rather than a computation: if the reference
# lists or the extinction tables ever move, this is what says so.
EXPECTED_GROUPS = {'oP': 68, 'tP': 23, 'oC': 9, 'cP': 7, 'tI': 8, 'oI': 8, 'hP': 8, 'mP': 8,
                   'cI': 5, 'cF': 5, 'oF': 5, 'hR': 2, 'mC': 2, 'aP': 1}


def _hkl_ref(bravais_lattice):
    lattice_system = LATTICE_SYSTEM[bravais_lattice]
    return np.load(os.path.join(BASE, 'mlindex', 'models', f'{lattice_system}_1', 'data',
                                f'hkl_ref_{bravais_lattice}.npy'))


def _needs_cctbx():
    try:
        import cctbx.sgtbx  # noqa: F401
    except ImportError:
        return True
    return False


requires_cctbx = pytest.mark.skipif(_needs_cctbx(), reason='cctbx is not a runtime dependency')


def test_every_lattice_is_present():
    counts = get_absence_counts()
    assert set(counts) == set(LATTICE_SYSTEM)


@pytest.mark.parametrize('bravais_lattice,n_groups', sorted(EXPECTED_GROUPS.items()))
def test_group_counts_match_the_handoff(bravais_lattice, n_groups):
    assert get_n_groups_searched(bravais_lattice) == n_groups


def test_triclinic_is_the_structural_zero():
    """aP has one extinction group, so it can remove nothing and the merit cannot move.

    The whole diagnostic uses this as its negative control -- an effect on triclinic is a harness
    bug rather than a result -- so it is asserted here as well as in the analysis.
    """
    counts = get_absence_counts('aP')
    assert len(counts) == 1
    assert set(counts.values()) == {0}


@pytest.mark.parametrize('bravais_lattice', sorted(LATTICE_SYSTEM))
def test_exactly_one_generic_group_per_lattice(bravais_lattice):
    """The group that removes nothing is what makes `delta_merit_extinction` a plain subtraction.

    S03's point B scored the cell against the full reference list. That is only the merit at the
    generic group if exactly one group removes no lines and its list is `hkl_ref` itself -- so if
    this ever fails, every delta in S04 is measured against the wrong baseline.
    """
    counts = get_absence_counts(bravais_lattice)
    assert sum(1 for value in counts.values() if value == 0) == 1
    assert counts[get_generic_group(bravais_lattice)] == 0


@pytest.mark.parametrize('bravais_lattice', sorted(LATTICE_SYSTEM))
def test_counts_are_within_the_reference_list(bravais_lattice):
    counts = get_absence_counts(bravais_lattice)
    n_reference = _hkl_ref(bravais_lattice).shape[0]
    assert all(0 <= value < n_reference for value in counts.values())


@requires_cctbx
@pytest.mark.parametrize('bravais_lattice', sorted(LATTICE_SYSTEM))
def test_committed_lookup_regenerates(bravais_lattice):
    """The committed table must still be what the reference lists say it is."""
    rebuilt = build_absence_counts(_hkl_ref(bravais_lattice), bravais_lattice)
    assert rebuilt == get_absence_counts(bravais_lattice)


@requires_cctbx
@pytest.mark.parametrize('bravais_lattice', sorted(LATTICE_SYSTEM))
def test_masks_agree_with_counts_and_are_subsets(bravais_lattice):
    """Two routes to the same number, plus the subset property the in-range count depends on.

    `absent_in_range` counts dropped lines by masking the FULL list rather than by calculating the
    narrowed one, which is only valid if each group's list is a subset of `hkl_ref` in its own
    order.
    """
    hkl_ref = _hkl_ref(bravais_lattice)
    counts = build_absence_counts(hkl_ref, bravais_lattice)
    masks = build_group_masks(hkl_ref, bravais_lattice)
    assert set(masks) == set(counts)
    for key, mask in masks.items():
        assert int((~mask).sum()) == counts[key]
        assert mask.shape == (hkl_ref.shape[0],)


def test_absent_in_range_uses_a_strict_cutoff():
    """`get_M20` counts N over `q2_ref_calc < q2_calc[:, -1]`, so a line ON the cutoff is excluded.

    If this drifted to `<=`, the fraction and the merit would disagree about the same window.
    """
    q2_ref_calc = np.array([[1.0, 2.0, 3.0, 4.0]])
    keep = np.array([True, False, True, False])
    dropped, in_range = absent_in_range(q2_ref_calc, keep, np.array([3.0]))
    assert in_range[0] == 2      # 1.0 and 2.0; 3.0 sits on the cutoff and is excluded
    assert dropped[0] == 1       # only 2.0 is both dropped and inside


def test_absent_in_range_counts_only_dropped_lines():
    q2_ref_calc = np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]])
    keep = np.array([True, True, True])
    dropped, in_range = absent_in_range(q2_ref_calc, keep, np.array([10.0, 10.0]))
    assert list(dropped) == [0, 0]
    assert list(in_range) == [3, 3]
