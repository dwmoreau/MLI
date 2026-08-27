"""Mighell-Santoro degeneracy, against the paper it comes from.

Campaign 1 shipped `is_degenerate` null and so excluded degenerates at a *measured* zero rather
than a known one (R7, C2-Q-002). These are the tests that stop that recurring silently: if the
detector stopped detecting, `test_santoros_worked_example_is_degenerate` fails rather than the
column quietly filling with False.
"""

import numpy as np
import pytest

from mlindex.utilities import LatticeDegeneracy as LD


def _cell_from_scalars(S11, S22, S33, S23, S13, S12):
    """A cell (a, b, c, alpha, beta, gamma) in radians with the given Santoro scalars."""
    a, b, c = np.sqrt(S11), np.sqrt(S22), np.sqrt(S33)
    return (a, b, c,
            float(np.arccos(S23 / (b * c))),
            float(np.arccos(S13 / (a * c))),
            float(np.arccos(S12 / (a * b))))


def _santoro_boundary_cell(slack_factor=1.0):
    """Santoro & Mighell (1970) p. 126, condition (3c) at its boundary.

    With S23 = S22/2 in force, the reduction is unambiguous while S12 != 2*S13. At S12 = 2*S13
    "the two cells are indistinguishable", and the paper's matrix (9) is then an end-centred
    monoclinic lattice while the cells it came from are triclinic. `slack_factor` scales S12 away
    from that boundary.
    """
    S11, S22, S33 = 25.0, 36.0, 49.0
    S23 = S22 / 2
    S13 = 5.0
    return _cell_from_scalars(S11, S22, S33, S23, S13, 2 * S13 * slack_factor)


GENERIC_TRICLINIC = (5.0, 7.0, 11.0,
                     float(np.radians(83.0)), float(np.radians(74.0)), float(np.radians(66.0)))


def test_scalars_follow_the_papers_equation_one():
    # S11 = a.a, S22 = b.b, S33 = c.c, S23 = b.c, S13 = a.c, S12 = a.b
    cell = (3.0, 4.0, 5.0, np.radians(80.0), np.radians(85.0), np.radians(70.0))
    s = LD.scalars_from_cell(cell)
    assert np.isclose(s['S11'], 9.0)
    assert np.isclose(s['S22'], 16.0)
    assert np.isclose(s['S33'], 25.0)
    assert np.isclose(s['S23'], 4.0 * 5.0 * np.cos(np.radians(80.0)))
    assert np.isclose(s['S13'], 3.0 * 5.0 * np.cos(np.radians(85.0)))
    assert np.isclose(s['S12'], 3.0 * 4.0 * np.cos(np.radians(70.0)))


def test_cell_type_follows_the_sign_of_the_off_diagonal_scalars():
    # "the S_ij (i != j) are either all acute or all obtuse ... If one or more of the S_ij is
    # zero, the cell will be considered to be of Type II."
    acute = (5.0, 6.0, 7.0, np.radians(70.0), np.radians(75.0), np.radians(80.0))
    obtuse = (5.0, 6.0, 7.0, np.radians(110.0), np.radians(105.0), np.radians(100.0))
    right = (5.0, 6.0, 7.0, np.radians(90.0), np.radians(90.0), np.radians(90.0))
    assert LD.cell_type(LD.scalars_from_cell(acute)) == 1
    assert LD.cell_type(LD.scalars_from_cell(obtuse)) == 2
    assert LD.cell_type(LD.scalars_from_cell(right)) == 2


def test_santoros_worked_example_is_degenerate():
    cell = _santoro_boundary_cell()
    assert LD.degenerate_conditions(cell) == ('3c',)
    flagged, accidental, systematic = LD.is_degenerate(cell, 'aP')
    assert flagged is True
    assert accidental == ('3c',)
    assert systematic == ()


def test_moving_off_the_boundary_removes_the_degeneracy():
    # The left-column equality S23 = S22/2 still holds; only the right-column inequality is no
    # longer tight. That is exactly the distinction the definition rests on, so it must show up.
    cell = _santoro_boundary_cell(slack_factor=0.85)
    assert LD.triggered_conditions(cell) == ('3c',)      # still triggered
    assert LD.degenerate_conditions(cell) == ()          # but not on the boundary
    assert LD.is_degenerate(cell, 'aP')[0] is False


def test_a_generic_triclinic_cell_is_not_degenerate():
    assert LD.degenerate_conditions(GENERIC_TRICLINIC) == ()
    assert LD.is_degenerate(GENERIC_TRICLINIC, 'aP')[0] is False


@pytest.mark.parametrize('bravais_lattice, cell', [
    ('cP', (5.0, 5.0, 5.0, np.pi / 2, np.pi / 2, np.pi / 2)),
    ('cI', (5.0, 5.0, 5.0, np.pi / 2, np.pi / 2, np.pi / 2)),
    ('cF', (5.0, 5.0, 5.0, np.pi / 2, np.pi / 2, np.pi / 2)),
    ('hP', (5.0, 5.0, 8.0, np.pi / 2, np.pi / 2, np.radians(120.0))),
    ('tP', (5.0, 5.0, 8.0, np.pi / 2, np.pi / 2, np.pi / 2)),
    ])
def test_symmetry_required_equalities_are_systematic_not_degenerate(bravais_lattice, cell):
    # "may occur accidentally or systematically depending on the particular geometrical
    # properties of a lattice". A cubic lattice has a = b = c *by construction*, so the equality
    # it triggers is symmetry the indexer is entitled to find, not a degeneracy.
    flagged, accidental, systematic = LD.is_degenerate(cell, bravais_lattice)
    assert flagged is False
    assert accidental == ()


def test_the_systematic_probe_holds_the_lattice_type():
    # If the probe broke a = b for a tetragonal lattice it would report every such equality
    # accidental, and every high-symmetry entry would be flagged.
    rng = np.random.default_rng(0)
    cell = (5.0, 5.0, 8.0, np.pi / 2, np.pi / 2, np.pi / 2)
    for probe in LD._perturbed_cells(cell, 'tP', rng):
        assert np.isclose(probe[0], probe[1])            # a = b preserved
        assert np.isclose(probe[3], np.pi / 2)           # angles untouched
        assert not np.isclose(probe[0], 5.0)             # but a really did move


def test_reduction_uses_the_primitive_setting():
    # Niggli reduction is defined on the primitive cell. Reducing a centred conventional cell as
    # though it were primitive describes a different lattice, so the volume must fall by the
    # centring multiplicity.
    conventional = (8.0, 8.0, 8.0, np.pi / 2, np.pi / 2, np.pi / 2)

    def volume(cell):
        a, b, c, al, be, ga = cell
        return a * b * c * np.sqrt(
            1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
            + 2 * np.cos(al) * np.cos(be) * np.cos(ga))

    assert np.isclose(volume(LD.reduced_cell(conventional, 'cP')), volume(conventional))
    assert np.isclose(volume(LD.reduced_cell(conventional, 'cI')), volume(conventional) / 2)
    assert np.isclose(volume(LD.reduced_cell(conventional, 'cF')), volume(conventional) / 4)


def test_every_bravais_lattice_has_a_holohedry_and_free_parameters():
    from mlindex.command_line.run import BRAVAIS_LATTICES
    assert set(LD.BRAVAIS_HOLOHEDRY) == set(BRAVAIS_LATTICES)
    assert set(LD.FREE_PARAMETERS) == set(BRAVAIS_LATTICES)


@pytest.mark.parametrize('slack_factor, expect_higher_symmetry', [(1.0, True), (0.85, False)])
def test_flagged_cells_have_higher_metric_symmetry_by_cctbxs_own_search(slack_factor,
                                                                       expect_higher_symmetry):
    """An independent check of the claim the definition actually makes.

    `is_degenerate` says a lattice of *higher metric symmetry* indexes the same peak positions.
    cctbx's `lattice_symmetry.group` searches for that symmetry by a completely different route,
    so agreement is evidence the condition algebra is not merely self-consistent. At a tight
    tolerance the boundary cell comes back C-centred monoclinic -- which is what Santoro's matrix
    (9) is said to be -- and the off-boundary cell comes back triclinic.
    """
    from cctbx import sgtbx, uctbx
    from cctbx.sgtbx import lattice_symmetry

    cell = _santoro_boundary_cell(slack_factor=slack_factor)
    flagged = LD.is_degenerate(cell, 'aP')[0]
    assert flagged is expect_higher_symmetry

    a, b, c, al, be, ga = cell
    unit_cell = uctbx.unit_cell((a, b, c, np.degrees(al), np.degrees(be), np.degrees(ga)))
    group = lattice_symmetry.group(unit_cell, max_delta=0.05)
    symbol = str(sgtbx.space_group_info(group=group)).split('(')[0].strip()

    if expect_higher_symmetry:
        assert group.order_z() > 1
        assert symbol.startswith('C')          # end-centred monoclinic, per the paper
    else:
        assert group.order_z() == 1
        assert symbol == 'P 1'
