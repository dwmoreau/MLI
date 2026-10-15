import numpy as np
import pytest
from pathlib import Path

from conftest import load_test_case, _TEST_DATA_DIR
from mlindex.utilities.Reindexing import selling_reduction

EXPECTED_DIR = Path(__file__).parent / "expected"


def test_selling_reduction(unique_test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in (
        load_test_case(row) for _, row in unique_test_metadata.iterrows()
    ):
        result_uc, _, _ = selling_reduction(unit_cell[np.newaxis])
        expected = np.load(EXPECTED_DIR / f"selling_reduction_{bl}.npy")
        np.testing.assert_allclose(
            result_uc[0],
            expected,
            rtol=1e-10,
            err_msg=f"selling_reduction mismatch for {bl}",
        )


def _monoclinic_cells(rng, n):
    """Random monoclinic cells, many of them very oblique, (n, 6) in radians."""
    lengths = rng.uniform(4, 25, size=(n, 3))
    beta = np.deg2rad(rng.uniform(90, 170, size=n))
    return np.column_stack([lengths, np.full(n, np.pi / 2), beta, np.full(n, np.pi / 2)])


def _ac_mesh(cells):
    """a and c of each full cell as two-dimensional vectors in the plane normal to b."""
    a = np.column_stack([cells[:, 0], np.zeros(len(cells))])
    c = np.column_stack([cells[:, 2] * np.cos(cells[:, 4]), cells[:, 2] * np.sin(cells[:, 4])])
    return a, c


@pytest.mark.parametrize('centring', ['P', 'I'])
def test_monoclinic_standardization_keeps_the_lattice_and_reduces_it(centring):
    import itertools
    from mlindex.utilities.Reindexing import monoclinic_standardization, niggli_cell, same_lattice

    rng = np.random.default_rng(20261014)
    cells = _monoclinic_cells(rng, 200)
    reduced = monoclinic_standardization(cells, centring)

    assert np.all(reduced[:, 4] >= np.pi / 2 - 1e-12)
    np.testing.assert_allclose(reduced[:, [1, 3, 5]], cells[:, [1, 3, 5]])
    volume = lambda uc: uc[:, 0] * uc[:, 1] * uc[:, 2] * np.sin(uc[:, 4])
    np.testing.assert_allclose(volume(reduced), volume(cells), rtol=1e-9)
    degrees = lambda uc: np.column_stack([uc[:, :3], np.rad2deg(uc[:, 3:])])
    for before, after in zip(degrees(cells), degrees(reduced)):
        assert same_lattice(niggli_cell(before, centring), niggli_cell(after, centring),
                            1e-6, 1e-4), (before, after)

    # No basis change of the a-c mesh with small entries, among those that keep the centring
    # point at (a + c) / 2 for an I cell, gives a shorter pair of axes.
    a, c = _ac_mesh(reduced)
    best = np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1)
    for m in itertools.product(range(-3, 4), repeat=4):
        m = np.array(m).reshape(2, 2)
        if abs(round(np.linalg.det(m))) != 1:
            continue
        # The new axes are m[0, 0] a + m[0, 1] c and m[1, 0] a + m[1, 1] c; the centring point
        # (a + c) / 2 stays a half-sum of them when both columns of m sum to an odd number.
        if centring == 'I' and not np.all(m.sum(axis=0) % 2 == 1):
            continue
        new_a = m[0, 0] * a + m[0, 1] * c
        new_c = m[1, 0] * a + m[1, 1] * c
        product = np.linalg.norm(new_a, axis=1) * np.linalg.norm(new_c, axis=1)
        assert np.all(product >= best * (1 - 1e-9)), m


@pytest.mark.parametrize('centring', ['P', 'I'])
def test_monoclinic_standardization_is_idempotent(centring):
    from mlindex.utilities.Reindexing import monoclinic_standardization

    cells = _monoclinic_cells(np.random.default_rng(7), 100)
    once = monoclinic_standardization(cells, centring)
    np.testing.assert_allclose(monoclinic_standardization(once, centring), once, atol=1e-12)


def _keeps_centring(m, centring):
    """Does the a-c basis change m (new a = m[0, 0] a + m[0, 1] c, new c = m[1, 0] a + m[1, 1] c)
    keep the centring point: (a + c) / 2 for I, (a + b) / 2 for C?"""
    if centring == 'I':
        return np.all(m.sum(axis=0) % 2 == 1)
    if centring == 'C':
        return m[0, 0] % 2 == 1 and m[0, 1] % 2 == 0
    return True


def test_a_C_centred_mesh_reduces_to_its_shortest_axes_keeping_the_centring():
    import itertools
    from mlindex.utilities.Reindexing import niggli_cell, reduce_monoclinic_mesh, same_lattice

    cells = _monoclinic_cells(np.random.default_rng(3), 200)
    a, c = _ac_mesh(cells)
    a, c = reduce_monoclinic_mesh(a, c, 'C')
    best = np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1)
    for m in itertools.product(range(-3, 4), repeat=4):
        m = np.array(m).reshape(2, 2)
        if abs(round(np.linalg.det(m))) != 1 or not _keeps_centring(m, 'C'):
            continue
        product = (np.linalg.norm(m[0, 0] * a + m[0, 1] * c, axis=1)
                   * np.linalg.norm(m[1, 0] * a + m[1, 1] * c, axis=1))
        assert np.all(product >= best * (1 - 1e-9)), m
    beta = np.rad2deg(np.arccos(np.abs(np.sum(a * c, axis=1))
                                / (np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1))))
    for before, la, lc, b in zip(cells, np.linalg.norm(a, axis=1), np.linalg.norm(c, axis=1),
                                 180 - beta):
        stored = [*before[:3], 90, np.rad2deg(before[4]), 90]
        assert same_lattice(niggli_cell(stored, 'C'), niggli_cell([la, before[1], lc, 90, b, 90], 'C'),
                            1e-6, 1e-4)


_MONOCLINIC_KEYS = {
    'mP': ['P 1 - 1 e.g. P 1 2 1', 'P 1 21 1 e.g. P 1 21 1', 'P 1 c 1 e.g. P 1 c 1',
           'P 1 a 1 e.g. P 1 a 1', 'P 1 n 1 e.g. P 1 n 1', 'P 1 21/c 1 e.g. P 1 21/c 1',
           'P 1 21/a 1 e.g. P 1 21/a 1', 'P 1 21/n 1 e.g. P 1 21/n 1'],
    'mC': ['I 1 - 1 e.g. I 1 2 1', 'I 1 a 1 e.g. I 1 a 1'],
    }


def _allowed_d_spacings(cell, symbol, d_min):
    """The d-spacings of the reflections a space group allows, to d_min, sorted."""
    from cctbx import crystal, miller

    symmetry = crystal.symmetry(unit_cell=cell, space_group_symbol=symbol)
    allowed = miller.build_set(symmetry, anomalous_flag=False, d_min=d_min
                               ).remove_systematic_absences()
    return np.sort(allowed.d_spacings().data().as_numpy_array())


@pytest.mark.parametrize('bravais_lattice', ['mP', 'mC'])
def test_a_monoclinic_cell_prints_reduced_with_its_symbol_moved_with_it(bravais_lattice):
    """The printed cell is the stored lattice, reduced, beta >= 90, mC in the C setting; and the
    printed cell and space group allow exactly the reflections the stored ones do, d-spacing for
    d-spacing -- which checks the basis change and the symbol together, glide letters included."""
    from mlindex.utilities.Reindexing import niggli_cell, printed_setting, same_lattice

    stored_centring, printed_centring = ('I', 'C') if bravais_lattice == 'mC' else ('P', 'P')
    for cell in _monoclinic_cells(np.random.default_rng(11), 25):
        cell = [*cell[:3], 90.0, float(np.rad2deg(cell[4])), 90.0]
        for key in _MONOCLINIC_KEYS[bravais_lattice]:
            printed, printed_key = printed_setting(cell, bravais_lattice, key)
            assert printed[4] >= 90 - 1e-9
            np.testing.assert_allclose([printed[3], printed[5]], [90, 90], atol=1e-9)
            assert same_lattice(niggli_cell(cell, stored_centring),
                                niggli_cell(printed, printed_centring), 1e-6, 1e-4)
            extinction, _, example = printed_key.partition(' e.g. ')
            assert extinction.split()[0] == example.split()[0] == printed_centring
            assert ('-' in key.split()) == ('-' in extinction.split())
            # Not a simple fraction of an axis, so no reflection sits on the cut.
            d_min = min(cell[:3]) / 4.37
            np.testing.assert_allclose(
                _allowed_d_spacings(printed, example, d_min),
                _allowed_d_spacings(cell, key.partition(' e.g. ')[2], d_min), rtol=1e-9)


def test_a_triclinic_cell_prints_its_niggli_cell():
    from cctbx import uctbx
    from mlindex.utilities.Reindexing import printed_setting

    cell = (9.0, 8.0, 7.0, 95.0, 100.0, 115.0)
    printed, key = printed_setting(cell, 'aP', 'P - e.g. P 1')
    assert key == 'P - e.g. P 1'
    assert uctbx.unit_cell(printed).is_niggli_cell()
    np.testing.assert_allclose(uctbx.unit_cell(printed).volume(), uctbx.unit_cell(cell).volume())


def test_other_lattices_print_as_stored():
    from mlindex.utilities.Reindexing import printed_setting

    cell = (5.0, 5.0, 8.0, 90.0, 90.0, 120.0)
    assert printed_setting(cell, 'hP', 'P 6 - - e.g. P 6') == (cell, 'P 6 - - e.g. P 6')
