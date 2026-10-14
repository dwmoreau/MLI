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
