import json
import numpy as np
import pytest
from pathlib import Path

from conftest import load_test_case, BL_TO_LATTICE_SYSTEM, _TEST_DATA_DIR

from mlindex.utilities.UnitCellTools import (
    get_xnn_from_unit_cell,
    get_unit_cell_from_xnn,
    get_unit_cell_volume,
    get_hkl_matrix,
    get_partial_unit_cell,
    fix_unphysical,
)


def _xnn(unit_cell, lattice_system):
    """Convert full 6-component unit cell to partial xnn for the given lattice system."""
    uc_p = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
    return get_xnn_from_unit_cell(
        uc_p[np.newaxis], partial_unit_cell=True, lattice_system=lattice_system
    )


from mlindex.utilities.FigureOfMerits import (
    get_M20_from_xnn,
    get_M20_likelihood_from_xnn,
)
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.gsas import load_pkslst

EXPECTED_DIR = Path(__file__).parent / "expected"


# ---------------------------------------------------------------------------
# Helpers / parametrize IDs
# ---------------------------------------------------------------------------


def _cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


def _bl_ids(test_metadata):
    return list(test_metadata["bravais lattice"])


# ---------------------------------------------------------------------------
# xnn <-> unit_cell roundtrip
# ---------------------------------------------------------------------------


def test_xnn_roundtrip(test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(test_metadata):
        uc2d = unit_cell[np.newaxis]
        xnn = get_xnn_from_unit_cell(uc2d, lattice_system=lattice_system)
        recovered = get_unit_cell_from_xnn(xnn, lattice_system=lattice_system)[0]
        np.testing.assert_allclose(
            recovered, unit_cell, rtol=1e-10, err_msg=f"roundtrip failed for {bl}"
        )


def test_xnn_roundtrip_random_triclinic():
    rng = np.random.default_rng(12345)
    n = 100
    xnn = rng.uniform(0, 1, size=(n, 6))
    xnn = fix_unphysical(xnn=xnn, rng=rng, lattice_system="triclinic")
    uc = get_unit_cell_from_xnn(xnn, lattice_system="triclinic")
    xnn2 = get_xnn_from_unit_cell(uc, lattice_system="triclinic")
    np.testing.assert_allclose(xnn2, xnn, rtol=1e-10)


def test_xnn_roundtrip_grid_search():
    a_vals = np.linspace(2, 20, 5)
    b_vals = np.linspace(2, 20, 5)
    c_vals = np.linspace(2, 20, 5)
    angles_deg = [70.0, 90.0, 110.0]
    angles_rad = np.array(angles_deg) * np.pi / 180

    unit_cells = []
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                for alpha in angles_rad:
                    for beta in angles_rad:
                        for gamma in angles_rad:
                            uc = np.array([a, b, c, alpha, beta, gamma])
                            uc2d = uc[np.newaxis]
                            xnn = get_xnn_from_unit_cell(
                                uc2d, lattice_system="triclinic"
                            )
                            if not np.any(np.isnan(xnn)):
                                unit_cells.append((uc, xnn))

    for uc, xnn in unit_cells:
        recovered = get_unit_cell_from_xnn(xnn, lattice_system="triclinic")[0]
        np.testing.assert_allclose(recovered, uc, rtol=1e-10)


# ---------------------------------------------------------------------------
# Unit cell volume
# ---------------------------------------------------------------------------


def test_unit_cell_volume(unique_test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        computed = get_unit_cell_volume(unit_cell[np.newaxis])[0]
        expected = np.load(EXPECTED_DIR / f"unit_cell_volume_{bl}.npy")
        np.testing.assert_allclose(
            computed, expected, rtol=1e-10, err_msg=f"volume mismatch for {bl}"
        )


# ---------------------------------------------------------------------------
# get_hkl_matrix
# ---------------------------------------------------------------------------


def test_get_hkl_matrix(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        result = get_hkl_matrix(hkl_ref, lattice_system)
        fname = expected_dir / f"hkl_matrix_{bl}.npy"
        np.testing.assert_array_equal(
            result, np.load(fname), err_msg=f"hkl_matrix mismatch for {bl}"
        )


# ---------------------------------------------------------------------------
# Q2Calculator
# ---------------------------------------------------------------------------


def test_q2_calculator(unique_test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        expected = np.load(EXPECTED_DIR / f"q2_calculator_{bl}.npy")
        np.testing.assert_allclose(
            q2_calc, expected, rtol=1e-10, err_msg=f"q2_calculator mismatch for {bl}"
        )


# ---------------------------------------------------------------------------
# fast_assign
# ---------------------------------------------------------------------------


def test_fast_assign(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_ref = np.sum(hkl2 * xnn, axis=1)[np.newaxis]
        result = fast_assign(q2_obs.astype(np.float64), q2_ref.astype(np.float64))
        fname = expected_dir / f"fast_assign_{bl}.npy"
        np.testing.assert_array_equal(
            result, np.load(fname), err_msg=f"fast_assign mismatch for {bl}"
        )


# ---------------------------------------------------------------------------
# Figure of merits
# ---------------------------------------------------------------------------


def test_m20(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        hkl_assign = fast_assign(
            q2_obs.astype(np.float64), q2_calc[np.newaxis].astype(np.float64)
        )[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]
        result = get_M20_from_xnn(q2_obs, xnn, hkl_assigned, hkl_ref, lattice_system)
        expected = np.load(expected_dir / f"m20_{bl}.npy")
        np.testing.assert_allclose(
            result, expected, rtol=1e-10, err_msg=f"M20 mismatch for {bl}"
        )


def test_m20_likelihood(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        hkl_assign = fast_assign(
            q2_obs.astype(np.float64), q2_calc[np.newaxis].astype(np.float64)
        )[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]
        log_lik, prob, M = get_M20_likelihood_from_xnn(
            q2_obs, xnn, hkl_assigned, lattice_system, bl
        )
        expected = np.load(expected_dir / f"m20_likelihood_{bl}.npy")
        np.testing.assert_allclose(
            np.stack([log_lik, M.flatten()]),
            expected,
            rtol=1e-10,
            err_msg=f"M20_likelihood mismatch for {bl}",
        )


# ---------------------------------------------------------------------------
# Peak loading / unit conversion
# ---------------------------------------------------------------------------


def test_load_pkslst(test_metadata):
    for _, row in test_metadata.iterrows():
        peak_file = str(row["peak list file"])
        if not peak_file.endswith(".pkslst"):
            continue
        case_name = peak_file.replace(".pkslst", "")
        wavelength = float(row["wavelength"])
        pkslst_path = _TEST_DATA_DIR / case_name / peak_file

        result = load_pkslst(str(pkslst_path), wavelength)
        expected = np.load(EXPECTED_DIR / f"load_pkslst_{case_name}.npy")
        np.testing.assert_allclose(
            np.sort(result),
            expected,
            rtol=1e-12,
            err_msg=f"load_pkslst mismatch for {case_name}",
        )



# ---------------------------------------------------------------------------
# SpaceGroups: get_spacegroup_hkl_ref
# ---------------------------------------------------------------------------

_SPACEGROUP_HKL_REF_TEST_HKL = np.array(
    [
        [h, k, l]
        for h in range(8)
        for k in range(8)
        for l in range(8)
        if not (h == 0 and k == 0 and l == 0)
    ],
    dtype=np.int32,
)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64])
def test_spacegroup_hkl_ref_dtypes(dtype):
    # The production reference lists are float64 under mlindex/models and int64 under
    # mlindex/data, and Boost.Python accepts numpy int32 but rejects both of those. An int32-only
    # fixture therefore passes while every real run raises, which is how a broken systematic
    # absence check survived in assign_extinction_group. Pin all three dtypes.
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    hkl = _SPACEGROUP_HKL_REF_TEST_HKL.astype(dtype)
    for bl in BL_TO_LATTICE_SYSTEM:
        result = get_spacegroup_hkl_ref(hkl, bl)
        reference = get_spacegroup_hkl_ref(_SPACEGROUP_HKL_REF_TEST_HKL, bl)
        assert list(result.keys()) == list(reference.keys()), f"{bl}: key mismatch"
        for key in reference:
            np.testing.assert_array_equal(
                np.asarray(result[key], dtype=np.int64), reference[key].astype(np.int64),
                err_msg=f"{bl} [{key}]: {dtype} disagrees with int32",
            )


def test_spacegroup_hkl_ref():
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    for bl in BL_TO_LATTICE_SYSTEM:
        result = get_spacegroup_hkl_ref(_SPACEGROUP_HKL_REF_TEST_HKL, bl)
        with open(EXPECTED_DIR / f"spacegroup_hkl_ref_{bl}_keys.json") as f:
            expected_keys = json.load(f)
        assert list(result.keys()) == expected_keys, f"{bl}: key mismatch"
        for idx, key in enumerate(expected_keys):
            expected = np.load(EXPECTED_DIR / f"spacegroup_hkl_ref_{bl}_{idx}.npy")
            np.testing.assert_array_equal(
                result[key], expected, err_msg=f"{bl} [{key}]: hkl array mismatch"
            )


def test_load_peaks_npy(test_metadata):
    from mlindex.command_line.run import _load_peaks
    import types

    for _, row in test_metadata.iterrows():
        peak_file = str(row["peak list file"])
        if not (peak_file.endswith(".pkslst") or peak_file.endswith(".pklst")):
            continue
        bl = str(row["bravais lattice"])
        case_name = peak_file.replace(".pkslst", "").replace(".pklst", "")
        npy_path = _TEST_DATA_DIR / case_name / f"{case_name}_peak_list.npy"
        args = types.SimpleNamespace(
            peaks=None,
            peak_file=str(npy_path),
            peak_units="q2",
            wavelength=None,
            zero_error=False,
        )
        result = _load_peaks(args)
        expected = np.sort(np.load(npy_path))[:20]
        assert len(result) <= 20, f"{bl}: more than 20 peaks loaded"
        assert result[0] <= result[-1], f"{bl}: peaks not sorted"
        np.testing.assert_array_equal(
            result, expected, err_msg=f"load_peaks_npy mismatch for {bl}"
        )
