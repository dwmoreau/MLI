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
    return get_xnn_from_unit_cell(uc_p[np.newaxis], partial_unit_cell=True,
                                   lattice_system=lattice_system)
from mlindex.utilities.FigureOfMerits import get_M20_from_xnn, get_M20_likelihood_from_xnn
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.gsas import load_pkslst

EXPECTED_DIR = Path(__file__).parent / 'expected'


# ---------------------------------------------------------------------------
# Helpers / parametrize IDs
# ---------------------------------------------------------------------------

def _cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


def _bl_ids(test_metadata):
    return list(test_metadata['bravais lattice'])


# ---------------------------------------------------------------------------
# xnn <-> unit_cell roundtrip
# ---------------------------------------------------------------------------

def test_xnn_roundtrip(test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(test_metadata):
        uc2d = unit_cell[np.newaxis]
        xnn = get_xnn_from_unit_cell(uc2d, lattice_system=lattice_system)
        recovered = get_unit_cell_from_xnn(xnn, lattice_system=lattice_system)[0]
        np.testing.assert_allclose(recovered, unit_cell, rtol=1e-10,
                                   err_msg=f'roundtrip failed for {bl}')


def test_xnn_roundtrip_random_triclinic():
    rng = np.random.default_rng(12345)
    n = 100
    xnn = rng.uniform(0, 1, size=(n, 6))
    xnn = fix_unphysical(xnn=xnn, rng=rng, lattice_system='triclinic')
    uc = get_unit_cell_from_xnn(xnn, lattice_system='triclinic')
    xnn2 = get_xnn_from_unit_cell(uc, lattice_system='triclinic')
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
                            xnn = get_xnn_from_unit_cell(uc2d, lattice_system='triclinic')
                            if not np.any(np.isnan(xnn)):
                                unit_cells.append((uc, xnn))

    for uc, xnn in unit_cells:
        recovered = get_unit_cell_from_xnn(xnn, lattice_system='triclinic')[0]
        np.testing.assert_allclose(recovered, uc, rtol=1e-10)


# ---------------------------------------------------------------------------
# Unit cell volume
# ---------------------------------------------------------------------------

def test_unit_cell_volume(test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(test_metadata):
        a, b, c = unit_cell[:3]
        calpha = np.cos(unit_cell[3])
        cbeta  = np.cos(unit_cell[4])
        cgamma = np.cos(unit_cell[5])
        expected = a * b * c * np.sqrt(1 - calpha**2 - cbeta**2 - cgamma**2 + 2*calpha*cbeta*cgamma)
        computed = get_unit_cell_volume(unit_cell[np.newaxis])[0]
        np.testing.assert_allclose(computed, expected, rtol=1e-10,
                                   err_msg=f'volume mismatch for {bl}')


# ---------------------------------------------------------------------------
# get_hkl_matrix
# ---------------------------------------------------------------------------

def test_get_hkl_matrix(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f'hkl_ref_{bl}.npy')
        result = get_hkl_matrix(hkl_ref, lattice_system)
        fname = expected_dir / f'hkl_matrix_{bl}.npy'
        np.testing.assert_array_equal(result, np.load(fname),
                                      err_msg=f'hkl_matrix mismatch for {bl}')


# ---------------------------------------------------------------------------
# Q2Calculator
# ---------------------------------------------------------------------------

def test_q2_calculator(test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f'hkl_ref_{bl}.npy')
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        q2_calc_sorted = np.sort(q2_calc)

        # Most observed peaks should be close to some calculated peak.
        # Allow up to 2 unmatched peaks per pattern (impurity/absence peaks).
        unmatched = [
            q2 for q2 in q2_obs
            if np.min(np.abs(q2_calc_sorted - q2)) / q2 >= 5e-2
        ]
        assert len(unmatched) <= 2, (
            f'{bl}: {len(unmatched)} unmatched peaks (> 5% off): {unmatched}'
        )


# ---------------------------------------------------------------------------
# fast_assign
# ---------------------------------------------------------------------------

def test_fast_assign(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f'hkl_ref_{bl}.npy')
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_ref = np.sum(hkl2 * xnn, axis=1)[np.newaxis]
        result = fast_assign(q2_obs.astype(np.float64), q2_ref.astype(np.float64))
        fname = expected_dir / f'fast_assign_{bl}.npy'
        np.testing.assert_array_equal(result, np.load(fname),
                                      err_msg=f'fast_assign mismatch for {bl}')


# ---------------------------------------------------------------------------
# Figure of merits
# ---------------------------------------------------------------------------

def test_m20(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f'hkl_ref_{bl}.npy')
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        hkl_assign = fast_assign(q2_obs.astype(np.float64),
                                  q2_calc[np.newaxis].astype(np.float64))[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]
        result = get_M20_from_xnn(q2_obs, xnn, hkl_assigned, hkl_ref, lattice_system)
        expected = np.load(expected_dir / f'm20_{bl}.npy')
        np.testing.assert_allclose(result, expected, rtol=1e-10,
                                   err_msg=f'M20 mismatch for {bl}')


def test_m20_likelihood(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f'hkl_ref_{bl}.npy')
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        hkl_assign = fast_assign(q2_obs.astype(np.float64),
                                  q2_calc[np.newaxis].astype(np.float64))[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]
        log_lik, prob, M = get_M20_likelihood_from_xnn(q2_obs, xnn, hkl_assigned,
                                                        lattice_system, bl)
        expected = np.load(expected_dir / f'm20_likelihood_{bl}.npy')
        np.testing.assert_allclose(
            np.stack([log_lik, M.flatten()]), expected, rtol=1e-10,
            err_msg=f'M20_likelihood mismatch for {bl}'
        )


# ---------------------------------------------------------------------------
# Peak loading / unit conversion
# ---------------------------------------------------------------------------

def test_load_pkslst(test_metadata):
    for _, row in test_metadata.iterrows():
        peak_file = str(row['peak list file'])
        if not peak_file.endswith('.pkslst'):
            continue
        case_name = peak_file.replace('.pkslst', '')
        wavelength = float(row['wavelength'])
        pkslst_path = _TEST_DATA_DIR / case_name / peak_file

        result = load_pkslst(str(pkslst_path), wavelength)
        expected = np.load(EXPECTED_DIR / f'load_pkslst_{case_name}.npy')
        np.testing.assert_allclose(np.sort(result), expected, rtol=1e-12,
                                   err_msg=f'load_pkslst mismatch for {case_name}')


def test_convert_d_to_q2():
    from mlindex.command_line.run import _convert_to_q2
    peaks = np.array([5.0])
    result = _convert_to_q2(peaks, 'd', wavelength=None)
    np.testing.assert_allclose(result, np.array([1.0 / 25.0]), rtol=1e-15)


def test_convert_q_to_q2():
    from mlindex.command_line.run import _convert_to_q2
    peaks = np.array([0.2])
    result = _convert_to_q2(peaks, 'q', wavelength=None)
    np.testing.assert_allclose(result, np.array([0.04]), rtol=1e-15)


def test_convert_2theta_to_q2(test_metadata):
    import ast
    import pandas as pd
    from mlindex.command_line.run import _convert_to_q2

    for _, row in test_metadata.iterrows():
        peak_file = str(row['peak list file'])
        if not peak_file.endswith('.csv'):
            continue
        bl = str(row['bravais lattice'])
        wavelength = float(row['wavelength'])
        pattern_name = peak_file.replace('.csv', '')
        path = _TEST_DATA_DIR / pattern_name / peak_file
        df = pd.read_csv(path, index_col=0)
        theta2 = np.array(ast.literal_eval(df.loc['theta2'].iloc[0]))
        peak_positions = np.array(ast.literal_eval(df.loc['peak_positions'].iloc[0]))
        q2_obs_expected = np.sort(peak_positions ** 2)[:20]

        # Pick peaks near the observed peak positions for 2theta conversion check
        q2_from_2theta = _convert_to_q2(theta2, '2theta', wavelength=wavelength)
        q2_from_2theta_sorted = np.sort(q2_from_2theta)
        q2_from_q = (np.array(ast.literal_eval(df.loc['q'].iloc[0])) ** 2)
        np.testing.assert_allclose(q2_from_2theta_sorted, np.sort(q2_from_q),
                                   rtol=1e-6, err_msg=f'2theta→q2 mismatch for {bl}')


def test_load_peaks_npy(test_metadata):
    from mlindex.command_line.run import _load_peaks
    import types
    for _, row in test_metadata.iterrows():
        peak_file = str(row['peak list file'])
        if not (peak_file.endswith('.pkslst') or peak_file.endswith('.pklst')):
            continue
        bl = str(row['bravais lattice'])
        case_name = peak_file.replace('.pkslst', '').replace('.pklst', '')
        npy_path = _TEST_DATA_DIR / case_name / f'{case_name}_peak_list.npy'
        args = types.SimpleNamespace(
            peaks=None,
            peak_file=str(npy_path),
            peak_units='q2',
            wavelength=None,
            zero_error=False,
            triplets_file=None,
        )
        result, _ = _load_peaks(args)
        expected = np.sort(np.load(npy_path))[:20]
        assert len(result) <= 20, f'{bl}: more than 20 peaks loaded'
        assert result[0] <= result[-1], f'{bl}: peaks not sorted'
        np.testing.assert_array_equal(result, expected,
                                      err_msg=f'load_peaks_npy mismatch for {bl}')
