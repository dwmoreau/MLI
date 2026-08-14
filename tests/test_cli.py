import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from conftest import load_test_case, _TEST_DATA_DIR

EXPECTED_DIR = Path(__file__).parent / "expected"


def _aP_q2(test_metadata):
    row = test_metadata[test_metadata["bravais lattice"] == "aP"].iloc[0]
    q2_obs, unit_cell, wavelength, bl, lattice_system = load_test_case(row)
    return q2_obs


def _compare_json(result_path, expected_path):
    result = pd.read_json(result_path)
    expected = pd.read_json(expected_path)
    assert list(result.columns) == list(expected.columns), "column mismatch"
    for col in result.select_dtypes(include="number").columns:
        np.testing.assert_allclose(
            result[col].values,
            expected[col].values,
            rtol=1e-6,
            err_msg=f"CLI output mismatch in column {col}",
        )
    for col in result.select_dtypes(exclude="number").columns:
        assert list(result[col]) == list(expected[col]), f"column {col} mismatch"


@pytest.mark.slow
def test_run_analytical_aP(test_metadata, tmp_path):
    q2 = _aP_q2(test_metadata)
    peak_file = tmp_path / "aP_q2.npy"
    np.save(peak_file, q2)
    output_file = tmp_path / "analytic_results.json"

    cmd = [
        sys.executable,
        "-m",
        "mlindex.command_line.run_analytical",
        "--peak-file",
        str(peak_file),
        "--peak-units",
        "q2",
        "--bravais-lattices",
        "aP",
        "--seed",
        "12345",
        "--output-file",
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI exited {result.returncode}:\n{result.stderr}"
    assert output_file.exists(), "output JSON not written"

    _compare_json(output_file, EXPECTED_DIR / "run_analytical_aP.json")


@pytest.mark.slow
def test_run_ml_aP(test_metadata, tmp_path, models_available):
    if not models_available:
        pytest.skip("ML models not available")

    q2 = _aP_q2(test_metadata)
    peak_file = tmp_path / "aP_q2.npy"
    np.save(peak_file, q2)

    cmd = [
        sys.executable,
        "-m",
        "mlindex.command_line.run",
        "--peak-file",
        str(peak_file),
        "--peak-units",
        "q2",
        "--bravais-lattices",
        "aP",
        "--nproc",
        "1",
        "--seed",
        "12345",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, f"CLI exited {result.returncode}:\n{result.stderr}"
    output_file = tmp_path / "indexing_results.json"
    assert output_file.exists(), "output JSON not written"

    _compare_json(output_file, EXPECTED_DIR / "run_ml_aP.json")
