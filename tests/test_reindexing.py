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
