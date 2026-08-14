import numpy as np
import pytest
from pathlib import Path

from conftest import load_test_case, _TEST_DATA_DIR

from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
from mlindex.utilities.UnitCellTools import (
    get_xnn_from_unit_cell,
    get_hkl_matrix,
    get_partial_unit_cell,
)


def _xnn(unit_cell, lattice_system):
    uc_p = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
    return get_xnn_from_unit_cell(
        uc_p[np.newaxis], partial_unit_cell=True, lattice_system=lattice_system
    )


def _cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


def test_candidate_opt_loss(test_metadata):
    rng = np.random.default_rng(42)
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn_true = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn_true, axis=1)

        mask = q2_calc > 0
        q2_pos = q2_calc[mask]
        hkl_pos = hkl_ref[mask]
        order = np.argsort(q2_pos)[:20]
        q2_exact = q2_pos[order]
        hkl_exact = hkl_pos[order]

        q2_obs_2d = q2_exact[np.newaxis]
        hkl_3d = hkl_exact[np.newaxis]

        eps = 1e-4
        xnn_pert = xnn_true + eps * rng.standard_normal(xnn_true.shape)

        loss = CandidateOptLoss(q2_obs_2d, lattice_system)
        loss.update(hkl_3d, xnn_pert)
        delta = loss.gauss_newton_step(xnn_pert)
        xnn_refined = xnn_pert + delta

        np.testing.assert_allclose(
            xnn_refined,
            xnn_true,
            rtol=1e-6,
            atol=1e-10,
            err_msg=f"GN step failed to recover xnn for {bl}",
        )
