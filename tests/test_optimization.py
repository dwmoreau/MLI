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


# ---------------------------------------------------------------------------
# gauss_newton_step robustness
# ---------------------------------------------------------------------------
#
# Real runs put ~100,000 candidates through this, many of them ill-conditioned,
# and an escaping exception ends the whole run. Two failure modes used to be
# reachable: np.linalg.matrix_rank sat outside the try block and raises on a
# non-finite Hessian, and numpy's batched inv raises for the *entire* batch when
# any single member is singular -- which left every candidate with a zero step
# rather than just the bad one.


def _loss_for(n_entries, n_peaks=8, lattice_system="orthorhombic", seed=0):
    rng = np.random.default_rng(seed)
    q2_obs = np.sort(rng.uniform(0.05, 3.0, size=(n_entries, n_peaks)), axis=1)
    loss = CandidateOptLoss(q2_obs, lattice_system=lattice_system)
    hkl = rng.integers(0, 5, size=(n_entries, n_peaks, 3)).astype(float)
    xnn = rng.uniform(0.01, 0.5, size=(n_entries, loss.uc_length))
    loss.update(hkl, xnn)
    return loss, xnn


def test_gauss_newton_step_is_finite_on_healthy_input():
    loss, xnn = _loss_for(40)
    delta = loss.gauss_newton_step(xnn)
    assert delta.shape == (40, loss.uc_length)
    assert np.isfinite(delta).all()


@pytest.mark.parametrize(
    "description, corrupt",
    [
        ("NaN unit cell", lambda loss, xnn: xnn.__setitem__(3, np.nan)),
        ("Inf unit cell", lambda loss, xnn: xnn.__setitem__(3, np.inf)),
        ("zero sigma", lambda loss, xnn: loss.sigma.__setitem__((5, 2), 0.0)),
        ("NaN sigma", lambda loss, xnn: loss.sigma.__setitem__((5, 2), np.nan)),
        ("rank-deficient hkl", lambda loss, xnn: loss.hkl2.__setitem__((7, slice(None), slice(None)), 0.0)),
        ("all-zero sigma row", lambda loss, xnn: loss.sigma.__setitem__(9, 0.0)),
    ],
)
def test_gauss_newton_step_never_raises_on_degenerate_candidates(description, corrupt):
    """A degenerate candidate must not take the run down."""
    loss, xnn = _loss_for(40)
    corrupt(loss, xnn)
    # hessian_prefactor is derived from sigma, so rebuild it after corrupting.
    with np.errstate(all="ignore"):
        loss.hessian_prefactor = (1 / loss.sigma ** 2)[:, :, np.newaxis, np.newaxis]
        delta = loss.gauss_newton_step(xnn)
    assert delta.shape == (40, loss.uc_length), description
    assert np.isfinite(delta).all(), f"{description} produced a non-finite step"


def test_one_singular_candidate_does_not_zero_the_whole_batch():
    """The failure has to be isolated, not batch-wide.

    numpy's batched inv refuses the entire batch on a single singular member. If
    that is not caught per candidate, every candidate loses its refinement step
    and the optimizer silently stops making progress.
    """
    loss, xnn = _loss_for(40, seed=3)
    healthy = loss.gauss_newton_step(xnn)
    assert np.count_nonzero(healthy.any(axis=1)) > 30, "fixture is not healthy enough"

    # Make candidate 11 exactly rank-deficient.
    loss.hkl2[11, :, 1:] = 0.0
    with np.errstate(all="ignore"):
        delta = loss.gauss_newton_step(xnn)

    assert np.isfinite(delta).all()
    moved = np.count_nonzero(delta.any(axis=1))
    assert moved > 30, (
        f"only {moved} of 40 candidates got a step; one degenerate candidate "
        f"appears to have zeroed the whole batch"
    )
