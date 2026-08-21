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


# ---------------------------------------------------------------------------
# gauss_newton_step_zero_error robustness
# ---------------------------------------------------------------------------
#
# This path used to wrap its invertibility test in a bare `except:` that printed
# the offending candidate and then hit `assert False`, so one non-finite Hessian
# ended the run by construction. It is reachable: when wavelength/2 * sqrt(q2)
# exceeds 1 the arcsin is NaN and the whole Hessian follows.


def test_zero_error_step_shape_and_finiteness():
    loss, xnn = _loss_for(30)
    delta = loss.gauss_newton_step_zero_error(xnn, wavelength=1.5405)
    # one extra column for the zero-point parameter
    assert delta.shape == (30, loss.uc_length + 1)
    assert np.isfinite(delta).all()


def test_zero_error_step_accepts_a_starting_zeropoint():
    loss, xnn = _loss_for(30)
    zeropoint = np.full(30, 1e-4)
    delta = loss.gauss_newton_step_zero_error(xnn, wavelength=1.5405, zeropoint=zeropoint)
    assert delta.shape == (30, loss.uc_length + 1)
    assert np.isfinite(delta).all()


@pytest.mark.parametrize(
    "description, corrupt",
    [
        ("NaN unit cell", lambda loss, xnn: xnn.__setitem__(3, np.nan)),
        ("Inf unit cell", lambda loss, xnn: xnn.__setitem__(3, np.inf)),
        ("zero sigma", lambda loss, xnn: loss.sigma.__setitem__((5, 2), 0.0)),
        ("rank-deficient hkl", lambda loss, xnn: loss.hkl2.__setitem__((7, slice(None), slice(None)), 0.0)),
    ],
)
def test_zero_error_step_never_raises_on_degenerate_candidates(description, corrupt):
    loss, xnn = _loss_for(30)
    corrupt(loss, xnn)
    with np.errstate(all="ignore"):
        delta = loss.gauss_newton_step_zero_error(xnn, wavelength=1.5405)
    assert delta.shape == (30, loss.uc_length + 1), description
    assert np.isfinite(delta).all(), f"{description} produced a non-finite step"


def test_zero_error_step_survives_an_unreachable_wavelength():
    """wavelength/2 * sqrt(q2) > 1 makes arcsin NaN for the affected peaks.

    That used to poison the Hessian and trip the `assert False`. It must now
    simply skip the candidates it cannot refine.
    """
    loss, xnn = _loss_for(30)
    with np.errstate(all="ignore"):
        delta = loss.gauss_newton_step_zero_error(xnn, wavelength=50.0)
    assert delta.shape == (30, loss.uc_length + 1)
    assert np.isfinite(delta).all()
    # Nothing is refinable here, so every step should be zero rather than junk.
    assert not delta.any()


def _standardization_fixture(lattice_system, unit_cells):
    """A Candidates carrying only what standardize_cell touches.

    The full constructor wants an hkl_ref, a Q2Calculator and a peak list, none of which the
    standardization reads. Building one would test the constructor, not the method.
    """
    from mlindex.optimization.Candidates import Candidates
    from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell

    candidates = object.__new__(Candidates)
    candidates.lattice_system = lattice_system
    candidates.rng = np.random.default_rng(0)
    candidates.minimum_unit_cell = 2.0
    candidates.maximum_unit_cell = 100.0
    # Angles are radians throughout this codebase -- get_unit_cell_volume takes np.cos of them
    # directly -- so the degrees the cases are written in are converted here.
    partial = np.stack([
        get_partial_unit_cell(
            np.concatenate([np.asarray(unit_cell[:3], dtype=float),
                            np.deg2rad(np.asarray(unit_cell[3:], dtype=float))]),
            lattice_system=lattice_system,
            )
        for unit_cell in unit_cells
        ])
    candidates.best_xnn = get_xnn_from_unit_cell(
        partial, partial_unit_cell=True, lattice_system=lattice_system
        )
    return candidates


@pytest.mark.parametrize(
    "lattice_system, unit_cells",
    [
        # A long, very oblique c axis, which the Selling reduction inside
        # monoclinic_standardization shortens: c 20 -> 10.09 and beta 150 -> 97.5 deg.
        ("monoclinic", [[8.0, 5.0, 20.0, 90.0, 150.0, 90.0]]),
        # An unreduced triclinic cell, so the Selling reduction has something to do.
        ("triclinic", [[9.0, 8.0, 7.0, 95.0, 100.0, 115.0],
                       [11.0, 6.0, 10.0, 85.0, 78.0, 98.0]]),
        ],
    )
def test_standardize_cell_writes_back_when_nothing_fails(lattice_system, unit_cells):
    """The write-back used to sit inside `if np.sum(failed) > 0`.

    So on any run where no candidate NaN'd out of the final xnn conversion -- the common case --
    the standardization was computed and then discarded, and best_xnn kept its unstandardized
    value. That silently disabled monoclinic standardization and the triclinic Selling reduction
    for {mP, mC, aP}, which is the hard stratum, on essentially every production run.
    """
    candidates = _standardization_fixture(lattice_system, unit_cells)
    before = candidates.best_xnn.copy()

    candidates.standardize_cell()

    assert np.isfinite(candidates.best_xnn).all()
    assert not np.allclose(candidates.best_xnn, before), (
        'standardize_cell left best_xnn untouched; the write-back is conditional again'
        )


def test_standardize_cell_is_a_no_op_off_monoclinic_and_triclinic():
    candidates = _standardization_fixture(
        'orthorhombic', [[8.0, 9.0, 10.0, 90.0, 90.0, 90.0]])
    before = candidates.best_xnn.copy()
    candidates.standardize_cell()
    assert np.array_equal(candidates.best_xnn, before)


def _downsample_manager():
    """An OptimizerManager stub carrying only what _downsample_computation touches."""
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    manager = OptimizerManager.__new__(OptimizerManager)
    manager.lattice_system = 'cubic'
    manager.n_ranks = 1
    manager.zero_error = False
    manager.opt_params = {'downsample_radius': 1e-9, 'dump_candidates': None}
    return manager


def test_a_dropped_nan_cell_does_not_shift_the_spacegroups():
    """The NaN filter sliced four arrays and not the spacegroup list beside them, while
    the reciprocal-volume sort indexes the *filtered* arrays -- so one dropped row moved
    every later spacegroup onto a different candidate. It reaches the reported answer:
    in this construction the bad label lands on the highest-M20 candidate."""
    manager = _downsample_manager()
    xnn = [np.array([[1.0], [2.0], [np.nan], [3.0]])]
    M20 = [np.array([10.0, 20.0, 999.0, 30.0])]
    Minfo = [np.array([1.0, 2.0, 3.0, 4.0])]
    n_indexed = [np.array([5, 6, 7, 8])]
    spacegroup = ['A', 'B', 'BAD', 'D']

    manager._downsample_computation(M20, Minfo, xnn, n_indexed, spacegroup,
                                    n_top_candidates=10)

    assert 'BAD' not in manager.top_spacegroup
    assert manager.top_M20.tolist() == [30.0, 20.0, 10.0]
    assert manager.top_spacegroup == ['D', 'B', 'A']
