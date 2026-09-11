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


def _repairable(scores=None, bad=(2,), n=5):
    """A Candidates carrying only what fix_bad_conversions touches."""
    from mlindex.optimization.Candidates import Candidates

    candidates = Candidates.__new__(Candidates)
    candidates.rng = np.random.default_rng(0)
    cells = np.arange(1.0, n + 1.0).reshape(n, 1)
    candidates.reciprocal_unit_cell = cells.copy()
    for row in bad:
        candidates.reciprocal_unit_cell[row] = np.nan
    candidates.xnn = (10.0*cells).copy()
    candidates.unit_cell = candidates.xnn.copy()
    if scores is not None:
        candidates.best_M20 = np.asarray(scores, dtype=float)
    return candidates


def test_a_cell_that_will_not_convert_is_replaced_by_one_that_does():
    """A NaN cell cannot be refined, scored or ranked, so carrying it costs a candidate slot for
    the rest of the search."""
    candidates = _repairable(scores=[1.0, 9.0, 0.0, 5.0, 3.0])

    candidates.fix_bad_conversions()

    assert not np.isnan(candidates.reciprocal_unit_cell).any()
    assert candidates.xnn[2, 0] in {10.0, 20.0, 40.0, 50.0}
    # The donor is copied consistently into all three representations.
    donor = np.flatnonzero(candidates.xnn[:, 0] == candidates.xnn[2, 0])[0]
    assert candidates.unit_cell[2, 0] == candidates.unit_cell[donor, 0]


def test_donors_are_preferred_by_rank_and_not_by_the_size_of_the_score():
    """M20 is heavy-tailed -- it reaches 1e12 on a saturated fit -- so weighting by the value
    would put nearly all the probability on one blown-up candidate and every repair would return
    the same cell. Ranks are scale-free: the best donor is n times as likely as the worst, whatever
    the numbers are."""
    from mlindex.optimization.Candidates import Candidates

    counts = {}
    for seed in range(2000):
        candidates = _repairable(scores=[1.0, 9.0, 0.0, 5.0, 3.0])
        candidates.rng = np.random.default_rng(seed)
        candidates.fix_bad_conversions()
        cell = float(candidates.xnn[2, 0])
        counts[cell] = counts.get(cell, 0) + 1

    # Donors ranked worst to best are 10, 50, 40, 20, so frequencies must be ordered likewise.
    assert counts[20.0] > counts[40.0] > counts[50.0] > counts[10.0]
    # A blown-up score must not swamp the draw the way a value-weighted rule would.
    blown = {}
    for seed in range(2000):
        candidates = _repairable(scores=[1.0, 1e13, 0.0, 5.0, 3.0])
        candidates.rng = np.random.default_rng(seed)
        candidates.fix_bad_conversions()
        cell = float(candidates.xnn[2, 0])
        blown[cell] = blown.get(cell, 0) + 1
    assert blown[20.0] < 0.5*sum(blown.values())


def test_the_draw_is_uniform_before_any_score_exists():
    """The repair runs from `update_unit_cell_from_xnn`, which is reached before `assign_hkls` has
    produced an M20. There is nothing to rank by there, and that must not raise."""
    candidates = _repairable(scores=None)

    candidates.fix_bad_conversions()

    assert not np.isnan(candidates.reciprocal_unit_cell).any()


def test_more_bad_than_good_still_repairs_every_one():
    """With too few donors to go round they are reused, rather than leaving NaN cells behind."""
    candidates = _repairable(scores=[1.0, 2.0, 0.0, 0.0, 0.0], bad=(2, 3, 4))

    candidates.fix_bad_conversions()

    assert not np.isnan(candidates.reciprocal_unit_cell).any()
    assert set(candidates.xnn[:, 0]) <= {10.0, 20.0}


def test_every_cell_failing_to_convert_is_refused_rather_than_left_as_nan():
    """There is nothing to repair from. Carrying on would hand NaN cells to the refinement, where
    they are silently dropped much later by the deduplication's own filter."""
    candidates = _repairable(scores=[0.0, 0.0], bad=(0, 1), n=2)

    with pytest.raises(ValueError, match='failed to convert'):
        candidates.fix_bad_conversions()


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
    n_indexed = [np.array([5, 6, 7, 8])]
    spacegroup = ['A', 'B', 'BAD', 'D']

    manager._downsample_computation(M20, xnn, n_indexed, spacegroup,
                                    n_top_candidates=10)

    assert 'BAD' not in manager.top_spacegroup
    assert manager.top_M20.tolist() == [30.0, 20.0, 10.0]
    assert manager.top_spacegroup == ['D', 'B', 'A']


def test_a_short_return_does_not_desynchronise_the_pool():
    """The MPI manager used to size its receive buffers from the number of candidates it
    *sent* a rank. The rank prunes before returning, so the four numeric arrays arrived
    zero-padded up to the outgoing count while `best_spacegroup` -- sent by pickle, which
    carries its own length -- arrived at the true, shorter one. The two then disagreed:
    the `zip` against the NaN filter silently truncated to the list, and the
    reciprocal-volume sort indexed the full-length arrays, so the last candidates ran off
    the end of the list.

    Reproduced end to end before the fix: `mpiexec -n 6 ... --mpi --bravais-lattices aP`
    raised `IndexError: list index out of range` on the manager rank, which never reached
    the global barrier, so the other five ranks blocked there forever and the whole job
    hung with every rank spinning at 100% CPU. This pins the arithmetic underneath it."""
    manager = _downsample_manager()
    # Four rows of numbers against three spacegroups: what a padded receive looked like.
    xnn = [np.array([[1.0], [2.0], [3.0], [0.0]])]
    M20 = [np.array([10.0, 20.0, 30.0, 0.0])]
    n_indexed = [np.array([5, 6, 7, 0])]
    spacegroup = ['A', 'B', 'C']

    with pytest.raises(ValueError, match='spacegroup'):
        manager._downsample_computation(M20, xnn, n_indexed, spacegroup,
                                        n_top_candidates=10)


def test_a_matched_return_still_downsamples():
    """The guard above must not fire on the ordinary case."""
    manager = _downsample_manager()
    xnn = [np.array([[1.0], [2.0], [3.0]])]
    M20 = [np.array([10.0, 20.0, 30.0])]
    n_indexed = [np.array([5, 6, 7])]
    spacegroup = ['A', 'B', 'C']

    manager._downsample_computation(M20, xnn, n_indexed, spacegroup,
                                    n_top_candidates=10)

    assert manager.top_M20.tolist() == [30.0, 20.0, 10.0]
    assert manager.top_spacegroup == ['C', 'B', 'A']


def test_a_collapsed_neighbourhood_keeps_each_survivor_s_spacegroup():
    """The deduplication permutes rows and drops them, and the spacegroups are a list beside
    the arrays rather than a column of them. Row identity is carried through the collapse so
    the list is indexed once, at the end; this pins that a survivor keeps its own label."""
    manager = _downsample_manager()
    manager.opt_params['downsample_radius'] = 1e-3
    # The first two cells are near-duplicates, so one of them is collapsed away.
    xnn = [np.array([[1.0], [1.0000001], [5.0]])]
    M20 = [np.array([10.0, 20.0, 30.0])]
    n_indexed = [np.array([5, 6, 7])]
    spacegroup = ['A', 'B', 'C']

    manager._downsample_computation(M20, xnn, n_indexed, spacegroup,
                                    n_top_candidates=10)

    assert manager.top_M20.tolist() == [30.0, 20.0]
    assert manager.top_spacegroup == ['C', 'B']
    assert manager.top_n_indexed.tolist() == [7, 6]


def test_the_downsample_hook_is_inert_and_sees_the_pool_before_truncation():
    """The shipped indexer keeps twenty candidates a lattice; a benchmark needs every
    survivor and the size of the pool they came from. The hook is where a research subclass
    reads them, and it must do nothing at all unless something overrides it."""
    from mlindex.optimization.MPIOptimizer import OptimizerManager

    seen = {}

    class Recording(OptimizerManager):
        def _on_downsample(self, survivors, order, n_entering, n_top_candidates):
            seen.update(survivors=survivors, order=order, n_entering=n_entering)

    manager = Recording.__new__(Recording)
    manager.lattice_system = 'cubic'
    manager.n_ranks = 1
    manager.zero_error = False
    manager.opt_params = {'downsample_radius': 1e-9}
    xnn = [np.array([[1.0], [2.0], [np.nan], [3.0]])]
    M20 = [np.array([10.0, 20.0, 999.0, 30.0])]
    n_indexed = [np.array([5, 6, 7, 8])]
    spacegroup = ['A', 'B', 'BAD', 'D']

    manager._downsample_computation(M20, xnn, n_indexed, spacegroup,
                                    n_top_candidates=1)

    # Truncation keeps one; the hook saw all three that reached deduplication.
    assert manager.top_M20.tolist() == [30.0]
    assert seen['n_entering'] == 3
    assert seen['survivors']['spacegroup'] == ['A', 'B', 'D']
    assert seen['survivors']['M20'].tolist() == [10.0, 20.0, 30.0]
    assert seen['order'].tolist() == [2, 1, 0]
    # The base class reads nothing and returns nothing.
    assert OptimizerManager._on_downsample(manager, {}, None, 0, 0) is None


def _triclinic_candidates(xnn_values, seed=7):
    """A triclinic Candidates over a synthetic peak list, built straight from given cells.

    Triclinic because it is the only lattice system whose repair actually fires here:
    `fix_unphysical` dispatches the box systems -- cubic, orthorhombic, tetragonal, hexagonal --
    to `fix_unphysical_box`, which leaves an out-of-range cell alone. Writing this test on cubic
    would pass whether or not the bug is present.
    """
    import numpy as np
    from mlindex.optimization.Candidates import Candidates
    from mlindex.utilities.Q2Calculator import Q2Calculator

    hkl_ref = np.load(_TEST_DATA_DIR.parent / "hkl_ref_aP.npy")
    xnn_true = np.array([[0.02, 0.015, 0.01, 0.001, 0.002, 0.0015]])
    q2_ref = Q2Calculator(
        lattice_system="triclinic", hkl=hkl_ref, tensorflow=False, representation="xnn"
    ).get_q2(xnn_true)[0]
    q2_obs = np.sort(q2_ref[q2_ref > 0])[:20]

    opt_params = {
        "minimum_uc": 2.0,
        "maximum_uc": 60.0,
        "assignment_threshold": 0.95,
        "figure_of_merit": "M20",
    }
    return q2_obs, Candidates(
        q2_obs=q2_obs,
        xnn=np.asarray(xnn_values, dtype=float),
        hkl_ref=hkl_ref,
        lattice_system="triclinic",
        bravais_lattice="aP",
        opt_params=opt_params,
        rng=np.random.default_rng(seed),
        fom=None,
        zero_error=False,
        wavelength=None,
    )


def test_best_cell_and_best_score_describe_the_same_candidate_after_repair():
    """`best_M20` must be the score OF `best_xnn`, including for cells the constructor repaired.

    `fix_out_of_range_candidates` replaces any cell it considers unphysical. If `best_xnn` were
    copied before that ran while `best_M20` were computed after it, the two would describe
    different candidates -- silently, and only for the repaired ones, which are also the ones most
    likely to go unimproved by the search and so to carry the mismatched pair all the way to the
    prune, where it decides survival.

    Recomputing the score from the stored cell through the optimizer's own route is the check:
    same q2_calculator, same fast_assign, same get_M20, so anything but exact equality means the
    cell and its score have come apart.
    """
    import numpy as np
    from mlindex.utilities.FigureOfMerits import get_M20
    from mlindex.utilities.numba_functions import fast_assign

    rng = np.random.default_rng(11)
    xnn = np.concatenate([
        np.array([[0.02, 0.015, 0.01, 0.001, 0.002, 0.0015]]),
        rng.normal(0.01, 0.02, size=(32, 6)),        # many of these are unphysical
    ])
    q2_obs, candidates = _triclinic_candidates(xnn)

    q2_ref_calc = candidates.q2_calculator.get_q2(candidates.best_xnn)
    hkl_assign = fast_assign(q2_obs, q2_ref_calc)
    q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
    recomputed = get_M20(q2_obs, q2_calc, q2_ref_calc)

    differing = int(np.sum(~((np.isnan(recomputed) & np.isnan(candidates.best_M20))
                             | (recomputed == candidates.best_M20))))
    assert differing == 0, f"{differing} candidates whose stored cell does not give their score"


def _allocation_lattices():
    from mlindex.command_line.run import BRAVAIS_LATTICES

    return list(BRAVAIS_LATTICES)


def test_every_requested_lattice_is_allocated_exactly_once():
    """A lattice dropped from the plan is a lattice that never gets indexed.

    _run_mp reads its results dict by lattice name, so a missing one is a KeyError
    after all the work, and a duplicated one is the same lattice searched twice.
    """
    from mlindex.command_line.run import allocate_lattice_groups

    all_bl = _allocation_lattices()
    for n_procs in range(1, 21):
        for requested in (all_bl, all_bl[:1], all_bl[:5]):
            plan = allocate_lattice_groups(requested, n_procs)
            allocated = [bl for bl_list, _ in plan for bl in bl_list]
            assert sorted(allocated) == sorted(requested), (
                f"n_procs={n_procs}, requested={requested}: got {allocated}")


def test_the_plan_never_asks_for_more_processes_than_it_was_given():
    from mlindex.command_line.run import allocate_lattice_groups

    all_bl = _allocation_lattices()
    for n_procs in range(1, 21):
        plan = allocate_lattice_groups(all_bl, n_procs)
        used = sum(group_size for _, group_size in plan)
        assert used <= n_procs, f"n_procs={n_procs}: plan uses {used}"
        assert all(group_size >= 1 for _, group_size in plan)


def test_more_processes_never_make_the_plan_slower():
    """The regression that motivated dropping the no-split rule.

    Refusing to split a lattice until every lattice had its own process left the
    predicted makespan flat at 9.87 s from eight processes through fourteen -- six
    processes that bought nothing, because nothing can beat mP running alone -- and
    then halved it at fifteen. The planner must be free to split, so that each added
    process is worth at least as much as the last.
    """
    from mlindex.command_line.run import allocate_lattice_groups, _group_cost

    all_bl = _allocation_lattices()
    previous = None
    for n_procs in range(1, 25):
        plan = allocate_lattice_groups(all_bl, n_procs)
        makespan = max(_group_cost(bl_list, size) for bl_list, size in plan)
        if previous is not None:
            assert makespan <= previous + 1e-9, (
                f"n_procs={n_procs} is slower than {n_procs - 1}: "
                f"{makespan:.2f} vs {previous:.2f}")
        previous = makespan


def test_cheap_lattices_share_a_process_and_expensive_ones_are_split():
    """cI and cP cost 0.01 s; mP costs 9.87 s and is the makespan on its own.

    Giving each of the fourteen its own process wastes most of them. At fourteen the
    planner should be pooling the cheap lattices and spending what it saves on mP.
    """
    from mlindex.command_line.run import allocate_lattice_groups

    plan = allocate_lattice_groups(_allocation_lattices(), 14)
    sizes = {bl: size for bl_list, size in plan for bl in bl_list}
    shared = {bl for bl_list, _ in plan for bl in bl_list if len(bl_list) > 1}

    assert sizes['mP'] > 1, f"the heaviest lattice was not split: {plan}"
    assert {'cI', 'cP'} <= shared, (
        f"the two cheapest lattices each took a whole process: {plan}")


def test_the_heaviest_group_comes_first_because_the_caller_runs_it():
    """setup_lattice_groups runs group 0 in the calling process and spawns the rest.

    If group 0 were not the heaviest, the caller would finish early and sit idle
    while a spawned group was still going.
    """
    from mlindex.command_line.run import allocate_lattice_groups, _group_cost

    all_bl = _allocation_lattices()
    for n_procs in (2, 4, 8, 14, 18):
        plan = allocate_lattice_groups(all_bl, n_procs)
        costs = [_group_cost(bl_list, group_size) for bl_list, group_size in plan]
        assert costs == sorted(costs, reverse=True), f"n_procs={n_procs}: {costs}"


def test_every_lattice_has_a_cost_entry():
    """allocate_lattice_groups indexes _BL_COST directly, so a missing key is a
    KeyError at startup rather than a bad plan."""
    from mlindex.command_line.run import BRAVAIS_LATTICES, _BL_COST

    assert sorted(_BL_COST) == sorted(BRAVAIS_LATTICES)
    for bl, cost in _BL_COST.items():
        assert len(cost) == 2 and all(c >= 0 for c in cost), f"{bl}: {cost}"


def test_the_cost_table_measurement_script_still_matches_the_optimizer():
    """`measure_bl_cost` times two methods by name; a rename must fail loudly here.

    The script wraps `_generate_candidates_xnn` and `_run_loop` on an optimizer
    instance to separate the cost that divides with group size from the cost that
    does not. Both are private, so nothing else would notice them being renamed --
    and the script would then either crash mid-measurement or, worse, silently
    report zero for a phase that had moved.
    """
    from mlindex.optimization.MPIOptimizer import OptimizerBase, OptimizerManager

    assert hasattr(OptimizerManager, '_generate_candidates_xnn'), (
        "measure_bl_cost times OptimizerManager._generate_candidates_xnn by name")
    assert hasattr(OptimizerBase, '_run_loop'), (
        "measure_bl_cost times OptimizerBase._run_loop by name")


def test_the_cost_table_measurement_script_names_real_patterns():
    """Its default patterns must exist, or the script fails only once run."""
    from mlindex.scripts.measure_bl_cost import DEFAULT_PATTERNS, _test_data_dir

    for name in DEFAULT_PATTERNS:
        path = _test_data_dir().joinpath(name, f"{name}_peak_list.npy")
        assert path.is_file(), f"default pattern {name} is missing at {path}"
