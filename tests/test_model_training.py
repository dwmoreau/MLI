import numpy as np
import pytest
from pathlib import Path

from conftest import load_test_case, _TEST_DATA_DIR

EXPECTED_DIR = Path(__file__).parent / "expected"

N_GENERATE = 10


def _cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


def _canonical_order(unit_cells):
    """Sort generated unit cells into a stable order for comparison.

    The generators emit candidates in an order that depends on sorts keyed on
    quantities (reciprocal volume, q2) whose last ulp moves between machines, so
    two environments can produce the same candidates in a different order. The
    order carries no meaning downstream -- every candidate is refined and
    re-ranked afterwards -- so comparisons are made order insensitive.
    """
    return unit_cells[np.lexsort(unit_cells.T[::-1])]


# Tolerance for comparing generated candidates against the stored fixtures.
#
# Both generators run a quantized network through onnxruntime in float32, so the
# candidates they emit are not bit-reproducible across environments. Two distinct
# effects show up, and the comparison has to accommodate both without going so
# loose that a real change slips past:
#
#   * continuous drift. Every candidate moves a little -- measured 7e-7 for aP
#     and 5e-5 for mC against fixtures built from the same models. RELATIVE
#     covers this.
#   * discrete flips. The network scores a discretised set of cells and the
#     generator keeps the best ones; a candidate sitting on the selection
#     boundary can drop in or out for a last-ulp score difference, replacing one
#     entry outright. Measured on cP (one candidate of ten moved 9.98 A) and hR
#     (2.7e-2). No tolerance on a value can express that, so a small number of
#     candidates are allowed to differ entirely.
#
# A genuine regression in a generator moves most or all candidates, which this
# still catches: it requires all but MAX_UNMATCHED of them to match closely.
RELATIVE = 1e-4
MAX_UNMATCHED = 1


def _assert_candidates_match(result, expected, label):
    """Every candidate but at most MAX_UNMATCHED must appear in expected."""
    result, expected = _canonical_order(result), _canonical_order(expected)
    assert result.shape == expected.shape, (
        f"{label}: shape {result.shape} != fixture {expected.shape}"
    )
    unmatched = [
        index
        for index, candidate in enumerate(result)
        if not np.any(np.all(
            np.isclose(expected, candidate, rtol=RELATIVE, atol=RELATIVE), axis=1))
    ]
    assert len(unmatched) <= MAX_UNMATCHED, (
        f"{label}: {len(unmatched)} of {len(result)} candidates have no counterpart "
        f"in the fixture within rtol={RELATIVE}; at most {MAX_UNMATCHED} may differ. "
        f"Unmatched rows: {[result[i].tolist() for i in unmatched]}"
    )


@pytest.fixture(scope="session")
def all_optimizers(models_available, models_dir):
    if not models_available:
        pytest.skip("ML models not available")
    from mlindex.optimization.MPOptimizer import LocalComm
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    from mlindex.optimization.UtilitiesOptimizer import (
        get_cubic_optimizer,
        get_hexagonal_optimizer,
        get_rhombohedral_optimizer,
        get_tetragonal_optimizer,
        get_orthorhombic_optimizer,
        get_monoclinic_optimizer,
        get_triclinic_optimizer,
    )

    comm = LocalComm(1)
    bl_to_factory = {
        "cF": get_cubic_optimizer,
        "cI": get_cubic_optimizer,
        "cP": get_cubic_optimizer,
        "hP": get_hexagonal_optimizer,
        "hR": get_rhombohedral_optimizer,
        "tI": get_tetragonal_optimizer,
        "tP": get_tetragonal_optimizer,
        "oC": get_orthorhombic_optimizer,
        "oF": get_orthorhombic_optimizer,
        "oI": get_orthorhombic_optimizer,
        "oP": get_orthorhombic_optimizer,
        "mC": get_monoclinic_optimizer,
        "mP": get_monoclinic_optimizer,
        "aP": get_triclinic_optimizer,
    }
    optimizers = {}
    for bl, factory in bl_to_factory.items():
        opt = factory(
            bl, "1", 1, comm, optimizer_class=OptimizerManager, seed=12345,
            models_directory=models_dir,
        )
        opt.wrapper.setup_random()
        optimizers[bl] = opt
    return optimizers


def test_random_generator_generate(unique_test_metadata, all_optimizers):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        opt = all_optimizers[bl]
        rng = np.random.default_rng(12345)
        result = opt.wrapper.random_unit_cell_generator[bl].generate(
            N_GENERATE,
            rng,
            q2_obs,
            model="random",
        )
        expected = np.load(EXPECTED_DIR / f'random_gen_{bl}.npy')
        # allclose rather than array_equal: the unit cells come out of a chain of floating point
        # reductions whose accumulation order depends on the BLAS the machine happens to link
        # against, so the last couple of ulps move between environments while the code and the
        # model files are untouched. Observed drift is ~3e-14 on values of order 10, so a
        # tolerance well below anything crystallographically meaningful still catches a real
        # regression.
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9,
                                   err_msg=f'random_generator mismatch for {bl}')


def test_random_forest_generate(unique_test_metadata, all_optimizers):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        opt = all_optimizers[bl]
        sg = opt.wrapper.data_params["split_groups"][0]
        rng = np.random.default_rng(12345)
        result = opt.wrapper.random_forest_generator[sg].generate(
            N_GENERATE,
            rng,
            q2_obs,
        )
        expected = np.load(EXPECTED_DIR / f"random_forest_{bl}.npy")
        np.testing.assert_array_equal(
            result, expected, err_msg=f"random_forest mismatch for {bl}"
        )


def test_mi_templates_generate(unique_test_metadata, all_optimizers):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        opt = all_optimizers[bl]
        rng = np.random.default_rng(12345)
        result = opt.wrapper.miller_index_templator[bl].generate(
            N_GENERATE,
            rng,
            q2_obs,
        )
        expected = np.load(EXPECTED_DIR / f"mi_templates_{bl}.npy")
        _assert_candidates_match(result, expected, f"mi_templates {bl}")


def test_integral_filter_generate(unique_test_metadata, all_optimizers):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        opt = all_optimizers[bl]
        sg = opt.wrapper.data_params["split_groups"][0]
        rng = np.random.default_rng(12345)
        result = opt.wrapper.integral_filter_generator[sg].generate(
            N_GENERATE,
            rng,
            q2_obs,
            batch_size=2,
        )
        expected = np.load(EXPECTED_DIR / f"integral_filter_{bl}.npy")
        _assert_candidates_match(result, expected, f"integral_filter {bl}")


def test_candidate_matcher_rejects_a_real_regression():
    """Guard on the tolerance above: it must not let a genuine change through.

    _assert_candidates_match deliberately tolerates one replaced candidate,
    because a cell sitting on the network's selection boundary can drop in or
    out for a last-ulp score difference. That allowance is only defensible if
    the check still fails for anything larger, which is what this pins.
    """
    rng = np.random.default_rng(0)
    expected = rng.uniform(3.0, 12.0, size=(10, 3))

    _assert_candidates_match(expected.copy(), expected, "identical")

    one_replaced = expected.copy()
    one_replaced[4] = [99.0, 98.0, 97.0]
    _assert_candidates_match(one_replaced, expected, "one boundary flip")

    two_replaced = expected.copy()
    two_replaced[4] = [99.0, 98.0, 97.0]
    two_replaced[7] = [88.0, 87.0, 86.0]
    with pytest.raises(AssertionError):
        _assert_candidates_match(two_replaced, expected, "two flips")

    # A uniform drift far above float32 inference noise is a regression.
    with pytest.raises(AssertionError):
        _assert_candidates_match(expected * 1.01, expected, "1 percent drift")

    # So is a change in how many candidates come back.
    with pytest.raises(AssertionError):
        _assert_candidates_match(expected[:9], expected, "wrong count")
