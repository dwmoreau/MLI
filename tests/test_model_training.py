import numpy as np
import pytest
from pathlib import Path

from conftest import load_test_case, _TEST_DATA_DIR

EXPECTED_DIR = Path(__file__).parent / "expected"

N_GENERATE = 10


def _cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


@pytest.fixture(scope="session")
def all_optimizers(models_available):
    if not models_available:
        pytest.skip("ML models not available")
    from mlindex.optimization.MPOptimizer import LocalComm
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    from mlindex.optimization.UtilitiesOptimizer import (
        _resolve_project_path,
        get_cubic_optimizer,
        get_hexagonal_optimizer,
        get_rhombohedral_optimizer,
        get_tetragonal_optimizer,
        get_orthorhombic_optimizer,
        get_monoclinic_optimizer,
        get_triclinic_optimizer,
    )

    comm = LocalComm(1)
    project_path = _resolve_project_path()
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
            bl, "1", 1, comm, project_path, optimizer_class=OptimizerManager, seed=12345
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
        expected = np.load(EXPECTED_DIR / f"random_gen_{bl}.npy")
        np.testing.assert_array_equal(
            result, expected, err_msg=f"random_generator mismatch for {bl}"
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
        np.testing.assert_array_equal(
            result, expected, err_msg=f"mi_templates mismatch for {bl}"
        )


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
        np.testing.assert_array_equal(
            result, expected, err_msg=f"integral_filter mismatch for {bl}"
        )
