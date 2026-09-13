"""The template regressor's input sets: their widths, the builder against independent calls, loading."""
import copy
import csv
import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import load_test_case
from mlindex.model_training import MITemplates as templates_module
from mlindex.model_training.MITemplates import (
    CONTEXT_MERITS,
    MERIT_NAMES,
    MITemplates,
    TEMPLATE_INPUT_SETS,
    n_template_features,
    regressor_inputs,
    template_features,
)
from mlindex.utilities.FigureOfMerits import (
    get_assignment_posterior,
    get_assignment_sigma,
    get_M20_likelihood,
    merit_set,
)
from mlindex.utilities.UnitCellTools import (
    get_reciprocal_unit_cell_from_xnn,
    get_unit_cell_volume,
)

# aP has the widest reference list and twenty calibration peaks; cF calibrates on ten.
LATTICES = ("aP", "cF")
# Enough cells to exercise every family without spending seconds per test.
N_CELLS = 500
ALL_FAMILIES = sorted({family for families in TEMPLATE_INPUT_SETS.values() for family in families})

# Written out rather than derived, so a change to a family's width has to be made here too.
WIDTHS_AT_20_PEAKS = {
    "rho": 22,
    "rho_sigma": 23,
    "posterior_sigma": 23,
    "merits": 31,
    "merits_structure": 35,
    "merits_structure_context": 39,
    "scalars_only": 19,
}


@pytest.fixture(scope="module")
def templators(models_available, models_dir, test_metadata):
    if not models_available:
        pytest.skip("ML models not available")
    from mlindex.optimization.MPOptimizer import LocalComm
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    from mlindex.optimization.UtilitiesOptimizer import (
        get_cubic_optimizer,
        get_triclinic_optimizer,
    )

    factories = {"aP": get_triclinic_optimizer, "cF": get_cubic_optimizer}
    loaded = {}
    for bl in LATTICES:
        optimizer = factories[bl](
            bl, "1", 1, LocalComm(1), optimizer_class=OptimizerManager, seed=12345,
            models_directory=models_dir,
        )
        row = test_metadata[test_metadata["bravais lattice"] == bl].iloc[-1]
        q2_obs = load_test_case(row)[0][: optimizer.n_peaks]
        templator = optimizer.wrapper.miller_index_templator[bl]
        xnn, q2_calc, q2_ref_calc = templator.generate_xnn(q2_obs, np.random.default_rng(3))
        cells = (xnn[:N_CELLS], q2_calc[:N_CELLS], q2_ref_calc[:N_CELLS])
        loaded[bl] = (templator, optimizer.wrapper.data_params, q2_obs, cells)
    return loaded


@pytest.mark.parametrize("bl", LATTICES)
@pytest.mark.parametrize("input_set", sorted(TEMPLATE_INPUT_SETS))
def test_each_input_set_builds_the_width_it_declares(templators, bl, input_set):
    templator, _, q2_obs, cells = templators[bl]
    n_calibration = templator.template_params["n_peaks_calibration"]
    per_peak_families = sum(
        family in ("rho", "posterior") for family in TEMPLATE_INPUT_SETS[input_set])
    expected = WIDTHS_AT_20_PEAKS[input_set] - per_peak_families * (20 - n_calibration)

    probe = copy.copy(templator)
    probe.template_params = dict(templator.template_params, template_inputs=input_set)
    inputs = probe._inputs(q2_obs, *cells)

    assert n_template_features(probe.template_params) == expected
    assert inputs.shape == (cells[0].shape[0], expected)
    assert inputs.dtype == np.float32


@pytest.mark.parametrize("bl", LATTICES)
def test_the_builder_matches_independent_calls(templators, bl):
    """Each family equals what its own function returns when called alone.

    The posterior is called here without the sigma and nearest-line distances the builder hands
    it, so this also shows that sharing one scan between sigma and the posterior changes nothing.
    """
    templator, _, q2_obs, (xnn, q2_calc, q2_ref_calc) = templators[bl]
    lattice_system = templator.lattice_system
    q2_calibration = q2_obs[: templator.template_params["n_peaks_calibration"]]
    features = template_features(
        ALL_FAMILIES, q2_calibration, xnn, q2_calc, q2_ref_calc, lattice_system, bl)

    np.testing.assert_array_equal(
        features["posterior"],
        get_assignment_posterior(q2_calibration, q2_ref_calc, lattice_system))
    np.testing.assert_array_equal(
        features["log_sigma"][:, 0],
        np.log(get_assignment_sigma(q2_calibration, q2_ref_calc, lattice_system)[0]))

    merits = merit_set(q2_calibration, q2_ref_calc)
    for column, name in enumerate(MERIT_NAMES):
        np.testing.assert_array_equal(features["merits"][:, column], merits[name])
    for column, (name, best) in enumerate(CONTEXT_MERITS):
        np.testing.assert_array_equal(
            features["context"][:, column], merits[name] - best(merits[name]))
    assert np.all(np.isfinite(features["structure"]))


@pytest.mark.parametrize("bl", LATTICES)
def test_the_rho_input_set_is_the_matrix_the_shipped_regressors_were_fitted_on(templators, bl):
    templator, _, q2_obs, (xnn, q2_calc, q2_ref_calc) = templators[bl]
    lattice_system = templator.lattice_system
    q2_calibration = q2_obs[: templator.template_params["n_peaks_calibration"]]

    reciprocal_volume = get_unit_cell_volume(
        get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True, lattice_system=lattice_system),
        partial_unit_cell=True, lattice_system=lattice_system)
    _, rho, _ = get_M20_likelihood(q2_calibration, q2_calc, bl, reciprocal_volume)
    q2_calc_max = q2_calc.max(axis=1)
    N_pred = np.count_nonzero(q2_ref_calc < q2_calc_max[:, np.newaxis], axis=1)
    shipped = np.concatenate(
        (rho, N_pred[:, np.newaxis], q2_calc_max[:, np.newaxis]), axis=1).astype(np.float32)

    features = template_features(
        TEMPLATE_INPUT_SETS["rho"], q2_calibration, xnn, q2_calc, q2_ref_calc, lattice_system, bl)
    np.testing.assert_array_equal(regressor_inputs(features, "rho"), shipped)


def test_every_regressor_input_matrix_comes_from_one_builder():
    assert "self._inputs(" in inspect.getsource(MITemplates.generate)
    assert "self._inputs(" in inspect.getsource(MITemplates._get_inputs_worker)
    assert "regressor_inputs(" in inspect.getsource(MITemplates._inputs)
    # The shipped statistic is computed in exactly one place, the builder.
    assert inspect.getsource(templates_module).count("get_M20_likelihood(") == 1


def _copy_shipped_aP_template(models_dir, destination):
    source = Path(models_dir) / "triclinic_1" / "template"
    for path in source.glob("aP_*_triclinic_1.*"):
        shutil.copy(path, destination / path.name)
    return destination / "aP_template_params_triclinic_1.csv"


def _rewrite_params(path, **columns):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = dict(rows[-1], **columns)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_a_model_saved_before_the_setting_loads_as_rho(templators, models_dir, tmp_path):
    templator, data_params, _, _ = templators["aP"]
    params_path = _copy_shipped_aP_template(models_dir, tmp_path)
    with open(params_path, "r", encoding="utf-8", newline="") as handle:
        assert "template_inputs" not in next(csv.reader(handle))

    loaded = MITemplates(
        "aP", data_params, {"tag": "triclinic_1", "template_inputs": "posterior_sigma"},
        templator.hkl_ref, str(tmp_path), 0)
    loaded.load_from_tag()
    assert loaded.template_params["template_inputs"] == "rho"


def test_an_unknown_input_set_is_refused_on_load(templators, models_dir, tmp_path):
    templator, data_params, _, _ = templators["aP"]
    _rewrite_params(_copy_shipped_aP_template(models_dir, tmp_path), template_inputs="nonsense")

    loaded = MITemplates("aP", data_params, {"tag": "triclinic_1"}, templator.hkl_ref,
                         str(tmp_path), 0)
    with pytest.raises(ValueError, match="nonsense"):
        loaded.load_from_tag()
