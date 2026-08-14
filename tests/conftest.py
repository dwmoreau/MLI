import ast
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

BL_TO_LATTICE_SYSTEM = {
    "cF": "cubic",
    "cI": "cubic",
    "cP": "cubic",
    "hP": "hexagonal",
    "hR": "rhombohedral",
    "tI": "tetragonal",
    "tP": "tetragonal",
    "oC": "orthorhombic",
    "oF": "orthorhombic",
    "oI": "orthorhombic",
    "oP": "orthorhombic",
    "mC": "monoclinic",
    "mP": "monoclinic",
    "aP": "triclinic",
}

_TEST_DATA_DIR = Path(__file__).parent.parent / "mlindex" / "data" / "test_data"


def load_test_case(row):
    """Returns (q2_obs, unit_cell, wavelength, bravais_lattice, lattice_system).

    unit_cell is [a, b, c, alpha, beta, gamma] with angles in radians.
    q2_obs is sorted, length <= 20.
    """
    bl = str(row["bravais lattice"])
    wavelength = float(row["wavelength"])
    peak_file = str(row["peak list file"])

    # Unit cell from registry CSV (degrees → radians for angles)
    a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
    alpha = float(row["alpha"]) * np.pi / 180
    beta = float(row["beta"]) * np.pi / 180
    gamma = float(row["gamma"]) * np.pi / 180
    unit_cell = np.array([a, b, c, alpha, beta, gamma])

    if peak_file.endswith(".csv"):
        pattern_name = peak_file.replace(".csv", "")
        path = _TEST_DATA_DIR / pattern_name / peak_file
        df = pd.read_csv(path, index_col=0)
        peak_positions = np.array(ast.literal_eval(df.loc["peak_positions"].iloc[0]))
        q2_obs = np.sort(peak_positions**2)[:20]
    else:
        case_name = peak_file.replace(".pkslst", "").replace(".pklst", "")
        npy_path = _TEST_DATA_DIR / case_name / f"{case_name}_peak_list.npy"
        q2_obs = np.sort(np.load(npy_path))[:20]

    return q2_obs, unit_cell, wavelength, bl, BL_TO_LATTICE_SYSTEM[bl]


@pytest.fixture(scope="session")
def test_data_dir():
    return _TEST_DATA_DIR


@pytest.fixture(scope="session")
def unique_test_metadata(test_metadata):
    """One row per Bravais lattice — for tests that write/read expected files by BL name."""
    return test_metadata.drop_duplicates(subset=["bravais lattice"], keep="last")


@pytest.fixture(scope="session")
def test_metadata():
    return pd.read_csv(_TEST_DATA_DIR / "gsasII_tutorials.csv")


@pytest.fixture(scope="session")
def all_test_cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


@pytest.fixture(scope="session")
def aP_data(test_metadata):
    row = test_metadata[test_metadata["bravais lattice"] == "aP"].iloc[0]
    return load_test_case(row)


@pytest.fixture(scope="session")
def models_available():
    try:
        from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir

        _resolve_models_dir()
        return True
    except (FileNotFoundError, ImportError):
        return False


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow CLI integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip = pytest.mark.skip(reason="Pass --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
