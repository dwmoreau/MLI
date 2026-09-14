"""The template regressor's fit is reproducible from the templater's seed."""
import numpy as np
import pandas as pd

from mlindex.model_training.MITemplates import MITemplates


def test_the_regressor_fit_is_reproducible_from_the_templater_seed(tmp_path):
    """Two fits on identical training rows give identical rankers.

    Above 10 000 rows scikit-learn turns early stopping on and holds out a random tenth of the
    rows, so an unseeded regressor is a different model on every fit.
    """
    rng = np.random.default_rng(0)
    n_rows = 12000
    distance = rng.uniform(0, 0.05, n_rows)
    N_pred = rng.integers(20, 200, n_rows)
    q2_calc_max = rng.uniform(0.05, 0.5, n_rows)
    probability = rng.uniform(0, 1, (n_rows, 20))
    probability[:, 0] -= 10 * distance
    cache = np.concatenate(
        (distance[:, np.newaxis], N_pred[:, np.newaxis], q2_calc_max[:, np.newaxis], probability),
        axis=1)
    cache_directory = tmp_path / "data_cache"
    cache_directory.mkdir()
    for split in ("train", "val"):
        np.save(cache_directory / f"tP_{split}.npy", cache)
    grid = np.logspace(-7, np.log10(0.05), 400)
    curve = tmp_path / "curve.npy"
    np.save(curve, np.stack((grid, -np.log10(grid))))

    data_params = {"lattice_system": "tetragonal", "unit_cell_length": 2,
                   "unit_cell_indices": np.array([0, 2]), "hkl_ref_length": 10, "n_peaks": 20}
    data = pd.DataFrame({"augmented": [False, False], "train": [True, False]})
    inputs = np.concatenate(
        (probability, N_pred[:, np.newaxis], q2_calc_max[:, np.newaxis]), axis=1).astype(np.float32)
    predictions = []
    for _ in range(2):
        templator = MITemplates(
            "tP", data_params,
            {"tag": "seeded", "load_training_data": True, "roc_file_name": str(curve),
             "n_entries_train": 10},
            np.zeros((10, 3)), str(tmp_path), 7)
        templator.calibrate_templates(data)
        predictions.append(templator.hgbc_regressor.predict(inputs))
    np.testing.assert_array_equal(predictions[0], predictions[1])
