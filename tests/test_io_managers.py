"""Saving a model as ONNX: the exported graph must predict what the scikit-learn model predicts."""
import numpy as np
import pytest

from mlindex.utilities.IOManagers import SKLearnManager


def test_a_hist_gradient_boosting_regressor_round_trips_through_onnx(tmp_path):
    """The template regressor's model class, exported and reloaded the way MITemplates does it.

    A gradient-boosted tree stores a per-node missing-value flag that skl2onnx writes as a mix of
    integers and booleans, which protobuf 7 refuses; the export has to succeed regardless, and the
    graph it writes has to agree with the model it came from.
    """
    pytest.importorskip("skl2onnx")
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 23)).astype(np.float32)
    y = X[:, 0] - 0.5 * X[:, 22] + 0.1 * rng.normal(size=2000)
    model = HistGradientBoostingRegressor(max_iter=25, max_leaf_nodes=31).fit(X, y)

    filename = str(tmp_path / "regressor")
    SKLearnManager(filename=filename, model_type="onnx").save(model=model, n_features=X.shape[1])
    loaded = SKLearnManager(filename=filename, model_type="onnx")
    loaded.load()

    assert loaded.model.get_inputs()[0].shape[1] == X.shape[1]
    np.testing.assert_allclose(loaded.predict(X)[:, 0], model.predict(X), rtol=1e-5, atol=1e-5)
