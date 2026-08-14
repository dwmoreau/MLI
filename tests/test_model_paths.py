"""Model directory resolution.

These tests are fully offline and need neither the downloaded models nor a network
connection: they build a small fake models tree in tmp_path.

The fixture deliberately uses a directory that contains a space and does *not* end in
'mlindex/models'. That is the case the old `_resolve_project_path()` got silently wrong:
it returned `models_dir.parent.parent` and Wrapper re-appended 'mlindex/models', so any
other path shape resolved somewhere else entirely.
"""

import os
from pathlib import Path

import pytest

from mlindex import paths

LATTICE_SYSTEMS = (
    "cubic", "hexagonal", "monoclinic", "orthorhombic",
    "rhombohedral", "tetragonal", "triclinic",
)
MODEL_SUBDIRS = ("data", "integral_filter", "random_forest", "template", "random", "augmentor")


def _build_models_tree(root):
    for lattice_system in LATTICE_SYSTEMS:
        for subdir in MODEL_SUBDIRS:
            (root / f"{lattice_system}_1" / subdir).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def fake_models_dir(tmp_path):
    return _build_models_tree(tmp_path / "some" / "custom place")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Neither the real env var nor the real XDG dir should leak into these tests."""
    monkeypatch.delenv("MLINDEX_MODELS_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


def _resolve():
    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir

    return _resolve_models_dir()


# --- paths helpers ---------------------------------------------------------------


def test_default_models_dir_uses_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.default_models_dir() == tmp_path / "mlindex" / "models"


def test_default_models_dir_falls_back_to_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert paths.default_models_dir() == tmp_path / ".local" / "share" / "mlindex" / "models"


def test_env_value_is_stripped_of_quotes_and_space(monkeypatch):
    monkeypatch.setenv("MLINDEX_MODELS_DIR", '  "/tmp/some place"  ')
    assert paths.models_dir_from_env() == Path("/tmp/some place")


def test_env_unset_or_blank_is_none(monkeypatch):
    assert paths.models_dir_from_env() is None
    monkeypatch.setenv("MLINDEX_MODELS_DIR", "   ")
    assert paths.models_dir_from_env() is None


def test_looks_like_models_dir(fake_models_dir, tmp_path):
    assert paths.looks_like_models_dir(fake_models_dir)
    # A partial tree like the one bundled in the wheel must not count as a models dir.
    partial = tmp_path / "partial"
    (partial / "cubic_1" / "data").mkdir(parents=True)
    assert not paths.looks_like_models_dir(partial)


# --- resolution ------------------------------------------------------------------


def test_env_var_returns_the_directory_itself(monkeypatch, fake_models_dir):
    """The regression test for the MLINDEX_MODELS_DIR bug."""
    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(fake_models_dir))
    assert _resolve() == fake_models_dir


def test_env_var_missing_path_raises(monkeypatch, tmp_path):
    missing = tmp_path / "not_here"
    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(missing))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve()


def test_env_var_pointing_at_parent_suggests_the_child(monkeypatch, fake_models_dir):
    """Pointing one level too high is the most common user mistake."""
    parent = fake_models_dir.parent
    (parent / "models").symlink_to(fake_models_dir, target_is_directory=True)
    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(parent))
    with pytest.raises(FileNotFoundError) as excinfo:
        _resolve()
    assert "Did you mean" in str(excinfo.value)
    assert str(parent / "models") in str(excinfo.value)


def test_env_var_wrong_dir_without_candidate_points_at_downloader(monkeypatch, tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(unrelated))
    with pytest.raises(FileNotFoundError, match="mlindex.download_models"):
        _resolve()


def test_xdg_branch(monkeypatch, tmp_path):
    _build_models_tree(tmp_path / "mlindex" / "models")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert _resolve() == tmp_path / "mlindex" / "models"


def test_env_var_beats_xdg(monkeypatch, tmp_path, fake_models_dir):
    _build_models_tree(tmp_path / "mlindex" / "models")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(fake_models_dir))
    assert _resolve() == fake_models_dir


def test_resolve_project_path_is_deprecated(monkeypatch, fake_models_dir):
    from mlindex.optimization.UtilitiesOptimizer import _resolve_project_path

    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(fake_models_dir))
    with pytest.deprecated_call():
        _resolve_project_path()


# --- Wrapper round-trip ----------------------------------------------------------


@pytest.fixture
def quiet_wrapper(monkeypatch):
    """Wrapper that skips loading real model files."""
    from mlindex.model_training.Wrapper import Wrapper

    monkeypatch.setattr(Wrapper, "setup_from_tag", lambda self, load_bravais_lattice="all": None)
    monkeypatch.setattr(Wrapper, "setup", lambda self: None)
    return Wrapper


def test_wrapper_results_dir_uses_models_directory(quiet_wrapper, fake_models_dir):
    wrapper = quiet_wrapper(data_params={
        "tag": "cubic_1",
        "base_directory": None,
        "models_directory": str(fake_models_dir),
        "load_from_tag": True,
    })
    assert Path(wrapper.save_to["results"]) == fake_models_dir / "cubic_1"
    assert Path(wrapper.save_to["data"]) == fake_models_dir / "cubic_1" / "data"


def test_models_directory_is_not_persisted(quiet_wrapper, fake_models_dir):
    """Wrapper.save() writes data_params to CSV; a machine-specific path must not ride along."""
    wrapper = quiet_wrapper(data_params={
        "tag": "cubic_1",
        "base_directory": None,
        "models_directory": str(fake_models_dir),
        "load_from_tag": True,
    })
    assert "models_directory" not in wrapper.data_params


def test_legacy_base_directory_still_works(quiet_wrapper, tmp_path):
    """Protects mlindex/scripts/run_training_*.py, which pass only base_directory."""
    base = tmp_path / "project"
    _build_models_tree(base / "mlindex" / "models")
    wrapper = quiet_wrapper(data_params={
        "tag": "cubic_1",
        "base_directory": str(base),
        "load_from_tag": True,
    })
    assert Path(wrapper.save_to["results"]) == base / "mlindex" / "models" / "cubic_1"


def test_missing_models_raises_and_creates_nothing(quiet_wrapper, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        quiet_wrapper(data_params={
            "tag": "cubic_1",
            "base_directory": None,
            "models_directory": str(empty),
            "load_from_tag": True,
        })
    # The real regression: it used to mkdir its way to a wrong path instead of failing.
    assert not (empty / "cubic_1").exists()


def test_training_path_creates_dirs(quiet_wrapper, tmp_path):
    target = tmp_path / "training"
    quiet_wrapper(data_params={
        "tag": "cubic_1",
        "base_directory": None,
        "models_directory": str(target),
        "load_from_tag": False,
    })
    for subdir in ("augmentor", "data", "random", "random_forest", "template", "integral_filter"):
        assert (target / "cubic_1" / subdir).is_dir()


# --- factory wiring --------------------------------------------------------------


def test_factories_pass_models_dir_into_data_params(monkeypatch, fake_models_dir):
    """env var -> _resolve_models_dir -> factory -> data_params, with no ONNX or cctbx."""
    from mlindex.optimization import UtilitiesOptimizer as uo

    captured = {}

    class FakeManager:
        def __init__(self, data_params, *args, **kwargs):
            captured.update(data_params)

    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(fake_models_dir))
    uo.get_cubic_optimizer(
        "cP", "1", 1, comm=None,
        project_path=None,
        optimizer_class=FakeManager,
        models_directory=uo._resolve_models_dir(),
    )
    assert captured["models_directory"] == fake_models_dir
    assert captured["tag"] == "cubic_1"
