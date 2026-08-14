"""mlindex.download_models.

Fully offline: huggingface_hub is never called for real, and no model files are needed.
The download itself is exercised through the `_hf_imports` seam.
"""

import json
from pathlib import Path

import pytest

from mlindex.command_line import download_models as dm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MLINDEX_MODELS_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


# --- target directory precedence -------------------------------------------------


def test_models_dir_argument_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(tmp_path / "from_env"))
    assert dm._resolve_target_dir(str(tmp_path / "from_arg")) == tmp_path / "from_arg"


def test_env_var_used_when_no_argument(monkeypatch, tmp_path):
    monkeypatch.setenv("MLINDEX_MODELS_DIR", str(tmp_path / "from_env"))
    assert dm._resolve_target_dir(None) == tmp_path / "from_env"


def test_default_used_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert dm._resolve_target_dir(None) == tmp_path / "mlindex" / "models"


# --- existing-install guard ------------------------------------------------------


def test_guard_reports_empty(tmp_path):
    assert dm._existing_install_state(tmp_path / "missing") == "empty"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert dm._existing_install_state(empty) == "empty"


def test_guard_blocks_foreign_data(tmp_path):
    target = tmp_path / "documents"
    target.mkdir()
    (target / "notes.txt").write_text("important")
    assert dm._existing_install_state(target) == "foreign_data"


def test_guard_allows_resume_over_previous_download(tmp_path):
    target = tmp_path / "models"
    (target / "cubic_1").mkdir(parents=True)
    assert dm._existing_install_state(target) == "previous_download"


def test_guard_allows_resume_over_hf_metadata_only(tmp_path):
    """An interrupted download may have written only the metadata folder so far."""
    target = tmp_path / "models"
    (target / ".cache" / "huggingface").mkdir(parents=True)
    assert dm._existing_install_state(target) == "previous_download"


def test_guard_ignores_dotfiles(tmp_path):
    target = tmp_path / "models"
    target.mkdir()
    (target / ".DS_Store").write_bytes(b"\x00")
    assert dm._existing_install_state(target) == "empty"


# --- metadata --------------------------------------------------------------------


def test_shipped_metadata_has_hf_keys():
    import mlindex

    meta = json.loads((Path(mlindex.__file__).parent / "model_metadata.json").read_text())
    assert meta["hf_repo_id"]
    assert meta["model_revision"]
    # The GitHub fallback keys stay for one release; --source github depends on them.
    assert meta["repo_url"]
    assert meta["models_branch"]


def test_metadata_unreadable_falls_back_to_defaults(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr("builtins.open", boom)
    assert dm._load_metadata() == {}
    assert "WARNING" in capsys.readouterr().err


# --- the snapshot_download call --------------------------------------------------


class _Recorder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return "/fake/snapshot"


@pytest.fixture
def recorder(monkeypatch):
    """Replace the huggingface_hub seam with a recording fake."""
    rec = _Recorder()

    class FakeErrors:
        class RevisionNotFoundError(Exception):
            pass

        class RepositoryNotFoundError(Exception):
            pass

        class LocalEntryNotFoundError(Exception):
            pass

        class HfHubHTTPError(Exception):
            pass

    monkeypatch.setattr(dm, "_hf_imports", lambda: (rec, FakeErrors))
    return rec


def test_snapshot_download_kwargs(recorder, tmp_path):
    target = tmp_path / "models"
    dm._download_hf("owner/repo", "v1", target, redownload=False)

    assert recorder.kwargs["repo_id"] == "owner/repo"
    assert recorder.kwargs["revision"] == "v1"
    assert recorder.kwargs["repo_type"] == "model"
    assert recorder.kwargs["local_dir"] == str(target)
    # Removed in huggingface_hub 1.x -- passing any of these is a TypeError for users.
    assert "local_dir_use_symlinks" not in recorder.kwargs
    assert "resume_download" not in recorder.kwargs
    # cache_dir is unused when local_dir is set; passing it implies a second 545 MB copy.
    assert "cache_dir" not in recorder.kwargs


def test_force_does_not_imply_redownload(recorder, tmp_path, monkeypatch):
    """--force only bypasses the directory guard; it must not refetch 545 MB."""
    monkeypatch.setattr(
        "sys.argv",
        ["mlindex.download_models", "--models-dir", str(tmp_path / "models"), "--force"],
    )
    dm.main()
    assert recorder.kwargs["force_download"] is False


def test_redownload_flag_sets_force_download(recorder, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["mlindex.download_models", "--models-dir", str(tmp_path / "models"), "--redownload"],
    )
    dm.main()
    assert recorder.kwargs["force_download"] is True


def test_revision_and_repo_id_overrides(recorder, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["mlindex.download_models", "--models-dir", str(tmp_path / "models"),
         "--revision", "main", "--repo-id", "someone/else"],
    )
    dm.main()
    assert recorder.kwargs["revision"] == "main"
    assert recorder.kwargs["repo_id"] == "someone/else"


def test_defaults_come_from_shipped_metadata(recorder, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["mlindex.download_models", "--models-dir", str(tmp_path / "models")],
    )
    dm.main()
    meta = dm._load_metadata()
    assert recorder.kwargs["repo_id"] == meta["hf_repo_id"]
    assert recorder.kwargs["revision"] == meta["model_revision"]


def test_foreign_directory_blocks_without_force(recorder, tmp_path, monkeypatch, capsys):
    target = tmp_path / "documents"
    target.mkdir()
    (target / "notes.txt").write_text("important")
    monkeypatch.setattr(
        "sys.argv", ["mlindex.download_models", "--models-dir", str(target)]
    )
    with pytest.raises(SystemExit) as excinfo:
        dm.main()
    assert excinfo.value.code == 1
    assert "unrelated files" in capsys.readouterr().err
    assert recorder.kwargs is None


def test_revision_not_found_is_self_diagnosing(monkeypatch, tmp_path, capsys):
    class FakeErrors:
        class RevisionNotFoundError(Exception):
            pass

        class RepositoryNotFoundError(Exception):
            pass

        class LocalEntryNotFoundError(Exception):
            pass

        class HfHubHTTPError(Exception):
            pass

    def raiser(**kwargs):
        raise FakeErrors.RevisionNotFoundError()

    monkeypatch.setattr(dm, "_hf_imports", lambda: (raiser, FakeErrors))
    with pytest.raises(SystemExit):
        dm._download_hf("owner/repo", "v9", tmp_path / "models")
    err = capsys.readouterr().err
    assert "--revision main" in err


@pytest.mark.parametrize(
    "message, expected, unexpected",
    [
        # huggingface_hub wraps transport failures in OSError, so a rate limit must not
        # be reported as a disk-space problem.
        ("Network error: HTTP status client error (429 Too Many Requests)",
         "rate-limited", "free space"),
        ("Network error: Connection reset by peer", "network error", "free space"),
        ("[Errno 28] No space left on device", "free space", "rate-limited"),
    ],
)
def test_oserror_is_classified(monkeypatch, tmp_path, capsys, message, expected, unexpected):
    class FakeErrors:
        class RevisionNotFoundError(Exception):
            pass

        class RepositoryNotFoundError(Exception):
            pass

        class LocalEntryNotFoundError(Exception):
            pass

        class HfHubHTTPError(Exception):
            pass

    def raiser(**kwargs):
        raise OSError(message)

    monkeypatch.setattr(dm, "_hf_imports", lambda: (raiser, FakeErrors))
    with pytest.raises(SystemExit):
        dm._download_hf("owner/repo", "v1", tmp_path / "models")
    err = capsys.readouterr().err
    assert expected in err
    assert unexpected not in err


def test_missing_huggingface_hub_message(monkeypatch, tmp_path, capsys):
    def no_hub():
        raise ImportError("No module named 'huggingface_hub'")

    monkeypatch.setattr(dm, "_hf_imports", no_hub)
    with pytest.raises(SystemExit):
        dm._download_hf("owner/repo", "v1", tmp_path / "models")
    err = capsys.readouterr().err
    assert "pip install" in err
    assert "--source github" in err
