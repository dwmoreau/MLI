"""Properties of the repository itself, rather than of anything it computes.

Two of them, and each is something that has gone wrong here or would go wrong silently.

**The per-peak assignment network is gone and must stay gone.** It was 43 quantized ONNX graphs,
105 MB of the model download and 182 MB resident per process, and a closed-form posterior replaced
it. Deleting the files is not what makes that stick -- code that still reaches for them would fail
at a user's first run, three call frames from the cause. So the shipped package is scanned for any
means of loading one.

**Model files are on the right side of git-lfs, and the line runs both ways.** The rules in
`.gitattributes` are a mixture of 129 enumerated paths and a few basename globs, so renaming a
directory or a filename stem silently drops files out of lfs and into the pack -- and a blob in
the pack cannot be removed again without rewriting history. The other direction matters too: the
`hkl_ref_*.npy` reference lists are the only model files `pyproject.toml` puts in the wheel, and
if they were lfs-tracked a wheel built from a checkout that never ran `git lfs pull` would ship
130-byte pointer files in place of the data.
"""
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parent.parent
PACKAGE = REPOSITORY / "mlindex"

# Names that only a per-peak assignment network needs. `calibration_data=` is deliberately not
# here: that is ONNX post-training quantization, an unrelated use of the word which the surviving
# model still needs.
RETIRED = (
    "predict_hkl",
    "calibration_onnx_model",
    "calibration_model",
    "calibration_params",
    "_calibration_weights_",
    "build_calibration_model",
    "train_calibration",
    "model_builder_calibration",
    "PairwiseDifferenceCalculator",
)

# An IPython session transcript rather than importable code -- it opens with three exclamation
# marks and uses %lprun, so it cannot be imported or run, and the pasted profiler output quotes
# code as it stood in April. Rewriting quoted output would falsify the record it is.
NOT_CODE = {PACKAGE / "scripts" / "ProfileOptimizer.py"}


def _shipped_sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        if path in NOT_CODE or "models" in path.relative_to(PACKAGE).parts:
            continue
        yield path


def test_no_shipped_code_can_load_a_peak_assignment_network():
    offenders = []
    for path in _shipped_sources():
        text = path.read_text(encoding="utf-8")
        for name in RETIRED:
            if name in text:
                line = next(
                    number
                    for number, content in enumerate(text.splitlines(), start=1)
                    if name in content
                )
                offenders.append(f"{path.relative_to(REPOSITORY)}:{line} names {name!r}")
    assert not offenders, (
        "The per-peak assignment network was retired and no longer ships; these would fail at "
        "a user's first run:\n  " + "\n  ".join(offenders)
    )


def test_no_calibration_graph_ships_in_the_model_tree():
    graphs = sorted(PACKAGE.glob("models/*/*/*/*_calibration_weights_*_quantized.onnx"))
    assert not graphs, "\n".join(str(path.relative_to(REPOSITORY)) for path in graphs)


def _model_file_filters():
    """Map every tracked model path to its git-lfs filter, from `.gitattributes` alone.

    `git check-attr` needs neither the file nor its lfs object, so this works on a checkout that
    has not run `git lfs pull`.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "mlindex/models"],
        cwd=REPOSITORY, capture_output=True, text=True, check=True,
    ).stdout.split()
    if not tracked:
        return {}
    attrs = subprocess.run(
        ["git", "check-attr", "--stdin", "filter"],
        cwd=REPOSITORY, input="\n".join(tracked), capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    filters = {}
    for line in attrs:
        path, _, value = line.rsplit(": ", 2)
        filters[path] = value
    return filters


def test_every_model_file_the_wheel_does_not_ship_is_covered_by_git_lfs():
    filters = _model_file_filters()
    if not filters:
        pytest.skip("no model files are tracked in this checkout")
    missed = [path for path, value in filters.items()
              if value != "lfs" and not path.endswith(".npy")]
    assert not missed, (
        "These model files match no git-lfs rule in .gitattributes and would be committed as "
        "plain blobs, which cannot be removed without rewriting history:\n  "
        + "\n  ".join(missed)
    )


def test_the_reference_lists_the_wheel_ships_are_not_git_lfs():
    """pyproject.toml's package-data glob, which must be real data in a built wheel."""
    filters = _model_file_filters()
    if not filters:
        pytest.skip("no model files are tracked in this checkout")
    shipped = {path: value for path, value in filters.items()
               if "/data/hkl_ref_" in path}
    assert shipped, "pyproject.toml ships models/*/data/hkl_ref_*.npy; none are tracked"
    lfs_tracked = [path for path, value in shipped.items() if value == "lfs"]
    assert not lfs_tracked, (
        "These are in the wheel via pyproject.toml's package-data, so an lfs pointer would be "
        "shipped in place of the data:\n  " + "\n  ".join(lfs_tracked)
    )
