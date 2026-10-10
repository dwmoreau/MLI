import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from conftest import load_test_case, _TEST_DATA_DIR

EXPECTED_DIR = Path(__file__).parent / "expected"


def _aP_q2(test_metadata):
    row = test_metadata[test_metadata["bravais lattice"] == "aP"].iloc[0]
    q2_obs, unit_cell, wavelength, bl, lattice_system = load_test_case(row)
    return q2_obs


# A real candidate from the S02 grid that cctbx refuses: mC with beta = 149.9 degrees, badly
# non-reduced, the shape the optimizer produces for wrong low-symmetry candidates.
# metric_subgroups raises "Unsuitable value for rational rotation matrix" on it, and because
# _conventional_cell sits on run.py's unconditional output path an unguarded call killed the whole
# run after all the indexing work and before any output was written -- 10 of 16 workers died this
# way on one condition bundle. 0.06% of pooled candidates trip it, which is ~5% of entries.
# Its M20 of 13.2 is above the reporting threshold, so it is a candidate a user would have seen.
_CCTBX_HOSTILE_CELL = {
    'M20': 13.17996795388552, 'n_indexed': 18, 'bravais_lattice': 'mC',
    'volume': 3061.6961360816513,
    'a': 6.040842733459646, 'b': 18.908296254985537, 'c': 53.46502669011422,
    'alpha': 90.0, 'beta': 149.9105411673058, 'gamma': 90.0,
    'spacegroup': 'I 1 2 1', 'score': 0.2, 'M_sym': 10.0,
}


def test_conventional_cell_survives_a_cell_cctbx_refuses():
    from mlindex.command_line.run import _conventional_cell

    # Confirm the fixture still provokes cctbx, so this test cannot pass vacuously.
    from cctbx import crystal as cctbx_crystal
    from cctbx.sgtbx.lattice_symmetry import metric_subgroups
    entry = dict(_CCTBX_HOSTILE_CELL)
    with pytest.raises(Exception):
        metric_subgroups(
            cctbx_crystal.symmetry(
                unit_cell=(entry['a'], entry['b'], entry['c'],
                           entry['alpha'], entry['beta'], entry['gamma']),
                space_group_symbol='P 1'),
            0.1, enforce_max_delta_for_generated_two_folds=True)

    kept = _conventional_cell([entry])

    assert len(kept) == 1, 'a candidate cctbx cannot analyse must still be reported'
    for key in ('bravais_lattice', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'M20'):
        assert kept[0][key] == entry[key], f'{key} must survive un-promoted, not be mangled'


def test_conventional_cell_still_promotes_and_survives_a_mixed_pool():
    # The guard must not have disabled promotion: a cell with cubic metric labelled aP should still
    # be promoted, and it must do so in the same pool as the hostile cell above.
    from mlindex.command_line.run import _conventional_cell

    cubic_metric = {
        'M20': 40.0, 'n_indexed': 20, 'bravais_lattice': 'aP', 'volume': 512.0,
        'a': 8.0, 'b': 8.0, 'c': 8.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
        'spacegroup': 'P 1', 'score': 0.9, 'M_sym': 30.0,
    }
    kept = _conventional_cell([dict(_CCTBX_HOSTILE_CELL), cubic_metric])

    assert len(kept) == 2
    promoted = [e for e in kept if e['M20'] == 40.0][0]
    assert promoted['bravais_lattice'] != 'aP', 'cubic-metric cell should have been promoted'
    survivor = [e for e in kept if e['M20'] != 40.0][0]
    assert survivor['bravais_lattice'] == 'mC'


def test_of_two_copies_of_a_cell_the_higher_scored_is_kept_not_the_higher_m20():
    from mlindex.command_line.run import _conventional_cell

    cell = {'n_indexed': 18, 'bravais_lattice': 'oP', 'volume': 600.0, 'a': 6.0, 'b': 10.0,
            'c': 10.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0, 'spacegroup': 'P 2 2 2'}
    high_m20 = dict(cell, M20=30.0, M_sym=30.0, score=0.1)
    high_score = dict(cell, M20=12.0, M_sym=12.0, score=0.8, c=10.001)
    kept = _conventional_cell([high_m20, high_score])
    assert len(kept) == 1
    assert kept[0]['score'] == 0.8


def test_candidates_with_equal_scores_are_ordered_by_m_sym():
    from mlindex.command_line.run import _conventional_cell

    cell = {'n_indexed': 18, 'bravais_lattice': 'oP', 'volume': 600.0, 'a': 6.0, 'b': 10.0,
            'c': 10.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0, 'spacegroup': 'P 2 2 2',
            'score': 1.0}
    higher_m20 = dict(cell, M20=40.0, M_sym=20.0)
    higher_m_sym = dict(cell, M20=30.0, M_sym=25.0, a=7.0)
    kept = _conventional_cell([higher_m20, higher_m_sym])
    assert [entry['M_sym'] for entry in kept] == [25.0, 20.0]


def _compare_json(result_path, expected_path):
    result = pd.read_json(result_path)
    expected = pd.read_json(expected_path)
    assert list(result.columns) == list(expected.columns), "column mismatch"
    for col in result.select_dtypes(include="number").columns:
        np.testing.assert_allclose(
            result[col].values,
            expected[col].values,
            rtol=1e-6,
            err_msg=f"CLI output mismatch in column {col}",
        )
    for col in result.select_dtypes(exclude="number").columns:
        assert list(result[col]) == list(expected[col]), f"column {col} mismatch"


@pytest.mark.slow
def test_run_analytical_aP(test_metadata, tmp_path):
    q2 = _aP_q2(test_metadata)
    peak_file = tmp_path / "aP_q2.npy"
    np.save(peak_file, q2)
    output_file = tmp_path / "analytic_results.json"

    cmd = [
        sys.executable,
        "-m",
        "mlindex.command_line.run_analytical",
        "--peak-file",
        str(peak_file),
        "--peak-units",
        "q2",
        "--bravais-lattices",
        "aP",
        "--seed",
        "12345",
        "--output-file",
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI exited {result.returncode}:\n{result.stderr}"
    assert output_file.exists(), "output JSON not written"

    _compare_json(output_file, EXPECTED_DIR / "run_analytical_aP.json")


@pytest.mark.slow
def test_run_ml_aP(test_metadata, tmp_path, models_available, models_dir):
    if not models_available:
        pytest.skip("ML models not available")
    # The subprocess resolves models on its own (MLINDEX_MODELS_DIR, then $XDG_DATA_HOME, then
    # the package tree), so without this a developer with a downloaded tree tests models the
    # expected file was never generated against -- exactly what the `models_dir` fixture pins
    # against for the in-process tests.
    env = {**os.environ, "MLINDEX_MODELS_DIR": str(models_dir)}

    q2 = _aP_q2(test_metadata)
    peak_file = tmp_path / "aP_q2.npy"
    np.save(peak_file, q2)

    cmd = [
        sys.executable,
        "-m",
        "mlindex.command_line.run",
        "--peak-file",
        str(peak_file),
        "--peak-units",
        "q2",
        "--bravais-lattices",
        "aP",
        "--nproc",
        "1",
        "--seed",
        "12345",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert result.returncode == 0, f"CLI exited {result.returncode}:\n{result.stderr}"
    output_file = tmp_path / "indexing_results.json"
    assert output_file.exists(), "output JSON not written"

    _compare_json(output_file, EXPECTED_DIR / "run_ml_aP.json")


@pytest.mark.slow
def test_the_answer_is_repeatable_at_a_fixed_process_count(
        test_metadata, tmp_path, models_available, models_dir):
    """A given --nproc must always give the same answer.

    The answer is allowed to change with --nproc: above a certain process count the
    planner splits a heavy lattice and stripes its candidates, which is worth roughly
    2x at fourteen processes and is the deliberate trade. What is not allowed is two
    runs of the same command disagreeing, which is what a benchmark relies on.

    Run at --nproc 8, where the plan does split a lattice, so the striped path is
    exercised rather than only the one-process-per-group path.
    """
    if not models_available:
        pytest.skip("ML models not available")
    # Without this the subprocess resolves models itself and a developer with a
    # downloaded tree tests a different model tree from the one in the checkout.
    env = {**os.environ, "MLINDEX_MODELS_DIR": str(models_dir)}

    q2 = _aP_q2(test_metadata)
    peak_file = tmp_path / "aP_q2.npy"
    np.save(peak_file, q2)

    digests = []
    for repeat in range(2):
        run_dir = tmp_path / f"repeat_{repeat}"
        run_dir.mkdir()
        cmd = [
            sys.executable, "-m", "mlindex.command_line.run",
            "--peak-file", str(peak_file),
            "--peak-units", "q2",
            "--nproc", "8",
            "--seed", "12345",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(run_dir), env=env)
        assert result.returncode == 0, (
            f"run {repeat} exited {result.returncode}:\n{result.stderr}")
        output_file = run_dir / "indexing_results.json"
        assert output_file.exists(), f"run {repeat} wrote no output"
        digests.append(hashlib.sha256(output_file.read_bytes()).hexdigest())

    assert digests[0] == digests[1], (
        "two runs of the same command disagree: "
        f"{digests[0][:16]} then {digests[1][:16]}")


def _models_tree_without_the_ranker(models_dir, directory):
    """A model tree holding every lattice directory of `models_dir` and no ranker, as links."""
    from mlindex.paths import LATTICE_DIR_GLOB
    from mlindex.utilities.Ranker import RANKER_DIRECTORY

    directory.mkdir()
    for lattice_dir in models_dir.glob(LATTICE_DIR_GLOB):
        if lattice_dir.name != RANKER_DIRECTORY:
            try:
                os.symlink(lattice_dir, directory / lattice_dir.name, target_is_directory=True)
            except OSError:
                pytest.skip('this system cannot create directory links')
    return directory


def test_without_ranker_files_the_ranking_falls_back_to_m_sym_and_says_so(tmp_path, monkeypatch):
    from mlindex.utilities.Ranker import load_ranker

    (tmp_path / 'cubic_1' / 'abnn').mkdir(parents=True)
    monkeypatch.setenv('MLINDEX_MODELS_DIR', str(tmp_path))
    ranker = load_ranker()
    assert ranker.name.startswith('M_sym (fallback') and 'download_models' in ranker.name


@pytest.mark.slow
def test_run_ml_without_ranker_files_ranks_by_m_sym_and_says_so(test_metadata, tmp_path,
                                                                models_available, models_dir):
    if not models_available:
        pytest.skip("ML models not available")
    tree = _models_tree_without_the_ranker(models_dir, tmp_path / 'models')
    peak_file = tmp_path / "aP_q2.npy"
    np.save(peak_file, _aP_q2(test_metadata))
    output_file = tmp_path / "indexing_results.json"
    cmd = [sys.executable, "-m", "mlindex.command_line.run", "--peak-file", str(peak_file),
           "--peak-units", "q2", "--bravais-lattices", "aP", "--nproc", "1", "--seed", "12345",
           "--output-file", str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env={**os.environ, "MLINDEX_MODELS_DIR": str(tree)})
    assert result.returncode == 0, result.stderr
    assert "Ranked by: M_sym (fallback" in result.stdout
    output = pd.read_json(output_file)
    np.testing.assert_array_equal(output['score'], output['M_sym'])
    assert (np.diff(output['score'].to_numpy()) <= 0).all()


@pytest.mark.slow
def test_run_ml_with_zero_error_ranks_its_candidates(test_metadata, tmp_path, models_available,
                                                     models_dir):
    """The ranking inputs are computed with each candidate's zero-point."""
    if not models_available:
        pytest.skip("ML models not available")
    row = test_metadata[test_metadata["bravais lattice"] == "aP"].iloc[0]
    peak_file = tmp_path / "aP_q2.npy"
    np.save(peak_file, _aP_q2(test_metadata))
    output_file = tmp_path / "indexing_results.json"
    cmd = [sys.executable, "-m", "mlindex.command_line.run", "--peak-file", str(peak_file),
           "--peak-units", "q2", "--bravais-lattices", "aP", "--nproc", "1", "--seed", "12345",
           "--zero-error", "--wavelength", str(row["wavelength"]),
           "--output-file", str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env={**os.environ, "MLINDEX_MODELS_DIR": str(models_dir)})
    assert result.returncode == 0, result.stderr
    output = pd.read_json(output_file)
    assert len(output) and np.isfinite(output['score']).all()
    assert "Ranked by: learned ranker" in result.stdout
