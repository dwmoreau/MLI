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


# A real candidate from the S02 grid: mC with beta = 149.9 degrees, badly non-reduced, the shape the
# optimizer produces for wrong low-symmetry candidates. cctbx's metric_subgroups refused it, and an
# unguarded call on run.py's output path once killed the whole run after all the indexing work.
# Duplicate removal reduces every printed cell with cctbx, so it must still survive.
_CCTBX_HOSTILE_CELL = {
    'M20': 13.17996795388552, 'n_indexed': 18, 'bravais_lattice': 'mC',
    'volume': 3061.6961360816513,
    'a': 6.040842733459646, 'b': 18.908296254985537, 'c': 53.46502669011422,
    'alpha': 90.0, 'beta': 149.9105411673058, 'gamma': 90.0,
    'spacegroup': 'I 1 2 1', 'score': 0.2, 'M_sym': 10.0,
}


def _monoclinic(a, b, c, beta, **fields):
    """A printed mP entry, and its cell in the basis (a, b, a + c): one lattice, another beta."""
    entry = {'n_indexed': 18, 'bravais_lattice': 'mP', 'a': a, 'b': b, 'c': c, 'alpha': 90.0,
             'beta': beta, 'gamma': 90.0, 'spacegroup': 'P 1 21/c 1', 'M20': 20.0, 'M_sym': 20.0,
             'score': 0.5, **fields}
    entry['volume'] = a*b*c*np.sin(np.radians(beta))
    cosine = np.cos(np.radians(beta))
    c_other = np.sqrt(a**2 + c**2 + 2*a*c*cosine)
    beta_other = np.degrees(np.arccos((a**2 + a*c*cosine)/(a*c_other)))
    return entry, dict(entry, c=c_other, beta=beta_other)


def test_a_cell_cctbx_once_refused_is_still_reported():
    from mlindex.command_line.run import _remove_duplicates

    entry = dict(_CCTBX_HOSTILE_CELL)
    kept = _remove_duplicates([entry])

    assert len(kept) == 1, 'a candidate cctbx finds awkward must still be reported'
    for key in ('bravais_lattice', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'M20'):
        assert kept[0][key] == entry[key], f'{key} must be printed as found'


def test_a_cell_in_another_basis_is_a_duplicate_and_the_higher_scored_is_kept():
    """P17: (a, b, c) and (a, b, a + c) are one monoclinic lattice with different printed
    parameters; comparing the parameters as printed kept both, and each copy took a row."""
    from mlindex.command_line.run import _remove_duplicates

    reduced, oblique = _monoclinic(7.0, 9.0, 11.0, 105.0)
    assert abs(oblique['beta'] - reduced['beta']) > 10, 'the fixture must differ as printed'
    kept = _remove_duplicates([dict(reduced, score=0.3), dict(oblique, score=0.6)])
    assert len(kept) == 1
    assert kept[0]['score'] == 0.6


def test_cells_half_a_percent_apart_are_one_cell_and_two_percent_apart_are_two():
    from mlindex.command_line.run import _remove_duplicates

    entry, _ = _monoclinic(7.0, 9.0, 11.0, 105.0)
    near = dict(entry, b=9.0*1.002, score=0.4)
    far = dict(entry, b=9.0*1.02, score=0.3)
    kept = _remove_duplicates([entry, near, far])
    assert [e['b'] for e in kept] == [9.0, 9.0*1.02]


def test_one_metric_in_two_bravais_lattices_is_two_answers():
    from mlindex.command_line.run import _remove_duplicates

    orthorhombic = {'n_indexed': 18, 'bravais_lattice': 'oP', 'volume': 693.0, 'a': 7.0, 'b': 9.0,
                    'c': 11.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
                    'spacegroup': 'P 2 2 2', 'M20': 20.0, 'M_sym': 20.0, 'score': 0.9}
    monoclinic = dict(orthorhombic, bravais_lattice='mP', spacegroup='P 1 2 1', score=0.5)
    kept = _remove_duplicates([monoclinic, orthorhombic])
    assert [e['bravais_lattice'] for e in kept] == ['oP', 'mP']


def test_two_mc_cells_whose_basis_change_moves_the_centring_point_are_two_answers():
    """mC cells are printed in the I setting, centring point (a + b + c)/2. With a + c in place of
    c the same point lands on the A face, so the two cells are one primitive lattice only if the
    centring is ignored; compared as the lattices they are, both rows stay (STATUS P-F-209)."""
    from mlindex.command_line.run import _remove_duplicates

    entry, moved = _monoclinic(14.988, 13.371, 17.384, 100.42)
    entry = dict(entry, bravais_lattice='mC', spacegroup='I 1 2/m 1', score=0.6)
    other = dict(moved, bravais_lattice='mC', spacegroup='I 1 2/m 1', score=0.4)
    assert len(_remove_duplicates([entry, other])) == 2
    as_primitive = [dict(e, bravais_lattice='mP') for e in (entry, other)]
    assert len(_remove_duplicates(as_primitive)) == 1


def test_a_cell_is_printed_in_the_lattice_it_was_found_in():
    """P17 dropped metric_subgroups promotion: a triclinic cell with a cubic metric stays aP."""
    from mlindex.command_line.run import _remove_duplicates

    cubic_metric = {
        'M20': 40.0, 'n_indexed': 20, 'bravais_lattice': 'aP', 'volume': 512.0,
        'a': 8.0, 'b': 8.0, 'c': 8.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
        'spacegroup': 'P 1', 'score': 0.9, 'M_sym': 30.0,
    }
    kept = _remove_duplicates([dict(cubic_metric)])
    assert kept == [cubic_metric]


def test_of_two_copies_of_a_cell_the_higher_scored_is_kept_not_the_higher_m20():
    from mlindex.command_line.run import _remove_duplicates

    cell = {'n_indexed': 18, 'bravais_lattice': 'oP', 'volume': 600.0, 'a': 6.0, 'b': 10.0,
            'c': 10.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0, 'spacegroup': 'P 2 2 2'}
    high_m20 = dict(cell, M20=30.0, M_sym=30.0, score=0.1)
    high_score = dict(cell, M20=12.0, M_sym=12.0, score=0.8, c=10.001)
    kept = _remove_duplicates([high_m20, high_score])
    assert len(kept) == 1
    assert kept[0]['score'] == 0.8


def test_candidates_with_equal_scores_are_ordered_by_m_sym():
    from mlindex.command_line.run import _remove_duplicates

    cell = {'n_indexed': 18, 'bravais_lattice': 'oP', 'volume': 600.0, 'a': 6.0, 'b': 10.0,
            'c': 10.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0, 'spacegroup': 'P 2 2 2',
            'score': 1.0}
    higher_m20 = dict(cell, M20=40.0, M_sym=20.0)
    higher_m_sym = dict(cell, M20=30.0, M_sym=25.0, a=7.0)
    kept = _remove_duplicates([higher_m20, higher_m_sym])
    assert [entry['M_sym'] for entry in kept] == [25.0, 20.0]


def _cubic_pool(M20, M_sym):
    """A cP pool as `OptimizerBase.result()` hands it over: best M20 first, a full cell a row."""
    M20 = np.asarray(M20, dtype=float)
    a = 3.0 + np.arange(M20.shape[0])
    right = np.full(M20.shape[0], np.pi/2)
    return {'pool': {'M20': M20, 'M_sym': np.asarray(M_sym, dtype=float),
                     'n_indexed': np.full(M20.shape[0], 20),
                     'spacegroup': np.full(M20.shape[0], 'P m -3 m'),
                     'unit_cell': np.stack([a, a, a, right, right, right], axis=1)}}


def test_the_learned_ranker_shortlists_a_lattice_by_its_score_not_by_m20():
    """A candidate M20 puts 25th still reaches the output when it scores best. P16: by M20, a
    correct cell M20 ranked below twentieth in its lattice was scored and then thrown away."""
    from mlindex.command_line.run import SHORTLIST_DEPTH, _collect_results
    from mlindex.utilities.Ranker import LearnedRanker

    n = SHORTLIST_DEPTH + 5
    result = _cubic_pool(M20=np.linspace(30.0, 10.0, n), M_sym=np.full(n, 10.0))
    scores = np.linspace(0.0, 1.0, n)
    rows = LearnedRanker.__new__(LearnedRanker).shortlist(result['pool'], scores,
                                                          SHORTLIST_DEPTH)
    entries = _collect_results(result, 'cP', [], scores, rows)

    assert [entry['score'] for entry in entries] == sorted(scores)[::-1][:SHORTLIST_DEPTH]
    assert entries[0]['a'] == 3.0 + n - 1
    assert entries[0]['M20'] == 10.0


def test_the_learned_ranker_s_shortlist_breaks_equal_scores_by_m_sym():
    """As the pooled list does (`_remove_duplicates`), so the shortlist keeps what it would rank
    first: the learned ranker's calibrated probability is often exactly 1.0 for several cells."""
    from mlindex.command_line.run import SHORTLIST_DEPTH
    from mlindex.utilities.Ranker import LearnedRanker

    n = SHORTLIST_DEPTH + 1
    M_sym = np.full(n, 10.0)
    M_sym[-1] = 50.0
    pool = _cubic_pool(M20=np.linspace(30.0, 10.0, n), M_sym=M_sym)['pool']
    rows = LearnedRanker.__new__(LearnedRanker).shortlist(pool, np.ones(n), SHORTLIST_DEPTH)

    assert len(rows) == SHORTLIST_DEPTH
    assert rows[0] == n - 1


def test_ranking_by_m_sym_shortlists_a_lattice_by_m20():
    """Ranked by M_sym, a shortlist by M_sym costs orthorhombic and monoclinic 2-3 points of
    first place (P16), so the fallback and `run_analytical` keep the twenty best by M20: the
    pool's first rows."""
    from mlindex.utilities.Ranker import MSymRanker

    n = 25
    pool = _cubic_pool(M20=np.linspace(30.0, 10.0, n), M_sym=np.linspace(0.0, 50.0, n))['pool']
    rows = MSymRanker('test').shortlist(pool, pool['M_sym'], 20)

    assert rows.tolist() == list(range(20))


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
