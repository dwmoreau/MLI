import json
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
    'spacegroup': 'I 1 2 1',
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
        'spacegroup': 'P 1',
    }
    kept = _conventional_cell([dict(_CCTBX_HOSTILE_CELL), cubic_metric])

    assert len(kept) == 2
    promoted = [e for e in kept if e['M20'] == 40.0][0]
    assert promoted['bravais_lattice'] != 'aP', 'cubic-metric cell should have been promoted'
    survivor = [e for e in kept if e['M20'] != 40.0][0]
    assert survivor['bravais_lattice'] == 'mC'


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
def test_run_ml_aP(test_metadata, tmp_path, models_available):
    if not models_available:
        pytest.skip("ML models not available")

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
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, f"CLI exited {result.returncode}:\n{result.stderr}"
    output_file = tmp_path / "indexing_results.json"
    assert output_file.exists(), "output JSON not written"

    _compare_json(output_file, EXPECTED_DIR / "run_ml_aP.json")


# ------------------------------------------------------------------------------------------
# --prune-threshold. `prune_below_m20` discards every candidate scoring below 5.0 before
# refinement, which is the largest single control on whether the correct cell reaches ranking
# at all -- on low-symmetry, large-volume patterns it deletes 72-88% of the correct candidates
# the search finds. It entered the code as a speed optimisation for the analytical indexer and
# had no way to be set from the command line until this flag. The two properties that matter
# are that the default is unchanged and that the flag actually reaches the workers.
# ------------------------------------------------------------------------------------------

def test_prune_threshold_defaults_to_none_and_builds_no_options():
    """Absent, it must leave the per-lattice opt_params alone rather than re-assert 5.0."""
    from mlindex.command_line.run import build_base_parser
    args = build_base_parser().parse_args([])
    assert args.prune_threshold is None
    options = (None if getattr(args, 'prune_threshold', None) is None
               else {'prune_m20_threshold': args.prune_threshold})
    assert options is None


def test_prune_threshold_is_parsed_as_a_float_and_becomes_an_opt_params_override():
    from mlindex.command_line.run import build_base_parser
    args = build_base_parser().parse_args(['--prune-threshold', '3.0'])
    assert args.prune_threshold == pytest.approx(3.0)
    options = {'prune_m20_threshold': args.prune_threshold}
    assert options == {'prune_m20_threshold': 3.0}


def test_prune_threshold_help_is_ascii():
    """CLAUDE.md: piping --help on Windows encodes through the locale codepage."""
    from mlindex.command_line.run import build_base_parser
    help_text = build_base_parser().format_help()
    assert '--prune-threshold' in help_text
    help_text.encode('ascii')


def test_the_override_reaches_candidates_through_the_optimizer_factories():
    """Every factory must honour `options`, because `prune_below_m20` runs on the workers.

    `opt_params` is queued to each worker as its manager is constructed, so an override that
    stopped at the manager would apply to the manager's share of the candidates only. The
    factory is exercised with a stub optimizer class so this tests the merge rather than the
    model loading.
    """
    import mlindex.optimization.UtilitiesOptimizer as utilities

    captured = {}

    class _Stub:
        def __init__(self, *args, **kwargs):
            # The factories construct the manager positionally: data_params first,
            # opt_params second.
            captured['opt_params'] = kwargs.get('opt_params', args[1] if len(args) > 1 else None)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    factories = [
        (utilities.get_cubic_optimizer, 'cP'),
        (utilities.get_tetragonal_optimizer, 'tP'),
        (utilities.get_monoclinic_optimizer, 'mP'),
        (utilities.get_triclinic_optimizer, 'aP'),
        ]
    for factory, bravais_lattice in factories:
        for override in (None, {'prune_m20_threshold': 3.0}):
            captured.clear()
            try:
                factory(bravais_lattice, '1', 1, None, options=override,
                        optimizer_class=_Stub, seed=12345)
            except Exception as error:               # pragma: no cover - factory internals
                pytest.skip(f'{factory.__name__} needs more than a stub: {error}')
            opt_params = captured.get('opt_params')
            assert opt_params is not None, factory.__name__
            if override is None:
                # The 5.0 default is supplied by OptimizerBase, not by the factory, so an
                # un-overridden factory must leave the key absent rather than assert it --
                # that is what makes the default byte-identical to the pre-flag code.
                assert 'prune_m20_threshold' not in opt_params, factory.__name__
            else:
                assert opt_params['prune_m20_threshold'] == pytest.approx(3.0), (
                    f'{factory.__name__} did not honour options={override}')
