"""The driver that measures the clump discount, and the properties that make it runnable alone."""
import os
import subprocess
import sys

import numpy as np
import pytest

from mlindex.scripts import run_clump_discount as driver


def test_the_help_is_pure_ascii():
    driver.build_parser().format_help().encode('ascii')


def test_cubic_is_refused_with_the_reason(tmp_path):
    """One free parameter means a random direction is +1 or -1, so every member lands at exactly
    +/- delta/2 and half of each group is coincident whatever separation is asked for."""
    with pytest.raises(SystemExit, match='one free parameter'):
        driver.main(['--stage', 'reduce', '--bravais-lattices', 'cP',
                     '--out-dir', str(tmp_path), '--roc-dir', str(tmp_path)])


def test_the_first_separation_must_be_zero(tmp_path):
    """It is the identical-start case the rest is read against."""
    with pytest.raises(SystemExit, match='must be 0'):
        driver.main(['--stage', 'measure', '--bravais-lattices', 'oP',
                     '--separation-ratios', '0.1,0.3', '--dataset-directory', str(tmp_path),
                     '--out-dir', str(tmp_path), '--roc-dir', str(tmp_path)])


def test_the_reduce_stage_imports_no_mpi():
    """Measuring needs a cluster; reducing is seconds on a laptop and must not need a launcher."""
    probe = ('import sys\n'
             'from mlindex.scripts import run_clump_discount\n'
             'print("mpi4py" in sys.modules)\n')
    environment = {k: v for k, v in os.environ.items() if k != 'KERAS_BACKEND'}
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                            check=True, env=environment)
    assert result.stdout.strip() == 'False', result.stdout


def test_alpha_is_one_when_a_group_never_fails_together():
    """Independence is alpha = 1; the solve must return it rather than something clipped."""
    radii = np.array([1e-4, 1e-3])
    success = np.array([0.5, 0.5])
    # four groups of two at one shell, and in every group at least one member converged
    correct = np.ones((4, 1, 2, 2), dtype=bool)
    correct[..., 1] = False
    distance = np.full((4, 1, 2, 2), 1e-4)
    assert np.isnan(driver.solve_alpha(correct, distance, radii, success, 2))


def test_alpha_is_clipped_into_the_range_a_clump_can_occupy():
    """k_eff lies between 1 and k, so alpha lies in [1/k, 1]; outside is the solve failing."""
    radii = np.array([1e-4, 1e-3])
    success = np.array([0.5, 0.5])
    correct = np.zeros((40, 1, 4, 4), dtype=bool)      # every group fails entirely
    correct[0, 0, 0, 0] = True                         # except one, so the rate is not 0 or 1
    distance = np.full((40, 1, 4, 4), 1e-4)
    alpha = driver.solve_alpha(correct, distance, radii, success, 4)
    assert 1.0/4 <= alpha <= 1.0


def test_the_shells_are_picked_where_the_curve_can_resolve_a_correlation():
    radii = np.logspace(-4, -2, 40)
    success = np.linspace(0.95, 0.02, 40)
    picked, rates = driver.pick_radii(radii, success, driver.TARGET_SUCCESS_RATES)
    assert picked.size == len(driver.TARGET_SUCCESS_RATES)
    assert rates.max() < 0.95 and rates.min() > 0.02
