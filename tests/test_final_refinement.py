"""The final refinement step: which peaks it is taken on, and the zero-point."""
import os

import numpy as np

from mlindex.optimization.Candidates import Candidates
from mlindex.utilities.Q2Calculator import Q2Calculator

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVELENGTH = 1.5406
A_TRUE = 8.0


def _cubic(zeropoint, n=8, seed=0):
    """Candidates at the true cubic cell, on a pattern whose peaks carry a zero-point shift in theta.

    The peaks are shifted exactly, by moving each theta, not through the linearised correction
    the code applies, so the test does not agree with the code by construction.
    """
    hkl_ref = np.load(os.path.join(BASE, 'mlindex', 'models', 'cubic_1', 'data',
                                   'hkl_ref_cF.npy'))
    xnn_true = np.array([[1.0 / A_TRUE ** 2]])
    q2_ref = Q2Calculator(lattice_system='cubic', hkl=hkl_ref, tensorflow=False,
                          representation='xnn').get_q2(xnn_true)[0]
    q2_true = np.sort(q2_ref[q2_ref > 0])[:20]
    theta = np.arcsin(WAVELENGTH / 2 * np.sqrt(q2_true))
    noise = np.random.default_rng(seed).normal(0, 1e-5, size=theta.size)
    q2_obs = np.sort((2 * np.sin(theta + zeropoint + noise) / WAVELENGTH) ** 2)
    opt_params = {'minimum_uc': 2.0, 'maximum_uc': 60.0, 'assignment_threshold': 0.95,
                  'figure_of_merit': 'M20'}
    candidates = Candidates(q2_obs=q2_obs, xnn=np.repeat(xnn_true, n, axis=0), hkl_ref=hkl_ref,
                            lattice_system='cubic', bravais_lattice='cF', opt_params=opt_params,
                            rng=np.random.default_rng(seed), fom=None, zero_error=True,
                            wavelength=WAVELENGTH)
    candidates.best_zeropoint[:] = zeropoint
    return candidates


def test_the_peak_filter_sees_the_zero_point():
    """At 1.1 degrees of 2-theta the unshifted lines miss most peaks: without the zero-point the
    filter admits 3 of these 20 true peaks, with it all 20."""
    candidates = _cubic(zeropoint=1e-2)

    assert np.all(candidates.indexed_peaks().sum(axis=1) == 20)
    candidates.calculate_peaks_indexed()
    assert np.all(candidates.n_indexed == 20)


def test_the_final_step_is_taken_on_the_filtered_peaks(monkeypatch):
    """The step and the reported count read one filter, so they cannot disagree again."""
    candidates = _cubic(zeropoint=1e-2)
    calls = []
    original = Candidates.indexed_peaks

    def spy(self):
        calls.append(True)
        return original(self)

    monkeypatch.setattr(Candidates, 'indexed_peaks', spy)
    candidates.refine_cell()

    assert calls
