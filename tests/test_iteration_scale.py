"""S06 -- the iteration schedule scale factor, and the property the pilot's design rests on.

`iteration_scale` is DWMM's compute lever: run the search for half, or a quarter, of the
iterations. Three things have to hold for the pilot that prices it to mean anything.

1. **Scale 1.0 is exactly the shipped schedule.** Anything else and every campaign-1 comparison
   and every earlier campaign-2 number is measured against a different search.
2. **No block scales to zero.** `random_subsampling` is what makes the search stochastic; a
   lattice system whose only random block vanished would become a one-shot deterministic solve
   rather than a cheaper search, and would look like a catastrophic ceiling loss for the wrong
   reason.
3. **The arms nest.** The half arm must draw the *same* peak subsets as the full arm's first k
   iterations, so the three schedules are prefixes of one another rather than three independent
   searches. That is what makes the pilot a paired comparison -- F-137 established that two arms
   of the same configuration are not otherwise comparable.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.optimization.MPIOptimizer import OptimizerBase, _scaled_iterations

# The shipped schedules, read off UtilitiesOptimizer's factories. Kept here as data so a change
# to one of them fails this test rather than silently moving what the pilot measured.
SHIPPED_RANDOM_ITERATIONS = {
    'cubic': 5,
    'tetragonal': 30,
    'hexagonal': 30,
    'rhombohedral': 30,
    'orthorhombic': 50,
    'monoclinic': 60,
    'triclinic': 60,
    }


def _schedule(n_random, n_drop=17, n_peaks=20):
    return [
        {'worker': 'deterministic', 'n_iterations': 1},
        {'worker': 'random_subsampling', 'n_iterations': n_random,
         'n_peaks': n_peaks, 'n_drop': n_drop, 'uniform_sampling': False},
        ]


class _Recorder:
    """A stand-in for `Candidates` that records the iteration calls and no-ops the rest.

    The post-loop block is six method calls that do real linear algebra on real candidates; none
    of it is what this test is about, so it is stubbed. What is *not* stubbed is the loop itself,
    which is the code under test.
    """

    def __init__(self):
        self.calls = []

    def deterministic(self, iteration_info):
        self.calls.append('deterministic')

    def random_subsampling(self, iteration_info):
        self.calls.append('random_subsampling')

    def random_subsampling_power(self, iteration_info):
        self.calls.append('random_subsampling_power')

    def random_power(self, iteration_info):
        self.calls.append('random_power')

    def prune_below_m20(self, threshold=None):
        pass

    def refine_cell(self):
        pass

    def standardize_cell(self):
        pass

    def correct_off_by_two(self):
        pass

    def assign_extinction_group(self):
        pass

    def calculate_peaks_indexed(self):
        pass


class _LoopHarness(OptimizerBase):
    """Runs `OptimizerBase._run_loop` against a recorder, and nothing else."""

    def __init__(self, opt_params):
        self.opt_params = opt_params
        self.recorder = _Recorder()

    def generate_candidates_rank(self):
        return self.recorder

    def downsample_candidates(self, candidates, n_top_candidates):
        pass


def _run(n_random, scale=None):
    opt_params = {
        'iteration_info': _schedule(n_random),
        'convergence_testing': False,
        'redistribution_testing': False,
        }
    if scale is not None:
        opt_params['iteration_scale'] = scale
    harness = _LoopHarness(opt_params)
    harness._run_loop(n_top_candidates=20)
    return harness.recorder.calls


@pytest.mark.parametrize('lattice_system,n_random', sorted(SHIPPED_RANDOM_ITERATIONS.items()))
def test_scale_one_is_the_shipped_schedule(lattice_system, n_random):
    # The default path: no `iteration_scale` key at all, which is what every shipped opt_params
    # dict looks like. It must be indistinguishable from an explicit 1.0.
    assert _run(n_random) == _run(n_random, scale=1.0)
    assert _run(n_random) == ['deterministic'] + ['random_subsampling'] * n_random


@pytest.mark.parametrize('lattice_system,n_random', sorted(SHIPPED_RANDOM_ITERATIONS.items()))
def test_reduced_schedules_are_prefixes(lattice_system, n_random):
    full = _run(n_random, scale=1.0)
    for scale in (0.5, 0.25):
        reduced = _run(n_random, scale=scale)
        assert reduced == full[:len(reduced)], f'{lattice_system} at {scale} is not a prefix'
        assert len(reduced) < len(full)


@pytest.mark.parametrize('lattice_system,n_random', sorted(SHIPPED_RANDOM_ITERATIONS.items()))
def test_the_deterministic_pass_is_never_scaled_away(lattice_system, n_random):
    for scale in (1.0, 0.5, 0.25, 0.01):
        calls = _run(n_random, scale=scale)
        assert calls.count('deterministic') == 1
        # And the random block survives too, however small the scale. Cubic's five passes are the
        # binding case: at 0.01 the arithmetic gives 0.05, and a block that rounded to zero would
        # turn the search deterministic instead of cheap.
        assert calls.count('random_subsampling') >= 1


def test_scaled_iterations_arithmetic():
    block = {'n_iterations': 60}
    assert _scaled_iterations(block, 1.0) == 60
    assert _scaled_iterations(block, 0.5) == 30
    assert _scaled_iterations(block, 0.25) == 15
    # Cubic, where the schedule is too short for a scale factor to be fine-grained. The pilot
    # excludes cubic from its per-lattice comparison for exactly this reason.
    assert _scaled_iterations({'n_iterations': 5}, 0.5) == 2
    assert _scaled_iterations({'n_iterations': 5}, 0.25) == 1
    with pytest.raises(ValueError):
        _scaled_iterations(block, 0.0)
    with pytest.raises(ValueError):
        _scaled_iterations(block, -1.0)


def test_scale_one_short_circuits_exactly():
    # Not `round(n * 1.0)`, which would be an identity by luck rather than by construction. The
    # shipped count is returned unchanged, whatever it is.
    for n_iterations in (0, 1, 3, 5, 7, 30, 50, 60, 999):
        assert _scaled_iterations({'n_iterations': n_iterations}, 1.0) == n_iterations


def test_the_random_draws_nest():
    """The claim the pilot's pairing rests on, tested on the draw itself.

    `Candidates.random_subsampling` takes its subset from `self.rng` with shapes fixed by the peak
    count and the candidate count -- neither of which the iteration changes -- so a half-length
    run drawing from the same seed must produce the first half of the full run's subsets. If that
    ever stops being true, the three arms are three independent searches and every paired number
    in the pilot is wrong.
    """
    from mlindex.optimization.Candidates import vectorized_subsampling

    q2_obs = np.linspace(0.02, 0.4, 20)
    arg = 1.0 / q2_obs
    p = np.repeat((arg / arg.sum())[np.newaxis], 64, axis=0)

    def draws(n_iterations, seed=4321):
        rng = np.random.default_rng(seed)
        return [vectorized_subsampling(p, 3, rng) for _ in range(n_iterations)]

    full = draws(60)
    for n_iterations in (30, 15):
        reduced = draws(n_iterations)
        assert len(reduced) == n_iterations
        for position, drawn in enumerate(reduced):
            assert np.array_equal(drawn, full[position])
