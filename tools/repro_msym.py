#!/usr/bin/env python
"""Reproducer: get_M_rev_sym materialises an (n_candidates, n_ref, n_peaks) array.

DEVELOPMENT TOOL -- not part of the installed package.

Claim under test
----------------
`get_M_rev_sym` (FigureOfMerits.py) computes M_tilde, M_rev and M_sym.  Its
reversed term scores every reference line against its nearest observed peak::

    nearest = np.min(np.abs(q2_ref_calc[:, :, None] - q2_obs[None, None]), axis=2)

which is O(n_candidates x n_ref x n_peaks) in *memory*, not only in time -- 1.6 GB
for 10 000 candidates against a 1 000-line reference list.  `S06_zoo_cost.csv`
prices M_sym at 24.3 get_M20-equivalents, and F-070 makes that cost the reason
the merit cannot enter the inner loop.

Two things are wasted there.  The nearest observed peak to a value can be found
by binary search in the sorted peak list instead of by scanning all n_peaks, and
-- more importantly -- `nearest` is only ever read at the entries `in_range`
selects, which is roughly a tenth of the reference list (S01 audit A: N is 12-100
of a 99-999 line list).  The other nine tenths are computed and thrown away.

Everything the row sums touch must stay bit-identical, so the summed arrays are
rebuilt at full (n_candidates, n_ref) shape and reduced by the same `.sum(axis=1)`
as before: numpy's pairwise summation groups differently if the zeros are
dropped, and that is a real ULP difference, not a theoretical one.

Usage
-----
Capture the real arguments for a lattice (writes <out>/msym_<BL>.npz)::

    python tools/repro_msym.py capture --peak-file 11bmb_3844_peak_list.npy \
        --peak-units q2 --bravais-lattice mP

Line-profile the shipped implementation at 1 000 and 10 000 candidates::

    python tools/repro_msym.py profile --capture-file captures/msym_mP.npz

Benchmark every variant against the capture::

    python tools/repro_msym.py bench --capture-file captures/msym_mP.npz

Check bit-identity on randomised inputs, including the edge cases the capture
cannot contain (empty in-range rows, explicit weights, ties, unsorted peaks)::

    python tools/repro_msym.py fuzz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from numba import jit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_hook import capture_get_M20
from microbench import benchmark, exact_match, tuple_match

from mlindex.utilities.FigureOfMerits import get_M20
from mlindex.utilities.FigureOfMerits import get_M_rev_sym


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def current(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """The shipped implementation, copied verbatim for a like-for-like timing."""
    n_peaks = q2_obs.shape[0]
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)

    q_max = q2_calc[:, -1]
    q_min = np.take_along_axis(
        q2_ref_calc,
        np.argmin(np.abs(q2_ref_calc - q2_obs[0]), axis=1)[:, np.newaxis],
        axis=1,
    )[:, 0]

    in_range = (q2_ref_calc >= q_min[:, np.newaxis]) & (q2_ref_calc <= q_max[:, np.newaxis])
    row_weights = np.ones(q2_ref_calc.shape[1]) if weights is None else weights
    n_cal = (in_range*row_weights[np.newaxis, :]).sum(axis=1)
    q_n = np.max(np.where(in_range, q2_ref_calc, -np.inf), axis=1)

    nearest = np.min(np.abs(q2_ref_calc[:, :, np.newaxis] - q2_obs[np.newaxis, np.newaxis]), axis=2)
    reversed_sum = (np.where(in_range, nearest, 0.0)*row_weights[np.newaxis, :]).sum(axis=1)

    good = (n_cal > 0) & np.isfinite(q_n) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    M_tilde = np.zeros(q2_calc.shape[0])
    M_rev = np.zeros(q2_calc.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = (q_n - q_min)/(2*np.where(n_cal > 0, n_cal, 1))
        M_tilde[good] = epsilon[good]/discrepancy[good]
        discrepancy_reversed = reversed_sum/np.where(n_cal > 0, n_cal, 1)
        epsilon_reversed = (q2_obs[-1] - q2_obs[0])/(2*n_peaks)
        usable = good & (discrepancy_reversed > 0)
        M_rev[usable] = epsilon_reversed/discrepancy_reversed[usable]
    return M_tilde, M_rev, M_tilde*M_rev


def control(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """Byte-identical copy of ``current``. The noise floor -- see microbench.py."""
    n_peaks = q2_obs.shape[0]
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)

    q_max = q2_calc[:, -1]
    q_min = np.take_along_axis(
        q2_ref_calc,
        np.argmin(np.abs(q2_ref_calc - q2_obs[0]), axis=1)[:, np.newaxis],
        axis=1,
    )[:, 0]

    in_range = (q2_ref_calc >= q_min[:, np.newaxis]) & (q2_ref_calc <= q_max[:, np.newaxis])
    row_weights = np.ones(q2_ref_calc.shape[1]) if weights is None else weights
    n_cal = (in_range*row_weights[np.newaxis, :]).sum(axis=1)
    q_n = np.max(np.where(in_range, q2_ref_calc, -np.inf), axis=1)

    nearest = np.min(np.abs(q2_ref_calc[:, :, np.newaxis] - q2_obs[np.newaxis, np.newaxis]), axis=2)
    reversed_sum = (np.where(in_range, nearest, 0.0)*row_weights[np.newaxis, :]).sum(axis=1)

    good = (n_cal > 0) & np.isfinite(q_n) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    M_tilde = np.zeros(q2_calc.shape[0])
    M_rev = np.zeros(q2_calc.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = (q_n - q_min)/(2*np.where(n_cal > 0, n_cal, 1))
        M_tilde[good] = epsilon[good]/discrepancy[good]
        discrepancy_reversed = reversed_sum/np.where(n_cal > 0, n_cal, 1)
        epsilon_reversed = (q2_obs[-1] - q2_obs[0])/(2*n_peaks)
        usable = good & (discrepancy_reversed > 0)
        M_rev[usable] = epsilon_reversed/discrepancy_reversed[usable]
    return M_tilde, M_rev, M_tilde*M_rev


def _nearest_scan(values, q2_obs):
    """min_j |values - q2_obs[j]|, by the same full scan the shipped version does."""
    return np.min(np.abs(values[:, np.newaxis] - q2_obs[np.newaxis, :]), axis=1)


def _nearest_search(values, q2_obs_sorted):
    """min_j |values - q2_obs[j]|, by binary search in the sorted peak list.

    |v - p| is a V in p over a sorted peak list, so the minimum is attained at one
    of the two peaks bracketing v. Both candidate distances are formed by the same
    subtract-and-abs as the full scan, so the value returned is the same float, not
    merely the same number.
    """
    position = np.searchsorted(q2_obs_sorted, values)
    upper = np.minimum(position, q2_obs_sorted.size - 1)
    lower = np.maximum(position - 1, 0)
    return np.minimum(np.abs(values - q2_obs_sorted[lower]),
                      np.abs(values - q2_obs_sorted[upper]))


def _rev_sym_masked(q2_obs, q2_calc, q2_ref_calc, weights, nearest_of):
    """Shared body of the candidate variants; ``nearest_of`` computes the reversed term."""
    n_peaks = q2_obs.shape[0]
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)

    q_max = q2_calc[:, -1]
    deviation = np.subtract(q2_ref_calc, q2_obs[0])
    np.abs(deviation, out=deviation)
    q_min = np.take_along_axis(
        q2_ref_calc, np.argmin(deviation, axis=1)[:, np.newaxis], axis=1)[:, 0]
    del deviation

    in_range = q2_ref_calc >= q_min[:, np.newaxis]
    in_range &= q2_ref_calc <= q_max[:, np.newaxis]

    counts = in_range.sum(axis=1)
    n_cal = (counts.astype(float) if weights is None
             else (in_range*weights[np.newaxis, :]).sum(axis=1))
    q_n = np.max(np.where(in_range, q2_ref_calc, -np.inf), axis=1)

    scored = np.zeros_like(q2_ref_calc, dtype=np.result_type(q2_ref_calc, q2_obs, 0.0))
    selected = q2_ref_calc[in_range]
    scored[in_range] = nearest_of(selected)
    reversed_sum = (scored.sum(axis=1) if weights is None
                    else (scored*weights[np.newaxis, :]).sum(axis=1))

    good = (n_cal > 0) & np.isfinite(q_n) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    M_tilde = np.zeros(q2_calc.shape[0])
    M_rev = np.zeros(q2_calc.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = (q_n - q_min)/(2*np.where(n_cal > 0, n_cal, 1))
        M_tilde[good] = epsilon[good]/discrepancy[good]
        discrepancy_reversed = reversed_sum/np.where(n_cal > 0, n_cal, 1)
        epsilon_reversed = (q2_obs[-1] - q2_obs[0])/(2*n_peaks)
        usable = good & (discrepancy_reversed > 0)
        M_rev[usable] = epsilon_reversed/discrepancy_reversed[usable]
    return M_tilde, M_rev, M_tilde*M_rev


def masked_scan(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """Score only the in-range reference lines; still a full scan over the peaks."""
    return _rev_sym_masked(q2_obs, q2_calc, q2_ref_calc, weights,
                           lambda values: _nearest_scan(values, q2_obs))


def masked_search(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """Score only the in-range reference lines, by binary search in the sorted peaks."""
    q2_obs_sorted = np.sort(q2_obs)
    return _rev_sym_masked(q2_obs, q2_calc, q2_ref_calc, weights,
                           lambda values: _nearest_search(values, q2_obs_sorted))


def full_search(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """Binary search, but over every reference line -- isolates the masking's own effect."""
    q2_obs_sorted = np.sort(q2_obs)
    n_peaks = q2_obs.shape[0]
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    q_max = q2_calc[:, -1]
    q_min = np.take_along_axis(
        q2_ref_calc,
        np.argmin(np.abs(q2_ref_calc - q2_obs[0]), axis=1)[:, np.newaxis],
        axis=1,
    )[:, 0]
    in_range = (q2_ref_calc >= q_min[:, np.newaxis]) & (q2_ref_calc <= q_max[:, np.newaxis])
    row_weights = np.ones(q2_ref_calc.shape[1]) if weights is None else weights
    n_cal = (in_range*row_weights[np.newaxis, :]).sum(axis=1)
    q_n = np.max(np.where(in_range, q2_ref_calc, -np.inf), axis=1)
    # Written through an ``empty_like`` so the result keeps the reference array's memory
    # order: ``np.where`` follows its inputs' layout, and ``.sum(axis=1)`` over an F-ordered
    # array does not group its additions the way it does over a C-ordered one.
    nearest = np.empty_like(q2_ref_calc, dtype=np.result_type(q2_ref_calc, q2_obs, 0.0))
    nearest[...] = _nearest_search(
        np.asarray(q2_ref_calc).reshape(-1), q2_obs_sorted).reshape(q2_ref_calc.shape)
    reversed_sum = (np.where(in_range, nearest, 0.0)*row_weights[np.newaxis, :]).sum(axis=1)
    good = (n_cal > 0) & np.isfinite(q_n) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    M_tilde = np.zeros(q2_calc.shape[0])
    M_rev = np.zeros(q2_calc.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = (q_n - q_min)/(2*np.where(n_cal > 0, n_cal, 1))
        M_tilde[good] = epsilon[good]/discrepancy[good]
        discrepancy_reversed = reversed_sum/np.where(n_cal > 0, n_cal, 1)
        epsilon_reversed = (q2_obs[-1] - q2_obs[0])/(2*n_peaks)
        usable = good & (discrepancy_reversed > 0)
        M_rev[usable] = epsilon_reversed/discrepancy_reversed[usable]
    return M_tilde, M_rev, M_tilde*M_rev


@jit
def _reversed_scores_kernel(q2_ref_calc, q_max, q2_obs_sorted, scored, in_range, q_n, counts,
                            q_min, first_peak):
    """One row-local pass: q_min, the in-range mask, N_cal, q_N and the reversed scores.

    Everything here is per element, so no float addition is reassociated and every value
    written is the same float the shipped expression produces. The row is read twice but
    is 8 KB, so the second read is out of L1 rather than out of memory.

    No fastmath, for the reason numba_functions.py gives: it licenses the compiler to
    assume NaN and Inf never occur, and q2_ref_calc is xnn @ hkl2.T, where NaN is common.
    """
    n_candidates, n_ref = q2_ref_calc.shape
    n_peaks = q2_obs_sorted.size
    for candidate in range(n_candidates):
        # np.argmin returns the first occurrence of the minimum, so the update is a
        # strict '<' and the scan runs forwards.
        best_index = 0
        best_deviation = np.inf
        for reference in range(n_ref):
            deviation = abs(q2_ref_calc[candidate, reference] - first_peak)
            if deviation < best_deviation:
                best_deviation = deviation
                best_index = reference
        lower_bound = q2_ref_calc[candidate, best_index]
        q_min[candidate] = lower_bound
        upper_bound = q_max[candidate]

        count = 0
        highest = -np.inf
        for reference in range(n_ref):
            value = q2_ref_calc[candidate, reference]
            # NaN fails both comparisons, exactly as it does in the array expression.
            if value >= lower_bound and value <= upper_bound:
                in_range[candidate, reference] = True
                count += 1
                if value > highest:
                    highest = value
                # Branchless lower_bound: the trip count depends only on n_peaks, so
                # there is no data-dependent branch for the predictor to miss. The
                # textbook `while left < right` form mispredicts about half its
                # compares and measured 65.8 ms against this one's 41.2 ms on the
                # 10 000-candidate mP capture.
                left = 0
                width = n_peaks
                while width > 1:
                    half = width >> 1
                    left += half*(q2_obs_sorted[left + half - 1] < value)
                    width -= half
                left += q2_obs_sorted[left] < value
                below = left - 1
                if below < 0:
                    below = 0
                above = left
                if above > n_peaks - 1:
                    above = n_peaks - 1
                distance_below = abs(value - q2_obs_sorted[below])
                distance_above = abs(value - q2_obs_sorted[above])
                scored[candidate, reference] = (distance_above if distance_above < distance_below
                                                else distance_below)
            else:
                in_range[candidate, reference] = False
                scored[candidate, reference] = 0.0
        counts[candidate] = count
        q_n[candidate] = highest


def kernel(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """The masked search, fused into a single numba pass over the reference array."""
    n_peaks = q2_obs.shape[0]
    n_candidates, n_ref = q2_ref_calc.shape
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    q_max = q2_calc[:, -1]

    dtype = np.result_type(q2_ref_calc, q2_obs, 0.0)
    # Both keep the reference array's memory order: `.sum(axis=1)` groups its additions
    # differently over an F-ordered array, and the shipped expressions build these two
    # arrays from q2_ref_calc, so they inherit its order.
    scored = np.empty_like(q2_ref_calc, dtype=dtype)
    in_range = np.empty_like(q2_ref_calc, dtype=bool)
    q_n = np.empty(n_candidates, dtype=dtype)
    q_min = np.empty(n_candidates, dtype=q2_ref_calc.dtype)
    counts = np.empty(n_candidates, dtype=np.int64)
    _reversed_scores_kernel(q2_ref_calc, np.ascontiguousarray(q_max), np.sort(q2_obs),
                            scored, in_range, q_n, counts, q_min, q2_obs[0])

    n_cal = (counts.astype(float) if weights is None
             else (in_range*weights[np.newaxis, :]).sum(axis=1))
    reversed_sum = (scored.sum(axis=1) if weights is None
                    else (scored*weights[np.newaxis, :]).sum(axis=1))

    good = (n_cal > 0) & np.isfinite(q_n) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    M_tilde = np.zeros(q2_calc.shape[0])
    M_rev = np.zeros(q2_calc.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = (q_n - q_min)/(2*np.where(n_cal > 0, n_cal, 1))
        M_tilde[good] = epsilon[good]/discrepancy[good]
        discrepancy_reversed = reversed_sum/np.where(n_cal > 0, n_cal, 1)
        epsilon_reversed = (q2_obs[-1] - q2_obs[0])/(2*n_peaks)
        usable = good & (discrepancy_reversed > 0)
        M_rev[usable] = epsilon_reversed/discrepancy_reversed[usable]
    return M_tilde, M_rev, M_tilde*M_rev


def shipped(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """What FigureOfMerits.get_M_rev_sym actually does today."""
    return get_M_rev_sym(q2_obs, q2_calc, q2_ref_calc, weights)


VARIANTS = {
    'current': current,
    'control': control,
    'full_search': full_search,
    'masked_scan': masked_scan,
    'masked_search': masked_search,
    'kernel': kernel,
    'shipped': shipped,
    }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture(args):
    """Record the real (q2_obs, q2_calc, q2_ref_calc) get_M_rev_sym would be handed.

    The capture point is `Candidates.get_M20` as called from `assign_hkls`: the same place
    in the loop where the merit would be evaluated, with the optimiser's own arrays. See
    `capture_hook.py` for why it is not `_retention_fom_values`, which is on `fom` only.
    """
    from mlindex.command_line import run as run_module

    os.makedirs(args.out_dir, exist_ok=True)
    run_args = argparse.Namespace(
        peaks=args.peaks, peak_file=args.peak_file, peak_units=args.peak_units,
        wavelength=args.wavelength, zero_error=False,
        output_file=os.path.join(args.out_dir, 'capture_results.json'),
        bravais_lattices=[args.bravais_lattice], seed=args.seed, mpi=False, nproc=1,
        )
    with capture_get_M20(args.max_rows) as record:
        peak_list = run_module._load_peaks(run_args)
        run_module._run_mp(run_args, peak_list, n_procs=1, seed=args.seed)

    if record.arrays is None:
        raise SystemExit(f'no candidates were scored for {args.bravais_lattice}; nothing captured')
    shapes = record.shapes
    best = {'arrays': record.arrays[:3]}
    candidate_rows = np.array([shape[0] for shape in shapes])
    print(f'\n{len(shapes)} invocations, {candidate_rows.sum()} candidate rows total')
    print(f'   candidates per call: median {int(np.median(candidate_rows))}, '
          f'max {int(candidate_rows.max())}')
    print(f'   n_ref {shapes[0][1]}, n_peaks {shapes[0][2]}')

    q2_obs, q2_calc, q2_ref_calc = best['arrays']
    out_path = os.path.join(args.out_dir, f'msym_{args.bravais_lattice}.npz')
    np.savez(out_path, q2_obs=q2_obs, q2_calc=q2_calc, q2_ref_calc=q2_ref_calc,
             n_calls=len(shapes), total_rows=candidate_rows.sum())
    print(f'wrote {out_path}: q2_calc {q2_calc.shape}, q2_ref_calc {q2_ref_calc.shape}')


def load_capture(path, n_candidates):
    """The capture, resized to exactly ``n_candidates`` rows.

    Rows are tiled when the capture is short of the requested size. A tiled row is a
    real row -- same reference list, same assignment, same in-range fraction -- so the
    work per row is the work the indexer really does; only the number of them is set
    by the benchmark rather than by the run.
    """
    with np.load(path) as handle:
        q2_obs = handle['q2_obs']
        q2_calc = handle['q2_calc']
        q2_ref_calc = handle['q2_ref_calc']
    available = q2_calc.shape[0]
    index = np.arange(n_candidates) % available
    return q2_obs, np.ascontiguousarray(q2_calc[index]), np.ascontiguousarray(q2_ref_calc[index])


# ---------------------------------------------------------------------------
# Line profile
# ---------------------------------------------------------------------------

def profile(args):
    from line_profiler import LineProfiler

    for n_candidates in args.sizes:
        q2_obs, q2_calc, q2_ref_calc = load_capture(args.capture_file, n_candidates)
        for name in args.variants:
            function = VARIANTS[name]
            # The candidate variants delegate their body, so the helpers have to be
            # registered too or the report is one line reading "99.9%".
            profiler = LineProfiler(function, _rev_sym_masked, _nearest_search, _nearest_scan)
            profiler.enable_by_count()
            for _ in range(args.repeats):
                function(q2_obs, q2_calc, q2_ref_calc)
            profiler.disable_by_count()
            print('=' * 100)
            print(f'{name}: {n_candidates} candidates x {q2_ref_calc.shape[1]} reference '
                  f'lines x {q2_obs.size} peaks, {args.repeats} repeats')
            print('=' * 100)
            profiler.print_stats(summarize=True)


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

def bench(args):
    compare = tuple_match([exact_match, exact_match, exact_match])
    for n_candidates in args.sizes:
        q2_obs, q2_calc, q2_ref_calc = load_capture(args.capture_file, n_candidates)
        names = args.variants
        variants = {name: (lambda function=VARIANTS[name]:
                           function(q2_obs, q2_calc, q2_ref_calc))
                    for name in names}
        work = (f'{n_candidates} candidates x {q2_ref_calc.shape[1]} reference lines x '
                f'{q2_obs.size} peaks; peak intermediate in `current` '
                f'{8*n_candidates*q2_ref_calc.shape[1]*q2_obs.size/1e9:.2f} GB')
        benchmark(f'get_M_rev_sym -- {os.path.basename(args.capture_file)}, '
                  f'{n_candidates} candidates',
                  variants, reference='current', control='control',
                  rounds=args.rounds, compare=compare, work=work)


# ---------------------------------------------------------------------------
# Cost against get_M20
# ---------------------------------------------------------------------------

def cost(args):
    """The same ratio S06_zoo_cost.csv reports: seconds per candidate against get_M20.

    That table prices M_sym at 24.3 get_M20-equivalents, and FomCombiner's
    `affordable_features` inherits the 2x budget it fails. Both merits are timed here on
    one capture, on one machine, in one harness, so the ratio is comparable within this
    run even though the absolute seconds are not comparable to that table's.
    """
    for n_candidates in args.sizes:
        q2_obs, q2_calc, q2_ref_calc = load_capture(args.capture_file, n_candidates)
        # get_M20 zeroes the out-of-range entries of q2_ref_calc in place, so every
        # repeat needs its own copy and the copies are made outside the timed region.
        copies = [q2_ref_calc.copy() for _ in range(4*args.rounds + 8)]
        counter = {'index': 0}

        def call_m20():
            counter['index'] += 1
            return get_M20(q2_obs, q2_calc, copies[counter['index'] - 1])

        variants = {
            'get_M20': call_m20,
            'get_M20_control': call_m20,
            'get_M_rev_sym (before)': lambda: current(q2_obs, q2_calc, q2_ref_calc),
            'get_M_rev_sym (now)': lambda: shipped(q2_obs, q2_calc, q2_ref_calc),
            }
        results = benchmark(
            f'cost against get_M20 -- {os.path.basename(args.capture_file)}, '
            f'{n_candidates} candidates',
            variants, reference='get_M20', control='get_M20_control',
            rounds=args.rounds, compare=lambda reference, candidate: (True, 'not compared'),
            work=f'{n_candidates} candidates x {q2_ref_calc.shape[1]} reference lines x '
                 f'{q2_obs.size} peaks')
        baseline = results['get_M20']['median']
        print()
        for name in ('get_M_rev_sym (before)', 'get_M_rev_sym (now)'):
            print(f'   {name:<24} {results[name]["median"]/baseline:6.2f} get_M20-equivalents')


# ---------------------------------------------------------------------------
# Fuzz
# ---------------------------------------------------------------------------

def _fuzz_cases(rng, n_cases):
    """Randomised inputs, deliberately including what a capture cannot contain."""
    for case in range(n_cases):
        n_candidates = int(rng.integers(1, 40))
        n_peaks = int(rng.integers(2, 21))
        n_ref = int(rng.integers(1, 60))
        q2_obs = np.sort(rng.uniform(0.01, 2.0, n_peaks))
        q2_ref_calc = rng.uniform(0.0, 3.0, (n_candidates, n_ref))
        q2_calc = np.sort(rng.uniform(0.01, 2.0, (n_candidates, n_peaks)), axis=1)
        weights = None if case % 3 else rng.uniform(0.2, 1.0, n_ref)
        label = f'random {n_candidates}x{n_ref}x{n_peaks}, weights={weights is not None}'
        if case % 7 == 0:
            # No reference line in range: q_max below every reference line.
            q2_ref_calc = q2_ref_calc + 10.0
            label += ', empty in-range rows'
        if case % 5 == 0:
            # Ties in the nearest-peak search, and repeated reference values.
            q2_ref_calc[:, ::2] = q2_obs[0]
            q2_ref_calc[:, 1::3] = 0.5*(q2_obs[0] + q2_obs[-1])
            label += ', ties'
        if case % 11 == 0:
            q2_calc[:] = 0.0
            label += ', all-zero q2_calc'
        if case % 13 == 0:
            q2_obs = rng.permutation(q2_obs)
            label += ', unsorted peaks'
        if case % 17 == 0:
            q2_ref_calc = np.asfortranarray(q2_ref_calc)
            label += ', F-ordered reference'
        if case % 19 == 0:
            # No sorted order to binary-search, so `shipped` must take its fallback.
            q2_obs = q2_obs.copy()
            q2_obs[len(q2_obs)//2] = np.nan
            label += ', NaN peak (fallback path)'
        if case % 23 == 0:
            q2_ref_calc = np.round(q2_ref_calc*4).astype(np.int64)
            q2_calc = np.round(q2_calc*4).astype(np.int64)
            label += ', integer input (fallback path)'
        yield label, q2_obs, q2_calc, q2_ref_calc, weights


def fuzz(args):
    rng = np.random.default_rng(args.seed)
    failures = 0
    for label, q2_obs, q2_calc, q2_ref_calc, weights in _fuzz_cases(rng, args.cases):
        expected = current(q2_obs, q2_calc, q2_ref_calc, weights)
        for name in args.variants:
            if name in ('current', 'control'):
                continue
            got = VARIANTS[name](q2_obs, q2_calc, q2_ref_calc, weights)
            ok, detail = tuple_match([exact_match]*3)(expected, got)
            if not ok:
                failures += 1
                print(f'MISMATCH {name}: {label}\n   {detail}')
    print(f'{args.cases} cases x {len([n for n in args.variants if n not in ("current", "control")])} '
          f'variants: {failures} mismatches')
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)

    cap = sub.add_parser('capture')
    cap.add_argument('--peaks', type=str, default=None)
    cap.add_argument('--peak-file', type=str, default=None)
    cap.add_argument('--peak-units', type=str, default='q2',
                     choices=['d', 'q', 'q2', '2theta'])
    cap.add_argument('--wavelength', type=float, default=None)
    cap.add_argument('--bravais-lattice', type=str, default='mP')
    cap.add_argument('--seed', type=int, default=12345)
    cap.add_argument('--max-rows', type=int, default=20000)
    cap.add_argument('--out-dir', type=str, default='captures')
    cap.set_defaults(func=capture)

    prof = sub.add_parser('profile')
    prof.add_argument('--capture-file', type=str, required=True)
    prof.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    prof.add_argument('--repeats', type=int, default=3)
    prof.add_argument('--variants', type=str, nargs='+', default=['current'],
                      choices=list(VARIANTS))
    prof.set_defaults(func=profile)

    run = sub.add_parser('bench')
    run.add_argument('--capture-file', type=str, required=True)
    run.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    run.add_argument('--rounds', type=int, default=15)
    run.add_argument('--variants', type=str, nargs='+', default=list(VARIANTS),
                     choices=list(VARIANTS))
    run.set_defaults(func=bench)

    cst = sub.add_parser('cost')
    cst.add_argument('--capture-file', type=str, required=True)
    cst.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    cst.add_argument('--rounds', type=int, default=15)
    cst.set_defaults(func=cost)

    fzz = sub.add_parser('fuzz')
    fzz.add_argument('--cases', type=int, default=300)
    fzz.add_argument('--seed', type=int, default=3)
    fzz.add_argument('--variants', type=str, nargs='+', default=list(VARIANTS),
                     choices=list(VARIANTS))
    fzz.set_defaults(func=fuzz)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
