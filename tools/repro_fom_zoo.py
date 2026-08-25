#!/usr/bin/env python
"""Reproducer: what the rest of the figure-of-merit zoo costs, and why.

DEVELOPMENT TOOL -- not part of the installed package.

`repro_msym.py` did `get_M_rev_sym`. This does the same job for every other merit
`FigureOfMerits.compute_all` evaluates -- the set `run_fom_zoo_eval.py` priced into
`S06_zoo_cost.csv` and the S08 combiner is built on.

The point of the `cost` subcommand is to pick targets by measurement rather than by
reading the code and guessing. It prices each merit on real captured arguments at
several candidate counts and reports the ratio to `get_M20`, which is the unit
`S06_zoo_cost.csv` and `FomCombiner.affordable_features` are written in.

Usage
-----
Capture everything compute_all needs for a lattice (writes <out>/zoo_<BL>.npz)::

    python tools/repro_fom_zoo.py capture --peak-file 11bmb_3844_peak_list.npy \\
        --peak-units q2 --bravais-lattice mP

Price every merit at 1 000 and 10 000 candidates::

    python tools/repro_fom_zoo.py cost --capture-file captures/zoo_mP.npz

Line-profile the expensive ones::

    python tools/repro_fom_zoo.py profile --capture-file captures/zoo_mP.npz \\
        --merits n_over M_wu M_1

Benchmark the rewrites against the originals, and check bit-identity on randomised
inputs including the edge cases a capture cannot contain::

    python tools/repro_fom_zoo.py bench --capture-file captures/zoo_mP.npz
    python tools/repro_fom_zoo.py fuzz
"""

import argparse
import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_hook import capture_get_M20
from microbench import benchmark, exact_match, tuple_match

from mlindex.utilities import FigureOfMerits as fom

# A capture, resized. `hkl` and `hkl_ref` are only in the benchmark-pool captures, which are
# the ones the cross-validated family can be priced on at all.
Case = namedtuple('Case', 'q2_obs q2_calc q2_ref_calc xnn hkl hkl_ref '
                          'lattice_system bravais_lattice')


# ---------------------------------------------------------------------------
# The merits, as called on the arrays a capture holds
# ---------------------------------------------------------------------------
# One entry per merit compute_all evaluates. `get_M20` is handed a copy because it
# zeroes q2_ref_calc in place. Merits that come out of one call share that call here,
# exactly as compute_all shares it, so the table prices calls and not columns.

def merit_calls(case, fom=fom):
    q2_obs, q2_calc, q2_ref_calc = case.q2_obs, case.q2_calc, case.q2_ref_calc
    xnn, lattice_system, bravais_lattice = case.xnn, case.lattice_system, case.bravais_lattice
    volume = np.ones(q2_calc.shape[0])
    calls = {
        'M20': lambda: fom.get_M20(q2_obs, q2_calc, q2_ref_calc.copy()),
        'Minfo': lambda: fom.get_M20_likelihood(q2_obs, q2_calc, bravais_lattice, volume),
        'M_tilde/M_rev/M_sym': lambda: fom.get_M_rev_sym(q2_obs, q2_calc, q2_ref_calc),
        'X_N': lambda: fom.get_X_N(q2_obs, q2_calc, q2_ref_calc),
        'M_wu': lambda: fom.get_M_wu(q2_obs, q2_calc, q2_ref_calc),
        'M_star': lambda: fom.get_M_star(q2_obs, q2_calc, volume, lattice_system),
        'M_star_corrected': lambda: fom.get_M_star(q2_obs, q2_calc, volume, lattice_system,
                                                   corrected=True),
        'M_1': lambda: fom.get_M_1(q2_obs, q2_calc, q2_ref_calc),
        'M_nn': lambda: fom.get_M_nn(q2_obs, q2_calc, q2_ref_calc),
        'M_info_clipped': lambda: fom.get_M_info_clipped(q2_obs, q2_calc, xnn, lattice_system,
                                                         bravais_lattice),
        'nll_exponential': lambda: fom.get_nll_exponential(q2_obs, q2_calc, xnn, lattice_system,
                                                           bravais_lattice),
        'null_tail_nll': lambda: fom.get_null_tail_nll(q2_obs, q2_calc, xnn, lattice_system,
                                                       bravais_lattice),
        'bic': lambda: fom.get_bic(q2_obs, q2_calc, xnn, lattice_system, bravais_lattice),
        'n_over/max_gap': lambda: fom.get_n_over(q2_obs, q2_calc, q2_ref_calc),
        'zone_dominance': lambda: fom.get_zone_dominance(xnn, lattice_system),
        'N_cal': lambda: fom.get_N_cal(q2_ref_calc, np.zeros(q2_calc.shape[0]), q2_calc[:, -1]),
        'delta_dewolff61': lambda: fom.get_delta_dewolff61(q2_obs, xnn, lattice_system,
                                                           bravais_lattice),
        'n_dewolff61': lambda: fom.get_n_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice),
        'F_N_q': lambda: fom.get_F_N(q2_obs, q2_calc, q2_ref_calc),
        'chi2_taupin': lambda: fom.get_chi2(q2_obs, q2_calc, lattice_system, variant='taupin'),
        'chi2_fixed': lambda: fom.get_chi2(q2_obs, q2_calc, lattice_system, variant='fixed'),
        'chi2_entrywise': lambda: fom.get_chi2(q2_obs, q2_calc, lattice_system, sigma=1e-3,
                                               variant='entrywise'),
        'compute_all': lambda: fom.compute_all(q2_obs, q2_calc, q2_ref_calc, xnn, lattice_system,
                                               bravais_lattice),
        # S11 block B. `get_assignment_posterior` is given the sigma call's own outputs, which
        # is how the caller that wants both pays for the nearest-line scan once.
        'assignment_sigma': lambda: fom.get_assignment_sigma(q2_obs, q2_ref_calc, lattice_system),
        'assignment_posterior': lambda: fom.get_assignment_posterior(
            q2_obs, q2_ref_calc, lattice_system),
        }
    if case.hkl_ref is not None:
        # S10. Only priceable on a benchmark-pool capture, which is the only one carrying the
        # reference lines the hold-out score needs.
        #
        # `insample_fom` and `cv_fom__random` are NOT here: the cross-validated and in-sample
        # families lost in campaign 1 and are not ported to this branch (CHERRY_PICK.md), so
        # the functions do not exist to price.
        calls['holdout_fom'] = lambda: fom.get_holdout_fom(
            q2_obs[-5:] + 0.01, xnn, case.hkl_ref, lattice_system, bravais_lattice)
    return calls


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture(args):
    """Record everything compute_all needs, from a real run, at the real sizes.

    The capture point is `Candidates.get_M20` as called from `assign_hkls`; see
    `capture_hook.py` for why, and for the two traps it exists to avoid.
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
    q2_obs, q2_calc, q2_ref_calc, xnn = record.arrays
    lattice_system, bravais_lattice = record.lattice_system, record.bravais_lattice
    out_path = os.path.join(args.out_dir, f'zoo_{args.bravais_lattice}.npz')
    np.savez(out_path, q2_obs=q2_obs, q2_calc=q2_calc, q2_ref_calc=q2_ref_calc, xnn=xnn,
             lattice_system=lattice_system, bravais_lattice=bravais_lattice)
    cutoff = q2_calc[:, -1]
    fraction = (q2_ref_calc < cutoff[:, np.newaxis]).sum(axis=1).mean()/q2_ref_calc.shape[1]
    print(f'\n{len(record.shapes)} invocations, max {record.rows} candidates')
    print(f'wrote {out_path}: q2_calc {q2_calc.shape}, q2_ref_calc {q2_ref_calc.shape}, '
          f'xnn {xnn.shape}, {lattice_system}/{bravais_lattice}')
    print(f'   mean fraction of reference lines below the cut-off: {fraction:.3f}')


def capture_benchmark(args):
    """Capture from the frozen benchmark pool -- the candidates the analysis actually scores.

    `capture` records what the *indexer's inner loop* holds, which is mid-refinement. The S06
    feature matrix and the S08 combiner are built on the dumped pool instead: converged,
    pruned candidates, with the reference list rebuilt per extinction group. The two differ in
    the one property that decides what these merits cost -- the fraction of reference lines
    below the cut-off -- so both are worth pricing, and this is the one the question is about.

    Follows `run_fom_zoo_eval.measure_cost` exactly, so the capture is the same construction
    the cost column in `S06_zoo_cost.csv` was measured on.
    """
    import pandas as pd
    from mlindex.model_training import FomBenchmark

    os.makedirs(args.out_dir, exist_ok=True)
    candidates = FomBenchmark.load_candidates(
        args.benchmark_dir, bundles=[args.bundle],
        columns=list(FomBenchmark.ZOO_CANDIDATE_COLUMNS),
        )
    entries = FomBenchmark.load_entries(args.benchmark_dir)
    entries = entries.loc[entries['condition_bundle'] == args.bundle]
    peaks = entries.set_index('entry_id')['q2_obs']

    collected = []
    for entry_id in pd.unique(candidates['entry_id']):
        group = candidates.loc[(candidates['entry_id'] == entry_id)
                               & (candidates['bravais_lattice'] == args.bravais_lattice)]
        if not group.shape[0]:
            continue
        for spacegroup, chunk in group.groupby('spacegroup', sort=False):
            q2_obs = np.asarray(peaks.loc[entry_id],
                                dtype=np.float64)[:int(chunk['n_peaks'].iloc[0])]
            xnn = np.vstack([np.asarray(v, dtype=np.float64) for v in chunk['xnn']])
            q2_ref_calc, _, hkl, q2_calc = FomBenchmark.assign_lines(
                q2_obs, xnn, args.lattice_system, args.bravais_lattice, spacegroup,
                )
            hkl_ref = FomBenchmark.hkl_ref_for(
                args.lattice_system, args.bravais_lattice, spacegroup)
            collected.append((q2_obs, q2_calc, q2_ref_calc, xnn, hkl, hkl_ref))
        rows = sum(case[1].shape[0] for case in collected)
        if rows >= args.max_rows:
            break

    # One extinction group per capture: the merits are timed on one (q2_obs, q2_calc,
    # q2_ref_calc) triple, and reference lists differ in length between groups.
    (q2_obs, q2_calc, q2_ref_calc, xnn, hkl,
     hkl_ref) = max(collected, key=lambda case: case[1].shape[0])
    out_path = os.path.join(args.out_dir, f'pool_{args.bravais_lattice}.npz')
    np.savez(out_path, q2_obs=q2_obs, q2_calc=q2_calc, q2_ref_calc=q2_ref_calc, xnn=xnn,
             hkl=hkl, hkl_ref=hkl_ref,
             lattice_system=args.lattice_system, bravais_lattice=args.bravais_lattice)
    cutoff = q2_calc[:, -1]
    fraction = (q2_ref_calc < cutoff[:, np.newaxis]).sum(axis=1).mean()/q2_ref_calc.shape[1]
    print(f'\n{len(collected)} extinction groups seen; kept the largest')
    print(f'wrote {out_path}: q2_calc {q2_calc.shape}, q2_ref_calc {q2_ref_calc.shape}')
    print(f'   mean fraction of reference lines below the cut-off: {fraction:.3f}')


def load_capture(path, n_candidates):
    """The capture resized to exactly ``n_candidates`` rows, tiling if it is short.

    A tiled row is a real row -- same reference list, same assignment, same in-range
    fraction -- so the work per row is the work the indexer really does; only how many
    of them there are is set by the benchmark.
    """
    with np.load(path) as handle:
        q2_obs = handle['q2_obs']
        q2_calc = handle['q2_calc']
        q2_ref_calc = handle['q2_ref_calc']
        xnn = handle['xnn']
        hkl = handle['hkl'] if 'hkl' in handle else None
        hkl_ref = handle['hkl_ref'] if 'hkl_ref' in handle else None
        lattice_system = str(handle['lattice_system'])
        bravais_lattice = str(handle['bravais_lattice'])
    index = np.arange(n_candidates) % q2_calc.shape[0]
    return Case(q2_obs, np.ascontiguousarray(q2_calc[index]),
                np.ascontiguousarray(q2_ref_calc[index]), np.ascontiguousarray(xnn[index]),
                None if hkl is None else np.ascontiguousarray(hkl[index]), hkl_ref,
                lattice_system, bravais_lattice)


# ---------------------------------------------------------------------------
# Cost survey
# ---------------------------------------------------------------------------

def _time(call, repeats):
    call()
    timings = []
    for _ in range(repeats):
        import time
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


# The provenance a price is meaningless without. Campaign 1's cost table recorded none of
# it, which is a large part of why it could not be defended (C2-F-001), so `--csv` carries
# the machine, the library versions, the capture and the revision beside every number.
def _environment():
    import platform
    import numba
    import scipy
    return (f'{platform.processor() or platform.machine()} ({platform.machine()}), '
            f'Python {platform.python_version()}, numpy {np.__version__}, '
            f'numba {numba.__version__}, scipy {scipy.__version__}')


CSV_COLUMNS = ('merit', 'seconds_per_candidate', 'ms_per_call', 'cost_vs_M20', 'regime',
               'lattice', 'capture', 'capture_origin', 'n_candidates', 'n_ref_lines',
               'n_peaks', 'in_range_fraction', 'revision', 'hardware', 'date')


def cost(args):
    import csv as csv_module
    import datetime

    module = _baseline_module(args.revision) if args.revision else fom
    csv_rows = []
    for n_candidates in args.sizes:
        case = load_capture(args.capture_file, n_candidates)
        calls = merit_calls(case, module)
        rows = []
        for name, call in calls.items():
            rows.append((name, _time(call, args.repeats)))
        baseline = dict(rows)['M20']
        rows.sort(key=lambda row: -row[1])
        cutoff = case.q2_calc[:, -1]
        fraction = float((case.q2_ref_calc < cutoff[:, np.newaxis]).sum(axis=1).mean()
                         / case.q2_ref_calc.shape[1])
        print('=' * 78)
        print(f'{os.path.basename(args.capture_file)}: {n_candidates} candidates x '
              f'{case.q2_ref_calc.shape[1]} reference lines x {case.q2_obs.size} peaks'
              + (f'   [module as of {args.revision}]' if args.revision else ''))
        print(f'   {fraction:.1%} of reference lines below the cut-off')
        print('=' * 78)
        print(f'{"merit":<24} {"ms/call":>10} {"vs get_M20":>12}')
        for name, seconds in rows:
            print(f'{name:<24} {seconds*1e3:10.3f} {seconds/baseline:11.2f}x')
        if args.csv:
            for name, seconds in rows:
                csv_rows.append({
                    'merit': name,
                    'seconds_per_candidate': seconds/n_candidates,
                    'ms_per_call': seconds*1e3,
                    'cost_vs_M20': seconds/baseline,
                    'regime': args.regime or '',
                    'lattice': args.lattice or str(case.bravais_lattice),
                    'capture': os.path.basename(args.capture_file),
                    'capture_origin': args.capture_origin or '',
                    'n_candidates': n_candidates,
                    'n_ref_lines': case.q2_ref_calc.shape[1],
                    'n_peaks': case.q2_obs.size,
                    'in_range_fraction': round(fraction, 6),
                    'revision': args.revision or 'working tree',
                    'hardware': _environment(),
                    'date': datetime.date.today().isoformat(),
                    })

    if args.csv:
        exists = os.path.exists(args.csv)
        directory = os.path.dirname(args.csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # newline='' because csv writes its own line terminator; without it Windows
        # produces \r\r\n and a blank row between every record.
        with open(args.csv, 'a' if exists else 'w', newline='', encoding='utf-8') as handle:
            writer = csv_module.DictWriter(handle, fieldnames=CSV_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerows(csv_rows)
        print(f'\n{"appended" if exists else "wrote"} {len(csv_rows)} rows to {args.csv}')


# ---------------------------------------------------------------------------
# Line profile
# ---------------------------------------------------------------------------

# Keyed by the name in `merit_calls`, so `--merits` takes the same names `cost` prints.
PROFILE_TARGETS = {
    'n_over/max_gap': ['get_n_over', '_sorted_lines_in_range'],
    'M_wu': ['get_M_wu', '_sorted_lines_in_range'],
    'M_1': ['get_M_1', '_sorted_lines_in_range'],
    'M_nn': ['get_M_nn'],
    'X_N': ['get_X_N'],
    'F_N_q': ['get_F_N'],
    'N_cal': ['get_N_cal'],
    'M20': ['get_M20'],
    'Minfo': ['get_M20_likelihood'],
    'bic': ['get_bic'],
    'M_info_clipped': ['get_M_info_clipped'],
    'null_tail_nll': ['get_null_tail_nll'],
    'nll_exponential': ['get_nll_exponential'],
    'delta_dewolff61': ['get_delta_dewolff61', 'get_dewolff61_terms', 'get_dewolff61_axes'],
    'n_dewolff61': ['get_n_dewolff61', 'get_dewolff61_terms', 'get_dewolff61_axes'],
    'chi2_fixed': ['get_chi2'],
    'chi2_taupin': ['get_chi2'],
    'chi2_entrywise': ['get_chi2'],
    'zone_dominance': ['get_zone_dominance'],
    'M_star': ['get_M_star'],
    'M_tilde/M_rev/M_sym': ['get_M_rev_sym', '_reversed_line_terms'],
    'compute_all': ['compute_all'],
    'assignment_sigma': ['get_assignment_sigma'],
    'assignment_posterior': ['get_assignment_posterior', 'get_assignment_sigma'],
    'holdout_fom': ['get_holdout_fom', '_predictive_terms', '_reduce_predictive'],
    }


def profile(args):
    from line_profiler import LineProfiler

    for n_candidates in args.sizes:
        case = load_capture(args.capture_file, n_candidates)
        calls = merit_calls(case)
        for name in args.merits:
            functions = [getattr(fom, attribute) for attribute in PROFILE_TARGETS[name]]
            profiler = LineProfiler(*functions)
            profiler.enable_by_count()
            for _ in range(args.repeats):
                calls[name]()
            profiler.disable_by_count()
            print('=' * 90)
            print(f'{name}: {n_candidates} candidates x {case[2].shape[1]} reference lines x '
                  f'{case[0].size} peaks, {args.repeats} repeats')
            print('=' * 90)
            profiler.print_stats(summarize=True)



# ---------------------------------------------------------------------------
# The originals, copied verbatim, so every rewrite has something to be checked against
# ---------------------------------------------------------------------------

def before_M_wu(q2_obs, q2_calc, q2_ref_calc):
    """get_M_wu as it stood before the rewrite."""
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    cutoff = q2_calc[:, -1]
    lines, count = fom._sorted_lines_in_range(q2_ref_calc, cutoff)
    finite = np.isfinite(lines)
    previous = np.concatenate(
        [np.zeros((lines.shape[0], 1)), np.where(finite, lines, 0.0)[:, :-1]], axis=1
    )
    gaps = np.where(finite, np.where(finite, lines, 0.0) - previous, 0.0)
    q_n = np.max(np.where(np.isfinite(lines), lines, 0.0), axis=1)
    good = (count > 0) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    merit = np.zeros(q2_calc.shape[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        g_bar = np.sum(gaps**2/4, axis=1)/np.where(q_n > 0, q_n, 1)
        merit[good] = g_bar[good]/discrepancy[good]
    return merit


def before_M_1(q2_obs, q2_calc, q2_ref_calc):
    """get_M_1 as it stood before the rewrite."""
    cutoff = q2_calc[:, -1]
    lines, count = fom._sorted_lines_in_range(q2_ref_calc, cutoff)
    n_candidates, n_peaks = q2_calc.shape
    upper_index = np.stack(
        [np.searchsorted(lines[row], q2_obs) for row in range(n_candidates)], axis=0
    )
    n_lines = lines.shape[1]
    upper_index = np.clip(upper_index, 1, n_lines - 1)
    upper = np.take_along_axis(lines, upper_index, axis=1)
    lower = np.take_along_axis(lines, upper_index - 1, axis=1)
    upper = np.where(np.isfinite(upper), upper, lower)
    epsilon = np.abs(upper - lower)/2
    delta = np.abs(q2_obs[np.newaxis] - q2_calc)
    merit = np.zeros(n_candidates)
    mean_delta = np.mean(delta, axis=1)
    good = (count > 1) & (mean_delta > 0) & (q2_calc.sum(axis=1) != 0)
    merit[good] = np.mean(epsilon, axis=1)[good]/mean_delta[good]
    return merit


def before_n_over(q2_obs, q2_calc, q2_ref_calc, tolerance_factor=0.5):
    """get_n_over as it stood before the rewrite."""
    cutoff = q2_calc[:, -1]
    lines, count = fom._sorted_lines_in_range(q2_ref_calc, cutoff)
    finite = np.isfinite(lines)
    previous = np.concatenate(
        [np.zeros((lines.shape[0], 1)), np.where(finite, lines, 0.0)[:, :-1]], axis=1
    )
    local_gap = np.where(finite, np.where(finite, lines, 0.0) - previous, np.inf)
    nearest = np.min(
        np.abs(np.where(finite, lines, np.inf)[:, :, np.newaxis] - q2_obs[np.newaxis, np.newaxis]),
        axis=2,
    )
    unaccounted = finite & (nearest > tolerance_factor*local_gap)
    n_over = unaccounted.sum(axis=1)
    max_gap = np.zeros(lines.shape[0], dtype=int)
    for row in range(lines.shape[0]):
        run = best = 0
        for flag in unaccounted[row, : count[row]]:
            run = run + 1 if flag else 0
            best = max(best, run)
        max_gap[row] = best
    return n_over, max_gap


# name -> (before, after, comparator). The rewrites are checked against these, never against
# each other, and `bench` times the pair.
REWRITTEN = {
    'M_wu': (before_M_wu, lambda o, c, r: fom.get_M_wu(o, c, r), exact_match),
    'M_1': (before_M_1, lambda o, c, r: fom.get_M_1(o, c, r), exact_match),
    'n_over/max_gap': (before_n_over, lambda o, c, r: fom.get_n_over(o, c, r),
                       tuple_match([exact_match, exact_match])),
    }


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

# Priced against the whole module as it was, not against a copied variant, so the comparison
# covers the parts of each call nobody touched as well.
MODULE_LEVEL = {
    'compute_all': lambda m, c: m.compute_all(c.q2_obs, c.q2_calc, c.q2_ref_calc.copy(), c.xnn,
                                              c.lattice_system, c.bravais_lattice)['features'],
    'assignment_sigma': lambda m, c: dict(zip(
        ('sigma', 'd1'), m.get_assignment_sigma(c.q2_obs, c.q2_ref_calc, c.lattice_system))),
    'assignment_posterior': lambda m, c: {
        'posterior': m.get_assignment_posterior(c.q2_obs, c.q2_ref_calc, c.lattice_system)},
    }


def bench(args):
    baseline = (_baseline_module(args.revision)
                if any(name in MODULE_LEVEL for name in args.merits) else None)
    for n_candidates in args.sizes:
        case = load_capture(args.capture_file, n_candidates)
        q2_obs, q2_calc, q2_ref_calc = case.q2_obs, case.q2_calc, case.q2_ref_calc
        def columns_match(expected, got):
            bad = [name for name in expected
                   if name not in got
                   or not np.array_equal(np.asarray(expected[name]), np.asarray(got[name]),
                                         equal_nan=True)]
            return (not bad,
                    f'all {len(expected)} bit-identical' if not bad else f'{bad} differ')

        for name in args.merits:
            if name in MODULE_LEVEL:
                call = MODULE_LEVEL[name]
                benchmark(f'{name} -- {os.path.basename(args.capture_file)}, '
                          f'{n_candidates} candidates',
                          {'before': lambda c=call: c(baseline, case),
                           'control': lambda c=call: c(baseline, case),
                           'after': lambda c=call: c(fom, case)},
                          reference='before', control='control', rounds=args.rounds,
                          compare=columns_match,
                          work=f'{n_candidates} candidates x {q2_ref_calc.shape[1]} reference '
                               f'lines x {q2_obs.size} peaks; `before` is {args.revision}')
                continue
            before, after, compare = REWRITTEN[name]
            variants = {
                'before': lambda: before(q2_obs, q2_calc, q2_ref_calc),
                'control': lambda: before(q2_obs, q2_calc, q2_ref_calc),
                'after': lambda: after(q2_obs, q2_calc, q2_ref_calc),
                }
            benchmark(f'{name} -- {os.path.basename(args.capture_file)}, '
                      f'{n_candidates} candidates',
                      variants, reference='before', control='control',
                      rounds=args.rounds, compare=compare,
                      work=f'{n_candidates} candidates x {q2_ref_calc.shape[1]} reference '
                           f'lines x {q2_obs.size} peaks')


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
        label = f'random {n_candidates}x{n_ref}x{n_peaks}'
        if case % 7 == 0:
            q2_ref_calc = q2_ref_calc + 10.0
            label += ', no line in range'
        if case % 5 == 0:
            q2_ref_calc[:, ::2] = q2_obs[0]
            q2_ref_calc[:, 1::3] = 0.5*(q2_obs[0] + q2_obs[-1])
            label += ', ties and repeated lines'
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
            q2_ref_calc = -q2_ref_calc
            label += ', negative reference lines'
        if case % 23 == 0:
            q2_obs = q2_obs.copy()
            q2_obs[len(q2_obs)//2] = np.nan
            label += ', NaN peak (fallback path)'
        yield label, q2_obs, q2_calc, q2_ref_calc


def fuzz(args):
    """Randomised inputs, every rewritten function against the same function at `--revision`.

    The captures cannot contain a row with no line in range, a peak list with a NaN in it, an
    F-ordered reference array or an all-zero q2_calc, and every one of those is a path through
    the code. Comparing against git rather than against copies means the assignment family is
    covered too, without a second set of verbatim duplicates to keep in step.
    """
    baseline = _baseline_module(args.revision)
    rng = np.random.default_rng(args.seed)
    failures = 0
    for label, q2_obs, q2_calc, q2_ref_calc in _fuzz_cases(rng, args.cases):
        for name in args.merits:
            before, after, compare = REWRITTEN[name]
            ok, detail = compare(before(q2_obs, q2_calc, q2_ref_calc),
                                 after(q2_obs, q2_calc, q2_ref_calc))
            if not ok:
                failures += 1
                print(f'MISMATCH {name}: {label}\n   {detail}')
        for system in ('monoclinic', 'cubic'):
            expected = {}
            got = {}
            for module, into in ((baseline, expected), (fom, got)):
                sigma, d1 = module.get_assignment_sigma(q2_obs, q2_ref_calc, system)
                into['sigma'], into['d1'] = sigma, d1
                into['posterior'] = module.get_assignment_posterior(q2_obs, q2_ref_calc, system)
                into['posterior_shared'] = module.get_assignment_posterior(
                    q2_obs, q2_ref_calc, system, sigma=sigma, d1=d1)
                into['posterior_scaled'] = module.get_assignment_posterior(
                    q2_obs, q2_ref_calc, system, sigma_multiplier=8.0)
            for key in expected:
                left, right = expected[key], got[key]
                if not np.array_equal(left, right, equal_nan=True):
                    failures += 1
                    print(f'MISMATCH assignment.{key} ({system}): {label}')
    print(f'{args.cases} cases x {len(args.merits)} merits + the assignment family: '
          f'{failures} mismatches')
    return failures


# ---------------------------------------------------------------------------
# Verify against the pre-change code
# ---------------------------------------------------------------------------

def _baseline_module(revision):
    """`FigureOfMerits` as of a git revision, imported under its own name.

    The point of comparing against git rather than against hand-copied variants is that
    nothing can drift: it is the whole module as it was, including the parts of compute_all
    nobody touched, so `verify` covers every feature and not only the rewritten ones.
    """
    import importlib.util
    import subprocess
    import tempfile

    # Run git from the repository root rather than from wherever the caller happens to stand.
    root = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], cwd=str(Path(__file__).resolve().parent),
        capture_output=True, check=True, text=True).stdout.strip()
    source = subprocess.run(
        ['git', 'show', f'{revision}:mlindex/utilities/FigureOfMerits.py'], cwd=root,
        capture_output=True, check=True, text=True).stdout
    # The module is fully read by exec_module, so the file can go immediately afterwards.
    # The previous NamedTemporaryFile(delete=False) was never removed and leaked one .py
    # per invocation of bench, fuzz and verify.
    with tempfile.TemporaryDirectory() as directory:
        location = os.path.join(directory, 'FigureOfMerits_baseline.py')
        with open(location, 'w', encoding='utf-8') as handle:
            handle.write(source)
        specification = importlib.util.spec_from_file_location(
            'FigureOfMerits_baseline', location)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
    return module


def _named_outputs(module, case):
    """Every column each family produces, keyed by name, for one module and one capture."""
    q2_obs, q2_calc, q2_ref_calc = case.q2_obs, case.q2_calc, case.q2_ref_calc
    xnn, lattice_system, bravais_lattice = case.xnn, case.lattice_system, case.bravais_lattice
    outputs = {f'compute_all.{name}': value for name, value in
               module.compute_all(q2_obs, q2_calc, q2_ref_calc.copy(), xnn, lattice_system,
                                  bravais_lattice)['features'].items()}
    sigma, d1 = module.get_assignment_sigma(q2_obs, q2_ref_calc, lattice_system)
    outputs['assignment.sigma'] = sigma
    outputs['assignment.d1'] = d1
    outputs['assignment.sigma_robust'] = module.get_assignment_sigma(
        q2_obs, q2_ref_calc, lattice_system, robust=True)[0]
    outputs['assignment.posterior'] = module.get_assignment_posterior(
        q2_obs, q2_ref_calc, lattice_system)
    outputs['assignment.posterior_shared'] = module.get_assignment_posterior(
        q2_obs, q2_ref_calc, lattice_system, sigma=sigma, d1=d1)
    if case.hkl_ref is not None:
        # The insample.* and cv.* families are deliberately absent -- not ported, so there is
        # nothing on either side of the comparison to verify. See CHERRY_PICK.md.
        for name, value in module.get_holdout_fom(
                q2_obs[-5:] + 0.01, xnn, case.hkl_ref, lattice_system, bravais_lattice).items():
            outputs[f'holdout.{name}'] = value
    return outputs


def verify(args):
    """Every column every family produces, this revision against `--revision`, bit for bit.

    Comparing against the module as it was at a git revision rather than against hand-copied
    variants is the point: it covers the parts nobody touched as well, so a rewrite that
    perturbs a neighbour cannot pass.
    """
    baseline = _baseline_module(args.revision)
    for n_candidates in args.sizes:
        case = load_capture(args.capture_file, n_candidates)
        expected = _named_outputs(baseline, case)
        got = _named_outputs(fom, case)
        missing = set(expected) ^ set(got)
        shared = sorted(set(expected) & set(got))
        bad = {}
        for name in shared:
            left, right = np.asarray(expected[name]), np.asarray(got[name])
            if left.shape != right.shape:
                bad[name] = f'shape {right.shape} vs {left.shape}'
                continue
            # NaN is a legitimate value here -- a voided cross-validation fold writes it -- so
            # equality alone would report a match as a mismatch.
            differing = int(np.count_nonzero((left != right)
                                             & ~(np.isnan(left) & np.isnan(right))))
            if differing:
                bad[name] = f'{differing} of {left.size} values differ'
        values = sum(np.asarray(expected[name]).size for name in shared)
        print(f'{os.path.basename(args.capture_file)}, {n_candidates} candidates, '
              f'{len(shared)} columns / {values} values vs {args.revision}')
        if missing:
            print(f'   COLUMN SET DIFFERS: {sorted(missing)}')
        for name, detail in bad.items():
            print(f'   MISMATCH {name}: {detail}')
        if not bad and not missing:
            print(f'   all {len(shared)} columns bit-identical, {values} values')


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

    bcap = sub.add_parser('capture-pool')
    bcap.add_argument('--benchmark-dir', type=str,
                      default=os.path.join('mlindex', 'data', 'fom_benchmark'))
    bcap.add_argument('--bundle', type=str, default='error1_cont0')
    bcap.add_argument('--bravais-lattice', type=str, default='mP')
    bcap.add_argument('--lattice-system', type=str, default='monoclinic')
    bcap.add_argument('--max-rows', type=int, default=20000)
    bcap.add_argument('--out-dir', type=str, default='captures')
    bcap.set_defaults(func=capture_benchmark)

    cst = sub.add_parser('cost')
    cst.add_argument('--capture-file', type=str, required=True)
    cst.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    cst.add_argument('--repeats', type=int, default=5)
    cst.add_argument('--revision', type=str, default=None,
                     help='price the module as of this git revision instead of the working tree')
    cst.add_argument('--csv', type=str, default=None,
                     help='append the table to this CSV, with provenance columns')
    cst.add_argument('--regime', type=str, default=None,
                     help='population the capture came from, e.g. "inner loop" or "frozen pool"')
    cst.add_argument('--lattice', type=str, default=None,
                     help='lattice class label for the CSV; defaults to the capture own value')
    cst.add_argument('--capture-origin', type=str, default=None,
                     help='where the capture came from, e.g. "S02 2026-08-25 seed 12345"')
    cst.set_defaults(func=cost)

    prof = sub.add_parser('profile')
    prof.add_argument('--capture-file', type=str, required=True)
    prof.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    prof.add_argument('--repeats', type=int, default=3)
    prof.add_argument('--merits', type=str, nargs='+', required=True,
                      choices=list(PROFILE_TARGETS))
    prof.set_defaults(func=profile)

    bch = sub.add_parser('bench')
    bch.add_argument('--capture-file', type=str, required=True)
    bch.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    bch.add_argument('--rounds', type=int, default=11)
    bch.add_argument('--merits', type=str, nargs='+',
                     default=list(REWRITTEN) + list(MODULE_LEVEL),
                     choices=list(REWRITTEN) + list(MODULE_LEVEL))
    bch.add_argument('--revision', type=str, default='HEAD')
    bch.set_defaults(func=bench)

    fzz = sub.add_parser('fuzz')
    fzz.add_argument('--cases', type=int, default=400)
    fzz.add_argument('--seed', type=int, default=5)
    fzz.add_argument('--merits', type=str, nargs='+', default=list(REWRITTEN),
                     choices=list(REWRITTEN))
    fzz.add_argument('--revision', type=str, default='HEAD')
    fzz.set_defaults(func=fuzz)

    vfy = sub.add_parser('verify')
    vfy.add_argument('--capture-file', type=str, required=True)
    vfy.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    vfy.add_argument('--revision', type=str, default='HEAD')
    vfy.set_defaults(func=verify)

    args = parser.parse_args()
    # `fuzz` returns a mismatch count; without this it printed MISMATCH and still exited 0,
    # so it could never fail the acceptance gate that names it.
    failures = args.func(args)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
