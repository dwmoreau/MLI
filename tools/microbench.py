"""Shared harness for the isolated hotspot reproducers.

DEVELOPMENT TOOL -- not part of the installed package.

Every reproducer in tools/repro_*.py uses this so they all share one timing
methodology, arrived at after a naive "best of N, run sequentially" harness was
shown to be indefensible:

* **Interleaved.** Each round times every variant once, rather than all repeats
  of one variant and then the next. Drift over the timing window -- background
  load, thermal throttling, frequency scaling -- then hits every variant inside
  the same round instead of landing on whichever ran last.
* **Order-randomised.** The order within a round is reshuffled every round, so
  position in the round cannot bias a variant.
* **Paired.** Ratios are formed within a round and reported as a median with a
  [p10, p90] interval, not as a single best-of number.
* **Null-calibrated.** Every reproducer registers a ``control`` variant that is
  a byte-identical copy of the reference implementation, compiled/defined
  separately. Its true ratio is 1.000x by construction, so the interval it
  reports is the harness's noise floor on that machine at that moment. A
  variant whose interval overlaps the control's has *not* been shown to differ
  from the reference, however good its median looks.

The control is the load-bearing part. Without it there is no way to tell a 3%
effect from a 3% measurement artefact, and the reproducers exist to make claims
that survive scrutiny rather than claims that happen to be true.
"""

import os
import time

import numpy as np

__all__ = ['benchmark', 'exact_match', 'allclose_match', 'tuple_match']


# ---------------------------------------------------------------------------
# Correctness comparators
# ---------------------------------------------------------------------------

def exact_match(reference, candidate):
    """Bit-identical array comparison."""
    if isinstance(reference, np.ndarray):
        if reference.shape != candidate.shape:
            return False, f'shape {candidate.shape} vs {reference.shape}'
        if np.array_equal(reference, candidate):
            return True, 'bit-identical'
        return False, f'{int(np.count_nonzero(reference != candidate))} entries differ'
    if reference == candidate:
        return True, 'equal'
    return False, 'differ'


def allclose_match(reference, candidate, tolerance=1e-9, atol=0.0):
    """Relative comparison, for variants that legitimately reassociate floats.

    ``atol`` is not optional decoration. A purely relative test on a quantity
    that is legitimately near zero manufactures enormous ratios out of nothing:
    comparing two Gauss-Newton steps of 3.5e-17 and 3.4e-17 -- both meaning "this
    candidate has converged, do not move it" -- reported a relative difference of
    2.3e+04 and disqualified a correct implementation. Give an absolute floor
    scaled to the quantity the result is actually used against.
    """
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        return False, f'shape {candidate.shape} vs {reference.shape}'
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not np.array_equal(np.isfinite(reference), np.isfinite(candidate)):
        return False, 'nan/inf pattern differs'
    if not finite.any():
        return True, 'all non-finite, patterns match'
    scale = np.maximum(np.abs(reference[finite]), atol if atol > 0 else 1e-300)
    worst = float(np.max(np.abs(reference[finite] - candidate[finite]) / scale))
    return worst <= tolerance, f'max rel diff {worst:.2e}'


def tuple_match(comparators):
    """Compare a tuple of outputs element-wise with per-element comparators."""
    def compare(reference, candidate):
        if len(reference) != len(candidate):
            return False, 'different number of outputs'
        details = []
        ok = True
        for index, comparator in enumerate(comparators):
            element_ok, detail = comparator(reference[index], candidate[index])
            ok = ok and element_ok
            details.append(f'[{index}] {detail}')
        return ok, '; '.join(details)
    return compare


def list_match(reference, candidate):
    """Exact comparison for python lists (used for the spacegroup lists)."""
    if list(reference) == list(candidate):
        return True, 'identical'
    return False, f'lists differ (len {len(candidate)} vs {len(reference)})'


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def benchmark(title, variants, reference='current', control='control',
              rounds=30, seed=7, compare=exact_match, work=None, quiet=False):
    """Time ``variants`` against each other and report with a noise floor.

    ``variants`` maps a name to a zero-argument callable returning that
    implementation's result. ``reference`` names the implementation everything
    is compared against; ``control`` names the byte-identical copy of it that
    calibrates the noise floor.

    Returns a dict of name -> {'ratio', 'low', 'high', 'median', 'ok', 'detail'}.
    """
    names = list(variants)
    if reference not in names:
        raise ValueError(f'reference {reference!r} is not among {names}')

    print('=' * 100)
    print(title)
    print('=' * 100)
    if work:
        print(work)
    try:
        print(f'load average before: {tuple(round(v, 2) for v in os.getloadavg())}')
    except OSError:
        pass

    # --- correctness, before any timing --------------------------------
    reference_result = variants[reference]()
    checks = {}
    for name in names:
        if name == reference:
            checks[name] = (True, 'reference')
            continue
        ok, detail = compare(reference_result, variants[name]())
        checks[name] = (ok, detail)

    # --- timing --------------------------------------------------------
    rng = np.random.default_rng(seed)
    timings = {name: [] for name in names}
    for _ in range(rounds):
        order = list(names)
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter()
            variants[name]()
            timings[name].append(time.perf_counter() - start)

    base = np.array(timings[reference])
    results = {}
    print(f'\n{"variant":<22} {"median(s)":>10} {"speedup vs " + reference:>28}   '
          f'{"correctness":<44}')
    for name in names:
        times = np.array(timings[name])
        ratios = base / times
        low = float(np.percentile(ratios, 10))
        high = float(np.percentile(ratios, 90))
        ok, detail = checks[name]
        results[name] = {'ratio': float(np.median(ratios)), 'low': low, 'high': high,
                         'median': float(np.median(times)), 'ok': ok, 'detail': detail}
        interval = f'{np.median(ratios):6.3f}x [{low:5.3f}, {high:5.3f}]'
        flag = ' ' if ok else '!'
        print(f'{name:<22} {np.median(times):10.5f} {interval:>28} {flag} {detail:<44}')

    # --- verdict against the noise floor -------------------------------
    if control in results:
        floor_low = results[control]['low']
        floor_high = results[control]['high']
        print(f'\nnoise floor from {control!r} (true value 1.000x): '
              f'[{floor_low:.3f}, {floor_high:.3f}]')
        for name in names:
            if name in (reference, control):
                continue
            entry = results[name]
            if not entry['ok']:
                verdict = 'DISQUALIFIED -- output differs from the reference'
            elif entry['low'] > floor_high:
                verdict = f'FASTER  ({entry["ratio"]:.3f}x, clears the noise floor)'
            elif entry['high'] < floor_low:
                verdict = f'SLOWER  ({entry["ratio"]:.3f}x, clears the noise floor)'
            else:
                verdict = 'INDISTINGUISHABLE from the reference'
            print(f'   {name:<22} {verdict}')
    else:
        print('\nWARNING: no control variant registered, so there is no noise floor '
              'and none of the ratios above can be called real.')
    return results
