"""S14 items 1-3 -- what the two retention mechanisms cost, and what they change.

Two mechanisms put correct cells outside the pool before ranking ever sees it, and both
are now implementable behind an opt_params flag:

  * **Multi-FOM iterate retention** (`retention_foms`). `iteration_worker_common` keeps,
    per candidate, the single iterate with the best M20 over 100 iterations. If the final
    score is not M20 that is the wrong iterate. Each extra merit named keeps one more.
  * **The deduplication tiebreak** (`dedup_tiebreak_foms`). `_downsample_chunk` keeps the
    highest-M20 member of each xnn neighbourhood and deletes the rest (F-065, R2). Each
    extra merit named additionally rescues the member that maximises it.

WHY COST COMES FIRST. Execution time is a first-class constraint on which merits may be
retained (DWMM, 2026-08-21): the inner loop evaluates its merit for ~1 500-4 000 candidates
over 100 iterations, `get_M20` alone is already ~16% of `_run_loop` (F-001), and the
acceptance gate asks for <= 2x `get_M20` and <= 25% end-to-end. Of everything this project
has priced only the distilled MLP (0.17x, F-092) and block A (0.66x amortised, F-134) fit
that budget -- against `M_sym` 24.3x, `ho_M20` 8.7x and all seventeen merits 145x -- and
block A is entry-level, so it is constant within an entry and cannot re-rank inside one at
all (R16). So a merit is priced here before it is retained anywhere.

Two traps this harness is built around. `get_M20` zeroes the out-of-range entries of
`q2_ref_calc` in place, so it is timed against fresh copies and anything computed beside it
runs first. And F-095: an inner-loop merit is handed the optimiser's own `q2_calc` /
`q2_ref_calc` rather than rederiving them from the Miller indices, because M20's cut-off is
itself one of the reference lines and the two routes to it differ by an ULP that moves a
line across its own boundary -- up to 18% of M20 on 0.1% of rows.

The deduplication tiebreak is deliberately *not* held to the 2x budget: it runs once per
(entry, Bravais lattice), not a hundred times over every candidate, and every merit it can
use is already carried into the chunk.

    python run_fom_retention_mechanisms.py --stage cost     --peak-file peaks.npy
    python run_fom_retention_mechanisms.py --stage validate --peak-file peaks.npy

Bounds. This is a *cost and behaviour* measurement on whatever entries it is given; it says
nothing about the ceiling, which needs the NERSC run and a labelled pool. Ratios are
per-call on one machine and one BLAS; what transfers is the ordering and the order of
magnitude, not the third digit.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from mlindex.optimization.Candidates import RETENTION_FOMS
from mlindex.optimization.MPIOptimizer import DEDUP_TIEBREAK_FOMS
from mlindex.utilities.FigureOfMerits import get_M20
from mlindex.utilities.FigureOfMerits import get_M20_likelihood
from mlindex.utilities.FigureOfMerits import get_M_rev_sym
from mlindex.utilities.UnitCellTools import get_unit_cell_volume

BROADENING_TAG = '1'


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage', choices=['cost', 'validate', 'report'], required=True)
    parser.add_argument('--peak-file', default=None,
                        help='numpy array of q2 values (1/Angstrom^2); cost and validate only')
    parser.add_argument('--bravais-lattices', default='mP,oP,tP',
                        help='comma separated; the cost of the reversed merits scales with '
                             'the reference list, which is what differs between these')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--artifact-dir', default='docs/fom/artifacts')
    parser.add_argument('--tag', default='S14_retention')
    return parser.parse_args()


def _build_candidates(bravais_lattice, q2_obs, seed, options=None):
    """A real Candidates for one lattice, at the production candidate count.

    Goes through setup_mp_optimizers with one process so nothing is spawned and the
    opt_params, the reference list and the generators are exactly production's.
    """
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers
    from mlindex.optimization.MPOptimizer import shutdown_mp_workers

    optimizers, processes, task_queues = setup_mp_optimizers(
        1, BROADENING_TAG, n_candidates_scale=1, seed=seed, options=options)
    shutdown_mp_workers(processes, task_queues)
    optimizer = optimizers[bravais_lattice]
    optimizer.q2_obs = np.asarray(q2_obs, dtype=float)[:optimizer.n_peaks]
    xnn = optimizer._generate_candidates_xnn()
    return optimizer, optimizer.generate_candidates_common(xnn)


def _one_iteration(candidates, iteration_info):
    """One inner-loop iteration, dispatched exactly as `_run_loop` dispatches it."""
    worker = getattr(candidates, iteration_info['worker'])
    worker(iteration_info)


def _dominant_iteration_info(optimizer):
    """The stanza that runs the most iterations -- for mP that is random_subsampling at 60
    of 61, so it is what the loop's cost is. Every stanza calls the same merit code."""
    return max(optimizer.opt_params['iteration_info'],
               key=lambda info: info['n_iterations'])


def _timed(function, repeats):
    """Best-of-`repeats` wall clock. Best rather than mean: this is a cost floor, and the
    tail is the machine's other work, not the merit's."""
    best = np.inf
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def run_cost(args):
    peaks = np.load(args.peak_file)
    rows = []
    for bravais_lattice in args.bravais_lattices.split(','):
        optimizer, candidates = _build_candidates(bravais_lattice, peaks, args.seed)
        n_candidates = candidates.n
        # The arrays assign_hkls holds at the moment the merit would be evaluated.
        from mlindex.utilities.numba_functions import fast_assign
        q2_ref_calc = optimizer_q2 = candidates.q2_calculator.get_q2(candidates.xnn)
        hkl_assign = fast_assign(candidates.q2_obs, q2_ref_calc)
        q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
        n_ref = q2_ref_calc.shape[1]

        # get_M20 mutates q2_ref_calc, so every repeat gets its own copy and the copies are
        # made outside the timed region.
        copies = [q2_ref_calc.copy() for _ in range(args.repeats)]
        counter = {'i': 0}

        def call_m20():
            get_M20(candidates.q2_obs, q2_calc, copies[counter['i']])
            counter['i'] += 1

        m20_time = _timed(call_m20, args.repeats)

        reciprocal_volume = get_unit_cell_volume(
            candidates.reciprocal_unit_cell, partial_unit_cell=True,
            lattice_system=candidates.lattice_system)
        minfo_time = _timed(
            lambda: get_M20_likelihood(candidates.q2_obs, q2_calc,
                                       candidates.bravais_lattice, reciprocal_volume),
            args.repeats)
        # get_M_rev_sym materialises (n_candidates, n_ref, n_peaks) floats. At production
        # sizes that is hundreds of gigabytes, so it is priced on a slice and the per-row
        # cost extrapolated -- the extrapolation is the point, not an approximation of a
        # number that could have been measured.
        slice_n = int(min(n_candidates, max(1, 2e8 // (8 * n_ref * candidates.n_peaks))))
        rev_time = _timed(
            lambda: get_M_rev_sym(candidates.q2_obs, q2_calc[:slice_n],
                                  q2_ref_calc[:slice_n]),
            max(2, args.repeats // 2))
        rev_time_full = rev_time * n_candidates / slice_n
        rev_bytes = 8.0 * n_candidates * n_ref * candidates.n_peaks

        # What the loop actually costs, with and without the merit, so the ratio can be
        # read against the 25% end-to-end gate as well as the 2x per-merit one.
        iteration_info = _dominant_iteration_info(optimizer)
        loop_time = _timed(
            lambda: _one_iteration(candidates, iteration_info), args.repeats)

        rows.append({
            'bravais_lattice': bravais_lattice,
            'lattice_system': candidates.lattice_system,
            'n_candidates': int(n_candidates),
            'n_ref': int(n_ref),
            'n_peaks': int(candidates.n_peaks),
            'get_M20_s': m20_time,
            'iteration_s': loop_time,
            'get_M20_frac_of_iteration': m20_time / loop_time,
            'Minfo_ratio': minfo_time / m20_time,
            'M_rev_sym_ratio': rev_time_full / m20_time,
            'M_rev_sym_priced_on': int(slice_n),
            'M_rev_sym_peak_GB': rev_bytes / 1e9,
            })
        print(f'[{bravais_lattice}] n_cand {n_candidates}  n_ref {n_ref}  '
              f'get_M20 {m20_time*1e3:.2f} ms  Minfo {minfo_time/m20_time:.3f}x  '
              f'M_rev_sym {rev_time_full/m20_time:.1f}x ({rev_bytes/1e9:.1f} GB)',
              flush=True)

    # The same measurement through the real loop: how much a retained merit costs the
    # iteration it sits in, which is what the 25% gate is about.
    loop_rows = []
    for bravais_lattice in args.bravais_lattices.split(','):
        for foms in [('M20',), ('M20', 'Minfo')]:
            optimizer, candidates = _build_candidates(
                bravais_lattice, peaks, args.seed, options={'retention_foms': foms})
            iteration_info = _dominant_iteration_info(optimizer)
            seconds = _timed(
                lambda: _one_iteration(candidates, iteration_info), args.repeats)
            loop_rows.append({'bravais_lattice': bravais_lattice,
                              'retention_foms': ','.join(foms),
                              'iteration_s': seconds,
                              'n_iterations_total': int(sum(
                                  info['n_iterations']
                                  for info in optimizer.opt_params['iteration_info'])),
                              })
            print(f'[{bravais_lattice}] retention {foms} -> {seconds*1e3:.1f} ms/iteration',
                  flush=True)

    _write(args, 'cost', {'per_merit': rows, 'per_iteration': loop_rows})


def _counts(optimizer):
    """Pre- and post-deduplication row counts, from the dump streams session 1 added.

    Also the split by `retained_by`, which is the column that makes item 1's ceiling a
    restriction inside one run rather than a comparison between two.
    """
    predownsample = optimizer.drain_predownsample_dump()
    downsampled = optimizer.drain_candidate_dump()
    entering = int(sum(record['xnn'].shape[0] for record in predownsample))
    surviving = int(sum(record['xnn'].shape[0] for record in downsampled))
    provenance = {}
    for record in predownsample:
        values, counts = np.unique(record['retained_by'], return_counts=True)
        for value, count in zip(values, counts):
            provenance[int(value)] = provenance.get(int(value), 0) + int(count)
    return entering, surviving, provenance


def run_validate(args):
    """Run one lattice end to end under four configurations and report what moved.

    The first two configurations differ only in whether the new keys are spelled out, so
    they must agree bit for bit; that is the safety argument for the defaults, checked
    rather than asserted.
    """
    from mlindex.optimization.MPOptimizer import run_mp_bl
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers
    from mlindex.optimization.MPOptimizer import shutdown_mp_workers

    peaks = np.load(args.peak_file)
    configurations = [
        ('default', None),
        ('explicit_M20', {'retention_foms': ('M20',),
                          'dedup_tiebreak_foms': ('M20',)}),
        ('retain_Minfo', {'retention_foms': ('M20', 'Minfo')}),
        ('tiebreak_Minfo', {'dedup_tiebreak_foms': ('M20', 'Minfo')}),
        ('both', {'retention_foms': ('M20', 'Minfo'),
                  'dedup_tiebreak_foms': ('M20', 'Minfo')}),
        ]
    results = {}
    for name, options in configurations:
        options = dict(options or {})
        options['dump_predownsample'] = True
        per_lattice = {}
        # A fresh optimizer per repeat, at the same seed. Re-running an existing one would
        # be a different computation -- self.rng has advanced -- so the repeats would not
        # be repeats of anything.
        for repeat in range(args.repeats):
            optimizers, processes, task_queues = setup_mp_optimizers(
                1, BROADENING_TAG, n_candidates_scale=1, seed=args.seed, options=options)
            for bravais_lattice in args.bravais_lattices.split(','):
                optimizer = optimizers[bravais_lattice]
                optimizer.opt_params['dump_candidates'] = True
                started = time.perf_counter()
                run_mp_bl(optimizer, bravais_lattice, task_queues, q2=peaks,
                          zero_error=False, wavelength=None, n_top=20)
                elapsed = time.perf_counter() - started
                entering, surviving, provenance = _counts(optimizer)
                record = per_lattice.setdefault(bravais_lattice, {
                    'seconds': np.inf, 'seconds_all': []})
                record['seconds'] = min(record['seconds'], elapsed)
                record['seconds_all'].append(elapsed)
                record['n_entering_dedup'] = entering
                record['n_surviving_dedup'] = surviving
                record['n_by_retention_track'] = provenance
                record['top_M20'] = optimizer.top_M20.tolist()
                record['top_unit_cell'] = optimizer.top_unit_cell.tolist()
                record['top_spacegroup'] = list(optimizer.top_spacegroup)
            shutdown_mp_workers(processes, task_queues)
        for bravais_lattice, record in per_lattice.items():
            print(f'[{name}/{bravais_lattice}] {record["n_entering_dedup"]} -> '
                  f'{record["n_surviving_dedup"]} candidates, best M20 '
                  f'{record["top_M20"][0]:.4f}, {record["seconds"]:.1f} s '
                  f'(best of {args.repeats}), tracks {record["n_by_retention_track"]}',
                  flush=True)
        results[name] = per_lattice

    identical = all(
        results['default'][bl]['top_M20'] == results['explicit_M20'][bl]['top_M20']
        and results['default'][bl]['top_unit_cell']
        == results['explicit_M20'][bl]['top_unit_cell']
        for bl in results['default'])
    print(f'\ndefault == explicit ("M20",) on every lattice: {identical}')
    results['default_equals_explicit_M20'] = bool(identical)
    baseline = results['default']
    for name in ('retain_Minfo', 'tiebreak_Minfo', 'both'):
        for bl, record in results[name].items():
            reference = baseline[bl]
            print(f'  {name}/{bl}: entering x{record["n_entering_dedup"]/reference["n_entering_dedup"]:.2f}, '
                  f'surviving x{record["n_surviving_dedup"]/reference["n_surviving_dedup"]:.2f}, '
                  f'wall clock {100*(record["seconds"]/reference["seconds"] - 1):+.0f}%')
    _write(args, 'validate', results)


def _write(args, stage, payload):
    directory = Path(args.artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{args.tag}_{stage}.json'
    # F-051's trap: _resolve_models_dir prefers $XDG_DATA_HOME/mlindex/models over the
    # repo's own mlindex/models, so a run from inside the repo can silently use a
    # downloaded tree. Two arms are only comparable if they used the same one, and that
    # cannot be checked afterwards unless it is written down.
    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
    payload = {
        'stage': stage,
        'machine': platform.platform(),
        'models_dir': str(_resolve_models_dir()),
        'numpy': np.__version__,
        'seed': args.seed,
        'peak_file': args.peak_file,
        'retention_foms_known': list(RETENTION_FOMS),
        'dedup_tiebreak_foms_known': list(DEDUP_TIEBREAK_FOMS),
        'result': payload,
        }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f'wrote {path}')


def run_report(args):
    """Turn the two JSON runs into the committed tables, so nothing lives only in a log."""
    import pandas as pd

    directory = Path(args.artifact_dir)
    cost = json.load(open(directory / f'{args.tag}_cost.json', encoding='utf-8'))
    validate = json.load(open(directory / f'{args.tag}_validate.json', encoding='utf-8'))

    per_merit = pd.DataFrame(cost['result']['per_merit'])
    per_iteration = pd.DataFrame(cost['result']['per_iteration'])
    pivot = per_iteration.pivot(index='bravais_lattice', columns='retention_foms',
                                values='iteration_s')
    pivot['iteration_overhead_pct'] = 100*(pivot['M20,Minfo']/pivot['M20'] - 1)
    # What the merit itself adds to a whole run: the per-iteration delta times the number
    # of iterations. Everything above that in the end-to-end delta is the extra rows the
    # post-prune chain has to walk, which is F-143's point.
    iterations = per_iteration.groupby('bravais_lattice')['n_iterations_total'].max()
    # From the per-call ratio rather than by differencing the two loop timings: at a few
    # percent the difference of two ~50 ms measurements is inside the run-to-run spread and
    # comes out negative on two lattices, while the per-call ratio is a best-of-five on the
    # merit alone and is stable. Both are in the CSV; only this one is quoted.
    per_merit_indexed = per_merit.set_index('bravais_lattice')
    pivot['merit_seconds_per_run'] = (
        per_merit_indexed['Minfo_ratio']*per_merit_indexed['get_M20_s']*iterations)
    pivot['merit_seconds_per_run_differenced'] = (
        pivot['M20,Minfo'] - pivot['M20'])*iterations
    per_merit = per_merit.merge(
        pivot.reset_index()[['bravais_lattice', 'iteration_overhead_pct',
                             'merit_seconds_per_run',
                             'merit_seconds_per_run_differenced']],
        on='bravais_lattice', how='left')
    per_merit.to_csv(directory / f'{args.tag}_cost.csv', index=False)

    rows = []
    baseline = validate['result']['default']
    for name, per_lattice in validate['result'].items():
        if not isinstance(per_lattice, dict) or name == 'default_equals_explicit_M20':
            continue
        for bravais_lattice, record in per_lattice.items():
            reference = baseline[bravais_lattice]
            rows.append({
                'configuration': name,
                'bravais_lattice': bravais_lattice,
                'n_entering_dedup': record['n_entering_dedup'],
                'n_surviving_dedup': record['n_surviving_dedup'],
                'entering_ratio': record['n_entering_dedup']/reference['n_entering_dedup'],
                'surviving_ratio': record['n_surviving_dedup']/reference['n_surviving_dedup'],
                'seconds': record['seconds'],
                'wall_clock_pct': 100*(record['seconds']/reference['seconds'] - 1),
                'top_M20': record['top_M20'][0],
                'top_M20_matches_default': bool(
                    record['top_M20'] == reference['top_M20']),
                'rank1_matches_default': bool(
                    record['top_M20'][0] == reference['top_M20'][0]
                    and record['top_unit_cell'][0] == reference['top_unit_cell'][0]),
                # How much of what run.py prints is the same candidate in the same slot.
                'top20_slots_identical': int(sum(
                    a == b for a, b in zip(reference['top_unit_cell'],
                                           record['top_unit_cell']))),
                'seconds_added': record['seconds'] - reference['seconds'],
                })
    pool = pd.DataFrame(rows)
    # The decomposition F-143 rests on: of the extra wall clock retention costs, how much
    # is the merit and how much is the candidates it keeps.
    merit = per_merit.set_index('bravais_lattice')['merit_seconds_per_run']
    retain = pool['configuration'] == 'retain_Minfo'
    pool.loc[retain, 'merit_share_of_added'] = (
        pool.loc[retain, 'bravais_lattice'].map(merit)
        / pool.loc[retain, 'seconds_added'])
    pool.to_csv(directory / f'{args.tag}_pool.csv', index=False)
    print(per_merit.to_string(index=False))
    print()
    print(pool.to_string(index=False))
    print(f'\nwrote {args.tag}_{{cost,pool}}.csv to {directory}')


def main():
    args = _parse_args()
    if args.stage == 'report':
        run_report(args)
        return
    if args.peak_file is None:
        raise SystemExit('--peak-file is required for the cost and validate stages')
    if args.stage == 'cost':
        run_cost(args)
    else:
        run_validate(args)


if __name__ == '__main__':
    main()
