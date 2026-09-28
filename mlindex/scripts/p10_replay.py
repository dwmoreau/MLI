"""P10: does the final refinement step help, and at which peak-assignment threshold?

    python -m mlindex.scripts.p10_replay \
        --split-manifest docs/fom_campaign2/artifacts/S06_split_manifest.parquet \
        --split fom-train --population hard --per-lattice 120 --n-procs 4 \
        --out-dir $SCRATCH/p10_replay/hard

RESEARCH CODE THAT NEEDS TO BE DELETED -- P10, when P10 closes. On NERSC it is run by
`submit_p10_replay.sh`.

The search is run ONCE per pattern and Bravais lattice, up to and including the M20 cut. That
state -- the candidates and the search's random generator -- is copied, and the rest of the
pipeline (`refine_cell`, `standardize_cell`, `correct_off_by_two`, `assign_extinction_group`,
`calculate_peaks_indexed`, deduplication) is replayed from an identical copy under each setting
in `VARIANTS`. Every setting therefore starts from the same candidates, and a difference between
two of them is caused by the final step alone.

Each setting is written as an ordinary benchmark pool, one directory per setting, through the
harness's own writers, labeller and manifest. Score, reduce and compare them with
`run_benchmark.py` exactly as any arm:

    python -m mlindex.scripts.run_benchmark --stage sidecars --pool <out-dir>/t0.95
    python -m mlindex.scripts.run_benchmark --stage contrast --scores M_sym,M20 \
        --arm t0.95=<out-dir>/t0.95 --arm no_step=<out-dir>/no_step ... --reference t0.95

`t0.95` is the shipped setting; `t0.95_noA` is the shipped setting before the minimum-peaks rule
(cb132cc). Runs from the working tree, whose commit the manifest records.
"""

import argparse
import copy
import json
import platform
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import Benchmark, BenchmarkConditions, BenchmarkPatterns
from mlindex.model_training import BenchmarkRuns as runs
from mlindex.model_training.BenchmarkOptimizer import BenchmarkOptimizer
from mlindex.optimization import Candidates as candidates_module
from mlindex.utilities.Digests import q2_digest
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

# name: (assignment_threshold, minimum-peaks rule on, final step taken)
VARIANTS = {
    'no_step': (0.95, True, False),
    't0.95_noA': (0.95, False, True),
    't0.00': (0.0, True, True),
    't0.50': (0.50, True, True),
    't0.80': (0.80, True, True),
    't0.90': (0.90, True, True),
    't0.95': (0.95, True, True),
    't0.99': (0.99, True, True),
    't0.999': (0.999, True, True),
    }
CUT = 1.5


class ReplayOptimizer(BenchmarkOptimizer):
    """Runs the search once, then the tail of the pipeline once per variant."""

    def _run_loop(self, n_top_candidates):
        self._reseed_for_pattern()
        candidates = self.generate_candidates_rank()
        for iteration_info in self.opt_params['iteration_info']:
            for _ in range(iteration_info['n_iterations']):
                getattr(candidates, iteration_info['worker'])(iteration_info)
        candidates.prune_below_m20(threshold=self.opt_params['prune_m20_threshold'])
        self.variant_records = {}
        shipped_rule = candidates_module.MIN_EXCESS_PEAKS
        for name, (threshold, rule, step) in VARIANTS.items():
            replay = copy.deepcopy(candidates)
            replay.assignment_threshold = threshold
            candidates_module.MIN_EXCESS_PEAKS = shipped_rule if rule else -10**6
            try:
                if step:
                    replay.refine_cell()
            finally:
                candidates_module.MIN_EXCESS_PEAKS = shipped_rule
            replay.standardize_cell()
            replay.correct_off_by_two()
            replay.assign_extinction_group()
            replay.calculate_peaks_indexed()
            self.downsample_candidates(replay, n_top_candidates)
            records = self.drain()
            for record in records:
                record['assignment_threshold'] = float(threshold)
            self.variant_records[name] = records


def _worker(part, n_procs, args):
    from mlindex.optimization.MPOptimizer import (
        run_mp_bl, setup_mp_optimizers, shutdown_mp_workers)

    design = runs.POPULATIONS[args.population]
    chosen = runs.draw_entries(args.split_manifest, args.per_lattice, args.seed, split=args.split,
                               bravais_lattices=design['bravais_lattices'])
    source_rows = runs.load_source_rows(chosen)
    second_phase_pool = BenchmarkPatterns.build_second_phase_pool(source_rows)
    stripe = source_rows.iloc[part::n_procs].reset_index(drop=True)
    bundles = list(args.bundles or design['bundles'])

    optimizers, processes, task_queues = setup_mp_optimizers(
        1, BenchmarkPatterns.BROADENING_TAG, 1, seed=args.search_seed,
        options={'prune_m20_threshold': CUT}, optimizer_class=ReplayOptimizer)
    entry_rows = {name: [] for name in VARIANTS}
    failures = []
    started, done = time.perf_counter(), 0
    try:
        for bundle in bundles:
            condition = BenchmarkConditions.BY_TAG[bundle]
            records = {name: [] for name in VARIANTS}
            for _, entry in stripe.iterrows():
                try:
                    pattern = BenchmarkPatterns.prepare_peak_list(
                        entry, condition, args.seed, hkl=runs._hkl_of(entry),
                        second_phase_pool=second_phase_pool)
                except ContaminantPlacementError as error:
                    failures.append({'entry_id': entry['identifier'],
                                     'condition_bundle': bundle, 'reason': str(error)})
                    continue
                q2 = np.asarray(pattern.q2_obs, dtype=np.float64)
                digest = q2_digest(q2)
                pool_size_full = {name: 0 for name in VARIANTS}
                for lattice in BRAVAIS_LATTICES:
                    optimizer = optimizers[lattice]
                    optimizer.dump_context = {'entry_id': entry['identifier'],
                                              'condition_bundle': bundle, 'q2_digest': digest}
                    run_mp_bl(optimizer, lattice, task_queues, q2=q2, zero_error=False,
                              wavelength=None, n_top=Benchmark.N_TOP_CANDIDATES)
                    for name, drained in optimizer.variant_records.items():
                        records[name] += drained
                        pool_size_full[name] += sum(int(r['M20'].shape[0]) for r in drained)
                for name in VARIANTS:
                    entry_rows[name].append(runs.entry_record(
                        entry, condition, pattern, digest, args.split,
                        pool_size_full=pool_size_full[name]))
                done += 1
                elapsed = time.perf_counter() - started
                print(f'part {part:02d}: {done}/{len(bundles)*stripe.shape[0]} patterns, '
                      f'{elapsed/done:.1f} s each', flush=True)
            for name in VARIANTS:
                runs._write_bundle(Benchmark.part_dir(Path(args.out_dir)/name, part), bundle,
                                   records[name], entry_rows[name])
    finally:
        shutdown_mp_workers(processes, task_queues)
    for name in VARIANTS:
        directory = Benchmark.part_dir(Path(args.out_dir)/name, part)
        directory.mkdir(parents=True, exist_ok=True)
        Benchmark.write_entry_table(
            pd.DataFrame(entry_rows[name], columns=list(Benchmark.ENTRY_COLUMNS)), directory)
        with open(directory/'failures.json', 'w', encoding='utf-8') as handle:
            json.dump(failures, handle, indent=2, sort_keys=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--split-manifest', required=True, metavar='PATH')
    parser.add_argument('--split', default='fom-train', choices=runs.SPLITS)
    parser.add_argument('--population', default='hard', choices=tuple(runs.POPULATIONS))
    parser.add_argument('--per-lattice', type=int, default=120, metavar='N')
    parser.add_argument('--bundles', default=None, metavar='A,B')
    parser.add_argument('--seed', type=int, default=12345, metavar='N')
    parser.add_argument('--search-seed', type=int, default=12345, metavar='N')
    parser.add_argument('--n-procs', type=int, default=4, metavar='N')
    parser.add_argument('--out-dir', required=True, metavar='PATH')
    args = parser.parse_args(argv)
    args.bundles = [b for b in (args.bundles or '').split(',') if b] or None
    out_dir = Path(args.out_dir)
    for name in VARIANTS:
        runs._refuse_an_occupied_directory(out_dir/name)
    commit = runs._commit()

    workers = [Process(target=_worker, args=(part, args.n_procs, args))
               for part in range(args.n_procs)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode for worker in workers):
        raise SystemExit('a worker failed; its output is above. Nothing is stamped complete.')

    design = runs.POPULATIONS[args.population]
    bundles = list(args.bundles or design['bundles'])
    for name, (threshold, rule, step) in VARIANTS.items():
        pool_dir = out_dir/name
        failures = runs._collect_failures(pool_dir)
        Benchmark.consolidate(pool_dir)
        n_source = int(pd.read_parquet(next(pool_dir.glob('entries*.parquet')),
                                       columns=['entry_id'])['entry_id'].nunique())
        Benchmark.write_manifest(
            pool_dir, population=args.population, bundles=bundles,
            bravais_lattices=list(BRAVAIS_LATTICES), split=args.split,
            true_lattices=list(design['bravais_lattices']), n_source_entries=n_source,
            n_patterns_refused=len(failures), per_lattice=int(args.per_lattice),
            seed=int(args.seed), search_seed=int(args.search_seed), prune_threshold=CUT,
            pool_size=1, ensemble=runs.ensemble_record(), n_pools=int(args.n_procs),
            n_top_candidates=int(Benchmark.N_TOP_CANDIDATES),
            broadening_tag=BenchmarkPatterns.BROADENING_TAG,
            condition_set_digest=BenchmarkConditions.condition_set_digest(),
            split_manifest=str(args.split_manifest),
            split_manifest_sha256=runs.file_digest(args.split_manifest),
            degeneracy_rule='not_evaluated', commit=commit, arch=platform.machine(),
            platform=platform.platform(), python_version=platform.python_version(),
            numpy_version=np.__version__, mlindex_version='replay',
            replay_variant=name, assignment_threshold=threshold, minimum_peaks_rule=rule,
            final_step=step)
        Benchmark.stamp_complete(pool_dir, n_source_entries=n_source, n_bundles=len(bundles))
        print(f'wrote {pool_dir}')


if __name__ == '__main__':
    main()
