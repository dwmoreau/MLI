"""Measure whether the candidate search carries random state between patterns.

The indexer is asked for one pattern at a time, so a single run of the command line
tool cannot show this: it indexes one pattern and stops. A benchmark run is nothing
but the case this measures -- many patterns through one long-lived set of optimizers.

The probe runs a pattern twice: once through optimizers that have already indexed a
different pattern, and once from a clean start. Any difference is state carried
across patterns. It should be zero.

    python -m mlindex.scripts.measure_search_reproducibility --nproc 1
    python -m mlindex.scripts.measure_search_reproducibility --nproc 4 \
        --bravais-lattices aP,mP,tP,cF

Set MLINDEX_MODELS_DIR to the model tree you mean to measure. On a machine with a
downloaded tree in XDG_DATA_HOME that tree wins by default, and it may not be the one
in the checkout.
"""
import argparse
import ast

import numpy as np
import pandas as pd

from importlib.resources import files

DEFAULT_BRAVAIS_LATTICES = 'aP,mP,tP,cF'


def _test_data_dir():
    return files('mlindex').joinpath('data', 'test_data')


def load_pattern(bravais_lattice):
    """The 20-peak q2 list of the first test pattern registered for this lattice."""
    with _test_data_dir().joinpath('gsasII_tutorials.csv').open(encoding='utf-8') as handle:
        registry = pd.read_csv(handle)
    rows = registry[registry['bravais lattice'] == bravais_lattice]
    if rows.empty:
        raise SystemExit(f'no test pattern registered for {bravais_lattice}')
    peak_file = str(rows.iloc[0]['peak list file'])
    if not peak_file.endswith('.csv'):
        raise SystemExit(f'{bravais_lattice} maps to {peak_file}, which this probe does not read')
    name = peak_file[:-len('.csv')]
    with _test_data_dir().joinpath(name, peak_file).open(encoding='utf-8') as handle:
        frame = pd.read_csv(handle, index_col=0)
    positions = np.array(ast.literal_eval(frame.loc['peak_positions'].iloc[0]))
    return np.sort(positions ** 2)[:20]


def run_sequence(patterns, bravais_lattices, n_procs, seed):
    """Index each pattern in turn through one long-lived set of optimizers."""
    from mlindex.optimization.MPOptimizer import (
        setup_mp_optimizers, run_mp_bl, shutdown_mp_workers)

    optimizers, processes, task_queues = setup_mp_optimizers(
        n_procs, '1', n_candidates_scale=1, seed=seed)
    try:
        results = []
        for q2_obs in patterns:
            per_pattern = {}
            for bravais_lattice in bravais_lattices:
                run_mp_bl(optimizers[bravais_lattice], bravais_lattice, task_queues,
                          q2=q2_obs, zero_error=False, wavelength=None, n_top=20)
                per_pattern[bravais_lattice] = np.array(
                    optimizers[bravais_lattice].top_M20)
            results.append(per_pattern)
        return results
    finally:
        shutdown_mp_workers(processes, task_queues)


def main():
    parser = argparse.ArgumentParser(
        description='Measure whether the search carries random state between patterns')
    parser.add_argument('--nproc', type=int, default=1,
                        help='Worker pool size (default: 1)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Search seed (default: 12345)')
    parser.add_argument('--bravais-lattices', type=str, default=DEFAULT_BRAVAIS_LATTICES,
                        help=f'Comma-separated lattices to score '
                             f'(default: {DEFAULT_BRAVAIS_LATTICES})')
    parser.add_argument('--first', type=str, default='tP',
                        help='Lattice whose test pattern is indexed first (default: tP)')
    parser.add_argument('--subject', type=str, default='aP',
                        help='Lattice whose test pattern is compared (default: aP)')
    args = parser.parse_args()

    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    first = load_pattern(args.first)
    subject = load_pattern(args.subject)

    after = run_sequence([first, subject], bravais_lattices, args.nproc, args.seed)[1]
    alone = run_sequence([subject], bravais_lattices, args.nproc, args.seed)[0]

    print(f'pool size {args.nproc}, seed {args.seed}: '
          f'{args.subject} pattern after a {args.first} pattern, against on its own')
    print(f"{'lattice':>8} {'rows':>6} {'differing':>10}  max |dM20|")
    total_rows = total_differing = 0
    for bravais_lattice in bravais_lattices:
        a, b = after[bravais_lattice], alone[bravais_lattice]
        n = min(a.size, b.size)
        differing = int(np.count_nonzero(a[:n] != b[:n])) + abs(a.size - b.size)
        worst = float(np.max(np.abs(a[:n] - b[:n]))) if n else 0.0
        total_rows += max(a.size, b.size)
        total_differing += differing
        print(f'{bravais_lattice:>8} {max(a.size, b.size):>6} {differing:>10}  {worst:.6g}')
    print(f'\n{total_differing} of {total_rows} candidate rows differ because another '
          f'pattern was indexed first.')
    return 0 if total_differing == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
