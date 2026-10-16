"""Search measured patterns as `mlindex.run` and `mlindex.run_analytical` do, and keep every
printed list.

The search and the lists are the programs' own: the ML search of `mlindex.run --nproc 1` with the
models loaded once, or `run_analytical.search_serial`, then `run.printed_rows` under each ranking.
Each pattern's peaks are read as `--peak-units q --zero-error --wavelength <its wavelength>` reads
them. The patterns are a CNRS/opXRD-format dataframe (`pandas.read_json`) with `file_name`,
`peak_positions` (q = 1/d), `primary_wavelength`, `unit_cell`, `bravais_lattice` and `spacegroup`.

    python -m mlindex.scripts.run_real_benchmark --patterns real.json --indexer ml \\
        --part 0 --n-parts 8 --out-dir results/real

writes `printed_<indexer>_<part>.parquet`: one row per printed candidate, with its cell as the
search stores it (radians) and the pattern's truth, and `environment_<indexer>_<part>.json`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

SEED = 12345


def rankers(indexer):
    """The rankings each program's list is read under, by name."""
    from mlindex.utilities.Ranker import LearnedRanker, MeritRanker, load_ranker

    if indexer == 'analytical':
        from mlindex.command_line.run_analytical import RANKER
        return {'M_sym': RANKER, 'M20': MeritRanker('M20', 'benchmark')}
    learned = load_ranker()
    if not isinstance(learned, LearnedRanker):
        raise FileNotFoundError(f'No learned ranker: {learned.name}')
    return {'learned': learned, 'M_sym': MeritRanker('M_sym', 'benchmark'),
            'M20': MeritRanker('M20', 'benchmark')}


def search(patterns, indexer, part, n_parts):
    """Every printed row of part `part` of `n_parts` of `patterns`, as a frame."""
    from mlindex.command_line.run import _convert_to_q2, allocate_lattice_groups, printed_rows
    from mlindex.command_line.run_analytical import N_REF_HKL_GUESS, search_serial
    from mlindex.optimization.MPOptimizer import (
        run_lattice_groups, setup_lattice_groups, shutdown_lattice_groups)

    by_name = rankers(indexer)
    if indexer == 'ml':
        groups, processes = setup_lattice_groups(
            allocate_lattice_groups(list(BRAVAIS_LATTICES), 1), '1', 1, seed=SEED)
    rows = []
    try:
        for pattern in patterns.iloc[part::n_parts].itertuples():
            q2 = np.sort(_convert_to_q2(np.asarray(pattern.peak_positions)[:20], 'q', None))
            wavelength = pattern.primary_wavelength
            if indexer == 'ml':
                results = run_lattice_groups(groups, q2, True, wavelength, 20)
                results = {bl: results[bl] for bl in BRAVAIS_LATTICES}
            else:
                results = search_serial(q2, BRAVAIS_LATTICES,
                                        dict.fromkeys(BRAVAIS_LATTICES, N_REF_HKL_GUESS), True,
                                        wavelength, SEED)
            for ranking, ranker in by_name.items():
                for position, entry in enumerate(printed_rows(results, ranker)):
                    pool = results[entry['bravais_lattice']]['pool']
                    rows.append({
                        'entry_id': Path(pattern.file_name).name.split('.')[0],
                        'ranking': ranking, 'position': position,
                        'bravais_lattice': entry['bravais_lattice'], 'score': entry['score'],
                        'M20': entry['M20'], 'M_sym': entry['M_sym'],
                        'unit_cell': np.asarray(pool['unit_cell'][entry['row']]),
                        'bravais_lattice_true': pattern.bravais_lattice,
                        'unit_cell_true': np.asarray(pattern.unit_cell),
                        'spacegroup_true': pattern.spacegroup})
            print(f'{indexer} part {part}: {pattern.file_name}', flush=True)
    finally:
        if indexer == 'ml':
            shutdown_lattice_groups(groups, processes)
    return pd.DataFrame(rows)


def main(argv=None):
    from mlindex.model_training.BenchmarkPrinted import environment

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--patterns', required=True, metavar='PATH')
    parser.add_argument('--indexer', default='ml', choices=('ml', 'analytical'))
    parser.add_argument('--part', type=int, default=0, metavar='K',
                        help='Search patterns K, K+N, K+2N, ... (default: 0).')
    parser.add_argument('--n-parts', type=int, default=1, metavar='N', help='(default: 1)')
    parser.add_argument('--out-dir', required=True, metavar='PATH')
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.indexer}_{args.part:03d}'
    (out_dir / f'environment_{tag}.json').write_text(
        json.dumps(environment(), indent=2), encoding='utf-8')
    search(pd.read_json(args.patterns), args.indexer, args.part, args.n_parts).to_parquet(
        out_dir / f'printed_{tag}.parquet', index=False)


if __name__ == '__main__':
    main()
