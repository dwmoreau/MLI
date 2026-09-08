"""S18: how many peaks does the final refinement's filter admit, per arm, on real-run pools?

    python mlindex/scripts/run_fom_admitted_peaks.py --pool <base pool> --pool <variant pool> [...]

`refine_cell` admits every peak whose assignment statistic exceeds a threshold and takes one
least-squares step on those peaks alone; `n_indexed` on the persisted pool is that count on the
final cell. A cell has 1-6 free parameters (`N_FREE_PARAMETERS`); a step on fewer admitted
peaks than that is singular (zero step) and on barely more is poorly conditioned. This tabulates
the count per pool, per lattice, split by correctness, and the share of candidates at or below
the parameter count -- the test of the hypothesis in C2-F-171's follow-up: a calibrated
posterior at 0.99 admits fewer peaks on hard patterns than the over-confident `rho` at 0.95,
and starves the refinement where the cells are hardest.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

N_FREE = {'aP': 6, 'mP': 4, 'mC': 4, 'oP': 3, 'oC': 3, 'oI': 3, 'oF': 3,
          'tP': 2, 'tI': 2, 'hP': 2, 'hR': 2, 'cP': 1, 'cI': 1, 'cF': 1}


def tabulate(pool):
    files = sorted(glob.glob(os.path.join(pool, 'candidates_*.parquet')))
    frame = pd.concat([pd.read_parquet(f, columns=['bravais_lattice', 'n_indexed', 'is_correct',
                                                   'in_top_n'])
                       for f in files], ignore_index=True)
    frame['n_free'] = frame['bravais_lattice'].map(N_FREE)
    rows = []
    for (lattice, correct), g in frame.groupby(['bravais_lattice', 'is_correct'], sort=False):
        k = N_FREE[lattice]
        n = g['n_indexed'].to_numpy()
        rows.append(dict(pool=os.path.basename(os.path.dirname(pool)), bravais_lattice=lattice,
                         is_correct=bool(correct), n_candidates=int(len(g)),
                         n_free=k, admitted_mean=float(n.mean()), admitted_median=float(np.median(n)),
                         share_le_n_free=float((n <= k).mean()),
                         share_le_n_free_plus2=float((n <= k + 2).mean()),
                         share_zero=float((n == 0).mean())))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pool', action='append', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    table = pd.concat([tabulate(p) for p in args.pool], ignore_index=True)
    pd.set_option('display.width', 220)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f'-> {args.out}')


if __name__ == '__main__':
    main()
