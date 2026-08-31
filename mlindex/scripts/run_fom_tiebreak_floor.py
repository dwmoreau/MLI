"""S08 -- the tie-break floor: what a score that knows nothing already achieves.

    python mlindex/scripts/run_fom_tiebreak_floor.py \\
        --pool mlindex/data/fom_full_c2_pool \\
        --artifact-dir docs/fom_campaign2/artifacts

A **constant** score puts every candidate in a tie, so the outcome is decided entirely by the
tie-break -- score descending, then Bravais lattice in a fixed order, then `candidate_id`. Because
that order starts at cubic and the dominant failure mode is symmetry lowering, the tie-break hands
out a usable prior for free: campaign 1 measured a constant score at **0.2657** of top-10 on its
general population against a uniform random score's 0.0916, and both at exactly **0.00 %** on the
hard stratum, whose lattices sort last.

It is a property of the **population**, not of the metric, so it has to be measured for whatever
population is being reported on. METRICS.md section 8 requires it beside any rank metric: a merit
scoring 0.30 of top-10 has not beaten a coin, it has barely beaten a constant.

**This cannot be measured on a subsampled pool, and that is why it needs its own run.** Under a
constant score the rank is decided by the tie-break order, which is unrelated to the merits the
retention rule ranked on -- so retention is effectively random with respect to it and a correct
candidate is scored against a fraction of its true field. Benchmark B retains 8 206 of 26 734
survivors a cell, so a tie-break floor measured there would be flattered by ~3.3x of thinning.
`--pool` must be a fully retained pool; the script refuses a thinned one.
"""
import argparse
import os

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics


# Rank metrics only. A constant score has no meaningful threshold -- every candidate carries the
# same value, so any cut either accepts all of them or none.
REPORTED = ('top1', 'top5', 'top10', 'mrr', 'found')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='S08 -- the tie-break floor, on a fully retained pool')
    parser.add_argument('--pool', type=str, required=True,
                        help='A FULLY RETAINED pool. A subsampled one is refused: the tie-break '
                             'order is unrelated to the retention rule, so thinning flatters it')
    parser.add_argument('--artifact-dir', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', type=str, default='S08_tiebreak_floor')
    return parser.parse_args(argv)


def constant_score(frame):
    """Every candidate identical, so the tie-break decides everything."""
    return np.ones(frame.shape[0], dtype=np.float64)


class UniformScore:
    """A uniform random score, drawn reproducibly.

    Keyed on the candidate's own identity rather than on a running RNG, so the answer does not
    depend on how many shards the pool was written in or the order they were read -- the same
    property `subsample_negatives` needs for its own draw, and R17 one level down.
    """

    def __init__(self, seed):
        self.seed = int(seed)

    def __call__(self, frame):
        keys = (frame['entry_id'].astype(str) + '|' + frame['condition_bundle'].astype(str)
                + '|' + frame['bravais_lattice'].astype(str) + '|'
                + frame['candidate_id'].astype(str))
        digests = pd.util.hash_array(keys.to_numpy(dtype=object), hash_key=f'{self.seed:016d}')
        return (digests % np.uint64(1 << 53)).astype(np.float64)/float(1 << 53)


def main(argv=None):
    args = _parse_args(argv)
    depth, subsampled = FomBenchmark.subsample_depth(args.pool)
    if subsampled:
        raise SystemExit(
            f'{args.pool} is negatively subsampled at K={depth}. The tie-break floor cannot be '
            f'measured on it: a constant score is ranked entirely by the tie-break, which is '
            f'unrelated to the merits the retention rule kept, so the pool is thinned at random '
            f'with respect to it and the floor comes out flattered. Generate a fully retained '
            f'pool (submit_fom_full_retained.sh) and point at that.')

    rows = []
    for name, score in (('constant', constant_score),
                        ('uniform_random', UniformScore(args.seed)),
                        ('M20', 'M20')):
        result = FomMetrics.evaluate(args.pool, score=score, n_bootstrap=0,
                                     score_columns=('condition_bundle',))
        for scope, table in (('aggregate', result.aggregate), ('hard', result.hard)):
            block = table.iloc[0]
            rows.append(dict(score=name, scope=scope,
                             n_entries=int(block['n_entries']),
                             n_reachable=int(block['n_found']),
                             **{metric: float(block[metric]) for metric in REPORTED}))
        print(f'{name:16s} top10 {result.metric("top10"):.4f}  '
              f'hard {float(result.hard["top10"].iloc[0]):.4f}')

    table = pd.DataFrame(rows)
    os.makedirs(args.artifact_dir, exist_ok=True)
    path = os.path.join(args.artifact_dir, f'{args.tag}.csv')
    table.to_csv(path, index=False)
    print(f'\n{table.round(4).to_string(index=False)}')
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
