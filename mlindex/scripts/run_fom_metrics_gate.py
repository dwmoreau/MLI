"""S08 acceptance condition 1 -- the metrics module reproduces the pipeline's own ranking.

    python mlindex/scripts/run_fom_metrics_gate.py \\
        --pool mlindex/data/fom_benchmark_c2 \\
        --artifact-dir docs/fom_campaign2/artifacts

Writes `S08_metrics_gate.csv`: one row per check, with the measured value, the tolerance and a
verdict, so the gate is an artefact rather than a test that printed something once.

**This is not campaign 1's gate and does not port from it.** `fom`'s `run_fom_metrics_gate.py`
compares the module's per-lattice operating point against S02's published Benchmark A numbers, and
that comparison does not exist here: no campaign-2 number has been published for it to reproduce,
and a stratified slice's operating point is a property of the slice.

What campaign 2 has instead is **better**, and it is why the check is sharper rather than weaker.
The pool stores `final_rank` -- the pipeline's own 0-based rank by descending M20 over all
survivors of each (entry, lattice), computed *before* subsampling -- so the module's ranking can be
checked against the ranking the indexer actually produced, on every row, rather than against a
summary statistic. If those agree, the module is ranking the pool the way the program does; if they
disagree, no aggregate built on them means anything.

Six checks, and each is a way the module could be wrong while looking right:

  rank_order          the module's order reproduces `final_rank`. Tested as monotonicity of M20
                      along stored rank rather than as sequence equality, because `final_rank`'s
                      own tie-break is not recorded and ties are common -- a sequence comparison
                      would fail on ties that are not disagreements.
  in_top_n            the stored flag is `final_rank < 20`. It is what `pool_subset='in_top_n'`
                      selects on, so a drifted flag silently changes which pool is being reported.
  pool_depth          the operating point is identical whether the pool is every survivor or the
                      top twenty per lattice. METRICS.md section 1 says it must be -- a pooled
                      top-ten member is necessarily inside its own lattice's top twenty -- so a
                      difference means the pooling is wrong, not that the pool is deeper.
  ceilings            `ceiling_reranker` IS `threshold_only` and `ceiling_rescorer` IS `found`.
                      Identities, not approximations (METRICS.md section 3).
  loss_decomposition  the five outcome buckets partition the entries exactly once.
  degeneracy          the module's degenerate rate matches the entry table it was read from,
                      which is the check that C2-F-080's broadcast is wired up and not merely
                      present.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics


# The pipeline keeps the best twenty candidates per Bravais lattice, and `in_top_n` is that flag.
N_TOP_CANDIDATES = 20

# Exact identities are checked exactly. Only the rank reproduction gets a tolerance, and it is a
# rate rather than a numerical one.
EXACT = 0.0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='S08 acceptance condition 1: the module reproduces the pipeline ranking')
    parser.add_argument('--pool', type=str, required=True,
                        help='A schema-v3 pool, or a slice of one')
    parser.add_argument('--artifact-dir', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Accept threshold. Fixed a priori, NOT selected here -- selection is '
                             'S09 business and happens on fom-train (PROTOCOL section 8). The '
                             'checks below are identities and do not depend on its value')
    parser.add_argument('--tag', type=str, default='S08_metrics_gate')
    return parser.parse_args(argv)


def rank_reproduction(pool):
    """Does M20 fall monotonically along the pipeline's own `final_rank`?

    `final_rank` is the indexer's 0-based rank by descending M20 within each (entry, lattice),
    computed before subsampling -- so on a thinned pool it has gaps, but it stays monotone. Sorting
    retained rows by it must therefore give a non-increasing M20 sequence.

    Tested as monotonicity rather than as sequence equality on purpose. `final_rank`'s own
    tie-break is not recorded anywhere, and ties in M20 are common; comparing the two orders
    element by element would report every tie as a disagreement and the check would fail for a
    reason that is not a defect. Monotonicity is the property that actually has to hold, and it
    is violated by any real reordering.
    """
    violations = pools = rows = ties = 0
    worst = 0.0
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        bundle = FomBenchmark.bundle_from_candidate_path(path)
        frame = pd.read_parquet(
            path, columns=['entry_id', 'bravais_lattice', 'candidate_id', 'M20', 'final_rank'])
        frame['condition_bundle'] = bundle
        for _, group in frame.groupby(['entry_id', 'condition_bundle', 'bravais_lattice'],
                                      sort=False):
            pools += 1
            rows += group.shape[0]
            ordered = group.sort_values('final_rank')['M20'].to_numpy(dtype=np.float64)
            if ordered.size < 2:
                continue
            step = np.diff(ordered)
            ties += int(np.sum(step == 0.0))
            rising = step > 0.0
            if rising.any():
                violations += 1
                worst = max(worst, float(step[rising].max()))
    return dict(pools=pools, rows=rows, violations=violations, ties=ties, worst_rise=worst)


def in_top_n_consistency(pool):
    """`in_top_n` must be exactly `final_rank < 20`.

    It is what `pool_subset='in_top_n'` selects on, so a flag that has drifted from the rank it
    claims to summarise changes which pool every 'as the pipeline reports it' number describes,
    without changing anything visible.
    """
    disagreements = rows = 0
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        frame = pd.read_parquet(path, columns=['final_rank', 'in_top_n'])
        rows += frame.shape[0]
        expected = frame['final_rank'].to_numpy() < N_TOP_CANDIDATES
        disagreements += int(np.sum(frame['in_top_n'].to_numpy(dtype=bool) != expected))
    return dict(rows=rows, disagreements=disagreements)


def _row(check, what, value, tolerance, passed, detail=''):
    return dict(check=check, what=what, value=value, tolerance=tolerance,
                verdict='PASS' if passed else 'FAIL', detail=detail)


def build_rows(pool, threshold):
    rows = []

    rank = rank_reproduction(pool)
    rate = rank['violations']/max(rank['pools'], 1)
    rows.append(_row(
        'rank_order', "M20 is non-increasing along the pipeline's own final_rank",
        rate, EXACT, rank['violations'] == 0,
        f"{rank['violations']} of {rank['pools']} (entry, bundle, lattice) pools violate it over "
        f"{rank['rows']} rows; {rank['ties']} adjacent ties; worst rise {rank['worst_rise']:.3e}"))

    flag = in_top_n_consistency(pool)
    rows.append(_row(
        'in_top_n', f'in_top_n equals final_rank < {N_TOP_CANDIDATES}',
        flag['disagreements']/max(flag['rows'], 1), EXACT, flag['disagreements'] == 0,
        f"{flag['disagreements']} disagreements over {flag['rows']} rows"))

    everything = FomMetrics.evaluate(pool, score='M20', threshold=threshold, n_bootstrap=0)
    reported = FomMetrics.evaluate(pool, score='M20', threshold=threshold, n_bootstrap=0,
                                   pool_subset='in_top_n')

    gap = abs(everything.metric('operating_point') - reported.metric('operating_point'))
    rows.append(_row(
        'pool_depth', 'the operating point is identical over all survivors and over the top 20/lattice',
        gap, EXACT, gap == 0.0,
        f"all {everything.metric('operating_point'):.6f} against in_top_n "
        f"{reported.metric('operating_point'):.6f}; ceilings differ as they should, "
        f"{everything.metric('ceiling_rescorer'):.4f} against "
        f"{reported.metric('ceiling_rescorer'):.4f}"))

    for name, left, right in (('reranker', 'ceiling_reranker', 'threshold_only'),
                              ('rescorer', 'ceiling_rescorer', 'found')):
        difference = abs(everything.metric(left) - everything.metric(right))
        rows.append(_row(
            f'ceiling_{name}', f'{left} is identically {right}', difference, EXACT,
            difference == 0.0, f'{everything.metric(left):.6f}'))

    aggregate = everything.aggregate.iloc[0]
    total = float(sum(aggregate[bucket] for bucket in
                      ('operating_point', 'lost_rank_failure', 'lost_threshold_failure',
                       'lost_both', 'lost_not_found', 'degenerate_only')))
    rows.append(_row(
        'loss_decomposition', 'the five outcome buckets and the degenerates partition the entries',
        abs(total - 1.0), 1e-12, abs(total - 1.0) < 1e-12, f'they sum to {total:.12f}'))

    entries = FomBenchmark.load_entries(pool)
    if 'is_degenerate' in entries.columns:
        per_entry = entries.drop_duplicates(subset=['entry_id'])
        expected = float(pd.Series(per_entry['is_degenerate']).fillna(False).mean())
        seen = float(everything.metric('degenerate_only'))
        # Not an identity: `degenerate_only` is over (entry x bundle) cells and only counts an
        # entry whose *only* correct candidates are degenerate, so it is bounded above by the
        # entry rate rather than equal to it. What would be wrong is zero against a non-zero rate,
        # which is the C2-F-080 failure mode -- the flag present but never reaching the candidates.
        ok = seen <= expected + 1e-9 and (expected == 0.0 or seen > 0.0)
        rows.append(_row(
            'degeneracy', 'the degenerate rate is non-zero and bounded by the entry table',
            seen, expected, ok,
            f'{seen:.4f} of cells against {expected:.4f} of source entries'))

    rows.append(_row(
        'rank_exactness', 'the pool certifies rank metrics for this score',
        1.0 if everything.meta['ranks_exact'] else 0.0, 1.0,
        bool(everything.meta['ranks_exact']),
        f"K={everything.meta['subsample_top_k']}, "
        f"{everything.meta['rank_exactness'] or 'exact'}"))
    return rows, everything


def main(argv=None):
    args = _parse_args(argv)
    rows, result = build_rows(args.pool, args.threshold)
    table = pd.DataFrame(rows)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f'{args.tag}.csv'
    table.to_csv(path, index=False)

    provenance = dict(pool=str(args.pool), threshold=float(args.threshold),
                      n_cells=int(result.meta['n_entries']),
                      n_source_entries=int(result.meta['n_clusters']),
                      n_candidates=int(result.meta['n_candidates_seen']),
                      bundles=result.meta['bundles'],
                      entry_digest=result.meta['entry_digest'],
                      subsample_top_k=result.meta['subsample_top_k'])
    with open(artifact_dir / f'{args.tag}.json', 'w', encoding='utf-8') as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    print(table[['check', 'value', 'tolerance', 'verdict']].to_string(index=False))
    print(f'\n{result.meta["n_candidates_seen"]} candidates, {result.meta["n_entries"]} cells, '
          f'{result.meta["n_clusters"]} source entries')
    print(f'wrote {path}')
    failed = table.loc[table['verdict'] == 'FAIL']
    if failed.shape[0]:
        for row in failed.itertuples():
            print(f'FAIL {row.check}: {row.detail}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
