"""S05 acceptance gate: does the metrics module reproduce S02's live measurement?

S02 measured, live on the mirror harness, an operating point of 70.01% and a ceiling of 92.75%
at C1 nominal, CNRS-weighted. Both numbers came from the pooled cross-lattice top-20-per-lattice
output of a full indexing run. The metrics module has to recover them from the frozen candidate
pool alone. If it cannot, either the pool is incomplete or the pooling logic differs, and both
must be resolved before anything downstream quotes a number.

The gate is reported in three parts, because one of them is known not to close and the reason is
recorded rather than hidden:

1. **Ceiling.** Comparable only against `pool_subset='in_top_n'`, the top-twenty-per-lattice
   subset S02's pool consisted of. The full survivor pool is deeper and its ceiling is
   correspondingly higher; that difference is a property of the dump, not a disagreement.
2. **Operating point over the eleven lattices Q27 could not have touched.** The
   `standardize_cell` write-back fix (Q27, `7ab633d`) changed `best_xnn` for the monoclinic and
   triclinic systems *after* S02 was measured, so S02's mP, mC and aP numbers no longer describe
   this pool. The complement is the like-for-like comparison.
3. **The all-lattice operating point**, with the residual attributed per lattice. Each lattice's
   contribution to the aggregate delta is `w_bl/sum(w) * delta_bl`, so the total is apportioned
   with nothing left unexplained.

    python mlindex/scripts/run_fom_metrics_gate.py \
        --benchmark-dir mlindex/data/fom_benchmark --artifact-dir docs/fom/artifacts

Run it with the laptop development env, /Users/DWMoreau/miniforge3/envs/mli/bin/python.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomMetrics  # noqa: E402


# docs/fom/artifacts/S02_mirror_verdict.json, bundle error1_cont0, commit a6cdc4f.
S02_OPERATING_POINT = 0.7001215976014005
S02_CEILING = 0.9275374010447518
GATE_TOLERANCE = 0.005
GATE_BUNDLE = 'error1_cont0'


def _parse_args():
    parser = argparse.ArgumentParser(
        description='S05 acceptance gate for the metrics module')
    parser.add_argument('--benchmark-dir', type=str, default='mlindex/data/fom_benchmark',
                        help='Consolidated pool: entries.parquet and candidates_*.parquet')
    parser.add_argument('--artifact-dir', type=str, default='docs/fom/artifacts',
                        help='Where to write the gate report and tables')
    parser.add_argument('--reference', type=str,
                        default='docs/fom/artifacts/S02_mirror_per_lattice.csv',
                        help='S02 per-lattice table to attribute the residual against')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Operating threshold on the score (de Wolff M20 > 10)')
    parser.add_argument('--top-n', type=int, default=10, help='Pooled rank cut')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', type=str, default='S05_gate')
    return parser.parse_args()


def commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                       text=True).strip()
    except Exception:
        return 'unknown'


def weighted(values, weights):
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    present = np.isfinite(values) & (weights > 0)
    if not present.any():
        return np.nan
    return float((values[present]*weights[present]).sum()/weights[present].sum())


def per_lattice_comparison(result, reference):
    """Join the per-lattice table to S02's and apportion the aggregate delta across lattices."""
    ours = result.stratum('bravais_lattice').set_index('level')
    table = pd.DataFrame({
        'bravais_lattice': list(FomMetrics.BRAVAIS_LATTICES),
        }).set_index('bravais_lattice')
    table['cnrs_total'] = [FomMetrics.CNRS_WEIGHTS[lattice] for lattice in table.index]
    table['n_entries'] = ours['n_entries']
    table['operating_point'] = ours['operating_point']
    table['ceiling_rescorer'] = ours['ceiling_rescorer']
    table['operating_point_s02'] = reference['operating_point']
    table['ceiling_s02'] = reference['found']
    table['delta_operating_point'] = table['operating_point'] - table['operating_point_s02']
    table['delta_ceiling'] = table['ceiling_rescorer'] - table['ceiling_s02']
    total_weight = table['cnrs_total'].sum()
    # Sums to the aggregate delta, so the residual is apportioned with no remainder.
    table['weighted_contribution'] = table['delta_operating_point']*table['cnrs_total']/total_weight
    table['q27_affected'] = [lattice in FomMetrics.Q27_AFFECTED_LATTICES for lattice in table.index]
    return table.reset_index()


def restricted_operating_point(table, affected):
    """CNRS-weighted operating point over the affected or unaffected lattices only."""
    rows = table.loc[table['q27_affected'] == affected]
    return (weighted(rows['operating_point'], rows['cnrs_total']),
            weighted(rows['operating_point_s02'], rows['cnrs_total']),
            float(rows['cnrs_total'].sum()))


def bundle_table(benchmark_dir, args):
    """Aggregate and hard-stratum metrics for every evaluable bundle.

    One `evaluate` call over all of them, so the hard stratum -- which spans four bundles by
    definition -- is computed on a single consistent entry set rather than stitched together.
    """
    result = FomMetrics.evaluate(
        benchmark_dir, score='M20', threshold=args.threshold, top_n=args.top_n,
        pool_subset='all', n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
    rows = []
    aggregate = result.aggregate.iloc[0].to_dict()
    aggregate['bundle'] = 'all bundles'
    aggregate['condition'] = 'aggregate'
    rows.append(aggregate)
    hard = result.hard.iloc[0].to_dict()
    hard['bundle'] = 'all bundles'
    hard['condition'] = 'hard stratum'
    rows.append(hard)
    for level, group in result.per_entry.groupby('condition_bundle', observed=True):
        row = FomMetrics._scope_row(group, True, True, scope='bundle')
        row['bundle'] = level
        row['condition'] = FomMetrics.BUNDLE_LABELS.get(level, '?')
        rows.append(row)
        hard_group = group.loc[group['is_hard']]
        if hard_group.shape[0]:
            hard_row = FomMetrics._scope_row(hard_group, True, True, scope='bundle hard')
            hard_row['bundle'] = level
            hard_row['condition'] = FomMetrics.BUNDLE_LABELS.get(level, '?') + ' hard'
            rows.append(hard_row)
    columns = ['bundle', 'condition', 'n_entries', 'operating_point', 'operating_point_ci_low',
               'operating_point_ci_high', 'ceiling_reranker', 'ceiling_rescorer',
               'headroom_reranker', 'headroom_rescorer', 'top1', 'top5', 'top10', 'mrr',
               'threshold_only', 'reported', 'false_positive', 'precision', 'off_by_two',
               'weight_coverage']
    table = pd.DataFrame(rows)
    return result, table[[column for column in columns if column in table.columns]]


def plot_gate(bundles, comparison, out_path):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axis = axes[0]
    rows = bundles.loc[bundles['condition'].str.match(r'^C\d$')].sort_values('condition')
    positions = np.arange(rows.shape[0])
    axis.plot(positions, 100*rows['ceiling_rescorer'], 'o-', color='#1f77b4',
              label='ceiling (perfect re-scorer)')
    # Dashed and hollow, because it sits within a percentage point of the operating point on
    # every bundle: re-ranking alone recovers almost nothing, and the overlap is the result.
    axis.plot(positions, 100*rows['ceiling_reranker'], '^--', color='#2ca02c',
              markerfacecolor='none', label='ceiling (perfect re-ranker)')
    axis.errorbar(positions, 100*rows['operating_point'],
                  yerr=[100*(rows['operating_point'] - rows['operating_point_ci_low']),
                        100*(rows['operating_point_ci_high'] - rows['operating_point'])],
                  fmt='s-', color='#d62728', capsize=2, label='operating point')
    for value, style, label in ((FomMetrics.CNRS_OPERATING_POINT, '-', 'CNRS operating point'),
                                (FomMetrics.CNRS_CEILING, '--', 'CNRS ceiling')):
        axis.axhline(100*value, color='0.35', linestyle=style, linewidth=1.0,
                     label=f'{label} ({100*value:.1f}%)')
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{row.condition}\n{row.bundle}" for row in rows.itertuples()],
                         fontsize=7, rotation=30, ha='right')
    axis.set_ylabel('percent of entries (CNRS-weighted)')
    axis.set_title('M20 on the frozen pool, by condition bundle')
    axis.legend(fontsize=7, loc='lower left')
    axis.grid(alpha=0.3)

    axis = axes[1]
    order = np.argsort(-comparison['cnrs_total'].to_numpy())
    rows = comparison.iloc[order]
    positions = np.arange(rows.shape[0])
    colors = ['#d62728' if affected else '#1f77b4' for affected in rows['q27_affected']]
    axis.barh(positions, 100*rows['delta_operating_point'], color=colors)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{row.bravais_lattice} (w={int(row.cnrs_total)})"
                          for row in rows.itertuples()], fontsize=8)
    axis.axvline(0.0, color='0.2', linewidth=1.0)
    axis.set_xlabel('S05 minus S02 operating point (percentage points)')
    axis.set_title('Residual by lattice; red = touched by the Q27 fix')
    axis.grid(alpha=0.3, axis='x')

    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def write_report(path, payload, comparison, bundles):
    lines = [
        '# S05 gate -- the metrics module against S02\'s live measurement',
        '',
        '*Generated by `mlindex/scripts/run_fom_metrics_gate.py`. Do not edit by hand.*',
        '',
        f"Commit `{payload['commit']}`, seed {payload['seed']}, "
        f"{payload['n_bootstrap']} bootstrap resamples over source entries.",
        '',
        f"**Gate: {'PASSED' if payload['gate_passed'] else 'PARTIAL'}.** "
        f"Ceiling within {100*GATE_TOLERANCE:.1f} pp: "
        f"{'yes' if payload['ceiling_passed'] else 'no'}. "
        f"Operating point over the eleven Q27-unaffected lattices within "
        f"{100*GATE_TOLERANCE:.1f} pp: {'yes' if payload['unaffected_passed'] else 'no'}. "
        f"All-lattice operating point within {100*GATE_TOLERANCE:.1f} pp: "
        f"{'yes' if payload['operating_point_passed'] else 'no'}.",
        '',
        '## 1. The two headline numbers',
        '',
        '| quantity | S05 on the frozen pool | S02 live | delta (pp) |',
        '|---|---|---|---|',
        f"| operating point (all lattices) | {payload['operating_point']:.4f} | "
        f"{S02_OPERATING_POINT:.4f} | {100*payload['delta_operating_point']:+.2f} |",
        f"| ceiling, `in_top_n` pool | {payload['ceiling_in_top_n']:.4f} | {S02_CEILING:.4f} | "
        f"{100*payload['delta_ceiling']:+.2f} |",
        f"| ceiling, all survivors | {payload['ceiling_all']:.4f} | {S02_CEILING:.4f} | "
        f"{100*(payload['ceiling_all'] - S02_CEILING):+.2f} |",
        '',
        'The operating point is identical under both pool definitions '
        f"({payload['operating_point']:.4f} and {payload['operating_point_all']:.4f}), as it "
        'must be: a member of the pooled top ten is necessarily inside its own lattice\'s top '
        'twenty. Pool depth therefore explains the ceiling difference and none of the '
        'operating-point residual.',
        '',
        '## 2. Where the residual is',
        '',
        f"Over the eleven lattices the Q27 `standardize_cell` fix could not have touched "
        f"(weight {payload['unaffected_weight']:.0f}/599): S05 "
        f"{payload['unaffected_operating_point']:.4f} against S02 "
        f"{payload['unaffected_operating_point_s02']:.4f}, "
        f"**{100*payload['unaffected_delta']:+.2f} pp**.",
        '',
        f"Over the three it did (weight {payload['affected_weight']:.0f}/599): S05 "
        f"{payload['affected_operating_point']:.4f} against S02 "
        f"{payload['affected_operating_point_s02']:.4f}, "
        f"**{100*payload['affected_delta']:+.2f} pp**.",
        '',
        '| lattice | w | n | S05 op | S02 op | delta (pp) | contribution (pp) | Q27 |',
        '|---|---|---|---|---|---|---|---|',
        ]
    for row in comparison.sort_values('cnrs_total', ascending=False).itertuples():
        lines.append(
            f'| {row.bravais_lattice} | {int(row.cnrs_total)} | {int(row.n_entries)} | '
            f'{row.operating_point:.4f} | {row.operating_point_s02:.4f} | '
            f'{100*row.delta_operating_point:+.2f} | {100*row.weighted_contribution:+.3f} | '
            f"{'yes' if row.q27_affected else ''} |"
            )
    lines += [
        '',
        f"The contributions sum to {100*payload['delta_operating_point']:+.2f} pp, which is the "
        'aggregate delta.',
        '',
        '## 3. M20 on every evaluable bundle',
        '',
        f"C0 (`error0_cont0`) is **excluded**: its M20 is arithmetically degenerate (F-054). "
        f"Bundles evaluated: {', '.join(payload['bundles'])}.",
        '',
        '| bundle | condition | n | op | ceiling (re-rank) | ceiling (re-score) | top1 | MRR |',
        '|---|---|---|---|---|---|---|---|',
        ]
    for row in bundles.itertuples():
        lines.append(
            f'| {row.bundle} | {row.condition} | {int(row.n_entries)} | '
            f'{row.operating_point:.4f} | {row.ceiling_reranker:.4f} | '
            f'{row.ceiling_rescorer:.4f} | {row.top1:.4f} | {row.mrr:.4f} |'
            )
    lines += [
        '',
        '## 4. The loss decomposition -- STATUS Q2',
        '',
        'Weighted and unweighted, because S02 reported a weighted operating point beside an '
        'unweighted decomposition and the two do not reconcile.',
        '',
        '| quantity | CNRS-weighted | unweighted |',
        '|---|---|---|',
        ]
    for name in ('operating_point', 'lost_rank_failure', 'lost_threshold_failure', 'lost_both',
                 'lost_not_found'):
        lines.append(f"| {name} | {payload['loss_weighted'][name]:.4f} | "
                     f"{payload['loss_unweighted'][name]:.4f} |")
    lines += [
        '',
        f"Threshold failures outnumber rank failures "
        f"{payload['loss_weighted']['lost_threshold_failure']/max(payload['loss_weighted']['lost_rank_failure'], 1e-9):.0f}"
        ':1 weighted on this bundle. **F-043: this split inverts between bundles**, so it is a '
        'function of the conditions and must not be quoted as one number.',
        '',
        '## 5. The hard stratum is mostly a generation failure',
        '',
        "The stratum the S05 handoff says carries the paper's claim -- {mP, mC, aP} x volume "
        'decile >= 8 x C2-C5 -- contains '
        f"{payload['hard_n_entries']} entry-condition rows over "
        f"{payload['hard_n_source_entries']} source entries, and "
        f"**{100*payload['hard_lost_not_found']:.1f}% of them have no correct candidate in the "
        'pool at all**. That is a candidate-generation failure, not a figure-of-merit failure, '
        'and no amount of re-ranking or re-scoring can touch it.',
        '',
        '| quantity | value |',
        '|---|---|',
        f"| rows / source entries | {payload['hard_n_entries']} / "
        f"{payload['hard_n_source_entries']} |",
        f"| rows with a reachable solution | {payload['hard_n_found']} "
        f"({payload['hard_n_source_entries_found']} source entries) |",
        f"| ceiling (perfect re-scorer) | {payload['hard_ceiling_rescorer']:.4f} "
        f"[{payload['hard_ceiling_ci'][0]:.4f}, {payload['hard_ceiling_ci'][1]:.4f}] |",
        f"| operating point | {payload['hard_operating_point']:.4f} "
        f"[{payload['hard_operating_point_ci'][0]:.4f}, "
        f"{payload['hard_operating_point_ci'][1]:.4f}] |",
        '| operating point given a reachable solution | '
        f"{payload['hard_operating_point_given_found']:.4f} |",
        f"| CNRS weight coverage | {payload['hard_weight_coverage']:.3f} |",
        '',
        'Two consequences for S06. A hard-stratum gain must be quoted as a fraction of the '
        f"{100*payload['hard_ceiling_rescorer']:.1f}% ceiling rather than in percentage points "
        'against 100. And the effective sample is the '
        f"{payload['hard_n_source_entries_found']} source entries that have a reachable "
        'solution, so the stratum cannot resolve small differences. See STATUS F-059.',
        '',
        '## 6. Policies actually applied',
        '',
        f"- C0 excluded from every metric; {payload['n_score_above_1e9']} scores above 1e9 and "
        f"{payload['n_non_finite_score']} non-finite scores seen in what remained.",
        '- Calibration: '
        + (payload['calibration_skipped_reason'] or 'computed')
        + '. Raw M20 is not a probability, so it gets no ECE or Brier score.',
        '- Volume deciles are within-lattice, reproducing the frozen split manifest exactly.',
        '- Bootstrap resamples source entries, not (entry, condition) rows and never candidates.',
        '',
        ]
    with open(path, 'w', encoding='utf-8') as report:
        report.write('\n'.join(lines) + '\n')


def main():
    args = _parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading {benchmark_dir}')
    gate = FomMetrics.evaluate(
        benchmark_dir, score='M20', threshold=args.threshold, top_n=args.top_n,
        bundles=[GATE_BUNDLE], pool_subset='in_top_n', n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        )
    deep = FomMetrics.evaluate(
        benchmark_dir, score='M20', threshold=args.threshold, top_n=args.top_n,
        bundles=[GATE_BUNDLE], pool_subset='all', n_bootstrap=0, seed=args.seed,
        )
    reference = pd.read_csv(args.reference).set_index('bravais_lattice')
    comparison = per_lattice_comparison(gate, reference)
    unaffected = restricted_operating_point(comparison, affected=False)
    affected = restricted_operating_point(comparison, affected=True)

    print('Evaluating every bundle')
    everything, bundles = bundle_table(benchmark_dir, args)

    loss_weighted = gate.loss.loc[(gate.loss['stratum'] == 'all')].iloc[0]
    unweighted_gate = FomMetrics.evaluate(
        benchmark_dir, score='M20', threshold=args.threshold, top_n=args.top_n,
        bundles=[GATE_BUNDLE], pool_subset='in_top_n', weights=None, n_bootstrap=0,
        seed=args.seed,
        )
    loss_unweighted = unweighted_gate.loss.loc[unweighted_gate.loss['stratum'] == 'all'].iloc[0]
    loss_names = ('operating_point', 'lost_rank_failure', 'lost_threshold_failure', 'lost_both',
                  'lost_not_found')
    hard_rows = everything.per_entry.loc[everything.per_entry['is_hard']]
    hard_loss = everything.loss.loc[everything.loss['stratum'] == 'hard'].iloc[0]

    payload = dict(
        commit=commit_hash(), seed=args.seed, n_bootstrap=args.n_bootstrap,
        bundle=GATE_BUNDLE, threshold=args.threshold, top_n=args.top_n,
        n_entries=int(gate.aggregate['n_entries'].iloc[0]),
        operating_point=gate.metric('operating_point'),
        operating_point_ci=[float(gate.aggregate['operating_point_ci_low'].iloc[0]),
                            float(gate.aggregate['operating_point_ci_high'].iloc[0])],
        operating_point_all=deep.metric('operating_point'),
        operating_point_unweighted=unweighted_gate.metric('operating_point'),
        ceiling_in_top_n=gate.metric('ceiling_rescorer'),
        ceiling_all=deep.metric('ceiling_rescorer'),
        ceiling_reranker=gate.metric('ceiling_reranker'),
        s02_operating_point=S02_OPERATING_POINT, s02_ceiling=S02_CEILING,
        delta_operating_point=gate.metric('operating_point') - S02_OPERATING_POINT,
        delta_ceiling=gate.metric('ceiling_rescorer') - S02_CEILING,
        unaffected_operating_point=unaffected[0], unaffected_operating_point_s02=unaffected[1],
        unaffected_weight=unaffected[2], unaffected_delta=unaffected[0] - unaffected[1],
        affected_operating_point=affected[0], affected_operating_point_s02=affected[1],
        affected_weight=affected[2], affected_delta=affected[0] - affected[1],
        hard_operating_point=everything.metric('operating_point', scope='hard'),
        hard_operating_point_ci=[float(everything.hard['operating_point_ci_low'].iloc[0]),
                                 float(everything.hard['operating_point_ci_high'].iloc[0])],
        hard_operating_point_given_found=everything.metric('operating_point_given_found',
                                                           scope='hard'),
        hard_ceiling_rescorer=everything.metric('ceiling_rescorer', scope='hard'),
        hard_ceiling_ci=[float(everything.hard['found_ci_low'].iloc[0]),
                         float(everything.hard['found_ci_high'].iloc[0])],
        hard_n_entries=int(everything.hard['n_entries'].iloc[0]),
        hard_n_found=int(everything.hard['n_found'].iloc[0]),
        hard_n_source_entries=int(hard_rows['entry_id'].nunique()),
        hard_n_source_entries_found=int(hard_rows.loc[hard_rows['found'], 'entry_id'].nunique()),
        hard_lost_not_found=float(hard_loss['lost_not_found']),
        hard_weight_coverage=float(everything.hard['weight_coverage'].iloc[0]),
        loss_weighted={name: float(loss_weighted[name]) for name in loss_names},
        loss_unweighted={name: float(loss_unweighted[name]) for name in loss_names},
        bundles=everything.meta['bundles'],
        bundles_excluded=everything.meta['bundles_excluded'],
        n_candidates_seen=everything.meta['n_candidates_seen'],
        n_non_finite_score=everything.meta['n_non_finite_score'],
        n_score_above_1e9=everything.meta['n_score_above_1e9'],
        calibration_skipped_reason=everything.meta['calibration_skipped_reason'],
        entry_digest=gate.meta['entry_digest'],
        )
    payload['ceiling_passed'] = abs(payload['delta_ceiling']) <= GATE_TOLERANCE
    payload['unaffected_passed'] = abs(payload['unaffected_delta']) <= GATE_TOLERANCE
    payload['operating_point_passed'] = abs(payload['delta_operating_point']) <= GATE_TOLERANCE
    payload['gate_passed'] = bool(payload['ceiling_passed'] and payload['unaffected_passed']
                                 and payload['operating_point_passed'])

    comparison.to_csv(artifact_dir / f'{args.tag}_per_lattice.csv', index=False,
                      encoding='utf-8')
    bundles.to_csv(artifact_dir / 'S05_metrics_by_bundle.csv', index=False, encoding='utf-8')
    everything.loss.to_csv(artifact_dir / 'S05_loss_decomposition.csv', index=False,
                           encoding='utf-8')
    everything.by_stratum.to_csv(artifact_dir / 'S05_metrics_by_stratum.csv', index=False,
                                 encoding='utf-8')
    with open(artifact_dir / f'{args.tag}.json', 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    plot_gate(bundles, comparison, artifact_dir / 'S05_metrics.png')
    write_report(artifact_dir / f'{args.tag}.md', payload, comparison, bundles)

    print(f"\nGate: {'PASSED' if payload['gate_passed'] else 'PARTIAL'}")
    print(f"  ceiling (in_top_n)      {payload['ceiling_in_top_n']:.4f} vs {S02_CEILING:.4f} "
          f"({100*payload['delta_ceiling']:+.2f} pp) "
          f"{'PASS' if payload['ceiling_passed'] else 'FAIL'}")
    print(f"  operating point, 11 BL  {payload['unaffected_operating_point']:.4f} vs "
          f"{payload['unaffected_operating_point_s02']:.4f} "
          f"({100*payload['unaffected_delta']:+.2f} pp) "
          f"{'PASS' if payload['unaffected_passed'] else 'FAIL'}")
    print(f"  operating point, all    {payload['operating_point']:.4f} vs "
          f"{S02_OPERATING_POINT:.4f} ({100*payload['delta_operating_point']:+.2f} pp) "
          f"{'PASS' if payload['operating_point_passed'] else 'FAIL'}")
    print(f"  hard stratum            op {payload['hard_operating_point']:.4f}, ceiling "
          f"{payload['hard_ceiling_rescorer']:.4f}, n={payload['hard_n_entries']}")
    print(f"\nWrote {args.tag}.json, {args.tag}.md, {args.tag}_per_lattice.csv, "
          'S05_metrics_by_bundle.csv, S05_loss_decomposition.csv, S05_metrics_by_stratum.csv, '
          f'S05_metrics.png to {artifact_dir}')


if __name__ == '__main__':
    main()
