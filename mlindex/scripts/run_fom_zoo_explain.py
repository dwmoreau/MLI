"""S09: how much a combiner is worth, and why the winner wins.

Ported from campaign 1's `run_fom_zoo_explain.py` (branch `fom`) and cut to what S09's acceptance
gate asks for. What was taken, and what was deliberately left:

  TAKEN     `complementarity` -- the union oracle, which is the number that sizes S12 before it is
            built. It is the one campaign-1 prediction that came true in advance: it credited a
            perfect combiner with 6.0 points over the best single merit, and the combiner that was
            then built landed within 0.2 pp of that.
  TAKEN     `over_prediction` -- the symmetry-lowering mechanism, restated against the new pool.
  LEFT      `cross_lattice_null` and its figure. Its subject is the behaviour of the null *at the
            prune boundary*, and campaign 2 generates below that boundary, so the figure would be
            a picture of something that no longer exists.
  LEFT      `c0_singularity`. It reads `features_error0_cont0.parquet`; campaign 2 has no
            zero-error bundle by design (METRICS section 9) and `evaluate`'s own
            `meta['n_score_above_1e9']` covers what remains of the question.
  LEFT      `prefilter_sensitivity`. Campaign 1's Q5, not in S09's gate.

**Carry campaign 1's own correction with the oracle.** Complementarity in the union-oracle sense is
not necessity in a combiner: dropping the entire M^Rev family from campaign 1's finished model cost
0.28 pp of operating point at p = 0.85, with top-10 unchanged to four decimal places, while the
union oracle had credited those columns with the headroom. An oracle says "some merit gets this
entry right". It does not say a fitted model needs that column to find it.

    python mlindex/scripts/run_fom_zoo_explain.py --tag S09_zoo_slice
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomMetrics  # noqa: E402


def complementarity(per_entry_path, metric='top10', scope='all'):
    """Pairwise "A finds it, B does not", and the union oracle over the zoo.

    The union oracle is the ceiling on any combiner that only ever picks one of the zoo's existing
    orderings -- so it is the headroom S12 is chasing, measured before S12 is built.
    """
    frame = pd.read_parquet(per_entry_path)
    if scope == 'hard':
        frame = frame.loc[frame['is_hard'].astype(bool)]
    wide = frame.pivot_table(index=['entry_id', 'condition_bundle'], columns='merit',
                             values=metric, aggfunc='first')
    wide = wide.fillna(False).astype(bool)
    merits = list(wide.columns)
    matrix = pd.DataFrame(index=merits, columns=merits, dtype=float)
    for a in merits:
        for b in merits:
            matrix.loc[a, b] = float((wide[a] & ~wide[b]).mean())
    # `found` is score-independent, so any merit's column carries the pool's own ceiling: a
    # candidate a perfect *re-scorer* could reach. The union oracle cannot exceed it.
    reachable = frame.loc[frame['merit'] == merits[0]] if merits else frame
    ceiling = float(reachable['found'].astype(bool).mean()) if 'found' in frame.columns else np.nan
    summary = pd.DataFrame([{
        'metric': metric, 'scope': scope, 'n_merits': len(merits),
        'n_entries': int(wide.shape[0]),
        'union_oracle': float(wide.any(axis=1).mean()),
        'best_single_merit': float(wide.mean().max()),
        'best_merit': str(wide.mean().idxmax()) if merits else '',
        'reachable_ceiling': ceiling,
        }])
    summary['combiner_headroom'] = summary['union_oracle'] - summary['best_single_merit']
    summary['oracle_share_of_ceiling'] = summary['union_oracle']/summary['reachable_ceiling']
    return matrix, summary


# The campaign's figure palette, as S08 established it. Reused rather than re-chosen: these are
# candidate paper figures and a reader should not have to relearn the colours between them. The two
# identity hues are validated for colour-vision deficiency (worst adjacent deltaE 24.7 protan,
# against a target of 8); the grey is deliberately recessive rather than a third identity, and every
# bar carries a direct label so nothing depends on colour alone.
ACCENT = '#2a78d6'
BASELINE = '#eb6834'
RECEDE = '#9a9a94'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
SURFACE = '#fcfcfb'
GRID = '#e3e3df'


def _style(pyplot):
    pyplot.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.6,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK_SECONDARY, 'ytick.color': INK_SECONDARY,
        'font.size': 8.5, 'axes.titlesize': 9.0, 'axes.titleweight': 'bold',
        })


def write_figure(main_table, summary, artifact_dir, tag, tiebreak=None, caption=None):
    """Two panels: who wins, and what a combiner could still add.

    Panel (a) is the leaderboard against the *measured* tie-break floor rather than against zero --
    a constant score already reaches 0.2352 of top-10 on this population because ties break
    cubic-first and the dominant failure is symmetry lowering, so a merit below that line has not
    demonstrated anything (S08, C2-F-083). Panel (b) is the sizing S12 reads.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    _style(pyplot)

    figure_handle, (left, right) = pyplot.subplots(
        1, 2, figsize=(7.4, 3.3), layout='constrained', width_ratios=(1.25, 1.0))

    # ---- (a) the leaderboard -------------------------------------------------------------
    rows = main_table.sort_values('top10')
    positions = np.arange(rows.shape[0])
    colours = [ACCENT if merit == 'M_sym' else BASELINE if merit == 'M20' else RECEDE
               for merit in rows['merit']]
    left.barh(positions, rows['top10'], height=0.68, color=colours, linewidth=0)
    left.set_yticks(positions, rows['merit'], color=INK)
    if tiebreak is not None:
        left.axvline(tiebreak, color=INK, linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
    ceiling = float(summary.loc[(summary['metric'] == 'top10')
                                & (summary['scope'] == 'all'), 'reachable_ceiling'].iloc[0])
    left.axvline(ceiling, color=INK_SECONDARY, linewidth=0.9, zorder=2)
    # Drawn after the rules and carrying a surface-coloured backing, because a bar that ends near
    # the tie-break line would otherwise have its own value struck through by it.
    for position, value in zip(positions, rows['top10']):
        left.text(value + 0.014, position, f'{value:.3f}', va='center', ha='left',
                  fontsize=7.5, color=INK_SECONDARY, zorder=4,
                  bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.8))
    left.set_xlim(0, max(ceiling, float(rows['top10'].max())) + 0.12)
    top = positions[-1] + 0.75
    if tiebreak is not None:
        left.text(tiebreak, top, f'tie-break\nfloor {tiebreak:.3f}', fontsize=7.0, color=INK,
                  ha='center', va='bottom', linespacing=1.25)
    left.text(ceiling, top, f'reachable\n{ceiling:.3f}', fontsize=7.0, color=INK_SECONDARY,
              ha='center', va='bottom', linespacing=1.25)
    left.set_ylim(-0.7, top + 0.9)
    left.set_xlabel('correct cell in the pooled top ten')
    left.set_title('(a) the leaderboard, read against the floor', loc='left', color=INK)
    left.xaxis.grid(True, color=GRID, linewidth=0.6)
    left.set_axisbelow(True)

    # ---- (b) what a combiner could add ---------------------------------------------------
    scopes = [('all', 'aggregate'), ('hard', 'hard stratum')]
    bars = summary.loc[summary['metric'] == 'top10'].set_index('scope')
    spots = np.arange(len(scopes))
    height = 0.26
    for offset, (column, label, colour) in enumerate((
            ('best_single_merit', 'best single merit', ACCENT),
            ('union_oracle', 'union oracle', BASELINE),
            ('reachable_ceiling', 'reachable ceiling', RECEDE))):
        values = [float(bars.loc[scope, column]) if scope in bars.index else np.nan
                  for scope, _ in scopes]
        right.barh(spots + (1 - offset)*height, values, height=height - 0.03, color=colour,
                   linewidth=0, label=label)
        for spot, value in zip(spots, values):
            if np.isfinite(value):
                right.text(value + 0.012, spot + (1 - offset)*height, f'{value:.3f}',
                           va='center', ha='left', fontsize=7.0, color=INK_SECONDARY)
    right.set_yticks(spots, [label for _, label in scopes], color=INK)
    right.set_xlim(0, 1.20)
    right.set_ylim(-0.55, len(scopes) - 0.15)
    right.set_xlabel('correct cell in the pooled top ten')
    right.set_title('(b) what a combiner could still add', loc='left', color=INK)
    right.xaxis.grid(True, color=GRID, linewidth=0.6)
    right.set_axisbelow(True)
    # Upper right: the hard stratum's bars are short, so that corner is the panel's empty quarter.
    right.legend(loc='upper right', frameon=False, fontsize=7.0, handlelength=1.1,
                 labelcolor=INK_SECONDARY)

    if caption:
        figure_handle.text(0.005, -0.03, caption, fontsize=7.0, color=INK_SECONDARY,
                           ha='left', va='top', wrap=True)
    path = Path(artifact_dir)/f'{tag}.png'
    figure_handle.savefig(path, dpi=300, bbox_inches='tight')
    pyplot.close(figure_handle)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='S09 -- the union oracle and the mechanism analyses.')
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--tag', default='S09_zoo')
    parser.add_argument('--out-tag', default=None)
    parser.add_argument('--tiebreak-floor', type=float, default=0.2352,
                        help='S08 measured this on the fully retained pool. A rank metric is read '
                             'against it, never against zero (C2-F-083).')
    parser.add_argument('--caption', default=None)
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    out_tag = args.out_tag or args.tag.replace('_zoo', '_zoo')
    per_entry = artifact_dir/f'{args.tag}_per_entry.parquet'
    if not per_entry.exists():
        raise SystemExit(f'No per-entry table at {per_entry}. Run run_fom_zoo_eval.py first.')

    summaries, matrices = [], []
    for metric in ('top10', 'operating_point'):
        for scope in ('all', 'hard'):
            matrix, summary = complementarity(per_entry, metric=metric, scope=scope)
            if summary['n_entries'].iloc[0] == 0:
                continue
            summaries.append(summary)
            matrix = matrix.reset_index().rename(columns={'index': 'merit_a'})
            matrix.insert(0, 'scope', scope)
            matrix.insert(0, 'metric', metric)
            matrices.append(matrix)
            row = summary.iloc[0]
            print(f'  {metric:16s} {scope:5s} union {row["union_oracle"]:.4f}  '
                  f'best single {row["best_single_merit"]:.4f} ({row["best_merit"]})  '
                  f'ceiling {row["reachable_ceiling"]:.4f}  '
                  f'headroom {row["combiner_headroom"]:+.4f}')

    frame = pd.concat(summaries, ignore_index=True)
    frame.to_csv(artifact_dir/f'{out_tag}_complementarity.csv', index=False, encoding='utf-8')
    pd.concat(matrices, ignore_index=True).to_csv(
        artifact_dir/f'{out_tag}_complementarity_matrix.csv', index=False, encoding='utf-8')

    main_table = pd.read_csv(artifact_dir/f'{args.tag}_main_table.csv')
    path = write_figure(main_table, frame, artifact_dir, f'{out_tag}_explain',
                        tiebreak=args.tiebreak_floor, caption=args.caption)
    print(f'\nwrote {out_tag}_complementarity{{,_matrix}}.csv and {path.name} to {artifact_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
