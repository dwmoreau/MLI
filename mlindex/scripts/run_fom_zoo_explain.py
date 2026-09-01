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


def symmetry_lowering(artifact_dir, tag, merits):
    """Where a wrong candidate outranks the correct cell, is the winner of LOWER symmetry?

    Campaign 1's central mechanism claim: the failure is symmetry lowering, not volume inflation,
    and `M_rev` wins because lowering the symmetry at fixed volume increases the number of distinct
    calculated lines -- so a lower-symmetry cell over-predicts lines without over-predicting volume,
    and `M_rev` is the only classical merit that scores line over-prediction directly.

    **This needs no pool pass.** `reduce_pool` already stores the Bravais lattice of the top-ranked
    candidate and of the best correct one, so the whole analysis is arithmetic over the reduction
    -- which means it costs nothing at pool scale either.

    `FomMetrics.BRAVAIS_LATTICES` runs high symmetry to low, so a *higher* index is a
    *lower*-symmetry cell. Getting that backwards would invert the headline claim and nothing else
    would catch it, so it is asserted in a test.
    """
    rank = {lattice: position for position, lattice in enumerate(FomMetrics.BRAVAIS_LATTICES)}
    rows = []
    for merit in merits:
        path = Path(artifact_dir)/f'{tag}_reduced_{merit}_fom-dev.parquet'
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        beaten = frame.loc[frame['has_correct_all'].astype(bool)
                           & ~frame['top_is_correct_all'].astype(bool)]
        for scope, subset in (('all reachable', beaten),
                              ('hard stratum', beaten.loc[beaten['is_hard'].astype(bool)])):
            if not subset.shape[0]:
                continue
            true_rank = subset['bravais_lattice'].map(rank)
            top_rank = subset['bravais_lattice_top_all'].map(rank)
            rows.append({
                'merit': merit, 'scope': scope, 'n_entries': int(subset.shape[0]),
                'frac_wrong_cell_lower_symmetry': float((top_rank > true_rank).mean()),
                'frac_wrong_cell_higher_symmetry': float((top_rank < true_rank).mean()),
                'frac_same_lattice': float((top_rank == true_rank).mean()),
                'median_score_top': float(subset['score_top_all'].median()),
                'median_score_best_correct': float(subset['score_best_correct_all'].median()),
                })
    return pd.DataFrame(rows)


def scale_transfer(artifact_dir, tag, merits):
    """Cross-lattice against within-lattice: is the advantage fit quality, or scale?

    Campaign 1 measured ~90 % of `M_sym`'s advantage over M20 as **cross-lattice scale transfer**
    rather than better fit: within a lattice the whole zoo lands between 0.4407 and 0.6852 of
    top-10, while across lattices it spreads 0.000-0.6175. The mechanism composes with the
    symmetry-lowering result rather than competing with it -- within a lattice every candidate
    already shares the true Bravais lattice, so "the wrong winner has lower symmetry" cannot
    discriminate there, and `M_rev` collapses to about M20. **The over-prediction signal is
    inherently cross-lattice**, and that sentence is the whole explanation.

    Reads the `per_bl` reductions beside the `cross_bl` ones. Never the headline: ranking within a
    single lattice is a different and much easier problem than the one `run.py` solves
    (METRICS section 1).
    """
    rows = []
    for merit in merits:
        values = {}
        for pool, suffix in (('cross_bl', ''), ('per_bl', '_per_bl')):
            path = Path(artifact_dir)/f'{tag}{suffix}_reduced_{merit}_fom-dev.parquet'
            meta_path = Path(artifact_dir)/f'{tag}{suffix}_reduced_meta.json'
            if not (path.exists() and meta_path.exists()):
                continue
            import json
            meta = json.loads(meta_path.read_text(encoding='utf-8'))[f'{merit}|fom-dev']
            result = FomMetrics.summarise_per_entry(pd.read_parquet(path), meta, n_bootstrap=0)
            values[pool] = (float(result.metric('top10')), float(result.metric('top1')))
        if len(values) == 2:
            rows.append({'merit': merit,
                         'top10_cross_lattice': values['cross_bl'][0],
                         'top10_within_lattice': values['per_bl'][0],
                         'top1_cross_lattice': values['cross_bl'][1],
                         'top1_within_lattice': values['per_bl'][1]})
    frame = pd.DataFrame(rows)
    if frame.shape[0] and 'M20' in set(frame['merit']):
        baseline = frame.loc[frame['merit'] == 'M20'].iloc[0]
        frame['advantage_cross_lattice_pp'] = 100*(frame['top10_cross_lattice']
                                                   - baseline['top10_cross_lattice'])
        frame['advantage_within_lattice_pp'] = 100*(frame['top10_within_lattice']
                                                    - baseline['top10_within_lattice'])
        # What share of the advantage over M20 exists only across lattices. Undefined where the
        # merit has no advantage to apportion, which is most of the zoo on this pool.
        total = frame['advantage_cross_lattice_pp']
        frame['share_of_advantage_cross_lattice'] = np.where(
            np.abs(total) > 1e-9,
            1.0 - frame['advantage_within_lattice_pp']/total.replace(0, np.nan), np.nan)
    return frame


def floor_arm(artifact_dir, tag):
    """Floored against unfloored `M_sym`, on a fully retained pool -- the only place it is honest.

    The S09 handoff requires `M_sym` reported both ways, because campaign 1's number and S03's
    stored columns were computed unfloored and a floored `M_sym` is not like for like against them.

    **It cannot be run on Benchmark B** (C2-F-084). The negative subsampler ranked on the *floored*
    merit, so a saturated fit scored 0.0, ranked last, and was retained at only the 5 % Bernoulli
    rate; unfloored, those same rows read 1e11-1e14 and rank *first*. The unfloored arm would
    therefore be scored against a field with its own strongest rivals deleted, and would come out
    flattered -- understating what the floor is worth, which is the opposite of a conservative
    error. On a fully retained pool nothing was deleted and the comparison means what it says.

    Rank metrics only: the retained pool is `fom-dev` alone, so no threshold can be selected on it.
    """
    import json
    rows = []
    meta_path = Path(artifact_dir)/f'{tag}_reduced_meta.json'
    if not meta_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    metas = json.loads(meta_path.read_text(encoding='utf-8'))
    results = {}
    for merit in ('M20', 'M_sym', 'M_sym_unfloored'):
        path = Path(artifact_dir)/f'{tag}_reduced_{merit}_fom-dev.parquet'
        key = f'{merit}|fom-dev'
        if not (path.exists() and key in metas):
            continue
        meta = metas[key]
        if meta.get('subsampled'):
            raise ValueError(
                f'{tag} is a SUBSAMPLED pool. The unfloored arm is not interpretable there '
                f'(C2-F-084); run it on a fully retained pool.')
        result = FomMetrics.summarise_per_entry(pd.read_parquet(path), meta, n_bootstrap=1000)
        results[merit] = result
        for scope, block in (('all', result.aggregate), ('hard', result.hard)):
            if block.shape[0]:
                row = block.iloc[0]
                rows.append({'merit': merit, 'scope': scope, 'n_entries': row['n_entries'],
                             'n_found': row['n_found'], 'top1': row['top1'],
                             'top10': row['top10'], 'mrr': row['mrr'],
                             'ceiling_rescorer': row['ceiling_rescorer']})
    tests = []
    if 'M_sym' in results and 'M_sym_unfloored' in results:
        for scope in (None, 'hard'):
            for metric in ('top1', 'top10'):
                test = FomMetrics.mcnemar(results['M_sym'], results['M_sym_unfloored'],
                                          metric=metric, subset=scope)
                interval = FomMetrics.paired_delta_ci(
                    results['M_sym'], results['M_sym_unfloored'], metric=metric, subset=scope)
                tests.append({'comparison': 'M_sym (floored) - M_sym (unfloored)',
                              'scope': scope or 'all', **dict(test),
                              'ci_low': interval['ci_low'], 'ci_high': interval['ci_high']})
    return pd.DataFrame(rows), pd.DataFrame(tests)


HARD_COUNTS = ('X_N', 'n_over', 'max_gap')
SOFT_COUNTS = ('X_N_soft', 'n_over_soft', 'max_gap_soft')


def counting_arm(artifact_dir, tag, reference=('M20', 'M_sym')):
    """Hard against posterior-based counting merits, on a fully retained pool (C2-Q-025).

    **Why a fully retained pool is the only honest venue.** The soft counts were not in the
    subsampler's retention rule, so on Benchmark B the candidates that would outrank a correct one
    under them were kept at 5 % and every rank metric comes out optimistic (C2-R-013). This is the
    remedy that rebuild row names.

    **Why the comparison is not just "which scores higher".** DWMM's objection is that a merit built
    on a poor statistic cannot be informative however well it counts, and that these merits were
    never meant to work alone. So three things are reported: the standalone rank metric, the tie
    structure -- the mechanism C2-F-095 identified -- and the union-oracle contribution, which is
    what a merit is worth *in combination* and is the question that actually bears on S12.
    """
    import json
    meta_path = Path(artifact_dir)/f'{tag}_reduced_meta.json'
    if not meta_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    metas = json.loads(meta_path.read_text(encoding='utf-8'))

    rows, flags = [], {}
    for merit in list(reference) + list(HARD_COUNTS) + list(SOFT_COUNTS):
        key = f'{merit}|fom-dev'
        path = Path(artifact_dir)/f'{tag}_reduced_{merit}_fom-dev.parquet'
        if not (path.exists() and key in metas):
            continue
        meta = metas[key]
        if meta.get('subsampled'):
            raise ValueError(
                f'{tag} is a SUBSAMPLED pool. A merit outside the retention rule is not '
                f'rank-exact there (C2-R-013); run this on a fully retained pool.')
        frame = pd.read_parquet(path)
        result = FomMetrics.summarise_per_entry(frame, meta, n_bootstrap=0)
        ties = frame['n_ties_at_best_correct_all'].replace(0, np.nan)
        rows.append({
            'merit': merit,
            'family': ('reference' if merit in reference
                       else 'hard' if merit in HARD_COUNTS else 'soft'),
            'n_entries': int(result.metric('n_entries')),
            'top1': float(result.metric('top1')), 'top10': float(result.metric('top10')),
            'mrr': float(result.metric('mrr')),
            'median_ties_at_best_correct': float(ties.median()),
            'median_pool_size': float(frame['n_candidates_all'].median()),
            })
        flags[merit] = result.per_entry.set_index(
            ['entry_id', 'condition_bundle']).sort_index()['top10'].astype(bool)

    if not flags:
        return pd.DataFrame(rows), pd.DataFrame()

    wide = pd.DataFrame(flags)
    present = [m for m in reference if m in wide.columns]
    oracle = []
    for label, members in (('reference only', present),
                           ('reference + hard counts',
                            present + [m for m in HARD_COUNTS if m in wide.columns]),
                           ('reference + soft counts',
                            present + [m for m in SOFT_COUNTS if m in wide.columns]),
                           ('everything', list(wide.columns))):
        if not members:
            continue
        oracle.append({'set': label, 'n_merits': len(members),
                       'union_oracle_top10': float(wide[members].any(axis=1).mean())})
    oracle = pd.DataFrame(oracle)
    if oracle.shape[0]:
        base = float(oracle.loc[oracle['set'] == 'reference only', 'union_oracle_top10'].iloc[0])
        oracle['adds_over_reference_pp'] = 100*(oracle['union_oracle_top10'] - base)
    return pd.DataFrame(rows), oracle


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
    parser.add_argument('--counting-arm-tag', default=None,
                        help='Tag of a FULLY RETAINED pool reduction carrying both the hard and '
                             'the posterior-based counting merits (C2-Q-025).')
    parser.add_argument('--floor-arm-tag', default=None,
                        help='Tag of a FULLY RETAINED pool reduction carrying M_sym_unfloored. '
                             'Reports the floored/unfloored comparison the handoff requires; '
                             'refuses a subsampled pool (C2-F-084).')
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

    if args.counting_arm_tag:
        arm, oracle = counting_arm(artifact_dir, args.counting_arm_tag)
        if arm.shape[0]:
            arm.to_csv(artifact_dir/f'{out_tag}_counting_arm.csv', index=False, encoding='utf-8')
            oracle.to_csv(artifact_dir/f'{out_tag}_counting_oracle.csv', index=False,
                          encoding='utf-8')
            print('\nhard against posterior-based counting merits, fully retained pool')
            for row in arm.itertuples():
                print(f'  {row.merit:14s} {row.family:9s} top1 {row.top1:.4f}  '
                      f'top10 {row.top10:.4f}  mrr {row.mrr:.4f}  '
                      f'median ties {row.median_ties_at_best_correct:8.0f} '
                      f'of {row.median_pool_size:.0f}')
            print('\n  union oracle, top-10')
            for row in oracle.itertuples():
                print(f'    {row.set:26s} {row.union_oracle_top10:.4f}  '
                      f'({row.adds_over_reference_pp:+.3f} pp over the reference merits)')

    if args.floor_arm_tag:
        arm, tests = floor_arm(artifact_dir, args.floor_arm_tag)
        if arm.shape[0]:
            arm.to_csv(artifact_dir/f'{out_tag}_floor_arm.csv', index=False, encoding='utf-8')
            tests.to_csv(artifact_dir/f'{out_tag}_floor_arm_mcnemar.csv', index=False,
                         encoding='utf-8')
            print('\nthe M_rev support floor, on a fully retained pool')
            for row in arm.itertuples():
                print(f'  {row.merit:18s} {row.scope:5s} top1 {row.top1:.4f}  '
                      f'top10 {row.top10:.4f}  mrr {row.mrr:.4f}  (n={int(row.n_entries)})')
            for row in tests.itertuples():
                print(f'  floored - unfloored  {row.scope:5s} {row.metric:6s} '
                      f'{row.delta:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]  '
                      f'{int(row.n_a_only)}/{int(row.n_b_only)}  p {row.p_value:.3g}')

    frame = pd.concat(summaries, ignore_index=True)
    frame.to_csv(artifact_dir/f'{out_tag}_complementarity.csv', index=False, encoding='utf-8')
    pd.concat(matrices, ignore_index=True).to_csv(
        artifact_dir/f'{out_tag}_complementarity_matrix.csv', index=False, encoding='utf-8')

    main_table = pd.read_csv(artifact_dir/f'{args.tag}_main_table.csv')
    merits = list(main_table['merit'])

    symmetry = symmetry_lowering(artifact_dir, args.tag, merits)
    if symmetry.shape[0]:
        symmetry.to_csv(artifact_dir/f'{out_tag}_symmetry.csv', index=False, encoding='utf-8')
        print('\nsymmetry lowering, where a wrong candidate outranks the correct cell')
        for row in symmetry.itertuples():
            print(f'  {row.merit:16s} {row.scope:14s} n={row.n_entries:4d}  '
                  f'lower symmetry {row.frac_wrong_cell_lower_symmetry:.3f}  '
                  f'same lattice {row.frac_same_lattice:.3f}')

    transfer = scale_transfer(artifact_dir, args.tag, merits)
    if transfer.shape[0]:
        transfer.to_csv(artifact_dir/f'{out_tag}_scale_transfer.csv', index=False,
                        encoding='utf-8')
        print('\ncross-lattice against within-lattice top-10')
        for row in transfer.itertuples():
            print(f'  {row.merit:16s} cross {row.top10_cross_lattice:.4f}  '
                  f'within {row.top10_within_lattice:.4f}')
    else:
        print('\n(no per_bl reductions found -- run the eval driver with --pool-mode per_bl '
              'to size the cross-lattice half)')
    path = write_figure(main_table, frame, artifact_dir, f'{out_tag}_explain',
                        tiebreak=args.tiebreak_floor, caption=args.caption)
    print(f'\nwrote {out_tag}_complementarity{{,_matrix}}.csv and {path.name} to {artifact_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
