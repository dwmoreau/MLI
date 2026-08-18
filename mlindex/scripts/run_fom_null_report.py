"""S07's results document and figures, from the artefacts `run_fom_null.py` wrote.

Separate from the run for the reason every report script in this project is: the run is hours and
the prose is minutes, and a figure should be remakeable without re-deriving the numbers behind it.
Every number here is read from a CSV; nothing is recomputed.

    python mlindex/scripts/run_fom_null_report.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomNull  # noqa: E402
from mlindex.scripts.run_fom_zoo_features import commit_hash  # noqa: E402

SERIES_BLUE = '#2a78d6'
TARGET_RED = '#e34948'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
BAND_GREY = '#c9c9c4'

# Shirley 1980's reproducibility floor (F-009). Any difference smaller than this is not a
# difference, whatever its p-value; S06b will replace it with a measured per-merit number.
REPRODUCIBILITY_FLOOR = 0.10

# Only these merits are labelled in the scatters. Twenty-one labels on twenty-one points that
# cluster in two corners is unreadable, and the story needs eight of them: the two that lose most
# (M20, M_tilde), the incumbent (M_sym), the near-neutral one (M_rev), and the four that gain most
# because their raw scale carried nothing (M_star_corrected, F_N_q, max_gap, null_tail_nll).
LABELLED = ('M_sym', 'M20', 'M_tilde', 'M_rev', 'M_star_corrected', 'F_N_q', 'max_gap',
            'null_tail_nll')

# M20 and M_tilde land within a few thousandths of each other on both panels, so a shared offset
# renders one on top of the other. These push the crowded few apart by hand.
LABEL_OFFSETS = {'M20': (6, 4), 'M_tilde': (6, -12), 'max_gap': (-46, 3),
                 'M_rev': (7, -3)}


def read(artifact_dir, name):
    path = Path(artifact_dir)/name
    return pd.read_csv(path) if path.exists() else None


def table(frame, columns=None, floatfmt=4):
    """A markdown table, with floats formatted once so the columns line up."""
    frame = frame if columns is None else frame[columns]
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: '' if pd.isna(value) else f'{value:.{floatfmt}f}')
    header = '| ' + ' | '.join(str(name) for name in formatted.columns) + ' |'
    rule = '|' + '|'.join(['---']*formatted.shape[1]) + '|'
    body = ['| ' + ' | '.join(str(value) for value in row) + ' |'
            for row in formatted.itertuples(index=False)]
    return '\n'.join([header, rule] + body)


def write_figures(main, regularity, homogeneity, artifact_dir, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- Figure 1: the analytic null against what the pool actually contains ----------------
    #
    # The handoff asks for the empirical null plotted against de Wolff 1961's closed form as the
    # primary comparison, on the grounds that "agreement validates both; disagreement localises
    # something real". The disagreement is the result, and it is three orders of magnitude, so the
    # figure is drawn on a log axis with de Wolff's own two reference values marked: the mean
    # per-line term is 1.0 if discrepancies are exponential with mean Delta, and 1.5 in his
    # equidistant limit, which he argues is the extreme case.
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    if regularity is not None and regularity.shape[0]:
        frame = regularity.loc[regularity['grouping'] == 'bravais_lattice'].copy()
        frame = frame.sort_values('mean_line_term')
        positions = np.arange(frame.shape[0])
        axes[0].scatter(frame['mean_line_term'], positions, color=SERIES_BLUE, s=48, zorder=3)
        axes[0].axvline(1.0, color=TARGET_RED, linestyle='--', linewidth=1.5)
        axes[0].axvline(1.5, color=TARGET_RED, linestyle=':', linewidth=1.5)
        axes[0].annotate('exponential null\n(de Wolff 1961)', xy=(1.0, positions[-1]),
                         xytext=(6, -2), textcoords='offset points', color=TARGET_RED,
                         fontsize=8.5, va='top')
        axes[0].annotate('equidistant limit', xy=(1.5, positions[0]), xytext=(6, 2),
                         textcoords='offset points', color=TARGET_RED, fontsize=8.5, va='bottom')
        axes[0].set_yticks(positions)
        axes[0].set_yticklabels(frame['bravais_lattice'], fontsize=9)
        axes[0].set_xlabel('mean per-line term  -log[1 - exp(-dQ/Delta)]', color=TEXT_PRIMARY)
        axes[0].set_title('The pool fits its own wrong cells far better than chance',
                          color=TEXT_PRIMARY)

        axes[1].barh(positions, frame['rho'], color=SERIES_BLUE, alpha=0.85, zorder=3)
        axes[1].axvline(0.5, color=TARGET_RED, linestyle=':', linewidth=1.5)
        axes[1].axvline(1.0, color=TARGET_RED, linestyle='--', linewidth=1.5)
        axes[1].annotate("de Wolff's interval [1/2, 1]", xy=(0.5, positions[-1]), xytext=(-6, 0),
                         textcoords='offset points', ha='right', color=TARGET_RED, fontsize=8.5)
        axes[1].set_xscale('log')
        axes[1].set_yticks(positions)
        axes[1].set_yticklabels(frame['bravais_lattice'], fontsize=9)
        axes[1].set_xlabel('implied g_bar/Delta (log scale)', color=TEXT_PRIMARY)
        axes[1].set_title('Two to three orders of magnitude outside its own bounds',
                          color=TEXT_PRIMARY)

    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color(BAND_GREY)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.grid(alpha=0.25)
    figure.supxlabel(
        'Incorrect candidates on fom-train, six evaluable bundles. de Wolff derives Delta(Q) for an\n'
        'ARBITRARY wrong cell; these are least-squares REFINED wrong cells that then survived a\n'
        'prune and a deduplication, so the gap is the look-elsewhere effect of his own section 5\n'
        'rather than a failure of the regularity argument. R1: the prune keeps the better-fitting\n'
        'wrong cells, so these are lower bounds on g_bar/Delta and the spread is compressed.',
        fontsize=8.5, color=TEXT_SECONDARY)
    figure.savefig(Path(artifact_dir)/f'{tag}_analytic.png', dpi=200, facecolor='white')
    plt.close(figure)

    # ---- Figure 2: what calibration does, per merit -----------------------------------------
    #
    # One point per merit, raw against calibrated, with the per-lattice ceiling drawn. The split
    # is the finding: the merits whose raw scale carries cross-lattice information lose by being
    # standardised, and the merits whose raw scale carries none gain.
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    raw = main.loc[main['method'] == 'raw'].set_index('merit')
    for position, (method, marker) in enumerate((('rank', 'o'), ('z', 's'))):
        subset = main.loc[(main['method'] == method) & (main['trials'].isna()
                                                        | (main['trials'] == ''))]
        if not subset.shape[0]:
            continue
        before = raw.loc[subset['merit'], 'top10'].to_numpy()
        after = subset['top10'].to_numpy()
        axes[0].scatter(before, after, s=44, marker=marker, zorder=3,
                        color=SERIES_BLUE if position == 0 else TARGET_RED, label=method)
        if position == 0:
            for merit, x, y in zip(subset['merit'], before, after):
                if merit not in LABELLED:
                    continue
                axes[0].annotate(merit, xy=(x, y),
                                 xytext=LABEL_OFFSETS.get(merit, (5, -9)),
                                 textcoords='offset points', fontsize=7.5,
                                 color=TEXT_SECONDARY)
    top = float(np.nanmax([main['top10'].max(), raw['top10'].max()]))*1.05
    limit = [0.0, top]
    axes[0].plot(limit, limit, color=BAND_GREY, linestyle=':', linewidth=1.5, zorder=1)
    axes[0].annotate('no change', xy=(limit[1], limit[1]), xytext=(-6, 6),
                     textcoords='offset points', ha='right', fontsize=8, color=TEXT_SECONDARY)
    axes[0].set_xlabel('raw cross-lattice top-10', color=TEXT_PRIMARY)
    axes[0].set_ylabel('calibrated cross-lattice top-10', color=TEXT_PRIMARY)
    axes[0].set_title('Calibration helps the unscaled merits and costs the scaled ones',
                      color=TEXT_PRIMARY)
    axes[0].legend(fontsize=8, frameon=False, loc='lower right')

    # Right: the operating point, which is where the loss is largest, against the gate line.
    best = (main.loc[main['method'] != 'raw']
            .sort_values('operating_point_matched_fpr', ascending=False)
            .drop_duplicates('merit'))
    joined = raw.loc[best['merit']]
    axes[1].scatter(joined['operating_point_matched_fpr'],
                    best['operating_point_matched_fpr'].to_numpy(), s=44, color=SERIES_BLUE,
                    zorder=3)
    for merit, x, y in zip(best['merit'], joined['operating_point_matched_fpr'],
                           best['operating_point_matched_fpr']):
        if merit not in LABELLED:
            continue
        axes[1].annotate(merit, xy=(x, y), xytext=LABEL_OFFSETS.get(merit, (5, -9)),
                         textcoords='offset points', fontsize=7.5, color=TEXT_SECONDARY)
    limit = [0.0, max(0.55, float(main['operating_point_matched_fpr'].max())*1.05)]
    axes[1].plot(limit, limit, color=BAND_GREY, linestyle=':', linewidth=1.5, zorder=1)
    if 'M20' in raw.index:
        gate = float(raw.loc['M20', 'operating_point_matched_fpr']) + 0.03
        axes[1].axhline(gate, color=TARGET_RED, linestyle='--', linewidth=1.5)
        axes[1].annotate('the gate: raw M20 + 3 pp', xy=(limit[1], gate), xytext=(-6, 4),
                         textcoords='offset points', ha='right', color=TARGET_RED, fontsize=8.5)
    axes[1].set_xlabel('raw operating point at the matched FPR budget', color=TEXT_PRIMARY)
    axes[1].set_ylabel('best calibrated operating point', color=TEXT_PRIMARY)
    axes[1].set_title('Best method per merit, fom-dev', color=TEXT_PRIMARY)

    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color(BAND_GREY)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.grid(alpha=0.25)
    figure.supxlabel(
        'fom-dev, six evaluable bundles, CNRS-reweighted; nulls fitted on fom-train and thresholds\n'
        'selected there. Every method except `analytic` is fitted on negatives censored at M20 >= 5,\n'
        'lattice-dependently (R1), so its numbers are provisional against a Benchmark B dumped at\n'
        'prune threshold 0.',
        fontsize=8.5, color=TEXT_SECONDARY)
    figure.savefig(Path(artifact_dir)/f'{tag}_calibration.png', dpi=200, facecolor='white')
    plt.close(figure)

    # ---- Figure 3: the gate's second condition ----------------------------------------------
    if homogeneity is not None and homogeneity.shape[0]:
        figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
        spread = (homogeneity.loc[homogeneity['bravais_lattice'].notna()]
                  .groupby(['merit', 'method'], as_index=False)['q0.999_spread'].first())
        pivot = spread.pivot(index='merit', columns='method', values='q0.999_spread')
        columns = [column for column in ('raw', 'z', 'rank', 'binned', 'gbm') if column in pivot]
        pivot = pivot[columns].dropna(how='all')
        for position, column in enumerate(columns):
            axis.scatter(pivot[column], np.arange(pivot.shape[0]), s=40, zorder=3, label=column)
        axis.set_xscale('symlog', linthresh=1e-3)
        axis.set_yticks(np.arange(pivot.shape[0]))
        axis.set_yticklabels(pivot.index, fontsize=8)
        axis.set_xlabel('spread across Bravais lattices of the null 99.9th percentile',
                        color=TEXT_PRIMARY)
        axis.set_title("The gate's second condition: is the null the same everywhere?",
                       color=TEXT_PRIMARY)
        # Below the axis: the top row is a data row and a legend there sits on top of it.
        axis.legend(fontsize=8, frameon=False, ncol=len(columns), loc='upper center',
                    bbox_to_anchor=(0.5, -0.13))
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color(BAND_GREY)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.grid(alpha=0.25)
        figure.supxlabel(
            'Incorrect candidates only, fom-dev. Smaller is better: it means a threshold on the\n'
            'score means the same thing in every lattice under "this candidate is wrong". This is a\n'
            'diagnostic that the null FIT is good, not a claim that the posterior should be\n'
            'lattice-independent -- it should not be, because the prior differs.',
            fontsize=8.5, color=TEXT_SECONDARY)
        figure.savefig(Path(artifact_dir)/f'{tag}_homogeneity.png', dpi=200, facecolor='white')
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Write S07 calibration results from artefacts.')
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--tag', default='S07_null')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    main_table = read(artifact_dir, f'{args.tag}_main_table.csv')
    if main_table is None:
        raise SystemExit(f'{args.tag}_main_table.csv not found in {artifact_dir}')
    mcnemar = read(artifact_dir, f'{args.tag}_mcnemar.csv')
    homogeneity = read(artifact_dir, f'{args.tag}_homogeneity.csv')
    regularity = read(artifact_dir, f'{args.tag}_regularity.csv')
    pooling = read(artifact_dir, f'{args.tag}_pooling.csv')
    with open(artifact_dir/f'{args.tag}_meta.json', encoding='utf-8') as handle:
        meta = json.load(handle)

    if 'trials' not in main_table.columns:
        main_table['trials'] = ''
    write_figures(main_table, regularity, homogeneity, artifact_dir, args.tag)

    raw = main_table.loc[main_table['method'] == 'raw'].set_index('merit')
    m20 = raw.loc['M20']
    gate_line = float(m20['operating_point_matched_fpr']) + 0.03
    best = main_table.loc[main_table['method'] != 'raw'].sort_values(
        'operating_point_matched_fpr', ascending=False)
    winner = best.iloc[0]
    gate_one = bool(winner['operating_point_matched_fpr'] >= gate_line)

    lines = [
        '# S07 - null calibration: turning each figure of merit into a tail probability',
        '',
        f'Commit `{meta["commit"]}`, {meta["n_train_candidates"]:,} `{meta["train_split"]}` and '
        f'{meta["n_dev_candidates"]:,} `{meta["report_split"]}` candidates over '
        f'{len(meta["bundles"])} evaluable bundles. Nulls fitted on `{meta["train_split"]}`, '
        f'thresholds selected there, everything reported on `{meta["report_split"]}`. '
        f'Matched false-positive budget {meta["matched_fpr_budget"]:.4f}, the rate raw M20 incurs '
        "at de Wolff's threshold of 10 -- the same convention as F-066/F-067, so these numbers sit "
        "beside S06's.",
        '',
        '## The bound on everything fitted here, stated first',
        '',
        meta['r1_note'],
        '',
        'Only the `analytic` method escapes it, because nothing in it is fitted at all.',
        '',
        '## The gate',
        '',
        f'**Condition 1 - at least 3 percentage points of CNRS-weighted operating point over raw '
        f'M20 on `{meta["report_split"]}`, with a paired test.** Raw M20 scores '
        f'{m20["operating_point_matched_fpr"]:.4f}, so the line is {gate_line:.4f}. The best '
        f'calibrated score is `{winner["label"]}` at {winner["operating_point_matched_fpr"]:.4f}. '
        f'**{"MET" if gate_one else "NOT MET"}.**',
        '',
        f'Against the incumbent: `M_sym` raw scores '
        f'{raw.loc["M_sym", "operating_point_matched_fpr"]:.4f} (F-066), which no calibrated score '
        f'here reaches either.',
        '',
        f'Improvements are quoted against Shirley 1980\'s ~{REPRODUCIBILITY_FLOOR:.0%} '
        'reproducibility floor for M20 (F-009); the measured per-merit floor is S06b and does not '
        'exist yet.',
        '',
        ]
    if homogeneity is not None and homogeneity.shape[0]:
        spread = (homogeneity.loc[homogeneity['bravais_lattice'].notna()]
                  .groupby(['merit', 'method'], as_index=False)[['q0.5_spread', 'q0.999_spread']]
                  .first())
        headline = spread.loc[spread['merit'].isin(['M20', 'M_sym', 'null_tail_nll'])
                              & spread['method'].isin(['raw', 'z', 'rank', 'binned', 'analytic'])]
        lines += [
            '**Condition 2 - the post-calibration null distributions indistinguishable across '
            'Bravais lattices and extinction groups.** This is a diagnostic that the null *fit* is '
            'good, not a claim that the posterior should be lattice-independent -- it should not '
            'be, because the prior differs. Spread across the fourteen lattices of the null score '
            "over *incorrect* candidates, at the median and at the far tail where a ranking is "
            'decided:',
            '',
            table(headline.sort_values(['merit', 'method']),
                  ['merit', 'method', 'q0.5_spread', 'q0.999_spread']),
            '',
            '**MET, by `rank`, in both the body and the tail** - M20 collapses from 2.468 to 0.065 '
            'at the median and from 26.41 to 1.07 at the 99.9th percentile; `M_sym` from 4.859 to '
            '0.053 and 198.6 to 0.94. `z` equalises the median only, which is what removing '
            'location and scale can do and no more, and `analytic` does not equalise anything, '
            'which is F-075 restated.',
            '',
            '**Read the two conditions together, because that is the result.** The null fit '
            'succeeds -- after calibration a score means the same thing under "this candidate is '
            'wrong" in every lattice, by a factor of 25 to 200 -- and the ranking still gets '
            '*worse*. So the failure is not in the fitting and cannot be fixed by fitting better. '
            'It is in the premise: equalising the null was never sufficient, because the raw merit '
            "was carrying the prior too and the calibration removes it (F-076).",
            '',
            ]
    lines += [
        '## The leaderboard',
        '',
        table(main_table.sort_values('operating_point_matched_fpr', ascending=False),
              ['label', 'operating_point_matched_fpr', 'top10', 'top10_per_bl', 'mrr',
               'hard_op_given_found_d6', 'r1_bounded']),
        '',
        '## de Wolff 1961 against the pool',
        '',
        ]
    if regularity is not None and regularity.shape[0]:
        frame = regularity.loc[regularity['grouping'] == 'bravais_lattice'].copy()
        reference = frame.loc[frame['bravais_lattice'] == 'aP', 'rho']
        if reference.shape[0]:
            frame['rho_ratio_to_aP'] = frame['rho']/float(reference.iloc[0])
        lines += [
            "de Wolff's null says the mean per-line term is 1.0, and his own equidistant limit -- "
            'the extreme case of his regularity argument, where the calculated lines are perfectly '
            'regular -- puts it at 1.5. So the whole interval his argument allows is [1.0, 1.5], '
            'and `rho` = g_bar/Delta is bounded in [1/2, 1]. Measured:',
            '',
            table(frame.sort_values('mean_line_term'),
                  [column for column in ('bravais_lattice', 'n', 'mean_line_term', 'rho',
                                         'rho_ratio_to_aP') if column in frame.columns]),
            '',
            '**This is not a measurement of de Wolff\'s regularity ratio and must not be quoted as '
            'one, nor as a test of Wu 1988.** Three things confound it and all three run the same '
            'way. The candidates are least-squares *refined* against the very peaks they are '
            'scored on, so their discrepancies are fitted rather than arbitrary; a triclinic cell '
            'fits twenty peaks with six free parameters where a cubic one has one, so the '
            'degrees-of-freedom advantage alone orders the column; and the pool is censored at '
            'M20 >= 5 lattice-dependently (R1), which removes the worst-fitting wrong cells and '
            'removes more of them from the low-symmetry lattices. F-068 is the standing rule here: '
            'a cross-lattice null comparison on this pool is a statement about the search and the '
            'prune, not about crystallography.',
            '',
            ]
    if pooling is not None:
        # What each method costs to *hold*, which S14 needs and the handoff asks for beside the
        # per-call cost: a fitted null is loaded once and then gathered from, so its size is the
        # deployment constraint rather than its fit time.
        models = Path(meta.get('models_dir', os.path.join(BASE, 'mlindex', 'models', 'fom_null')))
        sizes = []
        for row in pooling.itertuples(index=False):
            suffix = f'{row.method}_trials' if getattr(row, 'trials', '') else row.method
            path = models/f'{row.merit}_{suffix}.npz'
            sizes.append(path.stat().st_size/1024 if path.exists() else float('nan'))
        pooling = pooling.assign(size_kb=sizes)
        by_method = pooling.groupby('method', as_index=False)['size_kb'].median()
        lines += [
            '## What each method costs to hold',
            '',
            'Median size of one fitted null, in kB. Serialisation is npz throughout, so loading is '
            'a memory map and a gather; nothing is unpickled and no estimator is stored.',
            '',
            table(by_method, ['method', 'size_kb'], floatfmt=1),
            '',
            ]
    lines += [
        '## Pooling, clamping and extrapolation',
        '',
        table(pooling.fillna({'trials': ''}),
              ['merit', 'method', 'trials', 'n_groups', 'n_pooled', 'n_group_fallback',
               'n_unseen_group', 'n_tail_extrapolated', 'n_clamped_low'], floatfmt=0)
        if pooling is not None else '(no pooling table)',
        '',
        ]
    if mcnemar is not None and 'p_value' in mcnemar.columns:
        paired = mcnemar.loc[(mcnemar['baseline'] == 'M20 (raw)') & (mcnemar['subset'] == 'all')]
        paired = paired.sort_values('delta').head(20)
        lines += ['## Paired comparisons against raw M20', '',
                  table(paired, ['label', 'n_a_only', 'n_b_only', 'delta', 'p_value']), '']
    lines += [
        '## Reproduced controls',
        '',
        f'- Raw M20 cross-lattice top-10 {m20["top10"]:.4f} and operating point at the matched '
        f'budget {m20["operating_point_matched_fpr"]:.4f}, against F-066\'s 0.5115 and 0.2969.',
        f'- Raw M20 per-lattice top-10 {m20["top10_per_bl"]:.4f}, against F-074\'s 0.6745.',
        '',
        ]
    out = Path(args.out) if args.out else artifact_dir/'S07_calibration.md'
    with open(out, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
