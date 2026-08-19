"""S10's results document and figure, assembled from the CSVs the analysis stages wrote.

Every number here is read from an artefact; nothing is recomputed. The run is minutes and the prose
is seconds, and a figure should be remakeable without re-deriving what is behind it.

    python mlindex/scripts/run_fom_cv_report.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SERIES_BLUE = '#2a78d6'
TARGET_RED = '#e34948'
ACCENT_GREEN = '#2c8c66'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
BAND_GREY = '#c9c9c4'

# Shirley 1980's ~10% reproducibility floor (F-009). A difference below it is not a difference.
REPRODUCIBILITY_FLOOR = 0.10

# What a score with no information already earns, because reduce_pool breaks ties cubic-first and
# F-069 makes that a good prior (F-083). Every rank metric is read against this, not against zero.
CONSTANT_SCORE_TOP10 = 0.2657


def read(path):
    """None for a missing artefact, so a partial run still reports."""
    if not os.path.exists(path):
        return None
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    return pd.read_csv(path)


def table(frame, columns=None, floatfmt=4):
    if frame is None or not len(frame):
        return ['_(not produced)_', '']
    frame = frame[columns] if columns else frame
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: '' if pd.isna(value) else f'{value:.{floatfmt}f}')
    header = '| ' + ' | '.join(str(column) for column in formatted.columns) + ' |'
    rule = '|' + '|'.join(['---']*len(formatted.columns)) + '|'
    body = ['| ' + ' | '.join(str(value) for value in row) + ' |'
            for row in formatted.itertuples(index=False)]
    return [header, rule] + body + ['']


def write_figures(main, scaling_cross, scaling_within, cost, artifact_dir, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    figure.subplots_adjust(hspace=0.42, wspace=0.28)

    panel = axes[0][0]
    if main is not None:
        frame = main.sort_values('top10')
        colours = [BAND_GREY if row == 'baseline' else SERIES_BLUE for row in frame['family']]
        panel.barh(frame['merit'], frame['top10'], color=colours)
        panel.axvline(CONSTANT_SCORE_TOP10, color=TEXT_SECONDARY, linestyle=':', linewidth=1.0)
        panel.text(CONSTANT_SCORE_TOP10, -0.6, ' constant score (F-083)', fontsize=7.5,
                   color=TEXT_SECONDARY)
        for baseline, colour in (('M20', TARGET_RED), ('M_sym', ACCENT_GREEN)):
            row = main.loc[main['merit'] == baseline]
            if row.shape[0]:
                panel.axvline(float(row['top10'].iloc[0]), color=colour, linestyle='--',
                              linewidth=1.0)
        panel.set_xlabel('top-10 on fom-dev', fontsize=9)
        panel.tick_params(labelsize=7.5)
        panel.grid(axis='x', alpha=0.25)
    panel.set_title('(a) Predictive merits against the incumbents', fontsize=10.5, loc='left',
                    color=TEXT_PRIMARY)

    panel = axes[0][1]
    if scaling_cross is not None:
        panel.plot(scaling_cross['n_free'], scaling_cross['median_penalty'], marker='o',
                   color=SERIES_BLUE, label='cross-lattice')
    if scaling_within is not None and len(scaling_within):
        panel.plot(scaling_within['n_free'], scaling_within['median_penalty_ratio'], marker='s',
                   color=ACCENT_GREEN, label='identical peaks (relative)')
    panel.axhline(1.0, color=TEXT_SECONDARY, linestyle=':', linewidth=1.0)
    panel.set_xlabel('free cell parameters', fontsize=9)
    panel.set_ylabel('is_M / cv_M on wrong candidates', fontsize=9)
    panel.tick_params(labelsize=8)
    panel.legend(fontsize=8)
    panel.grid(axis='y', alpha=0.25)
    panel.set_title('(b) Does the penalty scale with the degrees of freedom?', fontsize=10.5,
                    loc='left', color=TEXT_PRIMARY)

    panel = axes[1][0]
    if scaling_cross is not None:
        width = 0.38
        positions = np.arange(scaling_cross.shape[0])
        panel.bar(positions - width/2, scaling_cross['median_is_M'], width, color=BAND_GREY,
                  label='in-sample (is_M)')
        panel.bar(positions + width/2, scaling_cross['median_cv_M'], width, color=SERIES_BLUE,
                  label='held out (cv_M)')
        panel.axhline(1.0, color=TARGET_RED, linestyle='--', linewidth=1.0)
        panel.text(-0.4, 1.05, "de Wolff's arbitrary-cell null", fontsize=7.5, color=TARGET_RED)
        panel.set_xticks(positions)
        panel.set_xticklabels(scaling_cross['n_free'])
        panel.set_xlabel('free cell parameters', fontsize=9)
        panel.set_ylabel('median merit, wrong candidates', fontsize=9)
        panel.legend(fontsize=8)
        panel.tick_params(labelsize=8)
        panel.grid(axis='y', alpha=0.25)
    panel.set_title('(c) How far cross-validation moves a wrong cell towards the null',
                    fontsize=10.5, loc='left', color=TEXT_PRIMARY)

    panel = axes[1][1]
    if cost is not None and main is not None:
        merged = []
        for _, row in cost.iterrows():
            name = str(row['merit'])
            # A cost row prices the whole family; plot it against that family's best merit, which
            # is the de Wolff form in every case.
            key = {'M20': 'M20', 'is_* (all in-sample)': 'is_M20',
                   'ho_* (5 peaks)': 'ho_M20'}.get(name)
            if key is None and name.startswith('cv_* ('):
                key = f"cv_M20__{name[len('cv_* ('):-1]}"
            match = main.loc[main['merit'] == key] if key else None
            if match is not None and match.shape[0]:
                merged.append((name, float(row['cost_vs_M20']), float(match['top10'].iloc[0])))
        if merged:
            for name, x, y in merged:
                panel.scatter(x, y, color=SERIES_BLUE, s=42)
                panel.annotate(name, (x, y), fontsize=7.5, xytext=(4, 4),
                               textcoords='offset points', color=TEXT_SECONDARY)
        panel.axvline(2.0, color=TARGET_RED, linestyle='--', linewidth=1.0)
        panel.text(2.1, panel.get_ylim()[0], ' inner-loop budget', fontsize=7.5, color=TARGET_RED)
        panel.set_xscale('log')
        panel.set_xlabel('cost, multiples of get_M20', fontsize=9)
        panel.set_ylabel('top-10 on fom-dev', fontsize=9)
        panel.tick_params(labelsize=8)
        panel.grid(alpha=0.25)
    panel.set_title('(d) What each merit costs for what it returns', fontsize=10.5, loc='left',
                    color=TEXT_PRIMARY)

    figure.savefig(os.path.join(artifact_dir, f'{tag}_fom.png'), dpi=200, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="S10's results document.")
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--tag', default='S10_cv')
    args = parser.parse_args()

    def load(name):
        return read(os.path.join(args.artifact_dir, f'{args.tag}_{name}'))

    main_table = load('main_table.csv')
    mcnemar = load('mcnemar.csv')
    by_stratum = load('by_stratum.csv')
    scaling_cross = load('scaling_cross.csv')
    scaling_within = load('scaling_within.csv')
    gate = load('gate.csv')
    gate_mcnemar = load('gate_mcnemar.csv')
    confound = load('confound.csv')
    correlation = load('confound_correlation.csv')
    cost = load('cost.csv')
    combiner = load('combiner_table.csv')
    combiner_mcnemar = load('combiner_mcnemar.csv')
    importance = load('combiner_importance.csv')
    holdout_gate = load('holdout_gate.csv')
    build = load('build.csv')
    meta = load('main_meta.json') or {}
    scaling_meta = load('scaling_meta.json') or {}

    lines = []
    add = lines.append
    add('# S10 - Cross-validated and hold-out figures of merit')
    add('')
    add(f'Commit `{meta.get("commit", "unknown")}`, seed {meta.get("seed", "?")}, '
        f'{len(meta.get("bundles", []))} evaluable bundles, fold schemes '
        f'{", ".join(meta.get("schemes", []))}. Thresholds selected on '
        f'`{meta.get("train_split", "fom-train")}` and reported on '
        f'`{meta.get("report_split", "fom-dev")}`.')
    add('')
    add('## The bounds on everything here, stated first')
    add('')
    add('- **R1** - the pool is censored at M20 >= 5, so a predictive merit that would rank a '
        'low-M20 candidate highly cannot be evaluated on it.')
    add('- **R10** - every candidate has already been Gauss-Newton refined against the peaks it '
        'is then scored on. That refinement advantage is exactly what cross-validation exists to '
        'remove, so the size of what it removes is a measurement here rather than a nuisance - '
        'but the "wrong cell" in this pool is a refined survivor, not de Wolff\'s arbitrary cell.')
    add('- **R5** - cubic is scored on ten peaks and everything else on twenty, which is the '
        'confound the degrees-of-freedom claim runs through. `cv_tail_nll`\'s Gamma(n, 1) form '
        'and the identical-peaks control in section 3 are the two mitigations.')
    add('- **R13 (new)** - Variant A\'s hold-out lines were never stored and had to be '
        're-synthesised. They carry no contaminants and no second-phase lines, and come from a '
        'second noise draw. Its numbers are optimistic by an unmeasured amount.')
    add('- **F-083** - a constant score already scores '
        f'{CONSTANT_SCORE_TOP10:.4f} on top-10. Every rank metric below is read against that '
        'line, not against zero.')
    add(f'- **F-009** - M20\'s reproducibility floor is ~{REPRODUCIBILITY_FLOOR:.0%}. A '
        'difference below it is not a difference.')
    add('')

    add('## 1. The leaderboard')
    add('')
    lines.extend(table(main_table, [
        'merit', 'family', 'n_entries', 'operating_point', 'operating_point_matched_fpr',
        'top10', 'top1', 'mrr', 'precision', 'reported', 'hard_operating_point_given_found',
        ]))
    add('### Paired tests against the incumbents')
    add('')
    lines.extend(table(mcnemar, [
        'merit', 'baseline', 'metric', 'n_a_only', 'n_b_only', 'delta', 'p_value',
        ], floatfmt=6))

    add('## 2. Where it wins and loses')
    add('')
    lines.extend(table(by_stratum, None))

    add('## 3. Does the penalty scale with the degrees of freedom?')
    add('')
    add('The gate\'s second condition. The penalty is `is_M / cv_M` on incorrect candidates: the '
        'same statistic on the peaks the cell was fitted to, over the same statistic on peaks it '
        'was not. One means cross-validation found no fitting advantage to remove.')
    add('')
    add('**Cross-lattice** - confounded with everything else that differs between lattices:')
    add('')
    lines.extend(table(scaling_cross, None))
    add('**Identical peaks** - the same entry\'s candidates at different parameter counts, so the '
        'peaks, the noise and the entry\'s difficulty are held fixed by construction:')
    add('')
    lines.extend(table(scaling_within, None))
    add(f'Monotone cross-lattice: **{scaling_meta.get("monotone_cross_lattice")}**. '
        f'Monotone on identical peaks: **{scaling_meta.get("monotone_identical_peaks")}**.')
    add('')

    add('## 4. The acceptance stratum')
    add('')
    add('The handoff asks for a gain where "a larger-volume, lower-symmetry candidate outranks '
        'the correct one". F-069 measured that stratum: only 40.5% of wrong winners are larger '
        '(the median is 20% *smaller*) while 85.5% are lower symmetry. The primary stratum is '
        'therefore symmetry lowering, and the literal conjunction is reported beside it '
        'unchanged, so no earlier number changes meaning (STATUS section 6).')
    add('')
    lines.extend(table(gate, None, floatfmt=6))
    add('Paired, on the same entries:')
    add('')
    lines.extend(table(gate_mcnemar, [
        'stratum', 'merit', 'baseline', 'metric', 'n_entries', 'n_a_only', 'n_b_only', 'delta',
        'p_value',
        ], floatfmt=6))

    add('## 5. Is it measuring over-fitting capacity, or something duller?')
    add('')
    add('Discrimination as an AUC, pooled and then inside bands that hold a confound fixed.')
    add('')
    lines.extend(table(confound, None))
    add('Spearman correlation between the merits:')
    add('')
    lines.extend(table(correlation, None, floatfmt=3))

    add('## 6. Cost')
    add('')
    lines.extend(table(cost, None, floatfmt=8))

    add('## 7. What it adds to S08')
    add('')
    lines.extend(table(combiner, [
        'arm', 'n_features', 'operating_point', 'top10', 'precision', 'hard_op_given_found_d6',
        ]))
    lines.extend(table(combiner_mcnemar, [
        'comparison', 'metric', 'n_a_only', 'n_b_only', 'delta', 'p_value',
        ], floatfmt=6))
    if importance is not None:
        add('The CV columns by permutation importance, against the full ordering:')
        add('')
        lines.extend(table(importance.loc[importance['is_cv']], None, floatfmt=6))
        add(f'Their best rank among all {importance.shape[0]} features: '
            f'{int(importance.reset_index(drop=True).index[importance["is_cv"].values][0]) + 1}.')
        add('')

    add('## 8. Variant A: rebuilding what the benchmark discarded')
    add('')
    add('The surplus peaks were never stored. They were reconstructed by replaying the '
        'generator, with the noise on the extra lines drawn from a separate stream so the frozen '
        'twenty are untouched, and the replay was gated against the stored peak list before any '
        'Variant A number was produced.')
    add('')
    lines.extend(table(holdout_gate, None, floatfmt=6))

    add('## 9. Build')
    add('')
    lines.extend(table(build, None, floatfmt=6))

    path = os.path.join(args.artifact_dir, f'{args.tag}_fom.md')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    write_figures(main_table, scaling_cross, scaling_within, cost, args.artifact_dir, args.tag)
    print(f'wrote {path} and {args.tag}_fom.png')


if __name__ == '__main__':
    main()
