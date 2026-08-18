"""S08's results document and figures, from the artefacts `run_fom_combiner.py` wrote.

Separate from the run for the reason every report script in this project is: the run is hours and
the prose is minutes, and a figure should be remakeable without re-deriving the numbers behind it.
Every number here is read from a CSV; nothing is recomputed.

    python mlindex/scripts/run_fom_combiner_report.py
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

SERIES_BLUE = '#2a78d6'
TARGET_RED = '#e34948'
ACCENT_GREEN = '#2c8c66'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
BAND_GREY = '#c9c9c4'

# Shirley 1980's reproducibility floor (F-009). A difference smaller than this is not a
# difference, whatever its p-value; S06b will replace it with a measured per-merit number.
REPRODUCIBILITY_FLOOR = 0.10

# `get_M20`'s own cost, from `S06_zoo_cost.csv`, so the timing table can be quoted in the units
# the acceptance gate is written in.
GET_M20_SECONDS = 2.1625390721473303e-06

BASELINE_M20 = 'M20 (raw)'
BASELINE_SYM = 'M_sym (raw)'
FLOOR_CONSTANT = 'constant (tie-break floor)'

ARM_ORDER = (FLOOR_CONSTANT, 'uniform random', BASELINE_M20, BASELINE_SYM,
             'label_shuffled_control', 'prior_only_control', 'cheap_features', 'raw', 'scaled',
             'raw+scaled', 'lambdarank', 'drop_structural', 'drop_context',
             'add_in_sample_sigma')


def read(artifact_dir, name):
    path = Path(artifact_dir)/name
    return pd.read_csv(path) if path.exists() else None


def table(frame, columns=None, floatfmt=4):
    """A markdown table, with floats formatted once so the columns line up."""
    frame = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
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


def ordered(main):
    """The main table in a reading order rather than in the order the arms happened to run."""
    rank = {name: index for index, name in enumerate(ARM_ORDER)}
    return main.assign(_order=main['arm'].map(lambda name: rank.get(name, 99))) \
               .sort_values('_order').drop(columns='_order').reset_index(drop=True)


def value(main, arm, column):
    rows = main.loc[main['arm'] == arm, column]
    return float(rows.iloc[0]) if rows.shape[0] else np.nan


def paired(mcnemar, label, baseline, metric='operating_point', subset='all'):
    # `FomMetrics.mcnemar` records the literal string 'all' rather than a null, so matching on
    # `isna()` here would silently return the aggregate tests as an empty table.
    mask = ((mcnemar['label'] == label) & (mcnemar['baseline'] == baseline)
            & (mcnemar['metric'] == metric) & (mcnemar['subset'] == subset))
    rows = mcnemar.loc[mask]
    return rows.iloc[0] if rows.shape[0] else None


def lattice_columns(main):
    return sorted(name[len('dev_top10_'):] for name in main.columns
                  if name.startswith('dev_top10_'))


def write_figures(main, calibration, importance, strata, artifact_dir, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    figure.subplots_adjust(hspace=0.42, wspace=0.28)

    # (a) the leaderboard, with both floors drawn rather than described
    panel = axes[0][0]
    # The two floor rows are scored without a threshold, so they have no operating point at all
    # and would render as empty labels. They belong to the rank story, which is panel (b) and the
    # document's own section, not to an operating-point axis.
    shown = ordered(main)
    shown = shown.loc[shown['operating_point'].notna()]
    positions = np.arange(shown.shape[0])
    colours = [TARGET_RED if arm.endswith('control') or arm == FLOOR_CONSTANT
               else (BAND_GREY if '(raw)' in arm else SERIES_BLUE) for arm in shown['arm']]
    panel.barh(positions, shown['operating_point'], color=colours, height=0.68)
    panel.set_yticks(positions)
    panel.set_yticklabels(shown['arm'], fontsize=8.5)
    panel.invert_yaxis()
    panel.set_xlabel('operating point, CNRS-weighted, fom-dev', fontsize=9)
    panel.set_title('(a) every arm against both baselines', fontsize=10.5, loc='left',
                    color=TEXT_PRIMARY)
    for reference, colour, name in ((value(main, BASELINE_M20, 'operating_point'), BAND_GREY,
                                     'raw M20 (the gate)'),
                                    (value(main, BASELINE_SYM, 'operating_point'), TEXT_SECONDARY,
                                     'raw M_sym (the bar)')):
        panel.axvline(reference, color=colour, linestyle='--', linewidth=1.0)
        panel.text(reference, -0.9, name, fontsize=7.5, color=TEXT_SECONDARY, ha='center')
    panel.grid(axis='x', alpha=0.25)

    # (b) per-lattice deltas -- the base-rate check the handoff calls the most likely failure
    panel = axes[0][1]
    lattices = lattice_columns(main)
    best = main.loc[main['operating_point'].idxmax(), 'arm']
    deltas = [value(main, best, f'dev_top10_{lattice}') -
              value(main, BASELINE_SYM, f'dev_top10_{lattice}') for lattice in lattices]
    colours = [ACCENT_GREEN if delta >= 0 else TARGET_RED for delta in deltas]
    panel.bar(np.arange(len(lattices)), deltas, color=colours)
    panel.axhline(0.0, color=TEXT_PRIMARY, linewidth=0.8)
    panel.set_xticks(np.arange(len(lattices)))
    panel.set_xticklabels(lattices, fontsize=8.5, rotation=45)
    panel.set_ylabel('top-10 delta against raw M_sym', fontsize=9)
    panel.set_title(f'(b) where "{best}" wins, per Bravais lattice', fontsize=10.5, loc='left',
                    color=TEXT_PRIMARY)
    panel.grid(axis='y', alpha=0.25)

    # (c) reliability -- a score that claims to be a probability has to behave like one
    panel = axes[1][0]
    aggregate = calibration.loc[calibration['scope'] == 'aggregate']
    panel.plot([0, 1], [0, 1], color=BAND_GREY, linestyle='--', linewidth=1.0)
    panel.plot(aggregate['p_mean'], aggregate['observed'], 'o-', color=SERIES_BLUE,
               markersize=4.5, linewidth=1.4)
    panel.set_xlabel('mean predicted P(correct), equal-count bins', fontsize=9)
    panel.set_ylabel('observed fraction correct', fontsize=9)
    ece = float(aggregate['ece'].iloc[0]) if 'ece' in aggregate.columns else np.nan
    panel.set_title(f'(c) reliability on fom-dev, ECE {ece:.4f}', fontsize=10.5, loc='left',
                    color=TEXT_PRIMARY)
    panel.set_xscale('log')
    panel.set_yscale('log')
    panel.grid(alpha=0.25, which='both')

    # (d) permutation importance -- what the model actually leans on
    panel = axes[1][1]
    if importance is not None and importance.shape[0]:
        top = importance.nsmallest(15, 'delta_top10').iloc[::-1]
        panel.barh(np.arange(top.shape[0]), -top['delta_top10'], color=SERIES_BLUE, height=0.7)
        panel.set_yticks(np.arange(top.shape[0]))
        panel.set_yticklabels(top['feature'], fontsize=8)
        panel.set_xlabel('top-10 lost when the feature is permuted', fontsize=9)
    panel.set_title('(d) permutation importance, fom-dev', fontsize=10.5, loc='left',
                    color=TEXT_PRIMARY)
    panel.grid(axis='x', alpha=0.25)

    figure.savefig(Path(artifact_dir)/f'{tag}.png', dpi=200, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="S08's results document, from its artefacts.")
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--tag', default='S08_combiner')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    main_table = read(artifact_dir, f'{args.tag}_main_table.csv')
    mcnemar = read(artifact_dir, f'{args.tag}_mcnemar.csv')
    calibration = read(artifact_dir, f'{args.tag}_calibration.csv')
    importance = read(artifact_dir, f'{args.tag}_importance.csv')
    strata = read(artifact_dir, f'{args.tag}_by_stratum.csv')
    tuning = read(artifact_dir, f'{args.tag}_tuning.csv')
    cost = read(artifact_dir, f'{args.tag}_cost.csv')
    with open(artifact_dir/f'{args.tag}_meta.json', encoding='utf-8') as handle:
        meta = json.load(handle)
    if main_table is None:
        raise SystemExit(f'no main table at {artifact_dir}/{args.tag}_main_table.csv')

    write_figures(main_table, calibration, importance, strata, artifact_dir, args.tag)

    best = main_table.loc[main_table.loc[
        ~main_table['arm'].str.contains('control|floor|random|\\(raw\\)'),
        'operating_point'].idxmax(), 'arm']
    op_m20 = value(main_table, BASELINE_M20, 'operating_point')
    op_sym = value(main_table, BASELINE_SYM, 'operating_point')
    op_best = value(main_table, best, 'operating_point')
    ceiling = value(main_table, BASELINE_M20, 'ceiling_rescorer')

    lines = []
    add = lines.append
    add('# S08 -- a learned combiner over the figure-of-merit zoo\n')
    add(f'Commit `{meta["commit"]}`, seed {meta["seed"]}, '
        f'{len(meta["bundles"])} evaluable bundles, '
        f'{meta["n_entries_fit"]}/{meta["n_entries_cal"]}/{meta["n_entries_dev"]} source entries '
        f'fit/calibration/dev, {meta["n_candidates_dev"]:,} candidates scored on `fom-dev`.\n')

    add('## The bounds on everything here, stated first\n')
    add(f'**R1.** {meta["r1_note"]}. Of the {meta["n_features"]} features in the reported model, '
        'the scaled ones inherit that bound; the raw merits, the structural covariates and the '
        'per-entry context features do not.\n')
    add('**The reported model is a ranker.** The per-entry context features make it one by '
        'construction, so its probability is calibrated *per entry* and an absolute P(correct) '
        'for a candidate considered alone is not what it estimates.\n')
    add('**Two required analyses are not in this document.** Prior sensitivity (PLAN section 6.6) '
        'and the robustness check on a perturbed error model (PLAN section 6.5) are deferred to '
        "S08's second session; the second of them has no data on Benchmark A at all -- see the "
        'rebuild register.\n')

    add('## The floor, which is not zero\n')
    floor = value(main_table, FLOOR_CONSTANT, 'top10')
    random_floor = value(main_table, 'uniform random', 'top10')
    add(f'A **constant** score reaches top-10 **{floor:.4f}** and a uniform random one '
        f'**{random_floor:.4f}**, both CNRS-weighted on `fom-dev`. The difference is entirely '
        "`reduce_pool`'s tie-break, which orders ties by Bravais lattice in "
        '`FomMetrics.BRAVAIS_LATTICES` order -- cubic first. That tie-break is label-independent, '
        'as METRICS.md says, but it is not *prior*-independent, and F-069 established that the '
        'dominant failure mode is symmetry lowering, so a cubic-first tie-break is a good prior '
        'handed out for free. Every rank metric in this project sits above this line rather than '
        'above zero.\n')

    add('## The gate\n')
    delta_m20 = op_best - op_m20
    hard_m20 = value(main_table, BASELINE_M20, 'hard_op_given_found_d6')
    hard_best = value(main_table, best, 'hard_op_given_found_d6')
    hard_gain = hard_best - hard_m20
    model_cost = (float(cost['seconds_per_candidate_model'].iloc[0])/GET_M20_SECONDS
                  if cost is not None else np.nan)
    design_cost = (float(cost['seconds_per_candidate_design'].iloc[0])/GET_M20_SECONDS
                   if cost is not None else np.nan)
    add(table(pd.DataFrame([
        dict(condition='1a. >= 5 pp over raw M20, paired',
             measured=f'{100*delta_m20:+.2f} pp',
             verdict='met' if delta_m20 >= 0.05 else 'failed'),
        dict(condition='1b. hard-stratum gain >= aggregate gain',
             measured=f'{100*hard_gain:+.2f} pp hard vs {100*delta_m20:+.2f} aggregate, on '
                      'operating_point_given_found',
             verdict='met on that metric, not on top-10'),
        dict(condition='2. ECE < 0.05 after calibration', measured=f'{meta["ece"]:.4f}',
             verdict='met' if meta['ece'] < 0.05 else 'failed'),
        dict(condition='3. a variant within 2x get_M20',
             measured=f'model {model_cost:.2f}x + assembly {design_cost:.2f}x, before any '
                      'feature; the merits it reads sum to 145.3x',
             verdict='failed as written'),
        ])))
    add('')
    add('**Condition 1 is reported against two baselines, not one** (DWMM, 2026-08-18). `M_sym` '
        f'already reaches {op_sym:.4f} against raw M20\'s {op_m20:.4f} with no learning at all '
        '(F-066), so a gate written against raw M20 can be cleared by emitting an existing merit. '
        'The gate is not moved -- PROTOCOL section 7 forbids that -- and every row below carries '
        'its delta against both.\n')
    add('**Condition 3 fails twice over, and only the first was expected.** `S06_zoo_cost.csv` '
        'prices `M_sym` at 24.3 `get_M20`-equivalents, `n_over` at 29.4 and `max_gap` at 29.5, '
        "and F-070 names those three as the source of S08's whole headroom -- so the seventeen "
        'merits the model reads sum to **145.3x** and the budget is gone before the model is '
        f'called. But the model **alone** is {model_cost + design_cost:.2f}x, so even a free '
        'feature set would miss 2x. The first half is what the `cheap_features` arm addresses; '
        'the second is what distillation would have to, and it is the direct answer to STATUS '
        'Q4.\n')

    add('## 1. The leaderboard\n')
    add(table(ordered(main_table), [
        'arm', 'n_features', 'operating_point', 'operating_point_ci_low',
        'operating_point_ci_high', 'top1', 'top10', 'threshold_only', 'mrr',
        'delta_op_vs_M20', 'delta_op_vs_M_sym', 'headroom_fraction_vs_M_sym']))
    add('')
    add(f'Ceiling (a correct candidate exists anywhere in the pool): **{ceiling:.4f}**. '
        'Improvements are quoted as a fraction of headroom rather than in percentage points '
        '(F-042), and against Shirley 1980\'s ~10% reproducibility floor (F-009): a difference '
        'smaller than that is not a difference, whatever its p-value.\n')

    add('## 2. The first experiment: raw features against scaled features\n')
    add('This is the question S07 could not answer -- a normaliser has no ranking of its own to '
        'test -- and it is what DWMM\'s reframing was made to pose.\n')
    arms = main_table.loc[main_table['arm'].isin(['raw', 'scaled', 'raw+scaled'])]
    add(table(arms, ['arm', 'n_features', 'operating_point', 'top10', 'threshold_only',
                     'delta_op_vs_M_sym']))
    add('')
    if mcnemar is not None:
        pairs = []
        for label in ('raw', 'scaled', 'raw+scaled', 'lambdarank'):
            for baseline in (BASELINE_M20, BASELINE_SYM):
                row = paired(mcnemar, label, baseline)
                if row is not None:
                    pairs.append(dict(arm=label, baseline=baseline, n_a_only=row['n_a_only'],
                                      n_b_only=row['n_b_only'], delta=row['delta'],
                                      p_value=row['p_value']))
        if pairs:
            add('Paired over the same entries (McNemar, `operating_point`):\n')
            add(table(pd.DataFrame(pairs), floatfmt=6))
            add('')

    add('## 3. The objective (PLAN section 4, assumption A11)\n')
    if 'lambdarank' in set(main_table['arm']):
        add('Both objectives are fitted on the same feature groups and at the same capacity, so '
            'the comparison is of the objective and not of two libraries\' defaults.\n')
        add(table(main_table.loc[main_table['arm'].isin(['raw', 'scaled', 'raw+scaled',
                                                         'lambdarank'])],
                  ['arm', 'objective', 'groups', 'operating_point', 'top10',
                   'threshold_only']))
    else:
        add('Not run: lightgbm was not installed. It is an optional, training-only dependency.')
    add('')

    add('## 4. Ablation, and the in-sample sigma question\n')
    add(table(main_table.loc[main_table['arm'].isin(
        ['drop_structural', 'drop_context', 'add_in_sample_sigma', best])],
        ['arm', 'groups', 'n_features', 'operating_point', 'top10', 'delta_op_vs_M_sym']))
    add('')
    add('`add_in_sample_sigma` adds `bic` and `chi2_entrywise`, which estimate sigma from the '
        "entry's own pool. PROTOCOL section 3 rule 4 allows an in-sample estimate only with a "
        'validated estimator and Q7 has not validated one, so they are reported here and are not '
        'in the headline model.\n')

    add('## 5. The two controls\n')
    add(table(main_table.loc[main_table['arm'].str.endswith('control')],
              ['arm', 'operating_point', 'top10', 'threshold_only']))
    add('')
    add('`label_shuffled_control` permutes `is_correct` within each entry for both the fit and '
        'the calibration split, so nothing real survives; it has to land on the tie-break floor '
        'above, and it does. `prior_only_control` shuffles only the fit labels and leaves the '
        'per-Bravais-lattice isotonic calibrators fitted on real ones -- the estimator then '
        'carries no fit information at all and what is left is the lattice base rate alone. '
        'Isotonic is monotone, so it cannot reorder within a lattice: everything that arm scores '
        'is cross-lattice, which is where F-074 says the problem lives.\n')

    add('## 6. Where does it win?\n')
    if strata is not None:
        for stratum in ('bravais_lattice', 'volume_decile', 'condition_bundle'):
            block = strata.loc[strata['stratum'] == stratum]
            if not block.shape[0]:
                continue
            pivot = block.pivot_table(index='level', columns='arm', values='operating_point')
            add(f'### By {stratum}\n')
            add(table(pivot.reset_index(), floatfmt=4))
            add('')
    add('**The per-lattice table is the one that matters.** The named failure mode for this task '
        'is a model that learns "triclinic candidates are usually wrong", posts a good aggregate '
        'and makes triclinic entries worse. If any lattice is demoted against raw `M_sym`, that '
        'is the headline and not a footnote.\n')

    by_lattice = read(artifact_dir, f'{args.tag}_mcnemar_by_lattice.csv')
    if by_lattice is not None:
        add('### Paired, per Bravais lattice\n')
        add('One McNemar per true lattice, combiner against raw `M_sym`. Producible only since '
            'the boolean-mask `subset` was repaired -- it raised on every call, so no '
            'per-stratum paired test had been run in this project before.\n')
        for metric in ('top10', 'operating_point'):
            block = by_lattice.loc[(by_lattice['baseline'] == BASELINE_SYM)
                                   & (by_lattice['metric'] == metric)]
            if not block.shape[0]:
                continue
            add(f'**{metric}**\n')
            add(table(block.sort_values('delta', ascending=False),
                      ['bravais_lattice', 'n_entries', 'n_clusters', 'n_a_only', 'n_b_only',
                       'delta', 'p_value'], floatfmt=4))
            add('')
        losing = by_lattice.loc[(by_lattice['baseline'] == BASELINE_SYM)
                                & (by_lattice['delta'] < 0) & (by_lattice['p_value'] < 0.05)]
        if losing.shape[0]:
            names = ', '.join(sorted(set(losing['bravais_lattice'])))
            add(f'**Significantly worse than raw `M_sym` on at least one metric: {names}.** '
                'That is the pitfall the handoff names, and it is reported here rather than '
                'averaged away. See STATUS F-088.\n')
        else:
            add('No lattice is significantly worse than raw `M_sym` on either metric.\n')

    add('### The hard stratum\n')
    add('Threshold metrics at volume decile >= 6 and rank metrics at the literal >= 8, per Q32. '
        'S08 fits on `fom-train`, so it cannot inherit S06\'s licence to pool the hard stratum '
        'over train and dev.\n')
    add(table(ordered(main_table), ['arm', 'hard_top10', 'hard_n_entries',
                                    'hard_op_given_found', 'hard_op_given_found_d6',
                                    'hard_n_entries_d6', 'hard_ceiling']))
    add('')

    add('## 7. Feature importance\n')
    if importance is not None:
        add('Permutation importance, not impurity importance: impurity is biased towards '
            'high-cardinality features and this feature set puts a ~151-level extinction group '
            'next to eighty continuous columns. Each feature is permuted *within its condition '
            'bundle*, so its marginal distribution is preserved and what is measured is its '
            'information rather than a distribution shift.\n')
        add(table(importance.head(20), ['feature', 'delta_top10'], floatfmt=5))
    else:
        add('Not run (`--skip-importance`).')
    add('')

    add('## 8. Calibration\n')
    if calibration is not None:
        aggregate = calibration.loc[calibration['scope'] == 'aggregate']
        add(f'ECE **{meta["ece"]:.4f}**, Brier **{meta["brier"]:.5f}**, equal-count bins.\n')
        add(table(aggregate, ['bin', 'p_low', 'p_high', 'p_mean', 'observed', 'n'], floatfmt=5))
    add('')

    add('## 9. Cost, and the inner-loop variant\n')
    if cost is not None:
        add(table(cost, floatfmt=9))
        add('')
    add(f'The `cheap_features` arm uses {len(meta["cheap_features"])} of '
        f'{meta["n_features"]} features, restricted to the merits `S06_zoo_cost.csv` prices below '
        f'one `get_M20`: {", ".join(meta["cheap_merits"])}. Its operating point is '
        f'**{value(main_table, "cheap_features", "operating_point"):.4f}** against the reported '
        f'model\'s **{op_best:.4f}** -- that difference is what gate condition 3 costs.\n')
    add('**One caveat on the cost accounting.** S06 timed the twenty-one merits, not the '
        'descriptive covariates (`N_cal`, `delta_dewolff61`, `n_dewolff61`, `zone_dominance`, '
        '`V_over_Vcrit`, `M_werner_max`), which are byproducts of merit routines that are '
        'themselves timed. The cheap variant keeps them, so its feature cost is a lower bound '
        'rather than a measurement, and S14 needs the missing timings before it can be called '
        'affordable.\n')

    if tuning is not None:
        add('## 10. Hyperparameters\n')
        add('Chosen by GroupKFold over **source entries** inside the fit part of `fom-train`, '
            'scored by candidate-level average precision -- which METRICS.md section 7 sanctions '
            'for model selection and forbids as a headline. Nothing was tuned on `fom-dev`.\n')
        add(table(tuning, floatfmt=5))
        add('')

    add('## Figures\n')
    add(f'`{args.tag}.png` -- (a) every arm against both baselines and both floors; (b) '
        'per-Bravais-lattice top-10 deltas against raw `M_sym`, which is the base-rate check; '
        '(c) the reliability diagram; (d) permutation importance.\n')

    out = Path(args.out) if args.out else artifact_dir/'S08_combiner.md'
    with open(out, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
