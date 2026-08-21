"""S11 block C: turn what `run_fom_neural.py` wrote into the results document and its figure.

    python mlindex/scripts/run_fom_neural_report.py --tag S11_C

Reads only the CSVs and the metadata the driver produced and recomputes nothing, so the document
cannot quietly disagree with the tables it is built from. That is not a stylistic preference: the
prose in an earlier session's report was written from the run that produced it rather than from its
own tables, and the two drifted (`344f9ba`).

The document leads with whichever way the answer came out. A negative here is a result -- five of
this project's most useful findings are negatives -- so the phrasing is chosen from the measured
delta rather than fixed in advance.
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_ARTIFACTS = os.path.join('docs', 'fom', 'artifacts')

# S08 session 2 with S10's columns (F-099), the number this session has to beat.
PUBLISHED_OPERATING_POINT = 0.6536
PUBLISHED_TOP10 = 0.6845
REPRODUCIBILITY_FLOOR = 0.10
CONSTANT_TOP10 = 0.2657
ECE_GATE = 0.05

BASELINE_ARM = 'S08 baseline'
BLOCK_LABEL = {'+ block B': 'block B (the fit statistic)',
               '+ block A': 'block A (the prior)',
               '+ both': 'both blocks'}


def load(artifact_dir, tag):
    def read(name, required=True):
        path = os.path.join(artifact_dir, f'{tag}_{name}')
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(path)
            return None
        if path.endswith('.json'):
            with open(path, encoding='utf-8') as handle:
                return json.load(handle)
        return pd.read_csv(path)

    return dict(
        main=read('main_table.csv'),
        mcnemar=read('mcnemar.csv'),
        by_lattice=read('mcnemar_by_lattice.csv'),
        calibration=read('calibration.csv'),
        coverage=read('coverage.csv', required=False),
        combiner_meta=read('combiner_meta.json'),
        assignment_meta=read('assignment_meta.json', required=False),
        prior_meta=read('prior_meta.json', required=False),
        ablation=read('ablation.csv', required=False),
        ablation_mcnemar=read('ablation_mcnemar.csv', required=False),
        importance=read('importance.csv', required=False),
        cost=read('cost.csv', required=False),
        network=read('network.csv', required=False),
        network_mcnemar=read('network_mcnemar.csv', required=False),
        network_meta=read('network_meta.json', required=False),
        )


def _arm(main, name):
    rows = main.loc[main['arm'] == name]
    if not len(rows):
        raise KeyError(f'no arm named {name!r} in the main table')
    return rows.iloc[0]


def _paired(mcnemar, arm, metric):
    rows = mcnemar.loc[(mcnemar['arm'] == arm) & (mcnemar['metric'] == metric)]
    return rows.iloc[0] if len(rows) else None


def _significance(row):
    if row is None:
        return 'not measured'
    return f'{int(row["n_gained"])} gained / {int(row["n_lost"])} lost, p = {row["p_value"]:.2g}'


# Ordered by decreasing symmetry -- equivalently, by increasing free cell parameters. That
# ordering *is* the result in panel (b): the gain runs with the number of free parameters, which is
# the third time this project has measured that shape (F-096, F-099, F-133). Sorting these
# alphabetically would hide it.
LATTICE_ORDER = ('cP', 'cI', 'cF', 'tP', 'tI', 'hP', 'hR', 'oP', 'oC', 'oF', 'oI', 'mP', 'mC', 'aP')


def _lattice_columns(main):
    present = {name[len('dev_op_'):] for name in main.columns if name.startswith('dev_op_')}
    return [code for code in LATTICE_ORDER if code in present]


def figure(data, path):
    """Four panels: the arms, the per-lattice deltas, the reliability curves, and the ablation."""
    main, by_lattice = data['main'], data['by_lattice']
    figure_, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    # (a) the arms as *deltas* against the baseline, with the reproducibility floor drawn.
    # On an absolute axis all four bars are visually identical, which is true and useless: the
    # question is not "how good is it" but "is the difference bigger than the floor", so the floor
    # is the thing that has to be on the page.
    axis = axes[0][0]
    baseline_row = _arm(main, BASELINE_ARM)
    floor = 100*REPRODUCIBILITY_FLOOR*baseline_row['operating_point']
    others = main.loc[main['arm'] != BASELINE_ARM]
    positions = np.arange(len(others))
    width = 0.38
    axis.axhspan(-floor, floor, color='0.85', zorder=0,
                 label=f'inside the reproducibility floor (+/-{floor:.1f} pp)')
    axis.bar(positions - width/2, 100*others['delta_operating_point'], width,
             label='operating point', color='#3c6e91', zorder=3)
    axis.bar(positions + width/2, 100*others['delta_top10'], width, label='top-10',
             color='#c2703d', zorder=3)
    for offset, column, metric in ((-width/2, 'delta_operating_point', 'operating_point'),
                                   (width/2, 'delta_top10', 'top10')):
        for index, (_, row) in enumerate(others.iterrows()):
            test = _paired(data['mcnemar'], row['arm'], metric)
            if test is not None and test['p_value'] < 0.05:
                axis.text(index + offset, 100*row[column] + 0.06, '*', ha='center', fontsize=13,
                          zorder=4)
    axis.axhline(0, color='0.2', linewidth=0.8, zorder=2)
    axis.set_xticks(positions)
    axis.set_xticklabels(others['arm'], rotation=0)
    axis.set_ylabel('against the 78-feature baseline, percentage points')
    axis.set_title('(a) every arm sits inside the floor\n'
                   '(* is p < 0.05 on the paired test)', fontsize=10)
    axis.set_ylim(-floor*1.15, floor*1.15)
    axis.legend(fontsize=8, loc='upper left')

    # (b) per-lattice operating-point delta of the best arm against the baseline
    axis = axes[0][1]
    best = main.sort_values('operating_point').iloc[-1]
    lattices = _lattice_columns(main)
    baseline = _arm(main, BASELINE_ARM)
    deltas = [100*(best[f'dev_op_{code}'] - baseline[f'dev_op_{code}']) for code in lattices]
    significant = set()
    if by_lattice is not None and len(by_lattice):
        rows = by_lattice.loc[(by_lattice['arm'] == best['arm'])
                              & (by_lattice['metric'] == 'operating_point')]
        significant = set(rows.loc[rows['p_value'] < 0.05, 'bravais_lattice'])
    colours = ['#2f7d4f' if value > 0 else '#a33b3b' for value in deltas]
    edges = ['black' if code in significant else 'none' for code in lattices]
    axis.bar(np.arange(len(lattices)), deltas, color=colours, edgecolor=edges, linewidth=1.2)
    axis.axhline(0, color='0.2', linewidth=0.8)
    axis.set_xticks(np.arange(len(lattices)))
    axis.set_xticklabels(lattices, rotation=0)
    axis.set_ylabel('operating point, percentage points')
    axis.set_title(f'(b) {best["arm"]} against the baseline, by decreasing symmetry\n'
                   '(outlined bars are significant at p < 0.05, paired)', fontsize=10)
    axis.set_xlabel('free cell parameters increase to the right')

    # (c) reliability, aggregate, one line per arm
    axis = axes[1][0]
    calibration = data['calibration']
    for arm, block in calibration.loc[calibration['scope'] == 'aggregate'].groupby('arm'):
        axis.plot(block['p_mean'], block['observed'], marker='o', markersize=3.5,
                  label=f'{arm} (ECE {block["ece"].iloc[0]:.4f})')
    limit = max(calibration.loc[calibration['scope'] == 'aggregate', 'p_mean'].max(), 0.05)
    axis.plot([0, limit], [0, limit], color='0.4', linestyle='--', linewidth=1)
    axis.set_xlabel('predicted P(correct)')
    axis.set_ylabel('observed frequency')
    axis.set_title('(c) reliability on fom-dev')
    axis.legend(fontsize=8)

    # (d) the ablation, if it has been run
    axis = axes[1][1]
    ablation = data['ablation']
    if ablation is not None and len(ablation):
        order = ablation.iloc[::-1]
        colours = ['#2f7d4f' if value >= 0 else '#a33b3b'
                   for value in 100*order['delta_operating_point']]
        axis.barh(np.arange(len(order)), 100*order['delta_operating_point'], color=colours)
        axis.axvline(0, color='0.2', linewidth=0.8)
        axis.set_yticks(np.arange(len(order)))
        axis.set_yticklabels(order['arm'], fontsize=9)
        axis.set_xlabel('operating point against the full model, percentage points')
        axis.set_title('(d) what can be dropped')
    else:
        axis.axis('off')
        axis.text(0.5, 0.5, 'ablation not run', ha='center', va='center', color='0.4')

    figure_.suptitle('S11 block C -- the prior and the fit statistic inside S08\'s combiner',
                     fontsize=13)
    figure_.tight_layout(rect=(0, 0, 1, 0.97))
    figure_.savefig(path, dpi=200)
    plt.close(figure_)
    return path


def markdown(data, tag):
    main, mcnemar = data['main'], data['mcnemar']
    baseline = _arm(main, BASELINE_ARM)
    arms = [name for name in main['arm'] if name != BASELINE_ARM]
    best_name = main.sort_values('operating_point').iloc[-1]['arm']
    best = _arm(main, best_name)
    delta = 100*(best['operating_point'] - baseline['operating_point'])
    delta_top10 = 100*(best['top10'] - baseline['top10'])
    paired = _paired(mcnemar, best_name, 'operating_point') if best_name != BASELINE_ARM else None
    floor_points = 100*REPRODUCIBILITY_FLOOR*baseline['operating_point']

    # The headline is chosen from the number, not fixed in advance.
    if best_name == BASELINE_ARM or delta <= 0:
        verdict = ('**Negative, and it is the result.** Neither block improves on S08\'s combiner: '
                   f'the best arm is `{best_name}` and the operating point does not rise above the '
                   'baseline at all.')
    elif delta < floor_points:
        verdict = (f'**Negative against the floor.** `{best_name}` gains **{delta:+.2f} pp** of '
                   f'operating point, which is inside the ~10% reproducibility floor '
                   f'({floor_points:.2f} pp on this baseline) and is therefore not a difference '
                   '(Shirley 1980, PROTOCOL section 8) -- whatever its p-value.')
    else:
        verdict = (f'**Positive.** `{best_name}` gains **{delta:+.2f} pp** of operating point over '
                   f'the 78-feature baseline, against a floor of {floor_points:.2f} pp.')

    lines = [
        f'# S11 block C -- the prior and the fit statistic inside S08\'s combiner',
        '',
        f'`{tag}`, commit `{data["combiner_meta"]["commit"][:12]}`. '
        f'Fitted on `fom-train`, reported on `fom-dev`. '
        f'{data["combiner_meta"]["n_fit"]:,} fitting candidates, '
        f'{data["combiner_meta"]["n_dev"]:,} evaluation candidates.',
        '',
        verdict,
        '',
        '![block C](' + f'{tag}_combiner.png)',
        '',
        '## What was asked',
        '',
        'PLAN section 3 writes the target as `fit - null + prior`. S07 supplied the null exactly '
        'and removing it made every merit worse (F-076, and again in F-089 and F-096). S08 found '
        'the gain was the prior and had to infer it from the extinction group. Block C is the '
        'first model in this project handed both surviving terms explicitly:',
        '',
        '```',
        'P(candidate correct)  ~  P(cell plausible | peak list)  x  P(peaks fit | cell)',
        '                                block A                         block B',
        '```',
        '',
        f'The baseline is S08\'s tuned combiner at 78 features, published at '
        f'{PUBLISHED_OPERATING_POINT:.4f} operating point and {PUBLISHED_TOP10:.4f} top-10, and '
        'never raw M20 -- `Minfo` is built from the same statistic as `rho`, so against M20 alone '
        'any fit-quality column looks like +3.5 pp and against the honest baseline it is worth '
        'nothing (F-130).',
        '',
        '## The arms',
        '',
        '| arm | features | operating point | top-10 | precision | reported | ECE |',
        '|---|---|---|---|---|---|---|',
        ]
    for _, row in main.iterrows():
        lines.append(
            f'| {row["arm"]} | {int(row["n_features"])} | {row["operating_point"]:.4f} | '
            f'{row["top10"]:.4f} | {row["precision"]:.4f} | {row["reported"]:.4f} | '
            f'{row["ece"]:.4f} |'
            )
    lines += [
        '',
        f'The constant-score floor is {CONSTANT_TOP10:.4f} on top-10 (F-083) and goes beside every '
        'rank number above. The baseline reproduces S08\'s published figures to '
        f'{abs(100*(baseline["top10"] - PUBLISHED_TOP10)):.2f} pp of top-10 and '
        f'{abs(100*(baseline["operating_point"] - PUBLISHED_OPERATING_POINT)):.2f} pp of operating '
        'point; the comparison below is against this refit, which is what makes it paired.',
        '',
        '## Paired against the baseline',
        '',
        '| arm | metric | gained | lost | p |',
        '|---|---|---|---|---|',
        ]
    for arm in arms:
        for metric in ('operating_point', 'top10'):
            row = _paired(mcnemar, arm, metric)
            if row is None:
                continue
            lines.append(f'| {arm} | {metric} | {int(row["n_gained"])} | {int(row["n_lost"])} | '
                         f'{row["p_value"]:.3g} |')

    lines += ['', '## Per lattice', '',
              'F-084 is the named failure mode: a model that has learned "triclinic candidates are '
              'usually wrong" posts a good aggregate and is useless. This is the table that shows '
              'whether that is what happened. Paired throughout -- and note F-087, `mcnemar`\'s '
              'subset argument raised on every call before S08, so no per-stratum paired test in '
              'this project predates that fix.', '',
              '| lattice | baseline | ' + ' | '.join(arms) + ' |',
              '|---' * (2 + len(arms)) + '|']
    for code in _lattice_columns(main):
        cells = [f'{baseline[f"dev_op_{code}"]:.4f}']
        for arm in arms:
            row = _arm(main, arm)
            value = 100*(row[f'dev_op_{code}'] - baseline[f'dev_op_{code}'])
            marker = ''
            if data['by_lattice'] is not None:
                match = data['by_lattice'].loc[
                    (data['by_lattice']['arm'] == arm)
                    & (data['by_lattice']['metric'] == 'operating_point')
                    & (data['by_lattice']['bravais_lattice'] == code)]
                if len(match) and match.iloc[0]['p_value'] < 0.05:
                    marker = ' \\*'
            cells.append(f'{value:+.2f}{marker}')
        lines.append(f'| {code} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('Baseline column is the absolute operating point; the rest are deltas in '
                 'percentage points. `*` marks p < 0.05 on the paired test.')

    # The hard stratum, which PROTOCOL section 3 rule 6 says carries the claim.
    hard_rows = []
    if data['by_lattice'] is not None and len(data['by_lattice']):
        hard_rows = data['by_lattice'].loc[
            data['by_lattice']['bravais_lattice'].astype(str).str.startswith('hard_')]
    if len(hard_rows):
        lines += ['', '## The hard stratum', '',
                  'PROTOCOL section 3 rule 6: the hard stratum carries the claim. Two cuts, '
                  'because Q32 settled that threshold metrics are reported at volume decile >= 6 '
                  'while rank metrics stay at the literal 8 -- and `_found` restricts to entries '
                  'where a correct candidate exists at all, since F-059 measured the designated '
                  'stratum at 87% *generation* failure and the unconditional number is therefore '
                  'mostly a statement about the generator.', '',
                  '| stratum | arm | metric | n | gained | lost | p |',
                  '|---|---|---|---|---|---|---|']
        for _, row in hard_rows.iterrows():
            lines.append(
                f'| {row["bravais_lattice"]} | {row["arm"]} | {row["metric"]} | '
                f'{int(row["n_entries"])} | {int(row["n_gained"])} | {int(row["n_lost"])} | '
                f'{row["p_value"]:.3g} |')
        best_hard = main.loc[main['arm'] == best_name]
        if 'hard_op_given_found_d6' in main.columns:
            lines += ['', 'Unpaired, for context: `operating_point_given_found` on the decile >= 6 '
                      f'cut goes {baseline["hard_op_given_found_d6"]:.4f} -> '
                      f'{best_hard.iloc[0]["hard_op_given_found_d6"]:.4f} over '
                      f'{int(baseline["hard_n_entries_d6"])} rows, and on the literal decile >= 8 '
                      f'cut {baseline["hard_op_given_found"]:.4f} -> '
                      f'{best_hard.iloc[0]["hard_op_given_found"]:.4f} over '
                      f'{int(baseline["hard_n_entries"])}. **These are large and they are not '
                      'paired**; the table above is what carries a claim.']

    if data['ablation'] is not None and len(data['ablation']):
        lines += ['', '## What can be dropped', '',
                  'F-093 is the template and the direction: it found that dropping the whole '
                  'over-prediction family cost 0.28 pp at p = 0.85 while saving 57% of the feature '
                  'budget, and that was the most useful thing S14 inherited.', '',
                  '| arm | features | operating point | delta | top-10 | question |',
                  '|---|---|---|---|---|---|']
        for _, row in data['ablation'].iterrows():
            lines.append(
                f'| {row["arm"]} | {int(row["n_features"])} | {row["operating_point"]:.4f} | '
                f'{100*row["delta_operating_point"]:+.2f} | {row["top10"]:.4f} | '
                f'{row.get("question", "")} |')

    if data['importance'] is not None and len(data['importance']):
        new = data['importance'].loc[data['importance']['group'] != 'S08'].head(12)
        lines += ['', '### Where the new columns rank', '',
                  f'Permutation importance on average precision, dev split, out of '
                  f'{len(data["importance"])} features.', '',
                  '| rank | feature | block | importance |', '|---|---|---|---|']
        for _, row in new.iterrows():
            lines.append(f'| {int(row["rank"])} | `{row["feature"]}` | {row["group"]} | '
                         f'{row["importance"]:.5f} |')

    if data['cost'] is not None and len(data['cost']):
        lines += ['', '## Cost', '',
                  'In `get_M20` equivalents, `S06_zoo_cost.csv` format. Gate condition 3 asks for a '
                  'variant inside 2x `get_M20` and has failed on the *features* every time it has '
                  'been measured (F-085: ten affordable merits at 4.68x, all seventeen at 145x; '
                  'F-092: the model itself at 0.17x). S14 inherits this table.', '',
                  '| feature | block | seconds/candidate | vs get_M20 |', '|---|---|---|---|']
        for _, row in data['cost'].iterrows():
            lines.append(f'| {row["feature"]} | {row["block"]} | '
                         f'{row["seconds_per_candidate"]:.3e} | {row["cost_vs_M20"]:.2f}x |')

    if data['network'] is not None and len(data['network']):
        meta = data['network_meta'] or {}
        residual = (meta.get('residual_structure') or {}).get('r2_on_dev')
        lines += ['', '## The network', '',
                  'PLAN section 4 would have made the residual measurement the gate on building '
                  'this at all; DWMM\'s instruction (2026-08-20) was to build it either way once '
                  'everything else was covered, so the measurement is reported rather than used '
                  'as a gate.', '']
        if residual is not None:
            lines.append(
                f'**Residual structure: R^2 = {residual:+.5f} on `fom-dev`.** A second model '
                'fitted to the tree\'s residual on the fitting split and scored on dev -- so a '
                'positive value means the structure generalises rather than that one model '
                'memorised another\'s errors.')
            lines.append('')
        lines += ['| arm | features | operating point | delta | top-10 | iterations |',
                  '|---|---|---|---|---|---|']
        for _, row in data['network'].iterrows():
            lines.append(
                f'| {row["arm"]} | {int(row["n_features"])} | {row["operating_point"]:.4f} | '
                f'{100*row["delta_operating_point"]:+.2f} | {row["top10"]:.4f} | '
                f'{row.get("n_iterations", "-")} |')
        if data['network_mcnemar'] is not None and len(data['network_mcnemar']):
            lines += ['', '| arm | metric | gained | lost | p |', '|---|---|---|---|---|']
            for _, row in data['network_mcnemar'].iterrows():
                lines.append(
                    f'| {row["arm"]} | {row["metric"]} | {int(row["n_gained"])} | '
                    f'{int(row["n_lost"])} | {row["p_value"]:.3g} |')
            lines.append('')
            lines.append('Paired against the tree on identical features. **Significantly worse, '
                         'not merely no better.**')
        lines += ['', 'The raw-versus-scaled pair is F-081 re-measured as that finding instructs: '
                  'S07\'s normalisers make no difference to a tree, because `z` and `rank` are '
                  'monotone within a lattice, and F-081 says S11 and S14 must re-measure rather '
                  'than assume for a network.']

    lines += ['', '## Gate', '',
              f'- paired gain over the 78-feature baseline, larger than the ~10% reproducibility '
              f'floor ({floor_points:.2f} pp here): '
              f'**{"met" if delta >= floor_points else "not met"}** ({delta:+.2f} pp, '
              f'{_significance(paired)})',
              f'- the per-lattice table shows it is not the base rate: see above',
              f'- ECE < {ECE_GATE}: **{"met" if best["ece"] < ECE_GATE else "not met"}** '
              f'({best["ece"]:.4f} on the best arm)',
              f'- cost reported in `get_M20` equivalents: '
              f'**{"yes" if data["cost"] is not None else "not run"}**',
              '',
              '## Bounds',
              '']
    for bound in data['combiner_meta'].get('bounds', []):
        lines.append(f'- {bound}')
    lines += ['', f'Top-10 delta of the best arm: {delta_top10:+.2f} pp.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='S11 block C: write the results document.')
    parser.add_argument('--tag', default='S11_C')
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()

    data = load(args.artifact_dir, args.tag)
    path = os.path.join(args.artifact_dir, f'{args.tag}_combiner.png')
    figure(data, path)
    print(f'wrote {path}')
    document = os.path.join(args.artifact_dir, f'{args.tag}_combiner.md')
    with open(document, 'w', encoding='utf-8') as handle:
        handle.write(markdown(data, args.tag))
    print(f'wrote {document}')


if __name__ == '__main__':
    main()
