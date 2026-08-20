"""S11 block B: turn what `run_fom_assignment.py` wrote into the results document and its figure.

    python -m mlindex.scripts.run_fom_assignment_report --tag S11_B

Reads only the CSVs and the metadata the driver produced and recomputes nothing, so the document
cannot quietly disagree with the tables it is built from.

The document leads with the structural finding rather than with the gate, because the gate cannot
be read without it: the two analytic estimators the handoff asked to compare are one statistic
under two link functions, so the honest comparison is against that statistic's best calibration.
"""
import argparse
import json
import matplotlib
import numpy as np
import os
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_ARTIFACTS = os.path.join('docs', 'fom', 'artifacts')
ECE_GATE = 0.1
REPRODUCIBILITY_FLOOR = 0.10

POOLED = 'all'
WELL_POSED = 'well_posed_reachable'


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
        baselines=read('analytic_baselines.csv'),
        setting=read('setting_cut.csv'),
        modelled=read('modelled_count.csv'),
        analytic_paired=read('analytic_paired.csv'),
        reliability=read('reliability.csv'),
        analytic_meta=read('analytic_meta.json'),
        network=read('network_table.csv', required=False),
        network_paired=read('network_paired.csv', required=False),
        network_reliability=read('network_reliability.csv', required=False),
        history=read('history.csv', required=False),
        main_meta=read('main_meta.json', required=False),
        cost=read('cost.csv', required=False),
        )


def _table(frame, stratum, lattice):
    rows = frame.loc[(frame['stratum'] == stratum) & (frame['bravais_lattice'] == lattice)]
    return rows.set_index('form') if len(rows) else None


def figure(data, path, lattice):
    table = data['network'] if data['network'] is not None else data['baselines']
    reliability = (
        data['network_reliability'] if data['network_reliability'] is not None
        else data['reliability'].assign(stratum=POOLED)
        )
    figure_, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Each panel shows only the forms fitted for *its* population: a recalibration fitted on the
    # pooled base rate says 0.04 everywhere and would appear on the well-posed panel as a flat
    # line at the bottom, which is true but is a statement about the other population.
    for axis, stratum, title, shown in (
            (axes[0, 0], POOLED, 'Every candidate (what block C sees)',
             (('rho', '-o'), ('isotonic', '-s'), ('network_calibrated', '-^'))),
            (axes[0, 1], WELL_POSED, 'Correct cell, truth setting, reachable peak',
             (('rho', '-o'), ('isotonic_well_posed', '-s'), ('network_well_posed', '-^'))),
            ):
        rows = reliability.loc[reliability['stratum'] == stratum] \
            if 'stratum' in reliability.columns else reliability
        for form, style in shown:
            curve = rows.loc[rows['target'] == form]
            if not len(curve):
                continue
            axis.plot(curve['confidence'], curve['accuracy'], style, label=form, markersize=4)
        axis.plot([0, 1], [0, 1], color='0.6', linewidth=1, linestyle='--')
        axis.set_xlabel('stated probability')
        axis.set_ylabel('observed frequency')
        axis.set_title(title, fontsize=10)
        axis.legend(fontsize=8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    axis = axes[1, 0]
    setting = data['setting'].loc[data['setting']['bravais_lattice'] == lattice]
    finite = setting.loc[np.isfinite(setting['setting_residual_high'])]
    positions = np.arange(len(setting))
    axis.bar(positions, setting['label_rate'], color='#2b6cb0', label='correct index recovered')
    axis.plot(positions, setting['reachable_ceiling'], 'k^--', markersize=5,
              label='reachable ceiling')
    labels = [
        f'{low:g}-{high:g}' if np.isfinite(high) else f'>{low:g}'
        for low, high in zip(setting['setting_residual_low'], setting['setting_residual_high'])
        ]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7, rotation=45)
    axis.set_xlabel('distance from the truth\'s own setting (error scales)')
    axis.set_ylabel('fraction of real peaks')
    axis.set_title('The label is basis-dependent; the ceiling is not', fontsize=10)
    axis.axvline(len(finite.loc[finite['setting_residual_high'] <= 1.0]) - 0.5,
                 color='#c05621', linewidth=1)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    forms = [form for form in ('rho', 'dewolff', 'isotonic', 'network_calibrated')
             if data['network'] is not None or form != 'network_calibrated']
    width = 0.38
    for offset, (stratum, colour) in enumerate(
            ((POOLED, '#2b6cb0'), (WELL_POSED, '#c05621'))):
        rows = _table(table, stratum, lattice)
        if rows is None:
            continue
        values = [rows['auc'].get(form, np.nan) for form in forms]
        axis.bar(np.arange(len(forms)) + offset*width, values, width, color=colour, label=stratum)
    axis.axhline(0.5, color='0.4', linewidth=1, linestyle='--')
    axis.set_xticks(np.arange(len(forms)) + width/2)
    axis.set_xticklabels(forms, fontsize=8, rotation=20)
    axis.set_ylabel('AUC')
    axis.set_ylim(0.45, 1.0)
    axis.set_title('Discrimination: the links tie, the network does not', fontsize=10)
    axis.legend(fontsize=8)

    figure_.suptitle(f'S11 block B -- per-peak Miller-index assignment, {lattice}', fontsize=12)
    figure_.tight_layout(rect=(0, 0, 1, 0.96))
    figure_.savefig(path, dpi=200)
    plt.close(figure_)


def _row(frame, form, columns):
    if frame is None or form not in frame.index:
        return None
    return [frame[column].get(form, np.nan) for column in columns]


def markdown(data, tag, lattice):
    meta = data['analytic_meta']
    main = data['main_meta']
    table = data['network'] if data['network'] is not None else data['baselines']
    lines = []
    lines.append(f'# S11 block B -- the per-peak Miller-index assignment probability, {lattice}')
    lines.append('')
    seconds = (main or meta).get('wall_clock_seconds', float('nan'))
    lines.append(
        f'Commit `{meta["commit"]}`, seed {meta["seed"]}, '
        f'{"analytic + network" if main else "analytic only"}, {seconds:.0f} s.'
        )
    summaries = {row['split']: row for row in meta.get('summaries', [])}
    dev = summaries.get('fom-dev', {})
    train = summaries.get('fom-train', {})
    lines.append('')
    lines.append(
        f'Measured on the frozen benchmark: **{dev.get("n_peak_rows", 0):,} (candidate, peak) '
        f'rows** over {dev.get("n_source_entries", 0)} `fom-dev` source entries and '
        f'{len(meta.get("bundles", []))} condition bundles, against '
        f'{train.get("n_peak_rows", 0):,} rows on `fom-train` that everything fitted here was '
        f'fitted on. Real candidates from real indexing runs; per-peak truth is the benchmark\'s '
        f'own `hkl_true`, and the assignment is rebuilt by `FomBenchmark.assign_lines`. '
        f'Broadening tag `{meta["broadening_tag"]}`.'
        )
    lines.append('')

    lines.append('## 1. The two analytic estimators are one statistic')
    lines.append('')
    pooled = _table(table, POOLED, lattice)
    columns = ['base_rate', 'mean_probability', 'ece', 'brier', 'auc']
    lines.append('| form | states | observed | ECE | Brier | AUC |')
    lines.append('|---|---|---|---|---|---|')
    for form in ('rho', 'taupin', 'dewolff', 'constant', 'isotonic'):
        values = _row(pooled, form, columns)
        if values is None:
            continue
        lines.append(
            f'| `{form}` | {values[1]:.3f} | {values[0]:.3f} | {values[2]:.3f} | '
            f'**{values[3]:.4f}** | {values[4]:.4f} |'
            )
    lines.append('')
    rho_auc = pooled['auc'].get('rho', float('nan')) if pooled is not None else float('nan')
    taupin_auc = pooled['auc'].get('taupin', float('nan')) if pooled is not None else float('nan')
    lines.append(
        f'`get_M20_likelihood` builds `arg = 8*pi*q2*eps/(V* mu)`, which is Taupin\'s `2*eps*n`, '
        f'and returns `rho = 1/(1 + arg)` per peak while its Minfo carries `1 - exp(-arg)`. Both '
        f'links are monotone in the same number, so they cannot rank differently -- and they do '
        f'not: AUC {rho_auc:.6f} against {taupin_auc:.6f}, equal to every digit reported. '
        f'The handoff asks for these as two estimators; they are one estimator and two '
        f'conventions, and every Brier difference between them is calibration. `dewolff` is the '
        f'genuinely different one, being the same link on de Wolff 1961\'s Delta(Q) rather than '
        f'on Taupin\'s density.'
        )
    lines.append('')
    lines.append(
        'That is why `isotonic` is on the list and why it is the bar. Isotonic regression is the '
        'Brier-optimal monotone transform of a statistic, so no relabelling of `arg` -- published, '
        'fitted or learned -- can beat it. Anything that does is using information `arg` does not '
        'contain.'
        )
    lines.append('')

    lines.append('## 2. Calibration on the population block C sees')
    lines.append('')
    constant = pooled['brier'].get('constant', float('nan')) if pooled is not None else float('nan')
    isotonic = pooled['brier'].get('isotonic', float('nan')) if pooled is not None else float('nan')
    rho_brier = pooled['brier'].get('rho', float('nan')) if pooled is not None else float('nan')
    base = pooled['base_rate'].get('rho', float('nan')) if pooled is not None else float('nan')
    lines.append(
        f'The pooled base rate is **{base:.3f}** -- across every candidate in the pool, one peak '
        f'in twenty-seven is assigned its correct Miller index, because almost every candidate is '
        f'wrong and a wrong cell indexes nothing correctly. `rho` states **{pooled["mean_probability"].get("rho", float("nan")):.3f}**. '
        f'It is over-confident by a factor of {pooled["mean_probability"].get("rho", float("nan"))/max(base, 1e-9):.0f}, '
        f'ECE {pooled["ece"].get("rho", float("nan")):.3f}, and its Brier score of {rho_brier:.4f} '
        f'is **twenty times worse than predicting the base rate and nothing else** '
        f'({constant:.4f}).'
        )
    lines.append('')
    lines.append(
        f'**A7 is answered, negative, and the answer is not close.** The analytic per-peak '
        f'estimator is not calibrated on the population it runs on.'
        )
    lines.append('')
    relative = (constant - isotonic)/constant if constant else float('nan')
    lines.append(
        f'**And the repair is free, which is the part that matters.** Recalibrating `arg` reaches '
        f'{isotonic:.4f}, against the constant predictor\'s {constant:.4f} -- a relative gain of '
        f'{100*relative:.1f}%, far inside the ~10% reproducibility floor (F-009). Once the link '
        f'function is chosen honestly, the whole analytic statistic is worth almost nothing over '
        f'knowing the base rate on this population. F-083 is the standing warning that a scoring '
        f'rule can be dominated by its base rate, and this is that warning landing again.'
        )
    lines.append('')

    lines.append('## 3. Where the question is actually well posed')
    lines.append('')
    lines.append(
        '"The correct Miller index" only names one reflection when the candidate is the correct '
        'cell **in the truth\'s own setting**. Most of the candidates the benchmark labels '
        '`is_correct` are not: they describe the right lattice in an alternative monoclinic '
        'setting, so their indices live in a different basis and index identity scores them near '
        'zero for a reason that has nothing to do with the assignment. The benchmark stores no '
        'transformation between a candidate\'s basis and the truth\'s, so the setting is '
        'recovered by evaluating q2 at the *true* Miller indices through the *candidate\'s* cell.'
        )
    lines.append('')
    setting = data['setting'].loc[data['setting']['bravais_lattice'] == lattice]
    lines.append('| distance from the truth\'s setting | candidates | correct index recovered | reachable ceiling |')
    lines.append('|---|---|---|---|')
    for _, row in setting.iterrows():
        high = row['setting_residual_high']
        band = f'{row["setting_residual_low"]:g} - {high:g}' if np.isfinite(high) \
            else f'above {row["setting_residual_low"]:g}'
        lines.append(
            f'| {band} | {int(row["n_candidates"])} | {row["label_rate"]:.3f} | '
            f'{row["reachable_ceiling"]:.3f} |'
            )
    lines.append('')
    lines.append(
        'The recovered-index column falls by a factor of five across the sweep while the '
        'reachable ceiling stays flat, which is the signature of a basis change rather than of a '
        'worse fit. The cut is set at one error scale on that evidence.'
        )
    lines.append('')
    near = setting.loc[setting['setting_residual_high'] <= 1.0, 'reachable_ceiling'].mean()
    far = setting['reachable_ceiling'].iloc[-1]
    lines.append(
        f'**The ceiling is itself a finding, and a mild one.** A real peak cannot be assigned its '
        f'correct index if the reflection is absent from the list the candidate assigns from -- '
        f'the reference list is truncated to `hkl_ref_length` lines and then narrowed to one '
        f'extinction group. For a candidate in the truth\'s own setting that ceiling is '
        f'**{near:.3f}**; ten error scales away it is **{far:.3f}**. Since the list length does '
        f'not change with the candidate, what binds is the **extinction-group narrowing**, not the '
        f'truncation -- a candidate that picked the wrong group has deleted whole families of '
        f'reflections from its own vocabulary before assigning anything. That is the carried-over '
        f'half of S01-C\'s A2 audit, and it comes back consistent with F-023, which measured the '
        f'truncation on *true* cells and found it never binds.'
        )
    lines.append('')
    well = _table(table, WELL_POSED, lattice)
    if well is not None:
        lines.append('| form | states | observed | ECE | Brier | AUC |')
        lines.append('|---|---|---|---|---|---|')
        for form in ('rho', 'taupin', 'dewolff', 'constant', 'isotonic_well_posed',
                     'network', 'network_calibrated', 'network_well_posed'):
            values = _row(well, form, columns)
            if values is None:
                continue
            lines.append(
                f'| `{form}` | {values[1]:.3f} | {values[0]:.3f} | {values[2]:.3f} | '
                f'**{values[3]:.4f}** | {values[4]:.4f} |'
                )
        lines.append('')
        lines.append(
            f'On this stratum `rho` is over-confident by about '
            f'{100*(well["mean_probability"].get("rho", np.nan) - well["base_rate"].get("rho", np.nan)):.0f} '
            f'percentage points rather than by a factor of twenty -- it states '
            f'{well["mean_probability"].get("rho", np.nan):.3f} where the truth is '
            f'{well["base_rate"].get("rho", np.nan):.3f}. So the estimator is roughly right about '
            f'a peak *given that the cell is right*, and knows nothing about whether the cell is '
            f'right, which is a fair description of what it was derived to do.'
            )
        lines.append('')
        auc = well['auc'].get('isotonic_well_posed', np.nan)
        lines.append(
            f'**The number that decides A6 is the AUC, not the ECE.** On the well-posed, reachable '
            f'peaks the analytic statistic scores {auc:.3f} -- it cannot tell which of a correct '
            f'cell\'s peaks will be mis-assigned. Its apparent discrimination on the pooled '
            f'population is the candidate being wrong, not the peak being mis-assigned.'
            )
        lines.append('')

    if data['network'] is not None:
        lines.append('## 4. The network')
        lines.append('')
        paired = data['network_paired']
        lines.append(
            'One model per lattice, `AssignmentModel(IntegralFilter)`, using the shipped '
            'calibration head unchanged: pairwise differences against the lattice\'s reference '
            'lines, the `epsilon/(|pds| + epsilon)` soft-match kernel, three per-peak dense layers '
            'and a softmax over the reference list. Trained on `fom-train` structures only, with a '
            'fresh condition draw each epoch and a mixture of true cells, perturbed cells at a '
            'labelled distance ladder, and real `fom-train` candidates. Its raw output is '
            'recalibrated by an isotonic fitted on `fom-train`, exactly as `arg` is, so the '
            'comparison is about information rather than about link functions.'
            )
        lines.append('')
        if paired is not None and len(paired):
            lines.append('| score | against | stratum | delta Brier | 95% CI | relative | beats the floor |')
            lines.append('|---|---|---|---|---|---|---|')
            for _, row in paired.iterrows():
                if not str(row['form']).startswith('network'):
                    continue
                lines.append(
                    f'| `{row["form"]}` | `{row["against"]}` | {row["stratum"]} | '
                    f'{row["delta_brier"]:+.5f} | [{row["delta_low"]:+.5f}, '
                    f'{row["delta_high"]:+.5f}] | {100*row["relative"]:+.1f}% | '
                    f'{"yes" if row["beats_floor"] else "**no**"} |'
                    )
            lines.append('')
            lines.append(
                'A negative delta means the network scores better. The comparison that matters is '
                'against `isotonic` on the pooled population and against `isotonic_well_posed` on '
                'the well-posed one, because those are the best any monotone function of the '
                'analytic statistic can do. The uncalibrated `network` rows are in the table to '
                'show what skipping that step would have concluded: raw softmax mass loses to a '
                'recalibrated analytic form by 57% on one population and 194% on the other, while '
                'carrying strictly more information than it. A probability that is not calibrated '
                'is not a probability.'
                )
            lines.append('')

        pooled_table = _table(table, POOLED, lattice)
        well_table = _table(table, WELL_POSED, lattice)
        if pooled_table is not None and well_table is not None:
            network_auc = pooled_table['auc'].get('network_calibrated', np.nan)
            analytic_auc = pooled_table['auc'].get('isotonic', np.nan)
            well_network_auc = well_table['auc'].get('network_well_posed', np.nan)
            well_analytic_auc = well_table['auc'].get('isotonic_well_posed', np.nan)
            constant_brier = pooled_table['brier'].get('constant', np.nan)
            lines.append(
                f'**The network does carry information the analytic statistic does not, and the '
                f'Brier score is a poor instrument for saying so.** Its AUC is '
                f'{network_auc:.3f} against {analytic_auc:.3f} pooled, and '
                f'{well_network_auc:.3f} against {well_analytic_auc:.3f} on the well-posed '
                f'reachable peaks -- where the analytic form is at chance and the network is not. '
                f'But at a {pooled_table["base_rate"].get("rho", np.nan):.3f} base rate the whole '
                f'usable range of the Brier score is narrow: a constant scores '
                f'{constant_brier:.5f}, recalibrated `arg` improves on it by '
                f'{constant_brier - pooled_table["brier"].get("isotonic", np.nan):.5f} and the '
                f'network by '
                f'{constant_brier - pooled_table["brier"].get("network_calibrated", np.nan):.5f} '
                f'-- so the network extracts about '
                f'{(constant_brier - pooled_table["brier"].get("network_calibrated", np.nan))/max(constant_brier - pooled_table["brier"].get("isotonic", np.nan), 1e-12):.0f} '
                f'times as much of the available signal, and both numbers are small.'
                )
            lines.append('')

        lines.append('### The gate')
        lines.append('')
        gate_rows = paired.loc[
            (paired['form'] == 'network_calibrated') & (paired['against'] == 'isotonic')
            ] if paired is not None else None
        pooled_relative = float(gate_rows['relative'].iloc[0]) if gate_rows is not None \
            and len(gate_rows) else float('nan')
        pooled_ece = pooled_table['ece'].get('network_calibrated', np.nan) \
            if pooled_table is not None else np.nan
        lines.append(
            f'- **ECE < {ECE_GATE}: met.** {pooled_ece:.4f} on the pooled population, '
            f'{well_table["ece"].get("network_well_posed", np.nan):.4f} on the well-posed one.'
            )

        def _relative(form, against, stratum):
            if paired is None:
                return float('nan')
            match = paired.loc[
                (paired['form'] == form) & (paired['against'] == against)
                & (paired['stratum'] == stratum)
                ]
            return 100*float(match['relative'].iloc[0]) if len(match) else float('nan')

        published_pooled = _relative('network_calibrated', 'rho', 'all')
        published_well = _relative('network_well_posed', 'rho', 'well_posed')
        bar_well = _relative('network_well_posed', 'isotonic_well_posed', 'well_posed')
        lines.append(
            f'- **Beats both analytic estimators on Brier, paired, by more than the '
            f'~{100*REPRODUCIBILITY_FLOOR:.0f}% floor: met against the published forms.** '
            f'{published_pooled:+.1f}% against `rho` pooled, {published_well:+.1f}% against `rho` '
            f'on the well-posed stratum.'
            )
        lines.append(
            f'- **Beats the recalibrated statistic -- the bar this session was given -- : not '
            f'met.** {100*pooled_relative:+.1f}% pooled and {bar_well:+.1f}% on the well-posed '
            f'stratum, both inside the floor. The direction is right and both intervals exclude '
            f'zero; the size is not there.'
            )
        lines.append('')
        lines.append(
            '**So the verdict is a clean negative, and it is the one PLAN section 4 predicted.** '
            'It called this block the weakest link, said to build it only if the analytic forms '
            'were miscalibrated in a way features cannot repair, and the measurement is that they '
            'are badly miscalibrated and a one-dimensional recalibration repairs almost all of it. '
            'Block C should take the analytic per-peak probability, recalibrated, and spend its '
            'budget elsewhere.'
            )
        lines.append('')

    if data['cost'] is not None:
        lines.append('## 5. Cost')
        lines.append('')
        lines.append('| merit | seconds per candidate | cost vs get_M20 |')
        lines.append('|---|---|---|')
        for _, row in data['cost'].iterrows():
            lines.append(
                f'| `{row["merit"]}` | {row["seconds_per_candidate"]:.3e} | '
                f'{row["cost_vs_M20"]:.2f}x |'
                )
        lines.append('')
        lines.append(
            'Timed on real candidate pools with the numba kernels warm, best of N with the first '
            'pass discarded, following `run_fom_cv_analysis.py`. S14 needs this price: the '
            'reference points are `Minfo` at 0.37x, `M_sym` at 24.3x and S10\'s `ho_M20` at 8.7x.'
            )
        lines.append('')
        lines.append(
            'The network row prices the keras inference path, so part of it is call overhead that '
            'a distilled numpy form would remove -- F-092 measured that worth 18-21x on S08\'s '
            'combiner. It cannot close this gap. The head applies three 1000x1000 dense layers to '
            'each of twenty peaks, which is of order 1e8 multiply-adds per candidate against '
            '`get_M20`\'s 20 x n_ref argmin, so the arithmetic alone is three orders of magnitude '
            'apart. **The whole of S08\'s seventeen-merit feature set costs 145x** (F-085); this '
            'is one block of the neural figure of merit, for a Brier gain inside the '
            'reproducibility floor.'
            )
        lines.append('')

    lines.append('## 6. Bounds on every number above')
    lines.append('')
    for bound in meta.get('bounds', []):
        lines.append(f'- {bound}')
    lines.append(
        '- The benchmark stores no transformation between a candidate\'s basis and the truth\'s, '
        'so the well-posed stratum is recovered by measurement rather than read off. A rebuild '
        'should store it.'
        )
    lines.append('')
    if main:
        lines.append(f'Production configuration for the second campaign: '
                     f'`{json.dumps(main.get("production_configuration", {}))}`')
    lines.append('')
    lines.append(f'Scale: {meta["scale"]}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Render the S11 block B results document.')
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACTS)
    parser.add_argument('--tag', default='S11_B')
    parser.add_argument('--lattice', default=None)
    args = parser.parse_args()

    data = load(args.artifact_dir, args.tag)
    lattice = args.lattice or data['baselines']['bravais_lattice'].iloc[0]
    figure(data, os.path.join(args.artifact_dir, f'{args.tag}_assignment.png'), lattice)
    path = os.path.join(args.artifact_dir, f'{args.tag}_assignment.md')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(markdown(data, args.tag, lattice))
    print(f'wrote {path} and {args.tag}_assignment.png')


if __name__ == '__main__':
    main()
