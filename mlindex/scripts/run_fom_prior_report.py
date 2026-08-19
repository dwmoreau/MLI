"""S11 session 1 results document: regenerate `S11_A_prior.md` and its figure from the CSVs.

    python -m mlindex.scripts.run_fom_prior_report

Reads only what `run_fom_prior.py` wrote, so every number in the document is traceable to a file on
disk and none of it lives in a notebook (PROTOCOL section 5). Nothing is recomputed here; if a
number is wrong the fit is re-run, not the report.
"""
import argparse
import json
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_ARTIFACTS = os.path.join('docs', 'fom', 'artifacts')

# S12's gate. A prior does not have to be sharp, it has to be honest, so these are the numbers the
# session is judged on and accuracy is not one of them.
ECE_GATE = 0.1
COVERAGE_TOLERANCE = 0.10


def load(artifact_dir, tag):
    import pandas as pd

    main = pd.read_csv(os.path.join(artifact_dir, f'{tag}_main_table.csv'))
    reliability = pd.read_csv(os.path.join(artifact_dir, f'{tag}_reliability.csv'))
    per_lattice = pd.read_csv(os.path.join(artifact_dir, f'{tag}_per_lattice.csv'))
    with open(os.path.join(artifact_dir, f'{tag}_meta.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    bootstrap_path = os.path.join(artifact_dir, f'{tag}_bootstrap.csv')
    bootstrap = pd.read_csv(bootstrap_path) if os.path.exists(bootstrap_path) else None
    return main, reliability, per_lattice, meta, bootstrap


def figure(main, reliability, per_lattice, path, arm=None):
    """Reliability, information gain per head, the volume interval check, and the per-lattice split.

    Publication quality from first production (PROTOCOL section 5) -- these are candidate paper
    figures and nobody wants to remake them.
    """
    arm = arm or main['arm'].iloc[0]
    row = main.loc[main['arm'] == arm].iloc[0]
    targets = [name for name in ('bravais', 'system', 'centring', 'n_free', 'high_symmetry')
               if f'{name}_ece' in main.columns]

    figure_, axes = plt.subplots(2, 2, figsize=(11, 9))
    figure_.suptitle(
        f'S11 block A -- volume and Bravais prior from the peak list ({arm} arm)\n'
        'development scale; calibration is the gate, accuracy is not',
        fontsize=12,
        )

    axis = axes[0, 0]
    subset = reliability.loc[(reliability['arm'] == arm)]
    for target in targets:
        rows = subset.loc[subset['target'] == target].sort_values('confidence')
        if rows.empty:
            continue
        axis.plot(rows['confidence'], rows['accuracy'], marker='o', markersize=4, label=target)
    axis.plot([0, 1], [0, 1], color='0.4', linestyle='--', linewidth=1, label='perfect')
    axis.set_xlabel('mean predicted confidence')
    axis.set_ylabel('observed accuracy')
    axis.set_title('Reliability, after temperature scaling')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    axis = axes[0, 1]
    gains = [row[f'{target}_information_gain_bits'] for target in targets]
    entropies = [row[f'{target}_base_rate_entropy_bits'] for target in targets]
    positions = np.arange(len(targets))
    axis.barh(positions, entropies, color='0.85', label='base-rate entropy')
    axis.barh(positions, gains, height=0.5, color='#2b6cb0', label='gain over base rate')
    axis.axvline(0.0, color='0.3', linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(targets)
    axis.set_xlabel('bits')
    axis.set_title('Information the peak list carries, in bits')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis='x')

    axis = axes[1, 0]
    levels = sorted(int(name.rsplit('_', 1)[1]) for name in main.columns
                    if name.startswith('volume_coverage_'))
    observed = [row[f'volume_coverage_{level}'] for level in levels]
    axis.plot([0, 100], [0, 100], color='0.4', linestyle='--', linewidth=1, label='nominal')
    axis.fill_between([0, 100], [-COVERAGE_TOLERANCE*100, 100 - COVERAGE_TOLERANCE*100],
                      [COVERAGE_TOLERANCE*100, 100 + COVERAGE_TOLERANCE*100],
                      color='0.85', label=f'+/-{int(COVERAGE_TOLERANCE*100)} pp gate')
    axis.plot(levels, np.asarray(observed)*100, marker='s', color='#2b6cb0',
              label='highest-density set')
    axis.set_xlabel('nominal coverage (%)')
    axis.set_ylabel('observed coverage (%)')
    axis.set_xlim(40, 100)
    axis.set_ylim(40, 100)
    axis.set_title('Volume: predictive-interval coverage')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    axis = axes[1, 1]
    rows = per_lattice.loc[per_lattice['arm'] == arm].sort_values('information_gain_bits')
    colours = ['#c05621' if value < 0 else '#2b6cb0' for value in rows['information_gain_bits']]
    axis.barh(np.arange(len(rows)), rows['information_gain_bits'], color=colours)
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows['bravais_lattice'], fontsize=8)
    axis.axvline(0.0, color='0.3', linewidth=1)
    axis.set_xlabel('bits gained over the base rate')
    axis.set_title('Per Bravais lattice -- negative means confidently wrong')
    axis.grid(alpha=0.3, axis='x')

    figure_.tight_layout(rect=(0, 0, 1, 0.94))
    figure_.savefig(path, dpi=200)
    plt.close(figure_)


def markdown(main, per_lattice, meta, tag, bootstrap=None):
    targets = [name for name in ('bravais', 'system', 'centring', 'n_free', 'high_symmetry')
               if f'{name}_ece' in main.columns]
    lines = []
    lines.append('# S11 block A -- volume scale and Bravais lattice from the peak list')
    lines.append('')
    lines.append(f'Commit `{meta["commit"]}`, seed {meta["seed"]}, stage `{meta["stage"]}`, '
                 f'{meta["wall_clock_seconds"]:.0f} s.')
    lines.append('')
    lines.append(f'**{meta["scale"]}** Fitted on {meta["n_fit_structures"]:,} source structures '
                 f'({meta["n_calibration_structures"]:,} held back for temperature scaling), '
                 f'evaluated on {meta["n_evaluation_rows"]:,} `fom-dev` entry-conditions from the '
                 f'benchmark\'s own peak lists. Broadening tag `{meta["broadening_tag"]}`, '
                 f'{len(meta["condition_bundles"])} condition bundles.')
    lines.append('')
    lines.append('The gate is **calibration, not accuracy** (S12, S13). A prior does not have to '
                 'be sharp; it has to be honest. Accuracy is reported below and is explicitly not '
                 'the criterion.')
    lines.append('')

    lines.append('## 1. The heads')
    lines.append('')
    lines.append('| arm | head | ECE | accuracy | base-rate entropy (bits) | gain over base rate '
                 '(bits) | temperature |')
    lines.append('|---|---|---|---|---|---|---|')
    for _, row in main.iterrows():
        for target in targets:
            lines.append(
                f'| {row["arm"]} | `{target}` | {row[f"{target}_ece"]:.4f} | '
                f'{row[f"{target}_accuracy"]:.3f} | '
                f'{row[f"{target}_base_rate_entropy_bits"]:.3f} | '
                f'**{row[f"{target}_information_gain_bits"]:+.3f}** | '
                f'{row[f"{target}_temperature"]:.2f} |'
                )
    lines.append('')
    lines.append(f'ECE gate is < {ECE_GATE}. The gain column is the honest statement of how much '
                 'lattice information a realistic peak list carries: base-rate entropy minus the '
                 'model\'s cross entropy, both in bits, with the base rate taken from the '
                 '*training* population so the baseline cannot borrow from the evaluation split. '
                 'A negative value means the head is worse than knowing nothing, which is the '
                 'failure mode a prior must not have.')
    lines.append('')

    if bootstrap is not None:
        lines.append('### Clustered intervals')
        lines.append('')
        lines.append('Resampled over **source entries**, not rows: each `fom-dev` entry appears once '
                     'per condition bundle and its six rows share a crystal and correlated noise, so '
                     'treating them as independent would shrink every interval by about the square '
                     'root of six (PROTOCOL section 8).')
        lines.append('')
        lines.append('| head | gain (bits) | 95% CI | ECE | 95% CI | accuracy | 95% CI |')
        lines.append('|---|---|---|---|---|---|---|')
        for _, row in bootstrap.iterrows():
            lines.append(
                f'| `{row["target"]}` | {row["information_gain_bits"]:+.3f} | '
                f'[{row["gain_ci_low"]:+.3f}, {row["gain_ci_high"]:+.3f}] | '
                f'{row["ece"]:.4f} | [{row["ece_ci_low"]:.4f}, {row["ece_ci_high"]:.4f}] | '
                f'{row["accuracy"]:.3f} | '
                f'[{row["accuracy_ci_low"]:.3f}, {row["accuracy_ci_high"]:.3f}] |'
                )
        lines.append('')
        lines.append(f'{int(bootstrap["n_entries"].iloc[0]):,} source entries carrying '
                     f'{int(bootstrap["n_rows"].iloc[0]):,} entry-conditions.')
        lines.append('')

    lines.append('## 2. Volume')
    lines.append('')
    levels = sorted(int(name.rsplit('_', 1)[1]) for name in main.columns
                    if name.startswith('volume_coverage_'))
    header = ' | '.join(f'{level}% coverage' for level in levels)
    lines.append(f'| arm | CRPS | NLL | median abs log ratio | {header} |')
    lines.append('|---|---|---|---|' + '---|'*len(levels))
    for _, row in main.iterrows():
        cells = ' | '.join(f'{row[f"volume_coverage_{level}"]*100:.1f}%' for level in levels)
        lines.append(
            f'| {row["arm"]} | {row["volume_crps"]:.4f} | {row["volume_nll"]:.3f} | '
            f'{row["volume_median_abs_log_ratio"]:.3f} | {cells} |'
            )
    lines.append('')
    lines.append('Intervals are highest-density sets over the branch grid, taken in log volume '
                 'because the branches are spaced geometrically. Highest-density rather than '
                 'central on purpose: a pattern consistent with V and with 2V is bimodal, and S12 '
                 'names averaging that away as the failure that matters most.')
    lines.append('')

    lines.append('## 3. The extraction grid')
    lines.append('')
    lines.append('| arm | n_volumes | n_filters | sigma | filter spacing | sigma / spacing | '
                 'at floor | branches per decade |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for _, row in main.iterrows():
        lines.append(
            f'| {row["arm"]} | {meta["n_volumes"]} | {meta["n_filters"]} | {row["sigma"]:.5f} | '
            f'{row["filter_spacing"]:.5f} | {row["sigma_over_spacing"]:.2f} | '
            f'{"**yes**" if row["sigma_at_floor"] else "no"} | '
            f'{row["branches_per_decade"]:.1f} |'
            )
    lines.append('')
    lines.append('`sigma / spacing == 2.00` means the width safeguard is holding sigma **up**, so '
                 'the filter grid is too coarse to resolve the peaks it renders and the model is '
                 'reading a smeared pattern. This is the one thing a single grid spanning all '
                 'fourteen lattices makes worse than a per-split-group one: the pooled q2 range is '
                 'wider, so the same filter count buys coarser spacing.')
    lines.append('')

    lines.append('## 4. Per Bravais lattice')
    lines.append('')
    lines.append('An aggregate alone is the anti-pattern PROTOCOL section 10 names, and F-084 '
                 'warns specifically that a model which has learned "triclinic is usually wrong" '
                 'looks fine in aggregate. Training is class balanced, so the base rate is not '
                 'available to be learned here at all.')
    lines.append('')
    lines.append('| arm | lattice | n | accuracy | mean P(truth) | gain over base rate (bits) |')
    lines.append('|---|---|---|---|---|---|')
    for _, row in per_lattice.iterrows():
        lines.append(
            f'| {row["arm"]} | {row["bravais_lattice"]} | {int(row["n"])} | '
            f'{row["accuracy"]:.3f} | {row["mean_probability_on_truth"]:.3f} | '
            f'{row["information_gain_bits"]:+.3f} |'
            )
    lines.append('')

    lines.append('## 5. Bounds on every number above')
    lines.append('')
    for bound in meta['bounds']:
        lines.append(f'- {bound}')
    lines.append('- Training is **class balanced**, which removes the Bravais base rate. The base '
                 'rate is part of the prior the combiner wants (F-086), so block C takes the '
                 'likelihood from this model and is given the base rate back as its own feature.')
    lines.append('- The pool this is evaluated on is censored at M20 >= 5 (**R1**), though that '
                 'bites on candidates rather than on entries, and this model scores entries.')
    lines.append('')
    lines.append(f'Production configuration for the second campaign: '
                 f'`{json.dumps(meta["production_configuration"])}`.')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Regenerate the S11 block A results document.')
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACTS)
    parser.add_argument('--tag', default='S11_A_prior')
    parser.add_argument('--arm', default=None)
    args = parser.parse_args()

    main_table, reliability, per_lattice, meta, bootstrap = load(args.artifact_dir, args.tag)
    figure(main_table, reliability, per_lattice,
           os.path.join(args.artifact_dir, f'{args.tag}.png'), arm=args.arm)
    document = markdown(main_table, per_lattice, meta, args.tag, bootstrap=bootstrap)
    path = os.path.join(args.artifact_dir, f'{args.tag}.md')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(document)
    print(f'wrote {path} and {args.tag}.png')


if __name__ == '__main__':
    main()
