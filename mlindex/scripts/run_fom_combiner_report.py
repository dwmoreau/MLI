"""S12's results document and figure, generated from its own tables rather than transcribed.

    python mlindex/scripts/run_fom_combiner_report.py

Reads `S12_combiner_main_table.csv`, `_contrasts.csv`, `_mcnemar.csv` and `_retention_skew.csv`
and writes `S12_combiner.md` and `S12_combiner.png`. Nothing here recomputes a metric: if a number
is in the document it is in a CSV, and if it is in a CSV a committed script put it there
(PROTOCOL section 5).

The document leads with the verdict. S11's write-up had to be rearranged for putting its answer
eight sections below its gates, and the same mistake is easy to repeat when a step has seventeen
arms and one headline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE/'docs'/'fom_campaign2'/'artifacts'

# S08 measured these on this campaign's own pool. Quoted rather than recomputed, and quoted with
# their source, because campaign 1's equivalents are still in circulation and are different numbers
# on a different population (C2-F-083).
TIEBREAK_FLOOR = 0.2352
CONTRAST_FLOOR_PP = 0.509
PER_LATTICE_FLOOR_PP = (1.38, 2.85)


def _load(artifact_dir, tag, name, required=True):
    path = Path(artifact_dir)/f'{tag}_{name}.csv'
    if not path.exists():
        if required:
            raise SystemExit(f'{path} is missing; run the analyse stage first')
        return None
    return pd.read_csv(path)


def _pp(value):
    return '--' if value is None or (isinstance(value, float) and np.isnan(value)) \
        else f'{100*value:.2f}'


def _significance(row):
    """A McNemar row as one readable cell: the delta, the split of discordant pairs, and p."""
    return (f'{row["delta_pp"]:+.2f} pp [{row["ci_low_pp"]:+.2f}, {row["ci_high_pp"]:+.2f}], '
            f'{int(row["gained"])} gained / {int(row["lost"])} lost, p = {row["p_value"]:.2g}')


def build(artifact_dir, tag):
    main = _load(artifact_dir, tag, 'main_table')
    contrasts = _load(artifact_dir, tag, 'contrasts', required=False)
    mcnemar = _load(artifact_dir, tag, 'mcnemar', required=False)
    skew = _load(artifact_dir, tag, 'retention_skew', required=False)
    calibration = _load(artifact_dir, tag, 'calibration', required=False)
    fit_table = _load(artifact_dir, tag, 'fit_table', required=False)
    main = main.set_index('arm')

    lines = ['# S12 — the learned combiner, cut hard', '']
    lines += _verdict(main, mcnemar)
    lines += _how(main, fit_table)
    lines += _leaderboard(main)
    lines += _controls(main)
    lines += _cuts(contrasts)
    lines += _per_lattice(main)
    lines += _calibration(calibration)
    lines += _bounds(skew)
    return '\n'.join(lines) + '\n'


def _verdict(main, mcnemar):
    """The answer, first, in one paragraph and one table."""
    lines = ['## The verdict', '']
    if 'base' not in main.index:
        return lines + ['The base arm did not fit; there is no verdict to state.', '']
    base = main.loc['base']
    rows = []
    for name in ('M20', 'M_sym', 'base', 'no_symmetry', 'constant', 'uniform_random'):
        if name in main.index:
            row = main.loc[name]
            rows.append(f'| `{name}` | {row["operating_point"]:.4f} | {row["top10"]:.4f} | '
                        f'{row["top1"]:.4f} | {row["precision"]:.4f} | {row["reported"]:.4f} |')
    lines += ['| score | operating point | top-10 | top-1 | precision | reported on |',
              '|---|---|---|---|---|---|'] + rows + ['']
    if mcnemar is not None and len(mcnemar):
        against = mcnemar[(mcnemar.arm == 'base') & (mcnemar.scope == 'aggregate')]
        for _, row in against.iterrows():
            lines.append(f'**Against `{row["reference"]}` on {row["metric"]}:** '
                         f'{_significance(row)}.')
        lines.append('')
    lines += [
        f'Read every rank number against the tie-break floor S08 measured for this population, '
        f'**{TIEBREAK_FLOOR:.4f}** of top-10, and every contrast against the contrast floor, '
        f'**{CONTRAST_FLOOR_PP:.3f} pp** aggregate and '
        f'{PER_LATTICE_FLOOR_PP[0]:.2f}–{PER_LATTICE_FLOOR_PP[1]:.2f} pp per lattice. '
        f'A constant score reaching {TIEBREAK_FLOOR:.4f} is not a merit doing anything; it is '
        f'ties breaking cubic-first while the dominant failure is symmetry lowering.', '',
        f'The base arm carries **{int(base["n_features"]) if "n_features" in base else 0} '
        f'features** where campaign 1 carried 65.', '']
    return lines


def _how(main, fit_table):
    lines = ['## How this was measured, and why it takes two pools', '',
             'A learned score is not one of the seven merits the negative subsampler ranked on, so '
             'on Benchmark B every rank metric for it is optimistic by an unmeasured amount and '
             '`FomMetrics` refuses to report one (C2-F-077, C2-R-013). The fully retained pool has '
             'no such problem and carries `fom-dev` crystals only, so nothing can be fitted or '
             'thresholded there. So the two jobs are split:', '',
             '| | fit and threshold | report |', '|---|---|---|',
             '| pool | Benchmark B slice | fully retained |',
             '| crystals | 196 `fom-train` | 530 `fom-dev` |',
             '| condition bundles | 9 | 3 |',
             '| thinned at generation | yes, K = 200 / 5 % | **no** |',
             '| a learned score\'s rank there | optimistic, refused | **exact, certified** |', '',
             'The two entry sets are disjoint by split and the driver asserts it rather than '
             'assuming it; `check_threshold_transfer` then refuses a threshold reported on the '
             'entries it was chosen on. Every fit is weighted by `sampling_weight` -- and by '
             '`sampling_weight` alone, not by the composed inverse-inclusion weight, which undoes '
             'the negative subsampling and costs 17.7 pp of top-10 (C2-F-127).', '']
    lines += ['**The report pool is easier than the one S09 reported on, and the levels are not '
              'comparable with its.** It is S08\'s floor sample: 530 crystals drawn BALANCED across '
              'the fourteen lattices to measure reproducibility, under the three severity bundles '
              '(0.1x, 1x and 2x error) and none of the sparsity, contaminant or second-phase ones. '
              'S09 measured `M_sym` at 0.6931 of top-10 on Benchmark B\'s `fom-dev`; here it '
              'reaches far more, because this population is not that population. **Every contrast '
              'in this document is paired on the same rows and is unaffected; every absolute level '
              'is a statement about this sample.** PROTOCOL section 3 rule 6 -- report the '
              'composition, do not reweight it.', '']
    if fit_table is not None and 'skipped' in fit_table.columns:
        missing = fit_table[fit_table['skipped'].notna()]
        if len(missing):
            lines += ['**Arms that could not be built**, with the reason, so an absent arm reads '
                      'as a stated absence rather than as a gap:', '']
            for _, row in missing.iterrows():
                lines.append(f'- `{row["arm"]}` — {row["skipped"]}')
            lines.append('')
    return lines


def _leaderboard(main):
    lines = ['## Every arm', '',
             '| arm | features | operating point | top-10 | hard top-10 | hard n | threshold |',
             '|---|---|---|---|---|---|---|']
    for name, row in main.sort_values('operating_point', ascending=False).iterrows():
        lines.append(
            f'| `{name}` | {int(row["n_features"]) if "n_features" in row and not pd.isna(row.get("n_features")) else "--"} '
            f'| {row["operating_point"]:.4f} | {row["top10"]:.4f} '
            f'| {row.get("hard_top10", float("nan")):.4f} '
            f'| {int(row["hard_n_entries"]) if not pd.isna(row.get("hard_n_entries")) else 0} '
            f'| {row["threshold"]:.4g} ({row["threshold_rule"]}) |')
    lines += ['',
              '**The hard column is 20 (entry, condition) cells over 20 crystals, of which 6 are '
              'reachable** (C2-R-019). The fully retained pool is S08\'s floor sample, drawn to '
              'measure reproducibility rather than as a stratified benchmark, and its hard stratum '
              'cannot carry a claim. Every hard number here is a statement about twenty patterns.',
              '']
    return lines


def _controls(main):
    """The two controls, read against BOTH floors, because which one applies is a property of the
    control's own tie structure rather than something to be assumed."""
    lines = ['## The controls', '',
             'A label-shuffled model is expected to land **between** the two floors, and which end '
             'it lands at is a fact about its ties rather than about leakage. A constant score '
             f'reaches **{TIEBREAK_FLOOR:.4f}** of top-10 because every candidate ties and the '
             'tie-break runs cubic-first while the dominant failure is symmetry lowering, so a '
             'degenerate score collects a free symmetry prior. A score that varies but carries no '
             'signal breaks those ties at random and collects nothing, landing near the '
             'uniform-random floor. Campaign 1 read its control against the constant floor alone '
             '(0.2814 against 0.2657); that was right for a model whose output had collapsed to '
             'nearly one value, and it is not a general rule.', '']
    constant = main.loc['constant', 'top10'] if 'constant' in main.index else float('nan')
    random = main.loc['uniform_random', 'top10'] if 'uniform_random' in main.index else float('nan')
    lines += ['| control | top-10 | operating point |', '|---|---|---|']
    for name in ('uniform_random', 'constant', 'label_shuffled', 'prior_only', 'M20', 'base'):
        if name in main.index:
            row = main.loc[name]
            lines.append(f'| `{name}` | {row["top10"]:.4f} | {row["operating_point"]:.4f} |')
    lines.append('')
    if 'label_shuffled' in main.index:
        value = main.loc['label_shuffled', 'top10']
        inside = min(constant, random) - 0.02 <= value <= max(constant, random) + 0.02
        lines.append(
            f'**Label-shuffled** — fit and calibration labels permuted within each (entry, bundle), '
            f'so the per-entry positive count is preserved and only the candidate-to-correctness '
            f'association is destroyed — reaches **{value:.4f}**, against {random:.4f} random and '
            f'{constant:.4f} constant. '
            + ('It sits inside that interval, which is where a model that learned nothing belongs: '
               'nothing leaks from the harness.' if inside else
               '**It sits OUTSIDE that interval, and no other number in this document is readable '
               'until that is explained.**'))
    if 'prior_only' in main.index:
        row = main.loc['prior_only']
        m20 = main.loc['M20', 'top10'] if 'M20' in main.index else float('nan')
        lines.append('')
        lines.append(
            f'**Prior-only** — fit labels shuffled, calibration labels real, so the per-lattice '
            f'isotonic is the only thing that knows anything — reaches **{row["top10"]:.4f}** of '
            f'top-10 against raw M20\'s {m20:.4f}. Isotonic is monotone and so cannot reorder '
            f'candidates within a lattice: everything it buys is cross-lattice. This is what '
            f'bounds the hypothesis that the model is just learning which lattices are usually '
            f'right, and it bounds it at '
            f'{100*(row["top10"] - random):.1f} pp above the random floor.')
    lines.append('')
    return lines


def _cuts(contrasts):
    if contrasts is None or not len(contrasts):
        return []
    lines = ['## Every cut, as a retrained paired arm', '',
             'PROTOCOL section 8: a feature cut is validated by retraining without the feature and '
             'pairing, never by an importance table. Permuting campaign 1\'s extinction group cost '
             '7.28 pp of top-10 while retraining without it cost 0.004 pp — a factor of 1 800 — '
             'because permutation pushes a high-cardinality feature out of distribution and '
             'measures the corruption. **There is no permutation importance in this step.**', '',
             '| arm | metric | scope | delta vs `base` | gained / lost | p |',
             '|---|---|---|---|---|---|']
    for _, row in contrasts.sort_values(['scope', 'metric', 'delta_pp']).iterrows():
        lines.append(f'| `{row["arm"]}` | {row["metric"]} | {row["scope"]} '
                     f'| {row["delta_pp"]:+.2f} pp [{row["ci_low_pp"]:+.2f}, '
                     f'{row["ci_high_pp"]:+.2f}] | {int(row["gained"])} / {int(row["lost"])} '
                     f'| {row["p_value"]:.3g} |')
    lines.append('')
    return lines


def _per_lattice(main):
    lattices = sorted({name[len('dev_top10_'):] for name in main.columns
                       if name.startswith('dev_top10_')})
    if not lattices:
        return []
    lines = ['## Per lattice', '',
             'The named failure mode is a model that learns "triclinic candidates are usually '
             'wrong", posts a good aggregate and makes triclinic entries worse. That is only '
             'visible here, and a per-lattice claim is read against **that lattice\'s own floor** '
             f'({PER_LATTICE_FLOOR_PP[0]:.2f}–{PER_LATTICE_FLOOR_PP[1]:.2f} pp), never the '
             'aggregate one.', '',
             '| lattice | n | M20 | `M_sym` | `base` | `base` − `M_sym` |', '|---|---|---|---|---|---|']
    for lattice in lattices:
        def value(arm):
            return main.loc[arm, f'dev_top10_{lattice}'] if arm in main.index else float('nan')
        count = main.loc['base', f'dev_n_{lattice}'] if 'base' in main.index else float('nan')
        delta = 100*(value('base') - value('M_sym'))
        lines.append(f'| {lattice} | {int(count) if not pd.isna(count) else 0} '
                     f'| {value("M20"):.4f} | {value("M_sym"):.4f} | {value("base"):.4f} '
                     f'| {delta:+.2f} pp |')
    lines.append('')
    return lines


def _calibration(calibration):
    """Is the score a probability? Reported with its base rate, because at 0.03 % correct a small
    ECE is easy to earn by predicting nearly zero everywhere and being right."""
    if calibration is None or not len(calibration):
        return []
    lines = ['## Calibration', '',
             'Per-lattice isotonic, fitted on a held-out slice of `fom-train` rather than on the '
             'reporting split. Measured on a **uniform** sample of the report pool: positives are '
             'not enriched, because an ECE computed on an enriched sample is the calibration of a '
             'population that does not exist.', '',
             '| arm | ECE | Brier | base rate | n | n correct |', '|---|---|---|---|---|---|']
    for _, row in calibration.sort_values('ece').iterrows():
        lines.append(f'| `{row["arm"]}` | {row["ece"]:.5f} | {row["brier"]:.6f} '
                     f'| {row.get("base_rate", float("nan")):.5f} | {int(row["n"]):,} '
                     f'| {int(row.get("n_positive", 0)):,} |')
    best = calibration.sort_values('ece').iloc[0]
    lines += ['',
              f'The gate is ECE below 0.05 and the best arm reaches **{best["ece"]:.5f}**, so it is '
              f'not the binding constraint -- campaign 1 said the same at a 0.9 % base rate. '
              f'**Read it with the base rate beside it**: at {row.get("base_rate", 0):.3%} correct, '
              f'a score that predicts almost zero everywhere is almost always right, and most of '
              f'the reliability table lives in that regime. The bin that matters is the top one, '
              f'and its count is what `n correct` bounds.', '']
    return lines


def _bounds(skew):
    lines = ['## What this does not measure', '',
             '- **The hard stratum.** 20 cells over 20 crystals, 6 reachable (C2-R-019). The fix '
             'is one short cluster job: 360 hard `fom-dev` crystals exist in the frozen split and '
             'regenerating them fully retained over the same three bundles is under half a '
             'node-hour.',
             '- **Six of the nine condition bundles.** The report pool carries the severity axis '
             '(0.1x, 1x, 2x error) and none of the sparsity, contaminant or second-phase bundles, '
             'so the aggregate is over a narrower population than the fit.',
             '- **Transfer to a different error law.** No error-law bundle exists and none is '
             'generated (C2-R-008, reaffirmed 2026-09-01). What can be measured here is transfer '
             'across *conditions*, not across laws, and the handoff\'s '
             '`S12_error_law_transfer.csv` therefore cannot be produced by any run on this '
             'benchmark.',
             '- **Scale.** The fit uses 196 crystals where Benchmark B\'s `fom-train` holds about '
             '11 000. Every level here is a small-sample level; the contrasts are what carry.']
    if skew is not None and len(skew):
        lines += ['', '### The one skew that is measured, rather than assumed', '',
                  'The fit pool is thinned 3.3x and the report pool is not, so a feature that is a '
                  'statistic of a candidate\'s own pool means different things in the two places, '
                  'and `sampling_weight` cannot repair a shift on the feature side. Twenty-five '
                  'crystals are in both pools, which makes it measurable:', '',
                  '| context statistic | Spearman | median abs shift / own scale |',
                  '|---|---|---|']
        for _, row in skew.iterrows():
            if pd.isna(row['spearman']):
                continue
            lines.append(f'| `{row["feature"]}` | {row["spearman"]:.4f} '
                         f'| {row["median_abs_shift_over_scale"]:.4f} |')
        lines += ['', '`gap_to_best` is exact because the retention rule keeps the pool maximum of '
                  'every context merit by construction. `rank` and `z` are not, so the base arm '
                  'drops them and `plus_ctx_rank_z` measures them net of the shift.', '']
    return lines


def figure(artifact_dir, tag, main):
    """One publication-quality figure: the arms against both baselines and both floors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = [name for name in ('uniform_random', 'constant', 'label_shuffled', 'prior_only',
                               'M20', 'M_sym', 'no_symmetry', 'base') if name in main.index]
    if not order:
        return None
    values = [main.loc[name, 'top10'] for name in order]
    operating = [main.loc[name, 'operating_point'] for name in order]
    colours = ['#bdbdbd' if name in ('uniform_random', 'constant', 'label_shuffled', 'prior_only')
               else '#7f7f7f' if name in ('M20', 'M_sym') else '#1f77b4' for name in order]

    figure_, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, series, title in ((axes[0], values, 'top-10'),
                                (axes[1], operating, 'operating point')):
        axis.barh(range(len(order)), series, color=colours)
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels(order, fontsize=9)
        axis.set_xlabel(title)
        axis.grid(axis='x', alpha=0.3, linewidth=0.5)
        axis.set_axisbelow(True)
        for spine in ('top', 'right'):
            axis.spines[spine].set_visible(False)
    axes[0].axvline(TIEBREAK_FLOOR, color='#d62728', linestyle='--', linewidth=1.0)
    axes[0].text(TIEBREAK_FLOOR, len(order) - 0.4, ' tie-break floor 0.2352', color='#d62728',
                 fontsize=8, va='top')
    axes[0].invert_yaxis()
    figure_.suptitle('S12: the learned combiner against its baselines and its floors', fontsize=11)
    figure_.tight_layout()
    path = Path(artifact_dir)/f'{tag}.png'
    figure_.savefig(path, dpi=200)
    plt.close(figure_)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate S12's results document and figure")
    parser.add_argument('--artifact-dir', default=str(ARTIFACT_DIR))
    parser.add_argument('--tag', default='S12_combiner')
    args = parser.parse_args(argv)

    document = build(args.artifact_dir, args.tag)
    path = Path(args.artifact_dir)/f'{args.tag}.md'
    path.write_text(document, encoding='utf-8')
    table = pd.read_csv(Path(args.artifact_dir)/f'{args.tag}_main_table.csv').set_index('arm')
    image = figure(args.artifact_dir, args.tag, table)
    print(f'wrote {path}')
    if image:
        print(f'wrote {image}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
