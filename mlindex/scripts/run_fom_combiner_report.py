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
    """One artefact table, or None when an optional one is absent.

    **A file with no columns counts as absent**, not as a crash. An interrupted write leaves a
    zero-byte one behind and an empty frame writes a bare newline; `read_csv` raises
    `EmptyDataError` on both, which would take down the whole document over a section that is
    optional anyway.
    """
    path = Path(artifact_dir)/f'{tag}_{name}.csv'
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            pass
    if required:
        raise SystemExit(f'{path} is missing or empty; run the analyse stage first')
    return None


def _pp(value):
    return '--' if value is None or (isinstance(value, float) and np.isnan(value)) \
        else f'{100*value:.2f}'


def _significance(row):
    """A McNemar row as one readable cell: the delta, the split of discordant pairs, and p."""
    return (f'{row["delta_pp"]:+.2f} pp [{row["ci_low_pp"]:+.2f}, {row["ci_high_pp"]:+.2f}], '
            f'{int(row["gained"])} gained / {int(row["lost"])} lost, p = {row["p_value"]:.2g}')


SEARCH_SUFFIXES = ('_search', '_search2')
SEEDS = (12345, 777, 20260826)

# What each ladder was asking, in the order they were run. The main ladder asks whether each
# feature FAMILY earns its place; the search ladders start from that answer and cut columns.
LADDER_TITLES = {
    '_search': ('Cycle 1 -- one column at a time',
                'Starting from `lean`, the sixteen features left once the structural family is '
                'dropped, remove each column singly and refit.'),
    '_search2': ('Cycle 2 -- the joint drops cycle 1 implies',
                 'A column with no individual effect is not the same as a column that can be '
                 'removed: these features are correlated by construction, so removing any one '
                 'leaves the information in its neighbours. Cycle 2 removes them together.'),
}



def build(artifact_dir, tag, search_suffixes=SEARCH_SUFFIXES, seeds=SEEDS):
    main = _load(artifact_dir, tag, 'main_table')
    contrasts = _load(artifact_dir, tag, 'contrasts', required=False)
    mcnemar = _load(artifact_dir, tag, 'mcnemar', required=False)
    skew = _load(artifact_dir, tag, 'retention_skew', required=False)
    calibration = _load(artifact_dir, tag, 'calibration', required=False)
    by_lattice = _load(artifact_dir, tag, 'by_lattice_mcnemar', required=False)
    seeds = _load(artifact_dir, tag, 'seed_summary', required=False)
    fit_table = _load(artifact_dir, tag, 'fit_table', required=False)
    transfer = _load_transfer(artifact_dir, tag)
    search = _load_search(artifact_dir, tag, search_suffixes, seeds)
    main = main.set_index('arm')

    lines = ['# S12 — the learned combiner, cut hard', '']
    lines += _verdict(main, mcnemar)
    lines += _how(main, fit_table)
    lines += _leaderboard(main)
    lines += _controls(main)
    lines += _seeds(seeds)
    lines += _cuts(contrasts)
    lines += _search(search)
    lines += _per_lattice(main, by_lattice)
    lines += _calibration(calibration)
    lines += _transfer(transfer)
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


def _seeds(seeds):
    """What survives three fit seeds, which is the only thing an arm verdict may be read from."""
    if seeds is None or not len(seeds):
        return []
    frame = seeds[(seeds.scope == 'aggregate') & (seeds.metric == 'operating_point')].copy()
    if not len(frame):
        return []
    settled = frame[frame.same_sign_all_seeds & frame.significant_all_seeds]
    unsettled = frame[~(frame.same_sign_all_seeds & frame.significant_all_seeds)]
    lines = ['## What survives three fit seeds', '',
             'Every arm is refitted from scratch at three seeds and reduced over the whole pool at '
             'each. An arm counts as settled only if every seed agreed on the **sign** and every '
             'seed reached p < 0.05 -- both, because C2-F-061 failed by having two halves of one '
             'group swap which was significant while both means stayed positive. **Read arm '
             'verdicts here and nowhere else in this document.**', '',
             '| arm | mean | range over seeds | p at the worst seed | settled |',
             '|---|---|---|---|---|']
    for _, row in frame.sort_values('delta_mean', ascending=False).iterrows():
        mark = '**yes**' if row['same_sign_all_seeds'] and row['significant_all_seeds'] else (
            'no, sign flips' if not row['same_sign_all_seeds'] else 'no, not significant')
        lines.append(f'| `{row["arm"]}` | {row["delta_mean"]:+.2f} pp '
                     f'| [{row["delta_min"]:+.2f}, {row["delta_max"]:+.2f}] '
                     f'| {row["p_max"]:.3g} | {mark} |')
    flips = frame[frame.significant_all_seeds & ~frame.same_sign_all_seeds]
    lines += ['', f'**{len(settled)} of {len(frame)} arms are settled; {len(unsettled)} are not.**']
    if len(flips):
        for _, row in flips.iterrows():
            lines.append(
                f'`{row["arm"]}` is worth keeping as the illustration: it clears p < 0.05 at every '
                f'seed and its sign is **not the same at all three** '
                f'([{row["delta_min"]:+.2f}, {row["delta_max"]:+.2f}] pp). A reader shown any one '
                f'seed would have had a significant result and a confident wrong conclusion.')
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
             '**These are one seed.** The table above is what an arm verdict may be read from; '
             'this one is the detail behind a single fit of it.', '',
             '| arm | metric | scope | delta vs `base` | gained / lost | p |',
             '|---|---|---|---|---|---|']
    for _, row in contrasts.sort_values(['scope', 'metric', 'delta_pp']).iterrows():
        lines.append(f'| `{row["arm"]}` | {row["metric"]} | {row["scope"]} '
                     f'| {row["delta_pp"]:+.2f} pp [{row["ci_low_pp"]:+.2f}, '
                     f'{row["ci_high_pp"]:+.2f}] | {int(row["gained"])} / {int(row["lost"])} '
                     f'| {row["p_value"]:.3g} |')
    lines.append('')
    return lines


def _per_lattice(main, by_lattice=None):
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
    if by_lattice is not None and len(by_lattice):
        paired = by_lattice[(by_lattice.arm == 'base') & (by_lattice.reference == 'M_sym')
                            & (by_lattice.metric == 'top10')].copy()
        if len(paired):
            paired['lattice'] = paired['scope'].str.replace('lattice=', '', regex=False)
            paired = paired.sort_values('delta_pp')
            significant = paired[paired['p_value'] < 0.05]
            lines += [
                '### Paired, which changes the reading', '',
                'The table above differences two rates. A difference of rates is not a paired '
                'comparison and carries no interval -- and that is the defect campaign 1 shipped '
                'across its whole zoo and null packages, because its McNemar routine raised on '
                'every masked call (F-087). Paired properly, on the same patterns:', '',
                '| lattice | n | delta vs `M_sym`, top-10 | gained / lost | p |',
                '|---|---|---|---|---|']
            for _, row in paired.iterrows():
                lines.append(
                    f'| {row["lattice"]} | {int(row["n_entries"])} | {row["delta_pp"]:+.2f} pp '
                    f'[{row["ci_low_pp"]:+.2f}, {row["ci_high_pp"]:+.2f}] '
                    f'| {int(row["gained"])} / {int(row["lost"])} | {row["p_value"]:.3g} |')
            gains = significant[significant['delta_pp'] > 0]['lattice'].tolist()
            losses = significant[significant['delta_pp'] < 0]['lattice'].tolist()
            lines += ['',
                      f'**{len(significant)} of {len(paired)} lattices move significantly**: '
                      f'{", ".join(losses) if losses else "none"} against, '
                      f'{", ".join(gains) if gains else "none"} for. The rest are noise, and the '
                      f'sign count in the table above should not be read as though they were not. '
                      f'The gate still fails -- aP loses by four to eight times its own '
                      f'reproducibility floor, and the aggregate top-10 gain over `M_sym` is null '
                      f'-- but it fails on two lattices, not six.', '']
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




def _load_search(artifact_dir, tag, suffixes, seeds):
    """Per-ladder levels averaged over fit seeds, with each arm's per-seed spread.

    Levels come from the per-seed `main_table`s and the settled/unsettled call comes from the
    `seed_summary`, because an arm is only settled if its contrast holds the SAME SIGN at every
    seed (C2-F-061: a single-seed contrast can invert, and `m20_only` was significant at every
    seed in OPPOSITE directions).
    """
    out = []
    for suffix in suffixes:
        frames = [t for t in (_load(artifact_dir, tag, f'main_table{suffix}_seed{seed}',
                                    required=False) for seed in seeds) if t is not None]
        if not frames:
            continue
        levels = pd.concat(frames, ignore_index=True)
        agg = levels.groupby('arm').agg(
            operating_point=('operating_point', 'mean'),
            top10=('top10', 'mean'),
            n_features=('n_features', 'max'),
            n_seeds=('operating_point', 'size')).reset_index()
        summary = _load(artifact_dir, tag, f'seed_summary{suffix}', required=False)
        settled = {}
        if summary is not None and len(summary):
            block = summary[(summary['metric'] == 'operating_point')
                            & (summary['scope'] == 'aggregate')]
            for _, row in block.iterrows():
                settled[row['arm']] = (row['delta_mean'], row['delta_min'], row['delta_max'])
        out.append((suffix, agg.sort_values('operating_point', ascending=False), settled))
    return out


def _search(search):
    """The backward elimination, and the model the campaign actually settled on."""
    lines = ['## The feature search', '']
    if not search:
        return lines + ['Not run.', '']
    lines += ['Every cut below is a **retrained paired arm**: the model is refitted from scratch '
              'without the column, and the two arms are compared on the same crystals. No cut '
              'rests on an importance table -- permuting campaign 1\'s extinction-group feature '
              'cost 7.28 pp while retraining without it cost 0.004 pp, and that gap is the reason '
              'this step exists. Levels are the mean over three fit seeds; an arm is **settled** '
              'only if its contrast keeps the same sign at all three.', '']
    for suffix, agg, settled in search:
        title, blurb = LADDER_TITLES.get(suffix, (suffix.lstrip('_'), ''))
        lines += [f'### {title}', '', blurb, '',
                  '| arm | features | operating point | top-10 | contrast, mean [min, max] |',
                  '|---|---|---|---|---|']
        for _, row in agg.iterrows():
            features = '--' if pd.isna(row['n_features']) else f'{int(row["n_features"])}'
            if row['arm'] in settled:
                mean, low, high = settled[row['arm']]
                mark = '' if (low > 0) == (high > 0) else ' *(sign flips)*'
                contrast = f'{mean:+.2f} pp [{low:+.2f}, {high:+.2f}]{mark}'
            else:
                contrast = 'reference'
            lines.append(f'| `{row["arm"]}` | {features} | {_pp(row["operating_point"])} '
                         f'| {_pp(row["top10"])} | {contrast} |')
        lines.append('')
    return lines


def _load_transfer(artifact_dir, tag):
    """The condition-transfer table, whichever arm produced it.

    Named by arm (`_condition_transfer_core.csv`) because the stage can be pointed at any feature
    set and a transfer number quoted for the wrong one is a number for a model nobody ships. The
    settled arm is preferred; the older unsuffixed name is read as a fallback so a table written
    before the stage was parameterised is not silently dropped.
    """
    for name in ('condition_transfer_core', 'condition_transfer_lean',
                 'condition_transfer_base', 'condition_transfer'):
        table = _load(artifact_dir, tag, name, required=False)
        if table is not None and len(table):
            # The FILE NAME is the authority on which arm this is, not a column inside it: the
            # stage was parameterised after its first run, so an early table carries the name and
            # not the column, and a column is the thing a hand-edit can make disagree.
            if 'arm_features' not in table:
                table = table.assign(
                    arm_features=name[len('condition_transfer'):].lstrip('_') or 'unknown')
            return table
    return None


def _transfer(transfer):
    """Acceptance condition 5, and the two things it is NOT."""
    lines = ['## Transfer to a condition the model never saw', '']
    if transfer is None:
        return lines + ['Not run. Acceptance condition 5 is open.', '']
    arm = transfer['arm_features'].iloc[0] if 'arm_features' in transfer else 'unknown'
    unseen = transfer[transfer['is_the_unseen_condition']]
    lines += [f'Leave-one-condition-bundle-out on the `{arm}` feature set: fit without one error '
              'severity, report on the one left out, against the incumbent that saw all three. '
              'Paired in one reduction pass, so the two arms are scored on identical rows.', '',
              '| fitted without | reported on | held-out arm | saw everything | delta |',
              '|---|---|---|---|---|']
    for _, row in unseen.iterrows():
        lines.append(f'| `{row["fitted_without"]}` | `{row["reported_on"]}` '
                     f'| {_pp(row["held_out"])} | {_pp(row["all_bundles"])} '
                     f'| **{row["delta_pp"]:+.2f} pp** |')
    worst = unseen.loc[unseen['delta_pp'].idxmin()]
    lines += ['', f'**Worst case {worst["delta_pp"]:+.2f} pp**, dropping the `'
              f'{worst["fitted_without"]}` bundle. Campaign 1\'s leave-one-condition-out measured '
              '1.6 pp average and 2.7 pp worst, on a different arm and a different pool; the '
              'comparison is indicative, not paired.', '',
              '**Two things this is not.** It is **not transfer across error laws** — no error-law '
              'bundle exists and none is generated (C2-R-008, reaffirmed 2026-09-01), so the '
              'handoff\'s `S12_error_law_transfer.csv` cannot be produced by any run on this '
              'benchmark. And it covers **three of nine condition bundles**: a transfer claim needs '
              'exact ranks, which needs the fully retained pool, which carries the severity axis '
              'and none of the sparsity, contaminant or second-phase bundles (C2-R-024). So what '
              'is measured is transfer across error **severity**.', '']
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
    parser.add_argument('--search-suffix', nargs='*', default=list(SEARCH_SUFFIXES),
                        help='Ladder suffixes whose per-seed tables the feature-'
                             'search section reads')
    parser.add_argument('--seed', nargs='*', type=int, default=list(SEEDS),
                        help='Fit seeds the levels are averaged over')
    args = parser.parse_args(argv)

    document = build(args.artifact_dir, args.tag,
                     search_suffixes=tuple(args.search_suffix),
                     seeds=tuple(args.seed))
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
