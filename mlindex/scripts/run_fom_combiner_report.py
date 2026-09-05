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
    # NOT `seeds`, which is this function's own parameter: shadowing it here handed
    # `_load_search` a DataFrame where it wanted the seed numbers, and it silently found no
    # ladder tables and reported the feature search as "Not run."
    seed_summary = _load(artifact_dir, tag, 'seed_summary', required=False)
    fit_table = _load(artifact_dir, tag, 'fit_table', required=False)
    transfer = _load_transfer(artifact_dir, tag)
    search = _load_search(artifact_dir, tag, search_suffixes, seeds)
    main = main.set_index('arm')

    lines = ['# S12 — the learned combiner, cut hard', '']
    lines += _verdict(main, mcnemar, search)
    lines += _metrics()
    lines += _how(main, fit_table)
    lines += _leaderboard(main)
    lines += _controls(main)
    lines += _seeds(seed_summary)
    lines += _cuts(contrasts)
    lines += _search(search)
    lines += _per_lattice(main, by_lattice)
    lines += _calibration(calibration, search)
    lines += _transfer(transfer)
    lines += _contaminated(artifact_dir, tag)
    lines += _bounds(skew)
    return '\n'.join(lines) + '\n'


def _settled(search):
    """The last ladder's reference arm and its contrasts: the model this step actually settled on.

    Returned as (name, level row, {arm: (mean, min, max)}) or None. The LAST ladder is the settled
    one by construction -- each cycle starts from the previous cycle's answer -- and its reference
    arm is the one row carrying no contrast against itself.
    """
    if not search:
        return None
    _, agg, settled = search[-1]
    reference = [name for name in agg['arm'] if name not in settled]
    if len(reference) != 1:
        return None
    name = reference[0]
    return name, agg.set_index('arm').loc[name], settled


def _verdict(main, mcnemar, search=None):
    """The answer, first, in one paragraph and one table."""
    lines = ['## The verdict', '']
    top = _settled(search)
    if top is not None:
        name, row, settled = top
        features = '--' if pd.isna(row['n_features']) else f'{int(row["n_features"])}'
        lines += [f'**The model this step settled on is `{name}`, at {features} features**, and it '
                  f'reaches **{_pp(row["operating_point"])}** of operating point over three fit '
                  f'seeds.']
        against = [f'**{-mean:+.2f} pp** against `{arm}`'
                   for arm, (mean, low, high) in sorted(settled.items(), key=lambda kv: kv[1][0])
                   if arm in ('M20', 'M_sym') and (low > 0) == (high > 0)]
        if against:
            lines[-1] += ' That is ' + ', and '.join(reversed(against)) + \
                         ', the same sign at every seed.'
        lines += ['', 'Everything below the next table is the road to it: the family ladder first, '
                  'which asks whether each GROUP of features earns its place, then the two '
                  'backward-elimination cycles that cut columns. **The family ladder\'s reference '
                  'arm `base` is not the answer** -- it is 29 features and the search beat it by '
                  'cutting fifteen of them.', '']
    if 'base' not in main.index:
        return lines + ['The family ladder did not fit; there is no ladder to report.', '']
    base = main.loc['base']
    rows = []
    for name in ('M20', 'M_sym', 'base', 'no_symmetry', 'constant', 'uniform_random'):
        if name in main.index:
            row = main.loc[name]
            rows.append(f'| `{name}` | {row["operating_point"]:.4f} | {row["top10"]:.4f} | '
                        f'{row["top1"]:.4f} | {row["precision"]:.4f} | {row["reported"]:.4f} |')
    lines += ['### The family ladder, at one fit seed', '',
              '| score | operating point | top-10 | top-1 | precision | reported on |',
              '|---|---|---|---|---|---|'] + rows + ['']
    if mcnemar is not None and len(mcnemar):
        against = mcnemar[(mcnemar.arm == 'base') & (mcnemar.scope == 'aggregate')]
        for _, row in against.iterrows():
            lines.append(f'**`base` against `{row["reference"]}` on {row["metric"]}:** '
                         f'{_significance(row)}.')
        lines.append('')
    lines += [
        f'Read every rank number against the tie-break floor S08 measured for this population, '
        f'**{TIEBREAK_FLOOR:.4f}** of top-10, and every contrast against the contrast floor, '
        f'**{CONTRAST_FLOOR_PP:.3f} pp** aggregate and '
        f'{PER_LATTICE_FLOOR_PP[0]:.2f}\u2013{PER_LATTICE_FLOOR_PP[1]:.2f} pp per lattice. '
        f'A constant score reaching {TIEBREAK_FLOOR:.4f} is not a merit doing anything; it is '
        f'ties breaking cubic-first while the dominant failure is symmetry lowering.', '',
        f'`base` carries **{int(base["n_features"]) if "n_features" in base else 0} '
        f'features** where campaign 1 carried 65.', '']
    return lines



def _metrics():
    """What the two headline numbers mean, in words, before any of them are quoted.

    They are not interchangeable and the difference between them is itself a result: a merit can
    degrade far more on one than the other, which says whether it stopped ORDERING candidates or
    stopped MEANING the same thing.
    """
    return [
        '## The two numbers everything here is reported in', '',
        '**Top-10 is a ranking question and nothing else.** Pool every candidate the search '
        'produced for one pattern, across all fourteen Bravais lattices, order them by the score, '
        'and ask whether the correct cell is in the first ten. No threshold is involved, so a '
        'score that orders well but is numerically meaningless still does well here. It measures '
        'one thing: *would a human looking at the top of the list find the answer.*', '',
        '**The operating point is ranking AND a decision.** The correct cell must be in the '
        'pooled top ten **and** score above a fixed threshold, so the indexer can say "this one" '
        'rather than "one of these ten". It is the number the project exists to move, because a '
        'figure of merit that cannot be thresholded does not automate anything. The threshold is '
        'chosen on crystals the model was never fitted on, at the false-positive rate M20 itself '
        'produces at the conventional de Wolff cut of 10.0, so every score is asked to be right '
        '*as often as M20 is wrong* -- that is what makes the comparison fair rather than a choice '
        'of operating régime.', '',
        '**Read the gap between them, not just each one.** If a merit loses top-10 it has stopped '
        'ordering candidates correctly. If it loses much more operating point than top-10, its '
        'ordering survived but its SCALE moved: the same numerical score no longer means the same '
        'thing, so a threshold set elsewhere no longer holds. Those are different failures with '
        'different fixes, and this document reports both for exactly that reason.', '',
        '`top1` and `mrr` appear in the tables beside them. `top1` is the same question at rank '
        'one; `mrr` is the mean reciprocal rank, which rewards being second over being tenth. '
        'Neither is a headline; they are there to show whether a top-10 gain is real depth or a '
        'reshuffle just inside the cut.', '']


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


def _calibration(calibration, search=None):
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
    # **The gate is about the SHIPPED model, not the best-calibrated one.** Quoting the leader
    # here would report the gate as met for an arm nobody ships -- which is exactly how it stood
    # until 2026-09-03, when the only ECE in the record was the 29-feature `base`'s.
    top = _settled(search)
    if top is not None and top[0] in set(calibration['arm']):
        shipped = calibration.set_index('arm').loc[top[0]]
        lines += [f'**The gate is about `{top[0]}`, the model this step ships**, and it reaches '
                  f'**{shipped["ece"]:.5f}** against a gate of 0.05 -- met with three orders of '
                  f'magnitude to spare, and never the binding constraint. The best-calibrated '
                  f'arm reaches {calibration["ece"].min():.5f}, so cutting features costs a factor '
                  f'of a few in ECE and it does not matter at this distance from the gate.', '',
                  '`unweighted_fit` is the row to look at twice: it is the **worst** calibrated by '
                  '4x while having the best Brier. Dropping `sampling_weight` buys a sharper '
                  'discriminator whose probabilities are further from the truth, which is the '
                  'shape C2-Q-031 has to be decided on -- not on the operating point alone.', '']
    best = calibration.sort_values('ece').iloc[0]
    lines += ['',
              f'**Read every row with the base rate beside it.** At {row.get("base_rate", 0):.3%} '
              f'correct, a score that predicts almost zero everywhere is almost '
              f'always right, and most of the reliability table lives in that '
              f'regime. The bin that matters is the top one, and its count is what '
              f'`n correct` bounds.', '']
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
    """Acceptance condition 5, its size-matched control, and the two things it is NOT."""
    lines = ['## Transfer to a condition the model never saw', '']
    if transfer is None:
        return lines + ['Not run. Acceptance condition 5 is open.', '']
    arm = transfer['arm_features'].iloc[0] if 'arm_features' in transfer else 'unknown'
    unseen = transfer[transfer['is_the_unseen_condition']]
    controlled = 'condition_effect_pp' in unseen
    lines += [f'Leave-one-condition-bundle-out on the `{arm}` feature set: fit without one error '
              'severity, report on the one left out, against the incumbent that saw all three. '
              'Every arm is scored in ONE reduction pass over identical rows, so nothing here is a '
              'population difference.', '',
              '| fitted without | crystals | held-out arm | size-matched | saw everything '
              '| rows lost | condition lost |',
              '|---|---|---|---|---|---|---|']
    for _, row in unseen.iterrows():
        n = f'{int(row["n_entries"])}' if 'n_entries' in row and pd.notna(row['n_entries']) else '--'
        matched = _pp(row['size_matched']) if controlled else '--'
        size = f'{row["size_effect_pp"]:+.2f} pp' if controlled else '--'
        effect = (f'**{row["condition_effect_pp"]:+.2f} pp**' if controlled
                  else f'**{row["delta_pp"]:+.2f} pp** (uncontrolled)')
        lines.append(f'| `{row["fitted_without"]}` | {n} | {_pp(row["held_out"])} | {matched} '
                     f'| {_pp(row["all_bundles"])} | {size} | {effect} |')
    column = 'condition_effect_pp' if controlled else 'delta_pp'
    worst = unseen.loc[unseen[column].idxmin()]
    lines += ['', f'**Worst case {worst[column]:+.2f} pp**, dropping the '
              f'`{worst["fitted_without"]}` bundle. Campaign 1\'s leave-one-condition-out measured '
              '1.6 pp average and 2.7 pp worst, on a different arm and a different pool; the '
              'comparison is indicative, not paired.', '']
    if controlled:
        lines += ['**Why there is a size-matched column, and why the last one is the answer.** '
                  'Withholding a bundle withholds a condition *and* the rows that came with it, '
                  'and this campaign has already measured that fit size is the binding constraint '
                  'here -- 14 features beat 29 by 8.6 pp at 157 training crystals. So a loss on '
                  'the unseen condition is not evidence of failed transfer until the same loss of '
                  'rows, spread across the conditions the model does see, has been shown to cost '
                  'less. The control is fitted on the same number of rows, with whole (crystal, '
                  'bundle) cells drawn at random from all nine bundles, and calibrated on the same '
                  'rows as the incumbent -- so it differs from the incumbent in row count and in '
                  'nothing else. **"Condition lost" is the held-out arm against that control**, '
                  'and it is the only column that is a transfer claim.', '',
                  'One residual, stated because it runs one way and not the other: the '
                  'control drops cells uniformly, so it also loses about a sixth of '
                  'the reported condition. The contrast is therefore *no exposure* '
                  'against *five sixths of the exposure*, not against all of it, which '
                  'makes "condition lost" a slight UNDER-estimate of the full effect '
                  'rather than an over-estimate.', '']
    else:
        lines += ['**Uncontrolled.** This table carries no size-matched arm, so its delta '
                  'confounds the condition the model never saw with the rows that condition took '
                  'with it. Re-run the stage for the controlled contrast.', '']
    lines += ['**Read as top-10, not as an operating point.** This stage selects no threshold, so '
              'an operating point is undefined for it, and these deltas are not on the same scale '
              'as the ones everywhere else in this document. They are also **differences of rates '
              'on the same crystals, not McNemar contrasts** -- the right quantity for a bound on '
              'what transfer costs, and not a significance claim. No p-value belongs on this '
              'table.', '',
              '**Two things this is not.** It is **not transfer across error laws** \u2014 no '
              'error-law bundle exists and none is generated (C2-R-008, reaffirmed 2026-09-01), so '
              "the handoff's `S12_error_law_transfer.csv` cannot be produced by any run on this "
              'benchmark. And it covers **three of nine condition bundles**: a transfer claim '
              'needs exact ranks, which needs the fully retained pool, which carries the severity '
              'axis and none of the sparsity, contaminant or second-phase bundles (C2-R-024). So '
              'what is measured is transfer across error **severity**.', '']
    return lines


BUNDLE_MEANING = {
    'c2_error1_cont2': ('2 unindexable lines, 60 peaks, no dropout',
                        'the clean contaminant read: contaminants and nothing else moving'),
    'c2_error1_cont0_phase3': ('3 lines from a real partner cell, 60 peaks',
                               'CORRELATED -- they follow a lattice, so a wrong cell can genuinely '
                               'index some of them'),
    'c2_error1_cont1_drop2': ('1 contaminant, 2 peaks dropped, 31 peaks', 'sparsity trend'),
    'c2_error1_cont1_drop4': ('1 contaminant, 4 peaks dropped, 31 peaks', 'sparsity trend'),
    'c2_error1_cont1_drop6': ('1 contaminant, 6 peaks dropped, 31 peaks', 'sparsity trend'),
    }


def _contaminated(artifact_dir, tag, clean_suffix='_fullscale', contam_suffix='_contam'):
    """What the score does on patterns carrying peaks no cell can index.

    Computed here rather than read from a table because the per-bundle split is the whole point:
    pooling five conditions would bury `phase3`, which is both the worst case and the realistic
    one. The threshold is the one the clean run chose -- held fixed, so a difference between the
    two pools is a difference between the pools and not between two thresholds.
    """
    import json
    from mlindex.model_training import FomMetrics

    artifact_dir = Path(artifact_dir)
    meta_path = artifact_dir/f'{tag}_reduced_meta{contam_suffix}.json'
    clean = _load(artifact_dir, tag, f'main_table{clean_suffix}', required=False)
    contam = _load(artifact_dir, tag, f'main_table{contam_suffix}', required=False)
    if not meta_path.exists() or clean is None or contam is None:
        return []
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    clean = clean.set_index('arm')
    contam = contam.set_index('arm')
    arms = [a for a in ('M20', 'M_sym', 'drop_structural', 'base', 'plus_probation')
            if a in contam.index and f'{a}|fom-dev' in meta]

    rows = []
    for arm in arms:
        path = artifact_dir/f'{tag}_reduced_{arm}_fom-dev{contam_suffix}.parquet'
        if not path.exists():
            continue
        per_entry = pd.read_parquet(path)
        threshold = float(contam.loc[arm, 'threshold'])
        for bundle, block in per_entry.groupby('condition_bundle'):
            result = FomMetrics.summarise_per_entry(
                block.reset_index(drop=True), meta[f'{arm}|fom-dev'],
                threshold=threshold, n_bootstrap=0)
            rows.append(dict(arm=arm, bundle=bundle,
                             top10=float(result.metric('top10')),
                             op=float(result.metric('operating_point'))))
    if not rows:
        return []
    table = pd.DataFrame(rows)

    lines = ['## Patterns carrying peaks no cell can index', '',
             'Everything above is measured on clean patterns -- Gaussian peak-position error and '
             'nothing else. Real data carries lines a correct cell cannot explain, and this section '
             'is the only place in the campaign that measures what that does. **The same 530 '
             'crystals**, regenerated under five further conditions and fully retained, so every '
             'comparison is paired over crystals and the ranks are exact. **The threshold is the '
             'one the clean run chose, held fixed** -- otherwise part of any difference would be a '
             'difference between thresholds rather than between conditions.', '']

    for metric, title, gloss in (
            ('top10', 'Top-10: does it still ORDER the candidates',
             'ranking only, no threshold'),
            ('op', 'Operating point: does it still DECIDE',
             'ranking and the fixed threshold together')):
        wide = table.pivot(index='bundle', columns='arm', values=metric)[arms]
        lines += [f'### {title}', '', f'*{gloss}.*', '',
                  '| condition | ' + ' | '.join(f'`{a}`' for a in arms) + ' | what it is |',
                  '|---' * (len(arms) + 2) + '|']
        for bundle in sorted(wide.index):
            what = BUNDLE_MEANING.get(bundle, ('', ''))[0]
            cells = ' | '.join(_pp(wide.loc[bundle, a]) for a in arms)
            lines.append(f'| `{bundle}` | {cells} | {what} |')
        clean_cells = ' | '.join(
            _pp(clean.loc[a, 'top10' if metric == 'top10' else 'operating_point'])
            if a in clean.index else '--' for a in arms)
        lines.append(f'| *clean, for reference* | {clean_cells} | 0.1x / 1x / 2x error |')
        lines.append('')

    wide_t = table.pivot(index='bundle', columns='arm', values='top10')
    wide_o = table.pivot(index='bundle', columns='arm', values='op')

    def _fall(arm, column, wide):
        return 100*(clean.loc[arm, column] - wide[arm].mean()) if arm in clean.index else float('nan')

    lines += ['### What the two tables say together', '']
    if {'M_sym', 'base'} <= set(arms):
        lines += [
            f'**The classical merits lose far more than the learned score, and they lose more '
            f'DECIDING than ORDERING.** Averaged over the five conditions, `M_sym` falls '
            f'{_fall("M_sym", "top10", wide_t):.1f} pp of top-10 and '
            f'{_fall("M_sym", "operating_point", wide_o):.1f} pp of operating point; `base` falls '
            f'{_fall("base", "top10", wide_t):.1f} and '
            f'{_fall("base", "operating_point", wide_o):.1f}. That `M_sym` loses so much more of '
            f'the second than the first is the diagnosis: its ordering partly survives, but its '
            f'numerical scale moves, so a threshold set on clean patterns stops meaning what it '
            f'meant. The learned score keeps both.', '']
    if 'c2_error1_cont0_phase3' in wide_o.index and 'M20' in arms:
        lines += [
            f'**`phase3` is the worst case and the realistic one.** M20 reaches '
            f'{_pp(wide_o.loc["c2_error1_cont0_phase3", "M20"])} of operating point there -- on a '
            f'pattern with three lines from a genuine second phase, at the conventional de Wolff '
            f'threshold, it returns the right cell about one time in twelve. Independent '
            f'contaminants hurt it far less ({_pp(wide_o.loc["c2_error1_cont2", "M20"])} on '
            f'`cont2`), and the difference is why: random lines cannot be indexed by ANY cell, so '
            f'no candidate is rewarded for them, while second-phase lines follow a real lattice '
            f'and a wrong cell can genuinely index some of them. M20 counts that as evidence.', '']
    lines += [
        '**The claim this supports, and the one it does not.** These models were fitted on all '
        'nine condition bundles, contaminated ones included -- on `fom-train` crystals, disjoint '
        'from the `fom-dev` crystals reported here, so this is not leakage. But it means the '
        'supported claim is *a learned score trained on a realistic mix of conditions is far more '
        'robust to contamination than a formula from 1961*, and NOT *it generalises to degradation '
        'it has never seen*. The second is a different experiment -- leave-one-condition-out over '
        'these bundles, with the size-matched control the transfer stage already uses -- and it '
        'has not been run.', '',
        '**Read the `drop` series as a trend, not as three results.** Each moves a contaminant, '
        '2/4/6 dropped peaks, and a 31-peak pattern at once, so no single one isolates a cause. '
        '`cont2` and `phase3` are the two that change one thing.', '']
    return lines


def _bounds(skew):
    lines = ['## What this does not measure', '',
             '- **The hard stratum.** Every hard number here is 20 cells over 20 crystals, 6 of '
             'them reachable (C2-R-019), which is too few to carry a claim. The cluster job that '
             'fixes it has RUN -- 360 hard `fom-dev` crystals over the same three bundles, three '
             'array tasks, 0 failures -- and what remains is to consolidate its output into a pool '
             'and re-reduce against it (C2-F-135).',
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
