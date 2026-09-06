"""S14's results document and figure, generated from its own tables rather than transcribed.

    python mlindex/scripts/run_fom_neural_report.py

Reads the `S14_neural_*` tables `run_fom_neural_score.py` writes and `S14_prior_interface.csv`,
and writes `S14_neural_score.md` and `S14_neural_score.png`. Nothing here recomputes a metric
(PROTOCOL section 5). The document leads with the verdict against the tree, reports rank and
threshold separately (handoff item 2), and puts every per-lattice number beside that lattice's
own contrast floor from S08.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.scripts import run_fom_combiner_report as s12_report

BASE = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE/'docs'/'fom_campaign2'/'artifacts'
TIEBREAK_FLOOR = s12_report.TIEBREAK_FLOOR
CONTRAST_FLOOR_PP = s12_report.CONTRAST_FLOOR_PP
_load = s12_report._load
_pp = s12_report._pp
_significance = s12_report._significance

LATTICES = ('cP', 'cI', 'cF', 'tP', 'tI', 'hP', 'hR', 'oP', 'oC', 'oF', 'oI', 'mP', 'mC', 'aP')
NETWORK = 'network'
REFERENCES = ('tree', 'tree_fullscale', 'M_sym', 'M20')


def lattice_floors(artifact_dir):
    """S08's per-lattice contrast floor (`M_sym` against M20 on top-10), as {lattice: pp}."""
    path = Path(artifact_dir)/'S08_floor_by_lattice.csv'
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    table = table[(table['merit'] == 'M_sym') & (table['baseline'] == 'M20')
                  & (table['metric'] == 'top10')]
    return {row['bravais_lattice']: float(row['se_pp']) for _, row in table.iterrows()}


def _row(table, reference, arm, metric, scope='aggregate'):
    if table is None:
        return None
    hit = table[(table['reference'] == reference) & (table['arm'] == arm)
                & (table['metric'] == metric) & (table['scope'] == scope)]
    return hit.iloc[0] if hit.shape[0] else None


def build(artifact_dir, tag):
    main = _load(artifact_dir, tag, 'main_table').set_index('arm')
    contrasts = _load(artifact_dir, tag, 'contrasts', required=False)
    mcnemar = _load(artifact_dir, tag, 'mcnemar', required=False)
    by_lattice = _load(artifact_dir, tag, 'by_lattice_mcnemar', required=False)
    rates = _load(artifact_dir, tag, 'answer_rates', required=False)
    calibration = _load(artifact_dir, tag, 'calibration', required=False)
    seed_summary = _load(artifact_dir, tag, 'seed_summary', required=False)
    fit_table = _load(artifact_dir, tag, 'fit_table', required=False)
    cost = _load(artifact_dir, tag, 'cost', required=False)
    interface = _load(artifact_dir, 'S14_prior', 'interface', required=False)
    floors = lattice_floors(artifact_dir)

    lines = ['# S14 — the neural scoring network', '']
    lines += _verdict(main, mcnemar, by_lattice, floors, seed_summary)
    lines += s12_report._metrics()
    lines += _how(main, fit_table)
    lines += _leaderboard(main)
    lines += _rank_and_threshold(main, mcnemar)
    lines += _controls(main)
    lines += _super_additivity(main, contrasts)
    lines += _block_a_as_features(main, contrasts, seed_summary)
    lines += _seeds(seed_summary)
    lines += _per_lattice(main, by_lattice, floors)
    lines += _answer_rates(rates)
    lines += _calibration(calibration, main)
    lines += _interface(interface)
    lines += _cost(cost)
    lines += _bounds(main, fit_table)
    return '\n'.join(lines) + '\n'


def _verdict(main, mcnemar, by_lattice, floors, seed_summary):
    lines = ['## The verdict', '']
    if NETWORK not in main.index:
        return lines + ['The `network` arm has not been reduced yet.', '']
    net = main.loc[NETWORK]
    lines.append(f'**The network reaches {_pp(net["operating_point"])} of operating point and '
                 f'{_pp(net["top10"])} of top-10** on the fully retained pool, from '
                 f'{int(net.get("n_features", 0))} inputs and none of the classical merits.')
    for reference in REFERENCES:
        if reference not in main.index:
            continue
        ref = main.loc[reference]
        parts = []
        for metric in ('operating_point', 'top10'):
            row = _row(mcnemar, reference, NETWORK, metric)
            if row is not None:
                parts.append(f'{metric.replace("_", " ")} {_significance(row)}')
        label = {'tree': "S12's tree refitted on the same rows",
                 'tree_fullscale': "S12's shipped full-scale tree (`plus_probation`)",
                 'M_sym': '`M_sym`', 'M20': 'M20'}[reference]
        lines.append(f'- **Against {label}** ({_pp(ref["operating_point"])} / '
                     f'{_pp(ref["top10"])}): ' + ('; '.join(parts) if parts else 'not paired'))
    lines.append('')
    lines.append(f'Read every contrast against the contrast floor S08 measured on this pool, '
                 f'**{CONTRAST_FLOOR_PP} pp** aggregate, and every rank number against the '
                 f'tie-break floor **{TIEBREAK_FLOOR}** of top-10.')
    if by_lattice is not None and 'tree' in main.index:
        sig = by_lattice[(by_lattice['reference'] == 'tree') & (by_lattice['arm'] == NETWORK)
                         & (by_lattice['metric'] == 'top10') & (by_lattice['p_value'] < 0.05)]
        wins = sorted(sig.loc[sig['delta_pp'] > 0, 'scope'].str.replace('lattice=', ''))
        losses = sorted(sig.loc[sig['delta_pp'] < 0, 'scope'].str.replace('lattice=', ''))
        lines.append('')
        lines.append(f'**Per lattice, paired against the refitted tree on top-10:** '
                     f'{len(wins)} lattice(s) significantly better ({", ".join(wins) or "none"}), '
                     f'{len(losses)} significantly worse ({", ".join(losses) or "none"}). The '
                     f'gate asks for a per-lattice win over the tree beyond that lattice\'s own '
                     f'floor; the per-lattice section below reads each against it.')
    if seed_summary is not None:
        settled = seed_summary[(seed_summary['reference'] == 'tree')
                               & (seed_summary['arm'] == NETWORK)
                               & (seed_summary['scope'] == 'aggregate')]
        if settled.shape[0]:
            lines.append('')
            lines.append('**Across fit seeds:** ' + '; '.join(
                f'{row["metric"]} {row["delta_mean"]:+.2f} pp [{row["delta_min"]:+.2f}, '
                f'{row["delta_max"]:+.2f}] over {int(row["n_seeds"])} seed(s), '
                f'{"settled" if row["settled"] else "not settled"}'
                for _, row in settled.iterrows()))
    lines.append('')
    return lines


def _how(main, fit_table):
    lines = ['## How this was measured', '',
             'The same two-pool design as S12 (decision 2026-09-01): every model is fitted and its '
             'threshold chosen on the Benchmark B slice\'s `fom-train` crystals, and reported on '
             'the fully retained pool\'s `fom-dev` crystals, where a learned score\'s rank is exact '
             '(C2-R-013). The two entry sets are disjoint by split and the driver asserts it. Every '
             'fit is weighted by `sampling_weight` except the arm that says otherwise.', '']
    if fit_table is not None:
        n_rows = fit_table['n_rows_fit'].dropna()
        n_pos = fit_table['n_positive_fit'].dropna()
        if n_rows.shape[0]:
            lines.append(f'Fit rows: **{int(n_rows.iloc[0]):,}** with **{int(n_pos.iloc[0]):,}** '
                         f'correct, on every arm. `tree_fullscale` is the exception: it is S12\'s '
                         f'shipped model, fitted on 2 381 244 rows (C2-F-143) and loaded here with '
                         f'that count asserted, so it is the campaign\'s reference LEVEL rather than '
                         f'a paired arm at equal fit size -- `tree` is that.')
        lines.append('')
        lines.append('| arm | kind | features | what it asks |')
        lines.append('|---|---|---|---|')
        for _, row in fit_table.iterrows():
            if 'skipped' in fit_table.columns and isinstance(row.get('skipped'), str):
                lines.append(f'| `{row["arm"]}` | {row.get("kind", "")} | -- | '
                             f'SKIPPED: {row["skipped"]} |')
                continue
            lines.append(f'| `{row["arm"]}` | {row.get("kind", "")} | '
                         f'{int(row["n_features"])} | {row.get("purpose", "")} |')
        lines.append('')
    return lines


def _leaderboard(main):
    lines = ['## Every arm', '',
             '| arm | operating point | top-10 | threshold only | precision | reported on | '
             'hard top-10 | hard n |', '|---|---|---|---|---|---|---|---|']
    for arm, row in main.sort_values('operating_point', ascending=False).iterrows():
        lines.append(f'| `{arm}` | {_pp(row["operating_point"])} | {_pp(row["top10"])} | '
                     f'{_pp(row["threshold_only"])} | {_pp(row["precision"])} | '
                     f'{_pp(row["reported"])} | {_pp(row.get("hard_top10", np.nan))} | '
                     f'{int(row["hard_n_entries"]) if "hard_n_entries" in row and np.isfinite(row["hard_n_entries"]) else "--"} |')
    lines += ['', '**The hard column is 20 cells over 20 crystals, 6 reachable** (C2-R-019); it '
              'carries no claim until the hard pool arrives from NERSC.', '']
    return lines


def _rank_and_threshold(main, mcnemar):
    lines = ['## Rank and threshold, reported separately', '',
             'Handoff item 2. Campaign 1\'s neural gain was entirely threshold (top-10 moved 22 '
             'gained / 22 lost); an entry-level input cannot reorder candidates inside an entry, so '
             'a rank gain is the evidence that a per-candidate prior re-ranks. `top10` is the rank '
             'half with no threshold; `threshold_only` is the score half with no rank.', '',
             '| arm | reference | top-10 (rank) | threshold only (scale) | operating point |',
             '|---|---|---|---|---|']
    if mcnemar is None:
        return lines + ['| -- | -- | not reduced | | |', '']
    for arm in sorted(set(mcnemar['arm'])):
        for reference in REFERENCES:
            cells = []
            for metric in ('top10', 'threshold_only', 'operating_point'):
                row = _row(mcnemar, reference, arm, metric)
                cells.append(_significance(row) if row is not None else '--')
            if any(cell != '--' for cell in cells):
                lines.append(f'| `{arm}` | `{reference}` | ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def _controls(main):
    lines = ['## The controls', '', '| control | top-10 | operating point |', '|---|---|---|']
    for name in ('uniform_random', 'constant', 'label_shuffled', 'M20', 'tree', NETWORK):
        if name in main.index:
            lines.append(f'| `{name}` | {_pp(main.loc[name, "top10"])} | '
                         f'{_pp(main.loc[name, "operating_point"])} |')
    lines += ['', 'A label-shuffled network must land between the uniform-random and constant '
              'floors; anything above the constant floor would be leakage from the harness.', '']
    return lines


def _super_additivity(main, contrasts):
    lines = ['## Super-additivity: block A, block B, both', '',
             'Handoff item 3. Campaign 1 found neither block significant alone and the two together '
             'significant -- the first direct evidence for `P(cell plausible) x P(peaks fit)`. '
             '`drop_B` is block A alone (the prior), `drop_A` is block B alone (the per-peak '
             'posteriors); both are paired against `network`, which carries both.', '',
             '| arm | operating point | top-10 | vs `network`, operating point | vs `network`, top-10 |',
             '|---|---|---|---|---|']
    for arm in ('drop_B', 'drop_A', 'plus_prior_claimed', 'plus_asg_sigma', 'unweighted_fit',
                'tree_plus_blocks'):
        if arm not in main.index:
            continue
        cells = []
        for metric in ('operating_point', 'top10'):
            row = _row(contrasts, NETWORK, arm, metric)
            cells.append(_significance(row) if row is not None else '--')
        lines.append(f'| `{arm}` | {_pp(main.loc[arm, "operating_point"])} | '
                     f'{_pp(main.loc[arm, "top10"])} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'A positive delta means the arm beats `network`. `tree_plus_blocks` is the tree '
              'given the network\'s inputs as well as its own: if it matches the network, the gain '
              'is the inputs; if the network beats it, the architecture is doing something.', '']
    return lines


RATIO_ARMS = ('tree_ratio_marginal', 'tree_ratio_claimed', 'tree_ratio_volume_only',
              'tree_ratio_dof_only', 'tree_plus_joint', 'tree_plus_blocks')


def _block_a_as_features(main, contrasts, seed_summary):
    lines = ['## Block A as features in the combiner (DWMM\'s redirect, 2026-09-05)', '',
             'The network is stopped; what only a network can do -- predict a volume and a symmetry '
             'from the peak list -- enters S12\'s tree instead. Two ratio features per candidate: '
             '`log(v_candidate / v_inferred)` (the lattice-marginal E[log V], or E[log V | claimed '
             'lattice], as separate arms) and `dof_candidate / E[dof]` from the prior\'s free-'
             'parameter head. Beside them, the principled readout (the joint P(V, lattice) at the '
             'claimed pair) and the whole block-A output. Every arm is S12\'s `plus_probation` '
             'feature set plus the named columns, refitted on the same rows and paired against '
             '`tree`.', '',
             '| arm | features | operating point | top-10 | vs `tree`, operating point | vs `tree`, top-10 |',
             '|---|---|---|---|---|---|']
    for arm in RATIO_ARMS:
        if arm not in main.index:
            continue
        cells = []
        for metric in ('operating_point', 'top10'):
            row = _row(contrasts, 'tree', arm, metric)
            cells.append(_significance(row) if row is not None else '--')
        lines.append(f'| `{arm}` | {int(main.loc[arm, "n_features"]) if np.isfinite(main.loc[arm, "n_features"]) else "--"} | '
                     f'{_pp(main.loc[arm, "operating_point"])} | {_pp(main.loc[arm, "top10"])} | '
                     + ' | '.join(cells) + ' |')
    lines.append('')
    if seed_summary is not None:
        rows = seed_summary[(seed_summary['reference'] == 'tree') & (seed_summary['scope'] == 'aggregate')
                            & seed_summary['arm'].isin(RATIO_ARMS)]
        if rows.shape[0]:
            lines += ['Over the fit seeds, against `tree`:', '',
                      '| arm | metric | mean | range | p at the worst seed | settled |',
                      '|---|---|---|---|---|---|']
            for _, row in rows.sort_values(['metric', 'delta_mean'], ascending=[True, False]).iterrows():
                lines.append(f'| `{row["arm"]}` | {row["metric"]} | {row["delta_mean"]:+.2f} pp | '
                             f'[{row["delta_min"]:+.2f}, {row["delta_max"]:+.2f}] | '
                             f'{row["p_max"]:.3g} | {"**yes**" if row["settled"] else "no"} |')
            lines.append('')
    return lines


def _seeds(seed_summary):
    lines = ['## What survives three fit seeds', '']
    if seed_summary is None:
        return lines + ['One fit seed so far. An arm verdict is read only from a table that has '
                        'all three (PROTOCOL section 8).', '']
    lines += ['| reference | arm | metric | mean | range over seeds | p at the worst seed | settled |',
              '|---|---|---|---|---|---|---|']
    subset = seed_summary[seed_summary['scope'] == 'aggregate']
    for _, row in subset.sort_values(['reference', 'metric', 'delta_mean'],
                                     ascending=[True, True, False]).iterrows():
        lines.append(f'| `{row["reference"]}` | `{row["arm"]}` | {row["metric"]} | '
                     f'{row["delta_mean"]:+.2f} pp | [{row["delta_min"]:+.2f}, '
                     f'{row["delta_max"]:+.2f}] | {row["p_max"]:.3g} | '
                     f'{"**yes**" if row["settled"] else "no"} |')
    lines.append('')
    return lines


def _per_lattice(main, by_lattice, floors):
    lines = ['## Per lattice', '',
             'The named failure mode is a model that learns "triclinic candidates are usually '
             'wrong", posts a good aggregate and makes triclinic entries worse. Each lattice is '
             'read against its OWN contrast floor from S08 (`S08_floor_by_lattice.csv`), never the '
             'aggregate one.', '']
    if NETWORK in main.index:
        lines += ['| lattice | n | M20 | `M_sym` | `tree` | `network` | network − tree | floor (pp) |',
                  '|---|---|---|---|---|---|---|---|']
        for lattice in LATTICES:
            key = f'dev_top10_{lattice}'
            if key not in main.columns:
                continue
            cells = [f'{int(main.loc[NETWORK, f"dev_n_{lattice}"])}']
            for arm in ('M20', 'M_sym', 'tree', NETWORK):
                cells.append(_pp(main.loc[arm, key]) if arm in main.index else '--')
            delta = (100*(main.loc[NETWORK, key] - main.loc['tree', key])
                     if 'tree' in main.index else np.nan)
            cells.append('--' if np.isnan(delta) else f'{delta:+.2f} pp')
            cells.append(f'{floors[lattice]:.2f}' if lattice in floors else '--')
            lines.append(f'| {lattice} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    if by_lattice is not None:
        for reference in ('tree', 'M_sym'):
            rows = by_lattice[(by_lattice['reference'] == reference)
                              & (by_lattice['arm'] == NETWORK) & (by_lattice['metric'] == 'top10')]
            if not rows.shape[0]:
                continue
            lines += [f'### Paired against `{reference}`, top-10', '',
                      '| lattice | n | delta | gained / lost | p | floor (pp) | in floors |',
                      '|---|---|---|---|---|---|---|']
            for _, row in rows.sort_values('delta_pp').iterrows():
                lattice = row['scope'].replace('lattice=', '')
                floor = floors.get(lattice, np.nan)
                lines.append(f'| {lattice} | {int(row["n_entries"])} | {row["delta_pp"]:+.2f} pp '
                             f'[{row["ci_low_pp"]:+.2f}, {row["ci_high_pp"]:+.2f}] | '
                             f'{int(row["gained"])} / {int(row["lost"])} | {row["p_value"]:.3g} | '
                             f'{floor:.2f} | '
                             f'{"--" if np.isnan(floor) or floor == 0 else f"{row["delta_pp"]/floor:+.1f}"} |')
            lines.append('')
    return lines


def _answer_rates(rates):
    lines = ['## Every score forced to answer on the same fraction of patterns', '',
             'C2-F-142\'s check: a calibrated probability and a raw merit are not comparable at a '
             'fixed threshold, and the incomparability flatters the calibrated one. Here every arm\'s '
             'threshold is set so it reports on 75 % and on 90 % of patterns.', '']
    if rates is None:
        return lines + ['Not computed.', '']
    lines += ['| arm | top-10 | operating point @75 % | precision @75 % | operating point @90 % | '
              'precision @90 % |', '|---|---|---|---|---|---|']
    for _, row in rates.sort_values('op_at_90', ascending=False).iterrows():
        lines.append(f'| `{row["arm"]}` | {_pp(row["top10"])} | {_pp(row["op_at_75"])} | '
                     f'{_pp(row["precision_at_75"])} | {_pp(row["op_at_90"])} | '
                     f'{_pp(row["precision_at_90"])} |')
    lines.append('')
    return lines


def _calibration(calibration, main=None):
    lines = ['## Calibration, and saturation at the calibrator\'s maximum', '',
             'Measured on a uniform sample of the report pool, positives not enriched. `top at max` '
             'is the share of PATTERNS whose top candidate carries the arm\'s maximum calibrated '
             'score -- S12\'s tree put 55.2 % of clean patterns there, a step rather than a ranking '
             'at high confidence, which the handoff names as the opening for a continuous head. '
             'The network\'s head is continuous and it still saturates: the per-lattice isotonic '
             'maps its top bin to the maximum for any model (C2-F-151).', '']
    if calibration is None:
        return lines + ['Not computed.', '']
    lines += ['| arm | ECE | Brier | base rate | top at max | n | n correct |',
              '|---|---|---|---|---|---|---|']
    for _, row in calibration.sort_values('ece').iterrows():
        top = (main.loc[row['arm'], 'top_score_at_max']
               if main is not None and row['arm'] in main.index
               and 'top_score_at_max' in main.columns else np.nan)
        lines.append(f'| `{row["arm"]}` | {row["ece"]:.5f} | {row["brier"]:.6f} | '
                     f'{row["base_rate"]:.5f} | {_pp(top) if np.isfinite(top) else "--"} % | '
                     f'{int(row["n"]):,} | {int(row["n_positive"]):,} |')
    lines.append('')
    return lines


def _interface(interface):
    lines = ['## The prior\'s untrained classes, before and after', '',
             'Gate condition 1. The shipped prior was trained on eleven lattices and kept a '
             'fourteen-class head; the three cubic classes never saw a positive and read as '
             'probabilities of about e^-19 for every pattern. `raw_head` is that head as campaign 1 '
             'consumed it; `support_masked` renormalises over the trained lattices and reads NaN '
             'outside them; a `main14` model, where present, is the fourteen-lattice retrain. Macro '
             'F1 with the predicted share beside it, per gate condition 3 -- never recall alone.', '']
    if interface is None:
        return lines + ['`S14_prior_interface.csv` not present.', '']
    for (model, readout), group in interface.groupby(['model', 'readout'], sort=False):
        f1_support = group.loc[group['in_support'], 'f1'].mean()
        f1_all = group['f1'].mean()
        lines += [f'### `{model}`, `{readout}` -- macro F1 over the support {f1_support:.3f}, '
                  f'over all fourteen {f1_all:.3f}', '',
                  '| lattice | in support | n | precision | recall | F1 | predicted share | '
                  'true share | median log P | median rank | max P |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for _, row in group.iterrows():
            lines.append(f'| {row["bravais_lattice"]} | {"yes" if row["in_support"] else "**no**"} '
                         f'| {int(row["n"])} | {row["precision"]:.3f} | {row["recall"]:.3f} | '
                         f'{row["f1"]:.3f} | {row["predicted_share"]:.3f} | {row["true_share"]:.3f} '
                         f'| {row["median_log_probability"]:.2f} | {row["median_rank"]:.0f} | '
                         f'{row["max_probability"]:.2e} |')
        lines.append('')
    return lines


def _cost(cost):
    lines = ['## Cost, in `get_M20` units', '',
             'Recorded, not gating (decision 2026-08-25). Block A is one forward pass per pattern, '
             'amortised over the pattern\'s own pool; block B and the network are per candidate.', '']
    if cost is None:
        return lines + ['Not computed.', '']
    lines += ['| step | microseconds per candidate | `get_M20` units |', '|---|---|---|']
    for _, row in cost.iterrows():
        lines.append(f'| {row["step"]} | {row["microseconds_per_candidate"]:.1f} | '
                     f'{row["get_M20_units"]:.2f} |')
    lines.append('')
    return lines


def _bounds(main, fit_table):
    lines = ['## What this does not measure', '',
             '- **The hard stratum.** 20 cells over 20 crystals, 6 reachable (C2-R-019). The '
             'consolidated hard pool from NERSC is what removes this bound.',
             '- **Full scale.** The networks are fitted on the slice\'s `fom-train` (157 crystals); '
             'S12\'s shipped tree on ~11 000. `tree` refitted on the slice is the fair pair; the '
             'full-scale network waits on the NERSC input export (`submit_fom_neural_inputs.sh`).',
             '- **Six of the nine condition bundles.** The report pool carries the severity axis only.',
             '- **Transfer to a different error law** (C2-R-008), and to unseen degradation.',
             '- **The fourteen-lattice prior**, until the retrain lands; every network here reads '
             'the shipped eleven-lattice prior through its support mask.', '']
    return lines


def figure(artifact_dir, tag, main, by_lattice, floors):
    """Per-lattice paired delta of the network against the refitted tree, with its floor."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = main.sort_values('operating_point')
    axis = axes[0]
    axis.barh(order.index, 100*order['operating_point'], color='#4c72b0', label='operating point')
    axis.barh(order.index, 100*order['top10'], color='none', edgecolor='#333333', label='top-10')
    axis.axvline(100*TIEBREAK_FLOOR, color='grey', linestyle=':', label='tie-break floor')
    axis.set_xlabel('% of patterns')
    axis.set_title('Every arm on the fully retained pool')
    axis.legend(loc='lower right', fontsize=8)

    axis = axes[1]
    if by_lattice is not None:
        rows = by_lattice[(by_lattice['reference'] == 'tree') & (by_lattice['arm'] == NETWORK)
                          & (by_lattice['metric'] == 'top10')].copy()
        rows['lattice'] = rows['scope'].str.replace('lattice=', '')
        rows = rows.set_index('lattice').reindex([l for l in LATTICES if l in
                                                  set(rows['lattice'])])
        y = np.arange(rows.shape[0])
        lower = np.clip(rows['delta_pp'] - rows['ci_low_pp'], 0, None)
        upper = np.clip(rows['ci_high_pp'] - rows['delta_pp'], 0, None)
        axis.errorbar(rows['delta_pp'], y, xerr=[lower, upper], fmt='o', color='#c44e52')
        for k, lattice in enumerate(rows.index):
            floor = floors.get(lattice)
            if floor:
                axis.plot([-floor, floor], [k, k], color='grey', alpha=0.4, linewidth=6)
        axis.set_yticks(y)
        axis.set_yticklabels(rows.index)
        axis.axvline(0, color='black', linewidth=0.8)
        axis.set_xlabel('network − tree, top-10, pp (paired, 95 % CI; grey = that lattice\'s floor)')
        axis.set_title('Per lattice against the refitted tree')
    fig.suptitle('S14 — the neural scoring network')
    fig.tight_layout()
    path = Path(artifact_dir)/f'{tag}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate S14's results document and figure")
    parser.add_argument('--artifact-dir', default=str(ARTIFACT_DIR))
    parser.add_argument('--tag', default='S14_neural')
    parser.add_argument('--document', default='S14_neural_score')
    args = parser.parse_args(argv)
    document = build(args.artifact_dir, args.tag)
    path = Path(args.artifact_dir)/f'{args.document}.md'
    path.write_text(document, encoding='utf-8')
    main_table = _load(args.artifact_dir, args.tag, 'main_table').set_index('arm')
    by_lattice = _load(args.artifact_dir, args.tag, 'by_lattice_mcnemar', required=False)
    image = figure(args.artifact_dir, args.document, main_table, by_lattice,
                   lattice_floors(args.artifact_dir))
    print(f'wrote {path}\nwrote {image}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
