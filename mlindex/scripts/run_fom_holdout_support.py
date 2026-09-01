"""S10a: the validity gates, before any hold-out merit is allowed to carry a claim.

    python mlindex/scripts/run_fom_holdout_support.py \\
        --pool mlindex/data/fom_benchmark_c2 --sidecar /tmp/s10a_sample

Four gates and two realism confirmations. Each can invalidate every number S10b and S10c would
otherwise produce, and none of them is in the inherited S10 handoff.

**Gate 1 -- does the reference list reach the surplus peaks?** A hold-out merit scores against the
candidate's extinction-group reference list, which is finite. Where that list stops short of the
peak being asked about, `ho_*` measures the list running out rather than the cell mispredicting,
and the merit is a statement about `hkl_ref_length` rather than about crystallography.

**Gate 2 -- does `M_rev` have support on a surplus window?** `M_rev` counts calculated lines the
observations fail to account for, over `[q_I, q_N]`. Handed the surplus, that window is the surplus
interval, which is short. `get_M_rev_sym` declares the merit undefined below ten lines (C2-F-059,
C2-F-062), so if the floor fires broadly then `ho_M_sym` -- the out-of-sample form of the merit
S09 crowned -- is undefined exactly where it was supposed to be measured. DWMM, 2026-09-01:
measure the support first, then choose a rule.

**Gate 3 -- is the statistic comparable across peak budgets?** A merit that sums over peaks grows
with the number of peaks, so a sweep would read that growth as a gain. This reports the scale of
each column against the budget so an extensive one cannot be swept by accident.

**Gate 4 -- cubic is fitted on ten peaks, so peaks 11-20 are already hold-out for it** (R5). The
stored `q2_holdout` does not contain them. This is a live lead rather than bookkeeping: S09 found
`M_sym` *loses* on cF (-8.93 pp, 1 gained / 11 lost, C2-F-096) with "cubic is scored on ten peaks"
the leading untested hypothesis, and cubic is the one lattice system where a hold-out merit has ten
extra peaks available for free.

**Realism A -- do the surplus peaks carry measurement error?** `SCHEMA.md` and `FomPatterns` both
say the surplus is drawn from the window's own noise stream, and campaign 1's equivalent was
re-synthesised noiselessly from the true structure (R13). Asserted in two places, checked in none.
Residuals are measured against the *true* cell for the window and for the surplus separately: if
the surplus were noiseless its residuals would sit at machine epsilon.

**Realism B -- how many contaminants actually reach the surplus?** `add_contaminants` draws below
the window maximum, so the only junk lines above it are ones displaced out of the window when the
list was re-truncated. This measures the rate before and after S10a's offline seeding.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.utilities.FigureOfMerits import get_hkl_matrix


BASE = Path(__file__).resolve().parents[2]
ARTIFACTS = BASE/'docs'/'fom_campaign2'/'artifacts'

# The support below which `get_M_rev_sym` declares M_rev undefined and returns 0.0.
MIN_N_CAL = FomBenchmark.M_REV_MIN_N_CAL

ENTRY_COLUMNS = ('entry_id', 'condition_bundle', 'split', 'bravais_lattice_true',
                 'lattice_system_true', 'q2_obs', 'hkl_true', 'q2_holdout', 'hkl_holdout',
                 'xnn_true', 'n_contaminants', 'second_phase_lines', 'n_peaks_available')


def _sidecar_frame(sidecar):
    frames = [pd.read_parquet(path) for path in sorted(Path(sidecar).glob('candidates*.parquet'))]
    return pd.concat(frames, ignore_index=True)


def _true_residuals(entries):
    """|q2_observed - q2_from_the_true_cell|, for the window and for the surplus separately.

    The true cell is stored as a full six-component `xnn_true` whatever the lattice, so the
    triclinic design matrix is the right one for every row -- the symmetry-implied zeros are
    already in the stored vector. Contaminant lines carry `(0, 0, 0)` and have no true position at
    all, so they are excluded here rather than scored against the origin.
    """
    rows = []
    for record in entries.itertuples():
        xnn = np.asarray(record.xnn_true, dtype=np.float64)
        if xnn.size != 6:
            continue
        for label, q2, hkl in (
                ('window', np.asarray(record.q2_obs, dtype=np.float64),
                 np.asarray(record.hkl_true).reshape(-1, 3)),
                ('surplus', np.asarray(record.q2_holdout, dtype=np.float64),
                 np.asarray(record.hkl_holdout).reshape(-1, 3))):
            if q2.size == 0 or hkl.shape[0] != q2.size:
                continue
            real = np.abs(hkl).sum(axis=1) > 0
            if not real.any():
                continue
            q2_calc = get_hkl_matrix(hkl[real], 'triclinic').astype(np.float64) @ xnn
            residual = np.abs(q2[real] - q2_calc)
            rows.append({'entry_id': record.entry_id,
                         'condition_bundle': record.condition_bundle,
                         'bravais_lattice': record.bravais_lattice_true,
                         'window': label,
                         'n_lines': int(real.sum()),
                         'median_abs_residual': float(np.median(residual)),
                         'mean_abs_residual': float(np.mean(residual))})
    return pd.DataFrame(rows)


def gate_reference_reach(merits, keys, n_extra_values):
    rows = []
    for n_extra in n_extra_values:
        column = FomBenchmark.holdout_column('ho_ref_reach', n_extra)
        if column not in merits.columns:
            continue
        frame = pd.concat([keys, merits[column]], axis=1).dropna(subset=[column])
        for lattice, group in frame.groupby('bravais_lattice', sort=True):
            rows.append({'gate': 'reference_reach', 'n_extra': n_extra,
                         'bravais_lattice': lattice, 'n_scored': int(len(group)),
                         'fraction_reaching': float(group[column].mean()),
                         'fraction_short': float(1.0 - group[column].mean())})
    return pd.DataFrame(rows)


def gate_mrev_support(merits, keys, n_extra_values):
    rows = []
    for n_extra in n_extra_values:
        column = FomBenchmark.holdout_column('ho_N_cal', n_extra)
        if column not in merits.columns:
            continue
        frame = pd.concat([keys, merits[column]], axis=1).dropna(subset=[column])
        for lattice, group in frame.groupby('bravais_lattice', sort=True):
            values = group[column].to_numpy()
            rows.append({'gate': 'mrev_support', 'n_extra': n_extra,
                         'bravais_lattice': lattice, 'n_scored': int(values.size),
                         'median_N_cal': float(np.median(values)),
                         'p10_N_cal': float(np.percentile(values, 10)),
                         'fraction_below_floor': float(np.mean(values < MIN_N_CAL))})
    return pd.DataFrame(rows)


def gate_budget_scale(merits, n_extra_values):
    """How each column's central value moves with the budget. An extensive column grows with it."""
    rows = []
    for name in FomBenchmark.HOLDOUT_MERIT_NAMES:
        values = {}
        for n_extra in n_extra_values:
            column = FomBenchmark.holdout_column(name, n_extra)
            if column in merits.columns:
                series = merits[column].dropna()
                if len(series):
                    values[n_extra] = float(series.median())
        if len(values) < 2:
            continue
        low, high = min(values), max(values)
        ratio = values[high]/values[low] if values[low] else np.nan
        rows.append({'gate': 'budget_scale', 'merit': name,
                     f'median_at_n{low}': values[low], f'median_at_n{high}': values[high],
                     'ratio_high_over_low': ratio,
                     'extensive': bool(np.isfinite(ratio) and ratio > 2.0)})
    return pd.DataFrame(rows)


def gate_cubic_budget(entries):
    """Peaks 11-20 are hold-out for a cubic candidate and are not in `q2_holdout` (R5)."""
    rows = []
    for lattice, group in entries.groupby('bravais_lattice_true', sort=True):
        system = group['lattice_system_true'].iloc[0]
        stored = group['q2_holdout'].apply(len)
        extra_in_window = 10 if system == 'cubic' else 0
        rows.append({'gate': 'cubic_budget', 'bravais_lattice': lattice,
                     'lattice_system': system, 'n_entries': int(len(group)),
                     'peaks_fitted': 10 if system == 'cubic' else 20,
                     'unused_window_peaks': extra_in_window,
                     'median_stored_surplus': float(stored.median()),
                     'median_surplus_available': float(stored.median() + extra_in_window)})
    return pd.DataFrame(rows)


def contaminant_rates(entries, seed=FomBenchmark.HOLDOUT_CONTAMINANT_SEED):
    rows = []
    for (bundle, lattice), group in entries.groupby(
            ['condition_bundle', 'bravais_lattice_true'], sort=True):
        before = after = added = surplus = 0
        for record in group.itertuples():
            hkl = np.asarray(record.hkl_holdout).reshape(-1, 3)
            before += int((np.abs(hkl).sum(axis=1) == 0).sum())
            surplus += hkl.shape[0]
            row = {'q2_holdout': record.q2_holdout, 'hkl_holdout': record.hkl_holdout,
                   'q2_obs': record.q2_obs, 'n_contaminants': record.n_contaminants,
                   'second_phase_lines': record.second_phase_lines}
            _, hkl_new, n_added = FomBenchmark.contaminated_holdout(
                pd.Series(row), record.entry_id, seed=seed)
            after += int((np.abs(hkl_new).sum(axis=1) == 0).sum())
            added += n_added
        rows.append({'condition_bundle': bundle, 'bravais_lattice': lattice,
                     'n_entries': int(len(group)), 'surplus_lines': surplus,
                     'contaminants_before': before, 'contaminants_after': after,
                     'contaminants_seeded': added,
                     'rate_before': before/max(surplus, 1),
                     'rate_after': after/max(surplus + added, 1)})
    return pd.DataFrame(rows)


# The categorical slots, validated for all-pairs separation in both colour-vision and normal
# vision (dataviz `scripts/validate_palette.js`, three slots, --pairs all). Three is the cap for
# an all-pairs form; the grouping below is built to fit it rather than the palette stretched to
# fit fourteen lattices. The aqua slot sits below 3:1 on this surface, so every series carries a
# visible direct label -- the relief rule, and what a reader wants here anyway.
SERIES = {'hard (aP, mP, mC)': '#2a78d6', 'cubic (cF, cI, cP)': '#eb6834',
          'the other eight': '#1baf7a'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#d9d8d4', '#fcfcfb'
SLOT1, SLOT2 = '#2a78d6', '#eb6834'

HARD = ('aP', 'mP', 'mC')
CUBIC = ('cF', 'cI', 'cP')


def _series_of(lattice):
    if lattice in HARD:
        return 'hard (aP, mP, mC)'
    if lattice in CUBIC:
        return 'cubic (cF, cI, cP)'
    return 'the other eight'


def _grouped(frame, value):
    frame = frame.copy()
    frame['series'] = frame['bravais_lattice'].map(_series_of)
    return frame.groupby(['series', 'n_extra'])[value].mean().unstack(0)


def _style(axis, x_grid=False):
    axis.grid(axis='x' if x_grid else 'y', color=GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ('top', 'right'):
        axis.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        axis.spines[spine].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=7.5, length=3)


def _line_panel(axis, table, title, ylabel, marker_at=5):
    """One gate against the peak budget. Three series, so a shared legend rather than end labels.

    Direct labels were tried first, as the relief rule prefers, and all three series converge at
    the right-hand end of both gates -- so the labels collided with each other and with the next
    panel. The relief the aqua slot needs is discharged by the table view instead: every value
    plotted here is a row of the CSV beside it.
    """
    for name, colour in SERIES.items():
        if name not in table.columns:
            continue
        column = table[name].dropna()
        axis.plot(column.index, 100*column.to_numpy(), color=colour, linewidth=2.0,
                  marker='o', markersize=5.5, label=name, zorder=3)
    axis.axvline(marker_at, color=MUTED, linewidth=1.0, linestyle=':', zorder=1)
    axis.set_title(title, fontsize=10.5, color=INK, loc='left', fontweight='semibold', pad=8)
    axis.set_ylabel(ylabel, fontsize=8.5, color=MUTED)
    axis.set_xlabel('surplus peaks scored  (total peaks in the pattern)', fontsize=8.5,
                    color=MUTED)
    axis.set_xscale('log')
    axis.set_xticks([1, 2, 3, 5, 10, 20])
    axis.set_xticklabels(['1\n(21)', '2\n(22)', '3\n(23)', '5\n(25)', '10\n(30)', '20\n(40)'],
                         fontsize=7.5)
    axis.set_ylim(-3, 104)
    _style(axis)


def _paired_bars(axis, index, left, right, labels, title, xlabel, log=False):
    positions = np.arange(len(index))
    axis.barh(positions - 0.19, left, height=0.34, color=SLOT1, label=labels[0], zorder=3)
    axis.barh(positions + 0.19, right, height=0.34, color=SLOT2, label=labels[1], zorder=3)
    axis.set_yticks(positions)
    axis.set_yticklabels(index, fontsize=7.5)
    if log:
        axis.set_xscale('log')
    axis.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    axis.set_title(title, fontsize=10.5, color=INK, loc='left', fontweight='semibold', pad=8)
    axis.legend(fontsize=8, frameon=False, loc='lower right', labelcolor=MUTED)
    _style(axis, x_grid=True)


def write_figure(tables, path, subtitle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), facecolor=SURFACE,
                                layout='constrained')
    # constrained_layout reserves the header band through `rect`, not `top`.
    figure.get_layout_engine().set(hspace=0.10, wspace=0.09,
                                   rect=(0.012, 0.040, 0.988, 0.845))

    _line_panel(axes[0][0], _grouped(tables['mrev_support'], 'fraction_below_floor'),
                'A   M_rev loses its support', 'candidates where the floor fires  (%)')
    _line_panel(axes[0][1], _grouped(tables['reference_reach'], 'fraction_short'),
                'B   The reference list runs out', 'candidates whose list stops short  (%)')

    residuals = tables['residuals'].pivot_table(
        index='condition_bundle', columns='window', values='median_abs_residual', aggfunc='median')
    order = list(residuals['window'].sort_values().index)
    labels = [name.replace('c2_', '') for name in order]
    _paired_bars(axes[1][0], order, residuals.loc[order, 'window'],
                 residuals.loc[order, 'surplus'],
                 ('fitted window', 'surplus peaks'),
                 'C   The surplus carries the window\'s error',
                 'median |q2 observed - q2 from the true cell|   (1/Angstrom^2)', log=True)
    axes[1][0].set_yticklabels(labels, fontsize=7.5)

    contaminants = tables['contaminants'].groupby('condition_bundle').sum(numeric_only=True)
    contaminants = contaminants.loc[[name for name in order if name in contaminants.index]]
    before = 100*contaminants['contaminants_before']/contaminants['surplus_lines']
    after = 100*contaminants['contaminants_after']/(
        contaminants['surplus_lines'] + contaminants['contaminants_seeded'])
    _paired_bars(axes[1][1], list(contaminants.index), before, after,
                 ('as generated', 'after seeding'),
                 'D   Contaminants cannot reach the surplus',
                 'contaminant lines as a share of the surplus  (%)')
    axes[1][1].set_yticklabels([name.replace('c2_', '') for name in contaminants.index],
                               fontsize=7.5)

    handles, names = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, names, fontsize=9, frameon=False, ncol=3, labelcolor=MUTED,
                  loc='upper left', bbox_to_anchor=(0.055, 0.918))
    figure.suptitle('S10a   Can the surplus peaks carry a figure of merit at all?',
                    fontsize=14, color=INK, x=0.055, ha='left', y=0.975, fontweight='semibold')
    figure.text(0.055, 0.937, subtitle, fontsize=9, color=MUTED, ha='left', va='top')
    figure.text(0.055, 0.026,
                'Dotted line marks five surplus peaks - a 25-peak pattern, which is what real '
                'data typically supplies.',
                fontsize=7.5, color=MUTED, ha='left')
    figure.text(0.055, 0.008,
                'A and B are the two gates a hold-out merit must clear. Both fail outright on '
                'cubic; A costs a fifth to a third of all other candidates at five surplus peaks.',
                fontsize=7.5, color=MUTED, ha='left')
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='S10a validity gates for the hold-out merits')
    parser.add_argument('--pool', type=str, required=True)
    parser.add_argument('--sidecar', type=str, required=True)
    parser.add_argument('--tag', type=str, default='S10a')
    parser.add_argument('--artifact-dir', type=str, default=str(ARTIFACTS))
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    artifacts = Path(args.artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    entries = FomBenchmark.load_entries(args.pool)
    keep = [name for name in ENTRY_COLUMNS if name in entries.columns]
    entries = entries[keep]

    merits = _sidecar_frame(args.sidecar)
    n_extra_values = sorted({int(column.split('__n')[1]) for column in merits.columns
                             if '__n' in column})
    keys = merits[['entry_id', 'condition_bundle', 'bravais_lattice']]

    tables = {
        'reference_reach': gate_reference_reach(merits, keys, n_extra_values),
        'mrev_support': gate_mrev_support(merits, keys, n_extra_values),
        'budget_scale': gate_budget_scale(merits, n_extra_values),
        'cubic_budget': gate_cubic_budget(entries),
        'contaminants': contaminant_rates(entries),
        'residuals': _true_residuals(entries),
        }
    for name, frame in tables.items():
        path = artifacts/f'{args.tag}_holdout_{name}.csv'
        frame.to_csv(path, index=False)
        print(f'{path}: {len(frame)} rows')

    meta = {'pool': args.pool, 'sidecar': args.sidecar,
            'n_candidates_scored': int(len(merits)), 'n_entries': int(len(entries)),
            'n_extra_values': n_extra_values, 'min_n_cal': MIN_N_CAL,
            'contaminant_seed': FomBenchmark.HOLDOUT_CONTAMINANT_SEED}
    (artifacts/f'{args.tag}_holdout_support_meta.json').write_text(
        json.dumps(meta, indent=2), encoding='utf-8')

    figure_path = artifacts/f'{args.tag}_holdout_support.png'
    write_figure(tables, figure_path,
                 f'{len(merits):,} candidates sampled from {args.pool}, '
                 f'{len(entries):,} pattern-conditions')
    print(f'{figure_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
