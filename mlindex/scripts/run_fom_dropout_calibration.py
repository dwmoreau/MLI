"""S04 -- choose the --n-dropout values for condition bundles C4 and C5.

C4 is specified as "aggressive dropout: 20 peaks spanning higher q2", with the parameter chosen so
the median q2_20 is about 1.5x the nominal, and C5 as "as aggressive as the entry allows". Neither
value can be guessed: the achieved dropout is capped per entry by how many surplus peaks it has,
and the q2 stretch that follows depends on the lattice's peak density. This measures both.

No indexing happens here. Peak selection is the same `select_peaks_with_dropout` the generation
drivers call, over the same entries `run_fom_dump.py` will sample -- `sample_entries` depends only
on the seed and the per-lattice count -- with the same identifier-derived RNG, so the numbers
describe the bundles that will actually be generated rather than a proxy for them.

    python run_fom_dropout_calibration.py --artifact-dir ../../docs/fom/artifacts

Reports, per (Bravais lattice, n_drop): the median and upper-decile q2_20 stretch, the achieved
hole count against the requested one, and the fraction of entries where availability binds. The
handoff asks for the achieved distribution, not just the parameter, so all three are written out.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.command_line.run import BRAVAIS_LATTICES

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fom_mirror as mirror


# The handoff's target for C4: the median q2_20 should sit about 1.5x the nominal q2_20.
C4_TARGET_STRETCH = 1.5

# The values S04 actually runs (DWMM, 2026-08-16). The first sweep of this script is what settled
# them: the 1.5x target lands at n_drop = 10, but 10 is also the ceiling the mechanism now enforces
# (ErrorAdder.MAX_INTERIOR_DROPOUT), so pinning C4 there would have made C4 and C5 the same bundle.
# C4 steps back to 6 and C5 takes the ceiling, which puts the 1.5x target on C5 rather than C4.
C4_N_DROPOUT = 6
C5_N_DROPOUT = 10

# Only what the selection needs. The full READ_COLUMNS set pulls the ground-truth cell and the
# spacegroup as well, and dataset_aP.parquet is 1.2 GB.
CALIBRATION_COLUMNS = ['identifier', 'train', 'bravais_lattice', f'q2_{mirror.BROADENING_TAG}']


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate the S04 dropout conditions C4 and C5')
    parser.add_argument('--n-drop-values', type=str, default='0,2,4,6,8,10,12,14,16,20',
                        help='Comma-separated n_dropout values to sweep')
    parser.add_argument('--n-entries-per-bl', type=int, default=500,
                        help='Must match the generation run, or a different entry set is measured')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Must match the generation run, for the same reason')
    parser.add_argument('--n-peaks', type=int, default=mirror.N_PEAKS)
    parser.add_argument('--bravais-lattices', type=str, default=','.join(BRAVAIS_LATTICES))
    parser.add_argument('--artifact-dir', type=str, required=True)
    parser.add_argument('--tag', type=str, default='S04_dropout_calibration')
    return parser.parse_args()


def measure_lattice(bravais_lattice, n_drop_values, args):
    """One row per (entry, n_drop): what the selection actually did to that entry."""
    entries = mirror.sample_entries(bravais_lattice, args.n_entries_per_bl, args.seed,
                                    columns=CALIBRATION_COLUMNS)
    rows = []
    for _, entry in entries.iterrows():
        q2_full = np.asarray(entry[f'q2_{mirror.BROADENING_TAG}'], dtype=float)
        q2_full = q2_full[q2_full > 0]
        nominal = q2_full[:args.n_peaks]
        n_surplus = q2_full.size - args.n_peaks
        for n_drop in n_drop_values:
            # The generation drivers seed from the identifier, not from a counter, so the same
            # entry sees the same draw in every bundle. Reproduce that here.
            rng = np.random.default_rng(
                mirror.derived_seed(f'noise:{entry["identifier"]}', args.seed))
            selected = mirror.select_peaks_with_dropout(
                q2_full, args.n_peaks, n_drop, rng)
            rows.append({
                'bravais_lattice': bravais_lattice,
                'identifier': entry['identifier'],
                'n_drop': n_drop,
                'n_peaks_available': int(q2_full.size),
                'n_surplus': int(n_surplus),
                'n_holes_achieved': int(np.sum(~np.isin(nominal, selected))),
                # Capped means the entry could not deliver what was asked, whichever limit
                # bound -- its surplus, or the nominal window itself at large n_drop.
                'capped': bool(int(np.sum(~np.isin(nominal, selected))) < n_drop),
                'q2_20_stretch': float(selected[-1] / nominal[-1]),
                })
    return pd.DataFrame(rows)


def summarize(per_entry):
    grouped = per_entry.groupby(['bravais_lattice', 'n_drop'])
    summary = grouped.agg(
        n_entries=('identifier', 'size'),
        median_stretch=('q2_20_stretch', 'median'),
        p90_stretch=('q2_20_stretch', lambda column: column.quantile(0.9)),
        mean_holes=('n_holes_achieved', 'mean'),
        median_holes=('n_holes_achieved', 'median'),
        fraction_capped=('capped', 'mean'),
        ).reset_index()
    return summary


def pooled(per_entry):
    """Across all lattices, equally weighted per entry.

    Not reweighted to the CNRS distribution: this is a parameter choice, not a reported result, and
    PLAN 6.4's reweighting applies to claims about performance.
    """
    grouped = per_entry.groupby('n_drop')
    return grouped.agg(
        n_entries=('identifier', 'size'),
        median_stretch=('q2_20_stretch', 'median'),
        p90_stretch=('q2_20_stretch', lambda column: column.quantile(0.9)),
        mean_holes=('n_holes_achieved', 'mean'),
        fraction_capped=('capped', 'mean'),
        ).reset_index()


def measured_at(pooled_summary, n_drop):
    """The measured properties of a decided bundle value, or None if it is off the sweep."""
    row = pooled_summary.loc[pooled_summary['n_drop'] == n_drop]
    return None if row.empty else row.iloc[0]


# Light-mode categorical slots 1 and 8 from the data-viz reference palette, plus its text inks.
# Fourteen lattices are deliberately NOT fourteen hues: past eight slots a categorical palette
# cannot hold colourblind separation, and here it would be pointless anyway -- the per-lattice
# curves lie almost on top of each other. The spread is drawn as a band and only the extremes are
# named, which is what the reader actually needs. Every lattice is still in the CSV and the
# markdown table.
SERIES_BLUE = '#2a78d6'
TARGET_RED = '#e34948'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
BAND_GREY = '#c9c9c4'


def _label_extremes(axis, summary, column, formatter):
    """Name the highest and lowest lattice at the right-hand end of the sweep."""
    final_x = summary['n_drop'].max()
    final = summary.loc[summary['n_drop'] == final_x].set_index('bravais_lattice')[column]
    for bravais_lattice, offset in ((final.idxmax(), 4), (final.idxmin(), -4)):
        axis.annotate(f'{bravais_lattice} {formatter(final[bravais_lattice])}',
                      xy=(final_x, final[bravais_lattice]), xytext=(6, offset),
                      textcoords='offset points', va='center', fontsize=8,
                      color=TEXT_SECONDARY)


def write_figure(summary, pooled_summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spread = summary.groupby('n_drop').agg(
        low_stretch=('median_stretch', 'min'), high_stretch=('median_stretch', 'max'),
        low_holes=('mean_holes', 'min'), high_holes=('mean_holes', 'max'),
        ).reset_index()

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    n_drop = pooled_summary['n_drop']

    axes[0].fill_between(spread['n_drop'], spread['low_stretch'], spread['high_stretch'],
                         color=BAND_GREY, alpha=0.55, linewidth=0)
    axes[0].plot(n_drop, pooled_summary['median_stretch'],
                 color=SERIES_BLUE, linewidth=2, marker='o', markersize=4)
    axes[0].axhline(C4_TARGET_STRETCH, color=TARGET_RED, linestyle='--', linewidth=1.5)
    axes[0].annotate(f'C4 target {C4_TARGET_STRETCH:g}x', xy=(0.02, C4_TARGET_STRETCH),
                     xycoords=('axes fraction', 'data'), va='bottom',
                     color=TARGET_RED, fontsize=9)
    axes[0].set_ylabel(r'median $q^2_{20}$ / nominal $q^2_{20}$', color=TEXT_PRIMARY)
    axes[0].set_title('How far the window stretches', color=TEXT_PRIMARY)
    _label_extremes(axes[0], summary, 'median_stretch', lambda value: f'{value:.2f}x')

    axes[1].plot(n_drop, n_drop, color=BAND_GREY, linestyle=':', linewidth=1.5)
    axes[1].annotate('requested', xy=(n_drop.iloc[-1], n_drop.iloc[-1]), xytext=(-4, 6),
                     textcoords='offset points', ha='right', fontsize=8, color=TEXT_SECONDARY)
    axes[1].fill_between(spread['n_drop'], spread['low_holes'], spread['high_holes'],
                         color=BAND_GREY, alpha=0.55, linewidth=0)
    axes[1].plot(n_drop, pooled_summary['mean_holes'],
                 color=SERIES_BLUE, linewidth=2, marker='o', markersize=4)
    axes[1].set_ylabel('mean holes achieved', color=TEXT_PRIMARY)
    axes[1].set_title('Where availability binds', color=TEXT_PRIMARY)
    _label_extremes(axes[1], summary, 'mean_holes', lambda value: f'{value:.0f}')

    for axis in axes:
        axis.set_xlabel('requested n_dropout', color=TEXT_PRIMARY)
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color(BAND_GREY)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.margins(x=0.10)
    # supxlabel rather than suptitle(y=...): constrained_layout reserves space for it, where a
    # negative-y suptitle is simply clipped off the canvas.
    figure.supxlabel(
        'Line is the pooled median over all 5 955 entries; band spans the 14 Bravais lattices, '
        'with the extremes named. Per-lattice values are in the CSV.',
        fontsize=8.5, color=TEXT_SECONDARY)
    figure.savefig(path, dpi=200, facecolor='white')
    plt.close(figure)


def main():
    args = _parse_args()
    n_drop_values = [int(value) for value in args.n_drop_values.split(',')]
    bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
    invalid = [bl for bl in bravais_lattices if bl not in BRAVAIS_LATTICES]
    if invalid:
        raise SystemExit(f"Unknown Bravais lattices: {', '.join(invalid)}")

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    per_lattice = []
    for bravais_lattice in bravais_lattices:
        measured = measure_lattice(bravais_lattice, n_drop_values, args)
        per_lattice.append(measured)
        print(f'{bravais_lattice}: {measured["identifier"].nunique()} entries, '
              f'median available {measured["n_peaks_available"].median():.0f} peaks', flush=True)
    per_entry = pd.concat(per_lattice, ignore_index=True)

    summary = summarize(per_entry)
    pooled_summary = pooled(per_entry)
    c4_row = measured_at(pooled_summary, C4_N_DROPOUT)
    c5_row = measured_at(pooled_summary, C5_N_DROPOUT)

    summary.to_csv(artifact_dir / f'{args.tag}.csv', index=False)
    pooled_summary.to_csv(artifact_dir / f'{args.tag}_pooled.csv', index=False)
    write_figure(summary, pooled_summary, artifact_dir / f'{args.tag}.png')

    availability = per_entry.loc[per_entry['n_drop'] == n_drop_values[0]].groupby(
        'bravais_lattice').agg(
        median_available=('n_peaks_available', 'median'),
        fraction_with_no_surplus=('n_surplus', lambda column: float((column <= 0).mean())),
        ).reset_index()

    lines = [
        '# S04 dropout calibration -- choosing n_dropout for C4 and C5',
        '',
        'Peak selection only; no indexing. Same entries, seed and per-entry RNG as the generation',
        'run, so these are the bundles that will be generated, not a proxy.',
        '',
        f'- **C4 (sparse): `--n-dropout {C4_N_DROPOUT}`** -- median q2_20 '
        f'{c4_row["median_stretch"]:.2f}x nominal, {c4_row["mean_holes"]:.1f} holes of 20.',
        f'- **C5 (aggressive): `--n-dropout {C5_N_DROPOUT}`** -- median q2_20 '
        f'{c5_row["median_stretch"]:.2f}x nominal, {c5_row["mean_holes"]:.1f} holes of 20. This is '
        f'`ErrorAdder.MAX_INTERIOR_DROPOUT`, the ceiling the mechanism enforces.',
        '',
        f'The handoff put a {C4_TARGET_STRETCH}x median-stretch target on C4. That target lands at '
        f'n_dropout 10, which is also the ceiling, so pinning C4 there would have made C4 and C5 '
        'the same bundle. C4 steps back to 6 and the target moves to C5 (DWMM, 2026-08-16).',
        '',
        'Why the ceiling exists: the median entry carries ~60 peaks, so the entry\'s own surplus '
        'almost never binds. Swept to 20, the mechanism removes 18.9 of the nominal 20 and refills '
        'from peaks 21-40 -- a wholesale translation of the window to high angle, not holes punched '
        'in it. Ten keeps half the nominal window.',
        '',
        '## Pooled over all lattices',
        '',
        pooled_summary.round(3).to_markdown(index=False),
        '',
        '## Peak availability, which is what caps the achieved dropout',
        '',
        availability.round(3).to_markdown(index=False),
        '',
        '## Per Bravais lattice',
        '',
        summary.round(3).to_markdown(index=False),
        ]
    with open(artifact_dir / f'{args.tag}.md', 'w', encoding='utf-8') as report:
        report.write('\n'.join(lines) + '\n')

    print()
    print(pooled_summary.round(3).to_string(index=False))
    print()
    print(f'C4: --n-dropout {C4_N_DROPOUT}  '
          f'({c4_row["median_stretch"]:.2f}x, {c4_row["mean_holes"]:.1f} holes)')
    print(f'C5: --n-dropout {C5_N_DROPOUT}  '
          f'({c5_row["median_stretch"]:.2f}x, {c5_row["mean_holes"]:.1f} holes)')
    print(f'wrote {artifact_dir / (args.tag + ".md")}')


if __name__ == '__main__':
    main()
