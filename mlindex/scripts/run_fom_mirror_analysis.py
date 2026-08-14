"""S02 -- characterise the synthetic condition grid and freeze the benchmark manifest.

Consumes the per-entry shards written by run_fom_mirror.py and reports, per condition bundle:

1. Operating point and ceiling -- how often the correct cell is reported (pooled top 10 and
   M20 > 10) versus found anywhere -- and the headroom between them, which is what a perfect
   re-ranker of the same pool could recover. Downstream steps should quote gains as a fraction of
   that headroom rather than in percentage points.

2. The same per Bravais lattice and per volume decile, because an aggregate is dominated by whatever
   is abundant and can conceal large offsetting per-lattice errors.

3. Where the lost entries go -- rank failure, threshold failure, both, or not found. Note F-043:
   this split inverts between bundles, so it is a function of the conditions and not a single
   number.

**Conditions are justified physically, not by matching the sealed benchmark** (DWMM, 2026-08-12).
The CNRS operating point and ceiling are printed beside every bundle as a sanity check and are
*never* tuned to or selected on; doing so would fit the data generator to the answers of the
benchmark S16 is supposed to test blind. Table 1's per-lattice *totals* are used as reweighting
weights, which is the benchmark's population composition and is mandated by PLAN 6.4; its ML-Index
column is not a target. The nominal bundle is PLAN 6.3's C1 -- error multiplier 1, the characterised
sigma(q2) -- designated before any of this comparison existed.

Aggregates are re-weighted to the CNRS lattice distribution throughout, because the validation pool
is 78% aP/mP/oP where the benchmark is 55%.

    python mlindex/scripts/run_fom_mirror_analysis.py
    python mlindex/scripts/run_fom_mirror_analysis.py --select-bundle error1_cont0

Run it with the development env:
    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import mlindex  # noqa: E402

# Draft Table 1 (draft_v1.2.pdf p.17), read from the rendered page rather than the text layer.
# 'total' is the entry count per Bravais lattice and doubles as the CNRS re-weighting weight;
# 'ml_index' is the count ML-Index reports under the operating-point criterion.
CNRS_TABLE1 = {
    'cP': {'ml_index': 17, 'total': 21},
    'cI': {'ml_index': 2, 'total': 4},
    'cF': {'ml_index': 17, 'total': 22},
    'tP': {'ml_index': 46, 'total': 56},
    'tI': {'ml_index': 19, 'total': 30},
    'hP': {'ml_index': 32, 'total': 36},
    'hR': {'ml_index': 11, 'total': 19},
    'oP': {'ml_index': 101, 'total': 124},
    'oC': {'ml_index': 10, 'total': 13},
    'oF': {'ml_index': 2, 'total': 2},
    'oI': {'ml_index': 9, 'total': 16},
    'mP': {'ml_index': 108, 'total': 149},
    'mC': {'ml_index': 37, 'total': 51},
    'aP': {'ml_index': 39, 'total': 56},
    }

CNRS_OPERATING_POINT = 450 / 599
CNRS_CEILING = 533 / 599

# REPORT-ONLY (DWMM, 2026-08-12). These two numbers are printed beside every bundle as a sanity
# check -- a mirror wildly unlike the benchmark is worth noticing -- but nothing is tuned to them
# and no bundle is selected on them. Choosing conditions to reproduce CNRS's per-lattice results
# would fit the data generator to the sealed benchmark's answers, which would make any FOM developed
# here implicitly tuned to CNRS and destroy S16's independence as a blind test.
#
# Conditions are justified physically instead, and the bundles are PLAN section 6.3's, which were
# designated before any of this comparison existed. Error multiplier 1 is the characterised
# sigma(q2) for this instrument population, so C1 is a measured nominal rather than a fitted one.
CONDITION_LABELS = {
    'error0_cont0': 'C0 ideal',
    'error1_cont0': 'C1 nominal',
    'error2_cont0': 'C2 noisy',
    'error1_cont2': 'C3 contaminated',
}
NOMINAL_BUNDLE = 'error1_cont0'

# Table 1's per-lattice totals are the benchmark's population composition and are used as
# reweighting weights per PLAN section 6.4. Its ML-Index column is *not* a target. cI (4 entries)
# and oF (2) carry no information and are excluded from the per-lattice comparison.
COMPARISON_LATTICES = ['tP', 'oP', 'mP', 'mC', 'aP']

TOP_N = 10
M20_THRESHOLD = 10.0
N_BOOTSTRAP = 2000
N_VOLUME_DECILES = 10
SPLIT_FRACTIONS = {'fom-train': 0.6, 'fom-dev': 0.2, 'fom-test': 0.2}

ARTIFACT_DIRECTORY = Path(BASE) / 'docs' / 'fom' / 'artifacts'
MIRROR_DIRECTORY = Path(mlindex.__path__[0]) / 'characterization' / 'fom' / 'mirror'


def _parse_args():
    parser = argparse.ArgumentParser(description='S02 mirror scoring and manifest')
    parser.add_argument('--mirror-dir', type=str, default=str(MIRROR_DIRECTORY))
    parser.add_argument('--out-dir', type=str, default=str(ARTIFACT_DIRECTORY))
    parser.add_argument('--select-bundle', type=str, default=None,
                        help='Bundle tag to freeze the manifest against, e.g. error1_cont1. '
                             'Omit to report every bundle without freezing a manifest')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Seed for the split assignment and the bootstrap')
    parser.add_argument('--rtol-suffix', type=str, default='',
                        help="'' for the headline rtol 1e-2, '_rtol0.02' for the draft's 2%%")
    return parser.parse_args()


def load_bundles(mirror_dir):
    frames = {}
    for path in sorted(Path(mirror_dir).glob('entries_*.parquet')):
        # entries_<bundle>_shard<i>of<n>.parquet
        bundle = path.stem[len('entries_'):].rsplit('_shard', 1)[0]
        frames.setdefault(bundle, []).append(pd.read_parquet(path))
    return {bundle: pd.concat(parts, ignore_index=True) for bundle, parts in frames.items()}


def classify(entries, suffix=''):
    # The four-way decomposition of every entry. rank is 0-based and -1 when the correct cell is
    # absent from the pool entirely, which is a generation failure and not a ranking one.
    rank = entries[f'rank_best_correct{suffix}'].to_numpy()
    score = entries[f'M20_best_correct{suffix}'].to_numpy()
    found = rank >= 0
    in_top_n = found & (rank < TOP_N)
    over_threshold = found & (score > M20_THRESHOLD)
    return pd.DataFrame({
        'found': found,
        'operating_point': in_top_n & over_threshold,
        'rank_only': in_top_n,
        'threshold_only': over_threshold,
        'lost_rank_failure': found & ~in_top_n & over_threshold,
        'lost_threshold_failure': found & in_top_n & ~over_threshold,
        'lost_both': found & ~in_top_n & ~over_threshold,
        'lost_not_found': ~found,
        }, index=entries.index)


def bootstrap_ci(flags, rng, n_bootstrap=N_BOOTSTRAP):
    # Bootstrap over entries, never over candidates (PROTOCOL section 8).
    flags = np.asarray(flags, dtype=float)
    if flags.size == 0:
        return np.nan, np.nan
    draws = rng.integers(0, flags.size, size=(n_bootstrap, flags.size))
    means = flags[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_lattice_table(entries, labels, rng):
    rows = []
    for bravais_lattice, group in entries.groupby('bravais_lattice_true', sort=False):
        group_labels = labels.loc[group.index]
        target = CNRS_TABLE1.get(bravais_lattice)
        row = {
            'bravais_lattice': bravais_lattice,
            'n_entries': int(group.shape[0]),
            'cnrs_total': target['total'] if target else 0,
            'cnrs_operating_point': target['ml_index'] / target['total'] if target else np.nan,
            }
        for column in ['operating_point', 'found', 'rank_only', 'threshold_only',
                       'lost_rank_failure', 'lost_threshold_failure', 'lost_both',
                       'lost_not_found']:
            row[column] = float(group_labels[column].mean())
        low, high = bootstrap_ci(group_labels['operating_point'], rng)
        row['operating_point_ci_low'] = low
        row['operating_point_ci_high'] = high
        low, high = bootstrap_ci(group_labels['found'], rng)
        row['ceiling_ci_low'] = low
        row['ceiling_ci_high'] = high
        row['off_by_two'] = float(entries.loc[group.index, 'off_by_two_any'].mean()) \
            if 'off_by_two_any' in entries.columns else np.nan
        row['delta_vs_cnrs'] = row['operating_point'] - row['cnrs_operating_point']
        rows.append(row)
    table = pd.DataFrame(rows)
    order = [bl for bl in CNRS_TABLE1 if bl in set(table['bravais_lattice'])]
    return table.set_index('bravais_lattice').loc[order].reset_index()


def cnrs_weighted(table, column):
    # Re-weight to the CNRS lattice distribution. Lattices absent from the run are dropped and the
    # weights renormalised over what is present, so a partial run is not silently biased.
    present = table.dropna(subset=[column])
    weights = present['cnrs_total'].to_numpy(dtype=float)
    if weights.sum() == 0:
        return np.nan
    return float(np.sum(weights * present[column].to_numpy()) / weights.sum())


def volume_decile(entries):
    # Deciles within each lattice, so 'high volume' means high for that lattice rather than
    # tracking the lattice's overall size.
    decile = np.zeros(entries.shape[0], dtype=np.int64)
    positions = {index: position for position, index in enumerate(entries.index)}
    for _, group in entries.groupby('bravais_lattice_true', sort=False):
        ranks = group['volume_true'].rank(method='first', pct=True)
        assigned = np.clip((ranks.to_numpy() * N_VOLUME_DECILES).astype(np.int64),
                           0, N_VOLUME_DECILES - 1)
        decile[[positions[index] for index in group.index]] = assigned
    return pd.Series(decile, index=entries.index)


def per_decile_table(entries, labels):
    frame = labels.copy()
    frame['volume_decile'] = volume_decile(entries)
    frame['bravais_lattice'] = entries['bravais_lattice_true']
    columns = ['operating_point', 'found', 'lost_rank_failure', 'lost_threshold_failure',
               'lost_both', 'lost_not_found']
    table = frame.groupby('volume_decile')[columns].mean().reset_index()
    table['n_entries'] = frame.groupby('volume_decile').size().to_numpy()
    return table


def bundle_summary(bundles, labels_by_bundle, rng):
    rows = []
    for bundle in sorted(bundles):
        table = per_lattice_table(bundles[bundle], labels_by_bundle[bundle], rng)
        operating_point = cnrs_weighted(table, 'operating_point')
        ceiling = cnrs_weighted(table, 'found')
        comparison = table.loc[table['bravais_lattice'].isin(COMPARISON_LATTICES)]
        worst = comparison['delta_vs_cnrs'].abs().max() if comparison.shape[0] else np.nan
        # Headroom is what a perfect re-ranker of this pool could recover: everything the pipeline
        # found but did not report. It is the denominator downstream steps should quote gains
        # against -- "recovered 40% of the reachable gap" rather than "gained 8 points" -- because a
        # bundle with more headroom flatters any improvement measured in absolute points. Expressed
        # against the bundle's own headroom, deliberately: normalising by the sealed benchmark's
        # headroom would make every reported gain a function of the number we are not allowed to
        # target.
        headroom = ceiling - operating_point
        rows.append({
            'bundle': bundle,
            'n_entries': int(bundles[bundle].shape[0]),
            'operating_point': operating_point,
            'operating_point_error': operating_point - CNRS_OPERATING_POINT,
            'ceiling': ceiling,
            'ceiling_error': ceiling - CNRS_CEILING,
            'headroom': headroom,
            'worst_gate_lattice_delta': worst,
            'condition': CONDITION_LABELS.get(bundle, ''),
            })
    return pd.DataFrame(rows)


def assign_splits(entries, seed):
    # By source entry, never by candidate, and stratified by (lattice x volume decile) so no
    # stratum is missing from a split. identifier is one row per CIF entry, so a split on it is
    # already a split by source entry.
    rng = np.random.default_rng(seed)
    manifest = entries[['identifier', 'bravais_lattice_true', 'volume_true']].copy()
    manifest['volume_decile'] = volume_decile(entries).to_numpy()
    manifest['split'] = ''
    names = list(SPLIT_FRACTIONS)
    fractions = np.array([SPLIT_FRACTIONS[name] for name in names])
    for _, group in manifest.groupby(['bravais_lattice_true', 'volume_decile'], sort=False):
        order = rng.permutation(group.shape[0])
        boundaries = np.round(np.cumsum(fractions) * group.shape[0]).astype(int)
        assignment = np.empty(group.shape[0], dtype=object)
        start = 0
        for name, stop in zip(names, boundaries):
            assignment[order[start:stop]] = name
            start = stop
        manifest.loc[group.index, 'split'] = assignment
    return manifest.rename(columns={'bravais_lattice_true': 'bravais_lattice'})


def commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                       text=True).strip()
    except Exception:
        return 'unknown'


def plot_grid(summary, per_lattice, out_path):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axis = axes[0]
    for target, value, style in [('operating point', CNRS_OPERATING_POINT, '-'),
                                 ('ceiling', CNRS_CEILING, '--')]:
        axis.axhline(100 * value, color='0.35', linestyle=style, linewidth=1.0,
                     label=f'CNRS {target} ({100*value:.1f}%)')
    order = np.argsort(summary['bundle'].to_numpy())
    positions = np.arange(order.size)
    axis.plot(positions, 100 * summary['ceiling'].to_numpy()[order], 'o-', color='#1f77b4',
              label='synthetic ceiling')
    axis.plot(positions, 100 * summary['operating_point'].to_numpy()[order], 's-', color='#d62728',
              label='synthetic operating point')
    axis.set_xticks(positions)
    axis.set_xticklabels(summary['bundle'].to_numpy()[order], rotation=45, ha='right', fontsize=8)
    axis.set_ylabel('CNRS-weighted success (%)')
    axis.set_title('Condition grid against the two CNRS numbers')
    axis.legend(fontsize=8, frameon=False)
    axis.grid(alpha=0.25)

    axis = axes[1]
    lattices = per_lattice['bravais_lattice'].to_numpy()
    positions = np.arange(lattices.size)
    axis.bar(positions - 0.2, 100 * per_lattice['cnrs_operating_point'].to_numpy(), width=0.4,
             color='0.55', label='CNRS Table 1')
    axis.bar(positions + 0.2, 100 * per_lattice['operating_point'].to_numpy(), width=0.4,
             color='#d62728', label='synthetic')
    for gate_lattice in COMPARISON_LATTICES:
        if gate_lattice in set(lattices):
            axis.axvspan(np.flatnonzero(lattices == gate_lattice)[0] - 0.45,
                         np.flatnonzero(lattices == gate_lattice)[0] + 0.45,
                         color='#ffe08a', alpha=0.35, zorder=0)
    axis.set_xticks(positions)
    axis.set_xticklabels(lattices, fontsize=8)
    axis.set_ylabel('operating point (%)')
    axis.set_title('Per-lattice profile, selected bundle (shaded = gate lattices)')
    axis.legend(fontsize=8, frameon=False)
    axis.grid(alpha=0.25, axis='y')

    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def write_markdown(summary, per_lattice, verdict, out_path):
    # Generated, not hand-written, so it cannot drift from the CSVs (PROTOCOL section 5).
    selected = verdict['bundle']
    lines = [
        '# S02 — synthetic condition grid',
        '',
        f'Generated by `run_fom_mirror_analysis.py` at commit `{verdict["commit"][:12]}`. '
        f'Do not edit by hand.',
        '',
        'Conditions are justified physically, not by matching the sealed benchmark (STATUS §6). '
        'The CNRS operating point (75.1%) and ceiling (89.0%) appear below as a sanity check only; '
        'nothing is tuned to or selected on them. Table 1\'s per-lattice totals are used as '
        'reweighting weights, per PLAN §6.4.',
        '',
        '## Condition grid',
        '',
        'CNRS-weighted. Headroom is ceiling − operating point: what a perfect re-ranker of the same '
        'pool could recover, and the denominator downstream gains should be quoted against.',
        '',
        '| bundle | condition | operating point | ceiling | headroom |',
        '|---|---|---|---|---|',
        ]
    for _, row in summary.sort_values('ceiling', ascending=False).iterrows():
        label = row['condition'] if isinstance(row['condition'], str) else ''
        marker = ' **← frozen**' if row['bundle'] == selected else ''
        lines.append(f'| `{row["bundle"]}`{marker} | {label} | {100*row["operating_point"]:.1f}% | '
                     f'{100*row["ceiling"]:.1f}% | {100*row["headroom"]:.1f} pts |')

    lines += [
        '',
        f'## Frozen bundle: `{selected}` ({verdict.get("condition", "")})',
        '',
        f'- entries: {verdict["n_entries"]}',
        f'- operating point {100*verdict["operating_point"]:.1f}%, '
        f'ceiling {100*verdict["ceiling"]:.1f}%, headroom {100*verdict["headroom"]:.1f} points',
        f'- splits: ' + ', '.join(f'{k} {v}' for k, v in sorted(verdict['split_counts'].items())),
        '',
        'Loss decomposition — note F-043: this split **inverts** between bundles, so it is a '
        'function of the conditions and not a property of the problem.',
        '',
        '| channel | fraction of all entries |',
        '|---|---|',
        ]
    for channel, value in verdict['loss_decomposition'].items():
        lines.append(f'| {channel.replace("lost_", "").replace("_", " ")} | {100*value:.1f}% |')

    lines += [
        '',
        '## Per Bravais lattice',
        '',
        'The `vs CNRS` column is informational. Note the systematic: the centred lattices are much '
        'easier synthetically than in the benchmark and aP/hP are harder, with opposite signs that '
        'partially cancel under weighting — an aggregate match would conceal both (F-042).',
        '',
        '| lattice | n | operating point | ceiling | CNRS (ref) | vs CNRS |',
        '|---|---|---|---|---|---|',
        ]
    for _, row in per_lattice.iterrows():
        lines.append(f'| {row["bravais_lattice"]} | {int(row["n_entries"])} | '
                     f'{100*row["operating_point"]:.1f}% | {100*row["found"]:.1f}% | '
                     f'{100*row["cnrs_operating_point"]:.1f}% | '
                     f'{100*row["delta_vs_cnrs"]:+.1f} pts |')
    lines += ['', 'Artefacts: `S02_mirror_bundles.csv`, `S02_mirror_per_lattice.csv`, '
              '`S02_mirror_per_decile.csv`, `S02_mirror_verdict.json`, `S02_mirror.png`, '
              '`S02_mirror_manifest.parquet`.', '']
    with open(out_path, 'w') as handle:
        handle.write('\n'.join(lines))


def main():
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_bundles(args.mirror_dir)
    if not bundles:
        raise SystemExit(f'No entries_*.parquet under {args.mirror_dir}')
    labels_by_bundle = {bundle: classify(frame, args.rtol_suffix)
                        for bundle, frame in bundles.items()}

    summary = bundle_summary(bundles, labels_by_bundle, rng)
    summary.to_csv(out_dir / 'S02_mirror_bundles.csv', index=False)
    print(summary.to_string(index=False))

    selected = args.select_bundle
    if selected is None:
        # Deliberately no "closest bundle" suggestion. Ranking bundles by their distance to the
        # sealed benchmark's numbers is the selection-by-tuning this analysis must not do; the
        # nominal bundle is designated physically (PLAN 6.3's C1, the characterised sigma).
        print(f'\nNominal condition is {NOMINAL_BUNDLE} (PLAN 6.3 C1, the characterised sigma(q2)). '
              f'Pass --select-bundle to freeze a manifest.')
        return

    if selected not in bundles:
        raise SystemExit(f'Bundle {selected} not found; have {sorted(bundles)}')

    entries = bundles[selected]
    labels = labels_by_bundle[selected]
    per_lattice = per_lattice_table(entries, labels, rng)
    per_lattice.to_csv(out_dir / 'S02_mirror_per_lattice.csv', index=False)
    per_decile_table(entries, labels).to_csv(out_dir / 'S02_mirror_per_decile.csv', index=False)

    manifest = assign_splits(entries, args.seed)
    manifest['condition_bundle'] = selected
    manifest['broadening_tag'] = '1'
    manifest['commit'] = commit_hash()
    manifest['seed'] = args.seed
    manifest.to_parquet(out_dir / 'S02_mirror_manifest.parquet', index=False)

    plot_grid(summary, per_lattice, out_dir / 'S02_mirror.png')

    selected_row = summary.loc[summary['bundle'] == selected].iloc[0]
    comparison = per_lattice.loc[per_lattice['bravais_lattice'].isin(COMPARISON_LATTICES)]
    verdict = {
        'bundle': selected,
        'n_entries': int(entries.shape[0]),
        'operating_point': float(selected_row['operating_point']),
        'operating_point_target': CNRS_OPERATING_POINT,
        'ceiling': float(selected_row['ceiling']),
        'ceiling_target': CNRS_CEILING,
        'condition': CONDITION_LABELS.get(selected, ''),
        'is_nominal': selected == NOMINAL_BUNDLE,
        'headroom': float(selected_row['headroom']),
        # Informational only -- see the CONDITION_LABELS comment. Not a pass/fail criterion.
        'cnrs_comparison_per_lattice': {row['bravais_lattice']: float(row['delta_vs_cnrs'])
                                        for _, row in comparison.iterrows()},
        'loss_decomposition': {column: float(labels[column].mean()) for column in
                               ['lost_rank_failure', 'lost_threshold_failure', 'lost_both',
                                'lost_not_found']},
        'split_counts': manifest['split'].value_counts().to_dict(),
        'commit': commit_hash(),
        }
    with open(out_dir / 'S02_mirror_verdict.json', 'w') as verdict_file:
        json.dump(verdict, verdict_file, indent=2)
    write_markdown(summary, per_lattice, verdict, out_dir / 'S02_mirror.md')
    print('\n' + json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
