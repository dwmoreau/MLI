"""S06 -- read the iteration pilot, and say whether the ceiling moved.

The decision rule is the handoff's and it is deliberately conservative: **take the reduction only
if the ceiling does not move**, with an interval, per lattice. The reason is not statistical
caution. Reducing the iterations changes the candidate pool relative to the configuration that
ships, so a merit selected on a cheap pool is being chosen against a distribution users will never
see. If the ceiling moves at all, the honest options are to keep the full schedule or to change
the shipped default too -- not to quietly develop against a different search.

WHAT IS PAIRED WITH WHAT. Every arm runs the same entries under the same per-pattern seeds, so
the comparison is McNemar over discordant pairs, not two independent proportions. The bootstrap
unit is the source entry throughout; the pilot runs one condition bundle, so entry and row
coincide here, but the reporting code says so rather than relying on it.

POWER IS PART OF THE RESULT. "No detectable change" is a finding only if the pilot could have
detected one, so every row carries the minimum effect it had 80 % power against. Where the
discordant count is zero the interval is the rule of three, which is the honest statement of what
a null result on n entries is worth.

CUBIC IS EXCLUDED from the pooled per-lattice comparison. It runs five random passes on ten peaks
where triclinic runs sixty on twenty, so a scale factor does not mean the same thing there -- at a
quarter schedule cubic's block rounds to a single pass. It is reported in its own right and left
out of the aggregate.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

# The reference arm every other arm is contrasted against: the shipped schedule.
REFERENCE_SCALE = 1.0

# Excluded from the aggregate, reported on its own. See the module docstring.
CUBIC = ('cF', 'cI', 'cP')


def wilson(successes, total, z=1.96):
    """Wilson score interval. Behaves at 0 and at 1, where Wald gives a zero-width interval."""
    if total == 0:
        return (float('nan'), float('nan'))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(reference, arm):
    """Paired comparison of two boolean arrays over the same entries.

    Returns the discordant counts, the paired difference in the reduced arm's favour, its
    standard error, an exact two-sided binomial p, and the minimum effect the comparison had
    80 % power against.

    `b` is where the full schedule finds a correct cell and the reduced one does not -- the
    direction that refuses the reduction. `c` is the reverse, which is not evidence *for* the
    reduction so much as evidence that the search is noisy (C2-R-005: two runs of the same 972
    hard pattern-conditions agreed on 328 and disagreed on 130).
    """
    from scipy import stats

    reference = np.asarray(reference, dtype=bool)
    arm = np.asarray(arm, dtype=bool)
    n = reference.size
    b = int(np.sum(reference & ~arm))
    c = int(np.sum(~reference & arm))
    delta = (c - b) / n if n else float('nan')
    if b + c == 0:
        # Rule of three: with no discordant pair in n entries, the 95 % bound on the per-entry
        # discordance rate is 3/n, and that -- not zero -- is what the null result is worth.
        return {'n': n, 'b': b, 'c': c, 'delta': delta, 'se': float('nan'), 'p': 1.0,
                'mde_80': 3.0 / n if n else float('nan'), 'basis': 'rule_of_three'}
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    se = np.sqrt(b + c - (b - c) ** 2 / n) / n
    # The classic paired-proportion approximation: with `b + c` discordant pairs observed, an
    # effect this size would be detected 80 % of the time at alpha = 0.05.
    mde_80 = (1.96 + 0.84) * np.sqrt(b + c) / n
    return {'n': n, 'b': b, 'c': c, 'delta': delta, 'se': float(se), 'p': p,
            'mde_80': float(mde_80), 'basis': 'mcnemar'}


def load_arms(out_root, scales):
    """The pilot's two tables per arm, concatenated, with the arm on every row."""
    from mlindex.scripts.run_fom_iteration_pilot import scale_tag

    root = Path(BASE) / out_root
    lattice, entry = [], []
    for scale in scales:
        tag = scale_tag(scale)
        lattice_path = root / f'lattice_{tag}.parquet'
        entry_path = root / f'entry_{tag}.parquet'
        if not (lattice_path.exists() and entry_path.exists()):
            raise SystemExit(f'arm {tag} is missing from {root}; run it before reporting')
        lattice.append(pd.read_parquet(lattice_path))
        entry.append(pd.read_parquet(entry_path))
    return pd.concat(lattice, ignore_index=True), pd.concat(entry, ignore_index=True)


def restrict_to_common_entries(entries, scales):
    """Keep only entries every arm actually ran.

    The arms are paired by construction -- same entries, same per-pattern seeds -- but an arm that
    was stopped early, or one entry that failed in one arm and not another, breaks that silently:
    the unpaired ceiling of a truncated arm is then computed over a different, and here a
    systematically easier, set of crystals. The pilot samples in lattice order, so the first half
    of a truncated arm is all high-symmetry, whose ceiling is 1.000 by construction.

    Restricting up front means every column in the table -- paired and unpaired alike -- describes
    the same crystals, and the count is printed so a restriction is visible rather than inferred.
    """
    common = None
    for scale in scales:
        ids = set(entries.loc[entries['iteration_scale'] == scale, 'entry_id'])
        common = ids if common is None else (common & ids)
    dropped = entries['entry_id'].nunique() - len(common)
    if dropped:
        print(f'restricting to {len(common)} entries run by every arm ({dropped} dropped)',
              flush=True)
    return entries[entries['entry_id'].isin(common)].reset_index(drop=True), sorted(common)


def ceiling_table(entries, scales):
    """Ceiling, rank and wall clock per (lattice, arm), each reduced arm paired against the full.

    The unit is the source entry and `reachable` is the ceiling: a correct cell exists ANYWHERE
    in the pool. `pooled_rank` is the outcome -- where it lands in the list `run.py` prints -- and
    is reported beside it because a schedule that holds the ceiling while pushing the correct cell
    down the list has still changed what a merit has to work with.
    """
    reference = entries[entries['iteration_scale'] == REFERENCE_SCALE]
    reference_by_entry = reference.set_index('entry_id')['reachable']
    # The outcome, paired the same way. The ceiling is what the handoff calls decisive, but a
    # schedule that holds the ceiling while pushing the correct cell down the printed list has
    # still changed what a merit has to work with -- and that is the quantity this campaign is
    # about. Reported beside the ceiling, never instead of it.
    reference_top10 = reference.set_index('entry_id')['pooled_rank'].between(0, 9)
    rows = []
    for scale in scales:
        arm = entries[entries['iteration_scale'] == scale]
        for lattice in ['ALL', 'ALL_NONCUBIC'] + sorted(entries['bravais_lattice_true'].unique()):
            if lattice == 'ALL':
                subset = arm
            elif lattice == 'ALL_NONCUBIC':
                subset = arm[~arm['bravais_lattice_true'].isin(CUBIC)]
            else:
                subset = arm[arm['bravais_lattice_true'] == lattice]
            if subset.empty:
                continue
            reachable = subset['reachable'].to_numpy(dtype=bool)
            paired = reference_by_entry.reindex(subset['entry_id']).to_numpy(dtype=bool)
            low, high = wilson(int(reachable.sum()), reachable.size)
            found = subset[subset['pooled_rank'] >= 0]['pooled_rank'].to_numpy()
            row = {
                'iteration_scale': scale,
                'bravais_lattice': lattice,
                'n_entries': int(reachable.size),
                'n_reachable': int(reachable.sum()),
                'ceiling': float(reachable.mean()),
                'ceiling_ci_low': low,
                'ceiling_ci_high': high,
                # The printed-list outcome, reported beside the ceiling rather than instead of it.
                'top10': float(np.mean((subset['pooled_rank'] >= 0)
                                       & (subset['pooled_rank'] < 10))),
                'median_rank_when_found': float(np.median(found)) if found.size else float('nan'),
                'seconds_per_entry': float(subset['seconds'].mean()),
                }
            row.update({f'paired_{name}': value
                        for name, value in mcnemar(paired, reachable).items()})
            in_top10 = subset['pooled_rank'].between(0, 9).to_numpy()
            paired_top10 = reference_top10.reindex(subset['entry_id']).to_numpy(dtype=bool)
            row.update({f'top10_{name}': value
                        for name, value in mcnemar(paired_top10, in_top10).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def pool_table(lattice_rows, scales):
    """Survivor counts per (lattice, arm) -- the input to S06's negative-subsampling sizing.

    These are the only measured pool sizes at the generation cut of 1.5 that campaign 2 has, and
    K cannot be sized without them: S05's projection of 1.5 billion survivor rows is ~590 per
    lattice, which would make the default K of 500 very nearly a no-op.
    """
    rows = []
    for scale in scales:
        arm = lattice_rows[lattice_rows['iteration_scale'] == scale]
        for lattice in ['ALL'] + sorted(lattice_rows['bravais_lattice'].unique()):
            subset = arm if lattice == 'ALL' else arm[arm['bravais_lattice'] == lattice]
            if subset.empty:
                continue
            sizes = subset['pool_size'].to_numpy()
            rows.append({
                'iteration_scale': scale,
                'bravais_lattice': lattice,
                'n_cells': int(sizes.size),
                'pool_mean': float(sizes.mean()),
                'pool_median': float(np.median(sizes)),
                'pool_p90': float(np.percentile(sizes, 90)),
                'pool_max': int(sizes.max()),
                'pool_total': int(sizes.sum()),
                'seconds_mean': float(subset['seconds'].mean()),
                })
    return pd.DataFrame(rows)


def figure(ceiling, pools, path, scales):
    """Two panels, because the result has two halves and one of them is the decisive one.

    Left: does the ceiling move. Right: what the reduction buys against what it costs -- which is
    where the answer actually lives, since the ceiling turns out not to move at all.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mlindex.scripts.run_fom_iteration_pilot import scale_tag

    reduced = [scale for scale in scales if scale != REFERENCE_SCALE]
    lattices = [value for value in ceiling['bravais_lattice'].unique()
                if value not in ('ALL', 'ALL_NONCUBIC')]
    # Cubic last and shaded: five random passes on ten peaks is a different schedule, so a scale
    # factor does not mean there what it means for triclinic's sixty.
    order = sorted(lattices, key=lambda bl: (bl in CUBIC, bl))
    palette = ['#22223b', '#3f7d92', '#c1666b', '#7f9c6a']
    colours = {scale: palette[index % len(palette)] for index, scale in enumerate(scales)}
    positions = np.arange(len(order))

    figure_, axes = plt.subplots(1, 2, figsize=(14.5, 5.4),
                                 gridspec_kw={'width_ratios': [1.15, 1.0]})

    # ---- left: the ceiling, per lattice -------------------------------------------------
    ax = axes[0]
    for bl in CUBIC:
        if bl in order:
            ax.axvspan(order.index(bl) - 0.5, order.index(bl) + 0.5, color='0.94', zorder=0)
    offsets = np.linspace(-0.16, 0.16, len(scales)) if len(scales) > 1 else [0.0]
    for offset, scale in zip(offsets, scales):
        arm = ceiling[ceiling['iteration_scale'] == scale].set_index('bravais_lattice')
        values = np.array([arm['ceiling'].get(bl, np.nan) for bl in order])
        low = values - np.array([arm['ceiling_ci_low'].get(bl, np.nan) for bl in order])
        high = np.array([arm['ceiling_ci_high'].get(bl, np.nan) for bl in order]) - values
        ax.errorbar(positions + offset, values, yerr=[low, high], fmt='o', ms=5,
                    color=colours[scale], capsize=3, lw=1.2, label=scale_tag(scale), zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_ylim(0.60, 1.03)
    ax.set_ylabel('ceiling: a correct cell anywhere in the pool')
    ax.set_title('The ceiling does not move, on any lattice')
    ax.legend(title='schedule', frameon=False, loc='lower left')
    aggregate = ceiling[(ceiling['bravais_lattice'] == 'ALL_NONCUBIC')
                        & (ceiling['iteration_scale'].isin(reduced))]
    if not aggregate.empty:
        row = aggregate.iloc[0]
        ax.text(0.98, 0.06,
                f"non-cubic aggregate, paired: {row['paired_b']:.0f} lost / "
                f"{row['paired_c']:.0f} gained of {row['n_entries']:.0f}\n"
                f"change {row['paired_delta'] * 100:+.2f} pp, "
                f"detectable at 80 % power: {row['paired_mde_80'] * 100:.1f} pp\n"
                f"per lattice n = 30, so the per-lattice bound is 10 pp",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color='0.25')
    ax.text(0.985, 0.965, 'shaded: cubic', transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='0.45')

    # ---- right: the trade -----------------------------------------------------------------
    # The decisive panel. A reduction is worth taking only if it buys wall clock; this one buys
    # very little and enlarges the pool that then has to be stored, labelled and ranked.
    ax = axes[1]
    reference_pools = (pools[pools['iteration_scale'] == REFERENCE_SCALE]
                       .set_index('bravais_lattice'))
    reference_seconds = (ceiling[(ceiling['iteration_scale'] == REFERENCE_SCALE)]
                         .set_index('bravais_lattice')['seconds_per_entry'])
    width = 0.7 / max(1, len(reduced))
    for index, scale in enumerate(reduced):
        arm_pools = pools[pools['iteration_scale'] == scale].set_index('bravais_lattice')
        arm_ceiling = (ceiling[ceiling['iteration_scale'] == scale]
                       .set_index('bravais_lattice')['seconds_per_entry'])
        pool_change = np.array([
            100.0 * (arm_pools['pool_mean'].get(bl, np.nan)
                     / reference_pools['pool_mean'].get(bl, np.nan) - 1.0) for bl in order])
        time_change = np.array([
            100.0 * (arm_ceiling.get(bl, np.nan) / reference_seconds.get(bl, np.nan) - 1.0)
            for bl in order])
        ax.bar(positions + index * width - 0.35 + width / 2, pool_change, width,
               color=colours[scale], label=f'{scale_tag(scale)}: pool size', zorder=2)
        ax.plot(positions + index * width - 0.35 + width / 2, time_change, 'D', ms=6.5,
                color='#c1666b', markeredgecolor='white', markeredgewidth=1.1,
                label=f'{scale_tag(scale)}: wall clock', zorder=4)
    ax.axhline(0.0, color='0.35', lw=1.0, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_ylabel('change against the full schedule (%)')
    ax.set_title('What it buys, and what it costs')
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    # Headroom so the markers below the axis are never crowded by the frame.
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.12 * (high - low), high + 0.06 * (high - low))

    figure_.suptitle('S06 — the iteration lever, priced', y=0.985, fontsize=13)
    # The mechanism, as a figure caption rather than inside a panel, where it sat on the data.
    figure_.text(0.5, 0.945,
                 'Halving the schedule leaves the ceiling untouched and the pool 17.8 % larger: a '
                 'shorter search leaves candidates less converged, so deduplication collapses '
                 'fewer of them\nand the post-cut block spends back what the search saved. '
                 'Wall clock falls 4.7 % in aggregate — and rises on hP, hR and tI.',
                 ha='center', va='top', fontsize=9, color='0.3')
    figure_.tight_layout(rect=(0, 0, 1, 0.90))
    figure_.savefig(path, dpi=200)
    plt.close(figure_)


def report(args, scales):
    lattice_rows, entries = load_arms(args.out_root, scales)
    entries, common = restrict_to_common_entries(entries, scales)
    lattice_rows = lattice_rows[lattice_rows['entry_id'].isin(common)].reset_index(drop=True)
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ceiling = ceiling_table(entries, scales)
    pools = pool_table(lattice_rows, scales)
    ceiling.to_csv(artifact_dir / 'S06_iteration_pilot.csv', index=False)
    pools.to_csv(artifact_dir / 'S06_iteration_pilot_pools.csv', index=False)
    figure(ceiling, pools, artifact_dir / 'S06_iteration_pilot.png', scales)

    show = ceiling[ceiling['bravais_lattice'].isin(['ALL', 'ALL_NONCUBIC'])]
    with pd.option_context('display.width', 200, 'display.max_columns', 40):
        print(show[['iteration_scale', 'bravais_lattice', 'n_entries', 'ceiling',
                    'ceiling_ci_low', 'ceiling_ci_high', 'seconds_per_entry',
                    'paired_b', 'paired_c', 'paired_delta', 'paired_p', 'paired_mde_80']]
              .to_string(index=False))
        print()
        print(show[['iteration_scale', 'bravais_lattice', 'top10', 'top10_b', 'top10_c',
                    'top10_delta', 'top10_p', 'top10_mde_80']].to_string(index=False))
    print()
    print(pools[pools['bravais_lattice'] == 'ALL'].to_string(index=False))
    print(f'\nwrote {artifact_dir}/S06_iteration_pilot.{{csv,png}} '
          f'and S06_iteration_pilot_pools.csv')
