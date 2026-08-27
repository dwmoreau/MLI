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


def ceiling_table(entries, scales):
    """Ceiling, rank and wall clock per (lattice, arm), each reduced arm paired against the full.

    The unit is the source entry and `reachable` is the ceiling: a correct cell exists ANYWHERE
    in the pool. `pooled_rank` is the outcome -- where it lands in the list `run.py` prints -- and
    is reported beside it because a schedule that holds the ceiling while pushing the correct cell
    down the list has still changed what a merit has to work with.
    """
    reference = entries[entries['iteration_scale'] == REFERENCE_SCALE]
    reference_by_entry = reference.set_index('entry_id')['reachable']
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mlindex.scripts.run_fom_iteration_pilot import scale_tag

    lattices = [value for value in ceiling['bravais_lattice'].unique()
                if value not in ('ALL', 'ALL_NONCUBIC')]
    order = sorted(lattices, key=lambda bl: (bl in CUBIC, bl))
    colours = {scale: colour for scale, colour in
               zip(scales, ['#22223b', '#4a7c93', '#c1666b', '#7f9c6a', '#b08968'])}

    figure_, axes = plt.subplots(1, 3, figsize=(15.5, 5.0),
                                 gridspec_kw={'width_ratios': [2.1, 1.0, 1.0]})

    ax = axes[0]
    width = 0.8 / max(1, len(scales))
    positions = np.arange(len(order))
    for index, scale in enumerate(scales):
        arm = ceiling[ceiling['iteration_scale'] == scale].set_index('bravais_lattice')
        values = [arm['ceiling'].get(bl, np.nan) for bl in order]
        low = [arm['ceiling'].get(bl, np.nan) - arm['ceiling_ci_low'].get(bl, np.nan)
               for bl in order]
        high = [arm['ceiling_ci_high'].get(bl, np.nan) - arm['ceiling'].get(bl, np.nan)
                for bl in order]
        ax.bar(positions + index * width, values, width, label=scale_tag(scale),
               color=colours[scale], yerr=[low, high], capsize=2, error_kw={'lw': 0.8})
    ax.set_xticks(positions + width * (len(scales) - 1) / 2)
    ax.set_xticklabels(order)
    ax.set_ylabel('ceiling: correct cell anywhere in the pool')
    ax.set_title('The ceiling, per lattice, with Wilson intervals')
    ax.legend(title='schedule', frameon=False)
    for bl in CUBIC:
        if bl in order:
            ax.axvspan(order.index(bl) - 0.5, order.index(bl) + 0.5, color='0.92', zorder=0)
    ax.text(0.99, 0.02, 'shaded: cubic, excluded from the aggregate\n(5 passes on 10 peaks)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.35')

    ax = axes[1]
    aggregate = ceiling[ceiling['bravais_lattice'] == 'ALL_NONCUBIC']
    ax.axhline(0.0, color='0.6', lw=0.8)
    ax.errorbar(range(len(aggregate)), aggregate['paired_delta'] * 100,
                yerr=aggregate['paired_se'] * 100, fmt='o', color='#22223b', capsize=3)
    for position, row in enumerate(aggregate.itertuples()):
        ax.annotate(f'MDE {row.paired_mde_80 * 100:.1f} pp',
                    (position, row.paired_delta * 100), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=8, color='0.35')
    ax.set_xticks(range(len(aggregate)))
    ax.set_xticklabels([scale_tag(value) for value in aggregate['iteration_scale']])
    ax.set_ylabel('paired change in ceiling (pp)')
    ax.set_title('Paired against the full schedule\n(non-cubic aggregate)')

    ax = axes[2]
    for index, scale in enumerate(scales):
        arm = pools[(pools['iteration_scale'] == scale) & (pools['bravais_lattice'] != 'ALL')]
        arm = arm.set_index('bravais_lattice').reindex(order)
        ax.bar(positions + index * width, arm['pool_median'], width,
               label=scale_tag(scale), color=colours[scale])
    ax.set_xticks(positions + width * (len(scales) - 1) / 2)
    ax.set_xticklabels(order, rotation=90)
    ax.set_yscale('log')
    ax.set_ylabel('median survivors per (entry, lattice)')
    ax.set_title('Pool size at the generation cut of 1.5')

    figure_.tight_layout()
    figure_.savefig(path, dpi=200)
    plt.close(figure_)


def report(args, scales):
    lattice_rows, entries = load_arms(args.out_root, scales)
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
                    'ceiling_ci_low', 'ceiling_ci_high', 'top10', 'seconds_per_entry',
                    'paired_b', 'paired_c', 'paired_delta', 'paired_p', 'paired_mde_80']]
              .to_string(index=False))
    print()
    print(pools[pools['bravais_lattice'] == 'ALL'].to_string(index=False))
    print(f'\nwrote {artifact_dir}/S06_iteration_pilot.{{csv,png}} '
          f'and S06_iteration_pilot_pools.csv')
