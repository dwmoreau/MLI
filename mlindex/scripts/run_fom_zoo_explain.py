"""S06 steps 5-7: is the figure of merit even applicable, where does it fail, and why.

The leaderboard is `run_fom_zoo_eval.py`. This is the half the phase note calls the point: the
diagnostics that say *why* one merit beats another, which is the input to Phase 4's design.

Five measurements, each answering a question the handoff poses:

  * **V/V_crit (section 3c).** Werner 1976: above a critical volume a figure of merit reports the
    precision of the data rather than the correctness of the cell. F-062 promoted this from one
    stratifier among many to a headline, because in the hard region the correct cell's M20 is
    4.4-8.1 and the *wrong* winner outscores it by a median factor of 1.43 -- which is what
    "the FOM was never applicable here" looks like. Reported as a sweep over g_min, because
    V_crit is proportional to 1/g_min and so the whole sweep is a rescale of one stored column
    (Q14). `M_werner_frac` needs no floor at all: g_min multiplies every candidate equally, so
    its ranking within an entry is exactly g_min-invariant.
  * **Cross-lattice bias (section 4, F-002).** Wu 1988 Table 1 predicts mean M20/M'20 of 1.82 for
    cubic falling to 1.00 for triclinic, from the uniform-spacing approximation alone. `run.py`
    pools all fourteen lattices and sorts on raw M20, so if that ratio is real the pooled ranking
    carries it. Measured here on the *incorrect* candidates, which is the null this project has
    to standardise.
  * **The over-prediction axis (section 4).** The assumption behind several later tasks is that
    the dominant failure is a large, low-symmetry cell out-scoring the correct one. Confirmed or
    refuted here, and flagged loudly if refuted.
  * **Complementarity (section 4).** For how many entries does merit A rank the correct cell in
    the top ten when merit B does not, and what does a union oracle over the whole zoo reach?
    That bounds what S08's combiner can achieve and says which merits are worth including.
  * **The C0 singularity (F-054).** Zero error means zero residual, so M20 diverges on the
    control bundle. Every merit with a residual denominator has the same singularity; which ones
    is a measurement, and S06 owes it.

    python mlindex/scripts/run_fom_zoo_explain.py

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.model_training import FomMetrics  # noqa: E402
from mlindex.utilities.FigureOfMerits import WU88_M20_RATIO  # noqa: E402
from mlindex.scripts.run_fom_zoo_eval import bundle_frames  # noqa: E402
from mlindex.scripts.run_fom_zoo_eval import evaluate_merit  # noqa: E402
from mlindex.scripts.run_fom_zoo_features import EVALUABLE_BUNDLES  # noqa: E402
from mlindex.scripts.run_fom_zoo_features import commit_hash  # noqa: E402

# The merits the section 3d explanation is actually about, one per design axis. The band analysis
# runs two `evaluate` passes per merit, so it is deliberately not all twenty-one.
EXPLAINED = ('M20', 'M_1', 'M_wu', 'M_rev', 'M_sym', 'null_tail_nll', 'M_werner_frac', 'M_star')

# The columns the pool geometry table needs beyond the score-independent ones.
GEOMETRY_COLUMNS = ('M20', 'volume', 'volume_ratio_to_truth', 'V_over_Vcrit', 'X_N')

# g_min enters V_over_Vcrit multiplicatively, so a sweep over it is a sweep of the cut on one
# stored column: "above V_crit at g_min = g" is "stored value above 1/g".
#
# The range is set by where the boundary actually falls, which is a measurement, not a guess.
# g_min is a mean discrepancy in q2 units, so it is O(1e-4 / Angstrom^2), not O(1) -- measured on
# this pool the median correct candidate crosses V/V_crit = 1 at g_min = 1.5e-4. A first attempt
# swept 0.1 to 10 and reported "100% above V_crit" at every point, which was arithmetic about the
# units rather than a result. Two decades either side of the crossing is what shows the shape.
G_MIN_SWEEP = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)

# Bands for the in-domain/out-of-domain comparison are **quantiles** of V/V_crit, not a cut at 1.
# V/V_crit is monotone in g_min, so a quantile band is exactly g_min-invariant and the comparison
# needs no floor chosen (Q14); a cut at 1 would need one and would put every candidate on the same
# side of it at most floors.
VCRIT_BANDS = 4

SERIES_BLUE = '#2a78d6'
TARGET_RED = '#e34948'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
BAND_GREY = '#c9c9c4'


def pool_geometry(benchmark_dir, feature_dir, bundles, keep_entry_ids, entries):
    """One row per (entry, condition): what the correct cell looks like, and what beat it.

    The ranking facts come from `FomMetrics.reduce_pool`, which owns the canonical total order
    (score descending, then Bravais lattice, then candidate_id). The *identity* of the winning
    candidate is not something `reduce_pool` returns, so it is recovered here by applying the
    same order -- and cross-checked against the Bravais lattice `reduce_pool` reports, so a
    divergence is caught rather than assumed away.
    """
    rows = []
    mismatches = 0
    for frame in bundle_frames(benchmark_dir, feature_dir, bundles, keep_entry_ids,
                               GEOMETRY_COLUMNS):
        values = frame['M20'].to_numpy(dtype=np.float64)
        reduced = FomMetrics.reduce_pool(frame, values, pool='cross_bl')

        order = pd.Categorical(frame['bravais_lattice'],
                               categories=FomMetrics.BRAVAIS_LATTICES).codes
        rank = np.lexsort((frame['candidate_id'].to_numpy(), order, -values))
        ordered = frame.iloc[rank]
        top = ordered.groupby(['entry_id', 'condition_bundle'], sort=False).head(1)
        top = top.set_index(['entry_id', 'condition_bundle'])

        correct = frame.loc[FomMetrics.as_bool(frame['is_correct'])]
        correct_order = pd.Categorical(correct['bravais_lattice'],
                                       categories=FomMetrics.BRAVAIS_LATTICES).codes
        correct = correct.iloc[np.lexsort((correct['candidate_id'].to_numpy(), correct_order,
                                           -correct['M20'].to_numpy()))]
        correct = correct.groupby(['entry_id', 'condition_bundle'], sort=False).head(1)
        correct = correct.set_index(['entry_id', 'condition_bundle'])

        reduced = reduced.set_index(['entry_id', 'condition_bundle'])
        mismatches += int((top['bravais_lattice'].reindex(reduced.index)
                           != reduced['bravais_lattice_top_all']).sum())

        block = pd.DataFrame(index=reduced.index)
        block['has_correct'] = reduced['has_correct_all'].to_numpy()
        block['rank_best_correct'] = reduced['rank_best_correct_all'].to_numpy()
        block['top_is_correct'] = reduced['top_is_correct_all'].to_numpy()
        block['bravais_lattice_top'] = reduced['bravais_lattice_top_all'].to_numpy()
        for column in ('M20', 'V_over_Vcrit', 'volume_ratio_to_truth', 'X_N'):
            block[f'top_{column}'] = top[column].reindex(reduced.index).to_numpy()
            block[f'correct_{column}'] = correct[column].reindex(reduced.index).to_numpy()
        rows.append(block.reset_index())
    geometry = pd.concat(rows, ignore_index=True)

    context = FomMetrics.entry_context(entries)
    keep = ['entry_id', 'condition_bundle', 'bravais_lattice', 'volume_decile', 'is_hard',
            'condition_label', 'cnrs_weight']
    geometry = geometry.merge(context[keep], on=['entry_id', 'condition_bundle'], how='left')
    return geometry, mismatches


def vcrit_sweep(geometry):
    """What fraction of correct cells, and of the cells that beat them, sit above V_crit.

    V_over_Vcrit is stored at g_min = 1 and is linear in g_min, so "above V_crit at g_min = g" is
    "stored value above 1/g". The sweep is therefore exact rather than interpolated.
    """
    reachable = geometry.loc[geometry['has_correct'].astype(bool)]
    rows = []
    for g_min in G_MIN_SWEEP:
        cut = 1.0/g_min
        for scope, frame in (('all reachable', reachable),
                             ('hard stratum', reachable.loc[reachable['is_hard']])):
            if not frame.shape[0]:
                continue
            wrong = frame.loc[~frame['top_is_correct'].astype(bool)]
            rows.append({
                'g_min': g_min,
                'scope': scope,
                'n_entries': int(frame.shape[0]),
                'median_correct_V_over_Vcrit':
                    float(frame['correct_V_over_Vcrit'].median()*g_min),
                'median_winner_V_over_Vcrit':
                    float(frame['top_V_over_Vcrit'].median()*g_min),
                'frac_correct_above_Vcrit':
                    float((frame['correct_V_over_Vcrit'] > cut).mean()),
                'frac_winner_above_Vcrit':
                    float((frame['top_V_over_Vcrit'] > cut).mean()),
                'frac_wrong_winner_above_Vcrit':
                    float((wrong['top_V_over_Vcrit'] > cut).mean()) if wrong.shape[0] else np.nan,
                })
    return pd.DataFrame(rows)


def cross_lattice_null(benchmark_dir, feature_dir, bundles, keep_entry_ids, merits):
    """The distribution of each merit over *incorrect* candidates, per Bravais lattice.

    This is the null S07 has to standardise, and F-002's prediction is about its location: Wu
    1988 puts cubic 1.82x above triclinic on M20 from the uniform-spacing approximation alone.
    """
    accumulated = {}
    systems = {}
    for frame in bundle_frames(benchmark_dir, feature_dir, bundles, keep_entry_ids, merits):
        wrong = frame.loc[~FomMetrics.as_bool(frame['is_correct'])]
        for lattice, group in wrong.groupby('bravais_lattice', sort=False):
            # Wu's table is per crystal *system*, not per Bravais lattice, so the system has to
            # be carried through from the pool rather than inferred from the lattice label.
            systems[lattice] = group['lattice_system'].iloc[0]
            store = accumulated.setdefault(lattice, {name: [] for name in merits})
            for name in merits:
                # Subsample: the medians are over millions of rows and a 2% sample moves the
                # quantiles far below the reproducibility floor, while the full arrays do not fit.
                values = group[name].to_numpy(dtype=np.float64)
                store[name].append(values[::50])

    rows = []
    for lattice, store in accumulated.items():
        row = {'bravais_lattice': lattice, 'lattice_system': systems[lattice]}
        for name, chunks in store.items():
            values = np.concatenate(chunks) if chunks else np.array([])
            values = values[np.isfinite(values)]
            row[f'{name}_median'] = float(np.median(values)) if values.size else np.nan
            row[f'{name}_p10'] = float(np.percentile(values, 10)) if values.size else np.nan
            row[f'{name}_p90'] = float(np.percentile(values, 90)) if values.size else np.nan
        # How hard `prune_below_m20` truncates this lattice's null. A p10 sitting at the cut means
        # the distribution has had its lower tail removed and its median is not a location
        # estimate of anything (F-068).
        m20 = np.concatenate(store['M20']) if store.get('M20') else np.array([])
        m20 = m20[np.isfinite(m20)]
        row['frac_within_10pct_of_prune'] = (
            float(((m20 >= 5.0) & (m20 < 5.5)).mean()) if m20.size else np.nan
            )
        row['n_sampled'] = int(sum(len(chunk) for chunk in store[merits[0]]))
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values('bravais_lattice').reset_index(drop=True)

    # Wu's ratio is quoted against triclinic, so ours is too.
    reference = frame.loc[frame['bravais_lattice'] == 'aP']
    for name in merits:
        base = float(reference[f'{name}_median'].iloc[0]) if reference.shape[0] else np.nan
        frame[f'{name}_ratio_to_aP'] = frame[f'{name}_median']/base
    frame['wu88_predicted_ratio'] = frame['lattice_system'].map(WU88_M20_RATIO)
    return frame


def over_prediction(geometry):
    """Where a wrong candidate outranks the correct one, is it bigger and less symmetric?

    Several later tasks assume it is. `volume_ratio_to_truth` is the candidate's volume over the
    true one, so the winner's value *is* V_wrong/V_correct up to the correct candidate's own
    (near-unit) ratio, which is reported beside it.
    """
    beaten = geometry.loc[geometry['has_correct'].astype(bool)
                          & ~geometry['top_is_correct'].astype(bool)]
    rows = []
    for scope, frame in (('all reachable', beaten),
                         ('hard stratum', beaten.loc[beaten['is_hard']])):
        if not frame.shape[0]:
            continue
        ratio = (frame['top_volume_ratio_to_truth']
                 / frame['correct_volume_ratio_to_truth'].replace(0, np.nan))
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        symmetry = FomMetrics.BRAVAIS_LATTICES
        rank = {lattice: position for position, lattice in enumerate(symmetry)}
        # BRAVAIS_LATTICES runs high symmetry to low, so a higher index is a lower-symmetry cell.
        moved_down = (frame['bravais_lattice_top'].map(rank)
                      > frame['bravais_lattice'].map(rank))
        rows.append({
            'scope': scope,
            'n_entries': int(frame.shape[0]),
            'median_V_wrong_over_V_correct': float(ratio.median()),
            'p25': float(ratio.quantile(0.25)),
            'p75': float(ratio.quantile(0.75)),
            'frac_wrong_cell_larger': float((ratio > 1).mean()),
            'frac_wrong_cell_lower_symmetry': float(moved_down.mean()),
            'frac_same_lattice': float(
                (frame['bravais_lattice_top'] == frame['bravais_lattice']).mean()),
            'median_top_M20': float(frame['top_M20'].median()),
            'median_correct_M20': float(frame['correct_M20'].median()),
            })
    return pd.DataFrame(rows)


# Oishi-Tomiyasu 2013 excluded candidates on these before comparing figures of merit, and warned
# in the same breath that doing so flatters every one of them. Q5 settled our policy as "not in
# the headline", so this exists to measure the size of the flattery once.
PREFILTER = dict(M_tilde_min=3.0, M_rev_min=1.0, N_cal_min=12.0, N_cal_max=120.0)


def prefilter_mask(frame):
    return ((frame['M_tilde'] >= PREFILTER['M_tilde_min'])
            & (frame['M_rev'] >= PREFILTER['M_rev_min'])
            & (frame['N_cal'] >= PREFILTER['N_cal_min'])
            & (frame['N_cal'] <= PREFILTER['N_cal_max'])).to_numpy()


def prefilter_sensitivity(benchmark_dir, feature_dir, bundles, keep_entry_ids, entries, merits,
                          n_bootstrap, seed, split_label):
    """How much does an Oishi-Tomiyasu-style pre-filter flatter each merit (Q5)?

    Reported on the rank metrics, which need no threshold and so need nothing selected on
    `fom-train`. `ceiling_rescorer` is the column that matters most: a pre-filter that lifts
    top-10 by deleting correct candidates has made the problem easier by making it unsolvable,
    and that shows up as the ceiling falling rather than as the rate rising.
    """
    rows = []
    for name in merits:
        for label, filtered in (('unfiltered', False), ('O-T pre-filter', True)):
            needed = (name, 'M_tilde', 'M_rev', 'N_cal')

            def shards(filtered=filtered, needed=needed):
                for frame in bundle_frames(benchmark_dir, feature_dir, bundles, keep_entry_ids,
                                           needed):
                    yield frame.loc[prefilter_mask(frame)] if filtered else frame

            result = FomMetrics.evaluate(
                shards(), score=name, higher_is_better=True, threshold=None, entries=entries,
                strata=(), split=split_label, n_bootstrap=n_bootstrap, seed=seed,
                )
            row = result.aggregate.iloc[0]
            rows.append({
                'merit': name, 'pool': label, 'n_entries': row['n_entries'],
                'top1': row['top1'], 'top10': row['top10'], 'mrr': row['mrr'],
                'ceiling_rescorer': row['ceiling_rescorer'],
                })
    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(index='merit', columns='pool',
                             values=['top10', 'ceiling_rescorer'])
    wide.columns = [f'{metric}_{pool}' for metric, pool in wide.columns]
    wide['top10_flattery'] = wide['top10_O-T pre-filter'] - wide['top10_unfiltered']
    wide['ceiling_cost'] = (wide['ceiling_rescorer_O-T pre-filter']
                            - wide['ceiling_rescorer_unfiltered'])
    return frame, wide.reset_index()


def complementarity(per_entry_path, metric='top10'):
    """Pairwise 'A finds it, B does not', and the union oracle over the whole zoo.

    The union oracle is the ceiling on any combiner that only ever picks one of the zoo's
    orderings, which is what S08 needs to know before it starts.
    """
    frame = pd.read_parquet(per_entry_path)
    wide = frame.pivot_table(index=['entry_id', 'condition_bundle'], columns='merit',
                             values=metric, aggfunc='first')
    wide = wide.fillna(False).astype(bool)
    merits = list(wide.columns)
    matrix = pd.DataFrame(index=merits, columns=merits, dtype=float)
    for a in merits:
        for b in merits:
            matrix.loc[a, b] = float((wide[a] & ~wide[b]).mean())
    union = float(wide.any(axis=1).mean())
    best = wide.mean().max()
    return matrix, union, float(best), wide


def c0_singularity(feature_dir, merits):
    """Which merits inherit M20's zero-residual divergence on the control bundle (F-054)."""
    path = Path(feature_dir)/'features_error0_cont0.parquet'
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    rows = []
    for name in merits:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        rows.append({
            'merit': name,
            'n': int(values.size),
            'n_non_finite': int((~finite).sum()),
            'frac_above_1e9': float((np.abs(values[finite]) > 1e9).mean()) if finite.any() else np.nan,
            'max_finite_abs': float(np.max(np.abs(values[finite]))) if finite.any() else np.nan,
            'p99_abs': float(np.percentile(np.abs(values[finite]), 99)) if finite.any() else np.nan,
            })
    frame = pd.DataFrame(rows)
    return frame.sort_values('frac_above_1e9', ascending=False) if frame.shape[0] else frame


def write_figures(null_frame, vcrit, matrix, geometry, artifact_dir, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- Figure 1: why the cross-lattice null is not measurable on this pool ----------------
    #
    # This started as "does Wu's 1.82-to-1.00 inflation reproduce?" and the honest answer is that
    # the question cannot be asked here (F-068), so the figure shows the reason rather than the
    # apparent answer. Left: where each lattice's null sits *relative to the prune boundary*, with
    # the p10-to-p90 span drawn, so a distribution truncated at the cut is visible as such.
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    frame = null_frame.dropna(subset=['M20_median']).sort_values('M20_median')
    positions = np.arange(frame.shape[0])
    # "Pinned" means the tenth percentile sits *at* the cut, i.e. the lower tail was removed.
    # A p10 well below 5 means the opposite: that lattice kept its tail, because extinction-group
    # assignment moved its M20 down after the pre-assignment prune had already let it through.
    pinned = (frame['M20_p10'] >= 4.95).to_numpy()
    axes[0].hlines(positions, frame['M20_p10'], frame['M20_p90'], color=BAND_GREY,
                   linewidth=5, alpha=0.8)
    axes[0].scatter(frame['M20_median'], positions, color=SERIES_BLUE, s=42, zorder=3,
                    label='median')
    axes[0].scatter(frame.loc[~pinned, 'M20_p10'], positions[~pinned],
                    color=TEXT_SECONDARY, s=18, zorder=3, marker='|', linewidths=1.5,
                    label='p10, tail retained')
    axes[0].scatter(frame.loc[pinned, 'M20_p10'], positions[pinned],
                    color=TARGET_RED, s=60, zorder=4, marker='|', linewidths=2.5,
                    label='p10 pinned at the cut: tail removed')
    axes[0].axvline(5.0, color=TARGET_RED, linestyle='--', linewidth=1.5)
    axes[0].annotate('prune_below_m20 = 5', xy=(5.0, positions[0] - 0.7), xytext=(4, 0),
                     textcoords='offset points', color=TARGET_RED, fontsize=8.5, va='bottom')
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(frame['bravais_lattice'], fontsize=9)
    axes[0].set_xlabel('M20 over incorrect candidates (p10 - median - p90)', color=TEXT_PRIMARY)
    axes[0].set_title('The null sits on the censoring boundary', color=TEXT_PRIMARY)
    # Room below the last row for the legend, rather than letting it sit on top of cF and hR.
    axes[0].set_ylim(-3.0, positions[-1] + 0.6)
    axes[0].legend(fontsize=7.5, frameon=False, loc='lower left', ncol=3,
                   columnspacing=1.4, handletextpad=0.4)

    both = null_frame.dropna(subset=['M20_ratio_to_aP', 'wu88_predicted_ratio'])
    cubic = both['lattice_system'] == 'cubic'
    axes[1].scatter(both.loc[~cubic, 'wu88_predicted_ratio'], both.loc[~cubic, 'M20_ratio_to_aP'],
                    color=SERIES_BLUE, s=38, zorder=3, label='20 peaks')
    axes[1].scatter(both.loc[cubic, 'wu88_predicted_ratio'], both.loc[cubic, 'M20_ratio_to_aP'],
                    color=TARGET_RED, s=38, zorder=3, marker='s',
                    label='10 peaks - a different statistic')
    limit = [0.9, max(2.0, float(both[['wu88_predicted_ratio', 'M20_ratio_to_aP']].max().max()))]
    axes[1].plot(limit, limit, color=BAND_GREY, linestyle=':', linewidth=1.5, zorder=1)
    axes[1].annotate('Wu 1988 exactly', xy=(limit[1], limit[1]), xytext=(-6, 6),
                     textcoords='offset points', ha='right', fontsize=8, color=TEXT_SECONDARY)
    for row in both.itertuples():
        axes[1].annotate(row.bravais_lattice,
                         xy=(row.wu88_predicted_ratio, row.M20_ratio_to_aP), xytext=(5, -3),
                         textcoords='offset points', fontsize=8, color=TEXT_SECONDARY)
    axes[1].set_xlabel('Wu 1988 Table 1 predicted M20/M-prime-20', color=TEXT_PRIMARY)
    axes[1].set_ylabel('measured median, relative to aP', color=TEXT_PRIMARY)
    axes[1].set_title('Below the line everywhere - but see the caption', color=TEXT_PRIMARY)
    axes[1].legend(fontsize=7.5, frameon=False, loc='upper left')

    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color(BAND_GREY)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.grid(alpha=0.25)
    figure.supxlabel(
        'Incorrect candidates only, fom-dev, six evaluable bundles. THIS DOES NOT REFUTE WU\n'
        '(F-068): the pool is censored at M20 >= 5 and the cut bites unequally, because the prune\n'
        'tests the pre-extinction-group M20 while the stored value is post-assignment (F-049). The\n'
        'low-symmetry lattices have their lower tail removed (red p10 marks) and the high-symmetry\n'
        'ones do not, which manufactures the ordering on the right. Cubic is scored on ten peaks.',
        fontsize=8.5, color=TEXT_SECONDARY)
    figure.savefig(Path(artifact_dir)/f'{tag}_cross_lattice.png', dpi=200, facecolor='white')
    plt.close(figure)

    # ---- Figure 2: is the comparison meaningful at all? -------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for scope, colour, marker in (('all reachable', SERIES_BLUE, 'o'),
                                  ('hard stratum', TARGET_RED, 's')):
        frame = vcrit.loc[vcrit['scope'] == scope]
        if not frame.shape[0]:
            continue
        axes[0].plot(frame['g_min'], frame['frac_correct_above_Vcrit'], color=colour,
                     marker=marker, markersize=5, linewidth=2, label=f'correct cell, {scope}')
        axes[0].plot(frame['g_min'], frame['frac_wrong_winner_above_Vcrit'], color=colour,
                     marker=marker, markersize=5, linewidth=2, linestyle='--',
                     label=f'wrong winner, {scope}')
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\bar{g}_{min}$ (unchosen: Q14)', color=TEXT_PRIMARY)
    axes[0].set_ylabel(r'fraction above $V_{crit}$', color=TEXT_PRIMARY)
    axes[0].set_title('Werner: above $V_{crit}$ the merit reports precision', color=TEXT_PRIMARY)
    axes[0].legend(fontsize=7.5, frameon=False)

    reachable = geometry.loc[geometry['has_correct'].astype(bool)]
    beaten = reachable.loc[~reachable['top_is_correct'].astype(bool)]
    pairs = beaten[['correct_M20', 'top_M20']].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].scatter(pairs['correct_M20'][::7], pairs['top_M20'][::7], s=5, alpha=0.18,
                    color=SERIES_BLUE, linewidths=0)
    top_limit = float(np.nanpercentile(pairs.to_numpy(), 99))
    axes[1].plot([0, top_limit], [0, top_limit], color=BAND_GREY, linestyle=':', linewidth=1.5)
    axes[1].axhline(10.0, color=TARGET_RED, linestyle='--', linewidth=1.2)
    axes[1].axvline(10.0, color=TARGET_RED, linestyle='--', linewidth=1.2)
    axes[1].set_xlim(0, top_limit)
    axes[1].set_ylim(0, top_limit)
    axes[1].set_xlabel('M20 of the correct cell', color=TEXT_PRIMARY)
    axes[1].set_ylabel('M20 of the cell that beat it', color=TEXT_PRIMARY)
    axes[1].set_title('Everything below 10, and the wrong one higher', color=TEXT_PRIMARY)

    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color(BAND_GREY)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        axis.grid(alpha=0.25)
    figure.supxlabel(
        r'$V/V_{crit}$ is stored at $\bar{g}_{min}=1$ and is linear in it, so the sweep is exact '
        'rather than interpolated. Right: entries with a reachable solution whose top-ranked '
        'candidate is nonetheless wrong, every seventh point drawn.',
        fontsize=8.5, color=TEXT_SECONDARY)
    figure.savefig(Path(artifact_dir)/f'{tag}_vcrit.png', dpi=200, facecolor='white')
    plt.close(figure)

    # ---- Figure 3: complementarity, which bounds what a combiner can do ---------------------
    if matrix is None or not matrix.shape[0]:
        return
    order = matrix.mean(axis=1).sort_values(ascending=False).index
    ordered = matrix.loc[order, order]
    figure, axis = plt.subplots(figsize=(9.5, 8.2), constrained_layout=True)
    image = axis.imshow(ordered.to_numpy(), cmap='magma_r', vmin=0.0)
    axis.set_xticks(np.arange(len(order)))
    axis.set_yticks(np.arange(len(order)))
    axis.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
    axis.set_yticklabels(order, fontsize=8)
    axis.set_xlabel('...when this one does not', color=TEXT_PRIMARY)
    axis.set_ylabel('this merit ranks the correct cell top-10...', color=TEXT_PRIMARY)
    axis.set_title('What each merit finds that another misses', color=TEXT_PRIMARY)
    figure.colorbar(image, ax=axis, shrink=0.8, label='fraction of entries')
    figure.supxlabel(
        'fom-dev, six condition bundles. Cell (row a, column b) is the fraction of entries where a '
        'ranks the correct cell top-10 and b does not; the diagonal is zero by construction.\n'
        'A pale ROW means that merit never succeeds where another fails -- it is dominated and '
        'adds nothing to a combiner. A pale COLUMN means the opposite: nothing rescues the\n'
        'entries that merit loses, so it subsumes the others. M_sym has the brightest row and one '
        'of the palest columns, which is what being the best single merit looks like.',
        fontsize=8.5, color=TEXT_SECONDARY)
    figure.savefig(Path(artifact_dir)/f'{tag}_complementarity.png', dpi=200, facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description='Why one figure of merit beats another (S06 sections 3c, 3d and 4).'
        )
    parser.add_argument('--benchmark-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--feature-dir',
                        default=os.path.join(BASE, 'mlindex', 'data', 'fom_features'))
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--bundles', nargs='+', default=list(EVALUABLE_BUNDLES))
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--per-entry', default=None,
                        help='S06_zoo_per_entry.parquet from run_fom_zoo_eval.py.')
    parser.add_argument('--n-bootstrap', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--skip-bands', action='store_true')
    parser.add_argument('--skip-prefilter', action='store_true')
    parser.add_argument('--reuse-geometry', action='store_true',
                        help='Load the geometry table from a previous run instead of walking '
                             'the pool again. It depends only on the pool and M20, so it is '
                             'stable across reruns of everything downstream of it.')
    parser.add_argument('--tag', default='S06_explain')
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    per_entry_path = Path(args.per_entry or artifact_dir/'S06_zoo_per_entry.parquet')

    entries = FomBenchmark.load_entries(args.benchmark_dir)
    dev_ids = set(entries.loc[entries['split'] == args.report_split, 'entry_id'])

    geometry_path = artifact_dir/f'{args.tag}_geometry.parquet'
    if args.reuse_geometry and geometry_path.exists():
        geometry = pd.read_parquet(geometry_path)
        print(f'pool geometry: reused {geometry.shape[0]:,} rows from {geometry_path.name}')
    else:
        print('pool geometry')
        geometry, mismatches = pool_geometry(
            args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries,
            )
        print(f'  {geometry.shape[0]:,} (entry, condition) rows; '
              f'{int(geometry["has_correct"].sum()):,} reachable; '
              f'{mismatches} top-candidate disagreements with reduce_pool')
        geometry.to_parquet(geometry_path, index=False)

    print('\nV/V_crit sweep over g_min')
    vcrit = vcrit_sweep(geometry)
    print(vcrit.to_string(index=False))
    vcrit.to_csv(artifact_dir/f'{args.tag}_vcrit.csv', index=False, encoding='utf-8')

    print('\ncross-lattice null over incorrect candidates')
    null_frame = cross_lattice_null(
        args.benchmark_dir, args.feature_dir, args.bundles, dev_ids,
        ('M20', 'M_1', 'M_wu', 'null_tail_nll'),
        )
    print(null_frame[['bravais_lattice', 'M20_median', 'M20_ratio_to_aP',
                      'wu88_predicted_ratio']].to_string(index=False))
    null_frame.to_csv(artifact_dir/f'{args.tag}_cross_lattice.csv', index=False,
                      encoding='utf-8')

    print('\nover-prediction')
    over = over_prediction(geometry)
    print(over.to_string(index=False))
    over.to_csv(artifact_dir/f'{args.tag}_over_prediction.csv', index=False, encoding='utf-8')

    matrix = pd.DataFrame()
    if per_entry_path.exists():
        print('\ncomplementarity')
        matrix, union, best, wide = complementarity(per_entry_path)
        print(f'  union oracle over the zoo (top-10): {union:.4f}; best single merit: {best:.4f}')
        matrix.to_csv(artifact_dir/f'{args.tag}_complementarity.csv', encoding='utf-8')
        pd.DataFrame([{'union_oracle_top10': union, 'best_single_merit_top10': best,
                       'n_merits': matrix.shape[0], 'n_entries': int(wide.shape[0])}]
                     ).to_csv(artifact_dir/f'{args.tag}_union_oracle.csv', index=False,
                              encoding='utf-8')
    else:
        print(f'\ncomplementarity skipped: {per_entry_path} not found '
              '(run run_fom_zoo_eval.py first)')

    print('\nC0 singularity (F-054)')
    singular = c0_singularity(args.feature_dir, EXPLAINED + ('M_tilde', 'M_info_clipped', 'bic'))
    if singular.shape[0]:
        print(singular.to_string(index=False))
        singular.to_csv(artifact_dir/f'{args.tag}_c0_singularity.csv', index=False,
                        encoding='utf-8')
    else:
        print('  skipped: no C0 feature matrix. Build one with '
              '--bundles error0_cont0 --limit-entries 300')

    if not args.skip_prefilter:
        print('\npre-filter sensitivity (Q5: measured once, not adopted)')
        long_frame, wide = prefilter_sensitivity(
            args.benchmark_dir, args.feature_dir, args.bundles, dev_ids, entries,
            ('M20', 'M_1', 'M_wu', 'null_tail_nll'), args.n_bootstrap, args.seed,
            args.report_split,
            )
        print(wide.round(4).to_string(index=False))
        long_frame.to_csv(artifact_dir/f'{args.tag}_prefilter.csv', index=False,
                          encoding='utf-8')
        wide.to_csv(artifact_dir/f'{args.tag}_prefilter_summary.csv', index=False,
                    encoding='utf-8')

    bands = pd.DataFrame()
    if not args.skip_bands:
        print('\nper-merit discrimination inside and outside the domain (V/V_crit)')
        band_rows = []
        quantiles = pd.qcut(geometry['top_V_over_Vcrit'], VCRIT_BANDS,
                            labels=[f'V/Vcrit Q{i + 1}' for i in range(VCRIT_BANDS)])
        for band in quantiles.cat.categories:
            band_ids = set(geometry.loc[quantiles == band, 'entry_id']) & dev_ids
            if not band_ids:
                print(f'  {band}: empty, skipped')
                continue
            for name in EXPLAINED:
                result = evaluate_merit(
                    args.benchmark_dir, args.feature_dir, args.bundles, band_ids,
                    entries, name, True, None, (), (), args.n_bootstrap, args.seed,
                    args.report_split,
                    )
                row = result.aggregate.iloc[0]
                band_rows.append({
                    'band': band, 'merit': name, 'n_entries': row['n_entries'],
                    'top1': row['top1'], 'top10': row['top10'], 'mrr': row['mrr'],
                    'ceiling_rescorer': row['ceiling_rescorer'],
                    })
                print(f'  {band:14s} {name:16s} top10 {row["top10"]:.4f}  '
                      f'ceiling {row["ceiling_rescorer"]:.4f}')
        bands = pd.DataFrame(band_rows)
        bands.to_csv(artifact_dir/f'{args.tag}_vcrit_bands.csv', index=False, encoding='utf-8')

    print('\nfigures')
    write_figures(null_frame, vcrit, matrix, geometry, artifact_dir, args.tag)
    print(f'  wrote {args.tag}_{{cross_lattice,vcrit,complementarity}}.png')
    print(f'\ncommit {commit_hash()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
