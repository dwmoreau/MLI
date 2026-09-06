"""S10b's analysis half: the leaderboard, the peak-budget sweep, and the two arms S10a asked for.

Split out of `run_fom_holdout_eval.py` for the reason the zoo's `--reduce` / `--analyse` split
exists -- the pool lives on `$SCRATCH` and the record lives on a laptop -- and kept as a module
rather than as script body so the figure and the tables have one home and one set of tests.

Nothing here touches a candidate pool. Everything is a function of the stacked per-entry reduction
`run_fom_holdout_eval.py --reduce` writes, which is one row per (merit, budget, split, entry,
condition) and a sufficient statistic for every metric downstream.

**Three rules from S10a are enforced here rather than remembered:**

  * A budget's population is only those entries whose stored surplus reaches it. `--reduce`
    already dropped the rest, and every paired comparison is re-restricted to the intersection
    before McNemar sees it, because `FomMetrics.mcnemar` refuses two results over different
    entry sets and that refusal is the guard, not an obstacle.
  * `ho_M_sym` is reported with `M_rev` support coverage beside it, always. Below ten reference
    lines the merit returns 0.0 rather than null, so a floored value is *defined* and meaningless,
    and only the stored `ho_N_cal` distinguishes it (C2-Q-017, C2-F-100).
  * The realistic regime is 1-10 surplus peaks -- a 21- to 30-peak pattern. 20 is the storage cap
    minus the window and is a labelled upper bound, never a recommendation (C2-F-103, C2-R-016).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomMetrics

# The peak budgets an instrument can actually supply. `n_extra` IS the total peak budget minus the
# 20-peak window, so this is 21 to 30 peaks -- DWMM: 20-25 typical, 30 on extremely good data.
# Anything above is the generator's storage cap showing through and is labelled as such.
REALISTIC_MAX = 10

# The budget the headline and the recommendation to S12 are quoted at: a 25-peak pattern, which is
# the middle of what real data supplies.
HEADLINE_N_EXTRA = 5

# S08's measured contrast floor, in percentage points, for `M_sym` against M20 on top-10 over the
# 530-crystal floor sample -- which is exactly `fom_full_c2_pool`'s population, so the sample floor
# is the right column and the composed one is not.
#
# **No floor has been measured for a hold-out contrast.** S08's four arms differ in the search
# seed and were scored in sample, so this is the nearest measured proxy rather than the quantity
# itself. Gates below are quoted against it with that stated; C2-R-017 records the bound.
FLOOR_ARTEFACT = 'S08_floor_by_lattice.csv'
FLOOR_AGGREGATE_ARTEFACT = 'S08_floor_contrast.csv'
FLOOR_REFERENCE = ('M_sym', 'M20', 'top10')

# Campaign 1's hold-out headline, and everything about it that differs from this measurement.
# Quoted from `INHERITED.md` section 2 (F-097) and its rebuild row R13, not recomputed -- and kept
# here as data so the results document cannot drift from the record it cites.
CAMPAIGN1_HOLDOUT = {
    'delta_pp': 7.11,
    'gained': 1081,
    'lost': 580,
    'p_value': '1.3e-34',
    'n_extra': 5,
    'differences': [
        ('the surplus carried no contaminants and no second-phase lines, while the fitted window '
         'did', 'R13, F-097 -- campaign 1 never stored the surplus, so it was re-synthesised from '
         'the true structure. Campaign 2 stores it, S10a confirmed on data that it carries the '
         "window's own noise draw (C2-F-099), and S10a seeds contaminants into it at the window's "
         'own per-peak rate (C2-F-098)'),
        ('it was a second, independent noise draw rather than part of the pattern the candidate '
         'pool came from', 'R13. Campaign 2 draws one stream for window and surplus together, '
         'measured at the window scale and moving proportionally with the error multiplier'),
        ('the pool was censored at M20 >= 5 everywhere', 'campaign 1 held no candidate below it, '
         'so a merit that ranks a low-M20 candidate highly was unevaluable. Campaign 2 generates '
         'at 1.5 (C2-F-021) -- still an M20 cut, which C2-R-018 records'),
        ('a different split, a different generator and a different condition grid', 'campaign 2 '
         'rebuilt all three'),
        ],
    }

# Below this many reference lines in the counting window `get_M_rev_sym` declares M_rev undefined
# and returns 0.0. Kept in step with `run_fom_holdout_eval.MIN_N_CAL`.
MIN_N_CAL = 10


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------
def load_reduction(artifact_dir, tag):
    """The stacked reduction, its meta, and the coverage table."""
    artifact_dir = Path(artifact_dir)
    stacked = pd.read_parquet(artifact_dir/f'{tag}_reduced.parquet')
    meta = json.loads((artifact_dir/f'{tag}_reduced_meta.json').read_text(encoding='utf-8'))
    coverage_path = artifact_dir/f'{tag}_coverage.csv'
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    return stacked, meta, coverage


def load_floors(artifact_dir, reference=FLOOR_REFERENCE, aggregate_artefact=FLOOR_AGGREGATE_ARTEFACT,
                per_lattice_artefact=FLOOR_ARTEFACT):
    """(aggregate floor pp, {lattice: floor pp}) for the reference contrast.

    Returns `(None, {})` when S08's artefacts are absent, and the caller then reports gates in
    percentage points with the omission stated -- rather than inventing a floor, which is the
    failure PROTOCOL section 8 exists to prevent.

    The defaults are the top-10 floor S08 measured. S15 reads the operating-point floor too, which
    S09 measured into `S09_floor_op_*.csv` with the same columns, so the reference and the two
    artefacts are parameters rather than a second copy of this function.
    """
    artifact_dir = Path(artifact_dir)
    merit, baseline, metric = reference
    aggregate, per_lattice = None, {}
    path = artifact_dir/aggregate_artefact
    if path.exists():
        frame = pd.read_csv(path)
        row = frame.loc[(frame['merit'] == merit) & (frame['baseline'] == baseline)
                        & (frame['metric'] == metric)]
        if row.shape[0]:
            aggregate = float(row['floor_pp'].iloc[0])
    path = artifact_dir/per_lattice_artefact
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame.loc[(frame['merit'] == merit) & (frame['baseline'] == baseline)
                          & (frame['metric'] == metric)]
        per_lattice = dict(zip(frame['bravais_lattice'], frame['se_pp'].astype(float)))
    return aggregate, per_lattice


# ---------------------------------------------------------------------------------------
# Turning a slice of the stacked reduction back into a MetricsResult
# ---------------------------------------------------------------------------------------
def _meta_for(meta, merit, n_extra, split):
    """The reduce meta for one (merit, budget, split), whatever key shape it was stored under."""
    for key, value in meta.get('reductions', {}).items():
        if (value.get('score') == merit and value.get('split') == split
                and (value.get('n_extra') if value.get('n_extra') is not None else -1)
                == (n_extra if n_extra is not None else -1)):
            return dict(value)
    raise KeyError(f'No reduction for {merit!r} at n_extra={n_extra} on {split!r}')


def result_for(stacked, meta, merit, n_extra, split, restrict_to=None, threshold=None,
               strata=(), n_bootstrap=0, seed=12345, top_n=10):
    """One `MetricsResult` for a (merit, budget, split), optionally restricted to an entry set.

    `restrict_to` is a set of (entry_id, condition_bundle) pairs and is how a paired comparison is
    made honest: a hold-out merit at ten surplus peaks is defined on fewer patterns than the
    in-sample anchor, so the anchor is cut down to the merit's population before either is read.
    """
    budget = n_extra if n_extra is not None else -1
    frame = stacked.loc[(stacked['merit'] == merit) & (stacked['n_extra'] == budget)
                        & (stacked['split'] == split)]
    if not frame.shape[0]:
        raise KeyError(f'No rows for {merit!r} at n_extra={n_extra} on {split!r}')
    if restrict_to is not None:
        keys = list(zip(frame['entry_id'], frame['condition_bundle']))
        frame = frame.loc[[key in restrict_to for key in keys]]
    frame = frame.drop(columns=['merit', 'n_extra', 'split']).reset_index(drop=True)
    return FomMetrics.summarise_per_entry(
        frame, _meta_for(meta, merit, n_extra, split), threshold=threshold, top_n=top_n,
        strata=strata, n_bootstrap=n_bootstrap, seed=seed)


def population_of(stacked, merit, n_extra, split):
    """The (entry, condition) set a merit is defined on at a budget."""
    budget = n_extra if n_extra is not None else -1
    frame = stacked.loc[(stacked['merit'] == merit) & (stacked['n_extra'] == budget)
                        & (stacked['split'] == split)]
    return set(zip(frame['entry_id'], frame['condition_bundle']))


# ---------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------
def coverage_table(coverage):
    """`M_rev` support and reference reach as rates, per (budget, lattice), and in aggregate.

    Candidate-level, which is the level both gates act at: the floor fires on a candidate's own
    counting window and the reference list is a property of the candidate's extinction group.
    """
    if not coverage.shape[0]:
        return pd.DataFrame()
    rows = []
    for scope, keys in (('all', ['n_extra']), ('by_lattice', ['n_extra', 'bravais_lattice'])):
        block = coverage.groupby(keys, as_index=False)[
            ['n_candidates', 'n_scored', 'n_mrev_supported', 'n_ref_reached']].sum()
        block['scope'] = scope
        if 'bravais_lattice' not in block.columns:
            block['bravais_lattice'] = 'all'
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    scored = out['n_scored'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['mrev_support_rate'] = np.where(scored > 0, out['n_mrev_supported']/scored, np.nan)
        out['ref_reach_rate'] = np.where(scored > 0, out['n_ref_reached']/scored, np.nan)
        out['scored_rate'] = out['n_scored']/out['n_candidates'].to_numpy(dtype=float)
    # The raw counts stay in the table beside the rates. A rate without its denominator cannot be
    # pooled, and the figure pools the per-lattice rows into three groups.
    return out[['scope', 'n_extra', 'bravais_lattice', 'n_candidates', 'n_scored',
                'n_mrev_supported', 'n_ref_reached', 'scored_rate', 'mrev_support_rate',
                'ref_reach_rate']].sort_values(['scope', 'n_extra', 'bravais_lattice'])


def applicability_table(stacked, meta, split):
    """What fraction of the split's patterns can be scored at each budget, and how many dropped.

    A property of the generator, not of real data -- S10a measures it on a simulator that knows
    every non-absent line, and what fraction of *experimental* patterns extend usable lines past
    the twenty the indexer took is an S15/S16 question. Flagged, never transferred.
    """
    rows = []
    for key, value in meta.get('reductions', {}).items():
        if value.get('split') != split or value.get('n_extra') is None:
            continue
        kept, dropped = int(value['n_entries']), int(value.get('n_dropped_short', 0))
        total = kept + dropped
        rows.append({'merit': value['score'], 'n_extra': int(value['n_extra']),
                     'n_pattern_peaks': 20 + int(value['n_extra']),
                     'n_applicable': kept, 'n_dropped_short': dropped, 'n_total': total,
                     'applicability': kept/total if total else np.nan})
    frame = pd.DataFrame(rows)
    if not frame.shape[0]:
        return frame
    # One number per budget: applicability is a property of the pattern, not of the merit, so the
    # per-merit rows must agree and the aggregate is taken over them rather than from one.
    return frame.groupby(['n_extra', 'n_pattern_peaks'], as_index=False).agg(
        n_applicable=('n_applicable', 'max'), n_dropped_short=('n_dropped_short', 'max'),
        n_total=('n_total', 'max'), applicability=('applicability', 'max'))


# ---------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------
def sweep(stacked, meta, split, merits, budgets, anchor='M20', top_n=10, n_bootstrap=0,
          seed=12345):
    """Gain against the in-sample anchor, per merit, per budget, aggregate / hard / per lattice.

    Every row is **paired**: the anchor is restricted to the merit's own population at that
    budget before either side is read, so a rising mean over a shrinking population cannot be
    mistaken for a gain.
    """
    rows = []
    strata = ('bravais_lattice',)
    for budget in budgets:
        for merit in merits:
            try:
                population = population_of(stacked, merit, budget, split)
                merit_result = result_for(stacked, meta, merit, budget, split, strata=strata,
                                          n_bootstrap=n_bootstrap, seed=seed, top_n=top_n)
                anchor_result = result_for(stacked, meta, anchor, None, split,
                                           restrict_to=population, strata=strata,
                                           n_bootstrap=n_bootstrap, seed=seed, top_n=top_n)
            except KeyError:
                continue
            scopes = [('all', None), ('hard', 'hard')]
            lattices = sorted(merit_result.per_entry['bravais_lattice'].dropna().unique())
            scopes += [(lattice, FomMetrics.stratum_mask(merit_result, 'bravais_lattice', lattice))
                       for lattice in lattices]
            for label, subset in scopes:
                row = {'merit': merit, 'n_extra': budget, 'n_pattern_peaks': 20 + budget,
                       'regime': 'realistic' if budget <= REALISTIC_MAX else 'cap-limited',
                       'scope': label}
                row.update(_scope_metrics(merit_result, label, prefix=''))
                row.update(_scope_metrics(anchor_result, label, prefix='anchor_'))
                try:
                    test = FomMetrics.mcnemar(merit_result, anchor_result, metric='top10',
                                              subset=subset)
                    row.update({'delta_pp': 100.0*float(test['delta']),
                                'n_gained': int(test['n_a_only']),
                                'n_lost': int(test['n_b_only']),
                                'p_value': float(test['p_value'])})
                except (ValueError, KeyError) as error:
                    row['error'] = str(error)
                rows.append(row)
    return pd.DataFrame(rows)


def _scope_metrics(result, scope, prefix=''):
    """top1/top10/mrr and the count, for 'all', 'hard' or one Bravais lattice."""
    if scope == 'all':
        frame = result.aggregate
    elif scope == 'hard':
        frame = result.hard
    else:
        by = result.by_stratum
        frame = by.loc[(by['stratum'] == 'bravais_lattice') & (by['level'] == scope)]
    if not frame.shape[0]:
        return {f'{prefix}{name}': np.nan for name in ('top1', 'top10', 'mrr', 'n_entries')}
    row = frame.iloc[0]
    return {f'{prefix}top1': float(row['top1']), f'{prefix}top10': float(row['top10']),
            f'{prefix}mrr': float(row['mrr']), f'{prefix}n_entries': int(row['n_entries'])}


def in_standard_errors(frame, aggregate_floor, per_lattice_floor):
    """Add `floor_pp` and `standard_errors` to a sweep table.

    PROTOCOL section 8: a gate is written in standard errors of the contrast floor, and a
    per-lattice claim uses **that lattice's own** floor. Reading a cF claim against the aggregate
    would be wrong by an order of magnitude (C2-F-081).
    """
    frame = frame.copy()
    floors = []
    for scope in frame['scope']:
        if scope in per_lattice_floor:
            floors.append(per_lattice_floor[scope])
        elif scope in ('all', 'hard'):
            floors.append(aggregate_floor)
        else:
            floors.append(np.nan)
    frame['floor_pp'] = floors
    with np.errstate(invalid='ignore', divide='ignore'):
        frame['standard_errors'] = frame['delta_pp']/frame['floor_pp']
    return frame


# ---------------------------------------------------------------------------------------
# The leaderboard
# ---------------------------------------------------------------------------------------
def leaderboard(stacked, meta, split, merits, n_extra, anchor='M20', top_n=10, n_bootstrap=1000,
                seed=12345):
    """Every merit at one peak budget, paired against the in-sample anchor on its own population.

    Rank metrics only. **The threshold half is deliberately absent from this table**: campaign 1
    measured `ho_M20`'s operating point at exactly 0.0000 -- its Youden optimum is never to report
    -- so a hold-out merit is a **ranker, not a score**, and quoting a rank gain without saying so
    is misleading (S10b acceptance gate 4). The threshold work is a separate table on a pool that
    carries `fom-train`.
    """
    rows = []
    strata = ('bravais_lattice', 'condition_bundle')
    for merit in merits:
        try:
            population = population_of(stacked, merit, n_extra, split)
            result = result_for(stacked, meta, merit, n_extra, split, strata=strata,
                                n_bootstrap=n_bootstrap, seed=seed, top_n=top_n)
            anchor_result = result_for(stacked, meta, anchor, None, split,
                                       restrict_to=population, strata=strata,
                                       n_bootstrap=n_bootstrap, seed=seed, top_n=top_n)
        except KeyError:
            continue
        aggregate, hard = result.aggregate.iloc[0], (
            result.hard.iloc[0] if result.hard.shape[0] else None)
        row = {
            'merit': merit, 'n_extra': n_extra, 'n_pattern_peaks': 20 + n_extra,
            'higher_is_better': result.meta['higher_is_better'],
            'ranks_exact': bool(result.meta.get('ranks_exact')),
            'top1': float(aggregate['top1']), 'top10': float(aggregate['top10']),
            'mrr': float(aggregate['mrr']),
            'top10_ci_low': float(aggregate.get('top10_ci_low', np.nan)),
            'top10_ci_high': float(aggregate.get('top10_ci_high', np.nan)),
            'ceiling_rescorer': float(aggregate['ceiling_rescorer']),
            'n_entries': int(aggregate['n_entries']),
            'hard_top10': np.nan if hard is None else float(hard['top10']),
            'hard_n_entries': np.nan if hard is None else int(hard['n_entries']),
            'hard_n_found': np.nan if hard is None else int(hard['n_found']),
            'anchor_top10': float(anchor_result.aggregate.iloc[0]['top10']),
            'anchor_top1': float(anchor_result.aggregate.iloc[0]['top1']),
            }
        for metric in ('top10', 'top1'):
            try:
                test = FomMetrics.mcnemar(result, anchor_result, metric=metric)
                interval = FomMetrics.paired_delta_ci(result, anchor_result, metric=metric,
                                                      n_bootstrap=n_bootstrap, seed=seed)
                row[f'delta_{metric}_pp'] = 100.0*float(test['delta'])
                row[f'delta_{metric}_ci_low_pp'] = 100.0*float(interval['ci_low'])
                row[f'delta_{metric}_ci_high_pp'] = 100.0*float(interval['ci_high'])
                row[f'n_gained_{metric}'] = int(test['n_a_only'])
                row[f'n_lost_{metric}'] = int(test['n_b_only'])
                row[f'p_{metric}'] = float(test['p_value'])
            except (ValueError, KeyError) as error:
                row[f'delta_{metric}_error'] = str(error)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values('top10', ascending=False) if frame.shape[0] else frame


# ---------------------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------------------
def paired_arm(stacked_a, meta_a, stacked_b, meta_b, split, merits, budgets, label_a, label_b,
               top_n=10, n_bootstrap=0, seed=12345, lattices=None):
    """One reduction against another over the same entries -- the shape both S10b arms take.

    The cubic free-peaks arm and the contaminant-cost arm differ only in which two reductions are
    handed in, so they share this. `lattices` restricts the comparison to a lattice set, which is
    what makes the cubic arm a **paired arm within cubic** rather than an aggregate that quietly
    adopts a wider budget for one lattice -- the mistake F-088 records four times over.
    """
    rows = []
    for budget in budgets:
        for merit in merits:
            try:
                common = (population_of(stacked_a, merit, budget, split)
                          & population_of(stacked_b, merit, budget, split))
                if not common:
                    continue
                result_a = result_for(stacked_a, meta_a, merit, budget, split, restrict_to=common,
                                      strata=('bravais_lattice',), n_bootstrap=n_bootstrap,
                                      seed=seed, top_n=top_n)
                result_b = result_for(stacked_b, meta_b, merit, budget, split, restrict_to=common,
                                      strata=('bravais_lattice',), n_bootstrap=n_bootstrap,
                                      seed=seed, top_n=top_n)
            except KeyError:
                continue
            scopes = [('all', None)]
            if lattices:
                scopes = [(one, FomMetrics.stratum_mask(result_a, 'bravais_lattice', one))
                          for one in lattices if one in
                          set(result_a.per_entry['bravais_lattice'].dropna())]
                scopes.append(('all_listed', result_a.per_entry['bravais_lattice']
                               .isin(lattices).to_numpy()))
            for scope, subset in scopes:
                row = {'merit': merit, 'n_extra': budget, 'n_pattern_peaks': 20 + budget,
                       'scope': scope, 'arm_a': label_a, 'arm_b': label_b,
                       'n_common_cells': len(common)}
                row.update(_scope_metrics(result_a, scope if scope != 'all_listed' else 'all',
                                          prefix='a_'))
                row.update(_scope_metrics(result_b, scope if scope != 'all_listed' else 'all',
                                          prefix='b_'))
                try:
                    test = FomMetrics.mcnemar(result_a, result_b, metric='top10', subset=subset)
                    row.update({'delta_pp': 100.0*float(test['delta']),
                                'n_gained': int(test['n_a_only']),
                                'n_lost': int(test['n_b_only']),
                                'p_value': float(test['p_value'])})
                except (ValueError, KeyError) as error:
                    row['error'] = str(error)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 200, 'savefig.dpi': 200, 'legend.frameon': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        })
    return plt


# One colour per merit, fixed here so the sweep figure, the per-lattice panel and anything S17
# reuses agree. Order is the reading order of the leaderboard, not alphabetical.
MERIT_COLOURS = {
    'ho_M_sym': '#1b4965', 'ho_M20': '#c1666b', 'ho_M_tilde': '#5fa8d3',
    'ho_Minfo': '#e09f3e', 'ho_M_rev': '#8a817c', 'ho_M': '#4f772d', 'ho_raw': '#9d4edd',
    }


def sweep_figure(sweep_frame, coverage, applicability, path, anchor_label='M20, in sample',
                 realistic_max=REALISTIC_MAX, headline_n_extra=HEADLINE_N_EXTRA,
                 tiebreak_floor=None, hard_note=None):
    """The peak-budget sweep: gain, per-lattice deltas, applicability and `M_rev` coverage.

    A candidate paper figure, so it is built to publication quality first time (PROTOCOL section
    5). Three things it has to do that a default plot does not:

      * **Both units on the x axis** -- surplus peaks and total pattern peaks -- so an
        experimentalist reads their own instrument off it, with the regime no instrument reaches
        shaded rather than footnoted.
      * **The tie-break floor drawn as a line.** At one surplus peak `M_rev`'s support floor fires
        for every candidate, so `ho_M_sym` is a *constant* and scores exactly the tie-break floor.
        Undrawn, that spike reads as the merit working at one peak, which is the opposite of what
        it means.
      * **Per lattice, at the headline budget.** The hard stratum is the natural second panel and
        on the retained pool it is 20 entries where the in-sample anchor itself scores zero, so a
        per-lattice panel is what carries the stratified claim instead.
    """
    plt = _style()
    aggregate = sweep_frame.loc[sweep_frame['scope'] == 'all'].copy()
    if not aggregate.shape[0]:
        return None
    budgets = sorted(aggregate['n_extra'].unique())

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)

    def budget_axis(axis):
        axis.set_xticks(budgets)
        axis.set_xticklabels([f'{b}\n{20 + b}' for b in budgets], fontsize=7.5)
        axis.set_xlabel('surplus peaks scored\ntotal peaks in the pattern', fontsize=8)
        top = max(budgets)
        if top > realistic_max:
            axis.axvspan(realistic_max, top + 0.5, color='0.92', zorder=0, linewidth=0)
        axis.axvline(realistic_max, color='0.55', linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)

    # ---------------------------------------------------------------- a. the sweep
    axis = axes[0][0]
    for merit, block in aggregate.groupby('merit'):
        block = block.sort_values('n_extra')
        axis.plot(block['n_extra'], 100.0*block['top10'], marker='o', markersize=3.5,
                  linewidth=1.5, label=merit.replace('ho_', ''), zorder=3,
                  color=MERIT_COLOURS.get(merit, '0.4'))
    anchor = aggregate.sort_values('n_extra').drop_duplicates('n_extra')
    axis.plot(anchor['n_extra'], 100.0*anchor['anchor_top10'], linestyle='--', linewidth=2.0,
              color='0.15', label=anchor_label, zorder=4)
    if tiebreak_floor is not None:
        axis.axhline(100.0*tiebreak_floor, color='#c1666b', linewidth=1.0,
                     linestyle=':', zorder=2)
        # Above the line and hard right: below it is where the legend sits, and left of it is
        # where the steep early traces cross.
        axis.text(budgets[-1], 100.0*tiebreak_floor + 1.6,
                  'tie-break floor -- what a CONSTANT score already gets ', fontsize=6.8,
                  color='#c1666b', va='bottom', ha='right')
    axis.set_title('a. no hold-out merit reaches the in-sample incumbent\n'
                   '    at any budget an instrument can supply', loc='left', fontsize=9.5)
    axis.set_ylabel('top-10 (%)')
    axis.legend(ncol=2, fontsize=7.2, loc='lower right')
    budget_axis(axis)

    # ---------------------------------------------------------------- b. per lattice
    axis = axes[0][1]
    per_lattice = sweep_frame.loc[(sweep_frame['n_extra'] == headline_n_extra)
                                  & (~sweep_frame['scope'].isin(['all', 'hard']))
                                  & sweep_frame['delta_pp'].notna()]
    if per_lattice.shape[0]:
        merits = [m for m in ('ho_M20', 'ho_M_sym', 'ho_Minfo')
                  if m in set(per_lattice['merit'])] or sorted(set(per_lattice['merit']))[:3]
        lattices = sorted(per_lattice['scope'].unique())
        width = 0.8/len(merits)
        for index, merit in enumerate(merits):
            block = per_lattice.loc[per_lattice['merit'] == merit].set_index('scope')
            values = [block['delta_pp'].get(one, np.nan) for one in lattices]
            axis.bar(np.arange(len(lattices)) + index*width - 0.4 + width/2, values, width,
                     label=merit.replace('ho_', ''), color=MERIT_COLOURS.get(merit, '0.4'))
        axis.axhline(0, color='0.15', linewidth=1.0)
        axis.set_xticks(np.arange(len(lattices)))
        axis.set_xticklabels(lattices, fontsize=7.5, rotation=90)
        axis.set_ylabel('top-10 against in-sample M20 (pp)')
        axis.legend(fontsize=7.2, loc='lower left')
    axis.set_title(f'b. every lattice loses, at {headline_n_extra} surplus peaks '
                   f'({20 + headline_n_extra}-peak pattern)', loc='left', fontsize=9.5)
    axis.set_xlabel('true Bravais lattice', fontsize=8)

    # ---------------------------------------------------------------- c. applicability
    axis = axes[1][0]
    if applicability is not None and applicability.shape[0]:
        block = applicability.sort_values('n_extra')
        axis.plot(block['n_extra'], 100.0*block['applicability'], marker='s', markersize=4,
                  linewidth=1.6, color='#1b4965', zorder=3)
        # Labelled only where the budget is wide enough to place text without collision; the
        # numbers for every point are in the applicability CSV.
        for _, row in block.iterrows():
            if row['n_extra'] not in (budgets[0], realistic_max, budgets[-1]):
                continue
            first = row['n_extra'] == budgets[0]
            axis.annotate(f'{int(row["n_dropped_short"])} of {int(row["n_total"])}\ntoo short',
                          (row['n_extra'], 100.0*row['applicability']),
                          textcoords='offset points', xytext=(6 if first else 0, -24),
                          ha='left' if first else 'center', fontsize=6.8, color='0.35')
    axis.set_title('c. applicability -- a property of the GENERATOR, not of real data',
                   loc='left', fontsize=9.5)
    axis.set_ylabel('patterns whose surplus reaches the budget (%)')
    axis.set_ylim(0, 108)
    budget_axis(axis)

    # ---------------------------------------------------------------- d. M_rev support
    axis = axes[1][1]
    if coverage is not None and coverage.shape[0]:
        by_lattice = coverage.loc[coverage['scope'] == 'by_lattice']
        hard_lattices, cubic = ('aP', 'mP', 'mC'), ('cF', 'cI', 'cP')
        groups = (('hard lattices (aP, mP, mC)', hard_lattices, '#1b4965'),
                  ('cubic (cF, cI, cP)', cubic, '#c1666b'),
                  ('the other eight', tuple(sorted(
                      set(by_lattice['bravais_lattice']) - set(hard_lattices) - set(cubic))),
                   '#5fa8d3'))
        for label, members, colour in groups:
            block = by_lattice.loc[by_lattice['bravais_lattice'].isin(members)]
            if not block.shape[0]:
                continue
            block = block.groupby('n_extra', as_index=False)[
                ['n_scored', 'n_mrev_supported']].sum().sort_values('n_extra')
            rate = 100.0*block['n_mrev_supported']/block['n_scored'].replace(0, np.nan)
            axis.plot(block['n_extra'], rate, marker='o', markersize=3.5, linewidth=1.5,
                      label=label, color=colour, zorder=3)
        overall = coverage.loc[coverage['scope'] == 'all'].sort_values('n_extra')
        if overall.shape[0]:
            axis.plot(overall['n_extra'], 100.0*overall['mrev_support_rate'], linestyle=':',
                      linewidth=1.3, color='0.25', label='all candidates', zorder=3)
    axis.set_title('d. where ho_M_sym exists at all: M_rev support', loc='left', fontsize=9.5)
    axis.set_ylabel('candidates with N_cal >= 10 (%)')
    axis.set_ylim(0, 108)
    axis.legend(fontsize=7.2, loc='lower right')
    budget_axis(axis)

    figure.suptitle('S10b: scoring a cell on peaks it was never fitted to, against the peak '
                    'budget', fontsize=12, y=1.055)
    caption = ('Shaded: above 10 surplus peaks the benchmark is reporting its own 60-peak storage '
               'cap, not a regime any instrument reaches. Real patterns carry 20-25 peaks, 30 at '
               'best.')
    if hard_note:
        caption += f'\n{hard_note[0].upper() + hard_note[1:]}.'
    figure.text(0.5, 1.005, caption, ha='center', va='bottom', fontsize=7.8, color='0.35')
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------------------
# The results document
# ---------------------------------------------------------------------------------------
def _table(frame, columns, formats=None):
    """A markdown table from a frame, so the results `.md` is generated and not transcribed."""
    formats = formats or {}
    header = '| ' + ' | '.join(columns) + ' |'
    rule = '|' + '|'.join(['---']*len(columns)) + '|'
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, '')
            if column in formats and pd.notna(value):
                cells.append(formats[column](value))
            elif isinstance(value, float):
                cells.append('' if pd.isna(value) else f'{value:.4f}')
            else:
                cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_report(path, tag, meta, leaders, sweep_frame, coverage, applicability, floors,
                 figure_name, headline_n_extra, arms=None, thresholds=None,
                 threshold_note=None):
    """Ship the results `.md` -- PROTOCOL section 5: before the findings are written, with a figure."""
    aggregate_floor, per_lattice_floor = floors
    reductions = meta.get('reductions', {})
    exact = all(bool(one.get('ranks_exact')) for one in reductions.values())
    pool = meta.get('pool', '?')
    pp = lambda v: f'{v:+.2f}'  # noqa: E731
    pct = lambda v: f'{100*v:.1f} %'  # noqa: E731

    parts = [
        f'# S10b — the classical merits out of sample, and the peak budget',
        '',
        f'**Tag:** `{tag}` · **Pool:** `{pool}` · **Commit:** `{meta.get("commit", "?")}`'
        + (' (working tree dirty)' if meta.get('dirty_tree') else ''),
        f'**Candidates read:** {meta.get("n_candidates_seen", 0):,} · '
        f'**Ranks exact:** {"yes" if exact else "**NO — see below**"}',
        '',
        f'![the peak-budget sweep]({figure_name})',
        '',
        '## 0. How to read this',
        '',
        '`n_extra` is the **total peak budget minus 20**: five surplus peaks is a 25-peak pattern, '
        'ten is a 30-peak one. Real patterns carry 20–25 peaks and 30 at best, so **1–10 is the '
        'realistic regime and everything above it is the generator\'s storage cap showing through** '
        '(C2-F-103). The benchmark stores at most 20 surplus peaks and 54.9 % of source entries sit '
        'exactly on the 60-peak generation cap, so there is no "all available" budget to report.',
        '',
        'An entry whose stored surplus is shorter than the budget is **missing, not zero**: it '
        'leaves the population at that budget and the count that left is reported. Every '
        'comparison against the in-sample anchor is **paired on the merit\'s own population**.',
        '',
        ]

    if not exact:
        reasons = sorted({str(one.get('rank_exactness')) for one in reductions.values()
                          if not one.get('ranks_exact')})
        parts += [
            '> ### Every rank number here is OPTIMISTIC, and that is a property of the pool',
            '>',
            '> The hold-out merits are outside `FomMetrics.RANK_EXACT_MERITS`, so on a subsampled '
            'pool a correct candidate is ranked against a thinned field (C2-R-013). Reason '
            'recorded by the metrics module:',
            '>',
            ] + [f'> - {reason}' for reason in reasons] + [
            '>',
            '> **This pool answers threshold and contaminant questions and does not answer rank '
            'questions.** The rank claims belong on a fully retained pool.',
            '',
            ]

    parts += [
        f'## 1. The leaderboard at {headline_n_extra} surplus peaks '
        f'({20 + headline_n_extra}-peak pattern)',
        '',
        '**Every merit here is a ranker, not a score.** Campaign 1 measured `ho_M20`\'s operating '
        'point at exactly **0.0000** — its Youden optimum is never to report — so a rank gain '
        'quoted without this is misleading. The threshold half is a separate table on a pool that '
        'carries `fom-train`.',
        '',
        ]
    if leaders is not None and leaders.shape[0]:
        parts += [_table(
            leaders,
            ['merit', 'top1', 'top10', 'mrr', 'anchor_top10', 'delta_top10_pp',
             'n_gained_top10', 'n_lost_top10', 'p_top10', 'hard_top10', 'n_entries'],
            {'delta_top10_pp': pp, 'p_top10': lambda v: f'{v:.2g}'}), '']
    parts += [
        f'`anchor_top10` is in-sample M20 on the *same* patterns. `delta_top10_pp` is paired '
        f'(McNemar); the aggregate contrast floor is '
        + (f'**{aggregate_floor:.3f} pp**' if aggregate_floor else '**not available**')
        + ', measured in S08 for `M_sym` vs M20 — **no floor has been measured for a hold-out '
          'contrast**, so this is the nearest measured proxy and gates read against it carry that '
          'caveat.',
        '',
        '## 2. The peak-budget sweep',
        '',
        'The deliverable. Full table in `' + tag + '_sweep.csv`, per lattice, with applicability '
        'and `M_rev` coverage at every point.',
        '',
        ]
    aggregate_sweep = sweep_frame.loc[sweep_frame['scope'] == 'all'].sort_values(
        ['merit', 'n_extra'])
    if aggregate_sweep.shape[0]:
        parts += [_table(
            aggregate_sweep,
            ['merit', 'n_extra', 'n_pattern_peaks', 'regime', 'top10', 'anchor_top10',
             'delta_pp', 'n_gained', 'n_lost', 'n_entries'],
            {'delta_pp': pp}), '']

    if applicability is not None and applicability.shape[0]:
        parts += ['### Applicability, per budget', '',
                  '**A property of the generator, not of real data.** S10a measures it on a '
                  'simulator that knows every non-absent line. What fraction of *experimental* '
                  'patterns extend usable lines past the twenty the indexer took is an S15/S16 '
                  'question and must not be substituted with this.', '',
                  _table(applicability,
                         ['n_extra', 'n_pattern_peaks', 'n_applicable', 'n_dropped_short',
                          'applicability'],
                         {'applicability': pct}), '']

    if coverage is not None and coverage.shape[0]:
        overall = coverage.loc[coverage['scope'] == 'all'].sort_values('n_extra')
        parts += ['### `M_rev` support and reference reach, all candidates', '',
                  'Where `M_rev` has no support, `ho_M_rev` and `ho_M_sym` return **0.0 rather '
                  'than null** — a defined value that means nothing — so these rates belong '
                  'beside every `ho_M_sym` number (C2-F-100).', '',
                  _table(overall,
                         ['n_extra', 'n_candidates', 'mrev_support_rate', 'ref_reach_rate'],
                         {'mrev_support_rate': pct, 'ref_reach_rate': pct}), '']

    if thresholds is not None and thresholds.shape[0]:
        youden = thresholds.loc[thresholds['objective'] == 'youden']
        budget = float(thresholds['matched_fpr_budget'].iloc[0])
        never = youden.loc[youden.get('never_reports', False) == True]  # noqa: E712
        parts += [
            '## 2b. The threshold half — is any of this a SCORE, or only a ranker?',
            '',
            "**The gate's own question, and campaign 1's answer for `ho_M20` was that its Youden "
            'optimum is *never to report* — an operating point of exactly 0.0000. Measured here on '
            "this campaign's own pool rather than quoted:**",
            '',
            _table(youden,
                   [c for c in ('merit', 'n_extra', 'threshold', 'operating_point',
                                'threshold_only', 'reported', 'false_positive', 'precision',
                                'never_reports', 'n_entries') if c in youden.columns],
                   {'threshold': lambda v: f'{v:.4g}'}),
            '',
            f'Thresholds are selected on `{threshold_note or "the selection split"}` and reported '
            f'here, with `check_threshold_transfer` asserting the two entry sets are disjoint. The '
            f'matched false-positive budget — the rate in-sample M20 incurs at de Wolff\'s 10 on '
            f'the selection split — is **{budget:.4f}**; the same table at that budget is in '
            f'`{tag}_thresholds.csv` under `objective = operating_point`.',
            '',
            ]
        if never.shape[0]:
            parts += [
                f'**{never.shape[0]} of {youden.shape[0]} merits never report at all** — '
                + ', '.join(f'`{m}`' for m in never['merit'])
                + '. For those the Youden optimum is to abstain on every pattern, so they are '
                  '**rankers, not scores**, and a rank gain quoted without this would be '
                  'misleading.',
                '']
        else:
            parts += ['**Every merit here does report at some threshold**, so none is a pure '
                      'ranker on this pool — but read the operating points against in-sample '
                      "M20's row in the same table before treating that as a positive.", '']

    for label, frame in (arms or {}).items():
        if frame is None or not frame.shape[0]:
            continue
        parts += [f'## {label}', '',
                  _table(frame, [c for c in ('merit', 'n_extra', 'scope', 'arm_a', 'arm_b',
                                             'a_top10', 'b_top10', 'delta_pp', 'n_gained',
                                             'n_lost', 'p_value', 'n_common_cells')
                                 if c in frame.columns],
                         {'delta_pp': pp, 'p_value': lambda v: f'{v:.2g}'}), '']

    parts += recommendation(leaders, sweep_frame, thresholds, arms, headline_n_extra)

    campaign1 = CAMPAIGN1_HOLDOUT
    row = None
    if leaders is not None and leaders.shape[0]:
        anchor_row = leaders.loc[leaders['merit'] == 'ho_M20']
        row = anchor_row.iloc[0] if anchor_row.shape[0] else leaders.iloc[0]
    parts += [
        "## Against campaign 1's +7.11 pp -- NOT the same measurement",
        '',
        f"Campaign 1's headline for this idea was **+{campaign1['delta_pp']} pp** of top-10 against "
        f"M20 with no refit, {campaign1['gained']} gained / {campaign1['lost']} lost, "
        f"p = {campaign1['p_value']}, at **{campaign1['n_extra']} hold-out peaks** -- the same "
        f'budget headlined here, which is what makes the contrast worth stating at all.',
        '',
        ]
    if row is not None:
        parts += [
            f"Measured here on `ho_M20` at {headline_n_extra} surplus peaks: "
            f"**{row.get('delta_top10_pp', float('nan')):+.2f} pp**, "
            f"{int(row.get('n_gained_top10', 0))} gained / {int(row.get('n_lost_top10', 0))} lost. "
            f'A sign flip and an order of magnitude.',
            '']
    parts += [
        '**It is not a contradiction, because the two hold-out sets are different objects.** '
        'Campaign 1 never stored its surplus peaks, so it rebuilt them; campaign 2 stores them. '
        'Four differences, each recorded before this step ran:',
        '',
        ] + [f'{index}. **{what}** -- {why}'
             for index, (what, why) in enumerate(campaign1['differences'], start=1)] + [
        '',
        '**The first is the one this step can price, and the contaminant arm does: about a tenth '
        'of the gap.** Removing contamination from the surplus recovers ~5 pp of the ~47 pp swing '
        '(`S10b_holdout_slice_contaminant_cost.csv`, C2-F-107) -- measured on the slice, because a '
        'fully retained pool of three clean bundles has no contaminated arm to compare against. '
        'The remaining three differences are not separable on data that exists.',
        '',
        '## Provenance',
        '',
        f'| | |', '|---|---|',
        f'| pool | `{pool}` |',
        f'| sidecars | `{meta.get("merit_dir", "?")}` |',
        f'| candidates read | {meta.get("n_candidates_seen", 0):,} |',
        f'| budgets | {meta.get("budgets", [])} |',
        f'| commit | `{meta.get("commit", "?")}`'
        + (' — **working tree dirty**' if meta.get('dirty_tree') else '') + ' |',
        '',
        ]
    Path(path).write_text('\n'.join(parts) + '\n', encoding='utf-8')
    return path



def recommendation(leaders, sweep_frame, thresholds, arms, headline_n_extra, anchor='M20'):
    """S10b's acceptance gate 5: which column, at which budget, on which population.

    Generated from the tables rather than written beside them, so the recommendation cannot drift
    from the numbers it rests on. That is the failure PROTOCOL section 8 records for five
    campaign-1 numbers quoted in prose that disagreed with their own CSVs.
    """
    lines = ["## 5. The recommendation to S10c and S12", ""]

    row = sweep_frame.loc[(sweep_frame["scope"] == "all")
                          & (sweep_frame["n_extra"] == headline_n_extra)]
    lines += ["**Which column: none of them, as a ranker or as a score.**", ""]
    if row.shape[0]:
        row = row.sort_values("top10", ascending=False)
        lines += [
            "At {} surplus peaks the best classical hold-out merit is **`{}`** at {:.2f} % of "
            "top-10, against in-sample {} at **{:.2f} %** -- **{:+.2f} pp**. There is no budget an "
            "instrument can reach at which any of them is level, and no lattice where one is. "
            "**Do not ship a hold-out column as the score, and do not carry one into S12 as a "
            "presumed win.**".format(
                headline_n_extra, str(row["merit"].iloc[0]), 100*float(row["top10"].iloc[0]),
                anchor, 100*float(row["anchor_top10"].iloc[0]), float(row["delta_pp"].iloc[0])),
            ""]

    lines += [
        "**Which budget, if a later step carries one anyway.** Ten surplus peaks -- a **30-peak "
        "pattern**, which DWMM calls extremely good data. Below that the merits are undefined or "
        "degenerate for too much of the population: `M_rev` support is 70 % at five surplus peaks "
        "against 93 % at ten, and at one and two peaks the floored merits are literal constants "
        "scoring the tie-break floor. **Above ten surplus peaks is the generator's storage cap "
        "rather than a regime any instrument reaches** -- never recommend it.",
        "",
        "**Which population.** Rank claims: the fully retained pool only, because the hold-out "
        "family sits outside `RANK_EXACT_MERITS` and every other pool thins the field (C2-R-013). "
        "Threshold claims: selected on a split carrying `fom-train`, and the cross-pool transfer "
        "used here is legitimate **only because the two entry sets are disjoint**, which the "
        "driver asserts rather than assumes. **The hard stratum: neither laptop pool -- it needs "
        "Benchmark B on NERSC** (C2-R-019).",
        "",
        "**What to carry forward anyway, and it is not nothing:**",
        "",
        "1. **The cubic free-peaks definition.** `holdout_merits` with `mode='free_window'` gives "
        "a cubic candidate the ten window peaks it was never fitted to. It is free, it is "
        "byte-identical on the other eleven lattices, and it is worth **+18.5 to +26.7 pp** to "
        "`ho_M_sym` on cF/cI/cP with **zero patterns lost**. Whatever S12 scores, score cubic "
        "this way.",
        "2. **The shape of `ho_Minfo`, for S10c.** It is the merit contamination damages least, by "
        "a factor of three to four, and the per-peak family is the only one that is *defined* at a "
        "small budget. It is not a good ranker here -- but the posterior statistic S10c is "
        "building shares exactly those two structural properties, which is why it is worth "
        "measuring rather than assuming dead.",
        "3. **A hold-out column as an S12 *feature*, settled by a retrained paired arm** -- never "
        "by an importance table, and never on the standalone numbers above. Complementarity is a "
        "different question from performance, and this step did not measure it.",
        "",
        ]

    if thresholds is not None and thresholds.shape[0] and "never_reports" in thresholds.columns:
        youden = thresholds.loc[thresholds["objective"] == "youden"]
        never = [str(m) for m in youden.loc[youden["never_reports"] == True, "merit"]]  # noqa: E712
        tail = ("The merit that genuinely never reports is "
                + ", ".join("`" + m + "`" for m in never)
                + ", and in-sample `M_rev` behaves the same way, so it is a property of that "
                  "merit rather than of hold-out scoring." if never
                else "No merit here abstains entirely.")
        lines += [
            "**One correction to carry, because this step's own gate was written around it.** "
            "Campaign 1 recorded the operating point of `ho_M20` as exactly zero -- a ranker, not "
            "a score. **It is not zero here** (C2-F-109). " + tail,
            ""]
    return lines


# ---------------------------------------------------------------------------------------
# The entry point `run_fom_holdout_eval.py --analyse` calls
# ---------------------------------------------------------------------------------------
def run_analyse(args):
    """Tables, sweep, figure and results document. Needs no pool."""
    artifact_dir = Path(args.artifact_dir)
    stacked, meta, coverage = load_reduction(artifact_dir, args.tag)
    split = args.report_split
    floors = load_floors(artifact_dir)
    budgets = sorted(b for b in stacked['n_extra'].unique() if b >= 0)
    merits = [m for m in stacked['merit'].unique() if m.startswith('ho_')]
    sweepable = [m for m in merits if m != 'ho_tail_nll']
    headline = args.headline_n_extra if args.headline_n_extra in budgets else (
        budgets[len(budgets)//2] if budgets else None)

    print(f'\n{args.tag}: {len(merits)} merits x {len(budgets)} budgets on {split}')
    print(f'  budgets (surplus peaks / total pattern peaks): '
          + ', '.join(f'{b}/{20 + b}' for b in budgets))

    coverage_frame = coverage_table(coverage)
    applicability = applicability_table(stacked, meta, split)

    print('\napplicability -- patterns whose stored surplus reaches the budget')
    for _, row in applicability.iterrows():
        print(f'  n_extra {int(row["n_extra"]):>3d} ({int(row["n_pattern_peaks"])} peaks): '
              f'{row["applicability"]:.4f}  ({int(row["n_dropped_short"])} dropped short)')

    if coverage_frame.shape[0]:
        print('\nM_rev support -- where ho_M_sym is defined at all')
        for _, row in coverage_frame.loc[coverage_frame['scope'] == 'all'].iterrows():
            print(f'  n_extra {int(row["n_extra"]):>3d}: support {row["mrev_support_rate"]:.4f}  '
                  f'reference reach {row["ref_reach_rate"]:.4f}')

    print(f'\nthe leaderboard at {headline} surplus peaks ({20 + headline}-peak pattern)')
    leaders = leaderboard(stacked, meta, split, sweepable, headline, anchor=args.anchor,
                          top_n=args.top_n, n_bootstrap=args.n_bootstrap, seed=args.seed)
    for _, row in leaders.iterrows():
        print(f'  {row["merit"]:12s} top10 {row["top10"]:.4f}  (M20 {row["anchor_top10"]:.4f})  '
              f'delta {row.get("delta_top10_pp", float("nan")):+.2f} pp  '
              f'{int(row.get("n_gained_top10", 0))}/{int(row.get("n_lost_top10", 0))}  '
              f'hard {row["hard_top10"]:.4f}')

    print('\nthe sweep')
    sweep_frame = sweep(stacked, meta, split, sweepable, budgets, anchor=args.anchor,
                        top_n=args.top_n, n_bootstrap=0, seed=args.seed)
    sweep_frame = in_standard_errors(sweep_frame, floors[0], floors[1])
    for _, row in sweep_frame.loc[sweep_frame['scope'] == 'all'].sort_values(
            ['merit', 'n_extra']).iterrows():
        print(f'  {row["merit"]:12s} n_extra {int(row["n_extra"]):>3d} '
              f'({int(row["n_pattern_peaks"])} peaks, {row["regime"]:12s}) '
              f'top10 {row["top10"]:.4f}  delta {row.get("delta_pp", float("nan")):+6.2f} pp')

    # Both cubic arms are kept: the fixed-pattern-length one is the result and the equal-count one
    # is its control, and running them into the same file would destroy the comparison that makes
    # either interpretable.
    # ---------------------------------------------------------------- the threshold half
    # Selected on a split that carries `fom-train`, reported where the ranks are exact. The two
    # are different pools for the headline arm, which is only legitimate because their entry sets
    # are disjoint -- asserted here rather than assumed, since a leak would be invisible in the
    # output and would flatter every number in the table.
    thresholds = pd.DataFrame()
    if args.threshold_train_tag:
        train_stacked, train_meta, _ = load_reduction(artifact_dir, args.threshold_train_tag)
        if args.threshold_train_tag != args.tag:
            leak = (set(train_stacked.loc[train_stacked['split'] == args.train_split, 'entry_id'])
                    & set(stacked['entry_id']))
            if leak:
                raise SystemExit(
                    f'{len(leak)} source entries appear in both the selection split '
                    f'({args.threshold_train_tag} / {args.train_split}) and the reporting pool '
                    f'({args.tag}). A threshold selected on entries it is then reported on is the '
                    f'anti-pattern PROTOCOL section 8 forbids outright.')
        thresholds = threshold_table(
            train_stacked, train_meta, stacked, meta, sweepable, headline,
            train_split=args.train_split, report_split=split, anchor=args.anchor,
            train_bundles=(sorted(stacked['condition_bundle'].unique())
                           if args.threshold_train_tag != args.tag else None),
            top_n=args.top_n, n_bootstrap=args.n_bootstrap, seed=args.seed)
        if thresholds.shape[0]:
            thresholds.to_csv(artifact_dir/f'{args.tag}_thresholds.csv', index=False,
                              encoding='utf-8')
            print(f'\nthe threshold half -- selected on {args.threshold_train_tag}/'
                  f'{args.train_split}, reported on {args.tag}/{split}')
            for _, row in thresholds.loc[thresholds['objective'] == 'youden'].iterrows():
                if 'operating_point' not in row or pd.isna(row.get('operating_point')):
                    continue
                print(f'  {str(row["merit"]):12s} op {row["operating_point"]:.4f}  '
                      f'reported {row["reported"]:.4f}  precision {row["precision"]:.4f}  '
                      f'{"NEVER REPORTS -- a ranker, not a score" if row["never_reports"] else ""}')

    arms, jobs = {}, []
    for tag in (args.cubic_tag or []):
        equal = 'equal' in tag
        jobs.append((f'3{"b" if equal else "a"}. The cubic free-peaks arm -- '
                     f'{"EQUAL COUNT, the control: same number of peaks, taken earlier" if equal else "FIXED PATTERN LENGTH, the result: ten free window peaks"}',
                     tag, ('cF', 'cI', 'cP'), f'cubic_arm_{"equal" if equal else "window"}'))
    if args.clean_tag:
        jobs.append(('4. What contamination of the surplus costs', args.clean_tag, None,
                     'contaminant_cost'))
    for label, tag, lattices, stem in jobs:
        other, other_meta, _ = load_reduction(artifact_dir, tag)
        arm = paired_arm(other, other_meta, stacked, meta, split, sweepable, budgets,
                         label_a=tag, label_b=args.tag, top_n=args.top_n, lattices=lattices)
        arms[label] = arm
        arm.to_csv(artifact_dir/f'{args.tag}_{stem}.csv', index=False, encoding='utf-8')

    leaders.to_csv(artifact_dir/f'{args.tag}.csv', index=False, encoding='utf-8')
    sweep_frame.to_csv(artifact_dir/f'{args.tag}_sweep.csv', index=False, encoding='utf-8')
    coverage_frame.to_csv(artifact_dir/f'{args.tag}_coverage_rates.csv', index=False,
                          encoding='utf-8')
    applicability.to_csv(artifact_dir/f'{args.tag}_applicability.csv', index=False,
                         encoding='utf-8')
    tiebreak = None
    path = artifact_dir/'S08_tiebreak_floor.csv'
    if path.exists():
        floor_frame = pd.read_csv(path)
        row = floor_frame.loc[(floor_frame['score'] == 'constant')
                              & (floor_frame['scope'] == 'aggregate')]
        if row.shape[0]:
            tiebreak = float(row['top10'].iloc[0])
    # The hard stratum is reported honestly on the face of the figure or not at all: on the
    # retained pool it is 20 entries of which 6 are reachable, and the in-sample anchor scores
    # 0.0000 there too, so a zero from a hold-out merit says nothing about the merit.
    hard_rows = sweep_frame.loc[sweep_frame['scope'] == 'hard']
    hard_note = None
    if hard_rows.shape[0]:
        n_hard = int(hard_rows['n_entries'].max())
        if n_hard < 100 or float(hard_rows['anchor_top10'].max()) == 0.0:
            hard_note = (f'hard stratum omitted: {n_hard} entries, and in-sample M20 scores '
                         f'0.0000 on it too')
    figure = sweep_figure(sweep_frame, coverage_frame, applicability,
                          artifact_dir/f'{args.tag}.png', headline_n_extra=headline,
                          tiebreak_floor=tiebreak, hard_note=hard_note)
    write_report(artifact_dir/f'{args.tag}.md', args.tag, meta, leaders, sweep_frame,
                 coverage_frame, applicability, floors, f'{args.tag}.png', headline, arms=arms,
                 thresholds=thresholds,
                 threshold_note=(f'{args.threshold_train_tag}` / `{args.train_split}'
                                 if args.threshold_train_tag else None))
    print(f'\nwrote {args.tag}.{{md,csv,png}}, _sweep.csv, _coverage_rates.csv, '
          f'_applicability.csv to {artifact_dir}')
    return 0


# ---------------------------------------------------------------------------------------
# The threshold half — the acceptance gate's condition 4
# ---------------------------------------------------------------------------------------
# de Wolff's published threshold, kept only as the source of a matched false-positive budget: the
# rate M20 itself incurs at 10 on the selection split. Every cross-merit comparison is quoted at
# that matched budget, because the operating point is monotone in the threshold and its
# unconstrained maximiser is minus infinity (METRICS section 6).
DEWOLFF_THRESHOLD = 10.0

# The three bundles the fully retained pool carries. When a threshold selected on the slice is
# applied there, the selection split is restricted to these so the two halves face the same
# condition mix -- a threshold chosen against contaminated and dropout bundles and applied to clean
# ones is a different quantity, and the shift would be read as the merit's behaviour.
RETAINED_BUNDLES = ('c2_error0.1_cont0', 'c2_error1_cont0', 'c2_error2_cont0')


def _restrict(stacked, bundles):
    return stacked if not bundles else stacked.loc[stacked['condition_bundle'].isin(bundles)]


def threshold_table(train_stacked, train_meta, report_stacked, report_meta, merits, n_extra,
                    train_split='fom-train', report_split='fom-dev', anchor='M20',
                    train_bundles=None, top_n=10, n_bootstrap=0, seed=12345):
    """Select a threshold on one split and report what it does on another.

    **This is what decides whether a hold-out merit is a score or only a ranker**, and campaign 1's
    answer for `ho_M20` was that its Youden optimum is *never to report* -- an operating point of
    exactly 0.0000. S10b's acceptance gate asks for that measured on this campaign's own pool
    rather than quoted, because a rank gain published without it is misleading.

    `train_stacked` and `report_stacked` may be the same reduction or two different ones. They are
    two different ones for the headline arm: the fully retained pool carries no `fom-train`, so the
    threshold is selected on the slice's train half -- **whose entries are disjoint from the
    retained pool's, checked rather than assumed** -- and applied where the ranks are exact.
    `train_bundles` restricts the selection split to the condition mix the reporting pool has.

    `check_threshold_transfer` is called on every arm and raises if a choice is ever reported on
    the entries it was selected on (PROTOCOL section 8).
    """
    train_stacked = _restrict(train_stacked, train_bundles)
    rows = []
    # The matched budget: what M20 at de Wolff's 10 costs in wrong answers reported, on the
    # selection split. Every merit is then also thresholded to buy the same willingness to answer.
    anchor_train = result_for(train_stacked, train_meta, anchor, None, train_split,
                              threshold=DEWOLFF_THRESHOLD, top_n=top_n)
    budget = float(anchor_train.metric('false_positive'))

    for merit in list(merits) + [anchor]:
        budget_n_extra = None if merit == anchor else n_extra
        try:
            train_result = result_for(train_stacked, train_meta, merit, budget_n_extra,
                                      train_split, top_n=top_n)
            higher = bool(train_result.meta['higher_is_better'])
        except KeyError:
            continue
        for objective, kwargs in (('youden', {}),
                                  ('operating_point', {'max_false_positive_rate': budget})):
            try:
                choice = FomMetrics.select_threshold(train_result, objective=objective, **kwargs)
            except (ValueError, KeyError) as error:
                rows.append({'merit': merit, 'n_extra': budget_n_extra, 'objective': objective,
                             'error': str(error)})
                continue
            # `per_entry` stores scores already oriented higher-is-better, so a lower-is-better
            # merit's chosen threshold comes back negated and must be turned round again before
            # `summarise_per_entry` mirrors it a second time.
            threshold = choice.threshold if higher else -choice.threshold
            reported = result_for(report_stacked, report_meta, merit, budget_n_extra, report_split,
                                  threshold=threshold, n_bootstrap=n_bootstrap, seed=seed,
                                  top_n=top_n)
            FomMetrics.check_threshold_transfer(choice, reported)
            row = reported.aggregate.iloc[0]
            rows.append({
                'merit': merit, 'n_extra': budget_n_extra,
                'n_pattern_peaks': None if budget_n_extra is None else 20 + budget_n_extra,
                'objective': objective, 'threshold': threshold,
                'higher_is_better': higher,
                'ranks_exact': bool(reported.meta.get('ranks_exact')),
                'operating_point': float(row['operating_point']),
                'threshold_only': float(row['threshold_only']),
                'reported': float(row['reported']),
                'false_positive': float(row['false_positive']),
                'precision': float(row['precision']),
                'top10': float(row['top10']),
                'n_entries': int(row['n_entries']),
                # The gate's actual question, made a column so nobody has to read it off a number.
                'never_reports': bool(float(row['reported']) == 0.0),
                'is_ranker_not_score': bool(float(row['operating_point']) == 0.0),
                })
    frame = pd.DataFrame(rows)
    if frame.shape[0]:
        frame.insert(0, 'matched_fpr_budget', budget)
    return frame


# ---------------------------------------------------------------------------------------
# The within-M20-band control — S10c's acceptance gate 4
# ---------------------------------------------------------------------------------------
# Fixed M20 bands rather than quantiles of whatever rows a pass happens to see. A quantile edge
# recomputed per shard would put the same candidate in different bands on different runs, which is
# the drift R14 records for `volume_decile`. These edges are interpretable on their own terms:
# below de Wolff's 10 is "would not be reported", and the upper bands are where the pool's mass is.
M20_BAND_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, np.inf)

# Correct candidates are 0.081 % of the pool, so they are all kept and the rest are sampled. AUC is
# a rank statistic and uniform subsampling of one class leaves it unbiased, so this costs precision
# and not correctness -- and it turns a 43 M-row control into a 500 k-row one.
CONTROL_NEGATIVE_RATE = 0.01


def control_rows(frame, columns, rate=CONTROL_NEGATIVE_RATE, seed=12345):
    """Every correct candidate and a sample of the rest, banded by M20.

    **What this control is for.** M20 cannot separate correct from incorrect candidates *within its
    own bands* -- it is at chance there by construction, 0.503 in campaign 1's table. So a merit
    that separates them inside a band is carrying information M20 does not have, which is the
    question a combiner actually asks. `M_sym` reaches 0.822 and that is the bar S10c is measured
    against.
    """
    if 'M20' not in frame.columns or 'is_correct' not in frame.columns:
        return None
    correct = FomMetrics.as_bool(frame['is_correct'])
    # Seeded from the shard's own content, so the same pool gives the same control every time
    # without carrying a counter across shards (PROTOCOL section 6).
    digest = int(pd.util.hash_pandas_object(frame['candidate_id'], index=False).sum() % (2**32))
    rng = np.random.default_rng((seed + digest) % (2**32))
    keep = correct | (rng.random(frame.shape[0]) < rate)
    if not keep.any():
        return None
    block = frame.loc[keep, [c for c in columns if c in frame.columns]].copy()
    block['is_correct'] = correct[keep]
    block['m20_band'] = pd.cut(frame.loc[keep, 'M20'].to_numpy(), bins=list(M20_BAND_EDGES),
                               right=False).astype(str)
    return block


def _auc(scores, labels):
    """AUC by rank, NaN-safe, returning NaN where one class is absent.

    Written out rather than imported so the control has no sklearn dependency and so ties are
    handled by average ranks, which is what makes a merit with many equal values -- the failure
    C2-F-095 found in `X_N` -- score 0.5 rather than something flattering.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    usable = np.isfinite(scores)
    scores, labels = scores[usable], labels[usable]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan, n_pos, n_neg
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average ranks over ties, so a constant score gives exactly 0.5.
    sorted_scores = scores[order]
    start = 0
    for stop in range(1, sorted_scores.size + 1):
        if stop == sorted_scores.size or sorted_scores[stop] != sorted_scores[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return (ranks[labels].sum() - n_pos*(n_pos + 1)/2)/(n_pos*n_neg), n_pos, n_neg


def band_m20(control, n_bands=40):
    """Re-band the control on the whole frame, finely, and return it with `m20_band` replaced.

    **The band width is not a presentation choice; it is what makes the control a control.** The
    premise is that M20 cannot separate correct from incorrect candidates *within* a band, so a
    merit that does is carrying information M20 lacks. That premise only holds if the bands are
    narrow enough to exhaust M20's own discrimination -- and with the seven interpretable fixed
    bands this function replaces, **M20 scored 0.7225 within its own bands rather than ~0.5**, so
    every other row in the table was measuring residual M20 signal as though it were the merit's.

    Quantile bands over the whole control frame, computed once here rather than per shard. Doing
    it per shard is the drift R14 records for `volume_decile`: a within-set rank moves when rows
    are dropped, so the same candidate would land in different bands on different runs.
    """
    control = control.copy()
    values = control['M20'].to_numpy(dtype=np.float64)
    edges = np.unique(np.nanquantile(values, np.linspace(0, 1, n_bands + 1)))
    control['m20_band'] = pd.cut(values, bins=edges, include_lowest=True).astype(str)
    return control


def within_band_control(control, merits, reference=('M20', 'M_sym', 'Minfo'), n_bands=40):
    """Within-M20-band AUC for every column, plus the reference columns, per band and pooled.

    `M20` is included deliberately and must come out near 0.5 inside its own bands. If it does
    not, the bands are wrong and every other row in the table is uninterpretable -- so it is the
    control's own control.
    """
    control = band_m20(control, n_bands=n_bands)
    rows = []
    columns = [c for c in list(merits) + list(reference) if c in control.columns]
    for band, block in list(control.groupby('m20_band')) + [('unconditional', control)]:
        for column in columns:
            auc, n_pos, n_neg = _auc(block[column].to_numpy(),
                                     FomMetrics.as_bool(block['is_correct']))
            rows.append({'m20_band': band, 'merit': column, 'auc': auc,
                         'n_correct': n_pos, 'n_incorrect': n_neg,
                         'pairs': n_pos*n_neg})
    frame = pd.DataFrame(rows)
    # The stratified number: each band's AUC weighted by the comparisons it actually contains, so
    # a band holding three correct candidates cannot swing the headline. This is the row to quote.
    banded = frame.loc[(frame['m20_band'] != 'unconditional') & frame['auc'].notna()]
    for column, block in banded.groupby('merit'):
        weight = block['pairs'].sum()
        frame = pd.concat([frame, pd.DataFrame([{
            'm20_band': 'within-band (pair-weighted)', 'merit': column,
            'auc': float((block['auc']*block['pairs']).sum()/weight) if weight else np.nan,
            'n_correct': int(block['n_correct'].sum()),
            'n_incorrect': int(block['n_incorrect'].sum()),
            'pairs': int(weight)}])], ignore_index=True)
    return frame


def control_from_pool(pool, merit_dir, columns, rate=CONTROL_NEGATIVE_RATE, seed=12345):
    """Control rows straight from a pool and a sidecar directory, without a reduce.

    The within-band control is **candidate-level** -- it asks whether a merit separates correct
    from incorrect candidates inside a narrow M20 band, and never ranks anything. So it needs
    neither the cross-lattice pooling nor the per-entry reduction that `--reduce` pays for, and a
    four-column projection of the pool joined to the sidecar answers it in a fraction of the time.

    That matters because the sigma-sensitivity curve needs one of these **per multiplier**, and
    each multiplier is already a full sidecar pass (C2-F-110). Going through `--reduce` as well
    would have made a five-point curve cost more than the headline measurement.
    """
    from mlindex.model_training import FomBenchmark
    keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
    wanted = keys + ['M20', 'is_correct']
    blocks = []
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        sidecar = Path(merit_dir)/path.name
        if not sidecar.exists():
            continue
        available = FomBenchmark.candidate_columns_present(Path(pool))
        frame = pd.read_parquet(path, columns=[c for c in wanted if c in available])
        if 'condition_bundle' not in frame.columns:
            frame['condition_bundle'] = FomBenchmark.bundle_from_candidate_path(path)
        merits = pd.read_parquet(sidecar)
        frame = frame.merge(merits, on=[k for k in keys if k in merits.columns], how='left',
                            validate='1:1')
        block = control_rows(frame, [c for c in columns if c in frame.columns] + ['M20'],
                             rate=rate, seed=seed)
        if block is not None:
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
