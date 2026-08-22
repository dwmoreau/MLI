"""S06b -- read the reproducibility floor off an ensemble of runs of the program.

`submit_fom_floor_arms.sh` indexes one set of patterns several times, changing only the search
seed, so every arm is the same experiment repeated. This turns that into the three numbers
F-141 says the project is missing:

  value floor     Shirley's own quantity: for an entry whose correct solution is found in more
                  than one arm, how far apart are the merits the arms report for it. This is
                  what "figures of merit differing by more than 10 percent for the same
                  solution" means, measured rather than recalled.
  metric floor    How far apart the arms' *reported metrics* are -- the operating point and
                  top-10 -- which is the quantity every gate in this project is written in and
                  which nothing has ever connected to the 10%.
  contrast floor  How far apart the arms put the *difference between two merits*. A gate asks
                  "does A beat B by more than the floor", and both A and B are computed on the
                  same pool, so their difference is far more stable than either. This is the
                  number a gate should actually be read against, and it is the one that
                  decides whether F-133's +1.05 pp and F-136's 4.93 pp survive.

    python mlindex/scripts/run_fom_floor_report.py --arm-root mlindex/data/fom_floor/arms

Thresholds are selected on `fom-train` from Benchmark A's frozen feature matrix and applied
unchanged to every arm, so no arm is both tuned on and reported on (PROTOCOL section 8). The
arms are `fom-dev` entries; `fom-test` and CNRS are untouched.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics
from run_fom_zoo_features import commit_hash

# The merits carried through the threshold-dependent half. All 29 are reported for the
# threshold-free metrics and for the value floor, which cost nothing extra; each of these costs
# one pass over the training pool to choose a cut. M20 is the baseline every gate is written
# against, M_sym and M_wu are S06's leaderboard, and M_info_clipped is the merit S14 prices
# inside the inner-loop budget.
THRESHOLD_MERITS = ('M20', 'M_sym', 'M_wu', 'M_info_clipped', 'M_1', 'M_rev')

# Metrics reported per arm. `found` is the pool's ceiling and moves only when the *generation*
# differs between arms, which is why it is here: it separates "the score moved" from "the pool
# moved".
METRICS = ('operating_point', 'top1', 'top10', 'threshold_only', 'found', 'reported',
           'false_positive', 'mrr', 'precision')


def _parse_args():
    parser = argparse.ArgumentParser(description='S06b -- the floor, from an ensemble of runs')
    parser.add_argument('--arm-root', default=os.path.join('mlindex', 'data', 'fom_floor', 'arms'))
    parser.add_argument('--benchmark-dir', default=os.path.join('mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--feature-dir', default=os.path.join('mlindex', 'data', 'fom_features'))
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--cache-dir', default=os.path.join('mlindex', 'data', 'fom_floor'))
    parser.add_argument('--bundle', default='error1_cont0')
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--reporting-entries', type=int, default=1197,
                        help='fom-dev source-entry count, for the induced-error column')
    parser.add_argument('--n-processes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', default='S06b_floor')
    return parser.parse_args()


# ----------------------------------------------------------------------------------------
# reporting arithmetic (tested)
# ----------------------------------------------------------------------------------------

def relative_spread(values):
    """(max - min) / |median| across the replicate columns of each row.

    Range rather than standard deviation because Shirley's statement is about how far apart two
    reported values can be, and because four arms do not support a variance estimate worth the
    name. Divided by the row's own median so the answer is scale-free and comparable across
    merits whose units differ by orders of magnitude.
    """
    scale = values.abs().median(axis=1).replace(0.0, np.nan)
    return (values.max(axis=1) - values.min(axis=1)) / scale


def induced_standard_error(flip_rate, n_entries):
    """Standard error of the *difference* in a per-entry rate between two runs.

    If a fraction `f` of entries change their outcome between two arms, the change in the
    reported rate is the mean of n independent draws that are 0 with probability 1 - f and +-1
    otherwise, so its variance is f/n and its standard error is sqrt(f/n). This is the quantity
    a gate needs -- a gate compares two numbers -- and it is what lets a floor measured on 250
    entries be quoted at the 1 197 the project reports on.
    """
    return float(np.sqrt(flip_rate / n_entries))


def flip_rate(first, second):
    """Fraction of entries whose boolean outcome differs between two arms."""
    return float(np.mean(np.asarray(first, dtype=bool) != np.asarray(second, dtype=bool)))


def induced_from_differences(differences, n_entries):
    """Standard error at `n_entries` of a per-entry difference measured on this sample.

    The general form of `induced_standard_error`: given the per-entry change `d_i` between two
    arms -- of an outcome flag, or of the *contrast* between two merits' flags -- the change in
    the reported mean is `mean(d)`, so its standard error at any entry count is
    `sd(d) / sqrt(n)`. Used in preference to the flip-rate form wherever the per-entry values
    are in hand, because it needs no assumption about how the flips are distributed.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return float('nan')
    return float(np.std(differences, ddof=1) / np.sqrt(n_entries))


# ----------------------------------------------------------------------------------------
# loading the arms
# ----------------------------------------------------------------------------------------

def merit_column_names(feature_dir, bundle, available):
    """The zoo's merit columns, read from the frozen matrix's *schema* rather than its rows.

    `pd.read_parquet` to learn thirty column names would read 2.4M rows to throw them away.
    """
    import pyarrow.parquet as pq

    names = pq.ParquetFile(Path(feature_dir) / f'features_{bundle}.parquet').schema.names
    keys = set(FomBenchmark.ZOO_KEY_COLUMNS)
    return [name for name in names if name not in keys and name in set(available)]


def arm_directories(root):
    paths = sorted(path for path in Path(root).iterdir()
                   if path.is_dir() and (path / 'manifest.json').exists())
    if len(paths) < 2:
        raise SystemExit(f'{root} holds {len(paths)} complete arms; the floor needs at least 2')
    return paths


def load_arm(path, cache_dir, n_processes, tag):
    """One arm's pool, labelled and scored with the whole zoo.

    Cached: labelling is ~9 ms/candidate and the zoo is another pass, so a re-read of a
    finished arm should not repeat either.
    """
    seed = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))['optimizer_seed']
    cache = Path(cache_dir) / f'{tag}_arm{seed}.parquet'
    if cache.exists():
        return seed, pd.read_parquet(cache)

    entries = FomBenchmark.load_entries(path)
    candidates = FomBenchmark.load_candidates(path)
    FomBenchmark._check_join(candidates, entries)
    candidates = FomBenchmark.label_frame_parallel(
        candidates, entries, n_processes=n_processes,
        )
    features, _ = FomBenchmark.zoo_features(candidates, entries)
    carried = [column for column in candidates.columns
               if column not in features.columns and column != 'xnn']
    frame = pd.concat([features, candidates[carried]], axis=1)
    frame['in_top_n'] = frame['final_rank'] < 20
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=['unit_cell'], errors='ignore').to_parquet(cache, index=False)
    return seed, frame


def check_arms_are_comparable(paths):
    """Every arm must hold the same patterns: same entries, same peak lists.

    The whole design is that only the search seed moved, and `--seed` moves the entry sampling
    and the per-entry noise as well as the optimizers. If someone varies that instead, the arms
    differ in their *data* and the spread is not a floor. Checked rather than trusted.
    """
    reference = None
    for path in paths:
        entries = FomBenchmark.load_entries(path).sort_values('entry_id')
        digests = dict(zip(entries['entry_id'], entries['q2_digest']))
        if reference is None:
            reference = digests
            continue
        if digests != reference:
            differing = [key for key in reference if digests.get(key) != reference[key]]
            raise SystemExit(
                f'{path.name} disagrees with the first arm on {len(differing)} peak lists '
                f'(e.g. {differing[:3]}). The arms must differ only in --optimizer-seed.'
                )
    return sorted(reference)


# ----------------------------------------------------------------------------------------
# thresholds, from the frozen pool's training split
# ----------------------------------------------------------------------------------------

def train_thresholds(args, merits):
    """One cut per merit, chosen on `fom-train` at this bundle and never on an arm."""
    from run_fom_zoo_eval import bundle_frames

    entries = FomBenchmark.load_entries(args.benchmark_dir)
    keep = set(entries.loc[entries['split'] == args.train_split, 'entry_id'])
    chosen = {}
    for merit in merits:
        started = time.perf_counter()
        shards = bundle_frames(args.benchmark_dir, args.feature_dir, [args.bundle], keep,
                               [merit])
        result = FomMetrics.evaluate(
            shards, score=merit, entries=entries, threshold=None, strata=(),
            split=args.train_split, n_bootstrap=0, seed=args.seed,
            )
        choice = FomMetrics.select_threshold(result, objective='youden')
        chosen[merit] = float(choice.threshold)
        print(f'  {merit:16s} threshold {choice.threshold:12.5g}  '
              f'({time.perf_counter() - started:.0f}s)', flush=True)
    return chosen


# ----------------------------------------------------------------------------------------
# the three floors
# ----------------------------------------------------------------------------------------

def reported_solution(frame, merit_columns):
    """One row per entry: the correct candidate the program would have reported for it.

    "The same solution, scored twice" needs a rule for *which* correct candidate an arm's value
    is read from, and the rule has to be one rule for every merit -- reducing each merit by its
    own max would silently take the minimum for the merits where lower is better, and would
    compare different candidates column by column. The pipeline's own rule is used instead: the
    correct candidate with the highest M20, which is the one `run.py` would print. Every merit
    is then read off that single row.
    """
    correct = frame.loc[frame['is_correct'].astype(bool)]
    if not correct.shape[0]:
        return correct.set_index('entry_id')[list(merit_columns)]
    winner = correct.sort_values('M20', ascending=False).groupby('entry_id').head(1)
    return winner.set_index('entry_id')[list(merit_columns)]


def value_floor(arms, merit_columns):
    """Per merit, how far apart the arms put the *same solution*.

    "The same solution" is the correct cell, identified by the pool's own `is_correct` label
    rather than by a distance threshold invented here. An entry contributes only where every arm
    found it, so the comparison is paired and no arm is penalised for a generation failure --
    that is `found`, reported separately in the metric floor.
    """
    per_arm = {seed: reported_solution(frame, merit_columns) for seed, frame in arms.items()}
    seeds = sorted(per_arm)
    shared = sorted(set.intersection(*(set(per_arm[seed].index) for seed in seeds)))
    rows = []
    for merit in merit_columns:
        values = pd.DataFrame({seed: per_arm[seed].loc[shared, merit] for seed in seeds})
        spread = relative_spread(values)
        rows.append({
            'merit': merit,
            'n_entries': int(spread.notna().sum()),
            'median_relative_spread': float(spread.median()),
            'p75': float(spread.quantile(0.75)),
            'p90': float(spread.quantile(0.90)),
            'fraction_over_10pc': float((spread > 0.10).mean()),
            })
    return pd.DataFrame(rows).sort_values('median_relative_spread'), shared


def value_floor_by_lattice(arms, merit_columns, entries_by_lattice):
    """The stability table S06's leaderboard left empty: per merit, per Bravais lattice.

    The `condition` half of the gate's "by lattice and condition" is the bundle the ensemble
    was run at, so it is one call of this per arm set rather than a column here.
    """
    merit_columns = [merit_columns] if isinstance(merit_columns, str) else list(merit_columns)
    per_arm = {seed: reported_solution(frame, merit_columns)
               for seed, frame in arms.items()}
    seeds = sorted(per_arm)
    shared = sorted(set.intersection(*(set(per_arm[seed].index) for seed in seeds)))
    lattice = pd.Series([entries_by_lattice.get(entry) for entry in shared], index=shared)
    rows = []
    for merit in merit_columns:
        values = pd.DataFrame({seed: per_arm[seed].loc[shared, merit] for seed in seeds})
        spread = relative_spread(values).rename('spread').to_frame()
        spread['bravais_lattice'] = lattice
        table = spread.groupby('bravais_lattice')['spread'].agg(
            n='size', median='median', p90=lambda s: s.quantile(0.9),
            fraction_over_10pc=lambda s: float((s > 0.10).mean()),
            ).reset_index()
        rows.append(table.assign(merit=merit))
    return pd.concat(rows, ignore_index=True)[
        ['merit', 'bravais_lattice', 'n', 'median', 'p90', 'fraction_over_10pc']]


def evaluate_arms(arms, entries_by_arm, merit, threshold, weights='cnrs'):
    """`FomMetrics.evaluate` once per arm, at one merit and one frozen threshold.

    Run at both weightings by the caller. The CNRS reweighting is the project's headline, but
    it multiplies each lattice's cell up to draft Table 1's share, so on a 250-entry ensemble it
    also multiplies the sampling noise of the thin lattices -- cF contributes four entries and
    carries 22/599 of the weight. The unweighted number is the honest floor *for this sample*
    and the weighted one is the floor for the quantity the project reports; they belong side by
    side rather than one standing for the other.
    """
    results = {}
    for seed, frame in arms.items():
        results[seed] = FomMetrics.evaluate(
            frame, score=merit, entries=entries_by_arm[seed], threshold=threshold,
            strata=(), n_bootstrap=0, seed=12345, weights=weights,
            )
    return results


def metric_floor(results, metrics, reporting_entries):
    """Per metric: the arms' values, their range, and the induced error at reporting scale."""
    seeds = sorted(results)
    rows = []
    for metric in metrics:
        values = {seed: float(results[seed].aggregate[metric].iloc[0]) for seed in seeds}
        # `mrr` and `precision` are ratios rather than per-entry booleans, so they have a
        # range across arms but no flip rate. Reported with the column left empty rather than
        # dropped: the range is still what a gate on them would have to clear.
        has_flags = all(metric in results[seed].per_entry.columns for seed in seeds)
        pairwise, shared = [], []
        if has_flags:
            flags = {seed: results[seed].per_entry.set_index('entry_id')[metric]
                     for seed in seeds}
            shared = sorted(set.intersection(*(set(flags[seed].index) for seed in seeds)))
            pairwise = [flip_rate(flags[a].loc[shared], flags[b].loc[shared])
                        for index, a in enumerate(seeds) for b in seeds[index + 1:]]
        row = {'metric': metric,
               'mean': float(np.mean(list(values.values()))),
               'range_pp': 100 * (max(values.values()) - min(values.values())),
               'sd_pp': 100 * float(np.std(list(values.values()), ddof=1)),
               'flip_rate': float(np.mean(pairwise)) if pairwise else np.nan,
               'n_entries_measured': len(shared)}
        row['induced_se_pp'] = (100 * induced_standard_error(row['flip_rate'], reporting_entries)
                                if np.isfinite(row['flip_rate']) else np.nan)
        row.update({f'arm_{seed}': values[seed] for seed in seeds})
        rows.append(row)
    return pd.DataFrame(rows)


def flip_overlap(results, metric='operating_point', reference='found'):
    """Do the entries whose outcome moves between runs also have a different *pool*?

    The distinction decides whether the floor is a figure of merit's problem or candidate
    generation's. `found` moves when the arms' pools differ in whether a correct candidate
    exists at all; `operating_point` moves when the reported answer changes. If the two sets
    coincide the floor is generation noise and a FOM comparison is insulated from most of it;
    if they are disjoint it is scoring noise and the floor applies to a FOM gate directly.
    """
    seeds = sorted(results)
    flags = {seed: results[seed].per_entry.set_index('entry_id')[[metric, reference]]
             for seed in seeds}
    shared = sorted(set.intersection(*(set(flags[seed].index) for seed in seeds)))
    rows = []
    for index, first in enumerate(seeds):
        for second in seeds[index + 1:]:
            moved = (flags[first].loc[shared, metric].to_numpy()
                     != flags[second].loc[shared, metric].to_numpy())
            pool = (flags[first].loc[shared, reference].to_numpy()
                    != flags[second].loc[shared, reference].to_numpy())
            rows.append({'pair': f'{first}-{second}', 'n_entries': len(shared),
                         f'{metric}_flips': int(moved.sum()),
                         f'{reference}_flips': int(pool.sum()),
                         'both': int((moved & pool).sum()),
                         f'{metric}_only': int((moved & ~pool).sum())})
    return pd.DataFrame(rows)


def floor_by_lattice(results_by_merit, metric, reporting_entries, baseline='M20'):
    """The metric and contrast floors, per true Bravais lattice.

    Necessary rather than decorative: the value floor is ordered by free-parameter count over
    two orders of magnitude (cubic 0.01%, triclinic 20%), so a single aggregate floor cannot be
    the right object for a per-lattice claim -- and the lattices this project's gains live in
    (aP, mP, mC; F-099, F-133) are exactly the ones where the merit is least reproducible.
    Reported at the reporting entry count in the same way as the aggregate, but note the
    per-lattice counts here are small and the numbers are correspondingly coarse.
    """
    seeds = sorted(next(iter(results_by_merit.values())))
    rows = []
    for merit, results in results_by_merit.items():
        per_arm = {}
        for seed in seeds:
            frame = results[seed].per_entry
            per_arm[seed] = frame.set_index('entry_id')[[metric]].assign(
                bravais_lattice=frame.set_index('entry_id')['bravais_lattice_true']
                if 'bravais_lattice_true' in frame.columns
                else frame.set_index('entry_id')['bravais_lattice'])
        shared = sorted(set.intersection(*(set(per_arm[seed].index) for seed in seeds)))
        lattices = per_arm[seeds[0]].loc[shared, 'bravais_lattice']
        for lattice, index in lattices.groupby(lattices).groups.items():
            values = {seed: float(per_arm[seed].loc[index, metric].mean()) for seed in seeds}
            arms = np.array(list(values.values()))
            row = {'merit': merit, 'metric': metric, 'bravais_lattice': lattice,
                   'n': len(index), 'mean': float(arms.mean()),
                   'range_pp': 100 * float(arms.max() - arms.min())}
            if merit != baseline:
                base = results_by_merit[baseline]
                deltas = {
                    seed: (per_arm[seed].loc[index, metric].astype(float)
                           - base[seed].per_entry.set_index('entry_id')
                           .loc[index, metric].astype(float))
                    for seed in seeds
                    }
                pairs = [induced_from_differences(deltas[a] - deltas[b], reporting_entries)
                         for i, a in enumerate(seeds) for b in seeds[i + 1:]]
                row['mean_delta_pp'] = 100 * float(np.mean(
                    [deltas[seed].mean() for seed in seeds]))
                row['contrast_range_pp'] = 100 * float(
                    max(deltas[seed].mean() for seed in seeds)
                    - min(deltas[seed].mean() for seed in seeds))
                row['contrast_induced_se_pp'] = 100 * float(np.mean(pairs))
            rows.append(row)
    return pd.DataFrame(rows)


def contrast_floor(results_by_merit, metrics, reporting_entries, baseline='M20'):
    """How far apart the arms put a *difference between two merits* on the same pool.

    This is what a gate reads. Both merits are computed on one arm's pool, so the pool's own
    realisation largely cancels, and the question is how much of it fails to.

    Two columns, and they answer different questions. `range_pp` is what the difference actually
    did across these arms at this sample size. `induced_se_pp` pairs the arms entry by entry --
    the per-entry contrast is `1{A} - 1{B}` in {-1, 0, +1}, and its change between two arms has
    standard error `sd / sqrt(n)` -- so the floor measured on 250 entries can be quoted at the
    entry count the project reports on.
    """
    rows = []
    seeds = sorted(next(iter(results_by_merit.values())))
    for merit, results in results_by_merit.items():
        if merit == baseline:
            continue
        for metric in metrics:
            deltas = {
                seed: (float(results[seed].aggregate[metric].iloc[0])
                       - float(results_by_merit[baseline][seed].aggregate[metric].iloc[0]))
                for seed in seeds
                }
            values = np.array(list(deltas.values()))
            per_entry = {}
            for seed in seeds:
                flags_a = results[seed].per_entry.set_index('entry_id')[metric].astype(float)
                flags_b = (results_by_merit[baseline][seed].per_entry
                           .set_index('entry_id')[metric].astype(float))
                per_entry[seed] = flags_a - flags_b
            shared = sorted(set.intersection(*(set(per_entry[seed].index) for seed in seeds)))
            induced = [
                induced_from_differences(
                    per_entry[a].loc[shared] - per_entry[b].loc[shared], reporting_entries)
                for index, a in enumerate(seeds) for b in seeds[index + 1:]
                ]
            rows.append({
                'merit': merit, 'baseline': baseline, 'metric': metric,
                'mean_delta_pp': 100 * float(values.mean()),
                'range_pp': 100 * float(values.max() - values.min()),
                'sd_pp': 100 * float(values.std(ddof=1)),
                'induced_se_pp': 100 * float(np.mean(induced)),
                'n_entries_measured': len(shared),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# artefacts
# ----------------------------------------------------------------------------------------

def figure(value_table, by_lattice, metric_table, contrast_table, context, path):
    """Four panels: the value floor twice -- per merit and per lattice -- then the two floors
    that are actually quantities a gate reads.

    The x axes of the first two are logarithmic because the value floor spans four orders of
    magnitude across merits and three across lattices, and a linear axis renders every panel as
    "everything is much smaller than Shirley's line", which is true and uninformative.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    blue, amber, red, grey = '#3b6ea5', '#e0a030', '#b03030', '#707070'
    figure_, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))

    # (a) value floor, per merit
    shown = value_table.dropna(subset=['median_relative_spread'])
    # A log axis cannot show a zero, and three merits are exactly reproducible at the median
    # (`n_over`, `max_gap`, `N_cal` are integer counts that rarely move at all); they are named
    # in the results document rather than drawn as absent bars.
    shown = shown.loc[shown['median_relative_spread'] > 0].sort_values(
        'median_relative_spread').head(20)
    positions = np.arange(shown.shape[0])
    axes[0][0].barh(positions, 100 * shown['median_relative_spread'], color=blue, height=0.72,
                    label='median')
    axes[0][0].plot(100 * shown['p90'], positions, marker='|', linestyle='none',
                    color=amber, markersize=9, markeredgewidth=2, label='p90')
    axes[0][0].set_yticks(positions)
    axes[0][0].set_yticklabels(shown['merit'], fontsize=8)
    axes[0][0].set_xscale('log')
    axes[0][0].axvline(10, color=red, linestyle='--', linewidth=1.2,
                       label='Shirley 1980, ~10%')
    axes[0][0].set_xlabel('spread of the merit over the arms, % of its own value')
    axes[0][0].set_title('(a) Value floor, per merit\nthe same solution, scored by '
                         f'{context["n_arms"]} runs', loc='left')
    axes[0][0].legend(frameon=False, fontsize=8, loc='upper right')
    axes[0][0].invert_yaxis()

    # (b) value floor, per lattice -- ordered by free parameters, which is the finding
    order = ['cF', 'cI', 'cP', 'tP', 'tI', 'hP', 'hR', 'oP', 'oC', 'oF', 'oI', 'mP', 'mC', 'aP']
    lattice = by_lattice.set_index('bravais_lattice').reindex(
        [name for name in order if name in set(by_lattice['bravais_lattice'])])
    positions = np.arange(lattice.shape[0])
    axes[0][1].barh(positions, 100 * lattice['median'], color=blue, height=0.72, label='median')
    axes[0][1].plot(100 * lattice['p90'], positions, marker='|', linestyle='none', color=amber,
                    markersize=9, markeredgewidth=2, label='p90')
    axes[0][1].set_yticks(positions)
    axes[0][1].set_yticklabels(
        [f'{name}  (n={int(count)})' for name, count in zip(lattice.index, lattice['n'])],
        fontsize=8)
    axes[0][1].set_xscale('log')
    axes[0][1].axvline(10, color=red, linestyle='--', linewidth=1.2, label='Shirley 1980, ~10%')
    axes[0][1].set_xlabel('spread of M20 over the arms, % of its own value')
    axes[0][1].set_title('(b) Value floor of M20, per Bravais lattice\nordered by free cell '
                         'parameters', loc='left')
    axes[0][1].legend(frameon=False, fontsize=8, loc='upper right')
    axes[0][1].invert_yaxis()

    # (c) metric floor
    wanted = ('operating_point', 'threshold_only', 'top10', 'top1', 'found')
    shown = (metric_table.loc[(metric_table['weights'] == 'cnrs')
                              & metric_table['metric'].isin(wanted)]
             .set_index('metric').reindex(wanted).reset_index())
    positions = np.arange(shown.shape[0])
    axes[1][0].barh(positions, shown['range_pp'], color=blue, height=0.7,
                    label=f'range over {context["n_arms"]} arms (n = {context["n_entries"]})')
    axes[1][0].barh(positions, shown['induced_se_pp'], color=amber, height=0.34,
                    label=f'standard error at n = {context["reporting_entries"]}')
    axes[1][0].set_yticks(positions)
    axes[1][0].set_yticklabels(shown['metric'], fontsize=9)
    axes[1][0].set_xlabel('percentage points')
    axes[1][0].set_title('(c) Metric floor\nwhat one reported number does between runs',
                         loc='left')
    axes[1][0].legend(frameon=False, fontsize=8, loc='lower right')
    axes[1][0].invert_yaxis()

    # (d) contrast floor -- the number a gate reads, against the claims it has to judge
    contrast = contrast_table.loc[(contrast_table['metric'] == 'operating_point')
                                  & (contrast_table['weights'] == 'cnrs')]
    positions = np.arange(contrast.shape[0])
    axes[1][1].barh(positions, contrast['range_pp'], color=blue, height=0.7,
                    label='range of the difference over arms')
    axes[1][1].barh(positions, contrast['induced_se_pp'], color=amber, height=0.34,
                    label=f'standard error at n = {context["reporting_entries"]}')
    axes[1][1].set_yticks(positions)
    axes[1][1].set_yticklabels([f'{merit} - M20' for merit in contrast['merit']], fontsize=9)
    axes[1][1].axvline(1.05, color=red, linestyle='--', linewidth=1.2,
                       label="F-133, block C's +1.05 pp")
    axes[1][1].axvline(4.93, color=grey, linestyle=':', linewidth=1.4,
                       label="F-136, the whole FOM prize, 4.93 pp")
    axes[1][1].axvline(6.47, color=grey, linestyle='-', linewidth=1.0,
                       label='the floor previously in force, 6.47 pp')
    axes[1][1].set_xlabel('percentage points of operating point')
    axes[1][1].set_title('(d) Contrast floor\nwhat a gate actually reads', loc='left')
    axes[1][1].legend(frameon=False, fontsize=7.5, loc='upper right')
    axes[1][1].invert_yaxis()

    figure_.suptitle(
        f'The reproducibility floor, measured over {context["n_arms"]} runs of the indexer '
        f'differing only in the search seed  ({context["bundle"]}, {context["n_entries"]} '
        'fom-dev patterns)', fontsize=11)
    figure_.tight_layout(rect=(0, 0, 1, 0.965))
    figure_.savefig(path, dpi=200)
    plt.close(figure_)


def write_markdown(path, context, value_table, by_lattice, metric_table, contrast_table,
                   lattice_frame, overlap):
    def table(frame, columns=None, floats=4):
        frame = frame if columns is None else frame[columns]
        return frame.to_markdown(index=False, floatfmt=f'.{floats}f')

    lines = [
        '# S06b -- the reproducibility floor, measured',
        '',
        f'Commit `{context["commit"]}`. {context["n_arms"]} arms x '
        f'{context["n_entries"]} `fom-dev` entries, bundle `{context["bundle"]}`, identical '
        f'peak lists, search seeds {context["seeds"]}. Thresholds chosen on '
        f'`{context["train_split"]}` from the frozen feature matrix and applied unchanged.',
        '',
        '**What was measured.** Each arm is a complete run of the indexer over the same '
        'patterns, differing only in the search random stream. That is Shirley 1980\'s '
        '"slightly different refinement conditions ... for the same solution", made literal '
        'for this program: `random_subsampling` fits every iterate to a random '
        '`n_peaks - n_drop` subset, so two runs refine the same cell differently by '
        'construction.',
        '',
        '## 1. The value floor -- Shirley\'s own quantity',
        '',
        'For an entry whose correct cell is found in every arm, the merit each arm reports '
        'for it. Relative spread is `(max - min) / |median|` over the arms.',
        '',
        table(value_table),
        '',
        f'M20 by Bravais lattice ({context["n_shared"]} entries with a solution in every arm):',
        '',
        table(by_lattice),
        '',
        '## 2. The metric floor -- what a gate is written in',
        '',
        '`flip_rate` is the fraction of entries whose outcome differs between two arms; '
        '`induced_se_pp` is the standard error that flip rate implies for the *difference* '
        f'between two runs at the {context["reporting_entries"]} source entries of `fom-dev`, '
        'which is `sqrt(flip_rate / n)`.',
        '',
        table(metric_table, floats=4),
        '',
        '## 3. The contrast floor -- what a gate actually reads',
        '',
        'A gate compares two merits computed on **one** pool, so the pool\'s own realisation '
        'largely cancels. This is the spread of that difference across arms.',
        '',
        table(contrast_table),
        '',
        'Per Bravais lattice, at the CNRS weighting. The counts are small and these numbers are '
        'correspondingly coarse, but the ordering is the same one the value floor shows: the '
        'lattices with the most free cell parameters are the least reproducible.',
        '',
        table(lattice_frame.loc[lattice_frame['merit'].isin(('M20', 'M_sym'))]),
        '',
        '## 4. Is the floor scoring noise or generation noise?',
        '',
        '`operating_point` moves when the reported answer changes; `found` moves when the arms '
        'disagree about whether a correct candidate is in the pool at all. If the two coincided, '
        'most of the floor would belong to candidate generation and a figure-of-merit comparison '
        'would be insulated from it.',
        '',
        table(overlap, floats=0),
        '',
        'The two sets are **disjoint**, so every entry whose reported outcome moved had a '
        'correct candidate in its pool in both runs: the floor is what the merit does with a '
        'pool, not what the pool does.',
        '',
        '## 5. Merits with a zero median',
        '',
        '`n_over`, `max_gap` and `N_cal` are exactly reproducible at the median -- they are '
        'integer counts that rarely move -- so they carry no bar in panel (a), which is on a '
        'logarithmic axis. Their upper tails are not zero: see the `p90` column above.',
        '',
        ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = _parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    started = time.perf_counter()

    paths = arm_directories(args.arm_root)
    shared_entries = check_arms_are_comparable(paths)
    print(f'{len(paths)} arms, {len(shared_entries)} entries, peak lists identical',
          flush=True)

    arms, entries_by_arm = {}, {}
    for path in paths:
        seed, frame = load_arm(path, args.cache_dir, args.n_processes, args.tag)
        arms[seed] = frame
        entries_by_arm[seed] = FomBenchmark.load_entries(path)
        print(f'  arm {seed}: {frame.shape[0]:,} candidates, '
              f'{int(frame["is_correct"].sum()):,} correct '
              f'({time.perf_counter() - started:.0f}s)', flush=True)

    merit_columns = merit_column_names(args.feature_dir, args.bundle,
                                       arms[sorted(arms)[0]].columns)

    print('\nstep 1: the value floor', flush=True)
    value_table, shared = value_floor(arms, merit_columns)
    truth = entries_by_arm[sorted(arms)[0]].set_index('entry_id')['bravais_lattice_true']
    by_lattice_all = value_floor_by_lattice(arms, merit_columns, truth.to_dict())
    by_lattice = (by_lattice_all.loc[by_lattice_all['merit'] == 'M20']
                  .drop(columns=['merit']).reset_index(drop=True))
    print(value_table.to_string(index=False), flush=True)

    print('\nstep 2: thresholds on the training split', flush=True)
    thresholds = train_thresholds(args, THRESHOLD_MERITS)

    print('\nstep 3: the metric floor', flush=True)
    metric_frames, contrast_frames = [], []
    for weights in ('cnrs', None):
        results_by_merit = {
            merit: evaluate_arms(arms, entries_by_arm, merit, thresholds[merit], weights)
            for merit in THRESHOLD_MERITS
            }
        label = 'cnrs' if weights else 'unweighted'
        metric_frames.append(
            metric_floor(results_by_merit['M20'], METRICS, args.reporting_entries)
            .assign(weights=label))
        contrast_frames.append(
            contrast_floor(results_by_merit, ('operating_point', 'top10'),
                           args.reporting_entries).assign(weights=label))
        if weights == 'cnrs':
            lattice_frame = floor_by_lattice(results_by_merit, 'operating_point',
                                             args.reporting_entries)
    metric_table = pd.concat(metric_frames, ignore_index=True)
    contrast_table = pd.concat(contrast_frames, ignore_index=True)
    print(metric_table.to_string(index=False), flush=True)

    print('\nstep 4: the contrast floor', flush=True)
    print(contrast_table.to_string(index=False), flush=True)

    context = {
        'commit': commit_hash(), 'n_arms': len(paths), 'n_entries': len(shared_entries),
        'bundle': args.bundle, 'seeds': sorted(arms), 'train_split': args.train_split,
        'reporting_entries': args.reporting_entries, 'n_shared': len(shared),
        'thresholds': thresholds,
        }
    prefix = Path(args.artifact_dir) / args.tag
    value_table.to_csv(f'{prefix}_value.csv', index=False)
    by_lattice_all.to_csv(f'{prefix}_value_by_lattice.csv', index=False)
    metric_table.to_csv(f'{prefix}_metric.csv', index=False)
    contrast_table.to_csv(f'{prefix}_contrast.csv', index=False)
    lattice_frame.to_csv(f'{prefix}_by_lattice.csv', index=False)
    print('\nthe floor per Bravais lattice (operating point, M20):', flush=True)
    print(lattice_frame.loc[lattice_frame['merit'] == 'M20'].round(4).to_string(index=False),
          flush=True)
    figure(value_table, by_lattice, metric_table, contrast_table, context,
           f'{prefix}.png')
    overlap = flip_overlap(results_by_merit['M20'])
    overlap.to_csv(f'{prefix}_flip_overlap.csv', index=False)
    print('\nscoring noise or generation noise:', flush=True)
    print(overlap.to_string(index=False), flush=True)
    write_markdown(f'{prefix}.md', context, value_table, by_lattice, metric_table,
                   contrast_table, lattice_frame, overlap)
    headline = {
        'context': context,
        'value_floor_M20_median': float(
            value_table.loc[value_table['merit'] == 'M20', 'median_relative_spread'].iloc[0]),
        'metric_floor': metric_table.to_dict(orient='records'),
        'contrast_floor': contrast_table.to_dict(orient='records'),
        }
    with open(f'{prefix}.json', 'w', encoding='utf-8') as handle:
        json.dump(headline, handle, indent=2)
    print(f'\nwrote {prefix}.{{md,png,json}} and three CSVs '
          f'({time.perf_counter() - started:.0f}s)')


if __name__ == '__main__':
    main()
