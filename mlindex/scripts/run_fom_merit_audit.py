"""Which figures of merit does campaign 2 carry, and which are noise? (The merit audit.)

DWMM's brief for S12 is that there are too many traditional figures of merit in the combiner and
that campaign 2 should "just use what was effective and discard the rest as noise". This script is
the evidence behind that cut. It runs ahead of S02 rather than inside a numbered step, because
S02 (the port), S09 (the evaluation) and S12 (the combiner) each need the same list and should not
each re-decide it.

Campaign 1 evaluated 22 distinct merits. Its own tables show 23 rows, but `M20 @ 10 (literature)`
is M20 at de Wolff's published threshold rather than a separate merit (C2-F-006).

Four questions, each answered from per-entry outcomes rather than from an importance table --
PROTOCOL section 8 forbids cutting a feature on permutation importance, which campaign 1 proved
can be wrong by a factor of 1800 on this very feature set:

  1. Redundancy. Which merits carry a per-entry outcome vector another merit already carries?
  2. Usefulness. Which merits rank below a constant score, i.e. below the tie-break floor?
  3. Sufficiency. How few merits reach the union oracle over all 22?
  4. Where the value sits. Is a merit's contribution concentrated on lattices that are already
     easy? An aggregate hides this, and the campaign's gains live in aP, mP and mC.

Aggregates here are UNWEIGHTED, per PROTOCOL section 3 rule 6. Campaign 1's published tables are
weighted to the sealed benchmark's lattice distribution, which campaign 2 forbids; the first
output quantifies how much that choice moved each merit.

    python mlindex/scripts/run_fom_merit_audit.py --artifact-dir docs/fom_campaign2/artifacts

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

# Campaign 1's zoo evaluation, read as artefacts rather than as the prose that cites them.
CAMPAIGN1 = os.path.join('docs', 'fom_campaign1', 'artifacts')
PER_ENTRY = 'S06_zoo_per_entry.parquet'
MAIN_TABLE = 'S06_zoo_main_table.csv'
COMBINER_TABLE = 'S08_combiner_main_table.csv'

# A merit that is not a merit: M20 evaluated at de Wolff's published threshold. It shares M20's
# ranking exactly and only differs in where the cut is placed, so it is excluded from every
# ranking statistic here (C2-F-006).
NOT_A_MERIT = ('M20 @ 10 (literature)',)

# The lattices where campaign 2's gains are expected to live. Kept explicit so the per-lattice
# table always reports them, whatever else is dropped.
LOW_SYMMETRY = ('aP', 'mP', 'mC')


def load_inputs(campaign1_dir):
    """The three artefacts the audit reads. All are campaign 1 outputs and none are in git."""
    per_entry = pd.read_parquet(os.path.join(campaign1_dir, PER_ENTRY))
    main_table = pd.read_csv(os.path.join(campaign1_dir, MAIN_TABLE))
    combiner = pd.read_csv(os.path.join(campaign1_dir, COMBINER_TABLE))
    return per_entry, main_table, combiner


def outcome_matrix(per_entry, metric='top10', subset=None):
    """One column per merit, one row per (entry, condition bundle), holding a 0/1 outcome.

    The pivot is over (entry_id, condition_bundle) rather than entry_id alone because an entry
    appears once per bundle and the bundles are different noise realisations of it. Splitting is
    by source entry everywhere else in this campaign; here every merit sees the same rows, so the
    comparison is paired by construction and the row definition only has to be consistent.
    """
    frame = per_entry if subset is None else per_entry[subset]
    pivot = frame.pivot_table(
        index=['entry_id', 'condition_bundle'], columns='merit', values=metric, aggfunc='first'
        )
    pivot = pivot.drop(columns=[c for c in NOT_A_MERIT if c in pivot.columns])
    return pivot.fillna(0).astype(int)


def identical_groups(pivot):
    """Merits whose per-entry outcome vectors are equal bit for bit.

    This is the strongest form of redundancy available: not a correlation that happens to be high
    on this pool, but the same answer on every single entry. M_nn against M20 is the case that
    matters, and it is not a coincidence of the sample -- Oishi-Tomiyasu's 2021 nearest-neighbour
    form reduces analytically to de Wolff's, which tests/test_fom_literature.py already asserts.
    """
    groups = {}
    for merit in pivot.columns:
        groups.setdefault(tuple(pivot[merit].values), []).append(merit)
    return [names for names in groups.values() if len(names) > 1]


def unique_wins(pivot):
    """Entries a merit gets into the top ten that no other merit in the pool does.

    Zero unique wins does not by itself condemn a merit -- M20 scores zero, because everything it
    finds M_sym also finds -- but a merit with no unique wins AND a rate below the tie-break floor
    is contributing nothing that is not already there.
    """
    values = pivot.values
    counts = {}
    for index, merit in enumerate(pivot.columns):
        others = np.delete(values, index, axis=1).max(axis=1)
        counts[merit] = int(((values[:, index] == 1) & (others == 0)).sum())
    return pd.Series(counts, name='unique_wins')


def greedy_oracle(pivot, max_steps=10, stop_gain=0.0005):
    """Forward selection against the union oracle -- the ceiling a perfect selector would reach.

    This is an upper bound on any combiner over the selected merits, not a prediction of one. It
    answers the sufficiency question directly: if k merits already reach the union of all 22, the
    remaining merits have no entry left to contribute, whatever an importance table says about
    them.
    """
    values = pivot.values
    merits = list(pivot.columns)
    union = values.max(axis=1).mean()
    covered = np.zeros(len(values), dtype=int)
    chosen, rows = [], []
    for step in range(max_steps):
        best = None
        for index, merit in enumerate(merits):
            if merit in chosen:
                continue
            reached = np.maximum(covered, values[:, index]).mean()
            if best is None or reached > best[0]:
                best = (reached, merit, index)
        reached, merit, index = best
        gain = reached - covered.mean()
        chosen.append(merit)
        covered = np.maximum(covered, values[:, index])
        rows.append({
            'step': step + 1,
            'merit': merit,
            'oracle': reached,
            'gain': gain,
            'entries_gained': int(round(gain*len(values))),
            'fraction_of_union': reached/union if union else np.nan,
            })
        if gain < stop_gain:
            break
    return pd.DataFrame(rows), union


def correlated_pairs(pivot, threshold=0.85):
    """Pairs whose outcome vectors agree closely enough to suspect one is carrying the other."""
    correlation = pivot.corr()
    rows = []
    for left, right in itertools.combinations(pivot.columns, 2):
        value = correlation.loc[left, right]
        if value > threshold:
            rows.append({'merit_a': left, 'merit_b': right, 'phi': value})
    return pd.DataFrame(rows).sort_values('phi', ascending=False), correlation


def build_summary(per_entry, main_table, floor):
    """One row per merit: what campaign 1 published, what the rule now in force says, and both
    redundancy measures. The published/unweighted pair is the point -- they are different numbers
    and the record only carries the first."""
    pivot = outcome_matrix(per_entry)
    hard_pivot = outcome_matrix(per_entry, subset=per_entry['is_hard'])
    published = main_table.set_index('merit')
    unweighted = per_entry.groupby('merit')[['top1', 'top10', 'operating_point']].mean()

    summary = pd.DataFrame(index=pivot.columns)
    summary['top10_published_weighted'] = published['top10'].reindex(summary.index)
    summary['top10_unweighted'] = unweighted['top10'].reindex(summary.index)
    summary['weighting_delta_pp'] = (
        summary['top10_unweighted'] - summary['top10_published_weighted']
        )*100
    summary['operating_point_published'] = published['operating_point'].reindex(summary.index)
    summary['top10_hard_unweighted'] = (
        per_entry[per_entry['is_hard']].groupby('merit')['top10'].mean().reindex(summary.index)
        )
    summary['unique_wins'] = unique_wins(pivot).reindex(summary.index)
    summary['unique_wins_hard'] = unique_wins(hard_pivot).reindex(summary.index)
    summary['phi_with_M20'] = pivot.corr()['M20'].reindex(summary.index)
    # The tie-break floor is a property of the population, not of the metric: a constant score
    # already reaches it because ties break cubic-first and the dominant failure is symmetry
    # lowering. Compared on the published weighted basis, which is the basis the floor is on.
    summary['below_tie_break_floor'] = summary['top10_published_weighted'] < floor
    return summary.sort_values('top10_unweighted', ascending=False), pivot, hard_pivot


def per_lattice_table(per_entry):
    """Top-10 by Bravais lattice. The aggregate hides that the strongest-looking merits earn it
    on high-symmetry lattices that are already solved, and are below M20 on mP and mC."""
    table = per_entry[~per_entry['merit'].isin(NOT_A_MERIT)].pivot_table(
        index='merit', columns='bravais_lattice', values='top10', aggfunc='mean'
        )
    return table.sort_values(list(LOW_SYMMETRY), ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Audit campaign 1 merits for redundancy and usefulness, and recommend the set '
            'campaign 2 carries. Reads campaign 1 artefacts; writes CSVs for the results doc.'
            ),
        )
    parser.add_argument(
        '--campaign1-dir', default=CAMPAIGN1,
        help='directory holding campaign 1 zoo and combiner artefacts',
        )
    parser.add_argument(
        '--artifact-dir', default=os.path.join('docs', 'fom_campaign2', 'artifacts'),
        help='where to write the audit CSVs',
        )
    parser.add_argument(
        '--prefix', default='S00_merit_audit', help='basename prefix for the outputs',
        )
    parser.add_argument(
        '--correlation-threshold', type=float, default=0.85,
        help='report merit pairs above this phi correlation as candidate duplicates',
        )
    args = parser.parse_args()

    per_entry, main_table, combiner = load_inputs(args.campaign1_dir)
    floor_row = combiner[combiner['arm'] == 'constant (tie-break floor)']
    floor = float(floor_row['top10'].iloc[0])
    random_floor = float(combiner[combiner['arm'] == 'uniform random']['top10'].iloc[0])

    summary, pivot, hard_pivot = build_summary(per_entry, main_table, floor)
    greedy, union = greedy_oracle(pivot)
    greedy_hard, union_hard = greedy_oracle(hard_pivot, max_steps=6, stop_gain=0.002)
    pairs, _ = correlated_pairs(pivot, args.correlation_threshold)
    lattice = per_lattice_table(per_entry)

    os.makedirs(args.artifact_dir, exist_ok=True)
    outputs = {
        'summary': summary,
        'greedy': greedy,
        'greedy_hard': greedy_hard,
        'correlated_pairs': pairs,
        'per_lattice': lattice,
        }
    for name, frame in outputs.items():
        path = os.path.join(args.artifact_dir, f'{args.prefix}_{name}.csv')
        frame.to_csv(path, encoding='utf-8')
        print(f'wrote {path}')

    print(f'\nmerits audited: {len(summary)}  entries x bundles: {len(pivot)}')
    print(f'tie-break floor (constant score): {floor:.4f}   uniform random: {random_floor:.4f}')
    print(f'union oracle over all merits: {union:.4f}   hard stratum: {union_hard:.4f}')
    print('\nidentical outcome vectors:')
    for names in identical_groups(pivot):
        print(f'  {names}')
    print('\ngreedy forward selection:')
    print(greedy.to_string(index=False))
    print('\nbelow the tie-break floor, with zero unique wins:')
    condemned = summary[summary['below_tie_break_floor'] & (summary['unique_wins'] == 0)]
    print('  ' + ', '.join(condemned.index))
    print('\nlow-symmetry lattices, where the campaign gains live:')
    print((lattice[list(LOW_SYMMETRY)]*100).round(1).to_string())


if __name__ == '__main__':
    main()
