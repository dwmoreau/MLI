"""Reducing a candidate pool to per-pattern outcomes, and those to comparable numbers.

The unit of measurement is a **pattern-condition**: one source crystal under one condition bundle.
A pool holds every candidate the indexer generated for it, across all fourteen Bravais lattices,
and the reduction answers one question per pattern-condition -- where in the pooled ranking the
best correct cell sits -- from which every reported metric follows.

Three properties this module exists to keep:

* **Ranking is pooled across lattices**, because that is the question the indexer answers. Ranking
  within a lattice and taking the best is a different and much easier problem.
* **The tie-break is total and label-independent.** Ties are broken by lattice order and then
  candidate id, never by whether a candidate is correct -- which would make every ranking
  optimistic -- and never by input order, which would make the numbers depend on the order shards
  happened to be read in.
* **Several scores are reduced in one pass over the pool**, each certified under its own name, so
  comparing two scores costs one pass rather than one per score.

"No correct candidate in the pool" is a generation failure and is reported separately from a
ranking failure. Aggregates are unweighted means over pattern-conditions, and every number is also
reported per Bravais lattice, because an aggregate alone is dominated by the easy lattices.
"""

import numpy as np
import pandas as pd

from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

# Rank metrics reported for every scope. `found` is the share with a correct cell anywhere in the
# pool -- the ceiling a perfect re-ranker reaches, and the line between a generation failure and a
# ranking one.
RANK_METRICS = ('found', 'top1', 'top5', 'top10', 'mrr')


def as_bool(values):
    """A boolean array from a column that may be object-typed or hold nulls."""
    return pd.Series(values).fillna(False).to_numpy(dtype=bool)


def _group_codes(entry_id, condition_bundle):
    """Integer codes for (entry_id, condition_bundle), and the keys in code order.

    The two fields are factorised separately and combined arithmetically rather than pasted into
    one string: this runs over millions of rows, factorising object tuples is far slower, and a
    string separator is not safe -- numpy truncates object string concatenation at a NUL byte.
    """
    entry_codes, entry_keys = pd.factorize(pd.Series(entry_id).astype(str).to_numpy(), sort=True)
    bundle_codes, bundle_keys = pd.factorize(
        pd.Series(condition_bundle).astype(str).to_numpy(), sort=True)
    stride = max(len(bundle_keys), 1)
    codes, composites = pd.factorize(entry_codes.astype(np.int64)*stride + bundle_codes, sort=True)
    keys = [(entry_keys[composite//stride], bundle_keys[composite % stride])
            for composite in composites]
    return codes.astype(np.int64), keys


def _count(group_code, flags, n_groups):
    return np.bincount(group_code, weights=np.asarray(flags, dtype=float),
                       minlength=n_groups).astype(np.int64)


# Position of each Bravais lattice in the canonical order, which is what breaks a score tie.
_LATTICE_POSITION = {lattice: position for position, lattice in enumerate(BRAVAIS_LATTICES)}


def lattice_order_of(bravais_lattice):
    """Position in the canonical lattice order, refusing anything not in it.

    An unrecognised lattice must not sort to one end: it would take every tie or lose every one,
    and the effect would look like a property of the ranking.
    """
    values = np.asarray(bravais_lattice)
    unknown = sorted(set(values.tolist()) - set(_LATTICE_POSITION))
    if unknown:
        raise ValueError(f'Unknown Bravais lattice in the candidate frame: {unknown}')
    return np.array([_LATTICE_POSITION[value] for value in values.tolist()], dtype=np.int64)


def _sort_key(values):
    """Descending by score, with NaN last.

    Non-finite scores follow numpy's ordering otherwise: +inf ranks first, which is right, because
    for M20 it means a vanishing residual. NaN carries no information and is sent to the end.
    """
    key = -np.asarray(values, dtype=np.float64).copy()
    key[np.isnan(values)] = np.inf
    return key


def _reduce_subset(entry_code, values, lattice, lattice_order, candidate_id, correct, degenerate,
                   subset_mask, n_groups):
    """One ranking pass over a subset of the pool, and the gathers that hang off it."""
    if subset_mask is not None:
        keep = np.flatnonzero(subset_mask)
        entry_code = entry_code[keep]
        values = values[keep]
        lattice = np.asarray(lattice)[keep]
        lattice_order = lattice_order[keep]
        candidate_id = candidate_id[keep]
        correct = correct[keep]
        degenerate = degenerate[keep]

    sort_key = _sort_key(values)
    order = np.lexsort((candidate_id, lattice_order, sort_key, entry_code))
    entry_sorted = entry_code[order]
    values_sorted = np.asarray(values)[order]
    lattice_sorted = np.asarray(lattice)[order]
    correct_sorted = correct[order]
    degenerate_sorted = degenerate[order]

    counts = np.bincount(entry_sorted, minlength=n_groups)
    starts = np.searchsorted(entry_sorted, np.arange(n_groups), side='left')
    rank_sorted = np.arange(entry_sorted.size) - starts[entry_sorted]

    columns = {'n_candidates': counts}

    present = counts > 0
    first = starts[present]
    score_top = np.full(n_groups, np.nan)
    top_is_correct = np.zeros(n_groups, dtype=bool)
    lattice_top = np.array([None]*n_groups, dtype=object)
    score_top[present] = values_sorted[first]
    top_is_correct[present] = correct_sorted[first]
    lattice_top[present] = lattice_sorted[first]
    columns['score_top'] = score_top
    columns['top_is_correct'] = top_is_correct
    columns['bravais_lattice_top'] = lattice_top

    # A crystal whose lattice is Mighell-Santoro degenerate has correct cells that are correct by
    # an accident of the lattice rather than by the search finding it, so they are excluded from
    # the headline outcome and counted separately.
    mask = correct_sorted & ~degenerate_sorted
    rank_best = np.full(n_groups, -1, dtype=np.int64)
    score_best = np.full(n_groups, np.nan)
    lattice_best = np.array([None]*n_groups, dtype=object)
    position_best = np.full(n_groups, -1, dtype=np.int64)
    hits = np.flatnonzero(mask)
    if hits.size:
        groups_hit = entry_sorted[hits]
        _, first_hit = np.unique(groups_hit, return_index=True)
        positions = hits[first_hit]
        winners = groups_hit[first_hit]
        rank_best[winners] = rank_sorted[positions]
        score_best[winners] = values_sorted[positions]
        lattice_best[winners] = lattice_sorted[positions]
        position_best[winners] = positions
    columns['rank_best_correct'] = rank_best
    columns['score_best_correct'] = score_best
    columns['bravais_lattice_best_correct'] = lattice_best
    columns['has_correct'] = rank_best >= 0
    columns['n_correct'] = _count(entry_sorted, mask, n_groups)
    columns['n_correct_incl_degenerate'] = _count(entry_sorted, correct_sorted, n_groups)
    # How many candidates share the winning score, so an outcome that hinged on a tie can be
    # counted rather than argued about.
    with np.errstate(invalid='ignore'):
        tied = values_sorted == score_best[entry_sorted]
    columns['n_ties_at_best_correct'] = _count(entry_sorted, tied, n_groups)

    # Who sits above the best correct cell, split by lattice. The dominant ranking failure of this
    # pipeline is cross-lattice, so a change that lifts correct cells within their own lattice buys
    # nothing if it lifts the other lattices' candidates equally. Measurable from a real run and
    # understated by a replay, which is why it is computed here.
    above = np.arange(entry_sorted.size) < position_best[entry_sorted]
    same_lattice = lattice_sorted == lattice_best[entry_sorted]
    columns['n_same_lattice_above_best_correct'] = _count(
        entry_sorted, above & same_lattice, n_groups)
    columns['n_other_lattice_above_best_correct'] = _count(
        entry_sorted, above & ~same_lattice, n_groups)
    return columns


def reduce_pool(frame, values, depths=('all', 'in_top_n')):
    """Rank a pool's candidates and reduce them to one row per (entry, condition bundle).

    `values` is the score to rank by, one per row of `frame`. Both reporting depths come from the
    same pass: `all` is the whole retained pool a re-ranker would see, and `in_top_n` is the top
    twenty per lattice that production hands to its final sort -- a strict superset of the printed
    list, because promotion and deduplication across lattices can only raise a correct cell's rank.
    """
    entry_code, keys = _group_codes(frame['entry_id'], frame['condition_bundle'])
    lattice = frame['bravais_lattice'].to_numpy()
    lattice_order = lattice_order_of(lattice)
    candidate_id = frame['candidate_id'].to_numpy(dtype=np.int64)
    correct = as_bool(frame['is_correct'])
    degenerate = (as_bool(frame['is_degenerate']) if 'is_degenerate' in frame
                  else np.zeros(len(frame), dtype=bool))
    in_top_n = as_bool(frame['in_top_n'])

    masks = {'all': None, 'in_top_n': in_top_n}
    columns = {}
    for depth in depths:
        reduced = _reduce_subset(entry_code, np.asarray(values, dtype=np.float64), lattice,
                                 lattice_order, candidate_id, correct, degenerate,
                                 masks[depth], n_groups=len(keys))
        for name, column in reduced.items():
            columns[f'{name}_{depth}'] = column
    out = pd.DataFrame(columns)
    out.insert(0, 'entry_id', [key[0] for key in keys])
    out.insert(1, 'condition_bundle', [key[1] for key in keys])
    return out


def reduce_many(frame, scores, depths=('all', 'in_top_n')):
    """Reduce several scores in one pass over the pool, each under its own name.

    `scores` maps a name to a column of `frame` or to an array. One pass rather than one per score
    is the difference between comparing five rankings and re-reading a hundred gigabytes five
    times. Each score is reduced independently, so no score inherits another's ranking.
    """
    reductions = {}
    for name, values in scores.items():
        column = frame[values].to_numpy() if isinstance(values, str) else np.asarray(values)
        if column.shape[0] != frame.shape[0]:
            raise ValueError(
                f'score {name!r} has {column.shape[0]} values for {frame.shape[0]} candidates')
        reductions[name] = reduce_pool(frame, column, depths=depths)
    return reductions


def derive_flags(per_entry, depth='all', top_n=10):
    """The per-pattern outcome flags, for one reporting depth.

    Split from the reduction because these are the only quantities that depend on `top_n`, so a
    sweep over it re-reads nothing.
    """
    if depth not in ('all', 'in_top_n'):
        raise ValueError(f"depth must be 'all' or 'in_top_n', got {depth!r}")
    frame = per_entry.copy()
    for name in ('rank_best_correct', 'score_best_correct', 'has_correct', 'n_correct',
                 'bravais_lattice_best_correct', 'n_ties_at_best_correct', 'score_top',
                 'top_is_correct', 'bravais_lattice_top', 'n_candidates',
                 'n_same_lattice_above_best_correct', 'n_other_lattice_above_best_correct'):
        frame[name] = frame[f'{name}_{depth}']

    rank = frame['rank_best_correct'].to_numpy()
    found = frame['has_correct'].to_numpy(dtype=bool)
    frame['found'] = found
    frame['top1'] = found & (rank < 1)
    frame['top5'] = found & (rank < 5)
    frame['top10'] = found & (rank < top_n)
    frame['reciprocal_rank'] = np.where(found, 1.0/(np.maximum(rank, 0) + 1.0), 0.0)
    # The pool held a correct cell and the ranking failed to surface it: a ranking loss, as
    # distinct from a generation loss, which is `~found`.
    frame['lost_to_ranking'] = found & ~frame['top10'].to_numpy(dtype=bool)
    frame['beaten_cross_lattice'] = (
        frame['lost_to_ranking'].to_numpy(dtype=bool)
        & (frame['n_other_lattice_above_best_correct'].to_numpy() > 0))
    return frame


def unweighted_mean(frame, column):
    """The plain mean of a per-pattern flag. One aggregate, unweighted over lattices."""
    values = frame[column].to_numpy(dtype=np.float64)
    if values.size == 0:
        return np.nan
    return float(np.nanmean(values))


def summarise(flagged, scope='aggregate'):
    """One row of metrics for one scope."""
    row = {'scope': scope, 'n_entries': int(flagged.shape[0])}
    if flagged.shape[0] == 0:
        row.update({name: np.nan for name in RANK_METRICS})
        return row
    for name in RANK_METRICS:
        row[name] = unweighted_mean(flagged, 'reciprocal_rank' if name == 'mrr' else name)
    row['n_clusters'] = int(flagged['cluster'].nunique()) if 'cluster' in flagged else np.nan
    return row


def scope_masks(flagged, entries=None):
    """Every scope a result is reported at: the aggregate, each lattice, each condition bundle.

    An aggregate alone is dominated by the easy lattices and will make two rankings look alike, so
    a per-lattice row is produced for every claim rather than on request.
    """
    masks = {'aggregate': np.ones(flagged.shape[0], dtype=bool)}
    if entries is not None and 'bravais_lattice_true' in entries:
        merged = flagged.merge(entries[['entry_id', 'condition_bundle', 'bravais_lattice_true']],
                               on=['entry_id', 'condition_bundle'], how='left')
        lattice = merged['bravais_lattice_true'].to_numpy()
        for value in BRAVAIS_LATTICES:
            mask = lattice == value
            if mask.any():
                masks[f'bravais_lattice={value}'] = mask
    bundles = flagged['condition_bundle'].to_numpy()
    for value in sorted(set(bundles.tolist())):
        masks[f'condition_bundle={value}'] = bundles == value
    return masks


def summarise_by_scope(flagged, entries=None):
    """A metric row per scope, aggregate first."""
    rows = []
    for scope, mask in scope_masks(flagged, entries=entries).items():
        rows.append(summarise(flagged.loc[mask], scope=scope))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Comparing two arms
# ---------------------------------------------------------------------------------------------

def mcnemar(a, b):
    """The paired test for two binary outcomes over the same patterns.

    Only the disagreements carry information, which is the point: two rankings agreeing on nine
    tenths of a benchmark differ only where they disagree, and pooling the agreements into the
    test would dilute exactly the signal being measured.
    """
    from scipy.stats import binomtest

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f'paired arms must cover the same patterns: {a.shape} against {b.shape}')
    b_wins = int(np.count_nonzero(~a & b))
    a_wins = int(np.count_nonzero(a & ~b))
    discordant = a_wins + b_wins
    p_value = (1.0 if discordant == 0
               else float(binomtest(b_wins, discordant, 0.5).pvalue))
    return {'n_pairs': int(a.size), 'a_only': a_wins, 'b_only': b_wins,
            'n_discordant': discordant, 'delta': float(b.mean() - a.mean()),
            'p_value': p_value}


def _bootstrap_replicates(clusters, n_bootstrap, seed):
    """Cluster draws reused by every metric, so intervals are comparable across them.

    Resampling is over **source crystals**, not over pattern-conditions: one crystal under nine
    conditions is one draw, not nine, and treating it as nine would understate every interval.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    return rng.integers(0, unique.size, size=(n_bootstrap, unique.size))


def paired_delta_ci(a, b, clusters, n_bootstrap=1000, seed=12345):
    """A bootstrap interval for `mean(b) - mean(a)`, resampling source crystals."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique, codes = np.unique(clusters, return_inverse=True)
    replicates = _bootstrap_replicates(clusters, n_bootstrap, seed)
    order = np.argsort(codes, kind='stable')
    codes_sorted = codes[order]
    starts = np.searchsorted(codes_sorted, np.arange(unique.size), side='left')
    ends = np.searchsorted(codes_sorted, np.arange(unique.size), side='right')
    difference = b - a
    per_cluster_sum = np.add.reduceat(difference[order], starts) if unique.size else np.zeros(0)
    per_cluster_n = (ends - starts).astype(np.float64)
    deltas = np.empty(replicates.shape[0])
    for index, draw in enumerate(replicates):
        deltas[index] = per_cluster_sum[draw].sum() / max(per_cluster_n[draw].sum(), 1.0)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
