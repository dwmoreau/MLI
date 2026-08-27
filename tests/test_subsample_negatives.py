"""S06 -- the negative subsampler, and the one property that makes it safe to use.

Negative subsampling is what makes a 20 000-crystal pool fit its disk budget, and it is also the
easiest way to quietly change what the benchmark measures. Three things have to hold.

* **No correct candidate is ever dropped.** The base rate is under 1 %, they are the whole signal,
  and dropping one turns a ranking failure into what looks like a generation failure -- a
  distinction METRICS.md keeps in separate buckets for exactly this reason.
* **Rank metrics stay exact to depth K**, for every merit the pool reports, which is what the
  top-K union buys over a plain sample.
* **The weights reproduce full-pool aggregates.** Without `sampling_weight` every fit on the pool
  is biased by whatever the retention rate happened to be; S07's gate 6 is the version of this
  test that runs on the real pool.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training.FomBenchmark import subsample_negatives


def _pool(n_per_group=400, n_groups=6, correct_rate=0.01, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for group in range(n_groups):
        n = n_per_group
        frames.append(pd.DataFrame({
            'entry_id': f'E{group // 2:03d}',
            'condition_bundle': 'c2_error1_cont0' if group % 2 else 'c2_error2_cont0',
            'bravais_lattice': ['aP', 'mP', 'oP'][group % 3],
            'candidate_id': np.arange(n),
            'M20': rng.gamma(2.0, 2.0, n),
            # A second merit that agrees with M20 only loosely, so the union is a real union.
            'M_sym': rng.gamma(2.0, 2.0, n),
            'is_correct': rng.random(n) < correct_rate,
            }))
    return pd.concat(frames, ignore_index=True)


def test_every_correct_candidate_survives():
    pool = _pool(correct_rate=0.05)
    thinned = subsample_negatives(pool, merit_columns=('M20',), top_k=20, negative_rate=0.02)
    assert int(pool['is_correct'].sum()) > 0, 'fixture has no positives to protect'
    assert int(thinned['is_correct'].sum()) == int(pool['is_correct'].sum())
    assert (thinned.loc[thinned['is_correct'], 'retained_reason'] == 'correct').all()
    assert (thinned.loc[thinned['is_correct'], 'sampling_weight'] == 1.0).all()


@pytest.mark.parametrize('top_k', [5, 20, 100])
def test_rank_metrics_are_exact_to_depth_k(top_k):
    # The property K is for. Every merit's top K must be present, row for row, so a top-N metric
    # for any N <= K computed on the thinned pool equals the one computed on the full pool.
    pool = _pool()
    thinned = subsample_negatives(pool, merit_columns=('M20', 'M_sym'), top_k=top_k,
                                  negative_rate=0.01)
    keys = ['entry_id', 'condition_bundle', 'bravais_lattice']
    for merit in ('M20', 'M_sym'):
        for key, group in pool.groupby(keys, sort=False):
            expected = set(group.nlargest(top_k, merit)['candidate_id'])
            kept = thinned
            for name, value in zip(keys, key):
                kept = kept[kept[name] == value]
            assert expected <= set(kept['candidate_id']), (merit, key)


def test_the_weights_reproduce_the_full_pool_count():
    # The check S07's gate 6 runs on the real pool: a weighted count over the thinned pool
    # estimates the full pool's size. Sampling is random, so this is a tolerance, not equality --
    # and the tolerance is what the standard error of a 5 % Bernoulli sample allows.
    pool = _pool(n_per_group=4000, n_groups=6, correct_rate=0.005)
    thinned = subsample_negatives(pool, merit_columns=('M20',), top_k=50, negative_rate=0.05)
    estimated = float(thinned['sampling_weight'].sum())
    assert abs(estimated - pool.shape[0]) / pool.shape[0] < 0.05
    assert thinned.shape[0] < pool.shape[0], 'nothing was thinned; the fixture is wrong'


def test_retention_is_reproducible_and_pool_keyed():
    # Same pool, same rows -- whichever shard it lands in and however the frame is ordered.
    pool = _pool()
    first = subsample_negatives(pool, top_k=20, negative_rate=0.1)
    second = subsample_negatives(pool, top_k=20, negative_rate=0.1)
    pd.testing.assert_frame_equal(first, second)

    shuffled = pool.sample(frac=1.0, random_state=17).reset_index(drop=True)
    third = subsample_negatives(shuffled, top_k=20, negative_rate=0.1)
    key = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']
    assert (set(map(tuple, first[key].to_numpy()))
            == set(map(tuple, third[key].to_numpy())))


def test_rate_one_keeps_everything_but_still_writes_the_bookkeeping():
    pool = _pool()
    thinned = subsample_negatives(pool, top_k=10, negative_rate=1.0)
    assert thinned.shape[0] == pool.shape[0]
    assert set(thinned['retained_reason']) <= {'correct', 'top_k', 'sampled'}
    assert (thinned['sampling_weight'] == 1.0).all()


def test_a_missing_merit_column_raises():
    # A quiet fall-back to M20 would advertise exactness at depth K for a merit that was never
    # ranked on, which is the kind of claim nothing downstream could check.
    with pytest.raises(ValueError):
        subsample_negatives(_pool(), merit_columns=('M20', 'not_a_merit'))
    with pytest.raises(ValueError):
        subsample_negatives(_pool(), negative_rate=0.0)


def test_precedence_is_correct_then_top_k_then_sampled():
    pool = _pool(n_per_group=100, n_groups=2, correct_rate=0.0)
    # Force the single highest-M20 row of one pool to be correct, so it qualifies both ways.
    top_row = pool['M20'].idxmax()
    pool.loc[top_row, 'is_correct'] = True
    thinned = subsample_negatives(pool, top_k=10, negative_rate=0.01)
    match = thinned[(thinned['candidate_id'] == pool.loc[top_row, 'candidate_id'])
                    & (thinned['entry_id'] == pool.loc[top_row, 'entry_id'])
                    & (thinned['bravais_lattice'] == pool.loc[top_row, 'bravais_lattice'])]
    assert (match['retained_reason'] == 'correct').all()
