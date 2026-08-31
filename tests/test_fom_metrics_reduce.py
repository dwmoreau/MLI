"""The reduce/analyse split: one pool pass answers every threshold, on any machine.

`evaluate` is `reduce_to_per_entry` followed by `summarise_per_entry`, and the seam matters
operationally rather than aesthetically. Benchmark B is 122 GB and lives on `$SCRATCH`; the record,
the figures and the paper live on a laptop with ~14 GB free. The reduction is one row per
(entry, condition) -- a few hundred megabytes for the whole pool -- and it carries everything the
downstream half reads. So the cluster reduces and the laptop analyses, through the same code.

That only holds if the reduction really is a sufficient statistic. These tests pin it:

  * composing the two halves reproduces `evaluate` exactly, metric for metric;
  * a threshold sweep off one reduction matches independent `evaluate` calls at each threshold;
  * a reduction certified at one depth refuses to be summarised at a deeper one.
"""

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomMetrics

from tests.test_fom_metrics import _tiny


ROWS = [
    ('A', 'oP', 9.0, True), ('A', 'mP', 7.0, False), ('A', 'aP', 12.0, False),
    ('B', 'oP', 4.0, False), ('B', 'mP', 8.0, True), ('B', 'aP', 3.0, False),
    ('C', 'oP', 15.0, True), ('C', 'mP', 2.0, False), ('C', 'aP', 6.0, False),
    ('D', 'oP', 1.0, False), ('D', 'mP', 5.0, False), ('D', 'aP', 11.0, False),
    ]


def _pool():
    return _tiny(ROWS, entries={'A': 'oP', 'B': 'mP', 'C': 'oP', 'D': 'aP'})


def _metrics_of(result):
    return result.aggregate.iloc[0].to_dict()


def test_composing_the_two_halves_reproduces_evaluate():
    """The seam is exact, not merely close -- `evaluate` is these two calls and nothing else."""
    candidates, entries = _pool()
    direct = FomMetrics.evaluate(candidates, score='score', entries=entries, threshold=5.0,
                                 n_bootstrap=0)

    reduced, calibration_rows, meta = FomMetrics.reduce_to_per_entry(
        candidates, score='score', entries=entries)
    composed = FomMetrics.summarise_per_entry(reduced, meta, threshold=5.0, n_bootstrap=0,
                                              calibration_rows=calibration_rows)

    left, right = _metrics_of(direct), _metrics_of(composed)
    assert set(left) == set(right)
    for name, value in left.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            np.testing.assert_allclose(right[name], value, rtol=0, atol=0,
                                       err_msg=f'{name} differs across the seam')
        else:
            assert right[name] == value
    # The entry digest is what `mcnemar` pairs on, so a difference here would silently prevent
    # a reduced result from being compared against a directly evaluated one.
    assert composed.meta['entry_digest'] == direct.meta['entry_digest']


def test_one_reduction_answers_every_threshold():
    """The property the NERSC split rests on: sweep thresholds without touching the pool again."""
    candidates, entries = _pool()
    reduced, _, meta = FomMetrics.reduce_to_per_entry(candidates, score='score', entries=entries)

    for threshold in (2.0, 5.0, 8.0, 12.0):
        swept = FomMetrics.summarise_per_entry(reduced, meta, threshold=threshold, n_bootstrap=0)
        fresh = FomMetrics.evaluate(candidates, score='score', entries=entries,
                                    threshold=threshold, n_bootstrap=0)
        for name in ('operating_point', 'top1', 'top10', 'reported', 'false_positive',
                     'threshold_only', 'found', 'ceiling_reranker', 'ceiling_rescorer'):
            np.testing.assert_allclose(
                swept.aggregate.iloc[0][name], fresh.aggregate.iloc[0][name], rtol=0, atol=0,
                err_msg=f'{name} differs at threshold {threshold}')


def test_the_reduction_survives_a_round_trip_through_parquet(tmp_path):
    """It has to cross a machine boundary, so it has to survive being written and read."""
    pytest.importorskip('pyarrow')
    candidates, entries = _pool()
    reduced, _, meta = FomMetrics.reduce_to_per_entry(candidates, score='score', entries=entries)

    path = tmp_path/'reduced.parquet'
    reduced.to_parquet(path, index=False)
    restored = pd.read_parquet(path)

    here = FomMetrics.summarise_per_entry(reduced, meta, threshold=5.0, n_bootstrap=0)
    there = FomMetrics.summarise_per_entry(restored, meta, threshold=5.0, n_bootstrap=0)
    np.testing.assert_allclose(there.aggregate.iloc[0]['operating_point'],
                               here.aggregate.iloc[0]['operating_point'], rtol=0, atol=0)
    assert there.meta['entry_digest'] == here.meta['entry_digest']


def test_summarising_deeper_than_the_reduction_was_certified_refuses():
    """A depth check evaded after the fact would be worse than no depth check.

    `rank_exactness` runs inside the reduce half against the pool's retention depth K. Asking for
    a deeper `top_n` in the summarise half would quietly report a rank the pool cannot answer --
    the C2-F-077 failure, reintroduced through the back door.
    """
    candidates, entries = _pool()
    # A subsampled pool only certifies a rank for a merit the subsampler ranked on, so the score
    # has to be one of those seven for the reduction to be admitted at all.
    candidates = candidates.rename(columns={'score': 'M20'})
    reduced, _, meta = FomMetrics.reduce_to_per_entry(
        candidates, score='M20', entries=entries, subsample_top_k=10)
    assert meta['subsampled'] is True
    with pytest.raises(ValueError, match='certified for top_n'):
        FomMetrics.summarise_per_entry(reduced, meta, top_n=50, n_bootstrap=0)


def test_a_fully_retained_reduction_may_be_summarised_at_any_depth():
    """The refusal is a property of a *subsampled* pool, not a blanket rule."""
    candidates, entries = _pool()
    reduced, _, meta = FomMetrics.reduce_to_per_entry(candidates, score='score', entries=entries)
    assert meta['subsampled'] is False
    FomMetrics.summarise_per_entry(reduced, meta, top_n=50, n_bootstrap=0)
