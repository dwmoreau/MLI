"""Gate condition 4: a subset of Benchmark B re-runs bit-identically.

This is the property campaign 1 could not have at all. Its R17 records that "a pool is
reproducible only at a fixed (n_pools, pool_size, shard, n_shards, seed)", and that single fact
forced a within-run restriction on every result in its final phase.

It is checked here rather than in `run_fom_dump_gate.py` because it is not a property of a
finished pool -- it needs a second generation run to compare against. The gate script's docstring
points here.

**Run at `--pool-size 2`, deliberately.** C2-F-047 is that S05's gates ran at `--pool-size 1`,
where the manager is the only rank, so multiprocessing -- the mode the benchmark actually
generates in -- was never exercised, and both known reproducibility defects lived there.

Skipped without the model set, which is a 545 MB download the test suite does not require.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST = os.path.join('docs', 'fom_campaign2', 'artifacts', 'S06_split_manifest.parquet')


def _models_present():
    try:
        from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
        return os.path.isdir(os.path.join(_resolve_models_dir(), 'cubic_1', 'integral_filter'))
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not os.path.exists(MANIFEST),
                       reason='docs/ is git-ignored and absent on this machine'),
    pytest.mark.skipif(not _models_present(), reason='model set not downloaded'),
    ]


def _entry_ids(n):
    manifest = pd.read_parquet(MANIFEST)
    return sorted(manifest.loc[manifest['bravais_lattice'] == 'cP', 'identifier'])[:n]


def _generate(out_dir, entry_ids, n_pools, pool_size, tmp_path, extra=()):
    """One pool, written to `out_dir`. cP only: its pools are small, so this stays a test."""
    from mlindex.scripts.run_fom_dump import main

    ids_path = tmp_path / f'ids_{os.path.basename(out_dir)}.csv'
    pd.DataFrame({'identifier': entry_ids}).to_csv(ids_path, index=False)
    main(['--condition', 'nominal',
          '--split-manifest', MANIFEST,
          '--bravais-lattices', 'cP',
          '--entry-ids-file', str(ids_path),
          '--n-pools', str(n_pools), '--pool-size', str(pool_size),
          '--predownsample-entries', '0',
          '--out-dir', str(out_dir)] + list(extra))
    import glob
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(f'{out_dir}/candidates_*.parquet'))]
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(['entry_id', 'bravais_lattice', 'candidate_id'],
                             kind='stable', ignore_index=True)


def _entries(out_dir):
    """The entry table of a generated pool, keyed and ordered so two runs are comparable."""
    import glob
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(f'{out_dir}/entries_*.parquet'))]
    return pd.concat(frames, ignore_index=True).sort_values(
        'entry_id', kind='stable', ignore_index=True)


def _differing_columns(left, right):
    """Column names that differ. Object columns hold arrays, so `.equals` is not enough."""
    assert list(left.columns) == list(right.columns)
    differing = []
    for column in left.columns:
        if left[column].dtype != object:
            same = left[column].equals(right[column])
        else:
            same = all(
                np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
                if isinstance(a, np.ndarray) else (a == b) or (pd.isna(a) and pd.isna(b))
                for a, b in zip(left[column], right[column]))
        if not same:
            differing.append(column)
    return differing


def test_a_subset_reproduces_the_full_run_candidate_for_candidate(tmp_path):
    """The gate condition itself: three entries, then the middle two alone."""
    full_ids = _entry_ids(3)
    subset_ids = full_ids[:2]

    full = _generate(tmp_path / 'full', full_ids, 1, 2, tmp_path)
    subset = _generate(tmp_path / 'subset', subset_ids, 1, 2, tmp_path)

    restricted = full[full['entry_id'].isin(subset_ids)].reset_index(drop=True)
    assert restricted.shape[0] > 0, 'the full run produced no candidates for the subset entries'
    assert restricted.shape == subset.shape, (
        f'subset has {subset.shape[0]} rows against {restricted.shape[0]} in the full run')
    assert not _differing_columns(restricted, subset)


def test_the_number_of_pools_does_not_change_the_pool(tmp_path):
    """R17 says it does. For `n_pools` it no longer does, and this stops that regressing.

    NOTE the axis. This varies `--n-pools` at a FIXED `--pool-size`, which is the only half of
    "topology" the pool is invariant to -- see the test below.
    """
    ids = _entry_ids(2)
    one_pool = _generate(tmp_path / 'one', ids, 1, 2, tmp_path)
    two_pools = _generate(tmp_path / 'two', ids, 2, 2, tmp_path)
    assert one_pool.shape == two_pools.shape
    assert not _differing_columns(one_pool, two_pools)


def test_pool_size_DOES_change_the_pool_and_is_part_of_its_identity(tmp_path):
    """The other half of the axis, and it is not invariant. C2-F-069.

    `_reseed_for_pattern` keys on (q2 digest, Bravais lattice, **rank**), and the number of ranks
    IS `pool_size` -- so changing it is a different stochastic search, exactly as changing the
    backend is (C2-F-009). Measured on six entries: 1x2, 2x2 and 3x2 all give 1 989 candidates and
    15 correct; 1x3 gives 2 054 and 1x4 gives 2 071 and 14 correct.

    This is asserted rather than fixed because the seeds are what they are and the benchmark is
    generated at one pool size. What the assertion buys is that nobody reads
    "bit-identical across pool topology" as covering this axis -- it does not, and an earlier
    version of this file tested only `n_pools` while the record claimed both.
    """
    ids = _entry_ids(2)
    two = _generate(tmp_path / 'ps2', ids, 1, 2, tmp_path)
    four = _generate(tmp_path / 'ps4', ids, 1, 4, tmp_path)
    assert two.shape != four.shape or _differing_columns(two, four), (
        'pool_size no longer changes the pool. If the seeding was made rank-independent that is '
        'an improvement -- update C2-F-069 and the driver guard rather than deleting this test.')


def test_a_requeue_at_a_different_pool_size_is_refused(tmp_path):
    """The dangerous case: `_pool_complete` skips finished pools, so a requeue at a different
    `--pool-size` would keep the old ones and generate the rest under a different search."""
    from mlindex.scripts.run_fom_dump import refuse_a_changed_pool_size

    out = tmp_path / 'bundle'
    out.mkdir()
    (out / 'manifest.json').write_text(json.dumps({'pool_size': 4}), encoding='utf-8')
    refuse_a_changed_pool_size(str(out), 4)          # the same size is fine
    with pytest.raises(SystemExit, match='different searches'):
        refuse_a_changed_pool_size(str(out), 2)


def test_the_guard_is_silent_on_a_fresh_directory(tmp_path):
    from mlindex.scripts.run_fom_dump import refuse_a_changed_pool_size
    assert refuse_a_changed_pool_size(str(tmp_path / 'absent'), 4) is None


def test_optimizer_seed_moves_the_search_and_nothing_else(tmp_path):
    """`--optimizer-seed` is what S08 measures the reproducibility floor with.

    The floor is the spread of a reported number over runs differing ONLY in the search, so the
    flag has to do exactly one thing. Both halves are asserted, and the first is the half that is
    easy to omit: if the seed leaked into `prepare_peak_list` or `sample_entries` the runs would
    also differ in their patterns, and the spread would be generation noise wearing the floor's
    name (METRICS.md section 8).
    """
    ids = _entry_ids(2)
    base = _generate(tmp_path / 'base', ids, 1, 2, tmp_path)
    moved = _generate(tmp_path / 'moved', ids, 1, 2, tmp_path,
                      extra=['--optimizer-seed', '777'])

    # Same patterns. `q2_digest` is the peak list's own checksum, so this is the strong form.
    left, right = _entries(tmp_path / 'base'), _entries(tmp_path / 'moved')
    assert list(left['entry_id']) == list(right['entry_id'])
    assert list(left['q2_digest']) == list(right['q2_digest'])

    # A different search. Comparing `xnn` rather than the row count: the candidate count can
    # coincide, but the cells the search reaches cannot.
    assert base.shape[0] > 0 and moved.shape[0] > 0
    assert 'xnn' in _differing_columns(base, moved) or base.shape != moved.shape


def test_optimizer_seed_defaults_to_seed(tmp_path):
    """An invocation written before the flag existed must still produce the pool it produced."""
    ids = _entry_ids(2)
    implicit = _generate(tmp_path / 'implicit', ids, 1, 2, tmp_path)
    explicit = _generate(tmp_path / 'explicit', ids, 1, 2, tmp_path,
                         extra=['--optimizer-seed', '12345'])
    assert implicit.shape == explicit.shape
    assert not _differing_columns(implicit, explicit)
