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


def _generate(out_dir, entry_ids, n_pools, pool_size, tmp_path):
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
          '--out-dir', str(out_dir)])
    import glob
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(f'{out_dir}/candidates_*.parquet'))]
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(['entry_id', 'bravais_lattice', 'candidate_id'],
                             kind='stable', ignore_index=True)


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


def test_the_pool_topology_does_not_change_the_pool(tmp_path):
    """R17 says it does. It no longer does, and this is what stops that regressing."""
    ids = _entry_ids(2)
    one_pool = _generate(tmp_path / 'one', ids, 1, 2, tmp_path)
    two_pools = _generate(tmp_path / 'two', ids, 2, 2, tmp_path)
    assert one_pool.shape == two_pools.shape
    assert not _differing_columns(one_pool, two_pools)
