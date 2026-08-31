"""S08 -- the Benchmark B slicer, and the three ways a slice could be quietly useless.

The slice exists so the metrics module can be developed and gated off the cluster. That makes it a
piece of experimental apparatus rather than a convenience: a slice that silently misses the hard
stratum, or that is drawn proportionally so the rare lattices carry one entry each, passes a gate
that means nothing. And a slice that carries `fom-test` breaks the seal PROTOCOL section 3 rule 2
holds until S15 -- irreversibly, because the entries would then be on a laptop.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'mlindex', 'scripts'))

from mlindex.model_training import FomMetrics
import run_fom_pool_subset as subset


BUNDLES = ('c2_error1_cont0', 'c2_error2_cont0')
LATTICES = ('cF', 'cI', 'aP', 'mP', 'mC', 'oP')


def _entries(n_per_lattice=30, seed=1):
    """One row per (entry, bundle), as the pool's entry table is."""
    rng = np.random.default_rng(seed)
    rows = []
    for lattice in LATTICES:
        # cF is deliberately short, the way the real source population is (C2-R-010).
        count = 4 if lattice == 'cF' else n_per_lattice
        for index in range(count):
            entry_id = f'{lattice}{index:03d}'
            split = ('fom-test' if index % 5 == 0 else
                     'fom-dev' if index % 5 == 1 else 'fom-train')
            # Drawn once per source entry, not per row: the decile is frozen on the split
            # manifest and is a property of the crystal, so it is identical across that entry's
            # condition bundles. Drawing it per row would make the fixture disagree with the
            # thing being tested -- which is R14 in miniature.
            decile = int(rng.integers(0, 10))
            volume = float(rng.uniform(100, 900))
            for bundle in BUNDLES:
                rows.append(dict(
                    entry_id=entry_id, condition_bundle=bundle, split=split,
                    bravais_lattice_true=lattice, lattice_system_true='triclinic',
                    volume_decile=decile, volume_true=volume,
                    ))
    return pd.DataFrame(rows)


def _candidates(entries, per_cell=12, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for row in entries.itertuples():
        for candidate in range(per_cell):
            rows.append(dict(
                entry_id=row.entry_id, condition_bundle=row.condition_bundle,
                bravais_lattice=row.bravais_lattice_true, candidate_id=candidate,
                M20=float(rng.gamma(2.0, 2.0)), is_correct=bool(candidate == 0),
                ))
    return pd.DataFrame(rows)


def _write_pool(root, entries, candidates):
    root.mkdir(parents=True, exist_ok=True)
    entries.to_parquet(root / 'entries.parquet', index=False)
    for bundle in candidates['condition_bundle'].unique():
        for lattice in candidates['bravais_lattice'].unique():
            block = candidates.loc[(candidates['condition_bundle'] == bundle)
                                   & (candidates['bravais_lattice'] == lattice)]
            if block.empty:
                continue
            block.to_parquet(root / f'candidates_{bundle}_{lattice}.parquet', index=False)
    with open(root / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(dict(schema_version='3', top_k=200, subsampled=True,
                       n_candidates=int(candidates.shape[0])), handle)
    return root


@pytest.fixture
def pool(tmp_path):
    entries = _entries()
    return _write_pool(tmp_path / 'pool', entries, _candidates(entries)), entries


def test_the_sealed_split_is_refused_rather_than_filtered(pool):
    """Naming fom-test raises. It is refused at the argument, not dropped downstream, because a
    filter that silently drops it leaves no evidence that it was ever asked for."""
    _, entries = pool
    with pytest.raises(SystemExit, match='sealed'):
        subset.select_entries(entries, 5, 0.5, ('fom-train', 'fom-test'), seed=0)


def test_no_sealed_entry_reaches_the_slice(pool):
    root, entries = pool
    chosen = subset.select_entries(entries, 8, 0.5, subset.ALLOWED_SPLITS, seed=0)
    assert not chosen['entry_id'].isin(
        entries.loc[entries['split'] == 'fom-test', 'entry_id']).any()


def test_the_draw_is_balanced_across_lattices_not_proportional(pool):
    """Every lattice gets its quota, capped only by what exists.

    This is the property the slice is for. The real split runs 600 entries for each of aP, mC and
    mP against 20 for cF, so a proportional draw gives cF under two entries and no per-lattice
    number can be exercised on it at all.
    """
    _, entries = pool
    chosen = subset.select_entries(entries, 8, 0.5, subset.ALLOWED_SPLITS, seed=0)
    counts = chosen.groupby('bravais_lattice_true').size()
    plentiful = [lattice for lattice in LATTICES if lattice != 'cF']
    assert set(counts.loc[plentiful]) == {8}
    # cF has four source entries in total and three of them are outside the sealed split, so its
    # quota is capped by availability rather than unmet.
    assert 0 < counts['cF'] < 8


def test_the_hard_stratum_is_selected_for(pool):
    """A hard lattice's quota reserves half its places for the high-volume deciles.

    Left to a uniform draw the hard stratum arrives at whatever rate the deciles happen to give,
    which is how a slice ends up unable to exercise the numbers every headline claim rests on.
    """
    _, entries = pool
    chosen = subset.select_entries(entries, 10, 0.5, subset.ALLOWED_SPLITS, seed=0)
    for lattice in subset.HARD_LATTICES:
        rows = chosen.loc[chosen['bravais_lattice_true'] == lattice]
        n_hard = int((rows['volume_decile'] >= subset.HARD_MIN_DECILE).sum())
        available = entries.loc[(entries['bravais_lattice_true'] == lattice)
                                & (entries['split'].isin(subset.ALLOWED_SPLITS))
                                & (entries['volume_decile'] >= subset.HARD_MIN_DECILE)]
        assert n_hard == min(5, available['entry_id'].nunique())


def test_the_draw_is_deterministic_given_the_seed(pool):
    _, entries = pool
    first = subset.select_entries(entries, 6, 0.5, subset.ALLOWED_SPLITS, seed=7)
    second = subset.select_entries(entries, 6, 0.5, subset.ALLOWED_SPLITS, seed=7)
    pd.testing.assert_frame_equal(first, second)
    other = subset.select_entries(entries, 6, 0.5, subset.ALLOWED_SPLITS, seed=8)
    assert not first['entry_id'].equals(other['entry_id'])


def test_end_to_end_slice_is_a_valid_pool(pool, tmp_path):
    """The slice reads back through the ordinary loader and carries only the chosen entries."""
    root, _ = pool
    out_dir = tmp_path / 'slice'
    subset.main(['--pool', str(root), '--out-dir', str(out_dir),
                 '--entries-per-lattice', '6', '--seed', '3'])

    from mlindex.model_training import FomBenchmark
    entries = FomBenchmark.load_entries(out_dir)
    candidates = FomBenchmark.load_candidates(out_dir)
    assert set(entries['split']) <= set(subset.ALLOWED_SPLITS)
    assert set(candidates['entry_id']) == set(entries['entry_id'])
    # Every (entry, bundle) cell keeps all of its candidates: the slice selects entries, never
    # candidates, so nothing inside a retained cell may be thinned.
    per_cell = candidates.groupby(['entry_id', 'condition_bundle']).size()
    assert set(per_cell) == {12}

    manifest = FomBenchmark.load_manifest(out_dir)
    assert manifest['subset_of'] == str(root)
    assert manifest['n_candidates'] == candidates.shape[0]
    # The retention depth survives, because the metrics module refuses a rank metric without it.
    assert manifest['top_k'] == 200
    depth, subsampled = FomBenchmark.subsample_depth(out_dir)
    assert (depth, subsampled) == (200, True)


def test_hard_stratum_constants_match_the_metrics_module():
    """The slicer restates the hard stratum rather than importing it, to stay light on a login
    node. A drifted copy would select for a stratum the metrics module does not report."""
    assert subset.HARD_LATTICES == FomMetrics.HARD_LATTICES
    assert subset.HARD_MIN_DECILE == FomMetrics.HARD_MIN_DECILE


def test_parallel_filtering_gives_the_same_slice(pool, tmp_path):
    """The files are independent, so the process count is a throughput knob and nothing else.

    Worth pinning rather than assuming: the whole pool is scanned either way -- about 1 % of
    entries are wanted and nothing is sorted on `entry_id`, so there is no index to skip on -- and
    the only reason to reach for processes is the 122 GB of I/O. A knob that changed the output
    would be worse than no knob.
    """
    root, _ = pool
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    for out_dir, processes in ((serial, '1'), (parallel, '4')):
        subset.main(['--pool', str(root), '--out-dir', str(out_dir),
                     '--entries-per-lattice', '6', '--seed', '3', '--processes', processes])

    from mlindex.model_training import FomBenchmark
    left = FomBenchmark.load_candidates(serial).sort_values(
        ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']).reset_index(drop=True)
    right = FomBenchmark.load_candidates(parallel).sort_values(
        ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert FomBenchmark.load_manifest(serial)['n_candidates'] == \
        FomBenchmark.load_manifest(parallel)['n_candidates']
