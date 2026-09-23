"""The mix driver's command line, and the properties that make it runnable by one person.

Nothing here needs models or a dataset: the fit stage is fed a pool built by hand, which is also
the only way to check the arithmetic of a mix against an answer worked out on paper.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from mlindex.scripts import run_ensemble_refine as driver


def test_the_help_is_pure_ascii():
    """Redirecting --help on Windows encodes through the locale codepage and raises otherwise."""
    driver.build_parser().format_help().encode('ascii')


def test_importing_the_driver_starts_no_mpi_and_pins_no_threads():
    """It used to set KERAS_BACKEND and nine thread-count variables at module scope, before its
    own imports, so merely importing it pinned the importing process -- a pytest session included
    -- to one thread. Checked in a fresh interpreter, because this one has already imported it."""
    probe = (
        'import os, sys\n'
        'from mlindex.scripts import run_ensemble_refine\n'
        'print("mpi4py" in sys.modules, os.environ.get("OMP_NUM_THREADS"),'
        ' os.environ.get("KERAS_BACKEND"))\n'
        )
    # Started from an environment with these cleared, because another test module in the same
    # session imports something that sets them and a child would inherit them.
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('OMP_NUM_THREADS', 'KERAS_BACKEND')}
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                            check=True, env=environment)
    assert result.stdout.split() == ['False', 'None', 'None'], result.stdout


def test_counts_for_a_mix_always_sum_to_the_budget():
    """Rounding each share on its own lets the budget drift, and more candidates always score
    higher, so a mix that rounded up would win on that alone."""
    for budget in (100, 101, 999, 4000, 6000):
        for mix in ((1/3, 1/3, 1/3), (0.05, 0.7, 0.25), (1.0, 0.0, 0.0), (0.0, 0.5, 0.5)):
            counts = driver.counts_for_mix(mix, budget)
            assert counts.sum() == budget
            assert counts.min() >= 0


def test_the_grid_covers_the_simplex_and_every_mix_sums_to_one():
    grid = driver.mix_grid(0.02, 3)
    assert grid.shape == (1326, 3)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert np.any(np.all(np.isclose(grid, [1.0, 0.0, 0.0]), axis=1))


def test_stacking_takes_the_first_candidates_of_each_generator():
    distance = np.arange(2*10*3, dtype=float).reshape(2, 10, 3)
    pool = driver.stack_pools(distance, np.array([2, 3, 0]))
    assert pool.shape == (2, 5)
    np.testing.assert_array_equal(pool[0], [0.0, 3.0, 1.0, 4.0, 7.0])


def _write_pools(directory, n_crystals=12, depth=8):
    """A pool where one generator is plainly better, so the fit has a knowable right answer."""
    rng = np.random.default_rng(0)
    distance = np.empty((n_crystals, depth, 3))
    distance[:, :, 0] = rng.uniform(2e-4, 4e-4, (n_crystals, depth))    # trees: middling
    distance[:, :, 1] = rng.uniform(1e-4, 1.2e-4, (n_crystals, depth))  # abnn: close
    distance[:, :, 2] = rng.uniform(5e-3, 9e-3, (n_crystals, depth))    # templates: hopeless
    # positions spread far enough apart that nothing clumps, so the fit's answer is decided by
    # the distances alone and this test is about the selection, not the crowding
    rng_pos = np.random.default_rng(1)
    np.savez_compressed(
        directory/'cP_pools.npz',
        xnn=rng_pos.uniform(0, 1, (n_crystals, depth, 3, 1)).astype(np.float32),
        xnn_true=np.zeros((n_crystals, 1)),
        distances=distance,
        identifiers=np.array([f'X{index:03d}' for index in range(n_crystals)], dtype=object),
        generator_names=np.array(['trees', 'abnn', 'templates'], dtype=object),
        )
    (directory/'pools_manifest.json').write_text(json.dumps({
        'commit': 'test', 'population': 'synthetic',
        'lattices': {'cP': {
            'n_crystals': n_crystals, 'per_generator': depth, 'shipped_budget': 4,
            'shipped_mix': {'trees': 0.45, 'abnn': 0.45, 'templates': 0.10},
            'generator_names': ['trees', 'abnn', 'templates'], 'n_peaks': 10}},
        }), encoding='utf-8')


def _write_discount(directory):
    np.savez(directory/'cP_clump_discount.npz', delta=1e-6,
             k=np.array([1.0, 2.0, 8.0]), alpha=np.array([1.0, 0.6, 0.2]))


def _write_curve(directory):
    radii = np.logspace(-4, -2, 40)
    success = np.clip(np.linspace(0.9, 0.0, 40), 0.0, 1.0)
    np.save(directory/'cP_roc_peaks10_drop8_iter100_sampQ2.npy', np.vstack((radii, success)))


def test_the_fit_stage_runs_without_mpi_and_picks_the_generator_that_is_closer(tmp_path):
    pools = tmp_path/'pools'
    pools.mkdir()
    roc = tmp_path/'roc'
    roc.mkdir()
    out = tmp_path/'out'
    _write_pools(pools)
    _write_curve(roc)
    _write_discount(roc)

    assert driver.main([
        '--stage', 'fit', '--bravais-lattices', 'cP', '--pools', str(pools),
        '--roc-dir', str(roc), '--clump-discount', str(roc),
        '--out-dir', str(out), '--step', '0.1',
        ]) == 0

    frame = pd.read_csv(out/'ensemble_mix.csv')
    whole = frame.loc[frame['split'] == 'all']
    # abnn's candidates are ten times closer than the others', so it must take the largest share.
    for _, row in whole.iterrows():
        assert row['best_abnn'] > row['best_trees'], row.to_dict()
        assert row['best_abnn'] > row['best_templates'], row.to_dict()


def test_the_fit_refuses_pools_it_was_never_given(tmp_path):
    out = tmp_path/'out'
    with pytest.raises(SystemExit, match='no pools manifest'):
        driver.main(['--stage', 'fit', '--pools', str(tmp_path/'nothing'),
                     '--roc-dir', str(tmp_path), '--clump-discount', str(tmp_path),
                     '--out-dir', str(out)])


def test_a_lattice_the_package_does_not_know_is_refused(tmp_path):
    with pytest.raises(SystemExit, match='not Bravais lattices'):
        driver.main(['--stage', 'fit', '--bravais-lattices', 'xQ', '--pools', str(tmp_path),
                     '--roc-dir', str(tmp_path), '--clump-discount', str(tmp_path),
                     '--out-dir', str(tmp_path)])


def test_the_fit_refuses_to_run_without_a_clump_discount(tmp_path):
    """A score that assumes candidates are independent always names a corner, so this is not a
    convenience default -- there is no sensible value to fall back to."""
    with pytest.raises(SystemExit, match='not independent'):
        driver.main(['--stage', 'fit', '--pools', str(tmp_path), '--roc-dir', str(tmp_path),
                     '--out-dir', str(tmp_path)])


def test_the_generate_stage_says_what_it_needs(tmp_path):
    with pytest.raises(SystemExit, match='needs --dataset-directory'):
        driver.main(['--stage', 'generate', '--pools', str(tmp_path)])


def test_spreading_the_score_over_processes_does_not_change_it():
    """The parallel path is a partition of crystals, so it must agree to the last bit.

    It is the only thing standing between a 67-hour fit and a half-hour one, and a subtle
    disagreement would not look like a failure -- it would look like a different answer.
    """
    rng = np.random.default_rng(7)
    n_crystals, depth = 9, 12
    distance = rng.uniform(1e-4, 5e-3, (n_crystals, depth, 3))
    xnn = rng.uniform(0, 1e-3, (n_crystals, depth, 3, 4)).astype(np.float32)
    grid = driver.mix_grid(0.25, 3)
    curve = np.vstack((np.logspace(-4, -2, 20), np.linspace(0.9, 0.0, 20)))
    discount = (2e-4, np.array([1.0, 2.0, 8.0]), np.array([1.0, 0.6, 0.2]))

    serial = driver.score_every_mix(distance, xnn, grid, 8, curve, discount, 1)
    parallel = driver.score_every_mix(distance, xnn, grid, 8, curve, discount, 4)
    assert parallel.shape == (grid.shape[0], n_crystals)
    np.testing.assert_array_equal(serial, parallel)


def test_a_lattice_short_of_crystals_says_so(tmp_path, capsys):
    """Ten of the fourteen lattices cannot supply 10 000, and a quiet shortfall is a wrong run."""
    rng = np.random.default_rng(3)
    n_rows = 5
    pd.DataFrame({
        'identifier': [f'X{i}' for i in range(n_rows)],
        'train': [True]*n_rows,
        f'q2_{driver.BROADENING_TAG}': [rng.uniform(0.01, 1.0, 30) for _ in range(n_rows)],
        'reindexed_xnn': [rng.uniform(0, 1, 6) for _ in range(n_rows)],
        'reindexed_unit_cell': [np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])]*n_rows,
        }).to_parquet(tmp_path/'dataset_cP.parquet')

    entries, _ = driver.load_entries('cP', 1000, str(tmp_path), 1, 'nominal')
    assert len(entries) == n_rows
    assert 'WARNING' in capsys.readouterr().out


def _write_dataset(path, n_rows, seed):
    rng = np.random.default_rng(seed)
    pd.DataFrame({
        'identifier': [f'{path.stem}_{i}' for i in range(n_rows)],
        'train': [True]*n_rows,
        f'q2_{driver.BROADENING_TAG}': [np.sort(rng.uniform(0.01, 1.0, 30))
                                        for _ in range(n_rows)],
        'reindexed_xnn': [rng.uniform(0.01, 1, 6) for _ in range(n_rows)],
        'reindexed_unit_cell': [np.array([5.0, 6.0, 7.0, 90.0, 90.0, 90.0])]*n_rows,
        }).to_parquet(path)


def test_the_second_phase_bundle_gets_the_partner_pool_it_needs(tmp_path):
    """`--bundle second_phase` was offered and could not run.

    The bundle adds lines from a real partner cell, so prepare_peak_list needs a pool of them
    and raises without one. load_entries never passed it, so the bundle failed on the first
    crystal every time -- and it is the hardest contaminant, which is the case the mix most
    needs to be tested against.
    """
    for bravais_lattice in ('cP', 'oP', 'tP'):
        _write_dataset(tmp_path/f'dataset_{bravais_lattice}.parquet', 6, seed=hash(bravais_lattice) % 1000)

    pool = driver.load_second_phase_pool(str(tmp_path), 1)
    identifiers, lines = pool
    assert len(identifiers) == len(lines) == 18, 'every lattice should contribute its crystals'
    # the partner comes from the whole set, not from the lattice being fitted
    assert any(name.startswith('dataset_oP') for name in identifiers)
    assert any(name.startswith('dataset_cP') for name in identifiers)

    with pytest.raises(ValueError, match='partner pool'):
        driver.load_entries('cP', 2, str(tmp_path), 1, 'second_phase')

    entries, _ = driver.load_entries('cP', 2, str(tmp_path), 1, 'second_phase',
                                     second_phase_pool=pool)
    assert 'q2' in entries.columns


def test_the_difficulty_grid_puts_the_second_phase_at_three_contaminants():
    """DWMM's convention: three contaminants IS the second phase.

    Three independently placed lines and three from one real partner cell are not the same
    object -- the partner's lines are mutually consistent with some other lattice, which is what
    makes them hard to reject -- so the top of the axis uses the correlated mechanism.
    """
    for level in (0, 1, 2):
        condition = driver.grid_condition(level, 4)
        assert condition.n_contaminants == level
        assert condition.second_phase_lines == 0
    top = driver.grid_condition(3, 4)
    assert top.n_contaminants == 0, 'three contaminants is not three independent lines'
    assert top.second_phase_lines == 3
    # every cell keeps the nominal peak error and a distinct tag
    tags = {driver.grid_condition(c, d).tag
            for c in driver.CONTAMINANT_LEVELS for d in driver.DROPOUT_LEVELS}
    assert len(tags) == len(driver.CONTAMINANT_LEVELS)*len(driver.DROPOUT_LEVELS)
    assert all(driver.grid_condition(c, d).error_multiplier == 1.0
               for c in driver.CONTAMINANT_LEVELS for d in driver.DROPOUT_LEVELS)
    with pytest.raises(ValueError):
        driver.grid_condition(5, 4)
    with pytest.raises(ValueError):
        driver.grid_condition(1, 3)


def test_the_grid_draws_a_difficulty_per_crystal_and_keeps_every_one(tmp_path):
    """--n-entries X must still deliver X crystals, and the draw must not depend on X.

    A crystal whose observed range is narrow cannot take every contaminant asked for. Dropping
    it would bias the sample toward crystals that can, so the draw steps down instead and the
    delivered difficulty is recorded on the row.
    """
    for bravais_lattice in ('cP', 'oP', 'tP'):
        _write_dataset(tmp_path/f'dataset_{bravais_lattice}.parquet', 40,
                       seed=abs(hash(bravais_lattice)) % 1000)
    pool = driver.load_second_phase_pool(str(tmp_path), 1)
    assert driver.needs_second_phase_pool('grid'), 'the grid reaches the second phase'
    assert not driver.needs_second_phase_pool('nominal')

    entries, _ = driver.load_entries('cP', 24, str(tmp_path), 1, 'grid',
                                     second_phase_pool=pool)
    assert len(entries) == 24, 'every crystal asked for must come back'
    # what is recorded is what was DELIVERED, which can fall short of the level drawn: a crystal
    # with few spare interior peaks yields 3 where 4 was asked. So the bound is the range, not
    # the set -- on real data the delivered dropout does take values outside DROPOUT_LEVELS.
    assert entries['n_contaminants'].between(0, max(driver.CONTAMINANT_LEVELS)).all()
    assert entries['n_dropout'].between(0, max(driver.DROPOUT_LEVELS)).all()
    assert entries['n_dropout'].nunique() > 1, 'the difficulty must actually vary'
    assert entries['n_contaminants'].nunique() > 1

    # the draw is keyed on the crystal, so asking for fewer gives the same crystals the same cell
    fewer, _ = driver.load_entries('cP', 12, str(tmp_path), 1, 'grid', second_phase_pool=pool)
    merged = fewer.merge(entries, on='identifier', suffixes=('_few', '_many'))
    assert len(merged) == 12
    assert (merged['n_dropout_few'] == merged['n_dropout_many']).all()
    assert (merged['n_contaminants_few'] == merged['n_contaminants_many']).all()
