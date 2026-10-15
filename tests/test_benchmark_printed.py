"""The benchmark's replay of the printed list (`BenchmarkPrinted`), and the outcomes and contrasts
read from it."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import models_tree_without_the_ranker
from mlindex.model_training import Benchmark
from mlindex.model_training import BenchmarkMetrics as metrics

FROZEN_SPLIT = Path('docs/fom_campaign2/artifacts/S06_split_manifest.parquet')
DATASETS = Path('mlindex/data/generated_datasets')


def _entries(n, lattice='cP', degenerate=False):
    return pd.DataFrame({'entry_id': [f'E{i}' for i in range(n)],
                         'condition_bundle': 'b1_error1_cont0',
                         'bravais_lattice_true': lattice, 'is_degenerate': degenerate})


def _printed(rows):
    return pd.DataFrame(rows, columns=['entry_id', 'condition_bundle', 'position', 'is_correct'])


def test_a_pattern_is_read_at_its_first_correct_printed_row():
    entries = _entries(4)
    printed = _printed([
        ('E0', 'b1_error1_cont0', 0, True), ('E0', 'b1_error1_cont0', 1, True),
        ('E1', 'b1_error1_cont0', 0, False), ('E1', 'b1_error1_cont0', 6, True),
        ('E2', 'b1_error1_cont0', 0, False)])
    flags = metrics.printed_flags(printed, entries).set_index('entry_id')

    assert flags['top1'].tolist() == [True, False, False, False]
    assert flags['top5'].tolist() == [True, False, False, False]
    assert flags['top10'].tolist() == [True, True, False, False]
    assert flags['listed'].tolist() == [True, True, False, False]
    np.testing.assert_allclose(flags['reciprocal_rank'], [1.0, 1/7, 0.0, 0.0])


def test_a_degenerate_pattern_counts_as_a_miss():
    flags = metrics.printed_flags(_printed([('E0', 'b1_error1_cont0', 0, True)]),
                                  _entries(1, degenerate=True))

    assert not flags['top1'].any() and flags['reciprocal_rank'].tolist() == [0.0]


def test_the_balanced_contrast_weights_lattices_equally_and_has_no_spread_without_a_difference():
    a = pd.concat([_entries(2, 'cP'), _entries(8, 'aP').assign(
        entry_id=lambda f: 'A' + f['entry_id'])], ignore_index=True)
    a['top1'] = False
    b = a.copy()
    b.loc[b['bravais_lattice_true'] == 'cP', 'top1'] = True

    delta, se = metrics.balanced_delta_se(a, b, 'top1', np.random.default_rng(1))
    assert delta == pytest.approx(0.5)
    assert se == pytest.approx(0.0)
    assert metrics.balanced_mean(b, 'top1') == pytest.approx(0.5)
    with pytest.raises(ValueError, match='different patterns'):
        metrics.balanced_delta_se(a, b.iloc[1:], 'top1', np.random.default_rng(1))


def test_a_lattice_pool_is_in_final_rank_order_with_its_ids():
    from mlindex.model_training.BenchmarkPrinted import lattice_pool

    block = pd.DataFrame({
        'candidate_id': [7, 3, 5], 'final_rank': [2, 0, 1], 'lattice_system': 'cubic',
        'M20': [5.0, 9.0, 7.0], 'n_indexed': [10, 12, 11], 'spacegroup': 'P m -3 m',
        'unit_cell': [np.array([4.0]), np.array([5.0]), np.array([6.0])],
        'is_correct': [False, True, False],
        'M_rev': 1.0, 'M_sym': [1.0, 3.0, 2.0], 'X_N': 1.0, 'n_over': 1.0, 'M_wu': 1.0,
        'F_N_q': 1.0, 'zone_dominance': 1.0, 'V_over_Vcrit': 1.0, 'n_dewolff61': 1.0,
        'n_absent_extra_in_range': [0, 1, 2], 'n_ref_in_range': [4, 4, 0]})
    pool, ids, labels = lattice_pool(block)

    assert ids.tolist() == [3, 5, 7]
    assert labels.tolist() == [True, False, False]
    assert pool['M20'].tolist() == [9.0, 7.0, 5.0]
    np.testing.assert_allclose(pool['unit_cell'][:, 0], [5.0, 6.0, 4.0])
    np.testing.assert_allclose(pool['unit_cell'][:, 3:], np.pi/2)
    np.testing.assert_allclose(pool['f_absent_extra'], [0.25, np.nan, 0.0])


def _run_printed(peak_file, directory, models):
    """`mlindex.run`'s printed list for `peak_file`, with the model tree `models`."""
    directory.mkdir()
    cmd = [sys.executable, '-m', 'mlindex.command_line.run', '--peak-file', str(peak_file),
           '--peak-units', 'q2', '--nproc', '1', '--seed', '12345']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(directory),
                            env={**os.environ, 'MLINDEX_MODELS_DIR': str(models)})
    assert result.returncode == 0, result.stderr
    return pd.read_json(directory/'indexing_results.json')


@pytest.mark.slow
@pytest.mark.skipif(not (FROZEN_SPLIT.is_file() and DATASETS.is_dir()),
                    reason='frozen split or source datasets not on this machine')
@pytest.mark.parametrize('lattice', ['aP', 'mC'])
def test_the_replayed_list_is_the_list_mlindex_run_prints(tmp_path, models_dir, monkeypatch,
                                                         lattice):
    """A fom-dev pattern searched by the benchmark and replayed gives `mlindex.run`'s printed
    list on the same peaks, row for row: under the learned ranker, and under M_sym, which
    `mlindex.run` prints without one. It fails if either the replay or the indexer's ranking,
    shortlist or duplicate removal changes."""
    from mlindex.model_training import BenchmarkPrinted, BenchmarkRuns
    from mlindex.optimization.Candidates import PRUNE_M20_THRESHOLD
    from mlindex.utilities.Digests import file_digest

    monkeypatch.setenv('MLINDEX_MODELS_DIR', str(models_dir))
    monkeypatch.setattr(BenchmarkRuns, '_provenance', lambda: {
        'commit': 'test', 'models_dir': 'test', 'models_digest': 'test', 'n_model_files': 0,
        'model_revision': 'test'})
    bundle = 'b1_error1_cont0'
    BenchmarkRuns.run_arm(tmp_path/'pool', FROZEN_SPLIT, file_digest(FROZEN_SPLIT),
                          population='general', true_lattices=[lattice], per_lattice=1,
                          bundles=[bundle], cut=PRUNE_M20_THRESHOLD, seed=12345,
                          search_seed=12345)
    replayed = BenchmarkPrinted.replay_bundle(tmp_path/'pool', bundle, BenchmarkPrinted.rankings())
    entry = Benchmark.load_entries(tmp_path/'pool').iloc[0]
    np.save(tmp_path/'peaks.npy', np.asarray(entry['q2_obs'], dtype=np.float64))
    trees = {'learned': models_dir,
             'M_sym': models_tree_without_the_ranker(models_dir, tmp_path/'no_ranker')}

    for ranking, models in trees.items():
        printed = _run_printed(tmp_path/'peaks.npy', tmp_path/ranking, models)
        ours = replayed.loc[replayed['ranking'] == ranking].reset_index(drop=True)
        assert ours.shape[0] == printed.shape[0] > 0, ranking
        assert ours['bravais_lattice'].tolist() == printed['bravais_lattice'].tolist(), ranking
        for column in ('score', 'M20', 'M_sym', 'volume'):
            np.testing.assert_allclose(ours[column], printed[column], rtol=1e-8, atol=1e-9,
                                       err_msg=f'{ranking} {column}')
