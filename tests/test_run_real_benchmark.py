"""`run_real_benchmark` prints what the programs print."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import models_tree_without_the_ranker
from mlindex.scripts import run_real_benchmark as real

FROZEN_SPLIT = Path('docs/fom_campaign2/artifacts/S06_split_manifest.parquet')
DATASETS = Path('mlindex/data/generated_datasets')
WAVELENGTH = 1.5406


def _pattern(lattice):
    """One fom-dev crystal of `lattice` in the real file's columns: its first twenty lines with a
    zero-point offset of 0.05 degrees of 2-theta at `WAVELENGTH`, as q."""
    from mlindex.model_training import BenchmarkRuns

    entry = BenchmarkRuns.load_source_rows(
        BenchmarkRuns.draw_entries(FROZEN_SPLIT, 1, 12345, bravais_lattices=[lattice])).iloc[0]
    q2 = np.asarray(entry['q2_1'], dtype=np.float64)
    theta2 = 2*np.arcsin(WAVELENGTH*np.sqrt(np.sort(q2[q2 > 0])[:20])/2) + np.radians(0.05)
    return pd.DataFrame([{
        'file_name': f'/x/{entry["identifier"]}.json',
        'peak_positions': 2*np.sin(theta2/2)/WAVELENGTH, 'primary_wavelength': WAVELENGTH,
        'unit_cell': entry['reindexed_unit_cell'], 'bravais_lattice': lattice,
        'spacegroup': entry['reindexed_spacegroup_symbol_hm']}])


def test_a_repeated_peak_is_dropped_and_the_next_one_taken():
    """Twenty-one peaks, one stored twice: the twenty distinct ones are searched, as q2."""
    q = np.linspace(0.1, 0.9, 21)
    np.testing.assert_allclose(real.peak_list(np.insert(q, 5, q[5])), q[:20]**2)
    np.testing.assert_allclose(real.peak_list(q), q[:20]**2)


@pytest.mark.slow
@pytest.mark.skipif(not (FROZEN_SPLIT.is_file() and DATASETS.is_dir()),
                    reason='frozen split or source datasets not on this machine')
@pytest.mark.parametrize('indexer', ['ml', 'analytical'])
def test_the_wrapper_prints_what_the_program_prints(tmp_path, models_dir, monkeypatch, indexer):
    """Row for row, the list `mlindex.run --zero-error` prints with and without the ranker, or
    `run_analytical --zero-error` prints, on the same peaks."""
    monkeypatch.setenv('MLINDEX_MODELS_DIR', str(models_dir))
    patterns = _pattern('oP')
    printed = real.search(patterns, indexer, 0, 1)
    np.save(tmp_path/'peaks.npy', patterns['peak_positions'].iloc[0])

    module = f'mlindex.command_line.{"run" if indexer == "ml" else "run_analytical"}'
    output = 'indexing_results.json' if indexer == 'ml' else 'analytic_results.json'
    trees = ({'learned': models_dir,
              'M_sym': models_tree_without_the_ranker(models_dir, tmp_path/'no_ranker')}
             if indexer == 'ml' else {'M_sym': models_dir})
    for ranking, models in trees.items():
        (tmp_path/ranking).mkdir()
        result = subprocess.run(
            [sys.executable, '-m', module, '--peak-file', str(tmp_path/'peaks.npy'),
             '--peak-units', 'q', '--nproc', '1', '--seed', '12345', '--zero-error',
             '--wavelength', str(WAVELENGTH)],
            capture_output=True, text=True, cwd=str(tmp_path/ranking),
            env={**os.environ, 'MLINDEX_MODELS_DIR': str(models)})
        assert result.returncode == 0, result.stderr
        expected = pd.read_json(tmp_path/ranking/output)
        ours = printed.loc[printed['ranking'] == ranking].reset_index(drop=True)
        assert ours.shape[0] == expected.shape[0] > 0, ranking
        assert ours['bravais_lattice'].tolist() == expected['bravais_lattice'].tolist(), ranking
        for column in ('score', 'M20', 'M_sym'):
            np.testing.assert_allclose(ours[column], expected[column], rtol=1e-8, atol=1e-9,
                                       err_msg=f'{indexer} {ranking} {column}')
