"""Run once to populate tests/expected/ with baseline outputs.

Usage:
    python tests/generate_expected.py             # fast tests only
    python tests/generate_expected.py --models    # include model training tests
    python tests/generate_expected.py --all       # include model training + CLI tests
"""
import argparse
import ast
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
EXPECTED_DIR = Path(__file__).parent / 'expected'
TEST_DATA_DIR = REPO_ROOT / 'mlindex' / 'data' / 'test_data'

EXPECTED_DIR.mkdir(exist_ok=True)

BL_TO_LATTICE_SYSTEM = {
    'cF': 'cubic',  'cI': 'cubic',  'cP': 'cubic',
    'hP': 'hexagonal', 'hR': 'rhombohedral',
    'tI': 'tetragonal', 'tP': 'tetragonal',
    'oC': 'orthorhombic', 'oF': 'orthorhombic',
    'oI': 'orthorhombic', 'oP': 'orthorhombic',
    'mC': 'monoclinic', 'mP': 'monoclinic',
    'aP': 'triclinic',
}


def load_test_case(row):
    bl = str(row['bravais lattice'])
    wavelength = float(row['wavelength'])
    peak_file = str(row['peak list file'])
    a, b, c = float(row['a']), float(row['b']), float(row['c'])
    alpha = float(row['alpha']) * np.pi / 180
    beta  = float(row['beta'])  * np.pi / 180
    gamma = float(row['gamma']) * np.pi / 180
    unit_cell = np.array([a, b, c, alpha, beta, gamma])
    if peak_file.endswith('.csv'):
        pattern_name = peak_file.replace('.csv', '')
        path = TEST_DATA_DIR / pattern_name / peak_file
        df = pd.read_csv(path, index_col=0)
        peak_positions = np.array(ast.literal_eval(df.loc['peak_positions'].iloc[0]))
        q2_obs = np.sort(peak_positions ** 2)[:20]
    else:
        case_name = peak_file.replace('.pkslst', '').replace('.pklst', '')
        npy_path = TEST_DATA_DIR / case_name / f'{case_name}_peak_list.npy'
        q2_obs = np.sort(np.load(npy_path))[:20]
    return q2_obs, unit_cell, wavelength, bl, BL_TO_LATTICE_SYSTEM[bl]


def all_cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


# ---------------------------------------------------------------------------
# Utility expected files (fast, no models)
# ---------------------------------------------------------------------------

def generate_utility_expected(test_metadata):
    from mlindex.utilities.UnitCellTools import (
        get_xnn_from_unit_cell, get_hkl_matrix, get_partial_unit_cell,
    )

    def _xnn(unit_cell, lattice_system):
        uc_p = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
        return get_xnn_from_unit_cell(uc_p[np.newaxis], partial_unit_cell=True,
                                      lattice_system=lattice_system)
    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.FigureOfMerits import get_M20_from_xnn, get_M20_likelihood_from_xnn

    print('Generating utility expected files ...')
    for q2_obs, unit_cell, wavelength, bl, lattice_system in all_cases(test_metadata):
        hkl_ref = np.load(TEST_DATA_DIR.parent / f'hkl_ref_{bl}.npy')
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)

        # hkl_matrix
        np.save(EXPECTED_DIR / f'hkl_matrix_{bl}.npy', hkl2)

        q2_calc = np.sum(hkl2 * xnn, axis=1)

        # fast_assign
        q2_ref = q2_calc[np.newaxis]
        fa = fast_assign(q2_obs.astype(np.float64), q2_ref.astype(np.float64))
        np.save(EXPECTED_DIR / f'fast_assign_{bl}.npy', fa)

        hkl_assign = fa[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]

        # m20
        m20 = get_M20_from_xnn(q2_obs, xnn, hkl_assigned, hkl_ref, lattice_system)
        np.save(EXPECTED_DIR / f'm20_{bl}.npy', m20)

        # m20_likelihood
        log_lik, prob, M = get_M20_likelihood_from_xnn(
            q2_obs, xnn, hkl_assigned, lattice_system, bl
        )
        np.save(EXPECTED_DIR / f'm20_likelihood_{bl}.npy',
                np.stack([log_lik, M.flatten()]))

        print(f'  {bl}: M20={m20[0]:.3f}')

    print('Done.')


def generate_pkslst_expected(test_metadata_all):
    from mlindex.utilities.gsas import load_pkslst

    print('Generating load_pkslst expected files ...')
    for _, row in test_metadata_all.iterrows():
        peak_file = str(row['peak list file'])
        if not peak_file.endswith('.pkslst'):
            continue
        case_name = peak_file.replace('.pkslst', '')
        wavelength = float(row['wavelength'])
        pkslst_path = TEST_DATA_DIR / case_name / peak_file
        result = load_pkslst(str(pkslst_path), wavelength)
        np.save(EXPECTED_DIR / f'load_pkslst_{case_name}.npy', np.sort(result))
        print(f'  {case_name}')
    print('Done.')


# ---------------------------------------------------------------------------
# Reindexing expected files (fast, no models)
# ---------------------------------------------------------------------------

def generate_reindexing_expected(test_metadata):
    from mlindex.utilities.Reindexing import selling_reduction

    print('Generating reindexing expected files ...')
    for q2_obs, unit_cell, wavelength, bl, lattice_system in all_cases(test_metadata):
        result_uc, _, _ = selling_reduction(unit_cell[np.newaxis])
        np.save(EXPECTED_DIR / f'selling_reduction_{bl}.npy', result_uc[0])
        print(f'  {bl}')
    print('Done.')


# ---------------------------------------------------------------------------
# Model training expected files (slow, requires models)
# ---------------------------------------------------------------------------

def generate_model_training_expected(test_metadata):
    from mlindex.optimization.MPOptimizer import LocalComm
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    from mlindex.optimization.UtilitiesOptimizer import (
        _resolve_project_path,
        get_cubic_optimizer, get_hexagonal_optimizer, get_rhombohedral_optimizer,
        get_tetragonal_optimizer, get_orthorhombic_optimizer,
        get_monoclinic_optimizer, get_triclinic_optimizer,
    )

    print('Loading ML models ...')
    comm = LocalComm(1)
    project_path = _resolve_project_path()
    bl_to_factory = {
        'cF': get_cubic_optimizer, 'cI': get_cubic_optimizer, 'cP': get_cubic_optimizer,
        'hP': get_hexagonal_optimizer,
        'hR': get_rhombohedral_optimizer,
        'tI': get_tetragonal_optimizer, 'tP': get_tetragonal_optimizer,
        'oC': get_orthorhombic_optimizer, 'oF': get_orthorhombic_optimizer,
        'oI': get_orthorhombic_optimizer, 'oP': get_orthorhombic_optimizer,
        'mC': get_monoclinic_optimizer, 'mP': get_monoclinic_optimizer,
        'aP': get_triclinic_optimizer,
    }

    print('Generating model training expected files ...')
    for q2_obs, unit_cell, wavelength, bl, lattice_system in all_cases(test_metadata):
        factory = bl_to_factory[bl]
        opt = factory(bl, '1', 1, comm, project_path,
                      optimizer_class=OptimizerManager, seed=12345)
        opt.wrapper.setup_random()
        sg = opt.wrapper.data_params['split_groups'][0]

        # random generator
        rng = np.random.default_rng(12345)
        result = opt.wrapper.random_unit_cell_generator[bl].generate(
            10, rng, q2_obs, model='random',
        )
        np.save(EXPECTED_DIR / f'random_gen_{bl}.npy', result)

        # random forest
        rng = np.random.default_rng(12345)
        result = opt.wrapper.random_forest_generator[sg].generate(10, rng, q2_obs)
        np.save(EXPECTED_DIR / f'random_forest_{bl}.npy', result)

        # MI templates
        rng = np.random.default_rng(12345)
        result = opt.wrapper.miller_index_templator[bl].generate(10, rng, q2_obs)
        np.save(EXPECTED_DIR / f'mi_templates_{bl}.npy', result)

        # integral filter
        rng = np.random.default_rng(12345)
        result = opt.wrapper.integral_filter_generator[sg].generate(
            10, rng, q2_obs, batch_size=2,
        )
        np.save(EXPECTED_DIR / f'integral_filter_{bl}.npy', result)

        print(f'  {bl}')
    print('Done.')


# ---------------------------------------------------------------------------
# CLI expected files (very slow, requires models)
# ---------------------------------------------------------------------------

def generate_cli_expected(test_metadata):
    import tempfile
    import os

    row = test_metadata[test_metadata['bravais lattice'] == 'aP'].iloc[0]
    q2_obs, unit_cell, wavelength, bl, lattice_system = load_test_case(row)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        peak_file = tmp_path / 'aP_q2.npy'
        np.save(peak_file, q2_obs)

        print('Generating run_analytical expected output ...')
        analytic_out = tmp_path / 'analytic_results.json'
        cmd = [
            sys.executable, '-m', 'mlindex.command_line.run_analytical',
            '--peak-file', str(peak_file),
            '--peak-units', 'q2',
            '--bravais-lattices', 'aP',
            '--seed', '12345',
            '--output-file', str(analytic_out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'run_analytical failed:\n{r.stderr}')
        else:
            import shutil
            shutil.copy(analytic_out, EXPECTED_DIR / 'run_analytical_aP.json')
            print('  Done.')

        print('Generating run ML expected output (this may take several minutes) ...')
        cmd = [
            sys.executable, '-m', 'mlindex.command_line.run',
            '--peak-file', str(peak_file),
            '--peak-units', 'q2',
            '--bravais-lattices', 'aP',
            '--nproc', '1',
            '--seed', '12345',
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        if r.returncode != 0:
            print(f'run ML failed:\n{r.stderr}')
        else:
            import shutil
            shutil.copy(tmp_path / 'indexing_results.json', EXPECTED_DIR / 'run_ml_aP.json')
            print('  Done.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate expected test outputs')
    parser.add_argument('--models', action='store_true',
                        help='Also generate model training expected files')
    parser.add_argument('--all', action='store_true',
                        help='Also generate CLI expected files (implies --models)')
    args = parser.parse_args()

    test_metadata_all = pd.read_csv(TEST_DATA_DIR / 'gsasII_tutorials.csv')
    # Use one row per BL for expected files (avoid name collisions from duplicate BL)
    test_metadata = test_metadata_all.drop_duplicates(subset=['bravais lattice'], keep='last')

    generate_utility_expected(test_metadata)
    generate_pkslst_expected(test_metadata_all)
    generate_reindexing_expected(test_metadata)

    if args.models or args.all:
        generate_model_training_expected(test_metadata)

    if args.all:
        generate_cli_expected(test_metadata_all)
