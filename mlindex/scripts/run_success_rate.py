import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['ATLAS_NUM_THREADS'] = '1'
os.environ['SKLEARN_N_JOBS'] = '1'
import numpy as np
import pandas as pd
import time
from mpi4py import MPI
import json
import sys

from mlindex.model_training.BenchmarkConditions import Condition
from mlindex.model_training.BenchmarkPatterns import prepare_peak_list
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import get_partial_unit_cell
from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
from mlindex.optimization.UtilitiesOptimizer import get_cubic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_hexagonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_monoclinic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_orthorhombic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_rhombohedral_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_tetragonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_triclinic_optimizer
from mlindex.optimization.UtilitiesOptimizer import lattice_budget
from mlindex.optimization.CandidateValidation import label_known_bl_batch


def parse_bool(argument, name):
    # bool() of any non-empty string is True, so 'False' used to select the training set.
    if argument in ['True', '1']:
        return True
    elif argument in ['False', '0']:
        return False
    raise ValueError(f'{name} must be True, False, 1 or 0, not {argument!r}')


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    split_comm = comm.Split(color=rank, key=rank)
    models_directory = _resolve_models_dir()

    broadening_tag = '1'
    n_entries = 10000

    bravais_lattice = sys.argv[1]
    q2_error_multiplier = int(sys.argv[2])
    n_contaminants = int(sys.argv[3])
    n_candidates_scale = float(sys.argv[4])
    use_train = parse_bool(sys.argv[5], 'use_train')
    n_iterations = int(sys.argv[6])
    # Optional. True replaces the ML generators with the random-cell generator alone, at the same
    # candidate budget: the no-ML baseline.
    if len(sys.argv) > 7:
        random = parse_bool(sys.argv[7], 'random')
    else:
        random = False
    if random:
        options = {'generator_info': [{
            'generator': 'random',
            'n_unit_cells': lattice_budget(bravais_lattice, n_candidates_scale),
            }]}
    else:
        options = None

    n_top_candidates = 20
    # Each entry's noise is seeded from its identifier and this seed, so it does not depend on
    # which rank the entry lands on.
    base_seed = 0
    condition = Condition(
        key='success_rate',
        label='success_rate',
        description='',
        axis=None,
        error_multiplier=q2_error_multiplier,
        n_contaminants=n_contaminants,
        )
    if bravais_lattice in ['cF', 'cI', 'cP']:
        optimizer = get_cubic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm, models_directory, options=options)
    elif bravais_lattice in ['hP']:
        optimizer = get_hexagonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm, models_directory, options=options)
    elif bravais_lattice in ['hR']:
        optimizer = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm, models_directory, options=options)
    elif bravais_lattice in ['tI', 'tP']:
        optimizer = get_tetragonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm, models_directory, options=options)
    elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
        optimizer = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm, models_directory, options=options)
    elif bravais_lattice in ['mC', 'mP']:
        optimizer = get_monoclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm, models_directory, options=options)
    elif bravais_lattice in ['aP']:
        optimizer = get_triclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm, models_directory, options=options)
    optimizer.opt_params['iteration_info'][1]['n_iterations'] = n_iterations

    if bravais_lattice in ['cF', 'cI', 'cP']:
        n_peaks = 10
    else:
        n_peaks = 20
    if rank == 0:
        read_columns = [
            'identifier',
            'lattice_system',
            'bravais_lattice',
            'train',
            f'q2_{broadening_tag}',
            'reindexed_spacegroup_symbol_hm',
            'reindexed_unit_cell',
            ]
        bravais_lattice_data = pd.read_parquet(
            f'/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/generated_datasets/dataset_{bravais_lattice}.parquet',
            columns=read_columns
            )
        if use_train:
            bravais_lattice_data = bravais_lattice_data.loc[bravais_lattice_data['train']]
        else:
            bravais_lattice_data = bravais_lattice_data.loc[~bravais_lattice_data['train']]
        peaks = bravais_lattice_data[f'q2_{broadening_tag}']
        bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(len) >= n_peaks]
        peaks = bravais_lattice_data[f'q2_{broadening_tag}']
        bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(np.count_nonzero) >= n_peaks]
        data = bravais_lattice_data
        if not n_entries is None and data.shape[0] > n_entries:
            data = data.iloc[:n_entries]
        for rank_index in range(1, n_ranks):
            comm.send(data.iloc[rank_index::n_ranks], dest=rank_index)
        data = data.iloc[0::n_ranks]
    else:
        data = comm.recv(source=0)

    n_entries = data.shape[0]
    report_counts = {
        'Not found': 0,
        'Found': 0,
        'Off by two': 0,
        'Found explainers': 0,
        }
    # Entries whose contaminants could not be placed; they are left out of the counts.
    n_skipped = 0
    for entry_index in range(n_entries):
        entry = data.iloc[entry_index]
        unit_cell_true = get_partial_unit_cell(
            np.array(entry['reindexed_unit_cell']),
            lattice_system=BL_TO_LATTICE_SYSTEM[bravais_lattice],
            )
        try:
            q2 = prepare_peak_list(entry, condition, base_seed, n_peaks=n_peaks).q2_obs
        except ContaminantPlacementError:
            n_skipped += 1
            continue
        start = time.time()
        optimizer.run(q2=q2, n_top_candidates=n_top_candidates)
        correct, candidate_off_by_two = label_known_bl_batch(
            unit_cell_true, optimizer.top_unit_cell, bravais_lattice
            )
        found = np.any(correct)
        off_by_two = np.any(candidate_off_by_two)
        found_explainer = np.any(optimizer.top_M20 > 1000)
        if found:
            report_counts['Found'] += 1
        elif off_by_two:
            report_counts['Off by two'] += 1
        elif found_explainer:
            report_counts['Found explainers'] += 1
        else:
            report_counts['Not found'] += 1
        #print()
        #print(entry)
        #print(found, off_by_two, found_explainer)
        #print(np.column_stack((
        #    optimizer.top_unit_cell,
        #    optimizer.top_M20,
        #    optimizer.top_spacegroup
        #    )))
        #print()
        stop = time.time()
        #print(report_counts, rank, stop - start)
        #print()
    report_counts_gathered = comm.gather(report_counts, root=0)
    n_skipped_gathered = comm.gather(n_skipped, root=0)

    if rank == 0:
        report_counts_all = {
            'Not found': 0,
            'Found': 0,
            'Off by two': 0,
            'Found explainers': 0,
            }
        for rank_index in range(n_ranks):
            for key in report_counts_gathered[rank_index].keys():
                report_counts_all[key] += report_counts_gathered[rank_index][key]
        print(report_counts_all)
        print(f'Skipped (contaminants could not be placed): {sum(n_skipped_gathered)}')
        suffix = '_random' if random else ''
        file_name = os.path.join(
            '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/characterization/success_rate',
            f'{bravais_lattice}_error{q2_error_multiplier}_cont{n_contaminants}_candscale{n_candidates_scale}_niters{n_iterations}{suffix}.json'
        )
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(report_counts_all, f)
