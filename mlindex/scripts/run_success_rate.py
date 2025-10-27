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

from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.optimization.UtilitiesOptimizer import get_cubic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_hexagonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_monoclinic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_orthorhombic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_rhombohedral_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_tetragonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_triclinic_optimizer
from mlindex.optimization.CandidateValidation import validate_candidate_known_bl


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    split_comm = comm.Split(color=rank, key=rank)

    broadening_tag = '1'
    n_entries = 10000

    bravais_lattice = sys.argv[1]
    q2_error_multiplier = int(sys.argv[2])
    n_contaminants = int(sys.argv[3])
    n_candidates_scale = float(sys.argv[4])
    use_train = bool(sys.argv[5])
    n_iterations = int(sys.argv[6])
    
    n_top_candidates = 20
    rng = np.random.default_rng(0)
    if bravais_lattice in ['cF', 'cI', 'cP']:
        optimizer = get_cubic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
    elif bravais_lattice in ['hP']:
        optimizer = get_hexagonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
    elif bravais_lattice in ['hR']:
        optimizer = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
    elif bravais_lattice in ['tI', 'tP']:
        optimizer = get_tetragonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
    elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
        optimizer = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
    elif bravais_lattice in ['mC', 'mP']:
        optimizer = get_monoclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
    elif bravais_lattice in ['aP']:
        optimizer = get_triclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
    optimizer.opt_params['iteration_info'][1]['n_iterations'] = n_iterations

    if rank == 0:
        data = []
        read_columns = [
            'lattice_system',
            'bravais_lattice',
            'train',
            f'q2_{broadening_tag}',
            f'reindexed_h_{broadening_tag}',
            f'reindexed_k_{broadening_tag}',
            f'reindexed_l_{broadening_tag}',
            'reindexed_spacegroup_symbol_hm',
            'reindexed_unit_cell',
            ]
        drop_columns = [
            f'q2_{broadening_tag}',
            f'reindexed_h_{broadening_tag}',
            f'reindexed_k_{broadening_tag}',
            f'reindexed_l_{broadening_tag}',
            ]
        if bravais_lattice in ['cF', 'cI', 'cP']:
            n_peaks = 10
        else:
            n_peaks = 20
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
        q2 = np.zeros((bravais_lattice_data.shape[0], n_peaks))
        hkl = np.zeros((bravais_lattice_data.shape[0], n_peaks, 3))
        for entry_index in range(bravais_lattice_data.shape[0]):
            q2[entry_index, :n_peaks] = np.array(bravais_lattice_data[f'q2_{broadening_tag}'].iloc[entry_index])[:n_peaks]
            hkl[entry_index, :n_peaks, 0] = np.array(bravais_lattice_data[f'reindexed_h_{broadening_tag}'].iloc[entry_index])[:n_peaks]
            hkl[entry_index, :n_peaks, 1] = np.array(bravais_lattice_data[f'reindexed_k_{broadening_tag}'].iloc[entry_index])[:n_peaks]
            hkl[entry_index, :n_peaks, 2] = np.array(bravais_lattice_data[f'reindexed_l_{broadening_tag}'].iloc[entry_index])[:n_peaks]
        
        q2, hkl = add_q2_error(q2, hkl, q2_error_multiplier, rng)
        q2, hkl = add_contaminants(q2, hkl, n_contaminants, rng)

        bravais_lattice_data['q2'] = list(q2)
        bravais_lattice_data['reindexed_hkl'] = list(hkl)
        bravais_lattice_data.drop(columns=drop_columns, inplace=True)
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
    for entry_index in range(n_entries):
        entry = data.iloc[entry_index]
        unit_cell_true = np.array(entry['reindexed_unit_cell'])
        start = time.time()
        optimizer.run(entry=entry, n_top_candidates=n_top_candidates)
        found = False
        off_by_two = False
        found_explainer = False
        for candidate_index in range(optimizer.top_unit_cell.shape[0]):
            correct, off_by_two = validate_candidate_known_bl(
                unit_cell_true=unit_cell_true,
                unit_cell_pred=optimizer.top_unit_cell[candidate_index],
                bravais_lattice_pred=bravais_lattice,
                )
            if correct:
                found = True
            if off_by_two:
                off_by_two = True
            if np.any(optimizer.top_M20 > 1000):
                found_explainer = True
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
        file_name = os.path.join(
            '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/characterization/success_rate',
            f'{bravais_lattice}_error{q2_error_multiplier}_cont{n_contaminants}_candscale{n_candidates_scale}_niters{n_iterations}.json'
        )
        with open(file_name, 'w') as f:
            json.dump(report_counts_all, f)
