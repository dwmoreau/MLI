"""
extremely small error:
cF:  100% {'Not found': 0, 'Found': 238, 'Off by two': 0, 'Found explainers': 0}
cI: 99.6% {'Not found': 1, 'Found': 249, 'Off by two': 0, 'Found explainers': 0}
cP: 99.8% {'Not found': 1, 'Found': 533, 'Off by two': 0, 'Found explainers': 0}
hP: 98.6% {'Not found': 4, 'Found': 493, 'Off by two': 0, 'Found explainers': 3}
hR: 99.0% {'Not found': 5, 'Found': 495, 'Off by two': 0, 'Found explainers': 0}
tI: 98.8% {'Not found': 6, 'Found': 494, 'Off by two': 0, 'Found explainers': 0}
tP: 97.2% {'Not found': 3, 'Found': 486, 'Off by two': 0, 'Found explainers': 11}
oC: 
oF: 97.0% {'Not found': 7, 'Found': 485, 'Off by two': 0, 'Found explainers': 8}
oI:
oP: 98.5% {'Not found': 1, 'Found': 447, 'Off by two': 0, 'Found explainers': 6}
mC:
mP:
aP:
"""
from mpi4py import MPI
import numpy as np
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

from Optimizer_mpi import OptimizerManager
from Optimizer_mpi import OptimizerWorker

from UtilitiesOptimizer import get_cubic_optimizer
from UtilitiesOptimizer import get_hexagonal_optimizer
from UtilitiesOptimizer import get_monoclinic_optimizer
from UtilitiesOptimizer import get_orthorhombic_optimizer
from UtilitiesOptimizer import get_rhombohedral_optimizer
from UtilitiesOptimizer import get_tetragonal_optimizer
from UtilitiesOptimizer import get_triclinic_optimizer
from UtilitiesOptimizer import validate_candidate


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    split_comm = comm.Split(color=rank, key=rank)

    load_data = True
    broadening_tag = '0.5'
    error_tag = '0.1'
    n_entries = 500
    q2_error_params = np.array([0.0001, 0.001]) / 1
    #q2_error_params = np.array([0.000000001, 0])
    n_top_candidates = 20
    #bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    bravais_lattices = ['oP']
    optimizer = dict.fromkeys(bravais_lattices)
    rng = np.random.default_rng()
    for bravais_lattice in bravais_lattices:
        if bravais_lattice in ['cF', 'cI', 'cP']:
            optimizer[bravais_lattice] = get_cubic_optimizer(bravais_lattice, broadening_tag, error_tag, split_comm)
        elif bravais_lattice in ['hP']:
            optimizer[bravais_lattice] = get_hexagonal_optimizer(bravais_lattice, broadening_tag, error_tag, split_comm)
        elif bravais_lattice in ['hR']:
            optimizer[bravais_lattice] = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, error_tag, split_comm)
        elif bravais_lattice in ['tI', 'tP']:
            optimizer[bravais_lattice] = get_tetragonal_optimizer(bravais_lattice, broadening_tag, error_tag, split_comm)
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer[bravais_lattice] = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, error_tag, split_comm)
        elif bravais_lattice in ['mC', 'mP']:
            optimizer[bravais_lattice] = get_monoclinic_optimizer(bravais_lattice, broadening_tag, error_tag, split_comm)
        elif bravais_lattice in ['aP']:
            optimizer[bravais_lattice] = get_triclinic_optimizer(bravais_lattice, broadening_tag, error_tag, split_comm)

    if rank == 0:
        if load_data:
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
            for bravais_lattice in bravais_lattices:
                n_peaks = 20
                bravais_lattice_data = pd.read_parquet(
                    f'/Users/DWMoreau/MLI/data/GeneratedDatasets/dataset_{bravais_lattice}.parquet',
                    columns=read_columns
                    )
                bravais_lattice_data = bravais_lattice_data.loc[~bravais_lattice_data['train']]
                peaks = bravais_lattice_data[f'q2_{broadening_tag}']
                bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(len) >= n_peaks]
                peaks = bravais_lattice_data[f'q2_{broadening_tag}']
                bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(np.count_nonzero) >= n_peaks]
                q2 = np.zeros((bravais_lattice_data.shape[0], n_peaks))
                hkl = np.zeros((bravais_lattice_data.shape[0], n_peaks, 3))
                for entry_index in range(bravais_lattice_data.shape[0]):
                    q2[entry_index] = np.array(bravais_lattice_data[f'q2_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                    hkl[entry_index, :, 0] = np.array(bravais_lattice_data[f'reindexed_h_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                    hkl[entry_index, :, 1] = np.array(bravais_lattice_data[f'reindexed_k_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                    hkl[entry_index, :, 2] = np.array(bravais_lattice_data[f'reindexed_l_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                sigma_error = q2_error_params[0] + q2 * q2_error_params[1]
                q2 += rng.normal(loc=0, scale=sigma_error)
                bravais_lattice_data['q2'] = list(q2)
                bravais_lattice_data['reindexed_hkl'] = list(hkl)
                bravais_lattice_data.drop(columns=drop_columns, inplace=True)
                data.append(bravais_lattice_data)
            data = pd.concat(data)
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
        print(entry)
        unit_cell_true = np.array(entry['reindexed_unit_cell'])
        for bravais_lattice in bravais_lattices:
            top_unit_cell, top_M20 = optimizer[bravais_lattice].run(entry, n_top_candidates)
            found = False
            off_by_two = False
            found_explainer = False
            print(np.column_stack((top_unit_cell, top_M20)))
            for candidate_index in range(top_unit_cell.shape[0]):
                correct, off_by_two = validate_candidate(
                    unit_cell_true=unit_cell_true,
                    unit_cell_pred=top_unit_cell[candidate_index],
                    bravais_lattice_pred=bravais_lattice,
                    )
                if correct:
                    found = True
                if off_by_two:
                    off_by_two = True
                if np.any(top_M20 > 1000):
                    found_explainer = True
            if found:
                report_counts['Found'] += 1
            elif off_by_two:
                report_counts['off_by_two'] += 1
            elif found_explainer:
                report_counts['Found explainers'] += 1
            else:
                report_counts['Not found'] += 1
            print(report_counts, rank)
            print()
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
