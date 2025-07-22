from mpi4py import MPI
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import time

from Utilities import get_peak_generation_info
from UtilitiesOptimizer import get_cubic_optimizer
from UtilitiesOptimizer import get_hexagonal_optimizer
from UtilitiesOptimizer import get_monoclinic_optimizer
from UtilitiesOptimizer import get_orthorhombic_optimizer
from UtilitiesOptimizer import get_rhombohedral_optimizer
from UtilitiesOptimizer import get_tetragonal_optimizer
from UtilitiesOptimizer import get_triclinic_optimizer
from UtilitiesOptimizer import validate_candidate_known_bl


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    split_comm = comm.Split(color=rank, key=rank)

    #np.seterr(all='raise')

    load_data = True
    broadening_tag = '1'
    n_entries = 1000
    n_candidates_scale = 50/100

    peak_generation_info = get_peak_generation_info()
    q2_error_params = 1 * peak_generation_info['q2_error_params']
    q2_broadening_params = peak_generation_info['broadening_params']
    n_contaminants = 1
    n_top_candidates = 20
    #bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['hP', 'hR', 'tI', 'tP']
    bravais_lattices = ['aP']
    optimizer = dict.fromkeys(bravais_lattices)
    rng = np.random.default_rng(0)
    for bravais_lattice in bravais_lattices:
        if bravais_lattice in ['cF', 'cI', 'cP']:
            optimizer[bravais_lattice] = get_cubic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
        elif bravais_lattice in ['hP']:
            optimizer[bravais_lattice] = get_hexagonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
        elif bravais_lattice in ['hR']:
            optimizer[bravais_lattice] = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
        elif bravais_lattice in ['tI', 'tP']:
            optimizer[bravais_lattice] = get_tetragonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer[bravais_lattice] = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
        elif bravais_lattice in ['mC', 'mP']:
            optimizer[bravais_lattice] = get_monoclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)
        elif bravais_lattice in ['aP']:
            optimizer[bravais_lattice] = get_triclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, split_comm)

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
                if bravais_lattice in ['cF', 'cI', 'cP']:
                    n_peaks_contaminants = 10
                else:
                    n_peaks_contaminants = 20
                bravais_lattice_data = pd.read_parquet(
                    f'/Users/DWMoreau/MLI/data/GeneratedDatasets/dataset_{bravais_lattice}.parquet',
                    columns=read_columns
                    )
                bravais_lattice_data = bravais_lattice_data.loc[~bravais_lattice_data['train']]
                peaks = bravais_lattice_data[f'q2_{broadening_tag}']
                bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(len) >= n_peaks]
                peaks = bravais_lattice_data[f'q2_{broadening_tag}']
                bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(np.count_nonzero) >= n_peaks]
                q2 = np.zeros((bravais_lattice_data.shape[0], n_peaks + n_contaminants))
                hkl = np.zeros((bravais_lattice_data.shape[0], n_peaks + n_contaminants, 3))
                for entry_index in range(bravais_lattice_data.shape[0]):
                    q2[entry_index, :n_peaks] = np.array(bravais_lattice_data[f'q2_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                    hkl[entry_index, :n_peaks, 0] = np.array(bravais_lattice_data[f'reindexed_h_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                    hkl[entry_index, :n_peaks, 1] = np.array(bravais_lattice_data[f'reindexed_k_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                    hkl[entry_index, :n_peaks, 2] = np.array(bravais_lattice_data[f'reindexed_l_{broadening_tag}'].iloc[entry_index])[:n_peaks]
                sigma_error = q2_error_params[0] + q2[:, :n_peaks] * q2_error_params[1]
                q2[:, :n_peaks] += rng.normal(loc=0, scale=sigma_error)
                q2[:, :n_peaks] = np.abs(q2[:, :n_peaks])
                breadth = q2_broadening_params[0] + q2_broadening_params[1] * q2[:, :n_peaks_contaminants]
                if n_contaminants > 0:
                    for entry_index in range(bravais_lattice_data.shape[0]):
                        status = True
                        while status:
                            q2_contaminants = rng.uniform(
                                low=0.5*q2[entry_index, 0],
                                high=q2[entry_index, n_peaks_contaminants - 1],
                                size=n_contaminants
                                )
                            difference = np.abs(
                                q2_contaminants[np.newaxis]
                                - q2[entry_index, :n_peaks_contaminants][:, np.newaxis]
                                ).min(axis=0)
                            status = np.any(difference[np.newaxis] < 0.5*breadth[entry_index][:, np.newaxis])

                        q2[entry_index, n_peaks:] = q2_contaminants
                        sort_indices = np.argsort(q2[entry_index])
                        q2[entry_index] = q2[entry_index, sort_indices]
                        hkl[entry_index, :, 0] = hkl[entry_index, sort_indices, 0]
                        hkl[entry_index, :, 1] = hkl[entry_index, sort_indices, 1]
                        hkl[entry_index, :, 2] = hkl[entry_index, sort_indices, 2]
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
        unit_cell_true = np.array(entry['reindexed_unit_cell'])
        for bravais_lattice in bravais_lattices:
            start = time.time()
            optimizer[bravais_lattice].run(entry=entry, n_top_candidates=n_top_candidates)
            found = False
            off_by_two = False
            found_explainer = False
            #print(np.column_stack((
            #    optimizer[bravais_lattice].top_unit_cell,
            #    optimizer[bravais_lattice].top_M20,
            #    optimizer[bravais_lattice].top_spacegroup
            #    )))
            for candidate_index in range(optimizer[bravais_lattice].top_unit_cell.shape[0]):
                correct, off_by_two = validate_candidate_known_bl(
                    unit_cell_true=unit_cell_true,
                    unit_cell_pred=optimizer[bravais_lattice].top_unit_cell[candidate_index],
                    bravais_lattice_pred=bravais_lattice,
                    )
                if correct:
                    found = True
                if off_by_two:
                    off_by_two = True
                if np.any(optimizer[bravais_lattice].top_M20 > 1000):
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
                #print(np.column_stack((
                #    optimizer[bravais_lattice].top_unit_cell,
                #    optimizer[bravais_lattice].top_M20,
                #    optimizer[bravais_lattice].top_spacegroup
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
