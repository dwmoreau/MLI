import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

from Utilities import get_xnn_from_unit_cell
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

    load_data = True
    broadening_tag = '1'
    error_tag = '1'
    n_entries = 100
    q2_error_params = np.array([0.0001, 0.001]) / 1
    #q2_error_params = np.array([0.000000001, 0])
    
    n_top_candidates = 100
    #bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['cF', 'cI', 'cP']
    bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    options = {
        'convergence_testing': True,
        'convergence_distances': np.logspace(-4, -1.5, 50),
        'convergence_candidates': 100,
        }
    optimizer = dict.fromkeys(bravais_lattices)
    rng = np.random.default_rng(0)
    for bravais_lattice in bravais_lattices:
        if bravais_lattice in ['cF', 'cI', 'cP']:
            optimizer[bravais_lattice] = get_cubic_optimizer(bravais_lattice, broadening_tag, split_comm, options)
        elif bravais_lattice in ['hP']:
            optimizer[bravais_lattice] = get_hexagonal_optimizer(bravais_lattice, broadening_tag, split_comm, options)
        elif bravais_lattice in ['hR']:
            optimizer[bravais_lattice] = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, split_comm, options)
        elif bravais_lattice in ['tI', 'tP']:
            optimizer[bravais_lattice] = get_tetragonal_optimizer(bravais_lattice, broadening_tag, split_comm, options)
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer[bravais_lattice] = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, split_comm, options)
        elif bravais_lattice in ['mC', 'mP']:
            optimizer[bravais_lattice] = get_monoclinic_optimizer(bravais_lattice, broadening_tag, split_comm, options)
        elif bravais_lattice in ['aP']:
            optimizer[bravais_lattice] = get_triclinic_optimizer(bravais_lattice, broadening_tag, split_comm, options)

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
                'reindexed_xnn',
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
                # SELECT THE TRAINING DATA FOR RADIUS OF CONVERGENCE 
                bravais_lattice_data = bravais_lattice_data.loc[bravais_lattice_data['train']]
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
                if not n_entries is None and bravais_lattice_data.shape[0] > n_entries:
                    bravais_lattice_data = bravais_lattice_data.sample(n=n_entries, random_state=rng)
                data.append(bravais_lattice_data)
            data = pd.concat(data)
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

    print('Calculating Xnn from unit cell - delete this line after dataset regeneration')
    unit_cell = np.stack(data['reindexed_unit_cell'])
    xnn = get_xnn_from_unit_cell(unit_cell, partial_unit_cell=False)
    data['reindexed_xnn'] = list(xnn)

    for bravais_lattice in bravais_lattices:
        bl_data = data[data['bravais_lattice'] == bravais_lattice]
        found = np.zeros((len(options['convergence_distances']), len(bl_data)))
        for entry_index in range(len(bl_data)):
            entry = bl_data.iloc[entry_index]
            #print(entry)
            unit_cell_true = np.array(entry['reindexed_unit_cell'])
            optimizer[bravais_lattice].run(entry=entry, n_top_candidates=n_top_candidates)
            for candidate_index in range(optimizer[bravais_lattice].top_unit_cell.shape[0]):
                correct, off_by_two = validate_candidate_known_bl(
                    unit_cell_true=unit_cell_true,
                    unit_cell_pred=optimizer[bravais_lattice].top_unit_cell[candidate_index],
                    bravais_lattice_pred=bravais_lattice,
                    )
                if correct:
                    distance_index = candidate_index // options['convergence_candidates']
                    found[distance_index, entry_index] += 1
        found_gathered = comm.gather(found, root=0)
        if rank == 0:
            found_gathered = np.concatenate(found_gathered, axis=1)
            found_rate = found_gathered.mean(axis=1) / options['convergence_candidates']
            fig, axes = plt.subplots(1, 1, figsize=(6, 3))
            axes.loglog(options['convergence_distances'], found_rate, marker='.')
            axes.set_ylabel('Success Rate')
            axes.set_xlabel('Xnn Perturbation Amount (1/$\mathrm{\AA}^2$)')
            axes.set_title(bravais_lattice)

            xlim = axes.get_xlim()
            ylim = axes.get_ylim()
            radius = np.zeros((5, 2))
            radius[:, 0] = [5, 10, 25, 50, 100]
            for radius_index in range(5):
                n = int(radius[radius_index, 0])
                radius[radius_index, 1] = options['convergence_distances'][np.argwhere(found_rate < 1/n)[0][0]]
                axes.plot(
                    [xlim[0], radius[radius_index, 1]], 1/n*np.ones(2),
                    linestyle='dotted', color=[0, 0, 0], linewidth=1
                    )
                axes.plot(
                    radius[radius_index, 1]*np.ones(2), [ylim[0], 1/n],
                    linestyle='dotted', color=[0, 0, 0], linewidth=1
                    )
                axes.annotate(
                    f'{n:3d} {int(100*1/n)}%: {radius[radius_index, 1]:0.5f}',
                    (2*xlim[0], 1/n),
                    verticalalignment='bottom'
                    )
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)
            fig.tight_layout()
            fig.savefig(f'figures/{bravais_lattice}_radius_of_convergence_err_{error_tag}.png')
            plt.close()
            np.save(f'data/radius_of_convergence_{error_tag}_{bravais_lattice}.npy', radius)
