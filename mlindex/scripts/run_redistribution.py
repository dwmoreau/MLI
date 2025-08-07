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
os.environ["KERAS_BACKEND"] = "torch"
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell
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
    cr_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/characterization/roc/data'
    convergence_radius = {
        'cF': np.load(os.path.join(cr_dir, 'cF_roc_peaks10_drop8_iter100_sampQ2.npy')),
        'cI': np.load(os.path.join(cr_dir, 'cI_roc_peaks10_drop8_iter100_sampQ2.npy')),
        'cP': np.load(os.path.join(cr_dir, 'cP_roc_peaks10_drop8_iter100_sampQ2.npy')),
        'hP': np.load(os.path.join(cr_dir, 'hP_roc_peaks20_drop17_iter100_sampQ2.npy')),
        'hR': np.load(os.path.join(cr_dir, 'hR_roc_peaks20_drop17_iter100_sampQ2.npy')),
        'tI': np.load(os.path.join(cr_dir, 'tI_roc_peaks20_drop17_iter100_sampQ2.npy')),
        'tP': np.load(os.path.join(cr_dir, 'tP_roc_peaks20_drop17_iter100_sampQ2.npy')),
        'oC': np.load(os.path.join(cr_dir, 'oC_roc_peaks20_drop16_iter100_sampQ2.npy')),
        'oF': np.load(os.path.join(cr_dir, 'oF_roc_peaks20_drop16_iter100_sampQ2.npy')),
        'oI': np.load(os.path.join(cr_dir, 'oI_roc_peaks20_drop16_iter100_sampQ2.npy')),
        'oP': np.load(os.path.join(cr_dir, 'oP_roc_peaks20_drop16_iter100_sampQ2.npy')),
        'mC': np.load(os.path.join(cr_dir, 'mC_roc_peaks20_drop14_iter100_sampQ2.npy')),
        'mP': np.load(os.path.join(cr_dir, 'mP_roc_peaks20_drop14_iter100_sampQ2.npy')),
        'aP': np.load(os.path.join(cr_dir, 'aP_roc_peaks20_drop11_iter100_sampQ2.npy'))
        }

    #bravais_lattices = ['cF']
    #bravais_lattices = ['cI']
    #bravais_lattices = ['cP']
    #bravais_lattices = ['hP']
    #bravais_lattices = ['hR']
    #bravais_lattices = ['tI']
    #bravais_lattices = ['tP']
    #bravais_lattices = ['oC']
    #bravais_lattices = ['oF']
    #bravais_lattices = ['oI']
    #bravais_lattices = ['oP']
    #bravais_lattices = ['mC']
    #bravais_lattices = ['mP']
    #bravais_lattices = ['aP']
    #bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP']
    bravais_lattices = ['cF', 'cI', 'cP', 'aP']

    n_entries = 10000
    options = {
        'redistribution_testing': True,
        'convergence_radius': convergence_radius,
        'max_neighbors_grid': np.arange(10, 210, 10),
        }
    rng = np.random.default_rng(0)
    for bravais_lattice in bravais_lattices:
        if bravais_lattice in ['cF', 'cI', 'cP']:
            optimizer = get_cubic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=options)
        elif bravais_lattice in ['hP']:
            optimizer = get_hexagonal_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=options)
        elif bravais_lattice in ['hR']:
            optimizer = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=options)
        elif bravais_lattice in ['tI', 'tP']:
            optimizer = get_tetragonal_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=options)
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=options)
        elif bravais_lattice in ['mC', 'mP']:
            optimizer = get_monoclinic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=options)
        elif bravais_lattice in ['aP']:
            optimizer = get_triclinic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=options)

        if rank == 0:
            read_columns = [
                'lattice_system',
                'bravais_lattice',
                'train',
                f'q2_{broadening_tag}',
                'reindexed_spacegroup_symbol_hm',
                'reindexed_unit_cell',
                'reindexed_xnn',
                ]
            drop_columns = [
                f'q2_{broadening_tag}',
                ]
            if bravais_lattice in ['cF', 'cI', 'cP']:
                n_peaks = 10
            else:
                n_peaks = 20
            bravais_lattice_data = pd.read_parquet(
                f'/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/generated_datasets/dataset_{bravais_lattice}.parquet',
                columns=read_columns
                )
            # SELECT THE TRAINING DATA FOR RADIUS OF CONVERGENCE 
            bravais_lattice_data = bravais_lattice_data.loc[bravais_lattice_data['train']]
            peaks = bravais_lattice_data[f'q2_{broadening_tag}']
            bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(len) >= n_peaks]
            peaks = bravais_lattice_data[f'q2_{broadening_tag}']
            bravais_lattice_data = bravais_lattice_data.loc[peaks.apply(np.count_nonzero) >= n_peaks]
            q2 = np.zeros((bravais_lattice_data.shape[0], n_peaks))
            for entry_index in range(bravais_lattice_data.shape[0]):
                q2[entry_index] = np.array(bravais_lattice_data[f'q2_{broadening_tag}'].iloc[entry_index])[:n_peaks]
            bravais_lattice_data['q2'] = list(add_q2_error(q2, None, 1, rng))
            bravais_lattice_data.drop(columns=drop_columns, inplace=True)
            if not n_entries is None and bravais_lattice_data.shape[0] > n_entries:
                bravais_lattice_data = bravais_lattice_data.sample(n=n_entries, random_state=rng)
            for rank_index in range(1, n_ranks):
                comm.send(bravais_lattice_data.iloc[rank_index::n_ranks], dest=rank_index)
            bravais_lattice_data = bravais_lattice_data.iloc[0::n_ranks]
        else:
            bravais_lattice_data = comm.recv(source=0)

        n_entries_rank = bravais_lattice_data.shape[0]
        results_rank = np.zeros((n_entries_rank, 2))
        for entry_index in range(n_entries_rank):
            entry = bravais_lattice_data.iloc[entry_index]
            max_neighbors, neighbor_radius = optimizer.run(
                entry=entry, n_top_candidates=20
                )
            results_rank[entry_index, 0] = max_neighbors
            results_rank[entry_index, 1] = neighbor_radius
            #print(rank, max_neighbors, neighbor_radius)
        results = comm.gather(results_rank)
        if rank == 0:
            results = np.vstack(results)
            np.save(
                os.path.join(
                    '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/characterization/redistribution',
                    f'redistribute_{bravais_lattice}.npy'
                ),
                results
                )
