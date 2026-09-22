import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['ATLAS_NUM_THREADS'] = '1'
os.environ['SKLEARN_N_JOBS'] = '1'

import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

import mlindex
from mlindex.optimization.GeneratorPools import generate_candidate_pools
from mlindex.optimization.UtilitiesOptimizer import get_cubic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_hexagonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_monoclinic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_orthorhombic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_rhombohedral_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_tetragonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_triclinic_optimizer
from mlindex.utilities.ErrorAdder import add_q2_error


def ensemble_refine(distance, generator_names, convergence_radius, rng):
    def target_function(params, distance_all, x, N_success, rng, n_total, return_F=False):
        # convergence radius has rows:
        # 0: distance
        # 1: success rate
        n_generators = distance_all.shape[1]
        n_gen = np.round(
            n_total * scipy.special.softmax(params[:n_generators]), decimals=0
            ).astype(int)
        n_total = n_gen.sum()
        distance = np.zeros(n_total)
        start = 0
        for generator_index in range(n_generators):
            distance[start: start + n_gen[generator_index]] = distance_all[:n_gen[generator_index], generator_index]
            start += n_gen[generator_index]

        # Calculate N_success(delta xnn) from the convergence radius
        # Calcuate N_gen(delta xnn)
        bins = np.concatenate([[0], x])
        distance_hist, _ = np.histogram(distance, bins=bins)
        N = np.cumsum(distance_hist)

        in_range = N_success != np.inf
        # Calculate target function
        F = (N[in_range] - N_success[in_range]) / N_success[in_range]
        term_0 = 0
        if np.max(F) < 0:
            term_0 += 100
        # term_1 represents efficiency. Integrate F. This gives a total number of excess entries.
        term_1 = -np.mean(
            np.trapezoid(F, x[in_range]) / np.trapezoid(x[in_range])
            )
        #if term_0 > 0:
        #    print(term_0, term_1)
        if return_F:
            return F, N
        else:
            return term_0 + term_1

    n_optimizations = 10
    n_generators = distance.shape[1]
    # Parameterization:
    # 0 -> n_generators-1: Logit for generator sampling fraction
    bounds = [[-np.inf, np.inf] for _ in range(n_generators)]
    n_total = distance.shape[0]

    x_opt = np.zeros((n_optimizations, n_generators))
    x = convergence_radius[0]
    success_rate = convergence_radius[1]
    N_success = 1/success_rate
    in_range = success_rate > 0.01
    N_success[~in_range] = np.inf
    for opt_index in range(n_optimizations):
        x0 = rng.normal(size=n_generators)
        initial_simplex = rng.normal(size=(n_generators+1, n_generators))    
        opt_results = scipy.optimize.minimize(
            target_function,
            x0=x0,
            method='Nelder-Mead',
            args=(distance, x, N_success, rng, n_total),
            options={'initial_simplex': initial_simplex},
            bounds=bounds,
            )
        x_opt[opt_index] = opt_results.x
    print(generator_names)
    print(np.round(scipy.special.softmax(x_opt.mean(axis=0)) * n_total))
    #print(scipy.special.softmax(opt_results.x[:-1]))
    #print(opt_results)
    output = {}
    F, N = target_function(
        x_opt.mean(axis=0), distance, x, N_success, rng, n_total, return_F=True
        )
    #fig, axes = plt.subplots(2, 1 ,figsize=(6, 4), sharex=True)
    #axes[0].plot(x[in_range], F[0])
    #axes[1].plot(x, N[0])
    #axes[1].plot(x, N_success)
    #plt.show()
    #print('Mean distance of the top 10 entries:')
    mean_distance = np.zeros(len(generator_names))
    for index, name in enumerate(generator_names):
        output[name] = x_opt.mean(axis=0)[index]
        output[f'{name}_mean_dist'] = np.sort(distance[:, index])[:10].mean()
        mean_distance[index] = np.sort(distance[:, index])[:10].mean()
        #print(name, output[f'{name}_mean_dist'])
    print(np.round(mean_distance / mean_distance.min(), decimals=1))
    print()
    return output


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    split_comm = comm.Split(color=rank, key=rank)
    project_path = Path(mlindex.__path__[0]).parent

    load_data = True
    broadening_tag = '1'
    n_trials = 10000
    rng = np.random.default_rng(0)
    #rng = np.random.default_rng()

    #bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP']
    #bravais_lattices = ['cF', 'cI', 'cP']
    #bravais_lattices = ['cP']
    #bravais_lattices = ['hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['cF', 'cI', 'cP']
    #bravais_lattices = ['oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['tI', 'tP','oP', 'mC', 'mP', 'aP']
    #bravais_lattices = ['mC', 'mP', 'aP']

    bravais_lattices = [sys.argv[1]]

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

    candidates_per_model = {
        'cF': 100,
        'cI': 100,
        'cP': 100,
        'hP': 1500,
        'hR': 1500,
        'tI': 1500,
        'tP': 1500,
        'oC': 2500,
        'oF': 2500,
        'oI': 2500,
        'oP': 2500,
        'mC': 4000,
        'mP': 4000,
        'aP': 4000,
        }

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

    for bravais_lattice in bravais_lattices:
        print(f'Loading optimizer for {bravais_lattice}')
        if bravais_lattice in ['cF', 'cI', 'cP']:
            optimizer = get_cubic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, project_path)
        elif bravais_lattice in ['hP']:
            optimizer = get_hexagonal_optimizer(bravais_lattice, broadening_tag, 1, split_comm, project_path)
        elif bravais_lattice in ['hR']:
            optimizer = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, 1, split_comm, project_path)
        elif bravais_lattice in ['tI', 'tP']:
            optimizer = get_tetragonal_optimizer(bravais_lattice, broadening_tag, 1, split_comm, project_path)
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, project_path)
        elif bravais_lattice in ['mC', 'mP']:
            optimizer = get_monoclinic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, project_path)
        elif bravais_lattice in ['aP']:
            optimizer = get_triclinic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, project_path)
        
        if rank == 0:
            if bravais_lattice in ['cF', 'cI', 'cP']:
                n_peaks = 10
            else:
                n_peaks = 20
            bravais_lattice_data = pd.read_parquet(
                f'/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/generated_datasets/dataset_{bravais_lattice}.parquet',
                columns=read_columns
                )
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
            if n_trials < len(bravais_lattice_data):
                bravais_lattice_data = bravais_lattice_data.sample(
                    n=n_trials,
                    replace=False,
                    random_state=rng
                    )
            for rank_index in range(1, n_ranks):
                comm.send(bravais_lattice_data.iloc[rank_index::n_ranks], dest=rank_index)
            bravais_lattice_data = bravais_lattice_data.iloc[0::n_ranks]
        else:
            bravais_lattice_data = comm.recv(source=0)
            
        output = []
        for trial_index in range(len(bravais_lattice_data)):
            xnn, xnn_true, generator_names = generate_candidate_pools(
                optimizer,
                bravais_lattice_data.iloc[trial_index],
                candidates_per_model=candidates_per_model[bravais_lattice],
                rng=rng,
                )
            distance = np.linalg.norm(xnn - xnn_true, axis=-1)
            output.append(ensemble_refine(
                distance,
                generator_names,
                convergence_radius[bravais_lattice],
                rng=rng,
                ))
        
        if rank == 0:
            for rank_index in range(1, n_ranks):
                output += comm.recv(source=rank_index)
            df = pd.DataFrame(output)
            df.to_csv(os.path.join(
                '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/characterization/ensemble',
                f'ensemble_{bravais_lattice}.csv'
            ))
        else:
            comm.send(output, dest=0)
        
