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
from tqdm import tqdm
import time
import sys

from mlindex.optimization.UtilitiesOptimizer import get_cubic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_hexagonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_monoclinic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_orthorhombic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_rhombohedral_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_tetragonal_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_triclinic_optimizer
from mlindex.optimization.CandidateValidation import validate_candidate
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell
from mlindex.utilities.Reindexing import reindex_entry_triclinic
from mlindex.utilities.ErrorAdder import add_q2_error


def evaluate_regression(optimizer, entry, candidates_per_model, rng):
    integral_filter_top_n = None
    n_sub_generators = dict()
    candidates_per_sub_model = dict()
    generator_names = []
    for generator_info in optimizer.opt_params['generator_info']:
        if generator_info['generator'] in n_sub_generators.keys():
            n_sub_generators[generator_info['generator']] += 1
        else:
            n_sub_generators[generator_info['generator']] = 1
            generator_names.append(generator_info['generator'])
    for key in n_sub_generators.keys():
        if n_sub_generators[key] == 1:
            candidates_per_sub_model[key] = candidates_per_model
        else:
            candidates_per_sub_model[key] = candidates_per_model // n_sub_generators[key]
    distance = np.full(
        (candidates_per_model, len(n_sub_generators.keys())),
        np.nan
        )

    tree_time = 0
    integral_filter_time = 0
    template_time = 0
    random_time = 0

    xnn_true = np.array(entry['reindexed_xnn'])[optimizer.wrapper.data_params['unit_cell_indices']]
    q2 = np.array(entry['q2'])[:optimizer.n_peaks]

    for generator_info in optimizer.opt_params['generator_info']:
        start_time = time.time()
        if generator_info['generator'] == 'trees':
            generator_unit_cells = optimizer.wrapper.random_forest_generator[generator_info['split_group']].generate(
                candidates_per_sub_model[generator_info['generator']], rng,  q2,
                )
            tree_time += time.time() - start_time
        elif generator_info['generator'] == 'integral_filter':
            if integral_filter_top_n is None:
                integral_filter_top_n = optimizer.wrapper.integral_filter_generator[generator_info['split_group']].model_params['n_volumes']
            generator_unit_cells = optimizer.wrapper.integral_filter_generator[generator_info['split_group']].generate(
                candidates_per_sub_model[generator_info['generator']], rng, q2,
                top_n=integral_filter_top_n,
                batch_size=2,
                )
            integral_filter_time += time.time() - start_time
        elif generator_info['generator'] == 'templates':
            generator_unit_cells = optimizer.wrapper.miller_index_templator[optimizer.bravais_lattice].generate(
                candidates_per_sub_model[generator_info['generator']], rng, q2, 
                )
            template_time += time.time() - start_time
        elif generator_info['generator'] == 'predicted_volume':
            generator_unit_cells = optimizer.wrapper.random_unit_cell_generator[optimizer.bravais_lattice].generate(
                candidates_per_sub_model[generator_info['generator']], rng, q2,
                model=generator_info['generator'],
                )
            random_time += time.time() - start_time

        generator_unit_cells = fix_unphysical(
            unit_cell=generator_unit_cells,
            rng=rng,
            minimum_unit_cell=optimizer.opt_params['minimum_uc'],
            maximum_unit_cell=optimizer.opt_params['maximum_uc'],
            lattice_system=optimizer.wrapper.data_params['lattice_system']
            )
        if optimizer.wrapper.data_params['lattice_system'] == 'triclinic':
            generator_unit_cells, _ = reindex_entry_triclinic(generator_unit_cells)
        generator_xnn = get_xnn_from_unit_cell(
            generator_unit_cells,
            partial_unit_cell=True,
            lattice_system=optimizer.wrapper.data_params['lattice_system']
            )
        generator_distance = np.linalg.norm(generator_xnn - xnn_true[np.newaxis], axis=1)
        generator_index = list(n_sub_generators.keys()).index(generator_info['generator'])
        if n_sub_generators[generator_info['generator']] == 1:
            distance[:, generator_index] = generator_distance
        else:
            start = np.argwhere(np.isnan(distance[:, generator_index]))[0][0]
            stop = start + candidates_per_sub_model[generator_info['generator']]
            distance[start: stop, generator_index] = generator_distance

    # Randomly permute the Tree distances because they are ordered based on the
    # split group or dominant zone bin in the case of the RF model.
    tree_index = list(n_sub_generators.keys()).index('trees')
    distance[:, tree_index] = rng.permutation(distance[:, tree_index])

    # The integral filter model distances are also ordered based on the split group.
    # There is also an ordering based on the "top_n" predictions. The first top_n predictions
    # are the top_n most probable unit cells. The rest of the predictions are based on
    # randomly sampling their Miller Indices.
    # Create groupings of the top_n and rest of the predictions. Permute separately. Then
    # combine with the top_n first.
    integral_filter_index = list(n_sub_generators.keys()).index('integral_filter')

    if integral_filter_top_n < candidates_per_sub_model['integral_filter']:
        n_lower = (candidates_per_sub_model['integral_filter'] - integral_filter_top_n)
        distance_top_n = np.zeros(integral_filter_top_n * n_sub_generators['integral_filter'])
        distance_lower = np.zeros(n_lower * n_sub_generators['integral_filter'])
    
        start = 0
        for sub_index in range(n_sub_generators['integral_filter']):
            distance_top_n[sub_index*integral_filter_top_n: (sub_index+1)*integral_filter_top_n] = distance[
                start: start + integral_filter_top_n,
                integral_filter_index
                ]
            distance_lower[sub_index*n_lower: (sub_index+1)*n_lower] = distance[
                start + integral_filter_top_n: start + candidates_per_sub_model['integral_filter'],
                integral_filter_index
                ]
            start += candidates_per_sub_model['integral_filter']
        n_total_candidates = n_sub_generators['integral_filter']*candidates_per_sub_model['integral_filter']
        distance[:n_total_candidates, integral_filter_index] = np.concatenate([
            rng.permutation(distance_top_n),
            rng.permutation(distance_lower)
            ])

    total_time = tree_time + integral_filter_time + template_time + random_time
    #print('Time Evaluation')
    #print('Tree', 'integral_filter', 'Template', 'Random')
    #print(
    #    np.round(tree_time / total_time, decimals=3),
    #    np.round(integral_filter_time / total_time, decimals=3),
    #    np.round(template_time / total_time, decimals=3),
    #    np.round(random_time / total_time, decimals=3),
    #    )
    #print('Distance Evaluation')
    #print('Tree', 'integral_filter', 'Template', 'Random')
    #print(
    #    np.round(np.mean(1000*distance[:, :, list(n_sub_generators.keys()).index('trees')]), decimals=3),
    #    np.round(np.mean(1000*distance[:, :, list(n_sub_generators.keys()).index('integral_filter')]), decimals=3),
    #    np.round(np.mean(1000*distance[:, :, list(n_sub_generators.keys()).index('templates')]), decimals=3),
    #    np.round(np.mean(1000*distance[:, :, list(n_sub_generators.keys()).index('predicted_volume')]), decimals=3),
    #    )
    return distance, generator_names


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
            optimizer = get_cubic_optimizer(bravais_lattice, broadening_tag, 1, split_comm)
        elif bravais_lattice in ['hP']:
            optimizer = get_hexagonal_optimizer(bravais_lattice, broadening_tag, 1, split_comm)
        elif bravais_lattice in ['hR']:
            optimizer = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, 1, split_comm)
        elif bravais_lattice in ['tI', 'tP']:
            optimizer = get_tetragonal_optimizer(bravais_lattice, broadening_tag, 1, split_comm)
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, 1, split_comm)
        elif bravais_lattice in ['mC', 'mP']:
            optimizer = get_monoclinic_optimizer(bravais_lattice, broadening_tag, 1, split_comm)
        elif bravais_lattice in ['aP']:
            optimizer = get_triclinic_optimizer(bravais_lattice, broadening_tag, 1, split_comm)
        
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
            distance, generator_names = evaluate_regression(
                optimizer,
                bravais_lattice_data.iloc[trial_index],
                candidates_per_model=candidates_per_model[bravais_lattice],
                rng=rng,
                )
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
        
