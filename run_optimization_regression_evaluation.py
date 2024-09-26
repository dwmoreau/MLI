import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from tqdm import tqdm

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
from Utilities import fix_unphysical
from Utilities import get_xnn_from_unit_cell
from Reindexing import reindex_entry_triclinic


def evaluate_regression(optimizer, data, candidates_per_model, N_entries, n_candidates_steps, n_evaluations, threshold):
    candidate_steps = np.round(
        np.linspace(10, candidates_per_model, n_candidates_steps),
        decimals=0
        ).astype(int)

    n_generators = len(optimizer.opt_params['generator_info'])
    efficiency = np.zeros((N_entries, n_candidates_steps, n_generators))
    failure_rate = np.zeros((N_entries, n_candidates_steps, n_generators))
    efficiency_nn = np.zeros((N_entries, n_candidates_steps))
    efficiency_rf = np.zeros((N_entries, n_candidates_steps))
    efficiency_all_models = np.zeros((N_entries, n_candidates_steps))
    #efficiency_pitf = np.zeros((N_entries, n_candidates_steps))
    failure_rate_nn = np.zeros((N_entries, n_candidates_steps))
    failure_rate_rf = np.zeros((N_entries, n_candidates_steps))
    failure_rate_all_models = np.zeros((N_entries, n_candidates_steps))
    #failure_rate_pitf = np.zeros((N_entries, n_candidates_steps))
    rng = np.random.default_rng()
    generator_labels = [None for i in range(n_generators)]
    for generator_index, generator_info in enumerate(optimizer.opt_params['generator_info']):
        if generator_info['generator'] in ['nn', 'trees', 'pitf']:
            generator_labels[generator_index] = f'{generator_info["generator"]} {generator_info["split_group"]}'
        elif generator_info['generator'] in ['random', 'distribution_volume', 'predicted_volume', 'templates']:
            generator_labels[generator_index] = f'{generator_info["generator"]} {optimizer.bravais_lattice}'

    for entry_index in tqdm(range(N_entries)):
        entry = data.iloc[entry_index]
        xnn_true = np.array(entry['reindexed_xnn'])[optimizer.indexer.data_params['unit_cell_indices']]
        q2 = np.array(entry['q2'])[:optimizer.n_peaks]
        distance_all_nn = []
        distance_all_rf = []
        distance_all_models = []
        #distance_all_pitf = []
        n_gen_nn = 0
        n_gen_rf = 0
        n_gen_all_models = 0
        #n_gen_pitf = 0
        for generator_index, generator_info in enumerate(optimizer.opt_params['generator_info']):
            if generator_info['generator'] in ['nn', 'trees']:
                generator_unit_cells = optimizer.indexer.unit_cell_generator[generator_info['split_group']].generate(
                    candidates_per_model, rng,  q2,
                    batch_size=1,
                    model=generator_info['generator'],
                    q2_scaler=optimizer.indexer.q2_scaler,
                    )
            elif generator_info['generator'] == 'pitf':
                generator_unit_cells = optimizer.indexer.pitf_generator[generator_info['split_group']].generate(
                    candidates_per_model, rng, q2,
                    batch_size=1,
                    )
            elif generator_info['generator'] == 'templates':
                generator_unit_cells = optimizer.indexer.miller_index_templator[optimizer.bravais_lattice].generate(
                    candidates_per_model, rng, q2, 
                    )
            elif generator_info['generator'] in ['random', 'distribution_volume', 'predicted_volume']:
                generator_unit_cells = optimizer.indexer.random_unit_cell_generator[optimizer.bravais_lattice].generate(
                    candidates_per_model, rng, q2,
                    model=generator_info['generator'],
                    )

            generator_unit_cells = fix_unphysical(
                unit_cell=generator_unit_cells,
                rng=rng,
                minimum_unit_cell=optimizer.opt_params['minimum_uc'],
                maximum_unit_cell=optimizer.opt_params['maximum_uc'],
                lattice_system=optimizer.indexer.data_params['lattice_system']
                )
            if optimizer.indexer.data_params['lattice_system'] == 'triclinic':
                generator_unit_cells, _ = reindex_entry_triclinic(generator_unit_cells)
            generator_xnn = get_xnn_from_unit_cell(
                generator_unit_cells,
                partial_unit_cell=True,
                lattice_system=optimizer.indexer.data_params['lattice_system']
                )

            distance_all = np.linalg.norm(generator_xnn - xnn_true[np.newaxis], axis=1)
            if generator_info['generator'] == 'nn':
                distance_all_nn.append(distance_all)
                n_gen_nn += 1
            elif generator_info['generator'] == 'trees':
                distance_all_rf.append(distance_all)
                n_gen_rf += 1
            elif generator_info['generator'] == 'pitf':
                distance_all_pitf.append(distance_all)
                n_gen_pitf += 1
            distance_all_models.append(distance_all)
            n_gen_all_models += 1

            for step_index, step_size in enumerate(candidate_steps):
                efficiency_step = np.zeros(n_evaluations)
                for eval_index in range(n_evaluations):
                    indices = rng.choice(candidates_per_model, step_size, replace=False)
                    distance = distance_all[indices]
                    efficiency_step[eval_index] = np.sum(distance < threshold) / step_size
                efficiency[entry_index, step_index, generator_index] = efficiency_step.mean()
                failure_rate[entry_index, step_index, generator_index] = np.sum(efficiency_step == 0, axis=0) / n_evaluations

        distance_all_nn = np.concatenate(distance_all_nn)
        distance_all_rf = np.concatenate(distance_all_rf)
        distance_all_models = np.concatenate(distance_all_models)
        #distance_all_pitf = np.concatenate(distance_all_pitf)
        for step_index, step_size in enumerate(candidate_steps):
            efficiency_step_nn = np.zeros(n_evaluations)
            efficiency_step_rf = np.zeros(n_evaluations)
            efficiency_step_all_models = np.zeros(n_evaluations)
            #efficiency_step_pitf = np.zeros(n_evaluations)
            for eval_index in range(n_evaluations):
                indices_nn = rng.choice(distance_all_nn.size, step_size, replace=False)
                indices_rf = rng.choice(distance_all_rf.size, step_size, replace=False)
                indices_all_models = rng.choice(distance_all_models.size, step_size, replace=False)
                #indices_pitf = rng.choice(distance_all_pitf.size, step_size, replace=False)
                distance_nn = distance_all_nn[indices_nn]
                distance_rf = distance_all_rf[indices_rf]
                distance_models = distance_all_models[indices_all_models]
                #distance_pitf = distance_all_pitf[indices_pitf]
                efficiency_step_nn[eval_index] = np.sum(distance_nn < threshold) / step_size
                efficiency_step_rf[eval_index] = np.sum(distance_rf < threshold) / step_size
                efficiency_step_all_models[eval_index] = np.sum(distance_models < threshold) / step_size
                #efficiency_step_pitf[eval_index] = np.sum(distance_pitf < threshold) / step_size
            efficiency_nn[entry_index, step_index] = efficiency_step_nn.mean()
            efficiency_rf[entry_index, step_index] = efficiency_step_rf.mean()
            efficiency_all_models[entry_index, step_index] = efficiency_step_all_models.mean()
            #efficiency_pitf[entry_index, step_index] = efficiency_step_pitf.mean()
            failure_rate_nn[entry_index, step_index] = np.sum(efficiency_step_nn == 0, axis=0) / n_evaluations
            failure_rate_rf[entry_index, step_index] = np.sum(efficiency_step_rf == 0, axis=0) / n_evaluations
            failure_rate_all_models[entry_index, step_index] = np.sum(efficiency_step_all_models == 0, axis=0) / n_evaluations
            #failure_rate_pitf[entry_index, step_index] = np.sum(efficiency_step_pitf == 0, axis=0) / n_evaluations

    color_cycle_indices = [0, 1, 2, 3, 9, 8, 5, 6, 7, 4]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex='row')
    for generator_index, generator_info in enumerate(optimizer.opt_params['generator_info']):
        color_index = generator_index % len(color_cycle_indices)
        if generator_info['generator'] == 'trees':
            linestyle = 'dotted'
        elif generator_info['generator'] == 'nn':
            linestyle = 'dashed'
        elif generator_info['generator'] == 'random':
            linestyle = 'solid'
        elif generator_info['generator'] == 'predicted_volume':
            linestyle = 'dashdot'
        elif generator_info['generator'] == 'templates':
            linestyle = (0, (5, 1))
        elif generator_info['generator'] == 'pitf':
            linestyle = (0, (3, 1, 1, 1))
        else:
            print(generator_info['generator'])
        axes[0, 0].plot(
            candidate_steps, 100 * efficiency[:, :, generator_index].mean(axis=0),
            label=generator_labels[generator_index], color=colors[color_cycle_indices[color_index]], linestyle=linestyle
            )
        axes[0, 1].plot(
            candidate_steps, 100 * failure_rate[:, :, generator_index].mean(axis=0),
            label=generator_labels[generator_index], color=colors[color_cycle_indices[color_index]], linestyle=linestyle
            )
        if generator_info['generator'] in ['predicted_volume', 'templates']:
            axes[1, 0].plot(
                candidate_steps, 100 * efficiency[:, :, generator_index].mean(axis=0),
                label=generator_labels[generator_index], color=colors[color_cycle_indices[color_index]], linestyle=linestyle
                )
            axes[1, 1].plot(
                candidate_steps, 100 * failure_rate[:, :, generator_index].mean(axis=0),
                label=generator_labels[generator_index], color=colors[color_cycle_indices[color_index]], linestyle=linestyle
                )
    axes[1, 0].plot(
        candidate_steps, 100 * efficiency_nn.mean(axis=0),
        label='All NN', color=colors[0], linestyle='dashed'
        )
    axes[1, 1].plot(
        candidate_steps, 100 * failure_rate_nn.mean(axis=0),
        label='All NN', color=colors[0], linestyle='dashed'
        )
    axes[1, 0].plot(
        candidate_steps, 100 * efficiency_rf.mean(axis=0),
        label='All RF', color=colors[1], linestyle='dotted'
        )
    axes[1, 1].plot(
        candidate_steps, 100 * failure_rate_rf.mean(axis=0),
        label='All RF', color=colors[1], linestyle='dotted'
        )
    #axes[1, 1].plot(
    #    candidate_steps, 100 * failure_rate_all_models.mean(axis=0),
    #    label='All Models', color=[0, 0, 0]
    #    )
    #axes[1, 0].plot(
    #    n_gen_pitf * candidate_steps, 100 * efficiency_pitf.mean(axis=0),
    #    label='All PITF', color=colors[2], linestyle=(0, (3, 1, 1, 1))
    #    )
    #axes[1, 1].plot(
    #    n_gen_pitf * candidate_steps, 100 * failure_rate_pitf.mean(axis=0),
    #    label='All PITF', color=colors[2], linestyle=(0, (3, 1, 1, 1))
    #    )

    fs = 16
    axes[0, 0].set_title(optimizer.lattice_system, fontsize=fs)
    axes[0, 1].set_title(optimizer.bravais_lattice, fontsize=fs)
    axes[1, 0].set_xlabel('Number of candidates', fontsize=fs)
    axes[1, 1].set_xlabel('Number of candidates', fontsize=fs)
    for row in range(2):
        axes[row, 0].set_ylabel('Efficiency (%)', fontsize=fs)
        axes[row, 1].set_ylabel('Failure Rate (%)', fontsize=fs)
        for col in range(2):
            axes[row, col].tick_params(axis='both', which='major', labelsize=fs)
    axes[0, 0].legend(frameon=False, ncols=2)
    axes[1, 1].legend(frameon=False, fontsize=fs-4)
    fig.tight_layout()
    fig.savefig(f'figures/{optimizer.bravais_lattice}_EfficiencyFailures.png')
    plt.show()


if __name__ == '__main__':
    load_data = True
    broadening_tag = '0.5'
    error_tag = '0.1'
    q2_error_params = np.array([0.0001, 0.001]) / 1
    n_top_candidates = 20
    #bravais_lattices = ['cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP']
    
    #bravais_lattices = ['cF', 'cI', 'cP']
    #candidates_per_model = 200
    #N_entries = 100
    #mult_factor = 1
    
    #bravais_lattices = ['tI', 'tP']
    #candidates_per_model = 500
    #N_entries = 100
    #mult_factor = 5

    #bravais_lattices = ['hP', 'hR']
    #candidates_per_model = 500
    #N_entries = 100
    #mult_factor = 5

    #bravais_lattices = ['oC', 'oF', 'oI', 'oP']
    #candidates_per_model = 500
    #N_entries = 20
    #mult_factor = 5

    #bravais_lattices = ['mC', 'mP']
    #candidates_per_model = 1000
    #N_entries = 20
    #mult_factor = 10

    bravais_lattices = ['aP']
    candidates_per_model = 800
    N_entries = 100
    mult_factor = 17

    optimizer = dict.fromkeys(bravais_lattices)
    rng = np.random.default_rng()
    for bravais_lattice in bravais_lattices:
        print(f'Loading optimizer for {bravais_lattice}')
        if bravais_lattice in ['cF', 'cI', 'cP']:
            optimizer[bravais_lattice] = get_cubic_optimizer(bravais_lattice, broadening_tag, error_tag, MPI.COMM_WORLD)
        elif bravais_lattice in ['hP']:
            optimizer[bravais_lattice] = get_hexagonal_optimizer(bravais_lattice, broadening_tag, error_tag, MPI.COMM_WORLD)
        elif bravais_lattice in ['hR']:
            optimizer[bravais_lattice] = get_rhombohedral_optimizer(bravais_lattice, broadening_tag, error_tag, MPI.COMM_WORLD)
        elif bravais_lattice in ['tI', 'tP']:
            optimizer[bravais_lattice] = get_tetragonal_optimizer(bravais_lattice, broadening_tag, error_tag, MPI.COMM_WORLD)
        elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            optimizer[bravais_lattice] = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, error_tag, MPI.COMM_WORLD)
        elif bravais_lattice in ['mC', 'mP']:
            optimizer[bravais_lattice] = get_monoclinic_optimizer(bravais_lattice, broadening_tag, error_tag, MPI.COMM_WORLD)
        elif bravais_lattice in ['aP']:
            optimizer[bravais_lattice] = get_triclinic_optimizer(bravais_lattice, broadening_tag, error_tag, MPI.COMM_WORLD)

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
    
    for bravais_lattice in bravais_lattices:
        evaluate_regression(
            optimizer[bravais_lattice],
            data[data['bravais_lattice'] == bravais_lattice],
            candidates_per_model=candidates_per_model,
            N_entries=N_entries,
            n_candidates_steps=20,
            n_evaluations=10,
            threshold=mult_factor*optimizer[bravais_lattice].opt_params['neighbor_radius']
            )
