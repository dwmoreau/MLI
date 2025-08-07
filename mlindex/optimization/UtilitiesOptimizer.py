from collections import namedtuple
import logging
import numpy as np


def get_logger(comm, optimization_tag):
    from mlindex.utilities.MPIFileHandler import MPIFileHandler
    logger = logging.getLogger(f'rank[{comm.rank}]')
    logger.setLevel(logging.DEBUG)                                                 
    mh = MPIFileHandler(f'logfile_{optimization_tag}.log')
    mh.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
    logger.addHandler(mh)
    return logger


def get_mpi_organizer(comm, bravais_lattices, manager_rank, serial):
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    mpi_organizer = namedtuple('mpi_organizer', ['manager', 'workers', 'color', 'split_comm'])
    mpi_organizers = dict.fromkeys(bravais_lattices)
    serial_split_comm = comm.Split(color=rank, key=0)
    for bl_index, bravais_lattice in enumerate(bravais_lattices):
        if serial[bl_index]:
            if rank == manager_rank[bl_index]:
                mpi_organizers[bravais_lattice] = mpi_organizer(
                    manager_rank[bl_index],
                    [manager_rank[bl_index]],
                    manager_rank[bl_index],
                    serial_split_comm
                    )
            else:
                mpi_organizers[bravais_lattice] = mpi_organizer(
                    manager_rank[bl_index],
                    [manager_rank[bl_index]],
                    rank,
                    None
                    )
        else:
            if rank == manager_rank[bl_index]:
                key = 0
            else:
                key = rank + 1
            mpi_organizers[bravais_lattice] = mpi_organizer(
                manager_rank[bl_index],
                [i for i in range(n_ranks)],
                bl_index,
                comm.Split(color=bl_index, key=key)
                )
    return mpi_organizers


def get_cubic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, fom=None, options=None):
    from mlindex.optimization.Optimizer import OptimizerManager
    data_params = {
        'tag': f'cubic_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        }
    template_params = {bravais_lattice: {'tag': f'cubic_{broadening_tag}'}}
    rf_params = {f'{bravais_lattice}_0': {'tag': f'cubic_{broadening_tag}'}}
    integral_filter_params = {f'{bravais_lattice}_0': {'tag': f'cubic_{broadening_tag}'}}
    random_params = {bravais_lattice: {'tag': f'cubic_{broadening_tag}'}}
    n_candidates = int(n_candidates_scale * 100)
    generator_info = [
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': int(0.4*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': int(0.5*n_candidates)},
        #{'generator': 'templates', 'n_unit_cells': int(0.1*n_candidates)},
        {'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.2*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 5,
        'n_peaks': 10,
        'n_drop': 8,
        'triplet_opt': True,
        'uniform_sampling': False,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 64,
        'neighbor_radius': 0.000026,
        'convergence_testing': False,
        'redistribution_testing': False,
        'downsample_radius': 0.002,
        'assignment_threshold': 0.95,
        'figure_of_merit': 'M20',
        }
    if not options is None:
        for key in options.keys():
            opt_params[key] = options[key]
    optimizer = OptimizerManager(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom
        )
    return optimizer


def get_tetragonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, fom=None, options=None):
    from mlindex.optimization.Optimizer import OptimizerManager
    data_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        }
    template_params = {bravais_lattice: {'tag': f'tetragonal_{broadening_tag}'}}
    rf_group_params = {'tag': f'tetragonal_{broadening_tag}'}
    rf_params = {
        f'{bravais_lattice}_0_00': rf_group_params,
        f'{bravais_lattice}_1_00': rf_group_params,
        f'{bravais_lattice}_0_01': rf_group_params,
        f'{bravais_lattice}_1_01': rf_group_params,
        }
    integral_filter_group_params = {'tag': f'tetragonal_{broadening_tag}'}
    integral_filter_params = {
        f'{bravais_lattice}_0_00': integral_filter_group_params,
        f'{bravais_lattice}_1_00': integral_filter_group_params,
        f'{bravais_lattice}_0_01': integral_filter_group_params,
        f'{bravais_lattice}_1_01': integral_filter_group_params,
        }
    random_params = {bravais_lattice: {'tag': f'tetragonal_{broadening_tag}'}}
    n_candidates = int(n_candidates_scale * 2000)
    generator_info = [
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/4*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/4*0.5*n_candidates)},
        #{'generator': 'templates', 'n_unit_cells': int(0.40*n_candidates)},
        {'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 30,
        'n_peaks': 20,
        'n_drop': 17,
        'uniform_sampling': False,
        'triplet_opt': True,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 52,
        'neighbor_radius': 0.000213,
        'convergence_testing': False,
        'redistribution_testing': False,
        'downsample_radius': 0.0001,
        'assignment_threshold': 0.95,
        'figure_of_merit': 'M20',
        }
    if not options is None:
        for key in options.keys():
            opt_params[key] = options[key]
    optimizer = OptimizerManager(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom
        )
    return optimizer


def get_hexagonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, fom=None, options=None):
    from mlindex.optimization.Optimizer import OptimizerManager
    data_params = {
        'tag': f'hexagonal_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        }
    template_params = {bravais_lattice: {'tag': f'hexagonal_{broadening_tag}'}}
    rf_group_params = {'tag': f'hexagonal_{broadening_tag}'}
    rf_params = {
        f'{bravais_lattice}_0_00': rf_group_params,
        f'{bravais_lattice}_0_01': rf_group_params,
        f'{bravais_lattice}_0_02': rf_group_params,
        f'{bravais_lattice}_0_03': rf_group_params,
        f'{bravais_lattice}_1_00': rf_group_params,
        f'{bravais_lattice}_1_01': rf_group_params,
        f'{bravais_lattice}_1_02': rf_group_params,
        f'{bravais_lattice}_1_03': rf_group_params,
        }
    integral_filter_group_params = {'tag': f'hexagonal_{broadening_tag}'}
    integral_filter_params = {
        f'{bravais_lattice}_0_00': integral_filter_group_params,
        f'{bravais_lattice}_0_01': integral_filter_group_params,
        f'{bravais_lattice}_0_02': integral_filter_group_params,
        f'{bravais_lattice}_0_03': integral_filter_group_params,
        f'{bravais_lattice}_1_00': integral_filter_group_params,
        f'{bravais_lattice}_1_01': integral_filter_group_params,
        f'{bravais_lattice}_1_02': integral_filter_group_params,
        f'{bravais_lattice}_1_03': integral_filter_group_params,
        }
    random_params = {bravais_lattice: {'tag': f'hexagonal_{broadening_tag}'}}
    n_candidates = int(n_candidates_scale * 2000)
    generator_info = [
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/8*0.5*n_candidates)},
        #{'generator': 'templates', 'n_unit_cells': int(0.4*n_candidates)},
        {'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 30,
        'n_peaks': 20,
        'n_drop': 17,
        'uniform_sampling': False,
        'triplet_opt': True,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 52,
        'neighbor_radius': 0.000213,
        'convergence_testing': False,
        'redistribution_testing': False,
        'downsample_radius': 0.0001,
        'assignment_threshold': 0.95,
        'figure_of_merit': 'M20',
        }
    if not options is None:
        for key in options.keys():
            opt_params[key] = options[key]
    optimizer = OptimizerManager(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom
        )
    return optimizer


def get_rhombohedral_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, fom=None, options=None):
    from mlindex.optimization.Optimizer import OptimizerManager
    data_params = {
        'tag': f'rhombohedral_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        }
    template_params = {bravais_lattice: {'tag': f'rhombohedral_{broadening_tag}'}}
    rf_group_params = {'tag': f'rhombohedral_{broadening_tag}'}
    rf_params = {
        f'{bravais_lattice}_00': rf_group_params,
        f'{bravais_lattice}_01': rf_group_params,
        }
    integral_filter_group_params = {'tag': f'rhombohedral_{broadening_tag}', 'quantitized_model': True}
    integral_filter_params = {
        f'{bravais_lattice}_00': integral_filter_group_params,
        f'{bravais_lattice}_01': integral_filter_group_params,
        }
    random_params = {bravais_lattice: {'tag': f'rhombohedral_{broadening_tag}'}}
    n_candidates = int(n_candidates_scale * 2000)
    generator_info = [
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
        #{'generator': 'trees', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': int(1/2*0.05*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(1/2*0.5*n_candidates)},
        #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': int(1/2*0.5*n_candidates)},
        #{'generator': 'templates', 'n_unit_cells': int(0.4*n_candidates)},
        {'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 30,
        'n_peaks': 20,
        'n_drop': 17,
        'uniform_sampling': False,
        'triplet_opt': True,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 52,
        'neighbor_radius': 0.000213,
        'convergence_testing': False,
        'redistribution_testing': False,
        'downsample_radius': 0.0001,
        'assignment_threshold': 0.95,
        'figure_of_merit': 'M20',
        }
    if not options is None:
        for key in options.keys():
            opt_params[key] = options[key]
    optimizer = OptimizerManager(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom
        )
    return optimizer


def get_orthorhombic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, fom=None, options=None):
    from mlindex.optimization.Optimizer import OptimizerManager
    data_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        }
    template_params = {bravais_lattice: {'tag': f'orthorhombic_{broadening_tag}'}}
    rf_group_params = {'tag': f'orthorhombic_{broadening_tag}'}
    integral_filter_group_params = {'tag': f'orthorhombic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'orthorhombic_{broadening_tag}'}}
    n_candidates = int(n_candidates_scale * 4000)
    if bravais_lattice == 'oF':
        rf_params = {
            f'{bravais_lattice}_0_00': rf_group_params,
            f'{bravais_lattice}_0_01': rf_group_params,
            }
        integral_filter_params = {
            f'{bravais_lattice}_0_00': integral_filter_group_params,
            f'{bravais_lattice}_0_01': integral_filter_group_params,
            }
        generator_info = [
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.75*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/2*0.75*n_candidates)},
            #{'generator': 'templates', 'n_unit_cells': int(0.15*n_candidates)},
            {'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
            ]
    elif bravais_lattice == 'oI':
        rf_params = {f'{bravais_lattice}_0_00': rf_group_params,}
        integral_filter_params = {f'{bravais_lattice}_0_00': integral_filter_group_params,}
        generator_info = [
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(0.05*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(0.75*n_candidates)},
            #{'generator': 'templates', 'n_unit_cells': int(0.15*n_candidates)},
            {'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
            ]
    elif bravais_lattice == 'oC':
        rf_params = {
            f'{bravais_lattice}_0_00': rf_group_params,
            f'{bravais_lattice}_2_00': rf_group_params,
            }
        integral_filter_params = {
            f'{bravais_lattice}_0_00': integral_filter_group_params,
            f'{bravais_lattice}_2_00': integral_filter_group_params,
            }
        generator_info = [
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.75*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': int(1/2*0.75*n_candidates)},
            #{'generator': 'templates', 'n_unit_cells': int(0.15*n_candidates)},
            {'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
            ]
    elif bravais_lattice == 'oP':
        rf_params = {
            f'{bravais_lattice}_0_00': rf_group_params,
            f'{bravais_lattice}_0_01': rf_group_params,
            f'{bravais_lattice}_0_02': rf_group_params,
            f'{bravais_lattice}_0_03': rf_group_params,
            }
        integral_filter_params = {
            f'{bravais_lattice}_0_00': integral_filter_group_params,
            f'{bravais_lattice}_0_01': integral_filter_group_params,
            f'{bravais_lattice}_0_02': integral_filter_group_params,
            f'{bravais_lattice}_0_03': integral_filter_group_params,
            }
        generator_info = [
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            #{'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.75*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.75*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/4*0.75*n_candidates)},
            #{'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/4*0.75*n_candidates)},
            #{'generator': 'templates', 'n_unit_cells': int(0.15*n_candidates)},
            {'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
            ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 50,
        'n_peaks': 20,
        'n_drop': 14,
        'uniform_sampling': False,
        'triplet_opt': True,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 46,
        'neighbor_radius': 0.000338,
        'convergence_testing': False,
        'redistribution_testing': False,
        'downsample_radius': 0.0001,
        'assignment_threshold': 0.95,
        'figure_of_merit': 'M20',
        }
    if not options is None:
        for key in options.keys():
            opt_params[key] = options[key]
    optimizer = OptimizerManager(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom
        )
    return optimizer


def get_monoclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, fom=None, options=None):
    from mlindex.optimization.Optimizer import OptimizerManager
    data_params = {
        'tag': f'monoclinic_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        }
    template_params = {bravais_lattice: {'tag': f'monoclinic_{broadening_tag}'}}
    rf_group_params = {'tag': f'monoclinic_{broadening_tag}'}
    integral_filter_group_params = {'tag': f'monoclinic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'monoclinic_{broadening_tag}'}}
    n_candidates = int(n_candidates_scale * 6000)
    if bravais_lattice == 'mC':
        rf_params = {
            f'{bravais_lattice}_0_02': rf_group_params,
            f'{bravais_lattice}_0_03': rf_group_params,
            f'{bravais_lattice}_1_02': rf_group_params,
            f'{bravais_lattice}_1_03': rf_group_params,
            f'{bravais_lattice}_4_02': rf_group_params,
            f'{bravais_lattice}_4_03': rf_group_params,
            }
        integral_filter_params = {
            f'{bravais_lattice}_0_02': integral_filter_group_params,
            f'{bravais_lattice}_0_03': integral_filter_group_params,
            f'{bravais_lattice}_1_02': integral_filter_group_params,
            f'{bravais_lattice}_1_03': integral_filter_group_params,
            f'{bravais_lattice}_4_02': integral_filter_group_params,
            f'{bravais_lattice}_4_03': integral_filter_group_params,
            }
        generator_info = [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.2*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            {'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
            ]
    elif bravais_lattice == 'mP':
        rf_params = {
            f'{bravais_lattice}_0_00': rf_group_params,
            f'{bravais_lattice}_0_01': rf_group_params,
            f'{bravais_lattice}_1_00': rf_group_params,
            f'{bravais_lattice}_1_01': rf_group_params,
            f'{bravais_lattice}_4_00': rf_group_params,
            f'{bravais_lattice}_4_01': rf_group_params,
            }
        integral_filter_params = {
            f'{bravais_lattice}_0_00': integral_filter_group_params,
            f'{bravais_lattice}_0_01': integral_filter_group_params,
            f'{bravais_lattice}_1_00': integral_filter_group_params,
            f'{bravais_lattice}_1_01': integral_filter_group_params,
            f'{bravais_lattice}_4_00': integral_filter_group_params,
            f'{bravais_lattice}_4_01': integral_filter_group_params,
            }
        generator_info = [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(1/6*0.1*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(1/6*0.65*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.2*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            {'generator': 'predicted_volume', 'n_unit_cells': int(0.05*n_candidates)},
            ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'n_peaks': 20,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 60,
        'n_peaks': 20,
        'n_drop': 14,
        'uniform_sampling': False,
        'triplet_opt': True,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 42,
        'neighbor_radius': 0.000547,
        'convergence_testing': False,
        'redistribution_testing': False,
        'downsample_radius': 0.0001,
        'assignment_threshold': 0.95,
        'figure_of_merit': 'M20',
        }
    if not options is None:
        for key in options.keys():
            opt_params[key] = options[key]
    optimizer = OptimizerManager(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom
        )
    return optimizer


def get_triclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, fom=None, options=None):
    from mlindex.optimization.Optimizer import OptimizerManager
    data_params = {
        'tag': f'triclinic_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        }
    template_params = {bravais_lattice: {'tag': f'triclinic_{broadening_tag}'}}
    rf_group_params = {'tag': f'triclinic_{broadening_tag}'}
    integral_filter_group_params = {'tag': f'triclinic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'triclinic_{broadening_tag}'}}
    rf_params = {
        f'{bravais_lattice}_00': rf_group_params,
        }
    integral_filter_params = {
        f'{bravais_lattice}_00': integral_filter_group_params,
        }
    n_candidates = int(n_candidates_scale * 6000)
    generator_info = [
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(0.1 * n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(0.4 * n_candidates)},
        {'generator': 'templates', 'n_unit_cells': int(0.45 * n_candidates)},
        #{'generator': 'random', 'n_unit_cells': n_candidates},
        {'generator': 'predicted_volume', 'n_unit_cells': int(0.05 * n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 60,
        'n_peaks': 20,
        'n_drop': 12,
        'uniform_sampling': False,
        'triplet_opt': True,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'max_neighbors': 23,
        'neighbor_radius': 0.000679,
        'convergence_testing': False,
        'redistribution_testing': False,
        'downsample_radius': 0.0001,
        'assignment_threshold': 0.95,
        'figure_of_merit': 'M20',
        }
    if not options is None:
        for key in options.keys():
            opt_params[key] = options[key]
    optimizer = OptimizerManager(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom
        )
    return optimizer


def get_optimizers(rank, mpi_organizers, broadening_tag, n_candidates_scale, logger=None):
    from mlindex.optimization.Optimizer import OptimizerWorker
    """
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'n_peaks': 20,
        'triplet_opt': True,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 100,
        'uniform_sampling': False,
        'triplet_opt': True,
        }
        ]
    opt_params = {
        'generator_info': generator_info,
        'iteration_info': iteration_info,
        'convergence_testing': False,
        'assignment_threshold': 0.95,
        }
    """
    fom = None
    bravais_lattices = mpi_organizers.keys()
    optimizer = dict.fromkeys(bravais_lattices)
    for bl_index, bravais_lattice in enumerate(bravais_lattices):
        if rank == mpi_organizers[bravais_lattice].manager:
            # These function calls return an OptimizerManager object
            if bravais_lattice in ['cF', 'cI', 'cP']:
                optimizer[bravais_lattice] = get_cubic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    fom,
                    )
            elif bravais_lattice in ['hP']:
                optimizer[bravais_lattice] = get_hexagonal_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    fom,
                    )
            elif bravais_lattice in ['hR']:
                optimizer[bravais_lattice] = get_rhombohedral_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    fom,
                    )
            elif bravais_lattice in ['tI', 'tP']:
                optimizer[bravais_lattice] = get_tetragonal_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    fom,
                    )
            elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
                optimizer[bravais_lattice] = get_orthorhombic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    fom,
                    )
            elif bravais_lattice in ['mC', 'mP']:
                optimizer[bravais_lattice] = get_monoclinic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    fom,
                    )
            elif bravais_lattice in ['aP']:
                optimizer[bravais_lattice] = get_triclinic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    fom,
                    )
            if not logger is None:
                logger.info(f'Loaded manager optimizer for {bravais_lattice}')
        elif rank in mpi_organizers[bravais_lattice].workers:
            optimizer[bravais_lattice] = OptimizerWorker(mpi_organizers[bravais_lattice].split_comm, fom)
            if not logger is None:
                logger.info(f'Loaded worker optimizer for {bravais_lattice}')
    return optimizer
