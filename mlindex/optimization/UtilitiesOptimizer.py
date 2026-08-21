from collections import namedtuple
import logging
import numpy as np
import os
from pathlib import Path
import warnings

from mlindex import paths


def _env_models_dir_error(env_dir):
    """Build the error message for an MLINDEX_MODELS_DIR that isn't a models directory.

    Users very often point the variable at the parent of the models directory, so
    check the two likely candidates and name the right one for them.
    """
    message = (
        f"{paths.ENV_VAR}={env_dir} does not look like a models directory.\n"
        "It must be the directory that directly contains the model subdirectories "
        "(cubic_1/, hexagonal_1/, ...)."
    )
    for candidate in (env_dir / 'mlindex' / 'models', env_dir / 'models'):
        if paths.looks_like_models_dir(candidate):
            message += f"\nDid you mean {candidate}?"
            break
    else:
        message += "\nRun 'mlindex.download_models' to fetch the models."
    return message


def _resolve_models_dir():
    """Return the directory that directly contains cubic_1/, hexagonal_1/, ...

    Resolution order:
    1. MLINDEX_MODELS_DIR env var, used as-is.
    2. XDG data home: ~/.local/share/mlindex/models
    3. Package directory fallback (editable installs / legacy repo checkouts).
    """
    env_dir = paths.models_dir_from_env()
    if env_dir is not None:
        if not env_dir.exists():
            raise FileNotFoundError(
                f"{paths.ENV_VAR}={env_dir} does not exist. "
                "Run 'mlindex.download_models' or set it to the directory "
                "containing model subdirectories (e.g. cubic_1/, hexagonal_1/, ...)."
            )
        if not paths.looks_like_models_dir(env_dir):
            raise FileNotFoundError(_env_models_dir_error(env_dir))
        return env_dir

    xdg_models = paths.default_models_dir()
    if paths.looks_like_models_dir(xdg_models):
        return xdg_models

    import mlindex
    pkg_models = Path(mlindex.__path__[0]) / 'models'
    # Check for a directory that only exists after mlindex.download_models (not just bundled hkl_ref)
    if not paths.looks_like_models_dir(pkg_models):
        raise FileNotFoundError(
            "ML models not found. Run 'mlindex.download_models' to fetch them.\n"
            f"Searched:\n  {xdg_models}\n  {pkg_models}\n"
            f"Or set the {paths.ENV_VAR} environment variable to the directory "
            "containing model subdirectories (e.g. cubic_1/, hexagonal_1/, ...)."
        )
    return pkg_models


def _resolve_project_path():
    """Deprecated: use _resolve_models_dir().

    Returns a base directory such that base_dir/mlindex/models/{tag} points at the
    models. That round-trip only works when the models live in a directory ending in
    'mlindex/models', which is why it is deprecated. Removed in 0.2.0.
    """
    warnings.warn(
        "_resolve_project_path() is deprecated and will be removed in 0.2.0; "
        "use _resolve_models_dir(), which returns the models directory itself.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _resolve_models_dir().parent.parent


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


def get_cubic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, project_path=None, fom=None, options=None, optimizer_class=None, seed=12345, models_directory=None):
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    _cls = optimizer_class or OptimizerManager
    data_params = {
        'tag': f'cubic_{broadening_tag}',
        'base_directory': project_path,
        'models_directory': models_directory,
        }
    template_params = {bravais_lattice: {'tag': f'cubic_{broadening_tag}'}}
    rf_params = {f'{bravais_lattice}_0': {'tag': f'cubic_{broadening_tag}'}}
    integral_filter_params = {f'{bravais_lattice}_0': {'tag': f'cubic_{broadening_tag}'}}
    random_params = {bravais_lattice: {'tag': f'cubic_{broadening_tag}'}}
    n_candidates = int(n_candidates_scale * 100)
    generator_info = [
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': int(0.45*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': int(0.45*n_candidates)},
        {'generator': 'templates', 'n_unit_cells': int(0.1*n_candidates)},
        #{'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 5,
        'n_peaks': 10,
        'n_drop': 8,
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
    optimizer = _cls(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_tetragonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, project_path=None, fom=None, options=None, optimizer_class=None, seed=12345, models_directory=None):
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    _cls = optimizer_class or OptimizerManager
    data_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'base_directory': project_path,
        'models_directory': models_directory,
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
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/4*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/4*0.7*n_candidates)},
        {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
        #{'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 30,
        'n_peaks': 20,
        'n_drop': 17,
        'uniform_sampling': False,
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
    optimizer = _cls(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_hexagonal_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, project_path=None, fom=None, options=None, optimizer_class=None, seed=12345, models_directory=None):
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    _cls = optimizer_class or OptimizerManager
    data_params = {
        'tag': f'hexagonal_{broadening_tag}',
        'base_directory': project_path,
        'models_directory': models_directory,
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
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/8*0.05*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/8*0.7*n_candidates)},
        {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
        #{'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 30,
        'n_peaks': 20,
        'n_drop': 17,
        'uniform_sampling': False,
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
    optimizer = _cls(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_rhombohedral_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, project_path=None, fom=None, options=None, optimizer_class=None, seed=12345, models_directory=None):
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    _cls = optimizer_class or OptimizerManager
    data_params = {
        'tag': f'rhombohedral_{broadening_tag}',
        'base_directory': project_path,
        'models_directory': models_directory,
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
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': int(1/2*0.05*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': int(1/2*0.7*n_candidates)},
        {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
        #{'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 30,
        'n_peaks': 20,
        'n_drop': 17,
        'uniform_sampling': False,
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
    optimizer = _cls(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_orthorhombic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, project_path=None, fom=None, options=None, optimizer_class=None, seed=12345, models_directory=None):
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    _cls = optimizer_class or OptimizerManager
    data_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'base_directory': project_path,
        'models_directory': models_directory,
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
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
            ]
    elif bravais_lattice == 'oI':
        rf_params = {f'{bravais_lattice}_0_00': rf_group_params,}
        integral_filter_params = {f'{bravais_lattice}_0_00': integral_filter_group_params,}
        generator_info = [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(0.05*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
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
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
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
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
            ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 50,
        'n_peaks': 20,
        'n_drop': 14,
        'uniform_sampling': False,
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
    optimizer = _cls(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_monoclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, project_path=None, fom=None, options=None, optimizer_class=None, seed=12345, models_directory=None):
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    _cls = optimizer_class or OptimizerManager
    data_params = {
        'tag': f'monoclinic_{broadening_tag}',
        'base_directory': project_path,
        'models_directory': models_directory,
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
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.4*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
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
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.4*n_candidates)},
            #{'generator': 'random', 'n_unit_cells': n_candidates},
            #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01*n_candidates)},
            ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        'n_peaks': 20,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 60,
        'n_peaks': 20,
        'n_drop': 14,
        'uniform_sampling': False,
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
    optimizer = _cls(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_triclinic_optimizer(bravais_lattice, broadening_tag, n_candidates_scale, comm, project_path=None, fom=None, options=None, optimizer_class=None, seed=12345, models_directory=None):
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    _cls = optimizer_class or OptimizerManager
    data_params = {
        'tag': f'triclinic_{broadening_tag}',
        'base_directory': project_path,
        'models_directory': models_directory,
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
        {'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(0.05 * n_candidates)},
        {'generator': 'integral_filter', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(0.4 * n_candidates)},
        {'generator': 'templates', 'n_unit_cells': int(0.55 * n_candidates)},
        #{'generator': 'random', 'n_unit_cells': n_candidates},
        #{'generator': 'predicted_volume', 'n_unit_cells': int(0.01 * n_candidates)},
        ]
    iteration_info = [
        {
        'worker': 'deterministic',
        'n_iterations': 1,
        },
        {
        'worker': 'random_subsampling',
        'n_iterations': 60,
        'n_peaks': 20,
        'n_drop': 12,
        'uniform_sampling': False,
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
    optimizer = _cls(
        data_params,
        opt_params,
        rf_params,
        template_params,
        integral_filter_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_optimizers(rank, mpi_organizers, broadening_tag, n_candidates_scale, logger=None, optimizer_class=None, seed=12345, options=None):
    """Build one optimizer per Bravais lattice.

    ``options`` overrides entries of ``opt_params`` on every lattice, through the
    ``options=`` argument the per-system factories already take. It has to be supplied
    here rather than assigned afterwards: ``opt_params`` is broadcast to the workers at
    construction (``OptimizerBase.__init__``, ``MPOptimizerManager._init_workers``), so a
    key read inside ``Candidates`` -- which runs on every rank -- would otherwise apply to
    the manager's share of the candidates and not the workers'. ``dump_candidates`` gets
    away with being set afterwards only because it is read on the manager alone.
    """
    from mlindex.optimization.MPIOptimizer import OptimizerWorker

    models_dir = _resolve_models_dir()
    # Legacy base_directory, kept populated for callers that still read it. It is inert
    # on the inference path, where models_directory determines the model location.
    project_path = models_dir.parent.parent

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
                    project_path,
                    fom,
                    options=options,
                    optimizer_class=optimizer_class,
                    seed=seed,
                    models_directory=models_dir,
                    )
            elif bravais_lattice in ['hP']:
                optimizer[bravais_lattice] = get_hexagonal_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    project_path,
                    fom,
                    options=options,
                    optimizer_class=optimizer_class,
                    seed=seed,
                    models_directory=models_dir,
                    )
            elif bravais_lattice in ['hR']:
                optimizer[bravais_lattice] = get_rhombohedral_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    project_path,
                    fom,
                    options=options,
                    optimizer_class=optimizer_class,
                    seed=seed,
                    models_directory=models_dir,
                    )
            elif bravais_lattice in ['tI', 'tP']:
                optimizer[bravais_lattice] = get_tetragonal_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    project_path,
                    fom,
                    options=options,
                    optimizer_class=optimizer_class,
                    seed=seed,
                    models_directory=models_dir,
                    )
            elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
                optimizer[bravais_lattice] = get_orthorhombic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    project_path,
                    fom,
                    options=options,
                    optimizer_class=optimizer_class,
                    seed=seed,
                    models_directory=models_dir,
                    )
            elif bravais_lattice in ['mC', 'mP']:
                optimizer[bravais_lattice] = get_monoclinic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    project_path,
                    fom,
                    options=options,
                    optimizer_class=optimizer_class,
                    seed=seed,
                    models_directory=models_dir,
                    )
            elif bravais_lattice in ['aP']:
                optimizer[bravais_lattice] = get_triclinic_optimizer(
                    bravais_lattice,
                    broadening_tag,
                    n_candidates_scale,
                    mpi_organizers[bravais_lattice].split_comm,
                    project_path,
                    fom,
                    options=options,
                    optimizer_class=optimizer_class,
                    seed=seed,
                    models_directory=models_dir,
                    )
            if not logger is None:
                logger.info(f'Loaded manager optimizer for {bravais_lattice}')
        elif rank in mpi_organizers[bravais_lattice].workers:
            optimizer[bravais_lattice] = OptimizerWorker(mpi_organizers[bravais_lattice].split_comm, fom, seed=seed + rank)
            if not logger is None:
                logger.info(f'Loaded worker optimizer for {bravais_lattice}')
    return optimizer
