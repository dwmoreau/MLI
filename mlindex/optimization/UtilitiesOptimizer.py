from collections import namedtuple
import logging
import numpy as np
import os
from pathlib import Path
import warnings

from mlindex import paths


# The smallest and largest cell edge, in Angstrom, a candidate may have; fix_unphysical enforces
# them wherever a cell is generated or moved.
MINIMUM_UNIT_CELL = 2
MAXIMUM_UNIT_CELL = 500

# Each Bravais lattice's candidate settings, and the only place they are set:
#   n_candidates     the budget at n_candidates_scale = 1
#   fractions        the share of it each generator makes: P09b's fit on a grid of contaminants
#                    and dropped peaks (per-pattern reduction, rounded to 0.01). OptimizerManager
#                    divides each share
#                    among the split groups with Allocation.generator_info_from_fractions, and a row
#                    whose fractions do not sum to one is refused there
#   max_neighbors,   Redistribution.redistribute_xnn caps every candidate's neighbourhood within
#   neighbor_radius  neighbor_radius at max_neighbors. Re-derived in P09c with the same score as
#                    the fractions (run_ensemble_refine --stage redistribution, per-pattern
#                    reduction); cubic keeps its earlier pair, since no clump discount can be
#                    measured there and the score cannot see redistribution
ENSEMBLE = {
    'cF': {'n_candidates': 100, 'fractions': {'trees': 0.80, 'abnn': 0.09, 'templates': 0.11},
           'max_neighbors': 64, 'neighbor_radius': 0.000026},
    'cI': {'n_candidates': 100, 'fractions': {'trees': 0.75, 'abnn': 0.09, 'templates': 0.16},
           'max_neighbors': 64, 'neighbor_radius': 0.000026},
    'cP': {'n_candidates': 100, 'fractions': {'trees': 0.67, 'abnn': 0.10, 'templates': 0.23},
           'max_neighbors': 64, 'neighbor_radius': 0.000026},
    'hP': {'n_candidates': 2000, 'fractions': {'trees': 0.44, 'abnn': 0.51, 'templates': 0.05},
           'max_neighbors': 64, 'neighbor_radius': 0.00122},
    'hR': {'n_candidates': 2000, 'fractions': {'trees': 0.57, 'abnn': 0.17, 'templates': 0.26},
           'max_neighbors': 46, 'neighbor_radius': 0.000609},
    'tI': {'n_candidates': 2000, 'fractions': {'trees': 0.66, 'abnn': 0.29, 'templates': 0.05},
           'max_neighbors': 66, 'neighbor_radius': 0.000753},
    'tP': {'n_candidates': 2000, 'fractions': {'trees': 0.56, 'abnn': 0.10, 'templates': 0.34},
           'max_neighbors': 45, 'neighbor_radius': 0.000421},
    'oC': {'n_candidates': 4000, 'fractions': {'trees': 0.52, 'abnn': 0.03, 'templates': 0.45},
           'max_neighbors': 50, 'neighbor_radius': 0.000689},
    'oF': {'n_candidates': 4000, 'fractions': {'trees': 0.34, 'abnn': 0.03, 'templates': 0.63},
           'max_neighbors': 68, 'neighbor_radius': 0.000547},
    'oI': {'n_candidates': 4000, 'fractions': {'trees': 0.46, 'abnn': 0.04, 'templates': 0.50},
           'max_neighbors': 65, 'neighbor_radius': 0.000582},
    'oP': {'n_candidates': 4000, 'fractions': {'trees': 0.71, 'abnn': 0.08, 'templates': 0.21},
           'max_neighbors': 58, 'neighbor_radius': 0.000766},
    'mC': {'n_candidates': 6000, 'fractions': {'trees': 0.29, 'abnn': 0.08, 'templates': 0.63},
           'max_neighbors': 44, 'neighbor_radius': 0.000892},
    'mP': {'n_candidates': 6000, 'fractions': {'trees': 0.45, 'abnn': 0.06, 'templates': 0.49},
           'max_neighbors': 46, 'neighbor_radius': 0.00126},
    'aP': {'n_candidates': 6000, 'fractions': {'trees': 0.20, 'abnn': 0.21, 'templates': 0.59},
           'max_neighbors': 22, 'neighbor_radius': 0.00132},
    }


def lattice_fractions(bravais_lattice, fractions):
    """A lattice's generator fractions: its ENSEMBLE row, unless `fractions` names the lattice.

    RESEARCH CODE THAT NEEDS TO BE DELETED -- `fractions` exists so P09c's benchmark runs can put
    the old and the new fractions side by side from one commit. It goes when P09c closes, and
    ENSEMBLE is again the only source.
    """
    unknown = sorted(set(fractions) - set(ENSEMBLE))
    if unknown:
        raise ValueError(f'fractions names unknown Bravais lattices {unknown}')
    return dict(fractions.get(bravais_lattice, ENSEMBLE[bravais_lattice]['fractions']))

def lattice_redistribution(bravais_lattice, redistribution):
    """A lattice's (max_neighbors, neighbor_radius): its ENSEMBLE row, unless `redistribution`
    names the lattice.

    RESEARCH CODE THAT NEEDS TO BE DELETED -- `redistribution` exists so P09c's benchmark runs can
    put the old and the re-derived constants side by side from one commit. It goes when P09c
    closes.
    """
    unknown = sorted(set(redistribution) - set(ENSEMBLE))
    if unknown:
        raise ValueError(f'redistribution names unknown Bravais lattices {unknown}')
    if bravais_lattice in redistribution:
        max_neighbors, neighbor_radius = redistribution[bravais_lattice]
    else:
        max_neighbors = ENSEMBLE[bravais_lattice]['max_neighbors']
        neighbor_radius = ENSEMBLE[bravais_lattice]['neighbor_radius']
    if not (int(max_neighbors) == max_neighbors and max_neighbors >= 1 and neighbor_radius >= 0):
        raise ValueError(f'{bravais_lattice}: max_neighbors must be a whole number >= 1 and '
                         f'neighbor_radius >= 0, got {max_neighbors}, {neighbor_radius}')
    return int(max_neighbors), float(neighbor_radius)

def lattice_budget(bravais_lattice, n_candidates_scale, budget_scale):
    """How many candidates a lattice generates: its ENSEMBLE budget times both scales.

    `n_candidates_scale` applies to every lattice; `budget_scale` maps a Bravais lattice to a
    further factor for that lattice alone, and a lattice it does not name keeps a factor of one.
    """
    unknown = sorted(set(budget_scale) - set(ENSEMBLE))
    if unknown:
        raise ValueError(f'budget_scale names unknown Bravais lattices {unknown}')
    return int(
        n_candidates_scale * budget_scale.get(bravais_lattice, 1) * ENSEMBLE[bravais_lattice]['n_candidates'])


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
    rf_group_params = {'tag': f'cubic_{broadening_tag}'}
    abnn_group_params = {'tag': f'cubic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'cubic_{broadening_tag}'}}
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
        'n_candidates_scale': n_candidates_scale,
        'iteration_info': iteration_info,
        'convergence_testing': False,
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
        rf_group_params,
        template_params,
        abnn_group_params,
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
    abnn_group_params = {'tag': f'tetragonal_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'tetragonal_{broadening_tag}'}}
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
        'n_candidates_scale': n_candidates_scale,
        'iteration_info': iteration_info,
        'convergence_testing': False,
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
        rf_group_params,
        template_params,
        abnn_group_params,
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
    abnn_group_params = {'tag': f'hexagonal_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'hexagonal_{broadening_tag}'}}
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
        'n_candidates_scale': n_candidates_scale,
        'iteration_info': iteration_info,
        'convergence_testing': False,
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
        rf_group_params,
        template_params,
        abnn_group_params,
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
    abnn_group_params = {'tag': f'rhombohedral_{broadening_tag}', 'quantitized_model': True}
    random_params = {bravais_lattice: {'tag': f'rhombohedral_{broadening_tag}'}}
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
        'n_candidates_scale': n_candidates_scale,
        'iteration_info': iteration_info,
        'convergence_testing': False,
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
        rf_group_params,
        template_params,
        abnn_group_params,
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
    abnn_group_params = {'tag': f'orthorhombic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'orthorhombic_{broadening_tag}'}}
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
        'n_candidates_scale': n_candidates_scale,
        'iteration_info': iteration_info,
        'convergence_testing': False,
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
        rf_group_params,
        template_params,
        abnn_group_params,
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
    abnn_group_params = {'tag': f'monoclinic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'monoclinic_{broadening_tag}'}}
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
        'n_candidates_scale': n_candidates_scale,
        'iteration_info': iteration_info,
        'convergence_testing': False,
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
        rf_group_params,
        template_params,
        abnn_group_params,
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
    abnn_group_params = {'tag': f'triclinic_{broadening_tag}'}
    random_params = {bravais_lattice: {'tag': f'triclinic_{broadening_tag}'}}
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
        'n_candidates_scale': n_candidates_scale,
        'iteration_info': iteration_info,
        'convergence_testing': False,
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
        rf_group_params,
        template_params,
        abnn_group_params,
        random_params,
        bravais_lattice,
        comm,
        fom,
        seed=seed,
        )
    return optimizer


def get_optimizers(rank, mpi_organizers, broadening_tag, n_candidates_scale, logger=None, optimizer_class=None, seed=12345, options=None):
    """Build one optimizer per Bravais lattice this rank manages.

    `options` is a flat dict merged over each lattice's `opt_params` after the
    defaults are built, which is how a driver script reaches settings that are
    deliberately not command-line flags -- they are research knobs, not user
    controls. The seven per-system factories already accept and merge it; this
    threads it through to them.
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
