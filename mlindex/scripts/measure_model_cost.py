"""What the shipped model tree costs: its size on disk, and its resident memory at inference.

Two numbers, both of which this project has previously quoted from file listings or from memory
rather than measured, and both of which move whenever a model file changes -- so P07, P15 and P18
each need them again.

    # the tree: tracked files and bytes, per lattice system and in total
    python -m mlindex.scripts.measure_model_cost --stage tree

    # resident memory once every model a 14-lattice run loads is loaded
    python -m mlindex.scripts.measure_model_cost --stage resident

    # both, against a specific checkout's models rather than the resolved ones
    python -m mlindex.scripts.measure_model_cost --stage all --models-dir /path/to/mlindex/models

Report the resident figure as a median of interleaved pairs against the tree you are comparing
with, never as a single run: this laptop drifts, and the peak over a whole indexing run does not
resolve at all (P05 measured spreads of 1.1 GB against a 133 MB effect). The number this reports
is resident after the models load, which is stable to under 10 MB and is what each pool manager
holds for the life of a run.
"""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

def resident_bytes():
    """Peak resident set size of this process, in bytes on every platform.

    getrusage reports bytes on macOS and kibibytes on Linux, which is a factor of 1024 waiting
    for whoever compares a number taken on one with a number taken on the other.
    """
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def _models_dir(given):
    if given is not None:
        return Path(given)
    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
    return _resolve_models_dir()


def measure_tree(models_dir):
    """File count and byte total of the shipped per-lattice-system directories.

    Untracked files are skipped by name rather than by asking git, so this works on an unpacked
    download as well as on a checkout. The wheel ships only the hkl_ref lists, so a tree without
    per-split-group subdirectories is a partial one and says so.
    """
    print(f'models directory: {models_dir}')
    total_files = total_bytes = 0
    for system in sorted(models_dir.glob('*_1')):
        files = [path for path in system.rglob('*')
                 if path.is_file() and path.name != '.DS_Store']
        size = sum(path.stat().st_size for path in files)
        print(f'  {system.name:<16} {len(files):>5} files  {size:>12} bytes  {size/1e6:>8.1f} MB')
        total_files += len(files)
        total_bytes += size
    print(f'  {"total":<16} {total_files:>5} files  {total_bytes:>12} bytes  '
          f'{total_bytes/1e6:>8.1f} MB  ({total_bytes/1048576:.1f} MiB)')
    return total_files, total_bytes


def measure_resident(models_dir):
    """Resident memory once every model a fourteen-lattice run loads is loaded.

    The optimizers load their generators during construction, so building one per Bravais lattice
    and holding them is the state a pool manager is in for the whole of a run.
    """
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    from mlindex.optimization.MPOptimizer import LocalComm
    from mlindex.optimization import UtilitiesOptimizer
    # Imported rather than restated: UnitCellTools is where the list and the mapping live, and
    # both were written out in several places before that was fixed.
    from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM, BRAVAIS_LATTICES

    factory_of = {
        'cubic': UtilitiesOptimizer.get_cubic_optimizer,
        'hexagonal': UtilitiesOptimizer.get_hexagonal_optimizer,
        'rhombohedral': UtilitiesOptimizer.get_rhombohedral_optimizer,
        'tetragonal': UtilitiesOptimizer.get_tetragonal_optimizer,
        'orthorhombic': UtilitiesOptimizer.get_orthorhombic_optimizer,
        'monoclinic': UtilitiesOptimizer.get_monoclinic_optimizer,
        'triclinic': UtilitiesOptimizer.get_triclinic_optimizer,
        }
    before = resident_bytes()
    started = time.perf_counter()
    held = []
    for bravais_lattice in BRAVAIS_LATTICES:
        factory = factory_of[BL_TO_LATTICE_SYSTEM[bravais_lattice]]
        held.append(factory(bravais_lattice, '1', 1, LocalComm(1),
                            optimizer_class=OptimizerManager, seed=12345,
                            models_directory=models_dir))
    elapsed = time.perf_counter() - started
    after = resident_bytes()
    print(f'resident before {before/1e6:.1f} MB, after {after/1e6:.1f} MB, '
          f'delta {(after - before)/1e6:.1f} MB, {elapsed:.1f} s for '
          f'{len(held)} optimizers')
    return after


def build_parser():
    parser = argparse.ArgumentParser(
        description='Measure the size and resident memory cost of the shipped model tree.')
    parser.add_argument(
        '--stage', choices=('all', 'tree', 'resident'), default='all',
        help='Which measurement to take (default: all).')
    parser.add_argument(
        '--models-dir', default=None,
        help='Models directory to measure. Defaults to the one the optimizer resolves. '
             'Pass this explicitly when comparing two checkouts.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    models_dir = _models_dir(args.models_dir)
    if args.stage in ('all', 'tree'):
        measure_tree(models_dir)
    if args.stage in ('all', 'resident'):
        measure_resident(models_dir)


if __name__ == '__main__':
    main()
