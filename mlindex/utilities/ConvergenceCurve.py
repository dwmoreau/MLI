"""The measured convergence-radius curves, and what a starting distance is worth on one.

A curve is the two-row array `run_convergence_radius.py` writes: row 0 the shell radii, in the
same units as xnn, and row 1 the fraction of candidate cells started at that radius which refined
to the true cell. Each lattice has its own curve, measured once over its own peak count and
dropout, and the curves are treated as a fixed input.

The curves are run output, not package data -- `mlindex/characterization/` is not tracked -- so
every entry point here takes the directory holding them rather than reconstructing one from a
path shape.
"""
import os

import numpy as np


# Each lattice's curve was measured at that lattice's own peak count and dropout, and the file is
# named after them. The rest of the name is the same for every lattice.
ROC_TAG = {
    'cF': 'peaks10_drop8',
    'cI': 'peaks10_drop8',
    'cP': 'peaks10_drop8',
    'hP': 'peaks20_drop17',
    'hR': 'peaks20_drop17',
    'tI': 'peaks20_drop17',
    'tP': 'peaks20_drop17',
    'oC': 'peaks20_drop16',
    'oF': 'peaks20_drop16',
    'oI': 'peaks20_drop16',
    'oP': 'peaks20_drop16',
    'mC': 'peaks20_drop14',
    'mP': 'peaks20_drop14',
    'aP': 'peaks20_drop11',
    }
N_ITERATIONS = 100
SAMPLING = 'sampQ2'


def curve_path(roc_directory, bravais_lattice):
    """The one place the curve filename is composed."""
    if bravais_lattice not in ROC_TAG:
        raise ValueError(f'no convergence curve is defined for {bravais_lattice!r}')
    name = (f'{bravais_lattice}_roc_{ROC_TAG[bravais_lattice]}'
            f'_iter{N_ITERATIONS}_{SAMPLING}.npy')
    return os.path.join(roc_directory, name)


def load_curve(roc_directory, bravais_lattice):
    """-> (radii, success_rate), each (n_shells,).

    Raises with the path it wanted, because the usual cause is a directory that does not hold the
    curves rather than a corrupted file.
    """
    path = curve_path(roc_directory, bravais_lattice)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'no convergence curve for {bravais_lattice} at {path}. The curves are run output and '
            f'are not shipped with the package; point --roc-dir at the directory holding them.'
            )
    curve = np.load(path)
    if curve.ndim != 2 or curve.shape[0] != 2:
        raise ValueError(f'{path} is not a two-row convergence curve; its shape is {curve.shape}')
    return curve[0], curve[1]


def success_of_distance(radii, success_rate, distance):
    """s(d) for every distance, by lookup on the curve's own shells.

    A candidate closer than the first shell takes the first shell's rate, which is the highest that
    was measured. A candidate further out than the LAST shell is worth nothing -- anything else
    extrapolates past where the curve was measured, and it matters: the last shell's rate is small
    but not zero, so crediting a few thousand hopeless candidates with it drives a combined
    probability to exactly 1 for every generator mix and the score stops separating them.

    `MITemplates` does the same lookup but clamps to the last shell instead of zeroing, for the
    regression targets its calibrator is fitted on. That difference is deliberate and is not
    reproduced here: every shipped template model was fitted against the clamped values, so
    changing them is a retrain, not a refactor. It is recorded in `STATUS.md` rather than carried
    as an argument nothing in this package sets.
    """
    index = np.searchsorted(radii, distance)
    past_last = index >= radii.size
    rate = success_rate[np.clip(index, 0, radii.size - 1)]
    return np.where(past_last, 0.0, rate)
