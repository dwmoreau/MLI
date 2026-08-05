"""Check that the extraction layer renders every volume branch at the same peak width.

The layer compares the observed peaks against a fixed filter grid for each trial volume scale. Peak p
of branch v lands at q2_p / v on that grid. It used to be rendered with a width of sigma / v, so the
same unit cell shape reached the shared attention embedding as a differently resolved pattern
depending on its volume. It is now rendered with a width of sigma on every branch.

This script drives the real ExtractionLayer.call() with weights from a trained checkpoint and checks
both properties: the width no longer depends on the branch, and the peak positions did not move. Run
it after any change to ExtractionLayer.

    python mlindex/scripts/check_extraction_layer.py [weights.h5]
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import sys

import h5py
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS = os.path.join(
    BASE, 'mlindex', 'models', 'monoclinic_1', 'integral_filter_fwhmfix', 'mP_0_01',
    'mP_0_01_pitf_weights_monoclinic_1.weights.h5',
    )
Q2_PEAK = 1.7
EXTRACTION_PEAK_LENGTH = 10


def load_extraction_weights(path):
    with h5py.File(path, 'r') as handle:
        group = handle['layers/extraction_layer/vars']
        return (
            np.array(group['0']).ravel(),
            np.array(group['1']).ravel(),
            float(np.array(group['2'])),
            )


def build_layer(volumes, filters, sigma):
    import keras
    from mlindex.model_training.Networks import ExtractionLayer
    model_params = {
        'n_volumes': volumes.size,
        'n_filters': filters.size,
        'extraction_peak_length': EXTRACTION_PEAK_LENGTH,
        }
    layer = ExtractionLayer(model_params, None, None, None, 1.0, name='extraction_layer')
    layer.volumes.assign(keras.ops.cast(volumes[:, np.newaxis], dtype='float32'))
    layer.filters.assign(keras.ops.cast(filters[np.newaxis], dtype='float32'))
    layer.sigma.assign(keras.ops.cast(sigma, dtype='float32'))
    return layer


def full_width_half_max(profile, grid):
    """Width of the peak in profile, interpolated so the answer is not quantized to the grid."""
    peak = profile.argmax()
    half = profile[peak]/2
    if profile[peak] <= 0:
        return np.nan
    crossings = []
    for step in (-1, 1):
        index = peak
        while 0 <= index + step < profile.size and profile[index + step] > half:
            index += step
        neighbour = index + step
        if not 0 <= neighbour < profile.size:
            return np.nan
        span = profile[index] - profile[neighbour]
        fraction = 0.0 if span == 0 else (profile[index] - half)/span
        crossings.append(grid[index] + fraction*(grid[neighbour] - grid[index]))
    return abs(crossings[1] - crossings[0])


def main():
    import keras
    weights_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    volumes, filters, sigma = load_extraction_weights(weights_path)
    print(f'weights:     {weights_path}')
    print(f'n_volumes:   {volumes.size}')
    print(f'n_filters:   {filters.size}')
    print(f'sigma:       {sigma:.5f}\n')

    layer = build_layer(volumes, filters, sigma)
    q2_obs = np.full((1, EXTRACTION_PEAK_LENGTH), Q2_PEAK, dtype='float32')
    metric = np.asarray(keras.ops.convert_to_numpy(layer.call(keras.ops.cast(q2_obs, 'float32'))))[0]

    print(f'A single peak at q2 = {Q2_PEAK}, rendered on each volume branch:\n')
    print('%7s %8s %10s %10s %10s %10s' % ('branch', 'v', 'argmax', 'q2/v', 'FWHM', 'was'))
    widths, shifts = [], []
    shown = [0, volumes.size//6, volumes.size//3, volumes.size//2,
             2*volumes.size//3, 5*volumes.size//6, volumes.size - 1]
    for branch in range(volumes.size):
        width = full_width_half_max(metric[branch], filters)
        expected = Q2_PEAK/volumes[branch]
        observed = filters[metric[branch].argmax()]
        if expected <= filters[-1]:
            widths.append(width)
            shifts.append(abs(observed - expected))
        if branch in shown:
            print('%7d %8.3f %10.4f %10.4f %10.5f %10.5f' % (
                branch, volumes[branch], observed, expected, width, width/volumes[branch]))

    widths = np.array(widths)
    widths = widths[np.isfinite(widths)]
    spacing = filters[1] - filters[0]
    print(f'\nFWHM spread now:      {widths.max()/widths.min():.4f}x')
    print(f'2.355 * sigma:        {2.355*sigma:.5f}')
    print(f'mean FWHM:            {widths.mean():.5f}')

    constant_width = np.isclose(widths.mean(), 2.355*sigma, rtol=0.05)
    unmoved = max(shifts) < spacing
    if not constant_width:
        print(f'\nFAIL: mean FWHM {widths.mean():.5f} is not 2.355 * sigma {2.355*sigma:.5f}')
        return 1
    if not unmoved:
        print(f'\nFAIL: peak centres moved by up to {max(shifts):.5f}, more than one filter '
              f'spacing ({spacing:.5f})')
        return 1
    print('\nPASS: constant FWHM across branches, peak centres unmoved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
