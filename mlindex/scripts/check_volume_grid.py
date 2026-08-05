"""Compare branch grids before and after changing how the volume branches are drawn.

The branch grid decides which trial volume scales the integral filter can render an entry at. Two
things about it matter and trade against each other. Its range decides whether an entry has any
usable branch at all -- entries outside it are the worst predicted group by a wide margin. Its
spacing decides how far an entry lands from its nearest branch, and error rises once that
misalignment passes roughly half the peak width.

Run this before retraining anything. It rebuilds the grid for every split group under both the old
and new settings and reports both quantities, so a change that helps one split group at another's
expense is visible before an hour of training rather than after.

    python mlindex/scripts/check_volume_grid.py [--tag monoclinic_1] [--blend 0.5]
                                                [--lower 0.001] [--upper 0.999]
"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq
import scipy.ndimage
import scipy.stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIT_CELL_INDICES = {'monoclinic': [0, 1, 2, 4], 'triclinic': [0, 1, 2, 3, 4, 5],
                     'orthorhombic': [0, 1, 2], 'tetragonal': [0, 2], 'hexagonal': [0, 2],
                     'rhombohedral': [0, 3], 'cubic': [0]}


def build_grid(reciprocal_volume, q2_obs_scale, n_volumes, lower, upper, blend):
    """Reproduces ExtractionLayer.__init__, including the data-median normalization."""
    srt = np.sort(reciprocal_volume)
    bins = np.linspace(srt[int(lower*srt.size)], srt[int(upper*srt.size)], 401)
    hist, _ = np.histogram(reciprocal_volume, bins=bins, density=True)
    rv = scipy.stats.rv_histogram(
        (scipy.ndimage.gaussian_filter1d(hist, sigma=3, mode='constant'), bins), density=True
        )
    if blend >= 1.0:
        volumes = (rv.ppf(np.linspace(0.001, 0.999, n_volumes))/q2_obs_scale**2)**(2/3)
    else:
        quantiles = np.linspace(0.001, 0.999, 20001)
        reciprocal_v = 1.0/(rv.ppf(quantiles)/q2_obs_scale**2)**(2/3)
        weight = np.abs(np.gradient(reciprocal_v, quantiles))**(1 - blend)
        cumulative = np.concatenate(
            [[0], np.cumsum((weight[1:] + weight[:-1])/2*np.diff(quantiles))])
        cumulative /= cumulative[-1]
        volumes = np.sort(1.0/np.interp(np.linspace(0, 1, n_volumes), cumulative, reciprocal_v))
    normalization = np.median((reciprocal_volume/q2_obs_scale**2)**(2/3))
    return volumes/normalization, normalization


def fit_sigma(q2_obs_scaled, scales, n_filters, extraction_peak_length):
    """Reproduces the sigma fit, which depends on the grid only through the normalization."""
    upper = np.sort(q2_obs_scaled[:, :extraction_peak_length].ravel())
    upper = upper[int(0.98*upper.size)]
    spacing = (upper - upper/n_filters)/(n_filters - 1)
    projected = np.sort(q2_obs_scaled[:, :extraction_peak_length]/scales[:, np.newaxis], axis=1)
    separations = np.diff(projected, axis=1).ravel()
    separations = separations[separations > 0]
    measured = np.quantile(separations, 0.05)/2
    return min(max(measured, 2*spacing), 6*spacing), spacing


def report(name, data, params, settings, xnn_offset):
    reciprocal_volume = 1.0/np.stack(data['reindexed_volume'].values).astype(float)
    q2 = np.stack([np.asarray(x)[:params['extraction_peak_length']] for x in data['q2'].values])
    q2_obs_scale = float(q2.std())
    print(f'\n{name}   n={len(data)}   q2_obs_scale={q2_obs_scale:.5f}')
    print('  %-26s %14s %10s %9s %9s %9s' % (
        'grid', 'v range', 'outside', 'med mis', '>0.5 sig', '>1.0 sig'))
    for label, (lower, upper, blend) in settings:
        volumes, normalization = build_grid(
            reciprocal_volume, q2_obs_scale, params['n_volumes'], lower, upper, blend)
        scaled = q2/q2_obs_scale
        entry_scales = (reciprocal_volume/q2_obs_scale**2)**(2/3)/normalization
        sigma, _ = fit_sigma(scaled, entry_scales, params['n_filters'],
                             params['extraction_peak_length'])
        outside = (entry_scales < volumes.min()) | (entry_scales > volumes.max())
        matched = np.abs(
            np.log(entry_scales)[:, np.newaxis] - np.log(volumes)[np.newaxis]).argmin(axis=1)
        misalignment = np.median(
            np.abs(scaled*(1/volumes[matched] - 1/entry_scales)[:, np.newaxis])/sigma, axis=1)
        print('  %-26s %6.3f-%-7.3f %9.2f%% %9.3f %8.1f%% %8.1f%%' % (
            label, volumes.min(), volumes.max(), 100*outside.mean(), np.median(misalignment),
            100*(misalignment > 0.5).mean(), 100*(misalignment > 1.0).mean()))
        print('  %-26s sigma %.5f    max bias term %.1f  (log-cosh clips at 75)' % (
            '', sigma, xnn_offset*(volumes.max() - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='monoclinic_1')
    parser.add_argument('--lattice-system', default='monoclinic')
    parser.add_argument('--lower', type=float, default=0.001)
    parser.add_argument('--upper', type=float, default=0.999)
    parser.add_argument('--blend', type=float, default=0.5)
    parser.add_argument('--n-volumes', type=int, default=150)
    parser.add_argument('--n-filters', type=int, default=1000)
    parser.add_argument('--extraction-peak-length', type=int, default=10)
    args = parser.parse_args()

    path = os.path.join(BASE, 'mlindex', 'models', args.tag, 'data', 'data.parquet')
    table = pq.read_table(path, columns=['split_group', 'train', 'q2', 'reindexed_xnn',
                                         'reindexed_volume'],
                          filters=[('train', '==', True)]).to_pandas()
    params = {'n_volumes': args.n_volumes, 'n_filters': args.n_filters,
              'extraction_peak_length': args.extraction_peak_length}
    settings = [('current (default)', (0.005, 0.990, 1.0)),
                ('proposed', (args.lower, args.upper, args.blend))]
    indices = UNIT_CELL_INDICES[args.lattice_system]
    for split_group in sorted(table['split_group'].unique()):
        data = table[table['split_group'] == split_group]
        if len(data) < 1000:
            print(f'\n{split_group}: only {len(data)} entries, skipped')
            continue
        xnn = np.stack(data['reindexed_xnn'].values)[:, indices]
        mean = np.median(xnn, axis=0)
        offset = float(np.max(np.abs(mean/np.median(np.abs(xnn - mean), axis=0))))
        report(split_group, data, params, settings, offset)


if __name__ == '__main__':
    main()
