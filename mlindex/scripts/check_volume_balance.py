"""Check how the integral filter's accuracy varies with unit cell size.

The 150 volume branches of the integral filter scan over a trial volume scale. Each validation entry
has a true scale, and so a branch that matches it. Binning the validation error by that branch shows
whether the model is uniformly accurate across cell sizes or is trading one end against the other.

Two errors are reported per bin:

  relative     |pred - true| / |true|. What matters for indexing, and what the verdict is based on.
  loss-space   |pred - true| / xnn_scale. The units the loss actually minimizes.

The two are not independent: loss-space = relative * |true|/xnn_scale, and the second factor rises
with cell size because xnn magnitude is tied to the volume scale. That factor is printed as well, so
the loss-space column can be read against its own floor. A model with perfectly flat relative error
still shows a rising loss-space column; that is arithmetic, not a defect.

    python mlindex/scripts/check_volume_balance.py [model_dir] [--n 8000]

Run it with the pytorch env, which has onnxruntime, h5py and pyarrow:
    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python3
"""
import argparse
import os

import h5py
import numpy as np
import onnxruntime
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(BASE, 'mlindex', 'models', 'monoclinic_1', 'integral_filter', 'mP_0_01')
DEFAULT_DATA = os.path.join(BASE, 'mlindex', 'models', 'monoclinic_1', 'data', 'data.parquet')
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 135), (135, 150)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', nargs='?', default=DEFAULT_DIR)
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--split-group', default='mP_0_01')
    parser.add_argument('--tag', default='monoclinic_1')
    parser.add_argument('--n', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    split_group, tag = args.split_group, args.tag
    prefix = os.path.join(args.model_dir, f'{split_group}_')
    print(f'model: {args.model_dir}')

    # The branch volumes are only stored in the keras checkpoint, not in the ONNX graph.
    with h5py.File(f'{prefix}pitf_weights_{tag}.weights.h5', 'r') as handle:
        volumes = np.array(handle['layers/extraction_layer/vars/0']).ravel()

    q2_obs_scale = float(np.load(f'{prefix}q2_obs_scale_{tag}.npy'))
    xnn_mean, xnn_scale = [np.ravel(a) for a in np.load(f'{prefix}xnn_scaler_{tag}.npy')]
    unit_cell_indices = [0, 1, 2, 4] if split_group[0] == 'm' else None
    if unit_cell_indices is None:
        raise SystemExit(f'unit_cell_indices not known for split group {split_group}')

    data = pq.read_table(
        args.data,
        columns=['split_group', 'train', 'q2', 'reindexed_xnn', 'reindexed_volume'],
        filters=[('split_group', '==', split_group), ('train', '==', False)],
        ).to_pandas()
    xnn_true = np.stack(data['reindexed_xnn'].values).astype(float)[:, unit_cell_indices]

    # Each entry's own volume scale, put on the same normalization as the branch volumes, then
    # matched to the nearest branch in log space.
    reciprocal_volume = 1.0/data['reindexed_volume'].values.astype(float)
    scales = (reciprocal_volume/q2_obs_scale**2)**(2/3)
    scales /= np.median(scales)/np.median(volumes)
    matched = np.abs(np.log(scales)[:, None] - np.log(volumes)[None, :]).argmin(axis=1)

    session = onnxruntime.InferenceSession(f'{prefix}pitf_weights_{tag}_quantized.onnx')
    input_name = session.get_inputs()[0].name
    n_sample = min(args.n, len(data))
    indices = np.random.default_rng(args.seed).choice(len(data), size=n_sample, replace=False)
    q2_all = np.stack([np.asarray(x)[:20] for x in data['q2'].values[indices]])/q2_obs_scale

    n_components = len(unit_cell_indices)
    loss_error = np.zeros(n_sample)
    relative_error = np.zeros(n_sample)
    for row_index, row in enumerate(q2_all):
        output = session.run(None, {input_name: row.astype(np.float32)[None]})[0][0]
        branch = output[:, n_components].argmax()
        predicted = output[branch, :n_components]*xnn_scale + xnn_mean
        true = xnn_true[indices[row_index]]
        loss_error[row_index] = np.median(np.abs((predicted - true)/xnn_scale))
        relative_error[row_index] = np.median(np.abs(predicted - true)/np.abs(true))

    matched_bin = matched[indices]
    print(f'\n{n_sample} validation entries\n')
    print('%-9s %6s %8s %12s %10s %12s' % (
        'branch', 'n', 'med v', '|true|/scale', 'rel err', 'loss-space'))
    relative_by_bin, loss_by_bin = [], []
    for low, high in BINS:
        mask = (matched_bin >= low) & (matched_bin < high)
        if not mask.sum():
            continue
        magnitude = np.median(np.median(np.abs(xnn_true[indices[mask]])/xnn_scale, axis=1))
        relative_by_bin.append(np.median(relative_error[mask]))
        loss_by_bin.append(np.median(loss_error[mask]))
        print('%-9s %6d %8.3f %12.3f %9.1f%% %12.4f' % (
            f'{low}-{high}', mask.sum(), np.median(volumes[matched_bin[mask]]), magnitude,
            100*relative_by_bin[-1], loss_by_bin[-1]))

    middle = np.median([loss_by_bin[index] for index in (2, 3)])
    ratio = max(loss_by_bin[0], loss_by_bin[-1])/middle
    magnitudes = [
        np.median(np.median(np.abs(xnn_true[indices[(matched_bin >= low) & (matched_bin < high)]])
                            / xnn_scale, axis=1))
        for low, high in BINS
        ]
    floor = max(magnitudes[0], magnitudes[-1])/np.median([magnitudes[2], magnitudes[3]])

    print('\nVERDICT METRICS (relative error is what matters for indexing)')
    print('  relative error, worst/best bin : %.2f      (baseline 1.40)' % (
        max(relative_by_bin)/min(relative_by_bin)))
    print('  overall median relative error  : %.1f%%     (baseline ~12%%, must not regress)' % (
        100*np.median(relative_error)))
    print('\nSECONDARY (reported, not gating)')
    print('  loss-space worst/best ratio    : %.2f      (baseline 3.10)' % ratio)
    print('  ...its floor at flat rel error : %.2f      (a ratio of 1.0 is not achievable)' % floor)
    print('  ratio relative to that floor   : %.2f      (baseline 1.23)' % (ratio/floor))


if __name__ == '__main__':
    main()
