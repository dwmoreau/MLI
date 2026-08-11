"""Which model of the calculated-line density is right? (S01 item 0, resolving Q3.)

Every probabilistic figure of merit in this repo rests on one quantity: how many calculated lines
a candidate puts below a given q2. get_M20_likelihood models it as Taupin's

    N(Q) = (4 pi / 3) Q^(3/2) / (V* mu)

with mu a single constant per Bravais lattice. DWMM tuned those constants by hand to match the
observed density of non-systematically-absent reflections in the training set (F-022), which is why
they drift from Taupin's published values -- hexagonal 14 against 24, triclinic 1.8 against 2.

de Wolff (1961) gives the same quantity with no free parameters at all:

    N(Q) = Q(C0 sqrt(Q) + C1 a* + C2 b* + C3 c*) / V*

Both are fits to the same thing, but de Wolff's carries a surface term in Q rather than only the
volume term in Q^(3/2), and the surface term dominates at low Q -- exactly where the first twenty
peaks live. The hypothesis (F-015) is that the hand-tuned constants are one number standing in for
that missing sqrt(Q)-dependent term, and are therefore biased low at small Q.

This script settles it. For validation entries it counts the true number of distinct calculated
lines below each observed peak and compares all three models against that count, resolved by peak
index so the bias can be read as a function of q2.

    python mlindex/scripts/run_fom_density_model.py

Run it with the development env:
    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.scripts.run_fom_audits import TAGS, load_entries, read_params  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_multiplicity_taupin88  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_n_dewolff61  # noqa: E402
from mlindex.utilities.UnitCellTools import get_hkl_matrix  # noqa: E402
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn  # noqa: E402
from mlindex.utilities.UnitCellTools import get_unit_cell_volume  # noqa: E402

# Taupin 1988's published multiplicities, mu = mu_lattice * mu_system, which the repo replaced.
TAUPIN_ORIGINAL = {
    'cF': 4*32, 'cI': 2*32, 'cP': 1*32,
    'hP': 1*24, 'hR': 1*24,
    'tI': 2*16, 'tP': 1*16,
    'oC': 2*8, 'oI': 2*8, 'oF': 4*8, 'oP': 1*8,
    'mC': 2*4, 'mP': 1*4,
    'aP': 1*2,
    }


def taupin_count(q2, reciprocal_volume, mu):
    """N(Q) = (4 pi / 3) Q^(3/2) / (V* mu), the density get_M20_likelihood integrates."""
    return (4*np.pi/3)*q2**1.5/(reciprocal_volume[:, None]*mu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--n-max', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, by_peak = [], []
    for bravais_lattice, tag in TAGS.items():
        params = read_params(tag)
        n_peaks = params['n_peaks']
        lattice_system = params['lattice_system']
        hkl_ref = np.load(os.path.join(
            BASE, 'mlindex', 'models', tag, 'data', f'hkl_ref_{bravais_lattice}.npy'))
        # The (0,0,0) sentinel would add one line at q2 = 0 to every count.
        hkl_ref = hkl_ref[:-1] if np.all(hkl_ref[-1] == 0) else hkl_ref
        entries = load_entries(bravais_lattice, tag, args.n_max, args.seed)
        if len(entries) == 0:
            continue

        xnn = np.stack(entries['reindexed_xnn'].values).astype(float)
        xnn = xnn[:, params['unit_cell_indices']]
        q2_obs = np.stack([np.asarray(row) for row in entries['q2'].values]).astype(float)

        # The true count. Audit A established that the reference list never saturates below the
        # 20th peak, so this is exact rather than truncated.
        q2_ref = xnn @ get_hkl_matrix(hkl_ref, lattice_system).T
        actual = (q2_ref[:, None, :] < q2_obs[:, :, None]).sum(axis=2).astype(float)

        # V* exactly as get_M20_likelihood_from_xnn computes it. It must NOT come from
        # get_dewolff61_axes: that returns the *hexagonal*-axis V* for hR, one third of the
        # rhombohedral one the repo uses, and this column is auditing the repo's model.
        # get_n_dewolff61 computes its own V* internally and is unaffected.
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=lattice_system)
        reciprocal_volume = get_unit_cell_volume(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system)

        models = {
            'dewolff61': get_n_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice),
            'taupin_repo': taupin_count(
                q2_obs, reciprocal_volume, get_multiplicity_taupin88(bravais_lattice)[0]),
            'taupin_published': taupin_count(
                q2_obs, reciprocal_volume, TAUPIN_ORIGINAL[bravais_lattice]),
            }

        valid = actual > 0
        for name, predicted in models.items():
            relative = np.where(valid, (predicted - actual)/np.maximum(actual, 1), np.nan)
            rows.append({
                'bravais_lattice': bravais_lattice,
                'model': name,
                'n_entries': len(entries),
                'median_rel_bias': float(np.nanmedian(relative)),
                'median_abs_rel_error': float(np.nanmedian(np.abs(relative))),
                # The first five peaks carry most of the indexing information and are where the
                # surface term matters most.
                'median_rel_bias_peaks_1_5': float(np.nanmedian(relative[:, :5])),
                'median_rel_bias_peaks_last5': float(np.nanmedian(relative[:, n_peaks - 5:])),
                })
            for peak in range(n_peaks):
                by_peak.append({
                    'bravais_lattice': bravais_lattice,
                    'model': name,
                    'peak_index': peak + 1,
                    'median_rel_bias': float(np.nanmedian(relative[:, peak])),
                    'median_actual_N': float(np.median(actual[:, peak])),
                    })
        print(f'{bravais_lattice}: {len(entries)} entries, median N at last peak '
              f'{np.median(actual[:, -1]):.0f}')

    summary = pd.DataFrame(rows)
    peaks = pd.DataFrame(by_peak)
    summary.to_csv(os.path.join(args.out, 'S01_density_model.csv'), index=False)
    peaks.to_csv(os.path.join(args.out, 'S01_density_model_by_peak.csv'), index=False)

    print('\n=== median relative bias, (predicted - actual)/actual')
    print(summary.pivot(index='bravais_lattice', columns='model',
                        values='median_rel_bias').to_string())
    print('\n=== median relative bias over the first five peaks')
    print(summary.pivot(index='bravais_lattice', columns='model',
                        values='median_rel_bias_peaks_1_5').to_string())

    order = list(TAGS.keys())
    figure, axes = plt.subplots(2, 7, figsize=(20, 6), sharey=True)
    colours = {'dewolff61': '#1b4965', 'taupin_repo': '#c1666b', 'taupin_published': '#8a9b68'}
    for index, bravais_lattice in enumerate(order):
        axis = axes[index//7, index % 7]
        subset = peaks[peaks['bravais_lattice'] == bravais_lattice]
        for name, colour in colours.items():
            model = subset[subset['model'] == name]
            axis.plot(model['peak_index'], 100*model['median_rel_bias'],
                      color=colour, marker='o', markersize=2.5, linewidth=1.2, label=name)
        axis.axhline(0, color='0.3', linewidth=0.8, zorder=0)
        axis.set_title(bravais_lattice, fontsize=11)
        axis.set_xlabel('peak index')
        axis.set_ylim(-100, 150)
        if index % 7 == 0:
            axis.set_ylabel('median bias in N (%)')
    axes[0, 0].legend(fontsize=8, frameon=False)
    figure.suptitle('Calculated-line count: predicted minus actual, by peak index', fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(args.out, 'S01_density_model.png'), dpi=200)
    print(f'\nwrote tables and figure to {args.out}')


if __name__ == '__main__':
    main()
