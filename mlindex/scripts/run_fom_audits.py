"""The three implementation audits that gate everything else in the FOM zoo (S01 Part A2).

de Wolff wrote a whole paper in 1972 because a published author counted N20 wrongly and reported
M20 = 9 where the correct value was 4.9. These audits ask the same question of our own code, and
they run before any new figure of merit is trusted, because a subtly wrong M20 invalidates every
comparison downstream.

  A  Does the reference-hkl truncation bind?  The reference lists are cut to hkl_ref_length per
     model tag (100 cubic, up to 1000 monoclinic). If a candidate has more allowed reflections
     below the cut-off than the list holds, N20 is under-counted and M20 is *inflated* -- and
     inflated precisely for the large low-symmetry cells that dominate the failure mode. Reported
     against the true cell, and then against the true cell inflated in volume, which is the regime
     where a false candidate lives. The volume headroom is the answer: the volume multiple at which
     the list first saturates.

  B  What is counted in N20?  All distinct calculated Q below the cut-off, symmetry equivalents and
     systematic absences removed (de Wolff 1972). Checks that property directly on the reference
     lists, and quantifies the (0,0,0) sentinel row, which gives q2_calc = 0, is therefore below
     every cut-off, and inflates N by exactly one.

  C  Which cut-off?  get_M20 uses the *calculated* position of the last observed peak's assignment,
     not the observed Q20, and not the largest calculated value among the assigned lines. All three
     conventions exist. This measures how far apart they are in M20.

    python mlindex/scripts/run_fom_audits.py [--out docs/fom/artifacts]

Run it with the development env:
    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref  # noqa: E402
from mlindex.utilities.UnitCellTools import get_hkl_matrix  # noqa: E402

# Every Bravais lattice, in the order they are reported, with the model tag whose data directory
# holds the truncated reference list actually used at inference.
TAGS = {
    'cP': 'cubic_1', 'cI': 'cubic_1', 'cF': 'cubic_1',
    'tP': 'tetragonal_1', 'tI': 'tetragonal_1',
    'hP': 'hexagonal_1',
    'hR': 'rhombohedral_1',
    'oP': 'orthorhombic_1', 'oC': 'orthorhombic_1',
    'oI': 'orthorhombic_1', 'oF': 'orthorhombic_1',
    'mP': 'monoclinic_1', 'mC': 'monoclinic_1',
    'aP': 'triclinic_1',
    }

# The volume multiples probed in audit A. A false candidate that over-predicts lines is a cell
# larger than the truth, so this is the direction that matters.
VOLUME_FACTORS = (1, 2, 4, 8, 16, 32)


def centring_allowed(hkl, bravais_lattice):
    """Systematic absences due to lattice centring alone, as a boolean mask over hkl.

    Two of these depend on the setting the repo actually uses, not on the Bravais label:

      mC  is stored in the *I* setting -- SpaceGroups.py:318 lists its spacegroups as I 1 2 1 and
          I 1 a 1 -- so the rule is h+k+l = 2n, not h+k = 2n.
      hR  is stored on rhombohedral axes, not hexagonal ones: get_hkl_matrix's rhombohedral branch
          is (sum h^2, hk+hl+kl), which is a = b = c, alpha = beta = gamma. On those axes the
          R centring is absorbed into the cell and there are no centring absences at all.
    """
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    if bravais_lattice in ('cP', 'tP', 'hP', 'oP', 'mP', 'aP', 'hR'):
        return np.ones(len(hkl), dtype=bool)
    elif bravais_lattice == 'oC':
        return (h + k) % 2 == 0
    elif bravais_lattice in ('cI', 'tI', 'oI', 'mC'):
        return (h + k + l) % 2 == 0
    elif bravais_lattice in ('cF', 'oF'):
        return ((h + k) % 2 == 0) & ((h + l) % 2 == 0)
    raise ValueError(f'unknown centring in {bravais_lattice}')


def read_params(tag):
    params = pd.read_csv(os.path.join(BASE, 'mlindex', 'models', tag, 'data', 'data_params.csv'))
    indices = params['unit_cell_indices'][0].split('[')[1].split(']')[0].split(',')
    return {
        'lattice_system': params['lattice_system'][0],
        'n_peaks': int(params['n_peaks'][0]),
        'hkl_ref_length': int(params['hkl_ref_length'][0]),
        'unit_cell_indices': np.array(indices, dtype=int),
        }


def load_entries(bravais_lattice, tag, n_max, seed):
    """Validation entries only: train == False and not augmented, per F-021."""
    table = pq.read_table(
        os.path.join(BASE, 'mlindex', 'models', tag, 'data', 'data.parquet'),
        columns=['q2', 'reindexed_xnn', 'reindexed_volume',
                 'reindexed_h', 'reindexed_k', 'reindexed_l'],
        filters=[('bravais_lattice', '==', bravais_lattice),
                 ('train', '==', False),
                 ('augmented', '==', False)],
        ).to_pandas()
    if len(table) > n_max:
        indices = np.random.default_rng(seed).choice(len(table), size=n_max, replace=False)
        table = table.iloc[np.sort(indices)].reset_index(drop=True)
    return table


def audit_a(rows_a, rows_headroom, bravais_lattice, params, hkl_ref, entries):
    """Does the truncation bind? Counts reference lines below the cut-off, true cell and inflated."""
    n_peaks = params['n_peaks']
    # The (0,0,0) sentinel is the last row and is handled separately in audit B; exclude it here so
    # that "how many real reflections does the list offer" is not off by one.
    real_ref = hkl_ref[:-1] if np.all(hkl_ref[-1] == 0) else hkl_ref
    n_available = len(real_ref)

    hkl_matrix = get_hkl_matrix(real_ref, params['lattice_system'])
    xnn = np.stack(entries['reindexed_xnn'].values).astype(float)[:, params['unit_cell_indices']]
    q2_obs = np.stack([np.asarray(row) for row in entries['q2'].values]).astype(float)
    cutoff = q2_obs[:, n_peaks - 1]

    q2_ref = xnn @ hkl_matrix.T
    counts = (q2_ref < cutoff[:, None]).sum(axis=1)
    rows_a.append({
        'bravais_lattice': bravais_lattice,
        'tag': TAGS[bravais_lattice],
        'n_entries': len(entries),
        'hkl_ref_length': params['hkl_ref_length'],
        'n_available': n_available,
        'N_median': float(np.median(counts)),
        'N_p95': float(np.percentile(counts, 95)),
        'N_max': int(counts.max()),
        'frac_saturating': float((counts >= n_available).mean()),
        'frac_above_90pct': float((counts > 0.9*n_available).mean()),
        })

    # A cell whose volume is f times larger has xnn scaled by f**(-2/3), so every calculated q2
    # shrinks by that factor and more of them fall below the same observed cut-off.
    for factor in VOLUME_FACTORS:
        scaled = (q2_ref*factor**(-2/3) < cutoff[:, None]).sum(axis=1)
        rows_headroom.append({
            'bravais_lattice': bravais_lattice,
            'volume_factor': factor,
            'N_median': float(np.median(scaled)),
            'N_p95': float(np.percentile(scaled, 95)),
            'frac_saturating': float((scaled >= n_available).mean()),
            })
    return counts


def audit_b(rows_b, bravais_lattice, params, hkl_ref, counts):
    """N20's contents: equivalents, absences, and the (0,0,0) sentinel."""
    has_sentinel = bool(np.all(hkl_ref[-1] == 0))
    real_ref = hkl_ref[:-1] if has_sentinel else hkl_ref

    # Two distinct hkl that give the same row of the hkl matrix produce the same calculated q2 for
    # every cell in the lattice system -- they are symmetry equivalents, and counting both would
    # double-count one calculated line.
    hkl_matrix = get_hkl_matrix(real_ref, params['lattice_system'])
    n_distinct = len(np.unique(hkl_matrix, axis=0))

    allowed = centring_allowed(real_ref.astype(int), bravais_lattice)

    # get_spacegroup_hkl_ref must preserve both properties per extinction group, since the
    # spacegroup pass in Candidates.py re-scores M20 against those lists.
    worst_duplicate_frac = 0.0
    n_spacegroups = 0
    try:
        by_spacegroup = get_spacegroup_hkl_ref(real_ref, bravais_lattice=bravais_lattice)
        n_spacegroups = len(by_spacegroup)
        for reference in by_spacegroup.values():
            if len(reference) == 0:
                continue
            matrix = get_hkl_matrix(reference, params['lattice_system'])
            duplicate_frac = 1 - len(np.unique(matrix, axis=0))/len(matrix)
            worst_duplicate_frac = max(worst_duplicate_frac, duplicate_frac)
    except Exception as error:  # gemmi may not cover every extinction group
        print(f'  {bravais_lattice}: get_spacegroup_hkl_ref raised {error!r}')

    rows_b.append({
        'bravais_lattice': bravais_lattice,
        'n_ref': len(hkl_ref),
        'has_000_sentinel': has_sentinel,
        'frac_duplicate_q2': float(1 - n_distinct/len(real_ref)),
        'frac_centring_absent': float((~allowed).mean()),
        'n_spacegroups': n_spacegroups,
        'worst_sg_duplicate_frac': float(worst_duplicate_frac),
        # The sentinel adds exactly one to N, so it deflates M20 by N/(N+1).
        'sentinel_pct_of_N': float(np.median(100/(counts + 1))) if has_sentinel else 0.0,
        })


def audit_c(rows_c, bravais_lattice, params, hkl_ref, entries):
    """The cut-off convention: calculated-last vs calculated-max vs observed.

    Reported as ratios of epsilon = Q_N/(2N), never as M20 itself. These entries are unaugmented,
    so no measurement error has been added and the mean discrepancy is at float precision; M20
    would come out around 1e14 and mean nothing. The discrepancy is common to all three
    conventions and cancels exactly in the ratio, so the ratio is the whole comparison and needs
    no error model -- which also keeps the audit sigma-free (PLAN 2.5).
    """
    n_peaks = params['n_peaks']
    lattice_system = params['lattice_system']
    real_ref = hkl_ref[:-1] if np.all(hkl_ref[-1] == 0) else hkl_ref

    xnn = np.stack(entries['reindexed_xnn'].values).astype(float)[:, params['unit_cell_indices']]
    q2_obs = np.stack([np.asarray(row) for row in entries['q2'].values]).astype(float)
    hkl_true = np.stack([
        np.stack([np.asarray(entries[f'reindexed_{axis}'].values[index]) for axis in 'hkl'], axis=1)
        for index in range(len(entries))
        ]).astype(float)

    q2_calc = np.sum(get_hkl_matrix(hkl_true, lattice_system)*xnn[:, None, :], axis=2)
    q2_ref = xnn @ get_hkl_matrix(real_ref, lattice_system).T

    conventions = {
        'calc_last': q2_calc[:, n_peaks - 1],   # what get_M20 does today
        'calc_max': q2_calc.max(axis=1),
        'obs_last': q2_obs[:, n_peaks - 1],
        }
    epsilon, counts = {}, {}
    for name, cutoff in conventions.items():
        below = q2_ref < cutoff[:, None]
        counts[name] = below.sum(axis=1)
        # Q_N is the largest calculated reference line below the cut-off, as in get_M20.
        q_n = np.max(np.where(below, q2_ref, 0), axis=1)
        epsilon[name] = np.where(counts[name] > 0, q_n/(2*np.maximum(counts[name], 1)), np.nan)

    def ratio(name):
        with np.errstate(divide='ignore', invalid='ignore'):
            values = epsilon[name]/epsilon['calc_last']
        return values[np.isfinite(values)]

    # An assignment set that is not sorted in q2 puts some assigned lines above the cut-off. It
    # cannot happen for a correct cell, whose peaks are generated in increasing q2 order, so a zero
    # here says nothing about false candidates -- that measurement needs the S04 dump.
    non_monotonic = q2_calc[:, n_peaks - 1] < q2_calc.max(axis=1) - 1e-12
    row = {
        'bravais_lattice': bravais_lattice,
        'frac_last_not_max': float(non_monotonic.mean()),
        'N_median_calc_last': float(np.median(counts['calc_last'])),
        'N_median_obs_last': float(np.median(counts['obs_last'])),
        }
    for name in ('calc_max', 'obs_last'):
        values = ratio(name)
        row[f'M20_ratio_{name}_median'] = float(np.median(values))
        row[f'M20_ratio_{name}_p05'] = float(np.percentile(values, 5))
        row[f'M20_ratio_{name}_p95'] = float(np.percentile(values, 95))
        row[f'M20_ratio_{name}_frac_differs'] = float((np.abs(values - 1) > 0.01).mean())
    rows_c.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--n-max', type=int, default=20000,
                        help='entries sampled per Bravais lattice; triclinic has millions')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows_a, rows_headroom, rows_b, rows_c = [], [], [], []
    for bravais_lattice, tag in TAGS.items():
        params = read_params(tag)
        hkl_ref = np.load(os.path.join(
            BASE, 'mlindex', 'models', tag, 'data', f'hkl_ref_{bravais_lattice}.npy'))
        entries = load_entries(bravais_lattice, tag, args.n_max, args.seed)
        print(f'{bravais_lattice} ({tag}): {len(entries)} validation entries, '
              f'{len(hkl_ref)} reference hkl')
        if len(entries) == 0:
            continue
        counts = audit_a(rows_a, rows_headroom, bravais_lattice, params, hkl_ref, entries)
        audit_b(rows_b, bravais_lattice, params, hkl_ref, counts)
        audit_c(rows_c, bravais_lattice, params, hkl_ref, entries)

    frames = {
        'S01_audit_a_truncation.csv': pd.DataFrame(rows_a),
        'S01_audit_a_volume_headroom.csv': pd.DataFrame(rows_headroom),
        'S01_audit_b_n20_contents.csv': pd.DataFrame(rows_b),
        'S01_audit_c_cutoff.csv': pd.DataFrame(rows_c),
        }
    for name, frame in frames.items():
        frame.to_csv(os.path.join(args.out, name), index=False)
        print(f'\n=== {name}')
        print(frame.to_string(index=False))
    print(f'\nwrote {len(frames)} tables to {args.out}')


if __name__ == '__main__':
    main()
