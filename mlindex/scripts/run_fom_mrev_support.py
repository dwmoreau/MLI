"""C2-Q-012 -- is the `M_rev` blow-up a numerical artefact or a legitimate extreme, and who eats it.

THE QUESTION. C2-F-038 measured one candidate carrying `M_rev` = 4.1e11 against a 99.99th
percentile of 11.6, and stopped there: it winsorised its own means and said the tail was not
explained. `M_sym` = `M_tilde` x `M_rev` is the leading classical merit (INHERITED section 1), so
every step from S09 on inherits whatever this turns out to be. Nothing anywhere bounds the value.

WHAT THIS SCRIPT ESTABLISHES, in four stages:

    mechanism   reproduce one blow-up from the pool that measured it, and show what it is made of
    support     M_rev against the size of its own counting window, and against the noise level,
                over all 69 876 033 candidates
    blast       how far one blow-up reaches once an entry-relative feature is computed from it
    floor       what a support floor voids, on exact N_cal over a random sample of pools

THE ANSWER, in one line: M_rev's denominator is a mean over the reference lines in [q_I, q_N],
and when that window holds no more lines than the cell has free parameters the refinement
interpolates them exactly, so the denominator is zero by construction. It is an artefact of a
saturated fit. See `FigureOfMerits.get_M_rev_sym` for the implementation note and C2-F-059 for
the numbers.

POOL. S03's threshold-0 arms (`mlindex/data/fom_symmetry_counts/`), because they are uncensored
and they already carry `M_rev` at two points plus `n_ref_in_range` -- PROTOCOL section 3 rule 8 in
its happy case, the expensive quantity persisted beside the data. `error0_cont0` is not among
their bundles, so rule 11 is not in play here and the tail is not that trap.

    python mlindex/scripts/run_fom_mrev_support.py --stage mechanism
    python mlindex/scripts/run_fom_mrev_support.py --stage support
    python mlindex/scripts/run_fom_mrev_support.py --stage blast
    python mlindex/scripts/run_fom_mrev_support.py --stage floor
    python mlindex/scripts/run_fom_mrev_support.py --stage figure

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training.FomBenchmark import fast_assign
from mlindex.scripts.run_fom_prune_criterion import ARMS, load_entries, load_hkl_ref
from mlindex.scripts.run_fom_symmetry_count import MERIT_COLUMNS, SOURCE_COLUMNS, shard_pairs
from mlindex.utilities import FigureOfMerits as fom
from mlindex.utilities.Q2Calculator import Q2Calculator

COUNT_ROOT = os.path.join('mlindex', 'data', 'fom_symmetry_counts')
ARTIFACT_DIR = os.path.join('docs', 'fom_campaign2', 'artifacts')

# What counts as a blow-up. Not a chosen threshold so much as a reading of the distribution: the
# 99.99th percentile of M_rev on this pool is 11.6, so 1e3 is two orders of magnitude clear of
# anything the merit does in normal use and there is nothing between 626.7 and 2.5e4.
BLOWUP = 1e3

# Free cell parameters per Bravais lattice -- the length of `xnn`. The saturation hypothesis is
# that the blow-up lives where the counting window holds no more reference lines than this.
FREE_PARAMETERS = {'cF': 1, 'cI': 1, 'cP': 1, 'tI': 2, 'tP': 2, 'hP': 2, 'hR': 2,
                   'oC': 3, 'oF': 3, 'oI': 3, 'oP': 3, 'mC': 4, 'mP': 4, 'aP': 6}

# The row C2-Q-012 was opened over, and the one the regression fixture is cut from. Named rather
# than rediscovered so the mechanism stage always reports on the same candidate.
WITNESS = ('JAVDOA', 'oI', 'general')


# The pool is 69.9 M rows and the laptop has 16 GB, so nothing here concatenates it. The numeric
# columns alone are ~1.8 GB and fit; the three string columns do not, and are read only for the
# handful of rows a stage actually needs them for.
NUMERIC_COLUMNS = ('is_correct', 'M_rev_B', 'M_rev_C', 'n_ref_in_range')
KEY_COLUMNS = ('entry_id', 'bravais_lattice', 'condition_bundle')


def shards():
    """(arm, path) for every count shard of both threshold-0 arms."""
    found = [(arm, path) for arm in ('general', 'hard')
             for path in sorted(glob.glob(os.path.join(COUNT_ROOT, arm, '*.parquet')))]
    if not found:
        raise SystemExit(f'no count shards under {COUNT_ROOT}; run run_fom_symmetry_count.py '
                         '--stage counts first')
    return found


def load_numeric():
    """Every candidate of both arms, numeric columns plus a lattice code. No strings.

    Returns (value, window, correct, lattice, arm_is_hard), each aligned and pool-length.
    `value` is the larger of the two points: a cell can carry a modest M_rev against the full
    reference list and a catastrophic one against the narrowed list, so reading only B understates
    the tail -- the same reason `run_fom_symmetry_count.run_magnitude` caps both ends.
    """
    value, window, correct, lattice, hard = [], [], [], [], []
    for arm, path in shards():
        frame = pd.read_parquet(path, columns=list(NUMERIC_COLUMNS) + ['bravais_lattice'])
        value.append(np.maximum(frame['M_rev_B'].to_numpy(dtype=np.float64),
                                frame['M_rev_C'].to_numpy(dtype=np.float64)))
        window.append(frame['n_ref_in_range'].to_numpy(dtype=np.int32))
        correct.append(frame['is_correct'].to_numpy(dtype=bool))
        lattice.append(frame['bravais_lattice'].map(FREE_PARAMETERS).to_numpy(dtype=np.int8))
        hard.append(np.full(frame.shape[0], arm == 'hard', dtype=bool))
    return (np.concatenate(value), np.concatenate(window), np.concatenate(correct),
            np.concatenate(lattice), np.concatenate(hard))


def load_tail(threshold=BLOWUP):
    """The blow-up rows, with their keys. Read shard by shard and filtered before anything is kept."""
    kept = []
    for arm, path in shards():
        frame = pd.read_parquet(path, columns=list(KEY_COLUMNS) + list(NUMERIC_COLUMNS)
                                + ['M20_B'])
        value = np.maximum(frame['M_rev_B'].to_numpy(), frame['M_rev_C'].to_numpy())
        hit = value > threshold
        if hit.any():
            part = frame.loc[hit].copy()
            part['arm'] = arm
            part['M_rev'] = value[hit]
            kept.append(part)
    if not kept:
        return pd.DataFrame(columns=list(KEY_COLUMNS) + ['arm', 'M_rev'])
    return pd.concat(kept, ignore_index=True).sort_values('M_rev', ascending=False)


# ---------------------------------------------------------------------------------------------
# stage: mechanism -- one blow-up, reproduced from its own pool and taken apart
# ---------------------------------------------------------------------------------------------

def run_mechanism(args):
    """Reproduce the witness candidate bit for bit, and print what its denominator is made of.

    This is the stage that answers artefact-or-extreme, and it answers it by observation rather
    than by inference: the recomputed `M_rev` must equal the stored one exactly before anything it
    prints about the mechanism is worth reading.
    """
    entry_id, bravais_lattice, arm = WITNESS
    found = None
    for merit_path, source_path, bundle in shard_pairs(arm):
        frame = pd.read_parquet(merit_path, columns=list(MERIT_COLUMNS))
        hit = ((frame['entry_id'] == entry_id) & (frame['bravais_lattice'] == bravais_lattice)
               & (frame['M_rev_B'] > BLOWUP)).to_numpy()
        if hit.any():
            source = pd.read_parquet(source_path, columns=list(SOURCE_COLUMNS))
            position = int(np.flatnonzero(hit)[0])
            found = (frame.iloc[position], source.iloc[position], bundle)
            break
    if found is None:
        raise SystemExit(f'{entry_id}/{bravais_lattice} carries no blow-up in the {arm} arm')
    row, source_row, bundle = found

    xnn = np.asarray(source_row['xnn'], dtype=np.float64)[np.newaxis, :]
    entries = load_entries(os.path.join(ARMS[arm]['root'], bundle)).set_index('entry_id')
    q2_obs = np.asarray(entries.loc[entry_id, 'q2_obs'],
                        dtype=np.float64)[:int(row['n_peaks'])]
    hkl_ref = load_hkl_ref(row['lattice_system'], row['bravais_lattice'])
    calculator = Q2Calculator(lattice_system=row['lattice_system'], hkl=hkl_ref,
                              tensorflow=False, representation='xnn')
    q2_ref_calc = np.ascontiguousarray(calculator.get_q2(xnn))
    q2_calc = np.take_along_axis(q2_ref_calc, fast_assign(q2_obs, q2_ref_calc), axis=1)

    M_tilde, M_rev, M_sym = fom.get_M_rev_sym(q2_obs, q2_calc, q2_ref_calc)
    M20 = fom.get_M20(q2_obs, q2_calc, q2_ref_calc.copy())
    q_min, in_range, counts, q_n, scored = fom._reversed_line_terms(
        q2_obs, q2_calc[:, -1], q2_ref_calc)
    window = np.flatnonzero(in_range[0])

    if M_rev[0] != row['M_rev_B']:
        raise SystemExit(f'recomputed {M_rev[0]!r} against stored {row["M_rev_B"]!r}: the '
                         'mechanism below would be describing a different candidate')

    print(f'=== {entry_id} / {bravais_lattice} / {bundle}, candidate {row["candidate_id"]}')
    print(f'    stored M_rev_B  {float(row["M_rev_B"])!r}')
    print(f'    recomputed      {float(M_rev[0])!r}   (exact)')
    print(f'    is_correct      {bool(row["is_correct"])}')
    print()
    print(f'    M20     {M20[0]:.6f}   <- correctly says the cell is bad')
    print(f'    M_tilde {M_tilde[0]:.6f}   <- unharmed: its denominator is over all '
          f'{len(q2_obs)} assigned lines')
    print(f'    M_rev   {M_rev[0]:.6g}')
    print(f'    M_sym   {M_sym[0]:.6g}   <- the product, so the blow-up reaches the leading merit')
    print()
    print(f'    reference list              {q2_ref_calc.shape[1]} lines')
    print(f'    free cell parameters        {xnn.shape[1]}')
    print(f'    N_cal, lines in [q_I, q_N]  {counts[0]}   <- the whole of the denominator')
    print(f'    distinct calculated lines assigned to the {len(q2_obs)} peaks: '
          f'{len(np.unique(q2_calc[0]))}')
    print()
    print('    the in-range reference lines, and each one distance to the nearest observed peak:')
    for line, distance in zip(q2_ref_calc[0, window], scored[0, window]):
        exact = ' <- bit-identical to an observed peak' if distance == 0.0 else ''
        print(f'      {float(line)!r:<24}  {float(distance)!r}{exact}')
    reversed_sum = scored[0].sum()
    epsilon_reversed = (q2_obs[-1] - q2_obs[0])/(2*len(q2_obs))
    print()
    print(f'    sum of those distances       {float(reversed_sum)!r}')
    print(f'    / N_cal = discrepancy_rev    {float(reversed_sum/counts[0])!r}')
    print(f'    epsilon_reversed             {float(epsilon_reversed)!r}  <- fixed, does NOT shrink with N_cal')
    print(f'    ratio                        {epsilon_reversed/(reversed_sum/counts[0]):.6g}')
    print()
    print('    The cell has three free parameters and its window holds three lines, so the')
    print('    refinement interpolates them exactly. The denominator is zero by construction,')
    print('    not by luck, and the value is not a measurement of anything.')


# ---------------------------------------------------------------------------------------------
# stage: support -- M_rev against the size of its own counting window
# ---------------------------------------------------------------------------------------------

def run_support(args):
    """M_rev binned by the size of its own counting window, over the whole threshold-0 pool.

    `n_ref_in_range` is the reference lines at or below the cut-off, which is an upper bound on
    N_cal -- N_cal additionally requires the line to sit at or above q_I. It is used here because
    it is already on disk for all 69.9 M rows; the `floor` stage computes exact N_cal on a sample
    and reports the gap between the two, so the bound is quantified rather than assumed.
    """
    value, window, correct, free, _ = load_numeric()

    rows = []
    for low, high, label in ((0, 2, '0-2'), (3, 4, '3-4'), (5, 9, '5-9'), (10, 19, '10-19'),
                             (20, 49, '20-49'), (50, 99, '50-99'), (100, 10**9, '100+')):
        keep = (window >= low) & (window <= high)
        if not keep.any():
            continue
        rows.append({
            'n_ref_in_range': label, 'n': int(keep.sum()),
            'fraction_of_pool': float(keep.mean()),
            'M_rev_median': float(np.median(value[keep])),
            'M_rev_p99_9': float(np.percentile(value[keep], 99.9)),
            'M_rev_max': float(value[keep].max()),
            'n_above_1e3': int((value[keep] > BLOWUP).sum()),
            'fraction_correct': float(correct[keep].mean()),
            })
    _write(pd.DataFrame(rows), 'INTERIM_mrev_support_by_window.csv', args)

    # The same cut against the saturation hypothesis: window size less the free cell parameters.
    # A candidate at or below zero has a window its own refinement can interpolate exactly.
    slack = window - free
    rows = []
    for low, high, label in ((-10**9, 0, '<=0'), (1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'),
                             (5, 9, '5-9'), (10, 10**9, '>=10')):
        keep = (slack >= low) & (slack <= high)
        if not keep.any():
            continue
        rows.append({
            'window_less_free_parameters': label, 'n': int(keep.sum()),
            'fraction_of_pool': float(keep.mean()),
            'M_rev_median': float(np.median(value[keep])),
            'M_rev_max': float(value[keep].max()),
            'n_above_1e3': int((value[keep] > BLOWUP).sum()),
            'n_correct': int(correct[keep].sum()),
            })
    _write(pd.DataFrame(rows), 'INTERIM_mrev_saturation.csv', args)

    # Per bundle, because the first question anyone asks is whether this is the zero-error trap
    # (PROTOCOL section 3 rule 11) wearing a different hat. It is not, and the rate rising with
    # error and contamination is what says so: the fit ABSORBS measurement error rather than being
    # caught out by it, because a window with no more lines than the cell has parameters is exactly
    # determined. Noise is not a defence.
    bundles = []
    for arm, path in shards():
        frame = pd.read_parquet(path, columns=['condition_bundle', 'M_rev_B', 'M_rev_C'])
        local = np.maximum(frame['M_rev_B'].to_numpy(), frame['M_rev_C'].to_numpy())
        frame = frame.assign(blowup=local > BLOWUP)
        bundles.append(frame.groupby('condition_bundle')['blowup'].agg(['size', 'sum']))
    per_bundle = pd.concat(bundles).groupby(level=0).sum()
    per_bundle = per_bundle.rename(columns={'size': 'n', 'sum': 'n_above_1e3'})
    per_bundle['per_million'] = per_bundle['n_above_1e3']/per_bundle['n']*1e6
    _write(per_bundle.sort_values('per_million', ascending=False).reset_index(),
           'INTERIM_mrev_by_bundle.csv', args)

    blowup = value > BLOWUP
    n_pool, n_blowup = len(value), int(blowup.sum())
    windows = sorted(set(window[blowup].tolist()))
    del value, window, correct, free, slack

    tail = load_tail()
    _write(tail[['arm', 'entry_id', 'bravais_lattice', 'condition_bundle', 'is_correct',
                 'M20_B', 'M_rev_B', 'M_rev_C', 'n_ref_in_range']],
           'INTERIM_mrev_tail.csv', args)
    print(f'\n{n_pool:,} candidates; {n_blowup} above {BLOWUP:g}, '
          f'{int(tail["is_correct"].sum())} of them correct')
    print(f'window sizes among the blow-ups: {windows}')


# ---------------------------------------------------------------------------------------------
# stage: blast -- how far one blow-up reaches
# ---------------------------------------------------------------------------------------------

def run_blast(args):
    """The rows an entry-relative feature carries the blow-up to.

    `FomCombiner.add_context` computes `ctx_M_sym_gap_to_best` as `value - max(pool)` and
    `ctx_M_sym_z` as `(value - median)/std`, over the pooled cross-lattice candidates of one
    (entry, condition) -- which is what `run.py` ranks. Both read a statistic of the whole pool, so
    one blow-up does not corrupt one row, it corrupts every row that shares its pool. The unit of
    exposure is the pool, not the candidate, which is the same reason PROTOCOL section 8 bootstraps
    over entries and not over candidates.

    Two passes, because the pool does not fit in memory with its key columns attached: the first
    collects the affected (entry, condition) pairs, which are few, and the second counts what
    shares a pool with one.
    """
    tail = load_tail()
    affected = set(zip(tail['entry_id'], tail['condition_bundle']))

    counts = {}
    for arm, path in shards():
        frame = pd.read_parquet(path, columns=['entry_id', 'condition_bundle', 'is_correct'])
        pools = pd.Series(list(zip(frame['entry_id'], frame['condition_bundle'])))
        reached = pools.isin(affected).to_numpy()
        record = counts.setdefault(arm, {'n_rows': 0, 'pools': set(), 'reached_pools': set(),
                                         'n_rows_reached': 0, 'n_correct_reached': 0,
                                         'entries_reached': set()})
        record['n_rows'] += frame.shape[0]
        record['pools'].update(pools)
        record['reached_pools'].update(pools[reached])
        record['n_rows_reached'] += int(reached.sum())
        record['n_correct_reached'] += int(frame['is_correct'].to_numpy()[reached].sum())
        record['entries_reached'].update(frame['entry_id'].to_numpy()[reached])

    rows = []
    for arm in sorted(counts):
        record = counts[arm]
        n_pools = len(record['pools'])
        rows.append({
            'arm': arm, 'n_rows': record['n_rows'], 'n_pools': n_pools,
            'n_blowup_rows': int((tail['arm'] == arm).sum()),
            'n_pools_affected': len(record['reached_pools']),
            'fraction_pools_affected': len(record['reached_pools'])/max(n_pools, 1),
            'n_rows_reached': record['n_rows_reached'],
            'fraction_rows_reached': record['n_rows_reached']/max(record['n_rows'], 1),
            'n_correct_reached': record['n_correct_reached'],
            'n_entries_reached': len(record['entries_reached']),
            })
    table = pd.DataFrame(rows)
    _write(table, 'INTERIM_mrev_blast_radius.csv', args)
    print()
    print(table.to_string(index=False))


# ---------------------------------------------------------------------------------------------
# stage: floor -- exact N_cal, and what a support floor costs
# ---------------------------------------------------------------------------------------------

def run_floor(args):
    """Exact N_cal on a random sample of (entry, condition) pools, and the cost of each floor.

    The window has to be recomputed to get N_cal exactly, so this runs on a sample rather than on
    all 69.9 M rows. What it is for is the cost side of the decision -- how many rows a floor
    voids and, the number that actually matters, how many CORRECT candidates it voids. The benefit
    side comes from `support`, which is measured on everything.
    """
    rng = np.random.default_rng(args.sample_seed)
    collected = []
    for arm in ('general', 'hard'):
        pairs = shard_pairs(arm)
        for index in rng.choice(len(pairs), size=min(args.n_shards, len(pairs)), replace=False):
            merit_path, source_path, bundle = pairs[index]
            frame = pd.read_parquet(merit_path, columns=list(MERIT_COLUMNS))
            source = pd.read_parquet(source_path, columns=list(SOURCE_COLUMNS))
            entries = load_entries(os.path.join(ARMS[arm]['root'], bundle)).set_index('entry_id')
            entry_id = str(rng.choice(pd.unique(frame['entry_id'])))
            rows = np.flatnonzero(frame['entry_id'].to_numpy() == entry_id)
            part = frame.iloc[rows]
            q2_obs_full = np.asarray(entries.loc[entry_id, 'q2_obs'], dtype=np.float64)
            for (bravais_lattice, lattice_system), _ in part.groupby(
                    ['bravais_lattice', 'lattice_system'], sort=False):
                local = np.flatnonzero(part['bravais_lattice'].to_numpy() == bravais_lattice)
                hkl_ref = load_hkl_ref(lattice_system, bravais_lattice)
                calculator = Q2Calculator(lattice_system=lattice_system, hkl=hkl_ref,
                                          tensorflow=False, representation='xnn')
                q2_obs = q2_obs_full[:int(part.iloc[local[0]]['n_peaks'])]
                xnn = np.stack([np.asarray(v, dtype=np.float64)
                                for v in source['xnn'].to_numpy()[rows[local]]])
                for start in range(0, xnn.shape[0], args.block):
                    block = slice(start, min(start + args.block, xnn.shape[0]))
                    q2_ref_calc = np.ascontiguousarray(calculator.get_q2(xnn[block]))
                    q2_calc = np.take_along_axis(
                        q2_ref_calc, fast_assign(q2_obs, q2_ref_calc), axis=1)
                    _, _, counts, _, _ = fom._reversed_line_terms(
                        q2_obs, q2_calc[:, -1], q2_ref_calc)
                    piece = part.iloc[local[block]]
                    collected.append(pd.DataFrame({
                        'arm': arm, 'n_cal': counts,
                        # `n_ref_in_range` as the count shards define it: strictly below the
                        # cut-off, with no lower bound. Recomputed rather than joined because the
                        # merit shards do not carry it, and the point of having it here is to
                        # measure how far it overstates N_cal, which is the bound the `support`
                        # stage carries.
                        'n_ref_in_range': (q2_ref_calc < q2_calc[:, -1:]).sum(axis=1),
                        'is_correct': piece['is_correct'].to_numpy(),
                        'M_rev': np.maximum(piece['M_rev_B'].to_numpy(),
                                            piece['M_rev_C'].to_numpy())}))
            print(f'  {arm} {bundle} {entry_id}: '
                  f'{sum(len(part) for part in collected):,} rows', flush=True)

    sample = pd.concat(collected, ignore_index=True)
    n_cal = sample['n_cal'].to_numpy()
    correct = sample['is_correct'].to_numpy()
    value = sample['M_rev'].to_numpy()
    rows = []
    for floor in (0, 3, 5, 8, 10, 15, 20):
        void = n_cal < floor
        rows.append({
            'min_n_cal': floor,
            'n_voided': int(void.sum()), 'fraction_voided': float(void.mean()),
            'n_correct_voided': int(correct[void].sum()),
            'fraction_of_correct_voided': float(correct[void].sum()/max(correct.sum(), 1)),
            'M_rev_max_kept': float(value[~void].max()) if (~void).any() else np.nan,
            })
    table = pd.DataFrame(rows)
    _write(table, 'INTERIM_mrev_floor_cost.csv', args)
    print(f'\nsampled {len(sample):,} candidates; {int(correct.sum()):,} correct')
    print(f'exact N_cal: median {np.median(n_cal):.0f}, and n_ref_in_range exceeds it by a '
          f'median of {np.median(sample["n_ref_in_range"].to_numpy() - n_cal):.0f}')
    print(table.to_string(index=False))


def _write(table, name, args):
    destination = os.path.join(args.artifact_dir, name)
    table.to_csv(destination, index=False)
    print(f'wrote {destination}  ({table.shape[0]} rows)')


def run_figure(args):
    """M_rev against its window size, which is the whole finding in one panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    value, window, _, _, _ = load_numeric()

    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    thinned = np.random.default_rng(0).choice(
        len(value), size=min(400_000, len(value)), replace=False)
    axes.scatter(window[thinned], np.maximum(value[thinned], 1e-3), s=1, alpha=0.06,
                 color='#4C72B0', linewidths=0, rasterized=True)
    blowup = value > BLOWUP
    axes.scatter(window[blowup], value[blowup], s=26, color='#D55E00', zorder=3,
                 label=f'M_rev > {BLOWUP:g}  ({int(blowup.sum())} rows, none correct)')
    axes.axvline(fom.M_REV_MIN_N_CAL, color='#333333', linestyle='--', linewidth=1.1,
                 label=f'support floor, min_n_cal = {fom.M_REV_MIN_N_CAL}\n'
                       '(applied to N_cal, which this axis overstates by a median of 1)')
    axes.set_yscale('log')
    axes.set_xscale('log')
    axes.set_xlabel('reference lines in the counting window  (n_ref_in_range)')
    axes.set_ylabel('M_rev')
    axes.set_title('M_rev blows up only where its denominator has almost no support\n'
                   f'{len(value):,} candidates, S03 threshold-0 arms', fontsize=10)
    axes.legend(loc='upper right', fontsize=8, framealpha=0.95)
    axes.grid(alpha=0.25, which='both', linewidth=0.4)
    figure.tight_layout()
    destination = os.path.join(args.artifact_dir, 'INTERIM_mrev_support.png')
    figure.savefig(destination, dpi=200)
    print(f'wrote {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stage', required=True,
                        choices=('mechanism', 'support', 'blast', 'floor', 'figure'))
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    parser.add_argument('--sample-seed', type=int, default=20260827,
                        help='Seed for the pool sample the floor stage measures on')
    parser.add_argument('--n-shards', type=int, default=4,
                        help='Shards sampled per arm by the floor stage')
    parser.add_argument('--block', type=int, default=2000)
    args = parser.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    {'mechanism': run_mechanism, 'support': run_support, 'blast': run_blast,
     'floor': run_floor, 'figure': run_figure}[args.stage](args)


if __name__ == '__main__':
    main()
