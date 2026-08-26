"""S04 Phase 2 -- what the symmetry prior is worth, measured by retrained paired arms.

THE QUESTION THIS EXISTS TO SETTLE. Campaign 1 concluded the extinction group is nearly free to
drop, on a retrained arm that cost **0.004 pp of top-10**. That number is real and it is one metric
of five: the same two arms in the same artefact differ by **2.23 pp of operating point** and by
**41 % of the hard-stratum arm** (C2-F-039). And the arm that produced it, `drop_structural`,
removes **sixteen** features at once -- `spacegroup`, `bravais_lattice`, `final_rank`, `n_entering`,
`log_volume` and eleven more -- so the cost belongs to the family and not to the symmetry prior.
Nobody has ever measured `spacegroup` on its own. That is C2-Q-013, and this script answers it.

It also settles S04's original question, which Phase 1 could only answer indirectly: is the absence
COUNT a better encoding than the categorical? Phase 1 refuted the mechanism the count was supposed
to work by (C2-F-034) but never put it in a model.

SIX ARMS, ALL RETRAINED AND PAIRED -- never an importance table (PROTOCOL section 8):

    full             raw + structural + context, 46 features. The incumbent.
    drop_spacegroup  full minus the extinction group ALONE.            <- C2-Q-013
    drop_symmetry    full minus `spacegroup` and `bravais_lattice`.
    drop_structural  raw + context. Campaign 1's arm, 30 features.     <- harness validation
    counts           full minus `spacegroup`, plus the absence counts. <- DWMM's proposal
    delta            full minus `spacegroup`, plus the merit movement. <- what Phase 1 found

WHAT IS REPORTED, AND WHY MORE THAN ONE COLUMN. Top-10, operating point, and the hard stratum, on
every arm. Reading one column is precisely the error that produced C2-F-039.

POOL. Benchmark A, `fom-train` to fit and `fom-dev` to report. This is the pool campaign 1 used, and
using it is the point: C2-Q-013 is a question about their number. It is prune-censored (R1), so
every level here carries that bound -- but every arm carries it equally and the CONTRAST is what is
being measured. `fom-test` is sealed until S15 and is structurally unreachable here: the feature
matrix was never computed for it, and `combiner_frames` inner-joins on it.

    python mlindex/scripts/run_fom_symmetry_arms.py --stage features
    python mlindex/scripts/run_fom_symmetry_arms.py --stage arms

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomCombiner
from mlindex.model_training import FomMetrics
from mlindex.scripts.run_fom_prune_criterion import (
    MAX_BLOCK_ELEMENTS, load_hkl_ref, load_spacegroup_sets, merits_on_reference, commit_hash, _map,
    )
from mlindex.utilities.ExtinctionCounts import (
    LATTICE_SYSTEM, absent_in_range, build_group_masks, get_absence_counts,
    )
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.numba_functions import fast_assign

BENCHMARK_A = os.path.join('mlindex', 'data', 'fom_benchmark')
FEATURE_DIR = os.path.join('mlindex', 'data', 'fom_features')
SYMMETRY_DIR = os.path.join('mlindex', 'data', 'fom_symmetry_benchmark')
ARTIFACT_DIR = os.path.join('docs', 'fom_campaign2', 'artifacts')

BASE_GROUPS = ('raw', 'structural', 'context')

# (label, extra groups, dropped columns, one-line purpose)
ARMS = (
    ('full', (), (), 'the incumbent: raw + structural + context'),
    ('drop_spacegroup', (), ('spacegroup',), 'the extinction group alone -- C2-Q-013'),
    ('drop_symmetry', (), ('spacegroup', 'bravais_lattice'), 'both symmetry categoricals'),
    ('drop_structural', None, (), "campaign 1's 16-feature family drop -- validation"),
    ('counts', ('counts',), ('spacegroup',), "DWMM's absence-count encoding"),
    ('delta', ('delta',), ('spacegroup',), 'the merit movement Phase 1 found carries signal'),
    )

# Campaign 1's own values, from `../fom_campaign1/artifacts/S08_combiner_meta.json`. Reused rather
# than rechosen so these arms sit on its split, its subsample and its false-positive budget, and
# the only thing that differs between them and its table is the feature set.
N_NEGATIVES = 40
SEED = 12345
HOLDOUT_FRACTION = 0.2
MATCHED_FPR_BUDGET = 0.23131313676123635
MODEL_PARAMS = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63)

# The six campaign 1 fitted and reported on. `error0_cont0` is `FomMetrics.CONTROL_BUNDLES` -- the
# zero-measurement-error control -- and `evaluate` drops it from every metric, so including it here
# would fit on rows no number is read from.
#
# It is also numerically degenerate, which is worth stating once so nobody re-investigates it: with
# no error added, a correct cell's mean |q2_obs - q2_calc| is at machine epsilon, so M20 -- a ratio
# with that in the denominator -- runs to 1e12-1e14 and is ill-conditioned. Recomputing it
# reproduces the stored value to a few percent RELATIVE, which on those magnitudes is 1e12 absolute.
# That accounted for all 62 rows of an earlier gate failure, across ten lattices, and none of them
# involved the extinction group (the recomputed value equalled the generic one). Excluding the
# bundle removes the rows and the question together.
BUNDLES = ('error1_cont0', 'error2_cont0', 'error1_cont2', 'error1_cont1_drop6',
           'error1_cont1_drop10', 'error1_cont0_phase3')


# ---------------------------------------------------------------------------------------------
# stage: features -- the S04 columns, computed on Benchmark A
# ---------------------------------------------------------------------------------------------

SYMMETRY_COLUMNS = ('n_absent_extra', 'n_absent_extra_in_range', 'f_absent_extra',
                    'n_groups_searched', 'delta_M20', 'delta_M_rev')


def _features_worker(job):
    """The S04 columns for one (bundle, lattice) shard of Benchmark A.

    Point B is the cell against the full reference list, point C against the group the pipeline
    chose. `delta` is C - B, which is the merit movement the assignment bought -- and it is a valid
    subtraction because exactly one group per lattice removes no lines, so the full list IS the
    generic group's list (C2-F-034).

    `M20_C` is gated against the pool's own stored `M20`, which SCHEMA.md defines as the
    post-assignment value. That is the same gate S03 used and it is what proves the recompute is
    the pipeline's own arithmetic rather than a second implementation of it.
    """
    shard, bundle, bravais_lattice, keep_ids, out_dir = job
    columns = ['entry_id', 'bravais_lattice', 'candidate_id', 'xnn', 'spacegroup', 'n_peaks', 'M20']
    frame = pd.read_parquet(shard, columns=columns)
    frame = frame.loc[frame['entry_id'].isin(keep_ids)]
    if not frame.shape[0]:
        return None

    entries = pd.read_parquet(os.path.join(BENCHMARK_A, 'entries.parquet'),
                              columns=['entry_id', 'condition_bundle', 'q2_obs'])
    entries = entries.loc[entries['condition_bundle'] == bundle].set_index('entry_id')

    lattice_system = LATTICE_SYSTEM[bravais_lattice]
    hkl_ref = load_hkl_ref(lattice_system, bravais_lattice)
    masks = build_group_masks(hkl_ref, bravais_lattice)
    spacegroup_sets = load_spacegroup_sets(lattice_system, bravais_lattice)
    counts = get_absence_counts(bravais_lattice)

    n_rows = frame.shape[0]
    out = {name: np.full(n_rows, np.nan) for name in
           ('n_absent_extra_in_range', 'n_ref_in_range', 'M20_B', 'M20_C', 'M_rev_B', 'M_rev_C')}
    position = np.arange(n_rows)

    for entry_id, group in frame.groupby('entry_id', sort=False):
        n_peaks = int(group['n_peaks'].iloc[0])
        q2_obs = np.asarray(entries.loc[entry_id, 'q2_obs'], dtype=np.float64)[:n_peaks]
        rows = position[frame.index.get_indexer(group.index)]
        xnn = np.stack(group['xnn'].to_numpy()).astype(np.float64)
        spacegroups = group['spacegroup'].to_numpy()

        chunk = max(1, MAX_BLOCK_ELEMENTS // max(hkl_ref.shape[0], 1))
        calculator = Q2Calculator(lattice_system=lattice_system, hkl=hkl_ref, tensorflow=False,
                                  representation='xnn')
        for start in range(0, xnn.shape[0], chunk):
            stop = min(start + chunk, xnn.shape[0])
            block = rows[start:stop]
            q2_ref_calc = calculator.get_q2(xnn[start:stop])
            cutoff = np.take_along_axis(
                q2_ref_calc, fast_assign(q2_obs, q2_ref_calc), axis=1)[:, -1]
            for spacegroup in pd.unique(spacegroups[start:stop]):
                local = np.flatnonzero(spacegroups[start:stop] == spacegroup)
                dropped, in_range = absent_in_range(
                    q2_ref_calc[local], masks[spacegroup], cutoff[local])
                out['n_absent_extra_in_range'][block[local]] = dropped
                out['n_ref_in_range'][block[local]] = in_range
            # get_M20 mutates q2_ref_calc, so point B is computed on a copy and the original is
            # left intact for the counts above -- which were taken first for the same reason.
            values = merits_on_reference(q2_obs, q2_ref_calc.copy())
            out['M20_B'][block] = values['M20']
            out['M_rev_B'][block] = values['M_rev']

        for spacegroup in pd.unique(spacegroups):
            local = np.flatnonzero(spacegroups == spacegroup)
            lines = spacegroup_sets[spacegroup]
            narrow = Q2Calculator(lattice_system=lattice_system, hkl=lines, tensorflow=False,
                                  representation='xnn')
            step = max(1, MAX_BLOCK_ELEMENTS // max(lines.shape[0], 1))
            for start in range(0, local.size, step):
                piece = local[start:start + step]
                values = merits_on_reference(q2_obs, narrow.get_q2(xnn[piece]))
                out['M20_C'][rows[piece]] = values['M20']
                out['M_rev_C'][rows[piece]] = values['M_rev']

    result = frame[['entry_id', 'bravais_lattice', 'candidate_id']].copy()
    result['condition_bundle'] = bundle
    result['n_absent_extra'] = [counts[key] for key in frame['spacegroup']]
    result['n_groups_searched'] = len(counts)
    result['n_absent_extra_in_range'] = out['n_absent_extra_in_range'].astype(np.int64)
    result['f_absent_extra'] = np.where(out['n_ref_in_range'] > 0,
                                        out['n_absent_extra_in_range']
                                        / np.maximum(out['n_ref_in_range'], 1), np.nan)
    result['delta_M20'] = out['M20_C'] - out['M20_B']
    result['delta_M_rev'] = out['M_rev_C'] - out['M_rev_B']

    # The gate does not just count: it keeps the offending rows. A bare count says the recompute
    # disagrees somewhere and gives the next session nothing to work with, and this pool is
    # POST-deduplication -- the one place the spacegroup mis-attachment of C2-F-036 could actually
    # show up.
    stored = frame['M20'].to_numpy()
    recomputed = out['M20_C']
    differs = (~np.isclose(stored, recomputed, rtol=0, atol=1e-9)
               & ~(np.isnan(stored) & np.isnan(recomputed)))
    offenders = pd.DataFrame()
    if differs.any():
        offenders = frame.loc[differs, ['entry_id', 'bravais_lattice', 'candidate_id',
                                        'spacegroup', 'n_peaks']].copy()
        offenders['condition_bundle'] = bundle
        offenders['M20_stored'] = stored[differs]
        offenders['M20_recomputed'] = recomputed[differs]
        offenders['M20_generic'] = out['M20_B'][differs]
        offenders['n_absent_extra'] = [counts[key] for key in offenders['spacegroup']]
    return result, int(differs.sum()), bundle, offenders


def run_features(args):
    entries = pd.read_parquet(os.path.join(BENCHMARK_A, 'entries.parquet'),
                              columns=['entry_id', 'split', 'condition_bundle'])
    # fom-test is sealed until S15. Excluded here as well as by the feature matrix's own coverage,
    # so the seal does not depend on a join being inner.
    keep = set(entries.loc[entries['split'].isin(('fom-train', 'fom-dev')), 'entry_id'].unique())
    print(f'{len(keep):,} fom-train + fom-dev crystals; {len(BUNDLES)} bundles '
          f'(control bundle excluded); fom-test not read')

    os.makedirs(SYMMETRY_DIR, exist_ok=True)
    jobs = []
    for shard in sorted(Path(BENCHMARK_A).glob('candidates_*.parquet')):
        stem = shard.stem[len('candidates_'):]
        bundle, bravais_lattice = stem.rsplit('_', 1)
        if bundle not in BUNDLES:
            continue
        jobs.append((str(shard), bundle, bravais_lattice, keep, SYMMETRY_DIR))

    started = time.time()
    pieces, mismatches, differing_total, rows_total = {}, [], 0, 0
    for done, result in enumerate(_map(_features_worker, jobs, args.processes), start=1):
        if result is None:
            continue
        frame, differing, bundle, offenders = result
        pieces.setdefault(bundle, []).append(frame)
        if offenders.shape[0]:
            mismatches.append(offenders)
        differing_total += differing
        rows_total += frame.shape[0]
        print(f'  [{done}/{len(jobs)}] {bundle} {frame.shape[0]:>8,} rows '
              f'({time.time() - started:.0f} s)')

    for bundle, frames in pieces.items():
        pd.concat(frames, ignore_index=True).to_parquet(
            Path(SYMMETRY_DIR) / f'symmetry_{bundle}.parquet', index=False)

    print(f'\n{rows_total:,} rows over {len(pieces)} bundles -> {SYMMETRY_DIR}')
    print(f'GATE recomputed M20 at the chosen group vs the stored M20: {differing_total} '
          f'differing of {rows_total:,} ({differing_total/max(rows_total, 1):.2e})')
    if differing_total:
        raise AssertionError(
            f'the recompute is not the pipeline arithmetic on {differing_total} rows; '
            f'see {Path(SYMMETRY_DIR) / "gate_mismatches.parquet"}')
    if mismatches:
        table = pd.concat(mismatches, ignore_index=True)
        destination = Path(SYMMETRY_DIR) / 'gate_mismatches.parquet'
        table.to_parquet(destination, index=False)
        print(f'  wrote {destination}')
        print('  by lattice: '
              + ', '.join(f'{k} {v}' for k, v in
                          table['bravais_lattice'].value_counts().items()))
        print('  by bundle:  '
              + ', '.join(f'{k} {v}' for k, v in
                          table['condition_bundle'].value_counts().items()))
        gap = (table['M20_recomputed'] - table['M20_stored']).abs()
        print(f'  |recomputed - stored|: median {gap.median():.4g}, max {gap.max():.4g}')
        print(f'  rows where the recompute is LOWER than stored: '
              f"{int((table['M20_recomputed'] < table['M20_stored']).sum())} of {table.shape[0]}")
    with open(Path(SYMMETRY_DIR) / 'manifest.json', 'w', encoding='utf-8') as _f:
        json.dump({'commit': commit_hash(), 'rows': rows_total, 'bundles': sorted(pieces),
                   'differing_M20': differing_total, 'columns': list(SYMMETRY_COLUMNS)}, _f,
                  indent=2)


# ---------------------------------------------------------------------------------------------
# stage: arms -- fit each arm on fom-train, report every arm on fom-dev
# ---------------------------------------------------------------------------------------------

def arm_groups(extra):
    """`drop_structural` is the one arm that removes a whole group rather than named columns."""
    return ('raw', 'context') if extra is None else BASE_GROUPS + tuple(extra)


def split_ids(entries, split, holdout_fraction, seed):
    """Source entries of one split, divided into a fit part and a threshold-selection part.

    By SOURCE ENTRY, never by candidate (PROTOCOL section 3 rule 5): one crystal appears under
    seven condition bundles with correlated noise, so splitting rows would put near-duplicates of
    the same pattern on both sides. Same construction and same seed as campaign 1's
    `run_fom_combiner.split_ids`, so the arms here sit on its split rather than a new one.
    """
    ids = np.array(sorted(set(entries.loc[entries['split'] == split, 'entry_id'])))
    if not holdout_fraction:
        return set(ids), set()
    rng = np.random.default_rng(seed)
    held = rng.permutation(ids.size) < int(round(holdout_fraction*ids.size))
    return set(ids[~held]), set(ids[held])


def load_split(bundles, covariates, groups, keep, n_negatives=None, seed=SEED):
    frames = list(FomCombiner.combiner_frames(
        BENCHMARK_A, FEATURE_DIR, bundles, keep, covariates, scalers=(), groups=groups,
        symmetry_dir=SYMMETRY_DIR))
    if n_negatives is not None:
        frames = [FomCombiner.subsample_negatives(frame, n_negatives, seed) for frame in frames]
    return frames


def _headline(result):
    """The reported metrics, aggregate and hard stratum. Unweighted -- there is no other option."""
    out = {metric: float(result.metric(metric))
           for metric in ('top10', 'operating_point', 'operating_point_given_found',
                          'false_positive', 'reported', 'ceiling_rescorer')}
    hard = result.hard
    for name, column in (('hard_top10', 'top10'), ('hard_operating_point', 'operating_point'),
                         ('hard_op_given_found', 'operating_point_given_found'),
                         ('hard_ceiling', 'ceiling_rescorer')):
        out[name] = float(hard[column].iloc[0]) if hard is not None and hard.shape[0] else np.nan
    out['hard_n_entries'] = (int(hard['n_entries'].iloc[0])
                             if hard is not None and hard.shape[0] else 0)
    return out


def run_arms(args):
    # Two projections of the same table. `entry_covariates` needs the peak list; `evaluate` needs
    # the truth columns it derives the volume decile and the hard stratum from -- and those are
    # entry-level context, never candidate features (`FORBIDDEN_SUFFIX = '_true'` blocks them from
    # every design matrix, checked on each fit and each score).
    covariate_entries = pd.read_parquet(os.path.join(BENCHMARK_A, 'entries.parquet'),
                                        columns=list(FomCombiner.ENTRY_COLUMNS))
    entries = pd.read_parquet(os.path.join(BENCHMARK_A, 'entries.parquet'))
    entries = entries.loc[entries['condition_bundle'].isin(BUNDLES)]
    bundles = list(BUNDLES)
    covariates = FomCombiner.entry_covariates(covariate_entries)
    fit_ids, cal_ids = split_ids(covariate_entries, 'fom-train', HOLDOUT_FRACTION, SEED)
    dev_ids = set(covariate_entries.loc[covariate_entries['split'] == 'fom-dev', 'entry_id'])
    print(f'fit {len(fit_ids):,} crystals | threshold-selection {len(cal_ids):,} | '
          f'report {len(dev_ids):,} (fom-dev) | fom-test sealed, never loaded')

    # Every group any arm needs, assembled once. A frame carries columns a given arm does not use;
    # the design matrix is built from that arm's own name list, so the surplus costs memory only.
    every_group = BASE_GROUPS + ('counts', 'delta')
    print('assembling frames ...')
    started = time.time()
    fit_frames = load_split(bundles, covariates, every_group, fit_ids, n_negatives=N_NEGATIVES,
                            seed=args.fit_seed)
    cal_frames = load_split(bundles, covariates, every_group, cal_ids)
    dev_frames = load_split(bundles, covariates, every_group, dev_ids)
    print(f'  fit {sum(f.shape[0] for f in fit_frames):,} rows after negative subsampling '
          f'(<= {N_NEGATIVES} per entry-bundle)')
    print(f'  select {sum(f.shape[0] for f in cal_frames):,} | '
          f'report {sum(f.shape[0] for f in dev_frames):,} candidates '
          f'({time.time() - started:.0f} s)')

    results, rows = {}, []
    for label, extra, drop, purpose in ARMS:
        started = time.time()
        groups = arm_groups(extra)
        combiner = FomCombiner.FomCombiner.fit(fit_frames, groups=groups, scalers=(),
                                               objective='pointwise', seed=args.fit_seed,
                                               drop=drop, **MODEL_PARAMS)
        # The threshold is chosen on held-out `fom-train` and never on the split it is reported
        # on (PROTOCOL section 8). The budget is campaign 1's, so every arm here is cut at the
        # same false-positive rate its numbers were.
        # `evaluate` applies the score callable itself, so the frames go in raw -- the same call
        # campaign 1's `evaluate_score` makes.
        selection = FomMetrics.evaluate(
            cal_frames, score=combiner.score,
            score_columns=list(combiner.score_columns),
            higher_is_better=True, threshold=0.0, entries=entries, split='fom-train',
            n_bootstrap=0)
        choice = FomMetrics.select_threshold(selection, objective='operating_point',
                                             max_false_positive_rate=MATCHED_FPR_BUDGET)
        result = FomMetrics.evaluate(
            dev_frames, score=combiner.score,
            score_columns=list(combiner.score_columns),
            higher_is_better=True, threshold=float(choice.threshold), entries=entries,
            split='fom-dev', n_bootstrap=args.n_bootstrap, seed=SEED)
        results[label] = result

        summary = _headline(result)
        summary.update(arm=label, n_features=combiner.meta['n_features'],
                       dropped=','.join(drop) or '-', groups='+'.join(groups),
                       threshold=float(choice.threshold), seconds=round(time.time() - started, 1),
                       purpose=purpose)
        rows.append(summary)
        print(f"  {label:16s} n={summary['n_features']:>3d}  top10 {summary['top10']:.6f}  "
              f"op {summary['operating_point']:.6f}  hard_op {summary['hard_operating_point']:.6f}"
              f"  hard_opgf {summary['hard_op_given_found']:.6f}  ({summary['seconds']:.0f} s)")

    table = pd.DataFrame(rows)[
        ['arm', 'groups', 'dropped', 'n_features', 'threshold', 'top10', 'operating_point',
         'operating_point_given_found', 'false_positive', 'hard_top10', 'hard_operating_point',
         'hard_op_given_found', 'hard_n_entries', 'seconds', 'purpose']]
    suffix = '' if args.fit_seed == SEED else f'_seed{args.fit_seed}'
    table['fit_seed'] = args.fit_seed
    table.to_csv(os.path.join(args.artifact_dir, f'S04_symmetry_arms{suffix}.csv'), index=False)
    _print_contrasts(table, results, args.artifact_dir, suffix, args.fit_seed)


def _print_contrasts(table, results, artifact_dir, suffix='', fit_seed=SEED):
    """Every arm against `full`, on all four metrics, paired by McNemar over the same crystals."""
    indexed = table.set_index('arm')
    base = indexed.loc['full']
    rows = []
    for label in table['arm']:
        if label == 'full':
            continue
        arm = indexed.loc[label]
        record = {'arm': label, 'vs': 'full', 'dropped': arm['dropped']}
        for metric in ('top10', 'operating_point', 'hard_operating_point',
                       'hard_op_given_found'):
            record[f'delta_{metric}_pp'] = 100.0*(arm[metric] - base[metric])
        # Paired over the same crystals, `full` against the ablation, on the aggregate and again
        # on the hard stratum. `n_a_only` is where `full` wins.
        # `tag`, not `suffix`: `suffix` is this function's own parameter and naming the loop
        # variable the same rebinds it to '_hard', so every seeded run wrote its contrasts over
        # one file.
        for metric, subset, tag in (('operating_point', None, ''), ('top10', None, ''),
                                    ('operating_point', 'hard', '_hard')):
            key = f'mcnemar_{metric}{tag}'
            try:
                test = FomMetrics.mcnemar(results['full'], results[label], metric=metric,
                                          subset=subset)
                record[f'{key}_p'] = float(test['p_value'])
                record[f'{key}_full_only'] = int(test['n_a_only'])
                record[f'{key}_arm_only'] = int(test['n_b_only'])
                record[f'{key}_method'] = str(test['method'])
            except Exception as error:                                        # noqa: BLE001
                record[f'{key}_p'] = np.nan
                record[f'{key}_full_only'] = record[f'{key}_arm_only'] = -1
                record[f'{key}_method'] = f'unavailable: {str(error)[:60]}'
        rows.append(record)
    contrasts = pd.DataFrame(rows)
    contrasts['fit_seed'] = fit_seed
    contrasts.to_csv(os.path.join(artifact_dir, f'S04_symmetry_arms_contrasts{suffix}.csv'),
                     index=False)

    print('\ncost of each ablation against `full`, in percentage points '
          '(negative = the ablation is worse)')
    print(f"  {'arm':16s} {'top10':>9s} {'op':>9s} {'hard_op':>9s} {'hard_opgf':>10s} "
          f"{'op p':>9s} {'full/arm':>10s}")
    for _, row in contrasts.iterrows():
        print(f"  {row['arm']:16s} {row['delta_top10_pp']:>+9.3f} "
              f"{row['delta_operating_point_pp']:>+9.3f} "
              f"{row['delta_hard_operating_point_pp']:>+9.3f} "
              f"{row['delta_hard_op_given_found_pp']:>+10.3f} "
              f"{row['mcnemar_operating_point_p']:>9.3f} "
              f"{row['mcnemar_operating_point_full_only']:>4d}/"
              f"{row['mcnemar_operating_point_arm_only']:<5d}")
    print('\n  (full/arm = entries where only `full` succeeds / only the ablation does. The hard '
          'stratum\n   carries 232 entry-conditions on `fom-dev`, so its columns move in steps of '
          'about 0.4 pp\n   and one entry flipping is the whole difference -- INHERITED section 2.)')


def main():
    parser = argparse.ArgumentParser(
        description='S04 Phase 2 -- what the symmetry prior is worth, by retrained paired arms.')
    parser.add_argument('--stage', required=True, choices=('features', 'arms'))
    parser.add_argument('--processes', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--fit-seed', type=int, default=SEED,
                        help='seed for the negative subsample and the model fit. The entry '
                             'split stays on campaign 1 seed 12345 whatever this is, so a '
                             'seed sweep varies the fit and never the crystals it is judged on.')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    args = parser.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    {'features': run_features, 'arms': run_arms}[args.stage](args)


if __name__ == '__main__':
    main()
