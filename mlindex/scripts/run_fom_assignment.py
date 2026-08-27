"""S13 -- the per-peak Miller-index assignment probability, on an uncensored campaign-2 pool.

    python mlindex/scripts/run_fom_assignment.py --stage table    --population general
    python mlindex/scripts/run_fom_assignment.py --stage analytic --population general
    python mlindex/scripts/run_fom_assignment.py --stage threshold --population general

Given a peak list and one candidate cell, this asks for the probability that each observed peak is
assigned its correct Miller index -- the number `Candidates.refine_cell` masks on, the number
`n_indexed` counts, and the vector `MITemplates`' template ranker takes as a feature.

WHAT IS DIFFERENT FROM CAMPAIGN 1's BLOCK B, which measured the same estimators (F-125 to F-132).

  * **The pool is uncensored.** Campaign 1 measured on the frozen benchmark, which the prune had
    already cut at M20 >= 5 -- R1, and every block-B number carried that bound. This reads S03's
    threshold-0 capture instead (`mlindex/characterization/fom/prune_capture`), where nothing has
    been deleted for scoring badly, so the population includes the candidates a lower cut admits
    and campaign 2 cares about (C2-F-022, C2-F-032).
  * **The seeding is per (entry, Bravais lattice)**, so any subset of it regenerates comparably
    (PROTOCOL section 6).
  * **There is no network arm.** Campaign 1 built one and it did not beat the recalibration of the
    analytic statistic it was competing with (F-128); campaign 2 does not retry a designed
    negative (PLAN section 8). What S13 asks about a network is a different question -- whether the
    *shipped* peak-assigner inside the IntegralFilter can be retired -- and that is measured end to
    end by `run_fom_assignment_arms.py`, not per peak.

THE ESTIMATORS. `rho`, `taupin` and `dewolff` are coincidence probabilities under a null: "could an
arbitrary cell have put a line this close". The first two are one statistic under two monotone
links -- `get_M20_likelihood` computes both from one `arg` -- so they are **identically ranked**
and differ only in calibration, and the honest bar for both is the isotonic recalibration of `arg`,
which is the best any monotone function of it can do. `posterior` asks a different question --
"given these calculated lines, which one produced this peak" -- and is therefore normalised over
the competing lines rather than against a null.

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark as Bench  # noqa: E402
from mlindex.scripts.run_fom_prune_rerun import ARMS  # noqa: E402
from mlindex.utilities import FigureOfMerits as fom  # noqa: E402
from mlindex.utilities.UnitCellTools import get_hkl_matrix  # noqa: E402
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn  # noqa: E402
from mlindex.utilities.UnitCellTools import get_unit_cell_volume  # noqa: E402

# S03's threshold-0 capture, and the entry tables that carry `q2_obs` and `hkl_true` beside it.
POPULATIONS = ARMS
CAPTURE_ROOT = os.path.join('mlindex', 'characterization', 'fom', 'prune_capture')
PEAKS_ROOT = os.path.join('mlindex', 'data', 'fom_assignment_c2')

# How close a candidate has to sit to the truth's own *setting* for "the correct Miller index" to
# mean anything, measured in error scales. Campaign 1's number, and its reason is R15: most
# candidates labelled `is_correct` are the right lattice in an alternative setting, so their Miller
# indices are expressed in a different basis from `hkl_true`. Pooling the two measures the basis
# convention rather than the assignment -- it moved campaign 1's base rate from 0.83 to 0.38.
#
# 1.0 was chosen by sweep, not by taste: the label rate falls from 0.86 below 1 to 0.66 between 1
# and 3 and to 0.19 above 3, while the reachable ceiling stays flat at ~0.9 throughout. A candidate
# refined onto the truth sits at ~1 error scale because that is what refinement leaves.
SETTING_TOLERANCE = 1.0
SETTING_CUTS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf)

# PROTOCOL section 3 rule 4: anything that uses a sigma reports a sensitivity curve over it. The
# posterior estimates sigma in sample rather than assuming it, so this asks what a mis-estimate of
# that scale costs, which is the honest form of the question for an estimator.
SIGMA_MULTIPLIERS = (0.25, 0.5, 2.0, 4.0)

# In report order. `constant` is not padding: on a population whose base rate is a few percent a
# constant predictor already scores a Brier of a few percent, so any claim that a probability is
# good has to clear that line first (F-083's standing warning).
ANALYTIC_FORMS = ('rho', 'taupin', 'dewolff', 'constant', 'isotonic',
                  'posterior', 'posterior_robust')


def commit_hash():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=BASE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


# -------------------------------------------------------------------------------------------
# Miller-index bookkeeping -- five functions taken from AssignmentModel.py at function
# granularity rather than porting the module, which is a per-Bravais-lattice network trainer this
# step does not build (PROTOCOL section 3 rule 10; CHERRY_PICK.md).
# -------------------------------------------------------------------------------------------
def hkl_reference(lattice_system, bravais_lattice, models_directory=None):
    """The lattice's frozen reference list: (hkl_ref_length, 3), `[0, 0, 0]` last."""
    hkl_ref = np.load(Bench._hkl_ref_path(lattice_system, bravais_lattice, models_directory))
    assert np.all(hkl_ref[-1] == 0), (
        'the reference list does not end in the [0, 0, 0] sentinel, so the unindexed class is '
        'not where Wrapper.setup_hkl and MITemplates believe it is'
        )
    return hkl_ref


def canonical_hkl(hkl, lattice_system):
    """Miller indices reduced to what a peak position can distinguish, as an (..., k) int array.

    `get_hkl_matrix` maps hkl to the design row whose dot product with xnn is q2, so two indices
    with equal rows have equal calculated q2 for *every* cell of the system. Equality of these
    rows is therefore the operational meaning of "the same reflection" here.

    **int64, and that is a correctness requirement rather than tidiness.** The design row is built
    from products of integers, so it is exact in float -- but `hl` for h = -1, l = 0 evaluates to
    **negative zero**, which compares equal to +0.0 and has different bytes, so any lookup keyed on
    the byte representation misses it. That sent one real reflection in twenty-two to the
    "unindexed" sentinel in campaign 1 before it was caught. Casting to int64 maps -0 to 0.
    """
    design = get_hkl_matrix(np.asarray(hkl, dtype=float), lattice_system)
    rounded = np.rint(design)
    assert np.allclose(design, rounded, atol=1e-9), (
        'the hkl design matrix should be integer-valued for integer Miller indices'
        )
    return rounded.astype(np.int64)


def hkl_class_index(hkl, hkl_ref, lattice_system):
    """Class index into `hkl_ref` for each Miller index, with the sentinel for anything absent."""
    hkl = np.asarray(hkl, dtype=float)
    reference = canonical_hkl(hkl_ref, lattice_system)
    lookup = {row.tobytes(): index for index, row in enumerate(np.ascontiguousarray(reference))}
    # The sentinel is the last row, [0, 0, 0], so an unmatched peak and a contaminant land on the
    # same class -- which is what they are: lines this reference list cannot index.
    sentinel = len(hkl_ref) - 1
    flat = np.ascontiguousarray(canonical_hkl(hkl.reshape(-1, 3), lattice_system))
    codes = np.fromiter(
        (lookup.get(row.tobytes(), sentinel) for row in flat), dtype=np.int64, count=len(flat),
        )
    return codes.reshape(hkl.shape[:-1])


def assignment_labels(hkl_assigned, hkl_true, lattice_system):
    """Per-peak truth: did the candidate assign this peak its correct Miller index?

    Contaminants carry `(0, 0, 0)` as their true index, so they can only be called correct by a
    candidate that assigned them the sentinel, which no assignment does -- the sentinel's
    calculated q2 is zero. They are therefore always False, which is the truth.
    """
    assigned = canonical_hkl(hkl_assigned, lattice_system)
    truth = canonical_hkl(hkl_true, lattice_system)
    return np.all(assigned == truth, axis=-1)


def setting_residuals(q2_obs, hkl_true, xnn, lattice_system):
    """How far each candidate is from describing these peaks at the *true* Miller indices.

    q2 for `hkl_true` computed through the **candidate's** cell, compared with the observed peaks
    and divided by the error scale the generator drew from. A candidate refined onto the truth
    comes back at ~1; the same lattice in a different setting comes back at tens to hundreds,
    because its indices label different reflections.

    Contaminant peaks carry `(0, 0, 0)` and are excluded -- they have no true reflection to place.
    """
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info

    real = np.any(np.asarray(hkl_true) != 0, axis=1)
    if not real.any():
        return np.full(len(xnn), np.inf)
    design = canonical_hkl(hkl_true, lattice_system).astype(float)
    q2_at_true = design@np.asarray(xnn).T
    params = get_peak_generation_info()['q2_error_params']
    scale = (params[0] + q2_obs*params[1])[:, np.newaxis]
    return np.median((np.abs(q2_at_true - q2_obs[:, np.newaxis])/scale)[real], axis=0)


def reachable_peaks(hkl_true, lattice_system, bravais_lattice, spacegroup,
                    models_directory=None):
    """Is each peak's true reflection even in the list the candidate assigns from?

    `assign_lines` picks the nearest line of `hkl_ref_for(..., spacegroup)`, the model's truncated
    reference list narrowed to one extinction group. A true reflection outside it cannot be
    recovered by any assignment rule, so this is a hard ceiling on the label and it belongs beside
    every number rather than inside the residual. Contaminants carry the sentinel row and would
    match trivially; they are marked unreachable, which is what they are.
    """
    hkl_true = np.asarray(hkl_true, dtype=float)
    reference = Bench.hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory)
    known = set(map(
        bytes, np.ascontiguousarray(canonical_hkl(reference, lattice_system))))
    rows = np.ascontiguousarray(canonical_hkl(hkl_true, lattice_system))
    contaminant = np.all(hkl_true == 0, axis=1)
    return np.array([row.tobytes() in known and not is_contaminant
                     for row, is_contaminant in zip(rows, contaminant)])


# -------------------------------------------------------------------------------------------
# The per-peak table
# -------------------------------------------------------------------------------------------
# Only what the per-peak table needs. The capture carries seven merits at the prune site and the
# deduplication radius beside them, and reading a 2.5 million-row shard whole costs ~1.5 GB of
# resident memory for columns nothing here touches.
CAPTURE_COLUMNS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'lattice_system',
                   'candidate_id', 'xnn', 'spacegroup', 'n_peaks', 'M20', 'm20_at_prune',
                   'is_correct', 'split')


def capture_shards(population):
    directory = Path(BASE) / CAPTURE_ROOT / population
    shards = sorted(directory.glob('predownsample_*.parquet'))
    if not shards:
        raise SystemExit(f'no capture pool under {directory}')
    return shards


def entry_table(population):
    """`q2_obs`, `hkl_true` and the truth columns for every pattern of the population."""
    root = Path(BASE) / POPULATIONS[population]
    frames = [pd.read_parquet(shard) for bundle_dir in sorted(root.iterdir())
              if bundle_dir.is_dir() for shard in sorted(bundle_dir.glob('entries_*.parquet'))]
    if not frames:
        raise SystemExit(f'no entry tables under {root}')
    entries = pd.concat(frames, ignore_index=True)
    return entries.set_index(['entry_id', 'condition_bundle'], drop=False)


def subsample(candidates, max_candidates, rng):
    """Cap the incorrect candidates per (entry, bundle, lattice); keep every correct one.

    Correct candidates are the entire positive class and a fraction of a percent of the pool, so
    they are never subsampled. The retained incorrect rows carry weight 1 and the count dropped is
    reported, so a bounded number is reported as bounded (PROTOCOL section 10).
    """
    if max_candidates is None:
        return candidates, 0
    chosen, dropped = [], 0
    keys = ['entry_id', 'condition_bundle', 'bravais_lattice']
    for _, group in candidates.groupby(keys, sort=False):
        if len(group) <= max_candidates:
            chosen.append(group)
            continue
        correct = group.loc[group['is_correct'].astype(bool)]
        rest = group.drop(index=correct.index)
        take = max(max_candidates - len(correct), 0)
        if len(rest) > take:
            dropped += len(rest) - take
            rest = rest.iloc[np.sort(rng.choice(len(rest), size=take, replace=False))]
        chosen.append(pd.concat([correct, rest]))
    return pd.concat(chosen, ignore_index=True), dropped


def collect_peaks(candidates, entries, models_directory=None, verbose=True):
    """One row per (candidate, observed peak), with the label and every analytic probability.

    Built once, so every form is scored on **identical rows** -- a Brier comparison between two
    scores computed over different subsamples is not paired and does not mean anything.

    Grouped by (entry, bundle, lattice, spacegroup) because that is the unit `assign_lines` works
    on: the extinction group decides which reference lines exist, so candidates that chose
    different groups have different vocabularies and cannot share one reference array.
    """
    blocks, n_groups = [], 0
    grouped = candidates.groupby(
        ['entry_id', 'condition_bundle', 'bravais_lattice', 'spacegroup'], sort=False)
    for (entry_id, bundle, bravais_lattice, spacegroup), group in grouped:
        entry = entries.loc[(entry_id, bundle)]
        lattice_system = group['lattice_system'].iloc[0]
        n_peaks = int(group['n_peaks'].iloc[0])
        q2_obs = np.asarray(entry['q2_obs'], dtype=np.float64)[:n_peaks]
        hkl_true = np.asarray(entry['hkl_true'], dtype=np.float64).reshape(-1, 3)[:n_peaks]
        xnn = np.stack([np.asarray(value, dtype=np.float64) for value in group['xnn']])

        q2_ref_calc, _, hkl_assigned, q2_calc = Bench.assign_lines(
            q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, models_directory,
            )
        # V* exactly as get_M20_likelihood_from_xnn computes it. It must NOT come from a stored
        # column: the shipped statistic is defined on the reciprocal volume of the cell in hand.
        reciprocal_volume = get_unit_cell_volume(
            get_reciprocal_unit_cell_from_xnn(
                xnn, partial_unit_cell=True, lattice_system=lattice_system),
            partial_unit_cell=True, lattice_system=lattice_system)

        argument = fom.get_assignment_argument(
            q2_obs, q2_calc, bravais_lattice, reciprocal_volume)
        dewolff = fom.get_assignment_probability_dewolff(
            q2_obs, q2_calc, xnn, lattice_system, bravais_lattice)
        # Computed here rather than in a second pass because it needs `q2_ref_calc`, the full set
        # of calculated lines, which is far too large to store and is already in hand. The sigma
        # and the nearest-line distances are computed once and handed to every posterior form,
        # which is what that argument pair is for.
        sigma, d1 = fom.get_assignment_sigma(q2_obs, q2_ref_calc, lattice_system)
        posterior = fom.get_assignment_posterior(
            q2_obs, q2_ref_calc, lattice_system, sigma=sigma, d1=d1)
        posterior_robust = fom.get_assignment_posterior(
            q2_obs, q2_ref_calc, lattice_system, robust=True)
        sigma_curve = {
            f'posterior_sigma{multiplier:g}': fom.get_assignment_posterior(
                q2_obs, q2_ref_calc, lattice_system, sigma=sigma, d1=d1,
                sigma_multiplier=multiplier)
            for multiplier in SIGMA_MULTIPLIERS
            }

        hkl_ref_full = hkl_reference(lattice_system, bravais_lattice, models_directory)
        label = assignment_labels(hkl_assigned, hkl_true[np.newaxis], lattice_system)
        setting_residual = setting_residuals(q2_obs, hkl_true, xnn, lattice_system)
        reachable = reachable_peaks(
            hkl_true, lattice_system, bravais_lattice, spacegroup, models_directory)

        n_candidates = q2_calc.shape[0]
        blocks.append(pd.DataFrame(dict(
            entry_id=np.repeat(group['entry_id'].to_numpy(), n_peaks),
            condition_bundle=bundle,
            candidate_id=np.repeat(group['candidate_id'].to_numpy(), n_peaks),
            split=np.repeat(group['split'].to_numpy(), n_peaks),
            bravais_lattice=bravais_lattice,
            lattice_system=lattice_system,
            bravais_lattice_true=entry['bravais_lattice_true'],
            spacegroup=spacegroup,
            is_correct=np.repeat(np.asarray(group['is_correct'], dtype=bool), n_peaks),
            M20=np.repeat(np.asarray(group['M20'], dtype=np.float64), n_peaks),
            m20_at_prune=np.repeat(
                np.asarray(group['m20_at_prune'], dtype=np.float64), n_peaks),
            setting_residual=np.repeat(setting_residual, n_peaks).astype(np.float32),
            same_setting=np.repeat(setting_residual < SETTING_TOLERANCE, n_peaks),
            peak_index=np.tile(np.arange(n_peaks), n_candidates),
            q2_obs=np.tile(q2_obs, n_candidates).astype(np.float32),
            q2_calc=q2_calc.reshape(-1).astype(np.float32),
            # float64 for the shared statistic: rho and taupin are derived from it, and in float32
            # the two links round differently enough to move their AUCs apart in the fourth
            # decimal -- which would look like an ordering difference that does not exist.
            argument=argument.reshape(-1),
            dewolff=dewolff.reshape(-1),
            posterior=posterior.reshape(-1),
            posterior_robust=posterior_robust.reshape(-1),
            **{name: values.reshape(-1) for name, values in sigma_curve.items()},
            assign_class=hkl_class_index(
                hkl_assigned, hkl_ref_full, lattice_system).reshape(-1).astype(np.int32),
            true_class=np.tile(hkl_class_index(
                hkl_true, hkl_ref_full, lattice_system), n_candidates).astype(np.int32),
            is_contaminant=np.tile(np.all(hkl_true == 0, axis=1), n_candidates),
            reachable=np.tile(reachable, n_candidates),
            label=label.reshape(-1),
            )))
        n_groups += 1
        if verbose and n_groups % 2000 == 0:
            print(f'    {n_groups} groups, {sum(len(b) for b in blocks)} peak rows', flush=True)

    peaks = pd.concat(blocks, ignore_index=True)
    peaks['rho'] = 1.0/(1.0 + peaks['argument'])
    peaks['taupin'] = np.exp(-peaks['argument'])
    return peaks


def run_table(args):
    rng = np.random.default_rng(args.seed)
    entries = entry_table(args.population)
    out_dir = Path(BASE) / args.peaks_root / args.population
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for shard in capture_shards(args.population):
        destination = out_dir / f'peaks_{shard.stem.split("_", 1)[1]}.parquet'
        if destination.exists() and not args.overwrite:
            print(f'{destination.name}: already done, skipping', flush=True)
            continue
        started = time.time()
        candidates = pd.read_parquet(shard, columns=list(CAPTURE_COLUMNS))
        n_available = len(candidates)
        candidates, dropped = subsample(candidates, args.max_candidates, rng)
        peaks = collect_peaks(candidates, entries, args.models_dir)
        peaks.to_parquet(destination, index=False)
        summary = dict(shard=shard.name, n_candidates_available=int(n_available),
                       n_candidates=int(len(candidates)), n_candidates_dropped=int(dropped),
                       n_peak_rows=int(len(peaks)),
                       n_source_entries=int(peaks['entry_id'].nunique()),
                       seconds=time.time() - started)
        summaries.append(summary)
        print(f'{destination.name}: {summary["n_peak_rows"]} rows from '
              f'{summary["n_candidates"]} of {n_available} candidates, '
              f'{summary["seconds"]:.0f}s', flush=True)
    (out_dir / 'table_provenance.json').write_text(json.dumps({
        'commit': commit_hash(), 'population': args.population, 'seed': args.seed,
        'max_candidates': args.max_candidates, 'platform': platform.platform(),
        'machine': platform.machine(), 'setting_tolerance': SETTING_TOLERANCE,
        'shards': summaries,
        }, indent=2), encoding='utf-8')
    print(f'wrote {out_dir}')


def load_peaks(peaks_root, population, columns=None):
    directory = Path(BASE) / peaks_root / population
    shards = sorted(directory.glob('peaks_*.parquet'))
    if not shards:
        raise SystemExit(f'no peak table under {directory}; run --stage table first')
    return pd.concat([pd.read_parquet(s, columns=columns) for s in shards], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage',
                        choices=['table', 'analytic', 'threshold', 'replay', 'n_indexed',
                                 'choose', 'figure', 'consumers'],
                        default='table')
    parser.add_argument('--population', default='general', choices=sorted(POPULATIONS))
    parser.add_argument('--max-candidates', type=int, default=25,
                        help='incorrect candidates kept per (entry, bundle, lattice); every '
                             'correct candidate is kept whatever this is')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--peaks-root', default=PEAKS_ROOT)
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--artifact-dir',
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--min-rows', type=int, default=500,
                        help='per-lattice rows below which a row is not reported; a lattice\n'
                             'with a handful of peaks carries no claim')
    parser.add_argument('--bootstrap', type=int, default=0,
                        help='entry-clustered bootstrap replicates for the pooled Brier')
    parser.add_argument('--posterior-threshold', type=float, default=0.99,
                        help='the mask cut the replay chose on fom-train, across both\n'
                             'populations by the max-min rule (--stage choose)')
    parser.add_argument('--mask-form', default='posterior',
                        help='the statistic --stage choose picks a cut for')
    parser.add_argument('--mask-metric-rtol', type=float, default=0.002,
                        help='the tightness the replay is scored at when choosing')
    parser.add_argument('--replay-grid', default=None,
                        help='comma-separated thresholds; default REPLAY_GRID')
    parser.add_argument('--replay-forms', default=None,
                        help='comma-separated statistics; default all three')
    parser.add_argument('--replay-weighted', action='store_true',
                        help='add a soft-weighted arm per statistic: no cut, every\n'
                             'peak in at weight p through sigma_reduction')
    parser.add_argument('--replay-suffix', default='',
                        help='distinguishes a focused rerun from the main sweep')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.stage == 'table':
        run_table(args)
    else:
        from mlindex.scripts import run_fom_assignment_report as report
        getattr(report, f'run_{args.stage}')(args)


if __name__ == '__main__':
    main()
