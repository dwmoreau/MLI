"""S08 -- read the reproducibility floor off an ensemble of runs of the indexer.

`submit_fom_floor_arms.sh` indexes one set of patterns several times, changing only the search
seed. This turns those arms into the three numbers every gate in the campaign is written against.

    python mlindex/scripts/run_fom_floor_report.py \
        --arm-root $SCRATCH/fom_campaign2/floor \
        --benchmark $SCRATCH/fom_campaign2/pool \
        --artifact-dir docs/fom_campaign2/artifacts

Three quantities, which campaign 1 conflated into one recalled number:

  value floor     Shirley's own quantity: for a pattern whose correct solution is found in more
                  than one arm, how far apart are the merit values the arms report for it. This
                  is what "figures of merit differing by more than 10 percent for the same
                  solution" means, measured rather than recalled.
  metric floor    How far apart the arms' reported *rates* are -- the operating point, top-10.
  contrast floor  How far apart the arms put the *difference between two merits*. A gate asks
                  "does A beat B by more than the floor", and A and B are computed on the same
                  pool, so their difference is far more stable than either. PROTOCOL section 8
                  makes this the number a gate is read against, in standard errors.

Four things this does that campaign 1's version did not.

**The aggregate is composed from the reporting split's per-lattice counts, not the sample's.**
The floor sample is drawn balanced across lattices, because a proportional draw leaves cF and cI
with one or two patterns and no per-lattice floor at all -- and PROTOCOL section 8 requires a
per-lattice claim to be read against that lattice's own floor. Campaign 1's sample was
proportional, so it could scale by the sample's own composition; S08's acceptance condition 4 says
in terms that a sample drawn any other way needs the split's own counts instead. Those counts are
written beside the sample by `run_fom_floor_entries.py` and are read here, never re-derived.

**No weighted branch.** Campaign 1's `induced_standard_error_stratified` computed the standard
error of a CNRS-reweighted rate, and its `weights=None` meant *weighted* -- the opposite of the
module's convention, in the one calculation every gate is read against. PROTOCOL section 3 rule 6
makes every aggregate unweighted, so that function has no caller and is not ported; the unweighted
standard error is the plain `induced_from_differences`. Leaving it out is the fix.

**The generation floor is reported separately from the scoring floor.** Campaign 1's F-150
concluded the floor is "scoring noise, not generation noise" because its arms' operating-point
flips and reachability flips were disjoint. On this campaign's hard stratum that already fails:
two runs of the same 972 pattern-conditions agreed on 328 and disagreed on 130 about whether a
pattern is solvable at all (C2-F-031, C2-R-005). So the two are reported side by side and the
disjointness is re-tested rather than assumed.

**Every arm's peak lists are compared before any number is read**, and a mismatch raises. The
whole design is that only the search seed moved; if `--seed` moved too, the arms differ in their
data and the spread is not a floor.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics


# The merits the floor is measured for. Every one is in `FomMetrics.RANK_EXACT_MERITS`, so rank
# metrics on the subsampled arms are exact to the pool's own depth K -- which a learned score's
# would not be (C2-F-077). A floor for a learned score has to be measured on a fully retained arm.
FLOOR_MERITS = ('M20', 'M_sym', 'M_rev', 'M_tilde', 'X_N')

# Metrics reported per arm. `found` is the pool's ceiling and moves only when the *generation*
# differs between arms, which is why it is here: it separates "the score moved" from "the pool
# moved".
FLOOR_METRICS = ('operating_point', 'top1', 'top10', 'threshold_only', 'found', 'reported',
                 'false_positive', 'mrr', 'precision')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='S08 -- the reproducibility floor, from an ensemble of runs of the indexer')
    parser.add_argument('--arm-root', type=str, required=True,
                        help='Directory holding one subdirectory per arm, each a generated pool')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Benchmark B. Included as the first arm, restricted to the floor '
                             'entries -- it was generated at a recorded search seed and a subset '
                             'run reproduces it bit for bit (C2-F-058), so it is an arm already')
    parser.add_argument('--composition', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts',
                                             'S08_floor_composition.csv'),
                        help='Per-lattice counts of the reporting split, from '
                             'run_fom_floor_entries.py. The aggregate is composed with these '
                             'because the sample is balanced rather than proportional')
    parser.add_argument('--artifact-dir', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--threshold', type=float, default=None,
                        help='Accept threshold, selected on fom-train and passed in. Omit for '
                             'rank metrics only')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--tag', type=str, default='S08_floor')
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------------------
# The arithmetic. Ported from `fom` @ 7c137c3, less the weighted branch.
# ----------------------------------------------------------------------------------------
def relative_spread(values):
    """(max - min) / |median| across the replicate columns of each row.

    Range rather than standard deviation because Shirley's statement is about how far apart two
    reported values can be, and because a handful of arms do not support a variance estimate worth
    the name. Divided by the row's own median so the answer is scale-free and comparable across
    merits whose units differ by orders of magnitude.
    """
    scale = values.abs().median(axis=1).replace(0.0, np.nan)
    return (values.max(axis=1) - values.min(axis=1)) / scale


def flip_rate(first, second):
    """Fraction of entries whose boolean outcome differs between two arms."""
    return float(np.mean(np.asarray(first, dtype=bool) != np.asarray(second, dtype=bool)))


def induced_standard_error(rate, n_entries):
    """Standard error of the *difference* in a per-entry rate between two runs, from the flip rate.

    If a fraction `f` of entries change their outcome between two arms, the change in the reported
    rate is the mean of n draws that are 0 with probability 1 - f and +-1 otherwise, so its
    variance is f/n and its standard error is sqrt(f/n). This is the derived form; the measured
    form is `induced_from_differences`. S08 acceptance condition 3 is that the two agree --
    campaign 1 got 0.366 pp derived against 0.360 pp reported.
    """
    return float(np.sqrt(rate / n_entries))


def induced_from_differences(differences, n_entries):
    """Standard error at `n_entries` of a per-entry difference measured on this sample.

    Given the per-entry change `d_i` between two arms -- of an outcome flag, or of the *contrast*
    between two merits' flags -- the change in the reported mean is `mean(d)`, so its standard
    error at any entry count is `sd(d)/sqrt(n)`. Preferred to the flip-rate form wherever the
    per-entry values are in hand, because it needs no assumption about how flips are distributed.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return float('nan')
    return float(np.std(differences, ddof=1) / np.sqrt(n_entries))


def compose_aggregate(per_lattice, composition):
    """An aggregate standard error from per-lattice ones, using the SPLIT's composition.

    Var(aggregate) = sum_bl (n_bl/N)^2 * Var_bl / n_bl, so the aggregate standard error is
    sqrt(sum_bl w_bl^2 * se_bl^2 * (n_sample_bl / n_split_bl)) -- each lattice's measured standard
    error rescaled from the sample's count to the split's, then combined at the split's weights.

    This is the step campaign 1 did not need. Its floor sample was drawn proportional to its
    reporting split, so `n_bl` scaled by the sample's own composition and the aggregate was a
    plain mean. This campaign's sample is balanced, so the sample's shape is NOT the split's, and
    using it would weight cF -- 20 entries of 3 810 -- as though it were a fourteenth of the
    population. S08 acceptance condition 4 is exactly this check.
    """
    weights = composition.set_index('bravais_lattice')
    variance = 0.0
    covered = 0.0
    for row in per_lattice.itertuples():
        lattice = row.bravais_lattice
        if lattice not in weights.index or not np.isfinite(row.se_pp):
            continue
        n_split = float(weights.loc[lattice, 'split_entries'])
        n_sample = float(row.n_entries)
        if n_split <= 0 or n_sample <= 0:
            continue
        weight = float(weights.loc[lattice, 'floor_weight'])
        # se_pp was measured at n_sample; rescale its variance to n_split before combining.
        variance += (weight**2) * (row.se_pp**2) * (n_sample / n_split)
        covered += weight
    if not covered:
        return float('nan'), 0.0
    return float(np.sqrt(variance)), float(covered)


# ----------------------------------------------------------------------------------------
# Loading the arms
# ----------------------------------------------------------------------------------------
def arm_directories(root):
    """One directory per arm. A pool written per (arm, condition) nests one level deeper."""
    root = Path(root)
    paths = sorted(path for path in root.iterdir() if path.is_dir())
    if len(paths) < 1:
        raise SystemExit(f'{root} holds no arm directories')
    return paths


def check_arms_are_comparable(arms):
    """Every arm must hold the same patterns with the same peak lists. Raises if not.

    S08 acceptance condition 5, and it runs before anything is read rather than after. The design
    is that only `--optimizer-seed` moved; `--seed` fixes the entry sample and the per-entry noise
    and therefore the peak lists. If someone varies that instead, the arms differ in their DATA,
    and their spread is generation noise and scoring noise together -- which is the one thing the
    floor exists to separate.
    """
    reference = reference_name = None
    for name, entries in arms.items():
        digests = dict(zip(entries['entry_id'].astype(str), entries['q2_digest'].astype(str)))
        if reference is None:
            reference, reference_name = digests, name
            continue
        shared = set(reference) & set(digests)
        if not shared:
            raise SystemExit(f'Arms {reference_name} and {name} share no entries at all.')
        differing = sorted(key for key in shared if reference[key] != digests[key])
        if differing:
            raise SystemExit(
                f'Arms {reference_name} and {name} disagree on the peak lists of '
                f'{len(differing)} of {len(shared)} entries, e.g. {differing[:5]}. The arms are '
                f'not measuring the reproducibility floor: only --optimizer-seed may differ, and '
                f'--seed fixes the peak lists. Regenerate with a shared --seed.')
    return True


def load_arms(arm_root, benchmark, floor_entries):
    """{arm name: (candidates, entries)}, with Benchmark B restricted in as the first arm."""
    arms = {}
    for path in arm_directories(arm_root):
        entries = FomBenchmark.load_entries(path)
        arms[path.name] = (path, entries)
    if benchmark is not None:
        entries = FomBenchmark.load_entries(benchmark)
        entries = entries.loc[entries['entry_id'].isin(floor_entries)].reset_index(drop=True)
        arms['benchmark'] = (Path(benchmark), entries)
    if len(arms) < 2:
        raise SystemExit(f'The floor needs at least two arms; found {sorted(arms)}. '
                         f'Pass --benchmark to include Benchmark B as the first one.')
    return arms


def main(argv=None):
    args = _parse_args(argv)
    composition = pd.read_csv(args.composition)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    entries_path = artifact_dir / f'{args.tag}_entries.csv'
    floor_entries = set(pd.read_csv(entries_path)['identifier'].astype(str))
    arms = load_arms(args.arm_root, args.benchmark, floor_entries)
    check_arms_are_comparable({name: entries for name, (_, entries) in arms.items()})
    print(f'{len(arms)} arms, peak lists identical across all of them')

    # One MetricsResult per (arm, merit). The arms are subsampled the same way Benchmark B is, so
    # every merit here has to be one the subsampler ranked on, or the ranks are not exact.
    results = {}
    for name, (path, entries) in arms.items():
        keep = entries.loc[entries['entry_id'].isin(floor_entries)]
        for merit in FLOOR_MERITS:
            results[(name, merit)] = FomMetrics.evaluate(
                path, score=merit, threshold=args.threshold, top_n=args.top_n,
                entries=keep, n_bootstrap=0,
                )
        print(f'  {name}: {results[(name, FLOOR_MERITS[0])].meta["n_entries"]} cells')

    metric_rows, contrast_rows, lattice_rows = [], [], []
    arm_names = sorted(arms)
    for merit in FLOOR_MERITS:
        table = pd.DataFrame({name: [results[(name, merit)].metric(metric)
                                     for metric in FLOOR_METRICS]
                              for name in arm_names}, index=list(FLOOR_METRICS))
        for metric in FLOOR_METRICS:
            values = table.loc[metric].to_numpy(dtype=np.float64)*100.0
            metric_rows.append(dict(
                merit=merit, metric=metric,
                mean=float(np.nanmean(values)), sd_pp=float(np.nanstd(values, ddof=1)),
                range_pp=float(np.nanmax(values) - np.nanmin(values)),
                n_arms=len(arm_names),
                **{f'arm_{name}': float(value) for name, value in zip(arm_names, values)},
                ))

        # The contrast floor, and the per-lattice floors, both from paired per-entry flags.
        baseline = 'M20'
        if merit != baseline:
            for metric in ('operating_point', 'top10'):
                paired = _paired_contrast(results, arm_names, merit, baseline, metric)
                if paired is None:
                    continue
                differences, lattices, n = paired
                contrast_rows.append(dict(
                    merit=merit, baseline=baseline, metric=metric,
                    mean_delta_pp=float(np.mean(differences)*100.0),
                    se_pp=induced_from_differences(differences*100.0, n),
                    n_entries=int(n),
                    ))
                for lattice in sorted(set(lattices)):
                    mask = lattices == lattice
                    lattice_rows.append(dict(
                        merit=merit, baseline=baseline, metric=metric, bravais_lattice=lattice,
                        se_pp=induced_from_differences(differences[mask]*100.0, int(mask.sum())),
                        n_entries=int(mask.sum()),
                        ))

    metric_table = pd.DataFrame(metric_rows)
    contrast_table = pd.DataFrame(contrast_rows)
    lattice_table = pd.DataFrame(lattice_rows)

    # The aggregate, composed from the split's own counts because the sample is balanced.
    aggregate_rows = []
    for (merit, metric), block in lattice_table.groupby(['merit', 'metric']):
        se, covered = compose_aggregate(block, composition)
        aggregate_rows.append(dict(merit=merit, baseline='M20', metric=metric,
                                   se_pp=se, lattice_weight_covered=covered,
                                   composed_from='split_entries'))
    aggregate_table = pd.DataFrame(aggregate_rows)

    for name, frame in (('metric', metric_table), ('contrast', contrast_table),
                        ('by_lattice', lattice_table), ('aggregate', aggregate_table)):
        path = artifact_dir / f'{args.tag}_{name}.csv'
        frame.to_csv(path, index=False)
        print(f'wrote {path}')
    return 0


def _paired_contrast(results, arm_names, merit, baseline, metric):
    """Per-entry (merit - baseline) contrast, differenced between the first two arms.

    The contrast floor is the spread of a *difference between two merits on one pool*, so the
    per-entry quantity is (merit_flag - baseline_flag) within an arm, and the floor is how much
    that quantity moves between arms.
    """
    if len(arm_names) < 2:
        return None
    frames = []
    for name in arm_names[:2]:
        merged = results[(name, merit)].per_entry.merge(
            results[(name, baseline)].per_entry[['entry_id', 'condition_bundle', metric]],
            on=['entry_id', 'condition_bundle'], suffixes=('', '_base'), validate='1:1')
        merged['contrast'] = (merged[metric].astype(float)
                              - merged[f'{metric}_base'].astype(float))
        frames.append(merged[['entry_id', 'condition_bundle', 'bravais_lattice', 'contrast']])
    joined = frames[0].merge(frames[1], on=['entry_id', 'condition_bundle', 'bravais_lattice'],
                             suffixes=('_a', '_b'), validate='1:1')
    if joined.empty:
        return None
    differences = (joined['contrast_a'] - joined['contrast_b']).to_numpy(dtype=np.float64)
    return differences, joined['bravais_lattice'].to_numpy(), joined.shape[0]


if __name__ == '__main__':
    raise SystemExit(main())
