"""S14 -- what the two destructive cuts delete, and whether the ceiling is real.

The project's ceiling is `ceiling_rescorer`: the fraction of entries with a correct
candidate anywhere in the pool. F-136 measures it at 0.6961 against a 0.6468 operating
point, so **4.93 pp is the entire prize for every figure of merit this project will ever
build**, while 30.4% of entries (90% on the hard stratum) have no correct candidate at
all. METRICS.md section 3 assigns that bucket here.

Two cuts delete candidates before ranking ever sees them, and neither has been measured:

  * `prune_below_m20` discards everything scoring below 5.0 (F-049, Q31, rebuild row R1).
  * deduplication keeps the highest-M20 member of each xnn neighbourhood and deletes the
    rest (F-065, rebuild row R2).

WHY THIS IS ONE MEASUREMENT PER ARM AND NOT A DIFFERENCE BETWEEN ARMS. The obvious design
-- run at threshold 5 and at threshold 0 and subtract -- does not work, and the reason is
worth stating because it is not obvious. The candidate set *entering* the prune is
bit-identical between the two arms, but everything after it is not: `refine_cell`,
`standardize_cell` and `correct_off_by_two` all call `fix_unphysical(rng=self.rng, ...)`,
which draws from one shared stream sized by the number of rows. At threshold 0 there are
~400x more rows, so the stream diverges immediately and the surviving cells differ. Measured
on three hard-stratum entries: only 57% of the threshold-5 rows appear bit-identically in
the threshold-0 run, and the rest differ by up to 27% in relative xnn -- genuinely different
cells, not perturbations.

So each cut is measured *within* one run instead:

  * Q31, from the threshold-0 arm alone. Every candidate carries `m20_at_prune`, the
    pre-extinction-group M20 the production rule actually tested (the stored `M20` is the
    post-assignment value, a different quantity -- F-049). A correct candidate with
    `m20_at_prune < 5.0` is one production would have deleted. No cross-arm comparison.
  * F-065, from the production arm alone. The pre-deduplication dump and the post-
    deduplication dump come from the same run, so their difference is exactly what the
    tiebreak destroyed.

    python run_fom_retention_report.py --stage gate    --t0-root ... --t5-root ...
    python run_fom_retention_report.py --stage analyse --t0-root ... --t5-root ...

Bounds to carry into anything written from this (PROTOCOL section 7):

  * Not bit-comparable with Benchmark A. The search RNG is seeded per pool and advances per
    entry, so a 243-entry run explores differently from the 5 955-entry bundle the frozen
    pool came from. What *is* exact is the peak lists -- noise is seeded per entry -- and
    the gate stage checks that.
  * `prune_below_m20` keeps the argmax when nothing clears the threshold, per rank. On the
    hard stratum that fallback, not the threshold, is what populates most lattices, so
    "would have been deleted" is an upper bound by at most one candidate per rank per
    (entry, lattice).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume

PRODUCTION_PRUNE_THRESHOLD = 5.0
N_BOOTSTRAP = 2000


def _parse_args():
    parser = argparse.ArgumentParser(description='S14 -- the prune and dedup cuts (Q31, F-065)')
    parser.add_argument('--stage', choices=['gate', 'analyse'], required=True)
    parser.add_argument('--t0-root', required=True,
                        help='Root holding one subdirectory per bundle, prune threshold 0')
    parser.add_argument('--t5-root', default=None,
                        help='Root holding one subdirectory per bundle at the production '
                             'threshold. Needed for the deduplication half (F-065)')
    parser.add_argument('--benchmark-dir', default='mlindex/data/fom_benchmark',
                        help='Frozen Benchmark A pool, for the peak-list reproduction gate')
    parser.add_argument('--artifact-dir', default='docs/fom/artifacts')
    parser.add_argument('--tag', default='S14_cuts')
    parser.add_argument('--rtol', type=float, default=1e-2,
                        help='Tolerance for validate_candidate_known_bl, matching S04')
    parser.add_argument('--n-processes', type=int, default=1,
                        help='Labelling is ~9 ms/candidate and is the whole cost here')
    return parser.parse_args()


def bundle_directories(root):
    """One subdirectory per condition bundle, named by its manifest."""
    directories = {}
    for path in sorted(Path(root).iterdir()):
        manifest = path / 'manifest.json'
        if path.is_dir() and manifest.exists():
            with open(manifest, encoding='utf-8') as handle:
                directories[json.load(handle)['bundle']] = path
    if not directories:
        raise SystemExit(f'No bundle directories with a manifest.json under {root}')
    return directories


def load_predownsample(bundle_dir):
    """Every candidate entering deduplication, with unit_cell derived rather than stored.

    The pre-deduplication population runs to ~58 000 rows per entry at threshold 0, so
    unit_cell / volume are left off disk and rebuilt here; they are an exact function of
    xnn and the lattice system.
    """
    shards = sorted(Path(bundle_dir).glob('predownsample_*.parquet'))
    if not shards:
        raise SystemExit(
            f'No predownsample_*.parquet under {bundle_dir}. The run needs '
            '--dump-predownsample; without it the rows this measures were never written.')
    frame = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    pieces = []
    for lattice_system, group in frame.groupby('lattice_system', sort=False):
        unit_cell = get_unit_cell_from_xnn(
            np.stack(group['xnn'].values), partial_unit_cell=True,
            lattice_system=lattice_system)
        pieces.append(group.assign(
            unit_cell=list(unit_cell),
            volume=get_unit_cell_volume(unit_cell, partial_unit_cell=True,
                                        lattice_system=lattice_system),
            ))
    return pd.concat(pieces).sort_index()


def _reachable(labelled, entries, mask=None):
    """Entries with a correct candidate, on their own true Bravais lattice.

    Restricted to the entry's own lattice for the same reason `run_fom_dump_consolidate`
    does it: "no correct candidate in the pool" is a generation failure and PROTOCOL
    section 8 wants that bucket kept separate from ranking failures.
    """
    truth = entries.set_index('entry_id')['bravais_lattice_true']
    frame = labelled if mask is None else labelled.loc[mask]
    frame = frame.loc[frame['is_correct'].fillna(False)]
    frame = frame.loc[frame['bravais_lattice'] == frame['entry_id'].map(truth)]
    return set(frame['entry_id'].unique())


def _bootstrap_ceiling(reached, all_entries, rng):
    """Cluster bootstrap over source entries, which is the unit PROTOCOL section 8 names."""
    entries = np.asarray(sorted(all_entries))
    hit = np.isin(entries, list(reached)).astype(float)
    draws = rng.integers(0, entries.size, size=(N_BOOTSTRAP, entries.size))
    samples = hit[draws].mean(axis=1)
    return float(hit.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run_gate(args):
    """Cheapest checks first, so a structural failure is not diagnosed as a physics result."""
    findings = []
    benchmark_entries = None
    benchmark_dir = Path(args.benchmark_dir)
    # The consolidated pool writes one entries.parquet; a raw per-pool dump writes
    # entries_<tag>.parquet. Accept either, and say so loudly if neither is there --
    # skipping this check silently is how a run gets read as reproducing patterns it
    # never compared against.
    entry_files = sorted(benchmark_dir.glob('entries*.parquet'))
    if entry_files:
        benchmark_entries = pd.concat(
            [pd.read_parquet(path,
                             columns=['entry_id', 'condition_bundle', 'q2_digest', 'q2_obs'])
             for path in entry_files], ignore_index=True)
    else:
        raise SystemExit(
            f'No entries*.parquet under {benchmark_dir}, so the peak-list reproduction '
            'gate cannot run. Point --benchmark-dir at the consolidated Benchmark A pool.')

    for label, root in (('t0', args.t0_root), ('t5', args.t5_root)):
        if root is None:
            continue
        for bundle, bundle_dir in bundle_directories(root).items():
            entries = FomBenchmark.load_entries(bundle_dir)
            predownsample = pd.concat(
                [pd.read_parquet(path) for path in
                 sorted(Path(bundle_dir).glob('predownsample_*.parquet'))],
                ignore_index=True)

            # 1. The peak lists must reproduce Benchmark A exactly. This is the one thing a
            #    targeted run *can* reproduce -- the noise is seeded per entry -- and it is
            #    what proves the same patterns are being indexed.
            if benchmark_entries is not None:
                reference = benchmark_entries.loc[
                    benchmark_entries['condition_bundle'] == bundle]
                joined = entries.merge(reference, on='entry_id', how='inner',
                                       suffixes=('', '_ref'))
                # Numerically, not by digest. The pool was generated on x86 and a laptop
                # check runs on arm64, where a contaminant position can land one ULP away
                # -- measured at 1.6e-16 relative on one peak of one entry, which is R9
                # and R13 arriving again. A digest comparison turns that into a hard
                # failure and hides the thing worth knowing, which is the magnitude. The
                # digest agreement is reported beside it and is expected to be exact only
                # when this runs on the same architecture as the pool.
                worst = 0.0
                for mine, theirs in zip(joined['q2_obs'], joined['q2_obs_ref']):
                    mine = np.asarray(mine, dtype=float)
                    theirs = np.asarray(theirs, dtype=float)
                    if mine.shape != theirs.shape:
                        worst = np.inf
                        break
                    worst = max(worst, float(np.max(
                        np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-300))))
                digest_bad = int((joined['q2_digest'] != joined['q2_digest_ref']).sum())
                findings.append({
                    'arm': label, 'bundle': bundle, 'check': 'peak_lists_reproduce_1e-12',
                    'n': len(joined), 'n_bad': digest_bad, 'pass': worst < 1e-12,
                    'detail': f'max relative peak difference {worst:.3e}; '
                              f'{digest_bad} of {len(joined)} digests differ (R9: expected '
                              f'off-architecture, expected 0 on x86)'})

            # 2. n_entering is recorded before the NaN filter and the dump after it, so a
            #    shortfall counts NaN cells rather than indicating a misplaced hook.
            counted = predownsample.groupby(['entry_id', 'bravais_lattice']).agg(
                rows=('candidate_id', 'size'), n_entering=('n_entering', 'first'))
            nan_rows = int((counted['n_entering'] - counted['rows']).sum())
            findings.append({
                'arm': label, 'bundle': bundle, 'check': 'n_entering_minus_rows_is_nan_count',
                'n': int(counted['rows'].sum()), 'n_bad': nan_rows,
                'pass': bool((counted['n_entering'] >= counted['rows']).all()),
                'detail': f'{nan_rows} rows dropped by the NaN filter'})

            # 3. The arm ran at the threshold it claims to have run at.
            with open(Path(bundle_dir) / 'manifest.json', encoding='utf-8') as handle:
                threshold = json.load(handle)['prune_threshold']
            expected = 0.0 if label == 't0' else PRODUCTION_PRUNE_THRESHOLD
            findings.append({
                'arm': label, 'bundle': bundle, 'check': f'prune_threshold_is_{expected}',
                'n': 1, 'n_bad': int(threshold != expected), 'pass': threshold == expected,
                'detail': f'manifest records {threshold}'})

    table = pd.DataFrame(findings)
    out = Path(args.artifact_dir) / f'{args.tag}_gate.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f'\nwrote {out}')
    if not table['pass'].all():
        raise SystemExit('gate FAILED -- do not read the analyse stage until this passes')
    print('gate PASSED')


def run_analyse(args):
    rng = np.random.default_rng(12345)
    prune_rows, dedup_rows, ceiling_rows = [], [], []

    for bundle, bundle_dir in bundle_directories(args.t0_root).items():
        entries = FomBenchmark.load_entries(bundle_dir)
        all_ids = set(entries['entry_id'])
        pre_prune = load_predownsample(bundle_dir)
        pre_prune = FomBenchmark.label_frame(pre_prune, entries, rtol=args.rtol,
                                             n_processes=args.n_processes)
        correct = pre_prune.loc[pre_prune['is_correct'].fillna(False)]

        # ---- Q31 -------------------------------------------------------------------
        would_survive = correct['m20_at_prune'] >= PRODUCTION_PRUNE_THRESHOLD
        prune_rows.append({
            'condition_bundle': bundle,
            'candidates_generated': len(pre_prune),
            'correct_generated': len(correct),
            'correct_surviving_prune': int(would_survive.sum()),
            'correct_deleted_by_prune': int((~would_survive).sum()),
            'fraction_deleted': float((~would_survive).mean()) if len(correct) else np.nan,
            'm20_at_prune_p10': float(correct['m20_at_prune'].quantile(0.10))
            if len(correct) else np.nan,
            'm20_at_prune_median': float(correct['m20_at_prune'].median())
            if len(correct) else np.nan,
            })

        reached_pre = _reachable(pre_prune, entries)
        reached_post = _reachable(pre_prune, entries,
                                  pre_prune['m20_at_prune'] >= PRODUCTION_PRUNE_THRESHOLD)
        for stage, reached in (('pre_prune', reached_pre), ('post_prune', reached_post)):
            value, low, high = _bootstrap_ceiling(reached, all_ids, rng)
            ceiling_rows.append({'condition_bundle': bundle, 'stage': stage,
                                 'n_entries': len(all_ids), 'n_reachable': len(reached),
                                 'ceiling': value, 'ci_low': low, 'ci_high': high})

        # ---- F-065, from the production arm ---------------------------------------
        if args.t5_root is None:
            continue
        production_dir = bundle_directories(args.t5_root).get(bundle)
        if production_dir is None:
            continue
        pre_dedup = FomBenchmark.label_frame(
            load_predownsample(production_dir), FomBenchmark.load_entries(production_dir),
            rtol=args.rtol, n_processes=args.n_processes)
        post_dedup = FomBenchmark.label_frame(
            FomBenchmark.load_candidates(production_dir),
            FomBenchmark.load_entries(production_dir),
            rtol=args.rtol, n_processes=args.n_processes)
        production_entries = FomBenchmark.load_entries(production_dir)
        production_ids = set(production_entries['entry_id'])

        reached_before = _reachable(pre_dedup, production_entries)
        reached_after = _reachable(post_dedup, production_entries)
        dedup_rows.append({
            'condition_bundle': bundle,
            'candidates_entering': len(pre_dedup),
            'candidates_surviving': len(post_dedup),
            'fraction_discarded': 1 - len(post_dedup) / max(1, len(pre_dedup)),
            'correct_entering': int(pre_dedup['is_correct'].fillna(False).sum()),
            'correct_surviving': int(post_dedup['is_correct'].fillna(False).sum()),
            'entries_reachable_before': len(reached_before),
            'entries_reachable_after': len(reached_after),
            'entries_losing_only_correct': len(reached_before - reached_after),
            })
        for stage, reached in (('pre_dedup', reached_before), ('post_dedup', reached_after)):
            value, low, high = _bootstrap_ceiling(reached, production_ids, rng)
            ceiling_rows.append({'condition_bundle': bundle, 'stage': stage,
                                 'n_entries': len(production_ids), 'n_reachable': len(reached),
                                 'ceiling': value, 'ci_low': low, 'ci_high': high})

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prune = pd.DataFrame(prune_rows)
    dedup = pd.DataFrame(dedup_rows)
    ceilings = pd.DataFrame(ceiling_rows)
    prune.to_csv(artifact_dir / f'{args.tag}_prune.csv', index=False)
    dedup.to_csv(artifact_dir / f'{args.tag}_dedup.csv', index=False)
    ceilings.to_csv(artifact_dir / f'{args.tag}_ceilings.csv', index=False)
    print(prune.to_string(index=False))
    print()
    print(dedup.to_string(index=False))
    print()
    print(ceilings.to_string(index=False))
    print(f'\nwrote {args.tag}_{{prune,dedup,ceilings}}.csv to {artifact_dir}')


def main():
    args = _parse_args()
    if args.stage == 'gate':
        run_gate(args)
    else:
        run_analyse(args)


if __name__ == '__main__':
    main()
