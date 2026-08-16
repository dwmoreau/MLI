"""S04 -- consolidate the per-bundle dumps into the frozen benchmark pool, and check the gate.

`run_fom_dump.py` writes one directory per condition bundle, each holding a parquet per (shard,
pool). This joins them into the single pool every downstream step reads, partitioned so that S05
onwards can pull one Bravais lattice at a time instead of the whole 20M-row grid:

    mlindex/data/fom_benchmark/candidates_<bundle>_<BL>.parquet
    mlindex/data/fom_benchmark/entries.parquet
    mlindex/data/fom_benchmark/manifest.json

    python run_fom_dump_consolidate.py --dump-root ../characterization/fom/benchmark \\
        --out-dir ../data/fom_benchmark --artifact-dir ../../docs/fom/artifacts

Flat names rather than the `{split}/candidates_{BL}.parquet` layout S04's handoff describes.
`FomBenchmark._read_glob` globs `candidates_*.parquet` from one root and `load_candidates` already
filters by split through the entry table, so a per-split directory tree would break the loader S03
delivered and gain nothing -- the split is a column, and a candidate's split is a property of its
entry, not of where it sits on disk.

Bundles are processed one at a time and never all held at once: a bundle is ~3.4M candidate rows,
the whole grid ~20M (F-049), and the checks below only ever need one bundle plus the accumulated
counts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.command_line.run import BRAVAIS_LATTICES
from mlindex.model_training import FomBenchmark


# The acceptance gate from S04's handoff.
MIN_CORRECT_PER_STRATUM = 200
# Lattices with too few validation entries to reach it; recorded rather than treated as failures.
SPARSE_LATTICES = ('cP', 'cI', 'cF', 'oF')
# C0 is the control: the idealised success rates the draft reports. If these are not reproduced,
# the generation path differs from the paper's and nothing downstream is trustworthy.
C0_TARGETS = {'monoclinic': 0.99, 'triclinic': 0.96}


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Consolidate the S04 bundle dumps and check the acceptance gate')
    parser.add_argument('--dump-root', type=str, required=True,
                        help='Directory holding one subdirectory per condition bundle')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Where the frozen pool is written')
    parser.add_argument('--artifact-dir', type=str, required=True)
    parser.add_argument('--tag', type=str, default='S04_row_counts')
    parser.add_argument('--rtol', type=float, default=1e-2,
                        help='Correctness tolerance for validate_candidate_known_bl')
    parser.add_argument('--n-processes', type=int, default=1,
                        help='Processes for candidate labelling, which is ~95%% of this script\'s '
                             'cost at ~9 ms/candidate. Serial over a full grid is ~50 h; use the '
                             'node (e.g. 128) and it is minutes')
    parser.add_argument('--no-align-entries', action='store_true',
                        help='Keep every bundle\'s own entry set instead of intersecting them. '
                             'Leaves conditions describing different crystals, so any paired '
                             'comparison must intersect by entry_id itself')
    return parser.parse_args()


def bundle_directories(dump_root):
    """Every subdirectory that actually holds a dump, in a stable order."""
    found = sorted(path for path in Path(dump_root).iterdir()
                   if path.is_dir() and any(path.glob('candidates_*.parquet')))
    if not found:
        raise SystemExit(f'No bundle directories with candidates_*.parquet under {dump_root}')
    return found


def common_entry_ids(bundle_dirs):
    """The entries every bundle has, so all bundles describe the same crystals.

    A condition can lose an entry: add_second_phase raises when a pattern is dense enough that no
    line of its assigned partner clears every peak's half breadth, and the first grid lost 33 of
    5 955 in C6 that way. Nothing else lost any. Left alone, that leaves C6 with a different entry
    set from the other six, and every downstream paired comparison has to remember to intersect --
    which is the kind of thing that is remembered until it is not, and then silently compares
    different crystals across conditions.

    Aligning here costs those 33 entries in the six bundles that did have them, and buys the
    invariant that a bundle is a condition applied to one fixed set of entries.

    Reads only the entry_id column, so this pre-pass is cheap against a 20M-row grid.
    """
    common = None
    per_bundle = {}
    for bundle_dir in bundle_dirs:
        ids = set(FomBenchmark.load_entries_ids(bundle_dir))
        per_bundle[bundle_dir.name] = ids
        common = ids if common is None else (common & ids)
    return common, per_bundle


def consolidate_bundle(bundle_dir, out_dir, rtol, keep_entry_ids=None, n_processes=1):
    """Label one bundle, write it partitioned by Bravais lattice, and return what the gate needs."""
    entries = FomBenchmark.load_entries(bundle_dir)
    candidates = FomBenchmark.load_candidates(bundle_dir)
    if keep_entry_ids is not None:
        entries = entries.loc[entries['entry_id'].isin(keep_entry_ids)].reset_index(drop=True)
        candidates = candidates.loc[
            candidates['entry_id'].isin(keep_entry_ids)].reset_index(drop=True)
    # Raises if any candidate's q2_digest disagrees with its entry: shards from different runs
    # would otherwise join silently, every column parsing and every number on the wrong pattern.
    FomBenchmark._check_join(candidates, entries)
    labelled = FomBenchmark.label_frame(candidates, entries, rtol=rtol,
                                        n_processes=n_processes)

    bundle = entries['condition_bundle'].iloc[0]
    if entries['condition_bundle'].nunique() != 1:
        raise SystemExit(
            f'{bundle_dir} mixes condition bundles: '
            f'{sorted(entries["condition_bundle"].unique())}. One directory per bundle.')

    for bravais_lattice, group in labelled.groupby('bravais_lattice'):
        FomBenchmark.write_candidate_shard(
            group.reset_index(drop=True), out_dir, f'{bundle}_{bravais_lattice}')

    truth = entries.set_index('entry_id')[['bravais_lattice_true', 'lattice_system_true', 'split']]
    counts = labelled.groupby(['bravais_lattice', 'is_correct', 'is_off_by_two']).size(
        ).reset_index(name='n_candidates')
    counts.insert(0, 'condition_bundle', bundle)

    # Per-entry reachability, on the entry's own true lattice. "No correct candidate in the pool"
    # is a generation failure, not a ranking failure, and PROTOCOL section 8 wants them separate.
    own = labelled.merge(truth, left_on='entry_id', right_index=True, how='left')
    own = own.loc[own['bravais_lattice'] == own['bravais_lattice_true']].copy()
    # Combine before grouping rather than reaching back into the frame from inside an agg lambda,
    # which only works while the group's index still aligns with the parent's.
    own['correct_and_kept'] = own['is_correct'].fillna(False) & own['in_top_n']
    reach = own.groupby('entry_id').agg(
        in_pool=('is_correct', 'any'),
        in_top_n=('correct_and_kept', 'any'),
        ).reset_index().merge(truth, left_on='entry_id', right_index=True, how='left')
    reach.insert(0, 'condition_bundle', bundle)

    return bundle, entries, counts, reach


def check_gate(counts, reach, entries):
    """The handoff's three conditions, each reported whether or not it passes."""
    findings = []

    correct = counts.loc[counts['is_correct']].groupby(
        ['condition_bundle', 'bravais_lattice'])['n_candidates'].sum().reset_index()
    short = correct.loc[(correct['n_candidates'] < MIN_CORRECT_PER_STRATUM)
                        & (~correct['bravais_lattice'].isin(SPARSE_LATTICES))]
    findings.append({
        'check': f'>= {MIN_CORRECT_PER_STRATUM} correct candidates per (BL x condition)',
        'passed': bool(short.empty),
        'detail': 'all well-populated strata met the floor' if short.empty else
                  '; '.join(f'{row.bravais_lattice}/{row.condition_bundle}={row.n_candidates}'
                            for row in short.itertuples()),
        })

    overlap = entries.groupby('entry_id')['split'].nunique()
    per_bundle_counts = entries.groupby('condition_bundle')['entry_id'].nunique()
    findings.append({
        'check': 'every bundle covers the same entries',
        'passed': bool(per_bundle_counts.nunique() == 1),
        'detail': f'{int(per_bundle_counts.iloc[0])} entries in each of '
                  f'{per_bundle_counts.shape[0]} bundles' if per_bundle_counts.nunique() == 1
                  else f'counts differ: {per_bundle_counts.to_dict()}',
        })

    findings.append({
        'check': 'splits disjoint by entry_id',
        'passed': bool((overlap <= 1).all()),
        'detail': 'no entry appears in two splits' if (overlap <= 1).all() else
                  f'{int((overlap > 1).sum())} entry_id in more than one split',
        })

    unassigned = int((entries['split'] == 'unassigned').sum())
    findings.append({
        'check': 'every entry carries a split from the frozen S02 manifest',
        'passed': unassigned == 0,
        'detail': 'all assigned' if unassigned == 0 else
                  f'{unassigned} entries fell through to "unassigned"',
        })

    control = reach.loc[reach['condition_bundle'].str.startswith('error0_cont0')]
    if control.empty:
        findings.append({
            'check': 'C0 control reproduces the draft idealised success rates',
            'passed': False,
            'detail': 'no error0_cont0 bundle present, so the control was not run',
            })
    else:
        rates = control.groupby('lattice_system_true')['in_pool'].mean()
        misses = {system: float(rates.get(system, float('nan')))
                  for system, target in C0_TARGETS.items()
                  if not rates.get(system, 0.0) >= target}
        findings.append({
            'check': 'C0 control reproduces the draft idealised success rates',
            'passed': not misses,
            'detail': ('; '.join(f'{system} {rates.get(system, float("nan")):.3f} '
                                 f'(target {C0_TARGETS[system]})'
                                 for system in C0_TARGETS)),
            })
    return pd.DataFrame(findings)


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle_dirs = bundle_directories(args.dump_root)
    keep_entry_ids, dropped_per_bundle = None, {}
    if not args.no_align_entries:
        keep_entry_ids, per_bundle = common_entry_ids(bundle_dirs)
        dropped_per_bundle = {name: len(ids) - len(keep_entry_ids)
                              for name, ids in per_bundle.items()}
        total_dropped = sum(dropped_per_bundle.values())
        print(f'aligning entry sets: {len(keep_entry_ids)} entries common to all '
              f'{len(bundle_dirs)} bundles', flush=True)
        if total_dropped:
            for name, dropped in sorted(dropped_per_bundle.items()):
                if dropped:
                    print(f'  {name}: dropping {dropped} entries absent from another bundle',
                          flush=True)
            print(f'  {total_dropped} entry-rows dropped in total, so every condition now '
                  'describes the same crystals', flush=True)

    all_entries, all_counts, all_reach, manifests = [], [], [], {}
    for bundle_dir in bundle_dirs:
        bundle, entries, counts, reach = consolidate_bundle(
            bundle_dir, out_dir, args.rtol, keep_entry_ids=keep_entry_ids,
            n_processes=args.n_processes)
        all_entries.append(entries)
        all_counts.append(counts)
        all_reach.append(reach)
        manifest_path = bundle_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, encoding='utf-8') as manifest_file:
                manifests[bundle] = json.load(manifest_file)
        print(f'{bundle}: {entries.shape[0]} entries, '
              f'{int(counts["n_candidates"].sum())} candidates', flush=True)

    entries = pd.concat(all_entries, ignore_index=True)
    counts = pd.concat(all_counts, ignore_index=True)
    reach = pd.concat(all_reach, ignore_index=True)
    FomBenchmark._to_parquet(entries, out_dir / 'entries.parquet')

    gate = check_gate(counts, reach, entries)
    FomBenchmark.write_manifest(
        out_dir,
        consolidated_from=[path.name for path in bundle_directories(args.dump_root)],
        bundles=sorted(manifests),
        per_bundle=manifests,
        n_entries=int(entries.shape[0]),
        n_entry_rows_per_bundle=entries.groupby('condition_bundle').size().to_dict(),
        n_candidates=int(counts['n_candidates'].sum()),
        rtol=args.rtol,
        entries_aligned=not args.no_align_entries,
        n_entries_common=None if keep_entry_ids is None else len(keep_entry_ids),
        n_entries_dropped_to_align=dropped_per_bundle,
        gate_passed=bool(gate['passed'].all()),
        )

    counts.to_csv(artifact_dir / f'{args.tag}.csv', index=False)
    reach.to_csv(artifact_dir / f'{args.tag}_reachability.csv', index=False)

    by_stratum = counts.pivot_table(
        index='bravais_lattice', columns='condition_bundle', values='n_candidates',
        aggfunc='sum', fill_value=0).reindex(BRAVAIS_LATTICES)
    correct_by_stratum = counts.loc[counts['is_correct']].pivot_table(
        index='bravais_lattice', columns='condition_bundle', values='n_candidates',
        aggfunc='sum', fill_value=0).reindex(BRAVAIS_LATTICES).fillna(0).astype(int)
    ceiling = reach.pivot_table(index='bravais_lattice_true', columns='condition_bundle',
                                values='in_pool', aggfunc='mean').reindex(BRAVAIS_LATTICES)

    lines = [
        f'# S04 benchmark pool: {"GATE PASSED" if gate["passed"].all() else "GATE FAILED"}',
        '',
        f'- entries: **{entries.shape[0]}** over {entries["condition_bundle"].nunique()} bundles',
        f'- candidates: **{int(counts["n_candidates"].sum())}**',
        f'- correctness tolerance: rtol {args.rtol}',
        '',
        '## Gate',
        '',
        gate.to_markdown(index=False),
        '',
        '## Correct candidates per (Bravais lattice x condition)',
        '',
        f'The floor is {MIN_CORRECT_PER_STRATUM}. '
        f'{", ".join(SPARSE_LATTICES)} are recorded rather than failed -- the frozen manifest '
        'caps them at 321/156/106/500 entries, so they may need the S04b scale-up or a '
        'merged-lattice treatment.',
        '',
        correct_by_stratum.to_markdown(),
        '',
        '## All candidates per (Bravais lattice x condition)',
        '',
        by_stratum.to_markdown(),
        '',
        '## Ceiling: fraction of entries with a correct candidate anywhere in their own lattice pool',
        '',
        'A generation statistic, not a ranking one. An entry missing here cannot be recovered by '
        'any figure of merit, so it bounds everything S06 onwards can claim.',
        '',
        ceiling.round(3).to_markdown(),
        ]
    with open(artifact_dir / f'{args.tag}.md', 'w', encoding='utf-8') as report:
        report.write('\n'.join(lines) + '\n')

    print()
    print(gate.to_string(index=False))
    print(f'\nwrote {out_dir} and {artifact_dir / (args.tag + ".md")}')
    raise SystemExit(0 if gate['passed'].all() else 1)


if __name__ == '__main__':
    main()
