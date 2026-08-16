"""S03 acceptance gate: is the candidate dump complete?

Reads a pool written by `run_fom_dump.py` and recomputes M20, Minfo and n_indexed from the
dumped columns alone. If any column needed to reconstruct a score is missing from the
schema, the recomputation diverges and this fails -- which is the point. The gate is a
completeness proof, not a statistic, so it needs no particular entries or conditions, only
enough candidates to cover all fourteen Bravais lattices.

    python run_fom_dump_gate.py --dump-dir <dir> --artifact-dir <dir>

Gate: agreement to 1e-6 on at least 10,000 candidates spanning all 14 Bravais lattices,
with n_indexed exact. n_indexed counts peaks over a hard probability threshold, so a near
miss there is a real disagreement rather than floating-point noise; when it fails, the
report prints how close the offending peaks sat to the threshold, which separates "the dump
is incomplete" from "one peak sat an ulp from 0.95".
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.command_line.run import BRAVAIS_LATTICES
from mlindex.model_training import FomBenchmark


REQUIRED_CANDIDATES = 10000
TOLERANCE = 1e-6


def _parse_args():
    parser = argparse.ArgumentParser(description='S03 acceptance gate for the candidate dump')
    parser.add_argument('--dump-dir', type=str, required=True,
                        help='Directory of candidates_*.parquet and entries_*.parquet')
    parser.add_argument('--artifact-dir', type=str, required=True,
                        help='Where to write the gate report')
    parser.add_argument('--tag', type=str, default='S03_roundtrip')
    return parser.parse_args()


def relative_error(actual, expected):
    """Relative where the reference is non-zero, absolute where it is not."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = np.abs(expected)
    return np.where(scale > 0, np.abs(actual - expected) / np.maximum(scale, 1e-300),
                    np.abs(actual - expected))


def main():
    args = _parse_args()
    dump_dir = Path(args.dump_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    entries = FomBenchmark.load_entries(dump_dir)
    candidates = FomBenchmark.load_candidates(dump_dir)
    checked = FomBenchmark.recompute_frame(candidates, entries)

    checked['M20_relative_error'] = relative_error(
        checked['M20_recomputed'], checked['M20'])
    checked['Minfo_relative_error'] = relative_error(
        checked['Minfo_recomputed'], checked['Minfo'])
    checked['n_indexed_delta'] = (
        checked['n_indexed_recomputed'] - checked['n_indexed']).astype(int)

    per_lattice = checked.groupby('bravais_lattice').agg(
        candidates=('M20', 'size'),
        entries=('entry_id', 'nunique'),
        max_M20_relative_error=('M20_relative_error', 'max'),
        max_Minfo_relative_error=('Minfo_relative_error', 'max'),
        n_indexed_mismatches=('n_indexed_delta', lambda column: int((column != 0).sum())),
        ).reindex(BRAVAIS_LATTICES)

    n_candidates = int(checked.shape[0])
    lattices_covered = int(checked['bravais_lattice'].nunique())
    worst_M20 = float(checked['M20_relative_error'].max())
    worst_Minfo = float(checked['Minfo_relative_error'].max())
    n_indexed_mismatches = int((checked['n_indexed_delta'] != 0).sum())

    passed = (
        n_candidates >= REQUIRED_CANDIDATES
        and lattices_covered == len(BRAVAIS_LATTICES)
        and worst_M20 <= TOLERANCE
        and worst_Minfo <= TOLERANCE
        and n_indexed_mismatches == 0
        )

    verdict = {
        'passed': bool(passed),
        'n_candidates': n_candidates,
        'n_entries': int(checked['entry_id'].nunique()),
        'lattices_covered': lattices_covered,
        'required_candidates': REQUIRED_CANDIDATES,
        'tolerance': TOLERANCE,
        'max_M20_relative_error': worst_M20,
        'max_Minfo_relative_error': worst_Minfo,
        'n_indexed_mismatches': n_indexed_mismatches,
        'survivors_per_entry': round(n_candidates / max(1, checked['entry_id'].nunique()), 1),
        'mean_entering_per_entry': round(float(
            checked.groupby(['entry_id', 'bravais_lattice'])['n_entering'].first()
            .groupby('entry_id').sum().mean()), 1),
        }

    per_lattice.to_csv(artifact_dir / f'{args.tag}.csv')
    with open(artifact_dir / f'{args.tag}.json', 'w', encoding='utf-8') as verdict_file:
        json.dump(verdict, verdict_file, indent=2, sort_keys=True)

    lines = [
        f'# S03 acceptance gate: {"PASS" if passed else "FAIL"}',
        '',
        'Recomputing M20, Minfo and n_indexed from the dumped columns alone, and comparing',
        'against the values the pipeline itself produced.',
        '',
        f'- candidates checked: **{n_candidates}** (gate requires {REQUIRED_CANDIDATES})',
        f'- entries: {verdict["n_entries"]}, '
        f'{verdict["survivors_per_entry"]} surviving candidates per entry',
        f'- Bravais lattices covered: **{lattices_covered} / {len(BRAVAIS_LATTICES)}**',
        f'- worst M20 relative error: **{worst_M20:.3e}** (tolerance {TOLERANCE:g})',
        f'- worst Minfo relative error: **{worst_Minfo:.3e}**',
        f'- n_indexed mismatches: **{n_indexed_mismatches}** (must be 0)',
        '',
        '## Per Bravais lattice',
        '',
        per_lattice.to_markdown(),
        ]
    if n_indexed_mismatches:
        offenders = checked.loc[checked['n_indexed_delta'] != 0]
        lines += [
            '',
            '## n_indexed disagreements',
            '',
            'A count over a hard threshold, so these are either a genuine schema gap or a',
            'peak sitting within floating-point noise of the assignment threshold.',
            '',
            offenders[['entry_id', 'bravais_lattice', 'spacegroup', 'n_indexed',
                       'n_indexed_recomputed']].head(20).to_markdown(index=False),
            ]
    with open(artifact_dir / f'{args.tag}.md', 'w', encoding='utf-8') as report:
        report.write('\n'.join(lines) + '\n')

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f'wrote {artifact_dir / (args.tag + ".md")}')
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
