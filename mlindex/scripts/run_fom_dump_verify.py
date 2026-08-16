"""S04 -- did the grid actually run properly? One command, four layers, cheapest first.

    python run_fom_dump_verify.py --dump-root ../characterization/fom/benchmark \\
        --artifact-dir ../../docs/fom/artifacts --out-dir ../data/fom_benchmark \\
        --slurm-job <arrayjobid>

Each layer is a superset of the doubt the previous one leaves. The order matters because the
expensive layers are meaningless if a cheap one fails -- a round trip over a bundle that is missing
half its pools proves only that the half present is self-consistent.

    1. SLURM      every array task COMPLETED, no ABORTED pool, no failure files
    2. structure  every bundle has all its pool tables AND its manifest.json, which is written only
                  after all pools join and is therefore the real completion marker
    3. content    5 955 entries per bundle carrying the frozen split, nothing 'unassigned'
    4. gate       the S03 round trip per bundle, then consolidation and the acceptance gate

Stops at the first failing layer unless --keep-going. Exits non-zero if anything failed, so it can
gate a script.

Layer 1 is skipped when sacct is unavailable, which is what happens off a login node; that is
reported rather than silently passed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


# S02's frozen manifest: 5 955 entries split 3575/1197/1183 by source entry. Read from the manifest
# rather than hardcoded, so this stays correct if the split is ever refrozen.
EXPECTED_BUNDLES = 7


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Verify an S04 grid ran properly, from SLURM state down to the acceptance gate')
    parser.add_argument('--dump-root', type=str, required=True,
                        help='Directory holding one subdirectory per condition bundle')
    parser.add_argument('--artifact-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str,
                        help='Consolidated pool destination. Omit to skip layer 4b')
    parser.add_argument('--split-manifest', type=str,
                        help='Frozen S02 manifest, to check every entry survived. '
                             'Defaults to the path run_fom_dump.py uses')
    parser.add_argument('--slurm-job', type=str,
                        help='Array job id, for the sacct check in layer 1')
    parser.add_argument('--slurm-out-dir', type=str, default='.',
                        help='Where the slurm-*.out files are, for the ABORTED scan')
    parser.add_argument('--expected-bundles', type=int, default=EXPECTED_BUNDLES)
    parser.add_argument('--skip-gate', action='store_true',
                        help='Structure and content only; skip the round trip and consolidation')
    parser.add_argument('--keep-going', action='store_true',
                        help='Run every layer even after one fails')
    return parser.parse_args()


class Report:
    """Accumulates pass/fail so every layer's verdict survives to one summary at the end."""

    def __init__(self):
        self.rows = []

    def add(self, layer, check, passed, detail=''):
        self.rows.append({'layer': layer, 'check': check, 'passed': passed, 'detail': detail})
        mark = 'PASS' if passed else 'FAIL'
        print(f'  [{mark}] {check}' + (f' -- {detail}' if detail else ''), flush=True)
        return passed

    def layer_passed(self, layer):
        rows = [row for row in self.rows if row['layer'] == layer]
        return all(row['passed'] for row in rows) if rows else True

    def failed(self):
        return [row for row in self.rows if not row['passed']]


def bundle_directories(dump_root):
    return sorted(path for path in Path(dump_root).iterdir()
                  if path.is_dir() and any(path.glob('candidates_*.parquet')))


def layer_slurm(args, report):
    """Did the scheduler think every task finished, and did any pool abort inside one?"""
    print('\nLayer 1 -- SLURM')
    if args.slurm_job:
        try:
            output = subprocess.check_output(
                ['sacct', '-j', args.slurm_job, '--noheader', '-P',
                 '--format=JobID,State,ExitCode,Elapsed,MaxRSS'],
                text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError) as error:
            report.add(1, 'sacct reachable', True,
                       f'skipped: {type(error).__name__}. Off a login node this is expected; '
                       'check the job state by hand')
            output = ''
        if output:
            # Only the top-level array tasks; .batch and .extern steps repeat their state.
            tasks = [line.split('|') for line in output.strip().splitlines()
                     if '.batch' not in line.split('|')[0]
                     and '.extern' not in line.split('|')[0]]
            bad = [task for task in tasks if task[1].strip() != 'COMPLETED']
            report.add(1, f'{len(tasks)} array tasks COMPLETED', not bad,
                       'all completed' if not bad else
                       '; '.join(f'{task[0]}={task[1]}' for task in bad))
            for task in tasks:
                print(f'        {task[0]:<20} {task[1]:<12} exit={task[2]:<6} '
                      f'elapsed={task[3]:<10} maxrss={task[4]}')
    else:
        report.add(1, 'sacct check', True, 'skipped: no --slurm-job given')

    # An aborted pool still writes its rows, so its bundle looks structurally fine. The log is the
    # only place that says so.
    logs = sorted(Path(args.slurm_out_dir).glob('slurm-*.out'))
    if not logs:
        report.add(1, 'slurm-*.out scanned for ABORTED pools', True,
                   f'skipped: no slurm-*.out under {args.slurm_out_dir}')
    else:
        aborted, failures = [], []
        for log in logs:
            text = log.read_text(encoding='utf-8', errors='replace')
            if 'ABORTED' in text:
                aborted.append(log.name)
            for line in text.splitlines():
                if 'failures by reason' in line:
                    failures.append(f'{log.name}: {line.strip()}')
        report.add(1, f'no aborted pool in {len(logs)} logs', not aborted,
                   'none' if not aborted else ', '.join(aborted))
        report.add(1, 'no per-entry failures', not failures,
                   'none' if not failures else ' | '.join(failures[:5]))


def layer_structure(args, report):
    """Every bundle complete, judged by manifest.json rather than by file count alone."""
    print('\nLayer 2 -- structure')
    bundles = bundle_directories(args.dump_root)
    report.add(2, f'{args.expected_bundles} bundle directories present',
               len(bundles) == args.expected_bundles,
               f'found {len(bundles)}: {[path.name for path in bundles]}')

    for bundle in bundles:
        manifest_path = bundle / 'manifest.json'
        n_candidates = len(list(bundle.glob('candidates_*.parquet')))
        n_entries = len(list(bundle.glob('entries_*.parquet')))
        n_failures = len(list(bundle.glob('failures_*.json')))

        if not manifest_path.exists():
            # Written only after every pool joins, so its absence means the task did not finish
            # even if SLURM reported COMPLETED.
            report.add(2, f'{bundle.name}: manifest.json', False,
                       f'MISSING -- task did not finish; {n_candidates} candidate shards on disk')
            continue
        with open(manifest_path, encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
        expected_pools = manifest.get('n_pools')
        ok = (n_candidates == n_entries == expected_pools)
        report.add(2, f'{bundle.name}: {expected_pools} pools wrote both tables', ok,
                   f'candidates={n_candidates} entries={n_entries}'
                   + (f' failure_files={n_failures}' if n_failures else ''))
        if n_failures:
            reasons = {}
            for path in bundle.glob('failures_*.json'):
                with open(path, encoding='utf-8') as failure_file:
                    for failure in json.load(failure_file):
                        reasons[failure['reason']] = reasons.get(failure['reason'], 0) + 1
            report.add(2, f'{bundle.name}: no dropped entries', False, f'{reasons}')


def layer_content(args, report):
    """The entries are all there and carry the split S02 froze."""
    from mlindex.model_training import FomBenchmark

    print('\nLayer 3 -- content')
    manifest_path = args.split_manifest
    if manifest_path is None:
        import run_fom_dump
        manifest_path = str(run_fom_dump.MANIFEST_PATH)
    expected_ids, expected_splits = None, None
    if Path(manifest_path).exists():
        frozen = pd.read_parquet(manifest_path)
        expected_ids = set(frozen['identifier'])
        expected_splits = frozen['split'].value_counts().to_dict()
        report.add(3, 'frozen split manifest readable', True,
                   f'{len(expected_ids)} entries, {expected_splits}')
    else:
        report.add(3, 'frozen split manifest readable', False,
                   f'not found at {manifest_path}; entry coverage cannot be checked')

    for bundle in bundle_directories(args.dump_root):
        entries = FomBenchmark.load_entries(bundle)
        unassigned = int((entries['split'] == 'unassigned').sum())
        report.add(3, f'{bundle.name}: no unassigned split', unassigned == 0,
                   f'{entries.shape[0]} entries' if unassigned == 0
                   else f'{unassigned} entries carry no split -- the pool is unusable downstream')
        if expected_ids is not None:
            missing = expected_ids - set(entries['entry_id'])
            report.add(3, f'{bundle.name}: all {len(expected_ids)} manifest entries present',
                       not missing,
                       'complete' if not missing
                       else f'{len(missing)} missing, e.g. {sorted(missing)[:3]}')
            got = entries['split'].value_counts().to_dict()
            report.add(3, f'{bundle.name}: split counts match the frozen manifest',
                       got == expected_splits, f'{got}')
        duplicated = int(entries['entry_id'].duplicated().sum())
        report.add(3, f'{bundle.name}: no duplicated entry_id', duplicated == 0,
                   'none' if duplicated == 0
                   else f'{duplicated} duplicates -- pools overlapped, or a shard was written twice')


def layer_gate(args, report):
    """The S03 round trip per bundle, then consolidation and the acceptance gate."""
    print('\nLayer 4 -- round trip and acceptance gate')
    here = Path(__file__).resolve().parent
    for bundle in bundle_directories(args.dump_root):
        result = subprocess.run(
            [sys.executable, str(here / 'run_fom_dump_gate.py'),
             '--dump-dir', str(bundle), '--artifact-dir', args.artifact_dir,
             '--tag', f'S04_roundtrip_{bundle.name}'],
            capture_output=True, text=True)
        detail = ''
        try:
            verdict = json.loads(result.stdout[result.stdout.index('{'):
                                               result.stdout.rindex('}') + 1])
            detail = (f'{verdict["n_candidates"]} candidates, '
                      f'{verdict["lattices_covered"]}/14 lattices, '
                      f'max rel err {verdict["max_M20_relative_error"]:.1e}, '
                      f'{verdict["n_indexed_mismatches"]} n_indexed mismatches')
        except (ValueError, KeyError):
            detail = (result.stderr.strip().splitlines() or ['no output'])[-1]
        report.add(4, f'{bundle.name}: round trip', result.returncode == 0, detail)

    if not args.out_dir:
        report.add(4, 'consolidation', True, 'skipped: no --out-dir given')
        return
    result = subprocess.run(
        [sys.executable, str(here / 'run_fom_dump_consolidate.py'),
         '--dump-root', args.dump_root, '--out-dir', args.out_dir,
         '--artifact-dir', args.artifact_dir],
        capture_output=True, text=True)
    print(result.stdout)
    # Consolidation prints its own gate table to stdout, echoed above, and exits non-zero when any
    # check fails. Its stderr is empty in that case, so quoting the last stderr line would report
    # a bare 'failed' over a table that says exactly what went wrong.
    stderr_tail = result.stderr.strip().splitlines()
    report.add(4, 'consolidation and acceptance gate', result.returncode == 0,
               'see the gate table above and S04_row_counts.md' if result.returncode == 0
               else (stderr_tail[-1] if stderr_tail else
                     'one or more acceptance checks failed -- see the gate table above'))


def main():
    args = _parse_args()
    report = Report()

    layer_slurm(args, report)
    for layer, run in ((1, None), (2, layer_structure), (3, layer_content),
                       (4, None if args.skip_gate else layer_gate)):
        if run is None:
            continue
        if not report.layer_passed(layer - 1) and not args.keep_going:
            print(f'\nLayer {layer - 1} failed; stopping. Fix it before the later layers mean '
                  'anything, or pass --keep-going.')
            break
        run(args, report)

    failures = report.failed()
    print('\n' + '=' * 88)
    if failures:
        print(f'FAILED -- {len(failures)} check(s):')
        for row in failures:
            print(f'  layer {row["layer"]}: {row["check"]} -- {row["detail"]}')
    else:
        print('PASSED -- every check')
    print('=' * 88)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
