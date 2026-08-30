"""The dump driver's split, entry-list, resume and labelling wiring.

Everything pinned here is silent when it fails, which is why it is pinned.

* The manifest is the ENTRY LIST, not merely a split lookup. `sample_entries` draws
  `rng.choice(size=n, replace=False)` and a draw of 3 000 is not a superset of a draw of 1 400, so
  no `--n-entries-per-bl` reproduces a manifest whose per-lattice counts run 106 to 3 000
  (C2-F-048). The driver used to sample and then intersect, which generated a quietly different
  benchmark whenever a sampling parameter drifted.
* `volume_decile` is READ from the frozen manifest and never recomputed (R14).
* A pool must not be subsampled unlabelled: the retention rule keeps every correct candidate, so
  with no `is_correct` column it would delete the entire signal at a base rate under 1 % and leave
  a pool indistinguishable from a generation failure.
* The manifest records what the run DID, not what it was asked to do.
* Resume must treat a truncated shard as absent. A pool killed mid-write leaves the entry table
  complete and the candidate shard unreadable, and resuming onto that loses its candidates.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.scripts.run_fom_dump import (_parse_args, _pool_complete, entries_from_manifest,
                                          label_and_subsample, load_manifest, manifest_sha256,
                                          preflight)

RIGHT = np.pi / 2


def _manifest_frame():
    return pd.DataFrame({
        'identifier': ['AAA', 'BBB', 'CCC'],
        'bravais_lattice': ['aP', 'mP', 'aP'],
        'volume_true': [100.0, 200.0, 300.0],
        'volume_decile': [0, 5, 9],
        'split': ['fom-train', 'fom-dev', 'fom-test'],
        'arm': ['core', 'core+mechanism', 'core'],
        })


def test_load_manifest_returns_the_frozen_decile(tmp_path):
    path = tmp_path / 'manifest.parquet'
    _manifest_frame().to_parquet(path, index=False)
    manifest = load_manifest(str(path))
    assert list(manifest.index) == ['AAA', 'BBB', 'CCC']
    assert int(manifest.loc['CCC', 'volume_decile']) == 9
    assert manifest.loc['BBB', 'split'] == 'fom-dev'


def test_a_manifest_without_the_decile_is_refused(tmp_path):
    # Loudly, because the alternative is a benchmark whose volume stratification silently comes
    # from whatever row set each analysis happened to hold -- which is R14 exactly.
    path = tmp_path / 'manifest.parquet'
    _manifest_frame().drop(columns=['volume_decile']).to_parquet(path, index=False)
    with pytest.raises(SystemExit, match='volume_decile'):
        load_manifest(str(path))


def test_a_missing_manifest_is_refused_rather_than_invented(tmp_path):
    with pytest.raises(SystemExit, match='must not be'):
        load_manifest(str(tmp_path / 'absent.parquet'))
    assert load_manifest(None) is None


def test_the_manifest_checksum_is_recorded_so_a_stale_copy_is_detectable(tmp_path):
    # `docs/` is git-ignored and reaches NERSC only through sync_record.sh push, so nothing else
    # would notice a stale split manifest at the far end.
    path = tmp_path / 'manifest.parquet'
    _manifest_frame().to_parquet(path, index=False)
    digest = manifest_sha256(str(path))
    assert isinstance(digest, str) and len(digest) == 64
    assert manifest_sha256(None) is None


def test_the_frozen_split_manifest_still_has_the_checksum_the_record_quotes():
    # The value S06 froze and every S07 runbook step checks against. If this fails, either the
    # manifest was regenerated -- which silently invalidates every result that cites it -- or the
    # file is damaged.
    path = os.path.join('docs', 'fom_campaign2', 'artifacts', 'S06_split_manifest.parquet')
    if not os.path.exists(path):
        pytest.skip('docs/ is git-ignored and absent on this machine')
    assert manifest_sha256(path) == (
        '3dd52c5eb2546dacca3034ebd2fd052dcd2acd4a8f9af24ce972fe4e0a210969')


def test_entries_are_read_from_the_manifest_and_a_partial_read_is_refused(tmp_path, monkeypatch):
    from mlindex.model_training import FomPatterns

    source = pd.DataFrame({
        'identifier': ['AAA', 'CCC', 'ZZZ'],
        'database': ['csd', 'csd', 'cod'],
        'bravais_lattice': ['aP', 'aP', 'aP'],
        })
    monkeypatch.setattr(FomPatterns, 'DATASET_DIRECTORY', tmp_path)
    source.to_parquet(tmp_path / 'dataset_aP.parquet', index=False)

    manifest = load_manifest_frame(_manifest_frame())
    entries = entries_from_manifest(manifest[manifest['bravais_lattice'] == 'aP'], ['aP'],
                                    ['identifier', 'database', 'bravais_lattice'])
    # Exactly the manifest's aP entries, in a stable order, and nothing else from the source.
    assert entries['identifier'].tolist() == ['AAA', 'CCC']

    # And a manifest entry the source does not carry is an abort, not a smaller pool.
    source.iloc[:1].to_parquet(tmp_path / 'dataset_aP.parquet', index=False)
    with pytest.raises(SystemExit, match='manifest entries are not in'):
        entries_from_manifest(manifest[manifest['bravais_lattice'] == 'aP'], ['aP'],
                              ['identifier', 'database', 'bravais_lattice'])


def load_manifest_frame(frame):
    return frame.set_index('identifier')


def test_the_subsampling_defaults_are_the_measured_ones():
    args = _parse_args([])
    # C2-F-051: 500 retained 35 % of the pool and advertised subsampling while doing almost none.
    assert args.top_k == 200
    assert args.negative_rate == 0.05
    assert args.no_subsample is False
    assert args.no_label is False
    assert args.prune_threshold == 1.5
    assert args.n_pools == 1 and args.n_shards == 1


def test_print_tag_does_not_require_an_output_directory():
    # The submit script calls --print-tag to learn which directory a bundle should be written to,
    # so requiring the directory first is circular. One manifest.json per --out-dir means two
    # bundles sharing a directory overwrite each other's.
    args = _parse_args(['--condition', 'sparse4', '--print-tag'])
    assert args.print_tag and args.out_dir is None


def test_an_unlabelled_run_that_would_subsample_is_refused_before_the_search():
    # A guard that fires at the end costs the whole bundle. Campaign 1 added abort-safety after
    # losing a run near the end of a 2.5 h bundle; a late refusal reintroduces the same loss.
    with pytest.raises(SystemExit, match='unlabelled'):
        preflight(_parse_args(['--out-dir', '/tmp/unused', '--no-label']))
    # Silent when the configuration is possible: labelling on, or subsampling off.
    assert preflight(_parse_args(['--out-dir', '/tmp/unused'])) is None
    assert preflight(_parse_args(['--out-dir', '/tmp/unused', '--no-label',
                                  '--no-subsample'])) is None


def test_an_out_of_range_shard_is_refused():
    with pytest.raises(SystemExit, match='--shard must be in'):
        preflight(_parse_args(['--out-dir', '/tmp/unused', '--shard', '4', '--n-shards', '4']))


def _pool_and_entries():
    """A pool the labeller can handle without models: cells and truth only."""
    hkl_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int16).reshape(-1)
    entry_rows = [{
        'entry_id': 'E1', 'q2_digest': 'aaaa', 'condition_bundle': 'c2_error1_cont0',
        'unit_cell_true': np.array([8.0, 11.0, 14.0, RIGHT, RIGHT, RIGHT]),
        'volume_true': 1232.0, 'bravais_lattice_true': 'oP',
        'lattice_system_true': 'orthorhombic', 'hkl_true': hkl_true,
        }]
    n = 40
    candidates = pd.DataFrame({
        'entry_id': ['E1'] * n,
        'q2_digest': ['aaaa'] * n,
        'condition_bundle': ['c2_error1_cont0'] * n,
        'bravais_lattice': ['oP'] * n,
        'lattice_system': ['orthorhombic'] * n,
        'candidate_id': np.arange(n),
        'M20': np.linspace(1.0, 9.0, n),
        'volume': np.linspace(500.0, 2000.0, n),
        })
    # One genuinely correct candidate, so the retention rule has a positive to protect.
    cells = [np.array([5.0 + 0.3 * i, 6.0 + 0.2 * i, 30.0]) for i in range(n)]
    cells[7] = np.array([8.0, 11.0, 14.0])
    candidates['unit_cell'] = cells
    candidates['xnn'] = [np.array([1 / c[0] ** 2, 1 / c[1] ** 2, 1 / c[2] ** 2]) for c in cells]
    return candidates, entry_rows


def test_no_subsample_labels_and_keeps_every_row_with_the_bookkeeping_columns():
    # Keeping everything is not a no-op: the bookkeeping columns are still written, so a pool
    # generated whole reads through exactly the same loader as a thinned one -- which is what the
    # held-back fully-retained shard the weight check needs is generated with.
    candidates, entry_rows = _pool_and_entries()
    args = _parse_args(['--out-dir', '/tmp/unused', '--no-subsample'])
    result, subsampled = label_and_subsample(candidates, entry_rows, args, 0)
    assert subsampled is False
    assert result.shape[0] == candidates.shape[0]
    assert set(result['retained_reason']) <= {'correct', 'top_k', 'sampled'}
    assert (result['sampling_weight'] == 1.0).all()
    assert int(result['is_correct'].sum()) == 1, 'the constructed positive was not labelled'
    assert result.loc[result['is_correct'], 'retained_reason'].tolist() == ['correct']


def test_labelling_happens_even_when_the_pool_is_kept_whole():
    # SCHEMA.md's rule is that labels are written at generation, never on load -- campaign 1
    # labelled a 57.4 M-row dump on every analysis pass and persisted nothing (R24).
    from mlindex.model_training import FomBenchmark

    candidates, entry_rows = _pool_and_entries()
    args = _parse_args(['--out-dir', '/tmp/unused', '--no-subsample'])
    result, _ = label_and_subsample(candidates, entry_rows, args, 0)
    assert FomBenchmark.has_labels(result)


def test_an_unlabelled_pool_passes_through_when_labelling_is_off():
    candidates, entry_rows = _pool_and_entries()
    args = _parse_args(['--out-dir', '/tmp/unused', '--no-label', '--no-subsample'])
    result, subsampled = label_and_subsample(candidates, entry_rows, args, 0)
    assert subsampled is False
    assert 'is_correct' not in result.columns


def test_resume_skips_a_complete_pool_and_regenerates_a_truncated_one(tmp_path):
    frame = pd.DataFrame({'entry_id': ['E1', 'E2']})
    for name in ('entries', 'candidates'):
        frame.to_parquet(tmp_path / f'{name}_tag_pool00.parquet', index=False)
    assert _pool_complete(tmp_path, 'tag_pool00', want_predownsample=False)

    # Asked for the pre-deduplication stream and it is absent: not complete. Without this, a task
    # requeued after --predownsample-entries changed would skip pools holding only survivors.
    assert not _pool_complete(tmp_path, 'tag_pool00', want_predownsample=True)

    # Present but truncated by a kill mid-write. Existence is not enough.
    (tmp_path / 'candidates_tag_pool00.parquet').write_bytes(b'PAR1 truncated')
    assert not _pool_complete(tmp_path, 'tag_pool00', want_predownsample=False)


def test_a_missing_pool_is_not_complete(tmp_path):
    assert not _pool_complete(tmp_path, 'absent_pool00', want_predownsample=False)


# ------------------------------------------------------------------------------------------
# The submit script's task table.
#
# It is bash, so nothing else checks it, and a bundle silently missing from the array is a hole in
# the benchmark that only shows up at consolidation -- after the node-hours are spent. Campaign 1
# kept a partial copy of the tag rule in bash which omitted two components; this pins the one
# thing bash still owns, which is the list of what runs.
# ------------------------------------------------------------------------------------------

SUBMIT_PATH = os.path.join('mlindex', 'scripts', 'submit_fom_dump.sh')


def _submit_tasks():
    import re
    with open(SUBMIT_PATH, encoding='utf-8') as handle:
        text = handle.read()
    body = re.search(r'TASKS=\((.*?)\n\)', text, re.S).group(1)
    return [row.split() for row in re.findall(r'"([^"]+)"', body)], text


@pytest.mark.skipif(not os.path.exists(SUBMIT_PATH), reason='submit script not present')
def test_the_array_covers_every_condition_bundle_exactly_once_per_shard():
    from mlindex.model_training import FomConditions

    tasks, text = _submit_tasks()
    conditions = {task[0] for task in tasks}
    assert conditions == {condition.key for condition in FomConditions.CONDITIONS}, (
        'the array does not run every condition bundle; a missing one is a hole in the '
        'benchmark that shows up only at consolidation')

    # Each bundle's shards must be the complete 0..n-1, or the bundle is generated with a gap.
    by_condition = {}
    for condition, _arm, shard, n_shards in tasks:
        by_condition.setdefault(condition, []).append((int(shard), int(n_shards)))
    for condition, shards in by_condition.items():
        n_shards = {n for _, n in shards}
        assert len(n_shards) == 1, f'{condition} mixes shard counts: {n_shards}'
        total = n_shards.pop()
        assert sorted(s for s, _ in shards) == list(range(total)), (
            f'{condition} does not cover shards 0..{total - 1}')


@pytest.mark.skipif(not os.path.exists(SUBMIT_PATH), reason='submit script not present')
def test_the_array_directive_matches_the_task_table():
    import re

    tasks, text = _submit_tasks()
    directive = re.search(r'#SBATCH --array=(\d+)-(\d+)', text)
    assert directive, 'no --array directive'
    low, high = int(directive.group(1)), int(directive.group(2))
    assert (low, high) == (0, len(tasks) - 1), (
        f'--array={low}-{high} but the table has {len(tasks)} tasks. A task index past the end '
        'reads an empty entry and runs the driver with no condition')


@pytest.mark.skipif(not os.path.exists(SUBMIT_PATH), reason='submit script not present')
def test_the_mechanism_arm_is_selected_by_flag_and_the_core_arm_is_not():
    from mlindex.model_training import FomConditions

    tasks, _ = _submit_tasks()
    for condition, arm, _shard, _n in tasks:
        axis = FomConditions.BY_KEY[condition].axis
        expected = 'mechanism' if axis in ('sparsity', 'error_shape') else 'core'
        assert arm == expected, (
            f'{condition} is on the {axis} axis but the array runs it as the {arm} arm; the arms '
            'nest, and running a mechanism bundle over the core entry set costs 5x the cells')


@pytest.mark.skipif(not os.path.exists(SUBMIT_PATH), reason='submit script not present')
def test_the_submit_script_does_not_wrap_the_driver_in_srun():
    # A bare `srun -n 1` pins CPU affinity to one core and strangles the 128 processes. Campaign 1
    # hit this and its calibration script says so in a header comment.
    with open(SUBMIT_PATH, encoding='utf-8') as handle:
        text = handle.read()
    driver_lines = [line for line in text.splitlines()
                    if 'run_fom_dump.py' in line and not line.strip().startswith('#')]
    assert driver_lines, 'the submit script never invokes the driver'
    assert not any('srun' in line for line in driver_lines), driver_lines


def test_a_lattice_that_produced_nothing_stops_the_run(tmp_path):
    """C2-F-071's second half: the loss must not be silent.

    `MAX_CONSECUTIVE_FAILURES` cannot see this. Entries arrive grouped by lattice but are striped
    across shards and pools, so a lattice of 1 400 entries gives each of 256 pools about five --
    below the threshold of 10. Every pool wrote, every task exited 0, and an entire Bravais lattice
    was absent from the benchmark until consolidation three days later.
    """
    from mlindex.scripts.run_fom_dump import refuse_a_missing_lattice

    given = pd.DataFrame({'identifier': ['A', 'B', 'C'],
                          'bravais_lattice': ['hR', 'hR', 'tP']})
    # Only tP produced an entry row; both hR entries failed.
    pd.DataFrame({'entry_id': ['C'], 'bravais_lattice_true': ['tP']}).to_parquet(
        tmp_path / 'entries_tag_pool00.parquet', index=False)
    with pytest.raises(SystemExit, match=r"produced no entry at all: \['hR'\]"):
        refuse_a_missing_lattice(str(tmp_path), 'tag', given)


def test_a_complete_shard_passes_the_lattice_check(tmp_path):
    from mlindex.scripts.run_fom_dump import refuse_a_missing_lattice

    given = pd.DataFrame({'identifier': ['A', 'B'], 'bravais_lattice': ['hR', 'tP']})
    pd.DataFrame({'entry_id': ['A', 'B'],
                  'bravais_lattice_true': ['hR', 'tP']}).to_parquet(
        tmp_path / 'entries_tag_pool00.parquet', index=False)
    assert refuse_a_missing_lattice(str(tmp_path), 'tag', given) is None


def test_the_lattice_check_is_silent_when_nothing_was_written(tmp_path):
    # A shard with no entries at all is a different failure and is reported elsewhere; this guard
    # must not turn it into a confusing message about lattices.
    from mlindex.scripts.run_fom_dump import refuse_a_missing_lattice

    given = pd.DataFrame({'identifier': ['A'], 'bravais_lattice': ['hR']})
    assert refuse_a_missing_lattice(str(tmp_path), 'tag', given) is None
