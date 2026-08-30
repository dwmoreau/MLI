"""The consolidator and the acceptance gate.

Both exist to catch things that are silent, so what is pinned here is that they *refuse* rather
than that they succeed.

* Two directories holding the same bundle overwrite each other's consolidated output. The run
  would report a bundle count below the directory count and otherwise look fine.
* `condition_bundle` must survive consolidation as a real column. Campaign 1's consolidator
  explicitly dropped it -- rebuild row R8 -- after which `entry_id` alone is not a key and a join
  on it fans every candidate out once per bundle.
* The floor table must exist BEFORE the gate is read. PROTOCOL section 7 forbids weakening a gate
  to pass it, and a floor discovered afterwards is exactly that.
* The floor is availability-aware, not a flat number with named exemptions. Campaign 1 exempted
  four lattices by name, which is a gate with a hole in it.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.scripts import run_fom_dump_consolidate as consolidate
from mlindex.scripts import run_fom_dump_gate as gate

RIGHT = np.pi / 2
MANIFEST_PATH = os.path.join('docs', 'fom_campaign2', 'artifacts', 'S06_split_manifest.parquet')


def _bundle(tmp_path, bundle, n_entries=2, name=None, shard='shard00of01', first_entry=0):
    directory = tmp_path / (name or bundle)
    directory.mkdir(parents=True, exist_ok=True)
    entries = pd.DataFrame([{
        'entry_id': f'E{i}', 'q2_digest': f'd{i}', 'condition_bundle': bundle,
        'q2_obs': np.linspace(0.01, 0.2, 20),
        'bravais_lattice_true': 'oP', 'lattice_system_true': 'orthorhombic',
        'second_phase_partner': None,
        'unit_cell_true': np.array([8.0, 11.0, 14.0, RIGHT, RIGHT, RIGHT]),
        'volume_true': 1232.0,
        } for i in range(first_entry, first_entry + n_entries)])
    candidates = pd.DataFrame([{
        'entry_id': f'E{i}', 'q2_digest': f'd{i}', 'condition_bundle': bundle,
        'bravais_lattice': 'oP', 'lattice_system': 'orthorhombic', 'candidate_id': j,
        'M20': 1.0 + j, 'is_correct': j == 0, 'is_off_by_two': False,
        'retained_reason': 'correct' if j == 0 else 'top_k', 'sampling_weight': 1.0,
        } for i in range(first_entry, first_entry + n_entries) for j in range(3)])
    entries.to_parquet(directory / f'entries_{bundle}_{shard}_pool00.parquet', index=False)
    candidates.to_parquet(directory / f'candidates_{bundle}_{shard}_pool00.parquet',
                          index=False)
    with open(directory / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump({'bundle': bundle, 'search_seed_scheme': 'per_entry_bravais', 'arch': 'arm64'},
                  handle)
    return directory


def test_consolidation_keeps_condition_bundle_as_a_real_column(tmp_path):
    # R8. Without it `entry_id` alone is not a key on the consolidated entry table.
    _bundle(tmp_path / 'dump', 'c2_error1_cont0')
    out = tmp_path / 'pool'
    consolidate.main(['--dump-root', str(tmp_path / 'dump'), '--out-dir', str(out)])
    entries = pd.read_parquet(out / 'entries.parquet')
    candidates = pd.read_parquet(out / 'candidates_c2_error1_cont0_oP.parquet')
    assert 'condition_bundle' in entries.columns
    assert 'condition_bundle' in candidates.columns
    assert candidates['condition_bundle'].nunique() == 1


def test_one_bundle_across_two_roots_is_merged_not_refused(tmp_path):
    """A supplementary run is a legitimate second directory for the same bundle.

    The earlier guard refused this outright, which was both too strict -- it forbids regenerating a
    lattice the first pass lost without redoing the other 93 % -- and too loose, since it said
    nothing about which files actually overlapped. The hazard is the same (shard, pool) generated
    twice, and that is now caught by filename in `_stream_paths`.
    """
    main_root, supp_root = tmp_path / 'main', tmp_path / 'supp'
    _bundle(main_root, 'c2_error1_cont0', n_entries=2, shard='shard00of02')
    _bundle(supp_root, 'c2_error1_cont0', n_entries=1, shard='shard00of03', first_entry=90)
    out = tmp_path / 'pool'
    consolidate.main(['--dump-root', str(main_root), str(supp_root), '--out-dir', str(out)])

    entries = pd.read_parquet(out / 'entries.parquet')
    assert entries.shape[0] == 3, 'the supplementary entries were not merged in'
    assert entries['entry_id'].nunique() == 3


def test_the_same_shard_in_two_roots_is_refused(tmp_path):
    """Because consolidating it would double those rows, silently."""
    main_root, dup_root = tmp_path / 'main', tmp_path / 'dup'
    _bundle(main_root, 'c2_error1_cont0', n_entries=2, shard='shard00of02')
    _bundle(dup_root, 'c2_error1_cont0', n_entries=2, shard='shard00of02')
    with pytest.raises(SystemExit, match='appears in both'):
        consolidate.main(['--dump-root', str(main_root), str(dup_root),
                          '--out-dir', str(tmp_path / 'pool')])


def test_a_directory_without_a_manifest_is_not_a_bundle(tmp_path):
    root = tmp_path / 'dump'
    (root / 'half-written').mkdir(parents=True)
    with pytest.raises(SystemExit, match='manifest.json'):
        consolidate.main(['--dump-root', str(root), '--out-dir', str(tmp_path / 'pool')])


def test_the_gate_refuses_to_run_without_a_floor_named_in_advance(tmp_path):
    _bundle(tmp_path / 'dump', 'c2_error1_cont0')
    out = tmp_path / 'pool'
    consolidate.main(['--dump-root', str(tmp_path / 'dump'), '--out-dir', str(out)])
    with pytest.raises(SystemExit, match='named BEFORE the run'):
        gate.main(['check', '--pool', str(out), '--artifact-dir', str(tmp_path / 'empty')])


@pytest.mark.skipif(not os.path.exists(MANIFEST_PATH),
                    reason='docs/ is git-ignored and absent on this machine')
def test_the_floor_is_capped_by_availability_rather_than_exempting_lattices():
    table = gate.build_floor(MANIFEST_PATH)
    core = table[table['arm'] == 'core'].drop_duplicates('bravais_lattice').set_index(
        'bravais_lattice')

    # Every stratum carries a floor. Campaign 1 named four lattices and skipped them entirely.
    assert (table['floor'] > 0).all()
    assert set(core.index) == {'aP', 'cF', 'cI', 'cP', 'hP', 'hR', 'mC',
                               'mP', 'oC', 'oF', 'oI', 'oP', 'tI', 'tP'}

    # cF has 106 eligible entries in existence (C2-F-048), so 200 is unreachable there and the
    # floor says so rather than the gate quietly passing it.
    assert int(core.loc['cF', 'n_entries']) == 106
    assert int(core.loc['cF', 'floor']) < gate.TARGET_CORRECT_PER_STRATUM
    assert bool(core.loc['cF', 'capped'])

    # A lattice with room reaches the full target and is not marked capped.
    assert int(core.loc['aP', 'floor']) == gate.TARGET_CORRECT_PER_STRATUM
    assert not bool(core.loc['aP', 'capped'])

    # The mechanism arm is a nested subset, so its floors are lower for the same lattice --
    # a floor read off the core arm would ask the mechanism bundles for entries they never had.
    mechanism = table[table['arm'] == 'mechanism'].drop_duplicates('bravais_lattice').set_index(
        'bravais_lattice')
    assert int(mechanism.loc['hP', 'n_entries']) < int(core.loc['hP', 'n_entries'])
    assert int(mechanism.loc['hP', 'floor']) < int(core.loc['hP', 'floor'])


@pytest.mark.skipif(not os.path.exists(MANIFEST_PATH),
                    reason='docs/ is git-ignored and absent on this machine')
def test_measured_reachability_overrides_the_default_and_is_recorded_as_such():
    import tempfile

    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as handle:
        handle.write('bravais_lattice,reachability_low\naP,0.95\n')
        path = handle.name
    table = gate.build_floor(MANIFEST_PATH, reachability_path=path)
    aP = table[table['bravais_lattice'] == 'aP'].iloc[0]
    other = table[table['bravais_lattice'] == 'mP'].iloc[0]
    assert aP['reachability_low'] == 0.95
    assert aP['reachability_source'] == 'measured'
    # And a lattice the calibration did not cover keeps the conservative default, labelled.
    assert other['reachability_low'] == gate.DEFAULT_REACHABILITY_LOW
    assert 'default' in other['reachability_source']
    os.unlink(path)


def test_the_weights_layer_fails_when_a_positive_was_not_protected():
    candidates = pd.DataFrame({
        'entry_id': ['E1', 'E1'], 'condition_bundle': ['b', 'b'], 'bravais_lattice': ['oP', 'oP'],
        'is_correct': [True, False], 'retained_reason': ['sampled', 'top_k'],
        'sampling_weight': [20.0, 1.0],
        })
    with pytest.raises(gate.GateFailure, match='not marked'):
        gate.layer_weights(candidates)


def test_the_weights_layer_says_when_it_is_only_partial():
    candidates = pd.DataFrame({
        'entry_id': ['E1'], 'condition_bundle': ['b'], 'bravais_lattice': ['oP'],
        'is_correct': [True], 'retained_reason': ['correct'], 'sampling_weight': [1.0],
        })
    # Without a fully-retained reference the weighted-count half cannot run, and saying so is the
    # point: a silently partial gate reads as a passed one.
    assert 'partial' in gate.layer_weights(candidates)


# ------------------------------------------------------------------------------------------
# Consolidation is streamed in Arrow, not concatenated in pandas. What that must not change.
# ------------------------------------------------------------------------------------------

def test_consolidation_never_holds_more_than_one_shard(tmp_path, monkeypatch):
    """The rewrite's whole point: memory bounded by a shard, not by a bundle.

    A core bundle is ~77 M rows and four of its 34 columns are list-valued, so the old
    `pd.concat([pd.read_parquet(p) for p in paths])` built hundreds of millions of Python objects
    before writing anything. Measured on a 32 MB pool the peak dropped 494 MB -> 150 MB; the ratio
    is what matters, since the old figure scales with the bundle and the new one does not.

    Asserted structurally rather than by measuring RSS, which is not reproducible in a test: no
    more than one shard table may be alive at once.
    """
    import pyarrow.parquet as pq

    root = tmp_path / 'dump'
    _bundle(root, 'c2_error1_cont0', n_entries=3)
    live = {'now': 0, 'peak': 0}
    real_read = pq.read_table

    def counting_read(*args, **kwargs):
        live['now'] += 1
        live['peak'] = max(live['peak'], live['now'])
        table = real_read(*args, **kwargs)
        live['now'] -= 1                      # released as soon as the caller rebinds it
        return table

    monkeypatch.setattr(consolidate.pq, 'read_table', counting_read)
    consolidate.main(['--dump-root', str(root), '--out-dir', str(tmp_path / 'pool')])
    assert live['peak'] <= 1


def test_consolidation_checks_the_join_per_shard_not_per_row(tmp_path):
    """A shard whose candidates disagree with the entry table is refused.

    The check is on the DISTINCT (entry_id, q2_digest) pairs -- a shard holds ~74 entries and
    millions of rows -- but it must still catch the case it exists for: a mis-joined shard is
    otherwise silent, because every column parses and the numbers attach to the wrong pattern.
    """
    root = tmp_path / 'dump'
    directory = _bundle(root, 'c2_error1_cont0', n_entries=2)
    path = next(directory.glob('candidates_*.parquet'))
    corrupted = pd.read_parquet(path)
    corrupted['q2_digest'] = 'not-the-right-digest'
    corrupted.to_parquet(path, index=False)
    with pytest.raises(SystemExit, match='q2_digest'):
        consolidate.main(['--dump-root', str(root), '--out-dir', str(tmp_path / 'pool')])


def test_an_entry_absent_from_the_entry_table_is_refused(tmp_path):
    root = tmp_path / 'dump'
    directory = _bundle(root, 'c2_error1_cont0', n_entries=2)
    path = next(directory.glob('candidates_*.parquet'))
    orphaned = pd.read_parquet(path)
    orphaned['entry_id'] = 'NOT_AN_ENTRY'
    orphaned.to_parquet(path, index=False)
    with pytest.raises(SystemExit, match='absent from the entry table'):
        consolidate.main(['--dump-root', str(root), '--out-dir', str(tmp_path / 'pool')])


def test_counts_are_grouped_by_the_TRUE_lattice_not_the_candidate_s(tmp_path):
    """Two different groupings, and mixing them would misreport every stratum.

    Files are partitioned by the candidate's lattice, because that is what a later step loads one
    of. Counts are grouped by the ENTRY's true lattice, because that is the stratum METRICS.md
    section 5 defines and what the acceptance floor is keyed on.
    """
    root = tmp_path / 'dump'
    _bundle(root, 'c2_error1_cont0', n_entries=2)
    entries, counts = consolidate.consolidate_bundle(
        root / 'c2_error1_cont0', tmp_path / 'pool', consolidate.ROW_GROUP_SIZE)
    assert list(counts['bravais_lattice_true'].unique()) == ['oP']
    assert int(counts['n_candidates'].sum()) == 6      # 2 entries x 3 candidates
    assert int(counts['n_correct'].sum()) == 2         # one per entry, by construction
    assert int(counts['n_entries'].iloc[0]) == 2
    assert int(counts['n_reachable_entries'].iloc[0]) == 2
