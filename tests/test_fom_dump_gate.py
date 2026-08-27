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


def _bundle(tmp_path, bundle, n_entries=2, name=None):
    directory = tmp_path / (name or bundle)
    directory.mkdir(parents=True, exist_ok=True)
    entries = pd.DataFrame([{
        'entry_id': f'E{i}', 'q2_digest': f'd{i}', 'condition_bundle': bundle,
        'q2_obs': np.linspace(0.01, 0.2, 20),
        'bravais_lattice_true': 'oP', 'lattice_system_true': 'orthorhombic',
        'second_phase_partner': None,
        'unit_cell_true': np.array([8.0, 11.0, 14.0, RIGHT, RIGHT, RIGHT]),
        'volume_true': 1232.0,
        } for i in range(n_entries)])
    candidates = pd.DataFrame([{
        'entry_id': f'E{i}', 'q2_digest': f'd{i}', 'condition_bundle': bundle,
        'bravais_lattice': 'oP', 'lattice_system': 'orthorhombic', 'candidate_id': j,
        'M20': 1.0 + j, 'is_correct': j == 0, 'is_off_by_two': False,
        'retained_reason': 'correct' if j == 0 else 'top_k', 'sampling_weight': 1.0,
        } for i in range(n_entries) for j in range(3)])
    entries.to_parquet(directory / f'entries_{bundle}_shard00of01_pool00.parquet', index=False)
    candidates.to_parquet(directory / f'candidates_{bundle}_shard00of01_pool00.parquet',
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


def test_two_directories_holding_one_bundle_are_refused(tmp_path):
    # The "one manifest.json per --out-dir" trap one level up: the second silently overwrites the
    # first, and the run still looks successful.
    root = tmp_path / 'dump'
    # Two sibling directories, differently named, holding the SAME bundle -- which is what a
    # re-run into a new directory looks like.
    _bundle(root, 'c2_error1_cont0', name='first')
    _bundle(root, 'c2_error1_cont0', name='second')
    with pytest.raises(SystemExit, match='both hold bundle'):
        consolidate.main(['--dump-root', str(root), '--out-dir', str(tmp_path / 'pool')])


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
