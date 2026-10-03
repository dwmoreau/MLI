"""Writing a benchmark pool, and reading back exactly what was written.

The reading half of this module was validated in P04a against a pool campaign 2 generated. These
cover the half that produces one, and the refusals that stop a broken pool from reporting as a
measurement rather than as an error.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import Benchmark


def _record(entry_id='AAAAAA', bundle='b1_error1_cont0', lattice='cP',
            lattice_system='cubic', m20=(30.0, 10.0), cells=(5.0, 7.0)):
    n = len(m20)
    return {
        'entry_id': entry_id, 'condition_bundle': bundle, 'q2_digest': 'deadbeefdeadbeef',
        'bravais_lattice': lattice, 'lattice_system': lattice_system,
        'n_peaks': 10, 'hkl_ref_length': 500, 'n_entering': 99,
        'assignment_threshold': 0.95, 'downsample_radius': 1e-4, 'prune_threshold': 1.5,
        'candidate_id': np.arange(n, dtype=np.int64),
        'xnn': np.array([[1.0/c**2] for c in cells]),
        'unit_cell': np.array([[c] for c in cells]),
        'volume': np.array([c**3 for c in cells]),
        'reciprocal_volume': np.array([c**-3 for c in cells]),
        'spacegroup': [f'SG{i}' for i in range(n)],
        'M20': np.array(m20, dtype=float),
        'n_indexed': np.arange(n, dtype=np.int64),
        'final_rank': np.arange(n, dtype=np.int64),
        'in_top_n': np.ones(n, dtype=bool),
        }


def _entries(entry_id='AAAAAA', bundle='b1_error1_cont0', a=5.0):
    return pd.DataFrame([{
        'entry_id': entry_id, 'condition_bundle': bundle, 'q2_digest': 'deadbeefdeadbeef',
        'source_db': 'csd', 'split': 'fom-dev', 'q2_obs': np.linspace(0.05, 0.5, 10),
        'n_peaks_available': 20, 'bravais_lattice_true': 'cP', 'lattice_system_true': 'cubic',
        'unit_cell_true': np.array([a, a, a, 90.0, 90.0, 90.0]), 'volume_true': a**3,
        'is_degenerate': False, 'pool_size_full': 99,
        }])


def _manifest(**overrides):
    base = {name: 'x' for name in Benchmark.IDENTITY_FIELDS}
    base.update(schema_version=Benchmark.SCHEMA_VERSION, pool_size=1, prune_threshold=1.5,
                search_seed=12345, seed=12345)
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Records to a frame
# ---------------------------------------------------------------------------


def test_block_wide_values_are_broadcast_and_per_candidate_arrays_are_not():
    frame = Benchmark.records_to_frame([_record(), _record(entry_id='BBBBBB')])

    assert frame.shape[0] == 4
    assert frame['entry_id'].tolist() == ['AAAAAA']*2 + ['BBBBBB']*2
    assert frame['n_entering'].tolist() == [99]*4
    assert frame['M20'].tolist() == [30.0, 10.0, 30.0, 10.0]
    # A cell stays one value per candidate, not one number per frame.
    assert [list(cell) for cell in frame['unit_cell']] == [[5.0], [7.0]]*2


def test_no_records_still_gives_the_declared_columns():
    """An arm that indexed nothing must not produce a frame whose columns depend on that."""
    frame = Benchmark.records_to_frame([])
    assert list(frame.columns) == list(Benchmark.CANDIDATE_COLUMNS)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def test_the_true_cell_is_labelled_correct_and_a_wrong_one_is_not():
    frame = Benchmark.records_to_frame([_record(cells=(5.0, 7.0))])
    labelled = Benchmark.label_frame(frame, _entries(a=5.0))

    assert labelled['is_correct'].tolist() == [True, False]


def test_a_cell_is_correct_only_in_the_true_lattice_but_in_any_of_its_settings():
    """The cell comparison slices the truth to the candidate's lattice system, so on its own it
    compares a cubic candidate with an orthorhombic truth on `a` alone. A different setting of the
    true cell -- here the axes in another order -- is the true cell and must still count."""
    def block(lattice, lattice_system, cell):
        record = _record(lattice=lattice, lattice_system=lattice_system, m20=(30.0,),
                         cells=(cell[0],))
        record['unit_cell'] = np.array([cell], dtype=float)
        record['xnn'] = np.array([cell], dtype=float)
        return record

    frame = Benchmark.records_to_frame([
        block('cP', 'cubic', [5.0]),
        block('oP', 'orthorhombic', [7.0, 5.0, 6.0]),
        block('oC', 'orthorhombic', [5.0, 6.0, 7.0]),
        ])
    entries = _entries().assign(bravais_lattice_true='oP', lattice_system_true='orthorhombic')
    entries['unit_cell_true'] = [np.array([5.0, 6.0, 7.0, 90.0, 90.0, 90.0])]
    labelled = Benchmark.label_frame(frame, entries)

    assert labelled['bravais_lattice'].tolist() == ['cP', 'oP', 'oC']
    assert labelled['is_correct'].tolist() == [False, True, False]


def test_a_candidate_whose_pattern_has_no_truth_is_refused():
    """Labelling it would leave `is_correct` false for a reason that has nothing to do with the
    cell, and false is what the overwhelming majority of rows carry legitimately."""
    frame = Benchmark.records_to_frame([_record(entry_id='ZZZZZZ')])

    with pytest.raises(ValueError, match='no row in the entry table'):
        Benchmark.label_frame(frame, _entries(entry_id='AAAAAA'))


def test_a_cell_of_the_wrong_width_is_refused_rather_than_compared():
    """A candidate carries the partial cell for its own lattice system. If that ever stops
    matching the sliced truth the labeller would compare different quantities and report it as an
    answer."""
    record = _record()
    record['unit_cell'] = np.array([[5.0, 90.0], [7.0, 90.0]])
    frame = Benchmark.records_to_frame([record])

    with pytest.raises(ValueError, match='cell parameters'):
        Benchmark.label_frame(frame, _entries())


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


def test_two_pools_stripe_an_arm_and_consolidate_into_one_shard_per_lattice(tmp_path):
    """An arm is generated by several pools at once. Each writes its own stripe under the same
    names, and consolidation streams them into the shard a reader rebuilds by name."""
    entries = pd.concat([_entries('AAAAAA'), _entries('BBBBBB')], ignore_index=True)
    for part, entry_id in enumerate(('AAAAAA', 'BBBBBB')):
        directory = Benchmark.part_dir(tmp_path, part)
        frame = Benchmark.records_to_frame([_record(entry_id=entry_id)])
        frame = Benchmark.label_frame(frame, entries)
        Benchmark.write_candidate_shard(frame, directory, 'b1_error1_cont0', 'cP')
        Benchmark.write_entry_table(entries.iloc[[part]], directory)

    Benchmark.consolidate(tmp_path)
    Benchmark.write_manifest(tmp_path, **_manifest())
    Benchmark.stamp_complete(tmp_path, n_source_entries=2, n_bundles=1, n_patterns_refused=0)

    assert not (tmp_path / Benchmark.PART_DIR).exists()
    assert Benchmark.available_bundles(tmp_path) == ['b1_error1_cont0']
    assert Benchmark.check_complete(tmp_path)['n_source_entries'] == 2

    read = Benchmark.load_candidates(tmp_path, 'b1_error1_cont0', sidecars=())
    assert read.shape[0] == 4
    assert sorted(read['entry_id'].unique()) == ['AAAAAA', 'BBBBBB']
    assert read['is_correct'].sum() == 2
    assert list(read.columns)[:6] == list(Benchmark.CANDIDATE_COLUMNS)[:6]

    read_entries = Benchmark.load_entries(tmp_path)
    assert sorted(read_entries['entry_id']) == ['AAAAAA', 'BBBBBB']
    Benchmark.check_peak_digests(read, read_entries)


def test_an_unconsolidated_arm_is_not_silently_readable(tmp_path):
    """Nothing may read the stripes as though they were the arm: they hold different crystals,
    and a pool killed halfway leaves some of them."""
    with pytest.raises(FileNotFoundError):
        Benchmark.consolidate(tmp_path)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_a_manifest_missing_an_identity_field_is_refused(tmp_path):
    """An omitted field cannot be refused a pairing it should be refused, and the omission is
    invisible at the point it matters."""
    incomplete = _manifest()
    del incomplete['arch']

    with pytest.raises(ValueError, match='arch'):
        Benchmark.write_manifest(tmp_path, **incomplete)


def test_arms_from_different_machines_are_not_paired():
    arms = {'a': _manifest(arch='x86_64'), 'b': _manifest(arch='arm64')}

    with pytest.raises(ValueError, match='arch'):
        Benchmark.manifest_identity(arms)


def test_the_axis_under_study_is_named_and_everything_else_still_checked():
    """A run-to-run floor varies the search seed on purpose; it must not also vary the commit."""
    arms = {'a': _manifest(search_seed=12345), 'b': _manifest(search_seed=202)}
    assert Benchmark.manifest_identity(arms, allow=('search_seed',))

    with pytest.raises(ValueError, match='search_seed'):
        Benchmark.manifest_identity(arms)

    moved = {'a': _manifest(search_seed=12345), 'b': _manifest(search_seed=202, commit='other')}
    with pytest.raises(ValueError, match='commit'):
        Benchmark.manifest_identity(moved, allow=('search_seed',))


def test_arms_drawn_from_different_splits_are_not_paired():
    """A setting chosen on fom-train and confirmed on fom-dev is two populations; pairing them
    would report the change of crystals as the effect of the setting."""
    arms = {'a': _manifest(split='fom-train'), 'b': _manifest(split='fom-dev')}
    with pytest.raises(ValueError, match='split'):
        Benchmark.manifest_identity(arms)

    narrowed = {'a': _manifest(true_lattices=['mP']), 'b': _manifest(true_lattices=['aP', 'mP'])}
    with pytest.raises(ValueError, match='true_lattices'):
        Benchmark.manifest_identity(narrowed)


def _split_manifest(path):
    rows = [{'identifier': f'{split[4:]}{lattice}{i}', 'bravais_lattice': lattice, 'split': split}
            for split in ('fom-train', 'fom-dev', 'fom-test') for lattice in ('mP', 'aP')
            for i in range(3)]
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_a_draw_takes_only_the_named_split_and_lattices(tmp_path):
    from mlindex.model_training.BenchmarkRuns import draw_entries

    manifest = _split_manifest(tmp_path/'split.parquet')
    chosen = draw_entries(manifest, 2, 12345, split='fom-train', bravais_lattices=['mP'])

    assert chosen.shape[0] == 2
    assert set(chosen['split']) == {'fom-train'}
    assert set(chosen['bravais_lattice']) == {'mP'}


@pytest.mark.parametrize('overrides, message', [
    ({'split': 'fom-test'}, 'sealed'),
    ({'population': 'hard', 'true_lattices': ['cP']}, 'not in the hard population'),
    ])
def test_an_arm_refuses_a_sealed_split_or_a_lattice_outside_its_population(
        tmp_path, overrides, message):
    """Refused before a single pattern is drawn, so a mistyped option costs nothing."""
    from mlindex.model_training.BenchmarkRuns import run_arm
    from mlindex.utilities.Digests import file_digest

    manifest = _split_manifest(tmp_path/'split.parquet')
    with pytest.raises(ValueError, match=message):
        run_arm(tmp_path/'arm', manifest, file_digest(manifest), **overrides)
    assert not (tmp_path/'arm').exists()


def test_an_arm_refuses_a_split_manifest_that_is_not_the_one_expected(tmp_path):
    """The split is proved before the run, not after: a different file could draw crystals from
    the sealed split, and nothing downstream would notice."""
    from mlindex.model_training.BenchmarkRuns import run_arm
    from mlindex.utilities.Digests import file_digest

    manifest = _split_manifest(tmp_path/'split.parquet')
    expected = file_digest(manifest)
    other = _split_manifest(tmp_path/'other.parquet')
    pd.read_parquet(other).iloc[:-1].to_parquet(other)

    with pytest.raises(ValueError, match='sha256'):
        run_arm(tmp_path/'arm', other, expected)
    assert not (tmp_path/'arm').exists()


def test_an_arm_is_refused_from_a_checkout_with_uncommitted_changes(monkeypatch):
    """The manifest names a commit; with tracked files modified, that commit is not the code
    that ran and the number cannot be attributed to anything readable."""
    from mlindex.model_training import BenchmarkRuns

    answers = {'rev-parse': 'abc123\n', 'status': ' M mlindex/optimization/Candidates.py\n'}
    monkeypatch.setattr(BenchmarkRuns, '_git', lambda *arguments: answers[arguments[0]])
    with pytest.raises(RuntimeError, match='uncommitted'):
        BenchmarkRuns._provenance()

    answers['status'] = ''
    provenance = BenchmarkRuns._provenance()
    assert provenance['commit'] == 'abc123'
    assert provenance['n_model_files'] > 0 and len(provenance['models_digest']) == 64


def test_a_models_digest_changes_with_any_file_and_ignores_other_directories(tmp_path):
    from mlindex.utilities.Digests import tree_digest

    (tmp_path/'cubic_1'/'data').mkdir(parents=True)
    (tmp_path/'cubic_1'/'data'/'a.npy').write_bytes(b'one')
    (tmp_path/'unrelated').mkdir()
    (tmp_path/'unrelated'/'b').write_bytes(b'two')
    before, n_files = tree_digest(tmp_path, ['cubic_1'])

    (tmp_path/'unrelated'/'b').write_bytes(b'three')
    assert tree_digest(tmp_path, ['cubic_1']) == (before, 1)
    (tmp_path/'cubic_1'/'data'/'a.npy').write_bytes(b'uno')
    assert tree_digest(tmp_path, ['cubic_1'])[0] != before


def test_one_arm_needs_no_agreement():
    assert Benchmark.manifest_identity({'a': _manifest()})


def test_a_manifest_round_trips_through_disk(tmp_path):
    Benchmark.write_manifest(tmp_path, **_manifest())
    loaded = Benchmark.load_manifest(tmp_path)

    assert loaded['schema_version'] == Benchmark.SCHEMA_VERSION
    assert loaded['candidate_columns'] == list(Benchmark.CANDIDATE_COLUMNS)
    assert Benchmark.manifest_identity({'a': loaded, 'b': _manifest()})


# ---------------------------------------------------------------------------
# The merit sidecar
# ---------------------------------------------------------------------------


def test_a_sidecar_column_carries_the_value_its_name_claims(tmp_path, models_dir):
    """The sidecar is joined back on the candidate key, so a column arriving under the wrong name
    would rank the pool by something else and read as a measurement. This covers the round trip
    and the naming; agreement with an independently computed merit is the test below."""
    from mlindex.model_training.BenchmarkRuns import (SIDECAR_MERITS, _hkl_reference,
                                                      merit_sidecar)
    from mlindex.utilities.FigureOfMerits import merit_set
    from mlindex.utilities.Q2Calculator import Q2Calculator
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    q2_obs = np.linspace(0.05, 0.5, 20)
    xnn = np.array([[0.04], [0.0402], [0.0399]])
    by_group = get_spacegroup_hkl_ref(_hkl_reference('cP', 'cubic'), bravais_lattice='cP')
    spacegroup = sorted(by_group)[0]
    calculator = Q2Calculator(lattice_system='cubic', hkl=by_group[spacegroup],
                              tensorflow=False, representation='xnn')
    expected = merit_set(q2_obs[:10], calculator.get_q2(xnn))

    frame = pd.DataFrame({
        'entry_id': ['AAAAAA']*3, 'condition_bundle': ['b1_error1_cont0']*3,
        'bravais_lattice': ['cP']*3, 'candidate_id': np.arange(3),
        'lattice_system': ['cubic']*3, 'xnn': list(xnn),
        'spacegroup': [spacegroup]*3, 'n_peaks': [10]*3, 'M20': expected['M20'],
        })
    entries = pd.DataFrame([{'entry_id': 'AAAAAA', 'condition_bundle': 'b1_error1_cont0',
                             'q2_obs': q2_obs}])
    Benchmark.write_candidate_shard(frame, tmp_path, 'b1_error1_cont0', 'cP')
    Benchmark.write_entry_table(entries, tmp_path)

    merit_sidecar(tmp_path)
    joined = Benchmark.load_candidates(tmp_path, 'b1_error1_cont0')

    for name in SIDECAR_MERITS:
        np.testing.assert_allclose(joined[name].to_numpy(), expected[name], rtol=0, atol=0,
                                   err_msg=name)


def test_several_bundles_and_several_pools_consolidate_without_colliding(tmp_path):
    """A floor arm carries three condition bundles and a hard arm five, written by several pools
    at once. Bundle tags contain both dots and underscores (`b1_error0.5_cont0`), and the reader
    rebuilds a shard name by splitting the lattice off the end -- so this pins that a tag survives
    the round trip through a filename."""
    bundles = ('b1_error0.5_cont0', 'b1_error1_cont0', 'b1_error2_cont0')
    entries = pd.concat([_entries('AAAAAA', bundle=bundle) for bundle in bundles]
                        + [_entries('BBBBBB', bundle=bundle) for bundle in bundles],
                        ignore_index=True)
    for part, entry_id in enumerate(('AAAAAA', 'BBBBBB')):
        directory = Benchmark.part_dir(tmp_path, part)
        for bundle in bundles:
            frame = Benchmark.records_to_frame([_record(entry_id=entry_id, bundle=bundle)])
            frame = Benchmark.label_frame(frame, entries)
            Benchmark.write_candidate_shard(frame, directory, bundle, 'cP')
        Benchmark.write_entry_table(
            entries.loc[entries['entry_id'] == entry_id].reset_index(drop=True), directory)

    Benchmark.consolidate(tmp_path)

    assert Benchmark.available_bundles(tmp_path) == sorted(bundles)
    read_entries = Benchmark.load_entries(tmp_path)
    assert read_entries.shape[0] == 6
    for bundle in bundles:
        read = Benchmark.load_candidates(tmp_path, bundle, sidecars=())
        assert sorted(read['entry_id'].unique()) == ['AAAAAA', 'BBBBBB']
        assert read['condition_bundle'].unique().tolist() == [bundle]
        assert read.shape[0] == 4


# ---------------------------------------------------------------------------
# The floor stage
# ---------------------------------------------------------------------------


def _floor_reduction(values):
    """One arm's per-entry reduction, built by the real reduction so the columns are the real ones.

    `values[i]` is whether crystal i's correct cell ranks first under the score.
    """
    from mlindex.model_training import BenchmarkMetrics as metrics

    rows = []
    for index, correct_first in enumerate(values):
        for position in range(2):
            rows.append({
                'entry_id': f'C{index:03d}', 'condition_bundle': 'b1_error1_cont0',
                'bravais_lattice': 'cP', 'candidate_id': position,
                'in_top_n': True, 'is_degenerate': False,
                'is_correct': (position == 0) if correct_first else (position == 1),
                'score': 10.0 - position,
                })
    frame = pd.DataFrame(rows)
    return metrics.reduce_many(frame, {'M20': 'score', 'M_sym': 'score'})


def test_a_floor_that_comes_out_zero_is_refused_rather_than_reported():
    """A floor of zero makes every gate read against it infinite or undefined, and the NaN it
    produces looks like a missing number rather than a broken one. At a real sample size it means
    too few arms or too few crystals, not a noiseless search."""
    from mlindex.model_training.BenchmarkRuns import floor_from_arms

    # Two arms that agree everywhere: the pairwise shift is identically zero.
    arms = {name: _floor_reduction([True]*8) for name in ('a', 'b')}

    with pytest.raises(ValueError, match='floor is'):
        floor_from_arms(arms, 'M_sym', 'M20')


def test_one_arm_cannot_produce_a_floor():
    from mlindex.model_training.BenchmarkRuns import floor_from_arms

    arms = {'a': _floor_reduction([True, False])}

    with pytest.raises(ValueError, match='at least two arms'):
        floor_from_arms(arms, 'M_sym', 'M20')


def test_the_floor_stage_refuses_arms_that_differ_in_more_than_the_seed(tmp_path, monkeypatch):
    """A floor is the spread between arms that differ only in the search seed. Nothing downstream
    can tell that spread from a machine-to-machine or commit-to-commit one."""
    from mlindex.scripts.run_benchmark import main

    for name, commit in (('armA', 'aaaa'), ('armB', 'bbbb')):
        directory = tmp_path / name
        entries = _entries()
        frame = Benchmark.label_frame(Benchmark.records_to_frame([_record()]), entries)
        Benchmark.write_candidate_shard(frame, directory, 'b1_error1_cont0', 'cP')
        Benchmark.write_entry_table(entries, directory)
        Benchmark.write_manifest(directory, **_manifest(commit=commit, search_seed=12345))
        Benchmark.stamp_complete(directory, n_source_entries=1, n_bundles=1,
                                 n_patterns_refused=0)

    with pytest.raises(ValueError, match='commit'):
        main(['--stage', 'floor', '--arm', f'armA={tmp_path/"armA"}',
              '--arm', f'armB={tmp_path/"armB"}', '--scores', 'M20,M_sym'])


def test_a_bundle_that_mostly_fails_is_refused_but_a_few_refusals_are_not():
    """The guard is for a condition that cannot be applied at all, not for the handful of large
    cells that legitimately have no partner. It is checked over the whole arm: at 128 pools a hard
    stripe is under three crystals, so any fraction of a stripe would abort on the first refusal."""
    from mlindex.model_training.BenchmarkRuns import _refuse_a_broken_bundle

    bundles = ['b1_error1_cont0_phase3']
    few = [{'entry_id': f'C{i}', 'condition_bundle': bundles[0], 'reason': 'x'} for i in range(20)]
    _refuse_a_broken_bundle(few, bundles, n_crystals=360)          # 5.6 %, the real tail

    many = [{'entry_id': f'C{i}', 'condition_bundle': bundles[0], 'reason': 'x'} for i in range(100)]
    with pytest.raises(RuntimeError, match='27.8 %'):
        _refuse_a_broken_bundle(many, bundles, n_crystals=360)


def test_refusals_are_counted_against_their_own_bundle():
    """A bundle is not condemned by another bundle's refusals."""
    from mlindex.model_training.BenchmarkRuns import _refuse_a_broken_bundle

    bundles = ['b1_error1_cont0_phase3', 'b1_error2_cont0']
    failures = [{'entry_id': f'C{i}', 'condition_bundle': bundles[0], 'reason': 'x'}
                for i in range(30)]
    _refuse_a_broken_bundle(failures, bundles, n_crystals=360)


def _shard(pool_dir, shard, n_shards, entry_ids, **overrides):
    """What `run_arm` leaves for one shard: its pools' stripes, with a sidecar, and its stamp."""
    entries = pd.concat([_entries(entry_id) for entry_id in entry_ids], ignore_index=True)
    for index, entry_id in enumerate(entry_ids):
        directory = Benchmark.part_dir(pool_dir, f'{shard:03d}_{index:03d}')
        frame = Benchmark.label_frame(Benchmark.records_to_frame([_record(entry_id=entry_id)]),
                                      entries)
        Benchmark.write_candidate_shard(frame, directory, 'b1_error1_cont0', 'cP')
        Benchmark.write_candidate_shard(frame[list(Benchmark.CANDIDATE_KEY)].assign(M_sym=1.0),
                                        directory/Benchmark.MERIT_SIDECAR, 'b1_error1_cont0', 'cP')
        Benchmark.write_entry_table(entries.iloc[[index]], directory)
    stamp = dict(_manifest(), bundles=['b1_error1_cont0'], n_source_entries=4, n_shards=n_shards,
                 shard=shard, n_shard_entries=len(entry_ids), failures=[])
    stamp.update(overrides)
    (pool_dir/Benchmark.SHARD_DIR).mkdir(parents=True, exist_ok=True)
    (pool_dir/Benchmark.SHARD_DIR/f'{shard:03d}.json').write_text(json.dumps(stamp),
                                                                 encoding='utf-8')


def test_an_arm_is_finalized_only_when_every_shard_is_stamped(tmp_path):
    """A shard that died leaves valid stripes and no stamp; merging without it would describe the
    crystals that happened to finish, and the stamp would say complete."""
    from mlindex.model_training.BenchmarkRuns import finalize_arm

    _shard(tmp_path, 0, 2, ['AAAAAA', 'BBBBBB'])
    with pytest.raises(RuntimeError, match=r'Shards \[1\] of 2 have no stamp'):
        finalize_arm(tmp_path)
    assert not (tmp_path/Benchmark.COMPLETION_NAME).exists()

    _shard(tmp_path, 1, 2, ['CCCCCC', 'DDDDDD'])
    metadata = finalize_arm(tmp_path)

    assert metadata['n_shard_entries'] == [2, 2]
    assert Benchmark.check_complete(tmp_path)['n_shards'] == 2
    read = Benchmark.load_candidates(tmp_path, 'b1_error1_cont0')
    assert sorted(read['entry_id'].unique()) == ['AAAAAA', 'BBBBBB', 'CCCCCC', 'DDDDDD']
    assert read['M_sym'].notna().all()
    assert not (tmp_path/Benchmark.PART_DIR).exists()
    with pytest.raises(FileExistsError, match='already finalized'):
        finalize_arm(tmp_path)


@pytest.mark.parametrize('overrides, message', [
    ({'commit': 'other'}, 'differs'),
    ({'n_shard_entries': 3}, 'hold 5 crystals'),
    ])
def test_shards_of_different_arms_or_the_wrong_size_are_not_merged(tmp_path, overrides, message):
    from mlindex.model_training.BenchmarkRuns import finalize_arm

    _shard(tmp_path, 0, 2, ['AAAAAA', 'BBBBBB'])
    _shard(tmp_path, 1, 2, ['CCCCCC', 'DDDDDD'], **overrides)
    with pytest.raises((ValueError, RuntimeError), match=message):
        finalize_arm(tmp_path)
    assert not (tmp_path/Benchmark.COMPLETION_NAME).exists()


def test_a_completion_stamp_the_entry_table_contradicts_is_refused(tmp_path):
    """A stamp that exists is not the same as a stamp that describes what is there."""
    Benchmark.write_entry_table(pd.concat([_entries('AAAAAA'), _entries('BBBBBB')]), tmp_path)
    Benchmark.stamp_complete(tmp_path, n_source_entries=2, n_bundles=2, n_patterns_refused=0)
    with pytest.raises(ValueError, match='holds 2'):
        Benchmark.check_complete(tmp_path)

    Benchmark.stamp_complete(tmp_path, n_source_entries=2, n_bundles=2, n_patterns_refused=2)
    assert Benchmark.check_complete(tmp_path)['n_patterns_refused'] == 2


def test_a_shard_is_refused_only_where_it_would_collide(tmp_path):
    """Several shards write into one arm, so another shard's stripes are expected there; this
    shard's own, or a finished arm, are not."""
    from mlindex.model_training.BenchmarkRuns import _refuse_an_occupied_directory

    (tmp_path/Benchmark.PART_DIR/'001_000').mkdir(parents=True)
    _refuse_an_occupied_directory(tmp_path, shard=0, n_shards=2)
    with pytest.raises(FileExistsError, match='001_000'):
        _refuse_an_occupied_directory(tmp_path, shard=1, n_shards=2)
    Benchmark.write_manifest(tmp_path, **_manifest())
    with pytest.raises(FileExistsError, match='manifest'):
        _refuse_an_occupied_directory(tmp_path, shard=0, n_shards=2)


def test_generating_into_an_occupied_directory_is_refused(tmp_path):
    """An arm that died leaves its finished pools' stripes behind. Re-running into the same place
    would consolidate them with the new ones, and nothing in the result would say which run a
    shard came from."""
    from mlindex.model_training.BenchmarkRuns import _refuse_an_occupied_directory

    _refuse_an_occupied_directory(tmp_path/'never_used')      # absent is fine
    (tmp_path/'arm').mkdir()
    _refuse_an_occupied_directory(tmp_path/'arm')             # empty is fine

    (tmp_path/'arm'/'parts').mkdir()
    with pytest.raises(FileExistsError, match='parts'):
        _refuse_an_occupied_directory(tmp_path/'arm')


# ---------------------------------------------------------------------------
# The merit sidecar against an independently computed one
# ---------------------------------------------------------------------------

CAMPAIGN_POOL = Path('mlindex/data/fom_full_c2_pool')


@pytest.mark.skipif(not CAMPAIGN_POOL.is_dir(), reason='campaign pool not on this machine')
@pytest.mark.parametrize('lattice', ['cP', 'oP', 'aP', 'hP'])
def test_the_sidecar_matches_merits_computed_independently(tmp_path, models_dir, lattice):
    """The campaign computed these merits with its own code on its own machine. Agreeing with the
    round trip through `merit_set` proves nothing -- that is the same function twice. This is the
    check that was missing when the sidecar shipped scoring every candidate against the full
    reference list and a twenty-peak window, which no lattice's pipeline used."""
    from mlindex.model_training.BenchmarkRuns import SIDECAR_MERITS, merit_sidecar

    bundle = 'c2_error1_cont0'
    shard = pd.read_parquet(CAMPAIGN_POOL/f'candidates_{bundle}_{lattice}.parquet').head(200)
    Benchmark.write_candidate_shard(shard, tmp_path, bundle, lattice)
    Benchmark.write_entry_table(Benchmark.load_entries(CAMPAIGN_POOL), tmp_path)

    merit_sidecar(tmp_path)

    mine = pd.read_parquet(tmp_path/'merits'/f'candidates_{bundle}_{lattice}.parquet')
    theirs = pd.read_parquet(CAMPAIGN_POOL/'merits'/f'candidates_{bundle}_{lattice}.parquet')
    joined = mine.merge(theirs, on=list(Benchmark.CANDIDATE_KEY),
                        suffixes=('_mine', '_theirs'), validate='1:1')
    assert joined.shape[0] == shard.shape[0]
    # The campaign spells the M_rev support `N_cal`; `merit_set` spells it `n_cal`, matching
    # PRUNE_CAPTURE_MERITS. Same quantity, so it is compared under both names.
    theirs_name = {'n_cal': 'N_cal'}
    for name in SIDECAR_MERITS:
        left = joined[f'{name}_mine'] if f'{name}_mine' in joined else joined[name]
        other = theirs_name.get(name, name)
        right = joined[f'{other}_theirs'] if f'{other}_theirs' in joined else joined[other]
        np.testing.assert_array_equal(left.to_numpy(), right.to_numpy(),
                                      err_msg=f'{lattice} {name}')


@pytest.mark.skipif(not CAMPAIGN_POOL.is_dir(), reason='campaign pool not on this machine')
@pytest.mark.parametrize('lattice', ['cF', 'hR', 'oP', 'mC', 'aP'])
def test_the_feature_sidecar_matches_the_campaign_s_stored_features(tmp_path, models_dir,
                                                                    lattice):
    """The campaign's stored structural sidecar and its dump-time absence counts were computed by
    its own code. Agreement with them, exactly, is what licenses fitting on these columns a model
    whose earlier version was fitted on theirs. Lattices with one extinction group (aP) and many
    (oP) take different paths through the absence counts."""
    from mlindex.model_training.BenchmarkRuns import SIDECAR_FEATURES, feature_sidecar

    bundle = 'c2_error1_cont0'
    shard = pd.read_parquet(CAMPAIGN_POOL/f'candidates_{bundle}_{lattice}.parquet').head(200)
    Benchmark.write_candidate_shard(shard, tmp_path, bundle, lattice)
    Benchmark.write_entry_table(Benchmark.load_entries(CAMPAIGN_POOL), tmp_path)

    feature_sidecar(tmp_path)

    mine = pd.read_parquet(tmp_path/'features'/f'candidates_{bundle}_{lattice}.parquet')
    theirs = pd.read_parquet(CAMPAIGN_POOL/'structural'/f'candidates_{bundle}_{lattice}.parquet')
    theirs = theirs.merge(
        shard[list(Benchmark.CANDIDATE_KEY) + ['n_absent_extra', 'n_groups_searched']],
        on=list(Benchmark.CANDIDATE_KEY))
    joined = mine.merge(theirs, on=list(Benchmark.CANDIDATE_KEY),
                        suffixes=('_mine', '_theirs'), validate='1:1')
    assert joined.shape[0] == shard.shape[0]
    for name in SIDECAR_FEATURES:
        np.testing.assert_array_equal(joined[f'{name}_mine'].to_numpy(dtype=float),
                                      joined[f'{name}_theirs'].to_numpy(dtype=float),
                                      err_msg=f'{lattice} {name}')


def test_a_feature_sidecar_on_a_different_reference_list_is_refused(tmp_path, models_dir):
    """The absence counts are counts over the reference list the search used. A models tree with
    a different list would give counts over different reflections, all finite and plausible."""
    from mlindex.model_training.BenchmarkRuns import _hkl_reference, feature_sidecar

    n_lines = _hkl_reference('cP', 'cubic').shape[0]
    frame = pd.DataFrame({
        'entry_id': ['AAAAAA'], 'condition_bundle': ['b1_error1_cont0'],
        'bravais_lattice': ['cP'], 'candidate_id': [0], 'lattice_system': ['cubic'],
        'xnn': [np.array([0.04])], 'spacegroup': ['P - - - e.g. P 2 3'], 'n_peaks': [10],
        'hkl_ref_length': [n_lines + 1], 'M20': [10.0],
        })
    Benchmark.write_candidate_shard(frame, tmp_path, 'b1_error1_cont0', 'cP')
    Benchmark.write_entry_table(pd.DataFrame([{
        'entry_id': 'AAAAAA', 'condition_bundle': 'b1_error1_cont0',
        'q2_obs': np.linspace(0.05, 0.5, 20)}]), tmp_path)

    with pytest.raises(ValueError, match='reference list'):
        feature_sidecar(tmp_path)


def test_a_sidecar_that_disagrees_with_its_pool_is_refused():
    """M20 exists on both sides, so recomputing it checks that the peak list, the reference lines
    and the cell are the ones the search used. Every other merit rides on the same three."""
    from mlindex.model_training.BenchmarkRuns import _refuse_a_disagreeing_sidecar

    stored = np.array([10.0, 20.0, 30.0])
    _refuse_a_disagreeing_sidecar(stored.copy(), stored, 'b1_error1_cont0', 'cP')

    with pytest.raises(ValueError, match='disagrees with the pool'):
        _refuse_a_disagreeing_sidecar(np.array([10.0, 20.0, 31.0]), stored,
                                      'b1_error1_cont0', 'cP')


# ---------------------------------------------------------------------------
# Arm against arm
# ---------------------------------------------------------------------------


def test_an_arm_contrast_pairs_two_arms_under_one_score():
    """The comparison every session after P04b needs: arm A is the code before a change, arm B
    after it. `contrast_table` compares two scores inside one arm and `floor_from_arms` compares a
    score-contrast across arms; neither answers this."""
    from mlindex.model_training.BenchmarkRuns import arm_contrast

    before = _floor_reduction([True]*4 + [False]*4)
    after = _floor_reduction([True]*7 + [False])
    table = arm_contrast({'before': before, 'after': after}, 'M20', 'before')
    row = table[(table.scope == 'aggregate') & (table.metric == 'top1')].iloc[0]

    assert row['reference_pct'] == 50.0
    assert row['arm_pct'] == 87.5
    assert row['delta_pp'] == pytest.approx(37.5)
    assert row['n_discordant'] == 3


def test_a_contrast_is_reported_in_multiples_of_the_measured_floor():
    """A gate is read in standard errors of the run-to-run floor, never in percentage points. If
    no floor is supplied the column is NaN rather than a number that looks like one."""
    from mlindex.model_training.BenchmarkRuns import arm_contrast, floors_from_table

    arms = {'before': _floor_reduction([True]*4 + [False]*4),
            'after': _floor_reduction([True]*7 + [False])}
    floor_table = pd.DataFrame([{'score': 'M20', 'metric': 'top1', 'scope': 'aggregate',
                                 'floor_pp': 2.5}])

    with_floor = arm_contrast(arms, 'M20', 'before',
                              floors=floors_from_table(floor_table, score='M20'))
    row = with_floor[(with_floor.scope == 'aggregate') & (with_floor.metric == 'top1')].iloc[0]
    assert row['standard_errors'] == pytest.approx(37.5/2.5)
    # A floor is measured for one metric; another metric is not read against it.
    other = with_floor[(with_floor.scope == 'aggregate') & (with_floor.metric == 'top10')].iloc[0]
    assert np.isnan(other['standard_errors'])
    assert other['verdict'] == ''

    without = arm_contrast(arms, 'M20', 'before')
    assert np.isnan(without[without.metric == 'top1'].iloc[0]['standard_errors'])


def test_an_arm_contrast_counts_what_it_rescued_and_what_it_broke():
    """The net change hides a trade: two crystals rescued and one broken reads as +1 either way."""
    from mlindex.model_training.BenchmarkRuns import arm_contrast

    arms = {'before': _floor_reduction([True, True, False, False]),
            'after': _floor_reduction([True, False, True, True])}
    row = arm_contrast(arms, 'M20', 'before').query("scope == 'aggregate' and metric == 'top1'")
    assert row['n_rescued'].item() == 2
    assert row['n_broken'].item() == 1
    assert row['n_discordant'].item() == 3


@pytest.mark.parametrize('standard_errors, expected', [
    (15.0, 'helps'), (2.01, 'helps'), (2.0, 'does not matter much'), (0.0, 'does not matter much'),
    (-2.0, 'does not matter much'), (-2.01, 'hurts'), (float('nan'), ''),
    ])
def test_the_verdict_follows_the_rule_fixed_before_any_arm_ran(standard_errors, expected):
    from mlindex.model_training.BenchmarkRuns import verdict
    assert verdict(standard_errors) == expected


def test_arms_that_ran_different_candidate_settings_pair_only_when_that_is_the_point():
    shipped = {'lattices': {'cP': {'n_candidates': 100}}}
    halved = {'lattices': {'cP': {'n_candidates': 50}}}
    arms = {'control': _manifest(ensemble=shipped), 'budget_half': _manifest(ensemble=halved)}
    with pytest.raises(ValueError, match='ensemble'):
        Benchmark.manifest_identity(arms)
    assert Benchmark.manifest_identity(arms, allow=('ensemble',))


def test_an_arm_contrast_needs_a_reference_that_exists_and_something_to_compare():
    from mlindex.model_training.BenchmarkRuns import arm_contrast

    arms = {'before': _floor_reduction([True, False])}
    with pytest.raises(ValueError, match='needs two arms'):
        arm_contrast(arms, 'M20', 'before')
    with pytest.raises(ValueError, match='No arm named'):
        arm_contrast({'a': arms['before'], 'b': arms['before']}, 'M20', 'missing')


def test_vary_names_a_field_the_identity_check_actually_compares(tmp_path):
    """Naming a field that is not compared permits nothing, so it is refused rather than accepted
    as though it had widened anything."""
    from mlindex.scripts.run_benchmark import main

    for name in ('a', 'b'):
        directory = tmp_path/name
        entries = _entries()
        frame = Benchmark.label_frame(Benchmark.records_to_frame([_record()]), entries)
        Benchmark.write_candidate_shard(frame, directory, 'b1_error1_cont0', 'cP')
        Benchmark.write_entry_table(entries, directory)
        Benchmark.write_manifest(directory, **_manifest(commit=name))
        Benchmark.stamp_complete(directory, n_source_entries=1, n_bundles=1,
                                 n_patterns_refused=0)

    with pytest.raises(SystemExit, match='does not compare'):
        main(['--stage', 'contrast', '--arm', f'a={tmp_path/"a"}', '--arm', f'b={tmp_path/"b"}',
              '--scores', 'M20', '--vary', 'wallclock'])

    # `commit` IS compared, so naming it permits the pair the floor stage would refuse.
    main(['--stage', 'contrast', '--arm', f'a={tmp_path/"a"}', '--arm', f'b={tmp_path/"b"}',
          '--scores', 'M20', '--vary', 'commit', '--out-dir', str(tmp_path/'out')])
    assert (tmp_path/'out'/'arm_contrast.csv').is_file()


def test_an_arm_can_be_read_from_reduced_tables_and_is_still_identity_checked(tmp_path):
    """Reducing a pool reads every candidate and must run where the pool is; contrasting two arms
    is arithmetic over a few thousand rows and should run anywhere. The manifest travels with the
    tables so the second form is still checked for comparability."""
    from mlindex.scripts.run_benchmark import main, load_arm, arm_manifest

    entries = _entries()
    pool = tmp_path/'pool'
    frame = Benchmark.label_frame(Benchmark.records_to_frame([_record()]), entries)
    Benchmark.write_candidate_shard(frame, pool, 'b1_error1_cont0', 'cP')
    Benchmark.write_entry_table(entries, pool)
    Benchmark.write_manifest(pool, **_manifest())
    Benchmark.stamp_complete(pool, n_source_entries=1, n_bundles=1, n_patterns_refused=0)

    main(['--stage', 'reduce', '--pool', str(pool), '--scores', 'M20',
          '--out-dir', str(tmp_path/'reduced')])

    prefix = tmp_path/'reduced'/'pool'
    assert (tmp_path/'reduced'/'pool_manifest.json').is_file()
    assert arm_manifest(prefix)['commit'] == _manifest()['commit']

    from_pool, _ = load_arm(pool, ['M20'])
    from_tables, entries_back = load_arm(prefix, ['M20'])
    pd.testing.assert_frame_equal(
        from_pool['M20'].reset_index(drop=True),
        from_tables['M20'][from_pool['M20'].columns].reset_index(drop=True))
    # The true lattice rides along, so a per-lattice contrast works without the pool.
    assert 'bravais_lattice_true' in entries_back.columns


def test_an_arm_with_no_manifest_beside_its_tables_is_refused(tmp_path):
    """Tables written before the manifest travelled with them cannot be checked for comparability,
    and pairing them would compare arms nothing has verified are comparable."""
    from mlindex.scripts.run_benchmark import main

    (tmp_path/'a').mkdir()
    for name in ('a', 'b'):
        pd.DataFrame({'entry_id': ['C0'], 'condition_bundle': ['b1_error1_cont0'],
                      'rank_best_correct_all': [0], 'has_correct_all': [True],
                      'n_candidates_all': [5]}).to_csv(
            tmp_path/'a'/f'{name}_per_entry_M20.csv', index=False)

    with pytest.raises(SystemExit, match='No manifest for arm'):
        main(['--stage', 'contrast', '--arm', f'a={tmp_path/"a"/"a"}',
              '--arm', f'b={tmp_path/"a"/"b"}', '--scores', 'M20', '--vary', 'commit'])


FROZEN_SPLIT = Path('docs/fom_campaign2/artifacts/S06_split_manifest.parquet')


@pytest.mark.slow
@pytest.mark.skipif(not FROZEN_SPLIT.is_file(), reason='frozen split not on this machine')
def test_an_arm_divided_into_shards_is_the_arm_undivided(tmp_path, models_dir, monkeypatch):
    """Every shard draws the whole arm and builds its second-phase partners from all of it, so how
    an arm is divided across nodes changes no candidate. A partner pool built per shard would
    leave a one-crystal shard with no partner at all."""
    from mlindex.model_training import BenchmarkRuns
    from mlindex.utilities.Digests import file_digest

    monkeypatch.setattr(BenchmarkRuns, '_provenance', lambda: {
        'commit': 'test', 'models_dir': 'test', 'models_digest': 'test', 'n_model_files': 0,
        'model_revision': 'test'})
    settings = dict(split_sha256=file_digest(FROZEN_SPLIT), true_lattices=['cP', 'hP'],
                    per_lattice=1, bundles=['b1_error1_cont0_phase3', 'b1_error1_cont0'])
    BenchmarkRuns.run_arm(tmp_path/'whole', FROZEN_SPLIT, **settings)
    for shard in (0, 1):
        BenchmarkRuns.run_arm(tmp_path/'sharded', FROZEN_SPLIT, shard=shard, n_shards=2,
                              **settings)
    BenchmarkRuns.finalize_arm(tmp_path/'sharded')

    for bundle in settings['bundles']:
        whole, sharded = (
            Benchmark.load_candidates(tmp_path/name, bundle, sidecars=('merits', 'features'))
            .sort_values(Benchmark.CANDIDATE_KEY, ignore_index=True)
            for name in ('whole', 'sharded'))
        assert whole.shape[0] > 0 and whole['entry_id'].nunique() == 2
        pd.testing.assert_frame_equal(whole, sharded)
    whole, sharded = (Benchmark.load_entries(tmp_path/name).sort_values(
        Benchmark.ENTRY_KEY, ignore_index=True) for name in ('whole', 'sharded'))
    pd.testing.assert_frame_equal(whole, sharded)
    assert whole['second_phase_partner'].notna().sum() == 2


def _cubic_reference(limit=40):
    hkl = np.array([(h, k, l) for h in range(7) for k in range(h + 1) for l in range(k + 1)
                    if 0 < h*h + k*k + l*l <= limit], dtype=float)
    return hkl[np.argsort((hkl**2).sum(axis=1), kind='stable')]


def test_a_true_cell_is_finished_as_the_search_finishes_a_candidate_and_stays_correct():
    """The added true cell is refined against the noisy peaks and assigned an extinction group
    by the search's own steps. It must come out still the true cell, with a real M20."""
    from mlindex.model_training.BenchmarkRuns import refine_true_cell
    from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch

    hkl = _cubic_reference()
    a = 5.0
    q2 = np.unique((hkl**2).sum(axis=1))[:10]/a**2
    q2_obs = q2*(1 + np.random.default_rng(0).normal(0, 2e-4, q2.size))
    opt_params = {'minimum_uc': 2, 'maximum_uc': 500, 'assignment_threshold': 0.95,
                  'figure_of_merit': 'M20'}
    candidates = refine_true_cell(q2_obs, [a, a, a, np.pi/2, np.pi/2, np.pi/2], 'cP', 'cubic',
                                  hkl, opt_params, np.random.default_rng(1))
    cell = 1/np.sqrt(candidates.best_xnn[:, 0])
    assert is_correct_known_bl_batch(np.array([a]), cell[:, np.newaxis], 'cubic').all()
    assert candidates.best_M20[0] > 10 and candidates.n_indexed[0] >= 8
    assert len(candidates.best_spacegroup) == 1
