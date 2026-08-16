"""The FOM benchmark dump is only useful if it is complete.

The acceptance gate for S03 is that M20, Minfo and n_indexed recompute from the dumped
columns alone and agree with what the pipeline stored. These tests are the fast half of
that: they check the pieces the round trip rests on -- that the reference lines regenerate
from (Bravais lattice, extinction group) rather than needing to be stored, that parquet
preserves the ragged float64 columns exactly, and that a mis-joined shard is caught. The
full-pipeline half is `mlindex/scripts/run_fom_dump.py` plus the gate script, which needs
models and minutes rather than seconds.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import BL_TO_LATTICE_SYSTEM
from conftest import load_test_case

from mlindex.model_training import FomBenchmark as fb
from mlindex.utilities.FigureOfMerits import get_M20
from mlindex.utilities.FigureOfMerits import get_M20_likelihood_from_xnn
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref
from mlindex.utilities.UnitCellTools import get_partial_unit_cell
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell


ASSIGNMENT_THRESHOLD = 0.95


def _xnn(unit_cell, lattice_system):
    partial = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
    return get_xnn_from_unit_cell(
        partial[np.newaxis], partial_unit_cell=True, lattice_system=lattice_system
        )


def _synthetic_pool(models_dir, q2_obs, unit_cell, bravais_lattice, lattice_system,
                    n_candidates=40, seed=0):
    """A candidate pool around the true cell, scored exactly as the pipeline scores one.

    Not a substitute for the pipeline -- it exists so the schema can be exercised for all
    fourteen Bravais lattices without loading models or spending minutes.
    """
    rng = np.random.default_rng(seed)
    xnn_true = _xnn(unit_cell, lattice_system)
    xnn = xnn_true * (1 + rng.normal(scale=0.01, size=(n_candidates, xnn_true.shape[1])))
    xnn = np.vstack([xnn_true, xnn])

    hkl_ref = np.load(
        models_dir / f'{lattice_system}_1' / 'data' / f'hkl_ref_{bravais_lattice}.npy')
    spacegroups = get_spacegroup_hkl_ref(hkl_ref, bravais_lattice=bravais_lattice)
    spacegroup = sorted(spacegroups.keys())[0]
    hkl_ref_sg = spacegroups[spacegroup]

    q2_ref_calc = Q2Calculator(
        lattice_system=lattice_system, hkl=hkl_ref_sg, tensorflow=False,
        representation='xnn').get_q2(xnn)
    hkl_assign = fast_assign(q2_obs, q2_ref_calc)
    hkl = np.take(hkl_ref_sg, hkl_assign, axis=0)
    q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
    _, probability, Minfo = get_M20_likelihood_from_xnn(
        q2_obs=q2_obs, xnn=xnn, hkl=hkl,
        lattice_system=lattice_system, bravais_lattice=bravais_lattice)
    n_indexed = np.sum(probability > ASSIGNMENT_THRESHOLD, axis=1, dtype=int)
    M20 = get_M20(q2_obs, q2_calc, q2_ref_calc)

    unit_cells = get_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system)
    reciprocal = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system)
    order = np.argsort(M20)[::-1]
    final_rank = np.empty(M20.shape[0], dtype=int)
    final_rank[order] = np.arange(M20.shape[0])
    return {
        'bravais_lattice': bravais_lattice,
        'lattice_system': lattice_system,
        'q2_digest': fb.q2_digest(q2_obs),
        'context': {'entry_id': f'test-{bravais_lattice}'},
        'n_peaks': int(q2_obs.size),
        'hkl_ref_length': int(hkl_ref.shape[0]),
        'n_entering': int(xnn.shape[0]),
        'assignment_threshold': ASSIGNMENT_THRESHOLD,
        'downsample_radius': 1e-4,
        'xnn': xnn,
        'unit_cell': unit_cells,
        'volume': get_unit_cell_volume(
            unit_cells, partial_unit_cell=True, lattice_system=lattice_system),
        'reciprocal_volume': get_unit_cell_volume(
            reciprocal, partial_unit_cell=True, lattice_system=lattice_system),
        'M20': M20,
        'Minfo': Minfo,
        'n_indexed': n_indexed,
        'spacegroup': [spacegroup] * xnn.shape[0],
        'final_rank': final_rank,
        'in_top_n': final_rank < 20,
        }


@pytest.fixture(scope='module')
def pool(models_dir, unique_test_metadata):
    """One synthetic pool per Bravais lattice, plus the matching entry table."""
    if models_dir is None:
        pytest.skip('models directory not available')
    records = []
    entry_rows = []
    for _, row in unique_test_metadata.iterrows():
        q2_obs, unit_cell, _, bravais_lattice, lattice_system = load_test_case(row)
        if q2_obs.size < 20:
            continue
        record = _synthetic_pool(
            models_dir, q2_obs, unit_cell, bravais_lattice, lattice_system)
        records.append(record)
        entry_rows.append({
            'entry_id': record['context']['entry_id'],
            'q2_digest': record['q2_digest'],
            'q2_obs': np.asarray(q2_obs, dtype=np.float64),
            'split': 'fom-train',
            'unit_cell_true': np.asarray(unit_cell, dtype=np.float64),
            'xnn_true': _xnn(unit_cell, lattice_system)[0],
            'volume_true': float(get_unit_cell_volume(
                unit_cell[np.newaxis], partial_unit_cell=False,
                lattice_system=lattice_system)[0]),
            'bravais_lattice_true': bravais_lattice,
            })
    return fb.records_to_frame(records), pd.DataFrame(entry_rows)


def test_records_to_frame_has_the_declared_schema(pool):
    candidates, _ = pool
    assert list(candidates.columns) == list(fb.CANDIDATE_COLUMNS)
    assert candidates.shape[0] > 0
    # xnn is a ragged list column, one to six components depending on lattice system.
    widths = {len(value) for value in candidates['xnn']}
    assert widths <= {1, 2, 3, 4, 6}


def test_records_to_frame_drops_empty_lattices():
    empty = {'xnn': np.zeros((0, 3)), 'context': {}, 'q2_digest': 'x'}
    assert fb.records_to_frame([empty]).shape[0] == 0


def test_reference_lines_match_a_direct_q2calculator(models_dir, unique_test_metadata):
    """The dump stores an extinction-group name instead of ~1000 calculated lines per row.

    That is only sound if the name regenerates the same lines, which is what this checks.
    """
    if models_dir is None:
        pytest.skip('models directory not available')
    for _, row in unique_test_metadata.iterrows():
        q2_obs, unit_cell, _, bravais_lattice, lattice_system = load_test_case(row)
        xnn = _xnn(unit_cell, lattice_system)
        hkl_ref = np.load(
            models_dir / f'{lattice_system}_1' / 'data' / f'hkl_ref_{bravais_lattice}.npy')
        spacegroups = get_spacegroup_hkl_ref(hkl_ref, bravais_lattice=bravais_lattice)
        spacegroup = sorted(spacegroups.keys())[0]
        expected = Q2Calculator(
            lattice_system=lattice_system, hkl=spacegroups[spacegroup],
            tensorflow=False, representation='xnn').get_q2(xnn)
        actual = fb.reference_lines(
            xnn, lattice_system, bravais_lattice, spacegroup, models_directory=models_dir)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                                   err_msg=f'reference lines differ for {bravais_lattice}')


def test_scores_recompute_from_the_dumped_columns(pool, models_dir):
    """The acceptance gate, on a synthetic pool spanning every Bravais lattice."""
    candidates, entries = pool
    checked = fb.recompute_frame(candidates, entries, models_directory=models_dir)
    # Every Bravais lattice with a 20-peak tutorial pattern must be represented, or the
    # gate is being met on a subset of the lattice systems.
    assert set(checked['bravais_lattice']) <= set(BL_TO_LATTICE_SYSTEM)
    assert checked['lattice_system'].nunique() == len(set(BL_TO_LATTICE_SYSTEM.values()))
    np.testing.assert_allclose(
        checked['M20_recomputed'], checked['M20'], rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        checked['Minfo_recomputed'], checked['Minfo'], rtol=1e-6, atol=0)
    # n_indexed is a count over a hard 0.95 threshold, so it is exact or it is wrong.
    mismatched = checked.loc[checked['n_indexed'] != checked['n_indexed_recomputed']]
    assert mismatched.empty, (
        f'{mismatched.shape[0]} rows disagree on n_indexed, '
        f'lattices {sorted(set(mismatched["bravais_lattice"]))}'
        )


def test_parquet_round_trip_preserves_the_scores(pool, tmp_path, models_dir):
    """float32 storage would silently break the 1e-6 gate; list columns must stay ragged."""
    pytest.importorskip('pyarrow')
    candidates, entries = pool
    fb.write_candidate_shard(candidates, tmp_path, 'unit')
    fb.write_entry_table(entries, tmp_path, 'unit')

    reloaded = fb.load_candidates(tmp_path)
    assert reloaded.shape[0] == candidates.shape[0]
    for column in ('M20', 'Minfo', 'volume', 'reciprocal_volume'):
        np.testing.assert_array_equal(
            reloaded[column].to_numpy(), candidates[column].to_numpy())
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(v) for v in reloaded['xnn']]),
        np.concatenate([np.asarray(v) for v in candidates['xnn']]))

    checked = fb.recompute_frame(reloaded, fb.load_entries(tmp_path),
                                 models_directory=models_dir)
    np.testing.assert_allclose(
        checked['M20_recomputed'], checked['M20'], rtol=1e-6, atol=0)


def test_join_rejects_shards_from_different_runs(pool, tmp_path):
    pytest.importorskip('pyarrow')
    candidates, entries = pool
    tampered = entries.copy()
    tampered.loc[0, 'q2_digest'] = 'deadbeefdeadbeef'
    fb.write_candidate_shard(candidates, tmp_path, 'unit')
    fb.write_entry_table(tampered, tmp_path, 'unit')
    with pytest.raises(ValueError, match='q2_digest'):
        fb.load_benchmark(tmp_path, label=False)


def test_q2_digest_separates_peak_lists():
    a = np.linspace(0.1, 1.0, 20)
    b = a.copy()
    b[7] += 1e-9
    assert fb.q2_digest(a) == fb.q2_digest(a.copy())
    assert fb.q2_digest(a) != fb.q2_digest(b)


def test_labels_find_the_true_cell(pool, models_dir):
    """The pool is seeded with the true cell, so at least one candidate must be correct."""
    candidates, entries = pool
    labelled = fb.label_frame(candidates, entries)
    for bravais_lattice, group in labelled.groupby('bravais_lattice'):
        assert bool(group['is_correct'].any()), (
            f'no candidate labelled correct for {bravais_lattice}, '
            'though the true cell is in the pool'
            )


def test_dumping_is_off_by_default():
    """PROTOCOL 5: anything on the production inference path is behind a flag, default off."""
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    import inspect

    source = inspect.getsource(OptimizerManager.__init__)
    assert "'dump_candidates': None" in source
