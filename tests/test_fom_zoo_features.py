"""The S06 feature matrix is only useful if it is the zoo, computed on the right lines.

`zoo_features` rebuilds each candidate's calculated lines from the dumped columns and evaluates
every merit on them. Two things can go wrong silently: the line assignment can be rebuilt
differently from the way the pipeline built it, in which case every merit is computed on the wrong
lines and nothing complains; and `get_M20`'s in-place write into `q2_ref_calc` can corrupt the
merits evaluated after it. Both are checked here rather than assumed.

The M20 round trip is the load-bearing one, and it is exact rather than approximate: `compute_all`
runs `get_M20` last and on a copy, so its M20 must equal the value the pipeline stored.
"""
import numpy as np
import pandas as pd
import pytest

from test_fom_benchmark_roundtrip import _synthetic_pool
from test_fom_benchmark_roundtrip import _xnn
from conftest import load_test_case

from mlindex.model_training import FomBenchmark as fb
from mlindex.utilities.FigureOfMerits import SIGMA_TREATMENT


@pytest.fixture(scope='module')
def zoo_pool(models_dir, unique_test_metadata):
    """A few synthetic pools with their entry table, enough to exercise several lattice systems."""
    if models_dir is None:
        pytest.skip('models directory not available')
    records = []
    entry_rows = []
    for _, row in unique_test_metadata.iterrows():
        q2_obs, unit_cell, _, bravais_lattice, lattice_system = load_test_case(row)
        if q2_obs.size < 20:
            continue
        records.append(_synthetic_pool(
            models_dir, q2_obs, unit_cell, bravais_lattice, lattice_system, n_candidates=12,
            ))
        entry_rows.append({
            'entry_id': records[-1]['context']['entry_id'],
            'q2_digest': records[-1]['q2_digest'],
            'q2_obs': np.asarray(q2_obs, dtype=np.float64),
            'split': 'fom-train',
            'unit_cell_true': np.asarray(unit_cell, dtype=np.float64),
            'xnn_true': _xnn(unit_cell, lattice_system)[0],
            'bravais_lattice_true': bravais_lattice,
            })
        if len(records) >= 4:
            break
    if not records:
        pytest.skip('no test case has twenty peaks')
    return fb.records_to_frame(records), pd.DataFrame(entry_rows)


def test_assign_lines_takes_q2_calc_out_of_q2_ref_calc(zoo_pool):
    """q2_calc must be the reference lines the assignment chose, not an independent evaluation."""
    candidates, entries = zoo_pool
    group = candidates.groupby(
        ['lattice_system', 'bravais_lattice', 'spacegroup'], sort=False
        ).get_group((candidates['lattice_system'].iloc[0], candidates['bravais_lattice'].iloc[0],
                     candidates['spacegroup'].iloc[0]))
    q2_obs = np.asarray(
        entries.set_index('entry_id')['q2_obs'].loc[group['entry_id'].iloc[0]], dtype=np.float64
        )[:int(group['n_peaks'].iloc[0])]
    xnn = np.vstack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
    q2_ref_calc, hkl_assign, hkl, q2_calc = fb.assign_lines(
        q2_obs, xnn, group['lattice_system'].iloc[0], group['bravais_lattice'].iloc[0],
        group['spacegroup'].iloc[0],
        )
    assert q2_calc.shape == (xnn.shape[0], q2_obs.size)
    assert hkl.shape == (xnn.shape[0], q2_obs.size, 3)
    np.testing.assert_array_equal(
        q2_calc, np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
        )


def test_zoo_features_reproduces_the_stored_M20_exactly(zoo_pool):
    """The gate. A mismatch means the lines were rebuilt wrongly, not that the tolerance is tight.

    `get_M20` is evaluated last and on a copy inside `compute_all`, so if the rest of the zoo had
    corrupted `q2_ref_calc` this is the column that would show it.
    """
    candidates, entries = zoo_pool
    features, _ = fb.zoo_features(candidates, entries)
    residual = np.abs(features['M20'].to_numpy() - candidates['M20'].to_numpy())
    scale = np.maximum(np.abs(candidates['M20'].to_numpy()), 1e-12)
    assert np.max(residual/scale) < 1e-9


def test_zoo_features_is_invariant_to_row_order(zoo_pool):
    """Grouping is an implementation detail; a shuffled pool must give the same feature per row."""
    candidates, entries = zoo_pool
    features, _ = fb.zoo_features(candidates, entries)
    shuffled = candidates.sample(frac=1.0, random_state=3).reset_index(drop=True)
    other, _ = fb.zoo_features(shuffled, entries)

    keys = list(fb.ZOO_KEY_COLUMNS)
    keys = [key for key in keys if key in features.columns]
    left = features.set_index(keys).sort_index()
    right = other.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(left, right[left.columns])


def test_every_feature_column_declares_a_sigma_treatment(zoo_pool):
    """A column that quietly defaults to 'free' is how an assumed sigma gets reported as free."""
    candidates, entries = zoo_pool
    features, treatments = fb.zoo_features(candidates, entries)
    computed = [column for column in features.columns if column not in fb.ZOO_KEY_COLUMNS]
    assert set(computed) == set(treatments)
    assert set(treatments.values()) <= {'free', 'in-sample', 'assumed'}
    # The fallback in compute_all must not be load-bearing: every base name is in the table.
    for name in computed:
        base = name.split('_pvalue')[0].split('_scale')[0]
        assert base in SIGMA_TREATMENT, f'{name} relies on the default treatment'


def test_g_min_scales_V_over_Vcrit_and_leaves_the_werner_ranking_alone(zoo_pool):
    """Q14's escape route, asserted rather than argued.

    V_crit is proportional to 1/g_min, so V/V_crit is linear in g_min and any other floor is a
    rescale of the stored column -- the whole sweep is free. M_werner_frac picks up the same
    global factor, so its *ranking within an entry* does not depend on g_min at all, which is why
    it can be the headline without a floor ever being chosen.
    """
    candidates, entries = zoo_pool
    one, _ = fb.zoo_features(candidates, entries, g_min=1.0)
    two, _ = fb.zoo_features(candidates, entries, g_min=2.0)

    np.testing.assert_allclose(
        two['V_over_Vcrit'].to_numpy(), 2*one['V_over_Vcrit'].to_numpy(), rtol=1e-12,
        )
    np.testing.assert_allclose(
        two['M_werner_frac'].to_numpy(), 2*one['M_werner_frac'].to_numpy(), rtol=1e-12,
        )
    ranks_one = one.groupby('entry_id')['M_werner_frac'].rank(method='first')
    ranks_two = two.groupby('entry_id')['M_werner_frac'].rank(method='first')
    pd.testing.assert_series_equal(ranks_one, ranks_two)


def test_min_discrepancy_only_moves_the_information_merits(zoo_pool):
    """The floor is a property of the input data (F-026), so it must not leak into the rest."""
    candidates, entries = zoo_pool
    bare, _ = fb.zoo_features(candidates, entries, min_discrepancy=0.0)
    floored, _ = fb.zoo_features(candidates, entries, min_discrepancy=1e-3)

    affected = {'M_info_clipped', 'null_tail_nll'}
    for column in bare.columns:
        if column in fb.ZOO_KEY_COLUMNS or column in affected:
            continue
        np.testing.assert_allclose(
            floored[column].to_numpy(), bare[column].to_numpy(), rtol=1e-12,
            err_msg=f'{column} moved with min_discrepancy',
            )
