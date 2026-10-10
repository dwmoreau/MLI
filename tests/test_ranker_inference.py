"""The packaged learned ranker, checked against what the research path computed.

`tests/expected/ranker_agreement.npz` holds one pattern per true lattice of a stored benchmark
pool: every candidate it keeps at the ranker's cut, what the indexer knows of each, the sixteen
inputs the research export computed, and the scikit-learn fit's raw and calibrated scores
(`generate_expected.py --ranker`). The package must reproduce them from the ONNX file and the
calibrators alone.
"""

from pathlib import Path

import numpy as np
import pytest

from mlindex.utilities.Ranker import LATTICE_FEATURE
from mlindex.utilities.Ranker import RANKER_DIRECTORY
from mlindex.utilities.Ranker import LearnedRanker

EXPECTED = Path(__file__).parent / 'expected' / 'ranker_agreement.npz'


@pytest.fixture(scope='module')
def agreement():
    with np.load(EXPECTED) as arrays:
        return {name: arrays[name] for name in arrays.files}


@pytest.fixture(scope='module')
def ranker(models_dir):
    if models_dir is None or not (models_dir / RANKER_DIRECTORY).is_dir():
        pytest.skip('no packaged ranker in the model tree')
    return LearnedRanker.load(models_dir / RANKER_DIRECTORY)


def _exported_inputs(agreement):
    columns = {str(name): agreement['inputs'][:, i]
               for i, name in enumerate(agreement['features'])}
    columns[LATTICE_FEATURE] = agreement['bravais_lattice']
    return columns


def test_the_packaged_ranker_reads_the_inputs_it_was_fitted_on(ranker, agreement):
    assert ranker.features == tuple(str(name) for name in agreement['features'])
    assert ranker.encoding == 'ordinal'


def test_the_packaged_ranker_scores_as_its_fit(ranker, agreement):
    """ONNX against scikit-learn, raw and calibrated, on 26 550 candidates of 14 patterns."""
    columns = _exported_inputs(agreement)
    raw = ranker.predict_batch(ranker.design_matrix(columns))
    np.testing.assert_allclose(raw, agreement['raw'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ranker.score(columns), agreement['calibrated'], rtol=0,
                               atol=1e-12)


def _pattern_pools(agreement, models_dir, pattern):
    """The pattern's candidates, lattice by lattice, as the indexer hands them to `score_pool`."""
    from mlindex.utilities.Q2Calculator import Q2Calculator
    from mlindex.utilities.Ranker import candidate_inputs
    from mlindex.utilities.SpaceGroups import get_spacegroup_keep_masks

    pools, rows = {}, {}
    in_pattern = agreement['pattern'] == pattern
    for lattice in dict.fromkeys(agreement['bravais_lattice'][in_pattern].tolist()):
        index = np.flatnonzero(in_pattern & (agreement['bravais_lattice'] == lattice))
        system = str(agreement['lattice_system'][index[0]])
        q2_obs = agreement['q2_obs'][pattern][:int(agreement['n_peaks'][index[0]])]
        xnn = agreement['xnn'][index]
        xnn = xnn[:, ~np.isnan(xnn[0])]
        hkl_ref = np.load(models_dir / f'{system}_1' / 'data' / f'hkl_ref_{lattice}.npy')
        q2_ref_calc_full = Q2Calculator(lattice_system=system, hkl=hkl_ref, tensorflow=False,
                                        representation='xnn').get_q2(xnn)
        spacegroups = agreement['spacegroup'][index]
        pools[lattice] = dict(
            candidate_inputs(q2_obs, xnn, q2_ref_calc_full, spacegroups,
                             get_spacegroup_keep_masks(hkl_ref, lattice), system, lattice),
            M20=agreement['M20'][index], n_indexed=agreement['n_indexed'][index],
            spacegroup=spacegroups)
        rows[lattice] = index
    return pools, rows


def test_the_inputs_the_indexer_computes_are_the_ones_the_ranker_was_fitted_on(models_dir,
                                                                               agreement):
    """`candidate_inputs` and the context gaps, from cells, groups and peaks alone, against the
    research export's values for the same candidates."""
    from mlindex.utilities.Ranker import context_gaps

    features = [str(name) for name in agreement['features']]
    for pattern in np.unique(agreement['pattern']):
        pools, rows = _pattern_pools(agreement, models_dir, pattern)
        columns = {name: np.concatenate([pools[lattice][name] for lattice in pools])
                   for name in pools[next(iter(pools))]}
        columns.update(context_gaps(columns))
        index = np.concatenate([rows[lattice] for lattice in pools])
        for name in features:
            if name in (LATTICE_FEATURE, 'group_frequency'):
                continue
            np.testing.assert_allclose(
                np.asarray(columns[name], dtype=np.float32),
                agreement['inputs'][index, features.index(name)], rtol=1e-6, atol=1e-6,
                err_msg=f'{name}, pattern {pattern}')


def test_the_indexer_scores_a_pattern_as_the_fit_does(ranker, models_dir, agreement):
    """`score_pool` on the indexer's inputs, one batch a pattern, against the fit's calibrated
    probabilities."""
    from mlindex.utilities.Ranker import score_pool

    for pattern in np.unique(agreement['pattern']):
        pools, rows = _pattern_pools(agreement, models_dir, pattern)
        scores = score_pool(pools, ranker)
        for lattice, index in rows.items():
            np.testing.assert_allclose(scores[lattice], agreement['calibrated'][index], rtol=0,
                                       atol=1e-9, err_msg=f'{lattice}, pattern {pattern}')
