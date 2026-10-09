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
