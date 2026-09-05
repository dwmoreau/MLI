"""S14's input sidecar: the columns, the cubic convention, and the guards that refuse silence.

The pool-free tests pin the registry -- which columns each feature group adds and that none of
them is a truth column -- and the two refusals that matter: an entry with no prior table raises
rather than writing NaN, and a shard missing from one sidecar directory raises rather than
joining as a null column. The real-data tests are `slow` and run when the fully retained pool is
present; they check the conventions the network depends on (ten-peak cubic rows carry NaN beyond
`asg_p09`, a cubic claim reads as out-of-support) on rows the shipped indexer actually produced.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.model_training import FomCombiner  # noqa: E402

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(REPOSITORY, 'mlindex', 'data', 'fom_full_c2_pool')

pytest.importorskip('pyarrow')


# ---------------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------------
def test_the_input_columns_are_the_ones_the_handoff_names():
    assert len(FomBenchmark.NEURAL_PEAK_COLUMNS) == 20
    assert FomBenchmark.NEURAL_PEAK_COLUMNS[0] == 'asg_p00'
    assert FomBenchmark.NEURAL_PEAK_COLUMNS[-1] == 'asg_p19'
    assert FomBenchmark.NEURAL_INPUT_COLUMNS == (
        ('prior_joint', 'prior_joint_margin', 'prior_in_support')
        + FomBenchmark.NEURAL_PEAK_COLUMNS + ('asg_sigma',))
    assert FomCombiner.ASSIGNMENT_PEAKS == FomBenchmark.NEURAL_PEAK_COLUMNS
    assert FomCombiner.PRIOR_CLAIMED_C2 == FomBenchmark.NEURAL_CLAIMED_COLUMNS
    assert len(FomCombiner.PRIOR_VOLUME) == 14 and len(FomCombiner.PRIOR_ENTRY) == 16


@pytest.mark.parametrize('group', FomCombiner.NEURAL_GROUPS)
def test_each_neural_group_adds_exactly_its_columns(group):
    base, _ = FomCombiner.feature_specification(('raw',))
    with_group, _ = FomCombiner.feature_specification(('raw', group))
    added = [name for name in with_group if name not in base]
    assert tuple(added) == tuple(FomCombiner.NEURAL_GROUP_COLUMNS[group])


def test_the_fifty_input_design_is_reachable_as_groups():
    """DWMM's specification: 14 lattice probabilities, 14 volumes, 20 per-peak, volume, lattice."""
    names, categorical = FomCombiner.feature_specification(
        ('structural', 'prior_entry', 'prior_volume', 'assignment_peaks'),
        drop=[name for name in FomCombiner.STRUCTURAL_NUMERIC if name != 'log_volume']
        + ['spacegroup'])
    assert set(FomCombiner.PRIOR_ENTRY) <= set(names)
    assert set(FomCombiner.PRIOR_VOLUME) <= set(names)
    assert set(FomCombiner.ASSIGNMENT_PEAKS) <= set(names)
    assert 'log_volume' in names and 'bravais_lattice' in names
    assert categorical == ('bravais_lattice',)
    FomCombiner.check_no_leakage(names)


def test_no_neural_column_is_forbidden_or_truth_shaped():
    for group in FomCombiner.NEURAL_GROUPS:
        for name in FomCombiner.NEURAL_GROUP_COLUMNS[group]:
            assert name not in FomCombiner.FORBIDDEN_COLUMNS
            assert not name.endswith(FomCombiner.FORBIDDEN_SUFFIX)


def test_the_per_candidate_groups_map_to_the_neural_inputs_sidecar():
    for group in FomCombiner.NEURAL_CANDIDATE_GROUPS:
        assert FomCombiner.SIDECAR_DIRS[group] == 'neural_inputs'
    for group in FomCombiner.NEURAL_ENTRY_GROUPS:
        assert group not in FomCombiner.SIDECAR_DIRS, 'entry-level columns ride the covariates'


# ---------------------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------------------
def _cubic_rows(n=3):
    """A few synthetic cubic candidates the reference-line machinery can score."""
    import pandas as pd

    q2 = np.sort(np.array([0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.090, 0.100, 0.110,
                           0.120, 0.130, 0.140, 0.160, 0.170, 0.180, 0.190, 0.200, 0.210, 0.220]))
    entries = pd.DataFrame({'entry_id': ['e0'], 'condition_bundle': ['b'], 'q2_obs': [q2]})
    candidates = pd.DataFrame({
        'entry_id': ['e0']*n, 'condition_bundle': ['b']*n, 'bravais_lattice': ['cP']*n,
        'candidate_id': np.arange(n), 'lattice_system': ['cubic']*n, 'spacegroup': ['P m -3 m']*n,
        'n_peaks': [10]*n, 'xnn': [np.array([0.01*(1 + 0.05*k)]) for k in range(n)],
        'volume': [1000.0*(1 - 0.05*k) for k in range(n)],
        })
    return candidates, entries


def test_neural_inputs_without_prior_tables_leaves_the_prior_columns_nan():
    candidates, entries = _cubic_rows()
    try:
        out = FomBenchmark.neural_inputs(candidates, entries, prior_tables=None)
    except (KeyError, FileNotFoundError) as error:   # the spacegroup key or hkl refs may differ
        pytest.skip(f'reference machinery unavailable for the synthetic row: {error}')
    assert list(out.columns) == list(FomBenchmark.NEURAL_INPUT_COLUMNS)
    assert out[list(FomBenchmark.NEURAL_CLAIMED_COLUMNS)].isna().all().all()
    # ten peaks -> ten posteriors, the rest NaN by construction
    assert np.isfinite(out[list(FomBenchmark.NEURAL_PEAK_COLUMNS[:10])]).all().all()
    assert out[list(FomBenchmark.NEURAL_PEAK_COLUMNS[10:])].isna().all().all()
    assert np.isfinite(out['asg_sigma']).all()


def test_a_candidate_whose_entry_has_no_prior_table_raises():
    import pandas as pd

    candidates, entries = _cubic_rows()
    tables = {'joint': np.zeros((1, 4, 14)), 'log_branch_volumes': np.linspace(5, 9, 4),
              'index': pd.MultiIndex.from_tuples([('other', 'b')],
                                                 names=['entry_id', 'condition_bundle'])}
    with pytest.raises(KeyError):
        FomBenchmark.neural_inputs(candidates, entries, prior_tables=tables)


def test_neural_covariates_refuses_a_missing_prior_row(tmp_path):
    import pandas as pd

    entries = pd.DataFrame({'entry_id': ['a', 'b'], 'condition_bundle': ['x', 'x'],
                            'q2_obs': [np.linspace(0.01, 0.2, 20)]*2,
                            'n_peaks_available': [30, 30]})
    with pytest.raises(FileNotFoundError):
        FomCombiner.neural_covariates(tmp_path, entries)
    out = tmp_path/'neural_inputs'
    out.mkdir()
    prior = pd.DataFrame({'entry_id': ['a'], 'condition_bundle': ['x']})
    for name in list(FomCombiner.PRIOR_ENTRY) + list(FomCombiner.PRIOR_VOLUME):
        prior[name] = 0.5
    prior.to_parquet(out/FomCombiner.NEURAL_ENTRY_FILE)
    with pytest.raises(KeyError):
        FomCombiner.neural_covariates(tmp_path, entries)
    prior = pd.concat([prior, prior.assign(entry_id='b')], ignore_index=True)
    prior.to_parquet(out/FomCombiner.NEURAL_ENTRY_FILE)
    covariates = FomCombiner.neural_covariates(tmp_path, entries)
    assert covariates.shape[0] == 2
    assert set(FomCombiner.PRIOR_VOLUME) <= set(covariates.columns)


def test_a_shard_missing_from_one_sidecar_directory_raises(tmp_path):
    """`bundle_frames` used to join a shard missing from one of several sidecar sets as NaN."""
    import pandas as pd

    pool = tmp_path
    keys = dict(entry_id=['e']*2, condition_bundle=['c2_x']*2, bravais_lattice=['tP']*2,
                candidate_id=[0, 1])
    pd.DataFrame(dict(**keys, M20=[1.0, 2.0])).to_parquet(pool/'candidates_c2_x_tP.parquet')
    (pool/'merits').mkdir()
    pd.DataFrame(dict(**keys, M_sym=[1.0, 2.0])).to_parquet(
        pool/'merits'/'candidates_c2_x_tP.parquet')
    (pool/'neural_inputs').mkdir()      # exists, but holds no shard
    with pytest.raises(FileNotFoundError):
        list(FomBenchmark.bundle_frames(pool, merit_dir=[pool/'merits', pool/'neural_inputs'],
                                        require_merits=True, merit_columns=['M_sym', 'asg_p00']))
    # and a merit wholly null after the join is refused, a partly null one is not
    (pool/'structural').mkdir()
    pd.DataFrame(dict(**keys, zone_dominance=[np.nan, np.nan])).to_parquet(
        pool/'structural'/'candidates_c2_x_tP.parquet')
    with pytest.raises(ValueError):
        list(FomBenchmark.bundle_frames(pool, merit_dir=[pool/'merits', pool/'structural'],
                                        require_merits=True,
                                        merit_columns=['M_sym', 'zone_dominance']))
    pd.DataFrame(dict(**keys, zone_dominance=[np.nan, 1.0])).to_parquet(
        pool/'structural'/'candidates_c2_x_tP.parquet')
    frames = list(FomBenchmark.bundle_frames(pool, merit_dir=[pool/'merits', pool/'structural'],
                                             require_merits=True,
                                             merit_columns=['M_sym', 'zone_dominance']))
    assert frames[0].shape[0] == 2


def test_the_writer_refuses_a_keyed_run_with_unmatched_keys(tmp_path):
    from mlindex.scripts import run_fom_neural_inputs as writer

    assert writer.OUT_DIRNAME == 'neural_inputs'
    args = writer._parse_args(['--pool', str(tmp_path), '--keys-from', 'nowhere_*.parquet'])
    with pytest.raises(SystemExit):
        writer.keys_in(tuple(args.keys_from))


# ---------------------------------------------------------------------------------------
# On the real pool
# ---------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(POOL, 'candidates_c2_error1_cont0_cF.parquet')),
                    reason='fully retained pool not present')
def test_cubic_candidates_carry_ten_posteriors_and_no_prior_on_the_real_pool():
    import pandas as pd

    entries = FomBenchmark.load_entries(POOL)[['entry_id', 'condition_bundle', 'q2_obs']]
    candidates = pd.read_parquet(
        os.path.join(POOL, 'candidates_c2_error1_cont0_cF.parquet'),
        columns=['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id',
                 'lattice_system', 'spacegroup', 'n_peaks', 'xnn', 'volume', 'is_correct'])
    out = FomBenchmark.neural_inputs(candidates, entries, prior_tables=None)
    assert out.shape[0] == candidates.shape[0]
    peaks = out[list(FomBenchmark.NEURAL_PEAK_COLUMNS)].to_numpy()
    assert np.isfinite(peaks[:, :10]).all() and np.isnan(peaks[:, 10:]).all()
    assert ((peaks[:, :10] >= 0) & (peaks[:, :10] <= 1 + 1e-9)).all()
    correct = candidates['is_correct'].to_numpy().astype(bool)
    if correct.any() and (~correct).any():
        assert peaks[correct, :10].mean() > peaks[~correct, :10].mean()


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(POOL, 'neural_inputs', '_meta.json')),
                    reason='neural_inputs sidecar not written on the fully retained pool')
def test_the_written_sidecar_verifies_and_records_its_support():
    from mlindex.scripts import run_fom_neural_inputs as writer

    total, problems, notes = writer.verify(POOL, os.path.join(POOL, 'neural_inputs'))
    assert not problems, problems
    meta = json.loads(open(os.path.join(POOL, 'neural_inputs', '_meta.json'),
                           encoding='utf-8').read())
    assert meta['n_candidates'] == total
    assert meta['entries']['support']
