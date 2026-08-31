"""S09's zoo driver: the merit set it admits, and the two refusals that keep its numbers honest.

The driver's numbers are only as good as two guards, and both fail *silently* if they are removed:

  * the pool's retention depth must reach `evaluate` explicitly. An iterable of frames carries no
    manifest, so `subsample_top_k='auto'` would take a subsampled pool for a fully retained one and
    certify a rank it cannot answer -- the C2-F-077 failure, through the back door.
  * `--unfloored` must be refused on a subsampled pool. The subsampler ranked on the *floored*
    `M_rev`, so the rows an unfloored ranking would put first are the ones it discarded at 95 %.
    The arm would come out flattered and would understate what the floor is worth (C2-F-084).

Neither raises on its own. Both are tested here because a wrong number that looks right is the
failure mode this campaign keeps paying for.
"""

import pytest

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics
from mlindex.scripts import run_fom_zoo_eval as zoo

SLICE_ROOT = FomBenchmark.Path(__file__).parent.parent/'mlindex'/'data'/'fom_benchmark_c2'


def test_the_zoo_is_exactly_the_merits_the_subsampler_ranked_on():
    """No caveated column in the main table -- that is the point of the cut, not a coincidence."""
    assert zoo.MERITS == FomBenchmark.REDUCED_MERIT_COLUMNS
    assert set(zoo.MERITS) <= set(FomMetrics.RANK_EXACT_MERITS)
    # Every one has a recorded direction, so none can be ranked backwards (C2-F-085).
    for merit in zoo.MERITS:
        assert isinstance(FomMetrics.orientation_of(merit), bool)
    # The probation merits and the negative control are gone, and ho_M20 is S10's.
    for dropped in ('M_wu', 'M_1', 'F_N_q', 'nll_exponential', 'ho_M20'):
        assert dropped not in zoo.MERITS


def test_the_unfloored_arm_is_refused_on_a_subsampled_pool():
    """The refusal is the finding, expressed as code (C2-F-084)."""
    if not (SLICE_ROOT/'manifest.json').exists():
        pytest.skip("Benchmark B's slice is absent (untracked).")
    depth, subsampled = FomBenchmark.subsample_depth(SLICE_ROOT)
    assert subsampled and depth == 200, 'the slice must be subsampled for this test to mean anything'
    with pytest.raises(SystemExit, match='FLOORED'):
        zoo.main(['--pool', str(SLICE_ROOT), '--unfloored', '--reduce', '--tag', 'unused'])


def test_merit_columns_asks_the_sidecar_for_the_unfloored_inputs():
    """The unfloored M_sym is a product, so the arm needs both factors and the support count."""
    plain = zoo.merit_columns(['M_sym'])
    assert plain == ['M_sym']
    unfloored = zoo.merit_columns(['M_sym'], unfloored=True)
    for needed in ('M_tilde', 'M_rev_unfloored', 'N_cal'):
        assert needed in unfloored
    # M20 is a stored candidate column, never a sidecar one, so it must not be projected from there.
    assert zoo.merit_columns(['M20']) == []


@pytest.mark.slow
def test_the_reduction_carries_the_pools_own_depth_and_certifies_the_rank():
    """`subsample_top_k` reaches `evaluate` from the manifest, not from 'auto'."""
    if not (SLICE_ROOT/'merits').exists():
        pytest.skip('merit sidecars absent; write them with run_fom_floor_merits.py')
    entries = FomBenchmark.load_entries(SLICE_ROOT)
    dev = set(entries.loc[entries['split'] == 'fom-dev', 'entry_id'])
    _, _, meta = zoo.reduce_one(str(SLICE_ROOT), 'M_sym', dev, entries, 'fom-dev')
    assert meta['subsample_top_k'] == 200
    assert meta['subsampled'] is True
    assert meta['ranks_exact'] is True


@pytest.mark.slow
def test_a_missing_sidecar_is_loud_rather_than_null():
    """A left join that misses becomes NaN, and NaN ranks last: silently, the worst merit in the zoo."""
    if not (SLICE_ROOT/'merits').exists():
        pytest.skip('merit sidecars absent.')
    frames = zoo.pool_frames(str(SLICE_ROOT), ['M_sym'], None, merit_dir=SLICE_ROOT/'no_such_dir')
    with pytest.raises(FileNotFoundError, match='No merit sidecar'):
        next(iter(frames))
