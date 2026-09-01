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


# ---------------------------------------------------------------------------------------
# The mechanism analyses
# ---------------------------------------------------------------------------------------
def test_bravais_lattices_runs_high_symmetry_to_low():
    """The direction the whole symmetry-lowering claim rests on.

    `symmetry_lowering` reads "is the wrong winner of LOWER symmetry?" as "is its index in
    `BRAVAIS_LATTICES` HIGHER?". If that ordering were reversed the headline mechanism claim would
    invert -- and it would still produce a plausible-looking number, so nothing else would catch
    it. Pinned against the physics rather than against the current tuple order.
    """
    order = list(FomMetrics.BRAVAIS_LATTICES)
    free_parameters = {'cP': 1, 'cI': 1, 'cF': 1, 'tP': 2, 'tI': 2, 'hP': 2, 'hR': 2,
                       'oP': 3, 'oC': 3, 'oI': 3, 'oF': 3, 'mP': 4, 'mC': 4, 'aP': 6}
    assert set(order) == set(free_parameters), 'the tuple and the physics must cover the same set'
    counts = [free_parameters[lattice] for lattice in order]
    # Non-decreasing: more free cell parameters means less symmetry, and it comes later.
    assert counts == sorted(counts), f'BRAVAIS_LATTICES is not ordered high symmetry to low: {order}'
    assert order[0].startswith('c') and order[-1] == 'aP'


def test_the_floor_arm_refuses_a_subsampled_pool():
    """C2-F-084, expressed where the comparison would otherwise be run."""
    from mlindex.scripts import run_fom_zoo_explain as explain
    artifact_dir = FomBenchmark.Path(__file__).parent.parent/'docs'/'fom_campaign2'/'artifacts'
    if not (artifact_dir/'S09_zoo_slice_reduced_meta.json').exists():
        pytest.skip('no slice reductions on disk')
    with pytest.raises(ValueError, match='SUBSAMPLED'):
        explain.floor_arm(artifact_dir, 'S09_zoo_slice')


def test_the_counting_arm_refuses_a_subsampled_pool_and_sizes_the_oracle_correctly(tmp_path):
    """C2-Q-025's analysis: the refusal, and the union-oracle arithmetic it reports.

    The oracle question is the one that actually bears on S12 -- what a merit is worth *in
    combination* -- so its arithmetic is pinned rather than eyeballed. Built so a naive
    implementation that unions the wrong column set gets a different answer.
    """
    import json
    import numpy as np
    import pandas as pd
    from mlindex.scripts import run_fom_zoo_explain as explain

    index = pd.MultiIndex.from_product([[f'E{i}' for i in range(10)], ['c2_error1_cont0']],
                                       names=['entry_id', 'condition_bundle'])
    # M20 gets the first five; the hard count adds one more; the soft count adds two more.
    truth = {'M20':          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             'X_N':          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             'X_N_soft':     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]}
    metas = {}
    for merit, flags in truth.items():
        frame = pd.DataFrame({
            'entry_id': index.get_level_values(0), 'condition_bundle': 'c2_error1_cont0',
            'cluster': np.arange(10), 'is_hard': False, 'bravais_lattice': 'oP',
            'volume_decile': 5, 'split': 'fom-dev',
            'has_correct_all': True, 'n_correct_all': 1,
            'rank_best_correct_all': np.where(np.array(flags) == 1, 1, 999),
            'score_best_correct_all': 1.0, 'score_top_all': 1.0,
            'top_is_correct_all': np.array(flags, dtype=bool),
            'n_candidates_all': 100, 'n_ties_at_best_correct_all': 1,
            'n_degenerate_all': 0, 'n_off_by_two_all': 0, 'n_non_finite_score_all': 0,
            'bravais_lattice_top_all': 'oP', 'bravais_lattice_best_correct_all': 'oP',
            })
        # `reduce_pool` emits each quantity in four forms: {all, in_top_n} x
        # {excluding, including degenerates}. `derive_flags` reads whichever the caller asked for,
        # so a fixture missing any of them fails on a KeyError rather than on its arithmetic.
        for column in list(frame.columns):
            if column.endswith('_all'):
                stem = column[:-len('_all')]
                frame[f'{stem}_in_top_n'] = frame[column]
                frame[f'{stem}_incl_degenerate_all'] = frame[column]
                frame[f'{stem}_incl_degenerate_in_top_n'] = frame[column]
        frame.to_parquet(tmp_path/f'T_reduced_{merit}_fom-dev.parquet', index=False)
        metas[f'{merit}|fom-dev'] = dict(score=merit, higher_is_better=True, pool='cross_bl',
                                         reduced_top_n=10, split='fom-dev', subsampled=False,
                                         subsample_top_k=None, ranks_exact=True,
                                         rank_exactness=None, source='test',
                                         hard_min_decile=8, bundles_excluded=[])
    (tmp_path/'T_reduced_meta.json').write_text(json.dumps(metas), encoding='utf-8')

    arm, oracle = explain.counting_arm(tmp_path, 'T', reference=('M20',))
    got = dict(zip(oracle['set'], oracle['union_oracle_top10']))
    assert got['reference only'] == pytest.approx(0.5)
    assert got['reference + hard counts'] == pytest.approx(0.6)
    assert got['reference + soft counts'] == pytest.approx(0.7)
    assert got['everything'] == pytest.approx(0.8)

    # And it must refuse a subsampled pool outright -- C2-R-013.
    metas['M20|fom-dev']['subsampled'] = True
    (tmp_path/'T_reduced_meta.json').write_text(json.dumps(metas), encoding='utf-8')
    with pytest.raises(ValueError, match='SUBSAMPLED'):
        explain.counting_arm(tmp_path, 'T', reference=('M20',))
