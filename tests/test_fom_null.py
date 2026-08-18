"""S07's nulls: the closed-form one must be the published one, and no transform may reorder a lattice.

Three classes of check, in rough order of how much damage the bug they catch would do.

**The analytic null is a claim about a published distribution**, so it is tested against de Wolff
1961's own statement of it and against de Wolff 1968's Table 2 rather than against itself. The
Table 2 test also pins the *mechanism* behind S07's headline negative result: with the line count
fixed, the analytic tail probability is a strictly monotone function of M20, which is why it cannot
reorder two candidates of the same N however far apart their lattices are.

**A standardisation that reorders candidates within a lattice is broken**, whatever it does to the
headline. `rank` and `z` condition on the Bravais lattice alone, so they are monotone within one by
construction and the transformed order must be the raw order exactly -- no inversions and no ties
that were not already there. `binned` conditions on volume and V/V_crit *inside* a lattice and is
deliberately not invariant; the difference between the two is asserted rather than assumed.

**A fitted null that cannot be reloaded is not a deliverable.** The npz round trip is exact, and it
carries no pickle, so a null outlives the pandas and sklearn versions that produced it.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import fixtures_fom_literature as fixtures

from mlindex.model_training import FomMetrics
from mlindex.model_training import FomNull


# ---------------------------------------------------------------------------------------
# A synthetic pool. Built by hand rather than from the models, because everything under test
# is a transform of a merit column and a few covariates -- no diffraction geometry is involved.
# ---------------------------------------------------------------------------------------
LATTICES = ('cF', 'mP', 'aP')


@pytest.fixture(scope='module')
def null_pool():
    """One shard: three lattices whose merit distributions differ in location and in scale.

    The scales are deliberately far apart -- that is the situation S07 exists for -- and each
    lattice carries a handful of correct candidates drawn from well above its own null, so the
    labels mean what they mean downstream.
    """
    rng = np.random.default_rng(4)
    rows = []
    for lattice, centre, spread, count in (('cF', 4.0, 1.0, 4000), ('mP', 9.0, 3.0, 12000),
                                           ('aP', 15.0, 6.0, 20000)):
        merit = rng.gamma(shape=3.0, scale=spread, size=count) + centre
        n_entries = count//200
        rows.append(pd.DataFrame({
            'entry_id': [f'E{index % n_entries:04d}' for index in range(count)],
            'condition_bundle': 'error1_cont0',
            'bravais_lattice': lattice,
            'lattice_system': {'cF': 'cubic', 'mP': 'monoclinic', 'aP': 'triclinic'}[lattice],
            'candidate_id': np.arange(count),
            'M20': merit,
            'null_tail_nll': merit*2.0,
            'n_peaks': 10 if lattice == 'cF' else 20,
            'spacegroup': rng.choice([f'{lattice} sg{index}' for index in range(3)], size=count),
            'volume': np.exp(rng.normal(6.0, 0.8, size=count)),
            'V_over_Vcrit': rng.gamma(2.0, 0.5, size=count),
            'n_entering': rng.integers(50, 5000, size=count),
            'in_top_n': True,
            'is_correct': False,
            'is_off_by_two': False,
            'is_degenerate': pd.Series([pd.NA]*count, dtype='object'),
            }))
    frame = pd.concat(rows, ignore_index=True)
    # One correct candidate per (entry, lattice), scoring above its own lattice's null.
    correct = frame.groupby(['entry_id', 'bravais_lattice'], sort=False).head(1).index
    frame.loc[correct, 'is_correct'] = True
    frame.loc[correct, 'M20'] = frame.loc[correct, 'M20']*3.0
    frame.loc[correct, 'null_tail_nll'] = frame.loc[correct, 'M20']*2.0
    return frame.reset_index(drop=True)


@pytest.fixture(scope='module')
def null_entries(null_pool):
    ids = sorted(null_pool['entry_id'].unique())
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        'entry_id': ids,
        'condition_bundle': 'error1_cont0',
        'split': ['fom-train' if index % 2 else 'fom-dev' for index in range(len(ids))],
        'bravais_lattice_true': rng.choice(LATTICES, size=len(ids)),
        'lattice_system_true': 'monoclinic',
        'volume_true': rng.uniform(100.0, 3000.0, size=len(ids)),
        })


# ---------------------------------------------------------------------------------------
# The analytic null
# ---------------------------------------------------------------------------------------
def test_log_gamma_sf_matches_scipy_and_survives_underflow():
    """scipy is the reference where it is exact, and the point of the helper is where it is not."""
    x = np.array([0.5, 5.0, 20.0, 39.0, 60.0, 200.0])
    reference = np.log(special.gammaincc(20.0, x))
    assert FomNull.log_gamma_sf(20.0, x) == pytest.approx(reference, rel=1e-8, abs=1e-9)

    # `null_tail_nll` reaches ~1400 against N = 20 (F-064), where gammaincc underflows to zero and
    # every candidate in the upper tail -- which is every candidate a ranking cares about --
    # would collapse to one infinite score.
    assert special.gammaincc(20.0, 1400.0) == 0.0
    deep = FomNull.log_gamma_sf(20.0, np.array([570.0, 1400.0]))
    assert np.all(np.isfinite(deep))
    assert deep[1] < deep[0] < 0


def test_exponential_null_gives_a_unit_mean_line_term():
    """de Wolff's null, taken literally: each per-line term is Exp(1), so the moments are (1, 1).

    If dQ/Delta is Exp(1) then exp(-dQ/Delta) is uniform, so 1 - exp(-dQ/Delta) is uniform and
    -log of it is Exp(1). Everything `analytic` does rests on this one line.
    """
    mean, variance = FomNull._line_term_moments([1.0])
    assert mean[0] == pytest.approx(1.0, rel=1e-6)
    assert variance[0] == pytest.approx(1.0, rel=1e-5)

    # de Wolff's equidistant limit, g_bar = Delta/2, has a closed form: sum_k 1/(k(1 + k/2)) = 3/2.
    assert FomNull._line_term_moments([0.5])[0][0] == pytest.approx(1.5, rel=1e-6)


def test_analytic_tail_probability_is_uniform_under_the_null():
    """Simulate the null de Wolff states and check the calibration comes back uniform.

    This is the whole claim of the `analytic` method in one test: if the per-line discrepancies
    really are exponential with mean Delta(Q), the reported tail probability is uniform on (0, 1)
    and therefore means the same thing in every lattice.
    """
    rng = np.random.default_rng(11)
    n_peaks = 20
    statistic = rng.gamma(shape=n_peaks, scale=1.0, size=20000)
    p = np.exp(-FomNull.analytic_neg_log_p('null_tail_nll', statistic,
                                           np.full(statistic.shape, n_peaks)))
    assert stats.kstest(p, 'uniform').pvalue > 0.01


def test_rho_inverts_the_mean_line_term():
    grid = np.array([0.002, 0.05, 0.5, 1.0, 2.0])
    mean, _ = FomNull._line_term_moments(grid)
    assert FomNull.rho_from_mean_term(mean) == pytest.approx(grid, rel=1e-2)


def test_analytic_null_reproduces_dewolff68_table2_and_inherits_its_failure():
    """The analytic tail probability on fourteen published indexings, and the case it cannot solve.

    de Wolff 1968's M20 is Q20/(2 eps_bar N20), so his tabulated mean discrepancy in units of the
    mean interval is exactly 1/M20 -- which makes the analytic statistic at fixed N a strictly
    monotone function of M20 and nothing else. That is not a coincidence to be checked once and
    forgotten: it is the mechanism behind S07's headline result, because a transform that is
    monotone in M20 at fixed N cannot reorder two candidates that share a line count, however far
    apart their lattices are.

    So the analytic null inherits Li6B4O9 (rows 8 and 9), where the *incorrect* indexing scores
    5.4 against the correct one's 5.3 and de Wolff notes there is "not the remotest analogy" between
    the two lattices. It is pinned here because a future method that claims to fix the cross-lattice
    problem must be checked against this row, not against the aggregate.
    """
    rows = fixtures.DEWOLFF68_TABLE2
    m20 = np.array([row[6] for row in rows], dtype=float)
    # 20 lines by construction: M20 is defined on the first twenty observed.
    statistic = -20.0*np.log1p(-np.exp(-1.0/m20))
    neg_log_p = FomNull.analytic_neg_log_p('null_tail_nll', statistic, np.full(m20.shape, 20))

    order = np.argsort(-m20, kind='stable')
    assert np.all(np.diff(neg_log_p[order]) <= 1e-9), 'analytic p must be monotone in M20 at fixed N'

    correct = np.array([row[7] == 'correct' for row in rows])
    incorrect = np.array([row[7] == 'incorrect' for row in rows])
    # It separates the classes on the whole, as M20 does...
    assert neg_log_p[correct].mean() > neg_log_p[incorrect].mean()
    # ...and fails on exactly the pair M20 fails on.
    li6b4o9_correct = next(index for index, row in enumerate(rows)
                           if row[0] == 8 and row[7] == 'correct')
    li6b4o9_incorrect = next(index for index, row in enumerate(rows)
                             if row[0] == 9 and row[7] == 'incorrect')
    assert neg_log_p[li6b4o9_incorrect] > neg_log_p[li6b4o9_correct]


def test_look_elsewhere_reduces_to_subtracting_log_of_the_trial_count():
    """de Wolff 1961 section 5 as arithmetic, and its two limits."""
    neg_log_p = np.array([10.0, 40.0, 200.0])
    corrected = FomNull.look_elsewhere(neg_log_p, np.full(3, 1000.0))
    assert corrected == pytest.approx(neg_log_p - np.log(1000.0), rel=1e-6)

    # One trial is no correction at all.
    assert FomNull.look_elsewhere(neg_log_p, np.ones(3)) == pytest.approx(neg_log_p, rel=1e-9)

    # Where m*p approaches one the linear form would go negative; the exact expression saturates
    # at "certain", i.e. -log p -> 0.
    saturated = FomNull.look_elsewhere(np.array([1.0]), np.array([10000.0]))
    assert 0.0 <= saturated[0] < 1e-6


# ---------------------------------------------------------------------------------------
# The transforms
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['rank', 'z'])
def test_lattice_conditioned_transforms_do_not_reorder_within_a_lattice(null_pool, method):
    """The invariant the whole ladder rests on: conditioning on the lattice cannot reorder inside it.

    Not merely "no inversions": no *ties* either. A quantile grid alone would quantise the upper
    tail, where every candidate worth reporting sits, and silently merge the top of the pool into
    one score -- which changes a ranking without ever inverting a pair. That is what the
    peaks-over-threshold fit in `_store_table` exists to prevent, and this is the test that would
    have caught its absence.
    """
    null = FomNull.FomNull.fit([null_pool], 'M20', method)
    scores = null.apply(null_pool)
    for lattice in LATTICES:
        rows = (null_pool['bravais_lattice'] == lattice).to_numpy()
        order = np.argsort(null_pool.loc[rows, 'M20'].to_numpy(), kind='stable')
        transformed = scores[rows][order]
        assert np.all(np.diff(transformed) > 0), f'{method} reordered or tied within {lattice}'


def test_binned_is_deliberately_not_within_lattice_invariant(null_pool):
    """Conditioning on volume *inside* a lattice is a different claim, and it must show up as one."""
    null = FomNull.FomNull.fit([null_pool], 'M20', 'binned', min_count=200)
    scores = null.apply(null_pool)
    rows = (null_pool['bravais_lattice'] == 'aP').to_numpy()
    order = np.argsort(null_pool.loc[rows, 'M20'].to_numpy(), kind='stable')
    assert np.any(np.diff(scores[rows][order]) < 0)


def test_the_null_is_fitted_on_incorrect_candidates_only(null_pool):
    """A null contaminated by the correct candidates is not a null."""
    null = FomNull.FomNull.fit([null_pool], 'M20', 'rank')
    n_incorrect = int((~null_pool['is_correct']).sum())
    assert null.meta['n_null'] == n_incorrect
    # The per-lattice tables partition the null; the root table covers all of it again, and is
    # stored so an unseen group has somewhere to land.
    assert sum(null.counts[lattice] for lattice in LATTICES) == n_incorrect
    assert null.counts['all'] == n_incorrect


def test_thin_groups_pool_up_the_hierarchy_and_say_so(null_pool):
    """A cell too thin for its own quantiles borrows its parent's, and the borrowing is recorded."""
    null = FomNull.FomNull.fit([null_pool], 'M20', 'binned', min_count=10**9)
    # Nothing can meet that count, so every finest cell must fall all the way to the root. The
    # per-lattice tables are still stored -- unconditionally, as the landing spot for an unseen
    # cell -- but nothing *resolves* to them.
    assert {entry['level'] for entry in null.pooled_from.values()} == {0}
    assert all(entry['source'] == 'all' for entry in null.pooled_from.values())
    assert set(null.tables) == {'all', *LATTICES}

    generous = FomNull.FomNull.fit([null_pool], 'M20', 'binned', min_count=1)
    assert len(generous.tables) > len(null.tables)
    assert {entry['level'] for entry in generous.pooled_from.values()} == {4}


def test_an_unseen_group_falls_back_rather_than_scoring_zero(null_pool):
    """A spacegroup `fom-train` never produced must not silently rank last."""
    null = FomNull.FomNull.fit([null_pool], 'M20', 'binned', min_count=200)
    unseen = null_pool.copy()
    unseen['spacegroup'] = 'a diffraction symbol never seen in training'
    scores = null.apply(unseen)
    assert np.all(np.isfinite(scores))
    assert null.diagnostics['n_unseen_group'] == 0
    assert null.diagnostics['n_group_fallback'] > 0


def test_non_finite_merits_stay_non_finite(null_pool):
    """A NaN merit carries no information and must not be mapped onto a finite score."""
    null = FomNull.FomNull.fit([null_pool], 'M20', 'rank')
    broken = null_pool.copy()
    broken.loc[broken.index[:5], 'M20'] = np.nan
    scores = null.apply(broken)
    assert np.all(np.isnan(scores[:5]))
    assert np.all(np.isfinite(scores[5:]))


# ---------------------------------------------------------------------------------------
# Persistence and the metrics contract
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['analytic', 'analytic_rho', 'z', 'rank', 'binned'])
def test_npz_round_trip_is_exact(null_pool, tmp_path, method):
    merit = 'null_tail_nll' if method in ('analytic', 'analytic_rho') else 'M20'
    null = FomNull.FomNull.fit([null_pool], merit, method, min_count=200)
    path = null.save(tmp_path/f'{merit}_{method}.npz')
    reloaded = FomNull.FomNull.load(path)

    assert reloaded.merit == null.merit and reloaded.method == null.method
    assert reloaded.required_columns == null.required_columns
    before = null.apply(null_pool)
    after = reloaded.apply(null_pool)
    assert np.array_equal(before, after, equal_nan=True)


def test_saved_nulls_contain_no_pickled_objects(null_pool, tmp_path):
    """Serialisation is arrays plus one JSON string, so loading executes nothing."""
    null = FomNull.FomNull.fit([null_pool], 'M20', 'rank')
    path = null.save(tmp_path/'M20_rank.npz')
    with np.load(path, allow_pickle=False) as data:
        # `labels` and `metadata` are the only object arrays and both are plain strings; everything
        # else must load with pickling refused outright.
        for name in data.files:
            if name in ('labels', 'metadata'):
                continue
            assert np.asarray(data[name]).dtype != object


def test_the_trial_correction_is_refused_on_a_score_that_is_not_a_probability(null_pool):
    """`z` returns a standardised merit, not a tail probability, so `1 - (1 - p)^m` is meaningless."""
    with pytest.raises(ValueError, match='tail probability'):
        FomNull.FomNull.fit([null_pool], 'M20', 'z', trials='n_entering')


def test_the_trial_correction_survives_the_round_trip(null_pool, tmp_path):
    null = FomNull.FomNull.fit([null_pool], 'M20', 'rank', trials='n_entering')
    assert 'n_entering' in null.required_columns
    reloaded = FomNull.FomNull.load(null.save(tmp_path/'M20_rank_trials.npz'))
    assert reloaded.trials == 'n_entering'
    assert np.array_equal(reloaded.apply(null_pool), null.apply(null_pool), equal_nan=True)


def test_apply_is_usable_as_an_evaluate_score(null_pool, null_entries):
    """The deliverable's contract: a fitted null is a `FomMetrics.evaluate` callable score."""
    null = FomNull.FomNull.fit([null_pool], 'M20', 'rank')
    result = FomMetrics.evaluate(
        [null_pool], score=null.apply, higher_is_better=True, score_columns=null.required_columns,
        entries=null_entries, strata=(), weights=None, n_bootstrap=0,
        )
    assert result.meta['n_candidates_seen'] == null_pool.shape[0]
    assert 0.0 <= result.metric('top10') <= 1.0


def test_scaled_features_puts_merits_on_one_scale_for_a_joint_predictor(null_pool, tmp_path):
    """S07's actual deliverable: comparable columns for S08, not a ranking.

    The property that matters to a combiner is that the same numeric value means the same thing in
    every lattice. Raw, the three lattices' merits sit on wildly different scales by construction of
    the fixture; scaled, their medians must collapse together.
    """
    nulls = [FomNull.FomNull.fit([null_pool], merit, 'z')
             for merit in ('M20', 'null_tail_nll')]
    scaled = FomNull.scaled_features(null_pool, nulls)
    assert 'M20__z' in scaled.columns and 'null_tail_nll__z' in scaled.columns
    assert scaled.shape[0] == null_pool.shape[0]
    # Nothing is dropped or reordered: this is a column append on the caller's own frame.
    assert list(null_pool.columns) == list(scaled.columns)[:null_pool.shape[1]]

    wrong = scaled.loc[~scaled['is_correct']]
    raw_spread = wrong.groupby('bravais_lattice')['M20'].median()
    scaled_spread = wrong.groupby('bravais_lattice')['M20__z'].median()
    assert scaled_spread.max() - scaled_spread.min() < raw_spread.max() - raw_spread.min()


def test_load_scalers_round_trips_a_directory(null_pool, tmp_path):
    for merit, method in (('M20', 'z'), ('M20', 'rank'), ('null_tail_nll', 'analytic')):
        FomNull.FomNull.fit([null_pool], merit, method).save(tmp_path/f'{merit}_{method}.npz')
    loaded = FomNull.load_scalers(tmp_path, methods=('z', 'analytic'))
    assert {(null.merit, null.label) for null in loaded} == {('M20', 'z'),
                                                             ('null_tail_nll', 'analytic')}
    assert FomNull.scaled_features(null_pool, loaded).shape[1] == null_pool.shape[1] + 2


def test_regularity_ratio_reports_one_bounded_number_per_lattice(null_pool):
    table = FomNull.regularity_ratio([null_pool], merit='null_tail_nll', min_count=100)
    assert set(table['bravais_lattice']) == set(LATTICES)
    assert np.all(table['rho'] >= FomNull.RHO_BOUNDS[0])
    assert np.all(table['rho'] <= FomNull.RHO_BOUNDS[1])
    # rho falls as the merit rises: a bigger per-line term means smaller discrepancies relative to
    # Delta, which is what "the wrong cells fit better than chance" looks like.
    assert table.sort_values('mean_line_term')['rho'].is_monotonic_decreasing


def test_homogeneity_reports_the_spread_the_gate_asks_about(null_pool):
    null = FomNull.FomNull.fit([null_pool], 'M20', 'rank')
    raw = FomNull.null_homogeneity([null_pool], 'M20')
    calibrated = FomNull.null_homogeneity([null_pool], null.apply)
    # The whole point of the transform: the null's median is wildly lattice-dependent before and
    # essentially identical after.
    assert calibrated['q0.5_spread'].iloc[0] < raw['q0.5_spread'].iloc[0]
