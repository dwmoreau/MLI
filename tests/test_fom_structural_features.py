"""S12's structural feature block, and the two name collisions it exists to keep apart.

`FomBenchmark.structural_features` is a cheaper route to nine of `compute_all`'s columns -- 255
microseconds a candidate against `zoo_features`' 558, because it computes six merits instead of
twenty-odd rather than because it computes them differently. That claim is only worth anything if
the columns are the *same* columns, so the first test here compares the two routes value for value
and demands exact equality.

Two quantities carry one name each in this repository and mean different things, and both reach
S12's design matrix:

  * **`N_cal`.** The merit sidecar's is `get_M_rev_sym`'s support count, over [q_I, q_N]. The one
    `compute_all` emits is `get_N_cal(ref, 0, q_N)`, over [0, q_N]. On real mP candidates they
    agree on 0.07 % of rows. `structural_features` emits the second as `N_cal_full` so a feature
    set can carry both, and so that joining the sidecar and calling it `N_cal` is not silently a
    different feature from the one campaign 1's numbers describe.
  * **the absence-count window.** `run_fom_symmetry_arms` counts against the generic reference list
    using the *generic* list's cutoff; `extinction_group_sweep` counts against the generic list
    using each *group's* own cutoff. A group's list is a subset, so its assigned N-th line sits
    further out, its window is wider, and its count is greater by one on 0.3-1.6 % of real rows.
    S12 takes S04's convention, because that is the one under which `n_absent_extra_in_range`
    earned the +0.522 pp that put it in the feature set (C2-F-041).
"""

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomBenchmark

SLICE_ROOT = FomBenchmark.Path(__file__).parent.parent / 'mlindex' / 'data' / 'fom_benchmark_c2'
RETAINED_ROOT = FomBenchmark.Path(__file__).parent.parent / 'mlindex' / 'data' / 'fom_full_c2_pool'

SHARED_WITH_COMPUTE_ALL = ('zone_dominance', 'V_over_Vcrit', 'M_werner_max', 'delta_dewolff61',
                           'n_dewolff61', 'M_wu', 'M_1', 'F_N_q')


def _pool(path, entries_root, rows=1500):
    """A real candidate block and its entry table, or a skip if the pool is not on this machine."""
    import pyarrow.parquet as pq
    if not path.exists():
        pytest.skip(f'{path} is not on this machine')
    frame = pd.DataFrame(next(pq.ParquetFile(path).iter_batches(batch_size=rows)).to_pydict())
    entries = FomBenchmark.load_entries(entries_root)
    bundle = frame['condition_bundle'].iloc[0]
    return frame.reset_index(drop=True), entries.loc[entries['condition_bundle'] == bundle]


# ---------------------------------------------------------------------------------------
# The claim the cheaper route rests on
# ---------------------------------------------------------------------------------------
@pytest.mark.slow
def test_every_shared_column_is_bit_identical_to_compute_all():
    """The saving is merits not computed, not merits computed differently.

    `assert_array_equal`, not `allclose`. If these were merely close, `structural_features` would
    be a second implementation of `compute_all`'s columns and every campaign-1 number carrying one
    of these names would stop being comparable with a campaign-2 one.
    """
    frame, entries = _pool(RETAINED_ROOT/'candidates_c2_error1_cont0_mP.parquet', RETAINED_ROOT)
    mine = FomBenchmark.structural_features(frame, entries)
    theirs, _ = FomBenchmark.zoo_features(frame, entries)
    for name in SHARED_WITH_COMPUTE_ALL:
        np.testing.assert_array_equal(
            mine[name].to_numpy(), theirs[name].to_numpy(),
            err_msg=f'{name} differs between structural_features and compute_all')
    # And the renamed one, which is the same call under a different key.
    np.testing.assert_array_equal(mine['N_cal_full'].to_numpy(), theirs['N_cal'].to_numpy())


@pytest.mark.slow
def test_N_cal_full_is_not_the_merit_sidecars_N_cal():
    """The collision, on real data, so the rename is evidenced rather than argued.

    Both count reference lines below the same cutoff; they differ in where the window starts.
    A test asserting only that the rename exists would pass even if the two were the same number.
    """
    frame, entries = _pool(RETAINED_ROOT/'candidates_c2_error1_cont0_mP.parquet', RETAINED_ROOT)
    mine = FomBenchmark.structural_features(frame, entries, probation=False, absences=False)
    merits = FomBenchmark.reduced_merits(frame, entries)
    agree = (mine['N_cal_full'].to_numpy() == merits['N_cal'].to_numpy()).mean()
    assert agree < 0.05, (
        f'N_cal_full and the sidecar N_cal agree on {agree:.3f} of rows. If they have become the '
        'same quantity the rename is misleading; if they have not, this threshold is wrong.')
    # The M_rev window starts at q_I rather than at zero, so it can only ever hold fewer lines.
    assert (merits['N_cal'].to_numpy() <= mine['N_cal_full'].to_numpy()).all()


# ---------------------------------------------------------------------------------------
# The absence counts, and the window they are counted in
# ---------------------------------------------------------------------------------------
@pytest.mark.slow
def test_the_absence_count_differs_from_the_sweep_only_by_the_cutoff_it_uses():
    """The two conventions differ, and the difference is exactly which cutoff bounds the window.

    `run_fom_symmetry_arms` -- and so this -- counts absent lines below the GENERIC list's own
    cutoff. `extinction_group_sweep` counts them below each GROUP's cutoff. The invariant worth
    pinning is not the size of the disagreement but its cause, so this recomputes with the group's
    cutoff and demands the sweep back exactly. If that ever stopped holding, one of the two routes
    would have changed what it counts rather than merely where it stops.

    Note the disagreement is **two-sided**, by up to nine lines on real mC candidates: a group's
    list is a subset, but `fast_assign` matches the last observed peak to the nearest line in
    whichever list it is given, and that line can sit either side of the generic list's. An earlier
    version of this test asserted a one-sided gap on a 3 000-row sample and was simply wrong.
    """
    sweep = RETAINED_ROOT/'extinction_sweep'/'candidates_c2_error1_cont0_oP.parquet'
    if not sweep.exists():
        pytest.skip('the extinction sweep is not on this machine')
    frame, entries = _pool(RETAINED_ROOT/'candidates_c2_error1_cont0_oP.parquet', RETAINED_ROOT)
    keys = list(FomBenchmark.ZOO_KEY_COLUMNS)
    mine = pd.concat([frame[keys], FomBenchmark.structural_features(
        frame, entries, probation=False)[['n_absent_extra_in_range']]], axis=1)
    theirs = pd.read_parquet(sweep, columns=keys + ['xg_M20_n_absent_in_range'])
    joined = mine.merge(theirs, on=keys, how='inner', validate='1:1')
    assert joined.shape[0] > 100

    # Recomputed with the sweep's own window, which must reproduce it exactly.
    from mlindex.utilities.ExtinctionCounts import absent_in_range, build_group_masks, \
        get_generic_group
    from mlindex.utilities.Q2Calculator import Q2Calculator
    lattice_system = frame['lattice_system'].iloc[0]
    hkl_ref = FomBenchmark.hkl_ref_for(lattice_system, 'oP', get_generic_group('oP'))
    masks = build_group_masks(hkl_ref, 'oP')
    calculator = Q2Calculator(lattice_system=lattice_system, hkl=hkl_ref, tensorflow=False,
                              representation='xnn')
    peaks = entries.set_index('entry_id')['q2_obs']
    group_window = np.full(frame.shape[0], -1, dtype=np.int64)
    for entry_id, block in frame.groupby('entry_id', sort=False):
        q2_obs = np.asarray(peaks.loc[entry_id],
                            dtype=np.float64)[:int(block['n_peaks'].iloc[0])]
        reference = calculator.get_q2(
            np.vstack([np.asarray(v, dtype=np.float64) for v in block['xnn']]))
        for spacegroup, rows in block.groupby('spacegroup', sort=False):
            local = np.flatnonzero(block.index.isin(rows.index))
            _, _, _, q2_calc = FomBenchmark.assign_lines(
                q2_obs, np.vstack([np.asarray(v, dtype=np.float64) for v in rows['xnn']]),
                lattice_system, 'oP', spacegroup)
            group_window[block.index[local]] = absent_in_range(
                reference[local], masks[spacegroup], q2_calc[:, -1])[0]
    np.testing.assert_array_equal(
        group_window, theirs.set_index(keys).loc[
            list(frame[keys].itertuples(index=False, name=None))
            ]['xg_M20_n_absent_in_range'].to_numpy(),
        err_msg='the sweep is no longer counting below its own group cutoff')

    # And the two windows really are close, so neither is a different quantity.
    agree = (joined['n_absent_extra_in_range'].to_numpy()
             == joined['xg_M20_n_absent_in_range'].to_numpy())
    assert agree.mean() > 0.9


@pytest.mark.slow
def test_triclinic_has_no_extra_absences_and_still_has_a_denominator():
    """aP has one extinction group, so it removes nothing -- S04's built-in negative control.

    The denominator must still be populated, or `f_absent_extra` is NaN for a whole lattice rather
    than the zero it is. That is the shape of failure campaign 1 shipped as a column of nulls.
    """
    frame, entries = _pool(RETAINED_ROOT/'candidates_c2_error1_cont0_aP.parquet', RETAINED_ROOT,
                           rows=400)
    features = FomBenchmark.structural_features(frame, entries, probation=False)
    assert (features['n_absent_extra_in_range'] == 0).all()
    assert (features['n_ref_in_range'] > 0).all()


# ---------------------------------------------------------------------------------------
# The contract a sidecar producer depends on
# ---------------------------------------------------------------------------------------
@pytest.mark.slow
def test_chunking_changes_nothing_including_across_a_group_boundary():
    """The producer streams row groups, so exactness on an arbitrary subset is a hard requirement.

    Split at a row that is *inside* one (entry, lattice, extinction group), not between two, since
    a split between groups would pass even if the reference list were being carried across chunks.
    """
    frame, entries = _pool(RETAINED_ROOT/'candidates_c2_error1_cont0_mP.parquet', RETAINED_ROOT,
                           rows=900)
    whole = FomBenchmark.structural_features(frame, entries)
    groups = frame['spacegroup'].to_numpy()
    interior = [row for row in range(1, frame.shape[0]) if groups[row - 1] == groups[row]]
    assert interior, 'every row starts a new extinction group; this frame cannot test chunking'
    # The one nearest the middle, so both halves are substantial.
    cut = min(interior, key=lambda row: abs(row - frame.shape[0]//2))
    halves = pd.concat([
        FomBenchmark.structural_features(frame.iloc[:cut].reset_index(drop=True), entries),
        FomBenchmark.structural_features(frame.iloc[cut:].reset_index(drop=True), entries),
        ], ignore_index=True)
    for name in whole.columns:
        np.testing.assert_array_equal(whole[name].to_numpy(), halves[name].to_numpy(),
                                      err_msg=f'{name} depends on how the frame was chunked')


def test_the_option_flags_select_exactly_their_column_blocks():
    """Each flag removes its own block and nothing else, so a sidecar's columns are readable from
    its `_meta.json` rather than from the data."""
    from mlindex.scripts import run_fom_structural_features as producer
    structural = FomBenchmark.STRUCTURAL_COLUMNS
    probation = FomBenchmark.PROBATION_MERIT_COLUMNS
    dropped = FomBenchmark.DROPPED_MERIT_COLUMNS
    absences = FomBenchmark.ABSENCE_COLUMNS
    assert producer.feature_columns(True, True, True) == (structural + probation + dropped
                                                          + absences)
    assert producer.feature_columns(False, True, True) == structural + dropped + absences
    assert producer.feature_columns(True, False, True) == structural + probation + dropped
    assert producer.feature_columns(True, True, False) == structural + probation + absences
    assert producer.feature_columns(False, False, False) == structural


def test_the_cut_merits_are_emitted_because_the_arm_that_licenses_the_cut_needs_them():
    """S00's audit cut ten merits from the zoo on per-entry outcomes. PROTOCOL section 8 says a
    cut is settled by a retrained paired arm, so the arm that RESTORES them has to be buildable --
    and it needs the columns. They cost 1 microsecond a candidate against the 193 the
    reference-line pass costs, so omitting them saves nothing and costs a second pass."""
    assert set(FomBenchmark.DROPPED_MERIT_COLUMNS) == {
        'M_star', 'M_star_corrected', 'M_info_clipped', 'null_tail_nll', 'nll_exponential',
        'M_werner_frac'}
    from mlindex.model_training import FomCombiner
    # Every one of them is in campaign 1's raw group and in none of campaign 2's.
    for name in FomBenchmark.DROPPED_MERIT_COLUMNS:
        assert name in FomCombiner.CAMPAIGN1_RAW_MERITS
        assert name not in FomCombiner.RAW_MERITS
        assert name in FomCombiner.EXCLUDED_MERITS, f'{name} is cut with no reason recorded'


def test_the_absence_columns_are_integers_and_the_ratio_is_not_stored():
    """`f_absent_extra` is derived where it is consumed. A stored ratio is a third column that can
    disagree with its own numerator, which is what `SCHEMA.md`'s recomputable rule is about."""
    assert FomBenchmark.ABSENCE_COLUMNS == ('n_absent_extra_in_range', 'n_ref_in_range')
    assert 'f_absent_extra' not in FomBenchmark.ABSENCE_COLUMNS
    assert 'f_absent_extra' not in FomBenchmark.STRUCTURAL_COLUMNS


def test_the_probation_merits_are_the_three_S00_left_undecided():
    """S09 dropped them because their rank on the subsampled pool is optimistic (C2-F-077), and
    recorded the drop as reversible on a fully retained pool. These are that reversal's inputs."""
    assert FomBenchmark.PROBATION_MERIT_COLUMNS == ('M_wu', 'M_1', 'F_N_q')
    for name in FomBenchmark.PROBATION_MERIT_COLUMNS:
        assert name not in FomBenchmark.REDUCED_MERIT_COLUMNS
