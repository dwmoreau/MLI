"""S08 -- the reproducibility floor's arithmetic, and the two refusals that protect it.

The floor is what every gate in this campaign is read against, so an error here does not produce a
wrong number in one place -- it silently rescales every verdict from S09 onward. Campaign 1 read
thirteen days of gates against a floor six times too large and refused a real result on it.

Two things are checked here that campaign 1's version could not have been:

* **The aggregate is composed from the reporting split's per-lattice counts**, because this
  campaign's floor sample is balanced across lattices rather than proportional to the split.
  Campaign 1's was proportional, so it could take a plain mean; S08 acceptance condition 4 says a
  sample drawn any other way needs the split's own counts instead.
* **Arms whose peak lists differ are refused.** Only the search seed may move between arms. If the
  base seed moved too, the arms differ in their data and their spread is generation noise and
  scoring noise together -- the one distinction the floor exists to make.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'mlindex', 'scripts'))

import run_fom_floor_entries as floor_entries
import run_fom_floor_report as floor_report


def _manifest(counts, seed=0):
    """A split manifest with `counts` entries per lattice, all in fom-dev."""
    rng = np.random.default_rng(seed)
    rows = []
    for lattice, count in counts.items():
        for index in range(count):
            rows.append(dict(identifier=f'{lattice}{index:04d}', bravais_lattice=lattice,
                             split='fom-dev', volume_decile=int(rng.integers(0, 10)),
                             volume_true=float(rng.uniform(100, 900))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Drawing the sample
# ----------------------------------------------------------------------------------------
def test_the_draw_is_balanced_and_caps_at_availability():
    """Every lattice gets the quota except those the source population cannot supply.

    cF has 20 entries in existence and cI 30 (C2-R-010), so a quota of 40 is unreachable for them
    and reachable for everything else. That is a cap, not a shortfall, and the composition table
    has to say which.
    """
    manifest = _manifest({'aP': 600, 'mP': 600, 'cF': 20, 'cI': 30, 'oP': 280})
    sample = floor_entries.draw_sample(manifest, 40, 'fom-dev', seed=1)
    counts = sample.groupby('bravais_lattice').size().to_dict()
    assert counts == {'aP': 40, 'mP': 40, 'cF': 20, 'cI': 30, 'oP': 40}


def test_the_composition_records_the_splits_counts_not_only_the_samples():
    """`floor_weight` must be the split's share. The whole point of writing this table is that the
    report cannot reach for the sample's shape by accident."""
    manifest = _manifest({'aP': 600, 'cF': 20})
    sample = floor_entries.draw_sample(manifest, 40, 'fom-dev', seed=1)
    table = floor_entries.composition(manifest, sample, 'fom-dev').set_index('bravais_lattice')
    assert table.loc['aP', 'split_entries'] == 600
    assert table.loc['cF', 'split_entries'] == 20
    assert table.loc['aP', 'floor_weight'] == pytest.approx(600/620)
    assert table.loc['cF', 'floor_weight'] == pytest.approx(20/620)
    # And the sample's own share is nothing like it -- which is the reason the column exists.
    assert table.loc['cF', 'sample_share'] == pytest.approx(20/60)


def test_the_draw_is_deterministic_given_the_seed():
    manifest = _manifest({'aP': 100, 'oP': 100})
    first = floor_entries.draw_sample(manifest, 20, 'fom-dev', seed=5)
    second = floor_entries.draw_sample(manifest, 20, 'fom-dev', seed=5)
    pd.testing.assert_frame_equal(first, second)
    assert not first['identifier'].equals(
        floor_entries.draw_sample(manifest, 20, 'fom-dev', seed=6)['identifier'])


# ----------------------------------------------------------------------------------------
# The floor arithmetic
# ----------------------------------------------------------------------------------------
def test_the_derived_and_measured_standard_errors_agree():
    """S08 acceptance condition 3, in miniature.

    Given per-entry differences that are +-1 at rate f and 0 otherwise, the flip-rate form
    sqrt(f/n) and the measured form sd(d)/sqrt(n) must land on the same number. Campaign 1 got
    0.366 pp derived against 0.360 pp reported on real arms.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    rate = 0.04
    flipped = rng.random(n) < rate
    differences = np.where(flipped, rng.choice([-1.0, 1.0], size=n), 0.0)
    derived = floor_report.induced_standard_error(rate, n)
    measured = floor_report.induced_from_differences(differences, n)
    assert measured == pytest.approx(derived, rel=0.05)


def test_the_flip_rate_counts_changed_outcomes():
    first = np.array([True, True, False, False])
    second = np.array([True, False, False, True])
    assert floor_report.flip_rate(first, second) == 0.5


def test_relative_spread_is_scale_free():
    """The value floor is quoted relative to the merit's own median, so two merits whose units
    differ by orders of magnitude are comparable."""
    values = pd.DataFrame({'a': [10.0, 1000.0], 'b': [11.0, 1100.0], 'c': [10.5, 1050.0]})
    spread = floor_report.relative_spread(values)
    assert spread.iloc[0] == pytest.approx(spread.iloc[1])


def test_the_aggregate_is_composed_from_the_split_not_the_sample():
    """A lattice that is 1/14 of the balanced sample but 1/300 of the split must contribute at the
    split's weight.

    This is the check that separates a valid aggregate from campaign 1's shortcut. Two lattices,
    equally sampled at 40 patterns each and with identical measured standard errors, but one is
    600 of the split and the other 20. Weighting them equally -- what the sample's own shape would
    give -- overstates the rare lattice's contribution thirty-fold.
    """
    per_lattice = pd.DataFrame([
        dict(bravais_lattice='aP', se_pp=1.0, n_entries=40),
        dict(bravais_lattice='cF', se_pp=1.0, n_entries=40),
        ])
    composition = pd.DataFrame([
        dict(bravais_lattice='aP', split_entries=600, floor_weight=600/620),
        dict(bravais_lattice='cF', split_entries=20, floor_weight=20/620),
        ])
    se, covered = floor_report.compose_aggregate(per_lattice, composition)
    expected = np.sqrt((600/620)**2*1.0*(40/600) + (20/620)**2*1.0*(40/20))
    assert se == pytest.approx(expected)
    assert covered == pytest.approx(1.0)
    # The naive equal-weight version is materially different, which is the point.
    naive = np.sqrt(2*(0.5**2)*1.0*(40/600))
    assert not se == pytest.approx(naive, rel=0.1)


def test_a_lattice_missing_from_the_composition_is_dropped_and_reported():
    """`lattice_weight_covered` is how a caller sees that the aggregate is incomplete, rather than
    getting a confident number over whatever happened to be present."""
    per_lattice = pd.DataFrame([
        dict(bravais_lattice='aP', se_pp=1.0, n_entries=40),
        dict(bravais_lattice='zz', se_pp=1.0, n_entries=40),
        ])
    composition = pd.DataFrame([dict(bravais_lattice='aP', split_entries=600, floor_weight=0.5)])
    se, covered = floor_report.compose_aggregate(per_lattice, composition)
    assert np.isfinite(se)
    assert covered == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------
# The refusal that makes the arms an ensemble rather than four different experiments
# ----------------------------------------------------------------------------------------
def _arm(digests, bundle='c2_error1_cont0'):
    """An arm's entry table: one row per (entry, condition bundle)."""
    return pd.DataFrame({'entry_id': [key[0] if isinstance(key, tuple) else key
                                      for key in digests],
                         'condition_bundle': [key[1] if isinstance(key, tuple) else bundle
                                              for key in digests],
                         'q2_digest': list(digests.values())})


def test_identical_peak_lists_pass():
    digests = {'E1': 'aaa', 'E2': 'bbb'}
    assert floor_report.check_arms_are_comparable({'a': _arm(digests), 'b': _arm(dict(digests))})


def test_differing_peak_lists_raise_rather_than_warn():
    """If the base seed moved, the arms differ in their DATA. That is not a floor, and it looks
    exactly like a large one -- so it raises before any number is read."""
    with pytest.raises(SystemExit, match='disagree on the peak lists'):
        floor_report.check_arms_are_comparable({
            'a': _arm({'E1': 'aaa', 'E2': 'bbb'}),
            'b': _arm({'E1': 'aaa', 'E2': 'DIFFERENT'}),
            })


def test_arms_sharing_no_entries_raise():
    with pytest.raises(SystemExit, match='share no entries'):
        floor_report.check_arms_are_comparable({
            'a': _arm({'E1': 'aaa'}), 'b': _arm({'E9': 'aaa'})})


def test_two_conditions_of_one_entry_are_not_compared_against_each_other():
    """The check keys on (entry, condition), not on the entry alone.

    Two condition bundles apply different noise to the same crystal, so their peak lists differ by
    design. Keyed on `entry_id` alone the check would compare a pattern against a deliberately
    different pattern and fail on correct data -- and the obvious repair, relaxing the check, would
    have thrown away the one guard that makes the arms an ensemble.
    """
    arm = _arm({('E1', 'c2_error1_cont0'): 'aaa', ('E1', 'c2_error2_cont0'): 'DIFFERENT'})
    assert floor_report.check_arms_are_comparable({'a': arm, 'b': arm.copy()})


def test_a_difference_within_one_condition_still_raises():
    """The relaxation must not go too far: the same (entry, condition) in two arms is the thing
    that has to be identical, and that check survives."""
    left = _arm({('E1', 'c2_error1_cont0'): 'aaa', ('E1', 'c2_error2_cont0'): 'bbb'})
    right = _arm({('E1', 'c2_error1_cont0'): 'MOVED', ('E1', 'c2_error2_cont0'): 'bbb'})
    with pytest.raises(SystemExit, match='disagree on the peak lists'):
        floor_report.check_arms_are_comparable({'a': left, 'b': right})


def test_every_floor_merit_is_one_the_subsampler_ranked_on():
    """The arms are subsampled the way Benchmark B is, so a merit outside the retention rule would
    have optimistic ranks and its floor would be measured on a thinned field (C2-F-077)."""
    from mlindex.model_training import FomMetrics
    assert set(floor_report.FLOOR_MERITS) <= set(FomMetrics.RANK_EXACT_MERITS)


# ----------------------------------------------------------------------------------------
# The two submit scripts. Neither can be executed here, so what is checkable is that they
# say what the design says -- and every one of these is a way a submitted job would run
# for its full walltime and produce something that cannot be used.
# ----------------------------------------------------------------------------------------
import re

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'mlindex', 'scripts')
ARMS_PATH = os.path.join(SCRIPTS, 'submit_fom_floor_arms.sh')
FULL_PATH = os.path.join(SCRIPTS, 'submit_fom_full_retained.sh')

# Benchmark B's own, from its manifest. A fully retained pool generated at anything else does not
# reproduce its candidates, and gate 6 would then be differencing two different candidate sets
# rather than auditing the retention rule.
BENCHMARK_SEED = 12345
BENCHMARK_OPTIMIZER_SEED = 12345
BENCHMARK_POOL_SIZE = 2


def _script(path):
    """The script's executable lines, comments stripped.

    Stripped because these scripts carry their runbook in the header -- `docs/` is git-ignored and
    never reaches NERSC, so the reasoning has to travel with the file. That means the header
    legitimately contains the very strings these tests forbid: it explains why the driver is not
    wrapped in `srun`, and why `--no-subsample` is not `--no-label`. Matching raw text would fail
    on the documentation for the rule it is checking.
    """
    with open(path, encoding='utf-8') as handle:
        lines = [line for line in handle
                 if not line.lstrip().startswith('#') or line.startswith('#SBATCH')]
    return ''.join(lines)


def _array_bounds(text):
    match = re.search(r'#SBATCH --array=(\d+)-(\d+)', text)
    assert match, 'no --array directive'
    return int(match.group(1)), int(match.group(2))


def test_the_full_retained_job_reproduces_benchmark_bs_search_exactly():
    """Gate 6 differences a thinned pool against a full one, which is only meaningful if both hold
    the same candidates. C2-F-058 gives that for a subset run at the same seed -- but only at the
    same seed, and only at the same pool_size, which is part of the benchmark's identity
    (C2-F-069) because the per-pattern search seed keys on the rank count."""
    text = _script(FULL_PATH)
    assert f'SEED={BENCHMARK_SEED}' in text
    assert f'OPTIMIZER_SEED={BENCHMARK_OPTIMIZER_SEED}' in text
    assert f'POOLSIZE={BENCHMARK_POOL_SIZE}' in text
    assert '--no-subsample' in text
    # Labelling must still run: the retention rule keeps every correct candidate, and the driver
    # refuses to subsample an unlabelled pool rather than deleting the positives silently.
    assert '--no-label' not in text


def test_the_full_retained_array_matches_its_condition_list():
    text = _script(FULL_PATH)
    conditions = re.search(r'CONDITIONS=\(([^)]*)\)', text)
    assert conditions, 'no CONDITIONS list'
    names = re.findall(r'"([^"]+)"', conditions.group(1))
    assert _array_bounds(text) == (0, len(names) - 1), (
        'a task index past the end of the list runs the driver with an empty condition')
    from mlindex.model_training import FomConditions
    assert set(names) <= set(FomConditions.BY_KEY)


def test_the_full_retained_job_covers_the_floor_sample():
    """It has to be the same patterns as the arms, or it serves only the gate and neither the
    tie-break floor nor a fully retained arm 1."""
    text = _script(FULL_PATH)
    assert 'S08_floor_entries.csv' in text
    assert '--entry-ids-file' in text


def test_the_floor_arms_share_a_base_seed_and_vary_only_the_optimizer_seed():
    """The whole design. If --seed moved too, the arms would differ in their peak lists and the
    spread would be generation noise and scoring noise together."""
    text = _script(ARMS_PATH)
    assert f'SEED={BENCHMARK_SEED}' in text
    assert '--optimizer-seed "$ARM_SEED"' in text
    assert '--seed "$SEED"' in text
    tasks = re.findall(r'"(\d+) (\w+)"', text)
    seeds = {seed for seed, _ in tasks}
    assert len(seeds) > 1, 'the arms do not vary the optimizer seed'
    assert str(BENCHMARK_OPTIMIZER_SEED) not in seeds, (
        "an arm at Benchmark B's own seed would duplicate arm 1 rather than adding one")
    assert _array_bounds(text) == (0, len(tasks) - 1)


def test_both_scripts_keep_benchmark_bs_pool_width():
    for path in (ARMS_PATH, FULL_PATH):
        assert f'POOLSIZE={BENCHMARK_POOL_SIZE}' in _script(path), path


def test_neither_script_wraps_the_driver_in_srun():
    """A bare `srun -n 1` pins CPU affinity to one core and strangles the 128 processes."""
    for path in (ARMS_PATH, FULL_PATH):
        assert 'srun' not in _script(path), path


# ----------------------------------------------------------------------------------------
# The figure. Drafted now, against the report's own output shape, so that when the arms land
# it is one command rather than a design problem solved under time pressure.
# ----------------------------------------------------------------------------------------
def _report_shaped(seed=0):
    """Synthetic tables in exactly the shape `main` writes, with campaign 1's structure: the floor
    rises with free cell parameters and barely moves with the condition."""
    rng = np.random.default_rng(seed)
    lattice = pd.DataFrame([
        dict(merit='M_sym', baseline='M20', metric='operating_point', bravais_lattice=name,
             se_pp=0.05*free**2.3*rng.uniform(0.8, 1.25), n_entries=40)
        for name, free in floor_report.FREE_PARAMETERS.items()])
    aggregate = pd.DataFrame([dict(merit='M_sym', baseline='M20', metric='operating_point',
                                   se_pp=0.61, lattice_weight_covered=1.0,
                                   composed_from='split_entries')])
    condition = pd.DataFrame([
        dict(merit='M_sym', metric='operating_point', condition_bundle=tag, mean=45.0,
             sd_pp=value, range_pp=value*2, n_arms=4)
        for tag, value in (('c2_error1_cont0', 0.58), ('c2_error2_cont0', 0.66),
                           ('c2_error0.1_cont0', 0.54))])
    return lattice, aggregate, condition


def test_the_figure_renders(tmp_path):
    pytest.importorskip('matplotlib')
    lattice, aggregate, condition = _report_shaped()
    out = tmp_path / 'floor.png'
    floor_report.figure(lattice, aggregate, condition, out)
    assert out.exists() and out.stat().st_size > 20_000


def test_the_figure_orders_lattices_by_free_parameters_not_alphabetically():
    """Alphabetical order would hide the one structural claim the figure exists to make."""
    order = list(floor_report.FREE_PARAMETERS)
    assert order != sorted(order)
    assert floor_report.FREE_PARAMETERS['cF'] == 1
    assert floor_report.FREE_PARAMETERS['aP'] == 6
    assert all(floor_report.FREE_PARAMETERS[name] >= 4
               for name in floor_report.GAIN_LATTICES)


def test_the_figure_survives_a_lattice_the_composition_does_not_name(tmp_path):
    """A pool missing a lattice must not take the figure down -- five of them are hard-capped by
    the source population and one could come back empty (C2-R-010)."""
    pytest.importorskip('matplotlib')
    lattice, aggregate, condition = _report_shaped()
    lattice = lattice.loc[~lattice['bravais_lattice'].isin(['cF', 'cI'])]
    out = tmp_path / 'partial.png'
    floor_report.figure(lattice, aggregate, condition, out)
    assert out.exists()


def test_the_figure_survives_an_absent_aggregate(tmp_path):
    """The aggregate needs the composition table; without it the per-lattice panel is still the
    useful half and must still render rather than raising."""
    pytest.importorskip('matplotlib')
    lattice, _, condition = _report_shaped()
    out = tmp_path / 'no_aggregate.png'
    floor_report.figure(lattice, pd.DataFrame(columns=['metric', 'se_pp']), condition, out)
    assert out.exists()


def test_condition_tags_are_shown_by_their_readable_key():
    """`c2_error1_cont0` on an axis is unreadable and every tag looks like every other."""
    assert floor_report._condition_label('c2_error1_cont0') == 'nominal'
    assert floor_report._condition_label('c2_error2_cont0') == 'noisy'
    # An unknown tag falls back to itself rather than raising or rendering blank.
    assert floor_report._condition_label('not_a_tag') == 'not_a_tag'
