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
def _arm(digests):
    return pd.DataFrame({'entry_id': list(digests), 'q2_digest': list(digests.values())})


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


def test_every_floor_merit_is_one_the_subsampler_ranked_on():
    """The arms are subsampled the way Benchmark B is, so a merit outside the retention rule would
    have optimistic ranks and its floor would be measured on a thinned field (C2-F-077)."""
    from mlindex.model_training import FomMetrics
    assert set(floor_report.FLOOR_MERITS) <= set(FomMetrics.RANK_EXACT_MERITS)
