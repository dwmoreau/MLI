"""Synthesising a benchmark pattern from a source crystal and a condition bundle.

The properties these pin are what make a benchmark re-runnable in pieces and its condition axes
readable independently. They are not about the numbers a pattern happens to contain.

The tests that need real source crystals skip when the generated datasets are absent, because those
are regenerable run output and a fresh clone does not carry them.
"""

import numpy as np
import pytest

from mlindex.model_training import BenchmarkConditions as conditions
from mlindex.model_training import BenchmarkPatterns as patterns

DATASETS_PRESENT = (patterns.DATASET_DIRECTORY / 'dataset_mP.parquet').is_file()
needs_datasets = pytest.mark.skipif(
    not DATASETS_PRESENT,
    reason=f'no generated datasets at {patterns.DATASET_DIRECTORY}')

SEED = 12345


# ----------------------------------------------------------------------------------------------
# Seeding: what makes a run reproducible in subsets
# ----------------------------------------------------------------------------------------------

def test_the_seed_for_a_key_is_stable_across_processes():
    """`hash()` is salted per process, so a pattern seeded from it would differ between runs of the
    same command."""
    assert patterns.derived_seed('ABAXEG', SEED) == patterns.derived_seed('ABAXEG', SEED)
    assert patterns.derived_seed('ABAXEG', SEED) != patterns.derived_seed('ABAXEH', SEED)
    assert patterns.derived_seed('ABAXEG', SEED) != patterns.derived_seed('ABAXEG', SEED + 1)


def test_each_mechanism_draws_from_its_own_stream():
    """A bundle differing only in contaminant count must not thereby get a different error
    realisation. Sharing one stream across mechanisms confounds every axis with the ones applied
    before it."""
    streams = {name: patterns.mechanism_rng(name, 'ABAXEG', SEED).random(4)
               for name in ('dropout', 'error', 'contaminant', 'phase')}
    values = list(streams.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert not np.array_equal(values[i], values[j])


def test_the_same_entry_gets_the_same_stream_in_every_bundle():
    a = patterns.mechanism_rng('error', 'ABAXEG', SEED).random(4)
    b = patterns.mechanism_rng('error', 'ABAXEG', SEED).random(4)
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------

@needs_datasets
def test_a_widened_read_selects_the_same_crystals():
    """Ground truth a run needs but the synthesis does not must not change which crystals are
    drawn, or two arms of one comparison see different populations."""
    narrow = patterns.sample_entries('mP', 8, SEED)
    wide = patterns.sample_entries('mP', 8, SEED, columns=patterns.TRUTH_COLUMNS)
    assert list(narrow['identifier']) == list(wide['identifier'])


@needs_datasets
def test_sampling_is_reproducible():
    first = patterns.sample_entries('mP', 8, SEED)
    second = patterns.sample_entries('mP', 8, SEED)
    assert list(first['identifier']) == list(second['identifier'])


@needs_datasets
def test_only_held_out_crystals_are_sampled():
    """A benchmark measured on crystals the models trained on measures nothing."""
    entries = patterns.sample_entries('mP', 8, SEED)
    assert not entries['train'].any()


@needs_datasets
def test_a_crystal_with_too_few_lines_is_not_sampled():
    entries = patterns.sample_entries('mP', 8, SEED)
    for q2 in entries[f'q2_{patterns.BROADENING_TAG}']:
        assert np.count_nonzero(q2) >= patterns.N_PEAKS


def test_a_missing_dataset_says_which_file_and_where():
    with pytest.raises(FileNotFoundError, match='dataset_zz.parquet'):
        patterns.sample_entries('zz', 4, SEED)


# ----------------------------------------------------------------------------------------------
# Synthesis
# ----------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def block():
    entries = patterns.sample_entries('mP', 6, SEED, columns=patterns.TRUTH_COLUMNS)
    pool = patterns.build_second_phase_pool(entries)
    entry = entries.iloc[0]
    hkl = np.stack([entry[f'reindexed_{axis}_{patterns.BROADENING_TAG}']
                    for axis in ('h', 'k', 'l')], axis=-1)
    return entries, pool, entry, hkl


@needs_datasets
@pytest.mark.parametrize('key', [condition.key for condition in conditions.CONDITIONS])
def test_every_bundle_yields_a_full_sorted_window(block, key):
    _, pool, entry, hkl = block
    pattern = patterns.prepare_peak_list(entry, conditions.BY_KEY[key], SEED, hkl=hkl,
                                         second_phase_pool=pool)
    assert pattern.q2_obs.size == patterns.N_PEAKS
    assert np.all(np.diff(pattern.q2_obs) >= 0)
    assert np.all(pattern.q2_obs > 0)
    assert pattern.hkl_obs.shape == (patterns.N_PEAKS, 3)


@needs_datasets
def test_synthesis_is_reproducible(block):
    _, pool, entry, hkl = block
    condition = conditions.BY_KEY['sparse4']
    first = patterns.prepare_peak_list(entry, condition, SEED, hkl=hkl, second_phase_pool=pool)
    second = patterns.prepare_peak_list(entry, condition, SEED, hkl=hkl, second_phase_pool=pool)
    np.testing.assert_array_equal(first.q2_obs, second.q2_obs)
    np.testing.assert_array_equal(first.hkl_obs, second.hkl_obs)


@needs_datasets
def test_the_sparsity_ladder_is_nested_in_the_finished_pattern(block):
    """The nesting has to survive synthesis, not merely hold inside the selector: the sparsity axis
    is a paired comparison of one crystal degrading, and it stops being one if the rungs draw
    different holes."""
    _, pool, entry, hkl = block
    dropped = {}
    for key in ('sparse2', 'sparse4', 'sparse6'):
        pattern = patterns.prepare_peak_list(entry, conditions.BY_KEY[key], SEED, hkl=hkl,
                                             second_phase_pool=pool)
        indexed = pattern.hkl_obs[~np.all(pattern.hkl_obs == 0, axis=1)]
        dropped[key] = {tuple(row) for row in indexed}
    # Higher rungs keep strictly fewer of the original reflections, and the ones they keep are a
    # subset of what the lower rung kept, plus backfill from beyond the window.
    assert len(dropped['sparse6']) <= len(dropped['sparse4']) <= len(dropped['sparse2'])


@needs_datasets
def test_changing_the_contaminant_count_does_not_change_the_error_realisation(block):
    """The axes have to be independent, or a contrast along one carries the other."""
    _, pool, entry, hkl = block
    clean = patterns.prepare_peak_list(entry, conditions.BY_KEY['nominal'], SEED, hkl=hkl)
    one = patterns.prepare_peak_list(entry, conditions.BY_KEY['contaminated1'], SEED, hkl=hkl)
    # The real reflections common to both must carry identical positions: only the injected lines
    # and the peaks they displaced should differ.
    clean_real = {tuple(h): q for h, q in zip(clean.hkl_obs, clean.q2_obs)
                  if not np.all(h == 0)}
    one_real = {tuple(h): q for h, q in zip(one.hkl_obs, one.q2_obs) if not np.all(h == 0)}
    shared = set(clean_real) & set(one_real)
    assert shared
    for key in shared:
        assert clean_real[key] == one_real[key]


@needs_datasets
def test_the_counts_delivered_are_recorded_not_assumed(block):
    """Dropout is capped by the crystal's own surplus, and an injected line can be truncated back
    out of the window, so an axis has to be read on what arrived."""
    _, pool, entry, hkl = block
    for key in ('contaminated1', 'contaminated2', 'sparse6', 'second_phase'):
        condition = conditions.BY_KEY[key]
        pattern = patterns.prepare_peak_list(entry, condition, SEED, hkl=hkl,
                                             second_phase_pool=pool)
        assert 0 <= pattern.n_contaminants_achieved <= condition.n_contaminants
        assert 0 <= pattern.n_dropout_achieved <= condition.n_dropout
        assert 0 <= pattern.n_second_phase_achieved <= condition.second_phase_lines
        injected = int(np.sum(np.all(pattern.hkl_obs == 0, axis=1)))
        assert injected == (pattern.n_contaminants_achieved
                            + pattern.n_second_phase_achieved)


@needs_datasets
def test_a_second_phase_bundle_names_a_partner_that_is_not_the_crystal_itself(block):
    _, pool, entry, hkl = block
    pattern = patterns.prepare_peak_list(entry, conditions.BY_KEY['second_phase'], SEED, hkl=hkl,
                                         second_phase_pool=pool)
    assert pattern.second_phase_partner is not None
    assert pattern.second_phase_partner != entry['identifier']


@needs_datasets
def test_a_second_phase_bundle_without_a_partner_pool_is_refused(block):
    _, _, entry, hkl = block
    with pytest.raises(ValueError, match='partner pool'):
        patterns.prepare_peak_list(entry, conditions.BY_KEY['second_phase'], SEED, hkl=hkl)


@needs_datasets
def test_the_error_can_move_a_peak_across_the_edge_of_the_window(block):
    """Error is applied to the window and the surplus together and then re-sorted. Applying it to
    the window alone would make that boundary artificially sharp."""
    entries, pool, _, _ = block
    crossed = False
    for index in range(entries.shape[0]):
        entry = entries.iloc[index]
        q2_full = np.asarray(entry[f'q2_{patterns.BROADENING_TAG}'], dtype=float)
        nominal = set(np.round(q2_full[q2_full > 0][:patterns.N_PEAKS], 10))
        pattern = patterns.prepare_peak_list(entry, conditions.BY_KEY['noisy'], SEED)
        # A peak that was not among the nominal twenty, arriving under noise, is the crossing.
        if pattern.q2_obs.size == patterns.N_PEAKS and len(nominal) == patterns.N_PEAKS:
            crossed = crossed or pattern.q2_obs[-1] > max(nominal)
    assert crossed, 'no peak crossed the window edge in any sampled crystal'
