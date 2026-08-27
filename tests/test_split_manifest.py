"""S06 -- the frozen split, and the drift it exists to prevent.

The campaign-1 defect these tests pin down (R14, F-108). `volume_decile` is a *within-lattice
percentile rank*, and a percentile rank rises when rows are dropped. Campaign 1 stored it in the
frozen manifest and then recomputed it downstream from whatever row set the caller happened to
hold, so once 33 entries were lost to unplaceable second-phase lines and the bundles were aligned
by intersection, 114 of 5 922 entries disagreed with the manifest -- every one of them moving up
by exactly one decile, and the hard stratum with them, from 286 entries to 298.

No campaign-1 number is wrong because of it; the definition was applied uniformly. What is wrong
is that "the hard stratum" denoted two different sets of entries in two different documents. The
test below is therefore in two halves: the join is stable under attrition, and the recompute is
NOT -- because a fix that silently stopped drifting would be indistinguishable from a fix that
stopped being applied.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.model_training.FomMetrics import entry_context, volume_decile
from mlindex.scripts.run_fom_split_manifest import (SPLIT_FRACTIONS, assign_arms, assign_splits,
                                                    check_manifest)
from mlindex.scripts.run_fom_split_manifest import volume_decile as manifest_volume_decile

LATTICES = ('cF', 'cP', 'oP', 'mP', 'mC', 'aP')


def _manifest(n_per_lattice=60, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for index, lattice in enumerate(LATTICES):
        frames.append(pd.DataFrame({
            'identifier': [f'{lattice}{position:04d}' for position in range(n_per_lattice)],
            'bravais_lattice': lattice,
            # Deliberately different volume scales per lattice: the decile is within-lattice, so
            # a global decile would put whole lattices in one bin and the test would not notice.
            'volume_true': rng.uniform(100, 4000) * (index + 1)
                           + rng.uniform(1, 100, n_per_lattice),
            }))
    manifest = pd.concat(frames, ignore_index=True)
    manifest['volume_decile'] = manifest_volume_decile(manifest)
    manifest['split'] = assign_splits(manifest, seed)
    manifest['arm'] = assign_arms(manifest, 0.2, seed)
    return manifest


def _entries(manifest, with_stored_decile=True):
    """An entry table shaped like the benchmark's, one row per (entry, condition bundle)."""
    frames = []
    for bundle in ('c2_error1_cont0', 'c2_error2_cont0'):
        frame = pd.DataFrame({
            'entry_id': manifest['identifier'].to_numpy(),
            'condition_bundle': bundle,
            'split': manifest['split'].to_numpy(),
            'bravais_lattice_true': manifest['bravais_lattice'].to_numpy(),
            'lattice_system_true': 'triclinic',
            'volume_true': manifest['volume_true'].to_numpy(),
            })
        if with_stored_decile:
            frame['volume_decile'] = manifest['volume_decile'].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_the_stored_decile_survives_attrition():
    # Gate 2. Re-deriving the stratification from an arbitrary SUBSET of rows must reproduce the
    # manifest exactly. This is the direct regression test for the campaign-1 drift.
    manifest = _manifest()
    entries = _entries(manifest)
    rng = np.random.default_rng(11)
    keep = rng.permutation(entries.shape[0])[:int(entries.shape[0] * 0.7)]
    subset = entries.iloc[np.sort(keep)].reset_index(drop=True)

    context = entry_context(subset)
    assert context.attrs['volume_decile_source'] == 'stored'
    expected = (manifest.set_index('identifier')['volume_decile']
                .reindex(subset['entry_id']).to_numpy())
    assert np.array_equal(context['volume_decile'].to_numpy(), expected)


def test_recomputing_the_decile_still_drifts_upward():
    """The other half, and the reason the first half means anything.

    The drift is compared against the recompute on the FULL row set, not against the manifest.
    That isolates attrition, which is the campaign-1 mechanism, from a second and unrelated effect
    the manifest comparison would mix in: an entry table carries one row per (entry, condition
    bundle), so a within-lattice rank over it runs over as many rows as there are bundles and
    lands a fraction of a decile away from a rank over the manifest's one-row-per-entry table.
    Both are consequences of recomputing rather than joining; only the first is one-directional,
    and a test that conflated them would fail for the wrong reason.
    """
    manifest = _manifest()
    entries = _entries(manifest).drop(columns=['volume_decile'])
    rng = np.random.default_rng(11)
    keep = np.sort(rng.permutation(entries.shape[0])[:int(entries.shape[0] * 0.7)])
    subset = entries.iloc[keep].reset_index(drop=True)

    full = entry_context(entries)
    recomputed = entry_context(subset)
    assert recomputed.attrs['volume_decile_source'] == 'recomputed'
    moved = recomputed['volume_decile'].to_numpy() - full['volume_decile'].to_numpy()[keep]
    assert (moved != 0).any(), 'the fixture is too small to show the drift at all'
    # And it moves in BOTH directions, which is not what the inherited record says. Campaign 1's
    # F-108 and the S06 handoff both state that "dropping rows can only raise a survivor's
    # within-lattice rank"; under attrition uncorrelated with volume it does not. A survivor's
    # new rank r' over n' survivors beats its old r/n only when the dropped rows sat mostly
    # ABOVE it, and random attrition drops them on both sides -- so the recompute is a two-sided
    # perturbation, not a conservative one. Campaign 1's 114 entries really did all move up,
    # which says its attrition was correlated with volume, not that the arithmetic forces it.
    #
    # The fix is unchanged -- join, do not recompute -- but the bound is wider than recorded:
    # a stratum defined on a recomputed decile can lose entries as well as gain them.
    assert (moved > 0).any() and (moved < 0).any()


def test_volume_decile_matches_the_campaign_one_rule():
    # The manifest's rule and the metrics module's rule have to be the same function, or the
    # stratum means one thing at freeze time and another at report time.
    manifest = _manifest()
    frame = manifest.rename(columns={'volume_true': 'volume_true'})[
        ['bravais_lattice', 'volume_true']]
    assert np.array_equal(np.asarray(manifest_volume_decile(manifest)),
                          np.asarray(volume_decile(frame)))


def test_the_split_is_disjoint_by_source_entry():
    # Gate 3, asserted in code rather than in a results document.
    manifest = _manifest()
    assert check_manifest(manifest)
    assert manifest['identifier'].is_unique
    assert manifest.groupby('identifier')['split'].nunique().max() == 1
    assert set(manifest['split']) == set(SPLIT_FRACTIONS)


def test_the_split_is_balanced_on_lattice_and_decile():
    manifest = _manifest(n_per_lattice=100)
    shares = (manifest.groupby(['bravais_lattice', 'split']).size()
              / manifest.groupby('bravais_lattice').size())
    for (lattice, split), share in shares.items():
        assert abs(share - SPLIT_FRACTIONS[split]) < 0.06, (lattice, split, share)
    # And no (lattice, decile) cell is missing from a split, which is what stratifying on the
    # decile buys and what a lattice-only split would not give.
    reached = manifest.groupby(['bravais_lattice', 'volume_decile'])['split'].nunique()
    assert reached.min() == 3


def test_the_mechanism_arm_nests_inside_the_core_arm():
    # `sample_entries` draws `rng.choice(size=n)`, so a smaller draw is NOT a subset of a larger
    # one and two arms sized independently would not be paired. Membership is assigned once, here.
    manifest = _manifest(n_per_lattice=100)
    mechanism = manifest[manifest['arm'] == 'core+mechanism']
    assert not mechanism.empty
    assert set(mechanism['identifier']) <= set(manifest['identifier'])
    # And the narrower arm keeps the wider one's shape, which is why it is drawn stratified.
    for lattice in LATTICES:
        share = ((mechanism['bravais_lattice'] == lattice).sum()
                 / (manifest['bravais_lattice'] == lattice).sum())
        assert abs(share - 0.2) < 0.06, (lattice, share)


def test_changing_the_mechanism_size_does_not_move_the_split():
    # Separate generators, so re-sizing one arm cannot silently reassign a single entry's split.
    manifest = _manifest(n_per_lattice=100)
    other = manifest.copy()
    other['arm'] = assign_arms(other, 0.35, 7)
    assert np.array_equal(manifest['split'].to_numpy(), other['split'].to_numpy())
    assert not np.array_equal(manifest['arm'].to_numpy(), other['arm'].to_numpy())


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_the_freeze_is_reproducible(seed):
    assert _manifest(seed=seed).equals(_manifest(seed=seed))
