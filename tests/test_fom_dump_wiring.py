"""S06 -- the dump driver's split, decile and subsampling wiring.

Three things the driver used to get wrong or not do at all, each of which is silent when it fails.

* It wrote `subsampled: true` into `manifest.json` whenever `--no-subsample` was absent, while the
  subsampler did not exist. A manifest that misdescribes its own pool is worse than one that omits
  the field, because every later loader believes it.
* It wrote `volume_decile = -1` on every entry row, so the column schema v3 exists to freeze was
  never populated from the frozen manifest.
* It would have subsampled an unlabelled pool. The retention rule keeps every correct candidate,
  so with no `is_correct` column it would delete the entire signal at a base rate under 1 %.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlindex.scripts.run_fom_dump import (_parse_args, load_manifest,
                                          preflight_subsampling, subsample_or_refuse)


def _manifest_frame():
    return pd.DataFrame({
        'identifier': ['AAA', 'BBB', 'CCC'],
        'bravais_lattice': ['aP', 'mP', 'aP'],
        'volume_true': [100.0, 200.0, 300.0],
        'volume_decile': [0, 5, 9],
        'split': ['fom-train', 'fom-dev', 'fom-test'],
        'arm': ['core', 'core+mechanism', 'core'],
        })


def test_load_manifest_returns_the_frozen_decile(tmp_path):
    path = tmp_path / 'manifest.parquet'
    _manifest_frame().to_parquet(path, index=False)
    manifest = load_manifest(str(path))
    assert list(manifest.index) == ['AAA', 'BBB', 'CCC']
    assert int(manifest.loc['CCC', 'volume_decile']) == 9
    assert manifest.loc['BBB', 'split'] == 'fom-dev'


def test_a_manifest_without_the_decile_is_refused(tmp_path):
    # Loudly, because the alternative is a benchmark whose volume stratification silently comes
    # from whatever row set each analysis happened to hold -- which is R14 exactly.
    path = tmp_path / 'manifest.parquet'
    _manifest_frame().drop(columns=['volume_decile']).to_parquet(path, index=False)
    with pytest.raises(SystemExit, match='volume_decile'):
        load_manifest(str(path))


def test_a_missing_manifest_is_refused_rather_than_invented(tmp_path):
    with pytest.raises(SystemExit, match='must not be'):
        load_manifest(str(tmp_path / 'absent.parquet'))
    assert load_manifest(None) is None


def test_the_subsampling_defaults_are_the_measured_ones():
    args = _parse_args(['--out-dir', '/tmp/unused'])
    # C2-F-051: 500 retained 35 % of the pool and advertised subsampling while doing almost none.
    assert args.top_k == 200
    assert args.negative_rate == 0.05
    assert args.no_subsample is False
    assert args.prune_threshold == 1.5


def _pool(n=200, labelled=False):
    frame = pd.DataFrame({
        'entry_id': ['E1'] * n,
        'condition_bundle': ['c2_error1_cont0'] * n,
        'bravais_lattice': ['aP'] * n,
        'candidate_id': np.arange(n),
        'M20': np.linspace(1.0, 9.0, n),
        })
    if labelled:
        frame['is_correct'] = frame['candidate_id'] == 0
    return frame


def test_the_driver_refuses_to_subsample_an_unlabelled_pool():
    # The order is label -> subsample -> consolidate, and it is forced by the retention rule
    # itself. Refusing is the whole point: a silent subsample of an unlabelled pool cannot be
    # detected afterwards, because a pool with no correct candidate is indistinguishable from a
    # generation failure.
    args = _parse_args(['--out-dir', '/tmp/unused'])
    with pytest.raises(SystemExit, match='unlabelled'):
        subsample_or_refuse([_pool()], args)


def test_the_driver_subsamples_a_labelled_pool_and_reports_that_it_did():
    args = _parse_args(['--out-dir', '/tmp/unused', '--top-k', '5', '--negative-rate', '0.01'])
    frames, subsampled = subsample_or_refuse([_pool(labelled=True)], args)
    assert subsampled is True
    thinned = frames[0]
    assert thinned.shape[0] < 200
    assert bool(thinned['is_correct'].any()), 'the positive was dropped'
    assert set(thinned['retained_reason']) <= {'correct', 'top_k', 'sampled'}


def test_no_subsample_reports_that_it_did_not():
    # The bug this pins: the manifest used to claim `subsampled: true` on a run that kept every
    # row, because it recorded the flag rather than the outcome.
    args = _parse_args(['--out-dir', '/tmp/unused', '--no-subsample'])
    frames, subsampled = subsample_or_refuse([_pool(labelled=True)], args)
    assert subsampled is False
    assert frames[0].shape[0] == 200


def test_the_refusal_happens_before_the_search_not_after():
    # A guard that fires at the end costs the whole bundle. Campaign 1 added abort-safety after
    # losing a run near the end of a 2.5 h bundle; a late refusal reintroduces the same loss.
    with pytest.raises(SystemExit, match='unlabelled'):
        preflight_subsampling(_parse_args(['--out-dir', '/tmp/unused']))
    # And it is silent when the configuration is possible.
    assert preflight_subsampling(_parse_args(['--out-dir', '/tmp/unused',
                                              '--no-subsample'])) is None
