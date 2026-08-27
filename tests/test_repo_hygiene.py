"""Campaign FOM output is neither git-lfs nor committable.

Two rules that are invisible until they are broken, and expensive afterwards.

`.gitattributes` carries four repo-wide GLOBS -- `*.xlsx`, `*_trees.npz`,
`*_pitf_weights_*_quantized.onnx`, `*_calibration_weights_*_quantized.onnx` -- which exist for the
shipped model set under `mlindex/models/<system>_1/`. They also catch anything a research step
happens to name the same way, and the campaign's own outputs collide: S12's combiner is a tree
model, and campaign 1 named its random forests `*_random_forest_regressor_trees.npz` exactly.

Taking those paths out of lfs is only safe if they cannot be committed at all -- a large file that
is not an lfs pointer is a large file in the pack, and it cannot be removed without rewriting
history. So both halves are pinned here, together.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _attr(path):
    result = subprocess.run(['git', 'check-attr', 'filter', '--', path],
                            cwd=REPO, capture_output=True, text=True)
    return result.stdout.strip().rsplit(': ', 1)[-1]


def _ignored(path):
    return subprocess.run(['git', 'check-ignore', '-q', path],
                          cwd=REPO, capture_output=True).returncode == 0


@pytest.fixture(scope='module', autouse=True)
def _require_git():
    if subprocess.run(['git', 'rev-parse', '--git-dir'], cwd=REPO,
                      capture_output=True).returncode != 0:
        pytest.skip('not a git checkout')


# Filenames a campaign step would plausibly produce, each chosen to match one of the four globs.
CAMPAIGN_OUTPUT = [
    'mlindex/models/fom_combiner_c2/combiner_random_forest_regressor_trees.npz',
    'mlindex/models/fom_neural_c2/score_pitf_weights_c2_quantized.onnx',
    'mlindex/models/fom_prior_c2/prior_calibration_weights_c2_quantized.onnx',
    'mlindex/data/fom_benchmark_c2/summary.xlsx',
    'mlindex/data/fom_benchmark_c2/candidates_c2_error1_cont0_aP.parquet',
    'mlindex/characterization/fom/benchmark/entries.parquet',
    ]


@pytest.mark.parametrize('path', CAMPAIGN_OUTPUT)
def test_campaign_output_is_not_git_lfs(path):
    assert _attr(path) != 'lfs', (
        f'{path} would be stored as a git-lfs object. The four repo-wide globs in '
        '.gitattributes are for the SHIPPED model set; campaign output must be unset from them.')


@pytest.mark.parametrize('path', CAMPAIGN_OUTPUT)
def test_campaign_output_cannot_be_committed(path):
    assert _ignored(path), (
        f'{path} is committable. Taking it out of lfs without ignoring it is worse than leaving '
        'it in: a large file that is not an lfs pointer goes into the pack permanently.')


# The shipped model set is the reason those globs exist. Unsetting them for campaign paths must
# not touch it -- it is ~545 MB the wheel ships, and it belongs in lfs.
SHIPPED = [
    'mlindex/models/cubic_1/random_forest/cP_0_random_forest_regressor_trees.npz',
    'mlindex/models/cubic_1/integral_filter/cP_0/cP_0_calibration_weights_cubic_1_quantized.onnx',
    'mlindex/data/GroupSpec_cubic.xlsx',
    ]


@pytest.mark.parametrize('path', SHIPPED)
def test_the_shipped_model_set_is_still_lfs(path):
    assert _attr(path) == 'lfs', f'{path} has been taken out of git-lfs; the wheel ships it'


@pytest.mark.parametrize('path', SHIPPED + [
    'mlindex/scripts/run_fom_dump.py',
    'mlindex/data/hkl_ref_aP.npy',
    'mlindex/data/test_data/example.npy',
    ])
def test_shipped_files_are_still_committable(path):
    assert not _ignored(path), f'{path} has been ignored; it is part of the package'


def test_no_generated_data_path_is_currently_tracked():
    """Ignoring a path does nothing to a file already in the index."""
    result = subprocess.run(
        ['git', 'ls-files', 'mlindex/characterization', 'mlindex/data/fom_benchmark',
         'mlindex/data/generated_datasets', 'mlindex/models/fom_combiner'],
        cwd=REPO, capture_output=True, text=True)
    tracked = [line for line in result.stdout.split('\n') if line.strip()]
    assert not tracked, (
        f'{len(tracked)} generated-data files are tracked, e.g. {tracked[:3]}. .gitignore does '
        'not untrack them -- they need `git rm --cached` and a deliberate decision.')
