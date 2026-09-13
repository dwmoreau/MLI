"""The template yield tool: its selection is production's, and its labels are the validator's."""
import numpy as np
import pandas as pd
import pytest

from test_model_training import (  # noqa: F401  (all_optimizers is a fixture)
    EXPECTED_DIR,
    N_GENERATE,
    _assert_candidates_match,
    _cases,
    all_optimizers,
)
from mlindex.optimization.CandidateValidation import (
    is_correct_known_bl_batch,
    validate_candidate_known_bl,
)
from mlindex.scripts import run_template_yield as tool
from mlindex.utilities.UnitCellTools import get_partial_unit_cell


def test_the_ranking_starts_with_what_production_selects(unique_test_metadata, all_optimizers):
    """The first n ranked templates are generate(n): on this machine exactly, and against the
    fixtures the pipeline wrote, to the tolerance the generator tests use."""
    for q2_obs, _, _, bl, _ in _cases(unique_test_metadata):
        templator = all_optimizers[bl].wrapper.miller_index_templator[bl]
        _, ranked = tool.ranked_templates(templator, q2_obs, 12345)
        np.testing.assert_array_equal(
            ranked[:N_GENERATE], templator.generate(N_GENERATE, np.random.default_rng(12345), q2_obs))
        _assert_candidates_match(
            ranked[:N_GENERATE], np.load(EXPECTED_DIR / f"mi_templates_{bl}.npy"),
            f"ranked templates {bl}")


def test_the_batch_label_agrees_with_the_scalar_validator(unique_test_metadata, all_optimizers):
    for q2_obs, unit_cell, _, bl, lattice_system in _cases(unique_test_metadata):
        templator = all_optimizers[bl].wrapper.miller_index_templator[bl]
        cells, _ = tool.ranked_templates(templator, q2_obs, 12345)
        cells = cells[:200]
        truth = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
        batch = is_correct_known_bl_batch(truth, cells, lattice_system)
        scalar = np.array([validate_candidate_known_bl(unit_cell, cell, bl)[0] for cell in cells])
        np.testing.assert_array_equal(batch, scalar, err_msg=bl)


@pytest.mark.parametrize("rtol", tool.DEFAULT_RTOLS)
def test_the_true_cell_is_found_and_a_cell_outside_the_tolerance_is_not(unique_test_metadata, rtol):
    for _, unit_cell, _, bl, lattice_system in _cases(unique_test_metadata):
        truth = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
        outside = truth * (1 + 3 * rtol)
        assert tool.first_correct_rank(truth, np.stack((outside, truth)), lattice_system, rtol) == 1, bl
        assert tool.first_correct_rank(truth, outside[np.newaxis], lattice_system, rtol) == -1, bl


def _manifest(tmp_path):
    identifiers = [f"crystal{index:04d}" for index in range(400)]
    frame = pd.DataFrame({
        "identifier": identifiers,
        "bravais_lattice": "tP",
        "split": ["fom-train"] * 300 + ["fom-dev"] * 100,
    })
    path = tmp_path / "split_manifest.parquet"
    frame.to_parquet(path)
    return path


def test_the_groups_are_disjoint_and_a_crystal_keeps_its_group_whatever_is_drawn(tmp_path):
    path = _manifest(tmp_path)
    fit = set(tool.draw_group(path, "fit", 200, 7, ["tP"])["identifier"])
    select = set(tool.draw_group(path, "select", 200, 7, ["tP"])["identifier"])
    report = set(tool.draw_group(path, "report", 200, 7, ["tP"])["identifier"])

    assert not fit & select
    assert 0.1 < len(select) / (len(fit) + len(select)) < 0.3
    assert all(int(name[len("crystal"):]) >= 300 for name in report)
    assert not report & (fit | select)

    smaller = tool.draw_group(path, "select", 300, 7, ["tP"])
    assert set(smaller["identifier"]) >= select


def test_the_template_count_is_read_from_the_factory(models_dir):
    if models_dir is None:
        pytest.skip("ML models not available")
    # get_triclinic_optimizer asks the templater for int(0.55 * 6000) cells.
    assert tool.production_template_count("aP", models_dir) == 3300


def test_help_is_ascii():
    tool.build_parser().format_help().encode("ascii")
