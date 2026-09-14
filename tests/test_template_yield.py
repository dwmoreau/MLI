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


def _per_entry(arm, found, first_rank, n_templates=5, n_ranked=50, oracle=True):
    """A yield run's rows for four tP patterns and one aP pattern."""
    lattices = ["tP", "tP", "tP", "tP", "aP"]
    return pd.DataFrame({
        "run": "seed1",
        "arm": arm,
        "entry_id": [f"crystal{index}" for index in range(5)],
        "condition_bundle": "b1_error1_cont0",
        "bravais_lattice": lattices,
        "n_templates": n_templates,
        "n_ranked": n_ranked,
        "first_rank_0.01": first_rank,
        "oracle_0.01": oracle,
        "refined_found": found,
        "refined_in_top_n": found,
    })


def test_a_template_counts_as_kept_only_within_the_production_depth():
    frame = _per_entry("rho", [True] * 5, [0, 4, 5, -1, 3])
    kept = tool.pattern_outcomes(frame, [0.01])["kept_0.01"].tolist()
    assert kept == [True, True, False, False, True]


def test_when_every_template_is_kept_the_unrefined_yield_is_the_ceiling():
    frame = _per_entry("rho", [True] * 5, [40, 40, 40, 40, 40], n_templates=60, n_ranked=50)
    assert tool.pattern_outcomes(frame, [0.01])["kept_0.01"].all()


def test_the_aggregate_is_an_unweighted_mean_over_lattices():
    frame = _per_entry("rho", [True, True, True, True, False], [0] * 5)
    _, aggregate = tool.summarise(tool.pattern_outcomes(frame, [0.01]))
    # tP finds 4 of 4 and aP 0 of 1: the unweighted mean is 0.5, the pooled share 0.8.
    assert aggregate["refined_found"].iloc[0] == pytest.approx(0.5)


def test_arms_are_paired_pattern_by_pattern():
    rho = _per_entry("rho", [True, False, False, True, False], [0] * 5)
    posterior = _per_entry("posterior_sigma", [True, True, True, False, False], [0] * 5)
    outcomes = tool.pattern_outcomes(pd.concat((rho, posterior)), [0.01])
    table = tool.paired(outcomes, "rho").set_index("scope")
    assert table.loc["tP", "b_only"] == 2 and table.loc["tP", "a_only"] == 1
    assert table.loc["tP", "delta"] == pytest.approx(0.25)
    assert table.loc["aggregate", "delta"] == pytest.approx((0.25 + 0.0) / 2)


def test_arms_that_cover_different_patterns_are_refused():
    rho = _per_entry("rho", [True] * 5, [0] * 5)
    posterior = _per_entry("posterior_sigma", [True] * 5, [0] * 5).iloc[:4]
    outcomes = tool.pattern_outcomes(pd.concat((rho, posterior)), [0.01])
    with pytest.raises(ValueError, match="different patterns"):
        tool.paired(outcomes, "rho")


def _write_run(directory, arms, search_seed):
    import json

    directory.mkdir(parents=True)
    frames = [_per_entry(arm, [True] * 5, [0] * 5).drop(columns="run") for arm in arms]
    table = pd.concat(frames, ignore_index=True)
    table["refused"] = ""
    table.to_parquet(directory / "per_entry.parquet", index=False)
    manifest = {field: "shared" for field in tool.SHARED_RUN_FIELDS}
    manifest.update(search_seed=search_seed, arms={arm: f"/models/{arm}" for arm in arms})
    with open(directory / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def test_a_floor_run_with_fewer_arms_is_read_through_the_arms_it_shares(tmp_path):
    _write_run(tmp_path / "seed12345", ["shipped", "rho", "posterior_sigma", "merits"], 12345)
    _write_run(tmp_path / "seed202", ["shipped", "rho", "posterior_sigma"], 202)
    runs = [str(tmp_path / "seed12345"), str(tmp_path / "seed202")]

    with pytest.raises(ValueError, match="arms"):
        tool.load_runs(runs)
    with pytest.raises(ValueError, match="merits"):
        tool.load_runs(runs, ["rho", "merits"])

    frame, _ = tool.load_runs(runs, ["rho", "posterior_sigma"])
    assert sorted(frame["arm"].unique()) == ["posterior_sigma", "rho"]
    assert sorted(frame["run"].unique()) == ["seed12345", "seed202"]


def test_the_floor_is_the_spread_of_the_paired_difference_between_runs():
    table = pd.DataFrame({"arm": "posterior_sigma", "reference": "rho", "scope": "aggregate",
                          "delta": [0.10, 0.20, 0.30]})
    row = tool.floor(table).iloc[0]
    assert row["delta_mean"] == pytest.approx(0.2)
    assert row["delta_sd"] == pytest.approx(0.1)
    assert row["delta_in_sds"] == pytest.approx(2.0)
    assert row["n_runs"] == 3
