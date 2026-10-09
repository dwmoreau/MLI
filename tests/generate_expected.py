"""Run once to populate tests/expected/ with baseline outputs.

Usage:
    python tests/generate_expected.py             # fast tests only
    python tests/generate_expected.py --models    # include model training tests
    python tests/generate_expected.py --all       # include model training + CLI tests
"""

import argparse
import ast
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from conftest import resolve_models_dir

REPO_ROOT = Path(__file__).parent.parent
EXPECTED_DIR = Path(__file__).parent / "expected"
TEST_DATA_DIR = REPO_ROOT / "mlindex" / "data" / "test_data"

EXPECTED_DIR.mkdir(exist_ok=True)

from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM


def load_test_case(row):
    bl = str(row["bravais lattice"])
    wavelength = float(row["wavelength"])
    peak_file = str(row["peak list file"])
    a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
    alpha = float(row["alpha"]) * np.pi / 180
    beta = float(row["beta"]) * np.pi / 180
    gamma = float(row["gamma"]) * np.pi / 180
    unit_cell = np.array([a, b, c, alpha, beta, gamma])
    if peak_file.endswith(".csv"):
        pattern_name = peak_file.replace(".csv", "")
        path = TEST_DATA_DIR / pattern_name / peak_file
        df = pd.read_csv(path, index_col=0)
        peak_positions = np.array(ast.literal_eval(df.loc["peak_positions"].iloc[0]))
        q2_obs = np.sort(peak_positions**2)[:20]
    else:
        case_name = peak_file.replace(".pkslst", "").replace(".pklst", "")
        npy_path = TEST_DATA_DIR / case_name / f"{case_name}_peak_list.npy"
        q2_obs = np.sort(np.load(npy_path))[:20]
    return q2_obs, unit_cell, wavelength, bl, BL_TO_LATTICE_SYSTEM[bl]


def all_cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


# ---------------------------------------------------------------------------
# Utility expected files (fast, no models)
# ---------------------------------------------------------------------------


_SPACEGROUP_HKL_REF_TEST_HKL = np.array(
    [
        [h, k, l]
        for h in range(8)
        for k in range(8)
        for l in range(8)
        if not (h == 0 and k == 0 and l == 0)
    ],
    dtype=np.int32,
)

_ALL_BRAVAIS_LATTICES = list(BL_TO_LATTICE_SYSTEM.keys())


def generate_spacegroup_hkl_ref_expected():
    import json
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    print("Generating spacegroup_hkl_ref expected files ...")
    for bl in _ALL_BRAVAIS_LATTICES:
        result = get_spacegroup_hkl_ref(_SPACEGROUP_HKL_REF_TEST_HKL, bl)
        keys = list(result.keys())
        with open(EXPECTED_DIR / f"spacegroup_hkl_ref_{bl}_keys.json", "w") as f:
            json.dump(keys, f)
        for idx, key in enumerate(keys):
            np.save(EXPECTED_DIR / f"spacegroup_hkl_ref_{bl}_{idx}.npy", result[key])
        print(f"  {bl}: {len(keys)} extinction groups")
    print("Done.")


def generate_utility_expected(test_metadata):
    from mlindex.utilities.UnitCellTools import (
        get_xnn_from_unit_cell,
        get_unit_cell_volume,
        get_hkl_matrix,
        get_partial_unit_cell,
    )

    def _xnn(unit_cell, lattice_system):
        uc_p = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
        return get_xnn_from_unit_cell(
            uc_p[np.newaxis], partial_unit_cell=True, lattice_system=lattice_system
        )

    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.FigureOfMerits import (
        get_M20_from_xnn,
        get_M20_likelihood_from_xnn,
    )

    print("Generating utility expected files ...")
    for q2_obs, unit_cell, wavelength, bl, lattice_system in all_cases(test_metadata):
        hkl_ref = np.load(TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)

        # unit_cell_volume
        np.save(EXPECTED_DIR / f"unit_cell_volume_{bl}.npy", get_unit_cell_volume(unit_cell[np.newaxis])[0])

        # hkl_matrix
        np.save(EXPECTED_DIR / f"hkl_matrix_{bl}.npy", hkl2)

        q2_calc = np.sum(hkl2 * xnn, axis=1)

        # q2_calculator
        np.save(EXPECTED_DIR / f"q2_calculator_{bl}.npy", q2_calc)

        # fast_assign
        q2_ref = q2_calc[np.newaxis]
        fa = fast_assign(q2_obs.astype(np.float64), q2_ref.astype(np.float64))
        np.save(EXPECTED_DIR / f"fast_assign_{bl}.npy", fa)

        hkl_assign = fa[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]

        # m20
        m20 = get_M20_from_xnn(q2_obs, xnn, hkl_assigned, hkl_ref, lattice_system)
        np.save(EXPECTED_DIR / f"m20_{bl}.npy", m20)

        # m20_likelihood
        log_lik, prob, M = get_M20_likelihood_from_xnn(
            q2_obs, xnn, hkl_assigned, lattice_system, bl
        )
        np.save(
            EXPECTED_DIR / f"m20_likelihood_{bl}.npy", np.stack([log_lik, M.flatten()])
        )

        print(f"  {bl}: M20={m20[0]:.3f}")

    print("Done.")


def generate_pkslst_expected(test_metadata_all):
    from mlindex.utilities.gsas import load_pkslst

    print("Generating load_pkslst expected files ...")
    for _, row in test_metadata_all.iterrows():
        peak_file = str(row["peak list file"])
        if not peak_file.endswith(".pkslst"):
            continue
        case_name = peak_file.replace(".pkslst", "")
        wavelength = float(row["wavelength"])
        pkslst_path = TEST_DATA_DIR / case_name / peak_file
        result = load_pkslst(str(pkslst_path), wavelength)
        np.save(EXPECTED_DIR / f"load_pkslst_{case_name}.npy", np.sort(result))
        print(f"  {case_name}")
    print("Done.")


# ---------------------------------------------------------------------------
# Reindexing expected files (fast, no models)
# ---------------------------------------------------------------------------


def generate_reindexing_expected(test_metadata):
    from mlindex.utilities.Reindexing import selling_reduction

    print("Generating reindexing expected files ...")
    for q2_obs, unit_cell, wavelength, bl, lattice_system in all_cases(test_metadata):
        result_uc, _, _ = selling_reduction(unit_cell[np.newaxis])
        np.save(EXPECTED_DIR / f"selling_reduction_{bl}.npy", result_uc[0])
        print(f"  {bl}")
    print("Done.")


# ---------------------------------------------------------------------------
# Model training expected files (slow, requires models)
# ---------------------------------------------------------------------------


def generate_model_training_expected(test_metadata):
    from mlindex.optimization.MPOptimizer import LocalComm
    from mlindex.optimization.MPIOptimizer import OptimizerManager
    from mlindex.optimization.UtilitiesOptimizer import (
        get_cubic_optimizer,
        get_hexagonal_optimizer,
        get_rhombohedral_optimizer,
        get_tetragonal_optimizer,
        get_orthorhombic_optimizer,
        get_monoclinic_optimizer,
        get_triclinic_optimizer,
    )

    print("Loading ML models ...")
    comm = LocalComm(1)
    # The same tree the tests pin to. Bare _resolve_models_dir() searches
    # MLINDEX_MODELS_DIR, then $XDG_DATA_HOME/mlindex/models, then the package -- so on a
    # machine that has ever run mlindex.download_models it silently regenerates every
    # fixture against a possibly older release than the checkout. It did exactly that
    # here, replacing all ten candidates in all fourteen abnn fixtures.
    models_dir = resolve_models_dir()
    bl_to_factory = {
        "cF": get_cubic_optimizer,
        "cI": get_cubic_optimizer,
        "cP": get_cubic_optimizer,
        "hP": get_hexagonal_optimizer,
        "hR": get_rhombohedral_optimizer,
        "tI": get_tetragonal_optimizer,
        "tP": get_tetragonal_optimizer,
        "oC": get_orthorhombic_optimizer,
        "oF": get_orthorhombic_optimizer,
        "oI": get_orthorhombic_optimizer,
        "oP": get_orthorhombic_optimizer,
        "mC": get_monoclinic_optimizer,
        "mP": get_monoclinic_optimizer,
        "aP": get_triclinic_optimizer,
    }

    print("Generating model training expected files ...")
    for q2_obs, unit_cell, wavelength, bl, lattice_system in all_cases(test_metadata):
        factory = bl_to_factory[bl]
        opt = factory(
            bl, "1", 1, comm, optimizer_class=OptimizerManager, seed=12345,
            models_directory=models_dir,
        )
        opt.wrapper.setup_random()
        sg = opt.wrapper.data_params["split_groups"][0]

        # random generator
        rng = np.random.default_rng(12345)
        result = opt.wrapper.random_unit_cell_generator[bl].generate(
            10,
            rng,
            q2_obs,
            model="random",
        )
        np.save(EXPECTED_DIR / f"random_gen_{bl}.npy", result)

        # random forest
        rng = np.random.default_rng(12345)
        result = opt.wrapper.random_forest_generator[sg].generate(10, rng, q2_obs)
        np.save(EXPECTED_DIR / f"random_forest_{bl}.npy", result)

        # MI templates
        rng = np.random.default_rng(12345)
        result = opt.wrapper.miller_index_templator[bl].generate(10, rng, q2_obs)
        np.save(EXPECTED_DIR / f"mi_templates_{bl}.npy", result)

        # ABNN
        rng = np.random.default_rng(12345)
        result = opt.wrapper.abnn_generator[sg].generate(
            10,
            rng,
            q2_obs,
            batch_size=2,
        )
        np.save(EXPECTED_DIR / f"abnn_{bl}.npy", result)

        # ABNN, resampling branch. The fixture above asks for ten cells against an
        # n_volumes of 100-200, so it takes the branch that assigns Miller indices once and
        # never resamples -- it has never covered the assignment half of the generator at all.
        # top_n=3 with ten cells forces it: three predicted cells, two full resampling passes
        # over them, then a partial pass for the remaining one.
        #
        # The peak list is truncated the way a manager truncates it -- ten lines for cubic,
        # twenty for the rest. The resampling branch requires that; the branch above happens to
        # tolerate a longer list, so the two disagree about their contract.
        rng = np.random.default_rng(12345)
        result = opt.wrapper.abnn_generator[sg].generate(
            10,
            rng,
            q2_obs[: opt.n_peaks],
            top_n=3,
            batch_size=2,
        )
        np.save(EXPECTED_DIR / f"abnn_resampled_{bl}.npy", result)

        print(f"  {bl}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI expected files (very slow, requires models)
# ---------------------------------------------------------------------------


def generate_cli_expected(test_metadata):
    import tempfile
    import os

    row = test_metadata[test_metadata["bravais lattice"] == "aP"].iloc[0]
    q2_obs, unit_cell, wavelength, bl, lattice_system = load_test_case(row)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        peak_file = tmp_path / "aP_q2.npy"
        np.save(peak_file, q2_obs)

        print("Generating run_analytical expected output ...")
        analytic_out = tmp_path / "analytic_results.json"
        cmd = [
            sys.executable,
            "-m",
            "mlindex.command_line.run_analytical",
            "--peak-file",
            str(peak_file),
            "--peak-units",
            "q2",
            "--bravais-lattices",
            "aP",
            "--seed",
            "12345",
            "--output-file",
            str(analytic_out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"run_analytical failed:\n{r.stderr}")
        else:
            import shutil

            shutil.copy(analytic_out, EXPECTED_DIR / "run_analytical_aP.json")
            print("  Done.")

        print("Generating run ML expected output (this may take several minutes) ...")
        cmd = [
            sys.executable,
            "-m",
            "mlindex.command_line.run",
            "--peak-file",
            str(peak_file),
            "--peak-units",
            "q2",
            "--bravais-lattices",
            "aP",
            "--nproc",
            "1",
            "--seed",
            "12345",
        ]
        # Pin the subprocess to the same tree, so both halves of this script and the
        # tests agree: the expected file is versioned with these models.
        env = {**os.environ, "MLINDEX_MODELS_DIR": str(resolve_models_dir())}
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, env=env)
        if r.returncode != 0:
            print(f"run ML failed:\n{r.stderr}")
        else:
            import shutil

            shutil.copy(
                tmp_path / "indexing_results.json", EXPECTED_DIR / "run_ml_aP.json"
            )
            print("  Done.")


def generate_ranker_expected(pool, fit_dir, group_frequency, bundle="b1_error1_cont1",
                             cut=3.5):
    """tests/expected/ranker_agreement.npz: one pattern per true lattice of a stored benchmark
    pool, every candidate the pool keeps at `cut`, as the research export computes its inputs.

    Holds what the indexer has for each candidate (cell, extinction group, M20, n_indexed, the
    pattern's peaks) beside the export's sixteen inputs and the fit's raw and calibrated scores,
    so the shipped feature builder and scorer can be checked against both.
    """
    from mlindex.model_training import Benchmark
    from mlindex.model_training.FomCombiner import FEATURES, FomCombiner, export_bundle
    from mlindex.utilities.Ranker import read_group_frequency

    entries = Benchmark.load_entries(pool, columns=["entry_id", "condition_bundle",
                                                    "bravais_lattice_true", "q2_obs"])
    entries = entries.loc[entries["condition_bundle"] == bundle]
    chosen = entries.sort_values("entry_id").drop_duplicates("bravais_lattice_true")
    ids = set(chosen["entry_id"])
    _, evaluation = export_bundle(
        pool, bundle, set(), ids, seeds=(), cut=cut, n_top=20, keep_all_depths=True, top_k=200,
        negative_rate=0.05, n_negatives=40, group_frequency=read_group_frequency(group_frequency))
    key = list(Benchmark.CANDIDATE_KEY)
    shards = pd.concat([
        pd.read_parquet(path, columns=key + ["lattice_system", "xnn", "n_peaks"])
        for _, path in Benchmark.candidate_shards(pool, bundle)])
    rows = evaluation.merge(shards, on=key, how="left", validate="1:1").sort_values(key)
    rows = rows.reset_index(drop=True)
    combiner = FomCombiner.load(fit_dir)
    pattern = rows["entry_id"].map({e: i for i, e in enumerate(chosen["entry_id"])}).to_numpy()
    q2_obs = np.full((len(chosen), 20), np.nan)
    for i, q2 in enumerate(chosen["q2_obs"]):
        q2_obs[i, :len(q2)] = q2
    np.savez_compressed(
        EXPECTED_DIR / "ranker_agreement.npz",
        pattern=pattern, q2_obs=q2_obs,
        bravais_lattice=rows["bravais_lattice"].to_numpy(dtype=str),
        lattice_system=rows["lattice_system"].to_numpy(dtype=str),
        spacegroup=rows["spacegroup"].to_numpy(dtype=str),
        # A lattice system's xnn has as many components as it has free parameters; padded.
        xnn=np.stack([np.pad(np.asarray(x, dtype=np.float64), (0, 6 - len(x)),
                             constant_values=np.nan) for x in rows["xnn"]]),
        n_peaks=rows["n_peaks"].to_numpy(dtype=np.int64),
        M20=rows["M20"].to_numpy(dtype=np.float64),
        n_indexed=rows["n_indexed"].to_numpy(dtype=np.float64),
        features=np.array(FEATURES),
        inputs=np.stack([rows[name].to_numpy(dtype=np.float32) if name != "bravais_lattice"
                         else np.full(rows.shape[0], np.nan, dtype=np.float32)
                         for name in FEATURES], axis=1),
        raw=combiner.raw_score(rows), calibrated=combiner.score(rows))
    print(f"  ranker_agreement.npz: {rows.shape[0]} candidates, {len(chosen)} patterns")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expected test outputs")
    parser.add_argument(
        "--models",
        action="store_true",
        help="Also generate model training expected files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also generate CLI expected files (implies --models)",
    )
    parser.add_argument(
        "--ranker",
        nargs=3,
        metavar=("POOL", "FIT_DIR", "GROUP_FREQUENCY_CSV"),
        help="Generate only the ranker agreement fixture, from a benchmark pool and a ranker fit",
    )
    args = parser.parse_args()

    if args.ranker:
        generate_ranker_expected(*args.ranker)
        sys.exit(0)

    test_metadata_all = pd.read_csv(TEST_DATA_DIR / "gsasII_tutorials.csv")
    # Use one row per BL for expected files (avoid name collisions from duplicate BL)
    test_metadata = test_metadata_all.drop_duplicates(
        subset=["bravais lattice"], keep="last"
    )

    generate_spacegroup_hkl_ref_expected()
    generate_utility_expected(test_metadata)
    generate_pkslst_expected(test_metadata_all)
    generate_reindexing_expected(test_metadata)

    if args.models or args.all:
        generate_model_training_expected(test_metadata)

    if args.all:
        generate_cli_expected(test_metadata_all)
