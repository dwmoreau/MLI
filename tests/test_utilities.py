import json
import numpy as np
import pytest
from pathlib import Path

from conftest import load_test_case, BL_TO_LATTICE_SYSTEM, _TEST_DATA_DIR

from mlindex.utilities.UnitCellTools import (
    get_xnn_from_unit_cell,
    get_unit_cell_from_xnn,
    get_unit_cell_volume,
    get_hkl_matrix,
    get_partial_unit_cell,
    fix_unphysical,
)


def _xnn(unit_cell, lattice_system):
    """Convert full 6-component unit cell to partial xnn for the given lattice system."""
    uc_p = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
    return get_xnn_from_unit_cell(
        uc_p[np.newaxis], partial_unit_cell=True, lattice_system=lattice_system
    )


from mlindex.utilities.FigureOfMerits import (
    get_M20_from_xnn,
    get_M20_likelihood_from_xnn,
)
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.gsas import load_pkslst

EXPECTED_DIR = Path(__file__).parent / "expected"


# ---------------------------------------------------------------------------
# Helpers / parametrize IDs
# ---------------------------------------------------------------------------


def _cases(test_metadata):
    return [load_test_case(row) for _, row in test_metadata.iterrows()]


def _bl_ids(test_metadata):
    return list(test_metadata["bravais lattice"])


# ---------------------------------------------------------------------------
# xnn <-> unit_cell roundtrip
# ---------------------------------------------------------------------------


def test_xnn_roundtrip(test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(test_metadata):
        uc2d = unit_cell[np.newaxis]
        xnn = get_xnn_from_unit_cell(uc2d, lattice_system=lattice_system)
        recovered = get_unit_cell_from_xnn(xnn, lattice_system=lattice_system)[0]
        np.testing.assert_allclose(
            recovered, unit_cell, rtol=1e-10, err_msg=f"roundtrip failed for {bl}"
        )


def test_xnn_roundtrip_random_triclinic():
    rng = np.random.default_rng(12345)
    n = 100
    xnn = rng.uniform(0, 1, size=(n, 6))
    xnn = fix_unphysical(xnn=xnn, rng=rng, lattice_system="triclinic")
    uc = get_unit_cell_from_xnn(xnn, lattice_system="triclinic")
    xnn2 = get_xnn_from_unit_cell(uc, lattice_system="triclinic")
    np.testing.assert_allclose(xnn2, xnn, rtol=1e-10)


def test_xnn_roundtrip_grid_search():
    a_vals = np.linspace(2, 20, 5)
    b_vals = np.linspace(2, 20, 5)
    c_vals = np.linspace(2, 20, 5)
    angles_deg = [70.0, 90.0, 110.0]
    angles_rad = np.array(angles_deg) * np.pi / 180

    unit_cells = []
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                for alpha in angles_rad:
                    for beta in angles_rad:
                        for gamma in angles_rad:
                            uc = np.array([a, b, c, alpha, beta, gamma])
                            uc2d = uc[np.newaxis]
                            xnn = get_xnn_from_unit_cell(
                                uc2d, lattice_system="triclinic"
                            )
                            if not np.any(np.isnan(xnn)):
                                unit_cells.append((uc, xnn))

    for uc, xnn in unit_cells:
        recovered = get_unit_cell_from_xnn(xnn, lattice_system="triclinic")[0]
        np.testing.assert_allclose(recovered, uc, rtol=1e-10)


# ---------------------------------------------------------------------------
# Unit cell volume
# ---------------------------------------------------------------------------


def test_unit_cell_volume(unique_test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        computed = get_unit_cell_volume(unit_cell[np.newaxis])[0]
        expected = np.load(EXPECTED_DIR / f"unit_cell_volume_{bl}.npy")
        np.testing.assert_allclose(
            computed, expected, rtol=1e-10, err_msg=f"volume mismatch for {bl}"
        )


# ---------------------------------------------------------------------------
# get_hkl_matrix
# ---------------------------------------------------------------------------


def test_get_hkl_matrix(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        result = get_hkl_matrix(hkl_ref, lattice_system)
        fname = expected_dir / f"hkl_matrix_{bl}.npy"
        np.testing.assert_array_equal(
            result, np.load(fname), err_msg=f"hkl_matrix mismatch for {bl}"
        )


# ---------------------------------------------------------------------------
# Q2Calculator
# ---------------------------------------------------------------------------


def test_q2_calculator(unique_test_metadata):
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(unique_test_metadata):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        expected = np.load(EXPECTED_DIR / f"q2_calculator_{bl}.npy")
        np.testing.assert_allclose(
            q2_calc, expected, rtol=1e-10, err_msg=f"q2_calculator mismatch for {bl}"
        )


# ---------------------------------------------------------------------------
# fast_assign
# ---------------------------------------------------------------------------


def test_fast_assign(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_ref = np.sum(hkl2 * xnn, axis=1)[np.newaxis]
        result = fast_assign(q2_obs.astype(np.float64), q2_ref.astype(np.float64))
        fname = expected_dir / f"fast_assign_{bl}.npy"
        np.testing.assert_array_equal(
            result, np.load(fname), err_msg=f"fast_assign mismatch for {bl}"
        )


def _fast_assign_reference(q2_obs, q2_ref):
    """Straightforward nearest-reference scan, as fast_assign was first written.

    fast_assign is now unrolled into four interleaved accumulators for speed, so
    the two behaviours its callers depend on are no longer obvious from reading
    it. This spells them out: the lowest reference index wins a tie, and a peak
    with nothing within 100.0 is assigned index 0.
    """
    hkl_assign = np.zeros((q2_ref.shape[0], q2_obs.size), dtype=np.uint16)
    for candidate_index in range(q2_ref.shape[0]):
        for obs_index in range(q2_obs.size):
            current_min = 100.0
            for ref_index in range(q2_ref.shape[1]):
                diff = abs(q2_obs[obs_index] - q2_ref[candidate_index, ref_index])
                if diff < current_min:
                    current_min = diff
                    hkl_assign[candidate_index, obs_index] = ref_index
    return hkl_assign


def test_fast_assign_breaks_ties_on_lowest_index():
    # 1.0 sits exactly between 0.5 and 1.5, and is equidistant from the pair at
    # indices 0 and 3, so only the lowest-index rule fixes the answer.
    q2_obs = np.array([1.0])
    q2_ref = np.array([[0.5, 2.0, 2.0, 1.5]])
    result = fast_assign(q2_obs, q2_ref)
    assert result[0, 0] == 0
    np.testing.assert_array_equal(result, _fast_assign_reference(q2_obs, q2_ref))


def test_fast_assign_returns_zero_when_nothing_is_within_the_bound():
    q2_obs = np.array([0.0])
    q2_ref = np.array([[500.0, 300.0, 400.0]])
    result = fast_assign(q2_obs, q2_ref)
    assert result[0, 0] == 0
    np.testing.assert_array_equal(result, _fast_assign_reference(q2_obs, q2_ref))


@pytest.mark.parametrize(
    "description, mutate",
    [
        ("all-NaN row", lambda a: a.__setitem__((1, slice(None)), np.nan)),
        ("single NaN", lambda a: a.__setitem__((1, 3), np.nan)),
        ("partial NaN row", lambda a: a.__setitem__((1, slice(0, 5)), np.nan)),
        ("all-Inf row", lambda a: a.__setitem__((1, slice(None)), np.inf)),
        ("single -Inf", lambda a: a.__setitem__((1, 2), -np.inf)),
        ("everything beyond the 100.0 bound", lambda a: a.__setitem__((1, slice(None)), 1e9)),
    ],
)
def test_fast_assign_is_well_defined_on_degenerate_input(description, mutate):
    """Degenerate q2_ref must give a defined, in-range answer -- never a crash.

    q2_ref is xnn @ hkl2.T and NaN xnn genuinely occurs (the Selling reduction
    produces it, which is why _downsample_computation filters for it), so this
    is a reachable input rather than a hypothetical. fast_assign deliberately
    carries no fastmath for this reason: under fastmath the compiler may assume
    NaN never appears and the result on these rows becomes whatever it happens
    to emit.
    """
    rng = np.random.default_rng(4)
    q2_obs = np.sort(rng.uniform(0.05, 3.0, size=6))
    q2_ref = rng.uniform(0.05, 3.0, size=(3, 12))
    mutate(q2_ref)

    result = fast_assign(q2_obs, q2_ref)

    assert result.shape == (3, 6)
    # Every entry must be usable as an index into hkl_ref, or the caller's
    # np.take would read out of bounds.
    assert np.all(result < q2_ref.shape[1]), description
    # Rows that were left alone must be unaffected by a neighbour's degeneracy.
    np.testing.assert_array_equal(
        result[[0, 2]], _fast_assign_reference(q2_obs, q2_ref)[[0, 2]]
    )
    # And the degenerate row itself must match the plain-Python semantics.
    np.testing.assert_array_equal(
        result, _fast_assign_reference(q2_obs, q2_ref), err_msg=description
    )


@pytest.mark.parametrize("n_ref", [1, 2, 3, 4, 5, 7, 8, 9, 17, 63, 64, 65])
def test_fast_assign_matches_reference_for_any_row_length(n_ref):
    # The unrolled loop handles four indices per step and finishes the remainder
    # separately, so lengths either side of a multiple of four are the ones that
    # would expose an off-by-one in the tail.
    rng = np.random.default_rng(20240814 + n_ref)
    q2_obs = np.sort(rng.uniform(0.05, 3.0, size=8))
    # Coarse rounding makes exact ties common, which is what pins tie-breaking.
    q2_ref = np.round(rng.uniform(0.0, 3.0, size=(11, n_ref)), 2)
    np.testing.assert_array_equal(
        fast_assign(q2_obs, q2_ref), _fast_assign_reference(q2_obs, q2_ref)
    )


# ---------------------------------------------------------------------------
# Figure of merits
# ---------------------------------------------------------------------------


def test_m20(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        hkl_assign = fast_assign(
            q2_obs.astype(np.float64), q2_calc[np.newaxis].astype(np.float64)
        )[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]
        result = get_M20_from_xnn(q2_obs, xnn, hkl_assigned, hkl_ref, lattice_system)
        expected = np.load(expected_dir / f"m20_{bl}.npy")
        np.testing.assert_allclose(
            result, expected, rtol=1e-10, err_msg=f"M20 mismatch for {bl}"
        )


def test_m20_likelihood(unique_test_metadata):
    expected_dir = EXPECTED_DIR
    for q2_obs, unit_cell, wavelength, bl, lattice_system in _cases(
        unique_test_metadata
    ):
        hkl_ref = np.load(_TEST_DATA_DIR.parent / f"hkl_ref_{bl}.npy")
        xnn = _xnn(unit_cell, lattice_system)
        hkl2 = get_hkl_matrix(hkl_ref, lattice_system)
        q2_calc = np.sum(hkl2 * xnn, axis=1)
        hkl_assign = fast_assign(
            q2_obs.astype(np.float64), q2_calc[np.newaxis].astype(np.float64)
        )[0]
        hkl_assigned = hkl_ref[hkl_assign][np.newaxis]
        log_lik, prob, M = get_M20_likelihood_from_xnn(
            q2_obs, xnn, hkl_assigned, lattice_system, bl
        )
        expected = np.load(expected_dir / f"m20_likelihood_{bl}.npy")
        np.testing.assert_allclose(
            np.stack([log_lik, M.flatten()]),
            expected,
            rtol=1e-10,
            err_msg=f"M20_likelihood mismatch for {bl}",
        )


# ---------------------------------------------------------------------------
# Peak loading / unit conversion
# ---------------------------------------------------------------------------


def test_load_pkslst(test_metadata):
    for _, row in test_metadata.iterrows():
        peak_file = str(row["peak list file"])
        if not peak_file.endswith(".pkslst"):
            continue
        case_name = peak_file.replace(".pkslst", "")
        wavelength = float(row["wavelength"])
        pkslst_path = _TEST_DATA_DIR / case_name / peak_file

        result = load_pkslst(str(pkslst_path), wavelength)
        expected = np.load(EXPECTED_DIR / f"load_pkslst_{case_name}.npy")
        np.testing.assert_allclose(
            np.sort(result),
            expected,
            rtol=1e-12,
            err_msg=f"load_pkslst mismatch for {case_name}",
        )



# ---------------------------------------------------------------------------
# SpaceGroups: get_spacegroup_hkl_ref
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


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64])
def test_spacegroup_hkl_ref_dtypes(dtype):
    # The production reference lists are float64 under mlindex/models and int64 under
    # mlindex/data, and Boost.Python accepts numpy int32 but rejects both of those. An int32-only
    # fixture therefore passes while every real run raises, which is how a broken systematic
    # absence check survived in assign_extinction_group. Pin all three dtypes.
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    hkl = _SPACEGROUP_HKL_REF_TEST_HKL.astype(dtype)
    for bl in BL_TO_LATTICE_SYSTEM:
        result = get_spacegroup_hkl_ref(hkl, bl)
        reference = get_spacegroup_hkl_ref(_SPACEGROUP_HKL_REF_TEST_HKL, bl)
        assert list(result.keys()) == list(reference.keys()), f"{bl}: key mismatch"
        for key in reference:
            np.testing.assert_array_equal(
                np.asarray(result[key], dtype=np.int64), reference[key].astype(np.int64),
                err_msg=f"{bl} [{key}]: {dtype} disagrees with int32",
            )


def test_spacegroup_hkl_ref():
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    for bl in BL_TO_LATTICE_SYSTEM:
        result = get_spacegroup_hkl_ref(_SPACEGROUP_HKL_REF_TEST_HKL, bl)
        with open(EXPECTED_DIR / f"spacegroup_hkl_ref_{bl}_keys.json") as f:
            expected_keys = json.load(f)
        assert list(result.keys()) == expected_keys, f"{bl}: key mismatch"
        for idx, key in enumerate(expected_keys):
            expected = np.load(EXPECTED_DIR / f"spacegroup_hkl_ref_{bl}_{idx}.npy")
            np.testing.assert_array_equal(
                result[key], expected, err_msg=f"{bl} [{key}]: hkl array mismatch"
            )


def test_load_peaks_npy(test_metadata):
    from mlindex.command_line.run import _load_peaks
    import types

    for _, row in test_metadata.iterrows():
        peak_file = str(row["peak list file"])
        if not (peak_file.endswith(".pkslst") or peak_file.endswith(".pklst")):
            continue
        bl = str(row["bravais lattice"])
        case_name = peak_file.replace(".pkslst", "").replace(".pklst", "")
        npy_path = _TEST_DATA_DIR / case_name / f"{case_name}_peak_list.npy"
        args = types.SimpleNamespace(
            peaks=None,
            peak_file=str(npy_path),
            peak_units="q2",
            wavelength=None,
            zero_error=False,
        )
        result = _load_peaks(args)
        expected = np.sort(np.load(npy_path))[:20]
        assert len(result) <= 20, f"{bl}: more than 20 peaks loaded"
        assert result[0] <= result[-1], f"{bl}: peaks not sorted"
        np.testing.assert_array_equal(
            result, expected, err_msg=f"load_peaks_npy mismatch for {bl}"
        )


def test_every_bravais_lattice_has_a_lattice_system():
    """The two constants describe the same fourteen lattices.

    They are read by the CLI, the multiprocessing planner, the peak-list builder and the test
    fixtures. A lattice present in one and missing from the other raises `KeyError` deep in a
    run rather than here.
    """
    from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES, BL_TO_LATTICE_SYSTEM

    assert len(BRAVAIS_LATTICES) == 14
    assert len(set(BRAVAIS_LATTICES)) == 14
    assert set(BRAVAIS_LATTICES) == set(BL_TO_LATTICE_SYSTEM)


def test_partial_unit_cell_agrees_whether_asked_by_lattice_or_by_system():
    """`get_partial_unit_cell` accepts either key and must slice identically for both.

    The two used to be separate if/elif chains, so a change to one could miss the other. They now
    share `BL_TO_LATTICE_SYSTEM`, and this is what fails if they are ever split again.
    """
    from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES, BL_TO_LATTICE_SYSTEM

    unit_cell = np.array([5.1, 6.2, 7.3, 1.1, 1.2, 1.3])
    for bravais_lattice in BRAVAIS_LATTICES:
        by_lattice = get_partial_unit_cell(unit_cell, bravais_lattice=bravais_lattice)
        by_system = get_partial_unit_cell(
            unit_cell, lattice_system=BL_TO_LATTICE_SYSTEM[bravais_lattice]
        )
        np.testing.assert_array_equal(
            by_lattice, by_system, err_msg=f"{bravais_lattice} slices differently by system"
        )


def test_partial_unit_cell_keeps_the_angle_each_system_actually_has():
    """Monoclinic keeps beta and drops alpha; rhombohedral keeps alpha and drops c.

    Getting either wrong compares the wrong angle and mislabels a candidate silently, which is
    why the free-parameter indices are asserted and not only used.
    """
    unit_cell = np.array([5.0, 6.0, 7.0, 1.1, 1.2, 1.3])
    expected = {
        "cubic": [5.0],
        "tetragonal": [5.0, 7.0],
        "hexagonal": [5.0, 7.0],
        "rhombohedral": [5.0, 1.1],
        "orthorhombic": [5.0, 6.0, 7.0],
        "monoclinic": [5.0, 6.0, 7.0, 1.2],
        "triclinic": [5.0, 6.0, 7.0, 1.1, 1.2, 1.3],
    }
    for lattice_system, want in expected.items():
        got = get_partial_unit_cell(unit_cell, lattice_system=lattice_system)
        np.testing.assert_array_equal(got, np.array(want), err_msg=lattice_system)


def test_xnn_axis_multipliers_describe_a_real_axis_rescaling():
    """The multipliers must be what actually happens to the metric when the axes are scaled.

    Reproducing the grid the optimizer used to build by hand is not enough: that grid could have
    been wrong. This scales a cell's axes directly, converts, and checks the metric agrees --
    which is what makes the square roots on the cross terms testable rather than folklore.
    """
    from mlindex.utilities.UnitCellTools import n_axis_factors, xnn_axis_multipliers

    cells = {
        'orthorhombic': np.array([5.0, 7.0, 11.0]),
        'monoclinic': np.array([5.0, 7.0, 11.0, np.deg2rad(103.0)]),
        'triclinic': np.array([5.0, 7.0, 11.0, np.deg2rad(88.0), np.deg2rad(103.0),
                               np.deg2rad(95.0)]),
        'tetragonal': np.array([5.0, 11.0]),
        'cubic': np.array([5.0]),
    }
    scalings = {
        'orthorhombic': np.array([2.0, 0.5, 3.0]),
        'monoclinic': np.array([2.0, 0.5, 3.0]),
        'triclinic': np.array([2.0, 0.5, 3.0]),
        'tetragonal': np.array([2.0, 0.5]),
        'cubic': np.array([2.0]),
    }

    for lattice_system, cell in cells.items():
        scale = scalings[lattice_system]
        assert scale.size == n_axis_factors(lattice_system)

        xnn = get_xnn_from_unit_cell(cell[np.newaxis], partial_unit_cell=True,
                                     lattice_system=lattice_system)

        # Scale the direct axis lengths, leaving the angles alone.
        scaled_cell = cell.copy()
        n_lengths = 1 if lattice_system in ('cubic', 'rhombohedral') else (
            2 if lattice_system in ('tetragonal', 'hexagonal') else 3)
        scaled_cell[:n_lengths] = cell[:n_lengths] * scale[:n_lengths]
        expected = get_xnn_from_unit_cell(scaled_cell[np.newaxis], partial_unit_cell=True,
                                          lattice_system=lattice_system)

        # A direct axis scaled by s scales the reciprocal axis by 1/s.
        multipliers = xnn_axis_multipliers(1.0 / scale, lattice_system)
        np.testing.assert_allclose(multipliers**2 * xnn, expected, rtol=1e-12,
                                   err_msg=lattice_system)


def test_the_identity_scaling_leaves_the_metric_alone():
    """Row zero of the optimizer's grid is the identity, and its acceptance test depends on it."""
    from mlindex.utilities.UnitCellTools import XNN_AXIS_PAIRS
    from mlindex.utilities.UnitCellTools import n_axis_factors, xnn_axis_multipliers

    for lattice_system, pairs in XNN_AXIS_PAIRS.items():
        ones = np.ones(n_axis_factors(lattice_system))
        np.testing.assert_array_equal(
            xnn_axis_multipliers(ones, lattice_system), np.ones(len(pairs)),
            err_msg=lattice_system)


def test_the_xnn_component_map_matches_the_hkl_design_matrix():
    """`q2` is `get_hkl_matrix(hkl) @ xnn`, so the two must agree on how many components a lattice
    system has and in what order. A mismatch multiplies the wrong coefficient by the wrong axis."""
    from mlindex.utilities.UnitCellTools import XNN_AXIS_PAIRS

    hkl = np.array([[1, 2, 3], [2, 0, 1]])
    for lattice_system, pairs in XNN_AXIS_PAIRS.items():
        design = get_hkl_matrix(hkl, lattice_system)
        assert design.shape[-1] == len(pairs), lattice_system


# ---------------------------------------------------------------------------
# Peak-list digests
# ---------------------------------------------------------------------------


def test_peak_list_bytes_are_the_same_for_any_memory_layout():
    """The search keys its generator on these bytes and the benchmark joins shards on their
    digest. Both are compared across machines and across processes, so the bytes must depend on
    the values alone -- not on dtype, stride or byte order."""
    from mlindex.utilities.Digests import peak_list_bytes

    values = [0.05, 0.1, 0.15, 0.2]
    contiguous = np.array(values, dtype=np.float64)
    strided = np.array([[v, np.nan] for v in values], dtype=np.float64)[:, 0]
    big_endian = np.array(values, dtype='>f8')

    assert peak_list_bytes(strided) == peak_list_bytes(contiguous)
    assert peak_list_bytes(big_endian) == peak_list_bytes(contiguous)
    assert peak_list_bytes(values) == peak_list_bytes(contiguous)


def test_q2_digest_is_stable_across_processes():
    """`hash()` is salted per process, so a digest built on it would differ on every run and a
    mis-joined shard would look like a fresh pattern. This value is pinned."""
    from mlindex.utilities.Digests import q2_digest

    digest = q2_digest(np.array([0.05, 0.1, 0.15, 0.2]))
    assert digest == '097ad568deb92441'
    assert len(digest) == 16


def test_q2_digest_separates_peak_lists_that_differ_in_one_line():
    from mlindex.utilities.Digests import q2_digest

    base = np.array([0.05, 0.1, 0.15, 0.2])
    moved = base.copy()
    moved[2] += 1e-9
    assert q2_digest(base) != q2_digest(moved)
