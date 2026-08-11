"""Validate the figure-of-merit implementations against the published numeric tables.

Run in both environments -- development first, then the runtime one, before anything is called
done:

    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python -m pytest tests/ -v
    /global/cfs/cdirs/m4064/dwmoreau/envs/onnx/bin/python -m pytest tests/ -v

Fixtures and their provenance are in fixtures_fom_literature.py. Each test states its tolerance
and why: several of the papers round an *input* harder than the output, so a five-per-cent target
is not always attainable from the printed values.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures_fom_literature as fixtures  # noqa: E402
from mlindex.utilities.FigureOfMerits import DEWOLFF61_COEFFICIENTS  # noqa: E402
from mlindex.utilities.FigureOfMerits import WU88_M20_RATIO  # noqa: E402
from mlindex.utilities.FigureOfMerits import WU88_SYMMETRY_FACTOR  # noqa: E402
from mlindex.utilities.FigureOfMerits import WU88_SYMMETRY_FACTOR_CORRECTED  # noqa: E402
from mlindex.utilities.FigureOfMerits import compute_all  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_M20  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_M_nn  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_delta_dewolff61  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_F_N  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_g_min_werner  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_hkl_multiplicity  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_laue_operations  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_n_dewolff61  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_V_over_Vcrit  # noqa: E402
from mlindex.utilities.UnitCellTools import get_hkl_matrix  # noqa: E402


# ================================================================================================
# de Wolff 1961 -- the analytic null
# ================================================================================================


def test_dewolff61_leading_terms_are_analytic():
    """C0 is half the reciprocal-lattice points in a sphere, divided down by the Laue order."""
    assert DEWOLFF61_COEFFICIENTS["aP"][0] == pytest.approx(2*np.pi/3, rel=1e-3)
    assert DEWOLFF61_COEFFICIENTS["mP"][0] == pytest.approx(np.pi/3, rel=1e-3)
    assert DEWOLFF61_COEFFICIENTS["oP"][0] == pytest.approx(np.pi/6, rel=1e-3)
    # de Wolff derives the monoclinic surface term as C2 = pi/4.
    assert DEWOLFF61_COEFFICIENTS["mP"][2] == pytest.approx(np.pi/4, rel=2e-3)
    # ...and the orthorhombic ones as pi/8.
    for index in (1, 2, 3):
        assert DEWOLFF61_COEFFICIENTS["oP"][index] == pytest.approx(np.pi/8, rel=2e-3)


def test_dewolff61_table1_transcribed_faithfully():
    """The primitive rows must equal the printed table, and centring must halve or quarter them."""
    assert DEWOLFF61_COEFFICIENTS["aP"] == fixtures.DEWOLFF61_TABLE1["triclinic"][1:]
    assert DEWOLFF61_COEFFICIENTS["mP"] == fixtures.DEWOLFF61_TABLE1["monoclinic_P"][1:]
    assert DEWOLFF61_COEFFICIENTS["oP"] == fixtures.DEWOLFF61_TABLE1["orthorhombic_P"][1:]
    assert DEWOLFF61_COEFFICIENTS["tP"] == fixtures.DEWOLFF61_TABLE1["tetragonal_P"][1:]
    assert DEWOLFF61_COEFFICIENTS["hP"] == fixtures.DEWOLFF61_TABLE1["hexagonal"][1:]
    assert DEWOLFF61_COEFFICIENTS["hR"] == fixtures.DEWOLFF61_TABLE1["rhombohedral_hex_axes"][1:]
    for centred, primitive, factor in [
        ("mC", "mP", 0.5), ("oC", "oP", 0.5), ("oI", "oP", 0.5), ("oF", "oP", 0.25),
        ("tI", "tP", 0.5),
    ]:
        for got, expected in zip(
            DEWOLFF61_COEFFICIENTS[centred], DEWOLFF61_COEFFICIENTS[primitive]
        ):
            assert got == pytest.approx(factor*expected, abs=1e-6)


def test_dewolff61_cubic_row_matches_enumeration():
    """de Wolff tabulates no cubic row; ours is derived, so it has to be checked against a count.

    In cubic the lines collapse onto integer values of h^2+k^2+l^2, so N is linear in Q and the
    leading term vanishes. The remaining coefficient is the density of attainable integers:
    5/6 for P, 11/24 for I, 1/3 for F.
    """
    limit = 60
    grid = np.arange(-limit, limit + 1)
    h, k, l = (axis.ravel() for axis in np.meshgrid(grid, grid, grid, indexing="ij"))
    squared = h*h + k*k + l*l
    rules = {
        "cP": np.ones(len(h), dtype=bool),
        "cI": (h + k + l) % 2 == 0,
        "cF": ((h + k) % 2 == 0) & ((h + l) % 2 == 0),
    }
    for bravais_lattice, mask in rules.items():
        assert DEWOLFF61_COEFFICIENTS[bravais_lattice][0] == 0.0
        attainable = np.unique(squared[mask])
        p = limit*limit
        attainable = attainable[(attainable > 0) & (attainable <= p)]
        density = len(attainable)/p
        assert DEWOLFF61_COEFFICIENTS[bravais_lattice][1] == pytest.approx(density, rel=0.01)


def test_dewolff61_sharan_line_count():
    """Section 5, Sharan's aluminium orthoarsenate: N = 117 at Q = 2000, 638 at Q = 6850.

    Asserted against the *formula*, not against the actual count of 115 at Q = 2000 -- de Wolff
    states the ~2% gap as the accuracy of equation (1).
    """
    q2 = np.array([q for q, _ in fixtures.DEWOLFF61_SHARAN_N])
    expected = np.array([n for _, n in fixtures.DEWOLFF61_SHARAN_N])
    got = get_n_dewolff61(q2, fixtures.DEWOLFF61_SHARAN_XNN, "orthorhombic", "oP")[0]
    # The printed coefficients carry two significant figures, so 1% is the achievable tolerance.
    assert got == pytest.approx(expected, rel=0.01)

    # And the coefficients themselves, which is what the units convention hangs on.
    reciprocal_axes = np.sqrt(fixtures.DEWOLFF61_SHARAN_XNN[0])
    reciprocal_volume = np.prod(reciprocal_axes)
    c0, c1, c2, c3 = DEWOLFF61_COEFFICIENTS["oP"]
    leading, surface = fixtures.DEWOLFF61_SHARAN_N_COEFFICIENTS
    assert c0/reciprocal_volume == pytest.approx(leading, rel=0.01)
    assert (c1*reciprocal_axes[0] + c2*reciprocal_axes[1] + c3*reciprocal_axes[2])/(
        reciprocal_volume
    ) == pytest.approx(surface, rel=0.01)


def test_dewolff61_sharan_expected_discrepancy():
    """Table 4's five Delta values, which pin down the factor of two (F-024).

    Section 5 prints Delta = 1/(0.00138 sqrt(Q) + 0.0173). That expression yields exactly twice
    each tabulated value -- it is the mean interval 2*Delta. Both are asserted so that a future
    change in either direction fails loudly.
    """
    q2 = np.array([q for q, _ in fixtures.DEWOLFF61_SHARAN_DELTA])
    expected = np.array([d for _, d in fixtures.DEWOLFF61_SHARAN_DELTA])
    got = get_delta_dewolff61(q2, fixtures.DEWOLFF61_SHARAN_XNN, "orthorhombic", "oP")[0]
    # The table rounds to two significant figures; 2% covers it.
    assert got == pytest.approx(expected, rel=0.02)

    printed_section5 = 1/(0.00138*np.sqrt(q2) + 0.0173)
    assert printed_section5 == pytest.approx(2*expected, rel=0.02)


def test_dewolff61_exponential_interval_assumption():
    """Table 3: 214 intervals of a 2-D anorthic net against 214 exp[-(x - 1/2)/18.8].

    de Wolff's own validation of the exponential null, and it also quantifies its known slight
    narrowing: real Q values are a little more regular than random, so the exponential is
    conservative. Assert the prediction reproduces his printed column, then check the direction of
    the residual over the body of the distribution.
    """
    two_delta = fixtures.DEWOLFF61_TABLE3_TWO_DELTA
    # Restricted to x <= 60. The two rows beyond that are not reliable: at x = 75 the extracted
    # prediction column reads 1 where the formula gives 4.1, while every row up to x = 60 agrees
    # within one count. A single mis-read digit in the tail of a 1961 table is far likelier than a
    # breakdown of the formula that its neighbours all satisfy, so those rows are not asserted on.
    for x, actual, printed in fixtures.DEWOLFF61_TABLE3:
        if x > 60:
            continue
        predicted = 214*np.exp(-(x - 0.5)/two_delta)
        assert predicted == pytest.approx(printed, abs=2.0), x
    # Over the body of the distribution the actual counts sit at or below the exponential -- the
    # narrowing de Wolff describes, and the reason the exponential null is mildly conservative.
    body = [(a, p) for x, a, p in fixtures.DEWOLFF61_TABLE3 if 30 <= x <= 60]
    assert sum(a for a, _ in body) <= sum(p for _, p in body)


# ================================================================================================
# de Wolff 1968 -- M20 itself
# ================================================================================================


def test_dewolff68_table2_m20_from_printed_inputs():
    """Given 10^4 Q20, 10^4 epsilon_bar and N20, M20 = Q20/(2 N20 epsilon_bar) must follow.

    This is the definition rather than our code path, but it is the check that catches a units
    error: the 10^4 factors cancel in the ratio, so it also confirms the fixture is internally
    consistent.

    Tolerance is 15% per row, with the *median* required under 5%. The mean discrepancy is printed
    to one decimal place, so rows quoting 0.5 or 1.9 carry up to 10% of rounding on the input
    alone; rows 1 and 6 are the two that need the loose bound.
    """
    errors = []
    for number, name, q20, epsilon, n20, _, m20, _ in fixtures.DEWOLFF68_TABLE2:
        computed = q20/(2*n20*epsilon)
        relative = abs(computed - m20)/m20
        errors.append(relative)
        assert relative < 0.15, f"row {number} ({name}): got {computed:.2f}, printed {m20}"
    assert np.median(errors) < 0.05


def _li6b4o9_cells():
    """The correct and incorrect Li6B4O9 cells, as xnn plus an allowed-hkl predicate.

    Both are built from the published cell constants rather than from the repo's reference lists,
    so this is an independent end-to-end check of the whole chain: cell -> xnn -> hkl matrix -> q2.
    Q is returned in the paper's units of 10^4 A^-2.
    """
    a, b, c, beta = fixtures.DEWOLFF68_LI6B4O9_CORRECT_CELL
    beta = np.radians(beta)
    # Monoclinic, b unique: a* = 1/(a sin beta), c* = 1/(c sin beta), cos beta* = -cos beta.
    a_star, b_star, c_star = 1/(a*np.sin(beta)), 1/b, 1/(c*np.sin(beta))
    correct_xnn = 1e4*np.array(
        [[a_star**2, b_star**2, c_star**2, 2*a_star*c_star*np.cos(np.pi - beta)]]
    )

    a, b, c = fixtures.DEWOLFF68_LI6B4O9_INCORRECT_CELL
    incorrect_xnn = 1e4*np.array([[1/a**2, 1/b**2, 1/c**2]])

    return (
        ("monoclinic", "mP", correct_xnn, lambda hkl: np.ones(len(hkl), dtype=bool)),
        # B-centred orthorhombic: h + l = 2n. Not one of our fourteen labels -- it is a
        # single-face centring, so it takes oC's de Wolff coefficients.
        ("orthorhombic", "oC", incorrect_xnn, lambda hkl: (hkl[:, 0] + hkl[:, 2]) % 2 == 0),
    )


def test_dewolff68_li6b4o9_published_q_calc_reproduced():
    """Both published indexings must reproduce their own printed 10^4 Q_calc column.

    If this fails, the cell conventions are wrong -- in particular the monoclinic b-unique setting
    and cos(beta*) = -cos(beta) -- and nothing downstream on this pair means anything.

    Tolerance 0.5% relative, not absolute. The cell constants are published to three or four
    significant figures (c = 3.32 A), and c* enters Q squared, so the l-bearing lines inherit about
    0.3% from the rounding of c alone: solving the 011 line backwards gives c = 3.327 rather than
    3.32. That is a limit of the printed data, not of the implementation, and it is why the M20
    test below uses de Wolff's own Q_calc column instead of recomputing it.
    """
    cells = _li6b4o9_cells()
    for column, (lattice_system, _, xnn, _) in zip((1, 3), cells):
        rows = [row for row in fixtures.DEWOLFF68_LI6B4O9 if row[column] is not None]
        hkl = np.array([row[column] for row in rows], dtype=float)
        printed = np.array([row[column + 1] for row in rows])
        computed = get_hkl_matrix(hkl, lattice_system) @ xnn[0]
        assert computed == pytest.approx(printed, rel=0.005), lattice_system


def _enumerate_reference(lattice_system, allowed, limit=14):
    """One representative per symmetry-distinct calculated line, for a self-contained N count."""
    grid = np.arange(-limit, limit + 1)
    hkl = np.stack(
        [axis.ravel() for axis in np.meshgrid(grid, grid, grid, indexing="ij")], axis=1
    )
    hkl = hkl[np.any(hkl != 0, axis=1)]
    hkl = hkl[allowed(hkl)]
    return np.unique(get_hkl_matrix(hkl.astype(float), lattice_system), axis=0)


def test_dewolff68_li6b4o9_m20_pair():
    """The famous counterexample: M20 must be ~5.3 correct against ~5.4 incorrect.

    de Wolff's whole point is that M20 does *not* separate these -- reproducing that failure is
    what makes the pair a fixture, and it is the S01 acceptance gate.

    Q_calc is taken from de Wolff's own printed column rather than recomputed, because his cell
    constants are rounded too hard to regenerate it (see the test above); N20 *is* computed here,
    by independent enumeration, and that is the part worth validating. It comes out at exactly his
    published 63 for the correct cell.

    Tolerance 15%: the printed Q_calc columns are themselves rounded (the incorrect one to whole
    units of 10^-4 A^-2), giving mean discrepancies of 2.34 and 2.93 against his stated 2.6 and
    3.0.
    """
    results, counts = {}, {}
    for label, (lattice_system, _, xnn, allowed) in zip(
        ("correct", "incorrect"), _li6b4o9_cells()
    ):
        column = 1 if label == "correct" else 3
        rows = [row for row in fixtures.DEWOLFF68_LI6B4O9 if row[column] is not None]
        q2_obs = np.array([row[0] for row in rows])
        q2_calc = np.array([row[column + 1] for row in rows])[np.newaxis]
        reference = _enumerate_reference(lattice_system, allowed)
        q2_ref = (reference @ xnn[0])[np.newaxis]
        counts[label] = int((q2_ref < q2_calc[0, -1]).sum())
        results[label] = get_M20(q2_obs, q2_calc, q2_ref.copy())[0]

    # de Wolff's Table 2 gives N20 = 63 for the correct indexing and 52 for the incorrect.
    assert counts["correct"] == pytest.approx(63, abs=3), counts
    assert counts["incorrect"] == pytest.approx(52, abs=6), counts

    expected = {"correct": 5.3, "incorrect": 5.4}
    for label, value in results.items():
        assert value == pytest.approx(expected[label], rel=0.15), f"{label}: {value:.2f}"
    # The property that makes this famous: M20 cannot tell them apart.
    assert abs(results["correct"] - results["incorrect"])/results["correct"] < 0.15


# ================================================================================================
# de Wolff 1972 -- how N20 is counted
# ================================================================================================


def test_dewolff72_counting_convention_is_self_consistent():
    """Khawas' error: counting only the calculated lines that matched an observation.

    The two published (N20, M20) pairs must be consistent with M20 = Q20/(2 N20 eps) at fixed
    Q20 and eps, i.e. the ratio of the merits must be the inverse ratio of the counts. This is a
    regression test on the definition, and it is the reason get_M20 counts every reference line
    below the cut-off rather than only the assigned ones.
    """
    khawas = fixtures.DEWOLFF72_KHAWAS
    merit_ratio = khawas["M20_reported"]/khawas["M20_correct"]
    count_ratio = khawas["N20_correct"]/khawas["N20_reported"]
    assert merit_ratio == pytest.approx(count_ratio, rel=0.05)


def test_get_M20_counts_unmatched_calculated_lines():
    """A reference line that explains no observation must still raise N and so lower M20."""
    q2_obs = np.linspace(0.1, 2.0, 20)
    q2_calc = q2_obs[np.newaxis] + 1e-4
    sparse = q2_obs[np.newaxis].copy()
    dense = np.concatenate([sparse, sparse + 0.05], axis=1)
    assert get_M20(q2_obs, q2_calc, dense.copy())[0] < get_M20(q2_obs, q2_calc, sparse.copy())[0]


# ================================================================================================
# Wu 1988
# ================================================================================================


def test_wu88_tables_transcribed():
    for system, value in fixtures.WU88_TABLE2_S.items():
        assert WU88_SYMMETRY_FACTOR[system] == pytest.approx(value)
    for system, value in fixtures.WU88_TABLE2_S_CORRECTED.items():
        assert WU88_SYMMETRY_FACTOR_CORRECTED[system] == pytest.approx(value)
    for system, ratios in fixtures.WU88_TABLE1_RATIO.items():
        assert WU88_M20_RATIO[system] == pytest.approx(np.mean(ratios), abs=0.03)
    # Rhombohedral is not a separate row in Wu's tables; we reuse hexagonal and say so.
    assert WU88_SYMMETRY_FACTOR["rhombohedral"] == WU88_SYMMETRY_FACTOR["hexagonal"]


def test_wu88_corrected_factor_is_S_over_ratio():
    """S' = S / (M20/M'20) -- the internal relation between his two tables."""
    for system in ("triclinic", "monoclinic", "orthorhombic", "tetragonal", "hexagonal", "cubic"):
        implied = fixtures.WU88_TABLE2_S[system]/np.mean(fixtures.WU88_TABLE1_RATIO[system])
        assert fixtures.WU88_TABLE2_S_CORRECTED[system] == pytest.approx(implied, rel=0.05)


# ================================================================================================
# Oishi-Tomiyasu 2013 and 2021
# ================================================================================================


@pytest.mark.parametrize("lattice_system", sorted(fixtures.LAUE_GROUP_ORDER))
def test_ot13_laue_group_orders(lattice_system):
    assert len(get_laue_operations(lattice_system)) == fixtures.LAUE_GROUP_ORDER[lattice_system]


@pytest.mark.parametrize("lattice_system", sorted(fixtures.OT13_TABLE1_MULTIPLICITY))
def test_ot13_table1_multiplicities(lattice_system):
    """Every multiplicity class printed in her Table 1, computed as a Laue-group orbit size."""
    cases = fixtures.OT13_TABLE1_MULTIPLICITY[lattice_system]
    hkl = np.array([case[0] for case in cases])
    expected = np.array([case[1] for case in cases])
    assert np.array_equal(get_hkl_multiplicity(hkl, lattice_system), expected)


def test_ot13_table2_documents_the_roundoff_instability():
    """N can exceed N_cal -- the instability her multiplicity weighting removes.

    Her cells are not published so the numbers cannot be recomputed; this asserts the property the
    table exists to demonstrate, and pins the worst case (tetragonal I, 109 against 61.1).
    """
    exceeds = [
        row for row in fixtures.OT13_TABLE2 if row[3] or row[6]
    ]
    assert len(exceeds) >= 6
    worst = max(
        max(row[1]/row[2], row[4]/row[5]) for row in fixtures.OT13_TABLE2
    )
    assert worst > 1.7


def test_ot21_nearest_neighbour_reduces_to_dewolff():
    """Equation (20) at s = 1 must be exactly de Wolff's epsilon = Q_n/(2N).

    The identity that ties the whole family together, and the reason it needs no sigma.
    """
    q2_obs = np.linspace(0.1, 2.0, 20)
    q2_calc = q2_obs[np.newaxis] + 3e-4
    q2_ref = np.linspace(0.01, 2.0, 137)[np.newaxis]
    assert get_M_nn(q2_obs, q2_calc, q2_ref, dimension=1) == pytest.approx(
        get_M20(q2_obs, q2_calc, q2_ref.copy()), rel=1e-12
    )

    from math import gamma
    for s, expected in fixtures.OT21_NEAREST_NEIGHBOUR_COEFFICIENT.items():
        coefficient = gamma(s/2 + 1)**(1/s)*gamma(1/s)/(np.sqrt(np.pi)*s)
        assert coefficient == pytest.approx(expected, rel=1e-6)


# ================================================================================================
# Werner 1976 -- the critical volume
# ================================================================================================


def test_werner76_g_min_from_decimal_quantisation():
    """Table 1's epsilon columns, from epsilon = |1/d^2 - 1/(d + 0.25 x 10^-n)^2|."""
    for decimals, d_index, eps_index in ((3, 1, 2), (2, 3, 4)):
        d_values = [row[d_index] for row in fixtures.WERNER76_TABLE1]
        printed = np.array([row[eps_index] for row in fixtures.WERNER76_TABLE1])/1e6
        step = 0.25*10.0**(-decimals)
        computed = np.abs(
            1/np.array(d_values)**2 - 1/(np.array(d_values) + step)**2
        )
        # The printed column is rounded to whole units of 10^-6, so 1e-6 absolute.
        assert computed == pytest.approx(printed, abs=1.5e-6)


def test_werner76_critical_volume():
    """V_crit = 3 m d_N / (8 pi g_min * 10): 226 A^3 at two decimals, 2262 at three.

    Note the first power of d_N. Shirley's quotation of this formula OCRs as d_N^2; the first
    power is what reproduces Werner's own numbers, and it is what follows from
    M_N = Q_N/(2 g N_N) with Q_N = 1/d_N^2 and N_N ~ (4 pi/3) d_N^-3 V/m (F-017).
    """
    volume = np.array([fixtures.WERNER76_VOLUME])
    multiplicity = fixtures.WERNER76_MULTIPLICITY
    for decimals, d_index, expected in (
        (2, 3, fixtures.WERNER76_V_CRIT_2DP),
        (3, 1, fixtures.WERNER76_V_CRIT_3DP),
    ):
        d_values = [row[d_index] for row in fixtures.WERNER76_TABLE1]
        g_min = get_g_min_werner(d_values, decimals)
        d_n = np.array([min(d_values)])
        ratio, _ = get_V_over_Vcrit(volume, d_n, g_min, multiplicity)
        v_crit = volume[0]/ratio[0]
        assert v_crit == pytest.approx(expected, rel=0.02), f"{decimals} dp: {v_crit:.0f}"


def test_werner76_threshold_six_recovers_377():
    """Lowering the acceptance threshold from M20 = 10 to 6 must give V_crit = 377 A^3."""
    d_values = [row[3] for row in fixtures.WERNER76_TABLE1]
    g_min = get_g_min_werner(d_values, 2)
    volume = np.array([fixtures.WERNER76_VOLUME])
    ratio, _ = get_V_over_Vcrit(volume, np.array([min(d_values)]), g_min, 4, threshold=6.0)
    assert volume[0]/ratio[0] == pytest.approx(fixtures.WERNER76_V_CRIT_2DP_AT_M6, rel=0.02)


def test_werner76_case_sits_above_its_critical_volume():
    """Werner's point: V = 765.7 A^3 against V_crit = 226, so M20 = 7 reports rounding, not truth.

    This is the published, quantitative statement of the pathology in the brief -- a results list
    dominated by high-volume low-symmetry cells at M20 5-10 is a list sitting above V_crit.
    """
    d_values = [row[3] for row in fixtures.WERNER76_TABLE1]
    g_min = get_g_min_werner(d_values, 2)
    volume = np.array([fixtures.WERNER76_VOLUME])
    ratio, m_max = get_V_over_Vcrit(volume, np.array([min(d_values)]), g_min, 4)
    assert ratio[0] > 3.0
    # The reported M20 exceeds the ceiling the data precision allows, which is why Werner calls it
    # an accidental effect.
    assert fixtures.WERNER76_M20_REPORTED > m_max[0]


# ================================================================================================
# Implementation invariants
# ================================================================================================


def _synthetic_pool(seed=0, n_candidates=40):
    """One correct candidate and a pool of wrong ones, orthorhombic, for invariant tests."""
    rng = np.random.default_rng(seed)
    xnn_true = np.array([0.02, 0.011, 0.007])
    grid = np.arange(-8, 9)
    hkl = np.stack(
        [axis.ravel() for axis in np.meshgrid(grid, grid, grid, indexing="ij")], axis=1
    ).astype(float)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    matrix = np.unique(get_hkl_matrix(hkl, "orthorhombic"), axis=0)

    q2_all = np.sort(matrix @ xnn_true)
    q2_obs = q2_all[:20] + 2e-5*rng.standard_normal(20)
    xnn = np.vstack(
        [xnn_true[np.newaxis], xnn_true[np.newaxis]*(1 + 0.4*rng.random((n_candidates, 3)))]
    )
    q2_ref = xnn @ matrix.T
    index = np.argmin(np.abs(q2_ref[:, :, np.newaxis] - q2_obs[np.newaxis, np.newaxis]), axis=1)
    q2_calc = np.take_along_axis(q2_ref, index, axis=1)
    return q2_obs, q2_calc, q2_ref, xnn


def test_compute_all_does_not_modify_its_arguments():
    """get_M20 writes zeros into q2_ref_calc in place. compute_all must contain that.

    get_M20 is inner-loop code and is deliberately left as it is, so ownership of the copy sits
    here instead.
    """
    q2_obs, q2_calc, q2_ref, xnn = _synthetic_pool()
    before = q2_ref.copy()
    compute_all(q2_obs, q2_calc, q2_ref, xnn, "orthorhombic", "oP")
    assert np.array_equal(q2_ref, before)


def test_compute_all_is_order_invariant():
    """Running it twice on the same arrays must give identical results.

    The failure this guards against is silent: if any FOM corrupted q2_ref_calc, the second call
    would see a zeroed reference list and every count would collapse.
    """
    q2_obs, q2_calc, q2_ref, xnn = _synthetic_pool()
    first = compute_all(q2_obs, q2_calc, q2_ref, xnn, "orthorhombic", "oP")["features"]
    second = compute_all(q2_obs, q2_calc, q2_ref, xnn, "orthorhombic", "oP")["features"]
    for name in first:
        assert np.allclose(first[name], second[name], equal_nan=True), name


def test_every_feature_declares_a_sigma_treatment():
    """PLAN 2.5: a sigma-dependent column must never be readable as sigma-free."""
    q2_obs, q2_calc, q2_ref, xnn = _synthetic_pool()
    result = compute_all(q2_obs, q2_calc, q2_ref, xnn, "orthorhombic", "oP")
    assert set(result["features"]) == set(result["sigma_treatment"])
    assert set(result["sigma_treatment"].values()) <= {"free", "in-sample", "assumed"}
    # The only 'assumed' entries may be the deliberate chi2 reference point.
    assumed = {k for k, v in result["sigma_treatment"].items() if v == "assumed"}
    assert all(name.startswith("chi2_fixed") for name in assumed), assumed


def test_get_M20_degenerate_guard_preserved():
    """get_M20 returns 0 for degenerate candidates; the ranking depends on that behaviour."""
    q2_obs = np.linspace(0.1, 2.0, 20)
    q2_calc = np.zeros((1, 20))
    q2_ref = np.linspace(0.01, 2.0, 50)[np.newaxis]
    assert get_M20(q2_obs, q2_calc, q2_ref.copy())[0] == 0.0


def test_the_correct_candidate_wins_on_the_position_only_merits():
    """A sanity floor: every ranking merit in the zoo must put the true cell first.

    This does not say a merit is *good* -- that is S06's job on real candidate pools -- but a merit
    that cannot do this is broken.
    """
    q2_obs, q2_calc, q2_ref, xnn = _synthetic_pool()
    features = compute_all(q2_obs, q2_calc, q2_ref, xnn, "orthorhombic", "oP")["features"]
    higher_is_better = [
        "M20", "M_tilde", "M_rev", "M_sym", "M_wu", "M_star", "M_1", "M_nn",
        "M_info_clipped", "null_tail_nll", "F_N_q",
    ]
    for name in higher_is_better:
        assert np.argmax(features[name]) == 0, f"{name} did not rank the true cell first"
    lower_is_better = ["X_N", "n_over", "bic", "chi2_fixed"]
    for name in lower_is_better:
        assert np.argmin(features[name]) == 0, f"{name} did not rank the true cell first"


def test_nll_exponential_is_not_a_ranking_merit():
    """F-025: the exponential null *density* is minimised by a perfect fit, so it ranks backwards.

    The S01 handoff calls it "a FOM in its own right". It is not, and this test pins the reason so
    that nobody reintroduces it as one. get_null_tail_nll is the form that discriminates.
    """
    q2_obs, q2_calc, q2_ref, xnn = _synthetic_pool()
    features = compute_all(q2_obs, q2_calc, q2_ref, xnn, "orthorhombic", "oP")["features"]
    assert np.argmax(features["nll_exponential"]) != 0
    assert np.argmax(features["null_tail_nll"]) == 0


# ================================================================================================
# Smith & Snyder 1979 -- F_N, and the case that M20 rates the model rather than the data
# ================================================================================================


def test_smith_snyder_table2_internal_consistency():
    """Every row must satisfy M20 = Q20/(2 N20 eps) for a physically sensible d20.

    |dQ| and N20 and M20 are all printed, so Q20 is determined; the check is that the implied d20
    lands in a plausible range for a powder pattern. It catches a units error in |dQ| (printed as
    10^5 |dQ|) or a transposed column, which is the realistic transcription failure here.
    """
    for label, compound, _, _, _, delta_q, n20, m20, _ in fixtures.SMITH_SNYDER79_TABLE2:
        q20 = m20*2*n20*delta_q*1e-5
        d20 = 1/np.sqrt(q20)
        assert 0.5 < d20 < 6.0, f"{label} ({compound}): implied d20 = {d20:.2f} A"


def test_smith_snyder_same_accuracy_different_M20():
    """Compounds F and R have identical |d2theta| yet M20 differs about six-fold.

    Smith & Snyder's sharpest statement of the cross-lattice problem, and the reason run.py's
    pooling of all fourteen Bravais lattices on raw M20 is biased (F-002). F is triclinic,
    R is cubic.
    """
    rows = {row[0]: row for row in fixtures.SMITH_SNYDER79_TABLE2}
    f_row, r_row = rows["F"], rows["R"]
    assert f_row[4] == r_row[4], "F and R should share the same mean |d2theta|"
    assert r_row[7]/f_row[7] == pytest.approx(6, rel=0.15)
    # F_N, by contrast, rates the data and so does not show the effect nearly as strongly.
    assert r_row[8]/f_row[8] < 2.0


def test_smith_snyder_less_accurate_pattern_scores_higher_M20():
    """Compounds N and O are both cubic; O is less accurate yet has ~3x the M20.

    The volume effect: O's cell is a third of N's. This is what S07 has to calibrate rather than
    remove -- Shirley's reply to Snyder is that the dependence is deliberate and correct in
    direction, and what is unknown is whether it is correct in magnitude (F-012).
    """
    rows = {row[0]: row for row in fixtures.SMITH_SNYDER79_TABLE2}
    n_row, o_row = rows["N"], rows["O"]
    assert o_row[4] > n_row[4], "O should be the less accurate pattern"
    assert o_row[3] < n_row[3], "O should be the smaller cell"
    assert o_row[7]/n_row[7] == pytest.approx(3, rel=0.15)


def test_smith_snyder_cubic_M20_falls_with_volume():
    """Within the cubic block M20 falls with cell volume, but only weakly -- Spearman rho = -0.43.

    Seven cubic compounds spanning V = 64 to 1927 A^3. The direction is Shirley's ~V^(-1/3), and
    the two extreme cells carry it: M (V = 1927) has M20 = 43 while O (V = 228) has 391.

    **The magnitude is not resolvable here, and that is the point.** Over seven compounds whose
    accuracy also varies by a factor of 2.5, the rank correlation is only -0.43, and the ordering
    is not monotone -- O at V = 228 beats every smaller cell. Multiplying out the accuracy factor
    (M20 x |dQ|, which leaves the pure geometry term Q20/2N) does not improve it either. So the
    literature's own best fixture cannot pin the exponent, which is exactly why S07 has to measure
    it at scale rather than adopt V^(-1/3) on authority (F-012).

    Asserted only as: negative, and weaker than one might assume.
    """
    cubic = [row for row in fixtures.SMITH_SNYDER79_TABLE2 if row[2] == "cubic"]
    volume = np.array([row[3] for row in cubic], dtype=float)
    merit = np.array([row[7] for row in cubic], dtype=float)
    order_v = np.argsort(np.argsort(volume))
    order_m = np.argsort(np.argsort(merit))
    correlation = np.corrcoef(order_v, order_m)[0, 1]
    assert -0.9 < correlation < -0.2, f"Spearman rho = {correlation:.2f}"
    # The extremes do carry the effect, even though the middle of the range does not.
    largest = max(cubic, key=lambda row: row[3])
    assert largest[7] == min(row[7] for row in cubic)


def _two_theta(d, wavelength=None):
    wavelength = fixtures.CU_KALPHA1 if wavelength is None else wavelength
    return np.degrees(2*np.arcsin(wavelength/(2*d)))


def test_smith_snyder_table1_two_theta_conversion():
    """Their |d2theta| column must follow from d_obs and d_cal at Cu K-alpha-1.

    The convention check that matters for get_F_N: F_N is defined in 2theta, so a wrong wavelength
    or a d-to-2theta slip would be invisible in any q-space test.

    **The tolerance has to be derived, not chosen**, and Smith & Snyder say why on their own p. 63:
    reporting d values rather than 2theta "introduces unnecessary errors as a result of round off",
    and their Fig. 2 plots exactly that uncertainty against the number of decimal places. Their
    d_obs is printed to three decimals for the first thirteen lines, so half a unit in the last
    place is +/- 0.0005 A, which at d ~ 2.9 A is +/- 0.011 deg in 2theta -- comparable to the whole
    discrepancy being tabulated. So each row is checked against its own round-off budget.
    """
    residuals, budgets = [], []
    for row in fixtures.SMITH_SNYDER79_TABLE1:
        _, _, _, _, _, d_obs, d_cal, delta_2theta, _, _, _, _ = row
        computed = abs(_two_theta(d_obs) - _two_theta(d_cal))
        decimals = len(str(d_obs).split('.')[1])
        half_step = 0.5*10.0**(-decimals)
        budget = abs(_two_theta(d_obs - half_step) - _two_theta(d_obs + half_step))
        assert abs(computed - delta_2theta) <= budget, (row, computed, budget)
        residuals.append(abs(computed - delta_2theta))
        budgets.append(budget)

    # Guard against the round-off budget making this vacuous. The typical residual sits about
    # fifteen times below the typical budget and no row exceeds a third of its own, which is only
    # true if the conversion is actually right; a wrong wavelength would push rows to the limit.
    assert np.median(budgets)/np.median(residuals) > 6, (np.median(budgets), np.median(residuals))
    assert max(r/b for r, b in zip(residuals, budgets)) < 0.5


def test_smith_snyder_table1_FN_formula():
    """F_N = (1/mean|d2theta|)(N/N_poss), row by row, against their printed column.

    Row 16 is skipped: its printed cumulative mean is 0.0049 where the row's own F_N requires
    0.0149 and every neighbour sits near 0.0150. That is a typo in the paper, not a failure of the
    formula -- the point of a fixture is to catch exactly this kind of thing, so it is named
    rather than quietly patched.
    """
    for row in fixtures.SMITH_SNYDER79_TABLE1:
        n_obs, _, _, _, _, _, _, _, mean_delta, merit, completeness, _ = row
        if n_obs in fixtures.SMITH_SNYDER79_TABLE1_TYPO_ROWS:
            continue
        assert (1/mean_delta)*completeness == pytest.approx(merit, rel=0.01), row


def test_smith_snyder_table1_row16_is_a_typo():
    """Pin the typo, so nobody 'fixes' the fixture to match the paper and breaks the test above."""
    row = [r for r in fixtures.SMITH_SNYDER79_TABLE1 if r[0] == 16][0]
    printed_mean, merit, completeness = row[8], row[9], row[10]
    assert printed_mean == 0.0049
    assert (1/printed_mean)*completeness > 3*merit          # printed value is absurd
    assert (1/0.0149)*completeness == pytest.approx(merit, rel=0.01)   # 0.0149 is the real value


def test_get_F_N_reproduces_published_F33():
    """End-to-end: get_F_N on their 33 lines must give their F_33 = 59 in published units.

    N_poss is supplied through a synthetic reference array with exactly 36 lines below the
    cut-off, which is what their space-group-aware count gives; that isolates the F_N arithmetic
    and the 2theta conversion from the reference-list enumeration, which other tests cover.
    """
    rows = fixtures.SMITH_SNYDER79_TABLE1
    q2_obs = np.array([1/row[5]**2 for row in rows])
    q2_calc = np.array([1/row[6]**2 for row in rows])[np.newaxis]

    cutoff = q2_calc[0, -1]
    n_possible = rows[-1][1]        # 36
    q2_ref = np.linspace(0.2*cutoff, 0.999*cutoff, n_possible)[np.newaxis]

    merit_2theta, merit_q = get_F_N(q2_obs, q2_calc, q2_ref, wavelength=fixtures.CU_KALPHA1)
    assert merit_2theta[0] == pytest.approx(fixtures.SMITH_SNYDER79_TABLE1_SUMMARY['F_33'], rel=0.02)
    # The q-space analogue is a different quantity in different units and must NOT match.
    assert not np.isclose(merit_q[0], merit_2theta[0], rtol=0.5)


def test_smith_snyder_table1_summary_is_consistent_with_the_rows():
    summary = fixtures.SMITH_SNYDER79_TABLE1_SUMMARY
    rows = fixtures.SMITH_SNYDER79_TABLE1
    assert len(rows) == summary['n_observed'] == 33
    assert rows[-1][1] == summary['n_possible'] == 36
    assert rows[-1][8] == pytest.approx(summary['mean_abs_delta_2theta'])
    assert rows[-1][9] == pytest.approx(summary['F_33'], rel=0.01)
