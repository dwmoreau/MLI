"""Numeric tables transcribed from the figure-of-merit literature, for use as test fixtures.

**Why this file exists** (PLAN 6.0). Literature fixtures cannot tell us whether a figure of merit
is *good*; they are the only way to know an implementation is *right*. They catch convention errors
-- Q in units of 10^4, which line q_N refers to, whose multiplicity definition -- that synthetic
data can never reveal, because synthetic data would be generated under the same wrong convention.
de Wolff wrote an entire 1972 paper because someone mis-counted N20 and reported M20 = 9 where the
right answer was 4.9.

Every entry cites paper, table and the PDF in /global/cfs/cdirs/m4064/dwmoreau/papers.

**Unit convention, and the trap.** de Wolff and Wu tabulate 10^4 * Q, where Q = 1/d^2 in A^-2.
Getting this wrong gives errors of exactly 10^4 and is invisible if you only check ratios. Values
below are stored **as printed**, in units of 10^4 A^-2, and the tests scale them explicitly.

**Tolerances.** The papers round to two or three significant figures, and in several places the
rounding of an input dominates the disagreement in the output -- de Wolff 1968 Table 2 quotes the
mean discrepancy to one decimal, so a printed 0.5 carries +/- 10% on its own. Each test states the
tolerance it uses and why.

**Smith & Snyder 1979 Table 2 is transcribed** (2026-08-11), read from the rendered page rather
than the text layer -- the extracted text of those pages is corrupted (column headers come out as
'"o~. "~o~s n w t do~ dca I 1~201 Ita0l', digits as '2.8?64'). Render with:

    envs/onnx/bin/python -c "import pypdfium2 as p; p.PdfDocument(PATH)[2].render(scale=3.2).to_pil().save(OUT)"

**Their Table 1 (K2SiF6, 33 lines) is now transcribed too** (2026-08-11), from the rendered page.
The earlier failed check was my own misread: the last row's F_N is 58.82, not 68.82, and 58.82
is exactly what (1/0.0156)(33/36) gives and what their stated F_33 = 59 rounds to.

**One genuine typo in the printed table**, flagged rather than silently corrected: row 16's
cumulative mean |d2theta| reads 0.0049 where every neighbour is near 0.0150. It cannot be right --
a running mean cannot dip by a factor of three and recover -- and the row's own F_N of 59.68
requires 0.0149, since (1/0.0149)(16/18) = 59.7 while (1/0.0049)(16/18) = 181. The fixture stores
0.0049 as printed and the test skips that row, naming it.
"""
import numpy as np

# ------------------------------------------------------------------------------------------------
# de Wolff, P. M. (1961). Acta Cryst. 14, 579-582. DeWolff_1961.pdf
# ------------------------------------------------------------------------------------------------

# Table 1: coefficients in N = Q(C0 sqrt(Q) + C1 a* + C2 b* + C3 c*) / V*, as printed.
# C- and I-centred lattices take half of every coefficient, F-centred a quarter.
DEWOLFF61_TABLE1 = {
    # bravais type: (unique axis, C0, C1, C2, C3)
    "triclinic": (None, 2.095, 0.0, 0.0, 0.0),
    "monoclinic_P": ("b", 1.047, 0.0, 0.786, 0.0),
    "orthorhombic_P": (None, 0.524, 0.393, 0.393, 0.393),
    "tetragonal_P": ("c", 0.214, 0.786, 0.0, 0.160),
    "hexagonal": ("c", 0.150, 0.681, 0.0, 0.113),
    "rhombohedral_hex_axes": ("c", 0.050, 0.227, 0.0, 0.038),
}

# Section 5: Sharan's (1959) aluminium orthoarsenate, an orthorhombic P indexing.
# Q = 50 h^2 + 74 k^2 + 88 l^2, in units of 10^4 sin^2(theta), i.e. xnn = (50, 74, 88).
#
# de Wolff derives N = Q(0.00092 sqrt(Q) + 0.0173) and reports N = 117 at Q = 2000 against an
# actual count of 115. **Assert on the formula, not the count** -- the ~2% gap is the stated
# accuracy of equation (1), not an error.
DEWOLFF61_SHARAN_XNN = np.array([[50.0, 74.0, 88.0]])
DEWOLFF61_SHARAN_N_COEFFICIENTS = (0.00092, 0.0173)   # (C0/V*, sum C_i x_i / V*)
DEWOLFF61_SHARAN_N = [(2000.0, 117.0), (6850.0, 638.0)]
DEWOLFF61_SHARAN_N_ACTUAL_COUNT_AT_2000 = 115

# Table 4, the tabulated average expected discrepancy Delta at five Q values.
#
# **Factor-of-two trap.** Section 5 prints "Delta = 1/(0.00138 sqrt(Q) + 0.0173)", and that
# expression reproduces none of the values below -- it gives exactly twice each of them, because it
# is the mean interval 2*Delta rather than Delta. The correct form is equation (4) halved,
# Delta = (1/2) V* / ((3/2) C0 sqrt(Q) + sum C_i x_i), which reproduces all five. LITERATURE.md and
# the S01 handoff both quote the section 5 form next to these values; see F-024.
DEWOLFF61_SHARAN_DELTA = [
    (197.0, 13.5),
    (999.0, 8.2),
    (2023.0, 6.3),
    (3057.0, 5.4),
    (6309.0, 4.0),
]

# Table 3: cumulative distribution of 214 intervals of a two-dimensional anorthic net with
# Q = 23h^2 + 38k^2 + 4hk, against 214 exp[-(x - 1/2)/18.8], i.e. 2*Delta = 18.8. Validates the
# exponential-interval assumption and quantifies its known slight narrowing.
DEWOLFF61_TABLE3_TWO_DELTA = 18.8
DEWOLFF61_TABLE3 = [
    # (x, actual number of intervals > x - 1/2, exponential prediction)
    (5, 180, 170), (10, 141, 129), (15, 107, 99), (20, 77, 76), (25, 59, 58),
    (30, 42, 44), (35, 29, 34), (40, 22, 26), (45, 19, 20), (50, 13, 15),
    (60, 9, 9), (75, 2, 1), (100, 1, 1),
]

# ------------------------------------------------------------------------------------------------
# de Wolff, P. M. (1968). J. Appl. Cryst. 1, 108-113. DeWolff_1968.pdf
# ------------------------------------------------------------------------------------------------

# Table 2: fourteen indexings with 10^4 Q20, 10^4 mean discrepancy, N20, X20 and M20.
# X20 is None for case 1, where de Wolff notes the number is immaterial.
DEWOLFF68_TABLE2 = [
    # (number, compound, 10^4 Q20, 10^4 epsilon_bar, N20, X20, M20, verdict)
    (1, "alpha-Li4B2O5", 1680, 0.5, 26, None, 60, "correct"),
    (2, "UO2WO4", 1304, 0.6, 35, 3, 33, "correct"),
    (3, "gamma-Cd(OH)2", 3669, 4.0, 25, 0, 18, "correct"),
    (4, "K2RuCl5.NO", 1271, 1.6, 30, 2, 13, "correct"),
    (5, "alpha-Li4B2O5", 2128, 2.0, 40, 5, 13, "correct"),
    (6, "beta-Li4B2O5", 2993, 1.9, 70, 0, 10, "correct"),
    (7, "NaB5O8.5H2O", 1082, 1.6, 50, 0, 6.7, "correct"),
    (8, "Li6B4O9", 1703, 2.6, 63, 2, 5.3, "correct"),
    (9, "Li6B4O9", 1703, 3.0, 52, 2, 5.4, "incorrect"),
    (10, "gamma-UO3", 1357, 1.3, 180, 1, 3.0, "incorrect"),
    (11, "Li3BO3", 2510, 2.3, 210, 0, 2.6, "incorrect"),
    (12, "gamma-Cd(OH)2", 3911, 4.4, 310, 3, 1.5, "incorrect"),
    (13, "delta-Ta2O5", 1621, 0.6, 38, 0, 36, "unconfirmed"),
    (14, "KTa5O13", 1484, 0.6, 33, 0, 37, "unconfirmed"),
]

# Table 3: Li6B4O9. THE famous counterexample -- the incorrect indexing scores M20 = 5.4 against
# the correct 5.3, and de Wolff notes "there is not the remotest analogy between correct and
# incorrect reciprocal lattice; there does not even exist a common zone". Only the intensities
# separate them: the incorrect indexing misses the strongest line of the pattern.
#
# Wu 1988 reports that his accurate expression does separate them, 7.7 against 3.8. Whether
# M^Rev, M^Sym and the analytic null tail do too is the S01 acceptance gate.
#
# Cell constants are from Table 2's companion listing:
DEWOLFF68_LI6B4O9_CORRECT_CELL = (9.18, 23.41, 3.32, 92.68)   # monoclinic P, b unique, beta in deg
DEWOLFF68_LI6B4O9_INCORRECT_CELL = (12.30, 13.37, 11.87)      # orthorhombic B-centred, h+l = 2n

# 10^4 Q from ASTM card 12-129, with both indexings. None means the indexing leaves that observed
# line unexplained, which is why X20 = 2 for both.
#
# **Two overbars were lost in extraction and have been restored:** the printed indices at
# 10^4 Q = 1161 and 1393 are (1 3 -1) and (2 2 -1), not (1 3 1) and (2 2 1). The paper marks
# negative indices with an overbar, which the text layer renders inconsistently -- it survives as
# 'T' at 995 and 1455 and vanishes at these two. The correction is forced by arithmetic, not
# guessed: with l negative the computed Q agrees to 0.25% and 0.20%, matching the systematic +0.2
# to +0.3% offset that every other l-bearing line shows from the rounding of c = 3.32 A; with l
# positive it is out by 5.6% and 9.0%.
#
# **One further extraction correction:** the incorrect indexing's line at 10^4 Q = 548.5 is
# (2 0 2), not (2 0 1). Again forced rather than guessed -- an exhaustive search over B-centred
# indices finds exactly one triple within 15 units of the printed 547.0, namely (2 0 2) at 548.3,
# while (2 0 1) gives 335.4. (2 0 1) also violates nothing, so this is a digit misread rather than
# a lost overbar.
DEWOLFF68_LI6B4O9 = [
    # (10^4 Q_obs, correct hkl, 10^4 Q_calc, incorrect hkl, 10^4 Q_calc)
    (192.9, (1, 2, 0), 191.9, (1, 1, 1), 193.0),
    (286.3, (1, 3, 0), 283.1, (0, 0, 2), 284.0),
    (358.7, None, None, (1, 2, 1), 362.0),
    (548.5, (2, 2, 0), 548.7, (2, 0, 2), 547.0),
    (644.4, (2, 3, 0), 640.0, (1, 3, 1), 642.0),
    (776.0, (1, 6, 0), 775.6, (2, 2, 2), 772.0),
    (886.0, None, None, (3, 2, 1), 888.0),
    (924.0, (0, 1, 1), 924.5, (1, 2, 3), 930.0),
    (995.0, (1, 0, -1), 994.5, None, None),
    (1061.0, (1, 0, 1), 1055.8, (2, 3, 2), 1052.0),
    (1075.0, (1, 1, 1), 1074.1, None, None),
    (1132.0, (2, 6, 0), 1132.5, (0, 0, 4), 1136.0),
    (1161.0, (1, 3, -1), 1158.7, (2, 4, 0), 1163.0),
    (1233.0, (3, 3, 0), 1234.7, (3, 0, 3), 1233.0),
    (1287.0, (1, 8, 0), 1286.4, (3, 1, 3), 1289.0),
    (1366.0, (3, 4, 0), 1362.4, (0, 2, 4), 1361.0),
    (1393.0, (2, 2, -1), 1393.6, (4, 1, 2), 1393.0),
    (1455.0, (1, 5, -1), 1450.5, (2, 1, 4), 1455.0),
    (1560.0, (0, 6, 1), 1562.9, (4, 2, 2), 1561.0),
    (1606.0, (2, 3, 1), 1607.5, (1, 4, 3), 1607.0),
    (1646.0, (2, 8, 0), 1643.2, (0, 3, 4), 1645.0),
    (1703.0, (1, 6, 1), 1712.5, (5, 0, 1), 1715.0),
]

# ------------------------------------------------------------------------------------------------
# de Wolff, P. M. (1972). J. Appl. Cryst. 5, 243. A one-page correction of Khawas.
# ------------------------------------------------------------------------------------------------
# Khawas counted only the calculated lines that matched an observation, so N20 never rose much
# above 20. All distinct calculated Q below the cut-off must be counted, whether or not they
# explain an observed line. A regression test against the mis-counting failure mode.
DEWOLFF72_KHAWAS = {
    "N20_reported": 24,
    "N20_correct": 44,
    "M20_reported": 9.0,
    "M20_correct": 4.9,
}

# ------------------------------------------------------------------------------------------------
# Wu, E. (1988). J. Appl. Cryst. 21, 530-535.
# ------------------------------------------------------------------------------------------------

# Table 1: the average M20/M'20 ratio for primitive cells. This is the cross-lattice inflation that
# run.py inherits by pooling all fourteen Bravais lattices and sorting on raw M20 (F-002). The
# paper lists four tetragonal rows over different parameter ranges; all are kept.
WU88_TABLE1_RATIO = {
    "cubic": [1.82],
    "tetragonal": [1.47, 1.43, 1.42, 1.45],
    "hexagonal": [1.41],
    "orthorhombic": [1.37],
    "monoclinic": [1.24],
    "triclinic": [1.00],
}

# Table 2: the symmetry factor S in M* = S/(V^(2/3) delta), and S' = S / (M20/M'20).
WU88_TABLE2_S = {
    "triclinic": 0.107, "monoclinic": 0.160, "orthorhombic": 0.176,
    "hexagonal": 0.328, "tetragonal": 0.264, "cubic": 0.580,
}
WU88_TABLE2_S_CORRECTED = {
    "triclinic": 0.107, "monoclinic": 0.129, "orthorhombic": 0.129,
    "hexagonal": 0.233, "tetragonal": 0.182, "cubic": 0.319,
}

# Wu's result on the Li6B4O9 pair: his accurate expression separates what M20 cannot.
WU88_LI6B4O9_M_PRIME = {"correct": 7.7, "incorrect": 3.8}

# ------------------------------------------------------------------------------------------------
# Oishi-Tomiyasu, R. (2013). J. Appl. Cryst. 46, 1277-1282.
# ------------------------------------------------------------------------------------------------

# Table 1: peak multiplicity of [hkl], the orbit size under Ci, C2h, D2h, D4h, D3d, D6h and Oh.
# One representative index triple per printed multiplicity class. The rhombohedral row assumes
# rhombohedral axes, which is also this repo's setting.
OT13_TABLE1_MULTIPLICITY = {
    "triclinic": [((1, 2, 3), 2)],
    "monoclinic": [((1, 0, 1), 2), ((0, 1, 0), 2), ((1, 2, 3), 4)],
    "orthorhombic": [
        ((1, 0, 0), 2), ((0, 1, 0), 2), ((0, 0, 1), 2),
        ((1, 1, 0), 4), ((1, 0, 1), 4), ((0, 1, 1), 4),
        ((1, 2, 3), 8),
    ],
    "tetragonal": [
        ((0, 0, 1), 2),
        ((1, 0, 0), 4), ((0, 1, 0), 4), ((1, 1, 0), 4), ((1, -1, 0), 4),
        ((1, 2, 0), 8), ((1, 0, 1), 8), ((0, 1, 1), 8), ((1, 1, 1), 8), ((1, -1, 1), 8),
        ((1, 2, 3), 16),
    ],
    "rhombohedral": [
        ((1, 1, 1), 2),
        ((1, 1, 2), 6), ((1, 2, 1), 6), ((1, 2, 2), 6),
        ((1, 1, 0), 6), ((1, 0, 1), 6), ((0, 1, 1), 6),
        ((1, 2, 3), 12),
    ],
    "hexagonal": [
        ((0, 0, 1), 2),
        ((1, 0, 0), 6), ((0, 1, 0), 6), ((1, 1, 0), 6), ((1, -1, 0), 6), ((1, -2, 0), 6),
        ((1, 2, 0), 12), ((0, 1, 1), 12), ((1, 0, 1), 12), ((1, 1, 1), 12),
        ((1, -1, 1), 12), ((1, -2, 1), 12),
        ((1, 2, 3), 24),
    ],
    "cubic": [
        ((1, 0, 0), 6), ((0, 0, 1), 6),
        ((1, 1, 1), 8),
        ((0, 1, 1), 12), ((1, 0, 1), 12),
        ((1, 1, 2), 24), ((1, -1, 2), 24), ((1, 2, 0), 24), ((1, 0, 2), 24), ((0, 1, 2), 24),
        ((1, 2, 3), 48),
    ],
}
LAUE_GROUP_ORDER = {
    "triclinic": 2, "monoclinic": 4, "orthorhombic": 8, "tetragonal": 16,
    "rhombohedral": 12, "hexagonal": 24, "cubic": 48,
}

# Table 2: N against N_cal([0, q_N]) for fourteen Bravais types, two cells each. The unit-cell
# parameters are **not published**, so these cannot be recomputed; they are kept as a qualitative
# fixture for the property the table exists to demonstrate -- that the raw count N can exceed the
# multiplicity-weighted N_cal, by up to 109 against 61.1 for tetragonal I, which is the round-off
# instability her equation (4) removes.
OT13_TABLE2 = [
    # (crystal system, N_A, N_cal_A, N_exceeds_A, N_B, N_cal_B, N_exceeds_B)
    ("cubic_F", 19, 13.0, True, 31, 40.0, False),
    ("cubic_I", 11, 13.0, False, 49, 26.1, True),
    ("cubic_P", 11, 13.0, False, 22, 24.0, False),
    ("hexagonal", 25, 26.0, False, 58, 59.0, False),
    ("rhombohedral", 56, 53.0, False, 61, 67.3, False),
    ("tetragonal_I", 27, 28.0, False, 109, 61.1, True),
    ("tetragonal_P", 49, 50.0, False, 38, 38.3, False),
    ("orthorhombic_F", 40, 41.0, False, 63, 58.0, True),
    ("orthorhombic_I", 37, 31.5, True, 146, 106.3, True),
    ("orthorhombic_C", 39, 40.0, False, 75, 70.0, True),
    ("orthorhombic_P", 34, 35.0, False, 87, 87.3, False),
    ("monoclinic_B", 42, 39.0, True, 98, 91.0, True),
    ("monoclinic_P", 59, 55.0, True, 98, 99.0, False),
    ("triclinic", 77, 78.0, False, 107, 108.0, False),
]

# ------------------------------------------------------------------------------------------------
# Oishi-Tomiyasu, Tanaka & Nakagawa (2021). J. Appl. Cryst. 54, 624-635.
# ------------------------------------------------------------------------------------------------
# Equation (20): the expected distance from a random point to the nearest of N computed points in
# s dimensions, epsilon = Gamma(s/2+1)^(1/s) Gamma(1/s) / (sqrt(pi) s) * (V/N)^(1/s).
# For s = 1 the coefficient must be exactly 1/2, recovering de Wolff's epsilon = Q_n/(2N).
OT21_NEAREST_NEIGHBOUR_COEFFICIENT = {1: 0.5, 2: 0.5, 3: 2.6789385347/ (36*np.pi)**(1/3)}

# ------------------------------------------------------------------------------------------------
# Werner, P.-E. (1976). J. Appl. Cryst. 9, 216-219.
# ------------------------------------------------------------------------------------------------

# Table 1: (UO)2P2O7, monoclinic a = 10.952, b = 12.764, c = 6.328 A, beta = 120.06 deg,
# V = 765.7 A^3. Twenty lines with d_calc rounded to three and to two decimal places, and the
# corresponding epsilon = |1/d^2 - 1/(d + Delta)^2| with Delta = 0.25 x 10^-n. Tabulated as
# 10^6 epsilon.
#
# Werner's point: V_crit from the two-decimal data is 226 A^3, far below the cell's own 765.7 A^3,
# so "figures of merit give information about rounding errors but not about the correctness of the
# trial cell". Its M20 = 7 is explicitly called an accidental effect.
WERNER76_TABLE1 = [
    # (hkl, d_calc to 3 dp, 10^6 eps_3, d_calc to 2 dp, 10^6 eps_2)
    ((1, 0, 1), 6.303, 2, 6.30, 20),
    ((2, 0, 1), 5.047, 4, 5.05, 39),
    ((5, 2, 1), 4.485, 6, 4.48, 56),
    ((0, 2, 1), 4.156, 7, 4.16, 69),
    ((1, 1, 1), 3.782, 9, 3.78, 92),
    ((3, 0, 1), 3.637, 10, 3.64, 104),
    ((5, 3, 1), 3.527, 11, 3.53, 114),
    ((0, 3, 1), 3.360, 13, 3.36, 132),
    ((2, 1, 2), 3.060, 17, 3.06, 174),
    ((1, 2, 2), 2.769, 24, 2.77, 235),
    ((0, 0, 2), 2.738, 24, 2.74, 243),
    ((2, 4, 1), 2.697, 25, 2.70, 254),
    ((2, 3, 2), 2.533, 31, 2.53, 308),
    ((5, 0, 2), 2.139, 51, 2.14, 509),
    ((4, 3, 0), 2.070, 56, 2.07, 563),
    ((2, 0, 2), 1.980, 64, 1.98, 643),
    ((3, 3, 2), 1.911, 72, 1.91, 716),
    ((3, 1, 3), 1.841, 80, 1.84, 801),
    ((0, 7, 1), 1.730, 96, 1.73, 964),
    ((0, 6, 2), 1.680, 105, 1.68, 1052),
]
WERNER76_CELL = (10.952, 12.764, 6.328, 120.06)
WERNER76_VOLUME = 765.7
WERNER76_MULTIPLICITY = 4
WERNER76_V_CRIT_2DP = 226.0
WERNER76_V_CRIT_3DP = 2262.0
WERNER76_V_CRIT_2DP_AT_M6 = 377.0     # recomputed with the threshold lowered from M20=10 to 6
WERNER76_M20_REPORTED = 7.0

# ------------------------------------------------------------------------------------------------
# Shirley, R. (1980). NBS Special Publication 567, 361-382. nbsspecialpublication567.pdf
# ------------------------------------------------------------------------------------------------
# Qualitative fixtures, not assertions: the shape of a real solution field, which is precisely the
# regime this project targets, described in 1980.
SHIRLEY80_SOLUTION_FIELDS = {
    "alpha-Cu-phthalocyanine": {
        "best_five_M20": [12.7, 9.7, 9.6, 8.6, 8.5],
        "correct_rank": 1,
        "note": "all of the first 20 solutions look acceptable as a d_obs-d_calc list",
    },
    "monosodium urate monohydrate": {
        "best_five_M20": [30, 21, 20, 19, 18],
        "correct_rank": 1,
        "note": "31 cells index all first 20 lines at M20 > 10; ~200 tolerably plausible",
    },
    "monoammonium urate": {
        "best_five_M20": [21, 15, 13],
        "correct_rank": 1,
        "note": None,
    },
}
# Section 2.2: the reproducibility floor. Any claimed improvement must clear this (F-009).
SHIRLEY80_REPRODUCIBILITY_FLOOR = 0.10


# ------------------------------------------------------------------------------------------------
# Smith, G. S. & Snyder, R. L. (1979). J. Appl. Cryst. 12, 60-65.
# ------------------------------------------------------------------------------------------------

# Table 1, summary values only, taken from the running text on p. 62 where they are stated
# unambiguously. K2SiF6 from NBS (1955): 33 of 36 possible lines observed, mean |d2theta| =
# 0.0156 deg, F_33 = 59, and de Wolff M20 = 143.11 (printed as the last line of the table).
SMITH_SNYDER79_TABLE1_SUMMARY = {
    "compound": "K2SiF6",
    "system": "cubic",
    "n_observed": 33,
    "n_possible": 36,
    "mean_abs_delta_2theta": 0.0156,
    "F_33": 59,
    "dewolff_M20": 143.11,
}

# Table 2: the single best fixture for S07, because it pairs both figures of merit with volume and
# crystal system. Nineteen compounds, labelled A-S as in the paper.
#
# Their own worked contrasts, which are the substantive claims and are asserted in the tests:
#   F and R have the *same* |d2theta| = 0.014 yet M20 differs about six-fold (34 against 202),
#   because R is cubic and F triclinic.
#   N and O are both cubic; O is less accurate (0.008 against 0.007) and less complete, yet has
#   nearly three times the M20 (391 against 135), because its cell is a third the volume.
# F20 shows neither effect, which is Smith & Snyder's whole point: F_N rates the data, M20 rates
# the model (Shirley's reply to Snyder, F-012).
SMITH_SNYDER79_TABLE2 = [
    # (label, compound, system, V in A^3, mean |d2theta| in deg, 10^5 |dQ|, N20, M20, F20)
    ("A", "beta-Mg(CH3COO)2", "triclinic", 946, 0.009, 4.1, 34, 24, 63),
    ("B", "K2Zn2V10O18", "triclinic", 892, 0.017, 7.4, 34, 14, 34),
    ("C", "Ni(NO3)2.6H2O", "triclinic", 478, 0.012, 6.3, 36, 26, 45),
    ("D", "CuSeO4.5H2O", "triclinic", 383, 0.015, 8.7, 27, 24, 50),
    ("E", "NaI.2H2O", "triclinic", 249, 0.007, 4.6, 23, 57, 120),
    ("F", "Pb(COO)2", "triclinic", 185, 0.014, 9.5, 24, 34, 58),
    ("G", "Mg2B2O5", "triclinic", 172, 0.010, 7.1, 36, 41, 54),
    ("H", "Cd(OH)NO3", "orthorhombic", 844, 0.007, 4.4, 25, 65, 117),
    ("I", "NaHgCl3", "orthorhombic", 708, 0.008, 4.5, 23, 56, 114),
    ("J", "KCaCl3", "orthorhombic", 572, 0.008, 5.6, 24, 64, 101),
    ("K", "Bi2S3", "orthorhombic", 502, 0.013, 9.4, 30, 35, 52),
    ("L", "NaMnF3", "orthorhombic", 255, 0.016, 14.0, 26, 42, 50),
    ("M", "NH4Al(SeO4)2.12H2O", "cubic", 1927, 0.015, 9.7, 20, 43, 68),
    ("N", "HBO2", "cubic", 701, 0.007, 7.0, 20, 135, 141),
    ("O", "ZnTe", "cubic", 228, 0.008, 9.7, 21, 391, 120),
    ("P", "Cr3Rh", "cubic", 102, 0.011, 12.4, 22, 285, 84),
    ("Q", "CsCaF3", "cubic", 93, 0.015, 17.4, 20, 154, 69),
    ("R", "KNiF3", "cubic", 65, 0.014, 16.9, 20, 202, 73),
    ("S", "LiBaF3", "cubic", 64, 0.011, 13.3, 20, 259, 91),
]


# Table 1: K2SiF6 from NBS (1955), the line-by-line worked example. Columns as printed:
# (N_obs, N_poss, h, k, l, d_obs, d_cal, |d2theta|, cumulative mean |d2theta|, F_N,
#  N_obs/N_poss, de Wolff M'_N).
#
# NBS cubic patterns are Cu K-alpha-1, lambda = 1.5405981 A, which reproduces their |d2theta|
# column from d_obs and d_cal to 1e-4 deg -- that is what makes this an end-to-end fixture for
# get_F_N in *published* units rather than in the q-space analogue.
#
# Row 16's cumulative mean is a typo in the paper: see the module docstring.
SMITH_SNYDER79_TABLE1_TYPO_ROWS = (16,)
SMITH_SNYDER79_TABLE1 = [
    (1, 1, 1, 1, 1, 4.699, 4.6956, 0.0138, 0.0138, 72.31, 1.00000, 344),
    (2, 3, 2, 2, 0, 2.877, 2.8754, 0.0172, 0.0155, 43.02, 0.66667, 411),
    (3, 4, 3, 1, 1, 2.453, 2.4522, 0.0125, 0.0145, 51.74, 0.75000, 612),
    (4, 5, 2, 2, 2, 2.349, 2.3478, 0.0204, 0.0160, 50.07, 0.80000, 590),
    (5, 6, 4, 0, 0, 2.034, 2.0332, 0.0173, 0.0162, 51.31, 0.83333, 751),
    (6, 7, 3, 3, 1, 1.866, 1.8658, 0.0045, 0.0143, 60.01, 0.85714, 1026),
    (7, 8, 4, 2, 0, 1.819, 1.8186, 0.0120, 0.0140, 62.72, 0.87500, 1083),
    (8, 9, 4, 2, 2, 1.661, 1.6601, 0.0310, 0.0161, 55.27, 0.88889, 1048),
    (9, 10, 5, 1, 1, 1.565, 1.5652, 0.0081, 0.0152, 59.21, 0.90000, 1241),
    (10, 11, 4, 4, 0, 1.438, 1.4377, 0.0139, 0.0151, 60.32, 0.90909, 1449),
    (11, 12, 5, 3, 1, 1.375, 1.3747, 0.0153, 0.0151, 60.72, 0.91667, 1544),
    (12, 13, 6, 0, 0, 1.356, 1.3555, 0.0292, 0.0163, 56.74, 0.92308, 1415),
    (13, 14, 6, 2, 0, 1.286, 1.2859, 0.0040, 0.0153, 60.59, 0.92857, 1671),
    (14, 16, 6, 2, 2, 1.2262, 1.2261, 0.0079, 0.0148, 59.16, 0.87500, 1773),
    (15, 17, 4, 4, 4, 1.1741, 1.1739, 0.0172, 0.0150, 59.02, 0.88235, 1884),
    (16, 18, 5, 5, 1, 1.1390, 1.1388, 0.0141, 0.0049, 59.68, 0.88889, 1991),
    (17, 20, 6, 4, 2, 1.0869, 1.0868, 0.0087, 0.0145, 58.50, 0.85000, 2122),
    (18, 21, 7, 3, 1, 1.0588, 1.0588, 0.0030, 0.0139, 61.70, 0.85714, 2352),
    (19, 22, 8, 0, 0, 1.0167, 1.0166, 0.0098, 0.0137, 63.14, 0.86364, 2586),
    (20, 23, 7, 3, 3, 0.9936, 0.9936, 0.0006, 0.0130, 66.77, 0.86957, 2862),
    (21, 24, 8, 2, 0, 0.9864, 0.9863, 0.0187, 0.0133, 65.82, 0.87500, 2822),
    (22, 25, 8, 2, 2, 0.9585, 0.9585, 0.0027, 0.0128, 68.68, 0.88000, 3114),
    (23, 26, 7, 5, 1, 0.9391, 0.9391, 0.0031, 0.0124, 71.39, 0.88462, 3366),
    (24, 27, 6, 6, 2, 0.9331, 0.9329, 0.0325, 0.0132, 67.19, 0.88889, 3163),
    (25, 28, 8, 4, 0, 0.9094, 0.9093, 0.0207, 0.0135, 66.00, 0.89286, 3254),
    (26, 29, 9, 1, 1, 0.8927, 0.8927, 0.0028, 0.0131, 68.36, 0.89655, 3497),
    (27, 30, 8, 4, 2, 0.8873, 0.8874, 0.0188, 0.0133, 67.55, 0.90000, 3491),
    (28, 31, 6, 6, 4, 0.8670, 0.8670, 0.0049, 0.0130, 69.35, 0.90323, 3755),
    (29, 32, 9, 3, 1, 0.8525, 0.8526, 0.0193, 0.0133, 68.35, 0.90625, 3843),
    (30, 33, 8, 4, 4, 0.8302, 0.8301, 0.0444, 0.0143, 63.59, 0.90909, 3836),
    (31, 34, 7, 7, 1, 0.8175, 0.8174, 0.0405, 0.0151, 60.21, 0.91176, 3822),
    (32, 35, 8, 6, 0, 0.8134, 0.8133, 0.0415, 0.0160, 57.26, 0.91429, 3744),
    (33, 36, 10, 2, 0, 0.7975, 0.7975, 0.0034, 0.0156, 58.82, 0.91667, 4011),
]
CU_KALPHA1 = 1.5405981
