import numpy as np

from mlindex.utilities.UnitCellTools import get_hkl_matrix
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.UnitCellTools import reciprocal_uc_conversion


# How each figure of merit treats the measurement error sigma, which this project never assumes is
# known (PLAN 2.5, F-008). Every function added to the zoo registers here, and compute_all reports
# the labels alongside the values so that a sigma-dependent number can never be read as if it were
# sigma-free.
#
#   free       the normalisation comes from the calculated-line density, a property of the
#              candidate rather than of the instrument. The whole classical family is of this type.
#   in-sample  sigma is estimated from the data being scored, by an estimator that must itself be
#              validated before the number is trusted.
#   assumed    sigma is taken from an external model. Reference points only, never a deliverable.
SIGMA_TREATMENT = {
    "M20": "free",
    "M20_likelihood": "free",
    "delta_dewolff61": "free",
    "n_dewolff61": "free",
    "nll_exponential": "free",
    "null_tail_nll": "free",
    "N_cal": "free",
    "M_tilde": "free",
    "M_rev": "free",
    "M_sym": "free",
    "X_N": "free",
}


# de Wolff (1961), Acta Cryst. 14, 579-582, Table 1: the coefficients in
#
#     N(Q) = Q(C0 sqrt(Q) + C1 a* + C2 b* + C3 c*) / V*
#
# with C1, C2, C3 multiplying the three reciprocal axes of de Wolff's conventional setting.
# C-centred and I-centred lattices take half of every coefficient, F-centred a quarter.
#
# Two departures from the printed table, both deliberate and both validated by enumeration in
# mlindex/scripts/run_fom_audits.py:
#
#   hR   de Wolff's row is for hexagonal axes, while this repo carries hR on rhombohedral axes
#        (get_hkl_matrix's rhombohedral branch is (sum h^2, hk+hl+kl), i.e. a = b = c and
#        alpha = beta = gamma). get_dewolff61_axes converts to hexagonal axes before applying it.
#
#   cP/cI/cF  de Wolff tabulates no cubic row. In cubic the calculated lines collapse onto integer
#        values of h^2+k^2+l^2, so N grows *linearly* in Q rather than as Q^(3/2): the leading
#        term vanishes and C0 = 0. The remaining coefficient is the asymptotic density of integers
#        that are attainable as h^2+k^2+l^2 under the centring rule, which is exact:
#            P   5/6     integers not of the form 4^a(8b+7)
#            I   11/24   of those, the even ones: 1/4 + (1/4)(5/6)
#            F   1/3     4*(sum of three squares) or n = 3 mod 8: (1/4)(5/6) + 1/8
#        Enumeration over |h|,|k|,|l| <= 140 gives 0.83347, 0.45847 and 0.33347. Note that the
#        halve-for-I, quarter-for-F rule does *not* hold here (11/24 is not half of 5/6), because
#        the collapse onto integers, not the point density, is what sets the count.
#        A consequence worth knowing: with C0 = 0, Delta(Q) is constant in Q for cubic.
DEWOLFF61_COEFFICIENTS = {
    "aP": (2.095, 0.0, 0.0, 0.0),
    "mP": (1.047, 0.0, 0.786, 0.0),
    "mC": (0.5235, 0.0, 0.393, 0.0),
    "oP": (0.524, 0.393, 0.393, 0.393),
    "oC": (0.262, 0.1965, 0.1965, 0.1965),
    "oI": (0.262, 0.1965, 0.1965, 0.1965),
    "oF": (0.131, 0.09825, 0.09825, 0.09825),
    "tP": (0.214, 0.786, 0.0, 0.160),
    "tI": (0.107, 0.393, 0.0, 0.080),
    "hP": (0.150, 0.681, 0.0, 0.113),
    "hR": (0.050, 0.227, 0.0, 0.038),
    "cP": (0.0, 5 / 6, 0.0, 0.0),
    "cI": (0.0, 11 / 24, 0.0, 0.0),
    "cF": (0.0, 1 / 3, 0.0, 0.0),
}


def get_dewolff61_axes(xnn, lattice_system, bravais_lattice):
    """The three reciprocal axes and V* in the setting de Wolff (1961) Table 1 assumes.

    Returns (a_star, b_star, c_star, reciprocal_volume), each of shape (n_candidates,). For every
    lattice except hR this is the repo's own setting; hR is converted from rhombohedral axes to
    hexagonal ones, since that is the setting de Wolff tabulates.
    """
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
    )
    if lattice_system == "rhombohedral":
        # Rhombohedral -> hexagonal axes, via the direct cell:
        #   a_hex = 2 a_rh sin(alpha_rh/2),  c_hex = a_rh sqrt(3 + 6 cos alpha_rh)
        unit_cell = reciprocal_uc_conversion(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
        )
        a_rh, alpha_rh = unit_cell[:, 0], unit_cell[:, 1]
        a_hex = 2 * a_rh * np.sin(alpha_rh / 2)
        c_hex = a_rh * np.sqrt(3 + 6 * np.cos(alpha_rh))
        a_star = 2 / (np.sqrt(3) * a_hex)
        c_star = 1 / c_hex
        # The hexagonal reciprocal cell has gamma* = 60 degrees.
        return a_star, a_star, c_star, a_star**2 * c_star * np.sin(np.pi / 3)

    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
    )
    if lattice_system == "cubic":
        a_star = reciprocal_unit_cell[:, 0]
        return a_star, a_star, a_star, reciprocal_volume
    elif lattice_system in ("tetragonal", "hexagonal"):
        a_star, c_star = reciprocal_unit_cell[:, 0], reciprocal_unit_cell[:, 1]
        return a_star, a_star, c_star, reciprocal_volume
    return (
        reciprocal_unit_cell[:, 0],
        reciprocal_unit_cell[:, 1],
        reciprocal_unit_cell[:, 2],
        reciprocal_volume,
    )


def get_dewolff61_terms(xnn, lattice_system, bravais_lattice):
    """The two Q-independent groupings that N(Q) and Delta(Q) are both built from.

    Returns (leading, surface, reciprocal_volume) where leading = C0 and
    surface = C1 a* + C2 b* + C3 c*, each of shape (n_candidates,).
    """
    if bravais_lattice not in DEWOLFF61_COEFFICIENTS:
        raise ValueError(f"no de Wolff 1961 coefficients for {bravais_lattice}")
    c0, c1, c2, c3 = DEWOLFF61_COEFFICIENTS[bravais_lattice]
    a_star, b_star, c_star, reciprocal_volume = get_dewolff61_axes(
        xnn, lattice_system, bravais_lattice
    )
    surface = c1 * a_star + c2 * b_star + c3 * c_star
    return c0 * np.ones(len(surface)), surface, reciprocal_volume


def get_n_dewolff61(q2, xnn, lattice_system, bravais_lattice):
    """de Wolff (1961) eq. (1a): the expected number of distinct calculated lines below Q.

    N(Q) = Q(C0 sqrt(Q) + C1 a* + C2 b* + C3 c*) / V*

    The sqrt(Q) term is the volume of the limiting sphere and the rest is its surface. It is the
    surface term that the repo's 4 pi q^2 V / mu density omits, and it dominates at low Q, which is
    exactly where the first twenty peaks live (F-015, F-022).

    q2 is (n_peaks,) or (n_candidates, n_peaks); xnn is (n_candidates, n_components). Returns
    (n_candidates, n_peaks). de Wolff states the accuracy of this formula as a few per cent; it is
    a smooth expectation, not a count, so it does not step at each line.
    """
    leading, surface, reciprocal_volume = get_dewolff61_terms(
        xnn, lattice_system, bravais_lattice
    )
    q2 = np.atleast_2d(q2)
    return (
        q2
        * (leading[:, np.newaxis] * np.sqrt(q2) + surface[:, np.newaxis])
        / reciprocal_volume[:, np.newaxis]
    )


def get_delta_dewolff61(q2, xnn, lattice_system, bravais_lattice):
    """de Wolff (1961) eq. (4): the expected discrepancy at Q for an *arbitrary* (wrong) cell.

    Delta(Q) = (1/2) V* / ((3/2) C0 sqrt(Q) + C1 a* + C2 b* + C3 c*)

    2 Delta = dQ/dN is the mean interval between successive calculated lines at Q; the factor of
    two is the inspection paradox, which de Wolff spells out in his footnote -- an observed line
    cannot be distinguished from the calculated lines around it, so each of the two sub-intervals
    it creates is itself an arbitrary interval.

    This is *local in Q*, where de Wolff 1968's M20 uses the single global number Q20/(2 N20). It
    is the analytic form of Shirley's per-line epsilon, it carries no free parameters and no sigma,
    and it supersedes the 4 pi q^2 V / mu density in get_M20_likelihood.

    Careful: de Wolff's section 5 prints this formula for his worked example *without* the leading
    one half -- as written there it gives 2 Delta, not Delta. His own tabulated Delta values in
    Table 4 follow the expression above (F-024).
    """
    leading, surface, reciprocal_volume = get_dewolff61_terms(
        xnn, lattice_system, bravais_lattice
    )
    q2 = np.atleast_2d(q2)
    denominator = (
        1.5 * leading[:, np.newaxis] * np.sqrt(q2) + surface[:, np.newaxis]
    )
    return 0.5 * reciprocal_volume[:, np.newaxis] / denominator


def get_nll_exponential(q2_obs, q2_calc, xnn, lattice_system, bravais_lattice):
    """The analytic null log-density: how these discrepancies score under "this candidate is wrong".

    de Wolff (1961) section 3: if the intervals between calculated lines are exponentially
    distributed -- free-path statistics -- then so are the discrepancies, and a discrepancy larger
    than x occurs with frequency exp(-x/Delta). Hence

        -log L_null = sum_i [ |dQ_i| / Delta(Q_i) + log Delta(Q_i) ]

    Closed form, no free parameters, no sigma.

    **This is not a figure of merit and must not be ranked on.** The S01 handoff describes it as
    "a FOM in its own right"; it is not, and the reason is elementary once written down: the
    exponential density peaks at zero, so a perfect fit is the *most* likely outcome under the
    null and -log L_null is minimised, not maximised, by a good candidate. It is also dominated by
    the log Delta term, which is a function of the candidate's volume and symmetry rather than of
    its agreement with the data. Measured on a real pool it ranks essentially by volume (F-025).

    What it *is* good for is the null itself -- it is the correct per-line null density, and S07
    needs exactly that to standardise other merits against. For ranking, use get_null_tail_nll,
    which is built from the same distribution and does discriminate.

    de Wolff validates the exponential assumption on 214 intervals of a two-dimensional anorthic
    net (his Table 3) and finds the real distribution slightly narrower -- real Q values are a
    little more regular than random -- so this null is mildly conservative, and more so for high
    symmetry, where he notes g can fall to Delta/2 in the equidistant limit.

    Returns (n_candidates,).
    """
    delta = get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)
    discrepancy = np.abs(np.atleast_2d(q2_obs) - q2_calc)
    return np.sum(discrepancy/delta + np.log(delta), axis=1)


def get_null_tail_nll(
    q2_obs, q2_calc, xnn, lattice_system, bravais_lattice, min_discrepancy=0.0
):
    """The discriminating form of the same exponential null: how improbably good the fit is.

        -log P(null) = -sum_i log[ 1 - exp(-|dQ_i| / Delta(Q_i)) ]

    Under "this candidate is wrong" the chance of a discrepancy at least as small as the one
    observed is 1 - exp(-|dQ|/Delta), so this is the negative log probability that an arbitrary
    cell would fit this well at every peak. Large means the agreement is too good to be chance,
    which is the direction a figure of merit should run.

    This is the analytic backbone S07 needs, and it is closed form, parameter-free and sigma-free.
    Structurally it is Taupin's information merit with de Wolff 1961's Delta(Q) in place of
    Taupin's 4 pi q^2 V / mu density -- the substitution that S01_density_model.md measured as
    worth 30-58% in the line count.

    **`min_discrepancy` matters here for the same reason it does in get_M_info_clipped.** The
    per-line term diverges as the discrepancy goes to zero, so one line landing exactly on a
    calculated position can swamp the other nineteen (F-026). Pass the resolution of the observed
    data as a floor. On de Wolff's Li6B4O9 table the effect is stark and purely an artefact of the
    printing: his incorrect indexing's Q_calc column is rounded to whole units where the correct
    one is rounded to tenths, producing three exact zeros and a merit sixteen times larger for the
    *wrong* cell.

    Two caveats it inherits. It is a *per-candidate* tail probability, so it is not yet a
    look-elsewhere-corrected significance: we generate thousands of candidates and report the best,
    and de Wolff himself notes that makes a good-looking false cell "fairly certain" (F-016). The
    extreme-value correction is S07's job. And because real Q sequences are more regular than
    exponential, especially at high symmetry, the null is optimistic in a symmetry-dependent way
    (F-015); measuring that is Q11.

    Returns (n_candidates,), larger being better.
    """
    delta = get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)
    discrepancy = np.maximum(np.abs(np.atleast_2d(q2_obs) - q2_calc), min_discrepancy)
    return -np.sum(np.log(1 - np.exp(-discrepancy/delta) + 1e-100), axis=1)


# Generators of the Laue group per crystal system, as integer matrices acting on hkl. The full
# group is closed from these; get_hkl_multiplicity asserts the resulting order. Oishi-Tomiyasu
# (2013) section 2 defines her peak multiplicity as the orbit size under exactly these groups:
# Ci (triclinic), C2h (monoclinic), D2h (orthogonal), D4h (tetragonal), D3d (rhombohedral),
# D6h (hexagonal) and Oh (cubic).
INVERSION = -np.eye(3, dtype=int)
LAUE_GENERATORS = {
    # Ci, order 2.
    "triclinic": ([INVERSION], 2),
    # C2h with unique axis b, matching the repo's monoclinic convention, order 4.
    "monoclinic": ([INVERSION, np.diag([-1, 1, -1])], 4),
    # D2h, order 8.
    "orthorhombic": ([INVERSION, np.diag([1, -1, -1]), np.diag([-1, 1, -1])], 8),
    # D4h: four-fold about c, plus a two-fold about a, order 16.
    "tetragonal": (
        [INVERSION, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), np.diag([1, -1, -1])],
        16,
    ),
    # D3d on rhombohedral axes: the three-fold is the cyclic permutation of hkl and the two-fold
    # swaps a pair, so the rotation part is the full symmetric group on three letters. Order 12.
    "rhombohedral": (
        [
            INVERSION,
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        ],
        12,
    ),
    # D6h: the six-fold acts on (h, k) as (h, k) -> (-k, h + k), which is what leaves
    # h^2 + hk + k^2 invariant. Order 24.
    "hexagonal": (
        [
            INVERSION,
            np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        ],
        24,
    ),
    # Oh: all signed permutations, order 48.
    "cubic": (
        [
            INVERSION,
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            np.diag([1, -1, -1]),
        ],
        48,
    ),
}


def get_laue_operations(lattice_system):
    """Every operation of the Laue group, closed from the generators above."""
    generators, order = LAUE_GENERATORS[lattice_system]
    operations = {np.eye(3, dtype=int).tobytes(): np.eye(3, dtype=int)}
    frontier = [np.eye(3, dtype=int)]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            product = generator @ current
            key = product.tobytes()
            if key not in operations:
                operations[key] = product
                frontier.append(product)
    if len(operations) != order:
        raise AssertionError(
            f"{lattice_system} Laue group closed to {len(operations)}, expected {order}"
        )
    return np.stack(list(operations.values()), axis=0)


def get_hkl_multiplicity(hkl, lattice_system):
    """Oishi-Tomiyasu 2013 Table 1: the number of Miller indices equivalent to each [hkl].

    Computed as the orbit size under the Laue group rather than by matching index patterns, which
    is both shorter and harder to get wrong. Validated against her printed Table 1 in
    tests/test_fom_literature.py.

    hkl is (n, 3); returns (n,) integers.
    """
    operations = get_laue_operations(lattice_system)
    hkl = np.asarray(hkl, dtype=int)
    # (n_operations, n, 3) -> the orbit of every hkl at once.
    orbits = np.einsum("sij,nj->nsi", operations, hkl)
    multiplicity = np.empty(len(hkl), dtype=int)
    for index, orbit in enumerate(orbits):
        multiplicity[index] = len(np.unique(orbit, axis=0))
    return multiplicity


def get_N_cal(q2_ref_calc, q_min, q_max, weights=None):
    """Oishi-Tomiyasu 2013 eq. (4): the multiplicity-weighted count of computed lines in a range.

        N_cal([q_min, q_max]) = sum_j 1 / m([h_j k_j l_j])

    summed over *every* Miller index whose computed line falls in the range, so a complete orbit
    contributes exactly 1 and the count is stable against the round-off that makes the raw count N
    jump (her Table 2 has cases where N exceeds N_cal by nearly a factor of two).

    **This repo's reference lists already carry one representative per orbit** -- Audit B measured
    `frac_duplicate_q2 = 0.000` for all fourteen Bravais lattices -- so the correct weight here is
    1 per entry, and N_cal reduces to a plain count of reference lines in range. That is what
    `weights=None` gives, and it means get_M20's existing N is already Oishi-Tomiyasu's N_cal
    rather than de Wolff's raw N: her fix (i) is in place, and the instability she documents does
    not apply to us. Pass explicit weights only when handing this a full, unreduced hkl list.

    q2_ref_calc is (n_candidates, n_ref); q_min and q_max are (n_candidates,).
    Returns (n_candidates,).
    """
    in_range = (q2_ref_calc >= q_min[:, np.newaxis]) & (q2_ref_calc <= q_max[:, np.newaxis])
    if weights is None:
        return in_range.sum(axis=1).astype(float)
    return (in_range*weights[np.newaxis, :]).sum(axis=1)


def get_M_rev_sym(q2_obs, q2_calc, q2_ref_calc, weights=None):
    """Oishi-Tomiyasu 2013 eqs (5), (7), (9)-(11): the restricted, reversed and symmetric FOMs.

    Replaces the dead `get_M20_sym_reversed`, which called an undefined `get_multiplicity` and
    hardcoded 'monoclinic' for the reference multiplicities (F-003).

    Three merits, all sigma-free:

      M_tilde  de Wolff's M_n with N_cal and the range restricted to [q_I, q_N], where q_I is the
               computed line nearest the *first* observed peak. Corrects for patterns missing
               low-index reflections, and for the calculated-line density growing with q rather
               than being uniform. Her best single FOM over 24 real patterns.
      M_rev    the same construction with the roles of observed and computed lines exchanged: it
               asks whether every *computed* line is accounted for by an observation. It is
               therefore blind to impurity peaks and sensitive to over-predicted reflections --
               exactly the axis on which over-prediction currently escapes punishment, and the
               reason this is the most likely quick win in the zoo.
      M_sym    M_tilde * M_rev, invariant under exchanging the two line sets. Rescued the true
               cell in both of her impurity-peak cases where M_tilde picked a false one.

    Arguments follow get_M20: q2_obs is (n_peaks,), q2_calc (n_candidates, n_peaks) the computed
    positions of the assigned lines, q2_ref_calc (n_candidates, n_ref) every reference line.
    `weights` is 1/m per reference entry and defaults to 1 -- see get_N_cal for why that is right
    here. Returns (M_tilde, M_rev, M_sym), each (n_candidates,).

    Unlike get_M20 this does not modify q2_ref_calc.
    """
    n_peaks = q2_obs.shape[0]
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)

    # The cut-off is the calculated position of the last assigned line, as in get_M20 -- Audit C
    # measured the alternatives as differing by less than the reproducibility floor.
    q_max = q2_calc[:, -1]
    # q_I: the computed line closest to the first observed peak.
    q_min = np.take_along_axis(
        q2_ref_calc,
        np.argmin(np.abs(q2_ref_calc - q2_obs[0]), axis=1)[:, np.newaxis],
        axis=1,
    )[:, 0]

    in_range = (q2_ref_calc >= q_min[:, np.newaxis]) & (q2_ref_calc <= q_max[:, np.newaxis])
    row_weights = np.ones(q2_ref_calc.shape[1]) if weights is None else weights
    n_cal = (in_range*row_weights[np.newaxis, :]).sum(axis=1)
    q_n = np.max(np.where(in_range, q2_ref_calc, -np.inf), axis=1)

    # Every reference line in range is scored against its nearest observed peak (eq. 10), which is
    # the reversal: observed lines that no computed line explains cost nothing, computed lines that
    # no observation explains cost everything.
    nearest = np.min(np.abs(q2_ref_calc[:, :, np.newaxis] - q2_obs[np.newaxis, np.newaxis]), axis=2)
    reversed_sum = (np.where(in_range, nearest, 0.0)*row_weights[np.newaxis, :]).sum(axis=1)

    good = (n_cal > 0) & np.isfinite(q_n) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    M_tilde = np.zeros(q2_calc.shape[0])
    M_rev = np.zeros(q2_calc.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilon = (q_n - q_min)/(2*np.where(n_cal > 0, n_cal, 1))
        M_tilde[good] = epsilon[good]/discrepancy[good]
        discrepancy_reversed = reversed_sum/np.where(n_cal > 0, n_cal, 1)
        epsilon_reversed = (q2_obs[-1] - q2_obs[0])/(2*n_peaks)
        usable = good & (discrepancy_reversed > 0)
        M_rev[usable] = epsilon_reversed/discrepancy_reversed[usable]
    return M_tilde, M_rev, M_tilde*M_rev


def get_X_N(q2_obs, q2_calc, q2_ref_calc, tolerance_factor=1.0):
    """de Wolff's X_N: how many observed lines below the cut-off are *not* explained.

    de Wolff reports X20 alongside M20 and never folds it in -- "M20 > 10 guarantees correctness
    provided there are few spurious lines (X20 not above 2)". Werner 1976 strengthens it to
    requiring all lines below Q20 to be indexed. Nothing in the repo computes it.

    Every observed peak in this pipeline receives an assignment, so "unindexed" needs a criterion.
    The sigma-free one implied by de Wolff's own framework is used here: a peak counts as
    unindexed when its discrepancy is no better than what an arbitrary cell would produce, that is
    when |dQ| exceeds tolerance_factor times the expected discrepancy Q_N/(2N). No external error
    model enters, and the threshold moves with the candidate's own line density.

    Returns (n_candidates,) integer counts.
    """
    cutoff = q2_calc[:, -1]
    in_range = q2_ref_calc < cutoff[:, np.newaxis]
    count = in_range.sum(axis=1)
    q_n = np.max(np.where(in_range, q2_ref_calc, 0.0), axis=1)
    expected = np.where(count > 0, q_n/(2*np.maximum(count, 1)), np.inf)

    below_cutoff = q2_obs[np.newaxis] <= cutoff[:, np.newaxis]
    unexplained = np.abs(q2_obs[np.newaxis] - q2_calc) > tolerance_factor*expected[:, np.newaxis]
    return (below_cutoff & unexplained).sum(axis=1)


def get_M20_from_xnn(q2_obs, xnn, hkl, hkl_ref, lattice_system):
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    q2_ref_calc = np.sum(hkl2_ref * xnn[:, np.newaxis, :], axis=2)
    return get_M20(q2_obs, q2_calc, q2_ref_calc)


def get_M20(q2_obs, q2_calc, q2_ref_calc):
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    smaller_ref_peaks = q2_ref_calc < q2_calc[:, -1][:, np.newaxis]
    np.putmask(q2_ref_calc, ~smaller_ref_peaks, 0)
    last_smaller_ref_peak = np.max(q2_ref_calc, axis=1)
    N = np.sum(smaller_ref_peaks, axis=1)

    # There is an unknown issue that causes q2_calc to be all zero
    # These cases are caught and the M20 score is returned as zero.
    # Also catch cases where N == 0 for all peaks
    good_indices = np.logical_and(q2_calc.sum(axis=1) != 0, N != 0)
    expected_discrepancy = np.zeros(q2_calc.shape[0])
    expected_discrepancy[good_indices] = last_smaller_ref_peak[good_indices] / (
        2 * N[good_indices]
    )
    M20 = expected_discrepancy / discrepancy
    return M20


def get_M20_likelihood_from_xnn(q2_obs, xnn, hkl, lattice_system, bravais_lattice):
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
    )
    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
    )
    log_likelihood, probability, M = get_M20_likelihood(
        q2_obs, q2_calc, bravais_lattice, reciprocal_volume
    )
    return log_likelihood, probability, M


def get_M20_likelihood(q2_obs, q2_calc, bravais_lattice, reciprocal_volume):
    # This was inspired by Taupin 1988
    # Probability that a peak is correctly assigned:
    # arg = Expected number of peaks within error from random unit cell
    # P = 1 / (1 + arg)
    mu, nu = get_multiplicity_taupin88(bravais_lattice)
    observed_difference2 = (np.sqrt(q2_obs[np.newaxis]) - np.sqrt(q2_calc)) ** 2
    # There is an upstream error where reciprocal volumes can be very small.
    # Adding 1e-100 here prevents division by zero errors
    arg = (
        8
        * np.pi
        * q2_obs
        * np.sqrt(observed_difference2)
        / (reciprocal_volume[:, np.newaxis] * mu + 1e-100)
    )
    probability = 1 / (1 + arg)
    # The 1e-100 factor prevents np.log(~0) = -infinity
    M = -1 / np.log(2) * np.sum(np.log(1 - np.exp(-arg) + 1e-100), axis=1)
    return -np.sum(np.log(probability + 1e-100), axis=1), probability, M


def get_multiplicity_taupin88(bravais_lattice):
    # The commented out returns come from Taupin 1988
    # The others are from empirically plotting the
    # non systematic absences
    if bravais_lattice == "cF":
        return 4 * 32, 1
    elif bravais_lattice == "cI":
        return 2 * 32, 1
    elif bravais_lattice == "cP":
        return 1 * 32, 1
    elif bravais_lattice == "hP":
        # return 1*24, 2
        return 1 * 14, 2
    elif bravais_lattice == "hR":
        # return 1*24, 2
        return 1 * 8, 2
    elif bravais_lattice == "tI":
        # return 2*16, 2
        return 2 * 13, 2
    elif bravais_lattice == "tP":
        # return 1*16, 2
        return 1 * 13, 2
    elif bravais_lattice in ["oC", "oI"]:
        # return 2*8, 3
        return 2 * 7, 3
    elif bravais_lattice == "oF":
        # return 4*8, 3
        return 4 * 7, 3
    elif bravais_lattice == "oP":
        # return 1*8, 3
        return 1 * 7, 3
    elif bravais_lattice == "mC":
        # return 2*4, 4
        return 2 * 3.2, 4
    elif bravais_lattice == "mP":
        # return 1*4, 4
        return 1 * 3.5, 4
    elif bravais_lattice == "aP":
        # return 1*2, 6
        return 1 * 1.8, 6


# ---------------------------------------------------------------------------------------------
# Per-peak assignment probability -- S11 block B / S01-C
#
# The question these answer is not "is this cell right" but "is *this peak* assigned to the right
# Miller index", one number per observed line. PLAN section 4's assumptions A6 and A7 turn on it,
# and the S11 handoff asks for two analytic estimators: the repo's rho = 1/(1 + eps*dN) and
# Taupin 1988's P = 1 - exp(-2*eps*n).
#
# **They are the same statistic under two link functions.** get_M20_likelihood already computes
#
#     arg = 8 pi q2_obs |sqrt(q2_obs) - sqrt(q2_calc)| / (V* mu)
#         = 2 eps n(q),    n(q) = dN/dq = 4 pi q^2 / (V* mu)
#
# and returns *both* 1/(1 + arg) per peak and -log2 prod(1 - exp(-arg)) as Minfo. So rho and
# Taupin's P are one number seen through 1/(1+x) and 1 - e^-x; they agree to first order, diverge
# where coincidence approaches certainty, and are **identically ranked** because both links are
# monotone. Any comparison between them is a comparison of calibration alone, and the best that
# any monotone function of arg can do is its isotonic recalibration -- which is why S11 measures
# that as the bar rather than the two raw forms (STATUS section 6, 2026-08-20).
#
# The genuinely different third form is the same link with de Wolff 1961's Delta(Q) in place of
# Taupin's 4 pi q^2 V / mu, which S01 measured under-counting lines by 30-58% (F-027).
#
# **Units differ between the two families, and this is not cosmetic.** The Taupin family's eps is
# a discrepancy in q, |sqrt(q2_obs) - sqrt(q2_calc)| (FigureOfMerits.py, get_M20_likelihood),
# while get_M20, get_nll_exponential, get_null_tail_nll and get_M_info_clipped all work in q^2.
# A per-peak comparison that does not say which is which is not reproducible.
# ---------------------------------------------------------------------------------------------


def get_assignment_argument(q2_obs, q2_calc, bravais_lattice, reciprocal_volume):
    """Taupin's 2*eps*n: the expected number of lines of a random cell within eps of this peak.

    Lifted verbatim out of get_M20_likelihood so the two probability forms below cannot drift
    from the shipped merit. get_assignment_probability(form='rho') reproduces
    get_M20_likelihood(...)[1] exactly, and a test asserts it.

    Returns (n_candidates, n_peaks), larger meaning the match is easier to get by chance.
    """
    mu, _ = get_multiplicity_taupin88(bravais_lattice)
    eps = np.abs(np.sqrt(np.atleast_1d(q2_obs))[np.newaxis] - np.sqrt(q2_calc))
    return (
        8*np.pi*q2_obs*eps/(np.atleast_1d(reciprocal_volume)[:, np.newaxis]*mu + 1e-100)
        )


def get_assignment_probability(q2_obs, q2_calc, bravais_lattice, reciprocal_volume, form='rho'):
    """Per-peak P(this peak is assigned its correct Miller index), from the Taupin density.

    `form` selects the link on the shared argument (see the note above):

      - 'rho'    -> 1/(1 + arg), the repo's own expression, what get_M20_likelihood returns and
                    what feeds refine_cell's peak selection and the assignment threshold.
      - 'taupin' -> exp(-arg), the complement of Taupin 1988 eq. (10)'s coincidence probability
                    1 - exp(-2 eps n). Note the orientation: Taupin's published P is the chance a
                    *random* line falls this close, so the probability the assignment is right is
                    one minus it. The handoff quotes the coincidence form; calibrating that
                    against a correctness label would report a reliability curve upside down.
      - 'arg'    -> the raw statistic, for fitting a recalibration to.

    Returns (n_candidates, n_peaks) in [0, 1] for the two probability forms.
    """
    argument = get_assignment_argument(q2_obs, q2_calc, bravais_lattice, reciprocal_volume)
    if form == 'arg':
        return argument
    if form == 'rho':
        return 1/(1 + argument)
    if form == 'taupin':
        return np.exp(-argument)
    raise ValueError(f"form must be 'rho', 'taupin' or 'arg', not {form!r}")


def get_assignment_probability_dewolff(
    q2_obs, q2_calc, xnn, lattice_system, bravais_lattice, min_discrepancy=0.0
):
    """The same question with de Wolff 1961's line density instead of Taupin's.

        P(correct) = exp(-|dQ| / Delta(Q))

    Under de Wolff's exponential interval statistics the chance that an arbitrary cell puts a line
    at least this close is 1 - exp(-|dQ|/Delta), so the complement is the probability the match is
    not a coincidence. Summing -log2(1 - P) over the peaks returns get_M_info_clipped with the
    neighbour clipping inactive, and -log(1 - P) returns get_null_tail_nll; a test asserts both,
    so this is the same family and not a fourth convention.

    S01-C names this the front-runner: it is parameter-free, sigma-free, local in Q, and F-027
    measured its density model beating the 4 pi q^2 V / mu form by 30-58% on exactly the count
    these probabilities are built from. Unlike the Taupin forms its discrepancy is in q^2.

    `min_discrepancy` floors |dQ| the way get_null_tail_nll's does (F-026); it does not bind on a
    probability the way it does on a log, but it is kept so the two agree term by term.

    Returns (n_candidates, n_peaks) in [0, 1].
    """
    delta = get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)
    discrepancy = np.maximum(np.abs(np.atleast_2d(q2_obs) - q2_calc), min_discrepancy)
    return np.exp(-discrepancy/delta)




# Free cell parameters per lattice system -- Taupin's nu, the divisor in the reduced chi-square.
N_FREE_PARAMETERS = {
    'cubic': 1, 'tetragonal': 2, 'hexagonal': 2, 'rhombohedral': 2,
    'orthorhombic': 3, 'monoclinic': 4, 'triclinic': 6,
    }


def get_assignment_sigma(q2_obs, q2_ref_calc, lattice_system, robust=False, chunk=256):
    """In-sample estimate of the measurement scale, from the candidate's own residuals.

    Taupin 1988's reduced chi-square: sigma^2 = sum(dQ_i^2)/(N - nu), where dQ_i is the distance
    from each observed peak to the nearest calculated line and nu is the number of free cell
    parameters. **Estimated per candidate from the data in front of it**, so nothing is assumed
    known and PROTOCOL section 3 rule 4 is respected -- this is the "in-sample estimation with a
    validated estimator" the rule allows, and Q7 is the question of which estimator to prefer.

    `robust=True` uses 1.4826 x median|dQ| instead, which mis-assigned peaks cannot inflate.
    Measured on mP it is *worse* calibrated than the chi-square form (ECE 0.117 against 0.052),
    which is worth understanding before preferring it.

    Returns (sigma, d1) -- the scale per candidate and the nearest-line distance per peak.
    """
    q2_obs = np.atleast_1d(np.asarray(q2_obs, dtype=np.float64))
    q2_ref_calc = np.atleast_2d(np.asarray(q2_ref_calc, dtype=np.float64))
    n_candidates, n_peaks = q2_ref_calc.shape[0], q2_obs.size
    d1 = np.empty((n_candidates, n_peaks), dtype=np.float64)
    for start in range(0, n_candidates, chunk):
        block = q2_ref_calc[start:start + chunk]
        for peak in range(n_peaks):
            d1[start:start + chunk, peak] = np.abs(block - q2_obs[peak]).min(axis=1)
    n_free = N_FREE_PARAMETERS[lattice_system]
    if robust:
        sigma = 1.4826*np.median(d1, axis=1)
    else:
        sigma = np.sqrt(np.sum(d1**2, axis=1)/max(n_peaks - n_free, 1))
    return np.maximum(sigma, 1e-300), d1


def get_assignment_posterior(q2_obs, q2_ref_calc, lattice_system, sigma=None,
                             sigma_multiplier=1.0, robust=False, chunk=256, d1=None):
    """P(each observed peak is assigned its correct Miller index) -- a posterior, not a null.

        P_i = exp(-d_i^2/2 sigma^2) / sum_j exp(-d_j^2/2 sigma^2),  evaluated at the nearest line

    **This asks a different question from `get_assignment_probability`, and that is the point.**
    The repo's rho and Taupin's P both answer "could an arbitrary cell have put a line this close"
    -- a coincidence probability under a null. A null has to have a base rate bolted onto it
    afterwards, which is why rho states 0.87 where the truth is 0.04 and why recalibrating it is
    worth twenty times its raw self (F-125). This answers "given these calculated lines and this
    peak, which line produced it", which is a posterior over the competing lines and is therefore
    calibrated by construction when the error model is right. Measured on mP it is calibrated
    **with nothing fitted**: ECE 0.052 against a recalibrated network's 0.051.

    Two properties worth knowing, both of which fall out rather than being arranged.

    **It reads local crowding, which is what actually causes a mis-assignment.** The competing
    line's distance enters directly, so a peak whose two nearest lines are 1e-5 apart is scored
    quite differently from one whose neighbours are 1e-2 away. rho cannot see this at all: its
    density is the smooth global 4 pi q^2 V/mu, so it inherits only the residual -- and the
    residual **alone** ranks mis-assignment at AUC 0.445, which is worse than chance, because in a
    crowded region the wrong line is close too. That is the whole of rho's 0.511.

    **A wrong cell is penalised without being told.** sigma is estimated from the candidate's own
    residuals, so a cell that fits badly gets a large sigma, a flat posterior and low confidence on
    every peak. Nothing has to detect that the cell is wrong.

    What it does **not** do, and cannot: be calibrated on a pool that is mostly wrong cells.
    P(peak right) = P(cell right) x P(peak right | cell right), and this is the second factor. The
    first is the figure of merit itself, which is the combiner's job. Any per-peak statistic that
    tries to be unconditionally calibrated is trying to solve indexing, which is how rho came to
    state 0.87 against a 0.04 base rate.

    `sigma_multiplier` scales the fitted sigma for the sensitivity curve PROTOCOL section 3 rule 4
    requires of anything that uses one. `sigma` overrides the in-sample estimate entirely, which is
    for testing and is **not** a licence to assume the generator's own error model.

    `sigma` and `d1` may both be passed straight from a previous `get_assignment_sigma` call, which
    is how a caller that wants *both* the scale and the posterior -- S11 block C, over ten million
    candidates -- pays for the nearest-line scan once instead of twice. That scan is the whole cost
    of this function; passing them halves it and changes no result.

    Returns (n_candidates, n_peaks) in (0, 1].
    """
    q2_obs = np.atleast_1d(np.asarray(q2_obs, dtype=np.float64))
    q2_ref_calc = np.atleast_2d(np.asarray(q2_ref_calc, dtype=np.float64))
    if sigma is None or d1 is None:
        estimated, distances = get_assignment_sigma(
            q2_obs, q2_ref_calc, lattice_system, robust=robust, chunk=chunk
            )
        sigma = estimated if sigma is None else sigma
        d1 = distances if d1 is None else d1
    sigma = np.broadcast_to(np.atleast_1d(np.asarray(sigma, dtype=np.float64)),
                            (q2_ref_calc.shape[0],))
    d1 = np.asarray(d1, dtype=np.float64)
    scale = 2*(sigma*sigma_multiplier)**2

    posterior = np.empty(d1.shape, dtype=np.float64)
    for start in range(0, q2_ref_calc.shape[0], chunk):
        stop = start + chunk
        block = q2_ref_calc[start:stop]
        block_scale = scale[start:stop][:, np.newaxis]
        for peak in range(q2_obs.size):
            # Subtracting the nearest distance before exponentiating is the standard log-sum-exp
            # shift: the nearest line's own term becomes exactly 1 and nothing underflows, so the
            # sum is exact where a direct exp(-d^2/2s^2) would be 0/0 for a well-fitting candidate.
            excess = np.abs(block - q2_obs[peak])**2 - (d1[start:stop, peak]**2)[:, np.newaxis]
            posterior[start:stop, peak] = 1.0/np.sum(np.exp(-excess/block_scale), axis=1)
    return posterior


def get_M20_sym_reversed(q2_obs, xnn, hkl, hkl_ref, lattice_system):
    """SUPERSEDED by get_M_rev_sym. Dead code, kept only so the name still resolves.

    This never ran: it calls an undefined `get_multiplicity` and hardcodes 'monoclinic' for the
    reference multiplicities (F-003). get_M_rev_sym implements Oishi-Tomiyasu 2013 eqs (5), (7),
    (9)-(11) properly, follows get_M20's calling convention, and is vectorised over candidates.
    """
    # This function is broken because there is no get_multiplicity function
    hkl2 = get_hkl_matrix(hkl, lattice_system)
    q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    hkl2_ref = get_hkl_matrix(hkl_ref, lattice_system)
    q2_ref_calc = np.sum(hkl2_ref * xnn[:, np.newaxis, :], axis=2)
    multiplicity = get_multiplicity(
        hkl.reshape((hkl.shape[0] * hkl.shape[1], hkl.shape[2])), lattice_system
    ).reshape(hkl.shape[:2])
    multiplicity_ref = get_multiplicity(hkl_ref, "monoclinic")

    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    smaller_ref_peaks = q2_ref_calc < q2_calc[:, -1][:, np.newaxis]
    last_smaller_ref_peak = np.zeros(q2_calc.shape[0])
    expected_discrepancy_reversed = (q2_obs[-1] - q2_obs[0]) / (2 * 20)
    discrepancy_reversed = np.zeros(q2_calc.shape[0])
    for i in range(q2_calc.shape[0]):
        q2_ref_smaller = q2_ref_calc[i, smaller_ref_peaks[i]]
        multiplicities_ref_smaller = multiplicity_ref[smaller_ref_peaks[i]]
        sort_indices = np.argsort(q2_ref_smaller)
        q2_ref_smaller = q2_ref_smaller[sort_indices]
        multiplicities_ref_smaller = multiplicities_ref_smaller[sort_indices]
        last_smaller_ref_peak[i] = q2_ref_smaller[-1]

        N_calc = np.sum(1 / multiplicities_ref_smaller)
        differences = np.min(
            np.abs(q2_ref_smaller[np.newaxis] - q2_obs[:, np.newaxis]), axis=0
        )
        discrepancy_reversed[i] = (
            np.sum(differences / multiplicities_ref_smaller) / N_calc
        )

    N = np.sum(smaller_ref_peaks, axis=1)
    expected_discrepancy = last_smaller_ref_peak / (2 * N)
    M20 = expected_discrepancy / discrepancy
    M20_reversed = expected_discrepancy_reversed / discrepancy_reversed
    M20_sym = M20 * M20_reversed
    return M20, M20_sym, M20_reversed


# ---------------------------------------------------------------------------------------------
# The rest of the zoo (S01 Part A items 5-15). Everything below follows get_M20's conventions:
# q2_obs is (n_peaks,), q2_calc is (n_candidates, n_peaks) holding the computed position of the
# line assigned to each observed peak, q2_ref_calc is (n_candidates, n_ref) holding every
# reference line. Nothing below modifies its arguments.
# ---------------------------------------------------------------------------------------------


# Wu 1988 Table 2: the symmetry factor S in M* = S / (V^(2/3) delta), and S' = S divided by the
# mean M20/M'20 ratio of his Table 1. S' is the version corrected for the uniform-spacing
# approximation, and is the one to use when comparing across crystal systems.
WU88_SYMMETRY_FACTOR = {
    "triclinic": 0.107,
    "monoclinic": 0.160,
    "orthorhombic": 0.176,
    "tetragonal": 0.264,
    "hexagonal": 0.328,
    "rhombohedral": 0.328,
    "cubic": 0.580,
}
WU88_SYMMETRY_FACTOR_CORRECTED = {
    "triclinic": 0.107,
    "monoclinic": 0.129,
    "orthorhombic": 0.129,
    "tetragonal": 0.182,
    "hexagonal": 0.233,
    "rhombohedral": 0.233,
    "cubic": 0.319,
}

# Wu 1988 Table 1: the mean M20/M'20 ratio per crystal system. This is the cross-lattice bias that
# run.py inherits when it pools all fourteen Bravais lattices and sorts on raw M20 (F-002).
WU88_M20_RATIO = {
    "triclinic": 1.00,
    "monoclinic": 1.24,
    "orthorhombic": 1.37,
    "tetragonal": 1.43,
    "hexagonal": 1.41,
    "rhombohedral": 1.41,
    "cubic": 1.82,
}

# Number of free cell parameters per crystal system: Taupin's nu, used for degrees of freedom.
N_CELL_PARAMETERS = {
    "cubic": 1,
    "tetragonal": 2,
    "hexagonal": 2,
    "rhombohedral": 2,
    "orthorhombic": 3,
    "monoclinic": 4,
    "triclinic": 6,
}

SIGMA_TREATMENT.update(
    {
        "M_wu": "free",
        "M_star": "free",
        "M_star_corrected": "free",
        "M_werner_max": "free",
        "M_info_clipped": "free",
        "M_1": "free",
        "n_over": "free",
        "max_gap": "free",
        "zone_dominance": "free",
        "M_nn": "free",
        "F_N": "free",
        "F_N_q": "free",
        "V_over_Vcrit": "free",
        "M_werner_frac": "free",
        "chi2_taupin": "in-sample",
        "chi2_entrywise": "in-sample",
        "bic": "in-sample",
        "chi2_fixed": "assumed",
    }
)

# S10's predictive merits. The normalisation of cv_M and cv_tail_nll is the calculated-line
# spacing of the *refit* cell, so they are sigma-free in exactly the sense the rest of the
# classical family is; only the chi2 forms, which divide by an estimated residual scale, are not.
SIGMA_TREATMENT.update(
    {
        "cv_raw": "free",
        "cv_M": "free",
        "cv_tail_nll": "free",
        "cv_n_scored": "free",
        "cv_n_voided": "free",
        "cv_max_leverage": "free",
        "ho_raw": "free",
        "ho_M": "free",
        "ho_tail_nll": "free",
        "ho_n_scored": "free",
        "is_raw": "free",
        "is_M": "free",
        "is_M20": "free",
        "cv_M20": "free",
        "ho_M20": "free",
        "is_tail_nll": "free",
        "is_n_scored": "free",
        "cv_n_predicted": "free",
        "ho_n_predicted": "free",
        "is_n_predicted": "free",
        "cv_chi2": "in-sample",
        "ho_chi2": "in-sample",
        "is_chi2": "in-sample",
    }
)


def _sorted_lines_in_range(q2_ref_calc, cutoff, floor=None):
    """Reference lines below the cut-off, sorted ascending, with the rest pushed to +inf.

    Returns (sorted_lines, count). Several of the FOMs below need the calculated lines as an
    ordered sequence rather than as a count, which is the whole point of Wu's and Shirley's
    refinements over de Wolff's.
    """
    in_range = q2_ref_calc < cutoff[:, np.newaxis]
    if floor is not None:
        in_range &= q2_ref_calc >= floor[:, np.newaxis]
    lines = np.where(in_range, q2_ref_calc, np.inf)
    return np.sort(lines, axis=1), in_range.sum(axis=1)


def get_M_wu(q2_obs, q2_calc, q2_ref_calc):
    """Wu 1988 eqs (5), (6): the de Wolff FOM with the exact mean arbitrary discrepancy.

        g_n  = sum_k (Q_(k) - Q_(k-1))^2 / 4 / Q_(N)
        M'_n = g_n / delta_n

    de Wolff approximates the mean arbitrary discrepancy by Q_N/(2N), which assumes the calculated
    lines are evenly spaced. Wu integrates it over the actual pattern, so M'_n depends on where the
    lines are and not merely how many there are. Two consequences we care about: it is continuous
    under small perturbations of the cell, which makes it a good inner-loop candidate, and it
    removes the uniform-spacing inflation that gives cubic a 1.82x advantage over triclinic on raw
    M20 (Wu's Table 1, F-002).

    Oishi-Tomiyasu found it a worse *ranker* than M_tilde precisely because that continuity lets
    lower-symmetry cells reach the highest values; both properties are worth having measured.

    Returns (n_candidates,).
    """
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    cutoff = q2_calc[:, -1]
    lines, count = _sorted_lines_in_range(q2_ref_calc, cutoff)

    # The k = 1 interval runs from Q = 0, matching Wu's sum starting at k = 1.
    finite = np.isfinite(lines)
    previous = np.concatenate(
        [np.zeros((lines.shape[0], 1)), np.where(finite, lines, 0.0)[:, :-1]], axis=1
    )
    gaps = np.where(finite, np.where(finite, lines, 0.0) - previous, 0.0)
    q_n = np.max(np.where(np.isfinite(lines), lines, 0.0), axis=1)

    good = (count > 0) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    merit = np.zeros(q2_calc.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        g_bar = np.sum(gaps**2/4, axis=1)/np.where(q_n > 0, q_n, 1)
        merit[good] = g_bar[good]/discrepancy[good]
    return merit


def get_M_star(q2_obs, q2_calc, volume, lattice_system, corrected=False):
    """Wu 1988 eq (9): M* = S / (V^(2/3) delta), the cheapest FOM in the literature.

    Uses Smith & Snyder's V ~ K_n d_n^3 to replace Q_N/(2N) by a closed form in the cell volume,
    so no line counting happens at all. Wu proposes it specifically as an intermediate testing
    criterion inside trial-and-error indexing, which is exactly this project's inner-loop slot.

    `volume` is the direct-space cell volume in A^3, shape (n_candidates,). With corrected=True the
    S' column of his Table 2 is used instead of S, which divides out the mean M20/M'20 ratio and so
    puts the crystal systems on a common footing.

    Returns (n_candidates,).
    """
    table = WU88_SYMMETRY_FACTOR_CORRECTED if corrected else WU88_SYMMETRY_FACTOR
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    merit = np.zeros(q2_calc.shape[0])
    good = (discrepancy > 0) & (volume > 0) & (q2_calc.sum(axis=1) != 0)
    merit[good] = table[lattice_system]/(volume[good]**(2/3)*discrepancy[good])
    return merit


def get_M_1(q2_obs, q2_calc, q2_ref_calc):
    """Shirley 1980 section 2.2: the de Wolff family with a *per-line local* epsilon.

        delta_i = |Q_obs_i - nearest calculated line|
        epsilon_i = half the separation between the two calculated lines bracketing Q_obs_i
        M_1 = <epsilon> / <delta>

    This is arguably the most natural member of the family and, as far as we can tell, has never
    been benchmarked at scale. Shirley's argument is that because epsilon_i is derived from the
    data rather than from an external error estimate, the ratio "resists spurious improvement with
    increasing volume" -- which is the literature's own answer to the sigma question (F-008).

    It is the empirical counterpart of get_delta_dewolff61, which gives the same local quantity
    analytically. Where both are computable they should agree, and they can be cross-plotted.

    Returns (n_candidates,).
    """
    cutoff = q2_calc[:, -1]
    lines, count = _sorted_lines_in_range(q2_ref_calc, cutoff)

    # For each observed peak, the bracketing pair of calculated lines. searchsorted on a row-sorted
    # array with the out-of-range entries at +inf gives the insertion point directly.
    n_candidates, n_peaks = q2_calc.shape
    upper_index = np.stack(
        [np.searchsorted(lines[row], q2_obs) for row in range(n_candidates)], axis=0
    )
    n_lines = lines.shape[1]
    upper_index = np.clip(upper_index, 1, n_lines - 1)
    upper = np.take_along_axis(lines, upper_index, axis=1)
    lower = np.take_along_axis(lines, upper_index - 1, axis=1)
    # A peak beyond the last in-range line has no upper bracket; fall back to the local gap below.
    upper = np.where(np.isfinite(upper), upper, lower)
    epsilon = np.abs(upper - lower)/2

    delta = np.abs(q2_obs[np.newaxis] - q2_calc)
    merit = np.zeros(n_candidates)
    mean_delta = np.mean(delta, axis=1)
    good = (count > 1) & (mean_delta > 0) & (q2_calc.sum(axis=1) != 0)
    merit[good] = np.mean(epsilon, axis=1)[good]/mean_delta[good]
    return merit


def get_M_info_clipped(
    q2_obs, q2_calc, xnn, lattice_system, bravais_lattice, min_discrepancy=0.0
):
    """Taupin 1988 eqs (20), (25): the information merit with neighbour clipping.

    The unclipped form (get_M20_likelihood) lets a single calculated line take credit for two close
    observed lines, which over-rewards crowded patterns -- the hard stratum. Taupin's fix replaces
    the interval 2*epsilon by dQ_minus + dQ_plus, each capped at half the spacing to the adjacent
    observed line.

    Two deliberate departures from Taupin, both of which keep this sigma-free (PLAN 2.5):

      - Taupin caps against chi_r * E_i, an a priori error estimate. We do not have one and are not
        allowed to assume one, so the half-width used here is the observed |dQ| itself, matching
        what the repo's existing unclipped form already does.
      - the line density is de Wolff 1961's Delta(Q) rather than Taupin's 4 pi q^2 V / mu, which
        was measured to under-count by 30-58% (S01_density_model.md).

    **`min_discrepancy` is not optional in practice.** The per-line term -log(1 - exp(-x)) diverges
    as x -> 0, so a single observation that happens to land exactly on a calculated line carries
    unbounded weight and can dominate the whole merit (F-026). Pass a floor on |dQ| expressing the
    resolution of the observed data -- how finely the peak positions are actually known. That is
    knowable, unlike sigma: it is the quantisation of the input, in the same spirit as Werner's
    g_min, not an error model. Defaults to 0.0, which reproduces the unclipped formula.

    Returns the merit in bits, (n_candidates,), larger being better.
    """
    delta = get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)
    discrepancy = np.maximum(np.abs(q2_obs[np.newaxis] - q2_calc), min_discrepancy)

    # Half the spacing to the observed neighbour on each side; the outermost peaks are capped on
    # one side only, so their gap is reused.
    spacing = np.diff(q2_obs)/2
    lower_cap = np.concatenate([spacing[:1], spacing])
    upper_cap = np.concatenate([spacing, spacing[-1:]])
    width = np.minimum(discrepancy, lower_cap[np.newaxis]) + np.minimum(
        discrepancy, upper_cap[np.newaxis]
    )

    # Taupin's argument is (interval)/(mean interval between calculated lines) = width/(2 Delta).
    argument = width/(2*delta)
    return -1/np.log(2)*np.sum(np.log(1 - np.exp(-argument) + 1e-100), axis=1)


def get_n_over(q2_obs, q2_calc, q2_ref_calc, tolerance_factor=0.5):
    """Calculated lines in range that no observation accounts for, and the longest such run.

    The ingredient of M_rev, useful on its own as a cheap over-prediction detector, and -- in the
    run form -- as a dominant-zone detector, since a dominant zone leaves long stretches of
    predicted lines unobserved.

    "Nearby" is defined from the local calculated-line spacing, not from an assumed error: a
    calculated line counts as unaccounted for when the nearest observed peak is further away than
    tolerance_factor times the local gap between calculated lines. Sigma-free by construction.

    Returns (n_over, max_gap), each (n_candidates,).
    """
    cutoff = q2_calc[:, -1]
    lines, count = _sorted_lines_in_range(q2_ref_calc, cutoff)
    finite = np.isfinite(lines)

    previous = np.concatenate(
        [np.zeros((lines.shape[0], 1)), np.where(finite, lines, 0.0)[:, :-1]], axis=1
    )
    local_gap = np.where(finite, np.where(finite, lines, 0.0) - previous, np.inf)
    nearest = np.min(
        np.abs(np.where(finite, lines, np.inf)[:, :, np.newaxis] - q2_obs[np.newaxis, np.newaxis]),
        axis=2,
    )
    unaccounted = finite & (nearest > tolerance_factor*local_gap)

    n_over = unaccounted.sum(axis=1)
    # Longest run of consecutive unaccounted-for calculated lines.
    max_gap = np.zeros(lines.shape[0], dtype=int)
    for row in range(lines.shape[0]):
        run = best = 0
        for flag in unaccounted[row, : count[row]]:
            run = run + 1 if flag else 0
            best = max(best, run)
        max_gap[row] = best
    return n_over, max_gap


def get_M_nn(q2_obs, q2_calc, q2_ref_calc, dimension=1):
    """Oishi-Tomiyasu, Tanaka & Nakagawa 2021: the recipe that generates the whole FOM family.

    A de Wolff-type FOM is epsilon/delta where epsilon is the expected distance from a random point
    to the nearest of N computed points in whatever space the data occupy. For s dimensions in a
    convex body of volume V,

        epsilon = Gamma(s/2 + 1)^(1/s) Gamma(1/s) / (sqrt(pi) s) * (V/N)^(1/s)

    For s = 1 this is exactly (1/2)(V/N), so with V = Q_N it reduces to de Wolff's Q_N/(2N) -- the
    test suite asserts that identity. The construction is explicitly scale free, which is the
    structural reason the whole family needs no sigma (F-013).

    Its real use here is as the correct normalisation for FOMs built on *derived* point processes
    -- the s != 1 cases. The repo's triplet FOM, which worked on differences of q2 values and was
    the original motivation for checking this, was removed in 2026-08-11 (F-033).

    Returns (n_candidates,).
    """
    from math import gamma

    s = dimension
    coefficient = gamma(s/2 + 1)**(1/s)*gamma(1/s)/(np.sqrt(np.pi)*s)
    discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    cutoff = q2_calc[:, -1]
    in_range = q2_ref_calc < cutoff[:, np.newaxis]
    count = in_range.sum(axis=1)
    q_n = np.max(np.where(in_range, q2_ref_calc, 0.0), axis=1)

    merit = np.zeros(q2_calc.shape[0])
    good = (count > 0) & (discrepancy > 0) & (q2_calc.sum(axis=1) != 0)
    epsilon = coefficient*(q_n[good]/count[good])**(1/s)
    merit[good] = epsilon/discrepancy[good]
    return merit


def get_zone_dominance(xnn, lattice_system):
    """Shirley 1980 section 3.3: S / V*^(2/3), where S is the smallest reciprocal net-cell area.

    The dominant zone is the powder zone whose reciprocal net cell has the smallest area. Shirley:
    "when S approaches half the geometric mean value (V*^(2/3)), this dominant zone will probably
    index about 10 out of the first 20 lines" -- leaving only ten lines to fix the other three
    powder constants, which is easy to do by chance. So a value near 0.5 flags a candidate whose
    apparent agreement may be an artefact of a single strong zone.

    Better founded than the min/max axis-length ratio the random-forest grouping currently uses.
    Only the three principal zones (a*b*, a*c*, b*c*) are considered, which is what Shirley's
    discussion assumes; higher-index zones would need an enumeration.

    Returns (n_candidates,).
    """
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
    )
    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
    )
    if lattice_system == "triclinic":
        a, b, c = reciprocal_unit_cell[:, 0], reciprocal_unit_cell[:, 1], reciprocal_unit_cell[:, 2]
        alpha, beta, gamma = (
            reciprocal_unit_cell[:, 3],
            reciprocal_unit_cell[:, 4],
            reciprocal_unit_cell[:, 5],
        )
        areas = np.stack(
            [a*b*np.sin(gamma), a*c*np.sin(beta), b*c*np.sin(alpha)], axis=1
        )
    elif lattice_system == "monoclinic":
        a, b, c = reciprocal_unit_cell[:, 0], reciprocal_unit_cell[:, 1], reciprocal_unit_cell[:, 2]
        beta = reciprocal_unit_cell[:, 3]
        areas = np.stack([a*b, a*c*np.sin(beta), b*c], axis=1)
    elif lattice_system == "orthorhombic":
        a, b, c = reciprocal_unit_cell[:, 0], reciprocal_unit_cell[:, 1], reciprocal_unit_cell[:, 2]
        areas = np.stack([a*b, a*c, b*c], axis=1)
    elif lattice_system in ("tetragonal", "hexagonal"):
        a, c = reciprocal_unit_cell[:, 0], reciprocal_unit_cell[:, 1]
        angle = np.pi/3 if lattice_system == "hexagonal" else np.pi/2
        areas = np.stack([a*a*np.sin(angle), a*c, a*c], axis=1)
    elif lattice_system == "rhombohedral":
        a, alpha = reciprocal_unit_cell[:, 0], reciprocal_unit_cell[:, 1]
        area = a*a*np.sin(alpha)
        areas = np.stack([area, area, area], axis=1)
    elif lattice_system == "cubic":
        a = reciprocal_unit_cell[:, 0]
        areas = np.stack([a*a, a*a, a*a], axis=1)
    return np.min(areas, axis=1)/reciprocal_volume**(2/3)


def get_g_min_werner(d_values, decimals):
    """Werner 1976's precision floor on the mean discrepancy, from decimal quantisation.

    If d values are reported to `decimals` places, a line at d cannot be located better than
    Delta = 0.25 * 10^(-decimals), so epsilon_i = |1/d^2 - 1/(d + Delta)^2| and g_min is their mean.

    Our q2 are full precision, so Werner's own floor does not bind for us; this exists to reproduce
    his Tables 1 and 2 and to document the construction. **The operational floor for this project's
    data is not yet chosen** -- the candidates are the peak-picking precision and the empirical
    reproducibility floor measured in S06 -- so get_V_over_Vcrit takes g_min as an argument rather
    than defaulting to anything.
    """
    step = 0.25*10.0**(-decimals)
    d_values = np.asarray(d_values, dtype=float)
    return float(np.mean(np.abs(1/d_values**2 - 1/(d_values + step)**2)))


def get_V_over_Vcrit(volume, d_n, g_min, multiplicity, threshold=10.0):
    """Werner 1976: is this comparison capable of discriminating at all?

        M_N,max ~ 3 m d_N / (8 pi g_min V)      V_crit ~ 3 m d_N / (8 pi g_min * threshold)

    Above V_crit, "figures of merit give information about rounding errors but not about the
    correctness of the trial cell". His worked case has V = 765.7 A^3 against V_crit = 226 A^3, and
    its M20 = 7 is explicitly an accidental effect.

    Note the power of d_N. Shirley's quotation of this formula OCRs as d_N^2; it is the first
    power, which follows from M_N = Q_N/(2 g N_N) with Q_N = 1/d_N^2 and N_N ~ (4 pi/3) d_N^-3 V/m.

    This gives a quantitative, published explanation of the pathology in the brief: a results list
    dominated by low-symmetry high-volume cells with M20 around 5-10 is a list of candidates
    sitting above V_crit, where the FOM was never applicable (F-017). V/V_crit is therefore a
    first-class quantity in its own right -- a stratification variable in S06 and a covariate in
    S07 -- not merely another FOM.

    Returns (V_over_Vcrit, M_max), each (n_candidates,).
    """
    v_crit = 3*multiplicity*d_n/(8*np.pi*g_min*threshold)
    m_max = 3*multiplicity*d_n/(8*np.pi*g_min*volume)
    return volume/v_crit, m_max


def get_M_werner_frac(merit, volume, d_n, g_min, multiplicity):
    """M_N / M_N,max: the FOM as a fraction of what the data precision allows.

    Precision-normalised and still sigma-free, because g_min is a floor on the *achievable*
    discrepancy rather than an error model for the measurement.
    """
    _, m_max = get_V_over_Vcrit(volume, d_n, g_min, multiplicity)
    return np.where(m_max > 0, merit/m_max, 0.0)


def get_F_N(q2_obs, q2_calc, q2_ref_calc, wavelength=None):
    """Smith & Snyder 1979: F_N = (1/|mean d(2theta)|) * (N / N_poss).

    The other classic FOM. Unlike M20 it is cumulative and defined for any pattern length, which is
    the property M20 lacks; Smith & Snyder recommend N = 30. It measures the quality of the *data*
    where M20 measures the reliability of the *model* (Shirley's reply to Snyder, F-012), so the
    two are complements rather than competitors.

    F_N is defined in 2theta and therefore needs a wavelength, which this pipeline frequently does
    not have -- a .npy peak file carries q2 only. Both forms are returned:

      F_N    in reciprocal degrees, computed only when `wavelength` is supplied, and directly
             comparable with published values. None otherwise.
      F_N_q  the same construction with |d(q)| in place of |d(2theta)|, q = sqrt(q2). Always
             available and **not comparable with published F_N values** -- it is in different
             units. Label it as such wherever it is reported.

    Shirley argues independently that F_N's wavelength dependence is a defect and that it should be
    standardised to Cu K-alpha-1; pass wavelength=1.540598 for that variant.

    Returns (F_N or None, F_N_q), each (n_candidates,).
    """
    n_peaks = q2_obs.shape[0]
    cutoff = q2_calc[:, -1]
    n_possible = np.maximum((q2_ref_calc < cutoff[:, np.newaxis]).sum(axis=1), 1)

    q_obs, q_calc = np.sqrt(q2_obs), np.sqrt(np.maximum(q2_calc, 0))
    mean_dq = np.mean(np.abs(q_obs[np.newaxis] - q_calc), axis=1)
    merit_q = np.where(mean_dq > 0, n_peaks/(mean_dq*n_possible), 0.0)

    merit_2theta = None
    if wavelength is not None:
        # 2 theta = 2 arcsin(lambda q / 2); q values beyond the Ewald limit are unobservable.
        argument_obs = np.clip(wavelength*q_obs/2, -1, 1)
        argument_calc = np.clip(wavelength*q_calc/2, -1, 1)
        two_theta_obs = np.degrees(2*np.arcsin(argument_obs))
        two_theta_calc = np.degrees(2*np.arcsin(argument_calc))
        mean_d2theta = np.mean(np.abs(two_theta_obs[np.newaxis] - two_theta_calc), axis=1)
        merit_2theta = np.where(mean_d2theta > 0, n_peaks/(mean_d2theta*n_possible), 0.0)
    return merit_2theta, merit_q


def estimate_sigma_entrywise(q2_obs, q2_calc_pool, quantile=0.1):
    """A per-entry, label-free estimate of the residual scale, from the best-fitting candidates.

    The idea (PLAN 2.5 treatment 2): whatever the instrument's error actually is, the candidates
    that fit this entry best cannot do better than it, so the residual scale of the leading tail of
    the candidate pool is an upper-bounded estimate of sigma that uses no external model and no
    labels. `quantile` selects how much of the pool counts as "best fitting".

    **This estimator is not yet validated.** PLAN 2.5 requires a calibration study before any
    sigma-hat is trusted, and Q7 is open. Anything built on it is labelled 'in-sample' and must be
    reported with a sigma-sensitivity curve.

    q2_calc_pool is (n_candidates, n_peaks) for one entry. Returns a scalar.
    """
    residual_scale = np.sqrt(np.mean((q2_obs[np.newaxis] - q2_calc_pool)**2, axis=1))
    cutoff = np.quantile(residual_scale, quantile)
    best = residual_scale[residual_scale <= cutoff]
    return float(np.mean(best)) if best.size else float(np.mean(residual_scale))


def get_chi2(q2_obs, q2_calc, lattice_system, sigma=None, variant="entrywise"):
    """Reduced chi-squared and its upper-tail p-value, in three explicitly labelled variants.

    dof = n - nu with nu the number of free cell parameters. Read PLAN 2.5 before using any of
    these: sigma is never known here, and each variant states what it does instead.

      'taupin'     Taupin's in-sample chi_r rescaling. With no a priori per-line error estimates
                   E_i to rescale, fitting the scale to the residuals makes the reduced chi-squared
                   identically 1 for every candidate, so it cannot rank. The informative output is
                   the fitted scale itself, which is returned as sigma_hat. Recorded as a negative
                   result rather than quietly dropped.
      'entrywise'  sigma from estimate_sigma_entrywise, a per-entry label-free estimator. Pass it
                   in via `sigma`. This is the variant with a chance of being useful, and it is the
                   one whose estimator needs validating (Q7).
      'fixed'      the repo's global model sigma(q2) = 0.00010 + 0.00058 q2. Included **only** as a
                   reference point to quantify what assuming sigma buys or costs. It is a median
                   over one instrument population fitted for *generating* data, and it is the most
                   insidious leakage path in the project (F-008), because the synthetic generator
                   uses this exact model. Never a deliverable on its own.

    Returns (chi2_reduced, p_value, sigma_hat), each (n_candidates,).
    """
    from scipy import stats

    n_peaks = q2_obs.shape[0]
    dof = max(n_peaks - N_CELL_PARAMETERS[lattice_system], 1)
    residual = q2_obs[np.newaxis] - q2_calc

    if variant == "fixed":
        scale = 0.00010 + 0.00058*q2_obs
        sigma_hat = np.full(q2_calc.shape[0], float(np.mean(scale)))
        chi2 = np.sum((residual/scale[np.newaxis])**2, axis=1)/dof
    elif variant == "taupin":
        sigma_hat = np.sqrt(np.sum(residual**2, axis=1)/dof)
        chi2 = np.ones(q2_calc.shape[0])
    elif variant == "entrywise":
        if sigma is None:
            raise ValueError("variant='entrywise' needs sigma from estimate_sigma_entrywise")
        sigma_hat = np.full(q2_calc.shape[0], float(sigma))
        chi2 = np.sum(residual**2, axis=1)/(sigma**2*dof)
    else:
        raise ValueError(f"unknown chi2 variant {variant!r}")
    return chi2, stats.chi2.sf(chi2*dof, dof), sigma_hat


def get_bic(q2_obs, q2_calc, xnn, lattice_system, bravais_lattice):
    """Bayesian information criterion with an explicit assignment-multiplicity penalty.

        BIC = -2 ln L + nu ln n + sum_i log(2 |dQ_i| dN_calc(q2_i))

    The last term is the log number of hkl assignments consistent with the data at each peak: a
    candidate that puts many calculated lines near an observation has many ways to fit it and
    should be charged for all of them. That is the complexity penalty M20 only implies, and it is
    the per-candidate counterpart of de Wolff's look-elsewhere argument (F-016).

    The line density dN_calc comes from get_delta_dewolff61 rather than from 4 pi q^2 V / mu, which
    was measured to under-count badly (S01_density_model.md). The likelihood uses an in-sample
    residual scale, so this is labelled 'in-sample' and not sigma-free; lower is better.

    Returns (n_candidates,).
    """
    n_peaks = q2_obs.shape[0]
    residual = q2_obs[np.newaxis] - q2_calc
    variance = np.maximum(np.mean(residual**2, axis=1), 1e-300)
    log_likelihood = -0.5*n_peaks*(np.log(2*np.pi*variance) + 1)

    delta = get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)
    # dN/dQ = 1/(2 Delta), so the number of assignments within |dQ| of the observation is
    # 2 |dQ| / (2 Delta). Floored at one: a peak always has at least the assignment it was given.
    assignments = np.maximum(np.abs(residual)/delta, 1.0)
    complexity = np.sum(np.log(assignments), axis=1)

    return -2*log_likelihood + N_CELL_PARAMETERS[lattice_system]*np.log(n_peaks) + complexity


# ---------------------------------------------------------------------------------------------
# S10: predictive figures of merit. Every merit above scores a candidate on the peaks it was
# fitted to; these two score it on peaks it was not. The distinction is the whole point -- a cell
# with six free parameters and a dense calculated spectrum can absorb any twenty peaks, and the
# question is whether it can then *predict* one it has not seen.
# ---------------------------------------------------------------------------------------------

# ln 2, the median of Exp(1). Under de Wolff's idealised null a held-out discrepancy is a free-path
# draw with mean Delta (de Wolff 1961 section 3), so median(|dQ|/Delta) = ln 2 and cv_M = 1 -- for
# every lattice, every cell size and every peak count. That is what makes it comparable across
# candidates without a fitted normalisation. tests/test_fom_cv.py asserts it on a construction that
# satisfies the null; on the benchmark's refined survivors it does not hold and is not expected to.
LOG_TWO = float(np.log(2.0))

# A held-out fold is voided when the retained peaks cannot determine the cell. This is not a
# nicety: gauss_newton_solve returns a *zero step* for a rank-deficient system, so an unvoided
# fold would silently report the full fit as its own refit and score perfectly.
CV_SCHEMES = ("random", "contiguous", "high_q")


def _cv_folds(n_peaks, n_folds, scheme, seed):
    """Which peaks are held out in each fold, as a list of index arrays.

    'random'      n_folds interleaved folds over a seeded permutation. Tests general predictive
                  accuracy. Every peak is held out exactly once.
    'contiguous'  n_folds contiguous blocks in ascending q2. Every peak is held out exactly once.
                  This exists to separate *contiguity* from *extrapolation* in the scheme below,
                  which would otherwise be confounded.
    'high_q'      the top block only, one fold. de Wolff's peaks are ascending in q2, so this is
                  the extrapolation test -- closest to the real failure, and the only scheme whose
                  out-of-fold sample is a block rather than the whole list.
    """
    order = np.arange(n_peaks)
    if scheme == "random":
        order = np.random.default_rng(seed).permutation(n_peaks)
        return [np.sort(order[start::n_folds]) for start in range(n_folds)]
    blocks = np.array_split(order, n_folds)
    if scheme == "contiguous":
        return list(blocks)
    if scheme == "high_q":
        return [blocks[-1]]
    raise ValueError(f"unknown cv scheme {scheme!r}, expected one of {CV_SCHEMES}")


def _refit_on_retained(q2_obs, xnn, hkl2_full, keep, pivot_tolerance=1e-12):
    """Weighted least-squares refit of xnn on the retained peaks, with the assignment frozen.

    Two things make this exact rather than iterative. The forward model q2 = hkl2 @ xnn is
    *linear*, so at fixed weights a single Gauss-Newton step from any starting point lands on the
    weighted optimum -- there is no convergence loop and no dependence on where it started.

    The weights are the part that has to be got right. CandidateOptLoss builds
    sigma = sqrt(q2 (|dQ| + eps)) from the residuals at its starting cell; carrying that over from
    the full fit would let the *held-out* peaks influence the refit through the weights, and would
    do so most for the cells with the most parameters -- which is exactly the effect being
    measured. So the weights are rebuilt here from the retained peaks alone: pass one uses the
    residual-free sigma_0 = sqrt(q2), pass two uses the pass-one retained residuals.

    The eps floor is relative rather than CandidateOptLoss's absolute 1e-10. With sixteen retained
    peaks, one peak landing on a calculated line would otherwise take a weight of ~1e12 and define
    the refit by itself.

    Returns (xnn_refit, ok), ok being False where the retained design is rank deficient.
    """
    from mlindex.utilities.numba_functions import gauss_newton_solve

    hkl2 = np.ascontiguousarray(hkl2_full[:, keep, :])
    q2 = np.ascontiguousarray(
        np.repeat(q2_obs[keep][np.newaxis], xnn.shape[0], axis=0)
    )

    sigma = np.ascontiguousarray(np.sqrt(np.maximum(q2, 0.0)) + 1e-300)
    step, ok = gauss_newton_solve(
        hkl2, q2, sigma, np.ascontiguousarray(xnn), pivot_tolerance
    )
    refit = xnn + step

    residual = np.abs(np.sum(hkl2 * refit[:, np.newaxis, :], axis=2) - q2)
    # Two floors, and both are needed. The relative one stops a single peak that happens to land on
    # a calculated line from taking a weight of ~1e12 and defining the refit by itself, which is
    # what CandidateOptLoss's absolute 1e-10 permits at sixteen retained peaks. The second stops
    # sigma reaching exactly zero when *every* residual does -- an exactly-fitting cell is not
    # hypothetical here (F-054's zero-error bundle) and 1/sigma^2 inside the kernel would divide
    # by zero for the whole candidate.
    floor = np.maximum(0.1*np.median(residual, axis=1, keepdims=True), 1e-12*np.maximum(q2, 0.0))
    sigma = np.ascontiguousarray(
        np.sqrt(np.maximum(q2, 0.0) * np.maximum(residual, floor)) + 1e-300
    )
    step, ok_second = gauss_newton_solve(
        hkl2, q2, sigma, np.ascontiguousarray(refit), pivot_tolerance
    )
    return refit + step, ok & ok_second


def _held_out_leverage(hkl2_full, keep, held):
    """Max leverage of the held-out peaks against the retained design, per candidate.

    h_i = x_i^T (X_ret^T X_ret)^-1 x_i, unweighted. For a retained point this is bounded by 1;
    for a held-out one it can exceed 1, and doing so means the prediction is an extrapolation
    rather than an interpolation.

    This is **reported, not used to void a fold.** The 'high_q' scheme holds out the top block on
    purpose, so its leverage is large by construction and voiding on it would delete the very
    measurement the scheme exists to make. It is a diagnostic for reading the result, and the
    contrast between 'contiguous' and 'high_q' is what it explains.
    """
    design = hkl2_full[:, keep, :]
    gram = np.matmul(np.swapaxes(design, 1, 2), design)
    out = hkl2_full[:, held, :]
    leverage = np.full(hkl2_full.shape[0], np.nan)
    try:
        solved = np.linalg.solve(
            gram + 1e-12*np.eye(gram.shape[-1])[np.newaxis], np.swapaxes(out, 1, 2)
        )
    except np.linalg.LinAlgError:
        return leverage
    values = np.sum(out * np.swapaxes(solved, 1, 2), axis=2)
    return np.max(values, axis=1)


def _dewolff_baseline(q2_ref_calc, cutoff):
    """Q_N/(2 N_cal): the expected discrepancy an arbitrary cell of this line density would give.

    get_M20's own baseline, lifted out so a held-out score can use it unchanged. `cutoff` is the
    calculated position of the line assigned to the *last observed* peak, which is what get_M20
    uses (`q2_calc[:, -1]`) -- taking it from the held-out peaks instead would make the baseline
    move with the fold. Returns (n_candidates,), zero where the candidate has no lines in range.

    The cut-off is **snapped onto the reference grid before it is used**, and that is what makes
    this exact rather than approximately right. The cut-off IS one of the reference lines -- the
    one the last observed peak was assigned -- so a bare `<` turns on whether that line reproduces
    itself to the last bit. It does not: get_M20 reaches it through `take_along_axis` on a matmul,
    while a caller holding only the assigned Miller indices reaches it through a sum over them,
    and the two differ by ~1e-16. Measured on 2.37M real candidates, that flipped N by one line
    for a handful of monoclinic cells and moved the merit by up to 1.8%. Snapping to the *nearest*
    reference line recovers the stored line itself -- the last peak's assignment is by definition
    the nearest line to its own calculated position -- so the comparison is get_M20's.
    """
    nearest = np.argmin(np.abs(q2_ref_calc - cutoff[:, np.newaxis]), axis=1)
    snapped = np.take_along_axis(q2_ref_calc, nearest[:, np.newaxis], axis=1)
    in_range = q2_ref_calc < snapped
    count = in_range.sum(axis=1)
    q_n = np.max(np.where(in_range, q2_ref_calc, 0.0), axis=1)
    expected = np.zeros(q2_ref_calc.shape[0])
    good = count > 0
    expected[good] = q_n[good]/(2*count[good])
    return expected


def _predictive_terms(q2_out, q2_assigned, xnn_used, lattice_system, bravais_lattice,
                      min_discrepancy=0.0):
    """|dQ| on the held-out peaks, and its ratio to the local calculated-line spacing there.

    Delta is evaluated at the cell that made the prediction -- the *refit* cell for the
    cross-validated form -- because that is the spectrum the held-out peak was assigned against.

    `min_discrepancy` floors |dQ| for the same reason it does in get_null_tail_nll: a held-out peak
    landing exactly on a calculated line sends the ratio to zero and the merit to infinity (F-026).
    Pass the resolution of the peak positions being scored.
    """
    discrepancy = np.maximum(
        np.abs(np.atleast_2d(q2_out) - q2_assigned), min_discrepancy
    )
    delta = get_delta_dewolff61(q2_out, xnn_used, lattice_system, bravais_lattice)
    return discrepancy, discrepancy/np.maximum(delta, 1e-300)


def _reduce_predictive(discrepancy, ratio, sigma_hat, prefix, expected=None):
    """The three normalisations the S10 handoff asks for, plus the one that transfers.

    Rows are candidates, columns are held-out peaks, NaN where a fold was voided.

      {p}_raw       median |dQ|. No normalisation at all, so it inherits M20's coincidence
                    problem -- a large cell with a dense spectrum always has a line nearby.
      {p}_chi2      median |dQ|/sigma_hat, sigma_hat from estimate_sigma_entrywise. **in-sample**.
      {p}_M         ln(2)/median(|dQ|/Delta). Keeps de Wolff's coincidence baseline while removing
                    the fitting advantage. Under de Wolff's *idealised* null -- an arbitrary cell
                    whose calculated lines are a Poisson process -- |dQ|/Delta is Exp(1), whose
                    median is ln 2, so this is 1 by construction for every lattice and cell size.
                    Real wrong candidates score above 1, for the two reasons the record already
                    names: they are refined survivors rather than arbitrary cells (R10, F-075) and
                    real calculated-Q sequences are more regular than exponential (F-015). How far
                    above is a measurement, not an assumption.
      {p}_M20       de Wolff's own statistic, computed on the held-out peaks: his global
                    Q_N/(2 N_cal) baseline over the *mean* held-out |dQ|. This is the merit the
                    project's baseline actually is, moved out of sample, and it exists because
                    {p}_M is not: {p}_M swaps the global baseline for the local Delta(Q) and the
                    mean for a median at the same time, so a difference against M20 could not be
                    attributed. With this column the two changes are separable.
      {p}_tail_nll  -sum log[1 - exp(-|dQ|/Delta)], the held-out counterpart of null_tail_nll.
                    Gamma(n_scored, 1) under the same null, which is what lets FomNull turn it
                    into a -log p that is comparable between a cubic candidate scored on ten peaks
                    and a triclinic one scored on twenty (R5).

    Larger is better for {p}_M and {p}_tail_nll; smaller is better for {p}_raw and {p}_chi2.
    """
    # Two masks, and keeping them apart matters. A peak has a *ratio* only where Delta(Q) is
    # finite, which fails when a refit cell comes out unphysical -- arccos of an out-of-range
    # argument -- while its *discrepancy* is perfectly well defined there. Sharing one mask made
    # the de Wolff column average over a different set of peaks from the one get_M20 averages
    # over, and the round trip came back 1.8e-2 instead of 1e-12.
    scored = np.isfinite(ratio)
    predicted = np.isfinite(discrepancy)
    n_scored = scored.sum(axis=1)
    n_predicted = predicted.sum(axis=1)
    good = n_scored > 0
    has_value = n_predicted > 0

    # A candidate whose every fold was voided has an all-NaN row, and np.nanmedian warns on one.
    # Fill those rows with zeros before reducing and discard the result afterwards, rather than
    # silencing the warning -- an all-NaN row that was *not* expected should still be visible.
    padded_ratio = np.where(scored, ratio, np.nan)
    padded_discrepancy = np.where(predicted, discrepancy, np.nan)
    padded_ratio[~good] = 0.0
    padded_discrepancy[~has_value] = 0.0
    median_ratio = np.where(good, np.nanmedian(padded_ratio, axis=1), np.nan)
    median_discrepancy = np.where(has_value, np.nanmedian(padded_discrepancy, axis=1), np.nan)
    tail = -np.sum(np.where(scored, np.log(1 - np.exp(-ratio) + 1e-100), 0.0), axis=1)

    # A perfect prediction -- every held-out peak exactly on a calculated line -- sends the merit
    # to infinity, exactly as M20 does when its mean discrepancy is zero (F-054). That is left as
    # an infinity rather than clipped, so it ranks first (which is correct) and is counted by
    # FomMetrics' n_non_finite_score diagnostic (which is how it stays visible). Callers with
    # rounded input pass min_discrepancy instead of relying on this.
    merit = np.zeros(ratio.shape[0])
    usable = good & np.isfinite(median_ratio)
    merit[usable] = np.where(
        median_ratio[usable] > 0, LOG_TWO/np.maximum(median_ratio[usable], 1e-300), np.inf
    )

    features = {
        f"{prefix}_raw": np.where(has_value, median_discrepancy, 0.0),
        f"{prefix}_M": merit,
        f"{prefix}_tail_nll": np.where(good, tail, 0.0),
        f"{prefix}_n_scored": n_scored.astype(float),
        f"{prefix}_n_predicted": n_predicted.astype(float),
    }
    if expected is not None:
        # get_M20's arithmetic exactly: a mean over the scored peaks, and the guard that returns
        # zero rather than dividing when a candidate's calculated lines have collapsed.
        padded_expected = np.where(predicted, expected, np.nan)
        padded_mean_discrepancy = np.where(predicted, discrepancy, np.nan)
        padded_expected[~has_value] = 0.0
        padded_mean_discrepancy[~has_value] = 0.0
        mean_discrepancy = np.where(has_value, np.nanmean(padded_mean_discrepancy, axis=1), np.nan)
        mean_expected = np.where(has_value, np.nanmean(padded_expected, axis=1), np.nan)
        dewolff = np.zeros(ratio.shape[0])
        usable_dewolff = has_value & np.isfinite(mean_discrepancy) & np.isfinite(mean_expected)
        dewolff[usable_dewolff] = np.where(
            mean_discrepancy[usable_dewolff] > 0,
            mean_expected[usable_dewolff]/np.maximum(mean_discrepancy[usable_dewolff], 1e-300),
            np.inf,
        )
        features[f"{prefix}_M20"] = dewolff
    if sigma_hat is not None and sigma_hat > 0:
        features[f"{prefix}_chi2"] = np.where(has_value, median_discrepancy/sigma_hat, 0.0)
    return features


def get_cv_fom(q2_obs, xnn, hkl, hkl_ref, lattice_system, bravais_lattice,
               scheme="random", n_folds=5, seed=12345, sigma_entrywise=None,
               min_discrepancy=0.0):
    """K-fold cross-validation inside the observed peak list: does this cell *predict*?

    Every classical figure of merit scores a candidate on the peaks it was refined against, so a
    cell with more free parameters is rewarded twice -- once for fitting better and once for
    having been able to. This measures the second effect directly and charges for it. Per fold:

      1. hold out a subset of the observed peaks;
      2. refit xnn on the rest, **with their existing Miller-index assignment frozen**. Re-assigning
         the retained peaks would let the cell chase the held-out ones implicitly, and there would
         be nothing left to measure;
      3. rebuild the calculated spectrum from the refit cell and assign **only the held-out peaks**
         to their nearest calculated line, which is what inference would do;
      4. score the discrepancy there.

    Each peak is held out exactly once (except under 'high_q'), so the per-peak terms are pooled
    across folds and reduced once rather than averaged twice. The reduction is a **median**, not a
    mean: a contaminant peak has no correct assignment under any cell and inflates the error for
    the true candidate as much as for a false one, and a median is the robustification inference
    can actually perform without knowing which peak is the contaminant.

    The prediction that validates the implementation is that the penalty scales with the number of
    free cell parameters (N_CELL_PARAMETERS): near-nil for cubic, largest for triclinic. If it does
    not scale, something is wrong -- most likely the assignment was not frozen, or folds are being
    voided silently.

    Arguments follow get_M20_from_xnn rather than get_M20: q2_obs is (n_peaks,), xnn is
    (n_candidates, n_free), hkl is (n_candidates, n_peaks, 3) as assigned by the full fit, and
    hkl_ref is (n_ref, 3) for the candidate's extinction group. Nothing is modified.

    Returns a dict of (n_candidates,) arrays; see _reduce_predictive for the columns.
    """
    q2_obs = np.asarray(q2_obs, dtype=np.float64)
    xnn = np.atleast_2d(np.asarray(xnn, dtype=np.float64))
    n_peaks = q2_obs.shape[0]
    n_candidates = xnn.shape[0]

    from mlindex.utilities.numba_functions import fast_assign

    hkl2_full = get_hkl_matrix(np.asarray(hkl), lattice_system)
    hkl2_ref = get_hkl_matrix(np.asarray(hkl_ref), lattice_system)

    discrepancy = np.full((n_candidates, n_peaks), np.nan)
    ratio = np.full((n_candidates, n_peaks), np.nan)
    expected = np.full((n_candidates, n_peaks), np.nan)
    voided = np.zeros(n_candidates)
    leverage = np.zeros(n_candidates)

    for held in _cv_folds(n_peaks, n_folds, scheme, seed):
        keep = np.setdiff1d(np.arange(n_peaks), held)
        if keep.size < hkl2_full.shape[2]:
            # Fewer retained peaks than free parameters: the refit is not defined at all.
            voided += held.size
            continue
        refit, ok = _refit_on_retained(q2_obs, xnn, hkl2_full, keep)
        ok = ok & np.all(np.isfinite(refit), axis=1)

        q2_ref_refit = np.matmul(refit, hkl2_ref.T)
        assign = fast_assign(q2_obs[held], q2_ref_refit)
        q2_assigned = np.take_along_axis(q2_ref_refit, assign, axis=1)

        fold_discrepancy, fold_ratio = _predictive_terms(
            q2_obs[held], q2_assigned, refit, lattice_system, bravais_lattice, min_discrepancy
        )
        fold_ratio = np.where(np.isfinite(fold_ratio), fold_ratio, np.nan)
        # de Wolff's baseline at the refit cell. The cut-off is the calculated position of the
        # line the *last observed* peak was assigned in the full fit, evaluated at the refit cell
        # -- get_M20's own q2_calc[:, -1], and no held-out information beyond what M20 uses.
        cutoff = np.sum(hkl2_full[:, -1, :]*refit, axis=1)
        fold_expected = _dewolff_baseline(q2_ref_refit, cutoff)
        discrepancy[np.ix_(ok, held)] = fold_discrepancy[ok]
        ratio[np.ix_(ok, held)] = fold_ratio[ok]
        expected[np.ix_(ok, held)] = fold_expected[ok][:, np.newaxis]
        voided += (~ok)*held.size
        leverage = np.maximum(leverage, _held_out_leverage(hkl2_full, keep, held))

    features = _reduce_predictive(discrepancy, ratio, sigma_entrywise, "cv", expected=expected)
    features["cv_n_voided"] = voided
    features["cv_max_leverage"] = leverage
    return features


def get_insample_fom(q2_obs, xnn, hkl, lattice_system, bravais_lattice,
                     q2_calc=None, q2_ref_calc=None, sigma_entrywise=None, min_discrepancy=0.0):
    """The same four statistics as get_cv_fom, computed on the peaks the cell WAS fitted to.

    This exists so the cross-validated numbers have an exactly comparable partner. `is_M/cv_M` is
    then a clean ratio -- same reduction, same normalisation, same estimator, differing only in
    whether the peak was in the fit -- and its scaling with the number of free cell parameters is
    S10's second acceptance condition. Comparing `cv_M` against M20 instead would confound the
    fitted/held-out question with de Wolff's global Q_N/(2N) baseline against the local Delta(Q).

    No refit and no re-assignment: the cell and the Miller indices are the ones the pipeline
    already produced, so `q2_calc` is just hkl2 @ xnn. Costs a fraction of one fold.

    `q2_calc` and `q2_ref_calc` are optional and buy one column: with them, `is_M20` is emitted,
    and it must reproduce the pipeline's stored M20 -- the round trip that proves the fold
    machinery scores the object the benchmark ranked.

    **Pass them; do not let this function derive them.** They are what `FomBenchmark.assign_lines`
    already returns, and taking `q2_calc` out of `q2_ref_calc` by the assignment is not the same
    float64 as summing hkl2 @ xnn over the assigned Miller indices. The difference is ~1e-16 and it
    does not matter to any merit here except the de Wolff one, where the cut-off IS a reference
    line: a 1e-16 shift moves a line across it, changes N by one, and moved M20 by up to 18% on
    the candidates where it fired.

    Returns a dict of (n_candidates,) arrays with the 'is' prefix.
    """
    q2_obs = np.asarray(q2_obs, dtype=np.float64)
    xnn = np.atleast_2d(np.asarray(xnn, dtype=np.float64))
    if q2_calc is None:
        hkl2 = get_hkl_matrix(np.asarray(hkl), lattice_system)
        q2_calc = np.sum(hkl2 * xnn[:, np.newaxis, :], axis=2)
    discrepancy, ratio = _predictive_terms(
        q2_obs, q2_calc, xnn, lattice_system, bravais_lattice, min_discrepancy
    )
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    features = _reduce_predictive(discrepancy, ratio, sigma_entrywise, "is")
    if q2_ref_calc is not None:
        # get_M20 itself, on the caller's own arrays. Not a reimplementation: this column's whole
        # job is to be comparable with the number the benchmark stored.
        features["is_M20"] = get_M20(q2_obs, q2_calc, np.asarray(q2_ref_calc).copy())
    return features


def get_holdout_fom(q2_obs_holdout, xnn, hkl_ref, lattice_system, bravais_lattice,
                    sigma_entrywise=None, min_discrepancy=0.0):
    """The literal hold-out: score the fitted cell on peaks beyond the window it was fitted to.

    This is approach 3 as the brief originally proposed it, and it needs no refit -- the cell was
    already refined against all of the peaks it was given, and these are not among them. Assign
    each extra peak to its nearest calculated line and score exactly as the cross-validated form
    does, so the two are directly comparable on the entries where both exist.

    Its limitations are real and were correctly anticipated. The extra peaks do not always exist;
    where they do they come from the high-q2 region, where peaks are broad, weak and overlapped;
    and on this benchmark they had to be re-synthesised, because Benchmark A stored only their
    count (STATUS section 7, R13). Implemented because it is the obvious thing and a referee will
    ask, not because it is expected to win.

    `q2_obs_holdout` is (n_holdout,), ascending. Returns a dict of (n_candidates,) arrays with the
    'ho' prefix.
    """
    q2_obs_holdout = np.asarray(q2_obs_holdout, dtype=np.float64)
    xnn = np.atleast_2d(np.asarray(xnn, dtype=np.float64))

    from mlindex.utilities.numba_functions import fast_assign

    hkl2_ref = get_hkl_matrix(np.asarray(hkl_ref), lattice_system)
    q2_ref_calc = np.matmul(xnn, hkl2_ref.T)
    assign = fast_assign(q2_obs_holdout, q2_ref_calc)
    q2_assigned = np.take_along_axis(q2_ref_calc, assign, axis=1)

    discrepancy, ratio = _predictive_terms(
        q2_obs_holdout, q2_assigned, xnn, lattice_system, bravais_lattice, min_discrepancy
    )
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    # The hold-out peaks lie *beyond* the fitted window, so the cut-off is the last of them rather
    # than the last fitted peak: N is the number of calculated lines the cell predicts out to the
    # point it is being asked about.
    expected = np.repeat(
        _dewolff_baseline(q2_ref_calc, np.full(xnn.shape[0], float(q2_obs_holdout[-1])))
        [:, np.newaxis], q2_obs_holdout.size, axis=1,
    )
    return _reduce_predictive(discrepancy, ratio, sigma_entrywise, "ho", expected=expected)


def compute_all(
    q2_obs,
    q2_calc,
    q2_ref_calc,
    xnn,
    lattice_system,
    bravais_lattice,
    wavelength=None,
    sigma_entrywise=None,
    g_min=None,
    min_discrepancy=0.0,
):
    """Every figure of merit in the zoo for one entry's candidate pool, as a tidy dict of arrays.

    S06 through S08 all want the whole vector at once, so this is the single entry point. Each key
    maps to an array of shape (n_candidates,); `sigma_treatment` maps each key to 'free',
    'in-sample' or 'assumed' so that a sigma-dependent column can never be read as sigma-free.

    **get_M20 is evaluated last, and on a copy.** It writes zeros into q2_ref_calc in place
    (FigureOfMerits.py:19) as part of its own arithmetic, which is fine for the inner loop -- it is
    performance-critical code and is deliberately left untouched -- but would silently corrupt
    every other FOM in this frame. Ordering it last and handing it a copy contains that, and
    tests/test_fom_literature.py asserts the frame is invariant to evaluation order.

    Optional arguments degrade gracefully: without `wavelength` the published-units F_N is omitted,
    without `sigma_entrywise` the entrywise chi-squared is omitted, and without `g_min` the Werner
    quantities are omitted, since this project has not yet chosen an operational precision floor.

    `min_discrepancy` floors |dQ| for the two information-type merits, which diverge when an
    observation lands exactly on a calculated line (F-026). Set it to the resolution of the peak
    positions being scored.
    """
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
    )
    reciprocal_volume = get_unit_cell_volume(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
    )
    volume = 1/np.maximum(reciprocal_volume, 1e-300)

    features = {}
    M_tilde, M_rev, M_sym = get_M_rev_sym(q2_obs, q2_calc, q2_ref_calc)
    features["M_tilde"] = M_tilde
    features["M_rev"] = M_rev
    features["M_sym"] = M_sym
    features["X_N"] = get_X_N(q2_obs, q2_calc, q2_ref_calc).astype(float)
    features["M_wu"] = get_M_wu(q2_obs, q2_calc, q2_ref_calc)
    features["M_star"] = get_M_star(q2_obs, q2_calc, volume, lattice_system)
    features["M_star_corrected"] = get_M_star(
        q2_obs, q2_calc, volume, lattice_system, corrected=True
    )
    features["M_1"] = get_M_1(q2_obs, q2_calc, q2_ref_calc)
    features["M_nn"] = get_M_nn(q2_obs, q2_calc, q2_ref_calc)
    features["M_info_clipped"] = get_M_info_clipped(
        q2_obs, q2_calc, xnn, lattice_system, bravais_lattice, min_discrepancy=min_discrepancy
    )
    features["nll_exponential"] = get_nll_exponential(
        q2_obs, q2_calc, xnn, lattice_system, bravais_lattice
    )
    features["null_tail_nll"] = get_null_tail_nll(
        q2_obs, q2_calc, xnn, lattice_system, bravais_lattice, min_discrepancy=min_discrepancy
    )
    features["bic"] = get_bic(q2_obs, q2_calc, xnn, lattice_system, bravais_lattice)
    n_over, max_gap = get_n_over(q2_obs, q2_calc, q2_ref_calc)
    features["n_over"] = n_over.astype(float)
    features["max_gap"] = max_gap.astype(float)
    features["zone_dominance"] = get_zone_dominance(xnn, lattice_system)
    features["N_cal"] = get_N_cal(
        q2_ref_calc, np.zeros(q2_calc.shape[0]), q2_calc[:, -1]
    )
    features["delta_dewolff61"] = np.mean(
        get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice), axis=1
    )
    features["n_dewolff61"] = get_n_dewolff61(
        q2_obs, xnn, lattice_system, bravais_lattice
    )[:, -1]

    merit_2theta, merit_q = get_F_N(q2_obs, q2_calc, q2_ref_calc, wavelength=wavelength)
    features["F_N_q"] = merit_q
    if merit_2theta is not None:
        features["F_N"] = merit_2theta

    chi2_taupin, p_taupin, sigma_taupin = get_chi2(
        q2_obs, q2_calc, lattice_system, variant="taupin"
    )
    features["chi2_taupin_scale"] = sigma_taupin
    chi2_fixed, p_fixed, _ = get_chi2(q2_obs, q2_calc, lattice_system, variant="fixed")
    features["chi2_fixed"] = chi2_fixed
    features["chi2_fixed_pvalue"] = p_fixed
    if sigma_entrywise is not None:
        chi2_entry, p_entry, _ = get_chi2(
            q2_obs, q2_calc, lattice_system, sigma=sigma_entrywise, variant="entrywise"
        )
        features["chi2_entrywise"] = chi2_entry
        features["chi2_entrywise_pvalue"] = p_entry

    if g_min is not None:
        d_n = 1/np.sqrt(np.maximum(q2_calc[:, -1], 1e-300))
        multiplicity = get_multiplicity_taupin88(bravais_lattice)[0]
        over_critical, m_max = get_V_over_Vcrit(volume, d_n, g_min, multiplicity)
        features["V_over_Vcrit"] = over_critical
        features["M_werner_max"] = m_max

    # Last, and on a copy -- see the docstring.
    features["M20"] = get_M20(q2_obs, q2_calc, q2_ref_calc.copy())
    if g_min is not None:
        features["M_werner_frac"] = np.where(
            features["M_werner_max"] > 0, features["M20"]/features["M_werner_max"], 0.0
        )

    return {
        "features": features,
        "sigma_treatment": {
            name: SIGMA_TREATMENT.get(name.split("_pvalue")[0].split("_scale")[0], "free")
            for name in features
        },
    }
