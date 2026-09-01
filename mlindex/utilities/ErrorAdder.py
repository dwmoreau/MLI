import numpy as np

from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn


class ContaminantPlacementError(RuntimeError):
    pass


# Interior dropout is capped here, not left to the caller. Measured over the 5 955 S04 entries, the
# median entry carries 60 stored peaks -- `GenerateDataset` truncates there, so 60 is a cap and
# not a line count (C2-F-103); for this argument only the stored length matters, and it is what
# refill draws from. So the entry's own surplus almost never binds below 20 -- and at
# n_drop = 20 the mechanism removes 18.9 of the nominal 20 and refills from peaks 21-40. That is no
# longer punching holes in the low-angle window; it is translating the window wholesale to high
# angle, which is the distinct attack this function's docstring contrasts itself against. Ten keeps
# half the nominal window, so what the mechanism does still matches what it claims to do.
MAX_INTERIOR_DROPOUT = 10

# Redraw budget for the surplus contaminant placement below. `FomPatterns` carries its own copy for
# the window mechanism; this one is separate rather than imported because `FomPatterns` imports this
# module, and the surplus draw is small enough that it never approaches the cap in practice.
SURPLUS_CONTAMINANT_MAX_ATTEMPTS = 2000


def q2_sigma_params(intercept=None, slope=None):
    """The sigma(q2) = intercept + slope * q2 model, defaulting to the repository's own.

    Campaign 2 varies the error along two axes and no others (DWMM, 2026-08-26): the
    *severity*, which is the multiplier and scales both terms together, and the *shape*,
    which is the intercept alone and changes sigma at low q2 relative to high q2. A
    multiplier cannot reach the second, which is why the parameters are exposed at all.

    The law is Gaussian throughout. Heavier tails and a systematic 2-theta zero shift were
    considered and declined, so robustness to a different error *model* -- as opposed to a
    different error *magnitude* -- stays untested rather than passed. That bound is
    C2-R-008 and it is carried forward from campaign 1's R11/R12, not discharged.

    Note also that the defaults are fitted to the sealed benchmark: get_peak_generation_info
    labels them `#CNRS` and keeps the smSFX alternative commented out beside them. Keeping
    them is deliberate (campaign-1 comparability) and bounded by C2-R-007.

    Passing neither argument reproduces the pre-2026-08-26 behaviour exactly, including the
    number of RNG draws consumed downstream.
    """
    if intercept is None or slope is None:
        from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
        defaults = get_peak_generation_info()['q2_error_params']
        if intercept is None:
            intercept = defaults[0]
        if slope is None:
            slope = defaults[1]
    return float(intercept), float(slope)


def add_q2_error(q2, hkl, multiplier, rng, intercept=None, slope=None):
    # intercept and slope default to the repository's own sigma(q2) model, so an omitted
    # pair is bit-identical to the behaviour before they existed. See q2_sigma_params.
    intercept, slope = q2_sigma_params(intercept, slope)
    sigma_error = multiplier * (intercept + q2 * slope)
    q2 += rng.normal(loc=0, scale=sigma_error)
    q2 = np.abs(q2)
    if hkl is None:
        return np.sort(q2, axis=1)
    else:
        sort_indices = np.argsort(q2, axis=1)
        q2 = np.take_along_axis(q2, sort_indices, axis=1)
        hkl = np.take_along_axis(hkl, sort_indices[:, :, np.newaxis], axis=1)
        return q2, hkl


def select_peaks_with_dropout(q2_full, n_peaks, n_drop, rng):
    # Interior dropout: delete peaks from *within* the nominal low-q2 window and backfill from
    # higher q2, which is what undetected weak reflections do to a real pattern. This is a different
    # attack from pushing the whole window outwards -- it punches holes in the low-angle region,
    # where the systematic-absence pattern lives and where the generators take their information, so
    # it degrades candidate *discovery* rather than just the fit. n_drop = 0 is a no-op.
    #
    # n_drop is the number of holes left in the NOMINAL first n_peaks, capped by how many surplus
    # peaks the entry has to backfill with and by MAX_INTERIOR_DROPOUT: a 22-peak entry can give up
    # 2 and no more, a 26-peak entry can give up 6, and a 20-peak entry cannot give up any. The
    # returned list is always as long as the input allows, so the fixed-length ONNX generator input
    # is never violated (F-044).
    #
    # This changed on 2026-08-16. It used to draw the deletions from a window of n_peaks + n_drop,
    # so a fraction n_drop/(n_peaks + n_drop) of them landed in the backfill region and did nothing,
    # leaving n_drop * n_peaks/(n_peaks + n_drop) holes rather than n_drop. The rationale was that
    # every peak in the window then has the same detection probability. The cost was that a hole
    # count could not be requested: at n_peaks=20 a 26-peak entry saturated at 4.62 holes on average
    # and no value of n_drop reached 6, because raising it widened the window as fast as it added
    # draws. S04's C5 needs "as aggressive as the entry allows", so the parameter now means what it
    # says. Callers still record the achieved count, since the cap binds on thin entries.
    #
    # NOTE: S02's 18 `_drop` mechanism bundles were generated under the old semantics and their
    # n_dropout is not comparable with S04's C4/C5. They are exploratory bundles, not the frozen C1
    # nominal the split manifest is pinned to, so nothing downstream depends on the difference.
    q2_full = np.asarray(q2_full, dtype=float)
    q2_full = q2_full[q2_full > 0]
    if n_drop <= 0:
        return q2_full[:n_peaks]
    n_surplus = q2_full.size - n_peaks
    if n_surplus <= 0:
        # Nothing to backfill from, so dropping a peak would shorten the list. Return what there is
        # and let the caller record that the requested dropout was not achieved.
        return q2_full[:n_peaks]
    # Three caps. The surplus is what the entry can backfill with; MAX_INTERIOR_DROPOUT is the
    # mechanism's own ceiling, above which it stops being interior dropout at all; n_peaks is the
    # arithmetic floor under both, and binds only if a caller shortens the nominal window.
    n_holes = min(n_drop, n_surplus, MAX_INTERIOR_DROPOUT, n_peaks)
    dropped = rng.choice(n_peaks, size=n_holes, replace=False)
    kept = np.delete(q2_full[:n_peaks], dropped)
    backfill = q2_full[n_peaks:n_peaks + n_holes]
    return np.sort(np.concatenate((kept, backfill)))


def select_peaks_with_nested_dropout(q2_full, n_peaks, n_drop, rng, n_holdout=0,
                                    max_drop=None):
    """Interior dropout as a *nested* family, plus the surplus peaks left over.

    Two things this does that `select_peaks_with_dropout` does not, both required by campaign 2
    (DWMM, 2026-08-26).

    **Nesting.** The hole permutation is always drawn at `max_drop`, and the holes for a smaller
    `n_drop` are its prefix. So the N = 2 holes are a subset of the N = 4 holes, which are a
    subset of the N = 6 holes, and the sparsity axis becomes a paired comparison of one crystal
    degrading progressively rather than three independent noise realisations (C2-F-005). A
    prefix of `rng.choice(..., replace=False)` is a subset by construction, which is the whole
    trick.

    **A constant draw count.** Because the permutation is drawn at `max_drop` whatever `n_drop`
    is, this function consumes the same number of random values for every bundle -- it depends
    on the entry's own surplus and on nothing else. `select_peaks_with_dropout` consumes
    `n_holes`, so the hole count shifted every draw after it and a bundle differing only in
    sparsity got a different error realisation too. Campaign 2 additionally gives each mechanism
    its own sub-stream, so the two fixes are belt and braces; this one is what makes the
    *sparsity* axis itself nested.

    **The surplus.** `n_holdout` peaks beyond the fitted window are returned alongside it, so a
    caller can carry them through the same noise mechanisms and store them. Campaign 1 stored
    only their count and had to re-synthesise them from the true structure afterwards, giving
    lines that carry no contaminants while the fitted window does (R13). They start *after* the
    backfill, because the backfilled peaks are in the window.

    At `n_drop == max_drop` this agrees with `select_peaks_with_dropout` exactly: the same
    `rng.choice` call, hence the same holes and the same window.

    Returns (window, holdout, n_holes_achieved). The window is `n_peaks` long whenever the entry
    has any surplus, so the fixed-length ONNX generator input is never violated (F-044).
    """
    if max_drop is None:
        max_drop = n_drop
    if n_drop > max_drop:
        raise ValueError(f'n_drop {n_drop} exceeds max_drop {max_drop}; the ladder must be '
                         'drawn at its maximum for the nesting to hold')
    q2_full = np.asarray(q2_full, dtype=float)
    q2_full = q2_full[q2_full > 0]
    n_surplus = q2_full.size - n_peaks

    # Drawn unconditionally whenever the entry has surplus, including when n_drop is 0, so this
    # stream's position depends on the entry alone and never on which bundle is being generated.
    if n_surplus > 0:
        n_holes_max = min(max_drop, n_surplus, MAX_INTERIOR_DROPOUT, n_peaks)
    else:
        n_holes_max = 0
    if n_holes_max > 0:
        holes_all = rng.choice(n_peaks, size=n_holes_max, replace=False)
    else:
        holes_all = np.empty(0, dtype=int)

    n_holes = int(min(max(n_drop, 0), n_holes_max))
    holes = holes_all[:n_holes]
    if n_holes > 0:
        kept = np.delete(q2_full[:n_peaks], holes)
        backfill = q2_full[n_peaks:n_peaks + n_holes]
        window = np.sort(np.concatenate((kept, backfill)))
    else:
        window = q2_full[:n_peaks]
    # After the backfill, not after the nominal window: the backfilled lines are in the window.
    holdout = q2_full[n_peaks + n_holes:n_peaks + n_holes + n_holdout]
    return window, holdout, n_holes


def add_contaminants(q2, hkl, n_contaminants, rng, random_n_contaminants=False, max_attempts=None,
                     low_angle_bias=1.0, return_overflow=False):
    # return_overflow hands back the real peaks pushed out of the window by the
    # inserted lines, which this function otherwise computes and discards. They are
    # not noise: the window is re-truncated to n_peaks after insertion, so a
    # contaminant landing inside it *displaces* a real reflection, and that
    # reflection belongs in the hold-out set. Reconstructing it afterwards by set
    # difference is the kind of recompute campaign 1 kept paying for. Default False,
    # so the return arity is unchanged for every existing caller.
    # The whole contaminant set is redrawn until every member clears every peak's half breadth,
    # so acceptance falls off exponentially in n_contaminants and a dense pattern can spin
    # forever. max_attempts=None keeps that unbounded behaviour; an integer caps the redraws
    # and raises instead, so a caller sweeping many entries can drop the ones that cannot be
    # contaminated rather than hanging on them.
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
    q2_broadening_params = get_peak_generation_info()['broadening_params']
    # Breadth is specified as a linear model in q
    # Breadth in q^2 comes from error propagation
    breadth_q = q2_broadening_params[0] + q2_broadening_params[1] * np.sqrt(q2)
    breadth = 2 * breadth_q * np.sqrt(q2)
    n_peaks = q2.shape[1]
    overflow_q2 = []
    overflow_hkl = []
    for entry_index in range(q2.shape[0]):
        status = True
        n_attempts = 0
        while status:
            if not max_attempts is None and n_attempts >= max_attempts:
                raise ContaminantPlacementError(
                    f'Could not place {n_contaminants} contaminants in entry {entry_index} '
                    f'within {max_attempts} attempts'
                    )
            n_attempts += 1
            high = q2[entry_index, -1]
            if random_n_contaminants:
                n_contaminants_add = rng.choice(n_contaminants)
            else:
                n_contaminants_add = n_contaminants
            # low_angle_bias biases the draw towards low q2 via q2 = low + (high-low) * u**bias.
            # bias = 1 is the original uniform draw; bias = 2 is uniform in q rather than q2, which
            # is roughly where a second phase's visible lines sit. Measured over 400 mP entries, the
            # fraction of contaminants landing within the first five real peaks -- the region the
            # generators index from -- is 35% at bias 1, 50% at 1.5, 61% at 2 and 72% at 3, with the
            # median landing position moving from 47% of the q2 window to 8%.
            low = 0.5*q2[entry_index, 0]
            if low_angle_bias == 1.0:
                q2_contaminants = rng.uniform(low=low, high=high, size=n_contaminants_add)
            else:
                q2_contaminants = low + (high - low) * rng.uniform(
                    size=n_contaminants_add)**low_angle_bias
            if n_peaks is None:
                difference = np.abs(
                    q2_contaminants[np.newaxis]
                    - q2[entry_index][:, np.newaxis]
                    ).min(axis=0)
            else:
                difference = np.abs(
                    q2_contaminants[np.newaxis]
                    - q2[entry_index, :n_peaks][:, np.newaxis]
                    ).min(axis=0)
            status = np.any(difference[np.newaxis] < 0.5*breadth[entry_index][:, np.newaxis])

        q2_new = np.concatenate((q2[entry_index], q2_contaminants))
        if not hkl is None:
            hkl_new = np.concatenate(
                (hkl[entry_index], np.zeros((n_contaminants_add, 3))),
                axis=0
            )
        sort_indices = np.argsort(q2_new)
        q2[entry_index] = q2_new[sort_indices][:n_peaks]
        overflow_q2.append(q2_new[sort_indices][n_peaks:])
        if not hkl is None:
            hkl[entry_index, :, 0] = hkl_new[sort_indices, 0][:n_peaks]
            hkl[entry_index, :, 1] = hkl_new[sort_indices, 1][:n_peaks]
            hkl[entry_index, :, 2] = hkl_new[sort_indices, 2][:n_peaks]
            overflow_hkl.append(hkl_new[sort_indices][n_peaks:])
        else:
            overflow_hkl.append(None)
    if hkl is None:
        result = q2
    else:
        result = (q2, hkl)
    if return_overflow:
        return result, overflow_q2, overflow_hkl
    return result


def add_second_phase(q2, hkl, partner_q2, n_lines, rng, low_angle_bias=1.0,
                     return_overflow=False):
    # return_overflow: as add_contaminants. The injected lines displace real peaks out
    # of the window, and those peaks belong in the hold-out set rather than nowhere.
    # add_contaminants draws each contaminant at an independent random position. Real contamination
    # is a second crystalline phase, so its lines are *correlated* -- they are consistent with some
    # other lattice, and a handful of them arrive together. Uniform contaminants are therefore
    # easier to reject than real ones, which makes bundle C3 optimistic. This injects lines from an
    # actual second cell instead.
    #
    # Which lines, given that PROTOCOL section 3 rule 3 forbids intensities and the datasets carry
    # only positions: a real second phase shows its strong low-angle reflections, so the selection
    # is random but weighted towards low q2 rather than being the k lowest lines outright. The
    # weighting reuses add_contaminants' low_angle_bias convention as a rank draw over the eligible
    # lines -- index = floor(n_eligible * u**bias), bias 1 being a uniform pick -- so the parameter
    # means the same thing in both mechanisms.
    #
    # Placement is NOT add_contaminants' rejection loop, and deliberately so. That function draws
    # from a continuum, so redrawing until nothing collides is the only option. Here the partner
    # offers a finite, known set of lines, so the ones that collide with a real peak are filtered
    # out up front and the draw is made from what remains. Rejection sampling on a discrete set
    # degenerates: with two eligible lines and two to place there is exactly one possible set, and
    # redrawing it 2000 times cannot change the answer. That cost four of fifty-six entries on the
    # first gate run before this was rewritten -- a 7% loss that would have biased C6 towards
    # entries whose partner happened to fit.
    #
    # Injected lines enter hkl as (0, 0, 0) before the list is re-sorted and truncated back to
    # n_peaks, exactly as contaminants do.
    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
    q2_broadening_params = get_peak_generation_info()['broadening_params']
    breadth_q = q2_broadening_params[0] + q2_broadening_params[1] * np.sqrt(q2)
    breadth = 2 * breadth_q * np.sqrt(q2)
    n_peaks = q2.shape[1]
    partner_q2 = np.asarray(partner_q2, dtype=float)
    partner_q2 = np.sort(partner_q2[partner_q2 > 0])
    overflow_q2 = []
    overflow_hkl = []

    for entry_index in range(q2.shape[0]):
        low = 0.5*q2[entry_index, 0]
        high = q2[entry_index, -1]
        # Only the partner's lines that would actually fall inside the observed pattern. A line
        # above the last peak is not observable and one far below the first is outside the
        # measured range; these are the same bounds add_contaminants draws within.
        eligible = partner_q2[(partner_q2 >= low) & (partner_q2 <= high)]
        if eligible.size == 0:
            raise ContaminantPlacementError(
                f'The second phase has no lines inside entry {entry_index}\'s observed range '
                f'[{low:.4f}, {high:.4f}]'
                )
        # A line too close to a real peak would not be resolved as its own reflection, so it is
        # not an observable contaminant. This is add_contaminants' criterion exactly -- the
        # distance to the *nearest* peak against *every* peak's half breadth, so in effect the
        # largest half breadth in the pattern -- rather than the line's own neighbour's breadth,
        # which would be the more physical test. Matching matters more than improving it here:
        # C3 and C6 differ only in how their lines are chosen, and a different rejection rule
        # would confound that comparison.
        separation = np.abs(
            eligible[np.newaxis] - q2[entry_index, :n_peaks][:, np.newaxis]
            ).min(axis=0)
        placeable = eligible[
            ~np.any(separation[np.newaxis] < 0.5*breadth[entry_index][:, np.newaxis], axis=0)
            ]
        if placeable.size == 0:
            raise ContaminantPlacementError(
                f'Every one of the second phase\'s {eligible.size} lines in entry '
                f'{entry_index}\'s range overlaps one of its peaks'
                )
        n_add = min(n_lines, placeable.size)

        # Rank draw rather than a position draw: the partner offers a discrete set of real lines,
        # not a continuum. Without replacement, by redrawing index collisions -- which terminates
        # because n_add <= placeable.size and every index remains reachable.
        chosen = set()
        while len(chosen) < n_add:
            if low_angle_bias == 1.0:
                index = int(rng.integers(placeable.size))
            else:
                index = int(placeable.size * rng.uniform()**low_angle_bias)
            chosen.add(min(index, placeable.size - 1))
        q2_second_phase = placeable[sorted(chosen)]

        q2_new = np.concatenate((q2[entry_index], q2_second_phase))
        if not hkl is None:
            hkl_new = np.concatenate(
                (hkl[entry_index], np.zeros((n_add, 3))),
                axis=0
                )
        sort_indices = np.argsort(q2_new)
        q2[entry_index] = q2_new[sort_indices][:n_peaks]
        overflow_q2.append(q2_new[sort_indices][n_peaks:])
        if not hkl is None:
            hkl[entry_index, :, 0] = hkl_new[sort_indices, 0][:n_peaks]
            hkl[entry_index, :, 1] = hkl_new[sort_indices, 1][:n_peaks]
            hkl[entry_index, :, 2] = hkl_new[sort_indices, 2][:n_peaks]
            overflow_hkl.append(hkl_new[sort_indices][n_peaks:])
        else:
            overflow_hkl.append(None)
    if hkl is None:
        result = q2
    else:
        result = (q2, hkl)
    if return_overflow:
        return result, overflow_q2, overflow_hkl
    return result


def surplus_contaminant_rate(n_injected, n_window):
    """Contaminants per peak in the window, which is the rate the surplus should carry too.

    Taking the rate from the entry's own bundle rather than from a new parameter is what keeps a
    clean bundle clean: `cont0` gives exactly 0.0 and nothing is drawn. `cont1` over a 20-peak
    window gives 0.05, so a 20-line surplus expects one contaminant, which is DWMM's "maybe 1
    here and there" (2026-09-01).
    """
    if n_window <= 0:
        return 0.0
    return float(n_injected)/float(n_window)


def add_surplus_contaminants(q2_holdout, hkl_holdout, expected_count, rng,
                             max_attempts=SURPLUS_CONTAMINANT_MAX_ATTEMPTS):
    """Inject contaminant lines into the surplus peaks, which the generator cannot reach.

    **This exists because `add_contaminants` draws in `[0.5*q2[0], q2[-1]]` -- always below the
    *window* maximum -- so no contaminant is ever placed above the fitted window.** The only junk
    lines that reach the surplus are real contaminants *displaced* out of the window when the list
    is re-truncated to `n_peaks`, which is ~1 line per 22 patterns and only in the bundles that
    inject at all. A real pattern's high-q2 tail is where unindexed junk is *most* likely, not
    least, so the stored surplus is cleaner than reality and a hold-out merit measured on it is
    optimistic. DWMM, 2026-09-01: "absolutely add error. Maybe 1 contaminant here and there."

    **Applied offline, and that is sound rather than a shortcut.** The indexer only ever sees the
    20-peak window, so the candidate pool is conditioned on the window alone; the surplus is stored
    metadata that no candidate depends on. Contaminating it therefore changes not one row of the
    880 M-candidate benchmark and needs no regeneration. Anything that touched the *window* would.

    Draws Poisson(`expected_count`) lines uniformly across the surplus range, tags each `(0,0,0)`
    exactly as the window mechanisms do, and re-sorts. Unlike the displaced lines already present
    -- which all sit at surplus position 0, immediately above the truncated window edge -- these
    land anywhere in the surplus, which is both more realistic and more searching for a merit
    scored on a short prefix.

    The half-breadth rejection is the window mechanism's, so an injected line is never placed on
    top of a real one where no instrument could resolve the two. The whole set is redrawn on any
    rejection, as `add_contaminants` does; `n` is small here so that cannot spin the way the window
    version can, and an exhausted budget returns the surplus unchanged rather than raising, since
    this runs over millions of entries in an analysis pass.

    Returns `(q2_holdout, hkl_holdout, n_added)`. The inputs are not modified.
    """
    q2_holdout = np.asarray(q2_holdout, dtype=np.float64)
    hkl_holdout = np.asarray(hkl_holdout).reshape(-1, 3)
    if q2_holdout.size == 0 or expected_count <= 0:
        return q2_holdout, hkl_holdout, 0

    n_add = int(rng.poisson(expected_count))
    if n_add <= 0:
        return q2_holdout, hkl_holdout, 0

    from mlindex.dataset_generation.EntryHelpers import get_peak_generation_info
    q2_broadening_params = get_peak_generation_info()['broadening_params']
    # Breadth is a linear model in q; the q2 breadth follows by error propagation. Same two lines
    # as add_contaminants, deliberately -- a second breadth model here would be a second thing to
    # keep in step with the generator.
    breadth_q = q2_broadening_params[0] + q2_broadening_params[1]*np.sqrt(q2_holdout)
    breadth = 2*breadth_q*np.sqrt(q2_holdout)

    low, high = float(q2_holdout[0]), float(q2_holdout[-1])
    if not high > low:
        return q2_holdout, hkl_holdout, 0

    placed = None
    for _ in range(int(max_attempts)):
        draw = rng.uniform(low=low, high=high, size=n_add)
        separation = np.abs(draw[np.newaxis] - q2_holdout[:, np.newaxis]).min(axis=0)
        if not np.any(separation[np.newaxis] < 0.5*breadth[:, np.newaxis]):
            placed = draw
            break
    if placed is None:
        return q2_holdout, hkl_holdout, 0

    q2_new = np.concatenate([q2_holdout, placed])
    hkl_new = np.concatenate([hkl_holdout, np.zeros((n_add, 3), dtype=hkl_holdout.dtype)], axis=0)
    order = np.argsort(q2_new)
    return q2_new[order], hkl_new[order], n_add


def perturb_xnn(xnn_true, convergence_candidates, convergence_distances, minimum_uc, maximum_uc, lattice_system, rng):
    size = (convergence_candidates, xnn_true.size)
    perturbed_unit_cells = []
    for distance in convergence_distances:
        perturbations = rng.uniform(low=-1, high=1, size=size)
        perturbations = distance * perturbations / np.linalg.norm(perturbations, axis=1)[:, np.newaxis]
        perturbed_xnn = xnn_true[np.newaxis] + perturbations
        perturbed_xnn = fix_unphysical(
            xnn=perturbed_xnn,
            rng=rng,
            minimum_unit_cell=minimum_uc,
            maximum_unit_cell=maximum_uc,
            lattice_system=lattice_system
            )
        perturbed_unit_cells.append(get_unit_cell_from_xnn(
            perturbed_xnn,
            partial_unit_cell=True,
            lattice_system=lattice_system
            ))
    return np.concatenate(perturbed_unit_cells, axis=0)
