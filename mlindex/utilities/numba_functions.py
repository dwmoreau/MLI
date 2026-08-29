from numba import jit
import numpy as np


# No fastmath here, deliberately. Bare fastmath=True implies LLVM's 'nnan' and
# 'ninf', which license the compiler to assume NaN and Inf never occur. They do:
# q2_ref is xnn @ hkl2.T, and NaN xnn is common enough that
# _downsample_computation filters for it explicitly. Under fastmath the old
# implementation disagreed with strict IEEE on rows containing NaN, so the
# assignment for such a candidate was whatever the optimiser happened to emit.
# Measured across all ten captured input shapes, dropping fastmath costs at most
# 0.5% -- inside the benchmark's noise floor on every one of them
# (tools/bench_fast_assign.py). Determinism on degenerate input is worth far
# more than that here, so it is not a trade.
@jit
def fast_assign(q2_obs, q2_ref):
    # The single running minimum this loop used to keep is a serial dependency:
    # every element has to wait on the previous compare, which pinned the scan
    # near one element per ~1.8 cycles no matter how many execution ports were
    # free. Four accumulators over interleaved indices give the out-of-order
    # engine four independent chains; measured 1.13-1.22x on real captured
    # inputs (tools/bench_fast_assign.py), with bit-identical output.
    #
    # Lane l covers indices l, l+4, l+8, ... Within a lane indices ascend and
    # the update stays a strict '<', so each lane holds the lowest index
    # attaining its own minimum. The merge below prefers a strictly smaller
    # value, or an equal value with a smaller index, which reproduces the
    # original "first index wins ties" behaviour exactly.
    #
    # Variants that lost on the same inputs, so they are not worth retrying
    # without new evidence: sorting each row and binary searching (0.15x),
    # swapping the loop order (0.85x), block min/max pruning (0.82x), a
    # vectorizable min pass plus an index lookup pass (0.55x), holding the
    # index in a local instead of storing in the branch (0.65x -- the branch is
    # rarely taken, so prediction beats a conditional move), and eight lanes
    # instead of four (0.85x -- the lane arrays spill to memory).
    n_obs = q2_obs.size
    n_candidates = q2_ref.shape[0]
    n_ref = q2_ref.shape[1]
    limit = n_ref - (n_ref % 4)
    hkl_assign = np.zeros((n_candidates, n_obs), dtype=np.uint16)
    for candidate_index in range(n_candidates):
        row = q2_ref[candidate_index]
        for obs_index in range(n_obs):
            obs_value = q2_obs[obs_index]
            min0 = 100.0
            min1 = 100.0
            min2 = 100.0
            min3 = 100.0
            idx0 = 0
            idx1 = 0
            idx2 = 0
            idx3 = 0
            for ref_index in range(0, limit, 4):
                d0 = abs(obs_value - row[ref_index])
                d1 = abs(obs_value - row[ref_index + 1])
                d2 = abs(obs_value - row[ref_index + 2])
                d3 = abs(obs_value - row[ref_index + 3])
                if d0 < min0:
                    min0 = d0
                    idx0 = ref_index
                if d1 < min1:
                    min1 = d1
                    idx1 = ref_index + 1
                if d2 < min2:
                    min2 = d2
                    idx2 = ref_index + 2
                if d3 < min3:
                    min3 = d3
                    idx3 = ref_index + 3
            current_min = min0
            best = idx0
            if min1 < current_min or (min1 == current_min and idx1 < best):
                current_min = min1
                best = idx1
            if min2 < current_min or (min2 == current_min and idx2 < best):
                current_min = min2
                best = idx2
            if min3 < current_min or (min3 == current_min and idx3 < best):
                current_min = min3
                best = idx3
            for ref_index in range(limit, n_ref):
                diff = abs(obs_value - row[ref_index])
                if diff < current_min:
                    current_min = diff
                    best = ref_index
            # No reference line came within the 100.0 initial bound, which the
            # original expressed by never assigning and leaving the zero.
            if current_min >= 100.0:
                best = 0
            hkl_assign[candidate_index, obs_index] = best
    return hkl_assign


# De Wolff's M20 needs two reductions per candidate over the reference lines:
# how many fall below the last assigned line, and the largest of those. Done in
# numpy that is about five passes over an (n_candidates x n_ref) array -- 48 MB
# for monoclinic, built 1299 times a pattern -- plus two boolean temporaries the
# same size, one of them only to hold the negation of the other. One pass here
# does the same work: measured 2.37x on get_M20 across the eleven non-cubic
# lattices and 4.2-4.7% on a whole serial pattern, bit-identical.
#
# No fastmath, for the reason given above fast_assign: q2_ref is xnn @ hkl2.T and
# NaN rows are common enough that _downsample_computation filters for them.
# `value < cutoff` is False for NaN, which is what the numpy version's comparison
# did, so NaN lines are excluded from both the count and the maximum.
#
# The maximum starts at 0.0 rather than -inf on purpose. The numpy version zeroed
# the excluded entries in place and then took the maximum over the whole row, so
# a row with no line below the cut-off, or one whose lines below it are all
# negative, yields 0.0. Starting from -inf would change both cases.
@jit
def lines_below_cutoff(q2_ref_calc, cutoff):
    n_candidates = q2_ref_calc.shape[0]
    n_ref = q2_ref_calc.shape[1]
    counts = np.zeros(n_candidates, dtype=np.int64)
    largest = np.zeros(n_candidates, dtype=np.float64)
    for candidate_index in range(n_candidates):
        limit = cutoff[candidate_index]
        count = 0
        best = 0.0
        for ref_index in range(n_ref):
            value = q2_ref_calc[candidate_index, ref_index]
            if value < limit:
                count += 1
                if value > best:
                    best = value
        counts[candidate_index] = count
        largest[candidate_index] = best
    return counts, largest


@jit(fastmath=True)
def fast_assign_top_n(q2_obs, q2_ref, top_n):
    n_obs = q2_obs.size
    n_candidates = q2_ref.shape[0]
    n_ref = q2_ref.shape[1]
    hkl_assign = np.zeros((n_candidates, n_obs, top_n), dtype=np.uint16)
    for candidate_index in range(1):
        for obs_index in range(n_obs):
            current_min = [100.0 for _ in range(top_n)]
            current_min_index = [0 for _ in range(top_n)]
            for ref_index in range(n_ref):
                diff = abs(q2_obs[obs_index] - q2_ref[candidate_index, ref_index])
                # bisect.bisect_left could be used here, but it is not supported by numba
                status = True
                bisect_index = top_n - 1
                diff_index = top_n
                # Most reference peaks are far away, so look through array backwards
                while status:
                    if diff < current_min[bisect_index]:
                        diff_index = bisect_index
                    else:
                        status = False
                    bisect_index -= 1
                    if bisect_index < 0:
                        status = False
                if diff_index < top_n:
                    current_min.insert(diff_index, diff)
                    current_min.pop()
                    current_min_index.insert(diff_index, ref_index)
                    current_min_index.pop()
            hkl_assign[candidate_index, obs_index, :] = current_min_index
    return hkl_assign


# fastmath is deliberately restricted rather than True. Bare fastmath=True implies
# LLVM's 'nnan' and 'ninf', which license the compiler to assume NaN and Inf never
# occur -- and it duly folds the np.isnan guards below to False, so a NaN unit cell
# sails through and produces a non-finite step. This set keeps the arithmetic
# relaxations and drops only the two flags that would break the guards this kernel
# exists to provide.
_SAFE_FASTMATH = {'nsz', 'arcp', 'contract', 'afn', 'reassoc'}


@jit(fastmath=_SAFE_FASTMATH)
def gauss_newton_solve(hkl2, q2_obs, sigma, xnn, pivot_tolerance):
    """Per-candidate Gauss-Newton step: build, factorise and solve independently.

    Replaces a four-step numpy pipeline that built a (n, n_peaks, k, k)
    intermediate, tested invertibility with a full eigendecomposition, and then
    inverted. Measured 9.3-9.6x with peak working memory falling from 224 MB to
    1.3 MB on a 39,753-candidate chunk (tools/repro_hessian.py).

    Robustness is the main reason this exists, not speed. numpy's batched inv,
    solve and cholesky all raise for the *entire batch* if any single member is
    singular, and the caller's except clause then leaves every candidate with a
    zero step -- one degenerate candidate out of tens of thousands costing all of
    them their refinement. Worse, np.linalg.matrix_rank was called outside that
    try, and it raises on a non-finite Hessian, so a single bad sigma took the
    process down. Here every candidate is independent: a failure writes zeros for
    that one row, which is exactly what the old code intended for a
    non-invertible candidate, and its neighbours are untouched.

    H = J^T W J is symmetric positive semi-definite by construction, so a Cholesky
    is valid and much cheaper than an eigendecomposition. A pivot at or below
    pivot_tolerance * max_diagonal marks the candidate failed rather than
    producing a huge step -- the scale-relative test that matrix_rank was standing
    in for. Unlike matrix_rank it also rejects the near-singular matrices that
    used to pass the rank check and then blow up inside inv.

    Returns (delta_gn, ok) so callers can see how many candidates were skipped
    rather than learning about it from a printed message.
    """
    n = hkl2.shape[0]
    n_peaks = hkl2.shape[1]
    k = hkl2.shape[2]
    delta_gn = np.zeros((n, k))
    ok = np.zeros(n, dtype=np.bool_)

    H = np.zeros((k, k))
    L = np.zeros((k, k))
    gradient = np.zeros(k)
    y = np.zeros(k)

    for candidate in range(n):
        bad_input = False
        for component in range(k):
            if not np.isfinite(xnn[candidate, component]):
                bad_input = True
        if bad_input:
            continue

        for i in range(k):
            gradient[i] = 0.0
            for j in range(k):
                H[i, j] = 0.0

        for peak in range(n_peaks):
            s = sigma[candidate, peak]
            if s == 0.0 or not np.isfinite(s):
                bad_input = True
                break
            weight = 1.0 / (s * s)
            prediction = 0.0
            for component in range(k):
                prediction += hkl2[candidate, peak, component] * xnn[candidate, component]
            scaled_residual = (prediction - q2_obs[candidate, peak]) * weight
            for i in range(k):
                hi = hkl2[candidate, peak, i]
                gradient[i] += scaled_residual * hi
                for j in range(i + 1):
                    H[i, j] += weight * hi * hkl2[candidate, peak, j]
        if bad_input:
            continue

        largest_diagonal = 0.0
        for i in range(k):
            if H[i, i] > largest_diagonal:
                largest_diagonal = H[i, i]
        if not (largest_diagonal > 0.0):
            continue
        floor = pivot_tolerance * largest_diagonal

        singular = False
        for i in range(k):
            for j in range(i + 1):
                total = H[i, j]
                for m in range(j):
                    total -= L[i, m] * L[j, m]
                if i == j:
                    if not (total > floor):
                        singular = True
                        break
                    L[i, i] = np.sqrt(total)
                else:
                    L[i, j] = total / L[j, j]
            if singular:
                break
        if singular:
            continue

        for i in range(k):
            total = -gradient[i]
            for m in range(i):
                total -= L[i, m] * y[m]
            y[i] = total / L[i, i]
        for i in range(k - 1, -1, -1):
            total = y[i]
            for m in range(i + 1, k):
                total -= L[m, i] * delta_gn[candidate, m]
            delta_gn[candidate, i] = total / L[i, i]

        finite = True
        for i in range(k):
            if not np.isfinite(delta_gn[candidate, i]):
                finite = False
        if not finite:
            for i in range(k):
                delta_gn[candidate, i] = 0.0
            continue
        ok[candidate] = True

    return delta_gn, ok


# ---------------------------------------------------------------------------------------
# Shared by every kernel below that has to score a calculated line against the peak list.
# ---------------------------------------------------------------------------------------

@jit
def nearest_peak_distance(q2_obs_sorted, value):
    """min_j |value - q2_obs_sorted[j]|, by binary search rather than by a scan.

    |value - peak| is a V over a sorted peak list, so the minimum is attained at one of the
    two peaks bracketing `value` and the other n_peaks - 2 need never be touched. Both
    candidate distances are formed by the same subtract-and-abs a full scan would use, so
    what comes back is the same float and not merely the same number -- which is what lets
    the callers claim bit-identity with the array expressions they replaced.

    The search is branchless: its trip count depends only on n_peaks, so the predictor has
    nothing to miss. The textbook `while left < right` form mispredicts about half its
    compares and measured 65.8 ms against this one's 41.2 ms inside `reversed_line_scores`
    on a 10,000-candidate capture.

    `q2_obs_sorted` must be sorted ascending and finite; callers check and fall back to the
    array expression when it is not.
    """
    n_peaks = q2_obs_sorted.size
    left = 0
    width = n_peaks
    while width > 1:
        half = width >> 1
        left += half*(q2_obs_sorted[left + half - 1] < value)
        width -= half
    left += q2_obs_sorted[left] < value

    below = left - 1
    if below < 0:
        below = 0
    above = left
    if above > n_peaks - 1:
        above = n_peaks - 1
    distance_below = abs(value - q2_obs_sorted[below])
    distance_above = abs(value - q2_obs_sorted[above])
    # One value selected by a conditional expression, not two stores in two branches:
    # which of the pair is nearer is a coin flip, so the branch form mispredicts half the
    # time and measured 2x slower.
    return distance_above if distance_above < distance_below else distance_below


# Same no-fastmath rule as fast_assign, and for the same reason: q2_ref is xnn @ hkl2.T,
# NaN rows are routine, and the whole point of this kernel is that it agrees with the
# array expression it replaced to the last bit.
@jit
def reversed_line_scores(q2_ref, q_max, q2_obs_sorted, first_peak,
                         scored, in_range, q_n, counts, q_min):
    """Everything get_M_rev_sym's reversed term needs, in one row-local pass.

    Fills, per candidate row of ``q2_ref``:

      q_min      the reference line closest to the first observed peak (de Wolff's q_I);
      in_range   whether each reference line lies in [q_min, q_max];
      counts     how many do, which is N_cal when the weights are all 1;
      q_n        the largest reference line in range, or -inf if none is;
      scored     min_j |q2_ref - q2_obs[j]| where in range and 0.0 where not.

    The row is read twice -- once to locate q_min, once to score against it -- but a row
    is n_ref*8 bytes, so the second read comes out of cache rather than out of memory.
    Nothing here reassociates a float addition: every value written is produced by the
    same subtract-and-abs as the array expression, and the row sums stay in numpy, so
    the result is bit-identical rather than merely close. See tools/repro_msym.py.
    """
    n_candidates, n_ref = q2_ref.shape
    n_peaks = q2_obs_sorted.size
    for candidate in range(n_candidates):
        # np.argmin returns the first occurrence of the minimum, so this scans forwards
        # and updates on a strict '<'.
        best_index = 0
        best_deviation = np.inf
        for reference in range(n_ref):
            deviation = abs(q2_ref[candidate, reference] - first_peak)
            if deviation < best_deviation:
                best_deviation = deviation
                best_index = reference
        lower_bound = q2_ref[candidate, best_index]
        q_min[candidate] = lower_bound
        upper_bound = q_max[candidate]

        count = 0
        highest = -np.inf
        for reference in range(n_ref):
            value = q2_ref[candidate, reference]
            # NaN fails both comparisons, which is what the array expression does too.
            if value >= lower_bound and value <= upper_bound:
                in_range[candidate, reference] = True
                count += 1
                if value > highest:
                    highest = value

                scored[candidate, reference] = nearest_peak_distance(q2_obs_sorted, value)
            else:
                in_range[candidate, reference] = False
                scored[candidate, reference] = 0.0
        counts[candidate] = count
        q_n[candidate] = highest


@jit
def sorted_line_gap_squares(lines, counts, scaled_gaps, q_n):
    """Wu 1988's g_n term: the squared spacing between consecutive calculated lines.

    `lines` is one row per candidate, the in-range calculated lines sorted ascending with
    the rest at +inf, and `counts` how many are in range -- what `_sorted_lines_in_range`
    returns. Fills `scaled_gaps` with (L_k - L_(k-1))^2 / 4, the k = 1 interval running
    from 0, and 0.0 past the end of the row; and `q_n` with the largest in-range line.

    `scaled_gaps` is filled at full row width and summed by numpy rather than accumulated
    here, because numpy's pairwise summation groups a full row differently from a short one
    even though every entry past `counts` is an exact zero. The zeros are load-bearing.
    """
    n_candidates, n_ref = lines.shape
    for candidate in range(n_candidates):
        count = counts[candidate]
        previous = 0.0
        highest = -np.inf
        for line in range(count):
            value = lines[candidate, line]
            gap = value - previous
            previous = value
            scaled_gaps[candidate, line] = gap*gap/4
            if value > highest:
                highest = value
        for line in range(count, n_ref):
            scaled_gaps[candidate, line] = 0.0
        # The out-of-range entries the array expression maxed over were zeros, so they
        # raise the maximum whenever every in-range line is negative -- or there are none.
        if count < n_ref and highest < 0.0:
            highest = 0.0
        q_n[candidate] = highest


@jit
def over_prediction_runs(lines, counts, q2_obs_sorted, tolerance_factor, n_over, max_gap):
    """Calculated lines no observation accounts for, and the longest consecutive run of them.

    Same inputs as `sorted_line_gap_squares`. A line counts as unaccounted for when the
    nearest observed peak is further away than `tolerance_factor` times the gap to the
    previous line -- so the whole test is local, and one forward pass over the sorted row
    computes the gap, the nearest peak, the count and the run length together.

    Both outputs are integers, so unlike the merits that sum floats there is no grouping to
    preserve here: reproducing the boolean `unaccounted` per line reproduces both exactly.
    """
    n_candidates = lines.shape[0]
    for candidate in range(n_candidates):
        count = counts[candidate]
        previous = 0.0
        unaccounted = 0
        run = 0
        longest = 0
        for line in range(count):
            value = lines[candidate, line]
            local_gap = value - previous
            previous = value
            if nearest_peak_distance(q2_obs_sorted, value) > tolerance_factor*local_gap:
                unaccounted += 1
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
        n_over[candidate] = unaccounted
        max_gap[candidate] = longest


@jit
def bracketing_line_indices(lines, q2_obs, upper_index):
    """Per candidate row, `np.searchsorted(lines[row], q2_obs)` for every observed peak.

    Replaces a Python loop that called numpy's searchsorted once per candidate row. `lines`
    is row-sorted with +inf padding, so the insertion point is the index of the first line
    at or above the peak, which is exactly what Shirley's bracketing pair needs.

    `q2_obs` is *not* required to be sorted -- each peak is searched on its own, as in the
    expression this replaces. The search is a plain lower_bound, matching side='left': it
    steps on a strict `<`, so an exact hit returns the index of the line itself.
    """
    n_candidates, n_lines = lines.shape
    n_peaks = q2_obs.size
    for candidate in range(n_candidates):
        for peak in range(n_peaks):
            value = q2_obs[peak]
            left = 0
            right = n_lines
            while left < right:
                middle = (left + right)//2
                if lines[candidate, middle] < value:
                    left = middle + 1
                else:
                    right = middle
            upper_index[candidate, peak] = left


@jit
def nearest_line_distances(q2_obs, q2_ref, d1):
    """d1[i, p] = min_j |q2_ref[i, j] - q2_obs[p]|, the nearest calculated line to each peak.

    Taupin's reduced chi-square and the assignment posterior are both built on this, and it
    was n_peaks separate passes over the reference array -- one per peak, each materialising
    a (chunk, n_ref) temporary. One pass carrying n_peaks running minima reads the row once
    instead of twenty times, which is the whole difference: the row is 8 KB and the second
    read would have come out of memory.

    A minimum is a selection, not an accumulation, so scan order cannot change the answer:
    the value written is the same float either way, and `abs(line - peak)` is formed in the
    same order the array expression used.

    **NaN has to be propagated deliberately**, which a running minimum does not do on its own:
    `NaN < running` is false, so a naive scan silently skips it and returns the minimum of the
    rest, where `.min()` returns NaN. A NaN reference line makes every peak's distance NaN, so
    one test per reference entry catches it -- n_peaks times cheaper than testing the distance
    -- and a NaN *peak* is a whole column, which the caller fills in. Cost of the test, measured
    on a 1,000-candidate pool capture: under 5%.
    """
    n_candidates, n_ref = q2_ref.shape
    n_peaks = q2_obs.size
    running = np.empty(n_peaks, dtype=np.float64)
    for candidate in range(n_candidates):
        for peak in range(n_peaks):
            running[peak] = np.inf
        undefined = False
        for reference in range(n_ref):
            value = q2_ref[candidate, reference]
            if value != value:
                undefined = True
            else:
                for peak in range(n_peaks):
                    distance = abs(value - q2_obs[peak])
                    if distance < running[peak]:
                        running[peak] = distance
        for peak in range(n_peaks):
            d1[candidate, peak] = np.nan if undefined else running[peak]


# np.exp returns exactly 0.0 for arguments at or below -745.1332191019412 -- the first
# nonzero result is the smallest subnormal, 5e-324. Anything below this bound therefore
# contributes an exact zero to the sum, so it can be filled in rather than computed. The
# bound is deliberately a little under the true one: it has to be safe, not tight, and no
# term between the two is skipped.
EXP_UNDERFLOW_BOUND = -746.0


@jit
def posterior_exponent_terms(q2_ref_block, q2_obs_peak, d1_peak, block_scale, terms, computable):
    """The log-sum-exp arguments for one peak, with the terms that underflow already zeroed.

    Fills, for every calculated line of every candidate in the block:

      terms       -(|line - peak|^2 - d1^2)/scale where np.exp of that is not identically
                  zero, and 0.0 where it is -- so the caller runs `np.exp(terms, out=terms,
                  where=computable)` and the zeros are already in place;
      computable  which entries the exponential still has to be taken of.

    On a real pool 98.8% of the entries fall below the bound: sigma is estimated from the
    candidate's own residuals, so for anything that fits at all the competing lines are many
    sigma away. The exponential is *not* what dominates, though -- fusing the four array
    passes that built its argument is worth more than skipping it (120.7 ms -> 19.0 ms on a
    1,000-candidate pool capture, of which skipping the exponential is 3.7 ms).

    Only multiplies, subtracts, divides and compares here, all exact under IEEE and all
    performed in the same order as the array expression, so the argument handed to numpy's
    exp is the same float it was handed before. The exponential itself stays in numpy: numba
    would call a different libm and the last bit is not guaranteed to agree.
    """
    n_candidates, n_ref = q2_ref_block.shape
    for candidate in range(n_candidates):
        scale = block_scale[candidate]
        nearest_squared = d1_peak[candidate]*d1_peak[candidate]
        for reference in range(n_ref):
            distance = abs(q2_ref_block[candidate, reference] - q2_obs_peak)
            value = -(distance*distance - nearest_squared)/scale
            # Written as `<=` rather than `>` so that NaN falls through to the branch that
            # keeps it: numpy's exp turns it into a NaN term and the sum propagates it, which
            # is what the expression this replaces did with a NaN peak or reference line.
            if value <= EXP_UNDERFLOW_BOUND:
                terms[candidate, reference] = 0.0
                computable[candidate, reference] = False
            else:
                terms[candidate, reference] = value
                computable[candidate, reference] = True
