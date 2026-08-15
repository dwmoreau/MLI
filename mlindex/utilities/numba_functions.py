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
