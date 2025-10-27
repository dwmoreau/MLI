from numba import jit
import numpy as np


@jit(fastmath=True)
def fast_assign(q2_obs, q2_ref):
    n_obs = q2_obs.size
    n_candidates = q2_ref.shape[0]
    n_ref = q2_ref.shape[1]
    hkl_assign = np.zeros((n_candidates, n_obs), dtype=np.uint16)
    for candidate_index in range(n_candidates):
        for obs_index in range(n_obs):
            current_min = 100.0
            for ref_index in range(n_ref):
                diff = abs(q2_obs[obs_index] - q2_ref[candidate_index, ref_index])
                if diff < current_min:
                    current_min = diff
                    hkl_assign[candidate_index, obs_index] = ref_index
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
