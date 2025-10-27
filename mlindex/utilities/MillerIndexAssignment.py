import numpy as np


def vectorized_resampling(softmaxes, rng):
    # This is a major performance bottleneck

    # This function randomly resamples the peaks using the algorithm
    #  1: Pick a peak at random
    #  2: Assign Miller index according to softmaxes
    #  3: Set the assigned Miller index softmax to zero for all other peaks
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]

    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    random_values = rng.random(size=(n_entries, n_peaks))
    point_order = rng.permutation(n_peaks)
    softmaxes_zeroed = softmaxes.copy()
    i = 0
    for point_index in point_order:
        # Fast random selection:
        #  1: make cummulative sum along the distribution's axis (this is a cdf)
        #  2: selection is the first point in cummulative sum greater than random value
        #    - fastest way to do this, convert to bool array and find first True with argmax
        #    - To account for adding zeros to the softmax array, the random values are scaled
        #      instead of scaling the softmax array

        # This line is slow (60% of execution time)
        cumsum = np.cumsum(softmaxes_zeroed[:, point_index, :], axis=1)
        q = cumsum >= (random_values[:, point_index] * cumsum[:, -1])[:, np.newaxis]
        hkl_assign[:, point_index] = np.argmax(q, axis=1)
        i += 1
        if i < n_peaks:
            np.put_along_axis(
                softmaxes_zeroed,
                hkl_assign[:, point_index][:, np.newaxis, np.newaxis],
                values=0,
                axis=2,
            )

    softmax = np.take_along_axis(softmaxes, hkl_assign[:, :, np.newaxis], axis=2)[
        :, :, 0
    ]
    return hkl_assign, softmax


def vectorized_subsampling(p, n_picks, rng):
    n_entries = p.shape[0]
    n_choices = p.shape[1]
    choices = np.repeat(np.arange(n_choices)[np.newaxis], repeats=n_entries, axis=0)
    chosen = np.zeros((n_entries, n_picks), dtype=int)
    for index in range(n_picks):
        # cumsum: n_entries, n_peaks
        # random_value: n_entries
        # q: n_entries, n_peaks
        n_peaks = p.shape[1]
        cumsum = p.cumsum(axis=1)
        random_value = rng.random(n_entries)
        q = cumsum >= random_value[:, np.newaxis]
        chosen_indices = q.argmax(axis=1)
        chosen[:, index] = choices[np.arange(n_entries), chosen_indices]
        p_flat = p.ravel()
        choices_flat = choices.ravel()
        delete_indices = np.arange(n_entries) * n_peaks + chosen_indices
        p = np.delete(p_flat, delete_indices).reshape((n_entries, n_peaks - 1))
        choices = np.delete(choices_flat, delete_indices).reshape(
            (n_entries, n_peaks - 1)
        )
    chosen = np.sort(chosen, axis=1)
    return chosen


def best_assign_nocommon_original(softmaxes):
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)

    peak_choice = np.argsort(np.max(softmaxes, axis=2), axis=1)
    for candidate_index in range(n_entries):
        softmaxes_zeroed = softmaxes[candidate_index].copy()
        for peak_index in peak_choice[candidate_index]:
            choice = np.argmax(softmaxes_zeroed[peak_index, :])
            hkl_assign[candidate_index, peak_index] = choice
            softmaxes_zeroed[:, hkl_assign[candidate_index, peak_index]] = 0

    softmax_assign = np.take_along_axis(softmaxes, hkl_assign[:, :, np.newaxis], axis=2)
    return hkl_assign, softmax_assign


def best_assign_nocommon(softmaxes):
    # This is three times faster than the version above.
    # It picks the first occurance as opposed to the best occurance.
    n_entries = softmaxes.shape[0]
    n_peaks = softmaxes.shape[1]
    hkl_assign = np.zeros((n_entries, n_peaks), dtype=int)
    softmax_assign = np.zeros((n_entries, n_peaks))
    for peak_index in range(n_peaks):
        softmaxes_peak = softmaxes[:, peak_index, :]
        hkl_assign[:, peak_index] = np.argmax(softmaxes_peak, axis=1)
        softmax_assign[:, peak_index] = np.take_along_axis(
            softmaxes_peak, hkl_assign[:, peak_index][:, np.newaxis], axis=1
        )[:, 0]
        np.put(softmaxes, hkl_assign[:, np.newaxis, :], 0)
    return hkl_assign, softmax_assign


def assign_hkl_triplets(triplets_obs, hkl_assign, triplet_hkl_ref, q2_ref_calc):
    top_n = hkl_assign.shape[2]
    n_candidates = hkl_assign.shape[0]
    n_triplets = triplets_obs.shape[0]
    hkl_assign_triplets = np.zeros((n_candidates, n_triplets), dtype=np.uint16)
    for candidate_index in range(n_candidates):
        hkl_assign_candidate = hkl_assign[candidate_index]
        q2_ref_calc_candidate = q2_ref_calc[candidate_index]
        for triplet_index in range(n_triplets):
            triplet_loop = triplets_obs[triplet_index]
            hkl_assign_0_top_n = hkl_assign_candidate[int(triplet_loop[0])]
            hkl_assign_1_top_n = hkl_assign_candidate[int(triplet_loop[1])]
            hkl_assign_pair = []
            for top_n_index_0 in range(top_n):
                hkl_assign_0 = hkl_assign_0_top_n[top_n_index_0]
                for top_n_index_1 in range(top_n):
                    hkl_assign_1 = hkl_assign_1_top_n[top_n_index_1]
                    if hkl_assign_0 < hkl_assign_1:
                        hkl_assign_pair += triplet_hkl_ref[hkl_assign_0][hkl_assign_1]
                    elif hkl_assign_0 > hkl_assign_1:
                        hkl_assign_pair += triplet_hkl_ref[hkl_assign_1][hkl_assign_0]
            if len(hkl_assign_pair) > 0:
                diff = np.abs(triplet_loop[2] - q2_ref_calc_candidate[hkl_assign_pair])
                min_index = np.argmin(diff)
                hkl_assign_triplets[candidate_index, triplet_index] = hkl_assign_pair[
                    min_index
                ]
    return hkl_assign_triplets
