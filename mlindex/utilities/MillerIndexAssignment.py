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
    """Draw n_picks peaks per entry without replacement, weighted by p.

    A chosen entry is zeroed rather than deleted, which removes the two
    np.delete-and-reshape calls this loop used to make per pick. Because
    positions no longer shift, the position index is the original index and the
    separate ``choices`` bookkeeping array is unnecessary too. Measured 1.13x
    with bit-identical output (tools/repro_subsampling.py).

    Two behaviours here are load-bearing and easy to lose. The cumulative
    distribution is deliberately *not* renormalised after a pick, and the draw is
    compared as ``cumsum >= random_value`` with random_value in [0, 1). Once
    enough mass has been taken that the remaining total falls below the draw,
    nothing satisfies the comparison and the pick falls through -- in the old
    compacted array np.argmax returned 0, which meant the first *surviving*
    entry. A zeroed array must therefore repair that case explicitly, or it
    re-picks an already-chosen index; a randomised cross-check against the old
    implementation caught exactly that on 203 of 300 cases.
    """
    n_entries, n_choices = p.shape
    p = p.copy()
    rows = np.arange(n_entries)
    alive = np.ones((n_entries, n_choices), dtype=bool)
    chosen = np.zeros((n_entries, n_picks), dtype=int)
    for index in range(n_picks):
        cumsum = p.cumsum(axis=1)
        random_value = rng.random(n_entries)
        q = cumsum >= random_value[:, np.newaxis]
        chosen_indices = q.argmax(axis=1)
        # Fall-through repair: no entry satisfied the draw, or argmax landed on
        # an already-taken position (possible only for an exact-zero draw).
        needs_repair = ~q.any(axis=1) | ~alive[rows, chosen_indices]
        if needs_repair.any():
            chosen_indices = chosen_indices.copy()
            chosen_indices[needs_repair] = alive[needs_repair].argmax(axis=1)
        chosen[:, index] = chosen_indices
        p[rows, chosen_indices] = 0.0
        alive[rows, chosen_indices] = False
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
