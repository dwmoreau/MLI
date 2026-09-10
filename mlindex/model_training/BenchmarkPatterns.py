"""Synthesising one benchmark pattern: a source crystal plus one condition bundle.

The mechanisms are applied in the order a real pattern acquires them -- some reflections are never
detected, the instrument adds a random error to the ones that are, contaminant lines are placed
relative to the peaks as observed, and a second phase is present in the sample rather than added by
the measurement.

Each mechanism draws from its own stream, keyed on the entry identifier, so the axes are
independent: a bundle that differs only in its contaminant count does not thereby receive a
different error realisation. Every seed derives from the entry id, so one entry gets the same noise
in every bundle and any subset of the benchmark regenerates identically -- which is what lets a
run be split across processes, resumed, or re-run over a handful of crystals.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import mlindex
from mlindex.model_training import BenchmarkConditions
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import add_second_phase
from mlindex.utilities.ErrorAdder import select_peaks_with_nested_dropout


# One instrument. Every optimizer factory interpolates this tag into its model directory name and
# only the `*_1` model set exists, so a run at another tag fails at model load.
BROADENING_TAG = '1'

N_PEAKS = 20

# A contaminant draw is rejected unless every line clears every peak's half breadth, and the whole
# set is redrawn on any rejection, so acceptance decays exponentially in the count. Uncapped that
# can spin forever on a crowded pattern; the cap turns it into a recorded skip.
CONTAMINANT_MAX_ATTEMPTS = 2000

DATASET_DIRECTORY = Path(mlindex.__path__[0]) / 'data' / 'generated_datasets'

READ_COLUMNS = [
    'identifier',
    'database',
    'bravais_lattice',
    'lattice_system',
    'train',
    'reindexed_unit_cell',
    'reindexed_volume',
    'reindexed_spacegroup_symbol_hm',
    f'q2_{BROADENING_TAG}',
    ]

# Ground truth the peak-list synthesis does not need but a labelled benchmark does.
TRUTH_COLUMNS = READ_COLUMNS + [
    'reindexed_xnn',
    'reindexed_extinction_group',
    f'reindexed_h_{BROADENING_TAG}',
    f'reindexed_k_{BROADENING_TAG}',
    f'reindexed_l_{BROADENING_TAG}',
    ]


def derived_seed(key, base_seed):
    """A stable seed for `key`. `hash()` will not do: it is salted per process."""
    digest = hashlib.sha256(f'{base_seed}:{key}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big')


def mechanism_rng(mechanism, entry_id, base_seed):
    """This entry's generator for one noise mechanism.

    A separate stream per mechanism is what keeps the condition axes independent. Sharing one
    stream across the mechanisms in a fixed order confounds every axis with the axes applied
    before it.
    """
    return np.random.default_rng(derived_seed(f'{mechanism}:{entry_id}', base_seed))


def sample_entries(bravais_lattice, n_entries, base_seed, columns=None,
                   dataset_directory=None):
    """Source crystals for one Bravais lattice, uniformly within it and reproducibly.

    Uniform within the lattice, so a per-lattice success rate keeps the natural volume
    distribution; stratifying by volume belongs to the split and to reporting, not here.

    `columns` widens the read for a caller that needs ground truth the synthesis does not use. The
    selection depends only on the seed and the count, so a widened read selects the same crystals.
    """
    directory = Path(dataset_directory) if dataset_directory is not None else DATASET_DIRECTORY
    path = directory / f'dataset_{bravais_lattice}.parquet'
    if not path.is_file():
        raise FileNotFoundError(
            f'No source dataset for {bravais_lattice} at {path}. '
            'Generate it, or point --dataset-directory at a tree that has it.')
    data = pd.read_parquet(path, columns=list(columns) if columns is not None else READ_COLUMNS)
    data = data.loc[~data['train']]
    peaks = data[f'q2_{BROADENING_TAG}']
    data = data.loc[peaks.apply(lambda q2: np.count_nonzero(q2) >= N_PEAKS)]
    data = data.sort_values('identifier', kind='stable', ignore_index=True)
    if data.shape[0] > n_entries:
        rng = np.random.default_rng(derived_seed(f'sample:{bravais_lattice}', base_seed))
        selected = np.sort(rng.choice(data.shape[0], size=n_entries, replace=False))
        data = data.iloc[selected].reset_index(drop=True)
    return data


def build_second_phase_pool(entries):
    """Candidate contaminating phases: the sampled crystals, across every Bravais lattice.

    Real contamination is not lattice-matched to the phase of interest, so a partner is drawn from
    the whole set rather than from within the entry's own lattice. Reusing the frame already in
    hand keeps the pool identical across shards, which it has to be for the choice to reproduce.
    """
    return (entries['identifier'].tolist(),
            [np.asarray(q2, dtype=float) for q2 in entries[f'q2_{BROADENING_TAG}']])


def choose_second_phase(entry_id, second_phase_pool, base_seed):
    """This entry's contaminating phase, deterministically and never itself."""
    rng = mechanism_rng('phase2_partner', entry_id, base_seed)
    identifiers, line_lists = second_phase_pool
    for _ in range(10):
        index = int(rng.integers(len(line_lists)))
        if identifiers[index] != entry_id:
            return identifiers[index], line_lists[index]
    raise ValueError(f'Could not draw a partner phase for {entry_id}')


class PreparedPattern:
    """One synthesised pattern: the observed window, and what was done to it.

    The three `_achieved` counts are what the pattern actually carries, which is not always what
    the condition asked for -- dropout is capped by the crystal's own surplus, and an injected line
    is lost if the re-sorted list truncates it back out of the window. An axis has to be read on
    what was delivered.
    """

    __slots__ = ('q2_obs', 'hkl_obs', 'n_dropout_achieved', 'n_contaminants_achieved',
                 'n_second_phase_achieved', 'second_phase_partner')

    def __init__(self, q2_obs, hkl_obs, n_dropout_achieved, n_contaminants_achieved,
                 n_second_phase_achieved, second_phase_partner):
        self.q2_obs = q2_obs
        self.hkl_obs = hkl_obs
        self.n_dropout_achieved = n_dropout_achieved
        self.n_contaminants_achieved = n_contaminants_achieved
        self.n_second_phase_achieved = n_second_phase_achieved
        self.second_phase_partner = second_phase_partner


def _hkl_for_values(q2_positive, hkl_positive, values):
    """The reflections behind `values`, looked up by position in the entry's own peak list.

    Selection through the mechanisms is by value rather than by index, so the reflections are
    recovered the same way. Stored peak lists are ascending, which `searchsorted` requires.
    """
    if hkl_positive is None:
        return None
    return hkl_positive[np.searchsorted(q2_positive, values)]


def _sigma_for(condition):
    """This condition's sigma(q2) intercept and slope."""
    from mlindex.utilities.ErrorAdder import q2_sigma_params

    intercept, slope = q2_sigma_params()
    return intercept * condition.intercept_scale, slope


def _unpack(result, hkl):
    """`add_contaminants` and `add_second_phase` return one array or two, depending on `hkl`."""
    if hkl is None:
        return result, None
    return result


def prepare_peak_list(entry, condition, base_seed, hkl=None, second_phase_pool=None,
                      n_peaks=N_PEAKS, max_drop=None):
    """Synthesise one pattern from one source crystal under one condition bundle.

    `hkl` is the ground-truth assignment for the crystal's full peak list, for callers that need to
    know which observed peak came from which reflection. A run that only needs unit cells can omit
    it.
    """
    entry_id = entry['identifier']
    if max_drop is None:
        max_drop = BenchmarkConditions.MAX_NESTED_DROPOUT

    q2_full = np.asarray(entry[f'q2_{BROADENING_TAG}'], dtype=float)
    positive = q2_full > 0
    q2_positive = q2_full[positive]
    hkl_positive = None if hkl is None else np.asarray(hkl, dtype=float)[positive]

    # 1. Dropout: which reflections were detected at all, before the instrument touches them.
    window, surplus, n_dropout_achieved = select_peaks_with_nested_dropout(
        q2_full, n_peaks, condition.n_dropout, mechanism_rng('dropout', entry_id, base_seed),
        max_drop=max_drop,
        )

    hkl_window = _hkl_for_values(q2_positive, hkl_positive, window)
    hkl_surplus = _hkl_for_values(q2_positive, hkl_positive, surplus)

    # 2. Error, applied to the window and the surplus together and then re-sorted, so a peak near
    #    the edge of the window can be displaced across it. Restricting the error to the window
    #    would make that boundary artificially sharp. The surplus is then discarded: a cell is
    #    scored on the peaks it was fitted to, and scoring beyond them was measured and lost.
    n_window = window.size
    extended = np.concatenate([window, surplus])[np.newaxis].copy()
    extended_hkl = (None if hkl_window is None
                    else np.concatenate([hkl_window, hkl_surplus])[np.newaxis].copy())
    intercept, slope = _sigma_for(condition)
    if condition.error_multiplier > 0:
        rng_error = mechanism_rng('error', entry_id, base_seed)
        if extended_hkl is None:
            extended = add_q2_error(extended, None, condition.error_multiplier, rng_error,
                                    intercept=intercept, slope=slope)
        else:
            extended, extended_hkl = add_q2_error(extended, extended_hkl,
                                                  condition.error_multiplier, rng_error,
                                                  intercept=intercept, slope=slope)

    window = extended[:, :n_window].copy()
    window_hkl = None if extended_hkl is None else extended_hkl[:, :n_window].copy()

    # 3. Contaminants, placed relative to the window as observed. An inserted line displaces a
    #    real peak out of the window, and the list is truncated back to n_peaks afterwards, so a
    #    contaminant drawn near the top of the range can be truncated out again -- which is why
    #    the count delivered is recorded rather than assumed.
    n_contaminants_achieved = 0
    if condition.n_contaminants > 0:
        rng_contaminant = mechanism_rng('contaminant', entry_id, base_seed)
        before = window[0].copy()
        result = add_contaminants(window, window_hkl, condition.n_contaminants, rng_contaminant,
                                  max_attempts=CONTAMINANT_MAX_ATTEMPTS)
        window, window_hkl = _unpack(result, window_hkl)
        n_contaminants_achieved = int(np.count_nonzero(np.isin(window[0], before, invert=True)))

    # 4. A second phase, last, because it is in the sample rather than added by the measurement.
    partner_id = None
    n_second_phase_achieved = 0
    if condition.second_phase_lines > 0:
        if second_phase_pool is None:
            raise ValueError('A second-phase bundle needs a partner pool; none was passed')
        partner_id, partner_q2 = choose_second_phase(entry_id, second_phase_pool, base_seed)
        rng_phase = mechanism_rng('phase', entry_id, base_seed)
        before = window[0].copy()
        result = add_second_phase(window, window_hkl, partner_q2, condition.second_phase_lines,
                                  rng_phase, low_angle_bias=condition.second_phase_bias)
        window, window_hkl = _unpack(result, window_hkl)
        n_second_phase_achieved = int(np.count_nonzero(np.isin(window[0], before, invert=True)))

    return PreparedPattern(
        q2_obs=window[0],
        hkl_obs=None if window_hkl is None else window_hkl[0],
        n_dropout_achieved=int(n_dropout_achieved),
        n_contaminants_achieved=n_contaminants_achieved,
        n_second_phase_achieved=n_second_phase_achieved,
        second_phase_partner=partner_id,
        )
