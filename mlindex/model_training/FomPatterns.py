"""Peak-list synthesis for Benchmark B: one pattern per (entry, condition).

On `fom` this lived in `mlindex/scripts/run_fom_mirror.py`, and `run_fom_dump.py` reached it with
a `sys.path.insert` followed by `import run_fom_mirror as mirror` -- a script importing a script.
It is a module here so the harness is importable, testable and pickleable under the `spawn` start
method, which CLAUDE.md requires and which a `sys.path` game does not survive reliably.

Three campaign-2 changes live in `prepare_peak_list`, and each fixes something campaign 1 paid for.

**Independent sub-streams per mechanism.** Campaign 1 ran dropout, error, contaminants and second
phase from a single `default_rng` seeded per entry, so a bundle differing only in
`n_contaminants` still received a different *error* realisation -- the contaminant rejection loop
consumes a variable number of draws and everything after it shifts. The condition axes were
nominally paired and actually were not. Each mechanism now draws from its own key, so changing one
condition parameter cannot perturb another mechanism.

**Nested sparsity.** N = 2, 4, 6 are prefixes of one hole set, so the sparsity axis is one crystal
degrading progressively (DWMM, 2026-08-26). See `ErrorAdder.select_peaks_with_nested_dropout`.

**The surplus peaks are produced here, carrying the window's noise.** Campaign 1 stored only
`n_peaks_available`, so its hold-out merit had to be reconstructed afterwards by replaying the
generator against the true structure alone -- giving lines that carry no contaminants while the
fitted window does, and a second noise draw rather than part of the same pattern. Its +7.11 pp is
optimistic by an unmeasured amount (R13). Two properties make the version here honest:

* The error is drawn for the window and the surplus **in one call**. `rng.normal(loc=0,
  scale=array)` is `standard_normal(n) * array` filled in order, so the first 20 values of a
  40-wide draw are bit-identical to a 20-wide draw -- the surplus genuinely continues the
  window's stream rather than starting a new one.
* **The boundary is not "peak 21".** `add_contaminants` and `add_second_phase` re-truncate to the
  window width after inserting, so an injected line landing inside the window displaces a real
  reflection *out* of it and the window's upper edge moves DOWN. Those displaced peaks are
  collected via `return_overflow` and belong to the hold-out. So does an injected line that
  itself fell above the window: it is part of the observed pattern beyond the fitted range, and
  it carries `(0, 0, 0)` as its reflection exactly as the schema wants.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import mlindex
from mlindex.model_training import FomConditions
from mlindex.utilities.ErrorAdder import add_contaminants
from mlindex.utilities.ErrorAdder import add_q2_error
from mlindex.utilities.ErrorAdder import add_second_phase
from mlindex.utilities.ErrorAdder import select_peaks_with_nested_dropout


# One instrument. The source data carries `0.5`, `1`, `1.5` and `sa`, but only the `*_1` model set
# exists on disk and every optimizer factory interpolates the tag into its model directory name,
# so a second-tag run fails at model load. DWMM's decision of 2026-08-26 closes the axis: tag `1`
# only, `sa` not used at all ("it is unrealistic"), and the interior-dropout condition is taken to
# stand in for the larger-broadening case. Measured for the record: moving from tag 1 to tag 1.5
# changes a mean of 1.00 of the first 20 lines and leaves 37.7 % of windows identical, and the
# 20-peak window reaches ~5 % further in q2 -- which is what dropout-with-backfill also does.
# See `artifacts/S05_broadening_tag_comparison.csv`.
BROADENING_TAG = '1'

N_PEAKS = 20
# Store at least 20 surplus peaks, per SCHEMA.md. A typical entry carries ~60 lines, so this is
# affordable, and S10 sweeps how many of them are worth scoring on -- campaign 1 imposed five a
# priori and never swept it.
N_HOLDOUT = 20

N_TOP_CANDIDATES = 20

# A contaminant draw is rejected unless it clears every peak's half breadth, and the whole set is
# redrawn on any rejection, so acceptance decays exponentially in the contaminant count. Uncapped
# this can spin forever on a crowded pattern; the cap turns that into a recorded skip.
CONTAMINANT_MAX_ATTEMPTS = 2000

# Give up on a shard rather than write a mostly-failed one, but tolerate isolated entry failures.
MAX_CONSECUTIVE_FAILURES = 10

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

# Ground truth the peak-list synthesis does not need but the benchmark schema requires.
DUMP_READ_COLUMNS = READ_COLUMNS + [
    'reindexed_xnn',
    'reindexed_extinction_group',
    f'reindexed_h_{BROADENING_TAG}',
    f'reindexed_k_{BROADENING_TAG}',
    f'reindexed_l_{BROADENING_TAG}',
    ]


def derived_seed(key, base_seed):
    """A stable seed for `key`, unlike `hash()`, which is salted per process.

    PROTOCOL §6 wants the per-entry seed derived from the entry id so the same entry gets the same
    noise in every condition bundle, and so any subset of the benchmark regenerates identically.
    """
    digest = hashlib.sha256(f'{base_seed}:{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def mechanism_rng(mechanism, entry_id, base_seed):
    """This entry's generator for one noise mechanism.

    The point of a separate stream per mechanism: a bundle that differs only in its contaminant
    count must not thereby receive a different error realisation. Campaign 1 shared one stream
    across all four mechanisms in a fixed order, so every condition axis was confounded with every
    axis applied before it.
    """
    return np.random.default_rng(derived_seed(f'{mechanism}:{entry_id}', base_seed))


def sample_entries(bravais_lattice, n_entries, base_seed, columns=None):
    """Entries for one Bravais lattice, uniformly within it and reproducibly.

    Uniform within the lattice, so the per-lattice success rate keeps the natural volume
    distribution. Volume stratification belongs to the split and to reporting, not to the sampling.

    `columns` widens the read for callers needing ground truth the peak-list synthesis does not
    use. The sampling depends only on the seed and the entry count, so a widened read selects the
    same entries.
    """
    data = pd.read_parquet(DATASET_DIRECTORY / f'dataset_{bravais_lattice}.parquet',
                           columns=list(columns) if columns is not None else READ_COLUMNS)
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
    """Candidate contaminating phases: the same entries, across every Bravais lattice.

    Real contamination is not lattice-matched to the phase of interest, so the partner is drawn
    from the whole sampled set rather than from within the entry's own lattice. Reusing the frame
    already in hand keeps the pool identical across pools and shards, which it must be for the
    partner choice to be reproducible.
    """
    return (entries['identifier'].tolist(),
            [np.asarray(q2, dtype=float) for q2 in entries[f'q2_{BROADENING_TAG}']])


def choose_second_phase(entry_id, second_phase_pool, base_seed):
    """This entry's contaminating phase, deterministically and excluding itself."""
    rng = mechanism_rng('phase2_partner', entry_id, base_seed)
    identifiers, line_lists = second_phase_pool
    for _ in range(10):
        index = int(rng.integers(len(line_lists)))
        if identifiers[index] != entry_id:
            return identifiers[index], line_lists[index]
    raise ValueError(f'could not draw a partner phase for {entry_id}')


class PreparedPattern:
    """One synthesised pattern: the fitted window, the surplus, and what was done to it."""

    __slots__ = ('q2_obs', 'hkl_obs', 'q2_holdout', 'hkl_holdout',
                 'n_dropout_achieved', 'second_phase_partner')

    def __init__(self, q2_obs, hkl_obs, q2_holdout, hkl_holdout,
                 n_dropout_achieved, second_phase_partner):
        self.q2_obs = q2_obs
        self.hkl_obs = hkl_obs
        self.q2_holdout = q2_holdout
        self.hkl_holdout = hkl_holdout
        self.n_dropout_achieved = n_dropout_achieved
        self.second_phase_partner = second_phase_partner


def _hkl_for_values(q2_positive, hkl_positive, values):
    """The reflections behind `values`, looked up by position in the entry's own peak list.

    Selection through the mechanisms is by value, not by index, so the reflections are recovered
    the same way. Peak lists are stored ascending, which `searchsorted` requires.
    """
    if hkl_positive is None:
        return None
    return hkl_positive[np.searchsorted(q2_positive, values)]


def prepare_peak_list(entry, condition, base_seed, hkl=None, second_phase_pool=None,
                      n_peaks=N_PEAKS, n_holdout=N_HOLDOUT, max_drop=None):
    """Synthesise one pattern under one condition bundle.

    Mechanisms are applied in the order a real pattern acquires them: some reflections are never
    detected, the instrument adds a random error, contaminant lines are placed relative to the
    peaks as observed, and a second phase is present in the sample rather than added by the
    measurement. Unlike campaign 1, each draws from its own stream, so the order no longer
    couples the axes to one another.

    `hkl` is the ground-truth assignment for the entry's full peak list, supplied by callers that
    need to know which observed peak came from which reflection. The dump does; a characterisation
    run need not.
    """
    entry_id = entry['identifier']
    if max_drop is None:
        max_drop = FomConditions.MAX_NESTED_DROPOUT

    q2_full = np.asarray(entry[f'q2_{BROADENING_TAG}'], dtype=float)
    positive = q2_full > 0
    q2_positive = q2_full[positive]
    hkl_positive = None if hkl is None else np.asarray(hkl, dtype=float)[positive]

    # 1. Dropout, nested and with a draw count that does not depend on the rung.
    window, holdout, n_dropout_achieved = select_peaks_with_nested_dropout(
        q2_full, n_peaks, condition.n_dropout, mechanism_rng('dropout', entry_id, base_seed),
        n_holdout=n_holdout, max_drop=max_drop,
        )

    hkl_window = _hkl_for_values(q2_positive, hkl_positive, window)
    hkl_holdout = _hkl_for_values(q2_positive, hkl_positive, holdout)

    # 2. Error, over the window and the surplus in ONE call, so the surplus continues the
    #    window's stream instead of starting a second, independent one (R13).
    n_window = window.size
    extended = np.concatenate([window, holdout])[np.newaxis].copy()
    extended_hkl = (None if hkl_window is None
                    else np.concatenate([hkl_window, hkl_holdout])[np.newaxis].copy())
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

    # add_q2_error re-sorts, so the fitted window is now simply the n_window lowest lines. A peak
    # crossing the boundary under noise is physically correct and is meant to be possible.
    window = extended[:, :n_window].copy()
    surplus = extended[0, n_window:].copy()
    window_hkl = None if extended_hkl is None else extended_hkl[:, :n_window].copy()
    surplus_hkl = None if extended_hkl is None else extended_hkl[0, n_window:].copy()

    displaced_q2 = []
    displaced_hkl = []

    # 3. Contaminants, placed relative to the fitted window. What they push out is kept.
    if condition.n_contaminants > 0:
        rng_contaminant = mechanism_rng('contaminant', entry_id, base_seed)
        result, overflow, overflow_hkl = add_contaminants(
            window, window_hkl, condition.n_contaminants, rng_contaminant,
            max_attempts=CONTAMINANT_MAX_ATTEMPTS, low_angle_bias=condition.contaminant_bias,
            return_overflow=True,
            )
        window, window_hkl = _unpack(result, window_hkl)
        displaced_q2.append(overflow[0])
        displaced_hkl.append(overflow_hkl[0])

    # 4. A second phase, last, because it is in the sample rather than added by the measurement.
    partner_id = None
    if condition.second_phase_lines > 0:
        if second_phase_pool is None:
            raise ValueError('a second-phase bundle needs a partner pool; none was passed')
        partner_id, partner_q2 = choose_second_phase(entry_id, second_phase_pool, base_seed)
        rng_phase = mechanism_rng('phase', entry_id, base_seed)
        result, overflow, overflow_hkl = add_second_phase(
            window, window_hkl, partner_q2, condition.second_phase_lines, rng_phase,
            low_angle_bias=condition.second_phase_bias, return_overflow=True,
            )
        window, window_hkl = _unpack(result, window_hkl)
        displaced_q2.append(overflow[0])
        displaced_hkl.append(overflow_hkl[0])

    # 5. The hold-out is the surplus plus everything the injected lines pushed out of the window,
    #    sorted. Defined relative to the FINAL window, which is why "peak 21" is the wrong rule.
    q2_holdout = np.concatenate([surplus] + displaced_q2) if displaced_q2 else surplus
    if surplus_hkl is None:
        hkl_holdout_final = None
    else:
        pieces = [surplus_hkl] + [np.asarray(part, dtype=float).reshape(-1, 3)
                                  for part in displaced_hkl if part is not None]
        hkl_holdout_final = np.concatenate(pieces, axis=0)
    order = np.argsort(q2_holdout)
    q2_holdout = q2_holdout[order]
    if hkl_holdout_final is not None:
        hkl_holdout_final = hkl_holdout_final[order]

    return PreparedPattern(
        q2_obs=window[0],
        hkl_obs=None if window_hkl is None else window_hkl[0],
        q2_holdout=q2_holdout,
        hkl_holdout=hkl_holdout_final,
        n_dropout_achieved=int(n_dropout_achieved),
        second_phase_partner=partner_id,
        )


def _sigma_for(condition):
    """This condition's sigma(q2) intercept and slope.

    The severity axis is the multiplier, applied inside `add_q2_error`; the shape axis is the
    intercept, scaled here. The slope is never varied -- a change to both is what the multiplier
    already is.
    """
    from mlindex.utilities.ErrorAdder import q2_sigma_params
    intercept, slope = q2_sigma_params()
    return intercept * condition.intercept_scale, slope


def _unpack(result, had_hkl):
    """Normalise the (q2,) / (q2, hkl) return shapes the ErrorAdder mechanisms use."""
    if had_hkl is None:
        return result, None
    return result[0], result[1]
