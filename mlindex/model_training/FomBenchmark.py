"""Benchmark A: the frozen candidate pool that the figure-of-merit work is developed against.

The indexer scores every candidate with M20 and then keeps the best twenty per Bravais
lattice. Comparing a *different* figure of merit against that output is impossible: the
candidates it would have ranked highly were discarded, and the columns needed to evaluate
it were never written. This module is the other half of the dump hook in
`MPIOptimizer._downsample_computation` -- it turns the hook's records into parquet, reads
them back, and recomputes the scores offline so a variant FOM can re-rank a frozen pool
instead of re-running the indexer.

Two tables, joined on ``(entry_id, condition_bundle)``:

    candidates_*.parquet   one row per surviving candidate
    entries.parquet        one row per indexed pattern *per condition*, with the ground truth

The bundle is not a candidate column -- the dump hook knows only the entry and the lattice --
so it is read back off the filename by `bundle_from_candidate_path` and attached by
`load_candidates`. Joining on ``entry_id`` alone was correct while a root held one bundle and
is wrong for a consolidated pool, where the same entry appears under every condition.

What is deliberately *not* stored, because it regenerates exactly and is large:

    q2_ref_calc     up to 1000 float64 per row. Rebuilt by `reference_lines` from the
                    Bravais lattice and the extinction group, which is what
                    `Candidates.assign_extinction_group` used to produce it.
    hkl_assign      rebuilt by `fast_assign` inside `recompute_scores`. Storing it would
                    only tell a failing round trip whether a `fast_assign` tie had
                    flipped; the round trip compares against the *pipeline's* M20, so it
                    catches the flip either way.

What is stored but censored, and matters when interpreting the negatives: the pool has
already passed `Candidates.prune_below_m20`, so no candidate scoring M20 < 5 is present.
See docs/fom/SCHEMA.md.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.utilities.FigureOfMerits import get_M20
from mlindex.utilities.FigureOfMerits import get_M20_likelihood_from_xnn
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref


SCHEMA_VERSION = '3'
# Version 3 is Benchmark B and is deliberately not backward compatible with campaign 1's
# Benchmark A. Every added column exists because campaign 1 could not answer a question without
# it; `docs/fom_campaign2/SCHEMA.md` is the specification and its "What changed from Benchmark A"
# table is the rationale, row by row. **Bump this whenever the column set changes** -- campaign 1
# shipped two different column sets under version '2', which is why a loader cannot trust it.

# Written by the dump hook, one row per surviving candidate.
CANDIDATE_COLUMNS = (
    'entry_id',
    'q2_digest',
    'bravais_lattice',
    'lattice_system',
    'candidate_id',
    'xnn',
    'unit_cell',
    'volume',
    'reciprocal_volume',
    'spacegroup',
    'hkl_ref_length',
    'n_peaks',
    'M20',
    'Minfo',
    'n_indexed',
    'final_rank',
    'in_top_n',
    'n_entering',
    'assignment_threshold',
    'downsample_radius',
    # --- schema v3 -------------------------------------------------------------------------
    # On the candidate row, so the join key is complete without the filename. Campaign 1's dump
    # hook ran per (entry, lattice) and never saw the condition, so it survived only in the path
    # -- and `(entry_id, q2_digest)` is not a substitute, because two sparse bundles leave 157
    # entries with identical peak lists (R8).
    'condition_bundle',
    # The pre-extinction-group, pre-refinement value the prune rule actually tested. Nothing
    # downstream can reconstruct it, and keeping it is what lets one run answer every higher
    # threshold by restriction -- the single best piece of design campaign 1 produced.
    'm20_at_prune',
    # The same for every rival cut criterion. C2-R-001: campaign 1 stored only M20 at the cut, so
    # S03 could not ask whether a different merit makes a better cut *at the cut*. Storing it now
    # means S07's own run answers it and no session has to re-run for it.
    'merit_at_prune',
    # `merit_preselection` was here and is deliberately NOT stored (DWMM, 2026-08-27). It would
    # measure how optimistic the stored merit is, given it is the best of ~100 stochastic
    # iterates -- but that bias applies to every candidate equally, it is fixed by an inner loop
    # campaign 2 does not change, and it runs in the conservative direction for the campaign's
    # own claim: every candidate sits at its M20-best, so a rival merit beating M20 here beats it
    # on M20's home ground. Nothing would be done differently for knowing it. See C2-F-046.
    'retained_by',
    'prune_threshold',
    # S04's absence counts, which replace the 158-level extinction-group categorical
    # (C2-F-041: +0.522 pp of operating point, p <= 0.004 at every fit seed).
    'n_absent_extra',
    # `n_absent_extra_in_range` is NOT a column here. It needs the candidate's own reference lines
    # and is recomputable offline from `xnn`, the peak list and the extinction group, so by this
    # schema's own rule it does not earn storage. It was briefly kept as "a defined home for the
    # analysis stage", which would have meant shipping a column that is null in every row of every
    # pool -- exactly what made campaign 1 exclude Mighell-Santoro degenerates at a *measured*
    # zero rather than a known one. It belongs in the analysis output that computes it.
    'n_groups_searched',
    # `xnn_pregen` was here and is deliberately NOT stored (DWMM, 2026-08-27). It existed to give
    # the null distribution of an *arbitrary* cell rather than of a refined survivor (campaign 1's
    # R10) -- which serves null calibration, a family campaign 2 dropped as a designed negative.
    # No campaign-2 step consumes it. See C2-F-046.
    # Negative subsampling bookkeeping. Without the weight every fit on this pool is biased.
    'sampling_weight',
    'retained_reason',
)

# Written by the pre-deduplication dump hook (S14), one row per candidate *entering*
# deduplication. A separate stream from CANDIDATE_COLUMNS on purpose: this population is
# ~2x the survivors at the production prune threshold and far larger at threshold 0, and
# unit_cell / volume / reciprocal_volume are all recoverable from xnn. Keeping it separate
# leaves Benchmark A's candidate schema and its loaders untouched.
PREDOWNSAMPLE_COLUMNS = (
    'entry_id',
    'q2_digest',
    'bravais_lattice',
    'lattice_system',
    'candidate_id',
    'xnn',
    'spacegroup',
    'hkl_ref_length',
    'n_peaks',
    'M20',
    'Minfo',
    'n_indexed',
    'm20_at_prune',
    # 0 for the iterate the M20 track kept -- the row production would have had -- and k
    # for the k-th entry of opt_params['retention_foms'] beyond M20 (S14 item 1). It is
    # what makes the ceiling before/after multi-FOM retention a restriction *inside* one
    # run: F-137 established that two arms of the same configuration are not comparable,
    # and a retention-on run differs from a retention-off one in exactly the way that
    # finding describes.
    'retained_by',
    'n_entering',
    'prune_m20_threshold',
    'downsample_radius',
    # --- schema v3 -------------------------------------------------------------------------
    'condition_bundle',
    'merit_at_prune',
    # Labels on THIS stream too. Campaign 1 wrote them into its consolidated shards but not onto
    # the 57.4 M-row pre-deduplication dump, so every re-analysis of its most valuable dataset
    # repeated a multi-hour labelling pass (R24).
    'is_correct',
    'sampling_weight',
)

# Written by the driver, one row per indexed pattern.
ENTRY_COLUMNS = (
    'entry_id',
    'q2_digest',
    'source_db',
    'split',
    'condition_bundle',
    'q2_obs',
    'n_peaks_available',
    'q2_error_multiplier',
    'n_contaminants',
    'contaminant_bias',
    'n_dropout',
    'n_dropout_achieved',
    'second_phase_lines',
    'second_phase_bias',
    'second_phase_partner',
    'xnn_true',
    'unit_cell_true',
    'volume_true',
    'bravais_lattice_true',
    'lattice_system_true',
    'spacegroup_true',
    'extinction_group_true',
    'hkl_true',
    # --- schema v3 -------------------------------------------------------------------------
    # The surplus peaks, beyond the fitted window, drawn from the SAME noise stream as the window
    # and carrying the same contaminants, second phase and dropout. Campaign 1 stored only
    # `n_peaks_available`, so its hold-out merit had to be reconstructed by replaying the
    # generator against the true structure alone -- lines carrying no contaminants while the
    # fitted window does, and a second noise draw rather than part of the same pattern. Its
    # +7.11 pp is optimistic by an unmeasured amount (R13).
    'q2_holdout',
    'hkl_holdout',
    # Read from the frozen split manifest, NEVER recomputed. A within-lattice rank rises when
    # rows are dropped, so recomputing moved 114 campaign-1 entries, all upward, and shifted the
    # hard stratum from 286 to 298 (R14).
    'volume_decile',
    # Survivors before subsampling, so a percentile has its true denominator rather than a count
    # of surviving rows.
    'pool_size_full',
    # The error model actually used. `error_law` is the constant 'gaussian' by decision (DWMM,
    # 2026-08-26) and `error_law_params` is [intercept, slope] of sigma(q2) -- the severity axis
    # is the multiplier and the shape axis is the intercept. Recorded rather than assumed so a
    # bundle's provenance is readable from the data. C2-R-008 bounds what is NOT varied here.
    'error_law',
    'error_law_params',
    'intercept_scale',
    # One instrument, recorded regardless. Tag '1' throughout; `sa` is not used (DWMM: "it is
    # unrealistic") and only the `*_1` model set exists on disk in any case.
    'broadening_tag',
    # Mighell-Santoro degeneracy, resolving C2-Q-002. **An entry-level property, not a
    # candidate-level one**, which is a deliberate departure from SCHEMA.md's placement: the
    # definition that resolves the question is a statement about the true lattice's Niggli
    # reduced cell, so it takes one value per pattern rather than one per candidate. Joining it
    # onto candidates is a join, not a recompute. See `mlindex/utilities/LatticeDegeneracy.py`.
    'is_degenerate',
    'degeneracy_conditions',
    'degeneracy_systematic',
)


# Attached by label_frame, and -- since the S04 consolidation -- written into the candidate shards
# themselves. Labelling costs ~9 ms/candidate, so a pool that already carries them must not be
# relabelled just because a loader defaults to label=True.
LABEL_COLUMNS = (
    'is_correct',
    'is_off_by_two',
    'xnn_distance_to_truth',
    'volume_ratio_to_truth',
    # schema v3. Without it "the correct Miller index" is a statement about a cell setting: a
    # monoclinic lattice admits many equivalent cells, and pooling without the setting cut moved
    # campaign 1's base rate from 0.83 to 0.38 (R15).
    'hkl_true_in_basis',
)
# `prior_target` was here and is NOT produced. R16 asks for a per-candidate prior label so a
# learned prior can be tested as a re-RANKER and not only as a re-scorer, and DWMM asked for it
# explicitly -- but nothing in the record says what the label IS, and inventing one mid-session is
# what C2-Q-002 shows the cost of. Open as C2-Q-015. It is left out rather than shipped null
# (C2-F-046), and every plausible definition is a function of columns this pool already stores --
# `volume`, `bravais_lattice`, `xnn`, and the truth beside them -- so it can be added offline once
# defined, with no regeneration.
# `is_degenerate` is NOT here. It was in campaign 1's label set and shipped null; campaign 2
# defines it on the true lattice alone, which makes it an entry column (see ENTRY_COLUMNS).


def q2_digest(q2_obs):
    """Stable short digest of a peak list, carried in both tables.

    The candidate and entry tables are written by different code paths and joined later.
    A mis-joined shard is otherwise silent -- every column still parses, the numbers are
    just attached to the wrong pattern. Eight bytes makes that detectable.
    """
    return hashlib.blake2b(
        np.ascontiguousarray(q2_obs, dtype=np.float64).tobytes(), digest_size=8
        ).hexdigest()


def records_to_frame(records):
    """Flatten the dump hook's per-(entry, Bravais lattice) records into a tidy frame.

    Each record holds parallel arrays for one lattice's survivors plus the scalars that
    apply to all of them. Empty records are dropped rather than contributing zero-row
    frames, because a lattice returning nothing is normal (cF and cI often do).
    """
    rows = []
    for record in records:
        n_candidates = record['xnn'].shape[0]
        if n_candidates == 0:
            continue
        context = record.get('context') or {}
        entry_id = context.get('entry_id')
        merit_names, merit_values = _merit_at_prune(record, n_candidates)
        for candidate_index in range(n_candidates):
            rows.append({
                'entry_id': entry_id,
                'q2_digest': record['q2_digest'],
                'bravais_lattice': record['bravais_lattice'],
                'lattice_system': record['lattice_system'],
                'candidate_id': candidate_index,
                'xnn': record['xnn'][candidate_index].astype(np.float64),
                'unit_cell': record['unit_cell'][candidate_index].astype(np.float64),
                'volume': float(record['volume'][candidate_index]),
                'reciprocal_volume': float(record['reciprocal_volume'][candidate_index]),
                'spacegroup': record['spacegroup'][candidate_index],
                'hkl_ref_length': record['hkl_ref_length'],
                'n_peaks': record['n_peaks'],
                'M20': float(record['M20'][candidate_index]),
                'Minfo': float(record['Minfo'][candidate_index]),
                'n_indexed': int(record['n_indexed'][candidate_index]),
                'final_rank': int(record['final_rank'][candidate_index]),
                'in_top_n': bool(record['in_top_n'][candidate_index]),
                'n_entering': record['n_entering'],
                'assignment_threshold': record['assignment_threshold'],
                'downsample_radius': record['downsample_radius'],
                'condition_bundle': context.get('condition_bundle'),
                'm20_at_prune': _scalar_at(record.get('m20_at_prune'), candidate_index),
                # A list per candidate rather than one column per criterion, with the order
                # recorded once in the manifest as `merit_at_prune_names`. C2-R-001 asks for
                # *every* rival cut criterion at the cut site, and a fixed column count keeps a
                # loader from having to discover the merit set from the schema.
                'merit_at_prune': (None if merit_values is None
                                   else merit_values[candidate_index].astype(np.float64)),
                'retained_by': int(_scalar_at(record.get('retained_by'), candidate_index) or 0),
                'prune_threshold': record.get('prune_m20_threshold'),
                'n_absent_extra': _scalar_at(record.get('n_absent_extra'), candidate_index),
                'n_absent_extra_in_range': _scalar_at(record.get('n_absent_extra_in_range'),
                                                      candidate_index),
                'n_groups_searched': _scalar_at(record.get('n_groups_searched'), candidate_index),
                # Overwritten by the subsampler. Defaulting to a kept row with unit weight means
                # a pool written without subsampling is still correct to fit on.
                'sampling_weight': 1.0,
                'retained_reason': 'all',
                })
    frame = pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))
    if merit_names:
        frame.attrs['merit_at_prune_names'] = merit_names
    return frame


def _scalar_at(values, index):
    """One element of a per-candidate array, or None when the producer did not supply it.

    The dump hook's optional columns are absent rather than null when a capture is off, so a
    frame builder that indexed them unconditionally would fail on a production-shaped record.
    """
    if values is None:
        return None
    return values[index]


def _merit_at_prune(record, n_candidates):
    """(names, values) for the rival cut criteria captured at the prune site.

    The dump hook emits one array per criterion under `merit_at_prune_<name>`. They are collapsed
    into a single list-valued column here, in a fixed, sorted order that the manifest records --
    so the column set does not change when the merit set does, and a loader can always say what
    the k-th entry means. C2-R-001 is the reason any of this is stored: campaign 1 kept only M20
    at the cut, so the question of whether a different merit makes a better cut could not be
    asked where the cut actually is.
    """
    from mlindex.optimization.Candidates import PRUNE_CAPTURE_MERITS

    present = {name[len('merit_at_prune_'):] for name in record
               if name.startswith('merit_at_prune_')}
    if not present:
        return (), None
    # ORDERED BY THE CAPTURE SITE'S OWN TUPLE, not alphabetically. This used to sort, while the
    # manifest wrote the capture order -- so a loader reading merit_at_prune[k] by the manifest's
    # k-th name got M_rev where it expected M_tilde, on four of seven entries, silently, and the
    # round-trip gate could not see it because it only checks M20 (C2-F-067). Anything unexpected
    # is appended in sorted order rather than dropped, so a new criterion is visible instead of
    # silently shifting the ones after it.
    names = tuple(name for name in PRUNE_CAPTURE_MERITS if name in present)
    names += tuple(sorted(present - set(PRUNE_CAPTURE_MERITS)))
    values = np.stack([record[f'merit_at_prune_{name}'] for name in names], axis=1)
    return names, values


def predownsample_records_to_frame(records):
    """Flatten the pre-deduplication records into a tidy frame.

    Built column-wise rather than row-wise, unlike ``records_to_frame``. At prune
    threshold 0 a single hard-stratum entry can contribute tens of thousands of rows
    across the fourteen lattices, and a per-row dict there costs both time and a large
    transient. The two builders are otherwise the same shape.
    """
    columns = {name: [] for name in PREDOWNSAMPLE_COLUMNS}
    for record in records:
        n_candidates = record['xnn'].shape[0]
        if n_candidates == 0:
            continue
        context = record.get('context') or {}
        columns['entry_id'].append(np.repeat(context.get('entry_id'), n_candidates))
        columns['q2_digest'].append(np.repeat(record['q2_digest'], n_candidates))
        columns['bravais_lattice'].append(np.repeat(record['bravais_lattice'], n_candidates))
        columns['lattice_system'].append(np.repeat(record['lattice_system'], n_candidates))
        columns['candidate_id'].append(np.arange(n_candidates, dtype=np.int64))
        columns['xnn'].append(list(record['xnn'].astype(np.float64)))
        columns['spacegroup'].append(np.asarray(record['spacegroup'], dtype=object))
        columns['hkl_ref_length'].append(np.repeat(record['hkl_ref_length'], n_candidates))
        columns['n_peaks'].append(np.repeat(record['n_peaks'], n_candidates))
        columns['M20'].append(record['M20'].astype(np.float64))
        columns['Minfo'].append(record['Minfo'].astype(np.float64))
        columns['n_indexed'].append(record['n_indexed'].astype(np.int64))
        columns['m20_at_prune'].append(record['m20_at_prune'].astype(np.float64))
        # `retained_by` is 0 for every row here: multi-merit iterate retention is not ported to
        # campaign 2 (it bought four entries of 972 for 57 % more rows -- F-155), so every row is
        # the one the M20 track kept, which is the production population. The column stays so a
        # loader written against campaign 1's shards still reads these.
        columns['retained_by'].append(
            record.get('retained_by', np.zeros(n_candidates)).astype(np.int64))
        columns['n_entering'].append(np.repeat(record['n_entering'], n_candidates))
        context = record.get('context') or {}
        columns['condition_bundle'].append(
            np.repeat(context.get('condition_bundle'), n_candidates))
        merit_names, merit_values = _merit_at_prune(record, n_candidates)
        columns['merit_at_prune'].append(
            [np.zeros(0)] * n_candidates if merit_values is None
            else list(merit_values.astype(np.float64)))
        columns['is_correct'].append(
            record.get('is_correct', np.zeros(n_candidates, dtype=bool)).astype(bool))
        columns['sampling_weight'].append(
            record.get('sampling_weight', np.ones(n_candidates)).astype(np.float64))
        columns['prune_m20_threshold'].append(
            np.repeat(record['prune_m20_threshold'], n_candidates))
        columns['downsample_radius'].append(
            np.repeat(record['downsample_radius'], n_candidates))

    if not columns['entry_id']:
        return pd.DataFrame(columns=list(PREDOWNSAMPLE_COLUMNS))

    data = {}
    for name, chunks in columns.items():
        if name in ('xnn', 'merit_at_prune'):
            data[name] = [row for chunk in chunks for row in chunk]
        else:
            data[name] = np.concatenate(chunks)
    return pd.DataFrame(data, columns=list(PREDOWNSAMPLE_COLUMNS))


def _hkl_ref_path(lattice_system, bravais_lattice, models_directory=None):
    if models_directory is None:
        from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
        models_directory = _resolve_models_dir()
    return Path(models_directory) / f'{lattice_system}_1' / 'data' / f'hkl_ref_{bravais_lattice}.npy'


_HKL_REF_CACHE = {}


def spacegroup_reference_sets(lattice_system, bravais_lattice, models_directory=None):
    """Every extinction group's reference Miller indices for one Bravais lattice.

    The whole dictionary `Candidates.assign_extinction_group` searches over, so a caller that
    has to reproduce that search -- rather than score one known group -- does not have to
    rebuild it. `get_spacegroup_hkl_ref` reaches into cctbx and is far too slow to call per
    row, hence the cache.
    """
    key = (str(models_directory), lattice_system, bravais_lattice)
    if key not in _HKL_REF_CACHE:
        path = _hkl_ref_path(lattice_system, bravais_lattice, models_directory)
        _HKL_REF_CACHE[key] = get_spacegroup_hkl_ref(
            np.load(path), bravais_lattice=bravais_lattice
            )
    return _HKL_REF_CACHE[key]


def hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory=None):
    """Reference Miller indices for one extinction group.

    `Candidates.assign_extinction_group` builds its calculated lines from the model's
    truncated `hkl_ref` filtered by the winning extinction group, so reproducing its M20
    means starting from exactly that list.
    """
    return spacegroup_reference_sets(
        lattice_system, bravais_lattice, models_directory
        )[spacegroup]


def reference_lines(xnn, lattice_system, bravais_lattice, spacegroup,
                    models_directory=None):
    """Calculated q2 for every reference line of one extinction group.

    Returns an (n_candidates, n_reference_lines) array. Callers must not reuse the result
    across `get_M20` calls -- see `recompute_scores`.
    """
    hkl_ref = hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory)
    return Q2Calculator(
        lattice_system=lattice_system,
        hkl=hkl_ref,
        tensorflow=False,
        representation='xnn',
        ).get_q2(np.atleast_2d(xnn))


def assign_lines(q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
                 models_directory=None):
    """Rebuild the calculated lines and the Miller-index assignment for one extinction group.

    Everything a figure of merit needs beyond the dumped columns, and the step both
    `recompute_scores` and `zoo_features` start from: the full reference list `q2_ref_calc`,
    the assignment `fast_assign` chooses, the assigned Miller indices, and the calculated
    position of the line assigned to each observed peak.

    Returns (q2_ref_calc, hkl_assign, hkl, q2_calc). `q2_ref_calc` is fresh, not cached --
    `get_M20` writes into it (FigureOfMerits.py, np.putmask), so it must not be shared.
    """
    xnn = np.atleast_2d(np.asarray(xnn, dtype=np.float64))
    q2_obs = np.asarray(q2_obs, dtype=np.float64)
    hkl_ref = hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory)

    q2_ref_calc = reference_lines(
        xnn, lattice_system, bravais_lattice, spacegroup, models_directory
        )
    hkl_assign = fast_assign(q2_obs, q2_ref_calc)
    hkl = np.take(hkl_ref, hkl_assign, axis=0)
    q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
    return q2_ref_calc, hkl_assign, hkl, q2_calc


def recompute_scores(q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
                     assignment_threshold, models_directory=None):
    """Recompute (M20, Minfo, n_indexed) for candidates sharing one extinction group.

    This is the round trip the acceptance gate measures: it must reproduce the values the
    pipeline stored, from the dumped columns alone.
    """
    xnn = np.atleast_2d(np.asarray(xnn, dtype=np.float64))
    q2_obs = np.asarray(q2_obs, dtype=np.float64)
    q2_ref_calc, hkl_assign, hkl, q2_calc = assign_lines(
        q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, models_directory
        )

    # get_M20 mutates q2_ref_calc in place (FigureOfMerits.py, np.putmask), so it gets the
    # array last and Minfo is computed from xnn before it is touched.
    _, probability, Minfo = get_M20_likelihood_from_xnn(
        q2_obs=q2_obs,
        xnn=xnn,
        hkl=hkl,
        lattice_system=lattice_system,
        bravais_lattice=bravais_lattice,
        )
    n_indexed = np.sum(probability > assignment_threshold, axis=1, dtype=int)
    M20 = get_M20(q2_obs, q2_calc, q2_ref_calc)
    return M20, Minfo, n_indexed, hkl_assign


def recompute_frame(candidates, entries, models_directory=None):
    """Recompute M20/Minfo/n_indexed for a whole candidate frame.

    Adds `M20_recomputed`, `Minfo_recomputed` and `n_indexed_recomputed`. Grouped by
    (Bravais lattice, extinction group) so the cctbx-backed reference list is built once
    per group rather than once per row.
    """
    # Keyed through `_join_keys`, NOT on `entry_id` alone. After consolidation `entries.parquet`
    # holds one row per (entry, bundle), so `entry_id` is not a key there: `peaks.loc[entry_id]`
    # returns one array per bundle and the recompute raises -- or, worse, would silently score a
    # candidate against another bundle's peak list if the shapes happened to line up. That is R8,
    # and `zoo_features` already reads the keys this way.
    join_keys = _join_keys(candidates, entries)
    peaks = entries.set_index(join_keys)['q2_obs']
    columns = ['M20_recomputed', 'Minfo_recomputed', 'n_indexed_recomputed']
    out = pd.DataFrame(index=candidates.index, columns=columns, dtype=float)

    group_keys = list(join_keys) + ['lattice_system', 'bravais_lattice', 'spacegroup', 'n_peaks']
    for key, group in candidates.groupby(group_keys, sort=False):
        peak_key = key[0] if len(join_keys) == 1 else key[:len(join_keys)]
        lattice_system, bravais_lattice, spacegroup, n_peaks = key[len(join_keys):]
        # Cubic models take ten peaks and everything else twenty; the optimizer truncates
        # with a plain prefix slice, so the entry's list is cut the same way here.
        q2_obs = np.asarray(peaks.loc[peak_key], dtype=np.float64)[:n_peaks]
        xnn = np.vstack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
        M20, Minfo, n_indexed, _ = recompute_scores(
            q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
            float(group['assignment_threshold'].iloc[0]), models_directory,
            )
        out.loc[group.index, 'M20_recomputed'] = M20
        out.loc[group.index, 'Minfo_recomputed'] = Minfo
        out.loc[group.index, 'n_indexed_recomputed'] = n_indexed

    result = candidates.copy()
    result[columns] = out
    result['n_indexed_recomputed'] = result['n_indexed_recomputed'].astype(int)
    return result


# The merits the negative subsampler ranks on, and the merit set S09 reports. Four calls, six
# derived columns, plus the stored `M20` -- the campaign-2 reduced core settled by DWMM on
# 2026-08-25 and priced at 6.0x `get_M20` in `artifacts/S02_zoo_cost.csv`.
#
# `subsample_negatives` MUST rank on all seven. K = 200 was measured as the size of the UNION over
# exactly this set, which is ~3.3x K rather than K (C2-F-051); ranking on M20 alone would retain a
# third of what the sizing assumed and would silently stop rank metrics on the other six being
# exact to depth K -- which is the property the whole retention rule exists to preserve.
REDUCED_MERIT_COLUMNS = ('M20', 'M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap')

# The support floor `get_M_rev_sym` applies to `M_rev`. Named here because a *stored* M_rev is
# floored by this value and nothing in the column itself says so (C2-F-062, C2-Q-017).
M_REV_MIN_N_CAL = 10

# Everything `reduced_merits` computes, and so everything a merit sidecar carries. The first six
# are the recomputable half of REDUCED_MERIT_COLUMNS -- M20 is stored, never recomputed. The last
# two are the audit pair: without them a stored `M_rev` of 0.0 means "the floor fired", "N_cal was
# zero" and "the candidate is degenerate" indistinguishably, and the floor cannot be undone after
# the fact.
RECOMPUTED_MERIT_COLUMNS = ('M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap',
                            'N_cal', 'M_rev_unfloored')

# The posterior-based counting merits (C2-Q-025). Written to their own sidecar rather than added
# to the set above, because the pool's existing sidecars are complete and verified and must not be
# invalidated -- and because these are NOT in the subsampler's retention rule, so a rank metric on
# them is only exact on a fully retained pool (C2-R-013).
SOFT_MERIT_COLUMNS = ('X_N_soft', 'n_over_soft', 'max_gap_soft')


def reduced_merits(candidates, entries, models_directory=None, soft=False):
    """The recomputable merits of the reduced core, plus the pair that audits the M_rev floor.

    Returns a frame aligned to `candidates`' index carrying `RECOMPUTED_MERIT_COLUMNS`: the six
    ranking merits `M_tilde`, `M_rev`, `M_sym`, `X_N`, `n_over` and `max_gap`, and the two audit
    columns `N_cal` and `M_rev_unfloored`. `M20` is not recomputed: it is already a stored column,
    and it is the one merit whose stored value is the *reported* one by construction.

    These are **not stored** in the pool. Every one is recomputable offline from `xnn`, the peak
    list and the extinction group, which is this schema's own rule for what does not earn a column
    (SCHEMA.md). They are computed here because the subsampler has to rank on them before the rows
    it would rank are thrown away.

    Grouped exactly as `recompute_frame` groups, so the cctbx-backed reference list is built once
    per extinction group rather than once per row, and the assignment is redone with `fast_assign`
    on the optimiser's own calculator output -- the route matters to the last bit, since rebuilding
    q2_calc from stored Miller indices differs by an ULP that can move a line across M20's cut-off
    (F-095).

    **`M_rev` is floored, and both sides of the floor come out of one call.** `get_M_rev_sym`
    zeroes `M_rev` wherever the counting window holds fewer than `M_REV_MIN_N_CAL` reference
    lines: below that the refinement interpolates them exactly and the merit reads 1e11-1e14 for a
    cell M20 correctly scores at 1.5 (C2-F-059). Asking for the *unfloored* value together with
    `n_cal` and applying the floor here is exactly equivalent, because the floor is a mask over
    the same arithmetic rather than a different one -- so both arms are available for the price of
    one evaluation. `M_sym` inherits the floor through the product, which is the intent; `M_tilde`
    is untouched by it, so the unfloored `M_sym` is `M_tilde * M_rev_unfloored` and is not stored
    separately.

    `get_M20` is deliberately NOT called here, though no longer for the reason this docstring gave
    until 2026-08-31. It used to zero the excluded lines of `q2_ref_calc` in place with
    `np.putmask`, so calling it changed what the other merits then saw; it was rewritten to take
    both reductions without touching its input and that ordering constraint is gone (see
    `FigureOfMerits.get_M20`). It is omitted now only because recomputing a stored column invites
    the two copies to disagree.
    """
    from mlindex.utilities.FigureOfMerits import get_M_rev_sym
    from mlindex.utilities.FigureOfMerits import get_n_over
    from mlindex.utilities.FigureOfMerits import get_X_N

    names = list(SOFT_MERIT_COLUMNS) if soft else list(RECOMPUTED_MERIT_COLUMNS)
    out = pd.DataFrame(index=candidates.index, columns=names, dtype=float)
    if candidates.empty:
        return out

    join_keys = _join_keys(candidates, entries)
    peaks = entries.set_index(join_keys)['q2_obs']
    group_keys = list(join_keys) + ['lattice_system', 'bravais_lattice', 'spacegroup', 'n_peaks']
    for key, group in candidates.groupby(group_keys, sort=False):
        peak_key = key[0] if len(join_keys) == 1 else key[:len(join_keys)]
        lattice_system, bravais_lattice, spacegroup, n_peaks = key[len(join_keys):]
        # Cubic models take ten peaks and everything else twenty; the optimizer truncates with a
        # plain prefix slice, so the entry's list is cut the same way here.
        q2_obs = np.asarray(peaks.loc[peak_key], dtype=np.float64)[:int(n_peaks)]
        xnn = np.vstack([np.asarray(value, dtype=np.float64) for value in group['xnn']])
        q2_ref_calc, _, _, q2_calc = assign_lines(
            q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, models_directory)

        if soft:
            from mlindex.utilities.FigureOfMerits import get_soft_counts
            values = get_soft_counts(q2_obs, q2_calc, q2_ref_calc, lattice_system)
            for name in names:
                out.loc[group.index, name] = np.asarray(values[name], dtype=float)
            continue

        # Unfloored, with the support count, so the floored arm is a mask over this one rather
        # than a second evaluation of identical arithmetic.
        M_tilde, M_rev_unfloored, _, N_cal = get_M_rev_sym(
            q2_obs, q2_calc, q2_ref_calc, min_n_cal=None, return_n_cal=True)
        M_rev = np.where(N_cal >= M_REV_MIN_N_CAL, M_rev_unfloored, 0.0)
        M_sym = M_tilde*M_rev
        n_over, max_gap = get_n_over(q2_obs, q2_calc, q2_ref_calc)
        X_N = get_X_N(q2_obs, q2_calc, q2_ref_calc)
        computed = (M_tilde, M_rev, M_sym, X_N, n_over, max_gap, N_cal, M_rev_unfloored)
        for name, values in zip(names, computed):
            out.loc[group.index, name] = np.asarray(values, dtype=float)
    return out


def with_reduced_merits(candidates, entries, models_directory=None):
    """`candidates` plus `RECOMPUTED_MERIT_COLUMNS`, for the subsampler. Not written to disk.

    The subsampler ranks on `REDUCED_MERIT_COLUMNS` only; the two audit columns ride along and are
    dropped again by the caller, which is why that drop list is `RECOMPUTED_MERIT_COLUMNS` rather
    than the ranking set.
    """
    merits = reduced_merits(candidates, entries, models_directory=models_directory)
    return pd.concat([candidates, merits], axis=1)


# The peak budgets the S10 sweep reports. `n_extra` IS the total peak budget minus the 20-peak
# window, so 5 is a 25-peak pattern and 10 a 30-peak one -- which is the range real data actually
# occupies (DWMM, 2026-09-01: "In many of the experimental data cases, only 20 - 25 peaks are
# actually recorded. On extremely good data, maybe 30").
#
# 20 is reachable HERE and on no instrument, and the reason is worth stating precisely, because
# the record got it wrong for a day: the benchmark stores up to 60 lines an entry **because
# `GenerateDataset.EntryGenerator` sets `self.peak_length = 60` and truncates to it**. That is a
# storage constant, not a property of a pattern, and 54.9 % of entries sit exactly on it -- so
# their true line count is unknown and >= 60. Two consequences: 20 is kept as a labelled upper
# bound and must never be quoted as achievable, and there is deliberately **no `all` budget**,
# since for the censored majority it would measure the constant rather than the data (C2-F-103).
HOLDOUT_N_EXTRA = (1, 2, 3, 5, 10, 20)
HOLDOUT_REALISTIC_MAX = 10

# Base seed for the offline surplus contamination. Fixed here rather than passed, so that two
# analyses of the same pool contaminate it identically; the per-entry stream is derived from the
# entry ID on top of it (PROTOCOL section 6).
HOLDOUT_CONTAMINANT_SEED = 12345

# One hold-out column set, per n_extra. Every name is registered in FomMetrics.HIGHER_IS_BETTER.
#
# `ho_tail_nll` is a SUM over surplus peaks and `ho_Minfo` would be one, so their scale grows with
# the window and a sweep would read that growth as a gain. `ho_Minfo` is therefore emitted as the
# per-peak MEAN of its terms -- which is what makes it comparable across n_extra and comparable
# with the intensive merits beside it -- and `ho_tail_nll` is kept in its native summed form and
# must be read at fixed n_extra only.
HOLDOUT_MERIT_NAMES = (
    'ho_M20', 'ho_M', 'ho_raw', 'ho_tail_nll',
    'ho_M_tilde', 'ho_M_rev', 'ho_M_sym',
    'ho_Minfo',
    # S10c's posterior family. `ho_evidence` is the denominator the posterior divides away, which
    # F-131 found worth more than the ratio; it is carried as a column of its own for the first
    # time, and a unit test pins the fact that it prefers a HALVED cell, so read it expecting the
    # same systematic inversion `X_N` has (C2-F-095) until the benchmark says otherwise.
    'ho_post', 'ho_post_logmean', 'ho_evidence',
    'ho_N_cal', 'ho_n_scored', 'ho_ref_reach',
    )


def holdout_column(name, n_extra):
    """`ho_M_sym` at 5 surplus peaks is `ho_M_sym__n5`. One place, so a reader is never guessing."""
    return f'{name}__n{n_extra}'


def holdout_columns(n_extra_values=HOLDOUT_N_EXTRA, names=HOLDOUT_MERIT_NAMES):
    return tuple(holdout_column(name, k) for k in n_extra_values for name in names)


def contaminated_holdout(entry_row, entry_id, seed=HOLDOUT_CONTAMINANT_SEED):
    """This entry's surplus peaks with contaminants seeded in, and the count added.

    **Once per (entry, condition), never per candidate or per Bravais lattice.** Every candidate of
    one pattern must face the same surplus or the comparison between them is not paired, so this is
    keyed on the entry and cached by the caller.

    The rate is the entry's own bundle rate -- injected lines per window peak -- so a `cont0` bundle
    draws nothing and is returned untouched. See `ErrorAdder.add_surplus_contaminants` for why the
    generator cannot produce these itself and why applying them offline is sound.
    """
    from mlindex.model_training.FomPatterns import derived_seed
    from mlindex.utilities.ErrorAdder import add_surplus_contaminants
    from mlindex.utilities.ErrorAdder import surplus_contaminant_rate

    q2_holdout = np.asarray(entry_row['q2_holdout'], dtype=np.float64)
    hkl_holdout = np.asarray(entry_row['hkl_holdout']).reshape(-1, 3)
    n_window = len(np.asarray(entry_row['q2_obs']))
    injected = int(entry_row.get('n_contaminants', 0) or 0)
    injected += int(entry_row.get('second_phase_lines', 0) or 0)
    rate = surplus_contaminant_rate(injected, n_window)
    if rate <= 0.0:
        return q2_holdout, hkl_holdout, 0
    # Keyed on the entry, not on (entry, bundle) -- the convention `FomPatterns.mechanism_rng`
    # already uses, so a bundle differing only in contaminant count does not thereby get a
    # different realisation. `entry_id` is passed rather than read off the row because the caller
    # indexes the entry table by the join keys, which makes it the index and not a column.
    rng = np.random.default_rng(derived_seed(f'surplus_contaminants:{entry_id}', seed))
    return add_surplus_contaminants(q2_holdout, hkl_holdout, rate*q2_holdout.size, rng)


def entry_id_of(entry_key, join_keys):
    """The entry id out of whatever `_join_keys` produced -- a scalar for one key, a tuple for two."""
    if len(join_keys) == 1:
        return entry_key
    return entry_key[list(join_keys).index('entry_id')]


# How a candidate's hold-out peak list is built. S10a gate 4 found that cubic candidates are
# fitted on TEN peaks where everything else gets twenty (R5, the `n_peaks` column), so for a cubic
# cell `q2_obs[10:]` are *already* peaks it was never fitted to -- and `q2_holdout` does not contain
# them. That is a free 50 % larger budget on exactly the two lattices where S09 found `M_sym` loses
# to M20 (cF -8.93 pp, 1 gained / 11 lost, C2-F-096, C2-F-101). C2-Q-026 asks whether using it
# repairs the loss, and it must be answered as a PAIRED arm: campaign 1 tried four remedies for a
# single losing lattice and all four were worse than nothing (F-088), so a wider cubic budget is a
# hypothesis to test and never a fix to adopt into an aggregate.
#
#   'surplus'      the uniform definition. Hold-out peaks are `q2_holdout[:n_extra]` for every
#                  lattice, which is what S10a measured and what every other step means by
#                  hold-out.
#   'free_window'  **fixed pattern length.** On a (20 + n_extra)-peak pattern, a candidate fitted
#                  on `n_peaks` peaks holds out everything above it: `q2_obs[n_peaks:20]` followed
#                  by `q2_holdout[:n_extra]`. Cubic gets 10 + n_extra peaks, everything else gets
#                  n_extra and is byte-identical to 'surplus'. This is what an instrument actually
#                  hands you, so it is the primary arm.
#   'free_equal'   **fixed hold-out budget.** The same concatenated list, truncated to `n_extra`,
#                  so cubic scores peaks 11-15 rather than 21-25 at the same count. Separates
#                  *which* peaks from *how many*, and is the secondary arm.
#
# **The budget axis does not mean the same thing in 'free_equal', and that is the point.** In
# 'surplus' and 'free_window' an `n_extra` of 5 always consumes five SURPLUS peaks, so it always
# describes a 25-peak pattern. In 'free_equal' it consumes five HOLD-OUT peaks wherever they come
# from -- and for a cubic candidate the first ten of those are window peaks, so `n_extra <= 10`
# consumes **no surplus at all** and describes a plain 20-peak pattern. That is not a defect in the
# axis; it is the arm's most useful property, because it means a cubic cell can be scored out of
# sample on data that carries no surplus whatever. Label it as hold-out peaks, never as surplus.
#
# The window peaks a cubic cell holds out carry the window's own contaminants already, because they
# are real observed lines; nothing is seeded into them and nothing needs to be (C2-F-098).
HOLDOUT_MODES = ('surplus', 'free_window', 'free_equal')


def holdout_peaks(q2_obs, q2_holdout, n_peaks, mode):
    """The peak list a candidate is scored on, and how many of it a budget of `n_extra` takes.

    Returns `(peaks, offset)`: `peaks` is the ordered hold-out list and `offset` is how many of it
    a budget consumes before the surplus, so a caller asks for `peaks[:offset + n_extra]`. Both are
    returned rather than one list per budget because the list is a property of the entry and the
    budget is a property of the sweep.
    """
    if mode not in HOLDOUT_MODES:
        raise ValueError(f'mode must be one of {HOLDOUT_MODES}, got {mode!r}')
    if mode == 'surplus':
        return q2_holdout, 0
    unused = np.asarray(q2_obs, dtype=np.float64)[int(n_peaks):]
    peaks = np.concatenate([unused, q2_holdout]) if unused.size else q2_holdout
    return peaks, (int(unused.size) if mode == 'free_window' else 0)


def holdout_merits(candidates, entries, n_extra_values=HOLDOUT_N_EXTRA, contaminate=True,
                   seed=HOLDOUT_CONTAMINANT_SEED, models_directory=None, mode='surplus',
                   sigma_multiplier=1.0):
    """Score every candidate on the surplus peaks it was never fitted to, at several peak budgets.

    S10's object. The cell was refined against the 20-peak window and these peaks are not among it,
    so **there is no refit anywhere here** -- that is the entire claim, and a refit would turn this
    into the cross-validated family campaign 2 dropped as a designed negative.

    Returns a frame aligned to `candidates`' index carrying `holdout_columns(n_extra_values)`. An
    entry whose surplus is shorter than `n_extra` gets **NaN, not zero**, at that budget: it is
    missing rather than bad, and a paired comparison has to drop it rather than score it as a
    failure. This is the pitfall campaign 1's restricted-population footnote exists for.

    Grouped exactly as `reduced_merits` groups -- by (entry, lattice, extinction group) -- so the
    cctbx-backed reference list is built once per group, and the reference lines are then handed to
    `get_holdout_fom` rather than rebuilt per budget, since they depend on the cell alone. That
    matmul is the whole cost of the merit, so passing it makes a seven-point sweep cost roughly one
    evaluation instead of seven.

    **`ho_M20` is not "the surplus peaks scored alone", and that is worth knowing before reading
    it.** de Wolff's baseline is Q_N/(2 N_cal), which must count reference lines below a cut-off, so
    it reaches over the whole reference list out to the last surplus peak. `ho_Minfo` and the
    posterior family are per-peak by construction and genuinely do restrict to an arbitrary subset;
    that difference is the reason Minfo is carried here as a baseline at all (DWMM, 2026-09-01).
    """
    from mlindex.utilities.FigureOfMerits import get_holdout_fom
    from mlindex.utilities.FigureOfMerits import get_M20_likelihood
    from mlindex.utilities.FigureOfMerits import get_M_rev_sym

    names = list(holdout_columns(n_extra_values))
    out = pd.DataFrame(index=candidates.index, columns=names, dtype=float)
    if candidates.empty:
        return out

    join_keys = _join_keys(candidates, entries)
    entry_index = entries.set_index(join_keys)
    surplus_cache, window_cache = {}, {}

    group_keys = list(join_keys) + ['lattice_system', 'bravais_lattice', 'spacegroup', 'n_peaks']
    for key, group in candidates.groupby(group_keys, sort=False):
        entry_key = key[0] if len(join_keys) == 1 else key[:len(join_keys)]
        lattice_system, bravais_lattice, spacegroup, n_peaks = key[len(join_keys):]

        if entry_key not in surplus_cache:
            row = entry_index.loc[entry_key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            surplus_cache[entry_key] = (
                contaminated_holdout(row, entry_id_of(entry_key, join_keys), seed=seed)
                if contaminate
                else (np.asarray(row['q2_holdout'], dtype=np.float64),
                      np.asarray(row['hkl_holdout']).reshape(-1, 3), 0)
                )
            window_cache[entry_key] = np.asarray(row['q2_obs'], dtype=np.float64)
        q2_holdout, _, _ = surplus_cache[entry_key]
        # The peak list depends on the candidate's own fitted-peak count, so it is built inside the
        # group loop rather than cached on the entry: `n_peaks` is a group key precisely because
        # cubic and non-cubic candidates of one pattern hold out different peaks (C2-F-101).
        q2_scored, offset = holdout_peaks(
            window_cache[entry_key], q2_holdout, n_peaks, mode)

        xnn = np.vstack([np.asarray(value, dtype=np.float64) for value in group['xnn']])
        hkl_ref = hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory)
        q2_ref_calc = reference_lines(
            xnn, lattice_system, bravais_lattice, spacegroup, models_directory)
        reciprocal_volume = np.asarray(group['reciprocal_volume'], dtype=np.float64)
        reach = q2_ref_calc.max(axis=1)

        # The peaks the cell was ACTUALLY FITTED TO, which is what sigma must be estimated from
        # for S10c's posterior columns. Cubic is fitted on ten and everything else on twenty (R5),
        # so this is not `q2_obs` and using `q2_obs` would give a cubic candidate a sigma fitted
        # partly to peaks it never saw -- the in-sample/out-of-sample line this whole step rests on.
        q2_fitted = window_cache[entry_key][:int(n_peaks)]
        # Fitted once per group, not once per budget: the window sigma is a property of the cell
        # and the peaks it was refined against, and neither moves as the sweep widens. Six budgets
        # would otherwise pay for six identical scans of the whole reference list.
        sigma_window = None
        if q2_fitted.size:
            from mlindex.utilities.FigureOfMerits import get_assignment_sigma
            sigma_window, _ = get_assignment_sigma(q2_fitted, q2_ref_calc, lattice_system)

        for n_extra in n_extra_values:
            # The budget is always quoted in SURPLUS peaks, so the sweep axis means the same thing
            # in every mode and a 25-peak pattern is n_extra = 5 whatever the candidate holds out.
            # `offset` is what the free-window mode adds on top, and it is free.
            if q2_holdout.size < n_extra:
                continue
            prefix = q2_scored[:offset + n_extra]
            values = _holdout_block(
                prefix, xnn, hkl_ref, q2_ref_calc, lattice_system, bravais_lattice,
                reciprocal_volume, reach, get_holdout_fom, get_M_rev_sym, get_M20_likelihood,
                q2_fitted=q2_fitted, sigma_window=sigma_window,
                sigma_multiplier=sigma_multiplier)
            for name, column in values.items():
                out.loc[group.index, holdout_column(name, n_extra)] = column
    return out


def _holdout_block(prefix, xnn, hkl_ref, q2_ref_calc, lattice_system, bravais_lattice,
                   reciprocal_volume, reach, get_holdout_fom, get_M_rev_sym, get_M20_likelihood,
                   q2_fitted=None, sigma_window=None, sigma_multiplier=1.0):
    """One peak budget's worth of columns. Split out so the group loop stays readable."""
    from mlindex.utilities.FigureOfMerits import get_holdout_posterior_fom
    from mlindex.utilities.numba_functions import fast_assign

    # q2_ref_calc is handed in rather than rebuilt: it is a property of the cell, not of the peaks.
    predictive = get_holdout_fom(
        prefix, xnn, hkl_ref, lattice_system, bravais_lattice, q2_ref_calc=q2_ref_calc)

    assign = fast_assign(prefix, q2_ref_calc)
    q2_calc = np.take_along_axis(q2_ref_calc, assign, axis=1)

    # The counting window becomes the surplus interval itself: _reversed_line_terms takes q_min from
    # the reference line nearest the FIRST peak it is given and q_max from the last assigned line,
    # so handing it the surplus asks exactly "does this cell over-predict lines out there". Whether
    # that window holds enough lines to support the merit is S10a's gate 2, which is why N_cal is
    # returned and stored rather than inferred from a floored M_rev of 0.0 (C2-Q-017).
    M_tilde, M_rev, M_sym, N_cal = get_M_rev_sym(
        prefix, q2_calc, q2_ref_calc, return_n_cal=True)

    # Minfo's terms are a sum over peaks, so the per-peak mean is what compares across budgets.
    _, _, Minfo = get_M20_likelihood(prefix, q2_calc, bravais_lattice, reciprocal_volume)
    Minfo = Minfo/max(prefix.size, 1)

    # S10c: sigma from the fitted window, spent on the surplus. `q2_ref_calc` is handed through
    # for the same reason every other column gets it -- it is a property of the cell, and building
    # it is the dominant cost of a seven-point sweep.
    if q2_fitted is None or np.asarray(q2_fitted).size == 0:
        nan = np.full(q2_ref_calc.shape[0], np.nan)
        posterior = {'ho_post': nan.copy(), 'ho_post_logmean': nan.copy(),
                     'ho_evidence': nan.copy()}
    else:
        posterior = get_holdout_posterior_fom(
            q2_fitted, prefix, xnn, hkl_ref, lattice_system, q2_ref_calc=q2_ref_calc,
            sigma=sigma_window, sigma_multiplier=sigma_multiplier)

    return {
        'ho_M20': predictive['ho_M20'],
        'ho_M': predictive['ho_M'],
        'ho_raw': predictive['ho_raw'],
        'ho_tail_nll': predictive['ho_tail_nll'],
        'ho_n_scored': predictive['ho_n_scored'],
        'ho_M_tilde': M_tilde,
        'ho_M_rev': M_rev,
        'ho_M_sym': M_sym,
        'ho_N_cal': np.asarray(N_cal, dtype=float),
        'ho_Minfo': Minfo,
        **posterior,
        # Gate 1: does the reference list reach the peak being asked about at all? Where it does
        # not, ho_* is measuring the list running out rather than the cell mispredicting.
        'ho_ref_reach': (reach >= float(prefix[-1])).astype(float),
        }


# The keys a feature row is identified by. candidate_id is only meaningful within its own
# (entry, lattice), so all four are needed to join a feature matrix back to the pool.
ZOO_KEY_COLUMNS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id')

# Columns zoo_features reads. Anything else is dead weight in a 10M-row projection.
ZOO_CANDIDATE_COLUMNS = (
    'entry_id',
    'bravais_lattice',
    'lattice_system',
    'candidate_id',
    'xnn',
    'spacegroup',
    'n_peaks',
    'M20',
    )


def zoo_features(candidates, entries, g_min=1.0, min_discrepancy=0.0,
                 sigma_entrywise=True, models_directory=None):
    """Every figure of merit in the zoo, for a whole candidate frame.

    `FigureOfMerits.compute_all` costs ~268 get_M20-equivalents (S01_fom_cost.csv), so it is
    evaluated **once** here and each merit is then ranked as a plain column. Computing it inside
    a `FomMetrics.evaluate(score=callable)` would pay that cost once per merit instead.

    Returns (features, sigma_treatment): a frame carrying ZOO_KEY_COLUMNS and one float column
    per feature, aligned to `candidates`' own index, and the map from feature name to
    'free' / 'in-sample' / 'assumed'. A column labelled anything but 'free' is not
    sigma-free and PROTOCOL section 3 rule 4 applies to it.

    Grouped by (entry, n_peaks) on the outside and (lattice, extinction group) inside, for two
    reasons. The reference list is cctbx-backed and is built once per extinction group rather
    than once per row; and `sigma_entrywise` is a property of the *entry's whole pool*, so the
    inner groups are collected before any merit is evaluated. n_peaks joins the outer key
    because the cubic models are scored on ten peaks and everything else on twenty, and the
    two cannot share a residual-scale estimate.

    `g_min` enters only the two Werner quantities and enters both multiplicatively:
    V_crit is proportional to 1/g_min, so `V_over_Vcrit` scales linearly in it and
    `M_werner_frac` scales linearly in it *uniformly across candidates*. So the stored
    columns are at the caller's `g_min` and any other floor is a rescale of them --
    `M_werner_frac`'s ranking within an entry does not depend on g_min at all, and only the
    V/V_crit = 1 boundary moves (Q14).
    """
    from mlindex.utilities.FigureOfMerits import compute_all
    from mlindex.utilities.FigureOfMerits import estimate_sigma_entrywise

    join_keys = _join_keys(candidates, entries)
    peaks = entries.set_index(join_keys)['q2_obs']

    collected = {}
    treatments = {}
    for outer_key, entry_group in candidates.groupby(list(join_keys) + ['n_peaks'], sort=False):
        n_peaks = int(outer_key[-1])
        peak_key = outer_key[0] if len(join_keys) == 1 else outer_key[:len(join_keys)]
        # The optimizer truncates the peak list with a plain prefix slice; cut it the same way.
        q2_obs = np.asarray(peaks.loc[peak_key], dtype=np.float64)[:n_peaks]

        inner_keys = ['lattice_system', 'bravais_lattice', 'spacegroup']
        prepared = []
        for (lattice_system, bravais_lattice, spacegroup), group in \
                entry_group.groupby(inner_keys, sort=False):
            xnn = np.vstack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
            q2_ref_calc, _, _, q2_calc = assign_lines(
                q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, models_directory
                )
            prepared.append(
                (group.index, lattice_system, bravais_lattice, xnn, q2_calc, q2_ref_calc)
                )

        sigma = None
        if sigma_entrywise:
            sigma = estimate_sigma_entrywise(
                q2_obs, np.vstack([item[4] for item in prepared])
                )

        for index, lattice_system, bravais_lattice, xnn, q2_calc, q2_ref_calc in prepared:
            output = compute_all(
                q2_obs, q2_calc, q2_ref_calc, xnn, lattice_system, bravais_lattice,
                sigma_entrywise=sigma, g_min=g_min, min_discrepancy=min_discrepancy,
                )
            treatments.update(output['sigma_treatment'])
            for name, values in output['features'].items():
                collected.setdefault(name, {}).update(zip(index, np.asarray(values, dtype=float)))

    features = pd.DataFrame(
        {name: pd.Series(values) for name, values in collected.items()}
        ).reindex(candidates.index)
    keys = [column for column in ZOO_KEY_COLUMNS if column in candidates.columns]
    return pd.concat([candidates[keys], features], axis=1), treatments



# The columns cv_features reads. Same projection as the zoo, and for the same reason: this is a
# 10M-row read and anything else on it is dead weight.
CV_CANDIDATE_COLUMNS = ZOO_CANDIDATE_COLUMNS


def cv_features(candidates, entries, schemes=('random', 'contiguous', 'high_q'), n_folds=5,
                seed=12345, sigma_entrywise=True, min_discrepancy=0.0, holdout_peaks=None,
                models_directory=None):
    """S10's predictive merits for a whole candidate frame: cross-validated, and held out.

    Built the same shape as `zoo_features` and for the same two reasons -- the reference list is
    cctbx-backed and is built once per extinction group rather than once per row, and the entrywise
    residual scale is a property of the entry's *whole* pool, so the inner groups are collected
    before anything is scored. It is a separate function rather than a branch of `zoo_features`
    because it costs ~29 get_M20-equivalents per fold scheme and nothing that already has the zoo
    should have to pay that to re-read it.

    `holdout_peaks`, when given, is a frame of (entry_id, condition_bundle, q2_holdout) from
    `run_fom_cv_holdout.py`. Entries missing from it, or carrying no surplus lines, get no `ho_`
    columns rather than zeros -- "this entry has no hold-out set" and "this candidate scored badly
    on its hold-out set" are different statements and must not be merged (the S10 handoff asks for
    the applicability fraction to be reported, which needs them separable).

    Returns (features, sigma_treatment), aligned to `candidates`' own index and keyed on
    ZOO_KEY_COLUMNS, exactly as `zoo_features` is, so the two matrices join on the same keys.
    """
    from mlindex.utilities.FigureOfMerits import SIGMA_TREATMENT
    from mlindex.utilities.FigureOfMerits import estimate_sigma_entrywise
    from mlindex.utilities.FigureOfMerits import get_cv_fom
    from mlindex.utilities.FigureOfMerits import get_holdout_fom
    from mlindex.utilities.FigureOfMerits import get_insample_fom

    join_keys = _join_keys(candidates, entries)
    peaks = entries.set_index(join_keys)['q2_obs']
    extra_peaks = None
    if holdout_peaks is not None:
        extra_keys = [key for key in join_keys if key in holdout_peaks.columns]
        extra_peaks = holdout_peaks.set_index(extra_keys)['q2_holdout']

    collected = {}
    for outer_key, entry_group in candidates.groupby(list(join_keys) + ['n_peaks'], sort=False):
        n_peaks = int(outer_key[-1])
        peak_key = outer_key[0] if len(join_keys) == 1 else outer_key[:len(join_keys)]
        # The optimizer truncates the peak list with a plain prefix slice; cut it the same way.
        q2_obs = np.asarray(peaks.loc[peak_key], dtype=np.float64)[:n_peaks]
        q2_holdout = None
        if extra_peaks is not None and peak_key in extra_peaks.index:
            q2_holdout = np.asarray(extra_peaks.loc[peak_key], dtype=np.float64)
            if q2_holdout.size == 0:
                q2_holdout = None

        inner_keys = ['lattice_system', 'bravais_lattice', 'spacegroup']
        prepared = []
        for (lattice_system, bravais_lattice, spacegroup), group in \
                entry_group.groupby(inner_keys, sort=False):
            xnn = np.vstack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
            q2_ref_calc, _, hkl, q2_calc = assign_lines(
                q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, models_directory
                )
            hkl_ref = hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory)
            prepared.append(
                (group.index, lattice_system, bravais_lattice, xnn, hkl, hkl_ref, q2_calc,
                 q2_ref_calc)
                )

        sigma = None
        if sigma_entrywise:
            sigma = estimate_sigma_entrywise(
                q2_obs, np.vstack([item[6] for item in prepared])
                )

        for (index, lattice_system, bravais_lattice, xnn, hkl, hkl_ref, q2_calc,
             q2_ref_calc) in prepared:
            # The in-sample partner first, so is_M/cv_M is a like-for-like ratio rather than a
            # comparison against a merit with a different baseline.
            # q2_calc and q2_ref_calc come straight from assign_lines rather than being
            # rederived, so is_M20 is get_M20 on exactly the arrays the pipeline used. See
            # get_insample_fom's docstring for why a 1e-16 rederivation is not good enough.
            features = dict(get_insample_fom(
                q2_obs, xnn, hkl, lattice_system, bravais_lattice,
                q2_calc=q2_calc, q2_ref_calc=q2_ref_calc,
                sigma_entrywise=sigma, min_discrepancy=min_discrepancy,
                ))
            for scheme in schemes:
                output = get_cv_fom(
                    q2_obs, xnn, hkl, hkl_ref, lattice_system, bravais_lattice,
                    scheme=scheme, n_folds=n_folds, seed=seed, sigma_entrywise=sigma,
                    min_discrepancy=min_discrepancy,
                    )
                features.update(
                    {f'{name}__{scheme}': values for name, values in output.items()}
                    )
            if q2_holdout is not None:
                features.update(get_holdout_fom(
                    q2_holdout, xnn, hkl_ref, lattice_system, bravais_lattice,
                    sigma_entrywise=sigma, min_discrepancy=min_discrepancy,
                    ))
            for name, values in features.items():
                collected.setdefault(name, {}).update(
                    zip(index, np.asarray(values, dtype=float))
                    )

    frame = pd.DataFrame(
        {name: pd.Series(values) for name, values in collected.items()}
        ).reindex(candidates.index)
    keys = [column for column in ZOO_KEY_COLUMNS if column in candidates.columns]
    treatments = {
        name: SIGMA_TREATMENT[name.split('__')[0]] for name in frame.columns
        }
    return pd.concat([candidates[keys], frame], axis=1), treatments


# label_frame's cost is almost entirely validate_candidate_known_bl: profiled at 600 candidates it
# is 95% of the time, calling np.isclose ~252 times per candidate as it searches the off-by-two
# multiplier and permutation space. That is ~9 ms/candidate, which over a full S04 grid of ~20M
# candidates is ~50 h and makes consolidation impossible in one job.
#
def label_frame(candidates, entries, rtol=1e-2, n_processes=1):
    """Attach correctness labels, which need ground truth the dump hook cannot see.

    **Batched, per (entry, lattice system).** The scalar `validate_candidate_known_bl` costs
    ~9 ms a candidate; `label_known_bl_batch` answers the same question for a whole block and was
    measured at 1 584x with zero disagreements over 57.4 M rows (F-166), extended in campaign 2 to
    return `is_off_by_two` as well and gated against the scalar routine on all seven lattice
    systems in `tests/test_candidate_validation_batch.py`. At Benchmark B's ~2.5 billion survivor
    rows the scalar form is not an option, and the process pool that used to wrap it is gone with
    it -- 128 processes multiplying a 1 584x speedup is not a trade anyone needs to make.

    `n_processes` is accepted and ignored, so callers written against the old signature still run.

    Four columns are per-candidate arithmetic:

    * `is_correct` / `is_off_by_two` from the batch labeller, which takes the *partial* truth for
      the candidate's own lattice system -- `TRUTH_SLICE` is an index list and not a range, since
      monoclinic takes beta and not alpha.
    * `xnn_distance_to_truth`, defined only when the candidate's lattice matches the true one,
      because the xnn vectors have different lengths and meanings otherwise. The true cell is
      converted to the candidate's partial form first: `xnn_true` is always the full six
      components while a candidate's `xnn` is partial, and comparing them as stored leaves the
      distance null for every lattice except triclinic, silently.
    * `hkl_true_in_basis`, the truth's reflections in **this candidate's** basis. Null for a
      candidate that is not correct -- not because it was not computed, but because no basis
      change relates a wrong cell to the truth and the quantity does not exist there (R15).

    `is_degenerate` is NOT set here. Campaign 1 shipped it null on the candidate table; campaign 2
    defines it on the true lattice's Niggli reduced cell, which makes it one value per pattern and
    therefore an entry column (C2-F-043, `mlindex/utilities/LatticeDegeneracy.py`).

    `prior_target` is not set here either, and it is not in `LABEL_COLUMNS`: it has no operational
    definition anywhere in the record -- see C2-Q-015. Every plausible definition is a function of
    columns this pool already stores, so it can be added offline once defined, without
    regenerating. Shipping it null in every row is the one option that is ruled out (C2-F-046).
    """
    from mlindex.optimization.CandidateValidation import basis_change_known_bl_batch
    from mlindex.optimization.CandidateValidation import hkl_in_candidate_basis
    from mlindex.optimization.CandidateValidation import label_known_bl_batch
    from mlindex.optimization.CandidateValidation import TRUTH_SLICE
    from mlindex.utilities.UnitCellTools import get_partial_unit_cell
    from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell

    result = candidates.copy()
    n_rows = result.shape[0]
    is_correct = np.zeros(n_rows, dtype=bool)
    is_off_by_two = np.zeros(n_rows, dtype=bool)
    xnn_distance = np.full(n_rows, np.nan)
    volume_ratio = np.full(n_rows, np.nan)
    hkl_in_basis = np.empty(n_rows, dtype=object)
    hkl_in_basis[:] = None
    if n_rows == 0:
        return _with_label_columns(result, is_correct, is_off_by_two, xnn_distance,
                                   volume_ratio, hkl_in_basis)

    truth = entries.set_index('entry_id')
    has_hkl_true = 'hkl_true' in truth.columns

    for (entry_id, lattice_system), group in result.groupby(
            ['entry_id', 'lattice_system'], sort=False):
        entry = truth.loc[entry_id]
        positions = result.index.get_indexer(group.index)
        unit_cell_true = np.asarray(entry['unit_cell_true'], dtype=np.float64)
        # Stacked per group, never across the frame: `unit_cell` is the PARTIAL cell for the
        # candidate's own lattice system, so it is one number for cubic and six for triclinic and
        # a pool holds all fourteen lattices. Stacking the column whole raises, and a pool that
        # happened to hold one system would have hidden that until the first real run.
        predicted = np.stack([np.asarray(cell, dtype=np.float64)
                              for cell in group['unit_cell']])
        predicted = predicted[:, :len(TRUTH_SLICE[lattice_system])]

        correct, off_by_two = label_known_bl_batch(
            unit_cell_true[TRUTH_SLICE[lattice_system]], predicted, lattice_system, rtol=rtol)
        is_correct[positions] = correct
        is_off_by_two[positions] = off_by_two
        volume_ratio[positions] = (group['volume'].to_numpy(dtype=np.float64)
                                   / float(entry['volume_true']))

        if group['bravais_lattice'].iloc[0] == entry['bravais_lattice_true']:
            partial_true = get_partial_unit_cell(unit_cell_true, lattice_system=lattice_system)
            xnn_true = get_xnn_from_unit_cell(
                partial_true[np.newaxis], partial_unit_cell=True,
                lattice_system=lattice_system)[0]
            xnn_pred = np.stack([np.asarray(xnn, dtype=np.float64) for xnn in group['xnn']])
            xnn_distance[positions] = np.linalg.norm(xnn_pred - xnn_true, axis=1)

        if has_hkl_true and correct.any():
            # Only the correct rows, which are under 1 % of the pool, so the per-candidate 3x3
            # inverse this needs costs nothing against the labelling itself.
            changes = basis_change_known_bl_batch(
                unit_cell_true[TRUTH_SLICE[lattice_system]], predicted, lattice_system, rtol=rtol)
            hkl_true = np.asarray(entry['hkl_true'], dtype=np.float64).reshape(-1, 3)
            for offset in np.flatnonzero(correct):
                reexpressed = hkl_in_candidate_basis(hkl_true, changes[offset])
                if reexpressed is not None:
                    hkl_in_basis[positions[offset]] = reexpressed.reshape(-1)

    return _with_label_columns(result, is_correct, is_off_by_two, xnn_distance,
                               volume_ratio, hkl_in_basis)


def _with_label_columns(frame, is_correct, is_off_by_two, xnn_distance, volume_ratio,
                        hkl_in_basis):
    frame['is_correct'] = is_correct
    frame['is_off_by_two'] = is_off_by_two
    frame['xnn_distance_to_truth'] = xnn_distance
    frame['volume_ratio_to_truth'] = volume_ratio
    frame['hkl_true_in_basis'] = hkl_in_basis
    return frame


# The group a candidate pool is defined over, and the group every rank in this schema is taken
# within. `final_rank` is "over all survivors of this (entry, lattice)", and the condition belongs
# in the key because one crystal appears once per bundle (R8).
POOL_KEY_COLUMNS = ('entry_id', 'condition_bundle', 'bravais_lattice')


def subsample_negatives(candidates, merit_columns=('M20',), top_k=200, negative_rate=0.05,
                        base_seed=12345, correct_column='is_correct'):
    """Thin the pool without changing what it measures.

    Three classes of row survive, in this precedence:

    * **`correct`** -- every correct candidate, unconditionally. The base rate is under 1 %, they
      are the entire signal, and no sampling rule may touch them. This is why the order of
      operations is forced: label, THEN subsample, then consolidate.
    * **`top_k`** -- every candidate inside the top *K* by *each* reported merit, unioned. Ranking
      metrics are then exact to depth *K* for every one of those merits, which is what makes a
      subsampled pool usable for the campaign's headline numbers rather than only for fitting.
    * **`sampled`** -- a Bernoulli sample of everything else, carrying `1 / negative_rate` as its
      weight so any fit or aggregate over the thinned pool is unbiased for the full one.

    `sampling_weight` is 1.0 for the first two classes because they are retained with certainty,
    and every fit must use the column -- without it the negatives are silently reweighted by
    whatever the retention rate happened to be.

    The RNG is keyed on the pool, not on the run: the same (entry, condition, lattice) thins the
    same way whichever shard it lands in and however many shards there are. Campaign 1 seeded once
    per pool and advanced with every entry, which is why no subset of its benchmark could be
    regenerated comparably (R17, PROTOCOL section 6).

    `negative_rate` of 1.0 keeps everything and is not a no-op: it still writes the bookkeeping
    columns, so a pool generated whole is readable by exactly the same loader as a thinned one.
    """
    if candidates.empty:
        return candidates
    missing = [column for column in merit_columns if column not in candidates.columns]
    if missing:
        raise ValueError(f'subsample_negatives cannot rank on absent columns: {missing}')
    if not 0.0 < negative_rate <= 1.0:
        raise ValueError(f'negative_rate must be in (0, 1], got {negative_rate}')

    result = candidates.copy()
    keys = [column for column in POOL_KEY_COLUMNS if column in result.columns]
    reason = np.full(result.shape[0], 'sampled', dtype=object)
    weight = np.full(result.shape[0], 1.0 / negative_rate)

    # Top-K by each merit, unioned. `rank` rather than `nlargest` so ties are resolved the same
    # way for every merit and a tie at the K-th place cannot silently drop a row.
    in_top_k = np.zeros(result.shape[0], dtype=bool)
    if keys:
        grouped = result.groupby(keys, sort=False)
        for column in merit_columns:
            ranks = grouped[column].rank(method='first', ascending=False)
            in_top_k |= (ranks <= top_k).to_numpy()
    else:
        for column in merit_columns:
            ranks = result[column].rank(method='first', ascending=False)
            in_top_k |= (ranks <= top_k).to_numpy()
    reason[in_top_k] = 'top_k'
    weight[in_top_k] = 1.0

    # And a Bernoulli draw over the rest, keyed per pool AND indexed by `candidate_id`.
    #
    # Indexing the draws by the candidate's own id rather than by its position in the frame is
    # what makes retention a property of the row instead of a property of the row ORDER. A
    # positional draw gives a different answer when the same pool arrives sorted differently,
    # concatenated from a different number of shards, or filtered upstream -- and the difference
    # is invisible, because either answer is a valid sample. Campaign 1's R17 is the same defect
    # one level up: a pool that cannot be regenerated row for row cannot be checked at all.
    keep = in_top_k.copy()
    if negative_rate >= 1.0:
        keep[:] = True
    elif keys and 'candidate_id' in result.columns:
        candidate_id = result['candidate_id'].to_numpy(dtype=np.int64)
        for key, group in result.groupby(keys, sort=False):
            positions = result.index.get_indexer(group.index)
            to_draw = ~in_top_k[positions]
            if not to_draw.any():
                continue
            ids = candidate_id[positions]
            rng = np.random.default_rng(
                _derived_pool_seed(key if isinstance(key, tuple) else (key,), base_seed))
            draws = rng.random(int(ids.max()) + 1)
            keep[positions[to_draw]] = draws[ids[to_draw]] < negative_rate
    else:
        raise ValueError(
            'subsample_negatives needs candidate_id and the pool key columns '
            f'{POOL_KEY_COLUMNS} to draw reproducibly; got {sorted(result.columns)}')

    # Correctness last, so it overrides both the reason and the weight of anything it touches.
    if correct_column in result.columns:
        correct = result[correct_column].to_numpy(dtype=bool)
        keep |= correct
        reason[correct] = 'correct'
        weight[correct] = 1.0

    result['retained_reason'] = reason
    result['sampling_weight'] = weight
    return result.loc[keep].reset_index(drop=True)


def _derived_pool_seed(key, base_seed):
    """A stable seed for one candidate pool. `hash()` is salted per process and cannot be used."""
    digest = hashlib.sha256(f"{base_seed}:{':'.join(str(part) for part in key)}".encode('utf-8'))
    return int.from_bytes(digest.digest()[:8], 'big')


def write_candidate_shard(frame, out_dir, shard_tag):
    path = Path(out_dir) / f'candidates_{shard_tag}.parquet'
    _to_parquet(frame, path)
    return path


def write_predownsample_shard(frame, out_dir, shard_tag):
    path = Path(out_dir) / f'predownsample_{shard_tag}.parquet'
    _to_parquet(frame, path)
    return path


def write_entry_table(frame, out_dir, shard_tag):
    path = Path(out_dir) / f'entries_{shard_tag}.parquet'
    _to_parquet(frame, path)
    return path


# The Arrow type a column must have on disk whatever this particular shard happens to hold.
#
# A column that is null in every row of a shard has no type for Arrow to infer, so pandas writes it
# as parquet type `null` rather than its real type -- and two shards of one run then disagree,
# which makes a strict multi-file read fail. `hkl_true_in_basis` is the one that bites: it is null
# for every candidate that is not correct, correctly so, and a shard containing no correct
# candidate at all has it null throughout. That happened often enough in the first Benchmark B run
# to stop consolidation dead (C2-F-073).
#
# Only columns that can legitimately be all-null need to be here. `xnn` and `unit_cell` are never
# null, so their type is always inferred.
NULLABLE_COLUMN_TYPES = ('hkl_true_in_basis', 'merit_at_prune')


def _nullable_column_types():
    """{column: Arrow type}, built lazily because pyarrow is an optional dependency here."""
    import pyarrow as pa

    return {'hkl_true_in_basis': pa.list_(pa.int16()),
            'merit_at_prune': pa.list_(pa.float64())}


def _typed_table(frame):
    """The frame as an Arrow table, with all-null columns given their declared type."""
    import pyarrow as pa

    declared = _nullable_column_types()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    fields, changed = [], False
    for field in table.schema:
        wanted = declared.get(field.name)
        if wanted is not None and pa.types.is_null(field.type):
            fields.append(pa.field(field.name, wanted))
            changed = True
        else:
            fields.append(field)
    if not changed:
        return table
    return table.cast(pa.schema(fields, metadata=table.schema.metadata))


def _to_parquet(frame, path, row_group_size=None):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Writing the FOM benchmark needs pyarrow. Install it with "
            "'pip install mlindex[fom]' or 'pip install pyarrow'."
            ) from error
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_typed_table(frame), path,
                   **({'row_group_size': row_group_size} if row_group_size else {}))


def write_manifest(out_dir, **run_metadata):
    """Record what makes the pool regenerable.

    The pool is reproducible only at a fixed pool topology: each optimizer's search RNG is
    seeded once per pool and advances with every entry it runs, so an entry's candidate set
    depends on its position in its pool's queue. Recording the seed alone is not enough.
    """
    path = Path(out_dir) / 'manifest.json'
    payload = dict(run_metadata)
    payload['schema_version'] = SCHEMA_VERSION
    payload['candidate_columns'] = list(CANDIDATE_COLUMNS)
    payload['entry_columns'] = list(ENTRY_COLUMNS)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as manifest_file:
        json.dump(payload, manifest_file, indent=2, sort_keys=True)
    return path


def load_manifest(root):
    """The pool's `manifest.json`, or `None` if the directory has none.

    Returns the dict as written. Callers that need one field should reach for a helper beside
    this one rather than indexing it directly, because a consolidated pool nests the per-bundle
    manifests under `bundle_manifests` and a single-bundle run does not.
    """
    path = Path(root) / 'manifest.json'
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as manifest_file:
        return json.load(manifest_file)


def subsample_depth(root):
    """The pool's negative-subsampling depth *K*, or `None` if it was not subsampled.

    `(top_k, subsampled)`, read from the manifest and from every per-bundle manifest a
    consolidated pool carries, because the pool is only exact to the *smallest* K any of its
    bundles was written at. A pool whose manifest says `subsampled: false` returns `None`, which
    means "no depth limit" rather than "unknown" -- `FomMetrics` distinguishes the two, since an
    absent manifest cannot be read as an unsubsampled pool.
    """
    manifest = load_manifest(root)
    if manifest is None:
        return None, None
    manifests = [manifest] + list(manifest.get('bundle_manifests', {}).values())
    subsampled = [bool(entry.get('subsampled')) for entry in manifests
                  if 'subsampled' in entry]
    if not subsampled:
        return None, None
    if not any(subsampled):
        return None, False
    depths = [entry.get('top_k') for entry in manifests if entry.get('subsampled')]
    depths = [int(depth) for depth in depths if depth is not None]
    return (min(depths) if depths else None), True


def load_entries(root):
    # `entries*` rather than `entries_*`: the per-pool shards a generation run writes are
    # entries_<tag>.parquet, but the consolidated pool is a single entries.parquet, and both must
    # load through the same function.
    return _read_glob(root, 'entries*.parquet')


def load_entries_ids(root):
    """Just the entry_id column, for callers that only need the entry set.

    Reading the whole entry table pulls q2_obs, hkl_true and the ground-truth cell for every
    pattern; the intersection pre-pass in consolidation needs none of it.
    """
    return _read_glob(root, 'entries*.parquet', columns=['entry_id'])['entry_id']


def bundle_from_candidate_path(path):
    """Which condition bundle a candidate shard belongs to, from its filename.

    The bundle is *not* a candidate column -- the dump hook runs per (entry, Bravais lattice)
    and knows nothing about the condition the driver applied -- so the filename is the only
    record of it on the candidate side. Both layouts have to be read: a generation run writes
    `candidates_<bundle>_shard<NN>ofNN_pool<NN>.parquet` and consolidation rewrites those as
    `candidates_<bundle>_<BL>.parquet`.

    This matters more than a convenience: after consolidation `entries.parquet` holds one row
    per (entry, bundle), so `entry_id` alone is no longer a key and a join on it silently
    fans out 7x. Nor is (entry_id, q2_digest) a key -- C4 and C5 leave 157 of 5 922 entries
    with an identical peak list, because both dropout counts cap on the same surplus.
    """
    from mlindex.command_line.run import BRAVAIS_LATTICES

    stem = Path(path).stem
    if not stem.startswith('candidates_'):
        raise ValueError(f'Not a candidate shard: {path}')
    tag = stem[len('candidates_'):]
    match = re.fullmatch(r'(?P<bundle>.+)_shard\d+of\d+_pool\d+', tag)
    if match:
        return match.group('bundle')
    bundle, _, last = tag.rpartition('_')
    if bundle and last in BRAVAIS_LATTICES:
        return bundle
    return tag


def available_bundles(root):
    """The condition bundles present under `root`, in sorted order."""
    paths = sorted(Path(root).glob('candidates_*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No candidates_*.parquet under {root}')
    return sorted({bundle_from_candidate_path(path) for path in paths})


def candidate_columns_present(root):
    """The candidate columns this pool actually has, read from one shard's schema.

    A column set is not fixed across the two campaigns: `is_degenerate` was a candidate column in
    Benchmark A and is an **entry** column here, because campaign 2's definition is a statement
    about the pattern's own true lattice and so takes one value per pattern (C2-F-043). A caller
    that projects a fixed column list onto a pool that lacks one gets `ArrowInvalid` from the
    parquet reader, not a missing column it can recover from.

    Reads the schema, not the rows.
    """
    # Imported here, as everywhere else in this module: pyarrow is an optional dependency and the
    # merit and labelling paths must import without it.
    import pyarrow.parquet as pq

    paths = sorted(Path(root).glob('candidates*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No candidates*.parquet under {root}')
    # `schema_arrow`, not `schema`. The parquet schema flattens list columns to their leaf
    # paths -- `xnn` becomes `xnn.list.element` -- so a caller projecting on this would drop
    # every list column it asked for and fail somewhere else entirely.
    return set(pq.ParquetFile(paths[0]).schema_arrow.names)


def load_candidates(root, split=None, bravais_lattices=None, columns=None, bundles=None):
    """Candidate shards under `root`, tagged with the bundle their filename names.

    `columns` is the only real memory knob -- the filters below run after the read, because
    parquet here carries no row-group statistics worth pushing a predicate into. Projecting
    to the nine columns the metrics need takes one bundle of ~3M candidates from ~2 GB to
    ~220 MB. `xnn` and `unit_cell` are what cost; ask for them deliberately.
    """
    paths = sorted(Path(root).glob('candidates*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No candidates*.parquet under {root}')
    wanted = None if bundles is None else set(bundles)
    projection = None if columns is None else _with_required(columns)
    frames = []
    for path in paths:
        bundle = bundle_from_candidate_path(path)
        if wanted is not None and bundle not in wanted:
            continue
        frame = _read_parquet(path, columns=projection)
        frame['condition_bundle'] = bundle
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f'No candidate shard under {root} for bundles {sorted(wanted)}')
    frame = pd.concat(frames, ignore_index=True)
    if bravais_lattices is not None:
        frame = frame.loc[frame['bravais_lattice'].isin(bravais_lattices)]
    if split is not None:
        entries = load_entries(root)
        keep = entries.loc[entries['split'] == split, 'entry_id']
        frame = frame.loc[frame['entry_id'].isin(set(keep))]
    return frame.reset_index(drop=True)


def bundle_frames(root, merit_dir=None, columns=None, require_merits=False):
    """Candidate frames for a pool, one condition bundle at a time, merit sidecars joined on.

    The route to `FomMetrics.evaluate` for any pool whose merits live in sidecars. `evaluate` given
    a *root* projects columns from the candidate parquet alone and so cannot see them; given an
    iterable of frames it takes whatever it is handed. Lifted here from
    `run_fom_floor_report.arm_frames`, which is now a thin wrapper, so the two cannot drift.

    **ONE FRAME PER BUNDLE, NOT PER FILE.** The pool is partitioned by (bundle, lattice) but the
    ranking is across all fourteen lattices at once -- that is the indexer's actual problem, and
    ranking within a lattice is a different and much easier one (PROTOCOL section 10's worst
    anti-pattern). Yielding a file at a time would reduce each lattice separately and the pooled
    ranking would be a fiction. `FomMetrics._combine_reductions` refuses it outright, which is how
    this was caught rather than shipped.

    A generator, so one bundle is resident at a time: Benchmark B is 122 GB and is not held.

    `columns` is projected. The pool's list-valued columns -- `xnn`, `unit_cell`,
    `merit_at_prune`, `hkl_true_in_basis` -- are what make an Arrow to pandas conversion
    expensive, one Python object per row per column, and none is needed once the merits are
    recomputed.

    `require_merits` turns the silent failure into a loud one. A missing sidecar, or a join that
    matches nothing, leaves the merit columns null -- and NaN sorts *last*, so the merit does not
    raise, it simply reports as the worst score in the zoo. That is indistinguishable from a real
    measurement, which is the same reason `evaluate` refuses an uncertifiable rank rather than
    warning about it.
    """
    root = Path(root)
    # A sequence, so a pool can carry more than one sidecar set -- the verified `merits/` and, for
    # C2-Q-025, a `merits_soft/` written separately rather than by invalidating the first.
    if merit_dir is None:
        merit_dirs = [root/'merits']
    elif isinstance(merit_dir, (str, Path)):
        merit_dirs = [Path(merit_dir)]
    else:
        merit_dirs = [Path(one) for one in merit_dir]
    available = candidate_columns_present(root)
    projection = None if columns is None else [name for name in columns if name in available]
    known = set(RECOMPUTED_MERIT_COLUMNS) | set(SOFT_MERIT_COLUMNS)
    wanted_merits = [] if columns is None else [name for name in columns if name in known]

    by_bundle = {}
    for path in sorted(root.glob('candidates*.parquet')):
        by_bundle.setdefault(bundle_from_candidate_path(path), []).append(path)

    for bundle, paths in sorted(by_bundle.items()):
        frames = []
        for path in paths:
            frame = _read_parquet(path, columns=projection)
            if 'condition_bundle' not in frame.columns:
                frame['condition_bundle'] = bundle
            found = False
            for one in merit_dirs:
                sidecar = one/path.name
                if not sidecar.exists():
                    continue
                merits = pd.read_parquet(sidecar)
                keys = [key for key in ZOO_KEY_COLUMNS if key in merits.columns]
                frame = frame.merge(merits, on=keys, how='left', validate='1:1')
                found = True
            if not found and require_merits:
                raise FileNotFoundError(
                    f'No merit sidecar for {path.name} in {merit_dirs}. Write it with '
                    f'run_fom_floor_merits.py --pool {root}; a missing sidecar would otherwise '
                    f'leave every recomputed merit null, and NaN ranks last rather than raising.')
            if require_merits:
                missing = [name for name in wanted_merits if name not in frame.columns
                           or frame[name].isna().any()]
                if missing:
                    raise ValueError(
                        f'{path.name}: merit columns {missing} are absent or partly null after '
                        f'the sidecar join. A left join that misses becomes NaN, and NaN sorts '
                        f'last -- the merit would report as uniformly worthless.')
            frames.append(frame)
        yield pd.concat(frames, ignore_index=True)


def load_benchmark(root, split=None, bravais_lattices=None, rtol=1e-2, label=True,
                   columns=None, bundles=None):
    """Candidates joined to their entries and labelled -- the frame S05 onwards consumes.

    Labelling is skipped when the shards already carry the label columns, which the S04
    consolidation writes. Relabelling a pool that has them would cost ~9 ms/candidate to
    reproduce what is already on disk.
    """
    entries = load_entries(root)
    candidates = load_candidates(
        root, split=split, bravais_lattices=bravais_lattices, columns=columns, bundles=bundles,
        )
    _check_join(candidates, entries)
    if label and not has_labels(candidates):
        candidates = label_frame(candidates, entries, rtol=rtol)
    return candidates.merge(
        entries.drop(columns=['q2_digest']), on=_join_keys(candidates, entries), how='left',
        validate='m:1',
        )


def has_labels(frame):
    """Does this frame already carry usable correctness labels?

    Every column of `LABEL_COLUMNS` has to be present, and `is_correct` has to be populated.
    Campaign 1's version filtered `is_degenerate` out of the check because that column shipped
    null and would otherwise have called every pool unlabelled; campaign 2 does not have that
    problem, because it does not ship a column it cannot fill -- `is_degenerate` is an entry
    column (C2-F-043) and `prior_target` is not produced at all (C2-Q-015).

    `hkl_true_in_basis` is null on most rows and that is correct: no basis change relates a wrong
    cell to the truth, so the quantity does not exist there. Only `is_correct` is checked for
    nulls, because it is the only one that is null exactly when the labelling did not run.
    """
    if any(column not in frame.columns for column in LABEL_COLUMNS):
        return False
    return not frame['is_correct'].isna().any()


def _with_required(columns):
    """Column projections must keep the join and filter keys, whatever else they ask for."""
    projection = list(columns)
    for required in ('entry_id', 'q2_digest', 'bravais_lattice'):
        if required not in projection:
            projection.append(required)
    return projection


def _join_keys(candidates, entries):
    """(entry_id, condition_bundle) for a consolidated pool, entry_id for a single-bundle one.

    A consolidated entry table holds the same entry under every condition, so `entry_id` alone
    is not a key there and joining on it fans every candidate out once per bundle. A generation
    run's shard directory holds one bundle, and its entry table predates the column, so the
    single key is still correct there.
    """
    if 'condition_bundle' in candidates.columns and 'condition_bundle' in entries.columns:
        return ['entry_id', 'condition_bundle']
    return ['entry_id']


def _check_join(candidates, entries):
    """Every candidate must find exactly one entry row, and agree with its peak list."""
    keys = _join_keys(candidates, entries)
    digests = entries.set_index(keys)['q2_digest']
    if digests.index.has_duplicates:
        n_duplicated = int(digests.index.duplicated().sum())
        raise ValueError(
            f'{n_duplicated} entry rows share {tuple(keys)}, so the entry table is not a key. '
            'A consolidated pool needs condition_bundle on both sides; load the candidates '
            'with load_candidates, which reads it from the filename.'
            )
    candidate_keys = (pd.MultiIndex.from_frame(candidates[keys]) if len(keys) > 1
                      else pd.Index(candidates['entry_id']))
    missing = candidate_keys.difference(digests.index)
    if len(missing):
        raise ValueError(
            f'{len(missing)} {tuple(keys)} present in the candidate shards but absent from the '
            f'entry table, e.g. {missing[:3].tolist()}'
            )
    mismatched = candidates.loc[
        candidates['q2_digest'].to_numpy() != digests.loc[candidate_keys].to_numpy()
        ]
    if mismatched.shape[0]:
        raise ValueError(
            f'{mismatched.shape[0]} candidate rows carry a q2_digest that disagrees with '
            'their entry. The shards do not belong to the same run.'
            )


def _read_glob(root, pattern, columns=None):
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No {pattern} under {root}')
    frames = [_read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _read_parquet(path, columns=None):
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Reading the FOM benchmark needs pyarrow. Install it with "
            "'pip install mlindex[fom]' or 'pip install pyarrow'."
            ) from error
    return pd.read_parquet(path, columns=columns)
