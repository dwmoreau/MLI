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


SCHEMA_VERSION = '2'

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
)


# Attached by label_frame, and -- since the S04 consolidation -- written into the candidate shards
# themselves. Labelling costs ~9 ms/candidate, so a pool that already carries them must not be
# relabelled just because a loader defaults to label=True.
LABEL_COLUMNS = (
    'is_correct',
    'is_off_by_two',
    'xnn_distance_to_truth',
    'volume_ratio_to_truth',
    'is_degenerate',
)


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
                })
    frame = pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))
    return frame


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
        columns['retained_by'].append(record['retained_by'].astype(np.int64))
        columns['n_entering'].append(np.repeat(record['n_entering'], n_candidates))
        columns['prune_m20_threshold'].append(
            np.repeat(record['prune_m20_threshold'], n_candidates))
        columns['downsample_radius'].append(
            np.repeat(record['downsample_radius'], n_candidates))

    if not columns['entry_id']:
        return pd.DataFrame(columns=list(PREDOWNSAMPLE_COLUMNS))

    data = {}
    for name, chunks in columns.items():
        if name == 'xnn':
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


def hkl_ref_for(lattice_system, bravais_lattice, spacegroup, models_directory=None):
    """Reference Miller indices for one extinction group.

    `Candidates.assign_extinction_group` builds its calculated lines from the model's
    truncated `hkl_ref` filtered by the winning extinction group, so reproducing its M20
    means starting from exactly that list. `get_spacegroup_hkl_ref` reaches into cctbx and
    is far too slow to call per row, hence the cache.
    """
    key = (str(models_directory), lattice_system, bravais_lattice)
    if key not in _HKL_REF_CACHE:
        path = _hkl_ref_path(lattice_system, bravais_lattice, models_directory)
        _HKL_REF_CACHE[key] = get_spacegroup_hkl_ref(
            np.load(path), bravais_lattice=bravais_lattice
            )
    return _HKL_REF_CACHE[key][spacegroup]


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
    peaks = entries.set_index('entry_id')['q2_obs']
    columns = ['M20_recomputed', 'Minfo_recomputed', 'n_indexed_recomputed']
    out = pd.DataFrame(index=candidates.index, columns=columns, dtype=float)

    group_keys = ['entry_id', 'lattice_system', 'bravais_lattice', 'spacegroup', 'n_peaks']
    for (entry_id, lattice_system, bravais_lattice, spacegroup, n_peaks), group in \
            candidates.groupby(group_keys, sort=False):
        # Cubic models take ten peaks and everything else twenty; the optimizer truncates
        # with a plain prefix slice, so the entry's list is cut the same way here.
        q2_obs = np.asarray(peaks.loc[entry_id], dtype=np.float64)[:n_peaks]
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
# The work is per-candidate and independent, so it parallelises exactly rather than being rewritten.
# That is the right trade here: validate_candidate_known_bl is production comparison logic that S02's
# labelling also depends on, and a node has 128 cores idle while this runs.
def _label_chunk(payload):
    """Module-level and picklable, so this survives the spawn start method."""
    candidates, truth, rtol = payload
    return label_frame(candidates, truth, rtol=rtol, n_processes=1)


def label_frame_parallel(candidates, entries, rtol=1e-2, n_processes=1, chunk_size=20000):
    """label_frame over a process pool. Identical output; only the wall-clock differs."""
    if n_processes <= 1 or candidates.shape[0] <= chunk_size:
        return label_frame(candidates, entries, rtol=rtol)
    from multiprocessing import Pool

    # Only the columns label_frame reads, so each chunk pickles a few hundred kB of ground truth
    # rather than the entry table's q2_obs and hkl_true arrays.
    truth = entries[['entry_id', 'unit_cell_true', 'volume_true', 'bravais_lattice_true']]
    chunks = [candidates.iloc[start:start + chunk_size].reset_index(drop=True)
              for start in range(0, candidates.shape[0], chunk_size)]
    with Pool(processes=n_processes) as pool:
        labelled = pool.map(_label_chunk, [(chunk, truth, rtol) for chunk in chunks])
    return pd.concat(labelled, ignore_index=True)


def label_frame(candidates, entries, rtol=1e-2, n_processes=1):
    """Attach correctness labels, which need ground truth the dump hook cannot see.

    `is_correct` and `is_off_by_two` come from `validate_candidate_known_bl`, which takes
    the *partial* unit cell for the candidate's own Bravais lattice -- the representation
    the optimizer produces. `xnn_distance_to_truth` is only defined when the candidate's
    lattice matches the true one, because the xnn vectors have different lengths and
    meanings otherwise.

    Note the two xnn representations do not match as stored: `xnn_true` comes from the
    dataset's `reindexed_xnn`, which is always the full six components, while a candidate's
    `xnn` is the partial form for its lattice system (one component for cubic, two for
    tetragonal, ...). The true cell is therefore converted to the candidate's partial form
    before the distance is taken. Comparing them directly would leave the distance null for
    every lattice except triclinic, silently.
    """
    if n_processes > 1:
        return label_frame_parallel(candidates, entries, rtol=rtol, n_processes=n_processes)

    from mlindex.optimization.CandidateValidation import validate_candidate_known_bl
    from mlindex.utilities.UnitCellTools import get_partial_unit_cell
    from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell

    truth = entries.set_index('entry_id')
    is_correct = np.zeros(candidates.shape[0], dtype=bool)
    is_off_by_two = np.zeros(candidates.shape[0], dtype=bool)
    xnn_distance = np.full(candidates.shape[0], np.nan)
    volume_ratio = np.full(candidates.shape[0], np.nan)

    for position, (_, row) in enumerate(candidates.iterrows()):
        entry = truth.loc[row['entry_id']]
        correct, off_by_two = validate_candidate_known_bl(
            unit_cell_true=np.asarray(entry['unit_cell_true'], dtype=np.float64),
            unit_cell_pred=np.asarray(row['unit_cell'], dtype=np.float64),
            bravais_lattice_pred=row['bravais_lattice'],
            rtol=rtol,
            )
        # validate_candidate_known_bl falls off the end of several branches, returning
        # None rather than False when nothing matches.
        is_correct[position] = bool(correct)
        is_off_by_two[position] = bool(off_by_two)
        volume_ratio[position] = row['volume'] / entry['volume_true']
        if row['bravais_lattice'] == entry['bravais_lattice_true']:
            partial_true = get_partial_unit_cell(
                np.asarray(entry['unit_cell_true'], dtype=np.float64),
                lattice_system=row['lattice_system'],
                )
            xnn_true = get_xnn_from_unit_cell(
                partial_true[np.newaxis], partial_unit_cell=True,
                lattice_system=row['lattice_system'],
                )[0]
            xnn_pred = np.asarray(row['xnn'], dtype=np.float64)
            xnn_distance[position] = np.linalg.norm(xnn_pred - xnn_true)

    result = candidates.copy()
    result['is_correct'] = is_correct
    result['is_off_by_two'] = is_off_by_two
    result['xnn_distance_to_truth'] = xnn_distance
    result['volume_ratio_to_truth'] = volume_ratio
    # PLAN 6.5 defines is_degenerate against the true cell's calculated lines "within
    # sigma(q2)", but PROTOCOL 3 forbids assuming sigma is known and the true cell's
    # Bravais lattice generally implies a different reference list. Left null pending Q27.
    result['is_degenerate'] = pd.NA
    return result


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


def _to_parquet(frame, path):
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Writing the FOM benchmark needs pyarrow. Install it with "
            "'pip install mlindex[fom]' or 'pip install pyarrow'."
            ) from error
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


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

    `is_degenerate` is deliberately excluded: it ships null (SCHEMA.md, Q28), so requiring it
    to be populated would call every pool unlabelled.
    """
    required = [column for column in LABEL_COLUMNS if column != 'is_degenerate']
    if any(column not in frame.columns for column in required):
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
