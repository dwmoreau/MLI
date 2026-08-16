"""Benchmark A: the frozen candidate pool that the figure-of-merit work is developed against.

The indexer scores every candidate with M20 and then keeps the best twenty per Bravais
lattice. Comparing a *different* figure of merit against that output is impossible: the
candidates it would have ranked highly were discarded, and the columns needed to evaluate
it were never written. This module is the other half of the dump hook in
`MPIOptimizer._downsample_computation` -- it turns the hook's records into parquet, reads
them back, and recomputes the scores offline so a variant FOM can re-rank a frozen pool
instead of re-running the indexer.

Two tables, joined on ``entry_id``:

    candidates_*.parquet   one row per surviving candidate
    entries.parquet        one row per indexed pattern, with the ground truth

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


def recompute_scores(q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
                     assignment_threshold, models_directory=None):
    """Recompute (M20, Minfo, n_indexed) for candidates sharing one extinction group.

    This is the round trip the acceptance gate measures: it must reproduce the values the
    pipeline stored, from the dumped columns alone.
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


def label_frame(candidates, entries, rtol=1e-2):
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
    return _read_glob(root, 'entries_*.parquet')


def load_candidates(root, split=None, bravais_lattices=None, columns=None):
    frame = _read_glob(root, 'candidates_*.parquet', columns=columns)
    if bravais_lattices is not None:
        frame = frame.loc[frame['bravais_lattice'].isin(bravais_lattices)]
    if split is not None:
        entries = load_entries(root)
        keep = entries.loc[entries['split'] == split, 'entry_id']
        frame = frame.loc[frame['entry_id'].isin(set(keep))]
    return frame.reset_index(drop=True)


def load_benchmark(root, split=None, bravais_lattices=None, rtol=1e-2, label=True):
    """Candidates joined to their entries and labelled -- the frame S05 onwards consumes."""
    entries = load_entries(root)
    candidates = load_candidates(root, split=split, bravais_lattices=bravais_lattices)
    _check_join(candidates, entries)
    if label:
        candidates = label_frame(candidates, entries, rtol=rtol)
    return candidates.merge(
        entries.drop(columns=['q2_digest']), on='entry_id', how='left', validate='m:1',
        )


def _check_join(candidates, entries):
    digests = entries.set_index('entry_id')['q2_digest']
    missing = set(candidates['entry_id']) - set(digests.index)
    if missing:
        raise ValueError(
            f'{len(missing)} entry_id present in the candidate shards but absent from the '
            f'entry table, e.g. {sorted(missing)[:3]}'
            )
    mismatched = candidates.loc[
        candidates['q2_digest'].to_numpy()
        != digests.loc[candidates['entry_id']].to_numpy()
        ]
    if mismatched.shape[0]:
        raise ValueError(
            f'{mismatched.shape[0]} candidate rows carry a q2_digest that disagrees with '
            'their entry. The shards do not belong to the same run.'
            )


def _read_glob(root, pattern, columns=None):
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Reading the FOM benchmark needs pyarrow. Install it with "
            "'pip install mlindex[fom]' or 'pip install pyarrow'."
            ) from error
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No {pattern} under {root}')
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)
