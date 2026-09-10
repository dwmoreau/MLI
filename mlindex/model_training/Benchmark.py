"""Reading a stored candidate pool: shards, sidecars, the entry table and the manifest.

A pool is one directory per arm, holding a candidate shard per (condition bundle, Bravais
lattice), an entry table with one row per pattern-condition and its ground truth, a manifest
recording how it was generated, and optional sidecars carrying columns computed after the fact.

Everything here **refuses rather than warns**. A missing sidecar, a join that matches nothing, an
arm with no completion stamp: each produces a frame that still parses and columns that still sort,
so the failure would arrive as a merit that looks like the worst in the zoo, or a number attached
to the wrong pattern. Those are indistinguishable from measurements, which is why they raise.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_NAME = 'manifest.json'
ENTRY_TABLE_NAME = 'entries.parquet'
COMPLETION_NAME = 'complete.json'
MERIT_SIDECAR = 'merits'

# The key a candidate row is identified by, and the key a sidecar joins on.
CANDIDATE_KEY = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']
ENTRY_KEY = ['entry_id', 'condition_bundle']

# Campaign 1's zero-error bundle. With no measurement error the residual denominator vanishes and
# M20 diverges arithmetically, so nothing may be fitted, reported or investigated on it. It is the
# largest thing on disk and a loader that globs a directory picks it up in silence, so it is
# refused by name rather than left to a reader to notice.
REFUSED_BUNDLES = ('error0_cont0',)


def _refuse_zero_error(bundles):
    banned = [bundle for bundle in bundles
              if any(bundle.endswith(suffix) or bundle == suffix for suffix in REFUSED_BUNDLES)]
    if banned:
        raise ValueError(
            f'Refusing the zero-error bundle(s) {banned}. With no measurement error M20 is '
            'ill-conditioned -- it reaches 1e12 to 1e14 and agrees only to a few percent '
            'relative -- so nothing may be fitted or reported on them.')


def load_manifest(pool_dir):
    """The manifest of a stored pool."""
    path = Path(pool_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f'No manifest at {path}; this is not a consolidated pool.')
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def check_complete(pool_dir, require=True):
    """Refuse an arm that was never stamped complete.

    A killed run is indistinguishable from a finished one by its contents: the shards it did
    write are valid, and pairing against them silently compares over whichever crystals happened
    to finish. `require=False` is for inspecting a partial arm deliberately.
    """
    path = Path(pool_dir) / COMPLETION_NAME
    if path.is_file():
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    if require:
        raise FileNotFoundError(
            f'No completion stamp at {path}. A killed run looks finished by its contents, so an '
            'unstamped arm is refused. Pass require=False to inspect one deliberately.')
    return None


def available_bundles(pool_dir):
    """The condition bundles a pool holds, from its shard names."""
    names = set()
    for path in Path(pool_dir).glob('candidates_*.parquet'):
        stem = path.stem[len('candidates_'):]
        names.add(stem.rsplit('_', 1)[0])
    return sorted(names)


def candidate_shards(pool_dir, bundle, bravais_lattices=None):
    """The shard paths for one condition bundle, in canonical lattice order."""
    from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

    lattices = list(BRAVAIS_LATTICES if bravais_lattices is None else bravais_lattices)
    paths = []
    for lattice in lattices:
        path = Path(pool_dir) / f'candidates_{bundle}_{lattice}.parquet'
        if path.is_file():
            paths.append((lattice, path))
    if not paths:
        raise FileNotFoundError(
            f'No candidate shards for bundle {bundle!r} in {pool_dir}. '
            f'Available bundles: {available_bundles(pool_dir)}')
    return paths


def load_entries(pool_dir, bundles=None, columns=None):
    """The entry table: one row per pattern-condition, with its ground truth."""
    path = Path(pool_dir) / ENTRY_TABLE_NAME
    if not path.is_file():
        raise FileNotFoundError(f'No entry table at {path}.')
    entries = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    if bundles is not None:
        _refuse_zero_error(bundles)
        entries = entries.loc[entries['condition_bundle'].isin(list(bundles))]
    else:
        _refuse_zero_error(sorted(entries['condition_bundle'].unique().tolist()))
    return entries.reset_index(drop=True)


def _check_join(frame, before, what):
    if frame.shape[0] != before:
        raise ValueError(
            f'The {what} join changed the row count from {before} to {frame.shape[0]}. A join '
            'that duplicates or drops rows silently re-attaches numbers to the wrong candidates.')


def _check_no_nulls(frame, columns, what):
    missing = [name for name in columns if frame[name].isna().any()]
    if missing:
        counts = {name: int(frame[name].isna().sum()) for name in missing}
        raise ValueError(
            f'The {what} join left nulls in {counts}. A null score sorts last, so the column '
            'would report as the worst ranking in the comparison -- indistinguishable from a '
            'measurement.')


def load_candidates(pool_dir, bundle, columns=None, bravais_lattices=None, sidecars=('merits',),
                    sidecar_columns=None):
    """Every candidate of one condition bundle, with its sidecar columns joined on.

    Shards are read in canonical lattice order so a pool reads the same way twice; the ranking
    does not depend on it, because the tie-break is total, but a stable order makes two runs
    comparable row for row.
    """
    _refuse_zero_error([bundle])
    frames = []
    for _, path in candidate_shards(pool_dir, bundle, bravais_lattices):
        frames.append(pd.read_parquet(path, columns=list(columns) if columns else None))
    frame = pd.concat(frames, ignore_index=True)

    for sidecar in sidecars or ():
        directory = Path(pool_dir) / sidecar
        if not directory.is_dir():
            raise FileNotFoundError(
                f'No {sidecar!r} sidecar at {directory}. It is required rather than optional: a '
                'missing merit column would rank last and read as a bad merit.')
        wanted = (sidecar_columns or {}).get(sidecar)
        pieces = []
        for _, path in candidate_shards(directory, bundle, bravais_lattices):
            pieces.append(pd.read_parquet(
                path, columns=(CANDIDATE_KEY + list(wanted)) if wanted else None))
        sidecar_frame = pd.concat(pieces, ignore_index=True)
        overlap = [name for name in sidecar_frame.columns
                   if name in frame.columns and name not in CANDIDATE_KEY]
        sidecar_frame = sidecar_frame.drop(columns=overlap)
        before = frame.shape[0]
        frame = frame.merge(sidecar_frame, on=CANDIDATE_KEY, how='left')
        _check_join(frame, before, f'{sidecar} sidecar')
        _check_no_nulls(frame, [name for name in sidecar_frame.columns
                                if name not in CANDIDATE_KEY], f'{sidecar} sidecar')
    return frame


def attach_entry_columns(candidates, entries, columns):
    """Broadcast per-pattern columns onto the candidates of that pattern.

    Ground truth lives on the entry table because it is one value per pattern, not per candidate,
    but the reduction needs some of it -- whether the crystal's lattice is degenerate, and which
    lattice is the true one -- alongside every row.
    """
    wanted = [name for name in columns if name not in candidates.columns]
    if not wanted:
        return candidates
    before = candidates.shape[0]
    merged = candidates.merge(entries[ENTRY_KEY + wanted], on=ENTRY_KEY, how='left')
    _check_join(merged, before, 'entry table')
    return merged


def check_peak_digests(candidates, entries):
    """The candidate and entry tables must agree about which pattern each row belongs to.

    They are written by different code paths and joined later, so a mis-join is otherwise silent:
    every column parses and the numbers simply attach to the wrong crystal.
    """
    if 'q2_digest' not in candidates.columns or 'q2_digest' not in entries.columns:
        raise ValueError('Both tables must carry q2_digest for the join to be checkable.')
    expected = entries.set_index(ENTRY_KEY)['q2_digest']
    keys = pd.MultiIndex.from_frame(candidates[ENTRY_KEY])
    wanted = expected.reindex(keys).to_numpy()
    got = candidates['q2_digest'].to_numpy()
    disagree = np.count_nonzero(wanted != got)
    if disagree:
        raise ValueError(
            f'{disagree} candidate rows carry a q2_digest that disagrees with their entry. The '
            'two tables are describing different peak lists for the same pattern.')
    return True
