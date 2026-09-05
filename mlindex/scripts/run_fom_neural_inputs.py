"""Compute S14's network inputs for a pool and persist them beside the data.

    python mlindex/scripts/run_fom_neural_inputs.py --pool mlindex/data/fom_full_c2_pool \\
        --stage entries                                   # block A, one forward pass per entry
    python mlindex/scripts/run_fom_neural_inputs.py --pool mlindex/data/fom_full_c2_pool \\
        --stage candidates --processes 8                  # block B + the claimed-pair readout
    python mlindex/scripts/run_fom_neural_inputs.py --pool ... --verify

Two files feed the network (`docs/fom_campaign2/handoffs/S14_neural_score.md`), and they live in
`<pool>/neural_inputs/`:

  prior_entries.parquet     one row per (entry_id, condition_bundle): the prior network's fourteen
                            lattice probabilities renormalised over its SUPPORT (NaN outside it),
                            the two entropies, and E[log V | lattice] for each lattice. Constant
                            within an entry, so it joins through `FomCombiner.neural_covariates`.
  prior_joint_tables.npz    the masked joint table per entry, kept so the per-candidate readout
                            below needs no forward pass and no keras in a worker.
  candidates_<bundle>_<BL>.parquet
                            one per candidate shard, keyed on the four zoo keys: the twenty
                            per-peak assignment posteriors, log sigma, and the prior read at the
                            candidate's own claimed (volume, lattice) pair
                            (`FomBenchmark.neural_inputs`).

**The `entries` stage runs first and alone.** It is the only part that needs keras
(`KERAS_BACKEND=torch`, set below before anything imports it), it takes seconds per thousand
entries, and the candidate workers read its output rather than recomputing it. A candidate whose
entry has no prior row makes `neural_inputs` raise rather than write NaN, because a NaN prior is
indistinguishable from an out-of-support claim.

**`--keys-from` is the NERSC path.** S12's full-scale fit frames carry the four keys of every row
they fitted and none of the columns this needs (`xnn`, `q2_obs`), and the pool they came from is
on the cluster. Given a glob of those frames, only rows whose keys appear in them are computed --
about 7.6 M rows rather than 880 M -- and `_meta.json` records how many keys were asked for and
how many were matched, per shard; a shortfall is a hard failure, since a key that is not matched
would become a NaN input on the laptop with no other symptom.

Cost: the assignment posterior is 44-180 microseconds a candidate depending on the lattice
(`S02_zoo_cost.csv`), dominated by the reference-line pass `assign_lines` shares with the
structural sidecar. The fully retained pool's 43.3 M candidates is one to two hours on eight
processes.
"""
import argparse
import json
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.model_training import FomCombiner  # noqa: E402
from mlindex.model_training.FomMetrics import BRAVAIS_LATTICES  # noqa: E402


JOIN_KEYS = ('entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id')
ENTRY_KEYS = ('entry_id', 'condition_bundle')

# All `neural_inputs` reads from the candidate side. `volume` is the claimed volume the prior is
# read at; the rest is what `assign_lines` needs.
CANDIDATE_COLUMNS = JOIN_KEYS + ('lattice_system', 'spacegroup', 'n_peaks', 'xnn', 'volume')
ENTRY_COLUMNS = ENTRY_KEYS + ('q2_obs',)

DEFAULT_PRIOR_DIR = os.path.join('mlindex', 'models', 'fom_prior', 'main', 'global')
OUT_DIRNAME = FomCombiner.SIDECAR_DIRS['prior_claimed']
ENTRY_FILE = FomCombiner.NEURAL_ENTRY_FILE
TABLES_FILE = 'prior_joint_tables.npz'

# Rows held at once per worker. `neural_inputs` groups by (entry, lattice, group, n_peaks) and is
# exact on any subset of a group, so a chunk boundary costs a rebuilt reference list and nothing
# else. Lower than the structural pass because the (n_candidates, n_ref) posterior block is held
# in float64 and the low-symmetry reference lists are long.
CHUNK_ROWS = 500_000

_ENTRY_CACHE = {}
_TABLES_CACHE = {}
_KEYS_CACHE = {}


# ---------------------------------------------------------------------------------------
# Stage 1: the entry-level block, one forward pass per (entry, bundle)
# ---------------------------------------------------------------------------------------
def entry_rows(pool, keys_from=None):
    """The (entry_id, condition_bundle, q2_obs) rows to run the prior on."""
    frame = FomBenchmark.load_entries(pool)
    keep = [name for name in ENTRY_COLUMNS if name in frame.columns]
    frame = frame[keep].drop_duplicates(subset=list(ENTRY_KEYS)).reset_index(drop=True)
    if keys_from:
        wanted = keys_in(keys_from)[list(ENTRY_KEYS)].drop_duplicates()
        frame = frame.merge(wanted, on=list(ENTRY_KEYS), how='inner').reset_index(drop=True)
    return frame


def write_entry_tables(pool, out_dir, prior_dir, keys_from=None, batch_size=256,
                       chunk=20_000):
    """`prior_entries.parquet` and `prior_joint_tables.npz` for every entry of the pool."""
    from mlindex.model_training import PriorNetwork as Prior

    entries = entry_rows(pool, keys_from)
    if not entries.shape[0]:
        raise SystemExit(f'{pool}: no entries to score')
    model = Prior.PriorNetwork.load_prior(prior_dir)
    q2 = np.stack([np.asarray(values, dtype=np.float64)[:model.model_params['peak_length']]
                   for values in entries['q2_obs']])
    if q2.shape[1] != model.model_params['peak_length']:
        raise ValueError(f'entries carry {q2.shape[1]} peaks; the prior takes '
                         f'{model.model_params["peak_length"]}')

    parts = {name: [] for name in ('joint', 'bravais_p', 'logv', 'bravais_entropy',
                                   'branch_entropy')}
    for start in range(0, q2.shape[0], chunk):
        tables = model.entry_tables(q2[start:start + chunk], batch_size=batch_size)
        for name in parts:
            parts[name].append(np.asarray(tables[name]))
    joint = np.concatenate(parts['joint'], axis=0).astype(np.float32)
    log_branch_volumes = np.asarray(tables['log_branch_volumes'], dtype=np.float64)

    out = pd.DataFrame({key: entries[key].to_numpy() for key in ENTRY_KEYS})
    bravais_p = np.concatenate(parts['bravais_p'], axis=0)
    logv = np.concatenate(parts['logv'], axis=0)
    for position, code in enumerate(BRAVAIS_LATTICES):
        out[f'prior_bravais_p_{code}'] = bravais_p[:, position].astype(np.float32)
    out['prior_branch_entropy'] = np.concatenate(parts['branch_entropy']).astype(np.float32)
    out['prior_bravais_entropy'] = np.concatenate(parts['bravais_entropy']).astype(np.float32)
    for position, code in enumerate(BRAVAIS_LATTICES):
        out[f'prior_logv_{code}'] = logv[:, position].astype(np.float32)
    # Every column FomCombiner will ask for, by name, before anything is written.
    for name in list(FomCombiner.PRIOR_ENTRY) + list(FomCombiner.PRIOR_VOLUME):
        assert name in out.columns, name

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir/ENTRY_FILE, index=False)
    np.savez(out_dir/TABLES_FILE, joint=joint, log_branch_volumes=log_branch_volumes,
             entry_id=out['entry_id'].to_numpy().astype(str),
             condition_bundle=out['condition_bundle'].to_numpy().astype(str),
             support=np.asarray(model.support, dtype=str),
             support_defaulted=np.asarray(model.support_defaulted))
    return dict(n_entries=int(out.shape[0]), support=list(model.support),
                support_defaulted=bool(model.support_defaulted),
                prior_dir=str(prior_dir), n_volumes=int(joint.shape[1]),
                cubic_readable=bool(np.isfinite(bravais_p[:, :3]).any()))


def load_prior_tables(out_dir):
    """The npz the entries stage wrote, as the dict `FomBenchmark.neural_inputs` takes."""
    out_dir = str(out_dir)
    if out_dir not in _TABLES_CACHE:
        stored = np.load(Path(out_dir)/TABLES_FILE)
        index = pd.MultiIndex.from_arrays(
            [stored['entry_id'].astype(object), stored['condition_bundle'].astype(object)],
            names=list(ENTRY_KEYS))
        _TABLES_CACHE[out_dir] = {
            'joint': stored['joint'].astype(np.float64),
            'log_branch_volumes': stored['log_branch_volumes'],
            'index': index,
            'support': tuple(stored['support'].tolist()),
            }
    return _TABLES_CACHE[out_dir]


# ---------------------------------------------------------------------------------------
# Stage 2: the per-candidate block
# ---------------------------------------------------------------------------------------
def _entries_for(pool):
    """The projected entry table, once per worker process rather than once per file."""
    if pool not in _ENTRY_CACHE:
        frame = FomBenchmark.load_entries(pool)
        keep = [name for name in ENTRY_COLUMNS if name in frame.columns]
        _ENTRY_CACHE[pool] = frame[keep]
    return _ENTRY_CACHE[pool]


def keys_in(patterns):
    """The four-key rows of every frame the glob(s) match, deduplicated."""
    key = tuple(patterns)
    if key not in _KEYS_CACHE:
        paths = []
        for pattern in patterns:
            matched = sorted(Path().glob(pattern)) if any(c in pattern for c in '*?[') \
                else [Path(pattern)]
            paths.extend(matched)
        if not paths:
            raise SystemExit(f'--keys-from matched nothing: {patterns}')
        frames = [pd.read_parquet(path, columns=list(JOIN_KEYS)) for path in paths]
        keys = pd.concat(frames, ignore_index=True).drop_duplicates()
        for name in ('entry_id', 'condition_bundle', 'bravais_lattice'):
            keys[name] = keys[name].astype(object)
        keys['candidate_id'] = keys['candidate_id'].astype(np.int64)
        _KEYS_CACHE[key] = keys.reset_index(drop=True)
    return _KEYS_CACHE[key]


def score_file(task):
    """One candidate file -> one sidecar, streamed. Module-level and picklable: spawn-safe."""
    path, out_path, pool, out_dir, chunk_rows, with_prior, keys_from = task
    entries = _entries_for(pool)
    tables = load_prior_tables(out_dir) if with_prior else None
    keys = keys_in(keys_from) if keys_from else None
    source = pq.ParquetFile(path)
    # `schema_arrow`, not `schema`: a list column such as `xnn` flattens to `xnn.list.element`
    # in the parquet schema and a membership test on it drops the column silently.
    projection = [name for name in CANDIDATE_COLUMNS if name in source.schema_arrow.names]

    pieces, held, written, requested = [], 0, 0, 0
    out = []
    if keys is not None:
        bundle = FomBenchmark.bundle_from_candidate_path(Path(path))
        lattice = Path(path).stem.split('_')[-1]
        keys = keys.loc[(keys['condition_bundle'] == bundle) & (keys['bravais_lattice'] == lattice)]
        requested = int(keys.shape[0])
        if not requested:
            return path, 0, 0
    for index in range(source.num_row_groups):
        block = source.read_row_group(index, columns=projection).to_pandas()
        if 'condition_bundle' not in block.columns:
            block['condition_bundle'] = FomBenchmark.bundle_from_candidate_path(Path(path))
        if keys is not None:
            block = block.merge(keys[list(JOIN_KEYS)], on=list(JOIN_KEYS), how='inner')
        pieces.append(block)
        held += block.shape[0]
        if held < chunk_rows and index < source.num_row_groups - 1:
            continue
        chunk = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
        pieces, held = [], 0
        if not chunk.shape[0]:
            continue
        features = FomBenchmark.neural_inputs(chunk, entries, prior_tables=tables)
        out.append(pd.concat([chunk[list(JOIN_KEYS)].reset_index(drop=True),
                              features.reset_index(drop=True).astype(np.float32)], axis=1))
        written += chunk.shape[0]

    if not out:
        return path, 0, requested
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(out, ignore_index=True)
    frame['prior_in_support'] = frame['prior_in_support'].fillna(0).astype(np.int8)
    frame.to_parquet(out_path, index=False)
    return path, written, requested


# ---------------------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------------------
def verify(pool, out_dir):
    """Check the sidecars. Returns (rows, problems, notes).

    Three things go wrong quietly and are checked from parquet metadata alone: a sidecar missing,
    a sidecar short of its candidate file (or of the keys it was asked for), and a column written
    wholly null because the computation raised for one group and was swallowed. Plus the entry
    table: every (entry, bundle) in `entries.parquet` has a prior row, and the support recorded
    in the npz is the one the tables were written under.
    """
    problems, notes, total = [], [], 0
    out_dir = Path(out_dir)
    meta_path = out_dir/'_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    expected_rows = meta.get('rows_per_file', {})
    keyed = bool(meta.get('keys_from'))

    entry_path = out_dir/ENTRY_FILE
    if not entry_path.exists():
        problems.append(f'{ENTRY_FILE}: MISSING')
    else:
        prior = pd.read_parquet(entry_path, columns=list(ENTRY_KEYS))
        entries = entry_rows(pool, meta.get('keys_from') or None)[list(ENTRY_KEYS)]
        missing = entries.merge(prior, on=list(ENTRY_KEYS), how='left', indicator=True)
        n_missing = int((missing['_merge'] != 'both').sum())
        if n_missing:
            problems.append(f'{ENTRY_FILE}: {n_missing} (entry, bundle) pairs of the entry '
                            f'table have no prior row')
        if (out_dir/TABLES_FILE).exists():
            stored = np.load(out_dir/TABLES_FILE)
            if stored['joint'].shape[0] != prior.shape[0]:
                problems.append(f'{TABLES_FILE}: {stored["joint"].shape[0]} tables against '
                                f'{prior.shape[0]} entry rows')
            notes.append(f'support {tuple(stored["support"].tolist())}'
                         f'{" (defaulted)" if bool(stored["support_defaulted"]) else ""}')
        else:
            problems.append(f'{TABLES_FILE}: MISSING')

    must_fill = list(FomBenchmark.NEURAL_PEAK_COLUMNS[:10]) + [FomBenchmark.NEURAL_SIGMA_COLUMN]
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        sidecar = out_dir/path.name
        lattice = path.stem.split('_')[-1]
        if keyed and expected_rows.get(path.name, 0) == 0:
            if sidecar.exists():
                problems.append(f'{path.name}: sidecar present but no keys were requested')
            continue
        if not sidecar.exists():
            problems.append(f'{path.name}: NO SIDECAR')
            continue
        expected = (expected_rows.get(path.name) if keyed
                    else pq.ParquetFile(path).metadata.num_rows)
        metadata = pq.ParquetFile(sidecar).metadata
        if expected is not None and metadata.num_rows != expected:
            problems.append(f'{path.name}: {metadata.num_rows} rows against {expected}')
            continue
        total += metadata.num_rows
        names = list(metadata.schema.names)
        nulls = {}
        for group in range(metadata.num_row_groups):
            row_group = metadata.row_group(group)
            for column in range(row_group.num_columns):
                stats = row_group.column(column).statistics
                if stats is not None:
                    nulls[names[column]] = nulls.get(names[column], 0) + stats.null_count
        check = list(must_fill)
        if not lattice.startswith('c') and 'prior_joint' in names:
            check.append('prior_joint')
        for name in check:
            if name not in names:
                problems.append(f'{path.name}: {name} absent')
            elif nulls.get(name, 0) == metadata.num_rows:
                problems.append(f'{path.name}: {name} is wholly null')
    return total, problems, notes


def _commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute S14's network inputs and persist them beside a pool")
    parser.add_argument('--pool', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None,
                        help=f'Where the sidecars go. Default is <pool>/{OUT_DIRNAME}')
    parser.add_argument('--stage', choices=('entries', 'candidates', 'all'), default='all')
    parser.add_argument('--prior-dir', type=str, default=DEFAULT_PRIOR_DIR,
                        help='The prior network checkpoint (weights, grid, params). Its recorded '
                             'support decides which lattices read as probabilities')
    parser.add_argument('--no-prior', action='store_true',
                        help='Write the assignment block only; the prior columns stay NaN')
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS)
    parser.add_argument('--keys-from', type=str, nargs='*', default=None,
                        help='Glob(s) of frames carrying the four zoo keys; only those rows are '
                             'computed. The NERSC path over S12\'s exported fit frames')
    parser.add_argument('--verify', action='store_true',
                        help='Check an existing set of sidecars instead of writing any')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    pool = Path(args.pool)
    out_dir = Path(args.out_dir) if args.out_dir else pool/OUT_DIRNAME
    with_prior = not args.no_prior

    if args.verify:
        total, problems, notes = verify(pool, out_dir)
        print(f'{pool}: {total} candidates carry neural inputs in {out_dir}')
        for note in notes:
            print(f'  note {note}')
        for problem in problems:
            print(f'  FAIL {problem}')
        print('all sidecars complete and populated' if not problems
              else f'{len(problems)} problem(s)')
        return 1 if problems else 0

    meta_path = out_dir/'_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    meta.update(pool=str(pool), commit=_commit(), chunk_rows=int(args.chunk_rows),
                keys_from=list(args.keys_from or []), numpy=np.__version__,
                pandas=pd.__version__,
                columns=list(FomBenchmark.NEURAL_INPUT_COLUMNS),
                entry_columns=list(FomCombiner.PRIOR_ENTRY) + list(FomCombiner.PRIOR_VOLUME))

    if args.stage in ('entries', 'all') and with_prior:
        print(f'{pool}: block A over the entry table ({args.prior_dir})', flush=True)
        meta['entries'] = write_entry_tables(pool, out_dir, args.prior_dir, args.keys_from)
        print(f'  {meta["entries"]["n_entries"]} entries; support {meta["entries"]["support"]}'
              f'{" (defaulted)" if meta["entries"]["support_defaulted"] else ""}', flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
    if args.stage == 'entries':
        return 0

    if with_prior and not (out_dir/TABLES_FILE).exists():
        raise SystemExit(f'{out_dir/TABLES_FILE} is missing: run --stage entries first')

    tasks = []
    for path in sorted(pool.glob('candidates*.parquet')):
        out_path = out_dir/path.name
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((str(path), str(out_path), str(pool), str(out_dir), int(args.chunk_rows),
                      with_prior, tuple(args.keys_from or ())))
    rows_per_file = dict(meta.get('rows_per_file', {}))
    if tasks:
        processes = max(1, min(int(args.processes), len(tasks)))
        print(f'{pool}: scoring {len(tasks)} files over {processes} process(es)')
        if processes == 1:
            results = map(score_file, tasks)
        else:
            handle = Pool(processes)
            results = handle.imap_unordered(score_file, tasks)
        for path, rows, requested in results:
            rows_per_file[Path(path).name] = int(rows)
            flag = '' if not args.keys_from or rows == requested else \
                f'   <-- {requested - rows} requested keys NOT FOUND'
            print(f'  {Path(path).name}: {rows}{flag}', flush=True)
            if args.keys_from and rows != requested:
                raise SystemExit(f'{Path(path).name}: {requested} keys requested, {rows} '
                                 f'matched. A key that is not in the pool would become a NaN '
                                 f'input on the laptop with no other symptom.')
        if processes > 1:
            handle.close()
            handle.join()
    else:
        print(f'{pool}: every sidecar is already written')

    meta['rows_per_file'] = rows_per_file
    meta['n_candidates'] = int(sum(rows_per_file.values()))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
    print(f'{pool}: {meta["n_candidates"]} candidates scored -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
