"""S03 -- the prune: which merit, at which stage, at what value.

`prune_below_m20` deletes every candidate scoring below M20 = 5.0 *mid-search*, before the final
refinement. It is the largest lossy cut in the pipeline: campaign 1 measured it deleting 72-88 %
of the correct candidates the search finds on hard patterns. It entered the codebase as a speed
optimisation and its effect on correct cells was first assessed three and a half months later.

DWMM's framing, 2026-08-25: the prune saves considerable time, but that time saving is not worth
it if we are throwing away the correct answer. So both halves go on the same table -- what a cut
costs in wall clock and what it costs in correct cells -- per merit, per stage, per lattice.

THE CUT DECOMPOSES INTO THREE POINTS, AND ALL THREE ARE RECOVERABLE FROM THE THRESHOLD-0 DUMP.
`OptimizerBase._run_loop` (MPIOptimizer.py:138-148) runs the cut, then `refine_cell`,
`standardize_cell`, `correct_off_by_two` and `assign_extinction_group` -- the last of which
replaces the stored M20 with the maximum over the lattice's extinction groups and can only raise
it. So:

  A  m20_at_prune   the value the rule tested: pre-refinement, against the FULL reference list.
                    Stored per candidate, which is what lets one run answer every threshold.
  B  post-refinement recomputed here from the stored (final) xnn against the FULL list.
  C  final           recomputed here from the stored xnn against the SPACEGROUP-NARROWED list.
                     Gated against the dump's own M20 column, which it must reproduce exactly.

B vs A separates refinement from the cut; C vs B separates extinction assignment from refinement.
That is the measurement that decides DWMM's question -- is M20 the wrong *merit* at the cut, or is
the cut simply at the wrong *stage*?

WHAT THIS CANNOT DO. `best_xnn` at the cut was never stored, so stage-A values exist for M20 and
for nothing else. The stage-A sweep over other criteria needs a capture at the cut site and a
re-run; see run_fom_prune_rerun.py. Every stage-A column here except `m20_at_prune` is absent by
construction, not by omission.

BOUNDS THAT TRAVEL WITH EVERY NUMBER (PROTOCOL section 7):
  * The hard arm's 243 entries are 183 `fom-train` + 60 `fom-dev` + 0 `fom-test`. Everything in
    this script is classical and unfitted, so all 243 are reportable; anything fitted (the blend)
    is selected on `fom-train` only.
  * It is a WITHIN-RUN RESTRICTION, not a second run. Restricting a threshold-0 run at 5.0 gives
    the candidates a threshold-5 run would have *admitted*, not the cells it would have
    *produced*. The direction is known and conservative: the restriction reaches 193 entries
    where the real threshold-5 run reaches 198.
  * The hard arm is mC/mP/aP only -- low symmetry throughout, and 99.8 % of its candidates are
    discarded at the production cut against 94.2 % at nominal conditions. The general-population
    arm is not a check on this, it is half the evidence. Both are reported.

    python mlindex/scripts/run_fom_prune_criterion.py --stage merits --arm general
    python mlindex/scripts/run_fom_prune_criterion.py --stage merits --arm hard
    python mlindex/scripts/run_fom_prune_criterion.py --stage cost
    python mlindex/scripts/run_fom_prune_criterion.py --stage stage
    python mlindex/scripts/run_fom_prune_criterion.py --stage retention
    python mlindex/scripts/run_fom_prune_criterion.py --stage dedup

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.utilities.FigureOfMerits import get_M20
from mlindex.utilities.FigureOfMerits import get_M_rev_sym
from mlindex.utilities.FigureOfMerits import get_X_N
from mlindex.utilities.FigureOfMerits import get_n_over
from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.Q2Calculator import Q2Calculator

# The two threshold-0 arms, both untracked run output already on disk. `labels` carries
# `is_correct` row-aligned to `root`'s predownsample shards, so nothing here relabels --
# PROTOCOL section 3 rule 8, and campaign 1 lost this four times.
ARMS = {
    'hard': {
        'root': os.path.join('mlindex', 'characterization', 'fom', 'retention', 't0'),
        'labels': os.path.join('mlindex', 'data', 'fom_prune_labels'),
        'layout': 'archived',
        'note': '243 entries (183 train / 60 dev), true lattices mC/mP/aP only, 4 hard bundles',
        },
    'general': {
        'root': os.path.join('mlindex', 'characterization', 'fom', 'allstrata', 't0'),
        'labels': os.path.join('mlindex', 'data', 'fom_prune_labels_allstrata'),
        'layout': 'archived',
        'note': '210 entries (163 train / 47 dev), all 14 lattices, 15 each, nominal conditions',
        },
    # The Phase 2 re-run (`run_fom_prune_rerun.py`). Same entries and the same peak lists, but a
    # pool of its own -- and it carries `merit_at_prune_*`, which campaign 1's dumps do not. That
    # is the whole reason it exists: it is the only pool on which a criterion can be swept AT the
    # cut rather than after it (C2-R-001). Labels are written inline, so no labels directory.
    'rerun-general': {
        'root': os.path.join('mlindex', 'characterization', 'fom', 'prune_capture', 'general'),
        'labels': None,
        'layout': 'rerun',
        # The re-run reuses the archived peak lists rather than regenerating them, so the entry
        # tables it was driven from are the ones the recompute must read too.
        'entries_root': os.path.join('mlindex', 'characterization', 'fom', 'allstrata', 't0'),
        'note': 'Phase 2 re-run of the general arm: 210 entries, all 14 lattices, threshold 0, '
                'seeded per (entry, Bravais lattice), merits captured at the cut',
        },
    'rerun-hard': {
        'root': os.path.join('mlindex', 'characterization', 'fom', 'prune_capture', 'hard'),
        'labels': None,
        'layout': 'rerun',
        'entries_root': os.path.join('mlindex', 'characterization', 'fom', 'retention', 't0'),
        'note': 'Phase 2 re-run of the hard arm: 243 entries x 4 bundles, threshold 0, '
                'seeded per (entry, Bravais lattice), merits captured at the cut',
        },
    }

MERIT_ROOT = os.path.join('mlindex', 'data', 'fom_prune_criterion')
ARTIFACT_DIR = os.path.join('docs', 'fom_campaign2', 'artifacts')

# The reduced core the decisions log settled on 2026-08-25: three calls returning six columns,
# plus get_M20 as the incumbent baseline. Nothing here needs `compute_all` or FomBenchmark, which
# belong to S08 -- PROTOCOL section 3 rule 10, take what the step uses.
MERITS = ('M20', 'M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap')

# Merits where a LOW value is the good one, so a cut keeps `value <= threshold` rather than
# `value >= threshold`. All three count things that should not be there.
HIGHER_IS_WORSE = ('X_N', 'n_over', 'max_gap')

PRODUCTION_PRUNE_THRESHOLD = 5.0

# Cap on how many (candidate x reference line) floats one q2_ref_calc block may hold, so a
# 68-extinction-group lattice with 1 000 reference lines cannot blow the worker's memory.
MAX_BLOCK_ELEMENTS = 20_000_000


# ---------------------------------------------------------------------------------------------
# the reference lists, and the merits on them
# ---------------------------------------------------------------------------------------------

def hkl_ref_path(lattice_system, bravais_lattice):
    """The reference list the optimiser itself loads, addressed the way the package does.

    Kept as a plain join under the package root rather than through importlib.resources so a
    development tree and an installed wheel resolve identically, and so the path is printable in
    an artefact.
    """
    return os.path.join(BASE, 'mlindex', 'models', f'{lattice_system}_1', 'data',
                        f'hkl_ref_{bravais_lattice}.npy')


_HKL_REF_CACHE = {}
_SPACEGROUP_CACHE = {}


def load_hkl_ref(lattice_system, bravais_lattice):
    key = (lattice_system, bravais_lattice)
    if key not in _HKL_REF_CACHE:
        _HKL_REF_CACHE[key] = np.load(hkl_ref_path(lattice_system, bravais_lattice))
    return _HKL_REF_CACHE[key]


def load_spacegroup_sets(lattice_system, bravais_lattice):
    """`assign_extinction_group`'s own narrowed lists, keyed by spacegroup label."""
    key = (lattice_system, bravais_lattice)
    if key not in _SPACEGROUP_CACHE:
        _SPACEGROUP_CACHE[key] = get_spacegroup_hkl_ref(
            load_hkl_ref(lattice_system, bravais_lattice), bravais_lattice=bravais_lattice)
    return _SPACEGROUP_CACHE[key]


def merits_on_reference(q2_obs, q2_ref_calc):
    """The reduced core, computed on one q2_ref_calc block. Returns a dict of (n_candidates,).

    ORDER IS LOAD-BEARING. `get_M20` is the only one of the four that modifies its input: it
    `np.putmask`s q2_ref_calc to zero outside the cut-off. So it is called LAST, on the array the
    other three have already read. Campaign 1 states this trap in three places and it is the
    reason the reversed and symmetric merits go first.

    The hkl assignment is redone here rather than read from a stored column because the merits
    must see the optimiser's own arrays: M20's cut-off is itself one of the reference lines, and
    a route that rebuilds q2_calc from Miller indices differs by an ULP that moves a line across
    its own boundary (F-095).
    """
    hkl_assign = fast_assign(q2_obs, q2_ref_calc)
    q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)

    M_tilde, M_rev, M_sym = get_M_rev_sym(q2_obs, q2_calc, q2_ref_calc)
    n_over, max_gap = get_n_over(q2_obs, q2_calc, q2_ref_calc)
    X_N = get_X_N(q2_obs, q2_calc, q2_ref_calc)
    M20 = get_M20(q2_obs, q2_calc, q2_ref_calc)   # no longer destroys q2_ref_calc

    return {'M20': M20, 'M_tilde': M_tilde, 'M_rev': M_rev, 'M_sym': M_sym,
            'X_N': X_N.astype(np.float64), 'n_over': n_over.astype(np.float64),
            'max_gap': max_gap.astype(np.float64)}


def merits_for_block(q2_obs, xnn, lattice_system, hkl):
    """Merits for one candidate block against one reference list, chunked to bound memory."""
    n_candidates = xnn.shape[0]
    n_ref = hkl.shape[0]
    chunk = max(1, min(n_candidates, MAX_BLOCK_ELEMENTS // max(n_ref, 1)))
    calculator = Q2Calculator(lattice_system=lattice_system, hkl=hkl, tensorflow=False,
                              representation='xnn')
    pieces = []
    for start in range(0, n_candidates, chunk):
        q2_ref_calc = calculator.get_q2(xnn[start:start + chunk])
        pieces.append(merits_on_reference(q2_obs, q2_ref_calc))
    return {name: np.concatenate([piece[name] for piece in pieces]) for name in MERITS}


# ---------------------------------------------------------------------------------------------
# stage: merits -- recompute at points B and C, and persist
# ---------------------------------------------------------------------------------------------

# Columns the recompute needs from the dump. Read explicitly so a schema change fails here
# rather than silently producing a column of NaN downstream.
PREDOWNSAMPLE_COLUMNS = ('entry_id', 'bravais_lattice', 'lattice_system', 'candidate_id', 'xnn',
                         'spacegroup', 'n_peaks', 'M20', 'Minfo', 'n_indexed', 'm20_at_prune',
                         'downsample_radius')

IDENTITY_COLUMNS = ('entry_id', 'bravais_lattice', 'candidate_id')


def bundle_directories(root):
    """{bundle name: directory}, for every bundle that actually carries a predownsample dump."""
    directories = {}
    for child in sorted(Path(root).iterdir()):
        if child.is_dir() and any(child.glob('predownsample_*.parquet')):
            directories[child.name] = child
    if not directories:
        raise SystemExit(f'no predownsample shards under {root}')
    return directories


def load_entries(bundle_dir):
    """The entry table for one bundle: q2_obs, split, and the truth columns."""
    shards = sorted(Path(bundle_dir).glob('entries_*.parquet'))
    return pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)


def label_path(labels_dir, bundle, shard_path):
    """`labels_{bundle}_{shard tag}.parquet`, the naming the labelling pass used."""
    return Path(labels_dir) / f'labels_{bundle}_{Path(shard_path).stem.split("_", 1)[1]}.parquet'


def join_labels(frame, labels):
    """Attach `is_correct`, asserting the row alignment rather than trusting it.

    The label shards were written by iterating the predownsample shards in order and appending a
    column, so they are row-aligned. Checking is cheap and a silent misalignment here would put
    correctness labels on the wrong candidates -- exactly the class of defect that corrupted the
    deduplication spacegroups (14b13a9).
    """
    if labels.shape[0] != frame.shape[0]:
        raise ValueError(f'{labels.shape[0]} label rows against {frame.shape[0]} candidate rows')
    for column in IDENTITY_COLUMNS:
        if not np.array_equal(labels[column].to_numpy(), frame[column].to_numpy()):
            raise ValueError(f'label shard is not row-aligned with the dump on {column}')
    if not np.array_equal(labels['M20'].to_numpy(), frame['M20'].to_numpy()):
        raise ValueError('label shard M20 disagrees with the dump M20')
    return labels['is_correct'].to_numpy()


def recompute_shard(shard_path, bundle, entries, labels_dir):
    """Points B and C for every candidate in one shard, plus the identity and label columns.

    Two layouts. Campaign 1's archived dumps keep `is_correct` in a parallel labels directory,
    row-aligned; the Phase 2 re-run writes it inline along with `merit_at_prune_*`, the point-A
    columns that make a criterion sweep at the cut possible at all. Anything matching
    `m20_at_prune` or `merit_at_prune_*` is carried through untouched.
    """
    stored = pq.ParquetFile(shard_path).schema_arrow.names
    wanted = [column for column in PREDOWNSAMPLE_COLUMNS if column in stored]
    carried = [column for column in stored
               if column == 'm20_at_prune' or column.startswith('merit_at_prune_')]
    inline = [column for column in ('is_correct', 'split', 'condition_bundle')
              if column in stored]
    frame = pd.read_parquet(shard_path,
                            columns=sorted(set(wanted + carried + inline), key=stored.index))
    if 'is_correct' not in frame:
        frame['is_correct'] = join_labels(frame, pd.read_parquet(
            label_path(labels_dir, bundle, shard_path)))
    truth = entries.set_index('entry_id')

    columns = {f'{merit}_{point}': np.full(frame.shape[0], np.nan)
               for point in ('B', 'C') for merit in MERITS}
    position = np.arange(frame.shape[0])

    for (bravais_lattice, lattice_system), lattice_rows in frame.groupby(
            ['bravais_lattice', 'lattice_system'], sort=False):
        hkl_ref = load_hkl_ref(lattice_system, bravais_lattice)
        spacegroup_sets = load_spacegroup_sets(lattice_system, bravais_lattice)
        for entry_id, group in lattice_rows.groupby('entry_id', sort=False):
            n_peaks = int(group['n_peaks'].iloc[0])
            q2_obs = np.asarray(truth.loc[entry_id, 'q2_obs'], dtype=np.float64)[:n_peaks]
            xnn = np.stack(group['xnn'].to_numpy()).astype(np.float64)
            rows = position[frame.index.get_indexer(group.index)]

            # Point B: the final cell scored against the full, un-narrowed list -- the quantity
            # the prune rule would test if the cut were simply moved after refine_cell.
            for name, values in merits_for_block(q2_obs, xnn, lattice_system, hkl_ref).items():
                columns[f'{name}_B'][rows] = values

            # Point C: the same cell against the extinction group the pipeline chose for it.
            # Grouped by spacegroup because each carries its own narrowed reference list.
            spacegroups = group['spacegroup'].to_numpy()
            for spacegroup in pd.unique(spacegroups):
                local = np.flatnonzero(spacegroups == spacegroup)
                block = merits_for_block(q2_obs, xnn[local], lattice_system,
                                         spacegroup_sets[spacegroup])
                for name, values in block.items():
                    columns[f'{name}_C'][rows[local]] = values

    keep = ['entry_id', 'bravais_lattice', 'lattice_system', 'candidate_id', 'n_peaks',
            'M20', 'Minfo', 'n_indexed', 'downsample_radius', 'is_correct'] + carried
    out = frame[keep].copy()
    if 'split' not in out:
        out['split'] = out['entry_id'].map(truth['split'])
    if 'condition_bundle' not in out:
        out['condition_bundle'] = bundle
    for name, values in columns.items():
        out[name] = values
    return out


def _merits_worker(job):
    shard_path, bundle_dir, bundle, labels_dir, out_dir = job
    if bundle is None:
        # Re-run layout: the shard carries its own bundle, and the entry table lives with the
        # archived arm the re-run was driven from.
        bundle = pd.read_parquet(shard_path, columns=['condition_bundle'])[
            'condition_bundle'].iloc[0]
        bundle_dir = os.path.join(bundle_dir, bundle)
    entries = load_entries(bundle_dir)
    frame = recompute_shard(shard_path, bundle, entries, labels_dir)
    destination = Path(out_dir) / f'merits_{Path(shard_path).stem.split("_", 1)[1]}.parquet'
    frame.to_parquet(destination, index=False)
    return {'shard': str(destination), 'rows': int(frame.shape[0]),
            'differing_M20': int(exact_mismatches(frame['M20'].to_numpy(),
                                                  frame['M20_C'].to_numpy())),
            'correct_rows': int(frame['is_correct'].sum())}


def exact_mismatches(stored, recomputed):
    """How many values differ, treating NaN as equal to NaN.

    A count, not a tolerance. The point of the gate is that the recompute walks the same route
    the pipeline walked -- same q2_ref_calc, same fast_assign, same get_M20 -- so anything but
    zero means the route diverged, and a tolerance would hide exactly the ULP-scale divergence
    the route was chosen to avoid.
    """
    both_nan = np.isnan(stored) & np.isnan(recomputed)
    return int(np.sum(~both_nan & (stored != recomputed)))


def run_merits(args):
    arm = ARMS[args.arm]
    out_dir = Path(BASE) / MERIT_ROOT / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if arm.get('layout') == 'rerun':
        # Flat directory, labels inline, entry tables borrowed from the archived arm.
        entries_root = os.path.join(BASE, arm['entries_root'])
        for shard in sorted((Path(BASE) / arm['root']).glob('predownsample_*.parquet')):
            jobs.append((str(shard), entries_root, None, None, str(out_dir)))
    else:
        for bundle, bundle_dir in bundle_directories(os.path.join(BASE, arm['root'])).items():
            for shard in sorted(bundle_dir.glob('predownsample_*.parquet')):
                jobs.append((str(shard), str(bundle_dir), bundle,
                             os.path.join(BASE, arm['labels']), str(out_dir)))
    if args.limit_shards:
        jobs = jobs[:args.limit_shards]
        print(f'SMOKE TEST: {len(jobs)} shards only -- not a result')

    started = time.time()
    results = []
    if args.processes == 1:
        for done, job in enumerate(jobs, start=1):
            results.append(_merits_worker(job))
            print(f'  {done}/{len(jobs)} shards, {time.time() - started:.0f}s', flush=True)
    else:
        from multiprocessing import Pool
        with Pool(processes=args.processes) as pool:
            for done, result in enumerate(pool.imap_unordered(_merits_worker, jobs), start=1):
                results.append(result)
                print(f'  {done}/{len(jobs)} shards, {time.time() - started:.0f}s', flush=True)

    rows = sum(result['rows'] for result in results)
    differing = sum(result['differing_M20'] for result in results)
    correct = sum(result['correct_rows'] for result in results)
    manifest = {
        'arm': args.arm,
        'note': arm['note'],
        'source': arm['root'],
        'labels': arm['labels'],
        'rows': rows,
        'correct_rows': correct,
        'shards': len(results),
        'merits': list(MERITS),
        'points': {'A': 'm20_at_prune, stored -- pre-refinement, full reference list',
                   'B': 'recomputed from the stored xnn against the full reference list',
                   'C': 'recomputed from the stored xnn against the spacegroup-narrowed list'},
        'gate_differing_M20_values': differing,
        'seconds': round(time.time() - started, 1),
        'commit': commit_hash(),
        }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)

    print(f'\n{arm["note"]}')
    print(f'{rows:,} rows, {correct:,} correct, {len(results)} shards, '
          f'{time.time() - started:.0f}s')
    print(f'GATE  recomputed point-C M20 vs the dump: {differing} differing values of {rows:,}')
    if differing:
        raise SystemExit('point-C M20 does not reproduce the pipeline value; route diverged')


def commit_hash():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                       text=True).strip()
    except Exception:
        return 'unknown'



# ---------------------------------------------------------------------------------------------
# stage: cost -- what does admitting one more candidate actually cost?
# ---------------------------------------------------------------------------------------------

# The pool sizes the block is timed at. The quantity wanted is the SLOPE -- seconds per admitted
# candidate -- so several sizes are timed and a line is fitted, rather than dividing one timing by
# its own n and inheriting the fixed setup cost as if it scaled.
COST_POOL_SIZES = (250, 500, 1000, 2000, 4000)

BROADENING_TAG = '1'


# The steps `_run_loop` runs after the cut, in order (MPIOptimizer.py:141-146). Timed
# separately, not as a block, because WHERE the cut is placed decides which of them run on the
# whole pool and which run only on survivors -- and they are not close to equal. A cut moved to
# after `refine_cell` still saves the extinction-group work; a cut moved to after
# `assign_extinction_group` saves almost nothing, because that step is the cost.
POST_PRUNE_STEPS = ('refine_cell', 'standardize_cell', 'correct_off_by_two',
                    'assign_extinction_group', 'calculate_peaks_indexed')

# Which steps are still downstream of a cut placed at each point. Point A is the production
# placement -- everything is downstream. Point C is after the extinction assignment, so only the
# indexed-peak count and deduplication remain.
DOWNSTREAM_OF = {
    'A': POST_PRUNE_STEPS,
    'B': ('assign_extinction_group', 'calculate_peaks_indexed'),
    'C': ('calculate_peaks_indexed',),
    }


def post_prune_block(candidates):
    """Run the steps in order, returning seconds per step.

    They must run in sequence -- each consumes what the last produced -- so this times them
    within one pass rather than timing any of them in isolation.
    """
    timings = {}
    for step in POST_PRUNE_STEPS:
        started = time.perf_counter()
        getattr(candidates, step)()
        timings[step] = time.perf_counter() - started
    return timings


def _candidate_pool_for_cost(arm, bravais_lattice, n_wanted):
    """Real post-search cells and their peak list, taken from the threshold-0 dump.

    Timing on generated-but-unrefined cells would measure the same arithmetic on a different
    distribution: `refine_cell` groups candidates by how many peaks they index, so the number of
    sub-solves it runs is data-dependent. These are the cells the block actually sees.
    """
    root = Path(BASE) / ARMS[arm]['root']
    bundle_dir = sorted(bundle_directories(root).values())[0]
    shard = sorted(bundle_dir.glob('predownsample_*.parquet'))[0]
    frame = pd.read_parquet(shard, columns=['entry_id', 'bravais_lattice', 'lattice_system',
                                            'xnn', 'n_peaks'])
    frame = frame[frame['bravais_lattice'] == bravais_lattice]
    entry_id = frame['entry_id'].value_counts().idxmax()
    frame = frame[frame['entry_id'] == entry_id]
    entries = load_entries(bundle_dir).set_index('entry_id')
    n_peaks = int(frame['n_peaks'].iloc[0])
    q2_obs = np.asarray(entries.loc[entry_id, 'q2_obs'], dtype=np.float64)[:n_peaks]
    xnn = np.stack(frame['xnn'].to_numpy()).astype(np.float64)
    if xnn.shape[0] < n_wanted:      # tile rather than shrink, so every lattice reports the
        repeats = int(np.ceil(n_wanted / xnn.shape[0]))   # same sizes and the slopes compare
        xnn = np.tile(xnn, (repeats, 1))
    return q2_obs, xnn[:n_wanted], entry_id


def run_cost(args):
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, shutdown_mp_workers

    optimizers, processes, task_queues = setup_mp_optimizers(
        1, BROADENING_TAG, n_candidates_scale=1, seed=12345)
    shutdown_mp_workers(processes, task_queues)

    lattices = args.bravais_lattices.split(',') if args.bravais_lattices else list(optimizers)
    rows = []
    for bravais_lattice in lattices:
        optimizer = optimizers[bravais_lattice]
        optimizer.zero_error = False
        optimizer.wavelength = None
        n_groups = len(load_spacegroup_sets(optimizer.lattice_system, bravais_lattice))
        n_ref = load_hkl_ref(optimizer.lattice_system, bravais_lattice).shape[0]
        q2_obs, pool, entry_id = _candidate_pool_for_cost(args.arm, bravais_lattice,
                                                         max(COST_POOL_SIZES))
        optimizer.q2_obs = q2_obs[:optimizer.n_peaks]

        for size in COST_POOL_SIZES:
            best = None
            for _ in range(args.repeats):
                # A fresh Candidates per repeat: the block mutates it, and correct_off_by_two
                # appends rows, so a second call would time a different pool.
                candidates = optimizer.generate_candidates_common(pool[:size].copy())
                timings = post_prune_block(candidates)
                if best is None or sum(timings.values()) < sum(best.values()):
                    best = timings
            rows.append(dict({'bravais_lattice': bravais_lattice,
                              'lattice_system': optimizer.lattice_system,
                              'n_extinction_groups': n_groups, 'n_reference_lines': n_ref,
                              'n_peaks': int(optimizer.n_peaks), 'entry_id': entry_id,
                              'pool_size': size, 'seconds_total': sum(best.values())},
                             **{f'seconds_{step}': best[step] for step in POST_PRUNE_STEPS}))
            print(f'  {bravais_lattice} n={size:5d}  {sum(best.values()):7.3f}s  '
                  f'(extinction {best["assign_extinction_group"]:6.3f}s over {n_groups} groups, '
                  f'{n_ref} lines)', flush=True)

    frame = pd.DataFrame(rows)
    fits = []
    for bravais_lattice, group in frame.groupby('bravais_lattice', sort=False):
        # Least squares on (pool size, seconds), per step. The slope is the per-candidate cost a
        # cut trades against correct cells; the intercept is fixed setup no cut can recover.
        record = {'bravais_lattice': bravais_lattice,
                  'lattice_system': group['lattice_system'].iloc[0],
                  'n_extinction_groups': int(group['n_extinction_groups'].iloc[0]),
                  'n_reference_lines': int(group['n_reference_lines'].iloc[0])}
        for step in POST_PRUNE_STEPS + ('total',):
            column = 'seconds_total' if step == 'total' else f'seconds_{step}'
            slope, intercept = np.polyfit(group['pool_size'], group[column], 1)
            record[f'per_candidate_{step}'] = float(slope)
            record[f'fixed_{step}'] = float(intercept)
        for point, steps in DOWNSTREAM_OF.items():
            record[f'per_candidate_downstream_{point}'] = float(
                sum(record[f'per_candidate_{step}'] for step in steps))
        fits.append(record)
    fits = pd.DataFrame(fits).sort_values('per_candidate_total')
    # `seconds_per_candidate` is the production placement: a cut at point A has every step
    # downstream of it, so this is what admitting one more candidate costs today.
    fits['seconds_per_candidate'] = fits['per_candidate_downstream_A']
    fits['relative_cost'] = fits['seconds_per_candidate'] / fits['seconds_per_candidate'].min()

    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(artifact_dir / 'S03_cost_timings.csv', index=False)
    fits.to_csv(artifact_dir / 'S03_cost_per_candidate.csv', index=False)
    print()
    print(fits[['bravais_lattice', 'n_extinction_groups', 'n_reference_lines',
                'seconds_per_candidate', 'relative_cost',
                'per_candidate_downstream_B', 'per_candidate_downstream_C']].to_string(index=False))
    calibrate_cost_model(fits, artifact_dir)
    print(f'\nwrote {artifact_dir}/S03_cost_{{timings,per_candidate}}.csv')


# The three real runs of the general arm, identical but for the threshold. `t35` and `t5` share a
# commit; `t0` was built at a different one, so its wall clock is indicative rather than exact --
# and it is quoted with that caveat every time.
REAL_THRESHOLD_RUNS = {'t5': 1305.5, 't35': 1371.8, 't0': 2053.8}


def calibrate_cost_model(fits, artifact_dir):
    """Turn per-candidate seconds into predicted wall clock, and check the result against reality.

    The per-candidate costs are measured serially; the runs they are checked against are parallel,
    so one constant separates them. Rather than assume it, it is FITTED from the three real runs
    and then over-determined: three threshold pairs give three independent estimates, and the
    fixed search cost each implies is a fourth check -- if the linear model were wrong, the
    implied search cost would move with the threshold, and it does not.
    """
    per_candidate = fits.set_index('bravais_lattice')['per_candidate_downstream_A']
    serial = {}
    for tag in REAL_THRESHOLD_RUNS:
        root = Path(BASE) / 'mlindex' / 'characterization' / 'fom' / 'allstrata' / tag
        total = 0.0
        rows = 0
        for shard in root.glob('*/predownsample_*.parquet'):
            counts = pd.read_parquet(shard, columns=['bravais_lattice'])[
                'bravais_lattice'].value_counts()
            for lattice, count in counts.items():
                total += count * per_candidate.get(lattice, 0.0)
            rows += int(counts.sum())
        serial[tag] = {'serial_post_cut_seconds': total, 'rows': rows,
                       'real_seconds': REAL_THRESHOLD_RUNS[tag]}

    tags = list(REAL_THRESHOLD_RUNS)
    estimates = []
    for first in range(len(tags)):
        for second in range(first + 1, len(tags)):
            a, b = serial[tags[first]], serial[tags[second]]
            if a['real_seconds'] == b['real_seconds']:
                continue
            estimates.append((a['serial_post_cut_seconds'] - b['serial_post_cut_seconds'])
                             / (a['real_seconds'] - b['real_seconds']))
    parallelism = float(np.mean(estimates))

    for tag, record in serial.items():
        record['predicted_post_cut_wall_seconds'] = record['serial_post_cut_seconds'] / parallelism
        record['implied_fixed_search_seconds'] = (record['real_seconds']
                                                  - record['predicted_post_cut_wall_seconds'])
    implied = [record['implied_fixed_search_seconds'] for record in serial.values()]

    model = {'effective_parallelism': parallelism,
             'effective_parallelism_estimates': [float(value) for value in estimates],
             'fixed_search_seconds': float(np.mean(implied)),
             'fixed_search_seconds_spread': float(max(implied) - min(implied)),
             'runs': serial,
             'note': ('per-candidate costs are measured serially; the real runs are parallel, so '
                      'one fitted constant separates them. t0 was built at a different commit '
                      'from t35/t5 and its wall clock is indicative.'),
             'machine': 'Apple M1 Pro, 10 cores (8 performance)'}
    with open(artifact_dir / 'S03_cost_model.json', 'w', encoding='utf-8') as handle:
        json.dump(model, handle, indent=2)

    print(f'\neffective parallelism of the post-cut block: {parallelism:.2f} '
          f'(estimates {", ".join(f"{value:.2f}" for value in estimates)})')
    print(f'fixed search cost implied by each run: '
          f'{", ".join(f"{value:.1f}s" for value in implied)} '
          f'-- spread {max(implied) - min(implied):.1f}s of {np.mean(implied):.0f}s')
    for tag, record in serial.items():
        print(f'  {tag}: {record["rows"]:>10,} rows, post-cut '
              f'{record["predicted_post_cut_wall_seconds"]:7.1f}s of '
              f'{record["real_seconds"]:.1f}s '
              f'({record["predicted_post_cut_wall_seconds"] / record["real_seconds"]:.1%})')


# ---------------------------------------------------------------------------------------------
# the criteria a cut could be made on
# ---------------------------------------------------------------------------------------------

# Point A carries M20 alone: `best_xnn` at the cut was never stored, so no other merit has a
# stage-A value and none can be invented from the dump. That absence is the finding, not an
# omission -- run_fom_prune_rerun.py is what fills it.
#
# de Wolff reports X_N beside M20 and never folds it in: "M20 > 10 guarantees correctness provided
# there are few spurious lines (X20 not above 2)". That is a *composite cut*, not a composite
# merit, and it is the one blend worth testing here because it costs nothing extra -- get_X_N is
# 0.91x get_M20 (C2-F-016) and both are already computed.
COMPOSITE_VETO = 2


def criteria_list(available=None):
    """[(label, point, base column, veto column or None)] -- every cut this stage sweeps.

    Point A is data-driven. On campaign 1's dumps it is `m20_at_prune` and nothing else, because
    that is all that was stored. On a Phase 2 re-run every merit has a `merit_at_prune_*` column
    and point A grows to match -- which is the only configuration in which "would a different
    merit make a better cut *at the cut*?" is a question this script can answer (C2-Q-008).
    """
    criteria = [('m20_at_prune', 'A', 'm20_at_prune', None)]
    if available is not None:
        for merit in MERITS:
            column = f'merit_at_prune_{merit}'
            if column in available and merit != 'M20':
                criteria.append((f'{merit}_A', 'A', column, None))
        if all(f'merit_at_prune_{merit}' in available for merit in ('M_sym', 'X_N')):
            criteria.append(('M_sym_A+X_N_veto', 'A', 'merit_at_prune_M_sym',
                             'merit_at_prune_X_N'))
    for point in ('B', 'C'):
        for merit in MERITS:
            criteria.append((f'{merit}_{point}', point, f'{merit}_{point}', None))
        criteria.append((f'M_sym_{point}+X_N_veto', point, f'M_sym_{point}', f'X_N_{point}'))
    return criteria


def required_columns(criteria):
    columns = []
    for _, _, base, veto in criteria:
        for column in (base, veto):
            if column is not None and column not in columns:
                columns.append(column)
    return columns


def criterion_scores(frame, base, veto):
    """Signed so that a cut always keeps `score >= threshold`.

    Three of the seven merits count things that should not be there -- unexplained observed lines,
    over-predicted calculated lines, the longest run of them -- so a low value is the good one and
    the sign has to be flipped before any threshold is comparable across criteria.

    A veto column expresses the composite cut as a score rather than as a second rule, by sending
    the vetoed candidates to -inf. They then fall out at every threshold, which is what a veto is.
    """
    if base.startswith('merit_at_prune_'):
        merit = base[len('merit_at_prune_'):]
    elif base.endswith(('_B', '_C')):
        merit = base[:-2]
    else:
        merit = 'M20'
    values = frame[base].to_numpy(dtype=np.float64)
    scores = -values if merit in HIGHER_IS_WORSE else values.copy()
    if veto is not None:
        scores = np.where(frame[veto].to_numpy(dtype=np.float64) <= COMPOSITE_VETO,
                          scores, -np.inf)
    return scores


# ---------------------------------------------------------------------------------------------
# stage: retention -- correct cells kept, against pool size and against wall clock
# ---------------------------------------------------------------------------------------------

# Target surviving fractions, per lattice. The x-axis reported is the ACHIEVED fraction, so
# approximate quantiles are all these need to be: they only have to place the sample points.
# Production at 5.0 keeps ~0.4 % of threshold-0's rows on the hard arm and ~2.8 % on the general
# arm, so the grid is dense where the decision lives.
TARGET_FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.028, 0.02, 0.01, 0.004, 0.002, 0.001)

# How many values per (lattice, criterion) to sample when placing the thresholds. Quantiles from
# 200 000 draws are accurate to well under the spacing of the grid above, and the counts that go
# into every reported number are exact rather than sampled.
QUANTILE_SAMPLE = 200_000


def merit_shards(merit_root, arm):
    shards = sorted((Path(BASE) / merit_root / arm).glob('merits_*.parquet'))
    if not shards:
        raise SystemExit(f'no merit shards under {merit_root}/{arm}; run --stage merits first')
    return shards


def _quantile_worker(job):
    """Per shard: a bounded random sample of each (lattice, criterion) score, and the row count."""
    shard, criteria, seed = job
    frame = pd.read_parquet(shard, columns=['bravais_lattice'] + required_columns(criteria))
    rng = np.random.default_rng(seed)
    samples = {}
    counts = {}
    for bravais_lattice, group in frame.groupby('bravais_lattice', sort=False):
        counts[bravais_lattice] = int(group.shape[0])
        take = min(group.shape[0], QUANTILE_SAMPLE)
        rows = (rng.choice(group.shape[0], size=take, replace=False)
                if take < group.shape[0] else np.arange(group.shape[0]))
        for label, _, base, veto in criteria:
            samples[(bravais_lattice, label)] = criterion_scores(group, base, veto)[rows]
    return samples, counts


def build_thresholds(shards, criteria, processes):
    """Per (lattice, criterion), the score at each target surviving fraction.

    A cut is a threshold on a score, and a threshold means something different on every merit, so
    comparing merits at matched threshold compares nothing. Matched *surviving pool size* is the
    comparison that has meaning -- it is the same amount of downstream work bought on each merit,
    which is exactly the axis DWMM's question is asked on.
    """
    jobs = [(str(shard), criteria, 12345 + index) for index, shard in enumerate(shards)]
    pooled = {}
    counts = {}
    for samples, shard_counts in _map(_quantile_worker, jobs, processes):
        for key, values in samples.items():
            pooled.setdefault(key, []).append(values)
        for lattice, count in shard_counts.items():
            counts[lattice] = counts.get(lattice, 0) + count

    thresholds = {}
    for key, pieces in pooled.items():
        values = np.concatenate(pieces)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        # `1 - fraction` because the grid is expressed as "keep this share", and a cut keeps the
        # upper tail of the score.
        thresholds[key] = np.quantile(values, [1.0 - f for f in TARGET_FRACTIONS],
                                      method='linear')
    return thresholds, counts


def _map(function, jobs, processes):
    if processes == 1:
        for job in jobs:
            yield function(job)
    else:
        from multiprocessing import Pool
        with Pool(processes=processes) as pool:
            for result in pool.imap_unordered(function, jobs):
                yield result


def _count_worker(job):
    """Per shard: exact survivor counts at every threshold, and every correct candidate's score.

    Counts are exact rather than sampled -- the sampling in build_thresholds only places the grid.
    The correct candidates are carried whole because there are 3 746 of them in the hard arm and
    30 807 in the general one, so reachability can be answered exactly from a table that fits in
    memory, instead of being recomputed from 57 million rows for every threshold.
    """
    shard, criteria, thresholds, lattice_order = job
    columns = ['entry_id', 'bravais_lattice', 'condition_bundle', 'split', 'is_correct']
    frame = pd.read_parquet(shard, columns=columns + required_columns(criteria))
    bundle = frame['condition_bundle'].iloc[0]

    lattice_index = frame['bravais_lattice'].map(
        {name: position for position, name in enumerate(lattice_order)}).to_numpy()
    entry_codes, entry_names = pd.factorize(frame['entry_id'], sort=True)
    n_fractions = len(TARGET_FRACTIONS)
    correct = frame['is_correct'].to_numpy()

    by_lattice = {}
    correct_scores = {}
    group_max = {}
    for label, _, base, veto in criteria:
        scores = criterion_scores(frame, base, veto)
        # A NaN score can never clear a cut, and NaN also breaks the ordering searchsorted needs.
        scores = np.where(np.isnan(scores), -np.inf, scores)

        # Per-row threshold: the cut is placed per lattice, so each row is compared against its
        # own lattice's grid.
        grid = np.full((len(lattice_order), n_fractions), np.inf)
        for position, lattice in enumerate(lattice_order):
            if (lattice, label) in thresholds:
                grid[position] = thresholds[(lattice, label)]
        survives = scores[:, None] >= grid[lattice_index]

        for position, lattice in enumerate(lattice_order):
            rows = lattice_index == position
            if rows.any():
                by_lattice[(bundle, lattice, label)] = (int(rows.sum()),
                                                        survives[rows].sum(axis=0))
        correct_scores[label] = scores[correct]

        # The arg-max rescue: prune_below_m20 keeps the best candidate when nothing clears the
        # bar (Candidates.py:450-451). Emulated at (entry, lattice) granularity, which is coarser
        # than production's per-rank pool, so it is a LOWER bound on how often the rescue fires.
        keys = entry_codes * len(lattice_order) + lattice_index
        order = np.lexsort((-scores, keys))
        first = np.concatenate([[True], keys[order][1:] != keys[order][:-1]])
        argmax_rows = order[first]
        group_max[label] = pd.DataFrame({
            'entry_id': entry_names.to_numpy()[entry_codes[argmax_rows]],
            'bravais_lattice': np.asarray(lattice_order)[lattice_index[argmax_rows]],
            'condition_bundle': bundle, 'criterion': label,
            'max_score': scores[argmax_rows], 'argmax_is_correct': correct[argmax_rows]})

    correct_frame = frame.loc[correct, ['entry_id', 'bravais_lattice', 'condition_bundle',
                                        'split']].copy()
    for label, values in correct_scores.items():
        correct_frame[label] = values
    return by_lattice, correct_frame, pd.concat(group_max.values(), ignore_index=True)


BRAVAIS_LATTICES = ('cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oC', 'oF', 'oI', 'oP', 'mC', 'mP',
                    'aP')

BOOTSTRAP_DRAWS = 1000

# Fitted in calibrate_cost_model from three real runs that differ only in the threshold; three
# independent pairs agree to 0.4 % and the fixed search cost each implies agrees to 0.7 s in
# 1 286 s. Overridden from S03_cost_model.json when that file is present.
EFFECTIVE_PARALLELISM = 8.80


def _threshold_matrix(thresholds, lattices, label):
    """(n_rows, n_fractions) of the threshold each row is compared against."""
    grid = np.full((len(BRAVAIS_LATTICES), len(TARGET_FRACTIONS)), np.inf)
    for position, lattice in enumerate(BRAVAIS_LATTICES):
        if (lattice, label) in thresholds:
            grid[position] = thresholds[(lattice, label)]
    index = pd.Series(lattices).map(
        {name: position for position, name in enumerate(BRAVAIS_LATTICES)}).to_numpy()
    return grid[index]


def _bootstrap_reachability(reachable, rng):
    """Bootstrap over ENTRIES, not candidates: one crystal is one draw (PROTOCOL section 8).

    `reachable` is (n_entries, n_fractions) boolean. Resampling rows resamples crystals, which is
    the unit the claim is about -- a crystal seen under several conditions is not several draws.
    """
    n_entries = reachable.shape[0]
    draws = rng.integers(0, n_entries, size=(BOOTSTRAP_DRAWS, n_entries))
    means = reachable[draws].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def summarise_retention(arm, criteria, thresholds, by_lattice, correct_frame, group_max,
                        cost_table):
    """One row per (bundle, criterion, lattice-or-ALL, target fraction)."""
    rows = []
    seconds = _cost_lookup(cost_table)
    rng = np.random.default_rng(12345)

    for label, point, _, _ in criteria:
        thresholds_correct = _threshold_matrix(thresholds, correct_frame['bravais_lattice'], label)
        correct_survives = correct_frame[label].to_numpy()[:, None] >= thresholds_correct

        for bundle in sorted(correct_frame['condition_bundle'].unique()):
            in_bundle = (correct_frame['condition_bundle'] == bundle).to_numpy()
            bundle_correct = correct_frame.loc[in_bundle]
            survives = correct_survives[in_bundle]

            # Reachability: an entry is reachable when at least one of its correct candidates
            # survives the cut. "No correct candidate in the pool" is a generation failure, not a
            # ranking failure, and it is kept in its own bucket (PROTOCOL section 8).
            entry_codes, entry_names = pd.factorize(bundle_correct['entry_id'], sort=True)
            reachable = np.zeros((entry_names.size, len(TARGET_FRACTIONS)), dtype=bool)
            np.logical_or.at(reachable, entry_codes, survives)

            rescued = _rescued(group_max, bundle, label, thresholds, entry_names)
            reachable_rescued = reachable | rescued
            low, high = _bootstrap_reachability(reachable, rng)

            for k, fraction in enumerate(TARGET_FRACTIONS):
                total = 0
                surviving = 0
                estimated_seconds = 0.0
                per_lattice = []
                for lattice in BRAVAIS_LATTICES:
                    key = (bundle, lattice, label)
                    if key not in by_lattice:
                        continue
                    lattice_total, lattice_surviving = by_lattice[key]
                    total += lattice_total
                    surviving += int(lattice_surviving[k])
                    estimated_seconds += _estimated_seconds(seconds, lattice, point,
                                                            lattice_total,
                                                            int(lattice_surviving[k]))
                    per_lattice.append((lattice, lattice_total, int(lattice_surviving[k])))

                base = {'arm': arm, 'condition_bundle': bundle, 'criterion': label,
                        'point': point, 'target_fraction': fraction}
                rows.append(dict(base, bravais_lattice='ALL',
                                 n_candidates=total, n_surviving=surviving,
                                 achieved_fraction=surviving / total if total else np.nan,
                                 n_correct=int(survives.shape[0]),
                                 n_correct_retained=int(survives[:, k].sum()),
                                 correct_retention=(float(survives[:, k].mean())
                                                    if survives.shape[0] else np.nan),
                                 n_entries=int(entry_names.size),
                                 n_reachable=int(reachable[:, k].sum()),
                                 reachability=float(reachable[:, k].mean()),
                                 reachability_ci_low=float(low[k]),
                                 reachability_ci_high=float(high[k]),
                                 n_reachable_with_rescue=int(reachable_rescued[:, k].sum()),
                                 estimated_seconds=estimated_seconds))
                for lattice, lattice_total, lattice_surviving in per_lattice:
                    here = (bundle_correct['bravais_lattice'] == lattice).to_numpy()
                    rows.append(dict(base, bravais_lattice=lattice,
                                     n_candidates=lattice_total, n_surviving=lattice_surviving,
                                     achieved_fraction=(lattice_surviving / lattice_total
                                                        if lattice_total else np.nan),
                                     n_correct=int(here.sum()),
                                     n_correct_retained=int(survives[here, k].sum()),
                                     correct_retention=(float(survives[here, k].mean())
                                                        if here.any() else np.nan),
                                     n_entries=np.nan, n_reachable=np.nan, reachability=np.nan,
                                     reachability_ci_low=np.nan, reachability_ci_high=np.nan,
                                     n_reachable_with_rescue=np.nan,
                                     estimated_seconds=_estimated_seconds(
                                         seconds, lattice, point, lattice_total,
                                         lattice_surviving)))
    return pd.DataFrame(rows)


def _cost_lookup(cost_table):
    """{lattice: {point: (upstream per candidate, downstream per candidate)}}."""
    if cost_table is None:
        return {}
    lookup = {}
    for _, row in cost_table.iterrows():
        total = row['per_candidate_downstream_A']
        lookup[row['bravais_lattice']] = {
            point: (total - row[f'per_candidate_downstream_{point}'],
                    row[f'per_candidate_downstream_{point}'])
            for point in DOWNSTREAM_OF}
    return lookup


def _estimated_seconds(lookup, lattice, point, n_candidates, n_surviving):
    """Wall clock the cut buys, with the steps upstream of it charged on the WHOLE pool.

    Moving a cut later does not make its upstream steps free -- it makes them unconditional. A cut
    at point B pays `refine_cell`, `standardize_cell` and `correct_off_by_two` on every candidate
    the search produced, and only the extinction assignment on the survivors. That is the trade
    the placement decision is actually making, and charging only the survivors would hide it.

    Divided by the effective parallelism fitted in S03_cost_model.json, so the number is
    comparable with a real run's wall clock rather than with a serial ideal.
    """
    if lattice not in lookup:
        return 0.0
    upstream, downstream = lookup[lattice][point]
    return (n_candidates * upstream + n_surviving * downstream) / EFFECTIVE_PARALLELISM


def _rescued(group_max, bundle, label, thresholds, entry_names):
    """Entries the arg-max rescue makes reachable, per threshold. A lower bound -- see the
    note in _count_worker."""
    rescued = np.zeros((entry_names.size, len(TARGET_FRACTIONS)), dtype=bool)
    here = group_max[(group_max['condition_bundle'] == bundle)
                     & (group_max['criterion'] == label)
                     & group_max['argmax_is_correct']]
    if here.empty:
        return rescued
    grid = _threshold_matrix(thresholds, here['bravais_lattice'], label)
    fires = here['max_score'].to_numpy()[:, None] < grid       # nothing cleared the bar
    codes = pd.Index(entry_names).get_indexer(here['entry_id'])
    keep = codes >= 0
    np.logical_or.at(rescued, codes[keep], fires[keep])
    return rescued


def entry_totals(arm):
    """(bundle, entries) from the arm's own entry tables -- the absolute denominator.

    `correct_frame` only knows entries that have at least one correct candidate somewhere in the
    threshold-0 pool, so reachability computed from it is conditional on the search having found
    the answer at all. Both denominators are reported: conditioning on reachability is the right
    way to read a *cut*, but the absolute rate is what an end-to-end claim is made against.
    """
    totals = {}
    # A re-run reuses the archived arm's entry tables, so the denominators come from there.
    root = Path(BASE) / ARMS[arm].get('entries_root', ARMS[arm]['root'])
    for bundle, bundle_dir in bundle_directories(root).items():
        totals[bundle] = int(load_entries(bundle_dir)['entry_id'].nunique())
    return totals


def run_retention(args):
    shards = merit_shards(args.merit_root, args.arm)
    criteria = criteria_list(available=set(pq.ParquetFile(shards[0]).schema_arrow.names))

    print(f'placing thresholds over {len(shards)} shards, {len(criteria)} criteria...',
          flush=True)
    started = time.time()
    thresholds, _ = build_thresholds(shards, criteria, args.processes)
    print(f'  {time.time() - started:.0f}s', flush=True)

    jobs = [(str(shard), criteria, thresholds, BRAVAIS_LATTICES) for shard in shards]
    by_lattice = {}
    correct_frames = []
    group_maxes = []
    print('counting survivors...', flush=True)
    for done, (shard_counts, correct_frame, group_max) in enumerate(
            _map(_count_worker, jobs, args.processes), start=1):
        for key, (total, surviving) in shard_counts.items():
            if key in by_lattice:
                previous_total, previous = by_lattice[key]
                by_lattice[key] = (previous_total + total, previous + surviving)
            else:
                by_lattice[key] = (total, surviving.copy())
        correct_frames.append(correct_frame)
        group_maxes.append(group_max)
        print(f'  {done}/{len(jobs)} shards, {time.time() - started:.0f}s', flush=True)

    correct_frame = pd.concat(correct_frames, ignore_index=True)
    group_max = pd.concat(group_maxes, ignore_index=True)

    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cost_path = artifact_dir / 'S03_cost_per_candidate.csv'
    cost_table = pd.read_csv(cost_path) if cost_path.exists() else None
    model_path = artifact_dir / 'S03_cost_model.json'
    if model_path.exists():
        global EFFECTIVE_PARALLELISM
        with open(model_path, encoding='utf-8') as handle:
            EFFECTIVE_PARALLELISM = json.load(handle)['effective_parallelism']
    if cost_table is None:
        print('NOTE: no S03_cost_per_candidate.csv -- the wall-clock column will be zero. '
              'Run --stage cost first.')

    summary = summarise_retention(args.arm, criteria, thresholds, by_lattice, correct_frame,
                                  group_max, cost_table)
    totals = entry_totals(args.arm)
    summary['n_entries_in_bundle'] = summary['condition_bundle'].map(totals)
    summary['reachability_absolute'] = (summary['n_reachable']
                                        / summary['n_entries_in_bundle'])

    destination = artifact_dir / f'S03_prune_retention_{args.arm}.csv'
    summary.to_csv(destination, index=False)

    thresholds_frame = pd.DataFrame([
        {'bravais_lattice': lattice, 'criterion': label,
         **{f'threshold_at_{fraction:g}': value
            for fraction, value in zip(TARGET_FRACTIONS, values)}}
        for (lattice, label), values in sorted(thresholds.items())])
    thresholds_frame.to_csv(artifact_dir / f'S03_prune_thresholds_{args.arm}.csv', index=False)

    print(f'\nwrote {destination}')
    print(f'wrote {artifact_dir}/S03_prune_thresholds_{args.arm}.csv')
    _print_retention_headline(summary)


def _print_retention_headline(summary):
    """The one comparison the stage exists to make, at the pool size production actually uses."""
    aggregate = summary[summary['bravais_lattice'] == 'ALL']
    # The share of the threshold-0 pool the production cut of 5.0 actually keeps on this arm.
    target = 0.028 if aggregate['arm'].iloc[0].endswith('general') else 0.004
    at_production = aggregate[aggregate['target_fraction'] == target]
    if at_production.empty:
        return
    table = (at_production.groupby('criterion')
             .agg(achieved_fraction=('achieved_fraction', 'mean'),
                  correct_retention=('correct_retention', 'mean'),
                  reachability=('reachability', 'mean'),
                  estimated_seconds=('estimated_seconds', 'sum'))
             .sort_values('reachability', ascending=False))
    print(f'\nat a matched surviving pool of ~{target:.1%} of the threshold-0 pool, '
          'averaged over bundles:')
    print(table.to_string())


# ---------------------------------------------------------------------------------------------
# stage: stage -- is M20 the wrong merit at the cut, or is the cut at the wrong stage?
# ---------------------------------------------------------------------------------------------

# How many incorrect rows per (lattice, bundle) to keep for the quantile columns. The counts that
# carry the result are exact; only the medians are sampled, and 50 000 draws pin a median far
# tighter than the spread being described.
INCORRECT_SAMPLE = 50_000


def _stage_worker(job):
    """Per shard: exact per-(entry, lattice, label) counts, plus a sample for the medians.

    Counts are per ENTRY rather than pooled, because the interval on every rate here has to be
    bootstrapped over entries -- one crystal is one draw, and the ~9 000 candidates it
    contributes are one search of one pattern, not 9 000 independent pieces of evidence
    (PROTOCOL section 8). Pooling first would throw that structure away.

    The correct candidates the cut deletes are also carried whole: there are a few thousand of
    them in 57 million rows, they are the subject, and every later question about them is cheaper
    to ask from a table than from another pass over the dump.
    """
    shard, seed = job
    columns = ['entry_id', 'bravais_lattice', 'condition_bundle', 'split', 'is_correct',
               'm20_at_prune', 'M20_B', 'M20_C', 'M20']
    frame = pd.read_parquet(shard, columns=columns)
    bundle = frame['condition_bundle'].iloc[0]
    kept_correct = (frame['is_correct']
                    & (frame['m20_at_prune'] >= PRODUCTION_PRUNE_THRESHOLD))
    kept = (frame.loc[kept_correct].groupby(['condition_bundle', 'entry_id']).size()
            .rename('n_kept'))

    frame = frame.loc[frame['m20_at_prune'] < PRODUCTION_PRUNE_THRESHOLD]
    frame = frame.assign(
        ge5_B=frame['M20_B'].to_numpy() >= PRODUCTION_PRUNE_THRESHOLD,
        ge5_C=frame['M20_C'].to_numpy() >= PRODUCTION_PRUNE_THRESHOLD,
        unmoved_B=np.isclose(frame['M20_B'].to_numpy(), frame['m20_at_prune'].to_numpy(),
                             rtol=0, atol=1e-9),
        label=np.where(frame['is_correct'].to_numpy(), 'correct', 'incorrect'))

    per_entry = (frame.groupby(['condition_bundle', 'bravais_lattice', 'entry_id', 'label'],
                               as_index=False)
                 .agg(n_deleted=('ge5_B', 'size'), n_ge5_B=('ge5_B', 'sum'),
                      n_ge5_C=('ge5_C', 'sum'), n_unmoved_B=('unmoved_B', 'sum')))

    rng = np.random.default_rng(seed)
    samples = []
    for (lattice, label), group in frame.groupby(['bravais_lattice', 'label'], sort=False):
        take = min(group.shape[0], INCORRECT_SAMPLE)
        chosen = rng.choice(group.shape[0], size=take, replace=False)
        samples.append(group.iloc[chosen][['condition_bundle', 'bravais_lattice', 'label',
                                           'm20_at_prune', 'M20_B', 'M20_C']])

    correct = frame.loc[frame['is_correct'], ['entry_id', 'bravais_lattice', 'condition_bundle',
                                              'split', 'm20_at_prune', 'M20_B', 'M20_C']].copy()
    return (correct, per_entry,
            pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(), kept)


def _rate_ci_by_entry(successes, totals, rng):
    """Bootstrap interval on a pooled rate, resampling ENTRIES.

    `successes` and `totals` are per-entry counts. Each draw resamples entries with replacement
    and re-pools, so an entry contributing many candidates moves the rate as one observation
    rather than as many -- which is the whole point of the rule, and the reason campaign 1's
    candidate-level intervals were too narrow.
    """
    n = successes.size
    if n == 0:
        return np.nan, np.nan
    draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    pooled = successes[draws].sum(axis=1) / np.maximum(totals[draws].sum(axis=1), 1)
    return float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))


def run_stage(args):
    shards = merit_shards(args.merit_root, args.arm)
    jobs = [(str(shard), 12345 + index) for index, shard in enumerate(shards)]
    started = time.time()
    correct, per_entry, samples, kept = [], [], [], []
    for done, pieces in enumerate(_map(_stage_worker, jobs, args.processes), start=1):
        for target, piece in zip((correct, per_entry, samples, kept), pieces):
            target.append(piece)
        print(f'  {done}/{len(jobs)} shards, {time.time() - started:.0f}s', flush=True)

    correct = pd.concat(correct, ignore_index=True)
    per_entry = pd.concat(per_entry, ignore_index=True)
    samples = pd.concat([piece for piece in samples if not piece.empty], ignore_index=True)
    kept = pd.concat(kept) if any(len(piece) for piece in kept) else pd.Series(dtype=int,
                                                                              name='n_kept')

    rng = np.random.default_rng(12345)
    rows = []
    for (bundle, lattice, label), group in per_entry.groupby(
            ['condition_bundle', 'bravais_lattice', 'label'], sort=False):
        totals = group['n_deleted'].to_numpy()
        n = int(totals.sum())
        ge5_b = int(group['n_ge5_B'].sum())
        ge5_c = int(group['n_ge5_C'].sum())
        low_c, high_c = _rate_ci_by_entry(group['n_ge5_C'].to_numpy(), totals, rng)
        sample = samples[(samples['condition_bundle'] == bundle)
                         & (samples['bravais_lattice'] == lattice)
                         & (samples['label'] == label)]
        rows.append({'arm': args.arm, 'condition_bundle': bundle, 'bravais_lattice': lattice,
                     'label': label, 'n_entries': int(group.shape[0]), 'n_deleted': n,
                     'median_at_prune': float(sample['m20_at_prune'].median()),
                     'median_B': float(sample['M20_B'].median()),
                     'median_C': float(sample['M20_C'].median()),
                     'frac_ge5_B': ge5_b / n, 'frac_ge5_C': ge5_c / n,
                     'frac_ge5_C_ci_low': low_c, 'frac_ge5_C_ci_high': high_c,
                     'frac_unmoved_B': int(group['n_unmoved_B'].sum()) / n})

    table = pd.DataFrame(rows).sort_values(['condition_bundle', 'bravais_lattice', 'label'])
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifact_dir / f'S03_prune_stage_{args.arm}.csv', index=False)
    per_entry.to_csv(artifact_dir / f'S03_prune_stage_per_entry_{args.arm}.csv', index=False)
    correct.to_csv(artifact_dir / f'S03_prune_stage_deleted_correct_{args.arm}.csv', index=False)

    _print_stage_headline(table, per_entry, correct, kept, args.arm, rng)
    print(f'\nwrote {artifact_dir}/S03_prune_stage_{args.arm}.csv')


def _print_stage_headline(table, per_entry, correct, kept, arm, rng):
    """Gate condition 2: what fraction of the correct candidates the cut deletes would have
    cleared the same bar after refinement and extinction assignment."""
    pooled = pd.DataFrame([
        {'label': label,
         'n_deleted': int(group['n_deleted'].sum()),
         'frac_ge5_B': float((group['frac_ge5_B'] * group['n_deleted']).sum()
                             / group['n_deleted'].sum()),
         'frac_ge5_C': float((group['frac_ge5_C'] * group['n_deleted']).sum()
                             / group['n_deleted'].sum())}
        for label, group in table.groupby('label', sort=False)]).set_index('label')
    intervals = []
    for label, group in per_entry.groupby('label', sort=False):
        by_entry = group.groupby('entry_id', as_index=False).sum(numeric_only=True)
        low, high = _rate_ci_by_entry(by_entry['n_ge5_C'].to_numpy(),
                                      by_entry['n_deleted'].to_numpy(), rng)
        intervals.append({'label': label, 'frac_ge5_C_ci_low': low, 'frac_ge5_C_ci_high': high})
    pooled = pooled.join(pd.DataFrame(intervals).set_index('label'))
    print(f'\n[{arm}] candidates the production cut deletes (m20_at_prune < 5.0):')
    print(pooled.to_string())

    per_lattice = table.pivot_table(index='bravais_lattice', columns='label',
                                    values='frac_ge5_C', aggfunc='mean')
    print('\nfraction reaching M20 >= 5.0 at point C, by lattice:')
    print(per_lattice.to_string())

    # Entries whose every correct candidate was deleted, and whether refinement returns one.
    all_correct = correct.groupby(['condition_bundle', 'entry_id']).size().rename('n_deleted')
    rescued = (correct[correct['M20_C'] >= PRODUCTION_PRUNE_THRESHOLD]
               .groupby(['condition_bundle', 'entry_id']).size().rename('n_back'))
    joined = pd.concat([all_correct, kept, rescued], axis=1).fillna(0)
    lost = joined[joined['n_kept'] == 0]
    print(f'\nentry-bundle cells whose every correct candidate the cut deletes: {len(lost)}')
    if len(lost):
        print(f'  of which point C would return at least one above 5.0: '
              f'{int((lost["n_back"] > 0).sum())} '
              f'({(lost["n_back"] > 0).mean():.1%})')


# ---------------------------------------------------------------------------------------------
# stage: dedup -- re-price the deduplication tie-break at the cut
# ---------------------------------------------------------------------------------------------

# The cuts the tie-break is priced at. Its damage roughly doubles between the production cut and
# threshold 0 (F-155: 0-4 entries against 3-9), so it gets worse exactly where a lower cut is
# heading, and the two cannot be chosen independently.
DEDUP_CUTS = (0.0, 2.5, 3.0, 3.5, 5.0)

DOWNSAMPLE_CHUNK_SIZE = 1000


def deduplicate(xnn, M20, Minfo, n_indexed, lattice_system, radius):
    """Production's own collapse over one (entry, lattice) pool. Returns the surviving rows.

    Not a reimplementation: it sorts by reciprocal volume, chunks at 1 000 and calls
    `MPIOptimizer._downsample_chunk` itself, so the densest-point rule, the first-maximum
    tie-breaks and the chunk boundaries are production's. Row identity travels in the
    `spacegroup` slot, which that function only ever permutes -- it never reads it.

    The NaN filter production applies before this point has already been applied to the dump, so
    it is asserted rather than repeated.
    """
    from mlindex.optimization.MPIOptimizer import _downsample_chunk
    from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
    from mlindex.utilities.UnitCellTools import get_unit_cell_volume

    if np.isnan(xnn).any():
        raise ValueError('NaN cell in a pre-deduplication pool; production filters these first')

    reciprocal_volume = get_unit_cell_volume(
        get_reciprocal_unit_cell_from_xnn(xnn, partial_unit_cell=True,
                                          lattice_system=lattice_system),
        partial_unit_cell=True, lattice_system=lattice_system)
    order = np.argsort(reciprocal_volume)
    identity = list(order)
    xnn, M20, Minfo, n_indexed = xnn[order], M20[order], Minfo[order], n_indexed[order]

    n_chunks = xnn.shape[0] // DOWNSAMPLE_CHUNK_SIZE + 1
    kept = []
    for chunk_index in range(n_chunks):
        start = chunk_index * DOWNSAMPLE_CHUNK_SIZE
        end = None if chunk_index == n_chunks - 1 else (chunk_index + 1) * DOWNSAMPLE_CHUNK_SIZE
        result = _downsample_chunk((xnn[start:end], M20[start:end], Minfo[start:end],
                                    n_indexed[start:end], identity[start:end], radius))
        kept += list(result[4])
    return np.asarray(kept, dtype=np.int64)


def _dedup_gate_worker(shard):
    """Reproduce one t5 shard's own `candidates_` output from its `predownsample_` rows.

    Both come from the same run, so this is not a comparison against a model of production -- it
    is production's input and production's output, with the emulator in between.
    """
    shard = Path(shard)
    entering = pd.read_parquet(shard)
    survivors = pd.read_parquet(
        shard.parent / f'candidates_{shard.stem.split("_", 1)[1]}.parquet')
    checked = 0
    mismatched = []
    for (entry_id, lattice), group in entering.groupby(['entry_id', 'bravais_lattice'],
                                                       sort=False):
        expected = survivors[(survivors['entry_id'] == entry_id)
                             & (survivors['bravais_lattice'] == lattice)]
        if expected.empty:
            continue
        kept = deduplicate(np.stack(group['xnn'].to_numpy()).astype(np.float64),
                           group['M20'].to_numpy(), group['Minfo'].to_numpy(),
                           group['n_indexed'].to_numpy(), group['lattice_system'].iloc[0],
                           float(group['downsample_radius'].iloc[0]))
        produced = np.stack(group['xnn'].to_numpy()).astype(np.float64)[kept]
        wanted = np.stack(expected['xnn'].to_numpy()).astype(np.float64)
        checked += 1
        if produced.shape != wanted.shape or not np.array_equal(produced, wanted):
            mismatched.append((entry_id, lattice, produced.shape[0], wanted.shape[0]))
    return checked, mismatched


def run_dedup(args):
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # --- the gate, on the real production-threshold run -------------------------------------
    t5_root = Path(BASE) / 'mlindex' / 'characterization' / 'fom' / 'allstrata' / 't5'
    gate_shards = sorted(t5_root.glob('*/predownsample_*.parquet'))
    if args.limit_shards:
        gate_shards = gate_shards[:args.limit_shards]
    checked = 0
    mismatched = []
    started = time.time()
    for done, (count, bad) in enumerate(_map(_dedup_gate_worker,
                                             [str(shard) for shard in gate_shards],
                                             args.processes), start=1):
        checked += count
        mismatched += bad
        print(f'  gate {done}/{len(gate_shards)} shards, {checked} pools, '
              f'{time.time() - started:.0f}s', flush=True)
    print(f'\nGATE  emulator vs the t5 run\'s own output: '
          f'{checked - len(mismatched)} of {checked} pools reproduced exactly')
    for entry_id, lattice, produced, wanted in mismatched[:10]:
        print(f'  MISMATCH {entry_id}/{lattice}: {produced} survivors against {wanted}')
    if mismatched:
        raise SystemExit('deduplication emulator does not reproduce production')

    # --- the reprice, on the threshold-0 arm --------------------------------------------------
    arm = ARMS[args.arm]
    jobs = []
    for bundle, bundle_dir in bundle_directories(Path(BASE) / arm['root']).items():
        for shard in sorted(bundle_dir.glob('predownsample_*.parquet')):
            jobs.append((str(shard), bundle, os.path.join(BASE, arm['labels'])))
    if args.limit_shards:
        jobs = jobs[:args.limit_shards]

    started = time.time()
    pieces = []
    for done, piece in enumerate(_map(_dedup_reprice_worker, jobs, args.processes), start=1):
        pieces.append(piece)
        print(f'  reprice {done}/{len(jobs)} shards, {time.time() - started:.0f}s', flush=True)
    per_entry = pd.concat(pieces, ignore_index=True)

    summary = (per_entry.groupby(['condition_bundle', 'prune_cut'], as_index=False)
               .agg(n_entries=('entry_id', 'nunique'),
                    reachable_before=('reachable_before', 'sum'),
                    reachable_after=('reachable_after', 'sum'),
                    correct_before=('n_correct_before', 'sum'),
                    correct_after=('n_correct_after', 'sum'),
                    pool_before=('pool_before', 'sum'),
                    pool_after=('pool_after', 'sum')))
    summary['entries_lost_to_tiebreak'] = (summary['reachable_before']
                                           - summary['reachable_after'])
    summary['arm'] = args.arm
    summary.to_csv(artifact_dir / f'S03_dedup_reprice_{args.arm}.csv', index=False)
    per_entry.to_csv(artifact_dir / f'S03_dedup_reprice_per_entry_{args.arm}.csv', index=False)
    print()
    print(summary.to_string(index=False))
    print(f'\nwrote {artifact_dir}/S03_dedup_reprice_{args.arm}.csv')


def _dedup_reprice_worker(job):
    """What the tie-break costs at each cut: entries that had a correct candidate entering
    deduplication and do not have one leaving it."""
    shard, bundle, labels_dir = job
    frame = pd.read_parquet(shard)
    frame['is_correct'] = join_labels(frame, pd.read_parquet(label_path(labels_dir, bundle,
                                                                       shard)))
    rows = []
    for entry_id, entry_pool in frame.groupby('entry_id', sort=False):
        for cut in DEDUP_CUTS:
            admitted = entry_pool.loc[entry_pool['m20_at_prune'] >= cut]
            if admitted.empty:
                continue
            correct_after = 0
            pool_after = 0
            for lattice, group in admitted.groupby('bravais_lattice', sort=False):
                kept = deduplicate(np.stack(group['xnn'].to_numpy()).astype(np.float64),
                                   group['M20'].to_numpy(), group['Minfo'].to_numpy(),
                                   group['n_indexed'].to_numpy(),
                                   group['lattice_system'].iloc[0],
                                   float(group['downsample_radius'].iloc[0]))
                pool_after += kept.size
                correct_after += int(group['is_correct'].to_numpy()[kept].sum())
            correct_before = int(admitted['is_correct'].sum())
            rows.append({'condition_bundle': bundle, 'entry_id': entry_id, 'prune_cut': cut,
                         'pool_before': int(admitted.shape[0]), 'pool_after': pool_after,
                         'n_correct_before': correct_before, 'n_correct_after': correct_after,
                         'reachable_before': bool(correct_before), 
                         'reachable_after': bool(correct_after)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# stage: figures
# ---------------------------------------------------------------------------------------------

# Lattices in symmetry order, high to low, so every per-lattice panel reads left to right as
# symmetry falls -- which is the axis the whole result is organised around.
LATTICE_ORDER = ('cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP', 'oF', 'oI', 'oC', 'oP', 'mC', 'mP',
                 'aP')

# The criteria worth drawing. The full sweep is in the CSV; a figure with seventeen curves says
# nothing, and these are the ones the decision turns on: the incumbent, the best classical merit
# at each later point, and M20 at each later point as the like-for-like control.
FIGURE_CRITERIA = (('m20_at_prune', 'M20 at the cut (production)', '#000000', '-'),
                   ('M20_B', 'M20 after refinement', '#0072B2', '--'),
                   ('M20_C', 'M20 after extinction assignment', '#0072B2', '-'),
                   ('M_sym_B', 'M_sym after refinement', '#D55E00', '--'),
                   ('M_sym_C', 'M_sym after extinction assignment', '#D55E00', '-'),
                   ('X_N_C', 'X_N after extinction assignment', '#009E73', '-'))


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'lines.linewidth': 1.2, 'axes.linewidth': 0.7,
        })
    return plt


def _pooled_stage(arm, artifact_dir):
    """Per-lattice fractions, pooled over bundles and weighted by how many rows each contributes."""
    table = pd.read_csv(artifact_dir / f'S03_prune_stage_{arm}.csv')
    rows = []
    for (lattice, label), group in table.groupby(['bravais_lattice', 'label']):
        total = group['n_deleted'].sum()
        rows.append({'bravais_lattice': lattice, 'label': label, 'n_deleted': int(total),
                     'frac_ge5_B': float((group['frac_ge5_B'] * group['n_deleted']).sum() / total),
                     'frac_ge5_C': float((group['frac_ge5_C'] * group['n_deleted']).sum() / total)})
    return pd.DataFrame(rows)


def figure_stage(artifact_dir):
    """The candidate paper figure: what the cut deletes, and what would have come back.

    Two panels because the answer is different on the two populations, and the difference is the
    result -- not a caveat on it. Panel widths follow the lattice counts so the bars are the same
    width in both, which is what makes the two readable against each other.
    """
    plt = _style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True,
                                gridspec_kw={'width_ratios': [14, 3.6], 'wspace': 0.08})
    handles = None
    for axis, arm, title in zip(axes, ('general', 'hard'),
                                ('general population\n210 entries, all 14 lattices',
                                 'hard stratum\n243 entries, mC/mP/aP')):
        pooled = _pooled_stage(arm, artifact_dir)
        lattices = [name for name in LATTICE_ORDER
                    if ((pooled['bravais_lattice'] == name)
                        & (pooled['label'] == 'correct')).any()]
        positions = np.arange(len(lattices))
        correct = pooled[pooled['label'] == 'correct'].set_index('bravais_lattice')
        incorrect = pooled[pooled['label'] == 'incorrect'].set_index('bravais_lattice')

        bars = [
            axis.bar(positions - 0.19, [correct.loc[name, 'frac_ge5_C'] for name in lattices],
                     0.36, color='#AECDE3', edgecolor='#0B5D91', linewidth=0.6, zorder=2),
            axis.bar(positions - 0.19, [correct.loc[name, 'frac_ge5_B'] for name in lattices],
                     0.36, color='#0B5D91', edgecolor='#0B5D91', linewidth=0.6, zorder=3),
            axis.bar(positions + 0.19, [incorrect.loc[name, 'frac_ge5_C'] for name in lattices],
                     0.36, color='#E5A87F', edgecolor='#C1571A', linewidth=0.6, zorder=2),
            ]
        handles = handles or bars
        for position, name in zip(positions, lattices):
            axis.annotate(f'{int(correct.loc[name, "n_deleted"]):,}',
                          (position - 0.19, correct.loc[name, 'frac_ge5_C']),
                          textcoords='offset points', xytext=(0, 2.5), ha='center',
                          fontsize=5.0, color='#0B5D91', zorder=4)
        axis.set_xticks(positions)
        axis.set_xticklabels(lattices)
        axis.set_title(title, fontsize=7.5, linespacing=1.3, pad=6)
        axis.set_ylim(0, 1.06)
        axis.set_xlim(-0.65, len(lattices) - 0.35)
        axis.tick_params(axis='y', labelleft=(arm == 'general'))
    axes[0].set_ylabel('reaches M20 $\\geq$ 5\nonce fitting finishes', linespacing=1.4)
    figure.legend(handles,
                  ['correct candidates, after refinement AND extinction assignment',
                   'correct candidates, after refinement alone',
                   'incorrect candidates, after both (the control)'],
                  loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False,
                  fontsize=6.5, handlelength=1.4, handleheight=0.8)
    figure.suptitle('Of the candidates the prune deletes, which would have cleared the same bar?',
                    fontsize=9, y=1.10)
    figure.text(0.5, -0.245,
                'Numbers above the blue bars are how many correct candidates the cut deletes on '
                'that lattice. Symmetry falls left to right.\nTriclinic has exactly one '
                'extinction group, so its two blue bars are equal by construction: only '
                'refinement can lift a triclinic cell.',
                ha='center', fontsize=6, color='#444444', linespacing=1.5)
    destination = artifact_dir / 'S03_prune_stage.png'
    figure.savefig(destination)
    plt.close(figure)
    return destination


def figure_retention(artifact_dir):
    """Reachability against pool size and against wall clock, for every criterion and stage.

    The lower row is the result. A criterion evaluated after refinement is a near-vertical line
    there: its cost hardly moves with its threshold, because the steps upstream of it have already
    run on every candidate. The incumbent, applied where the cut already is, sweeps the whole
    cost axis -- which is why lowering it is cheap and moving it is not.
    """
    plt = _style()
    arms = [arm for arm in ('general', 'hard')
            if (artifact_dir / f'S03_prune_retention_{arm}.csv').exists()]
    figure, axes = plt.subplots(2, len(arms), figsize=(3.5 * len(arms), 5.2), squeeze=False,
                                sharey=True)
    handles, labels = [], []

    for column, arm in enumerate(arms):
        data = pd.read_csv(artifact_dir / f'S03_prune_retention_{arm}.csv')
        data = data[data['bravais_lattice'] == 'ALL']
        pooled = (data.groupby(['criterion', 'target_fraction'], as_index=False)
                  .agg(achieved_fraction=('achieved_fraction', 'mean'),
                       reachability=('reachability', 'mean'),
                       low=('reachability_ci_low', 'mean'),
                       high=('reachability_ci_high', 'mean'),
                       estimated_seconds=('estimated_seconds', 'sum')))
        production = 0.028 if arm.endswith('general') else 0.004
        cells = ARM_CELLS[arm]

        for row, (xcolumn, xlabel) in enumerate(
                (('achieved_fraction', 'surviving share of the threshold-0 pool'),
                 ('estimated_seconds', f'post-cut wall clock (s, {cells} patterns)'))):
            axis = axes[row][column]
            for label, name, colour, style in FIGURE_CRITERIA:
                curve = pooled[pooled['criterion'] == label].sort_values(xcolumn)
                if curve.empty:
                    continue
                line, = axis.plot(curve[xcolumn], curve['reachability'], style, color=colour,
                                  marker='o', markersize=2.2)
                if row == 0 and column == 0:
                    handles.append(line)
                    labels.append(name)
                if label == 'm20_at_prune':
                    axis.fill_between(curve[xcolumn], curve['low'], curve['high'],
                                      color='#777777', alpha=0.16, linewidth=0)
            axis.set_xscale('log')
            axis.set_xlabel(xlabel)
            if column == 0:
                axis.set_ylabel('entries keeping a correct candidate')
            if row == 0:
                incumbent = pooled[(pooled['criterion'] == 'm20_at_prune')
                                   & (np.isclose(pooled['target_fraction'], production))]
                if not incumbent.empty:
                    position = float(incumbent['achieved_fraction'].iloc[0])
                    axis.axvline(position, color='#999999', linewidth=0.7, linestyle=':')
                    axis.annotate('production cut\n(M20 = 5)', (position, 0.02),
                                  textcoords='offset points', xytext=(4, 0), fontsize=6,
                                  color='#777777', va='bottom', linespacing=1.3)
                axis.set_title('general population, 210 patterns' if arm == 'general'
                               else 'hard stratum, 972 pattern-conditions', fontsize=7.5, pad=6)
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol=2,
                  frameon=False, fontsize=6.5, handlelength=2.2)
    figure.suptitle('What a cut keeps, against what it costs', fontsize=9, y=0.995)
    figure.text(0.5, -0.145,
                'Shaded band: 95 % bootstrap interval over entries on the incumbent. Wall clock '
                'charges the steps upstream of a cut on the WHOLE pool, which is why the\n'
                'later-stage criteria are near-vertical: their thresholds move what survives '
                'without moving what the run has already paid for.',
                ha='center', fontsize=6, color='#444444', linespacing=1.5)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    destination = artifact_dir / 'S03_prune_retention.png'
    figure.savefig(destination)
    plt.close(figure)
    return destination


def run_figures(args):
    artifact_dir = Path(BASE) / args.artifact_dir
    for destination in (figure_stage(artifact_dir), figure_retention(artifact_dir)):
        print(f'wrote {destination}')


# ---------------------------------------------------------------------------------------------
# stage: threshold -- the absolute sweep, which is what a deployment cut is expressed in
# ---------------------------------------------------------------------------------------------

# The retention sweep places cuts at matched surviving FRACTION, which is the only fair way to
# compare merits. A recommendation, though, has to be a number a person can read off a merit they
# already understand, so the incumbent gets a second sweep in its own units.
ABSOLUTE_CUTS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)

# Entry-bundle cells per arm: the denominator for an absolute reachability rate. Entries are
# balanced across lattices in both arms, so an unweighted rate over cells is already unweighted
# across lattices (PROTOCOL section 3 rule 6).
ARM_CELLS = {'general': 210, 'hard': 972, 'rerun-general': 210, 'rerun-hard': 972}


def _absolute_worker(job):
    shard, per_candidate = job
    frame = pd.read_parquet(shard, columns=['entry_id', 'bravais_lattice', 'condition_bundle',
                                            'split', 'is_correct', 'm20_at_prune'])
    counts = {}
    split_counts = {}
    for lattice, group in frame.groupby('bravais_lattice', sort=False):
        values = group['m20_at_prune'].to_numpy()
        counts[lattice] = np.array([int((values >= cut).sum()) for cut in ABSOLUTE_CUTS])
        for split, piece in group.groupby('split', sort=False):
            values = piece['m20_at_prune'].to_numpy()
            split_counts[(lattice, split)] = np.array(
                [int((values >= cut).sum()) for cut in ABSOLUTE_CUTS])
    correct = frame.loc[frame['is_correct']]
    best = (correct.groupby(['condition_bundle', 'entry_id', 'bravais_lattice', 'split'],
                            as_index=False)['m20_at_prune'].max())
    return counts, split_counts, best, int(frame.shape[0])


def run_threshold(args):
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cost_table = pd.read_csv(artifact_dir / 'S03_cost_per_candidate.csv').set_index(
        'bravais_lattice')
    with open(artifact_dir / 'S03_cost_model.json', encoding='utf-8') as handle:
        model = json.load(handle)
    parallelism = model['effective_parallelism']

    shards = merit_shards(args.merit_root, args.arm)
    per_candidate = cost_table['per_candidate_downstream_A'].to_dict()
    counts = {}
    split_counts = {}
    bests = []
    total = 0
    started = time.time()
    for done, (shard_counts, split_shard_counts, best, rows) in enumerate(
            _map(_absolute_worker, [(str(shard), per_candidate) for shard in shards],
                 args.processes), start=1):
        for lattice, values in shard_counts.items():
            counts[lattice] = counts.get(lattice, 0) + values
        for key, values in split_shard_counts.items():
            split_counts[key] = split_counts.get(key, 0) + values
        bests.append(best)
        total += rows
        print(f'  {done}/{len(shards)} shards, {time.time() - started:.0f}s', flush=True)
    best = pd.concat(bests, ignore_index=True)

    cells = ARM_CELLS[args.arm]
    rows = []
    for index, cut in enumerate(ABSOLUTE_CUTS):
        surviving = int(sum(values[index] for values in counts.values()))
        seconds = sum(values[index] * per_candidate.get(lattice, 0.0)
                      for lattice, values in counts.items()) / parallelism
        reachable = int((best['m20_at_prune'] >= cut).sum())
        row = {'arm': args.arm, 'prune_cut': cut, 'n_candidates': total,
               'n_surviving': surviving, 'surviving_share': surviving / total,
               'n_cells': cells, 'n_reachable_cells': reachable,
               'reachability': reachable / cells,
               'post_cut_seconds': seconds, 'post_cut_seconds_per_pattern': seconds / cells}
        for split in ('fom-train', 'fom-dev'):
            here = best[best['split'] == split]
            row[f'n_reachable_{split}'] = int((here['m20_at_prune'] >= cut).sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(artifact_dir / f'S03_prune_threshold_sweep_{args.arm}.csv', index=False)

    # Per lattice, in the same units -- C2-Q-006 asks whether one scalar is the right shape, and
    # the surviving share a single cut produces on each lattice is the evidence.
    per_lattice = []
    for index, cut in enumerate(ABSOLUTE_CUTS):
        for lattice, values in counts.items():
            here = best[best['bravais_lattice'] == lattice]
            per_lattice.append({'arm': args.arm, 'prune_cut': cut, 'bravais_lattice': lattice,
                                'n_surviving': int(values[index]),
                                'surviving_share': values[index] / values[0],
                                'n_reachable_cells': int((here['m20_at_prune'] >= cut).sum()),
                                'n_cells_with_a_correct_candidate': int(here.shape[0]),
                                'post_cut_seconds': (values[index]
                                                     * per_candidate.get(lattice, 0.0)
                                                     / parallelism)})
    pd.DataFrame(per_lattice).to_csv(
        artifact_dir / f'S03_prune_threshold_sweep_per_lattice_{args.arm}.csv', index=False)

    print()
    print(frame[['prune_cut', 'n_surviving', 'surviving_share', 'n_reachable_cells',
                 'reachability', 'post_cut_seconds', 'post_cut_seconds_per_pattern']]
          .to_string(index=False))
    split_test = split_aware_per_lattice_test(best, split_counts, per_candidate, parallelism,
                                              list(ABSOLUTE_CUTS))
    split_test.insert(0, 'arm', args.arm)
    split_test.to_csv(artifact_dir / f'S03_prune_per_lattice_split_test_{args.arm}.csv',
                      index=False)

    frontier, mapping = per_lattice_frontier(pd.DataFrame(per_lattice), cells)
    frontier.to_csv(artifact_dir / f'S03_prune_per_lattice_frontier_{args.arm}.csv', index=False)
    mapping.to_csv(artifact_dir / f'S03_prune_per_lattice_cuts_{args.arm}.csv', index=False)

    print()
    print(frame[['prune_cut', 'n_surviving', 'surviving_share', 'n_reachable_cells',
                 'reachability', 'post_cut_seconds', 'post_cut_seconds_per_pattern']]
          .to_string(index=False))
    _print_per_lattice_verdict(frame, frontier, args.arm)
    print(f'\n[{args.arm}] the same mapping fitted on fom-train and read on fom-dev:')
    print(split_test.to_string(index=False))
    print(f'\nwrote {artifact_dir}/S03_prune_threshold_sweep_{args.arm}.csv')
    print(f'wrote {artifact_dir}/S03_prune_threshold_sweep_per_lattice_{args.arm}.csv')
    print(f'wrote {artifact_dir}/S03_prune_per_lattice_frontier_{args.arm}.csv')
    print(f'wrote {artifact_dir}/S03_prune_per_lattice_cuts_{args.arm}.csv')
    print(f'wrote {artifact_dir}/S03_prune_per_lattice_split_test_{args.arm}.csv')


def split_aware_per_lattice_test(best, counts, per_candidate, parallelism, cuts):
    """Fit the per-lattice mapping on `fom-train`, evaluate it on `fom-dev`, against one scalar.

    The in-sample frontier below is an ORACLE: it chooses fourteen cuts on the same cells it is
    scored on. A per-lattice cut is a fourteen-parameter policy, so PROTOCOL section 8 requires it
    to be selected on one split and reported on another -- and campaign 1's most expensive habit
    was reading a design decision off the split it was tuned on.

    Returns the comparison table. Reading it is the point: if the fitted mapping does not beat the
    scalar out of sample, the in-sample frontier is measuring its own freedom, not a real gain.
    """
    lattices = sorted(best['bravais_lattice'].unique())

    def reachable(split, mapping):
        here = best[best['split'] == split]
        return int(sum((here[here['bravais_lattice'] == lattice]['m20_at_prune']
                        >= mapping[lattice]).sum() for lattice in lattices))

    def seconds(split, mapping):
        total = 0.0
        for lattice in lattices:
            values = counts.get((lattice, split))
            if values is None:
                continue
            total += (values[cuts.index(mapping[lattice])]
                      * per_candidate.get(lattice, 0.0))
        return total / parallelism

    fitted = []
    for price in np.concatenate([[0.0], np.logspace(-4, 5, 3000)]):
        mapping = {}
        for lattice in lattices:
            values = counts.get((lattice, 'fom-train'))
            if values is None:
                mapping[lattice] = cuts[-1]
                continue
            train = best[(best['split'] == 'fom-train')
                         & (best['bravais_lattice'] == lattice)]['m20_at_prune'].to_numpy()
            gain = np.array([int((train >= cut).sum()) for cut in cuts])
            cost = price * values * per_candidate.get(lattice, 0.0) / parallelism
            mapping[lattice] = cuts[int(np.argmax(gain - cost))]
        fitted.append({'dev_seconds': seconds('fom-dev', mapping),
                       'dev_reachable': reachable('fom-dev', mapping)})
    fitted = pd.DataFrame(fitted).drop_duplicates()

    rows = []
    for cut in cuts:
        mapping = {lattice: cut for lattice in lattices}
        budget = seconds('fom-dev', mapping)
        affordable = fitted[fitted['dev_seconds'] <= budget * 1.0001]
        rows.append({'global_cut': cut, 'dev_seconds': budget,
                     'global_dev_reachable': reachable('fom-dev', mapping),
                     'per_lattice_dev_reachable': (int(affordable['dev_reachable'].max())
                                                   if not affordable.empty else np.nan)})
    table = pd.DataFrame(rows)
    table['per_lattice_gain_cells'] = (table['per_lattice_dev_reachable']
                                       - table['global_dev_reachable'])
    return table


def per_lattice_frontier(per_lattice, cells):
    """The best (cost, reachability) a per-lattice cut can reach, and the cuts that reach it.

    Every entry has exactly one true lattice, so reachable cells add across lattices and the
    allocation separates: for a price lambda on seconds, each lattice independently takes the cut
    maximising `cells - lambda * seconds`. Sweeping lambda traces the whole frontier exactly --
    no search, and no risk of the coarse-grid artefact a fixed-budget optimiser produces.

    C2-Q-006 asks whether one scalar is the right shape for this parameter. The frontier is the
    answer: where it sits below the global curve, a per-lattice mapping buys the same reachability
    for less, and the gap is the cost of insisting on one number.
    """
    lattices = sorted(per_lattice['bravais_lattice'].unique())
    curves = {lattice: per_lattice[per_lattice['bravais_lattice'] == lattice]
              .sort_values('prune_cut').reset_index(drop=True) for lattice in lattices}
    points = {}
    for price in np.concatenate([[0.0], np.logspace(-4, 5, 2000)]):
        chosen = {}
        seconds = 0.0
        reachable = 0
        for lattice, curve in curves.items():
            value = (curve['n_reachable_cells'].to_numpy()
                     - price * curve['post_cut_seconds'].to_numpy())
            # Ties go to the CHEAPER cut: np.argmax takes the first maximum and the grid is
            # ordered by increasing cut, i.e. decreasing cost, so the tie-break is already right.
            best = int(np.argmax(value))
            chosen[lattice] = float(curve['prune_cut'].iloc[best])
            seconds += float(curve['post_cut_seconds'].iloc[best])
            reachable += int(curve['n_reachable_cells'].iloc[best])
        # Keep the cheapest allocation at each reachability level.
        if reachable not in points or seconds < points[reachable][0]:
            points[reachable] = (seconds, chosen)

    frontier = pd.DataFrame([
        {'n_reachable_cells': reachable, 'reachability': reachable / cells,
         'post_cut_seconds': seconds, 'post_cut_seconds_per_pattern': seconds / cells}
        for reachable, (seconds, _) in sorted(points.items())])
    mapping = pd.DataFrame([
        dict({'n_reachable_cells': reachable, 'post_cut_seconds': seconds}, **chosen)
        for reachable, (seconds, chosen) in sorted(points.items())])
    return frontier, mapping


def _print_per_lattice_verdict(global_curve, frontier, arm):
    """At matched reachability, what does insisting on one global scalar cost?"""
    print(f'\n[{arm}] a per-lattice cut against the best global scalar, at matched reachability:')
    print(f'{"cells":>7} {"global s":>10} {"per-lattice s":>14} {"cheaper by":>11}')
    for _, row in frontier.iterrows():
        affordable = global_curve[global_curve['n_reachable_cells'] >= row['n_reachable_cells']]
        if affordable.empty:
            continue
        global_seconds = float(affordable['post_cut_seconds'].min())
        if row['post_cut_seconds'] <= 0:
            continue
        print(f'{int(row["n_reachable_cells"]):7d} {global_seconds:10.1f} '
              f'{row["post_cut_seconds"]:14.1f} '
              f'{global_seconds / row["post_cut_seconds"]:10.1f}x')


# ---------------------------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description='S03 -- the prune: which merit, at which stage, at what value.')
    parser.add_argument('--stage', required=True,
                        choices=['merits', 'cost', 'stage', 'retention', 'dedup',
                                 'threshold', 'figures'])
    parser.add_argument('--arm', default='general', choices=sorted(ARMS),
                        help='which threshold-0 arm to read (default: general)')
    parser.add_argument('--processes', type=int, default=1,
                        help='worker processes over shards (default: 1)')
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    parser.add_argument('--merit-root', default=MERIT_ROOT,
                        help='where --stage merits wrote its per-candidate table')
    parser.add_argument('--bravais-lattices', default=None,
                        help='comma separated; default is all fourteen (cost stage)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='best-of-N timing repeats (cost stage)')
    parser.add_argument('--limit-shards', type=int, default=None,
                        help='process only the first N shards; for smoke tests, never for a result')
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.stage == 'merits':
        run_merits(args)
    elif args.stage == 'cost':
        run_cost(args)
    elif args.stage == 'retention':
        run_retention(args)
    elif args.stage == 'stage':
        run_stage(args)
    elif args.stage == 'dedup':
        run_dedup(args)
    elif args.stage == 'threshold':
        run_threshold(args)
    elif args.stage == 'figures':
        run_figures(args)
    else:
        raise SystemExit(f'--stage {args.stage} is not implemented yet')


if __name__ == '__main__':
    main()
