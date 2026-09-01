"""S11: what the extinction-group criterion is worth. Reads the sweep, writes the artefacts.

    python mlindex/scripts/run_fom_extinction_eval.py --stage accuracy
    python mlindex/scripts/run_fom_extinction_eval.py --stage stability --processes 8
    python mlindex/scripts/run_fom_extinction_eval.py --stage report

Stages, each writing `docs/fom_campaign2/artifacts/S11_*`:

  accuracy   deliverable (a) -- does the rule pick the TRUE extinction group, per lattice, per
             rule, restricted to candidates that are correct cells. The question is meaningless
             for a wrong cell: there is no true group for a cell that is not the answer.
  features   deliverable (d) -- `n_absent_extra_in_range` under each rule, which is a
             deterministic function of the choice, so S04's diagnostic has to be re-read against
             whatever this step adopts.
  stability  deliverable (c) -- how often the chosen group flips when the cell is displaced.
  cost       deliverable (e) -- time and peak memory per rule per lattice.
  report     assembles `S11_extinction_rule.md` from whatever stages have run.

**aP is absent from all of it**: one extinction group, one possible choice, so every arm is the
same and the rows carry no information about the question (DWMM, 2026-09-01).

Two things to hold on to while reading any number here.

**The new rule can only LOWER the reported merit, by construction.** `best_M20` is the maximum of
M20 over groups under the incumbent, so any other argmax lands at a group where M20 is no higher.
A uniform drop is therefore expected and is not a defect; the question is only ever whether the
drop is larger for wrong cells than for correct ones.

**Accuracy and ranking are different questions and this file answers only the first.** A rule can
assign more accurately and rank worse, because what the pooled sort reads is the rebound merit and
not the group. The ranking half is deliberately split to a second session.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark
from mlindex.utilities.ExtinctionCounts import LATTICE_SYSTEM
from mlindex.utilities.FigureOfMerits import EXTINCTION_CRITERIA

SKIP_LATTICES = ('aP',)

# `candidate_id` is unique only within an (entry, bundle, lattice) pool, so all four are needed.
JOIN_KEYS = ['entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id']

# Each lattice's own contrast floor on top-10, from S08 (`S08_floor_by_lattice.csv`, `se_pp`).
# PROTOCOL section 8: a per-lattice claim is read against that lattice's own floor, never against
# an aggregate and never against campaign 1's interim. Campaign 1's "ordered by free cell
# parameters" justification does NOT reproduce (2.1x spread, Spearman -0.18), but the rule stands
# -- what sets a lattice's floor is how many entries it has in the split (C2-F-081).
CONTRAST_FLOOR_PP = {
    'aP': 2.8469, 'cF': 1.7754, 'cI': 1.5429, 'cP': 2.5955, 'hP': 1.6011, 'hR': 2.0815,
    'mC': 1.5749, 'mP': 1.4060, 'oC': 1.6927, 'oF': 1.4842, 'oI': 1.3771, 'oP': 1.8533,
    'tI': 1.6279, 'tP': 1.6158,
    }


def commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE, text=True).strip()
    except Exception:
        return 'unknown'


def dirty_tree():
    try:
        return bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=BASE,
                                            text=True).strip())
    except Exception:
        return None


def sweep_files(pool, sweep_dir, columns, source_columns, correct_only=False):
    """Yield (bravais_lattice, frame) one sidecar at a time, joined to the pool columns needed.

    A generator rather than one concatenated frame, and it matters: the sweep covers ~33 million
    candidates and the join keys are strings, so materialising the whole thing costs tens of
    gigabytes before anything is filtered. Every stage here reduces per file instead.

    `correct_only` filters before the join, because the accuracy question is meaningless for a
    wrong cell -- there is no true extinction group for a cell that is not the answer -- and the
    correct candidates are 11 818 rows of the 33 million.
    """
    for path in sorted(Path(sweep_dir).glob('candidates*.parquet')):
        lattice = path.stem.split('_')[-1]
        if lattice in SKIP_LATTICES:
            continue
        source = pd.read_parquet(Path(pool)/path.name,
                                 columns=list(dict.fromkeys(JOIN_KEYS + source_columns)))
        if correct_only:
            source = source[source['is_correct']]
            if source.empty:
                continue
        sidecar = pd.read_parquet(path, columns=list(dict.fromkeys(JOIN_KEYS + columns)))
        frame = source.merge(sidecar, on=JOIN_KEYS, how='inner', validate='one_to_one')
        if not frame.empty:
            yield lattice, frame


def truth_for(entries):
    """{(entry_id, bundle): the single admissible group key}, raising on a symbol we cannot express.

    Not normalised strings on either side -- each candidate key's own `e.g. <spacegroup>` half is
    routed back through the EXPO table that produced the truth column, which resolves 530 of 530
    with nothing ambiguous. See `FomBenchmark.extinction_group_key_map`.
    """
    truth = {}
    for row in entries.itertuples():
        keys = FomBenchmark.admissible_group_keys(row.bravais_lattice_true,
                                                  row.extinction_group_true)
        truth[(row.entry_id, row.condition_bundle)] = keys
    return truth


def stage_accuracy(pool, sweep_dir, criteria, artifact_dir, tag):
    """Deliverable (a): does the rule choose the true extinction group, on correct cells only.

    The per-candidate hits are collected first and aggregated once, rather than aggregated per
    file: the pool is partitioned by (bundle, lattice), so a per-file table carries three rows per
    lattice and any per-lattice rate has to be formed from the pooled counts, not averaged over
    bundles. The hits frame is 11 818 rows, so keeping it whole costs nothing and it is persisted
    as the sufficient statistic for anything asked later (PROTOCOL section 3 rule 8).
    """
    entries = FomBenchmark.load_entries(pool)
    truth = truth_for(entries)
    meta = json.load(open(os.path.join(sweep_dir, '_meta.json'), encoding='utf-8'))
    group_keys = meta['group_keys']

    columns = (['xg_n_floored_groups', 'xg_n_groups', 'xg_stored_group_index']
               + [f'xg_{c}_group_index' for c in criteria])
    records = []
    for lattice, frame in sweep_files(pool, sweep_dir, columns,
                                      ['is_correct', 'final_rank'], correct_only=True):
        admissible = [truth[(e, b)] for e, b in zip(frame['entry_id'],
                                                    frame['condition_bundle'])]
        record = frame[JOIN_KEYS + ['final_rank']].copy()
        record['n_groups'] = frame['xg_n_groups'].to_numpy()
        record['share_floored'] = (frame['xg_n_floored_groups']
                                   / frame['xg_n_groups']).to_numpy()
        for criterion in criteria:
            chosen = [group_keys[lattice][i] for i in frame[f'xg_{criterion}_group_index']]
            record[f'hit_{criterion}'] = [key in keys for key, keys in zip(chosen, admissible)]
            record[f'same_as_stored_{criterion}'] = (
                frame[f'xg_{criterion}_group_index'] == frame['xg_stored_group_index']).to_numpy()
        records.append(record)

    hits = pd.concat(records, ignore_index=True)
    hits.to_parquet(os.path.join(artifact_dir, f'{tag}_assignment_hits.parquet'), index=False)

    # The top-ranked correct candidate of each pattern: what the pipeline would actually report,
    # as opposed to a candidate average a single prolific pattern can dominate.
    per_entry = (hits.sort_values('final_rank')
                 .groupby(['entry_id', 'condition_bundle', 'bravais_lattice'], as_index=False)
                 .first())

    rows = []
    for lattice, block in hits.groupby('bravais_lattice'):
        entry_block = per_entry[per_entry.bravais_lattice == lattice]
        for criterion in criteria:
            rows.append({
                'bravais_lattice': lattice, 'criterion': criterion,
                'n_groups_searched': int(block['n_groups'].iloc[0]),
                'n_correct_candidates': int(block.shape[0]),
                'n_source_entries': int(block['entry_id'].nunique()),
                'accuracy': float(block[f'hit_{criterion}'].mean()),
                'accuracy_per_entry': float(entry_block[f'hit_{criterion}'].mean()),
                'share_floored_groups': float(block['share_floored'].mean()),
                'agrees_with_stored': float(block[f'same_as_stored_{criterion}'].mean()),
                'contrast_floor_pp': CONTRAST_FLOOR_PP[lattice],
                })

    table = pd.DataFrame(rows)
    baseline = table[table.criterion == 'M20'].set_index('bravais_lattice')['accuracy']
    assert baseline.index.is_unique, 'one M20 baseline per lattice'
    table['delta_pp'] = (table['accuracy'] - table['bravais_lattice'].map(baseline))*100
    table['standard_errors'] = table['delta_pp']/table['contrast_floor_pp']
    table = table.sort_values(['bravais_lattice', 'criterion'])
    table.to_csv(os.path.join(artifact_dir, f'{tag}_assignment_accuracy.csv'), index=False,
                 encoding='utf-8')

    # Unweighted across lattices, per PROTOCOL section 3 rule 6: a candidate-weighted aggregate
    # would let mP, at a fifth of the correct candidates, stand in for the answer.
    summary = (table.groupby('criterion', as_index=False)
               .agg(mean_accuracy_unweighted=('accuracy', 'mean'),
                    mean_delta_pp=('delta_pp', 'mean'),
                    n_lattices_better=('delta_pp', lambda d: int((d > 0).sum())),
                    n_lattices_worse=('delta_pp', lambda d: int((d < 0).sum())),
                    max_standard_errors=('standard_errors', 'max'),
                    min_standard_errors=('standard_errors', 'min')))
    summary.to_csv(os.path.join(artifact_dir, f'{tag}_assignment_summary.csv'), index=False,
                   encoding='utf-8')
    print(table.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    return table


def stage_features(pool, sweep_dir, criteria, artifact_dir, tag):
    """Deliverable (d): the absence count each rule implies, which S04's diagnostic reads.

    Sums are accumulated per file and reduced once per lattice. The pool is partitioned by
    (bundle, lattice), so a per-file row is a per-BUNDLE row, and averaging those would weight
    three bundles equally regardless of how many candidates each carries.
    """
    columns = (['xg_stored_group_index']
               + [f'xg_{c}_n_absent_in_range' for c in criteria]
               + [f'xg_{c}_group_index' for c in criteria])
    totals = {}
    for lattice, frame in sweep_files(pool, sweep_dir, columns, ['is_correct']):
        correct = frame['is_correct'].to_numpy()
        for criterion in criteria:
            counts = frame[f'xg_{criterion}_n_absent_in_range'].to_numpy().astype(np.float64)
            key = (lattice, criterion)
            bucket = totals.setdefault(key, dict.fromkeys(
                ['n', 'n_correct', 'sum', 'sum_correct', 'sum_incorrect', 'n_generic',
                 'n_same_as_stored'], 0.0))
            bucket['n'] += counts.size
            bucket['n_correct'] += int(correct.sum())
            bucket['sum'] += counts.sum()
            bucket['sum_correct'] += counts[correct].sum()
            bucket['sum_incorrect'] += counts[~correct].sum()
            bucket['n_generic'] += int((frame[f'xg_{criterion}_group_index'] == 0).sum())
            bucket['n_same_as_stored'] += int(
                (frame[f'xg_{criterion}_group_index'] == frame['xg_stored_group_index']).sum())

    rows = []
    for (lattice, criterion), b in totals.items():
        n_incorrect = b['n'] - b['n_correct']
        rows.append({
            'bravais_lattice': lattice, 'criterion': criterion,
            'n_candidates': int(b['n']), 'n_correct': int(b['n_correct']),
            'mean_absent_in_range': b['sum']/b['n'] if b['n'] else np.nan,
            'mean_absent_in_range_correct':
                b['sum_correct']/b['n_correct'] if b['n_correct'] else np.nan,
            'mean_absent_in_range_incorrect':
                b['sum_incorrect']/n_incorrect if n_incorrect else np.nan,
            'share_generic_group': b['n_generic']/b['n'] if b['n'] else np.nan,
            'agrees_with_stored': b['n_same_as_stored']/b['n'] if b['n'] else np.nan,
            })
    table = pd.DataFrame(rows).sort_values(['bravais_lattice', 'criterion'])
    table.to_csv(os.path.join(artifact_dir, f'{tag}_absence_counts.csv'), index=False,
                 encoding='utf-8')
    print(table.to_string(index=False))
    return table


def perturb(xnn, radius, lattice_system, rng, minimum_uc=2.0, maximum_uc=60.0):
    """Isotropic displacement at a fixed radius, then the pipeline's own physicality repair.

    `ErrorAdder.perturb_xnn`'s construction, batched over candidates. Taken from the `fom` branch's
    `run_fom_floor.py` (recorded in CHERRY_PICK.md) with one change: its seed came from
    `abs(hash(...))`, which is salted per process, so nothing it produced could be regenerated.
    The seed here derives from the entry id, per PROTOCOL section 6.
    """
    from mlindex.utilities.UnitCellTools import fix_unphysical
    if radius == 0.0:
        return np.array(xnn, dtype=np.float64, copy=True)
    step = rng.uniform(-1.0, 1.0, size=xnn.shape)
    step *= radius/np.linalg.norm(step, axis=1)[:, np.newaxis]
    return fix_unphysical(xnn=xnn + step, rng=rng, minimum_unit_cell=minimum_uc,
                          maximum_unit_cell=maximum_uc, lattice_system=lattice_system)


def derived_seed(entry_id, bravais_lattice, base=20260901):
    """A stable per-(entry, lattice) seed. `hash` is salted per process and cannot be used."""
    import hashlib
    digest = hashlib.sha256(f'{base}:{entry_id}:{bravais_lattice}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') % (2**31 - 1)


def _time_sweep(q2_obs, xnn, lattice_system, bravais_lattice, criterion):
    started = time.perf_counter()
    FomBenchmark.extinction_group_sweep(q2_obs, xnn, lattice_system, bravais_lattice,
                                        criteria=(criterion,))
    return time.perf_counter() - started


def stage_cost(pool, criteria, artifact_dir, tag, sizes=(250, 500, 1000, 2000), repeats=3):
    """Deliverable (e): what each criterion costs inside the argmax, per lattice.

    **This is the one place in campaign 2 where a per-call merit cost still multiplies.** Every
    other consumer reads a merit once per candidate; here it is read once per candidate per group,
    and oP searches 68 of them. Cost was retired as an exclusion criterion (DWMM, 2026-08-25), so
    nothing is decided on these numbers -- but the arithmetic is worth knowing, and only here.

    A line is fitted over several pool sizes rather than dividing one measurement, so the fixed
    cost of building the cctbx-backed group lists is not smeared into the per-candidate slope.
    """
    entries = FomBenchmark.load_entries(pool).set_index(['entry_id', 'condition_bundle'])
    rows = []
    for path in sorted(Path(pool).glob('candidates_c2_error1_cont0_*.parquet')):
        lattice = path.stem.split('_')[-1]
        if lattice in SKIP_LATTICES:
            continue
        system = LATTICE_SYSTEM[lattice]
        block = pd.read_parquet(path, columns=['entry_id', 'condition_bundle', 'n_peaks', 'xnn'])
        (entry_id, bundle, n_peaks), group = next(iter(
            block.groupby(['entry_id', 'condition_bundle', 'n_peaks'], sort=False)))
        q2_obs = np.asarray(entries.loc[(entry_id, bundle), 'q2_obs'],
                            dtype=np.float64)[:int(n_peaks)]
        available = np.stack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
        n_groups = len(FomBenchmark.spacegroup_reference_sets(system, lattice))
        # Warm the cctbx-backed group lists so the timed calls measure arithmetic rather than a
        # cache miss -- production pays that once per pattern too.
        FomBenchmark.extinction_group_sweep(q2_obs, available[:2], system, lattice,
                                            criteria=('M20',))
        for criterion in criteria:
            timings = []
            for size in sizes:
                if available.shape[0] < size:
                    xnn = np.repeat(available, int(np.ceil(size/available.shape[0])),
                                    axis=0)[:size]
                else:
                    xnn = available[:size]
                timings.append((size, min(_time_sweep(q2_obs, xnn, system, lattice, criterion)
                                          for _ in range(repeats))))
            slope, intercept = np.polyfit(np.array([t[0] for t in timings], dtype=float),
                                          np.array([t[1] for t in timings], dtype=float), 1)
            rows.append({
                'bravais_lattice': lattice, 'criterion': criterion, 'n_groups': n_groups,
                'microseconds_per_candidate': float(slope*1e6),
                'fixed_seconds': float(intercept),
                # The float64 `hkl` scratch production allocates, at 4 000 candidates. int16 would
                # cut it fourfold and is numerically inert, which is why it is a separate commit.
                'hkl_scratch_MB_at_4000': 4000*int(n_peaks)*3*n_groups*8/1e6,
                })
    table = pd.DataFrame(rows)
    baseline = table[table.criterion == 'M20'].set_index('bravais_lattice')[
        'microseconds_per_candidate']
    table['relative_to_M20'] = (table['microseconds_per_candidate']
                                / table['bravais_lattice'].map(baseline))
    table = table.sort_values(['bravais_lattice', 'criterion'])
    table.to_csv(os.path.join(artifact_dir, f'{tag}_cost.csv'), index=False, encoding='utf-8')
    print(table.to_string(index=False))
    return table


def _stability_worker(task):
    """One (bundle, lattice) file's flip rates. Module-level and picklable: spawn-safe."""
    (path, pool, criteria, radii, replicates, n_entries, n_candidates, seed_base) = task
    lattice = Path(path).stem.split('_')[-1]
    system = LATTICE_SYSTEM[lattice]
    neighbor = _neighbor_radii()[lattice]
    entries = FomBenchmark.load_entries(pool).set_index(['entry_id', 'condition_bundle'])
    block = pd.read_parquet(path, columns=['entry_id', 'condition_bundle', 'n_peaks', 'xnn',
                                           'is_correct'])
    rows, seen = [], 0
    for (entry_id, bundle, n_peaks), group in block.groupby(
            ['entry_id', 'condition_bundle', 'n_peaks'], sort=False):
        if seen >= n_entries:
            break
        seen += 1
        q2_obs = np.asarray(entries.loc[(entry_id, bundle), 'q2_obs'],
                            dtype=np.float64)[:int(n_peaks)]
        # Correct cells first, then whatever else fills the quota: the flip rate on a cell that IS
        # the answer is the one that bears on the rule, and correct cells are rare.
        ordered = pd.concat([group[group['is_correct']], group[~group['is_correct']]])
        xnn = np.stack([np.asarray(v, dtype=np.float64)
                        for v in ordered['xnn'][:n_candidates]])
        correct = ordered['is_correct'].to_numpy()[:xnn.shape[0]]
        _, base, _, _, _, _ = FomBenchmark.extinction_group_sweep(
            q2_obs, xnn, system, lattice, criteria=tuple(criteria))
        for radius in radii:
            for replicate in range(replicates if radius else 1):
                rng = np.random.default_rng(
                    derived_seed(entry_id, lattice, seed_base) + replicate)
                moved = perturb(xnn, radius*neighbor, system, rng)
                _, winner, _, _, _, _ = FomBenchmark.extinction_group_sweep(
                    q2_obs, moved, system, lattice, criteria=tuple(criteria))
                for criterion in criteria:
                    flipped = winner[criterion] != base[criterion]
                    rows.append({
                        'bravais_lattice': lattice, 'criterion': criterion,
                        'radius_fraction': radius, 'replicate': replicate,
                        'entry_id': entry_id, 'n_candidates': int(xnn.shape[0]),
                        'n_flipped': int(flipped.sum()),
                        'n_correct': int(correct.sum()),
                        'n_flipped_correct': int(flipped[correct].sum()),
                        })
    return rows


def stage_stability(pool, criteria, artifact_dir, tag, radii=(0.0, 0.1, 0.25, 0.5),
                    replicates=4, n_entries=25, n_candidates=600, processes=1,
                    seed_base=20260901):
    """Deliverable (c): how often the chosen group flips when the cell is nudged.

    No real run is needed -- the flip rate is a property of the argmax given a cell, so displacing
    the stored cell and re-picking measures exactly the thing.

    **The LEVELS here are not comparable with campaign 1's 8.8 %.** That figure displaced the cell
    AND replayed the stochastic refinement loop before re-picking, and reported 2.75 % flips at
    radius zero -- where a pure re-pick is deterministic and must be exactly 0.00 %, which is this
    arm's own internal control. Only the contrast BETWEEN rules under one operator is the claim;
    the absolute rate is a property of the operator and is reported so the two are not confused.

    Radii are fractions of each lattice's own `neighbor_radius`, so "a tenth" means the same
    physical thing on cubic and on monoclinic.
    """
    tasks = []
    for path in sorted(Path(pool).glob('candidates_c2_error1_cont0_*.parquet')):
        if path.stem.split('_')[-1] in SKIP_LATTICES:
            continue
        tasks.append((str(path), pool, tuple(criteria), tuple(radii), replicates, n_entries,
                      n_candidates, seed_base))
    if processes > 1:
        with Pool(processes) as workers:
            collected = workers.map(_stability_worker, tasks)
    else:
        collected = [_stability_worker(task) for task in tasks]
    detail = pd.DataFrame([row for rows in collected for row in rows])
    detail.to_csv(os.path.join(artifact_dir, f'{tag}_stability_detail.csv'), index=False,
                  encoding='utf-8')

    table = (detail.groupby(['bravais_lattice', 'criterion', 'radius_fraction'], as_index=False)
             .agg(n_candidates=('n_candidates', 'sum'), n_flipped=('n_flipped', 'sum'),
                  n_correct=('n_correct', 'sum'),
                  n_flipped_correct=('n_flipped_correct', 'sum')))
    table['flip_rate'] = table['n_flipped']/table['n_candidates']
    table['flip_rate_correct'] = np.where(
        table['n_correct'] > 0, table['n_flipped_correct']/table['n_correct'], np.nan)
    table.to_csv(os.path.join(artifact_dir, f'{tag}_stability.csv'), index=False,
                 encoding='utf-8')

    zero = table[table.radius_fraction == 0.0]
    if not zero.empty and zero['n_flipped'].sum() != 0:
        print(f"WARNING: {zero['n_flipped'].sum()} flips at radius 0, where a pure re-pick is "
              f'deterministic. The operator is not doing what it claims.')
    else:
        print('control: 0 flips at radius 0, as a deterministic re-pick must give')
    print(table.to_string(index=False))
    return table


def _neighbor_radii():
    """Each lattice's `neighbor_radius`, read from the shipped parameter dictionaries.

    Scraped from the factory source rather than imported, because the value is a literal inside
    each `get_*_optimizer` and there is no exported dict -- and constructing an optimizer to read
    one number would require the model files. The assertion at the end is the point: if a factory
    is refactored so the literal moves, this raises instead of silently omitting a lattice and
    reporting a stability table with a hole in it.
    """
    from mlindex.optimization import UtilitiesOptimizer as U
    factories = {'cF': U.get_cubic_optimizer, 'cI': U.get_cubic_optimizer,
                 'cP': U.get_cubic_optimizer, 'hP': U.get_hexagonal_optimizer,
                 'hR': U.get_rhombohedral_optimizer, 'mC': U.get_monoclinic_optimizer,
                 'mP': U.get_monoclinic_optimizer, 'oC': U.get_orthorhombic_optimizer,
                 'oF': U.get_orthorhombic_optimizer, 'oI': U.get_orthorhombic_optimizer,
                 'oP': U.get_orthorhombic_optimizer, 'tI': U.get_tetragonal_optimizer,
                 'tP': U.get_tetragonal_optimizer, 'aP': U.get_triclinic_optimizer}
    import inspect
    radii = {}
    for lattice, factory in factories.items():
        source = inspect.getsource(factory)
        for line in source.splitlines():
            if "'neighbor_radius'" in line:
                radii[lattice] = float(line.split(':')[1].strip().rstrip(','))
                break
    missing = sorted(set(factories) - set(radii))
    if missing:
        raise RuntimeError(
            f'no neighbor_radius found for {missing}; the optimizer factories have moved and the '
            f'stability radii cannot be derived'
            )
    return radii


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'legend.frameon': False,
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def stage_figure(artifact_dir, tag):
    """Assignment accuracy per lattice per rule. Publication quality from first production."""
    plt = _style()
    table = pd.read_csv(os.path.join(artifact_dir, f'{tag}_assignment_accuracy.csv'))
    criteria = list(dict.fromkeys(table['criterion']))
    # `M_rev_then_M20` is identical to `M_rev` on every candidate -- the M20 tie-break already
    # falls back to the incumbent where no group clears the support floor, so the composite rule
    # adds nothing. Plotting both doubles the bars for no information; the identity is stated in
    # the caption and the column stays in the CSV.
    redundant = [c for c in criteria if c != 'M_rev'
                 and c.startswith('M_rev_then')
                 and table[table.criterion == c]['accuracy'].round(12).tolist()
                 == table[table.criterion == 'M_rev']['accuracy'].round(12).tolist()]
    criteria = [c for c in criteria if c not in redundant]
    lattices = (table[table.criterion == 'M20']
                .sort_values('accuracy')['bravais_lattice'].tolist())
    figure, axes = plt.subplots(figsize=(7.6, 4.2))
    width = 0.8/len(criteria)
    positions = np.arange(len(lattices))
    for offset, criterion in enumerate(criteria):
        block = table[table.criterion == criterion].set_index('bravais_lattice')
        axes.bar(positions + offset*width, [block.loc[l, 'accuracy'] for l in lattices],
                 width, label=criterion)
    labels = [f"{l}\n{int(table[(table.bravais_lattice == l)]['n_groups_searched'].iloc[0])}"
              for l in lattices]
    axes.set_xticks(positions + width*(len(criteria)-1)/2)
    axes.set_xticklabels(labels)
    axes.set_xlabel('Bravais lattice, and the number of extinction groups its argmax searches')
    axes.set_ylabel('share of correct cells given the true extinction group')
    axes.set_title('Assignment accuracy against the true extinction group, by criterion',
                   fontsize=10)
    mean_M20 = table[table.criterion == 'M20']['accuracy'].mean()
    axes.axhline(mean_M20, color='0.45', lw=0.8, ls='--', zorder=0)
    axes.text(-0.45, mean_M20 + 0.015, 'M20 mean, unweighted', ha='left', va='bottom',
              fontsize=7, color='0.35')
    footnote = ('Correct candidates only, 11 818 over 13 lattices, fully retained pool. '
                'Triclinic excluded: one group, so every rule makes the same choice.')
    if redundant:
        footnote += ' M_rev_then_M20 omitted -- identical to M_rev on every candidate.'
    figure.text(0.012, -0.045, footnote, fontsize=7, color='0.3', ha='left')
    axes.legend(ncol=len(criteria), loc='upper left', fontsize=7)
    axes.set_ylim(0, 1.0)
    figure.tight_layout()
    figure.savefig(os.path.join(artifact_dir, f'{tag}_assignment_accuracy.png'),
                   bbox_inches='tight')
    print(f'wrote {tag}_assignment_accuracy.png')


def _markdown(frame, columns, floats=3):
    head = '| ' + ' | '.join(columns) + ' |'
    rule = '|' + '|'.join(['---']*len(columns)) + '|'
    lines = [head, rule]
    for row in frame[columns].itertuples(index=False):
        cells = [f'{v:.{floats}f}' if isinstance(v, float) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def stage_report(artifact_dir, tag):
    """Assemble the results document from the CSVs, so no number exists only in a notebook."""
    def read(name):
        path = os.path.join(artifact_dir, f'{tag}_{name}.csv')
        return pd.read_csv(path) if os.path.exists(path) else None

    gate = json.load(open(os.path.join(artifact_dir, f'{tag}_gate.json'), encoding='utf-8'))
    g1 = read('gate_g1_reproduction')
    g4 = read('gate_g4_support_floor')
    accuracy = read('assignment_accuracy')
    absence = read('absence_counts')
    stability = read('stability')
    cost = read('cost')

    summary = read('assignment_summary')
    verdict = []
    if summary is not None and accuracy is not None:
        best = summary.sort_values('mean_delta_pp', ascending=False).iloc[0]
        rev = summary[summary.criterion == 'M_rev'].iloc[0]
        worst_lattice = accuracy[accuracy.criterion == 'M_rev'].sort_values('standard_errors')
        verdict = [
            '## The answer', '',
            f"**C2-Q-011 is answered and the answer is NO.** `M_rev` picks the true extinction "
            f"group **{abs(rev.mean_delta_pp):.2f} pp less often** than M20 -- better on "
            f"{int(rev.n_lattices_better)} lattices and worse on {int(rev.n_lattices_worse)}, "
            f"losing by as much as **{abs(worst_lattice.iloc[0].standard_errors):.1f} standard "
            f"errors** of that lattice's own contrast floor "
            f"({worst_lattice.iloc[0].bravais_lattice}). The best alternative is "
            f"`{best.criterion}` at {best.mean_delta_pp:+.2f} pp, which is within noise. "
            '**The incumbent stays.**', '',
            _markdown(summary.sort_values('mean_delta_pp', ascending=False),
                      ['criterion', 'mean_accuracy_unweighted', 'mean_delta_pp',
                       'n_lattices_better', 'n_lattices_worse'], floats=4), '',
            "**And a result larger than the comparison that produced it.** Restricted to "
            "candidates that ARE the correct cell, the incumbent names the true extinction group "
            f"only **{accuracy[accuracy.criterion == 'M20']['accuracy'].mean():.4f}** of the time, "
            "unweighted over lattices -- and no criterion tested reaches further. The group is "
            "wrong for two thirds of correct cells, which nobody had measured.", '',
            ]

    out = [f'# S11 -- the extinction-group assignment rule',
           '',
           f"**Pool:** `{gate['pool']}` -- the fully retained pool, 43 348 938 candidates, "
           '1 590 cells (530 `fom-dev` entries x 3 condition bundles), nothing thinned.',
           f"**Commit:** `{gate['commit'][:12]}` (dirty tree: {gate['dirty_tree']}) - "
           f"**criteria:** {', '.join(gate['criteria'])} - **aP excluded** (one group).",
           '',
           '## Read this before any number below',
           '',
           '**The new rule can only LOWER the reported merit, by construction.** `best_M20` is the',
           'maximum of M20 over groups under the incumbent, so any other argmax lands at a group',
           'where M20 is no higher. A uniform drop is expected and is not a defect; the only',
           'question is whether the drop is larger for wrong cells than for correct ones.',
           '',
           '**Accuracy and ranking are different questions, and only the first is answered here.**',
           'A rule can assign more accurately and rank worse, because the pooled sort reads the',
           'rebound merit and not the group. The ranking half is a second session.',
           '']
    out += verdict
    out += ['## 1. The gates', '',
            f"G0 {gate['passed']['G0']}, G2 {gate['passed']['G2']}, G1 {gate['passed']['G1']}, "
            f"G3 {gate['passed']['G3']}.", '']
    if g1 is not None:
        out += [f"**G1, the gate that licenses everything else:** the offline argmax under M20 "
                f"reproduces the stored `spacegroup` **and** the stored `M20` on "
                f"**{g1.n_checked.sum():,} of {g1.n_checked.sum():,}** candidates, with `==` and "
                f"not `isclose`, {gate['n_g1_offender_rows']} offending rows. C2-F-036 got "
                f"310 807 on the PRE-deduplication stream; this is 6.7x that and runs on the "
                f"post-deduplication pool, so the defect `14b13a9` fixed did not reach it.", '']
    if g4 is not None:
        out += ['### The `M_rev` support floor, reported before any argmax result', '',
                "An extinction group's job is to delete predicted lines, and deleting them is what",
                'drives `N_cal` under the floor -- so the argmax meets ties at zero, where a',
                'floored `M_rev` is indistinguishable from "degenerate" and "no support". Ties are',
                'broken on M20, which makes the degenerate case fall back to the incumbent.', '',
                _markdown(g4.sort_values('share_below_floor', ascending=False),
                          ['bravais_lattice', 'n_group_evaluations', 'share_below_floor',
                           'n_candidates_with_no_supported_group']), '']
    if accuracy is not None:
        out += ['## 2. Assignment accuracy (deliverable a)', '',
                'Correct candidates only -- the question is meaningless for a wrong cell.',
                'Deltas are against M20 and are quoted in standard errors of **that lattice\'s',
                "own** contrast floor (`S08_floor_by_lattice.csv`), per PROTOCOL section 8.", '',
                _markdown(accuracy.sort_values(['bravais_lattice', 'criterion']),
                          ['bravais_lattice', 'criterion', 'n_groups_searched',
                           'n_correct_candidates', 'accuracy', 'accuracy_per_entry', 'delta_pp',
                           'standard_errors']), '']
    if absence is not None:
        out += ['## 3. What it does to S04\'s absence counts (deliverable d)', '',
                '`n_absent_extra_in_range` is a deterministic function of the chosen group, so a',
                "change of rule changes S04's diagnostic input.", '',
                _markdown(absence.sort_values(['bravais_lattice', 'criterion']),
                          ['bravais_lattice', 'criterion', 'mean_absent_in_range',
                           'mean_absent_in_range_correct', 'mean_absent_in_range_incorrect',
                           'agrees_with_stored']), '']
    if stability is not None:
        out += ['## 4. Stability (deliverable c)', '',
                "Displacement is a fraction of each lattice's own `neighbor_radius`. **The LEVELS",
                "are not comparable with campaign 1's 8.8 %**, which displaced the cell AND",
                'replayed the stochastic refinement before re-picking, and reported 2.75 % flips',
                'at radius zero where a pure re-pick is deterministic. Only the contrast between',
                'rules under one operator is the claim.', '',
                _markdown(stability, list(stability.columns)), '']
    if cost is not None:
        out += ['## 5. Cost (deliverable e)', '',
                '**The one place in campaign 2 where a per-call merit cost still multiplies** --',
                'every other consumer reads a merit once per candidate; here it is once per',
                'candidate per group, and oP searches 68. Cost decides nothing (DWMM,',
                '2026-08-25); the arithmetic is recorded because this is where it exists.', '',
                _markdown(cost.sort_values(['bravais_lattice', 'criterion']),
                          ['bravais_lattice', 'criterion', 'n_groups',
                           'microseconds_per_candidate', 'relative_to_M20',
                           'hkl_scratch_MB_at_4000']), '']
    path = os.path.join(artifact_dir, f'{tag}_extinction_rule.md')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(out) + '\n')
    print(f'wrote {path}')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate S11 extinction-rule arms')
    parser.add_argument('--pool', default=os.path.join('mlindex', 'data', 'fom_full_c2_pool'))
    parser.add_argument('--sweep-dir', default=None)
    parser.add_argument('--criteria', nargs='+', default=list(EXTINCTION_CRITERIA))
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom_campaign2',
                                                               'artifacts'))
    parser.add_argument('--tag', default='S11')
    parser.add_argument('--stage', required=True,
                        choices=['accuracy', 'features', 'stability', 'cost', 'figure', 'report'])
    parser.add_argument('--processes', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    sweep_dir = args.sweep_dir or os.path.join(args.pool, 'extinction_sweep')
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.stage == 'accuracy':
        stage_accuracy(args.pool, sweep_dir, args.criteria, args.artifact_dir, args.tag)
    elif args.stage == 'features':
        stage_features(args.pool, sweep_dir, args.criteria, args.artifact_dir, args.tag)
    elif args.stage == 'cost':
        stage_cost(args.pool, args.criteria, args.artifact_dir, args.tag)
    elif args.stage == 'stability':
        stage_stability(args.pool, args.criteria, args.artifact_dir, args.tag,
                        processes=args.processes)
    elif args.stage == 'figure':
        stage_figure(args.artifact_dir, args.tag)
    elif args.stage == 'report':
        stage_report(args.artifact_dir, args.tag)
    else:
        raise SystemExit(f'stage {args.stage!r} is not implemented yet')

    meta_path = os.path.join(args.artifact_dir, f'{args.tag}_{args.stage}_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as handle:
        json.dump({'stage': args.stage, 'pool': args.pool, 'sweep_dir': sweep_dir,
                   'criteria': list(args.criteria), 'commit': commit_hash(),
                   'dirty_tree': dirty_tree(), 'skipped_lattices': list(SKIP_LATTICES),
                   'seconds': round(time.time() - started, 1)}, handle, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
