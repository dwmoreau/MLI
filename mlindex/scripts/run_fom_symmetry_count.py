"""S04 -- symmetry as a count, not a label.

DWMM's hypothesis, 2026-08-24: what carries information about a candidate is not *which* extinction
group it was assigned but **how many systematic absences that group imposes**. For a random incorrect
cell, deleting systematically absent lines is unlikely to help; for a correct cell, choosing the
right group should. So within a Bravais lattice the group with the fewest additional absences should
show no benefit and the group with the most should show the largest.

That is falsifiable, it has a built-in negative control -- triclinic has exactly one extinction group,
so its count is identically zero and any apparent effect there is a harness bug -- and PROTOCOL
section 7 makes the gate that the prediction is TESTED, not that it holds. A clean "no" retires the
symmetry prior instead of re-encoding it, which is the cheaper and cleaner claim.

WHY THE QUANTITY IS ALREADY ON DISK. S03 persisted, per candidate, every merit at two points:

  B  the refined cell scored against the FULL reference list
  C  the same cell scored against the extinction group the pipeline chose for it

Every one of the fourteen lattices has exactly one group that removes no lines -- the generic group,
whose reference list IS `hkl_ref`. So point B is the merit at the generic group, and

  delta_merit_extinction = {merit}_C - {merit}_B

for all seven merits, M20 and M_rev included, without recomputing a single one.

WHICH POOL, AND WHY NOT THE BIG ONE. Benchmark A (`mlindex/data/fom_benchmark/`) has 5 922 entries
against the threshold-0 arms' 210 and 243, and it CANNOT carry this test. Every row in it survived
the M20 cut and top-20-per-lattice retention, so it is conditioned on final M20 -- which is `M20_C`,
the very quantity on the left of the subtraction. Selecting on `M20_C` while measuring
`M20_C - M20_B` conditions on the outcome, and the bias runs WITH the hypothesis rather than merely
against precision: among survivors a large extinction gain compensates for a poor generic-list fit,
and the censoring bites least on the groups imposing most absences, because those raise M20 most.
A confirming result there would be uninterpretable. So the ordering is measured on the uncensored
threshold-0 arms only, and Benchmark A is used for the support table and nothing else -- where it is
the right pool, being the one campaign 1's combiner actually consumed.

    python mlindex/scripts/run_fom_symmetry_count.py --stage lookup
    python mlindex/scripts/run_fom_symmetry_count.py --stage counts     --arm general
    python mlindex/scripts/run_fom_symmetry_count.py --stage gate       --arm general
    python mlindex/scripts/run_fom_symmetry_count.py --stage support
    python mlindex/scripts/run_fom_symmetry_count.py --stage diagnostic --arm general
    python mlindex/scripts/run_fom_symmetry_count.py --stage figures

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

# S03's harness, reused rather than reimplemented: the reference lists, the merit block, the
# bundle walk, the entry bootstrap, the figure style. PROTOCOL section 5 -- code goes next to what
# it resembles, and this resembles run_fom_prune_criterion closely enough to share its plumbing.
from mlindex.scripts.run_fom_prune_criterion import (
    ARMS, MERITS, MAX_BLOCK_ELEMENTS, PRODUCTION_PRUNE_THRESHOLD,
    load_hkl_ref, load_spacegroup_sets, merits_on_reference, bundle_directories, load_entries,
    commit_hash, _map, _style,
    )
from mlindex.utilities.ExtinctionCounts import (
    LATTICE_SYSTEM, LOOKUP_PATH, absent_in_range, build_absence_counts, build_group_masks,
    get_absence_counts, get_generic_group,
    )
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.numba_functions import fast_assign

ARTIFACT_DIR = os.path.join('docs', 'fom_campaign2', 'artifacts')
COUNT_ROOT = os.path.join('mlindex', 'data', 'fom_symmetry_counts')
BENCHMARK_A = os.path.join('mlindex', 'data', 'fom_benchmark')

# The two merits the handoff asks the ordering to be read under. M20 is what the assignment argmax
# actually optimises; M_rev is the one campaign 1 found measures line over-prediction directly, and
# if the ordering appears under it and not under M20 that is the S11 result arriving early.
DIAGNOSTIC_MERITS = ('M20', 'M_rev')

BOOTSTRAP_DRAWS = 2000

# Cubic is scored on 10 peaks against everything else's 20 and carries a 100-line reference list
# against monoclinic's 1 000, so its counts and its merits are both different statistics (R5).
CUBIC = ('cF', 'cI', 'cP')


# ---------------------------------------------------------------------------------------------
# stage: lookup -- the static per-(lattice, group) table
# ---------------------------------------------------------------------------------------------

def run_lookup(args):
    """Build the committed absence table and prove it against cctbx in the same pass.

    The table is a fixed property of the reference lists, so it is built once here and read back
    at inference by `get_absence_counts`, which never imports cctbx. Regenerating it is how the
    test suite proves the committed copy has not drifted from the lists it describes.
    """
    counts, provenance = {}, {}
    for bravais_lattice, lattice_system in LATTICE_SYSTEM.items():
        hkl_ref = load_hkl_ref(lattice_system, bravais_lattice)
        counts[bravais_lattice] = build_absence_counts(hkl_ref, bravais_lattice)
        masks = build_group_masks(hkl_ref, bravais_lattice)

        # The mask and the count are two routes to the same number; if they ever disagree the
        # in-range count would be silently wrong while the full count stayed right.
        for key, value in counts[bravais_lattice].items():
            if int((~masks[key]).sum()) != value:
                raise AssertionError(f'{bravais_lattice} {key}: mask and count disagree')

        zero = [key for key, value in counts[bravais_lattice].items() if value == 0]
        if len(zero) != 1:
            raise AssertionError(f'{bravais_lattice}: {len(zero)} groups remove no lines, want 1')
        provenance[bravais_lattice] = {
            'lattice_system': lattice_system,
            'n_reference_lines': int(hkl_ref.shape[0]),
            'n_groups': len(counts[bravais_lattice]),
            'generic_group': zero[0],
            'max_absent_extra': int(max(counts[bravais_lattice].values())),
            }

    payload = {
        'note': ('Lines each extinction group removes from its Bravais lattice generic reference '
                 'list. Built by run_fom_symmetry_count.py --stage lookup; read at inference by '
                 'ExtinctionCounts.get_absence_counts, which does not import cctbx.'),
        'commit': commit_hash(),
        'reference_lists': os.path.join('mlindex', 'models', '{system}_1', 'data',
                                        'hkl_ref_{lattice}.npy'),
        'provenance': provenance,
        'counts': counts,
        }
    with open(LOOKUP_PATH, 'w', encoding='utf-8') as _f:
        json.dump(payload, _f, indent=2, sort_keys=True, ensure_ascii=False)
        _f.write('\n')

    total = sum(len(value) for value in counts.values())
    print(f'wrote {LOOKUP_PATH}')
    print(f'{len(counts)} lattices, {total} extinction groups')
    for bravais_lattice, meta in provenance.items():
        print(f"  {bravais_lattice:3s} n_ref={meta['n_reference_lines']:5d} "
              f"groups={meta['n_groups']:3d} max_absent={meta['max_absent_extra']:4d}")


# ---------------------------------------------------------------------------------------------
# stage: counts -- the per-candidate absence counts, joined onto S03's persisted merits
# ---------------------------------------------------------------------------------------------

MERIT_ROOT = os.path.join('mlindex', 'data', 'fom_prune_criterion')

# What the merit shard carries. `spacegroup` is NOT among them -- S03's recompute dropped it -- so
# it is rejoined from the source shard the merits were computed from.
MERIT_COLUMNS = (['entry_id', 'bravais_lattice', 'lattice_system', 'candidate_id', 'n_peaks',
                  'm20_at_prune', 'is_correct', 'split', 'condition_bundle']
                 + [f'{merit}_{point}' for merit in DIAGNOSTIC_MERITS for point in ('B', 'C')])

SOURCE_COLUMNS = ('entry_id', 'bravais_lattice', 'candidate_id', 'xnn', 'spacegroup')

IDENTITY = ['entry_id', 'bravais_lattice', 'candidate_id']


def shard_pairs(arm):
    """[(merit shard, source shard, bundle)], matched by the stem S03 derived one from the other.

    `recompute_shard` wrote `merits_<stem>` from `predownsample_<stem>` row for row, so the two
    align positionally and `spacegroup` can be carried across without a key join. That alignment is
    asserted per shard rather than trusted -- a silent misalignment would attach every count to the
    wrong candidate and nothing downstream would look wrong.
    """
    root = ARMS[arm]['root']
    pairs = []
    for shard in sorted(Path(os.path.join(MERIT_ROOT, arm)).glob('merits_*.parquet')):
        stem = shard.stem[len('merits_'):]
        bundle = stem.split('_shard')[0]
        source = Path(root) / bundle / f'predownsample_{stem}.parquet'
        if not source.exists():
            raise FileNotFoundError(f'no source shard for {shard}: looked for {source}')
        pairs.append((str(shard), str(source), bundle))
    if not pairs:
        raise FileNotFoundError(f'no merit shards for arm {arm} under {MERIT_ROOT}')
    return pairs


def counts_for_shard(merit_path, source_path, bundle, entries):
    """Absence counts for every candidate in one shard.

    The in-range count is the one with a mechanism attached, and it needs the candidate's own cell:
    a group removes a fixed set of Miller indices, but whether those lines fall inside the merit's
    counting window depends on the cell that placed them. So this is one Q2Calculator pass over the
    FULL reference list per candidate block -- the narrowed list is a mask over it, not a second
    calculation -- plus the assignment that fixes the window.
    """
    frame = pd.read_parquet(merit_path, columns=list(MERIT_COLUMNS))
    source = pd.read_parquet(source_path, columns=list(SOURCE_COLUMNS))
    if frame.shape[0] != source.shape[0] or not frame[IDENTITY].equals(source[IDENTITY]):
        raise AssertionError(f'{merit_path} is not row-aligned with {source_path}')
    frame['spacegroup'] = source['spacegroup'].to_numpy()

    truth = entries.set_index('entry_id')
    n_rows = frame.shape[0]
    dropped_window = np.full(n_rows, -1, dtype=np.int64)
    dropped_observed = np.full(n_rows, -1, dtype=np.int64)
    reference_window = np.full(n_rows, -1, dtype=np.int64)
    position = np.arange(n_rows)

    for (bravais_lattice, lattice_system), lattice_rows in frame.groupby(
            ['bravais_lattice', 'lattice_system'], sort=False):
        hkl_ref = load_hkl_ref(lattice_system, bravais_lattice)
        masks = build_group_masks(hkl_ref, bravais_lattice)
        calculator = Q2Calculator(lattice_system=lattice_system, hkl=hkl_ref, tensorflow=False,
                                  representation='xnn')
        chunk = max(1, MAX_BLOCK_ELEMENTS // max(hkl_ref.shape[0], 1))

        for entry_id, group in lattice_rows.groupby('entry_id', sort=False):
            n_peaks = int(group['n_peaks'].iloc[0])
            q2_obs = np.asarray(truth.loc[entry_id, 'q2_obs'], dtype=np.float64)[:n_peaks]
            rows = position[frame.index.get_indexer(group.index)]
            xnn = np.stack(source['xnn'].to_numpy()[rows]).astype(np.float64)
            spacegroups = group['spacegroup'].to_numpy()

            for start in range(0, xnn.shape[0], chunk):
                stop = min(start + chunk, xnn.shape[0])
                block = rows[start:stop]
                q2_ref_calc = calculator.get_q2(xnn[start:stop])

                # The window get_M20 actually counts N over: strictly below the last ASSIGNED
                # calculated line, not below the last observed peak. `fast_assign` is the same
                # routine assign_extinction_group uses, so the window is the pipeline's own.
                q2_calc = np.take_along_axis(q2_ref_calc, fast_assign(q2_obs, q2_ref_calc), axis=1)
                cutoff = q2_calc[:, -1]
                observed = np.full(stop - start, q2_obs[-1], dtype=np.float64)

                for spacegroup in pd.unique(spacegroups[start:stop]):
                    local = np.flatnonzero(spacegroups[start:stop] == spacegroup)
                    keep = masks[spacegroup]
                    n_dropped, n_in_range = absent_in_range(
                        q2_ref_calc[local], keep, cutoff[local])
                    dropped_window[block[local]] = n_dropped
                    reference_window[block[local]] = n_in_range
                    dropped_observed[block[local]] = absent_in_range(
                        q2_ref_calc[local], keep, observed[local])[0]

    if (dropped_window < 0).any():
        raise AssertionError(f'{merit_path}: {(dropped_window < 0).sum()} rows left uncounted')

    lookup = {lattice: get_absence_counts(lattice) for lattice in frame['bravais_lattice'].unique()}
    frame['n_absent_extra'] = [lookup[lattice][group] for lattice, group
                               in zip(frame['bravais_lattice'], frame['spacegroup'])]
    frame['n_groups_searched'] = frame['bravais_lattice'].map(
        {lattice: len(value) for lattice, value in lookup.items()}).astype(np.int64)
    frame['n_absent_extra_in_range'] = dropped_window
    frame['n_absent_extra_in_range_obs'] = dropped_observed
    frame['n_ref_in_range'] = reference_window
    frame['f_absent_extra'] = np.where(reference_window > 0,
                                       dropped_window / np.maximum(reference_window, 1), np.nan)
    for merit in DIAGNOSTIC_MERITS:
        frame[f'delta_{merit}'] = (frame[f'{merit}_C'].to_numpy()
                                   - frame[f'{merit}_B'].to_numpy())
    return frame.drop(columns=['lattice_system'])


def _counts_worker(job):
    merit_path, source_path, bundle, bundle_dir, out_dir = job
    frame = counts_for_shard(merit_path, source_path, bundle, load_entries(bundle_dir))
    destination = Path(out_dir) / f'counts_{Path(merit_path).stem[len("merits_"):]}.parquet'
    frame.to_parquet(destination, index=False)

    # The negative control, asserted rather than reported: triclinic has one extinction group, so
    # it can remove nothing and the merit cannot move. Anything else is a harness bug.
    triclinic = frame.loc[frame['bravais_lattice'] == 'aP']
    violations = 0
    if triclinic.shape[0]:
        violations = int((triclinic['n_absent_extra'].to_numpy() != 0).sum()
                         + (~np.isclose(triclinic['delta_M20'].to_numpy(), 0.0,
                                        rtol=0, atol=1e-12)).sum())
    return {'shard': str(destination), 'rows': int(frame.shape[0]),
            'aP_rows': int(triclinic.shape[0]), 'aP_violations': violations}


def run_counts(args):
    out_dir = Path(COUNT_ROOT) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ARMS[args.arm]['root']
    jobs = [(merit, source, bundle, os.path.join(root, bundle), str(out_dir))
            for merit, source, bundle in shard_pairs(args.arm)]

    started = time.time()
    results = []
    for done, result in enumerate(_map(_counts_worker, jobs, args.processes), start=1):
        results.append(result)
        print(f"  [{done}/{len(jobs)}] {Path(result['shard']).name} {result['rows']:>9,} rows "
              f"({time.time() - started:.0f} s)")

    rows = sum(result['rows'] for result in results)
    aP_rows = sum(result['aP_rows'] for result in results)
    violations = sum(result['aP_violations'] for result in results)
    print(f'\narm {args.arm}: {rows:,} rows over {len(results)} shards -> {out_dir}')
    print(f'triclinic control: {aP_rows:,} aP rows, {violations} violations')
    if violations:
        raise AssertionError('triclinic is not the structural zero it must be by construction')
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as _f:
        json.dump({'arm': args.arm, 'commit': commit_hash(), 'rows': rows,
                   'aP_rows': aP_rows, 'note': ARMS[args.arm]['note'],
                   'shards': [result['shard'] for result in results]}, _f, indent=2)


GATE_LABELS = ('correct', 'incorrect')


def _gate_worker(job):
    """Re-run the extinction argmax on a sample and check it reproduces the stored winner.

    Two things come out of one per-group loop:

    AGREEMENT. The handoff warns that deduplication mis-attached spacegroups whenever a NaN cell
    was dropped, that the fix (14b13a9) post-dates this pool, and that how often it fired is not
    recoverable because the NaN count per (entry, lattice) was never stored. It is not recoverable,
    but it IS boundable: `assign_extinction_group` is deterministic given the cell, the peaks and
    the reference list, so re-running it says how often the stored label is the one the rule would
    pick. Nobody has computed this.

    THE SEARCH NULL. `delta_merit_extinction` is a selected maximum, so it is optimistically biased
    by an amount that grows with the number of alternatives searched -- and that number runs from 1
    (aP) to 68 (oP). Carrying the null measured on INCORRECT cells is what makes a gain on oP
    comparable to a gain on cP, and the handoff is explicit that the two travel together or neither
    does.
    """
    merit_path, source_path, bundle, bundle_dir, per_cell, seed = job
    frame = pd.read_parquet(merit_path, columns=list(MERIT_COLUMNS))
    source = pd.read_parquet(source_path, columns=list(SOURCE_COLUMNS))
    frame['spacegroup'] = source['spacegroup'].to_numpy()
    truth = load_entries(bundle_dir).set_index('entry_id')
    rng = np.random.default_rng(seed)

    frame['label'] = np.where(frame['is_correct'].to_numpy(), 'correct', 'incorrect')
    rows = []
    for (bravais_lattice, lattice_system, label), group in frame.groupby(
            ['bravais_lattice', 'lattice_system', 'label'], sort=False):
        take = min(group.shape[0], per_cell)
        chosen = group.iloc[rng.choice(group.shape[0], size=take, replace=False)]
        spacegroup_sets = load_spacegroup_sets(lattice_system, bravais_lattice)
        keys = list(spacegroup_sets.keys())
        generic = get_generic_group(bravais_lattice)

        for entry_id, block in chosen.groupby('entry_id', sort=False):
            n_peaks = int(block['n_peaks'].iloc[0])
            q2_obs = np.asarray(truth.loc[entry_id, 'q2_obs'], dtype=np.float64)[:n_peaks]
            xnn = np.stack(source['xnn'].to_numpy()[
                frame.index.get_indexer(block.index)]).astype(np.float64)

            per_group = {merit: np.full((xnn.shape[0], len(keys)), np.nan)
                         for merit in DIAGNOSTIC_MERITS}
            for index, key in enumerate(keys):
                lines = spacegroup_sets[key]
                chunk = max(1, MAX_BLOCK_ELEMENTS // max(lines.shape[0], 1))
                calculator = Q2Calculator(lattice_system=lattice_system, hkl=lines,
                                          tensorflow=False, representation='xnn')
                for start in range(0, xnn.shape[0], chunk):
                    stop = min(start + chunk, xnn.shape[0])
                    values = merits_on_reference(q2_obs, calculator.get_q2(xnn[start:stop]))
                    for merit in DIAGNOSTIC_MERITS:
                        per_group[merit][start:stop, index] = values[merit]

            # The rule picks by argmax of M20, so that is what reproduction is judged on.
            winner = np.take(keys, np.argmax(per_group['M20'], axis=1))
            generic_index = keys.index(generic)
            record = {
                'bravais_lattice': bravais_lattice, 'label': label, 'condition_bundle': bundle,
                'entry_id': entry_id, 'n_groups_searched': len(keys), 'n': int(xnn.shape[0]),
                'n_agree': int((winner == block['spacegroup'].to_numpy()).sum()),
                }
            for merit in DIAGNOSTIC_MERITS:
                gain = per_group[merit].max(axis=1) - per_group[merit][:, generic_index]
                record[f'null_mean_{merit}'] = float(gain.mean())
                record[f'null_median_{merit}'] = float(np.median(gain))
                record[f'null_p95_{merit}'] = float(np.percentile(gain, 95))
            rows.append(record)
    return pd.DataFrame(rows)


def run_gate(args):
    root = ARMS[args.arm]['root']
    jobs = [(merit, source, bundle, os.path.join(root, bundle), args.sample, args.seed + index)
            for index, (merit, source, bundle) in enumerate(shard_pairs(args.arm))]
    pieces = [piece for piece in _map(_gate_worker, jobs, args.processes) if piece.shape[0]]
    table = pd.concat(pieces, ignore_index=True)

    summary = (table.groupby(['bravais_lattice', 'label'], as_index=False)
               .agg(n_entries=('entry_id', 'nunique'), n=('n', 'sum'), n_agree=('n_agree', 'sum'),
                    n_groups_searched=('n_groups_searched', 'first'),
                    **{f'null_mean_{merit}': (f'null_mean_{merit}', 'mean')
                       for merit in DIAGNOSTIC_MERITS},
                    **{f'null_p95_{merit}': (f'null_p95_{merit}', 'mean')
                       for merit in DIAGNOSTIC_MERITS}))
    summary['frac_agree'] = summary['n_agree'] / summary['n']
    summary.insert(0, 'arm', args.arm)

    destination = os.path.join(args.artifact_dir, f'S04_argmax_gate_{args.arm}.csv')
    summary.sort_values(['label', 'bravais_lattice']).to_csv(destination, index=False)
    table.to_csv(os.path.join(args.artifact_dir,
                              f'S04_argmax_gate_per_entry_{args.arm}.csv'), index=False)

    print(f'argmax reproduction, arm {args.arm} -> {destination}\n')
    print(f"{'bl':4s} {'label':10s} {'n':>9s} {'agree':>8s} {'groups':>7s} "
          f"{'null_M20':>9s} {'null_M_rev':>11s}")
    for _, row in summary.sort_values(['label', 'bravais_lattice']).iterrows():
        print(f"{row['bravais_lattice']:4s} {row['label']:10s} {row['n']:>9,} "
              f"{row['frac_agree']:>8.4f} {row['n_groups_searched']:>7d} "
              f"{row['null_mean_M20']:>9.4f} {row['null_mean_M_rev']:>11.4f}")
    pooled = summary['n_agree'].sum() / summary['n'].sum()
    print(f"\npooled agreement {pooled:.6f} over {summary['n'].sum():,} sampled candidates")


def run_support(args):
    """What supports the categorical, and what would support the count that replaces it.

    Measured on Benchmark A, `fom-train` only. That pool is prune-censored and so cannot carry the
    ordering test -- see the module docstring -- but it is the RIGHT pool for this question, because
    it is the one campaign 1's combiner actually consumed and the one C2-F-003's support numbers
    were computed from. Reading it here is a like-for-like re-measurement, not a new claim about the
    uncensored population.

    `fom-test` is sealed until S15 and `fom-dev` is a design set to be spent deliberately, so both
    are dropped before a single candidate row is read.
    """
    entries = pd.read_parquet(os.path.join(BENCHMARK_A, 'entries.parquet'),
                              columns=['entry_id', 'split'])
    train = set(entries.loc[entries['split'] == 'fom-train', 'entry_id'].unique())
    print(f'Benchmark A: {len(train):,} fom-train source crystals '
          f'({entries["entry_id"].nunique():,} total; dev and test not read)')

    # The bundle survives only in the filename -- Benchmark A never wrote it as a column, which is
    # campaign 1 rebuild-register row R8. It is recovered here because C2-F-003 measured this
    # support on ONE bundle, and pooling all seven counts a crystal at a level it reaches under any
    # condition. Both views are reported so the two numbers are reconcilable rather than merely
    # different.
    shards = sorted(Path(BENCHMARK_A).glob('candidates_*.parquet'))
    pieces = []
    for shard in shards:
        frame = pd.read_parquet(shard, columns=['entry_id', 'bravais_lattice', 'spacegroup',
                                                'is_correct'])
        stem = shard.stem[len('candidates_'):]
        frame['condition_bundle'] = stem.rsplit('_', 1)[0]
        pieces.append(frame.loc[frame['entry_id'].isin(train)])
    pool = pd.concat(pieces, ignore_index=True)
    lookup = get_absence_counts()
    pool['n_absent_extra'] = [lookup[lattice][group] for lattice, group
                              in zip(pool['bravais_lattice'], pool['spacegroup'])]
    print(f'{pool.shape[0]:,} fom-train candidates over {len(shards)} shards')

    rows = []
    for encoding, key in (('spacegroup', ['spacegroup']),
                          ('n_absent_extra', ['bravais_lattice', 'n_absent_extra'])):
        grouped = pool.groupby(key, sort=False)
        table = grouped.agg(n_candidates=('is_correct', 'size'),
                            n_positive=('is_correct', 'sum'),
                            n_crystals=('entry_id', 'nunique')).reset_index()
        positives = (pool.loc[pool['is_correct']].groupby(key, sort=False)['entry_id'].nunique()
                     .rename('n_crystals_positive').reset_index())
        table = table.merge(positives, on=key, how='left').fillna({'n_crystals_positive': 0})
        table.insert(0, 'encoding', encoding)
        rows.append(table)

        crystals = table['n_crystals'].to_numpy()
        print(f'\n{encoding}: {table.shape[0]} levels; '
              f'{int((table["n_positive"] == 0).sum())} with zero positives')
        print(f'  supporting crystals  min {crystals.min()} / p10 '
              f'{np.percentile(crystals, 10):.0f} / median {np.median(crystals):.0f} / p90 '
              f'{np.percentile(crystals, 90):.0f} / max {crystals.max()}')
        print(f'  levels under 30 crystals: {int((crystals < 30).sum())} of {table.shape[0]}')
        share = table.loc[~table.get("bravais_lattice", pd.Series("", index=table.index)).eq("x")]
        biggest = table.nlargest(1, 'n_candidates').iloc[0]
        print(f'  largest level carries {biggest["n_candidates"] / pool.shape[0]:.1%} '
              f'of the pool')

    support = pd.concat(rows, ignore_index=True)
    destination = os.path.join(args.artifact_dir, 'S04_absence_support.csv')
    support.to_csv(destination, index=False)

    # C2-F-003's view: one bundle at a time. The six condition bundles are the same crystals under
    # six condition draws, so they multiply rows and not support -- but they DO move which level a
    # crystal's correct candidate lands in, so the pooled count is legitimately higher than any
    # single bundle's. Reporting the per-bundle spread is what makes the two reconcilable.
    per_bundle = []
    for bundle, group in pool.groupby('condition_bundle', sort=True):
        table = (group.loc[group['is_correct']].groupby('spacegroup')['entry_id'].nunique()
                 .reindex(pool['spacegroup'].unique()).fillna(0))
        values = table.to_numpy()
        per_bundle.append({'condition_bundle': bundle, 'n_levels': int(values.size),
                           'levels_zero_positive': int((values == 0).sum()),
                           'crystals_min': float(values.min()),
                           'crystals_p10': float(np.percentile(values, 10)),
                           'crystals_median': float(np.median(values)),
                           'crystals_p90': float(np.percentile(values, 90)),
                           'crystals_max': float(values.max()),
                           'levels_under_30': int((values < 30).sum())})
    per_bundle = pd.DataFrame(per_bundle)
    per_bundle.to_csv(os.path.join(args.artifact_dir,
                                   'S04_absence_support_per_bundle.csv'), index=False)
    print('\nsupporting crystals per spacegroup level, POSITIVES only, one bundle at a time')
    print(f"  {'bundle':22s} {'zero':>5s} {'min':>5s} {'p10':>6s} {'med':>6s} {'p90':>6s} "
          f"{'max':>6s} {'<30':>5s}")
    for _, row in per_bundle.iterrows():
        print(f"  {row['condition_bundle']:22s} {row['levels_zero_positive']:>5.0f} "
              f"{row['crystals_min']:>5.0f} {row['crystals_p10']:>6.0f} "
              f"{row['crystals_median']:>6.0f} {row['crystals_p90']:>6.0f} "
              f"{row['crystals_max']:>6.0f} {row['levels_under_30']:>5.0f}")
    pooled = support.loc[support['encoding'] == 'spacegroup', 'n_crystals_positive'].to_numpy()
    print(f'  {"POOLED over 7 bundles":22s} {int((pooled == 0).sum()):>5d} '
          f'{pooled.min():>5.0f} {np.percentile(pooled, 10):>6.0f} {np.median(pooled):>6.0f} '
          f'{np.percentile(pooled, 90):>6.0f} {pooled.max():>6.0f} '
          f'{int((pooled < 30).sum()):>5d}')

    # The in-range count is a property of the candidate's own cell, so its support is read from the
    # uncensored arm where it was computed rather than re-derived on a censored one.
    in_range = []
    for arm in ('general', 'hard'):
        if not Path(os.path.join(COUNT_ROOT, arm)).exists():
            continue
        for shard in count_shards(arm):
            frame = pd.read_parquet(shard, columns=['entry_id', 'bravais_lattice', 'split',
                                                    'n_absent_extra_in_range', 'is_correct'])
            in_range.append(frame.loc[frame['split'] == 'fom-train'])
    if in_range:
        frame = pd.concat(in_range, ignore_index=True)
        table = (frame.groupby(['bravais_lattice', 'n_absent_extra_in_range'], sort=False)
                 .agg(n_candidates=('is_correct', 'size'), n_positive=('is_correct', 'sum'),
                      n_crystals=('entry_id', 'nunique')).reset_index())
        table.insert(0, 'encoding', 'n_absent_extra_in_range')
        table.to_csv(os.path.join(args.artifact_dir,
                                  'S04_absence_support_in_range.csv'), index=False)
        print(f'\nn_absent_extra_in_range (threshold-0 arms, fom-train): '
              f'{table.shape[0]} (lattice, value) cells, '
              f'{table["n_absent_extra_in_range"].nunique()} distinct values')
        print(f'  cells under 30 crystals: '
              f'{int((table["n_crystals"] < 30).sum())} of {table.shape[0]}')
    print(f'\nwrote {destination}')




# Candidates carried into a bootstrap draw, per (lattice, label). The POINT estimate always uses
# every candidate; this caps only the resampling, whose cost is a fresh rank sort per draw.
BOOTSTRAP_CANDIDATE_CAP = 50000
BOOTSTRAP_RESAMPLES = 500

DIAGNOSTIC_COLUMNS = ['entry_id', 'bravais_lattice', 'condition_bundle', 'split', 'is_correct',
                      'n_absent_extra', 'n_absent_extra_in_range', 'n_absent_extra_in_range_obs',
                      'n_ref_in_range', 'n_groups_searched'] + [f'delta_{m}'
                                                                for m in DIAGNOSTIC_MERITS]


def count_shards(arm):
    shards = sorted(Path(os.path.join(COUNT_ROOT, arm)).glob('counts_*.parquet'))
    if not shards:
        raise FileNotFoundError(f'no count shards for arm {arm}; run --stage counts first')
    return shards


def load_lattice(arm, bravais_lattice):
    """Every candidate of one Bravais lattice, across shards, with only the diagnostic columns.

    Read a lattice at a time rather than a shard at a time: the hard arm is 57 million rows and the
    statistic is within-lattice throughout, so nothing is ever needed across lattices at once.
    """
    frames = [pd.read_parquet(shard, columns=DIAGNOSTIC_COLUMNS,
                              filters=[('bravais_lattice', '==', bravais_lattice)])
              for shard in count_shards(arm)]
    frame = pd.concat(frames, ignore_index=True)
    frame['label'] = np.where(frame['is_correct'].to_numpy(), 'correct', 'incorrect')
    return frame


def _spearman(x, y):
    """Rank correlation, nan when either side has no spread. scipy is imported where it is used."""
    from scipy.stats import spearmanr
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _clustered_spearman(frame, count_column, delta_column, rng, bootstrap=True):
    """Point estimate on every candidate; interval by resampling SOURCE CRYSTALS, not candidates.

    One crystal searched once is one draw, and the several thousand candidates it contributes are
    one search of one pattern rather than thousands of independent pieces of evidence
    (PROTOCOL section 8). Campaign 1's candidate-level intervals were too narrow for exactly this
    reason, so the resampling unit here is `entry_id` -- the crystal -- and every candidate it
    contributed travels with it.

    The pool is thinned to `BOOTSTRAP_CANDIDATE_CAP` ONCE, before the draws, rather than inside
    each of them. Thinning inside the loop means materialising every resampled candidate before
    discarding most of them, which on a million-row lattice is a hundred-fold more copying than the
    statistic needs. Thinning first keeps whole entries intact -- the clustering is what matters,
    not the candidate count -- and the point estimate above still sees every row.
    """
    counts = frame[count_column].to_numpy()
    deltas = frame[delta_column].to_numpy()
    entry_ids = frame['entry_id'].to_numpy()
    if not bootstrap:
        return (_spearman(counts, deltas), np.nan, np.nan,
                int(pd.unique(entry_ids).size))

    # Thin BEFORE the point estimate, not after. Estimating the point on every candidate and the
    # interval on a subsample means the two describe different samples, and the interval can then
    # fail to bracket its own point -- which is not a presentation problem, it is two statistics
    # being quoted as one. The binding constraint here is the number of CRYSTALS, not the number of
    # candidates, so a thinned sample costs almost no precision.
    if counts.size > BOOTSTRAP_CANDIDATE_CAP:
        thinned = rng.choice(counts.size, size=BOOTSTRAP_CANDIDATE_CAP, replace=False)
        counts, deltas, entry_ids = counts[thinned], deltas[thinned], entry_ids[thinned]

    point = _spearman(counts, deltas)
    if not np.isfinite(point):
        return point, np.nan, np.nan, int(pd.unique(entry_ids).size)

    codes, entries = pd.factorize(entry_ids)
    order = np.argsort(codes, kind='stable')
    boundaries = np.searchsorted(codes[order], np.arange(entries.size + 1))
    blocks = [order[boundaries[index]:boundaries[index + 1]] for index in range(entries.size)]

    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = np.concatenate([blocks[index]
                                 for index in rng.integers(0, entries.size, entries.size)])
        value = _spearman(counts[picked], deltas[picked])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return point, np.nan, np.nan, entries.size
    return (point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)),
            entries.size)


def _paired_contrast(frame, count_column, delta_column, rng):
    """rho(correct) - rho(incorrect), with an interval that resamples crystals ONCE for both.

    This is the test statistic. A slope on correct cells alone proves nothing: an argmax over k
    alternatives gains something on any cell, and under M20 deleting an in-range line mechanically
    lowers N and so raises the merit whether or not the cell is right. What the hypothesis predicts
    is a DIFFERENCE -- rising for correct cells, flat for incorrect ones -- so the difference is
    what carries the interval.

    The two labels share entries, so the resample is paired: one draw of crystals, both rhos read
    off that draw. Bootstrapping the labels independently would ignore the correlation between them
    and give an interval of the wrong width.
    """
    labels = {}
    for label in ('correct', 'incorrect'):
        piece = frame[frame['label'] == label]
        counts = piece[count_column].to_numpy()
        deltas = piece[delta_column].to_numpy()
        entry_ids = piece['entry_id'].to_numpy()
        if counts.size > BOOTSTRAP_CANDIDATE_CAP:
            thinned = rng.choice(counts.size, size=BOOTSTRAP_CANDIDATE_CAP, replace=False)
            counts, deltas, entry_ids = counts[thinned], deltas[thinned], entry_ids[thinned]
        labels[label] = (counts, deltas, entry_ids)

    point = (_spearman(*labels['correct'][:2]) - _spearman(*labels['incorrect'][:2]))
    entries = pd.unique(np.concatenate([value[2] for value in labels.values()]))
    if entries.size < 3 or not np.isfinite(point):
        return point, np.nan, np.nan

    blocks = {}
    for label, (_, _, entry_ids) in labels.items():
        codes = pd.Categorical(entry_ids, categories=entries).codes
        order = np.argsort(codes, kind='stable')
        boundaries = np.searchsorted(codes[order], np.arange(entries.size + 1))
        blocks[label] = [order[boundaries[i]:boundaries[i + 1]] for i in range(entries.size)]

    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = rng.integers(0, entries.size, entries.size)
        values = []
        for label in ('correct', 'incorrect'):
            counts, deltas, _ = labels[label]
            index = np.concatenate([blocks[label][i] for i in picked])
            values.append(_spearman(counts[index], deltas[index]))
        if all(np.isfinite(value) for value in values):
            draws.append(values[0] - values[1])
    if not draws:
        return point, np.nan, np.nan
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _binned(frame, count_column, delta_column, n_bins=8):
    """Mean gain by absence-count bin -- what the figure draws, and what a reader checks by eye."""
    counts = frame[count_column].to_numpy()
    distinct = np.unique(counts)
    if distinct.size <= n_bins:
        assignment = np.searchsorted(distinct, counts)
        edges = distinct
    else:
        quantiles = np.unique(np.quantile(counts, np.linspace(0, 1, n_bins + 1)))
        assignment = np.clip(np.searchsorted(quantiles, counts, side='right') - 1,
                             0, quantiles.size - 2)
        edges = quantiles
    rows = []
    for index in np.unique(assignment):
        mask = assignment == index
        deltas = frame[delta_column].to_numpy()[mask]
        rows.append({'bin': int(index), 'bin_low': float(edges[min(index, edges.size - 1)]),
                     'n': int(mask.sum()),
                     'n_entries': int(frame['entry_id'].to_numpy()[mask].size and
                                      pd.unique(frame['entry_id'].to_numpy()[mask]).size),
                     'count_mean': float(counts[mask].mean()),
                     'delta_mean': float(deltas.mean()),
                     'delta_median': float(np.median(deltas)),
                     'delta_se': float(deltas.std(ddof=1) / np.sqrt(mask.sum()))
                     if mask.sum() > 1 else np.nan})
    return pd.DataFrame(rows)


def run_diagnostic(args):
    rng = np.random.default_rng(args.seed)
    lattices = sorted(pd.read_parquet(count_shards(args.arm)[0],
                                      columns=['bravais_lattice'])['bravais_lattice'].unique())
    ordering, binned, contrasts = [], [], []

    for bravais_lattice in lattices:
        frame = load_lattice(args.arm, bravais_lattice)

        # The negative control, asserted. aP has one extinction group, so it can remove nothing and
        # the merit cannot move. An effect here is a bug in the harness, not a result.
        if bravais_lattice == 'aP':
            for merit in DIAGNOSTIC_MERITS:
                moved = int((~np.isclose(frame[f'delta_{merit}'].to_numpy(), 0.0,
                                         rtol=0, atol=1e-12)).sum())
                if moved or int((frame['n_absent_extra'].to_numpy() != 0).sum()):
                    raise AssertionError(f'triclinic moved under {merit} on {moved} rows')
            print(f'aP control: {frame.shape[0]:,} rows, delta identically zero under '
                  f"{' and '.join(DIAGNOSTIC_MERITS)}")

        for label, group in frame.groupby('label', sort=False):
            for merit in DIAGNOSTIC_MERITS:
                for count_column in ('n_absent_extra_in_range', 'n_absent_extra',
                                     'n_absent_extra_in_range_obs'):
                    point, low, high, n_entries = _clustered_spearman(
                        group, count_column, f'delta_{merit}', rng,
                        bootstrap=count_column == 'n_absent_extra_in_range')
                    ordering.append({
                        'arm': args.arm, 'bravais_lattice': bravais_lattice, 'label': label,
                        'merit': merit, 'count': count_column, 'n_candidates': int(group.shape[0]),
                        'n_entries': int(n_entries),
                        'n_bootstrap_candidates': min(int(group.shape[0]),
                                                      BOOTSTRAP_CANDIDATE_CAP),
                        'n_groups_searched': int(group['n_groups_searched'].iloc[0]),
                        'spearman': point, 'ci_low': low, 'ci_high': high,
                        'delta_mean': float(group[f'delta_{merit}'].mean()),
                        'delta_median': float(group[f'delta_{merit}'].median()),
                        })
                table = _binned(group, 'n_absent_extra_in_range', f'delta_{merit}')
                table.insert(0, 'merit', merit)
                table.insert(0, 'label', label)
                table.insert(0, 'bravais_lattice', bravais_lattice)
                table.insert(0, 'arm', args.arm)
                binned.append(table)
        for merit in DIAGNOSTIC_MERITS:
            point, low, high = _paired_contrast(
                frame, 'n_absent_extra_in_range', f'delta_{merit}', rng)
            contrasts.append({
                'arm': args.arm, 'bravais_lattice': bravais_lattice, 'merit': merit,
                'n_groups_searched': int(frame['n_groups_searched'].iloc[0]),
                'n_correct': int((frame['label'] == 'correct').sum()),
                'n_incorrect': int((frame['label'] == 'incorrect').sum()),
                'n_entries': int(pd.unique(frame['entry_id'].to_numpy()).size),
                'contrast': point, 'ci_low': low, 'ci_high': high,
                'excludes_zero': bool(np.isfinite(low) and np.isfinite(high)
                                      and (low > 0 or high < 0)),
                })
        print(f'  {bravais_lattice}: {frame.shape[0]:,} candidates, '
              f"{int((frame['label'] == 'correct').sum()):,} correct")

    ordering = pd.DataFrame(ordering)
    binned = pd.concat(binned, ignore_index=True)
    ordering.to_csv(os.path.join(args.artifact_dir,
                                 f'S04_absence_counts_{args.arm}.csv'), index=False)
    binned.to_csv(os.path.join(args.artifact_dir,
                               f'S04_absence_binned_{args.arm}.csv'), index=False)

    # The test statistic is the CONTRAST. A slope present in both labels is a property of the
    # search -- an argmax over k alternatives gains something on any cell -- not of correctness.
    primary = ordering[ordering['count'] == 'n_absent_extra_in_range']
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(os.path.join(args.artifact_dir,
                                 f'S04_absence_contrast_{args.arm}.csv'), index=False)

    print(f'\nordering, arm {args.arm}, count = n_absent_extra_in_range')
    for merit in DIAGNOSTIC_MERITS:
        print(f'\n  merit {merit}')
        print(f"  {'bl':4s} {'grp':>4s} {'rho_correct':>22s} {'rho_incorrect':>22s} "
              f"{'contrast':>9s}")
        for bravais_lattice in lattices:
            rows = primary[(primary['bravais_lattice'] == bravais_lattice)
                           & (primary['merit'] == merit)].set_index('label')
            if 'correct' not in rows.index or 'incorrect' not in rows.index:
                continue
            good, bad = rows.loc['correct'], rows.loc['incorrect']
            print(f"  {bravais_lattice:4s} {int(good['n_groups_searched']):>4d} "
                  f"{good['spearman']:>8.3f} [{good['ci_low']:>6.3f},{good['ci_high']:>6.3f}] "
                  f"{bad['spearman']:>8.3f} [{bad['ci_low']:>6.3f},{bad['ci_high']:>6.3f}] "
                  f"{good['spearman'] - bad['spearman']:>9.3f}")
        print('\n    paired contrast rho(correct) - rho(incorrect), 95 % CI over crystals')
        for row in (r for r in contrasts if r['merit'] == merit):
            flag = '  *' if row['excludes_zero'] else ''
            print(f"    {row['bravais_lattice']:4s} {row['contrast']:>8.3f} "
                  f"[{row['ci_low']:>7.3f},{row['ci_high']:>7.3f}]{flag}")




def run_magnitude(args):
    """How BIG the extinction gain is, by label -- the quantity the ordering statistic hides.

    The rank correlation asks whether the gain rises with the absence count. It can be high for
    both labels at once, and is: deleting an in-range line lowers `get_M20`'s N, which raises the
    merit arithmetically whether or not the cell is right. What separates the labels is the SIZE of
    the gain, so it is reported beside the ordering rather than left in the figure.

    Two views, because the absolute one is partly a scale effect: correct cells sit at a higher
    merit to begin with, so a proportional gain is a larger absolute gain. The relative column
    divides it out. Rows where the baseline is zero are excluded from the relative mean only --
    `get_M20` returns exactly zero when its N is zero, which is a guard rather than a measurement.
    """
    rows = []
    for arm in ('general', 'hard'):
        if not Path(os.path.join(COUNT_ROOT, arm)).exists():
            continue
        lattices = sorted(pd.read_parquet(count_shards(arm)[0], columns=['bravais_lattice'])
                          ['bravais_lattice'].unique())
        for bravais_lattice in lattices:
            columns = ['entry_id', 'bravais_lattice', 'is_correct', 'n_groups_searched'] + [
                f'{merit}_{point}' for merit in DIAGNOSTIC_MERITS for point in ('B', 'C')]
            frame = pd.concat(
                [pd.read_parquet(shard, columns=columns,
                                 filters=[('bravais_lattice', '==', bravais_lattice)])
                 for shard in count_shards(arm)], ignore_index=True)
            for label, mask in (('correct', frame['is_correct'].to_numpy()),
                                ('incorrect', ~frame['is_correct'].to_numpy())):
                if not mask.any():
                    continue
                record = {'arm': arm, 'bravais_lattice': bravais_lattice, 'label': label,
                          'n': int(mask.sum()),
                          'n_entries': int(pd.unique(frame['entry_id'].to_numpy()[mask]).size),
                          'n_groups_searched': int(frame['n_groups_searched'].iloc[0])}
                for merit in DIAGNOSTIC_MERITS:
                    base = frame[f'{merit}_B'].to_numpy()[mask]
                    gain = frame[f'{merit}_C'].to_numpy()[mask] - base
                    usable = base > 1e-9

                    # M_rev blows up on a handful of degenerate cells -- one candidate in 1.45
                    # million carries M_rev = 4e11 -- and a single such row moves a mean over a
                    # million rows by eight orders of magnitude. So the headline statistic is
                    # winsorised at the 99.9th percentile and the raw mean is kept beside it, with
                    # the tail itself reported rather than quietly trimmed away. M20 is bounded and
                    # is unaffected either way.
                    value = frame[f'{merit}_C'].to_numpy()[mask]
                    cap = np.percentile(base, 99.9) if base.size else np.nan
                    cap_value = np.percentile(value, 99.9) if value.size else np.nan
                    # Both ends are capped. The blowup is not confined to the baseline: a cell can
                    # carry a modest M_rev against the full list and a catastrophic one against a
                    # narrowed list, so capping only B leaves the gain unbounded.
                    keep = (base <= cap) & (value <= cap_value)
                    record[f'{merit}_baseline_mean'] = float(base.mean())
                    record[f'{merit}_baseline_median'] = float(np.median(base))
                    record[f'{merit}_baseline_p99_9'] = float(cap)
                    record[f'{merit}_baseline_max'] = float(base.max())
                    record[f'{merit}_gain_mean_raw'] = float(gain.mean())
                    record[f'{merit}_gain_mean'] = float(gain[keep].mean()) if keep.any() else np.nan
                    record[f'{merit}_gain_median'] = float(np.median(gain))
                    record[f'{merit}_frac_moved'] = float(np.mean(np.abs(gain) > 1e-12))
                    trimmed = usable & keep
                    record[f'{merit}_gain_relative_mean'] = (
                        float(np.mean(gain[trimmed] / base[trimmed])) if trimmed.any() else np.nan)
                    record[f'{merit}_zero_baseline_frac'] = float(np.mean(~usable))
                    record[f'{merit}_value_p99_9'] = float(cap_value)
                    record[f'{merit}_value_max'] = float(value.max())
                    record[f'{merit}_n_above_1e3'] = int(((base > 1e3) | (value > 1e3)).sum())
                rows.append(record)

    table = pd.DataFrame(rows)
    destination = os.path.join(args.artifact_dir, 'S04_absence_magnitude.csv')
    table.to_csv(destination, index=False)

    for merit in DIAGNOSTIC_MERITS:
        print(f'\n=== {merit}: gain from assigning an extinction group '
                  f'(mean winsorised at the 99.9th pct of the baseline) ===')
        print(f"  {'arm':8s} {'bl':4s} {'grp':>4s} {'correct':>10s} {'incorrect':>10s} "
              f"{'ratio':>8s} {'rel_cor':>8s} {'rel_inc':>8s} {'rel_ratio':>9s}")
        for _, group in table.groupby(['arm', 'bravais_lattice'], sort=False):
            wide = group.set_index('label')
            if 'correct' not in wide.index or 'incorrect' not in wide.index:
                continue
            good, bad = wide.loc['correct'], wide.loc['incorrect']
            ratio = (good[f'{merit}_gain_mean'] / bad[f'{merit}_gain_mean']
                     if bad[f'{merit}_gain_mean'] else np.nan)
            rel_ratio = (good[f'{merit}_gain_relative_mean'] / bad[f'{merit}_gain_relative_mean']
                         if bad[f'{merit}_gain_relative_mean'] else np.nan)
            print(f"  {good['arm']:8s} {good['bravais_lattice']:4s} "
                  f"{int(good['n_groups_searched']):>4d} {good[f'{merit}_gain_mean']:>10.4f} "
                  f"{bad[f'{merit}_gain_mean']:>10.4f} {ratio:>8.1f} "
                  f"{good[f'{merit}_gain_relative_mean']:>8.4f} "
                  f"{bad[f'{merit}_gain_relative_mean']:>8.4f} {rel_ratio:>9.1f}"
                  f"{'   <- tail' if bad[f'{merit}_n_above_1e3'] else ''}")
    print(f'\nwrote {destination}')


def _lattice_order(table):
    """Lattices ordered by how many extinction groups the argmax searched, aP first.

    That is the axis the hypothesis is stated on -- fewest alternatives to most -- so it is the
    order the panels run in, and it puts the structural zero at the start where a reader meets the
    control before the claim.
    """
    order = (table.groupby('bravais_lattice')['n_groups_searched'].first()
             .sort_values(kind='stable'))
    return list(order.index)


def run_figures(args):
    plt = _style()
    from matplotlib.gridspec import GridSpec

    for arm in ('general', 'hard'):
        binned_path = os.path.join(args.artifact_dir, f'S04_absence_binned_{arm}.csv')
        ordering_path = os.path.join(args.artifact_dir, f'S04_absence_counts_{arm}.csv')
        if not (os.path.exists(binned_path) and os.path.exists(ordering_path)):
            print(f'skipping {arm}: run --stage diagnostic --arm {arm} first')
            continue
        binned = pd.read_csv(binned_path)
        ordering = pd.read_csv(ordering_path)
        primary = ordering[ordering['count'] == 'n_absent_extra_in_range']
        lattices = _lattice_order(primary)

        for merit in DIAGNOSTIC_MERITS:
            columns = 4
            rows = int(np.ceil(len(lattices) / columns))
            figure = plt.figure(figsize=(7.2, 1.75 * rows + 1.5))
            grid = GridSpec(rows + 1, columns, figure=figure,
                            height_ratios=[1] * rows + [1.35], hspace=0.55, wspace=0.32)

            for index, bravais_lattice in enumerate(lattices):
                axes = figure.add_subplot(grid[index // columns, index % columns])
                n_groups = int(primary.loc[primary['bravais_lattice'] == bravais_lattice,
                                           'n_groups_searched'].iloc[0])
                for label, colour, marker in (('incorrect', '#9aa0a6', 'o'),
                                              ('correct', '#1f4e79', 's')):
                    piece = binned[(binned['bravais_lattice'] == bravais_lattice)
                                   & (binned['label'] == label)
                                   & (binned['merit'] == merit)].sort_values('count_mean')
                    if piece.empty:
                        continue
                    axes.errorbar(piece['count_mean'], piece['delta_mean'],
                                  yerr=piece['delta_se'], color=colour, marker=marker,
                                  markersize=2.6, capsize=1.4, elinewidth=0.6, label=label)
                axes.axhline(0.0, color='#c0392b', linewidth=0.6, linestyle=':', zorder=0)
                title = f'{bravais_lattice}  ({n_groups} group{"s" if n_groups > 1 else ""})'
                if bravais_lattice in CUBIC:
                    title += '  *'
                axes.set_title(title, pad=3)
                if bravais_lattice == 'aP':
                    # One extinction group, so every gain is exactly zero and matplotlib would
                    # otherwise invent a +/-0.05 axis around a constant.
                    axes.set_ylim(-1, 1)
                    axes.set_xlim(-1, 1)
                    axes.text(0.5, 0.62, 'one group:', transform=axes.transAxes, ha='center',
                              va='center', fontsize=6.5, color='#c0392b')
                    axes.text(0.5, 0.44, 'zero by construction', transform=axes.transAxes,
                              ha='center', va='center', fontsize=6.5, color='#c0392b')
                if bravais_lattice == 'aP':
                    # The control panel has no data to obscure, so the key goes there.
                    axes.legend(loc='lower center', frameon=False, fontsize=6.5,
                                handlelength=1.2, borderpad=0.1)
                if index % columns == 0:
                    axes.set_ylabel(f'mean $\\Delta$ {merit}')
                if index // columns == rows - 1:
                    axes.set_xlabel('absences in range')

            # The bottom strip is the statistic the panels are a picture of: the rank correlation
            # for each label, with the interval that resamples crystals rather than candidates.
            summary = figure.add_subplot(grid[rows, :])
            positions = np.arange(len(lattices))
            for label, colour, offset, marker in (('incorrect', '#9aa0a6', -0.16, 'o'),
                                                  ('correct', '#1f4e79', 0.16, 's')):
                piece = (primary[(primary['label'] == label) & (primary['merit'] == merit)]
                         .set_index('bravais_lattice').reindex(lattices))
                summary.errorbar(
                    positions + offset, piece['spearman'],
                    yerr=[np.maximum(piece['spearman'] - piece['ci_low'], 0),
                          np.maximum(piece['ci_high'] - piece['spearman'], 0)],
                    fmt=marker, markersize=3.2, color=colour, capsize=1.8, elinewidth=0.7,
                    linestyle='none', label=label)
            summary.axhline(0.0, color='#c0392b', linewidth=0.6, linestyle=':')
            summary.set_xticks(positions)
            summary.set_xticklabels(lattices)
            summary.set_ylabel(f'Spearman $\\rho$\n(absences, $\\Delta$ {merit})')
            summary.set_xlabel('Bravais lattice, ordered by extinction groups searched'
                               '   (* cubic: 10 peaks, 100 reference lines)')
            summary.legend(loc='lower center', frameon=False, ncol=2,
                           bbox_to_anchor=(0.5, -0.62))

            figure.suptitle(
                f'Does the number of systematic absences order the gain from assigning an '
                f'extinction group?\n{merit}, {arm} arm, threshold-0 pool '
                f'({ARMS[arm]["note"]})', fontsize=8.5, y=0.995)
            destination = os.path.join(args.artifact_dir,
                                       f'S04_absence_counts_{arm}_{merit}.png')
            figure.savefig(destination)
            plt.close(figure)
            print(f'wrote {destination}')




def main():
    parser = argparse.ArgumentParser(
        description='S04 -- symmetry as a count, not a label. Tests whether the number of '
                    'systematic absences an extinction group imposes orders the merit gain it '
                    'buys, within each Bravais lattice.')
    parser.add_argument('--stage', required=True,
                        choices=('lookup', 'counts', 'gate', 'support', 'diagnostic',
                                 'magnitude', 'figures'))
    parser.add_argument('--arm', default='general', choices=('general', 'hard'),
                        help='which threshold-0 arm to read; the ordering test uses both')
    parser.add_argument('--processes', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--sample', type=int, default=20000,
                        help='candidates per (lattice, label) for the argmax gate')
    parser.add_argument('--seed', type=int, default=20260826)
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    {'lookup': run_lookup, 'counts': run_counts, 'gate': run_gate, 'support': run_support,
     'diagnostic': run_diagnostic, 'magnitude': run_magnitude,
     'figures': run_figures}[args.stage](args)


if __name__ == '__main__':
    main()
