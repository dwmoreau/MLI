"""S11's validity gates. G1 is the one that licenses every number the step reports.

    python mlindex/scripts/run_fom_extinction_sweep.py --gate --gate-rows 200000

Five gates, run in this order because each is cheaper than the next and a failure in an earlier
one makes the later ones meaningless:

  G0  the reference list on disk is still the one the pool recorded (`hkl_ref_length`)
  G2  the masked q2 route is bit-identical to the per-group one -- it is, and it is also SLOWER,
      so it stays off; the gate exists to keep that a measured fact rather than a memory
  G1  under criterion 'M20' the offline argmax reproduces the STORED `spacegroup` and the STORED
      `M20`, with `==` and not `isclose`
  G3  the offline sweep agrees with a real `Candidates.assign_extinction_group`, criterion by
      criterion -- the anti-drift gate
  G4  how often the `M_rev` support floor fires, per lattice and per group

**Why G1 is the gate.** C2-F-036 reproduced the assignment on 310 807 of 310 807 candidates, but
on the PRE-deduplication stream. This pool is post-deduplication, and `14b13a9` fixed a defect
where deduplication attached spacegroups to the wrong candidates whenever a NaN cell was dropped.
So a G1 failure here is not a bug in this script by default -- it is equally a candidate for that
defect having reached this pool. Hence offending ROWS are kept rather than a count: a count cannot
tell the two apart, and the per-lattice pattern of the failures can.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mlindex.model_training import FomBenchmark
from mlindex.utilities.ExtinctionCounts import LATTICE_SYSTEM, build_group_masks, get_absence_counts
from mlindex.utilities.FigureOfMerits import m_rev_support_floor
from mlindex.utilities.Q2Calculator import Q2Calculator

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_LATTICES = ('aP',)


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


def _lattices(pool, bravais_lattices=None):
    found = {}
    for path in sorted(Path(pool).glob('candidates*.parquet')):
        lattice = path.stem.split('_')[-1]
        if lattice in SKIP_LATTICES:
            continue
        if bravais_lattices and lattice not in bravais_lattices:
            continue
        found.setdefault(lattice, []).append(path)
    return found


def gate_reference_lists(pool, lattices):
    """G0: the reference list on disk is the one the pool was generated against."""
    rows = []
    for lattice, paths in lattices.items():
        system = LATTICE_SYSTEM[lattice]
        hkl_ref = np.load(FomBenchmark._hkl_ref_path(system, lattice))
        stored = pq.read_table(paths[0], columns=['hkl_ref_length']).column(
            'hkl_ref_length')[0].as_py()
        rows.append({'bravais_lattice': lattice, 'n_reference_lines': int(hkl_ref.shape[0]),
                     'stored_hkl_ref_length': int(stored),
                     'passes': int(hkl_ref.shape[0]) == int(stored)})
    return pd.DataFrame(rows)


def gate_masked_identity(pool, lattices, n_rows=2000):
    """G2: masking the full q2 matrix equals computing each group's q2 separately, bit for bit.

    Recorded per lattice rather than assumed, because it decides whether the masked route may ever
    be switched on. It is bit-identical and it is also 0.2-0.3x the SPEED, so it stays off -- but
    a later session tempted by the idea should find the measurement, not re-derive it.
    """
    rows = []
    for lattice, paths in lattices.items():
        system = LATTICE_SYSTEM[lattice]
        hkl_ref = np.load(FomBenchmark._hkl_ref_path(system, lattice))
        sets = FomBenchmark.spacegroup_reference_sets(system, lattice)
        masks = build_group_masks(hkl_ref, lattice)
        xnn = np.stack([np.asarray(v, dtype=np.float64) for v in
                        pq.ParquetFile(paths[0]).read_row_group(0, columns=['xnn']).to_pandas()
                        ['xnn'][:n_rows]])
        full = Q2Calculator(lattice_system=system, hkl=hkl_ref, tensorflow=False,
                            representation='xnn').get_q2(xnn)
        identical = 0
        for key in sets:
            direct = Q2Calculator(lattice_system=system, hkl=sets[key], tensorflow=False,
                                  representation='xnn').get_q2(xnn)
            identical += int(np.array_equal(direct, full[:, masks[key]]))
        rows.append({'bravais_lattice': lattice, 'n_groups': len(sets),
                     'n_groups_bit_identical': identical, 'n_candidates': int(xnn.shape[0]),
                     'passes': identical == len(sets)})
    return pd.DataFrame(rows)


def gate_reproduction(pool, lattices, gate_rows):
    """G1: the offline argmax under M20 reproduces the stored group and the stored M20 exactly."""
    entries = FomBenchmark.load_entries(pool).set_index(['entry_id', 'condition_bundle'])
    rows, offenders = [], []
    for lattice, paths in lattices.items():
        checked = matched_group = matched_M20 = 0
        for path in paths:
            if checked >= gate_rows:
                break
            source = pq.ParquetFile(path)
            for index in range(source.num_row_groups):
                if checked >= gate_rows:
                    break
                block = source.read_row_group(index, columns=[
                    'entry_id', 'condition_bundle', 'bravais_lattice', 'candidate_id',
                    'lattice_system', 'spacegroup', 'n_peaks', 'xnn', 'M20']).to_pandas()
                for keys, group in block.groupby(
                        ['entry_id', 'condition_bundle', 'lattice_system', 'n_peaks'], sort=False):
                    entry_id, bundle, system, n_peaks = keys
                    q2_obs = np.asarray(entries.loc[(entry_id, bundle), 'q2_obs'],
                                        dtype=np.float64)[:int(n_peaks)]
                    xnn = np.stack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
                    group_keys, winners, M20, _, _, _ = FomBenchmark.extinction_group_sweep(
                        q2_obs, xnn, system, lattice, criteria=('M20',))
                    winner = winners['M20']
                    chosen = np.array([group_keys[i] for i in winner])
                    stored_group = group['spacegroup'].to_numpy()
                    recomputed = M20[np.arange(xnn.shape[0]), winner]
                    stored_M20 = group['M20'].to_numpy()

                    same_group = chosen == stored_group
                    same_M20 = recomputed == stored_M20
                    checked += xnn.shape[0]
                    matched_group += int(same_group.sum())
                    matched_M20 += int(same_M20.sum())
                    bad = ~(same_group & same_M20)
                    if bad.any() and len(offenders) < 2000:
                        # Rows, not a count. A count cannot distinguish this script being wrong
                        # from the deduplication defect `14b13a9` fixed having reached the pool,
                        # and the per-lattice pattern of the failures can.
                        offenders.append(pd.DataFrame({
                            'entry_id': group['entry_id'].to_numpy()[bad],
                            'condition_bundle': group['condition_bundle'].to_numpy()[bad],
                            'bravais_lattice': lattice,
                            'candidate_id': group['candidate_id'].to_numpy()[bad],
                            'stored_group': stored_group[bad], 'recomputed_group': chosen[bad],
                            'stored_M20': stored_M20[bad], 'recomputed_M20': recomputed[bad],
                            }))
        rows.append({'bravais_lattice': lattice, 'n_checked': checked,
                     'n_group_reproduced': matched_group, 'n_M20_reproduced': matched_M20,
                     'passes': checked == matched_group == matched_M20})
    return (pd.DataFrame(rows),
            pd.concat(offenders, ignore_index=True) if offenders else pd.DataFrame())


def gate_production_agreement(pool, lattices, criteria, n_cells=3, n_rows=200):
    """G3: the sweep agrees with a real `Candidates`, criterion by criterion.

    The loop scaffolding is the one thing the two do NOT share, so it is the one thing that has to
    be gated rather than argued.
    """
    from mlindex.optimization.Candidates import Candidates
    entries = FomBenchmark.load_entries(pool).set_index(['entry_id', 'condition_bundle'])
    rows = []
    for lattice, paths in lattices.items():
        system = LATTICE_SYSTEM[lattice]
        hkl_ref = np.load(FomBenchmark._hkl_ref_path(system, lattice))
        block = pq.ParquetFile(paths[0]).read_row_group(0, columns=[
            'entry_id', 'condition_bundle', 'n_peaks', 'xnn']).to_pandas()
        seen = 0
        for keys, group in block.groupby(['entry_id', 'condition_bundle', 'n_peaks'], sort=False):
            if seen >= n_cells:
                break
            seen += 1
            entry_id, bundle, n_peaks = keys
            q2_obs = np.asarray(entries.loc[(entry_id, bundle), 'q2_obs'],
                                dtype=np.float64)[:int(n_peaks)]
            xnn = np.stack([np.asarray(v, dtype=np.float64) for v in group['xnn'][:n_rows]])
            group_keys, winners, _, _, _, _ = FomBenchmark.extinction_group_sweep(
                q2_obs, xnn, system, lattice, criteria=tuple(criteria))
            for criterion in criteria:
                opt_params = {'minimum_uc': 2.0, 'maximum_uc': 60.0,
                              'assignment_threshold': 0.95, 'figure_of_merit': 'M20',
                              'extinction_criterion': criterion}
                candidates = Candidates(
                    q2_obs=q2_obs, xnn=xnn, hkl_ref=hkl_ref, lattice_system=system,
                    bravais_lattice=lattice, opt_params=opt_params,
                    rng=np.random.default_rng(0), fom=None, zero_error=False, wavelength=None)
                candidates.best_xnn = xnn
                candidates.assign_extinction_group()
                agree = [group_keys[i] for i in winners[criterion]] == candidates.best_spacegroup
                rows.append({'bravais_lattice': lattice, 'criterion': criterion,
                             'entry_id': entry_id, 'n_candidates': int(xnn.shape[0]),
                             'passes': bool(agree)})
    return pd.DataFrame(rows)


def gate_support_floor(pool, lattices, n_rows=20000):
    """G4: how often the `M_rev` support floor fires, per lattice and per group.

    Reported BEFORE any argmax result, because an extinction group's whole job is to delete
    predicted lines and deleting them is exactly what drives `N_cal` under the floor. A rule whose
    wins come from ties at zero is not the same result as one whose wins come from its criterion,
    and only this distinguishes them (C2-F-059, C2-Q-017).
    """
    entries = FomBenchmark.load_entries(pool).set_index(['entry_id', 'condition_bundle'])
    floor = m_rev_support_floor()
    rows = []
    for lattice, paths in lattices.items():
        system = LATTICE_SYSTEM[lattice]
        block = pq.ParquetFile(paths[0]).read_row_group(0, columns=[
            'entry_id', 'condition_bundle', 'n_peaks', 'xnn']).to_pandas()[:n_rows]
        floored = total = all_floored = 0
        for keys, group in block.groupby(['entry_id', 'condition_bundle', 'n_peaks'], sort=False):
            entry_id, bundle, n_peaks = keys
            q2_obs = np.asarray(entries.loc[(entry_id, bundle), 'q2_obs'],
                                dtype=np.float64)[:int(n_peaks)]
            xnn = np.stack([np.asarray(v, dtype=np.float64) for v in group['xnn']])
            _, _, _, _, n_cal, _ = FomBenchmark.extinction_group_sweep(
                q2_obs, xnn, system, lattice, criteria=('M_rev',))
            below = n_cal < floor
            floored += int(below.sum())
            total += int(below.size)
            all_floored += int((below.all(axis=1)).sum())
        rows.append({'bravais_lattice': lattice, 'n_group_evaluations': total,
                     'n_below_floor': floored,
                     'share_below_floor': floored/total if total else np.nan,
                     'n_candidates_with_no_supported_group': all_floored})
    return pd.DataFrame(rows)


def run_gates(pool, artifact_dir, tag, gate_rows, criteria, bravais_lattices=None):
    lattices = _lattices(pool, bravais_lattices)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    started = time.time()

    print('G0 reference lists ...')
    g0 = gate_reference_lists(pool, lattices)
    print(g0.to_string(index=False))

    print('\nG2 masked-q2 identity ...')
    g2 = gate_masked_identity(pool, lattices)
    print(g2.to_string(index=False))

    print('\nG1 reproduction of the stored assignment ...')
    g1, offenders = gate_reproduction(pool, lattices, gate_rows)
    print(g1.to_string(index=False))

    print('\nG3 sweep vs production ...')
    g3 = gate_production_agreement(pool, lattices, criteria)
    print(g3.groupby('criterion')['passes'].agg(['sum', 'count']).to_string())

    print('\nG4 support floor ...')
    g4 = gate_support_floor(pool, lattices)
    print(g4.to_string(index=False))

    for name, frame in (('g0_reference_lists', g0), ('g2_masked_identity', g2),
                        ('g1_reproduction', g1), ('g3_production_agreement', g3),
                        ('g4_support_floor', g4)):
        frame.to_csv(os.path.join(artifact_dir, f'{tag}_gate_{name}.csv'), index=False,
                     encoding='utf-8')
    if not offenders.empty:
        offenders.to_csv(os.path.join(artifact_dir, f'{tag}_gate_g1_offenders.csv'), index=False,
                         encoding='utf-8')

    passed = {'G0': bool(g0['passes'].all()), 'G2': bool(g2['passes'].all()),
              'G1': bool(g1['passes'].all()), 'G3': bool(g3['passes'].all())}
    report = {
        'tag': tag, 'pool': str(pool), 'commit': commit_hash(), 'dirty_tree': dirty_tree(),
        'criteria': list(criteria), 'skipped_lattices': list(SKIP_LATTICES),
        'm_rev_support_floor': m_rev_support_floor(),
        'gate_rows_per_lattice': int(gate_rows),
        'passed': passed, 'seconds': round(time.time() - started, 1),
        'n_g1_offender_rows': int(offenders.shape[0]),
        }
    with open(os.path.join(artifact_dir, f'{tag}_gate.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    print('\n' + json.dumps(passed, indent=2))
    if not passed['G1']:
        print('\nG1 FAILED. Nothing downstream is meaningful until this is explained -- and the '
              'deduplication defect 14b13a9 fixed is as live a candidate as a bug in this script. '
              f'Offending rows: {tag}_gate_g1_offenders.csv')
    return 0 if all(passed.values()) else 1
