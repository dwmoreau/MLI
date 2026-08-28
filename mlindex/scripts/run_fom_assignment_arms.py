"""S13 -- the paired real runs behind the per-peak assignment statistic, and the local floor.

Two production decisions read a per-peak assignment probability, and S13 changes what that
probability is. Neither change can be settled by re-scoring a stored pool, because both alter what
the pipeline *produces*:

  * `Candidates.refine_cell` masks the peaks that enter the final Gauss-Newton step by
    `probability > assignment_threshold`, so a different statistic refines a different cell;
  * `IntegralFilter.generate` samples Miller-index assignments from the calibration network's
    softmax, so a different distribution generates different candidates.

So this script runs the indexer, once per arm, over the same patterns at the same per-(entry,
lattice) seeds, and reads what `run.py` would print. It is `run_fom_prune_confirm.py` with the
arm being a statistic rather than a threshold, and it reuses that script's outcome -- the rank of
the best correct candidate in the pooled, M20-sorted list -- so every top-N figure is a
restriction of one stored number.

THE FLOOR. PROTOCOL section 8 requires a gate to be read in standard errors of the contrast floor,
and requires the floor to be measured before the first gate is read. The campaign floor is S08's
and S08 waits on S07, so this measures a **local** floor for this harness: the same arm at four
search seeds, which is scoring-plus-generation noise on exactly the population the gate is read on.
Every number this produces is bounded by that substitution and the bound is stated in the record.

    # the floor: four seeds of the shipped arm
    python mlindex/scripts/run_fom_assignment_arms.py --arm baseline --search-seed 12345
    python mlindex/scripts/run_fom_assignment_arms.py --arm baseline --search-seed 777
    ...
    # the arms, paired against seed 12345
    python mlindex/scripts/run_fom_assignment_arms.py --arm mask --mask-threshold 0.5
    python mlindex/scripts/run_fom_assignment_arms.py --arm assigner
    python mlindex/scripts/run_fom_assignment_arms.py --stage report --population general

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch  # noqa: E402
from mlindex.scripts.run_fom_prune_confirm import N_TOP_CANDIDATES  # noqa: E402
from mlindex.scripts.run_fom_prune_confirm import best_correct_rank  # noqa: E402
from mlindex.scripts.run_fom_prune_rerun import ARMS  # noqa: E402
from mlindex.scripts.run_fom_prune_rerun import BRAVAIS_LATTICES  # noqa: E402
from mlindex.scripts.run_fom_prune_rerun import BROADENING_TAG  # noqa: E402
from mlindex.scripts.run_fom_prune_rerun import TRUTH_SLICE  # noqa: E402
from mlindex.scripts.run_fom_prune_rerun import bundle_directories  # noqa: E402

OUT_ROOT = os.path.join('mlindex', 'characterization', 'fom', 'assignment_arms')

# The populations, reusing S03's own entry tables so the two steps are measured on the same
# crystals. `general` is the all-strata set and `hard` is the retention set; both carry `q2_obs`,
# the split and the truth columns, so no peak list is regenerated here (C2-F-055's lesson).
POPULATIONS = ARMS

# What each arm sets in opt_params. `baseline` sets nothing at all, so it is the shipped path
# rather than a re-specification of it -- if a default moves, the baseline moves with it.
#
# `mask` swaps the statistic `refine_cell` and `n_indexed` read; its threshold is a separate flag
# because 0.95 was chosen against `rho` and does not transfer (Candidates.ASSIGNMENT_STATISTICS).
# `assigner` swaps the distribution `IntegralFilter.generate` resamples Miller indices from.
# `both` is not the sum of the two: they touch different stages and the second is upstream of the
# first, so it is run rather than inferred.
#
# `nofilter` is the arm the replay says is the real comparison. Once the statistic is no longer the
# broken one, the open question is not "which statistic should the filter read" but **whether a
# filter is worth having at all**: the replay puts admitting every peak at +3.60 pp against the
# shipped filter's +0.54 and the posterior's +4.87, so the shipped filter is already refuted and the
# contrast that decides a change is posterior-against-no-filter. A threshold of 0 admits everything,
# since both statistics are strictly positive, so the statistic is left at the shipped one and only
# the cut moves -- which also keeps the arm one option wide.
#
# **Do not read `n_indexed` from this arm.** `assignment_threshold` is shared by the mask and the
# reported count, so at 0 the count is identically the peak list length. It changes no ranking
# (C2-F-066) and it makes that column meaningless here.
ARM_OPTIONS = {
    'baseline': {},
    'nofilter': {'assignment_threshold': 0.0},
    'mask': {'assignment_statistic': 'posterior'},
    'assigner': {'hkl_source': 'posterior'},
    'both': {'assignment_statistic': 'posterior', 'hkl_source': 'posterior'},
    }


def derived_seed(entry_id, bravais_lattice, base_seed):
    """A stable per-(entry, lattice) seed, keyed on a base the floor varies.

    `run_fom_prune_rerun.derived_seed` fixes the base at 12345, which is right for a comparison
    of two arms and wrong for measuring the noise between two runs of the same arm. Same digest,
    same stability guarantee -- `hash` is salted per process and cannot be used -- with the base
    exposed. Called with 12345 it reproduces that function exactly, and a test asserts it.
    """
    digest = hashlib.sha256(
        f'{base_seed}:{entry_id}:{bravais_lattice}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') % (2 ** 31 - 1)


def peak_resident_bytes():
    """Peak RSS of this process and its children, or None where that cannot be read.

    `resource` is POSIX-only and CLAUDE.md forbids Unix-only modules in shipped code, so this is
    guarded rather than imported at the top: a Windows reader gets None and the memory column is
    empty, which is honest, instead of an ImportError on a research driver that ships inside the
    package. macOS reports ru_maxrss in bytes and Linux in kilobytes.
    """
    try:
        import resource
    except ImportError:
        return None
    scale = 1 if sys.platform == 'darwin' else 1024
    return scale*(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                  + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def commit_hash():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=BASE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def ranked_pool(optimizers, task_queues, entry, base_seed):
    """One pattern through all fourteen lattices, pooled and sorted exactly as `run.py` pools it.

    `run_fom_prune_confirm.ranked_pool` with the seed base exposed, and with the reported
    `n_indexed` carried out alongside the rank. `n_indexed` is the second production consumer of
    the statistic this script varies, and it is free here: the optimiser already computed it, so
    taking it from the run is a measurement of the shipped column rather than a recomputation of
    it (PROTOCOL section 3 rule 8).
    """
    from mlindex.optimization.MPOptimizer import run_mp_bl

    q2_obs = np.asarray(entry['q2_obs'], dtype=np.float64)
    unit_cell_true = np.asarray(entry['unit_cell_true'], dtype=np.float64)
    true_lattice = entry['bravais_lattice_true']

    M20, correct, n_indexed = [], [], []
    for bravais_lattice in BRAVAIS_LATTICES:
        optimizer = optimizers[bravais_lattice]
        run_mp_bl(optimizer, bravais_lattice, task_queues, q2_obs[:optimizer.n_peaks],
                  False, None, N_TOP_CANDIDATES,
                  run_seed=derived_seed(entry['entry_id'], bravais_lattice, base_seed))
        top_M20 = np.asarray(optimizer.top_M20, dtype=np.float64)
        M20.append(top_M20)
        n_indexed.append(np.asarray(optimizer.top_n_indexed, dtype=float))
        if bravais_lattice == true_lattice:
            system = optimizer.lattice_system
            correct.append(is_correct_known_bl_batch(
                unit_cell_true[TRUTH_SLICE[system]],
                np.asarray(optimizer.top_unit_cell, dtype=np.float64), system, rtol=0.01))
        else:
            # Only the true lattice can hold a correct cell, which is how every label in this
            # campaign is defined. Everything else is False by construction, not by measurement.
            correct.append(np.zeros(top_M20.size, dtype=bool))

    M20 = np.concatenate(M20)
    correct = np.concatenate(correct)
    n_indexed = np.concatenate(n_indexed)
    order = np.argsort(-M20, kind='stable')
    return {'pool_size': int(M20.size), 'n_correct': int(correct.sum()),
            'best_correct_rank': best_correct_rank(M20, correct),
            'n_indexed_top1': float(n_indexed[order[0]]) if n_indexed.size else float('nan'),
            'n_indexed_mean': float(n_indexed.mean()) if n_indexed.size else float('nan'),
            'n_indexed_correct': float(n_indexed[correct].mean()) if correct.any()
            else float('nan')}


def sampled_entry_ids(population, max_entries, seed=12345):
    """A fixed subsample of **source crystals**, identical for every arm.

    Drawn from the sorted unique entry ids of the whole population with a fixed seed, so two arms
    see the same crystals whatever order the shard files are visited in -- which is what makes the
    comparison paired. Subsampling by crystal and not by pattern-condition keeps all of a crystal's
    condition bundles together: one crystal under several conditions is one draw, not several
    (PROTOCOL section 8), and splitting it would break both the pairing and the cluster bootstrap.

    Returns None when no cap is asked for, which the caller reads as "take everything".
    """
    if not max_entries:
        return None
    root = Path(BASE) / POPULATIONS[population]
    ids = pd.concat(
        [pd.read_parquet(shard, columns=['entry_id'])
         for bundle_dir in sorted(root.iterdir()) if bundle_dir.is_dir()
         for shard in sorted(bundle_dir.glob('entries_*.parquet'))], ignore_index=True)
    unique = np.sort(ids['entry_id'].unique())
    if max_entries >= unique.size:
        return None
    chosen = np.random.default_rng(seed).choice(unique, size=max_entries, replace=False)
    return set(chosen.tolist())


def run_arm(args):
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, shutdown_mp_workers

    root = Path(BASE) / POPULATIONS[args.population]
    keep_entries = sampled_entry_ids(args.population, args.max_entries)
    out_dir = Path(BASE) / args.out_root / args.population / f'{args.arm}_s{args.search_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    options = dict(ARM_OPTIONS[args.arm])
    if 'assignment_statistic' in options:
        options['assignment_threshold'] = args.mask_threshold
    # Provenance beside the numbers, not in a notebook (PROTOCOL section 6). Written before the
    # run so a killed job still says what it was.
    (out_dir / 'provenance.json').write_text(json.dumps({
        'arm': args.arm, 'options': options, 'population': args.population,
        'search_seed': args.search_seed, 'commit': commit_hash(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'processes': args.processes, 'broadening_tag': BROADENING_TAG,
        'n_top_candidates': N_TOP_CANDIDATES,
        'models_dir': os.environ.get('MLINDEX_MODELS_DIR', 'package default'),
        'max_entries': args.max_entries,
        'n_entries_selected': None if keep_entries is None else len(keep_entries),
        }, indent=2), encoding='utf-8')

    optimizers, processes, task_queues = setup_mp_optimizers(
        args.processes, BROADENING_TAG, n_candidates_scale=1, seed=args.search_seed,
        options=options)

    jobs = [(bundle, shard)
            for bundle, bundle_dir in bundle_directories(root).items()
            for shard in sorted(bundle_dir.glob('entries_*.parquet'))]
    jobs = jobs[args.shard_offset::args.shard_stride]
    print(f'{args.arm} seed {args.search_seed} on {args.population}: {len(jobs)} shards, '
          f'options {options}', flush=True)

    started = time.time()
    try:
        for bundle, shard in jobs:
            tag = shard.stem.split('_', 1)[1]
            destination = out_dir / f'ranked_{tag}.parquet'
            if destination.exists():
                print(f'  {bundle}/{tag}: already done, skipping', flush=True)
                continue
            entries = pd.read_parquet(shard)
            if keep_entries is not None:
                entries = entries.loc[entries['entry_id'].isin(keep_entries)]
            if args.limit_entries:
                entries = entries.head(args.limit_entries)
            if not len(entries):
                continue
            rows = []
            for _, entry in entries.iterrows():
                at = time.perf_counter()
                record = ranked_pool(optimizers, task_queues, entry, args.search_seed)
                rows.append(dict(record, entry_id=entry['entry_id'], condition_bundle=bundle,
                                 split=entry['split'], arm=args.arm,
                                 search_seed=args.search_seed,
                                 seconds=time.perf_counter() - at,
                                 peak_rss_bytes=peak_resident_bytes()))
            pd.DataFrame(rows).to_parquet(destination, index=False)
            print(f'  {bundle}/{tag}: {len(rows)} patterns, '
                  f'{sum(r["seconds"] for r in rows):.0f}s of search, '
                  f'{time.time() - started:.0f}s elapsed', flush=True)
    finally:
        shutdown_mp_workers(processes, task_queues)
    print(f'wrote {out_dir}')


def true_lattices(population):
    """`bravais_lattice_true` per (entry, bundle), for the per-lattice stratification.

    The outcome this harness stores -- the rank of the best correct candidate in `run.py`'s pooled
    list -- is deliberately cross-lattice, because that is the list a user reads. Stratifying it
    still needs each pattern's own true lattice, and that lives in the entry tables rather than in
    the run output, so it is joined rather than stored twice.
    """
    root = Path(BASE) / POPULATIONS[population]
    frames = [pd.read_parquet(shard, columns=['entry_id', 'condition_bundle',
                                              'bravais_lattice_true'])
              for bundle_dir in sorted(root.iterdir()) if bundle_dir.is_dir()
              for shard in sorted(bundle_dir.glob('entries_*.parquet'))]
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ['entry_id', 'condition_bundle'])


def load_arm(out_root, population, arm, seed, lattices=None):
    directory = Path(BASE) / out_root / population / f'{arm}_s{seed}'
    shards = sorted(directory.glob('ranked_*.parquet'))
    if not shards:
        raise SystemExit(f'no output under {directory}')
    frame = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    if lattices is not None:
        frame = frame.merge(lattices, on=['entry_id', 'condition_bundle'], how='left')
    return frame


def paired_table(reference, arm, top_ns=(1, 5, 10, 20)):
    """Paired top-N contrast between two frames over the patterns both ran."""
    from scipy.stats import binomtest

    key = ['condition_bundle', 'entry_id']
    joined = reference.merge(arm, on=key, suffixes=('_ref', '_arm'))
    rows = []
    for top_n in top_ns:
        a = joined['best_correct_rank_ref'].between(0, top_n - 1).to_numpy()
        b = joined['best_correct_rank_arm'].between(0, top_n - 1).to_numpy()
        gained, lost = int((~a & b).sum()), int((a & ~b).sum())
        p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
        rows.append({'top_n': top_n, 'reference': a.mean(), 'arm': b.mean(),
                     'delta_pp': (b.mean() - a.mean())*100, 'gained': gained, 'lost': lost,
                     'p_mcnemar': p, 'n_patterns': int(joined.shape[0])})
    return pd.DataFrame(rows), joined


def run_report(args):
    seeds = [int(s) for s in args.floor_seeds.split(',')]
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---- the floor: the same arm at several seeds, every ordered pair contrasted -------------
    # Per lattice as well as pooled, because PROTOCOL section 8 requires a per-lattice claim to be
    # read against that lattice's own floor -- the floor is ordered by free cell parameters over
    # two orders of magnitude, and the lattices where the gains live are the least reproducible.
    lattices = true_lattices(args.population)
    floor_frames = {seed: load_arm(args.out_root, args.population, 'baseline', seed, lattices)
                    for seed in seeds}
    floor_rows = []
    for i, left in enumerate(seeds):
        for right in seeds[i + 1:]:
            for lattice in ['ALL_pooled'] + sorted(lattices['bravais_lattice_true'].unique()):
                a, b = floor_frames[left], floor_frames[right]
                if lattice != 'ALL_pooled':
                    a = a.loc[a['bravais_lattice_true'] == lattice]
                    b = b.loc[b['bravais_lattice_true'] == lattice]
                if len(a) < args.min_entries or len(b) < args.min_entries:
                    continue
                table, _ = paired_table(a, b)
                for _, row in table.iterrows():
                    floor_rows.append(dict(row, seed_a=left, seed_b=right,
                                           bravais_lattice_true=lattice))
    floor = pd.DataFrame(floor_rows)
    floor.to_csv(artifact_dir / f'S13_floor_{args.population}.csv', index=False)

    # The floor a gate is read against is the spread of |delta| between two runs that differ only
    # in the seed -- the contrast floor, not a merit's own value spread (PROTOCOL section 8).
    per_lattice_floor = (floor.assign(abs_delta=floor['delta_pp'].abs())
                         .groupby(['bravais_lattice_true', 'top_n'])['abs_delta']
                         .agg(['mean', 'max', 'std', 'count'])
                         .rename(columns={'mean': 'mean_abs_delta_pp',
                                          'max': 'max_abs_delta_pp',
                                          'std': 'sd_delta_pp', 'count': 'n_pairs'}))
    summary = per_lattice_floor.loc['ALL_pooled']
    print(f'CONTRAST FLOOR, {args.population}, {len(seeds)} seeds of the shipped arm')
    print(summary.to_string())
    print('\nper lattice, top-10 (the lattice a per-lattice claim is read against):')
    print(per_lattice_floor.xs(10, level='top_n')[['mean_abs_delta_pp', 'max_abs_delta_pp']]
          .to_string(), '\n')

    # ---- the arms, paired against the reference seed -----------------------------------------
    reference = floor_frames[seeds[0]]
    arm_rows = []
    for arm in args.arms.split(','):
        if arm == 'baseline':
            continue
        arm_frame = load_arm(args.out_root, args.population, arm, seeds[0], lattices)
        table, joined = paired_table(reference, arm_frame)
        joined.to_csv(artifact_dir / f'S13_arm_per_entry_{args.population}_{arm}.csv',
                      index=False)
        for lattice in ['ALL_pooled'] + sorted(lattices['bravais_lattice_true'].unique()):
            a, b = reference, arm_frame
            if lattice != 'ALL_pooled':
                a = a.loc[a['bravais_lattice_true'] == lattice]
                b = b.loc[b['bravais_lattice_true'] == lattice]
            if len(a) < args.min_entries:
                continue
            per_lattice_table, _ = paired_table(a, b)
            for _, row in per_lattice_table.iterrows():
                key = (lattice, row['top_n'])
                floor_at = (per_lattice_floor.loc[key, 'mean_abs_delta_pp']
                            if key in per_lattice_floor.index else np.nan)
                arm_rows.append(dict(row, arm=arm, bravais_lattice_true=lattice,
                                     floor_mean_abs_delta_pp=floor_at,
                                     delta_in_floors=row['delta_pp']/floor_at
                                     if floor_at else np.nan))
        print(f'--- {arm} against baseline (seed {seeds[0]}), pooled over lattices')
        print(table.to_string(index=False), '\n')

    arms_table = pd.DataFrame(arm_rows)
    arms_table.to_csv(artifact_dir / f'S13_arms_topn_{args.population}.csv', index=False)

    # ---- cost, and the reported n_indexed column ---------------------------------------------
    cost = []
    for arm in args.arms.split(','):
        frame = load_arm(args.out_root, args.population, arm, seeds[0])
        cost.append({'arm': arm, 'seconds_per_pattern': frame['seconds'].mean(),
                     'pool_size': frame['pool_size'].mean(),
                     'peak_rss_gb': frame['peak_rss_bytes'].max()/2**30
                     if frame['peak_rss_bytes'].notna().any() else np.nan,
                     'n_indexed_top1': frame['n_indexed_top1'].mean(),
                     'n_indexed_mean': frame['n_indexed_mean'].mean(),
                     'n_indexed_correct': frame['n_indexed_correct'].mean()})
    cost = pd.DataFrame(cost)
    cost.to_csv(artifact_dir / f'S13_arms_cost_{args.population}.csv', index=False)
    print(cost.to_string(index=False))
    print(f'\nwrote {artifact_dir}/S13_arms_topn_{args.population}.csv')


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage', choices=['run', 'report', 'diagnostic'], default='run')
    parser.add_argument('--arm', default='baseline', choices=sorted(ARM_OPTIONS))
    parser.add_argument('--arms', default='baseline,mask,assigner',
                        help='report stage: the arms to contrast against the baseline')
    parser.add_argument('--population', default='general', choices=sorted(POPULATIONS))
    parser.add_argument('--search-seed', type=int, default=12345)
    parser.add_argument('--floor-seeds', default='12345,777,20260827,4242',
                        help='report stage: the baseline seeds the floor is measured over')
    parser.add_argument('--mask-threshold', type=float, default=0.95,
                        help='assignment_threshold for the mask arm; 0.95 is the rho-era value '
                             'and does not transfer to the posterior')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--out-root', default=OUT_ROOT)
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--shard-stride', type=int, default=1)
    parser.add_argument('--shard-offset', type=int, default=0)
    parser.add_argument('--max-entries', type=int, default=None,
                        help='cap the number of source CRYSTALS, subsampled with a fixed seed so '
                             'every arm sees the same ones; all of a crystal\'s condition '
                             'bundles travel together')
    parser.add_argument('--limit-entries', type=int, default=None,
                        help='NOT a population cap: applied per shard FILE, so on the hard set '
                             '(60 files of 16-17 entries) any value above 17 does nothing. Use '
                             '--max-entries')
    parser.add_argument('--min-entries', type=int, default=20,
                        help='report stage: pattern-conditions below which a\n'
                             'per-lattice row is not reported')
    parser.add_argument('--diagnostic-entries', type=int, default=15,
                        help='entries the network/posterior diagnostic runs over')
    return parser.parse_args()


def main():
    args = _parse_args()
    {'report': run_report, 'diagnostic': run_diagnostic}.get(args.stage, run_arm)(args)



# -------------------------------------------------------------------------------------------
# Part 2's diagnostic: the calibration network against the analytic posterior, per peak
# -------------------------------------------------------------------------------------------
def run_diagnostic(args):
    """What the two distributions are, side by side, on the cells `generate` actually feeds them.

    **This is the diagnostic, not the verdict.** The network's job is stochastic candidate
    diversification -- `vectorized_resampling` draws a Miller-index labelling from it and each draw
    becomes a distinct candidate -- so its quality is the quality of the *cells* that come out, and
    that is measured end to end by the `assigner` arm. Campaign 1's block-B comparison scored
    per-peak correctness instead and its AUC advantage did not settle the question; this section
    exists so that mistake is not repeated silently.

    What it does settle, because nothing else measures it:

      * **wall clock**, per call, on the same cells. `predict_hkl` is 70.7 % of `generate` and the
        un-batched ONNX run is 99 % of that (ProfileOptimizer.py), so the swap's price is a real
        number rather than an inference from file sizes;
      * **what each distribution looks like** -- peak mass and entropy -- because that is what
        decides how diverse the resampled candidates are, and a distribution that is far sharper
        will diversify far less whatever its accuracy;
      * **per-peak accuracy given the truth's own cell**, which is the one place the question is
        well posed: both distributions are evaluated on `xnn_true` for the entry's own lattice, so
        the correct Miller index names one reference line.
    """
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, shutdown_mp_workers
    from mlindex.scripts.run_fom_assignment import hkl_class_index
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution
    from mlindex.utilities.Q2Calculator import Q2Calculator
    from mlindex.utilities.numba_functions import fast_assign

    root = Path(BASE)/POPULATIONS[args.population]
    entries = pd.concat(
        [pd.read_parquet(shard) for bundle_dir in sorted(root.iterdir()) if bundle_dir.is_dir()
         for shard in sorted(bundle_dir.glob('entries_*.parquet'))], ignore_index=True)
    # Stratified by true lattice rather than `head`, so the well-posed half -- which only exists
    # where the candidate lattice IS the entry's own -- covers more than whichever lattices happen
    # to come first in the file. With `head` a three-entry run scored one lattice.
    entries = (entries.groupby('bravais_lattice_true', sort=False, group_keys=False)
               .head(max(1, args.diagnostic_entries//14)))
    if len(entries) < args.diagnostic_entries:
        extra = entries.index
        entries = pd.concat([entries, entries.head(0)]) if len(extra) else entries

    optimizers, processes, task_queues = setup_mp_optimizers(
        1, BROADENING_TAG, n_candidates_scale=1, seed=args.search_seed)
    # Warm the numba kernel before anything is timed. Without this the first lattice measured
    # carries `posterior_exponent_terms`'s compile -- 120 ms against a 0.4 ms steady state on the
    # same shape -- and it reads as that lattice being slow rather than as the JIT.
    get_assignment_distribution(np.linspace(0.01, 0.5, 20),
                                np.sort(np.linspace(0.005, 0.6, 128))[np.newaxis], 'cubic')
    rows = []
    try:
        for bravais_lattice in BRAVAIS_LATTICES:
            optimizer = optimizers[bravais_lattice]
            split_groups = [info['split_group'] for info in optimizer.opt_params['generator_info']
                            if info['generator'] == 'integral_filter']
            if not split_groups:
                continue
            # One split group per lattice. The calibration model is per split group and there are
            # forty-three of them; the point here is the two distributions' shapes and prices,
            # which do not need every group, and saying so is cheaper than implying coverage.
            generator = optimizer.wrapper.integral_filter_generator[split_groups[0]]
            lattice_system = optimizer.lattice_system
            q2_calculator = Q2Calculator(lattice_system=lattice_system, hkl=generator.hkl_ref,
                                         tensorflow=False, representation='xnn')
            for _, entry in entries.iterrows():
                q2_obs = np.asarray(entry['q2_obs'], dtype=np.float64)[:optimizer.n_peaks]
                top_n = generator.model_params['n_volumes']
                xnn_pred, _ = generator.predict_xnn(top_n, q2_obs=q2_obs[np.newaxis],
                                                    batch_size=2)
                xnn_pred = xnn_pred[0]
                q2_ref_calc = q2_calculator.get_q2(xnn_pred)

                at = time.perf_counter()
                network = generator.predict_hkl(
                    np.repeat(q2_obs[np.newaxis], repeats=top_n, axis=0), xnn_pred, batch_size=2)
                network_seconds = time.perf_counter() - at

                at = time.perf_counter()
                posterior = get_assignment_distribution(
                    q2_obs, q2_ref_calc, lattice_system, normalise=False)
                posterior_seconds = time.perf_counter() - at

                nearest = fast_assign(q2_obs, q2_ref_calc)
                row = dict(entry_id=entry['entry_id'], condition_bundle=entry['condition_bundle'],
                           bravais_lattice=bravais_lattice, split_group=split_groups[0],
                           n_reference_lines=int(generator.hkl_ref.shape[0]), top_n=int(top_n),
                           network_seconds=network_seconds, posterior_seconds=posterior_seconds)
                for name, distribution in (('network', network), ('posterior', posterior)):
                    normalised = distribution/distribution.sum(axis=2, keepdims=True)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        entropy = -np.nansum(normalised*np.log(normalised), axis=2)
                    row[f'{name}_max_mass'] = float(normalised.max(axis=2).mean())
                    row[f'{name}_entropy'] = float(entropy.mean())
                    # For the posterior this is 1 by construction -- its exponent is monotone in
                    # the distance, so its argmax IS the nearest line. It is reported anyway
                    # because the network's value is the informative one: how often the dense
                    # stack overrides the nearest line, which is the joint conditioning the
                    # analytic form cannot reproduce.
                    row[f'{name}_agrees_with_nearest'] = float(
                        (np.argmax(distribution, axis=2) == nearest).mean())
                # The well-posed half: both distributions on the truth's own cell, where the
                # correct Miller index names one reference line.
                if bravais_lattice == entry['bravais_lattice_true']:
                    xnn_true = np.asarray(entry['xnn_true'], dtype=np.float64)[np.newaxis]
                    xnn_true = xnn_true[:, :generator.unit_cell_length]
                    q2_ref_true = q2_calculator.get_q2(xnn_true)
                    truth_class = hkl_class_index(
                        np.asarray(entry['hkl_true'], dtype=float).reshape(-1, 3)[:q2_obs.size],
                        generator.hkl_ref, lattice_system)
                    real = truth_class != generator.hkl_ref.shape[0] - 1
                    network_true = generator.predict_hkl(q2_obs[np.newaxis], xnn_true,
                                                         batch_size=2)
                    posterior_true = get_assignment_distribution(
                        q2_obs, q2_ref_true, lattice_system, normalise=False)
                    if real.any():
                        row['network_top1_true_cell'] = float(
                            (np.argmax(network_true[0], axis=1)[real] == truth_class[real]).mean())
                        row['posterior_top1_true_cell'] = float(
                            (np.argmax(posterior_true[0], axis=1)[real]
                             == truth_class[real]).mean())
                        row['n_real_peaks'] = int(real.sum())
                rows.append(row)
            print(f'  {bravais_lattice}: {len(entries)} entries', flush=True)
    finally:
        shutdown_mp_workers(processes, task_queues)

    diagnostic = pd.DataFrame(rows)
    artifact_dir = Path(BASE)/args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(artifact_dir/f'S13_assigner_diagnostic_{args.population}.csv', index=False)

    per_lattice = diagnostic.groupby('bravais_lattice').mean(numeric_only=True)
    print('\nper lattice (means), then the unweighted aggregate:')
    print(per_lattice[['n_reference_lines', 'network_seconds', 'posterior_seconds',
                       'network_max_mass', 'posterior_max_mass',
                       'network_agrees_with_nearest', 'posterior_agrees_with_nearest']]
          .to_string())
    well_posed = diagnostic.dropna(subset=['network_top1_true_cell'])
    if len(well_posed):
        by_lattice = well_posed.groupby('bravais_lattice')[
            ['network_top1_true_cell', 'posterior_top1_true_cell']].mean()
        print('\nper-peak top-1 on the truth\'s own cell, per lattice:')
        print(by_lattice.to_string())
        print(f'\nunweighted aggregate: network {by_lattice["network_top1_true_cell"].mean():.4f}, '
              f'posterior {by_lattice["posterior_top1_true_cell"].mean():.4f}')
    print(f'\nspeed-up: {per_lattice["network_seconds"].mean()/per_lattice["posterior_seconds"].mean():.1f}x '
          f'({per_lattice["network_seconds"].mean()*1000:.1f} ms -> '
          f'{per_lattice["posterior_seconds"].mean()*1000:.1f} ms per call)')
    print(f'\nwrote {artifact_dir}/S13_assigner_diagnostic_{args.population}.csv')


if __name__ == '__main__':
    main()
