"""S03 Phase 2b -- confirm the prune recommendation with two real runs.

Everything else in S03 measures **reachability**: whether a correct cell is still in the pool
after the cut. That is a ceiling, not an outcome. What a user sees is `run.py`'s printed list --
each Bravais lattice's top 20 pooled and sorted by M20 -- and campaign 1's F-171 found the two
can come apart badly: over 210 entries, lowering the threshold **alone** moved top-10 by +0.95 pp
(p = 0.73, i.e. nothing), the merit alone by +4.29 pp, and the two together by +7.62 pp.

So this script does not restrict one run; it runs the indexer **twice**, differing only in the
prune threshold, and reports what reaches the top of the printed list. A restriction of a
threshold-0 run gives the candidates a lower-threshold run would have *admitted*, not the cells
it would have *produced* -- `refine_cell`, `standardize_cell` and `correct_off_by_two` all draw
from an RNG stream sized by the surviving row count, so a real run diverges after the cut.

PAIRING. Both arms take the same seed for the same (entry, Bravais lattice), so the two runs see
the same generated candidates and the ONLY difference between them is where the cut falls. That
is what makes a paired test legitimate rather than a comparison of two independent searches.

WHAT IS MEASURED, per pattern and per arm:
  * the rank of the best correct candidate in the pooled, M20-sorted list -- so every top-N
    figure is a restriction of one number rather than a separate measurement;
  * wall clock, which is the other half of the question the cut is trading against.

    python mlindex/scripts/run_fom_prune_confirm.py --threshold 5.0 --shard-stride 5 --shard-offset 0
    python mlindex/scripts/run_fom_prune_confirm.py --threshold 3.0 --shard-stride 5 --shard-offset 0
    python mlindex/scripts/run_fom_prune_confirm.py --stage report --thresholds 5.0,3.0

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

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.optimization.CandidateValidation import is_correct_known_bl_batch
from mlindex.scripts.run_fom_prune_rerun import (ARMS, BRAVAIS_LATTICES, BROADENING_TAG,
                                                 TRUTH_SLICE, bundle_directories, derived_seed)

# What `run.py` uses (run.py:340, :422). Kept here as a constant rather than a flag: the point of
# this script is to reproduce the shipped answer, so the shipped value is the only correct one.
N_TOP_CANDIDATES = 20

OUT_ROOT = os.path.join('mlindex', 'characterization', 'fom', 'prune_confirm')


def ranked_pool(optimizers, task_queues, entry, threshold):
    """One pattern through all fourteen lattices, pooled and sorted exactly as `run.py` pools it.

    `run.py` collects each lattice's `top_unit_cell` / `top_M20` and sorts the union by M20
    descending (`_collect_results`, then `sort_values('M20', ascending=False)`). This reproduces
    that, and labels each row against the truth so the rank of the best correct candidate can be
    read straight off.
    """
    from mlindex.optimization.MPOptimizer import run_mp_bl

    q2_obs = np.asarray(entry['q2_obs'], dtype=np.float64)
    unit_cell_true = np.asarray(entry['unit_cell_true'], dtype=np.float64)
    true_lattice = entry['bravais_lattice_true']

    M20, correct, lattice_of = [], [], []
    for bravais_lattice in BRAVAIS_LATTICES:
        optimizer = optimizers[bravais_lattice]
        run_mp_bl(optimizer, bravais_lattice, task_queues, q2_obs[:optimizer.n_peaks],
                  False, None, N_TOP_CANDIDATES,
                  run_seed=derived_seed(entry['entry_id'], bravais_lattice))
        top_M20 = np.asarray(optimizer.top_M20, dtype=np.float64)
        M20.append(top_M20)
        lattice_of.append(np.full(top_M20.size, bravais_lattice))
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
    return {'pool_size': int(M20.size), 'n_correct': int(correct.sum()),
            'best_correct_rank': best_correct_rank(M20, correct)}


def best_correct_rank(M20, correct):
    """Where the best correct candidate sits in the list `run.py` prints, or -1 if absent.

    `run.py` sorts the pooled candidates by M20 descending and prints from the top, so rank 0 is
    the cell it names first. A **stable** sort is what reproduces it: ties then keep the order the
    lattices were assembled in, which is the same first-maximum convention `_collect_results` and
    `sort_values` produce. An unstable sort would reorder ties arbitrarily and make the rank of a
    tied correct candidate depend on the sort implementation.

    Returning the rank rather than a top-N flag means every top-N figure is a restriction of one
    stored number, so top-1, top-10 and top-20 cannot disagree with each other.
    """
    correct = np.asarray(correct, dtype=bool)
    if not correct.any():
        return -1
    order = np.argsort(-np.asarray(M20, dtype=np.float64), kind='stable')
    return int(np.argmax(correct[order]))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage', choices=['run', 'report', 'figure'], default='run')
    parser.add_argument('--threshold', type=float, default=5.0)
    parser.add_argument('--thresholds', default='5.0,3.0', help='report stage: the two arms')
    parser.add_argument('--arm', default='general', choices=sorted(ARMS))
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--out-root', default=OUT_ROOT)
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--shard-stride', type=int, default=1)
    parser.add_argument('--shard-offset', type=int, default=0)
    parser.add_argument('--limit-entries', type=int, default=None)
    return parser.parse_args()


def run_arm(args):
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, shutdown_mp_workers

    root = Path(BASE) / ARMS[args.arm]
    out_dir = Path(BASE) / args.out_root / args.arm / f't{args.threshold:g}'
    out_dir.mkdir(parents=True, exist_ok=True)

    options = {'prune_m20_threshold': args.threshold}
    optimizers, processes, task_queues = setup_mp_optimizers(
        args.processes, BROADENING_TAG, n_candidates_scale=1, seed=12345, options=options)

    jobs = [(bundle, shard)
            for bundle, bundle_dir in bundle_directories(root).items()
            for shard in sorted(bundle_dir.glob('entries_*.parquet'))]
    jobs = jobs[args.shard_offset::args.shard_stride]
    print(f'threshold {args.threshold:g}: {len(jobs)} shards for offset {args.shard_offset} '
          f'of stride {args.shard_stride}', flush=True)

    started = time.time()
    try:
        for bundle, shard in jobs:
            tag = shard.stem.split('_', 1)[1]
            destination = out_dir / f'ranked_{tag}.parquet'
            if destination.exists():
                print(f'  {bundle}/{tag}: already done, skipping', flush=True)
                continue
            entries = pd.read_parquet(shard)
            if args.limit_entries:
                entries = entries.head(args.limit_entries)
            rows = []
            for _, entry in entries.iterrows():
                # Per-pattern wall clock, which is the cost half of the trade the cut makes.
                # Timed around the whole fourteen-lattice pass, as a user experiences it.
                at = time.perf_counter()
                record = ranked_pool(optimizers, task_queues, entry, args.threshold)
                rows.append(dict(record, entry_id=entry['entry_id'], condition_bundle=bundle,
                                 split=entry['split'], threshold=args.threshold,
                                 seconds=time.perf_counter() - at))
            pd.DataFrame(rows).to_parquet(destination, index=False)
            print(f'  {bundle}/{tag}: {len(rows)} patterns, '
                  f'{sum(r["seconds"] for r in rows):.0f}s of search, '
                  f'{time.time() - started:.0f}s elapsed', flush=True)
    finally:
        shutdown_mp_workers(processes, task_queues)
    print(f'wrote {out_dir}')


def run_report(args):
    from scipy.stats import binomtest

    thresholds = [float(value) for value in args.thresholds.split(',')]
    frames = {}
    for threshold in thresholds:
        directory = Path(BASE) / args.out_root / args.arm / f't{threshold:g}'
        shards = sorted(directory.glob('ranked_*.parquet'))
        if not shards:
            raise SystemExit(f'no output under {directory}')
        frames[threshold] = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    high, low = max(thresholds), min(thresholds)
    key = ['condition_bundle', 'entry_id']
    joined = frames[high].merge(frames[low], on=key, suffixes=('_high', '_low'))
    print(f'{joined.shape[0]} patterns run at both thresholds ({high:g} and {low:g})\n')

    rows = []
    for top_n in (1, 5, 10, 20):
        a = (joined['best_correct_rank_high'].between(0, top_n - 1)).to_numpy()
        b = (joined['best_correct_rank_low'].between(0, top_n - 1)).to_numpy()
        gained, lost = int((~a & b).sum()), int((a & ~b).sum())
        p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
        rows.append({'top_n': top_n, f'threshold_{high:g}': a.mean(),
                     f'threshold_{low:g}': b.mean(), 'delta_pp': (b.mean() - a.mean()) * 100,
                     'gained': gained, 'lost': lost, 'p_mcnemar': p})
    table = pd.DataFrame(rows)

    seconds = {t: frames[t]['seconds'].sum() for t in thresholds}
    per_pattern = {t: frames[t]['seconds'].mean() for t in thresholds}
    pool = {t: frames[t]['pool_size'].mean() for t in thresholds}

    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifact_dir / f'S03_confirm_topn_{args.arm}.csv', index=False)
    joined.to_csv(artifact_dir / f'S03_confirm_per_entry_{args.arm}.csv', index=False)

    print(table.to_string(index=False))
    print(f'\nwall clock per pattern: {high:g} -> {per_pattern[high]:.2f}s, '
          f'{low:g} -> {per_pattern[low]:.2f}s '
          f'({(per_pattern[low]/per_pattern[high]-1)*100:+.1f} %)')
    print(f'ranked pool per pattern: {high:g} -> {pool[high]:.0f}, {low:g} -> {pool[low]:.0f}')
    print(f'\nwrote {artifact_dir}/S03_confirm_topn_{args.arm}.csv')


def run_figure(args):
    """The campaign's central picture: what the cut delivers, and what the ranking loses.

    Two bars per cut. The lower one is what a user gets; the upper is what was available for them
    to get. The gap between them is the prize a better merit is competing for, and the point of
    the figure is that lowering the cut grows the gap rather than the outcome.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'axes.axisbelow': True, 'grid.alpha': 0.25,
                         'grid.linewidth': 0.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                         'axes.linewidth': 0.7})

    thresholds = sorted((float(v) for v in args.thresholds.split(',')), reverse=True)
    arms = [a for a in ('general', 'hard')
            if (Path(BASE) / args.out_root / a / f't{thresholds[0]:g}').exists()]
    figure, axes = plt.subplots(1, len(arms), figsize=(3.3 * len(arms), 3.2), squeeze=False)

    for column, arm in enumerate(arms):
        axis = axes[0][column]
        available, ranked, labels = [], [], []
        for threshold in thresholds:
            directory = Path(BASE) / args.out_root / arm / f't{threshold:g}'
            frame = pd.concat([pd.read_parquet(s) for s in sorted(directory.glob('ranked_*.parquet'))],
                              ignore_index=True)
            available.append(int((frame['best_correct_rank'] >= 0).sum()))
            ranked.append(int(frame['best_correct_rank'].between(0, 9).sum()))
            labels.append(f'cut {threshold:g}')
            total = frame.shape[0]

        position = np.arange(len(thresholds))
        axis.bar(position, available, 0.52, color='#CFE3F2', edgecolor='#0B5D91', linewidth=0.7,
                 label='correct cell anywhere in the printed list', zorder=2)
        axis.bar(position, ranked, 0.52, color='#0B5D91', edgecolor='#0B5D91', linewidth=0.7,
                 label='correct cell in the top 10 (what the user gets)', zorder=3)
        for x, (a, r) in enumerate(zip(available, ranked)):
            axis.annotate(f'{a}', (x, a), textcoords='offset points', xytext=(0, 3),
                          ha='center', fontsize=7, color='#0B5D91')
            axis.annotate(f'{r}', (x, r), textcoords='offset points', xytext=(0, 3),
                          ha='center', fontsize=7, color='white' if r > 0.08 * max(available)
                          else '#0B5D91')
            if a > r:
                axis.annotate('', xy=(x + 0.33, r), xytext=(x + 0.33, a),
                              arrowprops=dict(arrowstyle='<->', color='#C1571A', lw=0.8))
                axis.annotate(f'{a - r} unranked', (x + 0.36, (a + r) / 2), fontsize=6.5,
                              color='#C1571A', va='center', ha='left')
        axis.set_xticks(position)
        axis.set_xticklabels(labels)
        axis.set_xlim(-0.5, len(thresholds) + 0.15)
        axis.set_ylim(0, max(available) * 1.16)
        # Per panel: the two arms have different denominators (210 and 972), so one shared
        # label would misstate one of them.
        axis.set_ylabel(f'pattern-conditions (of {total})')
        axis.set_title('general population' if arm == 'general'
                       else 'hard stratum', fontsize=8.5, pad=6)
    axes[0][0].legend(loc='upper left', frameon=False, bbox_to_anchor=(0.0, -0.12), ncol=1)
    figure.suptitle('Lowering the cut delivers the answer; the merit fails to rank it',
                    fontsize=9.5, y=1.02)
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / 'S03_confirm_available_vs_ranked.png'
    figure.savefig(destination)
    plt.close(figure)
    print(f'wrote {destination}')


def main():
    args = _parse_args()
    {'report': run_report, 'figure': run_figure}.get(args.stage, run_arm)(args)


if __name__ == '__main__':
    main()
