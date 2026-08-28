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
    parser.add_argument('--stage', choices=['run', 'cuts', 'report', 'figure'], default='run')
    parser.add_argument('--threshold', type=float, default=5.0)
    parser.add_argument('--threshold-map', default=None,
                        choices=['equal_share', 'lower_only'],
                        help='run stage: use the per-lattice mapping of this policy instead of '
                             '--threshold. Requires --stage cuts to have been run for this arm')
    parser.add_argument('--thresholds', default='5.0,3.0', help='report stage: the two arms')
    parser.add_argument('--cuts-from', default=None, choices=sorted(ARMS),
                        help='run stage: take the per-lattice cuts from this arm instead of the '
                             'one being run. The hard arm carries only three true Bravais '
                             'lattices (C2-R-002), so a policy derived there is fitted to its own '
                             'answer; deriving on `general` and applying to `hard` is the '
                             'out-of-sample form and is what the result should be read from')
    parser.add_argument('--arms', default=None,
                        help='report stage: comma-separated directory tags to compare, e.g. '
                             't5,t3,map-equal_share. Overrides --thresholds when given')
    parser.add_argument('--arm', default='general', choices=sorted(ARMS))
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--out-root', default=OUT_ROOT)
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--shard-stride', type=int, default=1)
    parser.add_argument('--shard-offset', type=int, default=0)
    parser.add_argument('--limit-entries', type=int, default=None)
    parser.add_argument('--split', default=None,
                        help='run stage: restrict to one split, e.g. fom-dev. On the hard arm '
                             'fom-train is 8.1 of 10.8 search-hours and no reported number comes '
                             'from it. Output goes to a split-tagged directory so a later full '
                             'run does not collide with it')
    return parser.parse_args()


MERIT_ROOT = os.path.join('mlindex', 'data', 'fom_prune_criterion')

# The production cut every arm here is measured against, and the value the equal-share policy
# matches its pooled surviving fraction to. Not a flag: 5.0 is what ships (decisions log,
# 2026-08-26) and the point of these arms is to be read against it.
BASELINE_CUT = 5.0


def derive_cuts(arm, policy, baseline=BASELINE_CUT, split='fom-train'):
    """Per-lattice M20 cuts from one rule with one free parameter, fitted on `fom-train` only.

    C2-F-033 found that a single global threshold is not a neutral quality filter: it is also an
    accidental CROSS-LATTICE filter, because M20's scale differs by lattice. At 5.0 the surviving
    share runs oF 1.3 % to cI 6.1 %, a 4.6x spread (C2-F-023), and lowering the cut globally
    switches that second function off faster than it delivers the first -- wrong-lattice pools
    regrow 171x against 31x for the right ones.

    Two policies, both one-parameter, neither fitted per lattice. Fourteen fitted cuts do not
    survive a train/dev split at these entry counts (C2-F-023, C2-R-002, C2-R-003), which is why
    neither of these is a fit:

      equal_share  every lattice keeps the same FRACTION of its own threshold-0 pool that the
                   global cut keeps overall. This is the direct undoing of the accidental filter:
                   it LOWERS the cut on the lattices the global value treats harshly and RAISES
                   it on the ones it treats leniently, so the pooled surviving count -- and
                   therefore the wall clock, which is the other half of the trade -- is matched to
                   the baseline by construction. The one parameter is the target share, and it is
                   not free either: it is read off the baseline.

      lower_only   the literal form the work order proposes -- hold the baseline everywhere and
                   lower it only where the cut is harsher than the pooled share. Strictly more
                   permissive than the baseline, so it is NOT cost-matched, and any gain it shows
                   is confounded with simply having a bigger pool. Run as the companion to
                   equal_share, not instead of it.

    **Selected on `fom-train` and never on the split the arms are reported on** (PROTOCOL section
    8). Uses no correctness label: the quantile is of the unlabelled candidate pool, so this is a
    rule that could be applied at inference on a pattern whose answer is unknown.
    """
    shards = sorted(Path(BASE).joinpath(MERIT_ROOT, arm).glob('merits_*.parquet'))
    if not shards:
        raise SystemExit(f'no merit shards under {MERIT_ROOT}/{arm}')
    columns = ['bravais_lattice', 'm20_at_prune', 'split']
    per_lattice = {}
    for shard in shards:
        frame = pd.read_parquet(shard, columns=columns)
        frame = frame.loc[frame['split'] == split]
        for lattice, part in frame.groupby('bravais_lattice', sort=False):
            per_lattice.setdefault(lattice, []).append(
                part['m20_at_prune'].to_numpy(dtype=np.float64))
    per_lattice = {k: np.concatenate(v) for k, v in per_lattice.items()}
    if not per_lattice:
        raise SystemExit(f'no {split} rows in {MERIT_ROOT}/{arm}')

    total = sum(v.size for v in per_lattice.values())
    kept = sum(int((v >= baseline).sum()) for v in per_lattice.values())
    target = kept/total

    rows, cuts = [], {}
    for lattice in sorted(per_lattice):
        values = per_lattice[lattice]
        share = float((values >= baseline).mean())
        # The cut that keeps exactly `target` of this lattice's own pool. `1 - target` because
        # the quantile is taken from below and the cut keeps the upper tail.
        equal = float(np.quantile(values, 1.0 - target))
        cut = equal if policy == 'equal_share' else min(equal, baseline)
        cuts[lattice] = round(cut, 4)
        rows.append({'arm': arm, 'policy': policy, 'bravais_lattice': lattice,
                     'n_train_candidates': int(values.size),
                     'share_at_baseline': share, 'target_share': target,
                     'cut': cuts[lattice], 'cut_moves': cuts[lattice] - baseline})
    return cuts, pd.DataFrame(rows), target


def run_cuts(args):
    """Derive both policies' cuts and write them where the run stage can read them."""
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tables, payload = [], {}
    for policy in ('equal_share', 'lower_only'):
        cuts, table, target = derive_cuts(args.arm, policy)
        payload[policy] = {'arm': args.arm, 'baseline': BASELINE_CUT, 'target_share': target,
                           'split': 'fom-train', 'cuts': cuts}
        tables.append(table)
        print(f'\n=== {policy}: target share {target:.5f} of each lattice own pool')
        print(table[['bravais_lattice', 'share_at_baseline', 'cut', 'cut_moves']]
              .to_string(index=False))
    pd.concat(tables, ignore_index=True).to_csv(
        artifact_dir / f'INTERIM_per_lattice_cuts_{args.arm}.csv', index=False)
    destination = artifact_dir / f'INTERIM_per_lattice_cuts_{args.arm}.json'
    with open(destination, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f'\nwrote {destination}')


def threshold_option(args):
    """What goes into `opt_params['prune_m20_threshold']`: a float, or a per-lattice mapping.

    `Candidates.prune_below_m20` takes either. Two other sites coerce the same option with
    `float()` -- `_record_candidate_dump` and the `at_prune` branch of `_downsample_computation`
    -- so a mapping would raise there; both are dump-only paths this harness does not enable.
    See C2-F-063 and C2-Q-022.
    """
    if not args.threshold_map:
        return float(args.threshold), f't{args.threshold:g}'
    source = args.cuts_from or args.arm
    path = Path(BASE) / args.artifact_dir / f'INTERIM_per_lattice_cuts_{source}.json'
    if not path.exists():
        raise SystemExit(f'{path} missing; run --stage cuts --arm {source} first')
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if args.threshold_map not in payload:
        raise SystemExit(f'{args.threshold_map} not in {path}: {sorted(payload)}')
    tag = f'map-{args.threshold_map}'
    if source != args.arm:
        tag += f'-from-{source}'
    return dict(payload[args.threshold_map]['cuts']), tag


def run_arm(args):
    from mlindex.optimization.MPOptimizer import setup_mp_optimizers, shutdown_mp_workers

    root = Path(BASE) / ARMS[args.arm]
    threshold, tag = threshold_option(args)
    if args.split:
        tag += f'-{args.split}'
    out_dir = Path(BASE) / args.out_root / args.arm / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    options = {'prune_m20_threshold': threshold}
    optimizers, processes, task_queues = setup_mp_optimizers(
        args.processes, BROADENING_TAG, n_candidates_scale=1, seed=12345, options=options)

    jobs = [(bundle, shard)
            for bundle, bundle_dir in bundle_directories(root).items()
            for shard in sorted(bundle_dir.glob('entries_*.parquet'))]
    jobs = jobs[args.shard_offset::args.shard_stride]
    print(f'{tag}: {len(jobs)} shards for offset {args.shard_offset} '
          f'of stride {args.shard_stride}', flush=True)
    if isinstance(threshold, dict):
        print('  per-lattice cuts: '
              + ', '.join(f'{k} {v:g}' for k, v in sorted(threshold.items())), flush=True)

    started = time.time()
    try:
        for bundle, shard in jobs:
            tag = shard.stem.split('_', 1)[1]
            destination = out_dir / f'ranked_{tag}.parquet'
            if destination.exists():
                print(f'  {bundle}/{tag}: already done, skipping', flush=True)
                continue
            entries = pd.read_parquet(shard)
            if args.split:
                # The reported contrast is read on `fom-dev` (PROTOCOL section 8), and on the hard
                # arm `fom-train` is 8.1 of the 10.8 search-hours for patterns no number comes
                # from. Restricting is a cost decision, not a validity one -- the cuts were derived
                # on the GENERAL arm's train split, so nothing here is contaminated either way --
                # and it can be lifted by re-running without the flag, which resumes rather than
                # repeats because finished shards are skipped.
                entries = entries.loc[entries['split'] == args.split]
            if args.limit_entries:
                entries = entries.head(args.limit_entries)
            if not entries.shape[0]:
                print(f'  {bundle}/{tag}: no rows after filters, skipping', flush=True)
                continue
            rows = []
            for _, entry in entries.iterrows():
                # Per-pattern wall clock, which is the cost half of the trade the cut makes.
                # Timed around the whole fourteen-lattice pass, as a user experiences it.
                at = time.perf_counter()
                record = ranked_pool(optimizers, task_queues, entry, threshold)
                rows.append(dict(record, entry_id=entry['entry_id'], condition_bundle=bundle,
                                 split=entry['split'], threshold=tag,
                                 seconds=time.perf_counter() - at))
            pd.DataFrame(rows).to_parquet(destination, index=False)
            print(f'  {bundle}/{tag}: {len(rows)} patterns, '
                  f'{sum(r["seconds"] for r in rows):.0f}s of search, '
                  f'{time.time() - started:.0f}s elapsed', flush=True)
    finally:
        shutdown_mp_workers(processes, task_queues)
    print(f'wrote {out_dir}')


def _arm_tags(args):
    """The directory tags to compare, baseline first.

    `--thresholds` keeps S03's two-scalar form working unchanged; `--arms` is the general form and
    is what a per-lattice arm needs, because its directory is named for a policy rather than a
    number. Baseline first in both cases: every contrast is read against the shipped cut.
    """
    if args.arms:
        return [tag.strip() for tag in args.arms.split(',') if tag.strip()]
    thresholds = sorted((float(v) for v in args.thresholds.split(',')), reverse=True)
    return [f't{t:g}' for t in thresholds]


def _load_arm(args, tag):
    directory = Path(BASE) / args.out_root / args.arm / tag
    shards = sorted(directory.glob('ranked_*.parquet'))
    if not shards:
        raise SystemExit(f'no output under {directory}')
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def run_report(args):
    """Every arm against the first one, paired over the patterns both ran.

    PAIRED, and the pairing is the whole point: both arms saw the same generated candidates for
    the same (entry, Bravais lattice) seed, so McNemar over the discordant patterns is testing the
    cut and nothing else. An unpaired comparison here would be comparing two searches.
    """
    from scipy.stats import binomtest

    tags = _arm_tags(args)
    if len(tags) < 2:
        raise SystemExit(f'need at least two arms to compare, got {tags}')
    frames = {tag: _load_arm(args, tag) for tag in tags}
    baseline = tags[0]
    key = ['condition_bundle', 'entry_id']

    rows = []
    for tag in tags[1:]:
        joined = frames[baseline].merge(frames[tag], on=key, suffixes=('_base', '_arm'))
        print(f'{tag} vs {baseline}: {joined.shape[0]} patterns run at both')
        for top_n in (1, 5, 10, 20):
            a = joined['best_correct_rank_base'].between(0, top_n - 1).to_numpy()
            b = joined['best_correct_rank_arm'].between(0, top_n - 1).to_numpy()
            gained, lost = int((~a & b).sum()), int((a & ~b).sum())
            p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
            rows.append({'arm': tag, 'baseline': baseline, 'top_n': top_n,
                         'baseline_rate': a.mean(), 'arm_rate': b.mean(),
                         'delta_pp': (b.mean() - a.mean())*100,
                         'gained': gained, 'lost': lost, 'p_mcnemar': p,
                         'n_patterns': int(joined.shape[0])})
        # Availability separately from ranking: "no correct candidate in the pool" is a generation
        # failure and "it was there and ranked below N" is a ranking failure, and PROTOCOL section 8
        # says to keep the buckets apart. A cut acts on the first; only the first is its own doing.
        avail_base = (joined['best_correct_rank_base'] >= 0).to_numpy()
        avail_arm = (joined['best_correct_rank_arm'] >= 0).to_numpy()
        gained, lost = int((~avail_base & avail_arm).sum()), int((avail_base & ~avail_arm).sum())
        rows.append({'arm': tag, 'baseline': baseline, 'top_n': -1,
                     'baseline_rate': avail_base.mean(), 'arm_rate': avail_arm.mean(),
                     'delta_pp': (avail_arm.mean() - avail_base.mean())*100,
                     'gained': gained, 'lost': lost,
                     'p_mcnemar': binomtest(gained, gained + lost, 0.5).pvalue
                     if gained + lost else 1.0,
                     'n_patterns': int(joined.shape[0])})

    table = pd.DataFrame(rows)
    table['metric'] = np.where(table['top_n'] < 0, 'available', 'top' + table['top_n'].astype(str))

    # Cost is PAIRED too, and it has to be. An arm run on a subset -- `--split fom-dev`, say --
    # otherwise gets compared against a baseline averaged over patterns it never ran, and the
    # difference then carries whatever makes those patterns faster or slower rather than the cut.
    # So each arm is priced on the patterns it shares with the baseline, and the baseline is
    # repriced on that same subset for every row.
    cost_rows = []
    for tag in tags:
        joined = frames[baseline].merge(frames[tag], on=key, suffixes=('_base', '_arm'))
        cost_rows.append({
            'arm': tag, 'n_patterns_paired': int(joined.shape[0]),
            'baseline_seconds': joined['seconds_base'].mean(),
            'seconds_per_pattern': joined['seconds_arm'].mean(),
            'baseline_pool': joined['pool_size_base'].mean(),
            'pool_per_pattern': joined['pool_size_arm'].mean(),
            'seconds_vs_baseline': (joined['seconds_arm'].mean()
                                    / joined['seconds_base'].mean() - 1)*100,
            'pool_vs_baseline': (joined['pool_size_arm'].mean()
                                 / joined['pool_size_base'].mean() - 1)*100})
    cost = pd.DataFrame(cost_rows)

    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = 'S03_confirm' if not args.arms else 'INTERIM_per_lattice_confirm'
    table.to_csv(artifact_dir / f'{stem}_topn_{args.arm}.csv', index=False)
    cost.to_csv(artifact_dir / f'{stem}_cost_{args.arm}.csv', index=False)
    for tag in tags[1:]:
        frames[baseline].merge(frames[tag], on=key, suffixes=('_base', '_arm')).to_csv(
            artifact_dir / f'{stem}_per_entry_{args.arm}_{tag}.csv', index=False)

    print()
    print(table[['arm', 'metric', 'baseline_rate', 'arm_rate', 'delta_pp',
                 'gained', 'lost', 'p_mcnemar']].to_string(index=False))
    print()
    print(cost.to_string(index=False))
    print(f'\nwrote {artifact_dir}/{stem}_topn_{args.arm}.csv')


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
    {'cuts': run_cuts, 'report': run_report, 'figure': run_figure}.get(args.stage, run_arm)(args)


if __name__ == '__main__':
    main()
