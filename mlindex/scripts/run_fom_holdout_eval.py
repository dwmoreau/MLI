"""S10b: rank the classical merits on the peaks the cell was never fitted to, and sweep the budget.

    # everything, on the slice
    python mlindex/scripts/run_fom_holdout_eval.py --pool mlindex/data/fom_benchmark_c2 \\
        --tag S10b_holdout_slice

    # the two stages apart, which is what a cluster run wants
    python mlindex/scripts/run_fom_holdout_eval.py --pool $SCRATCH/fom_campaign2/pool \\
        --tag S10b_holdout --reduce
    python mlindex/scripts/run_fom_holdout_eval.py --tag S10b_holdout --analyse

Written the way `run_fom_zoo_eval.py` is written, and for the same reason: the reduction is one row
per (entry, condition) and is a sufficient statistic for every threshold and metric downstream, so
a 122 GB pool reduces on NERSC and analyses wherever the record lives. S10b's own pitfall list asks
for this shape explicitly, so that the pool-scale pass over the full benchmark is one `sbatch`.

**One pool pass for the whole sweep, not one per column.** The zoo drives
`FomMetrics.reduce_to_per_entry` once per merit, which is right for seven columns and wrong for
this step: six merits over six peak budgets is 36 score columns, and 36 reads of a fully retained
pool is hours of I/O to answer a question that needs one. So the shard loop is written out here and
`FomMetrics.reduce_pool` is called once per column per shard, on a frame that is read once. Every
guard the metrics module applies is still applied -- `rank_exactness` is consulted per column and
refuses exactly as `reduce_to_per_entry` would, `_prepare_shard` drops the control bundles, and
`_combine_reductions` refuses a shard set that was not pooled across all fourteen lattices.

**This loop belongs in `FomMetrics` and is here on purpose.** A second session was editing
`FomMetrics.py` in this checkout while S10b ran, and PROTOCOL section 5 records a silent revert and
a four-function mis-sweep from exactly that. It moves in once the tree is quiet; C2-Q-027.

Three things the sweep has to carry that a merit table does not, all of them from S10a:

  * **Applicability.** An entry whose stored surplus is shorter than the budget is **missing, not
    zero** at that budget. It is dropped from the paired comparison and the count that dropped is
    reported. `ho_*` is null for exactly those rows, and null would otherwise rank last and read
    as a uniformly terrible merit.
  * **`M_rev` support coverage.** `get_M_rev_sym` returns 0.0, not null, below ten reference
    lines, so a floored `ho_M_sym` is a *defined* value that means nothing. The sidecar stores
    `ho_N_cal` per budget so the floor is auditable after the fact (C2-Q-017), and this driver
    reports the supported fraction beside every `ho_M_sym` number, per lattice, at every budget.
  * **Reference reach.** `ho_ref_reach` is 0 where the candidate's extinction-group reference list
    stops short of the peak being asked about, so `ho_*` there measures `hkl_ref_length` rather
    than crystallography. Cubic is the only lattice where this bites (C2-F-101).

`ho_tail_nll` is **extensive** -- it grew 27.75x from one surplus peak to twenty (S10a gate 3) --
so it is reported at a fixed budget and is refused entry to the sweep by `SWEEP_MERITS`.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.model_training import FomMetrics  # noqa: E402

# The merits that may be swept. S10a gate 3 measured each one's growth from 1 surplus peak to 20:
# anything extensive would have its growth read as a gain, so `ho_tail_nll` (27.75x) is excluded
# here rather than filtered downstream. `ho_Minfo` is already emitted as a per-peak mean, which is
# what makes it admissible at all.
SWEEP_MERITS = ('ho_M20', 'ho_M_tilde', 'ho_M_rev', 'ho_M_sym', 'ho_Minfo', 'ho_M', 'ho_raw')

# Reported at a fixed budget, never swept.
FIXED_BUDGET_MERITS = ('ho_tail_nll',)

# The in-sample incumbent, and the thing every hold-out number has to beat to be worth anything.
# It is a stored pool column, it is in `RANK_EXACT_MERITS`, and campaign 1's +7.11 pp was measured
# against exactly it -- on a differently constructed hold-out set, which is why the delta from that
# number is reported with its reason rather than as the same measurement (INHERITED R13/F-097).
INSAMPLE_ANCHOR = 'M20'

MERIT_NOTES = {
    'ho_M20': 'de Wolff M20 on the surplus -- campaign 1s column, and the anchor here',
    'ho_M_tilde': 'O-T (i)+(ii) on the surplus',
    'ho_M_rev': 'O-T (iii) reversed on the surplus -- needs a populated interval',
    'ho_M_sym': 'O-T symmetric on the surplus -- S09s winner in sample',
    'ho_Minfo': 'Taupin Minfo, per-peak mean -- the statistic that restricts to k peaks',
    'ho_M': 'the predictive form, normalised',
    'ho_raw': 'median |dQ| on the surplus -- lower is better',
    'ho_tail_nll': 'summed negative log-likelihood -- EXTENSIVE, fixed budget only',
    'M20': 'the in-sample incumbent, scored on the fitted window',
    }

# Diagnostics the sidecar carries that are not merits. `FomMetrics.HIGHER_IS_BETTER` deliberately
# refuses to give these a direction, so ranking on one raises rather than silently producing a
# number (C2-F-085's lesson, applied ahead of time).
DIAGNOSTIC_NAMES = ('ho_N_cal', 'ho_n_scored', 'ho_ref_reach')

# `get_M_rev_sym`'s support floor. Below this many reference lines in the counting window the merit
# is undefined and returns 0.0; the stored `ho_N_cal` is what makes that distinguishable afterwards.
MIN_N_CAL = 10

# What the reduction needs off the candidate row, beyond the merit columns themselves.
POOL_COLUMNS = tuple(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + (
    'condition_bundle', 'M20', 'lattice_system', 'n_peaks',
    )


def commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE, text=True).strip()
    except Exception:
        return 'unknown'


def dirty_tree():
    """Whether the working tree carries uncommitted changes, recorded beside the commit.

    S10a and S10b both ran against a checkout two sessions were editing, so a bare commit hash
    would misdescribe the code that produced these numbers.
    """
    try:
        return bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=BASE,
                                            text=True).strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------------------
# Applicability -- which entries can be scored at which budget, read off the entry table
# ---------------------------------------------------------------------------------------
def surplus_lengths(pool):
    """One row per (entry, condition): how many surplus peaks it actually stored.

    Read from the entry table rather than inferred from a null merit, so the reason an entry is
    absent at a budget is a fact about the pattern rather than an artefact of the scorer. This is
    the denominator every applicability number in the sweep is reported against.
    """
    entries = FomBenchmark.load_entries(pool)
    keys = ['entry_id', 'condition_bundle'] if 'condition_bundle' in entries.columns \
        else ['entry_id']
    out = entries[keys].copy()
    out['n_surplus'] = entries['q2_holdout'].apply(len).to_numpy()
    return out


# ---------------------------------------------------------------------------------------
# The reduce stage
# ---------------------------------------------------------------------------------------
def sweep_columns(pool, merit_dir, merits, budgets):
    """The `ho_*__nK` columns that actually exist in this pool's sidecars.

    A budget the sidecar was not written at is skipped with a message rather than producing a
    column of nulls that would rank last and read as a merit scoring zero everywhere.
    """
    sidecars = sorted(Path(merit_dir).glob('candidates*.parquet'))
    if not sidecars:
        raise SystemExit(
            f'No hold-out sidecars in {merit_dir}. Write them first:\n'
            f'  python mlindex/scripts/run_fom_holdout_merits.py --pool {pool} --processes 8')
    present = set(pd.read_parquet(sidecars[0]).columns)
    wanted, missing = [], []
    for budget in budgets:
        for merit in merits:
            name = FomBenchmark.holdout_column(merit, budget)
            (wanted if name in present else missing).append(name)
    if missing:
        print(f'  note: {len(missing)} column(s) absent from the sidecars and skipped, '
              f'e.g. {missing[:3]}')
    return wanted


def coverage_counts(frame, budgets):
    """Candidate-level support and reach counts, per (bundle, lattice, budget).

    Both gates S10a measured, recomputed on the full population rather than on its one-row-group
    sample, and carried through the reduction so the sweep can put coverage beside every number
    without a second pool pass (PROTOCOL section 3 rule 8).
    """
    rows = []
    lattice = frame['bravais_lattice'].to_numpy()
    bundle = frame['condition_bundle'].to_numpy()
    for budget in budgets:
        n_cal = FomBenchmark.holdout_column('ho_N_cal', budget)
        reach = FomBenchmark.holdout_column('ho_ref_reach', budget)
        if n_cal not in frame.columns:
            continue
        values = frame[n_cal].to_numpy(dtype=np.float64)
        scored = np.isfinite(values)
        supported = scored & (values >= MIN_N_CAL)
        reached = (scored & (frame[reach].to_numpy(dtype=np.float64) > 0.5)
                   if reach in frame.columns else scored)
        block = pd.DataFrame({
            'condition_bundle': bundle, 'bravais_lattice': lattice,
            'n_scored': scored.astype(np.int64),
            'n_mrev_supported': supported.astype(np.int64),
            'n_ref_reached': reached.astype(np.int64),
            'n_candidates': np.ones(scored.size, dtype=np.int64),
            })
        block = block.groupby(['condition_bundle', 'bravais_lattice'], as_index=False).sum()
        block.insert(0, 'n_extra', budget)
        rows.append(block)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_reduce(args):
    """One pass over the pool; every score column reduced from the same shard.

    Returns nothing -- it writes the stacked per-entry reduction and the coverage table, which is
    everything `--analyse` needs.
    """
    pool = Path(args.pool)
    merit_dir = Path(args.merit_dir) if args.merit_dir else pool/'holdout_merits'
    artifact_dir = Path(args.artifact_dir)
    budgets = tuple(args.n_extra) if args.n_extra else FomBenchmark.HOLDOUT_N_EXTRA
    merits = tuple(args.merits) if args.merits else SWEEP_MERITS + FIXED_BUDGET_MERITS

    columns = sweep_columns(pool, merit_dir, merits, budgets)
    diagnostics = [FomBenchmark.holdout_column(name, budget)
                   for budget in budgets for name in DIAGNOSTIC_NAMES]
    scores = list(columns) + [INSAMPLE_ANCHOR]

    entries = FomBenchmark.load_entries(pool)
    depth, subsampled = FomBenchmark.subsample_depth(pool)
    splits = {}
    for label in (args.train_split, args.report_split):
        ids = set(entries.loc[entries['split'] == label, 'entry_id'])
        if ids:
            splits[label] = ids
    if not splits:
        raise SystemExit(f'{pool} carries neither {args.train_split} nor {args.report_split}')
    print(f'{pool}: ' + ' / '.join(f'{len(v):,} {k}' for k, v in splits.items())
          + f' source entries; K={depth}, subsampled={subsampled}')

    # The exactness question is per score, and it is asked before any work is done. `ho_*` is
    # outside `RANK_EXACT_MERITS`, so on a subsampled pool every one of these refuses unless the
    # caller passed --allow-inexact-ranks, which is the intended behaviour: an optimistic rank is
    # indistinguishable from a good one (C2-R-013, METRICS section 1).
    exactness = {}
    for score in scores:
        base = score.split('__n')[0] if '__n' in score else score
        exact, reason = FomMetrics.rank_exactness(
            base if base in FomMetrics.RANK_EXACT_MERITS else score,
            args.top_n, depth if subsampled else None, subsampled)
        exactness[score] = (exact, reason)
        if not exact and not args.allow_inexact_ranks:
            raise SystemExit(
                f'Refusing to reduce {score!r} on this pool. {reason}\n'
                f'Rank claims for the hold-out family come from the fully retained pool. Pass '
                f'--allow-inexact-ranks only to produce THRESHOLD claims on the slice, and label '
                f'every table it feeds (S10b section 2).')

    orientation = {}
    for score in scores:
        orientation[score] = (FomMetrics.holdout_orientation_of(score) if score.startswith('ho_')
                              else FomMetrics.orientation_of(score))

    context = FomMetrics.entry_context(entries, hard_min_decile=args.hard_min_decile)
    degenerate = FomMetrics._degenerate_entries(entries)
    projection = list(POOL_COLUMNS)

    accumulated = {(score, split): [] for score in scores for split in splits}
    coverage, seen = [], 0
    started = time.perf_counter()
    for frame in FomBenchmark.bundle_frames(pool, merit_dir=merit_dir, columns=projection,
                                            require_merits=True):
        frame = FomMetrics._prepare_shard(frame, args.include_control, degenerate)
        if frame is None:
            continue
        # A bundle filter, for smoke-testing the assembly without paying for the whole pool. It is
        # NOT a reporting knob: dropping bundles changes the population every number is over, and
        # the meta records what survived so a filtered run cannot be mistaken for a full one.
        if args.bundles and frame['condition_bundle'].iloc[0] not in set(args.bundles):
            continue
        coverage.append(coverage_counts(frame, budgets))
        for split, ids in splits.items():
            shard = frame.loc[frame['entry_id'].isin(ids)]
            if not shard.shape[0]:
                continue
            for score in scores:
                if score not in shard.columns:
                    continue
                values = FomMetrics._shard_scores(shard, score, orientation[score])
                accumulated[(score, split)].append(
                    FomMetrics.reduce_pool(shard, values, pool=args.pool_mode))
        seen += frame.shape[0]
        print(f'  {frame["condition_bundle"].iloc[0]:28s} {frame.shape[0]:>9,} candidates '
              f'({time.perf_counter() - started:.0f} s)', flush=True)

    surplus = surplus_lengths(pool)
    stacked, metas = [], {}
    for (score, split), reductions in accumulated.items():
        if not reductions:
            continue
        per_entry = context.merge(FomMetrics._combine_reductions(reductions),
                                  on=['entry_id', 'condition_bundle'], how='inner',
                                  validate='1:1')
        per_entry = per_entry.merge(surplus, on=[c for c in ('entry_id', 'condition_bundle')
                                                 if c in surplus.columns], how='left')
        base, _, suffix = score.partition('__n')
        budget = int(suffix) if suffix else None
        # Missing, not zero. An entry that stored fewer surplus peaks than the budget was never
        # scored at it, so it leaves the population for that budget rather than counting as a
        # failure -- and the count that left is `n_dropped_short` on the meta.
        if budget is not None:
            keep = per_entry['n_surplus'].to_numpy() >= budget
            dropped = int((~keep).sum())
            per_entry = per_entry.loc[keep].reset_index(drop=True)
        else:
            dropped = 0
        # `entry_context` already carries `split`, so these are assigned rather than inserted --
        # `DataFrame.insert` raises on a name that exists, and losing a ten-minute pool pass to it
        # at the assembly step is not a mistake worth making twice.
        for name, value in (('n_extra', budget if budget is not None else -1),
                            ('merit', base), ('split', split)):
            per_entry[name] = value
        per_entry = per_entry[[name for name in ('split', 'merit', 'n_extra')]
                              + [c for c in per_entry.columns
                                 if c not in ('split', 'merit', 'n_extra')]]
        stacked.append(per_entry)
        exact, reason = exactness[score]
        metas[f'{score}|{split}'] = dict(
            score=base, n_extra=budget, split=split, higher_is_better=bool(orientation[score]),
            pool=args.pool_mode, reduced_top_n=int(args.top_n),
            subsample_top_k=(int(depth) if subsampled and depth is not None else None),
            subsampled=subsampled, ranks_exact=bool(exact), rank_exactness=reason,
            n_dropped_short=dropped, n_entries=int(per_entry.shape[0]),
            bundles_excluded=sorted(FomMetrics.CONTROL_BUNDLES) if not args.include_control else [],
            hard_min_decile=int(args.hard_min_decile), source=str(pool),
            )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(stacked, ignore_index=True).to_parquet(
        artifact_dir/f'{args.tag}_reduced.parquet', index=False)
    pd.concat(coverage, ignore_index=True).groupby(
        ['n_extra', 'condition_bundle', 'bravais_lattice'], as_index=False).sum().to_csv(
            artifact_dir/f'{args.tag}_coverage.csv', index=False, encoding='utf-8')
    (artifact_dir/f'{args.tag}_reduced_meta.json').write_text(
        json.dumps({'commit': commit_hash(), 'dirty_tree': dirty_tree(), 'pool': str(pool),
                    'merit_dir': str(merit_dir), 'n_candidates_seen': int(seen),
                    'budgets': list(budgets), 'bundles_filter': args.bundles,
                    'reductions': metas},
                   indent=2, sort_keys=True, default=str), encoding='utf-8')
    print(f'\n{seen:,} candidates -> {len(metas)} reductions in {artifact_dir}/{args.tag}_*')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='S10b -- the classical merits scored out of sample, and the peak budget.')
    parser.add_argument('--pool', default=os.path.join(BASE, 'mlindex', 'data',
                                                       'fom_benchmark_c2'),
                        help='Benchmark root. Needed for --reduce only.')
    parser.add_argument('--merit-dir', default=None,
                        help='Hold-out sidecar directory. Default is <pool>/holdout_merits.')
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--merits', nargs='+', default=None,
                        help='Hold-out merits to evaluate. Defaults to the seven sweepable ones '
                             'plus ho_tail_nll at fixed budget.')
    parser.add_argument('--n-extra', type=int, nargs='*', default=None,
                        help='Peak budgets, as surplus-peak counts. n_extra IS the total peak '
                             'budget minus 20, so 5 is a 25-peak pattern. Default is the S10 grid.')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--hard-min-decile', type=int, default=FomMetrics.HARD_MIN_DECILE)
    parser.add_argument('--pool-mode', choices=('cross_bl', 'per_bl'), default='cross_bl',
                        help="'per_bl' ranks within each Bravais lattice. Never the headline.")
    parser.add_argument('--include-control', action='store_true')
    parser.add_argument('--allow-inexact-ranks', action='store_true',
                        help='Proceed on a subsampled pool. The hold-out merits are outside the '
                             'seven the subsampler ranked on, so their ranks there are optimistic '
                             '(C2-R-013). Use for THRESHOLD claims on the slice only, and label '
                             'every table it feeds. Do not pass it on a retained pool -- if it '
                             'refuses there, the pool is wrong, not the flag.')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--anchor', default=INSAMPLE_ANCHOR,
                        help='The in-sample baseline every hold-out merit is paired against.')
    parser.add_argument('--headline-n-extra', type=int, default=5,
                        help='The budget the leaderboard is quoted at. 5 surplus peaks is a '
                             '25-peak pattern, the middle of what real data supplies.')
    parser.add_argument('--threshold-train-tag', default=None,
                        help='A --reduce tag carrying the selection split, for the threshold half. '
                             'Pass the SLICE tag when reporting on the fully retained pool: that '
                             'pool has no fom-train, and the two entry sets are disjoint, which is '
                             'asserted rather than assumed. Selection is restricted to the '
                             "reporting pool's own condition bundles so both halves face the same "
                             'mix. Omit to skip the threshold half entirely.')
    parser.add_argument('--cubic-tag', nargs='+', default=None,
                        help='One or two --reduce tags produced with a cubic free-peaks '
                             'definition, compared here as PAIRED arms within cF/cI/cP '
                             '(C2-Q-026). Pass both the fixed-pattern-length arm and the '
                             'equal-count control: the first is the result and the second is what '
                             'separates "more peaks" from "earlier peaks". A tag containing '
                             '"equal" is labelled as the control.')
    parser.add_argument('--clean-tag', default=None,
                        help='A --reduce tag produced from --no-contaminate sidecars. The '
                             'difference is what contamination costs each merit. The clean arm is '
                             'a diagnostic upper bound, not a competing result: a deployed indexer '
                             'cannot know which observed peaks are spurious.')
    parser.add_argument('--bundles', nargs='+', default=None,
                        help='Restrict the pool pass to these condition bundles. For smoke-testing '
                             'the assembly cheaply, not for reporting: it changes the population.')
    parser.add_argument('--reduce', action='store_true', help='Pool pass only.')
    parser.add_argument('--analyse', action='store_true', help='Analysis only.')
    parser.add_argument('--tag', default='S10b_holdout')
    args = parser.parse_args(argv)

    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    both = not (args.reduce or args.analyse)
    if args.reduce or both:
        run_reduce(args)
    if args.analyse or both:
        from mlindex.model_training import FomHoldoutReport
        return FomHoldoutReport.run_analyse(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
