"""S09: rank the classical merits on Benchmark B, and say where the ordering changes.

Ported from campaign 1's `run_fom_zoo_eval.py` (branch `fom`) and cut hard, per PROTOCOL section 3
rule 10. What changed, and why each change was forced rather than chosen:

  * **Seven merits, not twenty-three.** The S00 audit cut the zoo to ten and DWMM cut it again this
    session: the probation merits and the negative control are gone, and `ho_M20` is S10's. What
    remains is exactly `FomBenchmark.REDUCED_MERIT_COLUMNS`, which is exactly the set the negative
    subsampler ranked on -- so every rank metric here is *exact* to the pool's depth K = 200 and no
    column in the main table carries a caveat (C2-F-077, C2-R-013).
  * **No feature matrix.** Campaign 1 ran `compute_all` over the whole pool to build 22 columns.
    Six of the seven are recomputed once into sidecars beside the pool and read from there;
    `run_fom_zoo_features.py` is not ported.
  * **`werner_strict` and `M20 @ 10` are gone.** Neither is a distinct merit, and the two of them
    are why campaign 1's "13 of 21 merits abstain" has a denominator its own table contradicts
    (C2-F-006). Removing them makes the error unrepeatable rather than corrected.
  * **The hard stratum is `fom-dev` alone.** Campaign 1 pooled train+dev for its rank metrics
    because sixteen reachable dev entries cannot rank a zoo. S06 sized this campaign's stratum at
    ~258 reachable, so the licence is not needed and is explicitly not inherited.
  * **Every comparison is paired, including per lattice.** The mask argument `mcnemar` documents
    raised on every call in campaign 1, so no per-stratum paired test was ever run in that project.
    This is its first consumer.

**Two stages, because the pool and the record live on different machines.**

    --reduce    one pass over the pool per merit; writes the per-entry reduction. Needs the pool.
    --analyse   thresholds, tables, McNemar, intervals. Needs only the reductions.

The reduction is one row per (entry, condition) and is a sufficient statistic for every threshold
and every metric, so Benchmark B is reduced on NERSC and analysed wherever the record is. Passing
neither flag does both, which is what a local slice run wants.

    python mlindex/scripts/run_fom_zoo_eval.py --pool mlindex/data/fom_benchmark_c2 \\
        --tag S09_zoo_slice
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

# de Wolff's published threshold. Kept only as the source of the matched false-positive budget --
# the rate M20 itself incurs at 10 on fom-train -- and no longer as a table row of its own.
DEWOLFF_THRESHOLD = 10.0

# The zoo. Direction comes from `FomMetrics.HIGHER_IS_BETTER`, never from a literal here: three of
# the seven are lower-is-better and `evaluate` defaults to True, which is how X_N was ranked
# backwards through every S08 floor table (C2-F-085).
MERIT_NOTES = {
    'M20': 'de Wolff 1968 -- the baseline',
    'M_tilde': 'Oishi-Tomiyasu (i)+(ii): N_cal and a restricted range',
    'M_rev': 'O-T (iii) reversed -- the over-prediction penalty',
    'M_sym': 'O-T symmetric, M_tilde * M_rev -- campaign 1 winner',
    'X_N': 'de Wolff unindexed-line count',
    'n_over': 'calculated lines in range with no observation nearby',
    'max_gap': 'longest run of unaccounted calculated lines',
    }
MERITS = tuple(FomBenchmark.REDUCED_MERIT_COLUMNS)

# The unfloored comparison arm. NOT a member of MERITS and never in the main table: the subsampler
# ranked on the *floored* merit, so on a subsampled pool the candidates that would dominate an
# unfloored ranking were the ones discarded, and the arm comes out flattered (C2-F-084). It is
# evaluated only against a fully retained pool, where `--unfloored` is passed deliberately.
UNFLOORED = 'M_sym_unfloored'

# What the reduction needs off the candidate row. `xnn`, `unit_cell`, `merit_at_prune` and
# `hkl_true_in_basis` are list-valued and become one Python object per row per column in pandas;
# none is needed once the merits are in sidecars.
POOL_COLUMNS = tuple(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + (
    'condition_bundle', 'M20', 'Minfo', 'volume', 'lattice_system', 'n_peaks', 'spacegroup',
    'volume_ratio_to_truth', 'n_entering',
    )


def commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                       text=True).strip()
    except Exception:
        return 'unknown'


def merit_columns(merits, unfloored=False):
    """The sidecar columns to project for a given merit list."""
    known = (set(FomBenchmark.RECOMPUTED_MERIT_COLUMNS)
             | set(FomBenchmark.SOFT_MERIT_COLUMNS))
    wanted = [m for m in merits if m in known]
    if unfloored:
        wanted += ['M_tilde', 'M_rev_unfloored', 'N_cal']
    return sorted(set(wanted))


def pool_frames(pool, merits, entry_ids, merit_dir=None, unfloored=False):
    """Candidate frames for one pool, merits joined from the sidecars, filtered to `entry_ids`.

    `require_merits=True` is not optional here. A sidecar that is missing, or a join that matches
    nothing, leaves the merit column null -- and NaN sorts *last*, so the merit reports as the
    worst in the zoo rather than raising. That is indistinguishable from a measurement.
    """
    columns = list(POOL_COLUMNS) + merit_columns(merits, unfloored=unfloored)
    for frame in FomBenchmark.bundle_frames(pool, merit_dir=merit_dir, columns=columns,
                                            require_merits=True):
        if unfloored:
            frame[UNFLOORED] = frame['M_tilde'].to_numpy()*frame['M_rev_unfloored'].to_numpy()
        if entry_ids is not None:
            frame = frame.loc[frame['entry_id'].isin(entry_ids)]
        if frame.shape[0]:
            yield frame.reset_index(drop=True)


def reduce_one(pool, merit, entry_ids, entries, split_label, merit_dir=None, unfloored=False,
               pool_mode='cross_bl'):
    """One merit, one split: the pool pass. Everything downstream is a function of the result.

    `pool_mode='per_bl'` ranks within each Bravais lattice instead of across all fourteen. It is
    never the headline -- it is a different and much easier problem than the one `run.py` solves --
    and exists to measure the gap, which is where campaign 1 located ~90 % of `M_sym`'s advantage.
    """
    depth, subsampled = FomBenchmark.subsample_depth(Path(pool))
    frames = pool_frames(pool, [merit], entry_ids, merit_dir=merit_dir, unfloored=unfloored)
    return FomMetrics.reduce_to_per_entry(
        frames, score=merit, pool=pool_mode,
        higher_is_better=FomMetrics.orientation_of(merit) if merit in FomMetrics.HIGHER_IS_BETTER
        else True,
        entries=entries, split=split_label,
        # Explicit, never 'auto'. An iterable of frames carries no manifest, so 'auto' would take a
        # subsampled pool for a fully retained one and certify a rank it cannot answer.
        subsample_top_k=depth if subsampled else None,
        )


def reduction_path(artifact_dir, tag, merit, split):
    return Path(artifact_dir)/f'{tag}_reduced_{merit}_{split}.parquet'


def run_reduce(args, entries, splits):
    """The pool pass, once per (merit, split). The only stage that needs the pool."""
    artifact_dir = Path(args.artifact_dir)
    merits = list(args.merits) if args.merits else list(MERITS)
    if args.unfloored:
        merits = merits + [UNFLOORED]
    metas = {}
    for merit in merits:
        for split_label, entry_ids in splits.items():
            started = time.perf_counter()
            reduced, _, meta = reduce_one(
                args.pool, merit, entry_ids, entries, split_label,
                merit_dir=args.merit_dir, unfloored=args.unfloored or merit == UNFLOORED,
                pool_mode=args.pool_mode,
                )
            # Asserted rather than trusted: a reduction that silently lost its exactness
            # certificate is the one thing that cannot be detected downstream.
            assert meta['ranks_exact'], meta['rank_exactness']
            reduced.to_parquet(reduction_path(artifact_dir, args.tag, merit, split_label),
                               index=False)
            metas[f'{merit}|{split_label}'] = meta
            print(f'  reduced {merit:16s} {split_label:10s} {reduced.shape[0]:6d} cells '
                  f'from {meta["n_candidates_seen"]:,} candidates '
                  f'({time.perf_counter() - started:.0f} s)', flush=True)
    (artifact_dir/f'{args.tag}_reduced_meta.json').write_text(
        json.dumps(metas, indent=2, sort_keys=True, default=str), encoding='utf-8')
    print(f'wrote {len(metas)} reductions to {artifact_dir}')
    return metas


def load_reductions(artifact_dir, tag):
    metas = json.loads((Path(artifact_dir)/f'{tag}_reduced_meta.json').read_text(encoding='utf-8'))
    out = {}
    for key, meta in metas.items():
        merit, split_label = key.split('|')
        path = reduction_path(artifact_dir, tag, merit, split_label)
        out[(merit, split_label)] = (pd.read_parquet(path), meta)
    return out


def _summarise(reduction, threshold=None, strata=(), n_bootstrap=0, seed=12345, top_n=10):
    frame, meta = reduction
    return FomMetrics.summarise_per_entry(
        frame, meta, threshold=threshold, top_n=top_n, strata=strata,
        n_bootstrap=n_bootstrap, seed=seed,
        )


def _table_row(merit, result, threshold, choice):
    aggregate = result.aggregate.iloc[0]
    hard = result.hard.iloc[0] if result.hard.shape[0] else None
    return {
        'merit': merit,
        'higher_is_better': result.meta['higher_is_better'],
        'note': MERIT_NOTES.get(merit, ''),
        'threshold': threshold,
        'threshold_objective': None if choice is None else choice.objective,
        'threshold_split': None if choice is None else choice.split,
        'operating_point': aggregate['operating_point'],
        'operating_point_ci_low': aggregate['operating_point_ci_low'],
        'operating_point_ci_high': aggregate['operating_point_ci_high'],
        'false_positive_rate': aggregate['false_positive'],
        'precision': aggregate['precision'],
        'reported': aggregate['reported'],
        'top1': aggregate['top1'],
        'top10': aggregate['top10'],
        'mrr': aggregate['mrr'],
        'rank_only': aggregate['rank_only'],
        'threshold_only': aggregate['threshold_only'],
        'ceiling_rescorer': aggregate['ceiling_rescorer'],
        'n_entries': aggregate['n_entries'],
        'hard_operating_point': np.nan if hard is None else hard['operating_point'],
        'hard_operating_point_given_found':
            np.nan if hard is None else hard['operating_point_given_found'],
        'hard_top10': np.nan if hard is None else hard['top10'],
        'hard_ceiling_rescorer': np.nan if hard is None else hard['ceiling_rescorer'],
        'hard_n_entries': np.nan if hard is None else hard['n_entries'],
        'hard_n_found': np.nan if hard is None else hard['n_found'],
        }


def run_analyse(args, reductions):
    """Thresholds, the leaderboard, the paired tests and the ceilings. No pool required."""
    artifact_dir = Path(args.artifact_dir)
    train, dev = args.train_split, args.report_split
    strata = ('bravais_lattice', 'volume_decile', 'condition_bundle')
    merits = [m for m in (list(args.merits) if args.merits else list(MERITS))
              if (m, dev) in reductions]
    if args.unfloored and (UNFLOORED, dev) in reductions:
        merits = merits + [UNFLOORED]

    # ---------------------------------------------------------------- the loss decomposition
    # Reported first and by stratum because it orders everything after it: an aggregate that says
    # the loss is a threshold failure and a hard stratum that says both halves fail together are
    # different instructions to S12.
    m20_dev = _summarise(reductions[('M20', dev)], threshold=DEWOLFF_THRESHOLD, strata=strata,
                         n_bootstrap=args.n_bootstrap, seed=args.seed)
    m20_dev.loss.to_csv(artifact_dir/f'{args.tag}_loss_decomposition.csv', index=False,
                        encoding='utf-8')
    for scope in ('all', 'hard'):
        row = m20_dev.loss.loc[m20_dev.loss['stratum'] == scope]
        if row.shape[0]:
            row = row.iloc[0]
            print(f'  M20 {scope:5s} op {row["operating_point"]:.4f}  '
                  f'rank {row["share_rank_failure"]:.3f}  '
                  f'threshold {row["share_threshold_failure"]:.3f}  '
                  f'both {row["share_both"]:.3f}  '
                  f'(not found {row["lost_not_found"]:.3f})')

    # ---------------------------------------------------------------- the matched budget
    # The operating point cannot be maximised over the threshold -- it is monotone, so the
    # unconstrained maximiser is minus infinity. Every merit is therefore also thresholded at a
    # matched cost in wrong answers reported: the false-positive rate M20 itself incurs at de
    # Wolff's 10 on fom-train. Equal willingness to answer, so the operating points compare.
    m20_train = _summarise(reductions[('M20', train)], threshold=DEWOLFF_THRESHOLD)
    budget = float(m20_train.metric('false_positive'))
    print(f'\nmatched false-positive budget, M20 @ {DEWOLFF_THRESHOLD:g} on {train}: '
          f'{budget:.4f}')

    # ---------------------------------------------------------------- the leaderboard
    print(f'\nthe leaderboard: threshold selected on {train}, reported on {dev}')
    rows, results, thresholds = [], {}, {}
    for merit in merits:
        higher = reductions[(merit, dev)][1]['higher_is_better']
        train_result = _summarise(reductions[(merit, train)])
        choice = FomMetrics.select_threshold(train_result, objective='youden')
        # `per_entry` stores scores already oriented higher-is-better, so a lower-is-better
        # merit's chosen threshold comes back negated and has to be turned round again before
        # `summarise_per_entry` mirrors it a second time.
        threshold = choice.threshold if higher else -choice.threshold
        dev_result = _summarise(reductions[(merit, dev)], threshold=threshold, strata=strata,
                                n_bootstrap=args.n_bootstrap, seed=args.seed)
        FomMetrics.check_threshold_transfer(choice, dev_result)

        budgeted_choice = FomMetrics.select_threshold(
            train_result, objective='operating_point', max_false_positive_rate=budget)
        budgeted_threshold = (budgeted_choice.threshold if higher
                              else -budgeted_choice.threshold)
        budgeted = _summarise(reductions[(merit, dev)], threshold=budgeted_threshold)
        FomMetrics.check_threshold_transfer(budgeted_choice, budgeted)

        results[merit] = dev_result
        thresholds[merit] = choice
        row = _table_row(merit, dev_result, threshold, choice)
        row['threshold_matched_fpr'] = budgeted_threshold
        row['operating_point_matched_fpr'] = budgeted.metric('operating_point')
        row['false_positive_rate_matched'] = budgeted.metric('false_positive')
        # Campaign 1 flagged this and got its denominator wrong; the condition is recorded here so
        # the count is derivable from the table rather than quoted from prose (C2-F-006).
        row['abstains_always'] = bool(row['reported'] == 0.0 and row['operating_point'] == 0.0)
        rows.append(row)
        print(f'  {merit:16s} op {row["operating_point"]:.4f}  '
              f'op@fpr {row["operating_point_matched_fpr"]:.4f}  '
              f'top10 {row["top10"]:.4f}  hard|found '
              f'{row["hard_operating_point_given_found"]:.4f}  '
              f'{"ABSTAINS" if row["abstains_always"] else ""}')
    table = pd.DataFrame(rows).sort_values('operating_point_matched_fpr', ascending=False)

    # ---------------------------------------------------------------- paired comparisons
    # Every one of these is paired over the same entries, including the per-lattice rows. Campaign
    # 1 could not run the per-lattice ones at all -- `mcnemar`'s mask argument raised on every
    # call -- so all of its per-lattice claims are unpaired deltas with no interval (F-087).
    print('\npaired against M20 (McNemar + cluster-bootstrap interval over source entries)')
    comparisons = []
    lattices = sorted(results['M20'].per_entry['bravais_lattice'].dropna().unique())
    for merit in merits:
        if merit == 'M20':
            continue
        scopes = [('all', None), ('hard', 'hard')]
        scopes += [(f'bravais_lattice={lattice}',
                    FomMetrics.stratum_mask(results[merit], 'bravais_lattice', lattice))
                   for lattice in lattices]
        for label, subset in scopes:
            for metric in ('operating_point', 'top10'):
                try:
                    test = FomMetrics.mcnemar(results[merit], results['M20'], metric=metric,
                                              subset=subset)
                    interval = FomMetrics.paired_delta_ci(
                        results[merit], results['M20'], metric=metric, subset=subset,
                        n_bootstrap=args.n_bootstrap, seed=args.seed)
                except ValueError as error:
                    comparisons.append({'merit': merit, 'scope': label, 'metric': metric,
                                        'error': str(error)})
                    continue
                comparisons.append({
                    'merit': merit, 'scope': label, **dict(test),
                    'ci_low': interval['ci_low'], 'ci_high': interval['ci_high'],
                    })
        headline = [c for c in comparisons
                    if c.get('merit') == merit and c.get('scope') == 'all'
                    and c.get('metric') == 'top10']
        if headline and 'delta' in headline[0]:
            row = headline[0]
            print(f'  {merit:16s} top10 delta {row["delta"]:+.4f} '
                  f'[{row["ci_low"]:+.4f}, {row["ci_high"]:+.4f}]  '
                  f'{int(row["n_a_only"])}/{int(row["n_b_only"])}  p {row["p_value"]:.3g}')
    comparison_frame = pd.DataFrame(comparisons)

    # ---------------------------------------------------------------- ceilings, and the hard arm
    oracle_rows = []
    for scope, frame in (('aggregate', m20_dev.aggregate), ('hard', m20_dev.hard)):
        if frame.shape[0]:
            row = frame.iloc[0]
            oracle_rows.append({'scope': scope, **{
                key: row[key] for key in ('operating_point', 'ceiling_reranker',
                                          'ceiling_rescorer', 'headroom_reranker',
                                          'headroom_rescorer', 'degenerate_only', 'n_entries')}})
    for row in m20_dev.by_stratum.itertuples():
        oracle_rows.append({'scope': f'{row.stratum}={row.level}',
                            'operating_point': row.operating_point,
                            'ceiling_reranker': row.ceiling_reranker,
                            'ceiling_rescorer': row.ceiling_rescorer,
                            'headroom_reranker': row.headroom_reranker,
                            'headroom_rescorer': row.headroom_rescorer,
                            'degenerate_only': row.degenerate_only,
                            'n_entries': row.n_entries})
    oracle = pd.DataFrame(oracle_rows)

    # The hard stratum on `fom-dev` alone, on the rank metrics. S06 sized it so campaign 1's
    # train+dev pooling licence is not needed, and PROTOCOL forbids inheriting it anyway.
    print(f'\nthe hard stratum, rank metrics, {dev} alone')
    hard_rows = []
    for merit in merits:
        result = results[merit]
        if not result.hard.shape[0]:
            continue
        row = result.hard.iloc[0]
        hard_rows.append({'merit': merit, 'n_entries': row['n_entries'],
                          'n_found': row['n_found'], 'top1': row['top1'], 'top10': row['top10'],
                          'rank_only': row['rank_only'], 'mrr': row['mrr'],
                          'ceiling_rescorer': row['ceiling_rescorer']})
        print(f'  {merit:16s} top1 {row["top1"]:.4f}  top10 {row["top10"]:.4f}  '
              f'mrr {row["mrr"]:.4f}  (n={int(row["n_entries"])}, '
              f'{int(row["n_found"])} reachable)')
    hard_frame = pd.DataFrame(hard_rows)
    if hard_frame.shape[0]:
        hard_frame = hard_frame.sort_values('top10', ascending=False)

    # ---------------------------------------------------------------- per bundle, and per entry
    per_bundle = []
    for merit, result in results.items():
        frame = result.by_stratum
        for row in frame.loc[frame['stratum'] == 'condition_bundle'].itertuples():
            per_bundle.append({'merit': merit, 'condition_bundle': row.level,
                               'condition': FomMetrics.BUNDLE_LABELS.get(row.level, '?'),
                               'operating_point': row.operating_point,
                               'operating_point_given_found': row.operating_point_given_found,
                               'top1': row.top1, 'top10': row.top10, 'mrr': row.mrr,
                               'ceiling_rescorer': row.ceiling_rescorer,
                               'n_entries': row.n_entries})
    per_bundle_frame = pd.DataFrame(per_bundle)

    # Persisted because the union oracle and the complementarity matrix are joins over these
    # flags: recomputing them would mean re-running every merit (PROTOCOL section 3 rule 8).
    keep = ['entry_id', 'condition_bundle', 'bravais_lattice', 'volume_decile', 'is_hard',
            'top1', 'top10', 'rank_only', 'threshold_only', 'operating_point', 'found',
            'reported', 'false_positive', 'reciprocal_rank', 'rank_best_correct']
    per_entry = []
    for merit, result in results.items():
        subset = result.per_entry[[c for c in keep if c in result.per_entry.columns]].copy()
        subset.insert(0, 'merit', merit)
        per_entry.append(subset)
    per_entry_frame = pd.concat(per_entry, ignore_index=True)

    # ---------------------------------------------------------------- write
    table.to_csv(artifact_dir/f'{args.tag}_main_table.csv', index=False, encoding='utf-8')
    comparison_frame.to_csv(artifact_dir/f'{args.tag}_mcnemar.csv', index=False, encoding='utf-8')
    oracle.to_csv(artifact_dir/f'{args.tag}_oracle.csv', index=False, encoding='utf-8')
    per_bundle_frame.to_csv(artifact_dir/f'{args.tag}_per_bundle.csv', index=False,
                            encoding='utf-8')
    hard_frame.to_csv(artifact_dir/f'{args.tag}_hard_table.csv', index=False, encoding='utf-8')
    per_entry_frame.to_parquet(artifact_dir/f'{args.tag}_per_entry.parquet', index=False)
    (artifact_dir/f'{args.tag}_thresholds.json').write_text(
        json.dumps({'commit': commit_hash(), 'matched_fpr_budget': budget,
                    'choices': {m: c.to_dict() for m, c in thresholds.items()}},
                   indent=2, sort_keys=True, default=str),
        encoding='utf-8')
    print(f'\nwrote {args.tag}_{{main_table,mcnemar,oracle,hard_table,per_bundle,'
          f'loss_decomposition}}.csv and _per_entry.parquet to {artifact_dir}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='S09 -- rank the classical merit zoo on Benchmark B.')
    parser.add_argument('--pool', default=os.path.join(BASE, 'mlindex', 'data',
                                                       'fom_benchmark_c2'),
                        help='Benchmark root. Needed for --reduce only.')
    parser.add_argument('--merit-dir', nargs='+', default=None,
                        help='Merit sidecar directories. Defaults to <pool>/merits. Several may '
                             'be given, e.g. <pool>/merits <pool>/merits_soft -- they are joined '
                             'in turn on the four keys, so a pool can carry the verified set and '
                             'an experimental one without either invalidating the other.')
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--train-split', default='fom-train')
    parser.add_argument('--report-split', default='fom-dev')
    parser.add_argument('--merits', nargs='+', default=None,
                        help='The merits to evaluate. Defaults to the seven the subsampler ranked '
                             'on. Anything else -- a soft count, a learned score -- is only '
                             'rank-exact on a FULLY RETAINED pool (C2-R-013), and evaluate() will '
                             'refuse it on a subsampled one, which is the intended behaviour.')
    parser.add_argument('--unfloored', action='store_true',
                        help='Add the unfloored M_sym comparison arm. Only meaningful on a fully '
                             'retained pool: on a subsampled one the arm is flattered, because '
                             'the subsampler ranked on the floored merit (C2-F-084).')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--pool-mode', choices=('cross_bl', 'per_bl'), default='cross_bl',
                        help="'per_bl' ranks within each Bravais lattice. Never the headline -- a "
                             'different and much easier problem than the one run.py solves. Used '
                             'to size the cross-lattice half of a merit\'s advantage.')
    parser.add_argument('--reduce', action='store_true', help='Pool pass only.')
    parser.add_argument('--analyse', action='store_true', help='Analysis only.')
    parser.add_argument('--tag', default='S09_zoo')
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    both = not (args.reduce or args.analyse)

    if args.unfloored:
        depth, subsampled = FomBenchmark.subsample_depth(Path(args.pool))
        if subsampled:
            raise SystemExit(
                f'Refusing --unfloored on a subsampled pool (K = {depth}). The subsampler ranked '
                f'on the FLOORED M_rev, so a saturated fit scored 0.0, ranked last and was kept '
                f'at 5 % -- unfloored, those same rows rank first. The arm would be scored '
                f'against a field with its own strongest rivals removed and would come out '
                f'flattered, understating what the floor is worth. Run it on a fully retained '
                f'pool. See C2-F-084.')

    if args.reduce or both:
        entries = FomBenchmark.load_entries(args.pool)
        splits = {}
        for label in (args.train_split, args.report_split):
            ids = set(entries.loc[entries['split'] == label, 'entry_id'])
            if ids:
                splits[label] = ids
        print(f'{args.pool}: ' + ' / '.join(f'{len(v):,} {k}' for k, v in splits.items())
              + ' source entries')
        run_reduce(args, entries, splits)

    if args.analyse or both:
        reductions = load_reductions(artifact_dir, args.tag)
        if (args.report_split not in {split for _, split in reductions}):
            raise SystemExit(
                f'No reduction for the reporting split {args.report_split!r}. This pool carries '
                f'{sorted({s for _, s in reductions})}.')
        return run_analyse(args, reductions)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
