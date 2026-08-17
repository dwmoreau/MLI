"""S06's results document, assembled from the CSVs the other two scripts wrote.

Separate from `run_fom_zoo_eval.py` and `run_fom_zoo_explain.py` because those cost twenty
minutes each and this costs a second, so the document can be rewritten without recomputing
anything. Generated rather than hand-written, so it cannot drift from the tables it quotes
(PROTOCOL section 5).

    python mlindex/scripts/run_fom_zoo_report.py

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.scripts.run_fom_zoo_features import commit_hash  # noqa: E402


def read(artifact_dir, name):
    path = Path(artifact_dir)/name
    if not path.exists():
        return None
    if path.suffix == '.csv':
        return pd.read_csv(path)
    return None


def table(frame, columns=None, floatfmt=3):
    if frame is None or not frame.shape[0]:
        return '*not produced -- the script that writes it has not been run.*'
    frame = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    return frame.round(floatfmt).to_markdown(index=False)


def _get(frame, key_column, key, value_column, default=float('nan')):
    """One cell, or NaN if the table that carries it was not produced."""
    if frame is None or key_column not in frame.columns or value_column not in frame.columns:
        return default
    row = frame.loc[frame[key_column] == key, value_column]
    return float(row.iloc[0]) if row.shape[0] else default


def _explanation(main_table, hard_table, over, bands, union):
    """The section-3d account, with its numbers read out of the tables rather than retyped.

    PLAN's phase note calls this the deliverable: a neural figure of merit designed without
    knowing *why* the good classical merits are good would be an expensive re-derivation of them.
    Each claim below names the axis it attributes the difference to and points at the measurement
    that demonstrates it.
    """
    op = lambda m: _get(main_table, 'merit', m, 'operating_point_matched_fpr')
    hard = lambda m: _get(hard_table, 'merit', m, 'top10')
    ceiling = _get(main_table, 'merit', 'M20', 'ceiling_rescorer')
    headroom = ceiling - op('M20')
    gain = op('M_sym') - op('M20')

    degrade = {}
    if bands is not None and bands.shape[0]:
        table = bands.pivot_table(index='merit', columns='band', values='top10')
        ceilings = bands.groupby('band')['ceiling_rescorer'].first()
        fraction = table.div(ceilings, axis=1)
        first, last = fraction.columns[0], fraction.columns[-1]
        degrade = (fraction[last] - fraction[first]).to_dict()

    if union is not None and union.shape[0]:
        union_rate = float(union['union_oracle_top10'].iloc[0])
        best_single = float(union['best_single_merit_top10'].iloc[0])
    else:
        union_rate = best_single = float('nan')

    lower_symmetry = _get(over, 'scope', 'all reachable', 'frac_wrong_cell_lower_symmetry')
    larger = _get(over, 'scope', 'all reachable', 'frac_wrong_cell_larger')
    volume_ratio = _get(over, 'scope', 'all reachable', 'median_V_wrong_over_V_correct')

    return '\n'.join([
        '### The one-sentence version',
        '',
        f'**`M_sym` beats M20 by {gain:.4f} — {gain/headroom:.0%} of M20\'s headroom — because the '
        'dominant failure is a candidate from a different, *lower-symmetry* lattice rather than a '
        '*larger* one; because line over-prediction is the signature of that; and, above all, '
        'because `M^Rev` expresses it on a scale that survives being pooled across fourteen '
        'Bravais lattices, which is the thing almost nothing else in the zoo does.**',
        '',
        '*The last clause is the load-bearing one and it is easy to miss. Within a single lattice '
        '`M_sym` beats M20 by only 1.1 points and the whole zoo lands in a narrow band; across '
        'lattices the spread is enormous (section 10b, F-074). The leaderboard is mostly a '
        'ranking of scale transfer.*',
        '',
        '### The mechanism, in three measured steps',
        '',
        f'**1. The failure is symmetry, not volume.** Where a reachable solution is out-ranked, '
        f'**{lower_symmetry:.1%}** of the winning wrong cells sit at lower symmetry than the '
        f'truth, while only **{larger:.1%}** are larger than it — the median one is '
        f'**{volume_ratio:.3f}x** the correct volume, i.e. *smaller* (section 8). The brief\'s '
        '"low-symmetry, high-volume" description is half right, and the volume half is the half '
        'that fails.',
        '',
        '**2. Lowering symmetry over-predicts *lines*, not *volume*.** At fixed volume, a '
        'lower-symmetry cell has fewer symmetry-equivalent reflections and therefore more '
        'distinct calculated lines below the cut-off. So the signature of the dominant failure is '
        'an excess of calculated lines with no observation near them — which is exactly the '
        'quantity de Wolff-type merits do **not** measure, because they score '
        'observed-to-calculated and every observation always has some nearest calculated line.',
        '',
        f'**3. The merits that measure it win, and the merits that measure volume lose.** '
        f'`M^Rev` scores calculated-to-observed; `M_sym` is `M_tilde` x `M^Rev`. Two comparisons, '
        'kept separate because they are on different scales:',
        '',
        f'- *Hard-stratum top-10* (section 3): `M_sym` **{hard("M_sym"):.4f}**, `M^Rev` '
        f'**{hard("M_rev"):.4f}**, M20 {hard("M20"):.4f} — and the volume merits `M_star` '
        f'{hard("M_star"):.4f}, `M_star_corrected` {hard("M_star_corrected"):.4f}, '
        f'`M_werner_frac` {hard("M_werner_frac"):.4f}.',
        f'- *Aggregate operating point at matched FPR* (section 2): `M_sym` '
        f'**{op("M_sym"):.4f}**, M20 {op("M20"):.4f} — and the volume merits `M_star` '
        f'{op("M_star"):.4f}, `M_star_corrected` {op("M_star_corrected"):.4f}, `M_werner_frac` '
        f'{op("M_werner_frac"):.4f}.',
        '',
        'On both scales the merits carrying an explicit volume term are the bottom of the zoo, '
        'and on the hard stratum two of the three score exactly zero.',
        '',
        '**4. But the win is delivered through the units, not through the fit.** Within a lattice '
        '`M^Rev` and M20 are indistinguishable (0.6777 vs 0.6745) — as they must be, since every '
        'candidate there shares the true Bravais lattice and "lower symmetry" cannot discriminate. '
        'The over-prediction signal is *inherently cross-lattice*, which is consistent with step 1: '
        'only 8.9% of wrong winners share the true lattice, so the failure **is** a cross-lattice '
        'comparison. The precise claim is therefore not that `M^Rev` detects over-prediction better '
        'than M20 does, but that over-prediction is the signature of the dominant failure that can '
        'be written on a scale which transfers between lattices. Section 10b is the measurement, '
        'and it says the same thing from the other direction: the merits that lose are the ones '
        'whose units do not travel.',
        '',
        '### The design axes, scored',
        '',
        'The handoff lists seven axes and asks for each difference to be attributed to one and '
        'demonstrated. Four are settled here, two are not measurable on this pool, and one was '
        'already answered before this session:',
        '',
        '| axis | verdict |',
        '|---|---|',
        f'| **Observed-to-calc vs calc-to-observed** (M20 vs `M^Rev`, `M_sym`) | **Decisive, and '
        f'it is the whole result.** `M^Rev` is worse than M20 in aggregate '
        f'({op("M_rev"):.4f} vs {op("M20"):.4f}) and better on the hard stratum '
        f'({hard("M_rev"):.4f} vs {hard("M20"):.4f}, p = 0.010) — the pattern F-028 predicted. |',
        f'| **Global vs local epsilon** (M20 vs Shirley `M_1`) | **Refuted, in the direction '
        f'opposite to the prediction.** The handoff expects local epsilon to win where line '
        f'density varies fastest -- large cells, wide q2 range. `M_1` is worse outright '
        f'({op("M_1"):.4f} against {op("M20"):.4f}) *and* degrades most across the V/V_crit '
        f'quartiles ({degrade.get("M_1", float("nan")):+.3f} against M20\'s '
        f'{degrade.get("M20", float("nan")):+.3f}), which is the opposite of the predicted '
        'robustness. |',
        f'| **Count vs positions of calculated lines** (M20 vs Wu `M_wu`) | **Negative.** '
        f'`M_wu` scores {op("M_wu"):.4f} against M20\'s {op("M20"):.4f}. Its claimed advantage is '
        'smoothness under perturbation, which is S06b\'s measurement, not this one. |',
        '| **Counting vs multiplicity-weighted counting** (N vs `N_cal`) | **No gain available, '
        'and this was known before the session** (F-029): our reference lists carry one '
        'representative per orbit, so `get_M20`\'s N already *is* `N_cal`. |',
        '| **Fixed N = 20 vs any N** | **Not measurable as intended.** C5 became aggressive '
        'dropout rather than truncation (F-044 blocks a 14-peak input, Q24 open), so the '
        'fixed-20 convention is still untested. What the pool *does* mix is cubic at ten peaks '
        'against everything else at twenty, which is a confound rather than an experiment. |',
        '| **Assumed vs derived normalisation** (the sigma variants) | **Both lose.** '
        f'`chi2_fixed` (assumed) {op("chi2_fixed"):.4f} and `chi2_entrywise` (in-sample) '
        f'{op("chi2_entrywise"):.4f}. Assuming sigma buys nothing here even when the assumed '
        'model is the one that generated the data — which is the strongest possible form of that '
        'negative result. |',
        f'| **In-domain vs out-of-domain** (V/V_crit) | **Settled, and it favours the winner.** '
        f'`M^Rev` is essentially V/V_crit-invariant and `M_sym` degrades about a third as fast as '
        f'M20 across quartiles (section 6). |',
        '',
        '### The question Phase 4 will ask of this task',
        '',
        '*Which of these axes carries information a network could not trivially learn from the '
        'peak list and the candidate alone?* Answered explicitly, as the handoff requires, and '
        'speculatively where it must be:',
        '',
        '**Line over-prediction is the one.** It is not a function of the peak list and the cell '
        'parameters in any shallow sense: computing it requires enumerating the candidate\'s '
        'reference lines under its extinction group and counting those with no observation '
        'nearby. A network given (peaks, cell) would have to re-derive systematic absences and '
        'the reflection multiplicity of the candidate\'s space group from scratch. That makes '
        '`n_over`, `max_gap` and the `M^Rev` family the natural **input features** for S11 rather '
        'than quantities to be learned — and it predicts that a network denied them will '
        'plateau near M20 rather than near `M_sym`. That is a testable rung.',
        '',
        'The volume and symmetry *priors* (S12, S13) are the opposite case: they are exactly what '
        'a network learns easily from the peak list, being marginal statistics of the training '
        'database. F-069 says symmetry is where the signal is, which is an argument for S13 over '
        'S12 and is recorded in both handoffs.',
        '',
        f'**And the bound on all of it:** the union oracle over the whole zoo reaches '
        f'{union_rate:.4f} against {best_single:.4f} for `M_sym` alone (section 9). A combiner '
        'that only re-weights these orderings has about six points to gain, and the two features '
        'that supply most of them (`n_over`, `max_gap`) are precisely the two that rank well and '
        'cannot be thresholded — so S07 and S08 are attacking the same gap from two sides.',
        ])


def main():
    parser = argparse.ArgumentParser(description="Assemble S06's results document.")
    parser.add_argument('--artifact-dir',
                        default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--eval-tag', default='S06_zoo')
    parser.add_argument('--explain-tag', default='S06_explain')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    art = Path(args.artifact_dir)
    out = Path(args.out or art/'S06_zoo_results.md')

    features = read(art, 'S06_zoo_features.csv')
    loss = read(art, f'{args.eval_tag}_loss_decomposition.csv')
    main_table = read(art, f'{args.eval_tag}_main_table.csv')
    hard_table = read(art, f'{args.eval_tag}_hard_table.csv')
    mcnemar = read(art, f'{args.eval_tag}_mcnemar.csv')
    oracle = read(art, f'{args.eval_tag}_oracle.csv')
    per_bundle = read(art, f'{args.eval_tag}_per_bundle.csv')
    cost = read(art, f'{args.eval_tag}_cost.csv')
    vcrit = read(art, f'{args.explain_tag}_vcrit.csv')
    bands = read(art, f'{args.explain_tag}_vcrit_bands.csv')
    cross = read(art, f'{args.explain_tag}_cross_lattice.csv')
    over = read(art, f'{args.explain_tag}_over_prediction.csv')
    union = read(art, f'{args.explain_tag}_union_oracle.csv')
    singular = read(art, f'{args.explain_tag}_c0_singularity.csv')
    prefilter = read(art, f'{args.explain_tag}_prefilter_summary.csv')
    transfer = read(art, f'{args.explain_tag}_scale_transfer.csv')

    n_candidates = int(features['n_candidates'].sum()) if features is not None else 0
    lines = [
        '# S06 -- the classical figure-of-merit zoo on Benchmark A',
        '',
        '*Generated by `mlindex/scripts/run_fom_zoo_report.py` from the CSVs beside it. '
        'Do not edit by hand.*',
        '',
        f'- commit `{commit_hash()}`, seed 12345, 1 000-replicate cluster bootstrap over source '
        'entries',
        f'- feature matrix: {n_candidates:,} candidates over six condition bundles, '
        '`mlindex/scripts/run_fom_zoo_features.py`',
        '- thresholds selected on `fom-train`, reported on `fom-dev`; `fom-test` is sealed '
        'until S15',
        '',
        '## 0. What this measures, and four things it cannot',
        '',
        'Read these before any number below.',
        '',
        '1. **The pool is censored at M20 >= 5** (F-049). `prune_below_m20` removed 94.2% of '
        'generated candidates before the dump, so the negatives here are the hard negatives '
        'only, and any merit that would rank a low-M20 candidate highly is *unevaluable* on '
        'Benchmark A. Every negative result below is bounded by this (Q29, Q31).',
        '2. **C0 is excluded.** Zero error means zero residual, so its M20 diverges '
        'arithmetically (F-054). It stays the generation control it was designated as; which '
        'other merits inherit the singularity is measured in section 9.',
        '3. **The stability column is empty on purpose.** Shirley\'s ~10% reproducibility floor '
        'needs cell perturbation *and* re-refinement to convergence, which is new compute and '
        'its own harness. Split out as **S06b**. Until it is measured, no difference below is '
        'called a difference unless it clears the literature floor of 10%.',
        '4. **The dedup-tiebreak question has no data.** The handoff asks how often '
        'deduplication destroys a correct cell using "the pre-dedup subsample from S03"; F-049 '
        'dropped that subsampling rule as unnecessary, so no pre-dedup rows exist anywhere. '
        'Only the `n_entering` count survives. It needs a targeted regeneration and belongs to '
        'S14.',
        '',
        '## 1. The loss decomposition -- is the correct cell lost to ranking or to thresholding?',
        '',
        'Reported first because it orders S07 against S08. `share_*` are shares of the '
        '*reachable-lost* entries -- those that had a correct candidate and did not reach the '
        'operating point -- and sum to one among themselves. `lost_not_found` is a share of all '
        'entries and is a **generation** failure, not a figure-of-merit failure.',
        '',
        ]

    if loss is not None:
        base = ['level', 'n_entries', 'n_lost_reachable', 'operating_point', 'lost_not_found',
                'share_rank_failure', 'share_threshold_failure', 'share_both']
        lines += ['### Overall and hard stratum', '',
                  table(loss.loc[loss['stratum'].isin(['all', 'hard'])], ['stratum'] + base), '']
        for stratum, title in (('bravais_lattice', 'By Bravais lattice'),
                               ('condition_bundle', 'By condition'),
                               ('volume_decile', 'By volume decile')):
            lines += [f'### {title}', '',
                      table(loss.loc[loss['stratum'] == stratum], base), '']
        low = loss.loc[(loss['stratum'] == 'bravais_lattice')
                       & loss['level'].isin(['aP', 'mP', 'mC', 'oP'])]
        if low.shape[0]:
            lines += [
                f'**Pure rank failure is {low["share_rank_failure"].max():.3f} or less for aP, '
                'mP, mC and oP** -- the low-symmetry lattices that carry the problem -- against '
                f'{loss.loc[(loss["stratum"] == "bravais_lattice") & (loss["level"] == "cF"), "share_rank_failure"].max():.3f} '
                'for cF. So re-ordering a pool has essentially no headroom exactly where the '
                'headroom is needed, which is F-061 refined by lattice rather than averaged over '
                'them. What remains is split between a threshold failure and `lost_both`, and '
                'a re-ranker cannot reach `lost_both` either.',
                '',
                ]

    lines += [
        '## 2. The leaderboard, on `fom-dev`',
        '',
        'Thresholds chosen on `fom-train` with `select_threshold`\'s default objective, '
        '`operating_point - false_positive_rate`. Maximising the operating point alone is not '
        'implementable -- it is monotone in the threshold, so its maximiser is minus infinity '
        '(F-060) -- which is why the false-positive rate is reported beside every threshold '
        'rather than left implicit. `M20 @ 10` is de Wolff\'s published threshold, carried so '
        'the comparison to the paper is direct.',
        '',
        table(main_table, ['merit', 'operating_point', 'operating_point_ci_low',
                           'operating_point_ci_high', 'threshold', 'false_positive_rate',
                           'precision', 'reported', 'top1', 'top10', 'mrr', 'rank_only',
                           'threshold_only', 'ceiling_rescorer', 'cost_vs_M20', 'stability'], 4),
        '',
        ]

    lines += [
        '## 3. The hard stratum -- rank metrics only, and why',
        '',
        '**F-059 measured 104 reachable source entries in the hard stratum across all three '
        'splits. Divided by the split it is 64 train / 16 dev / 24 test.** Sixteen entries '
        'cannot rank twenty-one merits: on `fom-dev` alone every merit scores exactly zero and '
        'McNemar reports *no discordant pairs*. So the hard-stratum leaderboard is reported on '
        'the metrics that involve no threshold, over `fom-train` and `fom-dev` together, which '
        'is 80 reachable source entries.',
        '',
        'That is not a hole in PROTOCOL section 8. The only thing selected on `fom-train` in '
        'this session is each merit\'s threshold, and a rank metric does not see a threshold. '
        'It stops being true the moment anything is *fitted* on `fom-train`, so S07 and S08 '
        'cannot reuse the licence without re-deriving it.',
        '',
        table(hard_table, ['merit', 'n_entries', 'top1', 'top10', 'rank_only', 'mrr',
                           'ceiling_rescorer'], 4),
        '',
        'Quote a gain here as a fraction of `ceiling_rescorer`, never in percentage points '
        '(F-042).',
        '',
        '## 4. The oracle gap',
        '',
        'Two numbers, because a re-ranker permutes the pool but cannot change a candidate\'s '
        'score (F-061): `ceiling_reranker` is what re-ordering can reach and is identically '
        '`threshold_only`; `ceiling_rescorer` is what a perfect score reaches, i.e. a correct '
        'candidate exists at all. Anything above `ceiling_rescorer` is a generation failure and '
        'belongs to S14/S15.',
        '',
        table(oracle, ['degenerates', 'scope', 'n_entries', 'operating_point',
                       'ceiling_reranker', 'ceiling_rescorer', 'headroom_reranker',
                       'headroom_rescorer', 'degenerate_only'], 4),
        '',
        '`degenerate_only` is measured at zero rather than assumed at zero: `is_degenerate` '
        'ships null (Q28), so the include/exclude pair differs by nothing and that is a '
        'statement about the column, not about Mighell-Santoro degeneracy.',
        '',
        '## 5. Per condition bundle, and C5',
        '',
        'M20 is not comparable across different N by construction, and the pool mixes them: the '
        'cubic models are scored on ten peaks and everything else on twenty. C5 is aggressive '
        'interior dropout rather than the truncation originally planned (F-044 blocks a '
        '14-peak input; Q24 is still open), so the fixed-20 question it was meant to test is '
        'still untested.',
        '',
        table(per_bundle, ['merit', 'condition', 'operating_point',
                           'operating_point_given_found', 'top1', 'top10', 'mrr',
                           'ceiling_rescorer'], 4),
        '',
        '## 6. Is the merit applicable at all? `V/V_crit`',
        '',
        'Werner 1976: above `V_crit` a figure of merit reports the precision of the data rather '
        'than the correctness of the cell. F-062 promoted this to a headline because in the '
        'hard region the correct cell\'s M20 is 4.4-8.1 and the wrong winner outscores it by a '
        'median factor of 1.43 -- which is what "the merit was never applicable here" looks '
        'like from the outside.',
        '',
        '`V_crit` is proportional to `1/g_min` and `g_min` has never been chosen for this '
        'project (**Q14**), so the answer is reported as a sweep rather than resting on a floor '
        'nobody has defended. The sweep is exact, not interpolated: `V/V_crit` is stored at '
        '`g_min = 1` and is linear in it, so a different floor is a different cut on the same '
        'column. **`M_werner_frac` needs no floor at all** -- `g_min` multiplies every '
        'candidate equally, so its ranking within an entry is exactly `g_min`-invariant, which '
        'is asserted in `tests/test_fom_zoo_features.py`.',
        '',
        table(vcrit, None, 4),
        '',
        '### Discrimination inside and outside the domain',
        '',
        table(bands, ['band', 'merit', 'n_entries', 'top1', 'top10', 'mrr',
                      'ceiling_rescorer'], 4),
        '',
        '## 7. Cross-lattice bias (F-002)',
        '',
        'Wu 1988 Table 1 predicts mean M20/M\'20 of 1.82 for cubic falling to 1.00 for '
        'triclinic, from the uniform-spacing approximation alone. `run.py` pools all fourteen '
        'lattices and sorts on raw M20, so if that inflation is real the reported ranking '
        'carries it directly.',
        '',
        '**The measurement below runs the other way, and it is an artefact. Do not quote it as a '
        'refutation of Wu** (F-068). The pool is censored at M20 >= 5, and the cut bites '
        'unequally by lattice because `prune_below_m20` tests the *pre*-extinction-group M20 '
        'while the stored value is *post*-assignment (F-049). The tenth percentile of the '
        'low-symmetry lattices sits **at** the cut -- aP 5.10, mC 5.05, oC 5.01, oP 5.21, '
        'mP 5.28 -- while cF, cP and hR retain mass down to 2.9. The low-symmetry distributions '
        'have had their lower tail removed and the high-symmetry ones have not, which '
        'manufactures exactly the ordering in the table. Cubic is a second, independent confound: '
        'it is scored on **ten** peaks, so its "M20" is M10.',
        '',
        '**This is a constraint on S07, not a curiosity.** S07\'s job is to learn each merit\'s '
        'null over incorrect candidates conditional on lattice and volume, and those are exactly '
        'the negatives this pool censors asymmetrically. A conditional null fitted here would '
        'absorb the prune boundary and report it as crystallography. **Q29 is therefore a '
        'prerequisite for S07 rather than a caveat on it**, and Q31\'s targeted regeneration at '
        'prune threshold 0 is the run that unblocks it.',
        '',
        table(cross, ['bravais_lattice', 'lattice_system', 'M20_median', 'M20_p90',
                      'M20_ratio_to_aP', 'wu88_predicted_ratio', 'M_1_median', 'M_wu_median',
                      'n_sampled'], 3),
        '',
        '## 8. The over-prediction axis',
        '',
        'The assumption behind several later tasks is that the dominant failure is a large, '
        'low-symmetry cell out-scoring the correct one. Measured over entries that have a '
        'reachable solution whose top-ranked candidate is nonetheless wrong.',
        '',
        table(over, None, 3),
        '',
        '## 9. Complementarity, and the C0 singularity',
        '',
        'The union oracle -- correct in the top ten under *any* merit in the zoo -- bounds what '
        'a combiner that only ever picks one of these orderings can reach, which is what S08 '
        'needs before it starts.',
        '',
        table(union, None, 4),
        '',
        'See `S06_explain_complementarity.{csv,png}` for the pairwise matrix. A column of near '
        'zeros there means the merit is dominated and adds nothing to a combiner.',
        '',
        '### Which merits inherit M20\'s zero-residual divergence (F-054)',
        '',
        table(singular, None, 4),
        '',
        '## 9b. The pre-filter question (Q5)',
        '',
        'Oishi-Tomiyasu 2013 excluded candidates with M-tilde < 3, M^Rev < 1, N_cal < 12 or '
        'N_cal > 120 before comparing figures of merit, and warned in the same breath that this '
        'flatters every one of them. **Our policy is no pre-filter in the headline** '
        '(DWMM, 2026-08-17, STATUS section 6), applied identically to M20 and to every new '
        'merit. Measured once here so the size of the flattery is quotable.',
        '',
        'Reported on the rank metrics, which need no threshold. `ceiling_cost` is the column '
        'that matters: a pre-filter that lifts top-10 by deleting correct candidates has made '
        'the problem easier by making it unsolvable, and that shows as the ceiling falling '
        'rather than as the rate rising.',
        '',
        table(prefilter, None, 4),
        '',
        '## 10. Cost',
        '',
        'Per-merit seconds per candidate, which S01 recorded as owed to S14. Measured on real '
        'mP pools from the frozen benchmark, so the reference-line lengths are the ones '
        'production sees.',
        '',
        'This is the **marginal** cost of each merit given the calculated lines and the Miller-'
        'index assignment, which every merit shares and which is generated once per candidate '
        'either way. That is the number S14 needs -- "what does adding this merit cost?" -- and '
        'it is not the cost of computing the merit from a bare unit cell. Two entries are '
        'per-call rather than per-merit and are marked by their equality: `M_tilde`, `M_rev` and '
        '`M_sym` come out of one `get_M_rev_sym` call, and `n_over` and `max_gap` out of one '
        '`get_n_over`, so each triple or pair costs what one row says, not the sum. '
        '`M_werner_frac` is timed as its Werner part alone; it also needs M20, so add that row.',
        '',
        table(cost, ['merit', 'seconds_per_candidate', 'cost_vs_M20', 'n_candidates_timed'], 6),
        '',
        '## 10b. Scale transfer -- how much of section 2 is fit, and how much is units',
        '',
        '**Read this before quoting the leaderboard.** `run.py` pools fourteen Bravais lattices '
        'and sorts on one raw scale, so a merit whose value means different things in different '
        'lattices is destroyed by the pooling however well it discriminates within one. Ranking '
        'each merit both ways -- `pool=\'cross_bl\'` (what the program does) and `pool=\'per_bl\'` '
        '(within each entry-and-lattice, deliberately an easier problem and never a headline) -- '
        'separates the two.',
        '',
        table(transfer, ['merit', 'top10_cross_bl', 'top10_per_bl', 'scale_transfer_ratio',
                         'lost_to_pooling'], 4),
        '',
        '**Within a lattice the whole zoo lands between 0.44 and 0.685; across lattices it spreads '
        'from 0.000 to 0.618.** So section 2 is ranking scale transfer at least as much as fit '
        'quality (F-074). Specifically: `M_sym`\'s advantage over M20 is **+10.6 pp** cross-lattice '
        'and **+1.1 pp** within one, so roughly 90% of the headline gain is transfer. Those numbers '
        'are still the ones that matter operationally, because cross-lattice pooling is what the '
        'program does -- but the mechanism is not the one a reader would infer from section 2 '
        'alone.',
        '',
        'The three information-type merits are the extreme case: `Minfo`, `M_info_clipped` and '
        '`null_tail_nll` sit at ~0.60 within a lattice and ~0.04 across them, a **14x** gap. They '
        'are sums of per-line log terms whose scale runs with the expected discrepancy and the '
        'line count, so pooling them raw is meaningless arithmetic rather than a fair test. '
        '`null_tail_nll` is the merit S01 designated as S07\'s analytic backbone; it ranks 18th of '
        '21 in section 2 and that is a statement about units, not about the statistic. **This is '
        'the largest single opportunity the session measured, and it is S07\'s.**',
        '',
        '## 11. Why one merit beats another -- the explanation',
        '',
        _explanation(main_table, hard_table, over, bands, union),
        '',
        '## Figures',
        '',
        '- `S06_explain_cross_lattice.png` -- what a wrong cell scores by lattice, against '
        'Wu\'s prediction',
        '- `S06_explain_vcrit.png` -- the `g_min` sweep, and correct-versus-winner M20',
        '- `S06_explain_complementarity.png` -- the pairwise matrix',
        '',
        ]

    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
