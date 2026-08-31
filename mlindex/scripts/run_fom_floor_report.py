"""S08 -- read the reproducibility floor off an ensemble of runs of the indexer.

`submit_fom_floor_arms.sh` indexes one set of patterns several times, changing only the search
seed. This turns those arms into the three numbers every gate in the campaign is written against.

    python mlindex/scripts/run_fom_floor_report.py \
        --arm-root $SCRATCH/fom_campaign2/floor \
        --benchmark $SCRATCH/fom_campaign2/pool \
        --artifact-dir docs/fom_campaign2/artifacts

Three quantities, which campaign 1 conflated into one recalled number:

  value floor     Shirley's own quantity: for a pattern whose correct solution is found in more
                  than one arm, how far apart are the merit values the arms report for it. This
                  is what "figures of merit differing by more than 10 percent for the same
                  solution" means, measured rather than recalled.
  metric floor    How far apart the arms' reported *rates* are -- the operating point, top-10.
  contrast floor  How far apart the arms put the *difference between two merits*. A gate asks
                  "does A beat B by more than the floor", and A and B are computed on the same
                  pool, so their difference is far more stable than either. PROTOCOL section 8
                  makes this the number a gate is read against, in standard errors.

Four things this does that campaign 1's version did not.

**The aggregate is composed from the reporting split's per-lattice counts, not the sample's.**
The floor sample is drawn balanced across lattices, because a proportional draw leaves cF and cI
with one or two patterns and no per-lattice floor at all -- and PROTOCOL section 8 requires a
per-lattice claim to be read against that lattice's own floor. Campaign 1's sample was
proportional, so it could scale by the sample's own composition; S08's acceptance condition 4 says
in terms that a sample drawn any other way needs the split's own counts instead. Those counts are
written beside the sample by `run_fom_floor_entries.py` and are read here, never re-derived.

**No weighted branch.** Campaign 1's `induced_standard_error_stratified` computed the standard
error of a CNRS-reweighted rate, and its `weights=None` meant *weighted* -- the opposite of the
module's convention, in the one calculation every gate is read against. PROTOCOL section 3 rule 6
makes every aggregate unweighted, so that function has no caller and is not ported; the unweighted
standard error is the plain `induced_from_differences`. Leaving it out is the fix.

**The generation floor is reported separately from the scoring floor.** Campaign 1's F-150
concluded the floor is "scoring noise, not generation noise" because its arms' operating-point
flips and reachability flips were disjoint. On this campaign's hard stratum that already fails:
two runs of the same 972 pattern-conditions agreed on 328 and disagreed on 130 about whether a
pattern is solvable at all (C2-F-031, C2-R-005). So the two are reported side by side and the
disjointness is re-tested rather than assumed.

**Every arm's peak lists are compared before any number is read**, and a mismatch raises. The
whole design is that only the search seed moved; if `--seed` moved too, the arms differ in their
data and the spread is not a floor.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics


# The merits the floor is measured for. Every one is in `FomMetrics.RANK_EXACT_MERITS`, so rank
# metrics on the subsampled arms are exact to the pool's own depth K -- which a learned score's
# would not be (C2-F-077). A floor for a learned score has to be measured on a fully retained arm.
# All seven, not five: `n_over` and `max_gap` had no floor at all, and PROTOCOL section 8 requires
# a per-merit gate to be read against that merit's own floor. Three of the seven are
# lower-is-better and are oriented from `FomMetrics.HIGHER_IS_BETTER`, never from a literal here
# -- passing none at all is what reversed `X_N` in every table this script wrote (C2-F-085).
FLOOR_MERITS = ('M20', 'M_sym', 'M_rev', 'M_tilde', 'X_N', 'n_over', 'max_gap')

# Metrics reported per arm. `found` is the pool's ceiling and moves only when the *generation*
# differs between arms, which is why it is here: it separates "the score moved" from "the pool
# moved".
FLOOR_METRICS = ('operating_point', 'top1', 'top10', 'threshold_only', 'found', 'reported',
                 'false_positive', 'mrr', 'precision')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='S08 -- the reproducibility floor, from an ensemble of runs of the indexer')
    parser.add_argument('--arm-root', type=str, required=True,
                        help='Directory holding one subdirectory per arm, each a generated pool')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Benchmark B. Included as the first arm, restricted to the floor '
                             'entries -- it was generated at a recorded search seed and a subset '
                             'run reproduces it bit for bit (C2-F-058), so it is an arm already')
    parser.add_argument('--composition', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts',
                                             'S08_floor_composition.csv'),
                        help='Per-lattice counts of the reporting split, from '
                             'run_fom_floor_entries.py. The aggregate is composed with these '
                             'because the sample is balanced rather than proportional')
    parser.add_argument('--artifact-dir', type=str,
                        default=os.path.join('docs', 'fom_campaign2', 'artifacts'))
    parser.add_argument('--threshold', type=float, default=None,
                        help='Accept threshold, selected on fom-train and passed in. Omit for '
                             'rank metrics only')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--thresholds-json', type=str, default=None,
                        help="S09's {tag}_thresholds.json. Gives each merit its own operating "
                             'threshold, which is what the operating-point floor needs: one '
                             'scalar across merits that differ in scale by orders of magnitude is '
                             'the wrong cut for all but one of them. Overrides --threshold')
    parser.add_argument('--tag', type=str, default='S08_floor')
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------------------
# The arithmetic. Ported from `fom` @ 7c137c3, less the weighted branch.
# ----------------------------------------------------------------------------------------
def relative_spread(values):
    """(max - min) / |median| across the replicate columns of each row.

    Range rather than standard deviation because Shirley's statement is about how far apart two
    reported values can be, and because a handful of arms do not support a variance estimate worth
    the name. Divided by the row's own median so the answer is scale-free and comparable across
    merits whose units differ by orders of magnitude.
    """
    scale = values.abs().median(axis=1).replace(0.0, np.nan)
    return (values.max(axis=1) - values.min(axis=1)) / scale


def flip_rate(first, second):
    """Fraction of entries whose boolean outcome differs between two arms."""
    return float(np.mean(np.asarray(first, dtype=bool) != np.asarray(second, dtype=bool)))


def induced_standard_error(rate, n_entries):
    """Standard error of the *difference* in a per-entry rate between two runs, from the flip rate.

    If a fraction `f` of entries change their outcome between two arms, the change in the reported
    rate is the mean of n draws that are 0 with probability 1 - f and +-1 otherwise, so its
    variance is f/n and its standard error is sqrt(f/n). This is the derived form; the measured
    form is `induced_from_differences`. S08 acceptance condition 3 is that the two agree --
    campaign 1 got 0.366 pp derived against 0.360 pp reported.
    """
    return float(np.sqrt(rate / n_entries))


def induced_from_differences(differences, n_entries):
    """Standard error at `n_entries` of a per-entry difference measured on this sample.

    Given the per-entry change `d_i` between two arms -- of an outcome flag, or of the *contrast*
    between two merits' flags -- the change in the reported mean is `mean(d)`, so its standard
    error at any entry count is `sd(d)/sqrt(n)`. Preferred to the flip-rate form wherever the
    per-entry values are in hand, because it needs no assumption about how flips are distributed.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return float('nan')
    return float(np.std(differences, ddof=1) / np.sqrt(n_entries))


def compose_aggregate(per_lattice, composition):
    """An aggregate standard error from per-lattice ones, using the SPLIT's composition.

    Var(aggregate) = sum_bl (n_bl/N)^2 * Var_bl / n_bl, so the aggregate standard error is
    sqrt(sum_bl w_bl^2 * se_bl^2 * (n_sample_bl / n_split_bl)) -- each lattice's measured standard
    error rescaled from the sample's count to the split's, then combined at the split's weights.

    This is the step campaign 1 did not need. Its floor sample was drawn proportional to its
    reporting split, so `n_bl` scaled by the sample's own composition and the aggregate was a
    plain mean. This campaign's sample is balanced, so the sample's shape is NOT the split's, and
    using it would weight cF -- 20 entries of 3 810 -- as though it were a fourteenth of the
    population. S08 acceptance condition 4 is exactly this check.
    """
    weights = composition.set_index('bravais_lattice')
    variance = 0.0
    covered = 0.0
    for row in per_lattice.itertuples():
        lattice = row.bravais_lattice
        if lattice not in weights.index:
            continue
        weight = float(weights.loc[lattice, 'floor_weight'])
        # `se_at_split_pp` is already this lattice's floor rescaled from the sample's SOURCE-ENTRY
        # count to the split's, so only the weighting is left. Doing the rescale here from
        # `n_entries` was wrong twice over: it mixed units -- pattern-conditions against source
        # entries -- and it applied to a standard error that had not been clustered.
        rescaled = getattr(row, 'se_at_split_pp', None)
        if rescaled is None or not np.isfinite(rescaled):
            continue
        variance += (weight**2) * (rescaled**2)
        covered += weight
    if not covered:
        return float('nan'), 0.0
    return float(np.sqrt(variance)), float(covered)


# ----------------------------------------------------------------------------------------
# Loading the arms
# ----------------------------------------------------------------------------------------
def has_threshold(args):
    """Whether a threshold was supplied at all, by either route.

    Both the contrast table and the figure switch to the operating point only when there is a
    threshold to switch on. Testing `args.threshold` alone missed the `--thresholds-json` route and
    silently reported the top-10 contrast under an operating-point heading -- which is the exact
    shape of C2-F-085, a wrong number that raises nothing.
    """
    return args.threshold is not None or bool(getattr(args, '_thresholds', None))


def per_merit_threshold(args, merit):
    """This merit's own operating threshold, or the single scalar, or none.

    S08 measured the floor on top-10 *without* a threshold and left the operating-point version to
    S09, which is the step that selects one. But a single `--threshold` across the whole zoo is not
    a meaningful instruction: the merits differ in scale by orders of magnitude -- M20 selects
    12.2, `M_sym` 28.6, `X_N` 0 -- so one scalar is the wrong cut for every merit but one.

    `--thresholds-json` takes the file S09's eval driver writes and gives each merit its own. The
    stored value is in `per_entry`'s internal orientation, where every score is higher-is-better,
    so a lower-is-better merit's threshold is negated there and has to be turned back before
    `evaluate` mirrors it a second time.
    """
    if getattr(args, '_thresholds', None):
        choice = args._thresholds.get(merit)
        if choice is None:
            return None
        stored = float(choice['threshold'])
        return stored if FomMetrics.orientation_of(merit) else -stored
    return args.threshold


def arm_frames(root, merit_dir=None, columns=None):
    """One arm's candidate frames, merits joined on -- `FomBenchmark.bundle_frames` by another name.

    Kept as a name because this script and its tests read better for it, but the implementation
    moved so that S09's zoo drivers and this report cannot drift apart in how they join a sidecar.
    `require_merits` is deliberately not passed on: S08's arms predate the widened sidecar and a
    six-column one is complete for what this report asks of it.
    """
    return FomBenchmark.bundle_frames(root, merit_dir=merit_dir, columns=columns)


def arm_directories(root):
    """One directory per arm. A pool written per (arm, condition) nests one level deeper."""
    root = Path(root)
    paths = sorted(path for path in root.iterdir() if path.is_dir())
    if len(paths) < 1:
        raise SystemExit(f'{root} holds no arm directories')
    return paths


def check_arms_are_comparable(arms):
    """Every arm must hold the same patterns with the same peak lists. Raises if not.

    S08 acceptance condition 5, and it runs before anything is read rather than after. The design
    is that only `--optimizer-seed` moved; `--seed` fixes the entry sample and the per-entry noise
    and therefore the peak lists. If someone varies that instead, the arms differ in their DATA,
    and their spread is generation noise and scoring noise together -- which is the one thing the
    floor exists to separate.
    """
    reference = reference_name = None
    for name, entries in arms.items():
        # Keyed on (entry, condition). Two condition bundles apply different noise to the same
        # crystal, so their peak lists differ by design; pooling them would compare quantities that
        # are meant to differ and the check would fail on correct data.
        digests = dict(zip(zip(entries['entry_id'].astype(str),
                               entries['condition_bundle'].astype(str)),
                           entries['q2_digest'].astype(str)))
        if reference is None:
            reference, reference_name = digests, name
            continue
        shared = set(reference) & set(digests)
        if not shared:
            raise SystemExit(f'Arms {reference_name} and {name} share no entries at all.')
        differing = sorted(key for key in shared if reference[key] != digests[key])
        if differing:
            raise SystemExit(
                f'Arms {reference_name} and {name} disagree on the peak lists of '
                f'{len(differing)} of {len(shared)} entries, e.g. {differing[:5]}. The arms are '
                f'not measuring the reproducibility floor: only --optimizer-seed may differ, and '
                f'--seed fixes the peak lists. Regenerate with a shared --seed.')
    return True


def load_arms(arm_root, benchmark, floor_entries):
    """{arm name: (root, entries)}, with Benchmark B restricted in as the first arm.

    Benchmark B is an arm and costs nothing to be one: it was generated at a recorded search seed
    and a run restricted to a subset of entries reproduces it bit for bit (C2-F-058), so its rows
    for these patterns ARE the arm at that seed. Only the others had to be generated.
    """
    arms = {}
    for path in arm_directories(arm_root):
        entries = FomBenchmark.load_entries(path)
        arms[path.name] = (path, entries)
    if benchmark is not None:
        entries = FomBenchmark.load_entries(benchmark)
        entries = entries.loc[entries['entry_id'].isin(floor_entries)].reset_index(drop=True)
        arms['benchmark'] = (Path(benchmark), entries)
    if len(arms) < 2:
        raise SystemExit(f'The floor needs at least two arms; found {sorted(arms)}. '
                         f'Pass --benchmark to include Benchmark B as the first one.')
    return arms


# ----------------------------------------------------------------------------------------
# The figure
# ----------------------------------------------------------------------------------------
# Free cell parameters per Bravais lattice, which is the order the floor is expected to follow --
# campaign 1 found it spanning two orders of magnitude ordered this way, largest exactly on the
# low-symmetry lattices where this campaign's gains are expected to live. Plotting alphabetically
# would hide the one structural claim the figure exists to make.
FREE_PARAMETERS = {
    'cF': 1, 'cI': 1, 'cP': 1,
    'hP': 2, 'hR': 2, 'tI': 2, 'tP': 2,
    'oC': 3, 'oF': 3, 'oI': 3, 'oP': 3,
    'mC': 4, 'mP': 4,
    'aP': 6,
    }

# Where this campaign's gains are expected, from every measured result in campaign 1 and S03/S04.
# They are also the least reproducible lattices, which is the point of the figure.
GAIN_LATTICES = ('mC', 'mP', 'aP')

# Light-mode slots 1 and 3 of the reference categorical palette, plus its ink and surface. Two
# hues only: the story is one number per lattice, so the figure uses EMPHASIS -- the lattices where
# the gains live carry the accent and the rest recede -- rather than fourteen categorical hues.
# Validated: adjacent CVD Delta E 24.7 (protan), normal-vision 33.6, both clear of the floors.
ACCENT = '#2a78d6'
RECEDE = '#9a9a94'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
SURFACE = '#fcfcfb'
GRID = '#e3e3df'
CONDITION_HUES = ('#2a78d6', '#eb6834', '#1baf7a')


def _style(pyplot):
    """Hairline, recessive chrome. Thin marks and a grid one shade off the surface."""
    pyplot.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif', 'font.size': 8,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.6,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK_SECONDARY, 'ytick.color': INK_SECONDARY,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'legend.frameon': False, 'legend.fontsize': 7.5,
        })


def _condition_label(tag):
    """`c2_error1_cont0` -> `nominal`. Raw tags on an axis are unreadable and all look alike."""
    from mlindex.model_training import FomConditions
    condition = FomConditions.BY_TAG.get(tag)
    return condition.key if condition is not None else tag


def figure(by_lattice, aggregate, by_condition, out_path, metric='operating_point',
           merit='M_sym'):
    """The floor figure. Two panels on ONE shared axis, which is what makes it an argument.

    (a) The floor is **ordered by free cell parameters**, spanning orders of magnitude, and is
        largest exactly on the lattices where the campaign's gains are expected. That is why
        PROTOCOL section 8 requires a per-lattice claim to be read against that lattice's own
        floor: reading an aP result against an aggregate dominated by cubic lattices would accept
        noise as a result.

    (b) On the same scale, moving the **condition** barely moves it at all. Campaign 1 collapsed
        the operating point by a factor of 5.7 between conditions while the metric floor moved by
        1.4 pp (F-150), which is what makes a floor expressed as a fraction of the baseline wrong
        in shape as well as in size.

    The panels share an x-axis deliberately: the claim is not "the floor varies by lattice" and
    "the floor is stable in condition" as two separate facts, but that **one of these axes matters
    and the other does not**, and that comparison is only legible on one scale.

    The value floor is deliberately not a third panel. It is a percentage of a merit *value* while
    these are percentage points of a *rate*; putting two units on one figure is the mistake a
    second axis usually is. It goes in the results table.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pyplot

    _style(pyplot)
    figure_handle, (left, right) = pyplot.subplots(
        1, 2, figsize=(7.2, 3.2), sharex=True, layout='constrained',
        gridspec_kw=dict(width_ratios=[1.5, 1.0]))

    # ONE merit. Different merits have different floors -- M_sym's aggregate is 0.19 pp against
    # M_rev's 0.26 -- so averaging them would plot a quantity no gate is ever read against.
    rows = by_lattice.loc[(by_lattice['metric'] == metric)
                          & (by_lattice['merit'] == merit)].copy()
    if rows.empty:
        # An empty figure renders perfectly and says nothing, which is worse than not rendering.
        raise SystemExit(
            f'No per-lattice rows for metric {metric!r}; the table has '
            f'{sorted(by_lattice["metric"].unique())} for merits '
            f'{sorted(by_lattice["merit"].unique())}. Without --threshold the operating point is '
            f'identically top10 and is not reported, so the figure is drawn on top10.')
    rows = rows.groupby('bravais_lattice', as_index=False).agg(
        se_pp=('se_pp', 'mean'), n_entries=('n_entries', 'sum'))
    rows['free'] = rows['bravais_lattice'].map(FREE_PARAMETERS)
    rows = rows.dropna(subset=['free']).sort_values(['free', 'bravais_lattice'])
    positions = np.arange(rows.shape[0])
    colours = [ACCENT if name in GAIN_LATTICES else RECEDE
               for name in rows['bravais_lattice']]

    # height < 1 leaves the surface gap between adjacent bars; no borders on the marks.
    left.barh(positions, rows['se_pp'], height=0.7, color=colours, linewidth=0)
    left.set_yticks(positions)
    left.set_yticklabels([f'{name} ({int(free)})' for name, free
                          in zip(rows['bravais_lattice'], rows['free'])])
    left.invert_yaxis()
    left.set_xlabel('contrast floor, one standard error (pp)')
    # The title is COMPUTED, not asserted. Campaign 1 found this floor "ordered by free cell
    # parameters over two orders of magnitude"; on this campaign's pool it is not (C2-F-081). A
    # hardcoded caption would have shipped the inherited claim over the measurement.
    ratio = float(rows['se_pp'].max()/rows['se_pp'].min()) if rows['se_pp'].min() else float('nan')
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(rows['free'], rows['se_pp']).statistic)
        relation = f'rank correlation with free parameters {rho:+.2f}'
    except Exception:
        relation = ''
    left.set_title(f'(a) the floor spans {ratio:.1f}x across lattices\n{relation}',
                   loc='left', color=INK)
    left.xaxis.grid(True, color=GRID, linewidth=0.6)
    left.set_axisbelow(True)
    for spine in ('top', 'right', 'left'):
        left.spines[spine].set_visible(False)

    # No legend box: this is one series with emphasis, not two series, and the skill's own rule is
    # that a single series is named by the title or a direct label. The accent group is annotated
    # once, beside itself, which also keeps identity off colour alone.
    # Top right, which is the only clear space on this axis. The accented lattices are the longest
    # bars by construction -- that is the figure's point -- so there is nothing beside them and
    # nothing below them either: the bottom rows ARE them.
    # A figure-level caption, not an in-axes annotation. With the floor this flat there is no
    # clear space left inside panel (a) -- every bar reaches past the middle -- and a note placed
    # anywhere inside it lands on a mark.
    caption = ''
    accent_names = [name for name in rows['bravais_lattice'] if name in GAIN_LATTICES]
    if accent_names:
        accent_rank = rows['se_pp'].rank(ascending=False)
        worst = int(accent_rank[rows['bravais_lattice'].isin(GAIN_LATTICES)].min())
        caption = (f"{', '.join(accent_names)} (blue) are where this campaign's gains are "
                   f'expected; the highest floor among them ranks {worst} of {rows.shape[0]}.')

    composed = aggregate.loc[(aggregate['metric'] == metric)
                             & (aggregate['merit'] == merit), 'se_pp']
    aggregate_value = float(composed.iloc[0]) if composed.shape[0] else None
    if aggregate_value is not None:
        for axis in (left, right):
            axis.axvline(aggregate_value, color=INK, linewidth=0.9, zorder=3)
        left.annotate(f'aggregate {aggregate_value:.2f}', xy=(aggregate_value, -0.7),
                      xytext=(3, 0), textcoords='offset points',
                      color=INK, fontsize=7.5, ha='left', va='center')

    # Direct-label the two extremes only. A number on every bar is the anti-pattern.
    for position, row in zip(positions, rows.itertuples()):
        if position in (positions[0], positions[-1]):
            left.annotate(f'{row.se_pp:.2f}', xy=(row.se_pp, position),
                          xytext=(3, 0), textcoords='offset points',
                          va='center', fontsize=7.5, color=INK_SECONDARY)

    # (b) the same quantity under each condition, on the same scale. One hue, because the story is
    # that these are the SAME number -- three categorical hues would assert they are three things
    # to be told apart.
    conditions = by_condition.loc[(by_condition['metric'] == metric)
                                  & (by_condition['merit'] == merit)]
    if conditions.shape[0]:
        grouped = (conditions.groupby('condition_bundle', as_index=False)['sd_pp'].mean()
                   .sort_values('sd_pp'))
        spots = np.arange(grouped.shape[0])
        right.scatter(grouped['sd_pp'], spots, s=42, color=ACCENT, zorder=3,
                      edgecolors=SURFACE, linewidths=1.2)
        right.set_yticks(spots)
        right.set_yticklabels([_condition_label(tag) for tag in grouped['condition_bundle']])
        right.invert_yaxis()
        right.set_ylim(grouped.shape[0] - 0.4, -0.9)
        # Visible values: the relief rule, and three dots on a wide axis need their numbers.
        # Labels to the LEFT of each dot: everything below the smallest value is empty space, so
        # they cannot collide with each other, with the aggregate rule, or with the panel edge.
        # To the RIGHT of each dot. The dots sit near the left edge of a shared axis that runs to
        # the widest per-lattice floor, so everything to their right is empty while the space to
        # their left is occupied by the tick labels.
        for spot, value in zip(spots, grouped['sd_pp']):
            right.annotate(f'{value:.2f}', xy=(value, spot), xytext=(9, 0),
                           textcoords='offset points', ha='left', va='center', fontsize=7.5,
                           color=INK_SECONDARY)
        spread = float(grouped['sd_pp'].max() - grouped['sd_pp'].min())
        right.annotate(f'total spread {spread:.2f} pp', xy=(0.98, 0.04),
                       xycoords='axes fraction',
                       color=INK_SECONDARY, fontsize=7.5, ha='right', va='bottom')
    right.set_xlabel('contrast floor, one standard error (pp)')
    right.set_title('(b) the condition barely moves it', loc='left', color=INK)
    right.xaxis.grid(True, color=GRID, linewidth=0.6)
    right.set_axisbelow(True)
    for spine in ('top', 'right', 'left'):
        right.spines[spine].set_visible(False)

    if caption:
        figure_handle.text(0.005, -0.02, caption, fontsize=7.5, color=INK_SECONDARY,
                           ha='left', va='top')
    figure_handle.savefig(out_path, dpi=300, bbox_inches='tight')
    pyplot.close(figure_handle)
    return out_path


def main(argv=None):
    args = _parse_args(argv)
    composition = pd.read_csv(args.composition)
    args._thresholds = None
    if getattr(args, 'thresholds_json', None):
        import json
        args._thresholds = json.loads(
            Path(args.thresholds_json).read_text(encoding='utf-8'))['choices']
        print(f'per-merit thresholds from {args.thresholds_json}: '
              + ', '.join(f'{m} {c["threshold"]:.4g}' for m, c in sorted(args._thresholds.items())))

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    entries_path = artifact_dir / f'{args.tag}_entries.csv'
    floor_entries = set(pd.read_csv(entries_path)['identifier'].astype(str))
    arms = load_arms(args.arm_root, args.benchmark, floor_entries)
    check_arms_are_comparable({name: entries for name, (_, entries) in arms.items()})
    print(f'{len(arms)} arms, peak lists identical across all of them')

    # One MetricsResult per (arm, merit). The arms are subsampled the same way Benchmark B is, so
    # every merit here has to be one the subsampler ranked on, or the ranks are not exact.
    results = {}
    for name, (path, entries) in arms.items():
        keep = entries.loc[entries['entry_id'].isin(floor_entries)]
        for merit in FLOOR_MERITS:
            depth, subsampled = FomBenchmark.subsample_depth(path)
            wanted = list(FomMetrics.SCORE_INDEPENDENT_COLUMNS) + ['condition_bundle', merit]
            results[(name, merit)] = FomMetrics.evaluate(
                arm_frames(path, columns=wanted), score=merit,
                higher_is_better=FomMetrics.orientation_of(merit),
                threshold=per_merit_threshold(args, merit), top_n=args.top_n,
                entries=keep, n_bootstrap=0,
                # Read from the arm's own manifest rather than assumed: an iterable of frames
                # carries no manifest, and 'auto' would take it for a full pool.
                subsample_top_k=depth if subsampled else None,
                )
        print(f'  {name}: {results[(name, FLOOR_MERITS[0])].meta["n_entries"]} cells')

    split_counts = dict(zip(composition['bravais_lattice'], composition['split_entries']))
    metric_rows, contrast_rows, lattice_rows = [], [], []
    arm_names = sorted(arms)
    for merit in FLOOR_MERITS:
        table = pd.DataFrame({name: [results[(name, merit)].metric(metric)
                                     for metric in FLOOR_METRICS]
                              for name in arm_names}, index=list(FLOOR_METRICS))
        for metric in FLOOR_METRICS:
            values = table.loc[metric].to_numpy(dtype=np.float64)*100.0
            metric_rows.append(dict(
                merit=merit, metric=metric,
                mean=float(np.nanmean(values)), sd_pp=float(np.nanstd(values, ddof=1)),
                range_pp=float(np.nanmax(values) - np.nanmin(values)),
                n_arms=len(arm_names),
                **{f'arm_{name}': float(value) for name, value in zip(arm_names, values)},
                ))

        # The contrast floor, and the per-lattice floors, both from paired per-entry flags.
        #
        # `operating_point` is reported ONLY when a threshold was given. Without one it is
        # identically `top10` -- `derive_flags` accepts every candidate -- so reporting both would
        # publish the same number twice under two names, one of which a reader would take for the
        # headline criterion. Choosing a threshold is S09's, on `fom-train`.
        baseline = 'M20'
        metrics = ('top10',) + (('operating_point',) if has_threshold(args) else ())
        if merit != baseline:
            for metric in metrics:
                clustered = _paired_contrast(results, arm_names, merit, baseline, metric)
                if clustered is None:
                    continue
                n_source = int(clustered['entry_id'].nunique())
                contrast_rows.append(dict(
                    merit=merit, baseline=baseline, metric=metric,
                    # The effect: what the merit is actually worth against the baseline. Reported
                    # beside its floor so a gate is a ratio and not a comparison of two tables.
                    contrast_pp=float(clustered['contrast'].mean()*100.0),
                    floor_pp=_floor_from(clustered),
                    standard_errors=abs(float(clustered['contrast'].mean()*100.0))
                    / _floor_from(clustered) if _floor_from(clustered) else float('nan'),
                    # ~0 by construction: it is the same quantity measured twice. A value far from
                    # zero would mean the arms differ systematically, not by their search seed.
                    mean_shift_pp=float(clustered['shift'].mean()*100.0),
                    n_source_entries=n_source,
                    n_pairs=int(clustered['pair'].nunique()),
                    ))
                for lattice, block in clustered.groupby('bravais_lattice'):
                    target = split_counts.get(lattice)
                    lattice_rows.append(dict(
                        merit=merit, baseline=baseline, metric=metric, bravais_lattice=lattice,
                        contrast_pp=float(block['contrast'].mean()*100.0),
                        se_pp=_floor_from(block),
                        se_at_split_pp=_floor_from(block, n_target=target),
                        n_entries=int(block['entry_id'].nunique()),
                        n_split_entries=int(target) if target else None,
                        ))

    # Per condition, so "the floor barely moves with the condition" is re-measured rather than
    # inherited from F-150. Read off the condition_bundle stratum of each arm's own result, which
    # is already computed -- a separate evaluate() per bundle would be a second pass over the pool
    # for numbers the first pass produced.
    condition_rows = []
    for merit in FLOOR_MERITS:
        per_arm = {}
        for name in arm_names:
            stratum = results[(name, merit)].stratum('condition_bundle')
            for row in stratum.itertuples():
                for metric in ('operating_point', 'top10', 'found'):
                    per_arm.setdefault((row.level, metric), {})[name] = getattr(row, metric)
        for (bundle, metric), values in per_arm.items():
            series = np.array([values.get(name, np.nan) for name in arm_names],
                              dtype=np.float64)*100.0
            condition_rows.append(dict(
                merit=merit, metric=metric, condition_bundle=bundle,
                mean=float(np.nanmean(series)),
                sd_pp=float(np.nanstd(series, ddof=1)) if np.isfinite(series).sum() > 1 else np.nan,
                range_pp=float(np.nanmax(series) - np.nanmin(series)),
                n_arms=int(np.isfinite(series).sum()),
                ))

    # S08 acceptance condition 3: the floor derived from the flip rate must match the floor
    # measured from the spread. They are different estimators of one quantity -- sqrt(f/n) assumes
    # the flips are independent Bernoulli, sd(d)/sqrt(n) assumes nothing -- so agreement is
    # evidence that the arms differ by their search seed and not by something structural.
    # Campaign 1 got 0.366 derived against 0.360 reported and that is the standard to match.
    derivation_rows = []
    for merit in FLOOR_MERITS:
        for metric in ('top10', 'top1', 'found'):
            flips, measured = [], []
            for index, left_name in enumerate(arm_names):
                for right_name in arm_names[index + 1:]:
                    left = results[(left_name, merit)].per_entry
                    right = results[(right_name, merit)].per_entry
                    joined = left[['entry_id', 'condition_bundle', metric]].merge(
                        right[['entry_id', 'condition_bundle', metric]],
                        on=['entry_id', 'condition_bundle'], suffixes=('_a', '_b'),
                        validate='1:1')
                    # Clustered to the source entry, as everything else here is.
                    joined['flip'] = (joined[f'{metric}_a'].astype(bool)
                                      != joined[f'{metric}_b'].astype(bool))
                    joined['shift'] = (joined[f'{metric}_a'].astype(float)
                                       - joined[f'{metric}_b'].astype(float))
                    clustered = joined.groupby('entry_id', as_index=False)[['flip', 'shift']].mean()
                    n_source = clustered.shape[0]
                    # The derived form takes the number of ROWS the flip rate is computed over,
                    # not the number of clusters. Campaign 1's arms carried ONE condition bundle,
                    # so rows and entries coincided and the distinction never showed; on a
                    # three-condition pool, using the cluster count overstates the derived floor by
                    # exactly sqrt(3) and the check fails for a reason that is not a defect.
                    flips.append(induced_standard_error(
                        float(clustered['flip'].mean()), joined.shape[0])*100.0)
                    measured.append(float(np.std(clustered['shift'].to_numpy()*100.0, ddof=1)
                                          / np.sqrt(n_source)))
            if not flips:
                continue
            derived_pp, measured_pp = float(np.mean(flips)), float(np.mean(measured))
            derivation_rows.append(dict(
                merit=merit, metric=metric, derived_pp=derived_pp, measured_pp=measured_pp,
                ratio=derived_pp/measured_pp if measured_pp else float('nan'),
                relative_gap=abs(derived_pp - measured_pp)/measured_pp if measured_pp
                else float('nan'),
                n_pairs=len(flips)))
    derivation_table = pd.DataFrame(derivation_rows)

    metric_table = pd.DataFrame(metric_rows)
    condition_table = pd.DataFrame(condition_rows)
    contrast_table = pd.DataFrame(contrast_rows)
    lattice_table = pd.DataFrame(lattice_rows)

    # The aggregate, composed from the split's own counts because the sample is balanced.
    aggregate_rows = []
    for (merit, metric), block in lattice_table.groupby(['merit', 'metric']):
        se, covered = compose_aggregate(block, composition)
        # Composed at the SAME split weights as the standard error beside it. A plain mean here
        # would be a macro average over lattices sitting next to a split-weighted standard error,
        # and their ratio -- which is what a gate reads -- would be of two different populations.
        weights = composition.set_index('bravais_lattice')['floor_weight']
        aligned = block.set_index('bravais_lattice')
        shared = [name for name in aligned.index if name in weights.index]
        total = float(weights.loc[shared].sum())
        contrast = (float((aligned.loc[shared, 'contrast_pp']*weights.loc[shared]).sum()/total)
                    if total else float('nan'))
        aggregate_rows.append(dict(merit=merit, baseline='M20', metric=metric,
                                   contrast_pp=contrast, se_pp=se,
                                   standard_errors=abs(contrast)/se if se else float('nan'),
                                   lattice_weight_covered=covered,
                                   composed_from='split_entries'))
    aggregate_table = pd.DataFrame(aggregate_rows)

    for name, frame in (('metric', metric_table), ('derivation', derivation_table),
                        ('by_condition', condition_table),
                        ('contrast', contrast_table),
                        ('by_lattice', lattice_table), ('aggregate', aggregate_table)):
        path = artifact_dir / f'{args.tag}_{name}.csv'
        frame.to_csv(path, index=False)
        print(f'wrote {path}')

    if not lattice_table.empty:
        # The metric the tables actually carry: without a threshold the operating point is
        # identically top10 and is not reported at all.
        figure_metric = 'operating_point' if has_threshold(args) else 'top10'
        path = figure(lattice_table, aggregate_table, condition_table,
                      artifact_dir / f'{args.tag}.png', metric=figure_metric)
        print(f'wrote {path} (metric: {figure_metric})')
    return 0


def _paired_contrast(results, arm_names, merit, baseline, metric):
    """The contrast, and how far the arms move it. Clustered on the source entry.

    Returns a frame with one row per (source entry, arm pair): `contrast` is the mean
    (merit - baseline) outcome for that crystal, and `shift` is how much that contrast moved
    between the two arms of the pair.

    **Clustered on `entry_id`, not on the (entry, condition) row.** One crystal appears under
    every condition with correlated noise, so it is one draw and not three. Treating the rows as
    independent gives a standard error up to sqrt(n_conditions) -- 1.73x here -- too tight, and a
    floor that is too tight makes every gate in the campaign too permissive. It is the first
    pitfall the S08 handoff lists and PROTOCOL section 8 states it twice.

    **All pairs, not the first two.** Four arms give six ordered pairs, and averaging over them is
    most of why three extra arms were generated rather than one.
    """
    if len(arm_names) < 2:
        return None
    per_arm = {}
    for name in arm_names:
        merged = results[(name, merit)].per_entry.merge(
            results[(name, baseline)].per_entry[['entry_id', 'condition_bundle', metric]],
            on=['entry_id', 'condition_bundle'], suffixes=('', '_base'), validate='1:1')
        merged['contrast'] = (merged[metric].astype(float)
                              - merged[f'{metric}_base'].astype(float))
        per_arm[name] = merged[['entry_id', 'condition_bundle', 'bravais_lattice', 'contrast']]

    rows = []
    for index, left_name in enumerate(arm_names):
        for right_name in arm_names[index + 1:]:
            joined = per_arm[left_name].merge(
                per_arm[right_name], on=['entry_id', 'condition_bundle', 'bravais_lattice'],
                suffixes=('_a', '_b'), validate='1:1')
            if joined.empty:
                continue
            joined['shift'] = joined['contrast_a'] - joined['contrast_b']
            joined['contrast'] = joined['contrast_a']
            # To the source entry FIRST, then across entries. This is the clustering.
            clustered = (joined.groupby(['entry_id', 'bravais_lattice'], as_index=False)
                         [['shift', 'contrast']].mean())
            clustered['pair'] = f'{left_name}|{right_name}'
            rows.append(clustered)
    return pd.concat(rows, ignore_index=True) if rows else None


def _floor_from(clustered, n_target=None):
    """The floor, in pp, from clustered per-entry shifts.

    `sd(shift)/sqrt(n)` over SOURCE ENTRIES, averaged over the arm pairs so no pair's own draw
    decides the answer. `n_target` rescales from the sample's entry count to the reporting split's,
    which is what makes a floor measured on 40 crystals a statement about the 600 the split holds.
    """
    values = []
    for _, block in clustered.groupby('pair'):
        shift = block['shift'].to_numpy(dtype=np.float64)*100.0
        if shift.size < 2:
            continue
        n_sample = shift.size
        standard_error = float(np.std(shift, ddof=1)/np.sqrt(n_sample))
        if n_target:
            standard_error *= float(np.sqrt(n_sample/n_target))
        values.append(standard_error)
    return float(np.mean(values)) if values else float('nan')


if __name__ == '__main__':
    raise SystemExit(main())
