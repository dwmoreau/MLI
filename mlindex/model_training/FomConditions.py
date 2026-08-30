"""Benchmark B's condition set, defined once and read by everything that needs it.

Campaign 1 defined its conditions in four places -- `run_fom_mirror.bundle_tag`, a partial
reimplementation of the same rule in bash (`submit_fom_dump.sh`, which omitted the `bias` and
`pbias` components), `FomMetrics.BUNDLE_LABELS`, and `FomMetrics.HARD_BUNDLES` as a literal
tuple. Two implementations of one naming rule that had to agree, and two literals that had to
be edited in step with them. This module is the single definition; `bundle_tag` here is the
only implementation of the rule, and the submit script obtains its tag by asking Python for it
rather than rebuilding it.

**Tags are prefixed `c2_`.** Campaign 1's C4 tag is `error1_cont1_drop6`, which this campaign's
sparse-6 bundle would otherwise reproduce byte for byte; one `manifest.json` is written per
`--out-dir`, so two bundles sharing a directory silently overwrite each other. The prefix and a
separate output root are both required, not either.

What differs from campaign 1's grid, and why:

* **The control is not zero error.** With no measurement error the residual denominator vanishes
  and M20 diverges arithmetically -- 9.5 % of campaign 1's control scored above 1e9, 248 were
  non-finite, and `FomMetrics.CONTROL_BUNDLES` excluded it from every metric, so 13.9 M of
  26.4 M candidates were dead weight. A small non-zero multiplier keeps the control's purpose,
  a near-100 % ceiling proving nothing upstream is broken, and leaves every residual-denominator
  merit finite. PROTOCOL §3 rule 11.
* **Sparsity is N = 2, 4, 6 and the three are nested** (DWMM, 2026-08-26): the N = 2 holes are a
  subset of the N = 4 holes, which are a subset of the N = 6 holes, so the axis is a paired
  comparison of one crystal degrading progressively rather than three independent noise draws.
  The nesting is implemented in the peak-list preparation, not here. Campaign 1's `drop10` is
  dropped.
* **The error law is Gaussian and only Gaussian** (DWMM, 2026-08-26). The axes are *severity*,
  which is the multiplier and scales sigma(q2)'s intercept and slope together, and *shape*,
  which is the intercept alone. `student_t` and a systematic 2-theta zero shift were considered
  and declined, so robustness to a different error **model** stays untested rather than passed
  -- C2-R-008, carried forward from campaign 1's R11/R12 rather than discharged.
* **One instrument.** Broadening tag `1` only; the `sa` tag is not used at all (DWMM: "it is
  unrealistic"), and the dropout condition is taken to stand in for the larger-broadening case.
  Only the `*_1` model set exists on disk in any event, and every optimizer factory interpolates
  the tag into its model directory name, so a second-tag run fails at model load.
"""

TAG_PREFIX = 'c2'

# The nominal sigma(q2) intercept scale. 1.0 is the repository's own value, which
# `ErrorAdder.q2_sigma_params` resolves from `EntryHelpers.get_peak_generation_info`. Those
# defaults are labelled `#CNRS` at their definition and are therefore fitted to the sealed
# benchmark -- kept deliberately for campaign-1 comparability, and bounded by C2-R-007.
NOMINAL_INTERCEPT_SCALE = 1.0

# Gaussian throughout, by decision. Recorded on every entry row so a bundle's provenance is
# readable from the data rather than from this file.
ERROR_LAW = 'gaussian'


class Condition:
    """One condition bundle: a named point in the grid, and the tag it generates under."""

    def __init__(self, key, label, description, axis,
                 error_multiplier, n_contaminants,
                 intercept_scale=NOMINAL_INTERCEPT_SCALE, contaminant_bias=1.0,
                 n_dropout=0, second_phase_lines=0, second_phase_bias=2.0,
                 provisional=False):
        self.key = key
        self.label = label
        self.description = description
        self.axis = axis
        self.error_multiplier = float(error_multiplier)
        self.n_contaminants = int(n_contaminants)
        self.intercept_scale = float(intercept_scale)
        self.contaminant_bias = float(contaminant_bias)
        self.n_dropout = int(n_dropout)
        self.second_phase_lines = int(second_phase_lines)
        self.second_phase_bias = float(second_phase_bias)
        # Flags a bundle DWMM has asked for tentatively. It generates like any other; the flag
        # exists so the condition table says which rows are settled and which are for review.
        self.provisional = bool(provisional)

    @property
    def tag(self):
        return bundle_tag(self)

    def __repr__(self):
        return f'<Condition {self.key} {self.tag}>'


def bundle_tag(condition):
    """The one implementation of the tag rule.

    Accepts a Condition or anything carrying the same attribute names -- argparse's namespace
    does, which is what lets the dump script tag a run assembled from command-line flags rather
    than from the table below.

    Every non-default axis has to appear or two bundles collide on their output filenames. The
    defaults are omitted so a tag stays readable, and the `c2_` prefix makes a campaign-2 bundle
    unmistakable for a campaign-1 one.
    """
    error_multiplier = float(getattr(condition, 'error_multiplier'))
    tag = (f'{TAG_PREFIX}_error{error_multiplier:g}'
           f'_cont{int(getattr(condition, "n_contaminants")):d}')
    intercept_scale = float(getattr(condition, 'intercept_scale', NOMINAL_INTERCEPT_SCALE))
    if intercept_scale != NOMINAL_INTERCEPT_SCALE:
        tag += f'_icept{intercept_scale:g}'
    contaminant_bias = float(getattr(condition, 'contaminant_bias', 1.0))
    if contaminant_bias != 1.0:
        tag += f'_bias{contaminant_bias:g}'
    n_dropout = int(getattr(condition, 'n_dropout', 0))
    if n_dropout:
        tag += f'_drop{n_dropout:d}'
    second_phase_lines = int(getattr(condition, 'second_phase_lines', 0))
    if second_phase_lines:
        tag += f'_phase{second_phase_lines:d}'
        second_phase_bias = float(getattr(condition, 'second_phase_bias', 2.0))
        if second_phase_bias != 2.0:
            tag += f'_pbias{second_phase_bias:g}'
    return tag


CONDITIONS = (
    Condition(
        key='control', label='B0', axis='control',
        description='Near-noise-free ceiling. Proves nothing upstream is broken. NOT zero '
                    'error: M20 is ill-conditioned there and campaign 1 lost 13.9 M rows to it',
        error_multiplier=0.1, n_contaminants=0,
        ),
    Condition(
        key='nominal', label='B1', axis='reference',
        description='The reference condition every other bundle is contrasted against',
        error_multiplier=1, n_contaminants=0,
        ),
    Condition(
        key='noisy', label='B2', axis='error_severity',
        description='Twice the nominal sigma(q2). Scales intercept and slope together',
        error_multiplier=2, n_contaminants=0,
        ),
    Condition(
        key='error_shape', label='B3', axis='error_shape',
        description='Nominal severity, four times the sigma(q2) y-intercept. Raises sigma at '
                    'low q2 relative to high q2, which a multiplier cannot do. DWMM asked for '
                    'this one tentatively ("maybe alter the y-intercept"), so it is provisional',
        error_multiplier=1, n_contaminants=0, intercept_scale=4.0, provisional=True,
        ),
    Condition(
        key='contaminated', label='B4', axis='contamination',
        description='Two independently placed contaminant lines',
        error_multiplier=1, n_contaminants=2,
        ),
    Condition(
        key='sparse2', label='B5', axis='sparsity',
        description='Two interior peaks dropped and backfilled from beyond the 20th. Nested: '
                    'these holes are a subset of sparse4 and sparse6',
        error_multiplier=1, n_contaminants=1, n_dropout=2,
        ),
    Condition(
        key='sparse4', label='B6', axis='sparsity',
        description='Four interior peaks dropped. Nested: a superset of sparse2, a subset of '
                    'sparse6',
        error_multiplier=1, n_contaminants=1, n_dropout=4,
        ),
    Condition(
        key='sparse6', label='B7', axis='sparsity',
        description='Six interior peaks dropped. Nested: a superset of sparse4',
        error_multiplier=1, n_contaminants=1, n_dropout=6,
        ),
    Condition(
        key='second_phase', label='B8', axis='second_phase',
        description='Three lines from a real partner cell drawn from the database. Correlated, '
                    'unlike independently placed contaminants, which is what makes them hard to '
                    'reject',
        error_multiplier=1, n_contaminants=0, second_phase_lines=3,
        ),
    )

BY_KEY = {condition.key: condition for condition in CONDITIONS}
BY_TAG = {condition.tag: condition for condition in CONDITIONS}
BUNDLE_LABELS = {condition.tag: condition.label for condition in CONDITIONS}

# The nesting is a property of the sparsity axis and the peak-list preparation implements it by
# drawing MAX_NESTED_DROPOUT holes once and taking prefixes. Ordered ascending, because that is
# what "prefix" means here.
SPARSITY_LADDER = tuple(sorted(
    (condition.n_dropout for condition in CONDITIONS if condition.axis == 'sparsity')))
MAX_NESTED_DROPOUT = max(SPARSITY_LADDER) if SPARSITY_LADDER else 0

# Kept for the loader's benefit, and empty by construction: campaign 2 generates no zero-error
# bundle, so the arithmetic degeneracy PROTOCOL §3 rule 11 guards against cannot arise here. The
# rule still applies to campaign 1's inherited pool, where `error0_cont0` is on disk.
CONTROL_BUNDLES = ()

# The severe half of the grid. **This is the condition half of the hard stratum only.** PLAN §6
# defines the hard stratum by lattice, frozen volume decile, condition *and reachability at the
# generation cut*, and S06 owns the reachability half -- campaign 1's hard stratum held 104
# entries in total, so every threshold metric on its reporting split was exactly zero (R3).
HARD_BUNDLES = tuple(
    condition.tag for condition in CONDITIONS
    if condition.axis in ('error_severity', 'contamination', 'second_phase')
    or (condition.axis == 'sparsity' and condition.n_dropout >= 4)
    )


# Which arm a bundle belongs to. S07 runs a wide `core` arm over every crystal in the frozen
# manifest and a narrower `mechanism` arm over the nested ~15 % subset, so entry counts are
# comparable WITHIN an arm and not across one: 17 591 against 2 636 is the design, not a shortfall.
# Defined here because the consolidator and the acceptance gate both need it and a second copy
# would drift -- the same reason `bundle_tag` is the only implementation of the tag rule.
MECHANISM_AXES = ('sparsity', 'error_shape')


def bundle_arm(tag):
    """`core`, `mechanism`, or `unknown` for a tag this campaign did not generate."""
    condition = BY_TAG.get(tag)
    if condition is None:
        return 'unknown'
    return 'mechanism' if condition.axis in MECHANISM_AXES else 'core'


def tags():
    return tuple(condition.tag for condition in CONDITIONS)


def condition_row(condition):
    """One flat record per bundle, for the condition table shipped as an artefact."""
    return {
        'key': condition.key,
        'label': condition.label,
        'tag': condition.tag,
        'axis': condition.axis,
        'error_law': ERROR_LAW,
        'error_multiplier': condition.error_multiplier,
        'intercept_scale': condition.intercept_scale,
        'n_contaminants': condition.n_contaminants,
        'contaminant_bias': condition.contaminant_bias,
        'n_dropout': condition.n_dropout,
        'second_phase_lines': condition.second_phase_lines,
        'second_phase_bias': condition.second_phase_bias,
        'is_hard': condition.tag in HARD_BUNDLES,
        'provisional': condition.provisional,
        'description': condition.description,
        }
