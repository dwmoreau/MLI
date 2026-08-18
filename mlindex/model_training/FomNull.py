"""S07: each figure of merit as a tail probability under the incorrect-candidate null.

M20 = 8 means one thing for a 200 A^3 cubic cell and another for a 3000 A^3 triclinic one, and
`run.py` sorts fourteen Bravais lattices on one raw scale regardless. S06 measured what that costs:
within a lattice every merit in the zoo scores 0.44-0.685 on top-10, across lattices they spread
0.000-0.618, so most of the leaderboard is **scale transfer** rather than fit quality (F-074). The
fix is standard -- estimate the distribution of the statistic under "this candidate is wrong",
conditional on what shifts it, and report where the observed value sits in it.

Everything here is a **monotone transform within a group**, so per-lattice ranking is unchanged by
construction and only the cross-lattice comparison moves. That is asserted in the tests and is the
cheapest way to catch a transform that has started reading something it should not.

The output is always `-log p` with p = P(FOM_null >= FOM_observed | covariates), oriented so larger
is better whatever the merit's own direction. It composes additively with the log-evidence terms
S08/S12/S13 own (PLAN section 3), and it is a tail probability under the null, **not** P(correct) --
the prior is a separate factor and conflating the two is the easiest way to produce a number that
looks calibrated and ranks badly (S07 handoff).

Five methods, cheapest and most defensible first:

  * `analytic`     -- de Wolff 1961's null in closed form, no fitting whatsoever. For
                      `null_tail_nll` the per-line terms are Exp(1) under his exponential-interval
                      assumption, so the sum over N lines is exactly Gamma(N, 1) and p is an
                      incomplete gamma. Completely immune to R1 (STATUS section 7) because nothing
                      is fitted to the censored negatives. Its cross-lattice performance is a
                      *measurement of whether de Wolff's assumption holds*, and a negative result
                      there is itself the finding -- see `regularity_ratio`.
  * `analytic_rho` -- the same, with one bounded parameter per group: de Wolff's own regularity
                      ratio g_bar/Delta in [1/2, 1] (his section (d)), which is where the
                      cross-lattice bias Wu 1988 measured as 1.82 (cubic) -> 1.00 (triclinic) comes
                      from. Fewest parameters, extrapolates outside the training range, and directly
                      comparable to a published table.
  * `rank`         -- the empirical quantile transform within Bravais lattice. No covariates.
  * `binned`       -- the same, conditional on (extinction group, volume decile, V/V_crit band),
                      with sparse bins pooled up the hierarchy and every pooling recorded.
  * `gbm`          -- quantile regression over the same covariates, distilled onto the `binned`
                      grid so what is serialised is still a lookup table.

**R1 bounds the last three and the rho of the second.** The pool is censored at M20 >= 5 by
`prune_below_m20`, lattice-dependently, so a null fitted on these negatives partly learns the prune
boundary (F-049, F-068). The censoring keeps the *better-fitting* wrong cells, so every fitted null
here is biased towards "wrong cells fit well" and the bias is largest where the truncation bites
hardest, which is the low-symmetry lattices. Two things follow and both are in the results document
rather than only here: the numbers from the fitted methods are provisional, and the defence for the
*deployed* score is narrower than it looks -- `run.py` prunes identically at inference, so the
calibration population is the deployment population, but that says nothing about the true
crystallographic null.

Serialisation is plain npz -- grid edges, quantile tables and a JSON metadata blob -- so a fitted
null is version-independent, loads in milliseconds and never executes anything. No pickle, and no
sklearn estimator is ever stored: the GBM is distilled to a grid before it is saved.
"""
import json

import numpy as np
import pandas as pd
from scipy import special

from mlindex.model_training import FomMetrics

# The quantile grid an empirical null is stored on. Uniform in the bulk and geometric in the upper
# tail, because the tail is the half that decides a ranking: a candidate whose merit sits at the
# 99.999th percentile of wrong cells is the one being asked about, and a uniform grid would resolve
# it to three digits. `_LEVELS` are P(X_null <= x), so the tail of interest is the top end.
_TAIL = np.logspace(-7, -3, 129)
_LEVELS = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1025), 1.0 - _TAIL, _TAIL]))

# Above this quantile of the null the score comes from an exponential fit to the excess rather than
# from the stored grid, so the transform stays strictly monotone where the ranking is decided.
_TAIL_THRESHOLD = 0.999

# A group needs this many null candidates before its own quantiles are trusted; below it the group
# pools up the hierarchy. 2000 puts the 1/1000 tail on ~2 observations, which is the coarsest that
# is worth storing at all.
MIN_GROUP_COUNT = 2000

# g_bar/Delta is bounded in [1/2, 1] by de Wolff's *regularity* argument -- exponential intervals at
# one end, equidistant at the other. Measured on this pool it is not in that interval and is not
# close to it: the mean per-line term runs 3.5 (cF) to 7.1 (aP) where the exponential null predicts
# 1.0 and the equidistant limit 1.5, so the observed discrepancies are two to three orders of
# magnitude smaller than Delta(Q). That is not a failure of the regularity assumption, it is the
# look-elsewhere effect de Wolff identified in his own section 5 and nobody has quantified since:
# these candidates are not arbitrary wrong cells, they are least-squares *refined* wrong cells that
# then survived a prune and a deduplication, so the relevant null is the distribution of the best of
# many trials rather than of one draw.
#
# The range is therefore opened far below de Wolff's interval so the departure is measured rather
# than clamped into invisibility, and rho stops meaning "g_bar/Delta" and starts meaning "the
# effective per-line discrepancy scale, in units of Delta, after searching". The bounded quantity
# is still reported, and that it sits outside its own bounds is the result.
RHO_BOUNDS = (1e-5, 5.0)
_RHO_GRID = np.geomspace(RHO_BOUNDS[0], RHO_BOUNDS[1], 601)

METHODS = ('analytic', 'analytic_rho', 'z', 'rank', 'binned', 'gbm')

# Merits whose null has a closed form under de Wolff 1961. `scale` converts the stored merit into
# a sum of per-line terms that are Exp(1) under the null; `exact` records whether the identity is
# exact or an approximation whose direction is known.
#
# `nll_exponential` is deliberately absent even though its null is also Gamma: it is
# sum_i (dQ_i/Delta_i + log Delta_i), and the second sum is a per-candidate constant that is not
# stored anywhere and cannot be recovered from `delta_dewolff61`, which is a row *mean* over peaks.
NULL_LAWS = {
    'null_tail_nll': dict(
        scale=1.0, exact=True,
        note='-sum log[1 - exp(-dQ/Delta)]; each term is Exp(1) under the exponential null',
        ),
    'M_info_clipped': dict(
        scale=float(np.log(2.0)), exact=False,
        note=('Taupin eq. 25 in bits. Unclipped it is null_tail_nll/log(2) exactly; clipping only '
              'ever *reduces* the argument, so the analytic p is conservative'),
        ),
    }


# ---------------------------------------------------------------------------------------
# The analytic null
# ---------------------------------------------------------------------------------------
def log_gamma_sf(shape, x):
    """log P(Gamma(shape, 1) >= x), stable where `gammaincc` has underflowed to zero.

    This is not a nicety. `null_tail_nll` runs to ~1400 with a 99th percentile near 570 (F-064)
    against a shape of 10 or 20, and `gammaincc(20, 570)` is 0.0 in float64 -- so the whole upper
    tail, which is the only part that decides a ranking, would collapse to a single `inf`.

    Two regimes. Below the mean the survival function is O(1) and scipy is exact. Above it, the
    asymptotic series log Q(a, x) = -x + (a-1) log x - lgamma(a) + log(1 + (a-1)/x + ...) is used;
    it is an expansion in a/x and the terms are added while they shrink, which for x > 2a is
    accurate to well past float64's resolution of the result.
    """
    shape = np.asarray(shape, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    shape, x = np.broadcast_arrays(shape, x)
    out = np.full(x.shape, -np.inf)

    small = x <= np.maximum(shape, 1.0)*2.0
    with np.errstate(divide='ignore'):
        out[small] = np.log(special.gammaincc(shape[small], np.maximum(x[small], 0.0)))

    large = ~small & np.isfinite(x)
    if np.any(large):
        a = shape[large]
        z = x[large]
        series = np.ones(z.shape)
        term = np.ones(z.shape)
        for step in range(1, 64):
            term = term*(a - step)/z
            if not np.any(np.abs(term) > 1e-16*np.abs(series)):
                break
            series = series + term
        out[large] = (-z + (a - 1.0)*np.log(z) - special.gammaln(a)
                      + np.log(np.maximum(series, 1e-300)))
    return out


def analytic_neg_log_p(merit, values, n_peaks):
    """`-log p` for a merit whose null is Gamma(N, 1), with no fitting of any kind.

    Under de Wolff's exponential-interval assumption each per-line term of `null_tail_nll` is
    Exp(1), so their sum over the entry's N observed lines is Gamma(N, 1) exactly. p is then the
    survival function and nothing about the candidate's lattice or volume enters, which is the
    sense in which this construction is lattice-independent *by definition of the null*.

    Whether it is lattice-independent *in fact* is precisely what S07 measures: if real calculated-Q
    sequences are more regular than exponential -- de Wolff says they are, and more so at high
    symmetry -- then the true per-line mean is g_bar = rho*Delta with rho < 1 and this p is
    optimistic by a symmetry-dependent amount. `regularity_ratio` measures rho and `analytic_rho`
    corrects for it.
    """
    law = NULL_LAWS[merit]
    statistic = np.asarray(values, dtype=np.float64)*law['scale']
    shape = np.asarray(n_peaks, dtype=np.float64)
    return -log_gamma_sf(shape, np.maximum(statistic, 0.0))


def _line_term_moments(rho):
    """Mean and variance of one per-line term when dQ/Delta has mean `rho` rather than 1.

    The term is t = -log(1 - exp(-u)) with u ~ Exponential(mean rho). At rho = 1, u is Exp(1),
    1 - exp(-u) is uniform and t is Exp(1), so the moments must come back (1, 1) -- which is what
    the test asserts. Smaller rho means the discrepancies are systematically smaller than an
    exponential null predicts, i.e. the wrong cells fit better than chance, which is exactly de
    Wolff's regularity effect and pushes the mean term up.

    Quadrature rather than a closed form: the integral has no elementary antiderivative and this is
    evaluated a few hundred times at fit time and never at inference.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    # Substitute u = -rho*log(v) so v is uniform on (0, 1) and the exponential weight disappears;
    # the integrand is then smooth except at the endpoints, which Gauss-Legendre handles.
    nodes, weights = np.polynomial.legendre.leggauss(4096)
    v = 0.5*(nodes + 1.0)
    weights = 0.5*weights
    u = -rho[:, np.newaxis]*np.log(np.clip(v, 1e-300, 1.0))[np.newaxis]
    term = -np.log1p(-np.exp(-u) + 1e-300)
    mean = np.sum(term*weights, axis=1)
    second = np.sum(term**2*weights, axis=1)
    return mean, np.maximum(second - mean**2, 1e-12)


_RHO_MEAN, _RHO_VARIANCE = _line_term_moments(_RHO_GRID)


def rho_from_mean_term(mean_term):
    """Invert the per-line mean back to de Wolff's regularity ratio g_bar/Delta.

    The mean term is monotone decreasing in rho, so a scalar mean over a group's null candidates
    identifies it. Values outside `RHO_BOUNDS` are clamped and the caller counts the clamps.
    """
    mean_term = np.asarray(mean_term, dtype=np.float64)
    order = np.argsort(_RHO_MEAN)
    return np.clip(np.interp(mean_term, _RHO_MEAN[order], _RHO_GRID[order]), *RHO_BOUNDS)


def _rho_gamma_parameters(rho, n_peaks):
    """Moment-matched Gamma for the sum of N per-line terms at regularity ratio `rho`.

    The exact distribution of the sum has no closed form once rho != 1, but it is a sum of N i.i.d.
    positive terms, so a two-moment Gamma match is accurate in the body and in the direction that
    matters in the tail. N is 20 (10 for the cubic models), which is comfortably enough for the
    match to be the right shape rather than a convenience.
    """
    rho = np.asarray(rho, dtype=np.float64)
    n_peaks = np.asarray(n_peaks, dtype=np.float64)
    order = np.argsort(_RHO_GRID)
    mean = np.interp(rho, _RHO_GRID[order], _RHO_MEAN[order])
    variance = np.interp(rho, _RHO_GRID[order], _RHO_VARIANCE[order])
    shape = n_peaks*mean**2/variance
    scale = variance/mean
    return shape, scale


# ---------------------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------------------
# The conditioning hierarchy, coarsest last. Pooling drops covariates from the *right*, so a thin
# (lattice, extinction group, volume, V/V_crit) cell falls back to (lattice, extinction group) and
# ultimately to the lattice alone. `n_peaks` is deliberately not in the list: it is 10 for the cubic
# models and 20 for everything else, so it carries no information beyond `bravais_lattice`.
COVARIATE_LEVELS = (
    ('bravais_lattice', 'extinction_group', 'volume_bin', 'vcrit_bin'),
    ('bravais_lattice', 'extinction_group', 'volume_bin'),
    ('bravais_lattice', 'extinction_group'),
    ('bravais_lattice',),
    (),
    )

# What each method conditions on at its finest level.
METHOD_COVARIATES = {
    'z': ('bravais_lattice',),
    'rank': ('bravais_lattice',),
    'binned': COVARIATE_LEVELS[0],
    'gbm': COVARIATE_LEVELS[0],
    'analytic_rho': ('bravais_lattice', 'extinction_group'),
    'analytic': (),
    }

_BINNED_COLUMNS = {'volume_bin': 'volume', 'vcrit_bin': 'V_over_Vcrit',
                   'extinction_group': 'spacegroup'}


def required_columns(merit, method, trials=None):
    """Candidate columns `apply` will read, for `FomMetrics.evaluate(score_columns=...)`."""
    columns = {merit}
    if method in ('analytic', 'analytic_rho'):
        columns.add('n_peaks')
    for name in METHOD_COVARIATES[method]:
        columns.add(_BINNED_COLUMNS.get(name, name))
    if trials:
        columns.add(trials)
    return tuple(sorted(columns))


def look_elsewhere(neg_log_p, n_trials):
    """de Wolff 1961 section 5, as arithmetic: the null of the *best* of many cells, not of one.

    His own objection to his own calculation is that the probability of one arbitrary cell fitting
    a pattern this well was 0.01, but *"there are hundreds of choices possible for the indices of
    the three parameter-determining lines... Hence the existence of a false unit cell yielding an
    agreement at least similar is fairly certain."* We generate thousands of candidates per entry
    and report the best one, so the relevant null is the maximum's:

        P(max of m draws >= x) = 1 - (1 - p)^m

    which for small p is m*p, i.e. `-log p - log m`. That subtraction is the whole correction, and
    it is not a small one here: at C1 the triclinic generators enter deduplication with hundreds of
    times more candidates than the cubic ones, so a triclinic cell has to be far more improbable
    under its own null to be equally surprising. This is the term that makes a threshold mean the
    same thing in two lattices, and nobody appears to have applied it since he pointed it out.

    `n_trials` is the number of candidates that entered deduplication for that (entry, lattice) --
    `n_entering` in the dump -- which is known at inference and is not a property of the answer.
    """
    neg_log_p = np.asarray(neg_log_p, dtype=np.float64)
    trials = np.broadcast_to(
        np.maximum(np.asarray(n_trials, dtype=np.float64), 1.0), neg_log_p.shape)
    corrected = neg_log_p - np.log(trials)
    # Where the corrected value is small the linear approximation is not good enough: p is no
    # longer negligible and 1 - (1 - p)^m saturates at 1 rather than continuing to grow.
    near_one = corrected < 3.0
    if np.any(near_one):
        p = np.exp(-np.clip(neg_log_p[near_one], 0.0, 700.0))
        survival = -np.expm1(trials[near_one]*np.log1p(-np.clip(p, 0.0, 1.0 - 1e-16)))
        corrected[near_one] = -np.log(np.clip(survival, 1e-300, 1.0))
    return corrected


def _bin_edges(values, n_bins):
    """Quantile edges over the training null, with duplicates collapsed.

    Collapsing matters: `V_over_Vcrit` is heavily tied inside a lattice, and equal edges would put
    every candidate in one bin while claiming four.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _digitize(values, edges):
    return np.clip(np.searchsorted(edges, np.asarray(values, dtype=np.float64), side='right') - 1,
                   0, max(len(edges) - 2, 0))


class _Binner:
    """Turns a candidate frame into the group labels a fitted null is keyed on.

    The edges are learned once on `fom-train` and stored with the null, so `fom-dev` and deployment
    are binned by the training distribution rather than by their own -- otherwise the transform
    would depend on which candidates happen to be scored together.
    """

    def __init__(self, volume_edges=None, vcrit_edges=None):
        self.volume_edges = {} if volume_edges is None else volume_edges
        self.vcrit_edges = {} if vcrit_edges is None else vcrit_edges

    def fit(self, frame, n_volume_bins=10, n_vcrit_bins=4):
        lattices = frame['bravais_lattice'].to_numpy()
        for lattice in np.unique(lattices):
            rows = lattices == lattice
            if 'volume' in frame.columns:
                self.volume_edges[str(lattice)] = _bin_edges(
                    np.log(np.maximum(frame['volume'].to_numpy()[rows], 1e-300)), n_volume_bins,
                    )
            if 'V_over_Vcrit' in frame.columns:
                self.vcrit_edges[str(lattice)] = _bin_edges(
                    frame['V_over_Vcrit'].to_numpy()[rows], n_vcrit_bins,
                    )
        return self

    def labels(self, frame, covariates):
        """One string label per row for the requested covariate tuple.

        Concatenated through `Series.str.cat` rather than a comprehension: this runs over seven
        million rows, once per hierarchy level, once per evaluation, and a Python-level join here
        costs more than every quantile in the module put together.
        """
        if not covariates:
            return np.full(frame.shape[0], 'all', dtype=object)
        lattices = frame['bravais_lattice'].astype(str).to_numpy()
        parts = []
        for name in covariates:
            if name == 'bravais_lattice':
                parts.append(lattices)
            elif name == 'extinction_group':
                parts.append(frame['spacegroup'].astype(str).to_numpy())
            elif name == 'volume_bin':
                parts.append(self._per_lattice_bin(
                    np.log(np.maximum(frame['volume'].to_numpy(dtype=np.float64), 1e-300)),
                    lattices, self.volume_edges))
            elif name == 'vcrit_bin':
                parts.append(self._per_lattice_bin(
                    frame['V_over_Vcrit'].to_numpy(dtype=np.float64), lattices, self.vcrit_edges))
            else:
                parts.append(frame[name].astype(str).to_numpy())
        if len(parts) == 1:
            return parts[0].astype(object)
        head = pd.Series(parts[0], dtype='string')
        rest = [pd.Series(part, dtype='string') for part in parts[1:]]
        return head.str.cat(rest, sep='|').to_numpy(dtype=object)

    @staticmethod
    def _per_lattice_bin(values, lattices, edges_by_lattice):
        out = np.zeros(values.shape[0], dtype=np.int64)
        for lattice in np.unique(lattices):
            rows = lattices == lattice
            edges = edges_by_lattice.get(str(lattice))
            out[rows] = 0 if edges is None else _digitize(values[rows], edges)
        return out.astype(str)


# ---------------------------------------------------------------------------------------
# The fitted null
# ---------------------------------------------------------------------------------------
class FomNull:
    """One merit's null, fitted by one method, applied as `-log p`.

    Use it as a score directly:

        null = FomNull.load(path)
        result = FomMetrics.evaluate(shards, score=null.apply, higher_is_better=True,
                                     score_columns=null.required_columns)

    `apply` always returns higher-is-better, whatever the merit's own direction, because a tail
    probability has only one direction.
    """

    def __init__(self, merit, method, higher_is_better=True, tables=None, counts=None,
                 binner=None, pooled_from=None, rho=None, tail=None, meta=None, trials=None):
        self.merit = merit
        self.method = method
        self.trials = trials
        self.higher_is_better = bool(higher_is_better)
        self.tables = {} if tables is None else tables
        self.counts = {} if counts is None else counts
        self.binner = _Binner() if binner is None else binner
        self.pooled_from = {} if pooled_from is None else pooled_from
        self.rho = {} if rho is None else rho
        self.tail = {} if tail is None else tail
        self.meta = {} if meta is None else meta
        self.diagnostics = dict(n_applied=0, n_clamped_low=0, n_tail_extrapolated=0,
                                n_group_fallback=0, n_unseen_group=0)

    # -------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, frames, merit, method, higher_is_better=True, min_count=MIN_GROUP_COUNT,
            gbm_sample=300000, seed=12345, trials=None):
        """Fit on the incorrect-candidate null.

        `frames` is an iterable of candidate shards -- `run_fom_zoo_eval.bundle_frames` restricted
        to `fom-train`. The null sample is every candidate that is not correct, not off by two and
        not degenerate: the actual competitors the algorithm produces, which is the right null and
        is *not* the same as a random wrong cell, because it has already survived the search.
        """
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}, got {method!r}')
        if trials and method == 'z':
            raise ValueError(
                'The look-elsewhere correction turns P(one draw >= x) into P(best of m >= x), so '
                'it is only defined on a tail probability. `z` returns a standardised merit, which '
                'can be negative and is not one.'
                )
        covariates = METHOD_COVARIATES[method]
        # The trial count is read at apply time only: the null is the distribution of a *single*
        # wrong cell's merit, and the look-elsewhere factor turns that into the distribution of the
        # best of many afterwards. Fitting on the maximum instead would double-count the search.
        needed = set(required_columns(merit, method)) | {
            'is_correct', 'is_off_by_two', 'is_degenerate'}

        collected = []
        for frame in frames:
            missing = [column for column in needed if column not in frame.columns]
            if missing:
                raise ValueError(f'Candidate frame is missing {sorted(missing)} for '
                                 f'{merit!r}/{method!r}')
            keep = (~FomMetrics.as_bool(frame['is_correct'])
                    & ~FomMetrics.as_bool(frame['is_off_by_two'])
                    & ~FomMetrics.as_bool(frame['is_degenerate']))
            columns = sorted(needed - {'is_correct', 'is_off_by_two', 'is_degenerate'})
            collected.append(frame.loc[keep, columns].reset_index(drop=True))
        if not collected:
            raise ValueError('No candidate shards to fit on')
        null = pd.concat(collected, ignore_index=True)
        del collected

        # One orientation for the whole module: bigger is a better fit, so the tail of interest is
        # always the upper one.
        values = null[merit].to_numpy(dtype=np.float64)
        if not higher_is_better:
            values = -values
        finite = np.isfinite(values)

        binner = _Binner()
        if any(name in ('volume_bin', 'vcrit_bin') for name in covariates):
            binner.fit(null.loc[finite])

        fitted = cls(merit=merit, method=method, higher_is_better=higher_is_better, binner=binner,
                     trials=trials)
        fitted.meta = dict(
            n_null=int(finite.sum()), n_non_finite=int((~finite).sum()),
            covariates=list(covariates), min_count=int(min_count), seed=int(seed),
            levels=int(_LEVELS.size), r1_bounded=method != 'analytic', trials=trials,
            )
        if method == 'analytic':
            if merit not in NULL_LAWS:
                raise ValueError(f'No closed-form null is known for {merit!r}; '
                                 f'available: {sorted(NULL_LAWS)}')
            fitted.meta['law'] = NULL_LAWS[merit]['note']
            fitted.meta['law_exact'] = bool(NULL_LAWS[merit]['exact'])
            return fitted
        if method == 'analytic_rho':
            fitted._fit_rho(null.loc[finite], values[finite], min_count)
            return fitted
        fitted._fit_tables(null.loc[finite], values[finite], covariates, min_count)
        if method == 'gbm':
            fitted._distil_gbm(null.loc[finite], values[finite], covariates, gbm_sample, seed)
        return fitted

    def _fit_rho(self, null, values, min_count):
        """One regularity ratio per (lattice, extinction group), pooled where thin.

        rho is identified by the mean per-line term, which is the merit divided by the line count,
        so this is a single scalar per group with a known range -- the smallest fitted object in the
        ladder and the one that extrapolates.
        """
        if self.merit not in NULL_LAWS:
            raise ValueError(f'analytic_rho needs a closed-form null; {self.merit!r} has none')
        scale = NULL_LAWS[self.merit]['scale']
        n_peaks = null['n_peaks'].to_numpy(dtype=np.float64)
        mean_term = values*scale/np.maximum(n_peaks, 1.0)
        self._fit_by_level(null, mean_term, METHOD_COVARIATES['analytic_rho'], min_count,
                           self._store_rho)

    def _store_rho(self, label, sample):
        self.rho[label] = float(rho_from_mean_term(np.mean(sample)))
        self.counts[label] = int(sample.size)

    def _fit_tables(self, null, values, covariates, min_count):
        store = self._store_z if self.method == 'z' else self._store_table
        self._fit_by_level(null, values, covariates, min_count, store)

    def _store_z(self, label, sample):
        """Location and scale of the group's null, robustly -- the gentlest standardisation there is.

        The handoff asks for both a z-score and a tail probability, and the pair is the experiment:
        the z-score removes only the null's location and scale and leaves the merit's own shape
        intact, where the quantile transform replaces the shape entirely. If a merit's
        cross-lattice problem is that its scale differs between lattices, the two do the same
        thing; if the quantile transform is losing something the raw scale was carrying, only the
        z-score keeps it. Median and a symmetrised 16-84 spread rather than mean and standard
        deviation, because these distributions have tails that a second moment does not survive.
        """
        median = float(np.median(sample))
        spread = float(np.quantile(sample, 0.84) - np.quantile(sample, 0.16))/2.0
        if not np.isfinite(spread) or spread <= 0:
            spread = float(np.std(sample)) or 1.0
        self.tables[label] = np.array([median, spread], dtype=np.float64)
        self.counts[label] = int(sample.size)

    def _store_table(self, label, sample):
        """The group's quantile grid, plus an exponential fit to the excess over its 99.9th.

        The tail fit is not an embellishment. Everything a ranking cares about sits *above* the
        null's bulk -- a candidate worth reporting fits better than essentially every wrong cell --
        and a stored grid alone quantises exactly there, collapsing the top of the pool into ties
        and scrambling an ordering that was correct before calibration. Peaks-over-threshold on the
        excess restores a strictly monotone score above the threshold and extrapolates past the
        null's observed maximum instead of clamping every candidate beyond it to one value.
        """
        table = np.quantile(sample, _LEVELS).astype(np.float64)
        self.tables[label] = table
        self.counts[label] = int(sample.size)
        threshold = float(np.quantile(sample, _TAIL_THRESHOLD))
        excess = sample[sample > threshold] - threshold
        # Mean excess is the exponential MLE. A degenerate group (every value tied at the
        # threshold) falls back to the grid's own width so the score stays finite and increasing.
        scale = float(np.mean(excess)) if excess.size >= 8 else 0.0
        if not np.isfinite(scale) or scale <= 0:
            scale = max(float(table[-1] - table[0]), 1e-12)/100.0
        self.tail[label] = (threshold, scale)

    def _fit_by_level(self, null, statistic, covariates, min_count, store):
        """Fit the finest group that has the counts, and record every fallback.

        Pooling walks the hierarchy from `covariates` down to the empty tuple, so a cP cell with 40
        null candidates ends up on its lattice's table rather than on 40 points' worth of noise.
        Which groups were pooled and from where is stored, because "the quantiles for oF are really
        the cubic-system ones" is exactly the kind of thing that is invisible later otherwise.

        The label at a coarser level is constant within a finer group -- the hierarchy drops
        covariates from the right -- so the mapping is read off one row per finest group rather
        than by rescanning the null, which matters at six million rows.
        """
        levels = [level for level in COVARIATE_LEVELS if len(level) <= len(covariates)]
        if tuple(covariates) not in levels:
            levels = [tuple(covariates)] + levels
        labelled = pd.DataFrame(
            {f'level_{depth}': self.binner.labels(null, level)
             for depth, level in enumerate(levels)},
            )
        columns = list(labelled.columns)
        counts = {column: labelled[column].value_counts().to_dict() for column in columns}
        mapping = labelled.drop_duplicates(subset=columns[0])

        chosen = {}
        for row in mapping.itertuples(index=False):
            label = getattr(row, columns[0])
            for depth, column in enumerate(columns):
                source = getattr(row, column)
                n = int(counts[column].get(source, 0))
                if n >= min_count or depth == len(columns) - 1:
                    chosen[str(label)] = str(source)
                    self.pooled_from[str(label)] = dict(
                        level=len(levels[depth]), source=str(source), n=n,
                        )
                    break

        # One pass per distinct source, rather than one per finest label: many finest labels share
        # a source once pooling has fired.
        resolved = labelled[columns[0]].astype(str).map(chosen).to_numpy()
        for source in pd.unique(resolved):
            store(str(source), statistic[resolved == source])

        # The two coarsest levels are stored whether or not any training cell pooled that far,
        # because they are where an *unseen* cell lands. A spacegroup `fom-train` never produced
        # has no finest label to fall back from, and without a table on its lattice -- or in the
        # last resort on the whole pool -- it would score zero and rank below every candidate that
        # was merely bad. Fifteen extra tables against a silent, invisible failure.
        for column in columns[-2:]:
            for source in pd.unique(labelled[column]):
                if str(source) not in self.tables and str(source) not in self.rho:
                    store(str(source), statistic[labelled[column].to_numpy() == source])

    def _distil_gbm(self, null, values, covariates, gbm_sample, seed):
        """Quantile regression, then collapse it onto the binned grid.

        The GBM is only ever a way of *smoothing* the conditional quantiles across sparse cells; it
        is never what gets deployed. Distilling it onto the same group labels the binned method uses
        keeps the serialised artefact a lookup table, keeps inference one digitize plus one gather,
        and means the two methods differ in how the numbers were estimated rather than in what is
        stored.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        rng = np.random.default_rng(seed)
        if values.size > gbm_sample:
            take = rng.choice(values.size, size=gbm_sample, replace=False)
        else:
            take = np.arange(values.size)
        design, columns = self._design_matrix(null.iloc[take])
        target = values[take]

        # A coarse quantile grid: fifteen models per merit is what the fit costs, and the grid is
        # interpolated back onto `_LEVELS` afterwards, so nothing downstream sees the difference.
        grid = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                         0.99, 0.999])
        # Keyed on the *resolved* label, the one `_apply_tables` will look up: a finest cell that
        # pooled during `_fit_tables` must not acquire a private GBM table here, or the two halves
        # of the method would disagree about which group a candidate belongs to.
        sources = {label: entry['source'] for label, entry in self.pooled_from.items()}
        raw_labels = pd.Series(self.binner.labels(null, covariates), dtype=object)
        labels = raw_labels.map(sources).fillna(raw_labels).to_numpy(dtype=object)
        centres = pd.DataFrame(self._design_matrix(null, columns=columns)[0], columns=columns)
        centres['__label'] = labels
        centres = centres.groupby('__label', sort=False).mean()

        predictions = np.zeros((centres.shape[0], grid.size))
        for position, level in enumerate(grid):
            model = HistGradientBoostingRegressor(
                loss='quantile', quantile=float(level), max_iter=200, random_state=seed,
                )
            model.fit(design, target)
            predictions[:, position] = model.predict(centres[columns].to_numpy())
        # Quantile crossing is normal when each level is fitted independently; sorting is the
        # standard repair and is monotone, so it cannot invert a ranking.
        predictions = np.sort(predictions, axis=1)

        # The body of each group's curve is replaced; the upper-tail fit from `_fit_tables` is kept,
        # because a GBM trained on fifteen quantile levels has nothing to say above the 99.9th and
        # the excess distribution is what decides the ranking.
        for row, label in enumerate(centres.index):
            self.tables[str(label)] = np.interp(_LEVELS, grid, predictions[row])
        self.meta['gbm_sample'] = int(target.size)
        self.meta['gbm_levels'] = grid.tolist()

    def _design_matrix(self, frame, columns=None):
        """Numeric covariates for the GBM: log volume, V/V_crit, and the lattice as an integer.

        `n_peaks` is not among them and is not an omission: it is 10 for the cubic models and 20
        for every other lattice, so `lattice_code` already carries it exactly.
        """
        columns = columns or ['log_volume', 'V_over_Vcrit', 'lattice_code']
        lattice = pd.Series(frame['bravais_lattice'].astype(str).to_numpy()).map(
            {name: code for code, name in enumerate(FomMetrics.BRAVAIS_LATTICES)}).fillna(-1)
        built = {
            'log_volume': np.log(np.maximum(frame['volume'].to_numpy(dtype=np.float64), 1e-300)),
            'V_over_Vcrit': frame['V_over_Vcrit'].to_numpy(dtype=np.float64),
            'lattice_code': lattice.to_numpy(dtype=np.float64),
            }
        return np.column_stack([built[name] for name in columns]), columns

    # -------------------------------------------------------------- applying
    @property
    def required_columns(self):
        return required_columns(self.merit, self.method, self.trials)

    @property
    def label(self):
        return f'{self.method}+trials' if self.trials else self.method

    def apply(self, frame):
        """`-log p` for every row, higher being better. Signature is `FomMetrics.evaluate`'s."""
        values = frame[self.merit].to_numpy(dtype=np.float64)
        if not self.higher_is_better:
            values = -values
        if self.method == 'analytic':
            scores = analytic_neg_log_p(self.merit, values, frame['n_peaks'].to_numpy())
        elif self.method == 'analytic_rho':
            scores = self._apply_rho(frame, values)
        else:
            scores = self._apply_tables(frame, values)
        if self.trials:
            scores = look_elsewhere(scores, frame[self.trials].to_numpy())
        self.diagnostics['n_applied'] += int(values.size)
        # A non-finite merit cannot be calibrated; it keeps numpy's ordering (NaN last) rather than
        # being silently mapped onto a finite score.
        scores = np.where(np.isfinite(values), scores, np.nan)
        return scores

    def _apply_rho(self, frame, values):
        scale = NULL_LAWS[self.merit]['scale']
        labels = self._resolve_frame(frame, METHOD_COVARIATES['analytic_rho'])
        rho = np.array([self.rho.get(label, 1.0) for label in labels])
        shape, gamma_scale = _rho_gamma_parameters(rho, frame['n_peaks'].to_numpy())
        statistic = np.maximum(values*scale, 0.0)
        return -log_gamma_sf(shape, statistic/gamma_scale)

    def _apply_tables(self, frame, values):
        resolved = self._resolve_frame(frame, METHOD_COVARIATES[self.method])
        scores = np.zeros(values.shape[0])
        # One factorize and one groupby rather than a boolean mask per group: `binned` has
        # hundreds of groups over millions of rows, and a mask each would be a quadratic scan.
        codes, uniques = pd.factorize(resolved)
        positions = pd.Series(np.arange(values.size)).groupby(codes).indices
        for code, label in enumerate(uniques):
            rows = positions.get(code)
            if rows is None:
                continue
            table = self.tables.get(str(label))
            if table is None:
                # Only reachable when even the empty-covariate group is missing, which means the
                # null was never fitted. Scored zero and counted rather than silently ranked.
                self.diagnostics['n_unseen_group'] += int(rows.size)
                continue
            if self.method == 'z':
                scores[rows] = (values[rows] - table[0])/table[1]
            else:
                scores[rows] = self._score_from_table(str(label), table, values[rows])
        return scores

    def _score_from_table(self, label, table, values):
        """`-log p` from one group's grid, interpolated below the tail and fitted above it."""
        threshold, scale = self.tail.get(label, (table[-1], 1.0))
        out = np.empty(values.shape[0])

        upper = values > threshold
        # Continuous with the grid at the threshold by construction: -log(1 - _TAIL_THRESHOLD)
        # plus the exponential excess.
        out[upper] = -np.log1p(-_TAIL_THRESHOLD) + (values[upper] - threshold)/scale
        self.diagnostics['n_tail_extrapolated'] += int(np.sum(values > table[-1]))

        body = ~upper
        if np.any(body):
            level = np.interp(values[body], table, _LEVELS)
            # Below the null's own minimum every candidate is equally unremarkable under the null,
            # so p is 1 and the score is 0. The raw ordering there is continued linearly on the same
            # scale purely so the transform stays strictly monotone and cannot invent ties; the
            # values are negative, which is honest -- they are below the range where "how improbable
            # is this under the null" has any resolution at all.
            below = values[body] < table[0]
            score = -np.log(np.clip(1.0 - level, 1e-300, 1.0))
            score[below] = (values[body][below] - table[0])/max(scale, 1e-300)
            out[body] = score
            self.diagnostics['n_clamped_low'] += int(np.sum(below))
        return out

    def _resolve_frame(self, frame, covariates):
        """The label whose table each row should use, falling back down the hierarchy.

        Two distinct fallbacks meet here. At fit time a thin cell was pooled into a coarser one and
        `pooled_from` records where; at apply time a cell can be *unseen* -- a spacegroup or a
        V/V_crit band that `fom-train` never produced -- and has no entry at all. Both are answered
        the same way: take the finest level that has a table.
        """
        levels = [level for level in COVARIATE_LEVELS if len(level) <= len(covariates)]
        if tuple(covariates) not in levels:
            levels = [tuple(covariates)] + levels
        # `map` against plain dicts and `isin` against a set, rather than comprehensions: this is
        # called once per hierarchy level on every evaluation, over millions of rows.
        sources = {label: entry['source'] for label, entry in self.pooled_from.items()}
        fitted = set(self.tables) | set(self.rho)

        def resolve(labels):
            series = pd.Series(labels, dtype=object)
            mapped = series.map(sources).fillna(series)
            # `to_numpy` can hand back a read-only view of the Series' block; the fallback loop
            # writes into both of these.
            return (mapped.to_numpy(dtype=object, copy=True),
                    mapped.isin(fitted).to_numpy(copy=True))

        resolved, known = resolve(self.binner.labels(frame, levels[0]))
        for level in levels[1:]:
            if np.all(known):
                break
            missing = np.flatnonzero(~known)
            candidate, found = resolve(self.binner.labels(frame, level)[missing])
            self.diagnostics['n_group_fallback'] += int(np.sum(found))
            resolved[missing] = candidate
            known[missing] = found
        return resolved

    def _resolve(self, label):
        entry = self.pooled_from.get(str(label))
        return entry['source'] if entry else str(label)

    # -------------------------------------------------------------- persistence
    def save(self, path):
        """npz: grid, tables, counts and a JSON metadata blob. Never pickle.

        A fitted null outlives the sklearn and pandas versions that produced it, gets read by an
        inference-only environment, and is small enough that there is no reason for it to be
        anything but arrays.
        """
        labels = sorted(set(self.tables) | set(self.rho))
        payload = dict(
            levels=_LEVELS,
            labels=np.array(labels, dtype=object),
            counts=np.array([self.counts.get(label, 0) for label in labels], dtype=np.int64),
            rho=np.array([self.rho.get(label, np.nan) for label in labels], dtype=np.float64),
            tail=np.array([self.tail.get(label, (np.nan, np.nan)) for label in labels],
                          dtype=np.float64).reshape(-1, 2),
            metadata=np.array(json.dumps(dict(
                merit=self.merit, method=self.method, trials=self.trials,
                higher_is_better=self.higher_is_better,
                pooled_from=self.pooled_from, meta=self.meta,
                volume_edges={key: value.tolist()
                              for key, value in self.binner.volume_edges.items()},
                vcrit_edges={key: value.tolist()
                             for key, value in self.binner.vcrit_edges.items()},
                )), dtype=object),
            )
        if self.tables:
            # `z` stores two numbers per group and the quantile methods store the whole grid, so
            # rows are padded to a common width and trimmed again on load. Trailing NaN is the
            # padding marker, which is unambiguous because a fitted row never contains one.
            width = max(len(table) for table in self.tables.values())
            payload['tables'] = np.vstack([
                np.concatenate([table, np.full(width - len(table), np.nan)])
                if (table := self.tables.get(label)) is not None
                else np.full(width, np.nan)
                for label in labels])
        np.savez_compressed(str(path), **payload)
        return path

    @classmethod
    def load(cls, path):
        with np.load(str(path), allow_pickle=True) as data:
            metadata = json.loads(str(data['metadata'].item()))
            labels = [str(label) for label in data['labels']]
            counts = dict(zip(labels, data['counts'].tolist()))
            rho_values = data['rho']
            rho = {label: float(value) for label, value in zip(labels, rho_values)
                   if np.isfinite(value)}
            tail_values = data['tail']
            tail = {label: (float(row[0]), float(row[1]))
                    for label, row in zip(labels, tail_values) if np.all(np.isfinite(row))}
            tables = {}
            if 'tables' in data.files:
                stacked = data['tables']
                for row, label in enumerate(labels):
                    finite = np.isfinite(stacked[row])
                    if finite[0]:
                        tables[label] = stacked[row][:int(np.argmin(finite))
                                                     if not finite.all() else len(finite)]
        binner = _Binner(
            volume_edges={key: np.asarray(value, dtype=np.float64)
                          for key, value in metadata['volume_edges'].items()},
            vcrit_edges={key: np.asarray(value, dtype=np.float64)
                         for key, value in metadata['vcrit_edges'].items()},
            )
        return cls(merit=metadata['merit'], method=metadata['method'],
                   trials=metadata.get('trials'),
                   higher_is_better=metadata['higher_is_better'], tables=tables, counts=counts,
                   binner=binner, pooled_from=metadata['pooled_from'], rho=rho, tail=tail,
                   meta=metadata['meta'])


# ---------------------------------------------------------------------------------------
# Diagnostics S07 reports
# ---------------------------------------------------------------------------------------
def scaled_features(frame, nulls, prefix=''):
    """Add one comparably-scaled column per fitted null, for a joint predictor to combine.

    **This, and not a ranking, is what S07 hands to S08** (DWMM, 2026-08-17). Calibrating a merit
    to a tail probability and then *ranking on it* is measurably worse than the raw merit, because
    the raw scale was carrying the prior as well as the null and this removes both. But putting
    twenty-one merits onto one scale is a precondition for combining them at all: F-074 measured
    that they all discriminate about equally well *within* a lattice (0.44-0.685 top-10), so what a
    combiner has to exploit is their disagreement -- and on raw units it would first have to learn
    twenty-one lattice-dependent scale corrections from the training split before it could learn
    anything about crystallography.

    Returns a new frame with `{prefix}{merit}__{method}` columns appended. Nothing is materialised
    to disk: the fitted nulls are 5-150 kB and the transform is one interpolate plus one gather, so
    a caller applies them in its own loader rather than maintaining a second copy of a ten-million
    row feature matrix that can drift out of step with the first.
    """
    columns = {}
    for null in nulls:
        columns[f'{prefix}{null.merit}__{null.label}'] = null.apply(frame)
    return frame.assign(**columns)


def load_scalers(models_dir, merits=None, methods=('analytic', 'z', 'rank')):
    """Every fitted null on disk for the requested merits and methods, ready for `scaled_features`.

    Missing combinations are skipped rather than raising: `analytic` exists only for the two merits
    with a closed-form null, and asking for it across the zoo is the normal case.
    """
    from pathlib import Path

    loaded = []
    for path in sorted(Path(models_dir).glob('*.npz')):
        null = FomNull.load(path)
        if merits is not None and null.merit not in set(merits):
            continue
        if null.label not in set(methods):
            continue
        loaded.append(null)
    return loaded


def regularity_ratio(frames, merit='null_tail_nll', by=('bravais_lattice',), min_count=200):
    """de Wolff's g_bar/Delta per group, measured on the incorrect-candidate null.

    His section (d): real calculated-Q sequences are more regular than exponential, so
    g_bar < Delta with g_bar = Delta/2 in the equidistant limit, and assuming exponential therefore
    *over-estimates* reliability -- most for high symmetry. That is the origin of the cross-lattice
    bias Wu 1988 measured as 1.82 (cubic) falling to 1.00 (triclinic), and it is one bounded number
    per group with a known range of [1/2, 1].

    **Bounded by R1.** `prune_below_m20` keeps the better-fitting wrong cells, so the retained
    negatives fit better than a random wrong cell, mean_term is inflated and rho is biased *down* --
    towards "more regular" -- by an amount that is largest where the truncation bites hardest, which
    is the low-symmetry lattices. The ordering across lattices is therefore compressed towards the
    low-symmetry end and the absolute values are lower bounds, not point estimates.
    """
    scale = NULL_LAWS[merit]['scale']
    keys = ['spacegroup' if name == 'extinction_group' else name for name in by]
    rows = []
    collected = []
    for frame in frames:
        keep = (~FomMetrics.as_bool(frame['is_correct'])
                & ~FomMetrics.as_bool(frame['is_off_by_two'])
                & ~FomMetrics.as_bool(frame['is_degenerate']))
        collected.append(frame.loc[keep, sorted({merit, 'n_peaks', *keys})]
                         .reset_index(drop=True))
    null = pd.concat(collected, ignore_index=True)
    null = null.loc[np.isfinite(null[merit].to_numpy(dtype=np.float64))]
    mean_term = (null[merit].to_numpy(dtype=np.float64)*scale
                 / np.maximum(null['n_peaks'].to_numpy(dtype=np.float64), 1.0))
    null = null.assign(__mean_term=mean_term)
    for label, group in null.groupby(keys, sort=True):
        if group.shape[0] < min_count:
            continue
        sample = group['__mean_term'].to_numpy()
        rows.append(dict(
            zip(keys, label if isinstance(label, tuple) else (label,)),
            n=int(sample.size),
            mean_line_term=float(np.mean(sample)),
            rho=float(rho_from_mean_term(np.mean(sample))),
            rho_p25=float(rho_from_mean_term(np.quantile(sample, 0.75))),
            rho_p75=float(rho_from_mean_term(np.quantile(sample, 0.25))),
            ))
    return pd.DataFrame(rows)


def null_homogeneity(frames, score, quantiles=(0.5, 0.9, 0.99, 0.999), by='bravais_lattice',
                     score_columns=()):
    """The gate's second condition: is the calibrated null the same everywhere?

    After calibration the distribution of the score over *incorrect* candidates should be identical
    across Bravais lattices and extinction groups by construction, because that is what was
    standardised. This reports the quantiles per group and the largest pairwise gap, which is the
    diagnostic that the null *fit* is good.

    It is emphatically **not** a claim that the posterior should be lattice-independent -- it should
    not be, because the prior differs (S07 handoff, framing correction). Only the null is being
    equalised here.
    """
    collected = []
    for frame in frames:
        keep = (~FomMetrics.as_bool(frame['is_correct'])
                & ~FomMetrics.as_bool(frame['is_off_by_two'])
                & ~FomMetrics.as_bool(frame['is_degenerate']))
        subset = frame.loc[keep]
        if not subset.shape[0]:
            continue
        values = np.asarray(score(subset), dtype=np.float64) if callable(score) else \
            subset[score].to_numpy(dtype=np.float64)
        collected.append(pd.DataFrame({by: subset[by].astype(str).to_numpy(), 'score': values}))
    null = pd.concat(collected, ignore_index=True)
    null = null.loc[np.isfinite(null['score'])]
    rows = []
    for label, group in null.groupby(by, sort=True):
        row = {by: label, 'n': int(group.shape[0])}
        for level in quantiles:
            row[f'q{level:g}'] = float(np.quantile(group['score'], level))
        rows.append(row)
    table = pd.DataFrame(rows)
    for level in quantiles:
        column = f'q{level:g}'
        table[f'{column}_spread'] = float(table[column].max() - table[column].min())
    return table
