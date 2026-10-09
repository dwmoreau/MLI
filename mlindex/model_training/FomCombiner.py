"""The learned ranker: a calibrated probability that a candidate unit cell is correct.

One gradient-boosted classifier scores every candidate of a pattern, across all fourteen Bravais
lattices, from 30 inputs: seven merits, four systematic-absence counts, three further merits,
eleven structural quantities, the candidate's Bravais lattice, and four pool-context gaps (how far
the candidate sits below the pattern's best value of a merit). Its output is mapped to a
probability by an isotonic regression fitted separately for each Bravais lattice, on crystals the
classifier was not fitted on, using the rows it will score: the pool as a run at the cut leaves it.

The classifier's rows come from a benchmark pool (`Benchmark.py`) in two thinning steps, because a
pool holds of order one correct candidate in several thousand:

1. `thin_negatives`, per (pattern, condition, lattice): every correct candidate, the union of the
   `top_k` best candidates by each merit, and a Bernoulli sample of the rest. `sampling_weight` is
   the inverse of each row's inclusion probability (1 or 1/rate), and the fit is weighted by it.
2. `cap_negatives`, per (pattern, condition): every correct candidate and at most `n_negatives`
   wrong ones. This rebalancing is deliberately not weighted back: at the pool's base rate the
   trees learn almost nothing. The calibrators, fitted on unthinned rows, state the probability.

Both thinnings draw their random numbers from `keyed_uniform`, a hash of the candidate's key and a
seed, so whether a candidate is kept does not depend on the order or grouping its rows are read in.

The lattice enters the classifier through one of three encodings. `onehot` and `ordinal` export to
ONNX exactly; `native` uses the classifier's categorical splits, which `skl2onnx` writes as
ordinary threshold splits, so its ONNX export would score differently and none is written.

    from mlindex.model_training.FomCombiner import FomCombiner
    combiner = FomCombiner.fit(train_frame, encoding='onehot', seed=12345,
                               params=dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63))
    combiner.fit_calibrators(calibration_frame)
    probability = combiner.score(candidates)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlindex.model_training.Benchmark import CANDIDATE_KEY
from mlindex.model_training.Benchmark import ENTRY_KEY
from mlindex.model_training.BenchmarkMetrics import as_bool
from mlindex.model_training.BenchmarkMetrics import lattice_order_of
from mlindex.utilities.FigureOfMerits import HIGHER_IS_BETTER
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

RAW_MERITS = ('M20', 'M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap')
ABSENCE_FEATURES = ('n_absent_extra', 'n_absent_extra_in_range', 'f_absent_extra',
                    'n_groups_searched')
PROBATION_MERITS = ('M_wu', 'M_1', 'F_N_q')
STRUCTURAL_FEATURES = (
    'n_indexed', 'n_entering', 'final_rank', 'N_cal_full', 'zone_dominance', 'V_over_Vcrit',
    'delta_dewolff61', 'n_dewolff61', 'M_werner_max', 'log_volume', 'pool_size_full',
    )
LATTICE_FEATURE = 'bravais_lattice'
# The merits whose gap to the pattern's best value is an input.
CONTEXT_MERITS = ('M20', 'M_sym', 'n_over', 'max_gap')
CONTEXT_FEATURES = tuple(f'ctx_{merit}_gap_to_best' for merit in CONTEXT_MERITS)

FEATURES = (RAW_MERITS + ABSENCE_FEATURES + PROBATION_MERITS + STRUCTURAL_FEATURES
            + (LATTICE_FEATURE,) + CONTEXT_FEATURES)


def features_without(names):
    """`FEATURES` in their order, less `names`; refuses a name that is not one of them."""
    unknown = sorted(set(names) - set(FEATURES))
    if unknown:
        raise ValueError(f'not ranker inputs: {unknown}; the inputs are {list(FEATURES)}')
    if set(names) >= set(FEATURES):
        raise ValueError('removing every input leaves nothing to fit on')
    return tuple(name for name in FEATURES if name not in set(names))


LATTICE_ENCODINGS = ('onehot', 'ordinal', 'native')
# Only these encodings have an ONNX export that scores as the classifier does.
EXPORTABLE_ENCODINGS = ('onehot', 'ordinal')


# Columns no feature may be: labels, quantities derived from the true cell, properties of the
# synthetic conditions, the thinning weights, and constants of the generation run.
FORBIDDEN_COLUMNS = frozenset({
    'is_correct', 'is_off_by_two', 'is_degenerate', 'split', 'condition_bundle',
    'q2_error_multiplier', 'intercept_scale', 'n_contaminants', 'n_contaminants_achieved',
    'n_dropout', 'n_dropout_achieved', 'second_phase_lines', 'second_phase_achieved',
    'second_phase_partner', 'sampling_weight', 'm20_at_prune', 'merit_at_prune',
    'in_top_n', 'prune_threshold', 'downsample_radius', 'assignment_threshold', 'q2_digest',
    'ctx_pool_size',
    # The peak count of the crystal's whole simulated pattern: set by its true cell and symmetry,
    # and not available from a peak list.
    'n_peaks_available',
    })
FORBIDDEN_SUFFIX = '_true'

ALLOWED_WEIGHT_COLUMNS = ('sampling_weight',)
POOLED = '__pooled__'

# The classifier settings the tuned ones override.
DEFAULT_PARAMS = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=40,
                      l2_regularization=1.0)


def check_no_leakage(names):
    """Raise if any feature is a label, is derived from the truth, or describes the generator."""
    offenders = sorted({name for name in names
                        if name in FORBIDDEN_COLUMNS or name.endswith(FORBIDDEN_SUFFIX)})
    if offenders:
        raise ValueError(f'features that are unavailable at inference or derived from the truth: '
                         f'{offenders}')


def fit_weights(frame, column):
    """The per-row weight, refusing a column that is absent, unknown or not positive."""
    if column not in ALLOWED_WEIGHT_COLUMNS:
        raise ValueError(f'{column!r} is not an inclusion weight; allowed: '
                         f'{list(ALLOWED_WEIGHT_COLUMNS)}')
    if column not in frame.columns:
        raise ValueError(f'{column!r} is not in the frame; a subsampled frame cannot be fitted '
                         'unweighted without biasing the result')
    values = frame[column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f'{column!r} carries a non-positive or non-finite weight')
    return values


# ---------------------------------------------------------------------------------------------
# Building a training frame from a pool
# ---------------------------------------------------------------------------------------------
def keyed_uniform(frame, seed, stream=0):
    """A uniform draw in [0, 1) per candidate, fixed by its key, `seed` and `stream` alone.

    Different streams of one seed are independent draws, for two samplings that must not share
    their random numbers.
    """
    hash_key = f'{int(seed) % 10**12:012d}{int(stream):04d}'
    hashed = pd.util.hash_pandas_object(frame[CANDIDATE_KEY], index=False, hash_key=hash_key)
    return (hashed.to_numpy(dtype=np.uint64) >> np.uint64(11)).astype(np.float64) * 2.0**-53


def oriented(frame, merit):
    """The merit with its sign set so that larger is better."""
    values = frame[merit].to_numpy(dtype=np.float64)
    return values if HIGHER_IS_BETTER[merit] else -values


def top_k_mask(frame, top_k):
    """True for a candidate among the `top_k` best of its (pattern, condition, lattice) pool by any
    of the seven merits, each ranked best first."""
    keys = [frame['entry_id'], frame['condition_bundle'], frame['bravais_lattice']]
    in_top_k = np.zeros(frame.shape[0], dtype=bool)
    for merit in RAW_MERITS:
        ranks = pd.Series(oriented(frame, merit), index=frame.index).groupby(
            keys, sort=False).rank(method='first', ascending=False)
        in_top_k |= (ranks <= top_k).to_numpy()
    return in_top_k


def thin_negatives(frame, top_k, negative_rate, seed, in_top_k=None):
    """Every correct candidate, the `top_k` best by each merit, and a sample of the rest.

    Applied within each (pattern, condition, lattice) pool. Adds `sampling_weight`, the inverse
    inclusion probability: 1 for a row kept with certainty, `1/negative_rate` for a sampled one.
    `in_top_k` is `top_k_mask(frame, top_k)`, for a caller thinning one frame at several seeds.
    """
    if not 0.0 < negative_rate <= 1.0:
        raise ValueError(f'negative_rate must be in (0, 1], got {negative_rate}')
    if in_top_k is None:
        in_top_k = top_k_mask(frame, top_k)
    correct = as_bool(frame['is_correct'])
    certain = in_top_k | correct
    sampled = ~certain & (keyed_uniform(frame, seed) < negative_rate)
    keep = certain | sampled
    weight = np.where(certain, 1.0, 1.0/negative_rate)
    return frame.loc[keep].assign(sampling_weight=weight[keep]).reset_index(drop=True)


def negative_order(frame, seed):
    """Each wrong candidate's position among its pattern's wrong candidates, in draw order.

    Correct candidates get -1. Keeping the rows with `negative_order < n` is a uniform sample of
    `n` wrong candidates per (pattern, condition), and the samples for different `n` are nested.
    """
    codes = frame.groupby(ENTRY_KEY, sort=False).ngroup().to_numpy()
    correct = as_bool(frame['is_correct'])
    draw = keyed_uniform(frame, seed, stream=1)
    draw[correct] = -1.0
    # Sorted by pattern, then draw, the correct rows of each pattern come first.
    order = np.lexsort((draw, codes))
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1 if codes.size else 0))
    position = np.empty(codes.size, dtype=np.int64)
    position[order] = np.arange(codes.size) - starts[codes[order]]
    n_correct = np.bincount(codes[correct], minlength=starts.size)
    return np.where(correct, -1, position - n_correct[codes]).astype(np.int64)


def cap_negatives(frame, n_negatives, order_column='negative_order'):
    """Every correct candidate and the first `n_negatives` wrong ones per (pattern, condition).

    `order_column` is `negative_order`.
    """
    order = frame[order_column].to_numpy()
    keep = (order < 0) | (order < n_negatives)
    return frame.loc[keep].reset_index(drop=True)


def context_best(frame):
    """Each pattern's best (oriented) value of every context merit, one row per pattern."""
    best = pd.DataFrame({name: frame[name].to_numpy() for name in ENTRY_KEY})
    for merit in CONTEXT_MERITS:
        best[f'best_{merit}'] = oriented(frame, merit)
    return best.groupby(ENTRY_KEY, sort=False, as_index=False).max()


def add_context(frame, best=None):
    """The four `ctx_*_gap_to_best` columns: each candidate's oriented merit minus its
    pattern's best.

    `best` is `context_best` over the whole pattern, all fourteen lattices; it is computed from
    `frame` when not given, which is only right when `frame` holds every lattice of each pattern.
    """
    if best is None:
        best = context_best(frame)
    before = frame.shape[0]
    merged = frame.merge(best, on=ENTRY_KEY, how='left', validate='m:1')
    if merged.shape[0] != before:
        raise ValueError('the context join changed the row count')
    columns = {f'ctx_{merit}_gap_to_best':
               oriented(merged, merit) - merged[f'best_{merit}'].to_numpy(dtype=np.float64)
               for merit in CONTEXT_MERITS}
    return merged.drop(columns=[f'best_{merit}' for merit in CONTEXT_MERITS]).assign(**columns)


def add_derived(frame):
    """`log_volume` and `f_absent_extra`, which are computed from columns already on the row."""
    in_range = frame['n_ref_in_range'].to_numpy(dtype=np.float64)
    return frame.assign(
        log_volume=np.log(frame['volume'].to_numpy(dtype=np.float64)),
        f_absent_extra=np.where(
            in_range > 0,
            frame['n_absent_extra_in_range'].to_numpy(dtype=np.float64)/np.maximum(in_range, 1.0),
            np.nan))


def split_crystals(entries, fractions, rng):
    """Divide the source crystals of an entry table into named parts, within each true lattice.

    `fractions` maps a part name to its share; the shares must sum to one. Splitting by crystal
    keeps the conditions of one crystal, which share its structure, on one side of every boundary.
    Returns {name: set of entry_id}.
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f'fractions must sum to 1, got {fractions}')
    crystals = entries[['entry_id', 'bravais_lattice_true']].drop_duplicates('entry_id')
    parts = {name: set() for name in fractions}
    names = list(fractions)
    edges = np.cumsum([fractions[name] for name in names])[:-1]
    for lattice in BRAVAIS_LATTICES:
        ids = np.sort(crystals.loc[crystals['bravais_lattice_true'] == lattice,
                                   'entry_id'].to_numpy())
        if not ids.size:
            continue
        # A stratified position in [0, 1): the k-th of n crystals, in random order, lands in
        # [k/n, (k+1)/n). Each part gets its share of a lattice to within one crystal, and a
        # lattice too small to divide is assigned by chance rather than always to the first part.
        position = (rng.permutation(ids.size) + rng.random(ids.size))/ids.size
        assigned = np.searchsorted(edges, position, side='right')
        for index, name in enumerate(names):
            parts[name].update(ids[assigned == index].tolist())
    return parts


# ---------------------------------------------------------------------------------------------
# Exporting a pool, one condition bundle at a time
# ---------------------------------------------------------------------------------------------
# What is read from a pool's candidate shards, beside the key. The merit and feature sidecars are
# read whole; the entry table supplies `pool_size_full`, the one per-pattern input.
SHARD_COLUMNS = ('M20', 'n_indexed', 'final_rank', 'n_entering', 'volume', 'm20_at_prune',
                 'in_top_n', 'is_correct', 'spacegroup')
ENTRY_FEATURES = ('pool_size_full',)
# `spacegroup`, the candidate's extinction group, rides along as a plain column: not an input,
# but what a prior on the group is looked up by.
TRAINING_COLUMNS = tuple(CANDIDATE_KEY) + ('is_correct', 'sampling_weight', 'negative_order',
                                           'spacegroup')
EVALUATION_COLUMNS = tuple(CANDIDATE_KEY) + ('is_correct', 'in_top_n', 'spacegroup')


def _read_lattice(pool, bundle, lattice, entry_ids, columns=SHARD_COLUMNS,
                  sidecar_columns=None, truth_dir=None, truth_ids=()):
    """One lattice's candidates of one bundle, for the crystals in `entry_ids`. `sidecar_columns`
    maps a sidecar to the columns wanted from it; without it both sidecars are read whole.

    `truth_dir`, a `BenchmarkRuns.truth_pool` of the same bundle, adds its true cells for the
    crystals in `truth_ids`, those still correct after their refinement.
    """
    from mlindex.model_training import Benchmark

    if sidecar_columns is None:
        sidecar_columns = {Benchmark.MERIT_SIDECAR: None, Benchmark.FEATURE_SIDECAR: None}
    wanted = list(CANDIDATE_KEY) + list(dict.fromkeys(list(columns) + ['is_correct']))

    def read(directory, ids):
        frame = Benchmark.load_candidates(
            directory, bundle, columns=wanted, bravais_lattices=[lattice],
            sidecars=tuple(sidecar_columns), sidecar_columns=sidecar_columns)
        return frame.loc[frame['entry_id'].isin(ids)]

    frame = read(pool, entry_ids)
    shard = Path(truth_dir or '.') / f'candidates_{bundle}_{lattice}.parquet'
    if truth_dir is not None and len(truth_ids) and shard.exists():
        truth = read(truth_dir, truth_ids)
        frame = pd.concat([frame, truth.loc[as_bool(truth['is_correct'])]])
    return frame.reset_index(drop=True)


def _first_negatives(frame, n_negatives, seed):
    return frame.loc[negative_order(frame, seed) < n_negatives].reset_index(drop=True)


def _float32(frame):
    floats = frame.select_dtypes(include=['float64']).columns
    return frame.astype({name: np.float32 for name in floats})


def export_bundle(pool, bundle, entries, training_ids, evaluation_ids, seeds, cut, n_top,
                  keep_all_depths, top_k, negative_rate, n_negatives, truth_dir=None):
    """One condition bundle of a pool as ranker frames: training rows per seed, evaluation rows.

    Both are the pool as a run at `cut` would leave it (`restrict_at_cut`): `final_rank`,
    `pool_size_full` and the context gaps are recomputed over the restricted pool, the gaps over
    every lattice of the pattern.

    Training rows, for the crystals in `training_ids`: every depth, restricted at the cut by the
    same rule as the evaluation rows, so no row is kept for being correct. The bundle's true
    cells from `truth_dir` (a `BenchmarkRuns.truth_pool`), added where the search found none, are
    candidates of their patterns under that rule too: one that refines to an M20 below the cut
    is a cell the run would not have kept, and is not a row. Then thinned by `thin_negatives` and
    to the first `n_negatives` wrong candidates per pattern by `negative_order`, both draws keyed
    on the seed; a later `cap_negatives` at or below `n_negatives` selects from these rows exactly
    as it would from the whole pool.

    Evaluation rows, for the crystals in `evaluation_ids`: as the search left them, the `n_top`
    per lattice unless `keep_all_depths`.

    The pool is read one lattice at a time, twice: once for each pattern's best values and
    survivor counts, once for the rows. Returns ({seed: frame}, frame).
    """
    from mlindex.model_training import Benchmark
    from mlindex.model_training.BenchmarkMetrics import restrict_at_cut
    from mlindex.model_training.BenchmarkRuns import TRUTH_CANDIDATE_ID

    lattices = [lattice for lattice, _ in Benchmark.candidate_shards(pool, bundle)]
    wanted = set(training_ids) | set(evaluation_ids)
    context_columns = {Benchmark.MERIT_SIDECAR: ['M_sym', 'n_over', 'max_gap']}

    def restricted(frame):
        """(training rows, evaluation rows) of one lattice, each restricted to the cut by the
        one rule; the added true cells are training rows only."""
        added = frame['candidate_id'].to_numpy() == TRUTH_CANDIDATE_ID
        training = restrict_at_cut(frame.loc[frame['entry_id'].isin(training_ids)], cut,
                                   n_top=n_top)
        evaluation = restrict_at_cut(
            frame.loc[frame['entry_id'].isin(evaluation_ids) & ~added], cut, n_top=n_top)
        return training, evaluation

    best, survivors = {'training': [], 'evaluation': []}, {'training': [], 'evaluation': []}
    for lattice in lattices:
        frame = _read_lattice(pool, bundle, lattice, wanted, columns=('M20', 'm20_at_prune'),
                              sidecar_columns=context_columns, truth_dir=truth_dir,
                              truth_ids=training_ids)
        for name, rows in zip(('training', 'evaluation'), restricted(frame)):
            if rows.shape[0]:
                best[name].append(context_best(rows))
                survivors[name].append(rows.groupby(ENTRY_KEY, as_index=False).size())

    def combine(parts, how):
        return (pd.concat(parts).groupby(ENTRY_KEY, as_index=False).agg(how) if parts else None)

    best = {name: combine(parts, 'max') for name, parts in best.items()}
    survivors = {name: combine(parts, 'sum') for name, parts in survivors.items()}

    def in_restricted_pool(rows, name):
        """`pool_size_full` and the context gaps over the whole restricted pattern."""
        rows = rows.drop(columns='pool_size_full').merge(
            survivors[name].rename(columns={'size': 'pool_size_full'}), on=ENTRY_KEY,
            how='left')
        return add_context(rows, best[name])

    kept = {seed: [] for seed in seeds}
    evaluation_parts = []
    for lattice in lattices:
        frame = add_derived(Benchmark.attach_entry_columns(
            _read_lattice(pool, bundle, lattice, wanted, truth_dir=truth_dir,
                          truth_ids=training_ids), entries, ENTRY_FEATURES))
        training, evaluation = restricted(frame)
        if training.shape[0]:
            training = in_restricted_pool(training, 'training').reset_index(drop=True)
            in_top_k = top_k_mask(training, top_k)
            for seed in seeds:
                thinned = thin_negatives(training, top_k, negative_rate, seed, in_top_k=in_top_k)
                # The first `n_negatives` of the pattern overall are among the first
                # `n_negatives` of the lattices read so far, so truncating as we go loses none.
                kept[seed] = [_first_negatives(pd.concat(kept[seed] + [thinned],
                                                         ignore_index=True), n_negatives, seed)]
        if evaluation.shape[0]:
            evaluation = in_restricted_pool(evaluation, 'evaluation')
            if not keep_all_depths:
                evaluation = evaluation.loc[evaluation['in_top_n']]
            evaluation_parts.append(evaluation[list(EVALUATION_COLUMNS) + [
                name for name in FEATURES if name not in EVALUATION_COLUMNS]])

    training_frames = {}
    for seed in seeds:
        if not kept[seed]:
            continue
        frame = kept[seed][0]
        frame['negative_order'] = negative_order(frame, seed)
        training_frames[seed] = _float32(frame[list(TRAINING_COLUMNS) + [
            name for name in FEATURES if name not in TRAINING_COLUMNS]])
    evaluation_frame = (_float32(pd.concat(evaluation_parts, ignore_index=True))
                        if evaluation_parts else None)
    return training_frames, evaluation_frame


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------
def fit_calibration(raw, target, lattice, minimum=200):
    """Isotonic knots per Bravais lattice, and pooled ones for a lattice with too few rows.

    A lattice gets its own calibrator when it has at least `minimum` rows and both classes.
    Returns {lattice or POOLED: (thresholds, values)}.
    """
    from sklearn.isotonic import IsotonicRegression

    def knots(mask):
        fitted = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        fitted.fit(raw[mask], target[mask])
        return (np.asarray(fitted.X_thresholds_, dtype=np.float64),
                np.asarray(fitted.y_thresholds_, dtype=np.float64))

    raw = np.asarray(raw, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    lattice = np.asarray(lattice)
    calibrators = {POOLED: knots(np.ones(raw.size, dtype=bool))}
    for name in np.unique(lattice):
        mask = lattice == name
        if int(mask.sum()) >= minimum and np.unique(target[mask]).size > 1:
            calibrators[str(name)] = knots(mask)
    return calibrators


# The three ways one fitted classifier can rank a pool: through each lattice's isotonic map (the
# shipped design), through the one pooled map for every lattice, and by its raw score. The pooled
# map preserves the raw order except where its steps make ties; the per-lattice maps can reorder
# candidates of different lattices.
CALIBRATION_ARMS = ('per_lattice', 'pooled', 'raw')


def calibration_arms(raw, lattice, calibrators):
    """{arm: score} for each of `CALIBRATION_ARMS`, from one raw score."""
    raw = np.asarray(raw, dtype=np.float64)
    thresholds, values = calibrators[POOLED]
    return {'per_lattice': apply_calibration(raw, lattice, calibrators),
            'pooled': np.interp(raw, thresholds, values),
            'raw': raw}


def apply_calibration(raw, lattice, calibrators):
    """The calibrated probability: each row's raw score through its lattice's isotonic knots."""
    raw = np.asarray(raw, dtype=np.float64)
    lattice = np.asarray(lattice)
    out = np.empty(raw.size, dtype=np.float64)
    for name in np.unique(lattice):
        mask = lattice == name
        thresholds, values = calibrators.get(str(name), calibrators[POOLED])
        out[mask] = np.interp(raw[mask], thresholds, values)
    return out


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------
class FomCombiner:
    """A fitted classifier, its lattice encoding, and its per-lattice calibrators."""

    def __init__(self, encoding, features=FEATURES, model=None, calibrators=None, meta=None):
        if encoding not in LATTICE_ENCODINGS:
            raise ValueError(f'encoding must be one of {LATTICE_ENCODINGS}, got {encoding!r}')
        check_no_leakage(features)
        self.encoding = encoding
        self.features = tuple(features)
        self.model = model
        self.calibrators = calibrators or {}
        self.meta = meta or {}

    @property
    def matrix_names(self):
        """The design matrix's column names, with the lattice expanded as the encoding requires."""
        names = []
        for name in self.features:
            if name == LATTICE_FEATURE and self.encoding == 'onehot':
                names.extend(f'{LATTICE_FEATURE}={lattice}' for lattice in BRAVAIS_LATTICES)
            else:
                names.append(name)
        return tuple(names)

    @property
    def categorical_indices(self):
        if self.encoding != 'native' or LATTICE_FEATURE not in self.features:
            return None
        return [self.matrix_names.index(LATTICE_FEATURE)]

    def design_matrix(self, frame):
        """The float32 (n_candidates, n_columns) array the classifier and its ONNX export read."""
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise KeyError(f'frame is missing feature column(s): {missing}')
        columns = []
        for name in self.features:
            if name != LATTICE_FEATURE:
                columns.append(frame[name].to_numpy(dtype=np.float32)[:, np.newaxis])
                continue
            # Position in the canonical lattice order, highest symmetry first; refuses a lattice
            # outside it.
            position = lattice_order_of(frame[LATTICE_FEATURE].astype(str).to_numpy())
            if self.encoding == 'onehot':
                columns.append(np.eye(len(BRAVAIS_LATTICES), dtype=np.float32)[position])
            else:
                columns.append((position + 1).astype(np.float32)[:, np.newaxis])
        return np.concatenate(columns, axis=1)

    @classmethod
    def fit(cls, frame, encoding, seed, params=None, features=FEATURES):
        """Fit the classifier on a `cap_negatives` frame, weighted by `sampling_weight`."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        combiner = cls(encoding, features=features)
        settings = dict(DEFAULT_PARAMS)
        settings.update(params or {})
        matrix = combiner.design_matrix(frame)
        target = as_bool(frame['is_correct']).astype(np.int32)
        weights = fit_weights(frame, 'sampling_weight')
        combiner.model = HistGradientBoostingClassifier(
            categorical_features=combiner.categorical_indices, early_stopping=False,
            random_state=int(seed), **settings)
        combiner.model.fit(matrix, target, sample_weight=weights)
        combiner.meta = dict(seed=int(seed), encoding=encoding, params=settings,
                             n_rows=int(frame.shape[0]), n_positive=int(target.sum()),
                             weight_column='sampling_weight', weight_sum=float(weights.sum()),
                             n_iter=int(combiner.model.n_iter_))
        return combiner

    def fit_calibrators(self, frame, minimum=200):
        """Per-lattice isotonic calibration on rows the classifier was not fitted on, of the kind
        it will score."""
        self.calibrators = fit_calibration(
            self.raw_score(frame), as_bool(frame['is_correct']), frame[LATTICE_FEATURE],
            minimum=minimum)
        self.meta.update(n_calibration_rows=int(frame.shape[0]),
                         calibrated_lattices=sorted(set(self.calibrators) - {POOLED}))
        return self

    def raw_score(self, frame):
        """The classifier's probability, before calibration."""
        return self.model.predict_proba(self.design_matrix(frame))[:, 1].astype(np.float64)

    def score(self, frame):
        """The calibrated probability that each candidate is correct."""
        return apply_calibration(self.raw_score(frame), frame[LATTICE_FEATURE], self.calibrators)

    # -- persistence ----------------------------------------------------------------------
    def save(self, directory):
        """Write `model.joblib`, `calibrators.npz`, `specification.json` and, when the encoding
        allows, `model.onnx`. Refuses a directory that already holds a model."""
        import joblib

        directory = Path(directory)
        if (directory/'specification.json').exists():
            raise FileExistsError(f'{directory} already holds a model; each fit writes its own')
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory/'model.joblib')
        np.savez_compressed(directory/'calibrators.npz', **{
            f'{name}__{part}': array for name, (x, y) in self.calibrators.items()
            for part, array in (('x', x), ('y', y))})
        onnx_name = None
        if self.encoding in EXPORTABLE_ENCODINGS:
            from mlindex.utilities.IOManagers import SKLearnManager

            SKLearnManager(filename=str(directory/'model'), model_type='onnx').save(
                model=self.model, n_features=len(self.matrix_names))
            onnx_name = 'model.onnx'
        with open(directory/'specification.json', 'w', encoding='utf-8') as handle:
            json.dump(dict(features=list(self.features), matrix_names=list(self.matrix_names),
                           encoding=self.encoding, onnx=onnx_name, meta=self.meta),
                      handle, indent=2)
        return directory

    @classmethod
    def load(cls, directory):
        import joblib

        directory = Path(directory)
        with open(directory/'specification.json', encoding='utf-8') as handle:
            specification = json.load(handle)
        arrays = np.load(directory/'calibrators.npz')
        names = {key.rsplit('__', 1)[0] for key in arrays.files}
        calibrators = {name: (arrays[f'{name}__x'], arrays[f'{name}__y']) for name in names}
        combiner = cls(specification['encoding'], features=specification['features'],
                       model=joblib.load(directory/'model.joblib'), calibrators=calibrators,
                       meta=specification['meta'])
        if list(combiner.matrix_names) != specification['matrix_names']:
            raise ValueError(f'{directory}: the saved column order does not match the encoding')
        return combiner


def onnx_probability(path, matrix):
    """The positive-class probability the ONNX export at `path` gives for a design matrix."""
    from mlindex.utilities.IOManagers import SKLearnManager

    manager = SKLearnManager(filename=str(Path(path).with_suffix('')), model_type='onnx')
    manager.load()
    return np.asarray(manager.predict_proba(np.asarray(matrix)), dtype=np.float64)[:, 1]
