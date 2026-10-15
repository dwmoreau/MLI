"""The list `mlindex.run` prints, replayed from a benchmark pool.

A pool keeps every candidate that came through a pattern's search, and its sidecars hold the
learned ranker's inputs for each one. For every pattern this module rebuilds each lattice's pool
in the form the search hands to `run.printed_rows` -- best M20 first, the full cell,
`Ranker.CANDIDATE_INPUTS` -- and calls it, so the list is scored, shortlisted and de-duplicated by
the indexer's own code. Each printed row keeps its candidate key, and so the pool's label.

    from mlindex.model_training import BenchmarkPrinted
    printed = BenchmarkPrinted.replay_bundle(pool_dir, 'b1_error1_cont0',
                                             BenchmarkPrinted.rankings())
"""

import numpy as np
import pandas as pd

from mlindex.model_training import Benchmark
from mlindex.model_training import BenchmarkMetrics as metrics
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES

# The sidecar columns the replay reads: the inputs `candidate_inputs` computes in the search,
# with `f_absent_extra` stored as its two counts.
MERIT_COLUMNS = ('M_rev', 'M_sym', 'X_N', 'n_over')
FEATURE_COLUMNS = ('M_wu', 'F_N_q', 'zone_dominance', 'V_over_Vcrit', 'n_dewolff61',
                   'n_absent_extra_in_range', 'n_ref_in_range')
POOL_COLUMNS = Benchmark.CANDIDATE_KEY + ['lattice_system', 'final_rank', 'M20', 'n_indexed',
                                          'spacegroup', 'unit_cell', 'is_correct']
PRINTED_COLUMNS = Benchmark.ENTRY_KEY + ['ranking', 'position', 'bravais_lattice',
                                         'candidate_id', 'score', 'M20', 'M_sym', 'volume',
                                         'is_correct']
# The rankings compared: the packaged learned ranker, then the merits ranked as its fallback is.
RANKINGS = ('learned', 'M_sym', 'M20')
# Each ranking's outcomes, a column of `BenchmarkMetrics.printed_flags` each, by reported name.
OUTCOMES = {'top1': 'top1', 'top5': 'top5', 'top10': 'top10', 'mrr': 'reciprocal_rank',
            'listed': 'listed'}
# The paired contrasts reported, (ranking, reference): ranking minus reference.
CONTRASTS = (('learned', 'M_sym'), ('learned', 'M20'), ('M_sym', 'M20'))


def rankings():
    """The rankings of `RANKINGS`, by name: the packaged learned ranker, M_sym and M20.

    Refuses a model tree without the learned ranker rather than measuring the fallback twice.
    """
    from mlindex.utilities.Ranker import LearnedRanker, MeritRanker, load_ranker

    learned = load_ranker()
    if not isinstance(learned, LearnedRanker):
        raise FileNotFoundError(f'No learned ranker to measure: {learned.name}')
    return dict(zip(RANKINGS, (learned, MeritRanker('M_sym', 'benchmark'),
                               MeritRanker('M20', 'benchmark'))))


def lattice_pool(block):
    """One lattice's candidates of one pattern as the search's `pool`, and their candidate ids
    in the same order.

    `block` holds the lattice's rows with `POOL_COLUMNS` and the sidecar columns. The pool's order
    is `final_rank`, which is best M20 first over every survivor.
    """
    from mlindex.utilities.Ranker import CANDIDATE_INPUTS, absent_fraction
    from mlindex.utilities.UnitCellTools import get_full_unit_cell

    block = block.sort_values('final_rank', kind='stable')
    lattice_system = block['lattice_system'].iloc[0]
    pool = {name: block[name].to_numpy(dtype=np.float64)
            for name in CANDIDATE_INPUTS if name != 'f_absent_extra'}
    pool['f_absent_extra'] = absent_fraction(block['n_absent_extra_in_range'].to_numpy(),
                                             block['n_ref_in_range'].to_numpy())
    pool['M20'] = block['M20'].to_numpy(dtype=np.float64)
    pool['n_indexed'] = block['n_indexed'].to_numpy()
    pool['spacegroup'] = np.asarray(block['spacegroup'].tolist(), dtype=str)
    pool['unit_cell'] = get_full_unit_cell(
        np.stack(block['unit_cell'].to_numpy()).astype(np.float64), lattice_system)
    return pool, block['candidate_id'].to_numpy(), block['is_correct'].to_numpy(dtype=bool)


def replay_pattern(pattern, rankers):
    """The printed rows of one pattern under each ranking, as a list of dicts.

    `pattern` holds every candidate of one (entry, bundle), every lattice.
    """
    from mlindex.command_line.run import printed_rows

    entry_id = pattern['entry_id'].iloc[0]
    bundle = pattern['condition_bundle'].iloc[0]
    results, ids, labels = {}, {}, {}
    by_lattice = dict(tuple(pattern.groupby('bravais_lattice', sort=False)))
    for lattice in BRAVAIS_LATTICES:
        if lattice in by_lattice:
            pool, ids[lattice], labels[lattice] = lattice_pool(by_lattice[lattice])
            results[lattice] = {'pool': pool}
    rows = []
    for name, ranker in rankers.items():
        for position, entry in enumerate(printed_rows(results, ranker)):
            lattice, row = entry['bravais_lattice'], entry['row']
            rows.append({'entry_id': entry_id, 'condition_bundle': bundle, 'ranking': name,
                         'position': position, 'bravais_lattice': lattice,
                         'candidate_id': int(ids[lattice][row]),
                         'score': float(entry['score']), 'M20': float(entry['M20']),
                         'M_sym': float(entry['M_sym']), 'volume': float(entry['volume']),
                         'is_correct': bool(labels[lattice][row])})
    return rows


def replay_bundle(pool_dir, bundle, rankers, entry_ids=None):
    """The printed rows of every pattern of one condition bundle, under each of `rankers`.

    `entry_ids` restricts the replay to those crystals. Returns a frame of `PRINTED_COLUMNS`.
    """
    frame = Benchmark.load_candidates(
        pool_dir, bundle, columns=POOL_COLUMNS,
        sidecars=Benchmark.SIDECARS,
        sidecar_columns={Benchmark.MERIT_SIDECAR: list(MERIT_COLUMNS),
                         Benchmark.FEATURE_SIDECAR: list(FEATURE_COLUMNS)})
    if entry_ids is not None:
        frame = frame.loc[frame['entry_id'].isin(set(entry_ids))]
    rows = []
    for _, pattern in frame.groupby('entry_id', sort=True):
        rows += replay_pattern(pattern, rankers)
    return pd.DataFrame(rows, columns=PRINTED_COLUMNS)


def _replay_job(pool_dir, bundle):
    """`replay_bundle` in a process of its own, which loads its own rankers."""
    return replay_bundle(pool_dir, bundle, rankings())


def replay_pool(pool_dir, n_processes):
    """Every bundle of a pool replayed, a bundle a process, `n_processes` at a time."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    bundles = Benchmark.available_bundles(pool_dir)
    if n_processes <= 1:
        rankers = rankings()
        parts = [replay_bundle(pool_dir, bundle, rankers) for bundle in bundles]
    else:
        with ProcessPoolExecutor(max_workers=n_processes,
                                 mp_context=get_context('spawn')) as executor:
            parts = list(executor.map(_replay_job, [pool_dir]*len(bundles), bundles))
    return pd.concat(parts, ignore_index=True)


def pool_outcomes(pool_dir, entries):
    """The pool's own outcomes, beside the printed lists': `found`, a correct cell anywhere in the
    pool, and the tie-break floor, a constant score ranked over each lattice's twenty best by M20
    (`in_top_n`), so that ties break by lattice order. One row per entry of `entries`."""
    parts = []
    for bundle in Benchmark.available_bundles(pool_dir):
        frame = Benchmark.load_candidates(
            pool_dir, bundle, columns=Benchmark.CANDIDATE_KEY + ['in_top_n', 'is_correct'],
            sidecars=())
        frame = Benchmark.attach_entry_columns(frame, entries, ['is_degenerate'])
        reduced = metrics.reduce_pool(frame, np.zeros(frame.shape[0]))
        found = metrics.derive_flags(reduced, depth='all')[Benchmark.ENTRY_KEY + ['found']]
        constant = metrics.derive_flags(reduced, depth='in_top_n')
        parts.append(found.merge(constant[Benchmark.ENTRY_KEY + ['top1', 'top5', 'top10',
                                                                 'reciprocal_rank']],
                                 on=Benchmark.ENTRY_KEY, validate='1:1'))
    outcomes = pd.concat(parts, ignore_index=True)
    key = Benchmark.ENTRY_KEY + ['bravais_lattice_true']
    return entries[key].merge(outcomes, on=Benchmark.ENTRY_KEY, how='left',
                              validate='1:1').fillna({'found': False, 'top1': False,
                                                      'top5': False, 'top10': False,
                                                      'reciprocal_rank': 0.0})


def scopes(lattices):
    """(scope, kind, true lattices) of a report: each lattice, each lattice system, and all."""
    rows = [(lattice, 'lattice', (lattice,)) for lattice in lattices]
    for system in dict.fromkeys(BL_TO_LATTICE_SYSTEM[lattice] for lattice in lattices):
        rows.append((system, 'system', tuple(lattice for lattice in lattices
                                              if BL_TO_LATTICE_SYSTEM[lattice] == system)))
    return rows + [('all', 'all', tuple(lattices))]


def end_to_end_table(flags, pool, lattices, rng):
    """Levels and paired contrasts, in percent, per scope.

    `flags` maps each ranking to its `printed_flags`; `pool` is `pool_outcomes`. Every level is a
    mean over the scope's lattices of each lattice's mean; a contrast carries the bootstrap SE of
    `BenchmarkMetrics.balanced_delta_se`. The pool's `found` is the ceiling and its constant score
    the tie-break floor.
    """
    rows = []
    for scope, kind, members in scopes(lattices):
        def pick(frame):
            return frame.loc[frame['bravais_lattice_true'].isin(members)]

        n_patterns = int(pick(pool).shape[0])
        base = dict(scope=scope, scope_kind=kind, n_patterns=n_patterns,
                    n_crystals=int(pick(pool)['entry_id'].nunique()))
        rows.append({**base, 'metric': 'found', 'quantity': 'pool', 'kind': 'ceiling',
                     'value': 100*metrics.balanced_mean(pick(pool), 'found')})
        for metric, column in OUTCOMES.items():
            if metric != 'listed':
                rows.append({**base, 'metric': metric, 'quantity': 'constant',
                             'kind': 'tie-break floor',
                             'value': 100*metrics.balanced_mean(pick(pool), column)})
            for name, flagged in flags.items():
                rows.append({**base, 'metric': metric, 'quantity': name, 'kind': 'level',
                             'value': 100*metrics.balanced_mean(pick(flagged), column)})
            for name, reference in CONTRASTS:
                delta, se = metrics.balanced_delta_se(pick(flags[reference]), pick(flags[name]),
                                                      column, rng)
                rows.append({**base, 'metric': metric, 'quantity': f'{name} - {reference}',
                             'kind': 'contrast', 'value': 100*delta, 'se': 100*se})
    return pd.DataFrame(rows)


def provenance(pool_dir):
    """What the table is attributable to: the search (from the pool's manifest) and the replay
    (this checkout, its model tree, the ranker's files, the machine and its BLAS)."""
    import platform

    from mlindex.model_training.BenchmarkRuns import _provenance
    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
    from mlindex.utilities.Digests import tree_digest
    from mlindex.utilities.Ranker import RANKER_DIRECTORY

    manifest = Benchmark.load_manifest(pool_dir)
    replay = _provenance()
    ranker_digest, _ = tree_digest(_resolve_models_dir(), [RANKER_DIRECTORY])
    from threadpoolctl import threadpool_info

    # The library numpy loaded, which the list below first place depends on: on one machine and
    # one commit, Accelerate and OpenBLAS print different lists.
    blas = ', '.join(sorted({f"{info['internal_api']} {info.get('version')}"
                             for info in threadpool_info() if info['user_api'] == 'blas'}))
    return {
        'pool': str(pool_dir), 'population': manifest['population'], 'split': manifest['split'],
        'search_commit': manifest['commit'], 'search_arch': manifest.get('arch'),
        'seed': manifest['seed'], 'search_seed': manifest['search_seed'],
        'cut': manifest['prune_threshold'], 'search_model_revision': manifest.get('model_revision'),
        'search_models_digest': manifest.get('models_digest'),
        'replay_commit': replay['commit'], 'model_revision': replay['model_revision'],
        'models_digest': replay['models_digest'], 'ranker_digest': ranker_digest,
        'replay_arch': platform.machine(), 'replay_blas': blas,
        }


def end_to_end(pool_dir, rng, n_processes=1):
    """The end-to-end table of one pool, with its provenance on every row, and the printed rows.

    Refuses a pool searched at another cut than the indexer's, a modified checkout, and a search
    whose model files differ from the replay's.
    """
    from mlindex.model_training.BenchmarkRuns import POPULATIONS
    from mlindex.optimization.Candidates import PRUNE_M20_THRESHOLD

    Benchmark.check_complete(pool_dir)
    record = provenance(pool_dir)
    if float(record['cut']) != float(PRUNE_M20_THRESHOLD):
        raise ValueError(f'{pool_dir} was searched at cut {record["cut"]}; the indexer cuts at '
                         f'{PRUNE_M20_THRESHOLD}')
    if record['search_models_digest'] != record['models_digest']:
        raise ValueError(f'{pool_dir} was searched with model files {record["search_models_digest"]}'
                         f'; this model tree is {record["models_digest"]}')
    entries = Benchmark.load_entries(pool_dir, columns=Benchmark.ENTRY_KEY + [
        'bravais_lattice_true', 'is_degenerate'])
    printed = replay_pool(pool_dir, n_processes)
    flags = {name: metrics.printed_flags(printed.loc[printed['ranking'] == name], entries)
             for name in RANKINGS}
    lattices = [lattice for lattice in POPULATIONS[record['population']]['bravais_lattices']
                if lattice in set(entries['bravais_lattice_true'])]
    table = end_to_end_table(flags, pool_outcomes(pool_dir, entries), lattices, rng)
    return table.assign(**record), printed


# The figure's colours: the three rankings in fixed categorical order, a distinct marker each so
# that identity never rests on colour; the pool's ceiling and tie-break floor in neutral ink.
FIGURE_STYLE = {'learned': ('#2a78d6', 'o'), 'M_sym': ('#eb6834', 's'), 'M20': ('#1baf7a', '^')}
FIGURE_INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781',
              'grid': '#e1e0d9', 'surface': '#fcfcfb'}


def end_to_end_figure(table, path):
    """First place and top ten of each ranking, per lattice and over all lattices, with the
    ceiling and the tie-break floor, written to `path` (PNG)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scopes_shown = ['all'] + table.loc[table['scope_kind'] == 'lattice', 'scope'].unique().tolist()
    panels = (('top1', 'First place'), ('top10', 'Top ten'))
    figure, axes = plt.subplots(1, 2, figsize=(9, 0.42*len(scopes_shown) + 1.6), sharey=True,
                                facecolor=FIGURE_INK['surface'])
    y = np.arange(len(scopes_shown))[::-1]

    def values(metric, quantity):
        rows = table.loc[(table['metric'] == metric) & (table['quantity'] == quantity)]
        return rows.set_index('scope').loc[scopes_shown, 'value'].to_numpy()

    for axis, (metric, title) in zip(axes, panels):
        axis.set_facecolor(FIGURE_INK['surface'])
        axis.scatter(values('found', 'pool'), y, marker='|', s=420, linewidths=2,
                     color=FIGURE_INK['secondary'], label='ceiling (found)', zorder=2)
        axis.scatter(values(metric, 'constant'), y, marker='x', s=30, linewidths=1.2,
                     color=FIGURE_INK['muted'], label='tie-break floor', zorder=2)
        # Each ranking on its own line within the lattice's row, so equal values stay visible.
        for offset, (name, (colour, marker)) in zip((0.22, 0.0, -0.22), FIGURE_STYLE.items()):
            axis.scatter(values(metric, name), y + offset, marker=marker, s=64, color=colour,
                         edgecolors=FIGURE_INK['surface'], linewidths=1.5, label=name, zorder=3)
        axis.set_title(title, color=FIGURE_INK['primary'], fontsize=11, loc='left')
        axis.set_xlim(-3, 103)
        axis.set_xlabel('percent of patterns', color=FIGURE_INK['secondary'])
        axis.grid(axis='x', color=FIGURE_INK['grid'], linewidth=0.8)
        axis.axhline(y[0] - 0.5, color=FIGURE_INK['grid'], linewidth=0.8)
        axis.tick_params(colors=FIGURE_INK['muted'], length=0)
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)
        axis.spines['bottom'].set_color(FIGURE_INK['grid'])
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, ['all lattices'] + scopes_shown[1:], color=FIGURE_INK['primary'])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=5, frameon=False,
                  labelcolor=FIGURE_INK['secondary'])
    population = table['population'].iloc[0]
    figure.suptitle(f'fom-dev {population}: the printed list under each ranking',
                    color=FIGURE_INK['primary'], x=0.02, ha='left')
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(path, dpi=150, facecolor=FIGURE_INK['surface'])
    plt.close(figure)
