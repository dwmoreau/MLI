"""Choose how much of the candidate budget each generator should get.

The indexer draws candidate unit cells from three generators -- a random forest (`trees`), a
neural network (`abnn`) and a Miller-index template library (`templates`) -- and refines the
mixture. The shares are fixed numbers in `UtilitiesOptimizer.py`. This driver measures what those
shares should be, by generating each generator's candidates separately for patterns whose answer
is known and scoring every mixture of them against the measured convergence curve.

It runs in two stages, because generating candidates costs node-hours and does not depend on the
score, while scoring costs seconds and is the thing that gets changed:

  --stage generate   draw each generator's pool for many patterns and write it to disk. MPI.
  --stage fit        read those pools and score every mixture on a grid. Multiprocessing, not
                     MPI, over crystals; --nproc. One row per lattice, score, reduction and
                     split, written to ensemble_mix.csv.

The score is the expected number of INDEPENDENT candidates that converge. Candidates in a real
pool are not independent -- a clump that starts close together succeeds or fails together, and is
worth fewer tries than counting it would say -- so every candidate carries the measured clump
discount, which --clump-discount supplies and without which the fit refuses to run.

Worked commands. One lattice end to end on a laptop, which takes a few minutes:

    MLINDEX_MODELS_DIR=$PWD/mlindex/models python -m mlindex.scripts.run_ensemble_refine \\
        --stage all --bravais-lattices cP --n-entries 20 \\
        --dataset-directory mlindex/data/generated_datasets \\
        --roc-dir docs/fom_production/artifacts/P08_inputs/data \\
        --pools /tmp/pools --out-dir /tmp/mix

Generation for every lattice, on a node. Use mpiexec, not srun: a conda-built mpi4py does not
read srun's process management, so every task comes up alone and runs the whole job. The stage
refuses to start in that state.

    mpiexec -n 32 python -m mlindex.scripts.run_ensemble_refine --stage generate \\
        --dataset-directory mlindex/data/generated_datasets --pools results/pools

Then, refitting as often as the score changes, against those pools. Give --nproc the cores of
whatever it is running on: the work is one neighbour search per crystal per mix, so at production
scale a single core is days and a node is under an hour.

    python -m mlindex.scripts.run_ensemble_refine --stage fit --pools results/pools \\
        --roc-dir docs/fom_production/artifacts/P08_inputs/data --out-dir results/mix \\
        --clump-discount mlindex/characterization/clump_discount --nproc 128
"""
import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mlindex
from mlindex.optimization import UtilitiesOptimizer
from mlindex.optimization.GeneratorPools import generate_candidate_pools
from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
from mlindex.utilities.Allocation import largest_remainder
from mlindex.utilities.ClumpDiscount import CUBIC, clump_weights, load_clump_discount
from mlindex.utilities.ConvergenceCurve import load_curve
from mlindex.utilities.Digests import derived_seed
from mlindex.utilities.EnsembleObjective import expected_success_objective
from mlindex.utilities.Redistribution import redistribute_xnn
from mlindex.model_training import BenchmarkConditions
from mlindex.model_training import BenchmarkPatterns
from mlindex.utilities.ErrorAdder import ContaminantPlacementError
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM


BRAVAIS_LATTICES = ('cF', 'cI', 'cP', 'hP', 'hR', 'tI', 'tP',
                    'oC', 'oF', 'oI', 'oP', 'mC', 'mP', 'aP')
FACTORY_OF_SYSTEM = {
    'cubic': 'get_cubic_optimizer',
    'hexagonal': 'get_hexagonal_optimizer',
    'rhombohedral': 'get_rhombohedral_optimizer',
    'tetragonal': 'get_tetragonal_optimizer',
    'orthorhombic': 'get_orthorhombic_optimizer',
    'monoclinic': 'get_monoclinic_optimizer',
    'triclinic': 'get_triclinic_optimizer',
    }
# The cubic curves were measured on ten peaks and the rest on twenty, so a pattern is cut to the
# same count the curve for its lattice was measured at.
N_PEAKS = {'cF': 10, 'cI': 10, 'cP': 10}
DEFAULT_N_PEAKS = 20
BROADENING_TAG = '1'


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def check_mpi_world(comm, n_ranks):
    """Refuse to run one copy of the whole job per task because the launcher never reached MPI.

    A conda-built mpi4py does not read the process-management information srun hands out, so every
    task comes up in a world of size one, believes it is the only rank, takes the whole crystal
    list and does the entire job. Nothing about that shows up in the results: it is N times the
    cost for one job's work, and N processes writing one file.
    """
    launched = int(os.environ.get('SLURM_NTASKS') or 0)
    if n_ranks == 1 and launched > 1:
        raise SystemExit(
            f'the launcher started {launched} tasks but MPI reports a world of 1, so every task '
            f'would run the whole job. Launch with mpiexec rather than srun.'
            )
    return comm


def shipped_mix_and_budget(optimizer):
    """The shares and the budget this lattice ships with, read off the optimizer.

    Taken from `generator_info` rather than transcribed, so the comparison is always against what
    the package actually does. A generator split across several split groups contributes each of
    its entries, which is why the counts are summed per name rather than read off one row.
    """
    counts = {}
    order = []
    for generator_info in optimizer.opt_params['generator_info']:
        name = generator_info['generator']
        if name not in counts:
            counts[name] = 0
            order.append(name)
        counts[name] += generator_info['n_unit_cells']
    budget = sum(counts.values())
    return order, [counts[name]/budget for name in order], budget


# Crystals per chunk when distances are computed. Sized so the float64 temporary stays around a
# hundred megabytes at the deepest pool this driver generates.
DISTANCE_CHUNK = 64


# The difficulty grid. DWMM: the mix must be fitted against patterns the indexer will actually
# meet, not against one condition, so a run draws a cell per crystal rather than applying one
# bundle to all of them.
#
# Three contaminants IS the second phase, on DWMM's instruction: three lines from ONE real partner
# cell, mutually consistent with some other lattice. Independently placed lines are easier to
# reject than real ones (`ErrorAdder.add_second_phase`), so taking the top of the axis as the
# correlated mechanism keeps the hardest case honest rather than optimistic.
#
# Peak positional error stays at the nominal 1x across every cell. It is not a third axis here
# because it was measured not to matter -- 5-7 % on the fitted mix, nil on tP, and structurally
# invisible to the score (P-F-119) -- and varying it would dilute the two that do.
GRID_BUNDLE = 'grid'
CONTAMINANT_LEVELS = (0, 1, 2, 3)
DROPOUT_LEVELS = (0, 2, 4, 6)


def grid_condition(n_contaminants, n_dropout):
    """One cell of the difficulty grid, as a Condition the harness can synthesise."""
    if n_contaminants not in CONTAMINANT_LEVELS or n_dropout not in DROPOUT_LEVELS:
        raise ValueError(f'({n_contaminants}, {n_dropout}) is not a cell of the grid')
    correlated = 3 if n_contaminants == 3 else 0
    placed = 0 if n_contaminants == 3 else n_contaminants
    return BenchmarkConditions.Condition(
        key=f'grid_cont{n_contaminants}_drop{n_dropout}',
        label=f'G{n_contaminants}{n_dropout}',
        axis='difficulty_grid',
        description=(f'{n_contaminants} contaminant lines '
                     f'({"one real partner phase" if correlated else "independently placed"}) '
                     f'and {n_dropout} dropped peaks, at nominal peak error'),
        error_multiplier=1.0,
        n_contaminants=placed,
        n_dropout=n_dropout,
        second_phase_lines=correlated,
        is_hard=bool(n_contaminants or n_dropout),
        )


# Partner phases sampled per lattice for the second-phase bundle. The harness builds its pool
# from the crystals an arm drew, a few hundred a lattice; this matches that scale rather than
# pooling every training crystal, which would be half a million arrays held for the whole run.
SECOND_PHASE_POOL_PER_LATTICE = 300


def needs_second_phase_pool(bundle):
    """Whether this bundle will ever ask for a partner phase, so one place decides it."""
    if bundle == GRID_BUNDLE:
        # the top of the contaminant axis is the second phase, so the grid always needs one
        return any(grid_condition(level, 0).second_phase_lines > 0
                   for level in CONTAMINANT_LEVELS)
    return BenchmarkConditions.BY_KEY[bundle].second_phase_lines > 0


def load_second_phase_pool(dataset_directory, seed):
    """Candidate contaminating phases, drawn from every lattice.

    The `second_phase` bundle adds lines from a real partner cell, and real contamination is not
    lattice-matched -- so the partner comes from the whole set, exactly as `BenchmarkRuns` builds
    it, not from the lattice being fitted. Training crystals only, for the same reason
    `load_entries` uses them: a partner taken from the benchmark half would put benchmark data
    into a setting that is later reported against the benchmark.
    """
    frames = []
    for bravais_lattice in BRAVAIS_LATTICES:
        path = Path(dataset_directory)/f'dataset_{bravais_lattice}.parquet'
        if not path.is_file():
            continue
        data = pd.read_parquet(path, columns=['identifier', 'train', f'q2_{BROADENING_TAG}'])
        data = data.loc[data['train']].sort_values('identifier', kind='stable',
                                                   ignore_index=True)
        if data.shape[0] > SECOND_PHASE_POOL_PER_LATTICE:
            rng = np.random.default_rng(derived_seed(f'phase2:{bravais_lattice}', seed))
            data = data.iloc[np.sort(rng.choice(data.shape[0],
                                                size=SECOND_PHASE_POOL_PER_LATTICE,
                                                replace=False))]
        frames.append(data)
    if not frames:
        raise SystemExit(
            f'no source datasets under {dataset_directory}, so no partner phases can be drawn '
            f'for the second-phase bundle.'
            )
    return BenchmarkPatterns.build_second_phase_pool(
        pd.concat(frames, ignore_index=True))


def load_entries(bravais_lattice, n_entries, dataset_directory, seed, bundle,
                 second_phase_pool=None):
    """Training crystals with enough peaks, synthesised under one condition bundle.

    Training crystals, not benchmark ones: the mix is a setting being selected, and selecting it
    on the population it is later reported against would be reading the answer first. That is why
    this does not call `BenchmarkPatterns.sample_entries`, which selects the benchmark half.

    The synthesis itself IS the harness's, so a pattern here is the same object a benchmark arm
    would index -- same dropout rule, same error model, same contaminant placement.

    `bundle` is one of the harness's named bundles, or GRID_BUNDLE, which draws a difficulty per
    crystal instead: contaminants from CONTAMINANT_LEVELS and dropped peaks from DROPOUT_LEVELS,
    independently and uniformly, so all sixteen cells appear in one run. The draw is keyed on the
    crystal's identifier, so which difficulty a crystal gets does not depend on how many crystals
    were asked for or on what order they came in.

    Every row carries the difficulty that was DELIVERED, which is not always the one drawn -- see
    the degradation below -- so the fit can be re-cut by difficulty without regenerating.
    """
    path = Path(dataset_directory)/f'dataset_{bravais_lattice}.parquet'
    if not path.is_file():
        raise SystemExit(
            f'no source dataset for {bravais_lattice} at {path}. Point --dataset-directory at a '
            f'tree that has it.'
            )
    fixed = None if bundle == GRID_BUNDLE else BenchmarkConditions.BY_KEY[bundle]
    n_peaks = N_PEAKS.get(bravais_lattice, DEFAULT_N_PEAKS)
    data = pd.read_parquet(path, columns=[
        'identifier', 'train', f'q2_{BROADENING_TAG}', 'reindexed_xnn',
        # the labeller works from the conventional cell, so both forms of the truth come back
        'reindexed_unit_cell'])
    data = data.loc[data['train']]
    peaks = data[f'q2_{BROADENING_TAG}']
    data = data.loc[peaks.apply(lambda q2: np.count_nonzero(q2) >= n_peaks)]
    data = data.sort_values('identifier', kind='stable', ignore_index=True)
    # Drawn wider than asked, because a condition loses a few crystals it cannot be applied to.
    draw = min(data.shape[0], int(1.3*n_entries) + 8)
    if data.shape[0] > draw:
        rng = np.random.default_rng(derived_seed(f'sample:{bravais_lattice}', seed))
        data = data.iloc[np.sort(rng.choice(data.shape[0], size=draw, replace=False))]
    data = data.reset_index(drop=True)

    rows = []
    refused = 0
    degraded = 0
    for _, entry in data.iterrows():
        if fixed is not None:
            attempts = [fixed]
        else:
            draw_rng = np.random.default_rng(derived_seed(
                f'difficulty:{bravais_lattice}:{entry["identifier"]}', seed))
            n_contaminants = int(draw_rng.choice(CONTAMINANT_LEVELS))
            n_dropout = int(draw_rng.choice(DROPOUT_LEVELS))
            # A pattern whose observed range is narrow cannot take every contaminant asked for,
            # and add_contaminants raises rather than placing one on top of a peak. Dropping the
            # crystal would bias the sample: the ones that refuse are systematically the ones
            # with a narrow range, so they would be under-represented at high contamination and
            # over-represented at low. The draw steps down instead, and what was delivered is
            # recorded on the row.
            attempts = [grid_condition(level, n_dropout)
                        for level in range(n_contaminants, -1, -1)]
        pattern = None
        for index, condition in enumerate(attempts):
            try:
                pattern = BenchmarkPatterns.prepare_peak_list(
                    entry, condition, seed, n_peaks=n_peaks,
                    second_phase_pool=second_phase_pool)
            except ContaminantPlacementError:
                continue
            if index:
                degraded += 1
            break
        if pattern is None:
            refused += 1
            continue
        q2 = np.asarray(pattern.q2_obs, dtype=float)
        if np.count_nonzero(q2) < n_peaks:
            refused += 1
            continue
        rows.append({'identifier': entry['identifier'],
                     'reindexed_xnn': entry['reindexed_xnn'],
                     'reindexed_unit_cell': entry['reindexed_unit_cell'],
                     'q2': q2[:n_peaks],
                     # the second phase counts as contamination: DWMM's convention, and the two
                     # mechanisms deliver extra lines that the pattern cannot tell apart
                     'n_contaminants': int(pattern.n_contaminants_achieved
                                           + pattern.n_second_phase_achieved),
                     'n_dropout': int(pattern.n_dropout_achieved)})
        if len(rows) == n_entries:
            break
    if degraded:
        print(f'{bravais_lattice}: {degraded} of {len(rows)} crystals took fewer contaminants '
              f'than drawn; the delivered count is on the row', flush=True)
    if len(rows) < n_entries:
        # Ten of the fourteen lattices hold fewer than 10 000 usable training crystals -- cF has
        # 554 -- so a large --n-entries silently becomes 'all of them' on most of the run. Said
        # loudly here because the count otherwise appears as one line among thousands, and a
        # number of crystals is the number every later error bar is read against.
        print(f'WARNING: {bravais_lattice}: asked for {n_entries} crystals, only {len(rows)} '
              f'are available ({data.shape[0]} drawn, {refused} refused by the condition). '
              f'Every result for this lattice rests on {len(rows)}.', flush=True)
    return pd.DataFrame(rows), refused


def commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def generate(args):
    """Draw every generator's pool for every chosen lattice and write them to disk."""
    from mpi4py import MPI

    comm = check_mpi_world(MPI.COMM_WORLD, MPI.COMM_WORLD.Get_size())
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    split_comm = comm.Split(color=rank, key=rank)
    out = Path(args.pools)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    models_directory = Path(args.models_directory) if args.models_directory else _resolve_models_dir()
    package_root = Path(mlindex.__path__[0]).parent
    manifest = {
        'n_entries': args.n_entries, 'budget_scale': args.budget_scale, 'seed': args.seed,
        'broadening_tag': BROADENING_TAG, 'commit': commit(), 'n_ranks': n_ranks,
        'platform': platform.platform(), 'machine': platform.machine(),
        'numpy': np.__version__, 'models_directory': str(models_directory),
        'bundle': args.bundle,
        'population': (
            f'training crystals, a difficulty drawn per crystal from '
            f'{CONTAMINANT_LEVELS} contaminants x {DROPOUT_LEVELS} dropped peaks at nominal '
            f'peak error; three contaminants is the second phase'
            if args.bundle == GRID_BUNDLE else
            f'training crystals, condition bundle {args.bundle!r} '
            f'({BenchmarkConditions.BY_KEY[args.bundle].description})'),
        'lattices': {},
        }

    # Built once and across every lattice, not per lattice, because a partner phase is not
    # lattice-matched and rebuilding it per lattice would make the partner depend on which
    # lattices a run happened to ask for.
    second_phase_pool = None
    if rank == 0 and needs_second_phase_pool(args.bundle):
        second_phase_pool = load_second_phase_pool(args.dataset_directory, args.seed)

    for bravais_lattice in args.bravais_lattices:
        factory = getattr(
            UtilitiesOptimizer, FACTORY_OF_SYSTEM[BL_TO_LATTICE_SYSTEM[bravais_lattice]])
        optimizer = factory(bravais_lattice, BROADENING_TAG, 1, split_comm,
                            project_path=package_root, seed=args.seed,
                            models_directory=models_directory)
        names, fractions, budget = shipped_mix_and_budget(optimizer)
        per_generator = int(round(args.budget_scale*budget))

        if rank == 0:
            entries, refused = load_entries(bravais_lattice, args.n_entries,
                                            args.dataset_directory, args.seed, args.bundle,
                                            second_phase_pool=second_phase_pool)
            print(f'{bravais_lattice}: {len(entries)} crystals under {args.bundle!r} '
                  f'({refused} refused), {per_generator} candidates a generator, '
                  f'shipped budget {budget}', flush=True)
            for other in range(1, n_ranks):
                comm.send(entries.iloc[other::n_ranks], dest=other)
            mine = entries.iloc[0::n_ranks]
        else:
            mine = comm.recv(source=0)

        # The generators are seeded per crystal so that a subset of a run regenerates identically
        # and the rank count cannot change an answer.
        positions = []
        truths = []
        for index in range(len(mine)):
            entry = mine.iloc[index]
            rng = np.random.default_rng(derived_seed(
                f'pool:{bravais_lattice}:{entry["identifier"]}', args.seed))
            xnn, xnn_true, generated_names = generate_candidate_pools(
                optimizer, entry, candidates_per_model=per_generator, rng=rng)
            positions.append(xnn.astype(np.float32))
            truths.append(xnn_true)
        payload = {'xnn': positions, 'xnn_true': truths,
                   'identifiers': list(mine['identifier']),
                   'n_contaminants': list(mine['n_contaminants']),
                   'n_dropout': list(mine['n_dropout']),
                   'generator_names': generated_names}

        gathered = comm.gather(payload, root=0)
        del payload, positions, truths
        if rank == 0:
            blocks = [block for part in gathered for block in part['xnn']]
            xnn_true = np.stack([block for part in gathered for block in part['xnn_true']])
            identifiers = [name for part in gathered for name in part['identifiers']]
            # Carried into the pool so the fit can be cut by difficulty without regenerating:
            # what was DELIVERED, which is not always what was drawn.
            n_contaminants = [v for part in gathered for v in part['n_contaminants']]
            n_dropout = [v for part in gathered for v in part['n_dropout']]
            del gathered
            # Filled a crystal at a time and each block dropped as it is copied, so the gathered
            # copy and the stacked one are never both whole. At 10 000 crystals one aP pool is
            # 8.6 GB, and np.stack over the list would need both at once.
            xnn = np.empty((len(blocks),) + blocks[0].shape, dtype=np.float32)
            for index in range(len(blocks)):
                xnn[index] = blocks[index]
                blocks[index] = None
            del blocks
            # Distances at full precision because the score is computed from them; positions at
            # single precision because they are only ever used to count near neighbours, where the
            # radii are thousands of times coarser than float32's resolution here.
            #
            # In crystal-sized chunks: the whole-array form promotes the float32 positions against
            # the float64 truth, and that one temporary is 17 GB for aP at 10 000 crystals.
            distance = np.empty(xnn.shape[:-1], dtype=float)
            for start in range(0, xnn.shape[0], DISTANCE_CHUNK):
                stop = start + DISTANCE_CHUNK
                distance[start:stop] = np.linalg.norm(
                    xnn[start:stop] - xnn_true[start:stop, np.newaxis, np.newaxis, :], axis=-1)
            np.savez_compressed(
                out/f'{bravais_lattice}_pools.npz',
                xnn=xnn, xnn_true=xnn_true, distances=distance,
                identifiers=np.array(identifiers, dtype=object),
                n_contaminants=np.array(n_contaminants, dtype=int),
                n_dropout=np.array(n_dropout, dtype=int),
                generator_names=np.array(generated_names, dtype=object),
                )
            manifest['lattices'][bravais_lattice] = {
                'n_crystals': int(xnn.shape[0]), 'per_generator': per_generator,
                'shipped_budget': budget, 'shipped_mix': dict(zip(names, fractions)),
                'generator_names': generated_names,
                'n_peaks': N_PEAKS.get(bravais_lattice, DEFAULT_N_PEAKS),
                }
            # Rewritten after every lattice, not once at the end, so that a run which dies
            # partway leaves the lattices it finished in a state the fit stage can read.
            (out/'pools_manifest.json').write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
            print(f'{bravais_lattice}: wrote {xnn.shape[0]} crystals', flush=True)
    return 0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def mix_grid(step, n_generators):
    """Every mix on a regular grid over the simplex, as fractions summing to one."""
    n = int(round(1.0/step))
    if n_generators != 3:
        raise ValueError(f'the grid is written for three generators, not {n_generators}')
    return np.array([(a/n, b/n, (n - a - b)/n)
                     for a in range(n + 1) for b in range(n + 1 - a)])


def counts_for_mix(mix, budget):
    """Candidate counts that sum to the budget exactly.

    Rounding each share independently makes the total drift with the mix, and the score rises with
    the number of candidates, so a mix that happened to round up would win on that alone.
    """
    return largest_remainder(mix, budget)


def stack_pools(distance, counts):
    """(n_crystals, sum(counts)): the first counts[g] candidates of each generator, side by side."""
    return np.concatenate(
        [distance[:, :count, index] for index, count in enumerate(counts) if count], axis=1)


def stack_positions(xnn, counts):
    """The same selection, on positions: (n_crystals, sum(counts), n_cell_parameters)."""
    return np.concatenate(
        [xnn[:, :count, index] for index, count in enumerate(counts) if count], axis=1)


def score_crystals(distance, xnn, grid, budget, curve, discount):
    """(n_mixes, n_crystals): what each mix is worth to each crystal. Larger is better.

    The clump weights are recomputed for the candidate subset each mix selects, not once over the
    whole pool. Computed once they would not depend on the mix, the score would be linear in it,
    and its optimum would always be a corner -- which is the answer a score blind to crowding
    gives and the reason this term exists.

    That is also what makes the fit slow: one neighbour search per mix per crystal. Measured on
    this laptop, a 12 000-candidate pool costs 18 ms a crystal a mix, so a 0.02 grid over 10 000
    crystals is 67 hours on one core. This function is the serial unit; `score_every_mix` spreads
    it over crystals, which is the axis the work is independent along.
    """
    scores = np.empty((grid.shape[0], distance.shape[0]))
    for index, mix in enumerate(grid):
        counts = counts_for_mix(mix, budget)
        pool = stack_pools(distance, counts)
        positions = stack_positions(xnn, counts)
        weight = np.stack([clump_weights(positions[crystal], *discount)
                           for crystal in range(positions.shape[0])])
        scores[index] = expected_success_objective(pool, curve[0], curve[1], weight)
    return scores


def _score_chunk(payload):
    """Module-level and taking one picklable argument, because workers are spawned, not forked."""
    return score_crystals(*payload)


def score_every_mix(distance, xnn, grid, budget, curve, discount, nproc):
    """`score_crystals` over every crystal, spread across `nproc` processes.

    `nproc` has no default. A fit that quietly ran on one core would take days at production
    scale and look like a hang, so the caller says how many cores it is giving this.
    """
    n_crystals = distance.shape[0]
    if nproc <= 1 or n_crystals < 2*nproc:
        return score_crystals(distance, xnn, grid, budget, curve, discount)

    def chunks():
        # Built lazily: materialising every chunk first would hold a second copy of the pool,
        # which is gigabytes at production scale.
        for rows in np.array_split(np.arange(n_crystals), nproc):
            yield (distance[rows], xnn[rows], grid, budget, curve, discount)

    with multiprocessing.get_context('spawn').Pool(nproc) as pool:
        parts = list(pool.imap(_score_chunk, chunks()))
    return np.concatenate(parts, axis=1)


def choose_mix(scores, grid, reduction):
    """The mix a reduction picks, the pooled score over the grid, and how it got there.

    'pooled'       one mix for the lattice, against the mean over crystals.
    'per-pattern'  the best mix for each crystal, averaged. This is what the original fit did, by
                   averaging the logits it optimised; averaging the winning shares is the same
                   idea on a grid. Kept so that what it costs can be measured rather than assumed.

    For 'per-pattern' the third return says whether the average describes any pattern at all. A
    per-pattern optimum that hands almost the whole budget to one generator is not evidence for a
    blend, and an average of such optima is a mix no pattern asked for.
    """
    pooled = scores.mean(axis=1)
    if reduction == 'pooled':
        return grid[int(np.argmax(pooled))], pooled, {}
    if reduction == 'per-pattern':
        winners = grid[np.argmax(scores, axis=0)]
        return winners.mean(axis=0), pooled, {
            'share_all_or_nothing': float(np.mean(winners.max(axis=1) > 0.95)),
            'share_on_a_corner': float(np.mean(np.isclose(winners.max(axis=1), 1.0))),
            'winner_spread': float(np.mean(winners.std(axis=0))),
            }
    raise ValueError(f'unknown reduction {reduction!r}')


def screen(pooled, grid, chosen):
    """How decided the answer is, and whether it sits in a corner.

    A score that is flat gives every mix the same value, and `argmax` then returns whichever the
    grid happens to enumerate first. Counting how many mixes come within a hundredth of the whole
    range of the best one says whether there was anything to choose.
    """
    best = float(pooled.max())
    worst = float(pooled.min())
    spread = best - worst
    tied = int(np.count_nonzero(pooled >= best - 0.01*spread)) if spread > 0 else pooled.size
    return {
        'value_best': best,
        'value_worst': worst,
        'spread': spread,
        'n_tied': tied,
        'n_mixes': int(pooled.size),
        'determined': bool(tied < 0.02*pooled.size),
        'on_boundary': bool(np.any(np.isclose(chosen, 0.0)) or np.any(np.isclose(chosen, 1.0))),
        }


def fit(args):
    """Score every mix on the grid, under every reduction asked for."""
    pools = Path(args.pools)
    manifest_path = pools/'pools_manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'no pools manifest at {manifest_path}; run --stage generate first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    for bravais_lattice in args.bravais_lattices:
        path = pools/f'{bravais_lattice}_pools.npz'
        if not path.is_file():
            continue
        data = np.load(path, allow_pickle=True)
        names = [str(name) for name in data['generator_names']]
        distance = np.asarray(data['distances'], dtype=float)
        if not np.all(np.isfinite(distance)):
            raise SystemExit(f'{path} holds non-finite distances; regenerate it')
        if 'xnn' not in data:
            raise SystemExit(
                f'{path} has no candidate positions, so how crowded the pool is cannot be seen '
                f'and the mix cannot be scored. Regenerate it with the current driver.')
        # Left at the single precision they were written in. The generate stage stores them that
        # way on purpose -- they are only ever used to count near neighbours, where the radii are
        # thousands of times coarser -- and widening them here would double a 9 GB array.
        xnn = np.asarray(data['xnn'], dtype=np.float32)
        discount = load_clump_discount(args.clump_discount, bravais_lattice)
        info = manifest['lattices'][bravais_lattice]
        shipped = np.array([info['shipped_mix'][name] for name in names])
        curve = np.vstack(load_curve(args.roc_dir, bravais_lattice))
        grid = mix_grid(args.step, len(names))

        print(f'{bravais_lattice}: scoring {grid.shape[0]} mixes over '
              f'{distance.shape[0]} crystals on {args.nproc} processes', flush=True)

        budget = int(info['shipped_budget'])
        if budget <= distance.shape[1]:
            scores = score_every_mix(distance, xnn, grid, budget, curve, discount, args.nproc)
            # The shipped mix need not sit on the grid, so it is scored as a grid of one rather
            # than by a second copy of the same arithmetic.
            shipped_value = float(np.mean(score_every_mix(
                distance, xnn, shipped[np.newaxis], budget, curve, discount, args.nproc)))
            for reduction in args.reductions:
                for split, rows_of in _splits(distance.shape[0], args.split_seed):
                    chosen, pooled, how = choose_mix(scores[:, rows_of], grid, reduction)
                    measured = screen(pooled, grid, chosen)
                    measured.update(how)
                    rows.append(dict(
                        bravais_lattice=bravais_lattice,
                        bundle=manifest.get('bundle', 'unknown'),
                        reduction=reduction, split=split, budget=budget,
                        n_crystals=int(rows_of.size),
                        **{f'best_{name}': float(value)
                           for name, value in zip(names, chosen)},
                        **{f'shipped_{name}': float(value)
                           for name, value in zip(names, shipped)},
                        value_shipped=shipped_value, **measured))
        # Written after every lattice rather than once at the end: the low-symmetry lattices are
        # the slow ones and they come last, so a run that dies on aP would otherwise take the
        # thirteen finished lattices with it.
        pd.DataFrame(rows).to_csv(out/'ensemble_mix.csv', index=False)
    return 0


def _splits(n_crystals, seed):
    """The whole set, then two disjoint halves, so a mix that does not reproduce is visible."""
    everything = np.arange(n_crystals)
    shuffled = np.random.default_rng(seed).permutation(everything)
    half = n_crystals//2
    return [('all', everything),
            ('half-a', np.sort(shuffled[:half])),
            ('half-b', np.sort(shuffled[half:]))]


# ---------------------------------------------------------------------------
# Redistribution
# ---------------------------------------------------------------------------

# The settings searched, around each lattice's shipped pair: these max_neighbors values plus the
# shipped one and two and four times it, against the shipped radius times each factor. Wide enough
# that an answer on the edge of the grid says the grid was too small, which the table reports.
NEIGHBOR_VALUES = (2, 5, 10, 20)
RADIUS_FACTORS = (0.25, 0.5, 1, 2, 4, 8)


def redistribution_grid(max_neighbors, neighbor_radius):
    """(n_settings, 2) of (max_neighbors, neighbor_radius) around one lattice's shipped pair."""
    neighbors = sorted({*NEIGHBOR_VALUES, max_neighbors, 2*max_neighbors, 4*max_neighbors})
    return np.array([(n, factor*neighbor_radius) for n in neighbors for factor in RADIUS_FACTORS])


def score_redistribution(xnn, xnn_true, identifiers, bravais_lattice, counts, grid, curve,
                         discount, repeats, seed):
    """(1 + n_settings, n_crystals): the pool as generated, then after each setting's redistribution.

    Redistribution only moves candidates apart, so the clump weights are what can see it; they are
    recomputed on every redistributed pool. It draws random numbers, so each setting is the mean
    over `repeats` draws, and every setting uses the same `repeats` seeds for a crystal, so two
    settings differ by what they do and not by which draws they got.
    """
    positions = stack_positions(xnn, counts).astype(float)
    scores = np.empty((1 + grid.shape[0], positions.shape[0]))

    def value(cloud, truth):
        distance = np.linalg.norm(cloud - truth, axis=1)
        return expected_success_objective(
            distance, curve[0], curve[1], clump_weights(cloud, *discount))

    for crystal in range(positions.shape[0]):
        cloud, truth = positions[crystal], xnn_true[crystal]
        scores[0, crystal] = value(cloud, truth)
        for index, (max_neighbors, neighbor_radius) in enumerate(grid):
            total = 0.0
            for repeat in range(repeats):
                rng = np.random.default_rng(derived_seed(
                    f'redistribution:{bravais_lattice}:{identifiers[crystal]}:{repeat}', seed))
                moved = redistribute_xnn(
                    cloud, bravais_lattice, int(max_neighbors), float(neighbor_radius), rng,
                    minimum_unit_cell=UtilitiesOptimizer.MINIMUM_UNIT_CELL,
                    maximum_unit_cell=UtilitiesOptimizer.MAXIMUM_UNIT_CELL)
                total += value(moved, truth)
            scores[1 + index, crystal] = total/repeats
    return scores


def _score_redistribution_chunk(payload):
    """Module-level and taking one picklable argument, because workers are spawned, not forked."""
    return score_redistribution(*payload)


def score_redistribution_parallel(xnn, xnn_true, identifiers, bravais_lattice, counts, grid,
                                  curve, discount, repeats, seed, nproc):
    """`score_redistribution` over every crystal, spread across `nproc` processes."""
    n_crystals = xnn.shape[0]
    if nproc <= 1 or n_crystals < 2*nproc:
        return score_redistribution(xnn, xnn_true, identifiers, bravais_lattice, counts, grid,
                                    curve, discount, repeats, seed)

    def chunks():
        for rows in np.array_split(np.arange(n_crystals), nproc):
            yield (xnn[rows], xnn_true[rows], [identifiers[row] for row in rows],
                   bravais_lattice, counts, grid, curve, discount, repeats, seed)

    with multiprocessing.get_context('spawn').Pool(nproc) as pool:
        parts = list(pool.imap(_score_redistribution_chunk, chunks()))
    return np.concatenate(parts, axis=1)


def choose_redistribution(scores, grid, reduction):
    """The (max_neighbors, neighbor_radius) a reduction picks, and how decided it was.

    `scores` is what `score_redistribution` returns; its first row, the pool left as generated, is
    reported but never chosen, because switching the step off is a separate question.

    'pooled'       the setting with the best mean over crystals.
    'per-pattern'  each crystal's best setting, averaged, as the generator fractions were chosen.
                   A crystal on which every setting scores the same -- nothing it holds is crowded
                   enough to move -- has no best setting, and is left out of the average rather
                   than voting for whichever setting the grid lists first. How many were left out
                   is reported.
    """
    settings = scores[1:]
    pooled = settings.mean(axis=1)
    indifferent = np.ptp(settings, axis=0) == 0
    report = {'n_indifferent': int(indifferent.sum()),
              'value_pooled_best': float(pooled.max()),
              'n_tied': int(np.count_nonzero(pooled >= pooled.max() - 0.01*np.ptp(pooled)))
              if np.ptp(pooled) > 0 else int(pooled.size),
              'n_settings': int(pooled.size)}
    if reduction == 'pooled':
        chosen = grid[int(np.argmax(pooled))]
    elif reduction == 'per-pattern':
        if indifferent.all():
            return None, report
        winners = grid[np.argmax(settings[:, ~indifferent], axis=0)]
        chosen = winners.mean(axis=0)
    else:
        raise ValueError(f'unknown reduction {reduction!r}')
    report['at_grid_edge'] = bool(
        np.isclose(chosen[0], grid[:, 0].min()) or np.isclose(chosen[0], grid[:, 0].max())
        or np.isclose(chosen[1], grid[:, 1].min()) or np.isclose(chosen[1], grid[:, 1].max()))
    return (int(round(chosen[0])), float(chosen[1])), report


def fit_redistribution(args):
    """Re-derive every lattice's redistribution constants against the ensemble score.

    Each lattice's per-crystal scores are written to their own file, carrying what produced them,
    and the table is rebuilt from every such file after each lattice. So a second run over some
    lattices -- after a first one ran out of time -- adds to the table rather than replacing it.
    """
    pools = Path(args.pools)
    manifest_path = pools/'pools_manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'no pools manifest at {manifest_path}; run --stage generate first')
    pools_commit = json.loads(manifest_path.read_text(encoding='utf-8')).get('commit')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for bravais_lattice in args.bravais_lattices:
        if bravais_lattice in CUBIC:
            # No clump discount can be measured on cubic, so every weight is one and the score
            # cannot see what redistribution does; it moves nothing there at the shipped constants.
            print(f'{bravais_lattice}: skipped, no measured clump discount to see redistribution '
                  f'with', flush=True)
            continue
        path = pools/f'{bravais_lattice}_pools.npz'
        if not path.is_file():
            continue
        data = np.load(path, allow_pickle=True)
        names = [str(name) for name in data['generator_names']]
        chosen_rows = np.sort(np.random.default_rng(args.split_seed).permutation(
            data['xnn_true'].shape[0])[:args.n_crystals])
        xnn = np.asarray(data['xnn'][chosen_rows], dtype=np.float32)
        xnn_true = np.asarray(data['xnn_true'][chosen_rows], dtype=float)
        identifiers = [str(name) for name in data['identifiers'][chosen_rows]]

        ensemble = UtilitiesOptimizer.ENSEMBLE[bravais_lattice]
        fractions = UtilitiesOptimizer.lattice_fractions(bravais_lattice, {})
        budget = UtilitiesOptimizer.lattice_budget(bravais_lattice, 1, {})
        counts = counts_for_mix([fractions[name] for name in names], budget)
        if np.any(counts > xnn.shape[1]):
            raise SystemExit(f'{path} holds {xnn.shape[1]} candidates a generator, fewer than '
                             f'the {counts.max()} the shipped fractions ask of one')
        shipped = (ensemble['max_neighbors'], ensemble['neighbor_radius'])
        grid = redistribution_grid(*shipped)
        curve = np.vstack(load_curve(args.roc_dir, bravais_lattice))
        discount = load_clump_discount(args.clump_discount, bravais_lattice)
        print(f'{bravais_lattice}: {grid.shape[0]} settings x {args.repeats} draws over '
              f'{xnn.shape[0]} crystals on {args.nproc} processes', flush=True)

        scores = score_redistribution_parallel(
            xnn, xnn_true, identifiers, bravais_lattice, counts, grid, curve, discount,
            args.repeats, args.seed, args.nproc)
        # What produced these scores, kept with them so that it survives any later run.
        provenance = {
            'commit': commit(), 'pools': str(pools), 'pools_commit': pools_commit,
            'n_crystals': int(xnn.shape[0]), 'repeats': args.repeats, 'seed': args.seed,
            'split_seed': args.split_seed, 'budget': budget, 'fractions': fractions,
            'shipped_max_neighbors': shipped[0], 'shipped_neighbor_radius': shipped[1],
            'platform': platform.platform(), 'machine': platform.machine(),
            'numpy': np.__version__,
            }
        np.savez_compressed(out/f'{bravais_lattice}_redistribution_scores.npz', scores=scores,
                            grid=grid, identifiers=np.array(identifiers, dtype=object),
                            provenance=json.dumps(provenance, sort_keys=True))
        write_redistribution_table(out, args.reductions, args.split_seed)
        print(f'{bravais_lattice}: done', flush=True)
    return 0


def write_redistribution_table(out, reductions, split_seed):
    """Rebuild redistribution.csv from every lattice's scores file in `out`."""
    out = Path(out)
    rows = []
    for path in sorted(out.glob('*_redistribution_scores.npz')):
        data = np.load(path, allow_pickle=True)
        if 'provenance' not in data:
            raise SystemExit(f'{path} does not say what produced it; regenerate it')
        provenance = json.loads(str(data['provenance']))
        bravais_lattice = path.name.split('_')[0]
        scores, grid = data['scores'], data['grid']
        shipped = (provenance['shipped_max_neighbors'], provenance['shipped_neighbor_radius'])
        shipped_index = 1 + int(np.flatnonzero(
            (grid[:, 0] == shipped[0]) & np.isclose(grid[:, 1], shipped[1]))[0])
        for reduction in reductions:
            for split, rows_of in _splits(scores.shape[1], split_seed):
                constants, report = choose_redistribution(scores[:, rows_of], grid, reduction)
                rows.append(dict(
                    bravais_lattice=bravais_lattice, reduction=reduction, split=split,
                    n_crystals=int(rows_of.size), budget=provenance['budget'],
                    repeats=provenance['repeats'], commit=provenance['commit'],
                    best_max_neighbors=None if constants is None else constants[0],
                    best_neighbor_radius=None if constants is None else constants[1],
                    shipped_max_neighbors=shipped[0], shipped_neighbor_radius=shipped[1],
                    value_off=float(scores[0, rows_of].mean()),
                    value_shipped=float(scores[shipped_index, rows_of].mean()),
                    **report))
    pd.DataFrame(rows).to_csv(out/'redistribution.csv', index=False)
    return rows

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m mlindex.scripts.run_ensemble_refine',
        description='Measure how much of the candidate budget each generator should get.')
    parser.add_argument('--stage', default='all',
                        choices=('all', 'generate', 'fit', 'redistribution',
                                 'redistribution-table'),
                        help='generate writes candidate pools (MPI, slow); fit scores mixes '
                             'against them (no MPI); redistribution re-derives each lattice\'s '
                             'redistribution constants against the same pools and score, at '
                             'the generator fractions in ENSEMBLE; redistribution-table '
                             'rebuilds that stage\'s table from the scores files in --out-dir '
                             'without scoring anything. Default: all, which is generate then '
                             'fit.')
    parser.add_argument('--bravais-lattices', default=','.join(BRAVAIS_LATTICES), metavar='A,B',
                        help='Comma-separated. Default: all fourteen.')
    parser.add_argument('--pools', default=None, metavar='PATH',
                        help='Directory the candidate pools are written to and read from. '
                             'Needed by every stage except redistribution-table.')
    parser.add_argument('--out-dir', default=None, metavar='PATH',
                        help='Where the fit writes its table. Required for --stage fit and all.')
    parser.add_argument('--clump-discount', default=None, metavar='PATH',
                        help='Directory holding the measured clump discounts, one file a '
                             'lattice. They say what a candidate is worth when it is not alone, '
                             'and without them a mix cannot be scored. Run output, like the '
                             'curves. Required to fit.')
    parser.add_argument('--roc-dir', default=None, metavar='PATH',
                        help='Directory holding the measured convergence curves. They are run '
                             'output and are not shipped with the package. Required to fit.')
    generate_group = parser.add_argument_group('generate')
    generate_group.add_argument('--dataset-directory', default=None, metavar='PATH',
                                help='Directory holding dataset_{lattice}.parquet.')
    generate_group.add_argument('--n-entries', type=int, default=300, metavar='N',
                                help='Source crystals per lattice (default: 300).')
    generate_group.add_argument('--budget-scale', type=float, default=2.0, metavar='X',
                                help='Candidates per generator, as a multiple of the shipped '
                                     'budget (default: 2.0, so any mix can be scored at twice it).')
    generate_group.add_argument('--models-directory', default=None, metavar='PATH',
                                help='Model tree. Default: the usual resolution order.')
    generate_group.add_argument('--bundle', default='nominal',
                                choices=tuple(BenchmarkConditions.BY_KEY) + (GRID_BUNDLE,),
                                help='Conditions the patterns are synthesised under: one of '
                                     'the benchmark\'s named bundles (default: nominal), or '
                                     '\'grid\', which draws a difficulty per crystal -- 0, 1, 2 '
                                     'or 3 contaminants against 0, 2, 4 or 6 dropped peaks, all '
                                     'sixteen cells in one run, at nominal peak error. Three '
                                     'contaminants is the second phase. The delivered difficulty '
                                     'is stored per crystal so the fit can be cut by it.')
    generate_group.add_argument('--seed', type=int, default=12345, metavar='N')
    fit_group = parser.add_argument_group('fit')
    fit_group.add_argument('--reduction', action='append', dest='reductions', default=None,
                           choices=('pooled', 'per-pattern'),
                           help='How per-crystal scores become one mix. Repeat for several. '
                                'Default: pooled.')
    fit_group.add_argument('--step', type=float, default=0.02, metavar='X',
                           help='Grid spacing on the simplex (default: 0.02, so 1326 mixes).')
    fit_group.add_argument('--split-seed', type=int, default=12345, metavar='N',
                           help='Seed for the half-and-half stability check (default: 12345).')
    fit_group.add_argument('--nproc', type=int, default=1, metavar='N',
                           help='Processes the scoring is spread over, across crystals '
                                '(default: 1). The cost is one neighbour search per crystal per '
                                'mix; at 10000 crystals and the default grid that is tens of '
                                'hours on one core, so give this the cores of the node.')
    redistribution_group = parser.add_argument_group('redistribution')
    redistribution_group.add_argument('--n-crystals', type=int, default=1000, metavar='N',
                                      help='Crystals a lattice, drawn with --split-seed from the '
                                           'pool (default: 1000). The cost is about 130 '
                                           'redistributions of one pool per crystal.')
    redistribution_group.add_argument('--repeats', type=int, default=3, metavar='N',
                                      help='Draws averaged per setting, since redistribution is '
                                           'random (default: 3).')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.bravais_lattices = [name for name in args.bravais_lattices.split(',') if name]
    unknown = [name for name in args.bravais_lattices if name not in BRAVAIS_LATTICES]
    if unknown:
        raise SystemExit(f'not Bravais lattices this package knows: {", ".join(unknown)}')
    if args.pools is None and args.stage != 'redistribution-table':
        raise SystemExit(f'--stage {args.stage} needs --pools')

    if args.stage in ('all', 'generate'):
        if args.dataset_directory is None:
            raise SystemExit('--stage generate needs --dataset-directory')
        generate(args)
    if args.stage in ('all', 'fit'):
        args.reductions = args.reductions or ['pooled']
        if args.roc_dir is None:
            raise SystemExit('--stage fit needs --roc-dir')
        if args.out_dir is None:
            raise SystemExit('--stage fit needs --out-dir')
        if args.clump_discount is None:
            raise SystemExit(
                '--stage fit needs --clump-discount. Candidates in a real pool are not '
                'independent, and a score that assumes they are always names a corner.')
        fit(args)
    if args.stage == 'redistribution':
        missing = [flag for flag, value in (('--roc-dir', args.roc_dir),
                                            ('--out-dir', args.out_dir),
                                            ('--clump-discount', args.clump_discount))
                   if value is None]
        if missing:
            raise SystemExit(f'--stage redistribution needs {", ".join(missing)}. Without the '
                             f'clump discount redistribution is invisible to the score.')
        args.reductions = args.reductions or ['per-pattern', 'pooled']
        fit_redistribution(args)
    if args.stage == 'redistribution-table':
        if args.out_dir is None:
            raise SystemExit('--stage redistribution-table needs --out-dir')
        write_redistribution_table(args.out_dir, args.reductions or ['per-pattern', 'pooled'],
                                   args.split_seed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
