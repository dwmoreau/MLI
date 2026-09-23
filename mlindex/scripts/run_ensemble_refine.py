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
from mlindex.utilities.ClumpDiscount import clump_weights, load_clump_discount
from mlindex.utilities.ConvergenceCurve import load_curve
from mlindex.utilities.Digests import derived_seed
from mlindex.utilities.EnsembleObjective import expected_success_objective
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


# Partner phases sampled per lattice for the second-phase bundle. The harness builds its pool
# from the crystals an arm drew, a few hundred a lattice; this matches that scale rather than
# pooling every training crystal, which would be half a million arrays held for the whole run.
SECOND_PHASE_POOL_PER_LATTICE = 300


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
    would index -- same dropout rule, same error model, same contaminant placement. A crystal the
    condition cannot be applied to is skipped, as the harness skips it, rather than failing the run.
    """
    path = Path(dataset_directory)/f'dataset_{bravais_lattice}.parquet'
    if not path.is_file():
        raise SystemExit(
            f'no source dataset for {bravais_lattice} at {path}. Point --dataset-directory at a '
            f'tree that has it.'
            )
    condition = BenchmarkConditions.BY_KEY[bundle]
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
    for _, entry in data.iterrows():
        try:
            pattern = BenchmarkPatterns.prepare_peak_list(
                entry, condition, seed, n_peaks=n_peaks,
                second_phase_pool=second_phase_pool)
        except ContaminantPlacementError:
            refused += 1
            continue
        q2 = np.asarray(pattern.q2_obs, dtype=float)
        if np.count_nonzero(q2) < n_peaks:
            refused += 1
            continue
        rows.append({'identifier': entry['identifier'],
                     'reindexed_xnn': entry['reindexed_xnn'],
                     'reindexed_unit_cell': entry['reindexed_unit_cell'],
                     'q2': q2[:n_peaks]})
        if len(rows) == n_entries:
            break
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
        'population': f'training crystals, condition bundle {args.bundle!r} '
                      f'({BenchmarkConditions.BY_KEY[args.bundle].description})',
        'lattices': {},
        }

    # Built once and across every lattice, not per lattice, because a partner phase is not
    # lattice-matched and rebuilding it per lattice would make the partner depend on which
    # lattices a run happened to ask for.
    second_phase_pool = None
    if rank == 0 and BenchmarkConditions.BY_KEY[args.bundle].second_phase_lines > 0:
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
                   'identifiers': list(mine['identifier']), 'generator_names': generated_names}

        gathered = comm.gather(payload, root=0)
        del payload, positions, truths
        if rank == 0:
            blocks = [block for part in gathered for block in part['xnn']]
            xnn_true = np.stack([block for part in gathered for block in part['xnn_true']])
            identifiers = [name for part in gathered for name in part['identifiers']]
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
    """Candidate counts that sum to the budget exactly, by largest remainder.

    Rounding each share independently makes the total drift with the mix, and the score rises with
    the number of candidates, so a mix that happened to round up would win on that alone.
    """
    exact = np.asarray(mix, dtype=float)*budget
    counts = np.floor(exact).astype(int)
    short = budget - int(counts.sum())
    if short:
        counts[np.argsort(-(exact - counts))[:short]] += 1
    return counts


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
# Command line
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m mlindex.scripts.run_ensemble_refine',
        description='Measure how much of the candidate budget each generator should get.')
    parser.add_argument('--stage', default='all', choices=('all', 'generate', 'fit'),
                        help='generate writes candidate pools (MPI, slow); fit scores mixes '
                             'against them (no MPI, seconds). Default: all.')
    parser.add_argument('--bravais-lattices', default=','.join(BRAVAIS_LATTICES), metavar='A,B',
                        help='Comma-separated. Default: all fourteen.')
    parser.add_argument('--pools', required=True, metavar='PATH',
                        help='Directory the candidate pools are written to and read from.')
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
                                choices=tuple(BenchmarkConditions.BY_KEY),
                                help='Condition bundle the patterns are synthesised under, from '
                                     'the benchmark\'s own set (default: nominal). This is how '
                                     'the mix is tested for dependence on peak error, '
                                     'contaminants and dropout.')
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.bravais_lattices = [name for name in args.bravais_lattices.split(',') if name]
    unknown = [name for name in args.bravais_lattices if name not in BRAVAIS_LATTICES]
    if unknown:
        raise SystemExit(f'not Bravais lattices this package knows: {", ".join(unknown)}')
    args.reductions = args.reductions or ['pooled']

    if args.stage in ('all', 'generate'):
        if args.dataset_directory is None:
            raise SystemExit('--stage generate needs --dataset-directory')
        generate(args)
    if args.stage in ('all', 'fit'):
        if args.roc_dir is None:
            raise SystemExit('--stage fit needs --roc-dir')
        if args.out_dir is None:
            raise SystemExit('--stage fit needs --out-dir')
        if args.clump_discount is None:
            raise SystemExit(
                '--stage fit needs --clump-discount. Candidates in a real pool are not '
                'independent, and a score that assumes they are always names a corner.')
        fit(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
