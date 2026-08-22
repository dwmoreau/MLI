"""S06b -- the reproducibility floor, measured rather than quoted.

Every gate in this project is judged against a "~10% reproducibility floor". F-141 establishes
that this number was never measured: it is `0.10 x baseline operating point`, i.e. Shirley
1980's remark that *"slightly different refinement conditions may well yield figures of merit
differing by more than 10 percent for the same solution"* applied to a *fraction of entries*
rather than to a figure of merit. Those are different quantities and nothing connects them.

That matters because the floor is what refused S11's block C (+1.05 pp against 6.5) and what
makes F-136's central claim -- a perfect re-scorer gains 4.93 pp, so no figure of merit could
ever pass -- come out the way it does. If the induced spread in a *ranking metric* is smaller
than 4.93 pp, that conclusion inverts.

So this measures two things over the same ensemble and reports them side by side:

  value floor   Shirley's quantity. Per merit, the relative spread of the merit's own value
                across independent re-refinements of the same solution. The `stability`
                column S06's leaderboard left empty.
  metric floor  What a gate should actually be read against. The spread of the reported
                operating point and top-10 across independent re-refinements of the whole
                evaluated pool, at the size the project reports on.

Neither is derived from the other; that is the point.

    python mlindex/scripts/run_fom_floor.py --stage diagnose
    python mlindex/scripts/run_fom_floor.py --stage ensemble --n-entries 15 --radius-sweep

The floor itself is measured by `submit_fom_floor_arms.sh` and `run_fom_floor_report.py`; this
file holds the diagnosis that decided how, and the displacement study that ruled the obvious
operator out.

**The ensemble operator, and why it is not "perturb and refine back".** The obvious reading of
the brief -- perturb a stored cell, refine it back to convergence, and watch the merit move --
assumes the stored cell is a fixed point of the refinement. It is not, and F-142's attempt to
find one was measuring the wrong thing (see `--stage diagnose`, and F-147/F-148). The stored
cell is the arg-max over M20 of a hundred subsampled iterations, followed by at most one masked
Gauss-Newton step accepted only if M20 improved (`Candidates.iteration_worker_common`,
`Candidates.refine_cell`). Nothing in the pipeline drives a cell to the optimum of a
least-squares objective, so there is no fixed point for a perturbation to return to, and any
"refine back" operator measures the disagreement between M20-argmax selection and least squares
rather than a reproducibility floor.

The `ensemble` stage below is the closest stand-in that stays inside the frozen pool: displace
each stored cell, replay the production refinement loop with a fresh seed, and rescore. It was
built to be the measurement and is kept as a *diagnostic*, because its answer depends entirely
on the displacement radius and never plateaus -- 2% of M20 at radius 0 rising monotonically to
48% at twice the neighbour radius (F-149). There is no scale at which "the same solution" stops
being a choice, so no floor can be read off it.

What Shirley describes is available directly instead, without inventing an operator at all,
because *this pipeline's refinement conditions are themselves random*: `random_subsampling`
fits each iterate to `n_peaks - n_drop` peaks drawn with probability proportional to 1/q2 --
six of twenty for monoclinic, three of twenty for tetragonal. Two runs of this program on one
pattern refine the same solution under different conditions and report different merits, which
is Shirley's sentence made literal. So the floor is measured over **runs of the program**:
`submit_fom_floor_arms.sh` indexes one set of patterns several times under different search
seeds and identical peak lists, and `run_fom_floor_report.py` reads the spread. That also
measures the right thing for a gate, which compares two numbers computed on one pool: what
moves between arms is not only each merit's value but the *difference* between two merits.
"""
import argparse
import json
import multiprocessing
import os
import time

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics
from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
from mlindex.utilities.UnitCellTools import fix_unphysical
from run_fom_assignment import roc_auc
from run_fom_zoo_features import commit_hash
from run_fom_zoo_features import EVALUABLE_BUNDLES

# The production search parameters, per lattice system, as the seven factories in
# `UtilitiesOptimizer` set them. Held here rather than imported because building an optimizer
# would load ~3 GB of models to read five floats -- but *checked* against the factories by
# `tests/test_fom_floor.py::test_production_loop_matches_the_factories`, which parses them out
# of the source rather than trusting this table. The earlier version of this table was wrong
# for four of the seven systems (F-148), which is why the test exists.
PRODUCTION_LOOP = {
    'cubic':        {'n_peaks': 10, 'n_drop': 8,  'n_iterations': 5,
                     'neighbor_radius': 0.000026, 'downsample_radius': 0.002},
    'tetragonal':   {'n_peaks': 20, 'n_drop': 17, 'n_iterations': 30,
                     'neighbor_radius': 0.000213, 'downsample_radius': 0.0001},
    'hexagonal':    {'n_peaks': 20, 'n_drop': 17, 'n_iterations': 30,
                     'neighbor_radius': 0.000213, 'downsample_radius': 0.0001},
    'rhombohedral': {'n_peaks': 20, 'n_drop': 17, 'n_iterations': 30,
                     'neighbor_radius': 0.000213, 'downsample_radius': 0.0001},
    'orthorhombic': {'n_peaks': 20, 'n_drop': 14, 'n_iterations': 50,
                     'neighbor_radius': 0.000338, 'downsample_radius': 0.0001},
    'monoclinic':   {'n_peaks': 20, 'n_drop': 14, 'n_iterations': 60,
                     'neighbor_radius': 0.000547, 'downsample_radius': 0.0001},
    'triclinic':    {'n_peaks': 20, 'n_drop': 12, 'n_iterations': 60,
                     'neighbor_radius': 0.000679, 'downsample_radius': 0.0001},
    }

# Multiples of the lattice's own neighbour radius, for the displacement that removes the stored
# cell's incumbency. Swept rather than picked: the floor is the small-radius plateau, and a
# radius large enough to leave the basin measures the basin instead. 0.0 is retained as a
# diagnostic -- it is the one-sided case described in the module docstring, not a control.
RADIUS_MULTIPLES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_RADIUS_MULTIPLE = 0.25

MINIMUM_UC, MAXIMUM_UC = 2, 500
ASSIGNMENT_THRESHOLD = 0.95
N_REFINE_STEPS = 25

# Merits whose value floor is not a meaningful quantity: they are probabilities or test
# statistics that legitimately sit at or near zero, so a *relative* spread divides by nothing.
# They are still reported in the rank-stability table, which needs no scale.
UNSCALED_MERITS = ('chi2_fixed_pvalue', 'chi2_entrywise_pvalue', 'X_N')


def _parse_args():
    parser = argparse.ArgumentParser(description='S06b -- measure the reproducibility floor')
    parser.add_argument('--stage', choices=['diagnose', 'ensemble', 'refine-gain'],
                        required=True)
    parser.add_argument('--benchmark-dir', default=os.path.join('mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--feature-dir', default=os.path.join('mlindex', 'data', 'fom_features'))
    parser.add_argument('--out-dir', default=os.path.join('mlindex', 'data', 'fom_floor'))
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--split', default='fom-dev')
    parser.add_argument('--train-split', default='fom-train',
                        help='where every threshold is selected (PROTOCOL section 8)')
    parser.add_argument('--bundles', nargs='+', default=None,
                        help='default: the six evaluable bundles')
    parser.add_argument('--n-entries', type=int, default=400,
                        help='source entries sampled per bundle; 0 for all of them')
    parser.add_argument('--n-replicates', type=int, default=8)
    parser.add_argument('--n-processes', type=int, default=1)
    parser.add_argument('--radius-multiple', type=float, default=DEFAULT_RADIUS_MULTIPLE)
    parser.add_argument('--radius-sweep', action='store_true',
                        help='ensemble: run every radius in RADIUS_MULTIPLES (diagnostic)')
    parser.add_argument('--diagnose-per-lattice', type=int, default=25)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', default='S06b_floor')
    return parser.parse_args()


# ----------------------------------------------------------------------------------------
# the operators
# ----------------------------------------------------------------------------------------

def perturb(xnn, radius, lattice_system, rng):
    """Isotropic displacement at a fixed radius, then the pipeline's own physicality repair.

    `ErrorAdder.perturb_xnn`'s construction, batched over candidates rather than replicating
    one cell: draw uniform, normalise to the sphere, scale.
    """
    if radius == 0.0:
        return np.array(xnn, dtype=np.float64, copy=True)
    step = rng.uniform(-1.0, 1.0, size=xnn.shape)
    step *= radius / np.linalg.norm(step, axis=1)[:, np.newaxis]
    return fix_unphysical(xnn=xnn + step, rng=rng, minimum_unit_cell=MINIMUM_UC,
                          maximum_unit_cell=MAXIMUM_UC, lattice_system=lattice_system)


def replay(q2_obs, xnn, lattice_system, bravais_lattice, hkl_ref, seed, n_iterations=None):
    """Re-run the production refinement loop from these cells under a fresh seed.

    The operator is `Candidates` itself rather than a reimplementation of it, so "different
    refinement conditions" means exactly what the deployed program does: one deterministic
    all-peaks step, then `n_iterations` of `random_subsampling` -- each fitting the cell to
    `n_peaks - n_drop` peaks drawn with probability proportional to 1/q2 -- keeping the
    best-M20 iterate throughout, then `refine_cell`'s masked step under its accept-if-improved
    rule. The only thing that differs between replicates is the random stream.

    Note the loop is entered at a converged cell rather than at a generated one, so it is the
    *refinement* that is replayed and not the search. That is deliberate: the search is
    candidate generation, which this project measures elsewhere and which would change the
    pool's membership rather than its scores.

    `hkl_ref` is the model's truncated reference list, unfiltered by extinction group, because
    that is what `Candidates` holds during the loop; the extinction group is applied afterwards
    by the scorer.
    """
    from mlindex.optimization.Candidates import Candidates

    loop = PRODUCTION_LOOP[lattice_system]
    opt_params = {
        'minimum_uc': MINIMUM_UC,
        'maximum_uc': MAXIMUM_UC,
        'assignment_threshold': ASSIGNMENT_THRESHOLD,
        'figure_of_merit': 'M20',
        }
    candidates = Candidates(
        q2_obs=q2_obs, xnn=np.array(xnn, dtype=np.float64, copy=True), hkl_ref=hkl_ref,
        lattice_system=lattice_system, bravais_lattice=bravais_lattice,
        opt_params=opt_params, rng=np.random.default_rng(seed), fom=None,
        zero_error=False, wavelength=None,
        )
    candidates.deterministic({'n_peaks': loop['n_peaks']})
    iteration_info = {'n_peaks': loop['n_peaks'], 'n_drop': loop['n_drop'],
                      'uniform_sampling': False}
    for _ in range(loop['n_iterations'] if n_iterations is None else n_iterations):
        candidates.random_subsampling(iteration_info)
    candidates.refine_cell()
    return candidates.best_xnn


def reassign_extinction_group(q2_obs, xnn, lattice_system, bravais_lattice):
    """`Candidates.assign_extinction_group`, applied to a batch of replayed cells.

    Production picks the extinction group *after* the loop, by maximising M20 over every group
    of the lattice, and the stored `spacegroup` column is that winner. A replayed cell must
    therefore be given the same choice rather than inheriting the stored group: the loop
    optimises M20 against the model's *unfiltered* reference list, so holding the group fixed
    scores the landed cell under a list it was not selected against and reads as a loss that
    production would never have taken.

    Strict `>` reproduces `np.argmax`'s first-maximum rule over the dictionary's own order.
    """
    from mlindex.utilities.FigureOfMerits import get_M20
    from mlindex.utilities.numba_functions import fast_assign
    from mlindex.utilities.Q2Calculator import Q2Calculator

    groups = FomBenchmark.spacegroup_reference_sets(lattice_system, bravais_lattice)
    best_M20 = np.full(xnn.shape[0], -np.inf)
    best_group = np.empty(xnn.shape[0], dtype=object)
    for name, hkl_ref in groups.items():
        q2_ref_calc = Q2Calculator(
            lattice_system=lattice_system, hkl=hkl_ref, tensorflow=False,
            representation='xnn',
            ).get_q2(xnn)
        hkl_assign = fast_assign(q2_obs, q2_ref_calc)
        q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
        # get_M20 mutates q2_ref_calc in place; the array is fresh on every iteration.
        M20 = get_M20(q2_obs, q2_calc, q2_ref_calc)
        improved = M20 > best_M20
        best_M20[improved] = M20[improved]
        best_group[improved] = name
    return best_group, best_M20


def masked_step(q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
                models_directory=None):
    """One `Candidates.refine_cell` Gauss-Newton step, and the peak mask it was taken under.

    Split out of `refine_to_convergence` so the diagnose stage can ask the separate question
    of whether the *stored* cell is a stationary point of this objective at all.
    """
    from mlindex.utilities.FigureOfMerits import get_M20_likelihood_from_xnn

    _, _, hkl, _ = FomBenchmark.assign_lines(
        q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
        models_directory=models_directory)
    _, probability, _ = get_M20_likelihood_from_xnn(
        q2_obs=q2_obs, xnn=xnn, hkl=hkl, lattice_system=lattice_system,
        bravais_lattice=bravais_lattice)
    indexed = probability > ASSIGNMENT_THRESHOLD
    counts = np.sum(indexed, axis=1)
    step = np.zeros_like(xnn)
    # Grouped by how many peaks each candidate indexes, exactly as refine_cell does: the
    # design matrix has a different width per group and they cannot be batched together.
    for n in np.unique(counts):
        if n < xnn.shape[1] + 1:
            continue    # under-determined; leave these where they are
        rows = counts == n
        columns = np.argwhere(indexed[rows])[:, 1].reshape((int(rows.sum()), int(n)))
        target = CandidateOptLoss(np.take(q2_obs, columns), lattice_system=lattice_system)
        target.update(np.take_along_axis(hkl[rows], columns[:, :, np.newaxis], axis=1),
                      xnn[rows])
        step[rows] = target.gauss_newton_step(xnn[rows])
    return step, counts


def refine_to_convergence(q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
                          models_directory=None):
    """`refine_cell`'s objective, iterated -- the operator F-142 adopted, kept for diagnosis.

    It is *not* the ensemble operator. It has no fixed point at the pool's stored cells
    (F-147), and the accept-if-improved rule is deliberately omitted here because the question
    the diagnose stage asks is where the objective's optimum lies, not whether the pipeline
    would have moved to it.
    """
    xnn = np.array(xnn, dtype=np.float64, copy=True)
    for _ in range(N_REFINE_STEPS):
        step, _ = masked_step(q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
                              models_directory)
        xnn = xnn + step
        if not np.any(np.isfinite(step)) or np.nanmax(np.abs(step)) < 1e-14:
            break
    return xnn


# ----------------------------------------------------------------------------------------
# shared loading
# ----------------------------------------------------------------------------------------

# The pool columns the ensemble carries through, so a replicate frame is a drop-in for a
# benchmark shard in `FomMetrics.evaluate`. `in_top_n` is the pipeline's own rank flag and is
# meaningless once the cells move, so it is recomputed rather than inherited.
POOL_COLUMNS = (
    'entry_id', 'bravais_lattice', 'lattice_system', 'candidate_id', 'xnn', 'spacegroup',
    'M20', 'Minfo', 'n_peaks', 'volume', 'in_top_n', 'is_correct', 'is_off_by_two',
    'is_degenerate', 'volume_ratio_to_truth',
    )

_HKL_REF = {}


def hkl_ref_unfiltered(lattice_system, bravais_lattice):
    """The model's truncated reference list, cached per process."""
    key = (lattice_system, bravais_lattice)
    if key not in _HKL_REF:
        _HKL_REF[key] = np.load(FomBenchmark._hkl_ref_path(lattice_system, bravais_lattice))
    return _HKL_REF[key]


def peak_lists(entries, bundle):
    """(entry_id -> q2_obs) for **one** condition bundle.

    Keyed by the bundle, not by `entry_id` alone. A consolidated entry table holds one row per
    (entry, condition) and `set_index('entry_id')` silently keeps the last of the seven, so an
    earlier version of this harness refined every candidate against `error2_cont0`'s peak list
    whatever bundle it came from -- peaks up to 2.5% away in q2 (F-148). `FomBenchmark`'s own
    loaders take `(entry_id, condition_bundle)` for exactly this reason (F-056, SCHEMA.md).
    """
    frame = entries.loc[entries['condition_bundle'] == bundle]
    if frame['entry_id'].duplicated().any():
        raise ValueError(f'{bundle}: entry_id is not a key within one bundle')
    return frame.set_index('entry_id')['q2_obs'].to_dict()


def sample_entry_ids(entries, split, bundle, n_entries, seed):
    """Source entries, sampled once and reused across replicates and radii."""
    available = entries.loc[(entries['split'] == split)
                            & (entries['condition_bundle'] == bundle), 'entry_id']
    available = np.sort(pd.unique(available))
    if not n_entries or n_entries >= available.size:
        return available
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(available, size=n_entries, replace=False))


# ----------------------------------------------------------------------------------------
# stage: diagnose
# ----------------------------------------------------------------------------------------

def run_diagnose(args, entries):
    """Is the stored cell a fixed point of the refinement, and if not, why not?

    F-142 recorded a hypothesis and left it untested: that the stored cell is the end of an
    ordered chain -- `refine_cell` then `standardize_cell` then `correct_off_by_two` then
    `assign_extinction_group` -- whose last two stages move the cell *after* refinement, so a
    candidate they touched is no longer at the optimum that produced it. This tests it, against
    the pool's own `is_off_by_two` column and against whether the winning extinction group
    narrows the reference list at all.
    """
    bundle = (args.bundles or ['error1_cont0'])[0]
    peaks = peak_lists(entries, bundle)
    keep = set(entries.loc[entries['split'] == args.split, 'entry_id'])
    pool = FomBenchmark.load_candidates(
        args.benchmark_dir, bundles=[bundle],
        columns=['entry_id', 'bravais_lattice', 'lattice_system', 'candidate_id', 'xnn',
                 'spacegroup', 'M20', 'is_correct', 'is_off_by_two', 'n_peaks', 'n_indexed',
                 'hkl_ref_length'],
        )
    pool = pool.loc[pool['entry_id'].isin(keep)]
    rng = np.random.default_rng(args.seed)
    sample = pd.concat(
        [group.iloc[rng.choice(group.shape[0],
                               min(args.diagnose_per_lattice, group.shape[0]), replace=False)]
         for _, group in pool.groupby(['bravais_lattice'])],
        ignore_index=True,
        )
    print(f'{sample.shape[0]:,} candidates, {bundle}, {args.split}', flush=True)

    rows = []
    for (system, lattice, spacegroup, entry_id), group in sample.groupby(
            ['lattice_system', 'bravais_lattice', 'spacegroup', 'entry_id'], sort=False):
        n_peaks = int(group['n_peaks'].iloc[0])
        q2_obs = np.asarray(peaks[entry_id], dtype=np.float64)[:n_peaks]
        xnn0 = np.vstack([np.asarray(v, dtype=np.float64) for v in group['xnn']])

        stored, _, _, _ = FomBenchmark.recompute_scores(
            q2_obs, xnn0, system, lattice, spacegroup, ASSIGNMENT_THRESHOLD)
        step, counts = masked_step(q2_obs, xnn0, system, lattice, spacegroup)
        one_step, _, _, _ = FomBenchmark.recompute_scores(
            q2_obs, xnn0 + step, system, lattice, spacegroup, ASSIGNMENT_THRESHOLD)
        landed = refine_to_convergence(q2_obs, xnn0, system, lattice, spacegroup)
        converged, _, _, _ = FomBenchmark.recompute_scores(
            q2_obs, landed, system, lattice, spacegroup, ASSIGNMENT_THRESHOLD)

        n_reference = FomBenchmark.hkl_ref_for(system, lattice, spacegroup).shape[0]
        rows.append(pd.DataFrame({
            'bravais_lattice': lattice, 'lattice_system': system, 'entry_id': entry_id,
            'candidate_id': group['candidate_id'].to_numpy(),
            'is_off_by_two': group['is_off_by_two'].to_numpy(),
            'is_correct': group['is_correct'].to_numpy(),
            'extinction_narrows': n_reference < int(group['hkl_ref_length'].iloc[0]),
            'n_indexed': counts,
            'under_determined': counts < xnn0.shape[1] + 1,
            'M20_pool': group['M20'].to_numpy(),
            'M20_stored': stored, 'M20_one_step': one_step, 'M20_converged': converged,
            'step_relative': np.linalg.norm(step, axis=1) / np.linalg.norm(xnn0, axis=1),
            }))
    frame = pd.concat(rows, ignore_index=True)
    frame['round_trip'] = (np.abs(frame['M20_stored'] - frame['M20_pool'])
                           / np.abs(frame['M20_pool']))
    frame['moved'] = (np.abs(frame['M20_converged'] - frame['M20_stored'])
                      / np.abs(frame['M20_stored']))

    os.makedirs(args.artifact_dir, exist_ok=True)
    path = os.path.join(args.artifact_dir, 'S06b_fixed_point.csv')
    frame.to_csv(path, index=False)

    determinate = frame.loc[~frame['under_determined']]
    delta = frame['M20_converged'] - frame['M20_stored']
    print()
    print(f'round trip against the stored M20: median {frame["round_trip"].median():.1e}, '
          f'max {frame["round_trip"].max():.1e}')
    print(f'stationary at the stored cell (masked step < 1e-12 relative): '
          f'{float((determinate["step_relative"] < 1e-12).mean()):.3f} '
          f'of {determinate.shape[0]} determinate candidates')
    print(f'under-determined (fewer indexed peaks than free parameters): '
          f'{float(frame["under_determined"].mean()):.3f}')
    print(f'refining to convergence: M20 improves {float((delta > 0).mean()):.3f}, '
          f'worsens {float((delta < 0).mean()):.3f}, unchanged {float((delta == 0).mean()):.3f}')
    print()
    print("F-142's hypothesis -- movement is the chain's last two stages:")
    for predictor in ('is_off_by_two', 'extinction_narrows', 'under_determined'):
        table = frame.groupby(predictor)['moved'].agg(
            fraction_moved=lambda s: float((s > 1e-6).mean()), median='median', n='size')
        print(f'\n  by {predictor}:')
        print(table.round(4).to_string().replace('\n', '\n  '))
    print('\n  by lattice:')
    print(frame.groupby('bravais_lattice')['moved'].agg(
        fraction_moved=lambda s: float((s > 1e-6).mean()), median='median', n='size'
        ).round(4).to_string().replace('\n', '\n  '))
    print(f'\nwrote {path}')
    return frame


# ----------------------------------------------------------------------------------------
# stage: ensemble
# ----------------------------------------------------------------------------------------

def _replicate_frame(group, peaks_for_entry, entry_id, bundle, radius_multiple, replicate,
                     seed):
    """One replicate of one entry's whole pool: displace, replay, and score every candidate.

    The pool's *membership* is held fixed -- no re-pruning, no re-deduplication -- so this
    isolates the scoring and ranking noise from generation noise. A full re-run would move
    both, so what comes out of here is a lower bound on the spread of a re-run.
    """
    landed = np.empty(group.shape[0], dtype=object)
    shift = np.zeros(group.shape[0])
    regrouped = np.empty(group.shape[0], dtype=object)
    for (system, lattice), rows in group.groupby(['lattice_system', 'bravais_lattice'],
                                                 sort=False):
        n_peaks = int(rows['n_peaks'].iloc[0])
        q2_obs = np.asarray(peaks_for_entry, dtype=np.float64)[:n_peaks]
        xnn0 = np.vstack([np.asarray(v, dtype=np.float64) for v in rows['xnn']])
        # Derived from the entry id so the same entry meets the same noise in every condition
        # and paired comparisons stay valid (PROTOCOL section 6).
        stream = np.random.default_rng(
            [seed, abs(hash((entry_id, lattice))) % (2 ** 32), replicate]
            )
        radius = radius_multiple * PRODUCTION_LOOP[system]['neighbor_radius']
        moved = perturb(xnn0, radius, system, stream)
        result = replay(q2_obs, moved, system, lattice,
                        hkl_ref_unfiltered(system, lattice),
                        seed=int(stream.integers(0, 2 ** 31 - 1)))
        winner, _ = reassign_extinction_group(q2_obs, result, system, lattice)
        positions = group.index.get_indexer(rows.index)
        landed[positions] = list(result)
        regrouped[positions] = winner
        shift[positions] = (np.linalg.norm(result - xnn0, axis=1)
                            / np.linalg.norm(xnn0, axis=1))
    # The stored merits are renamed rather than carried under their own names: `zoo_features`
    # recomputes `M20` on the landed cell and a stale column beside it under the same name is
    # exactly the confusion this measurement cannot afford.
    return group.assign(xnn=list(landed), spacegroup=regrouped, xnn_shift=shift,
                        spacegroup_changed=regrouped != group['spacegroup'].to_numpy(),
                        condition_bundle=bundle, replicate=replicate,
                        radius_multiple=radius_multiple,
                        ).rename(columns={'M20': 'M20_pool', 'Minfo': 'Minfo_pool'})


def _entry_task(payload):
    """Module-level worker: every replicate of one entry. Spawn-safe, picklable arguments."""
    (group, peaks_for_entry, entry_id, bundle, radius_multiples, n_replicates, seed,
     entry_rows) = payload
    out = []
    for radius_multiple in radius_multiples:
        for replicate in range(n_replicates):
            frame = _replicate_frame(group, peaks_for_entry, entry_id, bundle,
                                     radius_multiple, replicate, seed)
            features, _ = FomBenchmark.zoo_features(frame, entry_rows)
            # Both are indexed by `frame`'s own index, which zoo_features reindexes onto.
            carried = [column for column in frame.columns
                       if column not in features.columns and column != 'xnn']
            out.append(pd.concat([features, frame[carried]], axis=1))
    return pd.concat(out, ignore_index=True)


def summarise_displacement(ensemble):
    """Per displacement radius: what the replay does to M20, and whether it plateaus.

    The question this answers is whether "displace and re-refine" isolates a floor. It does
    not: `spread` grows monotonically with the radius with no flat region, so any number read
    off it is a statement about the chosen radius rather than about the merit (F-149). The
    columns are kept because the shape of that growth is the useful part -- it is a direct
    measurement of how wide the basin the pipeline's own refinement can recover from is.

    `bias` is the median ratio of the replayed M20 to the pool's stored one, and it is below 1
    for a reason that is not noise: the stored value is the arg-max over a stochastic search,
    so any independent repeat regresses towards the mean.
    """
    rows = []
    for radius, group in ensemble.groupby('radius_multiple'):
        values = group.pivot_table(index=['entry_id', 'bravais_lattice', 'candidate_id'],
                                   columns='replicate', values='M20')
        scale = values.abs().median(axis=1).replace(0.0, np.nan)
        spread = (values.max(axis=1) - values.min(axis=1)) / scale
        ratio = group['M20'] / group['M20_pool']
        rows.append({
            'radius_multiple': float(radius),
            'n_candidates': int(values.shape[0]),
            'median_spread': float(spread.median()),
            'p75_spread': float(spread.quantile(0.75)),
            'median_bias': float(ratio.median()),
            'median_xnn_shift': float(group['xnn_shift'].median()),
            'spacegroup_changed': float(group['spacegroup_changed'].mean()),
            })
    return pd.DataFrame(rows)


def run_refine_gain(args, entries):
    """Shirley's criterion 1, tested: do false solutions resist refinement?

    LITERATURE section 5 item 24 -- Shirley 1980 argues that a wrong cell "resists attempts to
    improve it by refinement" while a right one does not, which makes the *gain* under
    re-refinement a merit in its own right. It was deferred to S06b because it needs the same
    machinery as the floor, and the displacement ensemble already carries it: every row holds
    the stored M20 and the M20 the replayed cell reaches, with the correctness label beside it.

    Reported as AUC against `is_correct`, beside M20's own AUC on the same rows, because the
    question is not whether the gain is informative but whether it adds anything to the merit
    it is computed from.
    """
    path = os.path.join(args.out_dir, f'{args.tag}_{(args.bundles or ["error1_cont0"])[0]}.parquet')
    ensemble = pd.read_parquet(path)
    print(f'{ensemble.shape[0]:,} rows from {path}', flush=True)

    ensemble = ensemble.assign(
        refine_gain=(ensemble['M20'] - ensemble['M20_pool']) / ensemble['M20_pool'],
        refine_gain_absolute=ensemble['M20'] - ensemble['M20_pool'],
        )
    rows = []
    for radius, group in ensemble.groupby('radius_multiple'):
        labels = group['is_correct'].to_numpy(dtype=bool)
        if labels.sum() < 20 or (~labels).sum() < 20:
            continue
        for name in ('M20_pool', 'M20', 'refine_gain', 'refine_gain_absolute'):
            rows.append({
                'radius_multiple': float(radius), 'score': name,
                'auc': roc_auc(group[name].to_numpy(dtype=float), labels),
                'n_correct': int(labels.sum()), 'n': int(labels.size),
                })
    table = pd.DataFrame(rows).pivot(index='radius_multiple', columns='score',
                                     values='auc').reset_index()
    out = os.path.join(args.artifact_dir, 'S06b_refine_gain.csv')
    table.to_csv(out, index=False)
    print(table.round(4).to_string(index=False))
    print(f'wrote {out}')
    return table


def run_ensemble(args, entries):
    os.makedirs(args.out_dir, exist_ok=True)
    bundles = args.bundles or list(EVALUABLE_BUNDLES)
    radius_multiples = (list(RADIUS_MULTIPLES) if args.radius_sweep
                        else [args.radius_multiple])
    started = time.perf_counter()

    for bundle in bundles:
        peaks = peak_lists(entries, bundle)
        entry_ids = sample_entry_ids(entries, args.split, bundle, args.n_entries, args.seed)
        pool = FomBenchmark.load_candidates(
            args.benchmark_dir, bundles=[bundle], columns=list(POOL_COLUMNS))
        pool = pool.loc[pool['entry_id'].isin(set(entry_ids))]
        entry_rows = entries.loc[entries['condition_bundle'] == bundle]
        print(f'{bundle}: {len(entry_ids):,} entries, {pool.shape[0]:,} candidates, '
              f'{len(radius_multiples)} radii x {args.n_replicates} replicates', flush=True)

        payloads = []
        for entry_id, group in pool.groupby('entry_id', sort=False):
            payloads.append((
                group.reset_index(drop=True), peaks[entry_id], entry_id, bundle,
                radius_multiples, args.n_replicates, args.seed,
                entry_rows.loc[entry_rows['entry_id'] == entry_id],
                ))

        collected = []
        if args.n_processes > 1:
            context = multiprocessing.get_context('spawn')
            with context.Pool(args.n_processes) as workers:
                for index, frame in enumerate(
                        workers.imap_unordered(_entry_task, payloads, chunksize=1)):
                    collected.append(frame)
                    if (index + 1) % 100 == 0:
                        print(f'  {index + 1}/{len(payloads)} entries, '
                              f'{time.perf_counter() - started:.0f}s', flush=True)
        else:
            for index, payload in enumerate(payloads):
                collected.append(_entry_task(payload))
                if (index + 1) % 25 == 0:
                    print(f'  {index + 1}/{len(payloads)} entries, '
                          f'{time.perf_counter() - started:.0f}s', flush=True)

        ensemble = pd.concat(collected, ignore_index=True)
        path = os.path.join(args.out_dir, f'{args.tag}_{bundle}.parquet')
        ensemble.to_parquet(path, index=False)
        print(f'  wrote {path}: {ensemble.shape[0]:,} rows '
              f'({time.perf_counter() - started:.0f}s)', flush=True)
        summary = summarise_displacement(ensemble)
        summary_path = os.path.join(args.artifact_dir, f'S06b_displacement_{bundle}.csv')
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False), flush=True)
        print(f'  wrote {summary_path}', flush=True)

    manifest = {
        'commit': commit_hash(), 'split': args.split, 'bundles': list(bundles),
        'n_entries': args.n_entries, 'n_replicates': args.n_replicates,
        'radius_multiples': radius_multiples, 'seed': args.seed,
        'operator': 'perturb + production replay (deterministic + random_subsampling + refine_cell)',
        'production_loop': PRODUCTION_LOOP,
        }
    with open(os.path.join(args.out_dir, f'{args.tag}_manifest.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = _parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    entries = FomBenchmark.load_entries(args.benchmark_dir)
    started = time.perf_counter()
    if args.stage == 'diagnose':
        run_diagnose(args, entries)
    elif args.stage == 'refine-gain':
        run_refine_gain(args, entries)
    else:
        run_ensemble(args, entries)
    print(f'stage {args.stage} finished in {time.perf_counter() - started:.1f}s')


if __name__ == '__main__':
    main()
