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
                when a converged candidate is perturbed off its cell and refined back. The
                `stability` column S06's leaderboard left empty.
  metric floor  What a gate should actually be read against. The spread of the reported
                operating point and top-10 across independent perturbation replicates of the
                whole evaluated pool.

Neither is derived from the other; that is the point.

    python mlindex/scripts/run_fom_floor.py --stage ensemble --n-per-cell 60
    python mlindex/scripts/run_fom_floor.py --stage report

The perturbation is isotropic in xnn at a radius swept in multiples of each lattice's own
`neighbor_radius`, following `ErrorAdder.perturb_xnn`'s construction and
`run_convergence_radius.py`'s precedent of sweeping rather than picking one. Sweeping matters:
a radius large enough to leave the basin measures the basin, not the floor, and the floor is
the small-radius plateau.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from mlindex.model_training import FomBenchmark
from mlindex.model_training import FomMetrics
from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
from mlindex.utilities.UnitCellTools import fix_unphysical
from run_fom_zoo_features import commit_hash
from run_fom_zoo_features import EVALUABLE_BUNDLES

# Multiples of the lattice's own neighbour radius. The largest is deliberately past the point
# where a perturbation should still return to its own cell, so the curve shows where the basin
# ends rather than assuming it.
# 0.0 is the control and is not optional: it perturbs by nothing and refines anyway, so the
# merits it produces must reproduce the stored ones. Without it a spread caused by the harness
# -- a refinement that moves an already-converged cell -- is indistinguishable from the
# reproducibility floor it is supposed to measure.
RADIUS_MULTIPLES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

# From UtilitiesOptimizer's per-system opt_params. Kept here rather than imported because
# building an optimizer would load ~3 GB of models to read two floats.
NEIGHBOR_RADIUS = {
    'cubic': 2.6e-5, 'tetragonal': 1.3e-4, 'hexagonal': 1.3e-4, 'rhombohedral': 1.3e-4,
    'orthorhombic': 2.9e-4, 'monoclinic': 5.47e-4, 'triclinic': 5.47e-4,
    }
MINIMUM_UC, MAXIMUM_UC = 2, 500
N_REFINE_STEPS = 25


def _parse_args():
    parser = argparse.ArgumentParser(description='S06b -- measure the reproducibility floor')
    parser.add_argument('--stage', choices=['ensemble', 'report'], required=True)
    parser.add_argument('--benchmark-dir', default=os.path.join('mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--out-dir', default=os.path.join('mlindex', 'data', 'fom_floor'))
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--split', default='fom-dev')
    parser.add_argument('--bundles', nargs='+', default=None,
                        help='default: the six evaluable bundles')
    parser.add_argument('--n-per-cell', type=int, default=60,
                        help='candidates sampled per (bundle, Bravais lattice)')
    parser.add_argument('--n-replicates', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tag', default='S06b_floor')
    return parser.parse_args()


def perturb(xnn, radius, lattice_system, rng):
    """Isotropic perturbation at a fixed radius, then the pipeline's own physicality repair.

    `ErrorAdder.perturb_xnn`'s construction, batched over candidates rather than replicating
    one cell: draw uniform, normalise to the sphere, scale.
    """
    if radius == 0.0:
        return np.array(xnn, dtype=np.float64, copy=True)
    step = rng.uniform(-1.0, 1.0, size=xnn.shape)
    step *= radius / np.linalg.norm(step, axis=1)[:, np.newaxis]
    return fix_unphysical(xnn=xnn + step, rng=rng, minimum_unit_cell=MINIMUM_UC,
                          maximum_unit_cell=MAXIMUM_UC, lattice_system=lattice_system)


def refine(q2_obs, xnn, lattice_system, bravais_lattice, spacegroup, assignment_threshold=0.95,
           models_directory=None):
    """`Candidates.refine_cell`'s operator, iterated to convergence.

    The pipeline fits the cell **only to peaks assigned above the probability threshold**
    (`Candidates.py:216-221`), not to all twenty. That distinction is the whole measurement:
    a plain all-peaks Gauss-Newton is a *different objective*, and refining a stored cell
    under it walks away from where the pipeline left it -- measured at a median 9.2% and a
    maximum 56% change in M20 at zero perturbation, which would have been reported as the
    reproducibility floor.

    Iterated rather than single-stepped because the brief asks where a perturbed cell lands
    when refined *back to convergence*; `refine_cell` itself takes one step because it runs
    after a hundred iterations that already converged the cell.

    Returns the landed xnn. The accept-if-improved rule is deliberately *not* applied here:
    it would make the radius-0 control vacuously pass by refusing every change, and the
    question is where refinement goes, not whether the pipeline would keep it.
    """
    from mlindex.utilities.FigureOfMerits import get_M20_likelihood_from_xnn

    xnn = np.array(xnn, dtype=np.float64, copy=True)
    for _ in range(N_REFINE_STEPS):
        _, _, hkl, _ = FomBenchmark.assign_lines(
            q2_obs, xnn, lattice_system, bravais_lattice, spacegroup,
            models_directory=models_directory)
        _, probability, _ = get_M20_likelihood_from_xnn(
            q2_obs=q2_obs, xnn=xnn, hkl=hkl, lattice_system=lattice_system,
            bravais_lattice=bravais_lattice)
        indexed = probability > assignment_threshold
        step = np.zeros_like(xnn)
        # Grouped by how many peaks each candidate indexes, exactly as refine_cell does: the
        # design matrix has a different width per group and they cannot be batched together.
        for n in np.unique(np.sum(indexed, axis=1)):
            if n < xnn.shape[1] + 1:
                continue    # under-determined; leave these where they are
            rows = np.sum(indexed, axis=1) == n
            columns = np.argwhere(indexed[rows])[:, 1].reshape((int(rows.sum()), int(n)))
            target = CandidateOptLoss(np.take(q2_obs, columns), lattice_system=lattice_system)
            target.update(np.take_along_axis(hkl[rows], columns[:, :, np.newaxis], axis=1),
                          xnn[rows])
            step[rows] = target.gauss_newton_step(xnn[rows])
        xnn = xnn + step
        if not np.any(np.isfinite(step)) or np.nanmax(np.abs(step)) < 1e-14:
            break
    return xnn


def sample_candidates(args, entries):
    """A stratified sample of the evaluated pool: n_per_cell per (bundle, Bravais lattice).

    Stratified rather than random because the floor is reported by lattice, and a random draw
    over a CNRS-weighted pool would leave cF and cI with almost nothing.
    """
    keep = set(entries.loc[entries['split'] == args.split, 'entry_id'])
    bundles = args.bundles or list(EVALUABLE_BUNDLES)
    rng = np.random.default_rng(args.seed)
    frames = []
    for bundle in bundles:
        pool = FomBenchmark.load_candidates(
            args.benchmark_dir, bundles=[bundle],
            columns=['entry_id', 'bravais_lattice', 'lattice_system', 'candidate_id',
                     'xnn', 'spacegroup', 'M20', 'is_correct', 'n_peaks'])
        pool = pool.loc[pool['entry_id'].isin(keep)]
        for (lattice,), group in pool.groupby(['bravais_lattice']):
            if group.shape[0] > args.n_per_cell:
                group = group.iloc[rng.choice(group.shape[0], args.n_per_cell, replace=False)]
            frames.append(group.assign(condition_bundle=bundle))
    return pd.concat(frames, ignore_index=True)


def run_ensemble(args, entries):
    os.makedirs(args.out_dir, exist_ok=True)
    sample = sample_candidates(args, entries)
    print(f'{sample.shape[0]:,} candidates sampled', flush=True)
    q2_by_entry = entries.set_index('entry_id')['q2_obs'].to_dict()

    rows = []
    started = time.perf_counter()
    # Grouped by (lattice_system, bravais_lattice, spacegroup) because assign_lines is defined
    # per extinction group -- its reference list is the spacegroup's, not the lattice's.
    groups = list(sample.groupby(['condition_bundle', 'lattice_system', 'bravais_lattice',
                                  'spacegroup', 'entry_id'], sort=False))
    for index, ((bundle, system, lattice, spacegroup, entry_id), group) in enumerate(groups):
        # Cubic candidates are scored on ten peaks and everything else on twenty (R5), and
        # the optimizer truncates with a plain prefix slice. Cut it the same way here, or the
        # refinement would fit lines the candidate was never scored against.
        n_peaks = int(group['n_peaks'].iloc[0])
        q2_obs = np.asarray(q2_by_entry[entry_id], dtype=np.float64)[:n_peaks]
        xnn0 = np.stack(group['xnn'].values).astype(np.float64)
        base = NEIGHBOR_RADIUS[system]
        for multiple in RADIUS_MULTIPLES:
            radius = base * multiple
            for replicate in range(args.n_replicates):
                rng = np.random.default_rng(
                    abs(hash((entry_id, lattice, spacegroup, multiple, replicate))) % (2 ** 32))
                moved = perturb(xnn0, radius, system, rng)
                landed = refine(q2_obs, moved, system, lattice, spacegroup)
                frame = group.assign(xnn=list(landed))
                features, _ = FomBenchmark.zoo_features(frame, entries)
                features = features.assign(
                    radius_multiple=multiple, replicate=replicate,
                    condition_bundle=bundle, bravais_lattice=lattice,
                    entry_id=entry_id, candidate_id=group['candidate_id'].to_numpy(),
                    is_correct=group['is_correct'].to_numpy(),
                    M20_stored=group['M20'].to_numpy(),
                    xnn_shift=np.linalg.norm(landed - xnn0, axis=1))
                rows.append(features)
        if (index + 1) % 50 == 0:
            print(f'  {index + 1}/{len(groups)} groups, '
                  f'{time.perf_counter() - started:.0f}s', flush=True)

    ensemble = pd.concat(rows, ignore_index=True)
    path = os.path.join(args.out_dir, f'{args.tag}_ensemble.parquet')
    ensemble.to_parquet(path, index=False)
    print(f'wrote {path}: {ensemble.shape[0]:,} rows')
    return ensemble


def main():
    args = _parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    entries = FomBenchmark.load_entries(args.benchmark_dir)
    started = time.perf_counter()
    if args.stage == 'ensemble':
        run_ensemble(args, entries)
    else:
        raise SystemExit('report stage not yet written')
    print(f'stage {args.stage} finished in {time.perf_counter() - started:.1f}s')


if __name__ == '__main__':
    main()
