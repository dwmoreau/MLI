"""Is the calculated-line density model a bug, and does it change anything? (S01 follow-up.)

F-027 measured the repo's density model, N(Q) = (4 pi/3) Q^(3/2) / (V* mu), as under-counting the
true number of calculated lines by 30-58% for most Bravais lattices. This script answers the two
questions that follow.

  1. **Is the tuning wrong, or is the functional form wrong?** DWMM tuned mu by hand to match the
     observed density of non-systematically-absent reflections (F-022). If a refit of mu against
     the actual counts lands near the value in the code, the tuning was fine and the residual is
     the functional form; if it lands far away, the tuning itself is off. Reported as
     mu_best_fit / mu_in_code.

  2. **Does it change a decision?** get_M20_likelihood's rho feeds exactly two places
     (Candidates.py:246 and :630). Only the first changes a result: refine_cell selects the peaks
     used for the final cell refinement by rho > assignment_threshold (0.95). n_indexed is
     reported but never ranked on -- run.py sorts on M20 alone. This measures how many peaks cross
     that threshold under the code's density against de Wolff 1961's.

Realistic measurement error is required or every discrepancy is float dust and rho saturates at 1.
The repo's characterised sigma(q2) = 0.00010 + 0.00058 q2 is used to *generate* it, which is what
that model is for (PLAN 2.5); nothing here scores with it.

    python mlindex/scripts/run_fom_density_impact.py

Run it with the development env:
    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.scripts.run_fom_audits import TAGS, load_entries, read_params  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_delta_dewolff61  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_multiplicity_taupin88  # noqa: E402
from mlindex.utilities.UnitCellTools import get_hkl_matrix  # noqa: E402
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn  # noqa: E402
from mlindex.utilities.UnitCellTools import get_unit_cell_volume  # noqa: E402

ASSIGNMENT_THRESHOLD = 0.95   # UtilitiesOptimizer.py, every lattice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    parser.add_argument('--n-max', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for bravais_lattice, tag in TAGS.items():
        params = read_params(tag)
        lattice_system, n_peaks = params['lattice_system'], params['n_peaks']
        hkl_ref = np.load(os.path.join(
            BASE, 'mlindex', 'models', tag, 'data', f'hkl_ref_{bravais_lattice}.npy'))
        hkl_ref = hkl_ref[:-1] if np.all(hkl_ref[-1] == 0) else hkl_ref
        entries = load_entries(bravais_lattice, tag, args.n_max, args.seed)
        if len(entries) == 0:
            continue

        xnn = np.stack(entries['reindexed_xnn'].values).astype(float)
        xnn = xnn[:, params['unit_cell_indices']]
        q2_true = np.stack([np.asarray(row) for row in entries['q2'].values]).astype(float)
        # V* exactly as get_M20_likelihood_from_xnn computes it. This must NOT come from
        # get_dewolff61_axes: for hR that returns the *hexagonal*-axis V*, which is one third of
        # the rhombohedral one this repo uses, and the whole point here is to audit the production
        # quantity. Every other lattice agrees to 1.0000.
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=lattice_system)
        reciprocal_volume = get_unit_cell_volume(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system)

        # --- question 1: refit mu against the actual counts -------------------------------------
        q2_ref = xnn @ get_hkl_matrix(hkl_ref, lattice_system).T
        actual = (q2_ref[:, None, :] < q2_true[:, :, None]).sum(axis=2).astype(float)

        def misfit(log_mu):
            predicted = (4*np.pi/3)*q2_true**1.5/(reciprocal_volume[:, None]*np.exp(log_mu))
            # Fit in log space so the residual is relative, which is how a density is judged.
            return np.mean((np.log(np.maximum(predicted, 1e-12))
                            - np.log(np.maximum(actual, 1e-12)))**2)

        mu_code = get_multiplicity_taupin88(bravais_lattice)[0]
        result = optimize.minimize_scalar(
            misfit, bounds=(np.log(0.05), np.log(5000)), method='bounded')
        mu_fit = float(np.exp(result.x))

        # Residual scatter that even the best mu cannot remove -- that is the functional form.
        best_predicted = (4*np.pi/3)*q2_true**1.5/(reciprocal_volume[:, None]*mu_fit)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio_best = np.log(best_predicted/np.maximum(actual, 1e-12))
        log_ratio_best = log_ratio_best[np.isfinite(log_ratio_best) & (actual > 0)]

        # --- question 2: does the threshold decision move? --------------------------------------
        # Generate realistic errors with the repo's own sigma model, per entry, seeded by row.
        rng = np.random.default_rng(args.seed)
        sigma = 0.00010 + 0.00058*q2_true
        q2_obs = q2_true + sigma*rng.standard_normal(q2_true.shape)
        # The candidate is the true cell, so |dQ| is pure measurement error -- the cleanest case,
        # and the one where an over-confident rho is least excusable.
        discrepancy_q = np.abs(np.sqrt(np.maximum(q2_obs, 0)) - np.sqrt(q2_true))

        # Repo: arg = 8 pi q2_obs |dq| / (V* mu); rho = 1/(1 + arg).
        arg_repo = 8*np.pi*q2_obs*discrepancy_q/(reciprocal_volume[:, None]*mu_code + 1e-100)
        # de Wolff: the same construction with his line density. dN/dq = 2 q dN/dQ = 2 q /(2 Delta),
        # so the expected number of lines within |dq| is 2 |dq| * q / Delta(Q).
        delta = get_delta_dewolff61(q2_obs, xnn, lattice_system, bravais_lattice)
        arg_dewolff = 2*discrepancy_q*np.sqrt(np.maximum(q2_obs, 0))/delta

        # The refit mu is the fair control: it uses the code's own functional form, so the only
        # thing that changes is the constant, and it is unbiased against the actual counts by
        # construction. That matters most for hR, where de Wolff's own row under-predicts by 35%
        # (F-027) and so cannot serve as the reference.
        arg_refit = 8*np.pi*q2_obs*discrepancy_q/(reciprocal_volume[:, None]*mu_fit + 1e-100)

        indexed_repo = (1/(1 + arg_repo)) > ASSIGNMENT_THRESHOLD
        indexed_dewolff = (1/(1 + arg_dewolff)) > ASSIGNMENT_THRESHOLD
        indexed_refit = (1/(1 + arg_refit)) > ASSIGNMENT_THRESHOLD

        rows.append({
            'bravais_lattice': bravais_lattice,
            'n_entries': len(entries),
            'n_peaks': n_peaks,
            'mu_in_code': mu_code,
            'mu_best_fit': mu_fit,
            'mu_ratio_code_over_fit': mu_code/mu_fit,
            'residual_spread_at_best_mu': float(np.std(log_ratio_best)),
            'mean_n_indexed_repo': float(indexed_repo.sum(axis=1).mean()),
            'mean_n_indexed_dewolff': float(indexed_dewolff.sum(axis=1).mean()),
            'mean_n_indexed_refit_mu': float(indexed_refit.sum(axis=1).mean()),
            'peaks_lost_vs_refit': float(
                indexed_refit.sum(axis=1).mean() - indexed_repo.sum(axis=1).mean()),
            'frac_entries_differing': float(
                (indexed_repo.sum(axis=1) != indexed_refit.sum(axis=1)).mean()),
        })
        print(f'{bravais_lattice}: mu code {mu_code:.4g}, refit {mu_fit:.4g} '
              f'(ratio {mu_code/mu_fit:.2f})')

    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(args.out, 'S01_density_impact.csv'), index=False)
    print('\n=== question 1: is the tuning wrong, or the form?')
    print(frame[['bravais_lattice', 'mu_in_code', 'mu_best_fit', 'mu_ratio_code_over_fit',
                 'residual_spread_at_best_mu']].to_string(index=False))
    print('\n=== question 2: does the rho > 0.95 decision move? (true cell, realistic noise)')
    print(frame[['bravais_lattice', 'n_peaks', 'mean_n_indexed_repo', 'mean_n_indexed_refit_mu',
                 'mean_n_indexed_dewolff', 'peaks_lost_vs_refit',
                 'frac_entries_differing']].to_string(index=False))
    print(f'\nwrote {args.out}/S01_density_impact.csv')


if __name__ == '__main__':
    main()
