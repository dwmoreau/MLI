"""The S01 acceptance gate: every figure of merit in the zoo, on de Wolff's Li6B4O9 pair.

de Wolff 1968 Table 3 is the famous counterexample. The same 22 observed lines admit a correct
monoclinic P indexing (a = 9.18, b = 23.41, c = 3.32 A, beta = 92.68 deg) and an incorrect
orthorhombic B-centred one (12.30 x 13.37 x 11.87 A), and M20 prefers the wrong one: 5.4 against
5.3. There is "not the remotest analogy between correct and incorrect reciprocal lattice; there
does not even exist a common zone" -- only the intensities separate them, and this project is
position-only by policy (PLAN 7).

Wu 1988 reports that his accurate expression does separate them, 7.7 against 3.8. Whether the
reversed, symmetric and analytic-null merits do too is exactly what S01 has to report.

Q values are taken from de Wolff's own printed columns. His cell constants are rounded to three or
four significant figures, too hard to regenerate the Q_calc column from (see
tests/test_fom_literature.py), so the cells are used only for the reference-line enumeration and
for the quantities that need a metric tensor.

    python mlindex/scripts/run_fom_li6b4o9.py

Run it with the development env:
    /global/cfs/cdirs/m4064/dwmoreau/envs/pytorch/bin/python
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'tests'))

import fixtures_fom_literature as fixtures  # noqa: E402
from mlindex.utilities.FigureOfMerits import compute_all  # noqa: E402
from mlindex.utilities.FigureOfMerits import estimate_sigma_entrywise  # noqa: E402
from mlindex.utilities.FigureOfMerits import get_M20  # noqa: E402
from mlindex.utilities.UnitCellTools import get_hkl_matrix  # noqa: E402

# Merits where a larger value is the better candidate. Everything else is either lower-is-better or
# a descriptive quantity rather than a ranking, and is reported without a verdict.
HIGHER_IS_BETTER = {
    'M20', 'M_tilde', 'M_rev', 'M_sym', 'M_wu', 'M_star', 'M_star_corrected', 'M_1', 'M_nn',
    'M_info_clipped', 'null_tail_nll', 'F_N_q', 'F_N', 'M_werner_frac',
    'chi2_fixed_pvalue', 'chi2_entrywise_pvalue',
}
LOWER_IS_BETTER = {'X_N', 'n_over', 'max_gap', 'bic', 'chi2_fixed', 'chi2_entrywise'}


def build_case(label):
    """(lattice_system, bravais_lattice, xnn, q2_obs, q2_calc, q2_ref) for one indexing.

    Everything is in de Wolff's units of 10^4 A^-2, which cancels out of every merit that is a
    ratio; the two that are not (nll and bic) are reported for contrast only.
    """
    if label == 'correct':
        a, b, c, beta = fixtures.DEWOLFF68_LI6B4O9_CORRECT_CELL
        beta = np.radians(beta)
        a_star, b_star, c_star = 1/(a*np.sin(beta)), 1/b, 1/(c*np.sin(beta))
        xnn = 1e4*np.array(
            [[a_star**2, b_star**2, c_star**2, 2*a_star*c_star*np.cos(np.pi - beta)]]
        )
        lattice_system, bravais_lattice, column = 'monoclinic', 'mP', 1
        def allowed(hkl):
            return np.ones(len(hkl), dtype=bool)
    else:
        a, b, c = fixtures.DEWOLFF68_LI6B4O9_INCORRECT_CELL
        xnn = 1e4*np.array([[1/a**2, 1/b**2, 1/c**2]])
        lattice_system, bravais_lattice, column = 'orthorhombic', 'oC', 3
        def allowed(hkl):
            # B-centred: h + l = 2n. Not one of the fourteen labels, so it borrows oC's de Wolff
            # coefficients -- both are single-face centrings and take the same factor of one half.
            return (hkl[:, 0] + hkl[:, 2]) % 2 == 0

    rows = [row for row in fixtures.DEWOLFF68_LI6B4O9 if row[column] is not None]
    q2_obs = np.array([row[0] for row in rows])
    q2_calc = np.array([row[column + 1] for row in rows])[np.newaxis]

    limit = 14
    grid = np.arange(-limit, limit + 1)
    hkl = np.stack(
        [axis.ravel() for axis in np.meshgrid(grid, grid, grid, indexing='ij')], axis=1
    )
    hkl = hkl[np.any(hkl != 0, axis=1)]
    hkl = hkl[allowed(hkl)]
    reference = np.unique(get_hkl_matrix(hkl.astype(float), lattice_system), axis=0)
    q2_ref = (reference @ xnn[0])[np.newaxis]
    return lattice_system, bravais_lattice, xnn, q2_obs, q2_calc, q2_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(BASE, 'docs', 'fom', 'artifacts'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for label in ('correct', 'incorrect'):
        lattice_system, bravais_lattice, xnn, q2_obs, q2_calc, q2_ref = build_case(label)
        sigma = estimate_sigma_entrywise(q2_obs, q2_calc, quantile=1.0)
        # g_min: Werner's decimal-quantisation floor. de Wolff's Q are printed to 0.1 in units of
        # 10^-4 A^-2, so the floor here is 0.05 in the same units. Documented, not assumed.
        # min_discrepancy = 0.5, half the coarser of the two printed Q_calc precisions. de Wolff
        # prints the correct indexing's Q_calc to 0.1 and the incorrect one's to 1.0, which leaves
        # three exact zeros in the incorrect column; without a floor those three lines contribute
        # 690 of its 725 nats and the information merits "prefer" the wrong cell sixteen to one
        # (F-026). The floor is a property of the printed table, not an error model.
        output = compute_all(
            q2_obs, q2_calc, q2_ref, xnn, lattice_system, bravais_lattice,
            sigma_entrywise=sigma, g_min=0.05, min_discrepancy=0.5,
        )
        results[label] = output['features']
        results[label + '_sigma'] = output['sigma_treatment']
        print(f'{label}: {q2_ref.shape[1]} reference lines enumerated, '
              f'N20 = {int((q2_ref[0] < q2_calc[0, -1]).sum())}')

    rows = []
    for name in sorted(results['correct']):
        correct = float(np.ravel(results['correct'][name])[0])
        incorrect = float(np.ravel(results['incorrect'][name])[0])
        if name in HIGHER_IS_BETTER:
            verdict = 'SEPARATES' if correct > incorrect else 'fails'
        elif name in LOWER_IS_BETTER:
            verdict = 'SEPARATES' if correct < incorrect else 'fails'
        else:
            verdict = '(not a ranking)'
        rows.append({
            'feature': name,
            'correct': correct,
            'incorrect': incorrect,
            'ratio_correct_over_incorrect': correct/incorrect if incorrect else np.nan,
            'verdict': verdict,
            'sigma_treatment': results['correct_sigma'][name],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(args.out, 'S01_li6b4o9.csv'), index=False)
    print('\n' + frame.to_string(index=False))

    separating = frame[frame['verdict'] == 'SEPARATES']['feature'].tolist()
    print(f'\n{len(separating)} of {(frame["verdict"] != "(not a ranking)").sum()} '
          f'ranking merits prefer the correct cell:')
    print('  ' + ', '.join(separating))

    # Cost, against get_M20's ~14.5 ms/call baseline (ProfileOptimizer.py:485). Measured on a
    # realistic pool rather than on this two-candidate pair.
    rng = np.random.default_rng(0)
    n_candidates, n_ref = 500, 750
    q2_obs = np.sort(rng.uniform(0.01, 0.2, 20))
    q2_ref = np.sort(rng.uniform(0.001, 0.25, (n_candidates, n_ref)), axis=1)
    index = np.argmin(
        np.abs(q2_ref[:, :, np.newaxis] - q2_obs[np.newaxis, np.newaxis]), axis=1
    )
    q2_calc = np.take_along_axis(q2_ref, index, axis=1)
    xnn = np.abs(rng.normal(0.01, 0.003, (n_candidates, 3)))

    def timed(function, repeats=5):
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        return (time.perf_counter() - start)/repeats

    baseline = timed(lambda: get_M20(q2_obs, q2_calc, q2_ref.copy()))
    whole = timed(lambda: compute_all(q2_obs, q2_calc, q2_ref, xnn, 'orthorhombic', 'oP'), 3)
    cost = pd.DataFrame([
        {'what': 'get_M20 (baseline)', 'seconds_per_call': baseline, 'ratio_to_M20': 1.0},
        {'what': 'compute_all (whole zoo)', 'seconds_per_call': whole,
         'ratio_to_M20': whole/baseline},
    ])
    cost.to_csv(os.path.join(args.out, 'S01_fom_cost.csv'), index=False)
    print(f'\ncost over {n_candidates} candidates x {n_ref} reference lines')
    print(cost.to_string(index=False))
    print(f'\nwrote tables to {args.out}')


if __name__ == '__main__':
    main()
