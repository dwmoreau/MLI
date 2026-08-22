"""S06b -- the reproducibility-floor harness.

Three of these tests exist because the first version of the harness was wrong in a way that
produced a plausible number (F-148): it refined every candidate against the wrong condition
bundle's peak list, and it carried a table of production search radii that disagreed with the
factories for four of the seven lattice systems. Both defects were invisible in the output --
the ensemble ran, the merits came out, and the spread looked like Shirley's 10%.

So the harness's two silent couplings to the rest of the codebase are pinned here: the peak
list must be keyed by (entry, bundle), and the production parameters must be the ones
`UtilitiesOptimizer` actually sets, checked by parsing them out of it rather than by trusting
a copy.
"""
import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'mlindex' / 'scripts'))

from conftest import BL_TO_LATTICE_SYSTEM

import run_fom_floor as floor


# ----------------------------------------------------------------------------------------
# the couplings that were wrong
# ----------------------------------------------------------------------------------------

def _factory_opt_params(source):
    """The `opt_params` / `iteration_info` literals one factory assigns, without running it.

    Building an optimizer to read five floats would load ~3 GB of models, so the harness holds
    a copy of them. This reads the real ones out of the source so the copy cannot drift.
    """
    tree = ast.parse(inspect.getsource(source))
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'opt_params' in targets and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                    found[key.value] = value.value
        if 'iteration_info' in targets and isinstance(node.value, ast.List):
            for element in node.value.elts:
                if not isinstance(element, ast.Dict):
                    continue
                entry = {k.value: v.value for k, v in zip(element.keys, element.values)
                         if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)}
                if entry.get('worker') == 'random_subsampling':
                    found.update({'n_peaks': entry['n_peaks'], 'n_drop': entry['n_drop'],
                                  'n_iterations': entry['n_iterations']})
    return found


def test_production_loop_matches_the_factories():
    """Every number in PRODUCTION_LOOP is the one the corresponding factory sets."""
    from mlindex.optimization import UtilitiesOptimizer

    for lattice_system, expected in floor.PRODUCTION_LOOP.items():
        factory = getattr(UtilitiesOptimizer, f'get_{lattice_system}_optimizer')
        actual = _factory_opt_params(factory)
        for key, value in expected.items():
            assert actual[key] == value, (
                f'{lattice_system}.{key}: harness says {value}, '
                f'get_{lattice_system}_optimizer says {actual[key]}'
                )


def test_peak_lists_are_keyed_by_the_condition_bundle():
    """One entry under two conditions has two peak lists, and the bundle picks between them.

    The consolidated entry table holds one row per (entry, condition), so `entry_id` alone is
    not a key -- `set_index('entry_id')` silently keeps whichever bundle happens to be last.
    An earlier harness did exactly that and refined C1 candidates against C4's peaks.
    """
    entries = pd.DataFrame({
        'entry_id': ['A', 'A', 'B', 'B'],
        'condition_bundle': ['error1_cont0', 'error2_cont0'] * 2,
        'q2_obs': [[1.0, 2.0], [1.5, 2.5], [3.0, 4.0], [3.5, 4.5]],
        })
    assert floor.peak_lists(entries, 'error1_cont0') == {'A': [1.0, 2.0], 'B': [3.0, 4.0]}
    assert floor.peak_lists(entries, 'error2_cont0') == {'A': [1.5, 2.5], 'B': [3.5, 4.5]}


def test_peak_lists_refuses_a_table_that_is_not_a_key():
    entries = pd.DataFrame({
        'entry_id': ['A', 'A'],
        'condition_bundle': ['error1_cont0', 'error1_cont0'],
        'q2_obs': [[1.0], [2.0]],
        })
    with pytest.raises(ValueError, match='not a key'):
        floor.peak_lists(entries, 'error1_cont0')


# ----------------------------------------------------------------------------------------
# the operators
# ----------------------------------------------------------------------------------------

def test_perturb_moves_by_exactly_the_radius():
    rng = np.random.default_rng(0)
    xnn = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    moved = floor.perturb(xnn, 1e-4, 'orthorhombic', rng)
    assert np.allclose(np.linalg.norm(moved - xnn, axis=1), 1e-4, rtol=1e-9)


def test_perturb_at_zero_radius_is_the_identity():
    rng = np.random.default_rng(0)
    xnn = np.array([[0.01, 0.02, 0.03]])
    assert np.array_equal(floor.perturb(xnn, 0.0, 'orthorhombic', rng), xnn)


@pytest.mark.parametrize('bravais_lattice', ['tP', 'oP', 'mP'])
def test_reassign_extinction_group_reproduces_assign_extinction_group(
        models_dir, models_available, bravais_lattice):
    """The harness's group search is `Candidates.assign_extinction_group`, batched.

    Production picks the extinction group after the loop by maximising M20 over every group of
    the lattice, and a replayed cell has to be given the same choice rather than inheriting the
    stored group. Checked against the real method rather than against a second copy of the
    arithmetic.
    """
    if not models_available:
        pytest.skip('needs the model tree for hkl_ref')
    from mlindex.optimization.Candidates import Candidates

    lattice_system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    hkl_ref = np.load(
        models_dir / f'{lattice_system}_1' / 'data' / f'hkl_ref_{bravais_lattice}.npy')
    rng = np.random.default_rng(3)
    q2_obs = np.sort(rng.uniform(0.01, 0.4, size=20))
    xnn = np.abs(rng.normal(0.02, 0.004, size=(6, {'tetragonal': 2, 'orthorhombic': 3,
                                                   'monoclinic': 4}[lattice_system])))

    candidates = Candidates(
        q2_obs=q2_obs, xnn=xnn.copy(), hkl_ref=hkl_ref, lattice_system=lattice_system,
        bravais_lattice=bravais_lattice,
        opt_params={'minimum_uc': 2, 'maximum_uc': 500, 'assignment_threshold': 0.95,
                    'figure_of_merit': 'M20'},
        rng=np.random.default_rng(0), fom=None, zero_error=False, wavelength=None,
        )
    candidates.assign_extinction_group()

    group, M20 = floor.reassign_extinction_group(
        q2_obs, candidates.best_xnn, lattice_system, bravais_lattice)
    assert list(group) == list(candidates.best_spacegroup)
    assert np.allclose(M20, candidates.best_M20, rtol=0, atol=0)


def test_replay_is_reproducible_from_its_seed(models_dir, models_available):
    """Two replays at one seed are identical, and two seeds are not.

    The ensemble's whole content is that the search seed moves and nothing else does, so a
    replay that ignored its seed would report a floor of zero and a replay that ignored its
    inputs would report noise.
    """
    if not models_available:
        pytest.skip('needs the model tree for hkl_ref')
    hkl_ref = np.load(models_dir / 'orthorhombic_1' / 'data' / 'hkl_ref_oP.npy')
    rng = np.random.default_rng(11)
    q2_obs = np.sort(rng.uniform(0.01, 0.4, size=20))
    xnn = np.abs(rng.normal(0.02, 0.004, size=(4, 3)))

    first = floor.replay(q2_obs, xnn, 'orthorhombic', 'oP', hkl_ref, seed=5, n_iterations=4)
    again = floor.replay(q2_obs, xnn, 'orthorhombic', 'oP', hkl_ref, seed=5, n_iterations=4)
    other = floor.replay(q2_obs, xnn, 'orthorhombic', 'oP', hkl_ref, seed=6, n_iterations=4)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_masked_step_leaves_under_determined_candidates_alone(models_dir, models_available):
    """A candidate indexing fewer peaks than it has free parameters cannot be refined.

    This is what F-142 measured as "half the stored cells are fixed points of the masked
    objective": those candidates are not stationary, they are immovable, and the diagnose
    stage separates the two.
    """
    if not models_available:
        pytest.skip('needs the model tree for hkl_ref')
    from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

    hkl_ref = np.load(models_dir / 'triclinic_1' / 'data' / 'hkl_ref_aP.npy')
    spacegroup = sorted(get_spacegroup_hkl_ref(hkl_ref, bravais_lattice='aP').keys())[0]
    rng = np.random.default_rng(7)
    q2_obs = np.sort(rng.uniform(0.01, 0.4, size=20))
    # Cells far from anything that indexes these peaks: nothing clears the 0.95 assignment
    # probability, so no candidate has the seven indexed peaks a triclinic fit needs.
    xnn = np.array([[0.031, 0.029, 0.027, 0.0011, 0.0009, 0.0013]]) * np.array([[1.0], [1.7]])
    step, counts = floor.masked_step(q2_obs, xnn, 'triclinic', 'aP', spacegroup)
    under_determined = counts < xnn.shape[1] + 1
    assert under_determined.any()
    assert np.array_equal(step[under_determined], np.zeros_like(step[under_determined]))


# ----------------------------------------------------------------------------------------
# the reporting arithmetic
# ----------------------------------------------------------------------------------------

def test_relative_spread_is_scale_free():
    from run_fom_floor_report import relative_spread

    values = pd.DataFrame({0: [10.0, 5.0], 1: [11.0, 5.5], 2: [12.0, 6.0]})
    spread = relative_spread(values)
    # (12 - 10) / 11 and (6 - 5) / 5.5 are the same number: the second row is the first,
    # halved.
    assert np.allclose(spread, spread.iloc[0])


def test_induced_from_differences_agrees_with_the_flip_rate_form():
    """The general form reduces to `sqrt(f / n)` on symmetric +-1 flips, which is its check."""
    from run_fom_floor_report import induced_from_differences
    from run_fom_floor_report import induced_standard_error

    rng = np.random.default_rng(0)
    flips = rng.choice([-1.0, 0.0, 1.0], size=200_000, p=[0.02, 0.96, 0.02])
    assert np.isclose(induced_from_differences(flips, 1000),
                      induced_standard_error(0.04, 1000), rtol=0.02)


def test_stratified_standard_error_exceeds_the_unweighted_one_when_a_thin_stratum_is_heavy():
    """CNRS reweighting amplifies a lattice the sample barely covers, and the s.e. must follow.

    The aggregate is a weighted mean over lattices, so a per-entry change in a thin stratum
    moves it by w/n rather than by 1/N. Treating it as unweighted understated the floor by 1.75x
    on the real ensemble, which is the difference between "block C's +1.05 pp is 1.8 standard
    errors" and "it is 1.0".
    """
    from run_fom_floor_report import induced_from_differences
    from run_fom_floor_report import induced_standard_error_stratified

    rng = np.random.default_rng(0)
    # cF is four entries here and carries 22/599 of the weight; oP is 200 and carries 124/599.
    lattices = np.array(['cF'] * 4 + ['oP'] * 200)
    differences = np.concatenate([rng.choice([-1.0, 1.0], size=4),
                                  np.zeros(200)])
    stratified = induced_standard_error_stratified(differences, lattices, 1000)
    plain = induced_from_differences(differences, 1000)
    assert stratified > plain


def test_stratified_standard_error_matches_the_plain_one_for_a_single_stratum():
    from run_fom_floor_report import induced_from_differences
    from run_fom_floor_report import induced_standard_error_stratified

    rng = np.random.default_rng(1)
    differences = rng.choice([-1.0, 0.0, 1.0], size=250, p=[0.02, 0.96, 0.02])
    lattices = np.array(['oP'] * 250)
    assert np.isclose(induced_standard_error_stratified(differences, lattices, 1000),
                      induced_from_differences(differences, 1000), rtol=1e-9)


def test_induced_standard_error_scales_with_the_entry_count():
    from run_fom_floor_report import induced_standard_error

    # A flip rate of 4% over 250 entries induces a bigger wobble than over 1 000.
    small = induced_standard_error(flip_rate=0.04, n_entries=250)
    large = induced_standard_error(flip_rate=0.04, n_entries=1000)
    assert np.isclose(small / large, 2.0, rtol=1e-9)
    assert np.isclose(small, np.sqrt(0.04 / 250), rtol=1e-9)
