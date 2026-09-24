"""The generator fractions are set in one table, and the counts derived from it are today's.

`_literal` holds, verbatim, the per-split-group count lines the seven optimizer factories carried
before the fractions moved into `ENSEMBLE`. Comparing against them -- rather
than against a second call of the new function -- is what makes the test able to fail.
"""
import pytest

from mlindex.optimization.UtilitiesOptimizer import ENSEMBLE
from mlindex.optimization.UtilitiesOptimizer import lattice_budget
from mlindex.utilities.Allocation import check_generator_fractions
from mlindex.utilities.Allocation import generator_info_from_fractions
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES


def _literal(bravais_lattice, n_candidates):
    """The factories' generator_info as written before 2026-09-24, one block per factory branch."""

    if bravais_lattice in ['cF', 'cI', 'cP']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': int(0.45*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0', 'n_unit_cells': int(0.45*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.1*n_candidates)},
            ]
    elif bravais_lattice in ['tI', 'tP']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            ]
    elif bravais_lattice in ['hP']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/8*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/8*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            ]
    elif bravais_lattice in ['hR']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_01', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            ]
    elif bravais_lattice in ['oF']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            ]
    elif bravais_lattice in ['oI']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            ]
    elif bravais_lattice in ['oC']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': int(1/2*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_2_00', 'n_unit_cells': int(1/2*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            ]
    elif bravais_lattice in ['oP']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/4*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/4*0.7*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.25*n_candidates)},
            ]
    elif bravais_lattice in ['mC']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_02', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_03', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_02', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_03', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_4_02', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_4_03', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.4*n_candidates)},
            ]
    elif bravais_lattice in ['mP']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(1/6*0.05*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_00', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_0_01', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_00', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_1_01', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_4_00', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_4_01', 'n_unit_cells': int(1/6*0.55*n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.4*n_candidates)},
            ]
    elif bravais_lattice in ['aP']:
        return [
            {'generator': 'trees', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(0.05 * n_candidates)},
            {'generator': 'abnn', 'split_group': f'{bravais_lattice}_00', 'n_unit_cells': int(0.4 * n_candidates)},
            {'generator': 'templates', 'n_unit_cells': int(0.55 * n_candidates)},
            ]
    raise KeyError(bravais_lattice)


BASE_BUDGET = {'cF': 100, 'cI': 100, 'cP': 100, 'tI': 2000, 'tP': 2000, 'hP': 2000, 'hR': 2000, 'oF': 4000, 'oI': 4000, 'oC': 4000, 'oP': 4000, 'mC': 6000, 'mP': 6000, 'aP': 6000}


def _model_split_groups(bravais_lattice):
    """The split groups the saved models have, read the way OptimizerManager reads them."""
    from mlindex.model_training.Wrapper import Wrapper
    from mlindex.optimization.UtilitiesOptimizer import _resolve_models_dir
    tag = f'{BL_TO_LATTICE_SYSTEM[bravais_lattice]}_1'
    wrapper = Wrapper(
        data_params={'tag': tag, 'base_directory': None,
                     'models_directory': _resolve_models_dir(), 'load_from_tag': True},
        rf_params={}, template_params={bravais_lattice: {'tag': tag, 'load_from_tag': True}},
        abnn_params={}, random_params={bravais_lattice: {'tag': tag, 'load_from_tag': True}})
    wrapper.setup_from_tag(load_bravais_lattice=bravais_lattice)
    return wrapper.data_params['split_groups']


def _derived(bravais_lattice, scale):
    """What OptimizerManager builds at this scale."""
    return generator_info_from_fractions(
        ENSEMBLE[bravais_lattice]['fractions'],
        lattice_budget(bravais_lattice, scale, {}),
        _model_split_groups(bravais_lattice))


def _key(row):
    return (row['generator'], row.get('split_group', ''))


@pytest.mark.parametrize('scale', [0.5, 1, 2])
@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_counts_match_the_literals_they_replaced(bravais_lattice, scale):
    """Same count for every (generator, split group). The order can differ: it now follows the
    model files, which list hP's groups in a different order from the old factory."""
    expected = _literal(bravais_lattice, int(scale*BASE_BUDGET[bravais_lattice]))
    assert sorted(_derived(bravais_lattice, scale), key=_key) == sorted(expected, key=_key)


@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_candidates_are_generated_in_the_model_files_split_group_order(bravais_lattice):
    groups = _model_split_groups(bravais_lattice)
    info = _derived(bravais_lattice, 1)
    assert [row['split_group'] for row in info if row['generator'] == 'trees'] == groups
    assert [row['split_group'] for row in info if row['generator'] == 'abnn'] == groups


def test_every_lattice_has_a_row():
    assert set(ENSEMBLE) == set(BRAVAIS_LATTICES)
    for row in ENSEMBLE.values():
        check_generator_fractions(row['fractions'])


def test_a_half_edited_row_is_refused():
    # trees raised from 0.05 to 0.10 without taking the 0.05 from anywhere else
    with pytest.raises(ValueError, match='sum to 1'):
        check_generator_fractions({'trees': 0.10, 'abnn': 0.70, 'templates': 0.25})


@pytest.mark.parametrize('fractions', [
    {'trees': 0.5, 'abnn': 0.5},
    {'trees': 0.5, 'abnn': 0.25, 'templates': 0.25, 'random': 0.0},
    {'trees': 0.5, 'abnn': 0.25, 'template': 0.25},
    {'trees': 1.1, 'abnn': -0.1, 'templates': 0.0},
    ])
def test_a_malformed_row_is_refused(fractions):
    with pytest.raises(ValueError):
        check_generator_fractions(fractions)


def test_a_zero_share_generator_is_left_out():
    info = generator_info_from_fractions(
        {'trees': 1.0, 'abnn': 0.0, 'templates': 0.0}, 100, ['cP_0'])
    assert info == [{'generator': 'trees', 'split_group': 'cP_0', 'n_unit_cells': 100}]


def test_a_budget_scale_reaches_only_the_lattices_it_names():
    scale = {'cF': 0.5, 'cI': 0.5, 'cP': 0.5}
    assert lattice_budget('cP', 1, scale) == 50
    assert lattice_budget('aP', 1, scale) == 6000
    # the same number the global scale gives, so a per-family run and a global one agree
    assert lattice_budget('oP', 1, {'oP': 2}) == lattice_budget('oP', 2, {}) == 8000


def test_a_budget_scale_naming_an_unknown_lattice_is_refused():
    with pytest.raises(ValueError, match='unknown Bravais lattices'):
        lattice_budget('cP', 1, {'cubic': 0.5})


def test_a_run_can_name_fractions_for_one_lattice_and_the_rest_keep_their_row():
    from mlindex.optimization.UtilitiesOptimizer import lattice_fractions

    corner = {'trees': 1.0, 'abnn': 0.0, 'templates': 0.0}
    assert lattice_fractions('cP', {'cP': corner}) == corner
    assert lattice_fractions('aP', {'cP': corner}) == ENSEMBLE['aP']['fractions']
    with pytest.raises(ValueError, match='unknown Bravais lattices'):
        lattice_fractions('cP', {'cubic': corner})


def test_an_arm_refuses_half_edited_fractions_before_it_indexes_anything():
    from mlindex.model_training.BenchmarkRuns import ensemble_record

    with pytest.raises(ValueError, match='sum to 1'):
        ensemble_record({}, True, {'cP': {'trees': 0.50, 'abnn': 0.45, 'templates': 0.10}}, {})
    record = ensemble_record({'cP': 0.5}, False, {}, {'oP': (10, 1e-4)})
    assert record['redistribute'] is False
    assert record['lattices']['cP']['n_candidates'] == 50
    assert record['lattices']['aP'] == {'n_candidates': 6000,
                                        'fractions': ENSEMBLE['aP']['fractions'],
                                        'max_neighbors': 23, 'neighbor_radius': 0.000679}
    assert (record['lattices']['oP']['max_neighbors'],
            record['lattices']['oP']['neighbor_radius']) == (10, 1e-4)


def test_fractions_on_the_command_line():
    from mlindex.scripts.run_benchmark import _parse_fractions

    assert _parse_fractions(['cP=0.67,0.10,0.23']) == {
        'cP': {'trees': 0.67, 'abnn': 0.10, 'templates': 0.23}}
    for bad in (['cP=0.5,0.5'], ['cubic=1,0,0'], ['cP']):
        with pytest.raises(ValueError):
            _parse_fractions(bad)


# The (max_neighbors, neighbor_radius) each factory set before the constants moved into ENSEMBLE.
FACTORY_REDISTRIBUTION = {
    'cubic': (64, 0.000026), 'tetragonal': (52, 0.000213), 'hexagonal': (52, 0.000213),
    'rhombohedral': (52, 0.000213), 'orthorhombic': (46, 0.000338),
    'monoclinic': (42, 0.000547), 'triclinic': (23, 0.000679),
    }


@pytest.mark.parametrize('bravais_lattice', BRAVAIS_LATTICES)
def test_redistribution_constants_match_the_factories_they_replaced(bravais_lattice):
    from mlindex.optimization.UtilitiesOptimizer import lattice_redistribution
    expected = FACTORY_REDISTRIBUTION[BL_TO_LATTICE_SYSTEM[bravais_lattice]]
    assert lattice_redistribution(bravais_lattice, {}) == expected


def test_a_run_can_name_redistribution_constants_for_one_lattice():
    from mlindex.optimization.UtilitiesOptimizer import lattice_redistribution
    assert lattice_redistribution('oP', {'oP': (10, 1e-4)}) == (10, 1e-4)
    assert lattice_redistribution('aP', {'oP': (10, 1e-4)}) == (23, 0.000679)
    for bad in ({'orthorhombic': (10, 1e-4)}, {'oP': (0, 1e-4)}, {'oP': (2.5, 1e-4)},
                {'oP': (10, -1.0)}):
        with pytest.raises(ValueError):
            lattice_redistribution('oP', bad)


def test_redistribution_constants_on_the_command_line():
    from mlindex.scripts.run_benchmark import _parse_redistribution

    assert _parse_redistribution(['oP=10,0.0001']) == {'oP': (10, 1e-4)}
    for bad in (['oP=10'], ['orthorhombic=10,0.0001'], ['oP']):
        with pytest.raises(ValueError):
            _parse_redistribution(bad)
