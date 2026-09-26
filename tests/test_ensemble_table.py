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


# The fractions the literals above encode -- the ones shipped before P09c moved ENSEMBLE to P09b's.
FACTORY_FRACTIONS = {system: dict(zip(('trees', 'abnn', 'templates'), shares)) for system, shares in {
    'cubic': (0.45, 0.45, 0.10), 'tetragonal': (0.05, 0.70, 0.25), 'hexagonal': (0.05, 0.70, 0.25),
    'rhombohedral': (0.05, 0.70, 0.25), 'orthorhombic': (0.05, 0.70, 0.25),
    'monoclinic': (0.05, 0.55, 0.40), 'triclinic': (0.05, 0.40, 0.55)}.items()}


def _derived(bravais_lattice, scale):
    """What OptimizerManager builds at this scale, at the fractions the literals were written for."""
    return generator_info_from_fractions(
        FACTORY_FRACTIONS[BL_TO_LATTICE_SYSTEM[bravais_lattice]],
        lattice_budget(bravais_lattice, scale),
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


def test_the_manifest_records_every_lattice_s_budget_and_fractions():
    from mlindex.model_training.BenchmarkRuns import ensemble_record

    record = ensemble_record()
    assert set(record['lattices']) == set(BRAVAIS_LATTICES)
    assert record['lattices']['aP'] == {'n_candidates': 6000,
                                        'fractions': ENSEMBLE['aP']['fractions']}
