"""The P09c run list: every run is a valid run_benchmark command, and none is run where it changes
nothing."""
import pytest

from mlindex.scripts import ensemble_arms, run_benchmark
from mlindex.utilities.Allocation import check_generator_fractions
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES


def test_the_old_settings_cover_every_lattice_and_are_valid():
    from mlindex.optimization.UtilitiesOptimizer import lattice_redistribution
    assert set(ensemble_arms.OLD_FRACTIONS) == set(BRAVAIS_LATTICES)
    for trees, abnn, templates in ensemble_arms.OLD_FRACTIONS.values():
        check_generator_fractions({'trees': trees, 'abnn': abnn, 'templates': templates})
    assert set(ensemble_arms.OLD_REDISTRIBUTION) == set(BRAVAIS_LATTICES)
    for lattice, pair in ensemble_arms.OLD_REDISTRIBUTION.items():
        assert lattice_redistribution(lattice, {lattice: pair}) == pair


@pytest.mark.parametrize('index', range(len(ensemble_arms.jobs())))
def test_every_run_is_a_command_run_benchmark_accepts(index):
    population, run = ensemble_arms.jobs()[index]
    argv = ensemble_arms.generate_argv(population, run, 'pools', 'tables', 'split.parquet', 4)
    args = run_benchmark.build_parser().parse_args(argv)
    assert args.population == population
    assert args.out_pool.endswith(f'{population}_{run}')
    fractions = run_benchmark._parse_fractions(args.fractions)
    redistribution = run_benchmark._parse_redistribution(args.redistribution)
    if run in ('old_fractions', 'shipped_before_p09c'):
        assert fractions['aP'] == {'trees': 0.05, 'abnn': 0.40, 'templates': 0.55}
        assert set(fractions) == set(BRAVAIS_LATTICES)
    else:
        assert fractions == {}
    if run in ('old_redistribution', 'shipped_before_p09c'):
        assert redistribution == ensemble_arms.OLD_REDISTRIBUTION
    else:
        assert redistribution == {}
    assert args.no_redistribution == (run == 'redistribution_off')
    scale = run_benchmark._parse_budget_scale(args.budget_scale)
    if run.startswith('budget_half_cubic'):
        assert scale == {'cF': 0.5, 'cI': 0.5, 'cP': 0.5}
    elif not run.startswith('budget_'):
        assert scale == {}


def test_the_hard_population_gets_only_the_runs_that_touch_its_lattices():
    hard = [run for population, run in ensemble_arms.jobs() if population == 'hard']
    assert hard == ['control', 'old_fractions', 'old_redistribution', 'redistribution_off',
                    'shipped_before_p09c',
                    'budget_half_monoclinic', 'budget_double_monoclinic',
                    'budget_half_triclinic', 'budget_double_triclinic']
    general = [run for population, run in ensemble_arms.jobs() if population == 'general']
    assert general == list(ensemble_arms.RUNS)


def test_the_submit_script_array_covers_every_run():
    from pathlib import Path
    script = Path(ensemble_arms.__file__).with_name('submit_ensemble_arms.sh')
    text = script.read_text(encoding='utf-8')
    assert f'#SBATCH --array=0-{len(ensemble_arms.jobs()) - 1}\n' in text


def test_every_comparison_names_runs_that_exist():
    for _, reference, arms in ensemble_arms.QUESTIONS:
        assert reference in ensemble_arms.RUNS
        assert set(arms) <= set(ensemble_arms.RUNS)


def test_nothing_is_generated_while_ensemble_holds_the_old_constants(monkeypatch):
    from mlindex.optimization import UtilitiesOptimizer
    old = {lattice: dict(row, max_neighbors=ensemble_arms.OLD_REDISTRIBUTION[lattice][0],
                         neighbor_radius=ensemble_arms.OLD_REDISTRIBUTION[lattice][1])
           for lattice, row in UtilitiesOptimizer.ENSEMBLE.items()}
    monkeypatch.setattr(UtilitiesOptimizer, 'ENSEMBLE', old)
    with pytest.raises(SystemExit, match='old redistribution constants'):
        ensemble_arms.main(['generate', '--index', '0', '--pools-dir', 'p', '--tables-dir', 't',
                            '--split-manifest', 's'])
