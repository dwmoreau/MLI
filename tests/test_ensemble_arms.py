"""The P09c run list: every run is a valid run_benchmark command, and none is run where it changes
nothing."""
import pytest

from mlindex.scripts import ensemble_arms, run_benchmark
from mlindex.utilities.Allocation import check_generator_fractions
from mlindex.utilities.UnitCellTools import BRAVAIS_LATTICES


def test_the_old_fractions_cover_every_lattice_and_each_row_sums_to_one():
    assert set(ensemble_arms.OLD_FRACTIONS) == set(BRAVAIS_LATTICES)
    for trees, abnn, templates in ensemble_arms.OLD_FRACTIONS.values():
        check_generator_fractions({'trees': trees, 'abnn': abnn, 'templates': templates})


@pytest.mark.parametrize('index', range(len(ensemble_arms.jobs())))
def test_every_run_is_a_command_run_benchmark_accepts(index):
    population, run = ensemble_arms.jobs()[index]
    argv = ensemble_arms.generate_argv(population, run, 'pools', 'tables', 'split.parquet', 4)
    args = run_benchmark.build_parser().parse_args(argv)
    assert args.population == population
    assert args.out_pool.endswith(f'{population}_{run}')
    fractions = run_benchmark._parse_fractions(args.fractions)
    if run == 'old_fractions':
        assert fractions['aP'] == {'trees': 0.05, 'abnn': 0.40, 'templates': 0.55}
        assert set(fractions) == set(BRAVAIS_LATTICES)
    else:
        assert fractions == {}
    scale = run_benchmark._parse_budget_scale(args.budget_scale)
    if run.startswith('budget_half_cubic'):
        assert scale == {'cF': 0.5, 'cI': 0.5, 'cP': 0.5}
    elif not run.startswith('budget_'):
        assert scale == {}


def test_the_hard_population_gets_only_the_runs_that_touch_its_lattices():
    hard = [run for population, run in ensemble_arms.jobs() if population == 'hard']
    assert hard == ['control', 'old_fractions', 'budget_half_monoclinic', 'budget_double_monoclinic',
                    'budget_half_triclinic', 'budget_double_triclinic']
    general = [run for population, run in ensemble_arms.jobs() if population == 'general']
    assert general == list(ensemble_arms.RUNS)


def test_the_submit_script_array_covers_every_run():
    from pathlib import Path
    script = Path(ensemble_arms.__file__).with_name('submit_ensemble_arms.sh')
    text = script.read_text(encoding='utf-8')
    assert f'#SBATCH --array=0-{len(ensemble_arms.batches()["fractions"]) - 1}\n' in text


def test_the_batches_hold_every_run_once():
    runs = [job for batch in ensemble_arms.batches().values() for job in batch]
    assert sorted(runs) == sorted(ensemble_arms.jobs())


def test_a_task_refuses_an_array_that_does_not_fit_its_batch():
    with pytest.raises(SystemExit, match='submit it with --array=0-17'):
        ensemble_arms.main(['generate', '--batch', 'budget', '--task', '0', '--array-size', '4',
                            '--pools-dir', 'p', '--tables-dir', 't', '--split-manifest', 's'])


def test_every_comparison_names_runs_that_exist():
    for _, reference, arms in ensemble_arms.QUESTIONS:
        assert reference in ensemble_arms.RUNS
        assert set(arms) <= set(ensemble_arms.RUNS)

