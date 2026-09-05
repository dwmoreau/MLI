"""S14's results document is generated from its tables; these pin that the generator reads them
the way the driver writes them, on a synthetic artefact directory."""
import numpy as np
import pandas as pd
import pytest

from mlindex.scripts import run_fom_neural_report as report

ARMS = ('network', 'tree', 'tree_fullscale', 'M_sym', 'M20', 'drop_A', 'drop_B',
        'label_shuffled', 'constant', 'uniform_random')


def _artifacts(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for arm in ARMS:
        row = dict(arm=arm, operating_point=rng.random(), top10=rng.random(),
                   threshold_only=rng.random(), precision=rng.random(), reported=rng.random(),
                   hard_top10=0.05, hard_n_entries=20, n_features=52)
        for lattice in report.LATTICES:
            row[f'dev_top10_{lattice}'] = rng.random()
            row[f'dev_op_{lattice}'] = rng.random()
            row[f'dev_n_{lattice}'] = 120
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path/'S14_neural_main_table.csv', index=False)

    def pair(reference, arm, metric, scope):
        return dict(reference=reference, arm=arm, metric=metric, scope=scope,
                    delta_pp=float(rng.normal()), ci_low_pp=-1.0, ci_high_pp=1.0, gained=3,
                    lost=2, p_value=0.5, method='exact')

    pd.DataFrame([pair('network', arm, metric, 'aggregate') for arm in ARMS if arm != 'network'
                  for metric in ('operating_point', 'top10', 'threshold_only')]).to_csv(
        tmp_path/'S14_neural_contrasts.csv', index=False)
    pd.DataFrame([pair(reference, 'network', metric, 'aggregate')
                  for reference in report.REFERENCES
                  for metric in ('operating_point', 'top10', 'threshold_only')]).to_csv(
        tmp_path/'S14_neural_mcnemar.csv', index=False)
    by_lattice = [dict(pair(reference, 'network', metric, f'lattice={lattice}'), n_entries=120)
                  for reference in ('tree', 'M_sym') for lattice in report.LATTICES
                  for metric in ('top10', 'operating_point')]
    pd.DataFrame(by_lattice).to_csv(tmp_path/'S14_neural_by_lattice_mcnemar.csv', index=False)
    pd.DataFrame([dict(arm=arm, top10=0.8, op_at_75=0.6, precision_at_75=0.7, op_at_90=0.7,
                       precision_at_90=0.6, reported_at_75=0.75, reported_at_90=0.9)
                  for arm in ARMS]).to_csv(tmp_path/'S14_neural_answer_rates.csv', index=False)
    pd.DataFrame([dict(arm=arm, kind='network', n_features=52, n_rows_fit=363421,
                       n_positive_fit=6221, purpose='x') for arm in ARMS]).to_csv(
        tmp_path/'S14_neural_fit_table.csv', index=False)
    pd.DataFrame([dict(merit='M_sym', baseline='M20', metric='top10', bravais_lattice=lattice,
                       contrast_pp=1.0, se_pp=2.0) for lattice in report.LATTICES]).to_csv(
        tmp_path/'S08_floor_by_lattice.csv', index=False)
    pd.DataFrame([dict(model='shipped_11', readout=readout, bravais_lattice=lattice,
                       in_support=not lattice.startswith('c'), n=100, precision=0.5, recall=0.5,
                       f1=0.5, predicted_share=0.07, true_share=0.07,
                       median_log_probability=-19.0 if lattice.startswith('c') else -4.0,
                       median_rank=12 if lattice.startswith('c') else 3,
                       max_probability=3.9e-5 if lattice.startswith('c') else 0.9)
                  for readout in ('raw_head', 'support_masked')
                  for lattice in report.LATTICES]).to_csv(
        tmp_path/'S14_prior_interface.csv', index=False)
    return tmp_path


def test_the_document_leads_with_the_verdict_and_reports_rank_and_threshold_apart(tmp_path):
    artifacts = _artifacts(tmp_path)
    text = report.build(artifacts, 'S14_neural')
    assert text.startswith('# S14 — the neural scoring network\n\n## The verdict')
    assert '## Rank and threshold, reported separately' in text
    assert 'threshold only' in text and 'top-10 (rank)' in text
    assert "S12's tree refitted on the same rows" in text
    assert "S12's shipped full-scale tree" in text
    assert '## Super-additivity' in text
    assert 'drop_B' in text and 'drop_A' in text


def test_per_lattice_rows_carry_each_lattices_own_floor(tmp_path):
    artifacts = _artifacts(tmp_path)
    text = report.build(artifacts, 'S14_neural')
    assert 'Paired against `tree`, top-10' in text
    assert '| aP | 120 |' in text
    floors = report.lattice_floors(artifacts)
    assert floors['aP'] == 2.0 and len(floors) == 14


def test_the_interface_section_flags_the_untrained_classes(tmp_path):
    artifacts = _artifacts(tmp_path)
    text = report.build(artifacts, 'S14_neural')
    assert '| cP | **no** |' in text
    assert 'macro F1 over the support' in text


def test_missing_optional_tables_do_not_break_the_document(tmp_path):
    artifacts = _artifacts(tmp_path)
    for name in ('contrasts', 'mcnemar', 'by_lattice_mcnemar', 'answer_rates', 'fit_table'):
        (artifacts/f'S14_neural_{name}.csv').unlink()
    (artifacts/'S14_prior_interface.csv').unlink()
    text = report.build(artifacts, 'S14_neural')
    assert 'One fit seed so far' in text
    assert 'Not computed.' in text


def test_the_figure_writes(tmp_path):
    pytest.importorskip('matplotlib')
    artifacts = _artifacts(tmp_path)
    main = pd.read_csv(artifacts/'S14_neural_main_table.csv').set_index('arm')
    by_lattice = pd.read_csv(artifacts/'S14_neural_by_lattice_mcnemar.csv')
    path = report.figure(artifacts, 'S14_neural_score', main, by_lattice,
                         report.lattice_floors(artifacts))
    assert path.exists() and path.stat().st_size > 0
