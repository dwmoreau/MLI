"""S15's logic, without a pool.

The driver's job is to run the real indexer at several cuts and score several merits over what it
produced, with nothing tuned and nothing paired across different peak lists. What is pinned here
is the part that fails silently: the design's completeness, the invocation, the provenance stamp,
the digest check, the floor arithmetic, the axis extraction, the restriction replay and the menu.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mlindex.model_training import FomConditions
from mlindex.model_training import FomEndToEnd as E2E
from mlindex.model_training import FomMetrics
from mlindex.scripts import run_fom_end_to_end as driver


# ---------------------------------------------------------------------------------------------
# design and naming
# ---------------------------------------------------------------------------------------------
def test_the_design_covers_the_grid_and_the_existing_arm():
    design = E2E.build_design()
    generated = design.loc[design['source'] == 'e2e']
    assert generated.loc[generated['population'] == 'general', 'n_cells'].sum() == 530*9*3
    assert generated.loc[generated['population'] == 'hard', 'n_cells'].sum() == 360*5*3
    hard = set(generated.loc[generated['population'] == 'hard', 'condition_bundle'])
    assert hard == set(FomConditions.HARD_BUNDLES)
    existing = design.loc[design['source'] == 'existing']
    assert set(existing['cut']) == {1.5} and set(existing['population']) == {'general'}
    assert existing.shape[0] == 8 and E2E.ERROR_SHAPE_TAG not in set(existing['condition_bundle'])
    assert design['is_real_run'].all()
    assert not design.duplicated(['population', 'cut', 'condition_bundle']).any()


def test_directory_names_are_deterministic_and_use_the_g_form():
    assert E2E.cut_label(5.0) == 'cut5' and E2E.cut_label(3.5) == 'cut3.5' and E2E.cut_label(5) == 'cut5'
    root = '/tmp/x'
    assert str(E2E.arm_dir(root, 'general', 5.0)).endswith('e2e/general/cut5')
    assert str(E2E.pool_dir(root, 'hard', 3.0)).endswith('e2e/hard/cut3_pool')
    assert str(E2E.bundle_dir(root, 'general', 3.5, 'c2_error1_cont0')).endswith(
        'e2e/general/cut3.5/c2_error1_cont0')


def test_the_generation_argv_is_a_real_fully_retained_timed_run_at_the_cut():
    argv = E2E.generate_argv('general', 3.5, 'nominal', '/root', 64, '/m.parquet', '/e.csv',
                             extra_opt_params={'hkl_source': 'posterior'})
    joined = ' '.join(argv)
    assert '--prune-threshold 3.5' in joined
    assert '--pool-size 2' in joined and '--optimizer-seed 12345' in joined and '--seed 12345' in joined
    assert '--no-subsample' in argv and '--predownsample-entries 0' in joined
    assert '--record-timing' in argv
    assert '--opt-param hkl_source=posterior' in joined
    assert argv[argv.index('--out-dir') + 1].endswith('e2e/general/cut3.5/c2_error1_cont0')
    # The argv parses through the generator's own parser, so a renamed flag fails here.
    from mlindex.scripts.run_fom_dump import _parse_args
    parsed = _parse_args(argv)
    assert parsed.prune_threshold == 3.5 and parsed.no_subsample and parsed.record_timing


def test_an_arm_without_the_completion_stamp_is_refused(tmp_path):
    E2E.write_provenance(tmp_path, population='general', cut=5.0)
    with pytest.raises(SystemExit, match='INCOMPLETE'):
        E2E.load_provenance(tmp_path)
    E2E.stamp_complete(tmp_path, bundle_seconds_total={'a': 1.0})
    payload = E2E.load_provenance(tmp_path)
    assert payload['complete'] and payload['cut'] == 5.0 and 'commit' in payload
    with pytest.raises(SystemExit, match='no provenance'):
        E2E.load_provenance(tmp_path/'nowhere')


# ---------------------------------------------------------------------------------------------
# sidecars and scores
# ---------------------------------------------------------------------------------------------
def test_sidecar_commands_add_the_slow_passes_only_when_a_model_reads_them():
    groups = ('raw', 'structural', 'context', 'counts', 'probation')
    commands = E2E.sidecar_commands('/arm', '/pool', 8, groups, python='py')
    scripts = [c[1].split('/')[-1] for c in commands]
    assert scripts[0] == 'run_fom_dump_consolidate.py'
    assert 'run_fom_holdout_merits.py' not in scripts
    assert not any('--soft' in c for c in commands)
    assert sum('--verify' in c for c in commands) == 2
    with_holdout = E2E.sidecar_commands('/arm', '/pool', 8, groups + ('holdout', 'soft'), python='py')
    scripts = [c[1].split('/')[-1] for c in with_holdout]
    assert 'run_fom_holdout_merits.py' in scripts and any('--soft' in c for c in with_holdout)
    assert sum('--verify' in c for c in with_holdout) == 3


def test_the_learned_groups_come_from_the_specification(tmp_path):
    (tmp_path/'specification.json').write_text(json.dumps({'groups': ['raw', 'holdout']}),
                                               encoding='utf-8')
    groups = E2E.learned_groups({'x': tmp_path})
    assert 'holdout' in groups and 'raw' in groups and 'structural' in groups


def test_the_report_split_guard_refuses_a_fit_crystal():
    entries = pd.DataFrame({'entry_id': ['A', 'B'], 'split': ['fom-dev', 'fom-train']})
    with pytest.raises(SystemExit, match='fom-train'):
        E2E.assert_report_split(entries)
    E2E.assert_report_split(entries.iloc[:1])


def test_the_reference_scores_are_the_two_merits_and_the_two_floors():
    scores = E2E.reference_scores(1)
    assert scores['M20'] == 'M20' and scores['M_sym'] == 'M_sym'
    frame = pd.DataFrame({'x': np.arange(5)})
    assert np.all(scores['constant'](frame) == 1.0)
    assert scores['uniform_random'](frame).shape == (5,)


# ---------------------------------------------------------------------------------------------
# gate 5
# ---------------------------------------------------------------------------------------------
def _entries(digests, bundle='c2_error1_cont0', partner=None):
    return pd.DataFrame({
        'entry_id': [f'E{i}' for i in range(len(digests))],
        'condition_bundle': bundle,
        'q2_digest': digests,
        'n_dropout_achieved': 0,
        'second_phase_partner': partner if partner is not None else [None]*len(digests),
        })


def test_the_digest_check_passes_on_identical_peak_lists_and_names_the_cell_that_differs():
    same = _entries(['a', 'b', 'c'])
    table = E2E.check_peak_digests({5.0: same, 3.0: same.copy(), 1.5: same.copy()})
    assert table['n_agree'].sum() == 3 and table['n_disagree'].sum() == 0
    assert table['cuts_compared'].iloc[0] == 'cut1.5,cut3,cut5'
    different = _entries(['a', 'b', 'X'])
    with pytest.raises(ValueError, match='E2'):
        E2E.check_peak_digests({5.0: same, 3.0: different})
    with pytest.raises(ValueError, match='missing'):
        E2E.check_peak_digests({5.0: same, 3.0: same.iloc[:2]})
    # A whole bundle one arm never generated is a note, not a failure: cut 1.5 has no error_shape.
    extra = pd.concat([same, _entries(['p', 'q', 'r'], bundle='c2_error1_cont0_icept4')],
                      ignore_index=True)
    table = E2E.check_peak_digests({5.0: extra, 1.5: same})
    absent = table.loc[table['note'] != '']
    assert absent.shape[0] == 1 and absent['condition_bundle'].iloc[0] == 'c2_error1_cont0_icept4'
    assert 'cut1.5' in absent['note'].iloc[0]
    assert table.loc[table['note'] == '', 'n_agree'].sum() == 3
    witness = _entries(['a', 'b', 'c'])
    witness.loc[1, 'n_dropout_achieved'] = 4
    with pytest.raises(ValueError, match='E1'):
        E2E.check_peak_digests({5.0: same, 3.0: witness})


def test_the_manifest_identity_check_allows_a_different_cut_and_commit_but_nothing_else():
    base = dict(seed=12345, optimizer_seed=12345, pool_size=2, search_seed_scheme='per_entry_bravais',
                split_manifest_sha256='abc', arch='x86_64', broadening_tag='1', iteration_scale=1.0,
                commit='c1', prune_threshold=5.0)
    other = dict(base, prune_threshold=3.0, commit='c2')
    table = E2E.check_manifest_identity({5.0: base, 3.0: other})
    assert table.shape[0] == 2
    consolidated = dict(bundle_manifests={'a': dict(base), 'b': dict(base)})
    E2E.check_manifest_identity({5.0: consolidated, 3.0: other})
    with pytest.raises(ValueError, match='pool_size'):
        E2E.check_manifest_identity({5.0: base, 3.0: dict(other, pool_size=4)})
    with pytest.raises(ValueError, match='arch'):
        E2E.check_manifest_identity({5.0: base, 3.0: dict(other, arch='arm64')})


# ---------------------------------------------------------------------------------------------
# floors and axes
# ---------------------------------------------------------------------------------------------
def test_floor_standard_errors_use_the_metric_and_the_lattice(tmp_path):
    pd.DataFrame([dict(merit='M_sym', baseline='M20', metric='top10', floor_pp=0.5086),
                  ]).to_csv(tmp_path/'S08_floor_contrast.csv', index=False)
    pd.DataFrame([dict(merit='M_sym', baseline='M20', metric='top10', bravais_lattice='cF', se_pp=1.78),
                  dict(merit='M_sym', baseline='M20', metric='top10', bravais_lattice='aP', se_pp=2.85),
                  dict(merit='M_sym', baseline='M20', metric='top10', bravais_lattice='mP', se_pp=2.0),
                  dict(merit='M_sym', baseline='M20', metric='top10', bravais_lattice='mC', se_pp=1.0),
                  ]).to_csv(tmp_path/'S08_floor_by_lattice.csv', index=False)
    pd.DataFrame([dict(merit='M_sym', baseline='M20', metric='operating_point', floor_pp=0.411),
                  ]).to_csv(tmp_path/'S09_floor_op_contrast.csv', index=False)
    pd.DataFrame([dict(merit='M_sym', baseline='M20', metric='operating_point', bravais_lattice='aP',
                       se_pp=2.6)]).to_csv(tmp_path/'S09_floor_op_by_lattice.csv', index=False)
    floors = E2E.load_floor_tables(tmp_path)
    assert floors['top10'][0] == pytest.approx(0.5086)
    assert floors['operating_point'][0] == pytest.approx(0.411)
    assert E2E.floor_for(floors, 'top10', 'aggregate')[0] == pytest.approx(0.5086)
    assert E2E.floor_for(floors, 'top10', 'bravais_lattice=cF')[0] == pytest.approx(1.78)
    assert E2E.floor_for(floors, 'operating_point', 'bravais_lattice=aP')[0] == pytest.approx(2.6)
    value, source = E2E.floor_for(floors, 'top10', 'hard')
    assert value == pytest.approx((2.85 + 1.0 + 2.0)/3) and 'aP/mC/mP' in source
    assert np.isnan(E2E.floor_for(floors, 'top10', 'bravais_lattice=zz')[0])
    assert E2E.in_floor_ses(5.744, 0.5086) == pytest.approx(11.29, abs=0.01)
    assert np.isnan(E2E.in_floor_ses(1.0, np.nan))


def test_the_success_curve_axes_are_read_off_the_condition_table():
    axes = E2E.success_curve_axes()
    error = axes.loc[axes['axis'] == 'error_scale'].sort_values('x')
    assert error['x'].tolist() == [0.1, 1.0, 2.0]
    contaminants = axes.loc[axes['axis'] == 'contaminant_count'].sort_values('x')
    assert contaminants['x'].tolist() == [0, 1, 2]
    assert contaminants.loc[contaminants['x'] == 1, 'caveat'].iloc[0]
    dropout = axes.loc[axes['axis'] == 'dropout'].sort_values('x')
    assert dropout['x'].tolist() == [0, 2, 4, 6]
    assert dropout.loc[dropout['x'] == 0, 'caveat'].iloc[0]
    for tag in axes['condition_bundle']:
        assert tag in FomConditions.BY_TAG
    assert E2E.ERROR_SHAPE_TAG not in set(axes['condition_bundle'])
    assert FomConditions.BY_KEY['second_phase'].tag not in set(axes['condition_bundle'])


# ---------------------------------------------------------------------------------------------
# the restriction replay
# ---------------------------------------------------------------------------------------------
def test_restriction_keeps_the_cut_the_fallback_and_recomputes_the_pool_position():
    frame = pd.DataFrame({
        'entry_id': ['A']*6,
        'condition_bundle': ['b']*6,
        'bravais_lattice': ['cP', 'cP', 'cP', 'aP', 'aP', 'oP'],
        'candidate_id': [0, 1, 2, 0, 1, 0],
        'm20_at_prune': [6.0, 3.0, 1.6, 2.0, 2.5, 9.0],
        'M20': [10.0, 12.0, 4.0, 3.0, 2.0, 30.0],
        'final_rank': [1, 0, 2, 0, 1, 0],
        'in_top_n': [True]*6,
        'pool_size_full': [6.0]*6,
        })
    out = E2E.restrict_at_cut(frame, 3.0, n_top=1)
    kept = set(zip(out['bravais_lattice'], out['candidate_id']))
    # cP keeps the two at or above 3.0; aP has none, so keeps its best (2.5); oP keeps its one.
    assert kept == {('cP', 0), ('cP', 1), ('aP', 1), ('oP', 0)}
    cp = out.loc[out['bravais_lattice'] == 'cP'].set_index('candidate_id')
    assert cp.loc[1, 'final_rank'] == 0 and cp.loc[0, 'final_rank'] == 1
    assert bool(cp.loc[1, 'in_top_n']) and not bool(cp.loc[0, 'in_top_n'])
    assert (out['pool_size_full'] == 4.0).all()
    with pytest.raises(KeyError, match='m20_at_prune'):
        E2E.restrict_at_cut(frame.drop(columns=['m20_at_prune']), 3.0)


# ---------------------------------------------------------------------------------------------
# levels, alignment and contrasts on a real reduction
# ---------------------------------------------------------------------------------------------
def _tiny(rows, bundle='c2_error1_cont0'):
    """(entry_id, lattice, score, is_correct) tuples -> a candidate frame and its entry table."""
    records, counter = [], {}
    for entry_id, lattice, score, is_correct in rows:
        counter[(entry_id, lattice)] = counter.get((entry_id, lattice), -1) + 1
        records.append(dict(entry_id=entry_id, condition_bundle=bundle, bravais_lattice=lattice,
                            candidate_id=counter[(entry_id, lattice)], score=float(score),
                            other=float(score) - 15.0, is_correct=bool(is_correct),
                            is_off_by_two=False, is_degenerate=False, in_top_n=True))
    candidates = pd.DataFrame(records)
    candidates['is_degenerate'] = candidates['is_degenerate'].astype('boolean')
    entries = pd.DataFrame([dict(entry_id=e, condition_bundle=bundle, split='fom-dev',
                                 bravais_lattice_true='oP', lattice_system_true='orthorhombic',
                                 volume_true=1000.0 + 10*i)
                            for i, e in enumerate(dict.fromkeys(c['entry_id'] for c in records))])
    return candidates, entries


def _reduce(candidates, entries, score):
    per_entry, _, meta = FomMetrics.reduce_to_per_entry(candidates, score=score, entries=entries)
    return per_entry, meta


def test_levels_alignment_and_the_contrast_sign_convention():
    rows = [(f'E{i}', 'oP', 20.0, True) for i in range(12)]
    candidates, entries = _tiny(rows)
    per_a, meta_a = _reduce(candidates, entries, 'score')
    per_b, meta_b = _reduce(candidates, entries, 'other')
    a = E2E.summarise(per_a, meta_a, 10.0, 'all', n_bootstrap=20)
    b = E2E.summarise(per_b, meta_b, 10.0, 'all', n_bootstrap=20)
    # `score` is 20 > 10 on every entry; `other` is 5 < 10 on every entry: A wins 12 / 0.
    row = E2E.contrast(b, a, 'operating_point')
    assert row['gained'] == 12 and row['lost'] == 0 and row['delta_pp'] == pytest.approx(100.0)
    row = E2E.contrast(a, b, 'operating_point')
    assert row['gained'] == 0 and row['lost'] == 12 and row['delta_pp'] == pytest.approx(-100.0)
    level = E2E.level_row(a, 'aggregate', population='general', cut=5.0, merit='x', pool_subset='all')
    assert level['operating_point'] == pytest.approx(1.0) and level['n_entries'] == 12
    assert level['scope'] == 'aggregate'
    strata = E2E.stratum_rows(a, 'bravais_lattice', population='general')
    assert strata and strata[0]['scope'] == 'bravais_lattice=oP'
    # Alignment: drop two cells from one side and both results are re-summarised on the rest.
    common = E2E.common_keys([per_a, per_b.iloc[:10]])
    assert len(common) == 10
    restricted, dropped = E2E.restrict_per_entry(per_a, common)
    assert restricted.shape[0] == 10 and dropped == 2
    # An empty stratum yields no row rather than a NaN one.
    assert E2E.contrast(a, b, 'operating_point', mask=np.zeros(12, dtype=bool)) is None
    masks = E2E.scope_masks(a)
    assert 'aggregate' in masks and 'bravais_lattice=oP' in masks
    assert f'condition_bundle=c2_error1_cont0' in masks
    assert masks['bravais_lattice=oP'].sum() == 12


# ---------------------------------------------------------------------------------------------
# the menu
# ---------------------------------------------------------------------------------------------
def _levels_and_contrasts():
    levels, contrasts = [], []
    ops = {('general', 5.0, 'M20'): 0.60, ('general', 3.0, 'M20'): 0.58,
           ('general', 5.0, 'M_sym'): 0.70, ('general', 3.0, 'M_sym'): 0.75,
           ('hard', 5.0, 'M20'): 0.05, ('hard', 3.0, 'M20'): 0.04,
           ('hard', 5.0, 'M_sym'): 0.08, ('hard', 3.0, 'M_sym'): 0.02}
    for (population, cut, merit), op in ops.items():
        levels.append(dict(population=population, cut=cut, merit=merit, pool_subset='in_top_n',
                           scope='aggregate', operating_point=op, top10=op + 0.1,
                           ceiling_rescorer=0.9))
        if (cut, merit) != (5.0, 'M20'):
            delta = 100*(op - ops[(population, 5.0, 'M20')])
            contrasts.append(dict(population=population, contrast_kind='pair', reference_cut=5.0,
                                  reference_merit='M20', cut=cut, merit=merit, metric='operating_point',
                                  pool_subset='in_top_n', scope='aggregate', delta_pp=delta,
                                  ci_low_pp=delta - 1, ci_high_pp=delta + 1, p_value=0.01,
                                  standard_errors=delta/0.5))
            if population == 'general':
                contrasts.append(dict(population='general', contrast_kind='pair', reference_cut=5.0,
                                      reference_merit='M20', cut=cut, merit=merit,
                                      metric='operating_point', pool_subset='in_top_n',
                                      scope='bravais_lattice=aP', delta_pp=-3.0, ci_low_pp=-4,
                                      ci_high_pp=-2, p_value=0.05, standard_errors=-1.2))
    return pd.DataFrame(levels), pd.DataFrame(contrasts)


def test_the_menu_applies_the_stated_rule_and_marks_the_incumbent():
    levels, contrasts = _levels_and_contrasts()
    cost = pd.DataFrame([dict(population='general', cut=5.0, condition_bundle='all',
                              seconds_per_entry_median=40.0, pool_size_full_median=100.0),
                         dict(population='general', cut=3.0, condition_bundle='all',
                              seconds_per_entry_median=50.0, pool_size_full_median=1000.0)])
    menu = E2E.build_menu(levels, contrasts, cost=cost, merits=('M20', 'M_sym'))
    assert menu.shape[0] == 4
    incumbent = menu.loc[menu['is_incumbent']]
    assert incumbent.shape[0] == 1 and np.isnan(incumbent['general_delta_pp_vs_incumbent'].iloc[0])
    # M_sym at 3.0 has the largest general operating point but its hard population falls by
    # 6 pp = 12 se below the incumbent, so the rule picks 5.0 for M_sym.
    rec = menu.loc[menu['recommended']].set_index('merit')
    assert rec.loc['M_sym', 'cut'] == 5.0
    assert rec.loc['M20', 'cut'] == 5.0
    row = menu.set_index(['cut', 'merit']).loc[(3.0, 'M_sym')]
    assert row['worst_lattice'] == 'aP' and row['worst_lattice_standard_errors'] == pytest.approx(-1.2)
    assert row['seconds_per_entry'] == 50.0 and row['seconds_vs_incumbent_pct'] == pytest.approx(25.0)
    assert (menu['rule'] == E2E.MENU_RULE).all()


# ---------------------------------------------------------------------------------------------
# the driver's argument surface
# ---------------------------------------------------------------------------------------------
def test_the_driver_defaults_to_s12s_full_scale_model_and_parses_learned_arms():
    args = driver._parse_args(['--stage', 'reduce', '--cut', '5.0'])
    learned = driver._learned(args)
    assert list(learned) == ['plus_probation'] and learned['plus_probation'].endswith(
        'plus_probation_seed12345')
    args = driver._parse_args(['--stage', 'reduce', '--cut', '5.0', '--learned', 'blocks=/x/y'])
    assert driver._learned(args) == {'blocks': '/x/y'}
    with pytest.raises(SystemExit, match='NAME=DIR'):
        driver._learned(driver._parse_args(['--stage', 'reduce', '--learned', 'oops']))
    args = driver._parse_args(['--stage', 'analyse', '--existing-pool', 'general:1.5=/p'])
    assert driver._existing_pools(args) == {('general', 1.5): [__import__('pathlib').Path('/p')]}


def test_the_entry_list_is_found_under_this_machines_artifact_dir_not_the_recorded_path(tmp_path):
    """The design travels from the laptop to NERSC; the path it records does not. Every task of
    the first grid submission failed on /Users/... not existing on Perlmutter (2026-09-06)."""
    (tmp_path/'S15_entries_hard.csv').write_text('identifier\nA\n', encoding='utf-8')
    design = {'entry_files': {'hard': {'path': '/Users/nobody/MLI/docs/x/S15_entries_hard.csv',
                                       'name': 'S15_entries_hard.csv'}}}
    args = driver._parse_args(['--stage', 'generate', '--population', 'hard', '--cut', '5.0',
                               '--artifact-dir', str(tmp_path)])
    assert driver._entries_file(args, design) == tmp_path/'S15_entries_hard.csv'
    # A design written before `name` existed still resolves by basename.
    del design['entry_files']['hard']['name']
    assert driver._entries_file(args, design) == tmp_path/'S15_entries_hard.csv'
    # An explicit --entries-file wins, which is how the pilot ran.
    args = driver._parse_args(['--stage', 'generate', '--population', 'hard', '--cut', '5.0',
                               '--entries-file', '/elsewhere/list.csv'])
    assert str(driver._entries_file(args, design)) == '/elsewhere/list.csv'
