"""S18 item 3 -- a laptop proxy for the MITemplates refit that can only run on NERSC.

    python mlindex/scripts/run_fom_templates_proxy.py --population general
    python mlindex/scripts/run_fom_templates_proxy.py --population hard

THE QUESTION. The Miller-index template ranker (`MITemplates.hgbc_regressor`) orders candidate
template cells by a predicted success probability, and its input is `[probability(20), N_pred,
q2_calc_max]` where `probability` is the per-peak assignment statistic `rho` that S13 refuted.
The refit that swaps the statistic needs the ranker's training target -- the ROC file
`roc_file_name` on NERSC, no local copy (C2-F-068, C2-R-012) -- so it cannot run here. The
S18 handoff's recommendation is not "swap the vector" but "add sigma": the posterior is exactly
invariant to a uniform rescaling of the residuals (derivation section 4.3), so it carries no
absolute fit quality, and the ranker's input has no other misfit term (`_generate_xnn_common`
computes residuals for downsampling and never hands them on).

THE PROXY. Same template cells, same inputs, same model class, same selection rule, a stand-in
label. For every entry: the shipped template model for its true lattice generates its template
cells exactly as production does (`generate_xnn`); the four inputs are built from those cells --
`rho` (the incumbent), `rho + log sigma`, `posterior + log sigma`, `posterior` alone (campaign 1's
F-131 control, expected to lose) -- and the label is the distance of the template cell to the
true cell in xnn space, which is what the real target is a function of (`calibrate_templates`
maps `distance` through the ROC to a success rate). A `HistGradientBoostingRegressor` with the
shipped hyperparameters is fitted on `fom-train` crystals to predict -log10(distance) over the
rows production trains on (distance < `max_distance`), and read on `fom-dev`:

  * `spearman`      rank agreement between predicted and true distance over ALL template cells;
  * `auc_<r>`       how well the prediction separates cells within radius r of the truth;
  * `yield_<r>`     the share of entries whose top-`n_templates` selection (production's share
                    for the lattice system) holds at least one cell within r -- the number that
                    matters, because that is what the ranker exists to do;
  * `oracle_<r>`    the same with a perfect ranker (any cell within r exists at all).

The substitution, stated: the real target is a smooth ROC of distance measured on refinement
outcomes; this is a monotone function of the same distance. Ranking by either orders templates
the same way when the model is exact; they can differ where the model is not. Arms are paired on
the same entries and the same template cells; the fit is per lattice system on the general
population (15 crystals per Bravais lattice, C2-F-073) and per Bravais lattice on the hard one.
PROTOCOL section 8: fitted on `fom-train`, reported on `fom-dev`, never the other way.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.scripts.run_fom_assignment import entry_table  # noqa: E402

TAG = 'S18'
ARTIFACT_DIR = os.path.join(BASE, 'docs', 'fom_campaign2', 'artifacts')
MODELS_DIR = os.path.join(BASE, 'mlindex', 'models')
BROADENING_TAG = '1'
ARMS = ('rho', 'rho_sigma', 'posterior_sigma', 'posterior')
# Production's template share per lattice system: `UtilitiesOptimizer.get_*_optimizer`'s
# `{'generator': 'templates', 'n_unit_cells': int(share * n_candidates)}` with n_candidates_scale 1.
N_TEMPLATES = {'cubic': int(0.1*100), 'tetragonal': int(0.25*100), 'hexagonal': int(0.25*100),
               'rhombohedral': int(0.25*100), 'orthorhombic': int(0.25*100),
               'monoclinic': int(0.4*100), 'triclinic': int(0.55*100)}
RADII = (0.001, 0.003, 0.01, 0.03)
# The shipped regressor's hyperparameters (MITemplates.template_params_defaults / calibrate_templates).
REGRESSOR = dict(loss='squared_error', learning_rate=0.1, max_leaf_nodes=31, max_depth=4,
                 min_samples_leaf=100, l2_regularization=10, max_iter=25)
MAX_DISTANCE = 0.05


def derived_seed(entry_id, base_seed):
    digest = hashlib.sha256(f'{entry_id}|{base_seed}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'little')


def load_templator(bravais_lattice, lattice_system, seed):
    """The shipped template model for one lattice, as production loads it (Wrapper.setup_from_tag)."""
    from mlindex.model_training.MITemplates import MITemplates
    from mlindex.utilities.IOManagers import read_params
    tag = f'{lattice_system}_{BROADENING_TAG}'
    root = os.path.join(MODELS_DIR, tag)
    params = read_params(os.path.join(root, 'data', 'data_params.csv'))
    indices = np.array(params['unit_cell_indices'].split('[')[1].split(']')[0].split(','), dtype=int)
    data_params = {'lattice_system': lattice_system, 'unit_cell_indices': indices,
                   'unit_cell_length': int(params['unit_cell_length']),
                   'hkl_ref_length': int(params['hkl_ref_length']), 'n_peaks': int(params['n_peaks'])}
    hkl_ref = np.load(os.path.join(root, 'data', f'hkl_ref_{bravais_lattice}.npy'))
    templator = MITemplates(bravais_lattice, data_params, {'tag': tag}, hkl_ref,
                            os.path.join(root, 'template'), seed)
    templator.load_from_tag()
    return templator


def template_inputs(templator, q2_obs, xnn_true):
    """Every template cell for one pattern with the four arms' inputs and the distance label.

    `generate_xnn` is production's own path (the shipped statistic, `rho`); the posterior and
    sigma are computed on the same cells from one nearest-line scan, as the refit would.
    """
    from scipy.spatial.distance import cdist
    from mlindex.utilities.FigureOfMerits import get_assignment_posterior, get_assignment_sigma
    from mlindex.utilities.Q2Calculator import Q2Calculator
    xnn, rho, N_pred, q2_calc_max = templator.generate_xnn(q2_obs)
    n_cal = templator.template_params['n_peaks_calibration']
    q2_obs_cal = q2_obs[:n_cal]
    q2_calculator = Q2Calculator(lattice_system=templator.lattice_system,
                                 hkl=templator.hkl_ref[:, :n_cal], tensorflow=False,
                                 representation='xnn')
    q2_ref_calc = q2_calculator.get_q2(xnn)
    sigma, d1 = get_assignment_sigma(q2_obs_cal, q2_ref_calc, templator.lattice_system)
    posterior = get_assignment_posterior(q2_obs_cal, q2_ref_calc, templator.lattice_system,
                                         sigma=sigma, d1=d1)
    # The entry table stores the full six-component xnn; the template model works in the
    # lattice's own free components (`unit_cell_indices`), which is what its training compared.
    xnn_true = np.asarray(xnn_true, dtype=np.float64)[np.asarray(templator.unit_cell_indices)]
    distance = cdist(xnn, xnn_true[np.newaxis])[:, 0]
    scalars = np.stack((N_pred, q2_calc_max), axis=1).astype(np.float64)
    log_sigma = np.log(np.asarray(sigma, dtype=np.float64))[:, np.newaxis]
    inputs = {'rho': np.concatenate((rho, scalars), axis=1),
              'rho_sigma': np.concatenate((rho, scalars, log_sigma), axis=1),
              'posterior_sigma': np.concatenate((posterior, scalars, log_sigma), axis=1),
              'posterior': np.concatenate((posterior, scalars), axis=1)}
    return inputs, distance


class _Templators:
    """The shipped template model per lattice, loaded once, reseeded per entry."""

    def __init__(self, seed):
        self.seed = seed
        self.loaded = {}

    def for_entry(self, row):
        key = (row.bravais_lattice_true, row.lattice_system_true)
        if key not in self.loaded:
            self.loaded[key] = load_templator(*key, self.seed)
        templator = self.loaded[key]
        templator.rng = np.random.default_rng(derived_seed(row.entry_id, self.seed))
        return templator


def _group_of(row, groups):
    return row.lattice_system_true if groups[0] == 'lattice_system' else row.bravais_lattice_true


def collect_training(entries, templators, groups, rows_per_pattern=None, progress=True):
    """The rows production trains on: fom-train cells within MAX_DISTANCE of the truth.

    Kept per group and per arm as float32 matrices, never as a frame of lists: at ~20 000 cells
    a pattern the full set does not fit beside anything else on a 16 GB laptop, and the far cells
    are not training rows anyway (`_sample_training_val_data`).
    """
    inputs = {arm: {} for arm in ARMS}
    targets, n_cells, n_entries = {}, {}, {}
    started = time.perf_counter()
    for i, row in enumerate(entries.itertuples(index=False)):
        templator = templators.for_entry(row)
        arm_inputs, distance = template_inputs(templator, np.asarray(row.q2_obs, dtype=np.float64),
                                               row.xnn_true)
        group = _group_of(row, groups)
        keep = distance < MAX_DISTANCE
        if rows_per_pattern and keep.sum() > rows_per_pattern:
            # Production caps its training set at `n_instances_train` rows over ~1 000 entries,
            # about a thousand a pattern; this keeps a fixed-seed random subset per pattern so
            # the hard population's ~15 M rows fit in memory. Every arm sees the same rows.
            chosen = np.random.default_rng(derived_seed(row.entry_id, 7)).choice(
                np.flatnonzero(keep), size=rows_per_pattern, replace=False)
            keep = np.zeros_like(keep)
            keep[chosen] = True
        n_cells[group] = n_cells.get(group, 0) + int(distance.size)
        n_entries[group] = n_entries.get(group, 0) + 1
        targets.setdefault(group, []).append(-np.log10(np.maximum(distance[keep], 1e-12)))
        for arm in ARMS:
            inputs[arm].setdefault(group, []).append(arm_inputs[arm][keep].astype(np.float32))
        if progress and (i + 1) % 50 == 0:
            print(f'  train {i + 1}/{len(entries)}, {time.perf_counter() - started:.0f} s', flush=True)
    return ({arm: {g: np.concatenate(v) for g, v in per.items()} for arm, per in inputs.items()},
            {g: np.concatenate(v) for g, v in targets.items()}, n_cells, n_entries)


def fit_models(inputs, targets, seed):
    """One regressor per (arm, group), the shipped hyperparameters."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = {}
    for arm in ARMS:
        for group, X in inputs[arm].items():
            if X.shape[0] < REGRESSOR['min_samples_leaf']*2:
                continue
            models[(arm, group)] = HistGradientBoostingRegressor(random_state=seed, **REGRESSOR) \
                .fit(X, targets[group])
    return models


def score_dev(entries, templators, groups, models, radii=RADII, progress=True):
    """Every fom-dev pattern, generated once and scored under every arm, streamed.

    Returns the per-entry yield rows and, per cell, (arm prediction, distance) for the AUC and
    Spearman -- two floats a cell an arm, which is what fits.
    """
    rows, cell_blocks = [], []
    started = time.perf_counter()
    for i, row in enumerate(entries.itertuples(index=False)):
        group = _group_of(row, groups)
        if not all((arm, group) in models for arm in ARMS):
            continue
        templator = templators.for_entry(row)
        arm_inputs, distance = template_inputs(templator, np.asarray(row.q2_obs, dtype=np.float64),
                                               row.xnn_true)
        n_top = N_TEMPLATES[row.lattice_system_true]
        block = {'entry_id': row.entry_id, 'condition_bundle': row.condition_bundle, 'group': group,
                 'distance': distance.astype(np.float32)}
        for arm in ARMS:
            pred = models[(arm, group)].predict(arm_inputs[arm].astype(np.float32))
            block[f'pred_{arm}'] = pred.astype(np.float32)
            order = np.argsort(-pred, kind='stable')[:n_top]
            record = dict(entry_id=row.entry_id, condition_bundle=row.condition_bundle, group=group,
                          bravais_lattice=row.bravais_lattice_true, arm=arm, n_cells=int(distance.size),
                          n_top=n_top, min_distance=float(distance.min()),
                          min_distance_selected=float(distance[order].min()))
            for radius in radii:
                within = distance < radius
                record[f'hit_{radius:g}'] = bool(within[order].any())
                record[f'oracle_{radius:g}'] = bool(within.any())
            rows.append(record)
        cell_blocks.append(pd.DataFrame(block))
        if progress and (i + 1) % 50 == 0:
            print(f'  dev {i + 1}/{len(entries)}, {time.perf_counter() - started:.0f} s', flush=True)
    return pd.DataFrame(rows), pd.concat(cell_blocks, ignore_index=True)


def summarise(per_entry, cells, groups, radii=RADII):
    """Per (group, arm): Spearman and AUC over the dev cells, yield and oracle over entries."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rows = []
    for group, g_cells in cells.groupby('group', sort=False):
        target = -np.log10(np.maximum(g_cells['distance'].to_numpy(dtype=np.float64), 1e-12))
        dist = g_cells['distance'].to_numpy(dtype=np.float64)
        for arm in ARMS:
            pred = g_cells[f'pred_{arm}'].to_numpy(dtype=np.float64)
            sub = per_entry.loc[(per_entry['group'] == group) & (per_entry['arm'] == arm)]
            row = {groups[0]: group, 'arm': arm, 'n_dev_entries': int(sub.shape[0]),
                   'n_dev_cells': int(dist.size),
                   'spearman': float(spearmanr(pred, target).correlation)}
            for radius in radii:
                hit = dist < radius
                row[f'auc_{radius:g}'] = (float(roc_auc_score(hit, pred))
                                          if 0 < hit.sum() < hit.size else np.nan)
                row[f'yield_{radius:g}'] = float(sub[f'hit_{radius:g}'].mean())
                row[f'oracle_{radius:g}'] = float(sub[f'oracle_{radius:g}'].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def paired_yield(per_entry, radii=RADII, reference='rho'):
    """Each arm against the incumbent on the same entries: gained / lost and an exact McNemar."""
    from scipy.stats import binomtest
    rows = []
    key = ['entry_id', 'condition_bundle']
    base = per_entry.loc[per_entry['arm'] == reference].set_index(key)
    for arm in ARMS:
        if arm == reference:
            continue
        other = per_entry.loc[per_entry['arm'] == arm].set_index(key)
        shared = base.index.intersection(other.index)
        for radius in radii:
            a = base.loc[shared, f'hit_{radius:g}'].to_numpy(dtype=bool)
            b = other.loc[shared, f'hit_{radius:g}'].to_numpy(dtype=bool)
            gained, lost = int(np.sum(~a & b)), int(np.sum(a & ~b))
            p = float(binomtest(min(gained, lost), gained + lost, 0.5).pvalue) if gained + lost else 1.0
            rows.append(dict(arm=arm, reference=reference, radius=radius, n_entries=int(len(shared)),
                             reference_yield=float(a.mean()), arm_yield=float(b.mean()),
                             delta_pp=100*float(b.mean() - a.mean()), gained=gained, lost=lost,
                             p_value=p))
    return pd.DataFrame(rows)


def run_figure(artifact_dir, populations=('general', 'hard'), radii=RADII):
    """Yield against radius per arm, one panel per population, from the CSVs alone."""
    from mlindex.model_training.FomHoldoutReport import _style
    plt = _style()
    artifact_dir = Path(artifact_dir)
    present = [p for p in populations
               if (artifact_dir/f'{TAG}_templates_proxy_{p}.csv').exists()]
    fig, panels = plt.subplots(1, len(present), figsize=(5.2*len(present), 3.6), squeeze=False)
    colours = {'rho': '#8a817c', 'rho_sigma': '#1b4965', 'posterior_sigma': '#e09f3e',
               'posterior': '#b5651d'}
    labels = {'rho': 'rho (shipped)', 'rho_sigma': 'rho + log sigma',
              'posterior_sigma': 'posterior + log sigma', 'posterior': 'posterior alone'}
    for ax, population in zip(panels[0], present):
        table = pd.read_csv(artifact_dir/f'{TAG}_templates_proxy_{population}.csv')
        paired = pd.read_csv(artifact_dir/f'{TAG}_templates_proxy_paired_{population}.csv')
        n = int(paired['n_entries'].iloc[0]) if paired.shape[0] else 0
        for _, row in table.iterrows():
            arm = row['arm']
            ax.plot(list(radii), [100*row[f'yield_{r:g}'] for r in radii], marker='o',
                    color=colours.get(arm, '#444444'), label=labels.get(arm, arm),
                    linewidth=1.6 if arm == 'rho' else 1.1)
        ax.plot(list(radii), [100*table.iloc[0][f'oracle_{r:g}'] for r in radii], linestyle=':',
                color='black', label='any cell within r (oracle)')
        ax.set_xscale('log')
        ax.set_xlabel('radius r in xnn space (1/A^2)')
        ax.set_ylabel('patterns with a selected template within r (%)')
        ax.set_title(f'{population}: {n} fom-dev pattern-conditions')
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
    panels[0][0].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    path = artifact_dir/f'{TAG}_templates_proxy.png'
    fig.savefig(path, dpi=160)
    print(f'-> {path}')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='S18: laptop proxy for the MITemplates refit')
    parser.add_argument('--population', choices=('general', 'hard'), default='general')
    parser.add_argument('--stage', choices=('run', 'figure'), default='run')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--max-entries', type=int, default=None, help='cap for a smoke test')
    parser.add_argument('--train-rows-per-pattern', type=int, default=4000,
                        help='fixed-seed cap on training rows per fom-train pattern (0 = none)')
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    parser.add_argument('--cells-dir', default=os.path.join(BASE, 'mlindex', 'characterization',
                                                            'fom', 'templates_proxy'),
                        help='where the cell-level predictions go (run output, not record)')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    artifact_dir = Path(args.artifact_dir)
    if args.stage == 'figure':
        run_figure(artifact_dir)
        return
    cells_dir = Path(args.cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    entries = entry_table(args.population).reset_index(drop=True)
    if args.max_entries:
        entries = entries.groupby('bravais_lattice_true', group_keys=False).head(
            max(1, args.max_entries//entries['bravais_lattice_true'].nunique()))
    groups = ['lattice_system'] if args.population == 'general' else ['bravais_lattice']
    train = entries.loc[entries['split'] == 'fom-train']
    dev = entries.loc[entries['split'] == 'fom-dev']
    print(f'{args.population}: {len(train)} fom-train and {len(dev)} fom-dev pattern-conditions, '
          f'{entries["entry_id"].nunique()} crystals, fit per {groups[0]}', flush=True)
    templators = _Templators(args.seed)
    inputs, targets, n_cells, n_entries = collect_training(train, templators, groups,
                                                           args.train_rows_per_pattern)
    models = fit_models(inputs, targets, args.seed)
    print(f'fitted {len(models)} models on '
          + ', '.join(f'{g}: {targets[g].size:,} rows of {n_cells[g]:,} cells / {n_entries[g]} patterns'
                      for g in sorted(targets)), flush=True)
    per_entry, cells = score_dev(dev, templators, groups, models)
    table = summarise(per_entry, cells, groups)
    paired = paired_yield(per_entry)
    numeric = [c for c in table.columns if c not in groups + ['arm']]
    aggregate = table.groupby('arm', sort=False)[numeric].mean(numeric_only=True).reset_index()
    aggregate.insert(0, 'scope', 'unweighted over ' + groups[0])
    stem = f'{TAG}_templates_proxy'
    table.to_csv(artifact_dir/f'{stem}_per_group_{args.population}.csv', index=False)
    aggregate.to_csv(artifact_dir/f'{stem}_{args.population}.csv', index=False)
    paired.to_csv(artifact_dir/f'{stem}_paired_{args.population}.csv', index=False)
    per_entry.to_parquet(artifact_dir/f'{stem}_per_entry_{args.population}.parquet', index=False)
    # The cell-level predictions are run output, not record: regenerable in minutes, tens of MB.
    cells.to_parquet(cells_dir/f'{stem}_cells_{args.population}.parquet', index=False)
    (artifact_dir/f'{stem}_provenance_{args.population}.json').write_text(
        json.dumps(dict(seed=args.seed, regressor=REGRESSOR, max_distance=MAX_DISTANCE,
                        radii=RADII, n_templates=N_TEMPLATES, arms=ARMS, groups=groups,
                        n_train_rows={g: int(v.size) for g, v in targets.items()},
                        n_train_cells=n_cells, n_train_patterns=n_entries,
                        n_dev_patterns=int(dev.shape[0]), n_dev_cells=int(cells.shape[0]),
                        max_entries=args.max_entries,
                        train_rows_per_pattern=args.train_rows_per_pattern), indent=2), encoding='utf-8')
    show = ['arm', 'spearman'] + [c for c in aggregate.columns if c.startswith(('yield_', 'oracle_', 'auc_'))]
    print(aggregate[show].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == '__main__':
    main()
