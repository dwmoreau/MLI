"""S10c's results document and figure, from the artefacts the runs produced."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlindex.model_training import FomHoldoutReport as HR

A = Path('docs/fom_campaign2/artifacts')
POST = ('ho_post', 'ho_post_logmean', 'ho_evidence')
COLOURS = {'ho_post': '#4f772d', 'ho_post_logmean': '#9d4edd', 'ho_evidence': '#e09f3e',
           'ho_M20': '#c1666b', 'ho_M_tilde': '#5fa8d3', 'ho_M_sym': '#1b4965',
           'ho_Minfo': '#8a817c', 'M20': '#111111'}

sweep = pd.read_csv(A/'S10c_holdout_sweep.csv')
control = HR.within_band_control(
    pd.read_parquet(A/'S10c_holdout_control.parquet'),
    [c for c in pd.read_parquet(A/'S10c_holdout_control.parquet').columns if c.endswith('__n5')],
    reference=('M20',), n_bands=400)
control.to_csv(A/'S10c_holdout_control.csv', index=False, encoding='utf-8')
band = control[control.m20_band == 'within-band (pair-weighted)'].copy()
band['merit'] = band['merit'].str.replace('__n5', '', regex=False)
sigma = pd.read_csv(A/'S10c_sigma_sensitivity.csv')
sigma['merit'] = sigma['merit'].str.replace('__n5', '', regex=False)
meta = json.loads((A/'S10c_holdout_reduced_meta.json').read_text())

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
plt.rcParams.update({'font.size': 9})

agg = sweep[sweep.scope == 'all']
for merit, blk in agg.groupby('merit'):
    blk = blk.sort_values('n_extra')
    ax[0].plot(blk.n_extra, 100*blk.top10, marker='o', ms=3.5, lw=1.5,
               label=merit.replace('ho_', ''), color=COLOURS.get(merit, '0.5'))
anc = agg.sort_values('n_extra').drop_duplicates('n_extra')
ax[0].plot(anc.n_extra, 100*anc.anchor_top10, '--', lw=2, color='0.15', label='M20, in sample')
ax[0].set(xlabel='surplus peaks scored', ylabel='top-10 (%)')
ax[0].set_title('a. as a ranker: still far behind M20', loc='left', fontsize=9.5)
ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.25, lw=.5)

b = band.sort_values('auc')
keep = b[b.merit.isin(list(POST) + ['ho_M20', 'ho_M_tilde', 'ho_Minfo', 'ho_M_sym', 'M20'])]
ax[1].barh(keep.merit, keep.auc, color=[COLOURS.get(m, '0.6') for m in keep.merit])
ax[1].axvline(0.5, color='0.2', lw=1.2)
ax[1].text(0.5, -0.75, ' chance', fontsize=7, color='0.3', va='top')
ax[1].set(xlabel='within-M20-band AUC', xlim=(0, 0.8))
ax[1].set_title('b. the control: information M20 does NOT have\n'
                '    retained pool, 400 bands', loc='left', fontsize=9.5)
ax[1].grid(alpha=.25, lw=.5, axis='x')

for merit, blk in sigma[sigma.merit.isin(POST + ('M20',))].groupby('merit'):
    blk = blk.sort_values('sigma_multiplier')
    ax[2].plot(blk.sigma_multiplier, blk.auc, marker='o', ms=4, lw=1.5,
               label=merit.replace('ho_', ''), color=COLOURS.get(merit, '0.5'))
ax[2].axhline(0.5, color='0.2', lw=1.0, ls=':')
ax[2].set(xscale='log', xlabel='sigma multiplier', ylabel='within-band AUC')
ax[2].set_xticks([0.25, 0.5, 1, 2, 4]); ax[2].set_xticklabels(['0.25', '0.5', '1', '2', '4'])
ax[2].set_title('c. and it does not hinge on sigma\n'
                '    SLICE -- read the shape, not the level', loc='left', fontsize=9.5)
ax[2].text(0.25, 0.503, ' levels sit ~0.13 below panel b: a subsampled pool keeps the\n'
           ' highest-scoring wrong candidates, so its negatives are harder (C2-F-111)',
           fontsize=6.2, color='0.35', va='bottom')
ax[2].legend(fontsize=7); ax[2].grid(alpha=.25, lw=.5)
fig.suptitle('S10c: a hold-out statistic built on the per-peak assignment posterior',
             fontsize=11.5)
fig.savefig(A/'S10c_holdout_posterior.png', dpi=200, bbox_inches='tight')

lead = pd.read_csv(A/'S10c_holdout.csv')
lead.to_csv(A/'S10c_holdout_posterior.csv', index=False, encoding='utf-8')
print('figure and csv written')
print(band.sort_values('auc', ascending=False)[['merit','auc']].round(4).to_string(index=False))
