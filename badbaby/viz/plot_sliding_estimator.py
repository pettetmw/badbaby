#!/usr/bin/env python

"""plot_sliding_estimator.py: visualize results of logit classification on
ERFs."""

import itertools
import os.path as op

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy import stats
import badbaby.return_dataframes as rd
from badbaby import defaults

# parameters
sns.set_style('ticks')
sns.set_palette('muted')
workdir = defaults.datapath
analysese = ['Individual', 'Oddball']
lp = defaults.lowpass
window = defaults.peak_window
groups = ['2mos', '6mos']
solver = 'lbfgs'
regex = r"[abcd_]$"

# Wrangle df1 & df2 covariates
df1, df2 = rd.return_dataframes('mmn')
df1.reset_index(inplace=True)
ma_ = df1.age < 80
rm_ids = df1[df1.simmInclude == 1].subjId
df1['group'] = ma_.map({True: '2mos', False: '6mos'})
df2['subjId'] = [xx.lstrip('BAD_') for xx in df2.subjId]
df1['dummy'] = [xx.rstrip(regex) for xx in df1.subjId]
covars = df1.merge(df2, left_on='dummy', right_on='subjId', validate='m:m')
covars['subjId'] = ['bad_%s' % xx for xx in covars.subjId_x]
covars = covars[['simmInclude', 'ses', 'age', 'gender',
                 'headSize', 'subjId',
                 'maternalEdu', 'maternalHscore',
                 'paternalEdu', 'paternalHscore',
                 'maternalEthno', 'paternalEthno', 'birthWeight',
                 'group', 'cdiAge', 'm3l', 'vocab']]

for iii, analysis in enumerate(analysese):
    if iii == 0:
        conditions = ['standard', 'ba', 'wa']
    else:
        conditions = ['standard', 'deviant']
    combos = list(itertools.combinations(conditions, 2))
    contrasts = ['-'.join(cc) for cc in combos]
    fi = op.join(defaults.datadir,
                 'SCORES_%d_%s_%s.nc'
                 % (lp, solver, analysis))
    scores = xr.open_dataarray(fi).to_dataframe('scores')
    scores.reset_index(inplace=True, level=[0, 1, 2])
    scores['subjId'] = [xx.lstrip('bad_') for xx in scores.subject]
    scores = scores.merge(df1, on='subjId')
    # plot timeseries scores over groups
    g = sns.relplot(x='time', y='scores', data=scores, style='group',
                    style_order=groups, kind='line',
                    row='contrasts', row_order=contrasts, legend='brief',
                    ci=99,
                    height=4, aspect=2, lw=1)
    g.fig.gca().set_xlim(scores.time.min(), scores.time.max())
    g.savefig(op.join(defaults.figsdir,
                      'GRPAVR-SCORES_%d_%s.png' % (lp, solver)))
    # plot mean-window AUC
    fi = op.join(defaults.datadir,
                 'AUC_%d_%s_%s.nc'
                 % (lp, solver, analysis))
    aucs = xr.open_dataarray(fi).to_dataframe('auc')
    aucs.reset_index(inplace=True, level=[0, 1])
    aucs = aucs.merge(covars, left_on='subject', right_on='subjId', how='outer')
    g = sns.catplot(x='subjId', y='auc', hue='group', row='contrasts',
                    data=aucs[aucs.cdiAge == 18], kind='bar', ci=None,
                    height=3,
                    aspect=6, dodge=False, legend=False)
    ax = g.fig.gca()
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(),
                       rotation=90, fontweight='light',
                       fontsize='small')
    ax.axhline(.5, color='k', linestyle='--', label='chance',
               alpha=.25)
    ax.set_xlabel('Subject')
    ax.legend()
    g.savefig(op.join(defaults.figsdir, 'IND-SCORES_%d_%s.png' % (lp, solver)))
    g = sns.catplot(x='group', y='auc',
                    data=aucs[aucs.cdiAge == 18], kind='boxen',
                    order=groups, row='contrasts', row_order=contrasts,
                    height=6, legend=False)
    ax = g.fig.gca()
    ax.axhline(.5, color='k', linestyle='--', label='chance',
               alpha=.25)
    ax.set_xlabel('Age')
    ax.legend()
    g.savefig(op.join(defaults.figsdir, 'GRP-SCORES_%d_%s.png' % (lp, solver)))
    rm = aucs[aucs.simmInclude == 1]
    rm = rm.pivot_table(index=['group', 'subject'])
    pieces = dict(zip(groups, [rm.loc[ag, :] for ag in groups]))
    for kk, vv in pieces.items():
        vv.insert(0, 'ids', np.unique([xx.strip(regex) for xx in vv.index]))
        pieces[kk].update((kk, vv.reset_index(drop=True)))
    result = pd.merge(pieces['2mos'][['ids', 'age', 'auc']], pieces['6mos'],
                      on='ids',
                      suffixes=groups)
    result.to_csv(op.join(defaults.datadir, 'RM-SCORES_%d_%s.csv' % (lp,
                                                                     solver)))
    for cc in contrasts:
        x = result.auc2mos.values
        y = result.auc6mos.values
        # Wilcoxon signed-rank test Alt H0 2- < 6-months
        stat, pval = stats.wilcoxon(x, y, alternative="less")
        print('%s (W, P-value): (%f, %f)' % (cc, stat, pval))
