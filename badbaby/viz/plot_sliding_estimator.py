#!/usr/bin/env python

"""plot_sliding_estimator.py: visualize results of logit classification on
ERFs."""

import itertools
import os.path as op

import pandas as pd
import seaborn as sns
import xarray as xr
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
idx = pd.IndexSlice

# Wrangle df1 & df2 covariates
df1, df2 = rd.return_dataframes('mmn')
df1.reset_index(inplace=True)
ma_ = df1.age < 80
auc_ds_ids = df1[df1.simmInclude == 1].subjId
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
    aucs_covars = aucs.merge(covars, left_on='subject', right_on='subjId',
                             how='outer')
    aucs_covars.to_csv(op.join(defaults.datadir, 'AUC_%s_%d_%s.csv' %
                               (analysis, lp, solver)))
    g = sns.catplot(x='subjId', y='auc', hue='group', row='contrasts',
                    data=aucs_covars[aucs_covars.cdiAge == 18], kind='bar',
                    ci=None,
                    height=3,
                    aspect=6, dodge=False, legend=True)
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
                    data=aucs_covars[aucs_covars.cdiAge == 18], kind='boxen',
                    order=groups, row='contrasts', row_order=contrasts,
                    height=6, legend=False)
    ax = g.fig.gca()
    ax.axhline(.5, color='k', linestyle='--', label='chance',
               alpha=.25)
    ax.set_xlabel('Age')
    ax.legend()
    g.savefig(op.join(defaults.figsdir, 'GRP-SCORES_%d_%s.png' % (lp, solver)))
