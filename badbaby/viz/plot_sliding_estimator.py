#!/usr/bin/env python

"""plot_sliding_estimator.py: visualize results of logit classification on
ERFs."""

import itertools
import os.path as op
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy import stats
from mne import grand_average
import badbaby.return_dataframes as rd
from badbaby import defaults

# parameters
sns.set_style('ticks')
sns.set_palette('muted')
workdir = defaults.datapath
analysese = ['Individual', 'Oddball']
lp = defaults.lowpass
window = defaults.peak_window
ages = [2, 6]
solver = 'liblinear'
regex = r"[abc]$"

# Wrangle MEG & CDI covariates
MEG, CDI = rd.return_dataframes('mmn')
ma_ = MEG.age < 80
MEG['group'] = ma_.map({True: '2mos', False: '6mos'})
CDI['dummy'] = [xx.lstrip('bad_') for xx in CDI.subjId]
MEG['dummy'] = [xx.lstrip('bad_') for xx in MEG.index]
COVAR = MEG.merge(CDI, on='dummy', left_index=True,
                  validate='m:m')
COVAR = COVAR[['simmInclude', 'ses', 'age', 'gender',
               'headSize',
               'maternalEdu', 'maternalHscore',
               'paternalEdu', 'paternalHscore',
               'maternalEthno', 'paternalEthno', 'birthWeight',
               'group', 'dummy', 'cdiAge', 'm3l', 'vocab']]
COVAR.insert(0, 'ids', COVAR.dummy.values)

for iii, analysis in enumerate(analysese):
    if iii == 0:
        conditions = ['standard', 'ba', 'wa']
        combine = False
        # alt = 'greater'
    else:
        conditions = ['standard', 'deviant']
        combine = True
        # alt = 'less'
    combos = list(itertools.combinations(conditions, 2))
    for cci, cc in enumerate(combos):
        contrast = '-'.join(cc)
        fi = op.join(defaults.datadir,
                         'AUC_%d_%s_%s.nc'
                         % (lp, solver, analysis))
        dfa = xr.open_dataarray(fi).to_dataframe('auc')
        dfa.reset_index(inplace=True, level=0, drop=True)
        fi = op.join(defaults.datadir,
                            'SCORES_%d_%s_%s.nc'
                            % (lp, solver, analysis))
        dfs = xr.open_dataarray(fi).to_dataframe('scores')
        dfs.reset_index(inplace=True, level=[0,1], drop=True)
        # combine frames
        
        # plot timeseries scores over groups
        fig, ax = plt.subplots(1, figsize=(6.6, 5))
        sns.lineplot(x='time', y='scores', hue='group',
                     data=pd.concat(scrs, ignore_index=True),
                     ax=ax)
        ax.set(xlabel='Time (s)', ylabel='Score (AUC)')
        ax.set_title(contrast)
        ax.axhline(.5, color='k', linestyle='--', label='chance', alpha=.25)
        ax.axvline(.0, color='k', linestyle='-', alpha=.25)
        ax.legend()
        fig.savefig(op.join(defaults.figsdir,
                            'GRPAVR-SCORES_%s_%d_%s.png' % (
                                contrast, lp, solver)))
        # plot mean-window AUC
        ag = pd.concat(aucs, ignore_index=True)
        ag['ids'] = [xx.lstrip('bad_') for xx in ag.subject]
        h = sns.catplot(x='ids', y='auc', hue='group',
                        data=ag, kind='bar',
                        order=sorted(ag.ids.values), height=8,
                        aspect=2, dodge=False, legend=False)
        h.set_xticklabels(h.ax.xaxis.get_majorticklabels(),
                          rotation=90, fontweight='light', fontsize='small')
        h.ax.axhline(.5, color='k', linestyle='--', label='chance',
                     alpha=.25)
        h.ax.set_xlabel('Subject')
        h.ax.set_ylabel('AUC')
        h.ax.set_title('%s' % contrast)
        plt.legend()
        plt.savefig(op.join(defaults.figsdir,
                            'IND-SCORES_%s_%d_%s.png' % (
                                contrast, lp, solver)))
        g = sns.catplot(x='group', y='auc',
                        data=ag, kind='box',
                        height=6, legend=False)
        g.ax.axhline(.5, color='k', linestyle='--', label='chance',
                     alpha=.25)
        g.ax.set_xlabel('Age')
        g.ax.set_ylabel('AUC')
        g.ax.set_title(contrast)
        plt.legend()
        plt.savefig(op.join(defaults.figsdir,
                            'GRP-SCORES_%s_%d_%s.png' % (
                                contrast, lp, solver)))
        
        ag = ag.pivot(index='subject', columns='group', values=['ids', 'auc'])
        ag.reset_index(inplace=True)
        rm = ag.join(ma_)
        x = rm[rm[('auc', '2mos')].notna()][('auc', '2mos')].values
        y = rm[rm[('auc', '6mos')].notna()][('auc', '6mos')].values
        rm['ids'] = [xx.lstrip('bad_') for xx in rm.subject]
        rm.to_csv(op.join(defaults.datadir, 'RM-SCORES_%s_%d_%s.csv' % (
            contrast, lp, solver)))
        # Wilcoxon signed-rank test Alt H0 2- < 6-months
        stat, pval = stats.wilcoxon(x, y, alternative="less")
        print('%s (W, P-value): (%f, %f)' % (contrast, stat, pval))


MEG.reset_index(inplace=True)
MEG['subject'] = ['bad_%s' % xx for xx in MEG.subjId]
MEG['ids'] =
MEG['subjId'] =
MEG['subjId'] = ['BAD_%s' % xx for xx in MEG.subjId]


rm_cohort = covars[covars.simmInclude == 1].subject.unique()
ma_ = MEG[MEG.simmInclude == 1][
            ['ses', 'gender', 'headSize', 'birthWeight', 'subject',
             'group']]
        ma_.columns = pd.MultiIndex.from_product([['ids'], ma_.columns])

# n_axs = len(scores[cci]) + 1
# n_rows = int(np.ceil(n_axs / 4.))
# fig = plt.figure(figsize=(14, 14))
# fig.subplots_adjust(hspace=0.5, wspace=0.5,
#                     bottom=0.1, left=0.1, right=0.98, top=0.95)
# n = 0
# for i in range(1, n_axs):
#     ax = fig.add_subplot(n_rows, 4, i)
#     ax.axhline(.5, color='k', linestyle='--', label='chance')
#     ax.axvline(.0, color='k', linestyle='-')
#     ax.plot(time, scores[cci][n], label='score')
#     ax.set(xlim=[tmin, tmax])
#     ax.set_title('%s' % subjects[n], fontsize=8)
#     n = n + 1
#     if n == len(scores[cci]):
#         ax.legend(bbox_to_anchor=(2.2, 0.75), loc='best')
#         break
# fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=16)
# fig.text(0.01, 0.5, 'Area under curve', va='center',
#          rotation='vertical', fontsize=16)
# fig.text(0.5, 0.98, '%s' % contrast,
#          ha='center', va='center', fontsize=16)
# fig.subplots_adjust()
# fig.savefig(op.join(defaults.figsdir,
#                     '%smos_%d_%s_%s_%s-scores.png' % (
#                     aix, lp, solver,
#                     contrast, tag)),
#             bbox_to_inches='tight')


for aix in ages:
    
    # Group averaged (ages) cv-score topomaps
    hs = grand_average(list(patterns[cs])).plot_joint(
        times=np.arange(win[0],
                        win[1], .05),
        title='patterns', **joint_kwargs)
    for hx, ch in zip(hs, ['mag', 'grad']):
        hx.savefig(op.join(defaults.figsdir,
                           '%dmos-avr-%s-%s-%s-topo_.png' %
                           (aix, solver, contrast, ch)),
                   bbox_inches='tight')