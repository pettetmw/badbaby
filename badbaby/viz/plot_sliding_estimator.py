#!/usr/bin/env python

"""plot_sliding_estimator.py: visualize results of logit classification on
ERFs."""

import itertools
import os.path as op
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
from mne.externals.h5io import read_hdf5
from scipy import stats

from badbaby import defaults
import badbaby.return_dataframes as rd

sns.set_style('ticks')
sns.set_palette('deep')
# parameters
workdir = defaults.datapath
analysese = ['Individual', 'Oddball']
lp = defaults.lowpass
window = defaults.peak_window
ages = [2, 6]
solver = 'liblinear'
regex = r"[abc]$"

# Wrangle MEG & CDI covariates
MEG, CDI = rd.return_dataframes('mmn')
MEG.reset_index(inplace=True)
MEG['subject'] = ['bad_%s' % xx for xx in MEG.subjId]
MEG['subjId'] = [re.split(regex, ss)[0].upper() for ss in MEG.subjId]
MEG['subjId'] = ['BAD_%s' % xx for xx in MEG.subjId]
s_ = MEG.age < 80
MEG['group'] = s_.map({True: '2mos', False: '6mos'})
covars = pd.merge(MEG[['simmInclude', 'ses', 'age', 'gender', 'headSize',
                       'maternalEdu', 'maternalHscore',
                       'paternalEdu', 'paternalHscore',
                       'maternalEthno', 'paternalEthno', 'birthWeight',
                       'subjId', 'subject', 'group']], CDI, on='subjId',
                  validate='m:m')
rm_cohort = covars[covars.simmInclude == 1].subject.unique()
# pd.set_option('mode.chained_assignment', 'raise')

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
        ctrst = '_vs_'.join(cc)
        title = ctrst.replace('_vs_', ' vs. ')
        aucs = list()
        scrs = list()
        for aix in ages:
            auc_fi = op.join(defaults.datadir,
                             '%smos_%d_%s_%s_AUC.nc'
                             % (aix, lp, solver, analysis))
            auc_df = xr.open_dataarray(auc_fi).to_dataframe('auc')
            auc_df.reset_index(inplace=True, level=1)
            scores_fi = op.join(defaults.datadir,
                                '%smos_%d_%s_%s_SCORES.nc'
                                % (aix, lp, solver, analysis))
            scores_df = xr.open_dataarray(scores_fi).to_dataframe('scores')
            scores_df.reset_index(inplace=True, level=(1, 2))
            auc_ctrst = auc_df.loc[ctrst].merge(MEG['subject'], on='subject',
                                                right_index=True)
            scrs_ctrst = scores_df.loc[ctrst].merge(MEG['subject'],
                                                    on='subject',
                                                    right_index=True)
            # combine AUC & scores group datasets
            subjs = MEG[MEG.group == '%dmos' % aix]['subject'].values
            aa = auc_ctrst[auc_ctrst.subject.isin(subjs)].reset_index(
                drop=True)
            aa['group'] = pd.Series(
                np.repeat(np.array('%dmos' % aix), len(aa)))
            aucs.append(aa)
            ss = scrs_ctrst[scrs_ctrst.subject.isin(subjs)].reset_index(
                drop=True)
            ss['group'] = pd.Series(
                np.repeat(np.array('%dmos' % aix), len(ss)))
            scrs.append(ss)
        # plot timeseries scores over groups
        fig, ax = plt.subplots(1, figsize=(6.6, 5))
        sns.lineplot(x='time', y='scores', hue='group',
                     data=pd.concat(scrs, ignore_index=True),
                     ax=ax)
        ax.set(xlabel='Time (s)', ylabel='Score (AUC)')
        ax.set_title(title)
        ax.axhline(.5, color='k', linestyle='--', label='chance', alpha=.25)
        ax.axvline(.0, color='k', linestyle='-', alpha=.25)
        ax.legend()
        fig.savefig(op.join(defaults.figsdir,
                            'GRPAVR-SCORES_%s_%d_%s.png' % (
                                ctrst, lp, solver)))
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
        h.ax.set_title('%s' % ctrst)
        plt.legend()
        plt.savefig(op.join(defaults.figsdir,
                            'IND-SCORES_%s_%d_%s.png' % (
                                ctrst, lp, solver)))
        g = sns.catplot(x='group', y='auc',
                        data=ag, kind='box',
                        height=6, legend=False)
        g.ax.axhline(.5, color='k', linestyle='--', label='chance',
                     alpha=.25)
        g.ax.set_xlabel('Age')
        g.ax.set_ylabel('AUC')
        g.ax.set_title(title)
        plt.legend()
        plt.savefig(op.join(defaults.figsdir,
                            'GRP-SCORES_%s_%d_%s.png' % (
                                ctrst, lp, solver)))
        ma_ = MEG[MEG.simmInclude == 1][
            ['ses', 'gender', 'headSize', 'birthWeight', 'subject',
             'group']]
        rm = ag.merge(ma_, on='subject', sort=True,
                      validate='1:1').reset_index()
        # Wilcoxon signed-rank test Alt H0 2- < 6-months
        x = rm[rm.group_x == '2mos']['auc'].values
        y = rm[rm.group_x == '6mos']['auc'].values
        stat, pval = stats.wilcoxon(x, y,
                                    alternative=alt)
        print('%s (W, P-value): (%f, %f)' % (ctrst, stat, pval))
        rm = auc_df[auc_df.subject.isin(rm_cohort)].reset_index()
        rm = rm.merge(covars, on='subject', validate='1:m')
        rm.drop(['contrast', 'simmInclude'], axis=1, inplace=True)
        XX = rm[rm.group == '2mos']
        YY = rm[rm.group == '6mos'][['subject', 'auc']]
        for _df in [XX, YY]:
            _df['ids'] = [xx.strip('bad_') for xx in _df.subject]
        DF = XX.merge(YY, on='ids', suffixes=('_2mos', '_6mos'))
        DF.to_csv(op.join(defaults.datadir, 'RM-SCORES_%s_%d_%s.csv' % (
            ctrst, lp, solver)))

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
# fig.text(0.5, 0.98, '%s' % ctrst,
#          ha='center', va='center', fontsize=16)
# fig.subplots_adjust()
# fig.savefig(op.join(defaults.figsdir,
#                     '%smos_%d_%s_%s_%s-scores.png' % (
#                     aix, lp, solver,
#                     ctrst, tag)),
#             bbox_to_inches='tight')
