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
from mne.externals.h5io import read_hdf5
from scipy import stats

from badbaby import defaults

workdir = defaults.datapath
plt.style.use('ggplot')
ages = [2, 6]
solver = 'liblinear'
tag = '2v'
if tag is '2v':
    combos = list(itertools.combinations(['standard', 'deviant'], 2))
else:
    combos = list(itertools.combinations(defaults.oddball_stimuli, 2))
lp = defaults.lowpass
window = defaults.peak_window

for aix in ages:
    hf_fname = op.join(defaults.datadir,
                       '%smos_%d_%s_%s-cv-scores.h5' % (aix, lp, solver, tag))
    hf = read_hdf5(hf_fname, title=solver)
    scores = hf['scores']
    auc = hf['auc']
    subjects = hf['subjects']
    time = hf['tvec']
    # Individual cv-score timeseries
    for cci, cc in enumerate(combos):
        this = '_vs_'.join(cc)
        tmin, tmax = defaults.epoching
        n_axs = len(scores[cci]) + 1
        n_rows = int(np.ceil(n_axs / 4.))
        fig = plt.figure(figsize=(14, 14))
        fig.subplots_adjust(hspace=0.5, wspace=0.5,
                            bottom=0.1, left=0.1, right=0.98, top=0.95)
        n = 0
        for i in range(1, n_axs):
            ax = fig.add_subplot(n_rows, 4, i)
            ax.axhline(.5, color='k', linestyle='--', label='chance')
            ax.axvline(.0, color='k', linestyle='-')
            ax.plot(time, scores[cci][n], label='score')
            ax.set(xlim=[tmin, tmax])
            ax.set_title('%s' % subjects[n], fontsize=8)
            n = n + 1
            if n == len(scores[cci]):
                ax.legend(bbox_to_anchor=(2.2, 0.75), loc='best')
                break
        fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=16)
        fig.text(0.01, 0.5, 'Area under curve', va='center',
                 rotation='vertical', fontsize=16)
        fig.text(0.5, 0.98, '%s' % this,
                 ha='center', va='center', fontsize=16)
        fig.subplots_adjust()
        fig.savefig(op.join(defaults.figsdir,
                            'ind-%s-%s-auc_lp-%d_%d-mos.png' % (tag, this, lp,
                                                                aix)),
                    bbox_to_inches='tight')

        # Group averaged (ages) cv-score timeseries
        score = np.asarray(scores[cci]).mean(axis=0)
        score_sem = stats.sem(np.asarray(scores[cci]))
        fig, ax = plt.subplots(1, figsize=(6.6, 5))
        ax.plot(time, score, label='score')
        ax.set(xlabel='Time (s)', ylabel='Area under curve (AUC)')
        ax.fill_between(time, score - score_sem, score + score_sem,
                        color='c', alpha=.3, edgecolor='none')
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.axvline(.0, color='k', linestyle='-')
        ax.legend()
        ax.set(xlim=[tmin, tmax])
        fig.tight_layout(pad=0.5)
        fig.savefig(op.join(defaults.figsdir,
                            '%dmos-avr_%s-%s-auc_%d.png' % (aix, tag, this,
                                                            lp)),
                    bbox_to_inches='tight')
        # Individual AUC values
        ds = pd.DataFrame(data=auc[cci], index=subjects, columns=['auc'])
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ds.plot(kind='bar', y='auc', ax=ax)
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        plt.xlabel('Subject')
        plt.ylabel('AUC')
        plt.title('%s' % this)
        plt.legend()
        plt.savefig(op.join(defaults.figsdir,
                            '%dmos_%s-%s-auc_%d.png' % (aix, tag, this, lp)),
                    bbox_inches='tight')

sns.set_style('ticks')
sns.set_palette('colorblind')
AA = list()
pvals = list()
# combine age cohort AUC responses
for aix, age in enumerate(ages):
    hf_fname = op.join(defaults.datadir,
                       '%smos_%d_%s_%s-cv-scores.h5' % (age, lp, solver, tag))
    hf = read_hdf5(hf_fname, title=solver)
    ids = [re.findall(r'\d+', ll)[0] for ll in hf['subjects']]
    BB = list()
    for cci, cc in enumerate(combos):
        # assert len(ids) == len(hf['auc'][cci])
        this = '_vs_'.join(cc)
        _df = pd.concat([pd.Series(hf['auc'][cci], name='auc'),
                         pd.Series([this for ii in hf['subjects']],
                                   name='contrast')], axis=1).reindex()
        BB.append(_df)
    df_ = pd.concat(BB, ignore_index=True)
    rep = len(df_) // len(ids)
    cols = pd.DataFrame(
        {
            'ids': np.reshape(np.repeat(np.array(ids)[:, np.newaxis],
                                        rep, 1), -1, order='F'),
            'age': np.reshape(
                np.repeat(np.array([age] * len(ids))[:, np.newaxis],
                          rep, 1), -1, order='F')
            })
    AA.append(cols.join(df_, how='inner'))
dff = pd.concat(AA, ignore_index=True)
# Plot & evaluate difference between AUC at 2- vs. 6-months ages
dff.age = dff.age.map({2: 'Two', 6: 'Six'})
for col in ['ids', 'age', 'contrast']:
    dff[col] = dff[col].astype('category')
sns.boxplot(x="contrast", y="auc", hue="age", notch=True, data=dff)
sns.despine(left=True)
for cci, cc in enumerate(combos):
    this = '_vs_'.join(cc)
     compare = dff[dff.contrast == this]
    # compare = dff[(dff.contrast == this) & (
    #            dff.ids != '309')]  # rm 1.5 IQR outlier  # noqa
    # Wilcoxon signed-rank test Alt H0 2- < 6-months
    compare = compare[compare.age == 'Two'].merge(compare[compare.age ==
                                                          'Six'],
                                                  on='ids')
    stat, pval = stats.wilcoxon(compare.auc_x, compare.auc_y,
                                alternative='less')
    print('%s (W, P-value): (%f, %f)' % (this, stat, pval))
