#!/usr/bin/env python

"""compute_sliding_estimator.py: temporal logit classification of oddball ERFs.
    Does per age x subject:
        1. combine CV deviant trials into deviant condition
        2. Run logistic regression
        3. K-fold cross val using area under curve as score
        4. Write xxx_cv-scores.h5 files to disk
        5. Generate & write plots to disk
        6. Evaluate AUC difference between 2- and 6-months
    Notes:
        https://martinos.org/mne/stable/auto_tutorials/machine-learning/plot_sensors_decoding.html?highlight=mvp
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__credits__ = ["Goedel", "Escher", "Bach"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Production"

import os.path as op
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from meeg_preprocessing import config
from meeg_preprocessing.utils import combine_events
from mne import read_epochs, EvokedArray, grand_average
from mne.decoding import (
    SlidingEstimator, cross_val_multiscore, LinearModel, get_coef
    )
from mne.externals.h5io import write_hdf5, read_hdf5
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import badbaby.return_dataframes as rd
from badbaby import defaults

workdir = defaults.datapath
plt.style.use('ggplot')
ages = [2, 6]
lp = defaults.lowpass
condition1, condition2 = 'standard', 'deviant'
window = defaults.peak_window
n_splits = 7  # how many folds to use for cross-validation
for aix in ages:
    rstate = np.random.RandomState(42)
    df = rd.return_dataframes('mmn', age=aix)[0]
    subjects = ['bad_%s' % ss for ss in df.index]
    scores = []
    auc = []
    evokeds = dict()
    hf_fname = op.join(defaults.datadir, '%smos_%d_cv-scores.h5' % (aix, lp))
    for subject in subjects:
        ep_fname = op.join(workdir, subject, 'epochs',
                           'All_%d-sss_%s-epo.fif' % (lp, subject))
        epochs = read_epochs(ep_fname)
        epochs.apply_baseline()
        # Combine trials into deviant condition
        epochs = combine_events(epochs, ['ba', 'wa'], {'deviant': 23})
        epochs.equalize_event_counts(epochs.event_id.keys())
        epochs.drop_bad()
        # Get the data and label
        le = LabelEncoder()
        X = epochs.get_data()
        y = le.fit_transform(epochs.events[:, -1])
        info = epochs.info
        ix = epochs.time_as_index(window[0])[0], \
             epochs.time_as_index(window[1])[0]
        # run logit with AUC because chance level same regardless of the class
        # balance
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(solver='liblinear',
                                               penalty='l2',
                                               max_iter=4000,
                                               multi_class='auto',
                                               random_state=rstate))
        time_decode = SlidingEstimator(clf, n_jobs=config.N_JOBS,
                                       scoring='roc_auc',
                                       verbose=True)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=rstate)
        scores.append(np.mean(cross_val_multiscore(time_decode, X=X, y=y, cv=cv,
                                                   n_jobs=config.N_JOBS),
                              axis=0))
        auc.append(scores[-1][ix[0]:ix[1]].mean())  # windowed AUC
        print("Subject %s : AUC cross val score : %.3f" % (
                subject, auc[-1].mean()))
        clf = make_pipeline(StandardScaler(),
                            LinearModel(LogisticRegression(solver='liblinear',
                                                           penalty='l2',
                                                           max_iter=4000,
                                                           multi_class='auto',
                                                           random_state=rstate)))  # noqa
        time_decode = SlidingEstimator(clf, n_jobs=config.N_JOBS,
                                       scoring=roc_auc_score,
                                       verbose=True)
        coef = get_coef(time_decode.fit(X, y), 'patterns_',
                        inverse_transform=True)
        evokeds[subject] = EvokedArray(coef, epochs.info,
                                       tmin=epochs.times[0])
    
    # Plot
    # Individual cv-score timeseries
    tmin, tmax = defaults.epoching
    n_axs = len(subjects) + 1
    n_rows = int(np.ceil(n_axs / 4.))
    fig = plt.figure(figsize=(14, 14))
    fig.subplots_adjust(hspace=0.5, wspace=0.5,
                        bottom=0.1, left=0.1, right=0.98, top=0.95)
    n = 0
    for i in range(1, n_axs):
        ax = fig.add_subplot(n_rows, 4, i)
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.axvline(.0, color='k', linestyle='-')
        ax.plot(epochs.times, scores[n], label='score')
        ax.set(xlim=[tmin, tmax])
        ax.set_title('%s' % subjects[n], fontsize=8)
        n = n + 1
        if n == len(subjects):
            ax.legend(bbox_to_anchor=(2.2, 0.75), loc='best')
            break
    fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=16)
    fig.text(0.01, 0.5, 'Area under curve', va='center',
             rotation='vertical', fontsize=16)
    fig.text(0.5, 0.98, '%s v %s' % (condition1, condition2),
             ha='center', va='center', fontsize=16)
    fig.subplots_adjust()
    plt.show()
    fig.savefig(op.join(defaults.figsdir,
                        'ind-auc_lp-%d_%d-mos.png' % (lp, aix)),
                bbox_to_inches='tight')
    # Group averaged (ages) cv-score topomaps
    joint_kwargs = dict(ts_args=dict(gfp=True, time_unit='s'),
                        topomap_args=dict(sensors=False, time_unit='s'))
    hs = grand_average(list(evokeds.values())).plot_joint(
            times=np.arange(window[0],
                            window[1], .05),
            title='patterns', **joint_kwargs)
    for hx, ch in zip(hs, ['mag', 'grad']):
        hx.savefig(op.join(defaults.figsdir,
                           'grp-estimator-%s-topo_%d-mos.png' %
                           (ch, aix)),
                   bbox_inches='tight')
    
    # Group averaged (ages) cv-score timeseries
    score = np.asarray(scores).mean(axis=0)
    score_sem = stats.sem(np.asarray(scores))
    fig, ax = plt.subplots(1, figsize=(6.6, 5))
    ax.plot(epochs.times, score, label='score')
    ax.set(xlabel='Time (s)', ylabel='Area under curve (AUC)')
    ax.fill_between(epochs.times, score - score_sem,
                    score + score_sem,
                    color='c', alpha=.3, edgecolor='none')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.axvline(.0, color='k', linestyle='-')
    ax.legend()
    ax.set(xlim=[tmin, tmax])
    fig.tight_layout(pad=0.5)
    fig.savefig(op.join(defaults.figsdir,
                        'grp-auc_%smos_%d_cv-score.png' % (aix, lp)),
                bbox_to_inches='tight')
    plt.show()
    
    # Individual AUC values
    auc = pd.DataFrame(data=auc, index=subjects,
                       columns=['auc'])
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    auc.plot(kind='bar', y='auc', ax=ax)
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    plt.xlabel('Subject')
    plt.ylabel('AUC')
    plt.title('%s Vs. %s' % (condition1, condition2))
    plt.legend()
    plt.savefig(op.join(defaults.figsdir, 'auc_%d_%dmos.png' % (lp, aix)),
                bbox_inches='tight')
    # Write out
    write_hdf5(hf_fname,
               dict(subjects=subjects,
                    scores=scores,
                    auc=auc,
                    patterns=np.array([vv.data for vv in evokeds.values()])),
               title='logit', overwrite=True)

# combine 2- & 6-months RM into Pandas DF
sns.set_style('ticks')
sns.set_palette('colorblind')
dfs = list()
for aix, age in enumerate(ages):
    hf = read_hdf5(op.join(defaults.datadir,
                           '%smos_%d_cv-scores.h5' % (age, lp)),
                   title='logit')
    vars = {
            'ids': [re.findall(r'\d+', ll)[0] for ll in hf['subjects']],
            'age': np.ones((len(hf['subjects']))) * age
            }
    dfs.append(pd.concat([pd.DataFrame(data=vars), hf['auc'].reset_index()],
                         sort=False, axis=1))
df = pd.concat(dfs)
# Plot & evaluate difference between AUC at 2- vs. 6-months ages
ax = sns.barplot(x='age', y='auc', data=df)
df.pivot(index='ids', columns='age',
         values='auc').dropna().plot.bar(figsize=(10, 7))
# Wilcoxon signed-rank test Alt H0 2- < 6-months
compare = df[df.age == 2].merge(df[df.age == 6], on='ids')
stat, pval = stats.wilcoxon(compare.auc_x, compare.auc_y, alternative='less')
print(pval)


