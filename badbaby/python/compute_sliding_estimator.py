#!/usr/bin/env python

"""compute_sliding_estimator.py: MVPA using samplewise logit to classify
oddball stimuli. Writes out xxx_cvscores.h5 files to disk"""
"""Notes:
    https://martinos.org/mne/stable/auto_tutorials/machine-learning
    /plot_sensors_decoding.html?highlight=mvp
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__credits__ = ["Goedel", "Escher", "Bach"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Production"

import os.path as op  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import pandas as pd
from meeg_preprocessing import config
from meeg_preprocessing.utils import combine_events
from mne import read_epochs, EvokedArray, grand_average
from mne.decoding import (
    SlidingEstimator, cross_val_multiscore, LinearModel, get_coef
    )
from mne.externals.h5io import write_hdf5
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import badbaby.python.return_dataframes as rd
from badbaby.python import defaults

workdir = defaults.datapath
plt.style.use('ggplot')
age = [2, 6]
lp = defaults.lowpass
condition1, condition2 = 'standard', 'deviant'
window = defaults.peak_window
n_splits = 5  # how many folds to use for cross-validation
for aix in age:
    rstate = np.random.RandomState(42)
    df = rd.return_dataframes('mmn', age=aix)[0]
    subjects = ['bad_%s' % ss for ss in df.index]
    scores = []
    auc = []
    evokeds = []
    hf_fname = op.join(defaults.datadir, '%smos_%d_cvscores.h5' % (aix, lp))
    for subject in subjects:
        ep_fname = op.join(workdir, subject, 'epochs',
                           'All_%d-sss_%s-epo.fif' % (lp, subject))
        epochs = read_epochs(ep_fname)
        epochs.apply_baseline()
        epochs = combine_events(epochs, ['ba', 'wa'], {'deviant': 23})
        epochs.equalize_event_counts(epochs.event_id.keys())
        epochs.drop_bad()
        # Get the data and labels
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
        evokeds.append(EvokedArray(coef, epochs.info, tmin=epochs.times[0]))
    
    # Plot
    # scores for individual subjects
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
                        'ind-auc_lp-%d_%d-mos.pdf' % (lp, aix)),
                bbox_to_inches='tight')
    # spatial patterns across subjects
    joint_kwargs = dict(ts_args=dict(gfp=True, time_unit='s'),
                        topomap_args=dict(sensors=False, time_unit='s'))
    hs = grand_average(evokeds).plot_joint(times=np.arange(window[0],
                                                           window[1], .05),
                                           title='patterns', **joint_kwargs)
    for hx, ch in zip(hs, ['mag', 'grad']):
        hx.savefig(op.join(defaults.figsdir,
                           'grp-estimator-%s-topo_%d-mos.pdf' %
                           (ch, aix)),
                   bbox_inches='tight')
    
    # cv scores across subjects
    score = np.asarray(scores).mean(axis=0)
    score_sem = sem(np.asarray(scores))
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
                        'grp-auc_%s-mos_lp-%d_cv-score.pdf' % (aix, lp)),
                bbox_to_inches='tight')
    plt.show()
    
    # plot & write out AUC measures
    auc = pd.DataFrame(data=auc, index=subjects,
                       columns=['auc'])
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    auc.plot(kind='bar', y='auc', ax=ax)
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    plt.xlabel('Subject')
    plt.ylabel('AUC')
    plt.title('%s Vs. %s' % (condition1, condition2))
    plt.legend()
    plt.savefig(op.join(defaults.figsdir, 'auc_lp-%d_%d-mos.pdf' % (aix, lp)),
                bbox_inches='tight')
    
    # Write logit data to disk
    write_hdf5(hf_fname,
               dict(subjects=subjects,
                    scores=scores,
                    auc=auc,
                    evokeds=evokeds),
               title='logit', overwrite=True)
