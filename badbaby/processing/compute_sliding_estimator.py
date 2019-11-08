#!/usr/bin/env python

"""compute_sliding_estimator.py: temporal logit classification of oddball ERFs.
    Notes:
        https://martinos.org/mne/stable/auto_tutorials/machine-learning
        /plot_sensors_decoding.html?highlight=mvp
"""

import itertools
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from meeg_preprocessing import config
from meeg_preprocessing.utils import combine_events
from mne import read_epochs, EvokedArray, grand_average
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore, LinearModel, get_coef
    )
from mne.externals.h5io import write_hdf5
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

import badbaby.return_dataframes as rd
from badbaby import defaults

workdir = defaults.datapath
plt.style.use('ggplot')
ages = [2, 6]
solver = 'liblinear'
cl = 'auto'
lp = defaults.lowpass
combine = True
if combine:
    tag = '2v'
    combos = list(itertools.combinations(['standard', 'deviant'], 2))
else:
    tag = '3v'
    combos = list(itertools.combinations(defaults.oddball_stimuli, 2))
window = defaults.peak_window
n_splits = 7  # how many folds to use for cross-validation
le = LabelEncoder()
for aix in ages:
    scores = {kk: list() for kk in combos}
    auc = {kk: list() for kk in combos}
    evokeds = {kk: list() for kk in combos}
    patterns = {kk: list() for kk in combos}
    rstate = np.random.RandomState(42)
    df = rd.return_dataframes('mmn', age=aix)[0]
    subjects = ['bad_%s' % ss for ss in df.index]

    hf_fname = op.join(defaults.datadir,
                       '%smos_%d_%s_%s-cv-scores.h5' % (aix, lp, solver, tag))
    for cii, cs in enumerate(combos):
        for si, subject in enumerate(subjects):
            ep_fname = op.join(workdir, subject, 'epochs',
                               'All_%d-sss_%s-epo.fif' % (lp, subject))
            epochs = read_epochs(ep_fname)
            epochs.apply_baseline()
            epochs.drop_bad()
            if combine:
                # Combine across deviant trials
                epochs = combine_events(epochs, ['ba', 'wa'], {'deviant': 23})
            epochs.equalize_event_counts(epochs.event_id.keys())
            info = epochs.info
            time = epochs.times
            ix = epochs.time_as_index(window[0])[0], \
                 epochs.time_as_index(window[1])[0]
            condition1, condition2 = list(epochs[cs].event_id.keys())
            clf = make_pipeline(RobustScaler(),
                                LogisticRegression(solver='liblinear',
                                                   penalty='l2',
                                                   max_iter=1000,
                                                   multi_class='auto',
                                                   random_state=rstate))
            time_decode = SlidingEstimator(clf, n_jobs=config.N_JOBS,
                                           scoring='roc_auc',
                                           verbose=True)
            # K-fold cross-validation with ROC area under curve score
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=rstate)
            # Get the data and label
            X = epochs[cs].get_data()
            y = le.fit_transform(epochs[cs].events[:, -1])
            # logit AUC b/c chance level same regardless of the class balance
            scores[cs].append(
                np.mean(cross_val_multiscore(time_decode, X=X, y=y,
                                             cv=cv,
                                             n_jobs=config.N_JOBS),
                        axis=0))
            # Average AUC score across MMN window samples
            auc[cs].append(scores[cs][-1][ix[0]:ix[1]].mean())
            print("Subject %s : %s vs. %s mean AUC: %.3f" % (
                subject, condition1, condition2, auc[cs][-1].mean()))
            clf = make_pipeline(StandardScaler(),
                                LinearModel(
                                    LogisticRegression(solver='liblinear',
                                                       penalty='l2',
                                                       max_iter=1000,
                                                       multi_class='auto',
                                                       random_state=rstate)))
            time_decode = SlidingEstimator(clf, n_jobs=config.N_JOBS,
                                           scoring=roc_auc_score,
                                           verbose=True)
            coef = get_coef(time_decode.fit(X, y), 'patterns_',
                            inverse_transform=True)
            # Get regression spatial patterns
            evokeds[cs].append(EvokedArray(coef, epochs.info,
                                           tmin=epochs.times[0]))
        # Group averaged (ages) cv-score topomaps
        joint_kwargs = dict(ts_args=dict(gfp=True, time_unit='s'),
                            topomap_args=dict(sensors=False, time_unit='s'))
        hs = grand_average(list(evokeds[cs])).plot_joint(
            times=np.arange(window[0],
                            window[1], .05),
            title='patterns', **joint_kwargs)
        this = '_vs_'.join(cs)
        for hx, ch in zip(hs, ['mag', 'grad']):
            hx.savefig(op.join(defaults.figsdir,
                               '%dmos-avr-%s-%s-%s-topo_.png' %
                               (aix, solver, this, ch)),
                       bbox_inches='tight')
    # Write age-cohort logit results to disk
    write_hdf5(hf_fname,
               dict(subjects=subjects,
                    scores=list(scores.values()),
                    auc=list(auc.values()),
                    tvec=time),
               title=solver, overwrite=True)
