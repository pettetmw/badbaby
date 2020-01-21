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
import xarray as xr
from meeg_preprocessing import config
from meeg_preprocessing.utils import combine_events
from mne import read_epochs, EvokedArray
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore, LinearModel, get_coef
    )
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

import badbaby.return_dataframes as rd
from badbaby import defaults

workdir = defaults.datapath
analysese = ['Individual', 'Oddball']
plt.style.use('ggplot')
ages = [2, 6]
solver = 'liblinear'
lp = defaults.lowpass
win = defaults.peak_window
n_splits = 10  # how many folds to use for cross-validation
le = LabelEncoder()
seed = np.random.RandomState(42)
joint_kwargs = dict(ts_args=dict(gfp=True, time_unit='s'),
                    topomap_args=dict(sensors=False, time_unit='s'))
df = rd.return_dataframes('mmn')[0]
subjects = ['bad_%s' % ss for ss in df[df.complete == 1].index]
for iii, analysis in enumerate(analysese):
    if iii == 0:
        conditions = ['standard', 'ba', 'wa']
        combine = False
    else:
        conditions = ['standard', 'deviant']
        combine = True
    events = list(itertools.combinations(conditions, 2))
    contrasts = ['-'.join(cc) for cc in events]
    files = [op.join(defaults.datadir,
                     'AUC_%d_%s_%s.nc' % (lp, solver, analysis)),
             op.join(defaults.datadir, 'SCORES_%d_%s_%s.nc' % (lp, solver,
                                                               analysis))
             ]
    aucs = {kk: list() for kk in events}
    scores = {kk: list() for kk in events}
    for cii, cs in enumerate(events):
        contrast = '-'.join(cs)
        print('Fitting estimator for %s: ' % contrast)
        for si, subject in enumerate(subjects):
            ep_fname = op.join(workdir, subject, 'epochs',
                               'All_%d-sss_%s-epo.fif' % (lp, subject))
            eps = read_epochs(ep_fname)
            eps.apply_baseline()
            eps.drop_bad()
            if combine:
                # Combine across deviant trials
                eps = combine_events(eps, ['ba', 'wa'], {'deviant': 23})
            eps.equalize_event_counts(eps.event_id.keys())
            info = eps.info
            time = eps.times
            ix = eps.time_as_index(win[0])[0], eps.time_as_index(win[1])[0]
            c1, c2 = list(eps[cs].event_id.keys())
            clf = make_pipeline(RobustScaler(),
                                LogisticRegression(solver=solver,
                                                   penalty='l2',
                                                   max_iter=1000,
                                                   multi_class='auto',
                                                   random_state=seed))
            time_decode = SlidingEstimator(clf, n_jobs=config.N_JOBS,
                                           scoring='roc_auc',
                                           verbose=True)
            # K-fold cross-validation with ROC area under curve score
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=seed)
            # Get the data and label
            X = eps[cs].get_data()
            y = le.fit_transform(eps[cs].events[:, -1])
            # AUC b/c chance level same regardless of the class balance
            scores[cs].append(
                np.mean(cross_val_multiscore(time_decode, X=X, y=y,
                                             cv=cv, n_jobs=config.N_JOBS),
                        axis=0))
            # Average AUC score across MMN win samples
            aucs[cs].append(scores[cs][-1][ix[0]:ix[1]].mean())
            print("     Subject %s : %s vs. %s mean AUC: %.3f" % (
                subject, c1, c2, aucs[cs][-1].mean()))
            clf = make_pipeline(StandardScaler(),
                                LinearModel(
                                    LogisticRegression(solver=solver,
                                                       penalty='l2',
                                                       max_iter=1000,
                                                       multi_class='auto',
                                                       random_state=seed)))
            time_decode = SlidingEstimator(clf, n_jobs=config.N_JOBS,
                                           scoring=roc_auc_score,
                                           verbose=True)
            coef = get_coef(time_decode.fit(X, y), 'patterns_',
                            inverse_transform=True)
            # Get regression spatial pattern
            pattern = EvokedArray(coef, eps.info, tmin=eps.times[0])
            pattern.save(op.join(workdir, subject, 'epochs',
                                 '%s_%d-sss_%s-logit.fif' % (contrast, lp,
                                                             subject)))
    # Write data to disk
    for ii, (ds, fi) in enumerate(zip([aucs, scores], files)):
        print('     writing %s: ' % op.basename(fi))
        if ii == 0:
            assert len(list(aucs.values())) == len(events)
            foo = xr.DataArray(np.array((list(aucs.values()))),
                               coords=[contrasts, subjects],
                               dims=['contrasts', 'subject']).to_netcdf(fi)
        if ii == 1:
            foo = xr.DataArray(np.array((list(scores.values()))),
                               coords=[contrasts, subjects, time],
                               dims=['contrasts', 'subject',
                                     'time']).to_netcdf(fi)
