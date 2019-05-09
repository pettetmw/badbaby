#!/usr/bin/env python

"""Compute temporal sliding estimator for oddball conditions."""

"""
===============================
Decoding (ML) across time (MEG)
===============================
Adapted from mne-python tutorial sliding_estimator.py
A sliding estimator fits a logistic regression model for every time point.
This script contrasts the oddball condition "standard" against "standard", 
resulting in an averaging effect across sensors.
The contrast across different sensors are combined into a single plot.
"""

__author__ = "Kambiz Tavabi"
__credits__ = ["Eric Larson"]
__copyright__ = "Copyright 2018, Seattle, Washington"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Development"

import os

import mne
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.epochs import combine_event_ids
from scipy.io import savemat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import badbaby.python.return_dataframes as rd
from badbaby.python import defaults

df = rd.return_dataframes("mmn")[0]
study_dir = defaults.paradigms["mmn"]
subjects = ["bad_%s" % ss for ss in df.subjId.values]
l_freq = "30"
n_jobs = 18
random_state = np.random.RandomState(42)


def run_time_decoding(subject_id, epochs, condition1, condition2):
    print("processing subject: %s (%s vs %s)"
          % (subject_id, condition1, condition2))

    datapath = os.path.join(study_dir, subject_id, "epochs")
    
    # We define the epochs and the labels
    epochs = mne.concatenate_epochs([epochs[condition1],
                                    epochs[condition2]])
    epochs.apply_baseline()

    # Let us restrict ourselves to the MEG channels, and also decimate to
    # make it faster (although we might miss some detail / alias)
    epochs.pick_types(meg=True).decimate(4, verbose="error")

    # Get the data and labels
    X = epochs.get_data()
    n_cond1 = len(epochs[condition1])
    n_cond2 = len(epochs[condition2])
    y = np.r_[np.ones(n_cond1), np.zeros(n_cond2)]

    # Use AUC because chance level is same regardless of the class balance
    se = SlidingEstimator(
        make_pipeline(StandardScaler(),
                      LogisticRegression(random_state=random_state)),
        scoring="roc_auc", n_jobs=n_jobs)
    cv = StratifiedKFold(random_state=random_state)
    scores = cross_val_multiscore(se, X=X, y=y, cv=cv)

    # let"s save the scores now
    a_vs_b = "%s_vs_%s" % (os.path.basename(condition1),
                           os.path.basename(condition2))
    fname_td = os.path.join(datapath, "%s_lowpass-%sHz-td-auc-%s.mat"
                            % (subject_id, l_freq, a_vs_b))
    savemat(fname_td, {"scores": scores, "times": epochs.times})


def combine_events(epochs, old_event_ids, new_event_id):
    """
    Parameters:
    epochs : instance of Epochs
        The epochs to operate on.
    old_event_ids : str, or list
        Conditions to collapse together.
    new_event_id : dict, or int
        A one-element dict (or a single integer) for the new condition.
    """
    # first, equalize trial counts (this will make a copy)
    assert len(old_event_ids) == 2
    e = epochs[old_event_ids]
    e.equalize_event_counts(old_event_ids)
    # second, collapse relevant types
    event_ids = epochs.event_id.keys()
    keeper = list(set(event_ids) - set(old_event_ids))
    combine_event_ids(e, old_event_ids, new_event_id, copy=False)
    return mne.concatenate_epochs([e, epochs[keeper]])


# parallelize inside the mne.decoding.SlidingEstimator class
# so we don"t dispatch manually to multiple jobs.
for subject in subjects:
    data_path = os.path.join(study_dir, subject, "epochs")
    eps = mne.read_epochs(os.path.join(data_path,
                                       "All_%s-sss_%s-epo.fif" % (
                                           l_freq, subject)))
    coll_eps = combine_events(eps, ["ba", "wa"],  {"deviant": 69})
    run_time_decoding(subject, coll_eps, "standard", "deviant")
