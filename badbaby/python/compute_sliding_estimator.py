#!/usr/bin/env python

"""Plot subject head positions."""

"""
=================
Sliding estimator
=================
Adapted from mne-python tutorial sliding_estimator.py
A sliding estimator fits a logistic legression model for every time point.
In this example, we contrast the condition 'famous' against 'scrambled'
using this approach. The end result is an averaging effect across sensors.
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

import numpy as np
from scipy.io import savemat
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


l_freq = '30'
n_jobs = 18
random_state = np.random.RandomState(42)


def run_time_decoding(subject_id, condition1, condition2):
    print("processing subject: %s (%s vs %s)"
          % (subject_id, condition1, condition2))

    data_path = os.path.join(meg_dir, subject_id, 'epochs')
    epochs = mne.read_epochs(os.path.join(data_path,
                             'All_%s-sss_%s-epo.fif' % (l_freq, subject_id)))

    # We define the epochs and the labels
    epochs = mne.concatenate_epochs([epochs[condition1],
                                    epochs[condition2]])
    epochs.apply_baseline()

    # Let us restrict ourselves to the MEG channels, and also decimate to
    # make it faster (although we might miss some detail / alias)
    epochs.pick_types(meg=True).decimate(4, verbose='error')

    # Get the data and labels
    X = epochs.get_data()
    n_cond1 = len(epochs[condition1])
    n_cond2 = len(epochs[condition2])
    y = np.r_[np.ones(n_cond1), np.zeros(n_cond2)]

    # Use AUC because chance level is same regardless of the class balance
    se = SlidingEstimator(
        make_pipeline(StandardScaler(),
                      LogisticRegression(random_state=random_state)),
        scoring='roc_auc', n_jobs=n_jobs)
    cv = StratifiedKFold(random_state=random_state)
    scores = cross_val_multiscore(se, X=X, y=y, cv=cv)

    # let's save the scores now
    a_vs_b = '%s_vs_%s' % (os.path.basename(condition1),
                           os.path.basename(condition2))
    fname_td = os.path.join(data_path, '%s_lowpass-%sHz-td-auc-%s.mat'
                            % (subject_id, l_freq, a_vs_b))
    savemat(fname_td, {'scores': scores, 'times': epochs.times})


# parallelize inside the mne.decoding.SlidingEstimator class
# so we don't dispatch manually to multiple jobs.

for subject in subjects:
    run_time_decoding(subject, 'fam', 'unfam')
