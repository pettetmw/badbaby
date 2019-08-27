#!/usr/bin/env python

"""Compute XDAWN components for oddball stimuli ERF sensor data.
    Per age x condition x subject:
        1. Compute XDAWN filter for auditory ERF
        2. Apply XDAWN filter to oddball ERFs
        3. Write out XDAWN TS and topographies to xxx_xdawn.h5 files
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"

from os import path as op

import matplotlib.pyplot as plt
import numpy as np
from meeg_preprocessing import config
from meeg_preprocessing.utils import combine_events
from mne import (
    read_epochs, compute_covariance,
    compute_rank
    )
from mne.cov import regularize
from mne.epochs import combine_event_ids
from mne.externals.h5io import write_hdf5
from mne.preprocessing import Xdawn
from pandas.plotting import scatter_matrix

from badbaby import return_dataframes as rd, defaults

# parameters
workdir = defaults.datapath
analysis = 'oddball'
conditions = ['standard', 'deviant']
plt.style.use('ggplot')
tmin, tmax = defaults.epoching
lp = defaults.lowpass
ages = [2, 6]
window = defaults.peak_window  # peak ERF latency window
for aix in ages:
    df = rd.return_dataframes('mmn', age=aix)[0]
    subjects = ['bad_%s' % ss for ss in df.index.values]
    print(df.info())
    scatter_matrix(df[['age', 'ses', 'headSize']], alpha=.8, grid=False)
    signals = {k: v for k, v in zip(conditions, [[], []])}
    topographies = {k: v for k, v in zip(conditions, [[], []])}
    for ii, cond in enumerate(conditions):
        print('     Fitting Xdawn for %d mos. %s data' % (aix, cond))
        hf_fname = op.join(defaults.datadir,
                           '%smos_%d-%s_%s_xdawn.h5' % (
                                   aix, lp, analysis, cond))
        patterns = list()
        for jj, subject in enumerate(subjects):
            print('     Subject: %s' % subject)
            ep_fname = op.join(workdir, subject, 'epochs',
                               'All_%d-sss_%s-epo.fif' % (lp, subject))
            cov_fname = op.join(workdir, subject, 'covariance',
                                '%s-%d-sss-cov.fif' % (subject, lp))
            # load trial data
            eps = read_epochs(ep_fname)
            times = eps.times
            assert eps.baseline is not None
            eps.pick_types(meg=True)
            # combine oddball ERFs into auditory ERF
            eps_copy = combine_event_ids(eps.copy(),
                                         [k for k in eps.event_id.keys()],
                                         {'All': 123})
            # Covariance
            signal_cov = compute_covariance(eps_copy, n_jobs=config.N_JOBS,
                                            rank='full', method='oas')
            rank = compute_rank(signal_cov, rank='full', info=eps_copy.info)
            signal_cov = regularize(signal_cov, eps_copy.info, rank=rank)
            # fit XDAWN on auditory ERF
            xd = Xdawn(signal_cov=signal_cov)
            xd.fit(eps_copy)
            event_ = list(xd.event_id_.keys())[0]
            # comtine deviant stimuli into single event
            eps = combine_events(eps, ['ba', 'wa'], {'deviant': 23})
            eps.equalize_event_counts(['standard', 'deviant'])
            if ii == jj == 0:
                signals = np.zeros((len(conditions), len(subjects),
                                    len(eps.times)))
            # apply XDAWN filters to oddball stimuli
            signals[ii, jj] = xd.transform(eps[cond])[0, 0]
            evo = eps[cond].average(method='median')
            patterns.append(xd.filters_[event_].dot(evo.data))
            write_hdf5(hf_fname,
                       dict(subjects=subjects,
                            signals=signals,
                            topographies=patterns,
                            times=times),
                       title='xdawn', overwrite=True)
