#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sssraw2xdawnfilt.py

"""
Created on Fri Jun 28 07:27:29 2019

@author: mpettet
"""

# !/usr/bin/env python

"""Gist to fit XDAWN filter to ASSR response and viz ERF topography."""

# based on xdawn_assr.py, except that it starts with raw_sss.fif
# (rather than raw.fif), so we skip maxwell filter

# __author__ = "Kambiz Tavabi"
# __copyright__ = "Copyright 2019, Seattle, Washington"
# __credits__ = ["Eric Larson"]
# __license__ = "MIT"
# __version__ = "1.0.1"
# __maintainer__ = "Kambiz Tavabi"
# __email__ = "ktavabi@uw.edu"
# __status__ = "Production"

import mne
import glob


event_id = {'Auditory': 1}
tmin, tmax = -0.5, 2.


def GetSsnData(aPFNm):
    # Given "a P(ath)F(ile) N(a)m(e)" to raw_sss.fif,
    # plot/return XDAWN responses to signal and noise
    try:
        raw = mne.io.read_raw_fif(aPFNm, allow_maxshield='yes')

        events = mne.find_events(raw)
        picks = mne.pick_types(raw.info, meg=True)
        tNCh = picks.size  # number of channels
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            baseline=(None, 0), reject=dict(grad=4000e-13),
                            preload=True)
        # epochs.average().plot()

        # signal_cov = mne.compute_covariance(epochs, tmin=0, tmax=1.5)
        # signal_cov = mne.cov.regularize(signal_cov, raw.info)
        signal_cov = mne.compute_covariance(epochs, method='oas', n_jobs=18)
        signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank='full')

        print('Fitting Xdawn')
        tNFC = 1  # number of xdawn filter components
        xd = mne.preprocessing.Xdawn(n_components=tNFC, signal_cov=signal_cov,
                                     correct_overlap=False, reg='ledoit_wolf')
        xd.fit(epochs)

        # fit() creates decomposition matrix called filters_ (or "unmixing")
        # and the inverse, called "patterns_" for remixing responses after
        # supressing contribution of selected filters_ components

        # apply() method restricts the reponse to those projected on the
        # filters_ components specified in the "include" arg.  The first tNFC
        # components are the "signal" for purposes of SSNR optimization.

        # calc "t(he) F(iltered) Evo(ked)" reponses
        # (by default, include=list(arange(0,tNFC)), i.e., the "signal")
        tFEvo = xd.apply(epochs)['Auditory'].average()

        noiseinclude = list(arange(tNFC, tNCh))  # a range to include
        # calc "t(he) Noi(se)"
        tNoi = xd.apply(epochs, include=noiseinclude)['Auditory'].average()
        
        # create arg to force both plots to have same fixed scaling
        ts_args = dict(ylim=dict(grad=[-100, 100], mag=[-500, 500]))
        tFEvo.plot_joint(ts_args=ts_args)
        tNoi.plot_joint(ts_args=ts_args)

#        # fit() also computes xd.evokeds_ which seems to be the same as
#        # epochs.average(), but it's calculated in a complicated way that
#        # compensates for overlap (when present).
#        # Keep this handy to compare with xdawn results.
#        xd.evokeds_['Auditory'].average().plot_joint(ts_args=ts_args)
#        epochs.average().plot_joint(ts_args=ts_args)

    except:
        tFEvo = None
        tNoi = None

    return tFEvo, tNoi


# With this above function defined, we can use a glob expression to
# to specify list of path-file name ("PFNm") strings

# this glob gets all a's and b's in bash, but not python:
#    bad_*{a,b}/sss_fif/*_am_raw_sss.fif

#aPFNmPattern = '/media/ktavabi/ALAYA/data/ilabs/badbaby/tone/bad_*b/sss_fif/*_am_raw_sss.fif'

aPFNmPattern = '/media/ktavabi/ALAYA/data/ilabs/badbaby/larson_eric_tone_raw_sss.fif'

# Additional code needed here to check exclusion list in defaults.py

# Start with second visits, labeled "B"
tPFNmsB = glob.glob(aPFNmPattern)

tTest = GetSsnData(tPFNmsB[0])
tFEvo = tTest[0]
tNoi = tTest[1]
