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
import mnefun
import glob
import os
from pathlib import Path
# some useful Path object idioms:
# use Path( aPathString ) to make Path object from pathname string
# use str( aPathObject ) to convert back to pathname string
# use Path( aString1 ) / Path( aString2 ) to join paths
# use also, e.g.: str( Path() / aString )
# cf. def PathGlob(...) below
from defaults import paradigms

def IfMkDir(aPath):
    if not Path(aPath).exists():
        Path(aPath).mkdir()

# helper function to make a (unsorted) list of
# subdir content in aPath found using aGlobStr:
def PathGlob(aPath, aGlobStr):
    return [str(p) for p in Path(aPath).glob(aGlobStr)]

def ParentDir(aPath):
    return str(Path(aPath).parent)

## e.g.,
#sbjSsnID = 'bad_116b'
## assuming prior "from defaults import paradigms" from /badbaby/python
#sbjSsnPath = str( Path( paradigms['assr'] ) / sbjSsnID )
#epoPath = PathGlob( sbjSsnPath, 'epochs/*-epo.fif')[0]
#sssPath = PathGlob( sbjSsnPath, 'sss_fif/*_am_raw_sss.fif')[0]
#pcaPath = PathGlob( sbjSsnPath, 'sss_pca_fif/*_am_*_raw_sss.fif')[0]
#listPath = PathGlob( sbjSsnPath, 'lists/*am-eve.fif')[0]


# make a list of paths to epoch data for all subjects and sessions;
# (this glob str will find any sbj id with 3 digits followed by "a" or "b")
#sbjSsnEpoPaths = PathGlob( tonePath, 'bad_????/epochs/*-epo.fif' )

# e.g., sbjSsnEpoPaths[16] == bad_316b/epochs/All_100-sss_bad_316b-epo.fif
# now, e.g., mne.read_epochs( sbjSsnEpoPaths[16] ).get_data().shape
# will be (106, 306, 781)


def Sss2Epo(sssPath):
    # needed to handle pilot 'bad_000' (larson_eric)
    sss = mne.io.read_raw_fif(sssPath, allow_maxshield='yes')
    events = mne.find_events(sss)
    picks = mne.pick_types(sss.info, meg=True)
    event_id = {'Auditory': 1} # is 'Auditory' correctly saved for use by Epo2Xdawn?
    tmin, tmax = -0.2, 1.1 # from process.py
    decim = 3 # from process.py
    epochs = mne.Epochs(sss, events, event_id, tmin, tmax, picks=picks,
        decim=decim, baseline=(None, 0), reject=dict(grad=4000e-13),
        preload=True)
    epoPath = sssPath.replace('sss_fif','epochs').replace('_raw_sss.fif','_epo.fif')
    epochs.save(epoPath)


def Epo2Xdawn(epoPath):
    # Given "a P(ath)F(ile) N(a)m(e)" to raw_sss.fif,
    # plot/return XDAWN responses to signal and noise
    epochs = mne.read_epochs(epoPath)
    signal_cov = mne.compute_covariance(epochs, method='oas', n_jobs=18)
    signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank='full')
    xd = mne.preprocessing.Xdawn(n_components=1, signal_cov=signal_cov,
                                 correct_overlap=False, reg='ledoit_wolf')
    xd.fit(epochs)

    # fit() creates decomposition matrix called filters_ (or "unmixing")
    # and the inverse, called "patterns_" for remixing responses after
    # supressing contribution of selected filters_ components

    # apply() method restricts the reponse to those projected on the
    # filters_ components specified in the "include" arg.  The first tNFC
    # components are the "signal" for purposes of SSNR optimization.

    # calc the signal reponses, as Evoked object
    # (by default, include=list(arange(0,1)), i.e., includes only one
    # "signal" component)
    signal = xd.apply(epochs)['Auditory'].average() # is 'Auditory' correctly saved by Sss2Epo?

    # calc the noise responses, as Evoked object
    noiseinclude = list(arange(1, epochs.info['nchan']))  # a range excluding signal "0"
    noise = xd.apply(epochs, include=noiseinclude)['Auditory'].average()

    # create arg to force both plots to have same fixed scaling
    ts_args = dict(ylim=dict(grad=[-100, 100], mag=[-500, 500]))
    signal.plot_joint(ts_args=ts_args)
    noise.plot_joint(ts_args=ts_args)

    ## fit() also computes xd.evokeds_ which seems to be the same as
    ## epochs.average(), but it's calculated in a complicated way that
    ## compensates for overlap (when present).
    ## Keep this handy to compare with xdawn results.
    #xd.evokeds_['Auditory'].average().plot_joint(ts_args=ts_args)
    #epochs.average().plot_joint(ts_args=ts_args)

    # save signal and noise
    # parent dir should be 'assr_results/' created by Sss2EPo above
    assrResPath = ParentDir(epoPath)

    xdawnEpoPath = str(Path(assrResPath) / 'xdawn_signal_epo.fif')
    signal.save(xdawnEpoPath)

    xdawnEpoPath = str(Path(assrResPath) / 'xdawn_noise_epo.fif')
    noise.save(xdawnEpoPath)


## With this above function defined, we can use a glob expression to
## to specify list of path-file name ("PFNm") strings
#
## this glob gets all a's and b's in bash, but not python:
##    bad_*{a,b}/sss_fif/*_am_raw_sss.fif
#
##aPFNmPattern = '/media/ktavabi/ALAYA/data/ilabs/badbaby/tone/bad_*b/sss_fif/*_am_raw_sss.fif'
#
#aPFNmPattern = '/media/ktavabi/ALAYA/data/ilabs/badbaby/larson_eric_tone_raw_sss.fif'
#
## Additional code needed here to check exclusion list in defaults.py
#
## Start with second visits, labeled "B"
#tPFNmsB = glob.glob(aPFNmPattern)
#
#tTest = GetSsnData(tPFNmsB[0])
#tFEvo = tTest[0]
#tNoi = tTest[1]
