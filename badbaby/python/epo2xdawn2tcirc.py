#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:05:05 2019

@author: pettetmw
"""

# epo2xdawn2tcirc.py

import mne
import mnefun
import os
from numpy import arange

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
    # plot/return XDAWN responses to signal and noise
    epochs = mne.read_epochs(epoPath)
    epochs.pick_types(meg=True)
    signal_cov = mne.compute_covariance(epochs, method='oas', n_jobs=18)
#    signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank='full')
    
    rank = mne.compute_rank(signal_cov, rank='full', info=epochs.info)
    signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank=rank)
    
    
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
    signal = xd.apply(epochs)['tone'].average() # is 'Auditory' correctly saved by Sss2Epo?

    # calc the noise responses, as Evoked object
    noiseinclude = list(arange(1, epochs.info['nchan']))  # a range excluding signal "0"
    noise = xd.apply(epochs, include=noiseinclude)['tone'].average()

    ## create arg to force both plots to have same fixed scaling
    #ts_args = dict(ylim=dict(grad=[-100, 100], mag=[-500, 500]))
    #signal.plot_joint(ts_args=ts_args)
    #noise.plot_joint(ts_args=ts_args)

    ## fit() also computes xd.evokeds_ which seems to be the same as
    ## epochs.average(), but it's calculated in a complicated way that
    ## compensates for overlap (when present).
    ## Keep this handy to compare with xdawn results.
    #xd.evokeds_['Auditory'].average().plot_joint(ts_args=ts_args)
    #epochs.average().plot_joint(ts_args=ts_args)

    # save signal and noise
    # first, replace "Auditory" tag with "signal" and "noise"
    signal.comment = 'signal'
    noise.comment = 'noise'
    xdawnPath = epoPath.replace('-epo.fif','_xdawn_ave.fif')
    mne.write_evokeds( xdawnPath, [ signal, noise ] )

