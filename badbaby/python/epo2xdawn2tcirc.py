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
from mne.externals.h5io import write_hdf5

def Sss2Epo(sssPath):
    # needed to handle pilot 'bad_000' (larson_eric)
    sss = mne.io.read_raw_fif(sssPath, allow_maxshield='yes')
    events = mne.find_events(sss)
    picks = mne.pick_types(sss.info, meg=True)
    event_id = {'Auditory': 1} # 'Auditory' only for bad_000; babies are 'tone'; cf Epo2Xdawn
    tmin, tmax = -0.2, 1.1 # from process.py
    decim = 3 # from process.py
    epochs = mne.Epochs(sss, events, event_id, tmin, tmax, picks=picks,
        decim=decim, baseline=(None, 0), reject=dict(grad=4000e-13),
        preload=True)
    epoPath = sssPath.replace('sss_fif','epochs').replace('_raw_sss.fif','_epo.fif')
    # if find/replace fails, prevent overwriting the input file
    # this needs better solution
    assert sssPath == epoPath 
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
    signal = xd.apply(epochs)['tone'].average() # 'tone' is for babies;
                                                # use 'Auditory' for bad_000 pilot

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
    # if find/replace fails, prevent overwriting the input file
    # this needs better solution
    assert epoPath == xdawnPath
    mne.write_evokeds( xdawnPath, [ signal, noise ] )

def Xdawn2Tcirc(xdawnPath,tmin=None,tmax=None,fundfreqhz=None):
    # create tcirc stats from xdawn 'signal' and 'noise' that have been
    # saved into xdawnPath by Epo2Xdawn
    
    signal = mne.read_evokeds(data,allow_maxshield=True)[0].get_data() # "[0]" is 'signal'
    signalTcirc=Tcirc(signal,tmin=0.5,tmax=1.0,fundfreqhz=20.) # signal t-circ stats
    noise = mne.read_evokeds(data,allow_maxshield=True)[1].get_data() # "[1]" is 'noise'
    noiseTcirc=Tcirc(noise,tmin=0.5,tmax=1.0,fundfreqhz=20.) # noise t-circ stats
    
    signalFtzs=None # fisher transformed z-score stats,
                    # for estimating within-subject longitudinal significance
    noiseFtzs=None
    sfreqhz=None # sampling frequency in Hz (from Evoked.info['sfreq']?)
    
    # save the results
    tcircPath = xdawnPath.replace('_xdawn_ave.fif','_tcirc.h5')
    # if find/replace fails, prevent overwriting the input file
    # this needs better solution
    assert xdawnPath == tcircPath
    write_hdf5(tcircPath,
               dict(signaltcirc=signalTcirc,
                    signalftzs=signalFtzs,
                    noisetcirc=noiseTcirc,
                    noiseftzs=noiseFtzs,
                    sfreqhz=sfreqhz),
               title='tcirc', overwrite=True)

def Tcirc(data,tmin=None,tmax=None,fundfreqhz=None):
    # create tcirc stats from data N-D array
    
    # if tmin==None, tmin= 0
    # if tmax==None, tmax= end of data along time dim
    # if fundfreqhz == None, fundfreqhz = 1 / (tmax-tmin), an appropriate
    #   default if tcirc to be estimated from epoch data
    #   For evokeds, use fundfreqhz = N / (tmax-tmin) to divide into N epochs
    
    # e.g., for ASSR: ... = Tcirc(...,tmin=0.5,tmax=1.0,fundfreqhz=20.),
    # will divide the 0.5 second epoch into ten epochs each 1/20.==0.05 sec duration
    
    # Be careful about trailing samples when converting from time to array
    # index values
    
    tcirc = None
    return tcirc # sampling frequency in Hz (from Evoked objects)
    
    

