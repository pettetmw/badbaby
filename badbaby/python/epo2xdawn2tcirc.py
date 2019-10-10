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


def Epo2Xdawn(epoPath,xdawnPath=None):
    # Compute and save (into xdawnPath) XDAWN responses to signal and noise,
    # given epoch data in epoPath; returns status string
    
    # Determine destination path
    if xdawnPath == None:
        # try to derive destination file name from source file name
        xdawnPath = epoPath.replace('-epo.fif','_xdawn_ave.fif')
        
    if xdawnPath == epoPath: # e.g., if find/replace fails, or if incorrect args
        # prevent overwriting epoPath
        errMsg = basename(epoPath) + ' --> error: xdawnPath would overwrite epoPath'
        print( errMsg )
        return errMsg
    
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
    
    try:
        mne.write_evokeds( xdawnPath, [ signal, noise ] )
    except:
        errMsg = basename(epoPath) + ' --> error writing ' + xdawnPath
        print( errMsg )
        return errMsg
    
    # Everything worked, so return status string
    return basename(epoPath) + ' --> ' + basename(xdawnPath)

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
    # create tcirc stats from data N-D array (n_epochs, n_channels, n_times)
    # note that np.fft.fft "axis" 
    
    
    # if tmin==None, tmin= 0
    # if tmax==None, tmax= end of data along time dim
    # if fundfreqhz == None, fundfreqhz = 1 / (tmax-tmin), an appropriate
    #   default if tcirc to be estimated from epoch data
    #   For evokeds, use fundfreqhz = N / (tmax-tmin) to divide into N epochs
    
    # e.g., for ASSR: ... = Tcirc(...,tmin=0.5,tmax=1.0,fundfreqhz=20.),
    # will divide the 0.5 second epoch into ten epochs each 1/20.==0.05 sec duration
    
    # Be careful about trailing samples when converting from time to array
    # index values

    tY = data[ :, :, :-1 ] # remove odd sample (make this conditional)
    #plt.plot(np.mean(tY,axis=-1),'r-')
    if tY.ndim == 2:
        tNCh = tY.shape
        tNEp = 5; # compute from fundfreqhz
        tY = np.reshape( tD, [ tNCh, -1, tNEp ] )
        
    tNCh, tNS, tNTrl = tY.shape
    
    tSR = # maybe needs argument
    
    tXFrq = np.round( np.fft.fftfreq( tNS, 1.0/tSR ), 2 ) # X Freq values for horizontal axis of plot
    
    tMYFFT = np.fft.fft( np.mean( tY, axis=1 ) ) / tNS # FFT of mean over trials of tY, Chan-by-Freq
    tYFFT = np.fft.fft( tY, axis=1 ) / tNS # FFT of tY along time sample dim, Chan-by-Freq-by-Trials
    
    tYFFTV = np.mean( np.stack( ( np.var( np.real(tYFFT), 0 ), np.var( np.imag(tYFFT), 0 ) ) ), 0 )
    #tYFFTV = np.var( abs(tYFFT), 0 )
    tcirc = abs(tMYFFT) / np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    
    return tcirc # sampling frequency in Hz (from Evoked objects)
    
    

