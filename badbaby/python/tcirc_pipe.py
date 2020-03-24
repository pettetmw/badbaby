#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:33:33 2020

@author: pettetmw
"""

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.externals.h5io import write_hdf5
#import mnefun
from fake import *

def TCirc(atcevop,aepop,tmin=None,tmax=None,fundfreqhz=None):
    # create a tcirc evoked file (atcevop) from epoched file (aepop)
    # note that np.fft.fft "axis" parameter defaults to -1
    
    # Be careful about trailing samples when converting from time to array
    # index values

    #plt.plot(np.mean(tY,axis=-1),'r-')
    
    epo = mne.read_epochs(aepop)
    
    info = epo.info
    sfreq = info['sfreq'] # to compute location of tmin,tmax
    imn = int( sfreq * tmin )
    imx = int( sfreq * tmax )
    
    tY = epo.get_data()[ :, :, imn:imx ] # remove odd sample (make this conditional)

    tNTrl, tNCh, tNS = tY.shape # number of trials, channels, and time samples
    
    tMYFFT = np.fft.fft( np.mean( tY, axis=0 ) ) / tNS # FFT of mean over trials of tY, Chan-by-Freq
    tYFFT = np.fft.fft( tY ) / tNS # FFT of tY , Trials-by-Chan-by-Freq
    
    # compute the mean of the variances along real and imaginary axis
    tYFFTV = np.mean( np.stack( ( np.var( np.real(tYFFT), 0 ), np.var( np.imag(tYFFT), 0 ) ) ), 0 )
    #tYFFTV = np.var( abs(tYFFT), 0 )
    numerator = abs(tMYFFT);
    denominator = np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    #tcirc = abs(tMYFFT) / np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    tcirc = numerator / denominator
    
    info['sfreq']=0.5
    mne.EvokedArray( tcirc, info ).save( atcevop )
    return tcirc 

def fPlotImage(p):
    # assumes instance of mne.Report called "report" in enclosing scope (see below)
    tF = mne.read_evokeds(p)[0].plot_image(xlim=[0,50])
    for a in zip( tF.get_axes()[0:2], [210,110] ) : a[0].set_ylim([0,a[1]])
    caption = os.path.basename(p)
    report.add_figs_to_section( tF, caption, section='TCirc channel-by-freq' )
    # print( 'plotting ' + os.path.basename(p) )

def fPlotERP(p):
    # assumes instance of mne.Report called "report" in enclosing scope (see below)
    tF = mne.read_epochs(p).plot_psd(fmax=50)
    for a in tF.get_axes()[0:2] : a.set_ylim([30,70])
    caption = os.path.basename(p)
    report.add_figs_to_section( tF, caption, section='TCirc channel-by-freq' )
    # print( 'plotting ' + os.path.basename(p) )

def makeReportFromTcircevos( areportp, *tcevops) :
    # assumes instance of mne.Report called "report" in enclosing scope (see below)
    # tMLFP = [ getLowgfpFromTcircevo(p) for p in tcevops ]
    tcevops = sorted( tcevops, key=getLowgfpFromTcircevo )
    [ fPlotImage(p) for p in tcevops ];
    report.save(areportp, open_browser=False, overwrite=True)
    
def makeCompreportFromEpos(areportp, aps) :
    # [ print( p ) for p in atcevops ]
    tps = sorted( aps, key=lambda ps: getLowgfpFromTcircevo(ps.split()[0]) )
    for ps in tps: 
        fPlotImage(ps.split()[0]) 
        fPlotImage(ps.split()[1])
        fPlotERP(ps.split()[2])
        fPlotERP(ps.split()[3])
    report.save(areportp, open_browser=False, overwrite=True)
    
def getLowgfpFromTcircevo( atcevop ):
    return mne.read_evokeds(atcevop)[0].data.mean(0)[0:3].sum()

# The rest assumes Makefile exists in this module's directory,
# with following rules:
    
# tcircevo/%b-ave.fif : tcircevo/%a-ave.fif
# 	echo $@

# 2MoReport : tcircevo/*a-ave.fif
# 	echo makeReportFromTcircevos 2MoTCircs.html $?

# 6MoReport : tcircevo/*b-ave.fif
# 	echo makeReportFromTcircevos 6MoTCircs.html $?

# tcircevo/%-ave.fif : ../tone/*/epochs/%-epo.fif
# 	echo makeTcircevoFromEpo $@ $<



# define wrapper to accept target path t, and prereq path p:
makeTcircevoFromEpo=lambda t,p: TCirc(t,p,tmin=0.5,tmax=1.0)



fakefnx.update( dict(makeTcircevoFromEpo=makeTcircevoFromEpo,
                     makeReportFromTcircevos=makeReportFromTcircevos) ) # and make it fake-able

#fakeout('../tone/*/epochs/*-epo.fif', [r'../tone/.+/epochs/','tcircevo/'], ['epo.fif','ave.fif'] ) # print only
#fakeit('../tone/*/epochs/*-epo.fif', [r'../tone/.+/epochs/','tcircevo/'], ['epo.fif','ave.fif'] ) # actually do it

# reports must be run from here so that 'report' variable has correct scope

# fakeout('2MoReport')
# report = mne.Report(); fakeit('2MoReport')
# fakeout('6MoReport')
# report = mne.Report(); fakeit('6MoReport')

# also handy for forcing or preventing updates:
# 
# lnx('touch tcircevo/*.fif')
# lnx('touch ../tone/*/epochs/*-epo.fif')

# Watch out for those "%" when pasting from Makefile... you really prefer "*"
# instead for bash globbing

# fakeit('tcircevo/*b-ave.fif')

# report = mne.Report();
# makeCompreportFromTcircevos( '2to6MoCompTc.html', fake('tcircevo/*b-ave.fif') )

report = mne.Report();
makeCompreportFromEpos( '2to6MoComp.html', fake('epofif/*b-epo.fif') )






















