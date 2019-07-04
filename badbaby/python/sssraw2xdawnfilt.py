#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sssraw2xdawnfilt.py

"""
Created on Fri Jun 28 07:27:29 2019

@author: mpettet
"""

#!/usr/bin/env python

"""Gist to fit XDAWN filter to ASSR response and viz ERF topography."""

# based on xdawn_assr.py, except that it starts with raw_sss.fif
# (rather than raw.fif), so we skip maxwell filter

#__author__ = "Kambiz Tavabi"
#__copyright__ = "Copyright 2019, Seattle, Washington"
#__credits__ = ["Eric Larson"]
#__license__ = "MIT"
#__version__ = "1.0.1"
#__maintainer__ = "Kambiz Tavabi"
#__email__ = "ktavabi@uw.edu"
#__status__ = "Production"

import mne
import glob


event_id = {'Auditory': 1}
tmin, tmax = -0.5, 2.

# ASSR
print('Loading ASSR')

def GetSsnData( aPFNm ):
    try:
        raw = mne.io.read_raw_fif( aPFNm, allow_maxshield='yes' );
        
        events = mne.find_events(raw)
        picks = mne.pick_types(raw.info, meg=True)
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            baseline=(None, 0), reject=dict(grad=4000e-13),
                            preload=True)
        #epochs.average().plot()
        
        # signal_cov = mne.compute_covariance(epochs, tmin=0, tmax=1.5)
        # signal_cov = mne.cov.regularize(signal_cov, raw.info)
        signal_cov = mne.compute_covariance(epochs, method='oas', n_jobs=18)
        signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank='full')
        
        print('Fitting Xdawn')
        xd = mne.preprocessing.Xdawn(n_components=1, signal_cov=signal_cov,
                                     correct_overlap=False, reg='ledoit_wolf')
        xd.fit(epochs)
        
        #xd.apply(epochs)['Auditory'].average().plot_topo()
        
        tEI = epochs.info;
        tEI['sfreq']=1; # we want each second to represent a different row of filter matrix
        
        tSig = np.dot( xd.filters_['Auditory'][0,:], epochs.average().data ).T;
        tNoi = np.dot( xd.filters_['Auditory'][1:,:].mean(0), epochs.average().data ).T;
        tSNTopo = np.array( [ xd.filters_['Auditory'][:,0].T, xd.filters_['Auditory'][:,1:].mean(1).T ] ).T;
        
#        # these two plots work, but better choice would be plot_joint()
#        mne.EvokedArray( tSNTopo, tEI ).plot_topomap(times=[0,1]);
#        figure(); plot( epochs.times, np.array([ tSig, tNoi ]).T ); legend( [ 'signal', 'noise' ] );
        
    except:
        tSig = None; tNoi = None; tSNTopo = None;
    
    return tSig, tNoi, tSNTopo


# this glob gets all a's and b's in bash, but not pythong: bad_*{a,b}/sss_fif/*_am_raw_sss.fif
aPFNmPattern = '/media/ktavabi/ALAYA/data/ilabs/badbaby/tone/bad_*b/sss_fif/*_am_raw_sss.fif'

# Additional code needed here to check exclusion list in defaults.py

# Start with second visits, labeled "B"
tPFNmsB = glob.glob( aPFNmPattern );
tBadB = [ GetSsnData( fp ) for fp in tPFNmsB ];

# Now repeat with first visits, "A"
tPFNmsA = [ x.replace( 'b/sss', 'a/sss' ).replace( 'b_am', 'a_am' ) for x in tPFNmsB ];
tBadA = [ GetSsnData( fp ) for fp in tPFNmsA ];

tSigAs = [ x[0] for x in tBadA ]; 
tNoiAs = [ x[1] for x in tBadA ];
tSNTopoAs = [ x[2] for x in tBadA ];

tSigBs = [ x[0] for x in tBadB ];
tNoiBs = [ x[1] for x in tBadB ];
tSNTopoBs = [ x[2] for x in tBadB ];








