# %%
import os
import numpy as np

from mne.time_frequency import tfr_morlet

import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
from hotelling.stats import hotelling_t2 as hott2
import mne
from mne.externals.h5io import write_hdf5


# %%
def t2circ_from_epo_tfr(epo_path):
    # epo_path : subject's epo file path
    if epo_path[-3:] == 'fif':
        epo = mne.read_epochs(epo_path)['tone']
        freqs = np.arange(34.0,48.0,2.0)
        n_cycles = freqs / 2.  
        # complex EpochsTFR
        tfr_epo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, output='complex')
    else:
        tfr_epo = mne.time_frequency.read_tfrs( epo_path )[0]

    info = tfr_epo.info
    ntrl, nch, nfrq, nsamp = tfr_epo.data.shape
    times = tfr_epo.times
    freqs = tfr_epo.freqs

    # # Hotelling's T-squared
    # hot_evo = np.ndarray((nch, nfrq, nsamp))
    # for s in np.arange(nsamp):
    #     for f in np.arange(nfrq):
    #         for c in np.arange(nch):
    #             x = [ np.real(tfr_epo.data[:,c,f,s]), np.imag(tfr_epo.data[:,c,f,s]) ]
    #             x = np.transpose(x)
    #             hot = hott2(x)
    #             hot_evo.data[c,f,s] = hot[0]
	# return hot_evo

    # T-squared-"circ", after Victor & Mast 1991
    x = np.real(tfr_epo.data)
    y = np.imag(tfr_epo.data)
    M = ntrl
    x_est = np.mean(x,0) # marginal sample means
    y_est = np.mean(y,0)
    xdev = x - np.tile(x_est,(M,1,1,1)) # marginal sample deviations
    ydev = y - np.tile(y_est,(M,1,1,1))
    Vi = np.sum( xdev**2 + ydev**2, 0 ) # pooled marginal variances
    Vg = ( x_est**2 + y_est**2 ) * M / 2 # variance attributable to sample means
    t2circ = Vg / Vi / M # distributed as F(2,2M-2)
    t2circ_evo = mne.time_frequency.AverageTFR(info,t2circ,times,freqs,ntrl)
    return t2circ_evo


# %%
# t2c = makeTFRfromEpo('/mnt/ilabs/badbaby/data/bad_116a/epochs/All_50-sss_bad_116a-tfr-epo-tfr.h5')
# print(t2c)

