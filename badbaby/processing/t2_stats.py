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
def t2_circ(data):
    # T-squared-"circ", after Victor & Mast 1991
    x = np.real(data)
    y = np.imag(data)
    m = data.shape[0]
    x_est = np.mean(x,0) # marginal sample means
    y_est = np.mean(y,0)
    x_dev = x - np.tile(x_est,(m,1,1,1)) # marginal sample deviations
    y_dev = y - np.tile(y_est,(m,1,1,1))
    v_i = np.sum( x_dev**2 + y_dev**2, 0 ) # pooled marginal variances (i=="indiv")
    v_g = ( x_est**2 + y_est**2 ) * m / 2 # variance attributable to sample means (g=="group")
    t2_circ = v_g / v_i / m # distributed as F(2,2M-2)
    return t2_circ

# %%
def t2_hot(data):
    # VERY slow
    ntrl, nch, nfrq, nsamp = data.shape

    # Hotelling's T-squared
    t2_hot = np.ndarray((nch, nfrq, nsamp))
    ss = np.arange(nsamp)
    fs = np.arange(nfrq)
    cs = np.arange(nch)
    for s in ss:
        for f in fs:
            for c in cs:
                x = [ np.real(data[:,c,f,s]), np.imag(data[:,c,f,s]) ]
                x = np.transpose(x)
                hot = hott2(x)
                t2_hot.data[c,f,s] = hot[0]
                #t2_hot[c,f,s] = hott2(np.transpose([ np.real(data[:,c,f,s]), np.imag(data[:,c,f,s]) ]))[0]
    return t2_hot

# %% Some code for testing (cf. test_t2circ_function.ipynb):
# def t2_evo_from_epo_path(epo_path):
#     # epo_path : subject's epo file path (can be Epochs or EpochsTFR)
#     # returns : AverageTFR with data from t2_circ() or t2_hot()

#     # Either: create the complex EpochsTFR from Epochs file
#     if epo_path[-3:] == 'fif': 
#         epo = mne.read_epochs(epo_path)['tone']
#         freqs = np.arange(34.0,48.0,2.0)
#         n_cycles = freqs / 2.
#         tfr_epo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
#             return_itc=False, average=False, output='complex')
#     # Or: read in the saved complex EpochsTFR
#     else: 
#         tfr_epo = mne.time_frequency.read_tfrs( epo_path )[0]
#
#     info = tfr_epo.info
#     ntrl = tfr_epo.data.shape[0]
#     times = tfr_epo.times
#     freqs = tfr_epo.freqs
#     t2 = t2_circ(tfr_epo.data)
#     #t2 = t2_hot(tfr_epo.data) # very slow
#     t2_evo = mne.time_frequency.AverageTFR(info,t2,times,freqs,ntrl)
#     return t2_evo
# 
# # read original time domain epochs (takes about 1 min)
# t2c = t2_evo_from_epo_path('/mnt/ilabs/badbaby/data/bad_116a/epochs/All_50-sss_bad_116a-epo.fif')
# print(t2c)

# # read saved tfr epochs (takes about 4 sec)
# t2c = t2_evo_from_epo_path('/mnt/ilabs/badbaby/data/bad_116a/epochs/All_50-sss_bad_116a-tfr-epo-tfr.h5')
# print(t2c)

