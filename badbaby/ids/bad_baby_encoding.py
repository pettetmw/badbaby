# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:49:39 2017

@authors: Eric larson
          Kambiz Tavabi <ktavabi@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import wavfile
from scipy.interpolate import interp1d

datapath = '/Users/ktavabi/Data/badbaby/'
event_id = {'Auditory': 1}
raw_fname = datapath + 'larson_eric_tone_raw.fif'
tmin, tmax = -0.5, 2.


#   Setup for reading the raw data
def preproc_raw(raw_fname, h_freq=50.):
    this_raw = mne.io.read_raw_fif(raw_fname, allow_maxshield='yes')
    mne.channels.fix_mag_coil_types(this_raw.info)
    this_raw = mne.preprocessing.maxwell_filter(this_raw)
    this_raw.filter(None, h_freq, fir_design='firwin', n_jobs='cuda')
    return this_raw


# ASSR
print('Loading ASSR')
raw = preproc_raw(raw_fname)
events = mne.find_events(raw)
picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13),
                    preload=True)
epochs.average().plot()
signal_cov = mne.compute_covariance(epochs, tmin=0, tmax=1.5)
signal_cov = mne.cov.regularize(signal_cov, raw.info)
print('Fitting Xdawn')
xd = mne.preprocessing.Xdawn(n_components=1, signal_cov=signal_cov,
                             correct_overlap=False, reg='ledoit_wolf')
xd.fit(epochs)
xd.apply(epochs)['Auditory'].average().plot_topo()

# Speech
print('Loading speech')
raw_speech = preproc_raw(datapath + 'larson_eric_story_01_raw.fif')
sfreq = wavfile.read(datapath + 'paradigm/inForest_part-1-rms.wav')[0]
assert sfreq == 24414
sfreq = 24414.0625  # corrected rate
sounds = np.concatenate([wavfile.read('inForest_part-%d-rms.wav' % k)[1]
                         for k in range(1, 6)])
envelope = sounds.copy()
envelope[envelope < 0] = 0.  # half-wave rectify + lowpass
envelope = mne.filter.filter_data(envelope, sfreq, 0., 10.,
                                  fir_design='firwin', n_jobs='cuda')
assert np.isfinite(envelope).all()
interpolator = interp1d(np.arange(len(envelope)) / sfreq, envelope,
                        fill_value=0., bounds_error=False, assume_sorted=True)
events_sound = mne.find_events(raw_speech, verbose=True)
epochs_sound = mne.Epochs(raw_speech, events=events_sound, event_id=1,
                          tmin=0., tmax=50., picks=picks, decim=10)
virtual_channels = xd.transform(epochs_sound[0])[0, 0]
assert np.isfinite(virtual_channels.all())
envelope_rs = interpolator(epochs_sound.times)
envelope_rs[0] = 0.
assert np.isfinite(envelope_rs).all()
est = mne.decoding.TimeDelayingRidge(-0.2, 0.4, epochs.info['sfreq'], 1.,
                                     'laplacian')
rf = mne.decoding.ReceptiveField(
    tmin=-0.200, tmax=0.400, sfreq=epochs.info['sfreq'], estimator=est)
print('Fitting model')
rf.fit(envelope_rs[:, np.newaxis], virtual_channels)
print(rf.coef_.shape)
fig, ax = plt.subplots(1)
ax.plot(rf.delays_ / epochs.info['sfreq'], rf.coef_[0])
