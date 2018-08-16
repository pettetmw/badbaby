# -*- coding: utf-8 -*-
"""
=========================================
Receptive Field Estimation and Prediction
=========================================

Second iteration of badbaby speech encoding script adapted from
plot_receptive_field example in mne-python. Use `mne.decoding.ReceptiveField`
class to fit a linear encoding model using the continuously-varying speech
envelope to MEG predict activity.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Kambiz Tavabi <ktavabi@uw.edu>
#

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.preprocessing import scale
from mne.decoding import ReceptiveField, TimeDelayingRidge


#   Setup for reading the raw data
def preproc_raw(raw_fname, h_freq=50.):
    this_raw = mne.io.read_raw_fif(raw_fname, allow_maxshield='yes')
    mne.channels.fix_mag_coil_types(this_raw.info)
    this_raw = mne.preprocessing.maxwell_filter(this_raw)
    this_raw.filter(None, h_freq, fir_design='firwin', n_jobs='cuda')
    if h_freq > 50:
        this_raw.notch_filter(np.arange(60, 241, 60),
                              filter_length='auto', phase='zero')
    return this_raw


event_id = {'Auditory': 1}
tmin, tmax = -0.5, 2.
decim = 2

# ASSR
print('Loading ASSR')
assr_raw = preproc_raw('larson_eric_tone_raw.fif', h_freq=50)
events = mne.find_events(assr_raw)
picks = mne.pick_types(assr_raw.info, meg=True)
epochs = mne.Epochs(assr_raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=3500e-13),
                    preload=True)
epochs.average().plot()
signal_cov = mne.compute_covariance(epochs, tmin=0, tmax=1.5)
signal_cov = mne.cov.regularize(signal_cov, assr_raw.info)
print('Fitting Xdawn')
xd = mne.preprocessing.Xdawn(n_components=1, signal_cov=signal_cov,
                             correct_overlap=False, reg='ledoit_wolf')
xd.fit(epochs)
xd.apply(epochs)['Auditory'].average().plot()

# Speech
print('Loading speech')
tmin, tmax = 0, 20
fmin, fmax = 2, 300
raw = preproc_raw('larson_eric_story_01_raw.fif', h_freq=100.)
mne.chpi.filter_chpi(raw)
n_channels = len(raw.ch_names)
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')
speech_events = mne.find_events(raw, verbose=True)
speech_epochs = mne.Epochs(raw, events=speech_events, event_id=1,
                           tmin=0., tmax=50., picks=picks)
virtual_channels = xd.transform(speech_epochs)[0, 0]
raw.crop(tmin, tmax).load_data()
raw.load_data().resample(1000, npad="auto")  # set sampling frequency to 1000Hz # noqa: E501
raw.plot_psd(area_mode='range', picks=None, average=False,
             fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, n_fft=2048)
sfreq = wavfile.read('inForest_part-1-rms.wav')[0]
assert sfreq == 24414
sfreq = 24414.0625  # corrected rate
speech = np.concatenate([wavfile.read('inForest_part-%d-rms.wav' % k)[1]
                         for k in range(1, 6)])
speech = mne.filter.resample(speech, down=decim, npad='auto')
envelope = speech.copy()
envelope[envelope < 0] = 0.  # half-wave rectify
assert np.isfinite(envelope).all()

# Plot brain and stimulus activity
fig, ax = plt.subplots()
lns = ax.plot(scale(raw[:, :300][0].T), color='k', alpha=.05)
ln1 = ax.plot(scale(envelope[:300]), color='r', lw=2, alpha=.5)
ln2 = ax.plot(scale(virtual_channels.T[:300]), color='b', lw=2)
ax.legend([lns[0], ln1[0], ln2[0]], ['MEG', 'Speech Envelope',
                                     'Virtual Channel'], frameon=False)
ax.set(title="Sample activity", xlabel="Time (s)")
mne.viz.tight_layout()

# Create and fit a receptive field model
# Time delays to use in the receptive field
tmin, tmax = -1, 0

# Initialize the model
interpolator = interp1d(np.arange(len(envelope)) / sfreq, envelope,
                        fill_value=0., bounds_error=False, assume_sorted=True)
envelope_rs = interpolator(speech_epochs.times)
envelope_rs[0] = 0.
assert np.isfinite(envelope_rs).all()
est = TimeDelayingRidge(tmin, tmax, epochs.info['sfreq'], 1., 'laplacian')
rf = ReceptiveField(tmin, tmax, epochs.info['sfreq'], estimator=est,
                    scoring='corrcoef')
n_delays = int((tmax - tmin) * sfreq) + 2

rf.fit(envelope_rs[:, np.newaxis], virtual_channels)
score = rf.score(envelope_rs[:, np.newaxis], virtual_channels)
coefs = rf.coef_[0, :]
times = rf.delays_ / float(rf.sfreq)

fig, ax = plt.subplots()
ax.plot(times, coefs)
ax.axhline(0, ls='--', color='r')
ax.set(title="WTF is this?", xlabel="time (s)",
       ylabel="($r$)")
mne.viz.tight_layout()
plt.show()