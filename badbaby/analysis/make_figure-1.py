# %%
import os
from os import path as op
import matplotlib.pyplot as plt
import mne
from badbaby import defaults

# %%
def read_in_evoked(filename, condition):
    """helper to read evoked file"""
    erf = read_evokeds(filename, condition=condition,
                       baseline=(None, 0))
    if erf.info['sfreq'] > 600.0:
        raise ValueError('Wrong sampling rate!')
    if len(erf.info['bads']) > 0:
        erf.interpolate_bads()
    return erf

# %%
plt.style.use('ggplot')
workdir = defaults.datadir
tmin, tmax = defaults.epoching
lp = defaults.lowpass
window = defaults.peak_window  # peak ERF latency window

# %%
df = defaults.return_dataframes('mmn')[0]
subjects = ['bad_%s' % ss for ss in df[df['age'] > 150].index]  # sixers
evokeds = {'standard': [], 'deviant': []}
for condition in evokeds.keys():
    evs = list()
    for si, subj in enumerate(subjects):
        print(' %s' % subj)
        evoked_file = op.join(defaults.datadir, subj, 'inverse',
                                'Oddball_%d-sss_eq_%s-ave.fif' % (lp, subj))
        evs.append(read_in_evoked(evoked_file)
    # do grand averaging
    print('  Doing %s averaging.' % condition)
    evokeds[condition].append(grand_average(evs))

# %%
peak = evokeds['deviant'].get_peak(ch_type='grad', tmin=window[0], tmax=window[1])


