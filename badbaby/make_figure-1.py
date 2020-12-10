# %%
import os
from os import path as op
import matplotlib.pyplot as plt
import mne
from mne.utils.numerics import grand_average
from badbaby import defaults

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
    for si, subj in enumerate(subjects):
        print(' %s' % subj)
        filename = op.join(defaults.datadir, subj, 'inverse','Oddball_%d-sss_eq_%s-ave.fif' % (lp, subj))
        erf = mne.read_evokeds(filename, condition=condition,
                       baseline=(None, 0))
        if erf.info['sfreq'] > 600.0:
            raise ValueError('Wrong sampling rate!')
        if len(erf.info['bads']) > 0:
            erf.interpolate_bads()
        evokeds[condition].append(erf)
print(evokeds)

# %%
ERFS = dict()
for kk in evokeds.keys():
    ERFS[kk]=mne.grand_average(evokeds[kk])
peak = ERFS['deviant'].get_peak(ch_type='grad', tmin=window[0], 
                                tmax=window[1])
mne.viz.plot_compare_evokeds(ERFS, combine='gfp')                               
# %%
