# %%
import os
from os import path as op
import matplotlib.pyplot as plt
import mne
from mne.utils.numerics import grand_average
from badbaby.defaults import (
    datadir,
    epoching,
    lowpass,
    peak_window,
    cohort_six
)

# TODO compute rolling window average ERF magnitude from stimulus onset and feed to classifer.

# %%
plt.style.use("ggplot")
workdir = datadir
tmin, tmax = epoching
lp = lowpass
window = peak_window  # peak ERF latency window

evokeds = {"standard": [], "deviant": []}
df = cohort_six
subjects = ["bad_%s" % pick for pick in df["id"]]
# averaging gist c/o Larson
conditions = ['standard', 'deviant']
evoked_dict = dict((key, list()) for key in conditions)
for condition in conditions:
    for subject in subjects:
        fname = op.join(datadir, subject, 'inverse',
                        f'Oddball_30-sss_eq_{subject}-ave.fif')
        evoked_dict[condition].append(mne.read_evokeds(fname, condition))
evoked = mne.combine_evoked(
    [mne.grand_average(evoked_dict[condition])
     for condition in conditions], weights=[-1, 1])
evoked.pick_types(meg=True)
evoked.plot_joint()

# %%
ERFS = dict()
for kk in evokeds.keys():
    ERFS[kk]=mne.grand_average(evokeds[kk])
peak = ERFS['deviant'].get_peak(ch_type='grad', tmin=window[0], 
                                tmax=window[1])
mne.viz.plot_compare_evokeds(ERFS, combine='gfp', ci=0.9)

#TODO 
# [] group level MMN interval windowing around deviant peak latency
# [] individual subject data for ERF conditions
