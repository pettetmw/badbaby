# %%
import os
from os import path as op

import matplotlib.pyplot as plt
import mne
from badbaby.defaults import (cohort_six, datadir, epoching, lowpass,
                              peak_window)
from mne.utils.numerics import grand_average


# %%
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
for kk in evoked_dict.keys():
    ERFS[kk] = mne.grand_average(evoked_dict[kk])
mne.viz.plot_compare_evokeds(ERFS, combine='gfp', ci=0.9)

# %%
ERF = mne.combine_evoked([ERFS[cc] for cc in conditions], weights=[-1, 1])
ERF.pick_types(meg=True)
ERF.plot_joint()

# TODO
# [] group level MMN interval windowing around deviant peak latency
# [] rolling average ERF magnitude in peak latency window for classifier.
# [] individual subject data for ERF conditions
