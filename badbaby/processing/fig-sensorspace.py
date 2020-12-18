# %%
from os import path as op
import matplotlib.pyplot as plt
import mne
from mne.utils.numerics import grand_average
from badbaby import defaults

plt.style.use("ggplot")
workdir = defaults.datadir
tmin, tmax = defaults.epoching
lp = defaults.lowpass
window = defaults.peak_window  # peak ERF latency window

# %%
evokeds = {"standard": [], "deviant": []}
for condition in evokeds.keys():
    for si, subj in enumerate(defaults.cohort_six):
        print(" %s" % subj)
        filename = op.join(
            defaults.datadir,
            subj,
            "inverse",
            "Oddball_%d-sss_eq_%s-ave.fif" % (lp, subj),
        )
        erf = mne.read_evokeds(
            filename,
            condition=condition,
            baseline=(None, 0)
        )
        evokeds[condition].append(erf)

# %%
ERFS = dict()
for kk in evokeds.keys():
    ERFS[kk] = grand_average(evokeds[kk])
peak = ERFS["deviant"].get_peak(
                                ch_type="grad",
                                tmin=window[0],
                                tmax=window[1]
                                )
mne.viz.plot_compare_evokeds(ERFS, combine="gfp", ci=0.9)

# TODO
# [] group level MMN interval windowing around deviant peak latency
# [] random individual (3) subjects topoplots w/ focus on peak deviant ERF
