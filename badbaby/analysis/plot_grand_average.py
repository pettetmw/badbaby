#!/usr/bin/env python

"""plot_grand_average.py: viz grand averaged ERF data."""

import os
from os import path as op
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from meeg_preprocessing import config

from badbaby import defaults

plt.style.use('ggplot')

# parameters
workdir = defaults.datapath
analysese = ['Individual', 'Oddball']
tmin, tmax = defaults.epoching
lp = defaults.lowpass
ages = [2, 6]
window = defaults.peak_window  # peak ERF latency window
for iii, analysis in enumerate(analysese):
    if iii == 0:
        conditions = ['standard', 'ba', 'wa']
    else:
        conditions = ['standard', 'deviant']
    evoked_dict = dict()
    picks_dict = dict()
    for aii, aix in enumerate(ages):
        evoked_dict[aix] = dict()
        picks_dict[aix] = dict()
        for ci, cond in enumerate(conditions):
            evo_in = op.join(workdir, '%s-%s_%d_%dmos_grp-ave.fif' % (analysis,
                                                                      cond, lp,
                                                                      aix))
            if not op.isfile(evo_in):
                os.system(op.join(Path(__file__).parents[0],
                                  'processing/compute_grand_average.py'))
            # Peak ERF gradiometer activity
            grandavr = mne.read_evokeds(evo_in)[0]
            ch, lat = grandavr.get_peak(ch_type='grad', tmin=window[0],
                                        tmax=window[1])
            picks_dict[aix][cond] = ch
            evoked_dict[aix][cond] = grandavr
            if cond in ['all', 'deviant']:
                print('     %s peak at:\n'
                      '         %s at %0.3fms' % (cond, ch, lat))
            # plot group ERF topography
            timing = [lat - .1, lat]
            hs = grandavr.plot_joint(title=cond, times=timing,
                                     ts_args=config.TS_ARGS,
                                     topomap_args=config.TOPOMAP_ARGS)
            for ii, hh in enumerate(hs):
                hh.savefig(op.join(defaults.figsdir,
                                   op.basename(evo_in)[:-4] +
                                   '_%d.png' % ii), bbox_inches='tight')
        #  compare group ERF time courses at peak locations
        pick = [grandavr.ch_names.index(kk) for kk in
                set(list(picks_dict[aix].values()))]
        hs = mne.viz.plot_compare_evokeds(evoked_dict[aix])
        for hh, kind in zip(hs, ['grad', 'mag']):
            hh.savefig(op.join(defaults.figsdir,
                               '%s_%d_%s_%dmos_grp-ave.png' % (analysis, lp,
                                                               kind, aix)),
                       bbox_inches='tight')
