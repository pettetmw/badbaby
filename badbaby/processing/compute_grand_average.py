#!/usr/bin/env python

"""Compute grand averaged evoked data.
    Per age x condition:
        1. Compute grand average ERF
        2. Write evoked data arrays to disk
        3. Write grand average evoked file to disk
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Production"

from os import path as op

import matplotlib.pyplot as plt
import numpy as np
from mne import (
    read_evokeds, grand_average
    )

from badbaby import return_dataframes as rd, defaults


def read_in_evoked(filename, condition):
    """helper to read evoked file"""
    erf = read_evokeds(filename, condition=condition,
                       baseline=(None, 0))
    if erf.info['sfreq'] > 600.0:
        raise ValueError('bad_%s - %dHz Wrong sampling rate!'
                         % (subj, erf.info['sfreq']))
    # chs = np.asarray(erf.info['ch_names'])
    # assert (all(chs == np.asarray(config.VV_ALL.values())))
    if len(erf.info['bads']) > 0:
        erf.interpolate_bads()
    return erf


plt.style.use('ggplot')

# parameters
workdir = defaults.datapath
analysis = 'Oddball'
conditions = ['standard', 'deviant']
tmin, tmax = defaults.epoching
lp = defaults.lowpass
ages = [2, 6]
window = defaults.peak_window  # peak ERF latency window

for aix in ages:
    rstate = np.random.RandomState(42)
    df = rd.return_dataframes('mmn', age=aix)[0]
    subjects = ['bad_%s' % ss for ss in df.index]
    print(df.info())
    for ci, cond in enumerate(conditions):
        evo_out = op.join(workdir, '%s_%d_%dmos_grp-ave.fif' % (cond, lp, aix))
        npz_out = op.join(workdir, '%s_%d_%dmos-ave.npz' % (cond, aix, lp))
        h5_out = op.join(workdir, '%s_%d_%dmos-ave.h5' % (cond, aix, lp))
        if not any([op.isfile(evo_out), op.isfile(npz_out)]):
            evokeds = list()
            print('     Loading %s - %s data' % (analysis, cond))
            for si, subj in enumerate(subjects):
                print('       %s' % subj)
                evoked_file = op.join(workdir, subj, 'inverse',
                                      '%s_%d-sss_eq_%s-ave.fif'
                                      % (analysis, lp, subj))
                evoked = read_in_evoked(evoked_file, condition=cond)
                evokeds.append(evoked)
                if subj == subjects[0]:
                    erf_data = np.zeros((len(subjects), len(evoked.info['chs']),
                                         len(evoked.times)))
                erf_data[si] = evoked.data
            np.savez(npz_out, erf_data=erf_data)
            # do grand averaging
            print('      Doing averaging...')
            grandavr = grand_average(evokeds)
            grandavr.save(evo_out)
