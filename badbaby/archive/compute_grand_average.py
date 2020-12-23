#!/usr/bin/env python

"""compute_grand_average.py: Compute grand average ERFs, write data to disk."""

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
    for aii, aix in enumerate(ages):
        df = rd.return_dataframes('mmn')[0]
        if aii == 0:  # 2 mos cohort
            subjects = ['bad_%s' % ss for ss in df[df['age'] < 81].index]
        else:
            subjects = ['bad_%s' % ss for ss in df[df['age'] > 81].index]
        for ci, cond in enumerate(conditions):
            evo_out = op.join(workdir, '%s-%s_%d_%dmos_grp-ave.fif' % (analysis,
                                                                       cond, lp,
                                                                       aix))
            npz_out = op.join(workdir,
                              '%s-%s_%d_%dmos-ave.npz' % (analysis, cond,
                                                          aix, lp))
            h5_out = op.join(workdir, '%s-%s_%d_%dmos-ave.h5' % (analysis, cond,
                                                                 aix, lp))
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
                        erf_data = np.zeros(
                            (len(subjects), len(evoked.info['chs']),
                             len(evoked.times)))
                    erf_data[si] = evoked.data
                np.savez(npz_out, erf_data=erf_data)
                # do grand averaging
                print('      Doing averaging...')
                grandavr = grand_average(evokeds)
                grandavr.save(evo_out)
