# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import os
from os import path as op
import numpy as np
import seaborn as sns
from mne import read_evokeds, grand_average
from meegproc import defaults, utils
import badbaby.defaults as params
import badbaby.return_dataframes as rd


def read_in_evoked(filename, condition):
    """helper to read evoked file"""
    erf = read_evokeds(filename, condition=condition,
                       baseline=(None, 0))
    if erf.info['sfreq'] > 600.0:
        raise ValueError('bad_%s - %dHz Wrong sampling rate!'
                         % (subj, erf.info['sfreq']))
    chs = np.asarray(erf.info['ch_names'])
    assert (all(chs == np.asarray(defaults.vv_all_ch_order)))
    if len(erf.info['bads']) > 0:
        erf.interpolate_bads()
    return erf, chs


# Some parameters
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
age = 2
data_dir = params.meg_dirs['mmn']
fig_dir = op.join(data_dir, 'figures')
if not op.isdir(fig_dir):
    os.mkdir(fig_dir)
meg_df, cdi_df = rd.return_dataframes('mmn', age=age)
# Remove rows with 0 entry for CDI measures.
for nm in ['M3L', 'VOCAB']:
    cdi_df = cdi_df[cdi_df[nm] != 0]
#  Confirm data is correct
print('\nDescriptive stats for Age(days) variable...\n',
      meg_df['Age(days)'].describe())
# CDI measures regplots
for nm, title in zip(['M3L', 'VOCAB'],
                     ['Mean length of utterance', 'Words understood']):
    g = sns.lmplot(x="CDIAge", y=nm, truncate=True, data=cdi_df)
    g.set_axis_labels("Age (months)", nm)
    g.ax.set(title=title)
    g.ax.grid(True)
    g.despine(offset=5, trim=True)

# Loop over groups & plot grand average ERFs
print('Plotting Grand Averages')
for ci, cond in enumerate(conditions):
    print('     Loading data for %s / %s' % (analysis, cond))
    file_out = op.join(data_dir, '%s_%s_%s-mos_%d_grand-ave.fif'
                       % (analysis, cond, age, lpf))
    if not op.isfile(file_out):
        print('      Doing averaging...')
        evokeds = list()
        for si, subj in enumerate(meg_df.Subject_ID.values):
            print('       %s' % subj)
            evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                                  '%s_%d-sss_eq_bad_%s-ave.fif'
                                  % (analysis, lpf, subj))
            evoked, ch_names = read_in_evoked(evoked_file, condition=cond)
            evokeds.append(evoked)
        # do grand averaging
        grandavr = grand_average(evokeds)
        grandavr.save(file_out)
    else:
        print('Reading...%s' % op.basename(file_out))
        grandavr = read_evokeds(file_out)[0]
    # peak ERF latency bn 100-550ms
    ch, lat = grandavr.get_peak(ch_type='mag', tmin=.15, tmax=.55)
    if cond in ['all', 'deviant']:
        print('     Peak latency for %s at:\n'
              '         %s at %0.3fms' % (cond, ch, lat))
    # plot ERF topography at peak latency and 100ms before
    timing = [lat - .1, lat]
    hs = grandavr.plot_joint(title=cond, times=timing,
                             ts_args=params.ts_args,
                             topomap_args=params.topomap_args)
    # for h, ch_type in zip(hs, ['grad', 'mag']):
    #     fig_out = op.join(fig_dir, '%s_%s_%s_%d_%s_grd-ave.eps'
    #                       % (analysis, cond, nm.replace(' ', ''),
    #                          lpf, ch_type))
    #     h.savefig(fig_out, dpi=240, format='eps')

