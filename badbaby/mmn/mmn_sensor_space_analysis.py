# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@uw.edu>

import os
from os import path as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import seaborn as sns
from scipy import stats

from mne import (read_evokeds, grand_average, combine_evoked)
from mne.stats import permutation_t_test
from mne.utils import _time_mask
from mne.evoked import _get_peak


def box_off(ax):
    """helper to format axis tick and border"""
    # Ensure axis ticks only show up on the bottom and left of the plot.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Remove the plot frame lines.
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    for axis in (ax.get_xaxis(), ax.get_yaxis()):
        for line in [ax.spines['left'], ax.spines['bottom']]:
            line.set_zorder(3)
        for line in axis.get_gridlines():
            line.set_zorder(1)
    ax.grid(True)


def r2(x, y):
    """helper to yield Pearson correlation coeffecient"""
    return stats.pearsonr(x, y)[0] ** 2


leg_kwargs = dict(frameon=True, columnspacing=0.1, labelspacing=0.1,
                  fontsize=8, fancybox=True, handlelength=2.0,
                  loc='upper left')

studydir = '/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch/inithp'
df = pd.read_csv('/media/ktavabi/ALAYA/data/ilabs/badbaby/simms_demographics.csv')
lpf = 40
fig_dir = op.join(studydir, 'figures')

df = df[df.MMN == 1]  # Pandas df slicing
groups = np.unique(df.Group).tolist()
remap = dict([(2, 'two'), (6, 'six')])
titles_comb = [remap[kind] for kind in [2, 6]]
N = max([len(np.where(df.Group == g)[0]) for g in groups])
ts_args = {'gfp': True}
topomap_args = {'outlines': 'skirt', 'sensors': False}

if not op.isdir(fig_dir):
    os.mkdir(fig_dir)

# Group ERFs
analysis = 'All'
conditions = ['All']
peak_lats = list()
for ii, group in enumerate(groups):
    subjects = df[df.Group == group].Subject_ID.values.tolist()
    n = len(subjects)
    assert n == len(np.where(df.Group == group)[0])
    evokeds = list()
    file_out = op.join(studydir, '%s_%s_%d_n%d_grand-ave.fif'
                       % (titles_comb[ii], analysis, lpf, n))
    for si, sub in enumerate(subjects):
        evoked_file = op.join(studydir, 'bad_%s' % sub, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif' % (analysis,
                                                               lpf,
                                                               sub))
        evoked = read_evokeds(evoked_file, condition=conditions[0],
                              baseline=(None, 0))
        times = evoked.times
        info = evoked.info
        if (evoked.info['sfreq']) != 600.0:
            print('bad_%s - %d' % (sub, evoked.info['sfreq']))
        evokeds.append(evoked.copy())
        if group == groups[0] and sub == subjects[0]:
            erf_data = np.zeros((len(groups), N,
                                 evoked.data.shape[0],
                                 evoked.data.shape[1]))
        erf_data[ii, si] = evoked.data
    del subjects
    if not op.isfile(file_out):
        grand_avr = grand_average(evokeds)
        grand_avr.save(file_out)
    else:
        grand_avr = read_evokeds(file_out)[0]
    assert grand_avr.nave == n
    del evokeds
    for ch in ['grad', 'mag']:
        if ch == 'grad':
            peak_lats.append(grand_avr.get_peak(ch_type=ch, merge_grads=True))
        else:
            peak_lats.append(grand_avr.get_peak(ch_type=ch))
    hs = grand_avr.plot_joint(title=titles_comb[ii],
                            times='peaks', ts_args=ts_args,
                            topomap_args=topomap_args)
    del grand_avr
    for h, ch_type in zip(hs, ['grad', 'mag']):
        fig_out = op.join(fig_dir, '%s_%s_%d_n%d_%s_grand-ave.fif'
                       % (titles_comb[ii], analysis, lpf, n, ch_type))
        h.savefig(fig_out, dpi=100, format='png')
plt.close('all')

# From group ERF peak detection above
these_grads = [ii[0] for ii in peak_lats if 'X' in ii[0]]
pairs = list()
for grad in these_grads:
    pairs.append([grad.replace('X', '%s' % i) for i in ['2', '3']])
assert all(len(pi) == 2 for pi in pairs)
timing = [ti[1] for ti in peak_lats if 'X' in ti[0]]
assert len(pairs) == len(timing) == len(groups)

analysis = 'Oddball'
conditions = ['standard', 'deviant']
for ii, group in enumerate(groups):
    subjects = df[df.Group == group].Subject_ID.values.tolist()
    n = len(subjects)
    assert n == len(np.where(df.Group == group)[0])
    deviants = list()
    standards = list()
    f_out = [op.join(studydir, '%s_%s_%d_n%d_grand-ave.fif'
                    % (titles_comb[ii], c, lpf, n))
            for c in conditions]
    for si, subj in enumerate(subjects):
        evoked_file = op.join(studydir, 'bad_%s' % subj, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif'
                              % (analysis, lpf, subj))
        deviant = read_evokeds(evoked_file, condition='deviant',
                               baseline=(None, 0))
        sfreq = deviant.info['sfreq']
        deviants.append(deviant)
        standard = read_evokeds(evoked_file, condition='standard',
                                baseline=(None, 0))
        standards.append(standard)
        assert sfreq == standard.info['sfreq']
        if group == groups[0] and subj == subjects[0]:
            amp_data = np.zeros((len(groups), N))
            lat_data = np.zeros((len(groups), N))
            standard_data = np.zeros((len(groups), N))
        # area under curve
        tmin, tmax = (timing[ii] - .05, timing[ii])
        mask = _time_mask(deviant.times, tmin=tmin,
                          tmax=tmax, sfreq=sfreq)
        dummy = deviant.copy().pick_channels(pairs[ii])
        amp_data[ii, si] = np.sum(np.abs(dummy.data[:, mask])) * \
                           len(dummy.data) * (1. / sfreq)
        lat_data[ii, si] = times[_get_peak(dummy.data, times,
                                           tmin=tmin,
                                           tmax=tmax)[1]]
        print("     Peak latency for %s at %.3f sec \n \n" % (subj,
                                                              lat_data[ii, si]))
        del dummy
        # for the standard stimulus
        dummy = standard.copy().pick_channels(pairs[ii])
        standard_data[ii, si] = np.sum(np.abs(deviant.data[:, mask])) * \
                                len(deviant.data) * (1. / deviant.info['sfreq'])
        del dummy
    for cond, evs, f in zip(conditions, [standards, deviants], f_out):
        if not op.isfile(f):
            grand_avr = grand_average(evs)
            grand_avr.save(op.join(studydir, '%s_%s_%d_n%d_grand-ave.fif'
                    % (titles_comb[ii], cond, lpf, n)))
        else:
            grand_avr = read_evokeds(file_out)[0]
        assert grand_avr.nave == n
        hs = grand_avr.plot_joint(title=titles_comb[ii],
                                  times='peaks', ts_args=ts_args,
                                  topomap_args=topomap_args)
        for h, ch_type in zip(hs, ['grad', 'mag']):
            fig_out = op.join(fig_dir, '%s_%s_%d_n%d_%s_grand-ave.png'
                              % (titles_comb[ii], c, lpf,
                                 n, ch_type))
            h.savefig(fig_out, dpi=300, format='png')
    del grand_avr
plt.close('all')

colors = '#adc3f4', '#DC143C'
conditions = ['standard', 'deviant']
legend = dict(zip(conditions, colors))
for ii, group in enumerate(groups):
    subjects = df[df.Group == group].Subject_ID.values.tolist()
    n = len(subjects)
    assert n == len(np.where(df.Group == group)[0])
    evoked_dict = dict()
    fnames = [op.join(studydir, '%s_%s_%d_n%d_grand-ave.fif'
                      % (titles_comb[ii], c, lpf, n))
              for c in conditions]
    evs = [read_evokeds(f)[0]
           for f in fnames]
    for jj, condition in enumerate(conditions):
        evoked_dict[condition] = evs[jj]
    title = titles_comb[ii] + ' - ' + \
            '{ }'.format(kk for kk in pairs) + ' - GFP'
    fig = mne.viz.plot_compare_evokeds(evoked_dict, colors=legend,
                                       )
    fig.gca().legend(**leg_kwargs)
    fig.gca().set_title(title)
    box_off(fig.gca())
    fig.tight_layout()
    fig.savefig(op.join(fig_dir, '%s_%s_%d_n%d_grand-ave_gfp.eps'
                        % (titles_comb[ii], c, lpf, n)))

# Enter data into panads frame
latencies = pd.concat([pd.Series(lat_data[ii, :len(df[df.Group == g])],
                                 index=df[df.Group == g].index)
                       for ii, g in enumerate(groups)])
df['dev_lats'] = latencies * 1e3
amplitudes = pd.concat([pd.Series(amp_data[ii, :len(df[df.Group == g])],
                                  index=df[df.Group == g].index)
                        for ii, g in enumerate(groups)])
df['dev_amps'] = amplitudes
the_standard = pd.concat([pd.Series(standard_data[ii, :len(df[df.Group == g])],
                                   index=df[df.Group == g].index)
                          for ii, g in enumerate(groups)])
df['standard_amps'] = the_standard


jpkwargs = {'x_ci': .95, 'truncate':True}
covar = 'Age(days)'
for ii, group in enumerate(groups):
    for measure, nm in zip(['dev_lats', 'dev_amps'], ['Latency', 'Strength']):
        g = sns.JointGrid(x=measure, y=covar,
                          data=df[df.Group == group])
    g.plot_joint(sns.regplot, **jpkwargs)
    g.annotate(stats.pearsonr)
    g.ax_marg_x.set_axis_off()
    g.ax_marg_y.set_axis_off()
    sns.despine(offset=10, trim=True)
    plt.title(titles_comb[ii] + ' - ' + nm)

