# coding: utf-8

from os import path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mne import read_evokeds, grand_average

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['grid.color'] = '0.75'
plt.rcParams['grid.linestyle'] = ':'
leg_kwargs = dict(frameon=False, columnspacing=0.1, labelspacing=0.1,
                  fontsize=10, fancybox=True, handlelength=2.0, loc=0,)


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


project_dir = '/home/ktavabi/Projects/badbaby/static'
data_dir = '/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch'
fig_dir = op.join(data_dir, 'figures')
# Read excel sheets into pandas dataframes
xl_a = pd.read_excel(op.join(project_dir, 'badbaby.xlsx'), sheet_name='MMN',
                     converters={'BAD': str})
xl_b = pd.read_excel(op.join(project_dir, 'badbaby.xlsx'),
                     sheet_name='simms_demographics')
# Exclude subjects
inclusion = xl_a['simms_inclusion'] == 1
xl_a = xl_a[inclusion]
subjects = pd.Series(np.intersect1d(xl_a['Subject_ID'].values,
                                    xl_b['Subject_ID'].values))
# Find intersection between dataframes for common subjects
xl_a = xl_a[xl_a['Subject_ID'].isin(subjects.tolist())]
xl_b = xl_b[xl_b['Subject_ID'].isin(subjects.tolist())]
simms_df = pd.merge(xl_a, xl_b)

# Some parameters
groups = np.unique(simms_df.Group).tolist()
remap = dict([(2, '2 months'), (6, '6 months')])
grp_nms = [remap[kind] for kind in [2, 6]]
ts_args = {'gfp': True}
topomap_args = {'outlines': 'skirt', 'sensors': False}
analysis = 'Oddball'
conditions = ['standard', 'deviant']
colors = dict(deviant="Crimson", standard="CornFlowerBlue")
lpf = 30

# Loop over groups to compute & plot grand averages
for ii, group in enumerate(groups):
    print(' Loading data for %s...' % grp_nms[ii])
    subjects = simms_df[simms_df.Group == group].Subject_ID.values.tolist()
    n = len(subjects)
    for ci, cond in enumerate(conditions):
        print('     Analysis-%s / condition-%s' % (analysis, cond))
        file_out = op.join(data_dir, '%s_%s_%smos_%d_grd-ave.fif'
                           % (analysis, cond, groups[ii], lpf))
        if not op.isfile(file_out):
            print('      Doing averaging...')
            evokeds = list()
            for si, subj in enumerate(subjects):
                print('       %s' % subj)
                evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                                      '%s_%d-sss_eq_bad_%s-ave.fif'
                                      % (analysis, lpf, subj))
                evoked = read_evokeds(evoked_file, condition=cond,
                                      baseline=(None, 0))
                if evoked.info['sfreq'] > 600.0:
                    raise ValueError('bad_%s - %dHz Wrong sampling rate!'
                                     % (subj, evoked.info['sfreq']))
                if len(evoked.info['bads']) > 0:
                    evoked.interpolate_bads()
                evokeds.append(evoked.copy())
            # do grand averaging
            grandavr = grand_average(evokeds)
            grandavr.save(file_out)
        else:
            print('Reading...%s' % op.basename(file_out))
            grandavr = read_evokeds(file_out)[0]
        # peak ERF latency bn 100-550ms
        ch, lat = grandavr.get_peak(ch_type='mag', tmin=.1, tmax=.55)
        if cond in ['all', 'deviant']:
            print('     Peak latency for %s in %s mos group:\n'
                  '         %s at %0.3fms' % (cond, group, ch, lat))
        # plot ERF topography at peak latency and 100ms before
        timing = [lat - .1, lat]
        hs = grandavr.plot_joint(title=grp_nms[ii] + ' ' + cond,
                                 times=timing, ts_args=ts_args,
                                 topomap_args=topomap_args)
        for h, ch_type in zip(hs, ['grad', 'mag']):
            fig_out = op.join(fig_dir, '%s_%s_%smos_%d_%s_grd-ave.eps'
                              % (analysis, cond, group, lpf, ch_type))
            h.savefig(fig_out, dpi=240, format='eps')
