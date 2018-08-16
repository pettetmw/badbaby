# coding: utf-8

from os import path as op
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mne import read_evokeds, grand_average
from mne.stats import (ttest_1samp_no_p, permutation_cluster_1samp_test,
                       bonferroni_correction, fdr_correction)
from mne.viz import plot_compare_evokeds

plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['axes.labelsize'] = 'small'
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['grid.color'] = '0.75'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['lines.markersize'] = '0.5'
leg_kwargs = dict(frameon=False, columnspacing=0.1, labelspacing=0.1,
                  fontsize=8, fancybox=False, handlelength=2.0,
                  loc='upper right')


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
    """helper to return Pearson correlation coeffecient"""
    return stats.pearsonr(x, y)[0] ** 2


data_dir = '/home/ktavabi/Projects/badbaby/static'
fig_dir = '/home/ktavabi/Projects/badbaby/static/figures'
work_dir = '/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch'
df = pd.read_excel(op.join(data_dir, 'badbaby.xlsx'), sheet_name='MMN',
                   converters={'BAD': str})
inclusion = df['Included'] == 1
df = df[inclusion]
df_demographs = pd.read_excel(op.join(data_dir, 'simms_demographics.xlsx'),
                              sheet_name='simms_demographics')
simms_subjects = pd.Series(np.intersect1d(df['Subject_ID'].values,
                                          df_demographs['Subject_ID'].values))
df_a = df[df['Subject_ID'].isin(simms_subjects.tolist())]
df_b = df_demographs[df_demographs['Subject_ID'].isin(simms_subjects.tolist())]
simms_df = pd.merge(df_a, df_b)
groups = np.unique(simms_df.Group).tolist()
remap = dict([(2, '2 months'), (6, '6 months')])
titles_comb = [remap[kind] for kind in [2, 6]]
N = max([len(np.where(simms_df.Group == g)[0]) for g in groups])
ts_args = {'gfp': True}
topomap_args = {'outlines': 'skirt', 'sensors': False}

# Group ERFs
analysis = 'All'
conditions = ['All']
lpf = 30
peak_lats = list()
peak_chs = list()
for ii, group in enumerate(groups):
    subjects = simms_df[simms_df.Group == group].Subject_ID.values.tolist()
    n = len(subjects)
    evokeds = list()
    file_out = op.join(work_dir, '%s_%smos_%d_grd-ave.fif'
                       % (analysis, group, lpf))
    for si, sub in enumerate(subjects):
        evoked_file = op.join(work_dir, 'bad_%s' % sub, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif' % (analysis,
                                                               lpf,
                                                               sub))
        evoked = read_evokeds(evoked_file, condition=conditions[0],
                              baseline=(None, 0))
        if evoked.info['sfreq'] > 600.0:
            raise ValueError('bad_%s - %dHz Wrong sampling rate!'
                             % (sub, evoked.info['sfreq']))
        evokeds.append(evoked.copy())

    # do grand averaging
    if not op.isfile(file_out):
        grandavr = grand_average(evokeds)
        grandavr.save(file_out)
    else:
        grandavr = read_evokeds(file_out)[0]

    # peak ERF latency from gradiometer RMS
    ch, lat = grandavr.get_peak(ch_type='mag', tmin=.1, tmax=.5)
    peak_chs.append(ch)
    peak_lats.append(lat)
    print('Peak latency for %s mos at \n %s at %0.3fms'
          % (group, peak_chs[ii], peak_lats[ii]))
    # look at 100ms rising slope to peak
    timing = [peak_lats[ii] - .1, peak_lats[ii]]
    hs = grandavr.plot_joint(title=titles_comb[ii],
                             times=timing, ts_args=ts_args,
                             topomap_args=topomap_args)
    for h, chs in zip(hs, ['grad', 'mag']):
        fig_out = op.join(fig_dir, '%s_%smos_%d_%s_grd-ave.png'
                          % (analysis, group, lpf, chs))
        h.savefig(fig_out, dpi=300, format='png')


# Group ERFs for oddball conditions
analysis = 'Oddball'
conditions = ['standard', 'deviant']
colors = dict(deviant="Crimson", standard="CornFlowerBlue")
lpf = 30
peak_lats = list()
peak_chs = list()
for ii, group in enumerate(groups):
    subjects = simms_df[simms_df.Group == group].Subject_ID.values.tolist()
    n = len(subjects)
    for ci, cond in enumerate(conditions):
        evokeds = list()
        file_out = op.join(work_dir, '%s_%s_%smos_%d_grd-ave.fif'
                           % (analysis, cond, groups[ii], lpf))
        print('Reading...%s' % op.basename(file_out))
        for si, subj in enumerate(subjects):
            evoked_file = op.join(work_dir, 'bad_%s' % subj, 'inverse',
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
        if not op.isfile(file_out):
            grandavr = grand_average(evokeds)
            grandavr.save(file_out)
        else:
            grandavr = read_evokeds(file_out)[0]
        # peak deviant ERF latency from gradiometer RMS
        ch, lat = grandavr.get_peak(ch_type='mag', tmin=.1, tmax=.4)
        if cond == 'deviant':
            peak_lats.append(lat)
            peak_chs.append(ch)
        timing = [lat - .15, lat]
        hs = grandavr.plot_joint(title=titles_comb[ii] + ' ' + cond,
                                 times=timing, ts_args=ts_args,
                                 topomap_args=topomap_args)
        for h, chs in zip(hs, ['grad', 'mag']):
            fig_out = op.join(fig_dir, '%s_%s_%smos_%d_%s_grd-ave.png'
                              % (analysis, cond, group, lpf, chs))
            h.savefig(fig_out, dpi=300, format='png')

# Lets see how sensor ts look across Groups
sensors = {'lh': ['MEG0111', 'MEG0112', 'MEG0113',
                  'MEG0121', 'MEG0122', 'MEG0123',
                  'MEG0341', 'MEG0342', 'MEG0343',
                  'MEG0321', 'MEG0322', 'MEG0323',
                  'MEG0331', 'MEG0332', 'MEG0333',
                  'MEG0131', 'MEG0132', 'MEG0133',
                  'MEG0211', 'MEG0212', 'MEG0213',
                  'MEG0221', 'MEG0222', 'MEG0223',
                  'MEG0411', 'MEG0412', 'MEG0413',
                  'MEG0421', 'MEG0422', 'MEG0423',
                  'MEG0141', 'MEG0142', 'MEG0143',
                  'MEG1511', 'MEG1512', 'MEG1513',
                  'MEG0241', 'MEG0242', 'MEG0243',
                  'MEG0231', 'MEG0232', 'MEG0233',
                  'MEG0441', 'MEG0442', 'MEG0443',
                  'MEG0431', 'MEG0432', 'MEG0433',
                  'MEG1541', 'MEG1542', 'MEG1543',
                  'MEG1521', 'MEG1522', 'MEG1523',
                  'MEG1611', 'MEG1612', 'MEG1613',
                  'MEG1621', 'MEG1622', 'MEG1623',
                  'MEG1811', 'MEG1812', 'MEG1813',
                  'MEG1821', 'MEG1822', 'MEG1823',
                  'MEG1531', 'MEG1532', 'MEG1533',
                  'MEG1721', 'MEG1722', 'MEG1723',
                  'MEG1641', 'MEG1642', 'MEG1643',
                  'MEG1631', 'MEG1632', 'MEG1633',
                  'MEG1841', 'MEG1842', 'MEG1843'],
           'rh': ['MEG1421', 'MEG1422', 'MEG1423',
                  'MEG1411', 'MEG1412', 'MEG1413',
                  'MEG1221', 'MEG1222', 'MEG1223',
                  'MEG1231', 'MEG1232', 'MEG1233',
                  'MEG1241', 'MEG1242', 'MEG1243',
                  'MEG1441', 'MEG1442', 'MEG1443',
                  'MEG1321', 'MEG1322', 'MEG1323',
                  'MEG1311', 'MEG1312', 'MEG1313',
                  'MEG1121', 'MEG1122', 'MEG1123',
                  'MEG1111', 'MEG1112', 'MEG1113',
                  'MEG1431', 'MEG1432', 'MEG1433',
                  'MEG2611', 'MEG2612', 'MEG2613',
                  'MEG1331', 'MEG1332', 'MEG1333',
                  'MEG1341', 'MEG1342', 'MEG1343',
                  'MEG1131', 'MEG1132', 'MEG1133',
                  'MEG1141', 'MEG1142', 'MEG1143',
                  'MEG2621', 'MEG2622', 'MEG2623',
                  'MEG2641', 'MEG2642', 'MEG2643',
                  'MEG2421', 'MEG2422', 'MEG2423',
                  'MEG2411', 'MEG2412', 'MEG2413',
                  'MEG2221', 'MEG2222', 'MEG2223',
                  'MEG2211', 'MEG2212', 'MEG2213',
                  'MEG2631', 'MEG2632', 'MEG2633',
                  'MEG2521', 'MEG2522', 'MEG2523',
                  'MEG2431', 'MEG2432', 'MEG2433',
                  'MEG2441', 'MEG2442', 'MEG2443',
                  'MEG2231', 'MEG2232', 'MEG2233']}
assert len(sensors['lh']) == len(sensors['rh'])
##############
# Statistics #
##############
analysis = 'Oddball'
conditions = ['standard', 'deviant']
colors = dict(deviant="Crimson", standard="CornFlowerBlue")
lpf = 30
alpha = 0.001
sigma = 1e-6
threshold_tfce = dict(start=0, step=0.5)
n_permutations = 1024
for gi, group in enumerate(groups):
    print(' Loading data for %s...' % titles_comb[gi])
    subjects = simms_df[simms_df.Group == group].Subject_ID.values.tolist()
    evoked_dict = dict()
    n = len(subjects)
    X = list()
    for ci, cond in enumerate(conditions):
        print('     \n%s...' % cond)
        evokeds = list()
        for si, subj in enumerate(subjects):
            print('      %s' % subj)
            evoked_file = op.join(work_dir, 'bad_%s' % subj, 'inverse',
                                  '%s_%d-sss_eq_bad_%s-ave.fif'
                                  % (analysis, lpf, subj))
            evoked = read_evokeds(evoked_file, condition=cond,
                                  baseline=(None, 0))
            if len(evoked.info['bads']) > 0:
                print('       Interpolating bad channels...')
                evoked.interpolate_bads()
            times = evoked.times
            ch_names = evoked.ch_names
            mags = np.asarray(evoked.copy().pick_types(meg='mag').ch_names)
            assert len(mags) == 102
            for hi, hem in enumerate(sensors.keys()):
                these_chs = np.intersect1d(mags, sensors[hem]).tolist()
                assert len(these_chs) == 27
                evo_cp = evoked.copy().pick_channels(these_chs)
                evo_cp.info.normalize_proj()  # likely not necessary
                if si == 0 and hi == 0:
                    data = np.zeros((n, len(sensors), evo_cp.data.shape[1]))
                # magnetometer RMS
                data[si, hi] = np.sqrt(np.mean(evo_cp.data, axis=0) ** 2)
            evokeds.append(evoked)
        evoked_dict[cond] = evokeds
        X.append(data)
    # |deviant - standard| per hem
    contrast = np.abs(X[1] - X[0])
    contrast = np.transpose(contrast, (1, 0, 2))
    #############################################
    # One-sample t-testing with "hat" adjustment#
    #############################################
    fig, axs = plt.subplots(1, len(sensors), sharex=True, sharey=True)
    for ii, key in enumerate(sensors.keys()):
        print('     Testing...\n'
              '      |deviant - standard| in %s hem...' % key)
        ts = ttest_1samp_no_p(contrast[ii], sigma=sigma)
        ps_unc = stats.distributions.t.sf(np.abs(ts), n - 1) * 2
        n_samples, n_tests = contrast[ii].shape
        threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
        reject_bonferroni, pval_bonferroni = bonferroni_correction(ps_unc,
                                                                   alpha=alpha)
        threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)

        reject_fdr, pval_fdr = fdr_correction(ps_unc, alpha=alpha,
                                              method='indep')
        threshold_fdr = np.min(np.abs(ts)[reject_fdr])
        axs[ii].plot(times, ts, label='$\mathrm{|deviant - standard|_{hat}}$',
                     color='g')
        xmin, xmax = axs[ii].get_xlim()
        axs[ii].hlines(threshold_uncorrected, xmin, xmax, linestyle='--',
                       colors='k',
                       label='p=%s (uncorrected)' % alpha, linewidth=2)
        axs[ii].hlines(threshold_bonferroni, xmin, xmax, linestyle='--',
                       colors='r',
                       label='p=%s (Bonferroni)' % alpha, linewidth=2)
        axs[ii].hlines(threshold_fdr, xmin, xmax, linestyle='--', colors='b',
                       label='p=%s (FDR)' % alpha, linewidth=2)
        axs[ii].set(title=key)
        if ii == 0:
            axs[ii].set(xlabel='Time (ms)', ylabel='t-value')
        else:
            axs[ii].legend(numpoints=1, facecolor=None, **leg_kwargs)
        box_off(axs[ii])
        axs[ii].axvline(c='k', linewidth=0.5, zorder=0)
        axs[ii].get_xticklabels()[0].set(horizontalalignment='right')
        fig.canvas.set_window_title(titles_comb[gi])
        fig.tight_layout()
    fig.savefig(op.join(fig_dir, '%s_%smos_%d_T-test.png'
                        % (analysis, group, lpf)),
                dpi=300, format='png')
    ##############################################
    # Permutation testing 1D temporal clustering #
    ##############################################
    # TFCE with "hat" correction #
    ##############################
    fig, axs = plt.subplots(1, len(sensors), sharex=True, sharey=True)
    for ii, key in enumerate(sensors.keys()):
        print('     Permutation testing...\n'
              '      |deviant - standard| in %s hem...' % key)
        stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
        t_tfce_hat, _, p_tfce_hat, H0 = permutation_cluster_1samp_test(
            contrast[ii], n_jobs=18, threshold=threshold_tfce,
            connectivity=None, n_permutations=n_permutations,
            stat_fun=stat_fun_hat, buffer_size=None, tail=1)
        sig_times = times[np.where(p_tfce_hat < alpha)[0]]
        sig_times = sig_times[np.logical_and(sig_times > .2, sig_times < .55)]
        axs[ii].plot(times, t_tfce_hat, color='k',
                     label='$\mathrm{|deviant - standard|_{hat,TFCE}}$')
        ymin, ymax = axs[ii].get_ylim()
        axs[ii].scatter(sig_times, np.ones(sig_times.shape) * ymin,
                        s=2, marker='o', color='r')
        axs[ii].set(title=key)
        if ii == 0:
            axs[ii].set(xlabel='Time (ms)',
                        ylabel='t-value')
        else:
            axs[ii].legend(numpoints=1, facecolor=None, **leg_kwargs)
        box_off(axs[ii])
        axs[ii].axvline(c='k', linewidth=0.5, zorder=0)
        axs[ii].get_xticklabels()[0].set(horizontalalignment='right')
        fig.canvas.set_window_title(titles_comb[gi])
        fig.tight_layout()
    fig.savefig(op.join(fig_dir, '%s_%smos_%d_TFCE.png'
                        % (analysis, group, lpf)),
                dpi=300, format='png')

    for hi, hem in enumerate(sensors.keys()):
        print('     Plotting %s data...' % hem)
        picks = [ch_names.index(jj) for jj in sensors[hem]]
        hs = plot_compare_evokeds(evokeds=evoked_dict, picks=picks,
                                  colors=colors, ci=.95,
                                  title=titles_comb[
                                            gi] + ' Oddball Stimuli')
        for h, chs in zip(hs, ['mag', 'grads']):
            h.savefig(op.join(fig_dir, '%s_%smos_%d_%s-%s.png'
                              % (analysis, group, lpf, hem, chs)),
                      dpi=300, format='png')


