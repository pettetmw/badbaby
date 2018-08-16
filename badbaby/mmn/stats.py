# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
from functools import partial
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mne.stats import (ttest_1samp_no_p, permutation_cluster_1samp_test,
                       bonferroni_correction, fdr_correction)


def box_off(ax):
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    for axis in (ax.get_xaxis(), ax.get_yaxis()):
        for line in [ax.spines['left'], ax.spines['bottom']]:
            line.set_zorder(3)
        for line in axis.get_gridlines():
            line.set_zorder(1)
    ax.grid(True)


def zscore_data(a, samples):
    # Get baseline data
    bl = (np.where(times == min(times))[0][0], np.where(times == 0)[0][0])
    ma_ind = np.ma.array(a[:, ..., bl[0]:bl[1]])
    bl_std = np.std(ma_ind, axis=-1)  # std across time
    return np.divide(a, bl_std[:, ..., np.newaxis])  # z-scored


def one_sample_parametric(a, n, hat=0., method='relative'):
    ts = ttest_1samp_no_p(a, sigma=hat, method=method)
    ps = stats.distributions.t.sf(np.abs(ts), n - 1) * 2
    return ts, ps


def one_sample_nonparam(a, threshold, connectivity=None,
                        nperm=1024, stat_fun=None, buffer_size=None,
                        tail=0, n_jobs=18):
    results = permutation_cluster_1samp_test(a, n_jobs=n_jobs,
                                             threshold=threshold,
                                             connectivity=connectivity,
                                             n_permutations=nperm,
                                             stat_fun=stat_fun,
                                             buffer_size=buffer_size,
                                             tail=tail)
    return results


plt.style.use('seaborn-notebook')  # ggplot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['grid.color'] = '0.75'
plt.rcParams['grid.linestyle'] = ':'
leg_kwargs = dict(frameon=False, columnspacing=0.1, labelspacing=0.1,
                  fontsize=8, fancybox=False, handlelength=2.0,
                  loc='best')

project_dir = '/home/ktavabi/Projects/badbaby/static'
data_dir = '/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch'
fig_dir = op.join(data_dir, 'figures')

# Some parameters
groups = [2, 6]
remap = dict([(2, '2 months'), (6, '6 months')])
grp_nms = [remap[kind] for kind in [2, 6]]
analysis = 'Oddball-matched'
conditions = ['standard', 'deviant']
lpf = 30
colors = ['CornFlowerBlue', 'Crimson']
alpha = 0.05
sigma = 1e-3
threshold_tfce = dict(start=0, step=0.5)
n_permutations = 1024

# Read in data
fname = op.join(data_dir, '%s_data.npz' % analysis)
npz = np.load(fname)
times = npz['times']
peak_idx = npz['peak_idx']
# sel_data, mag_data, or grad_data
data = npz['sel_data']  # grps x conds x sel x subjs x chans x times


# RMS
if np.ndim(data) == 6:
    this_data = np.sqrt(np.mean(np.square(data), axis=4))
    n_grps, n_conds, n_sel = data.shape[:3]
    n_samples, n_tests = data.shape[3:6:2]
    sel = ['left', 'right']
elif np.ndim(data) == 5:
    this_data = np.sqrt(np.mean(np.square(data), axis=3))
    n_grps, n_conds = data.shape[:2]
    n_samples, n_tests = data.shape[3:5:1]
    n_sel = 1
    sel = ['Gradiometers']
this_data = zscore_data(this_data, times)

# |deviant| - |standard| for grps x sel x subjs x times
contrast = np.abs(this_data[:, 1]) - \
           np.abs(this_data[:, 0])
tvals = np.zeros((n_grps, n_sel, n_tests))
if n_sel == 1:
    tvals = np.squeeze(tvals)
pvals = np.zeros_like(tvals)
t_tfce_hat = np.zeros_like(tvals)
p_tfce_hat = np.zeros_like(tvals)
thresh_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
thresh_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)
# TODO compare against baseline?
# bl = (np.where(times == min(times))[0][0], np.where(times == 0)[0][0])
# ma_ind = np.ma.array(data[:, ..., bl[0]:bl[1]])
# a = ma_ind.reshape((2, 2, 2*22*30*60))
# bl_mean = np.mean(a, axis=2)

#############################################
# One-sample t-testing with "hat" adjustment#
#############################################
print('Parametric testing...')
for ii in np.arange(n_grps):
    print(' in %s group' % grp_nms[ii])
    if n_sel == 1:
        axis = 1
        tvals[ii], pvals[ii] = one_sample_parametric(contrast[ii], n_samples,
                                                     hat=sigma)
    else:
        axis = 2
        for jj in np.arange(n_sel):
            print('  %s hem' % sel[jj])
            out = one_sample_parametric(contrast[ii, jj], n_samples, hat=sigma)
            tvals[ii, jj], pvals[ii, jj] = out

reject_bonferroni, pval_bonferroni = bonferroni_correction(pvals, alpha=alpha)
reject_fdr, pval_fdr = fdr_correction(pvals, alpha=alpha, method='indep')

thresh_fdr = np.min(np.abs(np.ma.array(data=tvals, mask=~reject_fdr)),
                    axis=axis)

##############################################
# Permutation testing 1D temporal clustering #
##############################################
# TFCE with "hat" correction #
##############################
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
print('Permutation testing...')
for ii in np.arange(n_grps):
    print(' in %s group' % grp_nms[ii])
    if n_sel == 1:
        out = one_sample_nonparam(contrast[ii], threshold=threshold_tfce,
                                  nperm=n_permutations,
                                  stat_fun=stat_fun_hat,
                                  buffer_size=None, tail=1)
        t_tfce_hat[ii], _, p_tfce_hat[ii], H0 = out
    else:
        for jj in np.arange(n_sel):
            print('  %s' % sel[jj])
            out = one_sample_nonparam(contrast[ii, jj], threshold=threshold_tfce,
                                      nperm=n_permutations,
                                      stat_fun=stat_fun_hat,
                                      buffer_size=None, tail=1)
            t_tfce_hat[ii, jj], _, p_tfce_hat[ii, jj], H0 = out
sig_times = p_tfce_hat < alpha

# plot oddball stimuli
tsec = times * 1e3
for ii, grp in enumerate(grp_nms):
    hs = list()
    f, axs = plt.subplots(1, len(sel), sharex=True,
                          sharey=True, figsize=(10, 6))
    f.canvas.set_window_title(grp_nms[ii])
    for jj in np.arange(len(sel)):
        if len(sel) == 1:
            ax = axs
        axs[jj].set_title(sel[jj])
        for ci, cond in enumerate(conditions):
            if np.ndim(data) == 6:
                mean = this_data[ii, ci, jj].mean(axis=0)
                std = np.std(this_data[ii, ci, jj], axis=0)
            else:
                mean = this_data[ii, ci].mean(axis=0)
                std = np.std(this_data[ii, ci], axis=0)
            upper = mean + std
            lower = mean - std
            hs.append(axs[jj].plot(tsec, mean, label=conditions[ci],
                                   color=colors[ci]))
            hs.append(axs[jj].fill_between(tsec, upper, lower,
                                           alpha=0.2, color=colors[ci],
                                           linestyle='None', zorder=1))
            if jj == 0:
                axs[jj].set_xlabel('Time (ms)')
                axs[jj].set_ylabel('Strength')
            else:
                axs[jj].legend(numpoints=1, facecolor=None, **leg_kwargs)
            axs[jj].set_xlim(tsec.min(), tsec.max())
            axs[jj].vlines(0, axs[jj].get_ylim()[0],
                           axs[jj].get_ylim()[1], lw=.5)
            axs[jj].hlines(0, tsec[0], tsec[-1], lw=.5)
            box_off(axs[jj])
    f.tight_layout()
    f.savefig(op.join(fig_dir, '%s_%smos_%d_zscored.png'
                      % (analysis, groups[ii], lpf)), dpi=300, format='png')

# plot contrast
if np.ndim(data) == 6:
    contrast_mean = np.mean(contrast, axis=2)
    contrast_stdd = np.std(contrast, axis=2)
else:
    contrast_mean = np.mean(contrast, axis=1)
    contrast_stdd = np.std(contrast, axis=1)
upper = contrast_mean + contrast_stdd
lower = contrast_mean - contrast_stdd
for ii, grp in enumerate(grp_nms):
    hs = list()
    f, axs = plt.subplots(2, len(sel), sharex=True, figsize=(10, 6))
    f.canvas.set_window_title(grp_nms[ii])
    if len(axs) == 2:
        ax1, ax2 = axs.flatten()
        ax1.set_title(sel[-1])
        hs.append(ax1.plot(times, contrast_mean[ii], label='contrast'))
        hs.append(ax1.fill_between(times, upper[ii], lower[ii],
                                   alpha=0.3, linestyle='None', zorder=1,
                                   label='$\mathrm{\pm 1 STD}$'))
        ax1.set_ylabel('magnitude')
        ax1.legend(numpoints=1, facecolor=None, **leg_kwargs)
        ax1.set_xlim(times.min(), times.max())
        ax1.vlines(0, ax1.get_ylim()[0], ax1.get_ylim()[1], lw=.5)
        ax1.hlines(0, times[0], times[-1], lw=.5)
        box_off(ax1)
        hs.append(ax2.plot(times, tvals[ii].squeeze(),
                           label='$\mathrm{\hat t}$'))
        hs.append(ax2.plot(times, t_tfce_hat[ii].squeeze(),
                           label='$\mathrm{\hat t_{TFCE}}$'))
        mask = times[sig_times[ii].squeeze()]
        ax2.scatter(mask, np.ones(mask.shape) * ax2.get_ylim()[0],
                    s=2, marker='o', color='r')
        ax2.hlines(thresh_fdr[ii],
                   times[0], times[-1], linestyle='--',
                   colors='b', label='p=%s (FDR)' % alpha, linewidth=1)
        ax2.hlines(thresh_uncorrected,
                   times[0], times[-1], linestyle='--',
                   colors='k', label='p=%s (uncorrected)' % alpha,
                   linewidth=1)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('t-value')
        ax2.legend(numpoints=1, facecolor=None, **leg_kwargs)
        box_off(ax2)
        f.tight_layout()
        f.savefig(op.join(fig_dir, '%s_%smos_%d_zscored-stats.png'
                          % (analysis, groups[ii], lpf)), dpi=300, format='png')
    else:
        ax1, ax2, ax3, ax4 = axs.flatten()
        for ai, ax in enumerate([ax1, ax2]):
            ax.set_title(sel[ai])
            hs.append(ax.plot(times, contrast_mean[ii, ai], label='contrast'))
            hs.append(ax.fill_between(times, upper[ii, ai], lower[ii, ai],
                                      alpha=0.3, linestyle='None', zorder=1,
                                      label='$\mathrm{\pm 1 STD}$'))
            if ai == 0:
                ax.set_ylabel('magnitude')
            else:
                ax.legend(numpoints=1, facecolor=None, **leg_kwargs)
            ax.set_xlim(times.min(), times.max())
            ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], lw=.5)
            ax.hlines(0, times[0], times[-1], lw=.5)
            box_off(ax)
        for ai, ax in enumerate([ax3, ax4]):
            hs.append(ax.plot(times, tvals[ii, ai],
                              label='$\mathrm{\hat t}$'))
            hs.append(ax.plot(times, t_tfce_hat[ii, ai],
                              label='$\mathrm{\hat t_{TFCE}}$'))
            mask = times[sig_times[ii, ai]]
            ax.scatter(mask, np.ones(mask.shape) * ax.get_ylim()[0],
                       s=2, marker='o', color='r')
            ax.hlines(thresh_fdr[ii, ai],
                      times[0], times[-1], linestyle='--',
                      colors='b', label='p=%s (FDR)' % alpha, linewidth=1)
            ax.hlines(thresh_uncorrected,
                      times[0], times[-1], linestyle='--',
                      colors='k', label='p=%s (uncorrected)' % alpha,
                      linewidth=1)
            if ai == 0:
                ax.set_xlabel('time (s)')
                ax.set_ylabel('t-value')
            else:
                ax.legend(numpoints=1, facecolor=None, **leg_kwargs)
            box_off(ax)
        f.tight_layout()
        f.savefig(op.join(fig_dir, '%s_%smos_%d_zscored-stats.png'
                          % (analysis, groups[ii], lpf)), dpi=300,
                  format='png')
