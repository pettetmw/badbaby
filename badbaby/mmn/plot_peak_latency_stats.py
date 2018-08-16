# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
from functools import partial
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
import seaborn as sns
import pandas as pd
from mne.stats import (ttest_1samp_no_p, permutation_cluster_1samp_test,
                       bonferroni_correction, fdr_correction)
from mne.evoked import _get_peak


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


plt.style.use('ggplot')  # seaborn-notebook
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
                  fontsize=10, fancybox=False, handlelength=2.0,
                  loc='best')
sns.set(style="whitegrid", palette="pastel", color_codes=True)

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
groups = np.unique(simms_df.Group).tolist()
remap = dict([(2, '2 months'), (6, '6 months')])
titles_comb = [remap[kind] for kind in [2, 6]]
subjects = np.unique(np.asarray([x[:3] for x in simms_df.Subject_ID.values]))
subjects = ['BAD_%s' % subj for subj in subjects]

dfs = list()
xl_c = pd.read_excel(op.join(project_dir, 'cdi_report_July_2018.xlsx'),
                     sheet_name='WS')
for age in np.unique(xl_c.CDIAge.values):
    _, _, mask = np.intersect1d(np.asarray(subjects),
                                xl_c[xl_c.CDIAge == age]
                                ['ParticipantId'].values,
                                return_indices=True)
    dfs.append(xl_c[xl_c.CDIAge == age].iloc[mask])
cdi_df = pd.concat(dfs, ignore_index=True)

# Some parameters
groups = [2, 6]
remap = dict([(2, '2 months'), (6, '6 months')])
grp_nms = [remap[kind] for kind in [2, 6]]
analysis = 'Individual'
conditions = ['standard', 'Ba', 'Wa']
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
sfreq = npz['sfreq']
peak_idx = npz['peak_idx']
# sel_data, mag_data, or grad_data
data = npz['sel_data']  # grp x cond x sel x subj x chan x time
if np.ndim(data) == 6:
    this_data = np.sqrt(np.mean(np.square(data), axis=4))
    n_grps, n_conds, n_sel = data.shape[:3]
    n_samples, n_tests = data.shape[3:6:2]
    chs_dim = 4
    selection = ['left', 'right']
elif np.ndim(data) == 5:
    this_data = np.sqrt(np.mean(np.square(data), axis=3))
    n_grps, n_conds = data.shape[:2]
    n_samples, n_tests = data.shape[3:5:1]
    chs_dim = 3
    selection = ['Gradiometers']
this_data = zscore_data(this_data, times)

#  Get from evokeds dependent measures (peak latency, AUC) as ndarrays
for ii, _ in enumerate(groups):
    for ci, _ in enumerate(conditions):
        if n_sel > 1:
            for jj in np.arange(n_sel):
                for kk in np.arange(n_samples):
                    if ii == 0 and ci == 0 and jj == 0 and kk == 0:
                        #  grp x cond x sel x subj
                        latencies = np.zeros(data.shape[:chs_dim])
                        locs = np.zeros_like(latencies)
                        amps = np.zeros_like(latencies)  # AUC -100 ms to peak
                    out = _get_peak(data[ii, ci, jj, kk], times,
                                    tmin=.15, tmax=.55)
                    locs[ii, ci, jj, kk] = out[0]
                    latencies[ii, ci, jj, kk] = out[1] # milliseconds
                    amps[ii, ci, jj, kk] = out[2]
                    b = np.where(np.isclose(times,
                                            out[1] * 1e-3, atol=1e-3))[0][0]
                    a = np.where(np.isclose(times,
                                            (out[1] - 100) * 1e-3,
                                            atol=1e-3))[0][0]
                    auc = (np.sum(np.abs(data[ii, ci, jj, kk, :, a : b]))
                           * (len(times) / sfreq))
                    amps[ii, ci, jj, kk] = auc

# plot dependents
box_kwargs = {'showmeans': True, 'meanline': False}
for nm, unit, measure in zip(['Latency', 'Amplitude'],
                             ['Time (ms)', 'Strength (fT)'],
                             [latencies, amps]):
    for ii, _ in enumerate(groups):
        f, axs = plt.subplots(1, n_sel, figsize=(10, 5), sharey=True)
        f.canvas.set_window_title(grp_nms[ii])
        for kk in np.arange(len(selection)):
            plot_data = [measure[ii, ci, kk] for ci in np.arange(n_conds)]
            axs[kk].boxplot(plot_data, widths=.5,
                            labels=conditions, **box_kwargs)
            axs[kk].set(title=selection[kk])
            axs[kk].set(ylabel=unit)
            box_off(axs[kk])
        f.tight_layout()
        f_out = '%s_%s-mos_%s.eps' % (analysis, groups[ii], nm)
        f.savefig(op.join(fig_dir, f_out.replace(' ', '_')),
                  dpi=240, format='eps')

# Paired t-tests Age 2 vs 6
thresh_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
thresh_bonferroni = stats.t.ppf(1.0 - alpha / n_conds * n_sel, n_samples - 1)

# erf strength: AUC
amp_ts, amp_ps = stats.ttest_rel(amps[0], amps[1], axis=2)
# erf peak latencies
lat_ts, lat_ps = stats.ttest_rel(latencies[0], latencies[1], axis=2)

# Non-parametric Wilcoxon signed rank
amp_ts_wilcox = np.zeros(amps.shape[1:3])
amp_ps_wilcox = np.zeros_like(amp_ts_wilcox)
for ci, _ in enumerate(conditions):
    for jj in np.arange(n_sel):
        amp_ts_wilcox[ci, jj], amp_ps_wilcox[ci, jj] = \
            stats.wilcoxon(amps[0, ci, jj], amps[1, ci, jj])

lat_ts_wilcox = np.zeros(amps.shape[1:3])
lat_ps_wilcox = np.zeros_like(amp_ts_wilcox)
for ci, _ in enumerate(conditions):
    for jj in np.arange(n_sel):
        lat_ts_wilcox[ci, jj], lat_ps_wilcox[ci, jj] = \
            stats.wilcoxon(latencies[0, ci, jj], latencies[1, ci, jj])

# Plot CDI measures
ws_measures = ['M3L', 'VOCAB', 'HOWUSE',
               'WORDEND', 'IRWORDS', 'OGWORDS',
               'COMBINE', 'COMPLEX']
cdi_ages = np.arange(18, 31, 3)
for nm in ws_measures:
    g = sns.lmplot(x="CDIAge", y=nm, truncate=True, data=cdi_df)
    g.set_axis_labels("Age (months)", nm)
    g.savefig(op.join(fig_dir, 'CDI_%s-by-age.eps' % nm),
              dpi=240, format='eps')
    plt.close(plt.gcf())

# Correlations: dependent measures vs. CDI measures
for feature in ws_measures:
    for age in cdi_ages:
        # feature
        Y = np.asarray(cdi_df[cdi_df.CDIAge == age][feature].values)
        assert Y.shape[-1] == n_samples
        for ii, _ in enumerate(groups):
            for jj in np.arange(len(selection)):
                for response, nm in zip([amps, latencies],
                                        ['strength(fT)', 'latency(ms)']):
                    for cc, cond in enumerate(conditions):
                        X = response[ii, cc, jj]  # response
                        X_intercept = sm.add_constant(X)
                        results = sm.OLS(Y, X_intercept).fit()
                        ols_ln = results.params[0] + results.params[1] * X
                        if results.f_pvalue < .05:
                            print('Regression results for %s...\n'
                                  % grp_nms[jj])
                            print(results.summary())
                            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                            ax.set(title=grp_nms[jj], ylabel=feature,
                                   xlabel=nm)
                            ax.plot(X, Y, 'o', markersize=10)
                            ax.plot(X, ols_ln, lw=1.5, ls='solid',
                                    color='Crimson', zorder=5)
                            box_off(ax)
                            ax.get_xticklabels()[0].\
                                set(horizontalalignment='right')
                            hb = Line2D(range(1), range(1), color='crimson')
                            ax.legend([hb], ['$\mathrm{r^{2}}=%.2f$\n'
                                             'p = %.3f' % (results.rsquared,
                                                           results.f_pvalue)],
                                      **leg_kwargs)
                            fig.tight_layout()
                            f_out = '%s-%d-x-%s_%s_%s-mos.eps' \
                                    % (feature, age, nm[:-4], cond, groups[ii])
                            fig.savefig(op.join(fig_dir, f_out), dpi=240,
                                        format='eps')
                            plt.close(plt.gcf())

# Individual subject difference scatters
cmap = plt.get_cmap('viridis')
clist = np.linspace(1, len(cmap.colors) - 1, n_samples).astype(int)
clist = np.asarray(cmap.colors)[clist]
deltas = [X[0] - X[1] for X in [amps, latencies]]  # 6 - 2 months
for nm, unit, delta in zip(['strength', 'latency'],
                           ['(fT)', '(ms)'], deltas):
    f, axs = plt.subplots(1, n_sel, figsize=(10, 5), sharex=True, sharey=True)
    f.canvas.set_window_title(nm)
    for kk in np.arange(len(selection)):
        for jj in np.arange(n_samples):
            axs[kk].plot(conditions, delta[:, kk, jj], label=subjects,
                         ls=':', lw=1.0,  marker='^',
                         color=clist[jj])
        axs[kk].set(title=selection[kk], ylabel='%s %s' % (nm, unit))
        axs[kk].grid(color='k', linestyle=':', linewidth=1, alpha=.2,
                     which='major')
        box_off(axs[kk])
        axs[kk].spines['left'].set_color('none')
        axs[kk].spines['top'].set_color('none')
    f.tight_layout()
    f.savefig(op.join(fig_dir, '%s_raw-delta.eps' % nm),
              dpi=240, format='eps')

# Dependents for MMN type activity
mmns = ['ba', 'wa']
mmn_ba = [X[:, 0] - X[:, 1] for X in [amps, latencies]]  # grp x sel x subj
mmn_wa = [X[:, 0] - X[:, 2] for X in [amps, latencies]]
# Put dependents into DF cond x grp x sel x subj
mmn_amps = np.concatenate((mmn_ba[0][np.newaxis, ...],
                           mmn_wa[0][np.newaxis, ...]), axis=0)
mmn_lats = np.concatenate((mmn_ba[1][np.newaxis, ...],
                           mmn_wa[1][np.newaxis, ...]), axis=0)
data = [np.transpose(X, (3, 1, 2, 0))
        for X in [mmn_amps, mmn_lats]]  # subj x grp x hem x cond
sz = data[-1].size
# interleave list --> tiled vector of levels for hemisphere
age = np.vstack(([2, 6], [2, 6]) * 2).reshape((-1,), order='F')
# tile and sort list --> tiled vector of levels for subjects
s = np.vstack((subjects, subjects) * 4).reshape((-1,), order='F')
hems = np.vstack((selection, selection)).reshape((-1,), order='F')
dep_df = pd.DataFrame({'subjID': s.tolist(),
                       'age': age.tolist() * n_samples,
                       'hemisphere': hems.tolist() * (sz//len(hems)),
                       'mmn': mmns * (sz//len(mmns)),
                       'amplitude': data[0].reshape(-1, order='C'),
                       'latency': data[1].reshape(-1, order='C')})

lh = dep_df[(dep_df.hemisphere == 'left')]
six = lh[lh.age == 6]
two = lh[lh.age == 2]
two.to_csv(op.join(data_dir, 'two_output.csv'))
six.to_csv(op.join(data_dir, 'six_output.csv'))
g = sns.lmplot(x='Age(days)', y='HC', data=xl_b, scatter=True,
               ci=95, truncate=True)
g.ax.spines['left'].set_color('none')
g.ax.spines['top'].set_color('none')
g.set_axis_labels(y_var='Head Circumference (cm)')
g.savefig(op.join(fig_dir, 'HCvsAge.eps'), dpi=240, format='eps')
