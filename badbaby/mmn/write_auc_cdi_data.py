# coding: utf-8

from os import path as op
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import mne
from mne import read_evokeds
from badbaby.return_dataframes import  return_simms_mmn_df
from badbaby.defaults import meg_dirs, project_dir


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


plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['axes.labelsize'] = 'small'
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['grid.color'] = '0.75'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['lines.markersize'] = '0.5'
leg_kwargs = dict(frameon=True, columnspacing=0.1, labelspacing=0.1,
                  fontsize=10, fancybox=False, handlelength=2.0,
                  loc='best')

static_dir = op.join(project_dir, 'static')
data_dir = meg_dirs['mmn']
fig_dir = op.join(data_dir, 'figures')

simms_df, simms_cdi = return_simms_mmn_df()
groups = np.unique(simms_df.Group).tolist()
remap = dict([(2, '2 months'), (6, '6 months')])
grp_names = [remap[kind] for kind in [2, 6]]
remap = dict([('lh', 'left'), ('rh', 'right')])
hemis = [remap[kind] for kind in ['lh', 'rh']]
subjects = np.unique(np.asarray([x[:3] for x in simms_df.Subject_ID.values]))

sns.set(style="whitegrid", palette="pastel", color_codes=True)
for nm in ['M3L', 'VOCAB']:
    g = sns.lmplot(x="CDIAge", y=nm, truncate=True, data=simms_cdi)
    g.set_axis_labels("Age (months)", nm)
    g.savefig(op.join(fig_dir, 'CDI_%s-by-age.png' % nm),
              dpi=300, format='png')
analysis = 'Oddball-matched'
colors = dict(deviant="Crimson", standard="CornFlowerBlue")
lpf = 30
peak_lats = list()
peak_amps = list()


for ii, group in enumerate(groups):
    subjects = simms_df[simms_df.Group == group].Subject_ID.values.tolist()
    for si, subj in enumerate(subjects):
        evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif'
                              % (analysis, lpf, subj))
        evoked = read_evokeds(evoked_file, condition='deviant',
                              baseline=(None, 0))
        if evoked.info['sfreq'] > 600.0:
            raise ValueError('bad_%s - %dHz Wrong sampling rate!'
                             % (subj, evoked.info['sfreq']))
        if len(evoked.info['bads']) > 0:
            evoked.interpolate_bads()
        times = evoked.times
        sfreq = evoked.info['sfreq']
        ch_names = evoked.ch_names
        pick_mag = mne.pick_types(evoked.info, meg='mag')
        assert pick_mag.shape[0] == 102
        pick_grad = mne.pick_types(evoked.info, meg='grad')
        assert pick_grad.shape[0] == 204
        # Do channel selection
        for sel, hem in enumerate(sensors.keys()):
            # subselect magnetometers channels
            picks = np.asarray(ch_names)[pick_mag]
            ch_selection = np.intersect1d(picks, sensors[hem]).tolist()
            assert len(ch_selection) == len(sensors[hem]) // 3
            evo_cp = evoked.copy().pick_channels(ch_selection)
            evo_cp.info.normalize_proj()
            _, lat, amp = evo_cp.get_peak(ch_type='mag', tmin=.2, tmax=.55,
                                          return_amplitude=True)
            if si == 0 and sel == 0:
                # subjects x hemispheres
                lats = np.zeros((len(subjects), len(sensors)))
                amps = np.zeros_like(lats)
            lats[si, sel] = lat
            amps[si, sel] = amp
    # subjects x hemispheres per group
    peak_lats.append(lats)
    peak_amps.append(amps)
# peak latency and amplitude difference between 2- and 6-months of age
latency_shifts = peak_lats[1] - peak_lats[0]
amplitude_diff = peak_amps[1] - peak_amps[0]
box_kwargs = {'showmeans': True, 'meanline': True}
for nm, unit, measure in zip(['Latency shift', 'Amplitude difference'],
                             ['ms', 'fT'],
                             [latency_shifts, amplitude_diff]):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set(title=nm)
    if nm is 'Latency shift':
        measures = np.asarray((measure[:, 0] * 1e3,
                               measure[:, 1] * 1e3))
    else:
        measures = np.asarray((measure[:, 0],
                               measure[:, 1]))
    plt.boxplot(measures.T.squeeze(), 1, labels=['Left', 'Right'],
                **box_kwargs)
    ax.set(ylabel=unit)
    box_off(ax)
    ax.get_xticklabels()[0].set(horizontalalignment='right')
    fig.tight_layout()
    f_out = '%s.png' % nm
    fig.savefig(op.join(fig_dir, f_out.replace(' ', '_')),
                dpi=300, format='png')

cdi_measure = np.asarray((simms_cdi[simms_cdi.CDIAge == 27].VOCAB.values,
                          simms_cdi[simms_cdi.CDIAge == 30].VOCAB.values))
two_mos_lats = peak_lats[0]
six_mos_lats = peak_lats[1]
for hi, hem in enumerate(sensors.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set(title=titles_comb[hi], ylabel='Vocabulary',
           xlabel='latency shift(ms)')
    X = two_mos_lats[:, hi] * 1e3  # response
    X_ = sm.add_constant(X)
    Y = cdi_measure[1]  # feature or predictor
    results = sm.OLS(Y, X_).fit()
    ols_ln = results.params[0] + results.params[1] * X
    print('Regression results for %s...\n' % titles_comb[hi])
    print(results.summary())
    ax.plot(X, Y, 'bo', markersize=10)
    ax.plot(X, ols_ln, lw=1.5, ls='solid', color='r', zorder=5)
    box_off(ax)
    ax.get_xticklabels()[0].set(horizontalalignment='right')
    ln = Line2D(range(1), range(1), color='k')
    ax.legend([ln], ['$\mathrm{r^{2}}=%.2f$\n'
                     'p = %.3f' % (results.rsquared,
                                   results.f_pvalue)],
              **leg_kwargs)
    fig.tight_layout()

plt.close('all')
