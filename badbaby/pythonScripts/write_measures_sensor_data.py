# -*- coding: utf-8 -*-

"""Write response variable data for oddball sensor space data.

    Script developed to extract peak ERF components AUC and latency for oddball
    responses to CV stimuli.
        1. Yield filtered dataframes for behavioral and MEG data.
        2. plot MEG covariate histograms.
        3. Extract evoked arrays for selected subjects.
        4. Write out ERF data in numpy array.
        5. Compute grand averaged ERF data.
        6. Plot group ERF timeseries and topography at peak (grad)
        latency between 150-510ms.
        7. Plot group ERF timecourse at peak location for each condition.
        8. Plot frequency of maximally responsive channels for latencies.
        9. Yield ND arrays of ERF peak latency & location from L-R selection
        of sensors.
        10. Plot PD of overall peak latencies.
        11. Plot ranked histogram of maximally responsive channels.
        12. Plot montage of maximally responsive channels.
        13. Compute area under the curve along "rising" slope (100 ms) to
        peak latency for intersection of L-R MEG sensors and montage of
        maximally responsive channels and storing as ND array.
        14. Plot PD of overall AUC.
        15. Write out ERF measurements as NPZ file.
        16. Plot group ERF timecourse average across channels in montage.

"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import mne
from mne import (read_evokeds, grand_average)
from mne.utils import _time_mask
from huma import defaults
import badbaby.pythonScripts.defaults as params
import badbaby.pythonScripts.return_dataframes as rd


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
    return erf


# Some parameters
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
age = 2
meg_dataDir = params.megPdg_dirs['mmn']

meg_df, cdi_df = rd.return_dataframes('mmn', age=age, ses=True)
print('\nDescriptive stats for Age(days) variable...\n',
      meg_df['age'].describe())
meg_df.hist(column=['ses', 'headSize', 'age'], layout=(3, 1),
            figsize=(8, 10), bins=50)
# Loop over subjects & plot grand average ERFs
subjects = meg_df.subjId.values
tmin, tmax = (.15, .51)      # peak ERF latency window
evoked_dict = dict()
picks_dict = dict()
for ci, cond in enumerate(conditions):
    print('     Loading data for %s / %s' % (analysis, cond))
    file_out = op.join(meg_dataDir, '%s_%s_%s-mos_%d_grand-ave.fif'
                       % (analysis, cond, age, lpf))
    print('      Doing averaging...')
    evokeds = list()
    for si, subj in enumerate(subjects):
        print('       %s' % subj)
        evoked_file = op.join(meg_dataDir, 'bad_%s' % subj, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif'
                              % (analysis, lpf, subj))
        evoked = read_in_evoked(evoked_file, condition=cond)
        evokeds.append(evoked)
        if subj == subjects[0]:
            erf_data = np.zeros((len(subjects), len(evoked.info['chs']),
                                 len(evoked.times)))
        erf_data[si] = evoked.data
    np.savez(op.join(meg_dataDir, '%s_%s_%s-mos_%d_evoked-arrays.npz'
                     % (analysis, cond, age, lpf)), erf_data=erf_data)
    # do grand averaging
    grandavr = grand_average(evokeds)
    grandavr.save(file_out)
    evoked_dict[cond] = evokeds
    # Grand average peak ERF gradiometer activity
    ch, lat = grandavr.get_peak(ch_type='grad', tmin=tmin, tmax=tmax)
    picks_dict[cond] = ch
    if cond in ['all', 'deviant']:
        print('     Peak latency for %s at:\n'
              '         %s at %0.3fms' % (cond, ch, lat))
    # plot group ERF topography
    timing = [lat - .1, lat]
    hs = grandavr.plot_joint(title=cond, times=timing,
                             ts_args=params.ts_args,
                             topomap_args=params.topomap_args)

#  compare group ERF time courses at peak locations
for pick in [grandavr.ch_names.index(kk) for kk in set(picks_dict.values())]:
    mne.viz.plot_compare_evokeds(evoked_dict, picks=pick,
                                 truncate_yaxis=True,
                                 vlines=[0, tmin, tmax],
                                 show_sensors=True, ci=True)

# Get ERF magnitudes and latencies...
# obs x conditions x sensor types x hemisphere
latencies = np.zeros((len(subjects), len(conditions), 2, 2))
auc = np.zeros_like(latencies)
channels = np.zeros_like(latencies, dtype='<U20')
naves = np.zeros((len(conditions), len(subjects)))
for ci, cond in enumerate(conditions):
    print('     Finding peak for %s: %s' % (analysis, cond))
    for ei, evoked in enumerate(evoked_dict[cond]):
        print('       %s' % evoked)
        dummy_info = evoked.info
        naves[ci, ei] = evoked.nave
        for ii, ch_type in enumerate(['grad', 'mag']):
            for jj, hem in enumerate(params.sensors.keys()):
                these_sensors = params.sensors[hem]
                ev = evoked.copy().pick_channels(these_sensors)
                # Find ERF peaks from L-R selection of sensors
                ch, lat = ev.get_peak(ch_type=ch_type,
                                      tmin=tmin, tmax=tmax)
                latencies[ei, ci, ii, jj] = lat
                channels[ei, ci, ii, jj] = ch

# Plots
# latencies
# An "interface" to matplotlib.axes.Axes.hist() method
# https://realpython.com/python-histograms/
_, ax = plt.subplots()
x = -np.log(latencies)
n, bins, patches = ax.hist(x=latencies.flat, bins='auto', color='#0504aa',
                           alpha=0.7, density=1)
# add a 'best fit' line
sigma = latencies.std()
mu = latencies.mean()
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, 'y', lw=2)
# Set a clean upper y-axis limit.
ax.grid(axis='y', alpha=0.3, ls=':')
ax.set_xlabel('Latency (AU)')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Latencies: $\mu=%.3f, \sigma=%.3f$' % (mu, sigma))

# frequency of maximally responsive channels
scores = Counter(channels.flat)
ch_names = np.array(list(scores.keys()))
counts = np.array(list(scores.values()))
x = np.arange(len(ch_names))
order = np.flipud(np.argsort(counts))
layout = ch_names[order].astype('<U20').tolist()
picks = mne.pick_channels(dummy_info['ch_names'], layout)
fig, ax = plt.subplots()
ax.bar(x, counts[order], color='#0504aa', width=.8)
ax.set_xticks(x)
ax.set_xticklabels(ch_names[order], fontdict={'fontsize': 6,
                                              'rotation': 90,
                                              'ha': 'right'})
ax.tick_params(axis='x', which='major', labelsize=10)
ax.grid(axis='y', alpha=0.75)
ax.set_xlabel('Channel')
ax.set_ylabel('Counts')
ax.set_title('Maximally responsive channels')
maxfreq = counts.max()
# Set a clean upper y-axis limit.
ax.set_ylim(
    ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
ax.axhline(.5 * maxfreq, color='r', lw=2, zorder=2)
mne.viz.plot_sensors(dummy_info, ch_type='grad',
                     ch_groups=picks[:, np.newaxis].T)

for ci, cond in enumerate(conditions):
    print('     Computing area under the curve for %s - %s' % (analysis, cond))
    for ei, evoked in enumerate(evoked_dict[cond]):
        print('       %s' % evoked)
        for ii, ch_type in enumerate(['grad', 'mag']):
            for jj, hem in enumerate(params.sensors.keys()):
                picks = list(set(layout).intersection(params.sensors[hem]))
                t0 = latencies[ei, ci, ii, jj] - .1
                ev = evoked.copy().pick_channels(picks)
                mask = _time_mask(ev.times, tmin=t0,
                                  tmax=latencies[ei, ci, ii, jj],
                                  sfreq=ev.info['sfreq'])
                auc[ei, ci, ii, jj] = (np.sum(np.abs(ev.data[:, mask])) *
                                       (len(ev.times) /
                                        ev.info['sfreq']))
                assert auc[ei, ci, ii, jj] > 0
# Area under curve
_, ax = plt.subplots()
x = -np.log(auc)
n, bins, patches = ax.hist(x=x.flat, bins='auto', color='#0504aa',
                           alpha=0.7, density=1)
# add a 'best fit' line
sigma = x.std()
mu = x.mean()
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, 'y', lw=2)
# Set a clean upper y-axis limit.
ax.grid(axis='y', alpha=0.3, ls=':')
ax.set_xlabel('Strength (AU)')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Area under curve: $\mu=%.3f, \sigma=%.3f$'
             % (mu, sigma))
# Write out ERF measurements
np.savez(op.join(params.dataDir,
                 '%s_%s-mos_%d_measures.npz' % (analysis, age, lpf)),
         auc=auc, latencies=latencies, channels=channels, naves=naves)

# Compare ERF datasets for subset of maximally responsive channels
picks = [evoked.ch_names.index(kk) for kk in layout]
mne.viz.plot_compare_evokeds(evoked_dict, picks=picks,
                             truncate_yaxis=True,
                             vlines=[0, tmin, tmax],
                             show_sensors=True, ci=True)

