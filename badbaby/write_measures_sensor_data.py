# -*- coding: utf-8 -*-

"""Write dependent measures for oddball sensor space data"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
import numpy as np
import seaborn as sns
import mne
from mne import (read_evokeds, grand_average)
from mne.stats import permutation_t_test
from mne.utils import _time_mask
from huma import defaults
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
    return erf


# Some parameters
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
age = 2
data_dir = params.meg_dirs['mmn']

meg_df, cdi_df = rd.return_dataframes('mmn', age=age, bezos=True)

#  Confirm data is correct
print('\nDescriptive stats for Age(days) variable...\n',
      meg_df['Age(days)'].describe())
meg_df.hist(column=['SES', 'HC', 'Age(days)'], layout=(3, 1),
            figsize=(8, 10), bins=50)
# Loop over subjects & plot grand average ERFs
subjects = meg_df.Subject_ID.values
tmin, tmax = (.09, .45)      # peak ERF latency window
print('Plotting Grand Averages')
for ci, cond in enumerate(conditions):
    print('     Loading data for %s / %s' % (analysis, cond))
    file_out = op.join(data_dir, '%s_%s_%s-mos_%d_grand-ave.fif'
                       % (analysis, cond, age, lpf))
    if not op.isfile(file_out):
        print('      Doing averaging...')
        evokeds = list()
        for si, subj in enumerate(subjects):
            print('       %s' % subj)
            evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                                  '%s_%d-sss_eq_bad_%s-ave.fif'
                                  % (analysis, lpf, subj))
            evoked = read_in_evoked(evoked_file, condition=cond)
            evokeds.append(evoked)
            if subj == subjects[0]:
                erf_data = np.zeros((len(subjects), len(evoked.info['chs']),
                                     len(evoked.times)))
            erf_data[si] = evoked.data
        np.savez(op.join(data_dir, '%s_%s_%s-mos_%d_evoked-arrays.npz'
                         % (analysis, cond, age, lpf)), erf_data=erf_data)
        # do grand averaging
        grandavr = grand_average(evokeds)
        grandavr.save(file_out)
    else:
        print('Reading...%s' % op.basename(file_out))
        grandavr = read_evokeds(file_out)[0]
    # peak ERF latency bn 100-550ms
    ch, lat = grandavr.get_peak(ch_type='mag', tmin=tmin, tmax=tmax)
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

#  For 'all' or 'deviant' conditions find maximally responsive sensors
Ds = dict()
for cond in conditions:
    if cond in ['all', 'deviant']:
        evoked = mne.Evoked(op.join(data_dir, '%s_%s_%s-mos_%d_grand-ave.fif'
                                    % (analysis, cond, age, lpf)))
        ds = np.load(op.join(data_dir, '%s_%s_%s-mos_%d_evoked-arrays.npz'
                             % (analysis, cond, age, lpf)))
        ds = ds['erf_data']
        times = evoked.times
        Ds[cond] = dict()
        for ch_type in ['grad', 'mag']:
            Ds[cond][ch_type] = dict()
            temporal_mask = np.logical_and(tmin <= times, times <= tmax)
            evoked_copy = evoked.copy().pick_types(meg=ch_type, exclude=[])
            info = evoked_copy.info
            picks = mne.pick_types(info, meg=ch_type, exclude=[])
            ixgrid = np.ix_(picks, temporal_mask)
            data = ds[:, ixgrid[0], ixgrid[1]]
            data = np.mean(data, axis=2)
            # T-test
            T0, p_values, H0 = permutation_t_test(data, n_permutations=1000,
                                                  n_jobs=18)
            # find significant sensors
            these_sensors = picks[p_values <= .05]
            print("Number of significant sensors : %d" % len(these_sensors))
            print("Sensors names : %s"
                  % np.asarray(grandavr.ch_names)[these_sensors])
            arr = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                                  info, tmin=0.)
            stats_picks = mne.pick_channels(info['ch_names'], these_sensors)
            mask = p_values[:, np.newaxis] <= 0.05
            fig = arr.plot_topomap(times=[0], ch_type=ch_type, mask=mask,
                                   scalings=1, res=240, contours=3, vmin=0.,
                                   vmax=np.max, cmap='Reds',
                                   units=r'$\mathrm{-log10}\ \mathit{p}$',
                                   cbar_fmt='-%0.1f', time_format=None,
                                   show_names=lambda x: x[0:] + ' ' * 20)
            fig.suptitle('cond-%s MEG-%s permutations' % (cond, ch_type))
            for kk in params.sensors.keys():
                Ds[cond][ch_type][kk] = \
                    list(set(np.asarray(info['ch_names'])[these_sensors]).
                         intersection(set(params.sensors[kk])))

# obs x conditions x sensor types x hemisphere
auc = np.zeros((len(subjects), len(conditions), 2, 2))
latencies = np.zeros_like(auc)
channels = np.zeros_like(auc, dtype='<U20')
naves = np.zeros((len(conditions), len(subjects)))
for ci, cond in enumerate(conditions):
    print('     Loading data for %s / %s' % (analysis, cond))
    for si, subj in enumerate(subjects):
        print('       %s' % subj)
        evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif'
                              % (analysis, lpf, subj))
        evoked = read_in_evoked(evoked_file, condition=cond)
        naves[ci, si] = evoked.nave
        for ii, ch_type in enumerate(['grad', 'mag']):
            for jj, hem in enumerate(params.sensors.keys()):
                these_sensors = params.sensors[hem]
                ev = evoked.copy().pick_channels(these_sensors)
                # Maximally responsive channels from L-R selection of sensors
                ch, lat = ev.get_peak(ch_type=ch_type,
                                      tmin=tmin, tmax=tmax)
                dummy = evoked.copy().pick_channels([ch])
                t0 = lat - .1
                mask = _time_mask(evoked.times, tmin=t0, tmax=lat,
                                  sfreq=evoked.info['sfreq'])
                auc[si, ci, ii, jj] = (np.sum(np.abs(dummy.data[:, mask])) *
                                       (len(evoked.times) /
                                        evoked.info['sfreq']))
                assert auc[si, ci, ii, jj] > 0
                latencies[si, ci, ii, jj] = lat
                channels[si, ci, ii, jj] = ch
                if len(Ds) > 0 and ch in Ds[cond][ch_type][hem]:
                    print(' %s in significant sensors' % ch)
np.savez(op.join(params.static_dir,
                 '%s_%s-mos_%d_measures.npz' % (analysis, age, lpf)),
         auc=auc, latencies=latencies, channels=channels, naves=naves)

# Compare ERF datasets for subset of maximally responsive channels
scores = Counter(channels.flat)
ch_names = np.array(list(scores.keys()))
counts = np.array(list(scores.values()))
x = np.arange(len(ch_names))
order = np.flipud(np.argsort(counts))
fig = plt.figure()
plt.bar(x, counts[order], color='#0504aa', width=.8)
plt.xticks(x, ch_names[order], rotation=90,
           ha='right')
plt.tick_params(axis='x', which='major', labelsize=10)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Channel')
plt.ylabel('Frequency')
plt.title('Peak location frequency')
maxfreq = counts.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.axhline(.5 * maxfreq, color='r', lw=2)

# Finds all unique elements and their positions
unique, pos = np.unique(channels, return_inverse=True)
counts = np.bincount(pos)  # Count the number of each unique element
maxpos = counts.argmax()   # Finds the positions of the maximum count
maxsort = counts.argsort()[::-1]
print(unique[maxpos], counts[maxpos])

layout = ch_names[order].astype('<U20').tolist()
evoked_dict = dict()
for cond in conditions:
    print('     Loading data for %s / %s' % (analysis, cond))
    evokeds = list()
    for si, subj in enumerate(subjects):
        print('       %s' % subj)
        evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif'
                              % (analysis, lpf, subj))
        evokeds.append(read_in_evoked(evoked_file, condition=cond))
    evoked_dict[cond] = evokeds
print(evoked_dict)

colors = {k: v for k, v in zip(conditions, ["CornFlowerBlue", "Crimson",
                                            'Teal'])}
for sensor in layout:
    pick = [evoked.ch_names.index(kk) for kk in layout]
    mne.viz.plot_compare_evokeds(evoked_dict, picks=pick, colors=colors,
                                 truncate_yaxis=True,
                                 vlines=[0, tmin, tmax],
                                 show_sensors=True, ci=True)

