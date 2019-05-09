# -*- coding: utf-8 -*-

""" Plot esnsor permutation-clustering for given condition(s)"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
import numpy as np
import mne
from mne.stats import permutation_t_test
import badbaby.defaults as params

# Some parameters
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
age = 2
tmin, tmax = (.09, .45)      # peak ERF latency window
data_dir = params.meg_dirs['mmn']
channels = ['grad']
sensor_clusters = dict()
for cond in conditions:
    evoked = mne.Evoked(op.join(data_dir, '%s_%s_%s-mos_%d_grand-ave.fif'
                                % (analysis, cond, age, lpf)))
    ds = np.load(op.join(data_dir, '%s_%s_%s-mos_%d_evoked-arrays.npz'
                         % (analysis, cond, age, lpf)))
    ds = ds['erf_data']
    times = evoked.times
    sensor_clusters[cond] = dict()
    for ch_type in channels:
        sensor_clusters[cond][ch_type] = dict()
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
        significant_sensors = picks[p_values <= .05]
        significant_sensors_names = [info['ch_names'][k] for k in
                                     significant_sensors]
        print("Number of significant sensors : %d" % len(significant_sensors))
        print("Sensors names : %s" % significant_sensors_names)
        # Extract mask and indices of active sensors in the layout
        stats_picks = mne.pick_channels(info['ch_names'],
                                        significant_sensors_names)
        mask = p_values[:, np.newaxis] <= 0.05

        arr = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                              info, tmin=0.)
        fig = arr.plot_topomap(times=[0.], ch_type=ch_type,
                               mask=mask,
                               scalings=1, res=240, contours=3, vmin=0.,
                               vmax=np.max, cmap='Reds',
                               units=r'$\mathrm{-log10}\ \mathit{p}$',
                               cbar_fmt='-%0.1f', time_format=None,
                               show_names=lambda x: x[0:] + ' ' * 20,
                               outlines='skirt')
        fig.suptitle('%s %s' % (cond, ch_type))
        for kk in params.sensors.keys():
            sensor_clusters[cond][ch_type][kk] = \
                list(set(np.asarray(info['ch_names'])[significant_sensors]).
                     intersection(set(params.sensors[kk])))
