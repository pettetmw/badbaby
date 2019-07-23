"""Plot XDAWN ERF sensor data for oddball stimuli."""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2019, Seattle, Washington'
__license__ = 'MIT'
__version__ = '0.2'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'

from os import path as op

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from mne.externals.h5io import read_hdf5
from scipy import stats

from badbaby.python import defaults

# parameters
workdir = defaults.datapath
analysis = 'oddball'
conditions = ['standard', 'deviant']
plt.style.use('ggplot')
sns.set(style='whitegrid')

tmin, tmax = defaults.epoching
lp = defaults.lowpass
age = [2, 6]
window = defaults.peak_window  # peak ERF latency window
ds = dict()

# read in xdawn virtual sensor data
for aix in age:
    for cc in conditions:
        h5 = op.join(defaults.datadir, '%dmos_%d-%s_%s_xdawn.h5'
                     % (aix, lp, analysis, cc))
        ds[aix] = read_hdf5(h5, 'xdawn')

# wrangle data via xarray into Pandas DF
dfx = list()
for aix in ds.keys():
    dsx = xr.DataArray(ds[aix]['signals'],
                       coords=[conditions,
                               ds[aix]['subjects'],
                               ds[aix]['times']],
                       dims=['condition', 'subject', 'time'])
    
    dfx.append(pd.concat([dsx.to_dataframe(name='amplitude')],
                         keys=[str(aix)],
                         names=['age']))
dfs = pd.concat(dfx)
dfs_tidy = pd.DataFrame(dfs.to_records())

# plot xdawn time series data
fig = sns.relplot(x='time', y='amplitude', hue='condition',
                  col='age', kind='line', data=dfs_tidy)
fig.savefig(op.join(defaults.figsdir,
                    '%s_lp-%d_%d-mos.pdf' % (analysis, lp, aix)),
            bbox_inches='tight')
# slice oddball data and merge into new (mmn) df
idx = pd.IndexSlice
right = dfs.loc(axis=0)[idx[:, 'standard']]
mmn = right.merge(dfs.loc(axis=0)[idx[:, 'deviant']],
                  on=['age', 'subject', 'time'], how='left',
                  suffixes=tuple('_%s' % ss for ss in conditions))
mmn['amplitude'] = mmn.amplitude_deviant - mmn.amplitude_standard
fig = sns.relplot(x='time', y='amplitude', col='age', kind='line',
                  data=pd.DataFrame(mmn.to_records()))
fig.savefig(op.join(defaults.figsdir, 'mmn_lp-%d_%d-mos.pdf' % (lp, aix)),
            bbox_inches='tight')

arr_2 = mmn.amplitude.unstack().xs('2').values
arr_6 = mmn.amplitude.unstack().xs('6').values
t_stat, p_val = stats.ttest_ind(arr_2, arr_6, axis=0)
