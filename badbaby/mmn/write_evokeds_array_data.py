# coding: utf-8

"""Script writes out numpy archive of nd evoked array data"""
from os import path as op
import time
import numpy as np
from mne import read_evokeds
from badbaby import defaults as params
from badbaby import return_dataframes as rd

# Some parameters
data_dir = params.meg_dirs['mmn']
df, cdi_df = rd.return_ford_mmn_dfs()
subjects = df.Subject_ID.values.tolist()
analysis = 'Individual'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
file_out = op.join(data_dir,
                   'ford_Analysis-%s_%d-ERF-data.npz' % (analysis, lpf))

# Loop over subjects and write ND data matrix
t0 = time.time()
for ci, cond in enumerate(conditions):
    print('   %s...' % cond)
    for si, subj in enumerate(subjects):
        print('     %s' % subj)
        evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                              '%s_%d-sss_eq_bad_%s-ave.fif'
                              % (analysis, lpf, subj))
        evoked = read_evokeds(evoked_file, condition=cond,
                              baseline=(None, 0))
        if len(evoked.info['bads']) > 0:
            print('     Interpolating bad channels...')
            evoked.interpolate_bads()
        times = evoked.times
        sfreq = evoked.info['sfreq']
        ch_names = evoked.info['ch_names']
        assert(all(np.asarray(ch_names) == np.asarray(params.vv_ch_order)))
        if ci == 0 and si == 0:
            data_array = np.zeros((len(conditions), len(subjects),
                                   evoked.data.shape[0],
                                   evoked.data.shape[1]))
        data_array[ci, si] = evoked.data
print(' Writing data...')
np.savez(file_out, data=data_array, times=times, sfreq=sfreq)
print('  Elapsed: %0.1f min' % ((time.time() - t0) / 60.))
