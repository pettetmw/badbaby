#!/usr/bin/env python

"""mnefun.py: MNEFUN preprocessing pipeline:
        1. Determine ACQ sampling rate and ECG channel.
        2. Write ACQ prebad channel to disk.
        3. Score.
        4. HP estimation yeilding annotation parameters.
        5. MF and move comp.
        6. Data & ERM covariances.
        7. Autoreject to threshold & reject noisy trials
        8. Compute ECG & ERM projectors
        9. Epoching & writing evoked data to disk.
"""

import os.path as op

import mnefun
import numpy as np
import pandas as pd

from badbaby import defaults
from badbaby.defaults import return_dataframes

try:
    # Use niprov as handler for events, or if it's not installed, ignore
    from niprov.mnefunsupport import handler
except ImportError:
    handler = None
df = return_dataframes('mmn')[0]
exclude = defaults.exclude
df.drop(df[df.index.isin(pd.Series(exclude, dtype=int))].index, inplace=True)
ecg_chs = np.unique(df['ecg'].tolist())
work_dir = defaults.datadir

for sr, decim in zip([1200, 1800], [2, 3]):
    for ch in ecg_chs:
        subjects = \
            df[(df['samplingRate'] == sr) & (df['ecg'] == ch)].index.tolist()
        if len(subjects) == 0:
            continue
        # noinspection PyTypeChecker
        print('    \nUsing %d Hz as sampling rate and\n'
              '    %s as ECG surrogate...' % (sr, ch))
        print('    %d ' % len(subjects), 'Subjects: ', subjects)
        tmin, tmax = defaults.epoching
        params = mnefun.Params(tmin=tmin, tmax=tmax, n_jobs=18,
                               n_jobs_fir='cuda', n_jobs_resample='cuda',
                               proj_sfreq=250, decim=decim,
                               hp_cut=defaults.highpass, hp_trans='auto',
                               lp_cut=defaults.lowpass, lp_trans='auto',
                               bmin=tmin, ecg_channel=ch)
        params.subjects = ['bad_%s' % ss for ss in subjects]
        # write prebad
        for si, subj in enumerate(subjects):
            bad_channels = df[df.index == subj]['badCh'].tolist()
            if op.exists(op.join(work_dir, params.subjects[si],
                                 'raw_fif')):
                prebad_file = op.join(work_dir, params.subjects[si],
                                      'raw_fif',
                                      '%s_prebad.txt' % params.subjects[si])
                if not op.exists(prebad_file):
                    if bad_channels[0] == 'None':
                        with open(prebad_file, 'w') as f:
                            f.write('')
                    else:
                        bads = ['%s' % ch for ch in
                                bad_channels[0].split(', ')]
                        with open(prebad_file, 'w') as output:
                            for ch_name in bads:
                                output.write('%s\n' % ch_name)
        # Set the niprov handler to deal with events:
        params.on_process = None
        params.structurals = [None] * len(params.subjects)
        # params.score = score
        params.dates = [None] * len(params.subjects)
        params.work_dir = work_dir
        params.subject_indices = np.arange(len(params.subjects))
        try:
            params.subject_indices = [params.subjects.index('bad_117b')]
        except ValueError:
            continue

        params.acq_ssh = 'kasga.ilabs.uw.edu'  # minea
        params.acq_dir = ['/brainstudio/bad_baby']
        # Set the parameters for head position estimation:
        params.coil_dist_limit = 0.01
        params.coil_t_window = 'auto'  # use the smallest reasonable window size
        # remove segments with < 3 good coils for at least 1 sec
        params.coil_bad_count_duration_limit = 1.  # sec
        # Annotation params
        params.rotation_limit = 20.  # deg/s
        params.translation_limit = 0.01  # m/s
        # Maxwell filter with mne-python
        params.sss_type = 'python'
        params.sss_regularize = 'in'
        params.tsss_dur = 4.
        params.int_order = 6
        params.st_correlation = .9
        params.trans_to = (0., 0., 0.06)
        # Covariance
        params.runs_empty = ['%s_erm']  # Define empty room runs
        # params.cov_method = 'ledoit_wolf'
        # params.compute_rank = True  # compute rank of the noise covar matrix
        # params.force_erm_cov_rank_full = False  # compute and use the
        # empty-room rank
        # Trial/CH rejection criteria
        params.ssp_ecg_reject = dict(grad=np.inf, mag=np.inf)
        params.autoreject_thresholds = True
        params.autoreject_types = ('mag', 'grad')
        params.flat = dict(grad=1e-13, mag=1e-15)
        params.auto_bad_reject = 'auto'
        params.auto_bad_flat = params.flat
        params.auto_bad_meg_thresh = 15
        # Proj
        params.get_projs_from = np.arange(1)
        params.proj_ave = True
        # params.proj_meg = 'combined'
        params.inv_names = ['%s']
        params.inv_runs = [np.arange(1)]
        params.ecg_t_lims = (-0.04, 0.04)
        params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                            [0, 0, 0],  # EOG
                            [1, 1, 0]]  # Continuous (from ERM)
        # Inverse options
        params.run_names = ['%s_' + defaults.run_name]
        params.get_projs_from = np.arange(1)
        params.inv_names = ['%s']
        params.inv_runs = [np.arange(1)]
        params.runs_empty = []
        # Conditioning
        params.in_names = ['standard', 'ba', 'wa']
        params.in_numbers = [103, 104, 105]
        params.analyses = [
            'All',
            'Individual',
            'Oddball'
            ]
        params.out_names = [
            ['All'],
            ['standard', 'ba', 'wa'],
            ['standard', 'deviant']
            ]
        params.out_numbers = [
            [1, 1, 1],  # Combine all trials
            [1, 2, 3],  # All conditions
            [1, 2, 2]  # oddball
            ]
        params.must_match = [
            [],
            [0, 1, 2],
            [0, 1, 2]
            ]
        cov = params.inv_names[0] + '-%.0f-sss-cov.fif' % params.lp_cut
        params.report_params.update(
            whitening=[
                dict(analysis='All', name='All', cov=cov),
                dict(analysis='Oddball', name='standard', cov=cov),
                dict(analysis='Oddball', name='deviant', cov=cov)
                ],
            sensor=[
                dict(analysis='All', name='All', times='peaks'),
                dict(analysis='Oddball', name='standard', times='peaks'),
                dict(analysis='Oddball', name='deviant', times='peaks')
                ],
            source=None,
            psd=True
            )
        # Set what will run
        default = False
        mnefun.do_processing(
            params,
            fetch_raw=True,
            push_raw=True,
            do_sss=True,
            do_score=True,
            fetch_sss=default,
            do_ch_fix=default,
            gen_ssp=default,
            apply_ssp=default,
            write_epochs=default,
            gen_covs=default,
            gen_fwd=default,
            gen_inv=default,
            gen_report=default,
            print_status=default,
            )
