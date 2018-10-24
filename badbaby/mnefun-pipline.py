# -*- coding: utf-8 -*-

""" mnefun script to process odd ball MEG data for Bad_baby experiment """

# Authors: Kambiz Tavabi <ktavabi@uw.edu>

from __future__ import print_function

import os.path as op
import numpy as np
import mnefun
import badbaby.return_dataframes as rd
# from score import score

try:
    # Use niprov as handler for events, or if it's not installed, ignore
    from niprov.mnefunsupport import handler
except ImportError:
    handler = None
#  TODO: verify & refactor
df = rd.return_dataframes('mmn')[0]
ecg_chs = np.unique(df['ECG'].tolist())
work_dir = meg_dirs['mmn']

for sr, decim in zip([1200, 1800], [2, 3]):
    for ch in ecg_chs:
        subjects = \
            df[(df['SR(Hz)'] == sr) & (df['ECG'] == ch)]['Subject_ID'].tolist()
        if len(subjects) == 0:
            continue
        # noinspection PyTypeChecker
        print('    \nUsing %d Hz as sampling rate and\n'
              '    %s as ECG surrogate...' % (sr, ch))
        print('    %d ' % len(subjects), 'Subjects: ', subjects)
        params = mnefun.Params(tmin=-0.1, tmax=0.6, n_jobs=18,
                               n_jobs_fir='cuda', n_jobs_resample='cuda',
                               proj_sfreq=200, decim=decim,
                               filter_length='30s', hp_cut=.1, hp_trans='auto',
                               lp_cut=30., lp_trans='auto', bmin=-0.1,
                               ecg_channel=ch)
        params.subjects = ['bad_%s' % ss for ss in subjects]
        # write prebad
        for si, subj in enumerate(subjects):
            bad_channels = df[df['Subject_ID'] == subj]['BAD'].tolist()
            if op.exists(op.join(work_dir, params.subjects[si],
                                 'raw_fif')):
                prebad_file = op.join(work_dir, params.subjects[si],
                                      'raw_fif',
                                      '%s_prebad.txt' % params.subjects[si])
                if not op.exists(prebad_file):
                    if bad_channels[0] == 'None':
                        with open(prebad_file, 'w') as f:
                            f.write("")
                    else:
                        bads = ["%s" % ch for ch in
                                bad_channels[0].split(', ')]
                        with open(prebad_file, 'w') as output:
                            for ch_name in bads:
                                output.write("%s\n" % ch_name)
        params.structurals = [None] * len(params.subjects)
        # params.score = score
        params.dates = [None] * len(params.subjects)
        params.subject_indices = np.arange(len(params.subjects))

        # SSH parameters for acquisition computer
        params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'  # minea
        params.acq_dir = ['/data101/bad_baby', '/sinuhe/data01/bad_baby',
                          '/sinuhe/data03/bad_baby', '/sinuhe_data01/bad_baby']
        params.sws_ssh = 'kam@kasga.ilabs.uw.edu'  # kasga
        params.sws_dir = '/data07/kam/bad_baby'

        # Set the niprov handler to deal with events:
        params.on_process = None

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

        # Trial/CH rejection criteria
        params.ssp_ecg_reject = dict(grad=np.inf, mag=np.inf)
        params.autoreject_thresholds = True
        params.autoreject_types = ('mag', 'grad')
        params.flat = dict(grad=1e-13, mag=1e-15)
        params.proj_nums = [[3, 3, 0],  # ECG: grad/mag/eeg
                            [0, 0, 0],  # EOG
                            [0, 0, 0]]  # Continuous (from ERM)
        # Inverse options
        params.run_names = ['%s_mmn']
        params.get_projs_from = np.arange(1)
        params.inv_names = ['%s']
        params.inv_runs = [np.arange(1)]
        params.runs_empty = []
        params.cov_method = 'empirical'
        # Conditioning
        params.in_names = ['standard', 'ba', 'wa']
        params.in_numbers = [103, 104, 105]
        params.analyses = [
            'All',
            'Individual',
            'Oddball',
            'Individual-matched',
            'Oddball-matched'
        ]
        params.out_names = [
            ['All'],
            ['standard', 'Ba', 'Wa'],
            ['standard', 'deviant'],
            ['standard', 'Ba', 'Wa'],
            ['standard', 'deviant']
        ]
        params.out_numbers = [
            [1, 1, 1],  # Combine all trials
            [1, 2, 3],  # All conditions
            [1, 2, 2],  # oddball
            [1, 2, 3],
            [1, 2, 2]
        ]
        params.must_match = [
            [],
            [],
            [],
            [0, 1, 2],
            [0, 1]
        ]
        default = False
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
            psd=True,
        )
        # Set what will run
        mnefun.do_processing(
            params,
            fetch_raw=default,
            push_raw=default,
            do_sss=default,
            do_score=default,
            fetch_sss=default,
            do_ch_fix=default,
            gen_ssp=default,
            apply_ssp=default,
            write_epochs=default,
            gen_covs=default,
            gen_fwd=default,
            gen_inv=default,
            gen_report=default,
            print_status=True
        )
