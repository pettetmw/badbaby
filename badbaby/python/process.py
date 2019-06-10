#!/usr/bin/env python

"""Infant auditory MEG data MNEFUN processing pipeline."""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2018, Seattle, Washington'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'
__status__ = 'Development'

import os.path as op
import numpy as np
import pandas as pd
import mnefun
import badbaby.python.return_dataframes as rd
from badbaby.python import defaults

# from score import score

try:
    # Use niprov as handler for events, or if it's not installed, ignore
    from niprov.mnefunsupport import handler
except ImportError:
    handler = None
df = rd.return_dataframes('assr')[0]  # could be 'mmn', 'assr', 'ids'
exclude = defaults.paradigms['exclusion']['assr']
df.drop(df[df.subjId.isin(pd.Series(exclude))].index, inplace=True)
ecg_chs = np.unique(df['ecg'].tolist())
work_dir = defaults.paradigms['assr']  # could be 'mmn', 'assr', 'ids'

for sr, decim in zip([1200, 1800], [2, 3]):
    for ch in ecg_chs:
        subjects = \
            df[(df['samplingRate'] == sr) &
               (df['ecg'] == ch)]['subjId'].tolist()
        if len(subjects) == 0:
            continue
        # noinspection PyTypeChecker
        print('    \nUsing %d Hz as sampling rate and\n'
              '    %s as ECG surrogate...' % (sr, ch))
        print('    %d ' % len(subjects), 'Subjects: ', subjects)
        params = mnefun.Params(tmin=-0.2, tmax=1.1, n_jobs=18,
                               n_jobs_fir='cuda', n_jobs_resample='cuda',
                               proj_sfreq=250, decim=decim,
                               hp_trans='auto', lp_cut=100.,
                               lp_trans='auto', bmin=-0.2,
                               ecg_channel=ch, auto_bad=.1)
        params.subjects = ['bad_%s' % ss for ss in subjects]
        # write prebad
        for si, subj in enumerate(subjects):
            bad_channels = df[df['subjId'] == subj]['badCh'].tolist()
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
        
        params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'  # minea
        params.acq_dir = ['/data101/bad_baby', '/sinuhe/data01/bad_baby',
                          '/sinuhe/data03/bad_baby', '/sinuhe_data01/bad_baby']
        params.sws_ssh = 'kam@kasga.ilabs.uw.edu'  # kasga
        params.sws_dir = '/data07/kam/sss_work'
        # Set the parameters for head position estimation:
        params.coil_dist_limit = 0.01
        params.coil_t_window = 'auto'  # use the smallest reasonable window size
        # remove segments with < 3 good coils for at least 1 sec
        params.coil_bad_count_duration_limit = 1.  # sec
        # Annotation params
        params.rotation_limit = 20.  # deg/s
        params.translation_limit = 0.01  # m/s
        # Do MF autobad
        params.mf_autobad = True
        # Maxwell filter with mne-python
        params.sss_type = 'python'
        params.sss_regularize = 'in'
        params.tsss_dur = 4.
        params.int_order = 6
        params.st_correlation = .9
        params.trans_to = (0., 0., 0.06)
        # Covariance
        params.runs_empty = ['%s_erm_01']  # Define empty room runs
        params.cov_method = 'ledoit_wolf'
        params.compute_rank = True  # compute rank of the noise covar matrix
        params.force_erm_cov_rank_full = False  # compute and use the
        # empty-room rank
        # Trial/CH rejection criteria
        params.ssp_ecg_reject = dict(grad=np.inf, mag=np.inf)
        params.autoreject_thresholds = True
        params.autoreject_types = ('mag', 'grad')
        params.flat = dict(grad=1e-13, mag=1e-15)
        # params.auto_bad_reject = dict(grad=100.0e-12, mag=90.0e-10)
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
        params.run_names = ['%s_am']
        params.get_projs_from = np.arange(1)
        params.inv_names = ['%s']
        params.inv_runs = [np.arange(1)]
        params.runs_empty = []
        # Conditioning
        params.in_names = ['tone']
        params.in_numbers = [1]
        params.analyses = ['All']
        params.out_names = [['Tone']]
        params.out_numbers = [[1]]  # Combine all trials
        params.must_match = [[]]
        cov = params.inv_names[0] + '-%.0f-sss-cov.fif' % params.lp_cut
        params.report_params.update(
                sensor=[dict(analysis='All', name='Tone', times='peaks')],
                source=None,
                source_alignment=False,
                psd=True,
                )
        # Set what will run
        default = False
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
                write_epochs=True,
                gen_covs=True,
                gen_fwd=default,
                gen_inv=default,
                gen_report=default,
                print_status=True
                )
