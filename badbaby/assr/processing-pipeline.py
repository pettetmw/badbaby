# -*- coding: utf-8 -*-

""" mnefun script to process AM tone MEG data for Bad_baby experiment """

# Authors: Kambiz Tavabi <ktavabi@uw.edu>

from __future__ import print_function

import mnefun
import numpy as np

from picks import tone_subjects as pickedSubjects
from score import score

sfreqs = [k for k in pickedSubjects.iterkeys() if k.startswith('sr')]

try:
    # Use niprov as handler for events, or if it's not installed, ignore
    from niprov.mnefunsupport import handler
except ImportError:
    handler = None

for lst in sfreqs:
    sr = int(lst[2:6])
    assert sr in [1200, 1800]
    if sr == 1200:
        decim = 2
    else:
        decim = 3
    ecg_channel = lst[-4:]
    subjects = pickedSubjects[lst]
    nsubjects = len(subjects)

    # noinspection PyTypeChecker
    params = mnefun.Params(tmin=-0.1, tmax=1., n_jobs=18,
                           n_jobs_fir='cuda', n_jobs_resample='cuda',
                           decim=decim, proj_sfreq=200,
                           filter_length='10s', hp_cut=32., hp_trans='auto',
                           lp_trans='auto', lp_cut=48.,
                           bmin=-0.1, auto_bad=20.,
                           ecg_channel='MEG%s' % ecg_channel)
    params.subjects = ['bad_%s' % s for s in subjects]
    params.structurals = [None] * nsubjects  # None means use sphere
    params.on_process = handler
    params.score = score
    params.dates = [(2013, 0, 00)] * len(params.subjects)
    params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)),
                                          [])
    params.plot_drop_logs = False
    params.acq_ssh = 'kambiz@172.28.161.8'  # minea
    params.acq_dir = ['/data101/bad_baby', '/sinuhe/data01/bad_baby',
                      '/sinuhe/data03/bad_baby', '/sinuhe_data01/bad_baby']
    params.sws_ssh = 'kam@172.25.148.15'  # kasga
    params.sws_dir = '/data07/kam/bad_baby'
    # SSS options
    params.sss_type = 'python'
    params.sss_regularize = 'svd'
    params.tsss_dur = 4.
    params.int_order = 6
    params.st_correlation = .9
    params.trans_to = (0, 0, 0.03)
    # Trial/CH rejection criteria
    params.reject = dict(grad=3500e-13, mag=4000e-15)
    params.flat = dict(grad=1e-13, mag=1e-15)
    params.auto_bad_reject = dict(grad=7000e-13, mag=8000e-15)
    params.auto_bad_flat = params.flat
    params.ssp_ecg_reject = params.auto_bad_reject
    params.get_projs_from = np.arange(1)
    params.proj_nums = [[3, 3, 0],  # ECG: grad/mag/eeg
                        [0, 0, 0],  # EOG
                        [0, 0, 0]]  # Continuous (from ERM)
    params.run_names = ['%s_am']
    params.inv_names = ['%s']
    params.inv_runs = [np.arange(1)]
    params.runs_empty = []
    params.cov_method = 'shrunk'
    # Conditioning
    params.in_names = ['tone']
    params.in_numbers = [103]
    params.analyses = ['AM_tone']
    params.out_names = [['tone']]
    params.out_numbers = [[1]]
    params.must_match = [[]]

    mnefun.do_processing(
        params,
        fetch_raw=False,  # Fetch raw recording files from acq machine
        do_score=False,  # do scoring
        push_raw=False,  # Push raw files and SSS script to SSS workstation
        do_sss=False,  # Run SSS remotely
        fetch_sss=False,  # Fetch SSSed files
        do_ch_fix=False,  # Fix channel ordering
        gen_ssp=False,  # Generate SSP vectors
        apply_ssp=False,  # Apply SSP vectors and filtering
        plot_psd=False,
        write_epochs=False,  # Write epochs to disk
        gen_covs=False,  # Generate covariances
        gen_fwd=False,  # Generate forward solutions (and source space if needed)
        gen_inv=False,  # Generate inverses
        gen_report=False,
        print_status=True
    )
