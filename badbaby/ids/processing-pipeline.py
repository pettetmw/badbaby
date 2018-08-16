# -*- coding: utf-8 -*-
# mnefun script to process IDS MEG data for Bad_baby experiment
# Authors: Kambiz Tavabi <ktavabi@uw.edu>
from __future__ import print_function

import mnefun
import numpy as np
from picks import speech_subjects as pickedSubjects

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
    params = mnefun.Params(n_jobs=18, decim=decim, proj_sfreq=200,
                           n_jobs_fir='cuda', n_jobs_resample='cuda',
                           filter_length='30s', lp_cut=80.,
                           ecg_channel='MEG%s' % ecg_channel)
    params.subjects = ['bad_%s' % s for s in subjects]
    params.structurals = [None] * nsubjects  # None means use sphere
    params.score = None  # defaults to passing events through
    params.dates = [None] * nsubjects  # None means more fully anonymize
    params.subject_indices = np.arange(
        nsubjects)  # Define which subjects to run

    # Set parameters for remotely connecting to acquisition computer
    params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'
    params.acq_dir = '/data101/bad_baby'
    # Set parameters for remotely connecting to SSS workstation ('sws')
    params.sws_ssh = 'kam@kasga.ilabs.uw.edu'
    params.sws_dir = '/data07/kam/badbaby'

    # Set the niprov handler to deal with events:
    params.on_process = handler

    params.run_names = ['%s_ids']
    params.get_projs_from = np.arange(1)
    params.inv_names = ['%s']
    params.inv_runs = [np.arange(1)]
    params.runs_empty = []

    params.proj_nums = [[3, 3, 0],  # ECG: grad/mag/eeg
                        [0, 0, 0],  # EOG
                        [0, 0, 0]]  # Continuous (from ERM)

    # Maxwell filter with mne-python
    params.sss_type = 'python'
    params.sss_regularize = 'svd'  # mc-svd SSS
    params.tsss_dur = 4.
    params.int_order = 6
    params.st_correlation = .9
    params.trans_to = (0, 0, 0.03)

    # Set what will execute
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
        plot_psd=False,  # Plot raw data power spectra
        write_epochs=False,  # Write epochs to disk
        gen_covs=False,  # Generate covariances
        gen_fwd=False,
        # Generate forward solutions (and source space if needed)
        gen_inv=False,  # Generate inverses
        gen_report=False,  # Write mne report html of results to disk
        print_status=True  # Print completeness status update
    )
