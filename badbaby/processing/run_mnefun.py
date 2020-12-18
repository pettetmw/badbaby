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

Subjects whose names are incorrect and need to be manually copied and renamed:

- bad_208a  bad_208_a
- bad_209a  bad_209
- bad_301a  bad_301
- bad_921a  bad_921
- bad_925a  bad_925
- bad_302a  bad_302
- bad_116a  bad_116
- bad_211a  bad_211

Subjects whose data were not on the server and needed to be uploaded were
[bad_114, bad_214, bad_110, bad_117a, bad_215a, bad_217, bad_119a].
Files were uploaded to brainstudio with variants of:
TODO priority(p)3 --files-from arg on ../static/missing.txt

    $ rsync -a --rsh="ssh -o KexAlgorithms=diffie-hellman-group1-sha1" --partial --progress --include="*_raw.fif" --include="*_raw-1.fif" --exclude="*" /media/ktavabi/ALAYA/data/ilabs/badbaby/*/bad_114/raw_fif/* larsoner@kasga.ilabs.uw.edu:/data06/larsoner/for_hank/brainstudio
    >>> mne.io.read_raw_fif('../mismatch/bad_114/raw_fif/bad_114_mmn_raw.fif', allow_maxshield='yes').info['meas_date'].strftime('%y%m%d')  # this neccessary?

Then repackaged manually into brainstudio/bad_baby/bad_*/*/ directories
based on the recording dates.

Subjects who did not complete preprocessing:

TODO add to defaults.exclude
- 223a: Preproc (Only 13/15 good ECG epochs found)

This is because for about half the time their HPI was no good, so throw them
out.

"""

import traceback

import numpy as np
import mnefun

from badbaby import defaults
from badbaby.defaults import df
from score import score, IN_NAMES, IN_NUMBERS

ecg_channel = dict((f'bad_{k}', v)
                   for k, v in zip(df['id'], df['ecg']))
work_dir = defaults.datadir

good, bad = list(), list()
subjects = sorted(f'bad_{id_}' for id_ in df['id'])
assert set(subjects) == set(ecg_channel)
assert len(subjects) == 76
subjects.pop(subjects.index('bad_223a'))  # cHPI is no good

# noinspection PyTypeChecker
tmin, tmax = defaults.epoching
params = mnefun.Params(
    tmin=tmin, tmax=tmax, n_jobs=18, n_jobs_fir='cuda', n_jobs_resample='cuda',
    proj_sfreq=250, decim=300., hp_cut=defaults.highpass, hp_trans='auto',
    lp_cut=defaults.lowpass, lp_trans='auto', bmin=tmin,
    ecg_channel=ecg_channel)
params.subjects = subjects
params.structurals = [None] * len(params.subjects)
params.score = score
params.dates = [None] * len(params.subjects)
params.work_dir = work_dir

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
params.st_correlation = .98
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
params.in_names = IN_NAMES
params.in_numbers = IN_NUMBERS
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
good, bad = list(), list()
use_subjects = params.subjects
for subject in use_subjects:
    params.subject_indices = [params.subjects.index(subject)]
    default = False
    try:
        mnefun.do_processing(
            params,
            fetch_raw=default,
            do_score=default,
            push_raw=default,
            do_sss=True,
            fetch_sss=default,
            do_ch_fix=default,
            gen_ssp=True,
            apply_ssp=True,
            write_epochs=True,
            gen_covs=default,
            gen_fwd=default,
            gen_inv=default,
            gen_report=default,
            print_status=default,
        )
    except Exception:
        raise
        traceback.print_exc()
        bad.append(subject)
    else:
        good.append(subject)
print(f'Successfuly processed {len(good)}/{len(good) + len(bad)}, bad:\n{bad}')
