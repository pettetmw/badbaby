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

    $ rsync -a --rsh="ssh -o KexAlgorithms=diffie-hellman-group1-sha1" --partial --progress --include="*_raw.fif" --include="*_raw-1.fif" --exclude="*" /media/ktavabi/ALAYA/data/ilabs/badbaby/*/bad_114/raw_fif/* larsoner@kasga.ilabs.uw.edu:/data06/larsoner/for_hank/brainstudio
    >>> mne.io.read_raw_fif('../mismatch/bad_114/raw_fif/bad_114_mmn_raw.fif', allow_maxshield='yes').info['meas_date'].strftime('%y%m%d')

Then repackaged manually into brainstudio/bad_baby/bad_*/*/ directories
based on the recording dates.

Subjects who did not complete preprocessing:

- 223a: Preproc (Only 13/15 good ECG epochs found)

This is because for about half the time their HPI was no good, so throw them
out.

"""

import os.path as op
import traceback
from pathlib import Path

import janitor  # noqa
import mnefun
import pandas as pd
import numpy as np

from score import score

static = op.join(Path(__file__).parents[1], "static")

columns = [
    "subjid",
    "behavioral",
    "complete",
    "ses",
    "age",
    "gender",
    "headsize",
    "maternaledu",
    "paternaledu",
    "maternalethno",
    "paternalethno",
    "ecg",
]

meg_features = (
    pd.read_excel(op.join(static, "meg_covariates.xlsx"), sheet_name="mmn")
    .clean_names()
    .select_columns(columns)
    .encode_categorical(columns=columns)
    .rename_columns({"subjid": "id"})
    .filter_on("complete == 1", complement=False)
)

ecg_channel = dict(
    (f"bad_{k}", v) for k, v in zip(meg_features["id"], meg_features["ecg"])
)

good, bad = list(), list()
subjects = sorted(f"bad_{id_}" for id_ in meg_features["id"])
assert set(subjects) == set(ecg_channel)
assert len(subjects) == 68

params = mnefun.read_params(
    "/home/ktavabi/Github/badbaby/badbaby/processing/oddball.yml"
)
params.n_jobs = "cuda"
params.n_jobs_fir = "cuda"
params.n_jobs_resample = "cuda"
params.proj_sfreq = 200.
params.decim = 2
params.acq_ssh = "kasga.ilabs.uw.edu"
params.acq_dir = ["/brainstudio/bad_baby"]

params.mf_prebad = {"default": ["MEG0743", "MEG1442"]}
params.mf_autobad = True
params.mf_autobad_type = "python"
params.coil_t_window = "auto"
params.coil_dist_limit = 0.01
params.coil_bad_count_duration_limit = 1.0  # sec
params.rotation_limit = 20.0  # deg/s
params.translation_limit = 0.01  # m/s
params.sss_type = "python"
params.hp_type = "python"
params.int_order = 6
params.ext_order = 3
params.tsss_dur = 90.0
params.st_correlation = 0.95
params.trans_to = "twa"
params.cont_as_esss = True
params.cont_hp = 20
params.cont_hp_trans = 2
params.cont_lp = 40
params.cont_lp_trans = 2
params.proj_sfreq = 200
params.proj_meg = "combined"
params.proj_ave = True
params.proj_nums = [
    [0, 0, 0],  # ECG: grad/meg/eeg
    [0, 0, 0],  # EOG  (combined saccade and blink events)
    [0, 0, 0],  # Continuous (from ERM)
    [0, 0, 0],  # HEOG (focus on saccades)
    [0, 0, 0],
]  # VEOG  (focus on blinks)


params.ecg_channel = ecg_channel
params.subjects = subjects
params.subject_indices = np.arange(len(params.subjects))
params.structurals = [None] * len(params.subjects)
params.score = score
params.dates = [None] * len(params.subjects)
params.work_dir = "/media/ktavabi/ALAYA/data/ilabs/badbaby"
# params.run_names = ["%s_mmn", "%s_am", "%s_ids"]
# params.runs_empty = ["%s_erm"]
params.subject_run_indices = [[0]] * len(params.subjects)
params.flat = dict(grad=1e-13)
params.auto_bad_flat = dict(grad=1e-13)
params.ssp_ecg_reject = dict(grad=np.inf, mag=np.inf)
params.ecg_t_lims = (-0.04, 0.04)
params.cov_method = "shrunk"
params.compute_rank = True
params.cov_rank = "full"
params.cov_rank_method = "compute_rank"
params.force_erm_cov_rank_full = False


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
            do_sss=default,
            fetch_sss=default,
            do_ch_fix=default,
            gen_ssp=default,
            apply_ssp=default,
            write_epochs=default,
            gen_covs=True,
            gen_fwd=default,
            gen_inv=default,
            gen_report=True,
            print_status=True,
        )
    except Exception:
        raise
        traceback.print_exc()
        bad.append(subject)
    else:
        good.append(subject)
print(f"Successfully processed {len(good)}/{len(good) + len(bad)}, bad:\n{bad}")
