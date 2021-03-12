#!/usr/bin/env python
"""mnefun.py: MNEFUN preprocessing pipeline:

        1. Determine ACQ sampling rate and ECG channel.
        2. Write ACQ prebad channel to disk.
        3. Score.
        4. HP estimation with annotation parameters.
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

from score import score

static = op.join(Path(__file__).parents[1], "static")

columns = [
    "subjid",
    "badch",
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

bads = dict(
    (f"bad_{k}", [v]) for k, v in zip(meg_features["id"], meg_features["badch"])
)

lst = [[1, 2, 0], [0, 0, 0], [1, 1, 0]]  # ECG|EOG|ERM: grad/meg/eeg
pool = [lst[:] for _ in range(len(meg_features))]
proj_nums = dict((f"bad_{k}", v) for k, v in zip(meg_features["id"], pool))


good, bad = list(), list()
subjects = sorted(f"bad_{id_}" for id_ in meg_features["id"])
assert set(subjects) == set(ecg_channel)
assert len(subjects) == 68

params = mnefun.read_params(
    op.join(Path(__file__).parents[1], "processing", "badbaby.yml")
)
params.ecg_channel = ecg_channel
params.subjects = ["bad_101", "bad_102"]

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
            write_epochs=default,
            gen_covs=default,
            gen_fwd=default,
            gen_inv=default,
            gen_report=default,
            print_status=True,
        )
    except Exception:
        raise
        traceback.print_exc()
        bad.append(subject)
    else:
        good.append(subject)
print(f"Successfully processed {len(good)}/{len(good) + len(bad)}, bad:\n{bad}")
