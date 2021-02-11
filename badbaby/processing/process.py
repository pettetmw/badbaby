#!/usr/bin/env python
"""MNEFUN processing pipeline"""

import os.path as op
import traceback
from pathlib import Path

import janitor  # noqa
import mnefun
import pandas as pd

from score import score

static = op.join(Path(__file__).parents[1], "static")
datadir = op.join(Path(__file__).parents[1], "data")
tabdir = op.join(Path(__file__).parents[1], "data", "tabdir")
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
    .filter_on("behavioral == 1", complement=False)
    .filter_on("complete == 1", complement=False)
)

ecg_channel = dict(
    (f"bad_{k}", v) for k, v in zip(meg_features["id"], meg_features["ecg"])
)

good, bad = list(), list()
subjects = sorted(f"bad_{id_}" for id_ in meg_features["id"])
assert set(subjects) == set(ecg_channel)
assert len(subjects) == 76
subjects.pop(subjects.index("bad_223a"))  # cHPI is no good

params = mnefun.read_params("badbaby/processing/params.yml")
params.ecg_channel = ecg_channel
params.subjects = subjects[0:2]
params.structurals = [None] * len(params.subjects)
params.score = score
params.dates = [None] * len(params.subjects)
params.work_dir = datadir

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
            do_score=True,
            push_raw=default,
            do_sss=default,
            fetch_sss=default,
            do_ch_fix=default,
            gen_ssp=default,
            apply_ssp=default,
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
