#!/usr/bin/env python

"""Auditory oddball paradigm parameters."""

import os.path as op
from pathlib import Path

import janitor  # noqa
import pandas as pd

static = op.join(Path(__file__).parents[0], "static")
figsdir = op.join(Path(__file__).parents[0], "writeup", "results", "figures")
datadir = "/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch"
run_name = "mmn"
epoching = (-0.1, 0.6)
lowpass = 55.0
highpass = 0.1
peak_window = (0.235, 0.53)
oddball_stimuli = ["standard", "ba", "wa"]
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
        ]
df = (
    pd.read_excel(op.join(
        static,
        "meg_covariates.xlsx"),
        sheet_name="mmn"
    )
    .clean_names()
    .select_columns(columns)
    .encode_categorical(columns=columns)
    .rename_columns({"subjid": "id"})
    .filter_on('behavioral == 1',
               complement=False)
    .filter_on('complete == 1',
               complement=False))

picks = ['bad_%s' % pick for pick in df["id"]]
cohort_six = df.filter_on('age > 150', complement=False)
