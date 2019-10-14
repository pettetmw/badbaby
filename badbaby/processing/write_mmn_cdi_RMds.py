#!/usr/bin/env python

"""write_mmn_cdi_RMds.py: write out mean MMN CV scores and CDI measures in
tidy format.
"""

import os.path as op
import re

import numpy as np
import pandas as pd

import badbaby.return_dataframes as rd
from badbaby import defaults

datadir = defaults.datadir

ages = [2, 6]
lp = defaults.lowpass

cdi_df = rd.return_dataframes('mmn', ses=True)[1]
dfs = list()
for age in ages:
    df_ = pd.read_hdf(op.join(datadir, '%dmos_%d_cv-scores.h5' % (age, lp)))
    df_['ag'] = np.ones((len(df_))) * age
    dfs.append(df_)

df = pd.concat(dfs, sort=False)

regex = r"[abc]$"
df['subjId'] = [re.split(regex, ss)[0].upper() for ss in df.index.values]
results = pd.merge(df, cdi_df, on='subjId')
results.to_csv(op.join(datadir, 'mmn-%d_cdi_RMds.csv' % lp))
