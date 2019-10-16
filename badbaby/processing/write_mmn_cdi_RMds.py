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
regex = r"[abc]$"

meg, cdi = rd.return_dataframes('mmn', ses=True)
ids = [re.split(regex, ss)[0] for ss in meg.index]
meg.reset_index(inplace=True)
meg['subjId'] = ['BAD_%s' % xx for xx in ids]

results = pd.merge(meg[['ses', 'age', 'gender', 'headSize', 'maternalEdu',
                        'maternalHscore', 'paternalEdu', 'paternalHscore',
                        'maternalEthno', 'paternalEthno', 'birthWeight',
                        'subjId']], cdi, on='subjId', validate='m:m')
dfs = list()
for age in ages:
    df_ = pd.read_hdf(op.join(datadir, '%dmos_%d_cv-scores.h5' % (age, lp)))
    df_['ag'] = np.ones((len(df_))) * age
    dfs.append(df_)

df = pd.concat(dfs, sort=False)
df['subjId'] = [re.split(regex, ss)[0].upper() for ss in df.index.values]
results = pd.merge(df, results, on='subjId', validate='m:m')
results.to_csv(op.join(datadir, 'mmn-%d_cdi_RMds.csv' % lp))
