#!/usr/bin/env python

"""write_mmn_cdi_RMds.py: write out MEG & CDI measurement in tidy-format.
"""

import itertools
import os.path as op
import re

import pandas as pd
import xarray as xr

import seaborn as sns
from badbaby import defaults
from badbaby import return_dataframes as rd

# Parameters
datadir = defaults.datadir
analysese = ['Individual', 'Oddball']
ages = [2, 6]
lp = defaults.lowpass
regex = r"[abc]$"
solver = 'lbfgs'

# covariates
meg, cdi = rd.return_dataframes('mmn', ses=True)
meg.reset_index(inplace=True)
meg.rename(columns={'subjId': 'sid'}, inplace=True)
meg['subjId'] = ['BAD_%s' % xx for xx in [
    re.split(regex, ss)[0] for ss in meg.sid]]
covs = pd.merge(meg[['ses', 'age', 'gender', 'headSize',
                     'maternalEdu', 'maternalHscore',
                     'paternalEdu', 'paternalHscore',
                     'maternalEthno', 'paternalEthno', 'birthWeight',
                     'subjId', 'sid']], cdi, on='subjId',
                validate='m:m')
covs['subject'] = ['bad_%s' % xx for xx in covs.sid]
covs.info()

# Wrangle MEG CDI vars
dfs = list()
for iii, analysis in enumerate(analysese):
    print('Reading data for %s analysis... ' % analysis)
    if iii == 0:
        conditions = ['standard', 'ba', 'wa']
    else:
        conditions = ['standard', 'deviant']
    combos = list(itertools.combinations(conditions, 2))
    fi_in = op.join(defaults.datadir,
                    'AUC_%d_%s_%s.nc'
                    % (lp, solver, analysis))
    ds = xr.open_dataarray(fi_in)
    dfs.append(ds.to_dataframe(name='AUC').reset_index())
df = pd.concat(dfs, axis=0, verify_integrity=True, ignore_index=True)
Ds = df.merge(covs, on='subject', validate='m:m')
Ds.info()
Ds.to_csv(op.join(datadir, 'cdi-meg_%d_%s_dataset.csv' % (lp, solver)))

