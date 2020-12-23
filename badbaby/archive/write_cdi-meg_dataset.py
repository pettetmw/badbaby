#!/usr/bin/env python

"""write_mmn_cdi_RMds.py: write out MEG & CDI measurement in tidy-format.
"""

import datetime
import itertools
import os.path as op
import re

import numpy as np
import pandas as pd
import xarray as xr

from badbaby import defaults
from badbaby import return_dataframes as rd

# Parameters
datadir = defaults.datadir
date = datetime.datetime.today()
date = '{:%m%d%Y}'.format(date)
analysese = ['Individual', 'Oddball']
ages = [2, 6]
lp = defaults.lowpass
regex = r"[0-9]+"
solver = 'lbfgs'


# covariates
meg, cdi = rd.return_dataframes('mmn', ses=True)
meg.reset_index(inplace=True)
meg['cdiId'] = ['BAD_%s' % xx for xx in [
    re.findall(regex, ss)[0] for ss in meg.subjId]]
cdi.rename(columns={'subjId': 'cdiId'}, inplace=True)
covs = pd.merge(meg[['ses', 'age', 'gender', 'headSize',
                     'maternalEdu', 'maternalHscore',
                     'paternalEdu', 'paternalHscore',
                     'maternalEthno', 'paternalEthno', 'birthWeight',
                     'subjId', 'cdiId']], cdi, on='cdiId',
                validate='m:m')
covs['megId'] = ['bad_%s' % xx for xx in covs.subjId]
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
df.rename(columns={'subject': 'megId'}, inplace=True)
Ds = df.merge(covs, on='megId', validate='m:m')
mapping = {'standard-ba': 'plosive',
           'standard-wa': 'aspirative',
           'standard-deviant': 'mmn',
           'ba-wa': 'deviant'}
Ds.replace(mapping, inplace=True)
Ds['vocab-asin'] = np.arcsin(np.sqrt(Ds.vocab.values/Ds.vocab.values.max()))  # noqa
Ds['m3l-asin'] = np.arcsin(np.sqrt(Ds.m3l.values/Ds.m3l.values.max()))
Ds.info()
Ds.to_csv(op.join(datadir, 'cdi-meg_%d_%s_tidy_%s.csv' % (lp, solver, date)))
