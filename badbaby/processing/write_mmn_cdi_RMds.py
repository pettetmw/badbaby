#!/usr/bin/env python

"""write_mmn_cdi_RMds.py: write out mean MMN CV scores and CDI measures in
tidy format.
"""

import os.path as op
import re
import pandas as pd
import seaborn as sns
import xarray as xr
import itertools

import badbaby.return_dataframes as rd
from badbaby import defaults

# Parameters
datadir = defaults.datadir
analysese = ['Individual', 'Oddball']
ages = [2, 6]
lp = defaults.lowpass
regex = r"[abc]$"
solver = 'liblinear'

# Wrangle MEG & CDI covariates
meg, cdi = rd.return_dataframes('mmn', ses=True)
meg.reset_index(inplace=True)
meg['subject'] = ['bad_%s' % xx for xx in meg.subjId]
meg['subjId'] = [re.split(regex, ss)[0].upper() for ss in meg.subjId]
meg['subjId'] = ['BAD_%s' % xx for xx in meg.subjId]
s_ = meg.age < 80
meg['group'] = s_.map({True: '2mos', False: '6mos'})
covars = pd.merge(meg[['simmInclude', 'ses', 'age', 'gender', 'headSize',
                       'maternalEdu', 'maternalHscore',
                       'paternalEdu', 'paternalHscore',
                       'maternalEthno', 'paternalEthno', 'birthWeight',
                       'subjId', 'subject', 'group']], cdi, on='subjId',
                  validate='m:m')
g = sns.catplot(x='group', y='age', height=6, aspect=2, kind='box', data=covars)
# Wrangle MEG response vars
for iii, analysis in enumerate(analysese):
    print('Reading data for %s analysis... ' % analysis)
    if iii == 0:
        conditions = ['standard', 'ba', 'wa']
    else:
        conditions = ['standard', 'deviant']
    combos = list(itertools.combinations(conditions, 2))
    dfs = list()
    for age in ages:
        fi_in = op.join(defaults.datadir,
                        '%smos_%d_%s_%s-slidingEstimator.nc'
                        % (age, lp, solver, analysis))
        print('    File in %s...' % op.basename(fi_in))
        ds = xr.open_dataarray(fi_in)
        dfs.append(ds.to_dataframe(name='AUC').reset_index())
    # ROC Area under curve
    # df = pd.concat(dfs, axis=1, verify_integrity=True, ignore_index = True)
    df_ = dfs[0].merge(dfs[1], on=['contrast', 'subject'],
                       suffixes=('_2mos', '_6mos'), validate='m:m')
    df = df_.merge(covars, on='subject', validate='m:m')
    df.to_csv(op.join(datadir, 'auc_%d_%s_%s-RM.csv' % (lp, solver,
                                                        analysis)))
