#!/usr/bin/env python

"""wrangle_CDI-AUC.py: write tidy formatted data as CSV file.
    Does:
        1.
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"

import os.path as op
import pandas as pd
import badbaby.return_dataframes as rd
from badbaby import defaults

datadir = defaults.datadir

ages = [2, 6]
lp = defaults.lowpass
condition1, condition2 = 'standard', 'deviant'
window = defaults.peak_window

cdi_df = rd.return_dataframes('mmn', ses=True)[1]
dfs = list()
for age in ages:
    dfs.append(pd.read_hdf(op.join(datadir, 'auc_%d_%dmos' % (lp, age))))

df = pd.concat(dfs, axis=1)
