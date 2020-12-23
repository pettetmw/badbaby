#!/usr/bin/env python

"""compute_ses_regression: OLS regression fit between SES and deviant ERF.
    Description:
        1. procedure
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"

import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from meeg_preprocessing import config
from meeg_preprocessing.utils import combine_events
from mne import read_epochs
from mne.stats import linear_regression, fdr_correction
from badbaby import return_dataframes as rd, defaults

plt.style.use('ggplot')

# parameters
workdir = defaults.datapath
analysis = 'Oddball'
conditions = ['standard', 'deviant']
tmin, tmax = defaults.epoching
lp = defaults.lowpass
ages = [2, 6]
window = defaults.peak_window  # peak ERF latency window
names = ['Intercept', 'SES']

for aix in ages:
    df = rd.return_dataframes('mmn', age=aix, ses=True)[0]
    subjects = ['bad_%s' % ss for ss in df.index.values]
    print(df.info())
    for ii, cond in enumerate(conditions):
        for jj, subject in enumerate(subjects):
            print('     Subject: %s' % subject)
            ep_fname = op.join(workdir, subject, 'epochs',
                               'All_%d-sss_%s-epo.fif' % (lp, subject))
            # load trial data
            eps = read_epochs(ep_fname)
            eps.apply_baseline()
            # Combine trials into deviant condition
            eps = combine_events(eps, ['ba', 'wa'], {'deviant': 23})
            eps.equalize_event_counts(eps.event_id.keys())
            eps.drop_bad()
            deviants = eps['deviant']
            n_obs = len(deviants)
            intercept = np.ones((n_obs, 1))
            # Design matrix for continues SES covariate
            regressor = np.linspace(8, 66, n_obs)
            dmat = np.concatenate((regressor[:, np.newaxis],
                                   intercept), axis=1)
            # regression between deviant ERF & SES
            res = linear_regression(deviants, design_matrix=dmat, names=names)
            # plot beta coeffs
            for nm in names:
                res[nm].beta.plot_joint(title=nm,
                                        ts_args=config.TS_ARGS,
                                        topomap_args=config.TOPOMAP_ARGS)
            # plot stats
            reject_H0, fdr_pvals = fdr_correction(
                res['SES'].p_val.data)
            evoked = res['SES'].beta
            evoked.plot_image(mask=reject_H0, time_unit='s')
