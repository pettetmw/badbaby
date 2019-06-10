#!/usr/bin/env python

"""Helper to return cohort specific dataframes """

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2018, Seattle, Washington"
__credits__ = ["Goedel", "Escher", "Bach"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Development"

import os.path as op
import pandas as pd
from badbaby.python import defaults as params

static = params.static


def return_dataframes(paradigm, age=None, cdi=False,
                      ses=False, longitudinal=False):
    """
    Return available MEG and corresponding CDI datasets.
    Parameters
    ----------
    paradigm:str
        Name of project paradigm. Can be mmn, assr, or ids.
    age:int
        If not None (default) filter cohort data based on age.
    longitudinal:bool
        Default False. If True then include only Simms-Mann longitudinal cohort.
    cdi:bool
        Default False, If True then include if CDI data is available.
    ses:bool
        Default False. If True then include only Bezos SES cohort.

    Returns
    -------
    tuple
        Pandas dataframes of available MEG and corresponding CDI datasets.
    """
    # Read excel sheets into pandas dataframes
    xl_meg = pd.read_excel(op.join(static, 'meg_covariates.xlsx'),
                           sheet_name=paradigm,
                           converters={'BAD': str})
    xl_meg = xl_meg[(xl_meg.acq == 1)]  # only Ss w MEG Acq
    # Subselect by cohort
    if ses:
        xl_meg = xl_meg[xl_meg['ses'] > 0]
    if longitudinal:
        xl_meg = xl_meg[xl_meg['simmInclude'] == 1]
    if cdi:
        xl_meg = xl_meg[(xl_meg.behavioral == 1)]  # only Ss w CDI data
    #  Filter by age
    if age == 2:
        xl_meg = xl_meg[xl_meg['age'] < 100]
    elif age == 6:
        xl_meg = xl_meg[xl_meg['age'] > 150]

    xl_meg = xl_meg.drop('notes', axis=1, inplace=False)
    xl_cdi = pd.read_excel(
        op.join(static, 'behavioral_data.xlsx'),
        sheet_name='Data')
    xl_cdi.drop(['dob', 'gender', 'language', 'cdiForm',
                 'examDate', 'vocabper', 'howuse', 'upstper',
                 'ufutper', 'umisper', 'ucmpper', 'uposper', 'wordend',
                 'plurper',
                 'possper', 'ingper', 'edper', 'irwords', 'irwdper', 'ogwords',
                 'combine', 'combper', 'cplxper'], axis=1, inplace=True)
    return xl_meg, xl_cdi
