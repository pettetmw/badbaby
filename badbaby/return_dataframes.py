# -*- coding: utf-8 -*-

"""Helpers to return cohort specific dataframes """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import os.path as op
import pandas as pd
from badbaby import defaults as params

static_dir = params.static_dir


def return_dataframes(paradigm, age=None, bezos=False, simms=False):
    """
    Return available MEG and corresponding CDI datasets.
    Parameters
    ----------
    paradigm:str
    Name of project paradigm. Can be mmn, tone, or ids.
    age:int
    If not None (default) filter cohort data based on age.
    simms:bool
    Default False. If True then include only Simms-Mann longitudinal cohort.
    bezos:bool
    Default False. If True then include only Bezos SES cohort.

    Returns
    -------
    tuple
        Pandas dataframes of available MEG and corresponding CDI datasets.
    """
    # Read excel sheets into pandas dataframes
    xl_meg = pd.read_excel(op.join(static_dir, 'badbaby_final_121818.xlsx'),
                           sheet_name=paradigm,
                           converters={'BAD': str})
    xl_meg = xl_meg[(xl_meg.complete == 1)]
    xl_meg.drop(['examDate', 'acq', 'sss',
                 'rejection', 'epoching'], axis=1, inplace=True)
    # Subselect by cohort
    if bezos:
        xl_meg = xl_meg[xl_meg['ses'] > 0]
    if simms:
        xl_meg = xl_meg[xl_meg['simmInclude'] == 1]
    if age == 2:
        #  Filter by age
        xl_meg = xl_meg[xl_meg['age'] < 80]
    elif age == 6:
        xl_meg = xl_meg[xl_meg['age'] > 80]

    xl_meg = xl_meg.drop('notes', axis=1, inplace=False)
    xl_cdi = pd.read_excel(
        op.join(static_dir, 'cdi_report_final_08292018.xlsx'),
        sheet_name='Data')
    xl_cdi.drop(['dob', 'gender', 'language', 'cdiForm',
                 'examDate', 'vocabper', 'howuse', 'upstper',
                 'ufutper', 'umisper', 'ucmpper', 'uposper', 'wordend',
                 'plurper',
                 'possper', 'ingper', 'edper', 'irwords', 'irwdper', 'ogwords',
                 'combine', 'combper', 'cplxper'], axis=1, inplace=True)
    return xl_meg, xl_cdi
