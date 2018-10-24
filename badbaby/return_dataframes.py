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
    xl_meg = pd.read_excel(op.join(static_dir, 'badbaby.xlsx'),
                           sheet_name=paradigm,
                           converters={'BAD': str})
    xl_meg = xl_meg[(xl_meg.complete == 1)]
    xl_meg.drop(['Exam date', 'BAD', 'ECG', 'SR(Hz)', 'ACQ', 'MC-SVD',
                 'Artifact rej', 'Epochs'], axis=1, inplace=True)
    # Subselect by cohort
    if bezos:
        xl_meg = xl_meg[xl_meg['SES'] > 0]
    if simms:
        xl_meg = xl_meg[xl_meg['simms_inclusion'] == 1]
    if age == 2:
        #  Filter by age
        xl_meg = xl_meg[xl_meg['Age(days)'] < 80]
    elif age == 6:
        xl_meg = xl_meg[xl_meg['Age(days)'] > 80]

    xl_meg = xl_meg.drop('Notes', axis=1, inplace=False)
    xl_cdi = pd.read_excel(
        op.join(static_dir, 'cdi_report_final_08292018.xlsx'),
        sheet_name='Data')
    xl_cdi.drop(['DOB', 'Gender', 'Language', 'CDIForm',
                 'CDIAgeCp', 'CDIDate', 'VOCPER', 'HOWUSE', 'UPSTPER',
                 'UFUTPER', 'UMISPER', 'UCMPPER', 'UPOSPER', 'WORDEND',
                 'PLURPER',
                 'POSSPER', 'INGPER', 'EDPER', 'IRWORDS', 'IRWDPER', 'OGWORDS',
                 'COMBINE', 'COMBPER', 'CPLXPER'], axis=1, inplace=True)
    return xl_meg, xl_cdi
