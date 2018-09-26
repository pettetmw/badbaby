# -*- coding: utf-8 -*-

"""Helpers to return cohort specific dataframes """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import os.path as op
import numpy as np
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
    xl_a = pd.read_excel(op.join(static_dir, 'badbaby.xlsx'),
                         sheet_name=paradigm,
                         converters={'BAD': str})
    xl_a = xl_a[(xl_a.complete == 1) & (xl_a.CDI == 1)]
    # Subselect by cohort
    if bezos:
        xl_a = xl_a[xl_a['SES'] > 0]
    elif simms:
        xl_a = xl_a[xl_a['simms_inclusion'] == 1]
    else:
        #  Filter by age
        if age == 2:
            xl_a = xl_a[xl_a['Age(days)'] < 80]
        elif age == 6:
            xl_a = xl_a[xl_a['Age(days)'] > 80]
    xl_a = xl_a.drop('Notes', axis=1, inplace=False)
    xl_a.dropna()
    df = xl_a
    subject_ids = df.Subject_ID.values
    subject_ids = set(['BAD_%s' % ss[:3] for ss in subject_ids.tolist()])
    xl_b = pd.read_excel(op.join(static_dir, 'cdi_report_final_08292018.xlsx'),
                         sheet_name='Data')
    participant_ids = np.unique(xl_b.ParticipantId.values)
    # Exclude subjects without CDI data
    out = np.intersect1d(np.asarray(list(subject_ids)),
                         participant_ids, return_indices=True)
    ma_union, ma_subject_ids, ma_participant_ids = out
    subject_ids = df.Subject_ID.values[ma_subject_ids]
    participant_ids = participant_ids[ma_participant_ids]
    assert subject_ids.shape == participant_ids.shape
    dfs = list()
    ages = np.unique(xl_b.CDIAge.values)
    for age in ages:
        _, _, mask = np.intersect1d(participant_ids,
                                    xl_b[xl_b.CDIAge == age]
                                    ['ParticipantId'].values,
                                    return_indices=True)
        dfs.append(xl_b[xl_b.CDIAge == age].iloc[mask])
    cdi_df = pd.concat(dfs, ignore_index=True)
    return df, cdi_df
