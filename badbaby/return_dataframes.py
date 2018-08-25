# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import os.path as op
import numpy as np
import pandas as pd
from badbaby.parameters import project_dir

static_dir = op.join(project_dir, 'badbaby', 'static')


def return_simms_mmn_dfs():
    """
       Return MMN exam and CDI dataframes for Simms-Mann cohort
       Returns
       -------
       tuple
           dataframe of selected cases from subject sample with
           completed MMN MEG data and dataframe of corresponding CDI data.
       """
    # Read excel sheets into pandas dataframes
    xl_a = pd.read_excel(op.join(static_dir, 'badbaby.xlsx'), sheet_name='MMN',
                         converters={'BAD': str})
    xl_b = pd.read_excel(op.join(static_dir, 'cdi_report_July_2018.xlsx'),
                         sheet_name='WS')
    # Exclude subjects
    df = xl_a[xl_a['simms_inclusion'] == 1]
    df = df.drop('Notes', axis=1)
    df.dropna()
    subjects = df['Subject_ID'].values.tolist()
    subjects = np.unique(np.asarray(['BAD_%s' % ss[:3] for ss in subjects]))
    dfs = list()
    ages = np.unique(xl_b.CDIAge.values)
    for age in ages:
        _, _, mask = np.intersect1d(subjects,
                                    xl_b[xl_b.CDIAge == age]
                                    ['ParticipantId'].values,
                                    return_indices=True)
        dfs.append(xl_b[xl_b.CDIAge == age].iloc[mask])
    cdi_df = pd.concat(dfs, ignore_index=True)
    return df, cdi_df


def return_ford_mmn_dfs():
    """
    Return MMN exam and CDI dataframes for Ford cohort
    Returns
    -------
    tuple
        dataframe of selected cases from subject sample with
        completed MMN MEG data and dataframe of corresponding CDI data.
    """
    # Read excel sheets into pandas dataframes
    xl_a = pd.read_excel(op.join(static_dir, 'badbaby.xlsx'), sheet_name='MMN',
                         converters={'BAD': str})
    xl_a = xl_a[(xl_a.complete == 1) & (xl_a.CDI == 1) & (xl_a.SES > 0)]
    xl_a = xl_a.drop('Notes', axis=1, inplace=False)
    xl_a.dropna()
    df = xl_a
    subject_ids = df.Subject_ID.values
    subject_ids = set(['BAD_%s' % ss[:3] for ss in subject_ids.tolist()])
    xl_b = pd.read_excel(op.join(static_dir, 'cdi_report_July_2018.xlsx'),
                         sheet_name='WS')
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


def return_mmn_df():
    """
    Return MMN exam dataframe
    Returns
    -------
    dataframe
        dataframe of selected cases from subject sample with
        completed MMN MEG data.
    """
    # Read excel sheets into pandas dataframes
    xl_a = pd.read_excel(op.join(static_dir, 'badbaby.xlsx'), sheet_name='MMN',
                         converters={'BAD': str})
    xl_a.drop('Notes', axis=1, inplace=True).dropna()
    # Exclude subjects
    return xl_a[xl_a.complete == 1]
