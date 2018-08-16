# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import os.path as op
import numpy as np
import pandas as pd
from badbaby.parameters import meg_dirs

static_dir = op.join()
fig_dir = op.join(data_dir, 'figures')

def return_simms_mmn_df():
    # Read excel sheets into pandas dataframes
    xl_a = pd.read_excel(op.join(project_dir, 'badbaby.xlsx'), sheet_name='MMN',
                         converters={'BAD': str})
    xl_b = pd.read_excel(op.join(project_dir, 'badbaby.xlsx'),
                         sheet_name='simms_demographics')
    # Exclude subjects
    inclusion = xl_a['simms_inclusion'] == 1
    xl_a = xl_a[inclusion]
    subjects = pd.Series(np.intersect1d(xl_a['Subject_ID'].values,
                                        xl_b['Subject_ID'].values))
    # Find intersection between dataframes for common subjects
    xl_a = xl_a[xl_a['Subject_ID'].isin(subjects.tolist())]
    xl_b = xl_b[xl_b['Subject_ID'].isin(subjects.tolist())]
    simms_df = pd.merge(xl_a, xl_b)
    groups = np.unique(simms_df.Group).tolist()
    remap = dict([(2, '2 months'), (6, '6 months')])
    titles_comb = [remap[kind] for kind in [2, 6]]
    subjects = np.unique(np.asarray([x[:3] for x in simms_df.Subject_ID.values]))
    subjects = ['BAD_%s' % subj for subj in subjects]

    dfs = list()
    xl_c = pd.read_excel(op.join(project_dir, 'cdi_report_July_2018.xlsx'),
                         sheet_name='WS')
    for age in np.unique(xl_c.CDIAge.values):
        _, _, mask = np.intersect1d(np.asarray(subjects),
                                    xl_c[xl_c.CDIAge == age]
                                    ['ParticipantId'].values,
                                    return_indices=True)
        dfs.append(xl_c[xl_c.CDIAge == age].iloc[mask])
    return pd.concat(dfs, ignore_index=True)

