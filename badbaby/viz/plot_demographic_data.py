#!/usr/bin/env python

"""plot_demographic_data.py: EDA gists for demographic descriptives."""

import itertools
import os.path as op
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.graphics.api as smg
from mne.externals.h5io import read_hdf5
from scipy import stats

import badbaby.return_dataframes as rd
from badbaby import defaults


def plot_correlation_matrix(data):
    """Statsmodel correlation plot helper
    Params:
        data: pandas frame of pairwise correlation of columns in original data.
    """
    return smg.plot_corr(data.values, xnames=data.columns)


pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
workdir = defaults.datapath
plt.style.use('ggplot')
ages = [2, 6]
solver = 'liblinear'
tag = '2v'
if tag is '2v':
    combos = list(itertools.combinations(['standard', 'deviant'], 2))
else:
    combos = list(itertools.combinations(defaults.oddball_stimuli, 2))
lp = defaults.lowpass

rm_sample = list()
for aix, age in enumerate([2, 6]):
    hf_fname = op.join(defaults.datadir,
                       '%smos_%d_%s_%s-cv-scores.h5' % (age, lp, solver, tag))
    subjs = read_hdf5(hf_fname, title=solver)['subjects']
    rm_sample.append([re.findall(r'\d+', idx)[0] for idx in subjs])
rm_sample = list(set(rm_sample[0]).intersection(rm_sample[1]))
rm_sample = ['BAD_%s' % idx for idx in rm_sample]

df_x, df_y = rd.return_dataframes('mmn', ses=True)
df_x.reset_index(inplace=True)
ids_x = [re.findall(r'\d+', idx)[0] for idx in df_x['subjId']]
df_x['subjId'] = ['BAD_%s' % idx for idx in ids_x]
df_x.drop(['badCh', 'ecg', 'samplingRate', 'sib1dob', 'sib1gender',
           'sib2dob', 'sib2gender', 'sib3dob', 'sib3gender',
           'birthWeight(lbs)'],
          axis=1, inplace=True)
for col in ['gender', 'maternalEdu', 'maternalHscore',
            'paternalEdu', 'paternalHscore',
            'maternalEthno', 'paternalEthno']:
    df_x[col] = df_x[col].astype('category')
split = (df_x.age.max() - df_x.age.min()) // 2
df_x['group'] = np.where(df_x['age'] < split, 'Two', 'Six')
df_x.info()
df_rm = df_x[df_x.subjId.isin(rm_sample)]
df_rm_grpby = df_rm[['group', 'gender', 'age',
                     'headSize', 'birthWeight',
                     'nSibs']].groupby(['group', 'gender'])
df_rm_grpby.describe()
df_rm_grpby.describe().to_csv(op.join(defaults.datadir,
                                      'RM-sample_demographic_desc.csv'))
df = df_x.merge(df_y, on='subjId', sort=True, validate='m:m')




sns.boxplot(x="contrast", y="auc", hue="group", notch=True, data=df)
sns.despine(left=True)

    # Split data on SES
    sesGrouping = df.ses <= df.ses.median()  # low ses True
    df['sesGroup'] = sesGrouping.map({True: 'low', False: 'high'})
    # Pairwise plots
    g = sns.pairplot(df, vars=['m3l', 'vocab'], diag_kind='kde',
                     hue='cdiAge', palette='tab20')
    plot_correlation_matrix(df[['cdiAge', 'm3l', 'vocab']].corr())
    # Gender
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    df.complete.groupby(df.gender).sum().plot. \
        pie(autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.35,
            subplots=False, ax=ax)
    ax.set(title='Gender', ylabel='MEG-CDI data acquired')
    fig.tight_layout()
    # Parental ethnicities
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    df.complete.groupby(df.maternalEthno).sum().plot. \
        pie(autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.35,
            subplots=False, ax=ax)
    ax.set(title='Mom Ethnicity')
    fig.tight_layout()
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    df.complete.groupby(df.paternalEthno).sum().plot. \
        pie(autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.35,
            subplots=False, ax=ax)
    ax.set(title='Dad Ethnicity')
    fig.tight_layout()
    # Head circumference
    scatter_kws = dict(s=50, linewidth=.5, edgecolor="w")
    g = sns.FacetGrid(df, hue="sesGroup",
                      hue_kws=dict(marker=["v", "^"]))
    g.map(plt.scatter, "age", "headSize",
          **scatter_kws).add_legend(title='SES')
    # CDI measures
    for nm, title in zip(['m3l', 'vocab'],
                         ['Mean length of utterance', 'Words understood']):
        g = sns.lmplot(x="cdiAge", y=nm, truncate=True, data=df)
        g.set_axis_labels("Age (months)", nm)
        g.ax.set(title=title)
        g.despine(offset=2, trim=True)
        h = sns.catplot(x='cdiAge', y=nm, hue='sesGroup',
                        data=df[df.cdiAge > 18],
                        kind='violin', scale_hue=True, bw=.2, linewidth=1,
                        scale='count', split=True, dodge=True, inner='quartile',
                        palette=sns.color_palette('pastel', n_colors=2,
                                                  desat=.5),
                        margin_titles=True, legend=False)
        h.add_legend(title='SES')
        h.fig.suptitle(title)
        h.despine(offset=2, trim=True)
    # Linear regression fit between CDI measures and SES scores
    cdi_ages = np.arange(21, 31, 3)
    for nm, tt in zip(['m3l', 'vocab'],
                      ['Mean length of utterance', 'Words understood']):
        fig, axs = plt.subplots(1, len(cdi_ages), figsize=(12, 6))
        hs = list()
        for fi, ax in enumerate(axs):
            ma_ = df.cdiAge == cdi_ages[fi]
            # response
            y_vals = np.squeeze(df[ma_][nm].values.reshape(-1, 1))
            # predictor
            x_vals = np.squeeze(df[ma_].ses.values.reshape(-1, 1))
            assert (y_vals.shape == x_vals.shape)
            hs.append(ax.scatter(x_vals, y_vals, c='CornFlowerBlue', s=50,
                                 zorder=5, marker='.', alpha=0.5))
            ax.set(xlabel='SES', title='%d Mos' % cdi_ages[fi])
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(x_vals, y_vals)
            hs.append(
                ax.plot(x_vals, intercept + slope * x_vals,
                        label='LS fit',
                        c="Grey", lw=2, alpha=0.5))
            ax.annotate('$\mathrm{r^{2}}=%.2f$''\n$\mathit{p = %.2f}$'
                        % (r_value ** 2, p_value),
                        xy=(0.05, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle='square', fc='w'))
            if fi == 0:
                ax.set(ylabel=tt)
                ax.legend(loc=1, frameon=False)
