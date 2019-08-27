#!/usr/bin/env python

"""Plot descriptives for demographic and covariates."""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__credits__ = ["Goedel", "Escher", "Bach"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Production"

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.graphics.api as smg
from scipy import stats
from sklearn import linear_model

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

analysis = 'oddball'
conditions = ['standard', 'deviant']
tmin, tmax = defaults.epoching
lp = defaults.lowpass
window = defaults.peak_window  # peak ERF latency window
workdir = defaults.datapath
ages = [2, 6]
meg_dependents = ['auc', 'latencies', 'naves']

for age in ages:
    mmn_df, cdi_df = rd.return_dataframes('mmn', age=age, ses=True)
    # assert cdi_df.subjId.unique().shape[0] == 71
    # assert mmn_df.shape[0] == 25
    # merge MEG & CDI frames
    re.split('_|-|', ll)
    mmn_df['merge_on'] = ['BAD_%s' % re.findall(r'\d+', ll)[0] for ll in
                          mmn_df.index]
    df = cdi_df.merge(mmn_df, left_on='subjId',
                      right_on='merge_on', sort=True, validate='m:1')
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
