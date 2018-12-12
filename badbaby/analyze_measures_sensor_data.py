# -*- coding: utf-8 -*-

"""Write dependent measures for oddball sensor space data"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import statsmodels.graphics.api as smg

import badbaby.defaults as params
import badbaby.return_dataframes as rd


def plot_correlation_matrix(data):
	"""Helper to return statsmodel correlation plot
	Parameters:
		data: pandas frame of pairwise correlation of columns in original data.
	"""
	return smg.plot_corr(data.values, xnames=data.columns)


def fit_linear_reg(feature, response):
    """Helper to fit linear model with sklearn
    Notes:
        As implemnted here this routine yields same results as
        scipy.stats.linregress
        Calculation of r-score stats on StackExchange
        https://tinyurl.com/yapl82m3
    """
    prediction_space = np.linspace(
        min(feature), max(feature), num=feature.shape[0]).reshape(-1, 1)
    reg = linear_model.LinearRegression().fit(feature, response)
    y_pred = reg.predict(prediction_space)
    r_val = reg.score(feature, response)  # equivalent to Pearson r-value
    n = feature.shape[0]
    t_val = np.sqrt(r_val) / np.sqrt((1 - r_val) / (n - 2))
    p_val = stats.t.sf(np.abs(t_val), n - 1) * 2  # Two-sided pvalue as Prob(
    # abs(t)>tt) using stats Survival Function (1-CDF — sometimes more
    # accurate))
    return reg, y_pred, prediction_space, r_val, t_val, p_val


# Some parameters
analysis = 'Individual-matched'
# encoding grouping variables
cond_keys = ['standard', 'Ba', 'Wa']
cond_vals = LabelEncoder().fit_transform(cond_keys) + 1
conditions = dict(zip(cond_keys, cond_vals))
ch_keys = ['grad', 'mag']
ch_vals = LabelEncoder().fit_transform(cond_keys) + 1
ch_types = dict(zip(ch_keys, ch_vals))
hem_keys = ['lh', 'rh']
hem_vals = LabelEncoder().fit_transform(hem_keys) + 1
hems = dict(zip(hem_keys, hem_vals))

lpf = 30
age = 2
meg_dependents = ['auc', 'latencies', 'naves']
data_dir = params.meg_dirs['mmn']

fname = op.join(params.static_dir, '%s_%s-mos_%d_measures.npz'
                % (analysis, age, lpf))
if not op.isfile(fname):
	raise RuntimeError('%s not found.' % fname)
data = np.load(fname)
mmn_xls, cdi_xls = rd.return_dataframes('mmn', age=age, bezos=True)

# Merge MMN features and CDI data
subj_ids = ['BAD_%s' % ss.split('a')[0]
            for ss in mmn_xls.Subject_ID.values]
mmn_xls.insert(len(mmn_xls.columns), 'ParticipantId', subj_ids)
mmn_cdi_df = pd.merge(cdi_xls, mmn_xls, on='ParticipantId', how='inner',
                      sort=True, validate='m:1').reindex()
# Split data on SES and CDIAge
ses_grouping = mmn_cdi_df.SES <= mmn_xls.SES.median()  # low SES True
mmn_cdi_df['ses_group'] = ses_grouping.map({True: 'low', False: 'high'})
mmn_cdi_df.drop(axis=1, columns=['BAD', 'ECG', 'SR(Hz)', 'complete', 'CDI',
                                 'simms_inclusion']).to_csv(
	op.join(params.static_dir, 'CDIdf_RM.csv'), sep='\t')
print('\nDescriptive stats for Age(days) variable...\n',
      mmn_cdi_df['Age(days)'].describe())

# Write out CSV for analysis in R
mmn_cdi_df.to_csv(op.join(params.static_dir, 'mmn-cdi_df.csv'), sep='\t')

# Plots
# Pairwise + density, and correlation matrix of CDI response variables
g = sns.pairplot(mmn_cdi_df, vars=['M3L', 'VOCAB'], diag_kind='kde',
                 hue='CDIAge', palette='tab20')
plot_correlation_matrix(mmn_cdi_df[['CDIAge', 'M3L', 'VOCAB']].corr())

# pie chart of gender
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
mmn_cdi_df.complete.groupby(mmn_cdi_df.Sex).sum().plot.pie(subplots=False,
                                                           ax=ax)
ax.set(title='Sex', ylabel='MEG-CDI data acquired')
fig.tight_layout()

# scatter head circumference vs. age
scatter_kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(mmn_cdi_df, hue="ses_group",
                  hue_kws=dict(marker=["v", "^"]))
g.map(plt.scatter, "Age(days)", "HC", **scatter_kws).add_legend(title='SES')

# CDI measure vs age regression & SES group within age distribution plots
for nm, title in zip(['M3L', 'VOCAB'],
                     ['Mean length of utterance', 'Words understood']):
	g = sns.lmplot(x="CDIAge", y=nm, truncate=True, data=mmn_cdi_df)
	g.set_axis_labels("Age (months)", nm)
	g.ax.set(title=title)
	g.despine(offset=2, trim=True)
	h = sns.catplot(x='CDIAge', y=nm, hue='ses_group',
	                data=mmn_cdi_df[mmn_cdi_df.CDIAge > 18],
	                kind='violin', scale_hue=True, bw=.2, linewidth=1,
	                scale='count', split=True, dodge=True, inner='quartile',
	                palette=sns.color_palette('pastel', n_colors=2, desat=.5),
	                margin_titles=True, legend=False)
	h.add_legend(title='SES')
	h.fig.suptitle(title)
	h.despine(offset=2, trim=True)

# Linear regression model fit between CDI measures and SES scores
ages = np.arange(21, 31, 3)
for nm, tt in zip(['M3L', 'VOCAB'],
                  ['Mean length of utterance', 'Words understood']):
    fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
    hs = list()
    for fi, ax in enumerate(axs):
        # response variable
        mask_y = mmn_cdi_df.CDIAge == ages[fi]
        y_vals = np.squeeze(mmn_cdi_df[mask_y][nm].values.reshape(-1, 1))
        mask_x = mmn_xls.Subject_ID.isin(
            mmn_cdi_df[mmn_cdi_df.CDIAge == ages[fi]].Subject_ID)
        # predictor
        x_vals = np.squeeze(mmn_xls[mask_x].SES.values.reshape(-1, 1))
        assert (y_vals.shape == x_vals.shape)
        hs.append(ax.scatter(x_vals, y_vals, c='CornFlowerBlue', s=50,
                             zorder=5, marker='.', alpha=0.5))
        ax.set(xlabel='SES', title='%d Mos' % ages[fi])
        if fi == 0:
            ax.set(ylabel=tt)
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(x_vals, y_vals)
        hs.append(ax.plot(x_vals, intercept + slope * x_vals, label='fitted '
                                                                    'line',
                          c="Grey", lw=2, alpha=0.5))
        ax.annotate('$\mathrm{r^{2}}=%.2f$''\n$\mathit{p = %.2f}$'
                    % (r_value ** 2, p_value),
                    xy=(0.05, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle='square', fc='w'))

# Create MEG measures dataframe
sz = data['auc'].size // 2
# interleave list --> tiled vector of levels for factors:
# subjs x conds x sensors x hemisphere
subjects = mmn_xls.Subject_ID.values
subj_ids = np.vstack((subjects, subjects) *
                     (sz // len(subjects))).reshape((-1,), order='F')
c_levels = np.vstack((list(conditions.values()), list(conditions.values())) *
                     len(ch_types)).reshape((-1,), order='F')
sns_levels = np.vstack((list(ch_types.values()),
                        list(ch_types.values()))).reshape((-1,), order='F')
hem_levels = np.vstack(list(hems.values())).reshape((-1,), order='F')
meg_df = pd.DataFrame({
    'Subject_ID': subj_ids.tolist(),
    'conditions': c_levels.tolist() * len(subjects),
    'ch_type': sns_levels.tolist() * (sz // 2),
    'hemisphere': hem_levels.tolist() * sz,
    'auc': (data['auc'].reshape(-1, order='C')),
    'latencies': data['latencies'].reshape(-1,
                                           order='C'),
    'channels': data['channels'].reshape(-1, order='C')
})
naves = np.transpose(data['naves'], (1, 0))
subj_ids = np.vstack([subjects] * len(conditions)).reshape((-1,), order='F')
c_levels = np.vstack(list(conditions.values())).reshape((-1,), order='F')

# Inspect number of averages for each stimulus condition
erf_naves = pd.DataFrame({
    'Subject_ID': subj_ids.tolist(),
    'conditions': c_levels.tolist() * len(subjects),
    'naves': data['naves'].reshape(-1, order='C')
})
erf_naves.replace({'conditions': {vv: kk for kk, vv in conditions.items()}},
                  inplace=True)
sns.catplot(x='conditions', y='naves', kind='swarm', palette='tab20',
            data=erf_naves)

#  Merge MEG with MMN xls dataframes
mmn_df = meg_df.merge(mmn_xls, on='Subject_ID', validate='m:1')
# Name grouping variables
ses_grouping = mmn_df.SES <= mmn_df.SES.median()
mmn_df['ses_group'] = ses_grouping.map({True: 1, False: 2})  # 1:low SES
mmn_df['ses_label'] = ses_grouping.map({True: 'low', False: 'high'})
mmn_df['condition_label'] = mmn_df.conditions
mmn_df.replace({'condition_label': {vv: kk for kk, vv in conditions.items()}},
               inplace=True)
mmn_df['hem_label'] = mmn_df.hemisphere
mmn_df.replace({'hem_label': {vv: kk for kk, vv in hems.items()}},
               inplace=True)
# Combine conditions
stim_grouping = mmn_df.conditions == 3
mmn_df['stimulus'] = stim_grouping.map({True: 1, False: 2})  # 1:standard
mmn_df['stim_label'] = mmn_df.stimulus
mmn_df.replace({'stimulus': {1: 'standard', 2: 'deviant'}}, inplace=True)
# Write out descriptives as csv
cols = ['auc', 'latencies', 'channels', 'SES', 'Age(days)', 'HC']
grpby = ['ses_label', 'stimulus', 'hem_label']
desc = mmn_df.loc[:, cols + grpby].groupby(grpby).describe()
desc.to_csv(op.join(params.static_dir, 'MMNdf_Descriptives'), sep='\t')
print('\nDescriptives...\n', desc)
# Write out channel specidfic data for R
mmn_df = mmn_df[mmn_df.ch_type == 3]
mmn_df.drop(axis=1, columns=['BAD', 'ECG', 'SR(Hz)', 'complete', 'CDI',
                             'simms_inclusion', 'ParticipantId',
                             'ch_type']).to_csv(
	op.join(params.static_dir, 'MMNdf_RM.csv'), sep='\t')

# Plots
#  Ball & stick plots of MEG measures for SES within condition
for nm, tt in zip(['auc', 'latencies'],
                  ['Strength', 'Latency']):
	h = sns.catplot(x='stimulus', y=nm, hue='ses_label',
	                data=mmn_df[mmn_df.ch_type == 3],
	                kind='point', ci='sd', dodge=True, legend=True,
	                palette=sns.color_palette('pastel', n_colors=2, desat=.5))
	h.fig.suptitle(tt)
	h.despine(offset=2, trim=True)
df = mmn_df.merge(mmn_cdi_df, on='ParticipantId', sort='True', validate='m:m')
# Pairwise + density, and correlation matrix of all response variables
g = sns.pairplot(df, vars=['M3L', 'VOCAB', 'latencies', 'auc'], diag_kind='kde',
                 hue='CDIAge', palette='tab20')
plot_correlation_matrix(df[['M3L', 'VOCAB', 'latencies', 'auc']].corr())
# Linear regression model fit between CDI measures and MEG latencies
ages = np.arange(21, 31, 3)
for nm, tt in zip(['M3L', 'VOCAB'],
                  ['Mean length of utterance', 'Words understood']):
    fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
    hs = list()
    for fi, ax in enumerate(axs):
        # response variable
        mask = df.CDIAge == ages[fi]
        y_vals = np.squeeze(df[mask][nm].values.reshape(-1, 1))
        # predictor
        x_vals = np.squeeze(df[mask].latencies.values.reshape(-1, 1))
        assert (y_vals.shape == x_vals.shape)
        hs.append(ax.scatter(x_vals, y_vals, c='CornFlowerBlue', s=50,
                             zorder=5, marker='.', alpha=0.5))
        ax.set(xlabel='Latency (sec)', title='%d Mos' % ages[fi])
        if fi == 0:
            ax.set(ylabel=tt)
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(x_vals, y_vals)
        hs.append(ax.plot(x_vals, intercept + slope * x_vals, label='fitted '
                                                                    'line',
                          c="Grey", lw=2, alpha=0.5))
        ax.annotate('$\mathrm{r^{2}}=%.2f$''\n$\mathit{p = %.2f}$'
                    % (r_value ** 2, p_value),
                    xy=(0.05, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle='square', fc='w'))

# Equivalent routine for GLM fit with Sklearn
# from sklearn.metrics import mean_squared_error, r2_score
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import researchpy as rp
# from patsy import dmatrices
# for nm, tt in zip(['M3L', 'VOCAB'],
#                   ['Mean length of utterance', 'Words understood']):
#     fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
#     hs = list()
#     for fi, ax in enumerate(axs):
#         # response variable
#         mask = mmn_cdi_df.CDIAge == ages[fi]
#         y = mmn_cdi_df[mask][nm].values.reshape(-1, 1)
#         X = mmn_cdi_df[mask].SES.values.reshape(-1, 1)  # feature
#         assert (y.shape == X.shape)
#         hs.append(ax.scatter(X, y, c='CornFlowerBlue', s=50,
#                              zorder=5, marker='.', alpha=0.5))
#         ax.set(xlabel='SES', title='%d Mos' % ages[fi])
#         if fi == 0:
#             ax.set(ylabel=tt)
#         reg = fit_linear_reg(X, y)
#         hs.append(ax.plot(reg[2], reg[1], c="Grey", lw=2, alpha=0.5))
#         ax.annotate('$\mathrm{r^{2}}=%.2f$''\n$\mathit{p = %.2f}$'
#                     % (reg[3], reg[5]),
#                     xy=(0.05, 0.9), xycoords='axes fraction',
#                     bbox=dict(boxstyle='square', fc='w'))
#         # The coefficients ax+b
#         print(' Model Coefficients: \n', reg[0].coef_)
#         # The mean squared error
#         print(' Mean squared error: %.2f'
#               % mean_squared_error(y, reg[1]))
#         # Explained variance score: 1 is perfect prediction
#         print('Variance score: %.2f' % r2_score(y, reg[1]))
#         print(' Coefficient of determination R^2: %.2f' % reg[3])
#         print(' t-statistic = %6.3f pvalue = %6.4f'
#               % (reg[4], reg[5]))

# OLS Regression F-tests (ANOVA)
# for nm, tt in zip(['M3L', 'VOCAB'],
#                   ['Mean length of utterance', 'Words understood']):
#     for ai in ages:
#         df = mmn_cdi_df[mmn_cdi_df.CDIAge == ai]
#         print('Testing variable: %s ' % nm, 'At %d mos of age...' % ai)
#         # This automatically include the main effects for each factor
#         formula = '%s ~ C(ses_group)' % nm
#         model = ols(formula, df).fit()
#         print(f"    Overall model p = {model.f_pvalue: .4f}")
#         if model.f_pvalue < .05:  # Fits the model with the interaction term
#             print('\n====================================================',
#                   ' Marginal means')
#             print(rp.summary_cont(df.groupby(['ses_group']))[nm])
#             print('\n----------------------------------------------------')
#             print(' \nModel Summary')
#             print(model.summary())
#             print('\n----------------------------------------------------')
#             print('     \nJB test for normality p-value: %6.3f'
#                   % sm.stats.stattools.jarque_bera(model.resid)[1])
#             # Seeing if the overall model is significant
#             print(f"    \nOverall model F({model.df_model: .0f},"
#                   f"    {model.df_resid: .0f}) = {model.fvalue: .3f}, "
#                   f"    p = {model.f_pvalue: .4f}")
#             print('\n----------------------------------------------------')
#             print('ANOVA Table')
#             aov_table = sm.stats.anova_lm(model, typ=2, robust='hc3')
#             print(aov_table)
#         else:
#             print('Go fish.')
