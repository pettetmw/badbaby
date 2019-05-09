# -*- coding: utf-8 -*-
"""Write out repeated measures MEG & CDI data files (.tsv).
    Script developed to visualize and export tidy ("long-form")
    each CDI & MEG dataframes.
        1. Combine filtered dataframes for behavioral and MEG data
        2. Visualize SES median split data
        3. Write out encoded files (.tsv) for R
    Dummy variables:
        CDI Age
            A->18 mos
            B->21
            C->24
            D->27
            E->30
        parental yrs of edu
            A->10
            B->12
            C->13
            D->14
            E->15
            F->16
            G->17
            H->18
            I->19
            J->20
            K->22
        parental Hollingshead scores
            A->3 HS
            B->4 GED
            C->5 College
            D->6 BS
            E->7 Graduate

"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as le
from sklearn import linear_model
import statsmodels.graphics.api as smg

import badbaby.python.defaults as params
import badbaby.python.return_dataframes as rd


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


pd.set_option('display.max_columns', None)
# Some parameters
analysis = 'Individual-matched'
stimuli = ['standard', 'Ba', 'Wa']
lpf = 30
age = 2
meg_dependents = ['auc', 'latencies', 'naves']
data_dir = params.megPdg_dirs['mmn']
fname = op.join(params.dataDir, '%s_%s-mos_%d_measures.npz'
                % (analysis, age, lpf))
if not op.isfile(fname):
    raise RuntimeError('%s not found.' % fname)
data = np.load(fname)

#################################################
# Combine filtered dataframes for CDI dataframe #
#################################################
# Get dataframes
mmn_xls, cdi_xls = rd.return_dataframes('mmn', age=age, ses=True)
assert cdi_xls.subjId.unique().shape[0] == 71
assert mmn_xls.subjId.values.shape[0] == 25
# Merge MEG & CDI features frames
mmn_xls.subjId = ['BAD_%s' % ss.split('a')[0]
                  for ss in mmn_xls.subjId.values]
cdi_df = pd.merge(cdi_xls, mmn_xls, on='subjId', how='inner',
                  sort=True, validate='m:1').reindex()
assert cdi_df.subjId.unique().shape[0] == 25

#######################################
# Visualize SES median split CDI data #
#######################################
# Split data on SES
sesGrouping = cdi_df.ses <= cdi_df.ses.median()  # low ses True
cdi_df['sesGroup'] = sesGrouping.map({True: 'low', False: 'high'})

# Pairwise + density, and correlation matrix of CDI response variables
g = sns.pairplot(cdi_df, vars=['m3l', 'vocab'], diag_kind='kde',
                 hue='cdiAge', palette='tab20')
plot_correlation_matrix(cdi_df[['cdiAge', 'm3l', 'vocab']].corr())

# pie chart of gender
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
cdi_df.complete.groupby(cdi_df.gender).sum().plot.\
    pie(autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.35,
        subplots=False, ax=ax)
ax.set(title='Gender', ylabel='MEG-CDI data acquired')
fig.tight_layout()

# pie chart of parental ethnicities
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
cdi_df.complete.groupby(cdi_df.maternalEthno).sum().plot.\
    pie(autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.35,
        subplots=False, ax=ax)
ax.set(title='Mom Ethnicity')
fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=(2, 2))
cdi_df.complete.groupby(cdi_df.paternalEthno).sum().plot.\
    pie(autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.35,
        subplots=False, ax=ax)
ax.set(title='Dad Ethnicity')
fig.tight_layout()

# scatter head circumference vs. age
scatter_kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(cdi_df, hue="sesGroup",
                  hue_kws=dict(marker=["v", "^"]))
g.map(plt.scatter, "age", "headSize", **scatter_kws).add_legend(title='SES')

# CDI measure vs age regression & SES group within age distribution plots
for nm, title in zip(['m3l', 'vocab'],
                     ['Mean length of utterance', 'Words understood']):
    g = sns.lmplot(x="cdiAge", y=nm, truncate=True, data=cdi_df)
    g.set_axis_labels("Age (months)", nm)
    g.ax.set(title=title)
    g.despine(offset=2, trim=True)
    h = sns.catplot(x='cdiAge', y=nm, hue='sesGroup',
                    data=cdi_df[cdi_df.cdiAge > 18],
                    kind='violin', scale_hue=True, bw=.2, linewidth=1,
                    scale='count', split=True, dodge=True, inner='quartile',
                    palette=sns.color_palette('pastel', n_colors=2, desat=.5),
                    margin_titles=True, legend=False)
    h.add_legend(title='SES')
    h.fig.suptitle(title)
    h.despine(offset=2, trim=True)

# Linear regression fit between CDI measures and SES scores
ages = np.arange(21, 31, 3)
for nm, tt in zip(['m3l', 'vocab'],
                  ['Mean length of utterance', 'Words understood']):
    fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
    hs = list()
    for fi, ax in enumerate(axs):
        # response variable
        mask_y = cdi_df.cdiAge == ages[fi]
        y_vals = np.squeeze(cdi_df[mask_y][nm].values.reshape(-1, 1))
        mask_x = mmn_xls.subjId.isin(
            cdi_df[cdi_df.cdiAge == ages[fi]].subjId)
        # predictor
        x_vals = np.squeeze(mmn_xls[mask_x].ses.values.reshape(-1, 1))
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

# factor encode
# subjects
subjects = cdi_df.subjId
subjects = dict(zip(subjects, le().fit_transform(subjects) + 1))
cdi_df.insert(0, 'subject', cdi_df.subjId.replace(subjects))
# Paternal yrs of edu
eduVals = list(set(cdi_df.paternalEdu.values.astype(np.int64)).
               union(cdi_df.maternalEdu.values.astype(np.int64)))
eduScores = dict(zip(eduVals, le().fit_transform(eduVals) + 1))
for col in ['maternalEdu', 'paternalEdu']:
    cdi_df[col] = cdi_df[col].map(eduScores)
# Paternal Hollingshead score
hhScores_vals = list(set(cdi_df.paternalHscore.values).
                     union(cdi_df.maternalHscore.values))
hhScores = dict(zip(hhScores_vals, le().fit_transform(hhScores_vals) + 1))
for col in ['maternalHscore', 'paternalHscore']:
    cdi_df[col] = cdi_df[col].map(hhScores)
# CDI age
# cdiAge_vals = cdi_xls.cdiAge.unique()
# cdiAge = dict(zip(cdiAge_vals, le().fit_transform(cdiAge_vals) + 1))
# cdi_df.cdiAge = cdi_df.cdiAge.map(cdiAge)
# # gender
# cdi_df.gender = cdi_df.gender.map({'M': 1, 'F': 2})
# Stimuli
stimuli = dict(zip(stimuli, le().fit_transform(stimuli) + 1))
# Channels
ch_types = dict(zip(['grad', 'mag'], le().fit_transform(['grad', 'mag']) + 1))
# Hemisphere
hems = dict(zip(['lh', 'rh'], le().fit_transform(['lh', 'rh']) + 1))

# Write out CSVs for R
cdi_df.drop(columns=['badCh', 'ecg', 'samplingRate', 'complete',
                     'behavioral', 'simmInclude',
                     'maternalEthno', 'paternalEthno',
                     'sib1dob', 'sib1gender',
                     'sib2dob', 'sib2gender', 'sib3dob', 'sib3gender',
                     'birthWeight(lbs)'],
            axis=1).to_csv(op.join(params.dataDir,
                                   'Ds-mmn-2mos_cdi_df.csv'), sep=',')


# Create MEG measures dataframe
# subjs x conds x sensors x hemisphere
sz = data['auc'].size
subjId = mmn_xls.subjId.values
nsubj = subjId.shape[0]
nstim = len(stimuli)
nsens = len(ch_types)
nhems = len(hems)
# interleave list --> tiled vector of levels for factors:
subjId = np.vstack((subjId, subjId) * (nstim * nsens * nhems // 2)).\
    reshape((-1,), order='F')
assert subjId.shape[0] == nsubj * nstim * nsens * nhems
stimulus = np.vstack((list(stimuli.keys()), list(stimuli.keys())) *
                     len(ch_types)).reshape((-1,), order='F')
channel = np.vstack((list(ch_types.keys()),
                     list(ch_types.keys()))).reshape((-1,), order='F')
hemisphere = np.vstack(list(hems.keys())).reshape((-1,), order='F')
meg_df = pd.DataFrame({
    'subjId': subjId.tolist(),
    'stimulus': stimulus.tolist() * nsubj,
    'channel': channel.tolist() * (sz // (nsens * nhems)),
    'hemisphere': hemisphere.tolist() * (sz // nhems),
    'auc': (data['auc'].reshape(-1, order='C')),
    'latencies': data['latencies'].reshape(-1,
                                           order='C'),
    'channels': data['channels'].reshape(-1, order='C')
})
assert meg_df.shape[0] == nsubj * nstim * nsens * nhems
naves = np.transpose(data['naves'], (1, 0))
subjId = np.vstack([mmn_xls.subjId.values] * len(stimuli)).reshape((-1,),
                                                                   order='F')
stimulus = np.vstack(list(stimuli.keys())).reshape((-1,), order='F')
# Inspect number of averages for each stimulus condition
erf_naves = pd.DataFrame({
    'subjId': subjId.tolist(),
    'stimulus': stimulus.tolist() * nsubj,
    'naves': data['naves'].reshape(-1, order='C')
})
assert erf_naves.shape[0] == nsubj * nstim

# factor encode
# subjects
subjects = meg_df.subjId
subjects = dict(zip(subjects, le().fit_transform(subjects) + 1))
meg_df.insert(0, 'subject', meg_df.subjId.replace(subjects))

# select gradiometer data
meg_df = meg_df[meg_df.channel == 'grad']
# Merge MEG with covariate dataframe
meg_df = meg_df.merge(mmn_xls, on='subjId', validate='m:1')
assert meg_df.shape[0] == sz // 2
# Split data on SES
sesGrouping = meg_df.ses <= meg_df.ses.median()
meg_df['sesGroup'] = sesGrouping.map({True: 'low', False: 'high'})
# Combine stimuli for oddball conditioning
stim_grouping = meg_df.stimulus == 'standard'
meg_df['oddballCond'] = stim_grouping.map({True: 'standard', False: 'deviant'})

# Descriptives MEG data
responses = ['age', 'headSize', 'ses',
             'birthWeight', 'maternalEdu', 'maternalHscore',
             'paternalEdu', 'paternalHscore',
             'auc', 'latencies']
grpby = ['oddballCond', 'hemisphere']
desc = meg_df[responses + grpby].groupby(grpby).describe()
print('\nDescriptives...\n', desc)

# Write out data for R
meg_df.drop(axis=1, columns=['channel', 'channels', 'badCh', 'ecg',
                             'samplingRate', 'complete', 'behavioral',
                             'simmInclude', 'sib1dob', 'sib1gender',
                             'sib2dob', 'sib2gender', 'sib3dob',
                             'sib3gender', 'birthWeight(lbs)']).to_csv(
    op.join(params.dataDir, 'Ds-mmn-2mos_meg_df.csv'), sep=',')

#######################################
# Visualize SES median split CDI data #
#######################################
# naves category plot
sns.catplot(x='stimulus', y='naves', kind='swarm', palette='tab20',
            data=erf_naves)

# Ball & stick plots of MEG measures for SES within condition
for nm, tt in zip(['auc', 'latencies'],
                  ['Strength', 'Latency']):
    h = sns.catplot(x='oddballCond', y=nm, hue='sesGroup',
                    data=meg_df,  # gradiometer data
                    kind='point', ci='sd', dodge=True, legend=True,
                    palette=sns.color_palette('pastel', n_colors=2, desat=.5))
    h.fig.suptitle(tt)
    h.despine(offset=2, trim=True)

# Correlation matrix for all response variables
# merge MEG & CDI dataframes
meg_df = meg_df.merge(cdi_df[['subjId', 'cdiAge', 'm3l', 'vocab']],
                      on='subjId', sort='True', validate='m:m')

plot_correlation_matrix(meg_df[['m3l', 'vocab', 'ses',
                                'latencies', 'auc']].corr())
# Pairwise + density
df = meg_df.copy()
# df.cdiAge = df.cdiAge.map(dict((k, v) for v, k in cdiAge.items()))
g = sns.pairplot(df, vars=['m3l', 'vocab', 'ses', 'latencies', 'auc'],
                 diag_kind='kde', hue='cdiAge', palette='tab20')

# Linear regression fit between CDI measures and MEG latencies
ages = np.arange(21, 31, 3)
for nm, tt in zip(['m3l', 'vocab'],
                  ['Mean length of utterance', 'Words understood']):
    fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
    hs = list()
    for fi, ax in enumerate(axs):
        # response variable
        deviants = df[(df.cdiAge == ages[fi]) & (df.oddballCond == 'deviant')]
        y_vals = np.squeeze(deviants[nm].values.reshape(-1, 1))
        # predictor
        x_vals = np.squeeze(deviants.latencies.values.reshape(-1, 1))
        assert x_vals.shape[0] == nsubj * nhems * 2  # 2-groups
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
# for nm, tt in zip(['m3l', 'vocab'],
#                   ['Mean length of utterance', 'Words understood']):
#     fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
#     hs = list()
#     for fi, ax in enumerate(axs):
#         # response variable
#         mask = cdi_df.cdiAge == ages[fi]
#         y = cdi_df[mask][nm].values.reshape(-1, 1)
#         X = cdi_df[mask].SES.values.reshape(-1, 1)  # feature
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
# for nm, tt in zip(['m3l', 'vocab'],
#                   ['Mean length of utterance', 'Words understood']):
#     for ai in ages:
#         df = cdi_df[cdi_df.cdiAge == ai]
#         print('Testing variable: %s ' % nm, 'At %d mos of age...' % ai)
#         # This automatically include the main effects for each factor
#         formula = '%s ~ C(sesGroup)' % nm
#         model = ols(formula, df).fit()
#         print(f"    Overall model p = {model.f_pvalue: .4f}")
#         if model.f_pvalue < .05:  # Fits the model with the interaction term
#             print('\n====================================================',
#                   ' Marginal means')
#             print(rp.summary_cont(df.groupby(['sesGroup']))[nm])
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
