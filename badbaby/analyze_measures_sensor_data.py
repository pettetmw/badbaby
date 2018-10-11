# -*- coding: utf-8 -*-

"""Write dependent measures for oddball sensor space data"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import os
from os import path as op
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
import researchpy as rp
import badbaby.defaults as params
import badbaby.return_dataframes as rd
from meegproc.utils import box_off


def fit_sklearn_glm(feature, target):
    """Helper to fit simple GLM"""
    prediction_space = np.linspace(min(feature), max(feature),
                                   num=feature.shape[0]).reshape(-1, 1)
    reg = linear_model.LinearRegression().fit(feature, target)
    y_pred = reg.predict(prediction_space)
    return reg, y_pred, prediction_space


def return_r_stats(rr, n):
    """Helper to compute t-stat for simple GLM"""
    """Notes:
        Calculation of t-value from r-value on StackExchange network
        https://tinyurl.com/yapl82m3
    """
    t_val = np.sqrt(rr) / np.sqrt((1 - rr) / (n - 2))
    # Two-sided pvalue as Prob(abs(t)>tt) using stats Survival Function (1-CDF — sometimes more accurate))  # noqa
    p_val = stats.t.sf(np.abs(t_val), n - 1) * 2
    return t_val, p_val


# Some parameters
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
age = 2
meg_dependents = ['auc', 'latencies', 'naves']
ch_type = ['grad', 'mag']
hems = ['lh', 'rh']
data_dir = params.meg_dirs['mmn']
fig_dir = op.join(data_dir, 'figures')
if not op.isdir(fig_dir):
    os.mkdir(fig_dir)
fname = op.join(data_dir, '%s_%s-mos_%d_measures.npz'
                % (analysis, age, lpf))
if not op.isfile(fname):
    raise RuntimeError('%s not found.' % fname)
data = np.load(fname)
mmn_features, cdi_df = rd.return_dataframes('mmn', age=age)

# Drop CDI cells with zero value
for nm in ['M3L', 'VOCAB']:
    cdi_df = cdi_df[cdi_df[nm] != 0]
cdi_df.drop(['DOB', 'Gender', 'Language', 'CDIForm',
             'CDIAgeCp', 'CDIDate', 'VOCPER', 'HOWUSE', 'UPSTPER',
             'UFUTPER', 'UMISPER', 'UCMPPER', 'UPOSPER', 'WORDEND', 'PLURPER',
             'POSSPER', 'INGPER', 'EDPER', 'IRWORDS', 'IRWDPER', 'OGWORDS',
             'COMBINE', 'COMBPER', 'CPLXPER'], axis=1, inplace=True)
mmn_features.drop(['Exam date', 'BAD', 'ECG', 'SR(Hz)', 'ACQ', 'MC-SVD',
                   'Artifact rej', 'Epochs', 'CDI', 'simms_inclusion'],
                  axis=1, inplace=True)
# Drop Subjects with zero value SES
mmn_features = mmn_features[mmn_features.SES > 0]
subj_ids = ['BAD_%s' % ss.split('a')[0]
            for ss in mmn_features.Subject_ID.values]
mmn_features.insert(len(mmn_features.columns), 'ParticipantId', subj_ids)

# Merge MMN features and CDI data
xx = pd.merge(mmn_features, cdi_df, on='ParticipantId',
              sort=True, validate='1:m').reindex()
# Split data on SES and CDIAge
ses_grouping = xx.SES < xx.SES.median()  # low SES True
xx['ses_group'] = ses_grouping.map({True: 'low', False: 'high'})

#  Some plots
print('\nDescriptive stats for Age(days) variable...\n',
      xx['Age(days)'].describe())
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
xx.complete.groupby(xx.Sex).sum().plot.pie(subplots=False, ax=ax)
ax.set(title='Sex', ylabel='Data acquired')
fig.tight_layout()
scatter_kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(xx, col="Sex", hue="ses_group",
                  hue_kws=dict(marker=["v", "^"]))
g = g.map(plt.scatter, "Age(days)", "HC", **scatter_kws)
g.add_legend(title='SES')
for nm, title in zip(['M3L', 'VOCAB'],
                     ['Mean length of utterance', 'Words understood']):
    g = sns.lmplot(x="CDIAge", y=nm, truncate=True, data=xx)
    g.set_axis_labels("Age (months)", nm)
    g.ax.set(title=title)
    g.despine(offset=2, trim=True)
    h = sns.catplot(x='CDIAge', y=nm, hue='ses_group',
                    col='Sex', data=xx[xx.CDIAge > 18],
                    kind='violin', scale_hue=True, bw=.2, linewidth=1,
                    scale='count', split=True, dodge=True, inner='quartile',
                    palette=sns.color_palette('pastel', n_colors=2, desat=.5),
                    margin_titles=True, legend=False)
    h.add_legend(title='SES')
    h.fig.suptitle(title)
    h.despine(offset=2, trim=True)

# Linear model of SES-CDI
ages = np.arange(21, 31, 3)
for nm, tt in zip(['M3L', 'VOCAB'],
                  ['Mean length of utterance', 'Words understood']):
    fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
    hs = list()
    for fi, ax in enumerate(axs):
        y = xx[xx.CDIAge == ages[fi]][nm].values.reshape(-1, 1)  # target
        mask = mmn_features.Subject_ID.isin(xx[xx.CDIAge ==
                                               ages[fi]].Subject_ID)
        X = mmn_features[mask].SES.values.reshape(-1, 1)  # feature
        assert(y.shape == X.shape)
        hs.append(ax.scatter(X, y, c='CornFlowerBlue', s=50,
                             zorder=5, marker='.', alpha=0.5))
        ax.set(xlabel='SES', title='%d Mos' % ages[fi])
        if fi == 0:
            ax.set(ylabel=tt)
        r, y_mod, space = fit_sklearn_glm(X, y)
        hs.append(ax.plot(space, y_mod, c="Grey", lw=2, alpha=0.5))
        ax.annotate('$\mathrm{r^{2}}=%.2f$''\n$\mathit{p = %.2f}$'
                    % (r.score(X, y), p),
                    xy=(0.05, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle='square', fc='w'))
        # The coefficients ax+b
        print(' Model Coefficients: \n', r.coef_)
        # The mean squared error
        print(' Mean squared error: %.2f'
              % mean_squared_error(y, y_mod))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(y, y_mod))
        t, p = return_r_stats(r.score(X, y), X.shape[0])
        print(' Coefficient of determination R^2: %.2f' % r.score(X, y))
        print(' t-statistic = %6.3f pvalue = %6.4f'
              % (t, p))

# OLS Regression ANOVA F-tests for Sex*SES interaction
for nm, tt in zip(['M3L', 'VOCAB'],
                  ['Mean length of utterance', 'Words understood']):
    for ai in ages:
        df = xx[xx.CDIAge == ai]
        print('Testing variable: %s ' % nm, 'At %d mos of age...' % ai)
        if model.f_pvalue < .05:  # Fits the model with the interaction term
            print(rp.summary_cont(df.groupby(['Sex', 'ses_group']))[nm])
            # This automatically include the main effects for each factor
            model = ols('VOCAB ~ C(Sex)* C(ses_group)', df).fit()
            model.summary()
            print('JB test for normality p-value: %6.3f'
                  % sm.stats.stattools.jarque_bera(model.resid)[1])
            # Seeing if the overall model is significant
            print(f"Overall model F({model.df_model: .0f},"
                  f"{model.df_resid: .0f}) = {model.fvalue: .3f}, "
                  f"p = {model.f_pvalue: .4f}")
            res = sm.stats.anova_lm(model, typ=2, robust='hc3')
        else:
            print(' Go fish you sorry SOB.')

# Put dependents into dataframe of obs x conditions x sensor types x hemisphere
sz = data['auc'].size // 2
# interleave list --> tiled vector of levels for factors
subjects = mmn_features.Subject_ID.values
ss = np.vstack((subjects, subjects) *
               (sz // len(subjects))).reshape((-1,), order='F')
cs = np.vstack((conditions, conditions) * len(ch_type)).reshape((-1,),
                                                                order='F')
# age = np.vstack(([2, 6], [2, 6]) * 2).reshape((-1,), order='F')
sn = np.vstack((ch_type, ch_type)).reshape((-1,), order='F')
hs = np.vstack(hems).reshape((-1,), order='F')
yy_ = pd.DataFrame({'subjID': ss.tolist(),
                    'conditions': cs.tolist() * len(subjects),
                    'ch_type': sn.tolist() * (sz // 2),
                    'hemisphere': hs.tolist() * sz,
                    'auc': data['auc'].reshape(-1, order='C'),
                    'latencies': data['latencies'].reshape(-1, order='C'),
                    'channels': data['channels'].reshape(-1, order='C')})
naves = np.transpose(data['naves'], (1, 0))
ss = np.vstack([subjects] * len(conditions)).reshape((-1,), order='F')
cs = np.vstack([conditions]).reshape((-1,), order='F')
yy_right = pd.DataFrame({'subjID': ss.tolist(),
                         'conditions': cs.tolist() * len(subjects),
                         'naves': naves.reshape(-1, order='C')})
yy = yy_.join(yy_right.set_index('subjID'), on='subjID',
              how='outer', rsuffix='_naves', sort=True).reindex()

pick = (yy.ch_type == 'grad')
fig, axs = plt.subplots(3, 1, sharey=False, figsize=(8, 10))
for col, ax in zip(meg_dependents, axs):
    yy[pick].hist(column=[col], bins=50,
                  alpha=0.5, ax=ax)

fig, axs = plt.subplots(3, 1, sharey=False, figsize=(8, 10))
ses.boxplot(column=['auc', 'latencies', 'naves'],
                 by=['conditions'], layout=(3, 1), ax=axs)
for ax in axs:
    box_off(ax)
