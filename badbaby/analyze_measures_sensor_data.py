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
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
import researchpy as rp
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import badbaby.defaults as params
import badbaby.return_dataframes as rd


def fit_sklearn_glm(feature, response):
    """Helper to fit simple GLM"""
    prediction_space = np.linspace(min(feature), max(feature),
                                   num=feature.shape[0]).reshape(-1, 1)
    reg = linear_model.LinearRegression().fit(feature, response)
    y_pred = reg.predict(prediction_space)
    return reg, y_pred, prediction_space


def return_r_stats(rr, n):
    """Helper to compute t-stat for simple GLM"""
    """Notes:
        Calculation of t-value from r-value on StackExchange network
        https://tinyurl.com/yapl82m3
    """
    t_val = np.sqrt(rr) / np.sqrt((1-rr) / (n-2))
    # Two-sided pvalue as Prob(abs(t)>tt) using stats Survival Function (1-CDF — sometimes more accurate))  # noqa
    p_val = stats.t.sf(np.abs(t_val), n-1) * 2
    return t_val, p_val


def return_vif(feature, response, formula):
    """Helper to compute variance inflation factors for OLS model"""
    vif = pd.DataFrame()
    vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['features'] = X.columns
    return vif


# Some parameters
analysis = 'Individual-matched'
# encoding features
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
print('\nDescriptive stats for Age(days) variable...\n',
      mmn_cdi_df['Age(days)'].describe())
#  Some plots
g = sns.pairplot(mmn_cdi_df, vars=['M3L', 'VOCAB'], diag_kind='kde',
                 hue='CDIAge', palette='tab20')
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
mmn_cdi_df.complete.groupby(mmn_cdi_df.Sex).sum().plot.pie(subplots=False,
                                                           ax=ax)
ax.set(title='Sex', ylabel='MEG-CDI data acquired')
fig.tight_layout()
scatter_kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(mmn_cdi_df, hue="ses_group",
                  hue_kws=dict(marker=["v", "^"]))
g = g.map(plt.scatter, "Age(days)", "HC", **scatter_kws).add_legend(title='SES')
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

# Linear model of SES-CDI
ages = np.arange(21, 31, 3)
for nm, tt in zip(['M3L', 'VOCAB'],
                  ['Mean length of utterance', 'Words understood']):
    fig, axs = plt.subplots(1, len(ages), figsize=(12, 6))
    hs = list()
    for fi, ax in enumerate(axs):
        # response variable
        y = mmn_cdi_df[mmn_cdi_df.CDIAge == ages[fi]][nm].values.reshape(-1,
                                                                         1)
        mask = mmn_xls.Subject_ID.isin(mmn_cdi_df[mmn_cdi_df.CDIAge ==
                                                 ages[fi]].Subject_ID)
        X = mmn_xls[mask].SES.values.reshape(-1, 1)  # feature
        assert (y.shape == X.shape)
        hs.append(ax.scatter(X, y, c='CornFlowerBlue', s=50,
                             zorder=5, marker='.', alpha=0.5))
        ax.set(xlabel='SES', title='%d Mos' % ages[fi])
        if fi == 0:
            ax.set(ylabel=tt)
        r, y_mod, space = fit_sklearn_glm(X, y)
        hs.append(ax.plot(space, y_mod, c="Grey", lw=2, alpha=0.5))
        t, p = return_r_stats(r.score(X, y), X.shape[0])
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
        print(' Coefficient of determination R^2: %.2f' % r.score(X, y))
        print(' t-statistic = %6.3f pvalue = %6.4f'
              % (t, p))

# OLS Regression F-tests (ANOVA)
for nm, tt in zip(['M3L', 'VOCAB'],
                  ['Mean length of utterance', 'Words understood']):
    for ai in ages:
        df = mmn_cdi_df[mmn_cdi_df.CDIAge == ai]
        print('Testing variable: %s ' % nm, 'At %d mos of age...' % ai)
        # This automatically include the main effects for each factor
        formula = '%s ~ C(ses_group)' % nm
        model = ols(formula, df).fit()
        print(f"Overall model p = {model.f_pvalue: .4f}")
        if model.f_pvalue < .05:  # Fits the model with the interaction term
            print('\n====================================================',
                  ' Marginal means')
            print(rp.summary_cont(df.groupby(['ses_group']))[nm])
            print('\n----------------------------------------------------')
            print(' \nModel Summary')
            print(model.summary())
            print('\n----------------------------------------------------')
            print('     \nJB test for normality p-value: %6.3f'
                  % sm.stats.stattools.jarque_bera(model.resid)[1])
            # Seeing if the overall model is significant
            print(f"    \nOverall model F({model.df_model: .0f},"
                  f"{model.df_resid: .0f}) = {model.fvalue: .3f}, "
                  f"p = {model.f_pvalue: .4f}")
            print('\n----------------------------------------------------')
            print('ANOVA Table')
            aov_table = sm.stats.anova_lm(model, typ=2, robust='hc3')
            print(aov_table)
        else:
            print('Go fish.')
# Put dependents into dataframe of obs x conditions x sensor types x hemisphere
sz = data['auc'].size // 2
# interleave list --> tiled vector of levels for factors
subjects = mmn_xls.Subject_ID.values
subj_ids = np.vstack((subjects, subjects) *
                     (sz // len(subjects))).reshape((-1,), order='F')
c_levels = np.vstack((list(conditions.values()), list(conditions.values())) *
                     len(ch_types)).reshape((-1,), order='F')
# age = np.vstack(([2, 6], [2, 6]) * 2).reshape((-1,), order='F')
sns_levels = np.vstack((list(ch_types.values()),
                        list(ch_types.values()))).reshape((-1,), order='F')
hem_levels = np.vstack(list(hems.values())).reshape((-1,), order='F')
meg_df = pd.DataFrame({'Subject_ID': subj_ids.tolist(),
                       'conditions': c_levels.tolist() * len(subjects),
                       'ch_type': sns_levels.tolist() * (sz // 2),
                       'hemisphere': hem_levels.tolist() * sz,
                       'auc': -np.log(data['auc'].reshape(-1, order='C')),
                       'latencies': data['latencies'].reshape(-1, order='C'),
                       'channels': data['channels'].reshape(-1, order='C')})
naves = np.transpose(data['naves'], (1, 0))
subj_ids = np.vstack([subjects] * len(conditions)).reshape((-1,), order='F')
c_levels = np.vstack(list(conditions.values())).reshape((-1,), order='F')
erf_naves = pd.DataFrame({'Subject_ID': subj_ids.tolist(),
                          'conditions': c_levels.tolist() * len(subjects),
                          'naves': data['naves'].reshape(-1, order='C')})

#  Merge MEG with MMN xls dataframes
mmn_df = meg_df.merge(mmn_xls, on='Subject_ID', validate='m:1')
g = sns.pairplot(mmn_df, vars=['auc', 'latencies'], hue='hemisphere',
                 height=2.5)

# split gradiometer data on conditions and hemispheres
grouped = mmn_df[mmn_df.ch_type == 3].groupby(['conditions', 'hemisphere'])
print('\nDescriptive stats for Age(days) variable...\n', grouped.describe())
ses_grouping = mmn_df.SES <= mmn_df.SES.median()
mmn_df['ses_group'] = ses_grouping.map({True: 1, False: 2})  # 1:low SES
mmn_df['ses_label'] = ses_grouping.map({True: 'low', False: 'high'})
mmn_df['condition_label'] = mmn_df.conditions
mmn_df.replace({'condition_label': {1: 'ba', 2: 'wa', 3: 'standard'}},
               inplace=True)
mmn_df['hem_label'] = mmn_df.hemisphere
mmn_df.replace({'hem_label': {1: 'lh', 2: 'rh'}}, inplace=True)

#  Ball & stick plots of MEG measures as function of conditions and SES
for nm, tt in zip(['auc', 'latencies'],
                  ['Strength', 'Latency']):
    h = sns.catplot(x='condition_label', y=nm, hue='ses_label',
                    data=mmn_df[mmn_df.ch_type == 3],
                    kind='point', ci='sd', dodge=True, legend=True,
                    palette=sns.color_palette('pastel', n_colors=2, desat=.5))
    h.fig.suptitle(tt)
    h.despine(offset=2, trim=True)


#  TODO: Compute VIFs
features_formula = "+".join(mmn_df.columns - ["auc"])

# Combine conditions
stim_grouping = mmn_df.conds_code == 3
mmn_df['stimulus'] = stim_grouping.map({True: 1, False: 2})  # 1:standard
for nm, tt in zip(['auc', 'latencies'],
                  ['Strength', 'Latency']):
    h = sns.catplot(x='stimulus', y=nm, hue='ses_group',
                    data=mmn_df[mmn_df.ch_type == 'grad'],
                    kind='point', ci='sd', dodge=True,
                    palette=sns.color_palette('pastel', n_colors=2, desat=.5))
    h.fig.suptitle(tt)
    h.despine(offset=2, trim=True)

# OLS Regression F-tests (ANOVA)
for nm, tt in zip(['auc', 'latencies'],
                  ['Strength', 'Latency']):
    # This automatically include the main effects for each factor
    model = ols('%s ~ C(ses_group)* C(conds_code)* C(hemisphere)' % nm,
                mmn_df[mmn_df.ch_type == 'grad']).fit()
    print(f"Overall model p = {model.f_pvalue: .4f}")
    if model.f_pvalue < .05:  # Fits the model with the interaction term
        print(rp.summary_cont(mmn_df.groupby(['ses_group', 'conditions',
                                              'hemisphere']))[nm])
        print(model.summary())
        #  variance inflation factor, VIF, for one exogenous variable
        print(' JB test for normality p-value: %6.3f'
              % sm.stats.stattools.jarque_bera(model.resid)[1])
        # Seeing if the overall model is significant
        print(f"    Overall model F({model.df_model: .0f},"
              f"    {model.df_resid: .0f}) = {model.fvalue: .3f}, "
              f"    p = {model.f_pvalue: .4f}")
        aov_table = sm.stats.anova_lm(model, typ=2, robust='hc3')
        print(aov_table)
    else:
        print(' Go fish.\n')

