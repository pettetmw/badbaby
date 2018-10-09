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
import badbaby.defaults as params
import badbaby.return_dataframes as rd
from meegproc.utils import box_off

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
df = xx.groupby(['ses_group', 'CDIAge', 'Sex'], as_index=False)
df_desc = df[['SES', 'HC', 'M3L', 'VOCAB']].agg([np.count_nonzero,
                                                 np.max, np.min, np.mean,
                                                 np.std, np.var, np.median,
                                                 stats.sem])
#  Some plots
print('\nDescriptive stats for Age(days) variable...\n',
      xx['Age(days)'].describe())
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
xx.complete.groupby(xx.Sex).sum().plot.pie(subplots=False, ax=ax)
ax.set(title='Sex', ylabel='Data acquired')
fig.axis('equal')
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
y = xx[xx.CDIAge == 30].M3L.values.reshape(-1, 1)  # target
mask_X = mmn_features.Subject_ID.isin(xx[xx.CDIAge == 30].Subject_ID)
X = mmn_features[mask_X].SES.values.reshape(-1, 1)  # feature
assert(y.shape == X.shape)
prediction_space = np.linspace(min(X), max(X), num=X.shape[0]).reshape(-1, 1)
reg = linear_model.LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(prediction_space)
plt.scatter(X, y)
plt.ylabel('Words Understood')
plt.xlabel('SES')
plt.scatter(X, y, c='CornFlowerBlue')
plt.plot(prediction_space, y_pred, c="Grey", lw=2)
# The coefficients ax+b
print(' Model Coefficients: \n', reg.coef_)
# The mean squared error
print(' Mean squared error: %.2f'
      % mean_squared_error(y, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, y_pred))
R_sq = reg.score(X, y)
n = X.shape[0]-2
print(' Coefficient of determination R^2: %.2f' % R_sq)
# Note Calculation of t-value from r-value on StackExchange network
# https://tinyurl.com/yapl82m3
t_val = np.sqrt(R_sq) / np.sqrt((1-R_sq) / n)
p_val = stats.t.sf(np.abs(t_val), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt) using stats Survival Function (1-CDF — sometimes more accurate))  # noqa
print(' t-statistic = %6.3f pvalue = %6.4f'
      % (t_val, p_val))


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
