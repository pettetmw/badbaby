# -*- coding: utf-8 -*-

"""Script plots CDI and MEG-MMN dependents for specified cohort"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import badbaby.defaults as params
import badbaby.return_dataframes as rd
from badbaby.set_viz_params import box_off, setup_pyplot


# Some parameters
leg_kwargs = setup_pyplot('seaborn-notebook')
data_dir = params.meg_dirs['mmn']
fig_dir = op.join(data_dir, 'figures')
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
agency = 'Ford'
csv_filename = '%s-%s-%d_dataset.csv' % (agency, analysis, lpf)
static_dir = params.static_dir
meg_df, cdi_df = rd.return_ford_mmn_dfs()  # Ford, SIMMS, or Bezos
df = pd.read_csv(op.join(static_dir, csv_filename), delimiter='\t')
df.drop(columns='Unnamed: 0', inplace=True)
sns.set(style="whitegrid", palette="muted", color_codes=True)
groups = np.unique(df.group.values).tolist()
outputs = ['age', 'ses']
features = ['lats', 'amps']
units = ['latency (sec)', 'strength (T)']
ws_measures = ['M3L', 'VOCAB']
cdi_ages = np.arange(18, 31, 3)

# plot dependents

plt.gcf().savefig(op.join(static_dir, 'figure',
                          '%s_%s_%d_sexes_pie.pdf' % (agency, analysis, lpf)),
                  dpi=240, format='pdf')

g.savefig(op.join(static_dir, 'figure',
                  '%s_age-x-hc_reg.pdf' % agency),
          dpi=240, format='pdf')


# violinplots
for feature, unit in zip(features, units):
    for ii, grouping in enumerate([groups[i:i + 2]
                                   for i in range(0, len(groups), 2)]):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
        for ai, (ax, grp) in enumerate(zip(axs, grouping)):
            group = df[df.group == grp]
            dataset = group[group[feature].notna()]
            g = sns.violinplot(x='hemisphere', y=feature, hue='condition',
                               data=dataset, ax=ax)
            g.legend().set_visible(False)
            g.set(title=grp, xlabel='Hemisphere', ylabel=unit,
                  xticklabels=['left', 'right'])
            box_off(ax)
        hs, labels = g.get_legend_handles_labels()
        fig.legend(hs, labels, **leg_kwargs)
        sns.despine(fig, left=True, bottom=True, offset=5, trim=True)
        fig.savefig(op.join(static_dir, 'figure', '%s_auc-x-%s_%s_violins.pdf'
                            % (agency, outputs[ii], analysis)),
                    dpi=240, format='pdf')
plt.show()
# Descriptives
lats_desc = df.groupby(['group', 'condition', 'hemisphere']). \
    agg({'lats': ['count', 'mean', 'std', 'skew']})
# t-tests
lats_ds = df.groupby(['group', 'condition', 'hemisphere'])
# Standard high vs low SES
mask = lats_ds.get_group(('Low SES', 'standard', 'rh')).lats.notna()
x = lats_ds.get_group(('Low SES', 'standard', 'rh'))
x = x[mask]['lats'].values
mask = lats_ds.get_group(('High SES', 'standard', 'rh')).lats.notna()
y = lats_ds.get_group(('High SES', 'standard', 'rh'))
y = y[mask]['lats'].values
ts, ps = stats.ttest_ind(x, y, equal_var=False)
# CDI between SES
cdi27 = cdi_df[cdi_df.CDIAge == 30]
ix_low = ['BAD_%s' % ss[:3]
          for ss in np.unique(df[df.group == 'Low SES'].Subject_ID.values)]
lowses27 = cdi27[cdi27.ParticipantId.isin(ix_low)].loc[:, ['M3L',
                                                           'VOCAB']].values

ix_high = ['BAD_%s' % ss[:3]
           for ss in np.unique(df[df.group == 'High SES'].Subject_ID.values)]
highses27 = cdi27[cdi27.ParticipantId.isin(ix_high)].loc[:, ['M3L',
                                                             'VOCAB']].values
ts, ps = stats.ttest_ind(lowses27, highses27, axis=0, equal_var=False)
# Correlations: dependent measures vs. CDI measures
for ii, group in enumerate(groups):
    print(' %s group' % group)
    for jj, key in enumerate(params.sensors.keys()):
        for response, nm in zip(features, units):
            for cc, cond in enumerate(conditions):
                y_ds = df[(df.group == group)
                          & (df.hemisphere == key)
                          & (df.condition == cond)]
                y_ds = y_ds[(y_ds[response].notna())]
                subjects = list(
                    set(y_ds[y_ds.group == group].Subject_ID.values))
                ids = list(set(['BAD_%s' %ss[:3] for ss in subjects]))
                for cdi_age in cdi_ages:
                    y_ds_copy = y_ds.copy()
                    x_ds = cdi_df[cdi_df.CDIAge == cdi_age]
                    on_these = np.intersect1d(np.asarray(ids),
                                              x_ds.ParticipantId.values)
                    y_ds_copy.insert(len(y_ds_copy.columns), 'ParticipantId',
                                     ['BAD_%s' % ss[:3] for ss in
                                      y_ds_copy.Subject_ID.values])
                    # combine MEG & CDI datasets
                    comb_df = pd.merge(y_ds_copy, x_ds, on='ParticipantId',
                                       how='inner',
                                       validate='m:1')
                    for ws in ws_measures:
                        # feature
                        Y = comb_df[response].values
                        # response
                        X = comb_df[ws].values
                        assert Y.shape == X.shape
                        print('     %d subjects...\n' % X.shape[0],
                              '         %s vs. %s' % (ws, response))
                        X_intercept = sm.add_constant(X)
                        results = sm.OLS(Y, X_intercept).fit()
                        ols_ln = results.params[0] + results.params[1] * X
                        if results.f_pvalue < .05:
                            print('Regression results for %s...\n'
                                  % groups[ii])
                            print(' %s at %d vs. %s-%s-%s' % (ws, cdi_age,
                                                              cond, key,
                                                              response))
                            print(results.summary())
                            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                            ax.plot(X, Y, 'o', markersize=12)
                            ax.plot(X, ols_ln, lw=3, ls='solid',
                                    color='c', zorder=5)
                            ax.set(title=groups[ii], ylabel=nm,
                                   xlabel=ws)
                            box_off(ax)
                            ax.get_xticklabels()[0]. \
                                set(horizontalalignment='right')
                            hb = Line2D(range(1), range(1), color='crimson')
                            ax.legend([hb], ['$\mathrm{r^{2}}=%.2f$\n'
                                             'p = %.3f' % (results.rsquared,
                                                           results.f_pvalue)],
                                      **leg_kwargs)
                            fig.tight_layout()
                            sns.despine(fig, left=True, bottom=True, offset=5,
                                        trim=True)
                            f_out = '%s_%s-%d-x-%s-%s_%s_%s_%s-%d_regplot.pdf' \
                                    % (agency, ws, cdi_age, response, key,
                                       groups[ii].replace(' ', ''),
                                       analysis, cond, lpf)
                            fig.savefig(op.join(static_dir, 'figure', f_out),
                                        dpi=240,
                                        format='pdf')
                            plt.close(plt.gcf())
