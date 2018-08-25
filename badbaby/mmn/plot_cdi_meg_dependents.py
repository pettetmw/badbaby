# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

from os import path as op
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import badbaby.parameters as params
import badbaby.return_dataframes as rd
from badbaby.set_viz_params import box_off, setup_pyplot

# Some parameters
leg_kwargs = setup_pyplot('seaborn-notebook')
data_dir = params.meg_dirs['mmn']
fig_dir = op.join(data_dir, 'figures')
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
csv_filename = 'Ford-%s-%d_dataset.csv' % (analysis, lpf)
static_dir = op.join(params.project_dir, 'badbaby', 'static')

meg_df, cdi_df = rd.return_ford_mmn_dfs()
df = pd.read_csv(op.join(static_dir, csv_filename), delimiter='\t')
df.drop(columns='Unnamed: 0', inplace=True)

# plot dependents
sns.set(style="whitegrid", palette="muted", color_codes=True)
groups = np.unique(df.group.values).tolist()
groups = [groups[i:i + 2] for i in range(0, len(groups), 2)]
outputs = ['age', 'ses']
features = ['lats', 'amps']
units = ['latency (sec)', 'strength (T)']
ws_measures = ['M3L', 'VOCAB']
cdi_ages = np.arange(18, 31, 3)

# violinplots
for feature, unit in zip(features, units):
    for ii, grouping in enumerate(groups):
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
        fig.savefig(op.join(fig_dir, 'Ford_auc-x-%s_%s_violins.pdf'
                            % (outputs[ii], analysis)),
                    dpi=240, format='pdf')

# Correlations: dependent measures vs. CDI measures
groups = np.unique(df.group.values)
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
                            print(results.summary())
                            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                            ax.set(title=groups[ii], ylabel=ws,
                                   xlabel=nm)
                            ax.plot(X, Y, 'o', markersize=10)
                            ax.plot(X, ols_ln, lw=1.5, ls='solid',
                                    color='Crimson', zorder=5)
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
                            f_out = 'Ford_%s-%d-x-%s-%s_%s_%s_%s-%d_regplot.pdf' \
                                    % (ws, cdi_age, response, key,
                                       groups[ii].replace(' ', ''),
                                       analysis, cond, lpf)
                            fig.savefig(op.join(fig_dir, f_out), dpi=240,
                                        format='pdf')
                            plt.close(plt.gcf())
