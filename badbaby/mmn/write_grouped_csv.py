# coding: utf-8

from os import path as op
import numpy as np
import pandas as pd
import mne
from mne import read_evokeds, grand_average
import badbaby.defaults as params
import badbaby.return_dataframes as rd


def read_in_evoked(filename):
    """helper to read evoked file"""
    erf = read_evokeds(filename, condition=cond,
                       baseline=(None, 0))
    if erf.info['sfreq'] > 600.0:
        raise ValueError('bad_%s - %dHz Wrong sampling rate!'
                         % (subj, erf.info['sfreq']))
    chs = np.asarray(erf.info['ch_names'])
    assert (all(chs == np.asarray(params.vv_ch_order)))
    if len(erf.info['bads']) > 0:
        erf.interpolate_bads()
    return erf, chs


# Some parameters
agency = 'SIMMS'
analysis = 'Individual-matched'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
csv_filename = '%s-%s-%d_dataset.csv' % (agency, analysis, lpf)
data_dir = params.meg_dirs['mmn']
fig_dir = op.join(data_dir, 'figures')
df_mmn, df_cdi = rd.return_simms_mmn_dfs()
groups = list()
# Group by...
by_age = df_mmn['Age(days)'].describe().median()  # AGE for Simms
mask = df_mmn['Age(days)'] < by_age
ages = dict([(2, '2 months'), (6, '6 months')])
ages_nms = [ages[kind] for kind in ages.keys()]
groups.append(df_mmn[mask].Subject_ID.values.tolist())
groups.append(df_mmn[~mask].Subject_ID.values.tolist())
names = [nm for nm in ages.values()]

if agency == 'Ford':
    by_ses = df_mmn['SES'].describe().median()  # SES for Ford/Bezos
    mask = df_mmn['SES'] < by_ses
    ses = dict([('lo', 'Low SES'), ('hi', 'High SES')])
    ses_nms = [ses[kind] for kind in ses.keys()]
    groups.append(df_mmn[mask].Subject_ID.values.tolist())
    groups.append(df_mmn[~mask].Subject_ID.values.tolist())
    names += [nm for nm in ses.values()]

# Loop over groups & get peak latencies and AUC
print('Getting ERF dependent measures...')
datasets = {nm: {} for nm in names}
for key in datasets.keys():
    for measure in ['locs', 'lats', 'amps']:
        datasets[key][measure] = list()
dfs = list()
measures = ['loc', 'lat', 'amp']
from_ = df_mmn.columns.tolist().index('Exam date')
to_ = df_mmn.columns.tolist().index('simms_inclusion')
for group, nm in zip(groups, names):
    subjects = df_mmn[df_mmn.Subject_ID.isin(group)]
    subjects = subjects.Subject_ID.values.tolist()
    n = len(subjects)
    drop = df_mmn.columns.tolist()[from_:to_]
    print(' Loading data for %d subjects in %s...' % (n, nm))
    for ci, cond in enumerate(conditions):
        print('     Analysis-%s / condition-%s' % (analysis, cond))
        for si, subj in enumerate(subjects):
            print('       %s' % subj)
            evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                                  '%s_%d-sss_eq_bad_%s-ave.fif'
                                  % (analysis, lpf, subj))
            evoked, ch_names = read_in_evoked(evoked_file)
            pick_mag = mne.pick_types(evoked.info, meg='mag')
            assert pick_mag.shape[0] == 102
            # Do channel selection
            for jj, (key, val) in enumerate(params.sensors.items()):
                # initialize data array
                if ci == 0 and si == 0 and jj == 0:
                    #  conditions x subjects x hemispheres
                    locs = np.zeros((len(conditions), len(subjects),
                                     2)).astype(str)
                    lats = np.zeros((len(conditions), len(subjects), 2))
                    amps = np.zeros_like(lats)  # AUC -100 ms to peak
                # subselect magnetometers channels
                picks = np.asarray(ch_names)[pick_mag]
                ch_selection = np.intersect1d(np.asarray(val), picks)
                assert len(ch_selection) == len(val) // 3
                evo = evoked.copy().pick_channels(ch_selection)
                evo.info.normalize_proj()
                # peak ERF latency bn 150-550ms
                out = evo.get_peak(ch_type='mag', tmin=.15, tmax=.55)
                locs[ci, si, jj] = out[0]
                lats[ci, si, jj] = out[1]  # milliseconds
                b = np.where(np.isclose(evoked.times,
                                        out[1], atol=1e-3))[0][0]
                a = np.where(np.isclose(evoked.times,
                                        (out[1] - .1),
                                        atol=1e-3))[0][0]
                auc = (np.sum(np.abs(evo.data[:, a: b]))
                       * (len(evoked.times) / evoked.info['sfreq']))
                amps[ci, si, jj] = auc
    xx_ = df_mmn.drop(columns=drop)
    df_group = list()
    for xx, yy in zip(['locs', 'lats', 'amps'], [locs, lats, amps]):
        #  --> subjects x conditions x hemispheres
        yy = np.transpose(yy, (2, 0, 1))
        datasets[nm][xx] = yy
        sz = yy[-1].size
        # tile and sort list --> tiled vector of levels for subjects
        ss = np.vstack((subjects, subjects) * (sz // n)).reshape((-1,),
                                                                 order='F')
        conds = np.tile(np.vstack((conditions, conditions)).reshape((-1,),
                                                                    order='F'),
                        n)
        hems = np.tile([ss for ss in params.sensors.keys()], sz)
        df_group.append(pd.DataFrame({'Subject_ID': ss.tolist(),
                                      'group': np.tile(nm, len(ss)),
                                      'condition': conds.tolist(),
                                      'hemisphere': hems.tolist(),
                                      xx: yy.reshape(-1, order='C')}))
    yy_ = pd.concat(df_group, axis=0, ignore_index=True)
    yy_ = pd.merge(xx_, yy_, how='inner', sort=False,
                   on='Subject_ID', validate='1:m')
    dfs.append(yy_)
df_merged = pd.concat(dfs, ignore_index=True, sort=False)
df_merged.to_csv(op.join(params.static_dir, csv_filename),
                 sep='\t', encoding='utf-8')

# Loop over groups & plot grand average ERFs
print('Plotting Grand Averages')
for ii, (group, nm) in enumerate(zip([two_mos_subjects, six_mos_subjects,
                                      low_ses_subjects, high_ses_subjects],
                                     names)):
    subjects = df_mmn[df_mmn.Subject_ID.isin(group)]
    subjects = subjects.Subject_ID.values.tolist()
    n = len(subjects)
    print(' Loading data for %d subjects in %s...' % (n, nm))
    for ci, cond in enumerate(conditions):
        print('     Analysis-%s / condition-%s' % (analysis, cond))
        file_out = op.join(data_dir, '%s_%s_%smos_%d_grd-ave.fif'
                           % (analysis, cond, nm.replace(' ', ''), lpf))
        if not op.isfile(file_out):
            print('      Doing averaging...')
            evokeds = list()
            for si, subj in enumerate(subjects):
                print('       %s' % subj)
                evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                                      '%s_%d-sss_eq_bad_%s-ave.fif'
                                      % (analysis, lpf, subj))
                evoked, ch_names = read_in_evoked(evoked_file)
                evokeds.append(evoked)
            # do grand averaging
            grandavr = grand_average(evokeds)
            grandavr.save(file_out)
        else:
            print('Reading...%s' % op.basename(file_out))
            grandavr = read_evokeds(file_out)[0]
        # peak ERF latency bn 100-550ms
        ch, lat = grandavr.get_peak(ch_type='mag', tmin=.15, tmax=.55)
        if cond in ['all', 'deviant']:
            print('     Peak latency for %s in %s group:\n'
                  '         %s at %0.3fms' % (cond, group, ch, lat))
        # plot ERF topography at peak latency and 100ms before
        timing = [lat - .1, lat]
        hs = grandavr.plot_joint(title=nm + ' ' + cond,
                                 times=timing, ts_args=params.ts_args,
                                 topomap_args=params.topomap_args)
        for h, ch_type in zip(hs, ['grad', 'mag']):
            fig_out = op.join(fig_dir, '%s_%s_%s_%d_%s_grd-ave.eps'
                              % (analysis, cond, nm.replace(' ', ''),
                                 lpf, ch_type))
            h.savefig(fig_out, dpi=240, format='eps')