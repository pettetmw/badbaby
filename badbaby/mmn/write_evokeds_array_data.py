# coding: utf-8

from os import path as op
import time
import numpy as np
import pandas as pd
import mne
from mne import read_evokeds
from mne.channels.layout import _merge_grad_data


project_dir = '/home/ktavabi/Projects/badbaby/static'
data_dir = '/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch'
# Read excel sheets into pandas dataframes
xl_a = pd.read_excel(op.join(project_dir, 'badbaby.xlsx'), sheet_name='MMN',
                     converters={'BAD': str})
xl_b = pd.read_excel(op.join(project_dir, 'badbaby.xlsx'),
                     sheet_name='simms_demographics')
# Exclude subjects
inclusion = xl_a['simms_inclusion'] == 1
xl_a = xl_a[inclusion]
subjects = pd.Series(np.intersect1d(xl_a['Subject_ID'].values,
                                    xl_b['Subject_ID'].values))
# Find intersection between dataframes for common subjects
xl_a = xl_a[xl_a['Subject_ID'].isin(subjects.tolist())]
xl_b = xl_b[xl_b['Subject_ID'].isin(subjects.tolist())]
simms_df = pd.merge(xl_a, xl_b)

# Some parameters
groups = np.unique(simms_df.Group).tolist()
remap = dict([(2, '2 months'), (6, '6 months')])
grp_nms = [remap[kind] for kind in [2, 6]]
analysis = 'Individual'
conditions = ['standard', 'Ba', 'Wa']
lpf = 30
# channel selections
sensors = {'lh': ['MEG0111', 'MEG0112', 'MEG0113',
                  'MEG0121', 'MEG0122', 'MEG0123',
                  'MEG0341', 'MEG0342', 'MEG0343',
                  'MEG0321', 'MEG0322', 'MEG0323',
                  'MEG0331', 'MEG0332', 'MEG0333',
                  'MEG0131', 'MEG0132', 'MEG0133',
                  'MEG0211', 'MEG0212', 'MEG0213',
                  'MEG0221', 'MEG0222', 'MEG0223',
                  'MEG0411', 'MEG0412', 'MEG0413',
                  'MEG0421', 'MEG0422', 'MEG0423',
                  'MEG0141', 'MEG0142', 'MEG0143',
                  'MEG1511', 'MEG1512', 'MEG1513',
                  'MEG0241', 'MEG0242', 'MEG0243',
                  'MEG0231', 'MEG0232', 'MEG0233',
                  'MEG0441', 'MEG0442', 'MEG0443',
                  'MEG0431', 'MEG0432', 'MEG0433',
                  'MEG1541', 'MEG1542', 'MEG1543',
                  'MEG1521', 'MEG1522', 'MEG1523',
                  'MEG1611', 'MEG1612', 'MEG1613',
                  'MEG1621', 'MEG1622', 'MEG1623',
                  'MEG1811', 'MEG1812', 'MEG1813',
                  'MEG1821', 'MEG1822', 'MEG1823',
                  'MEG1531', 'MEG1532', 'MEG1533',
                  'MEG1721', 'MEG1722', 'MEG1723',
                  'MEG1641', 'MEG1642', 'MEG1643',
                  'MEG1631', 'MEG1632', 'MEG1633',
                  'MEG1841', 'MEG1842', 'MEG1843',
                  'MEG1911', 'MEG1912', 'MEG1913',
                  'MEG1941', 'MEG1942', 'MEG1943',
                  'MEG1711', 'MEG1712', 'MEG1713'],
           'rh': ['MEG1421', 'MEG1422', 'MEG1423',
                  'MEG1411', 'MEG1412', 'MEG1413',
                  'MEG1221', 'MEG1222', 'MEG1223',
                  'MEG1231', 'MEG1232', 'MEG1233',
                  'MEG1241', 'MEG1242', 'MEG1243',
                  'MEG1441', 'MEG1442', 'MEG1443',
                  'MEG1321', 'MEG1322', 'MEG1323',
                  'MEG1311', 'MEG1312', 'MEG1313',
                  'MEG1121', 'MEG1122', 'MEG1123',
                  'MEG1111', 'MEG1112', 'MEG1113',
                  'MEG1431', 'MEG1432', 'MEG1433',
                  'MEG2611', 'MEG2612', 'MEG2613',
                  'MEG1331', 'MEG1332', 'MEG1333',
                  'MEG1341', 'MEG1342', 'MEG1343',
                  'MEG1131', 'MEG1132', 'MEG1133',
                  'MEG1141', 'MEG1142', 'MEG1143',
                  'MEG2621', 'MEG2622', 'MEG2623',
                  'MEG2641', 'MEG2642', 'MEG2643',
                  'MEG2421', 'MEG2422', 'MEG2423',
                  'MEG2411', 'MEG2412', 'MEG2413',
                  'MEG2221', 'MEG2222', 'MEG2223',
                  'MEG2211', 'MEG2212', 'MEG2213',
                  'MEG2631', 'MEG2632', 'MEG2633',
                  'MEG2521', 'MEG2522', 'MEG2523',
                  'MEG2431', 'MEG2432', 'MEG2433',
                  'MEG2441', 'MEG2442', 'MEG2443',
                  'MEG2231', 'MEG2232', 'MEG2233',
                  'MEG2311', 'MEG2312', 'MEG2313',
                  'MEG2321', 'MEG2322', 'MEG2333',
                  'MEG2531', 'MEG2532', 'MEG2533']}
assert len(sensors['lh']) == len(sensors['rh'])
peak_idx = list()
peak_chs = [['MEG1621', 'MEG1721'],  # LH 2, 6
            ['MEG2411', 'MEG2521']]  # RH 2, 6 homologs
# Loop over groups to compute & write RMS data matrix
t0 = time.time()
for gi, group in enumerate(groups):
    print(' Loading data for %s...' % grp_nms[gi])
    subjects = simms_df[simms_df.Group == group].Subject_ID.values.tolist()
    for ci, cond in enumerate(conditions):
        print('   %s...' % cond)
        for si, subj in enumerate(subjects):
            print('     %s' % subj)
            evoked_file = op.join(data_dir, 'bad_%s' % subj, 'inverse',
                                  '%s_%d-sss_eq_bad_%s-ave.fif'
                                  % (analysis, lpf, subj))
            evoked = read_evokeds(evoked_file, condition=cond,
                                  baseline=(None, 0))
            if len(evoked.info['bads']) > 0:
                print('     Interpolating bad channels...')
                evoked.interpolate_bads()
            times = evoked.times
            sfreq = evoked.info['sfreq']
            ch_names = evoked.ch_names
            pick_mag = mne.pick_types(evoked.info, meg='mag')
            assert pick_mag.shape[0] == 102
            pick_grad = mne.pick_types(evoked.info, meg='grad')
            assert pick_grad.shape[0] == 204
            if gi == 0 and ci == 0 and si == 0:
                mag_data = np.zeros((len(groups), len(conditions),
                                     len(subjects), len(pick_mag),
                                     len(times)))
                grad_data = np.zeros((len(groups), len(conditions),
                                     len(subjects), len(pick_grad) // 2,
                                     len(times)))
            mag_data[gi, ci, si] = evoked.copy().pick_types(meg='mag').data
            grad_data[gi, ci, si] = \
                _merge_grad_data(evoked.copy().pick_types(meg='grad').data)
            # Do channel selection
            for sel, hem in enumerate(sensors.keys()):
                # subselect channels
                picks = np.asarray(ch_names)[pick_mag]
                ch_selection = np.intersect1d(picks, sensors[hem]).tolist()
                assert len(ch_selection) == len(sensors[hem]) // 3
                if set(peak_chs[sel]).issubset(ch_selection):
                    peak_idx += [ch_selection.index(kk)
                                 for kk in peak_chs[sel]]
                # restrict evoked data to selection
                evo_cp = evoked.copy().pick_channels(ch_selection)
                evo_cp.info.normalize_proj()  # likely not necessary
                # init container matrix
                if gi == 0 and ci == 0 and si == 0 and sel == 0:
                    sel_data = np.zeros((len(groups), len(conditions),
                                         len(subjects), len(sensors),
                                         len(ch_selection), len(times)))
                sel_data[gi, ci, si, sel] = evo_cp.data
                
# grps x conds x hems x subjs x chans x times
sel_data = np.transpose(sel_data, axes=(0, 1, 3, 2, 4, 5))
peak_idx = np.unique(np.asarray(peak_idx))
print(' Writing data...')
file_out = op.join(data_dir, '%s_data.npz' % analysis)
np.savez(file_out, sel_data=sel_data, mag_data=mag_data,
         grad_data=grad_data, times=times, sfreq=sfreq, peak_idx=peak_idx)
print('  Elapsed: %0.1f min' % ((time.time() - t0) / 60.))
