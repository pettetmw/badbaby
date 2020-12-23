#!/usr/bin/env python

"""score.py: MNEFUN stimulus event scoring functions."""

import datetime
import os
from os import path as op
import re
import glob
import json

import numpy as np
from pytz import timezone

import mne
from mnefun import extract_expyfun_events
from expyfun.io import read_tab

from badbaby.defaults import tabdir

# Badbaby stimuli files
TRIGGER_MAP = {
    'Dp01bw6-rms': 103,
    'Dp01bw1-rms': 104,
    'Dp01bw10-rms': 105,
}
# I still don't know what's being unified
OTHER_TRIGGER_MAP = {
    'Dp01bw7-rms': 103,
    'Dp01bw1-rms': 104,
    'Dp01bw13-rms': 105,
}
IN_NAMES = ('std', 'ba', 'wa')
IN_NUMBERS = (103, 104, 105)


def score(p, subjects):
    """Use expyfun to extract events write MNE events file to disk."""
    for subj in subjects:
        print('  Running subject %s... ' % subj, end='')

        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)

        for run_name in p.run_names:
            print(subj)
            # Extract standard events
            fname = op.join(p.work_dir, subj, p.raw_dir,
                            (run_name % subj) + p.raw_fif_tag)
            events, _ = extract_expyfun_events(fname)[:2]
            events[:, 2] += 100
            # Find the right .tab file
            raw = mne.io.read_raw_fif(fname, allow_maxshield='yes')
            exp_subj = subj.split('_')[1].rstrip('ab')
            tab_files = sorted(glob.glob(op.join(tabdir, f'{exp_subj}_*.tab')))
            assert len(tab_files)
            good = np.zeros(len(tab_files), bool)
            got = list()
            for tab_file in tab_files:
                with open(tab_file, 'r') as fid:
                    header = fid.readline().lstrip('#').strip()
                    if 'runfile' in header:
                        # errant session that breaks things
                        assert subj == 'bad_130a'
                        header = re.sub("\"runfile.*\'\\)\"", "'1'", header)
                    header = json.loads(header.replace("'", '"'))
                assert header['participant'] == exp_subj
                if '.' in header['date']:
                    fmt = '%Y-%m-%d %H_%M_%S.%f'
                else:
                    fmt = '%Y-%m-%d %H_%M_%S'
                t_tab = datetime.datetime.strptime(
                    header['date'], fmt).replace(tzinfo=timezone('US/Pacific'))
                t_raw = raw.info['meas_date']
                # offsets between the Neuromag DAQ and expyfun computer
                off_minutes = abs((t_raw - t_tab).total_seconds() / 60.)
                got.append((
                    off_minutes, header['exp_name'], header['session']))
            # pick something in the right time frame, and the correct
            # session
            good = [m < 120 and
                    e == 'syllable' and
                    s in ('1', '3')
                    for m, e, s in got]
            if sum(good) == 2:
                idx = np.where(good)[0]
                sizes = [os.stat(tab_files[ii]).st_size for ii in idx]
                print(f'    Triaging based on file sizes: {sizes}')
                for ii in idx:
                    good[ii] = False
                good[idx[np.argmax(sizes)]] = True

            # We should only have one candidate file
            assert sum(good) == 1, sum(good)
            fname_tab = tab_files[np.where(good)[0][0]]
            data = read_tab(fname_tab, allow_last_missing=True)

            # Correct the triggers
            if subj in ('bad_921a', 'bad_925a'):
                use_map = OTHER_TRIGGER_MAP
            else:
                use_map = TRIGGER_MAP
            new_nums = np.array(
                [use_map[d['trial_id'][0][0]] for d in data], int)
            exp_times = [d['play'][0][1] for d in data]

            # Sometimes we are missing the last one
            assert len(data) >= len(events), (len(data), len(events))
            n_missed = len(data) - len(events)
            if n_missed:
                if subj == 'bad_117a':
                    sl = slice(n_missed - 1, -1, None)
                else:
                    sl = slice(None, -n_missed, None)
                data = data[sl]
                new_nums = new_nums[sl]
                exp_times = exp_times[sl]
                corr = np.corrcoef(events[:, 0], exp_times)[0, 1]
                assert corr > 0.99999999, corr
            wrong = new_nums != events[:, 2]
            if wrong.any():
                print(f'    Replacing {wrong.sum()}/{len(wrong)} TTL IDs')
                events[:, 2] = new_nums
            assert np.in1d(events[:, 2], IN_NUMBERS).all()
            print('    Counts: ' + '  '.join(
                f'{name.upper()}: {(events[:, 2] == num).sum()}'
                for name, num in zip(IN_NAMES, IN_NUMBERS)))

            # Write
            fname_out = op.join(out_dir, f'ALL_{run_name % subj}-eve.lst')
            mne.write_events(fname_out, events)
