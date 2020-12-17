#!/usr/bin/env python

"""score.py: MNEFUN stimulus event scoring functions."""

from __future__ import print_function

import datetime
import os
from os import path as op
import glob
import json

import numpy as np
from pytz import timezone

import mne
from mnefun import extract_expyfun_events
from expyfun.io import read_tab

from badbaby.defaults import tabdir


TRIGGER_MAP = {
    'Dp01bw6-rms': 103,
    'Dp01bw1-rms': 104,
    'Dp01bw10-rms': 105
}


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
            for ti, tab_file in enumerate(tab_files):
                with open(tab_file, 'r') as fid:
                    header = fid.readline().lstrip('#').strip()
                    header = json.loads(header.replace("'", '"'))
                assert header['participant'] == exp_subj
                t_tab = datetime.datetime.strptime(
                    header['date'], '%Y-%m-%d %H_%M_%S.%f'
                ).replace(tzinfo=timezone('US/Pacific'))
                t_raw = raw.info['meas_date']
                off_minutes = abs((t_raw - t_tab).total_seconds() / 60.)
                # tone/1 or IDS/2 or syllable/3
                good[ti] = (off_minutes < 120) and \
                    header['exp_name'] == 'syllable' and \
                    header['session'] in ('1', '3')
            assert sum(good) == 1, sum(good)
            # runtime expyfun logging
            data = read_tab(tab_files[np.where(good)[0][0]])

            # Correct the triggers
            new_nums = np.array(
                [TRIGGER_MAP[d['trial_id'][0][0]] for d in data], int)
            exp_times = [d['play'][0][1] for d in data]

            # Sometimes we are missing the last one
            assert len(data) in (len(events), len(events) + 1)
            if len(data) == len(events) + 1:
                data = data[:-1]
                new_nums = new_nums[:-1]
                # TODO map events sample(s) to expyfun time befor corrcoef
                corr = np.corrcoef(
                    events[:, 0] / raw.info['sfreq'], np.array(exp_times) * 10)[0, 1]
                assert corr > 0.9999999
            wrong = new_nums != events[:, 2]
            if wrong.any():
                print(f'    Replacing {wrong.sum()}/{len(wrong)} TTL IDs')
                events[:, 2] = new_nums

            # Write
            fname_out = op.join(out_dir, f'ALL_{run_name % subj}-eve.lst')
            mne.write_events(fname_out, events)
