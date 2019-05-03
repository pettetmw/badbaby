# -*- coding: utf-8 -*-

""" Scores trials of SPS experiment and do trial EQ """

# Authors: Kambiz Tavabi <ktavabi@uw.edu>

from __future__ import print_function

import os
from os import path as op
import numpy as np
import mne
from mnefun import extract_expyfun_events
from paradigm.expyfun.io import read_tab


def score(p, subjects):
    """Scoring function"""
    for subj in subjects:
        print('  Running subject %s... ' % subj, end='')

        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)

        for run_name in p.run_names:
            print(subj)
            fname = op.join(p.work_dir, subj, p.raw_dir,
                            (run_name % subj) + p.raw_fif_tag)
            events, _ = extract_expyfun_events(fname)[:2]
            events[:, 2] += 100
            fname_out = op.join(out_dir,
                                'ALL_' + (run_name % subj) + '-eve.lst')
            mne.write_events(fname_out, events)


def reconstruct_events(p, subjects):
    """Reconstruct events from expyfun tab file """
    for subj in subjects:
        print('  Running subject %s... ' % subj, end='')

        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        for run_name in p.run_names:
            print(subj)
            fname = op.join(p.work_dir, subj, p.raw_dir,
                            (run_name % subj) + p.raw_fif_tag)
            tab_file = op.join(p.work_dir, subj, p.list_dir,
                               (run_name % subj + '.tab'))

            evs, _ = extract_expyfun_events(fname)[:2]
            raw = mne.io.read_raw_fif(fname, allow_maxshield='yes')
            data = read_tab(tab_file)
            new_evs = np.zeros(evs.shape, dtype=np.int)
            for i in range(len(evs)):
                new_evs[i, 0] = raw.time_as_index(data[i]['play'][0][1])
                # classify event type based on expyfun stimulus
                new_evs[i, 1] = 0
                if data[i]['trial_id'][0][0] == 'Dp01bw6-rms':
                    trigger = 103
                elif data[i]['trial_id'][0][0] == 'Dp01bw1-rms':
                    trigger = 104
                elif data[i]['trial_id'][0][0] == 'Dp01bw10-rms':
                    trigger = 105
                new_evs[i, 2] = trigger
        fname_out = op.join(out_dir,
                            'ALL_' + (run_name % subj) + '-eve.lst')
        mne.write_events(fname_out, new_evs)
