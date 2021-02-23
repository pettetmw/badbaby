#!/usr/bin/env python

"""score.py: MNEFUN stimulus event scoring functions."""

from __future__ import print_function

import datetime
import glob
import json
import os
import re
from os import path as op

import mne
import numpy as np
from expyfun.io import read_tab
from mnefun import extract_expyfun_events
from pytz import timezone

# Badbaby stimuli files
TRIGGER_MAP = {
    "Dp01bw6-rms": 103,
    "Dp01bw1-rms": 104,
    "Dp01bw10-rms": 105,
}
OTHER_TRIGGER_MAP = {
    "Dp01bw7-rms": 103,
    "Dp01bw1-rms": 104,
    "Dp01bw13-rms": 105,
}
IN_NAMES = ("std", "ba", "wa")
IN_NUMBERS = (
    104,
    105,
    103,
)

tabdir = "/Users/ktavabi/Github/Projects/badbaby/badbaby/data/tabdir"


def score(p, subjects):
    """Use expyfun to extract events write MNE events file to disk."""
    for subj in subjects:
        print('  Running subject %s... ' % subj, end='')

        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)

        for run_name in p.run_names:
            # Extract standard events
            fname = op.join(
                p.work_dir, subj, p.raw_dir, (run_name % subj) + p.raw_fif_tag
            )
            events, _ = extract_expyfun_events(fname)[:2]
            events[:, 2] += 100
            fname_out = op.join(out_dir,
                                'ALL_' + (run_name % subj) + '-eve.lst')
            mne.write_events(fname_out, events)


def reconstruct_events(p, subjects):
    """Reconstruct events from expyfun tab file """
    for subj in subjects:
        print('  Running subject %s... ' % subj, end='')

            # Sometimes we are missing the last one
            assert len(data) >= len(events), (len(data), len(events))
            n_missed = len(data) - len(events)
            if n_missed:
                if subj in ("bad_117a"):
                    sl = slice(n_missed - 1, -1, None)
                else:
                    sl = slice(None, -n_missed, None)
                data = data[sl]
                new_nums = new_nums[sl]
                exp_times = exp_times[sl]
                corr = np.corrcoef(events[:, 0], exp_times)[0, 1]
                # The correlation should be 1 to numerical noise, if it's not, print the
                # correlation and force the user to check which >events are adjusted if you've hit
                # this assertion either it >means that we need to special case another subject, or
                # we need to adjust the assert to be more tolerant of numerical precision errors
                assert corr > 9e-20, corr
            wrong = new_nums != events[:, 2]
            if wrong.any():
                print(f"    Replacing {wrong.sum()}/{len(wrong)} TTL IDs")
                events[:, 2] = new_nums
            assert np.in1d(events[:, 2], IN_NUMBERS).all()
            print(
                "    Counts: " + "  ".join(
                    f"{name.upper()}: {(events[:, 2] == num).sum()}"
                    for name, num in zip(IN_NAMES, IN_NUMBERS)
                )
            )
        # Write
        fname_out = op.join(out_dir, f"ALL_{run_name % subj}-eve.lst")
        mne.write_events(fname_out, events)
