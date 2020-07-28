#!/usr/bin/env python

"""Auditory oddball paradigm parameters."""

import os.path as op
from pathlib import Path

static = op.join(Path(__file__).parents[0], 'data', 'static')
datadir = op.join(Path(__file__).parents[0], 'data')
figsdir = op.join(Path(__file__).parents[0], 'writeup', 'results', 'figures')

datapath = '/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch'
exclude = []
run_name = 'mmn'
epoching = (-0.1, 0.6)
lowpass = 55.0
highpass = 0.1
peak_window = (.235, .53)
oddball_stimuli = ['standard', 'ba', 'wa']



