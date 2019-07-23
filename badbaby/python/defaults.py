#!/usr/bin/env python

"""Auditory oddball paradigm parameters."""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2018, Seattle, Washington'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'
__status__ = 'Development'

import os.path as op
from pathlib import Path

static = op.join(Path(__file__).parents[1], 'static')
datadir = op.join(Path(__file__).parents[1], 'data')
figsdir = op.join(Path(__file__).parents[1], 'figures')
resultsdir = op.join(Path(__file__).parents[1], 'results')

datapath = '/media/ktavabi/ALAYA/data/ilabs/badbaby/mismatch'
exclude = []
run_name = 'mmn'
epoching = (-0.1, 0.6)
lowpass = 55.
highpass = 0.1
peak_window = (.235, .53)



