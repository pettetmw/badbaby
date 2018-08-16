# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

import numpy as np
from expyfun import stimuli
from expyfun import assert_version

assert_version('8511a4d')

def gen_am_tone(stim_fs=44100, stim_dur=1.0, stim_rms=0.01, carrier_freq=1000, 
                mod_ind=0.8, mod_freq=40):
    """Returns an RMS normalized amplitude modulated sinusoidal time series


    Args:
    stim_fs (int): sampling frequency for generated stimuli defaults 
                   to 44100 Hz.
    stim_dir : float
    stim_rms : float
    output_dir : string
    carrier_freq : int
    stim_db : int
    mod_ind : float
    mod_freq : int

    Returns:
    tone

    """
    t = np.arange(int(stim_fs * stim_dur)) / float(stim_fs)
    tone = np.sin(2 * np.pi * carrier_freq * t) * (1 + np.sin(2 * np.pi * mod_freq * t) * mod_ind)
    tone *= stim_rms * np.sqrt(2)
    tone = stimuli.window_edges(tone, stim_fs, 0.025, edges='both')
    return tone
