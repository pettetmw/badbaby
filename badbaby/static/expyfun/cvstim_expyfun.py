# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

"""Script for basic auditory testing using wav files

Notes:
Formula for converting decimal integer into number of bits borrowed from
http://www.exploringbinary.com/number-of-bits-in-a-decimal-integer
"""

import numpy as np
import os
from os import path as op
from expyfun import ExperimentController
from expyfun.stimuli import read_wav
from expyfun._trigger_controllers import decimals_to_binary
from expyfun import assert_version

assert_version('8511a4d')
n_trials = 80

fs = 24414
stim_dir = op.join(op.dirname(__file__), 'stimuli', 'cv')

with open(op.join(os.getcwd(), stim_dir, 'trial_list.txt'), 'r') as f:
    rows = (line.split('\t') for line in f)
    sound_files = {row[0]: row[1][0:].split('\n') for row in rows}
sound_files = {j: op.join(stim_dir, k[0])
               for j, k in sound_files.iteritems()}

wavs = [np.ascontiguousarray(read_wav(v)) for _, v in sorted(sound_files.items())]
trials = np.tile(np.arange(len(wavs)), n_trials)  # generate array of trial types
# convert length of wave files into number of bits
n_bits = int(np.floor(np.log2(len(wavs)))) + 1
# trial_types = lambda n: np.array(range(len(wavs))*n)
with ExperimentController('syllable', stim_db=80, stim_fs=fs, stim_rms=0.01,
                          check_rms=None, suppress_resamp=True) as ec:

    seed = int(ec.participant) if ec.participant else 555  # convert participant to int
    rng = np.random.RandomState(seed)  # seed generator with participant
    rng.shuffle(trials)  # randomly shuffle trial types
    last_time = -np.inf
    for trial in trials:
        # stamp trigger line prior to stimulus onset
        trial_name = op.basename(sound_files[str(trial + 1)][:-4])
        ec.clear_buffer()
        ec.load_buffer(wavs[trial][0])
        ec.identify_trial(ec_id=trial_name, ttl_id=decimals_to_binary([trial + 1], [n_bits]))
        # our next start time is our last start time, plus
        # the stimulus duration, plus min wait time, plus random amount
        stim_len = 1./fs * len(wavs[trial][0][0])  # in seconds
        when = last_time + stim_len + 1 + rng.rand(1)
        ec.write_data_line('soa', value=when - last_time)
        last_time = ec.start_stimulus(when=when)  # stamps stimulus onset
        ec.wait_secs(stim_len)  # wait through tone duration to stop the playback
        ec.stop()
        ec.trial_ok()
        ec.check_force_quit()  # make sure we're not trying to quit
