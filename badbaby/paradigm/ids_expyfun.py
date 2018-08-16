# -*- coding: utf-8 -*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

"""Script for infant basic auditory testing using infant directed speech (IDS)"""

import numpy as np
from os import path as op
from expyfun import ExperimentController
from expyfun.stimuli import read_wav
from expyfun._trigger_controllers import decimals_to_binary
from expyfun import assert_version

assert_version('8511a4d')

fs = 24414
stim_dir = op.join(op.dirname(__file__), 'stimuli', 'ids')
sound_files = ['inForest_part-1-rms.wav',
               'inForest_part-2-rms.wav',
               'inForest_part-3-rms.wav',
               'inForest_part-4-rms.wav',
               'inForest_part-5-rms.wav']

sound_files = {j: op.join(stim_dir, k)
               for j, k in enumerate(sound_files)}
wavs = [np.ascontiguousarray(read_wav(v)) for _, v in sorted(sound_files.items())]
# convert length of wave files into number of bits
n_bits = int(np.floor(np.log2(len(wavs)))) + 1
with ExperimentController('IDS', stim_db=75, stim_fs=fs, stim_rms=0.01,
                          check_rms=None, suppress_resamp=True) as ec:
    for ii, wav in enumerate(wavs):
        # stamp trigger line prior to stimulus onset
        ec.clear_buffer()
        ec.load_buffer(wav[0])
        ec.identify_trial(ec_id=str(ii), ttl_id=decimals_to_binary([ii], [n_bits]))
        # our next start time is our last start time, plus
        # the stimulus duration
        stim_len = 1./fs * len(wav[0][0])  # in seconds
        ec.start_stimulus()  # stamps stimulus onset
        ec.wait_secs(stim_len)  # wait through stimulus duration to stop the playback
        ec.stop()
        ec.trial_ok()
        ec.check_force_quit()  # make sure we're not trying to quit
