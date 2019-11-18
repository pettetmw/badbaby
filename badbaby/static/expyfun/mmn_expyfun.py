# -*- coding: utf-8 -*-

"""Script for basic auditory oddball paradigm.
    4:1 ratio of standards to deviants using designated wav files from HD.
    Stimulus sequence is psuedorandomized such that deviants never
    occur consecutively and are separated by at least 3 standards.
    Notes:
        Formula for converting decimal integer into number of bits from:
        http://www.exploringbinary.com/number-of-bits-in-a-decimal-integer
"""
# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# Credit: Ross Maddox <rkmaddox@uw.edu>
# License: simplified bsd-3 license


import numpy as np
from os import path as op
from paradigm.expyfun import ExperimentController
from paradigm.expyfun.stimuli import read_wav
from paradigm.expyfun._trigger_controllers import decimals_to_binary
from paradigm.expyfun import assert_version

assert_version('8511a4d')


def presentation_order(n_pres, n_standard_follow, seed):
    """Return psuedorandomized array"""
    rand = np.random.RandomState(seed)
    n_trial = sum(n_pres)
    ti = 0
    order = np.zeros(n_trial, dtype=int)
    while ti < n_trial:
        trial_type = np.where(sum(n_pres) * rand.rand()
                              < np.cumsum(n_pres))[0][0]
        order[ti] = trial_type
        ti += 1
        n_pres[trial_type] -= 1

        if trial_type > 0:
            order[ti:ti + n_standard_follow] = 0
            ti += n_standard_follow
    order = order + 2        
    order = np.insert(order, 0, np.ones(20) * 2)
    return order


stim_dir = op.join(op.dirname(__file__), 'stimuli/mmn')
sound_files = {2: op.join(stim_dir, 'Dp01bw6-rms.wav'),  # midpoint standard
               3: op.join(stim_dir, 'Dp01bw1-rms.wav'),  # ba endpoint
               4: op.join(stim_dir, 'Dp01bw10-rms.wav')}  # wa endpoint
wavs = [np.ascontiguousarray(read_wav(v))
        for _, v in sorted(sound_files.items())]

# Begin experiment
with ExperimentController('syllable', stim_db=80, stim_fs=24414, stim_rms=0.01,
                          check_rms=None, suppress_resamp=True) as ec:
    # convert participant to int
    seed = int(ec.participant) if ec.participant else 555
    trials = presentation_order([480, 100, 100], 3, seed)
    # convert number of unique trial types into number of bits
    n_bits = int(np.floor(np.log2(max(trials)))) + 1
    rng = np.random.RandomState(seed)  # seed generator with participant
    last_time = -np.inf
    for trial in trials:
        # stamp trigger line prior to stimulus onset
        trial_name = op.basename(sound_files[trial][:-4])
        wav = read_wav(op.join(stim_dir, sound_files[trial]))        
        ec.clear_buffer()
        ec.load_buffer(wav[0])
        ec.identify_trial(ec_id=trial_name,
                          ttl_id=decimals_to_binary([trial], [n_bits]))
        # our next start time is our last start time, plus
        # the stimulus duration, plus min wait time, plus random amount
        stim_len = float(len(wav[0])) / ec.fs  # in seconds
        when = last_time + stim_len + .5 + rng.rand(1)
        ec.write_data_line('soa', value=when - last_time)
        last_time = ec.start_stimulus(when=when)  # ustamps stimulus onset
        # wait through tone duration to stop the playback
        ec.wait_secs(stim_len)
        ec.stop()
        ec.trial_ok()
        ec.check_force_quit()  # make sure we're not trying to quit
