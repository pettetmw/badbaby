# -*- coding: utf-8 -1*-

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
#          simplified bsd-3 license

"""Script for basic auditory testing using AM sinusoidal tone"""

import numpy as np
from expyfun import ExperimentController
import gen_am_tone
from expyfun._trigger_controllers import decimals_to_binary
from expyfun import assert_version

assert_version('8511a4d')
# Experiment parameters
test_fs = 24414
tdt_fs = 24414
tone_stim = gen_am_tone.gen_am_tone(stim_fs=tdt_fs)

with ExperimentController('tone', stim_db=80, stim_fs=tdt_fs,
                          check_rms=None, suppress_resamp=True) as ec:
    seed = int(ec.participant) if ec.participant else 555  # convert participant to int
    rng = np.random.RandomState(seed)  # seed generator with participant
    last_time = -np.inf
    for trial in range(0, 120):
        # stamp trigger line prior to stimulus onset
        ec.clear_buffer()
        ec.load_buffer(tone_stim)
        ec.identify_trial(ec_id=str(trial), ttl_id=decimals_to_binary([2], [2]))
        # our next start time is our last start time, plus
        # the stimulus duration, plus min wait time, plus random amount
        when = last_time + 1 + .5 + rng.rand(1)
        ec.write_data_line('soa', value=when - last_time)  # write out new data
        # containing value for stimulus onset times i.e., difference
        # between flip times for consecutive trials
        last_time = ec.start_stimulus(when=when)  # stamps stimulus onset
        ec.wait_secs(1)  # wait through tone duration to stop the playback
        ec.stop()
        ec.trial_ok()
        ec.check_force_quit()  # make sure we're not trying to quit
