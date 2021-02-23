# Magnetoencephalography (MEG) examination of speech processing in infant auditory cortex

> DESCRIPTION: speech infants MEG.
>
> URL: https://github.com/ktavabi/badbaby
>
> EMAIL: ktavabi@gmail.com

This study was funded by philanthropic funding from [Simms
Mann](https://www.simmsmanninstitute.org) and [Bezos Family
Foundation](https://www.bezosfamilyfoundation.org).

MNEFUN processing pipeline

        1. Determine ACQ sampling rate and ECG channel.
        2. Write ACQ prebad channel to disk.
        3. Score.
        4. HP estimation yeilding annotation parameters.
        5. MF and move comp.
        6. Data & ERM covariances.
        7. Autoreject to threshold & reject noisy trials
        8. Compute ECG & ERM projectors
        9. Epoching & writing evoked data to disk.

Subjects whose names are incorrect and need to be manually copied and renamed:

- bad_208a  bad_208_a
- bad_209a  bad_209
- bad_301a  bad_301
- bad_921a  bad_921
- bad_925a  bad_925
- bad_302a  bad_302
- bad_116a  bad_116
- bad_211a  bad_211

Subjects whose data were not on the server and needed to be uploaded were
[bad_114, bad_214, bad_110, bad_117a, bad_215a, bad_217, bad_119a].
Files were uploaded to brainstudio with variants of:
XXX priority(p)3 --files-from arg on ../static/missing.txt

    $ rsync -a --rsh="ssh -o KexAlgorithms=diffie-hellman-group1-sha1" --partial --progress --include="*_raw.fif" --include="*_raw-1.fif" --exclude="*" /media/ktavabi/ALAYA/data/ilabs/badbaby/*/bad_114/raw_fif/* larsoner@kasga.ilabs.uw.edu:/data06/larsoner/for_hank/brainstudio
    >>> mne.io.read_raw_fif('../mismatch/bad_114/raw_fif/bad_114_mmn_raw.fif', allow_maxshield='yes').info['meas_date'].strftime('%y%m%d')  # this neccessary?

Then repackaged manually into brainstudio/bad_baby/bad_*/*/ directories
based on the recording dates.

Subjects who did not complete preprocessing:

TODO add to defaults.exclude
- 223a: Preproc (Only 13/15 good ECG epochs found)

This is because for about half the time their HPI was no good, so throw them
out.



## Branches

### Oddball
Consonant-vowel syllable stimuli used in an double-oddball [1] paradigm to examine the patterning of auditory evoked activity following changes in phonological voice-onset-timing [2] contrasts in young infants.

To process and generate manuscript figures you can run the following scripts on the raw MEG data from each paradigm.

   1. run-mnefun.py
   2. cHPI-positions
   3. sample-demographics
   4. sensor-data
   5. decoder-results
   
