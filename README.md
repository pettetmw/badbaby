# Magnetoencephalography (MEG) recordings of speech processing in infant auditory cortex

> DESCRIPTION: speech infants MEG.
>
> URL: https://github.com/ktavabi/badbaby
>
> EMAIL: ktavabi@gmail.com

This study was funded by funding from [Simms
Mann](https://www.simmsmanninstitute.org) and [Bezos Family
Foundation](https://www.bezosfamilyfoundation.org).

* `run_mnefun` wrappers for `mne-tools` MEEG proccessing tools to denoise raw data across three auditory stimulation blocks:

  1. _Tone:_ sinusoidal 1 Khz carrier signal with 40Hz amplitude modulations at 80% modulation depth.
  2. _MMN:_ double oddball syllabic segments with categorically differentiable VOT.
  3. _IDS:_ short infant directed speech narrative.

The `run_mnefun` script will (or should):

  1. Determine ACQ sampling rate and ECG channel
  2. Mark bad channels
  3. Score trial for epoching
  4. Estimate subject head position throughout ACQ
  5. Apply MaxFilter SSS and movement compensation
  6. Compute data covariance
  7. Use Autoreject to compute noisy trial thresholds
  8. Compute and apply ECG and empty room SSP

After denoising the `run_mnefun` script needs to be run for each stimulation block to compute epochs and generate individual subject `HTML` reports as specified in the session specific `YAML` files.

