# %% Imports
import os
from glob import glob
import numpy as np
import pandas as pd

from mne.time_frequency import tfr_morlet

from itertools import product as iprod
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
from hotelling.stats import hotelling_t2 as hott2
import mne
from mne.externals.h5io import write_hdf5
from mnefun import Params
params = Params()
epochs_dir = params.epochs_dir
inverse_dir = params.inverse_dir
epochs_tag = params.epochs_tag
inv_tag = params.inv_tag
eq_tag = params.eq_tag
#datadir = '/mnt/ilabs/tcirc/tone'
#lp = '100'
datadir = '/mnt/ilabs/badbaby/data'
lp = '50'
ave_dir = inverse_dir
ave_tag = '-ave'
tfr_tag = '-tfr'
grptags = ['a','b']
age2gid = { '2mo':'a', '6mo':'b' } # age-to-groupID ('a' and 'b' are parts of file names)

# %% Locate input files

# e.g., /mnt/ilabs/tcirc/tone/bad_116a/epochs/All_100-sss_bad_116a-epo.fif
# start with path name pattern, "pnp"
epo_pnp = [ os.path.join( datadir, '*', epochs_dir, 'All_'+lp+inv_tag+'_*' ), epochs_tag+'.fif' ]
epops = [ sorted(glob(epo_pnp[0]+g+epo_pnp[1])) for g in grptags ] # get the existing a's and b's

picks = [ [ os.path.basename(sidp)[-16:-8] for sidp in g ] for g in epops ]
#picks = [ picks[0][:3], picks[1][:3] ] # crop for quickie test

freqs = np.arange(38.0,44.0,2.0)
n_cycles = freqs / 2. # Coerce n_cycles to same time window duration

tfr_vnms = [ 'pow', 'itc', 't2c', 'hot' ] # time-freq response variable names (also part of file name)
fmn,fmx,tmn,tmx = freqs[0],freqs[-1],.2,.8 # freq and time limits (Hz and Sec)


# # %% loop over age and sbj
mne.utils.set_config('MNE_USE_CUDA', 'true')
for ig,g in enumerate(picks): # each group
    for iss,ss in enumerate(g): # each subject id path

        epo = mne.read_epochs( epops[ig][iss] )['tone']

        #TODO: need to select 

        tNTrl, tNCh, tNS = epo.get_data().shape
        
        # complex EpochsTFR
        tfr_epo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, average=False, output='complex')
        
        # power EpochsTFR
        pow_epo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, average=False, output='power')
        
        # power and ITC AverageTFR
        pow_evo, itc_evo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, average=True, output='power')
        
        tNTrl, tNCh, tNFrq, tNS = tfr_epo.data.shape
        
        tMFFT = np.mean( tfr_epo.data, axis=0 )

        hot_evo = itc_evo.copy()
        for iSmp in np.arange(tNS):
            for iFrq in np.arange(tNFrq):
                for iCh in np.arange(tNCh):
                    tX = [ np.real(tfr_epo.data[:,iCh,iFrq,iSmp]), np.imag(tfr_epo.data[:,iCh,iFrq,iSmp]) ]
                    tX = np.transpose(tX)
                    hot = hott2(tX)
                    hot_evo.data[iCh,iFrq,iSmp] = hot[0]

        # compute the mean of the variances along real and imaginary axis
        tVFFT = np.mean( np.stack( ( np.var( np.real(tfr_epo.data), 0 ), np.var( np.imag(tfr_epo.data), 0 ) ) ), 0 )
        numerator = abs(tMFFT)
        denominator = np.sqrt( tVFFT / ( tNTrl - 1 ) )
        tcirc = (numerator / denominator)**2
        tcirc_evo = itc_evo.copy()
        tcirc_evo.data = tcirc
        
        tfr = [ [ '-tfr-epo', '-pow-epo', '-pow-evo', '-itc-evo', '-hot-evo', '-t2c-evo'], 
        [ tfr_epo, pow_epo, pow_evo, itc_evo, hot_evo, tcirc_evo ] ]
        # QC:
        # tfr_epo.data.shape == (120, 306, 3, 781) # complex-valued tf response
        # pow_epo.data.shape == (120, 306, 3, 781) # power of  tfr_epo
        # pow_evo.data.shape == (306, 3, 781)
        # itc_evo.data.shape == (306, 3, 781)
        # tcirc_evo.data.shape == (306, 3, 781)

        # Loop over and save tfr variables as ".h5", which allows us to include
        # the frequency dimension (in addition to trials, channels, and time points)
        for iv,vn in enumerate(tfr[0]): # tfr[0] is the list of tfr variable names
            if vn[-3:] == 'evo':
                tfr_fname = os.path.join( datadir, ss, ave_dir, 'All_'+lp+inv_tag+'_'+eq_tag+'_'+ss+vn+tfr_tag+'.h5' )
            else:
                tfr_fname = os.path.join( datadir, ss, epochs_dir, 'All_'+lp+inv_tag+'_'+ss+vn+tfr_tag+'.h5' )
            #print(tfr_fname)
            tfr[1][iv].save(tfr_fname,overwrite=True) # tfr[1] is the list of tfr object variables

# %% build CSV

# Helpers to build pandas DataFrame by recursively nesting dict of categorical key columns
dict2df = lambda d : pd.DataFrame.from_records(iprod(*d.values()),columns=d.keys())
pdfcat = lambda adf : pd.concat(adf,ignore_index=True)

ttab = [] # the table to save
for ia,age in enumerate(age2gid.keys()):
    ttab.append([])
    for tfr in tfr_vnms:
        # hereafter, "sid" is subject id; aslo make sure that "tfrp" has been created above
        vn = '-'+tfr+'-evo'
        tfr_fname_patt = os.path.join( datadir, '*'+grptags[ia], ave_dir, 'All_'+lp+inv_tag+'_'+eq_tag+'_'+'*'+grptags[ia]+vn+tfr_tag+'.h5' )
        sidps = sorted(glob(tfr_fname_patt))
        sidps = sidps[:4] # shorty version for quick testing
        sids = [ os.path.basename(sidp)[-23:-16] for sidp in sidps ] # get id from full path
        tfrevos = [ mne.time_frequency.read_tfrs(sidp)[0] for sidp in sidps ] # tfr evo objects read from each path
        fps = [f >= fmn and f <= fmx for f in tfrevos[0].freqs] # frequency picks
        tps = [ t >= tmn and t < tmx for t in tfrevos[0].times ] # time picks
        freqs = tfrevos[0].freqs[fps] # the picked frequencies, for labeling
        # for this tfr variable, calc mean over grads and tps for each sid... 
        tfrdvs = np.concatenate( [ np.mean(np.mean(atfr.copy().pick('grad').data[:,:,tps],-1),0)[fps]
            for atfr in tfrevos ] ) # then stack the three fps from all the sids
        # make dataframe key fields
        df = dict2df( { 'age':[age], 'sid':sids, 'tfrvar':[tfr], 'freqs':freqs } )
        df['tfrval'] = tfrdvs # add data field
        ttab[ia].append(df) # append data frame to table
    ttab[ia]=pd.concat(ttab[ia],ignore_index=True) # stack the dfs from the different tfr variables
ttab=pd.concat(ttab,ignore_index=True) # stack the two age groups
ttab.to_csv('tfr_stats2.csv') # save in csv format in local directory
