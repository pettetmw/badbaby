#!/usr/bin/env python
# coding: utf-8

# # XDawnTcirc

# ## Compute within-subject Tcirc stat from XDAWN evoked averages

# In[1]:


dataPath = '/mnt/scratch/badbaby/tone/'


# ###### Create a list of the existing ```bad\*a``` paths from the list of ```bad\*b``` paths

# In[2]:


from glob import glob
from os.path import exists, basename


# In[3]:


bPaths = sorted(glob(dataPath + 'bad*b/epochs/*epo.fif')) # get the existing b's
aPaths = [ p.replace('b/epochs','a/epochs').replace('b-epo','a-epo') for p in bPaths ] # rename to create a's
aPaths = [ p for p in aPaths if exists(p) ] # cull to existing a's
bPaths = [ p.replace('a/epochs','b/epochs').replace('a-epo','b-epo') for p in aPaths ] # rename to create matching b's
epoPaths = aPaths + bPaths

all([ exists(p) for p in epoPaths ]) # test whether all exist

xdawnPaths = [ p.replace('-epo.fif','_xdawn_ave.fif') for p in epoPaths ]

# if replace fails, xdawnPath == epoPath, which we don't want to overwrite; so,
for epoPath, xdawnPath in zip( epoPaths, xdawnPaths ):
    assert xdawnPath != epoPath

tcircPaths = [ p.replace('_xdawn_ave.fif','_tcirc.h5') for p in xdawnPaths ]

# if replace fails, xdawnPath == epoPath, which we don't want to overwrite; so,
for xdawnPath, tcircPath in zip( xdawnPaths, tcircPaths ):
    assert tcircPath != xdawnPath

print( 'The following transformations will be performed:' )
for e,x,t in zip( epoPaths, xdawnPaths, tcircPaths ):
    print( e.replace(dataPath,'') + ' --> ' + x.replace(dataPath,'')+ ' --> ' + t.replace(dataPath,''))


# ###### So, all of these are valid input arguments for epo2xdawn2tcirc.Epo2XDawn

# ###### Now we import the necessary functions

# In[9]:


cd /home/pettetmw/Github/badbaby/badbaby


# In[5]:


import mne
import python.epo2xdawn2tcirc as e2x


# In[8]:


mne.set_log_level(verbose='CRITICAL')
for epoPath in epoPaths:
    try:
        e2x.Epo2Xdawn(epoPath)
        print ('Success processing ' + basename(epoPath) )
    except:
        print ('Error processing ' + basename(epoPath) )

mne.set_log_level(verbose='INFO')


# #### Victory!

# In[ ]:




