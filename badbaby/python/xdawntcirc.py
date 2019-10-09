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
aPaths = [ p.replace('b/epochs','a/epochs') for p in bPaths ] # rename to create a's
aPaths = [ p.replace('b-epo','a-epo') for p in aPaths ]
aPaths = [ p for p in aPaths if exists(p) ] # cull to existing a's
bPaths = [ p.replace('a/epochs','b/epochs') for p in aPaths ] # rename to create matching b's
bPaths = [ p.replace('a-epo','b-epo') for p in bPaths ]
epoPaths = aPaths + bPaths
all([ exists(p) for p in epoPaths ]) # test whether all exist


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




