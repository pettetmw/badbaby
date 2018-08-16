# -*- coding: utf-8 -*-

""" Global study parameters"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import os.path as op
project_dir = '/home/ktavabi/Projects/badbaby'
data_dir = '/media/ktavabi/ALAYA/data/ilabs/badbaby'
meg_dirs = {kk :op.join(data_dir, vv)
            for kk, vv in zip(['mmn', 'assr', 'ids'],
                              ['mismatch', 'tone', 'speech'])}
